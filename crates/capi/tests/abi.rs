//! C ABI 自身的冒烟测试 —— **不需要真样本**。
//!
//! 这里验的是「ABI 的形状」而不是「解码对不对」：
//! 版本号、错误通道、panic 围栏、空指针与 `size` 的处理。
//! 解码对不对由各 crate 自己的对账测试管，ABI 只负责别把错误传丢。

use std::ffi::{c_char, CString};
use std::mem::size_of;

use v2v_capi::common::*;
use v2v_capi::player::{v2v_player_info, v2v_player_open, V2vInfo};


/// 按种类找一条流 —— **别假设流的下标**：ASF 里音频常常排在视频前面，
/// 换个样本 `streams()[0]` 就不是视频了。
fn find_stream(
    d: *mut v2v_capi::demux::V2vDemux,
    kind: u32,
) -> v2v_capi::demux::V2vStreamInfo {
    let n = unsafe { v2v_capi::demux::v2v_demux_stream_count(d) };
    for i in 0..n {
        let mut si: v2v_capi::demux::V2vStreamInfo = unsafe { std::mem::zeroed() };
        si.size = size_of::<v2v_capi::demux::V2vStreamInfo>() as u32;
        assert_eq!(unsafe { v2v_capi::demux::v2v_demux_stream(d, i, &mut si) }, OK);
        if si.kind == kind {
            return si;
        }
    }
    panic!("这个样本里没有 kind={kind} 的流");
}

#[test]
fn version_is_major_minor_packed() {
    let v = v2v_capi::common::ABI_MAJOR;
    assert_eq!(v, 1, "主版本变了要同步改头文件里的 V2V_ABI_MAJOR");
    assert_eq!(size_of::<V2vInfo>() % 4, 0, "结构体要有自然的对齐");
}

#[test]
fn open_rejects_null_and_missing_file() {
    let h = unsafe { v2v_player_open(std::ptr::null()) };
    assert!(h.is_null());
    assert_eq!(v2v_last_error_kind(), KIND_BAD);
    assert!(!v2v_last_error().is_null(), "失败时也该有消息，不该是空指针");

    let p = CString::new("/definitely/not/here.wmv").unwrap();
    let h = unsafe { v2v_player_open(p.as_ptr() as *const c_char) };
    assert!(h.is_null());
}

#[test]
fn info_on_null_handle_is_arg_error_not_a_crash() {
    let mut info = V2vInfo {
        size: size_of::<V2vInfo>() as u32,
        duration_ms: 0,
        width: 0,
        height: 0,
        video_fourcc: 0,
        audio_sample_rate: 0,
        audio_channels: 0,
        has_audio: 0,
        flags: 0,
    };
    let rc = unsafe { v2v_player_info(std::ptr::null_mut(), &mut info) };
    assert_eq!(rc, ERR_ARG);
}

/// 带真样本的端到端：open → info → 逐帧/逐样本拉到 EOS。
///
/// 缺样本就 SKIP（仓规：真样本一律 `.gitignore`，靠环境变量找）。
/// `V2V_ASF_SAMPLE` 可以是 `:` 分隔的多个路径，这里取第一个存在的。
///
/// ⚠️ **路径要给绝对的**：cargo 跑测试时工作目录是**本 crate**（`crates/capi`），
/// 不是仓库根 —— 传相对路径会走到 SKIP 分支，看起来「绿了」其实什么都没验
/// （用时 0.00s 就是那个信号）。
#[test]
fn plays_a_real_sample_end_to_end() {
    let Some(path) = std::env::var("V2V_ASF_SAMPLE")
        .ok()
        .and_then(|s| s.split(':').find(|p| std::path::Path::new(p).is_file()).map(String::from))
    else {
        eprintln!("SKIP plays_a_real_sample_end_to_end：没设 V2V_ASF_SAMPLE");
        return;
    };
    use v2v_capi::player::{
        v2v_player_close, v2v_player_read_audio, v2v_player_read_video, V2vAudioFrame, V2vFrame,
    };

    let c = CString::new(path).unwrap();
    let h = unsafe { v2v_player_open(c.as_ptr()) };
    assert!(!h.is_null(), "打不开样本：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });

    let mut info = V2vInfo {
        size: size_of::<V2vInfo>() as u32,
        duration_ms: 0,
        width: 0,
        height: 0,
        video_fourcc: 0,
        audio_sample_rate: 0,
        audio_channels: 0,
        has_audio: 0,
        flags: 0,
    };
    assert_eq!(unsafe { v2v_player_info(h, &mut info) }, OK);
    assert!(info.width > 0 && info.height > 0, "宽高不该是 0");
    assert!(info.duration_ms > 0, "时长不该是 0");
    assert_eq!(info.flags & 1, 0, "这个样本不该被标成有 B 帧");

    // 逐帧拉视频：pts 必须单调不减，平面指针必须非空。
    // 只拉前 30 帧 —— 真样本可以上千帧，debug 下全解会把测试拖成几分钟。
    let mut frames = 0u32;
    let mut last_pts = i64::MIN;
    loop {
        if frames >= 30 {
            break;
        }
        let mut f = V2vFrame {
            size: size_of::<V2vFrame>() as u32,
            pts_ms: 0,
            key_frame: 0,
            planes: unsafe { std::mem::zeroed() },
        };
        match unsafe { v2v_player_read_video(h, &mut f) } {
            OK => {
                assert!(!f.planes.y.is_null() && !f.planes.u.is_null() && !f.planes.v.is_null());
                assert_eq!(f.planes.stride_y, f.planes.width, "stride 必须等于宽度");
                assert!(f.pts_ms >= last_pts, "pts 回退了：{} -> {}", last_pts, f.pts_ms);
                last_pts = f.pts_ms;
                frames += 1;
            }
            EOS => break,
            other => panic!("read_video 返回 {other}"),
        }
    }
    assert!(frames > 0, "一帧都没解出来");

    // 逐样本拉音频（没有音轨就是立刻 EOS）。同样只拉前 50 帧。
    let mut aframes = 0u32;
    loop {
        if aframes >= 50 {
            break;
        }
        let mut a = V2vAudioFrame {
            size: size_of::<V2vAudioFrame>() as u32,
            pts_ms: 0,
            planes: std::ptr::null(),
            channels: 0,
            samples: 0,
            sample_rate: 0,
        };
        match unsafe { v2v_player_read_audio(h, &mut a) } {
            OK => {
                assert!(!a.planes.is_null() && a.samples > 0);
                aframes += 1;
            }
            EOS => break,
            other => panic!("read_audio 返回 {other}"),
        }
    }

    unsafe { v2v_player_close(h) };
}

/// 容器层：嗅探 → 列流 → 逐样本拉到 EOS。缺样本就 SKIP（路径要给绝对的）。
#[test]
fn demuxes_a_real_sample() {
    use v2v_capi::demux::*;

    let Some(path) = std::env::var("V2V_ASF_SAMPLE")
        .ok()
        .and_then(|s| s.split(':').find(|p| std::path::Path::new(p).is_file()).map(String::from))
    else {
        eprintln!("SKIP demuxes_a_real_sample：没设 V2V_ASF_SAMPLE");
        return;
    };

    // 嗅探：只读文件头 64 字节（**别整文件读进来** —— 真样本可以上百 MB）。
    let mut head = [0u8; 64];
    let n = {
        use std::io::Read;
        std::fs::File::open(&path).unwrap().read(&mut head).unwrap()
    };
    let k = unsafe { v2v_probe(head.as_ptr(), n as u32) };
    assert_eq!(k, CONTAINER_ASF);

    let c = CString::new(path).unwrap();
    let h = unsafe { v2v_demux_open(c.as_ptr()) };
    assert!(!h.is_null(), "打不开：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });

    let n = unsafe { v2v_demux_stream_count(h) };
    assert!(n >= 1, "至少得有一条流");

    // 找视频流（别假设下标），extra 里应该带着 VC-1 的序列头。
    let si = find_stream(h, STREAM_VIDEO);
    assert!(si.width > 0 && si.height > 0);
    assert_ne!(si.fourcc, 0);
    assert!(si.extra_len > 0 && !si.extra.is_null(), "视频流该带序列头");

    // 逐样本拉：**每条流各自**的时间戳必须单调不减，data 必须非空。
    // ⚠️ 不能跨流比：容器按包交错吐，音频样本排在一个视频帧前面时，
    // 它的时间戳比上一帧视频小是正常的。
    let mut count = 0u32;
    let mut last = [i64::MIN; 256];
    loop {
        let mut s: V2vSample = unsafe { std::mem::zeroed() };
        s.size = size_of::<V2vSample>() as u32;
        match unsafe { v2v_demux_next(h, &mut s) } {
            OK => {
                assert!(!s.data.is_null() && s.data_len > 0);
                let slot = (s.stream as usize) & 0xFF;
                assert!(s.timestamp_ms >= last[slot], "流 {} 的时间戳回退", s.stream);
                last[slot] = s.timestamp_ms;
                count += 1;
            }
            EOS => break,
            other => panic!("next 返回 {other}"),
        }
    }
    assert!(count > 0, "一个样本都没读出来");

    unsafe { v2v_demux_close(h) };
}

/// 解封装 + 解码串起来：样本从容器来，帧从解码器来，全程只经 ABI。
/// 缺样本就 SKIP（路径要给绝对的）。
#[test]
fn decodes_a_real_sample_through_the_abi() {
    use v2v_capi::demux::*;
    use v2v_capi::vdec::*;

    let Some(path) = std::env::var("V2V_ASF_SAMPLE")
        .ok()
        .and_then(|s| s.split(':').find(|p| std::path::Path::new(p).is_file()).map(String::from))
    else {
        eprintln!("SKIP decodes_a_real_sample_through_the_abi：没设 V2V_ASF_SAMPLE");
        return;
    };

    let c = CString::new(path).unwrap();
    let d = unsafe { v2v_demux_open(c.as_ptr()) };
    assert!(!d.is_null());

    // 视频流的信息（extra 里就是 VC-1 的序列头）。
    let si = find_stream(d, STREAM_VIDEO);

    let dec = unsafe { v2v_vdec_open(CODEC_WMV3, si.extra, si.extra_len, si.width, si.height) };
    assert!(!dec.is_null(), "建解码器失败：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });

    // 只喂视频轨的样本，逐帧解（只解前 10 帧，理由同上一处）。
    let mut frames = 0u32;
    loop {
        if frames >= 10 {
            break;
        }
        let mut s: V2vSample = unsafe { std::mem::zeroed() };
        s.size = size_of::<V2vSample>() as u32;
        match unsafe { v2v_demux_next(d, &mut s) } {
            OK => {
                if s.stream != si.number {
                    continue;
                }
                let mut p: V2vPicture = unsafe { std::mem::zeroed() };
                p.size = size_of::<V2vPicture>() as u32;
                assert_eq!(unsafe { v2v_vdec_decode(dec, s.data, s.data_len, &mut p) }, OK);
                assert_eq!(p.planes.width, si.width);
                assert_eq!(p.planes.height, si.height);
                assert!(!p.planes.y.is_null() && !p.planes.u.is_null());
                frames += 1;
            }
            EOS => break,
            other => panic!("next 返回 {other}"),
        }
    }
    assert!(frames > 0, "一帧都没解出来");

    unsafe { v2v_vdec_close(dec) };
    unsafe { v2v_demux_close(d) };
}

/// ABI 不该改变任何一个像素：同一批样本分别喂给 ABI 与 Rust 直调，逐帧比字节。
///
/// 这不是「独立 oracle」（真正的 oracle 是 ffmpeg，由各 crate 自己的对账测试管），
/// 它验的是**中间那一层没把平面/stride 搞错** —— 那正是 ABI 最容易引入的 bug。
#[test]
fn abi_decode_matches_direct_rust_decode() {
    use v2v_capi::demux::*;
    use v2v_capi::vdec::*;

    let Some(path) = std::env::var("V2V_ASF_SAMPLE")
        .ok()
        .and_then(|s| s.split(':').find(|p| std::path::Path::new(p).is_file()).map(String::from))
    else {
        eprintln!("SKIP abi_decode_matches_direct_rust_decode：没设 V2V_ASF_SAMPLE");
        return;
    };

    let c = CString::new(path).unwrap();
    let d = unsafe { v2v_demux_open(c.as_ptr()) };
    assert!(!d.is_null());

    let si = find_stream(d, STREAM_VIDEO);

    // Rust 直调那一路：自己解析序列头、自己建解码器。
    let extra = unsafe { std::slice::from_raw_parts(si.extra, si.extra_len as usize) }.to_vec();
    let seq = v2v_vc1::SeqHeader::parse(&extra).unwrap();
    let mut direct = v2v_vc1::Decoder::new(seq, si.width, si.height).unwrap();

    let dec = unsafe { v2v_vdec_open(CODEC_WMV3, si.extra, si.extra_len, si.width, si.height) };
    assert!(!dec.is_null());

    let mut compared = 0u32;
    loop {
        if compared >= 10 {
            break;
        }
        let mut s: V2vSample = unsafe { std::mem::zeroed() };
        s.size = size_of::<V2vSample>() as u32;
        match unsafe { v2v_demux_next(d, &mut s) } {
            OK => {
                if s.stream != si.number {
                    continue;
                }
                let payload =
                    unsafe { std::slice::from_raw_parts(s.data, s.data_len as usize) }.to_vec();

                let mut p: V2vPicture = unsafe { std::mem::zeroed() };
                p.size = size_of::<V2vPicture>() as u32;
                assert_eq!(unsafe { v2v_vdec_decode(dec, s.data, s.data_len, &mut p) }, OK);

                direct.decode_frame(&payload).unwrap();
                let want = direct.frame();

                let got_y = unsafe {
                    std::slice::from_raw_parts(p.planes.y, (p.planes.width * p.planes.height) as usize)
                };
                let cw = p.planes.width.div_ceil(2);
                let ch = p.planes.height.div_ceil(2);
                let got_u =
                    unsafe { std::slice::from_raw_parts(p.planes.u, (cw * ch) as usize) };
                let got_v =
                    unsafe { std::slice::from_raw_parts(p.planes.v, (cw * ch) as usize) };

                assert_eq!(got_y, want.y.as_slice(), "第 {compared} 帧的 Y 平面不一致");
                assert_eq!(got_u, want.u.as_slice(), "第 {compared} 帧的 U 平面不一致");
                assert_eq!(got_v, want.v.as_slice(), "第 {compared} 帧的 V 平面不一致");
                compared += 1;
            }
            EOS => break,
            other => panic!("next 返回 {other}"),
        }
    }
    assert!(compared > 0, "一帧都没比到");

    unsafe { v2v_vdec_close(dec) };
    unsafe { v2v_demux_close(d) };
}

/// 音频解码：从容器里找出音频流 → 建解码器 → 喂几个载荷 → 逐帧取。
///
/// 只喂前若干个载荷（整条轨可能要几秒），验的是「队列 + 编码器延迟 + PTS」
/// 这一套形状对不对，不是音质。缺样本 / 样本没有音轨就 SKIP。
#[test]
fn decodes_audio_through_the_abi() {
    use v2v_capi::adec::*;
    use v2v_capi::demux::*;

    let Some(path) = std::env::var("V2V_ASF_SAMPLE")
        .ok()
        .and_then(|s| s.split(':').find(|p| std::path::Path::new(p).is_file()).map(String::from))
    else {
        eprintln!("SKIP decodes_audio_through_the_abi：没设 V2V_ASF_SAMPLE");
        return;
    };

    let c = CString::new(path).unwrap();
    let d = unsafe { v2v_demux_open(c.as_ptr()) };
    assert!(!d.is_null());

    // 找音频流。
    let n = unsafe { v2v_demux_stream_count(d) };
    let mut audio: Option<V2vStreamInfo> = None;
    for i in 0..n {
        let mut si: V2vStreamInfo = unsafe { std::mem::zeroed() };
        si.size = size_of::<V2vStreamInfo>() as u32;
        assert_eq!(unsafe { v2v_demux_stream(d, i, &mut si) }, OK);
        if si.kind == STREAM_AUDIO {
            audio = Some(si);
            break;
        }
    }
    let Some(a) = audio else {
        eprintln!("SKIP decodes_audio_through_the_abi：这个样本没有音轨");
        unsafe { v2v_demux_close(d) };
        return;
    };

    let cfg = V2vAdecCfg {
        size: size_of::<V2vAdecCfg>() as u32,
        format_tag: a.format_tag,
        channels: a.channels,
        sample_rate: a.sample_rate,
        avg_bytes_per_sec: a.avg_bytes_per_sec,
        block_align: a.block_align,
        extra: a.extra,
        extra_len: a.extra_len,
        drop_priming: 1,
    };
    let dec = unsafe { v2v_adec_open(&cfg) };
    assert!(!dec.is_null(), "建音频解码器失败：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });

    // 只喂前 20 个音频载荷。
    let mut fed = 0;
    let mut frames = 0u32;
    let mut last_pts = i64::MIN;
    loop {
        if fed >= 20 {
            break;
        }
        let mut s: V2vSample = unsafe { std::mem::zeroed() };
        s.size = size_of::<V2vSample>() as u32;
        match unsafe { v2v_demux_next(d, &mut s) } {
            OK => {
                if s.stream != a.number {
                    continue;
                }
                assert_eq!(unsafe { v2v_adec_feed(dec, s.data, s.data_len) }, OK);
                fed += 1;
                // 喂一个就把队列抽干。
                loop {
                    let mut f: V2vAdecFrame = unsafe { std::mem::zeroed() };
                    f.size = size_of::<V2vAdecFrame>() as u32;
                    match unsafe { v2v_adec_next(dec, &mut f) } {
                        OK => {
                            assert!(!f.planes.is_null());
                            assert!(f.samples > 0);
                            assert_eq!(f.channels, a.channels);
                            assert_eq!(f.sample_rate, a.sample_rate);
                            assert!(f.pts_ms >= last_pts, "pts 回退");
                            last_pts = f.pts_ms;
                            frames += 1;
                        }
                        EOS => break,
                        other => panic!("adec_next 返回 {other}"),
                    }
                }
            }
            EOS => break,
            other => panic!("next 返回 {other}"),
        }
    }
    assert!(fed > 0, "一个音频载荷都没喂到");
    assert!(frames > 0, "一帧音频都没解出来");
    // 一个载荷装好几帧 —— 这正是「必须队列」的理由。
    assert!(frames >= fed, "解出来的帧数不该少于载荷数");

    unsafe { v2v_adec_close(dec) };
    unsafe { v2v_demux_close(d) };
}
