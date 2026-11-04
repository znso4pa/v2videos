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
/// ⚠️ **路径要给绝对的**：cargo 跑测试时工作目录是**本 crate**（`crates/v2v-capi`），
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

/// 编码层：编两帧（首帧关键帧、第二帧 P），再自己解回来核对尺寸。
///
/// 这是「自造自读」的往返 —— **必要但不充分**（编码器与解码器的假设会一起错），
/// 真正的判据在各自 crate 的对账测试里。这里验的是 ABI 的形状：
/// 关键帧位对不对、平面有没有传错、输出缓冲的生命周期。
#[test]
fn encodes_video_through_the_abi() {
    use v2v_capi::vdec::*;
    use v2v_capi::venc::*;

    for codec in [CODEC_MP43, CODEC_MP4V] {
        let (w, h) = (64u32, 48u32);
        let cfg = V2vVencCfg {
            size: size_of::<V2vVencCfg>() as u32,
            width: w,
            height: h,
            qscale: 0,
            bitrate: 0,
            fps_num: 0,
            ac_pred: 1,
            // 0 = 不用 MPEG 矩阵量化：本仓的解码器只做 quant_type=0，
            // 开 1 编出来的流自己解不回来（对外部播放器没影响）。
            mpeg_quant: 0,
        };
        let enc = unsafe { v2v_venc_open(codec, &cfg) };
        assert!(!enc.is_null(), "codec {codec} 建编码器失败：{}", unsafe {
            std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
        });

        // 造两帧：一个水平渐变，第二帧整体右移一格（让 P 帧有东西可预测）。
        let mut planes = Gradient::new(w, h);
        let mut key_flags = Vec::new();
        let mut sizes = Vec::new();
        for i in 0..2 {
            planes.shift(i as u8);
            let p = planes.as_planes();
            let mut pkt: V2vPacket = unsafe { std::mem::zeroed() };
            pkt.size = size_of::<V2vPacket>() as u32;
            assert_eq!(
                unsafe { v2v_venc_encode(enc, &p, i32::from(i == 0), &mut pkt) },
                OK,
                "codec {codec} 第 {i} 帧编码失败：{}",
                unsafe { std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy() }
            );
            assert!(pkt.len > 0 && !pkt.data.is_null());
            key_flags.push(pkt.key_frame);
            sizes.push(pkt.len);
        }
        assert_eq!(key_flags, vec![1, 0], "codec {codec} 的关键帧位不对");
        assert!(sizes[1] < sizes[0], "codec {codec} 的 P 帧不该比 I 帧大");

        unsafe { v2v_venc_close(enc) };

        // 再把 I 帧解回来：尺寸必须一致，且至少不是全黑（说明真编进去了）。
        let enc = unsafe { v2v_venc_open(codec, &cfg) };
        let p = planes.as_planes();
        let mut pkt: V2vPacket = unsafe { std::mem::zeroed() };
        pkt.size = size_of::<V2vPacket>() as u32;
        assert_eq!(unsafe { v2v_venc_encode(enc, &p, 1, &mut pkt) }, OK);

        // 取封装器要的 header（MPEG-4 是 VOL），再用它建解码器把 I 帧解回来。
        let mut hdr = vec![0u8; 512];
        let hn = unsafe { v2v_venc_header(enc, hdr.as_mut_ptr(), hdr.len() as u32) };
        assert!(hn >= 0, "codec {codec} 取 header 失败");
        if codec == CODEC_MP4V {
            assert!(hn > 0, "MPEG-4 的 I 帧不带 VOL，header 不该是空的");
        }
        let dec = unsafe { v2v_vdec_open(codec, hdr.as_ptr(), hn as u32, w, h) };
        assert!(!dec.is_null(), "codec {codec} 建解码器失败：{}", unsafe {
            std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
        });
        let mut pic: V2vPicture = unsafe { std::mem::zeroed() };
        pic.size = size_of::<V2vPicture>() as u32;
        assert_eq!(
            unsafe { v2v_vdec_decode(dec, pkt.data, pkt.len, &mut pic) },
            OK,
            "codec {codec} 解不回来：{}",
            unsafe { std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy() }
        );
        assert_eq!(pic.planes.width, w);
        assert_eq!(pic.planes.height, h);
        unsafe { v2v_vdec_close(dec) };
        unsafe { v2v_venc_close(enc) };
    }
}

/// 一块可变的 YUV420P 测试图（只有测试用）。
struct Gradient {
    w: u32,
    h: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Gradient {
    fn new(w: u32, h: u32) -> Self {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        Self {
            w,
            h,
            y: vec![0; (w * h) as usize],
            u: vec![128; (cw * ch) as usize],
            v: vec![128; (cw * ch) as usize],
        }
    }

    fn shift(&mut self, k: u8) {
        for y in 0..self.h {
            for x in 0..self.w {
                self.y[(y * self.w + x) as usize] = ((x + y + k as u32) & 0xFF) as u8;
            }
        }
    }

    fn as_planes(&self) -> v2v_capi::common::V2vPlanes {
        v2v_capi::common::planes_of(&self.y, &self.u, &self.v, self.w, self.h)
    }
}

/// AAC：编一帧正弦，验 ASC 长度与码流非空。
#[test]
fn encodes_audio_through_the_abi() {
    use v2v_capi::aenc::*;

    let cfg = V2vAencCfg {
        size: size_of::<V2vAencCfg>() as u32,
        channels: 2,
        sample_rate: 44100,
        bitrate: 128_000,
        ms_stereo: 1,
        kb_window: 0,
        threads: 1,
    };
    let enc = unsafe { v2v_aenc_open(&cfg) };
    assert!(!enc.is_null(), "建 AAC 编码器失败：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });

    let n = AAC_FRAME_SAMPLES as usize;
    let mut l = vec![0f32; n];
    let mut r = vec![0f32; n];
    for i in 0..n {
        let s = (i as f32 * 0.05).sin() * 8000.0;
        l[i] = s;
        r[i] = -s;
    }
    let planes: [*const f32; 2] = [l.as_ptr(), r.as_ptr()];

    let mut pkt: V2vPacket = unsafe { std::mem::zeroed() };
    pkt.size = size_of::<V2vPacket>() as u32;
    assert_eq!(unsafe { v2v_aenc_encode(enc, planes.as_ptr(), n as u32, &mut pkt) }, OK);
    assert!(pkt.len > 0 && !pkt.data.is_null());
    assert_eq!(pkt.key_frame, 1, "AAC 每帧都该是独立的关键帧");

    // 采样点不对必须报参数错，不能悄悄补零。
    assert_eq!(
        unsafe { v2v_aenc_encode(enc, planes.as_ptr(), 512, &mut pkt) },
        ERR_ARG
    );

    // ASC：先问长度（给 0 字节），再取。
    let need = unsafe { v2v_aenc_asc(enc, std::ptr::null_mut(), 0) };
    assert_eq!(need, ERR_ARG, "缓冲不够时该报参数错");
    let mut asc = vec![0u8; 8];
    let got = unsafe { v2v_aenc_asc(enc, asc.as_mut_ptr(), asc.len() as u32) };
    assert!(got > 0, "ASC 长度是 {got}");
    assert_eq!(got as usize, 2, "44.1k 立体声的 ASC 是 2 字节");

    unsafe { v2v_aenc_close(enc) };
}

/// 封装层：编码 → 封成 ASF → 再用容器层读回来。
///
/// 这是**整条链的往返**（编码器 + 封装器 + 解封装器三者的假设会一起错），
/// 所以它只证明「ABI 把参数传对了」，不证明格式合规 —— 后者在各自 crate 的
/// 对账测试里（对 ffprobe / ffmpeg）。
#[test]
fn muxes_asf_and_reads_it_back() {
    use v2v_capi::demux::*;
    use v2v_capi::mux::*;
    use v2v_capi::venc::*;

    let (w, h) = (64u32, 48u32);
    let out = std::env::temp_dir().join("v2v_abi_mux_asf.wmv");
    let _ = std::fs::remove_file(&out);

    // ---- 编码几帧 ----
    let cfg = V2vVencCfg {
        size: size_of::<V2vVencCfg>() as u32,
        width: w,
        height: h,
        qscale: 0,
        bitrate: 0,
        fps_num: 0,
        ac_pred: 1,
        mpeg_quant: 0,
    };
    let enc = unsafe { v2v_venc_open(CODEC_MP43, &cfg) };
    assert!(!enc.is_null());

    let mut img = Gradient::new(w, h);
    let mut packets: Vec<(Vec<u8>, i32)> = Vec::new();
    for i in 0..4 {
        img.shift(i as u8);
        let p = img.as_planes();
        let mut pkt: V2vPacket = unsafe { std::mem::zeroed() };
        pkt.size = size_of::<V2vPacket>() as u32;
        assert_eq!(unsafe { v2v_venc_encode(enc, &p, i32::from(i == 0), &mut pkt) }, OK);
        // 立刻拷走：pkt.data 活到下一次 encode。
        let bytes = unsafe { std::slice::from_raw_parts(pkt.data, pkt.len as usize) }.to_vec();
        packets.push((bytes, pkt.key_frame));
    }
    unsafe { v2v_venc_close(enc) };

    // ---- 封成 ASF ----
    let vcfg = V2vMuxVideo {
        size: size_of::<V2vMuxVideo>() as u32,
        width: w,
        height: h,
        fourcc: u32::from_le_bytes(*b"MP43"),
        timescale: 0,
        codec: 0,
        avg_bitrate: 0,
        decoder_specific_info: std::ptr::null(),
        dsi_len: 0,
    };
    let path = CString::new(out.to_str().unwrap()).unwrap();
    let mux = unsafe { v2v_mux_open(MUX_ASF, path.as_ptr(), &vcfg, std::ptr::null()) };
    assert!(!mux.is_null(), "打开封装器失败：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });
    for (i, (bytes, key)) in packets.iter().enumerate() {
        assert_eq!(
            unsafe { v2v_mux_write_video(mux, bytes.as_ptr(), bytes.len() as u32, *key, i as i64 * 40, 40) },
            OK,
            "写第 {i} 帧失败：{}",
            unsafe { std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy() }
        );
    }
    assert_eq!(unsafe { v2v_mux_finish(mux) }, OK);
    // finish 之后不能再写。
    assert_eq!(
        unsafe { v2v_mux_write_video(mux, packets[0].0.as_ptr(), packets[0].0.len() as u32, 1, 0, 40) },
        ERR_ARG
    );
    unsafe { v2v_mux_close(mux) };

    // ---- 读回来 ----
    let meta = std::fs::metadata(&out).expect("产物没落盘");
    assert!(meta.len() > 0, "产物是空文件");

    let d = unsafe { v2v_demux_open(path.as_ptr()) };
    assert!(!d.is_null(), "读不回自己封的文件");
    let v = find_stream(d, STREAM_VIDEO);
    assert_eq!(v.width, w);
    assert_eq!(v.height, h);
    assert_eq!(v.fourcc, u32::from_le_bytes(*b"MP43"));

    let mut n = 0u32;
    let mut keys = 0u32;
    loop {
        let mut s: V2vSample = unsafe { std::mem::zeroed() };
        s.size = size_of::<V2vSample>() as u32;
        match unsafe { v2v_demux_next(d, &mut s) } {
            OK => {
                if s.key_frame != 0 {
                    keys += 1;
                }
                n += 1;
            }
            EOS => break,
            other => panic!("读回来时 next 返回 {other}"),
        }
    }
    assert_eq!(n, packets.len() as u32, "样本数对不上");
    assert_eq!(keys, 1, "只有第一帧是关键帧");
    unsafe { v2v_demux_close(d) };

    let _ = std::fs::remove_file(&out);
}

/// MP4 封装：只验产物是个像样的 MP4（没有 MP4 解封装器，读不回来）。
#[test]
fn muxes_mp4_with_a_valid_header() {
    use v2v_capi::mux::*;
    use v2v_capi::venc::*;

    let (w, h) = (64u32, 48u32);
    let out = std::env::temp_dir().join("v2v_abi_mux.mp4");
    let _ = std::fs::remove_file(&out);

    let cfg = V2vVencCfg {
        size: size_of::<V2vVencCfg>() as u32,
        width: w,
        height: h,
        qscale: 0,
        bitrate: 0,
        fps_num: 25,
        ac_pred: 1,
        mpeg_quant: 0,
    };
    let enc = unsafe { v2v_venc_open(CODEC_MP4V, &cfg) };
    assert!(!enc.is_null());

    // 先取 VOL（封装器建的时候就定死了，之后改不了）。
    let mut dsi = vec![0u8; 512];
    let dsi_len = unsafe { v2v_venc_header(enc, dsi.as_mut_ptr(), dsi.len() as u32) };
    assert!(dsi_len > 0);

    let mut img = Gradient::new(w, h);
    let mut packets: Vec<(Vec<u8>, i32)> = Vec::new();
    for i in 0..3 {
        img.shift(i as u8);
        let p = img.as_planes();
        let mut pkt: V2vPacket = unsafe { std::mem::zeroed() };
        pkt.size = size_of::<V2vPacket>() as u32;
        assert_eq!(unsafe { v2v_venc_encode(enc, &p, i32::from(i == 0), &mut pkt) }, OK);
        packets.push((
            unsafe { std::slice::from_raw_parts(pkt.data, pkt.len as usize) }.to_vec(),
            pkt.key_frame,
        ));
    }
    unsafe { v2v_venc_close(enc) };

    let vcfg = V2vMuxVideo {
        size: size_of::<V2vMuxVideo>() as u32,
        width: w,
        height: h,
        fourcc: 0,
        timescale: 1000,
        codec: MUX_VCODEC_MP4V,
        avg_bitrate: 0,
        decoder_specific_info: dsi.as_ptr(),
        dsi_len: dsi_len as u32,
    };
    let path = CString::new(out.to_str().unwrap()).unwrap();
    let mux = unsafe { v2v_mux_open(MUX_MP4, path.as_ptr(), &vcfg, std::ptr::null()) };
    assert!(!mux.is_null(), "打开 MP4 封装器失败：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });
    for (i, (bytes, key)) in packets.iter().enumerate() {
        assert_eq!(
            unsafe { v2v_mux_write_video(mux, bytes.as_ptr(), bytes.len() as u32, *key, i as i64 * 40, 40) },
            OK
        );
    }
    assert_eq!(unsafe { v2v_mux_finish(mux) }, OK);
    unsafe { v2v_mux_close(mux) };

    let bytes = std::fs::read(&out).expect("产物没落盘");
    assert!(bytes.len() > 32);
    // 第一个 box 必须是 `ftyp`。
    let size = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert!(size >= 8 && size as usize <= bytes.len());
    assert_eq!(&bytes[4..8], b"ftyp", "第一个 box 不是 ftyp");
    // 后面必须能找到 `mdat`（真有码流写进去了）。
    assert!(bytes.windows(4).any(|w| w == b"mdat"), "没找到 mdat");

    let _ = std::fs::remove_file(&out);
}

// ---------------------------------------------------------------------------
// MPEG-1/2（`mpeg12` feature）—— 真样本，缺就 SKIP
// ---------------------------------------------------------------------------

/// 先拿到样本路径，再决定验不验。缺样本就 SKIP（仓规：真样本一律 `.gitignore`，
/// 靠环境变量找）；⚠️ 路径要给绝对的，cargo 跑测试时工作目录是本 crate。
#[cfg(feature = "mpeg12")]
fn mpeg_sample() -> Option<String> {
    std::env::var("V2V_MPEG12_SAMPLE")
        .ok()
        .and_then(|s| s.split(':').find(|p| std::path::Path::new(p).is_file()).map(String::from))
}

/// MPEG-1/2 播放：open → info → 逐帧取到 EOS。
///
/// 验三件事：宽高与 info 一致、帧序**不倒着走**（重排真的管了）、指针在下次调用前有效。
#[cfg(feature = "mpeg12")]
#[test]
fn plays_mpeg_through_the_abi() {
    use v2v_capi::mpeg::*;
    use v2v_capi::player::V2vFrame;

    let Some(path) = mpeg_sample() else { return };
    let c = CString::new(path).unwrap();
    let h = unsafe { v2v_mpeg_open(c.as_ptr()) };
    assert!(!h.is_null(), "打开失败：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });

    let mut info: V2vMpegStreamInfo = unsafe { std::mem::zeroed() };
    info.size = size_of::<V2vMpegStreamInfo>() as u32;
    assert_eq!(unsafe { v2v_mpeg_info(h, &mut info) }, OK);
    assert!(info.width > 0 && info.height > 0, "宽高没取到");
    assert!(info.frame_ms > 0.0, "帧率没取到");
    assert!(info.picture_count > 0, "图像数没取到");

    let mut last_pts = i64::MIN;
    let mut got = 0u32;
    // 只取前 12 帧：这是形状测试，不是「解完整个文件」—— 全解在 debug 下能跑几十秒。
    for _ in 0..12 {
        let mut f: V2vFrame = unsafe { std::mem::zeroed() };
        f.size = size_of::<V2vFrame>() as u32;
        let rc = unsafe { v2v_mpeg_read_video(h, &mut f) };
        if rc == EOS {
            break;
        }
        assert_eq!(rc, OK, "取帧失败：{}", unsafe {
            std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
        });
        assert_eq!(f.planes.width, info.width, "帧宽与 info 不一致");
        assert_eq!(f.planes.height, info.height, "帧高与 info 不一致");
        assert!(!f.planes.y.is_null() && !f.planes.u.is_null() && !f.planes.v.is_null());
        assert!(f.pts_ms >= last_pts, "帧序回退：{} < {}", f.pts_ms, last_pts);
        last_pts = f.pts_ms;
        got += 1;
    }
    assert!(got > 0, "一帧都没取到");
    unsafe { v2v_mpeg_close(h) };
}

/// program stream 拆包：取前几个 PES，至少得有一个视频流。
///
/// 这里**不**要求 pts 单调：PES 是容器序，不保证按显示序排；
/// 也不要求 PES 边界等于帧边界（那正是「想要帧就用 v2v_mpeg_*」的理由）。
#[cfg(feature = "mpeg12")]
#[test]
fn demuxes_mpeg_ps_through_the_abi() {
    use v2v_capi::ps::*;

    let Some(path) = mpeg_sample() else { return };
    let c = CString::new(path).unwrap();
    let h = unsafe { v2v_ps_open(c.as_ptr()) };
    assert!(!h.is_null(), "打开失败：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });

    let mut video = 0u32;
    for _ in 0..200 {
        let mut p: V2vPes = unsafe { std::mem::zeroed() };
        p.size = size_of::<V2vPes>() as u32;
        let rc = unsafe { v2v_ps_next(h, &mut p) };
        if rc == EOS {
            break;
        }
        assert_eq!(rc, OK, "拆包失败：{}", unsafe {
            std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
        });
        assert!(!p.data.is_null() && p.len > 0, "PES 是空的");
        if (0xE0..=0xEF).contains(&p.stream_id) {
            video += 1;
        }
    }
    assert!(video > 0, "一个视频 PES 都没看到");
    unsafe { v2v_ps_close(h) };
}

/// 不是 MPEG 的字节：`open` 必须干净失败（返回 NULL + 错误通道有消息），不是崩。
/// 这条**不需要真样本** —— 验的是围栏，不是解码。
#[cfg(feature = "mpeg12")]
#[test]
fn mpeg_open_on_garbage_is_a_clean_failure() {
    use v2v_capi::mpeg::*;

    let p = std::env::temp_dir().join("v2v_abi_not_mpeg.mpg");
    std::fs::write(&p, b"not a program stream at all, not even close").unwrap();
    let c = CString::new(p.to_str().unwrap()).unwrap();

    let h = unsafe { v2v_mpeg_open(c.as_ptr()) };
    assert!(h.is_null(), "垃圾字节不该打开成功");
    assert!(!v2v_last_error().is_null(), "失败时错误通道必须有消息");
    unsafe { v2v_mpeg_close(h) }; // 传 NULL 必须是空操作

    // 空句柄上的每个入口都必须是 ERR_ARG，而不是段错误。
    let mut info: V2vMpegStreamInfo = unsafe { std::mem::zeroed() };
    info.size = size_of::<V2vMpegStreamInfo>() as u32;
    assert_eq!(unsafe { v2v_mpeg_info(std::ptr::null_mut(), &mut info) }, ERR_ARG);
    let mut f: v2v_capi::player::V2vFrame = unsafe { std::mem::zeroed() };
    f.size = size_of::<v2v_capi::player::V2vFrame>() as u32;
    assert_eq!(unsafe { v2v_mpeg_read_video(std::ptr::null_mut(), &mut f) }, ERR_ARG);

    let _ = std::fs::remove_file(&p);
}

/// PS 拆包对垃圾字节的反应：能打开（它就是读整份字节），但一个包都切不出来。
#[cfg(feature = "mpeg12")]
#[test]
fn ps_on_garbage_yields_eos_not_a_crash() {
    use v2v_capi::ps::*;

    let p = std::env::temp_dir().join("v2v_abi_not_ps.mpg");
    std::fs::write(&p, b"\x00\x00\x01\xb9garbage").unwrap();
    let c = CString::new(p.to_str().unwrap()).unwrap();

    let h = unsafe { v2v_ps_open(c.as_ptr()) };
    assert!(!h.is_null(), "读字节不该失败：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });
    let mut pes: V2vPes = unsafe { std::mem::zeroed() };
    pes.size = size_of::<V2vPes>() as u32;
    assert_eq!(unsafe { v2v_ps_next(h, &mut pes) }, EOS);
    assert_eq!(unsafe { v2v_ps_next(std::ptr::null_mut(), &mut pes) }, ERR_ARG);
    unsafe { v2v_ps_close(h) };
    unsafe { v2v_ps_close(std::ptr::null_mut()) };

    let _ = std::fs::remove_file(&p);
}

/// 出参的 `size` 没填 / 填得太小必须**返回错误**，不是「悄悄不写然后返回 OK」。
///
/// 这条钉的是头文件那句「不填（size = 0）会返回 V2V_ERR_ARG」：
/// 曾经代码只记一条错误就往下走，出口照样 `OK`，调用方拿到 `OK` 加一块
/// 没动过的内存 —— 最难查的一类 bug。
///
/// 不需要真样本：编码器不用打开文件，拿它换一个句柄就够。
#[test]
fn out_struct_size_is_enforced() {
    use v2v_capi::venc::*;

    let (w, h) = (32u32, 32u32);
    let cfg = V2vVencCfg {
        size: size_of::<V2vVencCfg>() as u32,
        width: w,
        height: h,
        qscale: 0,
        bitrate: 0,
        fps_num: 25,
        ac_pred: 1,
        mpeg_quant: 0,
    };
    let enc = unsafe { v2v_venc_open(CODEC_MP4V, &cfg) };
    assert!(!enc.is_null(), "建编码器失败：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });
    let img = Gradient::new(w, h);
    let planes = img.as_planes();

    // 忘了 `V2V_INIT` —— size 是 0。
    let mut pkt: V2vPacket = unsafe { std::mem::zeroed() };
    pkt.size = 0;
    assert_eq!(unsafe { v2v_venc_encode(enc, &planes, 1, &mut pkt) }, ERR_ARG);
    assert_eq!(v2v_last_error_kind(), KIND_BAD);

    // 连 4 字节都不到 —— 同样不写、同样报错。
    pkt.size = 2;
    assert_eq!(unsafe { v2v_venc_encode(enc, &planes, 1, &mut pkt) }, ERR_ARG);

    // 填对了才成功。
    pkt.size = size_of::<V2vPacket>() as u32;
    assert_eq!(unsafe { v2v_venc_encode(enc, &planes, 1, &mut pkt) }, OK);
    assert!(pkt.len > 0);

    unsafe { v2v_venc_close(enc) };
}

/// `v2v_planes_copy_to`：拷进带 padding 的目标、逐行对、padding 不动。
///
/// 这条不需要样本 —— 用 `Gradient` 造一帧就够。C 侧 `smoke.c` 里还有一份等价
/// 的（在真样本 720p 上跑），两份都要有：Rust 这份跑得快、C 那份证明头文件的
/// 签名没写错。
#[test]
fn planes_copy_respects_destination_stride() {
    let (w, h) = (64u32, 48u32);
    let img = Gradient::new(w, h);
    let src = img.as_planes();

    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (ys, us, vs) = (w + 16, cw + 8, cw + 8);
    let mut dy = vec![0x5Au8; (ys * h) as usize];
    let mut du = vec![0x5Au8; (us * ch) as usize];
    let mut dv = vec![0x5Au8; (vs * ch) as usize];

    let rc = unsafe {
        v2v_capi::common::v2v_planes_copy_to(
            &src,
            dy.as_mut_ptr(),
            ys,
            du.as_mut_ptr(),
            us,
            dv.as_mut_ptr(),
            vs,
        )
    };
    assert_eq!(rc, OK);

    let sy = unsafe { std::slice::from_raw_parts(src.y, (w * h) as usize) };
    let su = unsafe { std::slice::from_raw_parts(src.u, (cw * ch) as usize) };
    for r in 0..h as usize {
        assert_eq!(&dy[r * ys as usize..r * ys as usize + w as usize], &sy[r * w as usize..][..w as usize]);
        assert!(
            dy[r * ys as usize + w as usize..(r + 1) * ys as usize].iter().all(|b| *b == 0x5A),
            "亮度第 {r} 行的 padding 被写了"
        );
    }
    for r in 0..ch as usize {
        assert_eq!(&du[r * us as usize..r * us as usize + cw as usize], &su[r * cw as usize..][..cw as usize]);
        assert!(
            du[r * us as usize + cw as usize..(r + 1) * us as usize].iter().all(|b| *b == 0x5A),
            "U 第 {r} 行的 padding 被写了"
        );
    }

    // stride 传 0 = 用源的 stride。源紧排，所以整块等于源。
    let mut ty = vec![0u8; (w * h) as usize];
    let rc = unsafe {
        v2v_capi::common::v2v_planes_copy_to(&src, ty.as_mut_ptr(), 0, std::ptr::null_mut(), 0, std::ptr::null_mut(), 0)
    };
    assert_eq!(rc, OK);
    assert_eq!(ty, sy, "stride = 0 那条路与源不一致");

    // 三块目标都空 → 明确报错，不静默成功。
    let rc = unsafe {
        v2v_capi::common::v2v_planes_copy_to(&src, std::ptr::null_mut(), 0, std::ptr::null_mut(), 0, std::ptr::null_mut(), 0)
    };
    assert_eq!(rc, ERR_ARG);

    // 目标 stride 比宽度小 → 拒绝（不折行硬拷）。
    let rc = unsafe {
        v2v_capi::common::v2v_planes_copy_to(&src, dy.as_mut_ptr(), w - 1, std::ptr::null_mut(), 0, std::ptr::null_mut(), 0)
    };
    assert_eq!(rc, ERR_ARG);
}
