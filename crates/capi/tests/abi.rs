//! C ABI 自身的冒烟测试 —— **不需要真样本**。
//!
//! 这里验的是「ABI 的形状」而不是「解码对不对」：
//! 版本号、错误通道、panic 围栏、空指针与 `size` 的处理。
//! 解码对不对由各 crate 自己的对账测试管，ABI 只负责别把错误传丢。

use std::ffi::{c_char, CString};
use std::mem::size_of;

use v2v_capi::common::*;
use v2v_capi::player::{v2v_player_info, v2v_player_open, V2vInfo};

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
    let mut frames = 0u32;
    let mut last_pts = i64::MIN;
    loop {
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

    // 逐样本拉音频（没有音轨就是立刻 EOS）。
    loop {
        let mut a = V2vAudioFrame {
            size: size_of::<V2vAudioFrame>() as u32,
            pts_ms: 0,
            planes: std::ptr::null(),
            channels: 0,
            samples: 0,
            sample_rate: 0,
        };
        match unsafe { v2v_player_read_audio(h, &mut a) } {
            OK => assert!(!a.planes.is_null() && a.samples > 0),
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

    // 嗅探：读文件头 64 字节，必须认成 ASF。
    let head = std::fs::read(&path).unwrap();
    let k = unsafe { v2v_probe(head.as_ptr(), head.len().min(64) as u32) };
    assert_eq!(k, CONTAINER_ASF);

    let c = CString::new(path).unwrap();
    let h = unsafe { v2v_demux_open(c.as_ptr()) };
    assert!(!h.is_null(), "打不开：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });

    let n = unsafe { v2v_demux_stream_count(h) };
    assert!(n >= 1, "至少得有一条流");

    // 第一条应该是视频，且 extra 里带着 VC-1 的序列头。
    let mut si: V2vStreamInfo = unsafe { std::mem::zeroed() };
    si.size = size_of::<V2vStreamInfo>() as u32;
    assert_eq!(unsafe { v2v_demux_stream(h, 0, &mut si) }, OK);
    assert_eq!(si.kind, STREAM_VIDEO);
    assert!(si.width > 0 && si.height > 0);
    assert_ne!(si.fourcc, 0);
    assert!(si.extra_len > 0 && !si.extra.is_null(), "视频流该带序列头");

    // 逐样本拉：时间戳必须单调不减，data 必须非空。
    let mut count = 0u32;
    let mut last = i64::MIN;
    loop {
        let mut s: V2vSample = unsafe { std::mem::zeroed() };
        s.size = size_of::<V2vSample>() as u32;
        match unsafe { v2v_demux_next(h, &mut s) } {
            OK => {
                assert!(!s.data.is_null() && s.data_len > 0);
                assert!(s.timestamp_ms >= last, "时间戳回退");
                last = s.timestamp_ms;
                count += 1;
            }
            EOS => break,
            other => panic!("next 返回 {other}"),
        }
    }
    assert!(count > 0, "一个样本都没读出来");

    unsafe { v2v_demux_close(h) };
}
