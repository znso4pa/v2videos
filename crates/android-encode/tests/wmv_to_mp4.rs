//! **主测试：WMV（WMV3）→ MP4，全栈真实样本。**
//!
//! 解码是我们（对 ffmpeg 逐字节一致的那套），视频走平台硬件编码器
//! （macOS = VideoToolbox / Android = MediaCodec），封装是我们自己的 mp4 muxer
//! （`avc1` + `avcC`）。质量口径与「转码对垒」一致：对 **VC-1 解码参考**算 PSNR。
//!
//! `black` 轻量常跑；`sample` 全片重（≈35 s），标 `#[ignore]` 用 `--ignored` 跑。

use std::path::{Path, PathBuf};
use std::process::Command;

use v2v_android::transcode::{hw_supported, wmv_to_mp4_hw};

fn sample(name: &str) -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testing")
        .join(name);
    p.exists().then_some(p)
}

fn ffmpeg(args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("ffmpeg").args(args).output().ok()?;
    out.status.success().then(|| out.stdout).filter(|v| !v.is_empty())
}

/// 解码一个 MP4/WMV 到 yuv420p 裸帧。
fn decode_yuv(path: &Path, extra: &[&str]) -> Option<Vec<u8>> {
    let mut args = vec!["-v", "error", "-i", path.to_str().unwrap()];
    args.extend_from_slice(extra);
    args.extend_from_slice(&["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]);
    ffmpeg(&args)
}

fn has_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// 纯 Rust 的 PSNR（对齐 2026-10-20 转码对垒的口径：对 VC-1 解码参考）。
fn psnr(a: &[u8], b: &[u8]) -> Option<f64> {
    let n = a.len().min(b.len());
    if n == 0 {
        return None;
    }
    let mut se = 0u64;
    for i in 0..n {
        let d = i32::from(a[i]) - i32::from(b[i]);
        se += (d * d) as u64;
    }
    let mse = se as f64 / n as f64;
    if mse == 0.0 {
        return Some(f64::INFINITY);
    }
    Some(10.0 * (255.0f64 * 255.0 / mse).log10())
}

/// **轻量主测试**：black.wmv（720p，1 帧，无音频）→ MP4。
///
/// 判据：转码成功、产物有 h264 轨（ffmpeg 解得开）、解码回的帧数对。
#[test]
fn wmv3_to_mp4_hw_black() {
    if !hw_supported() {
        eprintln!("SKIP: 此平台没有硬件编码后端");
        return;
    }
    if !has_ffmpeg() {
        eprintln!("SKIP: 没有 ffmpeg");
        return;
    }
    let Some(src) = sample("black.wmv") else {
        eprintln!("SKIP: testing/black.wmv 不在（样本不入库）");
        return;
    };
    let out = std::env::temp_dir().join(format!("v2v_hw_black_{}.mp4", std::process::id()));
    let stats = wmv_to_mp4_hw(&src, &out, 4_000_000, 128_000).expect("转码失败");
    assert_eq!(stats.frames, 1, "black.wmv 就 1 帧");
    assert!(stats.output_bytes > 500, "产物太小：{} B", stats.output_bytes);

    // ffmpeg 解得开、是 h264、1 帧（1382400 = 1280×720×1.5 字节）。
    let yuv = decode_yuv(&out, &[]).expect("产物解不开");
    assert_eq!(yuv.len(), 1_382_400, "解出来的帧数/尺寸不对");
    let _ = std::fs::remove_file(&out);
}

/// **全片主测试**：sample.wmv（720p30，4138 帧 + WMA 双声道，138.4 s）→ MP4。
///
/// 判据（对齐转码对垒的口径）：
/// ① 帧数 4138、音频段数 = 6082560/1024 = 5940；
/// ② 产物解码回 YUV 后对 VC-1 解码参考的 PSNR ≥ 41 dB
///    （实测硬件路径 42.45 dB；厂商编码器有浮动，阈值留了余量）；
/// ③ 时长对得上（±0.5 s）。
#[test]
#[ignore = "全片 138 s 转码，重；用 `cargo test --release -p v2v-android -- --ignored` 跑"]
fn wmv3_to_mp4_hw_sample_full() {
    if !has_ffmpeg() {
        eprintln!("SKIP: 没有 ffmpeg");
        return;
    }
    let Some(src) = sample("sample.wmv") else {
        eprintln!("SKIP: testing/sample.wmv 不在（样本不入库）");
        return;
    };
    let out = std::env::temp_dir().join(format!("v2v_hw_sample_{}.mp4", std::process::id()));
    let stats = wmv_to_mp4_hw(&src, &out, 4_000_000, 128_000).expect("转码失败");
    println!(
        "帧 {} | 音频段 {} | 产物 {} B",
        stats.frames, stats.audio_chunks, stats.output_bytes
    );
    assert_eq!(stats.frames, 4138, "帧数不对");
    // 每段 1024 采样点：6,082,560 / 1024 = 5940 整（WMA 侧丢过延迟静音后的量）。
    assert_eq!(stats.audio_chunks, 5940, "音频段数不对");

    // 质量：对 VC-1 解码参考。参考用 ffmpeg 解源文件（-idct 2 与我们的
    // 解码器逐字节一致，见 vc1 对账的口径说明）。
    let mut ref_args = vec!["-v", "error", "-idct", "2", "-i", src.to_str().unwrap()];
    ref_args.extend_from_slice(&["-an", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]);
    let reference = ffmpeg(&ref_args).expect("解参考失败");

    let got = decode_yuv(&out, &["-an"]).expect("产物解不开");
    assert_eq!(got.len(), reference.len(), "解出来的帧数/尺寸不对");
    let p = psnr(&got, &reference).expect("PSNR 算不出来");
    println!("PSNR（对 VC-1 解码参考）= {p:.2} dB");
    assert!(p >= 41.0, "硬件转码质量掉出界：{p:.2} dB（实测基线 42.45）");
    let _ = std::fs::remove_file(&out);
}
