//! 端到端：**WMV → MP4，解码用我们的（对 ffmpeg 逐字节一致），视频用平台硬件
//! 编码器，封装用我们自己的 mp4 muxer** —— 「取优点」的完整形态。
//!
//! ```text
//! cargo run --release -p v2v-android --example wmv_to_mp4_hw -- \
//!     testing/sample.wmv /tmp/sample_hw.mp4 --video-bitrate 4000000
//! ```
//!
//! 量曲线用的三个旋钮（默认值就是实测最优的那一档，不传就别传）：
//!
//! ```text
//! --workers N        worker 数（默认按核数，封顶 16；实测 6 个就喂满硬件编码器）
//! --chunk-secs N     每块时长（秒，默认 3；2 s 会话太多、10 s 并行度不够）
//! --profile          把 [probe] 诊断打到 stderr（块数 / 音频线程耗时）
//! ```
//!
//! 后端按平台选：
//! - **macOS**：VideoToolbox（Apple Silicon 上就是硬件编码器）→ 长度前缀 NALU
//!   样本 + avcC → `v2v_mp4` 的 `VideoCodec::Avc1`；
//! - **Android**：MediaCodec（自带 MediaMuxer 的 `Mp4HwWriter`）；
//! - 其它：报 `Unsupported`。
//!
//! 音频两条路一致：WMA → f32 → 我们的 AAC-LC 编码器 → `esds` 轨。

use std::path::PathBuf;

use v2v_android::transcode::{wmv_to_mp4_hw_with, HwOptions};

struct Args {
    input: PathBuf,
    output: PathBuf,
    video_bitrate: u32,
    audio_bitrate: u32,
    opts: HwOptions,
}

fn parse_args() -> Option<Args> {
    let mut it = std::env::args().skip(1);
    let input = PathBuf::from(it.next()?);
    let output = PathBuf::from(it.next()?);
    let mut a = Args {
        input,
        output,
        video_bitrate: 4_000_000,
        audio_bitrate: 128_000,
        opts: HwOptions::default(),
    };
    while let Some(s) = it.next() {
        match s.as_str() {
            "--video-bitrate" => a.video_bitrate = it.next()?.parse().ok()?,
            "--audio-bitrate" => a.audio_bitrate = it.next()?.parse().ok()?,
            "--workers" => a.opts.workers = it.next()?.parse().ok()?,
            "--chunk-secs" => a.opts.chunk_secs = it.next()?.parse().ok()?,
            "--profile" => a.opts.profile = true,
            _ => return None,
        }
    }
    Some(a)
}

fn main() {
    let Some(args) = parse_args() else {
        eprintln!(
            "用法: wmv_to_mp4_hw <in.wmv> <out.mp4> \
             [--video-bitrate N] [--audio-bitrate N] \
             [--workers N] [--chunk-secs N] [--profile]"
        );
        std::process::exit(2);
    };

    let t = std::time::Instant::now();
    let stats = wmv_to_mp4_hw_with(
        &args.input,
        &args.output,
        args.video_bitrate,
        args.audio_bitrate,
        args.opts,
    )
    .expect("硬件转码失败");
    let dt = t.elapsed();
    println!(
        "视频 {} 帧 | 音频 {} 段 | 全链路 {:.2} s | {:.1} fps | 产物 {} B",
        stats.frames,
        stats.audio_chunks,
        dt.as_secs_f64(),
        stats.frames as f64 / dt.as_secs_f64(),
        stats.output_bytes
    );
}
