//! 端到端：**WMV → MP4，解码用我们的（对 ffmpeg 逐字节一致），视频用平台硬件
//! 编码器，封装用我们自己的 mp4 muxer** —— 「取优点」的完整形态。
//!
//! ```text
//! cargo run --release -p v2v-android --example wmv_to_mp4_hw -- \
//!     testing/sample.wmv /tmp/sample_hw.mp4 --video-bitrate 4000000
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



struct Args {
    input: PathBuf,
    output: PathBuf,
    video_bitrate: u32,
    audio_bitrate: u32,
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
    };
    while let Some(s) = it.next() {
        match s.as_str() {
            "--video-bitrate" => a.video_bitrate = it.next()?.parse().ok()?,
            "--audio-bitrate" => a.audio_bitrate = it.next()?.parse().ok()?,
            _ => return None,
        }
    }
    Some(a)
}

fn main() {
    let Some(args) = parse_args() else {
        eprintln!("用法: wmv_to_mp4_hw <in.wmv> <out.mp4> [--video-bitrate N] [--audio-bitrate N]");
        std::process::exit(2);
    };

    let t = std::time::Instant::now();
    let stats = v2v_android::transcode::wmv_to_mp4_hw(
        &args.input,
        &args.output,
        args.video_bitrate,
        args.audio_bitrate,
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
