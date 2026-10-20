//! 端到端示例：**WMV → MP4，解码用我们的（逐字节对 ffmpeg），编码用 Android 硬件**。
//!
//! ```text
//! cargo run --release -p v2v-android --example wmv_to_mp4_android -- \
//!     testing/sample.wmv /tmp/sample_hw.mp4 --video-bitrate 4000000 --audio-bitrate 128000
//! ```
//!
//! 非 Android 上运行会立即报 `Unsupported`（ Mp4HwWriter 的桩实现）——
//! 这是预期行为；真机路径用 NDK 交叉编译或在设备上跑。
//!
//! 流程：`v2v_player` 解封装+解码 → 视频 YUV420P 直喂 MediaCodec；
//! 音频走我们自己的 AAC 编码器（质量与软件栈一致），输出裸帧 + ASC 给 MediaMuxer。

use std::path::PathBuf;

use v2v_aac::{Encoder as AacEncoder, EncoderConfig as AacConfig, Frame as AacFrame};
use v2v_android::Mp4HwWriter;
use v2v_player::Player;

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
        eprintln!(
            "用法: wmv_to_mp4_android <in.wmv> <out.mp4> [--video-bitrate N] [--audio-bitrate N]"
        );
        std::process::exit(2);
    };

    let mut player = Player::open(&args.input).expect("打开输入失败");
    let info = player.info().clone();
    println!(
        "{}  {}x{}  时长 {} ms  音频 {} Hz × {}",
        args.input.display(),
        info.width,
        info.height,
        info.duration_ms,
        info.audio_sample_rate,
        info.audio_channels,
    );

    // 帧率从时间戳推（MediaCodec 只要整数）。用**一次性的 Player** 探 ——
    // 不能用主 player：它会把前 100 帧消费掉，输出就缺了开头。
    let fps = infer_fps(&args.input);
    let mut hw = Mp4HwWriter::open(
        &args.output,
        info.width,
        info.height,
        args.video_bitrate,
        fps,
        4, // GOP ≈ 4 s，与软件栈对垒时同参数
        info.has_audio,
    )
    .expect("打开硬件编码器失败");

    // ---- 音频：我们的 AAC 编码器（WMA → f32 → AAC-LC）----
    let mut aac = info.has_audio.then(|| {
        AacEncoder::new(AacConfig {
            channels: info.audio_channels as u8,
            sample_rate: info.audio_sample_rate,
            bitrate: args.audio_bitrate,
            ..AacConfig::default()
        })
        .expect("建 AAC 编码器")
    });
    if let (Some(enc), true) = (&aac, info.has_audio) {
        let asc = enc.audio_specific_config().to_bytes();
        hw.add_audio_track(&asc, info.audio_sample_rate, info.audio_channels as u8)
            .expect("加音频轨");
    }
    // AAC 编码器的样本缓冲：WMA 每次给 2048 帧，AAC-LC 每 1024 帧一帧 —— 攒着切。
    let mut pcm_buf: Vec<Vec<f32>> = Vec::new();
    let mut pcm_emitted = 0u64;
    let sample_rate = info.audio_sample_rate as u64;
    let ch = info.audio_channels as usize;

    let t = std::time::Instant::now();
    let mut vframes = 0u64;
    loop {
        let mut got = false;
        if let Some(f) = player.next_video().expect("视频解帧出错") {
            got = true;
            hw.write_video(&f.y, &f.u, &f.v, f.pts_ms * 1000, false)
                .expect("硬件编码视频出错");
            vframes += 1;
        }
        if let Some(enc) = &mut aac {
            if let Some(a) = player.next_audio().expect("音频解帧出错") {
                got = true;
                for (dst, src) in pcm_buf.iter_mut().zip(a.planes) {
                    dst.extend(src);
                }
                // 攒够 1024 帧一段就编一帧 AAC。
                let samples = pcm_buf.first().map_or(0, |p| p.len());
                while samples >= 1024 {
                    let mut planes = Vec::with_capacity(ch);
                    for p in &mut pcm_buf {
                        planes.push(p.drain(..1024).collect::<Vec<_>>());
                    }
                    let pts_us = (pcm_emitted * 1_000_000 / sample_rate) as i64;
                    pcm_emitted += 1024;
                    let bytes = enc.encode_frame(&AacFrame { planes }).expect("AAC 编码出错");
                    hw.write_aac(&bytes, pts_us).expect("写音频出错");
                    let samples = pcm_buf.first().map_or(0, |p| p.len());
                    if samples < 1024 {
                        break;
                    }
                }
            }
        }
        if !got {
            break;
        }
    }
    hw.finish().expect("收尾失败");
    let dt = t.elapsed();
    println!(
        "视频 {} 帧 | 用时 {:.2} s | {:.1} fps（硬件路径）",
        vframes,
        dt.as_secs_f64(),
        vframes as f64 / dt.as_secs_f64(),
    );
}

/// 从前 100 帧的时间戳差推帧率（容错：时间戳可能不从 0 开始）。
fn infer_fps(path: &std::path::Path) -> u32 {
    let Ok(mut p) = Player::open(path) else {
        return 30;
    };
    let mut ts = Vec::new();
    for _ in 0..100 {
        match p.next_video().expect("解帧出错") {
            Some(f) => ts.push(f.pts_ms),
            None => break,
        }
    }
    if ts.len() >= 2 {
        let span = *ts.last().unwrap() - ts[0];
        let steps = ts.len() - 1;
        if span > 0 {
            return ((steps as f64 * 1000.0 / span as f64).round() as u32).max(1);
        }
    }
    30
}
