//! 封装吞吐基准 —— 回答「封装是不是瓶颈」。
//!
//! 平时 `#[ignore]`（不进日常 `cargo test`），要量的时候显式跑：
//!
//! ```bash
//! cargo test -p v2v-mp4 --release --test mux_bench -- --ignored --nocapture
//! ```
//!
//! ## 为什么单独量封装
//!
//! 端到端慢的时候，第一反应容易怀疑「最后那一段」（封装）。但封装是
//! **纯内存追加**：`write_video` / `write_audio` 各是一次 `Vec::extend_from_slice`
//! 加一条 `SampleRec`，`finish()` 补 `mdat` 长度、追加一次 `moov`。
//! 它是 O(总字节数)，没有二次方行为，也不做 I/O。
//!
//! 所以这个基准的作用是**给出可对照的数**：封装吞吐（MB/s）与端到端帧率放在一起，
//! 就能一眼看出封装占多少。**别用「感觉」判断哪一段慢。**
//!
//! ⚠️ 参数写死在下面（样本数 / 样本大小），是因为它**不依赖任何真样本** ——
//! 换成别的数，结论只会在同一量级内浮动，不会翻转。

use std::time::Instant;

use v2v_mp4::{AudioCodec, AudioTrack, Mp4Muxer, VideoTrack};

/// 视频样本：模拟 1080p / 30 fps / 约 1.5 Mbps 的 mp4v 帧（关键帧更大）。
const VIDEO_FRAMES: usize = 900;
const P_FRAME_BYTES: usize = 6_000;
const I_FRAME_BYTES: usize = 60_000;
const GOP: usize = 12;

/// 音频样本：AAC 128 kbps / 48 kHz 的一个帧（1024 采样点）。
const AUDIO_FRAME_BYTES: usize = 384;
/// 视频每 2 帧配 1 个音频帧（48 kHz 下 AAC 帧率约 47/s，30 fps 视频约 1.56:1，
/// 取 2:1 是为了让循环简单，量级一致）。
const AUDIO_PER_VIDEO: usize = 2;

fn video_track() -> VideoTrack {
    VideoTrack {
        width: 1920,
        height: 1080,
        timescale: 30_000,
        codec: v2v_mp4::VideoCodec::Mp4v {
            decoder_specific_info: vec![0, 0, 1, 0xb0, 0x01],
        },
        avg_bitrate: 1_500_000,
    }
}

fn audio_track() -> AudioTrack {
    AudioTrack {
        channels: 2,
        sample_rate: 48_000,
        timescale: 48_000,
        codec: AudioCodec::Es {
            object_type: 0x40,
            decoder_specific_info: vec![0x11, 0x90],
        },
        avg_bitrate: 128_000,
        priming: 0,
    }
}

#[test]
#[ignore = "吞吐基准，日常测试不跑"]
fn mux_throughput() {
    let p_frame = vec![0x5au8; P_FRAME_BYTES];
    let i_frame = vec![0xa5u8; I_FRAME_BYTES];
    let a_frame = vec![0x3cu8; AUDIO_FRAME_BYTES];

    let mut m = Mp4Muxer::with_audio(video_track(), audio_track());

    // 只量「写入样本」这一段。
    let t_write = Instant::now();
    let mut audio_written = 0usize;
    for i in 0..VIDEO_FRAMES {
        let key = i % GOP == 0;
        let data = if key { &i_frame } else { &p_frame };
        m.write_video(data, key, 1001).expect("写视频样本");
        if i % AUDIO_PER_VIDEO == 0 {
            m.write_audio(&a_frame, 1024).expect("写音频样本");
            audio_written += 1;
        }
    }
    let write_elapsed = t_write.elapsed();

    let payload = VIDEO_FRAMES / GOP * I_FRAME_BYTES
        + (VIDEO_FRAMES - VIDEO_FRAMES / GOP) * P_FRAME_BYTES
        + audio_written * AUDIO_FRAME_BYTES;

    // 再量 `finish()`（补 mdat 长度 + 追加 moov）。
    let t_finish = Instant::now();
    let out = m.finish();
    let finish_elapsed = t_finish.elapsed();

    let write_mb = payload as f64 / (1024.0 * 1024.0);
    let total_mb = out.len() as f64 / (1024.0 * 1024.0);

    println!(
        "写入 {VIDEO_FRAMES} 视频帧 + {audio_written} 音频帧 = {write_mb:.1} MiB（mdat 载荷）"
    );
    println!(
        "  write_*  {:?}  →  {:.0} MiB/s",
        write_elapsed,
        write_mb / write_elapsed.as_secs_f64()
    );
    println!(
        "  finish() {:?}  →  {:.0} MiB/s（产出文件 {total_mb:.1} MiB）",
        finish_elapsed,
        total_mb / finish_elapsed.as_secs_f64()
    );
    println!(
        "  合计     {:?}  →  {:.0} MiB/s",
        write_elapsed + finish_elapsed,
        total_mb / (write_elapsed + finish_elapsed).as_secs_f64()
    );

    // 粗判据：封装是内存拷贝，慢于 50 MiB/s 就说明有意外开销（不是「慢」的绝对值，
    // 而是「不该这么慢」的下限 —— 换个机器只会在同一量级浮动）。
    let mbps = total_mb / (write_elapsed + finish_elapsed).as_secs_f64();
    assert!(
        mbps > 50.0,
        "封装吞吐只有 {mbps:.0} MiB/s —— 内存追加不该这么慢，先查是不是有隐藏拷贝"
    );
}
