//! `.ogg → .wav` 的端到端判据：**解封装 → Vorbis 解码 → WAV**，对着 `ffmpeg` 比。
//!
//! ## 为什么这条测试要在 CLI 这一层
//!
//! `crates/v2v-vorbis/tests/real_sample.rs` 已经证明「解码器解出来的波形对」。
//! 但那条测试**自己拆包、自己拼 PCM**，绕过了真正要用的那条路：`open_source`
//! 按魔数认容器、`classify_audio` 派发、`split_vorbis_headers` 拆头、
//! `audio_total_samples` 裁尾巴、`wav::wrap_pcm16` 写头。这些环节任何一个错了，
//! 解码器再对也没用 —— 而且**不会报错**（最典型的是单位差 32768 倍：
//! Vorbis 吐 ±1，WMA / AAC 吐 ±32768，接错了就是一片死寂）。
//!
//! ## 判据
//!
//! 1. 采样点数与 `ffmpeg` **完全相等**（不是「差不多」—— 尾巴裁剪错了就会差几十个）；
//! 2. 逐样本差 **≤ 1 LSB**（s16 舍入差，与 AAC 解码器同一个口径）；
//! 3. 峰值不是 0（静音骗局）。
//!
//! 需要 `V2V_OGG_SAMPLE` 与 `ffmpeg`；缺任一个就跳过。

use std::path::Path;
use std::process::{Command, Stdio};

use v2v_cli::{convert_file, ConvertOptions, OutputFormat};

fn sample_path() -> Option<std::path::PathBuf> {
    let p = std::env::var_os("V2V_OGG_SAMPLE")?;
    let p = std::path::PathBuf::from(p);
    if !p.is_file() {
        eprintln!("SKIP: V2V_OGG_SAMPLE 指向的文件不存在");
        return None;
    }
    Some(p)
}

fn has_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn unique_temp(stem: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{}_{n}", std::process::id()))
}

/// 把 `.wav` 的 `data` 段读成 i16（跳过 44 字节头 —— 那是我们自己写的，另有单测）。
fn wav_pcm(path: &Path) -> Vec<i16> {
    let b = std::fs::read(path).expect("读 wav");
    assert_eq!(&b[0..4], b"RIFF");
    assert_eq!(&b[8..12], b"WAVE");
    assert_eq!(&b[36..40], b"data");
    let n = u32::from_le_bytes(b[40..44].try_into().unwrap()) as usize;
    assert_eq!(n + 44, b.len(), "data 段的长度要和文件长度对得上");
    b[44..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[test]
fn ogg_to_wav_matches_ffmpeg_within_one_lsb() {
    let Some(input) = sample_path() else { return };
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }

    let out_dir = unique_temp("v2v_ogg_wav");
    std::fs::create_dir_all(&out_dir).unwrap();
    let out = out_dir.join("out.wav");
    let opts = ConvertOptions { format: OutputFormat::Wav, ..Default::default() };
    let rep = convert_file(&input, &out, &opts).expect("转码");
    println!(
        "{} → {}：{} 个采样点，{} 字节",
        input.display(),
        out.display(),
        rep.audio_payloads,
        rep.bytes_out
    );
    assert_eq!(rep.frames, 0, "WAV 这条路不该有视频帧");

    // ffmpeg 的同一段 PCM（s16le，与我们的 WAV 载荷同口径）。
    let raw = unique_temp("v2v_ogg_ref.s16");
    let st = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            input.to_str().unwrap(),
            "-f",
            "s16le",
            "-y",
            raw.to_str().unwrap(),
        ])
        .status()
        .expect("跑 ffmpeg");
    assert!(st.success(), "ffmpeg 解不开这个样本");
    let rb = std::fs::read(&raw).unwrap();
    let reference: Vec<i16> = rb
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let ours = wav_pcm(&out);
    // ---- 判据 1：长度 ----
    //
    // 必须**完全相等**：我们的解码器按块输出，最后一块会越过真实长度，
    // 靠 `granule_position` 裁掉。裁剪没生效的话这里会多出几十个采样。
    assert_eq!(
        ours.len(),
        reference.len(),
        "采样点数与 ffmpeg 不同（我们 {}，ffmpeg {}）—— 尾巴裁剪没生效？",
        ours.len(),
        reference.len()
    );

    // ---- 判据 3（先查）：不是静音 ----
    let peak = ours.iter().map(|v| (*v as i32).abs()).max().unwrap_or(0);
    assert!(peak > 1000, "输出峰值只有 {peak} —— 单位差（±1 vs ±32768）或全静音");

    // ---- 判据 2：逐样本 ----
    let mut worst = 0i32;
    let mut diff = 0usize;
    for (a, b) in ours.iter().zip(&reference) {
        let d = (*a as i32 - *b as i32).abs();
        if d > 0 {
            diff += 1;
        }
        worst = worst.max(d);
    }
    println!("最大差 {worst} LSB，不同的样本 {diff}/{}", ours.len());
    assert!(worst <= 1, "最大差 {worst} LSB —— 超过 s16 舍入的范围");

    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_file(&raw);
}
