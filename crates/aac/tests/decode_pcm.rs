//! AAC **解码器**对账 ffmpeg —— M3 的硬判据。
//!
//! 两条独立的路，缺一条都不算验过：
//!
//! 1. **我们的编码器 → 两个解码器**：位流是我们写的，语法错没错由 ffmpeg 判；
//!    解出来的 PCM 与 ffmpeg 的解逐样本比 —— 定标 / M/S / TNS / IMDCT 一起验。
//! 2. **ffmpeg 的编码器 → 两个解码器**：这条才是真判据。ffmpeg 会用它自己的
//!    窗序列、M/S、TNS 决策，覆盖到我们编码器**不产**的那些组合（逐带掩码、
//!    不同的分组、KBD 窗）。
//!
//! 容差 **±1 LSB**（int16 域）。比的是**同一个位流**的两个解码器，所以
//! 「编码器延迟」「priming」在两边是同一个偏移，不需要对齐 —— 只有样本数
//! 不一致才要停下来看（那说明有一边丢帧或补帧）。
//!
//! ```bash
//! cargo test -p v2v-aac --test decode_pcm -- --nocapture
//! ```
//!
//! 需要 `ffmpeg` 在 `PATH` 上（与 `ffmpeg_roundtrip.rs` 同一个前提）。

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use v2v_aac::{Decoder, Encoder, EncoderConfig, Frame};

const SR: u32 = 44_100;

fn has_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// 进程内唯一的临时文件（`cargo test` 多线程跑，共用一个路径会随机读到半截）。
fn unique_temp(stem: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{}_{n}", std::process::id()))
}

/// 纯正弦（不触发短窗、不触发 TNS）—— 用来把「基础路径」与「瞬态路径」分开。
fn pure_tone(n: usize, channels: usize) -> Vec<Vec<f32>> {
    (0..channels)
        .map(|c| {
            (0..n)
                .map(|i| {
                    let t = i as f32 / SR as f32;
                    (2.0 * std::f32::consts::PI * (440.0 + 110.0 * c as f32) * t).sin() * 4200.0
                })
                .collect()
        })
        .collect()
}

/// 一段用来打满各条路径的信号：低频正弦 + 每 0.25 s 一个瞬态 + 一点噪声。
///
/// 为什么要瞬态：短窗（`EIGHT_SHORT`）与 `LONG_START` / `LONG_STOP` 只有它才触发；
/// 纯正弦走的是同一条长窗路径，测不到最容易错的那一半。
fn test_signal(n: usize, channels: usize) -> Vec<Vec<f32>> {
    let mut state = 0x1234_5678u32;
    (0..channels)
        .map(|c| {
            (0..n)
                .map(|i| {
                    let t = i as f32 / SR as f32;
                    let tone = (2.0 * std::f32::consts::PI * (440.0 + 110.0 * c as f32) * t).sin();
                    // 每 0.25 s 一个短促的冲击（触发起始/短窗/终止那一串状态机）
                    let phase = i % (SR as usize / 4);
                    let click = if phase < 64 {
                        (1.0 - phase as f32 / 64.0) * if c == 0 { 0.9 } else { 0.7 }
                    } else {
                        0.0
                    };
                    state = state.wrapping_mul(1103515245).wrapping_add(12345);
                    let noise = ((state >> 16) as f32 / 32768.0 - 1.0) * 0.02;
                    (tone * 0.35 + click + noise) * 12_000.0
                })
                .collect()
        })
        .collect()
}

/// 写一个 s16le 的 WAV（`ffmpeg` 的输入）。
fn write_wav(path: &Path, planes: &[Vec<f32>], sr: u32) {
    let ch = planes.len();
    let n = planes[0].len();
    let mut data = Vec::with_capacity(n * ch * 2);
    for i in 0..n {
        for p in planes {
            let v = p[i].round().clamp(-32768.0, 32767.0) as i16;
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    let mut w = Vec::with_capacity(44 + data.len());
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&((36 + data.len()) as u32).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); // PCM
    w.extend_from_slice(&(ch as u16).to_le_bytes());
    w.extend_from_slice(&sr.to_le_bytes());
    w.extend_from_slice(&(sr * ch as u32 * 2).to_le_bytes());
    w.extend_from_slice(&((ch * 2) as u16).to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&(data.len() as u32).to_le_bytes());
    w.extend_from_slice(&data);
    std::fs::write(path, w).expect("写 WAV");
}

/// ffmpeg 编码成 ADTS 裸流（`-f adts`，不带 MP4 容器）。
///
/// ⚠️ **关掉 PNS 与强度立体声**（`-aac_pns 0 -aac_is 0`）。这两样产出的东西
/// **规范不要求 bit-exact**：PNS 是伪随机噪声（每个解码器的随机序列都不同）、
/// 强度立体声只保证左右声道的**能量关系**。把它们留在判据里，「±1 LSB」就变成
/// 一条量随机数的测试 —— 红绿都说明不了问题。TNS 留着，它是有确定答案的。
fn ffmpeg_encode(wav: &Path, out: &Path, bitrate: u32) {
    let r = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-i",
            wav.to_str().unwrap(),
            "-c:a",
            "aac",
            "-b:a",
            &bitrate.to_string(),
            "-aac_pns",
            "0",
            "-aac_is",
            "0",
            "-f",
            "adts",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("跑 ffmpeg 编码");
    assert!(
        r.status.success(),
        "ffmpeg 编码失败：{}",
        String::from_utf8_lossy(&r.stderr)
    );
}

/// ffmpeg 解码成 s16le（**每样本 2 字节**，与我们的 int16 量级一致）。
fn ffmpeg_decode(adts: &Path, channels: u8) -> Vec<Vec<f32>> {
    let raw_path = unique_temp("v2v_aac_ref.pcm");
    let _ = std::fs::remove_file(&raw_path);
    let r = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-i",
            adts.to_str().unwrap(),
            "-f",
            "s16le",
            "-ac",
            &channels.to_string(),
            raw_path.to_str().unwrap(),
        ])
        .output()
        .expect("跑 ffmpeg 解码");
    assert!(
        r.status.success(),
        "ffmpeg 解码失败：{}",
        String::from_utf8_lossy(&r.stderr)
    );
    let raw = std::fs::read(&raw_path).expect("读 ffmpeg 的输出");
    let ch = channels as usize;
    let n = raw.len() / 2 / ch;
    let mut planes = vec![Vec::with_capacity(n); ch];
    for i in 0..n {
        for (c, p) in planes.iter_mut().enumerate() {
            let o = (i * ch + c) * 2;
            p.push(f32::from(i16::from_le_bytes([raw[o], raw[o + 1]])));
        }
    }
    planes
}

/// ADTS 头（7 字节，`protection_absent = 1`）。profile 恒 LC、采样率恒 44.1k。
fn adts_header(len: usize, channels: u8) -> [u8; 7] {
    let total = (len + 7) as u32;
    let sfi = 4u8; // 44100
    let mut h = [0u8; 7];
    h[0] = 0xFF;
    h[1] = 0xF1; // MPEG-4, layer 0, protection_absent = 1
    h[2] = ((1u8) << 6) | (sfi << 2) | (channels >> 2); // profile = LC(1)
    h[3] = ((channels & 0x3) << 6) | ((total >> 11) as u8 & 0x03);
    h[4] = ((total >> 3) & 0xFF) as u8;
    h[5] = (((total & 0x7) as u8) << 5) | 0x1F;
    h[6] = 0xFC;
    h
}

fn wrap_adts(frames: &[Vec<u8>], channels: u8) -> Vec<u8> {
    let mut out = Vec::new();
    for f in frames {
        out.extend_from_slice(&adts_header(f.len(), channels));
        out.extend_from_slice(f);
    }
    out
}

/// 拆 ADTS：跳过 7（或 9）字节头，按 `aac_frame_length` 切出裸帧。
fn split_adts(data: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut p = 0usize;
    while p + 7 <= data.len() {
        assert_eq!(data[p], 0xFF, "ADTS 同步字不对（偏移 {p}）");
        assert_eq!(data[p + 1] & 0xF0, 0xF0, "ADTS 同步字不对（偏移 {p}）");
        let hdr = if data[p + 1] & 1 == 1 { 7 } else { 9 };
        let len = ((usize::from(data[p + 3] & 0x03)) << 11)
            | (usize::from(data[p + 4]) << 3)
            | (usize::from(data[p + 5]) >> 5);
        assert!(len > hdr && p + len <= data.len(), "ADTS 帧长越界");
        out.push(data[p + hdr..p + len].to_vec());
        p += len;
    }
    out
}

/// 从 ADTS 的头造 `AudioSpecificConfig`（解码器只认 ASC，不认 ADTS）。
fn asc_from_adts(data: &[u8]) -> Vec<u8> {
    let object_type = (data[2] >> 6) + 1; // profile 2 位，值 = object_type − 1
    let sfi = (data[2] >> 2) & 0x0F;
    let ch = ((data[2] & 0x01) << 2) | (data[3] >> 6);
    vec![
        (object_type << 3) | (sfi >> 1),
        ((sfi & 1) << 7) | (ch << 3),
    ]
}

fn decode_ours(frames: &[Vec<u8>], asc: &[u8]) -> Vec<Vec<f32>> {
    let mut dec = Decoder::new(asc).expect("建解码器");
    let mut out = vec![Vec::new(); dec.channels as usize];
    for f in frames {
        let pcm = dec.decode_frame(f).expect("解一帧");
        assert_eq!(pcm.len(), out.len());
        for (o, p) in out.iter_mut().zip(pcm) {
            o.extend_from_slice(&p);
        }
    }
    out
}

/// 两个解码器的逐样本差，**在 int16 域上按整数 LSB 比**。
///
/// ⚠️ 我们的输出是 `f32`（与编码器输入同一量级），ffmpeg 给的 s16le 是**取整过**的。
/// 直接拿 `f32` 减整数会把「ffmpeg 那半步取整」算成误差 —— 正确做法是两边都落到
/// int16 再比整数 LSB。判据「±1 LSB」说的就是这个：同一个位流、两个解码器，
/// 允许在**量化到 int16 之后**差 1 个最小刻度。
///
/// 返回 `(最大差, 首个越界样本)`。
fn compare(ours: &[Vec<f32>], theirs: &[Vec<f32>]) -> (i32, Option<(usize, usize)>) {
    let mut worst = 0i32;
    let mut first = None;
    for (c, (a, b)) in ours.iter().zip(theirs).enumerate() {
        assert_eq!(
            a.len(),
            b.len(),
            "第 {c} 声道样本数不一致：我们 {} / ffmpeg {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let x = x.round().clamp(-32768.0, 32767.0) as i32;
            let y = y.round() as i32;
            let d = (x - y).abs();
            if d > worst {
                worst = d;
                if d > 1 && first.is_none() {
                    first = Some((c, i));
                }
            }
        }
    }
    (worst, first)
}

/// 用我们自己的编码器把一段 PCM 编成裸帧。
fn encode_ours(src: &[Vec<f32>], channels: u8) -> (Vec<Vec<u8>>, Vec<u8>) {
    let n = src[0].len();
    let cfg = EncoderConfig {
        channels,
        sample_rate: SR,
        ..Default::default()
    };
    let mut enc = Encoder::new(cfg).unwrap();
    let asc = enc.audio_specific_config().to_bytes();
    let mut frames = Vec::new();
    let mut pos = 0usize;
    while pos < n {
        let take = (n - pos).min(1024);
        let planes: Vec<Vec<f32>> = src
            .iter()
            .map(|p| {
                let mut v = p[pos..pos + take].to_vec();
                v.resize(1024, 0.0);
                v
            })
            .collect();
        frames.push(enc.encode_frame(&Frame { planes }).unwrap());
        pos += take;
    }
    (frames, asc)
}

/// 「我们编 → 两个解码器解」，返回最大逐样本差（int16 LSB）。
fn our_bitstream_vs_ffmpeg(src: &[Vec<f32>], channels: u8, tag: &str) -> i32 {
    let (frames, asc) = encode_ours(src, channels);
    let path = unique_temp(&format!("v2v_aac_ours_{tag}_{channels}.aac"));
    std::fs::write(&path, wrap_adts(&frames, channels)).expect("写 ADTS");
    let theirs = ffmpeg_decode(&path, channels);
    let ours = decode_ours(&frames, &asc);
    let (worst, first) = compare(&ours, &theirs);
    eprintln!(
        "{tag} / {channels}ch：{} 帧，最大逐样本差 {worst} LSB",
        frames.len()
    );
    assert!(
        worst <= 1,
        "{tag} / {channels}ch：与 ffmpeg 解出的 PCM 差 {worst} LSB（首个越界样本 {first:?}）"
    );
    worst
}

/// **纯正弦**：只走长窗、不触发 TNS —— 把「基础路径」单独钉住。
///
/// 这条先红的话，问题在反量化 / IMDCT / 重叠，与短窗无关。
#[test]
fn decodes_own_pure_tone_like_ffmpeg() {
    if !has_ffmpeg() {
        eprintln!("SKIP：PATH 上没有 ffmpeg");
        return;
    }
    for &channels in &[1u8, 2u8] {
        let src = pure_tone(SR as usize * 2, channels as usize);
        our_bitstream_vs_ffmpeg(&src, channels, "纯正弦");
    }
}

/// **我们编 → 两个解码器解**：带瞬态（短窗 + 过渡窗），语法对不对由 ffmpeg 判，
/// 波形对不对由逐样本比。
#[test]
fn decodes_our_own_bitstream_like_ffmpeg() {
    if !has_ffmpeg() {
        eprintln!("SKIP：PATH 上没有 ffmpeg");
        return;
    }
    for &channels in &[1u8, 2u8] {
        let src = test_signal(SR as usize * 2, channels as usize);
        our_bitstream_vs_ffmpeg(&src, channels, "含瞬态");
    }
}

/// **ffmpeg 编 → 两个解码器解**：真判据。ffmpeg 的窗序列 / M/S / TNS 决策
/// 与我们不同，覆盖到我们编码器不产的那些组合。
#[test]
fn decodes_ffmpeg_bitstream_within_one_lsb() {
    if !has_ffmpeg() {
        eprintln!("SKIP：PATH 上没有 ffmpeg");
        return;
    }
    for &channels in &[1u8, 2u8] {
        let n = SR as usize * 2;
        let src = test_signal(n, channels as usize);
        let wav = unique_temp(&format!("v2v_aac_src_{channels}.wav"));
        let aac = unique_temp(&format!("v2v_aac_ff_{channels}.aac"));
        write_wav(&wav, &src, SR);
        ffmpeg_encode(&wav, &aac, if channels == 1 { 96_000 } else { 160_000 });

        let data = std::fs::read(&aac).expect("读 ffmpeg 的 AAC");
        let frames = split_adts(&data);
        let asc = asc_from_adts(&data);
        let ours = decode_ours(&frames, &asc);
        let theirs = ffmpeg_decode(&aac, channels);
        let (worst, first) = compare(&ours, &theirs);
        eprintln!(
            "ffmpeg 编 / {channels}ch：{} 帧，最大逐样本差 {worst} LSB",
            frames.len()
        );
        assert!(
            worst <= 1,
            "与 ffmpeg 解出的 PCM 差 {worst} LSB（首个越界样本 {first:?}）"
        );
    }
}
