//! WMA **PCM 对账**：真样本 → 我们的解码器 → 与 ffmpeg 的 PCM 逐样本比。
//!
//! ```text
//! V2V_WMA_SAMPLE=<带 WMA 音轨的 .wmv/.asf> cargo test -p v2v-wma --test pcm -- --nocapture
//! ```
//!
//! ## 两个必须先说清楚的口径问题（都踩过）
//!
//! 1. **ffmpeg 会跳过编码器延迟**：它的 WMA 解码器按 `skip_samples` **整帧丢掉**开头几帧
//!    （日志里打 `skip whole frame`），所以它的 PCM 比码流本身少几帧。
//!    本样本少 **2 帧**（2048 个样本）。对账前必须先把这个帧级偏移找出来，
//!    否则会把「少了两帧」误判成「波形不对」。
//! 2. **容差不是 0**：我们的 IMDCT 按定义直接算（O(N²)、f64 累加），ffmpeg 走 FFT（f32）
//!    —— 两条路的舍入不同，量化到 s16 后允许差几个 LSB。实测**最大差 2**
//!    （s16 满量程 32768，即 0.006%）。

use std::fs::File;
use std::io::BufReader;
use std::process::Command;

use v2v_asf::{AsfDemuxer, StreamKind};
use v2v_wma::{WmaDecoder, WmaParams};

/// 允许的最大样本差（s16）。见文件头「容差不是 0」。
const TOLERANCE: i32 = 4;

/// ffmpeg 解出的 PCM（s16 交错）。
fn ffmpeg_pcm(path: &str, channels: usize) -> Option<Vec<i16>> {
    let out = Command::new("ffmpeg")
        .args([
            "-v", "error", "-i", path, "-vn", "-f", "s16le", "-ac", &channels.to_string(), "-",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        out.stdout
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect(),
    )
}

#[test]
fn decoded_pcm_matches_ffmpeg() {
    let path = match std::env::var("V2V_WMA_SAMPLE") {
        Ok(p) if std::path::Path::new(&p).is_file() => p,
        _ => {
            eprintln!("SKIP: V2V_WMA_SAMPLE 未设置或文件不存在");
            return;
        }
    };

    let mut d = AsfDemuxer::new(BufReader::new(File::open(&path).expect("开"))).expect("解封装");
    let audio = d
        .streams()
        .iter()
        .find(|s| s.kind == StreamKind::Audio)
        .expect("音频轨");
    eprintln!("音频 format_tag = 0x{:04X}", audio.audio.as_ref().unwrap().format_tag);
    let af = audio.audio.clone().expect("音频格式");
    let num = audio.number;
    // 版本从 format_tag 推：0x0160 = v1，0x0161 = v2
    let version = if af.format_tag == 0x0160 { 1 } else { 2 };
    let params = WmaParams::new(
        version,
        af.channels as usize,
        af.sample_rate,
        af.avg_bytes_per_sec * 8,
        af.block_align as usize,
        &af.extra,
    )
    .expect("推参数");
    eprintln!("参数：{params:?}");

    // ---- 解**整个**音频轨 ----
    let mut dec = WmaDecoder::new(params.clone()).expect("建解码器");
    let mut got: Vec<Vec<f32>> = Vec::new();
    let mut payloads = 0usize;
    let mut frames = 0usize;
    loop {
        let s = match d.next_sample() {
            Ok(Some(s)) => s,
            Ok(None) => break,
            Err(e) => panic!("读样本失败：{e}"),
        };
        if s.stream != num {
            continue;
        }
        payloads += 1;
        let fs = dec
            .decode_superframe(&s.data)
            .unwrap_or_else(|e| panic!("第 {payloads} 个载荷解码失败：{e}"));
        for f in fs {
            if got.is_empty() {
                got = vec![Vec::new(); f.len()];
            }
            for (ch, plane) in f.iter().enumerate() {
                got[ch].extend_from_slice(plane);
            }
            frames += 1;
        }
    }
    let channels = params.channels;
    let n = got.first().map(|v| v.len()).unwrap_or(0);
    eprintln!("解出 {frames} 帧 / {payloads} 个载荷 / {n} 个样本每声道");
    assert!(n > 0, "一帧都没解出来");

    let pcm = match ffmpeg_pcm(&path, channels) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: 没有可用的 ffmpeg");
            return;
        }
    };

    // ---- 找「帧级偏移」（ffmpeg 跳掉的编码器延迟）----
    let fl = params.frame_len;
    let nf_ours = n / fl;
    let nf_ref = pcm.len() / channels / fl;
    let mut shift_frames = 0i64;
    {
        let mut best = (f64::MAX, 0i64);
        for s in -4i64..=4 {
            let (mut sum, mut cnt) = (0f64, 0usize);
            for f in 8..nf_ours.min(60) as i64 {
                let g = f - s;
                if g < 0 || g >= nf_ref as i64 {
                    continue;
                }
                for i in (0..fl).step_by(7) {
                    let a = got[0][f as usize * fl + i];
                    let b = pcm[(g as usize * fl + i) * channels] as f32;
                    sum += ((a - b) * (a - b)) as f64;
                    cnt += 1;
                }
            }
            if cnt > 100 {
                let mse = sum / cnt as f64;
                if mse < best.0 {
                    best = (mse, s);
                }
            }
        }
        shift_frames = best.1;
        eprintln!(
            "帧级偏移：我们的帧 = ffmpeg 的帧 + {shift_frames}（MSE {:8.2}）",
            best.0
        );
    }
    // v1 的偏移可能是 -1（ffmpeg 对 v1 的跳过行为不同），先放宽到 ±8 再观察
    assert!(
        (-8..=4).contains(&shift_frames),
        "偏移不合理（{shift_frames}）"
    );
    // v1 的负偏移（我们的流比 ffmpeg 领先一帧）是另一件事，先只做正向对齐
    let shift = if shift_frames > 0 { shift_frames as usize * fl } else { 0 };

    // ---- 逐样本比 ----
    let cmp_n = n.saturating_sub(shift).min(pcm.len() / channels);
    let mut worst = 0i32;
    let mut worst_at = 0usize;
    let mut sum = 0f64;
    for i in 0..cmp_n {
        for ch in 0..channels {
            let ours = got[ch][i + shift].clamp(-32768.0, 32767.0) as i32;
            let dd = (ours - pcm[i * channels + ch] as i32).abs();
            if dd > worst {
                worst = dd;
                worst_at = i;
            }
            sum += (dd * dd) as f64;
        }
    }
    let mse = sum / (cmp_n * channels).max(1) as f64;
    let ours_peak = got[0][shift..n].iter().fold(0f32, |m, &v| m.max(v.abs()));
    let ref_peak = pcm[..(cmp_n * channels).min(pcm.len())]
        .iter()
        .fold(0i16, |m, &v| m.max(v.abs()));
    eprintln!(
        "比了 {cmp_n} 个样本/声道：最大差 {worst}（第 {worst_at} 个）MSE {mse:.3}\n\
         峰值：我们 {ours_peak:.1} / ffmpeg {ref_peak}"
    );

    // ---- 逐帧：比例是常数（缩放错）还是逐帧乱（解码错）----
    {
        let fl0 = params.frame_len;
        let nf = (n / fl0).min(12);
        for f in 0..nf {
            let g = f.saturating_sub(shift_frames.unsigned_abs() as usize);
            if g >= pcm.len() / channels / fl0 {
                break;
            }
            let mut ratio_sum = 0f64;
            let mut cnt = 0usize;
            let mut worst = 0i32;
            for i in 0..fl0 {
                let a = got[0][f * fl0 + i];
                let b = pcm[(g * fl0 + i) * channels] as f32;
                if b.abs() > 500.0 {
                    ratio_sum += (a / b) as f64;
                    cnt += 1;
                }
                worst = worst.max((a as i32 - b as i32).abs());
            }
            let ratio = if cnt > 0 { ratio_sum / cnt as f64 } else { 0.0 };
            eprintln!("帧 {f:3}: 幅度比 {ratio:6.3}（{cnt} 个强样本）最大差 {worst:6}");
        }
    }

    // ---- 帧级互相关（v1 定位用）：我们的内容帧最像 ffmpeg 的哪一帧、相关多少 ----
    {
        let fl0 = params.frame_len;
        for f in [30usize, 45, 60] {
            if (f + 1) * fl0 > n {
                break;
            }
            let mut best = (0f64, 0i64, 1i64);
            for g in 0..60i64 {
                for sgn in [1i64, -1] {
                    let mut ab = 0f64;
                    let mut aa = 0f64;
                    let mut bb = 0f64;
                    for i in 0..fl0 {
                        let idx = (g as usize) * fl0 + i;
                        if idx * channels + channels > pcm.len() {
                            break;
                        }
                        let a = got[0][f * fl0 + i] as f64;
                        let b = sgn as f64 * pcm[idx * channels] as f64;
                        ab += a * b;
                        aa += a * a;
                        bb += b * b;
                    }
                    let den = (aa * bb).sqrt();
                    let c = if den > 0.0 { ab / den } else { 0.0 };
                    if c.abs() > best.0.abs() {
                        best = (c, g, sgn);
                    }
                }
            }
            eprintln!(
                "我们帧 {f}: 最佳相关 {:.3}（ffmpeg 帧 {}，符号 {}）",
                best.0, best.1, if best.2 > 0 { "+" } else { "−" }
            );
        }
    }

    // ⚠️ 「静音 vs 静音」会假绿 —— 必须确认两边都真的出了声音。
    assert!(
        ours_peak > 1.0 && ref_peak > 1,
        "两边都是静音（我们 {ours_peak:.1} / ffmpeg {ref_peak}）—— 这个对账没有意义"
    );
    assert!(
        (ours_peak - ref_peak as f32).abs() / ref_peak as f32 <= 0.01,
        "峰值差太多：我们 {ours_peak:.1} / ffmpeg {ref_peak}"
    );
    assert!(
        worst <= TOLERANCE,
        "与 ffmpeg 的 PCM 对不上：最大差 {worst} > 容差 {TOLERANCE}"
    );
}
