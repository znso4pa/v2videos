//! WMA **PCM 对账**：真样本 → 我们的解码器 → 与 ffmpeg 的 PCM 比。
//!
//! ```text
//! V2V_WMA_SAMPLE=<带 WMA 音轨的 .wmv/.asf> cargo test -p v2v-wma --test pcm -- --nocapture
//! ```
//!
//! 这一段是「音频解码跑通」的判据：**逐样本**对，容差 ±1 LSB（s16 量化）。

use std::fs::File;
use std::io::BufReader;
use std::process::Command;

use v2v_asf::{AsfDemuxer, StreamKind};
use v2v_wma::{BitReader, SuperframeState, WmaDecoder, WmaParams};

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

/// ⚠️ **还没跑通**：解码链已完整实现（帧头→LSP 指数→系数→IMDCT→加窗），
/// 但帧头位序还有错 —— 第 0 帧就 `run-level 越界`，解出来是静音。
/// 所以先 `#[ignore]`，等修好再让它进常规测试。
///
/// 另外注意：这个样本**开头 4096 个样本 ffmpeg 解出来也是静音**，
/// 所以对账必须瞄准「有声音的那一段」，否则「静音 vs 静音」会假绿
/// （第一版就踩了，见下面的峰值断言）。
#[test]
#[ignore = "WMA 帧解码还没跑通（帧头位序待修），见 TODO"]
fn first_frames_match_ffmpeg_pcm() {
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
    let af = audio.audio.clone().expect("音频格式");
    let num = audio.number;
    let params = WmaParams::new(
        2,
        af.channels as usize,
        af.sample_rate,
        af.avg_bytes_per_sec * 8,
        af.block_align as usize,
        &af.extra,
    )
    .expect("参数");
    eprintln!("参数：{params:?}");

    let mut dec = WmaDecoder::new(params.clone()).expect("建解码器");

    // 解前几帧（走完整的 superframe → 帧流程，含比特池搬运）
    let mut got: Vec<Vec<f32>> = Vec::new(); // 每个声道一串
    let mut payloads = 0usize;
    let mut decoded_frames = 0usize;
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
        let frames = match dec.decode_superframe(&s.data) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("第 {payloads} 个载荷解码失败：{e}");
                break;
            }
        };
        for f in frames {
            if got.is_empty() {
                got = vec![Vec::new(); f.len()];
            }
            for (ch, plane) in f.iter().enumerate() {
                got[ch].extend_from_slice(plane);
            }
            decoded_frames += 1;
        }
        if decoded_frames >= 60 {
            break;
        }
    }
    eprintln!("解出 {decoded_frames} 帧 / {payloads} 个载荷");

    let channels = params.channels;
    let n = got.first().map(|v| v.len()).unwrap_or(0);
    eprintln!("我们解出 {n} 个样本/声道（{payloads} 个载荷）");
    if n == 0 {
        panic!("一帧都没解出来");
    }

    // ffmpeg 的 PCM（交错 s16）
    let pcm = match ffmpeg_pcm(&path, channels) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: 没有可用的 ffmpeg");
            return;
        }
    };
    eprintln!("ffmpeg 解出 {} 个样本（交错）", pcm.len());

    // ---- 先定「帧级偏移」----
    //
    // ⚠️ ffmpeg 的 WMA 解码器会**整帧跳过**编码器延迟（`skip_samples`，
    // 日志里会打 `skip whole frame`），所以它的 PCM 比码流本身的帧少几帧。
    // 实测这个样本少 3 帧（3072 个样本）—— 对账前必须先把这个偏移找出来，
    // 否则会把「少了几帧」误判成「波形不对」。
    let fl0 = params.frame_len;
    let nf_ours = n / fl0;
    let nf_ref = pcm.len() / channels / fl0;
    let mut shift_frames = 0i64;
    {
        let mut best = (f64::MAX, 0i64);
        for s in -8i64..=8 {
            let mut sum = 0f64;
            let mut cnt = 0usize;
            for f in 8..nf_ours.min(40) as i64 {
                let g = f - s;
                if g < 0 || g >= nf_ref as i64 {
                    continue;
                }
                for i in (0..fl0).step_by(7) {
                    let a = got[0][(f as usize) * fl0 + i];
                    let b = pcm[((g as usize) * fl0 + i) * channels] as f32;
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
        eprintln!("帧级偏移：我们的帧 = ffmpeg 的帧 + {shift_frames}（MSE {:8.0}）", best.0);
    }
    let shift = (shift_frames * fl0 as i64).max(0) as usize;

    // 逐样本比（s16 量化后，按上面的偏移对齐）
    let mut worst = 0i32;
    let mut worst_at = 0usize;
    let mut sum = 0f64;
    let mut count = 0usize;
    let cmp_n = n.saturating_sub(shift);
    for i in 0..cmp_n {
        for ch in 0..channels {
            let ours = (got[ch][i + shift].clamp(-32768.0, 32767.0)) as i32;
            let idx = i * channels + ch;
            if idx >= pcm.len() {
                break;
            }
            let d = (ours - pcm[idx] as i32).abs();
            if d > worst {
                worst = d;
                worst_at = idx;
            }
            sum += (d * d) as f64;
            count += 1;
        }
    }
    let mse = sum / count.max(1) as f64;
    eprintln!(
        "前 {n} 个样本/声道：最大差 {worst}（第 {worst_at} 个交错样本）MSE {mse:.1}\n\
         我们前 8 个：{:?}\nffmpeg 前 8 个（左声道）：{:?}",
        &got[0][..8.min(n)],
        (0..8.min(n)).map(|i| pcm.get(i * channels).copied().unwrap_or(0)).collect::<Vec<_>>()
    );

    // ⚠️ 先确认**两边都不是静音** —— 否则「静音 vs 静音」会假绿。
    // （第一版就踩了：我们解出 4096 个零、ffmpeg 前 4096 个也是零，测试居然过了。）
    let ours_peak = got[0][shift..n].iter().fold(0f32, |m, &v| m.max(v.abs()));
    let ref_peak = pcm[..(cmp_n * channels).min(pcm.len())]
        .iter()
        .fold(0i16, |m, &v| m.max(v.abs()));
    eprintln!("峰值：我们 {ours_peak:.1} / ffmpeg {ref_peak}");
    assert!(
        ours_peak > 1.0 && ref_peak > 1,
        "两边都是静音（我们峰值 {ours_peak:.1}，ffmpeg {ref_peak}）—— 这个对账没有意义"
    );
    // ---- 帧映射：我们的每一帧，最像 ffmpeg 的哪一帧 ----
    {
        let fl = params.frame_len;
        let nf_ours = n / fl;
        let nf_ref = pcm.len() / channels / fl;
        for f in 10..nf_ours.min(24) {
            let mut best = (f64::MAX, 0usize);
            for g in 0..nf_ref.min(40) {
                let mut sum = 0f64;
                for i in 0..256 {
                    let a = got[0][f * fl + i];
                    let b = pcm[(g * fl + i) * channels] as f32;
                    sum += ((a - b) * (a - b)) as f64;
                }
                let mse = sum / 256.0;
                if mse < best.0 {
                    best = (mse, g);
                }
            }
            eprintln!("我们帧 {f:2} ↔ ffmpeg 帧 {:2}（MSE {:8.0}）", best.1, best.0);
        }
    }

    // ---- 逐帧看：错的是所有有声音的帧，还是只有某些帧 ----
    let fl = params.frame_len;
    for f in 0..(n / fl).min(20) {
        let mut worst = 0i32;
        for i in 0..fl {
            let idx = (f * fl + i) * channels;
            if idx >= pcm.len() {
                break;
            }
            let d = (got[0][f * fl + i] as i32 - pcm[idx] as i32).abs();
            worst = worst.max(d);
        }
        let peak = (0..fl)
            .map(|i| got[0][f * fl + i].abs())
            .fold(0f32, f32::max);
        eprintln!("帧 {f:2}: 最大差 {worst:6} 我们峰值 {peak:8.1}");
    }

    // ---- 对齐诊断：波形是不是整体错位（错位的话峰值会对、逐样本差会很大）----
    // 找一个**有声音**的窗口（静音段对齐没有意义：零对零总是相等）
    let mut anchor = 0usize;
    for i in 0..n {
        let idx = i * channels;
        if idx + channels < pcm.len() && pcm[idx].abs() > 500 {
            anchor = i;
            break;
        }
    }
    eprintln!("对齐窗口从第 {anchor} 个样本开始");
    let win = 4096usize.min(n - anchor);
    let mut best = (i64::MAX, 0i64);
    for shift in -(params.frame_len as i64 * 4)..=(params.frame_len as i64 * 4) {
        let mut sum = 0f64;
        let mut cnt = 0usize;
        for k in 0..win {
            let i = anchor + k;
            let j = i as i64 + shift;
            if j < 0 {
                continue;
            }
            let idx = j as usize * channels;
            if idx >= pcm.len() {
                break;
            }
            let d = got[0][i] - pcm[idx] as f32;
            sum += (d * d) as f64;
            cnt += 1;
        }
        if cnt > win / 2 {
            let mse = sum / cnt as f64;
            if mse < best.0 as f64 {
                best = (mse as i64, shift);
            }
        }
    }
    eprintln!("最佳对齐：shift={} 个样本（MSE {})", best.1, best.0);
    eprintln!("即：我们的样本 i 对应 ffmpeg 的 i+{}", best.1);

    assert!(worst <= 1, "与 ffmpeg 的 PCM 对不上：最大差 {worst}");
}
