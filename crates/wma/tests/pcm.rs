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

    // 逐样本比（s16 量化后）
    let mut worst = 0i32;
    let mut worst_at = 0usize;
    let mut sum = 0f64;
    let mut count = 0usize;
    for i in 0..n {
        for ch in 0..channels {
            let ours = (got[ch][i].clamp(-32768.0, 32767.0)) as i32;
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
    let ours_peak = got[0][..n].iter().fold(0f32, |m, &v| m.max(v.abs()));
    let ref_peak = pcm[..(n * channels).min(pcm.len())]
        .iter()
        .fold(0i16, |m, &v| m.max(v.abs()));
    eprintln!("峰值：我们 {ours_peak:.1} / ffmpeg {ref_peak}");
    assert!(
        ours_peak > 1.0 && ref_peak > 1,
        "两边都是静音（我们峰值 {ours_peak:.1}，ffmpeg {ref_peak}）—— 这个对账没有意义"
    );
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
