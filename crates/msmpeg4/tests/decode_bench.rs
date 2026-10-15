//! **解码性能**对表：我们 vs `ffmpeg`，同一把尺子。
//!
//! 口径（两边完全一样）：
//!
//! - 同一份样本、**同样的帧数**（`V2V_BENCH_FRAMES`，默认全解）
//! - 都**只解视频**（ffmpeg 加 `-map 0:v -an`）
//! - 都**单线程**（ffmpeg 加 `-threads 1`；我们的解码器本来就是单线程）
//! - ⚠️ **一处口径不完全对称，且偏向 ffmpeg**：我们的样本是**预载**的（解封装在计时之外），
//!   而 ffmpeg 那一次包含**进程启动 + 解封装**。所以帧数越多这个差越可忽略 ——
//!   量的时候请用**整份样本**，别用 `V2V_BENCH_FRAMES` 截几百帧（那样进程启动就占了大头）。
//! - 取 `V2V_BENCH_REPS` 次里**最快**的一次（默认 3）—— 机器上有别的负载时，
//!   最小值最接近「没有争抢」的那个数
//!
//! 为什么要单列这一张表：**解码速度是这条链路唯一还没量过的一环**。
//! 编码那边有 `bench.sh`（端到端），但解码混在里面看不见。
//!
//! ```bash
//! V2V_BENCH_SAMPLE=/path/x.wmv cargo test -p v2v-msmpeg4 --release \
//!     --test decode_bench -- --ignored --nocapture
//! ```
//!
//! ⚠️ 必须 `--release`：debug 下的数字没有意义（差一个数量级）。

mod common;

use std::process::Command;
use std::time::Instant;

use v2v_msmpeg4::{Decoder, Version};

/// 解我们这一侧：返回 `(解了多少帧, 秒)`。
fn ours(path: &str, limit: usize, reps: usize) -> Option<(usize, f64)> {
    let (samples, w, h) = common::all_video_frames(path)?;
    let version = common::video_fourcc(path)
        .and_then(|(_, _, fc)| Version::from_fourcc(fc))
        .unwrap_or_default();
    let n = samples.len().min(if limit == 0 { usize::MAX } else { limit });

    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        let mut dec = Decoder::with_version(version);
        for s in &samples[..n] {
            // 解不开就当这一帧没解 —— 本表量的是速度，正确性由对账测试管。
            let _ = dec.decode(s, w, h);
        }
        best = best.min(t0.elapsed().as_secs_f64());
    }
    Some((n, best))
}

/// 解 ffmpeg 那一侧：返回最快一次用的秒数。
fn ffmpeg(path: &str, limit: usize, reps: usize) -> Option<f64> {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-v", "error", "-threads", "1", "-i", path, "-map", "0:v", "-an",
        ]);
        if limit > 0 {
            cmd.args(["-frames:v", &limit.to_string()]);
        }
        // `-f null -` 仍然**跑完解码**（只是把输出丢掉），所以时间是可比的。
        cmd.args(["-f", "null", "-"]);
        let t0 = Instant::now();
        let out = cmd.output().ok()?;
        let dt = t0.elapsed().as_secs_f64();
        if !out.status.success() {
            return None;
        }
        best = best.min(dt);
    }
    Some(best)
}

#[test]
#[ignore = "性能对表：本机跑，样本由 V2V_BENCH_SAMPLE 给"]
fn decode_speed_vs_ffmpeg() {
    let paths: Vec<String> = std::env::var("V2V_BENCH_SAMPLE")
        .unwrap_or_default()
        .split(':')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    if paths.is_empty() {
        eprintln!("SKIP: V2V_BENCH_SAMPLE 未设置（用 `:` 分隔多个样本）");
        return;
    }
    let limit: usize = std::env::var("V2V_BENCH_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let reps: usize = std::env::var("V2V_BENCH_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    println!();
    println!("| 样本 | 编码 | 尺寸 | 帧数 | 我们 s | ffmpeg s | 倍数 | 我们 ms/帧 | ffmpeg ms/帧 |");
    println!("|---|---|---|---|---|---|---|---|---|");

    for p in &paths {
        let name = std::path::Path::new(p)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.clone());
        let (w, h, codec) = match common::video_fourcc(p) {
            Some((w, h, fc)) => (w, h, v2v_asf::fourcc_string(fc)),
            None => (0, 0, "?".into()),
        };
        let Some((n, our_s)) = ours(p, limit, reps) else {
            eprintln!("SKIP: 读不出 {name} 的视频样本");
            continue;
        };
        let Some(ff_s) = ffmpeg(p, limit, reps) else {
            eprintln!("SKIP: ffmpeg 解不了 {name}");
            continue;
        };
        // ⚠️ 帧数少时 ffmpeg 那一侧被**进程启动**（~20-30 ms）主导，比值没有意义 ——
        // 标出来，别让人拿它当结论。
        let shaky = if n < 500 { " ⚠️帧少" } else { "" };
        println!(
            "| `{name}` | {codec} | {w}×{h} | {n}{shaky} | {our_s:.3} | {ff_s:.3} | **{:.2}×** | {:.2} | {:.2} |",
            our_s / ff_s,
            our_s * 1000.0 / n as f64,
            ff_s * 1000.0 / n as f64,
        );
    }
    println!();
    eprintln!("（两边都含解封装、都单线程、都只解视频；取 {reps} 次里最快的一次）");
}
