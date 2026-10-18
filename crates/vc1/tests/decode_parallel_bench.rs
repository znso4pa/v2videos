//! **GOP 并行解码**的伸缩曲线 —— 回答「加解码 worker 到底有没有用」。
//!
//! ```bash
//! V2V_BENCH_SAMPLE=/path/x.wmv cargo test -p v2v-vc1 --release \
//!     --test decode_parallel_bench -- --ignored --nocapture
//! ```
//!
//! ## 为什么单独量这个
//!
//! 端到端在 8 线程下卡在 `≈ 纯解码（单线程）`，而把解码池从 1 加到 3 个 worker
//! **墙钟没有变化**。有两种可能，结论完全相反：
//!
//! 1. **解码本身不扩展**（派活/收集串行、或内存带宽打满）→ 该修流水线，不该动算法；
//! 2. **解码扩展了，但整体没有真正重叠**（解码与编码是「相加」而不是「取大」）
//!    → 该修的是两段之间的重叠。
//!
//! 只看端到端分不出这两者，所以这里把**解码单独拎出来**，用与 `v2v convert`
//! 完全相同的做法跑（每个 GOP 一条新解码器），量 1/2/4/6/8 个 worker 的墙钟。
//!
//! ⚠️ 判据是**伸缩比**（相对 1 worker），不是绝对秒数 —— 后者随机器状态漂。
//! 同一会话内背靠背跑出来的比值才有意义。

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use v2v_vc1::{Decoder, SeqHeader};

/// 一个 GOP 的帧数 —— 与 `v2v convert -g 12` 的默认口径一致。
const GOP: usize = 12;

/// 用 `workers` 个线程按 GOP 切分解码，返回墙钟秒数。
///
/// 做法与 `crates/cli/src/lib.rs` 的解码池一致：**每个 GOP 一条全新解码器**
/// （前提是「从任意 I 帧起解 == 整条链解到那里」，由 `gop_start.rs` 钉着）。
fn decode_parallel(
    frames: &Arc<Vec<Vec<u8>>>,
    seq: SeqHeader,
    (w, h): (u32, u32),
    workers: usize,
) -> f64 {
    let cursor = Arc::new(AtomicUsize::new(0));
    let n = frames.len();
    let t0 = Instant::now();

    std::thread::scope(|s| {
        for _ in 0..workers {
            let frames = Arc::clone(frames);
            let cursor = Arc::clone(&cursor);
            s.spawn(move || {
                loop {
                    let start = cursor.fetch_add(GOP, Ordering::Relaxed);
                    if start >= n {
                        break;
                    }
                    let end = (start + GOP).min(n);
                    let Ok(mut d) = Decoder::new(seq, w, h) else {
                        return;
                    };
                    for f in &frames[start..end] {
                        let _ = d.decode_frame(f);
                    }
                }
            });
        }
    });

    t0.elapsed().as_secs_f64()
}

#[test]
#[ignore = "性能基准：本机跑，样本由 V2V_BENCH_SAMPLE 给"]
fn decode_scaling_by_workers() {
    let path = std::env::var("V2V_BENCH_SAMPLE").unwrap_or_default();
    if path.is_empty() {
        eprintln!("SKIP: V2V_BENCH_SAMPLE 未设置");
        return;
    }
    let Some(track) = common::video_track(&path) else {
        eprintln!("SKIP: 读不出视频轨");
        return;
    };
    if track.fourcc != common::FOURCC_WMV3 {
        eprintln!("SKIP: 不是 WMV3");
        return;
    }
    let Some(seq) = SeqHeader::parse(&track.extra).ok() else {
        eprintln!("SKIP: 序列头解析失败");
        return;
    };
    let Some(all) = common::all_video_frames(&path, track.number) else {
        eprintln!("SKIP: 读不出视频样本");
        return;
    };
    let limit: usize = std::env::var("V2V_BENCH_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let n = if limit == 0 { all.len() } else { all.len().min(limit) };
    let frames = Arc::new(all[..n].to_vec());
    let (w, h) = (track.encoded_width, track.encoded_height);

    println!();
    println!("| worker | 墙钟 s | ms/帧 | 伸缩比 | 并行效率 |");
    println!("|---|---|---|---|---|");
    let mut base = 0.0;
    for workers in [1usize, 2, 4, 6, 8] {
        let t = decode_parallel(&frames, seq, (w, h), workers);
        if workers == 1 {
            base = t;
        }
        println!(
            "| {workers} | {t:.3} | {:.2} | {:.2}× | {:.0}% |",
            t * 1000.0 / n as f64,
            base / t,
            100.0 * (base / t) / workers as f64
        );
    }
    println!();
    println!("（{n} 帧 / {w}×{h} / GOP={GOP}；伸缩比相对 1 worker，同一会话内背靠背）");
}
