//! 拿一条 MPEG-1/2 program stream 和一份参考 YUV 逐帧比 —— **按显示序**。
//!
//! ```text
//! cargo run --release -p v2v-mpeg12 --example compare -- <sample.mpg> <ref.yuv>
//! ```
//!
//! 参考件：`ffmpeg -i sample.mpg -f rawvideo -pix_fmt yuv420p ref.yuv`。
//!
//! 报三样东西：**逐帧最差像素差**、**聚合 PSNR**、**逐帧 PSNR 的中位数**。
//! 为什么三个都要：只看聚合 PSNR 会被少数坏帧拖平，只看最差又会把 ±1 的
//! IDCT 舍入当成 bug。口径与 `tests/frames.rs` 一致（容差 2 = 舍入，>2 = 真错）。

use std::path::PathBuf;

use v2v_mpeg12::{
    PictureType, SequenceHeader, decode::decode_frame, split_pictures, split_units,
};
use v2v_mpegps::PsDemuxer;

/// 纯解码基准：解 `rounds` 遍，报 ms/帧（不读参考件、不逐像素比）。
fn run_bench(sample: &str, rounds: usize, threads: usize) -> std::process::ExitCode {
    let bytes = std::fs::read(sample).expect("read the program stream");
    let mut demux = PsDemuxer::new(bytes);
    let mut es = Vec::new();
    while let Some(packet) = demux.next_packet() {
        if packet.is_video() {
            es.extend_from_slice(&demux.data()[packet.range.clone()]);
        }
    }
    let units = split_units(&es);
    let mut sequence = units
        .iter()
        .find(|(code, _)| *code == 0xB3)
        .and_then(|(_, payload)| SequenceHeader::parse(payload))
        .expect("a sequence header");
    sequence.mpeg2 = v2v_mpeg12::has_sequence_extension(&es);
    if sequence.mpeg2 {
        if let Some(ext) = units
            .iter()
            .find(|(code, _)| *code == 0xB5)
            .and_then(|(_, payload)| v2v_mpeg12::SequenceExtension::parse(payload))
        {
            sequence.apply_sequence_extension(&ext);
        }
    }
    let pictures: Vec<_> = split_pictures(&es)
        .into_iter()
        .filter(|(h, _)| !matches!(h.picture_type, PictureType::D))
        .collect();
    let t0 = std::time::Instant::now();
    let mut frames = 0usize;
    for _ in 0..rounds {
        let mut last_ip: Option<v2v_mpeg12::decode::Frame> = None;
        let mut prev_ip: Option<v2v_mpeg12::decode::Frame> = None;
        let mut mt = if threads > 1 {
            Some(v2v_mpeg12::decode::MtDecoder::new(threads))
        } else {
            None
        };
        for (header, data) in &pictures {
            let kind = header.picture_type;
            let forward = if matches!(kind, PictureType::B) {
                prev_ip.as_ref()
            } else {
                last_ip.as_ref()
            };
            let f = match &mut mt {
                Some(m) => m
                    .decode(&sequence, header.clone(), data, forward, last_ip.as_ref())
                    .expect("解码"),
                None => decode_frame(&sequence, header.clone(), data, forward, last_ip.as_ref())
                    .expect("解码"),
            };
            if matches!(kind, PictureType::I | PictureType::P) {
                prev_ip = last_ip.take();
                last_ip = Some(f);
            }
            frames += 1;
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    let (w, h) = (sequence.width, sequence.height);
    eprintln!(
        "[bench] {} {w}x{h} {frames} 帧 / {} 遍：**{ms:.2} ms/帧**（threads={threads}）",
        sample, rounds
    );
    v2v_mpeg12::decode::prof::report();
    std::process::ExitCode::SUCCESS
}

fn main() {
    // `--prof` 是**开关**（不带值），先摘掉再解析位置参数 —— 否则它会顶掉 sample 的位置。
    let raw: Vec<String> = std::env::args().collect();
    v2v_mpeg12::decode::prof::enable(raw.iter().any(|a| a == "--prof"));
    let argv: Vec<String> = raw.iter().filter(|a| *a != "--prof").cloned().collect();
    // `--bench=N` 要在读参考件**之前**判掉：基准模式不读那 135 MB。
    if let Some(n) = argv.iter().find_map(|a| a.strip_prefix("--bench=")) {
        let rounds: usize = n.parse().unwrap_or(1);
        let threads: usize = argv
            .iter()
            .find_map(|a| a.strip_prefix("--threads="))
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let sample = argv.get(1).cloned().unwrap_or_default();
        run_bench(&sample, rounds, threads);
        return;
    }
    let mut it = argv.iter().skip(1).cloned();
    let Some(sample) = it.next().map(PathBuf::from) else {
        eprintln!("用法: compare <sample.mpg> <ref.yuv> [--dump <out.yuv>] [--threads=N] [--prof]");
        std::process::exit(2);
    };
    let Some(reference) = it.next().map(PathBuf::from) else {
        eprintln!("用法: compare <sample.mpg> <ref.yuv> [--dump <out.yuv>] [--threads=N] [--prof]");
        std::process::exit(2);
    };
    // `--dump`：把**我们解出来的**帧按显示序写成 YUV，方便拿去做宏块级的排查。
    let dump = match (it.next().as_deref(), it.next()) {
        (Some("--dump"), Some(p)) => Some(PathBuf::from(p)),
        _ => None,
    };
    let mut threads = 1usize;
    let mut rest: Vec<String> = Vec::new();
    for a in it {
        if let Some(n) = a.strip_prefix("--threads=") {
            threads = n.parse().unwrap_or(1);
        } else {
            rest.push(a);
        }
    }

    let bytes = std::fs::read(&sample).expect("读样本");
    let mut demux = PsDemuxer::new(bytes);
    let mut es = Vec::new();
    while let Some(packet) = demux.next_packet() {
        if packet.is_video() {
            es.extend_from_slice(&demux.data()[packet.range.clone()]);
        }
    }

    let units = split_units(&es);
    let mut sequence = units
        .iter()
        .find(|(code, _)| *code == 0xB3)
        .and_then(|(_, payload)| SequenceHeader::parse(payload))
        .expect("sequence header");
    sequence.mpeg2 = v2v_mpeg12::has_sequence_extension(&es);
    if sequence.mpeg2 {
        if let Some(ext) = units
            .iter()
            .find(|(code, _)| *code == 0xB5)
            .and_then(|(_, payload)| v2v_mpeg12::SequenceExtension::parse(payload))
        {
            sequence.apply_sequence_extension(&ext);
        }
    }

    let frame_size = match sequence.chroma_format {
        2 => (sequence.width * sequence.height * 2) as usize,
        3 => (sequence.width * sequence.height * 3) as usize,
        _ => (sequence.width * sequence.height * 3 / 2) as usize,
    };
    let want = std::fs::read(&reference).expect("读参考帧");
    let available = want.len() / frame_size;

    // 解码序 ≠ 显示序：I/P 先解、B 后解。按 (第几个 I 帧, temporal_reference) 排。
    let mut decoded: Vec<(u32, u32, PictureType, v2v_mpeg12::decode::Frame)> = Vec::new();
    let mut gop = 0u32;
    let mut last_ip: Option<v2v_mpeg12::decode::Frame> = None;
    let mut prev_ip: Option<v2v_mpeg12::decode::Frame> = None;
    let mut mt = if threads > 1 {
        Some(v2v_mpeg12::decode::MtDecoder::new(threads))
    } else {
        None
    };
    for (header, data) in split_pictures(&es) {
        let kind = header.picture_type;
        let tr = header.temporal_reference;
        if matches!(kind, PictureType::D) {
            continue;
        }
        if matches!(kind, PictureType::I) && !decoded.is_empty() {
            gop += 1;
        }
        let forward = if matches!(kind, PictureType::B) {
            prev_ip.as_ref()
        } else {
            last_ip.as_ref()
        };
        let frame = match &mut mt {
            Some(m) => m
                .decode(&sequence, header, data, forward, last_ip.as_ref())
                .unwrap_or_else(|e| panic!("解码失败：{e:?}")),
            None => decode_frame(&sequence, header, data, forward, last_ip.as_ref())
                .unwrap_or_else(|e| panic!("解码失败：{e:?}")),
        };
        if matches!(kind, PictureType::I | PictureType::P) {
            prev_ip = last_ip.take();
            last_ip = Some(frame.clone());
        }
        decoded.push((gop, tr, kind, frame));
    }
    decoded.sort_by_key(|(g, tr, _, _)| (*g, *tr));

    let mut dumped: Vec<u8> = Vec::new();
    let mut worst_all = 0i32;
    let mut sum_sq = 0f64;
    let mut pixels = 0usize;
    let mut per_frame = Vec::new();
    let mut over = 0usize;
    for (i, (_, tr, kind, frame)) in decoded.iter().enumerate() {
        if i >= available {
            break;
        }
        let mut ours = Vec::with_capacity(frame_size);
        ours.extend_from_slice(&frame.y);
        ours.extend_from_slice(&frame.u);
        ours.extend_from_slice(&frame.v);
        if dump.is_some() {
            dumped.extend_from_slice(&ours);
        }
        let theirs = &want[i * frame_size..(i + 1) * frame_size];
        let mut worst = 0i32;
        let mut se = 0f64;
        for (a, b) in ours.iter().zip(theirs) {
            let d = *a as i32 - *b as i32;
            let ad = d.abs();
            se += (d * d) as f64;
            worst = worst.max(ad);
        }
        sum_sq += se;
        pixels += ours.len();
        worst_all = worst_all.max(worst);
        if worst > 2 {
            over += 1;
        }
        let psnr = if se == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / (se / ours.len() as f64)).log10()
        };
        per_frame.push((i, *tr, kind, worst, psnr));
        if worst > 2 {
            // 坏在哪个宏块：亮度 / 色度分开统计（差 >2 才算，±1 是 IDCT 舍入）。
            let w = sequence.width as usize;
            let h = sequence.height as usize;
            let (ys, cs) = (w * h, (w / 2) * (h / 2));
            let mut ymb = std::collections::BTreeMap::new();
            let mut cmb = std::collections::BTreeMap::new();
            for (j, (a, b)) in ours.iter().zip(theirs).enumerate() {
                if (*a as i32 - *b as i32).abs() > 2 {
                    if j < ys {
                        let (x, y) = (j % w, j / w);
                        *ymb.entry((x / 16, y / 16)).or_insert(0usize) += 1;
                    } else {
                        let k = (j - ys) % cs;
                        let (x, y) = (k % (w / 2), k / (w / 2));
                        *cmb.entry((x / 8, y / 8)).or_insert(0usize) += 1;
                    }
                }
            }
            eprintln!(
                "  帧 {i} (tr={tr}, {kind:?}): 最差 {worst}, PSNR {psnr:.1} | Y {ymb:?} 色度 {cmb:?}"
            );
        }
    }

    if let Some(path) = &dump {
        std::fs::write(path, &dumped).expect("写 dump");
    }
    v2v_mpeg12::decode::prof::report();
    let mut sorted: Vec<f64> = per_frame.iter().map(|(_, _, _, _, p)| *p).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    let agg = 10.0 * (255.0 * 255.0 / (sum_sq / pixels as f64)).log10();
    println!(
        "{}: {} 帧（参考 {}）| 最差 {} | 聚合 PSNR {:.2} dB | 逐帧中位 {:.2} dB | 最差>2 的帧 {}",
        sample.display(),
        per_frame.len(),
        available,
        worst_all,
        agg,
        median,
        over
    );
}
