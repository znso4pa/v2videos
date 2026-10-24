//! 「Rust + FFI」这条路径的解码基准 —— 与 `v2v-mpeg12` 的 `compare --bench`
//! **同一套口径**（同样的 demux、同样的解码序 / 显示序维护、同样的轮数），
//! 这样两边数字可以直接对。
//!
//! ```text
//! cargo run --release -p v2v-mpeg12-simd --example bench -- <sample.mpg> [--rounds=N] [--threads=N]
//! cargo run --release -p v2v-mpeg12-simd --features unsafe-kernels --example bench -- <sample.mpg>
//! ```
//!
//! 同时打印 `kernels::available()`：它报的是**实际会用到**的内核，
//! 关掉 `unsafe-kernels` 时应当是 0。

use v2v_mpeg12::{PictureType, SequenceHeader, split_pictures, split_units};
use v2v_mpegps::PsDemuxer;

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(sample) = args.next() else {
        eprintln!(
            "用法: bench <sample.mpg> [--rounds=N] [--threads=N]"
        );
        return std::process::ExitCode::FAILURE;
    };
    let mut rounds = 20usize;
    let mut threads = 1usize;
    for arg in args {
        if let Some(v) = arg.strip_prefix("--rounds=") {
            rounds = v.parse().expect("--rounds=N");
        } else if let Some(v) = arg.strip_prefix("--threads=") {
            threads = v.parse().expect("--threads=N");
        }
    }

    let bytes = std::fs::read(&sample).expect("read the program stream");
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
            Some(v2v_mpeg12_simd::MtDecoder::new(threads))
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
                None => v2v_mpeg12_simd::decode_frame(
                    &sequence,
                    header.clone(),
                    data,
                    forward,
                    last_ip.as_ref(),
                )
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
        "[bench] {sample} {w}x{h} {frames} 帧 / {rounds} 遍：**{ms:.2} ms/帧**（threads={threads}）"
    );
    eprintln!(
        "[kernels] available=0b{:b}（KERNEL_AVG={}）",
        v2v_mpeg12_simd::kernels::available(),
        v2v_mpeg12_simd::kernels::KERNEL_AVG
    );
    std::process::ExitCode::SUCCESS
}
