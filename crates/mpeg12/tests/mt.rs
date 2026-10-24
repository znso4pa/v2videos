//! 多线程解码的**等价性**对账：`MtDecoder`（以及一次性的 `decode_frame_mt`）
//! 必须和单线程逐字节相同。
//!
//! ```bash
//! V2V_MPEG12_SAMPLE=<clip>.mpg V2V_MPEG12_THREADS=6 \
//!   cargo test -p v2v-mpeg12 --test mt -- --ignored --nocapture
//! ```
//!
//! 为什么盯「逐字节相同」而不是 PSNR：线程切分只改**谁算哪几行**，
//! 不该改动任何一行的值。只要有一位不同，就是切分、预填或合并错了 ——
//! 而 PSNR 会把一个错位的块平均掉。
//!
//! 两条路径都测：
//! 1. `MtDecoder`（worker 缓冲**跨帧复用**，热路径）；
//! 2. `decode_frame_mt`（每帧新建，一次性方便函数）。

use v2v_mpeg12::{
    PictureType, SequenceHeader,
    decode::{MtDecoder, decode_frame, decode_frame_mt},
    split_pictures, split_units,
};
use v2v_mpegps::PsDemuxer;

#[test]
#[ignore = "needs V2V_MPEG12_SAMPLE (a .mpg)"]
fn mt_matches_single_threaded() {
    let sample = std::env::var("V2V_MPEG12_SAMPLE").expect("set V2V_MPEG12_SAMPLE");
    let threads: usize = std::env::var("V2V_MPEG12_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);

    let mut demux = PsDemuxer::new(std::fs::read(&sample).expect("read the program stream"));
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

    let mut mt = MtDecoder::new(threads);
    let mut last_ip: Option<v2v_mpeg12::decode::Frame> = None;
    let mut prev_ip: Option<v2v_mpeg12::decode::Frame> = None;
    let mut n = 0usize;
    for (header, data) in split_pictures(&es) {
        let kind = header.picture_type;
        if matches!(kind, PictureType::D) {
            continue;
        }
        let forward = if matches!(kind, PictureType::B) {
            prev_ip.as_ref()
        } else {
            last_ip.as_ref()
        };
        let backward = last_ip.as_ref();
        let want =
            decode_frame(&sequence, header.clone(), data, forward, backward).expect("单线程解码");
        // ① 复用缓冲的 MtDecoder（热路径）。
        let got = mt
            .decode(&sequence, header.clone(), data, forward, backward)
            .expect("MtDecoder 解码");
        // ② 一次性的 decode_frame_mt。
        let got2 = decode_frame_mt(&sequence, header.clone(), data, forward, backward, threads)
            .expect("一次性多线程解码");
        assert_eq!(got.y.len(), want.y.len(), "帧 {n} 的 Y 长度不一致");
        assert_eq!(got.y, want.y, "帧 {n} 的 Y 不一致（切分/合并错了）");
        assert_eq!(got.u, want.u, "帧 {n} 的 U 不一致");
        assert_eq!(got.v, want.v, "帧 {n} 的 V 不一致");
        assert_eq!(got2.y, want.y, "帧 {n}：一次性路径的 Y 不一致");
        assert_eq!(got2.u, want.u, "帧 {n}：一次性路径的 U 不一致");
        assert_eq!(got2.v, want.v, "帧 {n}：一次性路径的 V 不一致");
        if matches!(kind, PictureType::I | PictureType::P) {
            prev_ip = last_ip.take();
            last_ip = Some(got);
        }
        n += 1;
    }
    eprintln!("{n} 帧逐字节一致（threads={threads}，MtDecoder + 一次性两条路径）");
    assert!(n >= 2, "只解了 {n} 帧");
}
