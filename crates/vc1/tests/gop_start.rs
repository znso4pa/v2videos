//! **「从任意 I 帧起解」等价于「整条链解到那里」吗？**
//!
//! 这条是**按 GOP 并行解码**的前提：如果每条 GOP 都可以用一条全新的解码器、
//! 从它的 I 帧开始独立解，并且像素与串行链完全一致，那并行就是安全的；
//! 只要有一个像素不同，就说明有**跨 GOP 的状态**（`rnd` 的翻转、dquant 的
//! `pqscale`、亮度补偿表 `luts` 的轮转链），并行必须先把它补上。
//!
//! 判据：**逐像素 0 差**，不是「看着差不多」。
//!
//! ```text
//! V2V_ASF_SAMPLE=<样本> cargo test --release -p v2v-vc1 --test gop_start -- --nocapture
//! ```

mod common;

use std::fs::File;
use std::io::BufReader;

use v2v_asf::AsfDemuxer;
use v2v_vc1::{Decoder, SeqHeader};

#[test]
fn fresh_decoder_at_key_frame_matches_serial_chain() {
    let samples = common::sample_list();
    if samples.is_empty() {
        eprintln!("SKIP: 没设 V2V_ASF_SAMPLE");
        return;
    }
    let path = &samples[0];
    let Some(track) = common::video_track(path) else {
        eprintln!("SKIP {path}: 取不到视频轨");
        return;
    };
    if track.fourcc != common::FOURCC_WMV3 {
        eprintln!("SKIP {path}: 不是 WMV3（fourcc {:#x}）", track.fourcc);
        return;
    }
    let seq = SeqHeader::parse(&track.extra).expect("序列头");

    // 取视频样本 + ASF 的关键帧标志（= I 帧）。
    let mut d = AsfDemuxer::new(BufReader::new(File::open(path).expect("开样本"))).expect("解封装");
    let mut frames: Vec<(Vec<u8>, bool)> = Vec::new();
    while let Some(s) = d.next_sample().expect("读样本") {
        if s.stream == track.number {
            frames.push((s.data, s.key_frame));
        }
    }
    assert!(!frames.is_empty(), "一个视频帧都没有");

    let (w, h) = (track.encoded_width, track.encoded_height);
    let mut serial = Decoder::new(seq, w, h).expect("建解码器");

    let mut checked = 0usize;
    let mut key_frames = 0usize;
    let mut bad = Vec::new();

    for (i, (data, key)) in frames.iter().enumerate() {
        serial.decode_frame(data).expect("串行解码");
        if !key {
            continue;
        }
        key_frames += 1;
        let mut fresh = Decoder::new(seq, w, h).expect("建解码器");
        fresh.decode_frame(data).expect("从 I 帧起解");
        let (a, b) = (serial.planes(), fresh.planes());
        let diff = a
            .y
            .iter()
            .zip(&b.y)
            .chain(a.u.iter().zip(&b.u))
            .chain(a.v.iter().zip(&b.v))
            .filter(|(x, y)| x != y)
            .count();
        if diff != 0 {
            bad.push((i, diff));
        }
        checked += 1;
    }

    eprintln!(
        "{path}: {key_frames} 个关键帧，抽查 {checked} 个；不一致 {} 个",
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "从 I 帧起解与整条链不一致（帧号, 差异像素数）：{bad:?}\n\
         说明有跨 GOP 的状态没补上 —— 按 GOP 并行之前必须先处理它"
    );
}
