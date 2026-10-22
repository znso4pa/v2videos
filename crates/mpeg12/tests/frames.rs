//! 对真样本的逐像素对账：解出的第一个 I 帧必须与参考实现的首帧一致。
//!
//! ```bash
//! V2V_MPEG12_SAMPLE=<clip>.mpg V2V_MPEG12_REF=<首帧>.yuv \
//!   cargo test -p v2v-mpeg12 --test frames -- --ignored
//! ```
//!
//! 参考件用 `ffmpeg -frames:v 1 -f rawvideo -pix_fmt yuv420p` 生成。
//! 判据是**逐像素**，容差 2：浮点 IDCT 与整数 IDCT 在舍入上会有 ±1 差别，
//! 那是实现选择；任何更大的差别都是 bug。

use v2v_mpeg12::{
    PictureType, SequenceHeader, decode::decode_frame, decode::decode_i_frame, split_pictures,
    split_units,
};
use v2v_mpegps::PsDemuxer;

#[test]
#[ignore = "needs V2V_MPEG12_SAMPLE (a .mpg) and V2V_MPEG12_REF (its first frame as I420)"]
fn the_first_i_frame_matches_the_reference() {
    let sample = std::env::var("V2V_MPEG12_SAMPLE").expect("set V2V_MPEG12_SAMPLE");
    let reference = std::env::var("V2V_MPEG12_REF").expect("set V2V_MPEG12_REF");

    let mut demux = PsDemuxer::new(std::fs::read(&sample).expect("read the program stream"));
    let mut es = Vec::new();
    while let Some(packet) = demux.next_packet() {
        if packet.is_video() {
            es.extend_from_slice(&demux.data()[packet.range.clone()]);
        }
    }

    let sequence = parse_sequence(&es);

    let mut decoded = None;
    for (header, data) in split_pictures(&es) {
        if matches!(header.picture_type, PictureType::I) {
            decoded = Some(decode_i_frame(&sequence, header, data).expect("decode the I frame"));
            break;
        }
    }
    let frame = decoded.expect("an I picture");

    let mut ours = Vec::new();
    ours.extend_from_slice(&frame.y);
    ours.extend_from_slice(&frame.u);
    ours.extend_from_slice(&frame.v);
    let expected = std::fs::read(&reference).expect("read the reference frame");

    assert_eq!(ours.len(), expected.len(), "frame size differs from the reference");
    let (mut worst, mut at) = (0i32, 0usize);
    for (i, (&a, &b)) in ours.iter().zip(&expected).enumerate() {
        let diff = (a as i32 - b as i32).abs();
        if diff > worst {
            worst = diff;
            at = i;
        }
    }
    assert!(
        worst <= 2,
        "pixel {at} differs by {worst}: the reconstruction is wrong, not just rounded differently"
    );
}

/// 逐帧对账：I → P → P …，每帧都以参考实现（ffmpeg）的同一帧为准。
///
/// ```bash
/// V2V_MPEG12_SAMPLE=<clip>.mpg V2V_MPEG12_REF_ALL=<all frames>.yuv \
///   cargo test -p v2v-mpeg12 --test frames every_frame -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs V2V_MPEG12_SAMPLE and V2V_MPEG12_REF_ALL (all frames, I420)"]
fn every_frame_matches_the_reference() {
    let sample = std::env::var("V2V_MPEG12_SAMPLE").expect("set V2V_MPEG12_SAMPLE");
    let reference = std::env::var("V2V_MPEG12_REF_ALL").expect("set V2V_MPEG12_REF_ALL");

    let mut demux = PsDemuxer::new(std::fs::read(&sample).expect("read the program stream"));
    let mut es = Vec::new();
    while let Some(packet) = demux.next_packet() {
        if packet.is_video() {
            es.extend_from_slice(&demux.data()[packet.range.clone()]);
        }
    }

    let sequence = parse_sequence(&es);

    let frame_size = (sequence.width * sequence.height * 3 / 2) as usize;
    let expected = std::fs::read(&reference).expect("read the reference frames");
    let available = expected.len() / frame_size;

    // 解码序 ≠ 显示序：I/P 先解、B 后解。按解码序维护两路参考帧，
    // 最后按 temporal_reference 排序得到显示序，再与参考件比。
    let mut prev_ip: Option<v2v_mpeg12::decode::Frame> = None;
    let mut last_ip: Option<v2v_mpeg12::decode::Frame> = None;
    let mut decoded: Vec<(u32, PictureType, v2v_mpeg12::decode::Frame)> = Vec::new();
    for (header, data) in split_pictures(&es) {
        let kind = header.picture_type;
        let tr = header.temporal_reference;
        if matches!(kind, PictureType::D) {
            continue;
        }
        let forward = if matches!(kind, PictureType::B) {
            prev_ip.as_ref()
        } else {
            last_ip.as_ref()
        };
        let backward = last_ip.as_ref();
        let frame =
            decode_frame(&sequence, header, data, forward, backward).expect("decode a picture");
        if matches!(kind, PictureType::I | PictureType::P) {
            prev_ip = last_ip.take();
            last_ip = Some(frame.clone());
        }
        decoded.push((tr, kind, frame));
    }
    decoded.sort_by_key(|(tr, _, _)| *tr);

    for (i, (tr, kind, frame)) in decoded.iter().enumerate() {
        if i >= available {
            break;
        }
        let mut ours = Vec::new();
        ours.extend_from_slice(&frame.y);
        ours.extend_from_slice(&frame.u);
        ours.extend_from_slice(&frame.v);
        let want = &expected[i * frame_size..(i + 1) * frame_size];

        let (mut worst, mut at) = (0i32, 0usize);
        let mut sum_sq = 0.0f64;
        for (j, (&a, &b)) in ours.iter().zip(want).enumerate() {
            let diff = (a as i32 - b as i32).abs();
            sum_sq += (diff * diff) as f64;
            if diff > worst {
                worst = diff;
                at = j;
            }
        }
        let mse = sum_sq / ours.len() as f64;
        let psnr = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };
        eprintln!("display {i} (tr={tr}, {kind:?}): worst {worst} at {at}, psnr {psnr:.1} dB");

        // I 帧是单帧重建，逐像素卡 ≤2；P/B 帧把 ±1 的 IDCT 舍入逐帧累积，
        // 所以用 PSNR + 一个较宽的硬上限，而不是逐像素 ≤2。
        match kind {
            PictureType::I => assert!(
                worst <= 2,
                "display {i} pixel {at} differs by {worst}: the reconstruction is wrong"
            ),
            _ => assert!(
                psnr >= 50.0 && worst <= 4,
                "display {i}: psnr {psnr:.1} dB, worst {worst} at {at}"
            ),
        }
    }

    assert!(
        decoded.len() >= 2,
        "only decoded {} frames; need at least an I and a P",
        decoded.len()
    );
}

/// 解析 sequence header，并把 MPEG-2 的 sequence extension 并进来。
fn parse_sequence(es: &[u8]) -> SequenceHeader {
    let units = split_units(es);
    let mut sequence = units
        .iter()
        .find(|(code, _)| *code == 0xB3)
        .and_then(|(_, payload)| SequenceHeader::parse(payload))
        .expect("a sequence header");
    sequence.mpeg2 = v2v_mpeg12::has_sequence_extension(es);
    if sequence.mpeg2 {
        if let Some(ext) = units
            .iter()
            .find(|(code, _)| *code == 0xB5)
            .and_then(|(_, payload)| v2v_mpeg12::SequenceExtension::parse(payload))
        {
            sequence.apply_sequence_extension(&ext);
        }
    }
    sequence
}
