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

use v2v_mpeg12::{PictureType, SequenceHeader, decode::decode_i_frame, split_pictures, split_units};
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

    let mut sequence = split_units(&es)
        .iter()
        .find(|(code, _)| *code == 0xB3)
        .and_then(|(_, payload)| SequenceHeader::parse(payload))
        .expect("a sequence header");
    sequence.mpeg2 = v2v_mpeg12::has_sequence_extension(&es);

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
