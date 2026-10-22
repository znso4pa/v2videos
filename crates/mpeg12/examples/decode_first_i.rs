//! 解出第一个 I 帧，按 I420 写到指定路径，用来和 ffmpeg 的首帧逐像素比。
//!
//! `cargo run -p v2v-mpeg12 --example decode_first_i -- <clip.mpg> <out.yuv>`

use v2v_mpeg12::{SequenceHeader, decode::decode_i_frame, split_pictures, split_units};
use v2v_mpegps::PsDemuxer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (input, output) = (&args[1], &args[2]);

    let mut demux = PsDemuxer::new(std::fs::read(input)?);
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
        .ok_or("no sequence header")?;
    sequence.mpeg2 = v2v_mpeg12::has_sequence_extension(&es);

    for (header, data) in split_pictures(&es) {
        if !matches!(header.picture_type, v2v_mpeg12::PictureType::I) {
            continue;
        }
        match decode_i_frame(&sequence, header, data) {
            Ok(frame) => {
                let mut out = Vec::new();
                out.extend_from_slice(&frame.y);
                out.extend_from_slice(&frame.u);
                out.extend_from_slice(&frame.v);
                std::fs::write(output, &out)?;
                println!("{}x{} → {} bytes", frame.width, frame.height, out.len());
                return Ok(());
            }
            Err(e) => {
                eprintln!("decode failed: stage={} bit={} detail={}", e.stage, e.bit, e.detail);
                return Err("decode failed".into());
            }
        }
    }
    Err("no I picture".into())
}
