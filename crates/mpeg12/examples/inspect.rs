//! 打印一个 `.mpg` 的解析结果，用来和 ffprobe 对照。
//!
//! `cargo run -p mpeg12 --example inspect -- <clip>.mpg`

use v2v_mpeg12::{SequenceHeader, has_sequence_extension, split_pictures, split_units};
use v2v_mpegps::PsDemuxer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: inspect <clip.mpg>")?;
    let file = std::fs::read(&path)?;

    let mut demux = PsDemuxer::new(file);
    let mut es = Vec::new();
    let mut audio_packets = 0usize;
    let mut first_pts = None;
    while let Some(packet) = demux.next_packet() {
        if packet.is_video() {
            if first_pts.is_none() {
                first_pts = packet.pts_ms;
            }
            es.extend_from_slice(&demux.data()[packet.range.clone()]);
        } else if packet.is_audio() {
            audio_packets += 1;
        }
    }

    let mut sequence: Option<SequenceHeader> = None;
    let mut slices = 0usize;
    for (code, payload) in split_units(&es) {
        match code {
            0xB3 if sequence.is_none() => sequence = SequenceHeader::parse(payload),
            0x01..=0xAF => slices += 1,
            _ => {}
        }
    }
    let Some(seq) = sequence else {
        return Err("no sequence header".into());
    };

    let pictures = split_pictures(&es);
    let types: String = pictures.iter().map(|(h, _)| h.picture_type.letter()).collect();
    println!("size       {}x{}", seq.width, seq.height);
    println!("fps        {}", seq.frame_rate_milli as f64 / 1000.0);
    println!("aspect     {}:{}", seq.aspect.0, seq.aspect.1);
    println!("bit_rate   {}", seq.bit_rate);
    println!("mpeg2      {}", has_sequence_extension(&es));
    println!("pictures   {}", pictures.len());
    println!("slices     {slices}");
    println!("types      {types}");
    println!("audio_pkts {audio_packets}");
    println!("first_pts  {first_pts:?}");
    Ok(())
}
