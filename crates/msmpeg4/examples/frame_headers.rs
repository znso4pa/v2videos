//! 读一个 ASF 文件，解析每一帧 MP43 视频帧头并统计。
//!
//! 用法：`cargo run -p v2v-msmpeg4 --example frame_headers -- <file.wmv>`

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::BufReader;

use v2v_asf::{AsfDemuxer, StreamKind};
use v2v_msmpeg4::{mb_count, parse_frame_header, PictureType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: frame_headers <file.wmv>")?;
    let mut d = AsfDemuxer::new(BufReader::new(File::open(&path)?))?;

    let video = d
        .streams()
        .iter()
        .find(|s| s.kind == StreamKind::Video)
        .ok_or("没有视频流")?;
    let vf = video.video.as_ref().ok_or("视频流没有格式信息")?;
    let video_number = video.number;
    let fourcc = v2v_asf::fourcc_string(vf.fourcc);
    let (mb_w, mb_h) = mb_count(vf.width, vf.height);
    let (width, height) = (vf.width, vf.height);
    println!(
        "视频流 #{video_number} {width}x{height} fourcc={fourcc} → {mb_w}x{mb_h} 宏块"
    );

    let mut i_frames = 0u64;
    let mut p_frames = 0u64;
    let mut q_hist: BTreeMap<u8, u64> = BTreeMap::new();
    let mut key_vs_i_mismatch = 0u64;
    let mut first_hex = String::new();
    let mut errors = 0u64;

    while let Some(s) = d.next_sample()? {
        if s.stream != video_number {
            continue;
        }
        if first_hex.is_empty() {
            first_hex = s
                .data
                .iter()
                .take(12)
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
        }
        match parse_frame_header(&s.data, mb_w, mb_h, v2v_msmpeg4::Version::V3, &mut Default::default()) {
            Ok((h, _)) => {
                match h.picture_type {
                    PictureType::I => i_frames += 1,
                    PictureType::P => p_frames += 1,
                }
                *q_hist.entry(h.qscale).or_insert(0) += 1;
                // 交叉校验：ASF 的关键帧标志应当 == I 帧。
                let is_i = h.picture_type == PictureType::I;
                if is_i != s.key_frame {
                    key_vs_i_mismatch += 1;
                }
            }
            Err(e) => {
                errors += 1;
                if errors <= 3 {
                    eprintln!("帧头解析失败：{e}");
                }
            }
        }
    }

    println!("首帧前 12 字节：{first_hex}");
    println!("I 帧 {i_frames} / P 帧 {p_frames} / 解析失败 {errors}");
    println!("I 帧 vs ASF 关键帧不一致：{key_vs_i_mismatch}");
    let qs: Vec<String> = q_hist.iter().map(|(q, n)| format!("{q}:{n}")).collect();
    println!("qscale 分布：{}", qs.join(" "));
    Ok(())
}
