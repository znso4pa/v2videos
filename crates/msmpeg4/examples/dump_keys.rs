//! 打印前几个关键帧的样本尺寸与首字节。
use std::fs::File;
use std::io::BufReader;
use v2v_asf::{AsfDemuxer, StreamKind};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: dump_keys <file>")?;
    let mut d = AsfDemuxer::new(BufReader::new(File::open(&path)?))?;
    let vnum = d.streams().iter().find(|s| s.kind == StreamKind::Video).unwrap().number;
    let mut k = 0usize;
    while let Some(s) = d.next_sample()? {
        if s.stream == vnum && s.key_frame {
            if k < 6 {
                println!("key#{k}: 大小={} 前8字节={:02x?}", s.data.len(), &s.data[..8.min(s.data.len())]);
            }
            k += 1;
            if k >= 6 { break; }
        }
    }
    Ok(())
}
