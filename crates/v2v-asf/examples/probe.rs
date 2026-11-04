//! 探测一个 ASF 文件：打印流信息与样本统计。
//!
//! 用法：`cargo run -p v2v-asf --example probe -- <file.wmv>`

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: probe <file>")?;
    let f = File::open(&path)?;
    let mut d = v2v_asf::AsfDemuxer::new(BufReader::new(f))?;

    let info = d.info();
    println!("file: {path}");
    println!(
        "  file_size={} packet_count={} duration_ms={} preroll_ms={} min_pkt={} max_pkt={} flags=0x{:x}",
        info.file_size,
        info.packet_count,
        info.duration_ms(),
        info.preroll_ms,
        info.min_packet_size,
        info.max_packet_size,
        info.flags
    );

    for s in d.streams() {
        if let Some(a) = &s.audio {
            println!(
                "  stream {} audio  tag=0x{:04x} ch={} rate={} bits={} block_align={}",
                s.number, a.format_tag, a.channels, a.sample_rate, a.bits_per_sample, a.block_align
            );
        } else if let Some(v) = &s.video {
            println!(
                "  stream {} video  fourcc={} (0x{:08x}) {}x{} bpp={} extra={}B",
                s.number,
                v2v_asf::fourcc_string(v.fourcc),
                v.fourcc,
                v.width,
                v.height,
                v.bits_per_pixel,
                v.extra.len()
            );
        } else {
            println!("  stream {} {:?}", s.number, s.kind);
        }
    }

    // stream -> (count, key_count, min_ts, max_ts, min_size, max_size)
    let mut counts: BTreeMap<u8, (u64, u64, i64, i64, usize, usize)> = BTreeMap::new();
    let mut total = 0u64;
    while let Some(s) = d.next_sample()? {
        total += 1;
        let e = counts
            .entry(s.stream)
            .or_insert((0, 0, i64::MAX, i64::MIN, usize::MAX, 0));
        e.0 += 1;
        if s.key_frame {
            e.1 += 1;
        }
        e.2 = e.2.min(s.timestamp_ms);
        e.3 = e.3.max(s.timestamp_ms);
        e.4 = e.4.min(s.data.len());
        e.5 = e.5.max(s.data.len());
    }
    println!("  total samples: {total}");
    for (st, (n, k, t0, t1, mn, mx)) in counts {
        println!("  stream {st}: samples={n} key={k} ts_ms=[{t0}..{t1}] size=[{mn}..{mx}]");
    }
    Ok(())
}
