//! 打印一个 ASF 文件里所有流的信息（含音频的 WAVEFORMATEX 字段与编解码器私有数据）。
//!
//! 用法：`cargo run -p v2v-asf --example dump_streams -- <文件.wmv|.wma>`
//!
//! 存在的理由：做音频解码时要拿到**真实的** `block_align` / `avg_bytes_per_sec` /
//! 私有数据，这些在 `ffprobe` 的输出里不齐全（它报的是它自己换算过的值）。

use std::env;
use std::fs::File;
use std::io::BufReader;

use v2v_asf::AsfDemuxer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("用法：dump_streams <文件>")?;
    let mut d = AsfDemuxer::new(BufReader::new(File::open(&path)?))?;

    let i = d.info();
    println!("文件：{path}");
    println!(
        "  大小 {} 字节 / 包数 {} / 包大小 {}..{} / preroll {} ms / 时长 {} ms",
        i.file_size,
        i.packet_count,
        i.min_packet_size,
        i.max_packet_size,
        i.preroll_ms,
        i.duration_ms()
    );

    for s in d.streams() {
        println!("流 #{} {:?}（time_offset {} ms）", s.number, s.kind, s.time_offset_ms);
        if let Some(a) = &s.audio {
            println!(
                "  音频：format_tag 0x{:04X} / {} Hz / {} ch / avg {} B/s / block_align {} / {} bit",
                a.format_tag,
                a.sample_rate,
                a.channels,
                a.avg_bytes_per_sec,
                a.block_align,
                a.bits_per_sample
            );
            println!("  私有数据 {} 字节：{:02x?}", a.extra.len(), a.extra);
        }
        if let Some(v) = &s.video {
            println!(
                "  视频：{}×{}（声明 {}×{}）/ {} bpp / fourcc {}",
                v.width,
                v.height,
                v.encoded_width,
                v.encoded_height,
                v.bits_per_pixel,
                v2v_asf::fourcc_string(v.fourcc)
            );
            println!("  私有数据 {} 字节", v.extra.len());
        }
    }

    // 样本计数与首末时间戳（按流分）
    let mut counts = vec![0usize; 128];
    let mut first = vec![None; 128];
    let mut last = vec![None; 128];
    let mut sizes = vec![(0usize, 0usize); 128];
    // 读到出错也要把「读了多少」打出来 —— 排查坏样本时这才是关键信息。
    loop {
        let s = match d.next_sample() {
            Ok(Some(s)) => s,
            Ok(None) => break,
            Err(e) => {
                eprintln!(
                    "读样本失败（已读 {} 个：音频 {} / 视频 {}）：{e}",
                    counts.iter().sum::<usize>(),
                    counts.get(1).copied().unwrap_or(0),
                    counts.get(2).copied().unwrap_or(0)
                );
                break;
            }
        };
        let k = s.stream as usize;
        if k >= 128 {
            continue;
        }
        counts[k] += 1;
        first[k].get_or_insert(s.timestamp_ms);
        last[k] = Some(s.timestamp_ms);
        let (mn, mx) = sizes[k];
        let n = s.data.len();
        sizes[k] = (
            if mn == 0 { n } else { mn.min(n) },
            mx.max(n),
        );
    }
    for (k, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        println!(
            "流 #{k}：{c} 个样本，时间戳 {}..{} ms，样本大小 {}..{} 字节",
            first[k].unwrap_or(0),
            last[k].unwrap_or(0),
            sizes[k].0,
            sizes[k].1
        );
    }
    Ok(())
}
