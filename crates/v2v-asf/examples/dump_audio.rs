//! 把某条**音频流**的样本原样导出一个裸文件（不做任何解析）。
//!
//! 用法：`cargo run -p v2v-asf --example dump_audio -- <文件> [流号=1] [输出.raw]`
//!
//! 存在的理由：做音频解码时要拿真样本的载荷字节反复试（位序、帧头字段），
//! 导出成裸文件后可以用脚本/其它语言随便切，不用每次都跑 Rust。

use std::env;
use std::fs::File;
use std::io::{BufReader, Write};

use v2v_asf::{AsfDemuxer, StreamKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("用法：dump_audio <文件> [流号] [输出.raw]")?;
    let want: u8 = args
        .next()
        .map(|s| s.parse().unwrap_or(1))
        .unwrap_or(1);
    let out = args.next().unwrap_or_else(|| "/tmp/audio.raw".into());

    let mut d = AsfDemuxer::new(BufReader::new(File::open(&path)?))?;
    let streams: Vec<(u8, StreamKind, usize)> = d
        .streams()
        .iter()
        .map(|s| (s.number, s.kind, s.audio.as_ref().map(|a| a.block_align as usize).unwrap_or(0)))
        .collect();
    eprintln!("流：{streams:?}（要导的是流 {want}）");

    let mut f = File::create(&out)?;
    let mut n = 0usize;
    let mut bytes = 0usize;
    loop {
        match d.next_sample() {
            Ok(Some(s)) if s.stream == want => {
                f.write_all(&s.data)?;
                n += 1;
                bytes += s.data.len();
            }
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(e) => {
                eprintln!("读样本中断（已导出 {n} 个）：{e}");
                break;
            }
        }
    }
    eprintln!("导出 {n} 个样本、{bytes} 字节 → {out}");
    Ok(())
}
