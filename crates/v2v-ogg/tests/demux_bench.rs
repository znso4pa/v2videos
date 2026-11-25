//! Ogg 解封装吞吐基准 —— 页层 / 包层 / 时间戳三层各占多少。
//!
//! 平时 `#[ignore]`，要量的时候显式跑：
//!
//! ```bash
//! V2V_OGG_SAMPLE=/path/x.ogg cargo test -p v2v-ogg --release --test demux_bench -- --ignored --nocapture
//! ```
//!
//! 除了吞吐，它还数**底层 `read` 调用次数** —— 页头 / 段表 / 页体是三次独立读，
//! 若底层是未缓冲的 `File`，一页就撞三次系统调用。这是「容器层慢」最常见的形态，
//! 而它**不会在任何产物字节里体现**，只能靠数调用来发现。
//!
//! 最后打印一个**输出摘要**（包长 + 字节 + 时间戳滚 FNV）：改动前后摘要相同
//! 才说明「只是更快，产物没变」。摘要单独跑一遍，不进计时区间。

use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::time::Instant;

use v2v_ogg::{OggDemuxer, PacketReader, PacketTiming};

/// 包住底层 reader，数 `read` 调用次数与字节数。
struct Counting<R> {
    inner: R,
    calls: u64,
}

impl<R: Read> Counting<R> {
    fn new(inner: R) -> Self {
        Self { inner, calls: 0 }
    }
}

impl<R: Read> Read for Counting<R> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        let n = self.inner.read(buf)?;
        self.calls += 1;
        Ok(n)
    }
}

struct ConstantLongBlock;

impl PacketTiming for ConstantLongBlock {
    fn advance(&self, _prev: &[u8], _cur: &[u8]) -> Option<u32> {
        Some(1024)
    }
}

fn sample() -> String {
    match std::env::var("V2V_OGG_SAMPLE") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SKIP: V2V_OGG_SAMPLE 未设置");
            std::process::exit(0);
        }
    }
}

/// 一遍完整解封装：开文件 → 读头 → 逐包。返回 (payload 字节, 包数, read 调用数)。
fn once(path: &str) -> (u64, usize, u64) {
    let mut c = Counting::new(File::open(path).unwrap());
    let mut d = OggDemuxer::open(&mut c).unwrap();
    d.set_timing(Box::new(ConstantLongBlock));
    let mut packets = 0usize;
    let mut bytes = 0u64;
    while let Some(s) = d.next_sample().unwrap() {
        packets += 1;
        bytes += s.data.len() as u64;
    }
    drop(d);
    (bytes, packets, c.calls)
}

/// 输出摘要：包长 + 字节 + 时间戳一起滚 FNV-1a。
fn digest(path: &str) -> u64 {
    let mut d = OggDemuxer::open(File::open(path).unwrap()).unwrap();
    d.set_timing(Box::new(ConstantLongBlock));
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let roll = |b: u8, h: &mut u64| {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    };
    while let Some(s) = d.next_sample().unwrap() {
        for b in s.data.iter() {
            roll(*b, &mut h);
        }
        for b in s.timestamp_ms.to_le_bytes() {
            roll(b, &mut h);
        }
        for b in (s.data.len() as u64).to_le_bytes() {
            roll(b, &mut h);
        }
    }
    h
}

#[test]
#[ignore]
fn demux_throughput() {
    let path = sample();
    let reps = 15;
    let mut best = f64::MAX;
    let mut best_packets = 0usize;
    let mut total_bytes = 0u64;
    let mut best_calls = 0u64;
    for _ in 0..reps {
        let t = Instant::now();
        let (bytes, packets, calls) = once(&path);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if ms < best {
            best = ms;
            best_calls = calls;
        }
        best_packets = packets;
        total_bytes = bytes;
    }
    let file_len = std::fs::metadata(&path).unwrap().len();
    eprintln!(
        "file {} B / {} 包 / payload {} B / 摘要 {:016x}",
        file_len,
        best_packets,
        total_bytes,
        digest(&path)
    );
    eprintln!(
        "demux {:.2} ms  ({:.1} MB/s 文件口径, {:.2} µs/包, {} 次 read)",
        best,
        file_len as f64 / 1e6 / (best / 1e3),
        best * 1e3 / best_packets as f64,
        best_calls
    );

    // 页层单独量：只读页，不拆包。
    let mut best_page = f64::MAX;
    let mut pages = 0usize;
    let mut calls = 0u64;
    for _ in 0..reps {
        let mut c = Counting::new(File::open(&path).unwrap());
        let t = Instant::now();
        let mut r = v2v_ogg::PageReader::new(&mut c);
        let mut n = 0usize;
        while r.next_page().unwrap().is_some() {
            n += 1;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if ms < best_page {
            best_page = ms;
        }
        pages = n;
        calls = c.calls;
    }
    eprintln!(
        "page  {:.2} ms / {} 页 / {} 次 read（{:.2} 次/页）",
        best_page,
        pages,
        calls,
        calls as f64 / pages as f64
    );

    // 包层单独量：页 → 包，不建 OggDemuxer（不做时间戳）。
    let mut best_pkt = f64::MAX;
    let mut n_pkt = 0usize;
    for _ in 0..reps {
        let mut c = Counting::new(File::open(&path).unwrap());
        let t = Instant::now();
        let mut r = PacketReader::new(&mut c);
        let mut n = 0usize;
        while r.next_packet().unwrap().is_some() {
            n += 1;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if ms < best_pkt {
            best_pkt = ms;
        }
        n_pkt = n;
    }
    eprintln!("packet {:.2} ms / {} 包", best_pkt, n_pkt);

    // 同一份数据放进内存（`Cursor`，零系统调用）—— 这是「读路径到底有多少是 syscall」的上界。
    let all = std::fs::read(&path).unwrap();
    let mut best_mem = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        let mut d = OggDemuxer::open(std::io::Cursor::new(&all[..])).unwrap();
        d.set_timing(Box::new(ConstantLongBlock));
        while d.next_sample().unwrap().is_some() {}
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if ms < best_mem {
            best_mem = ms;
        }
    }
    eprintln!("mem   {:.2} ms（同一份数据，无系统调用）", best_mem);
}
