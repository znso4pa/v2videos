//! 编码器单项计时：只读原始 YUV420P，只跑编码，不含解码与封装。
//!
//! 用法：`cargo run --release -p v2v-mp4v --example bench_encode -- \
//!        <raw.yuv> <宽> <高> <帧数> [qscale] [gop]`
//!
//! 打的是 `ms/帧` 与产物字节数的 **FNV 校验和** —— 优化前后校验和必须一致
//! （同一份输入、同一份参数下「同语义更快」的判据）。

use std::time::Instant;

use v2v_mp4v::{Encoder, EncoderConfig};
use v2v_kernel::mc::Frame;

fn fnv(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let raw = &args[0];
    let w: u32 = args[1].parse().expect("宽");
    let h: u32 = args[2].parse().expect("高");
    let n: usize = args[3].parse().expect("帧数");
    let qscale: u8 = args.get(4).map(|s| s.parse().expect("qscale")).unwrap_or(6);
    let gop: usize = args.get(5).map(|s| s.parse().expect("gop")).unwrap_or(250);

    let data = std::fs::read(raw).expect("读 raw");
    let ysize = (w * h) as usize;
    let csize = ysize / 4;
    let fsize = ysize + 2 * csize;
    assert!(data.len() >= fsize * n, "raw 不够 {n} 帧");

    let mut frames = Vec::with_capacity(n);
    for i in 0..n {
        let at = i * fsize;
        frames.push(Frame {
            width: w,
            height: h,
            y: data[at..at + ysize].to_vec(),
            u: data[at + ysize..at + ysize + csize].to_vec(),
            v: data[at + ysize + csize..at + fsize].to_vec(),
        });
    }
    drop(data);

    let cfg = EncoderConfig {
        width: w,
        height: h,
        qscale,
        keyframe_interval: gop,
        ..Default::default()
    };
    let mut enc = Encoder::new(cfg).expect("encoder");

    let mut sum = 0xcbf2_9ce4_8422_2325u64;
    let mut bytes = 0usize;
    let mut intra_mbs = 0usize;
    let mut intra_skips = 0usize;
    let t0 = Instant::now();
    for f in &frames {
        let out = enc.encode_frame(f).expect("encode");
        bytes += out.len();
        intra_mbs += enc.intra_mbs_last;
        intra_skips += enc.intra_skips_last;
        sum = fnv(&out, sum);
    }
    let dt = t0.elapsed().as_secs_f64();

    println!(
        "frames={n} qscale={qscale} gop={gop} total={dt:.3}s per_frame={:.3}ms bytes={bytes} fnv={sum:016x} intra_mbs={intra_mbs} intra_skips={intra_skips}",
        dt * 1000.0 / n as f64
    );
}
