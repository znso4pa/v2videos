//! 常驻线程池的**等价性**判据：`Pool::decode_into` 必须与
//! `v2v_mpeg12::decode::MtDecoder`（每帧 spawn 一遍的那条路）**逐字节相同**。
//!
//! ```bash
//! V2V_MPEG12_SAMPLE=<clip>.mpg \
//!   cargo test -p v2v-mpeg12-simd --test pool -- --ignored --nocapture
//! ```
//!
//! ## 为什么盯「逐字节」而不是 PSNR
//!
//! 池只改**谁来解哪几行、以及线程什么时候建**，不该改任何一行的值。有一位不同
//! 就说明：切片分配错了、缓冲续命错了（码流提前释放）、或者合并的区间错了。
//! 这三种错在 PSNR 上都会被平均掉。
//!
//! ## 为什么还有「连续多帧」那条
//!
//! 池的危险不在第一帧，而在**第二帧之后**：worker 是跨帧活着的，一块缓冲被
//! 复用、一个指针被重新解释。所以 `pool_reuse_matches_fresh_pool` 拿「建一次池
//! 解 N 帧」去对「每帧新建池解一帧」—— 这条专门抓「跨帧残留」。
//!
//! ⚠️ 这条测试**故意不并行**（`--test-threads=1` 不必）：池是 `!Sync` 的局部量，
//! 每条测试各建各的。

use v2v_mpeg12::decode::{MtDecoder, decode_frame, plane_dims};
use v2v_mpeg12::{PictureHeader, PictureType, SequenceHeader, split_pictures, split_units};
use v2v_mpeg12_simd::pool::Pool;
use v2v_mpegps::PsDemuxer;

/// 从 `.mpg` 里抽出视频 ES，并把 sequence header 那一小段解出来。
fn open_sample() -> Option<(Vec<u8>, SequenceHeader)> {
    let path = std::env::var_os("V2V_MPEG12_SAMPLE")?;
    let path = std::path::PathBuf::from(path);
    if !path.is_file() {
        eprintln!("SKIP: V2V_MPEG12_SAMPLE 指向的文件不存在");
        return None;
    }
    let mut demux = PsDemuxer::new(std::fs::read(&path).expect("读样本"));
    let mut es = Vec::new();
    while let Some(packet) = demux.next_packet() {
        if packet.is_video() {
            es.extend_from_slice(&demux.data()[packet.range.clone()]);
        }
    }
    if es.is_empty() {
        eprintln!("SKIP: 样本里没有视频");
        return None;
    }
    let units = split_units(&es);
    let mut sequence = units
        .iter()
        .find(|(code, _)| *code == 0xB3)
        .and_then(|(_, payload)| SequenceHeader::parse(payload))
        .expect("sequence header");
    sequence.mpeg2 = v2v_mpeg12::has_sequence_extension(&es);
    if sequence.mpeg2 {
        if let Some(ext) = units
            .iter()
            .find(|(code, _)| *code == 0xB5)
            .and_then(|(_, payload)| v2v_mpeg12::SequenceExtension::parse(payload))
        {
            sequence.apply_sequence_extension(&ext);
        }
    }
    Some((es, sequence))
}

/// 切组直接用 `v2v_mpeg12` 的那一份 —— **不要在测试里再写一遍**。
/// 两个说法迟早会漂，而漂了之后「两条路逐字节相同」这条判据验的就是错的东西。
fn groups_for(
    sequence: &SequenceHeader,
    data: &[u8],
    threads: usize,
) -> Vec<(usize, usize, Vec<u8>)> {
    v2v_mpeg12::decode::split_frame_groups(sequence, data, threads)
        .expect("样本太小 / 片太少，测不了并行")
}

fn same_frame(a: &v2v_mpeg12::decode::Frame, b: &v2v_mpeg12::decode::Frame) -> Option<String> {
    if a.width != b.width || a.height != b.height {
        return Some(format!("尺寸不同 {}x{} vs {}x{}", a.width, a.height, b.width, b.height));
    }
    for (name, x, y) in [("y", &a.y, &b.y), ("u", &a.u, &b.u), ("v", &a.v, &b.v)] {
        if x.len() != y.len() {
            return Some(format!("{name} 平面长度不同 {} vs {}", x.len(), y.len()));
        }
        if let Some(i) = x.iter().zip(y.iter()).position(|(p, q)| p != q) {
            return Some(format!("{name} 第一个不同在下标 {i}：{} vs {}", x[i], y[i]));
        }
    }
    None
}

/// 池解出来的帧必须与**单线程**逐字节相同。
#[test]
#[ignore = "needs V2V_MPEG12_SAMPLE (a .mpg)"]
fn pool_matches_single_threaded() {
    let Some((es, sequence)) = open_sample() else { return };
    let threads = 4usize;
    let mut pool = Pool::new(threads, &sequence);
    assert_eq!(pool.width(), threads);

    let mut checked = 0usize;
    let mut last_ip: Option<v2v_mpeg12::decode::Frame> = None;
    let mut prev_ip: Option<v2v_mpeg12::decode::Frame> = None;
    for (header, data) in split_pictures(&es) {
        if matches!(header.picture_type, PictureType::D) {
            continue;
        }
        // 单线程：拿同一条 `decode_frame`（它也自己做底色）。
        let forward = if matches!(header.picture_type, PictureType::B) {
            prev_ip.as_ref()
        } else {
            last_ip.as_ref()
        };
        let backward = last_ip.as_ref();
        let want = decode_frame(&sequence, header.clone(), data, forward, backward).expect("单线程解码");

        // 池：自己按行切、自己起手（`blank_frame_pub` + 未覆盖行的底色）。
        let (w, ph, cw, ch) = plane_dims(&want, sequence.chroma_format);
        let mut out = v2v_mpeg12::decode::blank_frame_pub(&sequence);
        let _ = (w, ph, cw, ch);
        let groups = groups_for(&sequence, data, threads);
        let first_start = groups.first().map(|g| g.0).unwrap_or(0);
        let last_end = groups.last().map(|g| g.1).unwrap_or(0);
        let mb_h = (sequence.height as usize).div_ceil(16);
        let intra = matches!(header.picture_type, PictureType::I);
        for (r0, r1) in [(0usize, first_start), (last_end, mb_h)] {
            if r0 >= r1 {
                continue;
            }
            match (intra, forward) {
                (true, _) | (false, None) => {
                    v2v_mpeg12::decode::fill_blank_rows_pub(&mut out, r0, r1, sequence.chroma_format)
                }
                (false, Some(f)) => {
                    v2v_mpeg12::decode::copy_rows_pub(&mut out, f, r0, r1, sequence.chroma_format)
                }
            }
        }
        pool.decode_into(&sequence, header.clone(), forward, backward, groups, &mut out)
            .expect("池解码");

        if let Some(why) = same_frame(&want, &out) {
            panic!("第 {checked} 帧（{:?}）与单线程不同：{why}", header.picture_type);
        }
        checked += 1;

        if matches!(header.picture_type, PictureType::I | PictureType::P) {
            prev_ip = last_ip.take();
            last_ip = Some(want);
        }
    }
    assert!(checked > 0, "一帧都没解出来");
    eprintln!("{checked} 帧逐字节一致（池 {threads} 线程 vs 单线程）");
}

/// **跨帧复用**必须与「每帧新建池」相同 —— 这条抓「池里残留了上一帧的数据」。
#[test]
#[ignore = "needs V2V_MPEG12_SAMPLE (a .mpg)"]
fn pool_reuse_matches_fresh_pool() {
    let Some((es, sequence)) = open_sample() else { return };
    let threads = 3usize;
    let pictures: Vec<_> = split_pictures(&es)
        .into_iter()
        .filter(|(h, _)| !matches!(h.picture_type, PictureType::D))
        .take(80)
        .collect();

    // 复用：一个池从头解到尾。
    let reused = run_pool(&sequence, &pictures, threads, true);
    // 每帧新建池（复用那条如果漏了「重建缓冲」就会被这条抓到）。
    let fresh = run_pool(&sequence, &pictures, threads, false);
    assert_eq!(reused.len(), fresh.len(), "帧数不同");
    for (i, (a, b)) in reused.iter().zip(fresh.iter()).enumerate() {
        if let Some(why) = same_frame(a, b) {
            panic!("第 {i} 帧：复用池 vs 新建池不同：{why}");
        }
    }
    eprintln!("{} 帧：复用池与新建池逐字节一致", reused.len());
}

fn run_pool(
    sequence: &SequenceHeader,
    pictures: &[(PictureHeader, &[u8])],
    threads: usize,
    reuse: bool,
) -> Vec<v2v_mpeg12::decode::Frame> {
    let mut out_frames = Vec::new();
    let mut pool = Pool::new(threads, sequence);
    let mut last_ip: Option<v2v_mpeg12::decode::Frame> = None;
    let mut prev_ip: Option<v2v_mpeg12::decode::Frame> = None;
    for (header, data) in pictures {
        if !reuse {
            pool = Pool::new(threads, sequence);
        }
        let forward = if matches!(header.picture_type, PictureType::B) {
            prev_ip.as_ref()
        } else {
            last_ip.as_ref()
        };
        let backward = last_ip.as_ref();
        let groups = groups_for(sequence, data, threads);
        let mb_h = (sequence.height as usize).div_ceil(16);
        let first_start = groups.first().map(|g| g.0).unwrap_or(0);
        let last_end = groups.last().map(|g| g.1).unwrap_or(0);
        let intra = matches!(header.picture_type, PictureType::I);
        let mut out = v2v_mpeg12::decode::blank_frame_pub(sequence);
        for (r0, r1) in [(0usize, first_start), (last_end, mb_h)] {
            if r0 >= r1 {
                continue;
            }
            match (intra, forward) {
                (true, _) | (false, None) => {
                    v2v_mpeg12::decode::fill_blank_rows_pub(&mut out, r0, r1, sequence.chroma_format)
                }
                (false, Some(f)) => {
                    v2v_mpeg12::decode::copy_rows_pub(&mut out, f, r0, r1, sequence.chroma_format)
                }
            }
        }
        pool.decode_into(sequence, header.clone(), forward, backward, groups, &mut out)
            .expect("池解码");
        if matches!(header.picture_type, PictureType::I | PictureType::P) {
            prev_ip = last_ip.take();
            last_ip = Some(out.clone());
        }
        out_frames.push(out);
    }
    out_frames
}

/// 池必须与 `MtDecoder`（每帧 spawn 那条路）出同样的字节 —— 这条才是「换实现
/// 不改产物」的判据。
#[test]
#[ignore = "needs V2V_MPEG12_SAMPLE (a .mpg)"]
fn pool_matches_mt_decoder() {
    let Some((es, sequence)) = open_sample() else { return };
    let threads = 4usize;
    let mut pool = Pool::new(threads, &sequence);
    let mut mt = MtDecoder::new(threads);

    let mut last_ip: Option<v2v_mpeg12::decode::Frame> = None;
    let mut prev_ip: Option<v2v_mpeg12::decode::Frame> = None;
    let mut n = 0usize;
    for (header, data) in split_pictures(&es) {
        if matches!(header.picture_type, PictureType::D) {
            continue;
        }
        let forward = if matches!(header.picture_type, PictureType::B) {
            prev_ip.as_ref()
        } else {
            last_ip.as_ref()
        };
        let backward = last_ip.as_ref();

        let want = mt.decode(&sequence, header.clone(), data, forward, backward).expect("MtDecoder");

        let groups = groups_for(&sequence, data, threads);
        let mb_h = (sequence.height as usize).div_ceil(16);
        let first_start = groups.first().map(|g| g.0).unwrap_or(0);
        let last_end = groups.last().map(|g| g.1).unwrap_or(0);
        let intra = matches!(header.picture_type, PictureType::I);
        let mut out = v2v_mpeg12::decode::blank_frame_pub(&sequence);
        for (r0, r1) in [(0usize, first_start), (last_end, mb_h)] {
            if r0 >= r1 {
                continue;
            }
            match (intra, forward) {
                (true, _) | (false, None) => {
                    v2v_mpeg12::decode::fill_blank_rows_pub(&mut out, r0, r1, sequence.chroma_format)
                }
                (false, Some(f)) => {
                    v2v_mpeg12::decode::copy_rows_pub(&mut out, f, r0, r1, sequence.chroma_format)
                }
            }
        }
        pool.decode_into(&sequence, header.clone(), forward, backward, groups, &mut out)
            .expect("池解码");
        if let Some(why) = same_frame(&want, &out) {
            panic!("第 {n} 帧（{:?}）：池 vs MtDecoder 不同：{why}", header.picture_type);
        }
        n += 1;
        if matches!(header.picture_type, PictureType::I | PictureType::P) {
            prev_ip = last_ip.take();
            last_ip = Some(want);
        }
    }
    assert!(n > 0);
    eprintln!("{n} 帧：池与 MtDecoder 逐字节一致");
}
