//! `MpgPlayer` 的显示序对账：解出来的帧序列必须与 ffmpeg 的**显示序**逐帧一致。
//!
//! ```bash
//! V2V_MPEG12_SAMPLE=<clip>.mpg V2V_MPEG12_REF_ALL=<all frames>.yuv \
//!   cargo test -p v2v-player --features mpeg12 --test mpeg -- --ignored --nocapture
//! ```
//!
//! 参考件：`ffmpeg -i x.mpg -f rawvideo -pix_fmt yuv420p all.yuv`。
//! 这条钉的是**播放器**（GOP 缓冲 + 按 `temporal_reference` 排显示序 + PTS），
//! 解码本身由 `crates/mpeg12/tests/frames.rs` 钉着。

#![cfg(feature = "mpeg12")]

use std::path::Path;

#[test]
#[ignore = "needs V2V_MPEG12_SAMPLE (a .mpg) and V2V_MPEG12_REF_ALL"]
fn display_order_matches_ffmpeg() {
    let sample = std::env::var("V2V_MPEG12_SAMPLE").expect("set V2V_MPEG12_SAMPLE");
    let reference = std::env::var("V2V_MPEG12_REF_ALL").expect("set V2V_MPEG12_REF_ALL");

    let mut p = v2v_player::mpeg::MpgPlayer::open(Path::new(&sample)).expect("打开");
    let (w, h) = p.size();
    let frame_size = (w * h * 3 / 2) as usize;
    let want = std::fs::read(&reference).expect("读参考帧");
    let available = want.len() / frame_size;

    let mut n = 0usize;
    let mut worst = 0i32;
    let mut last_pts = -1i64;
    while let Some(f) = p.next_video().expect("解一帧") {
        assert!(f.pts_ms > last_pts, "PTS 必须严格递增（帧 {n}）");
        last_pts = f.pts_ms;
        if n >= available {
            break;
        }
        let mut ours = Vec::with_capacity(frame_size);
        ours.extend_from_slice(&f.y);
        ours.extend_from_slice(&f.u);
        ours.extend_from_slice(&f.v);
        let theirs = &want[n * frame_size..(n + 1) * frame_size];
        let d = ours
            .iter()
            .zip(theirs)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap_or(0);
        worst = worst.max(d);
        assert!(d <= 2, "显示序第 {n} 帧差 {d}：顺序或重建不对");
        n += 1;
    }
    eprintln!("比了 {n} 帧（参考共 {available} 帧），最差 {worst}");
    assert!(n >= available, "只交出 {n} 帧，参考有 {available} 帧");
}

/// 切片级并行（`open_with_threads`）交出的帧必须与单线程**逐字节相同**。
///
/// 为什么这条要单独有：并行只改「谁来解哪几行」，不该改任何像素。解码器那一层
/// 已经由 `mpeg12/tests/mt.rs` 钉着，但**播放器**这一层多了一件事 ——
/// 按 `temporal_reference` 重排 + 参考帧的 `clone`，两者都得跟着对上。
/// 而且这条不需要参考件，只要一份样本。
#[test]
#[ignore = "needs V2V_MPEG12_SAMPLE (a .mpg)"]
fn threads_produce_identical_frames() {
    let sample = std::env::var("V2V_MPEG12_SAMPLE").expect("set V2V_MPEG12_SAMPLE");
    let path = Path::new(&sample);

    let mut single = v2v_player::mpeg::MpgPlayer::open_with_threads(path, 1).expect("打开");
    let mut parallel = v2v_player::mpeg::MpgPlayer::open_with_threads(path, 4).expect("打开");

    let mut n = 0usize;
    loop {
        let a = single.next_video().expect("单线程解一帧");
        let b = parallel.next_video().expect("并行解一帧");
        match (a, b) {
            (None, None) => break,
            (Some(a), Some(b)) => {
                assert_eq!(a.pts_ms, b.pts_ms, "第 {n} 帧 PTS 不同");
                assert_eq!(a.key_frame, b.key_frame, "第 {n} 帧关键帧标记不同");
                assert!(a.y == b.y && a.u == b.u && a.v == b.v, "第 {n} 帧像素不同");
                n += 1;
            }
            (a, b) => panic!("第 {n} 帧一有一没有：单线程 {:?} / 并行 {:?}", a.is_some(), b.is_some()),
        }
    }
    eprintln!("{n} 帧逐字节一致（1 线程 vs 4 线程）");
    assert!(n > 0, "一帧都没解出来");
}
