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
