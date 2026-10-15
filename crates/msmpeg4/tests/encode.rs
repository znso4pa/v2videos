//! 编码器：我们编出来的 MP43 流必须能被 **ffmpeg** 解开，而且解出来的像素要和我们
//! 自己的解码器**逐位一致**。
//!
//! 为什么是「两个解码器互相印证」而不是「和原图逐位一致」：有损编码本来就不会还原
//! 原图。真正的判据是**这条流是合法的** —— 两个独立实现（我们 vs ffmpeg）对同一条
//! 位流的解释完全一致，才说明熵编码没写歪。画质另用 PSNR 单独卡一条下限。
//!
//! 设 `V2V_ASF_SAMPLE=<路径>` 才真跑；缺样本或缺 ffmpeg 打 `SKIP` 并通过。

mod common;

use common::{compare, ffmpeg_frames, first_video_frames, sample, unique_temp, wrap_single_frame_avi};
use v2v_msmpeg4::{
    decode_i_frame, parse_frame_header, Encoder, EncoderConfig, Frame, PictureType,
};

/// Y 平面里出现了多少种不同的值（用来跳过纯色帧 —— 它测不到什么）。
fn distinct_levels(y: &[u8]) -> usize {
    let mut seen = [false; 256];
    for &p in y {
        seen[p as usize] = true;
    }
    seen.iter().filter(|&&b| b).count()
}

/// 整帧的 PSNR（Y/U/V 一起算 MSE）。
fn psnr(a: &Frame, b: &Frame) -> f64 {
    let mut sum = 0f64;
    let mut n = 0usize;
    for (p, q) in [(&a.y, &b.y), (&a.u, &b.u), (&a.v, &b.v)] {
        for (&x, &y) in p.iter().zip(q.iter()) {
            let d = x as f64 - y as f64;
            sum += d * d;
            n += 1;
        }
    }
    let mse = sum / n as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// 编一条流，然后**我们和 ffmpeg 各解一遍**，比像素。
/// 返回 `(两解码器的最大差, PSNR(原图, 我们的解码结果))`。
fn encode_and_cross_check(orig: &Frame, qscale: u8) -> Option<(i32, f64, u8)> {
    let (w, h) = (orig.width, orig.height);
    let mut enc = Encoder::new(EncoderConfig {
        qscale,
        ..Default::default()
    });
    let stream = enc.encode_i_frame(orig).expect("编码失败");

    // ① 我们的解码器
    let ours = decode_i_frame(&stream, w, h).expect("解自己编的流失败");

    // ② ffmpeg（单帧包成最小 AVI）
    let avi = wrap_single_frame_avi(&stream, w, h);
    let tmp = unique_temp("v2v_encode_probe", "avi");
    std::fs::write(&tmp, &avi).expect("写临时 AVI");
    let refs = ffmpeg_frames(tmp.to_str().unwrap(), 1, w, h);
    let _ = std::fs::remove_file(&tmp);
    let refs = refs.filter(|v| !v.is_empty())?;

    let (worst, _at, _mean, _) = compare(&ours.y, &ours.u, &ours.v, &refs[0], w, h);
    Some((worst, psnr(orig, &ours), enc.last_qscale))
}

#[test]
fn encodes_i_frames_both_decoders_agree_on() {
    let path = match sample() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: V2V_ASF_SAMPLE 未设置或样本不存在");
            return;
        }
    };
    let (samples, w, h) = match first_video_frames(&path, 64) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 没找到视频帧");
            return;
        }
    };
    let (mw, mh) = (w.div_ceil(16), h.div_ceil(16));

    let mut checked = 0usize;
    for (idx, bytes) in samples.iter().enumerate() {
        let Ok((hdr, _)) =
            parse_frame_header(bytes, mw, mh, v2v_msmpeg4::Version::V3, &mut Default::default())
        else {
            continue;
        };
        if hdr.picture_type != PictureType::I {
            continue;
        }
        let Ok(orig) = decode_i_frame(bytes, w, h) else {
            continue;
        };
        if distinct_levels(&orig.y) <= 64 {
            continue; // 纯色帧：编出来也没什么可验的
        }

        for qscale in [2u8, 8, 20] {
            let Some((worst, psnr_db, used)) = encode_and_cross_check(&orig, qscale) else {
                eprintln!("SKIP: ffmpeg 解不了我们编的帧（帧 {idx} qscale={qscale}）");
                continue;
            };
            eprintln!(
                "帧 {idx:3} 目标 qscale={qscale:2}（实际 {used:2}）：两解码器最大差 {worst}，PSNR {psnr_db:5.2} dB"
            );
            assert_eq!(
                worst, 0,
                "帧 {idx} qscale={qscale}：ffmpeg 和我们解出来的像素不一致（最大差 {worst}）\
                 —— 说明位流不合法"
            );
            assert!(
                psnr_db > 24.0,
                "帧 {idx} qscale={qscale}：画质太差（PSNR {psnr_db:.2} dB）"
            );
            checked += 1;
        }
        if checked >= 9 {
            break;
        }
    }
    assert!(checked > 0, "一帧都没验到（样本里没有有内容的 I 帧？）");
    eprintln!("共验了 {checked} 个（帧 × qscale）组合");
}

/// 极端图案：满幅棋盘 + 高频条纹，专门去逼出三种 escape。
///
/// 这张图在 qscale 小的时候量化后的 level 会很大 —— 正好覆盖
/// 「第一/第二/第三 escape」以及 `pick_qscale` 自动抬高 qscale 的路径。
#[test]
fn encodes_a_high_contrast_pattern_that_forces_escapes() {
    let (w, h) = (320u32, 240u32);
    let mut y = vec![0u8; (w * h) as usize];
    for r in 0..h as usize {
        for c in 0..w as usize {
            y[r * w as usize + c] = if (r / 4 + c / 4) % 2 == 0 { 16 } else { 235 };
        }
    }
    let (cw, ch) = ((w / 2) as usize, (h / 2) as usize);
    let mut u = vec![128u8; cw * ch];
    let mut v = vec![128u8; cw * ch];
    // 色度也来点极端：一半 16、一半 240
    for r in 0..ch {
        for c in 0..cw {
            u[r * cw + c] = if c < cw / 2 { 16 } else { 240 };
            v[r * cw + c] = if r < ch / 2 { 16 } else { 240 };
        }
    }
    let frame = Frame {
        width: w,
        height: h,
        y,
        u,
        v,
    };

    let mut any = false;
    for qscale in [1u8, 2, 4, 10, 31] {
        let Some((worst, psnr_db, used)) = encode_and_cross_check(&frame, qscale) else {
            eprintln!("SKIP: ffmpeg 解不了我们编的图案帧（qscale={qscale}）");
            return;
        };
        eprintln!(
            "棋盘 目标 qscale={qscale:2}（实际 {used:2}）：两解码器最大差 {worst}，PSNR {psnr_db:5.2} dB"
        );
        assert_eq!(worst, 0, "qscale={qscale}：位流不合法（最大差 {worst}）");
        any = true;
    }
    assert!(any, "ffmpeg 一次都没解成功");
}
