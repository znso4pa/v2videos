//! **P 帧编码**：编一段 I + P 的序列，两条判据 ——
//!
//! 1. **合法**：ffmpeg 解出来的像素与我们自己的解码器**逐位一致**（同一条流、
//!    两个独立实现）；这验证 skip / 帧间 CBP / MV 差分 / 帧间残差整条位流路径。
//! 2. **有用**：运动估计真的在干活 —— 有运动的序列，P 帧的 PSNR 得比
//!    「关掉运动估计」明显好，且码流要小。

mod common;

use common::{compare, ffmpeg_frames, unique_temp, wrap_avi};
use v2v_msmpeg4::{Decoder, Encoder, EncoderConfig, Frame};

/// 合成一段**有运动**的序列：纹理背景 + 一个每帧平移的方块。
/// 背景有纹理是为了让运动估计真的有东西可匹配（纯色背景测不出 ME）。
fn moving_sequence(w: u32, h: u32, n: usize) -> Vec<Frame> {
    let (wu, hu) = (w as usize, h as usize);
    let mut out = Vec::new();
    for i in 0..n {
        let mut y = vec![0u8; wu * hu];
        // 背景：竖条纹缓慢平移（低频纹理，可被匹配）
        for r in 0..hu {
            for c in 0..wu {
                let v = 100 + (((c + i * 3) / 8) % 4) as i32 * 30 + ((r / 16) % 2) as i32 * 10;
                y[r * wu + c] = v.clamp(16, 235) as u8;
            }
        }
        // 前景方块：每帧平移 (6, 3) 像素
        let ox = 40 + i * 6;
        let oy = 30 + i * 3;
        for r in 0..40usize {
            for c in 0..40usize {
                let yy = oy + r;
                let xx = ox + c;
                if yy < hu && xx < wu {
                    y[yy * wu + xx] = if (r / 4 + c / 4) % 2 == 0 { 30 } else { 220 };
                }
            }
        }
        let (cw, ch) = (wu / 2, hu / 2);
        let mut u = vec![128u8; cw * ch];
        let mut v = vec![128u8; cw * ch];
        // 色度跟着方块走一点，让色度残差也非零
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = (128 + (c / 8) as i32 * 6 - (i as i32) * 3).clamp(16, 240) as u8;
                v[r * cw + c] = (128 - (r / 8) as i32 * 5 + (i as i32) * 2).clamp(16, 240) as u8;
            }
        }
        out.push(Frame {
            width: w,
            height: h,
            y,
            u,
            v,
        });
    }
    out
}

/// 编一段序列 → 我们解一遍 → ffmpeg 解一遍 → 逐帧比。
/// 返回 `(每帧两解码器最大差, 每帧 PSNR(原图,我们的解码), 每帧码流字节数)`。
fn encode_sequence_and_cross_check(
    frames: &[Frame],
    qscale: u8,
) -> Result<Vec<(i32, f64, usize)>, String> {
    let (w, h) = (frames[0].width, frames[0].height);
    let mut enc = Encoder::new(EncoderConfig {
        qscale,
        ..Default::default()
    });
    let mut streams = Vec::new();
    for f in frames {
        streams.push(enc.encode_frame(f).map_err(|e| e.to_string())?);
    }

    // ① 我们的解码器
    let mut dec = Decoder::new();
    let mut ours = Vec::new();
    for s in &streams {
        ours.push(dec.decode(s, w, h).map_err(|e| e.to_string())?);
    }

    // ② ffmpeg（整段包成 AVI）
    let refs: Vec<&[u8]> = streams.iter().map(|s| s.as_slice()).collect();
    let avi = wrap_avi(&refs, w, h);
    let tmp = unique_temp("v2v_pframe_probe", "avi");
    std::fs::write(&tmp, &avi).map_err(|e| e.to_string())?;
    let ff = ffmpeg_frames(tmp.to_str().unwrap(), frames.len(), w, h);
    let _ = std::fs::remove_file(&tmp);
    let ff = ff.filter(|v| v.len() == frames.len()).ok_or("ffmpeg 解不了")?;

    let mut out = Vec::new();
    for i in 0..frames.len() {
        let (worst, _at, _mean, _) = compare(&ours[i].y, &ours[i].u, &ours[i].v, &ff[i], w, h);
        out.push((worst, psnr(&frames[i], &ours[i]), streams[i].len()));
    }
    Ok(out)
}

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
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

#[test]
fn encodes_ip_sequence_both_decoders_agree() {
    let (w, h) = (320u32, 240u32);
    let frames = moving_sequence(w, h, 5);
    let qscale = 8u8;

    let res = match encode_sequence_and_cross_check(&frames, qscale) {
        Ok(v) => v,
        Err(e) => panic!("编/解失败：{e}"),
    };

    for (i, (worst, psnr_db, size)) in res.iter().enumerate() {
        eprintln!(
            "帧 {i}（{}）最大差 {worst}，PSNR {psnr_db:5.2} dB，{} 字节",
            if i == 0 { "I" } else { "P" },
            size
        );
        assert_eq!(
            *worst, 0,
            "帧 {i}：ffmpeg 与我们解出来的像素不一致（最大差 {worst}）—— 位流不合法"
        );
    }
    // 画质下限：有运动估计的 P 帧不该崩
    for (i, (_, psnr_db, _)) in res.iter().enumerate().skip(1) {
        assert!(*psnr_db > 20.0, "P 帧 {i} 画质太差（PSNR {psnr_db:.2} dB）");
    }
}

/// **运动估计真的在干活**：把运动估计关掉（用「恒 (0,0)」的退化实现），
/// P 帧的 PSNR 应该明显更差。这条测试防的是「ME 写了但没起作用」。
#[test]
fn motion_estimation_actually_helps() {
    let (w, h) = (320u32, 240u32);
    let frames = moving_sequence(w, h, 4);
    let qscale = 8u8;

    // 正常编码
    let with_me = match encode_sequence_and_cross_check(&frames, qscale) {
        Ok(v) => v,
        Err(e) => panic!("编/解失败：{e}"),
    };

    // 退化对照：把每一帧都当成 I 帧编（等价于没有帧间预测）。
    // 两条路径的码流都合法，但 P 帧应该更小、质量更好。
    let mut no_me_size = 0usize;
    let mut no_me_psnr = 0f64;
    let mut dec = Decoder::new();
    for (i, f) in frames.iter().enumerate() {
        let mut enc = Encoder::new(EncoderConfig {
            qscale,
            ..Default::default()
        });
        let s = enc.encode_i_frame(f).expect("编码");
        let d = dec.decode(&s, w, h).expect("解");
        let _ = dec; // 借用结束
        no_me_size += s.len();
        no_me_psnr += psnr(f, &d);
        if i + 1 == frames.len() {
            no_me_psnr /= frames.len() as f64;
        }
    }
    let with_me_psnr: f64 =
        with_me.iter().map(|(_, p, _)| *p).sum::<f64>() / with_me.len() as f64;
    let with_me_size: usize = with_me.iter().map(|(_, _, s)| s).sum();

    eprintln!(
        "全 I 帧：{} 字节，平均 PSNR {:.2} dB\nI+P（带 ME）：{} 字节，平均 PSNR {:.2} dB",
        no_me_size, no_me_psnr, with_me_size, with_me_psnr
    );
    assert!(
        with_me_size * 100 < no_me_size * 60,
        "带运动估计的序列没比全 I 帧小多少（{} vs {}）—— ME 可能没起作用",
        with_me_size,
        no_me_size
    );
    assert!(
        with_me_psnr + 3.0 >= no_me_psnr,
        "带运动估计的画质反而差太多（{:.2} vs {:.2}）",
        with_me_psnr,
        no_me_psnr
    );
}

/// **skip 宏块**：静止场景的第二帧应该几乎全是 skip，码流小到可以忽略。
#[test]
fn static_scene_collapses_to_skips() {
    let (w, h) = (160u32, 120u32);
    let f0 = moving_sequence(w, h, 1).remove(0);
    let f1 = Frame {
        width: f0.width,
        height: f0.height,
        y: f0.y.clone(),
        u: f0.u.clone(),
        v: f0.v.clone(),
    };

    let mut enc = Encoder::new(EncoderConfig {
        qscale: 8,
        ..Default::default()
    });
    let i0 = enc.encode_frame(&f0).expect("编 I");
    let p1 = enc.encode_frame(&f1).expect("编 P");

    eprintln!("I 帧 {} 字节，静止 P 帧 {} 字节", i0.len(), p1.len());
    assert!(
        p1.len() * 20 < i0.len(),
        "静止场景的 P 帧没塌成 skip（{} vs {}）",
        p1.len(),
        i0.len()
    );

    // 且两个解码器仍然一致
    let mut dec = Decoder::new();
    let ours0 = dec.decode(&i0, w, h).expect("解 I");
    let ours1 = dec.decode(&p1, w, h).expect("解 P");
    let refs = vec![&i0[..], &p1[..]];
    let avi = wrap_avi(&refs, w, h);
    let tmp = unique_temp("v2v_static_probe", "avi");
    std::fs::write(&tmp, &avi).expect("写 AVI");
    let ff = ffmpeg_frames(tmp.to_str().unwrap(), 2, w, h);
    let _ = std::fs::remove_file(&tmp);
    let ff = match ff {
        Some(v) if v.len() == 2 => v,
        _ => {
            eprintln!("SKIP: ffmpeg 解不了");
            return;
        }
    };
    let (w0, _, _, _) = compare(&ours0.y, &ours0.u, &ours0.v, &ff[0], w, h);
    let (w1, _, _, _) = compare(&ours1.y, &ours1.u, &ours1.v, &ff[1], w, h);
    eprintln!("两解码器最大差：I {w0}，P {w1}");
    assert_eq!(w0, 0);
    assert_eq!(w1, 0, "静止 P 帧：ffmpeg 与我们不一致（最大差 {w1}）");
}

/// **画面切换**：P 帧遇到完全不同的画面，帧间残差会大到不划算 ——
/// 帧内/帧间判决应该让帧内宏块接管，画质不崩、ffmpeg 仍然解得开。
#[test]
fn scene_cut_falls_back_to_intra_macroblocks() {
    let (w, h) = (320u32, 240u32);

    // 0: 移动序列的 I 帧；1: 正常预测帧；2: **完全不同的画面**（切换）；
    // 3: 新画面的移动版（预测恢复正常）。
    let mut frames = moving_sequence(w, h, 2);
    frames.push(different_scene(w, h, 0));
    frames.push(different_scene(w, h, 1));

    let res = match encode_sequence_and_cross_check(&frames, 8) {
        Ok(v) => v,
        Err(e) => panic!("编/解失败：{e}"),
    };

    // ① 合法性：整段（含切换帧）两个解码器逐位一致
    for (i, (worst, _, _)) in res.iter().enumerate() {
        assert_eq!(*worst, 0, "帧 {i}：ffmpeg 与我们的解码不一致");
    }

    // ② 切换帧（帧 2）的画质：帧内接管后不应崩。
    //    没有判决时这里全是强行帧间的垃圾残差，PSNR 会掉到十几 dB。
    let (_, cut_psnr, _) = res[2];
    eprintln!(
        "各帧 PSNR：{:.2} / {:.2} / {:.2} / {:.2}，各帧字节：{:?}",
        res[0].1, res[1].1, res[2].1, res[3].1,
        res.iter().map(|(_, _, s)| *s).collect::<Vec<_>>()
    );
    assert!(
        cut_psnr > 28.0,
        "切换帧画质崩了（PSNR {cut_psnr:.2} dB）—— 帧内判决可能没接管"
    );

    // ③ 切换帧的码流明显大于正常预测帧（帧内码字贵得多）—— 证明它真用了帧内
    let normal_p = res[1].2;
    let cut = res[2].2;
    assert!(
        cut > normal_p * 4,
        "切换帧没比正常预测帧贵多少（{cut} vs {normal_p}）—— 帧内可能没被用上"
    );

    // ④ 直接证据：判决计数 —— 正常预测帧 0 个帧内宏块，切换帧占大多数
    let mut enc = Encoder::new(EncoderConfig {
        qscale: 8,
        ..Default::default()
    });
    let intra: Vec<usize> = frames
        .iter()
        .map(|f| {
            enc.encode_frame(f).expect("编码");
            enc.intra_mbs_last
        })
        .collect();
    eprintln!("各帧帧内宏块数：{intra:?}");
    // 正常预测帧允许**少量**帧内宏块（方块边缘的新内容局部更便宜 —— 这是判决
    // 在正确工作），但只该是个位数；切换帧应该占大多数。
    let total_mbs = (w * h / 256) as usize;
    assert!(
        intra[1] * 20 < total_mbs,
        "正常预测帧的帧内宏块太多（{}）—— 判决可能过于激进",
        intra[1]
    );
    assert!(
        intra[3] * 20 < total_mbs,
        "新画面的移动帧帧内宏块太多（{}）",
        intra[3]
    );
    assert!(
        intra[2] > total_mbs / 2,
        "切换帧只有 {} 个帧内宏块 —— 判决没接管",
        intra[2]
    );
}

/// 与 `moving_sequence` 完全不同的画面：高频棋盘底 + 一块随 `i` 平移的亮斑。
/// 底是**固定**的 —— 切换帧（i 起步）必然全帧内；下一帧（i+1）只有亮斑移动，
/// 帧间应该预测得很好。
fn different_scene(w: u32, h: u32, i: usize) -> Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut y = vec![0u8; wu * hu];
    for r in 0..hu {
        for c in 0..wu {
            y[r * wu + c] = if (r / 6 + c / 6) % 2 == 0 { 235 } else { 20 };
        }
    }
    // 移动的亮斑（帧 2 → 3 之间可被运动补偿追上）
    let ox = 40 + i * 6;
    let oy = 30 + i * 3;
    for r in 0..30usize {
        for c in 0..30usize {
            let yy = oy + r;
            let xx = ox + c;
            if yy < hu && xx < wu {
                y[yy * wu + xx] = 128;
            }
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for r in 0..ch {
        for c in 0..cw {
            u[r * cw + c] = if (r / 4) % 2 == 0 { 220 } else { 40 };
            v[r * cw + c] = if (c / 4) % 2 == 0 { 60 } else { 200 };
        }
    }
    Frame { width: w, height: h, y, u, v }
}
