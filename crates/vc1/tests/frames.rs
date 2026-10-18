//! 整帧对账：**真样本 + `ffmpeg` 的 rawvideo**，逐帧逐像素比。
//!
//! 判据是「**逐像素 0 差**」而不是「没报错」：解错一位、少滤一条边、取整偏半个像素，
//! 在肉眼下都「差不多」，但在这条判据下立刻现形。`ffmpeg` 当 oracle，
//! 它吐 `yuv420p` 裸帧到 stdout，我们边解边比。
//!
//! 帧数由 `V2V_MAX_FRAMES` 限（不设就全跑）；`V2V_FRAME_LOG=1` 会把每帧的
//! 「用掉多少位 / 差多少个像素」打出来。
//!
//! ⚠️ 样本一律走环境变量（`V2V_ASF_SAMPLE`），缺了就 `SKIP` 并通过 —— 仓规如此。

mod common;

use std::io::{BufReader, Read};
use std::process::{Command, Stdio};

use v2v_vc1::{Decoder, PictType, SeqHeader};

/// 从 `ffmpeg` 的裸帧流里读满一个 `yuv420p` 帧（读完返回 false = 到头了）。
fn read_frame(r: &mut impl Read, buf: &mut [u8]) -> bool {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => return false,
            Ok(n) => got += n,
            Err(_) => return false,
        }
    }
    true
}

/// 逐帧比「我们解出来的平面」与「ffmpeg 吐的裸帧」。
///
/// 我们的平面是**宏块对齐**的（宽 = mb_width*16，高 = mb_height*16），
/// ffmpeg 吐的是**裁剪到显示尺寸**的，所以按行比可见部分。
fn compare(
    ours: &v2v_msmpeg4::Planes,
    theirs: &[u8],
    w: usize,
    h: usize,
    mb_w: usize,
    mb_h: usize,
) -> (usize, usize, i32) {
    let (mut diff, mut first, mut maxd) = (0usize, usize::MAX, 0i32);
    let pw = mb_w * 16;
    for r in 0..h {
        for c in 0..w {
            let a = ours.y[r * pw + c] as i32;
            let b = theirs[r * w + c] as i32;
            if a != b {
                diff += 1;
                if first == usize::MAX {
                    first = r * w + c;
                }
                maxd = maxd.max((a - b).abs());
            }
        }
    }
    let cw = w / 2;
    let cw_full = pw / 2;
    let ch = h / 2;
    for r in 0..ch {
        for c in 0..cw {
            for (i, plane) in [&ours.u, &ours.v].iter().enumerate() {
                let a = plane[r * cw_full + c] as i32;
                let b = theirs[(w * h) + i * cw * ch + r * cw + c] as i32;
                if a != b {
                    diff += 1;
                    if first == usize::MAX {
                        first = w * h + i * cw * ch + r * cw + c;
                    }
                    maxd = maxd.max((a - b).abs());
                }
            }
        }
    }
    let _ = (mb_h, mb_w);
    (diff, first, maxd)
}

/// 按**宏块**（而不是像素）找第一处不一致 —— 宏块层一错就整块错，
/// 而环路滤波只改边上的两列/两行，两者要分开看。
fn first_mismatching_mb(
    ours: &v2v_msmpeg4::Planes,
    theirs: &[u8],
    w: usize,
    h: usize,
    mb_w: usize,
    mb_h: usize,
) -> Option<(usize, usize)> {
    let pw = mb_w * 16;
    for my in 0..mb_h {
        for mx in 0..mb_w {
            let (x0, y0) = (mx * 16, my * 16);
            let mut bad = false;
            for r in y0..(y0 + 16).min(h) {
                for c in x0..(x0 + 16).min(w) {
                    if ours.y[r * pw + c] != theirs[r * w + c] {
                        bad = true;
                        break;
                    }
                }
                if bad {
                    break;
                }
            }
            if bad {
                return Some((mx, my));
            }
        }
    }
    None
}

#[test]
fn frames_match_ffmpeg_pixel_exact() {
    let samples = common::sample_list();
    if samples.is_empty() {
        eprintln!("SKIP: 没设 V2V_ASF_SAMPLE");
        return;
    }
    if !common::has_ffmpeg() {
        eprintln!("SKIP: 没装 ffmpeg");
        return;
    }
    let limit: usize = std::env::var("V2V_MAX_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let verbose = std::env::var("V2V_FRAME_LOG").is_ok();

    let mut checked = 0usize;
    for path in &samples {
        let Some(track) = common::video_track(path) else {
            eprintln!("SKIP {path}: 取不到视频轨");
            continue;
        };
        if track.fourcc != common::FOURCC_WMV3 {
            eprintln!("SKIP {path}: 不是 WMV3（fourcc {:#x}）", track.fourcc);
            continue;
        }
        let seq = SeqHeader::parse(&track.extra)
            .unwrap_or_else(|e| panic!("{path}: 序列头解析失败: {e}"));
        let w = track.encoded_width as usize;
        let h = track.encoded_height as usize;
        let (mb_w, mb_h) = (w.div_ceil(16), h.div_ceil(16));
        let mut dec = Decoder::new(seq, track.encoded_width, track.encoded_height)
            .unwrap_or_else(|e| panic!("{path}: 解码器建不起来: {e}"));

        let frames = common::all_video_frames(path, track.number)
            .unwrap_or_else(|| panic!("{path}: 取不到视频帧"));

        // ffmpeg 的裸帧（stdout，边解边读，不落盘）。
        let mut child = Command::new("ffmpeg")
            .args(["-loglevel", "error", "-nostdin", "-i", path])
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("{path}: 起不了 ffmpeg: {e}"));
        let mut out = BufReader::new(child.stdout.take().unwrap());
        let mut raw = vec![0u8; w * h + 2 * (w / 2) * (h / 2)];

        let mut n = 0usize;
        let mut bad = 0usize;
        let mut overrun = 0usize;
        let mut tail_bits: Vec<usize> = Vec::new();
        while read_frame(&mut out, &mut raw) {
            if n >= frames.len() {
                eprintln!("{path}: ffmpeg 多吐了一帧（我们只有 {} 帧）", frames.len());
                break;
            }
            let bits = match dec.decode_frame(&frames[n]) {
                Ok(b) => b,
                Err(e) => {
                    // 出错时把「已经解出来的部分」与 ffmpeg 的那一帧比一遍：
                    // **第一处不一致的宏块**就是位流开始跑偏的地方（宏块层错了
                    // 只影响它自己和后面的，前面的像素仍然是对的）。
                    let (diff, first, _) = compare(dec.partial_planes(), &raw, w, h, mb_w, mb_h);
                    let (fx, fy) = (first % w, first / w);
                    let mb = first_mismatching_mb(dec.partial_planes(), &raw, w, h, mb_w, mb_h);
                    panic!(
                        "{path}: 第 {n} 帧解码失败（最后解到宏块 {:?}，位游标 {}）: {e}\n\
                         已解出的部分里有 {diff} 个像素与 ffmpeg 不同；首个不一致像素在 ({fx},{fy})，\
                         首个不一致的宏块是 {mb:?}",
                        dec.last_mb(),
                        dec.last_mb_bits()
                    )
                }
            };
            let avail = frames[n].len() * 8;
            if n < 16 {
                eprintln!("[diag] 第 {n} 帧：{} 字节", frames[n].len());
            }
            if bits > avail {
                overrun += 1;
            }
            tail_bits.push(avail.saturating_sub(bits));
            let (diff, first, maxd) = compare(dec.planes(), &raw, w, h, mb_w, mb_h);
            if diff != 0 {
                bad += 1;
                if bad <= 3 || verbose {
                    let (fy, fx) = (first / w, first % w);
                    // 按**宏块**报第一处不一致：宏块层错整块错，环路滤波只改边上两列/两行 ——
                    // 两者分开看才知道该往哪条路查。
                    let mb = first_mismatching_mb(dec.planes(), &raw, w, h, mb_w, mb_h);
                    eprintln!(
                        "{path}: 第 {n} 帧不一致：{diff} 个像素不同（最大差 {maxd}，\
                         首个像素在 ({fx}, {fy})，首个不一致宏块 {mb:?}，用掉 {bits}/{avail} 位）"
                    );
                }
            } else if verbose {
                eprintln!(
                    "{path}: 第 {n} 帧一致（用掉 {bits}/{avail} 位，剩 {} 位）",
                    avail.saturating_sub(bits)
                );
            }
            n += 1;
            if limit != 0 && n >= limit {
                break;
            }
        }
        // ⚠️ 我们提前停读之后，ffmpeg 会**堵在管道上**不退出 —— 必须先杀掉再 wait，
        // 否则 `wait()` 永远不返回（第一次跑就是这么挂住的）。
        let _ = child.kill();
        let _ = child.wait();
        if n == 0 {
            panic!("{path}: 一帧都没比过（ffmpeg 没吐裸帧？）");
        }
        let max_tail = tail_bits.iter().copied().max().unwrap_or(0);
        eprintln!(
            "{path}: 比过 {n} 帧，不一致 {bad} 帧，位游标越过帧尾 {overrun} 次，\
             帧尾剩余最多 {max_tail} 位"
        );
        assert_eq!(bad, 0, "{path}: {bad} 帧与 ffmpeg 不一致");
        assert_eq!(overrun, 0, "{path}: {overrun} 帧把位游标读过了帧尾");
        checked += 1;
    }
    assert!(checked > 0, "给了样本但没有一个是 WMV3");
}

/// 只跑 I 帧的语法层判据：**位游标正好落在帧尾**。
///
/// 这一条把「语法解对了」与「像素重建对不对」分开：I 帧的宏块层解完之后，
/// 游标必须停在帧数据的最后几个位里 —— 少读/多读一位都会让它跑偏几十位。
#[test]
fn i_frame_bit_cursor_lands_at_frame_end() {
    let samples = common::sample_list();
    if samples.is_empty() {
        eprintln!("SKIP: 没设 V2V_ASF_SAMPLE");
        return;
    }
    let limit: usize = std::env::var("V2V_MAX_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut checked = 0usize;
    for path in &samples {
        let Some(track) = common::video_track(path) else {
            continue;
        };
        if track.fourcc != common::FOURCC_WMV3 {
            continue;
        }
        let seq = SeqHeader::parse(&track.extra).unwrap();
        let mut dec = Decoder::new(seq, track.encoded_width, track.encoded_height).unwrap();
        let frames = common::all_video_frames(path, track.number).unwrap();
        let mut i_frames = 0usize;
        let mut worst_tail = 0usize;
        for (n, data) in frames.iter().enumerate() {
            let hdr = v2v_vc1::FrameHeader::parse(
                data,
                &seq,
                (track.encoded_width as usize).div_ceil(16),
                (track.encoded_height as usize).div_ceil(16),
                &mut Default::default(),
            );
            let is_i = matches!(hdr, Ok(ref h) if h.pict_type == PictType::I);
            let bits = dec
                .decode_frame(data)
                .unwrap_or_else(|e| panic!("{path}: 第 {n} 帧解码失败: {e}"));
            let avail = data.len() * 8;
            assert!(
                bits <= avail,
                "{path}: 第 {n} 帧位游标越界（{bits} > {avail}）"
            );
            if is_i {
                i_frames += 1;
                worst_tail = worst_tail.max(avail - bits);
            }
            if limit != 0 && n + 1 >= limit {
                break;
            }
        }
        eprintln!(
            "{path}: {i_frames} 个 I 帧，帧尾剩余最多 {worst_tail} 位（越少越紧）"
        );
        assert!(
            worst_tail < 64,
            "{path}: I 帧的位游标离帧尾太远（{worst_tail} 位）—— 宏块层多半少读了东西"
        );
        checked += 1;
    }
    assert!(checked > 0, "给了样本但没有一个是 WMV3");
}
