//! NEON ↔ 标量的**直接对账**：随机缓冲 × 全模式 × 全取整位，逐位必须一致。
//!
//! 这是比「真样本逐像素 0 差」更早的关卡 —— 后者只覆盖这几种模式在实际流里
//! 的取值组合，这里把参数空间（模式 × 取整 × 边界附近的余量行）全部扫一遍。
#![cfg(target_arch = "aarch64")]

use v2v_vc1::neon;
use v2v_vc1::test_support::BUF_STRIDE;

/// 确定性伪随机（可复现）。
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// 标量参照（与 `mc.rs::mspel_filter` 同构 —— 改那边必须同步这边）。
fn mspel_scalar(src: &[u8], off: isize, stride: isize, mode: u32, r: i32) -> i32 {
    let s = |k: isize| i32::from(src[(off + k * stride) as usize]);
    match mode {
        1 => (-4 * s(-1) + 53 * s(0) + 18 * s(1) - 3 * s(2) + 32 - r) >> 6,
        2 => (-1 * s(-1) + 9 * s(0) + 9 * s(1) - 1 * s(2) + 8 - r) >> 4,
        _ => (-3 * s(-1) + 18 * s(0) + 53 * s(1) - 4 * s(2) + 32 - r) >> 6,
    }
}

fn fill(lcg: &mut Lcg, buf: &mut [u8]) {
    for v in buf.iter_mut() {
        // 两头压到极端值：夹取边界最容易出偏差。
        *v = match lcg.next() % 8 {
            0 => 0,
            1 => 255,
            _ => (lcg.next() % 256) as u8,
        };
    }
}

#[test]
fn mspel_vertical_pass_matches_scalar() {
    let mut lcg = Lcg(1);
    let mut src = vec![0u8; 64 * BUF_STRIDE];
    for w in [8usize, 16] {
        let n = w + 3;
        for vmode in 0..4u32 {
            for shift in [1i32, 3, 5] {
                for rnd in [0i32, 1] {
                    fill(&mut lcg, &mut src);
                    for j in 0..w {
                    // `src -= 1`：整体左移一列（与 mc.rs 合流分支一致）。
                    let row0 = 3 * BUF_STRIDE + 3 + j * BUF_STRIDE - 1;
                    let mut got = vec![0i16; n];
                    let r = (1 << (shift - 1)) + rnd - 1;
                    neon::vertical_pass(&src, row0, n, vmode, shift, r, &mut got);
                    for i in 0..n {
                        let o = row0 + i;
                        // 标量在合流路径里存的是「去掉 mspel_filter 的 bias/shift」
                        // 的那套：`(v + r) >> shift`（见 mc.rs 合流分支）。
                        let v = match vmode {
                            1 => -4 * i32::from(src[o])
                                + 53 * i32::from(src[o + BUF_STRIDE])
                                + 18 * i32::from(src[o + 2 * BUF_STRIDE])
                                - 3 * i32::from(src[o + 3 * BUF_STRIDE]),
                            2 => -1 * i32::from(src[o])
                                + 9 * i32::from(src[o + BUF_STRIDE])
                                + 9 * i32::from(src[o + 2 * BUF_STRIDE])
                                - 1 * i32::from(src[o + 3 * BUF_STRIDE]),
                            _ => -3 * i32::from(src[o])
                                + 18 * i32::from(src[o + BUF_STRIDE])
                                + 53 * i32::from(src[o + 2 * BUF_STRIDE])
                                - 4 * i32::from(src[o + 3 * BUF_STRIDE]),
                        };
                        let want = ((v + r) >> shift) as i16;
                        assert_eq!(
                            got[i], want,
                            "vertical w={w} vmode={vmode} shift={shift} rnd={rnd} j={j} i={i}"
                        );
                    }
                }
            }
        }
    }
    }  // for j
    let _ = (&mut lcg, src);
}

#[test]
fn mspel_horizontal_pass_matches_scalar() {
    let mut lcg = Lcg(2);
    for w in [8usize, 16] {
        let row_len = w + 3;
        for hmode in 1..4u32 {
            for rnd in [0i32, 1] {
                let mut tmp = vec![0i16; row_len * w + 8];
                fill(&mut lcg, unsafe {
                    std::slice::from_raw_parts_mut(
                        tmp.as_mut_ptr() as *mut u8,
                        tmp.len() * 2,
                    )
                });
                let mut got = vec![0u8; w];
                let r2 = 64 - rnd;
                for j in 0..w {
                    let t = j * row_len + 1;
                    neon::horizontal_pass(&tmp, t, w, hmode, r2, &mut got, 0);
                    let base = t;
                    for i in 0..w {
                        let o = base + i;
                        let (a, b, c, d) = (
                            i32::from(tmp[o - 1]),
                            i32::from(tmp[o]),
                            i32::from(tmp[o + 1]),
                            i32::from(tmp[o + 2]),
                        );
                        let h = match hmode {
                            1 => -4 * a + 53 * b + 18 * c - 3 * d,
                            2 => -1 * a + 9 * b + 9 * c - 1 * d,
                            _ => -3 * a + 18 * b + 53 * c - 4 * d,
                        };
                        let want = ((h + r2) >> 7).clamp(0, 255) as u8;
                        assert_eq!(
                            got[i], want,
                            "horizontal w={w} hmode={hmode} rnd={rnd} j={j} i={i}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn mspel_single_direction_matches_scalar() {
    let mut lcg = Lcg(3);
    let mut src = vec![0u8; 64 * BUF_STRIDE];
    let row0 = 3 * BUF_STRIDE + 3;
    for w in [8usize, 16] {
        for mode in 1..4u32 {
            for rnd in [0i32, 1] {
                fill(&mut lcg, &mut src);
                // 竖直（r = 1 - rnd）
                let mut got = vec![0u8; w];
                neon::single_vertical(&src, row0, w, mode, 1 - rnd, &mut got, 0);
                for i in 0..w {
                    let off = (row0 + i) as isize;
                    let want =
                        mspel_scalar(&src, off, BUF_STRIDE as isize, mode, 1 - rnd).clamp(0, 255)
                            as u8;
                    assert_eq!(got[i], want, "single_v w={w} mode={mode} rnd={rnd} i={i}");
                }
                // 水平（r = rnd）
                let row = row0;
                let mut goth = vec![0u8; w];
                neon::single_horizontal(&src, row, w, mode, rnd, &mut goth, 0);
                for i in 0..w {
                    let off = (row + i) as isize;
                    let want = mspel_scalar(&src, off, 1, mode, rnd).clamp(0, 255) as u8;
                    assert_eq!(goth[i], want, "single_h w={w} mode={mode} rnd={rnd} i={i}");
                }
            }
        }
    }
}

#[test]
fn chroma_matches_scalar() {
    let mut lcg = Lcg(4);
    let mut src = vec![0u8; 64 * BUF_STRIDE];
    let base = 3 * BUF_STRIDE + 3;
    for x in 0..8i32 {
        for y in 0..8i32 {
            for rnd in [true, false] {
                fill(&mut lcg, &mut src);
                let a = (8 - x) * (8 - y);
                let b = x * (8 - y);
                let c = x * y;
                let d = (8 - x) * y;
                let bias: u32 = if rnd { 28 } else { 32 };
                let mut got = vec![0u8; 64];
                neon::chroma_mc8(&mut got, 8, 0, 0, &src, base, a, b, c, d, bias);
                for j in 0..8usize {
                    for i in 0..8usize {
                        let o = base + j * BUF_STRIDE + i;
                        let s = a * i32::from(src[o])
                            + b * i32::from(src[o + 1])
                            + c * i32::from(src[o + BUF_STRIDE])
                            + d * i32::from(src[o + BUF_STRIDE + 1]);
                        let want = ((s + bias as i32) >> 6) as u8;
                        assert_eq!(got[j * 8 + i], want, "chroma x={x} y={y} rnd={rnd} j={j} i={i}");
                    }
                }
            }
        }
    }
}

#[test]
fn hpel_rows_match_scalar() {
    let mut lcg = Lcg(5);
    let mut src = vec![0u8; 64 * BUF_STRIDE];
    let s = 3 * BUF_STRIDE + 3;
    for w in [8usize, 16, 24] {
        for rnd in [true, false] {
            fill(&mut lcg, &mut src);
            // avg2（横向邻居）
            let mut got = vec![0u8; w];
            neon::hpel_avg2_row(&src, s, &mut got, 0, w, !rnd);
            let r2 = i32::from(!rnd);
            for i in 0..w {
                let want = ((i32::from(src[s + i]) + i32::from(src[s + i + 1]) + r2) >> 1) as u8;
                assert_eq!(got[i], want, "avg2 w={w} rnd={rnd} i={i}");
            }
            // avg4（右下邻居）
            let r4: u32 = if rnd { 1 } else { 2 };
            let mut got4 = vec![0u8; w];
            neon::hpel_avg4_row(&src, s, &mut got4, 0, w, r4);
            for i in 0..w {
                let o = s + i;
                let want = ((i32::from(src[o])
                    + i32::from(src[o + 1])
                    + i32::from(src[o + BUF_STRIDE])
                    + i32::from(src[o + BUF_STRIDE + 1])
                    + r4 as i32)
                    >> 2) as u8;
                assert_eq!(got4[i], want, "avg4 w={w} rnd={rnd} i={i}");
            }
        }
    }
}
