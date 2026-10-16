//! H.263 / MPEG-4 的**正**离散余弦变换。
//!
//! 定标必须和 `idct.rs` 的 `simple_idct` 严格互逆：`simple_idct` 实现的是标准 IDCT
//! `f(x,y) = 1/4 ΣΣ C(u)C(v)·F(u,v)·cos((2x+1)uπ/16)·cos((2y+1)vπ/16)`
//! （`F(0,0) = 8` → 输出像素 1），所以正向变换取它的严格逆：
//! `F(u,v) = 1/4·C(u)C(v)·ΣΣ f(x,y)·cos((2x+1)uπ/16)·cos((2y+1)vπ/16)`。
//!
//! 用浮点算：编码器里**质量优先于逐位复现**（只有解码那边要求逐位一致）。
//! 日后换整数 `fdct`，只要保证同一套定标即可 —— `round_trips_through_our_idct`
//! 这条测试会把定标钉住。

use std::f64::consts::PI;

/// `COS[u][x] = cos((2x+1)uπ/16)`。
///
/// 表与输入块无关，缓存以避免在块循环中重复构造。
/// 保留原来的浮点表达式和运算顺序，不改变变换的定标与舍入。
fn cos_table() -> &'static [[f64; 8]; 8] {
    static T: std::sync::OnceLock<[[f64; 8]; 8]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0f64; 8]; 8];
        for (u, row) in t.iter_mut().enumerate() {
            for (x, v) in row.iter_mut().enumerate() {
                *v = ((2 * x + 1) as f64 * u as f64 * PI / 16.0).cos();
            }
        }
        t
    })
}

/// 正向 8×8 DCT。`pix` 是 8×8 像素块（`stride` = 一行字节数），
/// 结果写进 `out`（自然序 `out[v*8+u]`：`u` 水平频率、`v` 垂直频率）。
pub fn fdct(pix: &[u8], stride: usize, out: &mut [i32; 64]) {
    let mut tmp = [0i32; 64];
    for y in 0..8 {
        for x in 0..8 {
            tmp[y * 8 + x] = pix[y * stride + x] as i32;
        }
    }
    fdct_i(&tmp, 8, out)
}

/// 同 `fdct`，但输入是 **i32**（编码 P 帧时要对**残差**做正向变换，残差可负）。
pub fn fdct_i(pix: &[i32], stride: usize, out: &mut [i32; 64]) {
    // 走蝶形版（LL&M）。f64 那份留着当参照：`fdct_i_float`，只在测试里对拍。
    // 两者逐系数最大差 1、98.6% 完全相同（见 `llm_matches_float`）。
    fdct_llm(pix, stride, out);
}

/// **参照实现**：原来的 f64 分离式 DCT。只在测试里与蝶形版对拍。
#[allow(dead_code)]
pub fn fdct_i_float(pix: &[i32], stride: usize, out: &mut [i32; 64]) {
    let c = cos_table();
    let mut tmp = [[0f64; 8]; 8];

    // 行：G[y][u] = Σ_x f[y][x]·cos((2x+1)uπ/16)
    for y in 0..8 {
        for u in 0..8 {
            let mut s = 0.0;
            for x in 0..8 {
                s += pix[y * stride + x] as f64 * c[u][x];
            }
            tmp[y][u] = s;
        }
    }

    // 列 + 归一化：F[v][u] = 1/4·C(u)C(v)·Σ_y G[y][u]·cos((2y+1)vπ/16)
    let inv_sqrt2 = 1.0 / 2f64.sqrt();
    for u in 0..8 {
        let cu = if u == 0 { inv_sqrt2 } else { 1.0 };
        for v in 0..8 {
            let cv = if v == 0 { inv_sqrt2 } else { 1.0 };
            let mut s = 0.0;
            for y in 0..8 {
                s += tmp[y][u] * c[v][y];
            }
            out[v * 8 + u] = (s * cu * cv / 4.0).round() as i32;
        }
    }
}


// ───────────────────────── 蝶形版（LL&M，定标均匀）─────────────────────────
//
// 为什么是**这一份**而不是 ffmpeg 现在用的 `ff_fdct_ifast`：那个是 **AAN** 分解，
// 输出**逐系数定标不同**（DC 2×、u=1 3.9×…），是给「量化表吸收定标」的 JPEG 用的 ——
// 跟本仓 `simple_idct` 的标准定标对不上，直接搬过来会解出噪声。
// LL&M（Loeffler-Ligtenberg-Moschytz）是**整体乘一个常数**（这里 ×8），所以只要收尾
// 多移 3 位就与 f64 版同一套定标。
//
// 参考：ffmpeg n4.4 的 `libavcodec/jfdctint_template.c`（`ff_jpeg_fdct_islow`）——
// JPEG 参考实现里的 LL&M，注释明说「results scaled up by an overall factor of 8」。

const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 4;
/// 列趟的输出移位。上游是 `PASS1_BITS`；我们**再加 3 位**把「×8」收回去。
const OUT_SHIFT: u32 = PASS1_BITS + 3;

const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

#[inline(always)]
fn descale(x: i32, n: u32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// 一维蝶形（行趟）。下标 0/4 带 `PASS1_BITS` 的放大，其余不带。
#[inline(always)]
fn butterfly_row(d: [i32; 8]) -> [i32; 8] {
    let (t0, t7) = (d[0] + d[7], d[0] - d[7]);
    let (t1, t6) = (d[1] + d[6], d[1] - d[6]);
    let (t2, t5) = (d[2] + d[5], d[2] - d[5]);
    let (t3, t4) = (d[3] + d[4], d[3] - d[4]);
    let t10 = t0 + t3;
    let t13 = t0 - t3;
    let t11 = t1 + t2;
    let t12 = t1 - t2;
    let mut o = [0i32; 8];
    o[0] = (t10 + t11) << PASS1_BITS;
    o[4] = (t10 - t11) << PASS1_BITS;
    let z1 = (t12 + t13) * FIX_0_541196100;
    o[2] = descale(z1 + t13 * FIX_0_765366865, CONST_BITS - PASS1_BITS);
    o[6] = descale(z1 + t12 * -FIX_1_847759065, CONST_BITS - PASS1_BITS);
    let z1 = t4 + t7;
    let z2 = t5 + t6;
    let z3 = t4 + t6;
    let z4 = t5 + t7;
    let z5 = (z3 + z4) * FIX_1_175875602;
    let m4 = t4 * FIX_0_298631336;
    let m5 = t5 * FIX_2_053119869;
    let m6 = t6 * FIX_3_072711026;
    let m7 = t7 * FIX_1_501321110;
    let z1 = z1 * -FIX_0_899976223;
    let z2 = z2 * -FIX_2_562915447;
    let z3 = z3 * -FIX_1_961570560 + z5;
    let z4 = z4 * -FIX_0_390180644 + z5;
    o[7] = descale(m4 + z1 + z3, CONST_BITS - PASS1_BITS);
    o[5] = descale(m5 + z2 + z4, CONST_BITS - PASS1_BITS);
    o[3] = descale(m6 + z2 + z3, CONST_BITS - PASS1_BITS);
    o[1] = descale(m7 + z1 + z4, CONST_BITS - PASS1_BITS);
    o
}

/// 列趟：同一套蝶形，但**去掉 `PASS1_BITS` 的放大**（`OUT_SHIFT` 里已经含了那 3 位）。
#[inline(always)]
fn butterfly_col(d: [i32; 8]) -> [i32; 8] {
    let (t0, t7) = (d[0] + d[7], d[0] - d[7]);
    let (t1, t6) = (d[1] + d[6], d[1] - d[6]);
    let (t2, t5) = (d[2] + d[5], d[2] - d[5]);
    let (t3, t4) = (d[3] + d[4], d[3] - d[4]);
    let t10 = t0 + t3;
    let t13 = t0 - t3;
    let t11 = t1 + t2;
    let t12 = t1 - t2;
    let mut o = [0i32; 8];
    o[0] = descale(t10 + t11, OUT_SHIFT);
    o[4] = descale(t10 - t11, OUT_SHIFT);
    let z1 = (t12 + t13) * FIX_0_541196100;
    o[2] = descale(z1 + t13 * FIX_0_765366865, CONST_BITS + OUT_SHIFT);
    o[6] = descale(z1 + t12 * -FIX_1_847759065, CONST_BITS + OUT_SHIFT);
    let z1 = t4 + t7;
    let z2 = t5 + t6;
    let z3 = t4 + t6;
    let z4 = t5 + t7;
    let z5 = (z3 + z4) * FIX_1_175875602;
    let m4 = t4 * FIX_0_298631336;
    let m5 = t5 * FIX_2_053119869;
    let m6 = t6 * FIX_3_072711026;
    let m7 = t7 * FIX_1_501321110;
    let z1 = z1 * -FIX_0_899976223;
    let z2 = z2 * -FIX_2_562915447;
    let z3 = z3 * -FIX_1_961570560 + z5;
    let z4 = z4 * -FIX_0_390180644 + z5;
    o[7] = descale(m4 + z1 + z3, CONST_BITS + OUT_SHIFT);
    o[5] = descale(m5 + z2 + z4, CONST_BITS + OUT_SHIFT);
    o[3] = descale(m6 + z2 + z3, CONST_BITS + OUT_SHIFT);
    o[1] = descale(m7 + z1 + z4, CONST_BITS + OUT_SHIFT);
    o
}

/// **蝶形版正变换** —— 定标与 `fdct_i`（f64 版）完全相同（`F(0,0) = 8 × 均值`）。
pub fn fdct_llm(pix: &[i32], stride: usize, out: &mut [i32; 64]) {
    let mut tmp = [[0i32; 8]; 8];
    for y in 0..8 {
        let mut row = [0i32; 8];
        for x in 0..8 {
            row[x] = pix[y * stride + x];
        }
        tmp[y] = butterfly_row(row);
    }
    for u in 0..8 {
        let col = [
            tmp[0][u], tmp[1][u], tmp[2][u], tmp[3][u], tmp[4][u], tmp[5][u], tmp[6][u], tmp[7][u],
        ];
        let c = butterfly_col(col);
        for v in 0..8 {
            out[v * 8 + u] = c[v];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 蝶形版必须与 f64 版**同一套定标**，且逐系数只差个位数。
    #[test]
    fn llm_matches_float() {
        let mut c = [0i32; 64];
        for x in c.iter_mut() {
            *x = 100;
        }
        let (mut a, mut b) = ([0i32; 64], [0i32; 64]);
        fdct_i_float(&c, 8, &mut a);
        fdct_llm(&c, 8, &mut b);
        assert_eq!(a[0], 800, "f64 的定标：常数块 100 → DC 800");
        assert_eq!(b[0], 800, "蝶形版必须一样");
        let mut v = [0i32; 64];
        let mut st = 99991u64;
        let mut rnd = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let (mut worst, mut same, mut total) = (0i32, 0usize, 0usize);
        for _ in 0..2000 {
            for x in v.iter_mut() {
                *x = (rnd() % 511) as i32 - 255;
            }
            fdct_i_float(&v, 8, &mut a);
            fdct_llm(&v, 8, &mut b);
            for k in 0..64 {
                worst = worst.max((a[k] - b[k]).abs());
                same += (a[k] == b[k]) as usize;
                total += 1;
            }
        }
        eprintln!(
            "蝶形 vs f64：逐系数最大差 {worst}，完全相同 {:.1}%",
            100.0 * same as f64 / total as f64
        );
        assert!(worst <= 3, "蝶形 DCT 与 f64 差得太多：{worst}");
    }
    use crate::idct::idct_put;

    /// 常数块：`F(0,0)` 必须是 `8 × 像素值`（这正是解码端 `DC=8 → 像素 1` 的定标）。
    #[test]
    fn dc_of_a_flat_block_is_eight_times_the_level() {
        for v in [0u8, 1, 16, 100, 235, 255] {
            let pix = [v; 64];
            let mut out = [0i32; 64];
            fdct(&pix, 8, &mut out);
            assert_eq!(out[0], 8 * v as i32, "像素 {v} 的 DC 应当是 8×{v}");
            // 常数块的 AC 必须全 0
            assert!(out[1..].iter().all(|&x| x == 0), "常数块的 AC 应当全 0");
        }
    }

    /// 正向 + 我们自己的反向（无损路径：不量化）应当还原出原像素。
    #[test]
    fn round_trips_through_our_idct() {
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) & 0xff) as u8
        };
        for _ in 0..300 {
            let pix: Vec<u8> = (0..64).map(|_| next()).collect();
            let mut coeff = [0i32; 64];
            fdct(&pix, 8, &mut coeff);

            let mut block = [0i16; 64];
            for i in 0..64 {
                block[i] = coeff[i].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
            let mut got = vec![0u8; 64];
            idct_put(&mut block, &mut got, 8);

            let worst = pix
                .iter()
                .zip(got.iter())
                .map(|(&a, &b)| (a as i32 - b as i32).abs())
                .max()
                .unwrap();
            // 整数 IDCT 的舍入 + 16 位截断，允许 ±1
            assert!(worst <= 1, "正向+反向没能还原：最大差 {worst}");
        }
    }
}
