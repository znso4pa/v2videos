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

#[cfg(test)]
mod tests {
    use super::*;
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
