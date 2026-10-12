//! H.263 / MPEG-4 Part 2 的反离散余弦变换（`simple_idct`）。
//!
//! MP43 用的就是这一族：8 位、`int16` 输入，`ROW_SHIFT = 11`、`COL_SHIFT = 20`、
//! `DC_SHIFT = 3`。
//!
//! 两趟：先逐行 1D，再逐列 1D 并写出像素。整数运算 + 定标，输出即最终像素（已 clip 到 0..255）。
//! 定标与真 IDCT 同口径 —— 所以 `F(0,0)=8` 的直流块解出来恒为 `1`。

// 1D 变换的常数：cos(iπ/16)·√2·(1<<14) + 0.5
const W1: u32 = 22725;
const W2: u32 = 21407;
const W3: u32 = 19266;
const W4: u32 = 16383;
const W5: u32 = 12873;
const W6: u32 = 8867;
const W7: u32 = 4520;

const ROW_SHIFT: u32 = 11;
const COL_SHIFT: u32 = 20;
const DC_SHIFT: u32 = 3;

/// `(SUINT)w * x` —— 系数可正可负，按 C 的无符号回绕语义。
#[inline]
fn mul(w: u32, x: i16) -> u32 {
    w.wrapping_mul(x as i32 as u32)
}

/// 行变换（原地）。只有直流时走捷径。
fn idct_row(row: &mut [i16; 8]) {
    if row[1] == 0 && row[2] == 0 && row[3] == 0 && row[4] == 0 && row[5] == 0 && row[6] == 0
        && row[7] == 0
    {
        // temp = (row[0] << DC_SHIFT) & 0xffff，八份
        let t = (((row[0] as i32) << DC_SHIFT) as u16) as i16;
        *row = [t; 8];
        return;
    }

    let mut a0 = mul(W4, row[0]).wrapping_add(1 << (ROW_SHIFT - 1));
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;

    a0 = a0.wrapping_add(mul(W2, row[2]));
    a1 = a1.wrapping_add(mul(W6, row[2]));
    a2 = a2.wrapping_sub(mul(W6, row[2]));
    a3 = a3.wrapping_sub(mul(W2, row[2]));

    let mut b0 = mul(W1, row[1]);
    let mut b1 = mul(W3, row[1]);
    let mut b2 = mul(W5, row[1]);
    let mut b3 = mul(W7, row[1]);

    b0 = b0.wrapping_add(mul(W3, row[3]));
    b1 = b1.wrapping_sub(mul(W7, row[3]));
    b2 = b2.wrapping_sub(mul(W1, row[3]));
    b3 = b3.wrapping_sub(mul(W5, row[3]));

    if row[4] != 0 || row[5] != 0 || row[6] != 0 || row[7] != 0 {
        a0 = a0.wrapping_add(mul(W4, row[4])).wrapping_add(mul(W6, row[6]));
        a1 = a1.wrapping_sub(mul(W4, row[4])).wrapping_sub(mul(W2, row[6]));
        a2 = a2.wrapping_sub(mul(W4, row[4])).wrapping_add(mul(W2, row[6]));
        a3 = a3.wrapping_add(mul(W4, row[4])).wrapping_sub(mul(W6, row[6]));

        b0 = b0.wrapping_add(mul(W5, row[5])).wrapping_add(mul(W7, row[7]));
        b1 = b1.wrapping_sub(mul(W1, row[5])).wrapping_sub(mul(W5, row[7]));
        b2 = b2.wrapping_add(mul(W7, row[5])).wrapping_add(mul(W3, row[7]));
        b3 = b3.wrapping_add(mul(W3, row[5])).wrapping_sub(mul(W1, row[7]));
    }

    let s = ROW_SHIFT;
    row[0] = ((a0.wrapping_add(b0)) as i32 >> s) as i16;
    row[7] = ((a0.wrapping_sub(b0)) as i32 >> s) as i16;
    row[1] = ((a1.wrapping_add(b1)) as i32 >> s) as i16;
    row[6] = ((a1.wrapping_sub(b1)) as i32 >> s) as i16;
    row[2] = ((a2.wrapping_add(b2)) as i32 >> s) as i16;
    row[5] = ((a2.wrapping_sub(b2)) as i32 >> s) as i16;
    row[3] = ((a3.wrapping_add(b3)) as i32 >> s) as i16;
    row[4] = ((a3.wrapping_sub(b3)) as i32 >> s) as i16;
}

#[inline]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// 完整的 8×8 反变换：先逐行、再逐列，返回**每列 8 个未 clip 的输出**。
///
/// `put` 与 `add` 的唯一差别就在最后一步怎么用这些值 —— 前面（含列方向的 bias
/// `(1 << (COL_SHIFT-1))/W4` = 32）完全一样，所以两趟共用这一个函数。
fn idct_pass(block: &mut [i16; 64]) -> [[i32; 8]; 8] {
    // 行
    for y in 0..8 {
        let mut row = [0i16; 8];
        row.copy_from_slice(&block[y * 8..y * 8 + 8]);
        idct_row(&mut row);
        block[y * 8..y * 8 + 8].copy_from_slice(&row);
    }

    // 列
    let bias = ((1 << (COL_SHIFT - 1)) / W4) as i32; // = 32
    let mut cols = [[0i32; 8]; 8];
    for x in 0..8 {
        let c = |i: usize| block[8 * i + x];

        let mut a0 = W4.wrapping_mul((c(0) as i32 + bias) as u32);
        let mut a1 = a0;
        let mut a2 = a0;
        let mut a3 = a0;

        a0 = a0.wrapping_add(mul(W2, c(2)));
        a1 = a1.wrapping_add(mul(W6, c(2)));
        a2 = a2.wrapping_sub(mul(W6, c(2)));
        a3 = a3.wrapping_sub(mul(W2, c(2)));

        let mut b0 = mul(W1, c(1));
        let mut b1 = mul(W3, c(1));
        let mut b2 = mul(W5, c(1));
        let mut b3 = mul(W7, c(1));

        b0 = b0.wrapping_add(mul(W3, c(3)));
        b1 = b1.wrapping_sub(mul(W7, c(3)));
        b2 = b2.wrapping_sub(mul(W1, c(3)));
        b3 = b3.wrapping_sub(mul(W5, c(3)));

        if c(4) != 0 {
            a0 = a0.wrapping_add(mul(W4, c(4)));
            a1 = a1.wrapping_sub(mul(W4, c(4)));
            a2 = a2.wrapping_sub(mul(W4, c(4)));
            a3 = a3.wrapping_add(mul(W4, c(4)));
        }
        if c(5) != 0 {
            b0 = b0.wrapping_add(mul(W5, c(5)));
            b1 = b1.wrapping_sub(mul(W1, c(5)));
            b2 = b2.wrapping_add(mul(W7, c(5)));
            b3 = b3.wrapping_add(mul(W3, c(5)));
        }
        if c(6) != 0 {
            a0 = a0.wrapping_add(mul(W6, c(6)));
            a1 = a1.wrapping_sub(mul(W2, c(6)));
            a2 = a2.wrapping_add(mul(W2, c(6)));
            a3 = a3.wrapping_sub(mul(W6, c(6)));
        }
        if c(7) != 0 {
            b0 = b0.wrapping_add(mul(W7, c(7)));
            b1 = b1.wrapping_sub(mul(W5, c(7)));
            b2 = b2.wrapping_add(mul(W3, c(7)));
            b3 = b3.wrapping_sub(mul(W1, c(7)));
        }

        cols[x] = [
            (a0.wrapping_add(b0)) as i32 >> COL_SHIFT,
            (a1.wrapping_add(b1)) as i32 >> COL_SHIFT,
            (a2.wrapping_add(b2)) as i32 >> COL_SHIFT,
            (a3.wrapping_add(b3)) as i32 >> COL_SHIFT,
            (a3.wrapping_sub(b3)) as i32 >> COL_SHIFT,
            (a2.wrapping_sub(b2)) as i32 >> COL_SHIFT,
            (a1.wrapping_sub(b1)) as i32 >> COL_SHIFT,
            (a0.wrapping_sub(b0)) as i32 >> COL_SHIFT,
        ];
    }
    cols
}

/// 完整的 8×8 反变换，直接把结果写成像素（`stride` 为一行字节数）。
///
/// `block` 是**反量化后**的系数（扫描序已还原到自然序）。
pub fn idct_put(block: &mut [i16; 64], dest: &mut [u8], stride: usize) {
    let cols = idct_pass(block);
    for x in 0..8 {
        for y in 0..8 {
            dest[y * stride + x] = clip(cols[x][y]);
        }
    }
}

/// 同 `idct_put`，但把结果**加**到已有像素上（帧间残差的落点）。
///
/// 即 `dest = clip(dest + (x >> COL_SHIFT))`。
pub fn idct_add(block: &mut [i16; 64], dest: &mut [u8], stride: usize) {
    let cols = idct_pass(block);
    for x in 0..8 {
        for y in 0..8 {
            let cur = dest[y * stride + x] as i32;
            dest[y * stride + x] = (cur + cols[x][y]).clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 参考：教科书 2D IDCT（浮点）。定标与 `simple_idct` 同口径。
    fn ref_idct(block: &[i16; 64]) -> [f64; 64] {
        let mut out = [0f64; 64];
        for y in 0..8 {
            for x in 0..8 {
                let mut s = 0f64;
                for v in 0..8 {
                    for u in 0..8 {
                        let cu = if u == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                        let cv = if v == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                        s += cu
                            * cv
                            * block[v * 8 + u] as f64
                            * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
                            * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0).cos();
                    }
                }
                out[y * 8 + x] = s / 4.0;
            }
        }
        out
    }

    #[test]
    fn dc_only_block_is_uniform() {
        let mut block = [0i16; 64];
        block[0] = 8;
        let mut dest = vec![0u8; 8 * 8];
        idct_put(&mut block, &mut dest, 8);
        assert!(
            dest.iter().all(|&p| p == dest[0]),
            "直流块应当解出恒定的像素"
        );
    }

    #[test]
    fn matches_a_float_reference_within_rounding() {
        // 一串确定性的伪随机系数（避免引 rand 依赖）。
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) as i32 % 121 - 60) as i16 // -60..60
        };

        for _ in 0..200 {
            let mut block = [0i16; 64];
            for c in block.iter_mut() {
                *c = next();
            }
            let want = ref_idct(&block);

            let mut got_block = block;
            let mut dest = vec![0u8; 8 * 8];
            idct_put(&mut got_block, &mut dest, 8);

            for i in 0..64 {
                // 整数 IDCT 与浮点参考的差应当很小（舍入 + clip 边界）。
                let want_c = want[i].round().clamp(0.0, 255.0);
                let d = (dest[i] as f64 - want_c).abs();
                assert!(
                    d <= 2.0,
                    "第 {i} 个像素差 {d}（我们 {}，参考 {}）",
                    dest[i],
                    want_c
                );
            }
        }
    }

    #[test]
    fn zero_block_leaves_the_destination_untouched() {
        let base: Vec<u8> = (0..64).map(|i| (i * 7 % 256) as u8).collect();
        let mut block = [0i16; 64];
        let mut dest = base.clone();
        idct_add(&mut block, &mut dest, 8);
        assert_eq!(dest, base, "全零残差不该改动任何像素");
    }

    #[test]
    fn idct_add_matches_the_float_reference_on_a_base() {
        // add 的语义是 `base + IDCT(block)` 再 clip —— 与 put 的区别只在最后一步。
        // 系数取得小、base 取 128，保证不触到 0/255 的饱和，才好和浮点参考比。
        let mut seed = 0x2468_ACE0u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) as i32 % 17 - 8) as i16 // -8..8
        };

        for _ in 0..200 {
            let mut block = [0i16; 64];
            for c in block.iter_mut() {
                *c = next();
            }
            let want = ref_idct(&block);

            let base = vec![128u8; 64];
            let mut got_block = block;
            let mut got = base.clone();
            idct_add(&mut got_block, &mut got, 8);

            for i in 0..64 {
                let want_v = (base[i] as f64 + want[i]).round().clamp(0.0, 255.0);
                let d = (got[i] as f64 - want_v).abs();
                assert!(
                    d <= 2.0,
                    "第 {i} 个像素差 {d}（我们 {}，参考 {}）",
                    got[i],
                    want_v
                );
            }
        }
    }

    #[test]
    fn output_is_monotonic_in_dc() {
        let mut last = -1i32;
        for dc in 0..40i16 {
            let mut block = [0i16; 64];
            block[0] = dc;
            let mut dest = vec![0u8; 64];
            idct_put(&mut block, &mut dest, 8);
            let v = dest[0] as i32;
            assert!(v >= last, "直流增大时输出不该变小");
            last = v;
        }
    }
}
