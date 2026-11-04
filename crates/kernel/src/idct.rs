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

/// 逐通道的 `wrapping` 加/减。
#[inline(always)]
fn add<const L: usize>(a: [u32; L], b: [u32; L]) -> [u32; L] {
    std::array::from_fn(|i| a[i].wrapping_add(b[i]))
}
#[inline(always)]
fn sub<const L: usize>(a: [u32; L], b: [u32; L]) -> [u32; L] {
    std::array::from_fn(|i| a[i].wrapping_sub(b[i]))
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
    // 行（**原地**：原来每行先拷进临时数组、算完再拷回去 —— 每块 16 次 16 字节拷贝）
    for y in 0..8 {
        let row: &mut [i16; 8] = (&mut block[y * 8..y * 8 + 8]).try_into().unwrap();
        idct_row(row);
    }

    // 列：**4 列一批**（通道 = 列）。每个「系数 j 在 4 列上的值」正好是
    // `block[j*8 + off .. +4]` 这 4 个连续 i16 —— 一次向量加载就够，
    // 不用先把块拷成通道数组（上一版就是被那次拷贝吃掉了收益）。
    let mut cols = [[0i32; 8]; 8];
    for half in 0..2 {
        let off = half * 4;
        // 8 个通道向量（term(j, w) = w * 该系数在 4 列上的值）
        let t = |j: usize, w: u32| -> [u32; 4] {
            std::array::from_fn(|l| mul(w, block[j * 8 + off + l]))
        };
        let bias = ((1 << (COL_SHIFT - 1)) / W4) as i32;
        let init: [u32; 4] =
            std::array::from_fn(|l| W4.wrapping_mul((block[off + l] as i32 + bias) as u32));
        let (mut a0, mut a1, mut a2, mut a3) = (init, init, init, init);
        let (t2a, t2b) = (t(2, W2), t(2, W6));
        a0 = add(a0, t2a);
        a1 = add(a1, t2b);
        a2 = sub(a2, t2b);
        a3 = sub(a3, t2a);
        let (mut b0, mut b1, mut b2, mut b3) = (t(1, W1), t(1, W3), t(1, W5), t(1, W7));
        b0 = add(b0, t(3, W3));
        b1 = sub(b1, t(3, W7));
        b2 = sub(b2, t(3, W1));
        b3 = sub(b3, t(3, W5));
        let t4 = t(4, W4);
        a0 = add(a0, t4);
        a1 = sub(a1, t4);
        a2 = sub(a2, t4);
        a3 = add(a3, t4);
        b0 = add(b0, t(5, W5));
        b1 = sub(b1, t(5, W1));
        b2 = add(b2, t(5, W7));
        b3 = add(b3, t(5, W3));
        let t6 = t(6, W6);
        a0 = add(a0, t6);
        a1 = sub(a1, t(6, W2));
        a2 = add(a2, t(6, W2));
        a3 = sub(a3, t6);
        b0 = add(b0, t(7, W7));
        b1 = sub(b1, t(7, W5));
        b2 = add(b2, t(7, W3));
        b3 = sub(b3, t(7, W1));
        let sum = [
            add(a0, b0),
            add(a1, b1),
            add(a2, b2),
            add(a3, b3),
            sub(a3, b3),
            sub(a2, b2),
            sub(a1, b1),
            sub(a0, b0),
        ];
        for j in 0..8 {
            for l in 0..4 {
                cols[off + l][j] = sum[j][l] as i32 >> COL_SHIFT;
            }
        }
    }
    cols
}

/// 8 点 IDCT 的**列变换** —— 输入一列的 8 个系数，输出 8 个值。
///
/// 抽出来是因为 WMV2 的 ABT 那条路（`idct48_add`）只要**其中 4 列**，
/// 而 8×8 那条路要全部 8 列 —— 复刻第二份就是会漂的那一份。
fn idct_col8(c: &[i16; 8]) -> [i32; 8] {
    let bias = ((1 << (COL_SHIFT - 1)) / W4) as i32; // = 32
    let mut a0 = W4.wrapping_mul((c[0] as i32 + bias) as u32);
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;

    a0 = a0.wrapping_add(mul(W2, c[2]));
    a1 = a1.wrapping_add(mul(W6, c[2]));
    a2 = a2.wrapping_sub(mul(W6, c[2]));
    a3 = a3.wrapping_sub(mul(W2, c[2]));

    let mut b0 = mul(W1, c[1]);
    let mut b1 = mul(W3, c[1]);
    let mut b2 = mul(W5, c[1]);
    let mut b3 = mul(W7, c[1]);

    b0 = b0.wrapping_add(mul(W3, c[3]));
    b1 = b1.wrapping_sub(mul(W7, c[3]));
    b2 = b2.wrapping_sub(mul(W1, c[3]));
    b3 = b3.wrapping_sub(mul(W5, c[3]));

    if c[4] != 0 {
        a0 = a0.wrapping_add(mul(W4, c[4]));
        a1 = a1.wrapping_sub(mul(W4, c[4]));
        a2 = a2.wrapping_sub(mul(W4, c[4]));
        a3 = a3.wrapping_add(mul(W4, c[4]));
    }
    if c[5] != 0 {
        b0 = b0.wrapping_add(mul(W5, c[5]));
        b1 = b1.wrapping_sub(mul(W1, c[5]));
        b2 = b2.wrapping_add(mul(W7, c[5]));
        b3 = b3.wrapping_add(mul(W3, c[5]));
    }
    if c[6] != 0 {
        a0 = a0.wrapping_add(mul(W6, c[6]));
        a1 = a1.wrapping_sub(mul(W2, c[6]));
        a2 = a2.wrapping_add(mul(W2, c[6]));
        a3 = a3.wrapping_sub(mul(W6, c[6]));
    }
    if c[7] != 0 {
        b0 = b0.wrapping_add(mul(W7, c[7]));
        b1 = b1.wrapping_sub(mul(W5, c[7]));
        b2 = b2.wrapping_add(mul(W3, c[7]));
        b3 = b3.wrapping_sub(mul(W1, c[7]));
    }

    [
        (a0.wrapping_add(b0)) as i32 >> COL_SHIFT,
        (a1.wrapping_add(b1)) as i32 >> COL_SHIFT,
        (a2.wrapping_add(b2)) as i32 >> COL_SHIFT,
        (a3.wrapping_add(b3)) as i32 >> COL_SHIFT,
        (a3.wrapping_sub(b3)) as i32 >> COL_SHIFT,
        (a2.wrapping_sub(b2)) as i32 >> COL_SHIFT,
        (a1.wrapping_sub(b1)) as i32 >> COL_SHIFT,
        (a0.wrapping_sub(b0)) as i32 >> COL_SHIFT,
    ]
}

/// 完整的 8×8 反变换，直接把结果写成像素（`stride` 为一行字节数）。
///
/// `block` 是**反量化后**的系数（扫描序已还原到自然序）。
pub fn idct_put(block: &mut [i16; 64], dest: &mut [u8], stride: usize) {
    let cols = idct_pass(block);
    // 行外层 + 行切片，理由同 `idct_add`。
    for (y, row) in dest.chunks_mut(stride).take(8).enumerate() {
        for (x, px) in row[..8].iter_mut().enumerate() {
            *px = clip(cols[x][y]);
        }
    }
}

/// 同 `idct_put`，但把结果**加**到已有像素上（帧间残差的落点）。
///
/// 即 `dest = clip(dest + (x >> COL_SHIFT))`。
pub fn idct_add(block: &mut [i16; 64], dest: &mut [u8], stride: usize) {
    let cols = idct_pass(block);
    // ⚠️ **行外层 + 行切片**：`dest[y * stride + x]` 用计算下标写，每个像素
    // 都要过一次边界检查，而且 8 个像素跨 8 行（`stride` 是 320/368 这种）。
    // 换成行切片之后检查提到循环外，写是连续的 —— 值一个不差。
    for (y, row) in dest.chunks_mut(stride).take(8).enumerate() {
        for (x, px) in row[..8].iter_mut().enumerate() {
            *px = (*px as i32 + cols[x][y]).clamp(0, 255) as u8;
        }
    }
}

// ───────────────────────── WMV2 的 ABT（自适应块变换）─────────────────────────
//
// ⚠️ ABT 把 8×8 块**切成两个 4×8（或 8×4）的半块**，各做一次变换再落位 ——
// 所以需要 **4 点** 的 IDCT。ffmpeg 那边是 `ff_simple_idct84_add` / `ff_simple_idct48_add`。
//
// `84` = 4 行 × 8 列：先对**每一行**做 8 点 IDCT，再对**每一列**做 4 点 IDCT + 落位。
// `48` = 8 行 × 4 列：先对**每一行**做 4 点 IDCT，再对**每一列**做 8 点 IDCT + 落位。
//
// ⚠️ 常数与 `simple_idct.c` 一字不差（`C_FIX(x) = round(x·√2·2^12)`、`R_FIX` 同理）：
// 这两条路的**定标必须与 8×8 那条一致**，否则 ABT 块的亮度会整体偏。

/// `C_FIX(x) = round(x · √2 · 2^12)` —— 列变换用的定点常数。
const C1: i32 = 3_784; // C_FIX(0.6532814824)
const C2: i32 = 1_567; // C_FIX(0.2705980501)
const C3: i32 = 2_896; // C_FIX(0.5)
const C_SHIFT: u32 = 4 + 1 + 12;

/// `R_FIX(x) = round(x · √2 · 2^15)` —— 行变换用的定点常数。
const R1: i32 = 30_274; // R_FIX(0.6532814824)
const R2: i32 = 12_540; // R_FIX(0.2705980501)
const R3: i32 = 23_170; // R_FIX(0.5)
const R_SHIFT: u32 = 11;

/// 4 点 IDCT —— **行**（原地，写回 `row[0..4]`）。
///
/// ⚠️ `c0..c3` 在 C 里是 **`unsigned`**（`simple_idct.c` 的 `idct4row`）—— 那不是笔误：
/// 中间值**可以**为负，于是这一步按无符号回绕、`>>` 也是**逻辑**移位，最后写回 `int16`
/// 时再截断。照抄这个语义（而不是「修好」它），否则与 ffmpeg 的 ABT 块对不上。
fn idct4row(row: &mut [i16; 8]) {
    let (a0, a1, a2, a3) = (row[0] as i32, row[1] as i32, row[2] as i32, row[3] as i32);
    let bias = 1 << (R_SHIFT - 1);
    let c0 = (a0 + a2).wrapping_mul(R3).wrapping_add(bias) as u32;
    let c2 = (a0 - a2).wrapping_mul(R3).wrapping_add(bias) as u32;
    let c1 = a1.wrapping_mul(R1).wrapping_add(a3.wrapping_mul(R2)) as u32;
    let c3 = a1.wrapping_mul(R2).wrapping_sub(a3.wrapping_mul(R1)) as u32;
    row[0] = c0.wrapping_add(c1).wrapping_shr(R_SHIFT) as i16;
    row[1] = c2.wrapping_add(c3).wrapping_shr(R_SHIFT) as i16;
    row[2] = c2.wrapping_sub(c3).wrapping_shr(R_SHIFT) as i16;
    row[3] = c0.wrapping_sub(c1).wrapping_shr(R_SHIFT) as i16;
}

/// 4 点 IDCT —— **列**，**加到已有像素上**。
///
/// `col` 是这一列的 4 个系数；结果按 `stride` 往下走 4 行，加到 `dest` 上（带 clamp）。
fn idct4col_add(dest: &mut [u8], stride: usize, col: &[i16; 4]) {
    let (a0, a1, a2, a3) = (col[0] as i32, col[1] as i32, col[2] as i32, col[3] as i32);
    let bias = 1 << (C_SHIFT - 1);
    let c0 = (a0 + a2) * C3 + bias;
    let c2 = (a0 - a2) * C3 + bias;
    let c1 = a1 * C1 + a3 * C2;
    let c3 = a1 * C2 - a3 * C1;
    for (i, v) in [(c0 + c1) >> C_SHIFT, (c2 + c3) >> C_SHIFT, (c2 - c3) >> C_SHIFT, (c0 - c1) >> C_SHIFT]
        .into_iter()
        .enumerate()
    {
        let p = &mut dest[i * stride];
        *p = (*p as i32 + v).clamp(0, 255) as u8;
    }
}

/// **4 行 × 8 列**的半块：每行 8 点 IDCT，再每列 4 点 IDCT + 落位。
///
/// `block` 是 4 行 × 8 列的系数（按 `block[row * 8 + col]` 排）。
pub fn idct84_add(block: &mut [i16; 64], dest: &mut [u8], stride: usize) {
    for y in 0..4 {
        let mut row: [i16; 8] = block[y * 8..y * 8 + 8].try_into().expect("8 个");
        idct_row(&mut row);
        block[y * 8..y * 8 + 8].copy_from_slice(&row);
    }
    for x in 0..8 {
        let col = [block[x], block[8 + x], block[16 + x], block[24 + x]];
        idct4col_add(&mut dest[x..], stride, &col);
    }
}

/// **8 行 × 4 列**的半块：每行 4 点 IDCT，再每列 8 点 IDCT + 落位。
pub fn idct48_add(block: &mut [i16; 64], dest: &mut [u8], stride: usize) {
    for y in 0..8 {
        let mut row: [i16; 8] = block[y * 8..y * 8 + 8].try_into().expect("8 个");
        idct4row(&mut row);
        block[y * 8..y * 8 + 8].copy_from_slice(&row);
    }
    // 列：8 点 IDCT，但只落 **4 列**。
    for x in 0..4 {
        let col: [i16; 8] = std::array::from_fn(|y| block[y * 8 + x]);
        let out = idct_col8(&col);
        for (y, v) in out.into_iter().enumerate() {
            let p = &mut dest[y * stride + x];
            *p = (*p as i32 + v).clamp(0, 255) as u8;
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
