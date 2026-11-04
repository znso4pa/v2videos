//! WMV2 自己的 8×8 IDCT 与 ABT 的落位 —— **只有 WMV2 用**，所以留在编解码器里。
//!
//! 通用 8×8 的 `idct_put` / `idct_add` 在 `v2v-kernel::idct`（mp4v / mpeg12 也用它）。
//! ABT 的 `idct48_add` / `idct84_add` 也在 kernel —— 它们要用 kernel 内部的
//! `idct_col8`，而那不该为这一条路专门开成 `pub`。
//!
//! ⚠️ 这里的常数与 ffmpeg `wmv2dsp.c` 一字不差。别拿 `simple_idct` 替：解出来
//! 是「大体像、逐像素差几」，最难发现的那种错。

// ---- WMV2 自己的 8×8 IDCT（`wmv2dsp.c` 的 `wmv2_idct_*`）----
//
// ⚠️ **WMV2 的每一块都不走 `simple_idct`**：`ff_wmv2_common_init` 把
// `idsp.idct_add/idct_put` 换成了 `wmv2_idct_add/put`，常数与舍入另有一套。
// 拿 `simple_idct` 去解 WMV2 得到的是「大体像、逐像素差几」—— 最难发现的那种错。
//
// 定标也不同：行变换除以 2^8、列变换除以 2^14（行变换里那次 `>> 3` 是**精度扩展**）。

/// `wmv2dsp.c` 的 `W0..W7`：`2048·√2·cos(iπ/16)` 取整。
/// （上游的 `W4` 与 `W0` 同值 2048，那边两处都写 `W0`，这里同理只留一个常数。）
const K0: i32 = 2048;
const K1: i32 = 2841;
const K2: i32 = 2676;
const K3: i32 = 2408;
const K5: i32 = 1609;
const K6: i32 = 1108;
const K7: i32 = 565;

/// C 里的 `(int)(181U * d + 128) >> 8` —— `181U` 让整个乘法按 **u32** 走，
/// 转回 `int` 后再做**算术**右移。照着抄，别「修」。
#[inline]
fn wrap181(d: i32) -> i32 {
    (181u32.wrapping_mul(d as u32).wrapping_add(128) as i32) >> 8
}

/// WMV2 的 1D **行**变换（原地）。
fn wmv2_idct_row(b: &mut [i16; 8]) {
    let (v0, v1, v2, v3, v4, v5, v6, v7) = (
        b[0] as i32,
        b[1] as i32,
        b[2] as i32,
        b[3] as i32,
        b[4] as i32,
        b[5] as i32,
        b[6] as i32,
        b[7] as i32,
    );
    let a1 = K1 * v1 + K7 * v7;
    let a7 = K7 * v1 - K1 * v7;
    let a5 = K5 * v5 + K3 * v3;
    let a3 = K3 * v5 - K5 * v3;
    let a2 = K2 * v2 + K6 * v6;
    let a6 = K6 * v2 - K2 * v6;
    let a0 = K0 * v0 + K0 * v4;
    let a4 = K0 * v0 - K0 * v4;

    let s1 = wrap181(a1 - a5 + a7 - a3);
    let s2 = wrap181(a1 - a5 - a7 + a3);

    b[0] = ((a0 + a2 + a1 + a5 + 128) >> 8) as i16;
    b[1] = ((a4 + a6 + s1 + 128) >> 8) as i16;
    b[2] = ((a4 - a6 + s2 + 128) >> 8) as i16;
    b[3] = ((a0 - a2 + a7 + a3 + 128) >> 8) as i16;
    b[4] = ((a0 - a2 - a7 - a3 + 128) >> 8) as i16;
    b[5] = ((a4 - a6 - s2 + 128) >> 8) as i16;
    b[6] = ((a4 + a6 - s1 + 128) >> 8) as i16;
    b[7] = ((a0 + a2 - a1 - a5 + 128) >> 8) as i16;
}

/// WMV2 的 1D **列**变换（原地）—— `i` 是列号，走 `b[i], b[i+8], …`。
fn wmv2_idct_col(b: &mut [i16; 64], i: usize) {
    let g = |k: usize| b[i + 8 * k] as i32;
    let (v0, v1, v2, v3, v4, v5, v6, v7) = (g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7));
    // `+4) >> 3` 是**精度扩展**：把行变换的输出放大 3 位再算，最后统一 `>> 14`。
    let a1 = (K1 * v1 + K7 * v7 + 4) >> 3;
    let a7 = (K7 * v1 - K1 * v7 + 4) >> 3;
    let a5 = (K5 * v5 + K3 * v3 + 4) >> 3;
    let a3 = (K3 * v5 - K5 * v3 + 4) >> 3;
    let a2 = (K2 * v2 + K6 * v6 + 4) >> 3;
    let a6 = (K6 * v2 - K2 * v6 + 4) >> 3;
    let a0 = (K0 * v0 + K0 * v4) >> 3;
    let a4 = (K0 * v0 - K0 * v4) >> 3;

    let s1 = wrap181(a1 - a5 + a7 - a3);
    let s2 = wrap181(a1 - a5 - a7 + a3);

    b[i] = ((a0 + a2 + a1 + a5 + (1 << 13)) >> 14) as i16;
    b[i + 8] = ((a4 + a6 + s1 + (1 << 13)) >> 14) as i16;
    b[i + 16] = ((a4 - a6 + s2 + (1 << 13)) >> 14) as i16;
    b[i + 24] = ((a0 - a2 + a7 + a3 + (1 << 13)) >> 14) as i16;
    b[i + 32] = ((a0 - a2 - a7 - a3 + (1 << 13)) >> 14) as i16;
    b[i + 40] = ((a4 - a6 - s2 + (1 << 13)) >> 14) as i16;
    b[i + 48] = ((a4 + a6 - s1 + (1 << 13)) >> 14) as i16;
    b[i + 56] = ((a0 + a2 - a1 - a5 + (1 << 13)) >> 14) as i16;
}

fn wmv2_idct(block: &mut [i16; 64]) {
    for y in 0..8 {
        let mut row: [i16; 8] = block[y * 8..y * 8 + 8].try_into().expect("8 个");
        wmv2_idct_row(&mut row);
        block[y * 8..y * 8 + 8].copy_from_slice(&row);
    }
    for x in 0..8 {
        wmv2_idct_col(block, x);
    }
}

/// WMV2 的 IDCT + 落位（**覆盖**像素，帧内块用）。
pub fn wmv2_idct_put(block: &mut [i16; 64], dest: &mut [u8], stride: usize) {
    wmv2_idct(block);
    for y in 0..8 {
        let row = &block[y * 8..y * 8 + 8];
        let d = &mut dest[y * stride..y * stride + 8];
        for x in 0..8 {
            d[x] = row[x].clamp(0, 255) as u8;
        }
    }
}

/// WMV2 的 IDCT + 落位（**加到**已有像素上，帧间块用）。
pub fn wmv2_idct_add(block: &mut [i16; 64], dest: &mut [u8], stride: usize) {
    wmv2_idct(block);
    for y in 0..8 {
        let row = &block[y * 8..y * 8 + 8];
        let d = &mut dest[y * stride..y * stride + 8];
        for x in 0..8 {
            d[x] = (d[x] as i32 + row[x] as i32).clamp(0, 255) as u8;
        }
    }
}

// ---- 与上游 C 实现的**逐字节**对账（固定向量）----
//
// 这些期望值来自**上游 `wmv2dsp.c` 的 `wmv2_idct_add_c` 原样编译出来的小程序**：
// 拿 4000 个块（全零 / 只有 DC / 单行 / 满块随机，幅度到 ±2000）跑一遍，我们与它
// **逐字节相同**。这里固化成几个代表块，防的是**以后改这份 IDCT 时悄悄漂掉** ——
// 漂掉的症状是「整帧每块差 1」，在真样本对账里很难一眼归因。
//
// ⚠️ 期望值是 8 位像素，所以只取**没被 clip 饱和**的块（饱和会把差异盖住）。
#[cfg(test)]
mod ref_vectors {
    use super::*;

    fn check(block: [i16; 64], want: [u8; 64]) {
        let mut b = block;
        let mut add = [128u8; 64];
        wmv2_idct_add(&mut b, &mut add, 8);
        assert_eq!(add, want, "wmv2_idct_add 与上游 C 版不一致");
    }

    /// 只有 DC（负）—— 走的是「行变换里 a0 = W0·b0」那条。
    #[test]
    fn dc_only_negative() {
        let mut block = [0i16; 64];
        block[0] = -635;
        check(block, [49; 64]);
    }

    /// 满块小系数（六行以上非零）—— 四个象限的蝶形都过一遍。
    #[test]
    fn dense_small() {
        let block: [i16; 64] = [
            13, 2, 9, 19, -11, -17, 6, -11,
            -19, -20, 19, 5, -17, 17, -7, -6,
            -8, -5, 13, -15, -10, -4, -1, 3,
            18, 16, -8, -17, 18, -12, -3, -12,
            -13, 11, -7, 16, -7, 11, -2, -11,
            17, 7, -12, -12, 2, -3, 12, 9,
            17, 4, -12, -10, -16, 15, 9, -20,
            12, -10, -1, 3, 3, 9, -18, -15,
        ];
        let want: [u8; 64] = [
            125, 140, 128, 132, 111, 137, 131, 130,
            125, 125, 107, 93, 119, 123, 136, 136,
            125, 129, 125, 129, 117, 151, 150, 130,
            137, 122, 131, 112, 134, 140, 120, 113,
            138, 139, 103, 140, 134, 139, 127, 120,
            135, 145, 148, 143, 139, 139, 109, 139,
            134, 141, 138, 127, 155, 125, 118, 147,
            129, 133, 128, 117, 122, 140, 125, 118,
        ];
        check(block, want);
    }

    /// 满块、正负交错（`181U * ...` 那次无符号回绕真的会被走到）。
    #[test]
    fn dense_signed() {
        let block: [i16; 64] = [
            10, -10, 13, 7, -7, -11, -7, 17,
            16, -4, 14, -9, 5, 14, -1, -2,
            14, 3, 18, -4, 0, -19, -14, -15,
            6, 1, 12, 17, 15, -20, 16, -10,
            -12, -3, -7, 15, 17, -7, -11, -18,
            7, -8, 9, 12, 3, -12, -15, 3,
            15, -19, -5, 10, 15, -9, 19, 6,
            -14, -3, 7, 18, 14, 17, -19, -17,
        ];
        let want: [u8; 64] = [
            145, 145, 102, 124, 139, 138, 131, 156,
            128, 140, 157, 125, 113, 117, 141, 132,
            130, 115, 132, 128, 132, 125, 154, 132,
            119, 126, 137, 119, 139, 123, 138, 123,
            132, 113, 107, 130, 124, 127, 123, 113,
            132, 121, 150, 106, 148, 146, 119, 145,
            129, 148, 116, 113, 128, 121, 138, 123,
            116, 138, 123, 124, 136, 128, 126, 126,
        ];
        check(block, want);
    }
}
