//! VC-1 的**逆变换**：8×8 / 8×4 / 4×8 / 4×4 四套，外加各自的「只有 DC」捷径，
//! 以及把变换结果落到像素上的三个写回函数。
//!
//! 与 MPEG-4 那套**不是一回事**（`v2v-msmpeg4` 里的 `simple_idct` 别往这儿搬）：
//! VC-1 的变换核是整数蝶形，系数 12/16/6/9/15/4 那一组，两趟分别 `>> 3` 与 `>> 7`。
//! 定标也不一样，搬错就是「能出图但一片噪声」。
//!
//! 对照上游：`libavcodec/vc1dsp.c` 的 `vc1_inv_trans_{8x8,8x4,4x8,4x4}[_dc]_c()` 与
//! `idctdsp.c` 的 `put_pixels_clamped` / `put_signed_pixels_clamped` / `add_pixels_clamped`。
//! 这是 **VC-1（SMPTE 421M）规范里的变换**，不是某个实现的私有数据。
//!
//! ## 为什么照抄得这么死
//!
//! 判据是**逐像素 0 差**：任何一个常数、任何一处 `+4`/`+64` 的舍入偏置、任何一次
//! 移位的方向写错，出来都不是「差一点」而是整块偏色。所以这里一个数都没动 ——
//! 连「哪些项加 1」都按上游的写法保留（`(t8 - t4 + 1) >> 7` 那种）。
//!
//! ## 数据形状
//!
//! 四种尺寸都吃**同一个 8×8 的 `i16` 缓冲**（行距固定 8），区别只在用哪几行哪几列：
//! - `8x8`：整块，**就地**变换（调用方随后把结果加到像素上）
//! - `8x4`：用前 4 行、全 8 列，**直接写像素**（就地加）
//! - `4x8`：用全 8 行、前 4 列，直接写像素
//! - `4x4`：用前 4 行、前 4 列，直接写像素
//!
//! ABT 的 4×4 子块在**同一块缓冲里**以 `+off` 偏移取（上游传 `block + off`）——
//! 所以这几个函数吃的是**切片**而不是定长数组，偏移由调用方给。
//!
//! 中间量一律 `i32`（与 C 的 `int` 一致），**落回 `i16` 缓冲时按 C 的截断**。

/// `av_clip_uint8`：先加后夹。
#[inline(always)]
fn add_clip(dst: u8, v: i32) -> u8 {
    (i32::from(dst) + v).clamp(0, 255) as u8
}

/// 8×8 逆变换（**就地**改 `block`；`block` 至少 64 项，步长固定 8）。
pub fn inv_trans_8x8(block: &mut [i16]) {
    let mut temp = [0i16; 64];

    // ---- 第一趟：列（源步长 8，落在 temp 的行上）----
    for i in 0..8usize {
        let s = |k: usize| i32::from(block[i + k * 8]);
        let (t1, t2, t3, t4) = (
            12 * (s(0) + s(4)) + 4,
            12 * (s(0) - s(4)) + 4,
            16 * s(2) + 6 * s(6),
            6 * s(2) - 16 * s(6),
        );
        let (t5, t6, t7, t8) = (t1 + t3, t2 + t4, t2 - t4, t1 - t3);
        let (u1, u2, u3, u4) = (
            16 * s(1) + 15 * s(3) + 9 * s(5) + 4 * s(7),
            15 * s(1) - 4 * s(3) - 16 * s(5) - 9 * s(7),
            9 * s(1) - 16 * s(3) + 4 * s(5) + 15 * s(7),
            4 * s(1) - 9 * s(3) + 15 * s(5) - 16 * s(7),
        );
        let d = |k: usize| i * 8 + k;
        temp[d(0)] = ((t5 + u1) >> 3) as i16;
        temp[d(1)] = ((t6 + u2) >> 3) as i16;
        temp[d(2)] = ((t7 + u3) >> 3) as i16;
        temp[d(3)] = ((t8 + u4) >> 3) as i16;
        temp[d(4)] = ((t8 - u4) >> 3) as i16;
        temp[d(5)] = ((t7 - u3) >> 3) as i16;
        temp[d(6)] = ((t6 - u2) >> 3) as i16;
        temp[d(7)] = ((t5 - u1) >> 3) as i16;
    }

    // ---- 第二趟：行（源步长 1，落在 block 的列上）----
    for i in 0..8usize {
        let s = |k: usize| i32::from(temp[k * 8 + i]);
        let (t1, t2, t3, t4) = (
            12 * (s(0) + s(4)) + 64,
            12 * (s(0) - s(4)) + 64,
            16 * s(2) + 6 * s(6),
            6 * s(2) - 16 * s(6),
        );
        let (t5, t6, t7, t8) = (t1 + t3, t2 + t4, t2 - t4, t1 - t3);
        let (u1, u2, u3, u4) = (
            16 * s(1) + 15 * s(3) + 9 * s(5) + 4 * s(7),
            15 * s(1) - 4 * s(3) - 16 * s(5) - 9 * s(7),
            9 * s(1) - 16 * s(3) + 4 * s(5) + 15 * s(7),
            4 * s(1) - 9 * s(3) + 15 * s(5) - 16 * s(7),
        );
        let d = |k: usize| k * 8 + i;
        block[d(0)] = ((t5 + u1) >> 7) as i16;
        block[d(1)] = ((t6 + u2) >> 7) as i16;
        block[d(2)] = ((t7 + u3) >> 7) as i16;
        block[d(3)] = ((t8 + u4) >> 7) as i16;
        // ⚠️ 后四项的 `+1` 是上游的写法，不是笔误 —— 别「顺手统一」。
        block[d(4)] = ((t8 - u4 + 1) >> 7) as i16;
        block[d(5)] = ((t7 - u3 + 1) >> 7) as i16;
        block[d(6)] = ((t6 - u2 + 1) >> 7) as i16;
        block[d(7)] = ((t5 - u1 + 1) >> 7) as i16;
    }
}

/// 8×8 的「只有 DC」捷径：`dc` 三次缩放后**整块加同一个值**。
pub fn inv_trans_8x8_dc(dst: &mut [u8], stride: usize, block: &[i16], x: usize, y: usize) {
    let mut dc = i32::from(block[0]);
    dc = (3 * dc + 1) >> 1;
    dc = (3 * dc + 16) >> 5;
    for r in 0..8usize {
        for c in 0..8usize {
            dst[(y + r) * stride + x + c] = add_clip(dst[(y + r) * stride + x + c], dc);
        }
    }
}

/// 8×4 逆变换（用前 4 行 × 8 列，**直接写像素**）。
pub fn inv_trans_8x4(dst: &mut [u8], stride: usize, block: &mut [i16], x: usize, y: usize) {
    // 第一趟：4 行 × 8 列，就地（步长 8）
    for i in 0..4usize {
        let s = |k: usize| i32::from(block[i * 8 + k]);
        let (t1, t2, t3, t4) = (
            12 * (s(0) + s(4)) + 4,
            12 * (s(0) - s(4)) + 4,
            16 * s(2) + 6 * s(6),
            6 * s(2) - 16 * s(6),
        );
        let (t5, t6, t7, t8) = (t1 + t3, t2 + t4, t2 - t4, t1 - t3);
        let (u1, u2, u3, u4) = (
            16 * s(1) + 15 * s(3) + 9 * s(5) + 4 * s(7),
            15 * s(1) - 4 * s(3) - 16 * s(5) - 9 * s(7),
            9 * s(1) - 16 * s(3) + 4 * s(5) + 15 * s(7),
            4 * s(1) - 9 * s(3) + 15 * s(5) - 16 * s(7),
        );
        let d = |k: usize| i * 8 + k;
        block[d(0)] = ((t5 + u1) >> 3) as i16;
        block[d(1)] = ((t6 + u2) >> 3) as i16;
        block[d(2)] = ((t7 + u3) >> 3) as i16;
        block[d(3)] = ((t8 + u4) >> 3) as i16;
        block[d(4)] = ((t8 - u4) >> 3) as i16;
        block[d(5)] = ((t7 - u3) >> 3) as i16;
        block[d(6)] = ((t6 - u2) >> 3) as i16;
        block[d(7)] = ((t5 - u1) >> 3) as i16;
    }
    // 第二趟：8 列 × 4 行（步长 8），落在像素上
    for i in 0..8usize {
        let s = |k: usize| i32::from(block[k * 8 + i]);
        let (t1, t2) = (17 * (s(0) + s(2)) + 64, 17 * (s(0) - s(2)) + 64);
        let (t3, t4) = (22 * s(1) + 10 * s(3), 22 * s(3) - 10 * s(1));
        let base = y * stride + x + i;
        dst[base] = add_clip(dst[base], (t1 + t3) >> 7);
        dst[base + stride] = add_clip(dst[base + stride], (t2 - t4) >> 7);
        dst[base + 2 * stride] = add_clip(dst[base + 2 * stride], (t2 + t4) >> 7);
        dst[base + 3 * stride] = add_clip(dst[base + 3 * stride], (t1 - t3) >> 7);
    }
}

/// 8×4 的「只有 DC」捷径。
pub fn inv_trans_8x4_dc(dst: &mut [u8], stride: usize, block: &[i16], x: usize, y: usize) {
    let mut dc = i32::from(block[0]);
    dc = (3 * dc + 1) >> 1;
    dc = (17 * dc + 64) >> 7;
    for r in 0..4usize {
        for c in 0..8usize {
            dst[(y + r) * stride + x + c] = add_clip(dst[(y + r) * stride + x + c], dc);
        }
    }
}

/// 4×8 逆变换（用 8 行 × 前 4 列，**直接写像素**）。
pub fn inv_trans_4x8(dst: &mut [u8], stride: usize, block: &mut [i16], x: usize, y: usize) {
    // 第一趟：8 行 × 4 列，就地（步长 8）
    for i in 0..8usize {
        let s = |k: usize| i32::from(block[i * 8 + k]);
        let (t1, t2) = (17 * (s(0) + s(2)) + 4, 17 * (s(0) - s(2)) + 4);
        let (t3, t4) = (22 * s(1) + 10 * s(3), 22 * s(3) - 10 * s(1));
        let d = |k: usize| i * 8 + k;
        block[d(0)] = ((t1 + t3) >> 3) as i16;
        block[d(1)] = ((t2 - t4) >> 3) as i16;
        block[d(2)] = ((t2 + t4) >> 3) as i16;
        block[d(3)] = ((t1 - t3) >> 3) as i16;
    }
    // 第二趟：4 列 × 8 行（步长 8），落在像素上
    for i in 0..4usize {
        let s = |k: usize| i32::from(block[k * 8 + i]);
        let (t1, t2, t3, t4) = (
            12 * (s(0) + s(4)) + 64,
            12 * (s(0) - s(4)) + 64,
            16 * s(2) + 6 * s(6),
            6 * s(2) - 16 * s(6),
        );
        let (t5, t6, t7, t8) = (t1 + t3, t2 + t4, t2 - t4, t1 - t3);
        let (u1, u2, u3, u4) = (
            16 * s(1) + 15 * s(3) + 9 * s(5) + 4 * s(7),
            15 * s(1) - 4 * s(3) - 16 * s(5) - 9 * s(7),
            9 * s(1) - 16 * s(3) + 4 * s(5) + 15 * s(7),
            4 * s(1) - 9 * s(3) + 15 * s(5) - 16 * s(7),
        );
        let base = y * stride + x + i;
        dst[base] = add_clip(dst[base], (t5 + u1) >> 7);
        dst[base + stride] = add_clip(dst[base + stride], (t6 + u2) >> 7);
        dst[base + 2 * stride] = add_clip(dst[base + 2 * stride], (t7 + u3) >> 7);
        dst[base + 3 * stride] = add_clip(dst[base + 3 * stride], (t8 + u4) >> 7);
        dst[base + 4 * stride] = add_clip(dst[base + 4 * stride], (t8 - u4 + 1) >> 7);
        dst[base + 5 * stride] = add_clip(dst[base + 5 * stride], (t7 - u3 + 1) >> 7);
        dst[base + 6 * stride] = add_clip(dst[base + 6 * stride], (t6 - u2 + 1) >> 7);
        dst[base + 7 * stride] = add_clip(dst[base + 7 * stride], (t5 - u1 + 1) >> 7);
    }
}

/// 4×8 的「只有 DC」捷径。
pub fn inv_trans_4x8_dc(dst: &mut [u8], stride: usize, block: &[i16], x: usize, y: usize) {
    let mut dc = i32::from(block[0]);
    dc = (17 * dc + 4) >> 3;
    dc = (12 * dc + 64) >> 7;
    for r in 0..8usize {
        for c in 0..4usize {
            dst[(y + r) * stride + x + c] = add_clip(dst[(y + r) * stride + x + c], dc);
        }
    }
}

/// 4×4 逆变换（用前 4 行 × 前 4 列，**直接写像素**）。
pub fn inv_trans_4x4(dst: &mut [u8], stride: usize, block: &mut [i16], x: usize, y: usize) {
    // 第一趟：4 行 × 4 列，就地（步长 8）
    for i in 0..4usize {
        let s = |k: usize| i32::from(block[i * 8 + k]);
        let (t1, t2) = (17 * (s(0) + s(2)) + 4, 17 * (s(0) - s(2)) + 4);
        let (t3, t4) = (22 * s(1) + 10 * s(3), 22 * s(3) - 10 * s(1));
        let d = |k: usize| i * 8 + k;
        block[d(0)] = ((t1 + t3) >> 3) as i16;
        block[d(1)] = ((t2 - t4) >> 3) as i16;
        block[d(2)] = ((t2 + t4) >> 3) as i16;
        block[d(3)] = ((t1 - t3) >> 3) as i16;
    }
    // 第二趟：4 列 × 4 行（步长 8），落在像素上
    for i in 0..4usize {
        let s = |k: usize| i32::from(block[k * 8 + i]);
        let (t1, t2) = (17 * (s(0) + s(2)) + 64, 17 * (s(0) - s(2)) + 64);
        let (t3, t4) = (22 * s(1) + 10 * s(3), 22 * s(3) - 10 * s(1));
        let base = y * stride + x + i;
        dst[base] = add_clip(dst[base], (t1 + t3) >> 7);
        dst[base + stride] = add_clip(dst[base + stride], (t2 - t4) >> 7);
        dst[base + 2 * stride] = add_clip(dst[base + 2 * stride], (t2 + t4) >> 7);
        dst[base + 3 * stride] = add_clip(dst[base + 3 * stride], (t1 - t3) >> 7);
    }
}

/// 4×4 的「只有 DC」捷径。
pub fn inv_trans_4x4_dc(dst: &mut [u8], stride: usize, block: &[i16], x: usize, y: usize) {
    let mut dc = i32::from(block[0]);
    dc = (17 * dc + 4) >> 3;
    dc = (17 * dc + 64) >> 7;
    for r in 0..4usize {
        for c in 0..4usize {
            dst[(y + r) * stride + x + c] = add_clip(dst[(y + r) * stride + x + c], dc);
        }
    }
}

// ---- 三个写回函数（`idctdsp.c` 的同名函数）----

/// 把块值**当像素**写（I 帧）：夹到 `[0, 255]`。
pub fn put_pixels_clamped(block: &[i16; 64], dst: &mut [u8], stride: usize, x: usize, y: usize) {
    for r in 0..8usize {
        for c in 0..8usize {
            dst[(y + r) * stride + x + c] = block[r * 8 + c].clamp(0, 255) as u8;
        }
    }
}

/// 把块值当**有符号残差**写（P 帧的帧内块）：先 `+128` 再夹取。
///
/// ⚠️ 与 `put_pixels_clamped` 差一个 128 的偏置 —— 这是上游的语义，不是笔误。
pub fn put_signed_pixels_clamped(
    block: &[i16; 64],
    dst: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
) {
    for r in 0..8usize {
        for c in 0..8usize {
            let v = block[r * 8 + c];
            dst[(y + r) * stride + x + c] = if v < -128 {
                0
            } else if v > 127 {
                255
            } else {
                (v + 128) as u8
            };
        }
    }
}

/// 把块值**加到**像素上（P 帧的帧间残差）。
pub fn add_pixels_clamped(block: &[i16; 64], dst: &mut [u8], stride: usize, x: usize, y: usize) {
    for r in 0..8usize {
        for c in 0..8usize {
            dst[(y + r) * stride + x + c] = add_clip(dst[(y + r) * stride + x + c], i32::from(block[r * 8 + c]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 全零进 → 全零出（四种尺寸都该如此，且不该 panic）。
    #[test]
    fn zero_in_zero_out() {
        let mut b = [0i16; 64];
        inv_trans_8x8(&mut b);
        assert_eq!(b, [0i16; 64]);

        let mut px = vec![128u8; 16 * 16];
        let mut b = [0i16; 64];
        inv_trans_8x4(&mut px, 16, &mut b, 0, 0);
        inv_trans_4x8(&mut px, 16, &mut b, 0, 0);
        inv_trans_4x4(&mut px, 16, &mut b, 0, 0);
        assert!(px.iter().all(|&p| p == 128), "全零系数不该改像素");
    }

    /// **只有 DC** 时，变换结果必须与「DC 捷径」逐像素相同。
    ///
    /// 这条把两套实现钉在一起：`inv_trans_8x8` 走完整蝶形，
    /// `inv_trans_8x8_dc` 走三次缩放的捷径 —— 上游认为它们等价，我们也得等价，
    /// 不然「什么时候用捷径」就成了一个会改变像素的选择。
    #[test]
    fn dc_only_matches_the_full_transform() {
        for dc in [-2048i16, -1000, -255, -1, 0, 1, 255, 1000, 2047] {
            // 8×8
            let mut b = [0i16; 64];
            b[0] = dc;
            let mut full = vec![0u8; 8 * 8];
            let mut blk = b;
            inv_trans_8x8(&mut blk);
            for r in 0..8 {
                for c in 0..8 {
                    full[r * 8 + c] = add_clip(0, i32::from(blk[r * 8 + c]));
                }
            }
            let mut shortcut = vec![0u8; 8 * 8];
            inv_trans_8x8_dc(&mut shortcut, 8, &b, 0, 0);
            assert_eq!(full, shortcut, "8×8 只有 DC 时 dc={dc}");

            // 8×4
            let mut blk = b;
            let mut full = vec![0u8; 8 * 8];
            inv_trans_8x4(&mut full, 8, &mut blk, 0, 0);
            let mut shortcut = vec![0u8; 8 * 8];
            inv_trans_8x4_dc(&mut shortcut, 8, &b, 0, 0);
            assert_eq!(full, shortcut, "8×4 只有 DC 时 dc={dc}");

            // 4×8
            let mut blk = b;
            let mut full = vec![0u8; 8 * 8];
            inv_trans_4x8(&mut full, 8, &mut blk, 0, 0);
            let mut shortcut = vec![0u8; 8 * 8];
            inv_trans_4x8_dc(&mut shortcut, 8, &b, 0, 0);
            assert_eq!(full, shortcut, "4×8 只有 DC 时 dc={dc}");

            // 4×4
            let mut blk = b;
            let mut full = vec![0u8; 8 * 8];
            inv_trans_4x4(&mut full, 8, &mut blk, 0, 0);
            let mut shortcut = vec![0u8; 8 * 8];
            inv_trans_4x4_dc(&mut shortcut, 8, &b, 0, 0);
            assert_eq!(full, shortcut, "4×4 只有 DC 时 dc={dc}");
        }
    }

    /// ABT 的子块是「同一块缓冲 + 偏移」取出来的：偏移版的 4×4 与把子块搬到
    /// 缓冲开头再变换，结果必须逐像素相同。
    #[test]
    fn abt_subblock_offset_matches_a_copied_block() {
        let mut src = [0i16; 64];
        for (i, v) in src.iter_mut().enumerate() {
            *v = ((i as i32 * 37) % 401 - 200) as i16;
        }
        for off in [0usize, 4, 16, 20] {
            let mut a = vec![128u8; 16 * 16];
            let mut b = vec![128u8; 16 * 16];
            let mut blk = src;
            inv_trans_4x4(&mut a, 16, &mut blk[off..], 3, 5);
            // 把子块搬到开头
            let mut moved = [0i16; 64];
            for r in 0..4 {
                for c in 0..4 {
                    moved[r * 8 + c] = src[(off / 8 + r) * 8 + (off % 8) + c];
                }
            }
            inv_trans_4x4(&mut b, 16, &mut moved, 3, 5);
            assert_eq!(a, b, "off={off}");
        }
    }

    /// 三个写回函数的差别就是「覆盖 / +128 / 相加」。
    #[test]
    fn pixel_writers_differ_exactly_as_documented() {
        let mut block = [0i16; 64];
        block[0] = 200; // > 127
        let mut a = vec![10u8; 64];
        put_pixels_clamped(&block, &mut a, 8, 0, 0);
        assert_eq!(a[0], 200);
        let mut b = vec![10u8; 64];
        put_signed_pixels_clamped(&block, &mut b, 8, 0, 0);
        assert_eq!(b[0], 255, "有符号写回会夹到 255");
        let mut c = vec![10u8; 64];
        add_pixels_clamped(&block, &mut c, 8, 0, 0);
        assert_eq!(c[0], 210, "相加写回是叠加");
    }
}
