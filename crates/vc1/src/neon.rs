//! MC 插值的 NEON 内核（**仅 aarch64**）。
//!
//! ## 为什么值得写
//!
//! 采样里 `mc_1mv`（¼ 像素插值）+ `chroma_mc` + `load_ref` 占解码 **~37%**，
//! 而这四条路径全是「固定系数 × 连续内存」——正是 NEON 的甜点区。
//!
//! ## 位精确是硬约束
//!
//! NEON 版本与标量版必须**逐位一致**（`tests/mc_neon.rs` 随机参数 A/B +
//! 真样本逐像素 0 差两道判据都在盯）：
//! - 合流竖直趟的中间结果**不夹取**（i16 直存，上游如此；夹了在陡边差 1）；
//! - 水平趟用 **i32** 累加（53 × 16320 远超 i16，标量就是 i32）；
//! - 取整/移位常数逐条对齐标量（`mspel_filter` 按 mode 分 `(bias, shift)`）。
//!
//! 统一 **8 通道**结构：`u8x8` 载入 → `u16`/`i32` 加宽 → 乘加 → 移位 → 夹取 →
//! 窄化回 `u8x8`。
#![allow(unsafe_code)]

use crate::mc::BUF_STRIDE;

/// 标量参照：`mc.rs::mspel_filter` 的同构（余量像素 / A/B 测试都用它）。
#[inline(always)]
fn mspel_scalar(s: impl Fn(isize) -> i32, mode: u32, r: i32) -> i32 {
    match mode {
        1 => (-4 * s(-1) + 53 * s(0) + 18 * s(1) - 3 * s(2) + 32 - r) >> 6,
        2 => (-1 * s(-1) + 9 * s(0) + 9 * s(1) - 1 * s(2) + 8 - r) >> 4,
        _ => (-3 * s(-1) + 18 * s(0) + 53 * s(1) - 4 * s(2) + 32 - r) >> 6,
    }
}

/// 模式系数：`(c0, c1, c2, c3)` —— mode 1/3 是 ¼ 位移那组、mode 2 是 ½ 位移。
#[inline(always)]
fn taps(mode: u32) -> (i32, i32, i32, i32) {
    match mode {
        1 => (-4, 53, 18, -3),
        2 => (-1, 9, 9, -1),
        _ => (-3, 18, 53, -4),
    }
}

/// `mspel_filter` 的 `(bias, shift)`：mode 2 是 `(8-r, 4)`，其余 `(32-r, 6)`。
#[inline(always)]
fn bias_shift(mode: u32, r: i32) -> (i32, i32) {
    if mode == 2 {
        (8 - r, 4)
    } else {
        (32 - r, 6)
    }
}

/// 把 `u8x8` 零扩展成 `(i32x4 低, i32x4 高)`。
///
/// # Safety
/// 调用方保证 NEON 可用（aarch64）。
#[cfg(target_arch = "aarch64")]
unsafe fn widen_u8(
    x: std::arch::aarch64::uint8x8_t,
) -> (
    std::arch::aarch64::int32x4_t,
    std::arch::aarch64::int32x4_t,
) {
    use std::arch::aarch64::*;
    let u16x8 = vmovl_u8(x);
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u16x8))),
        vreinterpretq_s32_u32(vmovl_high_u16(u16x8)),
    )
}

/// 把 `i16x8` 符号扩展成 `(i32x4 低, i32x4 高)`。
///
/// # Safety
/// 调用方保证 NEON 可用（aarch64）。
#[cfg(target_arch = "aarch64")]
unsafe fn widen_i16(
    x: std::arch::aarch64::int16x8_t,
) -> (
    std::arch::aarch64::int32x4_t,
    std::arch::aarch64::int32x4_t,
) {
    use std::arch::aarch64::*;
    (vmovl_s16(vget_low_s16(x)), vmovl_high_s16(x))
}

/// 4 抽头乘加：`acc = Σ c_k · x_k`（i32 域，两个 4 通道组）。
///
/// # Safety
/// 调用方保证 NEON 可用（aarch64）。
#[cfg(target_arch = "aarch64")]
unsafe fn mac4(
    t: (std::arch::aarch64::int32x4_t, std::arch::aarch64::int32x4_t),
    u: (std::arch::aarch64::int32x4_t, std::arch::aarch64::int32x4_t),
    v: (std::arch::aarch64::int32x4_t, std::arch::aarch64::int32x4_t),
    w: (std::arch::aarch64::int32x4_t, std::arch::aarch64::int32x4_t),
    c0: i32,
    c1: i32,
    c2: i32,
    c3: i32,
) -> (
    std::arch::aarch64::int32x4_t,
    std::arch::aarch64::int32x4_t,
) {
    use std::arch::aarch64::*;
    let mul = |x: (int32x4_t, int32x4_t), k: i32| -> (int32x4_t, int32x4_t) {
        let k4 = vdupq_n_s32(k);
        (vmulq_s32(x.0, k4), vmulq_s32(x.1, k4))
    };
    let a = mul(t, c0);
    let b = mul(u, c1);
    let c = mul(v, c2);
    let d = mul(w, c3);
    let add =
        |p: (int32x4_t, int32x4_t), q: (int32x4_t, int32x4_t)| -> (int32x4_t, int32x4_t) {
            (vaddq_s32(p.0, q.0), vaddq_s32(p.1, q.1))
        };
    add(add(a, b), add(c, d))
}

/// 右移 + 夹取 + 窄化：`((x + bias) >> shift).clamp(0, 255)` → `u8x8`。
///
/// # Safety
/// 调用方保证 NEON 可用（aarch64）。
#[cfg(target_arch = "aarch64")]
unsafe fn shift_clamp_narrow(
    lo: std::arch::aarch64::int32x4_t,
    hi: std::arch::aarch64::int32x4_t,
    bias: i32,
    shift: i32,
) -> std::arch::aarch64::uint8x8_t {
    use std::arch::aarch64::*;
    let b4 = vdupq_n_s32(bias);
    // 运行时移位：vshlq 的负数 = 右移。
    let sh = vdupq_n_s32(-shift);
    let l = vshlq_s32(vaddq_s32(lo, b4), sh);
    let h = vshlq_s32(vaddq_s32(hi, b4), sh);
    let zero = vdupq_n_s32(0);
    let v255 = vdupq_n_s32(255);
    let l = vminq_s32(vmaxq_s32(l, zero), v255);
    let h = vminq_s32(vmaxq_s32(h, zero), v255);
    vqmovun_s16(vcombine_s16(vqmovn_s32(l), vqmovn_s32(h)))
}

/// 合流两趟的**竖直趟**：`src` 4 行 × `n = w+3` 列 → i16 中间结果（**不夹取**）。
/// 对应标量：`tmp[j*(w+3)+i] = ((v + r) >> shift) as i16`。
///
/// `row0` 已含「整体左移一列」的偏移（见 `mc.rs` 调用点）。
#[cfg(target_arch = "aarch64")]
pub fn vertical_pass(
    src: &[u8],
    row0: usize,
    n: usize,
    vmode: u32,
    shift: i32,
    r: i32,
    tmp: &mut [i16],
) {
    let (c0, c1, c2, c3) = taps(vmode);
    unsafe {
        use std::arch::aarch64::*;
        let mut i = 0usize;
        while i + 8 <= n {
            let a = vld1_u8(src.as_ptr().add(row0 + i));
            let b = vld1_u8(src.as_ptr().add(row0 + i + BUF_STRIDE));
            let c = vld1_u8(src.as_ptr().add(row0 + i + 2 * BUF_STRIDE));
            let d = vld1_u8(src.as_ptr().add(row0 + i + 3 * BUF_STRIDE));
            let (lo, hi) = mac4(
                widen_u8(a),
                widen_u8(b),
                widen_u8(c),
                widen_u8(d),
                c0,
                c1,
                c2,
                c3,
            );
            let r4 = vdupq_n_s32(r);
            let sh = vdupq_n_s32(-shift);
            let l = vshlq_s32(vaddq_s32(lo, r4), sh);
            let h = vshlq_s32(vaddq_s32(hi, r4), sh);
            // 标量是截断 `as i16`；|v| ≤ 16320 + r 远在 i16 域内，饱和窄化取同值。
            let p = vcombine_s16(vqmovn_s32(l), vqmovn_s32(h));
            vst1q_s16(tmp.as_mut_ptr().add(i), p);
            i += 8;
        }
        while i < n {
            let s = |k: isize| {
                i32::from(src[(row0 as isize + i as isize + k * BUF_STRIDE as isize) as usize])
            };
            let (c0, c1, c2, c3) = taps(vmode);
            let v = c0 * s(0) + c1 * s(1) + c2 * s(2) + c3 * s(3);
            tmp[i] = ((v + r) >> shift) as i16;
            i += 1;
        }
    }
}

/// 合流两趟的**水平趟**：tmp 4 列抽头 → u8 输出（clamp 0..255）。
/// 对应标量：`dst = ((h + r2) >> 7).clamp(0, 255)`，h 用 **i32** 累加。
///
/// `t` 是本行第一个输出的下标（抽头从 `t-1` 起，共 w 列）。
#[cfg(target_arch = "aarch64")]
pub fn horizontal_pass(
    tmp: &[i16],
    t: usize,
    w: usize,
    hmode: u32,
    r2: i32,
    dst: &mut [u8],
    drow: usize,
) {
    let (c0, c1, c2, c3) = taps(hmode);
    unsafe {
        use std::arch::aarch64::*;
        let mut i = 0usize;
        while i + 8 <= w {
            // ⚠️ 抽头从 `t-1` 起（标量是 `tmp[o-1..o+2]`）—— 第一版从 `t` 起
            // 差了一列，A/B 测试当场抓住（0 vs 255）。
            // t ≥ 1，`t + i + k` 恒非负 —— 用下标而不是 offset（顺带免 unsafe）。
            let ld = |k: isize| vld1q_s16(tmp.as_ptr().add((t as isize + i as isize + k) as usize));
            let (lo, hi) = mac4(
                widen_i16(ld(-1)),
                widen_i16(ld(0)),
                widen_i16(ld(1)),
                widen_i16(ld(2)),
                c0,
                c1,
                c2,
                c3,
            );
            let r4 = vdupq_n_s32(r2);
            let l = vshrq_n_s32(vaddq_s32(lo, r4), 7);
            let h = vshrq_n_s32(vaddq_s32(hi, r4), 7);
            let zero = vdupq_n_s32(0);
            let v255 = vdupq_n_s32(255);
            let l = vminq_s32(vmaxq_s32(l, zero), v255);
            let h = vminq_s32(vmaxq_s32(h, zero), v255);
            let u = vqmovun_s16(vcombine_s16(vqmovn_s32(l), vqmovn_s32(h)));
            vst1_u8(dst.as_mut_ptr().add(drow + i), u);
            i += 8;
        }
        while i < w {
            let o = t + i;
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
            dst[drow + i] = ((h + r2) >> 7).clamp(0, 255) as u8;
            i += 1;
        }
    }
}

/// 单向竖直（hmode == 0）：4 行抽头 → u8（clamp 0..255）。
/// 对应标量 `mspel_filter(src, off, BUF_STRIDE, vmode, r)`，`r = 1 - rnd`。
#[cfg(target_arch = "aarch64")]
pub fn single_vertical(
    src: &[u8],
    row0: usize,
    w: usize,
    vmode: u32,
    r: i32,
    dst: &mut [u8],
    drow: usize,
) {
    let (c0, c1, c2, c3) = taps(vmode);
    let (bias, shift) = bias_shift(vmode, r);
    unsafe {
        use std::arch::aarch64::*;
        let mut i = 0usize;
        while i + 8 <= w {
            // ⚠️ 系数 c0 打在 **s(-1)** 上：行基址要上移一行（第一版差一行，
            // A/B 测试当场抓住：224 vs 87）。
            let a = vld1_u8(src.as_ptr().add(row0 - BUF_STRIDE + i));
            let b = vld1_u8(src.as_ptr().add(row0 + i));
            let c = vld1_u8(src.as_ptr().add(row0 + i + BUF_STRIDE));
            let d = vld1_u8(src.as_ptr().add(row0 + i + 2 * BUF_STRIDE));
            let (lo, hi) = mac4(
                widen_u8(a),
                widen_u8(b),
                widen_u8(c),
                widen_u8(d),
                c0,
                c1,
                c2,
                c3,
            );
            let u = shift_clamp_narrow(lo, hi, bias, shift);
            vst1_u8(dst.as_mut_ptr().add(drow + i), u);
            i += 8;
        }
        while i < w {
            let off = (row0 + i) as isize;
            let s = |k: isize| i32::from(src[(off + k * BUF_STRIDE as isize) as usize]);
            dst[drow + i] = mspel_scalar(s, vmode, r).clamp(0, 255) as u8;
            i += 1;
        }
    }
}

/// 单向水平（vmode == 0）：一行内 4 列抽头 → u8。
/// 对应标量 `mspel_filter(src, off, 1, hmode, rnd)`。
#[cfg(target_arch = "aarch64")]
pub fn single_horizontal(
    src: &[u8],
    row: usize,
    w: usize,
    hmode: u32,
    rnd: i32,
    dst: &mut [u8],
    drow: usize,
) {
    let (c0, c1, c2, c3) = taps(hmode);
    let (bias, shift) = bias_shift(hmode, rnd);
    unsafe {
        use std::arch::aarch64::*;
        let mut i = 0usize;
        while i + 8 <= w {
            let base = row + i;
            let a = vld1_u8(src.as_ptr().add(base - 1));
            let b = vld1_u8(src.as_ptr().add(base));
            let c = vld1_u8(src.as_ptr().add(base + 1));
            let d = vld1_u8(src.as_ptr().add(base + 2));
            let (lo, hi) = mac4(
                widen_u8(a),
                widen_u8(b),
                widen_u8(c),
                widen_u8(d),
                c0,
                c1,
                c2,
                c3,
            );
            let u = shift_clamp_narrow(lo, hi, bias, shift);
            vst1_u8(dst.as_mut_ptr().add(drow + i), u);
            i += 8;
        }
        while i < w {
            let off = (row + i) as isize;
            let s = |k: isize| i32::from(src[(off + k) as usize]);
            dst[drow + i] = mspel_scalar(s, hmode, rnd).clamp(0, 255) as u8;
            i += 1;
        }
    }
}

/// 色度 8×8 双线性：`(a*o + b*(o+1) + c*(o+S) + d*(o+S+1) + bias) >> 6`。
/// 全程非负（u16 内不溢出：64×255 + 32 < 65536），一条 unsigned 流水到底。
#[cfg(target_arch = "aarch64")]
pub fn chroma_mc8(
    dst: &mut [u8],
    dstride: usize,
    dx: usize,
    dy: usize,
    src: &[u8],
    base: usize,
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    bias: u32,
) {
    unsafe {
        use std::arch::aarch64::*;
        for j in 0..8usize {
            let o = base + j * BUF_STRIDE;
            let p0 = vld1_u8(src.as_ptr().add(o));
            let p1 = vld1_u8(src.as_ptr().add(o + 1));
            let p2 = vld1_u8(src.as_ptr().add(o + BUF_STRIDE));
            let p3 = vld1_u8(src.as_ptr().add(o + BUF_STRIDE + 1));
            let mut acc = vmull_u8(p0, vdup_n_u8(a as u8));
            acc = vmlal_u8(acc, p1, vdup_n_u8(b as u8));
            acc = vmlal_u8(acc, p2, vdup_n_u8(c as u8));
            acc = vmlal_u8(acc, p3, vdup_n_u8(d as u8));
            acc = vaddq_u16(acc, vdupq_n_u16(bias as u16));
            let out = vshrn_n_u16(acc, 6);
            vst1_u8(dst.as_mut_ptr().add((dy + j) * dstride + dx), out);
        }
    }
}

/// hpel 半像素行：`(x + y + r2) >> 1`（r2 ∈ {0,1}，hadd / rhadd 一次到位）。
#[cfg(target_arch = "aarch64")]
pub fn hpel_avg2_row(src: &[u8], s: usize, dst: &mut [u8], d: usize, w: usize, round: bool) {
    unsafe {
        use std::arch::aarch64::*;
        let mut i = 0usize;
        while i + 16 <= w {
            let x = vld1q_u8(src.as_ptr().add(s + i));
            let y = vld1q_u8(src.as_ptr().add(s + i + 1));
            let o = if round {
                vrhaddq_u8(x, y)
            } else {
                vhaddq_u8(x, y)
            };
            vst1q_u8(dst.as_mut_ptr().add(d + i), o);
            i += 16;
        }
        while i < w {
            let r2 = i32::from(round);
            dst[d + i] = ((i32::from(src[s + i]) + i32::from(src[s + i + 1]) + r2) >> 1) as u8;
            i += 1;
        }
    }
}

/// hpel 四抽头行：`(a + b + c + d + r4) >> 2`（r4 ∈ {1,2}）。
/// u16 域求和（4×255 + 2 不溢出），右移两位再窄化 —— 与标量 i32 逐位一致。
#[cfg(target_arch = "aarch64")]
pub fn hpel_avg4_row(src: &[u8], s: usize, dst: &mut [u8], d: usize, w: usize, r4: u32) {
    unsafe {
        use std::arch::aarch64::*;
        let bias = vdupq_n_u16(r4 as u16);
        let mut i = 0usize;
        while i + 16 <= w {
            let t0 = vld1q_u8(src.as_ptr().add(s + i));
            let t1 = vld1q_u8(src.as_ptr().add(s + i + 1));
            let t2 = vld1q_u8(src.as_ptr().add(s + i + BUF_STRIDE));
            let t3 = vld1q_u8(src.as_ptr().add(s + i + BUF_STRIDE + 1));
            let ab = vaddl_u8(vget_low_u8(t0), vget_low_u8(t1));
            let cd = vaddl_u8(vget_low_u8(t2), vget_low_u8(t3));
            let abh = vaddl_high_u8(t0, t1);
            let cdh = vaddl_high_u8(t2, t3);
            let lo = vshrq_n_u16(vaddq_u16(vaddq_u16(ab, cd), bias), 2);
            let hi = vshrq_n_u16(vaddq_u16(vaddq_u16(abh, cdh), bias), 2);
            vst1q_u8(
                dst.as_mut_ptr().add(d + i),
                vcombine_u8(vmovn_u16(lo), vmovn_u16(hi)),
            );
            i += 16;
        }
        while i < w {
            let o = s + i;
            dst[d + i] = ((i32::from(src[o])
                + i32::from(src[o + 1])
                + i32::from(src[o + BUF_STRIDE])
                + i32::from(src[o + BUF_STRIDE + 1])
                + r4 as i32)
                >> 2) as u8;
            i += 1;
        }
    }
}
