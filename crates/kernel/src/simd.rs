//! **共享的 SIMD 内核**（NEON + 标量回退）。
//!
//! ⚠️ **这是本仓唯一允许 `unsafe` 的地方。** crate 根上挂着 `#![deny(unsafe_code)]`，
//! 只有本模块显式放开 —— 「别处不许写 unsafe」由编译器保证，不靠人记。
//!
//! ## 为什么这一个值得写（而解码器那批不值得）
//!
//! 解码器上试过一轮（半像素平均那几个内核），只快 1.3% —— 那边的热点是**串行位解析**，
//! 且内核按**行**调、一行 16 像素就是一条 NEON 指令。**编码器的运动估计正好相反**：
//! `sad_luma16_capped` 每个候选都是**一个 16×16 整块**（16 行 × 16 字节），每个宏块要评
//! 150+ 个候选 —— 调用开销被 16 行摊掉，向量宽度是真收益。
//!
//! ## 口径
//!
//! - **逐位等价**是硬要求：SAD 的值直接决定运动矢量，差 1 就是编码结果不同。
//!   所有实现（NEON / 标量）对同一输入必须给同一答案，测试里逐个对齐比。
//! - **提前退出的语义必须一字不差**：原实现每行算完就查一次 `cap`，命中就返回 `cap`。
//!   调用方只做 `s < best` 的比较、`best ≤ cap` 恒成立，所以「返回真值」与「返回 cap」
//!   等价 —— 但**绝不能返回小于真值的数**（那会让本该输的候选赢）。这里保持
//!   「每行查一次、命中返回 cap」的原语义，行为与旧实现逐位相同。
//! - 只在 `aarch64` 上走 NEON（目标是 macOS ARM 与 Android arm64）；别的架构落标量。

#![allow(unsafe_code)]

/// 一条 16×16 候选的 SAD（含半像素相位，插值在寄存器里做，不出临时数组）。
///
/// - `cur`：当前宏块（行优先 16×16，256 字节）。
/// - `refp`：参考平面；`base` 是参考块 (0,0) 的字节偏移，`pw` 是行距。
/// - `dxy`：bit0 = 水平半像素、bit1 = 垂直半像素（与 `put_pixels` 同一套约定）。
/// - `no_rounding`：`true` ⇒ 半像素用 `(a+b)>>1`（截断），`false` ⇒ `(a+b+1)>>1`；
///   全像素相位用 `(a+b+c+d+r2)>>2`，`r2` 取 1 / 2。
/// - `cap`：累计值一到就停、直接返回 `cap`（与旧实现同一语义）。
pub fn sad_block16(
    cur: &[u8; 256],
    refp: &[u8],
    pw: usize,
    base: usize,
    dxy: usize,
    no_rounding: bool,
    cap: u32,
) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: 调用方保证半像素相位最多读到 `base + (15+1)*pw + 16 + 1`，
        // 与旧实现的切片范围完全一致；`cur` 是定长 256。
        unsafe { neon::sad_block16(cur.as_ptr(), refp.as_ptr(), pw, base, dxy, no_rounding, cap) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar::sad_block16(cur, refp, pw, base, dxy, no_rounding, cap)
    }
}

/// 标量参考实现 —— 非 aarch64 平台**就是它**在跑，测试里也拿它当判据。
#[allow(dead_code)] // aarch64 上只有测试用它
mod scalar {
    pub fn sad_block16(
        cur: &[u8; 256],
        refp: &[u8],
        pw: usize,
        base: usize,
        dxy: usize,
        no_rounding: bool,
        cap: u32,
    ) -> u32 {
        let (hx, hy) = (dxy & 1 != 0, dxy & 2 != 0);
        let r1 = if no_rounding { 0u32 } else { 1u32 };
        let r2 = if no_rounding { 1u32 } else { 2u32 };
        let mut acc = 0u32;
        for r in 0..16usize {
            let row = base + r * pw;
            let crow = &cur[r * 16..r * 16 + 16];
            let mut rv = [0u8; 16];
            match (hx, hy) {
                (false, false) => rv.copy_from_slice(&refp[row..row + 16]),
                (true, false) => {
                    let t = &refp[row..row + 17];
                    for i in 0..16 {
                        rv[i] = ((t[i] as u32 + t[i + 1] as u32 + r1) >> 1) as u8;
                    }
                }
                (false, true) => {
                    let (t, b) = (&refp[row..row + 16], &refp[row + pw..row + pw + 16]);
                    for i in 0..16 {
                        rv[i] = ((t[i] as u32 + b[i] as u32 + r1) >> 1) as u8;
                    }
                }
                (true, true) => {
                    let t = &refp[row..row + 17];
                    let b = &refp[row + pw..row + pw + 17];
                    for i in 0..16 {
                        rv[i] = ((t[i] as u32
                            + t[i + 1] as u32
                            + b[i] as u32
                            + b[i + 1] as u32
                            + r2)
                            >> 2) as u8;
                    }
                }
            }
            for i in 0..16 {
                acc += crow[i].abs_diff(rv[i]) as u32;
            }
            if acc >= cap {
                return cap;
            }
        }
        acc
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    /// 一行的 16 通道绝对差之和（NEON 值进、标量出）。
    #[inline(always)]
    unsafe fn row_sad_v(cur: uint8x16_t, interp: uint8x16_t) -> u32 {
        // `vpadalq_u8`：把 u8x16 相邻两两相加、累进 u16x8（单行最多 16×255=4080，不溢出）。
        unsafe {
            let d = vabdq_u8(cur, interp);
            vaddvq_u16(vpadalq_u8(vdupq_n_u16(0), d)) as u32
        }
    }

    /// 半像素插值在**寄存器里**做：旧实现先写 `rv[16]` 再读一遍，这里省掉两趟内存。
    /// `vhaddq`（截断）/`vrhaddq`（+1 舍入）与 `(a+b+r1)>>1` 逐位相同。
    #[inline(always)]
    unsafe fn hpel2(a: uint8x16_t, b: uint8x16_t, no_rounding: bool) -> uint8x16_t {
        unsafe {
            if no_rounding {
                vhaddq_u8(a, b)
            } else {
                vrhaddq_u8(a, b)
            }
        }
    }

    /// 全像素相位：`(a+b+c+d+r2)>>2`。
    ///
    /// ⚠️ 四个数的平均**必须用 16 位宽算**：两次 `vrhadd` 会差 1
    /// （`((a+b+1)>>1 + (c+d+1)>>1 + 1)>>1 != (a+b+c+d+2)>>2`）。
    #[inline(always)]
    unsafe fn hpel4(
        a: uint8x16_t,
        b: uint8x16_t,
        c: uint8x16_t,
        d: uint8x16_t,
        r2: uint16x8_t,
    ) -> uint8x16_t {
        unsafe {
            let lo = vaddq_u16(
                vaddq_u16(
                    vaddl_u8(vget_low_u8(a), vget_low_u8(b)),
                    vaddl_u8(vget_low_u8(c), vget_low_u8(d)),
                ),
                r2,
            );
            let hi = vaddq_u16(vaddq_u16(vaddl_high_u8(a, b), vaddl_high_u8(c, d)), r2);
            vcombine_u8(
                vmovn_u16(vshrq_n_u16::<2>(lo)),
                vmovn_u16(vshrq_n_u16::<2>(hi)),
            )
        }
    }

    pub unsafe fn sad_block16(
        cur: *const u8,
        refp: *const u8,
        pw: usize,
        base: usize,
        dxy: usize,
        no_rounding: bool,
        cap: u32,
    ) -> u32 {
        let (hx, hy) = (dxy & 1 != 0, dxy & 2 != 0);
        let r2 = vdupq_n_u16(if no_rounding { 1 } else { 2 });
        let mut acc = 0u32;
        for r in 0..16usize {
            let row = base + r * pw;
            let crow = vld1q_u8(cur.add(r * 16));
            let p = refp.add(row);
            let sad = match (hx, hy) {
                (false, false) => row_sad_v(crow, vld1q_u8(p)),
                (true, false) => row_sad_v(
                    crow,
                    hpel2(vld1q_u8(p), vld1q_u8(p.add(1)), no_rounding),
                ),
                (false, true) => row_sad_v(
                    crow,
                    hpel2(vld1q_u8(p), vld1q_u8(p.add(pw)), no_rounding),
                ),
                (true, true) => {
                    let ip = hpel4(
                        vld1q_u8(p),
                        vld1q_u8(p.add(1)),
                        vld1q_u8(p.add(pw)),
                        vld1q_u8(p.add(pw + 1)),
                        r2,
                    );
                    row_sad_v(crow, ip)
                }
            };
            acc += sad;
            if acc >= cap {
                return cap;
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 便宜、确定的 PRNG（xorshift64）—— 测试里不引依赖。
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn sad_block16_matches_scalar() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        // 平面故意开得比块大、行距不是 16 的倍数 —— 行距不对齐才是真实情况。
        let (pw, ph) = (37usize, 24usize);
        let mut plane = vec![0u8; pw * ph];
        for b in plane.iter_mut() {
            *b = rng.next() as u8;
        }
        let mut cur = [0u8; 256];
        for b in cur.iter_mut() {
            *b = rng.next() as u8;
        }
        for dxy in 0usize..4 {
            for no_rounding in [false, true] {
                for (bx, by) in [(0usize, 0usize), (5usize, 3usize), (20usize, 7usize)] {
                    let base = by * pw + bx;
                    let got = sad_block16(&cur, &plane, pw, base, dxy, no_rounding, u32::MAX);
                    let want =
                        scalar::sad_block16(&cur, &plane, pw, base, dxy, no_rounding, u32::MAX);
                    assert_eq!(got, want, "dxy={dxy} nr={no_rounding} base=({bx},{by})");
                }
            }
        }
    }

    #[test]
    fn sad_block16_cap_semantics_match_scalar() {
        // cap 命中时**必须**返回 cap 本身（不是真值）—— 与旧实现同一语义。
        let mut rng = Rng(0xdead_beef_cafe_0001);
        let (pw, ph) = (33usize, 20usize);
        let mut plane = vec![0u8; pw * ph];
        for b in plane.iter_mut() {
            *b = rng.next() as u8;
        }
        let cur = [200u8; 256]; // 与随机平面差得远，保证提前退出真的发生
        for dxy in 0usize..4 {
            for cap in [10u32, 100u32, 1000u32, u32::MAX] {
                let got = sad_block16(&cur, &plane, pw, 0, dxy, false, cap);
                let want = scalar::sad_block16(&cur, &plane, pw, 0, dxy, false, cap);
                assert_eq!(got, want, "dxy={dxy} cap={cap}");
            }
        }
    }
}
