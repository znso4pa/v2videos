//! 原地 radix-2 复数 FFT（DIT，f64）—— **共享数值内核**。
//!
//! ## 为什么住在 `v2v-msmpeg4`
//!
//! 按 AGENTS.md 的不变量 3：`v2v-msmpeg4` 兼共享数值内核（`BitReader` / `BitWriter` /
//! FDCT / IDCT / VLC 表……），谁都要用的东西**不许复刻第二份**。这个 FFT 现在有两个
//! 用户，而且两边都不该各写一份：
//!
//! - **`v2v-aac`**：前向 MDCT —— `预旋转 → 2N 点 FFT → 后旋转 → 取实部 ×2`；
//! - **`v2v-wma`**：IMDCT —— `预旋转 → 2n 点 FFT → 后旋转 → 三段对称展开`。
//!
//! 两个变换的相位与定标**各自留在自己的 crate 里**（它们不一样），共用的只有这个核。
//!
//! ## 约定
//!
//! `X[k] = Σ_j x[j]·e^{-2πijk/len}`（教科书方向）。要**反号**那个方向（`+`）时用
//! 「输入取共轭 → FFT → 输出取共轭」拿 —— `v2v-wma` 的 `imdct` 就是这么做的
//! （它把共轭直接折进预旋转的相位里，省一趟）。
//!
//! ## 先要正确，再谈快
//!
//! 教科书版：位反转置换 + 逐级蝶形，每级现算 `half` 个旋转因子（`e^{-2πik/size}`）。
//! 曾经想把旋转因子换成递推，结果跨 block 的状态没重置 —— AAC 那边的
//! `fast_mdct_matches_the_definition` 当场抓住（相对偏差 2e13）。
//!
//! 搬到这里之后多了一条**直接对着 O(n²) 的 DFT** 钉的测试（`matches_direct_dft`）：
//! 它不依赖任何上层变换，换实现时先看它。

use std::f64::consts::PI;

/// 原地 radix-2 复数 FFT（DIT）。`re` 与 `im` 等长，长度必须是 2 的幂。
///
/// 位反转置换 → 逐级蝶形。旋转因子**每级现算**（`cos`/`sin`），不建全局表：
/// 一次调用是 `log2(len)` 级、合计 `len/2` 个旋转因子 —— 与「递推生成」相比多花
/// 一点三角函数，换掉的是「跨调用状态」这类难查的错。这条在 AAC 与 WMA 两条
/// 对账链路上都量过，不是瓶颈（见各自 crate 里的性能记录）。
pub fn fft_in_place(re: &mut [f64], im: &mut [f64]) {
    let len = re.len();
    debug_assert!(len.is_power_of_two());
    debug_assert_eq!(len, im.len());

    // 位反转置换
    let mut j = 0usize;
    for i in 1..len {
        let mut bit = len >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // 逐级蝶形
    let mut size = 2usize;
    while size <= len {
        let half = size / 2;
        let tw: Vec<(f64, f64)> = (0..half)
            .map(|k| {
                let a = -2.0 * PI * (k as f64) / (size as f64);
                (a.cos(), a.sin())
            })
            .collect();
        let mut start = 0usize;
        while start < len {
            for (k, &(w_re, w_im)) in tw.iter().enumerate() {
                let a = start + k;
                let b = a + half;
                let tr = re[b] * w_re - im[b] * w_im;
                let ti = re[b] * w_im + im[b] * w_re;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
            start += size;
        }
        size <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 直接按定义算的 DFT —— oracle。`X[k] = Σ_j x[j]·e^{-2πijk/n}`。
    fn direct_dft(re: &[f64], im: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = re.len();
        let mut out_re = vec![0f64; n];
        let mut out_im = vec![0f64; n];
        for k in 0..n {
            let mut s_re = 0f64;
            let mut s_im = 0f64;
            for j in 0..n {
                let a = -2.0 * PI * (k * j) as f64 / n as f64;
                let (c, s) = (a.cos(), a.sin());
                s_re += re[j] * c - im[j] * s;
                s_im += re[j] * s + im[j] * c;
            }
            out_re[k] = s_re;
            out_im[k] = s_im;
        }
        (out_re, out_im)
    }

    #[test]
    fn matches_direct_dft() {
        for &n in &[8usize, 16, 128] {
            let mut state = 0x2468_ace1u32;
            let mut next = || {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 16) as f64 / 32768.0 - 1.0
            };
            let re: Vec<f64> = (0..n).map(|_| next() * 100.0).collect();
            let im: Vec<f64> = (0..n).map(|_| next() * 100.0).collect();

            let (want_re, want_im) = direct_dft(&re, &im);
            let (mut got_re, mut got_im) = (re.clone(), im.clone());
            fft_in_place(&mut got_re, &mut got_im);

            let peak = want_re
                .iter()
                .chain(want_im.iter())
                .fold(0f64, |m, v| m.max(v.abs()));
            let worst = (0..n)
                .map(|k| {
                    (got_re[k] - want_re[k])
                        .abs()
                        .max((got_im[k] - want_im[k]).abs())
                })
                .fold(0f64, f64::max);
            let rel = worst / peak.max(1e-30);
            assert!(rel < 1e-12, "n={n}：与直接 DFT 的最大相对偏差 {rel:.3e}");
        }
    }

    #[test]
    fn conjugate_trick_gives_the_opposite_sign_dft() {
        // `conj(FFT(conj(x)))` 是**反号方向**的那个变换（`X[k] = Σ_j x[j]·e^{+2πijk/n}`）
        // —— `v2v-wma` 的 IMDCT 正是靠它把 DCT-IV 写成一次正向 FFT。
        //
        // ⚠️ 别把它当「逆变换」：把它连做两次得到的是 `n·x[(−k) mod n]`（翻转），
        // 不是 `n·x` —— 第一版测试就是这么写错的，这条断言把它当场抓住。
        let n = 64usize;
        let mut state = 0x1357_9bdfu32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as f64 / 32768.0 - 1.0
        };
        let re: Vec<f64> = (0..n).map(|_| next()).collect();
        let im: Vec<f64> = (0..n).map(|_| next()).collect();

        let (mut got_re, mut got_im) = (re.clone(), im.clone());
        for v in got_im.iter_mut() {
            *v = -*v;
        }
        fft_in_place(&mut got_re, &mut got_im);
        for v in got_im.iter_mut() {
            *v = -*v;
        }

        // 直接算 `+` 号方向的 DFT 当 oracle。
        let mut worst = 0f64;
        let mut peak = 0f64;
        for k in 0..n {
            let mut s_re = 0f64;
            let mut s_im = 0f64;
            for j in 0..n {
                let a = 2.0 * PI * (k * j) as f64 / n as f64;
                let (c, s) = (a.cos(), a.sin());
                s_re += re[j] * c - im[j] * s;
                s_im += re[j] * s + im[j] * c;
            }
            worst = worst
                .max((got_re[k] - s_re).abs())
                .max((got_im[k] - s_im).abs());
            peak = peak.max(s_re.abs()).max(s_im.abs());
        }
        let rel = worst / peak.max(1e-30);
        assert!(rel < 1e-12, "共轭技巧没给出反号方向的 DFT：相对偏差 {rel:.3e}");
    }
}
