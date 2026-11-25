//! 原地 radix-2 复数 FFT（DIT，f64）—— **共享数值内核**。
//!
//! ## 为什么住在 `v2v-kernel`
//!
//! 按 AGENTS.md 的不变量 3：共享数值内核只有一份，谁都要用的东西**不许复刻第二份**。
//! 这个 FFT 有两个用户，而且两边都不该各写一份：
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
//! 位反转置换（查表）后，把 radix-2 的级**两两融合**成一趟，旋转因子按级的长度缓存。
//!
//! ## 为什么是「融合 + 查表」，而不是换算法
//!
//! 归因（`fft_throughput`，新旧同进程交替量）：
//!
//! | 手段 | n=512 收益 |
//! |---|---|
//! | 位反转：现算 `while j & bit` → 预计算配对表 | 0.63 → 0.19 µs（**这一项才是大头**） |
//! | 相邻两级 radix-2 融合进一趟 | 只值 ×1.04 |
//! | 两者合计 | **×1.36~1.41**（n=128…2048 都成立） |
//!
//! 最初看着像访存瓶颈（每蝶形 6 载 4 存、4.3 周期/蝶形），于是先做融合、把访存砍近一半
//! —— 只换来 4%。真相是浮点已经跑到 **3.4 flops/cycle**，贴着 FP64 标量上限；真正在赔钱
//! 的是位反转里那个**圈数不定、分支预测吃不住**的 `while`。
//!
//! 换成 radix-4 / split-radix 能少算 13~20% 的乘法，但会**改掉每一个输出值的最后几位**，
//! 于是 AAC / WMA / Vorbis 三条线已有的对账判据全部作废。融合不同：中间量留在寄存器里，
//! **浮点运算的次数、顺序、舍入全部不变**，所以产物**逐位相同** —— 由
//! `fusion_is_bit_identical_to_separate_stages` 钉住（比 `to_bits()`，不是比差值）。
//!
//! ## 还能再快吗
//!
//! f64 标量已经到底（`-C target-cpu=native` 能再快一点，但那是 FMA 改了舍入 + 不可移植）。
//! 再往上是 NEON 的 2 条 f64 lane（上限 2×），代价是要把本模块挪进 `simd` 的 `unsafe` 段、
//! 并让 vorbis / aac / wma 打开 `unsafe-kernels` —— 等于改掉音频线的「纯标量」属性。
//!
//! 搬到这里之后多了一条**直接对着 O(n²) 的 DFT** 钉的测试（`matches_direct_dft`）：
//! 它不依赖任何上层变换，换实现时先看它。

use std::{f64::consts::PI, sync::OnceLock};

// 4096 点覆盖 AAC 与 WMA 的变换，也覆盖 Vorbis 最长块（8192 → 2048 点 FFT）。
// 这些级的表合计约 64 KiB。
static TWIDDLES: [OnceLock<Vec<(f64, f64)>>; 12] = [const { OnceLock::new() }; 12];

fn twiddles(size: usize) -> Vec<(f64, f64)> {
    (0..size / 2)
        .map(|k| {
            let angle = -2.0 * PI * k as f64 / size as f64;
            (angle.cos(), angle.sin())
        })
        .collect()
}

/// 取第 `size` 级的旋转因子；4096 点以内共用缓存。
fn twiddles_for(size: usize) -> &'static [(f64, f64)] {
    let slot = size.trailing_zeros() as usize;
    assert!(slot >= 1, "FFT 长度必须是至少为 2 的二次幂，拿到 {size}");
    let idx = slot - 1;
    assert!(
        idx < TWIDDLES.len(),
        "{size} 点超出旋转因子缓存的 {0} 级上限",
        TWIDDLES.len()
    );
    TWIDDLES[idx].get_or_init(|| twiddles(size))
}

/// 位反转置换。
///
/// ## 为什么查表，而不是现算
///
/// 现算是「`while j & bit != 0 { j ^= bit; bit >>= 1 }`」，那个 `while` 的圈数
/// 取决于 `j` 的尾部连 1 个数 —— 规律复杂、分支预测器吃不住。实测（`fft_throughput`）
/// n=512 时它单独占 **约 0.6 µs，是全趟 FFT 的 25%**（4.45 周期/元素）。
///
/// 换成预先算好的「(i, j) 且 i < j」配对表：没有分支、没有位运算，只剩访存。
fn bit_reverse_permute(re: &mut [f64], im: &mut [f64]) {
    for &(i, j) in bit_reverse_pairs(re.len()) {
        let (i, j) = (i as usize, j as usize);
        re.swap(i, j);
        im.swap(i, j);
    }
}

/// 预先算好的位反转配对（只留 `i < j` 那一半）。
fn bit_reverse_pairs(len: usize) -> &'static [(u32, u32)] {
    static PAIRS: [OnceLock<Vec<(u32, u32)>>; 12] = [const { OnceLock::new() }; 12];
    assert!(len >= 2, "位反转至少要 2 点，拿到 {len}");
    let idx = len.trailing_zeros() as usize - 1;
    assert!(
        idx < PAIRS.len(),
        "{len} 点超出位反转表的 {0} 级上限",
        PAIRS.len()
    );
    PAIRS[idx].get_or_init(|| {
        let shift = len.trailing_zeros();
        let mut out = Vec::with_capacity(len / 2);
        for i in 0..len {
            let j = (i.reverse_bits() >> (usize::BITS - shift)) as u32;
            if (i as u32) < j {
                out.push((i as u32, j));
            }
        }
        out
    })
}

/// 单级 radix-2 蝶形：块长 `size`，同一块内前后两半配对，旋转因子为 `W_size^k`。
fn radix2_stage(re: &mut [f64], im: &mut [f64], size: usize) {
    let half = size / 2;
    let tw = twiddles_for(size);
    for (re_chunk, im_chunk) in re.chunks_exact_mut(size).zip(im.chunks_exact_mut(size)) {
        let (lo_re, hi_re) = re_chunk.split_at_mut(half);
        let (lo_im, hi_im) = im_chunk.split_at_mut(half);
        for (((a_re, a_im), (b_re, b_im)), &(w_re, w_im)) in lo_re
            .iter_mut()
            .zip(lo_im)
            .zip(hi_re.iter_mut().zip(hi_im))
            .zip(tw)
        {
            let tr = *b_re * w_re - *b_im * w_im;
            let ti = *b_re * w_im + *b_im * w_re;
            *b_re = *a_re - tr;
            *b_im = *a_im - ti;
            *a_re += tr;
            *a_im += ti;
        }
    }
}

/// 把 `size` 与 `2·size` 两级融合成一趟。
///
/// 一个 `2·size` 的块切成四个 `size/2` 长的四分区 Q0..Q3。第一级把 (Q0,Q1) 与
/// (Q2,Q3) 各自配对，第二级再把 (Q0,Q2) 与 (Q1,Q3) 配对。中间量只活在寄存器里。
///
/// 第二级在 `k + size/2` 处的旋转因子是 `W_{2·size}^{k + size/2}`，直接按偏移取，
/// 不靠 `×(-i)` 推 —— 推出来的值虽然数学上相同，但会引入一次额外的乘法舍入，
/// 那样就不是逐位相同了。
fn fuse_two_stages(re: &mut [f64], im: &mut [f64], size: usize) {
    let block = 2 * size;
    let q = size / 2;
    let tw_inner = twiddles_for(size); // W_size^k，q 项
    let tw_outer = twiddles_for(block); // W_{2·size}^k，size = 2q 项
    let (outer_lo, outer_hi) = tw_outer.split_at(q);

    for (re_chunk, im_chunk) in re.chunks_exact_mut(block).zip(im.chunks_exact_mut(block)) {
        let (r0, rest) = re_chunk.split_at_mut(q);
        let (r1, rest) = rest.split_at_mut(q);
        let (r2, r3) = rest.split_at_mut(q);
        let (i0, rest) = im_chunk.split_at_mut(q);
        let (i1, rest) = rest.split_at_mut(q);
        let (i2, i3) = rest.split_at_mut(q);

        // 四分区两两配成「内层要配对的两组」，再与三张旋转因子表一起 zip ——
        // 这样一次也不做边界检查。这是数值内核，值得为它写这么丑的迭代器。
        let lo = r0.iter_mut().zip(i0).zip(r1.iter_mut().zip(i1));
        let hi = r2.iter_mut().zip(i2).zip(r3.iter_mut().zip(i3));
        let twiddles = tw_inner.iter().zip(outer_lo).zip(outer_hi);
        for (((q0, q1), (q2, q3)), ((&(ws_re, ws_im), &(wl_re, wl_im)), &(wh_re, wh_im))) in
            lo.zip(hi).zip(twiddles)
        {
            let ((r0k, i0k), (r1k, i1k)) = (q0, q1);
            let ((r2k, i2k), (r3k, i3k)) = (q2, q3);
            // 第一级：(Q0[k], Q1[k]) 与 (Q2[k], Q3[k])，同一个 W_size^k。
            let t0r = *r1k * ws_re - *i1k * ws_im;
            let t0i = *r1k * ws_im + *i1k * ws_re;
            let p0r = *r0k + t0r;
            let p0i = *i0k + t0i;
            let p1r = *r0k - t0r;
            let p1i = *i0k - t0i;

            let t2r = *r3k * ws_re - *i3k * ws_im;
            let t2i = *r3k * ws_im + *i3k * ws_re;
            let p2r = *r2k + t2r;
            let p2i = *i2k + t2i;
            let p3r = *r2k - t2r;
            let p3i = *i2k - t2i;

            // 第二级：(P0, P2) 用 W_block^k，(P1, P3) 用 W_block^{k+q}。
            let u0r = p2r * wl_re - p2i * wl_im;
            let u0i = p2r * wl_im + p2i * wl_re;
            *r0k = p0r + u0r;
            *i0k = p0i + u0i;
            *r2k = p0r - u0r;
            *i2k = p0i - u0i;

            let u1r = p3r * wh_re - p3i * wh_im;
            let u1i = p3r * wh_im + p3i * wh_re;
            *r1k = p1r + u1r;
            *i1k = p1i + u1i;
            *r3k = p1r - u1r;
            *i3k = p1i - u1i;
        }
    }
}

/// 原地 radix-2 复数 FFT（DIT）。`re` 与 `im` 等长，长度必须是 2 的幂。
///
/// 旋转因子按级缓存，**上限 4096 点**（12 级）。这是硬边界：所有调用方最长的
/// 变换是 Vorbis 8192 块对应的 2048 点，超了就是调用方搞错了，所以直接 panic
/// 而不是每次调用重新算表 —— 「静默变慢」比「当场报错」难查得多。
///
/// 级数是奇数时最后剩一级，单独跑一次 `radix2_stage`。
pub fn fft_in_place(re: &mut [f64], im: &mut [f64]) {
    let len = re.len();
    // 这两个是**真**断言而不是 `debug_assert`：release 下长度不对会算出静默的垃圾，
    // 而 `zip` 还会把多余的部分悄悄截掉。一次比较换掉一整类难查的错，值。
    assert!(len.is_power_of_two(), "FFT 长度必须是二次幂，拿到 {len}");
    assert_eq!(len, im.len(), "re / im 长度不一致：{len} vs {}", im.len());
    if len < 2 {
        return;
    }

    bit_reverse_permute(re, im);

    let mut size = 2usize;
    while size * 2 <= len {
        fuse_two_stages(re, im, size);
        size <<= 2;
    }
    if size <= len {
        radix2_stage(re, im, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 归因用：FFT 本体与位反转各占多少。
    ///
    /// 判据不是「快不快」，是「往哪儿使劲」—— 位反转占得少就别去动它。
    /// `cargo test -p v2v-kernel --release --lib fft_throughput -- --ignored --nocapture`
    #[test]
    #[ignore = "归因用"]
    fn fft_throughput() {
        use std::time::Instant;

        // 新旧两版**交替**跑、各自取最快一轮：机器上可能还有别的会话在抢 CPU，
        // 先量完一边再量另一边会把漂移算进比值里（n=128 一度量出 ×1.01 和 ×2.08
        // 两个互相矛盾的数，就是因为这个）。
        fn duel(
            n: usize,
            reps: usize,
            rounds: usize,
            mut new: impl FnMut(&mut [f64], &mut [f64]),
            mut old: impl FnMut(&mut [f64], &mut [f64]),
        ) -> (f64, f64) {
            let base_re: Vec<f64> = (0..n).map(|i| ((i * 37 % 101) as f64) * 1e-2).collect();
            let base_im: Vec<f64> = (0..n).map(|i| ((i * 53 % 97) as f64) * 1e-2).collect();
            let (mut re, mut im) = (base_re.clone(), base_im.clone());
            let (mut bn, mut bo) = (f64::INFINITY, f64::INFINITY);
            for _ in 0..rounds {
                // 每轮从同一份输入重来：否则数值会滚到 inf，量到的就不是同一个东西。
                re.copy_from_slice(&base_re);
                im.copy_from_slice(&base_im);
                let t = Instant::now();
                for _ in 0..reps {
                    new(&mut re, &mut im);
                }
                bn = bn.min(t.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(&re);

                re.copy_from_slice(&base_re);
                im.copy_from_slice(&base_im);
                let t = Instant::now();
                for _ in 0..reps {
                    old(&mut re, &mut im);
                }
                bo = bo.min(t.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(&re);
            }
            (bn * 1e3 / reps as f64, bo * 1e3 / reps as f64)
        }

        for &n in &[128usize, 256, 512, 1024, 2048] {
            // 小点数单趟太快，多跑几遍才压得住噪声。
            let reps = if n <= 256 { 8000 } else { 2000 };
            let (new, old) = duel(n, reps, 7, fft_in_place, fft_stage_by_stage);
            let (perm_new, perm_old) =
                duel(n, reps, 5, bit_reverse_permute, bit_reverse_permute_slow);
            eprintln!(
                "fft n={n:5}: 融合+查表 {new:6.2} µs | 逐级+现算 {old:6.2} µs | ×{:.2}   \
                 位反转 {perm_new:.2} / {perm_old:.2} µs",
                old / new
            );
        }
    }

    /// **逐位**参考实现：位反转之后一级一级地跑，不做融合。
    ///
    /// 这就是融合之前的生产代码（连位反转都是原来那个现算版本）。融合只该省访存，
    /// 不该动任何一个舍入 —— 所以判据是 `to_bits()` 全等，不是「差值很小」。
    fn bit_reverse_permute_slow(re: &mut [f64], im: &mut [f64]) {
        let len = re.len();
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
    }

    fn fft_stage_by_stage(re: &mut [f64], im: &mut [f64]) {
        bit_reverse_permute_slow(re, im);
        let len = re.len();
        let mut size = 2usize;
        while size <= len {
            radix2_stage(re, im, size);
            size <<= 1;
        }
    }

    #[test]
    fn fusion_is_bit_identical_to_separate_stages() {
        // 覆盖 2^1..2^11（奇数级与偶数级两条路径都要走到）。
        for shift in 1..=11u32 {
            let n = 1usize << shift;
            let mut state = 0x9e37_79b9u32 ^ n as u32;
            let mut next = || {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 8) as f64 / 65536.0 - 1.0
            };
            let re: Vec<f64> = (0..n).map(|_| next() * 1000.0).collect();
            let im: Vec<f64> = (0..n).map(|_| next() * 1000.0).collect();

            let (mut want_re, mut want_im) = (re.clone(), im.clone());
            fft_stage_by_stage(&mut want_re, &mut want_im);
            let (mut got_re, mut got_im) = (re.clone(), im.clone());
            fft_in_place(&mut got_re, &mut got_im);

            for k in 0..n {
                assert_eq!(
                    got_re[k].to_bits(),
                    want_re[k].to_bits(),
                    "n={n}：实部第 {k} 项不是逐位相同（融合改了舍入）"
                );
                assert_eq!(
                    got_im[k].to_bits(),
                    want_im[k].to_bits(),
                    "n={n}：虚部第 {k} 项不是逐位相同（融合改了舍入）"
                );
            }
        }
    }

    /// 非二次幂长度是调用方的错，不该被静默吞掉。
    #[test]
    #[should_panic(expected = "二次幂")]
    fn rejects_a_length_that_is_not_a_power_of_two() {
        let (mut re, mut im) = (vec![0.0; 6], vec![0.0; 6]);
        fft_in_place(&mut re, &mut im);
    }

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
        assert!(
            rel < 1e-12,
            "共轭技巧没给出反号方向的 DFT：相对偏差 {rel:.3e}"
        );
    }
}
