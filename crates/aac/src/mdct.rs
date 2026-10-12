//! MDCT（改进离散余弦变换）与 AAC 用的两种窗。
//!
//! ## 定标：与 ffmpeg 的 `mdct15` / `ff_mdct_init` 口径对齐
//!
//! AAC 的 MDCT 是 **2N 输入 → N 输出**，定义（ffmpeg 用的那一种）：
//!
//! ```text
//!   X[k] = Σ_{n=0}^{2N-1} x[n] · cos( π/N · (n + 1/2 + N/2) · (k + 1/2) ),   k = 0..N-1
//! ```
//!
//! **不加归一化系数** —— 定标全部由「量化 + 全局增益」那一趟吃掉，
//! 所以这里只要与解码端的 IMDCT 严格互为逆（也就是 50% 重叠相加后还原）即可。
//! 这条由 `round_trips_with_overlap` 测试钉住。
//!
//! ## 两种窗
//!
//! - **正弦窗**（`ff_sine_1024` / `ff_sine_128`）：`w[n] = sin( π/(2N) · (n + 1/2) )`。
//! - **KBD 窗**（`ff_aac_kbd_long_1024`）：Kaiser-Bessel derived，
//!   `ff_kbd_window_init(w, alpha, n)` 里 alpha 长窗是 **4.0**、短窗是 **6.0**。
//!
//! `use_kb_window` 是 ICS 里的一位，**每通道独立**。我们默认用正弦窗
//! （`use_kb_window = 0`）—— 它不需要 Bessel 函数，且 AAC 里用得很普遍。
//! KBD 的代码也留着（`kbd_window`），但默认不开。
//!
//! ## 为什么不用 FFT
//!
//! 直接按定义算是 O(N²)—— 对 1024 点来说是 100 万次乘加、每帧每通道。
//! 44.1 kHz 一个声道的音频每秒 43 帧，也就是每秒 4300 万次乘加 —— 现代 CPU 上
//! 这大约是几十毫秒，**可以接受**。换 FFT 会把这条链路变得难以对账，
//! 而这条链路的价值就在于「每一步都能手算验证」。

use std::f64::consts::PI;

/// 长块的样本数（MDCT 输出点数）。
pub const LONG_LEN: usize = 1024;

/// 正弦窗 `w[n] = sin( π/(2N) · (n + 1/2) )`，长度 `2N`。
///
/// AAC 用的是「半窗」——`sin` 的相位只走 π/2，所以两份窗相加（且一份倒序）恰好为 1。
/// 这是 MDCT 完美重建（TDAC）的前提。
pub fn sine_window(n: usize) -> Vec<f32> {
    (0..2 * n)
        .map(|i| (PI / (2.0 * n as f64) * (i as f64 + 0.5)).sin() as f32)
        .collect()
}

/// KBD（Kaiser-Bessel Derived）窗，与 `ff_kbd_window_init`（`libavcodec/kbdwin.c`）同口径。
///
/// `n` 是**块长**（AAC 长窗 = 1024），产出的窗长 `n`。
/// `alpha` 越大窗越「窄」（阻带衰减越好、主瓣越宽）：AAC 长窗用 4.0、短窗用 6.0。
///
/// 算法逐行对着 `kbd_window_init` 写：
///
/// ```text
///   alpha2 = 4 · (alpha·π/n)²
///   temp[i] = I0( sqrt( i·(n-i)·alpha2 ) )          i = 0..=n/2
///   scale   = 1 / ( Σ temp[i]·(1 + (i>0 && i<n/2)) + 1 )
///   w[i]    = sqrt( sum(i) · scale )                i = 0..=n/2
///   w[i]    = sqrt( sum(i) · scale )                i > n/2 时 sum 换成 Σ temp[n-i]
/// ```
///
/// ⚠️ 三个容易抄错的地方：
///
/// 1. `scale` 里的权重 `(1 + (i && i<n/2))` —— **两个端点算一次、中间算两次**
///    （窗是对称的，两端各只出现一次）。漏掉它就差半个窗的量级。
/// 2. 后半段（`i > n/2`）的累加和是 `Σ temp[n-i]`，**不是在 `temp` 上继续走**。
/// 3. `ff_kbd_window_init` 的 `n` 是**块长**，不是 `2N` —— 传 1024 得到 1024 点的窗，
///    而 AAC 的 MDCT 是 2N 点输入。窗在时域上要**两边各铺一次**
///    （前半上升、后半下降），见 `aac_window`。
pub fn kbd_window(n: usize, alpha: f64) -> Vec<f32> {
    let half = n / 2;
    let alpha2 = 4.0 * (alpha * std::f64::consts::PI / n as f64).powi(2);

    let mut temp = vec![0f64; half + 1];
    let mut scale = 0f64;
    for (i, t) in temp.iter_mut().enumerate() {
        *t = bessel_i0((i as f64 * (n - i) as f64 * alpha2).sqrt());
        // 两个端点各算一次，中间的算两次。
        scale += *t * if i > 0 && i < half { 2.0 } else { 1.0 };
    }
    let scale = 1.0 / (scale + 1.0);

    let mut w = vec![0f32; n];
    let mut sum = 0f64;
    for i in 0..=half {
        sum += temp[i];
        w[i] = (sum * scale).sqrt() as f32;
    }
    for i in (half + 1)..n {
        sum += temp[n - i];
        w[i] = (sum * scale).sqrt() as f32;
    }
    w
}

/// 把 `n` 点的 KBD 窗铺成 AAC 的 `2N` 点开窗表。
///
/// AAC 的 MDCT 吃 `2N` 个样本，窗也必须是 `2N` 长：
/// 前半 `n` 点直接铺、后半 `n` 点**倒序**铺（这样两份窗相加恰好为 1，即 TDAC）。
pub fn kbd_aac_window(n: usize, alpha: f64) -> Vec<f32> {
    let h = kbd_window(n, alpha);
    let mut w = vec![0f32; 2 * n];
    w[..n].copy_from_slice(&h);
    for i in 0..n {
        w[2 * n - 1 - i] = h[i];
    }
    w
}

/// 零阶修正贝塞尔函数 `I0(x)`（级数展开，`x` 不大时收敛很快）。
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    // `(x/2)^(2k) / (k!)^2`
    for k in 1..64 {
        term *= (x / 2.0) / k as f64;
        term *= (x / 2.0) / k as f64;
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

/// 直接法的 MDCT：`2N → N`，**无归一化**。
///
/// `input` 必须已经开过窗（长度 `2N`）。`out` 长度 `N`。
pub fn mdct(input: &[f32], out: &mut [f32]) {
    let n = out.len();
    debug_assert_eq!(input.len(), 2 * n);
    // 预先算好相位 —— 每帧重建 cos 表太亏（`cos` 是慢函数）。
    // `2N · N` 个 f64：N = 1024 时是 2M 个 = 16 MB，太大了，所以按行现算。
    let scale = PI / n as f64;
    for (k, o) in out.iter_mut().enumerate() {
        let base = scale * (k as f64 + 0.5);
        let mut s = 0f64;
        for (i, &x) in input.iter().enumerate() {
            s += x as f64 * (base * (i as f64 + 0.5 + n as f64 / 2.0)).cos();
        }
        *o = s as f32;
    }
}

/// 直接法的 IMDCT：`N → 2N`，**无归一化**。
///
/// 与 `mdct` 互为逆（差一个整体常数因子，由 50% 重叠相加的窗处理）。
/// 这里只用来做往返自测 —— 解码端的 IMDCT 不需要我们实现。
pub fn imdct(input: &[f32], out: &mut [f32]) {
    let n = input.len();
    debug_assert_eq!(out.len(), 2 * n);
    // 与 MDCT 的差：相位里的 `n/2` 从「输入侧」挪到「输出侧」，系数是 2/N。
    let scale = 2.0 / n as f64;
    for (i, o) in out.iter_mut().enumerate() {
        let mut s = 0f64;
        for (k, &x) in input.iter().enumerate() {
            s += x as f64
                * (PI / n as f64 * (i as f64 + 0.5 + n as f64 / 2.0) * (k as f64 + 0.5)).cos();
        }
        *o = (s * scale) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_window_satisfies_tdac() {
        // TDAC 的前提：`w[n]² + w[n+N]² == 1`（同一份窗、且倒序的那一份）。
        let n = 1024;
        let w = sine_window(n);
        for i in 0..n {
            let s = w[i] * w[i] + w[n + i] * w[n + i];
            assert!((s - 1.0).abs() < 1e-5, "第 {i} 项的平方和是 {s}");
        }
    }

    #[test]
    fn kbd_window_satisfies_tdac() {
        // `ff_kbd_window_init(.., 4.0, 1024)` 给的是 **1024 点**的窗，
        // 而 AAC 的 MDCT 吃 2048 个样本 —— 所以要铺成 2N 长再验 TDAC。
        let n = 1024;
        let w = kbd_aac_window(n, 4.0);
        assert_eq!(w.len(), 2 * n, "AAC 的开窗表是 2N 长");
        for i in 0..n {
            let s = f64::from(w[i]).powi(2) + f64::from(w[n + i]).powi(2);
            assert!((s - 1.0).abs() < 1e-6, "第 {i} 项的平方和是 {s}");
        }
    }

    #[test]
    fn kbd_aac_window_matches_ffmpeg_constants() {
        // `aactab.c` 里 `ff_kbd_window_init(ff_aac_kbd_long_1024, 4.0, 1024)`。
        // 只需要确认「端点接近 0、中段接近 1」这个形状 —— 数值本身由上面的 TDAC 钉住。
        let w = kbd_aac_window(1024, 4.0);
        // ⚠️ 形状**不是**「上升 + 平台 + 下降」——实测（`n=1024, alpha=4.0`）：
        //
        // ```text
        //   h[0]    = 0.000293   ← 起点接近 0
        //   h[511]  = 0.706
        //   h[1023] = 1.000000   ← 末尾才是峰值
        // ```
        //
        // 它是**单调从 0 升到 1**、没有平台；铺成 `2N` 之后
        // 前半 `[0, N)` 上升、`w[N]` 是峰值 1、后半 `[N, 2N)` 下降回 0。
        // 之前想当然地以为中段恒 1（那是 KBD 在**别的定标**下的形状），
        // 靠这条断言才发现。
        assert!(w[0] < 1e-3, "窗的起点应当接近 0，得到 {}", w[0]);
        assert!(
            (w[512] - 0.708).abs() < 1e-2,
            "半程处约 0.708，得到 {}",
            w[512]
        );
        let peak = w[1024];
        assert!((peak - 1.0).abs() < 1e-5, "`w[N]` 是峰值，得到 {peak}");
        assert!(w[2047] < 1e-3, "窗的终点应当接近 0，得到 {}", w[2047]);
        // 关于中点对称。
        for i in (0..2048).step_by(97) {
            assert!(
                (w[i] - w[2047 - i]).abs() < 1e-6,
                "窗应当关于中点对称：w[{i}] = {} vs w[{}] = {}",
                w[i],
                2047 - i,
                w[2047 - i]
            );
        }
    }

    #[test]
    fn sine_and_kbd_are_both_symmetric() {
        for w in [sine_window(128), kbd_aac_window(128, 6.0)] {
            let n = w.len();
            for i in 0..n / 2 {
                assert!(
                    (w[i] - w[n - 1 - i]).abs() < 1e-6,
                    "窗应当对称：第 {i} 项 {} vs 第 {} 项 {}",
                    w[i],
                    n - 1 - i,
                    w[n - 1 - i]
                );
            }
        }
    }

    #[test]
    fn mdct_round_trips_with_overlap() {
        // 真正要保证的性质：**开窗 → MDCT → IMDCT → 开窗 → 50% 重叠相加** 还原原信号。
        // 分段长度 = N，每次前进 N/2（50% 重叠），所以每个样本被两个块覆盖。
        let n = 64usize; // 小一点，直接法够快
        let w = sine_window(n);

        // 一段伪随机信号，长度 4N
        let total = 8 * n;
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) & 0xffff) as f32 / 32768.0 - 1.0
        };
        let src: Vec<f32> = (0..total).map(|_| next()).collect();

        let mut out = vec![0f32; total];
        let mut pos = 0usize;
        while pos + 2 * n <= total {
            // 开窗
            let mut buf = vec![0f32; 2 * n];
            for i in 0..2 * n {
                buf[i] = src[pos + i] * w[i];
            }
            // MDCT → IMDCT
            let mut spec = vec![0f32; n];
            mdct(&buf, &mut spec);
            let mut back = vec![0f32; 2 * n];
            imdct(&spec, &mut back);
            // 再开窗并叠加
            for i in 0..2 * n {
                out[pos + i] += back[i] * w[i];
            }
            pos += n;
        }

        // 中间部分（两端各差一个块）应当还原。
        //
        // ⚠️ **增益是 1.0**，不是 `N/2` ——
        // `mdct()` 不带归一化、`imdct()` 带 `2/N`，两者相乘恰好抵消。
        // 这与「展开成显式的余弦和」时的直觉相反（那时会看到 `N/2` 的因子），
        // 因为 `imdct` 里的 `2/N` 与 50% 重叠的两次覆盖各出一半，合起来正好归一。
        // 这一条正是在确认定标没写反 —— 写反了这里会是 `N/2` 或 `2/N`。
        let lo = 2 * n;
        let hi = total - 2 * n;
        let ratio: f64 = (lo..hi)
            .map(|i| {
                if src[i].abs() > 1e-3 {
                    (out[i] / src[i]) as f64
                } else {
                    0.0
                }
            })
            .filter(|r| *r != 0.0)
            .sum::<f64>()
            / (lo..hi).filter(|&i| src[i].abs() > 1e-3).count() as f64;
        assert!(
            (ratio - 1.0).abs() < 1e-3,
            "重叠相加应当无损（增益 1.0），量到 {ratio}"
        );
    }

    #[test]
    fn bessel_i0_matches_known_values() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
        // I0(1) = 1.2660658777520084
        assert!((bessel_i0(1.0) - 1.2660658777520084).abs() < 1e-9);
        // I0(2) = 2.2795853023360673
        assert!((bessel_i0(2.0) - 2.2795853023360673).abs() < 1e-9);
    }
}
