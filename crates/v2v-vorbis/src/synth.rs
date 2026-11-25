//! 窗、IMDCT、重叠相加。规范 4.3.1（窗）与 4.3.7 / 4.3.8（逆变换与叠加）。
//!
//! ## 窗
//!
//! 所有窗都是同一个斜率函数的组合：
//!
//! ```text
//!   slope(x) = sin(π/2 · sin²(π·(x + 0.5) / (2n)))
//! ```
//!
//! `n` 是**这段斜坡的宽度**（`left_n` / `right_n`），不是整块大小。
//!
//! 长块与短块**交叠**时（`previous_window_flag` / `next_window_flag` 为 0），
//! 长块那一侧的斜坡要**压窄**到短块的宽度 —— 这就是「hybrid 窗」。
//! 压窄之后两块的非零区正好等宽，重叠相加才满足 TDAC。
//!
//! ## 重叠相加的坐标
//!
//! 规范 4.3.8：叠加时**上一窗的 3/4 点对齐当前窗的 1/4 点**，返回的数据区间是
//! 「上一窗的中心 → 当前窗的中心」。记 `prev` 为上一块（长度 `Np`），`cur` 为当前块：
//!
//! - 两块窗口在时间轴上的起点差：`pos_cur - pos_prev = 3·Np/4 − N/4`；
//! - 于是 `cur[i]` 落在「以上一块中心为 0」的坐标里的位置是 `i + Np/4 − N/4`；
//! - 输出长度 = 两个中心之差 = `Np/4 + N/4`。
//!
//! 等长时 `Np = N` 退化成「重叠一半、每次出 `N/2`」；短→长（`Np=256, N=2048`）
//! 是 `64 + 512 = 576`，其中只有 128 个样本真的叠在一起 —— 规范明说这一点
//! 「不破坏变换的正交性」，但**返回的区间不能因此缩短**。
//!
//! 关键点：`pending` 只缓存上一块的**右半**（`Np/2` 个样本），而输出可能有
//! `Np/4 + N/4` 那么长，短→长时远大于 `Np/2`。所以不能「从 pending 里切一段」
//! 了事 —— 必须开一个 `Np/4 + N/4` 的输出缓冲，上一块的右半从下标 0 起加进去，
//! 当前块的样本按 `i + Np/4 − N/4` 的偏移加进去。

use v2v_kernel::ImdctPlan;

/// 一个块的窗。
pub struct Window {
    /// `n` 个系数。
    pub samples: Vec<f32>,
    /// 左斜坡起点（`[0, left_start)` 恒为 0）。
    pub left_start: usize,
    /// 右斜坡起点（`[right_start, n)` 恒为 0）。
    pub right_start: usize,
}

/// 生成一个块的窗。规范 4.3.1 第 4 步。
///
/// `long` 是这个 mode 的 `blockflag`；两个 `*_flag` 只在长块时有意义
/// （短块永远是对称的短窗）。
pub fn build_window(n: usize, bs0: usize, long: bool, prev_flag: bool, next_flag: bool) -> Window {
    let center = n / 2;
    let (left_start, left_end, left_n) = if long && !prev_flag {
        (n / 4 - bs0 / 4, n / 4 + bs0 / 4, bs0 / 2)
    } else {
        (0, center, n / 2)
    };
    let (right_start, right_end, right_n) = if long && !next_flag {
        (n * 3 / 4 - bs0 / 4, n * 3 / 4 + bs0 / 4, bs0 / 2)
    } else {
        (center, n, n / 2)
    };

    let mut samples = vec![0f32; n];
    for (i, s) in samples.iter_mut().enumerate() {
        *s = if i < left_start {
            0.0
        } else if i < left_end {
            slope((i - left_start) as f64 + 0.5, left_n)
        } else if i < right_start {
            1.0
        } else if i < right_end {
            // 右斜坡 = 左斜坡转 90°：sin(π/2 · cos²(…))，内层同样带 /2。
            let x = (i - right_start) as f64 + 0.5;
            let c = (std::f64::consts::PI * x / (2.0 * right_n as f64)).cos();
            (std::f64::consts::FRAC_PI_2 * c * c).sin() as f32
        } else {
            0.0
        };
    }
    Window { samples, left_start, right_start }
}

/// 规范 4.3.1 的斜率函数 `sin(π/2 · sin²(π(x+0.5)/(2n)))`。
///
/// **内层的 `2n` 不能写成 `n`**：写成 `n` 的话 `sin²` 在斜坡中点就过 1 回到 0，
/// 窗会在中间塌一个坑（测试 `mdct_round_trip_reconstructs_the_signal` 会立刻红）。
/// 这里 `n` 是**这段斜坡的宽度**，不是整块大小 —— hybrid 窗的斜坡被压窄到
/// `bs0/2`，除数必须跟着变，否则压窄后的斜坡升不到 1。
fn slope(x: f64, n: usize) -> f32 {
    let s = (std::f64::consts::PI * x / (2.0 * n as f64)).sin();
    (std::f64::consts::FRAC_PI_2 * s * s).sin() as f32
}

/// 一个声道的重叠状态：上一块的**右半**（已加窗），以及它的长度。
pub struct ChannelSynth {
    pending: Vec<f32>,
    prev_n: usize,
    /// IMDCT 输出的暂存（长度 = 块大小）。
    block: Vec<f32>,
    primed: bool,
}

impl Default for ChannelSynth {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelSynth {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            prev_n: 0,
            block: Vec::new(),
            primed: false,
        }
    }

    pub fn reset(&mut self) {
        self.pending.clear();
        self.prev_n = 0;
        self.primed = false;
    }

    /// IMDCT + 加窗 + 重叠相加。`coefs` 长度 = `n/2`；本块能输出的样本写进 `out`。
    ///
    /// 第一块只做 priming，`out` 被清空。
    ///
    /// **写进调用方的缓冲**（而不是返回 `Vec`）是为了让整条解码链零分配：
    /// 真样本上这是 5 万块，每块一个 4 KiB 的 `Vec` 分配 + 释放。
    pub fn synthesize_into(
        &mut self,
        plan: &mut ImdctPlan,
        coefs: &[f32],
        window: &Window,
        out: &mut Vec<f32>,
    ) {
        let n = window.samples.len();
        debug_assert_eq!(coefs.len(), n / 2);
        // ⚠️ **只在长度变了才 `resize`**。原来是无条件 `clear() + resize(n, 0.0)`，
        // 那是每块一次 8 KiB 的 memset —— 而 `ImdctPlan::apply` 本来就写满
        // 全部 `2n` 个样本（`put_imdct_sample` 对每个 `j ∈ [0, n)` 写两处，
        // 合起来正好覆盖 `[0, 2n)`；由 `imdct_apply_writes_every_sample` 钉住）。
        // 真样本上是 5 万块 × 8 KiB ≈ **400 MB 白写的 0**。
        if self.block.len() != n {
            self.block.resize(n, 0.0);
        }
        plan.apply(coefs, &mut self.block);
        for (s, w) in self.block.iter_mut().zip(&window.samples) {
            *s *= w;
        }
        self.overlap_add(window, out);
    }

    /// 把 `self.block`（**已经乘过窗**）叠进 `out`，`out` 里是本块完成的区间。
    fn overlap_add(&mut self, window: &Window, out: &mut Vec<f32>) {
        let n = self.block.len();
        if !self.primed {
            self.pending.clear();
            self.pending.extend_from_slice(&self.block[n / 2..]);
            self.prev_n = n;
            self.primed = true;
            out.clear();
            return;
        }

        let np = self.prev_n;
        let out_len = np / 4 + n / 4;
        // 长度没变时 `resize` 不清零 —— 所以下面要**自己清**上一块没盖到的那段。
        if out.len() != out_len {
            out.resize(out_len, 0.0);
        }

        // 上一块的右半 = 输出区间的前 `Np/2` 个样本（它自己的坐标里从中心开始）。
        let keep = self.pending.len().min(out_len);
        out[..keep].copy_from_slice(&self.pending[..keep]);
        // `[keep, out_len)` 上一轮留的是旧内容，必须清零：下面是对它做 `+=`。
        out[keep..].fill(0.0);

        // 当前块：`cur[i]` 落在输出下标 `i + Np/4 - N/4`。只加窗口非零的那一段，
        // 越界的部分直接丢掉（流不合法时也不至于 panic）。
        let shift = np as isize / 4 - n as isize / 4;
        for i in window.left_start..window.right_start.min(n) {
            let j = i as isize + shift;
            if j >= 0 && (j as usize) < out_len {
                out[j as usize] += self.block[i];
            }
        }

        self.pending.clear();
        self.pending.extend_from_slice(&self.block[n / 2..]);
        self.prev_n = n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// 测试用的一步合成：把 `synthesize_into` 包成「返回一个 Vec」。
    ///
    /// 生产路径不这么用（那是零分配的那一半），但测试里读起来更直白。
    fn synthesize(
        st: &mut ChannelSynth,
        plan: &mut ImdctPlan,
        coefs: &[f32],
        w: &Window,
    ) -> Vec<f32> {
        let mut out = Vec::new();
        st.synthesize_into(plan, coefs, w, &mut out);
        out
    }

    /// 规范意义上的 IMDCT（朴素 O(n²) 实现，直接从公式写）：
    ///
    /// ```text
    ///   y[j] = (2/M) · Σ_k X[k] · cos( (π/M)(j + 0.5 + M/2)(k + 0.5) )
    /// ```
    ///
    /// `2/M == 4/N`（`N = 2M` 是输出样本数），正是 MDCT / IMDCT 做 TDAC 时
    /// 需要的定标。内核的 `ImdctPlan` 是同一变换的快速实现，这条测试把两者的
    /// **定标与相位**钉在一起 —— 它们错了，整条链的输出会是「听起来对但电平不对」，
    /// 而那是最难查的一类。
    fn naive_imdct(coefs: &[f32]) -> Vec<f32> {
        let m = coefs.len();
        let n = 2 * m;
        let mut out = vec![0f32; n];
        for (j, o) in out.iter_mut().enumerate() {
            let mut s = 0f64;
            for (k, &c) in coefs.iter().enumerate() {
                let ang = PI / m as f64
                    * (j as f64 + 0.5 + m as f64 / 2.0)
                    * (k as f64 + 0.5);
                s += c as f64 * ang.cos();
            }
            *o = (s * 2.0 / m as f64) as f32;
        }
        out
    }

    #[test]
    fn imdct_plan_matches_the_textbook_transform() {
        for m in [16usize, 64, 128] {
            // 确定性伪随机系数（不用外部依赖）
            let coefs: Vec<f32> = (0..m)
                .map(|i| {
                    let x = (i as f32 * 0.37).sin() * 3.0 + ((i * 7919) % 11) as f32 * 0.1;
                    x
                })
                .collect();
            let want = naive_imdct(&coefs);
            let mut plan = ImdctPlan::new(m, 2.0 / m as f64);
            let mut got = vec![0f32; 2 * m];
            plan.apply(&coefs, &mut got);
            let max = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
            assert!(
                max / scale < 2e-3,
                "m={m}: 最大差 {max}（相对 {}），内核的定标/相位与教科书公式不一致",
                max / scale
            );
        }
    }

    /// TDAC 的充要条件（Princen-Bradley）：在**重叠区**里，
    /// `prev 的右半² + cur 的左半² = 1`。
    ///
    /// 注意这条**不是**「同一个窗的 `w[i]² + w[i+n/2]² = 1`」—— 那个只在窗左右对称
    /// 时成立，hybrid 窗（一侧被压窄到 `bs0/2`）必然不满足。写错这条测试会
    /// 逼你去改本来正确的窗。
    ///
    /// 判据用的坐标就是 `synthesize` 用的那套（`shift = Np/4 - N/4`）：
    /// 它和窗生成只要有一边错，这里就红。
    #[test]
    fn window_satisfies_princen_bradley() {
        // (prev 的 (n, long, prev_flag, next_flag), cur 的同样四元组)
        let cases = [
            // 短→短
            ((256usize, false, true, true), (256usize, false, true, true)),
            // 短→长（长块的 previous_window_flag = 0，左侧被压窄）
            ((256, false, true, true), (2048, true, false, true)),
            // 长→短（长块的 next_window_flag = 0，右侧被压窄）
            ((2048, true, true, false), (256, false, true, true)),
            // 长→长
            ((2048, true, true, true), (2048, true, true, true)),
        ];
        for (p, c) in cases {
            let wp = build_window(p.0, 256, p.1, p.2, p.3);
            let wc = build_window(c.0, 256, c.1, c.2, c.3);
            let np = p.0;
            let n = c.0;
            let out_len = np / 4 + n / 4;
            let shift = np as isize / 4 - n as isize / 4;
            for j in 0..out_len {
                let a = wp.samples.get(np / 2 + j).copied().unwrap_or(0.0);
                let k = j as isize - shift;
                let b = if k >= 0 && (k as usize) < n {
                    wc.samples[k as usize]
                } else {
                    0.0
                };
                let s = a * a + b * b;
                assert!(
                    (s - 1.0).abs() < 1e-4,
                    "prev n={np} cur n={n} j={j}: {a}² + {b}² = {s}"
                );
            }
        }
    }

    /// 交叠两个块，输出长度必须是 `Np/4 + N/4`（规范 4.3.8）。
    ///
    /// 窗的 `*_flag` 必须和「相邻块是长是短」**一致** —— 长块后面跟短块时，
    /// 长块的 `next_window_flag` 必须是 0。不然后半段的斜坡宽度对不上，
    /// 长度对了也是错的。
    #[test]
    fn overlap_length_matches_spec() {
        let short = build_window(256, 256, false, true, true);
        // 长块的左右两侧都是「邻块为短」，所以两个 flag 都是 0。
        let long_hybrid = build_window(2048, 256, true, false, false);
        let long_plain = build_window(2048, 256, true, true, true);
        let mut plan_s = ImdctPlan::new(128, 2.0 / 128.0);
        let mut plan_l = ImdctPlan::new(1024, 2.0 / 1024.0);

        // 短→长：256/4 + 2048/4 = 576
        let mut st = ChannelSynth::new();
        assert!(synthesize(&mut st, &mut plan_s, &vec![0f32; 128], &short).is_empty());
        let out = synthesize(&mut st, &mut plan_l, &vec![0f32; 1024], &long_hybrid);
        assert_eq!(out.len(), 576);

        // 长→短：也是 576
        let mut st = ChannelSynth::new();
        assert!(synthesize(&mut st, &mut plan_l, &vec![0f32; 1024], &long_hybrid).is_empty());
        let out = synthesize(&mut st, &mut plan_s, &vec![0f32; 128], &short);
        assert_eq!(out.len(), 576);

        // 等长：2048/4 + 2048/4 = 1024
        let mut st = ChannelSynth::new();
        synthesize(&mut st, &mut plan_l, &vec![0f32; 1024], &long_plain);
        let out = synthesize(&mut st, &mut plan_l, &vec![0f32; 1024], &long_plain);
        assert_eq!(out.len(), 1024);
    }

    /// 端到端判据：**朴素前向 MDCT → 本模块的 IMDCT + 加窗 + 叠加**，必须重建出原信号。
    ///
    /// 这一条把「窗形状、IMDCT 定标、叠加偏移量」三样一起钉住：任何一个错一格，
    /// 重建的波形都会歪（错一格是 O(1) 的误差，不是舍入级别）。块序列里故意夹了
    /// 两个短块，所以 hybrid 窗那条路径也被走到。
    ///
    /// 第一块只 priming，`x[0 .. center_0]` 收不回来 —— 这是规范规定的，不是 bug。
    #[test]
    fn mdct_round_trip_reconstructs_the_signal() {
        /// 教科书前向 MDCT（O(n²)，只为测试存在）：
        /// `X[k] = Σ_i s[i]·cos((π/M)(i + 0.5 + M/2)(k + 0.5))`，`M = n/2`。
        fn naive_mdct(s: &[f32]) -> Vec<f32> {
            let n = s.len();
            let m = n / 2;
            (0..m)
                .map(|k| {
                    let mut acc = 0f64;
                    for (i, &v) in s.iter().enumerate() {
                        let ang = PI / m as f64
                            * (i as f64 + 0.5 + m as f64 / 2.0)
                            * (k as f64 + 0.5);
                        acc += v as f64 * ang.cos();
                    }
                    acc as f32
                })
                .collect()
        }

        let bs0 = 16usize;
        let long_n = 64usize;
        // 长、长、短、短、长 —— 中间夹短块，把 hybrid 窗和短窗都覆盖到。
        let sizes = [long_n, long_n, bs0, bs0, long_n];

        // 每块窗口在时间轴上的起点：`pos_k = pos_{k-1} + 3·Np/4 − N/4`。
        let mut pos = vec![0i64];
        for k in 1..sizes.len() {
            let p = pos[k - 1] + 3 * sizes[k - 1] as i64 / 4 - sizes[k] as i64 / 4;
            pos.push(p);
        }

        let len = 160usize;
        let x: Vec<f32> = (0..len)
            .map(|i| {
                let f = i as f32;
                (f * 0.7).sin() * 0.6 + (f * 2.3).cos() * 0.3
            })
            .collect();

        let mut st = ChannelSynth::new();
        // 输出的写指针（以 `x` 的下标计）：第一块的输出从 `center_0` 开始。
        let mut at = (pos[0] + sizes[0] as i64 / 2) as usize;
        let mut checked = 0usize;

        for k in 0..sizes.len() {
            let n = sizes[k];
            let long = n == long_n;
            // flag 的含义是「相邻那块是不是长窗」，不是「这块」。
            let prev_flag = k == 0 || sizes[k - 1] == long_n;
            let next_flag = k + 1 == sizes.len() || sizes[k + 1] == long_n;
            let w = build_window(n, bs0, long, prev_flag, next_flag);

            let mut seg = vec![0f32; n];
            for (i, s) in seg.iter_mut().enumerate() {
                let t = pos[k] + i as i64;
                *s = if t >= 0 && (t as usize) < len { x[t as usize] } else { 0.0 };
            }
            for (s, wv) in seg.iter_mut().zip(&w.samples) {
                *s *= wv;
            }
            let coefs = naive_mdct(&seg);
            let mut plan = ImdctPlan::new(n / 2, 2.0 / (n / 2) as f64);
            let out = synthesize(&mut st, &mut plan, &coefs, &w);

            if k == 0 {
                assert!(out.is_empty(), "第一块只 priming");
                continue;
            }
            // 本块完成的区间长度必须是 `Np/4 + N/4`。
            assert_eq!(out.len(), sizes[k - 1] / 4 + n / 4);
            for (j, v) in out.iter().enumerate() {
                let t = at + j;
                assert!(t < len, "输出越界：t={t}");
                let want = x[t];
                assert!(
                    (v - want).abs() < 3e-3,
                    "k={k} t={t}: 重建出 {v}，原始是 {want}"
                );
            }
            checked += out.len();
            at += out.len();
        }
        // 必须真的比过一段像样的长度，别让「全是空块」蒙过去。
        assert!(checked > 60, "只比了 {checked} 个样本，覆盖太窄");
    }
}
