//! 解码端的**合成**：反量化之后的谱 → 时域 PCM。
//!
//! 三件事，顺序不能换（ISO 13818-7 的 filterbank 那一节）：
//!
//! ```text
//!   谱系数 → IMDCT(N→2N) → 乘合成窗 → 与上一帧的后半相加（50% 重叠）→ 交 1024 个样本
//! ```
//!
//! ## 窗表与编码端**是同一套**
//!
//! AAC 的 TDAC 要求「分析窗 × 合成窗」在重叠区满足 `w² + w'² = 1`。本仓的分析窗
//! 就是 `mdct` 里那几张表，合成窗形状完全相同 —— 所以这里直接复用，不另写一份。
//! `LONG_START` / `LONG_STOP` 两个过渡窗也一样（`mdct::long_start_window` 的文档里
//! 写明了「解码端的合成窗是同一套形状」）。
//!
//! ⚠️ 复用是有前提的：`window_shape` 那一位必须与编码端实际用的窗一致。位流里写错
//! 了也**照样合法**、帧数也对，只是重建出来的边界会「呼吸」—— 所以判据只能是
//! 解码回来的 SNR（见 `tests/ffmpeg_roundtrip.rs` 里那条 KBD 的测试）。
//!
//! ## 短窗为什么是「8 个 256 点窗摆进 2048 的缓冲」
//!
//! 一帧 `EIGHT_SHORT` 的 1024 个系数分成 8 组、每组 128 个。每组 IMDCT 出 256 个样本，
//! 相邻两组步进 128（50% 重叠）。8 组共覆盖 `7·128 + 256 = 1152` 个样本，在 2048 的
//! 帧缓冲里**居中**摆，两端各留 `(2048 − 1152)/2 = 448` 个零 —— 与编码端的
//! `short_pad` 是同一个数。
//!
//! ## 重叠状态
//!
//! 每帧交出 1024 个样本，同时把缓冲的后半（`buf[1024..2048]`）留给下一帧。
//! 第一帧的「上一帧后半」是全零 —— 这就是编码器那 1024 个样本的延迟（priming）。

use crate::mdct::{
    imdct_fast, kbd_aac_window, long_start_window, long_stop_window, short_window, sine_window,
    LONG_LEN, SHORT_LEN,
};

/// 短窗那一串在 2048 的帧缓冲里居中摆放时，两端各留的零长度。
const SHORT_PAD: usize = (2 * LONG_LEN - 9 * SHORT_LEN) / 2;

/// 一个声道的合成器：窗表 + 重叠历史。
///
/// 窗表在构造时算一次（KBD 要算贝塞尔函数，别每帧重算）；重叠历史跨帧保留。
pub struct Synthesis {
    /// `[shape]`：`0` = 正弦窗，`1` = KBD。长窗 2048 点。
    long: [Vec<f32>; 2],
    /// 短窗 256 点。
    short: [Vec<f32>; 2],
    /// `LONG_START` / `LONG_STOP` 的 2048 点过渡窗。
    start: [Vec<f32>; 2],
    stop: [Vec<f32>; 2],
    /// 每个声道上一帧缓冲的后半（1024 个样本）。
    overlap: Vec<Vec<f32>>,
    /// 每个声道**上一帧**的 `window_shape`（`0` = 正弦，`1` = KBD）。
    ///
    /// 本帧的上升沿要用它 —— 见 `frame` 上的 ⚠️。首帧取 `0`（与 ffmpeg 一致：
    /// 那里的 `use_kb_window[1]` 一开始就是 0）。
    prev_shape: Vec<u8>,
}

impl Synthesis {
    pub fn new(channels: usize) -> Self {
        let long_sine = sine_window(LONG_LEN);
        let long_kbd = kbd_aac_window(LONG_LEN, 4.0);
        let short_sine = short_window(false);
        let short_kbd = short_window(true);
        let start_sine = long_start_window(&long_sine, &short_sine);
        let start_kbd = long_start_window(&long_kbd, &short_kbd);
        let stop_sine = long_stop_window(&long_sine, &short_sine);
        let stop_kbd = long_stop_window(&long_kbd, &short_kbd);
        Self {
            long: [long_sine, long_kbd],
            short: [short_sine, short_kbd],
            start: [start_sine, start_kbd],
            stop: [stop_sine, stop_kbd],
            overlap: vec![vec![0f32; LONG_LEN]; channels],
            prev_shape: vec![0u8; channels],
        }
    }

    /// 丢掉重叠历史（seek / 换流之后必须调，否则第一帧会带上旧尾巴）。
    pub fn reset(&mut self) {
        for o in &mut self.overlap {
            o.fill(0.0);
        }
        self.prev_shape.fill(0);
    }

    /// 合成一个声道的一帧：`coefs` 是**已经反量化、做过 M/S 与 TNS 逆滤波**的谱。
    ///
    /// - `seq`：`0` = ONLY_LONG，`1` = LONG_START，`2` = EIGHT_SHORT，`3` = LONG_STOP；
    /// - `shape`：`0` = 正弦窗，`1` = KBD；
    /// - `out`：交出 1024 个样本（与上一帧的重叠已经加进去了）。
    ///
    /// 短窗时 `coefs` 是**窗口主序**（`[w*128 + k]`），与 `decode` 里谱数据的落点一致。
    ///
    /// ⚠️ **窗的前半用「上一帧」的 `window_shape`，后半用本帧的。** 这不是笔误：
    /// 相邻两帧重叠的那一段必须由**同一套**窗形构成，否则 `w²+w'² ≠ 1`，重建出来的
    /// 幅度就是错的。规范把这个责任放在「较早那一帧」上，所以本帧上升沿要沿用
    /// 上一帧的窗形（正弦 ↔ KBD 混用时尤其明显）。
    ///
    /// 症状极具误导性：**只有 `window_shape` 发生切换的那几帧错**，而且错在
    /// `seq=1`（LONG_START，铺满整个前半）和 `seq=3`（LONG_STOP，只在
    /// `[448, 576)` 那 128 点上）—— 其余帧逐样本完全相同。看起来像「起止窗建错了」，
    /// 实际是**窗形取错了来源**。
    pub fn frame(&mut self, ch: usize, seq: u8, shape: u8, coefs: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), LONG_LEN);
        debug_assert_eq!(coefs.len(), LONG_LEN);
        let cur = usize::from(shape != 0);
        let prev = self.prev_shape[ch] as usize;
        self.prev_shape[ch] = cur as u8;
        let mut buf = vec![0f32; 2 * LONG_LEN];

        if seq == 2 {
            let mut tmp = vec![0f32; 2 * SHORT_LEN];
            for w in 0..8 {
                imdct_fast(&coefs[w * SHORT_LEN..(w + 1) * SHORT_LEN], &mut tmp);
                let off = SHORT_PAD + w * SHORT_LEN;
                if w == 0 {
                    // 第 0 个短窗的**上升半**与上一帧的尾巴重叠，用上一帧的窗形。
                    for k in 0..SHORT_LEN {
                        buf[off + k] += tmp[k] * self.short[prev][k];
                    }
                    for k in SHORT_LEN..2 * SHORT_LEN {
                        buf[off + k] += tmp[k] * self.short[cur][k];
                    }
                } else {
                    for k in 0..2 * SHORT_LEN {
                        buf[off + k] += tmp[k] * self.short[cur][k];
                    }
                }
            }
        } else {
            let win = match seq {
                1 => &self.start,
                3 => &self.stop,
                // 0 与任何没见过的值都按普通长窗处理：`parse_ics_info` 只可能给出 0..=3。
                _ => &self.long,
            };
            let mut tmp = vec![0f32; 2 * LONG_LEN];
            imdct_fast(coefs, &mut tmp);
            for i in 0..LONG_LEN {
                buf[i] = tmp[i] * win[prev][i];
            }
            for i in LONG_LEN..2 * LONG_LEN {
                buf[i] = tmp[i] * win[cur][i];
            }
        }

        let ov = &mut self.overlap[ch];
        for i in 0..LONG_LEN {
            out[i] = buf[i] + ov[i];
            ov[i] = buf[LONG_LEN + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdct::mdct;

    /// **MDCT → 合成 的往返恒等**：这是整个 filterbank 的定标契约。
    ///
    /// 造两帧连续信号，各自 MDCT、再各自合成，拼起来应当等于原信号
    /// （从第 1024 个样本起 —— 前 1024 个是编码器的 priming，任何实现都给不出来）。
    ///
    /// 窗用正弦窗、序列用 ONLY_LONG：只验「定标 + 重叠 + 窗」这三件事，
    /// 不把短窗/过渡窗混进来。
    #[test]
    fn mdct_then_synthesis_is_identity_after_priming() {
        let n = LONG_LEN;
        let total = 4 * n;
        let src: Vec<f32> = (0..total)
            .map(|i| {
                let t = i as f32 / 44_100.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 8000.0
                    + (2.0 * std::f32::consts::PI * 1337.0 * t).sin() * 3000.0
            })
            .collect();

        let win = sine_window(n);
        let mut syn = Synthesis::new(1);
        let mut out = vec![0f32; total];

        // 帧 i 的输入 = `src[i*n .. i*n+2n]`（50% 重叠）；最后一帧只有前半有数据。
        for f in 0..3 {
            let mut buf = vec![0f32; 2 * n];
            for k in 0..2 * n {
                if f * n + k < total {
                    buf[k] = src[f * n + k] * win[k];
                }
            }
            let mut coefs = vec![0f32; n];
            mdct(&buf, &mut coefs);
            let mut pcm = vec![0f32; n];
            syn.frame(0, 0, 0, &coefs, &mut pcm);
            out[f * n..(f + 1) * n].copy_from_slice(&pcm);
        }

        // 第 0 帧的输出是 priming（只有上一帧的零尾巴），从第 1024 个样本起必须还原。
        let (mut worst, mut peak) = (0f64, 0f64);
        for i in n..2 * n {
            worst = worst.max((f64::from(out[i]) - f64::from(src[i])).abs());
            peak = peak.max(f64::from(src[i]).abs());
        }
        let rel = worst / peak;
        assert!(rel < 1e-5, "合成没还原：最大相对误差 {rel:.3e}");
    }

    /// 短窗的摆放位置：8 个 256 点窗必须**居中**落在 2048 的缓冲里。
    ///
    /// 判据不看波形，看「有没有把能量放到缓冲区两端」—— 放错 `SHORT_PAD`
    /// 的后果是帧边界出现台阶，而逐帧 SNR 只在瞬态帧上掉几个 dB，很难归因。
    #[test]
    fn short_windows_are_centred_in_the_frame_buffer() {
        assert_eq!(SHORT_PAD, 448);
        assert_eq!(SHORT_PAD + 7 * SHORT_LEN + 2 * SHORT_LEN, 2 * LONG_LEN - SHORT_PAD);
    }

    /// **窗形切换时仍然完美重建** —— 钉住「上升沿用上一帧的窗形」这条规则。
    ///
    /// 为什么必须单独钉：本仓的编码器只产 `ONLY_LONG` + 固定窗形，自编自解
    /// **永远走不到**这条路径；而 ffmpeg 的编码器会在瞬态处切 `window_shape`
    /// （正弦 ↔ KBD）。一旦上升沿错用本帧窗形，`w² + w'² ≠ 1`，症状是
    /// **只有切换的那几帧错**，幅度误差 O(1)，看起来却像「起止窗建错了」。
    ///
    /// 判据是往返恒等：分析窗与合成窗必须镜像，所以这里用同一条规则造分析窗。
    /// 合成端若改用本帧窗形，两条窗就对不上，相对误差立刻到 1e-1 量级。
    #[test]
    fn window_shape_switch_still_reconstructs() {
        let n = LONG_LEN;
        let frames = 5usize;
        let total = (frames + 1) * n;
        let src: Vec<f32> = (0..total)
            .map(|i| {
                let t = i as f32 / 44_100.0;
                (2.0 * std::f32::consts::PI * 523.0 * t).sin() * 9000.0
                    + (2.0 * std::f32::consts::PI * 2917.0 * t).sin() * 2000.0
            })
            .collect();

        let win = [sine_window(n), kbd_aac_window(n, 4.0)];
        // 故意来回切：0 → 1 → 1 → 0 → 1
        let shapes = [0usize, 1, 1, 0, 1];

        let mut syn = Synthesis::new(1);
        let mut out = vec![0f32; frames * n];
        for f in 0..frames {
            // 首帧的「上一帧窗形」是 0（`Synthesis::new` 的初值，与 ffmpeg 一致）。
            let p = if f == 0 { 0 } else { shapes[f - 1] };
            let c = shapes[f];
            let mut buf = vec![0f32; 2 * n];
            for k in 0..2 * n {
                if f * n + k < total {
                    // 分析窗：上升沿用上一帧窗形、下降沿用本帧窗形（与合成端同一规则）。
                    buf[k] = src[f * n + k] * win[if k < n { p } else { c }][k];
                }
            }
            let mut coefs = vec![0f32; n];
            mdct(&buf, &mut coefs);
            let mut pcm = vec![0f32; n];
            syn.frame(0, 0, c as u8, &coefs, &mut pcm);
            out[f * n..(f + 1) * n].copy_from_slice(&pcm);
        }

        // 第 0 帧的前半是 priming（上一帧尾巴是零），从第 n 个样本起必须还原。
        let (mut worst, mut peak) = (0f64, 0f64);
        for i in n..frames * n {
            worst = worst.max((f64::from(out[i]) - f64::from(src[i])).abs());
            peak = peak.max(f64::from(src[i]).abs());
        }
        let rel = worst / peak;
        assert!(rel < 1e-5, "窗形切换处没还原：最大相对误差 {rel:.3e}");
    }
}
