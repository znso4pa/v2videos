//! **TNS（时域噪声整形，Temporal Noise Shaping）** —— 只做长窗。
//!
//! ## 它解决什么
//!
//! 量化噪声在**频域**是平的，但人耳在意的是**时域**：一个瞬态（鼓点、辅音）
//! 之后，平铺的噪声会在时域上「糊」成一片可听见的 pre-echo。
//! TNS 的做法是：编码端用一条 **LPC 预测滤波器（MA）** 把频谱「白化」，
//! 于是量化噪声被这条滤波器的**逆（AR）**整形 —— 解码端做逆滤波时，
//! 噪声跟着信号包络走，落在被信号掩蔽的地方。
//!
//! ```text
//!   编码：  X'[k] = X[k] + Σ a[i]·X[k-i]        （MA，用**原始**谱）
//!   解码：  X[k]  = X'[k] − Σ a[i]·X[k-i]       （AR，用**已重建**的谱）
//! ```
//!
//! ## 口径（逐行对着 ffmpeg 的 `aacenc_tns.c` / `lpc.c` 写）
//!
//! | 环节 | 做法 |
//! |---|---|
//! | 分析范围 | 频带 `[tns_min_sfb_long, min(num_swb, min(tns_max_bands, max_sfb)))` |
//! | 窗 | Welch：`0.5 − 0.5·cos(2πi/(len−1))` |
//! | 自相关 | `autoc[j] = **1.0** + Σ w[i]·w[i−j]` —— 那个 `1.0` 是 ffmpeg 的正则项 |
//! | 预测增益 | `autoc[0] / avg_err`；只有落在 **1.4 ≤ gain < 1.624** 才用 TNS |
//! | 滤波器个数 | 长窗 **2** 个（各 `order/2 = 6` 阶、`sfb_len/2` 条带） |
//! | 方向 | 前/后两半的能量比较，往**能量小**的那头扫 |
//! | 量化 | 4 位表 `TNS_COEF_4BIT`（16 项）；**恒不压缩**（`coef_compress = 0`） |
//!
//! ⚠️ **为什么恒不压缩系数**：`coef_compress = 1` 时解码端会改用另一张表
//! （`tns_tmp2_map_1_4`，8 项），而编码端的 `compress_coeffs` 只是把索引整体
//! 平移 —— 两端的值就对不上了。ffmpeg 的注释也写着这条压缩「并不无损」。
//! 少一位不如少一个 bug。

use crate::tables;

/// 长窗的 TNS 阶数上限（AAC-LC）。
const ORDER_LONG: usize = 12;

/// 长窗的滤波器个数（ffmpeg：`order != TNS_MAX_ORDER ? 2 : 3`，LC 走 2）。
const N_FILT_LONG: usize = 2;

/// 预测增益的可用区间（`TNS_GAIN_THRESHOLD_LOW` / `_HIGH`）。
const GAIN_LOW: f64 = 1.4;
const GAIN_HIGH: f64 = 1.16 * 1.4;

/// 一个 TNS 滤波器。
pub struct Filter {
    /// 覆盖多少条频带（解码端从**最高**的频带往下数）。
    pub length: u8,
    pub order: u8,
    /// `true` = 从高频往低频扫。
    pub direction: bool,
    /// 量化后的系数索引（写进位流的就是它）。
    pub coef_idx: Vec<u8>,
    /// 由**量化值**推出的 LPC（编码端滤波用；解码端用同一套值）。
    pub lpc: Vec<f32>,
}

/// 一个通道的 TNS 参数。
pub struct Tns {
    pub filters: Vec<Filter>,
}

/// Welch 窗 + 自相关（`ff_lpc_calc_ref_coefs_f` + `lpc_compute_autocorr_c`）。
///
/// ⚠️ 两个细节都不是随手写的：窗是 `0.5 − 0.5·cos(2πi/(len−1))`（分母是 `len−1`，
/// 不是 `len`）；自相关的**初值是 1.0**（ffmpeg 拿它当正则项，避免全零谱把
/// Levinson 除爆）。
fn autocorr(x: &[f32], order: usize) -> Vec<f64> {
    let len = x.len();
    let denom = (len as f64 - 1.0).max(1.0);
    let w: Vec<f64> = (0..len)
        .map(|i| {
            let weight = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos();
            f64::from(x[i]) * weight
        })
        .collect();
    let mut autoc = vec![0f64; order + 1];
    for (j, a) in autoc.iter_mut().enumerate() {
        let mut sum = 1.0f64;
        for i in j..len {
            sum += w[i] * w[i - j];
        }
        *a = sum;
    }
    autoc
}

/// Levinson-Durbin：自相关 → **反射系数** + 各级预测误差（`compute_ref_coefs`）。
fn ref_coefs(autoc: &[f64], order: usize) -> (Vec<f64>, Vec<f64>) {
    let mut gen0 = vec![0f64; order];
    let mut gen1 = vec![0f64; order];
    for i in 0..order {
        gen0[i] = autoc[i + 1];
        gen1[i] = autoc[i + 1];
    }
    let mut refc = vec![0f64; order];
    let mut error = vec![0f64; order];
    let mut err = autoc[0];
    let div = if err != 0.0 { err } else { 1.0 };
    refc[0] = -gen1[0] / div;
    err += gen1[0] * refc[0];
    error[0] = err;
    for i in 1..order {
        for j in 0..order - i {
            gen1[j] = gen1[j + 1] + refc[i - 1] * gen0[j];
            gen0[j] = gen1[j + 1] * refc[i - 1] + gen0[j];
        }
        let div = if err != 0.0 { err } else { 1.0 };
        refc[i] = -gen1[0] / div;
        err += gen1[0] * refc[i];
        error[i] = err;
    }
    (refc, error)
}

/// 反射系数 → LPC（`compute_lpc_coefs`，`normalize = 0` 那条路：不除误差，
/// 直接做步进递推；注意第一步是 **`r = −refc[i]`**，那个负号不能漏）。
pub fn lpc_from_ref(refc: &[f32]) -> Vec<f32> {
    let order = refc.len();
    let mut lpc = vec![0f32; order];
    for i in 0..order {
        let r = -refc[i];
        lpc[i] = r;
        for j in 0..(i + 1) / 2 {
            let f = lpc[j];
            let b = lpc[i - 1 - j];
            lpc[j] = f + r * b;
            lpc[i - 1 - j] = b + r * f;
        }
    }
    lpc
}

/// 反射系数 → 量化索引 + 量化值（就近取 `TNS_COEF_4BIT`）。
fn quantize(refc: &[f64]) -> (Vec<u8>, Vec<f32>) {
    let mut idx = Vec::with_capacity(refc.len());
    let mut val = Vec::with_capacity(refc.len());
    for &c in refc {
        let mut best = 0usize;
        let mut best_err = f64::MAX;
        for (i, &t) in tables::TNS_COEF_4BIT.iter().enumerate() {
            let e = (c - f64::from(t)) * (c - f64::from(t));
            if e < best_err {
                best_err = e;
                best = i;
            }
        }
        idx.push(best as u8);
        val.push(tables::TNS_COEF_4BIT[best]);
    }
    (idx, val)
}

/// 这一帧值不值得做 TNS；值就返回参数。
///
/// 判据是**预测增益**：太小说明谱本来就白（TNS 没收益），太大说明这是
/// 纯音（整形反而会把噪声推到听得见的地方）—— 所以只取中间那段。
pub fn search(spec: &[f32], swb: &[u16], sr_idx: usize, max_sfb: usize) -> Option<Tns> {
    let mmm = (tables::TNS_MAX_BANDS_1024[sr_idx] as usize).min(max_sfb);
    let num_swb = swb.len() - 1;
    let sfb_start = (tables::TNS_MIN_SFB_LONG[sr_idx] as usize).min(mmm);
    let sfb_end = num_swb.min(mmm);
    if sfb_end <= sfb_start {
        return None;
    }
    let sfb_len = sfb_end - sfb_start;
    let coef_start = swb[sfb_start] as usize;
    let coef_end = swb[sfb_end] as usize;
    if coef_end <= coef_start {
        return None;
    }

    let seg = &spec[coef_start..coef_end];
    let autoc = autocorr(seg, ORDER_LONG);
    let (refc, error) = ref_coefs(&autoc, ORDER_LONG);
    let mut avg_err = 0f64;
    for e in &error {
        avg_err = (avg_err + e) / 2.0;
    }
    if avg_err <= 0.0 {
        return None;
    }
    let gain = autoc[0] / avg_err;
    if !gain.is_finite() || !(GAIN_LOW..GAIN_HIGH).contains(&gain) {
        return None;
    }

    // 前/后两半的能量：方向是「往能量小的那头扫」。
    let half = sfb_len / 2;
    let mut en = [0f32; 2];
    for b in sfb_start..sfb_end {
        let e: f32 = spec[swb[b] as usize..swb[b + 1] as usize]
            .iter()
            .map(|&v| v * v)
            .sum();
        if b > sfb_start + half {
            en[1] += e;
        } else {
            en[0] += e;
        }
    }

    let (idx, val) = quantize(&refc);
    let per = ORDER_LONG / N_FILT_LONG;
    let mut filters = Vec::with_capacity(N_FILT_LONG);
    for f in 0..N_FILT_LONG {
        let coef_idx = idx[f * per..(f + 1) * per].to_vec();
        let lpc = lpc_from_ref(&val[f * per..(f + 1) * per]);
        filters.push(Filter {
            length: (sfb_len / N_FILT_LONG) as u8,
            order: per as u8,
            direction: en[f] < en[1 - f],
            coef_idx,
            lpc,
        });
    }
    Some(Tns { filters })
}

/// 编码端的 **MA 滤波**（`ff_aac_apply_tns`）：用**原始**谱做参考。
///
/// 覆盖范围与解码端**逐字一致**：解码端从 `num_swb` 往下数 `length` 条带，
/// 再夹到 `min(tns_max_bands, max_sfb)` —— 两边算出的区间必须是同一个，
/// 否则滤波的谱位置对不上。
pub fn apply(spec: &mut [f32], tns: &Tns, swb: &[u16], sr_idx: usize, max_sfb: usize) {
    let mmm = (tables::TNS_MAX_BANDS_1024[sr_idx] as usize).min(max_sfb);
    let num_swb = swb.len() - 1;
    // 参考谱：**滤波前**的原始系数（ffmpeg 的 `pcoeffs`）。
    let orig = spec.to_vec();

    let mut bottom = num_swb;
    for filt in &tns.filters {
        let top = bottom;
        bottom = top.saturating_sub(filt.length as usize);
        let start = swb[bottom.min(mmm)] as usize;
        let end = swb[top.min(mmm)] as usize;
        if end <= start || filt.order == 0 {
            continue;
        }
        let size = end - start;
        let (start, inc): (isize, isize) = if filt.direction {
            (end as isize - 1, -1)
        } else {
            (start as isize, 1)
        };
        for m in 0..size {
            let k = (start + m as isize * inc) as usize;
            let mut acc = spec[k];
            for i in 1..=(filt.order as usize).min(m) {
                let j = (k as isize - i as isize * inc) as usize;
                acc += filt.lpc[i - 1] * orig[j];
            }
            spec[k] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn swb_44k() -> Vec<u16> {
        crate::tables::SWB_OFFSET_1024[4][..crate::tables::SWB_OFFSET_1024_LEN[4] as usize].to_vec()
    }

    /// 反射系数 → LPC 的步进递推：手算一个二阶的例子。
    ///
    /// `normalize = 0` 那条路：`r = −refc[i]`，然后
    /// `lpc[j] ← lpc[j] + r·lpc[i−1−j]`、`lpc[i−1−j] ← lpc[i−1−j] + r·lpc[j]`。
    #[test]
    fn lpc_from_ref_matches_hand_computation() {
        let lpc = lpc_from_ref(&[0.5, 0.25]);
        // i=0: lpc[0] = −0.5
        // i=1: r = −0.25；f = b = lpc[0] = −0.5 → lpc[0] = −0.5 + 0.125 = −0.375
        assert!((lpc[0] + 0.375).abs() < 1e-6, "lpc[0] = {}", lpc[0]);
        assert!((lpc[1] + 0.25).abs() < 1e-6, "lpc[1] = {}", lpc[1]);
    }

    /// Levinson 应当从自相关里还原出**已知的**一阶 AR 系数。
    ///
    /// 对 `x[k] = a·x[k−1] + w[k]` 这种过程，`r[j] = a^j·r[0]`，于是
    /// 一阶反射系数就是 `−a`。这条把「自相关 → Levinson → 反射系数」整条链
    /// 钉住 —— 它比「某个素材必须触发 TNS」有意义得多（后者只是在测调参窗口）。
    #[test]
    fn levinson_recovers_known_ar1_coefficient() {
        let a = 0.9f64;
        // 直接构造自相关（不必造信号）
        let r0 = 1.0;
        let autoc = vec![r0, a * r0, a * a * r0, a * a * a * r0];
        let (refc, err) = ref_coefs(&autoc, 3);
        assert!((refc[0] + a).abs() < 1e-9, "一阶反射系数 = {}（应为 {}）", refc[0], -a);
        // 误差逐级**不增**（完美 AR(1) 的第二级反射系数是 0，误差不再降）
        assert!(err[0] < autoc[0], "一级误差没降：{} vs {}", err[0], autoc[0]);
        assert!(err[1] <= err[0] + 1e-12, "二级误差反而涨了");
        assert!((refc[1]).abs() < 1e-9, "二阶反射系数应为 0（得到 {}）", refc[1]);
    }

    /// 平坦谱不该触发 TNS（增益 ≈ 1，落在 [1.4, 1.624) 之外）。
    #[test]
    fn flat_spectrum_does_not_trigger_tns() {
        let x = vec![0.5f32; 1024];
        let swb = swb_44k();
        assert!(search(&x, &swb, 4, swb.len() - 1).is_none());
    }

    /// **编解码契约**：编码端 MA 滤波 + 解码端 AR 逆滤波 ≈ 恒等。
    ///
    /// 这是 TNS 唯一真正的正确性判据 —— 符号、方向、覆盖范围任何一处不一致，
    /// 这个往返就会崩，而位流本身仍然「合法」（ffmpeg 也不会报错）。
    ///
    /// 这里**手工构造**滤波器（不走 `search`），把测试与那个很窄的增益窗口解耦。
    #[test]
    fn ma_filter_is_inverted_by_ar_filter() {
        let swb = swb_44k();
        let max_sfb = swb.len() - 1;

        // 一段有谱倾斜的谱（低通状），幅度够大
        let mut x = vec![0f32; 1024];
        let mut state = 999u32;
        for (k, v) in x.iter_mut().enumerate() {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((state >> 16) as f32 / 32768.0) - 1.0;
            let env = (-(k as f32) / 200.0).exp();
            *v = noise * env * 800.0;
        }

        // 手工造两个滤波器：4 阶、覆盖 12 条带（与 `search` 的结构一致）
        for (order, dir) in [(4usize, false), (4usize, true)] {
            let coef_idx: Vec<u8> = (0..order).map(|i| (i as u8 % 16)).collect();
            let val: Vec<f32> = coef_idx
                .iter()
                .map(|&i| tables::TNS_COEF_4BIT[i as usize])
                .collect();
            let t = Tns {
                filters: vec![Filter {
                    length: 12,
                    order: order as u8,
                    direction: dir,
                    coef_idx,
                    lpc: lpc_from_ref(&val),
                }],
            };

            let orig = x.clone();
            apply(&mut x, &t, &swb, 4, max_sfb);
            assert!(x != orig, "滤波之后谱没变？");

            // 解码端：AR 逆滤波（与 `aacdec_dsp_template.c` 的 `decode = 1` 同口径）
            let mmm = (tables::TNS_MAX_BANDS_1024[4] as usize).min(max_sfb);
            let num_swb = swb.len() - 1;
            let mut bottom = num_swb;
            for f in &t.filters {
                let top = bottom;
                bottom = top.saturating_sub(f.length as usize);
                let start = swb[bottom.min(mmm)] as usize;
                let end = swb[top.min(mmm)] as usize;
                if end <= start {
                    continue;
                }
                let size = end - start;
                let (start, inc): (isize, isize) = if f.direction {
                    (end as isize - 1, -1)
                } else {
                    (start as isize, 1)
                };
                for m in 0..size {
                    let k = (start + m as isize * inc) as usize;
                    for i in 1..=(f.order as usize).min(m) {
                        let j = (k as isize - i as isize * inc) as usize;
                        x[k] -= f.lpc[i - 1] * x[j];
                    }
                }
            }

            let err: f32 = orig.iter().zip(&x).map(|(a, b)| (a - b) * (a - b)).sum();
            let sig: f32 = orig.iter().map(|a| a * a).sum();
            assert!(
                err < sig * 1e-3,
                "MA→AR 往返没还原（err/sig = {}，direction = {dir}）—— 两端口径不一致",
                err / sig
            );
        }
    }
}
