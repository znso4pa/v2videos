//! 缩放：**分离式**重采样（先水平、再垂直），像素中心对齐，缩小时核自动拉伸。
//!
//! ## 为什么分离式
//!
//! 二维直接卷积是 O(n²·dw·dh) 次乘加；分离成两遍是 O(n·dw·sh + n·dw·dh)，
//! 在「缩一半」这种 sh ≈ dh 的情形下差一倍。更重要的是**中间结果不落 u8** ——
//! 水平那一遍的结果存成 1/256 单位的 `u16`，垂直那遍才右移回 8 位。
//! 落 u8 会每遍各舍入一次，恒定色块上也能抖出 ±1 LSB。
//!
//! ## 坐标为什么用定点而不是浮点
//!
//! `(d + 0.5) * src / dst - 0.5` 用 f64 算，换个平台可能差 1 ulp；差在整数边界
//! （正好 0.5 的地方）就会翻成另一个抽头，输出差一整个像素 —— 那种差异看着像
//! 「算法不同」，其实是浮点。这里把式子整体乘 256，全程 `i64`：
//!
//! ```text
//! c(1/256 像素) = ((2d + 1) · src · 128) / dst − 128
//! ```
//!
//! 整数除法对负数要**向下取整**（`div_euclid`），不是向零取整 —— 缩小时 `d = 0`
//! 的采样位置会落在 `-0.25` 这种负数上，向零取整会让第一个像素少缩一点。
//!
//! ## 缩小时核必须拉伸（否则混叠）
//!
//! 放大的核半径是固定的（tent 1 像素、Catmull-Rom 2 像素）。缩小**不能**沿用：
//! 一个目标像素覆盖了多个源像素，只取固定 2 / 4 个抽头等于**点采样**，高频会
//! 折回低频（摩尔纹、锯齿）。正确做法是把核按 `src/dst` 拉伸，让它覆盖整个
//! 采样足迹，再归一化 —— 这也是 swscale 的做法。
//!
//! 实测（源 16→8、脉冲在 x=3）：ffmpeg 给出 `96 / 32`，正是拉伸到半径 2 的 tent
//! 核归一化后的 `0.375 / 0.125`。不拉伸的实现只会给 `128 / 0`（脉冲整个落进一个
//! 目标像素），对 `testsrc2` 的 2:1 缩小只有 **35.5 dB** —— 差距就是混叠。
//!
//! ## 两个核
//!
//! - [`Algo::Bilinear`]：tent 核（三角形）。快，边缘发糊。
//! - [`Algo::Bicubic`]：Catmull-Rom（`B = 0, C = 0.5`）。比 ffmpeg 的默认 bicubic
//!   （`B = 0, C = 0.6`）更锐一点，代价是振铃稍明显。

use v2v_kernel::error::{Error, Result};
use v2v_kernel::mc::Frame;

/// 重采样核。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algo {
    /// 双线性（tent 核）。快，边缘糊。
    Bilinear,
    /// 双三次，Catmull-Rom。锐，代价是振铃。
    Bicubic,
}

impl Algo {
    /// 命令行 / 报告里用的名字。
    pub fn name(self) -> &'static str {
        match self {
            Self::Bilinear => "bilinear",
            Self::Bicubic => "bicubic",
        }
    }

    /// 放大时核的半径（1/256 像素）。
    fn base_support(self) -> i64 {
        match self {
            Self::Bilinear => 256,
            Self::Bicubic => 512,
        }
    }
}

/// 把一帧 4:2:0 缩放到 `dw × dh`。
///
/// `dw` / `dh` 为 0、或源的平面长度与宽高对不上（不是 4:2:0）→ `Error::Bad`。
pub fn resize(src: &Frame, dw: u32, dh: u32, algo: Algo) -> Result<Frame> {
    let (sw, sh) = (src.width as usize, src.height as usize);
    if sw == 0 || sh == 0 {
        return Err(Error::Bad("源帧的宽或高是 0"));
    }
    if dw == 0 || dh == 0 {
        return Err(Error::Bad("目标宽或高是 0"));
    }
    if src.y.len() != sw * sh {
        return Err(Error::Bad("亮度平面长度与宽高不符"));
    }
    let (cw, ch) = (chroma_len(sw), chroma_len(sh));
    if src.u.len() != cw * ch || src.v.len() != cw * ch {
        return Err(Error::Bad("色度平面长度与宽高不符（不是 4:2:0）"));
    }

    let (dw, dh) = (dw as usize, dh as usize);
    let (cdw, cdh) = (chroma_len(dw), chroma_len(dh));
    Ok(Frame {
        width: dw as u32,
        height: dh as u32,
        y: scale_plane(&src.y, sw, sh, dw, dh, algo),
        u: scale_plane(&src.u, cw, ch, cdw, cdh, algo),
        v: scale_plane(&src.v, cw, ch, cdw, cdh, algo),
    })
}

/// 4:2:0 的平面边长：向上取整（`ceil(n/2)`）。
fn chroma_len(n: usize) -> usize {
    (n + 1) / 2
}

/// 一条轴上的抽头：每个目标位置的**起始索引**与**16 位定点权重**（每行和为 65536）。
struct Taps {
    first: Vec<isize>,
    w: Vec<Vec<i32>>,
}

impl Taps {
    /// `dst` 个目标位置在 `src` 长度上的抽头。
    fn new(dst: usize, src: usize, algo: Algo) -> Self {
        // 核半径：放大时是核本身的半径，缩小时按 src/dst 拉伸（抗混叠）。
        let scale = (src as i64 * 256 + dst as i64 / 2) / dst as i64; // src/dst，1/256
        let sup = algo.base_support().max(scale);

        let mut first = Vec::with_capacity(dst);
        let mut w = Vec::with_capacity(dst);
        for d in 0..dst {
            let c = ((2 * d as i64 + 1) * src as i64) * 128 / dst as i64 - 128;
            // 抽头覆盖 [c − sup, c + sup]，两端都按 floor 取整（负数要向下）。
            let lo = (c - sup + 256).div_euclid(256);
            let hi = (c + sup).div_euclid(256);
            let mut row: Vec<i64> = (lo..=hi)
                .map(|i| kernel(algo, (c - i * 256).abs(), sup))
                .collect();
            normalize(&mut row);
            first.push(lo as isize);
            w.push(row.into_iter().map(|x| x as i32).collect());
        }
        Self { first, w }
    }
}

/// 归一化到 16 位定点、和恰为 65536。
///
/// 和必须**精确**是 65536：差一点的话恒定色块会被缩出 ±1 LSB 的波纹，而那种波纹
/// 看起来像「算法有噪声」，会把人往错的方向带。舍入残差补到权重最大那个抽头上。
fn normalize(w: &mut [i64]) {
    let sum: i64 = w.iter().sum();
    debug_assert!(sum > 0, "核权重全是 0：support 算错了");
    for x in w.iter_mut() {
        *x = (*x * 65536 + sum / 2) / sum;
    }
    let got: i64 = w.iter().sum();
    let mut max_at = 0;
    for (i, &v) in w.iter().enumerate() {
        if v > w[max_at] {
            max_at = i;
        }
    }
    w[max_at] += 65536 - got;
}

/// 核值（未归一化，单位 1/256 像素）。
///
/// `dist` 是到抽头的距离、`sup` 是半径，两者都是 1/256 像素 —— 所以 `dist/sup`
/// 就是核函数的自变量 `x`，与「拉伸」无关：拉伸体现在 `sup` 变大。
fn kernel(algo: Algo, dist: i64, sup: i64) -> i64 {
    if dist >= sup {
        return 0;
    }
    match algo {
        // tent：1 − |x|
        Algo::Bilinear => sup - dist,
        // Catmull-Rom：自变量是 `x = dist / (sup / 2)` —— 半径是 **2 像素**，
        // 不是 `dist / sup`。写成 `dist / sup` 等于把核压缩一半（实测掉到 33 dB）。
        //
        // 刻度用 **1/65536**，不能是 1/256：`u = dist·256/sup` 在放大时等于
        // `dist/2`，dist 为奇数时丢掉 0.5，核值就差一截。
        Algo::Bicubic => {
            let d = 65536i64; // x = 1.0
            let u = dist * 2 * d / sup;
            let u2 = u * u;
            let u3 = u2 * u;
            if u <= d {
                // 1.5x³ − 2.5x² + 1
                (3 * u3 - 5 * u2 * d + 2 * d * d * d) >> 33
            } else {
                // −0.5x³ + 2.5x² − 4x + 2
                (-u3 + 5 * u2 * d - 8 * u * d * d + 4 * d * d * d) >> 33
            }
        }
    }
}

/// 越界索引按边缘复制（clamp）。
fn at(i: isize, n: usize) -> usize {
    i.clamp(0, n as isize - 1) as usize
}

/// 单平面缩放。`csc` 里做 4:2:0 → 4:4:4 的色度上采样时也走这条（同一套坐标口径，
/// 免得「缩放」和「色度上采样」各有一套像素中心约定）。
pub(crate) fn scale_plane(
    src: &[u8],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    algo: Algo,
) -> Vec<u8> {
    // 1:1 直接复制 —— 不是优化，是**判据**：缩放不动就该逐字节不变。
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let tx = Taps::new(dw, sw, algo);
    let ty = Taps::new(dh, sh, algo);
    let mid = h_conv(src, sw, sh, &tx);
    v_conv(&mid, dw, sh, &ty)
}

/// 水平卷积 → 1/256 单位的中间平面。
fn h_conv(src: &[u8], sw: usize, sh: usize, t: &Taps) -> Vec<u16> {
    let dw = t.first.len();
    let mut out = vec![0u16; dw * sh];
    for y in 0..sh {
        let row = &src[y * sw..y * sw + sw];
        let dst = &mut out[y * dw..y * dw + dw];
        for (d, o) in dst.iter_mut().enumerate() {
            let mut sum = 0i64;
            for (k, &wk) in t.w[d].iter().enumerate() {
                sum += i64::from(row[at(t.first[d] + k as isize, sw)]) * i64::from(wk);
            }
            // 权重 16 位 → 结果是 1/65536 像素；右移 8 得中间平面的 1/256 单位。
            // 负权重会让中间值越界（振铃），必须 clamp。
            *o = ((sum + 128) >> 8).clamp(0, 65280) as u16;
        }
    }
    out
}

/// 垂直卷积：1/256 单位的中间平面 → 8 位。
fn v_conv(mid: &[u16], dw: usize, sh: usize, t: &Taps) -> Vec<u8> {
    let dh = t.first.len();
    let mut out = vec![0u8; dw * dh];
    for (d, dst) in out.chunks_mut(dw).enumerate() {
        for (x, o) in dst.iter_mut().enumerate() {
            let mut sum = 0i64;
            for (k, &wk) in t.w[d].iter().enumerate() {
                let row = at(t.first[d] + k as isize, sh) * dw;
                sum += i64::from(mid[row + x]) * i64::from(wk);
            }
            // 中间平面是 1/256 单位、权重 16 位 → 合计 2²⁴，右移 24 回到 8 位。
            *o = ((sum + (1 << 23)) >> 24).clamp(0, 255) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一帧 4:2:0，平面内容由闭包给（方便做解析函数）。
    fn frame(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Frame {
        let (cw, ch) = (chroma_len(w as usize), chroma_len(h as usize));
        Frame {
            width: w,
            height: h,
            y: (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .map(|(x, y)| f(x, y))
                .collect(),
            u: vec![128; cw * ch],
            v: vec![128; cw * ch],
        }
    }

    /// 1:1 缩放必须**逐字节**不变 —— 连舍入都不许有。
    #[test]
    fn same_size_is_byte_identical() {
        let src = frame(37, 23, |x, y| ((x * 7 + y * 13) % 256) as u8);
        for algo in [Algo::Bilinear, Algo::Bicubic] {
            let out = resize(&src, 37, 23, algo).unwrap();
            assert_eq!(out.y, src.y, "{algo:?} 1:1 改动了亮度");
            assert_eq!(out.u, src.u, "{algo:?} 1:1 改动了色度");
            assert_eq!((out.width, out.height), (37, 23));
        }
    }

    /// 3:1 缩小时脉冲必须落进**正确的那个**目标像素，峰值 = `255 / 3`。
    ///
    /// 缩小时不能要求「精确命中原像素」—— 抗混叠的代价就是混邻居（3 个源像素
    /// 合成 1 个，脉冲被摊成 `1/3`）。所以这里验两件事：
    ///
    /// 1. **位置**：源 x=7 的采样位置是 `d=2`（`(2.5)·3 − 0.5 = 7.0`），脉冲只该
    ///    出现在那一格 —— 差半个像素的实现会把它分到 d=1 / d=3；
    /// 2. **能量**：`255/3 = 85`，说明核被正确归一化（没归一化会得到 255 或更小）。
    #[test]
    fn three_to_one_shrink_puts_the_pulse_in_the_right_bucket() {
        let mut src = frame(24, 2, |_, _| 0);
        src.y[7] = 255;
        let out = resize(&src, 8, 2, Algo::Bilinear).unwrap();
        let row = &out.y[..8];
        assert!(
            (i32::from(row[2]) - 85).abs() <= 1,
            "峰值在 d=2 应是 85，实际 {}",
            row[2]
        );
        for (d, &v) in row.iter().enumerate() {
            if d != 2 {
                assert_eq!(v, 0, "d={d} 不该有脉冲能量");
            }
        }
    }

    /// 2× 放大的**解析值**：目标像素中心落在源像素之间的 1/4 处。
    ///
    /// `d = 2i + 1` 的采样位置是 `i + 0.25` → 双线性给 `(3A + B) / 4`；
    /// `d = 2i` 是 `i − 0.25` → `(A(i−1) + 3A(i)) / 4`。**不是 1/2** ——
    /// 差半个像素的实现会给出 `(A+B)/2`，正好差一个相位。
    #[test]
    fn bilinear_doubling_uses_quarter_phase() {
        // 垂直方向是常数，把水平相位单独隔离出来。
        let src = frame(16, 4, |x, _| (x * 16) as u8);
        let out = resize(&src, 32, 8, Algo::Bilinear).unwrap();
        let row = &out.y[..32];
        for i in 0..15usize {
            let a = i32::from(src.y[i]);
            let b = i32::from(src.y[i + 1]);
            assert_eq!(row[2 * i + 1] as i32, (3 * a + b + 2) / 4, "i={i} 的 1/4 点");
            let prev = if i == 0 { a } else { i32::from(src.y[i - 1]) };
            assert_eq!(row[2 * i] as i32, (prev + 3 * a + 2) / 4, "i={i} 的 3/4 点");
        }
    }

    /// 恒定色块缩放之后必须还是那个常数（任何抽头、任何尺寸）。
    #[test]
    fn constant_planes_stay_constant() {
        let src = frame(30, 18, |_, _| 137);
        for (w, h) in [(60u32, 36u32), (15, 9), (7, 5), (31, 19)] {
            for algo in [Algo::Bilinear, Algo::Bicubic] {
                let out = resize(&src, w, h, algo).unwrap();
                assert!(
                    out.y.iter().all(|&v| v == 137),
                    "{algo:?} {w}x{h} 把常数 137 缩成了 {}..{}",
                    out.y.iter().min().unwrap(),
                    out.y.iter().max().unwrap()
                );
            }
        }
    }

    /// 2:1 缩小的脉冲响应必须与「拉伸到半径 2 的 tent 核」一致。
    ///
    /// 实测 ffmpeg 在这个输入上给 `96 / 32`（`0.375 / 0.125`）。不拉伸核的实现
    /// 会把脉冲整个塞进一个目标像素（`128 / 0`）—— 那正是混叠的样子。
    #[test]
    fn shrink_stretches_the_kernel() {
        let mut src = frame(16, 2, |_, _| 0);
        src.y[3] = 255;
        let out = resize(&src, 8, 2, Algo::Bilinear).unwrap();
        let row = &out.y[..8];
        // 允许 ±1 LSB 的定点舍入。
        assert!((i32::from(row[1]) - 96).abs() <= 1, "左侧抽头是 {}", row[1]);
        assert!((i32::from(row[2]) - 32).abs() <= 1, "右侧抽头是 {}", row[2]);
        assert_eq!(row[0], 0);
        assert_eq!(row[3], 0);
    }

    /// 每个目标位置的权重和必须恰为 65536（否则恒定色块会抖）。
    #[test]
    fn tap_weights_sum_to_one() {
        for (dst, src) in [(32usize, 16usize), (8, 16), (10, 30), (7, 5), (1920, 480)] {
            for algo in [Algo::Bilinear, Algo::Bicubic] {
                let t = Taps::new(dst, src, algo);
                assert_eq!(t.first.len(), dst);
                for (d, row) in t.w.iter().enumerate() {
                    let sum: i32 = row.iter().sum();
                    assert_eq!(sum, 65536, "{algo:?} {src}->{dst} 第 {d} 个位置的权重和 {sum}");
                }
            }
        }
    }

    /// 尺寸不对要明确报错，不猜。
    #[test]
    fn bad_input_is_rejected() {
        let src = frame(16, 8, |_, _| 0);
        assert!(resize(&src, 0, 8, Algo::Bilinear).is_err());
        assert!(resize(&src, 16, 0, Algo::Bilinear).is_err());
        let mut broken = frame(16, 8, |_, _| 0);
        broken.u.pop();
        assert!(resize(&broken, 32, 16, Algo::Bilinear).is_err());
    }
}
