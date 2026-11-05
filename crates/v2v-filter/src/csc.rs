//! 色彩空间转换：YUV 4:2:0 / 4:4:4 ↔ RGB24，以及 NV12 的打包 / 解包。
//!
//! ## 系数是 BT.601 limited range，不是「随便一组」
//!
//! 视频里的 YUV 绝大多数是 **limited range**（Y 16..235、UV 16..240），不是 JPEG 那套
//! full range（0..255）。用错系的症状很具体：纯黑变成 `(0, 0, 0)` 还是 `(16, 16, 16)`
//! 取决于系数，而**两种都能出图**，只是暗部发灰、亮部发白 —— 肉眼要并排看才认得出。
//!
//! 这里用的是 swscale 的整数系数（`1.164383` / `1.596027` / `0.812968` / `0.391762` /
//! `2.017232`，各乘 65536 后取整），所以与 ffmpeg 的 `format=yuv420p` 是同一套。
//!
//! ## 为什么拆成 4:4:4 与 4:2:0 两层
//!
//! 色度下采样（4:2:0）**丢信息**，往返必然有损；把「逐像素的矩阵运算」和「色度
//! 抽样」混在一起，往返误差就说不清是谁的。分开之后：
//!
//! - [`yuv444_to_rgb24`] / [`rgb24_to_yuv444`] 是纯矩阵运算，**RGB 起点**的往返
//!   必须 **≤ 1 LSB**；
//! - [`rgb24_to_yuv420p`] 的 2×2 色度平均是**故意**的有损步骤，判据换成
//!   「恒定色度的区域往返 ≤ 1 LSB」。
//!
//! ⚠️ **往返的起点必须是 RGB，不能是 YUV。** limited range 的 YUV 立方体
//! （`Y 16..235`、`UV 16..240`）比 RGB 能表示的范围**大**：边角那些点要求
//! `R` 或 `B` 为负（超色域），进 RGB 一定被 clamp，再变回来当然对不上。
//! 拿 YUV 起点做这条判据会量出几十 LSB 的「偏差」—— 那不是转换错了，
//! 是判据问了一个没有答案的问题。
//!
//! ## 色度上采样走缩放那条路
//!
//! 4:2:0 → 4:4:4 的色度放大用的是 [`crate::scale::scale_plane`] 的双线性，
//! **不是**最近邻：最近邻会把 2×2 的色块放大成方块，彩色边缘出现阶梯。复用同一条
//! 路径的另一个好处是「像素中心」只有一份定义。

use v2v_kernel::error::{Error, Result};
use v2v_kernel::mc::Frame;

use crate::scale::{scale_plane, Algo};

// BT.601 limited range，各乘 65536 取整（与 swscale 同一套）。
const COEF_Y: i32 = 76309; // 1.164383
const COEF_RV: i32 = 104597; // 1.596027
const COEF_GU: i32 = -25673; // -0.391762
const COEF_GV: i32 = -53279; // -0.812968
const COEF_BU: i32 = 132205; // 2.017232

// 反向矩阵（RGB → YUV），同样 ×65536。
const R2Y_R: i32 = 16829; // 0.256788
const R2Y_G: i32 = 33039; // 0.504129
const R2Y_B: i32 = 6417; // 0.097906
const R2U_R: i32 = -9714; // -0.148223
const R2U_G: i32 = -19070; // -0.290993
const R2U_B: i32 = 28784; // 0.439216
const R2V_R: i32 = 28784; // 0.439216
const R2V_G: i32 = -24103; // -0.367788
const R2V_B: i32 = -4681; // -0.071427

fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// 4:2:0 的平面边长（向上取整）。
fn chroma_len(n: usize) -> usize {
    (n + 1) / 2
}

/// YUV 4:4:4 → RGB24（逐像素，无抽样）。
pub fn yuv444_to_rgb24(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
    let n = w.checked_mul(h).ok_or(Error::Bad("宽高相乘溢出"))?;
    if y.len() < n || u.len() < n || v.len() < n {
        return Err(Error::Bad("4:4:4 平面长度不足"));
    }
    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        let yy = (i32::from(y[i]) - 16) * COEF_Y;
        let uu = i32::from(u[i]) - 128;
        let vv = i32::from(v[i]) - 128;
        out[i * 3] = clamp_u8((yy + COEF_RV * vv + 32768) >> 16);
        out[i * 3 + 1] = clamp_u8((yy + COEF_GU * uu + COEF_GV * vv + 32768) >> 16);
        out[i * 3 + 2] = clamp_u8((yy + COEF_BU * uu + 32768) >> 16);
    }
    Ok(out)
}

/// RGB24 → YUV 4:4:4（逐像素，无抽样）。
pub fn rgb24_to_yuv444(rgb: &[u8], w: usize, h: usize) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let n = w.checked_mul(h).ok_or(Error::Bad("宽高相乘溢出"))?;
    if rgb.len() < n * 3 {
        return Err(Error::Bad("RGB24 长度不足"));
    }
    let mut y = vec![0u8; n];
    let mut u = vec![0u8; n];
    let mut v = vec![0u8; n];
    for i in 0..n {
        let r = i32::from(rgb[i * 3]);
        let g = i32::from(rgb[i * 3 + 1]);
        let b = i32::from(rgb[i * 3 + 2]);
        // 常数项也放进定点里（`16 << 16`），这样只有一次舍入。
        y[i] = clamp_u8(((16 << 16) + R2Y_R * r + R2Y_G * g + R2Y_B * b + 32768) >> 16);
        u[i] = clamp_u8(((128 << 16) + R2U_R * r + R2U_G * g + R2U_B * b + 32768) >> 16);
        v[i] = clamp_u8(((128 << 16) + R2V_R * r + R2V_G * g + R2V_B * b + 32768) >> 16);
    }
    Ok((y, u, v))
}

/// YUV 4:2:0 → RGB24。色度双线性放大到亮度尺寸。
pub fn yuv420p_to_rgb24(frame: &Frame) -> Result<Vec<u8>> {
    let (w, h) = (frame.width as usize, frame.height as usize);
    if w == 0 || h == 0 {
        return Err(Error::Bad("帧的宽或高是 0"));
    }
    let (cw, ch) = (chroma_len(w), chroma_len(h));
    if frame.y.len() != w * h {
        return Err(Error::Bad("亮度平面长度与宽高不符"));
    }
    if frame.u.len() != cw * ch || frame.v.len() != cw * ch {
        return Err(Error::Bad("色度平面长度与宽高不符（不是 4:2:0）"));
    }
    // 色度尺寸恰好是整数倍时 1:1 复制，否则双线性放大（与缩放的像素中心同口径）。
    let u = scale_plane(&frame.u, cw, ch, w, h, Algo::Bilinear);
    let v = scale_plane(&frame.v, cw, ch, w, h, Algo::Bilinear);
    yuv444_to_rgb24(&frame.y, &u, &v, w, h)
}

/// RGB24 → YUV 4:2:0。色度按 2×2 平均下采样（与 ffmpeg 默认一致）。
pub fn rgb24_to_yuv420p(rgb: &[u8], w: u32, h: u32) -> Result<Frame> {
    let (w, h) = (w as usize, h as usize);
    if w == 0 || h == 0 {
        return Err(Error::Bad("宽或高是 0"));
    }
    let (y444, u444, v444) = rgb24_to_yuv444(rgb, w, h)?;
    let (cw, ch) = (chroma_len(w), chroma_len(h));
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for cy in 0..ch {
        for cx in 0..cw {
            let mut su = 0u32;
            let mut sv = 0u32;
            let mut cnt = 0u32;
            // 奇数宽高时最后一列 / 行只有 1 个或 2 个样本 —— 按实际存在的算，
            // 别用「固定 4 个」去凑（那会把边缘的色度拉偏）。
            for dy in 0..2 {
                for dx in 0..2 {
                    let (x, yy) = (cx * 2 + dx, cy * 2 + dy);
                    if x < w && yy < h {
                        let i = yy * w + x;
                        su += u32::from(u444[i]);
                        sv += u32::from(v444[i]);
                        cnt += 1;
                    }
                }
            }
            u[cy * cw + cx] = ((su + cnt / 2) / cnt) as u8;
            v[cy * cw + cx] = ((sv + cnt / 2) / cnt) as u8;
        }
    }
    Ok(Frame {
        width: w as u32,
        height: h as u32,
        y: y444,
        u,
        v,
    })
}

/// YUV 4:2:0 → NV12（`Y` 平面 + 交错的 `UVUV…`）。
pub fn yuv420p_to_nv12(frame: &Frame) -> Result<Vec<u8>> {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let (cw, ch) = (chroma_len(w), chroma_len(h));
    if frame.y.len() != w * h || frame.u.len() != cw * ch || frame.v.len() != cw * ch {
        return Err(Error::Bad("平面长度与宽高不符（不是 4:2:0）"));
    }
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    out.extend_from_slice(&frame.y);
    for i in 0..cw * ch {
        out.push(frame.u[i]);
        out.push(frame.v[i]);
    }
    Ok(out)
}

/// NV12 → YUV 4:2:0。
pub fn nv12_to_yuv420p(nv12: &[u8], w: u32, h: u32) -> Result<Frame> {
    let (w, h) = (w as usize, h as usize);
    if w == 0 || h == 0 {
        return Err(Error::Bad("宽或高是 0"));
    }
    let (cw, ch) = (chroma_len(w), chroma_len(h));
    let (yn, cn) = (w * h, cw * ch);
    if nv12.len() < yn + 2 * cn {
        return Err(Error::Bad("NV12 长度不足"));
    }
    let mut u = vec![0u8; cn];
    let mut v = vec![0u8; cn];
    for i in 0..cn {
        u[i] = nv12[yn + 2 * i];
        v[i] = nv12[yn + 2 * i + 1];
    }
    Ok(Frame {
        width: w as u32,
        height: h as u32,
        y: nv12[..yn].to_vec(),
        u,
        v,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BT.601 limited range 的三个锚点：黑、白、纯红。
    ///
    /// 用错系数（比如 full range 的 JPEG 那套）时，黑会变成 `(16,16,16)`、
    /// 白会变成 `(235,235,235)` —— 都能出图，只是整体发灰。
    #[test]
    fn bt601_anchors_are_exact() {
        let (y, u, v) = ([16u8], [128u8], [128u8]);
        assert_eq!(yuv444_to_rgb24(&y, &u, &v, 1, 1).unwrap(), vec![0, 0, 0]);

        let (y, u, v) = ([235u8], [128u8], [128u8]);
        assert_eq!(
            yuv444_to_rgb24(&y, &u, &v, 1, 1).unwrap(),
            vec![255, 255, 255]
        );

        // 标准红（BT.601）：Y=81 U=90 V=240。
        let (y, u, v) = ([81u8], [90u8], [240u8]);
        let rgb = yuv444_to_rgb24(&y, &u, &v, 1, 1).unwrap();
        assert!(
            rgb[0] >= 250 && rgb[1] <= 5 && rgb[2] <= 5,
            "纯红解成了 {rgb:?}"
        );
    }

    /// RGB → YUV → RGB 往返，每个通道都必须 **≤ 2 LSB**。
    ///
    /// ⚠️ **起点必须是 RGB**，不能是 YUV。limited range 的 YUV 立方体
    /// （`Y 16..235`、`UV 16..240`）比 RGB 能表示的范围**大** —— 边角那些点对应
    /// 超色域颜色（例如 `Y=16,U=16,V=16` 要求 `R = B = −179`），进 RGB 必然被
    /// clamp，再怎么改系数也回不来。实测用 YUV 起点做这条判据会看到 **99 LSB**
    /// 的「偏差」，那不是转换错了，是**判据问了一个没有答案的问题**。
    ///
    /// ## 为什么门限是 2 而不是 1
    ///
    /// 门限由步长结构定，不是「调到能过」：中间那步把 YUV 存成 u8，色度的**一个**
    /// 量化步长在 RGB 的 B 通道上是 **2.017** 倍（`COEF_BU / 65536`）。两次量化各
    /// 贡献半个步长，最坏情况
    ///
    /// ```text
    /// 0.5 · (1.164 + 2.017 + 1) ≈ 2.09  →  2 LSB
    /// ```
    ///
    /// 实测最差点 `rgb=(0,105,180)` → `yuv=(87,177,77)` → `rgb=(1,105,182)`：
    /// Y 量化 +0.45、U 量化 +0.49，反变换时分别放大 1.164 与 2.017 倍，正好卡在
    /// 上界。灰度那条另测（[`gray_roundtrip_is_within_one_lsb`]）—— 那里没有色度
    /// 放大，门限才收到 1。
    #[test]
    fn rgb_roundtrip_is_within_two_lsb() {
        // 每通道步长 15，17³ = 4913 个点，覆盖立方体棱与面。
        let mut rgb = Vec::new();
        for r in (0..=255u32).step_by(15) {
            for g in (0..=255u32).step_by(15) {
                for b in (0..=255u32).step_by(15) {
                    rgb.extend_from_slice(&[r as u8, g as u8, b as u8]);
                }
            }
        }
        let n = rgb.len() / 3;
        let (y, u, v) = rgb24_to_yuv444(&rgb, n, 1).unwrap();
        let back = yuv444_to_rgb24(&y, &u, &v, n, 1).unwrap();
        let mut worst = 0i32;
        for i in 0..n * 3 {
            worst = worst.max((i32::from(rgb[i]) - i32::from(back[i])).abs());
        }
        assert!(worst <= 2, "往返最大偏差 {worst} LSB（上界 2）");
    }

    /// 灰度（`R = G = B`）往返 ≤ 1 LSB。
    ///
    /// 灰度时 `U = V = 128` 是精确的，误差只来自 Y 的 1.164 倍放大，
    /// 上限 `0.5·1.164 + 0.5 ≈ 1.08`。这条能把「系数抄错」与「色度步长放大」
    /// 分开：前者这里也会超 1，后者这里不受影响。
    #[test]
    fn gray_roundtrip_is_within_one_lsb() {
        let mut rgb = Vec::new();
        for v in 0..=255u8 {
            rgb.extend_from_slice(&[v, v, v]);
        }
        let (y, u, v) = rgb24_to_yuv444(&rgb, 256, 1).unwrap();
        // 灰度必须落在这条线上：U/V 恰好 128（差 1 都说明矩阵串了）。
        assert!(
            u.iter().all(|&x| x == 128) && v.iter().all(|&x| x == 128),
            "灰度算出来的色度不是 128"
        );
        let back = yuv444_to_rgb24(&y, &u, &v, 256, 1).unwrap();
        let mut worst = 0i32;
        for i in 0..256 * 3 {
            worst = worst.max((i32::from(rgb[i]) - i32::from(back[i])).abs());
        }
        assert!(worst <= 1, "灰度往返最大偏差 {worst} LSB");
    }

    /// NV12 打包 / 解包必须**逐字节**往返（纯重排，不该有任何损失）。
    #[test]
    fn nv12_roundtrip_is_byte_identical() {
        let f = Frame {
            width: 7,
            height: 5,
            y: (0..35).map(|i| i as u8).collect(),
            u: (0..12).map(|i| 100 + i as u8).collect(),
            v: (0..12).map(|i| 50 + i as u8).collect(),
        };
        let nv12 = yuv420p_to_nv12(&f).unwrap();
        assert_eq!(nv12.len(), 35 + 24);
        // 交错顺序是 U,V,U,V…（NV12 的约定；NV21 才是反过来）。
        assert_eq!(&nv12[35..39], &[100, 50, 101, 51]);
        let back = nv12_to_yuv420p(&nv12, 7, 5).unwrap();
        assert_eq!(back.y, f.y);
        assert_eq!(back.u, f.u);
        assert_eq!(back.v, f.v);
    }

    /// 4:2:0 往返：**色度恒定**的区域必须 ≤ 1 LSB（色度抽样本身有损，不在这里验）。
    #[test]
    fn yuv420_roundtrip_keeps_flat_chroma() {
        let f = Frame {
            width: 8,
            height: 6,
            y: (0..48).map(|i| (16 + i * 4) as u8).collect(),
            u: vec![128; 12],
            v: vec![128; 12],
        };
        let rgb = yuv420p_to_rgb24(&f).unwrap();
        let back = rgb24_to_yuv420p(&rgb, 8, 6).unwrap();
        let worst = (0..48)
            .map(|i| (i32::from(f.y[i]) - i32::from(back.y[i])).abs())
            .max()
            .unwrap();
        assert!(worst <= 1, "亮度往返最大偏差 {worst}");
        assert!(back.u.iter().all(|&x| x.abs_diff(128) <= 1), "色度漂了");
    }

    /// 尺寸不对要报错。
    #[test]
    fn bad_inputs_are_rejected() {
        assert!(yuv444_to_rgb24(&[0; 3], &[0; 3], &[0; 3], 2, 2).is_err());
        assert!(rgb24_to_yuv444(&[0; 5], 2, 2).is_err());
        assert!(nv12_to_yuv420p(&[0; 10], 4, 4).is_err());
    }
}
