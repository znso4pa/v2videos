//! 去隔行：`bob`（单场插值成帧）与 `weave`（两场交织成帧）。
//!
//! ## 为什么只有这两个
//!
//! 运动自适应去隔行（yadif 那一类）要**前后帧**做运动检测，属于「有时间维度的
//! 滤镜」，与这里「一帧进、一帧出」的形状不同；而且它的判据只能是「对着 yadif
//! 比 PSNR」—— 那是**另一种算法**的比较，不是正确性比较。所以先做两个**几何**
//! 算子：它们的正确答案是唯一的，能逐字节验。
//!
//! - [`bob`]：取一场（偶数行或奇数行），缺的行用上下相邻行平均补。**高度不变**，
//!   时间上每场各出一帧（帧率翻倍）—— 这里只做空间那一半，帧率由调用方决定。
//! - [`weave`]：把同一时刻的两场拼回一帧。**无损**，判据是逐字节。
//!
//! ## 色度也按同样规则处理
//!
//! 4:2:0 的色度平面是 `ceil(h/2)` 行，隔行信息在它上面同样是「隔一行取一行」。
//! 所以两个算子对三个平面用**同一套行规则**（只是行数不同），不搞特殊。

use v2v_kernel::error::{Error, Result};
use v2v_kernel::mc::Frame;

/// 哪一场。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    /// 顶场：第 0、2、4… 行。
    Top,
    /// 底场：第 1、3、5… 行。
    Bottom,
}

impl Field {
    /// 这一场保留的行是不是偶数行。
    fn keeps_even(self) -> bool {
        self == Self::Top
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Top => "top",
            Self::Bottom => "bottom",
        }
    }
}

/// 4:2:0 的色度平面边长。
fn chroma_len(n: usize) -> usize {
    (n + 1) / 2
}

/// 单场 → 完整帧：保留本场的行，其余行用**上下相邻行平均**补。
///
/// 边界（第 0 行或最后一行要插值时）用唯一存在的那个邻居，不补黑边 ——
/// 补黑会在画面最上/最下留一条暗边，那种边在缩略图上特别显眼。
pub fn bob(frame: &Frame, field: Field) -> Result<Frame> {
    let (w, h) = (frame.width as usize, frame.height as usize);
    check(frame)?;
    Ok(Frame {
        width: frame.width,
        height: frame.height,
        y: bob_plane(&frame.y, w, h, field),
        u: bob_plane(&frame.u, chroma_len(w), chroma_len(h), field),
        v: bob_plane(&frame.v, chroma_len(w), chroma_len(h), field),
    })
}

fn bob_plane(src: &[u8], w: usize, h: usize, field: Field) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        // 本场的行原样搬（**逐字节**），另一场的行插值。
        let keep = (y % 2 == 0) == field.keeps_even();
        let dst = &mut out[y * w..y * w + w];
        if keep {
            dst.copy_from_slice(&src[y * w..y * w + w]);
            continue;
        }
        let up = if y > 0 { y - 1 } else { y + 1 };
        let down = if y + 1 < h { y + 1 } else { y - 1 };
        for x in 0..w {
            let a = u32::from(src[up * w + x]);
            let b = u32::from(src[down * w + x]);
            dst[x] = ((a + b + 1) / 2) as u8;
        }
    }
    out
}

/// 两场交织成一帧：`top` 供偶数行、`bottom` 供奇数行。
///
/// 判据是**逐字节**：交织是纯重排，没有任何插值，改动一个字节就是错的。
pub fn weave(top: &Frame, bottom: &Frame) -> Result<Frame> {
    if top.width != bottom.width || top.height != bottom.height {
        return Err(Error::Bad("两场的尺寸不一致，无法交织"));
    }
    check(top)?;
    check(bottom)?;
    let (w, h) = (top.width as usize, top.height as usize);
    Ok(Frame {
        width: top.width,
        height: top.height,
        y: weave_plane(&top.y, &bottom.y, w, h),
        u: weave_plane(&top.u, &bottom.u, chroma_len(w), chroma_len(h)),
        v: weave_plane(&top.v, &bottom.v, chroma_len(w), chroma_len(h)),
    })
}

fn weave_plane(top: &[u8], bottom: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let src = if y % 2 == 0 { top } else { bottom };
        out[y * w..y * w + w].copy_from_slice(&src[y * w..y * w + w]);
    }
    out
}

/// 平面长度与宽高对不上就报错 —— 别在越界之后才 panic。
fn check(f: &Frame) -> Result<()> {
    let (w, h) = (f.width as usize, f.height as usize);
    if w == 0 || h == 0 {
        return Err(Error::Bad("帧的宽或高是 0"));
    }
    let (cw, ch) = (chroma_len(w), chroma_len(h));
    if f.y.len() != w * h || f.u.len() != cw * ch || f.v.len() != cw * ch {
        return Err(Error::Bad("平面长度与宽高不符（不是 4:2:0）"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32) -> Frame {
        let (cw, ch) = (chroma_len(w as usize), chroma_len(h as usize));
        Frame {
            width: w,
            height: h,
            y: (0..w * h).map(|i| (i * 7 % 256) as u8).collect(),
            u: (0..(cw * ch) as u32).map(|i| (100 + i % 50) as u8).collect(),
            v: (0..(cw * ch) as u32).map(|i| (200 - i % 50) as u8).collect(),
        }
    }

    /// bob 保留的那些行必须**逐字节**等于源（它只是挑行，不该重采样）。
    #[test]
    fn bob_keeps_its_own_field_byte_identical() {
        let f = frame(16, 10);
        for (field, even) in [(Field::Top, true), (Field::Bottom, false)] {
            let out = bob(&f, field).unwrap();
            for y in 0..10usize {
                if (y % 2 == 0) == even {
                    assert_eq!(
                        &out.y[y * 16..y * 16 + 16],
                        &f.y[y * 16..y * 16 + 16],
                        "{field:?} 第 {y} 行被改了"
                    );
                }
            }
        }
    }

    /// 插值行是上下两行的平均（四舍五入）。
    #[test]
    fn bob_interpolates_the_missing_field() {
        let f = frame(8, 8);
        let out = bob(&f, Field::Top).unwrap();
        for y in (1..8usize).step_by(2) {
            // 边界与实现同一套规则：最后一行没有「下一行」，就退回上一行。
            let up = if y > 0 { y - 1 } else { y + 1 };
            let down = if y + 1 < 8 { y + 1 } else { y - 1 };
            for x in 0..8usize {
                let a = u32::from(f.y[up * 8 + x]);
                let b = u32::from(f.y[down * 8 + x]);
                assert_eq!(out.y[y * 8 + x], ((a + b + 1) / 2) as u8, "第 {y} 行 {x} 列");
            }
        }
    }

    /// weave 是纯重排：偶数行来自 top、奇数行来自 bottom，一个字节都不许变。
    #[test]
    fn weave_is_a_pure_rearrangement() {
        let top = frame(16, 10);
        let bottom = frame(16, 10);
        let out = weave(&top, &bottom).unwrap();
        for y in 0..10usize {
            let src = if y % 2 == 0 { &top.y } else { &bottom.y };
            assert_eq!(&out.y[y * 16..y * 16 + 16], &src[y * 16..y * 16 + 16]);
        }
        assert_eq!(out.u.len(), top.u.len());
    }

    /// bob → weave 不是恒等（bob 丢了一半行），但「bob 顶场 + bob 底场 交织」
    /// 必须**还原原帧** —— 这条把两个算子的行选择口径钉在一起。
    #[test]
    fn weaving_the_two_bobs_restores_the_frame() {
        let f = frame(12, 8);
        let t = bob(&f, Field::Top).unwrap();
        let b = bob(&f, Field::Bottom).unwrap();
        let back = weave(&t, &b).unwrap();
        assert_eq!(back.y, f.y, "亮度没还原");
        assert_eq!(back.u, f.u, "U 没还原");
        assert_eq!(back.v, f.v, "V 没还原");
    }

    #[test]
    fn size_mismatch_is_rejected() {
        assert!(weave(&frame(16, 8), &frame(16, 10)).is_err());
        let mut broken = frame(16, 8);
        broken.y.pop();
        assert!(bob(&broken, Field::Top).is_err());
    }
}
