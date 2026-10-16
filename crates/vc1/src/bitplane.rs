//! 位平面（bitplane）解码。
//!
//! P 帧里「这个宏块跳过吗」「这个宏块是 4MV 吗」这类**逐宏块的一位标志**，VC-1 不是
//! 逐宏块写在宏块层里，而是整帧先压成一个位平面塞在**帧头**里。这是它相对
//! MS-MPEG-4 家族（MP43 / WMV1 / WMV2）最不一样的地方：帧头不是「几十个标量」，
//! 而是「标量 + 一两张整帧的位图」。
//!
//! 七种编码方式（`IMODE_*`）：原样（宏块层里逐块读）、2×1、2×3、3×2、跳行、跳列。
//! 其中 2×1 / 2×3 / 3×2 是「一次 VLC 出多个宏块」，跳行 / 跳列是「整行/整列共用一位」。
//! 前四种里的 `DIFF*` 还要再过一遍**差分算子**（沿行、列做异或累加）。
//!
//! 与上游的对照：这一份对应 `vc1.c` 的 `bitplane_decoding()` / `decode_rowskip()` /
//! `decode_colskip()`。**逐字节等价**是判据 —— 位平面错了，宏块层的跳过标志就全错，
//! 而且是「大部分帧看着还行」的那种错。

use v2v_msmpeg4::{BitReader, Result, Vlc};

use crate::tables;

/// 位平面的编码方式。数值即 `IMODE_*`（见 `tables.rs`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imode {
    /// 数据其实在**宏块层**里逐块读（同一帧内所有「测试」共用），帧头里一位都不占。
    Raw,
    /// 每码 2 个宏块（2×1）。
    Norm2,
    /// 同 2×1，外加差分算子。
    Diff2,
    /// 每码 6 个宏块（2×3 或 3×2，按宏块网格形状选）。
    Norm6,
    /// 同 6 个宏块，外加差分算子。
    Diff6,
    /// 逐行：一位管一整行。
    RowSkip,
    /// 逐列：一位管一整列。
    ColSkip,
}

impl Imode {
    fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            x if x == tables::IMODE_RAW as u32 => Self::Raw,
            x if x == tables::IMODE_NORM2 as u32 => Self::Norm2,
            x if x == tables::IMODE_DIFF2 as u32 => Self::Diff2,
            x if x == tables::IMODE_NORM6 as u32 => Self::Norm6,
            x if x == tables::IMODE_DIFF6 as u32 => Self::Diff6,
            x if x == tables::IMODE_ROWSKIP as u32 => Self::RowSkip,
            x if x == tables::IMODE_COLSKIP as u32 => Self::ColSkip,
            _ => return None,
        })
    }
}

/// 一张解好的位平面：`width × height` 个宏块，每块一位（存在 `u8` 里）。
///
/// `data` 的行距是 `width + 1`（与上游的 `mb_stride` 同构）。多出来的那一列
/// 是**布局余量**，位流语义不依赖它 —— 6 宏块编码要按 `y * stride + x` 撒点，
/// 留一列省得每行做边界判断。
#[derive(Debug, Clone)]
pub struct Bitplane {
    /// 编码方式。
    pub imode: Imode,
    /// 位流里的「取反」位：`DIFF*` 时它是差分算子的输入，其余是整张图的取反。
    pub invert: bool,
    /// 数据在宏块层里逐块读（`IMODE_RAW`）—— 此时本结构里的 `data` **全是 0**，
    /// 不要拿它当标志用。
    pub raw: bool,
    data: Vec<u8>,
    stride: usize,
    width: usize,
    height: usize,
}

impl Bitplane {
    /// 宏块 `(x, y)` 的标志（`0` / `1`）。
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.stride + x]
    }

    /// 宏块网格的宽（宏块数）。
    pub fn width(&self) -> usize {
        self.width
    }

    /// 宏块网格的高（宏块数）。
    pub fn height(&self) -> usize {
        self.height
    }

    /// 整块原始数据（行距 `width + 1`）。给宏块层按 `(x, y)` 直取用。
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// 行距。
    pub fn stride(&self) -> usize {
        self.stride
    }
}

fn vlc_imode() -> &'static Vlc {
    static V: std::sync::OnceLock<Vlc> = std::sync::OnceLock::new();
    V.get_or_init(|| Vlc::new(&tables::IMODE_VLC))
}

fn vlc_norm2() -> &'static Vlc {
    static V: std::sync::OnceLock<Vlc> = std::sync::OnceLock::new();
    V.get_or_init(|| Vlc::new(&tables::NORM2_VLC))
}

fn vlc_norm6() -> &'static Vlc {
    static V: std::sync::OnceLock<Vlc> = std::sync::OnceLock::new();
    V.get_or_init(|| Vlc::new(&tables::NORM6_VLC))
}

/// 逐行：先一位「整行都跳过吗」，是则整行清零，否则逐宏块读一位。
fn decode_rowskip(
    data: &mut [u8],
    x0: usize,
    y0: usize,
    count: usize,
    rows: usize,
    stride: usize,
    r: &mut BitReader,
) -> Result<()> {
    for y in y0..y0 + rows {
        if r.bit()? == 0 {
            for k in 0..count {
                data[y * stride + x0 + k] = 0;
            }
        } else {
            for k in 0..count {
                data[y * stride + x0 + k] = r.bit()? as u8;
            }
        }
    }
    Ok(())
}

/// 逐列：先一位「整列都跳过吗」，是则整列清零，否则逐宏块读一位。
fn decode_colskip(
    data: &mut [u8],
    x0: usize,
    cols: usize,
    height: usize,
    stride: usize,
    r: &mut BitReader,
) -> Result<()> {
    for x in x0..x0 + cols {
        if r.bit()? == 0 {
            for y in 0..height {
                data[y * stride + x] = 0;
            }
        } else {
            for y in 0..height {
                data[y * stride + x] = r.bit()? as u8;
            }
        }
    }
    Ok(())
}

/// 解一张 `width × height` 宏块的位平面。
pub(crate) fn decode(r: &mut BitReader, width: usize, height: usize) -> Result<Bitplane> {
    // 空网格直接挡掉：差分算子那一步要写 `data[0]`，空平面上就是越界。
    // （`FrameHeader::parse` 已经挡过一次，这里再挡一次 —— 这是 pub(crate) 的入口，
    // 不该指望调用方替它守住不变量。）
    if width == 0 || height == 0 {
        return Err(v2v_msmpeg4::Error::Bad("vc1: 位平面的网格尺寸为 0"));
    }
    let stride = width + 1;
    let invert = r.bit()? != 0;
    let imode = Imode::from_u32(vlc_imode().decode(r)? as u32)
        .ok_or(v2v_msmpeg4::Error::Bad("vc1: 非法的位平面 imode"))?;

    let mut data = vec![0u8; stride * height];
    // 宏块 `(x, y)` 在 `data` 里的下标。
    let at = |x: usize, y: usize| y * stride + x;
    let total = width * height;

    match imode {
        Imode::Raw => {
            // 位流里没有数据：宏块层自己逐块读。`invert` 也**不算数**（上游注释如此）。
            return Ok(Bitplane {
                imode,
                invert,
                raw: true,
                data,
                stride,
                width,
                height,
            });
        }
        Imode::Norm2 | Imode::Diff2 => {
            // 一码两个宏块，按**行优先**的长条走。
            let mut i = 0usize;
            if total & 1 != 0 {
                data[at(0, 0)] = r.bit()? as u8;
                i = 1;
            }
            while i < total {
                let code = vlc_norm2().decode(r)? as u32;
                data[at(i % width, i / width)] = (code & 1) as u8;
                i += 1;
                data[at(i % width, i / width)] = (code >> 1) as u8;
                i += 1;
            }
        }
        Imode::Norm6 | Imode::Diff6 => {
            // 6 个宏块一码，两种摆法：2×3（3 行 2 列）或 3×2（2 行 3 列）。
            // 选哪种只看宏块网格的形状 —— 判据是上游原样搬过来的。
            if height % 3 == 0 && width % 3 != 0 {
                let mut y = 0;
                while y < height {
                    let mut x = width & 1;
                    while x < width {
                        let code = vlc_norm6().decode(r)? as u32;
                        data[at(x, y)] = (code & 1) as u8;
                        data[at(x + 1, y)] = ((code >> 1) & 1) as u8;
                        data[at(x, y + 1)] = ((code >> 2) & 1) as u8;
                        data[at(x + 1, y + 1)] = ((code >> 3) & 1) as u8;
                        data[at(x, y + 2)] = ((code >> 4) & 1) as u8;
                        data[at(x + 1, y + 2)] = ((code >> 5) & 1) as u8;
                        x += 2;
                    }
                    y += 3;
                }
                // 网格宽是奇数时，最左那一列没被 2×3 盖到。
                if width & 1 != 0 {
                    decode_colskip(&mut data, 0, 1, height, stride, r)?;
                }
            } else {
                let mut y = height & 1;
                while y < height {
                    let mut x = width % 3;
                    while x < width {
                        let code = vlc_norm6().decode(r)? as u32;
                        for k in 0..3 {
                            data[at(x + k, y)] = ((code >> k) & 1) as u8;
                        }
                        for k in 0..3 {
                            data[at(x + k, y + 1)] = ((code >> (3 + k)) & 1) as u8;
                        }
                        x += 3;
                    }
                    y += 2;
                }
                // 3×2 盖不到的两条边：最左 `width % 3` 列，和（网格高为奇数时的）最上那一行
                // 从 `width % 3` 起往右的部分。
                let x = width % 3;
                if x != 0 {
                    decode_colskip(&mut data, 0, x, height, stride, r)?;
                }
                if height & 1 != 0 {
                    decode_rowskip(&mut data, x, 0, width - x, 1, stride, r)?;
                }
            }
        }
        Imode::RowSkip => decode_rowskip(&mut data, 0, 0, width, height, stride, r)?,
        Imode::ColSkip => decode_colskip(&mut data, 0, width, height, stride, r)?,
    }

    // 差分算子：沿行、沿列做异或累加。**只在 `width` 列上跑**（不碰布局余量那列）。
    if matches!(imode, Imode::Diff2 | Imode::Diff6) {
        let inv = invert as u8;
        data[0] ^= inv;
        for x in 1..width {
            data[x] ^= data[x - 1];
        }
        for y in 1..height {
            data[y * stride] ^= data[(y - 1) * stride];
            for x in 1..width {
                let left = data[y * stride + x - 1];
                let up = data[(y - 1) * stride + x];
                data[y * stride + x] ^= if left != up { inv } else { left };
            }
        }
    } else if invert {
        // 非差分：`invert` 就是整张图取反。
        for b in data.iter_mut() {
            *b ^= 1;
        }
    }

    Ok(Bitplane {
        imode,
        invert,
        raw: false,
        data,
        stride,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imode_discriminants_match_tables() {
        // `Imode as u8` 在测试与日志里都当数值用（跟 `ffmpeg` 的 `Imode:` 对账），
        // 所以枚举顺序**必须**与 `tables::IMODE_*` 一致。顺序被改过就在这里响。
        assert_eq!(Imode::Raw as u8, tables::IMODE_RAW);
        assert_eq!(Imode::Norm2 as u8, tables::IMODE_NORM2);
        assert_eq!(Imode::Diff2 as u8, tables::IMODE_DIFF2);
        assert_eq!(Imode::Norm6 as u8, tables::IMODE_NORM6);
        assert_eq!(Imode::Diff6 as u8, tables::IMODE_DIFF6);
        assert_eq!(Imode::RowSkip as u8, tables::IMODE_ROWSKIP);
        assert_eq!(Imode::ColSkip as u8, tables::IMODE_COLSKIP);
    }

    /// 位平面解码是**索引最密**的一段（按 `y * stride + x` 撒点、还要按网格形状分两种
    /// 摆法），所以拿随机位流 × 各种网格尺寸轰它 —— 判据是「允许 `Err`，禁止 `panic`」。
    ///
    /// 零样本、纯内存，所以 CI 里也真跑。
    #[test]
    fn decode_never_panics_on_random_bits() {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut ok = 0usize;
        for w in 1..14usize {
            for h in 1..14usize {
                for _ in 0..40 {
                    let bytes: Vec<u8> = (0..16).map(|_| next() as u8).collect();
                    let mut r = BitReader::new(&bytes);
                    match decode(&mut r, w, h) {
                        Ok(p) => {
                            ok += 1;
                            // 解出来的平面必须能按 (x, y) 取到 —— 顺带验一遍布局。
                            for y in 0..h {
                                for x in 0..w {
                                    assert!(p.get(x, y) <= 1);
                                }
                            }
                        }
                        Err(_) => {}
                    }
                }
            }
        }
        // 随机位流下「能解出来」的应该不少 —— 全 Err 说明测试根本没走到里面。
        assert!(ok > 100, "随机位流几乎没解出东西（{ok} 次），这一局没测到实处");
    }

    /// 空网格不许 panic（差分算子会写 `data[0]`）。
    #[test]
    fn empty_grid_is_rejected_not_panicking() {
        let bytes = [0u8; 16];
        let mut r = BitReader::new(&bytes);
        assert!(decode(&mut r, 0, 4).is_err());
        let mut r = BitReader::new(&bytes);
        assert!(decode(&mut r, 4, 0).is_err());
    }
}
