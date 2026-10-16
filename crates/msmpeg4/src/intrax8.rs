//! **IntraX8** —— WMV2 与 VC-1 共用的**内帧子解码器**。
//!
//! 与 MP43/WMV1/WMV2 那条宏块链路完全不同的一个东西：它按 **8×8 块**（不是 16×16 宏块）
//! 走，而且**自己带一套空间预测 + 自己的环路滤波**，解码时直接往帧缓冲上写像素
//! （不经过宏块层的残差 + 运动补偿那条路）。
//!
//! WMV2 那边：extradata 的 `j_type` 是**能力位**，I 帧头里的一位决定这一帧走不走它 ——
//! 置位时**整个 I 帧**都交给它（帧头剩下的字段都不读）。所以没有它，那类流解到
//! 第一个 IntraX8 的 I 帧就得停。
//!
//! 与上游的对照：`intrax8.c`（熵 / 预测 / 主循环）+ `intrax8dsp.c`（空间补偿与环路滤波）。
//! 表在 `intrax8_tables.rs`（生成）。
//!
//! ## 两个「照抄才对」的地方
//!
//! - **预测状态是跨帧的**：`prediction_table` 在 `common_init` 里一次分配、**每帧不清**，
//!   所以上一帧的 orient/run 会影响下一帧。这条决定了状态得挂在 `Decoder` 上而不是
//!   每帧现建。
//! - **`x8_ac_compensation` 写的是「转置」坐标**：`B(x,y) = block[x + y*8]`，
//!   x 是行号。上游如此（`idct_permutation` 在 C 版是恒等），照抄。

use std::sync::OnceLock;

use crate::bitreader::BitReader;
use crate::error::{Error, Result};
use crate::frame::Planes;
use crate::idct::wmv2_idct_add;
use crate::intrax8_tables as t;
use crate::tables::SCAN;
use crate::vlc::Vlc;

// ---- 空间补偿用的暂存区（`intrax8dsp.c` 顶上那张图）：
//
//     |66666666|
//    3|44444444|55555555|
//    - -+--------+--------+
//    1 2|XXXXXXXX|
//
// 1 是左列再左一列、2 是左列、3 是左上角那一个像素、4 是上一行的 8 个、
// 5 是上一行再往右的 8 个、6 是上上行的 8 个。
const AREA1: usize = 0;
const AREA2: usize = 8;
const AREA3: usize = 8 + 8;
const AREA4: usize = 8 + 8 + 1;
const AREA5: usize = 8 + 8 + 1 + 8;
const AREA6: usize = 8 + 8 + 1 + 16;

/// 边缘标志（`edges`）：`1` = 本行第一个块、`2` = 第一行、`4` = 本行最后一个块。
const EDGE_LEFT: u32 = 1;
const EDGE_TOP: u32 = 2;
const EDGE_RIGHT: u32 = 4;

/// 一张解好的 `[[code, len]]` 码表（`*_STRIDE` 是每张表的码字数）。
fn table(base: &'static [[u32; 2]], stride: usize, idx: usize) -> &'static [[u32; 2]] {
    &base[idx * stride..][..stride]
}

/// 全部码表树（建一次，全局共享 —— 与 `frame.rs` 的 `DecodeTables` 同理由）。
struct Vlcs {
    /// `[高量化][帧内/帧间][表号]`。
    ac: [[[Vlc; 8]; 2]; 2],
    /// `[高量化][表号]`。
    dc: [[Vlc; 8]; 2],
    orient_high: [Vlc; 2],
    orient_low: [Vlc; 4],
}

fn vlcs() -> &'static Vlcs {
    static V: OnceLock<Vlcs> = OnceLock::new();
    V.get_or_init(|| Vlcs {
        ac: std::array::from_fn(|hi| {
            std::array::from_fn(|inter| {
                std::array::from_fn(|sel| {
                    Vlc::new(table(
                        &t::X8_AC_QUANT,
                        t::X8_AC_QUANT_STRIDE,
                        (hi * 2 + inter) * 8 + sel,
                    ))
                })
            })
        }),
        dc: std::array::from_fn(|hi| {
            std::array::from_fn(|sel| {
                Vlc::new(table(&t::X8_DC_QUANT, t::X8_DC_QUANT_STRIDE, hi * 8 + sel))
            })
        }),
        orient_high: std::array::from_fn(|sel| {
            Vlc::new(table(&t::X8_ORIENT_HIGH, t::X8_ORIENT_HIGH_STRIDE, sel))
        }),
        orient_low: std::array::from_fn(|sel| {
            Vlc::new(table(&t::X8_ORIENT_LOW, t::X8_ORIENT_LOW_STRIDE, sel))
        }),
    })
}

/// IntraX8 的解码状态。
///
/// **要跨帧活着** —— `prediction_table` 上游只在初始化时清零一次。
pub(crate) struct IntraX8 {
    /// 逐「8×8 列」的预测状态：`mb_x * 2 + (mb_y & 1)`，两个 x8 行一组。
    prediction_table: Vec<u8>,
    mb_width: usize,
    mb_height: usize,

    // ---- 每帧
    quant: u32,
    /// ⚠️ **不是** `quant`：上游的 AC 系数乘的是 `dquant = 2 * qscale`（调用方传进来的），
    /// 而 DC 与直流填充用的是 `quant = dquant >> 1`。混用会让所有 AC 系数差一半。
    dquant: u32,
    qsum: u32,
    loopfilter: bool,
    quant_dc_chroma: u32,
    divide_quant_dc_luma: u32,
    divide_quant_dc_chroma: u32,
    use_quant_matrix: bool,

    // ---- 每个块
    mb_x: usize,
    mb_y: usize,
    orient: u32,
    raw_orient: u32,
    est_run: u32,
    edges: u32,
    chroma_orient: u32,
    flat_dc: bool,
    predicted_dc: i32,

    scratchpad: [u8; 42],
    block: [i16; 64],
    /// 每帧重置的「已经选过哪张表」（位流里只在第一次用到时给表号）。
    ac_vlc: [Option<&'static Vlc>; 4],
    dc_vlc: [Option<&'static Vlc>; 3],
    orient_vlc: Option<&'static Vlc>,
}

impl IntraX8 {
    /// 建一份状态（宏块网格尺寸变了要重建 —— 预测表是按宽度分配的）。
    pub(crate) fn new(mb_width: usize, mb_height: usize) -> Self {
        Self {
            prediction_table: vec![0u8; mb_width * 2 * 2],
            mb_width,
            mb_height,
            quant: 0,
            dquant: 0,
            qsum: 0,
            loopfilter: false,
            quant_dc_chroma: 0,
            divide_quant_dc_luma: 0,
            divide_quant_dc_chroma: 0,
            use_quant_matrix: false,
            mb_x: 0,
            mb_y: 0,
            orient: 0,
            raw_orient: 0,
            est_run: 0,
            edges: 0,
            chroma_orient: 0,
            flat_dc: false,
            predicted_dc: 0,
            scratchpad: [0; 42],
            block: [0; 64],
            ac_vlc: [None; 4],
            dc_vlc: [None; 3],
            orient_vlc: None,
        }
    }

    /// 宏块网格尺寸与这份状态对得上吗。
    pub(crate) fn fits(&self, mb_width: usize, mb_height: usize) -> bool {
        self.mb_width == mb_width && self.mb_height == mb_height
    }

    fn reset_vlc_tables(&mut self) {
        self.ac_vlc = [None; 4];
        self.dc_vlc = [None; 3];
        self.orient_vlc = None;
    }

    /// 选 AC 码表（位流里 3 位，**每帧第一次用到时才读**）。
    fn select_ac_table(&mut self, r: &mut BitReader, mode: usize) -> Result<()> {
        if self.ac_vlc[mode].is_some() {
            return Ok(());
        }
        let idx = r.u(3)? as usize;
        // 两个 mode 共用同一张表（上游如此：`mode >> 1`）。
        self.ac_vlc[mode] = Some(&vlcs().ac[(self.quant < 13) as usize][mode >> 1][idx]);
        Ok(())
    }

    fn get_orient_vlc(&mut self, r: &mut BitReader) -> Result<u32> {
        if self.orient_vlc.is_none() {
            // ⚠️ 方向表这里**名字是反的**：`quant < 13`（低量化）用的是 **lowquant**
            // 那 4 张（表号 2 位），高量化才用 highquant 那 2 张（表号 1 位）。
            // 上游写的是 `j_orient_vlc[w->quant < 13][table_index]`，而
            // `j_orient_vlc[0]` 挂的是 highquant、`[1]` 挂的是 lowquant。
            let low = (self.quant < 13) as usize;
            let idx = r.u(1 + low as u32)? as usize;
            self.orient_vlc = Some(if low == 1 {
                &vlcs().orient_low[idx]
            } else {
                &vlcs().orient_high[idx]
            });
        }
        Ok(self.orient_vlc.unwrap().decode(r)? as u32)
    }

    // ================= 熵解码 =================

    /// AC 的 `(run, level, final)`。对应上游 `x8_get_ac_rlf()`。
    fn get_ac_rlf(&mut self, r: &mut BitReader, mode: usize) -> Result<(u32, u32, bool)> {
        let i = self.ac_vlc[mode].unwrap().decode(r)? as i32;

        if i < 46 {
            // 0..22：run 与 level 的紧凑编码；23..45：同一套但 `final` 置位。
            let final_ = i > 22;
            let i = if final_ { i - 23 } else { i };
            let l = (0xE50000u32 >> (i & 0x1E)) & 3;
            let mask = 0x01030Fu32 >> (l << 3);
            Ok(((i as u32) & mask, l, final_))
        } else if i < 73 {
            let i = (i - 46) as usize;
            let sm = t::X8_AC_DECODE_TABLE[i];
            let e = r.u(sm & 0xF)?;
            let mask = (sm >> 8) & 0xff;
            let sm = sm >> 16;
            Ok((
                (sm & 0xff) + (e & mask),
                (sm >> 8) + (e & !mask),
                i > (58 - 46),
            ))
        } else if i < 75 {
            let e = r.u(5)? as usize;
            let mix = t::X8_CRAZY_MIX_RUNLEVEL[e];
            Ok(((mix >> 4) as u32, (mix & 0x0F) as u32, i & 1 == 0))
        } else {
            let level = r.u(7 - 3 * (i as u32 & 1))?;
            let run = r.u(6)?;
            let final_ = r.bit()? != 0;
            Ok((run, level, final_))
        }
    }

    /// DC 的 `(level, final)`。对应上游 `x8_get_dc_rlf()`。
    fn get_dc_rlf(&mut self, r: &mut BitReader, mode: usize) -> Result<(i32, bool)> {
        if self.dc_vlc[mode].is_none() {
            let idx = r.u(3)? as usize;
            // 4 个 mode 共用同一张表。
            self.dc_vlc[mode] = Some(&vlcs().dc[(self.quant < 13) as usize][idx]);
        }
        let i = self.dc_vlc[mode].unwrap().decode(r)? as i32;

        let final_ = i > 16;
        let i = if final_ { i - 17 } else { i };
        if i <= 0 {
            return Ok((0, final_));
        }
        let mut c = (i + 1) >> 1;
        if c > 1 {
            c -= 1;
        }
        let e = r.u(c as u32)? as i32;
        let i = t::X8_DC_INDEX_OFFSET[i as usize] as i32 + (e >> 1);
        let e = -(e & 1); // 0 或 -1
        Ok(((i ^ e) - e, final_))
    }

    // ================= 预测 =================

    /// 逐块的空间预测（`x8_get_prediction()`）。会改 `orient` / `est_run` / `edges`。
    fn get_prediction(&mut self) {
        // ⚠️ C 里的 `!(w->mb_y & 1)` 是**逻辑非**（0 或 1）；Rust 的 `!` 是**按位取反**
        // （`!1usize` = 0xFFFF…FE）。照抄 `!` 会算出天文数字的下标 —— 踩过一次。
        let not_odd_y = 1 - (self.mb_y & 1);
        self.edges = (if self.mb_x == 0 { EDGE_LEFT } else { 0 })
            | (if self.mb_y == 0 { EDGE_TOP } else { 0 })
            | (if self.mb_x >= 2 * self.mb_width - 1 {
                EDGE_RIGHT
            } else {
                0
            });

        match self.edges & 3 {
            0 => {}
            1 => {
                // 只有左边有邻居：拿上面那个块的（`prediction_table[!(mb_y & 1)]`）。
                self.est_run = (self.prediction_table[not_odd_y] >> 2) as u32;
                self.orient = 1;
                return;
            }
            2 => {
                // 只有上面有邻居：拿左边那个块的。
                self.est_run =
                    (self.prediction_table[2 * self.mb_x - 2] >> 2) as u32;
                self.orient = 2;
                return;
            }
            _ => {
                // 左上角：没有邻居，取常数。
                self.est_run = 16;
                self.orient = 0;
                return;
            }
        }

        let b = self.prediction_table[2 * self.mb_x + not_odd_y];
        let a = self.prediction_table[2 * self.mb_x - 2 + (self.mb_y & 1)];
        let c = self.prediction_table[2 * self.mb_x - 2 + not_odd_y];

        self.est_run = (b.min(a) >> 2) as u32;
        // ⚠️ 这个条件**与 `edges` 无关**，看着像笔误，但上游注释说它「成了标准」——
        // 照抄（x 与 y 都为奇数时才算左上对角那个）。
        if (self.mb_x & self.mb_y) != 0 {
            self.est_run = (c.min(b.min(a)) >> 2) as u32;
        }

        let (a, b, c) = ((a & 3) as u32, (b & 3) as u32, (c & 3) as u32);
        let i = (0xFFEAF4C4u32 >> (2 * b + 8 * a)) & 3;
        self.orient = if i != 3 {
            i
        } else {
            (0xFFEAD8u32 >> (2 * c + 8 * (self.quant > 12) as u32)) & 3
        };
    }

    /// 色度的预测（`x8_get_prediction_chroma()`）—— 只定 `chroma_orient` / `edges`。
    fn get_prediction_chroma(&mut self) {
        self.edges = (if self.mb_x >> 1 == 0 { EDGE_LEFT } else { 0 })
            | (if self.mb_y >> 1 == 0 { EDGE_TOP } else { 0 })
            | (if self.mb_x >= 2 * self.mb_width - 1 {
                EDGE_RIGHT
            } else {
                0
            });

        self.raw_orient = 0;
        if self.edges & 3 != 0 {
            self.chroma_orient = 4 << ((0xCCu32 >> self.edges) & 1);
            return;
        }
        // 左边那个块（同一个 2×2 组里的）。
        self.chroma_orient = ((self.prediction_table[2 * self.mb_x - 2] & 0x03) as u32) << 2;
    }

    fn update_predictions(&mut self, orient: u32, est_run: u32) {
        self.prediction_table[self.mb_x * 2 + (self.mb_y & 1)] =
            ((est_run << 2) + (orient == 4) as u32 + 2 * (orient == 8) as u32) as u8;
    }

    // ================= 空间补偿 =================

    /// 收集边缘像素 + 统计（`x8_setup_spatial_compensation()`）。
    ///
    /// `plane` 是整块平面、`off` 是块的起点 —— 读的是 `off` 左边的列与上面的行，
    /// 所以**不能**只传块本身那一片。
    fn setup_spatial_compensation(
        plane: &[u8],
        off: usize,
        stride: usize,
        dst: &mut [u8; 42],
        edges: u32,
    ) -> (i32, i32) {
        if edges & 3 == 3 {
            // 左上角那块：全 0x80。这会让 `range == 0`，必然走 flat_dc。
            dst.fill(0x80);
            return (0, 0x80 * (8 + 1 + 8 + 2));
        }

        let mut min_pix = 256i32;
        let mut max_pix = -1i32;
        let mut sum = 0i32;

        if edges & EDGE_LEFT == 0 {
            // 左边有块：左列（area2）与再左一列（area1）。
            let mut p = off - 1;
            for i in (0..8).rev() {
                dst[AREA1 + i] = plane[p - 1];
                let c = plane[p];
                sum += c as i32;
                min_pix = min_pix.min(c as i32);
                max_pix = max_pix.max(c as i32);
                dst[AREA2 + i] = c;
                p += stride;
            }
        }

        let mut last_c = 0u8;
        if edges & EDGE_TOP == 0 {
            // 上面有行：上一行（area4/5）与上上行（area6）。
            let p = off - stride;
            for i in 0..8 {
                let c = plane[p + i];
                sum += c as i32;
                min_pix = min_pix.min(c as i32);
                max_pix = max_pix.max(c as i32);
                last_c = c;
            }
            if edges & EDGE_RIGHT != 0 {
                // 本行最后一个块：右半边没有邻居，用最后一个像素铺满。
                dst[AREA5..AREA5 + 8].fill(last_c);
                dst[AREA4..AREA4 + 8].copy_from_slice(&plane[p..p + 8]);
            } else {
                dst[AREA4..AREA4 + 16].copy_from_slice(&plane[p..p + 16]);
            }
            dst[AREA6..AREA6 + 8].copy_from_slice(&plane[p - stride..p - stride + 8]);
        }

        if edges & 3 != 0 {
            let avg = (sum + 4) >> 3;
            if edges & EDGE_LEFT != 0 {
                // 左边没块：1/2/3 三个区都用均值。
                dst[AREA1..AREA1 + 8 + 8 + 1].fill(avg as u8);
            } else {
                // 上面没行：3/4/5/6 四个区都用均值。
                dst[AREA3..AREA3 + 1 + 16 + 8].fill(avg as u8);
            }
            sum += avg * 9;
        } else {
            // 左上角那个像素单独拿（它不在 min/max 里）。
            let c = plane[off - 1 - stride];
            dst[AREA3] = c;
            sum += c as i32;
        }

        let range = max_pix - min_pix;
        sum += dst[AREA5] as i32 + dst[AREA5 + 1] as i32;
        (range, sum)
    }

    /// 12 种空间补偿（`spatial_compensation_*`）：把 `src` 里攒好的边缘像素铺成 8×8。
    fn spatial_compensation(
        mode: u32,
        src: &[u8; 42],
        dst: &mut [u8],
        off: usize,
        stride: usize,
    ) {
        match mode {
            0 => {
                // 加权（零预测）：左右两侧各自按距离衰减，再按权重表混合。
                let mut left_sum = [[0u16; 8]; 2];
                let mut top_sum = [[0u16; 8]; 2];
                for i in 0..8usize {
                    let a = (src[AREA2 + 7 - i] as u32) << 4;
                    for j in 0..8usize {
                        let p = i.abs_diff(j);
                        left_sum[p & 1][j] =
                            left_sum[p & 1][j].wrapping_add((a >> (p >> 1)) as u16);
                    }
                }
                for i in 0..12usize {
                    let a = (src[AREA4 + i] as u32) << 4;
                    let (j0, j1) = match i {
                        0..=7 => (0usize, 8usize),
                        8..=9 => (5, 8),
                        _ => (7, 8),
                    };
                    for j in j0..j1 {
                        let p = i.abs_diff(j);
                        top_sum[p & 1][j] = top_sum[p & 1][j].wrapping_add((a >> (p >> 1)) as u16);
                    }
                }
                for i in 0..8usize {
                    // 181 ≈ sqrt(2)/2 —— 把「斜着」的那一份折算进来。
                    top_sum[0][i] += (top_sum[1][i] as u32 * 181 + 128 >> 8) as u16;
                    left_sum[0][i] += (left_sum[1][i] as u32 * 181 + 128 >> 8) as u16;
                }
                for y in 0..8usize {
                    for x in 0..8usize {
                        let w = &t::X8_ZERO_PREDICTION_WEIGHTS;
                        let v = (top_sum[0][x] as u32 * w[y * 16 + x * 2] as u32
                            + left_sum[0][y] as u32 * w[y * 16 + x * 2 + 1] as u32
                            + 0x8000)
                            >> 16;
                        dst[off + y * stride + x] = v as u8;
                    }
                }
            }
            1 => {
                for y in 0..8usize {
                    for x in 0..8usize {
                        dst[off + y * stride + x] = src[AREA4 + (2 * y + x + 2).min(15)];
                    }
                }
            }
            2 => {
                for y in 0..8usize {
                    for x in 0..8usize {
                        dst[off + y * stride + x] = src[AREA4 + 1 + y + x];
                    }
                }
            }
            3 => {
                for y in 0..8usize {
                    for x in 0..8usize {
                        dst[off + y * stride + x] = src[AREA4 + ((y + 1) >> 1) + x];
                    }
                }
            }
            4 => {
                for y in 0..8usize {
                    for x in 0..8usize {
                        dst[off + y * stride + x] =
                            (src[AREA4 + x] as u32 + src[AREA6 + x] as u32 + 1 >> 1) as u8;
                    }
                }
            }
            5 => {
                for y in 0..8usize {
                    for x in 0..8usize {
                        dst[off + y * stride + x] = if 2 * x < y {
                            src[AREA2 + 9 + 2 * x - y]
                        } else {
                            src[AREA4 + x - ((y + 1) >> 1)]
                        };
                    }
                }
            }
            6 => {
                for y in 0..8usize {
                    for x in 0..8usize {
                        dst[off + y * stride + x] = src[AREA3 + x - y];
                    }
                }
            }
            7 => {
                for y in 0..8usize {
                    for x in 0..8usize {
                        dst[off + y * stride + x] = if x > 2 * y {
                            (src[AREA3 - 1 + x - 2 * y] as u32 + src[AREA3 + x - 2 * y] as u32 + 1
                                >> 1) as u8
                        } else {
                            src[AREA2 + 8 - y + (x >> 1)]
                        };
                    }
                }
            }
            8 => {
                for y in 0..8usize {
                    for x in 0..8usize {
                        dst[off + y * stride + x] =
                            (src[AREA1 + 7 - y] as u32 + src[AREA2 + 7 - y] as u32 + 1 >> 1) as u8;
                    }
                }
            }
            9 => {
                for y in 0..8usize {
                    for x in 0..8usize {
                        dst[off + y * stride + x] = src[AREA2 + 6 - (x + y).min(6)];
                    }
                }
            }
            10 => {
                for y in 0..8usize {
                    for x in 0..8usize {
                        dst[off + y * stride + x] = (src[AREA2 + 7 - y] as u32 * (8 - x) as u32
                            + src[AREA4 + x] as u32 * x as u32
                            + 4
                            >> 3) as u8;
                    }
                }
            }
            _ => {
                for y in 0..8usize {
                    for x in 0..8usize {
                        dst[off + y * stride + x] = (src[AREA2 + 7 - y] as u32 * y as u32
                            + src[AREA4 + x] as u32 * (8 - y) as u32
                            + 4
                            >> 3) as u8;
                    }
                }
            }
        }
    }

    /// 8×8 环路滤波（`x8_loop_filter()`）。
    ///
    /// `a_stride` 是「跨过边界」的方向、`b_stride` 是「沿边界走」的方向：
    /// 横边用 `(stride, 1)`、竖边用 `(1, stride)`。
    fn loop_filter(plane: &mut [u8], off: usize, a_stride: usize, b_stride: usize, quant: u32) {
        let ql = ((quant + 10) >> 3) as i32;
        let mut p = off;
        for _ in 0..8 {
            let g = |k: i32| plane[(p as i32 + k * a_stride as i32) as usize] as i32;
            let (p0, p1, p2, p3, p4) = (g(-5), g(-4), g(-3), g(-2), g(-1));
            let (p5, p6, p7, p8, p9) = (g(0), g(1), g(2), g(3), g(4));

            let mut t = ((p1 - p2).abs() <= ql) as i32
                + ((p2 - p3).abs() <= ql) as i32
                + ((p3 - p4).abs() <= ql) as i32
                + ((p4 - p5).abs() <= ql) as i32;

            // 至少要有 1 分才可能凑够 6 分。
            let mut filtered = false;
            if t > 0 {
                t += ((p5 - p6).abs() <= ql) as i32
                    + ((p6 - p7).abs() <= ql) as i32
                    + ((p7 - p8).abs() <= ql) as i32
                    + ((p8 - p9).abs() <= ql) as i32
                    + ((p0 - p1).abs() <= ql) as i32;
                if t >= 6 {
                    let mut min = p1.min(p3).min(p5).min(p8);
                    let mut max = p1.max(p3).max(p5).max(p8);
                    if max - min < 2 * quant as i32 {
                        min = min.min(p2).min(p4).min(p6).min(p7);
                        max = max.max(p2).max(p4).max(p6).max(p7);
                        if max - min < 2 * quant as i32 {
                            let set = |plane: &mut [u8], k: i32, v: i32| {
                                plane[(p as i32 + k * a_stride as i32) as usize] = v as u8;
                            };
                            set(plane, -2, (4 * p2 + 3 * p3 + p7 + 4) >> 3);
                            set(plane, -1, (3 * p2 + 3 * p4 + 2 * p7 + 4) >> 3);
                            set(plane, 0, (2 * p2 + 3 * p5 + 3 * p7 + 4) >> 3);
                            set(plane, 1, (p2 + 3 * p6 + 4 * p7 + 4) >> 3);
                            filtered = true;
                        }
                    }
                }
            }
            if !filtered {
                let x0 = (2 * p3 - 5 * p4 + 5 * p5 - 2 * p6 + 4) >> 3;
                if x0.abs() < quant as i32 {
                    let x1 = (2 * p1 - 5 * p2 + 5 * p3 - 2 * p4 + 4) >> 3;
                    let x2 = (2 * p5 - 5 * p6 + 5 * p7 - 2 * p8 + 4) >> 3;
                    let x = x0.abs() - x1.abs().min(x2.abs());
                    let m = p4 - p5;
                    if x > 0 && (m ^ x0) < 0 {
                        let sign = m >> 31;
                        let mut m = (m ^ sign) - sign;
                        m >>= 1;
                        let mut x = 5 * x >> 3;
                        if x > m {
                            x = m;
                        }
                        let x = (x ^ sign) - sign;
                        plane[(p as i32 - a_stride as i32) as usize] =
                            (plane[(p as i32 - a_stride as i32) as usize] as i32 - x) as u8;
                        plane[p] = (plane[p] as i32 + x) as u8;
                    }
                }
            }
            p += b_stride;
        }
    }

    /// AC 系数的方向补偿（`x8_ac_compensation()`）。
    ///
    /// ⚠️ `B(x, y) = block[x + y*8]` —— **x 是行号**（转置着写的），上游如此。
    fn ac_compensation(block: &mut [i16; 64], direction: u32, dc_level: i32) {
        let t = |x: i32| (x * dc_level + 0x8000) >> 16;
        let sub = |block: &mut [i16; 64], i: usize, v: i32| {
            block[i] = block[i].wrapping_sub(v as i16);
        };
        let add = |block: &mut [i16; 64], i: usize, v: i32| {
            block[i] = block[i].wrapping_add(v as i16);
        };
        match direction {
            0 => {
                let v = t(3811); // h
                sub(block, 1, v);
                sub(block, 8, v);
                let v = t(487); // e
                sub(block, 2, v);
                sub(block, 16, v);
                let v = t(506); // f
                sub(block, 3, v);
                sub(block, 24, v);
                let v = t(135); // c
                sub(block, 4, v);
                sub(block, 32, v);
                add(block, 10, v); // B(2,1)
                add(block, 17, v); // B(1,2)
                add(block, 11, v); // B(3,1)
                add(block, 25, v); // B(1,3)
                let v = t(173); // d
                sub(block, 5, v);
                sub(block, 40, v);
                let v = t(61); // b
                sub(block, 6, v);
                sub(block, 48, v);
                add(block, 13, v);
                add(block, 41, v);
                let v = t(42); // a
                sub(block, 7, v);
                sub(block, 56, v);
                add(block, 12, v);
                add(block, 33, v);
                add(block, 36, v);
                let v = t(1084); // g
                add(block, 9, v);
            }
            1 => {
                sub(block, 8, t(6269));
                sub(block, 24, t(708));
                sub(block, 40, t(172));
                sub(block, 56, t(73));
            }
            _ => {
                sub(block, 1, t(6269));
                sub(block, 3, t(708));
                sub(block, 5, t(172));
                sub(block, 7, t(73));
            }
        }
    }

    /// 解一个 8×8 块（`x8_decode_intra_mb()`）。
    #[allow(clippy::too_many_arguments)]
    fn decode_intra_mb(
        &mut self,
        r: &mut BitReader,
        plane: &mut [u8],
        off: usize,
        stride: usize,
        chroma: bool,
    ) -> Result<()> {
        self.block = [0i16; 64];

        let dc_mode = if chroma { 2 } else { self.est_run.min(1) as usize };
        let (mut dc_level, final_) = self.get_dc_rlf(r, dc_mode)?;

        let mut zeros_only = false;
        let mut n = 0u32;
        let mut pos = 0usize;

        if !final_ {
            // ---- 有 AC ----
            let mut use_quant_matrix = self.use_quant_matrix;
            let mut ac_mode;
            let est_run;
            if chroma {
                ac_mode = 1;
                est_run = 64;
            } else {
                if self.raw_orient < 3 {
                    use_quant_matrix = false;
                }
                if self.raw_orient > 4 {
                    ac_mode = 0;
                    est_run = 64;
                } else if self.est_run > 1 {
                    ac_mode = 2;
                    est_run = self.est_run;
                } else {
                    ac_mode = 3;
                    est_run = 64;
                }
            }
            self.select_ac_table(r, ac_mode)?;

            // 扫描表按 orient 选（上游那张 12 项的查表）。
            let sel = (0x928548u32 >> (2 * self.orient)) & 3;
            let scantable: &[u8; 64] = match sel {
                0 => &SCAN[0],
                1 => &SCAN[2],
                _ => &SCAN[3],
            };

            loop {
                n += 1;
                if n >= est_run {
                    ac_mode = 3;
                    self.select_ac_table(r, 3)?;
                }
                let (run, level, fin) = self.get_ac_rlf(r, ac_mode)?;
                pos += run as usize + 1;
                if pos > 63 {
                    // 也顺带兜住 VLC 解错的情况（上游注释如此）。
                    return Err(Error::Bad("IntraX8: AC 系数越界"));
                }
                let mut level = (level as i32 + 1) * self.dquant as i32;
                level += self.qsum as i32;
                let sign = -(r.bit()? as i32);
                level = (level ^ sign) - sign;
                if use_quant_matrix {
                    level = (level * t::X8_QUANT_TABLE[pos] as i32) >> 8;
                }
                self.block[scantable[pos] as usize] = level as i16;
                if fin {
                    break;
                }
            }
        } else {
            // ---- 只有 DC ----
            if self.flat_dc && ((dc_level + 1) as u32) < 3 {
                let (divide_quant, dc_quant) = if !chroma {
                    (self.divide_quant_dc_luma, self.quant)
                } else {
                    (self.divide_quant_dc_chroma, self.quant_dc_chroma)
                };
                // 原意是 `dc_level += predicted_dc / quant`，但上游在取整上丢了精度。
                dc_level += (self.predicted_dc * divide_quant as i32 + (1 << 12)) >> 13;
                let pix = ((dc_level * dc_quant as i32 + 4) >> 3).clamp(0, 255) as u8;
                for y in 0..8usize {
                    plane[off + y * stride..off + y * stride + 8].fill(pix);
                }
                if !chroma {
                    self.update_predictions(self.orient, n);
                }
                if self.loopfilter {
                    self.filter_block(plane, off, stride, zeros_only);
                }
                return Ok(());
            }
            zeros_only = dc_level == 0;
        }

        self.block[0] = if !chroma {
            dc_level.wrapping_mul(self.quant as i32) as i16
        } else {
            dc_level.wrapping_mul(self.quant_dc_chroma as i32) as i16
        };

        // AC 的方向补偿：只有 DC 不平凡、且不在左上角时才做。
        if ((dc_level + 1) as u32) >= 3 && (self.edges & 3) != 3 {
            let direction = (0x6A017Cu32 >> (self.orient * 2)) & 3;
            if direction != 3 {
                let dc = self.block[0] as i32;
                Self::ac_compensation(&mut self.block, direction, dc);
            }
        }

        if self.flat_dc {
            let pix = self.predicted_dc.clamp(0, 255) as u8;
            for y in 0..8usize {
                plane[off + y * stride..off + y * stride + 8].fill(pix);
            }
        } else {
            Self::spatial_compensation(self.orient, &self.scratchpad, plane, off, stride);
        }
        if !zeros_only {
            // ⚠️ 用的是 **WMV2 自己那份 8×8 IDCT**（上游把 `wdsp.idct_add` 换掉了）。
            // 只切「8 行 × 8 列」那一片给它 —— 别把整个平面拷出去。
            let end = off + 7 * stride + 8;
            wmv2_idct_add(&mut self.block, &mut plane[off..end], stride);
        }

        if !chroma {
            self.update_predictions(self.orient, n);
        }
        if self.loopfilter {
            self.filter_block(plane, off, stride, zeros_only);
        }
        Ok(())
    }

    fn filter_block(&self, plane: &mut [u8], off: usize, stride: usize, zeros_only: bool) {
        // 跳过宏块时：内部边不滤、与邻居的边照滤（上游那两条 `(orient | 4) == 4` 之类）。
        if self.edges & EDGE_TOP == 0 && !(zeros_only && (self.orient | 4) == 4) {
            Self::loop_filter(plane, off, stride, 1, self.quant);
        }
        if self.edges & EDGE_LEFT == 0 && !(zeros_only && (self.orient | 8) == 8) {
            Self::loop_filter(plane, off, 1, stride, self.quant);
        }
    }

    /// 解一整帧（`ff_intrax8_decode_picture()`）。
    ///
    /// `quant` = 帧头的 `qscale`、`qsum` = `(qscale - 1) | 1`（WMV2 那边就这么传的）。
    pub(crate) fn decode_picture(
        &mut self,
        r: &mut BitReader,
        planes: &mut Planes,
        quant: u32,
        qsum: u32,
        loopfilter: bool,
    ) -> Result<()> {
        if quant == 0 {
            return Err(Error::Bad("IntraX8: quant 为 0"));
        }
        self.quant = quant;
        self.dquant = 2 * quant;
        self.qsum = qsum;
        self.loopfilter = loopfilter;
        self.use_quant_matrix = r.bit()? != 0;

        self.divide_quant_dc_luma = ((1 << 16) + (quant >> 1)) / quant;
        if quant < 5 {
            self.quant_dc_chroma = quant;
            self.divide_quant_dc_chroma = self.divide_quant_dc_luma;
        } else {
            self.quant_dc_chroma = quant + ((quant + 3) >> 3);
            self.divide_quant_dc_chroma =
                ((1 << 16) + (self.quant_dc_chroma >> 1)) / self.quant_dc_chroma;
        }
        self.reset_vlc_tables();

        let (mb_w, mb_h) = (self.mb_width, self.mb_height);
        let (pw, cw) = (planes.pw, planes.pw / 2);

        for y8 in 0..mb_h * 2 {
            let y_base = y8 * 8 * pw;
            let mut c_off = (y8 & !1) * 4 * cw;
            if r.bits_left() < 1 {
                return Err(Error::Eof);
            }
            for x8 in 0..mb_w * 2 {
                self.mb_x = x8;
                self.mb_y = y8;

                // ---- 亮度 ----
                self.get_prediction();
                let off = y_base + x8 * 8;
                self.setup_spatial_predictor(r, &planes.y, off, pw, false)?;
                self.decode_intra_mb(r, &mut planes.y, off, pw, false)?;

                // ---- 色度：每个 2×2 组里只做一次（奇数 x8、奇数 y8）----
                if (x8 & y8 & 1) == 1 {
                    self.get_prediction_chroma();
                    // 色度不读 VLC（所以到不了错误分支）。
                    self.setup_spatial_predictor(r, &planes.u, c_off, cw, true)?;
                    self.decode_intra_mb(r, &mut planes.u, c_off, cw, true)?;
                    self.setup_spatial_predictor(r, &planes.v, c_off, cw, true)?;
                    self.decode_intra_mb(r, &mut planes.v, c_off, cw, true)?;
                    c_off += 8;
                }
            }
        }
        Ok(())
    }

    /// 定 `orient` / `flat_dc` / `predicted_dc`（`x8_setup_spatial_predictor()`）。
    ///
    /// 先攒边缘像素（`setup_spatial_compensation`），再按「边缘起伏」决定预测方向：
    /// 起伏小就压平成直流（`flat_dc`），否则读一个方向码字。
    fn setup_spatial_predictor(
        &mut self,
        r: &mut BitReader,
        plane: &[u8],
        off: usize,
        stride: usize,
        chroma: bool,
    ) -> Result<()> {
        let (range, sum) = Self::setup_spatial_compensation(
            plane,
            off,
            stride,
            &mut self.scratchpad,
            self.edges,
        );
        let quant = if chroma {
            self.orient = self.chroma_orient;
            self.quant_dc_chroma
        } else {
            self.quant
        };

        self.flat_dc = false;
        if range < quant as i32 || range < 3 {
            self.orient = 0;
            // 上游注释：±1 的 IDCT 误差都可能把解码带偏，所以这里必须照抄。
            if range < 3 {
                self.flat_dc = true;
                let sum = sum + 9;
                // ((1 << 17) + 9) / (8 + 8 + 1 + 2) = 6899
                self.predicted_dc = (sum * 6899) >> 17;
            }
        }
        if chroma {
            return Ok(());
        }

        if range < 2 * self.quant as i32 {
            if self.edges & 3 == 0 {
                // 两边都有邻居：把「上」/「左」换成对应的斜向方向。
                if self.orient == 1 {
                    self.orient = 11;
                }
                if self.orient == 2 {
                    self.orient = 10;
                }
            } else {
                self.orient = 0;
            }
            self.raw_orient = 0;
        } else {
            self.raw_orient = self.get_orient_vlc(r)?;
            if self.raw_orient >= 12 {
                return Err(Error::Bad("IntraX8: 方向码字越界"));
            }
            self.orient = t::X8_PREDICTION_TABLE[self.orient as usize][self.raw_orient as usize]
                as u32;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 把 `(code, len)` 写成位流（左侧为高位）。
    fn pack(code: u32, len: u32) -> Vec<u8> {
        let mut bits = vec![0u8; (len as usize).div_ceil(8) + 1];
        for i in 0..len {
            let b = ((code >> (len - 1 - i)) & 1) as u8;
            bits[(i / 8) as usize] |= b << (7 - i % 8);
        }
        bits
    }

    /// 每张码表**逐符号往返**：把符号的码字写进去、解出来必须还是它。
    ///
    /// 这条钉的是 IntraX8 特有的一个坑：源表是 `{符号, 码长}`，而**表内顺序 ≠ 符号顺序**
    /// （`x8_ac_quant_table[1][1][7]` 的第 71 项符号是 58）。生成器现在按符号排好再落盘，
    /// 于是「下标即符号」这条不变式在 Rust 侧成立 —— 顺序一旦漂了，这条测试立刻响。
    #[test]
    fn every_code_table_round_trips() {
        let sets: [(&[[u32; 2]], usize, usize); 3] = [
            (&t::X8_AC_QUANT, t::X8_AC_QUANT_STRIDE, 32),
            (&t::X8_DC_QUANT, t::X8_DC_QUANT_STRIDE, 16),
            (&t::X8_ORIENT_LOW, t::X8_ORIENT_LOW_STRIDE, 4),
        ];
        for (base, stride, n) in sets {
            for tbl in 0..n {
                let slice = table(base, stride, tbl);
                let v = Vlc::new(slice);
                for (sym, e) in slice.iter().enumerate() {
                    let (code, len) = (e[0], e[1]);
                    let bits = pack(code, len);
                    let mut r = BitReader::new(&bits);
                    assert_eq!(v.decode(&mut r).unwrap(), sym, "表 {tbl} 的符号 {sym}");
                    assert_eq!(r.bit_pos(), len as usize, "表 {tbl} 符号 {sym} 的码长");
                }
            }
        }
    }

    /// 每张表都必须**前缀无关**（生成器只查了 Kraft 和与符号排列，这里再查一遍树）。
    #[test]
    fn every_code_table_is_prefix_free() {
        for tbl in 0..32 {
            let base = table(&t::X8_AC_QUANT, t::X8_AC_QUANT_STRIDE, tbl);
            for (i, a) in base.iter().enumerate() {
                for (j, b) in base.iter().enumerate() {
                    assert!(
                        i == j || a[1] <= b[1] || (a[0] >> (a[1] - b[1])) != b[0],
                        "表 {tbl}：符号 {j} 是符号 {i} 的前缀"
                    );
                }
            }
        }
    }

    /// 快表（偷看 9 位那条捷径）与逐位走树必须一致。
    #[test]
    fn fast_table_agrees_with_the_table() {
        for tbl in 0..32 {
            let base = table(&t::X8_AC_QUANT, t::X8_AC_QUANT_STRIDE, tbl);
            let v = Vlc::new(base);
            for pat in 0u32..512 {
                let bits = pack(pat, 9);
                let mut r = BitReader::new(&bits);
                let got = v.decode(&mut r).unwrap();
                // 期望值：覆盖这个 9 位前缀的那一项（前缀无关 ⇒ 至多一项）。
                let want = base.iter().enumerate().find(|(_, e)| {
                    e[1] <= 9
                        && (e[0] << (9 - e[1])) <= pat
                        && pat < (e[0] << (9 - e[1])) + (1 << (9 - e[1]))
                });
                if let Some((w, _)) = want {
                    assert_eq!(got, w, "9 位模式 {pat:09b}");
                }
            }
        }
    }
}
