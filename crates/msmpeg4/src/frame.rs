//! 帧解码：MB 类型 → 系数 → 预测 → 运动补偿 → IDCT → YUV420P。
//!
//! 这是解码链的最后一段，把前面几个核（VLC / 系数 / 预测 / IDCT / MC）串起来。
//! 位流按 MS-MPEG-4 v3 的语法解：帧头 → 逐宏块（类型 / MV / 系数）→ 预测 → 运动补偿 → IDCT。
//! 行为与 ffmpeg 交叉验证：整段真样本逐帧、逐像素对账（见 `tests/pframe.rs`）。
//!
//! ## 帧间（P）与帧内（I）的差别（一条一条对着源码核过）
//!
//! | 项 | 帧内 | 帧间 |
//! |---|---|---|
//! | MB 类型 VLC | `ff_msmp4_mb_i_vlc` | `ff_mb_non_intra_vlc[3]`（= `ff_table_mb_non_intra`），`bit6` 是 `mb_intra` |
//! | CBP 预测 | 有（`ff_msmpeg4_coded_block_pred`） | **无**，直接 `code & 0x3f` |
//! | DC | 走 DC VLC + 预测，且 `block[0] *= dc_scale` | **没有 DC 这一趟**，DC 是普通 AC 系数 |
//! | RL 表 | `rl_table_index` / `3 + rl_chroma_table_index` | `3 + rl_table_index` |
//! | `run_diff`（第二 escape） | 0 | 1 |
//! | 扫描表 | zig-zag 或 ac_pred 的 H/V 交替 | 恒 zig-zag（`inter_scantable`） |
//! | 反量化 | **独立一趟**：`i = 1..=nCoeffs`，DC 另乘 `dc_scale` | **没有独立的一趟** —— RL 表里已经乘好了（见下） |
//! | 落像素 | `idct_put`（覆盖） | `idct_add`（叠加在 MC 预测上） |
//!
//! ## 帧间的反量化只有**一次**，而且是在 RL 表里
//!
//! RL 表按 qscale 分档：第 `q` 档里存的是 `table_level * qmul + qadd`
//! （`qmul = q*2`、`qadd = (q-1)|1`；第 0 档是恒等）。帧内取第 0 档，帧间取第 `qscale` 档
//! —— 所以**帧间解出来就是最终系数**。
//!
//! 也就是说这个格式里帧间残差**没有**单独的「反量化」步骤：解出来的系数直接进 IDCT 叠加。
//! 多跑一趟反量化会让残差整整放大 `qmul` 倍（qscale = 2 时是 4 倍）—— 实测踩过，
//! 当时帧 7 起全错，把我们的残差和参考相减才看出比值正好是 `qmul`。
//!
//! ## 参考帧要留**未裁剪**的那份
//!
//! MC 可能引用到「真实高度以外、但仍在宏块对齐区域内」的行
//! （`v_edge_pos = mb_height*16`，比真实高度最多多出 15 行），
//! 所以 `Decoder` 内部存的是宏块对齐、未裁剪的平面；对外吐的 `Frame` 才是裁过的。

use crate::bitreader::BitReader;
use crate::block::RlTable;
use crate::error::{Error, Result};
use crate::grid::Grid;
use crate::header::{parse_frame_header, PictureType};
use crate::idct::{idct_add, idct_put};
use crate::predict::{ac_predict, coded_block_predict, dc_predict, AcVal};
use crate::tables;
use crate::vlc::Vlc;

/// MPEG-4 的标准 zig-zag 扫描顺序（`ff_zigzag_direct`）。
/// 帧内（无 ac_pred）与**所有帧间块**都用它（`inter_scantable` 也是 zig-zag）。
pub const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
    20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58,
    59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// `DC_MAX`：DC VLC 里那个 escape 符号（解到它就用 8 位明文补 level）。
const DC_MAX: usize = 119;

/// MV VLC 的真值个数；
/// 解到这个下标说明是 escape，后面明文读 6+6 位。
const MV_ESCAPE: usize = 1099;

/// 解码出来的一帧（YUV420P，尺寸已裁到真实宽高）。
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// 宏块对齐、**未裁剪**的平面。运动补偿要在它上面做（会读到真实高度以外的行）。
///
/// ⚠️ 这是**共享**的：`crates/mp4v` 的 P 帧编码也用它（连同下面那几个 MC 内核）。
/// 从解码器里提出来公开，是因为半像素搬运那一套在 H.263 / MPEG-4 系里是同一份 ——
/// 复刻第二份才是真会跑偏的那份。
pub struct Planes {
    pub pw: usize,
    pub ph: usize,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl Planes {
    /// 宏块对齐、清零的平面组。
    pub fn new(pw: usize, ph: usize) -> Self {
        Self {
            pw,
            ph,
            y: vec![0u8; pw * ph],
            u: vec![0u8; (pw / 2) * (ph / 2)],
            v: vec![0u8; (pw / 2) * (ph / 2)],
        }
    }
}

/// 解码**通路覆盖**统计。
///
/// 用途不是性能，是**证明每条分支都真的被样本走到了** —— 「测试全绿」和「每条路都测过」
/// 是两回事：一条从没进去过的分支，在测试里和一条写对的分支长得一模一样。
/// 所以整段样本对账时把这份统计打出来，并断言关键通路都非零。
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub frames_i: u64,
    pub frames_p: u64,
    /// 帧内宏块总数（I 帧的全部 + P 帧里的那些）。
    pub mb_intra: u64,
    /// **P 帧里的**帧内宏块 —— 少见，但 DC 预测的「邻块是帧间就清成 1024」只在这条路上起作用。
    pub mb_intra_in_p: u64,
    /// 帧间且 CBP 非 0（有残差）。
    pub mb_inter_coded: u64,
    /// 帧间但 6 块全不编码（只有 MC 预测）。
    pub mb_inter_empty: u64,
    /// skip 宏块（`use_skip_mb_code` 那条路，MV 恒 (0,0)）。
    pub mb_skip: u64,
    /// 用了 AC 预测的帧内块。
    pub blocks_ac_pred: u64,
    pub mv_zero: u64,
    pub mv_half_x: u64,
    pub mv_half_y: u64,
    pub mv_half_xy: u64,
    /// MC 走到边缘复制（`ff_emulated_edge_mc`）的宏块数。
    pub mc_edge_emu: u64,
    /// 用到第 1 张 MV 表的帧数（第 0 张用 `frames_p - mv_table_1` 推）。
    pub frames_mv_table_1: u64,
    /// 帧间 RL 表下标（`3 + rl_table_index`）各用到多少帧。
    pub frames_rl_inter: [u64; 3],
    /// 帧内 RL 表下标（亮度）各用到多少帧。
    pub frames_rl_intra: [u64; 3],
}

/// 一次 `ac_pred` 标志的读取记录（只给测试/调试用）。
///
/// 存在的理由：ffmpeg 编出来的 MP43 流里 `ac_pred` **恒为 0**（它不带 AC 预测），
/// 所以手头的样本根本走不到 AC 预测那条路。要验证它，只能把真流里的这一位翻成 1，
/// 再让 ffmpeg 解同一条流来对账 —— 而翻位需要先知道这一位在哪。
#[derive(Debug, Clone, Copy)]
pub struct AcPredRead {
    pub mbx: usize,
    pub mby: usize,
    /// 该位在**本帧 `data`** 里的位偏移（帧头第 0 位起算）。
    pub bit_pos: usize,
    pub value: bool,
    /// 该宏块是帧内**且 CBP 非 0**（只有这种才真的会解 AC 系数）。
    pub coded: bool,
}

/// 有状态的解码器：P 帧要参考帧，半像素补偿的取整方式也要跨帧记住。
///
/// 这两条都不是可选的：
/// - 参考帧 —— P 帧的运动补偿。
/// - `no_rounding` —— 它由 `flipflop_rounding` 逐帧翻转，而 `flipflop_rounding`
///   只从 **I 帧尾部**的 ext header 里读。所以「只看这一帧」是解不出来的。
pub struct Decoder {
    /// 上一帧的内部（宏块对齐、未裁剪）平面。
    prev: Option<Planes>,
    /// `no_rounding`：半像素 MC 用不用带舍入的平均。
    no_rounding: bool,
    /// `flipflop_rounding`：只由 I 帧尾部的 ext header 更新。
    flipflop_rounding: bool,
    /// 已解帧数（只用于调试输出）。
    frames: usize,
    /// 通路覆盖统计。
    stats: Stats,
    /// 是否记录 `ac_pred` 位的位置（默认关）。
    trace_ac_pred: bool,
    /// 记录下来的 `ac_pred` 位。
    ac_pred_reads: Vec<AcPredRead>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            prev: None,
            no_rounding: false,
            flipflop_rounding: false,
            frames: 0,
            stats: Stats::default(),
            trace_ac_pred: false,
            ac_pred_reads: Vec::new(),
        }
    }

    /// 开/关 `ac_pred` 位的位置记录（下一次 `decode` 开始生效，每次 `decode` 会清空）。
    pub fn trace_ac_pred(&mut self, on: bool) {
        self.trace_ac_pred = on;
        self.ac_pred_reads.clear();
    }

    /// 最近一次 `decode` 里读到的 `ac_pred` 位。
    pub fn ac_pred_reads(&self) -> &[AcPredRead] {
        &self.ac_pred_reads
    }

    /// 迄今解过的所有帧的通路覆盖统计。
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// 解一帧（I 或 P）。P 帧必须已经有参考帧（先解过一帧）。
    pub fn decode(&mut self, data: &[u8], width: u32, height: u32) -> Result<Frame> {
        let mb_w = width.div_ceil(16) as usize;
        let mb_h = height.div_ceil(16) as usize;
        let (header, used_bits) = parse_frame_header(data, mb_w as u32, mb_h as u32)?;
        let is_p = header.picture_type == PictureType::P;
        let frame_no = self.frames;
        self.frames += 1;
        self.ac_pred_reads.clear();

        if is_p {
            self.stats.frames_p += 1;
            self.stats.frames_rl_inter[header.rl_table_index as usize] += 1;
            if header.mv_table_index == 1 {
                self.stats.frames_mv_table_1 += 1;
            }
        } else {
            self.stats.frames_i += 1;
            self.stats.frames_rl_intra[header.rl_table_index as usize] += 1;
        }

        // ---- `no_rounding`（帧头解析的收尾）----
        // I 帧恒置 1；P 帧看 `flipflop_rounding`：为真则翻转，否则恒 0。
        if is_p {
            self.no_rounding = if self.flipflop_rounding {
                !self.no_rounding
            } else {
                false
            };
            if self.prev.is_none() {
                return Err(Error::Bad("P 帧需要参考帧（先解一帧 I）"));
            }
        } else {
            self.no_rounding = true;
        }
        let no_rounding = self.no_rounding;

        // DC 缩放表：**MP43 的亮度用 `OLD_Y_DC_SCALE`，不是 `Y_DC_SCALE`** ——
        // 后者是 WMV1/WMV2 的表；拿它解 MP43 会让 qscale >= 5 的帧整体偏暗（实测踩过）。
        let q = header.qscale as usize;
        let y_dc_scale = tables::OLD_Y_DC_SCALE[q] as i32;
        let c_dc_scale = tables::C_DC_SCALE[q] as i32;
        let slice_height = header.slice_height.max(1) as usize;

        // 反量化参数（帧内 / 帧间同一套）：
        //   qmul = qscale << 1；qadd = (qscale - 1) | 1。
        // 帧内只作用在 `1..=nCoeffs`（DC 另乘 dc_scale），帧间是 `0..=nCoeffs`（含 DC）。
        let qmul = (q as i32) << 1;
        let qadd = ((q as i32) - 1) | 1;

        let dump_mb: Option<(usize, usize)> = std::env::var("V2V_DUMP_MB").ok().and_then(|s| {
            let (a, b) = s.split_once(',')?;
            Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
        });
        let debug_mb = std::env::var("V2V_DEBUG_MB").is_ok();
        let dump_mv = std::env::var("V2V_DUMP_MV").is_ok();

        // 系数表：帧内用 `rl_table_index`（亮度）/ `3 + rl_chroma_table_index`（色度）；
        // 帧间一律 `3 + rl_table_index`。
        let rl_luma = rl_table(header.rl_table_index as usize)?;
        let rl_chroma = rl_table(3 + header.rl_chroma_table_index as usize)?;
        let rl_inter = rl_table(3 + header.rl_table_index as usize)?;
        let mb_intra_vlc = Vlc::new(&tables::MB_INTRA);
        let mb_non_intra_vlc = Vlc::new(&tables::MB_NON_INTRA);
        let dc_vlc = Vlc::new(&tables::DC[header.dc_table_index as usize][0]);
        let dc_vlc_c = Vlc::new(&tables::DC[header.dc_table_index as usize][1]);

        // MV VLC：两张表各 1100 个码（1099 真值 + 1 escape）。
        let mv_i = header.mv_table_index as usize;
        let (mv_code, mv_len, mv_x, mv_y) = if mv_i == 0 {
            (
                &tables::MV0_CODE[..],
                &tables::MV0_LEN[..],
                &tables::MV0_X[..],
                &tables::MV0_Y[..],
            )
        } else {
            (
                &tables::MV1_CODE[..],
                &tables::MV1_LEN[..],
                &tables::MV1_X[..],
                &tables::MV1_Y[..],
            )
        };
        let mv_pairs: Vec<[u32; 2]> = mv_code
            .iter()
            .zip(mv_len.iter())
            .map(|(&c, &l)| [c as u32, l as u32])
            .collect();
        let mv_vlc = Vlc::new(&mv_pairs);

        // 平面（按宏块对齐分配，最后再裁）
        let pw = mb_w * 16;
        let ph = mb_h * 16;

        // 参考帧的尺寸必须和当前帧一致。流中途换分辨率是无效的，而且 MC 会按**当前帧**的
        // 尺寸算 `src_x/src_y` 去索引**参考帧**的缓冲 —— 尺寸不一致就是越界（曾经真的 panic 过）。
        if let Some(p) = &self.prev {
            if p.pw != pw || p.ph != ph {
                return Err(Error::Bad("参考帧尺寸与当前帧不一致（流中途换分辨率？）"));
            }
        }

        let mut out = Planes {
            pw,
            ph,
            y: vec![0u8; pw * ph],
            u: vec![0u8; (pw / 2) * (ph / 2)],
            v: vec![0u8; (pw / 2) * (ph / 2)],
        };

        // DC 预测的**边界初值是 1024** —— 不是 0。用 0 会让第一个块的预测值算成 0，
        // 而不是 1024/scale。
        //
        // 每帧重建是**等价**的：DC 预测器虽然可以跨帧保留，但每个宏块要么是帧内
        // （写自己的槽），要么是非帧内（把槽清回 1024），所以每帧结束时所有槽都落在
        // 「本帧写过」或「1024」两种状态之一。
        let mut dc_luma: Grid<i16> = Grid::with_init(2 * mb_w, 2 * mb_h, 1024);
        let mut dc_cb: Grid<i16> = Grid::with_init(mb_w, mb_h, 1024);
        let mut dc_cr: Grid<i16> = Grid::with_init(mb_w, mb_h, 1024);
        let mut ac_luma: Grid<AcVal> = Grid::new(2 * mb_w, 2 * mb_h);
        let mut ac_cb: Grid<AcVal> = Grid::new(mb_w, mb_h);
        let mut ac_cr: Grid<AcVal> = Grid::new(mb_w, mb_h);
        let mut coded: Grid<u8> = Grid::new(2 * mb_w, 2 * mb_h);
        // MV 预测网格：每个宏块一格，边框 0。邻居是 **A=左、B=上、C=右上** ——
        // 注意**不是**常见的「左 / 左上 / 上」；这个差别只有中值恰好卡在中间时才暴露，
        // 光靠读代码很容易漏（实测踩过）。
        let mut mv_grid: Grid<(i32, i32)> = Grid::new(mb_w, mb_h);

        let mut r = BitReader::new(data);
        r.skip(used_bits as u32)?;

        for mby in 0..mb_h {
            let first_slice_line = mby % slice_height == 0;
            for mbx in 0..mb_w {
                let mb_intra: bool;
                let cbp: u32;
                let mut ac_pred = false;
                let mut mv = (0i32, 0i32);

                if is_p {
                    // ---- 跳过宏块：MV 恒 (0,0)、六块全不编码 ----
                    if header.use_skip_mb_code && r.bit()? != 0 {
                        let g = mv_grid.idx(mbx, mby);
                        mv_grid.data[g] = (0, 0);
                        if dump_mv && mbx < 8 && mby < 4 {
                            eprintln!("帧{frame_no} P MB({mbx},{mby}) SKIP mv=(0,0) q={q} nr={no_rounding}");
                        }
                        let _ = mc_mb(
                            &mut out,
                            self.prev.as_ref().unwrap(),
                            mbx,
                            mby,
                            (0, 0),
                            no_rounding,
                        );
                        self.stats.mb_skip += 1;
                        self.stats.mv_zero += 1;
                        continue;
                    }
                    let code = mb_non_intra_vlc.decode(&mut r)? as u32;
                    // `s->mb_intra = (~code & 0x40) >> 6;`
                    mb_intra = (code & 0x40) == 0;
                    cbp = code & 0x3f;
                    if mb_intra {
                        self.stats.mb_intra += 1;
                        self.stats.mb_intra_in_p += 1;
                    } else if cbp == 0 {
                        self.stats.mb_inter_empty += 1;
                    } else {
                        self.stats.mb_inter_coded += 1;
                    }
                    if !mb_intra {
                        let pred = pred_motion(&mv_grid, mbx, mby, first_slice_line);
                        mv = decode_motion(&mut r, &mv_vlc, mv_x, mv_y, pred)?;
                        if dump_mv && mbx < 8 && mby < 4 {
                            let sx = (mbx as i32) * 16 + (mv.0 >> 1);
                            let sy = (mby as i32) * 16 + (mv.1 >> 1);
                            eprintln!(
                                "帧{frame_no} P MB({mbx},{mby}) pred={pred:?} mv={mv:?} src=({sx},{sy}) intra=0 skip=0 q={q} nr={no_rounding}"
                            );
                        }
                    } else {
                        let bit_pos = r.bit_pos();
                        ac_pred = r.bit()? != 0;
                        if self.trace_ac_pred {
                            self.ac_pred_reads.push(AcPredRead {
                                mbx,
                                mby,
                                bit_pos,
                                value: ac_pred,
                                coded: cbp != 0,
                            });
                        }
                    }
                } else {
                    // ---- MB 类型（帧内）：只给编码块图，还要做 CBP 预测 ----
                    mb_intra = true;
                    self.stats.mb_intra += 1;
                    let code = mb_intra_vlc.decode(&mut r)? as u32;
                    let mut cbp_v = 0u32;
                    for i in 0..6usize {
                        let mut val = (code >> (5 - i)) & 1;
                        if i < 4 {
                            let gx = 2 * mbx + (i & 1);
                            let gy = 2 * mby + (i >> 1);
                            let xy = coded.idx(gx, gy);
                            let a = coded.data[xy - 1];
                            let b = coded.data[xy - 1 - coded.w];
                            let c = coded.data[xy - coded.w];
                            let pred = coded_block_predict(a, b, c);
                            val ^= pred as u32;
                            coded.data[xy] = val as u8;
                        }
                        cbp_v |= val << (5 - i);
                    }
                    cbp = cbp_v;
                    let bit_pos = r.bit_pos();
                    ac_pred = r.bit()? != 0;
                    if self.trace_ac_pred {
                        self.ac_pred_reads.push(AcPredRead {
                            mbx,
                            mby,
                            bit_pos,
                            value: ac_pred,
                            coded: cbp != 0,
                        });
                    }
                }

                if debug_mb && mbx == 0 && mby == 0 {
                    eprintln!(
                        "MB(0,0) intra={mb_intra} cbp={cbp:#06b} ac_pred={ac_pred} mv={mv:?} bits={}",
                        r.bit_pos()
                    );
                }

                // ---- 帧间：先做运动补偿铺底，残差随后叠加 ----
                if !mb_intra {
                    let g = mv_grid.idx(mbx, mby);
                    mv_grid.data[g] = mv;
                    let (dxy, emu) = mc_mb(
                        &mut out,
                        self.prev.as_ref().unwrap(),
                        mbx,
                        mby,
                        mv,
                        no_rounding,
                    );
                    match dxy {
                        0 => self.stats.mv_zero += 1,
                        1 => self.stats.mv_half_x += 1,
                        2 => self.stats.mv_half_y += 1,
                        _ => self.stats.mv_half_xy += 1,
                    }
                    if emu {
                        self.stats.mc_edge_emu += 1;
                    }
                }

                // ---- 6 个块 ----
                for n in 0..6usize {
                    let coded_flag = (cbp >> (5 - n)) & 1 != 0;
                    let luma = n < 4;
                    let scale = if luma { y_dc_scale } else { c_dc_scale };

                    // 邻块坐标
                    let (dc_grid, ac_grid) = if luma {
                        (&mut dc_luma, &mut ac_luma)
                    } else if n == 4 {
                        (&mut dc_cb, &mut ac_cb)
                    } else {
                        (&mut dc_cr, &mut ac_cr)
                    };
                    let (gx, gy) = if luma {
                        (2 * mbx + (n & 1), 2 * mby + (n >> 1))
                    } else {
                        (mbx, mby)
                    };
                    let xy = dc_grid.idx(gx, gy);

                    let mut block = [0i16; 64];

                    if mb_intra {
                        // ---- DC ----
                        let dc_v = if luma { &dc_vlc } else { &dc_vlc_c };
                        let dc_sym = dc_v.decode(&mut r)?;
                        let mut level = dc_sym as i32;
                        if debug_mb && mbx == 0 && mby == 0 {
                            eprintln!("    blk{n} DC sym={dc_sym} esc={}", dc_sym == DC_MAX);
                        }
                        if level == DC_MAX as i32 {
                            level = r.u(8)? as i32;
                            if r.bit()? != 0 {
                                level = -level;
                            }
                        } else if level != 0 && r.bit()? != 0 {
                            level = -level;
                        }

                        // 预测（缩放到 level 域后比较）
                        let mut a = dc_grid.data[xy - 1] as i32;
                        let mut b = dc_grid.data[xy - 1 - dc_grid.w] as i32;
                        let mut c = dc_grid.data[xy - dc_grid.w] as i32;
                        // 片的第一行、且 n 的 bit1 为 0 → 上方不可用
                        if first_slice_line && (n & 2) == 0 {
                            b = 1024;
                            c = 1024;
                        }
                        a = (a + scale / 2) / scale;
                        b = (b + scale / 2) / scale;
                        c = (c + scale / 2) / scale;
                        let (pred, dir) = dc_predict(a, b, c);
                        level += pred;
                        dc_grid.data[xy] = (level * scale) as i16;

                        // 帧内的 DC 要乘 dc_scale（亮度用 `y_dc_scale`、色度用 `c_dc_scale`）——
                        // 漏了这一步会让整帧偏暗一档（输出成了 level/8 而不是 level）。
                        block[0] = (level * scale) as i16;
                        if debug_mb && mbx == 0 && mby == 0 {
                            eprintln!("  blk{n} coded={coded_flag} level={level} pred={pred} dir={dir} scale={scale} bits={}", r.bit_pos());
                        }
                        if coded_flag {
                            if ac_pred {
                                self.stats.blocks_ac_pred += 1;
                            }
                            // ac_pred 为真时**不用 zig-zag**，改用水平/垂直交替扫描
                            // （dc_pred_dir 0=取左 → 垂直扫描；1=取上 → 水平扫描）。
                            let scan: &[u8; 64] = if ac_pred {
                                if dir == 0 {
                                    &tables::V_SCAN
                                } else {
                                    &tables::H_SCAN
                                }
                            } else {
                                &ZIGZAG
                            };
                            let rl = if luma { &rl_luma } else { &rl_chroma };
                            for (i, lv) in rl.decode_ac(&mut r, 1, 0, 0, 0)? {
                                block[scan[i] as usize] = lv;
                            }
                        }

                        // ---- AC 预测 ----
                        let nb_xy = if dir == 0 { xy - 1 } else { xy - ac_grid.w };
                        let neighbor = ac_grid.data[nb_xy];
                        let mut ac = [0i16; 16];
                        ac_predict(&mut block, &mut ac, &neighbor, dir, ac_pred);
                        ac_grid.data[xy] = ac;

                        // ---- 帧内 AC 反量化（`i = 1..=nCoeffs`，不含 DC）----
                        // 位置在 AC 预测**之后**、IDCT **之前**。超出范围的系数本来就是 0，
                        // 而这一步对 0 是恒等，所以直接扫 1..63 等价。
                        for i in 1..64usize {
                            let v = block[i] as i32;
                            if v != 0 {
                                block[i] = if v < 0 {
                                    v * qmul - qadd
                                } else {
                                    v * qmul + qadd
                                } as i16;
                            }
                        }
                    } else {
                        // ---- 帧间块：DC 只是普通系数，没有预测、没有 ac_pred ----
                        if !coded_flag {
                            continue; // `block_last_index[n] = -1`：不加残差
                        }
                        for (i, lv) in rl_inter.decode_ac(&mut r, qmul, qadd, 1, -1)? {
                            block[ZIGZAG[i] as usize] = lv;
                        }
                        // ---- 帧间**不再**反量化 ----
                        // 帧间的反量化**只做一次**，而且在 RL 表里就做完了（表按 qscale 分档）。
                        // 所以 `decode_ac(qmul, qadd, …)` 出来的就是**最终系数**，直接进 idct_add。
                        // 再加一趟会让残差整整放大 qmul 倍（qscale = 2 时 4 倍）—— 实测踩过。
                        if dump_mb == Some((mbx, mby)) {
                            let mut line = String::new();
                            for j in 0..64 {
                                line.push_str(&format!("{:6}", block[j]));
                            }
                            eprintln!("帧{frame_no} MB({mbx},{mby}) blk{n} inter {line}");
                        }
                    }

                    // 调试：帧内块的**自然序**系数（对照 ffmpeg `-debug 64`）。
                    // ⚠️ ffmpeg 的打印发生在**反量化之前**，所以打印出来的 `block[0]` 是未乘
                    // `dc_scale` 的 level、AC 也是未反量化的原始 level —— 我们这里是**反量化之后**的，
                    // 对账时要先 `block[0] /= scale`、再把 AC 逆回 `(v ∓ qadd) / qmul`。
                    if mb_intra && dump_mb == Some((mbx, mby)) {
                        let mut line = String::new();
                        for j in 0..64 {
                            let v = if j == 0 {
                                block[0] / scale as i16
                            } else {
                                block[j]
                            };
                            line.push_str(&format!("{v:6}"));
                        }
                        eprintln!(
                            "帧{frame_no} MB({mbx},{mby}) blk{n} scan={} {line}",
                            if ac_pred { "H/V" } else { "zz" }
                        );
                    }

                    // ---- IDCT → 像素 ----
                    let (plane, stride, px, py) = if luma {
                        (
                            &mut out.y,
                            pw,
                            mbx * 16 + (n & 1) * 8,
                            mby * 16 + (n >> 1) * 8,
                        )
                    } else if n == 4 {
                        (&mut out.u, pw / 2, mbx * 8, mby * 8)
                    } else {
                        (&mut out.v, pw / 2, mbx * 8, mby * 8)
                    };
                    let at = py * stride + px;
                    if mb_intra {
                        idct_put(&mut block, &mut plane[at..], stride);
                    } else {
                        idct_add(&mut block, &mut plane[at..], stride);
                    }
                }
            }
        }

        // ---- I 帧尾部可能有 17 位的 ext header（fps 5 + bit_rate 11 + flipflop_rounding 1）----
        // 只在 I 帧做这件事，且只在剩余位数落在 `[17, 25)` 时读；
        // 否则 `flipflop_rounding = 0`。它决定**后面那些 P 帧**的半像素取整方式。
        if !is_p {
            let left = data.len() as i64 * 8 - r.bit_pos() as i64;
            self.flipflop_rounding = if (17..25).contains(&left) {
                r.skip(16)?; // fps(5) + bit_rate(11)
                r.bit()? != 0
            } else {
                false
            };
        }

        // 留作下一帧的参考（未裁剪的那份）
        self.prev = Some(Planes {
            pw,
            ph,
            y: std::mem::take(&mut out.y),
            u: std::mem::take(&mut out.u),
            v: std::mem::take(&mut out.v),
        });

        // 裁到真实尺寸
        let ref_planes = self.prev.as_ref().unwrap();
        let (w, h) = (width as usize, height as usize);
        Ok(Frame {
            width,
            height,
            y: crop(&ref_planes.y, pw, w, h),
            u: crop(&ref_planes.u, pw / 2, w / 2, h / 2),
            v: crop(&ref_planes.v, pw / 2, w / 2, h / 2),
        })
    }
}

fn crop(src: &[u8], sw: usize, cw: usize, ch: usize) -> Vec<u8> {
    let mut out = vec![0u8; cw * ch];
    for row in 0..ch {
        out[row * cw..row * cw + cw].copy_from_slice(&src[row * sw..row * sw + cw]);
    }
    out
}

/// 解一帧 **I 帧**（一次性，不带参考帧）。P 帧请用 `Decoder`。
pub fn decode_i_frame(data: &[u8], width: u32, height: u32) -> Result<Frame> {
    let (mb_w, mb_h) = (width.div_ceil(16) as u32, height.div_ceil(16) as u32);
    let (header, _) = parse_frame_header(data, mb_w, mb_h)?;
    if header.picture_type != PictureType::I {
        return Err(Error::Unsupported("decode_i_frame 只支持 I 帧".into()));
    }
    Decoder::new().decode(data, width, height)
}

/// H.263 的中值预测（`mid_pred`）。
pub fn mid_pred(a: i32, b: i32, c: i32) -> i32 {
    let mut b = b;
    if a > b {
        if c > b {
            if c > a {
                b = a;
            } else {
                b = c;
            }
        }
    } else if b > c {
        if c > a {
            b = c;
        } else {
            b = a;
        }
    }
    b
}

/// 运动矢量预测（`ff_h263_pred_motion(s, 0, 0, ...)`，`block = 0`）。
///
/// 邻居是**左(A) / 上(B) / 右上(C)**；片首行特殊：第一个宏块取 0，其余只取左邻
/// （`if (s->first_slice_line && block<3)` 那一支）。
pub(crate) fn pred_motion(
    g: &Grid<(i32, i32)>,
    mbx: usize,
    mby: usize,
    first_slice_line: bool,
) -> (i32, i32) {
    let xy = g.idx(mbx, mby);
    let a = g.data[xy - 1];
    if first_slice_line {
        if mbx == 0 {
            (0, 0)
        } else {
            a
        }
    } else {
        let b = g.data[xy - g.w];
        let c = g.data[xy - g.w + 1];
        (mid_pred(a.0, b.0, c.0), mid_pred(a.1, b.1, c.1))
    }
}

/// 运动矢量的差分解码（`ff_msmpeg4_decode_motion`）。
///
/// 码字给的是「相对预测值、偏 +32」的偏移；escape 时明文读 6+6 位。
/// 最后按这个格式的怪规矩回绕（**不是**真正的模 64）。
fn decode_motion(
    r: &mut BitReader,
    vlc: &Vlc,
    mv_x: &[u8],
    mv_y: &[u8],
    pred: (i32, i32),
) -> Result<(i32, i32)> {
    let code = vlc.decode(r)?;
    let (dx, dy) = if code == MV_ESCAPE {
        (r.u(6)? as i32, r.u(6)? as i32)
    } else {
        (mv_x[code] as i32, mv_y[code] as i32)
    };
    let mut mx = dx + pred.0 - 32;
    let mut my = dy + pred.1 - 32;
    if mx <= -64 {
        mx += 64;
    } else if mx >= 64 {
        mx -= 64;
    }
    if my <= -64 {
        my += 64;
    } else if my >= 64 {
        my -= 64;
    }
    Ok((mx, my))
}

/// 边缘复制：把 `bw × bh` 的区域填成
/// `src[clamp(src_y+y)][clamp(src_x+x)]`。
fn edge_emu(
    src: &[u8],
    sstride: usize,
    src_x: i32,
    src_y: i32,
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; bw * bh];
    for y in 0..bh {
        let sy = (src_y + y as i32).clamp(0, h as i32 - 1) as usize;
        for x in 0..bw {
            let sx = (src_x + x as i32).clamp(0, w as i32 - 1) as usize;
            out[y * bw + x] = src[sy * sstride + sx];
        }
    }
    out
}

/// 半像素双线性搬运（`put_pixels` 一族）。
///
/// `dxy` 的 bit0 = 水平半像素、bit1 = 垂直半像素。带舍入时：
/// x2/y2 用 `(a+b+1)>>1`，xy2 用 `(a+b+c+d+2)>>2`；不带舍入各减 1。
fn put_pixels(
    dst: &mut [u8],
    dstride: usize,
    dx: usize,
    dy: usize,
    src: &[u8],
    sstride: usize,
    sx: usize,
    sy: usize,
    w: usize,
    h: usize,
    dxy: usize,
    no_rounding: bool,
) {
    let hx = dxy & 1 != 0;
    let hy = dxy & 2 != 0;
    let (r1, r2) = if no_rounding {
        (0u32, 1u32)
    } else {
        (1u32, 2u32)
    };
    for y in 0..h {
        for x in 0..w {
            // ⚠️ 这里必须把 `y` / `x` 一起加进去 —— 只写相对偏移会让每一行/列都复制块首行。
            let p = |oy: usize, ox: usize| src[(sy + y + oy) * sstride + sx + x + ox] as u32;
            let v = match (hx, hy) {
                (false, false) => p(0, 0),
                (true, false) => (p(0, 0) + p(0, 1) + r1) >> 1,
                (false, true) => (p(0, 0) + p(1, 0) + r1) >> 1,
                (true, true) => (p(0, 0) + p(0, 1) + p(1, 0) + p(1, 1) + r2) >> 2,
            };
            dst[(dy + y) * dstride + dx + x] = v as u8;
        }
    }
}

/// 一次运动补偿的参数：半像素相位、参考坐标、是否需要边缘复制。
pub struct McParams {
    pub dxy: usize,
    pub uvdxy: usize,
    pub src_x: i32,
    pub src_y: i32,
    pub need_emu: bool,
}

/// 算一个宏块的 MC 参数（H.263 / MPEG-4 的半像素一路）。
pub fn mc_params(mbx: usize, mby: usize, mv: (i32, i32), pw: usize, ph: usize) -> McParams {
    let (mx, my) = mv;
    let dxy = (((my & 1) << 1) | (mx & 1)) as usize;
    let src_x = (mbx as i32) * 16 + (mx >> 1);
    let src_y = (mby as i32) * 16 + (my >> 1);

    let h_edge = pw as i32;
    let v_edge = ph as i32;
    // 越界判据：`(unsigned)` 强制转换让负数必然触发（等价于「任一坐标为负就触发」）。
    let limit_x = (h_edge - (mx & 1) - 15).max(0);
    let limit_y = (v_edge - (my & 1) - 16 + 1).max(0);
    let need_emu = (src_x as u32) >= (limit_x as u32) || (src_y as u32) >= (limit_y as u32);
    let uvdxy = dxy | ((my as usize) & 2) | (((mx as usize) & 2) >> 1);
    McParams {
        dxy,
        uvdxy,
        src_x,
        src_y,
        need_emu,
    }
}

/// 16×16 亮度块的半像素搬运（写到 `dest` 的 `(dx,dy)` 处）。返回是否走了边缘复制。
pub fn mc_luma16(
    dest: &mut [u8],
    dstride: usize,
    dx: usize,
    dy: usize,
    refp: &Planes,
    p: &McParams,
    no_rounding: bool,
) -> bool {
    // 裁剪范围用**参考帧自己的**尺寸：它才是真正被索引的缓冲。
    let ybuf = if p.need_emu {
        edge_emu(&refp.y, refp.pw, p.src_x, p.src_y, 17, 17, refp.pw, refp.ph)
    } else {
        Vec::new()
    };
    let (ysrc, ystride, yx, yy) = if p.need_emu {
        (&ybuf[..], 17usize, 0usize, 0usize)
    } else {
        (&refp.y[..], refp.pw, p.src_x as usize, p.src_y as usize)
    };
    put_pixels(
        dest,
        dstride,
        dx,
        dy,
        ysrc,
        ystride,
        yx,
        yy,
        16,
        16,
        p.dxy,
        no_rounding,
    );
    p.need_emu
}

/// 8×8 色度块的半像素搬运（`which`：0 = U，1 = V）。半像素相位由 `uvdxy` 表示。
pub fn mc_chroma8(
    dest: &mut [u8],
    dstride: usize,
    dx: usize,
    dy: usize,
    refp: &Planes,
    which: usize,
    p: &McParams,
    no_rounding: bool,
) {
    let cw = refp.pw / 2;
    let (ew, eh) = (refp.pw >> 1, refp.ph >> 1);
    let uvsrc_x = p.src_x >> 1;
    let uvsrc_y = p.src_y >> 1;
    let plane: &[u8] = if which == 0 { &refp.u } else { &refp.v };
    let buf = if p.need_emu {
        edge_emu(plane, cw, uvsrc_x, uvsrc_y, 9, 9, ew, eh)
    } else {
        Vec::new()
    };
    let (src, sstride, sx, sy) = if p.need_emu {
        (&buf[..], 9usize, 0usize, 0usize)
    } else {
        (plane, cw, uvsrc_x as usize, uvsrc_y as usize)
    };
    put_pixels(
        dest,
        dstride,
        dx,
        dy,
        src,
        sstride,
        sx,
        sy,
        8,
        8,
        p.uvdxy,
        no_rounding,
    );
}

/// 一个宏块的 16×16 运动补偿（`MV_TYPE_16X16` 一路）。返回 `(dxy, 是否走了边缘复制)`。
pub fn mc_mb(
    dest: &mut Planes,
    refp: &Planes,
    mbx: usize,
    mby: usize,
    mv: (i32, i32),
    no_rounding: bool,
) -> (usize, bool) {
    let p = mc_params(mbx, mby, mv, dest.pw, dest.ph);
    let emu = mc_luma16(
        &mut dest.y,
        dest.pw,
        mbx * 16,
        mby * 16,
        refp,
        &p,
        no_rounding,
    );
    mc_chroma8(
        &mut dest.u,
        dest.pw / 2,
        mbx * 8,
        mby * 8,
        refp,
        0,
        &p,
        no_rounding,
    );
    mc_chroma8(
        &mut dest.v,
        dest.pw / 2,
        mbx * 8,
        mby * 8,
        refp,
        1,
        &p,
        no_rounding,
    );
    (p.dxy, emu)
}

/// `ff_rl_table` 的 6 张表。0/1/2 = 帧内亮度（低/高/中运动），3/4/5 = 色度 / 帧间。
/// 其中 2 与 5 是 MPEG-4 的通用表，目前没生成 —— 遇到就明确报 Unsupported，不猜。
fn rl_table(index: usize) -> Result<RlTable> {
    Ok(match index {
        0 => RlTable::new(&tables::RL0_VLC, &tables::RL0_RUN, &tables::RL0_LEVEL, 85),
        1 => RlTable::new(&tables::RL2_VLC, &tables::RL2_RUN, &tables::RL2_LEVEL, 119),
        2 => RlTable::new(
            &tables::RL_INTRA_VLC,
            &tables::RL_INTRA_RUN,
            &tables::RL_INTRA_LEVEL,
            67,
        ),
        3 => RlTable::new(&tables::RL1_VLC, &tables::RL1_RUN, &tables::RL1_LEVEL, 81),
        4 => RlTable::new(&tables::RL4_VLC, &tables::RL4_RUN, &tables::RL4_LEVEL, 99),
        // 5 = 帧间通用表 `ff_h263_rl_inter`（n=102, last=58）。P 帧的 `rl_table_index=2`
        // 会落到这里（`3 + 2`）。
        5 => RlTable::new(
            &tables::RL_INTER_VLC,
            &tables::RL_INTER_RUN,
            &tables::RL_INTER_LEVEL,
            58,
        ),
        _ => return Err(Error::Unsupported(format!("RL 表下标 {index}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_starts_and_ends_right() {
        assert_eq!(ZIGZAG[0], 0);
        assert_eq!(ZIGZAG[1], 1);
        assert_eq!(ZIGZAG[8], 17);
        assert_eq!(ZIGZAG[63], 63);
        // 是一个排列
        let mut seen = [false; 64];
        for &z in &ZIGZAG {
            assert!(!seen[z as usize]);
            seen[z as usize] = true;
        }
    }

    #[test]
    fn mid_pred_is_the_median() {
        assert_eq!(mid_pred(1, 2, 3), 2);
        assert_eq!(mid_pred(3, 2, 1), 2);
        assert_eq!(mid_pred(0, 0, 5), 0);
        assert_eq!(mid_pred(-4, -4, -9), -4);
        assert_eq!(mid_pred(7, -2, 3), 3);
        for a in -3..=3 {
            for b in -3..=3 {
                for c in -3..=3 {
                    let mut v = [a, b, c];
                    v.sort_unstable();
                    assert_eq!(mid_pred(a, b, c), v[1], "({a},{b},{c})");
                }
            }
        }
    }

    #[test]
    fn rejects_a_p_frame_without_a_reference() {
        // pict_type = P(1) + 1 = 2 → 前两位写 01
        let mut w = crate::testbit::BitWriter::new();
        w.put(1, 2);
        w.put(8, 5);
        w.put(0, 1); // use_skip_mb_code
        w.put(0, 1); // rl
        w.put(0, 1); // dc
        w.put(0, 1); // mv
        let data = w.finish();
        assert!(matches!(
            decode_i_frame(&data, 16, 16),
            Err(Error::Unsupported(_))
        ));
        // 有状态入口遇到没有参考帧的 P 帧要明确报错，不能瞎解
        assert!(matches!(
            Decoder::new().decode(&data, 16, 16),
            Err(Error::Bad(_))
        ));
    }
}
