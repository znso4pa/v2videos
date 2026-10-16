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
//! 当时整段从某一帧起全错，把我们的残差和参考相减才看出比值正好是 `qmul`。
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
use crate::block::AcCtx;
use crate::header::{
    decode012 as v2v_msmpeg4_decode012, parse_frame_header, parse_wmv2_frame, HeaderState,
    PictureType, Wmv2Flags, Wmv2Frame,
};
use crate::version::Version;
use crate::idct::{idct48_add, idct84_add, idct_add, idct_put, wmv2_idct_add, wmv2_idct_put};
use crate::predict::{ac_predict, coded_block_predict, dc_predict, AcVal};
use crate::tables;
use crate::vlc::Vlc;

/// MPEG-4 的标准 zig-zag 扫描顺序（`ff_zigzag_direct`）。
/// 帧内（无 ac_pred）与**所有帧间块**都用它（`inter_scantable` 也是 zig-zag）。
pub const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
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
#[derive(Debug, Clone)]
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
    /// WMV2 开 mspel 的帧数。
    pub frames_mspel: u64,
    /// 逐块 ABT 类型：[0] = 普通 8×8、[1] = 4×8、[2] = 8×4。
    pub blocks_abt: [u64; 3],
    /// ABT 的 `sub_cbp`（`{2,3,1}` 那个表）各出现多少次，下标 = 表值。
    pub abt_sub_cbp: [u64; 4],
    /// 读到了 `hshift = 1` 的宏块数。
    pub mb_hshift_1: u64,
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
    /// 位流的**版本**（MP41/42/43 / WMV1/2）。它决定帧头、MB 层与系数解码的分支。
    version: Version,
    /// **WMV2 的能力位** —— 来自容器的 `extradata`，只有 `Version::Wmv2` 才有。
    /// 它决定帧头怎么读（`j_type` / `abt` / `mspel` / `per_mb_rl` 那几段读不读）。
    wmv2: Option<Wmv2Flags>,
    /// 帧头解析的跨帧状态（WMV1 的码率要用）。
    hdr_state: HeaderState,
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

/// **建一次就够的码表树**。
///
/// ⚠️ 存在的理由：`Vlc::new` / `RlTable::new` 原来在**每帧开头**都重建一遍，
/// 而它们的内容**只取决于静态码表**、与帧数据无关。采样显示 `Vlc::new` 一项
/// 就独占**整个解码的 19%** —— 纯粹的白干（MV 那张更亏：先每帧造一个
/// 1100 项的 `Vec`，再拿它建树）。
///
/// 树是只读的，所以放 `OnceLock` 里全局共享。
struct DecodeTables {
    /// I 帧的宏块类型表。
    mb_intra: Vlc,
    /// P 帧的宏块类型表。
    mb_non_intra: Vlc,
    /// `DC[dc_table_index][0=亮度 / 1=色度]`。
    dc: [[Vlc; 2]; 2],
    /// `MV0` / `MV1`。
    mv: [Vlc; 2],
    /// 六张 RL 表（`for_standard` 的 0..5）。
    rl: [RlTable; 6],
    /// **WMV2 的 P 帧**宏块类型表（按 qscale 与 `cbp_index` 选出来的 0..=2）。
    wmv2_cbp: [Vlc; 3],
    /// **V2 私有的**两张宏块类型表。
    ///
    /// ⚠️ 这两张与下面四张 H.263 的**已经建好但还没有人读** —— MP41/MP42 的帧头
    /// 按版本分叉了，MB 层还没写（见 README 的能力矩阵）。表先备着，读它们的代码
    /// 一落地就能直接用；在那之前 `allow(dead_code)` 是为了保住「编译零警告」。
    #[allow(dead_code)]
    v2_mb_type: Vlc,
    #[allow(dead_code)]
    v2_intra_cbpc: Vlc,
    /// **V1 用的 H.263** 那四张（MCBPC 两张 + CBPY + MV）。理由同上。
    #[allow(dead_code)]
    h263_intra_mcbpc: Vlc,
    #[allow(dead_code)]
    h263_inter_mcbpc: Vlc,
    #[allow(dead_code)]
    h263_cbpy: Vlc,
    #[allow(dead_code)]
    h263_mv: Vlc,
}

impl DecodeTables {
    fn get() -> &'static Self {
        static T: std::sync::OnceLock<DecodeTables> = std::sync::OnceLock::new();
        T.get_or_init(|| {
            let vlc = |t: &'static [[u32; 2]]| Vlc::new(t);
            // H.263 那几张在表里是 `[u8; 2]`，先并成 `[u32; 2]`。
            let vlc_u8 = |t: &'static [[u8; 2]]| {
                let pairs: Vec<[u32; 2]> = t.iter().map(|&[c, l]| [c as u32, l as u32]).collect();
                Vlc::new(&pairs)
            };
            // MCBPC 是「两张平行数组」：code[] 与 bits[]。
            let mcbpc = |code: &'static [u8], bits: &'static [u8]| {
                let pairs: Vec<[u32; 2]> = code
                    .iter()
                    .zip(bits.iter())
                    .map(|(&c, &l)| [c as u32, l as u32])
                    .collect();
                Vlc::new(&pairs)
            };
            // MV 的两张表在 `tables` 里是 `(code, len)` 分开存的，先并成 `[u32; 2]`。
            let mv = |code: &'static [u16], len: &'static [u8]| {
                let pairs: Vec<[u32; 2]> = code
                    .iter()
                    .zip(len.iter())
                    .map(|(&c, &l)| [c as u32, l as u32])
                    .collect();
                Vlc::new(&pairs)
            };
            DecodeTables {
                mb_intra: vlc(&tables::MB_INTRA),
                mb_non_intra: vlc(&tables::MB_NON_INTRA),
                dc: [
                    [vlc(&tables::DC[0][0]), vlc(&tables::DC[0][1])],
                    [vlc(&tables::DC[1][0]), vlc(&tables::DC[1][1])],
                ],
                mv: [
                    mv(&tables::MV0_CODE, &tables::MV0_LEN),
                    mv(&tables::MV1_CODE, &tables::MV1_LEN),
                ],
                wmv2_cbp: [
                    vlc(&tables::WMV2_MB_NON_INTRA0),
                    vlc(&tables::WMV2_MB_NON_INTRA1),
                    vlc(&tables::WMV2_MB_NON_INTRA2),
                ],
                v2_mb_type: vlc(&tables::V2_MB_TYPE),
                v2_intra_cbpc: vlc(&tables::V2_INTRA_CBCPC),
                h263_intra_mcbpc: mcbpc(
                    &tables::H263_INTRA_MCBPC_CODE,
                    &tables::H263_INTRA_MCBPC_BITS,
                ),
                h263_inter_mcbpc: mcbpc(
                    &tables::H263_INTER_MCBPC_CODE,
                    &tables::H263_INTER_MCBPC_BITS,
                ),
                h263_cbpy: vlc_u8(&tables::H263_CBPY),
                h263_mv: vlc_u8(&tables::H263_MV),
                rl: [
                    RlTable::for_standard(0).expect("RL 表 0"),
                    RlTable::for_standard(1).expect("RL 表 1"),
                    RlTable::for_standard(2).expect("RL 表 2"),
                    RlTable::for_standard(3).expect("RL 表 3"),
                    RlTable::for_standard(4).expect("RL 表 4"),
                    RlTable::for_standard(5).expect("RL 表 5"),
                ],
            }
        })
    }
}

impl Decoder {
    pub fn new() -> Self {
        Self::with_version(Version::default())
    }

    /// 指定**版本**的解码器。
    ///
    /// 版本是**流**的属性（来自容器里的 FourCC），不是帧的属性 ——
    /// 所以它在构造时定，之后每帧沿用。
    pub fn with_version(version: Version) -> Self {
        Self {
            version,
            wmv2: None,
            hdr_state: HeaderState::default(),
            prev: None,
            no_rounding: false,
            flipflop_rounding: false,
            frames: 0,
            stats: Stats::default(),
            trace_ac_pred: false,
            ac_pred_reads: Vec::new(),
        }
    }

    /// **WMV2** 的解码器 —— 还要给容器里那条视频轨的 `extradata`
    /// （能力位与 `slice_height` 都在它的前 4 字节里，**不是位流的一部分**）。
    pub fn with_wmv2(extra: &[u8]) -> Result<Self> {
        Ok(Self {
            wmv2: Some(Wmv2Flags::from_extradata(extra)?),
            ..Self::with_version(Version::Wmv2)
        })
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
        // WMV2 的帧头是**另一套语法**（1 位的 picture type、能力位来自容器的 extradata、
        // 还有「整帧 skip」的检测）—— 从另一个入口拿，但**后面的 MB 循环共用**：
        // 版本差异只有几处（skip 表、宏块类型表下标、运动预测、ABT），
        // 复刻一整个循环才是真会漂的那一份。
        let (header, used_bits, wmv2) = if self.version == Version::Wmv2 {
            let f = self.wmv2.ok_or_else(|| {
                Error::Bad("WMV2 解码器要由 `Decoder::with_wmv2` 建（能力位在 extradata 里）")
            })?;
            let (frame, used) =
                parse_wmv2_frame(data, mb_w as u32, mb_h as u32, &f, self.no_rounding)?;
            match frame {
                // 整帧都是 skip 宏块：不解，直接把参考帧当输出（ffmpeg 的 `FRAME_SKIPPED`）。
                Wmv2Frame::Skipped => {
                    let p = self
                        .prev
                        .as_ref()
                        .ok_or_else(|| Error::Bad("整帧 skip 但还没有参考帧"))?;
                    self.stats.frames_p += 1;
                    return Ok(Frame {
                        width,
                        height,
                        y: crop(&p.y, p.pw, width as usize, height as usize),
                        u: crop(&p.u, p.pw / 2, width as usize / 2, height as usize / 2),
                        v: crop(&p.v, p.pw / 2, width as usize / 2, height as usize / 2),
                    });
                }
                Wmv2Frame::Frame(h) => {
                    if h.j_type {
                        return Err(Error::Unsupported(
                            "WMV2 的 IntraX8（`j_type`）是另一套内帧编解码器，还没做".into(),
                        ));
                    }
                    // ⚠️ WMV2 的 `no_rounding` 是帧头算出来的（P 帧 `^= 1`），不走
                    // MS-MPEG-4 那条「I 帧尾部 ext header + flipflop」的路。
                    self.no_rounding = h.no_rounding;
                    let fh = h.to_frame_header();
                    (fh, used, Some(*h))
                }
            }
        } else {
            let (h, used) =
                parse_frame_header(data, mb_w as u32, mb_h as u32, self.version, &mut self.hdr_state)?;
            (h, used, None)
        };
        let is_p = header.picture_type == PictureType::P;

        // ---- 系数解码的版本相关上下文 ----
        //
        // ⚠️ WMV1/WMV2 的第三 escape 长度**在帧内粘住**（一帧里第一次遇到时才读），
        // 所以它必须活在**帧**这一层，不能放进每块的循环里。
        // ffmpeg 在帧头解析的末尾把它归零 —— 这里等价于每帧新建一个。
        let mut ac_ctx = AcCtx {
            version: self.version,
            qscale: header.qscale,
            esc3_level_length: 0,
            esc3_run_length: 0,
        };

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
        //
        // ⚠️ **WMV2 不在这里动** —— 它的 `no_rounding` 在帧头里就定了（P 帧 `^= 1`），
        // 上面那一支已经写进 `self.no_rounding`。这里再按 flipflop 算一遍会把它冲掉，
        // 而 WMV2 的 `flipflop_rounding` 恒为假 → 所有 P 帧都变成「不取整」。
        if is_p {
            if !self.version.is_wmv2() {
                self.no_rounding = if self.flipflop_rounding {
                    !self.no_rounding
                } else {
                    false
                };
            }
            if self.prev.is_none() {
                return Err(Error::Bad("P 帧需要参考帧（先解一帧 I）"));
            }
        } else {
            self.no_rounding = true;
        }
        let no_rounding = self.no_rounding;

        // ---- DC 缩放表：**按版本选**（ffmpeg 的 `ff_msmpeg4_common_init`）----
        //
        // | 版本        | 亮度                            | 色度                            |
        // |------------|--------------------------------|--------------------------------|
        // | V1 / V2     | 默认（MPEG-1 那套）                | 默认                            |
        // | V3          | `ff_old_ff_y_dc_scale_table`（见下） | `ff_wmv1_c_dc_scale_table`      |
        // | WMV1 / WMV2 | `ff_wmv1_y_dc_scale_table`      | `ff_wmv1_c_dc_scale_table`      |
        //
        // ⚠️ 亮度那两张**在 `qscale >= 5` 时不同**（`<= 4` 都是 8）：`OLD_Y` 在 q=5 是 10、
        // `Y`（WMV1 的）是 8。所以拿错表的表现是**「低 qscale 的帧碰巧全对、高 qscale 的帧
        // 整片错」** —— 而且只在有帧内宏块的帧上冒出来，排查时极易被当成别的问题。
        //
        // V3 用 `OLD_Y` 是因为 MP43 走 `workaround_bugs` 的 autodetect（实测对着真样本
        // 逐像素验证过）；WMV1/WMV2 则明确用 wmv1 那张。色度两边都是 wmv1 的。
        let q = header.qscale as usize;
        let y_dc_scale = if self.version.is_wmv() {
            tables::Y_DC_SCALE[q] as i32
        } else {
            tables::OLD_Y_DC_SCALE[q] as i32
        };
        let c_dc_scale = tables::C_DC_SCALE[q] as i32;
        let slice_height = header.slice_height.max(1) as usize;

        // 反量化参数（帧内 / 帧间同一套）：
        //   qmul = qscale << 1；qadd = (qscale - 1) | 1。
        // 帧内只作用在 `1..=nCoeffs`（DC 另乘 dc_scale），帧间是 `0..=nCoeffs`（含 DC）。
        let qmul = (q as i32) << 1;
        let qadd = ((q as i32) - 1) | 1;

        // 系数表：帧内用 `rl_table_index`（亮度）/ `3 + rl_chroma_table_index`（色度）；
        // 帧间一律 `3 + rl_table_index`。
        //
        // ⚠️ **六张全建**，而不是只建这一帧要用的那三张：WMV1 的 `per_mb_rl_table`
        // 允许**每个宏块换一套表**（下标在宏块里读），所以下标必须可变。
        // 建表的代价是每张 ~190 次循环，六张 ≈ 1000 次 —— 相对一帧的解码可忽略。
        let t = DecodeTables::get();
        let rl_tables = &t.rl;
        // 本帧当前的三个下标 —— 会被 `per_mb_rl_table` 逐宏块改写（就地延续，与 ffmpeg 同）。
        let mut rl_luma_i = header.rl_table_index as usize;
        let mut rl_chroma_i = 3 + header.rl_chroma_table_index as usize;
        let mut rl_inter_i = 3 + header.rl_table_index as usize;
        let mb_intra_vlc = &t.mb_intra;
        let mb_non_intra_vlc = &t.mb_non_intra;
        let dc_vlc = &t.dc[header.dc_table_index as usize][0];
        let dc_vlc_c = &t.dc[header.dc_table_index as usize][1];
        let mv_i = header.mv_table_index as usize;
        let mv_vlc = &t.mv[mv_i];
        let (mv_x, mv_y) = if mv_i == 0 {
            (&tables::MV0_X[..], &tables::MV0_Y[..])
        } else {
            (&tables::MV1_X[..], &tables::MV1_Y[..])
        };

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

        // WMV2 的**逐帧**状态（只有 Wmv2 有；其余版本这些全是常量）。
        // `abt_type` 与 ffmpeg 一样是**帧内粘住**的：帧头给初值，逐宏块/逐块再改。
        let wmv2_abt_cap = self.wmv2.is_some_and(|f| f.abt);
        let wmv2_top_left_mv = self.wmv2.is_some_and(|f| f.top_left_mv);
        let wmv2_per_mb_abt = wmv2.as_ref().is_some_and(|w| w.per_mb_abt);
        let wmv2_mspel = wmv2.as_ref().is_some_and(|w| w.mspel);
        if wmv2_mspel {
            self.stats.frames_mspel += 1;
        }
        let mut wmv2_abt_type = wmv2.as_ref().map_or(0u8, |w| w.abt_type);
        // 环路滤波：只有 WMV2 会开（`loop_filter` 在 extradata 里）。逐宏块跑，
        // 所以要一张 skip 表（I 帧没有 → 空表 = 全不 skip）。
        let do_loop_filter = self.wmv2.is_some_and(|f| f.loop_filter) && wmv2.is_some();
        let skip_grid: &[bool] = wmv2.as_ref().map_or(&[], |w| w.mb_skip.as_slice());

        for mby in 0..mb_h {
            let first_slice_line = mby % slice_height == 0;
            for mbx in 0..mb_w {
                let mb_intra: bool;
                let cbp: u32;
                let mut ac_pred = false;
                let mut mv = (0i32, 0i32);
                // WMV2 的 `hshift`（mspel 的取整选择）与逐块 ABT 开关 —— 都是**逐宏块**状态。
                let mut hshift = 0u8;
                let mut per_block_abt = false;

                if is_p {
                    // ---- 跳过宏块：MV 恒 (0,0)、六块全不编码 ----
                    // ⚠️ WMV2 的 skip 位**不在宏块循环里**：整帧的 skip 表在帧头就读完了，
                    // 这里只是查表（`header.use_skip_mb_code` 对它恒为假，见 `to_frame_header`）。
                    let skip = match &wmv2 {
                        Some(w) => w.mb_skip[mby * mb_w + mbx],
                        None => header.use_skip_mb_code && r.bit()? != 0,
                    };
                    if skip {
                        let g = mv_grid.idx(mbx, mby);
                        mv_grid.data[g] = (0, 0);
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
                        // ⚠️ **skip 的宏块也要走环路滤波**（ffmpeg 是在 `ff_mpv_reconstruct_mb`
                        // 之后无条件调的）。它自己那份 `qp_c` 会算成 0 —— 于是「本宏块内部
                        // 那几条边不滤」，但**和邻居之间的边照样滤**（用的是邻居的 qscale）。
                        // 少了这一趟的表现：只有紧挨着 skip 宏块的那几条边差 ±1。
                        if do_loop_filter {
                            loop_filter_mb(
                                &mut out,
                                mbx,
                                mby,
                                header.qscale,
                                skip_grid,
                                mb_w,
                                mb_h,
                            );
                        }
                        continue;
                    }
                    let code = match &wmv2 {
                        // WMV2 的宏块类型表**按 qscale 换**（`cbp_table_index` 在帧头算好）。
                        Some(w) => t.wmv2_cbp[w.cbp_table_index].decode(&mut r)? as u32,
                        None => mb_non_intra_vlc.decode(&mut r)? as u32,
                    };
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
                        // ⚠️ **顺序按版本不同**，而且两边都「读错位置整帧就散」：
                        // - WMV1：先读逐宏块 RL 表下标，再做运动预测；
                        // - WMV2：先做运动预测（它可能多读一位 `type`），再读 RL 下标与 ABT 开关。
                        if let Some(w) = &wmv2 {
                            let pred = wmv2_pred_motion(
                                &mv_grid,
                                mbx,
                                mby,
                                first_slice_line,
                                w.mspel,
                                wmv2_top_left_mv,
                                &mut r,
                            )?;
                            if cbp != 0 {
                                if header.per_mb_rl_table {
                                    let t = v2v_msmpeg4_decode012(&mut r)? as usize;
                                    rl_luma_i = t;
                                    rl_chroma_i = 3 + t;
                                    rl_inter_i = 3 + t;
                                }
                                if wmv2_abt_cap && wmv2_per_mb_abt {
                                    per_block_abt = r.bit()? != 0;
                                    if !per_block_abt {
                                        wmv2_abt_type = v2v_msmpeg4_decode012(&mut r)?;
                                    }
                                }
                            }
                            mv = decode_motion(&mut r, &mv_vlc, mv_x, mv_y, pred)?;
                            if w.mspel && ((mv.0 | mv.1) & 1) != 0 {
                                hshift = r.bit()? as u8;
                                self.stats.mb_hshift_1 += u64::from(hshift);
                            }
                        } else {
                            // WMV1 的 `per_mb_rl_table`：**每个带系数的宏块自带 RL 表下标**。
                            // 帧间这条在**运动之前**读（帧内那条在 `ac_pred` 之后）——
                            // 位置不一样，读错地方整帧就散了。
                            if header.per_mb_rl_table && cbp != 0 {
                                let t = v2v_msmpeg4_decode012(&mut r)? as usize;
                                rl_luma_i = t;
                                rl_chroma_i = 3 + t;
                                rl_inter_i = 3 + t;
                            }
                            let pred = pred_motion(&mv_grid, mbx, mby, first_slice_line);
                            mv = decode_motion(&mut r, &mv_vlc, mv_x, mv_y, pred)?;
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
                        // 帧内这条在 `ac_pred` **之后**读（见帧间那条的说明）。
                        if header.per_mb_rl_table && cbp != 0 {
                            let t = v2v_msmpeg4_decode012(&mut r)? as usize;
                            rl_luma_i = t;
                            rl_chroma_i = 3 + t;
                            rl_inter_i = 3 + t;
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
                    if header.per_mb_rl_table && cbp != 0 {
                        let t = v2v_msmpeg4_decode012(&mut r)? as usize;
                        rl_luma_i = t;
                        rl_chroma_i = 3 + t;
                        rl_inter_i = 3 + t;
                    }
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

                // ---- 帧间：先做运动补偿铺底，残差随后叠加 ----
                if !mb_intra {
                    let g = mv_grid.idx(mbx, mby);
                    mv_grid.data[g] = mv;
                    // ⚠️ mspel 一开，**整套**半像素换掉（四抽头 + 逐宏块 `hshift`），
                    // 连边缘复制的判据都换 —— 见 `mc_mb_mspel`。
                    let (dxy, emu) = if wmv2_mspel {
                        let emu = mc_mb_mspel(
                            &mut out,
                            self.prev.as_ref().unwrap(),
                            width as usize,
                            height as usize,
                            mbx,
                            mby,
                            mv,
                            hshift,
                            no_rounding,
                        );
                        ((((mv.1 & 1) << 1) | (mv.0 & 1)) as usize, emu)
                    } else {
                        mc_mb(
                            &mut out,
                            self.prev.as_ref().unwrap(),
                            mbx,
                            mby,
                            mv,
                            no_rounding,
                        )
                    };
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
                // WMV2 的 ABT：一个 8×8 块被拆成**两个半块**（4×8 或 8×4），
                // 第二半解到 `abt_block2` 里 —— 它按宏块清零（ffmpeg 用完就 clear_block，
                // 效果一样：没解到的那半恒为 0）。
                for n in 0..6usize {
                    let coded_flag = (cbp >> (5 - n)) & 1 != 0;
                    let luma = n < 4;
                    let scale = if luma { y_dc_scale } else { c_dc_scale };
                    // 本块走哪种变换：0 = 普通 8×8，1 = 4×8（上 4 行 + 下 4 行），2 = 8×4。
                    let mut block_abt = 0u8;

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
                    // ⚠️ ABT 的第二个半块**每个块各有一份**（ffmpeg 是 `abt_block2[6][64]`，
                    // 而且**用完就 clear**）。共用一个缓冲会让「只解了第一个半块」的那些块
                    // （`sub_cbp == 1`）把**上一个块的**残差当自己的用 —— 输出会莫名其妙地
                    // 「多出一份上块的图案」。
                    let mut abt_block2 = [0i16; 64];

                    if mb_intra {
                        // ---- DC ----
                        let dc_v = if luma { &dc_vlc } else { &dc_vlc_c };
                        let dc_sym = dc_v.decode(&mut r)?;
                        let mut level = dc_sym as i32;
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
                        // ⚠️ 版本门：只有 `version < WMV1`（V1/V2/V3）才在片首行清 b/c。
                        // WMV1/WMV2 不清 —— 它们用上一行的 DC，这是 ffmpeg 里那处
                        // `s->msmpeg4_version < MSMP4_WMV1` 条件。
                        if first_slice_line && (n & 2) == 0 && !self.version.is_wmv() {
                            b = 1024;
                            c = 1024;
                        }
                        a = dc_unscale(a, scale);
                        b = dc_unscale(b, scale);
                        c = dc_unscale(c, scale);
                        let (pred, dir) = dc_predict(a, b, c, self.version);
                        level += pred;
                        dc_grid.data[xy] = (level * scale) as i16;

                        // 帧内的 DC 要乘 dc_scale（亮度用 `y_dc_scale`、色度用 `c_dc_scale`）——
                        // 漏了这一步会让整帧偏暗一档（输出成了 level/8 而不是 level）。
                        block[0] = (level * scale) as i16;
                        if coded_flag {
                            if ac_pred {
                                self.stats.blocks_ac_pred += 1;
                            }
                            // ac_pred 为真时**不用 zig-zag**，改用水平/垂直交替扫描
                            // （dc_pred_dir 0=取左 → 垂直扫描；1=取上 → 水平扫描）。
                            //
                            // ⚠️ **整套扫描表按版本换**：WMV1/WMV2 用的是
                            // `ff_wmv1_scantable[0..3]`，**不是** MPEG-4/H.263 那套。
                            // 换错的表现很阴：符号个数、位流位置、结构全对，
                            // 只有**系数落错位置** —— 于是「没有漂移却值不对」。
                            let scan: &[u8; 64] = if self.version.is_wmv() {
                                // [0] 帧间 / [1] 帧内默认 / [2] ac_pred 水平 / [3] ac_pred 垂直
                                if ac_pred {
                                    if dir == 0 {
                                        &tables::SCAN[3]
                                    } else {
                                        &tables::SCAN[2]
                                    }
                                } else {
                                    &tables::SCAN[1]
                                }
                            } else if ac_pred {
                                if dir == 0 {
                                    &tables::V_SCAN
                                } else {
                                    &tables::H_SCAN
                                }
                            } else {
                                &ZIGZAG
                            };
                            let rl = if luma { &rl_tables[rl_luma_i] } else { &rl_tables[rl_chroma_i] };
                            rl.decode_ac(
                                &mut r,
                                1,
                                0,
                                // ⚠️ 帧内的 run_diff **也按版本**：
                                // WMV1/WMV2 是 1、V1/V2/V3 是 0。它只影响**第二 escape**，
                                // 所以底本里没有第二 escape 的帧（比如干净的 I 帧）
                                // 会「看着全对」，错误只在有第二 escape 的帧上冒出来。
                                self.version.intra_run_diff(),
                                0,
                                &mut block,
                                scan,
                                &mut ac_ctx,
                            )?;
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
                        if wmv2.is_some() {
                            // ABT：先定本块的变换类型（`per_block_abt` 时逐块读），
                            // 再按 `sub_cbp` 决定两个半块各解不解。
                            if wmv2_abt_cap && per_block_abt {
                                wmv2_abt_type = v2v_msmpeg4_decode012(&mut r)?;
                            }
                            block_abt = wmv2_abt_type;
                            self.stats.blocks_abt[block_abt as usize] += 1;
                            if block_abt != 0 {
                                let scan: &[u8; 64] = if block_abt == 1 {
                                    &tables::ABT_SCAN_A
                                } else {
                                    &tables::ABT_SCAN_B
                                };
                                // `sub_cbp_table[3] = {2, 3, 1}` —— 位序不是递增的，照抄。
                                let sub_cbp = [2u32, 3, 1][v2v_msmpeg4_decode012(&mut r)? as usize];
                                self.stats.abt_sub_cbp[sub_cbp as usize] += 1;
                                if sub_cbp & 1 != 0 {
                                    rl_tables[rl_inter_i].decode_ac(
                                        &mut r,
                                        qmul,
                                        qadd,
                                        self.version.inter_run_diff(),
                                        -1,
                                        &mut block,
                                        scan,
                                        &mut ac_ctx,
                                    )?;
                                }
                                if sub_cbp & 2 != 0 {
                                    rl_tables[rl_inter_i].decode_ac(
                                        &mut r,
                                        qmul,
                                        qadd,
                                        self.version.inter_run_diff(),
                                        -1,
                                        &mut abt_block2,
                                        scan,
                                        &mut ac_ctx,
                                    )?;
                                }
                            } else {
                                let scan: &[u8; 64] = &tables::SCAN[0];
                                rl_tables[rl_inter_i].decode_ac(
                                    &mut r,
                                    qmul,
                                    qadd,
                                    self.version.inter_run_diff(),
                                    -1,
                                    &mut block,
                                    scan,
                                    &mut ac_ctx,
                                )?;
                            }
                        } else {
                            let scan: &[u8; 64] =
                                if self.version.is_wmv() { &tables::SCAN[0] } else { &ZIGZAG };
                            rl_tables[rl_inter_i].decode_ac(
                                &mut r,
                                qmul,
                                qadd,
                                self.version.inter_run_diff(),
                                -1,
                                &mut block,
                                scan,
                                &mut ac_ctx,
                            )?;
                        }
                        // ---- 帧间**不再**反量化 ----
                        // 帧间的反量化**只做一次**，而且在 RL 表里就做完了（表按 qscale 分档）。
                        // 所以 `decode_ac(qmul, qadd, …)` 出来的就是**最终系数**，直接进 idct_add。
                        // 再加一趟会让残差整整放大 qmul 倍（qscale = 2 时 4 倍）—— 实测踩过。
                    }

                    // 调试：帧内块的**自然序**系数（对照 ffmpeg `-debug 64`）。
                    // ⚠️ ffmpeg 的打印发生在**反量化之前**，所以打印出来的 `block[0]` 是未乘
                    // `dc_scale` 的 level、AC 也是未反量化的原始 level —— 我们这里是**反量化之后**的，

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
                    // ⚠️ WMV2 的**每一块**都走它自己那份 8×8 IDCT（常数与舍入都不同），
                    // **但 ABT 的两个半块例外** —— 那半块用的是 `simple_idct` 的 4×8 / 8×4。
                    match (mb_intra, block_abt) {
                        (true, _) => {
                            if wmv2.is_some() {
                                wmv2_idct_put(&mut block, &mut plane[at..], stride);
                            } else {
                                idct_put(&mut block, &mut plane[at..], stride);
                            }
                        }
                        (false, 0) => {
                            if wmv2.is_some() {
                                wmv2_idct_add(&mut block, &mut plane[at..], stride);
                            } else {
                                idct_add(&mut block, &mut plane[at..], stride);
                            }
                        }
                        // 4×8：上半块占 4 行，下半块从 `+4*stride` 起。
                        (false, 1) => {
                            idct84_add(&mut block, &mut plane[at..], stride);
                            idct84_add(&mut abt_block2, &mut plane[at + 4 * stride..], stride);
                        }
                        // 8×4：左半块占 4 列，右半块从 `+4` 起。
                        _ => {
                            idct48_add(&mut block, &mut plane[at..], stride);
                            idct48_add(&mut abt_block2, &mut plane[at + 4..], stride);
                        }
                    }
                }

                // ---- 环路滤波：**逐宏块、就地**（读的邻居必须已经解好）----
                if do_loop_filter {
                    loop_filter_mb(&mut out, mbx, mby, header.qscale, skip_grid, mb_w, mb_h);
                }
            }
        }



        // ---- I 帧尾部可能有 17 位的 ext header（fps 5 + bit_rate 11 + flipflop_rounding 1）----
        // 只在 I 帧做这件事，且只在剩余位数落在 `[17, 25)` 时读；
        // 否则 `flipflop_rounding = 0`。它决定**后面那些 P 帧**的半像素取整方式。
        //
        // ⚠️ **只有 `version < WMV1` 才在帧尾读**（ffmpeg 的 `h263dec.c` 里那处
        // `msmpeg4_version < MSMP4_WMV1`）。WMV1 的那一份在**帧头里**已经读过了
        // （见 `header.rs` 的 `parse_frame_header`）—— 在这里再读一遍会把帧尾
        // 那几个字节当成 ext header，凭空翻转 `flipflop_rounding`，
        // 于是**后面所有 P 帧**的半像素取整方式全错。
        if !is_p {
            self.flipflop_rounding = if self.version.is_wmv() {
                header.flipflop_rounding
            } else {
                let left = data.len() as i64 * 8 - r.bit_pos() as i64;
                if (17..25).contains(&left) {
                    r.skip(16)?; // fps(5) + bit_rate(11)
                    r.bit()? != 0
                } else {
                    false
                }
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
    let (header, _) = parse_frame_header(data, mb_w, mb_h, Version::V3, &mut HeaderState::default())?;
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

/// DC 预测器里的那一步「除以 scale」（`ff_msmpeg4_pred_dc` 的后半段）。
///
/// ⚠️ 它**分两条路**，而且结果对负数不一样：
/// - `scale == 8`（低 qscale 那一档）：`(a + 4) / 8` —— C 的 `/`，**向零取整**；
/// - 其余：`FASTDIV(a + scale/2, scale)` —— 那是 `((uint64_t)a * ff_inverse[b]) >> 32`，
///   `a` 先**符号扩展**到 64 位，于是对负数等价于**向下取整**。
///
/// 差 1 的 DC 就是「整块偏 1」，而且只在负 DC 上冒出来 —— 低 qscale 的样本根本碰不到。
/// `ff_inverse[b] = ceil(2^32 / b)`（经典的「除以常数」魔数）。
#[inline]
fn fastdiv(v: i32, scale: i32) -> i32 {
    let inv = ((1u64 << 32) + scale as u64 - 1) / scale as u64;
    ((v as i64 as u64).wrapping_mul(inv) >> 32) as u32 as i32
}
#[inline]
pub(crate) fn dc_unscale(v: i32, scale: i32) -> i32 {
    if scale == 8 {
        // 常见档位走这条：C 的 `/`，**向零取整**。
        (v + 4) / 8
    } else {
        // 其余走 `FASTDIV(a + scale/2, scale)` —— 见上面的说明。
        fastdiv(v + (scale >> 1), scale)
    }
}

/// WMV2 的运动矢量预测（`wmv2_pred_motion`）—— **不是** H.263 那条。
///
/// 三处差别，每一处都只在特定条件下才显形：
/// - 左邻与上邻「差得够大」（`max(|Δx|,|Δy|) >= 8`）时，它**多读一位** `type` 来选左邻还是上邻；
/// - `type == 2`（默认）时，片首行取**左邻**而不是中值；
/// - `mspel` 为真时干脆不比 —— `diff` 恒 0，于是永远走中值那条。
///
/// ⚠️ 这个函数**会读位**，所以它的调用位置是语法的一部分（见 MB 循环里的顺序说明）。
fn wmv2_pred_motion(
    g: &Grid<(i32, i32)>,
    mbx: usize,
    mby: usize,
    first_slice_line: bool,
    mspel: bool,
    top_left_mv: bool,
    r: &mut BitReader,
) -> Result<(i32, i32)> {
    let xy = g.idx(mbx, mby);
    let a = g.data[xy - 1];
    let b = g.data[xy - g.w];
    let c = g.data[xy - g.w + 1];
    let diff = if mbx != 0 && !first_slice_line && !mspel && top_left_mv {
        (a.0 - b.0).abs().max((a.1 - b.1).abs())
    } else {
        0
    };
    let ty = if diff >= 8 { r.bit()? } else { 2 };
    Ok(match ty {
        0 => a,
        1 => b,
        _ => {
            if first_slice_line {
                a
            } else {
                (mid_pred(a.0, b.0, c.0), mid_pred(a.1, b.1, c.1))
            }
        }
    })
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
///
/// ⚠️ **不分配**：旧版每块一次 `vec![0u8; bw*bh]`（luma 一次 MC 就 289 字节），
/// 实测 `edge_emu` 一个人占整条链路的 **5.1%**（还带 malloc/free 的抖动）。
/// 现在由调用方给一块**栈上**的 scratch —— 同样的值，少一趟堆。
///
/// 逐像素的两次 `clamp` 也去掉了：x 方向的越界形状**整块共用**，所以先把
/// 「左侧贴边 / 中间连续 / 右侧贴边」三段算出来，每行只做一次切片拷贝 + 两次填充。
fn edge_emu_into(
    out: &mut [u8],
    src: &[u8],
    sstride: usize,
    src_x: i32,
    src_y: i32,
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) {
    debug_assert!(out.len() >= bw * bh);
    debug_assert!(w > 0 && h > 0);

    // x 方向三段（对每一行都一样）：
    //   [0, left)        贴左边界，取 row[0]
    //   [left, bw-right) 连续区间，从 row[src_x + left] 起整段拷
    //   [bw-right, bw)   贴右边界，取 row[w-1]
    let left = (-src_x).clamp(0, bw as i32) as usize;
    let right = (src_x + bw as i32 - w as i32).clamp(0, bw as i32) as usize;
    let right = right.min(bw - left);
    let mid = bw - left - right;

    for y in 0..bh {
        let sy = (src_y + y as i32).clamp(0, h as i32 - 1) as usize;
        let row = &src[sy * sstride..sy * sstride + w];
        let d = &mut out[y * bw..y * bw + bw];
        d[..left].fill(row[0]);
        if mid > 0 {
            let sx0 = (src_x + left as i32) as usize;
            debug_assert!(sx0 + mid <= w, "连续段越出参考帧");
            d[left..left + mid].copy_from_slice(&row[sx0..sx0 + mid]);
        }
        d[bw - right..].fill(row[w - 1]);
    }
}

/// 半像素双线性搬运（`put_pixels` 一族）。
///
/// `dxy` 的 bit0 = 水平半像素、bit1 = 垂直半像素。带舍入时：
/// x2/y2 用 `(a+b+1)>>1`，xy2 用 `(a+b+c+d+2)>>2`；不带舍入各减 1。
///
/// ⚠️ 四种相位**分开写**、按**行切片**取源：旧版把相位判断放在内层循环里、
/// 还用一个逐像素算地址的闭包（每像素 1~4 次带边界检查的索引）。
/// **算法一字未改**（相加、移位、舍入常数全照旧），所以输出逐字节不变。
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
    // 横向半像素要多读一格（`src_x + w`）；纵向半像素才碰下一行。
    let sw = w + usize::from(hx);
    for y in 0..h {
        let base = (sy + y) * sstride + sx;
        let top = &src[base..base + sw];
        let dbase = (dy + y) * dstride + dx;
        let d = &mut dst[dbase..dbase + w];
        match (hx, hy) {
            (false, false) => d.copy_from_slice(&top[..w]),
            (true, false) => {
                for x in 0..w {
                    d[x] = ((top[x] as u32 + top[x + 1] as u32 + r1) >> 1) as u8;
                }
            }
            (false, true) => {
                let bbase = (sy + y + 1) * sstride + sx;
                let bot = &src[bbase..bbase + sw];
                for x in 0..w {
                    d[x] = ((top[x] as u32 + bot[x] as u32 + r1) >> 1) as u8;
                }
            }
            (true, true) => {
                let bbase = (sy + y + 1) * sstride + sx;
                let bot = &src[bbase..bbase + sw];
                for x in 0..w {
                    d[x] = ((top[x] as u32
                        + top[x + 1] as u32
                        + bot[x] as u32
                        + bot[x + 1] as u32
                        + r2)
                        >> 2) as u8;
                }
            }
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
    // scratch 在**栈上**（17×17 = 289 字节）—— 旧版这里每块一次 `vec![]`。
    let mut ybuf = [0u8; 17 * 17];
    let (ysrc, ystride, yx, yy) = if p.need_emu {
        edge_emu_into(
            &mut ybuf,
            &refp.y,
            refp.pw,
            p.src_x,
            p.src_y,
            17,
            17,
            refp.pw,
            refp.ph,
        );
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

/// 半像素插值 + SAD **融合**：`cur`（16×16 的当前宏块）与 `mv` 处的参考块比。
///
/// ⚠️ 公式与 `put_pixels` **逐字相同**（同一套舍入常数），所以值与
/// 「先 `mc_luma16` 拷进缓冲、再逐像素求和」**逐位相同** —— 只是省掉一趟写 + 一趟读。
/// 这条是运动估计的内层：`v2v-mp4v` 与 `v2v-msmpeg4` 两个编码器都靠它
/// （同一份实现，不是各写一份）。
///
/// **越界（需要边缘复制）时不走融合那条路** —— 那种情况仍交给 `mc_luma16`
/// （它要按边界钳位取像素），`tmp` 就是给它用的 16×16 缓冲。
///
/// ⚠️ 切片长度必须跟着**相位**走：横向半像素（`hx`）才多读一格（`src_x + 16`），
/// 纵向半像素（`hy`）才碰下一行。`need_emu` 的边界正是按这个定的 ——
/// 无条件多切一格会在「最后一行 + 最后一列」越界 panic。
pub fn sad_luma16(
    cur: &[u8; 256],
    refp: &Planes,
    p: &McParams,
    no_rounding: bool,
    tmp: &mut [u8; 256],
) -> u64 {
    if p.need_emu {
        mc_luma16(tmp, 16, 0, 0, refp, p, no_rounding);
        let mut acc = 0u64;
        for (a, b) in cur.iter().zip(tmp.iter()) {
            acc += (*a as i32 - *b as i32).unsigned_abs() as u64;
        }
        return acc;
    }
    let (r1, r2) = if no_rounding {
        (0u32, 1u32)
    } else {
        (1u32, 2u32)
    };
    let pw = refp.pw;
    let base = p.src_y as usize * pw + p.src_x as usize;
    let (hx, hy) = (p.dxy & 1 != 0, p.dxy & 2 != 0);
    let mut acc = 0u64;
    // ⚠️ 四个相位**在行循环外**分开写。原来是把 `hx`/`hy` 的判断放在最内层，
    // 每次索引 `top[c]`/`top[c+1]`/`bot[c]`/`bot[c+1]` —— 编译器无法证明
    // `c + 1` 不越界，于是 256 次里每次都带边界检查；而且 `if hx` 在内层
    // 会挡住向量化。分开之后用 `zip`/`windows(2)` 表达「正好 16 个」，
    // 边界检查消失，内层变成可向量化的纯算术。
    // **值必须逐位相同**：公式与舍入常数一个字没动。
    match (hx, hy) {
        (false, false) => {
            for r in 0..16usize {
                let top = &refp.y[base + r * pw..base + r * pw + 16];
                let crow = &cur[r * 16..r * 16 + 16];
                for (a, b) in crow.iter().zip(top) {
                    acc += (*a as i32 - *b as i32).unsigned_abs() as u64;
                }
            }
        }
        (true, false) => {
            for r in 0..16usize {
                let top = &refp.y[base + r * pw..base + r * pw + 17];
                let crow = &cur[r * 16..r * 16 + 16];
                for (a, t) in crow.iter().zip(top.windows(2)) {
                    let v = (t[0] as u32 + t[1] as u32 + r1) >> 1;
                    acc += (*a as i32 - v as i32).unsigned_abs() as u64;
                }
            }
        }
        (false, true) => {
            for r in 0..16usize {
                let top = &refp.y[base + r * pw..base + r * pw + 16];
                let bot = &refp.y[base + (r + 1) * pw..base + (r + 1) * pw + 16];
                let crow = &cur[r * 16..r * 16 + 16];
                for (a, (t, b)) in crow.iter().zip(top.iter().zip(bot)) {
                    let v = (*t as u32 + *b as u32 + r1) >> 1;
                    acc += (*a as i32 - v as i32).unsigned_abs() as u64;
                }
            }
        }
        (true, true) => {
            for r in 0..16usize {
                let top = &refp.y[base + r * pw..base + r * pw + 17];
                let bot = &refp.y[base + (r + 1) * pw..base + (r + 1) * pw + 17];
                let crow = &cur[r * 16..r * 16 + 16];
                for (a, (t, b)) in crow.iter().zip(top.windows(2).zip(bot.windows(2))) {
                    let v = (t[0] as u32 + t[1] as u32 + b[0] as u32 + b[1] as u32 + r2) >> 2;
                    acc += (*a as i32 - v as i32).unsigned_abs() as u64;
                }
            }
        }
    }
    acc
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
    let mut buf = [0u8; 9 * 9];
    let (src, sstride, sx, sy) = if p.need_emu {
        edge_emu_into(&mut buf, plane, cw, uvsrc_x, uvsrc_y, 9, 9, ew, eh);
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

// ---------- WMV2 的 mspel 运动补偿 ----------
//
// ⚠️ WMV2 的 P 帧只要 `mspel` 为真，**整套**半像素换掉：不是 H.263 那种两抽头平均，
// 而是**四抽头** `(9(s₀+s₁) − (s₋₁+s₂) + 8) >> 4`；每个宏块还有一个 `hshift` 位
// 在两种取整之间选。`no_rounding` 在这条路上**只管色度**（亮度由 `hshift` 决定）。
//
// 三处判据也各不相同，且都不是「顺手统一」的地方：
// - 坐标裁剪用**编码宽高**（ffmpeg 的 `s->width/height`），不是宏块对齐的那对；
// - 边缘复制的触发用**对齐**宽高（`h_edge_pos = mb_width*16`），窗口 19×19、原点偏 (−1,−1)；
// - 色度不跟着这套走 —— 它还是普通半像素，但**复不复用边缘缓冲由亮度那边决定**。

/// `rnd_avg32` —— `(a + b + 1) >> 1`。
#[inline]
fn rnd_avg(a: u8, b: u8) -> u8 {
    ((a as u32 + b as u32 + 1) >> 1) as u8
}

/// `wmv2_mspel8_h_lowpass` —— 横向四抽头，输出 `h` 行 × 8 列（输出行距固定 8）。
fn mspel8_h(dst: &mut [u8], src: &[u8], sstride: usize, sx: usize, sy: usize, h: usize) {
    for y in 0..h {
        let base = (sy + y) * sstride + sx;
        // 从 `sx - 1` 起取，于是 `s[0]` 就是 C 里的 `src[-1]` —— 避免下标算出负数。
        let s = &src[base - 1..base + 10];
        let d = &mut dst[y * 8..y * 8 + 8];
        for x in 0..8 {
            let v = 9 * (s[x + 1] as i32 + s[x + 2] as i32) - (s[x] as i32 + s[x + 3] as i32) + 8;
            d[x] = (v >> 4).clamp(0, 255) as u8;
        }
    }
}

/// `wmv2_mspel8_v_lowpass` —— 纵向四抽头，输出 8 行 × `w` 列（行距 `dstride`）。
/// 源要能往上读一行、往下读两行。
fn mspel8_v(
    dst: &mut [u8],
    dstride: usize,
    src: &[u8],
    sstride: usize,
    sx: usize,
    sy: usize,
    w: usize,
) {
    for x in 0..w {
        for y in 0..8 {
            let p = (sy + y) * sstride + sx + x;
            let up = src[p - sstride] as i32;
            let mid = src[p] as i32;
            let down = src[p + sstride] as i32;
            let down2 = src[p + 2 * sstride] as i32;
            let v = 9 * (mid + down) - (up + down2) + 8;
            dst[y * dstride + x] = (v >> 4).clamp(0, 255) as u8;
        }
    }
}

/// 8×8 的 mspel 相位搬运 —— `dxy` 0..=7 就是 `put_mspel_pixels_tab` 的下标
/// （`2*((my&1)<<1 | (mx&1)) + hshift`）。
fn mspel_put_pixels(
    dst: &mut [u8],
    dstride: usize,
    dx: usize,
    dy: usize,
    src: &[u8],
    sstride: usize,
    sx: usize,
    sy: usize,
    dxy: usize,
) {
    let mut half_h = [0u8; 11 * 8];
    let mut half_v = [0u8; 64];
    let mut half_hv = [0u8; 64];
    match dxy {
        // 整数位置：直接拷。
        0 => {
            for y in 0..8 {
                let s = &src[(sy + y) * sstride + sx..][..8];
                let d = &mut dst[(dy + y) * dstride + dx..][..8];
                d.copy_from_slice(s);
            }
        }
        // mc10 / mc20 / mc30：只有横向半像素。
        1..=3 => {
            mspel8_h(&mut half_h, src, sstride, sx, sy, 8);
            for y in 0..8 {
                let row = &half_h[y * 8..y * 8 + 8];
                // mc30 是「半像素与右邻一格平均」，所以源整体右移一列。
                let base = (sy + y) * sstride + sx + usize::from(dxy == 3);
                let s = &src[base..base + 8];
                let d = &mut dst[(dy + y) * dstride + dx..][..8];
                for x in 0..8 {
                    d[x] = if dxy == 2 { row[x] } else { rnd_avg(s[x], row[x]) };
                }
            }
        }
        // mc02：只有纵向半像素。
        4 => {
            mspel8_v(&mut half_v, 8, src, sstride, sx, sy, 8);
            for y in 0..8 {
                let d = &mut dst[(dy + y) * dstride + dx..][..8];
                d.copy_from_slice(&half_v[y * 8..y * 8 + 8]);
            }
        }
        // mc12 / mc22 / mc32：横竖都有。
        _ => {
            mspel8_h(&mut half_h, src, sstride, sx, sy - 1, 11);
            if dxy == 6 {
                // mc22：横向的 11 行里从第 2 行起做纵向 —— 直接落位。
                mspel8_v(&mut dst[dy * dstride + dx..], dstride, &half_h, 8, 0, 1, 8);
            } else {
                // mc12 / mc32：纵向那份与「横向再纵向」那份取平均。
                mspel8_v(&mut half_v, 8, src, sstride, sx + usize::from(dxy == 7), sy, 8);
                mspel8_v(&mut half_hv, 8, &half_h, 8, 0, 1, 8);
                for y in 0..8 {
                    let d = &mut dst[(dy + y) * dstride + dx..][..8];
                    for x in 0..8 {
                        d[x] = rnd_avg(half_v[y * 8 + x], half_hv[y * 8 + x]);
                    }
                }
            }
        }
    }
}

/// WMV2 的 mspel 宏块运动补偿（`ff_mspel_motion`）。返回是否走了边缘复制。
#[allow(clippy::too_many_arguments)]
pub fn mc_mb_mspel(
    dest: &mut Planes,
    refp: &Planes,
    coded_w: usize,
    coded_h: usize,
    mbx: usize,
    mby: usize,
    mv: (i32, i32),
    hshift: u8,
    no_rounding: bool,
) -> bool {
    let (mx, my) = mv;
    let mut dxy = 2 * (((my & 1) << 1) | (mx & 1)) as usize + hshift as usize;
    let mut src_x = mbx as i32 * 16 + (mx >> 1);
    let mut src_y = mby as i32 * 16 + (my >> 1);
    src_x = src_x.clamp(-16, coded_w as i32);
    src_y = src_y.clamp(-16, coded_h as i32);
    if src_x <= -16 || src_x >= coded_w as i32 {
        dxy &= !3;
    }
    if src_y <= -16 || src_y >= coded_h as i32 {
        dxy &= !4;
    }

    let need_emu =
        src_x < 1 || src_y < 1 || src_x + 17 >= dest.pw as i32 || src_y + 17 >= dest.ph as i32;
    let mut ybuf = [0u8; 19 * 19];
    let (ysrc, ystride, yx, yy) = if need_emu {
        edge_emu_into(
            &mut ybuf,
            &refp.y,
            refp.pw,
            src_x - 1,
            src_y - 1,
            19,
            19,
            refp.pw,
            refp.ph,
        );
        (&ybuf[..], 19usize, 1usize, 1usize)
    } else {
        (&refp.y[..], refp.pw, src_x as usize, src_y as usize)
    };
    for i in 0..4usize {
        let (ox, oy) = ((i & 1) * 8, (i >> 1) * 8);
        mspel_put_pixels(
            &mut dest.y,
            dest.pw,
            mbx * 16 + ox,
            mby * 16 + oy,
            ysrc,
            ystride,
            yx + ox,
            yy + oy,
            dxy,
        );
    }

    // 色度：普通半像素，但坐标按**编码宽高的一半**裁（`s->width>>1`），
    // 且边缘缓冲**复用亮度那个决定**。
    let mut cdxy = 0usize;
    if (mx & 3) != 0 {
        cdxy |= 1;
    }
    if (my & 3) != 0 {
        cdxy |= 2;
    }
    let (cw, ch) = (coded_w as i32 / 2, coded_h as i32 / 2);
    let mut csx = mbx as i32 * 8 + (mx >> 2);
    let mut csy = mby as i32 * 8 + (my >> 2);
    csx = csx.clamp(-8, cw);
    csy = csy.clamp(-8, ch);
    if csx == cw {
        cdxy &= !1;
    }
    if csy == ch {
        cdxy &= !2;
    }
    for which in 0..2usize {
        let plane: &[u8] = if which == 0 { &refp.u } else { &refp.v };
        let mut cbuf = [0u8; 9 * 9];
        let (csrc, cstride, cx, cy) = if need_emu {
            edge_emu_into(
                &mut cbuf,
                plane,
                refp.pw / 2,
                csx,
                csy,
                9,
                9,
                refp.pw / 2,
                refp.ph / 2,
            );
            (&cbuf[..], 9usize, 0usize, 0usize)
        } else {
            (plane, refp.pw / 2, csx as usize, csy as usize)
        };
        let dst = if which == 0 { &mut dest.u } else { &mut dest.v };
        put_pixels(
            dst,
            dest.pw / 2,
            mbx * 8,
            mby * 8,
            csrc,
            cstride,
            cx,
            cy,
            8,
            8,
            cdxy,
            no_rounding,
        );
    }
    need_emu
}

// ---------- H.263 的环路滤波（`ff_h263_loop_filter`）----------
//
// ⚠️ 只有 **WMV2** 会开它：`s->loop_filter` 全 ffmpeg 里只有 `wmv2dec.c` 的
// `decode_ext_header` 会赋值（从 extradata 读一位）。MP43 / WMV1 那条路上没人设它，
// 所以它们的样本没有这一步 —— 这也是为什么「少了它」在 MP43 上完全看不出来。
//
// 它**逐宏块、就地**跑：解完一个宏块就滤它内部的边、以及它和**已解**邻居之间的边。
// 所以「整帧解完再滤一遍」**不等价** —— 顺序和邻居取值都变了。必须跟着宏块循环走。

/// `ff_h263_loop_filter_strength`。
const LOOP_FILTER_STRENGTH: [i32; 32] = [
    0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11,
    12, 12, 12,
];

/// 滤波内核的公共部分：给定 `p0 p1 | p2 p3`，返回 `(d1, d2)`。
#[inline]
fn loop_filter_delta(p0: i32, p1: i32, p2: i32, p3: i32, strength: i32) -> (i32, i32) {
    // ⚠️ C 的 `/` 是**向零取整**（负数不是向下取整）—— Rust 的 `/` 同语义 ✓。
    let d = (p0 - p3 + 4 * (p2 - p1)) / 8;
    let d1 = if d < -2 * strength {
        0
    } else if d < -strength {
        -2 * strength - d
    } else if d < strength {
        d
    } else if d < 2 * strength {
        2 * strength - d
    } else {
        0
    };
    let ad1 = d1.abs() >> 1;
    let d2 = ((p0 - p3) / 4).clamp(-ad1, ad1);
    (d1, d2)
}

/// C 里那个 `if (p & 256) p = ~(p >> 31);` 的**截到 0..255** 写法。
///
/// 位 8 被置上说明「要么 > 255、要么为负」；`p >> 31` 在负数是 -1（取反得 0）、
/// 非负是 0（取反得 -1，赋给 `uint8_t` 即 255）。
#[inline]
fn clip_bit8(p: i32) -> u8 {
    if p & 256 != 0 {
        !(p >> 31) as u8
    } else {
        p as u8
    }
}

/// `h263_h_loop_filter_c` —— 滤**竖边**：`at` 指边**右侧**那一列的起点，纵向 8 行。
fn h263_h_loop_filter(plane: &mut [u8], at: usize, stride: usize, qscale: u8) {
    let strength = LOOP_FILTER_STRENGTH[qscale as usize];
    for y in 0..8usize {
        let row = at + y * stride;
        let (p0, p1, p2, p3) = (
            plane[row - 2] as i32,
            plane[row - 1] as i32,
            plane[row] as i32,
            plane[row + 1] as i32,
        );
        let (d1, d2) = loop_filter_delta(p0, p1, p2, p3, strength);
        plane[row - 1] = clip_bit8(p1 + d1);
        plane[row] = clip_bit8(p2 - d1);
        plane[row - 2] = (p0 - d2) as u8;
        plane[row + 1] = (p3 + d2) as u8;
    }
}

/// `h263_v_loop_filter_c` —— 滤**横边**：`at` 指边**下方**那一行的起点，横向 8 列。
fn h263_v_loop_filter(plane: &mut [u8], at: usize, stride: usize, qscale: u8) {
    let strength = LOOP_FILTER_STRENGTH[qscale as usize];
    for x in 0..8usize {
        let p = at + x;
        let (p0, p1, p2, p3) = (
            plane[p - 2 * stride] as i32,
            plane[p - stride] as i32,
            plane[p] as i32,
            plane[p + stride] as i32,
        );
        let (d1, d2) = loop_filter_delta(p0, p1, p2, p3, strength);
        plane[p - stride] = clip_bit8(p1 + d1);
        plane[p] = clip_bit8(p2 - d1);
        plane[p - 2 * stride] = (p0 - d2) as u8;
        plane[p + stride] = (p3 + d2) as u8;
    }
}

/// 一个宏块的环路滤波（`ff_h263_loop_filter`）。
///
/// `skip` 是逐宏块的 skip 表（`mb_w * mb_h`，I 帧全 `false`）—— ffmpeg 用它决定
/// 那条边要不要滤、以及用谁的 qscale。所有非 skip 宏块的 `qscale_table` 都等于帧的
/// `qscale`（MS-MPEG-4 没有 dquant），所以这里直接用帧的 `qscale` 代入。
#[allow(clippy::too_many_arguments)]
fn loop_filter_mb(
    out: &mut Planes,
    mbx: usize,
    mby: usize,
    qscale: u8,
    skip: &[bool],
    mb_w: usize,
    mb_h: usize,
) {
    let linesize = out.pw;
    let uvlinesize = out.pw / 2;
    let is_skip = |x: usize, y: usize| skip.get(y * mb_w + x).copied().unwrap_or(false);

    let dy = mby * 16 * linesize + mbx * 16;
    let dc = mby * 8 * uvlinesize + mbx * 8;

    let qp_c = if !is_skip(mbx, mby) {
        qscale
    } else {
        0
    };
    // 宏块内部的两条横边（左半 / 右半）—— 只滤亮度。
    if qp_c != 0 {
        h263_v_loop_filter(&mut out.y, dy + 8 * linesize, linesize, qp_c);
        h263_v_loop_filter(&mut out.y, dy + 8 * linesize + 8, linesize, qp_c);
    }

    if mby != 0 {
        let qp_tt = if is_skip(mbx, mby - 1) { 0 } else { qscale };
        let qp_tc = if qp_c != 0 { qp_c } else { qp_tt };
        if qp_tc != 0 {
            // 与上邻之间的横边（亮度左右两半 + 色度）。
            h263_v_loop_filter(&mut out.y, dy, linesize, qp_tc);
            h263_v_loop_filter(&mut out.y, dy + 8, linesize, qp_tc);
            h263_v_loop_filter(&mut out.u, dc, uvlinesize, qp_tc);
            h263_v_loop_filter(&mut out.v, dc, uvlinesize, qp_tc);
        }
        if qp_tt != 0 {
            // ⚠️ 上邻**内部**那条竖边的下半段 —— 它在解上邻时已经滤过一次，
            // 这里**再滤一次**。ffmpeg 就是这么写的，照抄（滤波不是幂等的）。
            h263_h_loop_filter(&mut out.y, dy - 8 * linesize + 8, linesize, qp_tt);
        }
        if mbx != 0 {
            let qp_dt = if qp_tt != 0 || is_skip(mbx - 1, mby - 1) {
                qp_tt
            } else {
                qscale
            };
            if qp_dt != 0 {
                h263_h_loop_filter(&mut out.y, dy - 8 * linesize, linesize, qp_dt);
                h263_h_loop_filter(&mut out.u, dc - 8 * uvlinesize, uvlinesize, qp_dt);
                h263_h_loop_filter(&mut out.v, dc - 8 * uvlinesize, uvlinesize, qp_dt);
            }
        }
    }

    if qp_c != 0 {
        // 宏块内部那条竖边的上半段；下半段留到下面那张 MB 来滤
        // （它滤的时候用的还是**本宏块**的 qscale）—— 所以最后一行要自己补。
        h263_h_loop_filter(&mut out.y, dy + 8, linesize, qp_c);
        if mby + 1 == mb_h {
            h263_h_loop_filter(&mut out.y, dy + 8 * linesize + 8, linesize, qp_c);
        }
    }

    if mbx != 0 {
        let qp_lc = if qp_c != 0 || is_skip(mbx - 1, mby) {
            qp_c
        } else {
            qscale
        };
        if qp_lc != 0 {
            h263_h_loop_filter(&mut out.y, dy, linesize, qp_lc);
            if mby + 1 == mb_h {
                h263_h_loop_filter(&mut out.y, dy + 8 * linesize, linesize, qp_lc);
                h263_h_loop_filter(&mut out.u, dc, uvlinesize, qp_lc);
                h263_h_loop_filter(&mut out.v, dc, uvlinesize, qp_lc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DC 预测里那一步除法的**两条路**（`ff_msmpeg4_pred_dc`）。
    ///
    /// ⚠️ 它们在**负数**上不一样，而且只有 `scale != 8` 才分得出来：
    /// `scale == 8` 用 C 的 `/`（向零取整），其余用 `FASTDIV`（向下取整）。
    /// 拿错的表现是「整帧色度每块差 1」—— 只在低 qscale 之外冒出来。
    #[test]
    fn dc_unscale_floors_negatives_on_the_fastdiv_path() {
        // scale == 8：向零取整，两边一致。
        for v in [-20, -10, 5, 20, 100] {
            assert_eq!(dc_unscale(v, 8), (v + 4) / 8, "scale=8 时应当就是 (v+4)/8");
        }
        // scale == 9：负数**向下取整**（`(v + 4) / 9` 会差 1）。
        assert_eq!(dc_unscale(-20, 9), -2);
        assert_eq!(dc_unscale(-10, 9), -1);
        assert_eq!(dc_unscale(-6, 9), -1);
        // 非负两边一致。
        assert_eq!(dc_unscale(5, 9), 1);
        assert_eq!(dc_unscale(20, 9), 2);
        assert_eq!(dc_unscale(100, 9), 11);
    }

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

    /// 直接按定义写的边缘复制 —— oracle（逐像素 `clamp`，就是被换掉的那一版）。
    fn direct_edge_emu(
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

    #[test]
    fn edge_emu_matches_the_definition() {
        // 换实现（每块分配 + 逐像素 clamp → 栈 scratch + 三段切片）时，
        // 只有「对着定义逐点比」能保证输出没变。四种块长 × 各种越界位置都要扫到：
        // 全在框内、左/右越界、上/下越界、完全在外（贴同一行/列）。
        let (w, h) = (32usize, 24usize);
        let src: Vec<u8> = (0..w * h).map(|i| (i * 7 % 251) as u8).collect();
        for &(bw, bh) in &[(17usize, 17usize), (9, 9)] {
            for sx in -20..=(w as i32 + 4) {
                for sy in -20..=(h as i32 + 4) {
                    let want = direct_edge_emu(&src, w, sx, sy, bw, bh, w, h);
                    let mut got = vec![0u8; bw * bh];
                    edge_emu_into(&mut got, &src, w, sx, sy, bw, bh, w, h);
                    assert_eq!(
                        got, want,
                        "{bw}×{bh} 在 ({sx},{sy}) 处与定义不一致"
                    );
                }
            }
        }
    }

    /// 直接按定义写的半像素搬运 —— oracle（逐像素取源，就是被换掉的那一版）。
    fn direct_put_pixels(
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
        let (r1, r2) = if no_rounding { (0u32, 1u32) } else { (1u32, 2u32) };
        for y in 0..h {
            for x in 0..w {
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

    #[test]
    fn sad_luma16_matches_the_two_pass_path() {
        // 融合版（插值 + SAD 一趟走完）必须与「先 `mc_luma16` 拷进缓冲、再逐像素
        // 求和」**逐位相同** —— 判据是 `u64` 完全相等。扫四个相位 × 两种舍入 ×
        // 若干位置，含越界（走边缘复制那条路）与贴着右/下边界的位置。
        let (pw, ph) = (48usize, 48usize);
        let mut refp = Planes::new(pw, ph);
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u8
        };
        for v in refp.y.iter_mut() {
            *v = next();
        }
        let mut cur = [0u8; 256];
        for v in cur.iter_mut() {
            *v = next();
        }

        let mut tmp = [0u8; 256];
        for &no_rounding in &[false, true] {
            for mbx in [0usize, 1, 2] {
                for mby in [0usize, 1, 2] {
                    for mv in [
                        (-8i32, -8i32),
                        (0, 0),
                        (1, 0),
                        (0, 1),
                        (1, 1),
                        (3, 5),
                        (-3, 7),
                        (15, 15),
                    ] {
                        let p = mc_params(mbx, mby, mv, pw, ph);
                        let got = sad_luma16(&cur, &refp, &p, no_rounding, &mut tmp);
                        // 参照：两趟（先搬后比）—— 就是被融合掉的那一版。
                        let mut buf = [0u8; 256];
                        mc_luma16(&mut buf, 16, 0, 0, &refp, &p, no_rounding);
                        let mut want = 0u64;
                        for i in 0..256usize {
                            want += (cur[i] as i32 - buf[i] as i32).unsigned_abs() as u64;
                        }
                        assert_eq!(
                            got, want,
                            "mb({mbx},{mby}) mv{mv:?} no_rounding={no_rounding}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn put_pixels_matches_the_definition() {
        // 四个相位 × 两种舍入 × 若干偏移，逐字节比 —— 这条钉住「输出逐字节不变」。
        let (sw, sh) = (48usize, 48usize);
        let src: Vec<u8> = (0..sw * sh).map(|i| (i * 13 % 253) as u8).collect();
        let (w, h) = (16usize, 16usize);
        let (dw, dh) = (24usize, 24usize);
        for dxy in 0..4usize {
            for no_rounding in [false, true] {
                for &(sx, sy) in &[(0usize, 0usize), (1, 0), (0, 1), (1, 1), (8, 7), (31, 30)] {
                    for &(dx, dy) in &[(0usize, 0usize), (3, 5)] {
                        let mut want = vec![0u8; dw * dh];
                        direct_put_pixels(
                            &mut want, dw, dx, dy, &src, sw, sx, sy, w, h, dxy, no_rounding,
                        );
                        let mut got = vec![0u8; dw * dh];
                        put_pixels(
                            &mut got, dw, dx, dy, &src, sw, sx, sy, w, h, dxy, no_rounding,
                        );
                        assert_eq!(
                            got, want,
                            "dxy={dxy} no_rounding={no_rounding} 源({sx},{sy}) 目标({dx},{dy})"
                        );
                    }
                }
            }
        }
    }
}
