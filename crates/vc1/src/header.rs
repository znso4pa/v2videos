//! VC-1 的两层头：**序列头**（Simple / Main profile 的 `STRUCT_C`，装在容器的 extradata 里）
//! 与**帧头**。
//!
//! 与 Advanced profile 的分工：Advanced 的序列头/入口点头是**带起始码的码流内**结构
//! （`0x0000010F` / `0x0000010E`），Simple / Main 则**只在容器里** —— 帧数据里没有序列头，
//! 解封装拿到的 extradata 就是全部。所以这一层对容器有依赖（要 extradata + 编码尺寸），
//! 但依赖的是「一段字节」和「两个数字」，不是某个具体容器。
//!
//! 与上游的对照：`vc1.c` 的 `ff_vc1_decode_sequence_header()`（Simple/Main 分支）与
//! `ff_vc1_parse_frame_header()`。

use v2v_msmpeg4::{BitReader, Error, Result};

use crate::bitplane::{self, Bitplane};
use crate::tables;

/// Profile。数值即 `PROFILE_*`（见 `tables.rs`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Simple,
    Main,
    /// WMV9 的私有档，上游也只给了个警告就照 Main 解。
    Complex,
    Advanced,
}

impl Profile {
    fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            x if x == tables::PROFILE_SIMPLE as u32 => Self::Simple,
            x if x == tables::PROFILE_MAIN as u32 => Self::Main,
            x if x == tables::PROFILE_COMPLEX as u32 => Self::Complex,
            x if x == tables::PROFILE_ADVANCED as u32 => Self::Advanced,
            _ => return None,
        })
    }
}

/// 量化器模式。数值即 `QUANT_*`（见 `tables.rs`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizerMode {
    /// 帧级隐式：`pq` 由 `pqindex` 查表得出，且 `pqindex < 9` 时每帧读一个 `halfpq` 位。
    Implicit,
    /// 帧级显式：多读一个 `pquantizer` 位。
    Explicit,
    /// 全帧非均匀量化。
    NonUniform,
    /// 全帧均匀量化。
    Uniform,
}

impl QuantizerMode {
    fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            x if x == tables::QUANT_FRAME_IMPLICIT as u32 => Self::Implicit,
            x if x == tables::QUANT_FRAME_EXPLICIT as u32 => Self::Explicit,
            x if x == tables::QUANT_NON_UNIFORM as u32 => Self::NonUniform,
            x if x == tables::QUANT_UNIFORM as u32 => Self::Uniform,
            _ => return None,
        })
    }

    /// `pqindex` -> `pq` 用哪一行 `PQUANT`。
    fn pquant_row(self) -> usize {
        match self {
            Self::Implicit => 0,
            _ => 1,
        }
    }
}

/// 帧级 MV 模式（P 帧）。数值即 `MV_PMODE_*`（见 `tables.rs`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvMode {
    /// 1 个 MV，双线性半像素。
    OneMvHpelBilin,
    /// 1 个 MV，四分之一像素。
    OneMv,
    /// 1 个 MV，半像素。
    OneMvHpel,
    /// 逐宏块在 1MV / 4MV 之间选（要解一张位平面）。
    MixedMv,
    /// 亮度补偿：本帧的参考帧要先按 `lumscale` / `lumshift` 映射一遍。
    IntensityComp,
}

impl MvMode {
    fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            x if x == tables::MV_PMODE_1MV_HPEL_BILIN as u32 => Self::OneMvHpelBilin,
            x if x == tables::MV_PMODE_1MV as u32 => Self::OneMv,
            x if x == tables::MV_PMODE_1MV_HPEL as u32 => Self::OneMvHpel,
            x if x == tables::MV_PMODE_MIXED_MV as u32 => Self::MixedMv,
            x if x == tables::MV_PMODE_INTENSITY_COMP as u32 => Self::IntensityComp,
            _ => return None,
        })
    }

    /// 该模式是否用**四分之一像素** MV（决定宏块层走哪套插值）。
    fn quarter_sample(self) -> bool {
        !matches!(self, Self::OneMvHpel | Self::OneMvHpelBilin)
    }

    /// 该模式是否允许**半像素**（`mspel`）。
    fn mspel(self) -> bool {
        !matches!(self, Self::OneMvHpelBilin)
    }
}

/// 序列头：Simple / Main（以及 Complex）profile 的全部能力位。
///
/// **没有尺寸** —— Simple/Main 的 extradata 里不带宽高，尺寸来自容器。
/// 所以解析帧头时要另外把宏块网格尺寸传进来。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqHeader {
    pub profile: Profile,
    /// `(帧率 - 2) / 4`，只影响后处理。
    pub frmrtq_postproc: u8,
    /// `(码率 - 32kbps) / 64kbps`，只影响后处理。
    pub bitrtq_postproc: u8,
    /// 环路滤波总开关。
    pub loop_filter: bool,
    /// 多分辨率（逐帧改分辨率）。
    pub multires: bool,
    /// 保留位（上游注释 `reserved`）—— 但它**是** IntraX8 的闸门：
    /// 置位时 I 帧的帧头里多一个 `x8_type` 位。
    pub res_x8: bool,
    /// 保留位，语义未公开（上游注释 `reserved`）。
    pub res_fasttx: bool,
    /// 色度用「快速」MC（只在色度上少做一次插值）。
    pub fastuvmc: bool,
    /// 允许扩展 MV 范围（帧头里多读一个 `mvrange`）。
    pub extended_mv: bool,
    /// 宏块级量化变化的粒度：`0` = 不变，`1` = 帧级，`2` = 宏块级。
    pub dquant: u8,
    /// 允许 ABT（4×4 / 4×8 / 8×4 变换）。
    pub vstransform: bool,
    /// 重叠平滑（overlap smoothing）。
    pub overlap: bool,
    /// 码流里有重同步标记。
    pub resync_marker: bool,
    /// 亮度/色度取值范围缩减（range reduction）。
    pub rangered: bool,
    /// 允许的最大 B 帧数。
    pub max_b_frames: u8,
    pub quantizer_mode: QuantizerMode,
    /// 帧级插值标志（`finterpflag`）：置位时每帧多读一个 `interpfrm` 位。
    pub finterpflag: bool,
    /// 保留位（上游注释 `reserved`）。
    pub res_rtm_flag: bool,
}

impl SeqHeader {
    /// 解一段 Simple / Main profile 的 `STRUCT_C`（通常 4~5 字节，来自容器 extradata）。
    ///
    /// 恰好用掉**前 32 位**；尾部的填充位不读也不校验（上游也只用 `show_bits_long` 记了个日志）。
    pub fn parse(extra: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(extra);
        let profile =
            Profile::from_u32(r.u(2)?).ok_or(Error::Bad("vc1: 非法的 profile"))?;
        if profile == Profile::Advanced {
            return Err(Error::Unsupported(
                "vc1: Advanced profile（序列头在码流里，不是 extradata）".into(),
            ));
        }
        // Simple / Main / Complex 走同一段语法。
        let res_y411 = r.bit()? != 0;
        let res_sprite = r.bit()? != 0;
        if res_y411 {
            return Err(Error::Bad("vc1: RES_Y411（老式隔行）不支持"));
        }
        if res_sprite {
            return Err(Error::Unsupported("vc1: sprite 帧不支持".into()));
        }

        let frmrtq_postproc = r.u(3)? as u8;
        let bitrtq_postproc = r.u(5)? as u8;
        let loop_filter = r.bit()? != 0;
        if loop_filter && profile == Profile::Simple {
            return Err(Error::Bad("vc1: Simple profile 不允许开环路滤波"));
        }
        let res_x8 = r.bit()? != 0;
        let multires = r.bit()? != 0;
        let res_fasttx = r.bit()? != 0;
        let fastuvmc = r.bit()? != 0;
        if profile == Profile::Simple && !fastuvmc {
            return Err(Error::Bad("vc1: Simple profile 必须开 FASTUVMC"));
        }
        let extended_mv = r.bit()? != 0;
        if profile == Profile::Simple && extended_mv {
            return Err(Error::Bad("vc1: Simple profile 不允许扩展 MV"));
        }
        let dquant = r.u(2)? as u8;
        let vstransform = r.bit()? != 0;
        if r.bit()? != 0 {
            return Err(Error::Bad("vc1: RES_TRANSTAB 必须为 0"));
        }
        let overlap = r.bit()? != 0;
        let resync_marker = r.bit()? != 0;
        let rangered = r.bit()? != 0;
        let max_b_frames = r.u(3)? as u8;
        let quantizer_mode =
            QuantizerMode::from_u32(r.u(2)?).ok_or(Error::Bad("vc1: 非法的 quantizer_mode"))?;
        let finterpflag = r.bit()? != 0;
        let res_rtm_flag = r.bit()? != 0;

        Ok(Self {
            profile,
            frmrtq_postproc,
            bitrtq_postproc,
            loop_filter,
            multires,
            res_x8,
            res_fasttx,
            fastuvmc,
            extended_mv,
            dquant,
            vstransform,
            overlap,
            resync_marker,
            rangered,
            max_b_frames,
            quantizer_mode,
            finterpflag,
            res_rtm_flag,
        })
    }

    /// 宏块级量化会变吗（`dquant == 2` 时宏块层要逐块读量化参数）。
    pub fn dquant_per_mb(&self) -> bool {
        self.dquant == 2
    }
}

/// 图像类型。B / BI 本实现还没做（宏块层不支持双向预测），解析到就报错。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictType {
    I,
    P,
}

/// 帧头解析要**跨帧**记住的东西。
///
/// 两样都不是「上一帧的某个字段」那么简单：
/// - `rnd`：位流里**没有**这个位。I 帧把它置 1，P 帧只是**翻转** —— 所以 P 帧的取整
///   方向是交替的（P、P、P… 依次 0/1 相间）。初值是 0（新建上下文）。
/// - `quarter_sample`：本帧算出来的，但**下一帧**做混合 MV 时要用它当 `qs_last`。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeaderState {
    pub rnd: bool,
    pub quarter_sample: bool,
}

/// P 帧独有的帧头字段。
#[derive(Debug, Clone)]
pub struct PFrameHeader {
    pub mv_mode: MvMode,
    /// 亮度补偿时才有：补偿模式下**实际**用哪套 MV 模式。
    pub mv_mode2: Option<MvMode>,
    /// 亮度补偿的缩放（6 位，`> 32` 表示放大）。
    pub lumscale: u8,
    /// 亮度补偿的偏移（6 位，有符号语义：`> 32` 为负）。
    pub lumshift: u8,
    pub quarter_sample: bool,
    /// 上一帧的 `quarter_sample`（混合 MV 的宏块层要用它）。
    pub qs_last: bool,
    pub mspel: bool,
    /// 宏块层的 MV 码表选择（2 位）。
    pub mv_table_index: u8,
    /// 宏块层的 CBP 码表选择（2 位）。
    pub cbptab: u8,
    /// 宏块级变换标志：`false` = 整帧都用 `tt`。
    pub ttmbf: bool,
    /// 本帧的变换类型（`TT_*`）。`ttmbf == false` 时恒为 `TT_8X8`。
    pub tt: u8,
    /// 逐宏块的「1MV 还是 4MV」位平面。`None` = 整帧都是 1MV（位流里没这张图）。
    pub mv_type_plane: Option<Bitplane>,
    /// 逐宏块的跳过标志位平面。
    pub skip_plane: Bitplane,
}

/// 帧头。
#[derive(Debug, Clone)]
pub struct FrameHeader {
    pub pict_type: PictType,
    /// 取整控制（见 `HeaderState::rnd`）。
    pub rnd: bool,
    pub pqindex: u8,
    /// 量化步长（`pqindex` 查表）。
    pub pq: u8,
    /// `pqindex < 9` 时的半步长标志。
    pub halfpq: bool,
    /// 宏块层用「带取整」的反量化。
    pub pquantizer: bool,
    /// 扩展 MV 范围（`extended_mv` 时才有）。
    pub mvrange: u8,
    /// MV 分量的合法范围：`[-range_x, range_x)`（`k_x` 是它的位数）。
    pub range_x: i32,
    pub range_y: i32,
    /// 多分辨率档位（`multires` 时才有）。
    pub respic: u8,
    /// 本帧是否做了取值范围缩减（`rangered` 时才有）。
    pub rangeredfrm: bool,
    /// 帧级量化变化。
    pub dquant: Option<Dquant>,
    /// `res_x8` 且 I 帧时才有：`true` 表示这一帧走 IntraX8 的语法。
    pub x8_type: bool,
    /// 宏块层的 AC 码表选择（`decode012`，0~2）。
    pub c_ac_table_index: u8,
    /// 亮度 AC 码表选择 —— **只有 I 帧有**。
    pub y_ac_table_index: Option<u8>,
    /// 宏块层的 DC 码表选择（1 位）。
    pub dc_table_index: u8,
    /// P 帧独有；I 帧为 `None`。
    pub p: Option<PFrameHeader>,
}

/// 帧级量化变化（`vop_dquant_decoding()`）。
///
/// 「有这个结构」= 位流里的 `dquantfrm` 为 1（`dquant == 2` 时这个位根本不读，
/// 变化信息在宏块层里，此时恒为 `None`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dquant {
    /// 变化范围（`DQPROFILE_*`：0 = 四条边，1 = 两条边，2 = 一条边，3 = 全部宏块）。
    pub dqprofile: u8,
    /// 变化发生在哪条边（`dqprofile` 为 1 / 2 时才有）。
    pub dqsbedge: u8,
    /// 全部宏块时：量化是二值的（只有 `pq` 与 `altpq` 两档）。
    pub dqbilevel: bool,
    /// 另一个量化步长。`None` = 位流里没给（每个宏块自带，此时半步长被强制关掉）。
    pub altpq: Option<u8>,
}

/// `get_unary(stop, max)`：逐位读，读到 `stop` 停，返回读过的位数（最多 `max` 位）。
fn unary(r: &mut BitReader, stop: u32, max: u32) -> Result<u32> {
    let mut i = 0;
    while i < max {
        if r.bit()? == stop {
            break;
        }
        i += 1;
    }
    Ok(i)
}

/// `decode012()`：一位 0 -> 0，否则再一位 -> 1 或 2。
fn decode012(r: &mut BitReader) -> Result<u8> {
    if r.bit()? == 0 {
        Ok(0)
    } else {
        Ok(r.bit()? as u8 + 1)
    }
}

/// 帧级量化变化（对应上游 `vop_dquant_decoding()`）。返回 `None` = 本帧不带。
///
/// **只对 P（和 B）帧调用** —— I 帧的帧头里没有这一段（上游也是这样）。
fn vop_dquant(
    r: &mut BitReader,
    seq: &SeqHeader,
    pq: u8,
    halfpq: &mut bool,
) -> Result<Option<Dquant>> {
    // `dquant == 2` 表示变化信息在**宏块层**里，帧头一位都不占。
    if seq.dquant == 2 {
        return Ok(None);
    }
    if r.bit()? == 0 {
        return Ok(None);
    }
    let dqprofile = r.u(2)? as u8;
    let mut dqsbedge = 0u8;
    let mut dqbilevel = false;
    match dqprofile {
        x if x == tables::DQPROFILE_SINGLE_EDGE as u8
            || x == tables::DQPROFILE_DOUBLE_EDGES as u8 =>
        {
            dqsbedge = r.u(2)? as u8;
        }
        x if x == tables::DQPROFILE_ALL_MBS as u8 => {
            dqbilevel = r.bit()? != 0;
            if !dqbilevel {
                // 全宏块但非二值 = 每个宏块自带量化 -> 半步长不适用，帧头也不给 `altpq`。
                *halfpq = false;
                return Ok(Some(Dquant {
                    dqprofile,
                    dqsbedge,
                    dqbilevel,
                    altpq: None,
                }));
            }
        }
        _ => {}
    }
    let pqdiff = r.u(3)? as u8;
    let altpq = if pqdiff == 7 {
        r.u(5)? as u8
    } else {
        pq.wrapping_add(pqdiff).wrapping_add(1)
    };
    Ok(Some(Dquant {
        dqprofile,
        dqsbedge,
        dqbilevel,
        altpq: Some(altpq),
    }))
}

impl FrameHeader {
    /// 解一帧的帧头。
    ///
    /// `mb_width` / `mb_height` 是**宏块网格**的尺寸（`(coded_width + 15) / 16`），
    /// 来自容器 —— Simple/Main 的序列头里没有尺寸。位平面的大小由它决定。
    ///
    /// `state` 是跨帧状态（见 `HeaderState`），本函数会就地更新它。
    pub fn parse(
        data: &[u8],
        seq: &SeqHeader,
        mb_width: usize,
        mb_height: usize,
        state: &mut HeaderState,
    ) -> Result<Self> {
        let mut r = BitReader::new(data);
        Self::parse_from(&mut r, seq, mb_width, mb_height, state)
    }

    /// 同 [`FrameHeader::parse`]，但**复用调用方的位读取器** —— 宏块层紧跟在帧头之后，
    /// 解完帧头要接着往下读，所以游标必须留在同一个读取器上。
    pub fn parse_from(
        r: &mut BitReader,
        seq: &SeqHeader,
        mb_width: usize,
        mb_height: usize,
        state: &mut HeaderState,
    ) -> Result<Self> {
        if mb_width == 0 || mb_height == 0 {
            return Err(Error::Bad("vc1: 宏块网格尺寸为 0"));
        }

        if seq.finterpflag {
            let _interpfrm = r.bit()?;
        }
        let _framecnt = r.u(2)?; // 帧序号，解码不用
        let rangeredfrm = if seq.rangered { r.bit()? != 0 } else { false };

        // 图像类型：一位定 P；否则（有 B 帧时才再读一位）定 I。
        let pict_type = if r.bit()? != 0 {
            PictType::P
        } else if seq.max_b_frames != 0 && r.bit()? == 0 {
            return Err(Error::Unsupported("vc1: B 帧（宏块层的双向预测还没做）".into()));
        } else {
            PictType::I
        };
        if pict_type == PictType::I {
            let _buffer_fullness = r.u(7)?;
        }

        // 取整控制：位流里没有这一位 —— I 帧置 1，P 帧翻转。
        match pict_type {
            PictType::I => state.rnd = true,
            PictType::P => state.rnd ^= true,
        }
        let rnd = state.rnd;

        let pqindex = r.u(5)? as u8;
        if pqindex == 0 {
            return Err(Error::Bad("vc1: pqindex 为 0"));
        }
        let pq = tables::PQUANT[seq.quantizer_mode.pquant_row()][pqindex as usize];
        let mut halfpq = if pqindex < 9 { r.bit()? != 0 } else { false };
        let pquantizer = match seq.quantizer_mode {
            QuantizerMode::Implicit => pqindex < 9,
            QuantizerMode::NonUniform => false,
            QuantizerMode::Explicit => r.bit()? != 0,
            QuantizerMode::Uniform => true,
        };

        let mvrange = if seq.extended_mv {
            unary(r, 0, 3)? as u8
        } else {
            0
        };
        // `k_x` / `k_y` 是 MV 分量的位数；范围是对称的。
        let k_x = mvrange + 9 + (mvrange >> 1);
        let k_y = mvrange + 8;
        let range_x = 1 << (k_x - 1);
        let range_y = 1 << (k_y - 1);

        // 多分辨率：逐帧改分辨率。只支持 I/P，而 P 也带这个字段
        // （上游的条件是「非 B 帧」，我们这里恒真）。
        let respic = if seq.multires { r.u(2)? as u8 } else { 0 };
        // IntraX8：`res_x8` 置位时 I 帧的帧头多一位。本实现还没做 IntraX8，
        // 但**位要读掉**，否则后面全错位。
        let x8_type = seq.res_x8 && pict_type == PictType::I && r.bit()? != 0;

        let mut p = None;
        // I 帧的帧头里没有帧级量化变化这一段（上游只在 P / B 分支里读）。
        let mut frame_dquant = None;
        if pict_type == PictType::P {
            let lowquant = if pq > 12 { 0 } else { 1 };
            let mv_mode = MvMode::from_u32(
                tables::MV_PMODE_TABLE[lowquant][unary(r, 1, 4)? as usize] as u32,
            )
            .ok_or(Error::Bad("vc1: 非法的 MV 模式"))?;

            let mut mv_mode2 = None;
            let mut lumscale = 0u8;
            let mut lumshift = 0u8;
            if mv_mode == MvMode::IntensityComp {
                mv_mode2 = Some(
                    MvMode::from_u32(
                        tables::MV_PMODE_TABLE2[lowquant][unary(r, 1, 3)? as usize] as u32,
                    )
                    .ok_or(Error::Bad("vc1: 非法的亮度补偿 MV 模式"))?,
                );
                lumscale = r.u(6)? as u8;
                lumshift = r.u(6)? as u8;
            }
            let qs_last = state.quarter_sample;
            let eff = mv_mode2.unwrap_or(mv_mode);
            let quarter_sample = eff.quarter_sample();
            let mspel = eff.mspel();
            state.quarter_sample = quarter_sample;

            // 混合 MV 才有一张「这个宏块是 4MV 吗」的位平面；否则整帧 1MV。
            let mixed = mv_mode == MvMode::MixedMv
                || (mv_mode == MvMode::IntensityComp && mv_mode2 == Some(MvMode::MixedMv));
            let mv_type_plane = if mixed {
                Some(bitplane::decode(r, mb_width, mb_height)?)
            } else {
                None
            };
            let skip_plane = bitplane::decode(r, mb_width, mb_height)?;

            let mv_table_index = r.u(2)? as u8;
            let cbptab = r.u(2)? as u8;
            // 帧级量化变化**只在 P（和 B）帧里**，而且排在位平面之后。
            let dquant = if seq.dquant != 0 {
                vop_dquant(r, seq, pq, &mut halfpq)?
            } else {
                None
            };

            let (ttmbf, tt) = if seq.vstransform {
                let ttmbf = r.bit()? != 0;
                let tt = if ttmbf {
                    tables::TTFRM_TO_TT[r.u(2)? as usize]
                } else {
                    tables::TT_8X8
                };
                (ttmbf, tt)
            } else {
                (true, tables::TT_8X8)
            };

            p = Some(PFrameHeader {
                mv_mode,
                mv_mode2,
                lumscale,
                lumshift,
                quarter_sample,
                qs_last,
                mspel,
                mv_table_index,
                cbptab,
                ttmbf,
                tt,
                mv_type_plane,
                skip_plane,
            });
            frame_dquant = dquant;
        }

        let (mut c_ac_table_index, mut y_ac_table_index, mut dc_table_index) = (0u8, None, 0u8);
        if !x8_type {
            c_ac_table_index = decode012(r)?;
            if pict_type == PictType::I {
                y_ac_table_index = Some(decode012(r)?);
            }
            dc_table_index = r.bit()? as u8;
        }

        Ok(Self {
            pict_type,
            rnd,
            pqindex,
            pq,
            halfpq,
            pquantizer,
            mvrange,
            range_x,
            range_y,
            respic,
            rangeredfrm,
            dquant: frame_dquant,
            x8_type,
            c_ac_table_index,
            y_ac_table_index,
            dc_table_index,
            p,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一段**持有**字节的位流（`BitReader` 借的是它，不能是临时值）。
    struct Bits(Vec<u8>);

    impl Bits {
        /// 把 `"1010"` 这样的串变成位流（左侧为高位）。
        fn new(s: &str) -> Self {
            let b: Vec<u8> = s.chars().map(|c| (c == '1') as u8).collect();
            let mut out = vec![0u8; b.len().div_ceil(8)];
            for (i, v) in b.iter().enumerate() {
                out[i / 8] |= v << (7 - i % 8);
            }
            Self(out)
        }

        fn bytes(&self) -> &[u8] {
            &self.0
        }

        fn reader(&self) -> BitReader<'_> {
            BitReader::new(&self.0)
        }
    }

    /// 把 `SeqHeader` 按位流布局**写回去**（顺带把布局写明白）。
    ///
    /// ⚠️ 这是**自造自读的往返** —— 按仓规「必要但不充分」：它能钉住「宽度/顺序在
    /// 读写两侧一致」，钉不住「布局与规范一致」（那要真样本 + `ffmpeg`，见
    /// `tests/header.rs`，那边逐帧比 3719 帧）。它的价值在于**覆盖真样本走不到的
    /// 分支**（Simple profile、`rangered`、`multires`…）与「非法组合要被拒」。
    fn build_seq(h: &SeqHeader) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        let mut put = |v: u32, n: u32| {
            for i in (0..n).rev() {
                bits.push(((v >> i) & 1) as u8);
            }
        };
        put(h.profile as u32, 2);
        put(0, 1); // res_y411
        put(0, 1); // res_sprite
        put(h.frmrtq_postproc as u32, 3);
        put(h.bitrtq_postproc as u32, 5);
        put(h.loop_filter as u32, 1);
        put(h.res_x8 as u32, 1);
        put(h.multires as u32, 1);
        put(h.res_fasttx as u32, 1);
        put(h.fastuvmc as u32, 1);
        put(h.extended_mv as u32, 1);
        put(h.dquant as u32, 2);
        put(h.vstransform as u32, 1);
        put(0, 1); // res_transtab
        put(h.overlap as u32, 1);
        put(h.resync_marker as u32, 1);
        put(h.rangered as u32, 1);
        put(h.max_b_frames as u32, 3);
        put(h.quantizer_mode as u32, 2);
        put(h.finterpflag as u32, 1);
        put(h.res_rtm_flag as u32, 1);
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        bits.chunks(8)
            .map(|c| c.iter().fold(0u8, |a, b| (a << 1) | b))
            .collect()
    }

    fn sample_seq() -> SeqHeader {
        SeqHeader {
            profile: Profile::Main,
            frmrtq_postproc: 7,
            bitrtq_postproc: 0,
            loop_filter: true,
            multires: false,
            res_x8: false,
            res_fasttx: true,
            fastuvmc: false,
            extended_mv: true,
            dquant: 0,
            vstransform: true,
            overlap: false,
            resync_marker: false,
            rangered: false,
            max_b_frames: 0,
            quantizer_mode: QuantizerMode::Implicit,
            finterpflag: false,
            res_rtm_flag: true,
        }
    }

    #[test]
    fn seq_header_round_trip() {
        let main = sample_seq();
        assert_eq!(SeqHeader::parse(build_seq(&main).as_slice()).unwrap(), main);

        // Simple profile：`fastuvmc` 必须开、`extended_mv` 必须关、环路滤波必须关。
        let simple = SeqHeader {
            profile: Profile::Simple,
            loop_filter: false,
            fastuvmc: true,
            extended_mv: false,
            res_x8: true,
            multires: true,
            resync_marker: true,
            rangered: false,
            quantizer_mode: QuantizerMode::NonUniform,
            ..main
        };
        assert_eq!(SeqHeader::parse(build_seq(&simple).as_slice()).unwrap(), simple);
    }

    #[test]
    fn seq_header_rejects_illegal() {
        let main = sample_seq();
        let rejects = |h: &SeqHeader, what: &str| {
            let bytes = build_seq(h);
            assert!(
                SeqHeader::parse(bytes.as_slice()).is_err(),
                "本该被拒：{what}"
            );
        };
        let simple = |loop_filter, fastuvmc, extended_mv| SeqHeader {
            profile: Profile::Simple,
            loop_filter,
            fastuvmc,
            extended_mv,
            ..main
        };
        rejects(&simple(true, true, false), "Simple + loop_filter");
        rejects(&simple(false, false, false), "Simple 不带 fastuvmc");
        rejects(&simple(false, true, true), "Simple + extended_mv");
        rejects(
            &SeqHeader {
                profile: Profile::Advanced,
                ..main
            },
            "Advanced profile（不在这一层范围内）",
        );
    }

    #[test]
    fn unary_consumes_the_right_number_of_bits() {
        // stop = 1：读 0 直到遇见 1。
        let b = Bits::new("10000000");
        let mut r = b.reader();
        assert_eq!(unary(&mut r, 1, 4).unwrap(), 0);
        assert_eq!(r.bit_pos(), 1, "首位就是 stop，只该吃 1 位");

        let b = Bits::new("00100000");
        let mut r = b.reader();
        assert_eq!(unary(&mut r, 1, 4).unwrap(), 2);
        assert_eq!(r.bit_pos(), 3);

        // 到上限就停：**没有** stop 也得停，且恰好吃 `max` 位。
        let b = Bits::new("00000000");
        let mut r = b.reader();
        assert_eq!(unary(&mut r, 1, 4).unwrap(), 4);
        assert_eq!(r.bit_pos(), 4);

        // stop = 0：读 1 直到遇见 0。
        let b = Bits::new("01110000");
        let mut r = b.reader();
        assert_eq!(unary(&mut r, 0, 3).unwrap(), 0);
        assert_eq!(r.bit_pos(), 1);

        let b = Bits::new("11100000");
        let mut r = b.reader();
        assert_eq!(unary(&mut r, 0, 3).unwrap(), 3);
        assert_eq!(r.bit_pos(), 3);
    }

    #[test]
    fn decode012_shapes() {
        let b = Bits::new("00000000");
        let mut r = b.reader();
        assert_eq!(decode012(&mut r).unwrap(), 0);
        assert_eq!(r.bit_pos(), 1);

        let b = Bits::new("10000000");
        let mut r = b.reader();
        assert_eq!(decode012(&mut r).unwrap(), 1);
        assert_eq!(r.bit_pos(), 2);

        let b = Bits::new("11000000");
        let mut r = b.reader();
        assert_eq!(decode012(&mut r).unwrap(), 2);
        assert_eq!(r.bit_pos(), 2);
    }

    /// `rnd` 位流里**没有**这一位：I 帧置 1，P 帧翻转。所以 I,P,P,P -> 1,0,1,0。
    ///
    /// 位流短到后面必然 `Eof`（要整帧解完得铺二十几位，而这里只验状态机）——
    /// 无所谓：`rnd` 在 `pict_type` 之后**立刻**就算出来了，错误发生在它之后。
    #[test]
    fn rnd_toggles_only_on_p_frames() {
        let seq = sample_seq();
        let mut st = HeaderState::default();
        let (mw, mh) = (2usize, 2usize);

        // 10 位：framecnt(2) + pict_type=0(I)(1) + fullness(7)。`max_b_frames == 0`
        // 时 I/P 只花这 1 位（不用再读「是不是 B」那一位）。
        let i = Bits::new("0000000000");
        let _ = FrameHeader::parse(i.bytes(), &seq, mw, mh, &mut st);
        assert!(st.rnd, "I 帧把 rnd 置 1");

        // 3 位：framecnt(2) + pict_type=1(P)(1)。
        let p = Bits::new("0010");
        for want in [false, true, false] {
            let _ = FrameHeader::parse(p.bytes(), &seq, mw, mh, &mut st);
            assert_eq!(st.rnd, want, "P 帧翻转 rnd");
        }
    }
}
