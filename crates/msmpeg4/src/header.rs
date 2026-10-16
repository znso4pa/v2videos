//! MS-MPEG-4（MP41/MP42/MP43 / WMV1-2）的帧头解析。
//!
//! **没有起始码**（V1 除外，见下）—— 每帧直接从位流开头读：
//!
//! ```text
//!   ── V1 专有 ──
//!     start_code   : 32 bit (必须是 0x00000100)
//!     frame_number : 5 bit  (跳过，解码用不着)
//!
//!   picture_type : 2 bit  (+1，所以 1 = I，2 = P)
//!   qscale       : 5 bit  (1..31)
//!
//!   ── I 帧 ──
//!     slice_code   : 5 bit
//!                    V1：1..=mb_height，直接就是 slice_height
//!                    其余：0x17 = 一个 slice；slice_height = mb_height / (code-0x16)
//!     ── 按版本分叉 ──
//!     V1/V2：rl_chroma = 2、rl = 2、dc = 0（**写死**，不占位）
//!     V3   ：rl_chroma = decode012、rl = decode012、dc = 1 位
//!     WMV1 ：ext_header（见下）、per_mb_rl_table（码率 > 50K 才有这一位）、
//!            rl_chroma / rl（**没有** per_mb_rl_table 时才读）、dc = 1 位
//!
//!   ── P 帧 ──
//!     ── 按版本分叉 ──
//!     V1   ：use_skip = 1（写死）、rl = 2、rl_chroma = 2、dc = 0、mv = 0
//!     V2   ：use_skip = 1 位、rl = 2、rl_chroma = 2、dc = 0、mv = 0
//!     V3   ：use_skip = 1 位、rl = decode012、rl_chroma = rl、dc = 1 位、mv = 1 位
//!     WMV1 ：use_skip = 1 位、per_mb_rl_table（同上）、rl / rl_chroma、dc、mv
//! ```
//!
//! ## ext header（17 位：fps 5 + 码率 11 + `flipflop_rounding` 1）
//!
//! ⚠️ **位置随版本不同，这是最容易搞错的一处**：
//!
//! - **V1/V2/V3**：在 **I 帧尾部**（所有 slice 解完之后），由 `h263dec.c` 调；
//!   判据是「帧内剩余位数落在 `[length, length+8)`」——`length` = 17（V3 及以上）
//!   或 16。见 [`parse_ext_header`]。
//! - **WMV1**：在 **I 帧帧头里**（slice_code 之后），窗口是固定的前 4 字节。
//!
//! 它给出的**码率**还会反过来决定 WMV1 帧头后面的字段读不读（`per_mb_rl_table`）。

use crate::bitreader::BitReader;
use crate::error::{Error, Result};
use crate::version::Version;

/// 码率高于它，WMV1 的帧头里才多一个 `per_mb_rl_table` 位（ffmpeg 的 `MBAC_BITRATE`）。
pub const MBAC_BITRATE: u32 = 50 * 1024;
/// `inter_intra_pred` 的码率上限（ffmpeg 的 `II_BITRATE`）。
pub const II_BITRATE: u32 = 128 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureType {
    I,
    P,
}

/// 帧头解析要用的**跨帧状态**。
///
/// 存在的唯一理由：**WMV1 的帧头里有的字段「读不读」取决于码率**，
/// 而码率是 ext header 给的 —— I 帧的 ext header 在帧头里、P 帧自己不读，
/// 所以 P 帧只能用**上一个 I 帧留下的**那个值。
/// （ffmpeg 那边这就是 `MpegEncContext.bit_rate` 这个上下文字段。）
#[derive(Debug, Clone, Copy, Default)]
pub struct HeaderState {
    /// 上一个 ext header 给出的码率（bit/s）。没读到过就是 0。
    pub bit_rate: u32,
}

#[derive(Debug, Clone)]
pub struct FrameHeader {
    pub picture_type: PictureType,
    /// 1..=31。
    pub qscale: u8,
    /// I 帧：一个 slice 覆盖多少宏块行（P 帧恒为整帧高）。
    pub slice_height: u32,
    pub rl_table_index: u8,
    pub rl_chroma_table_index: u8,
    pub dc_table_index: u8,
    pub mv_table_index: u8,
    pub use_skip_mb_code: bool,
    /// **每个宏块自带 RL 表下标**（WMV1 的 `per_mb_rl_table`）。开时帧头里的
    /// `rl_table_index` / `rl_chroma_table_index` 是无效的，改由每个宏块读。
    pub per_mb_rl_table: bool,
    /// **帧内块用邻块 DC/AC 预测**（H.263 的 `inter_intra_pred`）。
    ///
    /// WMV1 由「尺寸 + 码率」推出来，不是位流里的位：
    /// `宽×高 < 320×240 且 码率 ≤ 128K`。
    pub inter_intra_pred: bool,
    /// WMV1 在帧头里读到的 ext header 带来的 `flipflop_rounding`（其余版本是 false，
    /// 它们的那一份在帧尾，见 [`parse_ext_header`]）。
    pub flipflop_rounding: bool,
    /// 帧头里读到的码率（bit/s）——只有 WMV1 的 ext header 会填，其余是 0。
    pub bit_rate: u32,
}

// ════════════════════════════════════════════════════════════════════════════
// WMV2（WMV8）—— 帧头与 MS-MPEG-4 那套**不是一回事**
// ════════════════════════════════════════════════════════════════════════════

/// WMV2 的**能力位** —— 全部来自容器的 `extradata`（ASF 那条视频轨的私有数据前 4 字节）。
///
/// ⚠️ **它不是位流的一部分**：解码器在 init 时读一次，之后每帧沿用。
/// 所以它得由调用方（`Decoder`）带进来，不能从帧数据里推。
#[derive(Debug, Clone, Copy, Default)]
pub struct Wmv2Flags {
    /// 码率（`11 位 × 1024`）。ffmpeg 只用它推 `inter_intra_pred`，而 WMV2 那边
    /// 这个开关恒为 0 —— 所以它**解出来了但没有用途**，留着是为了把 extradata 的布局写全。
    #[allow(dead_code)]
    pub bit_rate: u32,
    /// 半像素 MV 时要不要再读一个 `hshift` 位。
    pub mspel: bool,
    /// 环路滤波 —— 只有 IntraX8 那条路用（本实现还没做），所以目前只解不用。
    #[allow(dead_code)]
    pub loop_filter: bool,
    /// 有没有 **ABT**（自适应块变换：4×8 / 8×4）。
    pub abt: bool,
    /// 有没有 **IntraX8**（`j_type` 那一套，一个独立的内帧编解码器）。
    pub j_type: bool,
    /// MV 预测里那个「左上差得够大就多读一位 type」的开关。
    pub top_left_mv: bool,
    pub per_mb_rl: bool,
    /// 3 位；`slice_height = mb_height / code`。
    pub code: u8,
}

impl Wmv2Flags {
    /// 从 `extradata` 的前 4 字节解出能力位。
    ///
    /// 字段顺序照 ffmpeg 的 `decode_ext_header`：fps(5) 码率(11) mspel(1)
    /// loop_filter(1) abt(1) j_type(1) top_left_mv(1) per_mb_rl(1) code(3)。
    pub fn from_extradata(extra: &[u8]) -> Result<Self> {
        if extra.len() < 4 {
            return Err(Error::Bad("WMV2 的 extradata 至少要 4 字节"));
        }
        let mut r = BitReader::new(&extra[..4]);
        let _fps = r.u(5)?;
        let bit_rate = r.u(11)? * 1024;
        let mspel = r.bit()? != 0;
        let loop_filter = r.bit()? != 0;
        let abt = r.bit()? != 0;
        let j_type = r.bit()? != 0;
        let top_left_mv = r.bit()? != 0;
        let per_mb_rl = r.bit()? != 0;
        let code = r.u(3)? as u8;
        if code == 0 {
            return Err(Error::Bad("WMV2 的 extradata 里 slice code 为 0"));
        }
        Ok(Self {
            bit_rate,
            mspel,
            loop_filter,
            abt,
            j_type,
            top_left_mv,
            per_mb_rl,
            code,
        })
    }
}

/// WMV2 一帧的**完整帧头**（两段合起来 —— ffmpeg 把它拆成 picture / secondary 两趟，
/// 那是为了在中间插 `ff_mpv_frame_start`；我们顺序解，一趟做完）。
#[derive(Debug, Clone)]
pub struct Wmv2Header {
    pub picture_type: PictureType,
    pub qscale: u8,
    pub slice_height: u32,
    /// **每个宏块是不是 skip** —— 整帧预先读出来（不在宏块循环里逐位读，这是 WMV2
    /// 与 MS-MPEG-4 最大的结构差异）。长度 = `mb_w * mb_h`。
    pub mb_skip: Vec<bool>,
    pub rl_table_index: u8,
    pub rl_chroma_table_index: u8,
    pub dc_table_index: u8,
    pub mv_table_index: u8,
    pub per_mb_rl_table: bool,
    /// 宏块类型表下标（0..=2）—— 由 `qscale` 与位流里的 `cbp_index` 一起选出来。
    pub cbp_table_index: usize,
    pub mspel: bool,
    /// ABT：`per_mb_abt` 为真时每个宏块自己读 `abt_type`。
    pub per_mb_abt: bool,
    pub abt_type: u8,
    /// 这一帧走 **IntraX8** —— 本实现不做（见 README 的能力矩阵）。
    pub j_type: bool,
    /// 本帧解完后 `no_rounding` 的新值（P 帧是 `^= 1`，I 帧恒 1）。
    pub no_rounding: bool,
}

impl Wmv2Header {
    /// 归一成**公共帧头** —— MB 循环只认这些字段，版本差异由调用方另行带着走。
    ///
    /// ⚠️ `use_skip_mb_code` 对 WMV2 是 **false**：它的逐宏块 skip 位**不在宏块循环里读**，
    /// 而是整帧预先读成一张表（见 `mb_skip`）—— 循环里再读一次就会错位。
    pub fn to_frame_header(&self) -> FrameHeader {
        FrameHeader {
            picture_type: self.picture_type,
            qscale: self.qscale,
            slice_height: self.slice_height,
            rl_table_index: self.rl_table_index,
            rl_chroma_table_index: self.rl_chroma_table_index,
            dc_table_index: self.dc_table_index,
            mv_table_index: self.mv_table_index,
            use_skip_mb_code: false,
            per_mb_rl_table: self.per_mb_rl_table,
            inter_intra_pred: false,
            flipflop_rounding: false,
            bit_rate: 0,
        }
    }
}

/// `parse_wmv2_frame` 的结果。
pub enum Wmv2Frame {
    /// **整帧都是 skip 宏块**（ffmpeg 的 `FRAME_SKIPPED`）—— 这一帧不解，
    /// 直接把参考帧当输出。位流里那一段全是 1 位。
    Skipped,
    /// 正常一帧。
    Frame(Box<Wmv2Header>),
}

/// `wmv2.h` 的 `wmv2_get_cbp_table_index`：`map[(q>10)+(q>20)][cbp_index]`。
///
/// ⚠️ 这个 map **只会选出 0/1/2** —— 第 4 张表（`ff_table_mb_non_intra`）解码侧用不到。
fn wmv2_cbp_table_index(qscale: u8, cbp_index: u8) -> usize {
    const MAP: [[u8; 3]; 3] = [[0, 2, 1], [1, 0, 2], [2, 1, 0]];
    let row = usize::from(qscale > 10) + usize::from(qscale > 20);
    MAP[row][(cbp_index as usize).min(2)] as usize
}

/// 解 WMV2 一帧的帧头，返回**结果与消费掉的位数**。
/// `no_rounding_in` 是**上一帧留下的**那个值（P 帧要翻转它）。
pub fn parse_wmv2_frame(
    data: &[u8],
    mb_width: u32,
    mb_height: u32,
    f: &Wmv2Flags,
    no_rounding_in: bool,
) -> Result<(Wmv2Frame, usize)> {
    let mut r = BitReader::new(data);

    // ---- 第一段：帧类型 / qscale / 「整帧 skip」检测 ----
    // ⚠️ `pict_type` 只有 **1 位**（MS-MPEG-4 是 2 位）。
    let pict = r.bit()? + 1;
    let picture_type = match pict {
        1 => PictureType::I,
        2 => PictureType::P,
        _ => return Err(Error::Bad("WMV2 的 picture type 不合法")),
    };
    if picture_type == PictureType::I {
        // I 帧多一个 7 位的字段（ffmpeg 只把它打日志，语义没用到）。
        let _code = r.u(7)?;
    }
    let qscale = r.u(5)? as u8;
    if qscale == 0 {
        return Err(Error::Bad("WMV2 的 qscale 为 0"));
    }

    if picture_type == PictureType::P && r.show(1)? != 0 {
        // 试探「整帧 skip」：`skip_type`(2) 之后按 25 位一组读，全 1 才算整帧跳过。
        // ⚠️ 这是**试探**（ffmpeg 用 `GetBitContext gb = s->gb` 拷一份来读），
        // 所以失败时要能退回原位 —— 用 `show`/`skip` 手工管位置。
        let save = r.bit_pos();
        let mut probe = BitReader::new(data);
        probe.skip(save as u32)?;
        if probe_all_skipped(&mut probe, mb_width, mb_height).is_some() {
            return Ok((Wmv2Frame::Skipped, save));
        }
    }

    // ---- 第二段：能力位 / 表选择 / 宏块 skip 表 ----
    let mut h = Wmv2Header {
        picture_type,
        qscale,
        slice_height: (mb_height / f.code.max(1) as u32).max(1),
        mb_skip: Vec::new(),
        rl_table_index: 0,
        rl_chroma_table_index: 0,
        dc_table_index: 0,
        mv_table_index: 0,
        per_mb_rl_table: false,
        cbp_table_index: 0,
        mspel: false,
        per_mb_abt: false,
        abt_type: 0,
        j_type: false,
        no_rounding: true,
    };

    match picture_type {
        PictureType::I => {
            h.j_type = f.j_type && r.bit()? != 0;
            if !h.j_type {
                h.per_mb_rl_table = f.per_mb_rl && r.bit()? != 0;
                if !h.per_mb_rl_table {
                    h.rl_chroma_table_index = decode012(&mut r)?;
                    h.rl_table_index = decode012(&mut r)?;
                }
                h.dc_table_index = r.bit()? as u8;
            }
            h.no_rounding = true;
        }
        PictureType::P => {
            h.mb_skip = parse_wmv2_mb_skip(&mut r, mb_width, mb_height)?;
            let cbp_index = decode012(&mut r)?;
            h.cbp_table_index = wmv2_cbp_table_index(qscale, cbp_index);
            h.mspel = f.mspel && r.bit()? != 0;
            if f.abt {
                // ⚠️ ffmpeg 这里是 `get_bits1() ^ 1` —— 位流里的 1 表示**逐宏块**读。
                h.per_mb_abt = r.bit()? == 0;
                if !h.per_mb_abt {
                    h.abt_type = decode012(&mut r)?;
                }
            }
            h.per_mb_rl_table = f.per_mb_rl && r.bit()? != 0;
            if !h.per_mb_rl_table {
                h.rl_table_index = decode012(&mut r)?;
                h.rl_chroma_table_index = h.rl_table_index;
            }
            h.dc_table_index = r.bit()? as u8;
            h.mv_table_index = r.bit()? as u8;
            h.no_rounding = no_rounding_in ^ true;
        }
    }

    Ok((Wmv2Frame::Frame(Box::new(h)), r.bit_pos()))
}

/// 试探「整帧 skip」：读得下去且全 1 就返回 `Some(())`。
fn probe_all_skipped(r: &mut BitReader, mb_width: u32, mb_height: u32) -> Option<()> {
    const SKIP_TYPE_COL: u32 = 3;
    let skip_type = r.u(2).ok()?;
    let mut run = if skip_type == SKIP_TYPE_COL {
        mb_width
    } else {
        mb_height
    };
    while run > 0 {
        let block = run.min(25);
        let v = r.u(block).ok()?;
        // `v + 1 != 1 << block` = 这一组不全是 1 → 不是整帧 skip。
        if v + 1 != 1u32 << block {
            return None;
        }
        run -= block;
    }
    Some(())
}

/// WMV2 的**逐宏块 skip 表** —— 位流里按 4 种粒度给。
///
/// | `skip_type` | 粒度 |
/// |---|---|
/// | 0 | 全不 skip（一个位都不读） |
/// | 1 | 逐宏块 1 位（MPEG 风格） |
/// | 2 | 逐行：先 1 位「整行 skip？」，不是再逐宏块读 |
/// | 3 | 逐列：同上，按列 |
fn parse_wmv2_mb_skip(r: &mut BitReader, mb_w: u32, mb_h: u32) -> Result<Vec<bool>> {
    let mut skip = vec![false; (mb_w * mb_h) as usize];
    let skip_type = r.u(2)?;
    match skip_type {
        0 => {}
        1 => {
            for v in skip.iter_mut() {
                *v = r.bit()? != 0;
            }
        }
        2 => {
            for y in 0..mb_h {
                if r.bit()? != 0 {
                    for x in 0..mb_w {
                        skip[(y * mb_w + x) as usize] = true;
                    }
                } else {
                    for x in 0..mb_w {
                        skip[(y * mb_w + x) as usize] = r.bit()? != 0;
                    }
                }
            }
        }
        _ => {
            for x in 0..mb_w {
                if r.bit()? != 0 {
                    for y in 0..mb_h {
                        skip[(y * mb_w + x) as usize] = true;
                    }
                } else {
                    for y in 0..mb_h {
                        skip[(y * mb_w + x) as usize] = r.bit()? != 0;
                    }
                }
            }
        }
    }
    Ok(skip)
}

/// `decode012` 这个三值编码：读 1 位，0 → 0；否则再读 1 位，0 → 1，1 → 2。
pub fn decode012(r: &mut BitReader) -> Result<u8> {
    if r.bit()? == 0 {
        Ok(0)
    } else if r.bit()? == 0 {
        Ok(1)
    } else {
        Ok(2)
    }
}

/// 解析一帧的头部。返回帧头与**消费掉的位数**。
///
/// `mb_width` / `mb_height` = 宏块数（`ceil(宽/16)` / `ceil(高/16)`），用来算 I 帧的 slice_height。
pub fn parse_frame_header(
    data: &[u8],
    mb_width: u32,
    mb_height: u32,
    version: Version,
    st: &mut HeaderState,
) -> Result<(FrameHeader, usize)> {
    let mut r = BitReader::new(data);

    // ---- V1 专有的起始码 + 帧号 ----
    if version == Version::V1 {
        let start = r.u(32)?;
        if start != 0x0000_0100 {
            return Err(Error::Bad("V1 的起始码不是 0x00000100"));
        }
        let _frame_number = r.u(5)?;
    }

    let pict = r.u(2)? + 1;
    let picture_type = match pict {
        1 => PictureType::I,
        2 => PictureType::P,
        _ => return Err(Error::Bad("invalid picture type")),
    };
    let qscale = r.u(5)? as u8;
    if qscale == 0 {
        return Err(Error::Bad("qscale == 0"));
    }

    let mut h = FrameHeader {
        picture_type,
        qscale,
        slice_height: 1,
        rl_table_index: 0,
        rl_chroma_table_index: 0,
        dc_table_index: 0,
        mv_table_index: 0,
        use_skip_mb_code: false,
        per_mb_rl_table: false,
        inter_intra_pred: false,
        flipflop_rounding: false,
        bit_rate: 0,
    };

    match picture_type {
        PictureType::I => {
            let code = r.u(5)?;
            if version == Version::V1 {
                // V1 这里直接就是 slice_height（1..=mb_height），不是那个 0x17 编码。
                if code == 0 || code > mb_height {
                    return Err(Error::Bad("V1 的 slice height 越界"));
                }
                h.slice_height = code;
            } else {
                if code < 0x17 {
                    return Err(Error::Bad("slice code < 0x17"));
                }
                // 0x17 = 一个 slice；0x18 = 两个 slice…
                h.slice_height = (mb_height / (code - 0x16)).max(1);
            }

            match version {
                Version::V1 | Version::V2 => {
                    // 写死，不占位流：这两版的 I 帧只走 RL 表 2 / DC 表 0。
                    h.rl_chroma_table_index = 2;
                    h.rl_table_index = 2;
                    h.dc_table_index = 0;
                }
                Version::V3 => {
                    h.rl_chroma_table_index = decode012(&mut r)?;
                    h.rl_table_index = decode012(&mut r)?;
                    h.dc_table_index = r.bit()? as u8;
                }
                Version::Wmv1 => {
                    // ⚠️ ext header 在**帧头里**（V1/V2/V3 是在帧尾）。
                    // 窗口固定 4 字节 —— ffmpeg 传的就是 `(2+5+5+17+7)/8`。
                    let (br, flip) = parse_ext_header(&mut r, 4, version);
                    if br > 0 {
                        st.bit_rate = br; // 留给后面那些 P 帧
                    }
                    h.bit_rate = st.bit_rate;
                    h.flipflop_rounding = flip;

                    if h.bit_rate > MBAC_BITRATE {
                        h.per_mb_rl_table = r.bit()? != 0;
                    }
                    if !h.per_mb_rl_table {
                        h.rl_chroma_table_index = decode012(&mut r)?;
                        h.rl_table_index = decode012(&mut r)?;
                    }
                    h.dc_table_index = r.bit()? as u8;
                    // WMV1 的 I 帧恒关。
                    h.inter_intra_pred = false;
                }
                Version::Wmv2 => {
                    return Err(Error::Bad(
                        "WMV2 的帧头走 `parse_wmv2_frame`（另一套语法，1 位的 picture type）",
                    ))
                }
            }
            let _ = mb_width;
        }
        PictureType::P => {
            // P 帧头**不设** slice_height —— 它保持初始化时的 `mb_height`，整帧一个 slice。
            // 这决定 `first_slice_line`：只有第 0 行是，于是 DC 预测的边界与 MV 预测的
            // 「首行只用左邻」都靠它。默认成 1 会让每一行都被当成 slice 首行。
            h.slice_height = mb_height;

            match version {
                Version::V1 | Version::V2 => {
                    // V1 写死 1；V2 读一位。
                    h.use_skip_mb_code = version == Version::V1 || r.bit()? != 0;
                    h.rl_table_index = 2;
                    h.rl_chroma_table_index = 2;
                    h.dc_table_index = 0;
                    h.mv_table_index = 0;
                }
                Version::V3 => {
                    h.use_skip_mb_code = r.bit()? != 0;
                    h.rl_table_index = decode012(&mut r)?;
                    h.rl_chroma_table_index = h.rl_table_index;
                    h.dc_table_index = r.bit()? as u8;
                    h.mv_table_index = r.bit()? as u8;
                }
                Version::Wmv1 => {
                    h.use_skip_mb_code = r.bit()? != 0;
                    // ⚠️ 码率来自**上一个 I 帧**的 ext header（P 帧自己不读那一段）。
                    // 所以这里用的是 `st.bit_rate`，不是本地读出来的东西 ——
                    // 读错了这个位，整帧的字段会从这儿开始全体错位。
                    h.bit_rate = st.bit_rate;
                    if h.bit_rate > MBAC_BITRATE {
                        h.per_mb_rl_table = r.bit()? != 0;
                    }
                    if !h.per_mb_rl_table {
                        h.rl_table_index = decode012(&mut r)?;
                        h.rl_chroma_table_index = h.rl_table_index;
                    }
                    h.dc_table_index = r.bit()? as u8;
                    h.mv_table_index = r.bit()? as u8;
                    // WMV1 的 P 帧才可能开 —— 由「尺寸 + 码率」推出来，不是位流里的位。
                    h.inter_intra_pred =
                        mb_width * mb_height * 256 < 320 * 240 && h.bit_rate <= II_BITRATE;
                }
                Version::Wmv2 => {
                    return Err(Error::Bad(
                        "WMV2 的帧头走 `parse_wmv2_frame`（另一套语法，1 位的 picture type）",
                    ))
                }
            }
        }
    }

    Ok((h, r.bit_pos()))
}

/// 读一个 ext header（17 位：fps 5 + 码率 11 + `flipflop_rounding` 1）。
///
/// `window_bytes` = ffmpeg 传进来的 `buf_size`：
/// - **WMV1 在帧头里**：固定 4 字节（`(2+5+5+17+7)/8`）；
/// - **V1/V2/V3 在 I 帧尾部**：整帧的字节数，由调用方给。
///
/// 判据与 ffmpeg 逐字一致：设 `left` = 窗口内还剩多少位，
/// `length` = 17（`version >= V3`，含 WMV1）或 16。只有 `left ∈ [length, length+8)`
/// 才读；否则 `flipflop_rounding` 归零。
///
/// 返回 `(码率 bit/s, flipflop_rounding)`。
pub fn parse_ext_header(r: &mut BitReader, window_bytes: usize, version: Version) -> (u32, bool) {
    let length = if version == Version::V1 || version == Version::V2 {
        16
    } else {
        17
    };
    let left = window_bytes * 8;
    let left = left.saturating_sub(r.bit_pos());
    if left < length || left >= length + 8 {
        return (0, false);
    }
    // 读得动才读 —— 这里不能因为位流短就报错：ffmpeg 是「读不到就当没有」。
    let (fps, br, flip) = match (r.u(5), r.u(11), r.bit()) {
        (Ok(f), Ok(b), Ok(fl)) => (f, b, fl != 0),
        _ => return (0, false),
    };
    let _ = fps;
    let bit_rate = br * 1024;
    // `flipflop_rounding` 只有 `version >= V3` 才有这一位。
    let flip = flip && version != Version::V1 && version != Version::V2;
    (bit_rate, flip)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode012_matches_the_three_way_code() {
        // 位串 "0" -> 0
        assert_eq!(decode012(&mut BitReader::new(&[0b0000_0000])).unwrap(), 0);
        // 位串 "10" -> 1
        assert_eq!(decode012(&mut BitReader::new(&[0b1000_0000])).unwrap(), 1);
        // 位串 "11" -> 2
        assert_eq!(decode012(&mut BitReader::new(&[0b1100_0000])).unwrap(), 2);
    }

    #[test]
    fn parses_a_minimal_i_frame_header() {
        // pict=0(I) qscale=8 slice=0x17 rl_chroma=0 rl=0 dc=0
        // 00 01000 10111 0 0 0  = 2+5+5+1+1+1 = 15 bits
        let mut w = crate::testbit::BitWriter::new();
        w.put(0, 2); // I
        w.put(8, 5); // qscale
        w.put(0x17, 5); // slice
        w.put(0, 1); // decode012 -> 0
        w.put(0, 1); // decode012 -> 0
        w.put(0, 1); // dc
        let data = w.finish();
        let mut st = HeaderState::default();
        let (h, used) = parse_frame_header(&data, 120, 68, Version::V3, &mut st).unwrap();
        assert_eq!(h.picture_type, PictureType::I);
        assert_eq!(h.qscale, 8);
        assert_eq!(h.slice_height, 68); // 一个 slice
        assert_eq!(h.rl_table_index, 0);
        assert_eq!(used, 15);
    }

    #[test]
    fn p_frame_slice_height_is_the_whole_frame() {
        // pict=1(P) qscale=8 skip=0 rl=0 dc=0 mv=0 → 2+5+1+1+1+1 = 11 bits
        let mut w = crate::testbit::BitWriter::new();
        w.put(1, 2); // P
        w.put(8, 5); // qscale
        w.put(0, 1); // use_skip_mb_code = 0
        w.put(0, 1); // decode012 → rl_table_index 0
        w.put(0, 1); // dc_table_index
        w.put(0, 1); // mv_table_index
        let data = w.finish();
        let mut st = HeaderState::default();
        let (h, used) = parse_frame_header(&data, 120, 68, Version::V3, &mut st).unwrap();
        assert_eq!(h.picture_type, PictureType::P);
        assert_eq!(
            h.slice_height, 68,
            "P 帧整帧一个 slice（保持初始化时的 mb_height）"
        );
        assert_eq!(used, 11);
    }

    #[test]
    fn rejects_qscale_zero() {
        let mut w = crate::testbit::BitWriter::new();
        w.put(0, 2);
        w.put(0, 5); // qscale = 0
        let data = w.finish();
        assert!(matches!(
            {
            let mut st = HeaderState::default();
            parse_frame_header(&data, 1, 1, Version::V3, &mut st)
        },
            Err(Error::Bad(_))
        ));
    }

    /// **V1 的起始码**：不是 `0x00000100` 就该报错。
    #[test]
    fn v1_requires_its_start_code() {
        let mut w = crate::testbit::BitWriter::new();
        w.put(0x0000_0100, 32); // 正确的起始码
        w.put(7, 5); // 帧号
        w.put(0, 2); // I
        w.put(8, 5); // qscale
        // ⚠️ V1 的 slice_height 是 **5 位**（1..=31），装不下 68 ——
        // 这一版一个 slice 最多 31 个宏块行，超了就得分片。
        w.put(4, 5); // slice_height = 4
        let data = w.finish();
        let mut st = HeaderState::default();
        let (h, _) = parse_frame_header(&data, 120, 68, Version::V1, &mut st).unwrap();
        assert_eq!(h.slice_height, 4);
        assert_eq!(h.rl_table_index, 2, "V1 的 I 帧 RL 表写死为 2");
        assert_eq!(h.rl_chroma_table_index, 2);
        assert_eq!(h.dc_table_index, 0);

        // 起始码错 → 报错
        let mut w2 = crate::testbit::BitWriter::new();
        w2.put(0x0000_0101, 32);
        let mut st2 = HeaderState::default();
        assert!(parse_frame_header(&w2.finish(), 120, 68, Version::V1, &mut st2).is_err());
    }

    /// **V1 的 P 帧**：`use_skip_mb_code` 写死为 1，不占位。
    #[test]
    fn v1_p_frame_hardcodes_skip_and_tables() {
        let mut w = crate::testbit::BitWriter::new();
        w.put(0x0000_0100, 32);
        w.put(0, 5); // 帧号
        w.put(1, 2); // P
        w.put(8, 5); // qscale
        // 后面什么都不写 —— V1 的 P 帧头到这里就结束了
        let data = w.finish();
        let mut st = HeaderState::default();
        let (h, used) = parse_frame_header(&data, 120, 68, Version::V1, &mut st).unwrap();
        assert_eq!(h.picture_type, PictureType::P);
        assert!(h.use_skip_mb_code, "V1 的 P 帧写死 use_skip = 1");
        assert_eq!(h.rl_table_index, 2);
        assert_eq!(h.mv_table_index, 0);
        assert_eq!(used, 32 + 5 + 2 + 5);
    }

    /// **V2 的 P 帧**：`use_skip` 读一位，但三张表仍写死。
    #[test]
    fn v2_p_frame_reads_only_the_skip_bit() {
        let mut w = crate::testbit::BitWriter::new();
        w.put(1, 2); // P
        w.put(8, 5); // qscale
        w.put(1, 1); // use_skip
        let data = w.finish();
        let mut st = HeaderState::default();
        let (h, used) = parse_frame_header(&data, 120, 68, Version::V2, &mut st).unwrap();
        assert!(h.use_skip_mb_code);
        assert_eq!(h.rl_table_index, 2, "V2 的 P 帧 RL 表写死为 2");
        assert_eq!(h.rl_chroma_table_index, 2);
        assert_eq!(h.dc_table_index, 0);
        assert_eq!(h.mv_table_index, 0);
        assert_eq!(used, 8);
    }

    /// **WMV1 的 I 帧**：帧头里带 17 位的 ext header，且码率决定后面读不读 `per_mb_rl_table`。
    #[test]
    fn wmv1_i_frame_reads_the_ext_header_in_the_header() {
        let mut w = crate::testbit::BitWriter::new();
        w.put(0, 2); // I
        w.put(8, 5); // qscale
        w.put(0x17, 5); // slice code
        // 到这里消费了 12 位，窗口 32 位 → left = 20 ∈ [17, 25) → 读 ext header
        w.put(3, 5); // fps
        w.put(200, 11); // 码率 200 * 1024 = 204800 > MBAC_BITRATE
        w.put(1, 1); // flipflop_rounding
        w.put(1, 1); // per_mb_rl_table = 1（因为码率 > 50K）
        w.put(0, 1); // dc_table_index
        let data = w.finish();
        let mut st = HeaderState::default();
        let (h, _) = parse_frame_header(&data, 120, 68, Version::Wmv1, &mut st).unwrap();
        assert_eq!(h.bit_rate, 200 * 1024);
        assert!(h.flipflop_rounding);
        assert!(h.per_mb_rl_table, "码率 > 50K 才读这一位，且它决定 rl 表读不读");
        assert_eq!(h.dc_table_index, 0);
        assert!(!h.inter_intra_pred, "WMV1 的 I 帧恒关 inter_intra_pred");
    }

    /// **WMV1 的低码率**：`per_mb_rl_table` 那一位根本不出现。
    #[test]
    fn wmv1_low_bitrate_has_no_per_mb_rl_bit() {
        let mut w = crate::testbit::BitWriter::new();
        w.put(0, 2);
        w.put(8, 5);
        w.put(0x17, 5);
        w.put(3, 5); // fps
        w.put(10, 11); // 码率 10 * 1024 = 10240 ≤ 50K → 没有 per_mb_rl_table 位
        w.put(1, 1); // flipflop
        w.put(0, 1); // rl_chroma decode012 → 0
        w.put(0, 1); // rl decode012 → 0
        w.put(1, 1); // dc
        let data = w.finish();
        let mut st = HeaderState::default();
        let (h, _) = parse_frame_header(&data, 120, 68, Version::Wmv1, &mut st).unwrap();
        assert_eq!(h.bit_rate, 10 * 1024);
        assert!(!h.per_mb_rl_table);
        assert_eq!(h.dc_table_index, 1);
    }

    /// **WMV1 的 P 帧**：`per_mb_rl_table` 那一位读不读，取决于**上一个 I 帧**的码率。
    #[test]
    fn wmv1_p_frame_uses_the_bitrate_from_the_last_i_frame() {
        // 码率 200K > 50K → 有 per_mb_rl_table 位
        let mut st = HeaderState {
            bit_rate: 200 * 1024,
        };
        let mut w = crate::testbit::BitWriter::new();
        w.put(1, 2); // P
        w.put(8, 5); // qscale
        w.put(0, 1); // use_skip = 0
        w.put(1, 1); // per_mb_rl_table = 1 → 后面不读 rl 表
        w.put(1, 1); // dc
        w.put(0, 1); // mv
        let (h, used) = parse_frame_header(&w.finish(), 120, 68, Version::Wmv1, &mut st).unwrap();
        assert!(h.per_mb_rl_table);
        assert_eq!(used, 11);

        // 同一个位流，码率低 → 那一位不存在，字段整体前移一位，解出来是别的东西
        let mut st2 = HeaderState { bit_rate: 10 * 1024 };
        let mut w2 = crate::testbit::BitWriter::new();
        w2.put(1, 2);
        w2.put(8, 5);
        w2.put(0, 1); // use_skip
        w2.put(0, 1); // rl decode012 → 0
        w2.put(1, 1); // dc
        w2.put(0, 1); // mv
        let (h2, used2) =
            parse_frame_header(&w2.finish(), 120, 68, Version::Wmv1, &mut st2).unwrap();
        assert!(!h2.per_mb_rl_table);
        assert_eq!(h2.dc_table_index, 1);
        // 2 + 5 + 1(use_skip) + 1(rl) + 1(dc) + 1(mv) = 11 —— 少了 per_mb_rl_table 那一位
        assert_eq!(used2, 11);
    }

    /// **WMV1 的 `inter_intra_pred`**：由「尺寸 + 码率」推出来，不是位流里的位。
    #[test]
    fn wmv1_inter_intra_pred_comes_from_size_and_bitrate() {
        // 120×68 宏块 = 1920×1088 像素 > 320×240 → 关（不论码率）
        let mut st = HeaderState {
            bit_rate: 100 * 1024,
        };
        let mut w = crate::testbit::BitWriter::new();
        w.put(1, 2);
        w.put(8, 5);
        w.put(0, 1); // use_skip
        w.put(1, 1); // per_mb_rl_table（码率 > 50K）
        w.put(0, 1); // dc
        w.put(0, 1); // mv
        let (h, _) = parse_frame_header(&w.finish(), 120, 68, Version::Wmv1, &mut st).unwrap();
        assert!(!h.inter_intra_pred, "尺寸超过 320×240 → 关");

        // 小尺寸（10×10 宏块 = 160×160 < 320×240）且码率 ≤ 128K → 开
        let mut st2 = HeaderState {
            bit_rate: 100 * 1024,
        };
        let mut w2 = crate::testbit::BitWriter::new();
        w2.put(1, 2);
        w2.put(8, 5);
        w2.put(0, 1);
        w2.put(1, 1); // per_mb_rl_table
        w2.put(0, 1);
        w2.put(0, 1);
        let (h2, _) = parse_frame_header(&w2.finish(), 10, 10, Version::Wmv1, &mut st2).unwrap();
        assert!(h2.inter_intra_pred, "小尺寸 + 低码率 → 开");
    }

    /// **ext header 的窗口判据**：位数落在 `[length, length+8)` 之外就不读。
    #[test]
    fn ext_header_window_matches_ffmpeg() {
        // 窗口 4 字节 = 32 位，已消费 12 位 → left = 20，读
        let mut w = crate::testbit::BitWriter::new();
        w.put(0, 12);
        w.put(3, 5);
        w.put(200, 11);
        w.put(1, 1);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        r.u(12).unwrap();
        let (br, flip) = parse_ext_header(&mut r, 4, Version::Wmv1);
        assert_eq!(br, 200 * 1024);
        assert!(flip);

        // 窗口只给 2 字节 = 16 位，已消费 12 位 → left = 4 < 17 → 不读
        let mut w2 = crate::testbit::BitWriter::new();
        w2.put(0, 12);
        w2.put(3, 5);
        w2.put(200, 11);
        let bytes2 = w2.finish();
        let mut r2 = BitReader::new(&bytes2);
        r2.u(12).unwrap();
        assert_eq!(parse_ext_header(&mut r2, 2, Version::Wmv1), (0, false));
    }

    /// V1/V2 的 ext header 是 **16 位**（没有 `flipflop_rounding`）。
    #[test]
    fn v1_v2_ext_header_is_sixteen_bits_without_flipflop() {
        let mut w = crate::testbit::BitWriter::new();
        w.put(0, 12);
        w.put(3, 5);
        w.put(200, 11);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        r.u(12).unwrap();
        let (br, flip) = parse_ext_header(&mut r, 4, Version::V2);
        assert_eq!(br, 200 * 1024);
        assert!(!flip, "V1/V2 没有 flipflop_rounding 这一位");
    }
}
