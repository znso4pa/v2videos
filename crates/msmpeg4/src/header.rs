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
                    return Err(Error::Unsupported(
                        "WMV2 的帧头是另一套（见 wmv2dec.c），还没接".into(),
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
                    return Err(Error::Unsupported(
                        "WMV2 的帧头是另一套（见 wmv2dec.c），还没接".into(),
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
