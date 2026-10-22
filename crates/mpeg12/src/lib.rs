//! MPEG-1 / MPEG-2 **视频** elementary stream 解析。
//!
//! 只做码流层：把 start code 之间的字节切成 sequence / picture / slice /
//! macroblock / block，并把头里的字段读出来。重建（IDCT、运动补偿）在这一层之上。
//!
//! ```text
//!   sequence header 00 00 01 B3  宽高、帧率、码率
//!   picture header  00 00 01 00  编码类型 I/P/B、temporal_reference
//!   slice           00 00 01 01..AF
//!   macroblock / block            由 slice 内部按位读
//!   sequence end    00 00 01 B7
//! ```
//!
//! 设计约束：**不用 unsafe、不调 FFI**，全部安全 Rust，按位读取走一个游标。

/// MPEG-1 的帧率表（`frame_rate_code` 1..=8），单位 1/1000 帧每秒。
const FRAME_RATE_TABLE: [u32; 9] = [0, 23_976, 24_000, 25_000, 29_970, 30_000, 50_000, 59_940, 60_000];

/// 宽高比表（`aspect_ratio_information` 1..=4）。
const ASPECT_TABLE: [(u32, u32); 5] = [(0, 0), (1, 1), (4, 3), (16, 9), (2, 21)];

/// 位读取器**复用 `v2v-msmpeg4` 的那一份**（架构不变量 3：共享数值内核只有一份）。
///
/// 这里只做再导出，不新增实现 —— 第二份一定会漂。
pub use v2v_msmpeg4::BitReader;

/// sequence header 里我们关心的字段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceHeader {
    pub width: u32,
    pub height: u32,
    pub aspect: (u32, u32),
    /// 帧率，单位 1/1000 帧每秒（`25000` = 25 fps）。
    pub frame_rate_milli: u32,
    pub bit_rate: u32,
    pub vbv_buffer_size: u32,
    /// 由调用方根据 sequence extension 判定（见 [`SequenceHeader::mpeg2_stream`]）。
    pub mpeg2: bool,
}

impl SequenceHeader {
    /// 这条流是不是 MPEG-2。
    ///
    /// 解析器本身不知道版本（sequence header 里没有版本位），由调用方在拿到
    /// 整条 elementary stream 时用 [`has_sequence_extension`] 判出来再传进来。
    pub fn mpeg2_stream(&self) -> bool {
        self.mpeg2
    }
}

impl SequenceHeader {
    /// 解析 `00 00 01 B3` **之后**的字节。
    pub fn parse(payload: &[u8]) -> Option<Self> {
        let mut r = BitReader::new(payload);
        let width = r.u(12).ok()?;
        let height = r.u(12).ok()?;
        let aspect_code = r.u(4).ok()? as usize;
        let frame_rate_code = r.u(4).ok()? as usize;
        if width == 0 || height == 0 || aspect_code >= ASPECT_TABLE.len() || frame_rate_code == 0 {
            return None;
        }
        let aspect = ASPECT_TABLE[aspect_code];
        let frame_rate_milli = *FRAME_RATE_TABLE.get(frame_rate_code)?;
        let bit_rate = r.u(18).ok()?;
        r.u(1).ok()?; // marker
        let vbv_buffer_size = r.u(10).ok()?;
        r.u(1).ok()?; // constrained_parameters_flag
        // 之后是 load_intra_quantiser_matrix / load_non_intra_quantiser_matrix
        // 两个标志（各带 64 字节表），这里不读 —— 它们不影响头字段。
        // **sequence header 里没有 MPEG-1/2 标志位**，版本要看后面有没有
        // sequence extension（`00 00 01 B5`），见 `has_sequence_extension`。

        Some(Self {
            width,
            height,
            aspect,
            frame_rate_milli,
            bit_rate,
            vbv_buffer_size,
            mpeg2: false,
        })
    }
}

/// 图像编码类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureType {
    I,
    P,
    B,
    D,
}

impl PictureType {
    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            1 => Self::I,
            2 => Self::P,
            3 => Self::B,
            4 => Self::D,
            _ => return None,
        })
    }

    /// ffprobe 用的单字母表示。
    pub fn letter(self) -> char {
        match self {
            Self::I => 'I',
            Self::P => 'P',
            Self::B => 'B',
            Self::D => 'D',
        }
    }
}

/// picture header 里我们关心的字段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PictureHeader {
    pub temporal_reference: u32,
    pub picture_type: PictureType,
    pub vbv_delay: u32,
    /// 仅 P/B 帧：前向运动矢量精度（0 = 半像素）。
    pub full_pel_forward: bool,
    pub forward_f_code: u32,
    /// 仅 B 帧：后向运动矢量精度与 f_code。
    pub full_pel_backward: bool,
    pub backward_f_code: u32,
}

impl PictureHeader {
    /// 解析 `00 00 01 00` **之后**的字节。
    pub fn parse(payload: &[u8]) -> Option<Self> {
        let mut r = BitReader::new(payload);
        let temporal_reference = r.u(10).ok()?;
        let picture_type = PictureType::from_code(r.u(3).ok()? as u8)?;
        let vbv_delay = r.u(16).ok()?;
        let (full_pel_forward, forward_f_code) = if matches!(picture_type, PictureType::P | PictureType::B)
        {
            (r.u(1).ok()? == 1, r.u(3).ok()?)
        } else {
            (false, 0)
        };
        let (full_pel_backward, backward_f_code) = if matches!(picture_type, PictureType::B) {
            (r.u(1).ok()? == 1, r.u(3).ok()?)
        } else {
            (false, 0)
        };
        Some(Self {
            temporal_reference,
            picture_type,
            vbv_delay,
            full_pel_forward,
            forward_f_code,
            full_pel_backward,
            backward_f_code,
        })
    }
}

/// 码流是不是 MPEG-2。
///
/// MPEG-1 和 MPEG-2 的 sequence header 布局相同，**没有版本位**；区别在于
/// MPEG-2 紧跟着一个 sequence extension（`00 00 01 B5`）。
pub fn has_sequence_extension(es: &[u8]) -> bool {
    split_units(es)
        .iter()
        .any(|(code, _)| *code == 0xB5)
}

/// 码流里的一个单元。
#[derive(Debug, Clone)]
pub enum Unit<'a> {
    Sequence(SequenceHeader),
    Picture(PictureHeader, &'a [u8]),
    SequenceEnd,
}

/// 按 start code 切分 elementary stream。
///
/// 返回 `(code, payload)`，`payload` 是该 start code 之后、下一个 start code
/// 之前的字节。**不解析 slice 内部**——那是宏块层的事。
pub fn split_units(es: &[u8]) -> Vec<(u8, &[u8])> {
    let mut out = Vec::new();
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 < es.len() {
        if es[i] == 0 && es[i + 1] == 0 && es[i + 2] == 1 {
            starts.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    for (n, &start) in starts.iter().enumerate() {
        let code = es[start + 3];
        let body_start = start + 4;
        let body_end = starts.get(n + 1).copied().unwrap_or(es.len());
        out.push((code, &es[body_start..body_end]));
    }
    out
}

/// 把码流切成图像，附带每帧的头。
///
/// 一个图像的范围从它的 picture start code 到下一个 picture start code
/// （或 sequence end）。slice 和宏块都在这个范围里。
pub fn split_pictures(es: &[u8]) -> Vec<(PictureHeader, &[u8])> {
    let units = split_units(es);
    let mut out = Vec::new();
    for (i, (code, payload)) in units.iter().enumerate() {
        if *code != 0x00 {
            continue;
        }
        let Some(header) = PictureHeader::parse(payload) else {
            continue;
        };
        // 这一帧的字节范围：从本 start code 起，到下一个 picture start code。
        let begin = payload.as_ptr() as usize - es.as_ptr() as usize - 4;
        let end = units
            .iter()
            .skip(i + 1)
            .find(|(code, _)| *code == 0x00)
            .map(|(_, p)| p.as_ptr() as usize - es.as_ptr() as usize - 4)
            .unwrap_or(es.len());
        out.push((header, &es[begin..end]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bits_msb_first() {
        let mut r = BitReader::new(&[0b1010_1100, 0b0011_0000]);
        assert_eq!(r.u(1).ok(), Some(1));
        assert_eq!(r.u(3).ok(), Some(0b010));
        assert_eq!(r.u(4).ok(), Some(0b1100));
        assert_eq!(r.u(4).ok(), Some(0b0011));
    }

    #[test]
    fn truncated_read_returns_none_instead_of_panicking() {
        let mut r = BitReader::new(&[0xFF]);
        assert_eq!(r.u(8).ok(), Some(0xFF));
        assert!(r.u(1).is_err());
    }

    #[test]
    fn reads_a_sequence_header() {
        // 320x240, aspect 1:1, 25 fps, bitrate 0, vbv 0
        let mut r = BitWriter::new();
        r.write(320, 12);
        r.write(240, 12);
        r.write(1, 4);
        r.write(3, 4);
        r.write(0, 18);
        r.write(1, 1);
        r.write(0, 10);
        r.write(0, 1);
        r.write(1, 1);
        let bytes = r.finish();

        let header = SequenceHeader::parse(&bytes).expect("a header");
        assert_eq!(header.width, 320);
        assert_eq!(header.height, 240);
        assert_eq!(header.aspect, (1, 1));
        assert_eq!(header.frame_rate_milli, 25_000);
    }

    #[test]
    fn reads_a_picture_header() {
        let mut r = BitWriter::new();
        r.write(7, 10); // temporal_reference
        r.write(1, 3); // I
        r.write(0xFFFF, 16); // vbv_delay
        let bytes = r.finish();

        let header = PictureHeader::parse(&bytes).expect("a header");
        assert_eq!(header.temporal_reference, 7);
        assert_eq!(header.picture_type, PictureType::I);
    }

    #[test]
    fn splits_pictures_by_their_start_codes() {
        let mut es = Vec::new();
        for (tr, kind) in [(0u32, 1u32), (1, 2), (2, 3)] {
            es.extend_from_slice(&[0, 0, 1, 0x00]);
            let mut w = BitWriter::new();
            w.write(tr, 10);
            w.write(kind, 3);
            w.write(0, 16);
            es.extend_from_slice(&w.finish());
            es.extend_from_slice(&[0xFF; 8]); // 假的 slice 数据
        }
        let pictures = split_pictures(&es);
        assert_eq!(pictures.len(), 3);
        assert_eq!(pictures[0].0.picture_type, PictureType::I);
        assert_eq!(pictures[1].0.picture_type, PictureType::P);
        assert_eq!(pictures[2].0.picture_type, PictureType::B);
    }

    /// 测试用的按位写入器。
    struct BitWriter {
        bytes: Vec<u8>,
        bit: usize,
    }

    impl BitWriter {
        fn new() -> Self {
            Self { bytes: Vec::new(), bit: 0 }
        }

        fn write(&mut self, value: u32, count: u32) {
            for i in (0..count).rev() {
                if self.bit % 8 == 0 {
                    self.bytes.push(0);
                }
                let bit = ((value >> i) & 1) as u8;
                let last = self.bytes.len() - 1;
                self.bytes[last] |= bit << (7 - (self.bit % 8));
                self.bit += 1;
            }
        }

        fn finish(self) -> Vec<u8> {
            self.bytes
        }
    }
}

pub mod decode;
pub mod tables;

#[cfg(test)]
mod table_tests {
    use super::tables::{
        EOB_BITS, EOB_CODE, ESCAPE_BITS, ESCAPE_CODE, MPEG1_RL, B_MB_TYPE, MV_CODE, P_MB_TYPE,
    };

    /// 前缀无关：没有一条码字是另一条的前缀。
    #[test]
    fn the_code_is_prefix_free() {
        for (i, a) in MPEG1_RL.iter().enumerate() {
            for (j, b) in MPEG1_RL.iter().enumerate() {
                if i == j || a.bits > b.bits {
                    continue;
                }
                let shift = b.bits - a.bits;
                assert_ne!(
                    b.code >> shift,
                    a.code,
                    "entry {i} ({:0width$b}/{} bits) is a prefix of entry {j}",
                    a.code,
                    a.bits,
                    width = a.bits as usize
                );
            }
        }
    }

    /// Kraft 和必须**不超过** 1，且恰好少一个 12 位码字。
    ///
    /// MPEG-1 的 TCOEFF 码表是**不完整**前缀码：有一个 12 位码字没有分配。
    /// 所以这里不能断言「等于 1」—— 那会把正确的表判成错的。真正要防的是
    /// **码字重复**（Kraft 和 > 1）和**多出来的洞**（抄漏了条目）。
    #[test]
    fn the_code_has_exactly_one_reserved_slot() {
        let mut sum = 0f64;
        for entry in MPEG1_RL.iter() {
            sum += 2f64.powi(-(entry.bits as i32));
        }
        sum += 2f64.powi(-(EOB_BITS as i32));
        sum += 2f64.powi(-(ESCAPE_BITS as i32));

        // 表里只有 run/level 码字（111 条），EOB 和逃逸码要额外加进来。
        assert!(sum <= 1.0 + 1e-12, "Kraft sum is {sum}: codewords overlap");
        let hole = 1.0 - sum;
        assert!(
            (hole - 2f64.powi(-12)).abs() < 1e-12,
            "one 12-bit slot should be unassigned, but the hole is {hole}"
        );
    }

    /// EOB 和逃逸码不能和任何 run/level 码字撞车。
    #[test]
    fn eob_and_escape_do_not_collide() {
        for entry in MPEG1_RL.iter() {
            assert!(
                !(entry.bits >= EOB_BITS && entry.code >> (entry.bits - EOB_BITS) == EOB_CODE),
                "an entry starts with the EOB code"
            );
            assert!(
                !(entry.bits >= ESCAPE_BITS && entry.code >> (entry.bits - ESCAPE_BITS) == ESCAPE_CODE),
                "an entry starts with the escape code"
            );
        }
    }

    /// 运动矢量码必须是前缀码 —— 否则「哪个码先匹配」会随表序漂。
    #[test]
    fn mv_code_is_prefix_free() {
        for (i, a) in MV_CODE.iter().enumerate() {
            for (j, b) in MV_CODE.iter().enumerate() {
                if i == j || a.bits > b.bits {
                    continue;
                }
                assert_ne!(b.code >> (b.bits - a.bits), a.code, "mv {i} is a prefix of {j}");
            }
        }
    }

    /// P 帧 `macroblock_type` 也必须是前缀码。
    #[test]
    fn p_mb_type_is_prefix_free() {
        for (i, a) in P_MB_TYPE.iter().enumerate() {
            for (j, b) in P_MB_TYPE.iter().enumerate() {
                if i == j || a.bits > b.bits {
                    continue;
                }
                assert_ne!(
                    b.code >> (b.bits - a.bits),
                    a.code,
                    "p mb_type {i} is a prefix of {j}"
                );
            }
        }
    }

    /// B 帧 `macroblock_type` 同样是前缀码。
    #[test]
    fn b_mb_type_is_prefix_free() {
        for (i, a) in B_MB_TYPE.iter().enumerate() {
            for (j, b) in B_MB_TYPE.iter().enumerate() {
                if i == j || a.bits > b.bits {
                    continue;
                }
                assert_ne!(
                    b.code >> (b.bits - a.bits),
                    a.code,
                    "b mb_type {i} is a prefix of {j}"
                );
            }
        }
    }
}

#[cfg(test)]
mod zigzag_tests {
    /// zigzag 必须是 0..63 的一个置换 —— 漏一个或重一个都会让块内系数错位。
    #[test]
    fn zigzag_is_a_permutation() {
        let mut seen = [false; 64];
        for value in super::decode::ZIGZAG {
            assert!(value < 64, "index out of range: {value}");
            assert!(!seen[value], "duplicate index {value}");
            seen[value] = true;
        }
        assert!(seen.iter().all(|&v| v));
    }
}
