//! MS-MPEG-4（MP41/MP42/MP43 / WMV1-2）的帧头解析。
//!
//! MP43（MS-MPEG-4 v3）**没有起始码** —— 每帧直接从位流开头读：
//!
//! ```text
//!   picture_type : 2 bit  (+1，所以 1 = I，2 = P)
//!   qscale       : 5 bit  (1..31)
//!   ── I 帧 ──
//!     slice_code   : 5 bit  (0x17 = 一个 slice；slice_height = mb_height / (code-0x16))
//!     rl_chroma    : decode012
//!     rl_luma      : decode012
//!     dc_table     : 1 bit
//!   ── P 帧 ──
//!     skip_mb_code : 1 bit
//!     rl_luma      : decode012   (rl_chroma = rl_luma)
//!     dc_table     : 1 bit
//!     mv_table     : 1 bit
//! ```
//!
//! 字段顺序按 MS-MPEG-4 v3 的帧头定义；行为与 ffmpeg 交叉验证（见 `tests/`）。

use crate::bitreader::BitReader;
use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureType {
    I,
    P,
}

#[derive(Debug, Clone)]
pub struct FrameHeader {
    pub picture_type: PictureType,
    /// 1..=31。
    pub qscale: u8,
    /// I 帧：一个 slice 覆盖多少宏块行（其余类型为 1）。
    pub slice_height: u32,
    pub rl_table_index: u8,
    pub rl_chroma_table_index: u8,
    pub dc_table_index: u8,
    pub mv_table_index: u8,
    pub use_skip_mb_code: bool,
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
) -> Result<(FrameHeader, usize)> {
    let mut r = BitReader::new(data);

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
    };

    match picture_type {
        PictureType::I => {
            let code = r.u(5)?;
            if code < 0x17 {
                return Err(Error::Bad("slice code < 0x17"));
            }
            // 0x17 = 一个 slice；0x18 = 两个 slice…
            h.slice_height = (mb_height / (code - 0x16)).max(1);
            h.rl_chroma_table_index = decode012(&mut r)?;
            h.rl_table_index = decode012(&mut r)?;
            h.dc_table_index = r.bit()? as u8;
        }
        PictureType::P => {
            // P 帧头**不设** slice_height —— 它保持初始化时的 `mb_height`，整帧一个 slice。
            // 这决定 `first_slice_line`：只有第 0 行是，于是 DC 预测的边界与 MV 预测的
            // 「首行只用左邻」都靠它。默认成 1 会让每一行都被当成 slice 首行。
            h.slice_height = mb_height;
            h.use_skip_mb_code = r.bit()? != 0;
            h.rl_table_index = decode012(&mut r)?;
            h.rl_chroma_table_index = h.rl_table_index;
            h.dc_table_index = r.bit()? as u8;
            h.mv_table_index = r.bit()? as u8;
        }
    }

    let _ = mb_width;
    Ok((h, r.bit_pos()))
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
        let (h, used) = parse_frame_header(&data, 120, 68).unwrap();
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
        let (h, used) = parse_frame_header(&data, 120, 68).unwrap();
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
            parse_frame_header(&data, 1, 1),
            Err(Error::Bad(_))
        ));
    }
}
