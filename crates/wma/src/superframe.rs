//! superframe 头解析 —— 对着 `wma_decode_superframe` 的比特池（bit reservoir）逻辑写。
//!
//! ## 结构（一包 = 一个 ASF 音频载荷）
//!
//! ```text
//! 不用比特池：整包就是一个帧。
//! 用比特池：
//!   4 位  superframe 序号（每个包 +1，8 回绕）
//!   4 位  本包的帧数（还要减去「上一包有没有残留」那一位，见下）
//!   (byte_offset_bits + 3) 位  bit_offset —— 本包**第一个完整帧**从这里开始
//!   ……帧数据……
//! ```
//!
//! ## 跨包残留（`last_superframe`）
//!
//! 包尾往往切在一帧中间。ffmpeg 把「没用完的那些位」存进 `last_superframe`，
//! 下一包再把开头 `bit_offset` 位接上去，凑成一整帧解出来。
//!
//! ⚠️ **残留长度取决于帧内实际消耗了多少位**，所以「精确帧数」只有帧解码器
//! 落地之后才能算 —— 本模块给的是**结构性**校验（起始位置在缓冲区内、
//! 帧数在合理区间、序号连得上）。

use crate::bitreader::BitReader;
use crate::{Error, Result, WmaParams};

/// 一个 superframe 的头。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SuperframeHeader {
    /// 4 位序号（`skip_bits(4)` 那个，每包 +1、8 回绕）。
    pub index: u32,
    /// 本包**总共**产出多少帧（含「接着上一包解完的那半帧」）。
    ///
    /// 对应 ffmpeg 的 `nb_frames = get_bits(4) - (last_superframe_len <= 0)`
    /// —— 那个 `-1` 是因为「有残留时，本包产出的第一帧其实是上一包那半帧」，
    /// 所以总产出并不减。本包**新起**的帧从 `frames_start_bit` 开始。
    pub nb_frames: i32,
    /// 本包第一个完整帧的起始位偏移。
    pub bit_offset: u32,
    /// 第一个完整帧的起始位（相对包首）：`bit_offset + 4 + 4 + byte_offset_bits + 3`。
    pub frames_start_bit: usize,
}

/// 跨包的状态（`last_superframe` 的搬运）。
///
/// ⚠️ ffmpeg 每解完一包都会 `last_superframe_len = buf_size - pos` —— 只要这一包
/// 解出了帧，残留就**非零**（帧边界切在包中间）。残留的**长度**取决于帧内实际
/// 消耗了多少位（要等帧解码器），但「有没有残留」这个布尔量现在就定得下来，
/// 而 superframe 头里那个 `-1` 用的正是它。
#[derive(Debug, Default, Clone)]
pub struct SuperframeState {
    /// 有没有「上一包留下、这一包要接着解」的半帧。
    pub residue: bool,
    /// 累积模式攒下的位数（只在 `nb_frames <= 0` 的攒位路径里增长）。
    pub accumulated_bits: usize,
}

impl SuperframeState {
    /// 解一包的头。
    ///
    /// `data` 是一个 ASF 音频载荷（长度应为 `block_align`）。
    /// 返回 `None` = 这一包只是「往残留缓冲里攒位」，不产帧。
    pub fn parse(&mut self, params: &WmaParams, data: &[u8]) -> Result<Option<SuperframeHeader>> {
        if !params.use_bit_reservoir {
            // 不用比特池：整包一个帧，没有头。
            return Ok(Some(SuperframeHeader {
                index: 0,
                nb_frames: 1,
                bit_offset: 0,
                frames_start_bit: 0,
            }));
        }

        let mut r = BitReader::new(data);
        let index = r.get(4)?;
        let raw = r.get(4)?;
        // `nb_frames = get_bits(4) - (last_superframe_len <= 0)`
        let nb_frames = raw as i32 - i32::from(!self.residue);
        if nb_frames <= 0 {
            // 攒位：ffmpeg 把 `buf_size - 1` 字节接进 last_superframe
            self.accumulated_bits += 8 * data.len() - 8;
            self.residue = true;
            return Ok(None);
        }

        let bit_offset = r.get(params.byte_offset_bits + 3)?;
        let frames_start_bit =
            bit_offset as usize + 4 + 4 + params.byte_offset_bits as usize + 3;
        if frames_start_bit > data.len() * 8 {
            return Err(Error::Bad("bit_offset 超出包长"));
        }

        // 有残留时，本包开头那 `bit_offset` 位属于「上一包那一帧」—— 先把它解掉，
        // 再解 `nb_frames - 1` 个新帧，总数还是 `nb_frames`。解完之后本包尾部
        // 又会留下新的半帧（帧边界切在包中间）。
        self.residue = true;
        self.accumulated_bits = 0;
        Ok(Some(SuperframeHeader {
            index,
            nb_frames,
            bit_offset,
            frames_start_bit,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> WmaParams {
        // 22.05 kHz 立体声 22 kbps：比特池开、byte_offset_bits = 8
        WmaParams::new(2, 2, 22050, 22008, 894, &[0, 0, 0, 0, 0x0e, 0, 0, 0, 0, 0]).unwrap()
    }

    #[test]
    fn first_packet_has_no_residue_so_one_fewer_frame() {
        // 序号 0、帧数 7、bit_offset 0（后续 8 位）
        let data = [0b0000_0111, 0b0000_0000, 0, 0];
        let mut st = SuperframeState::default();
        let h = st.parse(&params(), &data).unwrap().expect("有帧");
        assert_eq!(h.index, 0);
        assert_eq!(h.nb_frames, 6, "第一包没有残留 → 减 1");
        assert_eq!(h.bit_offset, 0);
        assert_eq!(h.frames_start_bit, 19);
    }

    #[test]
    fn later_packets_carry_a_residue_frame() {
        let data = [0b0000_0111, 0b0000_0000, 0, 0];
        let mut st = SuperframeState::default();
        let _ = st.parse(&params(), &data).unwrap(); // 第一包：6 帧
        let h = st.parse(&params(), &data).unwrap().expect("有帧");
        // 上一包留了半帧 → 本包的总产出不再减 1（那半帧算在本包头上）
        assert_eq!(h.nb_frames, 7);
        assert!(st.residue);
    }

    #[test]
    fn accumulation_packet_yields_no_frames() {
        // 序号 1、帧数 1，但没有残留 → nb_frames = 0 → 这一包只攒位
        let data = [0b0001_0001, 0, 0, 0];
        let mut st = SuperframeState::default();
        assert!(st.parse(&params(), &data).unwrap().is_none());
        assert!(st.accumulated_bits > 0);
    }
}
