//! WMA 的解码参数 —— 逐条对着 `ff_wma_init` / `ff_wma_get_frame_len_bits` 写。
//!
//! 这些参数**全部**由「WAVEFORMATEX + 编解码器私有数据」推出来，不读任何码流：
//! 采样率、声道数、码率、`block_align`，加上私有数据里的 `flags2`。

use crate::{av_log2, Error, Result};

/// `BLOCK_MIN_BITS`（`wma.h`）。
const BLOCK_MIN_BITS: u32 = 7;

/// 由流参数推出的解码参数。
#[derive(Debug, Clone, PartialEq)]
pub struct WmaParams {
    pub version: u8,
    pub channels: usize,
    pub sample_rate: u32,
    pub bit_rate: u32,
    pub block_align: usize,

    /// `frame_len = 1 << frame_len_bits`（一帧的样本数）。
    pub frame_len_bits: u32,
    pub frame_len: usize,
    /// superframe 头里 `bit_offset` 字段的位数（`byte_offset_bits + 3`）。
    pub byte_offset_bits: u32,
    /// 一个帧内可以有多少种块长（`ff_wma_init` 里按码率算）。
    pub nb_block_sizes: usize,

    pub use_exp_vlc: bool,
    pub use_bit_reservoir: bool,
    pub use_variable_block_len: bool,
    /// 噪声编码（高带不传系数、只传增益）。
    pub use_noise_coding: bool,
    /// 噪声编码的起始频率。
    pub high_freq: f32,
    /// 帧内第一个要传的系数下标（v1 是 3，v2 是 0）。
    pub coefs_start: usize,
}

impl WmaParams {
    /// 从「WAVEFORMATEX 的字段 + 编解码器私有数据」推参数。
    ///
    /// `extradata` 是 ASF 里那条音频流的私有数据（`AudioFormat::extra`）。
    pub fn new(
        version: u8,
        channels: usize,
        sample_rate: u32,
        bit_rate: u32,
        block_align: usize,
        extradata: &[u8],
    ) -> Result<Self> {
        if version != 1 && version != 2 {
            return Err(Error::Unsupported(format!("WMA 版本 {version}")));
        }
        if sample_rate == 0 || sample_rate > 50000 {
            return Err(Error::Bad("采样率不在 1..=50000"));
        }
        if channels == 0 || channels > 2 {
            return Err(Error::Bad("声道数不在 1..=2"));
        }
        if bit_rate == 0 {
            return Err(Error::Bad("码率为 0"));
        }
        if block_align == 0 {
            return Err(Error::Bad("block_align 为 0"));
        }

        // ---- flags2（wma_decode_init）----
        // v1：私有数据第 2 字节起；v2：第 4 字节起。长度不够就是 0。
        let flags2: u32 = if version == 1 {
            if extradata.len() >= 4 {
                u16::from_le_bytes([extradata[2], extradata[3]]) as u32
            } else {
                0
            }
        } else if extradata.len() >= 6 {
            u16::from_le_bytes([extradata[4], extradata[5]]) as u32
        } else {
            0
        };
        let mut use_variable_block_len = flags2 & 0x0004 != 0;
        // ffmpeg 的 issue1503 特例：v2 且私有数据够长、`RL16(extradata+4) == 0xd`
        // 时把变长块**关掉**。
        if version == 2
            && extradata.len() >= 8
            && u16::from_le_bytes([extradata[4], extradata[5]]) == 0xd
            && use_variable_block_len
        {
            use_variable_block_len = false;
        }

        // ---- 帧长（ff_wma_get_frame_len_bits）----
        let frame_len_bits = if sample_rate <= 16000 {
            9
        } else if sample_rate <= 22050 || (sample_rate <= 32000 && version == 1) {
            10
        } else if sample_rate <= 48000 || version < 3 {
            11
        } else if sample_rate <= 96000 {
            12
        } else {
            13
        };
        let frame_len = 1usize << frame_len_bits;

        // ---- 块长种类（ff_wma_init）----
        let nb_block_sizes = if use_variable_block_len {
            let mut nb = ((flags2 >> 3) & 3) as usize + 1;
            if bit_rate / channels as u32 >= 32000 {
                nb += 2;
            }
            let nb_max = (frame_len_bits - BLOCK_MIN_BITS) as usize;
            if nb > nb_max {
                nb = nb_max;
            }
            nb + 1
        } else {
            1
        };

        // ---- byte_offset_bits ----
        // ⚠️ 用 f32 走一遍 ffmpeg 的算式：`(int)(bps * frame_len / 8.0 + 0.5)`
        // —— 截断 + 0.5 的取整，不能换成四舍五入的函数。
        let bps = bit_rate as f32 / (channels as f32 * sample_rate as f32);
        let byte_offset_bits = av_log2((bps * frame_len as f32 / 8.0 + 0.5) as u32) + 2;

        // ---- 噪声编码与 high_freq ----
        // v2 的码率按「归一化采样率」分档。
        let sample_rate1 = if version == 2 {
            if sample_rate >= 44100 {
                44100
            } else if sample_rate >= 22050 {
                22050
            } else if sample_rate >= 16000 {
                16000
            } else if sample_rate >= 11025 {
                11025
            } else {
                8000
            }
        } else {
            sample_rate
        };
        let mut use_noise_coding = true;
        let mut high_freq = sample_rate as f32 * 0.5;
        let bps1 = if channels == 2 { bps * 1.6 } else { bps };
        if sample_rate1 == 44100 {
            if bps1 >= 0.61 {
                use_noise_coding = false;
            } else {
                high_freq *= 0.4;
            }
        } else if sample_rate1 == 22050 {
            if bps1 >= 1.16 {
                use_noise_coding = false;
            } else if bps1 >= 0.72 {
                high_freq *= 0.7;
            } else {
                high_freq *= 0.6;
            }
        } else if sample_rate1 == 16000 {
            if bps > 0.5 {
                high_freq *= 0.5;
            } else {
                high_freq *= 0.3;
            }
        } else if sample_rate1 == 11025 {
            high_freq *= 0.7;
        } else if sample_rate1 == 8000 {
            if bps <= 0.625 {
                high_freq *= 0.5;
            } else if bps > 0.75 {
                use_noise_coding = false;
            } else {
                high_freq *= 0.65;
            }
        } else if bps >= 0.8 {
            high_freq *= 0.75;
        } else if bps >= 0.6 {
            high_freq *= 0.6;
        } else {
            high_freq *= 0.5;
        }

        Ok(Self {
            version,
            channels,
            sample_rate,
            bit_rate,
            block_align,
            frame_len_bits,
            frame_len,
            byte_offset_bits,
            nb_block_sizes,
            use_exp_vlc: flags2 & 0x0001 != 0,
            use_bit_reservoir: flags2 & 0x0002 != 0,
            use_variable_block_len,
            use_noise_coding,
            high_freq,
            coefs_start: if version == 1 { 3 } else { 0 },
        })
    }

    /// 块长字段的位数：`n = av_log2(nb_block_sizes - 1) + 1`（`wma_decode_block`）。
    pub fn block_len_bits_field(&self) -> u32 {
        av_log2(self.nb_block_sizes as u32 - 1) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_len_bits_matches_ffmpeg_table() {
        // 对着 ff_wma_get_frame_len_bits：v2 的采样率分档
        for (rate, want) in [
            (8000u32, 9u32),
            (11025, 9),
            (16000, 9),
            (22050, 10),
            (32000, 11),
            (44100, 11),
            (48000, 11),
        ] {
            let p = WmaParams::new(2, 2, rate, 64000, 2048, &[]).unwrap();
            assert_eq!(p.frame_len_bits, want, "采样率 {rate}");
        }
    }

    #[test]
    fn flags2_bits_map_to_the_right_switches() {
        // v2 的 flags2 在私有数据第 4 字节起
        let extra = [0u8, 0, 0, 0, 0x0e, 0x00, 0, 0, 0, 0];
        let p = WmaParams::new(2, 2, 22050, 22008, 894, &extra).unwrap();
        assert!(!p.use_exp_vlc, "0x0e 的 bit0 = 0");
        assert!(p.use_bit_reservoir, "bit1");
        assert!(p.use_variable_block_len, "bit2");
        assert!(p.use_noise_coding, "22 kbps 立体声 22050Hz 要开噪声编码");
        assert_eq!(p.frame_len_bits, 10);
        assert_eq!(p.frame_len, 1024);
        assert_eq!(p.byte_offset_bits, 8);
        assert_eq!(p.nb_block_sizes, 3);
        assert_eq!(p.block_len_bits_field(), 2);
        assert_eq!(p.coefs_start, 0, "v2 从第 0 个系数起");
    }
}
