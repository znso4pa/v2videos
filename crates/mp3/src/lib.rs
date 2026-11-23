//! MPEG Audio Layer III. 输入是容器交出的连续码流，可在任意字节边界分块。
//! PCM 为逐声道 f32，刻度同 i16。文件标签、无缝裁剪和 seek 预滚由调用方处理。

#![forbid(unsafe_code)]

pub use v2v_kernel::{Error, Result};

#[cfg(feature = "decode")]
mod decode;
#[cfg(feature = "decode")]
mod synth;
#[cfg(feature = "decode")]
pub use decode::Decoder;

pub type Pcm = Vec<Vec<f32>>;
pub const FORMAT_TAG: u16 = 0x0055;

#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub channels: usize,
    pub sample_rate: u32,
    pub frame_len: usize,
    pub samples: usize,
    pub side_len: usize,
    pub crc_len: usize,
    pub rate_index: usize,
    pub mpeg1: bool,
    /// Joint stereo 的扩展位，普通 stereo / mono 为 0。
    pub stereo: u8,
}

fn unsupported(reason: &str) -> Error {
    Error::Unsupported(format!("MP3: {reason}"))
}

impl Header {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let h = u32::from_be_bytes(bytes.get(..4).ok_or(Error::Eof)?.try_into().unwrap());
        if h >> 21 != 0x7ff {
            return Err(Error::Bad("MP3 frame sync"));
        }
        if h >> 17 & 3 != 1 {
            return Err(unsupported("只支持 MPEG Audio Layer III"));
        }
        let version = match (h >> 19) & 3 {
            3 => 0,
            2 => 1,
            0 => 2,
            _ => return Err(Error::Bad("MP3 version")),
        };
        let mpeg1 = version == 0;
        let rate = (h >> 10 & 3) as usize;
        let sample_rate = [44100, 48000, 32000]
            .get(rate)
            .ok_or(Error::Bad("MP3 sample rate"))?
            >> version;
        const BITRATES: [[u32; 14]; 2] = [
            [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
            [
                32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
            ],
        ];
        let bitrate_index = (h >> 12 & 15) as usize;
        if bitrate_index == 0 {
            return Err(unsupported("free format"));
        }
        let bitrate = *BITRATES[usize::from(mpeg1)]
            .get(bitrate_index - 1)
            .ok_or(Error::Bad("MP3 bitrate"))?;
        let mode = h >> 6 & 3;
        let channels = if mode == 3 { 1 } else { 2 };
        let side_len = match (mpeg1, channels) {
            (true, 1) => 17,
            (true, _) => 32,
            (false, 1) => 9,
            (false, _) => 17,
        };
        let crc_len = if h >> 16 & 1 == 0 { 2 } else { 0 };
        let samples = if mpeg1 { 1152 } else { 576 };
        let frame_len = (samples as u32 * 125 * bitrate / sample_rate + (h >> 9 & 1)) as usize;
        if frame_len < 4 + crc_len + side_len {
            return Err(Error::Bad("MP3 side info exceeds frame"));
        }
        Ok(Self {
            channels,
            sample_rate,
            frame_len,
            samples,
            side_len,
            crc_len,
            rate_index: version * 3 + rate,
            mpeg1,
            stereo: if mode == 1 { (h >> 4 & 3) as u8 } else { 0 },
        })
    }
}

/// 容器载荷可能包含部分帧或多帧。流参数固定，遇到变化会返回 Unsupported。
pub struct Framer {
    pending: Vec<u8>,
    channels: usize,
    sample_rate: u32,
}

impl Framer {
    pub fn new(channels: usize, sample_rate: u32) -> Result<Self> {
        if !matches!(channels, 1 | 2)
            || !matches!(
                sample_rate,
                8000 | 11025 | 12000 | 16000 | 22050 | 24000 | 32000 | 44100 | 48000
            )
        {
            return Err(unsupported("声道数或采样率"));
        }
        Ok(Self {
            pending: Vec::new(),
            channels,
            sample_rate,
        })
    }

    pub fn feed(
        &mut self,
        mut data: &[u8],
        mut decode: impl FnMut(Header, &[u8]) -> Result<()>,
    ) -> Result<()> {
        while !data.is_empty() {
            if self.pending.is_empty() && data.len() >= 4 {
                let header = self.header(data)?;
                if let Some(frame) = data.get(..header.frame_len) {
                    decode(header, frame)?;
                    data = &data[header.frame_len..];
                    continue;
                }
            }
            if self.pending.len() < 4 {
                let take = data.len().min(4 - self.pending.len());
                self.pending.extend_from_slice(&data[..take]);
                data = &data[take..];
                if self.pending.len() < 4 {
                    break;
                }
            }
            let header = self.header(&self.pending)?;
            let take = data.len().min(header.frame_len - self.pending.len());
            self.pending.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.pending.len() == header.frame_len {
                let result = decode(header, &self.pending);
                self.pending.clear();
                result?;
            }
        }
        Ok(())
    }

    fn header(&self, data: &[u8]) -> Result<Header> {
        let h = Header::parse(data)?;
        if h.channels != self.channels || h.sample_rate != self.sample_rate {
            return Err(unsupported("帧的声道数或采样率与流声明不符"));
        }
        Ok(h)
    }

    /// 在真正的 EOF 调用，检测截断帧。
    pub fn finish(&self) -> Result<()> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            Err(Error::Eof)
        }
    }
}
