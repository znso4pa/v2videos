//! v2video 各**容器解封装器共用的数据模型**。
//!
//! 存在的理由：ASF / AVI / MP4 这些容器的字节布局完全不同，但「拆出来的东西」
//! 是同一种东西 —— **一条流 + 一串带时间戳与关键帧标志的裸样本**。
//! 把这层抽出来，上层（解码器、转码工具）就不必知道样本是从哪个容器来的，
//! 也不必为每个容器写一份类型转换。
//!
//! ```text
//!   .wmv / .avi / .mp4 ──► [各容器 crate] ──► Sample { stream, key_frame, timestamp_ms, data }
//! ```
//!
//! 零依赖，纯 std。

/// 流的类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamKind {
    Audio,
    Video,
    /// 命令流（ASF 有；别的容器一般没有）。
    Command,
    /// JFIF 静止图（ASF 有）。
    Jfif,
    Unknown,
}

/// 音频流格式（ASF 的 Type-Specific Data / AVI 的 `strf` 都是 WAVEFORMATEX）。
#[derive(Debug, Clone, Default)]
pub struct AudioFormat {
    /// `wFormatTag`。0x0161 = WMA v2，0x0160 = WMA v1，0x0162 = WMA Pro，0x0001 = PCM，0x0055 = MP3…
    pub format_tag: u16,
    pub channels: u16,
    pub sample_rate: u32,
    pub avg_bytes_per_sec: u32,
    pub block_align: u16,
    pub bits_per_sample: u16,
    /// `cbSize` 之后的编解码器私有数据。
    pub extra: Vec<u8>,
}

/// 视频流格式（ASF 的 Type-Specific Data / AVI 的 `strf` 都是 BITMAPINFOHEADER）。
#[derive(Debug, Clone, Default)]
pub struct VideoFormat {
    /// 编码器声明里带的宽高（可能与真实显示尺寸不同，见 `width` / `height`）。
    pub encoded_width: u32,
    pub encoded_height: u32,
    pub width: u32,
    pub height: u32,
    pub bits_per_pixel: u16,
    /// `biCompression`，也就是 FourCC（如 `MP43` / `WMV3`）。
    pub fourcc: u32,
    /// BITMAPINFOHEADER 之后的编解码器私有数据。
    pub extra: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Stream {
    /// 容器内的流编号。ASF 里是 1 起的流号（数据包 payload flags 的低 7 位）；
    /// AVI 里是 `strl` 的顺序（0 起）—— 各容器自己的编号体系，**只在同一容器内可比**。
    pub number: u8,
    pub kind: StreamKind,
    pub time_offset_ms: u64,
    pub audio: Option<AudioFormat>,
    pub video: Option<VideoFormat>,
}

/// 一条完整的、已重组的样本（视频一帧 / 音频一帧）。
#[derive(Debug, Clone)]
pub struct Sample {
    /// 与 `Stream::number` 对应。
    pub stream: u8,
    pub key_frame: bool,
    /// 毫秒。ASF 已减掉 preroll；AVI 从 0 起。
    pub timestamp_ms: i64,
    pub data: Vec<u8>,
}

/// 把 FourCC 渲染成可读字符串（不可打印字符用 `.`）。
pub fn fourcc_string(v: u32) -> String {
    let b = v.to_le_bytes();
    b.iter()
        .map(|&c| if (0x20..0x7f).contains(&c) { c as char } else { '.' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourcc_is_readable_and_never_panics() {
        assert_eq!(fourcc_string(u32::from_le_bytes(*b"MP43")), "MP43");
        assert_eq!(fourcc_string(0), "....");
        assert_eq!(fourcc_string(u32::MAX).chars().count(), 4);
    }
}
