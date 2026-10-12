//! 解封装器的数据模型。

/// ASF 的流类型（由 Stream Properties Object 里的 Stream Type GUID 决定）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamKind {
    Audio,
    Video,
    Command,
    Jfif,
    Unknown,
}

/// 音频流格式（Type-Specific Data 是一个 WAVEFORMATEX）。
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

/// 视频流格式（Type-Specific Data 是 11 字节前缀 + BITMAPINFOHEADER）。
#[derive(Debug, Clone, Default)]
pub struct VideoFormat {
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
    /// ASF 里的流编号（1 起），数据包的 payload flags 低 7 位就是它。
    pub number: u8,
    pub kind: StreamKind,
    pub time_offset_ms: u64,
    pub audio: Option<AudioFormat>,
    pub video: Option<VideoFormat>,
}

/// File Properties Object 里的内容 + Data Object 的包数。
#[derive(Debug, Clone, Default)]
pub struct FileInfo {
    pub file_size: u64,
    pub creation_time: u64,
    pub packet_count: u64,
    pub play_duration_100ns: u64,
    pub send_duration_100ns: u64,
    pub preroll_ms: u32,
    /// bit0 = Broadcast Flag。
    pub flags: u32,
    pub min_packet_size: u32,
    pub max_packet_size: u32,
    pub max_bitrate: u32,
}

impl FileInfo {
    /// 播放时长（毫秒），**已减掉 preroll** —— 与 `ffprobe` 报的 duration 同口径。
    /// 原始字段在 `play_duration_100ns`。
    pub fn duration_ms(&self) -> u64 {
        (self.play_duration_100ns / 10_000).saturating_sub(self.preroll_ms as u64)
    }

    pub fn is_broadcast(&self) -> bool {
        self.flags & 0x01 != 0
    }
}

/// 一条完整的、已重组的样本（视频一帧 / 音频一帧）。
#[derive(Debug, Clone)]
pub struct Sample {
    pub stream: u8,
    pub key_frame: bool,
    /// 毫秒，已减掉 preroll。
    pub timestamp_ms: i64,
    pub data: Vec<u8>,
}

/// 扩展流属性里声明的 payload 扩展（真样本里通常是 0 个）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct PayloadExt {
    pub type_byte: u8,
    pub size: u16,
}

/// 把 FourCC 渲染成可读字符串（不可打印字符用 `.`）。
pub fn fourcc_string(v: u32) -> String {
    let b = v.to_le_bytes();
    b.iter()
        .map(|&c| if (0x20..0x7f).contains(&c) { c as char } else { '.' })
        .collect()
}
