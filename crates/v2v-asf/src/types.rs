//! ASF 解封装器的数据模型。
//!
//! 容器无关的部分（`Stream` / `Sample` / 格式 / `StreamKind` / `fourcc_string`）
//! 在 `v2v-container` 里，与 AVI / MP4 共用；这里只留 **ASF 特有的**东西。

pub use v2v_container::{
    fourcc_string, AudioFormat, Sample, Stream, StreamKind, VideoFormat,
};

/// ASF File Properties Object 里的内容 + Data Object 的包数。
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

/// 扩展流属性里声明的 payload 扩展（真样本里通常是 0 个）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct PayloadExt {
    pub type_byte: u8,
    pub size: u16,
}
