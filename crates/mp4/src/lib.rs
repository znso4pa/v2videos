//! 纯 Rust 的 **MP4 / ISOBMFF 封装**。
//!
//! 链路里的最后一段（封装）。目标与另外几个容器 crate 一致：
//! **写出来的文件交给 ffprobe / ffmpeg 逐项核对**，不是「能打开就算过」。
//!
//! ## 为什么需要它
//!
//! 想把 `.wmv` 转成 `.mp4`，第一反应是「换个壳」。**但这条路不存在**：
//!
//! ```text
//! $ ffmpeg -i in.asf -c copy out.mp4
//! [mp4 @ ...] Could not find tag for codec msmpeg4v3 in stream #0,
//!             codec not currently supported in container
//! ```
//!
//! MP43（MS-MPEG-4 v3）**不是** MP4 认的编码 —— 视频必须重编码。
//! 而音频那边 WMA 同样进不了 MP4。所以 `.wmv → .mp4` 是「解码 → 重编码 → 重新封装」，
//! 三段都省不掉。
//!
//! ## 现在支持什么
//!
//! - **`mp4v`**（MPEG-4 Part 2）视频轨 —— 对应的编码器是下一步。
//! - **`mp4a`**（AAC）音频轨。
//! - 非分片 MP4（`ftyp` / `mdat` / `moov`），`moov` 在尾部。
//!
//! 零依赖、纯 std、无 FFI。
//!
//! ## 还没做
//!
//! 编辑列表（`edts` / `elst`，音频延迟补偿）、分片 MP4、`co64`（> 4 GiB）、
//! H.265 / AV1 之类的 sample entry、`udta` 标记。

mod mux;
mod writer;

pub use mux::{AudioTrack, Mp4Muxer, VideoTrack};

/// MP4 封装里的错误。
#[derive(Debug)]
pub enum Error {
    /// 参数或数据不合法。
    Bad(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Bad(s) => write!(f, "MP4 数据不合法：{s}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
