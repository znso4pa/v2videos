//! 纯 Rust 的 **WMA（v1 / v2）解码**。
//!
//! 目标与视频那半边一致：把 ASF 里那条音频轨解成 PCM，**与 ffmpeg 的 PCM 逐样本对账**。
//! 无 C、无 FFI；唯一的依赖是本仓的**共享数值内核**（FFT，见 `v2v-msmpeg4`）。
//!
//! ## 进度
//!
//! - ✅ **位读取器**（MSB 优先 —— 实测确认，见下）
//! - ✅ **参数推导**（`ff_wma_init` / `ff_wma_get_frame_len_bits` 逐条对着写）
//! - ✅ **superframe 头解析**（帧数 / `bit_offset` / 跨包残留的搬运规则）
//! - ✅ **码表**（`tables.rs`，由 `tools/gen_wma_tables.py` 从 ffmpeg 的数据文件生成，
//!   带 Kraft 和自检）
//! - ✅ **帧头 → 指数解码（LSP / VLC）→ 位分配 → 系数（6 张 RL 表）→ IMDCT**
//!   （IMDCT 走 **FFT**：`2n` 个输出折成 `n` 点 DCT-IV + 三段对称，再折成一次 `2n` 点 FFT）
//! - ✅ **v2 整轨对账**：真样本整轨与 ffmpeg PCM **最大差 3**
//! - ✅ **v1 整轨对账**：真样本整轨与 ffmpeg PCM **最大差 2**
//!
//! ## 三个「实测才知道」的点
//!
//! - **位序是 MSB 优先**，不是 LSB。WMA 常被说成 LSB 优先，但 ffmpeg 的
//!   `wmadec.c` 用的是普通 `init_get_bits` / `get_bits`（MSB）。拿真样本的
//!   第一个载荷试两种读法：MSB 读出来是一个序号与帧数都合理的头，LSB 读出来
//!   则两者都明显是垃圾。
//! - **不用比特池时，一个 ASF 载荷不是一帧**。`wma_decode_superframe` 里有
//!   `if (avctx->block_align) buf_size = avctx->block_align;`，然后 `return buf_size`
//!   —— 通用解码层（`decode.c` 的 `decode_simple_internal`）认这个返回值，
//!   把剩余字节留在 `avci->in_pkt` 里**接着调同一个解码器**。等价于
//!   「每 `block_align` 字节解一帧」—— 一个载荷出「载荷长度 ÷ `block_align`」帧。
//!   这一条曾让帧数只算出 1/3，是 v1 卡最久的原因（详见 `decoder.rs::decode_superframe`）。
//! - **开了比特池时，精确帧数只有帧解码器才能定**：superframe 末尾会把
//!   「这一包没用完的位」存进 `last_superframe` 留给下一包接着解
//!   （`last_superframe_len`），而残留长度取决于帧内实际消耗了多少位。

mod bitreader;
mod decoder;
mod params;
mod superframe;
pub mod tables;

pub use bitreader::BitReader;
pub use decoder::WmaDecoder;
pub use params::WmaParams;
pub use superframe::{SuperframeHeader, SuperframeState};

/// WMA 解码的错误类型。
#[derive(Debug)]
pub enum Error {
    /// 数据不合法（超出范围、越界读等）。
    Bad(&'static str),
    /// 合法但本实现还不支持。
    Unsupported(String),
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Bad(s) => write!(f, "WMA 数据不合法：{s}"),
            Error::Unsupported(s) => write!(f, "WMA 不支持：{s}"),
            Error::Io(e) => write!(f, "IO 错误：{e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// ffmpeg 的 `av_log2`：`x <= 1` 时返回 0，否则 `floor(log2(x))`。
pub(crate) fn av_log2(x: u32) -> u32 {
    if x <= 1 {
        0
    } else {
        31 - x.leading_zeros()
    }
}
