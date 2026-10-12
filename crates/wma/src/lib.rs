//! 纯 Rust 的 **WMA（v1 / v2）解码**。
//!
//! 目标与视频那半边一致：把 ASF 里那条音频轨解成 PCM，**与 ffmpeg 的 PCM 逐样本对账**。
//! 无 C、无 FFI、零依赖。
//!
//! ## 进度（一段一段啃，每段都要真样本对账）
//!
//! - ✅ **位读取器**（MSB 优先 —— 实测确认，见下）
//! - ✅ **参数推导**（`ff_wma_init` / `ff_wma_get_frame_len_bits` 逐条对着写）
//! - ✅ **superframe 头解析**（帧数 / `bit_offset` / 跨包残留的搬运规则）
//! - ⏳ 帧头 → 指数解码（LSP / VLC）→ 位分配 → 系数（16 张 RL 表）→ IMDCT
//!
//! ## 两个「实测才知道」的点
//!
//! - **位序是 MSB 优先**，不是 LSB。WMA 常被说成 LSB 优先，但 ffmpeg 的
//!   `wmadec.c` 用的是普通 `init_get_bits` / `get_bits`（MSB）。拿真样本的
//!   第一个载荷试两种读法：MSB 读出「superframe 序号 0、帧数 7」，LSB 读出
//!   「序号 7、帧数 0」—— 后者明显是垃圾。
//! - **精确帧数只有帧解码器才能定**：superframe 末尾会把「这一包没用完的位」
//!   存进 `last_superframe` 留给下一包接着解（`last_superframe_len`），
//!   而残留长度取决于帧内实际消耗了多少位。所以本模块只能给出**结构性**校验
//!   （起始位置在缓冲区内、帧数在合理区间），精确帧数等帧解码器落地后再验。

mod bitreader;
mod params;
mod superframe;

pub use bitreader::BitReader;
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
