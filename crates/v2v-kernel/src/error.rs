//! 错误类型。与 `v2v-asf` 同风格：能区分「数据坏了」和「本实现还不支持」。
//!
//! 这是**编解码器共用**的错误类型（aac / mp4v / vc1 / mpeg12 / msmpeg4 都用它），
//! 所以文案里不出现具体编解码器的名字 —— 否则一个 AAC 的解码失败会打印
//! 「bad msmpeg4」，读者会往错的方向查。

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// 位流提前结束。
    Eof,
    /// 位流内容不合法。
    Bad(&'static str),
    /// 结构合法但本实现不支持。
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eof => f.write_str("bitstream ended early"),
            Self::Bad(m) => write!(f, "bad stream: {m}"),
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
