//! Ogg 解封装的错误类型。

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// 字节流不是合法 Ogg（或页结构被破坏）。
    Bad(String),
    /// 合法但本实现不支持。
    Unsupported(String),
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Bad(s) => write!(f, "Ogg 数据不合法：{s}"),
            Error::Unsupported(s) => write!(f, "Ogg 不支持：{s}"),
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
