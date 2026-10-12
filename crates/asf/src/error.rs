//! 错误类型。**不引 anyhow/thiserror** —— 一个解封装器不值得为错误处理拉依赖。

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// 底层 IO 失败。
    Io(std::io::Error),
    /// 数据不合法（结构对不上、越界、字段自相矛盾）。
    Bad(&'static str),
    /// 结构合法，但本实现（还）不支持 —— 要能和「文件坏了」区分开。
    Unsupported(String),
    /// 数据提前结束。
    Eof,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Bad(m) => write!(f, "bad asf: {m}"),
            Self::Unsupported(m) => write!(f, "unsupported asf: {m}"),
            Self::Eof => f.write_str("unexpected end of data"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
