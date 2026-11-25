//! 错误类型。
//!
//! 与 `v2v-ogg` / `v2v-asf` 同风格：能区分「数据坏了」「本实现还不支持」，
//! 以及**位流读到头了**（[`Error::Eop`]）—— 最后这一条不是错误，是 Vorbis 的
//! 常规状态：规范明说「截断的包是正常模式」，音频包在 residue 中途读完要按
//! 「已经解出来的部分」出声音，而不是把整包丢掉。

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// 位流内容不合法（长度表凑不成树、码本编号越界、字段自相矛盾…）。
    Bad(String),
    /// 合法但本实现没做。
    Unsupported(String),
    /// 读到包尾。**不是错误**：调用方按规范决定「当作静音」还是「整包作废」。
    Eop,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Bad(s) => write!(f, "Vorbis 数据不合法：{s}"),
            Error::Unsupported(s) => write!(f, "Vorbis 不支持：{s}"),
            Error::Eop => f.write_str("Vorbis 包读完了"),
        }
    }
}

impl std::error::Error for Error {}

/// `ilog(x)`：`x` 最高有效位的位置（1 起）。规范 9.2.1，`ilog(0) = 0`。
///
/// 它同时是「读几个位」的口径：`ilog(n)` 位能表示 `0..n` 的编号。
#[inline]
pub fn ilog(x: u32) -> u32 {
    if x == 0 {
        0
    } else {
        32 - x.leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::ilog;

    /// 规范 9.2.1 给的例子，逐条钉住。
    #[test]
    fn ilog_matches_spec_examples() {
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(2), 2);
        assert_eq!(ilog(3), 2);
        assert_eq!(ilog(4), 3);
        assert_eq!(ilog(7), 3);
        assert_eq!(ilog(8), 4);
    }
}
