//! 两个极小的读取器：一个读内存切片（`Cursor`），一个带位置计数的流读取器（`StreamReader`）。
//! 刻意手写 —— 为的是「越界」返回一个能区分于 EOF 的错误，而不是静默给零。

use std::io::{Read, Seek, SeekFrom};

use crate::error::{Error, Result};
use crate::guid::Guid;

/// 在内存切片上按 ASF 的小端约定读取。
pub struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.buf.len() {
            return Err(Error::Bad("seek past end of buffer"));
        }
        self.pos = pos;
        Ok(())
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(Error::Eof);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.take(n).map(|_| ())
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    pub fn guid(&mut self) -> Result<Guid> {
        let b = self.take(16)?;
        let mut g = [0u8; 16];
        g.copy_from_slice(b);
        Ok(g)
    }
}

/// 带字节计数的流读取器 —— `data_offset` 这类「文件里的绝对位置」要靠它。
pub struct StreamReader<R: Read> {
    inner: R,
    pos: u64,
}

impl<R: Read> StreamReader<R> {
    pub fn new(inner: R) -> Self {
        Self { inner, pos: 0 }
    }

    /// 已消费的字节数（文件里的绝对位置）。
    #[allow(dead_code)]
    pub fn pos(&self) -> u64 {
        self.pos
    }

    pub fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.inner.read_exact(buf)?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    pub fn u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }

    pub fn u16(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    pub fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    pub fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    pub fn guid(&mut self) -> Result<Guid> {
        let mut g = [0u8; 16];
        self.read_exact(&mut g)?;
        Ok(g)
    }

    /// 读 `n` 字节到新 `Vec`（`n` 上限由调用方把关，避免一个坏长度就把内存吃光）。
    pub fn vec(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; n];
        self.read_exact(&mut v)?;
        Ok(v)
    }

    /// 尽量填满 `buf`，返回实际读到的字节数（0 = 正常 EOF）。
    /// ASF 的数据包是定长的，最后一块可能短读，所以不能直接用 `read_exact`。
    pub fn read_full(&mut self, buf: &mut [u8]) -> Result<usize> {
        let mut filled = 0;
        while filled < buf.len() {
            match self.inner.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => {
                    filled += n;
                    self.pos += n as u64;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(filled)
    }
}

/// 可 seek 的读法 —— **单独一个 impl 块**，`AsfDemuxer` 的主体仍然只要求 `Read`
/// （顺序读不需要可 seek；要求一收紧，所有非 seek 的读法都被连带拒绝）。
impl<R: Read + Seek> StreamReader<R> {
    /// 跳到文件里的绝对位置。
    pub fn seek_to(&mut self, pos: u64) -> Result<()> {
        self.inner.seek(SeekFrom::Start(pos)).map_err(Error::Io)?;
        self.pos = pos;
        Ok(())
    }

    /// 从**文件尾**往前偏移 `off` 字节（负数），返回落点的绝对位置。
    ///
    /// ASF 的 Simple Index 在文件**最后**，而头解析时并不知道它有多大 ——
    /// 只能从尾部倒着找。
    pub fn seek_from_end(&mut self, off: i64) -> Result<u64> {
        let p = self.inner.seek(SeekFrom::End(off)).map_err(Error::Io)?;
        self.pos = p;
        Ok(p)
    }
}
