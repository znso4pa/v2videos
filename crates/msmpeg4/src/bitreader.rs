//! MSB 优先的位读取器 —— MPEG-4 / H.263 系位流的约定。
//!
//! 刻意写成逐位取，先把**正确**钉死；后面真要提速再换 64 位窗口。

use crate::error::{Error, Result};

pub struct BitReader<'a> {
    data: &'a [u8],
    /// 绝对位位置。
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn bits_left(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.pos)
    }

    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    /// 取 `n` 位（`n <= 32`），MSB 在前。
    pub fn u(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.bits_left() < n as usize {
            return Err(Error::Eof);
        }
        let mut v = 0u32;
        for _ in 0..n {
            let byte = self.data[self.pos >> 3];
            let bit = (byte >> (7 - (self.pos & 7))) & 1;
            v = (v << 1) | bit as u32;
            self.pos += 1;
        }
        Ok(v)
    }

    pub fn bit(&mut self) -> Result<u32> {
        self.u(1)
    }

    /// 偷看 `n` 位但不前进 —— 前缀码查表要用。
    pub fn show(&mut self, n: u32) -> Result<u32> {
        let save = self.pos;
        let v = self.u(n);
        self.pos = save;
        v
    }

    pub fn skip(&mut self, n: u32) -> Result<()> {
        if self.bits_left() < n as usize {
            return Err(Error::Eof);
        }
        self.pos += n as usize;
        Ok(())
    }
}
