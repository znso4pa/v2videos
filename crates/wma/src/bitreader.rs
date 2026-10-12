//! MSB 优先的位读取器（WMA 用的就是这个方向，见模块文档里的实测）。

use crate::{Error, Result};

pub struct BitReader<'a> {
    buf: &'a [u8],
    /// 已读位数。
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// 读 `n` 位（`n <= 32`），MSB 在前。
    pub fn get(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(Error::Bad("一次最多读 32 位"));
        }
        if self.pos + n as usize > self.buf.len() * 8 {
            return Err(Error::Bad("读越界"));
        }
        let mut v = 0u32;
        for _ in 0..n {
            let byte = self.buf[self.pos >> 3];
            let bit = (byte >> (7 - (self.pos & 7))) & 1;
            v = (v << 1) | bit as u32;
            self.pos += 1;
        }
        Ok(v)
    }

    /// 读 1 位（`get_bits1`）。
    pub fn bit(&mut self) -> Result<u32> {
        self.get(1)
    }

    /// 已读位数（`get_bits_count`）。
    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        if self.pos + n > self.buf.len() * 8 {
            return Err(Error::Bad("跳过越界"));
        }
        self.pos += n;
        Ok(())
    }

    /// 剩余位数。
    pub fn bits_left(&self) -> usize {
        self.buf.len() * 8 - self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_msb_first() {
        let mut r = BitReader::new(&[0b1011_0010, 0b0100_0000]);
        assert_eq!(r.get(1).unwrap(), 1);
        assert_eq!(r.get(2).unwrap(), 0b01);
        assert_eq!(r.get(5).unwrap(), 0b1_0010);
        assert_eq!(r.bit_pos(), 8);
        assert_eq!(r.get(2).unwrap(), 0b01);
        assert_eq!(r.bits_left(), 6);
    }

    #[test]
    fn overread_is_an_error_not_a_panic() {
        let mut r = BitReader::new(&[0xFF]);
        assert!(r.get(8).is_ok());
        assert!(r.get(1).is_err());
        assert!(r.skip(1).is_err());
    }
}
