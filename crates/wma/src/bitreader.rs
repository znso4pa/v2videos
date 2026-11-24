//! 共享 MSB 位读取器的 WMA 接口。

use crate::{Error, Result};

pub struct BitReader<'a> {
    pub(crate) inner: v2v_kernel::BitReader<'a>,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            inner: v2v_kernel::BitReader::new(buf),
        }
    }

    /// 读 `n` 位（`n <= 32`），MSB 在前。
    pub fn get(&mut self, n: u32) -> Result<u32> {
        if n > 32 {
            return Err(Error::Bad("一次最多读 32 位"));
        }
        self.inner.u(n).map_err(|_| Error::Bad("读越界"))
    }

    /// 读 1 位（`get_bits1`）。
    pub fn bit(&mut self) -> Result<u32> {
        self.inner.bit().map_err(|_| Error::Bad("读越界"))
    }

    /// 已读位数（`get_bits_count`）。
    pub fn bit_pos(&self) -> usize {
        self.inner.bit_pos()
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        if n > self.bits_left() {
            return Err(Error::Bad("跳过越界"));
        }
        self.inner.set_pos(self.inner.bit_pos() + n);
        Ok(())
    }

    /// 剩余位数。
    pub fn bits_left(&self) -> usize {
        self.inner.bits_left()
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
