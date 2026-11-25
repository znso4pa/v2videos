//! **LSB 优先**的位读取器 —— Vorbis 的位打包约定。
//!
//! 与 MPEG 系（`v2v-kernel::BitReader`）**相反**，所以不能复用那一个：
//!
//! - 字节内**低位先出**：第 0 位是最先读到的；
//! - `n` 位整数也是**低位先出**：第一个读到的位是结果的最低位；
//! - 包尾不足一个字节的高位恒为 0（规范 2.1.8），所以「读进最后一个字节的
//!   填充位」读到的是 0，不是垃圾。
//!
//! 规范 2.1.6 的例子可以直接当判据：4 位写 `12` 得到 `[0 0 0 0 1 1 0 0]`
//! （按 MSB→LSB 显示），接着写 3 位 `-1` 得到 `[0 1 1 1 1 1 0 0]`。

use crate::error::{Error, Result};

pub struct BitReader<'a> {
    data: &'a [u8],
    /// 绝对位位置。
    bit: usize,
    /// 已经读到包尾（粘性：一次读穿之后，后面每次读都算 EOP）。
    eop: bool,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit: 0, eop: false }
    }

    pub fn bits_left(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.bit)
    }

    pub fn bit_pos(&self) -> usize {
        self.bit
    }

    pub fn is_eop(&self) -> bool {
        self.eop
    }

    /// 读 `n` 位（`n <= 32`），**低位先出**。
    ///
    /// `n == 0` 返回 0 且不前进（规范 2.1.9）；但**已经 EOP 之后**连 0 位读也失败
    /// （同一条的最后一句话）。
    pub fn read(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return if self.eop { Err(Error::Eop) } else { Ok(0) };
        }
        if self.eop || self.bits_left() < n as usize {
            self.eop = true;
            return Err(Error::Eop);
        }
        // 一次装 5 个字节（40 位）进 u64：32 位 + 最多 7 位偏移正好装得下。
        let byte = self.bit >> 3;
        let shift = (self.bit & 7) as u32;
        let mut w = 0u64;
        for i in 0..5usize {
            w |= (*self.data.get(byte + i).unwrap_or(&0) as u64) << (8 * i);
        }
        let v = ((w >> shift) & ((1u64 << n) - 1)) as u32;
        self.bit += n as usize;
        Ok(v)
    }

    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read(1)? != 0)
    }

    /// 读 `n` 位并按**有符号**解释（补码）。
    pub fn read_signed(&mut self, n: u32) -> Result<i32> {
        let v = self.read(n)?;
        if n == 0 {
            return Ok(0);
        }
        let shift = 32 - n;
        Ok(((v << shift) as i32) >> shift)
    }

    pub fn skip(&mut self, n: u32) -> Result<()> {
        if self.eop || self.bits_left() < n as usize {
            self.eop = true;
            return Err(Error::Eop);
        }
        self.bit += n as usize;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 规范 2.1.6 的完整例子：依次写 12（4 位）/ -1（3 位）/ 17（7 位）/ 6969（13 位）。
    ///
    /// 规范给的最终字节（按 MSB→LSB 显示）：
    ///   byte0 [1 1 1 1 1 1 0 0] = 0xFC
    ///   byte1 [0 1 0 0 1 0 0 0] = 0x48
    ///   byte2 [1 1 0 0 1 1 1 0] = 0xCE
    ///   byte3 [0 0 0 0 0 1 1 0] = 0x06
    #[test]
    fn spec_coding_example() {
        let data = [0xFC, 0x48, 0xCE, 0x06];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(4).unwrap(), 12);
        assert_eq!(r.read_signed(3).unwrap(), -1);
        assert_eq!(r.read(7).unwrap(), 17);
        assert_eq!(r.read(13).unwrap(), 6969);
        assert_eq!(r.bit_pos(), 27, "一共 27 位");
    }

    /// 规范 2.1.7 的解码例子：同一段字节按 2+2 位读，得到 'b00' 与 'b11'。
    ///
    /// 这条和上一条**故意用同一份字节**：同一串位按不同位宽读都成立，
    /// 说明位流没有对齐边界。
    #[test]
    fn spec_decoding_example() {
        let data = [0xFC, 0x48, 0xCE, 0x06];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(2).unwrap(), 0b00);
        assert_eq!(r.read(2).unwrap(), 0b11);
    }

    /// 低位先出：写 12（0b1100）后，流里的前四位是 0,0,1,1。
    #[test]
    fn integer_bits_are_lsb_first() {
        // byte = 0b0000_1100 = 12 → 前 4 位读出 12
        let data = [12u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(4).unwrap(), 12);
        // 剩下的高位是 0
        assert_eq!(r.read(4).unwrap(), 0);
    }

    /// 读穿包尾是粘性的，而且 `n == 0` 之后也失败。
    #[test]
    fn eop_is_sticky_and_zero_width_fails_after_it() {
        let data = [0xFFu8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(8).unwrap(), 0xFF);
        assert!(r.read(1).is_err());
        assert!(r.is_eop());
        assert!(r.read(0).is_err(), "EOP 之后连 0 位读也要失败");
    }

    #[test]
    fn zero_width_read_does_not_advance() {
        let data = [0xAAu8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(0).unwrap(), 0);
        assert_eq!(r.bit_pos(), 0);
    }
}
