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
    ///
    /// ⚠️ 原来是**逐位**读（每位一次取字节 + 移位 + 或）—— VLC 解码一次要 10~15 位，
    /// 采样里这条路径占整条解码的三成。现在一次取最多 5 个字节拼成 `u64` 再截，
    /// 语义与逐位读**逐位相同**（对账测试逐像素 0 差是判据）。
    pub fn u(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.bits_left() < n as usize {
            return Err(Error::Eof);
        }
        let v = self.peek_u(n);
        self.pos += n as usize;
        Ok(v)
    }

    /// 偷看 `n` 位但**不前进**（`n <= 32`）。调用方必须自己保证位数够。
    ///
    /// ⚠️ 走 **8 字节窗口**：一次 `get(byte..byte+8)`（一次边界检查）再 `u64::from_be_bytes`
    /// 装进来，而不是逐字节 `get().unwrap_or(&0)` 五次（五次边界检查）。
    /// 窗口装得下 `7 + 32 = 39` 位，够用。**只有文件尾部不足 8 字节时**才走逐字节慢路
    /// （那时多读的字节本来就只是凑数）。
    #[inline]
    pub fn peek_u(&self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        debug_assert!(self.bits_left() >= n as usize);
        let byte = self.pos >> 3;
        let shift = (self.pos & 7) as u32;
        let w = match self.data.get(byte..byte + 8) {
            Some(s) => u64::from_be_bytes(s.try_into().unwrap()),
            None => {
                // 慢路：尾部不足 8 字节，逐字节读、缺的补 0。
                let mut w = 0u64;
                for i in 0..5usize {
                    w = (w << 8) | *self.data.get(byte + i).unwrap_or(&0) as u64;
                }
                w << 24 // 对齐到「取高 40 位」的口径，见下面那一行
            }
        };
        ((w >> (64 - shift - n)) & ((1u64 << n) - 1)) as u32
    }

    /// 偷看 `n <= 16` 位（不前进）—— 只读 2 个字节，前缀码快表用这个。
    ///
    /// ⚠️ 窗口只有 16 位，所以**必须**满足 `位偏移 + n <= 16`（调用方保证）。
    /// `FAST_BITS = 9` 时最坏 7 + 9 = 16，正好装下。
    #[inline]
    pub fn peek_u16(&self, n: u32) -> u32 {
        debug_assert!((1..=16).contains(&n));
        debug_assert!(self.bits_left() >= n as usize);
        debug_assert!((self.pos & 7) as u32 + n <= 16);
        let byte = self.pos >> 3;
        let shift = (self.pos & 7) as u32;
        let pair = match self.data.get(byte..byte + 2) {
            Some(s) => u16::from_be_bytes(s.try_into().unwrap()) as u32,
            None => ((self.data[byte] as u32) << 8) | *self.data.get(byte + 1).unwrap_or(&0) as u32,
        };
        (pair >> (16 - shift - n)) & ((1u32 << n) - 1)
    }

    pub fn bit(&mut self) -> Result<u32> {
        if self.bits_left() < 1 {
            return Err(Error::Eof);
        }
        let byte = self.data[self.pos >> 3];
        let v = ((byte >> (7 - (self.pos & 7))) & 1) as u32;
        self.pos += 1;
        Ok(v)
    }

    /// 偷看 `n` 位但不前进 —— 前缀码查表要用。
    pub fn show(&mut self, n: u32) -> Result<u32> {
        if self.bits_left() < n as usize {
            return Err(Error::Eof);
        }
        Ok(self.peek_u(n))
    }

    pub fn skip(&mut self, n: u32) -> Result<()> {
        if self.bits_left() < n as usize {
            return Err(Error::Eof);
        }
        self.pos += n as usize;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 逐位读的参考实现 —— `peek_u` 的**穷举**对照（所有位置 × 所有位宽）。
    fn slow(data: &[u8], pos: usize, n: u32) -> u32 {
        let mut v = 0u32;
        for i in 0..n as usize {
            let p = pos + i;
            let b = (data[p >> 3] >> (7 - (p & 7))) & 1;
            v = (v << 1) | b as u32;
        }
        v
    }

    /// 字宽偷看必须与逐位读**逐位相同**：`peek_u` 是 VLC 快表的地基，
    /// 它错了就是整帧错位（而且症状离根因很远）。
    #[test]
    fn peek_matches_bit_by_bit() {
        let data: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37).wrapping_add(11)).collect();
        for pos in 0..64usize {
            for n in 1..=32u32 {
                if pos + n as usize > data.len() * 8 { continue; }
                let mut r = BitReader::new(&data);
                r.skip(pos as u32).unwrap();
                let got = r.peek_u(n);
                let want = slow(&data, pos, n);
                assert_eq!(got, want, "pos={pos} n={n}");
            }
        }
    }
}

#[cfg(test)]
mod tests_u16 {
    use super::*;

    /// `peek_u16` 与 `peek_u` 在 `n <= 16` 上必须一致（前者只是少读几个字节）。
    #[test]
    fn peek_u16_agrees_with_peek_u() {
        let data: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(29).wrapping_add(7)).collect();
        for pos in 0..64usize {
            for n in 1..=16u32 {
                if pos + n as usize > data.len() * 8 { continue; }
                if (pos & 7) as u32 + n > 16 { continue; } // 16 位窗口装不下
                let mut r = BitReader::new(&data);
                r.skip(pos as u32).unwrap();
                assert_eq!(r.peek_u16(n), r.peek_u(n), "pos={pos} n={n}");
            }
        }
    }
}
