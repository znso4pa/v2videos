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
//!
//! ## 为什么是「64 位窗口 + 按需补充」
//!
//! 第一版每次 `read()` 都装 5 个字节进 `u64`（`data.get()` × 5 + 循环 + 边界
//! 检查）。**码本走树时每读 1 位就调用一次 `read()`** —— 一份 4 分钟的真样本
//! 是上亿次 `read(1)`，光装字节就占了小一半时间。
//!
//! 现在维护一个「最低位 = 下一个要读的位」的 `u64` 窗口，一次补 8 个字节，
//! 之后每次读只是 `& mask` + `>>=`。`peek()` 让码本能「一次看 10 位查表」，
//! 不必逐位走树（见 `codebook::Codebook::fast`）。
//!
//! 语义**完全不变**：`bits_left()` 仍按「数据总位数 − 已消费位数」算，
//! 读穿包尾仍然是粘性的 `Eop`。

use crate::error::{Error, Result};

pub struct BitReader<'a> {
    data: &'a [u8],
    /// 下一个还没装进 `buf` 的字节下标。
    next_byte: usize,
    /// 位窗口：**最低位 = 下一个要读的位**。
    buf: u64,
    /// `buf` 里还有多少有效位。
    nbits: u32,
    /// 绝对位位置（`bit_pos()` 与 `bits_left()` 用）。
    bit: usize,
    /// 已经读到包尾（粘性：一次读穿之后，后面每次读都算 EOP）。
    eop: bool,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, next_byte: 0, buf: 0, nbits: 0, bit: 0, eop: false }
    }

    #[inline]
    pub fn bits_left(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.bit)
    }

    pub fn bit_pos(&self) -> usize {
        self.bit
    }

    pub fn is_eop(&self) -> bool {
        self.eop
    }

    /// 往窗口里补字节，直到装满（> 56 位）或数据用尽。
    ///
    /// 每次最多补 8 个字节，所以之后连续 56 次 `read(1)` 都不会再进这个循环。
    #[inline]
    fn refill(&mut self) {
        while self.nbits <= 56 && self.next_byte < self.data.len() {
            self.buf |= u64::from(self.data[self.next_byte]) << self.nbits;
            self.next_byte += 1;
            self.nbits += 8;
        }
    }

    /// 读 `n` 位（`n <= 32`），**低位先出**。
    ///
    /// `n == 0` 返回 0 且不前进（规范 2.1.9）；但**已经 EOP 之后**连 0 位读也失败
    /// （同一条的最后一句话）。
    #[inline]
    pub fn read(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return if self.eop { Err(Error::Eop) } else { Ok(0) };
        }
        if self.eop || self.bits_left() < n as usize {
            self.eop = true;
            return Err(Error::Eop);
        }
        if self.nbits < n {
            self.refill();
        }
        let v = (self.buf & ((1u64 << n) - 1)) as u32;
        self.buf >>= n;
        self.nbits -= n;
        self.bit += n as usize;
        Ok(v)
    }

    /// 看接下来 `n` 位但**不前进**。位不够（包尾附近）返回 `None`。
    ///
    /// 这是给码本查表用的：一次看 10 位、查表命中就 `consume`，不必逐位走树。
    #[inline]
    pub fn peek(&mut self, n: u32) -> Option<u32> {
        debug_assert!(n <= 32);
        if self.eop || self.bits_left() < n as usize {
            return None;
        }
        if self.nbits < n {
            self.refill();
        }
        Some((self.buf & ((1u64 << n) - 1)) as u32)
    }

    /// 前进 `n` 位。调用方必须保证 `peek(n)` 刚刚成功过。
    #[inline]
    pub fn consume(&mut self, n: u32) {
        debug_assert!(self.nbits >= n);
        self.buf >>= n;
        self.nbits -= n;
        self.bit += n as usize;
    }

    #[inline]
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
        let mut left = n as usize;
        while left > 0 {
            if self.nbits == 0 {
                self.refill();
                if self.nbits == 0 {
                    // 理论上到不了：上面已经确认 `bits_left >= n`。
                    self.eop = true;
                    return Err(Error::Eop);
                }
            }
            // 一次最多移 63 位：`nbits` 可能正好是 64（补窗口时 56+8），
            // 而 `u64 >> 64` 在 debug 下 panic、release 下会按 64 取模 —— 两种都不对。
            let take = left.min(self.nbits as usize).min(63);
            self.buf >>= take;
            self.nbits -= take as u32;
            self.bit += take;
            left -= take;
        }
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

    /// 窗口跨字节边界：32 位一次读完再逐位读，必须与逐位读一致。
    ///
    /// 这条钉的是「补窗口时 `nbits` 与实际位流错位」那类 bug —— 它不会报错，
    /// 只会让后面所有值整体偏几个位。
    #[test]
    fn wide_and_narrow_reads_agree_across_the_window() {
        let data: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37) ^ 0x5A).collect();
        // 一路 32 位读
        let mut wide = BitReader::new(&data);
        let mut wide_bits = Vec::new();
        for _ in 0..(data.len() * 8 / 32) {
            let v = wide.read(32).unwrap();
            for k in 0..32 {
                wide_bits.push((v >> k) & 1);
            }
        }
        // 一路 1 位读
        let mut narrow = BitReader::new(&data);
        let narrow_bits: Vec<u32> = (0..wide_bits.len()).map(|_| narrow.read(1).unwrap()).collect();
        assert_eq!(wide_bits, narrow_bits);

        // 一路乱序宽度读（3/1/7/2/13…）也要对得上
        let mut mixed = BitReader::new(&data);
        let widths = [3u32, 1, 7, 2, 13, 5, 1, 11];
        let mut got = Vec::new();
        for &w in widths.iter().cycle().take(40) {
            if mixed.bits_left() < w as usize {
                break;
            }
            let v = mixed.read(w).unwrap();
            for k in 0..w {
                got.push((v >> k) & 1);
            }
        }
        assert_eq!(got, narrow_bits[..got.len()]);
    }

    /// `peek` + `consume` 必须与 `read` 完全等价，且在包尾附近优雅地返回 `None`。
    #[test]
    fn peek_then_consume_matches_read() {
        let data = [0b1011_0011u8, 0b0100_1110];
        let mut a = BitReader::new(&data);
        let mut b = BitReader::new(&data);
        for n in [1u32, 3, 2, 4, 5, 1] {
            let peeked = b.peek(n);
            if let Some(p) = peeked {
                assert_eq!(p, a.read(n).unwrap());
                b.consume(n);
            } else {
                assert!(a.read(n).is_err());
            }
        }
        // 只剩 0 位时 peek 必须给 None（不能假装读到 0）
        assert_eq!(a.bits_left(), 0);
        assert_eq!(b.bits_left(), 0);
        assert!(b.peek(1).is_none());
    }

    /// `skip` 跨过 64 位窗口边界也要正确（一次 skip 大于窗口容量）。
    #[test]
    fn skip_can_cross_the_window() {
        let data: Vec<u8> = (0..32u8).collect();
        let mut r = BitReader::new(&data);
        r.skip(100).unwrap();
        assert_eq!(r.bit_pos(), 100);
        // 第 100 位起的 4 位：byte12 = 12 = 0b00001100，位偏移 4 → 高 4 位 = 0
        assert_eq!(r.read(4).unwrap(), 0);
        // 再 skip 到末尾之前
        r.skip(28).unwrap();
        assert_eq!(r.bit_pos(), 132);
    }
}
