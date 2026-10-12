//! MSB 优先的位写入器。
//!
//! 和 `bitreader.rs` 对称：MS-MPEG-4 的位流全是 MSB 在前，VLC 码也按 `(code, len)`
//! 从高位往下写。
//!
//! 编码器要用，所以是正式模块（不是 `#[cfg(test)]`）；测试里手工拼位流也复用它
//! （见 `testbit`）。

pub struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            bytes: Vec::new(),
            cur: 0,
            nbits: 0,
        }
    }

    /// 写 `n` 位（`n <= 32`），MSB 在前。
    pub fn put(&mut self, v: u32, n: u32) {
        debug_assert!(n <= 32, "一次最多写 32 位");
        for i in (0..n).rev() {
            let bit = ((v >> i) & 1) as u8;
            self.cur = (self.cur << 1) | bit;
            self.nbits += 1;
            if self.nbits == 8 {
                self.bytes.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    /// 写 `n` 位**有符号**值（补码）—— 第三 escape 的 level 就是这么写的。
    pub fn put_signed(&mut self, v: i32, n: u32) {
        let mask = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
        self.put((v as u32) & mask, n);
    }

    /// 已写出的位数（含未满一字节的那些）。
    ///
    /// 编码器里用来确认「收尾补齐后尾部不足 8 位」—— 不然 I 帧的 17 位 ext header
    /// 判据（剩余位数落在 `[17, 25)`）会被误触发。
    #[allow(dead_code)]
    pub fn bit_pos(&self) -> usize {
        self.bytes.len() * 8 + self.nbits as usize
    }

    /// 补齐到字节边界（MP43 每帧从字节边界开始）。
    pub fn align(&mut self) {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.put(0, pad as u32);
        }
    }

    /// 收尾（不足一字节补零）。
    pub fn finish(mut self) -> Vec<u8> {
        self.align();
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_msb_first() {
        let mut w = BitWriter::new();
        w.put(0b1, 1);
        w.put(0b10, 2);
        w.put(0b111, 3);
        assert_eq!(w.bit_pos(), 6);
        assert_eq!(w.finish(), vec![0b1_10_111_00]);
    }

    #[test]
    fn signed_uses_twos_complement() {
        let mut w = BitWriter::new();
        w.put_signed(-1, 8);
        assert_eq!(w.finish(), vec![0xff]);
        let mut w = BitWriter::new();
        w.put_signed(-128, 8);
        assert_eq!(w.finish(), vec![0x80]);
        let mut w = BitWriter::new();
        w.put_signed(127, 8);
        assert_eq!(w.finish(), vec![0x7f]);
    }
}
