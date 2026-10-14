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
    ///
    /// ⚠️ **一次拼完再吐整字节**，不是逐位循环：旧版每写一位都要一次移位 + 或 +
    /// 计数 + 判满，写满一个字节要走 8 圈（`append` 还按字节调它）。
    /// 实测 `put` 一个人占整条链路的 1.4% —— 换成 64 位累加器后同样的输出。
    pub fn put(&mut self, v: u32, n: u32) {
        debug_assert!(n <= 32, "一次最多写 32 位");
        if n == 0 {
            return;
        }
        // `cur` 是**右对齐**的尾位（`nbits` 位）。把它和新位拼进一个 64 位累加器：
        // 高位是 cur、低位是 `v` 的低 `n` 位。`nbits + n <= 7 + 32 = 39` 位，装得下。
        let mask = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
        let acc = (u64::from(self.cur) << n) | u64::from(v & mask);
        let mut total = u32::from(self.nbits) + n;
        while total >= 8 {
            total -= 8;
            self.bytes.push(((acc >> total) & 0xff) as u8);
        }
        self.cur = (acc & ((1u64 << total) - 1)) as u8;
        self.nbits = total as u8;
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

    /// 把另一个写入器写出的位（含未满一字节的尾位）原样接到自己后面。
    ///
    /// P 帧的帧内/帧间判决用：两条路径先各自写进临时写入器比大小，赢的那条
    /// 整段接进正式位流 —— 保证「比较时的位数」和「实际写出的位数」是同一个数。
    pub fn append(&mut self, other: &BitWriter) {
        for &b in &other.bytes {
            self.put(b as u32, 8);
        }
        if other.nbits > 0 {
            self.put(other.cur as u32, other.nbits as u32);
        }
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

    /// 逐位写的参考实现 —— 就是被换掉的那一版，换实现时对着它比。
    fn put_bitwise(out: &mut Vec<u8>, cur: &mut u8, nbits: &mut u8, v: u32, n: u32) {
        for i in (0..n).rev() {
            let bit = ((v >> i) & 1) as u8;
            *cur = (*cur << 1) | bit;
            *nbits += 1;
            if *nbits == 8 {
                out.push(*cur);
                *cur = 0;
                *nbits = 0;
            }
        }
    }

    #[test]
    fn put_matches_a_bit_by_bit_reference() {
        // 2000 次随机 (v, n)，n 覆盖 0 / 1 / 7 / 8 / 9 / 31 / 32 —— 位宽跨界、
        // 零位宽、整字节、跨 64 位累加器边界都在里面。判据是**字节完全相同**。
        let mut w = BitWriter::new();
        let mut want: Vec<u8> = Vec::new();
        let (mut cur, mut nbits) = (0u8, 0u8);
        let mut state = 0xdead_beefu32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        for i in 0..2000usize {
            let n = match i % 7 {
                0 => 0,
                1 => 1,
                2 => 7,
                3 => 8,
                4 => 9,
                5 => 31,
                _ => 32,
            };
            let v = next();
            w.put(v, n);
            put_bitwise(&mut want, &mut cur, &mut nbits, v, n);
        }
        // ⚠️ 参考实现也要走一遍 `finish()` 的补零：不满一字节的尾位由 `align`
        // 补零后才落成字节（第一版测试漏了这一步，把实现的正确行为当成 bug）。
        if nbits > 0 {
            let pad = 8 - u32::from(nbits);
            put_bitwise(&mut want, &mut cur, &mut nbits, 0, pad);
        }
        assert_eq!(w.finish(), want, "与逐位实现不一致");
    }

    #[test]
    fn append_is_bit_exact_across_the_seam() {
        // `append` 要在**位**这一层接得上（不是字节对齐）—— P 帧的帧内/帧间判决
        // 就是靠它把赢的那条临时位流整段接进正式位流。
        let mut a = BitWriter::new();
        a.put(0b101, 3);
        a.put(0b11, 2); // 共 5 位，不满一字节
        let mut b = BitWriter::new();
        b.put(0b1111, 4);
        b.put(0b0, 1);
        a.append(&b);
        assert_eq!(a.bit_pos(), 10);
        assert_eq!(a.finish(), vec![0b1011_1111, 0b1000_0000]);
    }
}
