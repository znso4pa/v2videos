//! 只给测试用的 MSB 优先位写入器 —— 用来手工拼一个位流去喂解析器。

pub struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u8,
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

    /// 收尾（不足一字节补零）。
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.cur <<= 8 - self.nbits;
            self.bytes.push(self.cur);
        }
        self.bytes
    }
}
