//! 前缀码（VLC）解码。
//!
//! 表都是 `[code, len]` 的形态（MS-MPEG-4 的 VLC 表全是这个形状）。这里按位建一棵二叉树，
//! 每个叶子挂符号下标。**表本身是数据，解码器是逻辑** —— 换表不用改这里。

use crate::bitreader::BitReader;
use crate::error::{Error, Result};

/// 由 `[code, len]` 表编译出来的前缀码树。
pub struct Vlc {
    /// 每个节点两个子：[0] = 读到位 0，[1] = 读到位 1。
    /// 值语义：`0` = 未设；`> 0` = 子节点下标；`< 0` = 叶子，符号 = `-值 - 1`。
    nodes: Vec<[i32; 2]>,
    /// 原表（编码侧要按符号反查 `(code, len)`）。
    table: Vec<[u32; 2]>,
    /// 正值为 `(符号 << 5) | 码长`，负值取反后是长码的续查节点。
    /// 零表示前缀未定义，逐位解码到原来的错误位置。
    fast: Vec<i32>,
}

/// 快表的位宽。取 9 与 ffmpeg 的 `*_VLC_BITS` 同档：大多数符号一次命中，
/// 表也就 2 KB。
const FAST_BITS: u32 = 9;

impl Vlc {
    /// `table[i] = [code, len]`；`len == 0` 的项视为未用（跳过）。
    pub fn new(table: &[[u32; 2]]) -> Self {
        let mut nodes: Vec<[i32; 2]> = vec![[0, 0]];
        for (sym, &[code, len]) in table.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let mut cur = 0usize;
            for i in (0..len).rev() {
                let bit = ((code >> i) & 1) as usize;
                if i == 0 {
                    nodes[cur][bit] = -(sym as i32) - 1;
                } else {
                    let next = nodes[cur][bit];
                    if next > 0 {
                        cur = next as usize;
                    } else {
                        nodes.push([0, 0]);
                        let idx = nodes.len() - 1;
                        nodes[cur][bit] = idx as i32;
                        cur = idx;
                    }
                }
            }
        }
        let mut fast = vec![0; 1 << FAST_BITS];
        for (sym, &[code, len]) in table.iter().enumerate() {
            if len == 0 || len > FAST_BITS {
                continue;
            }
            let entry = ((sym as i32) << 5) | len as i32;
            let base = (code << (FAST_BITS - len)) as usize;
            fast[base..base + (1 << (FAST_BITS - len))].fill(entry);
        }
        for &[code, len] in table.iter().filter(|entry| entry[1] > FAST_BITS) {
            let prefix = ((code >> (len - FAST_BITS)) & ((1 << FAST_BITS) - 1)) as usize;
            if fast[prefix] != 0 {
                continue;
            }
            let mut node = 0usize;
            for shift in (0..FAST_BITS).rev() {
                let bit = (prefix >> shift) & 1;
                let next = nodes[node][bit];
                if next <= 0 {
                    break;
                }
                node = next as usize;
                if shift == 0 {
                    fast[prefix] = -next;
                }
            }
        }
        Self {
            nodes,
            table: table.to_vec(),
            fast,
        }
    }

    /// 建树时用的 `(code, len)`，按符号下标取回（编码侧要用）。
    pub fn code_len(&self, sym: usize) -> Option<(u32, u32)> {
        self.table.get(sym).map(|e| (e[0], e[1]))
    }

    /// 从位流里解出一个符号（返回表下标）。
    ///
    /// 先查 `FAST_BITS` 位。长码从快表记录的节点继续，尾部或坏前缀从根开始。
    pub fn decode(&self, r: &mut BitReader) -> Result<usize> {
        let mut node = 0;
        if r.bits_left() >= FAST_BITS as usize {
            let entry = self.fast[r.peek_u16(FAST_BITS) as usize];
            if entry > 0 {
                r.set_pos(r.bit_pos() + (entry as usize & 31));
                return Ok((entry >> 5) as usize);
            }
            if entry < 0 {
                r.set_pos(r.bit_pos() + FAST_BITS as usize);
                node = (-entry) as usize;
            }
        }
        self.decode_from(node, || r.bit().ok())
            .ok_or(Error::Bad("invalid vlc code"))
    }

    /// 解一个符号，位从调用方给的闭包取，结束或未定义时返回 `None`。
    pub fn decode_bits<F: FnMut() -> Option<u32>>(&self, next_bit: F) -> Option<usize> {
        self.decode_from(0, next_bit)
    }

    fn decode_from<F: FnMut() -> Option<u32>>(&self, mut cur: usize, mut next_bit: F) -> Option<usize> {
        loop {
            let bit = next_bit()? as usize;
            let next = self.nodes[cur][bit];
            if next < 0 {
                return Some((-next - 1) as usize);
            }
            if next == 0 {
                return None;
            }
            cur = next as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resumes_long_codes_and_preserves_the_error_cursor() {
        let vlc = Vlc::new(&[[0, 1], [0b10_0000_0000, 10], [0b10_0000_0001, 10]]);
        let mut valid = BitReader::new(&[0x80, 0x40]);
        assert_eq!(vlc.decode(&mut valid).unwrap(), 2);
        assert_eq!(valid.bit_pos(), 10);
        assert_eq!(vlc.decode(&mut valid).unwrap(), 0);

        let mut invalid = BitReader::new(&[0xc0, 0]);
        assert!(vlc.decode(&mut invalid).is_err());
        assert_eq!(invalid.bit_pos(), 2);
    }

    #[test]
    fn decodes_a_tiny_hand_made_table() {
        // {a: "0", b: "10", c: "11"}
        let table: [[u32; 2]; 3] = [[0b0, 1], [0b10, 2], [0b11, 2]];
        let v = Vlc::new(&table);
        let mut r = BitReader::new(&[0b0_11_00000]);
        assert_eq!(v.decode(&mut r).unwrap(), 0); // "0"
        assert_eq!(v.decode(&mut r).unwrap(), 2); // "11"
    }

    #[test]
    fn rejects_an_undefined_prefix() {
        let table: [[u32; 2]; 1] = [[0b0, 1]];
        let v = Vlc::new(&table);
        let mut r = BitReader::new(&[0b1000_0000]);
        assert!(v.decode(&mut r).is_err());
    }
}
