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
}

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
        Self {
            nodes,
            table: table.to_vec(),
        }
    }

    /// 建树时用的 `(code, len)`，按符号下标取回（编码侧要用）。
    pub fn code_len(&self, sym: usize) -> Option<(u32, u32)> {
        self.table.get(sym).map(|e| (e[0], e[1]))
    }

    /// 从位流里解出一个符号（返回表下标）。
    pub fn decode(&self, r: &mut BitReader) -> Result<usize> {
        let mut cur = 0usize;
        loop {
            let bit = r.bit()? as usize;
            let next = self.nodes[cur][bit];
            if next < 0 {
                return Ok((-next - 1) as usize);
            }
            if next == 0 {
                return Err(Error::Bad("invalid vlc code"));
            }
            cur = next as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
