//! 码本：Huffman 解码树 + VQ 查表。规范第 3 章。
//!
//! ## 两个容易写错的地方
//!
//! 1. **码字的赋值顺序不是教科书里的 canonical Huffman。** 规范说「每个用到的条目
//!    按**条目顺序**分到当时最低的、还没被占用的码字」—— 也就是一棵**从左往右长**
//!    的树（规范 3.2.1 的图 4）。按「长度排序后再 canonical」会得到另一套码字：
//!    规范自己的例子（长度 `2,4,4,4,4,2,3,3`）里，条目 5 的码字是 `10`，
//!    而 canonical 会给 `01` —— 而 `01` 是条目 1..4 的前缀，根本不可能。
//! 2. **码字按位顺序读进树，但位流是 LSB 先出的**：码字的第一位是树的根分支，
//!    所以「逐位读、逐位走树」自然正确，不需要反转。

use crate::bit::BitReader;
use crate::error::{Error, Result};
use crate::ilog;

/// 一个码本。
pub struct Codebook {
    pub dimensions: usize,
    pub entries: usize,
    /// 每个条目的码长（0 = 未使用）。只用于诊断。
    lengths: Vec<u8>,
    /// 二叉树：`0` 未定义、`> 0` 子节点下标、`< 0` 叶子（符号 = `-值 - 1`）。
    nodes: Vec<[i32; 2]>,
    /// 单条目码本：读一个值恒返回它，并从流里**吞掉一位**。
    single: Option<usize>,
    /// VQ 查表（lookup type 1 / 2）。type 0 没有。
    pub vq: Option<Vq>,
}

/// VQ 查表：`entries` 个长度为 `dimensions` 的向量，已按规范展开成 float。
pub struct Vq {
    pub dimensions: usize,
    pub values: Vec<f32>,
}

impl Codebook {
    /// 从 setup 头里读一个码本。规范 3.2.1。
    pub fn read(bit: &mut BitReader) -> Result<Self> {
        let sync = bit.read(24)?;
        if sync != 0x0056_4342 {
            return Err(Error::Bad(format!(
                "码本同步字 {sync:#08x}（应为 0x564342）"
            )));
        }
        let dimensions = bit.read(16)? as usize;
        let entries = bit.read(24)? as usize;
        if dimensions == 0 || entries == 0 {
            return Err(Error::Bad(format!(
                "码本 dimensions={dimensions} entries={entries}"
            )));
        }
        // 资源上限，不是规范上限：坏头部不该让内存炸掉。
        if dimensions > 1024 || entries > (1 << 20) {
            return Err(Error::Unsupported(format!(
                "码本太大（dimensions={dimensions}, entries={entries}）"
            )));
        }

        let mut lengths = vec![0u8; entries];
        if bit.read_bool()? {
            // 有序：按长度升序，只写「每个长度有几个」
            let mut cur = 0usize;
            let mut len = bit.read(5)? as u8 + 1;
            while cur < entries {
                let number = bit.read(ilog((entries - cur) as u32))? as usize;
                if number == 0 || cur + number > entries {
                    return Err(Error::Bad(format!(
                        "有序码本的长度表越界（cur={cur}, number={number}, entries={entries}）"
                    )));
                }
                lengths[cur..cur + number].fill(len);
                cur += number;
                len += 1;
                if len > 32 {
                    return Err(Error::Bad("码长超过 32 位".into()));
                }
            }
        } else {
            let sparse = bit.read_bool()?;
            for l in lengths.iter_mut() {
                if sparse && !bit.read_bool()? {
                    continue; // 未使用
                }
                *l = bit.read(5)? as u8 + 1;
            }
        }

        let (nodes, single) = build_tree(&lengths)?;

        let lookup_type = bit.read(4)?;
        let vq = match lookup_type {
            0 => None,
            1 | 2 => Some(read_vq(bit, lookup_type, entries, dimensions)?),
            t => {
                return Err(Error::Unsupported(format!(
                    "码本 lookup type {t}（Vorbis I 只认 0 / 1 / 2）"
                )))
            }
        };

        Ok(Self { dimensions, entries, lengths, nodes, single, vq })
    }

    /// 解一个 Huffman 符号（条目下标）。规范 3.2.1「huffman decision tree」。
    pub fn decode_scalar(&self, bit: &mut BitReader) -> Result<usize> {
        if let Some(sym) = self.single {
            // 规范 errata 20150226：单条目码本「吞一位」，读到的值不计（0 / 1 都行）。
            bit.read(1)?;
            return Ok(sym);
        }
        let mut node = 0usize;
        for _ in 0..32 {
            let b = bit.read(1)? as usize;
            let next = self.nodes[node][b];
            if next < 0 {
                return Ok((-next - 1) as usize);
            }
            if next == 0 {
                return Err(Error::Bad("码本里没有这个码字".into()));
            }
            node = next as usize;
        }
        Err(Error::Bad("码字长度超过 32 位".into()))
    }

    /// 解一个 VQ 向量（VQ 上下文）。返回的是查表里的切片，不分配。
    pub fn decode_vector(&self, bit: &mut BitReader) -> Result<&[f32]> {
        let vq = self
            .vq
            .as_ref()
            .ok_or_else(|| Error::Bad("这个码本没有 VQ 查表，不能用在 VQ 上下文".into()))?;
        let e = self.decode_scalar(bit)?;
        let base = e * vq.dimensions;
        Ok(&vq.values[base..base + vq.dimensions])
    }

    /// 条目数（诊断用）。
    pub fn entry_count(&self) -> usize {
        self.entries
    }

    /// 码长表（诊断用）。
    pub fn codeword_lengths(&self) -> &[u8] {
        &self.lengths
    }
}

/// 按规范的「最左可用叶子」规则建树。规范 3.2.1 图 4 的那段文字。
///
/// 返回 `(树, 单条目符号)`；单条目码本不建树（没有位可读）。
fn build_tree(lengths: &[u8]) -> Result<(Vec<[i32; 2]>, Option<usize>)> {
    let used: Vec<(usize, u8)> = lengths
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l > 0)
        .map(|(i, &l)| (i, l))
        .collect();
    if used.is_empty() {
        return Err(Error::Bad("码本没有任何可用码字".into()));
    }
    if used.len() == 1 {
        // 规范 errata：单条目码本的码长必须是 1（写进去的值是 `len-1 == 0`）。
        if used[0].1 != 1 {
            return Err(Error::Bad(format!(
                "单条目码本的码长是 {}（规范要求 1）",
                used[0].1
            )));
        }
        return Ok((Vec::new(), Some(used[0].0)));
    }

    // Kraft 和必须恰好填满：多一个码字是「过满」，少一个是「没长完」，两者都不可解。
    let max_len = used.iter().map(|&(_, l)| l).max().unwrap() as u32;
    let mut kraft = 0u64;
    for &(_, l) in &used {
        kraft += 1u64 << (max_len - l as u32);
    }
    let full = 1u64 << max_len;
    if kraft != full {
        return Err(Error::Bad(format!(
            "码本长度表凑不成完整的树（Kraft 和 {kraft}/{full}）"
        )));
    }

    let mut nodes: Vec<[i32; 2]> = vec![[0, 0]];
    for (sym, len) in used {
        if !place(&mut nodes, 0, 0, len, sym as i32) {
            return Err(Error::Bad(format!(
                "码本长度表在条目 {sym}（长 {len}）处放不下 —— 树已经满了"
            )));
        }
    }
    Ok((nodes, None))
}

/// 从 `node`（深度 `depth`）往下找**最左**的空位，在深度 `len` 挂一个叶子。
///
/// 「最左」= 码字最小：每层先试 0 分支，试不通（那里是叶子、或者子树已满）再试 1。
fn place(nodes: &mut Vec<[i32; 2]>, node: usize, depth: u8, len: u8, sym: i32) -> bool {
    // 走到这里说明 `node` 是个内部节点。它可能已经比 `len` 还深（前一条更长的
    // 码字在这棵子树里长出了更深的枝），那样本层放不下 —— 换分支，别 panic。
    if depth >= len {
        return false;
    }
    for b in 0..2usize {
        let child = nodes[node][b];
        if child == 0 {
            if depth + 1 == len {
                nodes[node][b] = -sym - 1;
                return true;
            }
            nodes.push([0, 0]);
            let idx = nodes.len() - 1;
            nodes[node][b] = idx as i32;
            if place(nodes, idx, depth + 1, len, sym) {
                return true;
            }
            nodes[node][b] = 0;
            nodes.truncate(idx);
        } else if child > 0 && place(nodes, child as usize, depth + 1, len, sym) {
            return true;
        }
    }
    false
}

/// `float32_unpack`：Vorbis 自己的 32 位浮点（不是 IEEE）。规范 9.2.2。
fn float32_unpack(x: u32) -> f32 {
    let mantissa = (x & 0x001f_ffff) as f64;
    let negative = x & 0x8000_0000 != 0;
    let exponent = ((x & 0x7fe0_0000) >> 21) as i32;
    let v = mantissa * 2f64.powi(exponent - 788);
    (if negative { -v } else { v }) as f32
}

/// `lookup1_values(entries, dimensions)`：最大的 `v` 使 `v^dimensions <= entries`。
/// 规范 9.2.3。
fn lookup1_values(entries: usize, dimensions: usize) -> usize {
    if dimensions == 0 {
        return 0;
    }
    let mut v = 1usize;
    loop {
        // 用 checked 乘法判越界，别让 v^dimensions 溢出成「居然没超」。
        let mut p = 1usize;
        let mut ok = true;
        for _ in 0..dimensions {
            match p.checked_mul(v + 1) {
                Some(x) if x <= entries => p = x,
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            return v;
        }
        v += 1;
    }
}

/// 读 VQ 查表并展开成 `entries × dimensions` 的 float。规范 3.2.2。
fn read_vq(bit: &mut BitReader, lookup_type: u32, entries: usize, dimensions: usize) -> Result<Vq> {
    let minimum = float32_unpack(bit.read(32)?);
    let delta = float32_unpack(bit.read(32)?);
    let value_bits = bit.read(4)? + 1;
    let sequence_p = bit.read_bool()?;
    let lookup_values = if lookup_type == 1 {
        lookup1_values(entries, dimensions)
    } else {
        entries * dimensions
    };
    if lookup_values == 0 || lookup_values > (1 << 22) {
        return Err(Error::Unsupported(format!(
            "VQ 查表有 {lookup_values} 个值（超出本实现的资源上限）"
        )));
    }
    let mut multiplicands = Vec::with_capacity(lookup_values);
    for _ in 0..lookup_values {
        multiplicands.push(bit.read(value_bits)?);
    }

    let mut values = vec![0f32; entries * dimensions];
    if lookup_type == 1 {
        // 格点 VQ：每个维度从同一个乘法表里按 lookup_values 进制取一位。
        let mut divisor = 1usize;
        for i in 0..dimensions {
            for e in 0..entries {
                let off = (e / divisor) % lookup_values;
                let mut v = multiplicands[off] as f32 * delta + minimum;
                if i > 0 && sequence_p {
                    v += values[e * dimensions + i - 1];
                }
                values[e * dimensions + i] = v;
            }
            divisor = divisor.saturating_mul(lookup_values);
        }
    } else {
        // 显式 VQ：乘法表就是按向量逐元素排的。
        for e in 0..entries {
            let mut last = 0f32;
            for i in 0..dimensions {
                let v = multiplicands[e * dimensions + i] as f32 * delta + minimum + last;
                values[e * dimensions + i] = v;
                if sequence_p {
                    last = v;
                }
            }
        }
    }
    Ok(Vq { dimensions, values })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 规范 3.2.1 的例子：长度 `2,4,4,4,4,2,3,3` → 码字
    /// `00 / 0100 / 0101 / 0110 / 0111 / 10 / 110 / 111`。
    ///
    /// 这条**专门用来钉住「不是 canonical」**：canonical 会给条目 5 分 `01`，
    /// 而 `01` 是条目 1..4 的前缀。
    #[test]
    fn spec_huffman_assignment() {
        let lengths = [2u8, 4, 4, 4, 4, 2, 3, 3];
        let (nodes, single) = build_tree(&lengths).unwrap();
        assert!(single.is_none());

        let walk = |bits: &[u8]| -> usize {
            let mut node = 0usize;
            for &b in bits {
                let next = nodes[node][b as usize];
                if next < 0 {
                    return (-next - 1) as usize;
                }
                node = next as usize;
            }
            panic!("码字没走完就到了内部节点");
        };
        assert_eq!(walk(&[0, 0]), 0);
        assert_eq!(walk(&[0, 1, 0, 0]), 1);
        assert_eq!(walk(&[0, 1, 0, 1]), 2);
        assert_eq!(walk(&[0, 1, 1, 0]), 3);
        assert_eq!(walk(&[0, 1, 1, 1]), 4);
        assert_eq!(walk(&[1, 0]), 5);
        assert_eq!(walk(&[1, 1, 0]), 6);
        assert_eq!(walk(&[1, 1, 1]), 7);
    }

    #[test]
    fn kraft_must_be_exact() {
        // 三个 2 位码字只填了 3/4，树没长完 → 报错
        assert!(build_tree(&[2, 2, 2]).is_err());
        // 一个 1 位 + 两个 2 位 = 恰好满
        assert!(build_tree(&[1, 2, 2]).is_ok());
        // 四个 1 位码字 = 过满
        assert!(build_tree(&[1, 1, 1, 1, 1]).is_err());
    }

    #[test]
    fn single_entry_codebook_is_special() {
        let (nodes, single) = build_tree(&[0, 1, 0]).unwrap();
        assert!(nodes.is_empty());
        assert_eq!(single, Some(1));
        // 码长不是 1 的单条目码本按规范拒绝
        assert!(build_tree(&[0, 2, 0]).is_err());
    }

    #[test]
    fn float32_unpack_matches_spec_formula() {
        // 值 = mantissa * 2^(exponent - 788)：mantissa 1、exponent 788 就是 1.0
        assert_eq!(float32_unpack(1 | (788 << 21)), 1.0);
        assert_eq!(float32_unpack(1 | (787 << 21)), 0.5);
        assert_eq!(float32_unpack(1 | (789 << 21)), 2.0);
        // 符号位
        assert_eq!(float32_unpack(0x8000_0000 | 1 | (788 << 21)), -1.0);
        // 大 mantissa：2^20 * 2^0
        assert_eq!(float32_unpack(0x0010_0000 | (788 << 21)), 1048576.0);
    }

    #[test]
    fn lookup1_values_is_the_largest_v() {
        assert_eq!(lookup1_values(1, 1), 1);
        assert_eq!(lookup1_values(8, 1), 8);
        assert_eq!(lookup1_values(8, 2), 2); // 2^2=4 <= 8, 3^2=9 > 8
        assert_eq!(lookup1_values(9, 2), 3);
        assert_eq!(lookup1_values(27, 3), 3);
        assert_eq!(lookup1_values(26, 3), 2);
        assert_eq!(lookup1_values(1024, 10), 2); // 2^10 = 1024 <= 1024
        assert_eq!(lookup1_values(1023, 10), 1); // 2^10 > 1023 → 1
    }
}
