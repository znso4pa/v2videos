//! residue：频谱残差。规范第 8 章。
//!
//! 三种格式共用同一套「分类 → 按分类挑码本 → 逐分区解 VQ」的骨架，
//! 只在**分区内怎么摆**上不同：
//!
//! - **type 0**：分区内按码本维度**交错**（`v[off + i + j*step]`）；
//! - **type 1**：分区内顺序摆；
//! - **type 2**：先把 `ch` 个向量交织成一条 `ch*n` 的长向量，按 type 1 解完再拆开。
//!
//! type 2 的「全部 do_not_decode 就一个都不解、只要有一个要解就全解」是一条
//! 独立的规则（规范 8.6.5），别按 0/1 的直觉写。

use crate::bit::BitReader;
use crate::codebook::Codebook;
use crate::error::{Error, Result};

pub struct Residue {
    /// 0 / 1 / 2。
    kind: u8,
    begin: usize,
    end: usize,
    partition_size: usize,
    classifications: usize,
    classbook: usize,
    /// `books[分类][pass]`，`-1` 表示这个 pass 不解。
    books: Vec<[i16; 8]>,
}

impl Residue {
    /// 读 residue 配置。规范 8.6.1。
    pub fn read(bit: &mut BitReader, codebooks: &[Codebook]) -> Result<Self> {
        let kind = bit.read(16)? as u8;
        if kind > 2 {
            return Err(Error::Unsupported(format!(
                "residue type {kind}（Vorbis I 只有 0 / 1 / 2）"
            )));
        }
        let begin = bit.read(24)? as usize;
        let end = bit.read(24)? as usize;
        let partition_size = bit.read(24)? as usize + 1;
        let classifications = bit.read(6)? as usize + 1;
        let classbook = bit.read(8)? as usize;
        if classbook >= codebooks.len() {
            return Err(Error::Bad(format!(
                "residue 的 classbook {classbook} 越界（共 {} 个码本）",
                codebooks.len()
            )));
        }
        let mut cascade = vec![0u8; classifications];
        for c in cascade.iter_mut() {
            let low = bit.read(3)? as u8;
            let high = if bit.read_bool()? {
                bit.read(5)? as u8
            } else {
                0
            };
            *c = (high << 3) | low;
        }
        let mut books = vec![[-1i16; 8]; classifications];
        for (c, cascade_bits) in cascade.iter().enumerate() {
            for (pass, slot) in books[c].iter_mut().enumerate() {
                if cascade_bits & (1 << pass) != 0 {
                    let b = bit.read(8)? as usize;
                    if b >= codebooks.len() {
                        return Err(Error::Bad(format!(
                            "residue 的码本 {b} 越界（共 {} 个码本）",
                            codebooks.len()
                        )));
                    }
                    if codebooks[b].vq.is_none() {
                        return Err(Error::Bad(format!(
                            "residue 用了没有 VQ 查表的码本 {b}"
                        )));
                    }
                    *slot = b as i16;
                }
            }
        }
        Ok(Self { kind, begin, end, partition_size, classifications, classbook, books })
    }

    pub fn kind(&self) -> u8 {
        self.kind
    }

    /// 解一组声道。`views[j]` 是第 j 个声道的谱切片（长度 = 块大小 / 2）。
    ///
    /// 包尾读完按规范算**正常**：保留已经解出来的部分，剩下的保持 0。
    pub fn decode_views(
        &self,
        bit: &mut BitReader,
        codebooks: &[Codebook],
        views: &mut [&mut [f32]],
        do_not_decode: &[bool],
        scratch: &mut Scratch,
    ) -> Result<()> {
        match self.decode_inner(bit, codebooks, views, do_not_decode, scratch) {
            Ok(()) | Err(Error::Eop) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn decode_inner(
        &self,
        bit: &mut BitReader,
        codebooks: &[Codebook],
        views: &mut [&mut [f32]],
        do_not_decode: &[bool],
        scratch: &mut Scratch,
    ) -> Result<()> {
        debug_assert_eq!(do_not_decode.len(), views.len());
        if self.kind == 2 {
            // 交织成一条长向量，按 type 1 解，最后拆回去。
            if do_not_decode.iter().all(|&x| x) {
                return Ok(()); // 全都不解 → 一个都不解
            }
            let ch = views.len();
            let n = views.first().map(|v| v.len()).unwrap_or(0);
            // 把交织缓冲「借出来」：`decode_core` 还要用 `scratch` 的另一半，
            // 直接同时借两个字段做不到，`mem::take` 是最省事的切分。
            let mut flat = std::mem::take(&mut scratch.flat);
            flat.clear();
            flat.resize(n * ch, 0.0);
            {
                let mut one: [&mut [f32]; 1] = [flat.as_mut_slice()];
                self.decode_core(bit, codebooks, &mut one, &[false], scratch)?;
            }
            for j in 0..ch {
                let dst = &mut views[j];
                for i in 0..n {
                    dst[i] = flat[i * ch + j];
                }
            }
            scratch.flat = flat;
            return Ok(());
        }
        self.decode_core(bit, codebooks, views, do_not_decode, scratch)
    }

    /// 骨架：分类码字 → 每个 pass 按分类挑码本 → 逐分区解 VQ。规范 8.6.2。
    fn decode_core(
        &self,
        bit: &mut BitReader,
        codebooks: &[Codebook],
        views: &mut [&mut [f32]],
        do_not_decode: &[bool],
        scratch: &mut Scratch,
    ) -> Result<()> {
        let ch = views.len();
        let actual_size = views.first().map(|v| v.len()).unwrap_or(0);
        let begin = self.begin.min(actual_size);
        let end = self.end.min(actual_size);
        let n_to_read = end.saturating_sub(begin);
        if n_to_read == 0 {
            return Ok(());
        }
        let partitions = n_to_read / self.partition_size;
        if partitions == 0 {
            return Ok(());
        }
        let classwords = codebooks[self.classbook].dimensions;
        if classwords == 0 {
            // 分类码本的维度是 0 的话，`partition_count` 永远不前进 —— 死循环。
            // 这种流是非法的，明确报出来，别挂在那儿。
            return Err(Error::Bad("分类码本的维度是 0".into()));
        }
        let classifications_per_vector = self.classifications as u32;

        // 分类号按「声道 × 分区」排成一条，免得每包分配 `Vec<Vec<u8>>`。
        let Scratch { classifications, partition, .. } = scratch;
        let need = ch * partitions;
        classifications.clear();
        classifications.resize(need, 0);

        for pass in 0..8 {
            let mut partition_count = 0usize;
            while partition_count < partitions {
                if pass == 0 {
                    for j in 0..ch {
                        if do_not_decode[j] {
                            continue;
                        }
                        let t = codebooks[self.classbook].decode_scalar(bit)? as u32;
                        // 一个分类码字里塞了 classwords 个分类号，**从高位往低位**取。
                        unpack_classifications(
                            t,
                            classwords,
                            classifications_per_vector,
                            partitions,
                            partition_count,
                            &mut classifications[j * partitions..(j + 1) * partitions],
                        );
                    }
                }
                for _ in 0..classwords {
                    if partition_count >= partitions {
                        break;
                    }
                    for j in 0..ch {
                        if do_not_decode[j] {
                            continue;
                        }
                        let vqclass = classifications[j * partitions + partition_count] as usize;
                        let vqbook = self.books[vqclass][pass];
                        if vqbook >= 0 {
                            let offset = begin + partition_count * self.partition_size;
                            let book = &codebooks[vqbook as usize];
                            decode_partition(self.kind, book, bit, views[j], offset, self.partition_size, partition)?;
                        }
                    }
                    partition_count += 1;
                }
            }
        }
        Ok(())
    }
}

/// 解码时的可复用暂存 —— 免得每包都分配。
///
/// 大小由**流参数**决定（声道数 / 分区数 / 分区大小），所以按需扩容就够；
/// 内容每包都被完整重写，不需要清零。
#[derive(Default)]
pub struct Scratch {
    /// `ch * partitions` 个分类号（按声道分行）。
    classifications: Vec<u8>,
    /// type 2 的交织缓冲（`ch * n`）。
    flat: Vec<f32>,
    /// 一个分区的转置缓冲（`<= partition_size`）。
    partition: Vec<f32>,
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// 把一个分类码字拆成 `classwords` 个分类号，写进**本声道**那一行 `dst`。
///
/// 规范 8.6.2 的伪码是「从 `classwords-1` 往 0 取、每步取一个低位」，
/// 也就是**最高位落在 `start`**、最低位落在 `start + classwords - 1`。
///
/// ⚠️ **`start + classwords` 会越过本声道的分区数** —— `partitions` 不一定是
/// `classwords` 的整数倍。越过的位置按规范**丢掉**，但低位**仍然要除**：
/// 少除一次，剩下的几个位置就整体错位了。
///
/// 这条边界不判就是写穿行尾：最后一个声道直接 panic（实测 libvorbis 编的 `.ogg`：
/// `index out of bounds: the len is 15 but the index is 15`），前面的声道则会踩进
/// 下一个声道的分类号 —— 后者被后写的覆盖掉，所以只有最后一个声道会喊疼。
#[inline]
fn unpack_classifications(
    mut word: u32,
    classwords: usize,
    classes: u32,
    partitions: usize,
    start: usize,
    dst: &mut [u8],
) {
    debug_assert_eq!(dst.len(), partitions);
    for i in (0..classwords).rev() {
        let cls = (word % classes) as u8;
        word /= classes;
        if start + i < partitions {
            dst[start + i] = cls;
        }
    }
}

/// 解一个分区（规范 8.6.3 / 8.6.4 的差别就在这里的下标）。
///
/// 两条路径都先把这一分区的分量按**位流顺序**解进 `tmp`，再整段相加 ——
/// 这样内层就是两条 `chunks` 迭代器对齐的循环，没有逐元素的分支。
fn decode_partition(
    kind: u8,
    book: &Codebook,
    bit: &mut BitReader,
    v: &mut [f32],
    offset: usize,
    n: usize,
    tmp: &mut Vec<f32>,
) -> Result<()> {
    let dim = book.dimensions;
    tmp.clear();
    if kind == 0 {
        let step = n / dim;
        for _ in 0..step {
            tmp.extend_from_slice(book.decode_vector(bit)?);
        }
        let span = step * dim;
        let end = (offset + span).min(v.len());
        if offset >= end {
            return Ok(()); // 整段都在谱外：位已经消费了，值丢弃
        }
        let dst = &mut v[offset..end];
        // 转置相加：第 j 段（长度 step）加上每个分量的第 j 个元素。
        for (j, out_chunk) in dst.chunks_mut(step).enumerate() {
            for (s, e) in out_chunk.iter_mut().zip(tmp.chunks_exact(dim)) {
                *s += e[j];
            }
        }
    } else {
        // type 1（type 2 也走这里）：分区内顺序摆。
        let mut i = 0usize;
        while i < n {
            let entry = book.decode_vector(bit)?;
            for j in 0..dim {
                let k = offset + i + j;
                if k < v.len() {
                    v[k] += entry[j];
                }
            }
            i += dim;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::unpack_classifications;

    /// 位序：最高位落在 `start`。
    ///
    /// `word = 5`、`classes = 3` → 三进制 `012`，所以三个位置依次是 0 / 1 / 2。
    /// 写反了（从 `start` 开始放低位）就是 `2 / 1 / 0` —— 分类号错位会让
    /// 后面的分区挑错码本，而**不会**报任何错。
    #[test]
    fn classification_digits_are_most_significant_first() {
        let mut dst = [0xffu8; 3];
        unpack_classifications(5, 3, 3, 3, 0, &mut dst);
        assert_eq!(dst, [0, 1, 2]);
    }

    /// 末尾那一组字只有一部分落得下 —— 落不下的**丢掉**，前面的不能动。
    ///
    /// 这就是那个 panic 的最小复现：`partitions = 4`、`classwords = 3`、
    /// `start = 3`，老代码会去写 `dst[4]` 和 `dst[5]`。
    #[test]
    fn classification_word_stops_at_the_last_partition() {
        let mut dst = [0xeeu8; 4];
        // word = 5 → 高位到低位 0 / 1 / 2，只有 i=0（start+0=3）落得下。
        unpack_classifications(5, 3, 3, 4, 3, &mut dst);
        assert_eq!(dst, [0xee, 0xee, 0xee, 0]);
    }

    /// `start` 正好等于 `partitions`（一个都放不下）时也不能越界。
    #[test]
    fn classification_word_can_be_entirely_out_of_range() {
        let mut dst = [0x11u8; 2];
        unpack_classifications(12345, 4, 3, 2, 2, &mut dst);
        assert_eq!(dst, [0x11, 0x11]);
    }
}
