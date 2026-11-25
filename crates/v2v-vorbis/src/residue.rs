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

    /// 解一组向量。`out[j]` 是第 j 个声道在本 submap 里的谱（长度 = 块大小 / 2）。
    ///
    /// 包尾读完按规范算**正常**：保留已经解出来的部分，剩下的保持 0。
    pub fn decode(
        &self,
        bit: &mut BitReader,
        codebooks: &[Codebook],
        out: &mut [Vec<f32>],
        do_not_decode: &[bool],
    ) -> Result<()> {
        match self.decode_inner(bit, codebooks, out, do_not_decode) {
            Ok(()) | Err(Error::Eop) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn decode_inner(
        &self,
        bit: &mut BitReader,
        codebooks: &[Codebook],
        out: &mut [Vec<f32>],
        do_not_decode: &[bool],
    ) -> Result<()> {
        debug_assert_eq!(do_not_decode.len(), out.len());
        if self.kind == 2 {
            // 交织成一条长向量，按 type 1 解，最后拆回去。
            if do_not_decode.iter().all(|&x| x) {
                return Ok(()); // 全都不解 → 一个都不解
            }
            let ch = out.len();
            let n = out.first().map(|v| v.len()).unwrap_or(0);
            let mut flat = vec![0f32; n * ch];
            {
                let mut one: [&mut [f32]; 1] = [flat.as_mut_slice()];
                self.decode_core(bit, codebooks, &mut one, &[false])?;
            }
            for j in 0..ch {
                for i in 0..n {
                    out[j][i] = flat[i * ch + j];
                }
            }
            return Ok(());
        }
        let mut views: Vec<&mut [f32]> = out.iter_mut().map(|v| v.as_mut_slice()).collect();
        self.decode_core(bit, codebooks, &mut views, do_not_decode)
    }

    /// 骨架：分类码字 → 每个 pass 按分类挑码本 → 逐分区解 VQ。规范 8.6.2。
    fn decode_core(
        &self,
        bit: &mut BitReader,
        codebooks: &[Codebook],
        out: &mut [&mut [f32]],
        do_not_decode: &[bool],
    ) -> Result<()> {
        let ch = out.len();
        let actual_size = out.first().map(|v| v.len()).unwrap_or(0);
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
        let classifications_per_vector = self.classifications as u32;
        let mut classifications = vec![vec![0u8; partitions]; ch];

        for pass in 0..8 {
            let mut partition_count = 0usize;
            while partition_count < partitions {
                if pass == 0 {
                    for j in 0..ch {
                        if do_not_decode[j] {
                            continue;
                        }
                        let mut t = codebooks[self.classbook].decode_scalar(bit)? as u32;
                        // 一个分类码字里塞了 classwords 个分类号，**从高位往低位**取。
                        for i in (0..classwords).rev() {
                            classifications[j][i + partition_count] =
                                (t % classifications_per_vector) as u8;
                            t /= classifications_per_vector;
                        }
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
                        let vqclass = classifications[j][partition_count] as usize;
                        let vqbook = self.books[vqclass][pass];
                        if vqbook >= 0 {
                            let offset = begin + partition_count * self.partition_size;
                            let book = &codebooks[vqbook as usize];
                            self.decode_partition(book, bit, out[j], offset)?;
                        }
                    }
                    partition_count += 1;
                }
            }
        }
        Ok(())
    }

    /// 解一个分区（规范 8.6.3 / 8.6.4 的差别就在这里的下标）。
    fn decode_partition(
        &self,
        book: &Codebook,
        bit: &mut BitReader,
        v: &mut [f32],
        offset: usize,
    ) -> Result<()> {
        let dim = book.dimensions;
        let n = self.partition_size;
        if self.kind == 0 {
            let step = n / dim;
            for i in 0..step {
                let entry = book.decode_vector(bit)?;
                for j in 0..dim {
                    let k = offset + i + j * step;
                    if k < v.len() {
                        v[k] += entry[j];
                    }
                }
            }
        } else {
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
}
