//! setup 头：把码本 / floor / residue / mapping / mode 全部读出来。规范 4.2.4。

use crate::bit::BitReader;
use crate::codebook::Codebook;
use crate::error::{Error, Result};
use crate::floor::Floor1;
use crate::ilog;
use crate::residue::Residue;

pub enum Floor {
    Type1(Floor1),
}

/// 声道 → submap → (floor, residue) 的派发表，外加耦合步骤。
pub struct Mapping {
    pub submaps: usize,
    /// 每个声道落在哪个 submap（`submaps == 1` 时全是 0）。
    pub mux: Vec<u8>,
    pub submap_floor: Vec<u8>,
    pub submap_residue: Vec<u8>,
    pub magnitude: Vec<u8>,
    pub angle: Vec<u8>,
}

pub struct Mode {
    pub blockflag: bool,
    pub mapping: u8,
}

pub struct Setup {
    pub codebooks: Vec<Codebook>,
    pub floors: Vec<Floor>,
    pub residues: Vec<Residue>,
    pub mappings: Vec<Mapping>,
    pub modes: Vec<Mode>,
}

impl Setup {
    /// 解析 setup 头（第三个头包，`\x05vorbis` 开头那一包）。
    pub fn read(data: &[u8], channels: usize) -> Result<Self> {
        if data.len() < 7 {
            return Err(Error::Bad(format!("setup 头只有 {} 字节", data.len())));
        }
        if data[0] != 0x05 {
            return Err(Error::Bad("不是 setup 头（第一个字节不是 5）".into()));
        }
        if &data[1..7] != b"vorbis" {
            return Err(Error::Bad("setup 头缺 \"vorbis\" 标记".into()));
        }
        // 前 7 个字节是**字节对齐**的包头，位流从第 8 个字节开始。
        // 忘了跳这 7 个字节的话，第一个码本的同步字会读到 "vorbis" 的尾巴
        // （0x62726f = "bro"），报「同步字不对」—— 实测踩过。
        let mut bit = BitReader::new(data);
        bit.skip(7 * 8)?;

        // 码本
        let n = bit.read(8)? as usize + 1;
        let mut codebooks = Vec::with_capacity(n);
        for i in 0..n {
            codebooks.push(
                Codebook::read(&mut bit)
                    .map_err(|e| Error::Bad(format!("第 {i} 个码本：{e}")))?,
            );
        }

        // 时域变换（占位，必须全 0）
        let n = bit.read(6)? as usize + 1;
        for _ in 0..n {
            if bit.read(16)? != 0 {
                return Err(Error::Unsupported("时域变换不是 0（Vorbis I 只有 0）".into()));
            }
        }

        // floor
        let n = bit.read(6)? as usize + 1;
        let mut floors = Vec::with_capacity(n);
        for i in 0..n {
            let ty = bit.read(16)?;
            let f = match ty {
                0 => {
                    return Err(Error::Unsupported(
                        "floor type 0（LSP 表示）—— 本实现只做 floor 1，现存的编码器都用 floor 1"
                            .into(),
                    ))
                }
                1 => Floor::Type1(
                    Floor1::read(&mut bit, &codebooks)
                        .map_err(|e| Error::Bad(format!("第 {i} 个 floor：{e}")))?,
                ),
                t => {
                    return Err(Error::Unsupported(format!(
                        "floor type {t}（Vorbis I 只有 0 / 1）"
                    )))
                }
            };
            floors.push(f);
        }

        // residue
        let n = bit.read(6)? as usize + 1;
        let mut residues = Vec::with_capacity(n);
        for i in 0..n {
            residues.push(
                Residue::read(&mut bit, &codebooks)
                    .map_err(|e| Error::Bad(format!("第 {i} 个 residue：{e}")))?,
            );
        }

        // mapping
        let n = bit.read(6)? as usize + 1;
        let mut mappings = Vec::with_capacity(n);
        for i in 0..n {
            mappings.push(read_mapping(
                &mut bit,
                channels,
                floors.len(),
                residues.len(),
            ).map_err(|e| Error::Bad(format!("第 {i} 个 mapping：{e}")))?);
        }

        // mode
        let n = bit.read(6)? as usize + 1;
        let mut modes = Vec::with_capacity(n);
        for i in 0..n {
            let blockflag = bit.read_bool()?;
            let windowtype = bit.read(16)?;
            let transformtype = bit.read(16)?;
            let mapping = bit.read(8)? as u8;
            if windowtype != 0 || transformtype != 0 {
                return Err(Error::Unsupported(format!(
                    "mode {i} 的窗/变换类型不是 0（{windowtype}/{transformtype}）"
                )));
            }
            if mapping as usize >= mappings.len() {
                return Err(Error::Bad(format!(
                    "mode {i} 的 mapping {mapping} 越界（共 {} 个）",
                    mappings.len()
                )));
            }
            modes.push(Mode { blockflag, mapping });
        }

        if !bit.read_bool()? {
            return Err(Error::Bad("setup 头的 framing 位不是 1".into()));
        }

        Ok(Self { codebooks, floors, residues, mappings, modes })
    }
}

fn read_mapping(
    bit: &mut BitReader,
    channels: usize,
    floors: usize,
    residues: usize,
) -> Result<Mapping> {
    let ty = bit.read(16)?;
    if ty != 0 {
        return Err(Error::Unsupported(format!("mapping type {ty}（Vorbis I 只有 0）")));
    }
    let submaps = if bit.read_bool()? {
        bit.read(4)? as usize + 1
    } else {
        1
    };
    let mut magnitude = Vec::new();
    let mut angle = Vec::new();
    if bit.read_bool()? {
        let steps = bit.read(8)? as usize + 1;
        let bits = ilog(channels as u32 - 1);
        for _ in 0..steps {
            let m = bit.read(bits)? as u8;
            let a = bit.read(bits)? as u8;
            if a == m || m as usize >= channels || a as usize >= channels {
                return Err(Error::Bad(format!(
                    "耦合步骤不合法（magnitude {m}, angle {a}, 声道数 {channels}）"
                )));
            }
            magnitude.push(m);
            angle.push(a);
        }
    }
    let reserved = bit.read(2)?;
    if reserved != 0 {
        return Err(Error::Bad(format!("mapping 的保留位是 {reserved}（应为 0）")));
    }
    let mut mux = vec![0u8; channels];
    if submaps > 1 {
        for m in mux.iter_mut() {
            *m = bit.read(4)? as u8;
            if *m as usize >= submaps {
                return Err(Error::Bad(format!(
                    "声道 mux {} 超过 submap 数 {submaps}",
                    *m
                )));
            }
        }
    }
    let mut submap_floor = Vec::with_capacity(submaps);
    let mut submap_residue = Vec::with_capacity(submaps);
    for _ in 0..submaps {
        bit.read(8)?; // 时域配置占位
        let f = bit.read(8)? as u8;
        if f as usize >= floors {
            return Err(Error::Bad(format!("submap 的 floor {f} 越界（共 {floors} 个）")));
        }
        let r = bit.read(8)? as u8;
        if r as usize >= residues {
            return Err(Error::Bad(format!(
                "submap 的 residue {r} 越界（共 {residues} 个）"
            )));
        }
        submap_floor.push(f);
        submap_residue.push(r);
    }
    Ok(Mapping { submaps, mux, submap_floor, submap_residue, magnitude, angle })
}
