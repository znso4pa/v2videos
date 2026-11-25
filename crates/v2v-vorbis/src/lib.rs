//! 纯 Rust 的 **Vorbis I 解码**。
//!
//! 链路：`setup 头 → 码本 / floor / residue / mapping / mode`，
//! 然后每个音频包：`mode → 窗 → floor → residue → 逆耦合 → floor×residue → IMDCT → 叠加`。
//!
//! ```text
//!   Ogg 包 ──► [本 crate] ──► 逐声道 f32 PCM
//! ```
//!
//! 无 C、无 FFI；唯一依赖是本仓的共享数值内核（`v2v-kernel` 的 MDCT，与 AAC / WMA 同一份）。
//!
//! ## 为什么这个 crate 是 MIT
//!
//! Vorbis 的**全部**概率模型都在码流里（码本、floor、residue 的配置逐流携带），
//! 本实现没有用任何 FFmpeg 派生的码表 —— 唯一的常量表是 floor1 的反 dB 表，
//! 它是规范 10.1 那张表的**公式**（`10^((i-255)·7/256)`），见 `floor::inverse_db_table`。
//! 所以它和 `tables` 那一族的 LGPL 边界无关。
//!
//! ## 支持范围
//!
//! - ✅ floor 1、residue 0 / 1 / 2、声道耦合、长短块与 hybrid 窗、单条目码本
//! - ❌ **floor 0**（LSP）：setup 阶段明确报 `Unsupported` —— 现存编码器都用 floor 1
//! - ❌ 链式（chained）Ogg 的时间轴拼接：那是容器层的活，本 crate 只看单条逻辑流
//!
//! ## 输出的声道顺序
//!
//! 按**码流顺序**原样输出，不做 WAV 顺序重排（规范 4.3.9 描述的正是码流顺序）。
//! 需要别的顺序由调用方重排。

mod bit;
mod codebook;
mod error;
mod floor;
mod residue;
mod setup;
mod synth;

pub use error::{Error, Result};
pub use setup::{Floor, Mapping, Mode, Setup};

use bit::BitReader;
use floor::Floor1Packet;
use synth::{build_window, ChannelSynth};
use v2v_kernel::ImdctPlan;

/// 规范 9.1.1 的 `ilog`，同时供子模块以 `crate::ilog` 取用。
pub(crate) use error::ilog;

/// 识别头里的流参数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Info {
    pub channels: u8,
    pub sample_rate: u32,
    pub blocksize_0: usize,
    pub blocksize_1: usize,
    pub bitrate_nominal: i32,
}

impl Info {
    /// 解析识别头（`\x01vorbis` 开头那一包，30 字节）。
    pub fn parse(id: &[u8]) -> Result<Self> {
        if id.len() < 30 {
            return Err(Error::Bad(format!("识别头只有 {} 字节", id.len())));
        }
        if id[0] != 1 || &id[1..7] != b"vorbis" {
            return Err(Error::Bad("不是 Vorbis 识别头".into()));
        }
        let version = u32::from_le_bytes(id[7..11].try_into().unwrap());
        if version != 0 {
            return Err(Error::Unsupported(format!("Vorbis 版本 {version}（只认 0）")));
        }
        let channels = id[11];
        let sample_rate = u32::from_le_bytes(id[12..16].try_into().unwrap());
        let bitrate_nominal = i32::from_le_bytes(id[20..24].try_into().unwrap());
        let bs = id[28];
        let blocksize_0 = 1usize << (bs & 0x0F);
        let blocksize_1 = 1usize << (bs >> 4);
        if channels == 0 || sample_rate == 0 {
            return Err(Error::Bad("识别头里声道数或采样率为 0".into()));
        }
        if id[29] & 1 == 0 {
            return Err(Error::Bad("识别头的 framing 位不是 1".into()));
        }
        if blocksize_0 > blocksize_1 || blocksize_1 > 8192 {
            return Err(Error::Bad(format!(
                "块大小不合法（{blocksize_0} / {blocksize_1}）"
            )));
        }
        Ok(Self { channels, sample_rate, blocksize_0, blocksize_1, bitrate_nominal })
    }
}

/// Vorbis 解码器。
pub struct Decoder {
    info: Info,
    setup: Setup,
    /// 短块 / 长块的 IMDCT 计划，按需建。
    plans: [Option<ImdctPlan>; 2],
    synths: Vec<ChannelSynth>,
    /// 每个声道的谱（长度 = 块大小 / 2）。
    spectra: Vec<Vec<f32>>,
    /// 每个声道的 floor 解码结果（`None` = 本帧未使用）。
    floor_pkts: Vec<Option<Floor1Packet>>,
}

impl Decoder {
    /// 用识别头与 setup 头建解码器。注释头用不上，不必传。
    pub fn new(id_header: &[u8], setup_header: &[u8]) -> Result<Self> {
        let info = Info::parse(id_header)?;
        let setup = Setup::read(setup_header, info.channels as usize)?;
        let ch = info.channels as usize;
        Ok(Self {
            info,
            setup,
            plans: [None, None],
            synths: (0..ch).map(|_| ChannelSynth::new()).collect(),
            spectra: vec![Vec::new(); ch],
            floor_pkts: (0..ch).map(|_| None).collect(),
        })
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    pub fn channels(&self) -> usize {
        self.info.channels as usize
    }

    pub fn sample_rate(&self) -> u32 {
        self.info.sample_rate
    }

    /// 丢掉重叠状态。跳转（seek）或换一条逻辑流之后**必须**调：
    /// 否则新位置的第一块会和上一段的尾巴相加，听上去是一声爆音。
    pub fn reset(&mut self) {
        for s in &mut self.synths {
            s.reset();
        }
    }

    /// 解一个音频包，返回**逐声道**的 PCM。
    ///
    /// 长度可能为 0：第一个包只用来 priming（规范 4.3.8「Data is not returned
    /// from the first frame」），之后每包输出 `上一块/4 + 本块/4` 个样本。
    /// 包损坏（不是音频包、头部读到包尾）时返回空的 `Vec`，不报错 —— 规范要求
    /// 「忽略这个包，不尝试解成音频」。
    pub fn decode(&mut self, packet: &[u8]) -> Result<Vec<Vec<f32>>> {
        let mut bit = BitReader::new(packet);
        let Some(head) = read_packet_head(&mut bit, &self.setup)? else {
            return Ok(Vec::new());
        };

        let ch = self.channels();
        let n = if head.long { self.info.blocksize_1 } else { self.info.blocksize_0 };
        let half = n / 2;
        let window = build_window(
            n,
            self.info.blocksize_0,
            head.long,
            head.prev_flag,
            head.next_flag,
        );
        let mapping = &self.setup.mappings[self.setup.modes[head.mode].mapping as usize];

        // 1) floor：按声道顺序解。
        let mut no_residue = vec![false; ch];
        for i in 0..ch {
            let submap = mapping.mux[i] as usize;
            let Floor::Type1(f) = &self.setup.floors[mapping.submap_floor[submap] as usize];
            let pkt = f.decode(&mut bit, &self.setup.codebooks)?;
            no_residue[i] = pkt.is_none();
            self.floor_pkts[i] = pkt;
        }

        // 2) 未使用的声道经耦合会变成「有内容」——两个都得解。
        for i in 0..mapping.magnitude.len() {
            let m = mapping.magnitude[i] as usize;
            let a = mapping.angle[i] as usize;
            if !no_residue[m] || !no_residue[a] {
                no_residue[m] = false;
                no_residue[a] = false;
            }
        }

        // 3) residue：按 submap 顺序。
        for s in 0..mapping.submaps {
            let chans: Vec<usize> = (0..ch).filter(|&j| mapping.mux[j] as usize == s).collect();
            if chans.is_empty() {
                continue;
            }
            let residue_index = mapping.submap_residue[s] as usize;
            let residue = &self.setup.residues[residue_index];
            let mut out: Vec<Vec<f32>> = chans.iter().map(|_| vec![0f32; half]).collect();
            let dnd: Vec<bool> = chans.iter().map(|&j| no_residue[j]).collect();
            residue.decode(&mut bit, &self.setup.codebooks, &mut out, &dnd)?;
            for (k, &j) in chans.iter().enumerate() {
                self.spectra[j] = std::mem::take(&mut out[k]);
            }
        }

        // 4) 逆耦合（从后往前）。
        for i in (0..mapping.magnitude.len()).rev() {
            let m = mapping.magnitude[i] as usize;
            let a = mapping.angle[i] as usize;
            for k in 0..half {
                let mv = self.spectra[m][k];
                let av = self.spectra[a][k];
                let (nm, na) = if mv > 0.0 {
                    if av > 0.0 {
                        (mv, mv - av)
                    } else {
                        (mv + av, mv)
                    }
                } else if av > 0.0 {
                    (mv, mv + av)
                } else {
                    (mv - av, mv)
                };
                self.spectra[m][k] = nm;
                self.spectra[a][k] = na;
            }
        }

        // 5) floor 曲线 × residue → 谱 → IMDCT → 叠加。
        let bs0 = self.info.blocksize_0;
        let mut curve = vec![0f32; half];
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(ch);
        for i in 0..ch {
            curve.fill(0.0);
            if let Some(pkt) = &self.floor_pkts[i] {
                let submap = mapping.mux[i] as usize;
                let Floor::Type1(f) = &self.setup.floors[mapping.submap_floor[submap] as usize];
                f.curve(pkt, &mut curve);
            }
            for (c, s) in curve.iter_mut().zip(&self.spectra[i]) {
                *c *= s;
            }
            let plan = plan_for(&mut self.plans, half, bs0);
            out.push(self.synths[i].synthesize(plan, &curve, &window));
        }
        Ok(out)
    }
}

/// 一个音频包的包头（规范 4.3.1）。
struct PacketHead {
    mode: usize,
    long: bool,
    prev_flag: bool,
    next_flag: bool,
}

/// 读音频包头。`Ok(None)` = 这一包不是音频包，按规范丢弃。
fn read_packet_head(bit: &mut BitReader, setup: &Setup) -> Result<Option<PacketHead>> {
    let packet_type = match bit.read(1) {
        Ok(v) => v,
        Err(Error::Eop) => return Ok(None),
        Err(e) => return Err(e),
    };
    // 规范 4.3.1：音频包的包类型位必须是 0。别的值（含 1）是别的包类型。
    if packet_type != 0 {
        return Ok(None);
    }
    let mode_bits = ilog(setup.modes.len() as u32 - 1);
    let mode = match bit.read(mode_bits) {
        Ok(v) => v as usize,
        Err(Error::Eop) => return Ok(None),
        Err(e) => return Err(e),
    };
    let Some(m) = setup.modes.get(mode) else {
        return Err(Error::Bad(format!(
            "模式号 {mode} 越界（setup 里只有 {} 个）",
            setup.modes.len()
        )));
    };
    let (prev_flag, next_flag) = if m.blockflag {
        let prev = match bit.read_bool() {
            Ok(v) => v,
            Err(Error::Eop) => return Ok(None),
            Err(e) => return Err(e),
        };
        let next = match bit.read_bool() {
            Ok(v) => v,
            Err(Error::Eop) => return Ok(None),
            Err(e) => return Err(e),
        };
        (prev, next)
    } else {
        // 短块没有窗标志位，规范规定两边都按「不平滑」处理。
        (true, true)
    };
    Ok(Some(PacketHead { mode, long: m.blockflag, prev_flag, next_flag }))
}

/// 取（或按需建）短块 / 长块的 IMDCT 计划。
///
/// ## 定标是 1.0，**没有** `2/N`
///
/// 教科书的 IMDCT 带 `2/M`（`M = n/2`）才是正交归一的一对，所以这里很容易顺手
/// 乘上 `4/n` —— **那是错的**。Vorbis 的逆变换是**未归一化**的转置：规范 4.3.7
/// 只说「用逆 MDCT」，把定标留给了参考文献和参考实现。
///
/// 判据不是推出来的，是量出来的：拿真样本逐帧比幅度，`ref / ours` 恒等于
/// **512 = n/4**（长块 2048），波形相关系数 1.0000 —— 也就是说我们多乘了
/// `4/n` 这个因子。乘回 `n/4` 就等于 1。
fn plan_for<'a>(
    plans: &'a mut [Option<ImdctPlan>; 2],
    half: usize,
    blocksize_0: usize,
) -> &'a mut ImdctPlan {
    let slot = if half == blocksize_0 / 2 { 0 } else { 1 };
    plans[slot].get_or_insert_with(|| ImdctPlan::new(half, 1.0))
}

/// 把逐声道 f32 转成交织的 s16（截断到 `[-1, 1)`，四舍五入）。
///
/// 给调用方（CLI / 测试）用：解码器本身只出 f32，**不做量化决策**。
pub fn planar_to_s16(planar: &[Vec<f32>]) -> Vec<i16> {
    let ch = planar.len();
    let n = planar.first().map(|v| v.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        for v in planar.iter() {
            let x = v[i];
            let s = if x >= 1.0 {
                32767.0
            } else if x <= -1.0 {
                -32768.0
            } else {
                (x * 32768.0).round()
            };
            out.push(s as i16);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_rejects_garbage() {
        assert!(Info::parse(&[0u8; 30]).is_err());
        let mut id = vec![0u8; 30];
        id[0] = 1;
        id[1..7].copy_from_slice(b"vorbis");
        id[11] = 2;
        id[12..16].copy_from_slice(&44100u32.to_le_bytes());
        id[28] = 0xBB; // bs0 = 11, bs1 = 11
        id[29] = 1;
        let info = Info::parse(&id).unwrap();
        assert_eq!(info.channels, 2);
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.blocksize_0, 2048);
        assert_eq!(info.blocksize_1, 2048);
    }

    #[test]
    fn planar_to_s16_clamps_and_rounds() {
        let planar = vec![vec![0.0, 0.5, 1.0, -1.0, 2.0, -2.0]];
        assert_eq!(planar_to_s16(&planar), vec![0, 16384, 32767, -32768, 32767, -32768]);
    }
}
