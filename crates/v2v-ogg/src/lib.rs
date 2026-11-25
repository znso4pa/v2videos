//! 纯 Rust 的 **Ogg 解封装**。
//!
//! 与 `v2v-asf` / `v2v-avi` / `v2v-mp4` 同一层：把容器拆开，逐条吐出裸样本
//! （`v2v_container::Sample`）。上层解码器不必知道数据是从 Ogg 来的。
//!
//! ```text
//!   .ogg / .oga ──► [本 crate] ──► Sample { stream, key_frame, timestamp_ms, data }
//! ```
//!
//! 无 C、无 FFI、零依赖（除共用的 `v2v-container`）。
//!
//! 读路径**自带缓冲**（[`READ_BUFFER_SIZE`]，256 KiB）：一页是「页头 + 段表 + 页体」
//! 三次读，未缓冲时一页就撞三次系统调用。调用方**不需要**再套 `BufReader`
//! （套了只是多一次拷贝，结果一样）。
//!
//! ## 支持范围
//!
//! - **页层**：完整。多逻辑流（按 `serial` 分）同时可读、跨页续接的包、CRC 校验
//!   （可开）、截断与坏页明确报错。
//! - **流识别**：`\x01vorbis`（Vorbis）、`OpusHead`、`\x80theora`、`fLaC`、
//!   `Speex   `、`fishead`（Skeleton）。**只有 Vorbis 会真的往下解**，
//!   别的识别出来之后明确报 [`Error::Unsupported`]，不假装支持。
//! - **不支持**：链式（chained）Ogg 的时间轴拼接 —— 第二个 `BOS` 之后时间戳会
//!   从 0 重来，本实现照原样透出（不伪造连续时间轴）。
//!
//! ## 时间戳为什么需要外部喂「包时长」
//!
//! Ogg 只在**页**这一级给 `granule_position`（Vorbis 下是 PCM 采样数），
//! 而一页里通常有几十个包，**包自己不带时长**。要拆出每包的时间戳，必须知道
//! 「这个包推进了多少采样」—— 那是编解码器的知识：Vorbis 的推进量由**本包与
//! 上一包的块大小**决定，而块大小藏在 setup 头里。
//!
//! 所以这里定了个 [`PacketTiming`] 口子：容器只管拼包，时长由 `vorbis` crate
//! 实现。这样 `v2v-ogg` 不依赖任何编解码器，也不会为了读时间戳去把 setup 头
//! 解析第二遍。
//!
//! 时间戳口径与 `ffprobe` 对齐（实测 `.ogg` 逐包核对过）：
//!
//! - 第 1 个音频包**不产生输出**，它的 `pts = -块大小/2`（44.1 kHz 长块下是 −1024）；
//! - 之后每包 `pts` 顺推 `(上一包块大小 + 本包块大小) / 4`；
//! - 页末用 `granule_position` **校正累加器**（不改已经发出的 `pts`）——
//!   文件最后一包常被截短（`ffprobe` 报 `Skip Samples`），校正后不会再影响后续。

mod crc;
mod error;
mod mux;
mod page;

pub use crc::{crc32, crc32_page};
pub use error::{Error, Result};
pub use mux::{OggMuxer, DEFAULT_PAGE_PACKETS};
pub use page::{
    Page, PageFlags, PageReader, CAPTURE, GRANULE_NONE, HEADER_SIZE, MAX_BODY, READ_BUFFER_SIZE,
};
pub use v2v_container::{fourcc_string, AudioFormat, Sample, Stream, StreamKind, VideoFormat};

use std::collections::HashMap;
use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom};

/// AVI 里 Vorbis 用的 `wFormatTag`（Vorbis I 规范「AVI 封装」那一节定的），
/// 借它给 `AudioFormat` 一个可辨认的标识 —— Ogg 本身没有 WAVE 头。
pub const VORBIS_FORMAT_TAG: u16 = 0x674F;

/// 一个已重组的包。
#[derive(Debug, Clone)]
pub struct Packet {
    pub data: Vec<u8>,
    pub serial: u32,
    /// 这个包所在页的序号。
    pub page_sequence: u32,
    /// 这个包所在页的 `granule_position`。
    ///
    /// **只对该页最后一个完成的包有意义**（规范就是这么定的）：
    /// 一页里有 N 个包时，`granule` 描述的是第 N 个包的结束位置，
    /// 前面 N−1 个包的结束位置要自己往前推。
    pub page_granule: i64,
    pub bos: bool,
    pub eos: bool,
}

/// 包重组：把页拼成包。
///
/// 跨页续接靠 `continued` 标志 + 每 `serial` 一份半截包缓冲。
pub struct PacketReader<R> {
    pages: PageReader<R>,
    /// 每个 `serial` 上一页留下的半截包。
    partial: HashMap<u32, Vec<u8>>,
    /// 每个 `serial` 上一页的序号，用来发现缺页。
    last_seq: HashMap<u32, u32>,
    queue: VecDeque<Packet>,
    /// 因为缺页/标志不符而丢掉的半截包个数。
    discarded: u64,
    /// 页序号不连续（缺页）的次数。
    gaps: u64,
    eof: bool,
}

impl<R: Read> PacketReader<R> {
    pub fn new(r: R) -> Self {
        Self {
            pages: PageReader::new(r),
            partial: HashMap::new(),
            last_seq: HashMap::new(),
            queue: VecDeque::new(),
            discarded: 0,
            gaps: 0,
            eof: false,
        }
    }

    pub fn set_verify_crc(&mut self, on: bool) {
        self.pages.set_verify_crc(on);
    }

    pub fn crc_errors(&self) -> u64 {
        self.pages.crc_errors()
    }

    /// 丢掉过多少个半截包（缺页、或 `continued` 标志与实际情况不符）。
    pub fn discarded_partials(&self) -> u64 {
        self.discarded
    }

    /// 页序号跳变的次数。> 0 说明文件有缺页，音频会缺一截。
    pub fn sequence_gaps(&self) -> u64 {
        self.gaps
    }

    /// 下一个包；到文件尾返回 `None`。
    pub fn next_packet(&mut self) -> Result<Option<Packet>> {
        loop {
            if let Some(p) = self.queue.pop_front() {
                return Ok(Some(p));
            }
            if self.eof {
                return Ok(None);
            }
            self.fill()?;
        }
    }

    /// 读一页并拆成包填进队列。
    fn fill(&mut self) -> Result<()> {
        let page = match self.pages.next_page()? {
            Some(p) => p,
            None => {
                self.eof = true;
                // 文件尾还挂着半截包 → 那是被截断的流。
                for (_, left) in self.partial.drain() {
                    if !left.is_empty() {
                        self.discarded += 1;
                    }
                }
                return Ok(());
            }
        };

        // 缺页检测：同一 serial 的页序号应当连续。
        if let Some(&prev) = self.last_seq.get(&page.serial) {
            if page.sequence != prev.wrapping_add(1) {
                self.gaps += 1;
            }
        }
        self.last_seq.insert(page.serial, page.sequence);

        let mut prefix = self.partial.remove(&page.serial).unwrap_or_default();

        // 标志与实际不符的两种情形，都要把半截包丢掉：
        // - 本页 `continued` 但我们手里没有前缀 → 前面的页丢了；
        // - 本页不 `continued` 但手里有前缀 → 上一页的包永远等不到结尾了。
        let mut drop_first = false;
        if page.flags.continued() && prefix.is_empty() {
            drop_first = true;
        } else if !page.flags.continued() && !prefix.is_empty() {
            self.discarded += 1;
            prefix.clear();
        }

        let packets = page::split_packets(&page, &mut prefix);
        let n = packets.len();
        for (i, data) in packets.into_iter().enumerate() {
            if i == 0 && drop_first {
                // 这个「包」其实是某个包的后半截，解出来只会是噪声。
                self.discarded += 1;
                continue;
            }
            self.queue.push_back(Packet {
                data,
                serial: page.serial,
                page_sequence: page.sequence,
                page_granule: page.granule,
                bos: page.flags.bos() && i == 0,
                eos: page.flags.eos() && i + 1 == n,
            });
        }

        self.partial.insert(page.serial, prefix);
        Ok(())
    }
}

/// 「这个包推进了多少采样」——由编解码器提供。
///
/// 实现方必须对**每个**音频包都给得出答案；给不出（比如包是坏的）返回 `None`，
/// 容器层会保持时间戳不动并记一次 [`OggDemuxer::timing_misses`]。
pub trait PacketTiming {
    /// `prev` 是上一个音频包（第一个包时 `prev == cur`）。
    fn advance(&self, prev: &[u8], cur: &[u8]) -> Option<u32>;
}

/// 逻辑流里的编码类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Vorbis,
    Opus,
    Theora,
    Flac,
    Speex,
    /// Skeleton（元数据流，不含音频/视频）。
    Skeleton,
    Unknown,
}

impl Codec {
    fn identify(packet: &[u8]) -> Codec {
        if packet.starts_with(b"\x01vorbis") {
            Codec::Vorbis
        } else if packet.starts_with(b"OpusHead") {
            Codec::Opus
        } else if packet.starts_with(b"\x80theora") {
            Codec::Theora
        } else if packet.starts_with(b"fLaC") {
            Codec::Flac
        } else if packet.starts_with(b"Speex   ") {
            Codec::Speex
        } else if packet.starts_with(b"fishead\0") {
            Codec::Skeleton
        } else {
            Codec::Unknown
        }
    }

    /// 这种流的头一共有几个包（含识别头本身）。
    fn header_packets(self) -> usize {
        match self {
            // 识别头 + 注释头 + setup 头
            Codec::Vorbis => 3,
            // OpusHead + OpusTags
            Codec::Opus => 2,
            // 识别头 + 注释头 + setup 头
            Codec::Theora => 3,
            // `fLaC` 头包 + 若干 metadata block（数量由 block 头里的「最后一包」标志定）
            Codec::Flac => 1,
            Codec::Speex => 2,
            Codec::Skeleton => 1,
            Codec::Unknown => 1,
        }
    }
}

/// Vorbis 识别头里、容器这一层需要的字段。
///
/// **故意只读这几个**：块大小 / 模式表在 setup 头里，属于编解码器的活（见
/// [`PacketTiming`]）。这里读的是「流参数」，和 ASF 读 WAVEFORMATEX 同一个性质。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VorbisId {
    pub channels: u8,
    pub sample_rate: u32,
    pub bitrate_maximum: i32,
    pub bitrate_nominal: i32,
    pub bitrate_minimum: i32,
    /// 短块的对数（块大小 = `1 << blocksize_0`）。
    pub blocksize_0: u8,
    /// 长块的对数。
    pub blocksize_1: u8,
}

impl VorbisId {
    /// 从识别头（`\x01vorbis` 开头那个包）解析。
    pub fn parse(p: &[u8]) -> Result<Self> {
        // 7 字节魔数 + version(4) + channels(1) + rate(4) + 三个码率(12) + 块大小(1)
        if p.len() < 30 {
            return Err(Error::Bad(format!("Vorbis 识别头只有 {} 字节", p.len())));
        }
        if !p.starts_with(b"\x01vorbis") {
            return Err(Error::Bad("不是 Vorbis 识别头".into()));
        }
        let version = u32::from_le_bytes(p[7..11].try_into().unwrap());
        if version != 0 {
            return Err(Error::Unsupported(format!("Vorbis 版本 {version}（只认 0）")));
        }
        let channels = p[11];
        let sample_rate = u32::from_le_bytes(p[12..16].try_into().unwrap());
        let bitrate_maximum = i32::from_le_bytes(p[16..20].try_into().unwrap());
        let bitrate_nominal = i32::from_le_bytes(p[20..24].try_into().unwrap());
        let bitrate_minimum = i32::from_le_bytes(p[24..28].try_into().unwrap());
        let bs = p[28];
        let blocksize_0 = bs & 0x0F;
        let blocksize_1 = bs >> 4;
        if channels == 0 {
            return Err(Error::Bad("Vorbis 声道数为 0".into()));
        }
        if sample_rate == 0 {
            return Err(Error::Bad("Vorbis 采样率为 0".into()));
        }
        if !(6..=13).contains(&blocksize_0) || !(6..=13).contains(&blocksize_1) {
            return Err(Error::Bad(format!(
                "Vorbis 块大小指数越界（{blocksize_0} / {blocksize_1}，合法范围 6..=13）"
            )));
        }
        if blocksize_0 > blocksize_1 {
            return Err(Error::Bad(format!(
                "Vorbis 短块 {blocksize_0} 比长块 {blocksize_1} 还大"
            )));
        }
        Ok(Self {
            channels,
            sample_rate,
            bitrate_maximum,
            bitrate_nominal,
            bitrate_minimum,
            blocksize_0,
            blocksize_1,
        })
    }

    /// 长块的实际样本数。
    pub fn long_block(&self) -> u32 {
        1 << self.blocksize_1
    }

    /// 短块的实际样本数。
    pub fn short_block(&self) -> u32 {
        1 << self.blocksize_0
    }
}

/// 把 [`AudioFormat::extra`] 拆回三个头包：`(识别头, 注释头, setup 头)`。
///
/// `extra` 里是**三个头包原样拼接**（与 ffmpeg 的 `extradata` 同思路），
/// 而 Vorbis 解码器建实例只需要**识别头**（流参数）和 **setup 头**（码本 / floor /
/// residue / mapping / mode）—— 所以中间那个注释头必须能跳过去。
///
/// 不能靠「搜 `\x05vorbis`」：注释头里装的是 UTF-8 文本，理论上可以含任意字节，
/// 而这里的分界必须是**确定的**。注释头自带长度字段（vendor 串 + 逐条 comment），
/// 按规范 5.2 数一遍就知道边界在哪。
pub fn split_vorbis_headers(extra: &[u8]) -> Result<(&[u8], &[u8], &[u8])> {
    const ID_LEN: usize = 30; // 识别头长度由规范 4.2.2 定死
    if extra.len() < ID_LEN || !extra.starts_with(b"\x01vorbis") {
        return Err(Error::Bad("extra 不是以 Vorbis 识别头开头".into()));
    }
    let id = &extra[..ID_LEN];
    let rest = &extra[ID_LEN..];
    if !rest.starts_with(b"\x03vorbis") {
        return Err(Error::Bad("识别头后面不是注释头".into()));
    }

    // 注释头：7 字节魔数 + vendor 长度 + vendor + 条数 + 逐条 (长度 + 内容) + framing 位
    let be = |at: usize| -> Result<usize> {
        let s = rest
            .get(at..at + 4)
            .ok_or_else(|| Error::Bad("注释头被截断".into()))?;
        Ok(u32::from_le_bytes(s.try_into().unwrap()) as usize)
    };
    let mut p = 7usize;
    let vendor = be(p)?;
    p += 4;
    p = p
        .checked_add(vendor)
        .filter(|&p| p <= rest.len())
        .ok_or_else(|| Error::Bad("注释头的 vendor 串长度越界".into()))?;
    let count = be(p)?;
    p += 4;
    for _ in 0..count {
        let len = be(p)?;
        p = p
            .checked_add(4)
            .and_then(|p| p.checked_add(len))
            .ok_or_else(|| Error::Bad("注释头的 comment 长度溢出".into()))?;
        if p > rest.len() {
            return Err(Error::Bad("注释头被截断".into()));
        }
    }
    p += 1; // framing 位
    if p > rest.len() {
        return Err(Error::Bad("注释头缺 framing 位".into()));
    }

    let comment = &rest[..p];
    let setup = &rest[p..];
    if !setup.starts_with(b"\x05vorbis") {
        return Err(Error::Bad("注释头后面不是 setup 头".into()));
    }
    Ok((id, comment, setup))
}

/// Ogg 解封装器。
///
/// 用法是**两段式**：`open` 读完所有头（顺手把 Vorbis 那三个头塞进
/// `AudioFormat::extra`），调用方拿 `extra` 去建解码器、拿到 [`PacketTiming`]，
/// 再 `set_timing` 之后才开始 `next_sample`。
pub struct OggDemuxer<R> {
    packets: PacketReader<R>,
    streams: Vec<Stream>,
    /// 每个 `serial` 的编码类型。
    codecs: HashMap<u32, Codec>,
    /// 每个 `serial` 的识别头（只有 Vorbis 有）。
    vorbis_ids: HashMap<u32, VorbisId>,
    timing: Option<Box<dyn PacketTiming>>,
    /// 音频流（目前只支持单条）。
    audio_serial: Option<u32>,
    /// 音频流在 `streams` 里的编号。
    audio_stream: u8,
    sample_rate: u32,
    /// **已产出的 PCM 位置**（采样数）：上一个包输出结束的地方。
    /// 第一个包不产生输出，所以它跑完这里还是 0。
    sample_pos: i64,
    prev_packet: Vec<u8>,
    /// 上一个包来自哪一页、那一页的 granule —— 换页时用它校正累加器。
    last_page_sequence: Option<u32>,
    prev_page_granule: i64,
    /// 见过的最后一个有效 `granule_position` —— **这是总采样数的权威值**。
    last_granule: i64,
    timing_misses: u64,
    finished: bool,
}

impl<R: Read> OggDemuxer<R> {
    /// 读头部，返回解封装器。
    ///
    /// 会一直读到**所有流的头都齐了**为止；音频包一个都还没吐。
    /// 不校验页 CRC（与 `ffmpeg` 默认一致）—— 要校验用 [`Self::open_with`]。
    pub fn open(r: R) -> Result<Self> {
        Self::open_with(r, false)
    }

    /// 同 [`Self::open`]，但可以指定**从第一页起**就校验 CRC。
    ///
    /// 之所以要有这个入口：`open` 会把头部几页读掉，之后再
    /// `set_verify_crc(true)` 只能覆盖到音频页 —— 而「头坏了」恰恰是最该
    /// 被抓到的情况（它会让后面整条解码链按错的参数跑）。
    pub fn open_with(r: R, verify_crc: bool) -> Result<Self> {
        let mut packets = PacketReader::new(r);
        packets.set_verify_crc(verify_crc);
        let mut streams = Vec::new();
        let mut headers_seen: HashMap<u32, usize> = HashMap::new();
        let mut codecs: HashMap<u32, Codec> = HashMap::new();
        let mut vorbis_ids = HashMap::new();
        let mut extras: HashMap<u32, Vec<u8>> = HashMap::new();
        let mut pending: VecDeque<Packet> = VecDeque::new();
        let mut order: Vec<u32> = Vec::new();

        // 读到所有流都齐头为止。上限只是防「文件里全是骨架流」这种病态输入
        // 把内存吃光 —— 正常文件几个包就结束了。
        let mut guard = 0usize;
        loop {
            let all_done = !order.is_empty()
                && order.iter().all(|s| {
                    headers_seen.get(s).copied().unwrap_or(0)
                        >= codecs.get(s).copied().unwrap_or(Codec::Unknown).header_packets()
                });
            if all_done {
                break;
            }
            if guard > 4096 {
                return Err(Error::Bad("头部包超过 4096 个还没读完，文件可能损坏".into()));
            }
            guard += 1;

            let pkt = match packets.next_packet()? {
                Some(p) => p,
                None => {
                    if order.is_empty() {
                        return Err(Error::Bad("文件里没有任何 Ogg 逻辑流".into()));
                    }
                    break;
                }
            };

            let serial = pkt.serial;
            if !codecs.contains_key(&serial) {
                // 第一次见到这个 serial：这就是它的识别头。
                let codec = Codec::identify(&pkt.data);
                codecs.insert(serial, codec);
                order.push(serial);
                headers_seen.insert(serial, 0);
                extras.insert(serial, Vec::new());
                if codec == Codec::Vorbis {
                    vorbis_ids.insert(serial, VorbisId::parse(&pkt.data)?);
                }
            }

            let seen = headers_seen.entry(serial).or_insert(0);
            let total = codecs[&serial].header_packets();
            if *seen < total {
                *seen += 1;
                extras.entry(serial).or_default().extend_from_slice(&pkt.data);
            } else {
                // 头已经收完的流又来了包 —— 那是音频，先存着。
                pending.push_back(pkt);
            }
        }

        // 组装流表。
        let mut audio_serial = None;
        let mut audio_stream = 0u8;
        for (i, serial) in order.iter().enumerate() {
            let codec = codecs[serial];
            let kind = match codec {
                Codec::Vorbis | Codec::Opus | Codec::Flac | Codec::Speex => StreamKind::Audio,
                Codec::Theora => StreamKind::Video,
                Codec::Skeleton | Codec::Unknown => StreamKind::Unknown,
            };
            let audio = match codec {
                Codec::Vorbis => {
                    let id = vorbis_ids[serial];
                    let extra = extras.remove(serial).unwrap_or_default();
                    if audio_serial.is_none() {
                        audio_serial = Some(*serial);
                        audio_stream = i as u8;
                    }
                    Some(AudioFormat {
                        format_tag: VORBIS_FORMAT_TAG,
                        channels: id.channels as u16,
                        sample_rate: id.sample_rate,
                        // Vorbis 是变码率，没有「平均字节率」；标称码率除 8 只是给下游一个量级。
                        avg_bytes_per_sec: (id.bitrate_nominal.max(0) as u32) / 8,
                        block_align: 0,
                        bits_per_sample: 0,
                        extra,
                    })
                }
                _ => None,
            };
            streams.push(Stream {
                number: i as u8,
                kind,
                time_offset_ms: 0,
                audio,
                video: None,
            });
        }

        if audio_serial.is_none() {
            let names: Vec<&str> = order
                .iter()
                .map(|s| match codecs[s] {
                    Codec::Vorbis => "Vorbis",
                    Codec::Opus => "Opus",
                    Codec::Theora => "Theora",
                    Codec::Flac => "FLAC",
                    Codec::Speex => "Speex",
                    Codec::Skeleton => "Skeleton",
                    Codec::Unknown => "未知",
                })
                .collect();
            return Err(Error::Unsupported(format!(
                "文件里没有 Vorbis 音频流（找到：{}）",
                if names.is_empty() {
                    "无".to_string()
                } else {
                    names.join(" / ")
                }
            )));
        }

        let sample_rate = streams
            .iter()
            .find_map(|s| s.audio.as_ref().map(|a| a.sample_rate))
            .unwrap_or(0);

        // 头部之后已经读到的音频包要接在队列前面。
        let mut demuxer = Self {
            packets,
            streams,
            codecs,
            vorbis_ids,
            timing: None,
            audio_serial,
            audio_stream,
            sample_rate,
            sample_pos: 0,
            prev_packet: Vec::new(),
            last_page_sequence: None,
            prev_page_granule: GRANULE_NONE,
            last_granule: GRANULE_NONE,
            timing_misses: 0,
            finished: false,
        };
        demuxer.packets.queue = pending;
        Ok(demuxer)
    }

    /// 打开整页 CRC 校验（只影响**还没读的**页，见 [`Self::open_with`]）。
    pub fn set_verify_crc(&mut self, on: bool) {
        self.packets.set_verify_crc(on);
    }

    pub fn crc_errors(&self) -> u64 {
        self.packets.crc_errors()
    }

    /// 丢掉过多少个半截包（缺页、或 `continued` 标志与实际不符）。
    pub fn discarded_partials(&self) -> u64 {
        self.packets.discarded_partials()
    }

    /// 页序号跳变的次数。> 0 说明文件有缺页，音频会缺一截。
    pub fn sequence_gaps(&self) -> u64 {
        self.packets.sequence_gaps()
    }

    /// 时间戳推不出来的包数（[`PacketTiming`] 返回了 `None`）。
    pub fn timing_misses(&self) -> u64 {
        self.timing_misses
    }

    pub fn streams(&self) -> &[Stream] {
        &self.streams
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// 音频流的 `serial`。
    pub fn audio_serial(&self) -> Option<u32> {
        self.audio_serial
    }

    /// 每个 `serial` 的编码类型。
    pub fn codec_of(&self, serial: u32) -> Option<Codec> {
        self.codecs.get(&serial).copied()
    }

    /// 装上「包时长」口径。**必须在 `next_sample` 之前调用**，否则时间戳全是 0。
    pub fn set_timing(&mut self, t: Box<dyn PacketTiming>) {
        self.timing = Some(t);
    }

    /// 下一个音频包；到文件尾返回 `None`。
    pub fn next_sample(&mut self) -> Result<Option<Sample>> {
        if self.finished {
            return Ok(None);
        }
        let audio_serial = self.audio_serial.expect("open 保证了有音频流");

        loop {
            let pkt = match self.packets.next_packet()? {
                Some(p) => p,
                None => {
                    self.finished = true;
                    return Ok(None);
                }
            };
            if pkt.serial != audio_serial {
                continue; // 视频/元数据流：本实现不解，跳过
            }

            // 换页了：**上一页**的 granule 是权威值，用它校正累加器。
            // 不改已经发出的 pts —— 与 ffprobe 的口径一致（它也只截短最后一包的时长）。
            if let Some(prev_seq) = self.last_page_sequence {
                if prev_seq != pkt.page_sequence && self.prev_page_granule >= 0 {
                    self.sample_pos = self.prev_page_granule;
                }
            }
            self.last_page_sequence = Some(pkt.page_sequence);
            self.prev_page_granule = pkt.page_granule;
            if pkt.page_granule >= 0 {
                self.last_granule = pkt.page_granule;
            }

            let first = self.prev_packet.is_empty();
            let advance = match &self.timing {
                Some(t) => {
                    // 第一个包：当作「上一包就是自己」，推进量 = 块大小 / 2。
                    let prev = if first { &pkt.data[..] } else { &self.prev_packet[..] };
                    match t.advance(prev, &pkt.data) {
                        Some(a) => a as i64,
                        None => {
                            self.timing_misses += 1;
                            0
                        }
                    }
                }
                None => 0,
            };

            // 时间戳口径（与 ffprobe 逐包核对过）：
            // - 第 1 个包不产生输出，`pts = -推进量`，累加器不动；
            // - 之后每包 `pts = 累加器`（= 上一包输出结束处），然后累加器 += 推进量。
            let pts = if first {
                -advance
            } else {
                self.sample_pos
            };
            if !first {
                self.sample_pos += advance;
            }
            let timestamp_ms = samples_to_ms(pts, self.sample_rate);
            self.prev_packet.clear();
            self.prev_packet.extend_from_slice(&pkt.data);

            return Ok(Some(Sample {
                stream: self.audio_stream,
                key_frame: true, // 音频包都是「关键帧」：互不依赖
                timestamp_ms,
                data: pkt.data,
            }));
        }
    }

    /// 已产出的 PCM 采样数（= 最后一个包输出结束的位置，**未按末页 granule 截短**）。
    ///
    /// 第一个包不产生输出，所以只收到 1 个包时这里是 0。
    /// 想看「真正该输出多少采样」用 [`Self::total_samples`]。
    pub fn samples_produced(&self) -> i64 {
        self.sample_pos
    }

    /// 总采样数 —— 见过的最后一个有效 `granule_position`，**这是权威值**。
    ///
    /// 和 [`Self::samples_produced`] 的差就是「编码器标了可丢弃的尾巴」：
    /// 实测样本差 832 个采样（`ffprobe` 报 `Skip Samples 0 832`）。
    /// 没读到任何带 granule 的页时返回 `None`。
    pub fn total_samples(&self) -> Option<u64> {
        (self.last_granule >= 0).then_some(self.last_granule as u64)
    }

    /// Vorbis 识别头（若音频流是 Vorbis）。
    pub fn vorbis_id(&self) -> Option<VorbisId> {
        self.audio_serial.and_then(|s| self.vorbis_ids.get(&s).copied())
    }
}

/// 采样数 → 毫秒，四舍五入（负数向远离 0 的方向），与 `ffmpeg` 的 `av_rescale` 同口径。
fn samples_to_ms(samples: i64, rate: u32) -> i64 {
    if rate == 0 {
        return 0;
    }
    let rate = rate as i64;
    let num = samples * 1000;
    if num >= 0 {
        (num + rate / 2) / rate
    } else {
        -((-num + rate / 2) / rate)
    }
}

/// 扫文件尾拿**总采样数与采样率**。
///
/// 为什么单独一个函数：Ogg 没有全局头，总长度只体现在**最后一页的
/// `granule_position`** 上，顺序读要读完整份文件才知道。这里从尾部往回找
/// 最后一个 `OggS` 再解析 —— 与 `ffprobe` 的做法一致。
///
/// 返回 `(总采样数, 采样率)`。两样一起给是因为它们来自同一趟扫描：采样率在
/// 文件最前面那一页、granule 在最后一页，分开做就要读两遍头。
///
/// `r` 必须可 seek；调用后位置会被改到别处。
pub fn stream_length<R: Read + Seek>(r: &mut R) -> Result<Option<(u64, u32)>> {
    let end = r.seek(SeekFrom::End(0))?;
    if end < HEADER_SIZE as u64 {
        return Ok(None);
    }

    // 先拿采样率：识别头永远在文件最前面那一页。
    r.seek(SeekFrom::Start(0))?;
    let mut head = PageReader::new(&mut *r);
    let first = match head.next_page()? {
        Some(p) => p,
        None => return Ok(None),
    };
    let id = match VorbisId::parse(&first.body) {
        Ok(id) => id,
        Err(_) => return Ok(None),
    };

    // 尾部窗口：最后一页 ≤ 27 + 255 + 65025 字节，取 128 KiB 足够覆盖。
    const WINDOW: u64 = 128 * 1024;
    let start = end.saturating_sub(WINDOW);
    r.seek(SeekFrom::Start(start))?;
    let mut tail = Vec::new();
    r.read_to_end(&mut tail)?;

    // 从后往前找最后一个能解析通的页。
    let mut from = tail.len();
    while from > 0 {
        let idx = match tail[..from].windows(4).rposition(|w| w == CAPTURE) {
            Some(i) => i,
            None => break,
        };
        let mut pr = PageReader::new(&tail[idx..]);
        if let Ok(Some(p)) = pr.next_page() {
            if p.granule >= 0 {
                return Ok(Some((p.granule as u64, id.sample_rate)));
            }
        }
        from = idx;
    }
    Ok(None)
}

/// 扫文件尾拿总时长（毫秒）。
///
/// 就是 [`stream_length`] 换算一下 —— 不要各写一份尾部扫描，两处迟早会漂。
pub fn duration_ms<R: Read + Seek>(r: &mut R) -> Result<Option<u64>> {
    Ok(stream_length(r)?.map(|(samples, rate)| samples * 1000 / rate.max(1) as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 拼一份「识别头 + 注释头 + setup 头」，注释头里故意塞一个**含 `\x05vorbis`
    /// 的 comment** —— 靠搜魔数分界的那种写法会在这里断错位置。
    #[test]
    fn splits_vorbis_headers_by_length_not_by_magic() {
        let mut id = vec![0u8; 30];
        id[..7].copy_from_slice(b"\x01vorbis");
        id[11] = 2;
        id[12..16].copy_from_slice(&44100u32.to_le_bytes());
        id[28] = 0xBB;
        id[29] = 1;

        let vendor = b"v2v-test";
        let comments: [&[u8]; 2] = [b"TITLE=trap", b"X=\x05vorbis\x03vorbis"];
        let mut cm = Vec::new();
        cm.extend_from_slice(b"\x03vorbis");
        cm.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        cm.extend_from_slice(vendor);
        cm.extend_from_slice(&(comments.len() as u32).to_le_bytes());
        for c in comments {
            cm.extend_from_slice(&(c.len() as u32).to_le_bytes());
            cm.extend_from_slice(c);
        }
        cm.push(1); // framing 位

        let setup = b"\x05vorbis\x00\x01\x02payload".to_vec();
        let mut extra = id.clone();
        extra.extend_from_slice(&cm);
        extra.extend_from_slice(&setup);

        let (a, b, c) = split_vorbis_headers(&extra).unwrap();
        assert_eq!(a, &id[..]);
        assert_eq!(b, &cm[..]);
        assert_eq!(c, &setup[..]);
    }

    /// 截断的注释头必须报错，不能靠「后面看起来像 setup 头」蒙过去。
    #[test]
    fn truncated_comment_header_is_rejected() {
        let mut id = vec![0u8; 30];
        id[..7].copy_from_slice(b"\x01vorbis");
        let mut extra = id;
        extra.extend_from_slice(b"\x03vorbis");
        extra.extend_from_slice(&99u32.to_le_bytes()); // vendor 长度 99，但后面什么都没有
        assert!(split_vorbis_headers(&extra).is_err());
    }

    #[test]
    fn identifies_known_codecs() {
        assert_eq!(Codec::identify(b"\x01vorbis\x00\x00"), Codec::Vorbis);
        assert_eq!(Codec::identify(b"OpusHead\x01\x02"), Codec::Opus);
        assert_eq!(Codec::identify(b"\x80theora"), Codec::Theora);
        assert_eq!(Codec::identify(b"fLaC\x00"), Codec::Flac);
        assert_eq!(Codec::identify(b"fishead\0\x00"), Codec::Skeleton);
        assert_eq!(Codec::identify(b"\x7funknown"), Codec::Unknown);
    }

    #[test]
    fn ms_rounding_matches_ffmpeg_style() {
        // 第一个包：-1024 采样 @ 44.1k = -23.22 ms → -23
        assert_eq!(samples_to_ms(-1024, 44100), -23);
        assert_eq!(samples_to_ms(0, 44100), 0);
        // 1024 采样 = 23.219 ms → 23
        assert_eq!(samples_to_ms(1024, 44100), 23);
        // 44100 采样 = 1000 ms
        assert_eq!(samples_to_ms(44100, 44100), 1000);
        // 48000 下 1024 采样 = 21.33 ms → 21
        assert_eq!(samples_to_ms(1024, 48000), 21);
        assert_eq!(samples_to_ms(1000, 0), 0);
    }

    #[test]
    fn vorbis_id_rejects_garbage() {
        assert!(VorbisId::parse(b"\x01vorbis").is_err());
        let mut p = vec![0u8; 30];
        p[..7].copy_from_slice(b"\x01vorbis");
        p[11] = 2; // channels
        p[12..16].copy_from_slice(&44100u32.to_le_bytes());
        p[28] = 0xB8; // bs0 = 8, bs1 = 11
        let id = VorbisId::parse(&p).unwrap();
        assert_eq!(id.channels, 2);
        assert_eq!(id.sample_rate, 44100);
        assert_eq!(id.short_block(), 256);
        assert_eq!(id.long_block(), 2048);

        // 短块比长块大 → 报错
        p[28] = 0x8B; // bs0 = 11, bs1 = 8
        assert!(VorbisId::parse(&p).is_err());
        // 块大小指数越界
        p[28] = 0x5F; // bs0 = 15
        assert!(VorbisId::parse(&p).is_err());
    }
}
