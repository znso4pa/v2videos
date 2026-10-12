//! 纯 Rust 的 **AVI（RIFF）解封装**。
//!
//! 目标与 `v2v-asf` 一致：把容器拆开，逐条吐出**裸基本流样本**
//! （`v2v_container::Sample`）。上层（解码器 / 转码工具）不必知道样本来自哪个容器。
//!
//! ```text
//!   .avi ──► [本 crate] ──► Sample { stream, key_frame, timestamp_ms, data }
//! ```
//!
//! 无 C、无 FFI、零依赖（除共用的 `v2v-container`）。
//!
//! ## 支持范围（本实现**只做能验证的部分**）
//!
//! - `RIFF/AVI `，`LIST hdrl`（`avih` + 每条流的 `strl`：`strh` + `strf`）。
//! - `LIST movi` 里的**流数据块**：`NNdc` / `NNdb`（视频）、`NNwb`（音频）。
//!   支持 `movi` 内嵌套 `LIST`（有些封装器会把块再分组成 `rec ` 列表）。
//! - `idx1` 索引：用来给样本打**关键帧标志**（`AVIIF_KEYFRAME = 0x10`）。
//!   没有 `idx1` 时按「全部标关键帧」处理并记在 `has_index()` 里 —— 宁可保守，
//!   也不要凭空把关键帧标成非关键帧（那会让下游 seek 到解不开的位置）。
//! - **不认**：OpenDML（`AVIX` 分块、`indx` 索引）、`odml` 扩展、交错之外的
//!   奇技淫巧。遇到 `AVIX` 明确报 `Unsupported`，不假装支持。
//!
//! ## 时间戳
//!
//! AVI 不给每帧时间戳，只给**流的速率**（`strh` 的 `dwScale` / `dwRate`，
//! 单位是「每 `dwRate` 个 tick 有 `dwScale` 个采样」）。所以
//! `pts_ms = 帧号 × dwScale × 1000 / dwRate` —— 与 `ffprobe` 的口径一致。
//! （注意 AVI 的 pts 从 0 起，不像 ASF 要减 preroll。）

mod error;

pub use error::{Error, Result};
pub use v2v_container::{fourcc_string, AudioFormat, Sample, Stream, StreamKind, VideoFormat};

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use v2v_container::StreamKind as Kind;

/// 单个块的最大字节数（防御坏长度：一个 4GB 的 `dwSize` 不该把内存吃光）。
const MAX_CHUNK: u32 = 64 * 1024 * 1024;

/// `idx1` 里的关键帧标志。
const AVIIF_KEYFRAME: u32 = 0x10;

/// 一条流的速率信息（`strh` 的 `dwScale` / `dwRate`）。
#[derive(Debug, Clone, Copy, Default)]
struct Rate {
    scale: u32,
    rate: u32,
}

impl Rate {
    /// 第 `n` 帧的显示时间（毫秒）。
    fn pts_ms(&self, n: u64) -> i64 {
        if self.rate == 0 {
            return 0;
        }
        (n * self.scale as u64 * 1000 / self.rate as u64) as i64
    }
}

/// AVI 解封装器。
pub struct AviDemuxer<R: Read + Seek> {
    reader: R,
    streams: Vec<Stream>,
    rates: Vec<Rate>,
    /// 每条流已经吐了多少帧（算时间戳用）。
    frame_no: Vec<u64>,
    /// 关键帧表：块在文件里的绝对偏移 → 是不是关键帧。
    key_at: HashMap<u64, bool>,
    has_index: bool,
    /// `movi` 列表数据的结束偏移（绝对）。
    movi_end: u64,
    /// 当前读到哪，以及嵌套 `LIST` 的边界栈。
    pos: u64,
    limits: Vec<u64>,
    done: bool,
}

impl<R: Read + Seek> AviDemuxer<R> {
    /// 解析 `RIFF/AVI ` 头（`hdrl` + `movi` 边界 + `idx1`）。
    pub fn new(mut reader: R) -> Result<Self> {
        let size = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;
        if size < 12 {
            return Err(Error::Bad("文件短于 RIFF 头"));
        }

        let mut head = [0u8; 12];
        reader.read_exact(&mut head)?;
        if &head[0..4] != b"RIFF" {
            return Err(Error::Bad("不是 RIFF 文件"));
        }
        let riff_size = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as u64;
        let riff_end = (8 + riff_size).min(size);
        match &head[8..12] {
            b"AVI " => {}
            b"AVIX" => {
                return Err(Error::Unsupported(
                    "OpenDML（AVIX）分块文件：本实现只做单块 AVI".into(),
                ))
            }
            _ => return Err(Error::Bad("RIFF 类型不是 AVI")),
        }

        let mut streams: Vec<Stream> = Vec::new();
        let mut rates: Vec<Rate> = Vec::new();
        let mut key_at = HashMap::new();
        let mut idx1_raw: Option<Vec<u8>> = None;
        let mut movi: Option<(u64, u64)> = None;

        // ---- 顶层块遍历 ----
        let mut pos = 12u64;
        while pos + 8 <= riff_end {
            reader.seek(SeekFrom::Start(pos))?;
            let id = read_fourcc(&mut reader)?;
            let sz = read_u32(&mut reader)? as u64;
            let data = pos + 8;
            if data + sz > riff_end {
                break; // 尾部截断：能读到的都读了
            }
            match &id {
                b"LIST" => {
                    let kind = read_fourcc(&mut reader)?;
                    let body = data + 4;
                    match &kind {
                        b"hdrl" => {
                            let buf = read_vec(&mut reader, (sz - 4) as usize)?;
                            parse_hdrl(&buf, &mut streams, &mut rates)?;
                        }
                        b"movi" => movi = Some((body, data + sz)),
                        _ => {} // INFO / odml / JUNK…
                    }
                }
                b"idx1" => {
                    idx1_raw = Some(read_vec(&mut reader, sz as usize)?);
                }
                _ => {}
            }
            pos = data + sz + (sz & 1);
        }

        let (movi_start, movi_end) = movi.ok_or(Error::Bad("没有 movi 列表"))?;
        // idx1 的偏移基准**各家封装器不一致**：规范说「相对 movi 列表的数据起点」，
        // 但 ffmpeg 写的是「相对 movi 四字本身」（差 4 字节）。别猜 —— 拿索引里的
        // ckid 去两个候选位置各比一下，谁对得上就用谁。
        let mut has_index = false;
        if let Some(buf) = &idx1_raw {
            let entries = parse_idx1(buf);
            let base = pick_idx_base(&mut reader, movi_start, &entries)?;
            for (ckid, flags, off, _len) in &entries {
                let _ = ckid;
                key_at.insert(base + *off as u64, flags & AVIIF_KEYFRAME != 0);
            }
            has_index = true;
        }
        Ok(Self {
            reader,
            frame_no: vec![0; streams.len()],
            streams,
            rates,
            key_at,
            has_index,
            movi_end,
            pos: movi_start,
            limits: Vec::new(),
            done: false,
        })
    }

    pub fn streams(&self) -> &[Stream] {
        &self.streams
    }

    /// 有没有 `idx1` 索引（没有的话关键帧标志是「全标」的保守值）。
    pub fn has_index(&self) -> bool {
        self.has_index
    }

    /// 下一条样本；`None` = `movi` 读完。
    ///
    /// 顺序就是文件里的顺序（AVI 是交错的，所以视频/音频样本会交替出现）。
    pub fn next_sample(&mut self) -> Result<Option<Sample>> {
        if self.done {
            return Ok(None);
        }
        loop {
            let limit = *self.limits.last().unwrap_or(&self.movi_end);
            if self.pos + 8 > limit {
                if self.limits.pop().is_none() {
                    self.done = true;
                    return Ok(None);
                }
                continue;
            }
            self.reader.seek(SeekFrom::Start(self.pos))?;
            let id = read_fourcc(&mut self.reader)?;
            let sz = read_u32(&mut self.reader)?;
            let data = self.pos + 8;
            if sz > MAX_CHUNK {
                return Err(Error::Bad("块长度离谱（可能不是 AVI）"));
            }
            if data + sz as u64 > self.movi_end {
                self.done = true;
                return Ok(None);
            }
            self.pos = data + sz as u64 + (sz as u64 & 1);

            if &id == b"LIST" {
                // 嵌套列表：跳过类型四字，把边界压栈，继续在里面走。
                self.reader.seek(SeekFrom::Start(data))?;
                let _kind = read_fourcc(&mut self.reader)?;
                self.limits.push(data + sz as u64);
                self.pos = data + 4;
                continue;
            }

            // 流块：前两个字符是流号，第三个字符区分 d/w。
            let (Some(tens), Some(ones)) = (digit(id[0]), digit(id[1])) else {
                continue; // JUNK / 其它非流块
            };
            let idx = (tens * 10 + ones) as usize;
            if idx >= self.streams.len() {
                continue;
            }
            let is_audio_chunk = id[2] == b'w';
            let kind = self.streams[idx].kind;
            let wanted = matches!(
                (kind, is_audio_chunk),
                (Kind::Video, false) | (Kind::Audio, true)
            );
            if !wanted {
                continue;
            }

            let data = read_vec(&mut self.reader, sz as usize)?;
            // 帧号**每个槽都要走**（空块也占时间轴），所以先取号再决定要不要吐样本。
            //
            // 空块是真会遇到的：ffmpeg 的 AVI 封装器保留源的 1/1000 时基，
            // 60fps 的视频会被写成「每毫秒一个槽、只有约每 17 个槽有数据」——
            // 实测 2 秒 = 2009 个槽 / 118 个有数据。空块不产样本，但**必须**
            // 让后面的帧时间戳继续往前走。
            let n = self.frame_no[idx];
            self.frame_no[idx] += 1;
            if sz == 0 {
                continue;
            }
            let key_frame = self
                .key_at
                .get(&(self.pos - sz as u64 - (sz as u64 & 1) - 8))
                .copied()
                .unwrap_or(true);
            return Ok(Some(Sample {
                stream: self.streams[idx].number,
                key_frame,
                timestamp_ms: self.rates[idx].pts_ms(n),
                data,
            }));
        }
    }
}

fn digit(b: u8) -> Option<u8> {
    b.is_ascii_digit().then(|| b - b'0')
}

// ==================== RIFF 小工具 ====================

fn read_fourcc<R: Read>(r: &mut R) -> Result<[u8; 4]> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_vec<R: Read>(r: &mut R, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    r.read_exact(&mut v)?;
    Ok(v)
}

/// 内存游标（解析 `hdrl` / `idx1` 用）。
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, p: 0 }
    }
    fn u8(&mut self) -> Result<u8> {
        let v = *self.b.get(self.p).ok_or(Error::Bad("hdrl 截断"))?;
        self.p += 1;
        Ok(v)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes([self.u8()?, self.u8()?]))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes([
            self.u8()?,
            self.u8()?,
            self.u8()?,
            self.u8()?,
        ]))
    }
    fn fourcc(&mut self) -> Result<[u8; 4]> {
        Ok([self.u8()?, self.u8()?, self.u8()?, self.u8()?])
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            return Err(Error::Bad("hdrl 截断"));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn left(&self) -> usize {
        self.b.len() - self.p
    }
}

// ==================== hdrl ====================

/// 解析 `LIST hdrl` 的**数据部分**（不含 `hdrl` 四字）：`avih` + 每条流的 `strl`。
fn parse_hdrl(buf: &[u8], streams: &mut Vec<Stream>, rates: &mut Vec<Rate>) -> Result<()> {
    let mut c = Cur::new(buf);
    let mut n_streams: Option<usize> = None;

    while c.left() >= 8 {
        let id = c.fourcc()?;
        let sz = c.u32()? as usize;
        if c.left() < sz {
            return Err(Error::Bad("hdrl 子块越界"));
        }
        let body = c.take(sz)?;
        if sz % 2 == 1 {
            let _ = c.take(1); // 块补齐
        }
        match &id {
            b"avih" => {
                let mut a = Cur::new(body);
                let _us_per_frame = a.u32()?;
                let _max_bytes = a.u32()?;
                let _pad = a.u32()?;
                let _flags = a.u32()?;
                let _total = a.u32()?;
                let _initial = a.u32()?;
                n_streams = Some(a.u32()? as usize);
            }
            b"LIST" => {
                // 这里 body 的四个字是列表类型（"strl"）
                let mut s = Cur::new(body);
                let kind = s.fourcc()?;
                if &kind == b"strl" {
                    parse_strl(s.take(s.left())?, streams, rates)?;
                }
            }
            _ => {}
        }
    }
    if let Some(n) = n_streams {
        // 声明了流数但没解析出那么多 → 以解析结果为准（不静默补空流）
        let _ = n;
    }
    Ok(())
}

/// 解析一条 `strl` 列表的数据（不含 `strl` 四字）：`strh` + `strf`。
fn parse_strl(buf: &[u8], streams: &mut Vec<Stream>, rates: &mut Vec<Rate>) -> Result<()> {
    let mut c = Cur::new(buf);
    let mut kind = Kind::Unknown;
    let mut rate = Rate::default();
    let mut audio: Option<AudioFormat> = None;
    let mut video: Option<VideoFormat> = None;

    while c.left() >= 8 {
        let id = c.fourcc()?;
        let sz = c.u32()? as usize;
        if c.left() < sz {
            break;
        }
        let body = c.take(sz)?;
        if sz % 2 == 1 {
            let _ = c.take(1);
        }
        match &id {
            b"strh" => {
                let mut h = Cur::new(body);
                let fcc_type = h.fourcc()?;
                let _fcc_handler = h.fourcc()?;
                let _flags = h.u32()?;
                let _priority = h.u16()?;
                let _language = h.u16()?;
                let _initial_frames = h.u32()?;
                rate.scale = h.u32()?;
                rate.rate = h.u32()?;
                kind = match &fcc_type {
                    b"vids" => Kind::Video,
                    b"auds" => Kind::Audio,
                    b"txts" => Kind::Unknown,
                    _ => Kind::Unknown,
                };
            }
            b"strf" => {
                let mut f = Cur::new(body);
                if kind == Kind::Video {
                    // BITMAPINFOHEADER
                    let _size = f.u32()?;
                    let w = f.u32()? as i32;
                    let h = f.u32()? as i32;
                    let _planes = f.u16()?;
                    let bpp = f.u16()?;
                    let fourcc = f.u32()?;
                    let _image = f.u32()?;
                    let _xppm = f.u32()?;
                    let _yppm = f.u32()?;
                    let _clr_used = f.u32()?;
                    let _clr_imp = f.u32()?;
                    let extra = f.take(f.left())?.to_vec();
                    video = Some(VideoFormat {
                        encoded_width: w.max(0) as u32,
                        encoded_height: h.unsigned_abs(),
                        width: w.max(0) as u32,
                        height: h.unsigned_abs(),
                        bits_per_pixel: bpp,
                        fourcc,
                        extra,
                    });
                } else if kind == Kind::Audio {
                    // WAVEFORMATEX
                    let format_tag = f.u16()?;
                    let channels = f.u16()?;
                    let sample_rate = f.u32()?;
                    let avg = f.u32()?;
                    let block_align = f.u16()?;
                    let bits = f.u16()?;
                    let extra = if f.left() >= 2 {
                        let _cb = f.u16()?;
                        f.take(f.left())?.to_vec()
                    } else {
                        Vec::new()
                    };
                    audio = Some(AudioFormat {
                        format_tag,
                        channels,
                        sample_rate,
                        avg_bytes_per_sec: avg,
                        block_align,
                        bits_per_sample: bits,
                        extra,
                    });
                }
            }
            _ => {}
        }
    }

    let number = streams.len() as u8;
    streams.push(Stream {
        number,
        kind,
        time_offset_ms: 0,
        audio,
        video,
    });
    rates.push(rate);
    Ok(())
}

// ==================== idx1 ====================

/// `idx1` 每条 16 字节：`ckid(4) + dwFlags(4) + dwChunkOffset(4) + dwChunkLength(4)`。
fn parse_idx1(buf: &[u8]) -> Vec<([u8; 4], u32, u32, u32)> {
    buf.chunks_exact(16)
        .map(|e| {
            let ckid = [e[0], e[1], e[2], e[3]];
            let flags = u32::from_le_bytes([e[4], e[5], e[6], e[7]]);
            let off = u32::from_le_bytes([e[8], e[9], e[10], e[11]]);
            let len = u32::from_le_bytes([e[12], e[13], e[14], e[15]]);
            (ckid, flags, off, len)
        })
        .collect()
}

/// 判定 `idx1` 偏移的基准：`movi` 数据体起点，还是 `movi` 四字起点（差 4）。
///
/// 拿前几条索引的 `ckid` 去两个候选位置读四字比对，谁对得多用谁。
fn pick_idx_base<R: Read + Seek>(
    reader: &mut R,
    movi_start: u64,
    entries: &[([u8; 4], u32, u32, u32)],
) -> Result<u64> {
    let mut best = (0usize, movi_start);
    for base in [movi_start, movi_start.saturating_sub(4)] {
        let mut hits = 0usize;
        for (ckid, _flags, off, _len) in entries.iter().take(8) {
            if reader.seek(SeekFrom::Start(base + *off as u64)).is_err() {
                break;
            }
            match read_fourcc(reader) {
                Ok(got) if &got == ckid => hits += 1,
                _ => {}
            }
        }
        if hits > best.0 {
            best = (hits, base);
        }
    }
    Ok(best.1)
}

/// 按魔数判断是不是 AVI（给「自动选解封装器」用）。
pub fn is_avi(head: &[u8]) -> bool {
    head.len() >= 12 && &head[0..4] == b"RIFF" && &head[8..12] == b"AVI "
}
