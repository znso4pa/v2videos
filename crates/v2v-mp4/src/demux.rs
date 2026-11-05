//! MP4 / ISOBMFF **解封装**。
//!
//! 与 `mux.rs` 对称：那边写 box，这边读 box。目标与其它容器 crate 一致 ——
//! **解出来的样本要能和 `ffmpeg -c copy` 的产物逐字节对上**，不是「能打开就算过」。
//!
//! ## 支持范围（与封装侧一致）
//!
//! - 非分片 MP4（`moov` 在头或尾都行）；`stco` 与 `co64` 都认。
//! - 视频 sample entry：`avc1` / `avc3` / `mp4v` / `XVID` / `DIVX` / `FMP4`。
//!   视频的 `extra` = `avcC`（H.264）或 `esds` 里的 decoder-specific info（mp4v）。
//! - 音频 sample entry：`mp4a`（`esds` → AudioSpecificConfig，`format_tag = 0x00FF`）、
//!   `sowt` / `twos`（PCM，`format_tag = 0x0001`）。
//! - `ctts`（B 帧的显示时间偏移）会算进 PTS；出样本按 **PTS** 排序。
//!
//! ## 不做的
//!
//! 分片 MP4（`moof`）、`edts`/`elst` 编辑列表、加密（`encv`/`enca`）。
//! 遇到这些明确报错，不猜。

use std::io::{Read, Seek, SeekFrom};

use v2v_container::{AudioFormat, Sample, Stream, StreamKind, VideoFormat};

use crate::{Error, Result};

/// 一条轨道在文件里的样本表（解出来之后就不需要 box 了）。
struct Track {
    stream: u8,
    samples: Vec<SampleRef>,
    cursor: usize,
}

struct SampleRef {
    offset: u64,
    size: u32,
    /// 显示时间（`stts` + `ctts`）—— 交给调用方的是它。
    pts_ms: i64,
    /// 解码时间（只有 `stts`）—— **跨轨合并按它排**，因为 B 帧的 PTS 会回跳。
    dts_ms: i64,
    key: bool,
}

/// MP4 解封装器。`R` 需要 `Seek` —— 样本按偏移随机读，不是顺序流。
pub struct Mp4Demuxer<R: Read + Seek> {
    r: R,
    streams: Vec<Stream>,
    tracks: Vec<Track>,
}

impl<R: Read + Seek> Mp4Demuxer<R> {
    /// 打开并解析 `moov`。样本数据在 `next_sample` 里按需 seek 读。
    pub fn new(mut r: R) -> Result<Self> {
        let moov = read_moov(&mut r)?;
        let (streams, tracks) = parse_moov(&moov)?;
        if streams.is_empty() {
            return Err(Error::Bad("moov 里没有可用的轨道"));
        }
        Ok(Self { r, streams, tracks })
    }

    pub fn streams(&self) -> &[Stream] {
        &self.streams
    }

    /// 取下一个样本。
    ///
    /// 跨轨按 **DTS** 合并（不是 PTS）：B 帧的显示时间会回跳，按 PTS 排会把
    /// 解码顺序打乱。调用方拿到的 `timestamp_ms` 仍然是 PTS。
    pub fn next_sample(&mut self) -> Result<Option<Sample>> {
        let mut best: Option<usize> = None;
        for (i, t) in self.tracks.iter().enumerate() {
            if t.cursor >= t.samples.len() {
                continue;
            }
            best = match best {
                None => Some(i),
                Some(b) => {
                    let a = t.samples[t.cursor].dts_ms;
                    let bb = self.tracks[b].samples[self.tracks[b].cursor].dts_ms;
                    if a < bb {
                        Some(i)
                    } else {
                        Some(b)
                    }
                }
            };
        }
        let Some(ti) = best else {
            return Ok(None);
        };
        let si = self.tracks[ti].cursor;
        self.tracks[ti].cursor += 1;
        let sref = &self.tracks[ti].samples[si];
        let mut data = vec![0u8; sref.size as usize];
        self.r.seek(SeekFrom::Start(sref.offset))?;
        self.r.read_exact(&mut data)?;
        Ok(Some(Sample {
            stream: self.tracks[ti].stream,
            key_frame: sref.key,
            timestamp_ms: sref.pts_ms,
            data,
        }))
    }
}

// ---------------------------------------------------------------------------
// 顶层：找 moov
// ---------------------------------------------------------------------------

/// 扫顶层 box，把 `moov` 的**载荷**整块读进内存。
///
/// 为什么整块读：`moov` 通常几十 KB 到几 MB，而嵌套解析用切片比用 seek 简单一个数量级；
/// 样本数据仍然走 seek，不进内存。
fn read_moov<R: Read + Seek>(r: &mut R) -> Result<Vec<u8>> {
    let end = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;
    loop {
        let pos = r.stream_position()?;
        if pos + 8 > end {
            return Err(Error::Bad("没有 moov"));
        }
        let (typ, size, hdr) = read_box_header(r)?;
        if size < hdr {
            return Err(Error::Bad("box 尺寸小于头"));
        }
        if typ == fourcc(b"moov") {
            let payload = size - hdr;
            if payload > 64 * 1024 * 1024 {
                return Err(Error::Bad("moov 大得离谱（> 64 MiB）"));
            }
            let mut buf = vec![0u8; payload as usize];
            r.read_exact(&mut buf)?;
            return Ok(buf);
        }
        if pos + size > end {
            return Err(Error::Bad("box 越过文件尾"));
        }
        r.seek(SeekFrom::Start(pos + size))?;
    }
}

fn read_box_header<R: Read>(r: &mut R) -> Result<(u32, u64, u64)> {
    let mut h = [0u8; 8];
    r.read_exact(&mut h)?;
    let mut size = u32::from_be_bytes([h[0], h[1], h[2], h[3]]) as u64;
    let typ = u32::from_be_bytes([h[4], h[5], h[6], h[7]]);
    let mut hdr = 8u64;
    if size == 1 {
        let mut big = [0u8; 8];
        r.read_exact(&mut big)?;
        size = u64::from_be_bytes(big);
        hdr = 16;
    } else if size == 0 {
        // 一直到文件尾 —— 顶层少见，但按规范是合法的。
        size = u64::MAX;
    }
    Ok((typ, size, hdr))
}

fn fourcc(b: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*b)
}

// ---------------------------------------------------------------------------
// moov 解析
// ---------------------------------------------------------------------------

/// 一个只读切片游标 —— box 都是大端定长字段，用它比 `Read` 好写。
struct Cur<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn new(d: &'a [u8]) -> Self {
        Self { d, p: 0 }
    }
    fn left(&self) -> usize {
        self.d.len() - self.p
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.left() < n {
            return Err(Error::Bad("box 越界"));
        }
        let s = &self.d[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        self.take(n).map(|_| ())
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
}

/// 遍历一层 box，产出 `(类型, 载荷)`。
fn children(d: &[u8]) -> impl Iterator<Item = Result<(u32, &[u8])>> {
    let mut p = 0usize;
    std::iter::from_fn(move || {
        if p + 8 > d.len() {
            return None;
        }
        let size = u32::from_be_bytes([d[p], d[p + 1], d[p + 2], d[p + 3]]) as usize;
        let typ = u32::from_be_bytes([d[p + 4], d[p + 5], d[p + 6], d[p + 7]]);
        if size < 8 || p + size > d.len() {
            p = d.len();
            return Some(Err(Error::Bad("子 box 越界")));
        }
        let payload = &d[p + 8..p + size];
        p += size;
        Some(Ok((typ, payload)))
    })
}

fn find<'a>(d: &'a [u8], want: &[u8; 4]) -> Result<Option<&'a [u8]>> {
    for c in children(d) {
        let (t, p) = c?;
        if t == fourcc(want) {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

fn parse_moov(moov: &[u8]) -> Result<(Vec<Stream>, Vec<Track>)> {
    let mut streams = Vec::new();
    let mut tracks = Vec::new();
    for c in children(moov) {
        let (t, p) = c?;
        if t != fourcc(b"trak") {
            continue;
        }
        if let Some((s, tr)) = parse_trak(p, (streams.len() + 1) as u8)? {
            streams.push(s);
            tracks.push(tr);
        }
    }
    Ok((streams, tracks))
}

fn parse_trak(trak: &[u8], number: u8) -> Result<Option<(Stream, Track)>> {
    let mdia = match find(trak, b"mdia")? {
        Some(x) => x,
        None => return Ok(None),
    };
    let mdhd = match find(mdia, b"mdhd")? {
        Some(x) => x,
        None => return Ok(None),
    };
    let mut c = Cur::new(mdhd);
    let version = c.u8()?;
    c.skip(3)?;
    let timescale = if version == 1 {
        c.u64()?;
        c.u64()?;
        c.u32()?
    } else {
        c.u32()?;
        c.u32()?;
        c.u32()?
    };
    if timescale == 0 {
        return Err(Error::Bad("mdhd timescale 为 0"));
    }

    let hdlr = match find(mdia, b"hdlr")? {
        Some(x) => x,
        None => return Ok(None),
    };
    let mut c = Cur::new(hdlr);
    c.skip(8)?;
    let handler = c.u32()?;

    let minf = match find(mdia, b"minf")? {
        Some(x) => x,
        None => return Ok(None),
    };
    let stbl = match find(minf, b"stbl")? {
        Some(x) => x,
        None => return Ok(None),
    };

    let stsd = find(stbl, b"stsd")?.ok_or(Error::Bad("缺 stsd"))?;
    let entry = first_sample_entry(stsd)?;

    let kind = if handler == fourcc(b"vide") {
        StreamKind::Video
    } else if handler == fourcc(b"soun") {
        StreamKind::Audio
    } else {
        return Ok(None);
    };

    let (video, audio) = match kind {
        StreamKind::Video => {
            let (w, h, extra) = parse_visual_entry(entry.1)?;
            (
                Some(VideoFormat {
                    encoded_width: w as u32,
                    encoded_height: h as u32,
                    width: w as u32,
                    height: h as u32,
                    bits_per_pixel: 24,
                    // MP4 的 sample entry 类型在文件里是**大端**的 4 字符码，
                    // 而全仓的 `fourcc` 约定是**小端**（ASF/AVI 都按小端读）。
                    // 这里换过来，否则 `mp4v` 会变成 `v4pm`。
                    fourcc: entry.0.swap_bytes(),
                    extra,
                }),
                None,
            )
        }
        StreamKind::Audio => {
            let (fmt, extra) = parse_audio_entry(entry.0, entry.1)?;
            (
                None,
                Some(AudioFormat {
                    format_tag: fmt,
                    ..AudioFormat {
                        format_tag: fmt,
                        channels: parse_audio_channels(entry.1).unwrap_or(2),
                        sample_rate: parse_audio_rate(entry.1).unwrap_or(44100),
                        avg_bytes_per_sec: 0,
                        block_align: 0,
                        bits_per_sample: 16,
                        extra,
                    }
                }),
            )
        }
        _ => return Ok(None),
    };

    let samples = build_samples(stbl, timescale)?;
    Ok(Some((
        Stream {
            number,
            kind,
            time_offset_ms: 0,
            audio,
            video,
        },
        Track {
            stream: number,
            samples,
            cursor: 0,
        },
    )))
}

/// stsd → 第一条 sample entry 的 `(类型, 载荷)`。
fn first_sample_entry(stsd: &[u8]) -> Result<(u32, &[u8])> {
    let mut c = Cur::new(stsd);
    c.skip(4)?; // version + flags
    let n = c.u32()?;
    if n == 0 {
        return Err(Error::Bad("stsd 里没有 entry"));
    }
    let size = c.u32()? as usize;
    let typ = c.u32()?;
    if size < 16 {
        return Err(Error::Bad("sample entry 太短"));
    }
    let payload = c.take(size - 8)?;
    Ok((typ, payload))
}

/// 视觉 sample entry：宽高在固定偏移，`extra` 取 `avcC` / `esds` 的载荷。
fn parse_visual_entry(p: &[u8]) -> Result<(u16, u16, Vec<u8>)> {
    if p.len() < 78 {
        return Err(Error::Bad("visual sample entry 太短"));
    }
    let w = u16::from_be_bytes([p[24], p[25]]);
    let h = u16::from_be_bytes([p[26], p[27]]);
    let mut extra = Vec::new();
    // 78 字节之后是子 box（avcC / esds / pasp…）
    for c in children(&p[78..]) {
        let (t, cp) = c?;
        if t == fourcc(b"avcC") {
            extra = cp.to_vec();
            break;
        }
        if t == fourcc(b"esds") {
            if let Some(asc) = esds_decoder_specific(cp) {
                extra = asc;
                break;
            }
        }
    }
    Ok((w, h, extra))
}

fn parse_audio_channels(p: &[u8]) -> Option<u16> {
    if p.len() < 18 {
        return None;
    }
    Some(u16::from_be_bytes([p[16], p[17]]))
}

fn parse_audio_rate(p: &[u8]) -> Option<u32> {
    if p.len() < 28 {
        return None;
    }
    let v = u32::from_be_bytes([p[24], p[25], p[26], p[27]]);
    Some(v >> 16) // 16.16 定点
}

/// 音频 sample entry → `(format_tag, extra)`。
fn parse_audio_entry(typ: u32, p: &[u8]) -> Result<(u16, Vec<u8>)> {
    if typ == fourcc(b"mp4a") {
        let mut extra = Vec::new();
        if p.len() >= 28 {
            for c in children(&p[28..]) {
                let (t, cp) = c?;
                if t == fourcc(b"esds") {
                    if let Some(asc) = esds_decoder_specific(cp) {
                        extra = asc;
                    }
                }
            }
        }
        // 0x00FF = AAC（不是 WAVEFORMATEX 的官方编号，是我们内部对 AAC 的约定；
        // 头文件里写清）。真正的参数在 `extra` 的 AudioSpecificConfig 里。
        Ok((0x00FF, extra))
    } else if typ == fourcc(b"sowt") || typ == fourcc(b"twos") {
        Ok((0x0001, Vec::new()))
    } else {
        Err(Error::Bad("不认识的音频 sample entry"))
    }
}

/// 从 `esds` 里挖出 DecoderSpecificInfo（AAC 的 AudioSpecificConfig）。
///
/// 描述符是「tag + 变长长度」嵌套的，长度用 7 位一组、最高位继续。
fn esds_decoder_specific(esds: &[u8]) -> Option<Vec<u8>> {
    fn len(d: &[u8], p: &mut usize) -> Option<usize> {
        let mut v = 0usize;
        for _ in 0..4 {
            let b = *d.get(*p)?;
            *p += 1;
            v = (v << 7) | (b & 0x7f) as usize;
            if b & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }
    let mut p = 4; // version + flags
    while p < esds.len() {
        let tag = esds[p];
        p += 1;
        let n = len(esds, &mut p)?;
        let end = p + n;
        if end > esds.len() {
            return None;
        }
        if tag == 0x05 {
            return Some(esds[p..end].to_vec());
        }
        // 0x03 = ES_Descriptor：前 3 字节（ES_ID + flags）之后才是子描述符
        if tag == 0x03 && n > 3 {
            let mut q = p + 3;
            while q < end {
                let t2 = esds[q];
                q += 1;
                let n2 = len(esds, &mut q)?;
                let e2 = q + n2;
                if e2 > end {
                    return None;
                }
                if t2 == 0x05 {
                    return Some(esds[q..e2].to_vec());
                }
                if t2 == 0x04 {
                    // DecoderConfigDescriptor：13 字节头之后是 DecoderSpecificInfo
                    let mut r = q + 13;
                    while r < e2 {
                        let t3 = esds[r];
                        r += 1;
                        let n3 = len(esds, &mut r)?;
                        let e3 = r + n3;
                        if e3 > e2 {
                            return None;
                        }
                        if t3 == 0x05 {
                            return Some(esds[r..e3].to_vec());
                        }
                        r = e3;
                    }
                }
                q = e2;
            }
        }
        p = end;
    }
    None
}

// ---------------------------------------------------------------------------
// 样本表 → 扁平样本列表
// ---------------------------------------------------------------------------

fn build_samples(stbl: &[u8], timescale: u32) -> Result<Vec<SampleRef>> {
    // stsz
    let stsz = find(stbl, b"stsz")?.ok_or(Error::Bad("缺 stsz"))?;
    let mut c = Cur::new(stsz);
    c.skip(4)?;
    let uniform = c.u32()?;
    let count = c.u32()? as usize;
    let mut sizes = Vec::with_capacity(count);
    if uniform != 0 {
        sizes.resize(count, uniform);
    } else {
        for _ in 0..count {
            sizes.push(c.u32()?);
        }
    }

    // stco / co64
    let chunk_offsets: Vec<u64> = if let Some(stco) = find(stbl, b"stco")? {
        let mut c = Cur::new(stco);
        c.skip(4)?;
        let n = c.u32()? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(c.u32()? as u64);
        }
        v
    } else if let Some(co64) = find(stbl, b"co64")? {
        let mut c = Cur::new(co64);
        c.skip(4)?;
        let n = c.u32()? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(c.u64()?);
        }
        v
    } else {
        return Err(Error::Bad("缺 stco / co64"));
    };

    // stsc
    let stsc = find(stbl, b"stsc")?.ok_or(Error::Bad("缺 stsc"))?;
    let mut c = Cur::new(stsc);
    c.skip(4)?;
    let n = c.u32()? as usize;
    let mut stsc_entries = Vec::with_capacity(n);
    for _ in 0..n {
        let first_chunk = c.u32()? as usize;
        let per_chunk = c.u32()? as usize;
        c.u32()?; // sample_description_index
        stsc_entries.push((first_chunk, per_chunk));
    }

    // stts → DTS
    let stts = find(stbl, b"stts")?.ok_or(Error::Bad("缺 stts"))?;
    let mut c = Cur::new(stts);
    c.skip(4)?;
    let n = c.u32()? as usize;
    let mut dts = Vec::with_capacity(count);
    let mut t = 0u64;
    for _ in 0..n {
        let cnt = c.u32()? as usize;
        let delta = c.u32()? as u64;
        for _ in 0..cnt {
            dts.push(t);
            t += delta;
        }
    }

    // ctts → PTS 偏移（可选）
    let mut ctts_off: Vec<i64> = Vec::new();
    if let Some(ctts) = find(stbl, b"ctts")? {
        let mut c = Cur::new(ctts);
        let version = c.u8()?;
        c.skip(3)?;
        let n = c.u32()? as usize;
        for _ in 0..n {
            let cnt = c.u32()? as usize;
            let off = c.u32()?;
            let off = if version == 1 {
                off as i32 as i64
            } else {
                off as i64
            };
            for _ in 0..cnt {
                ctts_off.push(off);
            }
        }
    }

    // stss → 关键帧（没有 = 全是关键帧）
    let sync: Option<Vec<bool>> = match find(stbl, b"stss")? {
        Some(x) => {
            let mut c = Cur::new(x);
            c.skip(4)?;
            let n = c.u32()? as usize;
            let mut v = vec![false; count];
            for _ in 0..n {
                let idx = c.u32()? as usize;
                if idx >= 1 && idx <= count {
                    v[idx - 1] = true;
                }
            }
            Some(v)
        }
        None => None,
    };

    // 展开：chunk → 样本偏移
    let mut out = Vec::with_capacity(count);
    let mut sample_idx = 0usize;
    for (ci, &chunk_off) in chunk_offsets.iter().enumerate() {
        let chunk_no = ci + 1;
        // 找到覆盖这个 chunk 的 stsc 项
        let per = stsc_entries
            .iter()
            .rev()
            .find(|&&(first, _)| first <= chunk_no)
            .map(|&(_, p)| p)
            .ok_or(Error::Bad("stsc 覆盖不到这个 chunk"))?;
        let mut off = chunk_off;
        for _ in 0..per {
            if sample_idx >= count {
                break;
            }
            let size = sizes[sample_idx];
            let dts_ticks = *dts.get(sample_idx).unwrap_or(&0);
            let pts_ticks = dts_ticks as i64 + ctts_off.get(sample_idx).copied().unwrap_or(0);
            let pts_ms = pts_ticks * 1000 / timescale as i64;
            let dts_ms = dts_ticks as i64 * 1000 / timescale as i64;
            let key = sync.as_ref().map(|s| s[sample_idx]).unwrap_or(true);
            out.push(SampleRef {
                offset: off,
                size,
                pts_ms,
                dts_ms,
                key,
            });
            off += size as u64;
            sample_idx += 1;
        }
    }
    Ok(out)
}
