//! MP4 / ISOBMFF 封装器。
//!
//! ## 布局（非分片 MP4，`moov` 在尾部）
//!
//! ```text
//! ftyp
//! mdat   ← 样本按写入顺序排；每个样本一个 chunk（见 `stsc`）
//! moov
//!   mvhd
//!   trak(video)  tkhd / mdia(mdhd / hdlr / minf(vmhd / dinf / stbl))
//!   trak(audio)  tkhd / mdia(mdhd / hdlr / minf(smhd / dinf / stbl))
//! ```
//!
//! `moov` 在尾部是因为 `stbl` 里的表（`stsz` / `stco` / `stts`…）要等所有样本写
//! 完才能定。整个文件在内存里拼完再吐出来 —— 与 `AsfMuxer` 的做法一致。
//!
//! ## 表怎么填（v1 的取舍）
//!
//! - **一个样本一个 chunk**：`stsc` 恒为 `(first_chunk=1, samples_per_chunk=1, 1)`，
//!   `stco` 每个样本一项。表最简单，且**不依赖调用者的交错策略** —— 代价是
//!   `stco` 每样本 4 字节（2000 帧 = 8 KB，可接受）。ffmpeg 会把同一轨的连续样本
//!   合并成 chunk，但那样得先知道交错顺序。
//! - **`stts` 做游程压缩**：等时长的帧只占一项（75 帧 25fps → 1 项）。
//! - **全是关键帧时不写 `stss`**（规范：缺席 = 所有样本都是同步点）。
//!
//! ## 还没做
//!
//! - `edts` / `elst`（编辑列表）—— 音频编码器延迟的补偿就靠它。现在不写，
//!   A/V 可能有几十毫秒的偏移。
//! - `udta` / `meta`（写我们的编码器标记）。
//! - 分片 MP4（`moof`）、`co64`（> 4 GiB）。

use crate::writer::{put_descr_len, BoxBuilder};
use crate::{Error, Result};

/// 影片时间基（`mvhd` / `tkhd` 的时长单位）。1000 = 毫秒。
const MOVIE_TIMESCALE: u32 = 1000;

/// 视频轨参数 —— 目前只支持 `mp4v`（MPEG-4 Part 2）。
#[derive(Debug, Clone)]
pub struct VideoTrack {
    pub width: u16,
    pub height: u16,
    /// 时间基（每秒多少 tick）。喂进来的 `duration` 都用这个单位。
    pub timescale: u32,
    /// `esds` 里的 DecoderSpecificInfo：MPEG-4 的 **VOL 头**（含 `00 00 01 B0` 起始码）。
    pub decoder_specific_info: Vec<u8>,
    /// 平均码率（bit/s）。写进 `esds` 与 `btrt`。
    pub avg_bitrate: u32,
}

/// 音频轨参数 —— 目前只支持 `mp4a`（AAC）。
#[derive(Debug, Clone)]
pub struct AudioTrack {
    pub channels: u16,
    pub sample_rate: u32,
    /// 时间基（每秒多少 tick）。AAC 一般就等于采样率。
    pub timescale: u32,
    /// `esds` 里的 DecoderSpecificInfo：**AudioSpecificConfig**（AAC 是 2 字节起）。
    pub decoder_specific_info: Vec<u8>,
    pub avg_bitrate: u32,
}

/// 一个已写入 `mdat` 的样本。
#[derive(Debug, Clone, Copy)]
struct SampleRec {
    /// 在**整个文件**里的绝对偏移（`stco` 要的就是它）。
    offset: u32,
    size: u32,
    /// 在**本轨时间基**下的时长（`stts` 的 delta）。
    duration: u32,
    key: bool,
}

/// MP4 封装器。
///
/// ```no_run
/// use v2v_mp4::{Mp4Muxer, VideoTrack};
///
/// let mut m = Mp4Muxer::new(VideoTrack {
///     width: 320,
///     height: 240,
///     timescale: 12800,
///     decoder_specific_info: vec![0, 0, 1, 0xb0, 0x01], // VOL 头
///     avg_bitrate: 300_000,
/// });
/// m.write_video(&[0u8; 100], true, 512).unwrap(); // 512 tick @12800 = 25 fps
/// m.write_video(&[0u8; 80], false, 512).unwrap();
/// let mp4: Vec<u8> = m.finish();
/// # let _ = mp4;
/// ```
#[derive(Debug)]
pub struct Mp4Muxer {
    out: Vec<u8>,
    /// `mdat` box 头在 `out` 里的位置（收尾时回填大小）。
    mdat_at: usize,
    video: Option<VideoTrack>,
    audio: Option<AudioTrack>,
    video_samples: Vec<SampleRec>,
    audio_samples: Vec<SampleRec>,
}

impl Mp4Muxer {
    /// 只要视频轨。
    pub fn new(video: VideoTrack) -> Self {
        Self::assemble(Some(video), None)
    }

    /// 视频轨 + 音频轨。
    pub fn with_audio(video: VideoTrack, audio: AudioTrack) -> Self {
        Self::assemble(Some(video), Some(audio))
    }

    fn assemble(video: Option<VideoTrack>, audio: Option<AudioTrack>) -> Self {
        let mut out = Vec::new();
        out.extend_from_slice(&ftyp());
        let mdat_at = out.len();
        // `mdat` 的大小要等样本写完才知道 —— 先占位，`finish()` 回填。
        out.extend_from_slice(&[0u8; 8]);
        Self {
            out,
            mdat_at,
            video,
            audio,
            video_samples: Vec::new(),
            audio_samples: Vec::new(),
        }
    }

    /// 写一个视频样本。`duration` 的单位是 `VideoTrack::timescale`。
    pub fn write_video(&mut self, data: &[u8], key_frame: bool, duration: u32) -> Result<()> {
        if self.video.is_none() {
            return Err(Error::Bad("没有视频轨"));
        }
        if duration == 0 {
            return Err(Error::Bad("样本时长为 0"));
        }
        let rec = self.push(data, duration, key_frame)?;
        self.video_samples.push(rec);
        Ok(())
    }

    /// 写一个音频样本。`duration` 的单位是 `AudioTrack::timescale`。
    pub fn write_audio(&mut self, data: &[u8], duration: u32) -> Result<()> {
        if self.audio.is_none() {
            return Err(Error::Bad("没有音频轨"));
        }
        if duration == 0 {
            return Err(Error::Bad("样本时长为 0"));
        }
        let rec = self.push(data, duration, true)?;
        self.audio_samples.push(rec);
        Ok(())
    }

    fn push(&mut self, data: &[u8], duration: u32, key: bool) -> Result<SampleRec> {
        if data.is_empty() {
            return Err(Error::Bad("空样本"));
        }
        let offset = self.out.len();
        if offset > u32::MAX as usize {
            return Err(Error::Bad("超过 4 GiB —— 需要 co64，暂未实现"));
        }
        self.out.extend_from_slice(data);
        Ok(SampleRec {
            offset: offset as u32,
            size: data.len() as u32,
            duration,
            key,
        })
    }

    /// 收尾：回填 `mdat` 大小，追加 `moov`，返回整个文件。
    pub fn finish(mut self) -> Vec<u8> {
        let mdat_size = (self.out.len() - self.mdat_at) as u32;
        self.out[self.mdat_at..self.mdat_at + 4].copy_from_slice(&mdat_size.to_be_bytes());
        self.out[self.mdat_at + 4..self.mdat_at + 8].copy_from_slice(b"mdat");

        let vdur = track_duration_ms(&self.video_samples, self.video.as_ref().map(|v| v.timescale));
        let adur = track_duration_ms(&self.audio_samples, self.audio.as_ref().map(|a| a.timescale));
        let movie_duration = vdur.max(adur);

        let mut moov = BoxBuilder::new(b"moov");
        moov.child(mvhd(movie_duration, self.track_count() as u32 + 1));
        if let Some(v) = &self.video {
            moov.child(trak_video(v, 1, &self.video_samples));
        }
        if let Some(a) = &self.audio {
            moov.child(trak_audio(a, 2, &self.audio_samples));
        }
        self.out.extend_from_slice(&moov.finish());
        self.out
    }

    fn track_count(&self) -> usize {
        usize::from(self.video.is_some()) + usize::from(self.audio.is_some())
    }
}

/// 整轨时长（毫秒）—— 把各样本在**本轨时间基**下的时长加起来再换算。
fn track_duration_ms(samples: &[SampleRec], timescale: Option<u32>) -> u64 {
    let Some(ts) = timescale else { return 0 };
    let total: u64 = samples.iter().map(|s| s.duration as u64).sum();
    total * MOVIE_TIMESCALE as u64 / ts as u64
}

// ---------------------------------------------------------------- 顶层 box

fn ftyp() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"ftyp");
    b.bytes(b"isom"); // major_brand
    b.u32(512); // minor_version
    b.bytes(b"isom");
    b.bytes(b"iso2");
    b.bytes(b"mp41");
    b.finish()
}

fn mvhd(duration_ms: u64, next_track_id: u32) -> BoxBuilder {
    let mut b = BoxBuilder::new(b"mvhd");
    b.full(0, 0);
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(MOVIE_TIMESCALE);
    b.u32(duration_ms as u32);
    b.u32(0x0001_0000); // rate
    b.u16(0x0100); // volume
    b.u16(0); // reserved
    b.zeros(8); // reserved
    b.identity_matrix();
    b.zeros(24); // pre_defined
    b.u32(next_track_id);
    b
}

fn trak_video(v: &VideoTrack, track_id: u32, samples: &[SampleRec]) -> BoxBuilder {
    let media_duration: u64 = samples.iter().map(|s| s.duration as u64).sum();
    let track_duration = media_duration * MOVIE_TIMESCALE as u64 / v.timescale as u64;

    let mut trak = BoxBuilder::new(b"trak");
    trak.child(tkhd(track_id, track_duration, v.width, v.height, 0, 0));
    trak.child(mdia(
        b"vide",
        "VideoHandler",
        v.timescale,
        media_duration,
        minf_video(v, samples),
    ));
    trak
}

fn trak_audio(a: &AudioTrack, track_id: u32, samples: &[SampleRec]) -> BoxBuilder {
    let media_duration: u64 = samples.iter().map(|s| s.duration as u64).sum();
    let track_duration = media_duration * MOVIE_TIMESCALE as u64 / a.timescale as u64;

    let mut trak = BoxBuilder::new(b"trak");
    // 音频轨的 `tkhd`：宽高为 0、音量为满、`alternate_group` 为 1（与 ffmpeg 一致）。
    trak.child(tkhd(track_id, track_duration, 0, 0, 0x0100, 1));
    trak.child(mdia(
        b"soun",
        "SoundHandler",
        a.timescale,
        media_duration,
        minf_audio(a, samples),
    ));
    trak
}

fn tkhd(track_id: u32, duration: u64, width: u16, height: u16, volume: u16, alt: u16) -> BoxBuilder {
    let mut b = BoxBuilder::new(b"tkhd");
    b.full(0, 0x0000_0003); // track_enabled | track_in_movie
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(track_id);
    b.u32(0); // reserved
    b.u32(duration as u32);
    b.zeros(8); // reserved
    b.u16(0); // layer
    b.u16(alt); // alternate_group
    b.u16(volume);
    b.u16(0); // reserved
    b.identity_matrix();
    b.u32((width as u32) << 16); // 16.16
    b.u32((height as u32) << 16);
    b
}

fn mdia(
    handler: &[u8; 4],
    name: &str,
    timescale: u32,
    duration: u64,
    minf: BoxBuilder,
) -> BoxBuilder {
    let mut b = BoxBuilder::new(b"mdia");
    b.child(mdhd(timescale, duration));
    b.child(hdlr(handler, name));
    b.child(minf);
    b
}

fn mdhd(timescale: u32, duration: u64) -> BoxBuilder {
    let mut b = BoxBuilder::new(b"mdhd");
    b.full(0, 0);
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(timescale);
    b.u32(duration as u32);
    b.u16(0x55c4); // language = 'und'（ISO-639-2/T 打包）
    b.u16(0); // pre_defined
    b
}

fn hdlr(handler: &[u8; 4], name: &str) -> BoxBuilder {
    let mut b = BoxBuilder::new(b"hdlr");
    b.full(0, 0);
    b.u32(0); // pre_defined
    b.bytes(handler);
    b.zeros(12); // reserved
    b.bytes(name.as_bytes());
    b.u8(0); // 结尾 NUL
    b
}

// ---------------------------------------------------------------- minf / stbl

fn minf_video(v: &VideoTrack, samples: &[SampleRec]) -> BoxBuilder {
    let mut vmhd = BoxBuilder::new(b"vmhd");
    vmhd.full(0, 1); // flags=1：必须写 graphicsmode / opcolor
    vmhd.u16(0); // graphicsmode
    vmhd.zeros(6); // opcolor

    let mut minf = BoxBuilder::new(b"minf");
    minf.child(vmhd);
    minf.child(dinf());
    minf.child(stbl(&mp4v_entry(v), samples, true));
    minf
}

fn minf_audio(a: &AudioTrack, samples: &[SampleRec]) -> BoxBuilder {
    let mut smhd = BoxBuilder::new(b"smhd");
    smhd.full(0, 0);
    smhd.u16(0); // balance
    smhd.u16(0); // reserved

    let mut minf = BoxBuilder::new(b"minf");
    minf.child(smhd);
    minf.child(dinf());
    // 音频不写 `stss` —— 每个样本都是同步点。
    minf.child(stbl(&mp4a_entry(a), samples, false));
    minf
}

fn dinf() -> BoxBuilder {
    let mut url = BoxBuilder::new(b"url ");
    url.full(0, 1); // flags=1：数据就在本文件里

    let mut dref = BoxBuilder::new(b"dref");
    dref.full(0, 0);
    dref.u32(1);
    dref.child(url);

    let mut dinf = BoxBuilder::new(b"dinf");
    dinf.child(dref);
    dinf
}

fn stbl(sample_entry: &[u8], samples: &[SampleRec], with_stss: bool) -> BoxBuilder {
    let mut b = BoxBuilder::new(b"stbl");
    b.child(stsd(sample_entry));
    b.child(stts(samples));
    let ss = if with_stss { stss(samples) } else { None };
    if let Some(ss) = ss {
        b.child(ss);
    }
    b.child(stsc(samples));
    b.child(stsz(samples));
    b.child(stco(samples));
    b
}

fn stsd(sample_entry: &[u8]) -> BoxBuilder {
    let mut b = BoxBuilder::new(b"stsd");
    b.full(0, 0);
    b.u32(1);
    b.raw(sample_entry);
    b
}

/// `stts` —— 时长做**游程压缩**（等长的帧合成一项）。
fn stts(samples: &[SampleRec]) -> BoxBuilder {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for s in samples {
        match runs.last_mut() {
            Some(last) if last.1 == s.duration => last.0 += 1,
            _ => runs.push((1, s.duration)),
        }
    }
    let mut b = BoxBuilder::new(b"stts");
    b.full(0, 0);
    b.u32(runs.len() as u32);
    for (count, delta) in runs {
        b.u32(count);
        b.u32(delta);
    }
    b
}

/// `stss` —— 关键帧下标（1 起）。**全是关键帧时返回 `None`**（规范：缺席即全同步）。
fn stss(samples: &[SampleRec]) -> Option<BoxBuilder> {
    if samples.iter().all(|s| s.key) {
        return None;
    }
    let mut b = BoxBuilder::new(b"stss");
    b.full(0, 0);
    let keys: Vec<u32> = samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.key)
        .map(|(i, _)| i as u32 + 1)
        .collect();
    b.u32(keys.len() as u32);
    for k in keys {
        b.u32(k);
    }
    Some(b)
}

/// `stsc` —— 一个样本一个 chunk，所以恒为一项。
fn stsc(samples: &[SampleRec]) -> BoxBuilder {
    let mut b = BoxBuilder::new(b"stsc");
    b.full(0, 0);
    if samples.is_empty() {
        b.u32(0);
    } else {
        b.u32(1);
        b.u32(1); // first_chunk
        b.u32(1); // samples_per_chunk
        b.u32(1); // sample_description_index
    }
    b
}

fn stsz(samples: &[SampleRec]) -> BoxBuilder {
    let mut b = BoxBuilder::new(b"stsz");
    b.full(0, 0);
    b.u32(0); // sample_size = 0 → 逐样本给大小
    b.u32(samples.len() as u32);
    for s in samples {
        b.u32(s.size);
    }
    b
}

fn stco(samples: &[SampleRec]) -> BoxBuilder {
    let mut b = BoxBuilder::new(b"stco");
    b.full(0, 0);
    b.u32(samples.len() as u32);
    for s in samples {
        b.u32(s.offset);
    }
    b
}

// ---------------------------------------------------------------- sample entry

/// `compressorname`：32 字节，**首字节是长度**，其余是名字 + 补零。
fn compressor_name(name: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let n = name.len().min(31);
    out[0] = n as u8;
    out[1..1 + n].copy_from_slice(&name[..n]);
    out
}

fn mp4v_entry(v: &VideoTrack) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mp4v");
    b.zeros(6); // reserved
    b.u16(1); // data_reference_index
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    b.zeros(12); // pre_defined[3]
    b.u16(v.width);
    b.u16(v.height);
    b.u32(0x0048_0000); // horizresolution = 72 dpi
    b.u32(0x0048_0000); // vertresolution
    b.u32(0); // reserved
    b.u16(1); // frame_count
    b.bytes(&compressor_name(b"v2video mpeg4"));
    b.u16(0x0018); // depth = 24
    b.i16(-1); // pre_defined
    b.child(esds(1, 0x20, 0x04, v.avg_bitrate, &v.decoder_specific_info));
    b.child(btrt(v.avg_bitrate));
    b.finish()
}

fn mp4a_entry(a: &AudioTrack) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mp4a");
    b.zeros(6); // reserved
    b.u16(1); // data_reference_index
    b.zeros(8); // reserved（version / revision / vendor）
    b.u16(a.channels);
    b.u16(16); // samplesize
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    b.u32(a.sample_rate << 16); // 16.16
    b.child(esds(2, 0x40, 0x05, a.avg_bitrate, &a.decoder_specific_info));
    b.child(btrt(a.avg_bitrate));
    b.finish()
}

/// `btrt`（MPEG-4 系统）：缓冲提示。三个值都填同一个码率。
fn btrt(bitrate: u32) -> BoxBuilder {
    let mut b = BoxBuilder::new(b"btrt");
    b.u32(bitrate); // bufferSizeDB
    b.u32(bitrate); // maxBitrate
    b.u32(bitrate); // avgBitrate
    b
}

/// `esds`（MPEG-4 系统）—— 三个嵌套的描述符：
///
/// ```text
/// ES_Descriptor(0x03)
///   ES_ID(2) + flags(1)
///   DecoderConfigDescriptor(0x04)
///     objectTypeIndication(1) + streamType/upStream/reserved(1)
///     bufferSizeDB(3) + maxBitrate(4) + avgBitrate(4)
///     DecoderSpecificInfo(0x05)   ← VOL 头 / AudioSpecificConfig
///   SLConfigDescriptor(0x06)      ← 定长 [0x06, 0x01, 0x02]
/// ```
fn esds(es_id: u16, object_type: u8, stream_type: u8, bitrate: u32, dsi: &[u8]) -> BoxBuilder {
    let mut dsi_desc = Vec::new();
    dsi_desc.push(0x05);
    put_descr_len(&mut dsi_desc, dsi.len());
    dsi_desc.extend_from_slice(dsi);

    let mut dcd_body = Vec::new();
    dcd_body.push(object_type);
    dcd_body.push((stream_type << 2) | 0x01); // upStream=0, reserved=1
    dcd_body.extend_from_slice(&[0, 0, 0]); // bufferSizeDB
    dcd_body.extend_from_slice(&bitrate.to_be_bytes()); // maxBitrate
    dcd_body.extend_from_slice(&bitrate.to_be_bytes()); // avgBitrate
    dcd_body.extend_from_slice(&dsi_desc);

    let mut dcd = Vec::new();
    dcd.push(0x04);
    put_descr_len(&mut dcd, dcd_body.len());
    dcd.extend_from_slice(&dcd_body);

    let mut es_body = Vec::new();
    es_body.extend_from_slice(&es_id.to_be_bytes());
    es_body.push(0x00); // 无依赖流 / 无 URL / 无 OCR
    es_body.extend_from_slice(&dcd);
    es_body.extend_from_slice(&[0x06, 0x01, 0x02]); // SLConfigDescriptor

    let mut es = Vec::new();
    es.push(0x03);
    put_descr_len(&mut es, es_body.len());
    es.extend_from_slice(&es_body);

    let mut b = BoxBuilder::new(b"esds");
    b.full(0, 0);
    b.bytes(&es);
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video() -> VideoTrack {
        VideoTrack {
            width: 320,
            height: 240,
            timescale: 12800,
            decoder_specific_info: vec![0, 0, 1, 0xb0, 0x01, 0x00],
            avg_bitrate: 300_000,
        }
    }

    #[test]
    fn ftyp_matches_the_reference_layout() {
        let f = ftyp();
        assert_eq!(f.len(), 28);
        assert_eq!(&f[4..8], b"ftyp");
        assert_eq!(&f[8..12], b"isom");
        assert_eq!(&f[12..16], &512u32.to_be_bytes());
        assert_eq!(&f[16..], b"isomiso2mp41");
    }

    #[test]
    fn video_only_file_has_one_trak_and_valid_top_level_boxes() {
        let mut m = Mp4Muxer::new(video());
        m.write_video(&[1, 2, 3], true, 512).unwrap();
        m.write_video(&[4, 5], false, 512).unwrap();
        let f = m.finish();

        // 顶层：ftyp / mdat / moov，顺序且大小自洽
        let mut p = 0;
        let mut types = Vec::new();
        while p < f.len() {
            let size = u32::from_be_bytes(f[p..p + 4].try_into().unwrap()) as usize;
            types.push(String::from_utf8_lossy(&f[p + 4..p + 8]).to_string());
            p += size;
        }
        assert_eq!(types, vec!["ftyp", "mdat", "moov"]);
        assert_eq!(p, f.len(), "顶层 box 必须正好铺满整个文件");
    }

    #[test]
    fn mdat_size_covers_exactly_the_samples() {
        let mut m = Mp4Muxer::new(video());
        m.write_video(&[0u8; 10], true, 512).unwrap();
        m.write_video(&[0u8; 20], true, 512).unwrap();
        let f = m.finish();
        let mdat_size = u32::from_be_bytes(f[28..32].try_into().unwrap()) as usize;
        assert_eq!(mdat_size, 8 + 10 + 20, "mdat = 头 + 两个样本");
        // stco 的第一个偏移应当是 ftyp(28) + mdat 头(8)
        // stco 的搜索命中的是 type 字段：size(4) type(4) ver/flags(4) count(4) 之后才是条目
        let stco = f.windows(4).position(|w| w == b"stco").expect("有 stco");
        let count = u32::from_be_bytes(f[stco + 8..stco + 12].try_into().unwrap());
        assert_eq!(count, 2, "两个样本 → 两条 stco（一 chunk 一样本）");
        let first_off = u32::from_be_bytes(f[stco + 12..stco + 16].try_into().unwrap());
        assert_eq!(first_off as usize, 28 + 8, "ftyp(28) + mdat 头(8)");
        let second_off = u32::from_be_bytes(f[stco + 16..stco + 20].try_into().unwrap());
        assert_eq!(second_off as usize, 28 + 8 + 10, "第二个样本紧接在第一个之后");
    }

    #[test]
    fn stss_is_omitted_when_every_frame_is_a_key_frame() {
        let samples = vec![
            SampleRec { offset: 0, size: 1, duration: 512, key: true },
            SampleRec { offset: 1, size: 1, duration: 512, key: true },
        ];
        assert!(stss(&samples).is_none(), "全关键帧 → 不写 stss");

        let mixed = vec![
            SampleRec { offset: 0, size: 1, duration: 512, key: true },
            SampleRec { offset: 1, size: 1, duration: 512, key: false },
            SampleRec { offset: 2, size: 1, duration: 512, key: true },
        ];
        let b = stss(&mixed).expect("有非关键帧 → 要写").finish();
        assert_eq!(&b[4..8], b"stss");
        assert_eq!(u32::from_be_bytes(b[12..16].try_into().unwrap()), 2);
        assert_eq!(u32::from_be_bytes(b[16..20].try_into().unwrap()), 1);
        assert_eq!(u32::from_be_bytes(b[20..24].try_into().unwrap()), 3);
    }

    #[test]
    fn stts_run_length_compresses_equal_durations() {
        let samples: Vec<SampleRec> = (0..75)
            .map(|i| SampleRec { offset: i, size: 1, duration: 512, key: i == 0 })
            .collect();
        let b = stts(&samples).finish();
        assert_eq!(u32::from_be_bytes(b[12..16].try_into().unwrap()), 1, "只一项");
        assert_eq!(u32::from_be_bytes(b[16..20].try_into().unwrap()), 75);
        assert_eq!(u32::from_be_bytes(b[20..24].try_into().unwrap()), 512);
    }

    #[test]
    fn esds_uses_minimal_descriptor_lengths() {
        // 6 字节的 DSI → ES_Descriptor 总长 < 128，长度字段都应是 1 字节
        let b = esds(1, 0x20, 0x04, 300_000, &[0, 0, 1, 0xb0, 1, 0]).finish();
        assert_eq!(&b[4..8], b"esds");
        let body = &b[12..];
        assert_eq!(body[0], 0x03, "ES_Descriptor");
        assert_eq!(body[1], (body.len() - 2) as u8, "长度 = 后面剩下的字节数");
        assert!(body[1] < 0x80, "小于 128 时不该用多字节长度");
        assert_eq!(body[2..4], [0, 1], "ES_ID");
        assert_eq!(body[4], 0, "flags");
        assert_eq!(body[5], 0x04, "DecoderConfigDescriptor");
        // DCD 的长度只覆盖 objectType…DSI，不含后面那 3 字节 SLConfigDescriptor
        assert_eq!(body[6], (body.len() - 10) as u8, "DCD 长度不含 SLConfig");
        assert_eq!(body[7], 0x20, "objectTypeIndication = MPEG-4 Visual");
        assert_eq!(body[8], 0x11, "streamType=visual(4)<<2 | 1");
    }

    #[test]
    fn write_without_the_matching_track_is_an_error() {
        let mut m = Mp4Muxer::new(video());
        assert!(m.write_audio(&[1], 1024).is_err(), "没有音频轨");
        assert!(m.write_video(&[], false, 512).is_err(), "空样本");
        assert!(m.write_video(&[1], true, 0).is_err(), "零时长");
    }
}
