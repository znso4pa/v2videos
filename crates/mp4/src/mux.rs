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
//! ## 表怎么填
//!
//! - **chunk = 一段连续的字节**：按「本样本的偏移是否紧接上一个样本的末尾」
//!   切 chunk，`stsc` 对「每 chunk 样本数」再做游程压缩。
//!   这样**不依赖调用者的交错策略** —— 全视频再全音频就是各 1 个 chunk，
//!   逐帧交错就自动切成很多小 chunk，两种都能被正确读回。
//! - **`stts` 做游程压缩**：等时长的帧只占一项（75 帧 25fps → 1 项）。
//! - **`stsz` 等长时用 `default_sample_size`**：PCM 轨全靠这条，
//!   否则 13 万个样本的表就是 512 KB。
//! - **全是关键帧时不写 `stss`**（规范：缺席 = 所有样本都是同步点）。
//! - **描述符长度用最简形式**（`put_descr_len`）。ffmpeg 恒写 4 字节，
//!   我们按规范写最简 —— 已实测 ffmpeg 能正常读（见 `tests/real_sample.rs`）。
//!
//! ## 音频：`mp4a` 还是 PCM
//!
//! MP4 **不接受 WMA**（`Could not find tag for codec wmav1`），所以 WMV→MP4
//! 要么上 AAC 编码器，要么走未压缩 PCM。`sowt`（PCM S16LE）这条路已实测
//! 可用：把 ffmpeg 产的 MOV 的 `ftyp` 主品牌从 `qt  ` 改成 `isom` 之后，
//! ffprobe 照样认 `codec_name=pcm_s16le`，`ffmpeg -f s16le` 也能原样解出来。
//!
//! ## 还没做
//!
//! - `edts` / `elst`（编辑列表）—— 解码器延迟的补偿就靠它。
//!   ffmpeg 给 WMV2 的 MP4 就写了 `elst`（空编辑 40 ms + 正片）。
//! - `pasp`（像素宽高比）、`udta` / `meta`（编码器标记）。
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

/// 音频轨的编解码类型 —— 决定 `stsd` 里写哪种 sample entry。
#[derive(Debug, Clone)]
pub enum AudioCodec {
    /// `mp4a` + `esds`：ES 型编码（AAC / MP3…）。
    Es {
        /// `objectTypeIndication`：AAC = `0x40`，MP3 = `0x6B`。
        object_type: u8,
        /// `DecoderSpecificInfo`（AAC 是 AudioSpecificConfig）。
        decoder_specific_info: Vec<u8>,
    },
    /// `sowt`（小端）/ `twos`（大端）：**未压缩 PCM**，样本就是裸采样字节。
    ///
    /// **一个样本 = 一个采样点**（`stsz` 的 default size = 声道数 × 字节数），
    /// 时间基就等于采样率，每个样本时长 1。ffmpeg 的 `sowt` 轨就是这么填的
    /// （实测：`stsz` default 4、count 131072、`stts` 一项 `(131072, 1)`）。
    Pcm { bits: u16, big_endian: bool },
}

/// 音频轨参数。
#[derive(Debug, Clone)]
pub struct AudioTrack {
    pub channels: u16,
    pub sample_rate: u32,
    /// 时间基（每秒多少 tick）。PCM 就等于采样率；AAC 一般也是。
    pub timescale: u32,
    pub codec: AudioCodec,
    /// 平均码率（bit/s）。只写进 `esds` 与 `btrt`；PCM 用不上。
    pub avg_bitrate: u32,
    /// **编码器预热样本数**（`priming`）—— 解码器要跳过开头这么多个采样点。
    ///
    /// AAC 这类「MDCT + 50% 重叠」的编码器**必然有延迟**：编码器开头灌进去的
    /// 是补零的窗，解码出来的头一两个块是渐入段（实测 2048 个采样点）。
    /// 不补偿的话音频整体比视频晚 2048/44100 ≈ 46 ms，口型会对不上。
    ///
    /// 补偿用 `edts`/`elst`（编辑列表），**不是**「把开头编的静音扔掉」——
    /// 后者会让编码器失去预热、第一帧的质量反而变差。
    /// PCM 这类无延迟的编码填 0。
    pub priming: u32,
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

        let vdur = track_duration_ms(
            &self.video_samples,
            self.video.as_ref().map(|v| v.timescale),
        );
        let adur = track_duration_ms(
            &self.audio_samples,
            self.audio.as_ref().map(|a| a.timescale),
        );
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
    // `edts` / `elst`：告诉播放器「跳过开头 `priming` 个采样点」。
    //
    // ⚠️ `media_time` 用的单位是**本轨的 media timescale**（音频轨就是采样率），
    // 而 `segment_duration` 用的是**影片时间轴**（`mvhd` 的 timescale = 1000）。
    // 两个单位不一样 —— AGENTS.md 里就记着这条（当初在别的项目上踩过）。
    if a.priming > 0 {
        trak.child(edts(a.priming, a.timescale, track_duration));
    }
    trak.child(mdia(
        b"soun",
        "SoundHandler",
        a.timescale,
        media_duration,
        minf_audio(a, samples),
    ));
    trak
}

/// `edts`（edit box）—— 一条 `elst` 条目：「跳过开头的编码器预热，之后是正片」。
///
/// ## 两个单位不一样，这是最容易写错的地方
///
/// ```text
///   media_time       ← 本轨的 media timescale（音频轨 = 采样率）
///   segment_duration ← movie timescale（`mvhd` 的 1000 = 毫秒）
/// ```
///
/// 所以 `priming` 个采样点换算到影片时间轴是
/// `priming * MOVIE_TIMESCALE / media_timescale`。
/// **不能直接拿 `priming` 去减 `track_duration`** —— 那是笔「秒减采样点」的糊涂账。
///
/// AGENTS.md 里记着同一条坑（`elst.segment_duration` 是 movie 时间轴、
/// `media_time` 是媒体时间轴，两个单位不一样）。
fn edts(priming: u32, media_timescale: u32, track_duration: u64) -> BoxBuilder {
    // 预热段在**影片时间轴**上的长度。
    let priming_movie =
        u64::from(priming) * u64::from(MOVIE_TIMESCALE) / u64::from(media_timescale).max(1);

    let mut elst = BoxBuilder::new(b"elst");
    elst.full(0, 0); // version 0 / flags
    elst.u32(1); // entry_count
                 // segment_duration：正片在影片时间轴上的长度 = 总时长 − 预热段。
    elst.u32(track_duration.saturating_sub(priming_movie) as u32);
    // media_time：**从媒体时间轴的哪里开始播** = 预热了多少个采样点。
    // `-1` 在 `elst` 里是「空编辑」的约定值，但正片编辑用真实的起始点。
    elst.u32(priming);
    elst.u16(1); // media_rate_integer = 1.0
    elst.u16(0); // media_rate_fraction

    let mut edts = BoxBuilder::new(b"edts");
    edts.child(elst);
    edts
}

fn tkhd(
    track_id: u32,
    duration: u64,
    width: u16,
    height: u16,
    volume: u16,
    alt: u16,
) -> BoxBuilder {
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
    minf.child(stbl(&audio_entry(a), samples, false));
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

/// 把样本按「在文件里是否首尾相接」切成 chunk。
///
/// 这是 `stsc` / `stco` 的本意：**一个 chunk 是一段连续的字节**。
///
/// v1 曾用一个样本一个 chunk —— 对视频勉强能用，对 PCM 会直接爆：
/// 3 秒 44.1 kHz 立体声 = 13 万个样本 → 13 万条 `stco`（512 KB），
/// 而整条轨在文件里本来就是连续的一段，本该只有 1 条。
///
/// 按「是否连续」切还有个好处：**不依赖调用者的交错策略**。
/// 全视频再全音频 → 两条轨各 1 个 chunk；逐帧交错 → 自动切成很多小 chunk。
/// 两种都能被解码器正确读回。
fn chunk_runs(samples: &[SampleRec]) -> Vec<usize> {
    let mut runs: Vec<usize> = Vec::new();
    let mut cur = 0usize;
    for (i, s) in samples.iter().enumerate() {
        if i > 0 {
            let prev = samples[i - 1];
            let contiguous = prev.offset as u64 + prev.size as u64 == s.offset as u64;
            if !contiguous {
                runs.push(cur);
                cur = 0;
            }
        }
        cur += 1;
    }
    if cur > 0 {
        runs.push(cur);
    }
    runs
}

fn stbl(sample_entry: &[u8], samples: &[SampleRec], with_stss: bool) -> BoxBuilder {
    let runs = chunk_runs(samples);
    let mut b = BoxBuilder::new(b"stbl");
    b.child(stsd(sample_entry));
    b.child(stts(samples));
    // 音频不写 `stss` —— 每个样本都是同步点。
    if with_stss {
        if let Some(ss) = stss(samples) {
            b.child(ss);
        }
    }
    b.child(stsc(&runs));
    b.child(stsz(samples));
    b.child(stco(samples, &runs));
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

/// `stsc` —— 「每 chunk 样本数」做**游程压缩**。
fn stsc(runs: &[usize]) -> BoxBuilder {
    let mut rle: Vec<(u32, u32)> = Vec::new(); // (chunk 数, 每 chunk 样本数)
    for &n in runs {
        match rle.last_mut() {
            Some(last) if last.1 == n as u32 => last.0 += 1,
            _ => rle.push((1, n as u32)),
        }
    }

    let mut b = BoxBuilder::new(b"stsc");
    b.full(0, 0);
    b.u32(rle.len() as u32);
    let mut first_chunk = 1u32;
    for (count, per_chunk) in rle {
        b.u32(first_chunk);
        b.u32(per_chunk);
        b.u32(1); // sample_description_index
        first_chunk += count;
    }
    b
}

fn stsz(samples: &[SampleRec]) -> BoxBuilder {
    // 全部等长 → 用 default_sample_size，省掉整张表。
    // PCM 轨全靠这条：13 万个样本 × 4 字节的表就是 512 KB。
    let uniform = samples
        .first()
        .map(|s| s.size)
        .filter(|n| samples.iter().all(|s| s.size == *n));

    let mut b = BoxBuilder::new(b"stsz");
    b.full(0, 0);
    b.u32(uniform.unwrap_or(0)); // 0 = 逐样本给大小
    b.u32(samples.len() as u32);
    if uniform.is_none() {
        for s in samples {
            b.u32(s.size);
        }
    }
    b
}

/// `stco` —— 每个 chunk 的**首样本**偏移。
fn stco(samples: &[SampleRec], runs: &[usize]) -> BoxBuilder {
    let mut b = BoxBuilder::new(b"stco");
    b.full(0, 0);
    b.u32(runs.len() as u32);
    let mut i = 0usize;
    for &n in runs {
        b.u32(samples[i].offset);
        i += n;
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

/// `AudioSampleEntry` 的固定 28 字节头（`mp4a` / `sowt` / `twos` 共用）。
fn audio_sample_entry(b: &mut BoxBuilder, a: &AudioTrack, bits: u16) {
    b.zeros(6); // reserved
    b.u16(1); // data_reference_index
    b.zeros(8); // version / revision / vendor
    b.u16(a.channels);
    b.u16(bits); // samplesize
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    b.u32(a.sample_rate << 16); // 16.16
}

fn audio_entry(a: &AudioTrack) -> Vec<u8> {
    match &a.codec {
        AudioCodec::Es {
            object_type,
            decoder_specific_info,
        } => {
            let mut b = BoxBuilder::new(b"mp4a");
            audio_sample_entry(&mut b, a, 16);
            b.child(esds(
                2,
                *object_type,
                0x05,
                a.avg_bitrate,
                decoder_specific_info,
            ));
            b.child(btrt(a.avg_bitrate));
            b.finish()
        }
        AudioCodec::Pcm { bits, big_endian } => {
            // 未压缩：没有 `esds`，也没有 `btrt` —— 只有固定头。
            let mut b = BoxBuilder::new(if *big_endian { b"twos" } else { b"sowt" });
            audio_sample_entry(&mut b, a, *bits);
            b.finish()
        }
    }
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
        // stco 的搜索命中的是 type 字段：size(4) type(4) ver/flags(4) count(4) 之后才是条目
        let stco = f.windows(4).position(|w| w == b"stco").expect("有 stco");
        let count = u32::from_be_bytes(f[stco + 8..stco + 12].try_into().unwrap());
        assert_eq!(count, 1, "两个样本在文件里首尾相接 → 只该有 1 个 chunk");
        let first_off = u32::from_be_bytes(f[stco + 12..stco + 16].try_into().unwrap());
        assert_eq!(first_off as usize, 28 + 8, "ftyp(28) + mdat 头(8)");
    }

    #[test]
    fn contiguous_samples_merge_into_one_chunk_but_a_gap_splits_it() {
        let mk = |off, size| SampleRec {
            offset: off,
            size,
            duration: 1,
            key: true,
        };
        // 0..10 连着，然后跳到 100（中间有别轨的样本）
        let s = vec![mk(0, 10), mk(10, 20), mk(100, 5)];
        assert_eq!(chunk_runs(&s), vec![2, 1], "前两个一块，第三个自己一块");

        // 全部连续
        assert_eq!(chunk_runs(&[mk(0, 4), mk(4, 4), mk(8, 4)]), vec![3]);

        // 交错：每两个之间都跳
        assert_eq!(chunk_runs(&[mk(0, 1), mk(9, 1), mk(18, 1)]), vec![1, 1, 1]);

        // 空
        assert!(chunk_runs(&[]).is_empty());
    }

    #[test]
    fn stsz_falls_back_to_default_size_when_every_sample_is_equal() {
        let s: Vec<SampleRec> = (0..1000)
            .map(|i| SampleRec {
                offset: i * 4,
                size: 4,
                duration: 1,
                key: true,
            })
            .collect();
        let b = stsz(&s).finish();
        assert_eq!(
            u32::from_be_bytes(b[12..16].try_into().unwrap()),
            4,
            "default size"
        );
        assert_eq!(u32::from_be_bytes(b[16..20].try_into().unwrap()), 1000);
        assert_eq!(b.len(), 20, "省掉了整张表");

        // 不等长时必须逐项写
        let s = vec![
            SampleRec {
                offset: 0,
                size: 4,
                duration: 1,
                key: true,
            },
            SampleRec {
                offset: 4,
                size: 8,
                duration: 1,
                key: true,
            },
        ];
        let b = stsz(&s).finish();
        assert_eq!(u32::from_be_bytes(b[12..16].try_into().unwrap()), 0);
        assert_eq!(b.len(), 20 + 8);
    }

    #[test]
    fn stsc_run_length_compresses_the_chunking_pattern() {
        // chunk 大小：1,1,1,2,2,2 → 两项
        let b = stsc(&[1, 1, 1, 2, 2, 2]).finish();
        assert_eq!(u32::from_be_bytes(b[12..16].try_into().unwrap()), 2, "两项");
        assert_eq!(
            u32::from_be_bytes(b[16..20].try_into().unwrap()),
            1,
            "first_chunk"
        );
        assert_eq!(
            u32::from_be_bytes(b[20..24].try_into().unwrap()),
            1,
            "每 chunk 1 个"
        );
        assert_eq!(
            u32::from_be_bytes(b[28..32].try_into().unwrap()),
            4,
            "从第 4 个 chunk 起"
        );
        assert_eq!(
            u32::from_be_bytes(b[32..36].try_into().unwrap()),
            2,
            "每 chunk 2 个"
        );
    }

    #[test]
    fn pcm_audio_uses_sowt_without_esds() {
        let mut m = Mp4Muxer::with_audio(
            video(),
            AudioTrack {
                channels: 2,
                sample_rate: 44100,
                timescale: 44100,
                codec: AudioCodec::Pcm {
                    bits: 16,
                    big_endian: false,
                },
                avg_bitrate: 0,
                priming: 0,
            },
        );
        m.write_video(&[1, 2, 3], true, 512).unwrap();
        // 一个样本 = 一个采样点 = 2ch × 2B
        for _ in 0..500 {
            m.write_audio(&[0u8; 4], 1).unwrap();
        }
        let f = m.finish();
        assert!(f.windows(4).any(|w| w == b"sowt"), "有 sowt sample entry");
        assert_eq!(
            f.windows(4).filter(|w| *w == b"esds").count(),
            1,
            "只有视频轨写 esds；PCM 不写"
        );

        let stsz = f.windows(4).position(|w| w == b"stsz").expect("有 stsz");
        // 命中点是 type 串：ver/flags(+4) default_size(+8) count(+12)
        let audio_stsz = f[stsz + 4..].windows(4).position(|w| w == b"stsz").unwrap() + stsz + 4;
        assert_eq!(
            u32::from_be_bytes(f[audio_stsz + 8..audio_stsz + 12].try_into().unwrap()),
            4,
            "PCM 的 default_sample_size = 声道数 × 字节数"
        );
        assert_eq!(
            u32::from_be_bytes(f[audio_stsz + 12..audio_stsz + 16].try_into().unwrap()),
            500
        );
    }

    #[test]
    fn stss_is_omitted_when_every_frame_is_a_key_frame() {
        let samples = vec![
            SampleRec {
                offset: 0,
                size: 1,
                duration: 512,
                key: true,
            },
            SampleRec {
                offset: 1,
                size: 1,
                duration: 512,
                key: true,
            },
        ];
        assert!(stss(&samples).is_none(), "全关键帧 → 不写 stss");

        let mixed = vec![
            SampleRec {
                offset: 0,
                size: 1,
                duration: 512,
                key: true,
            },
            SampleRec {
                offset: 1,
                size: 1,
                duration: 512,
                key: false,
            },
            SampleRec {
                offset: 2,
                size: 1,
                duration: 512,
                key: true,
            },
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
            .map(|i| SampleRec {
                offset: i,
                size: 1,
                duration: 512,
                key: i == 0,
            })
            .collect();
        let b = stts(&samples).finish();
        assert_eq!(
            u32::from_be_bytes(b[12..16].try_into().unwrap()),
            1,
            "只一项"
        );
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
