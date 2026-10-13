//! v2video 的转码核心：**解封装 → 解码 → 编码 → 封装**。
//!
//! 命令行（`main.rs`）只是它的壳；这里能被测试直接调用，也能被别的程序当库用。
//!
//! ## 现状
//!
//! - 输入：**ASF（`.wmv`）/ AVI（`.avi`）** 里的 **MP43** 视频轨 —— 容器按魔数自动认。
//!   别的容器 / codec 会明确报 `Unsupported`，不猜、不静默跳过。
//! - 输出：**两种容器**（`ConvertOptions::format`）
//!
//!   | | 视频 | 音频 |
//!   |---|---|---|
//!   | `OutputFormat::Asf`（`.wmv`） | MP43 **重编码** | WMA **直通**（载荷原样搬） |
//!   | `OutputFormat::Mp4`（`.mp4`） | **MPEG-4 Part 2** 重编码（`v2v-mp4v`） | WMA **解码成 PCM**（`sowt`） |
//!
//! ## `.mp4` 那条路为什么必须重编码
//!
//! 不是因为「想换个编码器」，而是**没得选**：MP4 / 3GP 容器不收 MP43
//! （`mp4`/`3gp`/`3g2` 三个 muxer 都报 `codec not currently supported in container`），
//! `mov` 收但是单向街 —— 写出来的东西 ffmpeg 自己读不回来。所以视频必须过一遍
//! MPEG-4 Part 2 编码器（见 `crates/mp4v`）。
//!
//! 音频那边不用写编码器：**PCM（`sowt`）** 塞进 `isom` 品牌的文件里 ffmpeg 读得好好的
//! （见 `crates/mp4` 的模块文档）。代价是文件变大（44.1 kHz 立体声约 176 KB/s）。
//!
//! ## `.mp4` 的校验为什么是「未校验」
//!
//! `verify` 的做法是「把编出来的码流解回来和源帧比 PSNR」。`.wmv` 那条路能做到，
//! 因为我们**自己就有 MP43 解码器**；`.mp4` 这条路做不到 —— 我们只有 `mp4v`
//! **编码器**，没有解码器。所以 `psnr_db` 是 `None`，报告里显示「未校验」。
//!
//! 这不是「没测」：编码器的正确性由 `crates/mp4v/tests/ffmpeg_roundtrip.rs` 钉着
//! （ffmpeg 解得开、PSNR 与 ffmpeg 自己的 mpeg4 编码器同档、`esds` 里的 VOL 完整）。
//! 只是那条判据需要 ffmpeg，不能进 CLI 的运行时。

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

pub use v2v_asf::{AsfDemuxer, AsfMuxer, StreamKind, VideoStreamInfo};
pub use v2v_avi::AviDemuxer;
pub use v2v_container::{AudioFormat, Sample};
pub use v2v_msmpeg4::{Decoder, Encoder, EncoderConfig, Frame};

/// MP43 的 FourCC（`.wmv` 输出用的编码）。
pub const MP43: u32 = u32::from_le_bytes(*b"MP43");

/// 输出容器。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// MP43 in ASF（`.wmv`）。
    Asf,
    /// MPEG-4 Part 2（`mp4v`）in MP4（`.mp4`）。
    Mp4,
}

impl OutputFormat {
    pub fn extension(self) -> &'static str {
        match self {
            OutputFormat::Asf => "wmv",
            OutputFormat::Mp4 => "mp4",
        }
    }
}

/// 转码参数。
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// 目标 qscale（1..=31）。越大越小越糊。
    ///
    /// ⚠️ **两种输出容器的含义不完全一样**：`.wmv` 走 MP43，量化后的 level 必须塞进
    /// 8 位 escape，编码器会自己往上抬；`.mp4` 走 MPEG-4 Part 2，ESC3 是 12 位、没有这个约束。
    pub qscale: u8,
    /// 输出容器。
    pub format: OutputFormat,
    /// 周期 I 帧间隔（帧数）。`0` = 只首帧是 I。
    ///
    /// ⚠️ 两种容器的**默认含义不同**：`.wmv` 的默认是 250（有小 GOP）；
    /// `.mp4` 的默认是 0（**全 I 帧**，文件大但每帧都能独立解）。
    /// 给 `.mp4` 传个非零值就会真的编 P 帧 —— 低运动素材能压到 1/10 左右。
    pub keyframe_interval: usize,
    /// 转完把输出**解码回来**与源帧比 PSNR（慢一倍，但能证明「转出来的东西真能放」）。
    ///
    /// 只对 `.wmv` 有效；`.mp4` 没有解码器可比，见模块文档。
    pub verify: bool,
    /// 最多转多少帧（`None` = 全部）。做局部转换/预览用 —— 也让真样本的
    /// 端到端验证能在几秒内跑完。
    pub max_frames: Option<usize>,
    /// **音频直通**（默认开）：把源的音频载荷**原样**搬进输出，不重编码。
    ///
    /// 只对 `.wmv` 有效（输入输出都是 WMA 才有意义）。`.mp4` 里 WMA 装不进去，
    /// 这一项被忽略、音频一律解成 PCM。
    pub copy_audio: bool,
    /// `.mp4` 的帧率分子（写进 VOL 的 `vop_time_increment_resolution`）。
    /// `None` = 从视频样本的时间戳推。
    pub fps: Option<u32>,
    /// `.mp4` 的音频轨怎么走。
    ///
    /// 只对 `.mp4` 有效 —— `.wmv` 那条路的音频是**直通**（ASF 载荷原样搬）。
    pub audio: Mp4Audio,
    /// `.mp4` 走 AAC 时的目标码率（bit/s）。
    ///
    /// 这是**目标**不是保证：编码器不做码率控制，实际码率在它附近浮动。
    pub aac_bitrate: u32,
}

/// `.mp4` 的音频轨编码方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mp4Audio {
    /// **AAC-LC 编码**（默认）。体积能压到 PCM 的 1/10 上下。
    Aac,
    /// **未压缩 PCM**（`sowt`）。文件大，但一定能放、而且零编解码延迟。
    Pcm,
}

impl Default for Mp4Audio {
    fn default() -> Self {
        // 默认走 AAC —— 真样本里 PCM 占了整个 `.mp4` 的 3/4，
        // 而 AAC 解码器在哪儿都有（浏览器、手机、ffmpeg）。
        Self::Aac
    }
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            qscale: 6,
            format: OutputFormat::Asf,
            keyframe_interval: 250,
            verify: true,
            max_frames: None,
            copy_audio: true,
            fps: None,
            audio: Mp4Audio::default(),
            aac_bitrate: 128_000,
        }
    }
}

/// 一次转换的结果。
#[derive(Debug, Clone)]
pub struct ConvertReport {
    pub frames: usize,
    /// 关键帧数（含首帧）。`.mp4` 目前是**全 I 帧**，所以这里等于 `frames`。
    pub key_frames: usize,
    /// 写进输出的音频样本数。
    ///
    /// ⚠️ 两种容器的单位不同：`.wmv` 是**载荷数**（一个 ASF 载荷可能含多帧）；
    /// `.mp4` 是 **PCM 采样点**数（一个采样点 = 一条 `stsz` 记录，见 `crates/mp4`）。
    pub audio_payloads: usize,
    /// 没搬的音频样本数（`copy_audio = false`）。
    pub skipped_audio: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
    /// `verify` 打开时的平均 PSNR（dB，YUV 三平面合计）。
    ///
    /// `.mp4` 恒为 `None` —— 我们只有 `mp4v` 编码器、没有解码器，没法自己校验，见模块文档。
    pub psnr_db: Option<f64>,
}

/// 转码错误。
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Asf(v2v_asf::Error),
    /// 视频编解码错误。**MP43 与 MPEG-4 Part 2 共用这一个** ——
    /// `v2v_mp4v::Error` 就是 `v2v_msmpeg4::Error`（见那个 crate 的 `lib.rs`）。
    Msmpeg4(v2v_msmpeg4::Error),
    Avi(v2v_avi::Error),
    Mp4(v2v_mp4::Error),
    Wma(v2v_wma::Error),
    Unsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO 错误：{e}"),
            Error::Asf(e) => write!(f, "容器错误：{e}"),
            Error::Msmpeg4(e) => write!(f, "编解码错误：{e}"),
            Error::Avi(e) => write!(f, "AVI 错误：{e}"),
            Error::Mp4(e) => write!(f, "MP4 封装错误：{e}"),
            Error::Wma(e) => write!(f, "WMA 解码错误：{e}"),
            Error::Unsupported(s) => write!(f, "不支持：{s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
impl From<v2v_asf::Error> for Error {
    fn from(e: v2v_asf::Error) -> Self {
        Error::Asf(e)
    }
}
impl From<v2v_msmpeg4::Error> for Error {
    fn from(e: v2v_msmpeg4::Error) -> Self {
        Error::Msmpeg4(e)
    }
}
impl From<v2v_avi::Error> for Error {
    fn from(e: v2v_avi::Error) -> Self {
        Error::Avi(e)
    }
}
impl From<v2v_mp4::Error> for Error {
    fn from(e: v2v_mp4::Error) -> Self {
        Error::Mp4(e)
    }
}
impl From<v2v_wma::Error> for Error {
    fn from(e: v2v_wma::Error) -> Self {
        Error::Wma(e)
    }
}

/// 「能吐视频样本的东西」—— ASF / AVI 解封装器都实现它。
///
/// 存在的理由：转码管线只关心「视频流是谁、多大、什么 codec」和「下一个样本」，
/// 不该为每种容器写一份管线。加容器 = 加一个实现，管线不动。
trait VideoSource {
    /// `(流号, 宽, 高, fourcc)`；没有视频轨就 `None`。
    fn video_stream(&self) -> Option<(u8, u32, u32, u32)>;
    /// `(流号, WAVEFORMATEX)`；没有音频轨就 `None`。
    fn audio_stream(&self) -> Option<(u8, AudioFormat)>;
    fn next(&mut self) -> Result<Option<Sample>, Error>;
}

struct AsfSource(v2v_asf::AsfDemuxer<BufReader<File>>);

impl VideoSource for AsfSource {
    fn video_stream(&self) -> Option<(u8, u32, u32, u32)> {
        let v = self
            .0
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Video)?;
        let vf = v.video.as_ref()?;
        Some((v.number, vf.width, vf.height, vf.fourcc))
    }
    fn audio_stream(&self) -> Option<(u8, AudioFormat)> {
        let a = self
            .0
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Audio)?;
        Some((a.number, a.audio.clone()?))
    }
    fn next(&mut self) -> Result<Option<Sample>, Error> {
        Ok(self.0.next_sample()?)
    }
}

struct AviSource(v2v_avi::AviDemuxer<BufReader<File>>);

impl VideoSource for AviSource {
    fn video_stream(&self) -> Option<(u8, u32, u32, u32)> {
        let v = self
            .0
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Video)?;
        let vf = v.video.as_ref()?;
        Some((v.number, vf.width, vf.height, vf.fourcc))
    }
    fn audio_stream(&self) -> Option<(u8, AudioFormat)> {
        let a = self
            .0
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Audio)?;
        Some((a.number, a.audio.clone()?))
    }
    fn next(&mut self) -> Result<Option<Sample>, Error> {
        Ok(self.0.next_sample()?)
    }
}

/// 按**魔数**选解封装器（不看扩展名 —— 扩展名会骗人，字节不会）。
fn open_source(path: &Path) -> Result<Box<dyn VideoSource>, Error> {
    use std::io::{Read, Seek, SeekFrom};

    let mut f = BufReader::new(File::open(path)?);
    let mut head = [0u8; 16];
    let n = f.read(&mut head)?;
    // ⚠️ 探测读掉了开头 16 字节，必须倒回去 —— 解封装器都假设自己从 0 开始读。
    f.seek(SeekFrom::Start(0))?;

    if v2v_avi::is_avi(&head[..n]) {
        return Ok(Box::new(AviSource(v2v_avi::AviDemuxer::new(f)?)));
    }
    if v2v_asf::is_asf(&head[..n]) {
        return Ok(Box::new(AsfSource(v2v_asf::AsfDemuxer::new(f)?)));
    }
    Err(Error::Unsupported(
        "认不出容器（目前支持 ASF/.wmv 与 AVI/.avi）".into(),
    ))
}

/// 把一个文件转成 `opts.format` 指定的容器。
///
/// 逐帧流式：读一帧 → 解码 → 编码 → 写进封装器（封装器内部缓冲，最后一次性落盘）。
pub fn convert_file(
    input: &Path,
    output: &Path,
    opts: &ConvertOptions,
) -> Result<ConvertReport, Error> {
    if !(1..=31).contains(&opts.qscale) {
        return Err(Error::Unsupported(format!(
            "qscale {} 不在 1..=31",
            opts.qscale
        )));
    }
    match opts.format {
        OutputFormat::Asf => convert_to_asf(input, output, opts),
        OutputFormat::Mp4 => convert_to_mp4(input, output, opts),
    }
}

/// `.wmv`（MP43 in ASF）那条路。
fn convert_to_asf(
    input: &Path,
    output: &Path,
    opts: &ConvertOptions,
) -> Result<ConvertReport, Error> {
    let bytes_in = std::fs::metadata(input)?.len();

    let mut demux = open_source(input)?;

    // ---- 找视频轨；只认 MP43 ----
    let (vnum, w, h, fourcc) = demux
        .video_stream()
        .ok_or_else(|| Error::Unsupported("文件里没有视频轨（或没有格式信息）".into()))?;
    if fourcc != MP43 {
        return Err(Error::Unsupported(format!(
            "输入视频是 {}，本工具目前只解 MP43",
            v2v_asf::fourcc_string(fourcc)
        )));
    }

    let mut dec = Decoder::new();
    let mut enc = Encoder::new(EncoderConfig {
        qscale: opts.qscale,
        keyframe_interval: opts.keyframe_interval,
        ..Default::default()
    });
    // 音频：直通（只支持 WMA，因为输出要写同样的 WAVEFORMATEX）
    let audio = if opts.copy_audio {
        match demux.audio_stream() {
            Some((_, af)) if af.format_tag == 0x0160 || af.format_tag == 0x0161 => Some(af),
            Some((_, af)) => {
                return Err(Error::Unsupported(format!(
                    "音频是 0x{:04X}，直通只支持 WMA（0x0160/0x0161）",
                    af.format_tag
                )))
            }
            None => None,
        }
    } else {
        None
    };
    let audio_num = demux.audio_stream().map(|(n, _)| n);
    let mut mux = AsfMuxer::with_audio(
        VideoStreamInfo {
            width: w,
            height: h,
            fourcc: MP43,
        },
        audio,
    );
    // 校验用的第二个解码器（与解码源帧的那个分开 —— 状态独立）。
    let mut vdec = if opts.verify {
        Some(Decoder::new())
    } else {
        None
    };

    let mut frames = 0usize;
    let mut key_frames = 0usize;
    let mut audio_payloads = 0usize;
    let mut skipped_audio = 0usize;
    let mut sq_err = 0f64;
    let mut n_px = 0u64;

    while let Some(s) = demux.next()? {
        // 音频：直通（原样写载荷，时间戳照搬）
        if Some(s.stream) == audio_num {
            if opts.copy_audio {
                let pts = u32::try_from(s.timestamp_ms.max(0)).unwrap_or(0);
                mux.write_audio(&s.data, pts)?;
                audio_payloads += 1;
            } else {
                skipped_audio += 1;
            }
            continue;
        }
        if s.stream != vnum {
            continue;
        }
        let pts = u32::try_from(s.timestamp_ms.max(0))
            .map_err(|_| Error::Unsupported(format!("时间戳 {} ms 超出 u32", s.timestamp_ms)))?;

        let frame = dec.decode(&s.data, w, h)?;
        let stream = enc.encode_frame(&frame)?;
        mux.write_video(&stream, enc.last_key_frame, pts)?;

        if enc.last_key_frame {
            key_frames += 1;
        }
        if let Some(vd) = vdec.as_mut() {
            let back = vd.decode(&stream, w, h)?;
            for (a, b) in [
                (&frame.y, &back.y),
                (&frame.u, &back.u),
                (&frame.v, &back.v),
            ] {
                for (&x, &y) in a.iter().zip(b.iter()) {
                    let d = x as f64 - y as f64;
                    sq_err += d * d;
                }
                n_px += a.len() as u64;
            }
        }
        frames += 1;
        if let Some(max) = opts.max_frames {
            if frames >= max {
                break;
            }
        }
    }

    let wmv = mux.finish();
    write_out(output, &wmv)?;

    let psnr_db = if opts.verify && n_px > 0 {
        Some(psnr(sq_err, n_px as u64))
    } else {
        None
    };

    Ok(ConvertReport {
        frames,
        key_frames,
        audio_payloads,
        skipped_audio,
        bytes_in,
        bytes_out: wmv.len() as u64,
        psnr_db,
    })
}

// ==================== .mp4（MPEG-4 Part 2）那条路 ====================

/// 音频怎么进 `.mp4`。
///
/// 必须在读到第一个样本**之前**定下来 —— MP4 的 `stsd`（音频 sample entry）是
/// 封装器构造时就写死的，中途换不了。
struct AudioPlan {
    /// 源里的流号。
    stream: u8,
    /// 一个 PCM 采样点占多少字节（`声道数 × 2`）。
    ///
    /// ⚠️ MP4 的 `sowt` 轨里**一个采样点就是一条样本记录**（`stsz` 靠
    /// `default_sample_size` 压缩，`stts` 是一条 `(N, 1)`），所以写的时候要一个点一次。
    /// 见 `crates/mp4` 的模块文档。
    point: usize,
    /// 开头要丢掉的样本数（WMA 的编码器延迟）。
    skip: usize,
    /// `None` = 源本来就是 PCM，直接搬；`Some` = 要过 WMA 解码器。
    decoder: Option<v2v_wma::WmaDecoder>,
}

/// 决定音频轨怎么走。返回 `None` = 输出不带音频。
fn plan_audio(demux: &dyn VideoSource, enabled: bool) -> Result<Option<AudioPlan>, Error> {
    if !enabled {
        return Ok(None);
    }
    let Some((num, af)) = demux.audio_stream() else {
        return Ok(None);
    };
    let point = af.channels as usize * 2;
    match af.format_tag {
        // WMA v1 / v2：解成 16 位 PCM。
        0x0160 | 0x0161 => {
            let version = if af.format_tag == 0x0160 { 1 } else { 2 };
            let params = v2v_wma::WmaParams::new(
                version,
                af.channels as usize,
                af.sample_rate,
                af.avg_bytes_per_sec * 8,
                af.block_align as usize,
                &af.extra,
            )?;
            let decoder = v2v_wma::WmaDecoder::new(params.clone())?;
            // ⚠️ **编码器延迟**：WMA 编码器开头灌了两帧的静音，ffmpeg 的解码器
            // `skip_samples = frame_len * 2` 把它丢掉。不跟着丢，音频会整体晚
            // 两帧（44.1 kHz、帧长 2048 → 约 93 ms），与视频明显对不上。
            Ok(Some(AudioPlan {
                stream: num,
                point,
                skip: params.frame_len * 2,
                decoder: Some(decoder),
            }))
        }
        // 本来就是 16 位 PCM：不用解，直接搬。
        0x0001 if af.bits_per_sample == 16 => Ok(Some(AudioPlan {
            stream: num,
            point,
            skip: 0,
            decoder: None,
        })),
        other => Err(Error::Unsupported(format!(
            "音频是 0x{other:04X}，`.mp4` 这条路只支持 WMA（0x0160/0x0161）与 16 位 PCM（0x0001）\
             —— 用 --no-audio 只转视频"
        ))),
    }
}

/// 把 WMA 解出来的一帧（每声道一串 f32）交织成小端 16 位 PCM。
///
/// ⚠️ 解码器给的是**已经落在 `[-32768, 32767]` 的浮点**，不是 `[-1, 1)` ——
/// 直接截断即可，**不要**再乘 32768（乘了会整体溢出成噪声）。
/// 口径与 `crates/wma/tests/pcm.rs` 的对账测试一致。
fn wma_frame_to_pcm(frames: Vec<Vec<Vec<f32>>>) -> Vec<u8> {
    let mut out = Vec::new();
    for f in frames {
        let n = f.first().map_or(0, |p| p.len());
        out.reserve(n * f.len() * 2);
        for i in 0..n {
            for plane in &f {
                let v = plane[i].clamp(-32768.0, 32767.0) as i16;
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    out
}

/// `.mp4`（MPEG-4 Part 2 in MP4）那条路。
///
/// 与 `.wmv` 那条的三个实质差别：
///
/// 1. **视频必须重编码**成 MPEG-4 Part 2（MP43 进不了 MP4，见模块文档），而且目前是**全 I 帧**；
/// 2. **音频解成 PCM**（WMA 也进不了 MP4）；
/// 3. **没有校验** —— 我们只有 `mp4v` 编码器、没有解码器。
///
/// ## 帧率与时间基
///
/// 帧率必须在**建编码器之前**知道（要写进 VOL 的 `vop_time_increment_resolution`），
/// 而解封装器是单向流式的 —— 所以先预读开头几个样本定帧率，再把它们喂回去处理。
///
/// 视频轨的时间基取 **1000（毫秒）**，`stts` 里写**真实的帧间隔**。这样 29.97 / 23.976
/// 这类非整数帧率也不用去找最小公倍数，播放节奏就是源文件的节奏。
/// VOL 里声明的整数帧率只影响 VOP 头的 `time_mod` 记账，不影响播放。
fn convert_to_mp4(
    input: &Path,
    output: &Path,
    opts: &ConvertOptions,
) -> Result<ConvertReport, Error> {
    let bytes_in = std::fs::metadata(input)?.len();
    let mut demux = open_source(input)?;

    let (vnum, w, h, fourcc) = demux
        .video_stream()
        .ok_or_else(|| Error::Unsupported("文件里没有视频轨（或没有格式信息）".into()))?;
    if fourcc != MP43 {
        return Err(Error::Unsupported(format!(
            "输入视频是 {}，本工具目前只解 MP43",
            v2v_asf::fourcc_string(fourcc)
        )));
    }

    // 源里那条音频轨的格式（有没有音频、采样率多少）—— 封装器的 `stsd` 要用。
    let src_audio = demux.audio_stream().map(|(_, af)| af);
    let mut audio = plan_audio(demux.as_ref(), opts.copy_audio)?;
    if src_audio.is_none() {
        audio = None; // 没有音频轨就别留下一个「以为有」的 plan
    }
    let audio_num = audio.as_ref().map(|a| a.stream);
    // `--no-audio` 时源里的音频样本要计数，报告里说清楚跳过了多少。
    let skipped_num = if audio_num.is_none() {
        demux.audio_stream().map(|(n, _)| n)
    } else {
        None
    };

    // ---- 预读：攒够几个视频样本就能定帧率 ----
    //
    // 队列里存的是**原样**的样本，处理时按同样的顺序吐出来（用 VecDeque 的
    // `pop_front` —— 用 `Vec::pop` 会倒着放，那是个很难看出来的坑）。
    let mut queue: VecDeque<Sample> = VecDeque::new();
    let mut vts: Vec<i64> = Vec::new();
    while let Some(s) = demux.next()? {
        if s.stream == vnum {
            vts.push(s.timestamp_ms);
        }
        let enough = vts.len() >= FPS_PROBE_SAMPLES;
        queue.push_back(s);
        if enough {
            break;
        }
    }
    let fps = opts
        .fps
        .or_else(|| fps_from_timestamps(&vts))
        .unwrap_or(DEFAULT_FPS)
        .clamp(1, 120);

    // ---- 视频编码器：MPEG-4 Part 2，全 I 帧 ----
    let mut enc = v2v_mp4v::Encoder::new(v2v_mp4v::EncoderConfig {
        width: w,
        height: h,
        fps_num: fps,
        qscale: opts.qscale,
        keyframe_interval: opts.keyframe_interval,
        ..Default::default()
    })?;

    // ---- 封装器 ----
    //
    // ⚠️ `decoder_specific_info()` 必须在 `Mp4Muxer::new` 时给出去，之后改不了；
    // 它是「VOS + 视觉对象 + VOL + 用户数据」整段，`ffprobe` 靠它报 profile / level。
    const VTS: u32 = 1000;
    let vtrack = v2v_mp4::VideoTrack {
        width: w as u16,
        height: h as u16,
        timescale: VTS,
        decoder_specific_info: enc.decoder_specific_info(),
        avg_bitrate: 0,
    };
    // ---- 音频编码器（只在 `.mp4` + `Mp4Audio::Aac` 时建）----
    //
    // ⚠️ AAC 的**每一帧必须是 1024 个采样点**（长窗），而 WMA 解码器吐出来的
    // 是一段一段不等长的 PCM。所以中间要有个「攒够 1024 再编」的缓冲 ——
    // 见下面的 `aac_pending`。
    let mut aac = match (&audio, &src_audio, opts.audio) {
        (Some(_), Some(af), Mp4Audio::Aac) => {
            Some(v2v_aac::Encoder::new(v2v_aac::EncoderConfig {
                channels: af.channels as u8,
                sample_rate: af.sample_rate,
                bitrate: opts.aac_bitrate,
                kb_window: false,
            })?)
        }
        _ => None,
    };

    let mut mux = match (src_audio.as_ref(), aac.as_ref()) {
        (Some(af), Some(a)) => {
            v2v_mp4::Mp4Muxer::with_audio(
                vtrack,
                v2v_mp4::AudioTrack {
                    channels: af.channels,
                    sample_rate: af.sample_rate,
                    // AAC 的时间基取采样率：一帧 1024 个样本，`stts` 里写 1024。
                    // 这样「样本时长 × 时间基」正好落在秒上，不用管帧率。
                    timescale: af.sample_rate,
                    avg_bitrate: opts.aac_bitrate,
                    codec: v2v_mp4::AudioCodec::Es {
                        object_type: 0x40, // MPEG-4 Audio
                        decoder_specific_info: a.audio_specific_config().to_bytes(),
                    },
                    // **0** —— 我们的编码器**没有引入可测量的延迟**（实测出来的）。
                    //
                    // ⚠️ 这个 `0` 来之不易：我在这上面**绕了好几轮**，
                    // 一度以为是 2048、又改成 4096，两次都是**量错了**。
                    //
                    // 错在哪：我用「逐样本 MSE 最小」去找对齐。那个方法在这个素材上
                    // **不可靠** —— 它的最小值（MSE 开方 ≈ 3203）比源信号自身的
                    // RMS（≈ 2429）还大，也就是说那个「最优」根本不是对齐，
                    // 而是**噪声里挑出来的一个假峰**（归一化互相关只有 0.28）。
                    // 拿假峰当延迟，怎么调 `elst` 都只会更差。
                    //
                    // 换成**能量包络对齐**（每 1024 个样本算一个 RMS，
                    // 再对两条包络求归一化互相关）之后真值立刻现形：
                    // 相关 **0.98**、偏移 **0** —— 干净、可判读。
                    //
                    // **教训：对齐搜索不能用逐样本 MSE。**
                    // 它的能量面太平，随便一个偏移都能凑出「更小」的值；
                    // 要先降到「包络」这个更粗的尺度上，让真对齐**显著**地赢。
                    priming: 0,
                },
            )
        }
        (Some(af), None) => v2v_mp4::Mp4Muxer::with_audio(
            vtrack,
            v2v_mp4::AudioTrack {
                channels: af.channels,
                sample_rate: af.sample_rate,
                // `sowt` 的时间基就是采样率（一个采样点 = 一个 tick）。
                timescale: af.sample_rate,
                avg_bitrate: af.sample_rate * af.channels as u32 * 16,
                codec: v2v_mp4::AudioCodec::Pcm {
                    bits: 16,
                    big_endian: false,
                },
                // PCM 没有编码器延迟（样本就是裸采样）。
                priming: 0,
            },
        ),
        _ => v2v_mp4::Mp4Muxer::new(vtrack),
    };

    let mut dec = Decoder::new();
    let mut frames = 0usize;
    let mut key_frames = 0usize;
    let mut audio_points = 0usize;
    let mut skipped_audio = 0usize;
    // WMA 解码器与要跳的延迟样本数（`plan_audio` 里算好的）—— 拿出来单独持有，
    // 因为循环里要 `&mut`。
    let (mut wma_dec, mut skip_left) = match audio.as_mut() {
        Some(a) => (a.decoder.take(), a.skip),
        None => (None, 0),
    };
    // AAC 的待编样本缓冲（每声道一条，f32）。攒够 1024 个就编一帧。
    let mut aac_pending: Vec<Vec<f32>> = Vec::new();
    // 上一帧的视频时间戳（毫秒）—— `stts` 写的是**真实间隔**。
    let mut prev_vts: Option<i64> = None;
    let mut last_dur = (VTS as f64 / fps as f64).round().max(1.0) as u32;
    let mut stop = false;

    loop {
        let s = match queue.pop_front() {
            Some(s) => Some(s),
            None if !stop => demux.next()?,
            None => None,
        };
        let Some(s) = s else { break };

        // ---- 音频 ----
        if Some(s.stream) == audio_num {
            let plan = audio.as_ref().expect("audio_num 有值就一定有 plan");
            let mut pcm = match wma_dec.as_mut() {
                Some(d) => wma_frame_to_pcm(d.decode_superframe(&s.data)?),
                None => s.data.clone(),
            };
            if skip_left > 0 {
                let drop = (skip_left * plan.point).min(pcm.len());
                pcm.drain(..drop);
                skip_left -= drop / plan.point;
            }
            if let Some(enc) = aac.as_mut() {
                // ---- AAC：攒够 1024 个采样点再编一帧 ----
                //
                // 源 PCM 是「交织的小端 16 位」，AAC 编码器要的是
                // 「每声道一条、f32、范围 ±32768」——所以要先拆开、归一。
                push_aac_pcm(
                    &mut aac_pending,
                    &pcm,
                    plan.point,
                    src_audio.as_ref().expect("有 plan 就一定有源格式").channels as usize,
                );
                drain_aac_frames(
                    enc,
                    &mut aac_pending,
                    &mut mux,
                    &mut audio_points,
                    plan.point,
                )?;
                continue;
            }
            for c in pcm.chunks_exact(plan.point) {
                mux.write_audio(c, 1)?;
                audio_points += 1;
            }
            continue;
        }
        if Some(s.stream) == skipped_num {
            skipped_audio += 1;
            continue;
        }
        if s.stream != vnum {
            continue;
        }

        // ---- 视频 ----
        if let Some(p) = prev_vts {
            let d = s.timestamp_ms - p;
            if d > 0 {
                last_dur = d.min(u32::MAX as i64) as u32;
            }
        }
        prev_vts = Some(s.timestamp_ms);

        let frame = dec.decode(&s.data, w, h)?;
        // ⚠️ **`-g 0` 在 `.mp4` 这条路上就是「全 I 帧」** ——
        // 这条路的默认值是 0（每个 I 帧都独立可解，见 `ConvertOptions` 的说明），
        // 所以不能直接调 `encode_frame`（在 `mp4v` 里 `0` 的含义是「无 GOP」，会编 P 帧、
        // 而且第一个 P 帧会因为没有参考帧而报错）。
        let stream = if opts.keyframe_interval == 0 {
            enc.encode_i_frame(&frame)?
        } else {
            enc.encode_frame(&frame)?
        };
        // 关键帧位来自编码器 —— 全 I 帧时 `stss` 会被整个省掉。
        mux.write_video(&stream, enc.last_key_frame, last_dur)?;
        if enc.last_key_frame {
            key_frames += 1;
        }
        frames += 1;
        if let Some(max) = opts.max_frames {
            if frames >= max {
                stop = true;
                // 已经预读进队列的音频还要写完，所以不能直接 break。
            }
        }
    }

    // ---- 收尾：把 AAC 的尾巴（不足 1024 个采样点）补零编出去 ----
    if let Some(enc) = aac.as_mut() {
        flush_aac(enc, &mut aac_pending, &mut mux, &mut audio_points)?;
    }

    let mp4 = mux.finish();
    write_out(output, &mp4)?;

    Ok(ConvertReport {
        frames,
        key_frames,
        audio_payloads: audio_points,
        skipped_audio,
        bytes_in,
        bytes_out: mp4.len() as u64,
        // 没有 mp4v 解码器可比，见模块文档。
        psnr_db: None,
    })
}

/// 把一段「交织的小端 16 位 PCM」拆进 AAC 的待编缓冲。
///
/// ⚠️ 两个口径都要转：
///
/// 1. **交织 → 分声道**：源是 `L R L R ...`，AAC 编码器要 `[L...], [R...]`；
/// 2. **int16 → f32**：`crates/wma` 解出来的是 int16 量级，而 `v2v-aac` 要的是
///    同量级的 f32（**不归一化到 ±1**，因为 AAC 的 sf 定标本来就吸收整帧幅度）。
///
/// 尾部不足一个采样点的字节直接丢掉 —— `plan.point` 是「一个采样点的字节数」，
/// 只可能是 2 或 4，不会有半个点的残渣。
fn push_aac_pcm(pending: &mut Vec<Vec<f32>>, pcm: &[u8], point: usize, channels: usize) {
    if pending.is_empty() {
        pending.resize(channels, Vec::new());
    }
    for c in pcm.chunks_exact(point) {
        for (ch, p) in pending.iter_mut().enumerate() {
            let o = ch * 2;
            let v = i16::from_le_bytes([c[o], c[o + 1]]);
            p.push(f32::from(v));
        }
    }
}

/// 从待编缓冲里尽量多地编出 AAC 帧（每帧 1024 个采样点）并写进封装器。
///
/// ⚠️ **不足 1024 的部分要留着**，不能提前编 —— 提前编会补零，
/// 而补进去的零会在输出里变成一小段可听的静音（每 1024 个点补一次就够听出来）。
/// 剩下的尾巴由 `flush_aac` 在收尾时补零编出去。
fn drain_aac_frames(
    enc: &mut v2v_aac::Encoder,
    pending: &mut Vec<Vec<f32>>,
    mux: &mut v2v_mp4::Mp4Muxer,
    audio_samples: &mut usize,
    _point: usize,
) -> Result<(), Error> {
    loop {
        let have = pending.first().map_or(0, |p| p.len());
        if have < AAC_FRAME_SAMPLES {
            return Ok(());
        }
        let planes: Vec<Vec<f32>> = pending
            .iter_mut()
            .map(|p| p.drain(..AAC_FRAME_SAMPLES).collect())
            .collect();
        let frame = enc.encode_frame(&v2v_aac::Frame { planes })?;
        // AAC 一帧就是 1024 个采样点，时间基是采样率 —— 所以时长写 1024。
        mux.write_audio(&frame, AAC_FRAME_SAMPLES as u32)?;
        // 计数用**输出侧的采样点数**（与 PCM 那条路同单位，报告里好对比）。
        *audio_samples += AAC_FRAME_SAMPLES;
    }
}

/// 收尾：把不足一帧的尾巴补零编出去。
///
/// 不丢尾巴的话，输出的音频会比源短最多 1023 个采样点（约 23 ms）——
/// 听感上就是最后那一点点被切掉了。
fn flush_aac(
    enc: &mut v2v_aac::Encoder,
    pending: &mut Vec<Vec<f32>>,
    mux: &mut v2v_mp4::Mp4Muxer,
    audio_samples: &mut usize,
) -> Result<(), Error> {
    let have = pending.first().map_or(0, |p| p.len());
    if have == 0 {
        return Ok(());
    }
    // 补零到 1024（残留的尾巴本来就没有真实样本，补零是唯一合法的编法）。
    let planes: Vec<Vec<f32>> = pending
        .iter_mut()
        .map(|p| {
            let mut v = std::mem::take(p);
            v.resize(AAC_FRAME_SAMPLES, 0.0);
            v
        })
        .collect();
    let frame = enc.encode_frame(&v2v_aac::Frame { planes })?;
    mux.write_audio(&frame, AAC_FRAME_SAMPLES as u32)?;
    *audio_samples += AAC_FRAME_SAMPLES;
    Ok(())
}

/// AAC 长窗一帧的采样点数。
const AAC_FRAME_SAMPLES: usize = 1024;

/// 默认帧率 —— 推不出来时用（WMV 绝大多数是 25 或 30）。
const DEFAULT_FPS: u32 = 25;

/// 定帧率前预读多少个视频样本。
///
/// 只取前两个不够 —— ASF 里可能有**两条视频轨**（`wma1.asf` 就是），
/// 首两个样本的间隔可能落在两条轨的边界上，或者干脆是 0。
const FPS_PROBE_SAMPLES: usize = 8;

/// 从视频样本的时间戳推帧率。
///
/// 取**间隔的中位数**而不是第一个间隔 —— 时间戳偶尔会有一个不规整的
/// （ASF 的分组边界、重复帧），只看第一个会被带跑。
///
/// ⚠️ 这里推的是**整数**帧率（要写进 VOL 的 `vop_time_increment_resolution`，
/// 那个字段是整数）。12.5 fps 这种会被推成 12 或 13 —— 但**播放节奏不受影响**：
/// 真正决定节奏的是 `stts` 里逐帧写的**真实毫秒间隔**，这里只影响
/// `ffprobe` 报的 `r_frame_rate` 那个数。非整数帧率想声明准，得改成
/// 「时间基 = 1001 × 帧率」那一套，目前没做。
fn fps_from_timestamps(ts: &[i64]) -> Option<u32> {
    let mut ds: Vec<i64> = ts
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 0)
        .collect();
    if ds.is_empty() {
        return None;
    }
    ds.sort_unstable();
    let d = ds[ds.len() / 2];
    let fps = (1000.0 / d as f64).round() as i64;
    Some(fps.clamp(1, 120) as u32)
}

fn psnr(sq_err: f64, n: u64) -> f64 {
    let mse = sq_err / n as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

fn write_out(path: &Path, data: &[u8]) -> Result<(), Error> {
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(data)?;
    f.flush()?;
    Ok(())
}

/// 批量：逐个转，返回每个文件的 `(输入, 结果)`。**一个失败不影响其他**。
pub fn convert_batch(
    inputs: &[PathBuf],
    out_dir: &Path,
    opts: &ConvertOptions,
) -> Vec<(PathBuf, Result<(PathBuf, ConvertReport), Error>)> {
    inputs
        .iter()
        .map(|input| {
            let out = output_path_for(input, out_dir, opts.format);
            let r = convert_file(input, &out, opts).map(|rep| (out, rep));
            (input.clone(), r)
        })
        .collect()
}

/// 输出路径：`<out_dir>/<输入文件名去扩展>.<容器扩展名>`；与输入同路径时插一个 `.conv`。
pub fn output_path_for(input: &Path, out_dir: &Path, format: OutputFormat) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "out".into());
    let ext = format.extension();
    let out = out_dir.join(format!("{stem}.{ext}"));
    if out == input {
        out_dir.join(format!("{stem}.conv.{ext}"))
    } else {
        out
    }
}
