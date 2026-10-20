//! v2video 的转码核心：**解封装 → 解码 → 编码 → 封装**。
//!
//! 命令行（`main.rs`）只是它的壳；这里能被测试直接调用，也能被别的程序当库用。
//!
//! ## 现状
//!
//! - 输入：**ASF（`.wmv`）/ AVI（`.avi`）** 里的 **MP42 / MP43 / WMV1 / WMV2** 视频轨 —— 容器按魔数自动认。
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
pub use v2v_msmpeg4::{Decoder, Encoder, EncoderConfig, Frame, Version};

/// 认 FourCC → 版本，**只放逐像素对账过的版本进来**。
///
/// ⚠️ 解码器内部已经认五种版本，但经过**逐帧对 ffmpeg 逐像素 0 差**验证的只有：
///
/// | 版本 | 状态 |
/// |---|---|
/// | `MP43`（V3） | ✅ 多份真样本逐帧 0 差 |
/// | `WMV1` | ✅ 四份真样本逐帧 0 差 |
/// | `WMV2` | ✅ 三份真样本逐帧 0 差（含 `mspel` / ABT / 环路滤波那些分支） |
/// | `MP42`（V2） | ✅ 自造样本逐帧 0 差（`crates/msmpeg4/tests/v2.rs`，扫了 q=2/6/10/15 四档 —— 低 qscale 分不出 DC 缩放表，必须带高档） |
/// | `MP41`（V1） | ❌ 代码分支在，但**没有可造的流**（ffmpeg 只有 `msmpeg4v2`/`msmpeg4v3` 编码器，没有 v1），拿不到对账样本 |
///
/// `WMV3` 不走这里 —— 它不是 MS-MPEG-4 的版本，是**另一套骨架**（`v2v-vc1`），
/// 见 [`open_video_decoder`]。
///
/// 宁可明确拒绝，也不能静默产出错误画面 —— 那比不支持糟得多。
fn verified_version(fourcc: u32) -> Result<Version, Error> {
    match Version::from_fourcc(fourcc) {
        Some(v @ (Version::V2 | Version::V3 | Version::Wmv1 | Version::Wmv2)) => Ok(v),
        _ => Err(Error::Unsupported(format!(
            "输入视频是 {}，本工具目前只解 MP42 / MP43 / WMV1 / WMV2 / WMV3（MP41 没有可对账的样本，暂不支持）",
            v2v_asf::fourcc_string(fourcc)
        ))),
    }
}

/// 按版本建解码器。
///
/// ⚠️ **WMV2 要 `extradata`** —— 它的能力位（`mspel` / `abt` / `j_type` …）不在位流里，
/// 只在容器的编解码器私有数据里。别的版本只用 FourCC 就够。
fn make_decoder(version: Version, extra: &[u8]) -> Result<Decoder, Error> {
    Ok(match version {
        Version::Wmv2 => Decoder::with_wmv2(extra)?,
        _ => Decoder::with_version(version),
    })
}

/// MP43 的 FourCC（`.wmv` 输出用的编码）。
pub const MP43: u32 = u32::from_le_bytes(*b"MP43");

/// WMV3（VC-1）的 FourCC —— **只做输入**，不做出编码。
///
/// 不是因为不想：ffmpeg 里根本没有 VC-1 编码器，本仓也没有可对账的 oracle。
pub const WMV3: u32 = u32::from_le_bytes(*b"WMV3");

/// 输入视频解码器。
///
/// 存在的理由：VC-1 与 MS-MPEG-4 **不是同一套骨架**（帧头、宏块层、变换、MC 全不同），
/// 没法共用一个实现；但**下游只认一种形状** —— 所以这里只做「出口归一」：
/// 两种解码器都吐 `v2v_msmpeg4::Frame`（已裁到显示尺寸的 YUV420P），
/// 编码器与封装器一行都不用改。
///
/// ⚠️ **不要**为了「省一次拷贝」让下游认 `v2v_vc1::Frame` —— 那会把 VC-1 的类型
/// 漏进 MP43 / mp4v 的编码路径，等于让编码器知道输入是什么编解码器。
pub enum VideoDecoder {
    /// MP43 / WMV1 / WMV2。
    Msmpeg4(Decoder),
    /// VC-1（WMV3）。
    Vc1(v2v_vc1::Decoder),
}

impl VideoDecoder {
    /// 解一帧。`w` / `h` 是**显示**尺寸（只有 MS-MPEG-4 那一侧用得上 ——
    /// VC-1 的解码器在构造时就把尺寸记下来了）。
    pub fn decode(&mut self, data: &[u8], w: u32, h: u32) -> Result<Frame, Error> {
        match self {
            Self::Msmpeg4(d) => Ok(d.decode(data, w, h)?),
            Self::Vc1(d) => {
                d.decode_frame(data)?;
                let f = d.frame();
                Ok(Frame {
                    width: f.width,
                    height: f.height,
                    y: f.y,
                    u: f.u,
                    v: f.v,
                })
            }
        }
    }
}

/// 按 FourCC 建解码器。`extra` 是容器的编解码器私有数据，`w` / `h` 是显示尺寸。
///
/// | FourCC | 走哪条 |
/// |---|---|
/// | `MP43` / `WMV1` / `WMV2` | `v2v-msmpeg4`（`WMV2` 还要 `extra` 里的能力位） |
/// | `WMV3` | `v2v-vc1`（`extra` 是 `STRUCT_C` 序列头） |
///
/// 别的一律明确报 `Unsupported`。
fn open_video_decoder(
    fourcc: u32,
    extra: &[u8],
    w: u32,
    h: u32,
) -> Result<VideoDecoder, Error> {
    if fourcc == WMV3 {
        let seq = v2v_vc1::SeqHeader::parse(extra)
            .map_err(|e| Error::Unsupported(format!("VC-1 序列头解析失败：{e}")))?;
        let dec = v2v_vc1::Decoder::new(seq, w, h)
            .map_err(|e| Error::Unsupported(format!("VC-1 解码器建不起来：{e}")))?;
        return Ok(VideoDecoder::Vc1(dec));
    }
    let version = verified_version(fourcc)?;
    Ok(VideoDecoder::Msmpeg4(make_decoder(version, extra)?))
}

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
    /// 双声道 AAC 允不允许用 **M/S 立体声**（默认开）。
    ///
    /// 逐帧决策（两个声道相关时才用）。关掉它只为对比测量 —— 正常都该开着。
    pub ms_stereo: bool,
    /// **编码时用几个线程**（默认 **1** = 不并行）。
    ///
    /// `> 1` 时两条输出路都走 **GOP 并行编码**：视频按 GOP 切块、各起一个编码器。
    /// `.wmv` 的产物仍与 1 线程**逐字节相同**；`.mp4` 不相同（GOP 边界重置运动估计
    /// 与码控，差几个字节到几十 KB），质量基本不变。
    ///
    /// **并行解码**只对 `.mp4` + **WMV3 输入**开：每条 GOP 一条全新解码器、从 I 帧独立解
    /// （前提「从 I 帧起解 == 整条链」由 `v2v-vc1` 的 `gop_start` 测试钉着）。
    /// MP43 也验证过等价（`v2v-msmpeg4` 的 `gop_start`），但**开着更慢**（解码只有 VC-1 的
    /// 1/3 工作量，串行解码本来就与编码 worker 重叠），所以没开；WMV1/WMV2 的跨帧状态
    /// （WMV2 的 `IntraX8.prediction_table`）会让新解码器解出不同像素，更不开。
    /// 并行度的上限是 **GOP 数** —— 短视频或默认大 GOP 下 GOP 数 < 线程数，加速就封顶；
    /// 想要更多并行就把 `-g` 调小或喂长视频。AAC 仍按声道并行（双声道 → 2；封顶到声道数）。
    /// 默认 1：CLI 不替用户猜「这台机器现在还有多少算力」—— 要快就显式给 `--threads 2`。
    pub threads: usize,
    /// `.mp4` 的**视频**目标码率（bit/s）。`None` = 按固定的 `qscale` 编（默认）。
    ///
    /// 开着的时候 `qscale` 只当**起步值**，编码器逐帧调它去够目标码率。
    /// 只对 `.mp4` 有效 —— `.wmv` 那条路（MP43）还没有码率控制。
    pub video_bitrate: Option<u32>,
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
            ms_stereo: true,
            threads: 1,
            video_bitrate: None,
        }
    }
}

/// 一次转换的结果。
#[derive(Debug, Clone)]
pub struct ConvertReport {
    pub frames: usize,
    /// 关键帧数（含首帧）。`.wmv` 恒等于 `frames`（MP43 那条路全 I 帧）；
    /// `.mp4` 是编码器**真正标了**关键帧的帧数（`-g 0` 时才是全 I 帧）。
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
    /// `.mp4` 视频轨的**实际**平均码率（kbps）。开了 `--video-bitrate` 才有。
    ///
    /// 码率控制唯一值得看的数就是它 —— 落在目标附近才算真的生效了。
    pub video_kbps: Option<f64>,
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

/// 视频轨的入口信息。
///
/// `extra`（编解码器私有数据）**不是可有可无的**：WMV2 的**能力位**
/// （`mspel` / `abt` / `j_type` …）就在里面 —— 少了它连帧头都读不出来。
struct VideoTrack {
    number: u8,
    width: u32,
    height: u32,
    fourcc: u32,
    extra: Vec<u8>,
}

/// 「能吐视频样本的东西」—— ASF / AVI 解封装器都实现它。
///
/// 存在的理由：转码管线只关心「视频流是谁、多大、什么 codec」和「下一个样本」，
/// 不该为每种容器写一份管线。加容器 = 加一个实现，管线不动。
trait VideoSource {
    /// 视频轨；没有视频轨就 `None`。
    fn video_stream(&self) -> Option<VideoTrack>;
    /// `(流号, WAVEFORMATEX)`；没有音频轨就 `None`。
    fn audio_stream(&self) -> Option<(u8, AudioFormat)>;
    fn next(&mut self) -> Result<Option<Sample>, Error>;
}

struct AsfSource(v2v_asf::AsfDemuxer<BufReader<File>>);

impl VideoSource for AsfSource {
    fn video_stream(&self) -> Option<VideoTrack> {
        let v = self
            .0
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Video)?;
        let vf = v.video.as_ref()?;
        Some(VideoTrack {
            number: v.number,
            width: vf.width,
            height: vf.height,
            fourcc: vf.fourcc,
            extra: vf.extra.clone(),
        })
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
    fn video_stream(&self) -> Option<VideoTrack> {
        let v = self
            .0
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Video)?;
        let vf = v.video.as_ref()?;
        Some(VideoTrack {
            number: v.number,
            width: vf.width,
            height: vf.height,
            fourcc: vf.fourcc,
            extra: vf.extra.clone(),
        })
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

/// 并行那条路**最多攒多少字节的帧**在内存里（`--threads > 1` 时）。
///
/// 通道按 GOP 数限量是不够的：720p 一个 GOP 是 12 帧 × 1.38 MB ≈ 17 MB，
/// `2 × 10` 个 GOP 就是 330 MB —— 桌面（16 GB）无所谓，**Android 上这个量级会被 LMK 杀**。
/// 所以再加一道字节预算：排队 + 正在编的一起算，超了就压到预算之内（至少留 2 个 GOP，
/// 否则 worker 会饿死）。
const INFLIGHT_BUDGET_BYTES: usize = 128 << 20;

/// `.wmv`（MP43 in ASF）那条路。
fn convert_to_asf(
    input: &Path,
    output: &Path,
    opts: &ConvertOptions,
) -> Result<ConvertReport, Error> {
    let bytes_in = std::fs::metadata(input)?.len();

    let mut demux = open_source(input)?;

    // ---- 找视频轨；只认 MP43 ----
    let track = demux
        .video_stream()
        .ok_or_else(|| Error::Unsupported("文件里没有视频轨（或没有格式信息）".into()))?;
    let (vnum, w, h) = (track.number, track.width, track.height);

    let mut dec = open_video_decoder(track.fourcc, &track.extra, w, h)?;
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

    // ---- 视频：GOP 并行（只在 `--threads > 1` 时；否则走下面那条串行老路）----
    //
    // 单线程**逐字节不变**。并行时每个 GOP 起一个新编码器 —— MP43 编码器**没有跨 GOP
    // 的状态**（MV / DC 网格每帧重置，`ref_planes` 只在 GOP 内部有用，我们不写 ext header
    // 所以 `flipflop_rounding` 恒 0），所以这条也逐字节不变（有测试钉）。
    //
    // 解码（参考链，天然串行）留在主线程当生产者，编码交给 worker 池；结果带 GOP 序号回来。
    // ASF 是「攒载荷、按写入顺序打包」，所以并行这条路最后要按 pts 合并视频与音频再写 ——
    // 实测合并出来的顺序与串行读到的**完全一致**（带音频直通也是 md5 相同）。
    //
    // 并行度**上限就是 GOP 数**：`-g 12` 时 300 帧 = 25 个 GOP，8 线程够用；
    // `-g 250` 时 300 帧只有 2 个 GOP，再多的线程也白搭（这是格式约束，不是实现问题）。
    //
    // ⚠️ **这条路隐含一个前提：能同时开 N 个编码器实例。** 桌面上的软件编码器成立，
    // **Android 上不成立** —— MediaCodec 的并发实例数有限、硬件编码器通常一个编解码器一份，
    // 而且真机是 big.LITTLE（8 核里常常只有 2 个大的），线程数 ≠ 倍数。
    // 所以：`--threads 1`（下面那条串行老路）必须始终是第一等的，
    // 它是唯一在「只有一个编码器」时也成立的形状；并行只是**可选加速**，不是架构。
    if opts.threads > 1 {
        let nthreads = opts.threads;
        let gop_len = if opts.keyframe_interval == 0 {
            1
        } else {
            opts.keyframe_interval
        };
        let cfg = EncoderConfig {
            qscale: opts.qscale,
            keyframe_interval: opts.keyframe_interval,
            ..Default::default()
        };
        let verify = opts.verify;

        // 在途帧的上限：GOP 数与字节预算取小（见 `INFLIGHT_BUDGET_BYTES` 的说明）。
        let gop_bytes = gop_len * w as usize * h as usize * 3 / 2;
        let queued = (INFLIGHT_BUDGET_BYTES / gop_bytes.max(1))
            .clamp(2, 2 * nthreads);
        let (jtx, jrx) = std::sync::mpsc::sync_channel::<(usize, Vec<(u32, Frame)>)>(queued);
        let jrx = std::sync::Arc::new(std::sync::Mutex::new(jrx));
        let (rtx, rrx) = std::sync::mpsc::channel::<(usize, Vec<(u32, Vec<u8>, bool)>, f64, u64)>();

        // 一个 worker 的活：不停从队列领 GOP、编完把结果发回去。
        // 抽成闭包是为了**主线程也能跑它** —— 8 核上开 8 个 worker 再加一个解码线程
        // 会超订，所以 worker 数 = `threads - 1`，主线程喂完最后一个 GOP 就下场补上。
        let work = move |jrx: &std::sync::Arc<
            std::sync::Mutex<std::sync::mpsc::Receiver<(usize, Vec<(u32, Frame)>)>>,
        >,
                         rtx: &std::sync::mpsc::Sender<(
            usize,
            Vec<(u32, Vec<u8>, bool)>,
            f64,
            u64,
        )>|
              -> std::result::Result<(), v2v_msmpeg4::Error> {
            loop {
                let job = {
                    let rx = jrx
                        .lock()
                        .map_err(|_| v2v_msmpeg4::Error::Bad("worker 接收端锁坏了"))?;
                    rx.recv()
                };
                let Ok((idx, gop)) = job else { break };
                let mut e = Encoder::new(cfg);
                let mut out = Vec::with_capacity(gop.len());
                let (mut se, mut np) = (0f64, 0u64);
                let mut vd = if verify { Some(Decoder::new()) } else { None };
                for (pts, f) in &gop {
                    let st = e.encode_frame(f)?;
                    if let Some(v) = vd.as_mut() {
                        let back = v.decode(&st, w, h)?;
                        for (a, b) in [(&f.y, &back.y), (&f.u, &back.u), (&f.v, &back.v)] {
                            for (&x, &y) in a.iter().zip(b.iter()) {
                                let d = x as f64 - y as f64;
                                se += d * d;
                            }
                            np += a.len() as u64;
                        }
                    }
                    out.push((*pts, st, e.last_key_frame));
                }
                if rtx.send((idx, out, se, np)).is_err() {
                    break;
                }
            }
            Ok(())
        };

        // ⚠️ worker 数 = `nthreads`（不是 `nthreads - 1`）：主线程在读循环期间要解码、
        // 拿不到编码活，少起一个 worker 就等于少一个核。实测 `--threads 2` 因此只有
        // 1.17× 加速（10.19 vs 11.88 s），改成 2 个 worker 后接近 2×。
        let mut workers = Vec::with_capacity(nthreads);
        for _ in 0..nthreads {
            let jrx = std::sync::Arc::clone(&jrx);
            let rtx = rtx.clone();
            workers.push(std::thread::spawn(move || work(&jrx, &rtx)));
        }

        // 主线程：解封装 + 解码 → 攒够一个 GOP 就发出去；音频载荷先按序收着。
        let mut audio_buf: Vec<(u32, Vec<u8>)> = Vec::new();
        let mut gop: Vec<(u32, Frame)> = Vec::new();
        let mut n_gop = 0usize;
        while let Some(s) = demux.next()? {
            if Some(s.stream) == audio_num {
                if opts.copy_audio {
                    let pts = u32::try_from(s.timestamp_ms.max(0)).unwrap_or(0);
                    audio_buf.push((pts, s.data));
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
            gop.push((pts, frame));
            frames += 1;
            if gop.len() >= gop_len {
                jtx.send((n_gop, std::mem::take(&mut gop)))
                    .map_err(|_| Error::Unsupported("视频 worker 池提前退出".into()))?;
                n_gop += 1;
            }
            if let Some(max) = opts.max_frames {
                if frames >= max {
                    break;
                }
            }
        }
        if !gop.is_empty() {
            jtx.send((n_gop, gop))
                .map_err(|_| Error::Unsupported("视频 worker 池提前退出".into()))?;
            n_gop += 1;
        }
        drop(jtx);

        // 主线程下场编（队列已经封口，跑完自然退出）。
        let main_res = work(&jrx, &rtx);
        drop(rtx);
        for w in workers {
            match w.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(Error::Msmpeg4(e)),
                Err(_) => return Err(Error::Unsupported("视频 worker 线程 panic".into())),
            }
        }
        main_res.map_err(Error::Msmpeg4)?;

        // 结果乱序回来 → 按 GOP 序号归位，再按 pts 合并视频与音频。
        let mut gops: Vec<Vec<(u32, Vec<u8>, bool)>> = vec![Vec::new(); n_gop];
        for (idx, out, se, np) in rrx {
            if idx < gops.len() {
                gops[idx] = out;
                sq_err += se;
                n_px += np;
            }
        }
        let mut vid: Vec<(u32, &Vec<u8>, bool)> = Vec::with_capacity(frames);
        for g in &gops {
            for (pts, st, key) in g {
                vid.push((*pts, st, *key));
            }
        }
        vid.sort_by_key(|(p, _, _)| *p);
        let mut aud: Vec<(u32, &Vec<u8>)> = audio_buf.iter().map(|(p, d)| (*p, d)).collect();
        aud.sort_by_key(|(p, _)| *p);
        let (mut i, mut j) = (0usize, 0usize);
        while i < vid.len() || j < aud.len() {
            let take_video = j >= aud.len() || (i < vid.len() && vid[i].0 <= aud[j].0);
            if take_video {
                mux.write_video(vid[i].1, vid[i].2, vid[i].0)?;
                if vid[i].2 {
                    key_frames += 1;
                }
                i += 1;
            } else {
                mux.write_audio(aud[j].1, aud[j].0)?;
                j += 1;
            }
        }
        let wmv = mux.finish();
        write_out(output, &wmv)?;
        let psnr_db = if opts.verify && n_px > 0 {
            Some(psnr(sq_err, n_px))
        } else {
            None
        };
        return Ok(ConvertReport {
            frames,
            key_frames,
            audio_payloads,
            skipped_audio,
            bytes_in,
            bytes_out: wmv.len() as u64,
            psnr_db,
            video_kbps: None,
        });
    }

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
        // `.wmv`（MP43）那条路还没有码率控制。
        video_kbps: None,
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

/// 一个「将要写进 muxer 的载荷」+ 它的**交错序号**。
///
/// 序号 = `(样本序号, 样本内序号)`。串行版里「来自样本 i 的事件」全部排在
/// 「来自样本 i+1 的事件」之前，所以这个键与串行版的事件顺序**逐一对齐** ——
/// 排序回放之后 `mdat` 的排布就与原版逐字节相同。
/// 音频尾巴（`flush_aac`）排在所有样本之后，用 `usize::MAX`。
struct Pending {
    key: (usize, usize),
    kind: PendingKind,
}

enum PendingKind {
    Video {
        data: Vec<u8>,
        key_frame: bool,
        duration: u32,
    },
    Audio {
        data: Vec<u8>,
        duration: u32,
    },
}

/// 编**一个 GOP** 的帧，返回（载荷, 关键帧数, 码控统计）。
///
/// `all_i` 对应 `-g 0`：那条路上每一帧都是独立的 I 帧（见 `ConvertOptions` 的说明），
/// 所以调 `encode_i_frame` 而不是 `encode_frame`（后者的 `0` 含义是「无 GOP、全 P」）。
fn encode_items(
    enc: &mut v2v_mp4v::Encoder,
    items: Vec<(usize, Frame, u32)>,
    all_i: bool,
) -> std::result::Result<(Vec<Pending>, usize, (u64, u64)), String> {
    let mut out = Vec::with_capacity(items.len());
    let mut key_frames = 0usize;
    for (idx, frame, duration) in items {
        let stream = if all_i {
            enc.encode_i_frame(&frame)
        } else {
            enc.encode_frame(&frame)
        }
        .map_err(|e| e.to_string())?;
        // 关键帧位来自编码器 —— 全 I 帧时 `stss` 会被整个省掉。
        if enc.last_key_frame {
            key_frames += 1;
        }
        out.push(Pending {
            key: (idx, 0),
            kind: PendingKind::Video {
                data: stream,
                key_frame: enc.last_key_frame,
                duration,
            },
        });
    }
    Ok((out, key_frames, enc.rate_control_stats().unwrap_or((0, 0))))
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

    let track = demux
        .video_stream()
        .ok_or_else(|| Error::Unsupported("文件里没有视频轨（或没有格式信息）".into()))?;
    let (vnum, w, h) = (track.number, track.width, track.height);

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

    // ---- 视频编码器配置：MPEG-4 Part 2 ----
    //
    // 编码器**实例**在 worker 线程里建（见下面「视频 GOP 并行」）——
    // 配置是 `Copy` 的，每个线程各拿一份。
    let mp4v_cfg = v2v_mp4v::EncoderConfig {
        width: w,
        height: h,
        fps_num: fps,
        qscale: opts.qscale,
        bitrate: opts.video_bitrate,
        keyframe_interval: opts.keyframe_interval,
        ..Default::default()
    };
    // `esds` 要的 DSI（VOS + 视觉对象 + VOL + 用户数据）与编码器**状态**无关，
    // 用一个临时实例取一次就够 —— 编码器实例现在归 worker 线程所有。
    let dsi = v2v_mp4v::Encoder::new(mp4v_cfg)?.decoder_specific_info();

    // ---- 封装器 ----
    //
    // ⚠️ `decoder_specific_info()` 必须在 `Mp4Muxer::new` 时给出去，之后改不了；
    // 它是「VOS + 视觉对象 + VOL + 用户数据」整段，`ffprobe` 靠它报 profile / level。
    const VTS: u32 = 1000;
    let vtrack = v2v_mp4::VideoTrack {
        width: w as u16,
        height: h as u16,
        timescale: VTS,
        codec: v2v_mp4::VideoCodec::Mp4v {
            decoder_specific_info: dsi,
        },
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
                threads: opts.threads,
                ms_stereo: opts.ms_stereo,
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
                    // **1024** —— AAC 编解码链的**净延迟**（一个 AAC 帧的编码器前瞻）。
                    // 不声明它，音频整体晚 23 ms —— 而画质、帧数、能不能解开这些判据
                    // 全都不会报警，只有对齐偏移看得见。
                    //
                    // 用 `edts`/`elst` 声明（让解码器跳过去），**不是**把输入开头剪掉 ——
                    // 剪掉的是真内容，而且时长会跟着短。
                    //
                    // ⚠️ 这里曾经是 `0`，理由是「实测没有延迟」—— 那次测量是**错的**：
                    // 对齐搜索的包络粒度太粗，周期性素材上真峰被假峰盖过。
                    // 换成**逐样本**互相关（真峰必须显著地赢）才量准。细节见 `TODO.md`。
                    priming: 1024,
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

    let mut dec = open_video_decoder(track.fourcc, &track.extra, w, h)?;
    let mut frames = 0usize;
    let mut key_frames = 0usize;
    let mut skipped_audio = 0usize;

    // ---- 音频 worker：把「解 WMA + 编 AAC」整条链挪到另一个线程 ----
    //
    // 为什么能并行：AAC-LC **逐帧独立** —— 偏移搜索是「二分到最小可行偏移」，
    // `sf_offset_prev` 只决定从哪儿开始探、不决定答案（`v2v-aac` 的注释写死了这点），
    // WMA 解码也是逐帧流式的。所以音频那条链与视频那条互不依赖。
    //
    // ⚠️ **两边都不直接写 muxer**：`mdat` 是按 `write_video` / `write_audio` 的
    // **调用顺序**铺出来的（`finish()` 只补长度 + 追加 `moov`），所以并行之后必须
    // 把交错顺序**原样复现**。做法：给每个载荷记一个 `(样本序号, 样本内序号)`，
    // 收齐后按这个键排序回放 —— 与串行版的事件顺序逐一对齐。
    let a_point = audio.as_ref().map_or(0, |a| a.point);
    let a_channels = src_audio.as_ref().map_or(0, |af| af.channels as usize);
    // ---- 音频**两级流水线**：WMA 解码 → AAC 编码各占一个线程 ----
    //
    // 为什么拆：原来「WMA 解码 + AAC 编码」串在**同一个** worker 里，它的耗时是
    // **两者之和**。视频侧并行到 8 线程之后已经跑到 ~0.24 s，音频这条 0.47 s
    // 就成了 wall 的下界 —— 实测 `--pcm`（省掉 AAC 编码）wall 0.497 → **0.235**。
    // 拆成两级之后音频那条变成 **max(解码, 编码)**，与视频侧并列：
    // **wall = max(视频, 解码, 编码)**。
    //
    // ⚠️ **AAC 编码器是有状态的**（`sf_offset_prev` / `window_seq_prev` / 重叠历史），
    // 所以第二级必须是**单线程顺序**的 —— 这里正是，不要把它也拆开。
    let (tx_audio, rx_audio) = std::sync::mpsc::channel::<(usize, Vec<u8>)>();
    let (tx_pcm, rx_pcm) = std::sync::mpsc::channel::<(usize, Vec<u8>)>();
    let mut wma_dec = match audio.as_mut() {
        Some(a) => a.decoder.take(),
        None => None,
    };
    let mut skip_left = audio.as_ref().map_or(0, |a| a.skip);
    // 第一级：WMA 解码（无编码器时就是 PCM 直通）+ 跳过编码器延迟。
    let decode_worker = std::thread::spawn(move || -> std::result::Result<(), String> {
        while let Ok((idx, data)) = rx_audio.recv() {
            let mut pcm = match wma_dec.as_mut() {
                Some(d) => wma_frame_to_pcm(d.decode_superframe(&data).map_err(|e| e.to_string())?),
                None => data,
            };
            if skip_left > 0 {
                let drop = (skip_left * a_point).min(pcm.len());
                pcm.drain(..drop);
                skip_left -= drop / a_point;
            }
            if tx_pcm.send((idx, pcm)).is_err() {
                break; // 编码侧已退出（出错），由下面的 join 报出来
            }
        }
        Ok(())
    });
    // 第二级：AAC 编码（或 PCM 直通时按采样点切块）。
    let mut aac_enc = aac.take();
    let audio_worker = std::thread::spawn(move || -> std::result::Result<(Vec<Pending>, usize), String> {
        let mut out: Vec<Pending> = Vec::new();
        let mut pending: Vec<Vec<f32>> = Vec::new();
        // 每帧「攒够时的载荷序号」—— 见 `push_aac_pcm` 的说明。
        let mut marks: Vec<usize> = Vec::new();
        let mut audio_points = 0usize;
        while let Ok((idx, pcm)) = rx_pcm.recv() {
            if let Some(enc) = aac_enc.as_mut() {
                // ---- AAC：攒够 1024 个采样点再编一帧 ----
                //
                // 源 PCM 是「交织的小端 16 位」，AAC 编码器要的是
                // 「每声道一条、f32、范围 ±32768」——所以要先拆开、归一。
                push_aac_pcm(&mut pending, &mut marks, &pcm, a_point, a_channels, idx);
                {
                    // ⚠️ key 的「样本序号」用 `emit` 传进来的那个（不是外层的 `idx`）：
                    // 批量编码时一帧可能跨好几个载荷，用同一个 idx 会让 `mdat` 的
                    // 交错顺序变掉 —— 而 `mdat` 是按写入调用顺序铺的。
                    let mut emit = |d: &[u8], dur: u32, at: usize| {
                        out.push(Pending {
                            key: (at, 0),
                            kind: PendingKind::Audio {
                                data: d.to_vec(),
                                duration: dur,
                            },
                        });
                    };
                    drain_aac_frames(enc, &mut pending, &mut marks, &mut emit, &mut audio_points)
                        .map_err(|e| e.to_string())?;
                }
                continue;
            }
            for (j, c) in pcm.chunks_exact(a_point).enumerate() {
                out.push(Pending {
                    key: (idx, j),
                    kind: PendingKind::Audio {
                        data: c.to_vec(),
                        duration: 1,
                    },
                });
                audio_points += 1;
            }
        }
        // 收尾：把 AAC 的尾巴（不足 1024 个采样点）补零编出去。
        if let Some(enc) = aac_enc.as_mut() {
            {
                let mut emit = |d: &[u8], dur: u32, at: usize| {
                    out.push(Pending {
                        key: (at, 0),
                        kind: PendingKind::Audio {
                            data: d.to_vec(),
                            duration: dur,
                        },
                    });
                };
                flush_aac(enc, &mut pending, &mut marks, &mut emit, &mut audio_points)
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok((out, audio_points))
    });

    // ---- 视频 worker 池：GOP 并行 ----
    //
    // ⚠️ **这条不是逐字节不变的**（与上面音频 worker 不同）：`mv_prev`（运动估计的
    // 第三个起点）与码控的 `cur_qscale` 都**跨 GOP 延续**，按 GOP 并行就必须重置它们 ——
    // 代价是每个 GOP 的第一个 P 帧少一个搜索起点、码控从头收敛。
    // 所以**单线程（默认）仍沿用同一个编码器**、跨 GOP 连续，与老行为逐字节相同；
    // 只有 `--threads > 1` 才每个 GOP 起一个全新编码器。
    //
    // GOP 边界与编码器内部的判据一致（`frames_since_i >= keyframe_interval`，
    // 首帧恒 I）—— 所以按 `gop_len` 切出来的每一块，第一帧正好是 I 帧。
    // `-g 0`（全 I 帧）时每帧独立，`gop_len = 1`。
    let all_i = opts.keyframe_interval == 0;
    let gop_len = if all_i { 1 } else { opts.keyframe_interval };
    // 解码池与编码池**分核**。实测 1080p 上「解码 : 编码」约 **1 : 3**
    // （每帧解码 ~10 ms、mp4v 编码 ~30 ms），所以解码拿 1/4 的核才均衡 ——
    // 给多了反而把编码饿死（实测 6 解 / 2 编比 2 解 / 6 编慢一倍多）。
    // 并行解码的前提是「从任意 I 帧起解 == 整条链解到那里」，逐个编码器验证过：
    // - **WMV3（VC-1）**：`crates/vc1/tests/gop_start.rs`（真样本 278 个关键帧全查，0 差）→ **开**。
    // - **MP43**：`crates/msmpeg4/tests/gop_start.rs`（自编 6 条 GOP 全查，0 差）→ **也正确，
    //   但不开**：实测开了更慢（300 帧 8 线程：串行解码 2.09/2.24 s，池 2.58/2.93 s，
    //   产物 md5 相同）。原因是 MP43 的解码只有 VC-1 的 1/3 工作量 —— 串行解码本来就
    //   在**主线程**上与编码 worker 重叠着跑，再抽核出来反而把编码饿死。
    //   要开就把它加进下面这行（代码路径是通的，`DecKind::Mp43` 就是它）。
    // - **WMV1 / WMV2**：WMV2 挂着 `IntraX8.prediction_table`（上游只在初始化清一次、
    //   **跨帧累积**），新解码器是全零表，j_type 流会解出不同像素；WMV1 的 `bit_rate`
    //   有可能不在 I 帧上出现（`br == 0` 时延续旧值）—— 两者都没有可验证的流，串行。
    let dec_can_parallel = track.fourcc == WMV3;
    let par_decode = opts.threads > 1 && dec_can_parallel && gop_len > 1;
    let n_dec = if par_decode { (opts.threads / 4).max(1) } else { 0 };
    let nthreads = if par_decode {
        opts.threads.saturating_sub(n_dec).max(1)
    } else {
        opts.threads.max(1)
    };
    // 有界通道：解码（主线程）领先编码（worker）最多 `2 * nthreads` 个 GOP，
    // 解码帧就不会堆在内存里（一帧 320×240 的 YUV 是 115 KB）。
    let (vtx, vrx) = std::sync::mpsc::sync_channel::<Vec<(usize, Frame, u32)>>(2 * nthreads);
    let vrx = std::sync::Arc::new(std::sync::Mutex::new(vrx));
    let (otx, orx) = std::sync::mpsc::channel::<(Vec<Pending>, usize, (u64, u64))>();

    // 一个 worker 的活：不停从队列领一块 GOP、编完把结果发回去。
    //
    // 抽成闭包是为了**主线程也能跑它** —— 主线程解码那一趟约占全程 6%，跑完就闲着了，
    // 8 核上等于白丢一个核。所以 worker 数 = `threads − 1`，主线程喂完最后一块就下场补上。
    // （`.wmv` 那条路一开始就是这么写的；这条是补上的。）
    let work = move |vrx: &std::sync::Arc<
        std::sync::Mutex<std::sync::mpsc::Receiver<Vec<(usize, Frame, u32)>>>,
    >,
                     otx: &std::sync::mpsc::Sender<(Vec<Pending>, usize, (u64, u64))>|
          -> std::result::Result<(), String> {
        // 单线程时只建一次编码器，跨 GOP 连续（= 老行为）。
        let mut keep = if nthreads == 1 {
            Some(v2v_mp4v::Encoder::new(mp4v_cfg).map_err(|e| e.to_string())?)
        } else {
            None
        };
        loop {
            let job = {
                let rx = vrx.lock().map_err(|e| e.to_string())?;
                rx.recv()
            };
            let Ok(items) = job else { break };
            let (out, kf, stats) = match keep.as_mut() {
                Some(enc) => encode_items(enc, items, all_i)?,
                None => {
                    let mut enc =
                        v2v_mp4v::Encoder::new(mp4v_cfg).map_err(|e| e.to_string())?;
                    encode_items(&mut enc, items, all_i)?
                }
            };
            otx.send((out, kf, stats)).map_err(|e| e.to_string())?;
        }
        Ok(())
    };

    // ⚠️ **至少留一个 worker**：`threads == 1` 时 `nthreads - 1` 是 0，而 GOP 通道是
    // **有界**的（容量 `2 × nthreads`）—— 没有消费者时主线程喂到第 3 块就永久阻塞。
    // （`.wmv` 那条只在 `threads > 1` 时进，所以没这个坑；这条两条都要成立。）
    // ⚠️ **worker 数 = nthreads**（不是 nthreads-1）。
    //
    // 原来留一个核给主线程「最后下场编」，但主线程在**读循环期间**要解码、拿不到活 ——
    // 实测 `--threads 2` 只跑 1 个编码线程（7.31 → 7.14 s，等于没加速）。改成按
    // `nthreads` 起 worker 之后，2 线程才真的是 2 个编码线程。
    // `nthreads == 1` 时仍至少留一个 worker —— 否则有界通道没人消费，读循环会死锁。
    // 输出不受影响：`nthreads > 1` 时每个 GOP 本来就是全新编码器，worker 数不参与判决。
    let nworkers = nthreads.max(1);
    let mut vworkers = Vec::with_capacity(nworkers);
    for _ in 0..nworkers {
        let vrx = std::sync::Arc::clone(&vrx);
        let otx = otx.clone();
        vworkers.push(std::thread::spawn(move || work(&vrx, &otx)));
    }

    // 上一帧的视频时间戳（毫秒）—— `stts` 写的是**真实间隔**。
    let mut prev_vts: Option<i64> = None;
    let mut last_dur = (VTS as f64 / fps as f64).round().max(1.0) as u32;
    let mut stop = false;
    let mut sample_idx = 0usize;
    // 攒够一个 GOP 就派给 worker。
    let mut gop_buf: Vec<(usize, Frame, u32)> = Vec::with_capacity(gop_len);

    // ---- 视频**解码池**：按 GOP 独立解 ----
    //
    // 前提是「从任意 I 帧起解 == 整条链解到那里」，由 `crates/vc1/tests/gop_start.rs`
    // 钉着（真样本上抽查 50 个关键帧，逐像素 0 差）。
    //
    // ⚠️ **只对 VC-1 开**：MS-MPEG-4 那套的跨帧状态（`flipflop_rounding` / `Stats` /
    // WMV2 的 IntraX8）还没验过，别顺手打开。
    // ⚠️ `par_decode == false` 时**走下面那条串行老路，一个字节都不变**。
    let mut sample_gop: Vec<(usize, Vec<u8>, u32)> = Vec::with_capacity(gop_len);
    let mut gop_seq = 0usize;
    let mut dec_txs: Vec<std::sync::mpsc::SyncSender<Vec<(usize, Vec<u8>, u32)>>> = Vec::new();
    let mut dec_workers: Vec<std::thread::JoinHandle<()>> = Vec::new();
    let mut collector: Option<std::thread::JoinHandle<std::result::Result<(), String>>> = None;
    if par_decode {
        #[derive(Clone)]
        enum DecKind {
            Vc1,
            Mp43,
        }
        let kind = if track.fourcc == WMV3 { DecKind::Vc1 } else { DecKind::Mp43 };
        let extra = track.extra.clone();
        // 每个 worker 一条**专用**结果通道：派活与取活都是轮转，收端按同样的轮转顺序
        // 取就天然保持了全局顺序 —— 不需要重排缓冲。
        let mut res_rxs = Vec::new();
        for _ in 0..n_dec {
            let kind = kind.clone();
            let extra = extra.clone();
            let (jtx, jrx) = std::sync::mpsc::sync_channel::<Vec<(usize, Vec<u8>, u32)>>(2);
            let (rtx, rrx) =
                std::sync::mpsc::sync_channel::<std::result::Result<Vec<(usize, Frame, u32)>, String>>(2);
            res_rxs.push(rrx);
            dec_txs.push(jtx);
            dec_workers.push(std::thread::spawn(move || {
                while let Ok(gop) = jrx.recv() {
                    let r = (|| -> std::result::Result<Vec<(usize, Frame, u32)>, String> {
                        // 每条 GOP 一条**全新的**解码器（前提就是它和串行链等价）。
                        let mut out = Vec::with_capacity(gop.len());
                        match kind {
                            DecKind::Vc1 => {
                                let seq = v2v_vc1::SeqHeader::parse(&extra)
                                    .map_err(|e| e.to_string())?;
                                let mut d =
                                    v2v_vc1::Decoder::new(seq, w, h).map_err(|e| e.to_string())?;
                                for (idx, data, dur) in gop {
                                    d.decode_frame(&data).map_err(|e| e.to_string())?;
                                    let f = d.frame();
                                    out.push((
                                        idx,
                                        Frame {
                                            width: f.width,
                                            height: f.height,
                                            y: f.y,
                                            u: f.u,
                                            v: f.v,
                                        },
                                        dur,
                                    ));
                                }
                            }
                            DecKind::Mp43 => {
                                let mut d = Decoder::with_version(Version::V3);
                                for (idx, data, dur) in gop {
                                    let f = d.decode(&data, w, h).map_err(|e| e.to_string())?;
                                    out.push((
                                        idx,
                                        Frame {
                                            width: f.width,
                                            height: f.height,
                                            y: f.y,
                                            u: f.u,
                                            v: f.v,
                                        },
                                        dur,
                                    ));
                                }
                            }
                        }
                        Ok(out)
                    })();
                    let failed = r.is_err();
                    if rtx.send(r).is_err() || failed {
                        break;
                    }
                }
            }));
        }
        // 收端：按轮转顺序取，再把帧**重新按编码器的 GOP 长度**切块 ——
        // 这样编码器看到的分批与串行版完全一样，输出不受影响。
        let vtx_c = vtx.clone();
        collector = Some(std::thread::spawn(move || -> std::result::Result<(), String> {
            let mut batch: Vec<(usize, Frame, u32)> = Vec::with_capacity(gop_len);
            let mut cur = 0usize;
            loop {
                match res_rxs[cur].recv() {
                    Ok(Ok(gop)) => {
                        for it in gop {
                            batch.push(it);
                            if batch.len() >= gop_len {
                                vtx_c
                                    .send(std::mem::take(&mut batch))
                                    .map_err(|_| "视频线程提前退出".to_string())?;
                                batch = Vec::with_capacity(gop_len);
                            }
                        }
                        cur = (cur + 1) % n_dec;
                    }
                    Ok(Err(e)) => return Err(e),
                    Err(_) => break,
                }
            }
            if !batch.is_empty() {
                vtx_c.send(batch).map_err(|_| "视频线程提前退出".to_string())?;
            }
            Ok(())
        }));
    }

    loop {
        let s = match queue.pop_front() {
            Some(s) => Some(s),
            None if !stop => demux.next()?,
            None => None,
        };
        let Some(s) = s else { break };
        let idx = sample_idx;
        sample_idx += 1;

        // 最后一个已收视频样本的结束时刻（毫秒）。还没收到视频时给 `i64::MAX`，
        // 这样开头的音频不会被误裁。
        let video_end_ms = prev_vts.map_or(i64::MAX, |p| p + i64::from(last_dur));

        // ---- 音频：原样转给 worker（它自己解 WMA、编 AAC）----
        if Some(s.stream) == audio_num {
            // ⚠️ **`-n` 到了之后，队列里预读的音频样本不能无条件转发。**
            //
            // 上面为了探帧率预读了 8 个样本（见下面的注释）。`-n` 触发后主循环
            // 只是「不再解视频」，队列里那 8 个样本还会被走一遍 —— 其中音频会被
            // 照常转给 worker。实测后果：**10 s 的视频配了 12.1 s 的音频**，
            // 产物多出约 2 s，播放器尾部会多一段声音（视频已停）。
            //
            // 正解是按**时间戳**裁：只放行仍落在视频时长之内的音频样本。
            // `video_end_ms` 是最后一个已收视频样本的「起点 + 时长」，
            // 在 `stop` 置位之后就固定了（stop 后不再更新 `prev_vts` / `last_dur`）。
            if stop && s.timestamp_ms >= video_end_ms {
                continue;
            }
            if tx_audio.send((idx, s.data)).is_err() {
                break; // worker 已退出（出错），由下面的 join 报出来
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

        // `--max-frames` 到了就**不再解视频**，但队列里剩下的音频还要转给 worker。
        //
        // ⚠️ 这一句是必需的，不是防御性代码：上面为了探帧率**预读了 8 个样本**，
        // 那 8 个在 `stop` 之后仍然躺在 `queue` 里 —— 少了这句，`-n 6` 会产出 8 帧
        // （实测过，`.mp4` 那条路；`.wmv` 那条路没有预读，所以不受影响）。
        if stop {
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

        if par_decode {
            // ---- 并行解码：把**样本**攒成 GOP 派给解码池，本线程不解 ----
            // 新 GOP 的边界：当前样本是关键帧、且手上已经攒了东西。
            if s.key_frame && !sample_gop.is_empty() {
                let gop = std::mem::take(&mut sample_gop);
                dec_txs[gop_seq % dec_txs.len()]
                    .send(gop)
                    .map_err(|_| Error::Io(std::io::Error::other("解码线程提前退出")))?;
                gop_seq += 1;
                sample_gop = Vec::with_capacity(gop_len);
            }
            sample_gop.push((idx, s.data, last_dur));
            frames += 1;
        } else {
            let frame = dec.decode(&s.data, w, h)?;
            // ⚠️ **`-g 0` 在 `.mp4` 这条路上就是「全 I 帧」** ——
            // 这条路的默认值是 0（每个 I 帧都独立可解，见 `ConvertOptions` 的说明），
            // 所以 worker 那边不能直接调 `encode_frame`（在 `mp4v` 里 `0` 的含义是
            // 「无 GOP」，会编 P 帧、而且第一个 P 帧会因为没有参考帧而报错）——
            // `all_i` 就是为这个传进去的。
            gop_buf.push((idx, frame, last_dur));
            frames += 1;
            if gop_buf.len() >= gop_len {
                vtx.send(std::mem::take(&mut gop_buf))
                    .map_err(|_| Error::Io(std::io::Error::other("视频线程提前退出")))?;
            }
        }
        if let Some(max) = opts.max_frames {
            if frames >= max {
                stop = true;
                // 已经预读进队列的音频还要写完，所以不能直接 break。
            }
        }
    }
    // 最后一块不满一个 GOP 也要发出去。
    if par_decode {
        if !sample_gop.is_empty() {
            let gop = std::mem::take(&mut sample_gop);
            dec_txs[gop_seq % dec_txs.len()]
                .send(gop)
                .map_err(|_| Error::Io(std::io::Error::other("解码线程提前退出")))?;
        }
        // 关掉派活端 → worker 收工 → 收端取完最后一块也收工。
        drop(dec_txs);
        for h in dec_workers {
            let _ = h.join();
        }
        if let Some(c) = collector {
            match c.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    return Err(Error::Io(std::io::Error::other(format!("视频解码线程：{e}"))))
                }
                Err(_) => return Err(Error::Io(std::io::Error::other("视频解码线程 panic"))),
            }
        }
    } else if !gop_buf.is_empty() {
        vtx.send(gop_buf).map_err(|_| Error::Io(std::io::Error::other("视频线程提前退出")))?;
    }
    drop(vtx);

    // 关掉发送端：worker 把剩下的样本处理完、编掉 AAC 的尾巴之后退出。
    drop(tx_audio);

    // **主线程下场编**（队列已经封口，跑完自然退出）。音频那两条线程一直在并行跑着，
    // 这里不用等它们 —— 等它们会白白空出一个核。
    let main_res = work(&vrx, &otx);
    drop(otx);

    // 解码线程退出 → 它的 `tx_pcm` 随之 drop → 编码线程收完尾巴后自然退出。
    match decode_worker.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(Error::Io(std::io::Error::other(format!("音频解码线程：{e}")))),
        Err(_) => return Err(Error::Io(std::io::Error::other("音频解码线程 panic"))),
    }
    let (mut audio_events, audio_points) = match audio_worker.join() {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Err(Error::Io(std::io::Error::other(format!("音频编码线程：{e}")))),
        Err(_) => return Err(Error::Io(std::io::Error::other("音频编码线程 panic"))),
    };

    // 视频 worker 收尾：等全部退出，再把结果收齐（`orx.iter()` 在所有 `otx`
    // 都随线程一起 drop 之后自然结束）。
    for wk in vworkers {
        match wk.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(Error::Io(std::io::Error::other(format!("视频线程：{e}")))),
            Err(_) => return Err(Error::Io(std::io::Error::other("视频线程 panic"))),
        }
    }
    if let Err(e) = main_res {
        return Err(Error::Io(std::io::Error::other(format!("视频（主线程）：{e}"))));
    }
    let mut video_events: Vec<Pending> = Vec::with_capacity(frames);
    let mut rate_stats = (0u64, 0u64);
    for (out, kf, st) in orx.iter() {
        video_events.extend(out);
        key_frames += kf;
        rate_stats.0 += st.0;
        rate_stats.1 += st.1;
    }

    // ---- 音频裁到视频时长 ----
    //
    // ⚠️ `-n N` 只限**视频帧**，但音频载荷是按 ASF 包送来的，落在预读队列里的
    // 那部分会被照常解码、编码。实测后果：**10 s 的视频配了 12.1 s 的音频**
    // （570 个 AAC 帧对 480 个），产物多出约 2 s 尾巴。
    //
    // 修法：在回放前按**采样点数**裁掉超出视频时长的音频。目标采样点数 =
    // `frames / fps` 秒 × 采样率。`duration` 对 AAC 帧是 1024（采样点）、
    // 对 PCM 是 1（每采样点），单位一致，可以累计。
    //
    // 整帧裁掉（不拆帧）：AAC 的粒度是 1024 采样点 ≈ 21 ms，够细了。
    if let Some(af) = src_audio.as_ref() {
        let want = (u64::try_from(frames).unwrap_or(0) * u64::from(af.sample_rate) + fps as u64 - 1) / fps as u64;
        let mut acc = 0u64;
        audio_events.retain(|p| match &p.kind {
            PendingKind::Audio { duration, .. } => {
                if acc >= want {
                    return false;
                }
                acc += u64::from(*duration);
                true
            }
            PendingKind::Video { .. } => true,
        });
    }

    // ---- 按**原交错顺序**回放：`mdat` 的排布必须与串行版逐字节相同 ----
    let mut all: Vec<Pending> = Vec::with_capacity(video_events.len() + audio_events.len());
    all.extend(video_events);
    all.extend(audio_events);
    all.sort_by_key(|p| p.key); // 稳定排序；键本来就互不相同
    for p in all {
        match p.kind {
            PendingKind::Video {
                data,
                key_frame,
                duration,
            } => mux.write_video(&data, key_frame, duration)?,
            PendingKind::Audio { data, duration } => mux.write_audio(&data, duration)?,
        }
    }

    let mp4 = mux.finish();
    write_out(output, &mp4)?;

    // 码率控制开着时把**实际**码率报出去 —— 这是唯一能证明它生效的数。
    // 多线程时每个 worker 各有一份统计，这里按 (帧数, 位数) 求和。
    let video_kbps = {
        let (n, bits) = rate_stats;
        (n > 0).then(|| bits as f64 / n as f64 * f64::from(fps) / 1000.0)
    };

    Ok(ConvertReport {
        frames,
        key_frames,
        audio_payloads: audio_points,
        skipped_audio,
        bytes_in,
        bytes_out: mp4.len() as u64,
        // 没有 mp4v 解码器可比，见模块文档。
        psnr_db: None,
        video_kbps,
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
fn push_aac_pcm(
    pending: &mut Vec<Vec<f32>>,
    marks: &mut Vec<usize>,
    pcm: &[u8],
    point: usize,
    channels: usize,
    idx: usize,
) {
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
    // ⚠️ **逐帧记「攒够时的载荷序号」**：`emit` 的 key 要用它才能和串行版的事件
    // 顺序对齐（`mdat` 是按写入调用顺序铺的）。攒批之后一帧可能是在**更早**的
    // 载荷就攒够的 —— 用「当前载荷」会让交错顺序漂掉。
    let have = pending.first().map_or(0, |p| p.len());
    while (marks.len() + 1) * AAC_FRAME_SAMPLES <= have {
        marks.push(idx);
    }
}

/// 从待编缓冲里尽量多地编出 AAC 帧（每帧 1024 个采样点）并**交给 `emit`**。
///
/// ⚠️ **不足 1024 的部分要留着**，不能提前编 —— 提前编会补零，
/// 而补进去的零会在输出里变成一小段可听的静音（每 1024 个点补一次就够听出来）。
/// 剩下的尾巴由 `flush_aac` 在收尾时补零编出去。
///
/// ⚠️ 参数是 `emit` 回调而**不是** `&mut Mp4Muxer`：音频那条链现在跑在
/// **另一个线程**上（见 `convert_to_mp4` 的音频 worker），不能直接碰 muxer ——
/// 而 `mdat` 是按写入**调用顺序**铺的，并行后必须把交错顺序原样复现。
/// 交给回调，串行与并行两条路就共用同一份实现，不会漂出第二份。
fn drain_aac_frames(
    enc: &mut v2v_aac::Encoder,
    pending: &mut Vec<Vec<f32>>,
    marks: &mut Vec<usize>,
    emit: &mut dyn FnMut(&[u8], u32, usize),
    audio_samples: &mut usize,
) -> Result<(), Error> {
    /// 一批攒多少帧。
    ///
    /// ⚠️ **块要长**：`encode_frames` 把批按连续区间切给各线程，而**每个块的首帧
    /// 只能冷启动**偏移搜索（拿不到上一帧的种子）—— 块越长，冷启动的占比越低。
    /// 64 帧 ≈ 1.5 秒音频、内存里最多堆 64 × 2 × 1024 × 4 B = 512 KB PCM，可以接受。
    const BATCH: usize = 64;
    loop {
        let ready = marks.len().min(BATCH);
        if ready == 0 {
            return Ok(());
        }
        let mut frames: Vec<v2v_aac::Frame> = Vec::with_capacity(ready);
        let mut at: Vec<usize> = Vec::with_capacity(ready);
        for _ in 0..ready {
            let planes: Vec<Vec<f32>> = pending
                .iter_mut()
                .map(|p| p.drain(..AAC_FRAME_SAMPLES).collect())
                .collect();
            frames.push(v2v_aac::Frame { planes });
            at.push(marks.remove(0));
        }
        // ⚠️ `encode_frames` **按帧并行**（内部先预扫窗序列、再分块编）。
        // 输出与逐帧 `encode_frame` **不是**逐字节相同：`plan_group` 的三级细化
        // （12→3→1）是以「上一帧留下的 `sf_offset`」为锚的**近似搜索**，换锚点就
        // 会落到不同格点 —— 实测 350 帧里差 2 帧、合计 14 字节（SNR 相同）。
        // `v2v-aac` 的测试钉的是「帧数相同 + 每帧体积差 ≤ 16 字节」。
        let streams = enc.encode_frames(&frames)?;
        for (stream, idx) in streams.iter().zip(&at) {
            // AAC 一帧就是 1024 个采样点，时间基是采样率 —— 所以时长写 1024。
            emit(stream, AAC_FRAME_SAMPLES as u32, *idx);
            // 计数用**输出侧的采样点数**（与 PCM 那条路同单位，报告里好对比）。
            *audio_samples += AAC_FRAME_SAMPLES;
        }
    }
}

/// 收尾：把不足一帧的尾巴补零编出去。
///
/// 不丢尾巴的话，输出的音频会比源短最多 1023 个采样点（约 23 ms）——
/// 听感上就是最后那一点点被切掉了。
fn flush_aac(
    enc: &mut v2v_aac::Encoder,
    pending: &mut Vec<Vec<f32>>,
    marks: &mut Vec<usize>,
    emit: &mut dyn FnMut(&[u8], u32, usize),
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
    let streams = enc.encode_frames(&[v2v_aac::Frame { planes }])?;
    // 尾巴排在所有样本之后 —— 串行版用 `usize::MAX` 作载荷序号，这里沿用。
    marks.clear();
    emit(&streams[0], AAC_FRAME_SAMPLES as u32, usize::MAX);
    *audio_samples += AAC_FRAME_SAMPLES;
    Ok(())
}

/// AAC 长窗一帧的采样点数。
const AAC_FRAME_SAMPLES: usize = 1024;

/// 默认帧率 —— 推不出来时用（WMV 绝大多数是 25 或 30）。
const DEFAULT_FPS: u32 = 25;

/// 定帧率前预读多少个视频样本。
///
/// 只取前两个不够 —— ASF 里可能有**两条视频轨**，
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
