//! 顺时**播放**：把「解封装 → 解码」两层接成一个能逐帧 / 逐样本取数的组件。
//!
//! ## 为什么要有这一层
//!
//! 宿主（EntisGLS Android 移植）要的不是批量转码，而是**流式播放**：
//! open → 逐帧取视频、逐样本取音频 → EOS。现有链路每一层都是好的
//! （解封装流式、VC-1 / WMA 解码逐像素 / 逐样本对账过），但没有「播放」这一层 ——
//! 谁来交织、PTS 从哪来、EOS 怎么报、编码器延迟的静音怎么丢，这些归这里。
//!
//! ## 两条轨各自一条解封装器
//!
//! 视频与音频**各开一个 `AsfDemuxer`**（同一个文件、两个句柄）。理由：
//! 解封装是**顺序**的，单实例没法「先取视频再回头取音频」—— 要么缓存整条轨
//! （138 s 全片进内存，违背验收），要么要求宿主按时间顺序交替拉。
//! 两个句柄之后两条轨完全独立推进，内存各自有界，宿主想先拉谁就拉谁。
//!
//! ## 输出口径
//!
//! - 视频：YUV420P，三平面分开（宿主 1280×720 直接上传纹理）。
//! - 音频：WMA 解码器的**原生输出** —— 每声道一条 `Vec<f32>`，
//!   值域 `[-32768, 32767]`（不是 `[-1, 1)`，**别再乘 32768**，见
//!   `crates/cli/src/lib.rs::wma_frame_to_pcm` 的注释）。要不要交织成 i16 由宿主定。
//! - PTS：视频用容器时间戳（**已扣 preroll**）；音频按**已交出的样本数**推进
//!   （`样本数 × 1000 / 采样率`）—— 这是播放器对 PCM 的标准口径，
//!   也是 ffmpeg 报音频时间戳的算法。
//!
//! ## WMA 的编码器延迟
//!
//! WMA 编码器开头灌了两帧静音，ffmpeg 的解码器用 `skip_samples = frame_len × 2`
//! 丢掉；不跟着丢，音频整体晚约 93 ms（44.1 kHz、帧长 2048），与视频对不上。
//! 这里丢掉的方式与 CLI 一致：**只丢样本、不丢 PTS** —— PTS 按「应该有的」样本序号
//! 走，静音被丢掉之后时间轴才是直的。
//!
//! ## 范围
//!
//! 视频 WMV3（VC-1 Main profile）、音频 WMA v1 / v2 —— 正是宿主要的那两样。
//! 其它 fourcc 明确报「不支持」，不静默跳过。**seek 未做**（见 TODO）。

use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use v2v_asf::{AsfDemuxer, StreamKind};
use v2v_container::VideoFormat;

/// WMV3（VC-1）的 FourCC。只做输入，不做出编码。
const WMV3: u32 = u32::from_le_bytes(*b"WMV3");

/// 播放器自己的错误类型。
///
/// ## 为什么不复用 `v2v_msmpeg4::Error`
///
/// 那个类型已经是「编解码器的通用错误」，mp4v / aac 都在用 —— 但它**没有 `Io`**，
/// 而播放器要自己开文件；asf / wma 又各有一份自己的 `Error`（变体几乎一样但不通
/// 用）。与其往共享类型里加变体（会牵连别的 crate），不如播放器自己收口 ——
/// 到 FFI 那层反正要变成错误码，这里的 `Display` 就是给人看的消息。
#[derive(Debug)]
pub enum Error {
    /// 底层 IO 失败（打开 / 读文件）。
    Io(std::io::Error),
    /// 数据不合法（结构对不上、越界、字段自相矛盾）。
    Bad(String),
    /// 结构合法，但本实现（还）不支持。
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO: {e}"),
            Error::Bad(m) => write!(f, "{m}"),
            Error::Unsupported(m) => write!(f, "不支持：{m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<v2v_asf::Error> for Error {
    fn from(e: v2v_asf::Error) -> Self {
        match e {
            v2v_asf::Error::Io(e) => Error::Io(e),
            v2v_asf::Error::Bad(m) => Error::Bad(m.to_string()),
            v2v_asf::Error::Unsupported(m) => Error::Unsupported(m),
            v2v_asf::Error::Eof => Error::Bad("位流提前结束".into()),
        }
    }
}

impl From<v2v_msmpeg4::Error> for Error {
    fn from(e: v2v_msmpeg4::Error) -> Self {
        match e {
            v2v_msmpeg4::Error::Eof => Error::Bad("位流提前结束".into()),
            v2v_msmpeg4::Error::Bad(m) => Error::Bad(m.to_string()),
            v2v_msmpeg4::Error::Unsupported(m) => Error::Unsupported(m),
        }
    }
}

impl From<v2v_wma::Error> for Error {
    fn from(e: v2v_wma::Error) -> Self {
        match e {
            v2v_wma::Error::Io(e) => Error::Io(e),
            v2v_wma::Error::Bad(m) => Error::Bad(m.to_string()),
            v2v_wma::Error::Unsupported(m) => Error::Unsupported(m),
        }
    }
}

/// 流的静态信息（`v2v_info` 要的东西）。
#[derive(Debug, Clone)]
pub struct Info {
    /// 播放时长（ms，已扣 preroll）。
    pub duration_ms: u64,
    /// 显示宽高（不是编码器声明的那种）。
    pub width: u32,
    pub height: u32,
    pub has_audio: bool,
    /// 视频 FourCC（`WMV3`）。
    pub video_fourcc: u32,
    /// 音频采样率；无音频轨时是 0。
    pub audio_sample_rate: u32,
    /// 音频声道数；无音频轨时是 0。
    pub audio_channels: u16,
}

/// 解出来的一帧视频（YUV420P，三平面分开）。
pub struct VideoFrame {
    /// 容器时间戳（ms，已扣 preroll）。
    pub pts_ms: i64,
    /// 是不是关键帧（seek 的落点候选）。
    pub key_frame: bool,
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// 解出来的一帧音频（WMA 解码器的原生输出）。
pub struct AudioFrame {
    /// 按**已交出**的样本数推出来的（`样本数 × 1000 / 采样率`，ms）。
    ///
    /// 不是容器时间戳：一个 ASF 载荷可能装着好几个 WMA 帧，容器的粒度太粗。
    pub pts_ms: i64,
    pub channels: usize,
    pub sample_rate: u32,
    /// 每声道一条，值域 `[-32768, 32767]`（别再乘 32768）。
    pub planes: Vec<Vec<f32>>,
    /// 每声道的样本数（各平面等长）。
    pub samples: usize,
}

/// 顺时播放器。视频与音频各一条轨，互不阻塞、内存各自有界。
pub struct Player {
    video: VideoTrack,
    audio: Option<AudioTrack>,
    info: Info,
}

/// 视频轨：解封装 + VC-1 解码。
struct VideoTrack {
    demux: AsfDemuxer<BufReader<File>>,
    stream_num: u8,
    decoder: v2v_vc1::Decoder,
    done: bool,
    position_ms: i64,
}

/// 音频轨：解封装 + WMA 解码。
struct AudioTrack {
    demux: AsfDemuxer<BufReader<File>>,
    stream_num: u8,
    decoder: v2v_wma::WmaDecoder,
    channels: usize,
    sample_rate: u32,
    /// 还要丢多少个**每声道样本**（WMA 编码器延迟）。
    skip_left: usize,
    /// 已经交出的每声道样本数（PTS 的基准）。
    emitted: u64,
    done: bool,
    /// 已解出、还没交出的帧。
    ///
    /// ⚠️ **必须有**：一个 ASF 载荷常装着**好几个** WMA 帧（superframe），
    /// `decode_superframe` 返回的是一个列表。一次只交一个、剩下的直接丢，
    /// 音频就只剩四分之一 —— 实测 138 s 的音频只走到 34.8 s（751 帧 / 1538048 样本），
    /// 而且不报任何错，看起来像「音频轨就这么长」。
    pending: std::collections::VecDeque<Vec<Vec<f32>>>,
}

impl Player {
    /// 打开一个 `.wmv`。
    ///
    /// 文件会开**两次**（视频 / 音频各一个解封装实例），理由见模块文档；
    /// 都是 BufReader，不整文件进内存。
    pub fn open(path: &Path) -> Result<Player> {
        let video_demux = open_demux(path)?;
        let info = demux_info(&video_demux)?;

        // ---- 视频解码器（目前只有 WMV3）----
        let vstream = video_demux
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Video)
            .ok_or_else(|| Error::Bad("文件里没有视频轨".into()))?;
        let vfmt: &VideoFormat = vstream.video.as_ref().expect("视频轨必有格式");
        if vfmt.fourcc != WMV3 {
            return Err(Error::Unsupported(format!(
                "视频 fourcc 是 {}，播放器目前只支持 WMV3（VC-1）",
                v2v_container::fourcc_string(vfmt.fourcc)
            )));
        }
        let seq = v2v_vc1::SeqHeader::parse(&vfmt.extra)?;
        let decoder = v2v_vc1::Decoder::new(seq, info.width, info.height)?;
        let video = VideoTrack {
            stream_num: vstream.number,
            decoder,
            demux: video_demux,
            done: false,
            position_ms: 0,
        };

        // ---- 音频轨（有才建；各开一个解封装实例）----
        let audio = if info.has_audio {
            let demux = open_demux(path)?;
            let (num, af) = demux
                .streams()
                .iter()
                .find(|s| s.kind == StreamKind::Audio)
                .and_then(|s| s.audio.as_ref().map(|a| (s.number, a.clone())))
                .ok_or_else(|| Error::Bad("声明了音频轨但拿不到格式".into()))?;
            let version = match af.format_tag {
                0x0160 => 1u8,
                0x0161 => 2u8,
                other => {
                    return Err(Error::Unsupported(format!(
                        "音频 format_tag 0x{other:04X}，播放器目前只支持 WMA（0x0160/0x0161）"
                    )))
                }
            };
            let params = v2v_wma::WmaParams::new(
                version,
                af.channels as usize,
                af.sample_rate,
                af.avg_bytes_per_sec * 8,
                af.block_align as usize,
                &af.extra,
            )?;
            let decoder = v2v_wma::WmaDecoder::new(params.clone())?;
            Some(AudioTrack {
                stream_num: num,
                decoder,
                demux,
                channels: af.channels as usize,
                sample_rate: af.sample_rate,
                skip_left: params.frame_len * 2,
                emitted: 0,
                done: false,
                pending: std::collections::VecDeque::new(),
            })
        } else {
            None
        };

        Ok(Player { video, audio, info })
    }

    /// 静态信息。
    pub fn info(&self) -> &Info {
        &self.info
    }

    /// 下一帧视频；`None` = 视频轨到头（EOS）。
    pub fn next_video(&mut self) -> Result<Option<VideoFrame>> {
        self.video.next()
    }

    /// 下一帧音频；`None` = 音频轨到头（或本来就没有音频）。
    pub fn next_audio(&mut self) -> Result<Option<AudioFrame>> {
        match self.audio.as_mut() {
            Some(t) => t.next(),
            None => Ok(None),
        }
    }

    /// 已交出的最后一帧视频的时间戳（ms）。没取过帧时是 0。
    pub fn position_ms(&self) -> i64 {
        self.video.position_ms
    }
}

impl VideoTrack {
    fn next(&mut self) -> Result<Option<VideoFrame>> {
        if self.done {
            return Ok(None);
        }
        loop {
            match self.demux.next_sample().map_err(Error::from)? {
                None => {
                    self.done = true;
                    return Ok(None);
                }
                // 别的轨的载荷直接跳过（本轨用自己的解封装实例，只等自己这条）。
                Some(s) if s.stream != self.stream_num => continue,
                Some(s) => {
                    self.decoder.decode_frame(&s.data)?;
                    let f = self.decoder.frame();
                    let out = VideoFrame {
                        pts_ms: s.timestamp_ms,
                        key_frame: s.key_frame,
                        width: f.width,
                        height: f.height,
                        y: f.y.clone(),
                        u: f.u.clone(),
                        v: f.v.clone(),
                    };
                    self.position_ms = out.pts_ms;
                    return Ok(Some(out));
                }
            }
        }
    }
}

impl AudioTrack {
    fn next(&mut self) -> Result<Option<AudioFrame>> {
        loop {
            // ---- 先交缓存里的 ----
            if let Some(mut planes) = self.pending.pop_front() {
                let n = planes.first().map_or(0, |p| p.len());
                // 编码器延迟：丢样本，**不**丢 PTS（时间轴才直）。
                if self.skip_left > 0 && n > 0 {
                    let drop = self.skip_left.min(n);
                    for p in &mut planes {
                        p.drain(..drop);
                    }
                    self.skip_left -= drop;
                }
                let kept = planes.first().map_or(0, |p| p.len());
                if kept == 0 {
                    continue;
                }
                let pts =
                    (self.emitted as f64 * 1000.0 / self.sample_rate as f64).round() as i64;
                self.emitted += kept as u64;
                return Ok(Some(AudioFrame {
                    pts_ms: pts,
                    channels: self.channels,
                    sample_rate: self.sample_rate,
                    planes,
                    samples: kept,
                }));
            }
            if self.done {
                return Ok(None);
            }
            // ---- 缓存空了才读下一个载荷 ----
            match self.demux.next_sample().map_err(Error::from)? {
                None => {
                    self.done = true;
                    return Ok(None);
                }
                // 别的轨的载荷直接跳过。
                Some(s) if s.stream != self.stream_num => continue,
                Some(s) => {
                    for f in self.decoder.decode_superframe(&s.data)? {
                        self.pending.push_back(f);
                    }
                }
            }
        }
    }
}

/// 打开一个解封装实例（同一文件会被开两次，见模块文档）。
fn open_demux(path: &Path) -> Result<AsfDemuxer<BufReader<File>>> {
    let file = File::open(path).map_err(|e| Error::Bad(format!("打不开 {}: {e}", path.display())))?;
    AsfDemuxer::new(BufReader::new(file)).map_err(Error::from)
}

fn demux_info(demux: &AsfDemuxer<BufReader<File>>) -> Result<Info> {
    let duration_ms = demux.info().duration_ms();
    let vstream = demux
        .streams()
        .iter()
        .find(|s| s.kind == StreamKind::Video)
        .ok_or_else(|| Error::Bad("文件里没有视频轨".into()))?;
    let vfmt = vstream.video.as_ref().expect("视频轨必有格式");
    let audio = demux
        .streams()
        .iter()
        .find(|s| s.kind == StreamKind::Audio)
        .and_then(|s| s.audio.as_ref());
    Ok(Info {
        duration_ms,
        width: vfmt.width,
        height: vfmt.height,
        has_audio: audio.is_some(),
        video_fourcc: vfmt.fourcc,
        audio_sample_rate: audio.map_or(0, |a| a.sample_rate),
        audio_channels: audio.map_or(0, |a| a.channels),
    })
}

