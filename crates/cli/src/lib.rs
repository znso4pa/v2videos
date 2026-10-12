//! v2video 的转码核心：**解封装 → 解码 → 编码 → 封装**。
//!
//! 命令行（`main.rs`）只是它的壳；这里能被测试直接调用，也能被别的程序当库用。
//!
//! ## 现状
//!
//! - 输入：**ASF（`.wmv`）/ AVI（`.avi`）** 里的 **MP43** 视频轨 —— 容器按魔数自动认。
//!   别的容器 / codec 会明确报 `Unsupported`，不猜、不静默跳过。
//! - 输出：**MP43 in ASF**（`.wmv`）。
//! - 音频轨：目前**不搬**（WMA 解/编还没做），转换时跳过并计数，报告里说清楚。

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

pub use v2v_asf::{AsfDemuxer, AsfMuxer, StreamKind, VideoStreamInfo};
pub use v2v_avi::AviDemuxer;
pub use v2v_container::Sample;
pub use v2v_msmpeg4::{Decoder, Encoder, EncoderConfig, Frame};

/// MP43 的 FourCC（本工具唯一的输出编码）。
pub const MP43: u32 = u32::from_le_bytes(*b"MP43");

/// 转码参数。
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// 目标 qscale（1..=31）。越大越小越糊。编码器必要时会自己往上抬。
    pub qscale: u8,
    /// 周期 I 帧间隔（帧数）。`0` = 只首帧是 I。
    pub keyframe_interval: usize,
    /// 转完把输出**解码回来**与源帧比 PSNR（慢一倍，但能证明「转出来的东西真能放」）。
    pub verify: bool,
    /// 最多转多少帧（`None` = 全部）。做局部转换/预览用 —— 也让真样本的
    /// 端到端验证能在几秒内跑完。
    pub max_frames: Option<usize>,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            qscale: 6,
            keyframe_interval: 250,
            verify: true,
            max_frames: None,
        }
    }
}

/// 一次转换的结果。
#[derive(Debug, Clone)]
pub struct ConvertReport {
    pub frames: usize,
    /// 关键帧数（含首帧）。
    pub key_frames: usize,
    /// 跳过的音频样本数（音频还没做）。
    pub skipped_audio: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
    /// `verify` 打开时的平均 PSNR（dB，YUV 三平面合计）。
    pub psnr_db: Option<f64>,
}

/// 转码错误。
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Asf(v2v_asf::Error),
    Msmpeg4(v2v_msmpeg4::Error),
    Avi(v2v_avi::Error),
    Unsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "IO 错误：{e}"),
            Error::Asf(e) => write!(f, "容器错误：{e}"),
            Error::Msmpeg4(e) => write!(f, "编解码错误：{e}"),
            Error::Avi(e) => write!(f, "AVI 错误：{e}"),
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

/// 「能吐视频样本的东西」—— ASF / AVI 解封装器都实现它。
///
/// 存在的理由：转码管线只关心「视频流是谁、多大、什么 codec」和「下一个样本」，
/// 不该为每种容器写一份管线。加容器 = 加一个实现，管线不动。
trait VideoSource {
    /// `(流号, 宽, 高, fourcc)`；没有视频轨就 `None`。
    fn video_stream(&self) -> Option<(u8, u32, u32, u32)>;
    fn next(&mut self) -> Result<Option<Sample>, Error>;
}

struct AsfSource(v2v_asf::AsfDemuxer<BufReader<File>>);

impl VideoSource for AsfSource {
    fn video_stream(&self) -> Option<(u8, u32, u32, u32)> {
        let v = self.0.streams().iter().find(|s| s.kind == StreamKind::Video)?;
        let vf = v.video.as_ref()?;
        Some((v.number, vf.width, vf.height, vf.fourcc))
    }
    fn next(&mut self) -> Result<Option<Sample>, Error> {
        Ok(self.0.next_sample()?)
    }
}

struct AviSource(v2v_avi::AviDemuxer<BufReader<File>>);

impl VideoSource for AviSource {
    fn video_stream(&self) -> Option<(u8, u32, u32, u32)> {
        let v = self.0.streams().iter().find(|s| s.kind == StreamKind::Video)?;
        let vf = v.video.as_ref()?;
        Some((v.number, vf.width, vf.height, vf.fourcc))
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

/// 把一个文件转成 `.wmv`（MP43 in ASF）。
///
/// 逐帧流式：读一帧 → 解码 → 编码 → 写进封装器（封装器内部缓冲，最后一次性落盘）。
pub fn convert_file(
    input: &Path,
    output: &Path,
    opts: &ConvertOptions,
) -> Result<ConvertReport, Error> {
    if !(1..=31).contains(&opts.qscale) {
        return Err(Error::Unsupported(format!("qscale {} 不在 1..=31", opts.qscale)));
    }
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
    let mut mux = AsfMuxer::new(VideoStreamInfo {
        width: w,
        height: h,
        fourcc: MP43,
    });
    // 校验用的第二个解码器（与解码源帧的那个分开 —— 状态独立）。
    let mut vdec = if opts.verify { Some(Decoder::new()) } else { None };

    let mut frames = 0usize;
    let mut key_frames = 0usize;
    let mut skipped_audio = 0usize;
    let mut sq_err = 0f64;
    let mut n_px = 0u64;

    while let Some(s) = demux.next()? {
        if s.stream != vnum {
            skipped_audio += 1;
            continue;
        }
        let pts = u32::try_from(s.timestamp_ms.max(0)).map_err(|_| {
            Error::Unsupported(format!("时间戳 {} ms 超出 u32", s.timestamp_ms))
        })?;

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
    let mut f = BufWriter::new(File::create(output)?);
    f.write_all(&wmv)?;
    f.flush()?;

    let psnr_db = if opts.verify && n_px > 0 {
        let mse = sq_err / n_px as f64;
        Some(if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        })
    } else {
        None
    };

    Ok(ConvertReport {
        frames,
        key_frames,
        skipped_audio,
        bytes_in,
        bytes_out: wmv.len() as u64,
        psnr_db,
    })
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
            let out = output_path_for(input, out_dir);
            let r = convert_file(input, &out, opts).map(|rep| (out, rep));
            (input.clone(), r)
        })
        .collect()
}

/// 输出路径：`<out_dir>/<输入文件名去扩展>.wmv`；与输入同路径时加 `.conv` 后缀。
pub fn output_path_for(input: &Path, out_dir: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "out".into());
    let mut out = out_dir.join(format!("{stem}.wmv"));
    if out == input {
        out = out_dir.join(format!("{stem}.conv.wmv"));
    }
    out
}
