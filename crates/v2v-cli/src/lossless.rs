//! 无损剪切 / 拼接：**只搬样本，一个字节都不解码**。
//!
//! 这条路的意义在「不重编码」上：MP43 / WMV1 / WMV2 / VC-1 的码流原样搬过去，
//! 画质与源**逐字节相同**，而且快到不需要等编码器。代价是**只能在关键帧上切** ——
//! 从 P 帧起解等于从一个没有参考帧的位置起解，那不是「有损」，是「坏的」。
//!
//! ## 为什么值得做（而不是让人用 `ffmpeg -c copy`）
//!
//! ffmpeg 对 ASF 的 `-c copy` 剪切经常出问题：它按 `-ss` 就近 seek，索引与时间戳
//! 常常对不上（表现为「能打开、拖进度条就乱」「时长比内容长一截」）。这里不猜：
//! 索引由 `AsfMuxer` 按**实际写进去的包**重新算，时间戳从 0 重新起。
//!
//! ## 判据
//!
//! **输出解出的样本序列与源的对应区间逐字节相同**（`tests/lossless.rs`）。不靠 PSNR ——
//! PSNR 只能证明「看起来一样」，逐字节才能证明「真的没动过」。
//!
//! ## 起点为什么**向后**对齐到关键帧
//!
//! `--start 5` 的含义是「5 秒之前的东西不要」。向前对齐（取 5 秒前那个关键帧）会把
//! 用户明确要删掉的内容留在文件里 —— 那是更糟的错。所以对齐到 5 秒之后的**第一个**
//! 关键帧，并把**实际起点**报出来，让人看得见偏移了多少。
//!
//! ## 已知的体量开销（不是载荷问题）
//!
//! 整段复制（`--start 0` 不给 `--end`）出来的文件比源**大 1% 左右**。原因在封装层：
//! 我们的 `AsfMuxer` 把包长固定成 3200 字节，而真样本常用 8000 —— 包越多，包头
//! （每包约 50 字节）摊得越多。实测一份 138 秒的样本：源 18083 个包 / 产物 45711 个包，
//! 差值正好是那 1.1%。**载荷字节完全一致**，播放器不受影响。
//!
//! 修法是让包长跟随源（`min_packet_size` / `max_packet_size`），但那要把封装器里
//! 那个 `const` 变成字段、逐个调用点核对，属于「一次只动一个变量」之外的下一件事。

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use v2v_asf::{AsfDemuxer, AsfMuxer, StreamKind, VideoStreamInfo};
use v2v_container::AudioFormat;

use crate::{write_out, Error};

/// 拼接时「下一段从哪开始」的兜底帧间隔（毫秒）。
///
/// 只在容器声明的时长不可信时用到，见 `concat` 里的说明。
const NOMINAL_FRAME_MS: u64 = 40;

/// 一路输入的格式与流号。剪切 / 拼接都只需要这些。
struct Source {
    /// 视频流号（`Sample::stream` 与它比）。
    vnum: u8,
    /// 音频流号；没有音频轨就 `None`。
    anum: Option<u8>,
    video: VideoStreamInfo,
    /// 编解码器私有数据 —— 无损搬运必须原样带走（VC-1 的序列头就在里面）。
    video_extra: Vec<u8>,
    audio: Option<AudioFormat>,
}

type Demuxer = AsfDemuxer<BufReader<File>>;

fn open_asf(path: &Path) -> Result<Demuxer, Error> {
    AsfDemuxer::new(BufReader::new(File::open(path)?)).map_err(|e| {
        Error::Unsupported(format!(
            "{}：无损剪切 / 拼接目前只认 ASF/WMV —— 这个文件解不成 ASF（{e}）",
            path.display()
        ))
    })
}

fn source_of(d: &Demuxer, path: &Path) -> Result<Source, Error> {
    let v = d
        .streams()
        .iter()
        .find(|s| s.kind == StreamKind::Video)
        .ok_or_else(|| Error::Unsupported(format!("{}：没有视频轨", path.display())))?;
    let vf = v
        .video
        .as_ref()
        .ok_or_else(|| Error::Unsupported(format!("{}：视频轨没有格式信息", path.display())))?;
    let a = d.streams().iter().find(|s| s.kind == StreamKind::Audio);
    Ok(Source {
        vnum: v.number,
        anum: a.map(|s| s.number),
        video: VideoStreamInfo {
            width: vf.width,
            height: vf.height,
            fourcc: vf.fourcc,
        },
        video_extra: vf.extra.clone(),
        audio: a.and_then(|s| s.audio.clone()),
    })
}

/// 按源格式建封装器（视频私有数据一起带上）。
fn muxer_for(src: &Source) -> AsfMuxer {
    let mut m = AsfMuxer::with_audio(src.video.clone(), src.audio.clone());
    m.set_video_extra(src.video_extra.clone());
    m
}

/// 剪切结果。
#[derive(Debug, Clone)]
pub struct TrimReport {
    /// 实际起点（毫秒，相对**源**）。会向后对齐到关键帧，所以 `>= 请求的 start`。
    pub actual_start_ms: u64,
    /// 最后一个写进去的**视频样本**的时间戳（毫秒，相对源）。
    ///
    /// ⚠️ 这是**含**的，不是请求的终点：`--end 15000` 剪出来的内容到 14966 就没了
    /// （那之后没有落在区间内的帧）。别拿它当半开区间的右端。
    pub end_ms: u64,
    pub video_samples: usize,
    pub audio_samples: usize,
    pub key_frames: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
}

/// 按关键帧无损剪切。
///
/// `start_ms` / `end_ms` 都是**相对源**的毫秒；`end_ms == None` 表示剪到结尾。
/// 区间是 `[实际起点, end_ms)` —— 起点会被向后对齐到关键帧。
pub fn trim(
    input: &Path,
    output: &Path,
    start_ms: u64,
    end_ms: Option<u64>,
) -> Result<TrimReport, Error> {
    if let Some(end) = end_ms {
        if end <= start_ms {
            return Err(Error::Unsupported(format!(
                "终点 {end} ms 不在起点 {start_ms} ms 之后"
            )));
        }
    }
    let bytes_in = std::fs::metadata(input)?.len();
    let mut d = open_asf(input)?;
    let src = source_of(&d, input)?;
    let mut mux = muxer_for(&src);

    let (mut actual_start, mut last_video_pts) = (None, 0u64);
    let (mut video_samples, mut audio_samples, mut key_frames) = (0usize, 0usize, 0usize);
    // 两条流各自「越过终点」的标志。都越过了才停 —— 只看视频会切掉尾巴上的音频。
    let (mut v_done, mut a_done) = (false, src.anum.is_none());

    while let Some(s) = d.next_sample()? {
        let pts = s.timestamp_ms.max(0) as u64;

        // ---- 起点：等第一个「时间戳 >= start」的视频关键帧 ----
        let start = match actual_start {
            Some(v) => v,
            None => {
                if s.stream != src.vnum || !s.key_frame || pts < start_ms {
                    continue;
                }
                actual_start = Some(pts);
                pts
            }
        };

        // ---- 终点：越过就丢，两条流都越过才收工 ----
        if let Some(end) = end_ms {
            if pts >= end {
                if s.stream == src.vnum {
                    v_done = true;
                } else if Some(s.stream) == src.anum {
                    a_done = true;
                }
                if v_done && a_done {
                    break;
                }
                continue;
            }
        }

        // 音频可能比视频关键帧早几毫秒（交织顺序如此）—— 起点之前的一律丢，
        // 否则重定时会算出负的时间戳。
        if pts < start {
            continue;
        }
        let rel = u32::try_from(pts - start)
            .map_err(|_| Error::Unsupported(format!("时间戳 {} ms 超出 u32", pts - start)))?;

        if s.stream == src.vnum {
            mux.write_video(&s.data, s.key_frame, rel)?;
            video_samples += 1;
            key_frames += usize::from(s.key_frame);
            last_video_pts = pts - start;
        } else if Some(s.stream) == src.anum {
            mux.write_audio(&s.data, rel)?;
            audio_samples += 1;
        }
        // 别的流（ASF 允许有）不搬 —— 本仓的封装器只写视频 + 音频两条。
    }

    let actual_start = actual_start.ok_or_else(|| {
        Error::Unsupported(format!(
            "{start_ms} ms 之后没有关键帧 —— 整段都在请求的起点之前，或者这段流里没有关键帧"
        ))
    })?;
    if video_samples == 0 {
        return Err(Error::Unsupported("一个视频样本都没剪出来".into()));
    }

    let bytes = mux.finish();
    write_out(output, &bytes)?;
    Ok(TrimReport {
        actual_start_ms: actual_start,
        end_ms: actual_start + last_video_pts,
        video_samples,
        audio_samples,
        key_frames,
        bytes_in,
        bytes_out: bytes.len() as u64,
    })
}

/// 拼接结果。
#[derive(Debug, Clone)]
pub struct ConcatReport {
    pub segments: usize,
    pub video_samples: usize,
    pub audio_samples: usize,
    pub duration_ms: u64,
    pub bytes_out: u64,
}

/// 把多个 ASF/WMV 首尾相接（不重编码）。
///
/// 前置条件全部**明确校验**，不靠「应该没问题」：
/// 1. 每一段的视频四要素（FourCC / 宽 / 高 / 私有数据）必须完全一致；
/// 2. 音频 WAVEFORMATEX 必须完全一致（有一致的不一致就解不出声）；
/// 3. 每一段的**第一个视频样本必须是关键帧** —— 否则接缝处没有参考帧。
pub fn concat(inputs: &[PathBuf], output: &Path) -> Result<ConcatReport, Error> {
    if inputs.len() < 2 {
        return Err(Error::Unsupported(
            "拼接至少要两个文件（一个文件的「拼接」就是剪切）".into(),
        ));
    }

    let first = open_asf(&inputs[0])?;
    let head = source_of(&first, &inputs[0])?;
    drop(first);
    let mut mux = muxer_for(&head);

    let (mut video_samples, mut audio_samples, mut offset) = (0usize, 0usize, 0u64);
    // 真正写进去的最大时间戳 —— 报告里的时长用它，不是「加完最后一段偏移」的那个值
    // （后者比内容长一截，看着像多出来一段）。
    let mut last_written = 0u64;

    for path in inputs {
        let mut d = open_asf(path)?;
        let src = source_of(&d, path)?;
        ensure_compatible(&head, &src, path)?;

        let mut last_video_pts = None;
        let mut seen_video = false;

        while let Some(s) = d.next_sample()? {
            let pts = s.timestamp_ms.max(0) as u64;
            if s.stream == src.vnum && !seen_video {
                seen_video = true;
                if !s.key_frame {
                    return Err(Error::Unsupported(format!(
                        "{}：第一个视频样本不是关键帧 —— 拼接点必须能从关键帧起解，\
                         否则接缝之后一直到下一个关键帧都解不出来",
                        path.display()
                    )));
                }
            }
            let rel = u32::try_from(offset + pts).map_err(|_| {
                Error::Unsupported(format!("拼接后时间戳 {} ms 超出 u32", offset + pts))
            })?;
            if s.stream == src.vnum {
                mux.write_video(&s.data, s.key_frame, rel)?;
                video_samples += 1;
                last_video_pts = Some(pts);
            } else if Some(s.stream) == src.anum {
                mux.write_audio(&s.data, rel)?;
                audio_samples += 1;
            }
            last_written = last_written.max(u64::from(rel));
        }
        if !seen_video {
            return Err(Error::Unsupported(format!(
                "{}：没有视频样本",
                path.display()
            )));
        }

        // 下一段的偏移 = 本段声明的播放时长。
        //
        // ⚠️ 用**容器声明的时长**而不是「最后一帧的 pts」：后者不含最后一帧自己的
        // 显示时长，直接拿它当偏移会让下一段的第一帧和上一段的最后一帧**时间戳重合**。
        // 声明值不可信（为 0 或比内容还短）时退回「最后一帧 + 一个帧间隔」。
        //
        // 那个帧间隔保守取 40 ms（25 fps）：它只影响接缝处的几十毫秒，不影响任何
        // 样本的内容；真正的偏移来自容器声明值。
        let declared = d.info().duration_ms();
        let last = last_video_pts.unwrap_or(0);
        offset += declared.max(last + NOMINAL_FRAME_MS);
    }

    let bytes = mux.finish();
    write_out(output, &bytes)?;
    Ok(ConcatReport {
        segments: inputs.len(),
        video_samples,
        audio_samples,
        duration_ms: last_written,
        bytes_out: bytes.len() as u64,
    })
}

/// 校验两段的格式完全一致，不一致时说清**哪一个字段**。
fn ensure_compatible(head: &Source, cur: &Source, path: &Path) -> Result<(), Error> {
    let p = path.display();
    if head.video.fourcc != cur.video.fourcc {
        return Err(Error::Unsupported(format!(
            "{p}：视频编码是 {}，与第一段的 {} 不同 —— 不同编码器产出的码流不能直接串",
            fourcc_name(cur.video.fourcc),
            fourcc_name(head.video.fourcc)
        )));
    }
    if (head.video.width, head.video.height) != (cur.video.width, cur.video.height) {
        return Err(Error::Unsupported(format!(
            "{p}：尺寸 {}x{} 与第一段的 {}x{} 不同",
            cur.video.width, cur.video.height, head.video.width, head.video.height
        )));
    }
    if head.video_extra != cur.video_extra {
        return Err(Error::Unsupported(format!(
            "{p}：编解码器私有数据（{} 字节）与第一段（{} 字节）不同 —— \
             同一编码格式但参数不同，串起来解不出画面",
            cur.video_extra.len(),
            head.video_extra.len()
        )));
    }
    match (&head.audio, &cur.audio) {
        (None, None) => {}
        (Some(a), Some(b)) if same_audio(a, b) => {}
        (a, b) => {
            return Err(Error::Unsupported(format!(
                "{p}：音频格式与第一段不同（{} / {}）—— 同一个音频流属性没法描述两种音频",
                audio_desc(a.as_ref()),
                audio_desc(b.as_ref())
            )))
        }
    }
    Ok(())
}

fn same_audio(a: &AudioFormat, b: &AudioFormat) -> bool {
    a.format_tag == b.format_tag
        && a.channels == b.channels
        && a.sample_rate == b.sample_rate
        && a.block_align == b.block_align
        && a.bits_per_sample == b.bits_per_sample
        && a.extra == b.extra
}

fn audio_desc(a: Option<&AudioFormat>) -> String {
    match a {
        None => "无音频".into(),
        Some(a) => format!(
            "0x{:04X} {} ch / {} Hz / {} bit",
            a.format_tag, a.channels, a.sample_rate, a.bits_per_sample
        ),
    }
}

fn fourcc_name(f: u32) -> String {
    let b = f.to_le_bytes();
    if b.iter().all(|c| c.is_ascii_graphic() || *c == b' ') {
        format!("'{}'", String::from_utf8_lossy(&b))
    } else {
        format!("0x{f:08X}")
    }
}
