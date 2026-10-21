//! 把 WMV / ASF 里的 **WMA 音轨**整段解成 PCM16LE 的 WAV 字节。
//!
//! 存在的理由很具体：有些引擎自带的 WMA 解码器没做 FFT 加速（朴素 MDCT 实现），
//! 实测比实时慢 5.3×——一段 124 s 的片头要解 11 分钟，于是
//! 「WMV 能播画面但没声音」，只能靠外部工具预先把整个视频转成 MP4 绕过去。
//! 这里的解码器在同一个样本上是 **110× 实时**（见 `cargo run -p v2v-player
//! --example bench -- <file> --audio-only`），差了近三个数量级。
//!
//! 所以这条路比「转码」省得多：**不解视频、不写缓存、不重编码**，只把音轨
//! 解出来交给引擎——WMV 原样播放，声音就回来了。
//!
//! 输出是标准的 RIFF/WAVE（fmt PCM、16 bit 小端、声道交织），字段布局与常见
//! 引擎自己的实现一致，调用方拿到字节直接当「解好的 WAV」用即可。

use std::io::BufReader;
use std::path::Path;

use v2v_asf::{AsfDemuxer, StreamKind};
use v2v_player::{Error, Result};

/// WMA v1 / v2 的 `wFormatTag`（WMA Pro 是 0x0162，不支持）。
const WMA_V1: u16 = 0x0160;
const WMA_V2: u16 = 0x0161;

/// 这个文件里有没有我们能解的 WMA 音轨。
///
/// 只读头，不解码——给调用方做「值不值得走这条路」的快速判断。
pub fn has_wma_audio(input: &Path) -> Result<bool> {
    let file = std::fs::File::open(input).map_err(Error::Io)?;
    let demux = AsfDemuxer::new(BufReader::new(file))?;
    Ok(demux
        .streams()
        .iter()
        .any(|s| s.kind == StreamKind::Audio && is_wma(s.audio.as_ref().map(|a| a.format_tag))))
}

fn is_wma(tag: Option<u16>) -> bool {
    matches!(tag, Some(WMA_V1) | Some(WMA_V2))
}

/// 把整条 WMA 音轨解成 WAV（PCM16LE、交织）字节。
///
/// 没有 WMA 音轨时返回 [`Error::Unsupported`]——调用方据此退回自己的路径，
/// 别把「这文件本来就没声音」当成解码失败。
pub fn wma_to_wav_bytes(input: &Path) -> Result<Vec<u8>> {
    let file = std::fs::File::open(input).map_err(Error::Io)?;
    let mut demux = AsfDemuxer::new(BufReader::new(file))?;

    // 先把要用的值抄出来（`streams()` 是借用，不能拖过整个函数）。
    let (anum, af) = {
        let st = demux.streams();
        let a = st
            .iter()
            .find(|s| s.kind == StreamKind::Audio && is_wma(s.audio.as_ref().map(|f| f.format_tag)))
            .ok_or_else(|| Error::Unsupported("这个文件没有可解的 WMA 音轨".into()))?;
        (
            a.number,
            a.audio.as_ref().expect("音频轨必有格式").clone(),
        )
    };

    let version = if af.format_tag == WMA_V1 { 1 } else { 2 };
    let params = v2v_wma::WmaParams::new(
        version,
        af.channels as usize,
        af.sample_rate,
        af.avg_bytes_per_sec.saturating_mul(8),
        af.block_align as usize,
        &af.extra,
    )?;
    let mut dec = v2v_wma::WmaDecoder::new(params)?;

    let ch = af.channels as usize;
    let mut pcm: Vec<i16> = Vec::new();
    loop {
        let s = match demux.next_sample()? {
            Some(s) => s,
            None => break,
        };
        if s.stream != anum {
            continue;
        }
        for frame in dec.decode_superframe(&s.data)? {
            // 各声道等长；交织成 L R L R …（与 WAV 的 data 布局一致）。
            let n = frame.first().map_or(0, |p| p.len());
            pcm.reserve(n * ch);
            for i in 0..n {
                for plane in &frame {
                    // 解码器输出的值域就是 `[-32768, 32767]` 的 f32（见 `AudioFrame`），
                    // 所以这里是「取整 + 夹紧」，不是缩放。
                    let v = plane[i].round().clamp(-32768.0, 32767.0) as i16;
                    pcm.push(v);
                }
            }
        }
    }

    if pcm.is_empty() {
        return Err(Error::Bad("WMA 音轨解出来是空的".into()));
    }
    Ok(wav_pcm16le(&pcm, af.channels, af.sample_rate))
}

/// RIFF/WAVE + `fmt `(PCM) + `data`，16 bit 小端交织。
///
/// 与常见引擎自己的 WAV 实现逐字段一致 —— 调用方不需要知道这个 WAV 是谁产的。
fn wav_pcm16le(samples: &[i16], channels: u16, sample_rate: u32) -> Vec<u8> {
    const BYTES_PER_SAMPLE: u16 = 2;
    let block_align = channels.saturating_mul(BYTES_PER_SAMPLE);
    let byte_rate = sample_rate.saturating_mul(u32::from(block_align));
    let data_size = (samples.len() as u32).saturating_mul(2);

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36u32.saturating_add(data_size)).to_le_bytes());
    out.extend_from_slice(b"WAVE");

    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk 长度
    out.extend_from_slice(&1u16.to_le_bytes()); // audio_format = PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes()); // bits_per_sample

    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}
