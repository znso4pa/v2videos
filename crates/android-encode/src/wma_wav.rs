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
//!
//! ## 与 ffmpeg 参考解码对拍（两个真实样本）
//!
//! 内容**逐样本一致**：对齐后差异 RMS 0.71（约 -80 dB 相对信号），最大差
//! 1 个 LSB（编码器输出的取整方向差一步）。唯一区别是开头比 ffmpeg 多出
//! **固定的 8192 个样本**（与采样率无关）—— ffmpeg 会补偿 ASF preroll，
//! 而这里从第一块 PCM 起输出，与引擎自身音频路径、本仓转码路径（音频都
//! 从第一块 PCM 起计时，见 `transcode`）约定一致：引擎拿这份 WAV 换掉
//! 自己的解码器，A/V 同步关系不变；照 ffmpeg 裁掉反而会提前 93 ms。

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
    let byte_rate = af.sample_rate as u64 * ch as u64 * 2;

    // 直接往最终缓冲里写，不先攒 `Vec<i16>` 再拷贝 —— 峰值内存从「2 倍数据量」
    // 降到「1 倍」。手机上一次抽取 2 分钟 OP 约 23 MB（原先 46 MB）。
    //
    // 预分配用 ASF 头里的播放时长估：留 1 秒余量吸收块对齐误差；超过 64 MB 就
    // 不再一次性预占（长片用增量增长，宁可慢一点也不赌分配成功）。
    let est = (demux.info().duration_ms() as u64 * byte_rate / 1000) as usize;
    let mut out: Vec<u8> = Vec::with_capacity(44 + est.min(64 << 20) + byte_rate as usize);
    out.resize(44, 0); // 先占位，末尾回填头

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
            out.reserve(n * ch * 2);
            for i in 0..n {
                for plane in &frame {
                    // 解码器输出的值域就是 `[-32768, 32767]` 的 f32（见 `AudioFrame`），
                    // 所以这里是「取整 + 夹紧」，不是缩放。
                    let v = plane[i].round().clamp(-32768.0, 32767.0) as i16;
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }

    if out.len() <= 44 {
        return Err(Error::Bad("WMA 音轨解出来是空的".into()));
    }
    let data_size = (out.len() - 44) as u32;
    out[..44].copy_from_slice(&wav_header(af.channels, af.sample_rate, data_size));
    Ok(out)
}

/// 44 字节 RIFF/WAVE 头（`fmt ` PCM + `data`），16 bit 小端。
///
/// 与常见引擎自己的 WAV 实现逐字段一致 —— 调用方不需要知道这个 WAV 是谁产的。
fn wav_header(channels: u16, sample_rate: u32, data_size: u32) -> [u8; 44] {
    const BYTES_PER_SAMPLE: u16 = 2;
    let block_align = channels.saturating_mul(BYTES_PER_SAMPLE);
    let byte_rate = sample_rate.saturating_mul(u32::from(block_align));

    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&(36u32.saturating_add(data_size)).to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");

    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes()); // fmt chunk 长度
    h[20..22].copy_from_slice(&1u16.to_le_bytes()); // audio_format = PCM
    h[22..24].copy_from_slice(&channels.to_le_bytes());
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    h[32..34].copy_from_slice(&block_align.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits_per_sample

    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_size.to_le_bytes());
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16_at(h: &[u8; 44], i: usize) -> u16 {
        u16::from_le_bytes([h[i], h[i + 1]])
    }

    fn u32_at(h: &[u8; 44], i: usize) -> u32 {
        u32::from_le_bytes([h[i], h[i + 1], h[i + 2], h[i + 3]])
    }

    /// 44 字节头必须逐字段正确 —— 单声道/48k 没有真实样本可对拍，只能靠这个兜住。
    #[test]
    fn wav_header_layout() {
        let h = wav_header(2, 44_100, 1000);
        assert_eq!(&h[0..4], b"RIFF");
        assert_eq!(u32_at(&h, 4), 36 + 1000); // RIFF chunk size
        assert_eq!(&h[8..12], b"WAVE");
        assert_eq!(&h[12..16], b"fmt ");
        assert_eq!(u32_at(&h, 16), 16); // fmt chunk size
        assert_eq!(u16_at(&h, 20), 1); // PCM
        assert_eq!(u16_at(&h, 22), 2); // 声道
        assert_eq!(u32_at(&h, 24), 44_100);
        assert_eq!(u32_at(&h, 28), 44_100 * 4); // byte rate
        assert_eq!(u16_at(&h, 32), 4); // block align
        assert_eq!(u16_at(&h, 34), 16); // bits per sample
        assert_eq!(&h[36..40], b"data");
        assert_eq!(u32_at(&h, 40), 1000);

        // 单声道 48 kHz：block_align / byte_rate 都要跟着变
        let h = wav_header(1, 48_000, 2000);
        assert_eq!(u16_at(&h, 22), 1);
        assert_eq!(u32_at(&h, 28), 48_000 * 2);
        assert_eq!(u16_at(&h, 32), 2);
        assert_eq!(u32_at(&h, 40), 2000);
    }
}
