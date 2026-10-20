//! 端到端管线：**WMV → MP4，解码用我们的（对 ffmpeg 逐字节一致），视频用平台
//! 硬件编码器，封装用 v2v-mp4** —— 「取优点」的完整形态，示例与测试共用。
//!
//! 后端按编译目标选：
//! - **macOS**：VideoToolbox（`c/vt_encoder.c`，产长度前缀 NALU + avcC，
//!   走 `v2v_mp4::VideoCodec::Avc1`）；
//! - **Android**：MediaCodec（`c/media_encoder.c`，自带 MediaMuxer）；
//! - 其它：`Unsupported`。
//!
//! 音频两套后端一致：WMA → f32 → 我们的 AAC-LC → `esds` 轨。

use std::path::Path;

use v2v_aac::{Encoder as AacEncoder, EncoderConfig as AacConfig, Frame as AacFrame};
use v2v_player::{Error, Player, Result};

/// `v2v_mp4` 的错误只有 `&'static str` 一种 —— 转成带上下文的字符串。
#[cfg(any(target_os = "macos", target_os = "android"))]
fn mux_err(e: v2v_mp4::Error) -> Error {
    Error::Bad(format!("mp4 mux: {e}"))
}

/// 一次转码的结果。
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub frames: u64,
    /// 音频段数（每段 1024 个采样点）。
    pub audio_chunks: u64,
    pub output_bytes: u64,
}

/// 用平台硬件编码器把一个 WMV 转成 MP4。
pub fn wmv_to_mp4_hw(
    input: &Path,
    output: &Path,
    video_bitrate: u32,
    audio_bitrate: u32,
) -> Result<Stats> {
    let mut player = Player::open(input)?;
    let info = player.info().clone();

    let mut aac = info.has_audio.then(|| {
        AacEncoder::new(AacConfig {
            channels: info.audio_channels as u8,
            sample_rate: info.audio_sample_rate,
            bitrate: audio_bitrate,
            ..AacConfig::default()
        })
        .expect("建 AAC 编码器")
    });
    let asc = aac
        .as_ref()
        .map(|e| e.audio_specific_config().to_bytes());
    let ch = info.audio_channels as usize;
    // AAC 编码器的样本缓冲：**每声道一条长缓冲**（第一版把 planes 直接 extend
    // 进一个 Vec，段数无限增长，AAC 编码器一碰就炸 —— 详见 examples 的注释）。
    let mut pcm_buf: Vec<Vec<f32>> = vec![Vec::new(); ch];
    let mut stats = Stats {
        frames: 0,
        audio_chunks: 0,
        output_bytes: 0,
    };

    // 帧率从时间戳推（两个后端都只要整数）。用一次性的 Player 探，别动主 player。
    let fps = infer_fps(input);

    #[cfg(target_os = "macos")]
    {
        use crate::vt::VideoToolboxEncoder;
        use v2v_mp4::{Mp4Muxer, VideoCodec, VideoTrack};

        const TIMESCALE: u32 = 1000; // 视频轨时基：毫秒 × 1000 = 微秒，与 pts 同单位

        let mut enc =
            VideoToolboxEncoder::open(info.width, info.height, video_bitrate, fps, 4)?;

        // avcC 要等第一帧的**异步回调**送达（实测单帧样本上主循环结束时才到）
        // —— 统一走「pending → avcC 到位即建 muxer → 冲刷」，三处共用。
        let mut pending: Vec<(crate::vt::VtSample, u32)> = Vec::new();
        let mut mux: Option<Mp4Muxer> = None;
        let mut last_pts: Option<i64> = None;
        let mut audio_pending: Vec<Vec<u8>> = Vec::new();

        macro_rules! absorb {
            ($samples:expr) => {{
                for s in $samples {
                    let dur = last_pts.map_or(33, |p| (s.pts_us / 1000 - p).max(1) as u32);
                    last_pts = Some(s.pts_us / 1000);
                    pending.push((s, dur));
                }
                if mux.is_none() {
                    if let Some(avcc) = enc.avcc() {
                        let vt = VideoTrack {
                            width: info.width as u16,
                            height: info.height as u16,
                            timescale: TIMESCALE,
                            codec: VideoCodec::Avc1 {
                                avc_config: avcc.to_vec(),
                            },
                            avg_bitrate: video_bitrate,
                        };
                        mux = Some(match &asc {
                            Some(asc) => Mp4Muxer::with_audio(
                                vt,
                                v2v_mp4::AudioTrack {
                                    channels: info.audio_channels,
                                    sample_rate: info.audio_sample_rate,
                                    timescale: info.audio_sample_rate,
                                    codec: v2v_mp4::AudioCodec::Es {
                                        object_type: 0x40, // AAC-LC
                                        decoder_specific_info: asc.clone(),
                                    },
                                    avg_bitrate: audio_bitrate,
                                    // 我们的 AAC 有 2048 采样点的编码器延迟，
                                    // muxer 写进 edts/elst（与 CLI 口径一致）。
                                    priming: 2048,
                                },
                            ),
                            None => Mp4Muxer::new(vt),
                        });
                    }
                }
                if let Some(m) = &mut mux {
                    for (s, dur) in pending.drain(..) {
                        m.write_video(&s.data, s.key_frame, dur).map_err(mux_err)?;
                    }
                    for bytes in audio_pending.drain(..) {
                        m.write_audio(&bytes, 1024).map_err(mux_err)?;
                        stats.audio_chunks += 1;
                    }
                }
            }};
        }

        macro_rules! encode_audio_chunk {
            () => {{
                loop {
                    if pcm_buf.first().map_or(0, |p| p.len()) < 1024 {
                        break;
                    }
                    let mut planes = Vec::with_capacity(ch);
                    for p in &mut pcm_buf {
                        planes.push(p.drain(..1024).collect::<Vec<_>>());
                    }
                    let bytes = aac
                        .as_mut()
                        .expect("有音频就该有编码器")
                        .encode_frame(&AacFrame { planes })
                        .map_err(|e| Error::Bad(format!("AAC: {e}")))?;
                    if mux.is_some() {
                        mux.as_mut()
                            .unwrap()
                            .write_audio(&bytes, 1024)
                            .map_err(mux_err)?;
                        stats.audio_chunks += 1;
                    } else {
                        audio_pending.push(bytes);
                    }
                }
            }};
        }

        loop {
            let mut got = false;
            if let Some(f) = player.next_video()? {
                got = true;
                stats.frames += 1;
                let out = enc.encode(&f.y, &f.u, &f.v, f.pts_ms * 1000, false)?;
                absorb!(out);
            }
            if let Some(a) = player.next_audio()? {
                got = true;
                for (dst, src) in pcm_buf.iter_mut().zip(a.planes) {
                    dst.extend(src);
                }
                encode_audio_chunk!();
            }
            if !got {
                break;
            }
        }

        // EOS：吸干在途样本与尾段音频，再统一冲一遍。
        let out = enc.finish()?;
        absorb!(out);
        encode_audio_chunk!();

        let m = mux.ok_or_else(|| Error::Bad("一帧视频都没编出来（avcC 没拿到）".into()))?;
        let mp4 = m.finish();
        std::fs::write(output, &mp4).map_err(Error::Io)?;
        stats.output_bytes = mp4.len() as u64;
    }

    #[cfg(target_os = "android")]
    {
        use crate::Mp4HwWriter;
        let mut hw = Mp4HwWriter::open(output, info.width, info.height, video_bitrate, fps, 4, info.has_audio)?;
        if let (Some(enc), true) = (&aac, info.has_audio) {
            hw.add_audio_track(
                &enc.audio_specific_config().to_bytes(),
                info.audio_sample_rate,
                info.audio_channels as u8,
            )?;
        }
        let sample_rate = u64::from(info.audio_sample_rate);
        let mut pcm_emitted = 0u64;
        loop {
            let mut got = false;
            if let Some(f) = player.next_video()? {
                got = true;
                stats.frames += 1;
                hw.write_video(&f.y, &f.u, &f.v, f.pts_ms * 1000, false)?;
            }
            if let Some(enc) = &mut aac {
                if let Some(a) = player.next_audio()? {
                    got = true;
                    for (dst, src) in pcm_buf.iter_mut().zip(a.planes) {
                        dst.extend(src);
                    }
                    loop {
                        if pcm_buf.first().map_or(0, |p| p.len()) < 1024 {
                            break;
                        }
                        let mut planes = Vec::with_capacity(ch);
                        for p in &mut pcm_buf {
                            planes.push(p.drain(..1024).collect::<Vec<_>>());
                        }
                        let pts_us = (pcm_emitted * 1_000_000 / sample_rate) as i64;
                        pcm_emitted += 1024;
                        let bytes = enc.encode_frame(&AacFrame { planes }).map_err(Error::from)?;
                        hw.write_aac(&bytes, pts_us)?;
                        stats.audio_chunks += 1;
                    }
                }
            }
            if !got {
                break;
            }
        }
        hw.finish()?;
        stats.output_bytes = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    }

    #[cfg(not(any(target_os = "macos", target_os = "android")))]
    {
        let _ = (&aac, &asc, &mut pcm_buf, &ch, &mut stats, output);
        return Err(Error::Unsupported(
            "硬件编码后端只有 macOS（VideoToolbox）与 Android（MediaCodec）".into(),
        ));
    }

    Ok(stats)
}

/// 从前 100 帧的时间戳差推帧率（容错：时间戳可能不从 0 开始）。
/// 用**一次性的 Player** 探 —— 不能用主 player：它会把前 100 帧消费掉。
fn infer_fps(path: &Path) -> u32 {
    let Ok(mut p) = Player::open(path) else {
        return 30;
    };
    let mut ts = Vec::new();
    for _ in 0..100 {
        match p.next_video().map_err(Error::from) {
            Ok(Some(f)) => ts.push(f.pts_ms),
            Ok(None) => break,
            Err(_) => return 30,
        }
    }
    if ts.len() >= 2 {
        let span = *ts.last().unwrap() - ts[0];
        let steps = ts.len() - 1;
        if span > 0 {
            return ((steps as f64 * 1000.0 / span as f64).round() as u32).max(1);
        }
    }
    30
}
