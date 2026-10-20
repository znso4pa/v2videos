//! 端到端管线：**WMV → MP4，解码用我们的（对 ffmpeg 逐字节一致），视频用平台
//! 硬件编码器，封装用 v2v-mp4** —— 「取优点」的完整形态，示例与测试共用。
//!
//! 后端按编译目标选：
//! - **macOS**：VideoToolbox（`c/vt_encoder.c`，产长度前缀 NALU + avcC，
//!   走 `v2v_mp4::VideoCodec::Avc1`）。**GOP 并行解码**（VC-1 从任意 I 帧起解
//!   与整链解到那里逐像素一致，gop_start.rs 验过）+ 音频独立线程；
//! - **Android**：MediaCodec（`c/media_encoder.c`，自带 MediaMuxer），顺序版；
//! - 其它：`Unsupported`。
//!
//! 音频两套后端一致：WMA → f32 → 我们的 AAC-LC → `esds` 轨。

use std::path::Path;

#[cfg(any(target_os = "macos", target_os = "android"))]
use v2v_aac::{Encoder as AacEncoder, EncoderConfig as AacConfig, Frame as AacFrame};
use v2v_player::{Error, Result};

#[cfg(target_os = "macos")]
fn mux_err(e: v2v_mp4::Error) -> Error {
    Error::Bad(format!("mp4 mux: {e}"))
}

#[cfg(target_os = "android")]
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
    let profile = std::env::var("V2V_HW_PROFILE").is_ok();
    let t_start = std::time::Instant::now();
    let mut stats = Stats {
        frames: 0,
        audio_chunks: 0,
        output_bytes: 0,
    };

    #[cfg(target_os = "macos")]
    let mut encoded_samples: Vec<crate::vt::VtSample> = Vec::new();
    #[cfg(target_os = "macos")]
    {
        use crate::vt::VideoToolboxEncoder;
        use std::io::BufReader;
        use std::sync::{Arc, Mutex};
        use v2v_asf::{AsfDemuxer, StreamKind};
        use v2v_mp4::{Mp4Muxer, VideoCodec, VideoTrack};

        const TIMESCALE: u32 = 1000; // 视频轨时基：毫秒 × 1000 = 微秒，与 pts 同单位

        // ---- 解封装：一趟拿齐流信息、按 key_frame 切视频 GOP、音频载荷喂给音频线程 ----
        let file = std::fs::File::open(input).map_err(Error::Io)?;
        let mut demux = AsfDemuxer::new(BufReader::new(file)).map_err(Error::from)?;
        // 立刻把要用的值从 `streams` 里抄出来（避免借用拖过整个函数）。
        let (vnum, seq, vw, vh) = {
            let st = demux.streams();
            let v = st
                .iter()
                .find(|s| s.kind == StreamKind::Video)
                .ok_or_else(|| Error::Bad("没有视频轨".into()))?;
            let vf = v.video.as_ref().expect("视频轨必有格式");
            if vf.fourcc != u32::from_le_bytes(*b"WMV3") {
                return Err(Error::Unsupported(format!(
                    "视频 fourcc 0x{:08X}，硬件路径目前只支持 WMV3",
                    vf.fourcc
                )));
            }
            (
                v.number,
                v2v_vc1::SeqHeader::parse(&vf.extra)?,
                vf.width,
                vf.height,
            )
        };
        let (anum, af) = {
            let st = demux.streams();
            match st.iter().find(|s| s.kind == StreamKind::Audio) {
                Some(a) => (
                    Some(a.number),
                    Some(a.audio.as_ref().expect("音频轨必有格式").clone()),
                ),
                None => (None, None),
            }
        };
        let asc = af.as_ref().map(|_| {
            // AudioSpecificConfig：AAC-LC(2) + 采样率索引 + 声道数。
            // 真正的字节由下面的音频线程按参数生成（这里先占位拿声道数）。
            af.as_ref().unwrap();
            v2v_aac::AudioSpecificConfig {
                object_type: 2,
                sample_rate_index: v2v_aac::AudioSpecificConfig::sample_rate_index(
                    af.as_ref().unwrap().sample_rate,
                )
                .expect("采样率不在标准表里"),
                channels: af.as_ref().unwrap().channels as u8,
            }
            .to_bytes()
        });

        // ---- 音频线程：WMA 解码 + AAC 编码（与视频流水线**并行**——互不依赖；
        // 串行的话 AAC 那 5.8 s 就是纯叠加）。音频在 `mdat` 里的写入顺序无所谓
        // （`stco` 按轨各记各的偏移），整段攒完一次写。 ----
        let (aud_tx, aud_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let aac_thread = af.is_some().then(|| {
            let af = af.clone().expect("is_some 就该有格式");
            std::thread::spawn(move || -> std::result::Result<Vec<Vec<u8>>, String> {
                let mut enc = AacEncoder::new(AacConfig {
                    channels: af.channels as u8,
                    sample_rate: af.sample_rate,
                    bitrate: audio_bitrate,
                    ..AacConfig::default()
                })
                .map_err(|e| e.to_string())?;
                let params = v2v_wma::WmaParams::new(
                    2,
                    af.channels as usize,
                    af.sample_rate,
                    af.avg_bytes_per_sec * 8,
                    af.block_align as usize,
                    &af.extra,
                )
                .map_err(|e| e.to_string())?;
                let mut dec =
                    v2v_wma::WmaDecoder::new(params.clone()).map_err(|e| e.to_string())?;
                let mut pcm_buf: Vec<Vec<f32>> = vec![Vec::new(); af.channels as usize];
                let mut out = Vec::new();
                // WMA 编码器延迟：开头丢 frame_len×2 个采样（与 player / CLI 口径一致；
                // 不丢的话音频段多 4 段 —— 虽然靠 elst 也补得回来，但为了和顺序版
                // 产物完全同构，这里照丢）。
                let mut skip_left = params.frame_len * 2;
                while let Ok(payload) = aud_rx.recv() {
                    for planes in dec
                        .decode_superframe(&payload)
                        .map_err(|e| e.to_string())?
                    {
                        for (dst, src) in pcm_buf.iter_mut().zip(planes) {
                            dst.extend(src);
                        }
                        if skip_left > 0 {
                            let have = pcm_buf.first().map_or(0, |p| p.len());
                            let drop = skip_left.min(have);
                            for p in &mut pcm_buf {
                                p.drain(..drop);
                            }
                            skip_left -= drop;
                        }
                        loop {
                            if pcm_buf.first().map_or(0, |p| p.len()) < 1024 {
                                break;
                            }
                            let mut planes = Vec::with_capacity(af.channels as usize);
                            for p in &mut pcm_buf {
                                planes.push(p.drain(..1024).collect::<Vec<_>>());
                            }
                            let bytes = enc
                                .encode_frame(&AacFrame { planes })
                                .map_err(|e| e.to_string())?;
                            out.push(bytes);
                        }
                    }
                }
                Ok(out)
            })
        });

        // ---- 单趟解封装：视频样本按 key_frame 切 GOP ----
        //
        // ⚠️ GOP 并行解码的前提：VC-1 从任意 I 帧起解 == 整条链解到那里
        // （`crates/vc1/tests/gop_start.rs` 真样本 278 个关键帧全查 0 差）。
        let mut gops: Vec<Vec<(Vec<u8>, i64, u64)>> = Vec::new();
        let mut cur: Vec<(Vec<u8>, i64, u64)> = Vec::new();
        let (mut first_pts, mut last_pts) = (None, None);
        let mut frames = 0u64;
        loop {
            let s = match demux.next_sample().map_err(Error::from)? {
                Some(s) => s,
                None => break,
            };
            if s.stream == vnum {
                frames += 1;
                first_pts.get_or_insert(s.timestamp_ms);
                last_pts = Some(s.timestamp_ms);
                if s.key_frame && !cur.is_empty() {
                    gops.push(std::mem::take(&mut cur));
                }
                cur.push((s.data, s.timestamp_ms, frames - 1));
            } else if Some(s.stream) == anum {
                if aud_tx.send(s.data).is_err() {
                    break; // 音频线程已退出（出错），join 时报
                }
            }
        }
        if !cur.is_empty() {
            gops.push(cur);
        }
        drop(aud_tx);
        if profile {
            eprintln!(
                "[probe] 解封装完：样本 {} | GOP {} | 最大 GOP {}",
                frames,
                gops.len(),
                gops.iter().map(|g| g.len()).max().unwrap_or(0)
            );
        }
        // 帧率从首尾时间戳推（两个后端都只要整数）。
        let fps = match (first_pts, last_pts) {
            (Some(f), Some(l)) if l > f && frames > 1 => {
                (((frames - 1) as f64 * 1000.0 / (l - f) as f64).round() as u32).max(1)
            }
            _ => 30,
        };

        // ---- 有序提交的并行解码：worker 并行解，主线程按序喂 VT ----
        //
        // ⚠️ **提交顺序 = 帧间参考的顺序**（reordering 已关）。worker 解完的帧
        // 不能乱序喂 —— 实测乱序提交让 P 帧参考错帧，产物 71 → 239 MB。
        // 所以 worker 只负责**解码**，帧进 `ready`（按序号），主线程按
        // `next_submit` 顺序取出来喂硬件。in-flight 上限兜住内存
        // （解码后的帧 1.38 MB/帧，32 帧 ≈ 44 MB）。
        let mut enc = VideoToolboxEncoder::open(vw, vh, video_bitrate, fps, 4)?;

        struct Pipe {
            /// seq → (解好的帧, pts_ms)。
            ready: std::collections::BTreeMap<u64, (v2v_vc1::Frame, i64)>,
            /// 主线程已消费到哪（滑动窗口的左沿）。
            next_submit: u64,
            gops: std::collections::VecDeque<Vec<(Vec<u8>, i64, u64)>>,
        }
        let cap = 32usize;
        let state = Arc::new((
            Mutex::new(Pipe {
                ready: std::collections::BTreeMap::new(),
                next_submit: 0,
                gops: gops.into_iter().collect(),
            }),
            std::sync::Condvar::new(),
        ));

        let n_workers = std::thread::available_parallelism()
            .map(|n| n.get().max(1))
            .unwrap_or(4)
            .min(6);
        let mut handles = Vec::new();
        for _ in 0..n_workers {
            let state = Arc::clone(&state);
            let seq0 = seq.clone();
            handles.push(std::thread::spawn(move || -> std::result::Result<(), String> {
                let (lock, cvar) = (&state.0, &state.1);
                loop {
                    let chunk = {
                        let mut p = lock.lock().map_err(|e| e.to_string())?;
                        p.gops.pop_front()
                    };
                    let Some(chunk) = chunk else { break };
                    let mut dec =
                        v2v_vc1::Decoder::new(seq0.clone(), vw, vh).map_err(|e| e.to_string())?;
                    for (data, pts_ms, seq) in &chunk {
                        dec.decode_frame(data).map_err(|e| e.to_string())?;
                        let f = dec.frame();
                        // 滑动窗口：只准插入 next_submit + cap 之前的帧
                        //（in_flight 计数被主线程一减就下溢，cap 永远「已满」→ 死等）。
                        let mut p = lock.lock().map_err(|e| e.to_string())?;
                        while *seq >= p.next_submit + cap as u64 {
                            p = cvar.wait(p).map_err(|e| e.to_string())?;
                        }
                        p.ready.insert(*seq, (f, *pts_ms));
                        drop(p);
                        cvar.notify_all();
                    }
                }
                Ok(())
            }));
        }

        // ---- 主线程：按序取帧、喂硬件、收样本 ----
        if profile {
            eprintln!("[probe] worker 已启动（{} 个）", n_workers);
        }
        let (lock, cvar) = (&state.0, &state.1);
        let mut next_seq = 0u64;
        let mut t0 = std::time::Instant::now();
        while next_seq < frames {
            let f = {
                let mut p = lock.lock().map_err(|e| Error::Bad(format!("锁: {e}")))?;
                let mut last_dbg = std::time::Instant::now();
                while !p.ready.contains_key(&next_seq) {
                    let (np, timeout) = cvar
                        .wait_timeout(p, std::time::Duration::from_secs(2))
                        .map_err(|e| Error::Bad(format!("锁: {e}")))?;
                    p = np;
                    if timeout.timed_out() && last_dbg.elapsed().as_secs() >= 2 {
                        eprintln!(
                            "[debug] 主线程等 seq={} ready 的 seq 范围 {:?}..={:?} gops剩余={}",
                            next_seq,
                            p.ready.keys().next(),
                            p.ready.keys().next_back(),
                            p.gops.len()
                        );
                        last_dbg = std::time::Instant::now();
                    }
                }
                p.next_submit += 1; // 滑动窗口右移（worker 据此准入）
                let out = p.ready.remove(&next_seq).expect("contains_key 刚查过");
                cvar.notify_all();
                out
            };
            let (f, pts_ms) = f;
            let out = enc.encode(&f.y, &f.u, &f.v, pts_ms * 1000, false)?;
            encoded_samples.extend(out);
            next_seq += 1;
            if next_seq % 500 == 0 {
                eprintln!(
                    "[profile] 进度 {}/{}（本段 {:.2} s）",
                    next_seq,
                    frames,
                    t0.elapsed().as_secs_f64()
                );
                t0 = std::time::Instant::now();
            }
        }
        let _ = t0;
        // EOS：吸干在途样本。
        let tail = enc.finish()?;
        encoded_samples.extend(tail);
        drop(state);

        // ---- 封装：样本按提交序（= pts 序）写 ----
        let vt = VideoTrack {
            width: vw as u16,
            height: vh as u16,
            timescale: TIMESCALE,
            codec: VideoCodec::Avc1 {
                avc_config: enc.avcc().expect("编完一整条流 avcC 必然可得").to_vec(),
            },
            avg_bitrate: video_bitrate,
        };
        let audio_track = match (&af, &asc) {
            (Some(af), Some(asc)) => Some(v2v_mp4::AudioTrack {
                channels: af.channels,
                sample_rate: af.sample_rate,
                timescale: af.sample_rate,
                codec: v2v_mp4::AudioCodec::Es {
                    object_type: 0x40, // AAC-LC
                    decoder_specific_info: asc.clone(),
                },
                avg_bitrate: audio_bitrate,
                // 我们的 AAC 有 2048 采样点的编码器延迟，写进 edts/elst。
                priming: 2048,
            }),
            _ => None,
        };
        let mut m = match audio_track {
            Some(at) => Mp4Muxer::with_audio(vt, at),
            None => Mp4Muxer::new(vt),
        };
        let mut last_pts: Option<i64> = None;
        for s in &encoded_samples {
            let dur = last_pts.map_or(33, |p| (s.pts_us / 1000 - p).max(1) as u32);
            last_pts = Some(s.pts_us / 1000);
            m.write_video(&s.data, s.key_frame, dur).map_err(mux_err)?;
            stats.frames += 1;
        }
        let audio_bytes = match aac_thread {
            Some(h) => h
                .join()
                .map_err(|e| Error::Bad(format!("音频线程崩了: {e:?}")))?
                .map_err(Error::Bad)?,
            None => Vec::new(),
        };
        for bytes in &audio_bytes {
            m.write_audio(bytes, 1024).map_err(mux_err)?;
            stats.audio_chunks += 1;
        }
        let mp4 = m.finish();
        std::fs::write(output, &mp4).map_err(Error::Io)?;
        stats.output_bytes = mp4.len() as u64;
    }

    #[cfg(target_os = "android")]
    {
        use crate::Mp4HwWriter;
        use v2v_aac::{Encoder as AacEncoder, EncoderConfig as AacConfig, Frame as AacFrame};
        let fps = infer_fps(input);
        let mut hw = Mp4HwWriter::open(
            output,
            info.width,
            info.height,
            video_bitrate,
            fps,
            4,
            info.has_audio,
        )?;
        let mut aac = info.has_audio.then(|| {
            AacEncoder::new(AacConfig {
                channels: info.audio_channels as u8,
                sample_rate: info.audio_sample_rate,
                bitrate: audio_bitrate,
                ..AacConfig::default()
            })
            .expect("建 AAC 编码器")
        });
        if let (Some(enc), true) = (&aac, info.has_audio) {
            hw.add_audio_track(
                &enc.audio_specific_config().to_bytes(),
                info.audio_sample_rate,
                info.audio_channels as u8,
            )?;
        }
        let sample_rate = u64::from(info.audio_sample_rate);
        let ch = info.audio_channels as usize;
        let mut pcm_buf: Vec<Vec<f32>> = vec![Vec::new(); ch];
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
        let _ = (input, output, video_bitrate, audio_bitrate);
        return Err(Error::Unsupported(
            "硬件编码后端只有 macOS（VideoToolbox）与 Android（MediaCodec）".into(),
        ));
    }

    if profile {
        eprintln!(
            "[profile] 全链路 {:.2} s",
            t_start.elapsed().as_secs_f64()
        );
    }
    Ok(stats)
}

/// 从前 100 帧的时间戳差推帧率（容错：时间戳可能不从 0 开始）。
/// 用**一次性的 Player** 探 —— 不能用主 player：它会把前 100 帧消费掉。
#[cfg(target_os = "android")]
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
