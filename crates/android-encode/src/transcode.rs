//! 端到端管线：**WMV → MP4，解码用我们的（对 ffmpeg 逐字节一致），视频用平台
//! 硬件编码器，封装用 v2v-mp4** —— 「取优点」的完整形态，示例与测试共用。
//!
//! 后端按编译目标选：
//! - **macOS**：VideoToolbox（`c/vt_encoder.c`，产长度前缀 NALU + avcC，
//!   走 `v2v_mp4::VideoCodec::Avc1`）。**GOP 并行**：每个 worker 解完一段就地
//!   编一段，通道里只传压缩样本（VC-1 从任意 I 帧起解与整链解到那里逐像素一致，
//!   gop_start.rs 验过）+ 音频独立线程；
//! - **Android**：MediaCodec（`c/media_encoder.c`，自带 MediaMuxer），顺序版；
//! - 其它：`Unsupported`。
//!
//! 音频两套后端一致：WMA → f32 → 我们的 AAC-LC → `esds` 轨。

use std::path::Path;

#[cfg(any(target_os = "macos", target_os = "android"))]
use v2v_aac::{Encoder as AacEncoder, EncoderConfig as AacConfig, Frame as AacFrame};
use v2v_player::{Error, Result};

/// 当前平台有没有硬件编码后端（false 时 [`wmv_to_mp4_hw`] 返回 `Unsupported`，
/// 测试据此 SKIP —— Linux CI 上没有 VideoToolbox / MediaCodec）。
#[cfg(any(target_os = "macos", target_os = "android"))]
pub fn hw_supported() -> bool {
    true
}

/// 同上（非支持平台的桩）。
#[cfg(not(any(target_os = "macos", target_os = "android")))]
pub fn hw_supported() -> bool {
    false
}

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
    // 非 macOS / Android：没有硬件编码后端 —— 早退（整段实现在下面的 cfg 块里，
    // 这样 Linux CI 上既不会报「不可达语句」，也不会有未用变量）。
    #[cfg(not(any(target_os = "macos", target_os = "android")))]
    {
        let _ = (input, output, video_bitrate, audio_bitrate);
        return Err(Error::Unsupported(
            "硬件编码后端只有 macOS（VideoToolbox）与 Android（MediaCodec）".into(),
        ));
    }

    #[cfg(any(target_os = "macos", target_os = "android"))]
    {
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

            // ---- 把源 GOP 合并成更大的「编码块」----
            //
            // GOP 并行要求**每块第一帧是 IDR**（相邻块可能是不同 worker 编的，
            // 不能跨块参考）。源 GOP 常常只有半秒多 —— 逐 GOP 编就是每秒 1.5~2 个
            // IDR，同码率下 P 帧的位被 IDR 挤掉（实测 720p 203 个关键帧、1080p 291 个；
            // 旧版单会话只有 35 / 32 个，差的那 0.2~0.3 dB 就在这里）。
            //
            // 合并到 ~3 s 一块：块数仍远多于 worker（并行度不降），IDR 少掉四分之三；
            // 顺带把「每块一次 VT 会话创建」的次数也砍到四分之一。
            //
            // 块边界仍落在源 I 帧上（只在 GOP 之间切），所以「从块首起解 == 整链解到
            // 那里」这条前提不变。
            // 块长（秒）。`V2V_HW_CHUNK_SECS` 只用来量「块长 ↔ 并行度/IDR 数」的
            // 曲线：实测 2 s 会话太多、10 s 并行度不够，3 s 最好；正常别设。
            let chunk_secs: usize = std::env::var("V2V_HW_CHUNK_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);
            let target_frames = (fps as usize * chunk_secs).max(1);
            let mut chunks: Vec<Vec<(Vec<u8>, i64, u64)>> = Vec::new();
            let mut cur: Vec<(Vec<u8>, i64, u64)> = Vec::new();
            for g in gops {
                if !cur.is_empty() && cur.len() + g.len() > target_frames {
                    chunks.push(std::mem::take(&mut cur));
                }
                cur.extend(g);
            }
            if !cur.is_empty() {
                chunks.push(cur);
            }
            if profile {
                eprintln!(
                    "[probe] 合并成 {} 块（目标 {} 帧/块，最大 {} 帧）",
                    chunks.len(),
                    target_frames,
                    chunks.iter().map(|c| c.len()).max().unwrap_or(0)
                );
            }

            // ---- 每个 worker 自带硬件编码会话：解完一段就地编一段 ----
            //
            // 为什么**编码也下放**：解码后的 YUV 很大（1080p 一帧 3.1 MB），按「解好的
            // 帧」做背压，内存随并行度线性涨 —— 4 个 worker 就要 1.4 GB，实测直接被
            // 系统杀掉（6 个 worker 只剩 2 个能跑，并行度名存实亡）。而**编完的样本**
            // 只有 ~0.5 KB/帧：编码下放后 worker 之间只传压缩样本，内存从 GB 级掉到
            // MB 级，6 个 worker 才真正同时跑起来（1080p 47 → 32 → 26 s，见 TODO）。
            //
            // **每个 GOP 一个新会话**，两个理由：
            // ① GOP 首帧强制 IDR —— 段与段之间独立，样本按序拼接即可，不用跨段猜
            //    「这个样本属于哪一帧」；
            // ② 会话 `finish()` 会把在途样本全部吐出，段的边界就是确定的。
            // 代价：码控没有跨段历史，码率分布略毛糙（实测体积/PSNR 几乎不变）。
            //
            // 顺序保证：worker 把样本挂在该段**首帧的 seq** 上，主线程按 seq 顺序取段
            // 封装 —— 提交顺序 = 帧间参考的顺序（乱序提交会让 P 帧参考错帧，
            // 产物 71 → 239 MB，踩过）。
            struct Pipe {
                /// 块首帧 seq → 该块的压缩样本（已按 pts 序）。
                ready: std::collections::BTreeMap<u64, Vec<crate::vt::VtSample>>,
                chunks: std::collections::VecDeque<Vec<(Vec<u8>, i64, u64)>>,
                /// 第一个交出 avcC 的会话（所有会话配置相同，任取一份）。
                avcc: Option<Vec<u8>>,
                /// 任一 worker 失败的原因。主线程只在「等某一段」上阻塞 ——
                /// 没有这条通道的话，worker 一死那一段就永远不来，**挂死而不是报错**。
                failed: Option<String>,
            }
            let chunk_first_seq: Vec<u64> = chunks.iter().map(|c| c[0].2).collect();
            let state = Arc::new((
                Mutex::new(Pipe {
                    ready: std::collections::BTreeMap::new(),
                    chunks: chunks.into_iter().collect(),
                    avcc: None,
                    failed: None,
                }),
                std::sync::Condvar::new(),
            ));

            // 并行度：默认按核数，封顶 6（再多只是抢同一个硬件编码器）。
            // `V2V_HW_WORKERS` 只用来量并行度曲线 —— 正常别设。
            let n_workers = std::env::var("V2V_HW_WORKERS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|n| n.get().max(1))
                        .unwrap_or(4)
                })
                .clamp(1, 16);
            let mut handles = Vec::new();
            for _ in 0..n_workers {
                let state = Arc::clone(&state);
                let seq0 = seq.clone();
                handles.push(std::thread::spawn(move || {
                    let r = (|| -> std::result::Result<(), String> {
                        let (lock, cvar) = (&state.0, &state.1);
                        loop {
                            let chunk = {
                                let mut p = lock.lock().map_err(|e| e.to_string())?;
                                p.chunks.pop_front()
                            };
                            let Some(chunk) = chunk else { break };
                            let first_seq = chunk[0].2;
                            // 每段一个新**解码器**：从 I 帧起解，段间无状态
                            //（gop_start.rs 验过）。它是纯 Rust 的，建起来几乎不要钱。
                            let mut dec = v2v_vc1::Decoder::new(seq0.clone(), vw, vh)
                                .map_err(|e| e.to_string())?;
                            // 每段一个新**编码会话**。不是为了省事，是**码控**：
                            // 复用会话 + 每段 `finish()`（CompleteFrames）会让 VT 的码控
                            // 一路往上飘 —— 实测 720p 70 MB → 365 MB、帧数不变、单帧从
                            // 3 KB 爬到 140 KB。所以段的边界宁可付一次会话创建/销毁。
                            let mut enc = VideoToolboxEncoder::open(vw, vh, video_bitrate, fps, 4)
                                .map_err(|e| e.to_string())?;
                            let mut samples = Vec::new();
                            for (i, (data, pts_ms, _)) in chunk.iter().enumerate() {
                                dec.decode_frame(data).map_err(|e| e.to_string())?;
                                let f = dec.frame();
                                // 段首强制 IDR：这一段从此独立可解 —— 相邻段可能是
                                // 别的 worker 编的，不能跨段参考。
                                samples.extend(
                                    enc.encode(&f.y, &f.u, &f.v, pts_ms * 1000, i == 0)
                                        .map_err(|e| e.to_string())?,
                                );
                            }
                            // CompleteFrames：本段在途样本全部吐出，边界是确定的。
                            samples.extend(enc.finish().map_err(|e| e.to_string())?);
                            let avcc = enc.avcc().map(|a| a.to_vec());
                            // 先放掉会话再拿锁 —— `vt_free` 要 Invalidate，别占着全局锁。
                            drop(enc);
                            {
                                let mut p = lock.lock().map_err(|e| e.to_string())?;
                                // 整条流只写**一份** avcC。各会话的 SPS/PPS 必须一致
                                // （配置相同就该一致）；不一致说明有会话走了另一条编码
                                // 路径，后半段会用错配置 —— 宁可当场报错，也别静默产出
                                // 一个解不开的文件。
                                if let Some(a) = &avcc {
                                    match &p.avcc {
                                        None => p.avcc = Some(a.clone()),
                                        Some(prev) if prev != a => {
                                            let msg = "各编码会话产出的 avcC 不一致（SPS/PPS 不同）—— 一份 avcC 描述不了整条流";
                                            if p.failed.is_none() {
                                                p.failed = Some(msg.into());
                                            }
                                            cvar.notify_all();
                                            return Err(msg.into());
                                        }
                                        Some(_) => {}
                                    }
                                }
                                p.ready.insert(first_seq, samples);
                            }
                            cvar.notify_all();
                        }
                        Ok(())
                    })();
                    // 失败要**主动叫醒**主线程：它可能正阻塞在「等某一段」上。
                    if let Err(e) = &r {
                        let (lock, cvar) = (&state.0, &state.1);
                        if let Ok(mut p) = lock.lock() {
                            if p.failed.is_none() {
                                p.failed = Some(e.clone());
                            }
                        }
                        cvar.notify_all();
                    }
                    r
                }));
            }

            // ---- 主线程：按段序取压缩样本、收集 ----
            let (lock, cvar) = (&state.0, &state.1);
            let mut gi = 0usize;
            let mut t0 = std::time::Instant::now();
            while gi < chunk_first_seq.len() {
                let want = chunk_first_seq[gi];
                let samples = {
                    let mut p = lock.lock().map_err(|e| Error::Bad(format!("锁: {e}")))?;
                    while !p.ready.contains_key(&want) {
                        // worker 已经失败就别等了 —— 那一段永远不会来。
                        if let Some(e) = &p.failed {
                            return Err(Error::Bad(e.clone()));
                        }
                        p = cvar.wait(p).map_err(|e| Error::Bad(format!("锁: {e}")))?;
                    }
                    p.ready.remove(&want).expect("contains_key 刚查过")
                };
                encoded_samples.extend(samples);
                gi += 1;
                if profile && (gi % 4 == 0 || gi == chunk_first_seq.len()) {
                    eprintln!(
                        "[profile] 段 {}/{}（本段 {:.2} s）",
                        gi,
                        chunk_first_seq.len(),
                        t0.elapsed().as_secs_f64()
                    );
                    t0 = std::time::Instant::now();
                }
            }
            for h in handles {
                h.join()
                    .map_err(|e| Error::Bad(format!("worker 崩了: {e:?}")))?
                    .map_err(Error::Bad)?;
            }
            let avcc = {
                let p = lock.lock().map_err(|e| Error::Bad(format!("锁: {e}")))?;
                p.avcc.clone()
            };
            drop(state);

            // ---- 封装：样本按提交序（= pts 序）写 ----
            let vt = VideoTrack {
                width: vw as u16,
                height: vh as u16,
                timescale: TIMESCALE,
                codec: VideoCodec::Avc1 {
                    avc_config: avcc.ok_or_else(|| {
                        Error::Bad("没有任何视频样本，avcC 拿不到".into())
                    })?,
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

        if profile {
            eprintln!(
                "[profile] 全链路 {:.2} s",
                t_start.elapsed().as_secs_f64()
            );
        }
        Ok(stats)
    }
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
