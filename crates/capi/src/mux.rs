//! 封装（`v2v_mux_*`）：ASF / MP4。
//!
//! ## 两个封装器的入参形状**真的不一样**，这里不假装一样
//!
//! | | ASF | MP4 |
//! |---|---|---|
//! | 视频轨 | `width / height / fourcc` | `width / height / timescale / codec / VOL / bitrate` |
//! | 音频轨 | 整条 `WAVEFORMATEX`（含 `extra`） | `timescale / object_type / ASC / priming / bitrate` |
//! | 写入 | 按**毫秒** | 按 **timescale 的 tick** |
//!
//! 所以出参结构体里两组字段都在，**各自只看自己那组**（注释里写明了哪组给谁）。
//! 与其造一个「两边都能喂」的抽象（那必然丢掉某一侧的语义），不如把差异摆在台面上。
//!
//! ## `write` 同时给 `pts_ms` 和 `duration_ticks`
//!
//! 不是冗余：ASF 的写入接口按毫秒，MP4 按 tick。**音频**尤其不能换算 ——
//! AAC 一帧是 1024 个采样点，换成毫秒是 23.2 → 取整 23 → 再换回 tick 就变成 1014，
//! 每帧丢 10 个 tick，几十秒后就明显偏了。所以两个都给，各自用自己的那个。
//!
//! ## `finish` 之后句柄就废了
//!
//! 两个封装器的 `finish` 都是**消耗 self** 的（要拿最终字节），所以这里用
//! `Option` 装着。`finish` 之后再 `write` 会返回 `V2V_ERR_ARG`。

use std::os::raw::c_int;
use std::path::PathBuf;

use v2v_container::AudioFormat;

use crate::common::*;

/// 输出 ASF（`.wmv` / `.wma`）。
pub const MUX_ASF: u32 = 1;
/// 输出 MP4。
pub const MUX_MP4: u32 = 2;

/// MP4 视频轨的编解码类型：MPEG-4 Part 2（`mp4v`）。
pub const MUX_VCODEC_MP4V: u32 = 0;

/// 视频轨参数。ASF 看前三项，MP4 看后五项。
#[repr(C)]
pub struct V2vMuxVideo {
    pub size: u32,
    pub width: u32,
    pub height: u32,
    /// **ASF**：FourCC（MP43 编码时传 `'MP43'` 的小端）。
    pub fourcc: u32,
    /// **MP4**：时间基（每秒多少 tick）。0 视为 1000（即 tick 就是毫秒）。
    pub timescale: u32,
    /// **MP4**：`V2V_MUX_VCODEC_MP4V`。
    pub codec: u32,
    /// **MP4**：平均码率（写进 `esds` / `btrt`）。
    pub avg_bitrate: u32,
    /// **MP4**：VOL（`v2v_venc_header` 拿到的整段）。
    pub decoder_specific_info: *const u8,
    pub dsi_len: u32,
}

/// 音频轨参数。ASF 看 WAVEFORMATEX 那组，MP4 看时间基那组。
#[repr(C)]
pub struct V2vMuxAudio {
    pub size: u32,
    // ---- ASF ----
    pub format_tag: u32,
    pub channels: u32,
    pub sample_rate: u32,
    pub avg_bytes_per_sec: u32,
    pub block_align: u32,
    pub bits_per_sample: u32,
    pub extra: *const u8,
    pub extra_len: u32,
    // ---- MP4 ----
    /// 时间基。0 视为采样率（AAC / PCM 都该这么给）。
    pub timescale: u32,
    /// `object_type`（AAC = `0x40`）。
    pub object_type: u32,
    pub avg_bitrate: u32,
    /// AudioSpecificConfig。
    pub asc: *const u8,
    pub asc_len: u32,
    /// 编码器预热样本数：AAC = 1024，PCM = 0。
    pub priming: u32,
}

enum Mux {
    Asf(v2v_asf::AsfMuxer),
    Mp4(v2v_mp4::Mp4Muxer),
}

/// 封装句柄（不透明）。
pub struct V2vMux {
    /// `finish` 会把它取走（两个封装器的 `finish` 都消耗 self）。
    inner: Option<Mux>,
    /// 输出路径 —— `finish` 时整段字节往这里落盘。
    path: PathBuf,
}

/// 打开一个封装器。`audio` 传空表示只要视频轨。失败返回空指针。
///
/// # Safety
/// `path` 是 NUL 结尾的 UTF-8 串；`video` 必须可读；`audio` 要么为空指针，
/// 要么可读。
#[no_mangle]
pub unsafe extern "C" fn v2v_mux_open(
    container: u32,
    path: *const std::ffi::c_char,
    video: *const V2vMuxVideo,
    audio: *const V2vMuxAudio,
) -> *mut V2vMux {
    guard_ptr(|| {
        let p = match unsafe { cstr(path, "path") } {
            Ok(s) => s,
            Err(e) => {
                set_err(KIND_BAD, &e);
                return std::ptr::null_mut();
            }
        };
        if video.is_null() {
            set_err(KIND_BAD, "video 参数是空指针（视频轨是必填的）");
            return std::ptr::null_mut();
        }
        let v = unsafe { &*video };
        if (v.size as usize) < std::mem::size_of::<V2vMuxVideo>() {
            set_err(KIND_BAD, "video->size 太小，可能不是本版本的调用方");
            return std::ptr::null_mut();
        }
        let a = if audio.is_null() {
            None
        } else {
            let a = unsafe { &*audio };
            if (a.size as usize) < std::mem::size_of::<V2vMuxAudio>() {
                set_err(KIND_BAD, "audio->size 太小，可能不是本版本的调用方");
                return std::ptr::null_mut();
            }
            Some(a)
        };

        let built: Result<Mux, String> = match container {
            MUX_ASF => {
                let af = match a {
                    None => None,
                    Some(a) => {
                        let extra = match unsafe { slice(a.extra, a.extra_len) } {
                            Ok(b) => b.to_vec(),
                            Err(e) => return set_null(&e),
                        };
                        Some(AudioFormat {
                            format_tag: a.format_tag as u16,
                            channels: a.channels as u16,
                            sample_rate: a.sample_rate,
                            avg_bytes_per_sec: a.avg_bytes_per_sec,
                            block_align: a.block_align as u16,
                            bits_per_sample: a.bits_per_sample as u16,
                            extra,
                        })
                    }
                };
                let info = v2v_asf::VideoStreamInfo {
                    width: v.width,
                    height: v.height,
                    fourcc: v.fourcc,
                };
                Ok(Mux::Asf(v2v_asf::AsfMuxer::with_audio(info, af)))
            }
            MUX_MP4 => {
                if v.codec != MUX_VCODEC_MP4V {
                    set_err(KIND_UNSUPPORTED, "MP4 只做 MPEG-4 Part 2 的视频轨");
                    return std::ptr::null_mut();
                }
                let dsi = match unsafe { slice(v.decoder_specific_info, v.dsi_len) } {
                    Ok(b) => b.to_vec(),
                    Err(e) => return set_null(&e),
                };
                let vtrack = v2v_mp4::VideoTrack {
                    width: v.width as u16,
                    height: v.height as u16,
                    timescale: if v.timescale == 0 { 1000 } else { v.timescale },
                    codec: v2v_mp4::VideoCodec::Mp4v {
                        decoder_specific_info: dsi,
                    },
                    avg_bitrate: v.avg_bitrate,
                };
                match a {
                    None => Ok(Mux::Mp4(v2v_mp4::Mp4Muxer::new(vtrack))),
                    Some(a) => {
                        let asc = match unsafe { slice(a.asc, a.asc_len) } {
                            Ok(b) => b.to_vec(),
                            Err(e) => return set_null(&e),
                        };
                        let atrack = v2v_mp4::AudioTrack {
                            channels: a.channels as u16,
                            sample_rate: a.sample_rate,
                            timescale: if a.timescale == 0 {
                                a.sample_rate
                            } else {
                                a.timescale
                            },
                            avg_bitrate: a.avg_bitrate,
                            codec: v2v_mp4::AudioCodec::Es {
                                object_type: a.object_type as u8,
                                decoder_specific_info: asc,
                            },
                            priming: a.priming,
                        };
                        Ok(Mux::Mp4(v2v_mp4::Mp4Muxer::with_audio(vtrack, atrack)))
                    }
                }
            }
            other => {
                set_err(KIND_BAD, &format!("未知的输出容器编号 {other}"));
                return std::ptr::null_mut();
            }
        };
        match built {
            Ok(inner) => Box::into_raw(Box::new(V2vMux {
                inner: Some(inner),
                path: PathBuf::from(p),
            })),
            Err(e) => {
                set_err(KIND_BAD, &e);
                std::ptr::null_mut()
            }
        }
    })
}

/// 记一条错误并返回空指针（`guard_ptr` 里用）。
fn set_null(msg: &str) -> *mut V2vMux {
    set_err(KIND_BAD, msg);
    std::ptr::null_mut()
}

/// 关掉并释放。传空指针是空操作。**没 `finish` 就关 = 什么都不写出来。**
///
/// # Safety
/// `h` 必须是 [`v2v_mux_open`] 的返回值，且只能关一次。
#[no_mangle]
pub unsafe extern "C" fn v2v_mux_close(h: *mut V2vMux) {
    if h.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(h) });
}

/// 写一帧视频。
///
/// `pts_ms` 给 ASF 用；`duration_ticks` 给 MP4 用（按轨道时间基）。
///
/// # Safety
/// `data` 必须指向 `len` 字节可读内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_mux_write_video(
    h: *mut V2vMux,
    data: *const u8,
    len: u32,
    key_frame: c_int,
    pts_ms: i64,
    duration_ticks: u32,
) -> c_int {
    guard(|| {
        if h.is_null() {
            return arg_err("句柄是空指针");
        }
        let bytes = match unsafe { slice(data, len) } {
            Ok(b) => b,
            Err(e) => return arg_err(&e),
        };
        let hp = unsafe { &mut *h };
        let Some(m) = hp.inner.as_mut() else {
            return arg_err("已经 finish 过了，这个句柄不能再用");
        };
        match m {
            Mux::Asf(x) => {
                if !(0..=u32::MAX as i64).contains(&pts_ms) {
                    return arg_err("ASF 的时间戳必须落在 0..=u32::MAX 毫秒");
                }
                match x.write_video(bytes, key_frame != 0, pts_ms as u32) {
                    Ok(()) => OK,
                    Err(e) => rc(&e),
                }
            }
            Mux::Mp4(x) => match x.write_video(bytes, key_frame != 0, duration_ticks) {
                Ok(()) => OK,
                Err(e) => rc(&e),
            },
        }
    })
}

/// 写一帧音频。
///
/// `pts_ms` 给 ASF 用；`duration_ticks` 给 MP4 用。
///
/// # Safety
/// `data` 必须指向 `len` 字节可读内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_mux_write_audio(
    h: *mut V2vMux,
    data: *const u8,
    len: u32,
    pts_ms: i64,
    duration_ticks: u32,
) -> c_int {
    guard(|| {
        if h.is_null() {
            return arg_err("句柄是空指针");
        }
        let bytes = match unsafe { slice(data, len) } {
            Ok(b) => b,
            Err(e) => return arg_err(&e),
        };
        let hp = unsafe { &mut *h };
        let Some(m) = hp.inner.as_mut() else {
            return arg_err("已经 finish 过了，这个句柄不能再用");
        };
        match m {
            Mux::Asf(x) => {
                if !(0..=u32::MAX as i64).contains(&pts_ms) {
                    return arg_err("ASF 的时间戳必须落在 0..=u32::MAX 毫秒");
                }
                match x.write_audio(bytes, pts_ms as u32) {
                    Ok(()) => OK,
                    Err(e) => rc(&e),
                }
            }
            Mux::Mp4(x) => match x.write_audio(bytes, duration_ticks) {
                Ok(()) => OK,
                Err(e) => rc(&e),
            },
        }
    })
}

/// 收尾：把整段字节写到 `open` 时给的路径。之后句柄不可再用（`close` 照常）。
///
/// # Safety
/// `h` 必须是有效句柄。
#[no_mangle]
pub unsafe extern "C" fn v2v_mux_finish(h: *mut V2vMux) -> c_int {
    guard(|| {
        if h.is_null() {
            return arg_err("句柄是空指针");
        }
        let hp = unsafe { &mut *h };
        let Some(m) = hp.inner.take() else {
            return arg_err("已经 finish 过了");
        };
        let bytes = match m {
            Mux::Asf(x) => x.finish(),
            Mux::Mp4(x) => x.finish(),
        };
        match std::fs::write(&hp.path, &bytes) {
            Ok(()) => OK,
            Err(e) => {
                set_err(KIND_IO, &format!("写 {} 失败: {e}", hp.path.display()));
                ERR_FAILED
            }
        }
    })
}
