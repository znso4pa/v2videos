//! 顺时播放的 C ABI（`v2v_player_*`）。
//!
//! 这一层是**给播放器用的**：打开一个 `.wmv`，按自己的节奏交替拉视频帧与音频帧。
//! 下面的「解封装 / 解码」两层是给想自己交织的调用方用的；只想播的话用这一层，
//! 它已经把「一个载荷里装着好几帧音频」「编码器延迟的静音」「seek 后解码器重建」
//! 这些坑都处理掉了。
//!
//! ## 帧缓冲是**借用**的
//!
//! `read_video` / `read_audio` 给出的指针指向句柄内部，**活到本句柄的下一次
//! `read_*` 或 `seek_ms`**。想留就自己按 `stride` 拷一份（`stride` 恒等于宽度，
//! 所以就是逐行 `memcpy`）。
//! 这么做是为了省掉每帧一次全帧拷贝（720p 一帧 ≈ 1.4 MB）。
//!
//! ## 两个轨互不阻塞
//!
//! 视频与音频各自推进，先拉谁、拉多少都行。`seek_ms` 会把两轨一起挪。

use std::os::raw::c_int;
use std::path::Path;

use v2v_player::{AudioFrame, Player, VideoFrame};

use crate::common::*;

/// 播放器句柄（不透明）。
pub struct V2vPlayer {
    p: Player,
    /// 最近一次交出去的视频帧 —— 出参里的指针指向这里。
    frame: Option<VideoFrame>,
    /// 最近一次交出去的音频帧。
    audio: Option<AudioFrame>,
    /// 音频的平面指针表（出参的 `planes` 指向这里）。
    audio_ptrs: [*const f32; MAX_CHANNELS],
}

/// YUV420P 三平面（定义在公共层，解码 / 编码 / 播放三处共用）。
pub use crate::common::V2vPlanes;

/// 一帧视频。
#[repr(C)]
pub struct V2vFrame {
    pub size: u32,
    /// 容器时间戳（ms，已扣 preroll）。
    pub pts_ms: i64,
    /// 0 / 1。seek 的落点候选就是它。
    pub key_frame: i32,
    pub planes: V2vPlanes,
}

/// 一帧音频（每声道一条 float）。
#[repr(C)]
pub struct V2vAudioFrame {
    pub size: u32,
    /// 按**已交出**的样本数推出来的（不是容器时间戳：一个载荷常装着好几帧）。
    pub pts_ms: i64,
    /// `channels` 条，每条 `samples` 个。活到下次 `read_audio`。
    pub planes: *const *const f32,
    pub channels: u32,
    pub samples: u32,
    pub sample_rate: u32,
}

/// 流的静态信息。
#[repr(C)]
pub struct V2vInfo {
    pub size: u32,
    pub duration_ms: u64,
    pub width: u32,
    pub height: u32,
    /// 视频 FourCC（如 `'WMV3'` 的字节小端）。
    pub video_fourcc: u32,
    pub audio_sample_rate: u32,
    /// 0 = 没有音频轨。
    pub audio_channels: u32,
    pub has_audio: u32,
    /// 能力标志，见头文件里的 `V2V_INFO_*`。
    pub flags: u32,
}

fn planes_of(f: &VideoFrame) -> V2vPlanes {
    crate::common::planes_of(&f.y, &f.u, &f.v, f.width, f.height)
}

/// 打开一个文件。失败返回空指针（原因看 `v2v_last_error()`）。
///
/// # Safety
/// `path` 必须是 NUL 结尾的 UTF-8 字符串。
#[no_mangle]
pub unsafe extern "C" fn v2v_player_open(path: *const std::ffi::c_char) -> *mut V2vPlayer {
    guard_ptr(|| {
        let p = match unsafe { cstr(path, "path") } {
            Ok(s) => s,
            Err(e) => {
                set_err(KIND_BAD, &e);
                return std::ptr::null_mut();
            }
        };
        match Player::open(Path::new(p)) {
            Ok(pl) => Box::into_raw(Box::new(V2vPlayer {
                p: pl,
                frame: None,
                audio: None,
                audio_ptrs: [std::ptr::null(); MAX_CHANNELS],
            })),
            Err(e) => {
                rc(&e);
                std::ptr::null_mut()
            }
        }
    })
}

/// 关掉并释放句柄。传空指针是空操作。
///
/// # Safety
/// `h` 必须是 [`v2v_player_open`] 的返回值，且**只能关一次**。
#[no_mangle]
pub unsafe extern "C" fn v2v_player_close(h: *mut V2vPlayer) {
    if h.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(h) });
}

/// 取静态信息。
///
/// # Safety
/// `out` 必须指向一块首字段是 `size = sizeof(v2v_info)` 的可写内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_player_info(h: *mut V2vPlayer, out: *mut V2vInfo) -> c_int {
    guard(|| {
        if h.is_null() || out.is_null() {
            return arg_err("句柄或出参是空指针");
        }
        let hp = unsafe { &*h };
        let i = hp.p.info();
        let v = V2vInfo {
            size: std::mem::size_of::<V2vInfo>() as u32,
            duration_ms: i.duration_ms,
            width: i.width,
            height: i.height,
            video_fourcc: i.video_fourcc,
            audio_sample_rate: i.audio_sample_rate,
            audio_channels: u32::from(i.audio_channels),
            has_audio: u32::from(i.has_audio),
            flags: i.flags,
        };
        unsafe { write_sized(out, v) };
        OK
    })
}

/// 取下一帧视频。[`EOS`] = 视频轨到头。
///
/// 出参里的指针活到本句柄的下一次 `read_video` / `seek_ms` / `close`。
///
/// # Safety
/// `out` 必须指向一块首字段是 `size` 的可写内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_player_read_video(h: *mut V2vPlayer, out: *mut V2vFrame) -> c_int {
    guard(|| {
        if h.is_null() || out.is_null() {
            return arg_err("句柄或出参是空指针");
        }
        let hp = unsafe { &mut *h };
        match hp.p.next_video() {
            Ok(None) => EOS,
            Ok(Some(f)) => {
                hp.frame = Some(f);
                let fr = hp.frame.as_ref().expect("刚放进去");
                let v = V2vFrame {
                    size: std::mem::size_of::<V2vFrame>() as u32,
                    pts_ms: fr.pts_ms,
                    key_frame: i32::from(fr.key_frame),
                    planes: planes_of(fr),
                };
                unsafe { write_sized(out, v) };
                OK
            }
            Err(e) => rc(&e),
        }
    })
}

/// 取下一帧音频。[`EOS`] = 音频轨到头（或本来就没有音频）。
///
/// 出参里的指针活到本句柄的下一次 `read_audio` / `seek_ms` / `close`。
///
/// # Safety
/// `out` 必须指向一块首字段是 `size` 的可写内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_player_read_audio(
    h: *mut V2vPlayer,
    out: *mut V2vAudioFrame,
) -> c_int {
    guard(|| {
        if h.is_null() || out.is_null() {
            return arg_err("句柄或出参是空指针");
        }
        let hp = unsafe { &mut *h };
        match hp.p.next_audio() {
            Ok(None) => EOS,
            Ok(Some(a)) => {
                if a.channels > MAX_CHANNELS {
                    set_err(
                        KIND_UNSUPPORTED,
                        &format!("音频 {} 条声道，ABI 最多支持 {MAX_CHANNELS}", a.channels),
                    );
                    return ERR_UNSUPPORTED;
                }
                hp.audio = Some(a);
                let ar = hp.audio.as_ref().expect("刚放进去");
                let mut ptrs = [std::ptr::null::<f32>(); MAX_CHANNELS];
                for (i, p) in ar.planes.iter().enumerate() {
                    ptrs[i] = p.as_ptr();
                }
                hp.audio_ptrs = ptrs;
                let v = V2vAudioFrame {
                    size: std::mem::size_of::<V2vAudioFrame>() as u32,
                    pts_ms: ar.pts_ms,
                    planes: hp.audio_ptrs.as_ptr(),
                    channels: ar.channels as u32,
                    samples: ar.samples as u32,
                    sample_rate: ar.sample_rate,
                };
                unsafe { write_sized(out, v) };
                OK
            }
            Err(e) => rc(&e),
        }
    })
}

/// seek 到 `ms`，返回**实际落点**（关键帧的时间戳，≤ `ms`，最多差一个 GOP）。
/// 失败返回 -1。
///
/// 落点之后下一次 `read_video` 交出的就是落点那一帧。两轨一起挪。
///
/// # Safety
/// `h` 必须是有效句柄。
#[no_mangle]
pub unsafe extern "C" fn v2v_player_seek_ms(h: *mut V2vPlayer, ms: u64) -> i64 {
    if h.is_null() {
        set_err(KIND_BAD, "句柄是空指针");
        return -1;
    }
    let hp = unsafe { &mut *h };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| hp.p.seek_ms(ms))) {
        Ok(Ok(actual)) => {
            // 之前交出去的帧缓冲作废 —— 别让调用方拿着旧指针去读已经释放的内存。
            hp.frame = None;
            hp.audio = None;
            actual
        }
        Ok(Err(e)) => {
            rc(&e);
            -1
        }
        Err(_) => {
            set_err(KIND_BAD, "内部 panic 已在 ABI 边界拦下");
            -1
        }
    }
}

/// 已交出的最后一帧视频的时间戳（ms）。没取过帧时是 0。
///
/// # Safety
/// `h` 必须是有效句柄。
#[no_mangle]
pub unsafe extern "C" fn v2v_player_position_ms(h: *const V2vPlayer) -> i64 {
    if h.is_null() {
        return 0;
    }
    unsafe { &*h }.p.position_ms()
}
