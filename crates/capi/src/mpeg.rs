//! MPEG-1/2 的播放与 program stream 拆包（`mpeg12` feature，默认关）。
//!
//! ## 为什么单开一层，不并进 `v2v_player_*`
//!
//! MPEG-1/2 的解码是**图像级 + 引用管理**（B 帧要同时拿前后两个参考帧），
//! 与 VC-1 那条「一包进、一帧出」的路形状不同。硬塞进同一套符号会把
//! 重排语义丢掉，产出的帧序静默错 —— 那比不提供更坏。
//!
//! 所以这里给的是**已经管好重排**的高层入口（`v2v_player::mpeg::MpgPlayer`），
//! 调用方照样是「open → 逐帧取 → EOS」，只是不与 `v2v_player_*` 共用一个句柄类型。
//!
//! ## 为什么默认关
//!
//! 这套码表不小，只吃 WMV 的产物不该背上它。要就开 `mpeg12` feature。

use std::os::raw::c_int;
use std::path::Path;

use v2v_player::mpeg::MpgPlayer;
use v2v_player::VideoFrame;

use crate::common::*;

/// MPEG 播放句柄（不透明）。
pub struct V2vMpeg {
    p: MpgPlayer,
    /// 最近一次交出去的帧 —— 出参里的指针指向这里。
    frame: Option<VideoFrame>,
}

/// 流的静态信息。
#[repr(C)]
pub struct V2vMpegStreamInfo {
    pub size: u32,
    pub width: u32,
    pub height: u32,
    /// 文件里一共有多少张图像（含 B 帧）。
    pub picture_count: u32,
    /// 每帧多少毫秒（帧率推出来的）。
    pub frame_ms: f64,
}

/// 打开一个 MPEG-1/2 program stream。失败返回空指针。
///
/// # Safety
/// `path` 必须是 NUL 结尾的 UTF-8 字符串。
#[no_mangle]
pub unsafe extern "C" fn v2v_mpeg_open(path: *const std::ffi::c_char) -> *mut V2vMpeg {
    guard_ptr(|| {
        let p = match unsafe { cstr(path, "path") } {
            Ok(s) => s,
            Err(e) => {
                set_err(KIND_BAD, &e);
                return std::ptr::null_mut();
            }
        };
        match MpgPlayer::open(Path::new(p)) {
            Ok(pl) => Box::into_raw(Box::new(V2vMpeg { p: pl, frame: None })),
            Err(e) => {
                rc(&e);
                std::ptr::null_mut()
            }
        }
    })
}

/// 关掉并释放。传空指针是空操作。
///
/// # Safety
/// `h` 必须是 [`v2v_mpeg_open`] 的返回值，且只能关一次。
#[no_mangle]
pub unsafe extern "C" fn v2v_mpeg_close(h: *mut V2vMpeg) {
    if h.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(h) });
}

/// 取静态信息。
///
/// # Safety
/// `out` 必须指向一块首字段是 `size` 的可写内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_mpeg_info(h: *mut V2vMpeg, out: *mut V2vMpegStreamInfo) -> c_int {
    guard(|| {
        if h.is_null() || out.is_null() {
            return arg_err("句柄或出参是空指针");
        }
        let hp = unsafe { &*h };
        let (w, hh) = hp.p.size();
        let v = V2vMpegStreamInfo {
            size: std::mem::size_of::<V2vMpegStreamInfo>() as u32,
            width: w,
            height: hh,
            picture_count: hp.p.picture_count() as u32,
            frame_ms: hp.p.frame_ms(),
        };
        unsafe { write_sized(out, v) };
        OK
    })
}

/// 取下一帧。[`EOS`] = 到头。指针活到下一次 `read_video` / `close`。
///
/// # Safety
/// `out` 必须指向一块首字段是 `size` 的可写内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_mpeg_read_video(h: *mut V2vMpeg, out: *mut crate::player::V2vFrame) -> c_int {
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
                let v = crate::player::V2vFrame {
                    size: std::mem::size_of::<crate::player::V2vFrame>() as u32,
                    pts_ms: fr.pts_ms,
                    key_frame: i32::from(fr.key_frame),
                    planes: planes_of(&fr.y, &fr.u, &fr.v, fr.width, fr.height),
                };
                unsafe { write_sized(out, v) };
                OK
            }
            Err(e) => rc(&e),
        }
    })
}
