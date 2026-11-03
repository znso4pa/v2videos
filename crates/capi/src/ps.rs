//! MPEG-1/2 program stream 拆包（`v2v_ps_*`，`mpeg12` feature）。
//!
//! ## 形状与别处不同，如实暴露
//!
//! 这个拆包器**持有整个文件**（`PsDemuxer::new(Vec<u8>)`），不是流式 reader ——
//! 它靠随机找起始码来切 PES。所以 `open` 会把整个文件读进内存，
//! 出来的是一串 PES payload（而不是像 ASF/AVI 那样吐「已重组的样本」）。
//!
//! 帧边界由**视频解码器自己按起始码找**，这里不保证 PES 边界等于帧边界。
//! 真要逐帧，用上面的 `v2v_mpeg_*`（它已经管好了重排）。

use std::os::raw::c_int;
use std::path::Path;

use v2v_mpegps::PesPacket;

use crate::common::*;

/// 一个 PES payload。
#[repr(C)]
pub struct V2vPes {
    pub size: u32,
    /// PES 的 `stream_id`：`0xE0..=0xEF` 视频，`0xC0..=0xDF` 音频。
    pub stream_id: u32,
    /// 0 = 这个 PES 头里没有 PTS。
    pub has_pts: i32,
    pub pts_ms: i64,
    /// 借用的字节（指向整份文件缓冲），活到下一次 `next` / `close`。
    pub data: *const u8,
    pub len: u32,
}

/// PS 拆包句柄（不透明）。
pub struct V2vPs {
    d: v2v_mpegps::PsDemuxer,
    /// 最近一次交出去的包（`data` 指向整份缓冲，这里只是留着判重/生命周期）。
    cur: Option<PesPacket>,
}

/// 打开一个 program stream（**整文件读进内存**）。失败返回空指针。
///
/// # Safety
/// `path` 必须是 NUL 结尾的 UTF-8 字符串。
#[no_mangle]
pub unsafe extern "C" fn v2v_ps_open(path: *const std::ffi::c_char) -> *mut V2vPs {
    guard_ptr(|| {
        let p = match unsafe { cstr(path, "path") } {
            Ok(s) => s,
            Err(e) => {
                set_err(KIND_BAD, &e);
                return std::ptr::null_mut();
            }
        };
        match std::fs::read(Path::new(p)) {
            Ok(bytes) => Box::into_raw(Box::new(V2vPs {
                d: v2v_mpegps::PsDemuxer::new(bytes),
                cur: None,
            })),
            Err(e) => {
                set_err(KIND_IO, &format!("读 {p} 失败: {e}"));
                std::ptr::null_mut()
            }
        }
    })
}

/// 关掉并释放。传空指针是空操作。
///
/// # Safety
/// `h` 必须是 [`v2v_ps_open`] 的返回值，且只能关一次。
#[no_mangle]
pub unsafe extern "C" fn v2v_ps_close(h: *mut V2vPs) {
    if h.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(h) });
}

/// 取下一个 PES payload。[`EOS`] = 到头。
///
/// # Safety
/// `out` 必须指向一块首字段是 `size` 的可写内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_ps_next(h: *mut V2vPs, out: *mut V2vPes) -> c_int {
    guard(|| {
        if h.is_null() || out.is_null() {
            return arg_err("句柄或出参是空指针");
        }
        let hp = unsafe { &mut *h };
        let Some(p) = hp.d.next_packet() else {
            return EOS;
        };
        let (start, len, sid, pts) = (p.range.start, p.range.len(), p.stream_id, p.pts_ms);
        hp.cur = Some(p);
        // 指针指向整份文件缓冲 —— 它在句柄里，活到 close。
        let base = hp.d.data().as_ptr();
        let v = V2vPes {
            size: std::mem::size_of::<V2vPes>() as u32,
            stream_id: u32::from(sid),
            has_pts: i32::from(pts.is_some()),
            pts_ms: pts.map_or(0, |x| x as i64),
            data: unsafe { base.add(start) },
            len: len as u32,
        };
        unsafe { write_sized(out, v) }
    })
}
