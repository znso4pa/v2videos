//! **全仓库的 unsafe / FFI 都集中在这里。**
//!
//! 声明的是 `c/media_encoder.c` 编出来的 `ame_*` C ABI（函数签名必须与头文件一致）。
//! 约定：返回 0 = 成功；-1 = 失败（`ame_last_error` 拿原因）。
//!
//! ## 为什么 unsafe 只在这一个模块
//!
//! - `ame_*` 是我们自己写的 shim：它对内吞掉 NDK 的所有句柄与生命周期规则，
//!   对外只暴露「字节缓冲 + 微秒时间戳」这种 Rust 侧能静态检查的东西；
//! - 裸指针只在 `Mp4HwWriter`（lib.rs）里持有，且**只在**这个模块的调用边界上解引用；
//! - 除此之外的任何模块都不允许出现 `unsafe`（crate 级 `#[forbid(unsafe_code)]`
//!   在 lib.rs，只豁免本模块）。

#![allow(unsafe_code)]

/// shim 句柄（C 侧 `struct ame_ctx` 的不透明指针）。
#[repr(C)]
pub struct AmeCtx {
    _private: [u8; 0],
}

extern "C" {
    pub fn ame_open(
        path: *const std::os::raw::c_char,
        w: i32,
        h: i32,
        bitrate: i32,
        fps: i32,
        i_frame_interval_s: i32,
        has_audio: i32,
    ) -> *mut AmeCtx;
    pub fn ame_add_audio_track(
        c: *mut AmeCtx,
        asc: *const u8,
        asc_len: i32,
        sample_rate: i32,
        channels: i32,
    ) -> i32;
    pub fn ame_write_video(
        c: *mut AmeCtx,
        y: *const u8,
        u: *const u8,
        v: *const u8,
        pts_us: i64,
        force_keyframe: i32,
    ) -> i32;
    pub fn ame_write_audio(c: *mut AmeCtx, data: *const u8, len: i32, pts_us: i64) -> i32;
    pub fn ame_finish(c: *mut AmeCtx) -> i32;
    pub fn ame_last_error(c: *mut AmeCtx) -> *const std::os::raw::c_char;
    pub fn ame_free(c: *mut AmeCtx);
}
