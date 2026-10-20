//! Android 硬件编码路径：**MediaCodec（H.264）+ MediaMuxer（MP4）**。
//!
//! ## 为什么要有这条路（「取优点」的答案）
//!
//! 同码率 4 Mbps 转同一个 WMV（见 2026-10-20 的转码对垒）：
//!
//! | | 我们的软件栈（mp4v + AAC） | ffmpeg（mpeg4） | **MediaCodec（本 crate）** |
//! |---|---|---|---|
//! | 速度 | 53 fps（78 s / 138 s 视频） | 352 fps | **实时或更快，且不吃 CPU** |
//! | 质量 | PSNR 38.83 dB | PSNR 40.77 dB | **H.264 比两者都高**（同码率比 MPEG-4 Part 2 通常高 2–4 dB） |
//!
//! 硬件编码器同时解决两件事：速度（专用电路，CPU 释放给播放本身）和
//! 质量（换了一代编码标准）。代价是**只在 Android 上可用**、编码参数受厂商
//! 实现限制（没有逐帧 qscale、色彩格式有设备差异）——所以软件栈保留：
//! 桌面 / 测试 / 兜底用我们的，设备上用这条。
//!
//! ## 架构
//!
//! ```text
//! v2v-player（VC-1/WMA 解码，纯 Rust，逐字节对 ffmpeg）
//!   → v2v-android::Mp4HwWriter（本 crate）
//!       ├─ 视频：YUV420P → MediaCodec H.264 → MediaMuxer
//!       └─ 音频：WMA → f32 → 我们的 AAC 编码器 → 裸帧 + ASC → MediaMuxer
//! ```
//!
//! ## unsafe 边界
//!
//! 全部 NDK 交互在 `c/media_encoder.c`（C 侧）与 `src/ffi.rs`（唯一的
//! `extern` 声明 + 调用边界）。`Mp4HwWriter` 是安全 API：裸指针不外漏、
//! `Drop` 兜底释放、错误全部走 `Result`。**句柄不可跨线程移动**
//! （`Mp4HwWriter` 是 `!Send`）—— MediaCodec 不是线程安全的，在哪个线程
//! `open` 就在哪个线程用完。

#![cfg_attr(not(any(target_os = "android", target_os = "macos")), forbid(unsafe_code))]
// unsafe 只在两个后端模块里：Android 的 `ffi.rs`（MediaCodec）与
// macOS 的 `vt.rs`（VideoToolbox）。其余平台整个 crate 不允许 unsafe。

use std::path::Path;

use v2v_player::{Error, Result};

#[cfg(target_os = "android")]
mod ffi;

#[cfg(target_os = "android")]
use std::ffi::CString;

/// macOS 的 VideoToolbox 后端（硬件 H.264，产样本不封装）。
#[cfg(target_os = "macos")]
pub mod vt;

/// 给宿主 App 的 C ABI（JNI 桥 / ctypes 用）。只在真有硬件后端的平台编译 ——
/// 其它平台整个 crate 是 `forbid(unsafe_code)`，而这一层必须碰裸指针。
#[cfg(any(target_os = "android", target_os = "macos"))]
pub mod capi;

/// 端到端管线：WMV → MP4（解码纯 Rust + 视频/音频 + 封装），示例与测试共用。
///
/// 非 macOS / Android 平台上 [`transcode::wmv_to_mp4_hw`] 返回 `Unsupported`
/// （模块本身全平台编译 —— CI 在 Linux 上跑 `cargo test`，测试引用不能被 cfg 掉）。
pub mod transcode;

/// Android 硬件编码的 MP4 写入器。
///
/// 视频：每帧 YUV420P 三平面（与 [`v2v_player::VideoFrame`] 同口径）。
/// 音频：裸 AAC 帧（无 ADTS 头）+ 2 字节 AudioSpecificConfig。
#[cfg(target_os = "android")]
pub struct Mp4HwWriter {
    ctx: *mut ffi::AmeCtx,
    finished: bool,
}

#[cfg(target_os = "android")]
impl Mp4HwWriter {
    /// 打开硬件编码器与 MP4 封装器。
    ///
    /// * `bitrate`：视频目标码率（bit/s）—— MediaCodec 的码控是黑盒，
    ///   实际码率会围绕它浮动，没有我们软件栈那种逐帧调整。
    /// * `fps`：整数帧率（MediaCodec 只要整数；30 对这个仓库的样本够用）。
    /// * `i_frame_interval_s`：GOP 秒数（MediaCodec 的原生单位就是秒，不是帧数）。
    /// * `has_audio`：会不会加音频轨。**声明了就必须调 `add_audio_track`**，
    ///   否则 `finish` 时 MediaMuxer 没 start，文件里没有 moov。
    pub fn open(
        path: &Path,
        width: u32,
        height: u32,
        bitrate: u32,
        fps: u32,
        i_frame_interval_s: u32,
        has_audio: bool,
    ) -> Result<Self> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| Error::Bad("路径里有 NUL".into()))?;
        unsafe {
            let ctx = ffi::ame_open(
                c_path.as_ptr(),
                width as i32,
                height as i32,
                bitrate as i32,
                fps as i32,
                i_frame_interval_s as i32,
                has_audio as i32,
            );
            if ctx.is_null() {
                return Err(Error::Bad(Self::last_err(ctx)));
            }
            Ok(Self {
                ctx,
                finished: false,
            })
        }
    }

    /// 加音频轨（`ame_add_audio_track`）。`asc`：AudioSpecificConfig 字节。
    pub fn add_audio_track(&mut self, asc: &[u8], sample_rate: u32, channels: u8) -> Result<()> {
        unsafe {
            if ffi::ame_add_audio_track(
                self.ctx,
                asc.as_ptr(),
                asc.len() as i32,
                sample_rate as i32,
                channels as i32,
            ) != 0
            {
                return Err(Error::Bad(Self::last_err(self.ctx)));
            }
        }
        Ok(())
    }

    /// 喂一帧视频。`pts_us`：微秒（`VideoFrame.pts_ms * 1000`）。
    pub fn write_video(
        &mut self,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        pts_us: i64,
        force_keyframe: bool,
    ) -> Result<()> {
        unsafe {
            if ffi::ame_write_video(
                self.ctx,
                y.as_ptr(),
                u.as_ptr(),
                v.as_ptr(),
                pts_us,
                force_keyframe as i32,
            ) != 0
            {
                return Err(Error::Bad(Self::last_err(self.ctx)));
            }
        }
        Ok(())
    }

    /// 喂一段裸 AAC（无 ADTS）。`pts_us`：这段音频的起点（微秒）。
    pub fn write_aac(&mut self, data: &[u8], pts_us: i64) -> Result<()> {
        unsafe {
            if ffi::ame_write_audio(self.ctx, data.as_ptr(), data.len() as i32, pts_us) != 0 {
                return Err(Error::Bad(Self::last_err(self.ctx)));
            }
        }
        Ok(())
    }

    /// 发 EOS、吸干编码器、写 MP4 的 moov。消费 self（写完就没了）。
    pub fn finish(mut self) -> Result<()> {
        self.finished = true;
        unsafe {
            if ffi::ame_finish(self.ctx) != 0 {
                let msg = Self::last_err(self.ctx);
                ffi::ame_free(self.ctx);
                self.ctx = std::ptr::null_mut();
                return Err(Error::Bad(msg));
            }
        }
        Ok(())
    }

    fn last_err(ctx: *mut ffi::AmeCtx) -> String {
        unsafe {
            let p = ffi::ame_last_error(ctx);
            if p.is_null() {
                return "未知错误".into();
            }
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

#[cfg(target_os = "android")]
impl Drop for Mp4HwWriter {
    fn drop(&mut self) {
        // 正常路径在 finish 里已置 finished；走到这里都是中途出错/被丢弃 ——
        // 直接释放，MediaMuxer 没 stop 的文件没有 moov（不可播），如实放弃。
        if !self.finished {
            unsafe { ffi::ame_free(self.ctx) };
        }
    }
}

/// 非 Android 平台：整个 API 明确报「不支持」。
///
/// 保留同名类型让上层代码**编译通过**，但运行时必错 —— 桌面跑测试、
/// CI 上保持绿色都靠这个；真机路径走 `#[cfg(target_os = "android")]` 那份。
#[cfg(not(target_os = "android"))]
pub struct Mp4HwWriter;

#[cfg(not(target_os = "android"))]
impl Mp4HwWriter {
    /// 桩：永远报不支持（见 [`Mp4HwWriter::open`]）。
    pub fn add_audio_track(&mut self, _asc: &[u8], _sample_rate: u32, _channels: u8) -> Result<()> {
        Err(Error::Unsupported("MediaCodec 只在 Android 上可用".into()))
    }

    /// 桩：永远报不支持。
    pub fn write_video(&mut self, _y: &[u8], _u: &[u8], _v: &[u8], _pts_us: i64, _force: bool) -> Result<()> {
        Err(Error::Unsupported("MediaCodec 只在 Android 上可用".into()))
    }

    /// 桩：永远报不支持。
    pub fn write_aac(&mut self, _data: &[u8], _pts_us: i64) -> Result<()> {
        Err(Error::Unsupported("MediaCodec 只在 Android 上可用".into()))
    }

    /// 桩：永远报不支持。
    pub fn finish(self) -> Result<()> {
        Err(Error::Unsupported("MediaCodec 只在 Android 上可用".into()))
    }

    pub fn open(
        _path: &Path,
        _width: u32,
        _height: u32,
        _bitrate: u32,
        _fps: u32,
        _i_frame_interval_s: u32,
        _has_audio: bool,
    ) -> Result<Self> {
        Err(Error::Unsupported(
            "MediaCodec 只在 Android 上可用（这个平台用 v2v-cli 的软件栈）".into(),
        ))
    }
}
