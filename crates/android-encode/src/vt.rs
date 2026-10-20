//! VideoToolbox 后端（macOS）：安全封装 + FFI 声明。
//!
//! 与 MediaCodec 后端的差别：这里**只产样本**（长度前缀 NALU + avcC），
//! MP4 封装交给 Rust 侧的 `v2v_mp4`（`VideoCodec::Avc1`）—— 两个后端
//! 产物同构，验收口径一致。

#![allow(unsafe_code)]

use v2v_player::{Error, Result};

#[repr(C)]
pub struct VtCtx {
    _private: [u8; 0],
}

extern "C" {
    fn vt_open(w: i32, h: i32, bitrate: i32, fps: i32, gop_seconds: i32) -> *mut VtCtx;
    fn vt_encode(
        c: *mut VtCtx,
        y: *const u8,
        u: *const u8,
        v: *const u8,
        pts_us: i64,
        force_keyframe: i32,
    ) -> i32;
    fn vt_next(c: *mut VtCtx, data: *mut *mut u8, len: *mut i32, pts_us: *mut i64, key: *mut i32)
        -> i32;
    fn vt_avcc(c: *mut VtCtx, data: *mut *mut u8, len: *mut i32) -> i32;
    fn vt_finish(c: *mut VtCtx) -> i32;
    fn vt_last_error(c: *mut VtCtx) -> *const std::os::raw::c_char;
    fn vt_free(c: *mut VtCtx);
}

/// 一个编码出来的 H.264 样本（长度前缀 NALU，非 Annex-B）。
pub struct VtSample {
    pub data: Vec<u8>,
    pub pts_us: i64,
    pub key_frame: bool,
}

/// macOS 硬件 H.264 编码器（VideoToolbox）。
///
/// `!Send`：VTCompressionSession 不是线程安全的，哪个线程 `open`
/// 就在哪个线程用完。质量优先：`RealTime = false`（离线模式）、无 B 帧
/// （输出顺序 = 输入顺序）。
pub struct VideoToolboxEncoder {
    ctx: *mut VtCtx,
    /// 编出来的 avcC（SPS/PPS）—— 第一帧输出后可取。
    avcc: Option<Vec<u8>>,
}

impl VideoToolboxEncoder {
    /// 打开硬件编码器。`gop_seconds`：GOP 秒数（VT 的原生单位）。
    pub fn open(width: u32, height: u32, bitrate: u32, fps: u32, gop_seconds: u32) -> Result<Self> {
        unsafe {
            let ctx = vt_open(
                width as i32,
                height as i32,
                bitrate as i32,
                fps as i32,
                gop_seconds as i32,
            );
            if ctx.is_null() {
                return Err(Error::Bad(Self::last_err(ctx)));
            }
            Ok(Self { ctx, avcc: None })
        }
    }

    /// 喂一帧 YUV420P（三平面，行距 = width），并收走此刻已就绪的样本。
    pub fn encode(
        &mut self,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        pts_us: i64,
        force_keyframe: bool,
    ) -> Result<Vec<VtSample>> {
        unsafe {
            if vt_encode(self.ctx, y.as_ptr(), u.as_ptr(), v.as_ptr(), pts_us, force_keyframe as i32)
                != 0
            {
                return Err(Error::Bad(Self::last_err(self.ctx)));
            }
        }
        self.drain()
    }

    /// 发 EOS 并取走全部在途样本。之后句柄只能 Drop（样本仍可用 `avcc()` 取）。
    pub fn finish(&mut self) -> Result<Vec<VtSample>> {
        unsafe {
            if vt_finish(self.ctx) != 0 {
                return Err(Error::Bad(Self::last_err(self.ctx)));
            }
        }
        self.drain()
    }

    /// avcC（AVCDecoderConfigurationRecord）。没有样本输出前是 `None`。
    pub fn avcc(&self) -> Option<&[u8]> {
        self.avcc.as_deref()
    }

    fn drain(&mut self) -> Result<Vec<VtSample>> {
        let mut out = Vec::new();
        loop {
            let (mut data, mut len, mut pts_us, mut key) =
                (std::ptr::null_mut::<u8>(), 0i32, 0i64, 0i32);
            let r = unsafe { vt_next(self.ctx, &mut data, &mut len, &mut pts_us, &mut key) };
            if r == 1 {
                break;
            }
            if r != 0 {
                return Err(Error::Bad(Self::last_err(self.ctx)));
            }
            let bytes = unsafe { std::slice::from_raw_parts(data, len as usize) }.to_vec();
            if self.avcc.is_none() {
                // 第一帧出来时 avcC 一定已就绪（同一回调路径先于样本送达）。
                let (mut p, mut n) = (std::ptr::null_mut::<u8>(), 0i32);
                if unsafe { vt_avcc(self.ctx, &mut p, &mut n) } == 0 {
                    self.avcc = Some(
                        unsafe { std::slice::from_raw_parts(p, n as usize) }.to_vec(),
                    );
                }
            }
            out.push(VtSample {
                data: bytes,
                pts_us,
                key_frame: key != 0,
            });
        }
        Ok(out)
    }

    fn last_err(ctx: *mut VtCtx) -> String {
        unsafe {
            let p = vt_last_error(ctx);
            if p.is_null() {
                return "未知错误".into();
            }
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

impl Drop for VideoToolboxEncoder {
    fn drop(&mut self) {
        unsafe { vt_free(self.ctx) };
    }
}
