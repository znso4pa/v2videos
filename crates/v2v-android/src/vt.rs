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

#[repr(C)]
pub struct VtDecCtx {
    _private: [u8; 0],
}

extern "C" {
    fn vt_dec_open(avcc: *const u8, avcc_len: i32) -> *mut VtDecCtx;
    fn vt_dec_feed(c: *mut VtDecCtx, data: *const u8, len: i32, pts_us: i64) -> i32;
    fn vt_dec_next(
        c: *mut VtDecCtx,
        y: *mut *const u8,
        u: *mut *const u8,
        v: *mut *const u8,
        w: *mut i32,
        h: *mut i32,
        pts_us: *mut i64,
    ) -> i32;
    fn vt_dec_flush(c: *mut VtDecCtx) -> i32;
    fn vt_dec_is_hardware(c: *mut VtDecCtx) -> i32;
    fn vt_dec_last_error(c: *mut VtDecCtx) -> *const std::os::raw::c_char;
    fn vt_dec_free(c: *mut VtDecCtx);
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

    /// 只收已就绪的样本（不喂新帧）—— 多 worker 共享会话时用：
    /// **队列是全会话共享的**，drain 出来的样本可能来自任何 worker 喂的帧，
    /// 调用方必须把所有 worker 的收成都汇到同一个收集器（否则样本会按
    /// worker 被复制 —— 实测 369 MB 产物就是这么来的）。
    pub fn drain_ready(&mut self) -> Result<Vec<VtSample>> {
        self.drain()
    }

    /// 把在途帧全部编完、取走样本（`VTCompressionSessionCompleteFrames`）。
    ///
    /// ⚠️ 它**不是 EOS**（会话之后仍可 `encode()`），但**别**拿它当「段边界屏障」
    /// 去复用会话：每段调一次会让 VT 的码控一路飘高 —— 实测 720p 4 Mbps
    /// 70 MB → 365 MB、帧数不变、单帧从 3 KB 爬到 140 KB。段的边界宁可付一次
    /// 会话创建/销毁。
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

// # Safety
// `VTCompressionSession` 允许多线程调用 `VTCompressionSessionEncodeFrame`
// （Apple 文档：session 内部有锁），且本 crate 的所有调用点都用互斥锁
// **串行化**了 encode + drain。跨线程移动句柄因此是安全的。
unsafe impl Send for VideoToolboxEncoder {}

impl Drop for VideoToolboxEncoder {
    fn drop(&mut self) {
        unsafe { vt_free(self.ctx) };
    }
}

/// 一帧解出来的画面：YUV420P 三平面、紧排、行距 = `width`。
pub struct DecodedFrame {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub pts_us: i64,
}

/// macOS 硬件 H.264 解码器（VideoToolbox）。
///
/// 输入口径与 `VideoToolboxEncoder` 的产物、以及 `v2v_mp4` 解封装吐出的样本
/// 完全一致：**avcC + 长度前缀 NALU**（不是 Annex-B），所以这条路径不需要
/// 任何格式转换。
///
/// `!Send`：解码会话的回调跑在 VT 自己的线程上，本类型刻意不声明 `Send`
/// —— 调用点都是单线程「喂一个样本、取一批帧」，没有跨线程移动的需要。
pub struct VideoToolboxDecoder {
    ctx: *mut VtDecCtx,
}

impl VideoToolboxDecoder {
    /// 用 avcC（AVCDecoderConfigurationRecord，含 SPS/PPS）打开解码会话。
    pub fn open(avcc: &[u8]) -> Result<Self> {
        unsafe {
            let ctx = vt_dec_open(avcc.as_ptr(), avcc.len() as i32);
            if ctx.is_null() {
                // 打开失败时 ctx 已被 C 侧释放，错误串取不到了 —— 只能给个兜底。
                return Err(Error::Bad("VideoToolbox 解码会话打开失败".into()));
            }
            Ok(Self { ctx })
        }
    }

    /// 喂一个样本，并收走此刻已就绪的帧。
    ///
    /// ⚠️ 输出**不保证**与输入一一对应：解码器会缓冲（B 帧重排、延迟帧），
    /// 所以 `feed` 可能返回 0 帧，也可能返回多于 1 帧。收尾必须调 `flush`。
    pub fn feed(&mut self, sample: &[u8], pts_us: i64) -> Result<Vec<DecodedFrame>> {
        unsafe {
            if vt_dec_feed(self.ctx, sample.as_ptr(), sample.len() as i32, pts_us) != 0 {
                return Err(Error::Bad(Self::last_err(self.ctx)));
            }
        }
        self.drain()
    }

    /// 发 EOS 并把在途帧全部取出来（带 B 帧的流最后几帧只在这里出）。
    pub fn flush(&mut self) -> Result<Vec<DecodedFrame>> {
        unsafe {
            if vt_dec_flush(self.ctx) != 0 {
                return Err(Error::Bad(Self::last_err(self.ctx)));
            }
        }
        self.drain()
    }

    /// 这个会话实际用的是硬件还是软件解码器。
    ///
    /// `None` = 查不到（设备/系统版本没提供该属性）—— 别把它当成「软解」，
    /// 否则判据会在一台查不到属性的机器上莫名其妙地红。
    pub fn is_hardware(&self) -> Option<bool> {
        match unsafe { vt_dec_is_hardware(self.ctx) } {
            1 => Some(true),
            0 => Some(false),
            _ => None,
        }
    }

    fn drain(&mut self) -> Result<Vec<DecodedFrame>> {
        let mut out = Vec::new();
        loop {
            let (mut y, mut u, mut v) = (
                std::ptr::null::<u8>(),
                std::ptr::null::<u8>(),
                std::ptr::null::<u8>(),
            );
            let (mut w, mut h, mut pts_us) = (0i32, 0i32, 0i64);
            let r = unsafe {
                vt_dec_next(self.ctx, &mut y, &mut u, &mut v, &mut w, &mut h, &mut pts_us)
            };
            if r == 1 {
                break; // 暂无更多
            }
            if r != 0 {
                return Err(Error::Bad(Self::last_err(self.ctx)));
            }
            let (wu, hu) = (w as usize, h as usize);
            let clen = (wu / 2) * (hu / 2);
            // 三个平面各自从 C 侧给的指针拷 —— 不假设它们在内存里紧挨着，
            // 那样 C 侧一改布局这边就静默错位。
            out.push(DecodedFrame {
                y: unsafe { std::slice::from_raw_parts(y, wu * hu) }.to_vec(),
                u: unsafe { std::slice::from_raw_parts(u, clen) }.to_vec(),
                v: unsafe { std::slice::from_raw_parts(v, clen) }.to_vec(),
                width: w as u32,
                height: h as u32,
                pts_us,
            });
        }
        Ok(out)
    }

    fn last_err(ctx: *mut VtDecCtx) -> String {
        unsafe {
            let p = vt_dec_last_error(ctx);
            if p.is_null() {
                return "未知错误".into();
            }
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

impl Drop for VideoToolboxDecoder {
    fn drop(&mut self) {
        unsafe { vt_dec_free(self.ctx) };
    }
}
