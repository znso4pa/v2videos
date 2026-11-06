//! 视频编码（`v2v_venc_*`）：MP43 / MPEG-4 Part 2。
//!
//! ## GOP 由**调用方**决定，ABI 不猜
//!
//! 两个编码器内部都有 `keyframe_interval`，但这里**一律关掉**（传 0），
//! 改成「调用方说这一帧是不是关键帧」：
//!
//! - `force_key != 0` → 调 `encode_i_frame`，输出一定标成关键帧；
//! - 否则 → 调 `encode_frame`，输出一定不是关键帧（首帧由调用方自己标）。
//!
//! 理由：封装器要写载荷的关键帧位，而这个位必须与**实际编出来的东西**一致。
//! 让 ABI 去复刻编码器内部那套「距上个 I 帧多远」的判断，迟早会漂 ——
//! 实测过这类 off-by-one（12 帧 / GOP 5 变成 0、6 两个关键帧）。
//! 现在只有一个来源：调用方的 `force_key`。
//!
//! ## 输入帧会被拷一份
//!
//! 编码器的 `Frame` 持有 `Vec`，没有借用形态，所以这里必须拷。
//! 720p 一帧 ≈ 1.4 MB —— 相对编码本身的代价可以忽略。

use std::os::raw::c_int;

use v2v_kernel::mc::Frame;

use crate::common::*;

// 编号只有一份（在公共层）—— 解码与编码必须认同一张表。
pub use crate::common::{CODEC_MP43, CODEC_MP4V};

/// 建编码器要的参数。
///
/// 没用到的字段会被忽略（MP43 不看 `fps_num` / `ac_pred` / `mpeg_quant`）。
#[repr(C)]
pub struct V2vVencCfg {
    pub size: u32,
    pub width: u32,
    pub height: u32,
    /// 量化步长 1..=31。**0 = 用默认值**（MP43 是 8，MPEG-4 是 4）。
    pub qscale: u32,
    /// 目标码率（bit/s）。**0 = 不做码率控制**（整段固定 qscale）。
    /// 只有 MPEG-4 支持码率控制。
    pub bitrate: u32,
    /// MPEG-4 的时间分辨率（写进 VOL）。**0 = 25**。
    pub fps_num: u32,
    /// MPEG-4：逐宏块试 AC 预测。默认开。
    pub ac_pred: u32,
    /// MPEG-4：矩阵量化。**注意**：本仓的解码器只做 `quant_type = 0`，
    /// 传 1 编出来的流本仓自己解不了（外部播放器不受影响）。
    pub mpeg_quant: u32,
}

enum Venc {
    #[cfg(feature = "mp43")]
    Ms(v2v_msmpeg4::Encoder),
    #[cfg(feature = "mp4v")]
    Mp4v(v2v_mp4v::Encoder),
}

/// 视频编码句柄（不透明）。
pub struct V2vVenc {
    enc: Venc,
    width: u32,
    height: u32,
    /// 编过的帧数（判首帧用）。
    frames: u64,
    /// 最近一次输出的码流 —— 出参的指针指向这里。
    out: Vec<u8>,
    /// 封装器要的「解码器专用信息」（MPEG-4 是 VOL；MP43 没有）。
    header: Vec<u8>,
}

fn take_planes(p: &V2vPlanes) -> Result<Frame, String> {
    if p.y.is_null() || p.u.is_null() || p.v.is_null() {
        return Err("平面指针是空指针".into());
    }
    if p.width == 0 || p.height == 0 {
        return Err("宽高不能为 0".into());
    }
    let (w, h) = (p.width as usize, p.height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    // stride 必须等于 width —— 这是本仓的输出口径，不是可选项。
    if p.stride_y != p.width || p.stride_u != p.width.div_ceil(2) || p.stride_v != p.width.div_ceil(2)
    {
        return Err("stride 与宽高对不上（本仓只接受紧排平面）".into());
    }
    let y = unsafe { std::slice::from_raw_parts(p.y, w * h) }.to_vec();
    let u = unsafe { std::slice::from_raw_parts(p.u, cw * ch) }.to_vec();
    let v = unsafe { std::slice::from_raw_parts(p.v, cw * ch) }.to_vec();
    Ok(Frame {
        width: p.width,
        height: p.height,
        y,
        u,
        v,
    })
}

/// 建编码器。失败返回空指针。
///
/// # Safety
/// `cfg` 必须指向一块首字段是 `size` 的可读内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_venc_open(codec: u32, cfg: *const V2vVencCfg) -> *mut V2vVenc {
    guard_ptr(|| {
        if cfg.is_null() {
            set_err(KIND_BAD, "参数是空指针");
            return std::ptr::null_mut();
        }
        let cfg = unsafe { &*cfg };
        if (cfg.size as usize) < std::mem::size_of::<V2vVencCfg>() {
            set_err(KIND_BAD, "cfg->size 太小，可能不是本版本的调用方");
            return std::ptr::null_mut();
        }
        if cfg.width == 0 || cfg.height == 0 {
            set_err(KIND_BAD, "宽高不能为 0");
            return std::ptr::null_mut();
        }
        let qscale = if cfg.qscale == 0 {
            None
        } else if (1..=31).contains(&cfg.qscale) {
            Some(cfg.qscale as u8)
        } else {
            set_err(KIND_BAD, "qscale 必须在 1..=31（0 表示用默认值）");
            return std::ptr::null_mut();
        };

        #[allow(unreachable_patterns)]
        let built: Result<Venc, v2v_kernel::Error> = match codec {
            #[cfg(feature = "mp43")]
            CODEC_MP43 => {
                let mut ec = v2v_msmpeg4::EncoderConfig::default();
                if let Some(q) = qscale {
                    ec.qscale = q;
                }
                // GOP 由调用方驱动，见模块文档。
                ec.keyframe_interval = 0;
                Ok(Venc::Ms(v2v_msmpeg4::Encoder::new(ec)))
            }
            #[cfg(feature = "mp4v")]
            CODEC_MP4V => {
                let mut ec = v2v_mp4v::EncoderConfig {
                    width: cfg.width,
                    height: cfg.height,
                    ..Default::default()
                };
                if let Some(q) = qscale {
                    ec.qscale = q;
                }
                if cfg.bitrate > 0 {
                    ec.bitrate = Some(cfg.bitrate);
                }
                if cfg.fps_num > 0 {
                    ec.fps_num = cfg.fps_num;
                }
                ec.ac_pred = cfg.ac_pred != 0;
                ec.mpeg_quant = cfg.mpeg_quant != 0;
                ec.keyframe_interval = 0;
                v2v_mp4v::Encoder::new(ec).map(Venc::Mp4v)
            }
            other => {
                set_err(KIND_BAD, &format!("未知的编解码器编号 {other}"));
                return std::ptr::null_mut();
            }
        };
        match built {
            Ok(enc) => {
                let header = match &enc {
                    // MP4 的 esds 要这个；ASF 不需要（它的流属性里没有 extradata）。
                    #[cfg(feature = "mp4v")]
                    Venc::Mp4v(e) => e.decoder_specific_info(),
                    #[cfg(feature = "mp43")]
                    Venc::Ms(_) => Vec::new(),
                };
                Box::into_raw(Box::new(V2vVenc {
                    enc,
                    width: cfg.width,
                    height: cfg.height,
                    frames: 0,
                    out: Vec::new(),
                    header,
                }))
            }
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
/// `h` 必须是 [`v2v_venc_open`] 的返回值，且只能关一次。
#[no_mangle]
pub unsafe extern "C" fn v2v_venc_close(h: *mut V2vVenc) {
    if h.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(h) });
}

/// 编一帧。`force_key != 0` 表示这一帧要编成关键帧（I 帧）。
///
/// 首帧请传 `force_key = 1`（也可以传 0 —— 这里会当成关键帧处理）。
///
/// # Safety
/// `in` 指向一块可读的 `v2v_planes`；`out` 的首字段必须是 `size`。
#[no_mangle]
pub unsafe extern "C" fn v2v_venc_encode(
    h: *mut V2vVenc,
    planes: *const V2vPlanes,
    force_key: c_int,
    out: *mut V2vPacket,
) -> c_int {
    guard(|| {
        if h.is_null() || planes.is_null() || out.is_null() {
            return arg_err("句柄、输入帧或出参是空指针");
        }
        let hp = unsafe { &mut *h };
        let frame = match take_planes(unsafe { &*planes }) {
            Ok(f) => f,
            Err(e) => return arg_err(&e),
        };
        if frame.width != hp.width || frame.height != hp.height {
            return arg_err(&format!(
                "帧尺寸 {}x{} 与配置 {}x{} 不一致",
                frame.width, frame.height, hp.width, hp.height
            ));
        }
        let want_i = force_key != 0 || hp.frames == 0;
        let got: Result<Vec<u8>, v2v_kernel::Error> = match &mut hp.enc {
            #[cfg(feature = "mp43")]
            Venc::Ms(e) => {
                if want_i {
                    e.encode_i_frame(&frame)
                } else {
                    e.encode_frame(&frame)
                }
            }
            #[cfg(feature = "mp4v")]
            Venc::Mp4v(e) => {
                if want_i {
                    e.encode_i_frame(&frame)
                } else {
                    e.encode_frame(&frame)
                }
            }
        };
        match got {
            Ok(bytes) => {
                hp.frames += 1;
                hp.out = bytes;
                let v = V2vPacket {
                    size: std::mem::size_of::<V2vPacket>() as u32,
                    data: hp.out.as_ptr(),
                    len: hp.out.len() as u32,
                    key_frame: i32::from(want_i),
                };
                unsafe { write_sized(out, v) }
            }
            Err(e) => rc(&e),
        }
    })
}

/// 把「解码器专用信息」拷进 `buf`，返回字节数；`cap` 不够返回 `V2V_ERR_ARG`。
///
/// MPEG-4 返回的是 VOL（含 `00 00 01 B0` 起始码），封装 MP4 的 `esds` 要它；
/// MP43 返回 0 字节（ASF 不需要 extradata）。
///
/// # Safety
/// `buf` 要么为空（`cap` 必须为 0），要么指向 `cap` 字节可写内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_venc_header(h: *const V2vVenc, buf: *mut u8, cap: u32) -> c_int {
    guard(|| {
        if h.is_null() {
            return arg_err("句柄是空指针");
        }
        let hp = unsafe { &*h };
        let n = hp.header.len();
        if (cap as usize) < n {
            return arg_err(&format!("缓冲太小：要 {n} 字节，给了 {cap}"));
        }
        if n > 0 && buf.is_null() {
            return arg_err("缓冲是空指针");
        }
        if n > 0 {
            unsafe { std::ptr::copy_nonoverlapping(hp.header.as_ptr(), buf, n) };
        }
        n as c_int
    })
}
