//! 视频解码（`v2v_vdec_*`）：VC-1 / MS-MPEG-4 家族 / MPEG-4 Part 2。
//!
//! ## 为什么这四个能共用一套符号
//!
//! 它们的形状本来就一样：**一包字节进，一帧 YUV 出**。
//! `vc1::Decoder::decode_frame` / `msmpeg4::Decoder::decode` / `mp4v::Decoder::decode`
//! 只是参数多少的区别（msmpeg4 每帧还要宽高），不是语义的区别。
//!
//! ## 状态
//!
//! 解码器是**有状态**的：P 帧要参考前一帧。所以句柄必须按显示顺序喂样本、
//! 不能跳帧、不能并发用。要「从某个关键帧重来」，就关掉重开一个句柄
//! （内部的重建网格 / 参考帧没法半途清零 —— 这也是上层 seek 的做法）。
//!
//! ## 不做的事
//!
//! **不做错误隐藏**：解不出来就返回错误，不会给一帧「大概是这样」的画面。
//! 静默按错语法解出来的帧，比明确的失败坏得多。

use std::os::raw::c_int;

use v2v_msmpeg4::Version;

use crate::common::*;

/// VC-1（容器里的 FourCC 是 `WMV3`）。
pub const CODEC_WMV3: u32 = 0;
/// MS-MPEG-4 v3（`MP43`）。
pub const CODEC_MP43: u32 = 1;
/// MS-MPEG-4 v2（`MP42`）。
pub const CODEC_MP42: u32 = 2;
/// WMV7（`WMV1`）。
pub const CODEC_WMV1: u32 = 3;
/// WMV8（`WMV2`）。**必须**给 extradata（能力位在它的前 4 字节里）。
pub const CODEC_WMV2: u32 = 4;
/// MPEG-4 Part 2（`FMP4` / `DIVX` / `XVID`）。extradata 里是 VOL；
/// 传空则按「VOL 在码流里」处理（AVI 常见）。
pub const CODEC_MP4V: u32 = 5;

/// 解出来的一帧（显示尺寸，YUV420P 三块分开）。
struct Owned {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl From<v2v_msmpeg4::Frame> for Owned {
    fn from(f: v2v_msmpeg4::Frame) -> Self {
        Self {
            width: f.width,
            height: f.height,
            y: f.y,
            u: f.u,
            v: f.v,
        }
    }
}

impl From<v2v_vc1::Frame> for Owned {
    fn from(f: v2v_vc1::Frame) -> Self {
        Self {
            width: f.width,
            height: f.height,
            y: f.y,
            u: f.u,
            v: f.v,
        }
    }
}

enum Vdec {
    Vc1(v2v_vc1::Decoder),
    /// MS-MPEG-4 的 `decode` 每帧都要宽高 —— 存下来，调用方不用每帧传。
    Ms {
        d: v2v_msmpeg4::Decoder,
        w: u32,
        h: u32,
    },
    Mp4v(v2v_mp4v::Decoder),
}

/// 视频解码句柄（不透明）。
pub struct V2vVdec {
    dec: Vdec,
    /// 最近一次解出来的帧 —— 出参里的指针指向这里。
    pic: Option<Owned>,
}

/// 一帧解码结果。`planes` 活到本句柄的下一次 `v2v_vdec_decode` / `close`。
#[repr(C)]
pub struct V2vPicture {
    pub size: u32,
    pub planes: V2vPlanes,
}

/// 建一个解码器。失败返回空指针。
///
/// `extra` 是容器里的编解码器私有数据（`v2v_stream_info.extra` 原样传进来）。
/// `width` / `height` 用**显示尺寸**（`v2v_stream_info.width/height`），不是编码尺寸。
///
/// # Safety
/// `extra` 要么为空，要么指向 `extra_len` 字节可读内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_vdec_open(
    codec: u32,
    extra: *const u8,
    extra_len: u32,
    width: u32,
    height: u32,
) -> *mut V2vVdec {
    guard_ptr(|| {
        let extra = match unsafe { slice(extra, extra_len) } {
            Ok(b) => b,
            Err(e) => {
                set_err(KIND_BAD, &e);
                return std::ptr::null_mut();
            }
        };
        if width == 0 || height == 0 {
            set_err(KIND_BAD, "宽高不能为 0");
            return std::ptr::null_mut();
        }
        let built: Result<Vdec, v2v_msmpeg4::Error> = match codec {
            CODEC_WMV3 => v2v_vc1::SeqHeader::parse(extra)
                .and_then(|seq| v2v_vc1::Decoder::new(seq, width, height))
                .map(Vdec::Vc1),
            CODEC_MP43 | CODEC_MP42 | CODEC_WMV1 | CODEC_WMV2 => {
                let version = match codec {
                    CODEC_MP43 => Version::V3,
                    CODEC_MP42 => Version::V2,
                    CODEC_WMV1 => Version::Wmv1,
                    _ => Version::Wmv2,
                };
                let d = if codec == CODEC_WMV2 {
                    v2v_msmpeg4::Decoder::with_wmv2(extra)
                } else {
                    Ok(v2v_msmpeg4::Decoder::with_version(version))
                };
                d.map(|d| Vdec::Ms {
                    d,
                    w: width,
                    h: height,
                })
            }
            CODEC_MP4V => {
                if extra.is_empty() {
                    // AVI 的 `strf` 里常常没有 VOL —— 那就按「VOL 在码流里」来。
                    Ok(Vdec::Mp4v(v2v_mp4v::Decoder::new_from_stream()))
                } else {
                    v2v_mp4v::Vol::parse(extra)
                        .and_then(v2v_mp4v::Decoder::new)
                        .map(Vdec::Mp4v)
                }
            }
            other => {
                set_err(KIND_BAD, &format!("未知的编解码器编号 {other}"));
                return std::ptr::null_mut();
            }
        };
        match built {
            Ok(dec) => Box::into_raw(Box::new(V2vVdec { dec, pic: None })),
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
/// `h` 必须是 [`v2v_vdec_open`] 的返回值，且只能关一次。
#[no_mangle]
pub unsafe extern "C" fn v2v_vdec_close(h: *mut V2vVdec) {
    if h.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(h) });
}

/// 解一帧。`data` 是容器里的一个样本（裸帧数据，帧头在最前面）。
///
/// # Safety
/// `data` 必须指向 `len` 字节可读内存；`out` 的首字段必须是 `size`。
#[no_mangle]
pub unsafe extern "C" fn v2v_vdec_decode(
    h: *mut V2vVdec,
    data: *const u8,
    len: u32,
    out: *mut V2vPicture,
) -> c_int {
    guard(|| {
        if h.is_null() || out.is_null() {
            return arg_err("句柄或出参是空指针");
        }
        let bytes = match unsafe { slice(data, len) } {
            Ok(b) => b,
            Err(e) => return arg_err(&e),
        };
        let hp = unsafe { &mut *h };
        let got: Result<Owned, v2v_msmpeg4::Error> = match &mut hp.dec {
            Vdec::Vc1(d) => d.decode_frame(bytes).map(|_| Owned::from(d.frame())),
            Vdec::Ms { d, w, h } => d.decode(bytes, *w, *h).map(Owned::from),
            Vdec::Mp4v(d) => d.decode(bytes).map(Owned::from),
        };
        match got {
            Ok(f) => {
                hp.pic = Some(f);
                let p = hp.pic.as_ref().expect("刚放进去");
                let v = V2vPicture {
                    size: std::mem::size_of::<V2vPicture>() as u32,
                    planes: planes_of(&p.y, &p.u, &p.v, p.width, p.height),
                };
                unsafe { write_sized(out, v) };
                OK
            }
            Err(e) => rc(&e),
        }
    })
}
