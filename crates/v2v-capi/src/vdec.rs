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

use std::collections::VecDeque;
use std::os::raw::c_int;

#[cfg(feature = "mp43")]
use v2v_msmpeg4::Version;

use crate::common::*;

// 编号只有一份（在公共层）—— 解码与编码必须认同一张表。
pub use crate::common::{
    CODEC_H264_HW, CODEC_MP42, CODEC_MP43, CODEC_MP4V, CODEC_WMV1, CODEC_WMV2, CODEC_WMV3,
};

/// 解出来的一帧（显示尺寸，YUV420P 三块分开）。
struct Owned {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    /// 显示时间（微秒）。软件解码器一包一帧、顺序就是喂进来的顺序，所以恒为 0；
    /// 硬件解码器会重排（B 帧），必须把这个带出去，否则调用方分不清拿到的是哪一帧。
    pts_us: i64,
}

impl From<v2v_kernel::mc::Frame> for Owned {
    fn from(f: v2v_kernel::mc::Frame) -> Self {
        Self {
            width: f.width,
            height: f.height,
            y: f.y,
            u: f.u,
            v: f.v,
            pts_us: 0,
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
            pts_us: 0,
        }
    }
}

#[cfg(all(feature = "hw", target_os = "macos"))]
impl From<v2v_android::vt::DecodedFrame> for Owned {
    fn from(f: v2v_android::vt::DecodedFrame) -> Self {
        Self {
            width: f.width,
            height: f.height,
            y: f.y,
            u: f.u,
            v: f.v,
            pts_us: f.pts_us,
        }
    }
}

enum Vdec {
    #[cfg(feature = "wmv3")]
    Vc1(v2v_vc1::Decoder),
    /// MS-MPEG-4 的 `decode` 每帧都要宽高 —— 存下来，调用方不用每帧传。
    #[cfg(feature = "mp43")]
    Ms {
        d: v2v_msmpeg4::Decoder,
        w: u32,
        h: u32,
    },
    #[cfg(feature = "mp4v")]
    Mp4v(v2v_mp4v::Decoder),
    /// H.264 走**硬件**解码器（VideoToolbox）。与上面三个的差别不只是「谁在算」：
    /// 它是**异步**的 —— 喂一包可能一帧都不出，也可能一次出好几帧（B 帧重排），
    /// 所以句柄里带一个待发队列（见 `V2vVdec::pending`）。
    #[cfg(all(feature = "hw", target_os = "macos"))]
    H264Hw {
        d: v2v_android::vt::VideoToolboxDecoder,
        /// 容器声明的显示尺寸。解出来的帧必须跟它一致 —— 不一致说明 `extra`
        /// 和样本不是一路的（拿错 avcC 是最常见的成因）。
        w: u32,
        h: u32,
    },
}

/// 视频解码句柄（不透明）。
pub struct V2vVdec {
    dec: Vdec,
    /// 最近一次解出来的帧 —— 出参里的指针指向这里。
    pic: Option<Owned>,
    /// 已解出、还没交出去的帧。软件解码器永远是空的（一包一帧，直接给）；
    /// 硬件解码器靠它接住「一次 feed 吐多帧」和 flush 的尾巴。
    pending: VecDeque<Owned>,
}

/// 一帧解码结果。`planes` 活到本句柄的下一次 `v2v_vdec_decode` / `next` / `close`。
#[repr(C)]
pub struct V2vPicture {
    pub size: u32,
    pub planes: V2vPlanes,
    /// 这一帧的显示时间（微秒）。**新加的字段，追加在尾部** —— 老调用方把
    /// `size` 设成旧的结构体大小，`write_sized` 就只写旧的那部分，行为不变。
    /// 软件解码器恒为 0（一包一帧，顺序 = 喂进来的顺序）；硬件解码器给真值。
    pub pts_us: i64,
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
        let built: Result<Vdec, v2v_kernel::Error> = match codec {
            #[cfg(feature = "wmv3")]
            CODEC_WMV3 => v2v_vc1::SeqHeader::parse(extra)
                .and_then(|seq| v2v_vc1::Decoder::new(seq, width, height))
                .map(Vdec::Vc1),
            #[cfg(feature = "mp43")]
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
            #[cfg(feature = "mp4v")]
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
            #[cfg(not(feature = "wmv3"))]
            CODEC_WMV3 => {
                set_err(KIND_UNSUPPORTED, "这个产物没编进 VC-1（构建时缺 wmv3 feature）");
                return std::ptr::null_mut();
            }
            #[cfg(not(feature = "mp43"))]
            CODEC_MP43 | CODEC_MP42 | CODEC_WMV1 | CODEC_WMV2 => {
                set_err(KIND_UNSUPPORTED, "这个产物没编进 MS-MPEG-4（构建时缺 mp43 feature）");
                return std::ptr::null_mut();
            }
            #[cfg(not(feature = "mp4v"))]
            CODEC_MP4V => {
                set_err(KIND_UNSUPPORTED, "这个产物没编进 MPEG-4 Part 2（构建时缺 mp4v feature）");
                return std::ptr::null_mut();
            }
            #[cfg(all(feature = "hw", target_os = "macos"))]
            CODEC_H264_HW => {
                if extra.is_empty() {
                    set_err(
                        KIND_BAD,
                        "H.264 硬件解码需要 avcC（`v2v_stream_info.extra` 原样传进来）",
                    );
                    return std::ptr::null_mut();
                }
                // 这个分支不走 `built`：VideoToolbox 的错误类型是 `v2v_player::Error`，
                // 和 `v2v_kernel::Error` 不通用 —— 强行套一层会把「avcC 坏了」
                // 误报成「不支持」。
                match v2v_android::vt::VideoToolboxDecoder::open(extra) {
                    Ok(d) => {
                        return Box::into_raw(Box::new(V2vVdec {
                            dec: Vdec::H264Hw {
                                d,
                                w: width,
                                h: height,
                            },
                            pic: None,
                            pending: VecDeque::new(),
                        }))
                    }
                    Err(e) => {
                        rc(&e);
                        return std::ptr::null_mut();
                    }
                }
            }
            #[cfg(not(all(feature = "hw", target_os = "macos")))]
            CODEC_H264_HW => {
                set_err(
                    KIND_UNSUPPORTED,
                    "H.264 硬件解码要 hw feature + macOS（Android 侧还没写）",
                );
                return std::ptr::null_mut();
            }
            other => {
                set_err(KIND_BAD, &format!("未知的编解码器编号 {other}"));
                return std::ptr::null_mut();
            }
        };
        match built {
            Ok(dec) => Box::into_raw(Box::new(V2vVdec {
                dec,
                pic: None,
                pending: VecDeque::new(),
            })),
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
/// **`V2V_CODEC_H264_HW` 不能调这个** —— 硬件解码靠 pts 定显示顺序，喂 0 会
/// 拿到解码顺序（实测：第 1 帧就错到 26.8 dB，画面看着「有东西在动」，
/// 很难发现）。那种情况返回 [`ERR_ARG`] 并指路 [`v2v_vdec_decode_pts`]。
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
    unsafe { decode_impl(h, data, len, None, out) }
}

/// 解一帧，带**显示时间**。软件解码器上 `pts_us` 被忽略（它们一包一帧、
/// 顺序就是喂进来的顺序），硬件解码器上它是必需的。
///
/// 硬件解码器仍然可能一帧都不给（B 帧重排的延迟）—— 那返回 [`EOS`]，
/// 表示「收下了，但暂时没有画面」，**不是**流结束。接着调 [`v2v_vdec_next`]
/// 取可能多出来的帧，收尾别忘了 [`v2v_vdec_flush`]。
///
/// # Safety
/// 同 [`v2v_vdec_decode`]。
#[no_mangle]
pub unsafe extern "C" fn v2v_vdec_decode_pts(
    h: *mut V2vVdec,
    data: *const u8,
    len: u32,
    pts_us: i64,
    out: *mut V2vPicture,
) -> c_int {
    unsafe { decode_impl(h, data, len, Some(pts_us), out) }
}

/// [`v2v_vdec_decode`] / [`v2v_vdec_decode_pts`] 的共同实现。
///
/// `pts` 为 `None` 表示「调用方没给显示时间」—— 只有软件解码器接受这种调用。
unsafe fn decode_impl(
    h: *mut V2vVdec,
    data: *const u8,
    len: u32,
    pts: Option<i64>,
    out: *mut V2vPicture,
) -> c_int {
    // 软件解码器不看 pts（一包一帧，顺序就是喂进来的顺序）—— 只有硬件那条
    // 分支用它。没编进硬件解码时显式吃掉，免得变成 unused 警告。
    #[cfg(not(all(feature = "hw", target_os = "macos")))]
    let _ = pts;
    guard(|| {
        if h.is_null() || out.is_null() {
            return arg_err("句柄或出参是空指针");
        }
        let bytes = match unsafe { slice(data, len) } {
            Ok(b) => b,
            Err(e) => return arg_err(&e),
        };
        let hp = unsafe { &mut *h };
        #[allow(unreachable_patterns)]
        let got: Result<Option<Owned>, v2v_kernel::Error> = match &mut hp.dec {
            #[cfg(feature = "wmv3")]
            Vdec::Vc1(d) => d.decode_frame(bytes).map(|_| Some(Owned::from(d.frame()))),
            #[cfg(feature = "mp43")]
            Vdec::Ms { d, w, h } => d.decode(bytes, *w, *h).map(|f| Some(Owned::from(f))),
            #[cfg(feature = "mp4v")]
            Vdec::Mp4v(d) => d.decode(bytes).map(|f| Some(Owned::from(f))),
            #[cfg(all(feature = "hw", target_os = "macos"))]
            Vdec::H264Hw { d, w, h } => {
                let Some(pts_us) = pts else {
                    set_err(
                        KIND_BAD,
                        "H.264 硬件解码要用 v2v_vdec_decode_pts：喂 0 会拿到解码顺序，不是显示顺序",
                    );
                    return ERR_ARG;
                };
                match d.feed(bytes, pts_us) {
                    Ok(frames) => {
                        for f in frames {
                            if (f.width, f.height) != (*w, *h) {
                                set_err(
                                    KIND_BAD,
                                    &format!(
                                        "解出来的尺寸 {}x{} 与容器声明的 {}x{} 不一致 —— extra 和样本不是一路的？",
                                        f.width, f.height, *w, *h
                                    ),
                                );
                                return ERR_FAILED;
                            }
                            hp.pending.push_back(Owned::from(f));
                        }
                        Ok(None) // 帧已经进队列，由 emit 统一发
                    }
                    Err(e) => {
                        rc(&e);
                        return ERR_FAILED;
                    }
                }
            }
        };
        match got {
            Ok(Some(f)) => {
                hp.pending.push_back(f);
                emit(hp, out)
            }
            Ok(None) => emit(hp, out),
            Err(e) => rc(&e),
        }
    })
}

/// 把队首那一帧写成出参。队列空 → [`EOS`]（对 `decode` 是「这次没解出画面」，
/// 对 `next` 是「暂时没有了」）。
fn emit(hp: &mut V2vVdec, out: *mut V2vPicture) -> c_int {
    let Some(f) = hp.pending.pop_front() else {
        return EOS;
    };
    hp.pic = Some(f);
    let p = hp.pic.as_ref().expect("刚放进去");
    let v = V2vPicture {
        size: std::mem::size_of::<V2vPicture>() as u32,
        planes: planes_of(&p.y, &p.u, &p.v, p.width, p.height),
        pts_us: p.pts_us,
    };
    unsafe { write_sized(out, v) }
}

/// 取一帧已经解好的画面。**只有硬件解码器需要它**（一次喂进去可能吐出好几帧）：
/// 软件解码器一包一帧，帧已经在 `v2v_vdec_decode` 里交出去了，这里恒返回 [`EOS`]。
///
/// [`EOS`] 表示「队列空了」，**不是**流结束 —— 喂下一个样本再来。
///
/// # Safety
/// `h` 必须是 [`v2v_vdec_open`] 的返回值；`out` 的首字段必须是 `size`。
#[no_mangle]
pub unsafe extern "C" fn v2v_vdec_next(h: *mut V2vVdec, out: *mut V2vPicture) -> c_int {
    guard(|| {
        if h.is_null() || out.is_null() {
            return arg_err("句柄或出参是空指针");
        }
        emit(unsafe { &mut *h }, out)
    })
}

/// 告诉解码器「样本喂完了」，并把在途帧全部解出来排进队列。
///
/// 它干的事是**等异步解码收尾**（`VTDecompressionSessionFinishDelayedFrames` +
/// `WaitForAsynchronousFrames`），不是「排空重排缓冲」。
///
/// ⚠️ 实测（macOS 26 + Apple Silicon，45 帧带 B 帧的流）：**不调它也能拿到全部
/// 45 帧** —— 连「一次喂完再抽」那种最该出问题的顺序也一样。所以「不调会少帧」
/// 这个说法**没有证据**，别照着它写判据。留着它是按 VT 的协议来：销毁会话前
/// 必须等异步帧，否则在途的那几帧会被 `Invalidate` 丢掉。
///
/// 软件解码器上是空操作（返回 [`OK`]）：它们没有延迟帧，也没有第二个线程。
///
/// 调完之后照旧用 [`v2v_vdec_next`] 取，直到 [`EOS`]。
///
/// # Safety
/// `h` 必须是 [`v2v_vdec_open`] 的返回值。
#[no_mangle]
pub unsafe extern "C" fn v2v_vdec_flush(h: *mut V2vVdec) -> c_int {
    guard(|| {
        if h.is_null() {
            return arg_err("句柄是空指针");
        }
        let hp = unsafe { &mut *h };
        #[allow(unreachable_patterns)]
        match &mut hp.dec {
            #[cfg(all(feature = "hw", target_os = "macos"))]
            Vdec::H264Hw { d, .. } => match d.flush() {
                Ok(frames) => {
                    hp.pending.extend(frames.into_iter().map(Owned::from));
                    OK
                }
                Err(e) => {
                    rc(&e);
                    ERR_FAILED
                }
            },
            _ => OK,
        }
    })
}

