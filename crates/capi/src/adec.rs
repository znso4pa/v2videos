//! 音频解码（`v2v_adec_*`）：WMA v1 / v2。
//!
//! ## 为什么是「喂一个载荷 + 逐个取帧」两件事
//!
//! ASF 的一个音频载荷里常常装着**好几个** WMA 帧（superframe），而解码器一次
//! 吐一串。硬做成「一次调用 = 一帧」会让调用方要么丢帧、要么自己攒 ——
//! 实测过「一次只交一个、剩下直接丢」的写法：音频只剩四分之一，还不报错。
//!
//! 所以接口分成两个：`feed` 喂一个载荷（结果入队），`next` 从队列里取一帧。
//! `next` 返回 [`EOS`] 只表示**队列空了**（该喂下一个载荷），不是流结束。
//!
//! ## 编码器延迟
//!
//! WMA 编码器开头灌了两帧静音（`frame_len × 2` 个采样点）。ffmpeg 的解码器
//! 会把它丢掉；不丢的话音频整体晚约 90 ms，与视频对不上。
//! `drop_priming = 1` 就是干这个的 —— 默认建议开（与播放层一致）。
//!
//! ## PTS
//!
//! 按**已交出**的样本数推出来（`样本数 × 1000 / 采样率`），不是容器时间戳 ——
//! 一个载荷装着好几帧，容器的时间戳粒度太粗。

use std::collections::VecDeque;
use std::os::raw::c_int;

use crate::common::*;

/// 建解码器要的参数（与 `v2v_stream_info` 的音频字段一一对应）。
#[repr(C)]
pub struct V2vAdecCfg {
    pub size: u32,
    /// `wFormatTag`：`0x0160` = WMA v1，`0x0161` = WMA v2。其它值报不支持。
    pub format_tag: u32,
    pub channels: u32,
    pub sample_rate: u32,
    /// 容器里的 `avg_bytes_per_sec`（库内部 ×8 换成 bit/s）。
    pub avg_bytes_per_sec: u32,
    pub block_align: u32,
    /// 编解码器私有数据（`v2v_stream_info.extra` 原样传）。
    pub extra: *const u8,
    pub extra_len: u32,
    /// 1 = 丢掉编码器开头的静音（与 ffmpeg / 播放层一致）；0 = 原样交出。
    pub drop_priming: u32,
}

/// 音频解码句柄（不透明）。
pub struct V2vAdec {
    decoder: v2v_wma::WmaDecoder,
    /// 解出来、还没交出去的帧。
    pending: VecDeque<Vec<Vec<f32>>>,
    /// 最近一次交出去的那帧（出参的指针指向这里）。
    cur: Option<Vec<Vec<f32>>>,
    /// 平面指针表（出参的 `planes` 指向这里）。
    ptrs: [*const f32; MAX_CHANNELS],
    channels: usize,
    sample_rate: u32,
    /// 已交出的每声道样本数（PTS 的基准）。
    emitted: u64,
    /// 还要丢多少个每声道样本（编码器延迟）。
    skip_left: usize,
}

/// 一帧音频。`planes` 指向 `channels` 个指针，每条 `samples` 个样本，
/// 活到本句柄的下一次 `v2v_adec_next` / `close`。
///
/// 刻度同 int16（不是 `[-1, 1)`），并且**可能超出 ±32768** —— 转 int16 时自己 clamp。
#[repr(C)]
pub struct V2vAdecFrame {
    pub size: u32,
    pub planes: *const *const f32,
    pub channels: u32,
    pub samples: u32,
    pub sample_rate: u32,
    /// 按已交出的样本数推出来的（ms）。
    pub pts_ms: i64,
}

/// 建解码器。失败返回空指针。
///
/// # Safety
/// `cfg` 必须指向一块首字段是 `size` 的可读内存；`cfg->extra` 要么为空，
/// 要么指向 `extra_len` 字节可读内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_adec_open(cfg: *const V2vAdecCfg) -> *mut V2vAdec {
    guard_ptr(|| {
        if cfg.is_null() {
            set_err(KIND_BAD, "参数是空指针");
            return std::ptr::null_mut();
        }
        let cfg = unsafe { &*cfg };
        if (cfg.size as usize) < std::mem::size_of::<V2vAdecCfg>() {
            set_err(KIND_BAD, "cfg->size 太小，可能不是本版本的调用方");
            return std::ptr::null_mut();
        }
        let extra = match unsafe { slice(cfg.extra, cfg.extra_len) } {
            Ok(b) => b,
            Err(e) => {
                set_err(KIND_BAD, &e);
                return std::ptr::null_mut();
            }
        };
        let version = match cfg.format_tag {
            0x0160 => 1u8,
            0x0161 => 2u8,
            other => {
                set_err(
                    KIND_UNSUPPORTED,
                    &format!("音频 format_tag 0x{other:04X}，这里只做 WMA v1/v2"),
                );
                return std::ptr::null_mut();
            }
        };
        if cfg.channels as usize > MAX_CHANNELS {
            set_err(
                KIND_UNSUPPORTED,
                &format!("{} 条声道，ABI 最多 {MAX_CHANNELS}", cfg.channels),
            );
            return std::ptr::null_mut();
        }
        let params = match v2v_wma::WmaParams::new(
            version,
            cfg.channels as usize,
            cfg.sample_rate,
            cfg.avg_bytes_per_sec.saturating_mul(8),
            cfg.block_align as usize,
            extra,
        ) {
            Ok(p) => p,
            Err(e) => {
                rc(&e);
                return std::ptr::null_mut();
            }
        };
        let skip_left = if cfg.drop_priming != 0 {
            params.frame_len * 2
        } else {
            0
        };
        let decoder = match v2v_wma::WmaDecoder::new(params) {
            Ok(d) => d,
            Err(e) => {
                rc(&e);
                return std::ptr::null_mut();
            }
        };
        Box::into_raw(Box::new(V2vAdec {
            decoder,
            pending: VecDeque::new(),
            cur: None,
            ptrs: [std::ptr::null(); MAX_CHANNELS],
            channels: cfg.channels as usize,
            sample_rate: cfg.sample_rate,
            emitted: 0,
            skip_left,
        }))
    })
}

/// 关掉并释放。传空指针是空操作。
///
/// # Safety
/// `h` 必须是 [`v2v_adec_open`] 的返回值，且只能关一次。
#[no_mangle]
pub unsafe extern "C" fn v2v_adec_close(h: *mut V2vAdec) {
    if h.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(h) });
}

/// 喂一个载荷（容器里的一个音频样本），解出来的帧进队列。
///
/// # Safety
/// `data` 必须指向 `len` 字节可读内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_adec_feed(h: *mut V2vAdec, data: *const u8, len: u32) -> c_int {
    guard(|| {
        if h.is_null() {
            return arg_err("句柄是空指针");
        }
        let bytes = match unsafe { slice(data, len) } {
            Ok(b) => b,
            Err(e) => return arg_err(&e),
        };
        let hp = unsafe { &mut *h };
        match hp.decoder.decode_superframe(bytes) {
            Ok(frames) => {
                for f in frames {
                    hp.pending.push_back(f);
                }
                OK
            }
            Err(e) => rc(&e),
        }
    })
}

/// 取一帧。**返回 [`EOS`] 只表示队列空了**（喂下一个载荷再来），不是流结束。
///
/// # Safety
/// `out` 必须指向一块首字段是 `size` 的可写内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_adec_next(h: *mut V2vAdec, out: *mut V2vAdecFrame) -> c_int {
    guard(|| {
        if h.is_null() || out.is_null() {
            return arg_err("句柄或出参是空指针");
        }
        let hp = unsafe { &mut *h };
        loop {
            let Some(mut planes) = hp.pending.pop_front() else {
                return EOS;
            };
            let n = planes.first().map_or(0, |p| p.len());
            // 编码器延迟：丢样本、**不**丢 PTS（时间轴才直）。
            if hp.skip_left > 0 && n > 0 {
                let drop = hp.skip_left.min(n);
                for p in &mut planes {
                    p.drain(..drop);
                }
                hp.skip_left -= drop;
            }
            let kept = planes.first().map_or(0, |p| p.len());
            if kept == 0 {
                continue;
            }
            let pts = (hp.emitted as f64 * 1000.0 / hp.sample_rate as f64).round() as i64;
            hp.emitted += kept as u64;
            if planes.len() > MAX_CHANNELS {
                set_err(KIND_UNSUPPORTED, "解出来的声道数超过 ABI 上限");
                return ERR_UNSUPPORTED;
            }
            let mut ptrs = [std::ptr::null::<f32>(); MAX_CHANNELS];
            for (i, p) in planes.iter().enumerate() {
                ptrs[i] = p.as_ptr();
            }
            hp.ptrs = ptrs;
            // 上一帧已经交出去了，先放掉再换成新的 —— 免得两帧同时驻留。
            hp.cur.take();
            hp.cur = Some(planes);
            let v = V2vAdecFrame {
                size: std::mem::size_of::<V2vAdecFrame>() as u32,
                planes: hp.ptrs.as_ptr(),
                channels: hp.channels as u32,
                samples: kept as u32,
                sample_rate: hp.sample_rate,
                pts_ms: pts,
            };
            // 在 `loop` 里，块的尾表达式不是函数返回值 —— 必须显式 `return`。
            return unsafe { write_sized(out, v) };
        }
    })
}
