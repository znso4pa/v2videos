//! 音频编码（`v2v_aenc_*`）：AAC-LC。
//!
//! ## 一帧就是 1024 个采样点
//!
//! AAC 的帧长是**语法规定的**（1024），不是配置项。所以 `encode` 要求
//! `samples == 1024`，别的值直接报参数错 —— 悄悄补零或截断会让音频长度对不上，
//! 那种错在播放器里表现为「音画慢慢偏」，很难查。
//!
//! ## `AudioSpecificConfig` 要单独取
//!
//! 封装器（MP4 的 `esds`）需要它，但它在**建编码器时**就定了、与帧无关。
//! 所以单独给一个 `v2v_aenc_asc()`，别塞进每帧的返回里。

use std::os::raw::c_int;

use crate::common::*;

/// AAC 一帧的采样点数（语法规定，不是配置）。
pub const AAC_FRAME_SAMPLES: u32 = 1024;

/// 建编码器要的参数。
#[repr(C)]
pub struct V2vAencCfg {
    pub size: u32,
    /// 1 或 2。
    pub channels: u32,
    pub sample_rate: u32,
    /// 目标平均码率（bit/s）。这个编码器**不做**码率控制，它是「目标」不是「保证」。
    pub bitrate: u32,
    /// 双声道时允许 M/S 立体声（默认建议 1）。
    pub ms_stereo: u32,
    /// 用 KBD 窗而不是正弦窗（默认 0）。
    pub kb_window: u32,
    /// 编一帧用几个线程。0 = 1。
    pub threads: u32,
}

/// 音频编码句柄（不透明）。
pub struct V2vAenc {
    enc: v2v_aac::Encoder,
    channels: usize,
    /// 最近一次输出的码流 —— 出参的指针指向这里。
    out: Vec<u8>,
    /// `AudioSpecificConfig` 的字节（建编码器时就定了）。
    asc: Vec<u8>,
}

/// 建编码器。失败返回空指针。
///
/// # Safety
/// `cfg` 必须指向一块首字段是 `size` 的可读内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_aenc_open(cfg: *const V2vAencCfg) -> *mut V2vAenc {
    guard_ptr(|| {
        if cfg.is_null() {
            set_err(KIND_BAD, "参数是空指针");
            return std::ptr::null_mut();
        }
        let cfg = unsafe { &*cfg };
        if (cfg.size as usize) < std::mem::size_of::<V2vAencCfg>() {
            set_err(KIND_BAD, "cfg->size 太小，可能不是本版本的调用方");
            return std::ptr::null_mut();
        }
        let ec = v2v_aac::EncoderConfig {
            channels: cfg.channels as u8,
            sample_rate: cfg.sample_rate,
            bitrate: cfg.bitrate,
            kb_window: cfg.kb_window != 0,
            ms_stereo: cfg.ms_stereo != 0,
            threads: (cfg.threads as usize).max(1),
        };
        match v2v_aac::Encoder::new(ec) {
            Ok(enc) => {
                let asc = enc.audio_specific_config().to_bytes();
                Box::into_raw(Box::new(V2vAenc {
                    enc,
                    channels: cfg.channels as usize,
                    out: Vec::new(),
                    asc,
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
/// `h` 必须是 [`v2v_aenc_open`] 的返回值，且只能关一次。
#[no_mangle]
pub unsafe extern "C" fn v2v_aenc_close(h: *mut V2vAenc) {
    if h.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(h) });
}

/// 编一帧（必须正好 1024 个采样点）。
///
/// `planes` 指向 `channels` 个指针，每条 `samples` 个 float。
///
/// # Safety
/// `planes` 与它指向的每一条都必须可读；`out` 的首字段必须是 `size`。
#[no_mangle]
pub unsafe extern "C" fn v2v_aenc_encode(
    h: *mut V2vAenc,
    planes: *const *const f32,
    samples: u32,
    out: *mut V2vPacket,
) -> c_int {
    guard(|| {
        if h.is_null() || planes.is_null() || out.is_null() {
            return arg_err("句柄、输入平面或出参是空指针");
        }
        if samples != AAC_FRAME_SAMPLES {
            return arg_err(&format!(
                "AAC 一帧必须正好 {AAC_FRAME_SAMPLES} 个采样点，传进来的是 {samples}"
            ));
        }
        let hp = unsafe { &mut *h };
        let ptrs = unsafe { std::slice::from_raw_parts(planes, hp.channels) };
        let mut data = Vec::with_capacity(hp.channels);
        for (i, p) in ptrs.iter().enumerate() {
            if p.is_null() {
                return arg_err(&format!("第 {i} 条声道的平面是空指针"));
            }
            data.push(unsafe { std::slice::from_raw_parts(*p, samples as usize) }.to_vec());
        }
        let frame = v2v_aac::Frame { planes: data };
        match hp.enc.encode_frame(&frame) {
            Ok(bytes) => {
                hp.out = bytes;
                let v = V2vPacket {
                    size: std::mem::size_of::<V2vPacket>() as u32,
                    data: hp.out.as_ptr(),
                    len: hp.out.len() as u32,
                    // AAC 每帧都是独立的（没有帧间预测），所以每一帧都是关键帧。
                    key_frame: 1,
                };
                unsafe { write_sized(out, v) }
            }
            Err(e) => rc(&e),
        }
    })
}

/// 把 `AudioSpecificConfig` 拷进 `buf`，返回字节数。`cap` 不够返回 `V2V_ERR_ARG`。
///
/// # Safety
/// `buf` 要么为空（`cap` 必须为 0），要么指向 `cap` 字节可写内存。
#[no_mangle]
pub unsafe extern "C" fn v2v_aenc_asc(h: *const V2vAenc, buf: *mut u8, cap: u32) -> c_int {
    guard(|| {
        if h.is_null() {
            return arg_err("句柄是空指针");
        }
        let hp = unsafe { &*h };
        let n = hp.asc.len();
        if (cap as usize) < n {
            return arg_err(&format!("缓冲太小：要 {n} 字节，给了 {cap}"));
        }
        if n > 0 && buf.is_null() {
            return arg_err("缓冲是空指针");
        }
        if n > 0 {
            unsafe { std::ptr::copy_nonoverlapping(hp.asc.as_ptr(), buf, n) };
        }
        n as c_int
    })
}
