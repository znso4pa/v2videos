//! 硬件转码与 WMA→WAV 便捷入口（`hw` feature，默认关）。
//!
//! 这几个符号原本在硬件编码那个 crate 的 C ABI 里。收到这里是为了「全仓只有一个
//! 错误通道、一个版本号」—— 底下调的还是同一套函数，只有两处有意为之的差别：
//!
//! 1. 失败的种类现在区分得开（`V2V_ERR_UNSUPPORTED` / `V2V_ERR_FAILED`），
//!    以前一律是 `3`；
//! 2. [`v2v_has_wma_audio`] 出错时返回 `-1` 而不是 `1` —— 以前「有音轨」和
//!    「参数非法」都是 `1`，调用方分不出来。
//!
//! 为什么默认关：它带平台相关的 C shim（Android 的 MediaCodec / macOS 的
//! VideoToolbox），只想要解码的产物不该背上它，也不该因为它在这个平台上编不过
//! 而卡住整个 crate。

use std::os::raw::c_int;
use std::path::Path;

use crate::common::*;

/// 这个平台有没有可用的硬件编码后端。1 = 有，0 = 没有。
#[no_mangle]
pub extern "C" fn v2v_hw_supported() -> c_int {
    guard(|| c_int::from(v2v_android::transcode::hw_supported()))
}

/// 把一个文件转成 MP4（硬件 H.264 + AAC）。
///
/// `video_bitrate` / `audio_bitrate` 传 0 表示用默认值。
///
/// # Safety
/// `input` / `output` 必须是 NUL 结尾的 UTF-8 字符串。
#[no_mangle]
pub unsafe extern "C" fn v2v_transcode(
    input: *const std::ffi::c_char,
    output: *const std::ffi::c_char,
    video_bitrate: u32,
    audio_bitrate: u32,
) -> c_int {
    guard(|| {
        let (i, o) = match (unsafe { cstr(input, "input") }, unsafe { cstr(output, "output") }) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => return arg_err(&e),
        };
        if !v2v_android::transcode::hw_supported() {
            set_err(KIND_UNSUPPORTED, "这个平台没有可用的硬件编码后端");
            return ERR_UNSUPPORTED;
        }
        // 0 = 用默认值。这一步原来在旧的那层壳里做，收编时照搬 ——
        // 直接往下传 0 会让编码器拿到「码率 0」，那是另一个意思。
        let vbr = if video_bitrate == 0 { 4_000_000 } else { video_bitrate };
        let abr = if audio_bitrate == 0 { 128_000 } else { audio_bitrate };
        match v2v_android::transcode::wmv_to_mp4_hw(Path::new(i), Path::new(o), vbr, abr) {
            Ok(_) => OK,
            Err(e) => rc(&e),
        }
    })
}

/// 这个文件里有没有 WMA 音轨。1 = 有，0 = 没有，-1 = 出错（原因看错误通道）。
///
/// 出错**不**返回 1：旧的那层壳把「有音轨」和「参数非法」都返回 1，
/// 调用方分不出来，只能白跑一次转换。这里把「出错」挪到负数，
/// 0/1 的含义没变，检查 `== 1` 或 `> 0` 的代码都不用改。
///
/// # Safety
/// `input` 必须是 NUL 结尾的 UTF-8 字符串。
#[no_mangle]
pub unsafe extern "C" fn v2v_has_wma_audio(input: *const std::ffi::c_char) -> c_int {
    guard(|| {
        let i = match unsafe { cstr(input, "input") } {
            Ok(s) => s,
            Err(e) => {
                set_err(KIND_BAD, &e);
                return -1;
            }
        };
        match v2v_android::wma_wav::has_wma_audio(Path::new(i)) {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(e) => {
                rc(&e);
                -1
            }
        }
    })
}

/// 把音轨解成一段标准 WAV（RIFF/WAVE、PCM16LE、声道交织）。
///
/// **不解视频、不重编码** —— 适合「自带解码器太慢」的场合。
/// 返回的缓冲区所有权转移给调用方，用完必须 [`v2v_free`]。
///
/// # Safety
/// `input` 必须是 NUL 结尾的 UTF-8 字符串；`out_ptr` / `out_len` 必须可写。
#[no_mangle]
pub unsafe extern "C" fn v2v_wma_to_wav(
    input: *const std::ffi::c_char,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> c_int {
    guard(|| {
        if out_ptr.is_null() || out_len.is_null() {
            return arg_err("out_ptr / out_len 是空指针");
        }
        let i = match unsafe { cstr(input, "input") } {
            Ok(s) => s,
            Err(e) => return arg_err(&e),
        };
        match v2v_android::wma_wav::wma_to_wav_bytes(Path::new(i)) {
            Ok(vec) => {
                // 用 `into_boxed_slice` 而不是 `shrink_to_fit`：前者**保证**多余容量
                // 被搬进精确大小的分配，`v2v_free` 按 `len` 重建 `Box<[u8]>` 时
                // 释放布局才必然与分配一致。
                let boxed: Box<[u8]> = vec.into_boxed_slice();
                let len = boxed.len();
                let ptr = Box::into_raw(boxed) as *mut u8;
                unsafe {
                    *out_ptr = ptr;
                    *out_len = len;
                }
                OK
            }
            Err(e) => rc(&e),
        }
    })
}

/// 释放 [`v2v_wma_to_wav`] 返回的缓冲区（必须配对，且只释放一次）。
///
/// # Safety
/// `ptr` / `len` 必须来自一次成功的 [`v2v_wma_to_wav`] 调用。
#[no_mangle]
pub unsafe extern "C" fn v2v_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    unsafe {
        let slice = std::ptr::slice_from_raw_parts_mut(ptr, len);
        drop(Box::from_raw(slice));
    }
}
