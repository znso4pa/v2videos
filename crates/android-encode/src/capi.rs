//! 给宿主 App 的 C ABI —— JNI 桥（或 ctypes / 任何 FFI）都走这一层。
//!
//! 只暴露三件事：**转码**、**取最后一次错误**、**问硬件后端在不在**。
//! 路径用 UTF-8 C 字符串；`v2v_transcode` 返回 0 表示成功。
//!
//! 为什么单独一层而不是直接给 JNI 写：JNI 那层要在 C 里拼 `jstring` → `char*`，
//! 而这里只要 `const char*`；宿主 App 的 JNI 桥（那种 `-nostdlib` 的小库）
//! 用 `dlopen` + `dlsym` 就能拿到，不引入 Rust 的 C++ 运行时。
//!
//! ## 错误怎么拿
//!
//! [`v2v_last_error`] 返回**本线程**最后一次失败的原因（UTF-8，NUL 结尾）。
//! 指针在下次调用前有效 —— 别跨线程用，也别存起来。

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::path::Path;

use crate::transcode::{hw_supported, wmv_to_mp4_hw};

thread_local! {
    static LAST_ERR: RefCell<CString> = RefCell::new(c_empty());
}

fn c_empty() -> CString {
    CString::new("").expect("空串不会有 NUL")
}

fn set_err(msg: &str) {
    // 错误信息里混进 NUL 只会让 CString::new 失败，换成空格更利于排查。
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_else(|_| c_empty());
    LAST_ERR.with(|e| *e.borrow_mut() = c);
}

/// 把 C 字符串读成 `&str`；空指针 / 非 UTF-8 都报错。
///
/// # Safety
/// 调用方保证 `p` 指向一个 NUL 结尾的可读字符串。
unsafe fn cstr<'a>(p: *const c_char, what: &str) -> Result<&'a str, String> {
    if p.is_null() {
        return Err(format!("{what} 是空指针"));
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| format!("{what} 不是合法的 UTF-8"))
}

/// 把一个 WMV/ASF 转成 MP4（H.264 + AAC）。
///
/// * `video_bitrate` / `audio_bitrate` 传 0 表示用默认值（4 Mbps / 128 kbps）。
///
/// 返回码：
/// * `0` 成功
/// * `1` 参数非法（空指针、非 UTF-8）
/// * `2` 这个平台没有硬件编码后端（或硬件不可用）
/// * `3` 转码失败（具体原因看 [`v2v_last_error`]）
///
/// # Safety
/// `input` / `output` 必须是 NUL 结尾的可读字符串；调用期间不能被释放。
#[no_mangle]
pub extern "C" fn v2v_transcode(
    input: *const c_char,
    output: *const c_char,
    video_bitrate: u32,
    audio_bitrate: u32,
) -> i32 {
    let (inp, out) = unsafe {
        match (cstr(input, "input"), cstr(output, "output")) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => {
                set_err(&e);
                return 1;
            }
        }
    };
    if !hw_supported() {
        set_err("这个平台没有可用的硬件编码后端");
        return 2;
    }
    let vbr = if video_bitrate == 0 { 4_000_000 } else { video_bitrate };
    let abr = if audio_bitrate == 0 { 128_000 } else { audio_bitrate };

    match wmv_to_mp4_hw(Path::new(inp), Path::new(out), vbr, abr) {
        Ok(_) => 0,
        Err(e) => {
            set_err(&e.to_string());
            3
        }
    }
}

/// 本线程最后一次失败的原因（UTF-8，NUL 结尾）。成功时是空串。
#[no_mangle]
pub extern "C" fn v2v_last_error() -> *const c_char {
    LAST_ERR.with(|e| e.borrow().as_ptr())
}

/// 当前平台有没有可用的硬件编码后端（1 有 / 0 没有）。
#[no_mangle]
pub extern "C" fn v2v_hw_supported() -> i32 {
    i32::from(hw_supported())
}
