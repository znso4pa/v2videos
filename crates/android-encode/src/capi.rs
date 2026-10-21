//! 给宿主 App 的 C ABI —— JNI 桥（或 ctypes / 任何 FFI）都走这一层。
//!
//! 只暴露四件事：**转码**、**取最后一次错误**、**问硬件后端在不在**，
//! 以及 **把 WMA 音轨抽成 WAV 字节**（给解码器太慢的引擎替换音频路径用）。
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

    match guard("转码", || {
        wmv_to_mp4_hw(Path::new(inp), Path::new(out), vbr, abr)
    }) {
        Ok(_) => 0,
        Err(e) => {
            set_err(&e);
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

/// 这个文件里有没有我们能解的 WMA 音轨（1 有 / 0 没有）。
///
/// 只读 ASF 头，不解码——给调用方做「值不值得走 WMA→WAV 这条路」的快速判断。
/// 参数非法时返回 `1` 并把原因写进 [`v2v_last_error`]；其它情况只回 0/1。
///
/// # Safety
/// `input` 必须是 NUL 结尾的可读字符串；调用期间不能被释放。
#[no_mangle]
pub extern "C" fn v2v_has_wma_audio(input: *const c_char) -> i32 {
    let inp = unsafe {
        match cstr(input, "input") {
            Ok(a) => a,
            Err(e) => {
                set_err(&e);
                return 1;
            }
        }
    };
    match guard("读取 ASF 头", || crate::wma_wav::has_wma_audio(Path::new(inp))) {
        Ok(true) => 1,
        Ok(false) => 0,
        Err(e) => {
            set_err(&e);
            3
        }
    }
}

/// 把可能 panic 的解码/转码包起来 —— `extern "C"` 里 unwind 出去会直接 abort 进程，
/// 一个损坏的 WMV 不该让整个 App 挂掉。
fn guard<T, E: std::fmt::Display>(
    what: &str,
    f: impl FnOnce() -> std::result::Result<T, E>,
) -> std::result::Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err(format!("{what} 时发生 panic（文件可能损坏）")),
    }
}

/// 把整个 WMV/ASF 的 WMA 音轨解成 PCM16LE 的 WAV 字节，写到 `*out_ptr`/`*out_len`。
///
/// 这条比 [`v2v_transcode`] 轻得多：**不解视频、不重编码**，只把音轨解出来——
/// 引擎拿到字节直接当「解好的 WAV」用（标准 RIFF/WAVE、PCM16LE、声道交织，
/// 与常见引擎自己的实现逐字段一致）。适合「自带 WMA 解码器慢到没法用、只能靠预转 MP4 绕开」的场景。
///
/// 返回码：
/// * `0` 成功（`*out_ptr`/`*out_len` 已填，必须用 [`v2v_free`] 释放）
/// * `1` 参数非法（空指针、非 UTF-8、`out_ptr`/`out_len` 为 NULL）
/// * `3` 解码失败（具体原因看 [`v2v_last_error`]；没有 WMA 音轨也走这里）
///
/// # Safety
/// `input` 必须是 NUL 结尾的可读字符串；`out_ptr`/`out_len` 必须指向可写地址。
/// 返回的缓冲区所有权转移给调用方，用完后调 [`v2v_free`] 释放，否则泄漏。
#[no_mangle]
pub extern "C" fn v2v_wma_to_wav(
    input: *const c_char,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if out_ptr.is_null() || out_len.is_null() {
        set_err("out_ptr / out_len 是空指针");
        return 1;
    }
    let inp = unsafe {
        match cstr(input, "input") {
            Ok(a) => a,
            Err(e) => {
                set_err(&e);
                return 1;
            }
        }
    };
    match guard("解码 WMA 音轨", || {
        crate::wma_wav::wma_to_wav_bytes(Path::new(inp))
    }) {
        Ok(vec) => {
            // 用 `into_boxed_slice` 而不是 `shrink_to_fit`：前者**保证**多余容量被搬进
            // 精确大小的分配（文档明说），这样 `v2v_free` 按 `len` 重建 `Box<[u8]>`
            // 时，释放布局必然与分配一致。
            // 反例：`shrink_to_fit` 不保证容量 == len（分配器可拒绝收缩），
            // 那样 `Vec::from_raw_parts(ptr, len, len)` 就会按更小的布局释放更大的块 —— UB。
            let boxed: Box<[u8]> = vec.into_boxed_slice();
            let len = boxed.len();
            let ptr = Box::into_raw(boxed) as *mut u8; // 瘦指针，长度另带
            unsafe {
                *out_ptr = ptr;
                *out_len = len;
            }
            0
        }
        Err(e) => {
            set_err(&e);
            3
        }
    }
}

/// 释放 [`v2v_wma_to_wav`] 返回的缓冲区。
///
/// 必须配对调用：`ptr` / `len` 就是 `v2v_wma_to_wav` 写入的值。空指针或长度 0 直接忽略。
///
/// # Safety
/// `ptr` / `len` 必须来自一次成功的 [`v2v_wma_to_wav`] 调用，且只释放一次。
#[no_mangle]
pub extern "C" fn v2v_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    unsafe {
        // 与 `v2v_wma_to_wav` 里的 `Box::into_raw` 配对：`into_boxed_slice` 保证了
        // 分配大小恰好是 `len`，所以这里用 `len` 重建胖指针释放是合法的。
        let slice = std::ptr::slice_from_raw_parts_mut(ptr, len);
        drop(Box::from_raw(slice));
    }
}
