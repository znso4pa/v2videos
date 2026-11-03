//! C ABI 的公共层：返回码、错误通道、panic 围栏、结构体版本化。
//!
//! ## 为什么错误要分「码」和「种类」两样
//!
//! 返回码给机器判分支（成功 / 参数错 / 不支持 / 失败 / 到头），
//! `v2v_last_error_kind()` 给机器判**为什么**失败（IO / 数据坏 / 不支持），
//! `v2v_last_error()` 给人看。三样缺一不可：
//!
//! - 只有字符串 → 调用方只能 `strstr`，等于没有分类；
//! - 只有码 → 「文件坏了」和「这台机器 IO 出错」分不开，重试策略写不出来。
//!
//! ## 错误通道是**每线程**的
//!
//! 用 `thread_local!` 而不是全局 `static`：两个线程各播各的文件时互不覆盖。
//! 代价是**跨线程读不到别的线程的错误** —— 这是有意的，一个线程的错误对另一个
//! 线程没有意义。指针活到**本线程的下一次 ABI 调用**，别存起来。

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CStr, CString};

/// 成功。
pub const OK: c_int = 0;
/// **调用方**参数不对：空指针、非 UTF-8、结构体 `size` 太小、句柄为空。
pub const ERR_ARG: c_int = 1;
/// 结构合法，但本实现（还）不支持。
pub const ERR_UNSUPPORTED: c_int = 2;
/// 运行期失败：IO、数据不合法、内部 panic。细分看 [`v2v_last_error_kind`]。
pub const ERR_FAILED: c_int = 3;
/// 已经到头。只用于「取下一个」这类调用，**不是**错误。
pub const EOS: c_int = 4;

/// 底层 IO 失败。
pub const KIND_IO: c_int = 0;
/// 数据不合法（结构对不上、越界、字段自相矛盾）。
pub const KIND_BAD: c_int = 1;
/// 结构合法但不支持。
pub const KIND_UNSUPPORTED: c_int = 2;

thread_local! {
    static LAST_ERR: RefCell<(CString, c_int)> = RefCell::new((c_empty(), KIND_BAD));
}

fn c_empty() -> CString {
    CString::new("").expect("空串不会有 NUL")
}

/// 记一条错误（种类 + 人看的消息）。
pub fn set_err(kind: c_int, msg: &str) {
    // 消息里混进 NUL 只会让 `CString::new` 失败，换成空格更利于排查。
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_else(|_| c_empty());
    LAST_ERR.with(|e| *e.borrow_mut() = (c, kind));
}

/// 把某个 crate 的错误类型映射成 ABI 的三分类。
///
/// 每个错误类型各实现一次 —— 它们长得像但**不通用**（有的带 `Io`，有的没有）。
/// 集中在这里而不是散到各模块，是为了「一张派发表」那条规矩：
/// 想知道 `Bad` 到底映射成哪个码，只看这一个文件。
pub trait IntoAbi {
    /// [`KIND_IO`] / [`KIND_BAD`] / [`KIND_UNSUPPORTED`] 之一。
    fn kind(&self) -> c_int;
    /// 给人看的消息。
    fn message(&self) -> String;
}

/// 记下错误并返回对应的返回码。
pub fn rc<E: IntoAbi + ?Sized>(e: &E) -> c_int {
    rc_kind(e.kind(), &e.message())
}

/// 同 [`rc`]，但种类与消息已经拆开了（两个 crate 的错误类型不同时用它收口）。
pub fn rc_kind(kind: c_int, msg: &str) -> c_int {
    set_err(kind, msg);
    match kind {
        KIND_UNSUPPORTED => ERR_UNSUPPORTED,
        _ => ERR_FAILED,
    }
}

/// 参数错：记一条 `KIND_BAD` 并返回 [`ERR_ARG`]。
pub fn arg_err(msg: &str) -> c_int {
    set_err(KIND_BAD, msg);
    ERR_ARG
}

/// 把任何会 panic 的闭包关在 ABI 边界里。
///
/// 解码器吃的是别人的字节，一条 panic 穿过 FFI 就是未定义行为 ——
/// 所以**每个** `extern "C"` 入口都要经过这里，没有例外。
pub fn guard(f: impl FnOnce() -> c_int) -> c_int {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(rc) => rc,
        Err(_) => {
            set_err(KIND_BAD, "内部 panic 已在 ABI 边界拦下");
            ERR_FAILED
        }
    }
}

/// 同 [`guard`]，但用于返回句柄的入口（panic 时返回空指针）。
pub fn guard_ptr<T>(f: impl FnOnce() -> *mut T) -> *mut T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(p) => p,
        Err(_) => {
            set_err(KIND_BAD, "内部 panic 已在 ABI 边界拦下");
            std::ptr::null_mut()
        }
    }
}

/// 读一个 NUL 结尾的 UTF-8 路径。
///
/// # Safety
/// `p` 要么为空，要么指向一块 NUL 结尾的可读内存。
pub unsafe fn cstr<'a>(p: *const c_char, what: &str) -> Result<&'a str, String> {
    if p.is_null() {
        return Err(format!("{what} 是空指针"));
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| format!("{what} 不是合法的 UTF-8"))
}

/// 读一段「指针 + 长度」的字节；`len == 0` 时允许空指针。
///
/// # Safety
/// `ptr` 要么为空，要么指向至少 `len` 字节的可读内存。
pub unsafe fn slice<'a>(ptr: *const u8, len: u32) -> Result<&'a [u8], String> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err("字节指针是空指针但长度不为 0".into());
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len as usize) })
}

/// 按调用方填的 `size` 截断写入一个出参结构体。
///
/// 所有出参结构体的**首字段都是 `uint32_t size`**。调用方填自己的
/// `sizeof`，库写 `min(调用方的 size, 库自己的 size)` 字节 ——
/// 于是「库加了字段」对老调用方是安全的（多出来的字节不写），
/// 「调用方比库新」也是安全的（库只写自己知道的那部分）。
///
/// # Safety
/// `out` 要么为空，要么指向一块可写、且首 4 字节是一个 `u32 size` 的内存，
/// 大小至少 `size` 字节。
pub unsafe fn write_sized<T>(out: *mut T, val: T) {
    if out.is_null() {
        return;
    }
    let cap = unsafe { *(out as *const u32) } as usize;
    let n = cap.min(std::mem::size_of::<T>());
    if n < std::mem::size_of::<u32>() {
        // 连 `size` 都放不下 —— 调用方给的缓冲不合法，别写。
        set_err(KIND_BAD, "出参结构体的 size 字段不合法");
        return;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(&val as *const T as *const u8, out as *mut u8, n);
    }
}

// ---------------------------------------------------------------------------
// 跨层共用的出参结构
// ---------------------------------------------------------------------------

/// 一条音频最多几条声道。超过的流明确报不支持，**不静默截断**。
pub const MAX_CHANNELS: usize = 8;

/// 编解码器编号 —— **解码与编码共用这一张表**（只此一份，别处 re-export）。
pub const CODEC_WMV3: u32 = 0;
pub const CODEC_MP43: u32 = 1;
pub const CODEC_MP42: u32 = 2;
pub const CODEC_WMV1: u32 = 3;
pub const CODEC_WMV2: u32 = 4;
pub const CODEC_MP4V: u32 = 5;

/// YUV420P 三平面。**stride 恒等于 width**（色度是 `ceil(width/2)`）——
/// 本仓不产 padding，每行紧排。调用方可以据此逐行 `memcpy`。
///
/// 三个指针都是**借用**的：活到句柄的下一次同类调用。
#[repr(C)]
pub struct V2vPlanes {
    pub y: *const u8,
    pub u: *const u8,
    pub v: *const u8,
    pub width: u32,
    pub height: u32,
    pub stride_y: u32,
    pub stride_u: u32,
    pub stride_v: u32,
}

/// 把一帧的三块平面填成 [`V2vPlanes`]（stride 按 YUV420P 的规则算）。
pub fn planes_of(y: &[u8], u: &[u8], v: &[u8], width: u32, height: u32) -> V2vPlanes {
    V2vPlanes {
        y: y.as_ptr(),
        u: u.as_ptr(),
        v: v.as_ptr(),
        width,
        height,
        stride_y: width,
        stride_u: width.div_ceil(2),
        stride_v: width.div_ceil(2),
    }
}

/// 编出来的一包码流（视频一帧 / 音频一帧）。
///
/// `data` 是**借用**的：活到句柄的下一次 `encode` / `close`。
#[repr(C)]
pub struct V2vPacket {
    pub size: u32,
    pub data: *const u8,
    pub len: u32,
    /// 0 / 1。封装器写载荷的关键帧位要用它。
    pub key_frame: i32,
}

// ---------------------------------------------------------------------------
// 各 crate 的错误 → 三分类
// ---------------------------------------------------------------------------

impl IntoAbi for v2v_player::Error {
    fn kind(&self) -> c_int {
        match self {
            v2v_player::Error::Io(_) => KIND_IO,
            v2v_player::Error::Bad(_) => KIND_BAD,
            v2v_player::Error::Unsupported(_) => KIND_UNSUPPORTED,
        }
    }
    fn message(&self) -> String {
        self.to_string()
    }
}

impl IntoAbi for v2v_asf::Error {
    fn kind(&self) -> c_int {
        match self {
            v2v_asf::Error::Io(_) => KIND_IO,
            v2v_asf::Error::Bad(_) | v2v_asf::Error::Eof => KIND_BAD,
            v2v_asf::Error::Unsupported(_) => KIND_UNSUPPORTED,
        }
    }
    fn message(&self) -> String {
        self.to_string()
    }
}

impl IntoAbi for v2v_avi::Error {
    fn kind(&self) -> c_int {
        match self {
            v2v_avi::Error::Io(_) => KIND_IO,
            v2v_avi::Error::Bad(_) => KIND_BAD,
            v2v_avi::Error::Unsupported(_) => KIND_UNSUPPORTED,
        }
    }
    fn message(&self) -> String {
        self.to_string()
    }
}

impl IntoAbi for v2v_msmpeg4::Error {
    fn kind(&self) -> c_int {
        match self {
            v2v_msmpeg4::Error::Eof | v2v_msmpeg4::Error::Bad(_) => KIND_BAD,
            v2v_msmpeg4::Error::Unsupported(_) => KIND_UNSUPPORTED,
        }
    }
    fn message(&self) -> String {
        self.to_string()
    }
}

impl IntoAbi for v2v_wma::Error {
    fn kind(&self) -> c_int {
        match self {
            v2v_wma::Error::Io(_) => KIND_IO,
            v2v_wma::Error::Bad(_) => KIND_BAD,
            v2v_wma::Error::Unsupported(_) => KIND_UNSUPPORTED,
        }
    }
    fn message(&self) -> String {
        self.to_string()
    }
}

impl IntoAbi for v2v_mp4::Error {
    fn kind(&self) -> c_int {
        KIND_BAD
    }
    fn message(&self) -> String {
        self.to_string()
    }
}

// ---------------------------------------------------------------------------
// 版本与错误通道
// ---------------------------------------------------------------------------

/// ABI 的主版本号。**结构体布局或符号语义不兼容地变了**才 +1。
pub const ABI_MAJOR: u32 = 1;
/// ABI 的次版本号。**只加符号 / 只加结构体尾部字段**时 +1。
pub const ABI_MINOR: u32 = 0;

/// ABI 版本，`(major << 16) | minor`。
///
/// 调用方在 `open` 之前比一次：主版本不同就别继续（布局可能不一样），
/// 次版本比自己的低说明库比头文件新（多的符号用不上，但老符号照常可用）。
#[no_mangle]
pub extern "C" fn v2v_abi_version() -> u32 {
    (ABI_MAJOR << 16) | ABI_MINOR
}

/// 本线程最后一次失败的原因（UTF-8，NUL 结尾）。
///
/// 指针在**本线程的下一次 ABI 调用前**有效 —— 别跨线程用，也别存起来。
/// 成功时返回空串（不是空指针）。
#[no_mangle]
pub extern "C" fn v2v_last_error() -> *const c_char {
    LAST_ERR.with(|e| e.borrow().0.as_ptr())
}

/// 本线程最后一次失败的种类：[`KIND_IO`] / [`KIND_BAD`] / [`KIND_UNSUPPORTED`]。
#[no_mangle]
pub extern "C" fn v2v_last_error_kind() -> c_int {
    LAST_ERR.with(|e| e.borrow().1)
}
