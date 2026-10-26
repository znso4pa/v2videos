//! MPEG-1/2 解码内核：**允许 unsafe 与 FFI** 的加速变体。
//!
//! ## 为什么单独一个 crate
//!
//! `v2v-mpeg12` 是纯安全代码、零 unsafe，作为**参考实现（oracle）**留着 ——
//! 每个 unsafe 内核都要逐字节对它验。但真要抠最后那点性能，就绕不开 unsafe
//! （`std::arch` 内部函数）和 FFI（把内核交给 C / 汇编，或者让宿主从别的语言调）。
//! 把这些圈在一个 crate 里，好处是：
//!
//! 1. **审计范围明确** —— unsafe 只允许出现在这里，别处照旧零 unsafe；
//! 2. **能单独编 `.so` / `.a`** —— `crate-type` 里带了 `staticlib` / `cdylib`，
//!    宿主只链这一个 crate 就能拿到最小体积的内核；
//! 3. **有 oracle 兜底** —— 关掉 `unsafe-kernels` feature，行为与 `v2v-mpeg12`
//!    逐字节一致（有测试钉着）。
//!
//! ## unsafe 的规矩（本 crate 内）
//!
//! - 每个 unsafe 块上面必须写清**为什么安全**（不变量是什么）；
//! - 用 `is_x86_feature_detected!` / `is_aarch64_feature_detected!` 做**运行时**检测，
//!   没有编译期 `target_feature` 硬编码 —— 一份二进制要能跑在没 AVX 的机器上；
//! - 内核必须**逐字节或逐位等价**于安全路径（`_mm_avg_epu8` 这种正好是
//!   `(a+b+1)>>1` 的可以逐字节；IDCT 这类浮点/整数差异的只能卡容差，
//!   并且要把容差写进测试）。

#![allow(unsafe_code)]

use std::os::raw::c_char;

/// 版本号，给 FFI 用（宿主 dlopen 之后第一件事就是问它）。
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// FFI 表面：宿主（C / C++ / JNI 桥）直接用这些入口，不碰 Rust 类型。
// ---------------------------------------------------------------------------

/// 返回版本字符串（静态生命周期，调用方**不要 free**）。
#[no_mangle]
pub extern "C" fn v2v_mpeg12_simd_version() -> *const c_char {
    VERSION.as_ptr() as *const c_char
}

/// 本 crate 编进了哪些内核（bit 位掩码，给宿主查询用）。**不等于已装** ——
/// 见 [`kernels::available`] 与 [`kernels::installed`] 的区别。
///
/// - bit 0：半像素平均（`kernels::avg_rows`）—— 写了但**没装**（实测无收益）
/// - bit 1：IDCT 列通道（`kernels::idct_cols`）—— 已装（aarch64）
#[no_mangle]
pub extern "C" fn v2v_mpeg12_simd_kernels() -> u32 {
    kernels::available()
}

// ---------------------------------------------------------------------------
// 内核
// ---------------------------------------------------------------------------

pub mod kernels;

/// 把本 crate 的内核装进解码器（进程内调一次即可）。
pub use kernels::install;

/// 解码入口（转调 `v2v-mpeg12`，安全路径仍是 oracle）。
///
/// ⚠️ **别把它当成「加速版」**：这里只是转调，加速来自 [`install`] 装进去的内核
/// （目前只有 IDCT 列通道，实测收益在噪声内 —— 见 TODO 的 2026-10-27 一节）。
/// 要加新内核，得先在 `v2v-mpeg12` 里留一个挂载点（本 crate 依赖 mpeg12，
/// 反过来不行，所以挂载点必须在 mpeg12 侧）。
pub use v2v_mpeg12::decode::{decode_frame, MtDecoder};
