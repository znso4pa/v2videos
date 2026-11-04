//! FFI 表面能用在 C 侧：版本号、内核掩码、平均内核。
//!
//! 这条测试存在的理由：宿主（引擎 / App）是**从 C 侧 dlopen 这个 .so** 的，
//! 所以这几个 `extern "C"` 入口坏了不会在 Rust 侧暴露 —— 只能在这里钉住。

use std::ffi::CStr;
use std::os::raw::c_char;

extern "C" {
    fn v2v_mpeg12_simd_version() -> *const c_char;
    fn v2v_mpeg12_simd_kernels() -> u32;
}

#[test]
fn ffi_surface_is_usable() {
    // ⚠️ 必须**无条件**引用一个 crate 里的 Rust 符号。
    // 集成测试如果只写 `extern "C"` 声明、不碰任何 Rust 层符号，rustc 不会
    // 把这个 crate 的 rlib 真正拉进链接，`no_mangle` 符号就解析不到
    // （实测：关掉 `unsafe-kernels` 时必现 "Undefined symbols"）。
    assert_eq!(v2v_mpeg12_simd::VERSION, env!("CARGO_PKG_VERSION"));
    unsafe {
        let v = v2v_mpeg12_simd_version();
        assert!(!v.is_null(), "版本号不能是空指针");
        let s = CStr::from_ptr(v).to_str().expect("版本号是合法 UTF-8");
        assert!(!s.is_empty(), "版本号不能是空串");
        let mask = v2v_mpeg12_simd_kernels();
        // 掩码报的是**实际会用到**的内核，所以它必须跟着 feature 走 ——
        // 关掉 `unsafe-kernels` 时 `avg_rows` 走标量，这里就该是 0。
        #[cfg(feature = "unsafe-kernels")]
        assert_eq!(
            mask & v2v_mpeg12_simd::kernels::KERNEL_AVG,
            v2v_mpeg12_simd::kernels::KERNEL_AVG,
            "打开 unsafe-kernels 时半像素平均内核必须可用（本 crate 的门面）"
        );
        #[cfg(not(feature = "unsafe-kernels"))]
        assert_eq!(mask, 0, "关掉 unsafe-kernels 时不应报告任何内核");
    }
}
