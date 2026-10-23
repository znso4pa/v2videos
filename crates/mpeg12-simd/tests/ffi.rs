//! FFI 表面能用在 C 侧：版本号、内核掩码、平均内核。
//!
//! 这条测试存在的理由：宿主（rfvp / 引擎）是**从 C 侧 dlopen 这个 .so** 的，
//! 所以这几个 `extern "C"` 入口坏了不会在 Rust 侧暴露 —— 只能在这里钉住。

use std::ffi::CStr;
use std::os::raw::c_char;

extern "C" {
    fn v2v_mpeg12_simd_version() -> *const c_char;
    fn v2v_mpeg12_simd_kernels() -> u32;
}

#[test]
fn ffi_surface_is_usable() {
    unsafe {
        let v = v2v_mpeg12_simd_version();
        assert!(!v.is_null(), "版本号不能是空指针");
        let s = CStr::from_ptr(v).to_str().expect("版本号是合法 UTF-8");
        assert!(!s.is_empty(), "版本号不能是空串");
        let mask = v2v_mpeg12_simd_kernels();
        assert_eq!(mask & v2v_mpeg12_simd::kernels::KERNEL_AVG, v2v_mpeg12_simd::kernels::KERNEL_AVG,
            "半像素平均内核必须可用（本 crate 的门面）");
    }
}
