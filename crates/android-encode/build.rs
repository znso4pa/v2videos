//! 构建脚本：目标平台是 **Android** 时编译 C shim 并链 `mediandk`。
//!
//! **最低 API = 26**（`AMediaCodec_setParameters` 的要求；Android 8.0，2017 年起
//! 的设备都覆盖）。`min sdk` 要在 App 侧同步设成 26。
//!
//! 交叉编译时 cc crate 要知道用哪个编译器 —— 本脚本按序自动探测 NDK：
//! `$ANDROID_NDK_HOME` → `~/Library/Android/sdk/ndk/<最新>`，
//! 取里面 API 26 的 clang。也可以手动指：
//! `CC_aarch64_linux_android=<ndk>/.../bin/aarch64-linux-android26-clang`。
//!
//! 其它平台（macOS / Linux / Windows）这个 crate 照常编译 —— 运行时 API 返回
//! `Unsupported`，workspace 在开发机上永远绿色；FFI 代码用 NDK 交叉编译或在
//! 设备上验（`tests/hw.rs`）。

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "android" {
        println!("cargo:rerun-if-changed=build.rs");
        return;
    }

    println!("cargo:rustc-link-lib=dylib=mediandk");

    let mut build = cc::Build::new();
    build
        .file("c/media_encoder.c")
        .flag_if_supported("-std=c11");

    // 交叉编译时找 NDK 的 clang（用户手动设了 CC_* 就不用管）。
    let target = std::env::var("TARGET").unwrap_or_default();
    let cc_env = format!("CC_{}", target.replace('-', "_"));
    if std::env::var(&cc_env).is_err() && std::env::var("CC").is_err() {
        if let Some(clang) = find_ndk_clang(&target) {
            // NDK r28 起没有独立的 `*-ar` 了 —— 统一的 llvm-ar 顶上（cc-rs 会找
            // `aarch64-linux-android-ar`，找不到直接失败）。
            let ar = clang
                .parent()
                .map(|p| p.join("llvm-ar"))
                .unwrap_or_default();
            if ar.exists() {
                std::env::set_var(format!("AR_{}", target.replace('-', "_")), &ar);
            }
            build.compiler(clang);
        }
    }

    build.compile("ame_shim");
    println!("cargo:rerun-if-changed=c/media_encoder.c");
    println!("cargo:rerun-if-changed=c/media_encoder.h");
}

/// 在常见位置找 NDK 里 API 26 的 clang。
fn find_ndk_clang(target: &str) -> Option<std::path::PathBuf> {
    let triple = target.split('-').next()?; // aarch64 / armv7a... 简化：直接试名字
    let mut homes = Vec::new();
    if let Ok(h) = std::env::var("ANDROID_NDK_HOME") {
        homes.push(std::path::PathBuf::from(h));
    }
    if let Ok(home) = std::env::var("HOME") {
        let sdk = std::path::Path::new(&home).join("Library/Android/sdk/ndk");
        if let Ok(entries) = std::fs::read_dir(&sdk) {
            let mut versions: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
            versions.sort(); // 版本号排序，取最大
            homes.extend(versions);
        }
    }
    // 编译器名字按 NDK 规则：aarch64-linux-android26-clang / armv7a-linux-androideabi26-clang…
    let prefix = if triple == "aarch64" {
        "aarch64-linux-android26-clang"
    } else if triple == "armv7" {
        "armv7a-linux-androideabi26-clang"
    } else if triple == "x86_64" {
        "x86_64-linux-android26-clang"
    } else if triple == "i686" {
        "i686-linux-android26-clang"
    } else {
        return None;
    };
    for home in homes {
        let mut prebuilt = home.join("toolchains/llvm/prebuilt");
        if let Ok(entries) = std::fs::read_dir(&prebuilt) {
            for host in entries.filter_map(|e| e.ok()).map(|e| e.path()) {
                prebuilt = host.clone();
                let clang = host.join("bin").join(prefix);
                if clang.exists() {
                    return Some(clang);
                }
            }
        }
        let _ = prebuilt;
    }
    None
}
