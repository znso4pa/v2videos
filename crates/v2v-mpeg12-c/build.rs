fn main() {
    println!("cargo:rustc-check-cfg=cfg(v2v_c_mc8)");
    println!("cargo:rerun-if-changed=src/mc8_neon.c");

    #[cfg(feature = "neon")]
    {
        let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if arch == "aarch64" && matches!(os.as_str(), "macos" | "ios" | "linux" | "android") {
            cc::Build::new()
                .file("src/mc8_neon.c")
                .flag_if_supported("-fvisibility=hidden")
                .compile("v2v_mpeg12_mc8");
            println!("cargo:rustc-cfg=v2v_c_mc8");
        }
    }
}
