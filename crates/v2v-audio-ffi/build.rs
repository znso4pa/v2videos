use std::{env, process::Command};

fn main() {
    if env::var_os("CARGO_FEATURE_MPG123").is_none() {
        return;
    }
    println!("cargo:rerun-if-env-changed=V2V_MPG123_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    if let Some(dir) = env::var_os("V2V_MPG123_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir.to_string_lossy());
    } else {
        assert_eq!(
            env::var_os("HOST"),
            env::var_os("TARGET"),
            "交叉编译 mpg123 时请设置 V2V_MPG123_LIB_DIR 指向目标平台的动态库",
        );
        let pkg_config = env::var_os("PKG_CONFIG").unwrap_or_else(|| "pkg-config".into());
        let output = Command::new(pkg_config)
            .args(["--variable=libdir", "libmpg123"])
            .output()
            .expect("安装 pkg-config 与 libmpg123，或设置 V2V_MPG123_LIB_DIR");
        let dir = std::str::from_utf8(&output.stdout)
            .expect("libmpg123 的库路径必须是 UTF-8")
            .trim();
        assert!(
            output.status.success() && !dir.is_empty(),
            "找不到 libmpg123，请安装其开发包或设置 V2V_MPG123_LIB_DIR"
        );
        println!("cargo:rustc-link-search=native={dir}");
    }
    println!("cargo:rustc-link-lib=dylib=mpg123");
}
