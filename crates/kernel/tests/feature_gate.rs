//! 钉住「安全路径的依赖闭包里没有 unsafe 内核」这条**承诺**。
//!
//! ## 为什么要有这个测试
//!
//! `v2v-mpeg12` 的卖点之一是「参考实现、零 unsafe」。这句话在
//! `crates/kernel/src/simd.rs` 里只是一句**注释** —— 而注释不会失败。
//! 真正让它成立的是 `v2v-kernel` 的 `unsafe-kernels` feature：不开就是纯标量，
//! 一条 `unsafe` 都不编译。
//!
//! 所以这里用 `cargo tree -e features` 当判据，正反各钉一条：
//!
//! 1. `v2v-mpeg12` 的闭包里**不许**出现 `unsafe-kernels`；
//! 2. `v2v-msmpeg4` 的闭包里**必须**出现 —— 否则 NEON 会悄悄变成没人开，
//!    症状是「编码器莫名慢了」而没有任何报错。
//!
//! ⚠️ 第 2 条不是凑数：feature 忘了打开时，代码照样编译、测试照样全绿
//! （标量与 NEON 逐位等价），**只有性能掉下去**。那是最难发现的一类退化。
//!
//! ⚠️ 这里跑的是**子进程 cargo**（`cargo test` 本身不持有构建锁），和
//! `crates/capi/tests/` 里那几个嵌套 cargo 的测试是同一套路子。

use std::path::PathBuf;
use std::process::Command;

/// 工作区根（本文件在 `crates/kernel/tests/`）。
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("kernel 应该在工作区的 crates/ 下")
        .to_path_buf()
}

/// 某个包**自己的**依赖闭包里，`v2v-kernel` 打开了哪些 feature。
///
/// 用 `-p <包>` 单独解析：外层的 `--all-features` / `--features` 不会传染给子进程，
/// 所以拿到的就是「只依赖这个包时」的真实 feature 集合。
fn kernel_features_of(pkg: &str) -> String {
    let out = Command::new(env!("CARGO"))
        .current_dir(workspace_root())
        .args(["tree", "-p", pkg, "-e", "features", "--prefix", "none"])
        .output()
        .unwrap_or_else(|e| panic!("跑不动 cargo tree -p {pkg}：{e}"));
    assert!(
        out.status.success(),
        "cargo tree -p {pkg} 失败：\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn mpeg12_closure_has_no_unsafe_kernels() {
    let tree = kernel_features_of("v2v-mpeg12");
    assert!(
        !tree.contains(r#"v2v-kernel feature "unsafe-kernels""#),
        "v2v-mpeg12 的依赖闭包里出现了 unsafe-kernels —— \
         「参考实现零 unsafe」这句话就不再成立了。\n\
         要么是有人给 mpeg12 加了依赖（或给某个共享依赖加了 feature），\
         要么是 v2v-kernel 把 unsafe-kernels 设成了默认。\n\n{tree}"
    );
    // 反向确认这条断言不是「什么也没匹配到」的假阳性：
    // mpeg12 确实依赖 kernel，否则上面那句恒真。
    assert!(
        tree.contains("v2v-kernel"),
        "v2v-mpeg12 居然不依赖 v2v-kernel？这条测试的前提没了。\n\n{tree}"
    );
}

#[test]
fn msmpeg4_closure_does_enable_unsafe_kernels() {
    let tree = kernel_features_of("v2v-msmpeg4");
    assert!(
        tree.contains(r#"v2v-kernel feature "unsafe-kernels""#),
        "v2v-msmpeg4 没有打开 unsafe-kernels —— 它的运动估计会掉到标量 SAD 上，\
         编译和测试都不会报错，只是变慢。\n\
         检查 crates/msmpeg4/Cargo.toml 的 v2v-kernel 依赖行。\n\n{tree}"
    );
}
