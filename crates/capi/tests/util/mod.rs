//! 几条测试共用的辅助函数。
//!
//! 放 `tests/util/mod.rs` 而不是 `tests/util.rs`：cargo 只把 `tests/*.rs` 和
//! `tests/<dir>/main.rs` 当成测试目标，所以这个目录不会被单独编成一条测试。
//!
//! 这里的每一件都在回答同一个问题：**`cargo test` 不产出 `cdylib`**，所以凡是要
//! 碰真 `.so` 的测试都得自己嵌套跑一次 `cargo build`，再从 `<target>/debug` 里找
//! 产物。把这段抄两遍，两份迟早会漂（一份带 feature、一份不带）。

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

pub fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `.so` / `.dylib` / `.dll` 都可能 —— 测试不该假设平台。
pub fn dylib_in(dir: &Path) -> Option<PathBuf> {
    ["libv2v_capi.dylib", "libv2v_capi.so", "v2v_capi.dll"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.is_file())
}

/// 静态库。`staticlib` 是 `Cargo.toml` 里声明过的第三种产物。
pub fn staticlib_in(dir: &Path) -> Option<PathBuf> {
    ["libv2v_capi.a", "v2v_capi.lib"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.is_file())
}

/// 测试可执行文件在 `<target>/[<triple>/]debug/deps/` 下，往上两级就是产物目录。
pub fn debug_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.parent()?.to_path_buf())
}

/// 按给定 feature 编一次 `.so`，返回产物路径。
///
/// ⚠️ cargo 一个包只能出一个 `cdylib`：连着编两次，第二次会**覆盖**第一次的
/// `libv2v_capi.*`。所以调用方每次拿到路径就得立刻用完，不能先编两个再一起用。
pub fn build_product(feats: &[&str]) -> Option<PathBuf> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut b = Command::new(&cargo);
    b.args(["build", "-p", "v2v-capi"]);
    if !feats.is_empty() {
        b.arg("--features").arg(feats.join(","));
    }
    let out = b.output().expect("调不动 cargo");
    assert!(
        out.status.success(),
        "嵌套 cargo build（{feats:?}）失败：\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let dir = debug_dir()?;
    let p = dylib_in(&dir);
    if p.is_none() {
        eprintln!("SKIP：{} 下没找到 libv2v_capi", dir.display());
    }
    p
}
