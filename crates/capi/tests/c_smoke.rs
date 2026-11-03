//! 纯 C 的 ABI 冒烟：`dlopen` 打开**我们自己产的** `.so`，按 `v2v.h` 走一遍全链。
//!
//! 为什么值得单独写：`tests/abi.rs` 调的是 `v2v_capi::` 里的函数，**绕过了头文件** ——
//! 头文件把参数写反了、`size` 字段忘了、`stride` 语义写错了，它照样全绿。
//! 只有 C 编译器能回答「头文件自洽吗」，只有 `dlsym` 能回答「符号真的在 `.so` 里吗」。
//!
//! 三件东西缺一件就 SKIP（沿用仓里「缺东西打 SKIP」的约定，别让新机器变红）：
//! `cargo`、`cc`、`.so` 本身。
//!
//! ⚠️ `cargo test` **不会**产出 `cdylib`（只编 rlib 与测试夹具），所以这里自己
//! 嵌套跑一次 `cargo build`。feature 从 `CARGO_FEATURE_*` 原样透传 —— 不然
//! 外层带 `mpeg12`、嵌套按默认 feature 编，两边符号对不上。

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// 外层这次带了哪些 feature，原样传给嵌套的 `cargo build`。
fn features() -> Vec<String> {
    let mut v: Vec<String> = std::env::vars()
        .filter_map(|(k, _)| {
            k.strip_prefix("CARGO_FEATURE_")
                .map(|s| s.to_ascii_lowercase().replace('_', "-"))
        })
        .collect();
    v.sort();
    v
}

/// `.so` / `.dylib` / `.dll` 都可能 —— 测试不该假设平台。
fn dylib_in(dir: &Path) -> Option<PathBuf> {
    ["libv2v_capi.dylib", "libv2v_capi.so", "v2v_capi.dll"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.is_file())
}

/// 测试可执行文件在 `<target>/[<triple>/]debug/deps/` 下，往上两级就是产物目录。
fn debug_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.parent()?.to_path_buf())
}

/// 样本路径：`V2V_ASF_SAMPLE` 可以是 `:` 分隔的多个，取第一个存在的。
fn sample() -> Option<String> {
    std::env::var("V2V_ASF_SAMPLE")
        .ok()
        .and_then(|s| s.split(':').find(|p| Path::new(p).is_file()).map(String::from))
}

#[test]
fn c_smoke_through_dlopen() {
    // 1) 先保证 .so 在（`cargo test` 不产 cdylib）。
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut build = Command::new(&cargo);
    build.args(["build", "-p", "v2v-capi"]);
    let feats = features();
    if !feats.is_empty() {
        build.arg("--features").arg(feats.join(","));
    }
    let out = build.output().expect("调不动 cargo");
    assert!(
        out.status.success(),
        "嵌套 cargo build 失败：\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let Some(dir) = debug_dir() else { return };
    let Some(dylib) = dylib_in(&dir) else {
        eprintln!("SKIP：{} 下没找到 libv2v_capi", dir.display());
        return;
    };

    // 2) 编 C。这一步就是「头文件自洽」的判据：`-Werror` 下编不过就算失败。
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let bin = std::env::temp_dir().join(format!("v2v_abi_smoke_{}", std::process::id()));
    let mut c = Command::new(&cc);
    c.arg("-std=c11")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg("-I")
        .arg(manifest().join("include"))
        .arg(manifest().join("tests/smoke.c"))
        .arg("-o")
        .arg(&bin);
    if cfg!(target_os = "linux") {
        c.arg("-ldl");
    }
    match c.status() {
        Ok(s) => assert!(s.success(), "编 tests/smoke.c 失败 —— 头文件或 smoke 本身有问题"),
        Err(e) => {
            eprintln!("SKIP：没有可用的 C 编译器（{cc}）：{e}");
            return;
        }
    }

    // 3) 跑。给了样本就多走 open / 读帧 / seek / demux 那几段。
    let mut run = Command::new(&bin);
    run.arg(&dylib);
    if let Some(s) = sample() {
        run.arg(s);
    }
    let o = run.output().expect("C 程序跑不起来");
    let _ = std::fs::remove_file(&bin);
    let stdout = String::from_utf8_lossy(&o.stdout);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(
        o.status.success(),
        "C smoke 失败（退出码 {:?}）：\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        o.status.code()
    );
    // 没样本时只验了三条，别让「全绿」看起来像验了全链。
    if sample().is_none() {
        eprintln!("注意：没设 V2V_ASF_SAMPLE，只验了版本 / probe / 错误通道 / size 契约。");
    }
}
