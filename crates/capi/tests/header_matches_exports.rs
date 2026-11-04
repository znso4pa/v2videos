//! 头文件里的声明 ↔ `.so` 里的导出，必须一一对应。
//!
//! 头文件是手写的，`#[no_mangle]` 也是手写的，**中间没有任何机制保证两者一致**：
//!
//! - 头文件声明了、代码里忘了 `#[no_mangle]` → 调用方**链接时**才发现（静态链接
//!   报 undefined symbol，dlopen 的要到运行时 `dlsym` 返回 NULL 才发现）；
//! - 代码导出了、头文件没声明 → 隐形 API：没人知道它存在，也没人敢删。
//!
//! `tests/smoke.c` 只 `dlsym` 它用到的那几十个，**新加的函数它一个都不认识** ——
//! 所以它证明不了这件事。这里换个做法：把**头文件里的全部声明**抽出来，跟 `nm`
//! 看到的**全部导出**对一遍。
//!
//! 判据只对 `full`（`hw,mpeg12`）：那是「全部符号」的定义，`min` 是它的子集。
//! `min` 到底少了哪几族，`tests/smoke.c` 跑一遍就能看见，不在这里重复一份名单
//! —— 名单会漂，而漂了的名单比没有名单更坏。

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

mod util;
use util::{build_product, manifest};

/// 去掉 `/* ... */` 注释。头文件里大量符号名出现在注释的示例里，不去掉会误判。
fn strip_block_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
            // 注释换成空白：行号与「一行一个声明」的形状都不变。
            out.push(b'\n');
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).expect("头文件不是 UTF-8")
}

/// 从头文件抽出全部 `v2v_*` 函数声明。
///
/// 做法：每行找第一个 `(`，取它前面那串标识符。多行声明（参数换行）也能中，
/// 因为名字总在第一行、`(` 总在名字后面。宏 `V2V_INIT(s)` 的名字是大写
/// `V2V_`，不匹配前缀，自然被排除。
fn declared(header: &Path) -> BTreeSet<String> {
    let src = std::fs::read_to_string(header).expect("读不了 v2v.h");
    let src = strip_block_comments(&src);
    let mut out = BTreeSet::new();
    for line in src.lines() {
        let Some(paren) = line.find('(') else { continue };
        let head = &line[..paren];
        let name: String = head
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if name.starts_with("v2v_") {
            out.insert(name);
        }
    }
    out
}

/// 从 `.so` 里读导出的 `v2v_*` 符号（只取已定义、类型 `T` 的）。
fn exported(dylib: &Path) -> Option<BTreeSet<String>> {
    let mut cmd = Command::new("nm");
    // macOS 的 `-U` 是「不显示未定义」，Linux 用 `--defined-only`。
    if cfg!(target_os = "macos") {
        cmd.arg("-gU");
    } else {
        cmd.args(["-D", "--defined-only"]);
    }
    let out = cmd.arg(dylib).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut set = BTreeSet::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 3 || f[f.len() - 2] != "T" {
            continue;
        }
        // macOS 的符号带一个下划线前缀。
        let name = f[f.len() - 1].trim_start_matches('_');
        if name.starts_with("v2v_") {
            set.insert(name.to_string());
        }
    }
    Some(set)
}

fn report(kind: &str, set: &BTreeSet<String>) -> String {
    let mut s = format!("{kind}（{} 个）：", set.len());
    for n in set {
        s.push_str("\n    ");
        s.push_str(n);
    }
    s
}

#[test]
fn header_declarations_match_exports() {
    let header = manifest().join("include/v2v.h");
    let decl = declared(&header);
    assert!(
        decl.len() > 30,
        "只从头文件里抽出 {} 个声明 —— 解析规则大概没匹配上，先修测试而不是怀疑头文件",
        decl.len()
    );

    let Some(full) = build_product(&["hw", "mpeg12"]) else { return };
    let Some(exp) = exported(&full) else {
        eprintln!("SKIP：没有可用的 nm（或它读不了 {}）", full.display());
        return;
    };
    assert!(
        exp.len() > 30,
        "只从 {} 里读到 {} 个导出 —— nm 的输出格式大概变了，先修测试",
        full.display(),
        exp.len()
    );

    let only_decl: BTreeSet<String> = decl.difference(&exp).cloned().collect();
    let only_exp: BTreeSet<String> = exp.difference(&decl).cloned().collect();
    assert!(
        only_decl.is_empty() && only_exp.is_empty(),
        "头文件与导出对不上：\n\n{}\n\n{}",
        report("只在头文件里声明、`.so` 里没有", &only_decl),
        report("只在 `.so` 里导出、头文件没声明", &only_exp),
    );

    // `min` 必须是 `full` 的子集：它只能**少**几族符号，不能多出别的。
    let Some(min) = build_product(&[]) else { return };
    let Some(min_exp) = exported(&min) else { return };
    let extra: BTreeSet<String> = min_exp.difference(&exp).cloned().collect();
    assert!(
        extra.is_empty(),
        "min 里有 full 没有的符号（构建参数写错了？）：\n\n{}",
        report("只在 min 里", &extra)
    );
    let missing: BTreeSet<String> = exp.difference(&min_exp).cloned().collect();
    eprintln!(
        "header = {} 个声明 = full 的导出；min 少 {} 个（hw / mpeg12 那几族）。",
        decl.len(),
        missing.len()
    );
}
