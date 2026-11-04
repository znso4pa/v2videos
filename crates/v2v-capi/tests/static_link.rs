//! `staticlib` 是 `Cargo.toml` 里声明过的第三种产物，但**从来没人静态链过一次**。
//!
//! 声明了却不验，等于给宿主一个承诺而不兑现：静态链接的坑全在**系统库**上 ——
//! 我们的 `.a` 里有 C shim（VideoToolbox / MediaCodec），它们引用的系统符号
//! 必须由**调用方**补上，而 `cargo` 只在编 `cdylib` 时替我们加。
//!
//! 所以这里不写死「macOS 要哪几个 framework、Linux 要哪几个 .so」——那是猜。
//! 直接问 rustc：
//!
//! ```text
//! cargo rustc -p v2v-capi --features hw,mpeg12 -- --print native-static-libs
//! note: native-static-libs: -framework VideoToolbox ... -lSystem -lc -lm
//! ```
//!
//! 拿它报的那串去链，链得过、跑得通，才算这个产物真的能用；顺带那串就是
//! 宿主需要抄进自己构建脚本的东西（写进 README 的是这句话，不是某个平台的清单）。
//!
//! 同一份 `tests/smoke.c`，`-DV2V_STATIC` 之后符号在**链接期**解析 ——
//! 于是「静态库里缺某个声明过的函数」变成链接失败，链接器就是判据。

use std::path::PathBuf;
use std::process::Command;

mod util;
use util::{manifest, staticlib_in};

/// 编静态库，并问出「链它要补哪些系统库」。
fn build_static(feats: &[&str]) -> Option<(PathBuf, Vec<String>)> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut b = Command::new(&cargo);
    b.args(["rustc", "-p", "v2v-capi"]);
    if !feats.is_empty() {
        b.arg("--features").arg(feats.join(","));
    }
    b.args(["--", "--print", "native-static-libs"]);
    let out = b.output().expect("调不动 cargo");
    assert!(
        out.status.success(),
        "编静态库失败（{feats:?}）：\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // ⚠️ 这条 note 走 stderr。缓存命中也照样重放（验过），所以不用先 clean。
    let stderr = String::from_utf8_lossy(&out.stderr);
    let flags: Vec<String> = stderr
        .lines()
        .find_map(|l| l.trim().strip_prefix("note: native-static-libs:"))
        .map(|s| s.split_whitespace().map(String::from).collect())
        .unwrap_or_default();
    if flags.is_empty() {
        eprintln!("SKIP：rustc 没报 native-static-libs（版本太老？），静态链接要补哪些库无从得知");
        return None;
    }

    let dir = util::debug_dir()?;
    let lib = staticlib_in(&dir);
    if lib.is_none() {
        eprintln!("SKIP：{} 下没找到 libv2v_capi.a", dir.display());
    }
    lib.map(|l| (l, flags))
}

#[test]
fn static_lib_links_and_runs() {
    // 只用 full：`hw` / `mpeg12` 那几族在 min 里不存在，而静态引用是链接期的
    // 硬引用 —— 用 min 会直接链接失败（那正是「静态链接没有可选符号」的体现）。
    let Some((lib, native)) = build_static(&["hw", "mpeg12"]) else { return };

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let bin = std::env::temp_dir().join(format!("v2v_static_smoke_{}", std::process::id()));
    let mut c = Command::new(&cc);
    c.arg("-std=c11")
        .args(["-Wall", "-Wextra", "-Werror"])
        .arg("-DV2V_STATIC")
        .arg("-I")
        .arg(manifest().join("include"))
        .arg(manifest().join("tests/smoke.c"))
        .arg(&lib)
        .args(&native)
        .arg("-o")
        .arg(&bin);
    let out = match c.output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("SKIP：没有可用的 C 编译器（{cc}）：{e}");
            return;
        }
    };
    assert!(
        out.status.success(),
        "静态链接失败 —— 要么 `.a` 缺符号，要么 rustc 报的 native-static-libs 不够用：\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // 跑。给样本就走全链（argv[1] 直接是样本，静态模式没有库路径）。
    let mut run = Command::new(&bin);
    let out_mp4 = std::env::temp_dir().join(format!("v2v_static_chain_{}.mp4", std::process::id()));
    let sample = std::env::var("V2V_ASF_SAMPLE")
        .ok()
        .and_then(|s| s.split(':').find(|p| std::path::Path::new(p).is_file()).map(String::from));
    if let Some(s) = &sample {
        run.arg(s).arg(&out_mp4);
    }
    let o = run.output().expect("静态链出来的程序跑不起来");
    let stdout = String::from_utf8_lossy(&o.stdout).to_string();
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    let _ = std::fs::remove_file(&bin);
    let _ = std::fs::remove_file(&out_mp4);
    assert!(
        o.status.success(),
        "静态链接的程序跑失败（退出码 {:?}）：\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        o.status.code()
    );
    assert!(stdout.contains("PASS"), "没打 PASS：\n{stdout}");
    if sample.is_none() {
        eprintln!("注意：没设 V2V_ASF_SAMPLE，静态链接只验了「链得过 + 不用样本那几条」。");
    }
    eprintln!("静态链接要补的系统库（rustc 报的）：{}", native.join(" "));
}
