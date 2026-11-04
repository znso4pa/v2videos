//! 纯 C 的 ABI 冒烟：`dlopen` 打开**我们自己产的** `.so`，按 `v2v.h` 走一遍全链。
//!
//! 为什么值得单独写：`tests/abi.rs` 调的是 `v2v_capi::` 里的函数，**绕过了头文件** ——
//! 头文件把参数写反了、`size` 字段忘了、`stride` 语义写错了，它照样全绿。
//! 只有 C 编译器能回答「头文件自洽吗」，只有 `dlsym` 能回答「符号真的在 `.so` 里吗」。
//!
//! 缺东西就 SKIP（沿用仓里「缺东西打 SKIP」的约定，别让新机器变红）：`cargo`、`cc`、
//! `.so` 本身。
//!
//! ⚠️ `cargo test` **不会**产出 `cdylib`（只编 rlib 与测试夹具），所以这里自己
//! 嵌套跑 `cargo build`。**两个产物各跑一遍**：`min`（默认 feature）与
//! `full`（`hw,mpeg12`）—— `smoke.c` 里那几段可选检查只在 `full` 上才会真跑。

use std::path::{Path, PathBuf};
use std::process::Command;

mod util;
use util::{build_product, manifest};

/// 样本路径：`V2V_ASF_SAMPLE` 可以是 `:` 分隔的多个，取第一个存在的。
fn sample() -> Option<String> {
    std::env::var("V2V_ASF_SAMPLE")
        .ok()
        .and_then(|s| s.split(':').find(|p| Path::new(p).is_file()).map(String::from))
}

/// 编 `tests/smoke.c`。这一步就是「头文件自洽」的判据：`-Werror` 下编不过就算失败。
fn compile_smoke() -> Option<PathBuf> {
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
        Ok(s) => {
            assert!(s.success(), "编 tests/smoke.c 失败 —— 头文件或 smoke 本身有问题");
            Some(bin)
        }
        Err(e) => {
            eprintln!("SKIP：没有可用的 C 编译器（{cc}）：{e}");
            None
        }
    }
}

/// 跑一遍 C 程序。给了样本就多走 open / 读帧 / seek / demux / 音频那几段；
/// `chain` 为真再走「解码 → 编码 → 封装」那一段（只有 `min` 走 —— 两个产物的
/// 编码 / 封装是同一份代码，跑两遍只是让测试慢一倍）。
fn run_smoke(bin: &Path, dylib: &Path, label: &str, chain: bool) {
    let out_mp4 = std::env::temp_dir().join(format!(
        "v2v_abi_chain_{}_{}.mp4",
        label,
        std::process::id()
    ));
    let _ = std::fs::remove_file(&out_mp4);

    let mut run = Command::new(bin);
    run.arg(dylib);
    let sample = sample();
    if let Some(s) = &sample {
        run.arg(s);
        if chain {
            run.arg(&out_mp4);
        }
    }
    let o = run.output().expect("C 程序跑不起来");
    let stdout = String::from_utf8_lossy(&o.stdout).to_string();
    let stderr = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(
        o.status.success(),
        "[{label}] C smoke 失败（退出码 {:?}）：\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        o.status.code()
    );
    println!("--- {label} ---\n{stdout}");

    match sample {
        // 没样本时只验了四条，别让「全绿」看起来像验了全链。
        None => eprintln!(
            "注意：没设 V2V_ASF_SAMPLE，[{label}] 只验了版本 / probe / 错误通道 / size 契约。"
        ),
        Some(_) if chain => check_chain_output(&out_mp4, label),
        Some(_) => {}
    }
}

#[test]
fn c_smoke_through_dlopen() {
    let Some(bin) = compile_smoke() else { return };
    // 两个产物各跑一遍：`min` 是宿主默认会用的那个，`full` 才有 hw / mpeg12 那两族。
    // `min` 走全链（宿主默认会用的就是它）；`full` 只补 hw 那几段可选检查。
    for (label, feats, chain) in [
        ("min", &[][..], true),
        ("full", &["hw", "mpeg12"][..], false),
    ] {
        let Some(dylib) = build_product(feats) else {
            let _ = std::fs::remove_file(&bin);
            return;
        };
        run_smoke(&bin, &dylib, label, chain);
    }
    let _ = std::fs::remove_file(&bin);
}

/// C 写出来的 mp4 必须**真能被外部工具解开** —— 自己再读一遍自己的文件不算数。
///
/// `ffprobe` 说形状（两条轨、各 4 帧），`ffmpeg` 说「真能解」（解一遍到 null，
/// 有错就非零退出）。缺工具就 SKIP 那半段，但会明说。
fn check_chain_output(mp4: &Path, label: &str) {
    let bytes = std::fs::read(mp4).unwrap_or_else(|e| panic!("[{label}] C 没写出 {mp4:?}：{e}"));
    assert!(bytes.len() > 32, "[{label}] 产物太小：{} 字节", bytes.len());
    assert_eq!(&bytes[4..8], b"ftyp", "[{label}] 第一个 box 不是 ftyp");
    assert!(bytes.windows(4).any(|w| w == b"mdat"), "[{label}] 没找到 mdat");

    let Ok(probe) = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,nb_frames",
            "-of",
            "default=nw=1",
        ])
        .arg(mp4)
        .output()
    else {
        eprintln!("SKIP：没有 ffprobe，跳过「外部工具认得这个 mp4」那半段。");
        let _ = std::fs::remove_file(mp4);
        return;
    };
    let info = String::from_utf8_lossy(&probe.stdout).to_string();
    assert!(probe.status.success(), "[{label}] ffprobe 读不了 C 写的 mp4：{info}");
    assert!(info.contains("codec_name=mpeg4"), "[{label}] 视频轨不是 mpeg4：\n{info}");
    assert!(info.contains("codec_name=aac"), "[{label}] 音频轨不是 aac：\n{info}");
    assert_eq!(
        info.matches("nb_frames=4").count(),
        2,
        "[{label}] 两轨的帧数都该是 4：\n{info}"
    );

    // 「声明了 mpeg4」和「真能解」是两件事：解一遍到 null，有错就非零退出。
    if let Ok(dec) = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(mp4)
        .args(["-f", "null", "-"])
        .output()
    {
        let err = String::from_utf8_lossy(&dec.stderr);
        assert!(
            dec.status.success() && err.trim().is_empty(),
            "[{label}] ffmpeg 解不动 C 写的流：\n{err}"
        );
    }

    let _ = std::fs::remove_file(mp4);
}
