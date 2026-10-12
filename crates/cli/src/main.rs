//! `v2v` —— WMV/ASF 批量转码命令行。
//!
//! ```text
//! v2v convert [选项] <输入>...
//!
//! 选项：
//!   -o, --out-dir <DIR>   输出目录（默认：输入所在目录）
//!   -q, --qscale <1..31>  量化尺度（默认 6；越大越小越糊）
//!   -g, --gop <N>         周期 I 帧间隔，0=只首帧（默认 250）
//!   -n, --max-frames <N>  最多转多少帧（默认全部；预览/局部转换用）
//!       --no-verify       不校验输出（默认会解码回来比 PSNR）
//!   -h, --help            帮助
//! ```
//!
//! 退出码：全部成功 0；有失败 1；用法错误 2。

use std::path::PathBuf;
use std::process::ExitCode;

use v2v_cli::{convert_batch, ConvertOptions};

const USAGE: &str = "\
v2v —— WMV/ASF 批量转码（纯 Rust：解封装 → 解码 → 编码 → 封装）

用法：
  v2v convert [选项] <输入>...

选项：
  -o, --out-dir <DIR>   输出目录（默认：与输入同目录）
  -q, --qscale <1..31>  量化尺度（默认 6；越大文件越小、越糊）
  -g, --gop <N>         周期 I 帧间隔（默认 250；0 = 只首帧是 I）
  -n, --max-frames <N>  最多转多少帧（默认全部）
      --no-verify       跳过输出校验（默认解码回来与源比 PSNR）
  -h, --help            显示本帮助

输入支持：ASF（.wmv）里的 MP43 视频轨。音频轨暂不搬。
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(msg) => {
            eprintln!("用法错误：{msg}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// `Ok(true)` = 全部成功；`Ok(false)` = 有文件失败。
fn run(args: &[String]) -> Result<bool, String> {
    let mut it = args.iter();
    let cmd = it.next().ok_or("缺少子命令（目前只有 convert）")?;
    if cmd == "-h" || cmd == "--help" {
        print!("{USAGE}");
        return Ok(true);
    }
    if cmd != "convert" {
        return Err(format!("未知子命令 `{cmd}`"));
    }

    let mut opts = ConvertOptions::default();
    let mut out_dir: Option<PathBuf> = None;
    let mut inputs: Vec<PathBuf> = Vec::new();

    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" | "--out-dir" => {
                out_dir = Some(PathBuf::from(
                    it.next().ok_or("--out-dir 后面要跟目录")?,
                ));
            }
            "-q" | "--qscale" => {
                let v: u8 = it
                    .next()
                    .ok_or("--qscale 后面要跟数字")?
                    .parse()
                    .map_err(|_| "--qscale 要 1..=31 的整数".to_string())?;
                opts.qscale = v;
            }
            "-g" | "--gop" => {
                opts.keyframe_interval = it
                    .next()
                    .ok_or("--gop 后面要跟数字")?
                    .parse()
                    .map_err(|_| "--gop 要非负整数".to_string())?;
            }
            "-n" | "--max-frames" => {
                let v: usize = it
                    .next()
                    .ok_or("--max-frames 后面要跟数字")?
                    .parse()
                    .map_err(|_| "--max-frames 要非负整数".to_string())?;
                opts.max_frames = if v == 0 { None } else { Some(v) };
            }
            "--no-verify" => opts.verify = false,
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(true);
            }
            other if other.starts_with('-') => {
                return Err(format!("未知选项 `{other}`"));
            }
            _ => inputs.push(PathBuf::from(a)),
        }
    }

    if inputs.is_empty() {
        return Err("至少给一个输入文件".into());
    }
    if !(1..=31).contains(&opts.qscale) {
        return Err(format!("--qscale {} 不在 1..=31", opts.qscale));
    }

    // 输出目录：默认取第一个输入所在目录
    let out_dir = match out_dir {
        Some(d) => d,
        None => inputs[0]
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from(".")),
    };
    if !out_dir.is_dir() {
        std::fs::create_dir_all(&out_dir)
            .map_err(|e| format!("建不了输出目录 {}：{e}", out_dir.display()))?;
    }

    eprintln!(
        "v2v：{} 个文件 → {}（qscale {}，GOP {}，校验 {}）",
        inputs.len(),
        out_dir.display(),
        opts.qscale,
        if opts.keyframe_interval == 0 {
            "关".into()
        } else {
            opts.keyframe_interval.to_string()
        },
        if opts.verify { "开" } else { "关" }
    );

    let results = convert_batch(&inputs, &out_dir, &opts);
    let mut failed = 0usize;
    for (input, r) in &results {
        match r {
            Ok((out, rep)) => {
                let psnr = match rep.psnr_db {
                    Some(p) if p.is_finite() => format!("{p:.2} dB"),
                    Some(_) => "无损".into(),
                    None => "未校验".into(),
                };
                eprintln!(
                    "  ✓ {} → {}：{} 帧（{} 关键帧），{} → {} 字节，PSNR {}",
                    input.display(),
                    out.display(),
                    rep.frames,
                    rep.key_frames,
                    rep.bytes_in,
                    rep.bytes_out,
                    psnr
                );
                if rep.skipped_audio > 0 {
                    eprintln!("      （跳过 {} 个音频样本：音频轨暂不搬）", rep.skipped_audio);
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("  ✗ {}：{e}", input.display());
            }
        }
    }
    eprintln!(
        "完成：{} 成功 / {} 失败",
        results.len() - failed,
        failed
    );
    Ok(failed == 0)
}
