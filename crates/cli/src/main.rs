//! `v2v` —— WMV/ASF 批量转码命令行。
//!
//! ```text
//! v2v convert [选项] <输入>...
//!
//! 选项：
//!   -o, --out-dir <DIR>   输出目录（默认：输入所在目录）
//!   -q, --qscale <1..31>  量化尺度（默认 6；越大越小越糊）
//!       --mp4             输出 MP4（MPEG-4 Part 2）；默认是 WMV（MP43 in ASF）
//!       --fps <N>         只对 --mp4 有效：VOL 里声明的帧率（默认从时间戳推）
//!   -g, --gop <N>         周期 I 帧间隔，0=只首帧（默认 250；只对 WMV 有效）
//!   -n, --max-frames <N>  最多转多少帧（默认全部；预览/局部转换用）
//!       --no-audio        不搬/不转音频
//!       --no-verify       不校验输出（默认会解码回来比 PSNR；只对 WMV 有效）
//!   -h, --help            帮助
//! ```
//!
//! 退出码：全部成功 0；有失败 1；用法错误 2。

use std::path::PathBuf;
use std::process::ExitCode;

use v2v_cli::{convert_batch, ConvertOptions, OutputFormat};

const USAGE: &str = "\
v2v —— WMV/ASF 批量转码（纯 Rust：解封装 → 解码 → 编码 → 封装）

用法：
  v2v convert [选项] <输入>...

选项：
  -o, --out-dir <DIR>   输出目录（默认：与输入同目录）
  -q, --qscale <1..31>  量化尺度（默认 6；越大文件越小、越糊）
      --mp4             输出 .mp4（MPEG-4 Part 2 + PCM），默认输出 .wmv（MP43 in ASF）
      --fps <N>         --mp4 专用：VOL 里声明的帧率（默认从样本时间戳推）
  -g, --gop <N>         周期 I 帧间隔（默认 250；0 = 只首帧是 I）—— 只对 .wmv 有效
  -n, --max-frames <N>  最多转多少帧（默认全部）
      --no-audio        不搬/不转音频
      --no-verify       跳过输出校验（默认解码回来与源比 PSNR）—— 只对 .wmv 有效
  -h, --help            显示本帮助

输入支持：ASF（.wmv）/ AVI 里的 MP43 视频轨 + WMA 音频轨。

两条输出路：
  .wmv  视频重编码成 MP43，音频**直通**（零损失）。会做 PSNR 校验。
  .mp4  视频重编码成 MPEG-4 Part 2（**全 I 帧**，文件较大），音频解成 PCM。
        MP43 / WMA 都进不了 MP4 容器，所以这条路**必须**重编码；
        我们只有 mp4v 编码器没有解码器，所以这条路**不做** PSNR 校验。
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
                out_dir = Some(PathBuf::from(it.next().ok_or("--out-dir 后面要跟目录")?));
            }
            "-q" | "--qscale" => {
                let v: u8 = it
                    .next()
                    .ok_or("--qscale 后面要跟数字")?
                    .parse()
                    .map_err(|_| "--qscale 要 1..=31 的整数".to_string())?;
                opts.qscale = v;
            }
            "--mp4" => opts.format = OutputFormat::Mp4,
            "--fps" => {
                let v: u32 = it
                    .next()
                    .ok_or("--fps 后面要跟数字")?
                    .parse()
                    .map_err(|_| "--fps 要正整数".to_string())?;
                if v == 0 {
                    return Err("--fps 要正整数".into());
                }
                opts.fps = Some(v);
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
            "--no-audio" => opts.copy_audio = false,
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

    let verify = match opts.format {
        OutputFormat::Asf if opts.verify => "开",
        OutputFormat::Asf => "关",
        // `.mp4` 没有解码器可比，见 lib.rs 的模块文档。
        OutputFormat::Mp4 => "不支持（.mp4 无解码器）",
    };
    eprintln!(
        "v2v：{} 个文件 → {}（{}，qscale {}，GOP {}，校验 {}）",
        inputs.len(),
        out_dir.display(),
        match opts.format {
            OutputFormat::Asf => ".wmv / MP43",
            OutputFormat::Mp4 => ".mp4 / MPEG-4 Part 2（全 I 帧）",
        },
        opts.qscale,
        if opts.format == OutputFormat::Mp4 {
            "不适用".to_string()
        } else if opts.keyframe_interval == 0 {
            "关".into()
        } else {
            opts.keyframe_interval.to_string()
        },
        verify
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
                    "  ✓ {} → {}：{} 帧（{} 关键帧）+ {} 个音频样本，{} → {} 字节，PSNR {}",
                    input.display(),
                    out.display(),
                    rep.frames,
                    rep.key_frames,
                    rep.audio_payloads,
                    rep.bytes_in,
                    rep.bytes_out,
                    psnr
                );
                if rep.skipped_audio > 0 {
                    eprintln!(
                        "      （跳过 {} 个音频样本：--no-audio）",
                        rep.skipped_audio
                    );
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("  ✗ {}：{e}", input.display());
            }
        }
    }
    eprintln!("完成：{} 成功 / {} 失败", results.len() - failed, failed);
    Ok(failed == 0)
}
