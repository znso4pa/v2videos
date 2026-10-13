//! `v2v` —— WMV/ASF 批量转码命令行。
//!
//! ```text
//! v2v convert [选项] <输入>...
//!
//! 选项：
//!   -o, --out-dir <DIR>   输出目录（默认：输入所在目录）
//!   -q, --qscale <1..31>  量化尺度（默认 6；越大越小越糊）
//!       --threads <N>     编码线程数（默认 1）
//!       --mp4             输出 MP4（MPEG-4 Part 2）；默认是 WMV（MP43 in ASF）
//!       --video-bitrate N 只对 --mp4 有效：视频目标码率（bit/s），逐帧调 qscale 去够它
//!       --fps <N>         只对 --mp4 有效：VOL 里声明的帧率（默认从时间戳推）
//!   -g, --gop <N>         周期 I 帧间隔，0=只首帧（默认 250）
//!   -n, --max-frames <N>  最多转多少帧（默认全部；预览/局部转换用）
//!       --no-audio        不搬/不转音频
//!       --no-verify       不校验输出（默认会解码回来比 PSNR；只对 WMV 有效）
//!   -h, --help            帮助
//! ```
//!
//! 退出码：全部成功 0；有失败 1；用法错误 2。

use std::path::PathBuf;
use std::process::ExitCode;

use v2v_cli::{convert_batch, ConvertOptions, Mp4Audio, OutputFormat};

const USAGE: &str = "\
v2v —— WMV/ASF 批量转码（纯 Rust：解封装 → 解码 → 编码 → 封装）

用法：
  v2v convert [选项] <输入>...

选项：
  -o, --out-dir <DIR>   输出目录（默认：与输入同目录）
  -q, --qscale <1..31>  量化尺度（默认 6；越大文件越小、越糊）
      --mp4             输出 .mp4（MPEG-4 Part 2 + AAC），默认输出 .wmv（MP43 in ASF）
      --pcm             .mp4 的音频用未压缩 PCM（sowt）而不是 AAC
      --aac-bitrate N   .mp4 走 AAC 时的目标码率（默认 128000 bit/s）
      --threads N       编码线程数（默认 1）。现在只有 AAC 的**声道并行**吃它
                        （双声道 → 2；封顶到声道数，单声道给多少都是 1）。
                        输出与 1 线程**逐字节相同**，只是两个声道同时算。
      --video-bitrate N .mp4 的**视频**目标码率（bit/s；也可写 500k / 1.5M）。
                        开了就逐帧调 qscale 去够它，`-q` 退化成起步值。
                        不传 = 按固定的 -q 编（旧行为）。
      --fps <N>         --mp4 专用：VOL 里声明的帧率（默认从样本时间戳推）
  -g, --gop <N>         周期 I 帧间隔（默认 250；0 = 只首帧是 I）
                        对 .mp4 也有效 —— 传 0 就是全 I 帧（大但每帧独立可解）
  -n, --max-frames <N>  最多转多少帧（默认全部）
      --no-audio        不搬/不转音频
      --no-verify       跳过输出校验（默认解码回来与源比 PSNR）—— 只对 .wmv 有效
  -h, --help            显示本帮助

输入支持：ASF（.wmv）/ AVI 里的 MP43 视频轨 + WMA 音频轨。

两条输出路：
  .wmv  视频重编码成 MP43，音频**直通**（零损失）。会做 PSNR 校验。
  .mp4  视频重编码成 MPEG-4 Part 2（I+P，GOP 由 -g 控制），
        音频 WMA → PCM → **AAC-LC**（默认；`--pcm` 可退回未压缩 PCM）。
        MP43 / WMA 都进不了 MP4 容器，所以这条路**必须**重编码；
        我们只有 mp4v 编码器没有解码器，所以这条路**不做** PSNR 校验。
        -g 0 会编成全 I 帧（文件大得多，但每帧独立可解、适合剪辑）。
";

/// 解析码率：纯数字按 bit/s 算，也接受 `k` / `M` 后缀（`500k`、`1.5M`）。
fn parse_bitrate(s: &str) -> Result<u32, String> {
    let t = s.trim();
    let (num, mul) = match t.chars().last() {
        Some('k') | Some('K') => (&t[..t.len() - 1], 1_000.0),
        Some('m') | Some('M') => (&t[..t.len() - 1], 1_000_000.0),
        _ => (t, 1.0),
    };
    let v: f64 = num
        .parse()
        .map_err(|_| format!("--video-bitrate 看不懂 `{s}`（要 bit/s，也可写 500k / 1.5M）"))?;
    let bits = v * mul;
    if !(bits >= 1.0 && bits <= f64::from(u32::MAX)) {
        return Err(format!("--video-bitrate `{s}` 超出范围"));
    }
    Ok(bits.round() as u32)
}

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
            "--pcm" => opts.audio = Mp4Audio::Pcm,
            "--aac-bitrate" => {
                opts.aac_bitrate = it
                    .next()
                    .ok_or("--aac-bitrate 后面要跟数字")?
                    .parse()
                    .map_err(|_| "--aac-bitrate 要正整数（bit/s）".to_string())?;
            }
            "--threads" => {
                let v: usize = it
                    .next()
                    .ok_or("--threads 后面要跟数字")?
                    .parse()
                    .map_err(|_| "--threads 要正整数".to_string())?;
                if v == 0 {
                    return Err("--threads 要正整数".into());
                }
                opts.threads = v;
            }
            "--video-bitrate" => {
                let v = it.next().ok_or("--video-bitrate 后面要跟数字")?;
                let n = parse_bitrate(v)?;
                opts.video_bitrate = Some(n);
            }
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
    let vbr = match opts.video_bitrate {
        Some(b) => format!("，视频目标 {} kbps", b / 1000),
        None => String::new(),
    };
    eprintln!(
        "v2v：{} 个文件 → {}（{}，qscale {}，GOP {}，校验 {}{}）",
        inputs.len(),
        out_dir.display(),
        match opts.format {
            OutputFormat::Asf => ".wmv / MP43 + 音频直通".to_string(),
            OutputFormat::Mp4 => match opts.audio {
                Mp4Audio::Aac => format!(".mp4 / MPEG-4 Part 2 + AAC {}", opts.aac_bitrate),
                Mp4Audio::Pcm => ".mp4 / MPEG-4 Part 2 + PCM".to_string(),
            },
        },
        opts.qscale,
        if opts.keyframe_interval == 0 {
            "关（全 I 帧）".into()
        } else {
            opts.keyframe_interval.to_string()
        },
        verify,
        vbr
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
                    "  ✓ {} → {}：{} 帧（{} 关键帧）+ {} 个音频样本，{} → {} 字节，PSNR {}{}",
                    input.display(),
                    out.display(),
                    rep.frames,
                    rep.key_frames,
                    rep.audio_payloads,
                    rep.bytes_in,
                    rep.bytes_out,
                    psnr,
                    match rep.video_kbps {
                        // 码率控制开着才有 —— 这是「有没有够到目标」的唯一凭据。
                        Some(k) => format!("，视频实际 {k:.0} kbps"),
                        None => String::new(),
                    }
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

#[cfg(test)]
mod tests {
    use super::parse_bitrate;

    #[test]
    fn parses_bitrate_with_optional_suffix() {
        assert_eq!(parse_bitrate("300000").unwrap(), 300_000);
        assert_eq!(parse_bitrate("500k").unwrap(), 500_000);
        assert_eq!(parse_bitrate("1.5M").unwrap(), 1_500_000);
        assert_eq!(parse_bitrate(" 2K ").unwrap(), 2_000);
        assert_eq!(parse_bitrate("128k").unwrap(), 128_000);
    }

    #[test]
    fn rejects_garbage_and_out_of_range() {
        for bad in ["", "abc", "12x", "0", "-5", "999999M"] {
            assert!(
                parse_bitrate(bad).is_err(),
                "`{bad}` 不该被接受，却解析成了 {:?}",
                parse_bitrate(bad)
            );
        }
    }
}
