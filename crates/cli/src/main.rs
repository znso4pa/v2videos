//! `v2v` —— WMV/ASF 批量转码 + 基准测量命令行。
//!
//! ```text
//! v2v convert [选项] <输入>...
//! v2v bench   [选项] <样本>...
//!
//! convert 的选项：
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
//!
//! bench 的选项见 `BENCH_USAGE`。
//! ```
//!
//! 退出码：全部成功 0；有失败 / 有超过门限的性能回归 1；用法错误 2。

use std::path::PathBuf;
use std::process::ExitCode;

use v2v_cli::bench::{self, Options as BenchOptions};
use v2v_cli::{convert_batch, ConvertOptions, Mp4Audio, OutputFormat};

const USAGE: &str = "\
v2v —— WMV/ASF 批量转码（纯 Rust：解封装 → 解码 → 编码 → 封装）

用法：
  v2v convert [选项] <输入>...
  v2v bench   [选项] <样本>...   （性能测量：打 TSV、可选对着基线卡回归）

选项：
  -o, --out-dir <DIR>   输出目录（默认：与输入同目录）
  -q, --qscale <1..31>  量化尺度（默认 6；越大文件越小、越糊）
      --mp4             输出 .mp4（MPEG-4 Part 2 + AAC），默认输出 .wmv（MP43 in ASF）
      --pcm             .mp4 的音频用未压缩 PCM（sowt）而不是 AAC
      --aac-bitrate N   .mp4 走 AAC 时的目标码率（默认 128000 bit/s）
      --no-ms           关掉 AAC 的 M/S 立体声（默认开；只为对比测量）
      --threads N       编码线程数（默认 1）。`>1` 时两条输出路都走 **GOP 并行编码**：
                        每个 GOP 一个独立编码器（.wmv 产物仍与 1 线程逐字节相同；.mp4
                        不相同——GOP 边界重置运动估计/码控，差几字节到几十 KB）。
                        .mp4 且输入是 WMV3 时**解码也并行**（每条 GOP 一条全新解码器、
                        从 I 帧独立解）。MP43 也验证过等价，但它解码便宜、开池实测更慢，
                        所以仍是串行（与编码 worker 重叠着跑）；WMV1/WMV2 有跨帧状态，不开。
                        并行度上限是 **GOP 数**——短视频或默认大 GOP 下 GOP 数 < 线程数时
                        加速封顶（要更多并行就把 `-g` 调小或喂长视频）。AAC 仍按声道并行。
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

输入支持：ASF（.wmv）/ AVI 里的 MP42 / MP43 / WMV1 / WMV2 / mp4v（FMP4·DIVX·XVID）视频轨 + WMA 音频轨。

两条输出路：
  .wmv  视频重编码成 MP43，音频**直通**（零损失）。会做 PSNR 校验。
  .mp4  视频重编码成 MPEG-4 Part 2（I+P，GOP 由 -g 控制），
        音频 WMA → PCM → **AAC-LC**（默认；`--pcm` 可退回未压缩 PCM）。
        MP43 / WMA 都进不了 MP4 容器，所以这条路**必须**重编码；
        我们只有 mp4v 编码器没有解码器，所以这条路**不做** PSNR 校验。
        -g 0 会编成全 I 帧（文件大得多，但每帧独立可解、适合剪辑）。
";

/// `v2v bench` 的帮助。**单独一份**，因为它和 convert 的选项一个都不重叠 ——
/// 混在一个 USAGE 里会让人以为 `--out-dir` 对 bench 也有意义。
const BENCH_USAGE: &str = "\
v2v bench —— 性能测量：把结果写成 TSV，可选对着基线卡回归

用法：
  v2v bench [选项] <样本>...        （.wmv = VC-1 / WMV3；.mpg = MPEG-1/2）

选项：
      --out <FILE>          TSV 写到文件（默认 stdout）
      --check <BASELINE>    跑完与这份基线比；有超过门限的**比值**回归就退出码 1
      --baseline-dir <DIR>  把这次结果写成 <DIR>/<机器指纹>.tsv（目录不存在会建）
      --ratio-tolerance P   比值恶化的门限百分比（默认 15）—— 这条**会拦住人**
      --abs-tolerance P     绝对量恶化的提醒阈值（默认 30）—— 这条**只提醒**
      --reps N              每个用例跑 N 遍取最快（默认 3；大样本转码建议 1）
      --frames N            每个用例最多处理 N 个单位（默认全部）
      --cases LIST          只跑 case 名以这些前缀开头的（逗号分隔，如 decode,transcode）
      --no-ref              不跑 ffmpeg 参考实现（TSV 里 ref_per_unit 会留空）
  -h, --help                显示本帮助

用例（TSV 的 `case` 列）：
  decode/video/<样本>/<w>x<h>       VC-1 解码（Player::open 在计时之外）
  decode/audio/<样本>/<rate>hz      WMA 解码（单位是**每声道样本数**）
  transcode/mp4/<样本>              整条 .wmv → .mp4（-g 12、AAC 128k、不校验）
  decode/mpeg12/<样本>/<w>x<h>      MPEG-1/2 解码

为什么分「门」和「提醒」：
  绝对毫秒在 CI runner 上没有意义（同一份代码换台机器差 2×），只有
  「我们 / 参考实现」的比值对机器不敏感。所以比值恶化超过 --ratio-tolerance
  才失败；绝对量只打印提醒。基线按机器指纹分文件，换机器就不比 —— 见
  `bench/baseline/` 与 `crates/bench` 的模块文档。

例：
  # 量一遍，存成基线
  v2v bench --baseline-dir bench/baseline testing/sample.wmv
  # 改完代码再量，对着基线看有没有退化
  v2v bench --check bench/baseline/$(...).tsv testing/sample.wmv
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

/// `Ok(true)` = 全部成功（或没有超过门限的回归）；`Ok(false)` = 有文件失败 / 有回归。
fn run(args: &[String]) -> Result<bool, String> {
    let mut it = args.iter();
    let cmd = it.next().ok_or("缺少子命令（convert / bench）")?;
    match cmd.as_str() {
        "-h" | "--help" => {
            print!("{USAGE}");
            Ok(true)
        }
        "convert" => run_convert(&mut it),
        "bench" => run_bench(&mut it),
        other => Err(format!("未知子命令 `{other}`（有 convert / bench）")),
    }
}

/// `v2v bench`：解析参数后交给 `v2v_cli::bench`。
///
/// 这里只做**参数**的事 —— 测量、写 TSV、比基线全在库里，那样它才能被测试直接调用
/// （见 `crates/cli/src/bench.rs` 的 `mod tests`）。
fn run_bench(it: &mut std::slice::Iter<'_, String>) -> Result<bool, String> {
    let mut opts = BenchOptions::default();
    let mut samples: Vec<PathBuf> = Vec::new();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => {
                opts.out = Some(PathBuf::from(it.next().ok_or("--out 后面要跟文件")?));
            }
            "--check" => {
                opts.check = Some(PathBuf::from(it.next().ok_or("--check 后面要跟基线文件")?));
            }
            "--baseline-dir" => {
                opts.baseline_dir = Some(PathBuf::from(
                    it.next().ok_or("--baseline-dir 后面要跟目录")?,
                ));
            }
            "--ratio-tolerance" => {
                opts.thresholds.ratio_pct = parse_pct(it.next(), "--ratio-tolerance")?;
            }
            "--abs-tolerance" => {
                opts.thresholds.abs_pct = parse_pct(it.next(), "--abs-tolerance")?;
            }
            "--reps" => {
                let v: usize = it
                    .next()
                    .ok_or("--reps 后面要跟数字")?
                    .parse()
                    .map_err(|_| "--reps 要正整数".to_string())?;
                if v == 0 {
                    return Err("--reps 要正整数".into());
                }
                opts.reps = v;
            }
            "--frames" => {
                let v: usize = it
                    .next()
                    .ok_or("--frames 后面要跟数字")?
                    .parse()
                    .map_err(|_| "--frames 要正整数".to_string())?;
                if v == 0 {
                    return Err("--frames 要正整数".into());
                }
                opts.frames = Some(v);
            }
            "--cases" => {
                let v = it.next().ok_or("--cases 后面要跟逗号分隔的前缀")?;
                opts.cases = v
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();
                if opts.cases.is_empty() {
                    return Err("--cases 给了一串空前缀".into());
                }
            }
            "--no-ref" => opts.with_ref = false,
            "-h" | "--help" => {
                print!("{BENCH_USAGE}");
                return Ok(true);
            }
            other if other.starts_with('-') => {
                return Err(format!("未知选项 `{other}`"));
            }
            _ => samples.push(PathBuf::from(a)),
        }
    }
    bench::run(&samples, &opts)
}

/// 解析一个百分比（`15` / `15%` 都收）。
fn parse_pct(v: Option<&String>, flag: &str) -> Result<f64, String> {
    let raw = v.ok_or_else(|| format!("{flag} 后面要跟百分比"))?;
    let t = raw.trim().trim_end_matches('%');
    let n: f64 = t
        .parse()
        .map_err(|_| format!("{flag} 要一个非负数（可以带 %），读到 `{raw}`"))?;
    if !(n >= 0.0 && n.is_finite()) {
        return Err(format!("{flag} 要一个非负数，读到 `{raw}`"));
    }
    Ok(n)
}

/// `v2v convert`：解析参数后交给 `v2v_cli::convert_batch`。
fn run_convert(it: &mut std::slice::Iter<'_, String>) -> Result<bool, String> {
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
            "--no-ms" => opts.ms_stereo = false,
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
        // `.mp4` 这条路**还没接**校验（不是「没有解码器」—— `v2v-mp4v` 有解码器，
        // 只是 CLI 没把产物解回来比 PSNR）。见 lib.rs 的模块文档。
        OutputFormat::Mp4 => "未实现（.mp4 这条路没接 PSNR 校验）",
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
