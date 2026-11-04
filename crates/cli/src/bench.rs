//! `v2v bench` —— 把测量写成 TSV，可选地对着基线卡回归。
//!
//! ## 它测什么
//!
//! | 用例 | 样本 | 走哪条 |
//! |---|---|---|
//! | `decode/video/<stem>/<w>x<h>` | `.wmv`（WMV3） | [`Player::next_video`] |
//! | `decode/audio/<stem>/<rate>hz` | `.wmv`（有音频轨） | [`Player::next_audio`] |
//! | `transcode/mp4/<stem>` | `.wmv` | [`convert_file`]（`-g 12`、AAC 128k、不校验） |
//! | `decode/mpeg12/<stem>/<w>x<h>` | `.mpg` / `.mpeg` / `.dat` / `.vob` | `MpgPlayer::next_video` |
//!
//! ## 它不测什么
//!
//! - **MP43 / WMV1 / WMV2 的解码**：`Player` 只认 WMV3，而重写一遍解封装循环就等于
//!   多一份「同一件事的两个说法」。这些样本请用 `crates/msmpeg4/tests/decode_bench.rs`
//!   （那份还会同时量 ffmpeg 那一侧）。`v2v bench` 遇到会**明确跳过并说原因**，
//!   不会静默少一行。
//! - **编码器单独的耗时**：`transcode/mp4` 是解码 + 编码 + 封装 + 落盘一整条。
//!   要拆开看哪一段慢，用 `android-encode` 的分段计时。
//!
//! ## 口径（与 README 那些表一致）
//!
//! - `Player::open` / `MpgPlayer::open` 在**计时之外**（解封装不算）；
//! - 每个用例跑 `--reps` 遍，取**最快**的一遍（机器上有别的负载时，最小值最接近
//!   「没有争抢」的那个数）；
//! - 参考实现（`ffmpeg`）用 `-threads 1` + `-map 0:v -an` / `-map 0:a -vn`，
//!   并且**两边都减掉「处理 1 个单位」的那一趟**再折算成每单位毫秒 ——
//!   不减的话，短样本上算出来的「我们快 N 倍」全是进程启动灌水的分母。
//!
//! ## 为什么不在这里卡绝对毫秒
//!
//! 见 `v2v_bench::Thresholds` 的文档：绝对量只提醒，**比值才当门**。

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use v2v_bench::{compare, Fingerprint, Kind, Report, Row, Severity, Thresholds};
use v2v_player::Player;

use crate::{convert_file, ConvertOptions, Mp4Audio, OutputFormat};

/// 一次 `v2v bench` 的参数。
#[derive(Debug, Clone)]
pub struct Options {
    /// TSV 写到文件（`None` = stdout）。
    pub out: Option<PathBuf>,
    /// 跑完与这份基线比，有 `Fail` 就返回 `Ok(false)`。
    pub check: Option<PathBuf>,
    /// 把这次的结果写成 `<dir>/<机器指纹>.tsv`（建目录）。
    pub baseline_dir: Option<PathBuf>,
    pub thresholds: Thresholds,
    /// 每个用例跑几遍取最快（默认 3）。转码用例也会跑这么多遍，大样本请给 1。
    pub reps: usize,
    /// 每个用例最多处理多少个单位（默认全部）。
    ///
    /// ⚠️ 音频用例的单位是**每声道样本数**，但截断按解码帧粒度走（一帧 1024 / 2048 个
    /// 样本），所以实际会略超。截得太狠时 `ref_per_unit` 会留空 —— 见
    /// [`ffmpeg_per_unit`] 的说明。
    pub frames: Option<usize>,
    /// 只跑 case 名以这些前缀开头的用例（空 = 全跑）。
    pub cases: Vec<String>,
    /// 是否额外跑 `ffmpeg` 作为参考实现（默认开；没装就留空）。
    pub with_ref: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            out: None,
            check: None,
            baseline_dir: None,
            thresholds: Thresholds::default(),
            reps: 3,
            frames: None,
            cases: Vec::new(),
            with_ref: true,
        }
    }
}

impl Options {
    fn wants(&self, case: &str) -> bool {
        self.cases.is_empty() || self.cases.iter().any(|p| case.starts_with(p.as_str()))
    }
}

/// 跑一轮基准。
///
/// 返回 `Ok(true)` = 没有超过门限的回归（或没开 `--check`）；
/// `Ok(false)` = 有 `Fail`。`Err` = 用法 / IO / 解码错误。
pub fn run(samples: &[PathBuf], opts: &Options) -> Result<bool, String> {
    if samples.is_empty() {
        return Err("bench 至少给一个样本文件".into());
    }
    if opts.reps == 0 {
        return Err("--reps 要正整数".into());
    }

    let mut report = Report::new();
    let fp = Fingerprint::detect();
    fp.apply(&mut report);
    report.set("tool", "v2v bench");
    // 记一条当时的负载：不参与比较，只让基线**自证来源**。
    if let Some(load) = v2v_bench::load_average() {
        report.set("load", load);
    }
    // debug 的数字没有意义（解码差一个数量级），但**照样能验格式** ——
    // 所以提示一句、继续跑，而不是拒绝。
    if fp.profile != "release" {
        eprintln!(
            "⚠️ 这是 `{}` 构建：数字**不能**与任何基线比。要量性能请 `cargo run --release`。",
            fp.profile
        );
    }
    for (i, p) in samples.iter().enumerate() {
        let meta = std::fs::metadata(p).map_err(|e| format!("读不了 {}：{e}", p.display()))?;
        // 样本标识用**完整文件名**（带扩展名）：`sample.wmv` 与 `sample.avi`
        // 字节数碰巧相同时，光看 stem 分不出来。
        let name = p
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.display().to_string());
        report.add_sample(i, &name, meta.len());
    }

    for p in samples {
        measure_sample(&mut report, p, opts)?;
    }

    let tsv = report.to_tsv();
    match &opts.out {
        Some(path) => {
            std::fs::write(path, &tsv).map_err(|e| format!("写不了 {}：{e}", path.display()))?;
            eprintln!("TSV 已写入 {}", path.display());
        }
        // 只有 TSV 走 stdout —— 人读的进度一律走 stderr，否则 `> x.tsv` 会掺进废话。
        None => print!("{tsv}"),
    }

    if let Some(dir) = &opts.baseline_dir {
        std::fs::create_dir_all(dir).map_err(|e| format!("建不了 {}：{e}", dir.display()))?;
        let path = dir.join(format!("{}.tsv", Fingerprint::detect().slug()));
        std::fs::write(&path, &tsv).map_err(|e| format!("写不了 {}：{e}", path.display()))?;
        eprintln!("基线已写入 {}", path.display());
    }

    let Some(baseline_path) = &opts.check else {
        return Ok(true);
    };
    let text = std::fs::read_to_string(baseline_path)
        .map_err(|e| format!("读不了基线 {}：{e}", baseline_path.display()))?;
    let baseline = Report::parse_tsv(&text)
        .map_err(|e| format!("基线 {} 解析失败：{e}", baseline_path.display()))?;

    let findings = match compare(&baseline, &report, &opts.thresholds) {
        Ok(f) => f,
        Err(m) => {
            return Err(format!(
                "这两份报告**不能比**：{} 不同（基线 `{}`，现在 `{}`）。\n\
                 换机器 / 换样本 / 换 profile 之后的数不能相减 —— 那正是基线按机器分文件的原因。",
                m.field, m.base, m.now
            ))
        }
    };
    let fails = findings
        .iter()
        .filter(|f| f.severity == Severity::Fail)
        .count();
    let warns = findings.len() - fails;
    for f in &findings {
        eprintln!("{}", describe(f, &opts.thresholds));
    }
    if fails > 0 {
        eprintln!("✗ {fails} 条超过门限（比值），{warns} 条提醒");
        Ok(false)
    } else {
        eprintln!("✓ 没有超过门限的回归（{warns} 条提醒）");
        Ok(true)
    }
}

/// 一条发现的**人读**描述。结构化数据在 `v2v_bench` 里，文案在这里 —— 库不该管排版。
fn describe(f: &v2v_bench::Finding, th: &Thresholds) -> String {
    let pct = |base: f64, now: f64| (now / base - 1.0) * 100.0;
    match f.kind {
        Kind::Ratio => format!(
            "✗ {}：我们/参考 的比值 {:.3} → {:.3}（恶化 {:+.0}%，门限 {:.0}%）",
            f.case,
            f.base,
            f.now,
            pct(f.base, f.now),
            th.ratio_pct
        ),
        Kind::Absolute => format!(
            "· {}：{:.3} → {:.3} ms/单位（{:+.0}%，提醒阈值 {:.0}% —— 绝对量不当门）",
            f.case,
            f.base,
            f.now,
            pct(f.base, f.now),
            th.abs_pct
        ),
        Kind::MissingInCurrent => format!("· {}：基线里有、这次没跑", f.case),
        Kind::MissingInBaseline => format!("· {}：这次跑了、基线里没有（新用例，还没有基线）", f.case),
    }
}

/// 样本在 case 名里的标识：文件名去扩展名。
fn file_label(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "sample".into())
}

fn measure_sample(report: &mut Report, path: &Path, opts: &Options) -> Result<(), String> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if matches!(ext.as_str(), "mpg" | "mpeg" | "dat" | "vob") {
        measure_mpeg(report, path, opts)
    } else {
        measure_asf(report, path, opts)
    }
}

fn measure_asf(report: &mut Report, path: &Path, opts: &Options) -> Result<(), String> {
    let stem = file_label(path);
    // 先开一次看静态信息：有没有音频、分辨率多少、是不是我们认的编码。
    let probe = Player::open(path).map_err(|e| {
        format!(
            "{}：{e}\n\
             （`v2v bench` 走的是 `v2v-player`，它只解 WMV3 / VC-1。MP43 / WMV1 / WMV2 的样本\n\
             请用 `crates/msmpeg4/tests/decode_bench.rs` —— 那份还会同时量 ffmpeg 那一侧。）",
            path.display()
        )
    })?;
    let info = probe.info().clone();
    drop(probe);
    let (w, h) = (info.width, info.height);

    let video_case = format!("decode/video/{stem}/{w}x{h}");
    if opts.wants(&video_case) {
        let (n, ms, _) = decode_video(path, opts.frames, opts.reps)?;
        let mut row = Row::new(&video_case, n, ms, "frame");
        if opts.with_ref {
            if let Some(r) = ffmpeg_per_unit(path, false, n, opts.frames, opts.reps) {
                row = row.with_ref(r);
            }
        }
        eprintln!("· {video_case}：{n} 帧，{:.2} ms/帧", row.per_unit());
        report.push(row);
    }

    if info.has_audio && info.audio_sample_rate > 0 {
        let audio_case = format!("decode/audio/{stem}/{}hz", info.audio_sample_rate);
        if opts.wants(&audio_case) {
            let (n, ms) = decode_audio(path, opts.frames, opts.reps)?;
            if n > 0 {
                let mut row = Row::new(&audio_case, n, ms, "sample");
                if opts.with_ref {
                    if let Some(r) = ffmpeg_per_unit(path, true, n, opts.frames, opts.reps) {
                        row = row.with_ref(r);
                    }
                }
                eprintln!("· {audio_case}：{n} 样本，{:.6} ms/样本", row.per_unit());
                report.push(row);
            }
        }
    }

    let tc_case = format!("transcode/mp4/{stem}");
    if opts.wants(&tc_case) {
        let (n, ms) = transcode_mp4(path, opts.frames, opts.reps)?;
        let row = Row::new(&tc_case, n, ms, "frame");
        eprintln!("· {tc_case}：{n} 帧，{:.2} ms/帧", row.per_unit());
        report.push(row);
    }
    Ok(())
}

fn measure_mpeg(report: &mut Report, path: &Path, opts: &Options) -> Result<(), String> {
    use v2v_player::mpeg::MpgPlayer;

    let stem = file_label(path);
    let probe = MpgPlayer::open(path).map_err(|e| format!("{}：{e}", path.display()))?;
    let (w, h) = probe.size();
    drop(probe);

    let case = format!("decode/mpeg12/{stem}/{w}x{h}");
    if !opts.wants(&case) {
        return Ok(());
    }
    let mut best = f64::INFINITY;
    let mut frames = 0u64;
    for _ in 0..opts.reps {
        let mut p = MpgPlayer::open(path).map_err(|e| format!("{}：{e}", path.display()))?;
        let t0 = Instant::now();
        let mut n = 0u64;
        while let Some(_f) = p
            .next_video()
            .map_err(|e| format!("{}：{e}", path.display()))?
        {
            n += 1;
            if opts.frames.is_some_and(|l| n as usize >= l) {
                break;
            }
        }
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
        frames = n;
    }
    let mut row = Row::new(&case, frames, best, "frame");
    if opts.with_ref {
        if let Some(r) = ffmpeg_per_unit(path, false, frames, opts.frames, opts.reps) {
            row = row.with_ref(r);
        }
    }
    eprintln!("· {case}：{frames} 帧，{:.2} ms/帧", row.per_unit());
    report.push(row);
    Ok(())
}

/// 解视频：`(帧数, 最快一轮毫秒, (宽, 高))`。
///
/// ⚠️ `Player::open` 在**计时之外** —— 与 `player/examples/bench.rs` 和 README 的表同口径。
fn decode_video(
    path: &Path,
    limit: Option<usize>,
    reps: usize,
) -> Result<(u64, f64, (u32, u32)), String> {
    let mut best = f64::INFINITY;
    let mut frames = 0u64;
    let mut size = (0u32, 0u32);
    for _ in 0..reps {
        let mut p = Player::open(path).map_err(|e| format!("{}：{e}", path.display()))?;
        size = (p.info().width, p.info().height);
        let t0 = Instant::now();
        let mut n = 0u64;
        while let Some(_f) = p
            .next_video()
            .map_err(|e| format!("{}：{e}", path.display()))?
        {
            n += 1;
            if limit.is_some_and(|l| n as usize >= l) {
                break;
            }
        }
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
        frames = n;
    }
    Ok((frames, best, size))
}

/// 解音频：`(每声道样本数, 最快一轮毫秒)`。
fn decode_audio(path: &Path, limit: Option<usize>, reps: usize) -> Result<(u64, f64), String> {
    let mut best = f64::INFINITY;
    let mut samples = 0u64;
    for _ in 0..reps {
        let mut p = Player::open(path).map_err(|e| format!("{}：{e}", path.display()))?;
        let t0 = Instant::now();
        let mut n = 0u64;
        while let Some(f) = p
            .next_audio()
            .map_err(|e| format!("{}：{e}", path.display()))?
        {
            n += f.samples as u64;
            if limit.is_some_and(|l| n as usize >= l) {
                break;
            }
        }
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
        samples = n;
    }
    Ok((samples, best))
}

/// 整条 `.wmv → .mp4`：解码 + 编码 + 封装 + 落盘，**全在计时内**。
///
/// 参数固定成 README 那张表的口径：`-g 12`、AAC 128k、不校验、qscale 6。
/// 换参数就等于换用例，所以它们不开放成命令行开关 —— 要别的口径请写新 case。
fn transcode_mp4(path: &Path, limit: Option<usize>, reps: usize) -> Result<(u64, f64), String> {
    let mut opts = ConvertOptions::default();
    opts.format = OutputFormat::Mp4;
    opts.audio = Mp4Audio::Aac;
    opts.keyframe_interval = 12;
    opts.verify = false;
    opts.max_frames = limit;

    let out = std::env::temp_dir().join(format!(
        "v2v-bench-{}-{}.mp4",
        std::process::id(),
        file_label(path)
    ));
    let mut best = f64::INFINITY;
    let mut frames = 0u64;
    for _ in 0..reps {
        let t0 = Instant::now();
        let rep =
            convert_file(path, &out, &opts).map_err(|e| format!("{}：{e}", path.display()))?;
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
        frames = rep.frames as u64;
    }
    let _ = std::fs::remove_file(&out);
    Ok((frames, best))
}

/// `ffmpeg` 处理一遍要多少毫秒（`reps` 次里最快）。没装 / 跑挂了就 `None`。
fn ffmpeg_ms(path: &Path, audio: bool, limit: Option<usize>, reps: usize) -> Option<f64> {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let mut cmd = Command::new("ffmpeg");
        // `-threads 1`：ffmpeg 的 wmv3 解码器默认会开帧级线程，不钉住它量出来的
        // 是「能开几个核」而不是「解码器快慢」。我们的解码器本来就是单线程。
        cmd.args(["-v", "error", "-threads", "1", "-i"]).arg(path);
        if audio {
            cmd.args(["-map", "0:a", "-vn"]);
            if let Some(l) = limit {
                cmd.args(["-frames:a", &l.to_string()]);
            }
        } else {
            cmd.args(["-map", "0:v", "-an"]);
            if let Some(l) = limit {
                cmd.args(["-frames:v", &l.to_string()]);
            }
        }
        // `-f null -` 仍然**跑完解码**（只是把输出丢掉），所以时间可比。
        cmd.args(["-f", "null", "-"]);
        let t0 = Instant::now();
        let out = cmd.output().ok()?;
        if !out.status.success() {
            return None;
        }
        best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Some(best)
}

/// 参考实现的每单位毫秒。
///
/// ⚠️ 两边都**减掉「处理 1 个单位」的那一趟**：ffmpeg 每跑一次要付进程启动
/// （20–30 ms），短样本里它占了大半 —— 不减的话，153 帧的样本会算出「我们快 4 倍」
/// 这种纯属分母灌水的数。减掉之后才是纯吞吐的对比。
///
/// ⚠️ 减完如果**不剩什么**（`full - overhead <= 0`，说明这一趟量的全是启动噪声），
/// 返回 `None` 而不是一个小数 —— **宁可不给参考，也不给一个假数**。
/// 这正是 `--frames` 截得太狠时会发生的事：46 ms 的音频里，30 ms 启动占 65%。
fn ffmpeg_per_unit(
    path: &Path,
    audio: bool,
    n: u64,
    limit: Option<usize>,
    reps: usize,
) -> Option<f64> {
    if n == 0 {
        return None;
    }
    let full = ffmpeg_ms(path, audio, limit, reps)?;
    // 音频用「解 1 个音频帧」（1024 个样本）当启动基线，视频用「解 1 帧」。
    // 单位不同不影响这一步：要减的是**固定开销**，不是每单位成本。
    let overhead = ffmpeg_ms(path, audio, Some(1), reps)?;
    let pure = full - overhead;
    if pure <= 0.0 {
        return None;
    }
    Some(pure / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_filter_matches_by_prefix() {
        let mut o = Options::default();
        assert!(o.wants("decode/video/x/1280x720"), "空过滤 = 全跑");
        o.cases = vec!["decode".into()];
        assert!(o.wants("decode/video/x/1280x720"));
        assert!(!o.wants("transcode/mp4/x"));
        o.cases = vec!["transcode".into()];
        assert!(o.wants("transcode/mp4/x"));
        assert!(!o.wants("decode/video/x/1280x720"));
    }

    #[test]
    fn file_label_drops_extension_only() {
        assert_eq!(file_label(Path::new("/a/b/sample.wmv")), "sample");
        assert_eq!(file_label(Path::new("x.y.mpg")), "x.y");
    }

    #[test]
    fn rejects_empty_input_and_zero_reps() {
        assert!(run(&[], &Options::default()).is_err());
        let o = Options {
            reps: 0,
            ..Options::default()
        };
        assert!(run(&[PathBuf::from("/dev/null")], &o).is_err());
    }
}
