//! `v2v scan` —— 递归扫目录，认出**我们能转的**视频，并说明能转成什么。
//!
//! 拿到一个游戏目录 / 素材盘时，逐个试 `v2v convert` 是最慢的办法（每个都要解码
//! 一遍才失败）。这里**只读头部**：容器头 + MP4 的 `moov` + AVI 的 `hdrl`/`idx1`，
//! 一个样本都不读。
//!
//! ## 「能转」是头部级的判断
//!
//! 只读头部意味着这里说的是「**容器声称的编解码器我们有解码器**」，不是
//! 「真的解得开」。一个头部完好、帧数据烂掉的文件在这里仍然报可转 ——
//! 要确认得真跑一遍 `v2v convert`。这条边界写在文档里，不假装扫描器比它更强。
//!
//! ## 为什么不认扩展名
//!
//! 按**魔数**认容器（与 [`crate::open_source`] 同一套判断）。扩展名会骗人：
//! 游戏资源里 `.wmv` 是 MP4、`.bin` 是 ASF 的情况都见过，字节不会骗人。
//!
//! ## 边界
//!
//! - 目录遍历**跳过符号链接**（防环、防同一个文件被数两遍）；命令行上直接点名的照扫。
//! - 读不动的子目录**跳过不报**：扫描器的用途是「找出能转的东西」，不是审计权限。
//! - 认不出 / 文件坏了 → `unknown`，**不 panic、不中断**，继续扫下一个。

use std::path::{Path, PathBuf};

use crate::Error;

/// 一条视频流的格式（头部声明的）。
#[derive(Debug, Clone)]
pub struct VideoInfo {
    /// FourCC 的显示名（`WMV3` / `MP43` / `mp4v` …）。
    pub codec: String,
    pub width: u32,
    pub height: u32,
}

/// 一条音频流的格式。
#[derive(Debug, Clone)]
pub struct AudioInfo {
    /// 编解码器的显示名（`WMA2` / `AAC-LC` / `PCM` / `0x0055` …）。
    pub codec: String,
    pub channels: u16,
    pub sample_rate: u32,
}

/// 结论：这个文件能不能转、能转成什么。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// 能转。`targets` 是能去的输出容器（`"wmv"` = MP43/ASF，`"mp4"` = mp4v/MP4）。
    Convertible { targets: Vec<&'static str> },
    /// 容器认出来了，但某条流我们解不了。
    Unsupported(String),
    /// 连容器都认不出（或者文件根本读不动 / 是坏的）。
    Unknown(String),
}

impl Verdict {
    pub fn is_convertible(&self) -> bool {
        matches!(self, Verdict::Convertible { .. })
    }

    /// 给人看的一句话（表格里 `unknown` / `不支持的编码` 那一栏）。
    fn label(&self) -> String {
        match self {
            Verdict::Convertible { targets } => targets.join(","),
            Verdict::Unsupported(_) => "unsupported".into(),
            Verdict::Unknown(_) => "unknown".into(),
        }
    }

    /// 详细原因（`unsupported` / `unknown` 才有）。
    fn reason(&self) -> Option<&str> {
        match self {
            Verdict::Convertible { .. } => None,
            Verdict::Unsupported(s) | Verdict::Unknown(s) => Some(s),
        }
    }
}

/// 扫描到的一个文件。
#[derive(Debug, Clone)]
pub struct ScanEntry {
    pub path: PathBuf,
    pub bytes: u64,
    /// `"ASF"` / `"AVI"` / `"MP4"`；认不出容器就是 `None`。
    pub container: Option<&'static str>,
    pub video: Option<VideoInfo>,
    pub audio: Option<AudioInfo>,
    /// 容器声明的时长（毫秒）。`None` = 文件没写。
    pub duration_ms: Option<u64>,
    pub verdict: Verdict,
    /// 能转、但有前提（例如「音频不认识，只能 `--no-audio`」）。
    pub note: Option<String>,
}

/// 递归扫这些路径。目录会展开，文件直接探。
///
/// 返回**全部**结果（含不支持的）—— 要不要显示是 [`format_table`] / [`format_tsv`] 的事。
/// 根路径不存在这种「用户打错了」的错误往上抛；单个文件读不动只记进那一条。
pub fn scan(paths: &[PathBuf]) -> Result<Vec<ScanEntry>, Error> {
    let mut files = Vec::new();
    for p in paths {
        let meta = std::fs::metadata(p)?;
        if meta.is_dir() {
            walk(p, &mut files);
        } else {
            files.push(p.clone());
        }
    }
    // 目录遍历的顺序取决于文件系统的返回顺序 —— 排一下，同样的输入永远同样的输出。
    files.sort();
    Ok(files.iter().map(|p| probe(p)).collect())
}

/// 把一个目录树摊平成文件列表（**跳过符号链接**）。
fn walk(root: &Path, out: &mut Vec<PathBuf>) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        // 读不动就跳过这一层 —— 扫描器不该因为一个没权限的目录整个失败。
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                stack.push(e.path());
            } else {
                out.push(e.path());
            }
        }
    }
}

/// 探一个文件：认容器 → 取流 → 对照输入编码表判可转性。
fn probe(path: &Path) -> ScanEntry {
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mut entry = ScanEntry {
        path: path.to_path_buf(),
        bytes,
        container: None,
        video: None,
        audio: None,
        duration_ms: None,
        verdict: Verdict::Unknown("没探".into()),
        note: None,
    };

    // `open_source` 的 `new` 只解析头部（ASF 头 / AVI 的 hdrl+idx1 / MP4 的 moov），
    // 不读样本 —— 这正是扫描器要的成本形状。
    let src = match crate::open_source(path) {
        Ok(s) => s,
        Err(e) => {
            entry.verdict = Verdict::Unknown(e.to_string());
            return entry;
        }
    };

    entry.container = Some(src.container_name());
    entry.duration_ms = src.duration_ms();

    let Some(v) = src.video_stream() else {
        // 纯音频文件（或格式信息缺失）：能开，但没有我们要转的东西。
        entry.verdict = Verdict::Unsupported("没有视频轨".into());
        return entry;
    };
    entry.video = Some(VideoInfo {
        codec: crate::fourcc_string(v.fourcc),
        width: v.width,
        height: v.height,
    });

    let af = src.audio_stream().map(|(_, af)| af);
    entry.audio = af.as_ref().map(|af| AudioInfo {
        codec: audio_codec_name(af.format_tag),
        channels: af.channels,
        sample_rate: af.sample_rate,
    });

    // 视频能不能解 —— 查的是 `open_video_decoder` 那张**同一张表**。
    if let Err(e) = crate::classify_video(v.fourcc) {
        entry.verdict = Verdict::Unsupported(e.to_string());
        return entry;
    }

    let (targets, note) = output_targets(af.as_ref());
    entry.verdict = Verdict::Convertible { targets };
    entry.note = note;
    entry
}

/// 「视频能解」之后，再看音频决定能去哪些输出容器。
///
/// 规则是从两条输出路径的**实际约束**抄下来的，不是猜的：
/// - `.wmv`（ASF/MP43）的音频是**直通**，只收 WMA（`convert_to_asf` 里那条判断）；
/// - `--mp4`（mp4v）的音频要重编码成 AAC，能解 WMA / AAC / 16 位 PCM。
///
/// 两条路都吃「没有音频」，所以没音频的文件两个目标都能去。
fn output_targets(af: Option<&crate::AudioFormat>) -> (Vec<&'static str>, Option<String>) {
    let Some(af) = af else {
        return (vec!["wmv", "mp4"], None);
    };
    match af.format_tag {
        // WMA v1 / v2：`.wmv` 直通，`--mp4` 解成 PCM 再编 AAC。
        0x0160 | 0x0161 => (vec!["wmv", "mp4"], None),
        // AAC：`--mp4` 能**原样搬帧**（无损、也绕开我们 AAC 解码器缺的那几个语法）；
        // `.wmv` 没有 AAC 直通，只能 `--no-audio`。
        0x00FF => (
            vec!["mp4"],
            Some("音频是 AAC：`.wmv` 那条路的音频只直通 WMA，要转它得加 `--no-audio`".into()),
        ),
        // 16 位 PCM：`--mp4` 按采样点直接搬；`.wmv` 同样只收 WMA。
        0x0001 if af.bits_per_sample == 16 => (
            vec!["mp4"],
            Some("音频是 PCM：`.wmv` 那条路的音频只直通 WMA，要转它得加 `--no-audio`".into()),
        ),
        other => (
            vec!["wmv", "mp4"],
            Some(format!(
                "音频是 0x{other:04X}，两条路都不认 —— 只能加 `--no-audio` 只转视频"
            )),
        ),
    }
}

/// `wFormatTag` → 可读名。认不出就报十六进制（**不要**编一个名字）。
fn audio_codec_name(tag: u16) -> String {
    match tag {
        0x0160 => "WMA1".into(),
        0x0161 => "WMA2".into(),
        0x0162 => "WMAPro".into(),
        0x0001 => "PCM".into(),
        0x0055 => "MP3".into(),
        0x00FF => "AAC-LC".into(),
        other => format!("0x{other:04X}"),
    }
}

// ---------------------------------------------------------------------------
// 输出
// ---------------------------------------------------------------------------

/// TSV（可管道）。第一行是表头。
///
/// `all = false` 时只出**能转的**（`v2v scan` 的默认语义：只报我们处理得了的）。
pub fn format_tsv(entries: &[ScanEntry], all: bool) -> String {
    let mut s = String::new();
    s.push_str(
        "path\tcontainer\tvideo\twidth\theight\taudio\tchannels\trate\tduration_ms\tbytes\ttargets\tnote\n",
    );
    for e in entries.iter().filter(|e| all || e.verdict.is_convertible()) {
        let (vcodec, w, h) = match &e.video {
            Some(v) => (v.codec.as_str(), v.width.to_string(), v.height.to_string()),
            None => ("-", "-".into(), "-".into()),
        };
        let (acodec, ch, rate) = match &e.audio {
            Some(a) => (a.codec.as_str(), a.channels.to_string(), a.sample_rate.to_string()),
            None => ("-", "-".into(), "-".into()),
        };
        // 一行一个字段，**字段里不能有制表符**；路径里的换行会把 TSV 弄坏，所以换掉。
        let path = e.path.display().to_string().replace(['\t', '\n', '\r'], "?");
        let note = e
            .note
            .clone()
            .or_else(|| e.verdict.reason().map(str::to_string))
            .unwrap_or_default()
            .replace(['\t', '\n', '\r'], " ");
        s.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            path,
            e.container.unwrap_or("-"),
            vcodec,
            w,
            h,
            acodec,
            ch,
            rate,
            e.duration_ms.map_or("-".to_string(), |d| d.to_string()),
            e.bytes,
            e.verdict.label(),
            note,
        ));
    }
    s
}

/// 人类可读的对齐表格。
pub fn format_table(entries: &[ScanEntry], all: bool) -> String {
    let shown: Vec<&ScanEntry> = entries
        .iter()
        .filter(|e| all || e.verdict.is_convertible())
        .collect();
    if shown.is_empty() {
        return String::new();
    }
    // 路径列按最长的那条定宽，但**封顶**：路径很长时表格会被撑得没法看。
    let path_w = shown
        .iter()
        .map(|e| display_width(&e.path.display().to_string()))
        .max()
        .unwrap_or(10)
        .clamp(10, 60);

    let widths = [path_w, 4, 6, 11, 14, 10, 9, 0];
    // 时长与体积右对齐（数字看个位对齐才好扫），其余左对齐。
    let aligns = [false, false, false, false, false, true, true, false];

    let mut s = String::new();
    s.push_str(&row(
        &[
            "PATH".into(),
            "CONT".into(),
            "VIDEO".into(),
            "SIZE".into(),
            "AUDIO".into(),
            "DURATION".into(),
            "BYTES".into(),
            "TARGET".into(),
        ],
        &widths,
        &aligns,
    ));
    s.push('\n');

    for e in &shown {
        let (vcodec, size) = match &e.video {
            Some(v) => (v.codec.clone(), format!("{}x{}", v.width, v.height)),
            None => ("-".into(), "-".into()),
        };
        let audio = match &e.audio {
            Some(a) => format!("{} {}ch/{}", a.codec, a.channels, a.sample_rate / 1000),
            None => "-".into(),
        };
        s.push_str(&row(
            &[
                fit_path(&e.path.display().to_string(), path_w),
                e.container.unwrap_or("-").into(),
                vcodec,
                size,
                audio,
                fmt_duration(e.duration_ms),
                fmt_bytes(e.bytes),
                e.verdict.label(),
            ],
            &widths,
            &aligns,
        ));
        s.push('\n');
        // 前提与原因单独一行，缩进在下面 —— 挤进表格里会把对齐全毁掉。
        if let Some(note) = e.note.as_deref().or_else(|| e.verdict.reason()) {
            s.push_str(&" ".repeat(path_w));
            s.push_str("  └ ");
            s.push_str(note);
            s.push('\n');
        }
    }
    s
}

/// 把一行单元格按列宽拼起来（列间两个空格）。
fn row(cells: &[String], widths: &[usize], aligns: &[bool]) -> String {
    let mut s = String::new();
    for (i, (c, w)) in cells.iter().zip(widths).enumerate() {
        if i > 0 {
            s.push_str("  ");
        }
        let pad = w.saturating_sub(display_width(c));
        if aligns.get(i).copied().unwrap_or(false) {
            s.push_str(&" ".repeat(pad));
            s.push_str(c);
        } else {
            s.push_str(c);
            s.push_str(&" ".repeat(pad));
        }
    }
    s
}

/// 路径超宽就**保留尾部**（文件名比前面的目录名有用得多）。
fn fit_path(path: &str, width: usize) -> String {
    if display_width(path) <= width {
        return path.to_string();
    }
    let mut out: Vec<char> = Vec::new();
    let mut used = 1; // 省略号自己占一格
    for c in path.chars().rev() {
        let w = char_width(c);
        if used + w > width {
            break;
        }
        used += w;
        out.push(c);
    }
    out.reverse();
    format!("…{}", out.into_iter().collect::<String>())
}

/// 终端显示宽度。
///
/// 为什么不用 `unicode-width`：全仓**零第三方依赖**（AGENTS.md 的硬约束）。
/// 这里只需要「中文路径别把表格撑歪」，所以只认常用的宽字符区段 ——
/// 认不全的后果是**某个生僻字的行歪一格**，不是错数据。
fn display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// 一个字符占几格。宽字符（CJK / 全角 / 谚文 / 假名 / emoji）算两格。
fn char_width(c: char) -> usize {
    let u = c as u32;
    let wide = (0x1100..=0x115F).contains(&u)          // 谚文字母
        || (0x2E80..=0xA4CF).contains(&u)              // CJK 部首 + 统一表意 + 注音
        || (0xAC00..=0xD7A3).contains(&u)              // 谚文音节
        || (0xF900..=0xFAFF).contains(&u)              // CJK 兼容表意
        || (0xFE30..=0xFE6F).contains(&u)              // CJK 兼容形式
        || (0xFF00..=0xFF60).contains(&u)              // 全角标点与字母
        || (0xFFE0..=0xFFE6).contains(&u)              // 全角符号
        || (0x1F300..=0x1F64F).contains(&u)            // emoji
        || (0x20000..=0x3FFFD).contains(&u);           // CJK 扩展 B 及以后
    if wide {
        2
    } else {
        1
    }
}

/// `mm:ss` / `h:mm:ss`；没有就 `-`。
fn fmt_duration(ms: Option<u64>) -> String {
    let Some(ms) = ms else {
        return "-".into();
    };
    let total = ms / 1000;
    let (h, m, s) = (total / 3600, total % 3600 / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// 人类可读的字节数（`KiB` / `MiB` / `GiB`）。
fn fmt_bytes(n: u64) -> String {
    const K: f64 = 1024.0;
    let f = n as f64;
    if f >= K * K * K {
        format!("{:.1} GiB", f / K / K / K)
    } else if f >= K * K {
        format!("{:.1} MiB", f / K / K)
    } else if f >= K {
        format!("{:.1} KiB", f / K)
    } else {
        format!("{n} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_is_formatted_without_lying_about_zero() {
        assert_eq!(fmt_duration(None), "-");
        assert_eq!(fmt_duration(Some(0)), "0:00");
        assert_eq!(fmt_duration(Some(9_500)), "0:09");
        assert_eq!(fmt_duration(Some(3_723_000)), "1:02:03");
    }

    #[test]
    fn bytes_switch_units_at_1024() {
        assert_eq!(fmt_bytes(1023), "1023 B");
        assert_eq!(fmt_bytes(1024), "1.0 KiB");
        assert_eq!(fmt_bytes(1536 * 1024), "1.5 MiB");
    }

    /// 音频不认识时两条路都要靠 `--no-audio` —— 这条**必须**带上提示，
    /// 否则用户照着 `TARGET` 去转，会撞上一句「音频是 0x0055」。
    #[test]
    fn unknown_audio_still_offers_targets_but_warns() {
        let af = crate::AudioFormat {
            format_tag: 0x0055,
            channels: 2,
            ..Default::default()
        };
        let (targets, note) = output_targets(Some(&af));
        assert_eq!(targets, vec!["wmv", "mp4"]);
        assert!(note.unwrap().contains("--no-audio"));
    }

    #[test]
    fn aac_only_goes_to_mp4() {
        let af = crate::AudioFormat {
            format_tag: 0x00FF,
            channels: 2,
            ..Default::default()
        };
        let (targets, _) = output_targets(Some(&af));
        assert_eq!(targets, vec!["mp4"]);
    }
}
