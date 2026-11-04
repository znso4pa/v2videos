//! 性能测量的**格式与寿命管理** —— 不测量任何东西。
//!
//! 这个 crate 只回答三个问题：
//!
//! 1. 一次测量怎么写下来，才能被别的程序读（[`Report::to_tsv`]）；
//! 2. 两份测量凭什么可以放在一起比（[`Fingerprint`] / [`Report::meta`]）；
//! 3. 比出来「变慢了」的时候，哪一条该拦住人、哪一条只该提醒（[`compare`]）。
//!
//! ## 为什么要把这件事单独拎出来
//!
//! 仓库里有六处各自在计时：`vc1/tests/decode_bench.rs`、`vc1/tests/decode_parallel_bench.rs`、
//! `msmpeg4/tests/decode_bench.rs`、`mp4/tests/mux_bench.rs`、`player/examples/bench.rs`、
//! `mpeg12-simd/examples/simd_bench.rs`，再加上 `v2v bench` 子命令。
//! 它们的**测量方法各不相同**（有的对 ffmpeg、有的对同一份数据的两种实现），这是对的 ——
//! 每种测量的正确口径本来就不一样。
//!
//! 但**结果的寿命管理必须只有一套**。否则「三个月后 `22.33 ms/帧` 还算不算数」
//! 就会变成每个 crate 各说各话，而那张表恰恰是项目最硬的资产之一。
//!
//! ## 为什么零依赖
//!
//! - `serde`：一行就六列、格式自己定，手写读写比引一整套框架清楚，也便于在
//!   `#[ignore]` 测试里直接 `use`；
//! - `num_cpus`：`std::thread::available_parallelism` 从 1.59 起就有；
//! - `sha2`：700 MB 的样本算一遍要好几秒，而「同名 + 同字节数」已经足够判断
//!   「这是不是同一次测量的输入」。要更强的保证就用 [`Report::set`] 手工钉一个
//!   `sample.label`，别让哈希成为跑基准的前置成本。
//!
//! ## TSV 长什么样
//!
//! ```text
//! # v2v-bench 1
//! # sample.0	sample.wmv:1048576
//! # cpu	Example CPU
//! # arch	aarch64
//! # cores	8
//! # rustc	1.98.0
//! # profile	release
//! case	n	total_ms	per_unit	unit	ref_per_unit
//! decode/vc1/1280x720	4138	11400.000	2.754953	frame	2.754953
//! ```
//!
//! `#` 开头是元数据（`# <key>\t<value>`），第一行固定是格式版本。
//!
//! ⚠️ **`per_unit` 是派生列**（= `total_ms / n`），写出来只为方便人读与 grep；
//! 读回来时**以 `total_ms / n` 重算为准**。这样「手改一列让数字好看」不会生效 ——
//! 也避免了同一份数据里出现两个互相矛盾的说法。
#![deny(unsafe_code)]

use std::fmt::Write as _;
use std::process::Command;

/// TSV 的格式版本。列的含义变了才动它。
///
/// 读到一个不认识的版本就**报错而不是猜** —— 猜错的代价是拿两份不可比的数据
/// 下结论，那比读不出来坏得多。
pub const FORMAT_VERSION: &str = "1";

/// 一行测量结果。
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// 用例名，约定 `<op>/<codec>/<尺寸或条件>`，例如 `decode/vc1/1280x720`。
    ///
    /// 它是两份报告对齐的**唯一键**，所以改名等于换用例：`--check` 会报
    /// 「基线里没有 / 现在没有」，而不会拿错行去比。
    pub case: String,
    /// 单位数（帧数 / 样本数 / 文件数）。
    pub n: u64,
    /// 总耗时（毫秒）。
    pub total_ms: f64,
    /// `n` 的单位，`frame` / `sample` / `file` / `call`。
    pub unit: String,
    /// **参考实现**的每单位耗时（毫秒），没有参考实现就是 `None`。
    ///
    /// 存成「每单位毫秒」而不是「总毫秒」是有意的：参考实现（`ffmpeg`）与我们的
    /// 进程启动开销差得远，两边都先减掉「处理 1 个单位」的那一趟再折算，
    /// 才是纯吞吐的对比 —— 具体怎么减是测量方的责任，这里只负责把它记下来。
    pub ref_per_unit: Option<f64>,
}

impl Row {
    pub fn new(case: impl Into<String>, n: u64, total_ms: f64, unit: impl Into<String>) -> Self {
        Self {
            case: case.into(),
            n,
            total_ms,
            unit: unit.into(),
            ref_per_unit: None,
        }
    }

    /// 附上参考实现的每单位耗时。
    pub fn with_ref(mut self, ref_per_unit: f64) -> Self {
        self.ref_per_unit = Some(ref_per_unit);
        self
    }

    /// 每单位耗时（毫秒）。`n == 0` 时给 0 而不是 `NaN` —— 后者会毒化整张表。
    pub fn per_unit(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.total_ms / self.n as f64
        }
    }

    /// 我们 / 参考实现的比值。`< 1` = 我们更快。
    ///
    /// 这个数是**唯一能跨机器比较的量**：绝对毫秒在 CI runner 上能差 2×，
    /// 但同一台机器上「我们比参考实现慢几倍」是稳定的。
    pub fn ratio(&self) -> Option<f64> {
        match self.ref_per_unit {
            Some(r) if r > 0.0 => Some(self.per_unit() / r),
            _ => None,
        }
    }
}

/// 一次测量（一个或多个样本、若干用例）。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// 有序元数据。顺序稳定是为了 `to_tsv` 的 diff 稳定。
    pub meta: Vec<(String, String)>,
    pub rows: Vec<Row>,
}

impl Report {
    pub fn new() -> Self {
        Self::default()
    }

    /// 写一个元数据项（同 key 覆盖）。
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        if let Some(slot) = self.meta.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.meta.push((key, value));
        }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.meta
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// 记一个样本：`sample.<序号>` = `<文件名>:<字节数>`。
    ///
    /// 序号从 0 起、按命令行给的顺序，所以两份报告只要样本顺序一致就能对上。
    pub fn add_sample(&mut self, index: usize, name: &str, bytes: u64) {
        self.set(format!("sample.{index}"), format!("{name}:{bytes}"));
    }

    pub fn push(&mut self, row: Row) {
        self.rows.push(row);
    }

    /// 把元数据写成 TSV（不含行）。`v2v bench` 与各 bench 的 `--tsv` 共用。
    ///
    /// ⚠️ 数字用 `{}`（f64 的最短**无损**表示）而不是 `{:.6}`：实测踩过这个坑 ——
    /// 音频的 `ref_per_unit` 是 `8.9e-6` 量级，`{:.6}` 把它写成 `0.000009`，
    /// 精度掉了一半，而那个数正是比值的分母。TSV 的第一职责是被程序读回去，
    /// 可读性由各 bench 自己打的人类表补偿。
    pub fn to_tsv(&self) -> String {
        let mut s = String::with_capacity(256 + self.rows.len() * 64);
        let _ = writeln!(s, "# v2v-bench {FORMAT_VERSION}");
        for (k, v) in &self.meta {
            let _ = writeln!(s, "# {k}\t{v}");
        }
        let _ = writeln!(s, "case\tn\ttotal_ms\tper_unit\tunit\tref_per_unit");
        for r in &self.rows {
            let refs = match r.ref_per_unit {
                Some(v) => v.to_string(),
                None => String::new(),
            };
            let _ = writeln!(
                s,
                "{}\t{}\t{}\t{}\t{}\t{}",
                r.case,
                r.n,
                r.total_ms,
                r.per_unit(),
                r.unit,
                refs
            );
        }
        s
    }

    /// 从 TSV 读回来。见模块文档里那张表。
    ///
    /// 三条刻意的严格：版本不认识就报错；列数不对就报错；`per_unit` 列**不采信**
    /// （重算）。前两条防止「格式漂了却静默读成空表」，第三条防止一份数据里
    /// 出现两个说法。
    pub fn parse_tsv(text: &str) -> Result<Self, String> {
        let mut report = Self::new();
        let mut saw_header = false;
        let mut saw_version = false;
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim_end_matches('\r');
            if line.trim().is_empty() {
                continue;
            }
            let at = || format!("第 {} 行", lineno + 1);
            if let Some(rest) = line.strip_prefix("# v2v-bench ") {
                let ver = rest.trim();
                if ver != FORMAT_VERSION {
                    return Err(format!(
                        "{}：格式版本 `{ver}` 不认识（本程序只认 `{FORMAT_VERSION}`）",
                        at()
                    ));
                }
                saw_version = true;
                continue;
            }
            if let Some(rest) = line.strip_prefix("# ") {
                let mut it = rest.splitn(2, '\t');
                let key = it.next().unwrap_or_default();
                let value = it.next().unwrap_or_default();
                if key.is_empty() {
                    return Err(format!("{}：元数据行缺 key", at()));
                }
                report.set(key, value);
                continue;
            }
            if !saw_header {
                if line != "case\tn\ttotal_ms\tper_unit\tunit\tref_per_unit" {
                    return Err(format!(
                        "{}：表头不是预期的六列，读到 `{line}`",
                        at()
                    ));
                }
                saw_header = true;
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() != 6 {
                return Err(format!(
                    "{}：数据行要 6 列，读到 {} 列：`{line}`",
                    at(),
                    cols.len()
                ));
            }
            let n: u64 = cols[1]
                .parse()
                .map_err(|_| format!("{}：n 不是整数 `{}`", at(), cols[1]))?;
            let total_ms: f64 = cols[2]
                .parse()
                .map_err(|_| format!("{}：total_ms 不是数 `{}`", at(), cols[2]))?;
            // cols[3] 是 per_unit —— 故意不读，见方法文档。
            let ref_per_unit = if cols[5].trim().is_empty() {
                None
            } else {
                Some(
                    cols[5]
                        .parse()
                        .map_err(|_| format!("{}：ref_per_unit 不是数 `{}`", at(), cols[5]))?,
                )
            };
            report.push(Row {
                case: cols[0].to_string(),
                n,
                total_ms,
                unit: cols[4].to_string(),
                ref_per_unit,
            });
        }
        if !saw_version {
            return Err("缺 `# v2v-bench <版本>` 首行 —— 这不是本格式的文件".into());
        }
        if !saw_header && !report.rows.is_empty() {
            return Err("有数据行却没有表头".into());
        }
        Ok(report)
    }

    /// 按 case 名取一行（`--check` 对齐用）。
    pub fn row(&self, case: &str) -> Option<&Row> {
        self.rows.iter().find(|r| r.case == case)
    }
}

/// 这台机器的指纹 —— 「不同机器的数不能比」的落地方式。
///
/// 基线按指纹分文件（`bench/baseline/<slug>.tsv`），而不是一个全局文件：
/// 同一个提交在两台机器上量出的绝对毫秒可以差 2×，混在一个文件里就没法判断
/// 「是我改慢了」还是「换机器了」。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fingerprint {
    /// CPU 型号（`sysctl machdep.cpu.brand_string` / `/proc/cpuinfo`）。
    pub cpu: String,
    /// `std::env::consts::ARCH`。
    pub arch: String,
    /// 逻辑核数。
    pub cores: usize,
    /// `rustc -vV` 的 `release:` 行。
    pub rustc: String,
    /// `debug` / `release`。**debug 的数字没有意义**，`compare` 会拒绝跨这个比。
    pub profile: String,
}

impl Fingerprint {
    pub fn detect() -> Self {
        Self {
            cpu: cpu_brand(),
            arch: std::env::consts::ARCH.to_string(),
            cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            rustc: rustc_version(),
            profile: if cfg!(debug_assertions) {
                "debug".to_string()
            } else {
                "release".to_string()
            },
        }
    }

    /// 文件名安全的短标识，例如 `apple-m1-pro-aarch64-8c-1.98.0`。
    pub fn slug(&self) -> String {
        format!(
            "{}-{}-{}c-{}",
            slugify(&self.cpu),
            slugify(&self.arch),
            self.cores,
            slugify(&self.rustc)
        )
    }

    /// 写进 [`Report::meta`] 的那几项。
    pub fn apply(&self, report: &mut Report) {
        report.set("cpu", self.cpu.clone());
        report.set("arch", self.arch.clone());
        report.set("cores", self.cores.to_string());
        report.set("rustc", self.rustc.clone());
        report.set("profile", self.profile.clone());
    }
}

/// 小写、只留 `[a-z0-9.]`、其余折成单个 `-`。
///
/// `aarch64` 里的数字要留（`x86_64` 也一样），所以不能只留字母。
fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for ch in s.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() || c == '.' {
            out.push(c);
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "unknown".into()
    } else {
        out
    }
}

/// CPU 型号。拿不到就给 `unknown` —— 指纹不完整不该让基准跑不起来。
fn cpu_brand() -> String {
    if cfg!(target_os = "macos") {
        if let Ok(out) = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !s.is_empty() {
                    return s;
                }
            }
        }
    } else if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                if let Some(v) = rest.split(':').nth(1) {
                    let v = v.trim();
                    if !v.is_empty() {
                        return v.to_string();
                    }
                }
            }
        }
    }
    "unknown".into()
}

/// `rustc -vV` 的 `release:` 行。跑不动就给 `unknown`。
fn rustc_version() -> String {
    let Ok(out) = Command::new("rustc").arg("-vV").output() else {
        return "unknown".into();
    };
    if !out.status.success() {
        return "unknown".into();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("release: ").map(|v| v.trim().to_string()))
        .unwrap_or_else(|| "unknown".into())
}

/// 三个负载均值（1 / 5 / 15 分钟），拿不到就 `None`。
///
/// 记它不是为了参与比较（`compare` 不看这一项），而是为了让基线**自证来源**：
/// 一条 `# load 4.53 6.07 5.63` 就能让人看出「这份基线是在不空的机器上量的」。
/// 没有这个数，一年后没人说得清那次为什么偏高。
pub fn load_average() -> Option<String> {
    // Linux / macOS 都能从这一个文件拿到（macOS 上没有）。
    if let Ok(text) = std::fs::read_to_string("/proc/loadavg") {
        let head: Vec<&str> = text.split_whitespace().take(3).collect();
        if head.len() == 3 {
            return Some(head.join(" "));
        }
    }
    if cfg!(target_os = "macos") {
        let out = Command::new("sysctl").args(["-n", "vm.loadavg"]).output().ok()?;
        if !out.status.success() {
            return None;
        }
        // 形如 `{ 1.23 4.56 7.89 }`。
        let text = String::from_utf8_lossy(&out.stdout);
        let nums: Vec<&str> = text
            .trim()
            .trim_matches(|c| c == '{' || c == '}')
            .split_whitespace()
            .collect();
        if nums.len() == 3 {
            return Some(nums.join(" "));
        }
    }
    None
}

/// 一条「变慢了 / 对不上」的发现。
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub case: String,
    pub severity: Severity,
    pub kind: Kind,
    /// 基线里的数（比值或每单位毫秒；`MissingInBaseline` 时无意义）。
    pub base: f64,
    /// 现在的数。
    pub now: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// **该拦住人**：比值恶化超过阈值 —— 对机器不敏感，没有借口。
    Fail,
    /// **只提醒**：绝对量恶化，或某个用例在一边缺席。
    Warn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// 我们 / 参考实现的比值变差。
    Ratio,
    /// 每单位毫秒变差（没有参考实现可用时的兜底）。
    Absolute,
    /// 基线里有、这次没跑（用例被删 / 被跳过）。
    MissingInCurrent,
    /// 这次跑了、基线里没有（新用例还没有基线）。
    MissingInBaseline,
}

/// 门限，都是「恶化百分比」。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    /// 比值恶化的门限（默认 15%）。超了就是 [`Severity::Fail`]。
    ///
    /// 为什么可以比绝对量严：比值是两个数相除，机器整体快慢、后台负载、
    /// 频率漂移大部分会同时作用在分子分母上而抵消。留 15% 是因为
    /// 我们的测量本身取「N 次里最快」，重复测量的散布通常在 5% 以内。
    pub ratio_pct: f64,
    /// 绝对量恶化的门限（默认 30%）。超了只 [`Severity::Warn`]。
    ///
    /// 绝对毫秒在 CI runner 上没有意义（同一份代码换台机器差 2×），
    /// 所以它只能当「可能出事了，去看一眼」的提示，不能当门。
    pub abs_pct: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            ratio_pct: 15.0,
            abs_pct: 30.0,
        }
    }
}

/// 两份报告**凭什么不能比**。返回它而不是硬着头皮比。
#[derive(Clone, Debug, PartialEq)]
pub struct Mismatch {
    pub field: String,
    pub base: String,
    pub now: String,
}

/// 比较两份报告。
///
/// 顺序很重要：先确认能比（样本 / 架构 / profile / 核数），再逐用例比。
/// 能比的时候，**有参考实现的用例只报比值、不报绝对量** —— 同一个退化报两次
/// 只会让人以为出了两个问题。
pub fn compare(
    base: &Report,
    now: &Report,
    th: &Thresholds,
) -> Result<Vec<Finding>, Mismatch> {
    // 1) 先确认「同一把尺子」。
    for field in ["profile", "arch", "cores"] {
        let (Some(b), Some(n)) = (base.get(field), now.get(field)) else {
            continue; // 缺这项就跳过 —— 老文件可能没写，不该因此拒绝比较
        };
        if b != n {
            return Err(Mismatch {
                field: field.to_string(),
                base: b.to_string(),
                now: n.to_string(),
            });
        }
    }
    // 样本必须一一对应：换素材之后绝对量完全不可比。
    let base_samples: Vec<&str> = base
        .meta
        .iter()
        .filter(|(k, _)| k.starts_with("sample."))
        .map(|(_, v)| v.as_str())
        .collect();
    let now_samples: Vec<&str> = now
        .meta
        .iter()
        .filter(|(k, _)| k.starts_with("sample."))
        .map(|(_, v)| v.as_str())
        .collect();
    if !base_samples.is_empty() && base_samples != now_samples {
        return Err(Mismatch {
            field: "sample".into(),
            base: base_samples.join(", "),
            now: now_samples.join(", "),
        });
    }

    // 2) 逐用例。
    let mut out = Vec::new();
    for b in &base.rows {
        let Some(n) = now.row(&b.case) else {
            out.push(Finding {
                case: b.case.clone(),
                severity: Severity::Warn,
                kind: Kind::MissingInCurrent,
                base: b.per_unit(),
                now: f64::NAN,
            });
            continue;
        };
        match (b.ratio(), n.ratio()) {
            (Some(br), Some(nr)) if br > 0.0 => {
                // 比值是**唯一该当门**的量。
                if worse_pct(br, nr) > th.ratio_pct {
                    out.push(Finding {
                        case: b.case.clone(),
                        severity: Severity::Fail,
                        kind: Kind::Ratio,
                        base: br,
                        now: nr,
                    });
                }
            }
            _ => {
                // 没有参考实现 → 只剩绝对量，只能提醒。
                let (bp, np) = (b.per_unit(), n.per_unit());
                if worse_pct(bp, np) > th.abs_pct {
                    out.push(Finding {
                        case: b.case.clone(),
                        severity: Severity::Warn,
                        kind: Kind::Absolute,
                        base: bp,
                        now: np,
                    });
                }
            }
        }
    }
    for n in &now.rows {
        if base.row(&n.case).is_none() {
            out.push(Finding {
                case: n.case.clone(),
                severity: Severity::Warn,
                kind: Kind::MissingInBaseline,
                base: f64::NAN,
                now: n.per_unit(),
            });
        }
    }
    Ok(out)
}

/// `now` 比 `base` 恶化多少百分比（变小是好事，所以返回负值）。
fn worse_pct(base: f64, now: f64) -> f64 {
    if base <= 0.0 {
        return 0.0;
    }
    (now / base - 1.0) * 100.0
}

/// 让各 crate 的 bench 用**环境变量**决定要不要写 TSV 的开关。
///
/// 为什么是环境变量而不是命令行参数：`cargo test -- --ignored` 后面能传的东西
/// 有限，而这些 bench 大多是 `#[ignore]` 测试（不是 `examples/`）。
/// 统一的开关让「人读的表」与「机器读的 TSV」可以并存 —— 默认只打前者。
pub const TSV_ENV: &str = "V2V_BENCH_TSV";

/// 起一份带机器指纹的报告。各 bench 的样板代码（`tool` 之外的 meta 由调用方补）。
///
/// 顺带记一条 `load`（三个负载均值）—— 只作来源说明，`compare` 不看它。
pub fn start(tool: &str) -> Report {
    let mut report = Report::new();
    Fingerprint::detect().apply(&mut report);
    report.set("tool", tool);
    if let Some(load) = load_average() {
        report.set("load", load);
    }
    report
}

/// 环境变量给了 `V2V_BENCH_TSV` 就把这份报告写过去，返回写到的路径。
///
/// 没给就什么也不做 —— 这正是「不改变既有用法」的意思：不加这个变量时，
/// 各 bench 的行为与以前逐字节相同。
pub fn write_if_requested(report: &Report) -> std::io::Result<Option<std::path::PathBuf>> {
    let Some(path) = std::env::var_os(TSV_ENV) else {
        return Ok(None);
    };
    let path = std::path::PathBuf::from(path);
    std::fs::write(&path, report.to_tsv())?;
    Ok(Some(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(case: &str, n: u64, total_ms: f64, r: Option<f64>) -> Row {
        let mut row = Row::new(case, n, total_ms, "frame");
        row.ref_per_unit = r;
        row
    }

    fn sample_report(rows: Vec<Row>) -> Report {
        let mut r = Report::new();
        r.set("cpu", "Test CPU");
        r.set("arch", "aarch64");
        r.set("cores", "8");
        r.set("rustc", "1.98.0");
        r.set("profile", "release");
        r.add_sample(0, "a.wmv", 1000);
        for row in rows {
            r.push(row);
        }
        r
    }

    #[test]
    fn tsv_round_trip() {
        let mut r = sample_report(vec![
            row("decode/vc1/1280x720", 4138, 11400.0, Some(2.7549)),
            row("transcode/mp4", 500, 42000.0, None),
            // 音频那种量级：`{:.6}` 会把它写成 0.000009，精度掉一半。
            row("decode/audio/x/44100hz", 6_082_560, 480.858, Some(0.000008_912_3)),
        ]);
        r.set("note", "带制表符\t和换行不行，但普通文本可以");
        let text = r.to_tsv();
        let back = Report::parse_tsv(&text).expect("parse");
        assert_eq!(back.meta, r.meta);
        assert_eq!(back.rows.len(), 3);
        for (a, b) in back.rows.iter().zip(r.rows.iter()) {
            assert_eq!(a.case, b.case);
            assert_eq!(a.n, b.n);
            assert_eq!(a.unit, b.unit);
            // `{}` 是 f64 的最短无损表示，所以这里必须**严格相等** ——
            // 用 `abs() < eps` 会把精度回归放过去。
            assert_eq!(a.total_ms, b.total_ms, "{} 的 total_ms 丢了精度", a.case);
            assert_eq!(
                a.per_unit(),
                b.per_unit(),
                "{} 的 per_unit 丢了精度",
                a.case
            );
            assert_eq!(
                a.ref_per_unit, b.ref_per_unit,
                "{} 的 ref_per_unit 丢了精度",
                a.case
            );
        }
        assert_eq!(back.rows[1].ref_per_unit, None);
        // 幂等：再写一遍应当逐字节相同。
        assert_eq!(back.to_tsv(), text);
    }

    #[test]
    fn per_unit_column_is_recomputed_not_trusted() {
        // 手改 per_unit 列（第 4 列）不该影响读回来的数 —— 否则一份数据里
        // 会有两个互相矛盾的说法。
        let text = "# v2v-bench 1\n\
                    case\tn\ttotal_ms\tper_unit\tunit\tref_per_unit\n\
                    decode/x\t100\t1000.000\t0.001000\tframe\t\n";
        let r = Report::parse_tsv(text).expect("parse");
        assert!((r.rows[0].per_unit() - 10.0).abs() < 1e-9, "必须是 total_ms/n");
    }

    #[test]
    fn rejects_unknown_version_and_bad_columns() {
        assert!(Report::parse_tsv("# v2v-bench 99\n").is_err());
        assert!(Report::parse_tsv("case\tn\n").is_err());
        assert!(Report::parse_tsv("").is_err());
        let bad_cols = "# v2v-bench 1\n\
                        case\tn\ttotal_ms\tper_unit\tunit\tref_per_unit\n\
                        a\t1\t1.0\t1.0\tframe\n";
        assert!(Report::parse_tsv(bad_cols).is_err());
    }

    #[test]
    fn ratio_is_what_gets_gated() {
        let base = sample_report(vec![row("decode/x", 100, 1000.0, Some(10.0))]);
        // 绝对量涨 40%（超 abs 阈值），但参考实现也涨了同样的比例 → 比值不变。
        let now = sample_report(vec![row("decode/x", 100, 1400.0, Some(14.0))]);
        let f = compare(&base, &now, &Thresholds::default()).expect("comparable");
        assert!(
            f.is_empty(),
            "比值没变就不该报 —— 机器整体变慢不是退化：{f:?}"
        );
    }

    #[test]
    fn ratio_regression_fails() {
        let base = sample_report(vec![row("decode/x", 100, 1000.0, Some(10.0))]);
        let now = sample_report(vec![row("decode/x", 100, 1200.0, Some(10.0))]);
        let f = compare(&base, &now, &Thresholds::default()).expect("comparable");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::Fail);
        assert_eq!(f[0].kind, Kind::Ratio);
    }

    #[test]
    fn absolute_regression_only_warns() {
        let base = sample_report(vec![row("transcode/mp4", 100, 1000.0, None)]);
        let now = sample_report(vec![row("transcode/mp4", 100, 1400.0, None)]);
        let f = compare(&base, &now, &Thresholds::default()).expect("comparable");
        assert_eq!(f.len(), 1);
        assert_eq!(
            f[0].severity,
            Severity::Warn,
            "绝对量在 CI 上没有意义，只能提醒"
        );
        assert_eq!(f[0].kind, Kind::Absolute);
    }

    #[test]
    fn refuses_to_compare_across_machines_or_samples() {
        let base = sample_report(vec![row("decode/x", 100, 1000.0, None)]);
        let mut other = sample_report(vec![row("decode/x", 100, 1000.0, None)]);
        other.set("cores", "4");
        let e = compare(&base, &other, &Thresholds::default()).expect_err("核数不同");
        assert_eq!(e.field, "cores");

        let mut swapped = sample_report(vec![row("decode/x", 100, 1000.0, None)]);
        swapped.set("sample.0", "b.wmv:1000");
        let e = compare(&base, &swapped, &Thresholds::default()).expect_err("样本不同");
        assert_eq!(e.field, "sample");

        let mut debug = sample_report(vec![row("decode/x", 100, 1000.0, None)]);
        debug.set("profile", "debug");
        let e = compare(&base, &debug, &Thresholds::default()).expect_err("debug 对 release");
        assert_eq!(e.field, "profile");
    }

    #[test]
    fn missing_cases_are_warnings_both_ways() {
        let base = sample_report(vec![
            row("decode/x", 100, 1000.0, None),
            row("decode/gone", 100, 1000.0, None),
        ]);
        let now = sample_report(vec![
            row("decode/x", 100, 1000.0, None),
            row("decode/new", 100, 1000.0, None),
        ]);
        let f = compare(&base, &now, &Thresholds::default()).expect("comparable");
        assert!(f.iter().all(|x| x.severity == Severity::Warn));
        assert!(f.iter().any(|x| x.kind == Kind::MissingInCurrent));
        assert!(f.iter().any(|x| x.kind == Kind::MissingInBaseline));
    }

    #[test]
    fn slug_is_filename_safe_and_stable() {
        let fp = Fingerprint {
            cpu: "Apple M1 Pro".into(),
            arch: "aarch64".into(),
            cores: 8,
            rustc: "1.98.0".into(),
            profile: "release".into(),
        };
        assert_eq!(fp.slug(), "apple-m1-pro-aarch64-8c-1.98.0");
        // 两次调用必须一致，否则基线文件名会漂。
        assert_eq!(fp.slug(), fp.slug());

        let weird = Fingerprint {
            cpu: "  Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz ".into(),
            arch: "x86_64".into(),
            cores: 12,
            rustc: "1.98.0".into(),
            profile: "release".into(),
        };
        let s = weird.slug();
        assert!(
            s.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.'),
            "文件名里出现了不安全字符：{s}"
        );
        assert!(s.starts_with("intel-r-core-tm-i7-9750h-cpu-2.60ghz-x86-64-12c-"));
    }

    #[test]
    fn zero_n_does_not_poison_the_table() {
        let r = Row::new("empty", 0, 5.0, "frame");
        assert_eq!(r.per_unit(), 0.0);
        assert!(r.per_unit().is_finite());
        assert_eq!(r.ratio(), None);
    }
}
