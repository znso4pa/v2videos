//! 变异模糊测试的**共用件**：伪随机数发生器、四个变异算子、轮次驱动。
//!
//! ⚠️ 四个 `tests/fuzz.rs`（msmpeg4 / wma / asf / avi）用 `#[path]` **引同一份文件**，
//! 不各写一份 —— 复刻第二份就是会漂的那一份（AGENTS.md 不变量 3）。
//! 变异算子写错的后果比一般 bug 严重：它会让模糊测试**假绿**，
//! 而假绿的模糊测试比没有更糟 —— 它给的是「我查过了」的错觉。
//!
//! ## 房规（沿用 `crates/msmpeg4/tests/pframe.rs` 的两条）
//!
//! 1. **先打日志再调用**：坐标（轮次 / 算子 / 位置 / 参数 / 长度）在**调用之前**
//!    写到 stderr。被测代码 panic 时，最后一行打印就是复现坐标 —— 不用事后二分。
//! 2. **覆盖率要钉成断言**：一个每轮都在第 0 字节就 `Err` 的模糊测试毫无价值。
//!    所以 `drive()` 会统计「见过几种 Err」「有几轮走通了」，由调用方断言掉。
//!
//! ## 为什么不用 `rand`
//!
//! 零第三方依赖（不变量 2），而且这里要的只是「同一个种子 → 同一串变异体」。
//! splitmix64 是八行、分布够好、且**不会随版本变** —— 换个大版本的 `rand`
//! 可能就换了算法，那会让历史失败案例复现不出来。

#![allow(dead_code)] // 共用工具箱：每个测试目标只用到其中一部分

use std::collections::BTreeMap;
use std::fmt::Display;

/// splitmix64 —— 确定性伪随机数发生器（同一个种子永远给出同一串数）。
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `0..n`（`n == 0` 时返回 0）。
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    pub fn byte(&mut self) -> u8 {
        (self.next_u64() >> 24) as u8
    }
}

/// 变异算子 —— 就这四种。
///
/// 它们**各有各的针对性**，不是随便凑的：
///
/// | 算子 | 主要撞什么 |
/// |---|---|
/// | `Truncate` | 循环里「先读后判」的写法 —— 位流提前结束时的收尾路径 |
/// | `FlipBit` | 单个标志位翻转导致的分支错位（`ac_pred`、`mb_type`、`cbp`） |
/// | `Substitute` | **长度 / 偏移算错** —— 置 0 / 0xFF / 0x80 最容易撞出来 |
/// | `Splice` | 把别处的字节搬过来：字段之间**自相矛盾**（长度说是 N、数据只有 M） |
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Op {
    /// 截断到 `at` 字节。
    Truncate,
    /// 把第 `at` 个字节的第 `arg % 8` 位取反。
    FlipBit,
    /// 把第 `at` 个字节换成 `SUBSTITUTES[arg % 3]`。
    Substitute,
    /// 把从 `at` 起的 `arg` 个字节，换成从 `src` 起的同样多字节。
    Splice,
}

/// `Substitute` 用的三个值。
///
/// 三个都试：`0x00` 撞「长度为 0 / 计数为零」，`0xFF` 撞「长度超大 / 下标越界」，
/// `0x80` 撞「有符号数变成最小负数」（`sign_extend` 那一类）。
pub const SUBSTITUTES: [u8; 3] = [0x00, 0xFF, 0x80];

impl Op {
    pub fn name(self) -> &'static str {
        match self {
            Op::Truncate => "截断",
            Op::FlipBit => "翻位",
            Op::Substitute => "替换",
            Op::Splice => "拼接",
        }
    }
}

/// 一次变异的**全部坐标** —— 也是复现所需的全部信息。
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Mutation {
    pub op: Op,
    /// 动的位置（字节下标）。`Truncate` 是「从哪切」。
    pub at: usize,
    /// `FlipBit`：位号；`Substitute`：取哪个替换值；`Splice`：抄多长。
    pub arg: usize,
    /// 只有 `Splice` 用：从哪个字节开始抄。
    pub src: usize,
}

impl Mutation {
    /// 产出一个变异体。**纯函数**（不看 rng）—— 同一个 `Mutation` 永远给出同一份字节，
    /// 所以日志里那四个坐标就是完整的复现信息。
    pub fn apply(&self, base: &[u8]) -> Vec<u8> {
        let mut v = base.to_vec();
        match self.op {
            Op::Truncate => v.truncate(self.at),
            Op::FlipBit => {
                if self.at < v.len() {
                    v[self.at] ^= 1 << (self.arg % 8);
                }
            }
            Op::Substitute => {
                if self.at < v.len() {
                    v[self.at] = SUBSTITUTES[self.arg % SUBSTITUTES.len()];
                }
            }
            Op::Splice => {
                let len = self.arg.min(v.len());
                if len > 0 && self.at < v.len() {
                    let len = len.min(v.len() - self.at);
                    // 源区间允许与目标重叠 —— `copy_within` 自己处理得对。
                    let src = self.src % v.len();
                    let src = src.min(v.len() - len);
                    v.copy_within(src..src + len, self.at);
                }
            }
        }
        v
    }

    /// 日志用的一行（**调用之前**打出去）。
    pub fn coord(&self) -> String {
        match self.op {
            Op::Truncate => format!("{} 到 {} 字节", self.op.name(), self.at),
            Op::FlipBit => format!("{} 第 {} 字节的第 {} 位", self.op.name(), self.at, self.arg % 8),
            Op::Substitute => format!(
                "{} 第 {} 字节 → {:#04x}",
                self.op.name(),
                self.at,
                SUBSTITUTES[self.arg % SUBSTITUTES.len()]
            ),
            Op::Splice => format!(
                "{} 从 {} 抄 {} 字节到 {}",
                self.op.name(),
                self.src,
                self.arg,
                self.at
            ),
        }
    }
}

/// **系统性**变异：截断的每一个偏移、每个字节的每一位翻转、每个字节的三种替换值。
///
/// `stride > 1` 时按步长抽样（大输入用）—— 但**截断的偏移永远逐个试**：
/// 「位流在哪个位置结束」是最容易出错、也最便宜的一维，没有理由抽样。
pub fn systematic(len: usize, stride: usize) -> Vec<Mutation> {
    let stride = stride.max(1);
    let mut out = Vec::new();
    for at in 0..len {
        out.push(Mutation {
            op: Op::Truncate,
            at,
            arg: 0,
            src: 0,
        });
    }
    let mut at = 0;
    while at < len {
        for bit in 0..8 {
            out.push(Mutation {
                op: Op::FlipBit,
                at,
                arg: bit,
                src: 0,
            });
        }
        for k in 0..SUBSTITUTES.len() {
            out.push(Mutation {
                op: Op::Substitute,
                at,
                arg: k,
                src: 0,
            });
        }
        at += stride;
    }
    out
}

/// **随机**变异：主要是区间拼接（它没有「逐个试」的合理维度），
/// 另外撒一些跨段的截断 / 翻位 / 替换。
pub fn random(rng: &mut Rng, len: usize, rounds: usize) -> Vec<Mutation> {
    let mut out = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let op = match rng.below(4) {
            0 => Op::Truncate,
            1 => Op::FlipBit,
            2 => Op::Substitute,
            _ => Op::Splice,
        };
        let at = rng.below(len.max(1));
        let arg = match op {
            Op::Truncate => 0,
            Op::FlipBit => rng.below(8),
            Op::Substitute => rng.below(SUBSTITUTES.len()),
            // 长度从 1 到剩余长度的 1/3 —— 太长的拼接往往只是把整条都盖掉，
            // 退化成「换了一份数据」，撞不出字段间的自相矛盾。
            Op::Splice => 1 + rng.below((len / 3).max(1)),
        };
        out.push(Mutation {
            op,
            at,
            arg,
            src: rng.below(len.max(1)),
        });
    }
    out
}

/// 一批变异体跑完之后的**覆盖报告**。
///
/// 「没崩」本身不构成证据 —— 得有「我确实撞到了 N 条不同的错误通路」才算查过。
pub struct Report<K> {
    pub tag: &'static str,
    /// 一共跑了多少轮。
    pub total: usize,
    /// 有多少轮**没报错**（变异没破坏结构，正常解出来了）。
    pub ok: usize,
    /// 每种「错误种类」各出现多少次。空 map = 一轮都没报错（可疑）。
    pub errs: BTreeMap<K, usize>,
    /// 见过的**最短**变异体长度 —— 用来确认截断真的截到了很短。
    pub shortest: usize,
}

impl<K: Ord + Display> Report<K> {
    /// 打一份覆盖报告（测试里 `--nocapture` 才看得见，失败时自动带出来）。
    pub fn print(&self) {
        eprintln!(
            "[{}] 覆盖：共 {} 轮，其中没报错 {} 轮，最短变异体 {} 字节",
            self.tag, self.total, self.ok, self.shortest
        );
        if self.errs.is_empty() {
            eprintln!("[{}] ⚠️ 一轮都没报错 —— 变异没有生效，或者被测代码全都吃下去了", self.tag);
        }
        for (k, n) in &self.errs {
            eprintln!("[{0}]    {1}: {2} 轮", self.tag, k, n);
        }
    }

    /// 见过的错误**种类数**。
    pub fn kinds(&self) -> usize {
        self.errs.len()
    }
}

/// 跑一批变异体。
///
/// - `classify` 把具体的 `Error` 归成一种「错误种类」—— 种类要**粗**
///   （`Eof` / `Bad` / `Unsupported` 这一级），不然消息里的数字会把种类数撑虚。
/// - `run` 拿变异体跑一遍被测代码。**允许返回 `Err`，不允许 panic**
///   —— panic 会让测试直接红，这正是我们要的。
///
/// ⚠️ **先打日志再调用**（见模块头那条房规）。日志走 stderr（不带缓冲），
/// 所以 panic 时最后一行就是复现坐标。
pub fn drive<E, K>(
    tag: &'static str,
    base: &[u8],
    muts: &[Mutation],
    classify: impl Fn(&E) -> K,
    mut run: impl FnMut(&[u8]) -> Result<(), E>,
) -> Report<K>
where
    E: Display,
    K: Ord + Display,
{
    let mut errs: BTreeMap<K, usize> = BTreeMap::new();
    let mut ok = 0usize;
    let mut shortest = usize::MAX;

    for (i, m) in muts.iter().enumerate() {
        let data = m.apply(base);
        shortest = shortest.min(data.len());
        eprintln!(
            "[{tag}] 轮 {i}/{} 种子底本 {} 字节 → {}（{} 字节）",
            muts.len(),
            base.len(),
            m.coord(),
            data.len()
        );
        match run(&data) {
            Ok(()) => ok += 1,
            Err(e) => {
                eprintln!("[{tag}]    → Err: {e}");
                *errs.entry(classify(&e)).or_default() += 1;
            }
        }
    }

    Report {
        tag,
        total: muts.len(),
        ok,
        errs,
        shortest: if shortest == usize::MAX { 0 } else { shortest },
    }
}

/// 从 `V2V_FUZZ_ROUNDS` 读轮数（没设就用 `default`）。
///
/// 口径：CI / 日常跑用**小**轮数（几百轮，保证测试秒级完成）；
/// 要深挖时 `V2V_FUZZ_ROUNDS=200000 cargo test --release -- --ignored --nocapture`。
pub fn rounds(default: usize) -> usize {
    match std::env::var("V2V_FUZZ_ROUNDS") {
        Ok(s) => s.trim().parse().unwrap_or_else(|_| {
            panic!("V2V_FUZZ_ROUNDS 看不懂：`{s}`（要正整数）");
        }),
        Err(_) => default,
    }
}

/// 本机跑重局时用的默认轮数（`#[ignore]` 那一层）。
pub const HEAVY_ROUNDS: usize = 20_000;

/// CI / 日常那一层的默认轮数 —— 要能在几秒内跑完。
pub const LIGHT_ROUNDS: usize = 400;
