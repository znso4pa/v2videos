//! RL（run-level）前缀码容器 —— **与具体编解码器无关**。
//!
//! 一张 RL 表 = 前缀码 + 每个符号的 `(run, level)` + `last` 分界，外加三张派生表
//! （`max_level` / `max_run` / `index_run`）和一张快表。解码按 RL 语法走，
//! 三种 escape 的形状见各函数上的说明。
//!
//! 为什么在 `v2v-kernel`：`v2v-msmpeg4` 与 `v2v-mp4v` 用的是**同一个**容器与同一套
//! 解码/编码逻辑，只有表本身不同（各自 crate 的 `tables.rs` 里）。复刻第二份容器
//! 才是真会跑偏的那份。
//!
//! ⚠️ 表本身**不在这里**：`RlTable::new` 收的是调用方给的静态表。MS-MPEG-4 那 6 张
//! 标准的派发表（`RlTable::for_standard` 那张）是它的私有数据，留在 `v2v-msmpeg4`
//! 的 `block.rs` 里，做成自由函数 —— 因为**不许给外部类型写 inherent impl**。

use crate::bitreader::BitReader;
use crate::error::Result;
use crate::vlc::Vlc;

/// 一张 RL 表：前缀码 + 每个符号的 (run, level) + `last` 分界，外加两张派生表。
pub struct RlTable {
    pub vlc: Vlc,
    run: &'static [i8],
    level: &'static [i8],
    /// 下标 >= `last` 的符号 `last = 1`。
    last: usize,
    /// `max_level[last][run]` —— 第一 escape 用。
    max_level: [[i8; 64]; 2],
    /// `max_run[last][level]` —— 第二 escape 用。
    max_run: [[i8; 64]; 2],
    /// `index_run[last][run]` = 该 `(last, run)` 的**第一个**符号下标（没有则 = `n`）。
    /// 编码时靠它反查：同一个 run 下 level 从 1 起连续排，所以
    /// `符号 = index_run[last][run] + level - 1`。
    index_run: [[u16; 64]; 2],
    /// **快表**：偷看 `FAST_BITS` 位直接得 `(码长, run, level)` —— 把
    /// 「VLC 查表 + `run[]`/`level[]` 两次取数组 + `last` 比较」压成一次查表。
    /// 条目 0 = 码长超过 `FAST_BITS`（退回 `decode_symbol`）。
    ///
    /// 打包：`len | run << 5 | (level 的低 8 位) << 14`。escape 符号记成
    /// `(66, 0)` —— 与 `decode_symbol` 的返回一致（调用方就靠 `level == 0` 认它）。
    fast: Vec<u32>,
}

/// 快表位宽（与 `Vlc` 的 `FAST_BITS` 同档：多数符号一次命中）。
const FAST_BITS: u32 = 9;

impl RlTable {
    pub fn new(
        vlc_table: &'static [[u32; 2]],
        run: &'static [i8],
        level: &'static [i8],
        last: usize,
    ) -> Self {
        let n = run.len();
        let mut max_level = [[0i8; 64]; 2];
        let mut max_run = [[0i8; 64]; 2];
        let mut index_run = [[n as u16; 64]; 2];
        for i in 0..n {
            let l = level[i];
            let r = run[i] as usize;
            let li = usize::from(i >= last);
            if r < 64 {
                if index_run[li][r] == n as u16 {
                    index_run[li][r] = i as u16;
                }
                if l > max_level[li][r] {
                    max_level[li][r] = l;
                }
            }
            if (l as usize) < 64 && run[i] > max_run[li][l as usize] {
                max_run[li][l as usize] = run[i];
            }
        }
        // 快表：把每个「码长 ≤ FAST_BITS」的符号解成 (run, level) 直接塞进表。
        let mut fast = vec![0u32; 1 << FAST_BITS];
        for (sym, &[code, len]) in vlc_table.iter().enumerate() {
            if len == 0 || len > FAST_BITS {
                continue;
            }
            let (run_v, level_v) = if sym == n {
                (66u32, 0u32) // escape
            } else {
                let mut r = run[sym] as i32 + 1;
                if sym >= last {
                    r += 192;
                }
                (r as u32, (level[sym] as i32 as u32) & 0xff)
            };
            let entry = len | (run_v << 5) | (level_v << 14);
            let base = (code << (FAST_BITS - len)) as usize;
            for slot in fast[base..base + (1 << (FAST_BITS - len))].iter_mut() {
                *slot = entry;
            }
        }
        Self {
            vlc: Vlc::new(vlc_table),
            run,
            level,
            last,
            max_level,
            max_run,
            index_run,
            fast,
        }
    }

    /// 快路径：偷看 `FAST_BITS` 位一次拿到 `(run, level)`（level 已乘 qmul 加 qadd）。
    /// `None` = 码长超过快表（或位不够）→ 调用方退回 `decode_symbol`。
    #[inline]
    pub fn decode_symbol_fast(&self, r: &mut BitReader, qmul: i32, qadd: i32) -> Option<(i32, i32)> {
        if r.bits_left() < FAST_BITS as usize {
            return None;
        }
        let e = self.fast[r.peek_u16(FAST_BITS) as usize];
        if e == 0 {
            return None;
        }
        r.skip(e & 0x1f).ok()?;
        let run = ((e >> 5) & 0x1ff) as i32;
        // ⚠️ level 字节为 0 = **escape** 符号：它的 level 必须是**字面 0**
        // （调用方靠 `level == 0` 认它），不能走 `*qmul + qadd` —— 那会变成 qadd。
        let lv = (((e >> 14) & 0xff) as u8 as i8) as i32;
        let level = if lv == 0 { 0 } else { lv * qmul + qadd };
        Some((run, level))
    }

    /// **编码侧**的反查：`(last, run, level)` → 符号下标。
    /// 不在表里就返回 `n`（调用方走 escape）。
    pub fn index_of(&self, last: usize, run: usize, level: i32) -> usize {
        let n = self.n();
        if run > 63 || !(1..=63).contains(&level) {
            return n;
        }
        let idx = self.index_run[last][run] as usize;
        if idx >= n || level > self.max_level[last][run] as i32 {
            return n;
        }
        idx + level as usize - 1
    }

    /// 编码用：`max_level[last][run]`（第一 escape 要减它）。
    pub fn max_level_at(&self, last: usize, run: usize) -> i32 {
        self.max_level[last][run] as i32
    }

    /// 编码用：`max_run[last][level]`（第二 escape 要减它）。
    pub fn max_run_at(&self, last: usize, level: i32) -> i32 {
        if (0..64).contains(&level) {
            self.max_run[last][level as usize] as i32
        } else {
            0
        }
    }

    /// 编码用：符号下标 → `(code, len)`。
    pub fn code_len(&self, sym: usize) -> Option<(u32, u32)> {
        self.vlc.code_len(sym)
    }

    /// 表里符号的个数；这个下标就是 escape。
    pub fn escape_symbol(&self) -> usize {
        self.n()
    }

    /// 符号数（run/level 的长度）；`== n` 的那个前缀码是 escape。
    fn n(&self) -> usize {
        self.run.len()
    }

    /// 符号下标 → `(run, level, last)`。
    ///
    /// ⚠️ 这里的 `run` 是**表里的原值**（0 起），与 `decode_ac` 内部的约定不同 ——
    /// 那边用的是 `run + 1`（1 起，因为 `i` 从 -1 开始）。给外面的解码器（`v2v-mp4v`）
    /// 用时按需自己加 1。
    pub fn symbol(&self, i: usize) -> (usize, i32, bool) {
        (self.run[i] as usize, self.level[i] as i32, i >= self.last)
    }

    /// 表里有多少个真符号（escape 的下标就是它）。
    pub fn len(&self) -> usize {
        self.n()
    }

    pub fn is_empty(&self) -> bool {
        self.n() == 0
    }

    /// 解一个 RL 符号。返回 `(run, level, last)`，其中 **`run` 已打包**：
    /// `run = table_run + 1`，且 `last` 时再 `+ 192`。
    /// `level` 对帧内（qmul=1/qadd=0）就是表里的值。`level == 0` 表示 escape。
    pub fn decode_symbol(&self, r: &mut BitReader, qmul: i32, qadd: i32) -> Result<(i32, i32, bool)> {
        let s = self.vlc.decode(r)?;
        if s == self.n() {
            return Ok((66, 0, false)); // escape
        }
        let last = s >= self.last;
        let mut run = self.run[s] as i32 + 1;
        if last {
            run += 192;
        }
        // 表里存的是**未反量化**的 level，这里在取出时才乘（帧内 qmul=1/qadd=0，是恒等）。
        Ok((run, self.level[s] as i32 * qmul + qadd, last))
    }
}
