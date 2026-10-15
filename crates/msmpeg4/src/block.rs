//! 系数（AC）解码 —— MP43 里最硬的一核。
//!
//! 一串系数按 RL（run-level）前缀码走，外加**三种 escape**：
//!
//! ```text
//!   普通：解出 (run, level, last)
//!   escape 码之后看两位：
//!     1x → 第一 escape：再解一个 RL 码，run 照用，level += max_level[last][run] * qmul
//!     01 → 第二 escape：再解一个 RL 码，run += max_run[last][level/qmul]，level 用解出的
//!     00 → 第三 escape：last(1) + run(6) + level(8 位有符号) 明文给出
//! ```
//!
//! 反量化：`level > 0 ? level*qmul + qadd : level*qmul - qadd`。
//! MS-MPEG-4 v3 的第三 escape 是 1+6+8 位。**`run_diff` 是帧内/帧间的唯一差别**：
//! 帧内 0、帧间 1，只作用在第二 escape 的 run 上。

use crate::bitreader::BitReader;
use crate::version::Version;
use crate::error::{Error, Result};
use crate::tables;
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
}

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
        Self {
            vlc: Vlc::new(vlc_table),
            run,
            level,
            last,
            max_level,
            max_run,
            index_run,
        }
    }

    /// `ff_rl_table` 的 6 张表：0/1/2 = 帧内亮度（低 / 高 / 中运动），3/4/5 = 色度 / 帧间。
    ///
    /// ⚠️ **解码端与编码端共用这一份** —— 这张派发表以前在 `frame.rs` 与 `encode.rs`
    /// 里各写了一遍（逐项参数完全相同）。按 AGENTS.md「一张派发表只有一份真源」：
    /// 加表、改 `last`、换码表都只改这里。
    ///
    /// 2 与 5 是 MPEG-4 的通用表（`ff_mpeg4_rl_intra` / `ff_h263_rl_inter`，各 102 项）——
    /// 都已经生成好了。P 帧的 `rl_table_index = 2` 会落到 5（`3 + 2`）。
    ///
    /// 下标越界**明确报 Unsupported**，不猜一张表顶上（读坏帧头时 `rl_table_index`
    /// 可以到 3 以上）。
    pub fn for_standard(index: usize) -> Result<Self> {
        Ok(match index {
            0 => Self::new(&tables::RL0_VLC, &tables::RL0_RUN, &tables::RL0_LEVEL, 85),
            1 => Self::new(&tables::RL2_VLC, &tables::RL2_RUN, &tables::RL2_LEVEL, 119),
            2 => Self::new(
                &tables::RL_INTRA_VLC,
                &tables::RL_INTRA_RUN,
                &tables::RL_INTRA_LEVEL,
                67,
            ),
            3 => Self::new(&tables::RL1_VLC, &tables::RL1_RUN, &tables::RL1_LEVEL, 81),
            4 => Self::new(&tables::RL4_VLC, &tables::RL4_RUN, &tables::RL4_LEVEL, 99),
            5 => Self::new(
                &tables::RL_INTER_VLC,
                &tables::RL_INTER_RUN,
                &tables::RL_INTER_LEVEL,
                58,
            ),
            _ => return Err(Error::Unsupported(format!("RL 表下标 {index}"))),
        })
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

    /// 解一个 RL 符号。返回 `(run, level, last)`；`run` 的高位（bit7）带 `last`，
    /// 打包方式见下。`level == 0` 表示解到了 escape。
    /// 解一个 RL 符号。返回 `(run, level, last)`，其中 **`run` 已打包**：
    /// `run = table_run + 1`，且 `last` 时再 `+ 192`。
    /// `level` 对帧内（qmul=1/qadd=0）就是表里的值。`level == 0` 表示 escape。
    fn decode_symbol(&self, r: &mut BitReader, qmul: i32, qadd: i32) -> Result<(i32, i32, bool)> {
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

    /// 从 `i` 开始解一串系数，写入 `block`（已按扫描序展开到自然序由调用方做）。
    ///
    /// `run_diff`：它只加在**第二 escape** 的 run 上。取值**按版本不同**：
    /// 帧内 = `version.intra_run_diff()`、帧间 = `version.inter_run_diff()`
    /// （见 `Version` 的说明 —— 注意帧间那一条是「**只有 V2 是 0**」）。
    ///
    /// 返回 `(扫描序下标, level)` 列表；`block[scan_idx]` 由调用方按 `scan` 落位，
    /// 这里直接返回列表，避免把扫描表耦合进来。
    pub fn decode_ac(
        &self,
        r: &mut BitReader,
        qmul: i32,
        qadd: i32,
        run_diff: i32,
        start_i: i32,
        ctx: &mut AcCtx,
    ) -> Result<Vec<(usize, i16)>> {
        let mut out = Vec::new();
        let mut i = start_i;

        loop {
            let (run, mut level, _) = self.decode_symbol(r, qmul, qadd)?;
            let mut needs_sign = true;

            if level == 0 {
                // ---- escape ----
                if r.bit()? == 1 {
                    // 第一 escape：再解一个 RL 符号，run 照用，level 补上该 (last,run) 的上界
                    let (run2, lvl2, _) = self.decode_symbol(r, qmul, qadd)?;
                    let base = run2 - if run2 >= 192 { 192 } else { 0 };
                    let li = usize::from(run2 >= 192);
                    i += run2;
                    level = lvl2 + self.max_level[li][((base - 1) & 63) as usize] as i32 * qmul;
                } else if r.bit()? == 1 {
                    // 第二 escape：再解一个 RL 符号，run 再补上该 (last, level) 的上界（+run_diff）
                    let (run2, lvl2, _) = self.decode_symbol(r, qmul, qadd)?;
                    let li = usize::from(run2 >= 192);
                    let lv = (lvl2 / qmul).clamp(0, 63) as usize;
                    i += run2 + self.max_run[li][lv] as i32 + run_diff;
                    level = lvl2;
                } else if ctx.version.is_wmv() {
                    // 第三 escape（**WMV1/WMV2 的变长版**）
                    //
                    // ⚠️ 「level 用几位、run 用几位」**一帧只读一次**，之后同一帧里的
                    // 第三 escape 全部沿用（`esc3_level_length` 在帧头解析的末尾归零）。
                    // 所以这段必须跨块、跨宏块保持状态 —— 当成无状态写会从第二个
                    // 第三 escape 起全体错位。
                    let last = r.bit()? != 0;
                    if ctx.esc3_level_length == 0 {
                        let ll = if ctx.qscale < 8 {
                            // 码率低时频谱稀疏，长度用 3 位直接给，0 是转义。
                            let v = r.u(3)?;
                            if v == 0 {
                                8 + r.u(1)?
                            } else {
                                v
                            }
                        } else {
                            // 高 qscale：一元码，读到 1 为止（上限 8）。
                            //
                            // ⚠️ **不要再补一次 skip**：ffmpeg 那边用的是 `SHOW_UBITS`
                            // （**不消费**）来测位，所以它循环之后要显式 `SKIP_BITS(1)`
                            // 把那个终止的 1 吃掉。这里换成 `r.bit()`（**消费**）之后，
                            // 那一步就多余了 —— 多读一位会让**整个帧从此错位**，
                            // 症状是后面所有块莫名其妙地错，离根因很远。
                            let mut v = 2u32;
                            while v < 8 && r.bit()? == 0 {
                                v += 1;
                            }
                            v
                        };
                        ctx.esc3_level_length = ll;
                        ctx.esc3_run_length = r.u(2)? + 3;
                    }
                    let raw_run = r.u(ctx.esc3_run_length)? as i32;
                    let sign = r.bit()? != 0;
                    let mag = r.u(ctx.esc3_level_length)? as i32;
                    let l = if sign { -mag } else { mag };
                    level = if l > 0 {
                        l * qmul + qadd
                    } else {
                        l * qmul - qadd
                    };
                    i += raw_run + 1;
                    if last {
                        i += 192;
                    }
                    needs_sign = false;
                } else {
                    // 第三 escape（**V1/V2/V3 的定长版**）：last(1) + run(6) + level（有符号 8 位）
                    let last = r.bit()? != 0;
                    let raw_run = r.u(6)? as i32;
                    let l = (r.u(8)? as u8) as i8 as i32;
                    level = if l > 0 {
                        l * qmul + qadd
                    } else {
                        l * qmul - qadd
                    };
                    i += raw_run + 1;
                    if last {
                        i += 192;
                    }
                    needs_sign = false;
                }
            } else {
                i += run;
            }

            if needs_sign && r.bit()? != 0 {
                level = -level;
            }

            if i > 62 {
                i -= 192;
                if i & !63 != 0 {
                    return Err(Error::Bad("ac coefficient overflow"));
                }
                out.push((i as usize, level as i16));
                break;
            }
            out.push((i as usize, level as i16));
        }

        Ok(out)
    }
}

/// 解系数时的**版本相关上下文** —— 由调用方跨块持有。
///
/// 之所以要带着走：WMV1/WMV2 的第三 escape 长度**在帧内粘住**（见 `decode_ac`），
/// 一帧里第一次遇到时才读，之后所有块沿用。调用方（`Decoder`）每帧把它归零一次。
#[derive(Debug, Clone, Copy, Default)]
pub struct AcCtx {
    pub version: Version,
    /// 当前帧的 `qscale` —— 决定第三 escape 的长度前缀怎么读（`< 8` 与 `>= 8` 两条路）。
    pub qscale: u8,
    /// 第三 escape 的 level 位宽（0 = 这一帧还没遇到过第三 escape）。
    pub esc3_level_length: u32,
    /// 第三 escape 的 run 位宽。
    pub esc3_run_length: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testbit::BitWriter;
    use crate::tables;

    fn table0() -> RlTable {
        RlTable::new(&tables::RL0_VLC, &tables::RL0_RUN, &tables::RL0_LEVEL, 85)
    }

    #[test]
    fn derived_tables_are_consistent() {
        let t = table0();
        // max_level[last][run] 必须是该 (last,run) 下出现过的最大 level。
        for i in 0..t.run.len() {
            let li = usize::from(i >= t.last);
            let r = t.run[i] as usize;
            assert!(t.max_level[li][r] >= t.level[i], "max_level 不是上界");
        }
        // 有一个 (last=0, run=0) 的 level=16 —— 抽查。
        assert_eq!(t.max_level[0][0], 16);
    }

    #[test]
    fn decodes_a_normal_sequence() {
        let t = table0();
        // 符号 0：level=1, run=0, last=0（表里 {0x1,2}）；符号 85 起是 last=1。
        let mut w = BitWriter::new();
        let (c0, l0) = (tables::RL0_VLC[0][0], tables::RL0_VLC[0][1]);
        w.put(c0, l0);
        w.put(0, 1); // 符号
        // 一个 last=1 的符号
        let idx = 85;
        let (c1, l1) = (tables::RL0_VLC[idx][0], tables::RL0_VLC[idx][1]);
        w.put(c1, l1);
        w.put(0, 1); // 符号
        let mut data = w.finish();
        data.extend_from_slice(&[0, 0]);

        let mut r = BitReader::new(&data);
        let out = t.decode_ac(&mut r, 1, 0, 0, 0, &mut AcCtx::default()).unwrap();
        assert_eq!(out.len(), 2);
        // run 打包为 `table_run + 1`，所以 run=0 的符号落在扫描位 1（0 是直流）。
        assert_eq!(out[0], (1, 1));
        // 第二个符号带 last：i += run+1+192，再因 i>62 减 192 → run+2。
        assert_eq!(out[1].0, tables::RL0_RUN[idx] as usize + 2);
        assert_eq!(out[1].1, tables::RL0_LEVEL[idx] as i16);
    }

    #[test]
    fn decodes_the_third_escape() {
        let t = table0();
        // escape 码 = RL0_VLC 的最后一项（下标 132）。
        let (ce, le) = (
            tables::RL0_VLC[tables::RL0_VLC.len() - 1][0],
            tables::RL0_VLC[tables::RL0_VLC.len() - 1][1],
        );
        let mut w = BitWriter::new();
        w.put(ce, le);
        w.put(0, 1); // b0 = 0
        w.put(0, 1); // b1 = 0  → 第三 escape
        w.put(1, 1); // last = 1
        w.put(5, 6); // run = 5
        w.put(7, 8); // level = 7（有符号 8 位）
        let data = w.finish();

        let mut r = BitReader::new(&data);
        let out = t.decode_ac(&mut r, 1, 0, 0, 0, &mut AcCtx::default()).unwrap();
        // i = 0 + run(5) + 1 = 6，last → i += 192 → 198 → >62 → 198-192 = 6
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], (6, 7));
    }

    #[test]
    fn dequantizes_inter_levels() {
        let t = table0();
        let mut w = BitWriter::new();
        let (c0, l0) = (tables::RL0_VLC[0][0], tables::RL0_VLC[0][1]);
        w.put(c0, l0);
        w.put(1, 1); // 负号 → level = -1
        let (ce, le) = (
            tables::RL0_VLC[tables::RL0_VLC.len() - 1][0],
            tables::RL0_VLC[tables::RL0_VLC.len() - 1][1],
        );
        w.put(ce, le);
        w.put(0, 1);
        w.put(0, 1);
        w.put(1, 1); // last
        w.put(0, 6); // run 0
        w.put(1, 8); // level 1（有符号 8 位）
        let data = w.finish();

        // 帧内：qmul=1, qadd=0
        let mut r = BitReader::new(&data);
        let out = t.decode_ac(&mut r, 1, 0, 0, 0, &mut AcCtx::default()).unwrap();
        assert_eq!(out[0], (1, -1));
        // 帧间：qmul=2*qscale(=4), qadd=(qscale-1)|1(=1)
        let mut r = BitReader::new(&data);
        let out = t.decode_ac(&mut r, 4, 1, 0, 0, &mut AcCtx::default()).unwrap();
        assert_eq!(out[0], (1, -4 - 1)); // -1*4 - 1
    }

    #[test]
    fn second_escape_adds_run_diff() {
        // 第二 escape 是帧内/帧间唯一不同的地方：`i += run + max_run[...] + run_diff`，
        // 帧内 run_diff = 0、帧间 = 1。
        let t = table0();
        let (ce, le) = (
            tables::RL0_VLC[tables::RL0_VLC.len() - 1][0],
            tables::RL0_VLC[tables::RL0_VLC.len() - 1][1],
        );
        let (c0, l0) = (tables::RL0_VLC[0][0], tables::RL0_VLC[0][1]);
        let mut w = BitWriter::new();
        w.put(ce, le); // escape 码
        w.put(0, 1); // b0 = 0
        w.put(1, 1); // b1 = 1 → 第二 escape
        w.put(c0, l0); // 内层 RL 符号
        w.put(0, 1); // 内层符号位
        // 收尾用一个 last 符号（下标 85 起 last=1），否则循环会继续读
        let idx = 85;
        w.put(tables::RL0_VLC[idx][0], tables::RL0_VLC[idx][1]);
        w.put(0, 1);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        let intra = t.decode_ac(&mut r, 1, 0, 0, 0, &mut AcCtx::default()).unwrap();
        let mut r = BitReader::new(&data);
        let inter = t.decode_ac(&mut r, 1, 0, 1, 0, &mut AcCtx::default()).unwrap();

        assert_eq!(intra.len(), 2);
        assert_eq!(inter.len(), 2);
        assert_eq!(
            inter[0].0,
            intra[0].0 + 1,
            "帧间的第二 escape 应当让 run 多走 1"
        );
        // 收尾符号的位置也跟着 +1
        assert_eq!(inter[1].0, intra[1].0 + 1);
        // level 不受 run_diff 影响
        assert_eq!(inter[0].1, intra[0].1);
    }
}
