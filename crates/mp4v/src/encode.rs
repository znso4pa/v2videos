//! MPEG-4 Part 2（`mp4v`）**全 I 帧**编码器。
//!
//! 一条窄路径：不做 P 帧、不做运动估计、不做码率控制、不做 dquant —— 整帧一个 qscale。
//! 目的是把 `.wmv → .mp4` 打通（`msmpeg4` 只有解码，MP4 容器又不收 MP43），
//! 而不是一步做到 ffmpeg 的压缩率。
//!
//! ## 与 MP43 编码器的关系：**只有地基能复用**
//!
//! | 能复用（`v2v_msmpeg4`） | 不能复用 |
//! |---|---|
//! | `fdct`（正向 DCT，定标一致） | 宏块类型表：MP43 是 `MB_INTRA`，这里是 H.263 的 `ff_h263_intra_MCBPC` |
//! | `BitWriter`（MSB 优先） | DC 熵编码：MP43 是「符号 + 8 位 escape」，这里是「**位长** VLC」 |
//! | `RlTable` / `Vlc`（前缀码容器） | AC escape：MP43 有三种、含义不同；这里是统一表挑最短 |
//! | `Frame`、`Error`/`Result`、`ZIGZAG` | DC/AC **预测**：tie-break 判据不同（MP43 是 `<=`，这里是 `<`） |
//! | `tables::RL_INTRA_*`（同一张表） | DC 缩放表：MP43 用 `OLD_Y_DC_SCALE`，这里用 `ff_mpeg4_y_dc_scale_table` |
//!
//! ## 一帧的处理顺序（每一步的顺序都是语法的一部分，不能换）
//!
//! ```text
//! 1. 6 个块各做正向 DCT → 量化（DC 用 dc_scale，AC 用 H.263 的 qmul/qadd）
//! 2. **先**把 6 个块的 DC 预测全部算完（`ff_mpeg4_pred_dc`）—— 它会**就地更新** DC 网格，
//!    而块 1/2/3 要读块 0 刚写进去的值，所以必须按 n = 0,1,2,3,4,5 顺序走
//! 3. AC 预测（可选，见下）—— 同样按 n 顺序走，因为块 1 要读块 0 的 ac_val
//! 4. 由**扫描序里最后一个非零系数**定 cbp
//! 5. 写 MCBPC（I 表）→ ac_pred 位 → CBPY → 6 个块（每块：DC 再 AC）
//! ```
//!
//! ## `ac_pred` 是逐宏块试出来的，不是拍脑袋开的
//!
//! `ac_pred` 是**每宏块一个位**。开了之后，本块首行/首列会减掉邻居的首行/首列再编码。
//! 邻居和本块相似时省码字，不相似时反而费码字。ffmpeg 用一个近似的码率模型打分；
//! 这里**直接两种都写完、比位数**（`BitWriter::bit_pos`），选短的 —— 小宏块、算得准，
//! 而且不需要维护一套码率模型。
//!
//! 关键性质：`ac_val`（留给邻居的首行/首列）在**两种情况下都是「预测前」的值**
//! （解码端加完预测后正好还原成预测前的值），所以试两种**不用回滚网格**，
//! 这让「比位数」变得很便宜。
//!
//! ## 全 I 帧为什么不用重建
//!
//! 没有 P 帧就没有参考帧，所以**不需要反量化 + IDCT** 那一趟。
//! 代价是误差不累积（对全 I 帧反而是好事），而且少一整条容易错的路径。
//!
//! ## DC / AC 网格**每帧重建**
//!
//! ffmpeg 的 `dc_val` / `ac_val` 是跨帧复用的（`mpegvideo.c` 只在初始化时填 1024）。
//! 但全 I 帧时每个在帧内的格子都是「先写后读」，帧外的格子恒为 1024 / 0，
//! 所以每帧重建与跨帧复用**结果相同** —— 重建更省心，也就没有跨帧脏状态可错。

use v2v_msmpeg4::{fdct, BitWriter, Error, Frame, Result, RlTable, ZIGZAG};

use crate::header::{
    time_increment_bits, write_user_data, write_visual_object_header, write_vol, write_vop_header,
    VolParams, VopParams, PROFILE_SIMPLE_LEVEL1,
};
use crate::tables;

/// 越界的 DC 邻居值。ffmpeg 把整块 `dc_val_base` 填成 1024
/// （`mpegvideo.c:607`），所以帧外读到的是 1024 而不是 0。
const DC_NONE: i32 = 1024;

/// 一个块留给邻居的 16 个值：`[1..8]` 是**首列**（给右边用），`[9..16]` 是**首行**（给下面用）。
/// `[0]` 与 `[8]` 是 DC 的槽位，格式里不参与 AC 预测，恒 0（与 ffmpeg 的 `ac_val` 布局一致）。
type AcVal = [i16; 16];

/// MPEG-4 的 qscale 上限。
const QSCALE_MAX: u8 = 31;

/// 编码配置。
#[derive(Debug, Clone, Copy)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    /// 帧率分子 —— 写进 VOL 的 `vop_time_increment_resolution`，也决定 `time_increment_bits`。
    pub fps_num: u32,
    /// 整帧固定的量化步长（1..=31）。越大越糊、文件越小。
    ///
    /// 和 MP43 不同，这里**没有**「量化后 level 必须塞进 8 位 escape」的约束 ——
    /// MPEG-4 的 ESC3 是 12 位，所以 qscale 可以放心取小。
    pub qscale: u8,
    /// 要不要逐宏块试 AC 预测（见模块文档）。默认开。
    pub ac_pred: bool,
    /// VOL 里的 `low_delay`。恒 1 更省事（没有 B 帧）。
    pub low_delay: bool,
    /// 视觉对象序列头的 profile_and_level_indication。
    pub profile_and_level: u8,
    /// 用户数据（`00 00 01 B2` 之后）。空串表示不写。
    pub user_data: &'static str,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            fps_num: 25,
            // 全 I 帧没有 P 帧帮忙补，qscale 取 8 会明显糊；4 是「看着还行、文件别太离谱」的点。
            qscale: 4,
            ac_pred: true,
            low_delay: true,
            profile_and_level: PROFILE_SIMPLE_LEVEL1,
            user_data: "v2video",
        }
    }
}

/// MPEG-4 Part 2 编码器（全 I 帧）。
pub struct Encoder {
    cfg: EncoderConfig,
    mb_w: usize,
    mb_h: usize,
    time_increment_bits: u32,
    /// 已经编过多少帧 —— 决定 VOP 头里的 `time_mod` 与回绕位。
    frame_index: u64,
    /// 上一次时间基回绕的次数（`frame_index / fps_num` 的旧值）。
    last_time_base: u64,
}

impl Encoder {
    pub fn new(cfg: EncoderConfig) -> Result<Self> {
        if cfg.width == 0
            || cfg.height == 0
            || !cfg.width.is_multiple_of(2)
            || !cfg.height.is_multiple_of(2)
        {
            return Err(Error::Bad("尺寸必须是正的偶数（YUV420P）"));
        }
        if !(1..=QSCALE_MAX).contains(&cfg.qscale) {
            return Err(Error::Bad("qscale 必须在 1..=31"));
        }
        if cfg.fps_num == 0 {
            return Err(Error::Bad("帧率分子不能是 0"));
        }
        Ok(Self {
            mb_w: (cfg.width as usize).div_ceil(16),
            mb_h: (cfg.height as usize).div_ceil(16),
            time_increment_bits: time_increment_bits(cfg.fps_num),
            frame_index: 0,
            last_time_base: 0,
            cfg,
        })
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    /// 宏块数 `(宽, 高)`。
    pub fn mb_count(&self) -> (usize, usize) {
        (self.mb_w, self.mb_h)
    }

    /// 塞进 `esds` 的 `DecoderSpecificInfo`。
    ///
    /// **就是基本流前缀那一份字节**（VOS + 视觉对象 + VOL + 用户数据），不是只有 VOL ——
    /// 少了 VOS 那两级，`profile_and_level_indication` 就丢了，`ffprobe` 只能报
    /// `profile=unknown`（参考样本报的是 `Simple Profile` / `level 1`）。
    /// ffmpeg 的 `esds` 也是这么放的：它的 DSI 以 `00 00 01 B0 01` 开头。
    pub fn decoder_specific_info(&self) -> Vec<u8> {
        self.elementary_stream_header()
    }

    /// 整条基本流的前缀：VOS + 视觉对象 + VOL + 用户数据。
    ///
    /// 只在流的最前面写一次。每个 I 帧前面也可以重写一遍 VOL（让流能任意位置起播），
    /// 但那样每一帧都多 40 多字节；全 I 帧的流里由调用方按需决定（见 `encode_i_frame_with_vol`）。
    pub fn elementary_stream_header(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        write_visual_object_header(&mut w, self.cfg.profile_and_level);
        self.write_vol_into(&mut w);
        if !self.cfg.user_data.is_empty() {
            write_user_data(&mut w, self.cfg.user_data);
        }
        w.finish()
    }

    fn write_vol_into(&self, w: &mut BitWriter) {
        write_vol(
            w,
            &VolParams {
                width: self.cfg.width,
                height: self.cfg.height,
                fps_num: self.cfg.fps_num,
                low_delay: self.cfg.low_delay,
            },
        );
    }

    /// 编一个 I 帧（只含 VOP 头 + 宏块，不含 VOL）。
    pub fn encode_i_frame(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        self.encode_i_frame_with_vol(frame, false)
    }

    /// 同 `encode_i_frame`，但可以要求在 VOP 前重写一遍 VOL。
    pub fn encode_i_frame_with_vol(&mut self, frame: &Frame, with_vol: bool) -> Result<Vec<u8>> {
        if frame.width != self.cfg.width || frame.height != self.cfg.height {
            return Err(Error::Bad("帧尺寸与配置不一致"));
        }

        let (mb_w, mb_h) = (self.mb_w, self.mb_h);
        let (pw, ph) = (mb_w * 16, mb_h * 16);
        let (py, pu, pv) = pad_planes(frame, pw, ph);

        let qscale = self.cfg.qscale;
        let q = i32::from(qscale);
        let qmul = q << 1;
        let qadd = (q - 1) | 1;
        let y_dc_scale = i32::from(tables::Y_DC_SCALE[qscale as usize]);
        let c_dc_scale = i32::from(tables::C_DC_SCALE[qscale as usize]);

        let rl = intra_rl_table();

        let mut w = BitWriter::new();
        if with_vol {
            self.write_vol_into(&mut w);
        }
        let time_base = self.frame_index / u64::from(self.cfg.fps_num);
        write_vop_header(
            &mut w,
            &VopParams {
                time_mod: (self.frame_index % u64::from(self.cfg.fps_num)) as u32,
                time_incr: (time_base - self.last_time_base) as u32,
                time_increment_bits: self.time_increment_bits,
                qscale,
            },
        );
        self.last_time_base = time_base;
        self.frame_index += 1;

        // DC / AC 网格：每帧重建（见模块文档）。
        let mut dc = DcGrid::new(mb_w, mb_h, y_dc_scale, c_dc_scale);
        let mut ac = AcGrid::new(mb_w, mb_h);

        let mut blocks = [[0i32; 64]; 6];
        for mby in 0..mb_h {
            for mbx in 0..mb_w {
                // ---- 1. 正向 DCT + 量化 ----
                for (n, slot) in blocks.iter_mut().enumerate() {
                    let (plane, stride, px, py) = block_at(&py, &pu, &pv, pw, mbx, mby, n);
                    let mut coeff = [0i32; 64];
                    fdct(&plane[py * stride + px..], stride, &mut coeff);
                    quantize_block(&mut coeff, n, qmul, qadd, y_dc_scale, c_dc_scale);
                    *slot = coeff;
                }

                // ---- 2. DC 预测（顺序敏感：会就地更新网格）----
                let mut dc_diff = [0i32; 6];
                let mut dir = [0usize; 6];
                for n in 0..6 {
                    let (pred_level, d) = dc.predict(n, mbx, mby);
                    dc_diff[n] = blocks[n][0] - pred_level;
                    dir[n] = d;
                    dc.update(blocks[n][0], n, mbx, mby);
                }

                // ---- 3+4+5. AC 预测（两种都试）+ 定 cbp + 写宏块 ----
                // 试两种写法：普通扫描 vs AC 预测。哪种位流短用哪种。
                // `ac` 网格两种写法留下的一样（见 `write_mb` 的说明），所以先跑「预测版」再跑
                // 「普通版」不会互相污染。
                if self.cfg.ac_pred {
                    let mut predicted = BitWriter::new();
                    write_mb(
                        &mut predicted,
                        mbx,
                        mby,
                        &blocks,
                        &dc_diff,
                        &dir,
                        true,
                        &rl,
                        &mut ac,
                    )?;
                    let mut plain = BitWriter::new();
                    write_mb(
                        &mut plain, mbx, mby, &blocks, &dc_diff, &dir, false, &rl, &mut ac,
                    )?;
                    if predicted.bit_pos() < plain.bit_pos() {
                        w.append(&predicted);
                    } else {
                        w.append(&plain);
                    }
                } else {
                    write_mb(
                        &mut w, mbx, mby, &blocks, &dc_diff, &dir, false, &rl, &mut ac,
                    )?;
                }
            }
        }

        Ok(w.finish())
    }
}

/// 帧内 RL 表：`ff_mpeg4_rl_intra`，`n = 102`（下标 102 是 escape）、`last = 67`。
fn intra_rl_table() -> RlTable {
    RlTable::new(
        &tables::RL_INTRA_VLC,
        &tables::RL_INTRA_RUN,
        &tables::RL_INTRA_LEVEL,
        67,
    )
}

// ───────────────────────── 宏块 / 块 写出 ─────────────────────────

/// 写一个 I 宏块。`use_ac_pred` 决定走哪条路（两种写法各写一遍，由调用方比位数）。
///
/// 无论走哪条路，`ac_val` 网格都要更新，而且存的是**预测前**的值 ——
/// 解码端加完预测正好还原成它，所以两种写法留下的是同一份网格，
/// 试两种**不用回滚**（这是「比位数」能做得这么便宜的原因）。
#[allow(clippy::too_many_arguments)]
fn write_mb(
    w: &mut BitWriter,
    mbx: usize,
    mby: usize,
    blocks: &[[i32; 64]; 6],
    dc_diff: &[i32; 6],
    dir: &[usize; 6],
    use_ac_pred: bool,
    rl: &RlTable,
    ac: &mut AcGrid,
) -> Result<()> {
    // ---- AC 预测：按 n = 0..5 走（块 1 要读块 0 刚存进去的值）----
    let mut work = *blocks;
    let mut scan: [&[u8; 64]; 6] = [&ZIGZAG; 6];
    for n in 0..6 {
        let cell = cell_of(n, mbx, mby);
        let nb = ac.neighbor(n, dir[n], (cell.0 as usize, cell.1 as usize));
        // 存回预测前的首行/首列 —— 必须在读**下一个块**之前做。
        store_ac_val(ac, n, (cell.0 as usize, cell.1 as usize), &blocks[n]);
        if use_ac_pred {
            apply_ac_pred(&mut work[n], &nb, dir[n]);
        }
        scan[n] = if !use_ac_pred {
            &ZIGZAG
        } else if dir[n] == 0 {
            &tables::V_SCAN
        } else {
            &tables::H_SCAN
        };
    }

    // ---- cbp：由**该扫描序下**最后一个非零系数决定 ----
    let mut cbp = 0usize;
    for n in 0..6 {
        if has_ac(&work[n], scan[n]) {
            cbp |= 1 << (5 - n);
        }
    }

    // ---- 宏块头：MCBPC → ac_pred 位 → CBPY ----
    write_mcbpc(w, cbp & 3)?;
    w.put(u32::from(use_ac_pred), 1);
    write_cbpy(w, cbp >> 2)?;

    // ---- 6 个块：每块先 DC 再 AC ----
    for n in 0..6 {
        write_dc(w, dc_diff[n], n < 4)?;
        write_ac(w, rl, &work[n], scan[n])?;
    }
    Ok(())
}

/// 块在扫描序 `scan` 下有没有 AC 系数（扫描序下标 >= 1）。
fn has_ac(block: &[i32; 64], scan: &[u8; 64]) -> bool {
    (1..64).any(|i| block[scan[i] as usize] != 0)
}

// ───────────────────────── 量化 ─────────────────────────

/// 量化一个块的 64 个自然序系数（就地）。
///
/// - **DC**（下标 0）：`round(F(0,0) / dc_scale)`。
///   解码端 `block[0] *= dc_scale` 再进 IDCT（`mpegvideo.c:210`），
///   而我们的 `fdct` 给的是 `F(0,0) = 8 × 均值`，所以 `level = 8×均值 / dc_scale`
///   正好还原 —— `dc_scale` 最小是 8，`level` 落在 `[0, 255]`。
/// - **AC**：H.263 量化，`level` 使 `level*qmul ± qadd` 最接近 `F`。
///   和 MP43 是同一个式子（`qmul = 2*qscale`、`qadd = (qscale-1)|1`），
///   差别只在 MP43 把 level 限制在 8 位 escape 内，MPEG-4 的 ESC3 是 12 位、不用管。
fn quantize_block(
    coeff: &mut [i32; 64],
    n: usize,
    qmul: i32,
    qadd: i32,
    y_dc_scale: i32,
    c_dc_scale: i32,
) {
    let dc_scale = if n < 4 { y_dc_scale } else { c_dc_scale };
    coeff[0] = div_round(coeff[0], dc_scale);
    for c in coeff.iter_mut().skip(1) {
        *c = quantize_ac(*c, qmul, qadd);
    }
}

/// 取整除法（`b > 0`，`a` 可负）—— 四舍五入、远离零。
fn div_round(a: i32, b: i32) -> i32 {
    debug_assert!(b > 0);
    if a >= 0 {
        (a + b / 2) / b
    } else {
        -((-a + b / 2) / b)
    }
}

/// 反量化的逆：`level` 使 `level*qmul ± qadd` 最接近 `f`。
fn quantize_ac(f: i32, qmul: i32, qadd: i32) -> i32 {
    if f > 0 {
        div_round(f - qadd, qmul).max(0)
    } else if f < 0 {
        div_round(f + qadd, qmul).min(0)
    } else {
        0
    }
}

// ───────────────────────── DC 预测 ─────────────────────────

/// DC 预测网格。存的是 `level * dc_scale`（与 ffmpeg 的 `dc_val` 同口径），
/// 并且**截到 `[0, 2047]`** —— 解码端就是这么截的（`ff_mpeg4_pred_dc`），
/// 不跟着截，预测值会在极端块上分叉。
///
/// ⚠️ **Cb 与 Cr 必须是两张独立的表**（ffmpeg 是 `dc_val[1]` / `dc_val[2]`）。
/// 两者格位完全相同（都是 `(mbx, mby)`），合一张表就会让 Cb 的预测值串到 Cr 上 ——
/// 位流照样能编出来、单看也「像那么回事」，但解码端对不上。
struct DcGrid {
    /// 亮度：`2*mb_w × 2*mb_h` 个 8×8 块。
    luma: Vec<i32>,
    /// 色度：`[0] = Cb`、`[1] = Cr`，各 `mb_w × mb_h`。
    chroma: [Vec<i32>; 2],
    mb_w: usize,
    y_scale: i32,
    c_scale: i32,
}

impl DcGrid {
    fn new(mb_w: usize, mb_h: usize, y_scale: i32, c_scale: i32) -> Self {
        Self {
            luma: vec![DC_NONE; 2 * mb_w * 2 * mb_h],
            chroma: [vec![DC_NONE; mb_w * mb_h], vec![DC_NONE; mb_w * mb_h]],
            mb_w,
            y_scale,
            c_scale,
        }
    }

    /// 读一个格子的 DC 值；越界返回 `DC_NONE`。
    fn at(&self, n: usize, col: i32, row: i32) -> i32 {
        if col < 0 || row < 0 {
            return DC_NONE;
        }
        let (c, r) = (col as usize, row as usize);
        if n < 4 {
            let w = 2 * self.mb_w;
            if c >= w || r >= self.luma.len() / w {
                return DC_NONE;
            }
            self.luma[r * w + c]
        } else {
            let p = &self.chroma[n - 4];
            if c >= self.mb_w || r >= p.len() / self.mb_w {
                return DC_NONE;
            }
            p[r * self.mb_w + c]
        }
    }

    /// 算预测**水平**（不是系数）与方向：`0` = 取左（A）、`1` = 取上（C）。
    ///
    /// 判据是 `|a-b| < |b-c|` → 取上。注意 MP43 那边是 `<=`，差一个等号，
    /// 会让部分宏块的预测方向反过来 —— 两边的代码不能互相抄。
    fn predict(&self, n: usize, mbx: usize, mby: usize) -> (i32, usize) {
        let (col, row) = cell_of(n, mbx, mby);
        let a = self.at(n, col - 1, row);
        let b = self.at(n, col - 1, row - 1);
        let c = self.at(n, col, row - 1);
        let (pred, dir) = if (a - b).abs() < (b - c).abs() {
            (c, 1)
        } else {
            (a, 0)
        };
        let scale = if n < 4 { self.y_scale } else { self.c_scale };
        // 网格里存的是 level*scale，这里换回 level。`+scale/2` 是四舍五入。
        ((pred + (scale >> 1)) / scale, dir)
    }

    /// 把本块的**最终** DC level 写回网格（解码端就是这么存的）。
    fn update(&mut self, level: i32, n: usize, mbx: usize, mby: usize) {
        let scale = if n < 4 { self.y_scale } else { self.c_scale };
        let v = (level * scale).clamp(0, 2047);
        let (col, row) = cell_of(n, mbx, mby);
        if n < 4 {
            self.luma[row as usize * (2 * self.mb_w) + col as usize] = v;
        } else {
            self.chroma[n - 4][row as usize * self.mb_w + col as usize] = v;
        }
    }
}

/// 块 `n` 在 DC 网格里的格位 `(列, 行)`。
fn cell_of(n: usize, mbx: usize, mby: usize) -> (i32, i32) {
    if n < 4 {
        ((2 * mbx + (n & 1)) as i32, (2 * mby + (n >> 1)) as i32)
    } else {
        (mbx as i32, mby as i32)
    }
}

// ───────────────────────── AC 预测 ─────────────────────────

/// AC 预测网格：每个块留 16 个 i16 给右边/下面的邻居。
///
/// 与 `DcGrid` 同理，**Cb / Cr 必须分开**（ffmpeg 是 `ac_val[1]` / `ac_val[2]`）。
struct AcGrid {
    luma: Vec<AcVal>,
    chroma: [Vec<AcVal>; 2],
    mb_w: usize,
}

impl AcGrid {
    fn new(mb_w: usize, mb_h: usize) -> Self {
        Self {
            luma: vec![[0i16; 16]; 2 * mb_w * 2 * mb_h],
            chroma: [vec![[0i16; 16]; mb_w * mb_h], vec![[0i16; 16]; mb_w * mb_h]],
            mb_w,
        }
    }

    /// 取块 `n` 的邻居留给它的值。
    ///
    /// 越界（左邻居在帧外 / 上邻居在帧外）返回全 0 —— ffmpeg 的 `ac_val` 是
    /// 零初始化的，帧外格子从没被写过，读出来就是 0。
    fn neighbor(&self, n: usize, dir: usize, cell: (usize, usize)) -> AcVal {
        let (col, row) = cell;
        if dir == 0 {
            if col == 0 {
                return [0i16; 16];
            }
            self.get(n, col - 1, row)
        } else {
            if row == 0 {
                return [0i16; 16];
            }
            self.get(n, col, row - 1)
        }
    }

    fn get(&self, n: usize, col: usize, row: usize) -> AcVal {
        if n < 4 {
            self.luma[row * (2 * self.mb_w) + col]
        } else {
            self.chroma[n - 4][row * self.mb_w + col]
        }
    }

    /// 存回本块的首列/首行（**预测前**的值）。
    fn put(&mut self, n: usize, col: usize, row: usize, v: AcVal) {
        if n < 4 {
            self.luma[row * (2 * self.mb_w) + col] = v;
        } else {
            self.chroma[n - 4][row * self.mb_w + col] = v;
        }
    }
}

/// `ff_mpeg4_pred_ac` 的编码侧：把邻居的首行/首列**减**掉。
///
/// - `dir == 0`（取左）：减邻居的**首列**，落在本块的首列（`block[i*8]`）。
/// - `dir == 1`（取上）：减邻居的**首行**，落在本块的首行（`block[i]`）。
///
/// `i` 从 1 起 —— 下标 0 是 DC，不参与。
fn apply_ac_pred(block: &mut [i32; 64], nb: &AcVal, dir: usize) {
    if dir == 0 {
        for i in 1..8 {
            block[i * 8] -= i32::from(nb[i]);
        }
    } else {
        for i in 1..8 {
            block[i] -= i32::from(nb[8 + i]);
        }
    }
}

/// 把本块的首列/首行存回网格（存**预测前**的值）。
fn store_ac_val(ac: &mut AcGrid, n: usize, cell: (usize, usize), block: &[i32; 64]) {
    let mut v = [0i16; 16];
    for i in 1..8 {
        v[i] = block[i * 8] as i16; // 首列 → 给右边
        v[8 + i] = block[i] as i16; // 首行 → 给下面
    }
    ac.put(n, cell.0, cell.1, v);
}

// ───────────────────────── 熵编码 ─────────────────────────

/// 一个块的 AC：MPEG-4 的 RL 前缀码 + 三种 escape。
///
/// 与 MP43 的 `write_ac` **不是一回事**：MP43 的三种 escape 含义不同
/// （第一 escape 补 level、第二 escape 补 run、第三 escape 明文 1+6+8 位）。
/// 这里 ffmpeg 是预先算好一张「统一表」在 ESC0/ESC1/ESC2/ESC3 里挑最短的 ——
/// 我们**在写的时候现算**同样的挑选，省掉那张 8192 项的表。
fn write_ac(w: &mut BitWriter, rl: &RlTable, block: &[i32; 64], scan: &[u8; 64]) -> Result<()> {
    // 帧内 DC 已经单独写过，AC 从扫描序第 1 个开始。
    let Some(last_index) = (1..64).rev().find(|&i| block[scan[i] as usize] != 0) else {
        return Ok(());
    };
    let mut last_non_zero: i32 = 0;
    for i in 1..=last_index {
        let level = block[scan[i] as usize];
        if level == 0 {
            continue;
        }
        let run = (i as i32 - last_non_zero - 1) as usize;
        write_rl(w, rl, i == last_index, run, level)?;
        last_non_zero = i as i32;
    }
    Ok(())
}

/// 一个 `(last, run, level)` 的最省写法。
///
/// 四种候选（`init_uni_mpeg4_rl_tab` 的逻辑，现算而不是查表）：
///
/// | | 位流 | 长度 |
/// |---|---|---|
/// | ESC0 | `code` + sign | `len(code)+1` |
/// | ESC1 | `esc` `0` `code'` + sign，其中 `level' = level - max_level[last][run]` | `len(esc)+1+len(code')+1` |
/// | ESC2 | `esc` `10` `code''` + sign，其中 `run' = run - max_run[last][level] - 1` | `len(esc)+2+len(code'')+1` |
/// | ESC3 | `esc` `11` last(1) run(6) 1 level(12) 1 | `len(esc)+23` |
///
/// 严格取最短的（并列时取靠前的候选，与 ffmpeg 的 `len < best` 一致）。
fn write_rl(w: &mut BitWriter, rl: &RlTable, last: bool, run: usize, level: i32) -> Result<()> {
    let esc = rl.escape_symbol();
    let (esc_code, esc_len) = rl
        .code_len(esc)
        .ok_or(Error::Bad("RL 表里没有 escape 码"))?;
    let li = usize::from(last);
    let lvl = level.abs();
    let sign = u32::from(level < 0);

    // 候选：`(code, len)` —— 取最短。
    let mut best: Option<(u32, u32)> = None;
    let better = |code: u32, len: u32, best: &mut Option<(u32, u32)>| {
        if best.is_none_or(|(_, bl)| len < bl) {
            *best = Some((code, len));
        }
    };

    // ESC0
    let c0 = rl.index_of(li, run, lvl);
    if c0 != esc {
        if let Some((code, len)) = rl.code_len(c0) {
            better(code << 1 | sign, len + 1, &mut best);
        }
    }
    // ESC1
    let level1 = lvl - rl.max_level_at(li, run);
    if level1 > 0 {
        let c1 = rl.index_of(li, run, level1);
        if c1 != esc {
            if let Some((code, len)) = rl.code_len(c1) {
                let bits = ((esc_code << 1) << len | code) << 1 | sign;
                better(bits, esc_len + 1 + len + 1, &mut best);
            }
        }
    }
    // ESC2
    let run1 = run as i32 - rl.max_run_at(li, lvl) - 1;
    if run1 >= 0 {
        let c2 = rl.index_of(li, run1 as usize, lvl);
        if c2 != esc {
            if let Some((code, len)) = rl.code_len(c2) {
                let bits = ((esc_code << 2 | 0b10) << len | code) << 1 | sign;
                better(bits, esc_len + 2 + len + 1, &mut best);
            }
        }
    }
    // ESC3：`esc` `11` last run(6) 1 level(12) 1 —— 12 位有符号，用补码低 12 位。
    {
        let bits = ((((esc_code << 2 | 0b11) << 1 | u32::from(last)) << 6 | run as u32) << 1 | 1)
            << 12
            | ((level as u32) & 0xfff);
        let bits = bits << 1 | 1;
        better(bits, esc_len + 23, &mut best);
    }

    let (code, len) = best.ok_or(Error::Bad("AC 系数写不出来"))?;
    debug_assert!(len <= 32, "一次最多写 32 位，ESC3 是 {len} 位");
    w.put(code, len);
    Ok(())
}

/// `mpeg4_encode_dc`：VLC 的**符号是 DC 差的位长**，后面跟位长个位。
///
/// 值的写法是 `get_xbits` 语义：非负照写，负数是 `(-|v|) ^ ((1<<size)-1)`。
/// `size > 8` 时后面还有一个 marker 位（值为 1）。
fn write_dc(w: &mut BitWriter, level: i32, luma: bool) -> Result<()> {
    if !(-2047..=2047).contains(&level) {
        return Err(Error::Bad("DC 差超出 DC 表能表示的范围"));
    }
    let size = bit_length(level.unsigned_abs()) as usize;
    let tab = if luma {
        &tables::DC_LUM
    } else {
        &tables::DC_CHROM
    };
    let (code, len) = tab
        .get(size)
        .map(|e| (e[0], e[1]))
        .ok_or(Error::Bad("DC 位长超出 DC 表"))?;
    w.put(u32::from(code), u32::from(len));
    if size > 0 {
        let payload = if level < 0 {
            ((-level) as u32) ^ ((1u32 << size) - 1)
        } else {
            level as u32
        };
        w.put(payload, size as u32);
        if size > 8 {
            w.put(1, 1); // marker
        }
    }
    Ok(())
}

/// `|v|` 的位长（`v == 0` → 0）。
fn bit_length(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        32 - v.leading_zeros()
    }
}

/// I 宏块的 MCBPC：低 2 位是色度 CBP，第 3 位（值 4）表示「本宏块带 dquant」。
/// 全 I 帧不做 dquant，所以只用 0..=3。
fn write_mcbpc(w: &mut BitWriter, cbpc: usize) -> Result<()> {
    let code = *tables::INTRA_MCBPC_CODE
        .get(cbpc)
        .ok_or(Error::Bad("cbpc 越界"))?;
    let bits = *tables::INTRA_MCBPC_BITS
        .get(cbpc)
        .ok_or(Error::Bad("cbpc 越界"))?;
    w.put(u32::from(code), u32::from(bits));
    Ok(())
}

/// CBPY：`ff_h263_cbpy_tab[cbpy] = (code, len)`。
///
/// ⚠️ **不做取反**。H.263P 与 MPEG-4 的 **P** 帧要 `cbpy ^ 0xF`，
/// 但 MPEG-4 的 **I** 帧是直取（`mpeg4_decode_mb`：`cbp = (cbpc & 3) | (cbpy << 2)`）。
fn write_cbpy(w: &mut BitWriter, cbpy: usize) -> Result<()> {
    let e = tables::CBPY.get(cbpy).ok_or(Error::Bad("cbpy 越界"))?;
    w.put(u32::from(e[0]), u32::from(e[1]));
    Ok(())
}

// ───────────────────────── 像素搬运 ─────────────────────────

/// 把帧补齐到宏块对齐（边缘复制）。返回 `(y, u, v)` 三块平面。
fn pad_planes(frame: &Frame, pw: usize, ph: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let (cw, ch) = (w / 2, h / 2);
    let (pcw, pch) = (pw / 2, ph / 2);

    let mut y = vec![0u8; pw * ph];
    for r in 0..ph {
        let src = &frame.y[r.min(h - 1) * w..][..w];
        let dst = &mut y[r * pw..][..pw];
        dst[..w].copy_from_slice(src);
        // 右边补齐：边缘复制。
        dst[w..].fill(src[w - 1]);
    }

    let pad_chroma = |src: &[u8]| {
        let mut out = vec![0u8; pcw * pch];
        for r in 0..pch {
            let s = &src[r.min(ch - 1) * cw..][..cw];
            let d = &mut out[r * pcw..][..pcw];
            d[..cw].copy_from_slice(s);
            d[cw..].fill(s[cw - 1]);
        }
        out
    };
    (y, pad_chroma(&frame.u), pad_chroma(&frame.v))
}

/// 取宏块 `(mbx, mby)` 里第 `n` 个 8×8 块的平面、行距与左上角坐标。
fn block_at<'a>(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    pw: usize,
    mbx: usize,
    mby: usize,
    n: usize,
) -> (&'a [u8], usize, usize, usize) {
    if n < 4 {
        let px = mbx * 16 + (n & 1) * 8;
        let py = mby * 16 + (n >> 1) * 8;
        (y, pw, px, py)
    } else if n == 4 {
        (u, pw / 2, mbx * 8, mby * 8)
    } else {
        (v, pw / 2, mbx * 8, mby * 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(w: u32, h: u32, luma: u8, chroma: u8) -> Frame {
        Frame {
            width: w,
            height: h,
            y: vec![luma; (w * h) as usize],
            u: vec![chroma; (w * h / 4) as usize],
            v: vec![chroma; (w * h / 4) as usize],
        }
    }

    #[test]
    fn bit_length_is_the_index_of_the_top_set_bit_plus_one() {
        assert_eq!(bit_length(0), 0);
        assert_eq!(bit_length(1), 1);
        assert_eq!(bit_length(2), 2);
        assert_eq!(bit_length(3), 2);
        assert_eq!(bit_length(255), 8);
        assert_eq!(bit_length(256), 9);
    }

    #[test]
    fn quantize_ac_inverts_the_h263_dequant() {
        // 解码端：`level > 0 ? level*qmul + qadd : level*qmul - qadd`
        let (qmul, qadd) = (2 * 4, (4 - 1) | 1);
        for level in [-20i32, -3, -1, 0, 1, 3, 20] {
            let f = if level > 0 {
                level * qmul + qadd
            } else if level < 0 {
                level * qmul - qadd
            } else {
                0
            };
            assert_eq!(quantize_ac(f, qmul, qadd), level, "level {level} 没还原");
        }
        // 落在两个 level 之间时取最近的
        assert_eq!(quantize_ac(qmul + qadd - 1, qmul, qadd), 1);
        assert_eq!(quantize_ac(qmul + qadd - 4, qmul, qadd), 1); // 还在上半区
    }

    #[test]
    fn dc_quantize_rounds_to_nearest() {
        // dc_scale = 8：level = round(F/8)，而 F = 8×均值 → level == 均值。
        for mean in [0i32, 1, 127, 128, 200, 255] {
            assert_eq!(div_round(8 * mean, 8), mean);
        }
    }

    #[test]
    fn dc_prediction_tie_goes_to_left_not_top() {
        // MPEG-4 是 `|a-b| < |b-c|` → 取上；相等时取**左**。
        // 这和 MP43 的 `<=`（相等时取上）差一个等号，是最容易抄错的地方。
        let g = DcGrid::new(1, 1, 8, 8);
        let (_, dir) = g.predict(0, 0, 0);
        // 越界时 a = b = c = 1024 → |0| < |0| 为假 → 取左
        assert_eq!(dir, 0, "三者相等时应当取左");
    }

    #[test]
    fn dc_prediction_reads_left_and_top_neighbours() {
        let mut g = DcGrid::new(2, 1, 8, 8);
        // 手工放两个块：宏块 0 的亮度块 0 = level 100，亮度块 1 = level 200。
        // 网格存的是 `level * dc_scale`。
        g.update(100, 0, 0, 0);
        g.update(200, 1, 0, 0);
        // 宏块 1 的块 0：a = 左邻居（宏块 0 的块 1）= 1600，b = 越界 = 1024，c = 越界 = 1024。
        // |a-b| = 576，|b-c| = 0 → 576 < 0 为假 → 取**左**。
        let (pred, dir) = g.predict(0, 1, 0);
        assert_eq!(dir, 0, "差得远的邻居反而应当被选上（判据只看两个差谁更小）");
        assert_eq!(pred, 200, "1600/8 = 200");
    }

    #[test]
    fn dc_prediction_prefers_the_closer_of_left_and_top() {
        // 判据是 `|a-b| < |b-c|` → 取上（c）。所以「取上」要的是 **a 与 b 相近、
        // c 与 b 差得远**，而不是「c 和本块相近」—— 这是最容易理解反的一处。
        //
        // 预测宏块 (1,1) 的亮度块 0（格位 (2,2)）：
        //   a = 格位 (1,2) = 宏块 (0,1) 的块 1
        //   b = 格位 (1,1) = 宏块 (0,0) 的块 3
        //   c = 格位 (2,1) = 宏块 (1,0) 的块 2
        let mut g = DcGrid::new(2, 2, 8, 8);
        g.update(50, 1, 0, 1); // a
        g.update(50, 3, 0, 0); // b
        g.update(10, 2, 1, 0); // c
        let (pred, dir) = g.predict(0, 1, 1);
        assert_eq!(dir, 1, "a 与 b 相等、c 离得远 → 取上");
        assert_eq!(pred, 10, "取上就用 c 的 level");
    }

    #[test]
    fn ac_val_layout_matches_the_prediction_loops() {
        // 首列落在 `[1..8]`、首行落在 `[9..16]` —— 和 `apply_ac_pred` 的读法必须对上。
        let mut ac = AcGrid::new(1, 1);
        let mut block = [0i32; 64];
        for i in 1..8 {
            block[i * 8] = i as i32 * 3; // 首列
            block[i] = i as i32 * 5; // 首行
        }
        store_ac_val(&mut ac, 0, (0, 0), &block);
        let v = ac.get(0, 0, 0);
        for i in 1..8 {
            assert_eq!(v[i], (i as i32 * 3) as i16, "首列第 {i} 项");
            assert_eq!(v[8 + i], (i as i32 * 5) as i16, "首行第 {i} 项");
        }
    }

    #[test]
    fn out_of_frame_ac_neighbours_are_zero() {
        let ac = AcGrid::new(2, 2);
        assert_eq!(ac.neighbor(0, 0, (0, 0)), [0i16; 16], "左边越界");
        assert_eq!(ac.neighbor(0, 1, (0, 0)), [0i16; 16], "上边越界");
    }

    /// Cb 与 Cr 的格位相同（都是 `(mbx, mby)`），所以**必须各有一张表**。
    ///
    /// 曾经把它们合在一张里：编出来的位流照样「像那么回事」，ffmpeg 却在第 3 个宏块
    /// 起报 `Error at MB: 2`、PSNR 掉到 12 dB。根因就是 Cb 的预测值串到了 Cr 上。
    #[test]
    fn cb_and_cr_never_share_a_dc_cell() {
        let mut g = DcGrid::new(2, 1, 8, 8);
        g.update(90, 4, 0, 0); // Cb 宏块 0 = level 90
        g.update(170, 5, 0, 0); // Cr 宏块 0 = level 170
        assert_eq!(g.predict(4, 1, 0).0, 90, "Cb 读到的是 Cb");
        assert_eq!(g.predict(5, 1, 0).0, 170, "Cr 读到的是 Cr");
    }

    #[test]
    fn cb_and_cr_never_share_an_ac_cell() {
        let mut ac = AcGrid::new(2, 1);
        let mut cb = [0i32; 64];
        let mut cr = [0i32; 64];
        for i in 1..8 {
            cb[i * 8] = 7; // Cb 首列
            cr[i * 8] = -9; // Cr 首列
        }
        store_ac_val(&mut ac, 4, (0, 0), &cb);
        store_ac_val(&mut ac, 5, (0, 0), &cr);
        assert_eq!(ac.neighbor(4, 0, (1, 0))[1], 7, "Cb 的首列");
        assert_eq!(ac.neighbor(5, 0, (1, 0))[1], -9, "Cr 的首列");
    }

    #[test]
    fn applying_ac_prediction_subtracts_the_right_edge() {
        let mut block = [0i32; 64];
        for i in 1..8 {
            block[i * 8] = 10;
            block[i] = 20;
        }
        let mut nb = [0i16; 16];
        for i in 1..8 {
            nb[i] = i as i16;
            nb[8 + i] = 100 + i as i16;
        }
        let mut left = block;
        apply_ac_pred(&mut left, &nb, 0);
        for i in 1..8 {
            assert_eq!(left[i * 8], 10 - i as i32, "取左：减首列");
            assert_eq!(left[i], 20, "取左：首行不动");
        }
        let mut top = block;
        apply_ac_pred(&mut top, &nb, 1);
        for i in 1..8 {
            assert_eq!(top[i * 8], 10, "取上：首列不动");
            assert_eq!(top[i], 20 - (100 + i as i32), "取上：减首行");
        }
    }

    #[test]
    fn write_dc_matches_the_bit_length_vlc() {
        // VLC 的**符号是位长**，所以总位数 = VLC 长度 + 位长（+ size > 8 时的一个 marker）。
        // 亮度表：`DC_LUM[0..4] = (3,3) (3,2) (2,2) (2,3) (1,3)`，之后一路 `(1, size+1)`。
        for (level, bits) in [
            (0i32, 3u32), // size 0 → VLC 3 位
            (1, 3),       // size 1 → VLC 2 位 + 1
            (-1, 3),
            (3, 4),    // size 2 → VLC 2 位 + 2
            (255, 15), // size 8 → VLC 7 位 + 8（size 不 > 8，没有 marker）
            (256, 18), // size 9 → VLC 8 位 + 9 + marker
            (511, 18),
            (512, 20), // size 10 → VLC 9 位 + 10 + marker
        ] {
            let mut w = BitWriter::new();
            write_dc(&mut w, level, true).unwrap();
            assert_eq!(w.bit_pos() as u32, bits, "亮度 level {level}");
        }
        // 色度表短一点：`DC_CHROM[0] = (3,2)`。
        let mut w = BitWriter::new();
        write_dc(&mut w, 0, false).unwrap();
        assert_eq!(w.bit_pos(), 2, "色度 level 0");
    }

    #[test]
    fn write_dc_uses_xbits_negation() {
        // -1 在 size 1 下写成 0；-2 在 size 2 下写成 1；-3 在 size 2 下写成 0；3 照写。
        for (level, size, payload) in [(-1i32, 1u32, 0u32), (-2, 2, 1), (-3, 2, 0), (3, 2, 3)] {
            let mut w = BitWriter::new();
            write_dc(&mut w, level, true).unwrap();
            let nbits = w.bit_pos() as u32;
            let bytes = w.finish();
            let mut all = 0u32;
            for &x in &bytes {
                all = all << 8 | u32::from(x);
            }
            let total = bytes.len() as u32 * 8;
            // 载荷是「已写位」里的最后 size 位。
            let got = (all >> (total - nbits)) & ((1 << size) - 1);
            assert_eq!(got, payload, "level {level} 的载荷");
        }
    }

    #[test]
    fn escape3_costs_the_escape_code_plus_twenty_three_bits() {
        // 帧内表的 escape 码是 `0000011`（7 位，`ff_mpeg4_intra_vlc[102] = (3, 7)`），
        // 所以 ESC3 一共 7 + 2 + 1 + 6 + 1 + 12 + 1 = 30 位。
        // 用一个大 level 逼出 ESC3（12 位能表示 ±2047，但表里没有这么大的 level）。
        let rl = intra_rl_table();
        let mut w = BitWriter::new();
        write_rl(&mut w, &rl, true, 30, 2000).unwrap();
        assert_eq!(w.bit_pos(), 30);
    }

    #[test]
    fn short_levels_avoid_the_escape() {
        // (last=0, run=0, level=1) 在表里，应当只花 ESC0 的位数。
        let rl = intra_rl_table();
        let mut w = BitWriter::new();
        write_rl(&mut w, &rl, false, 0, 1).unwrap();
        assert!(w.bit_pos() <= 4, "最常用的系数不该走 escape");
    }

    #[test]
    fn a_flat_frame_produces_only_dc_and_a_zero_cbp() {
        // 纯色帧：6 个块的 AC 全 0 → cbp = 0 → 每块只有 DC。
        // 参考样本的第一帧正是这样（`cbpc=0`、`cbpy=0`）。
        let mut e = Encoder::new(EncoderConfig {
            width: 16,
            height: 16,
            ..Default::default()
        })
        .unwrap();
        let data = e.encode_i_frame(&flat_frame(16, 16, 128, 128)).unwrap();
        // VOP 起始码必须在
        assert!(data.windows(4).any(|x| x == [0x00, 0x00, 0x01, 0xB6]));
        // 只有 VOP 头 + 1 个宏块，应该很短
        assert!(data.len() < 32, "纯色宏块不该有 {} 字节", data.len());
    }

    #[test]
    fn encoder_rejects_bad_dimensions() {
        let cfg = EncoderConfig {
            width: 15,
            height: 16,
            ..Default::default()
        };
        assert!(Encoder::new(cfg).is_err());
        let cfg = EncoderConfig {
            width: 16,
            height: 0,
            ..Default::default()
        };
        assert!(Encoder::new(cfg).is_err());
    }

    #[test]
    fn decoder_specific_info_carries_the_vos_header() {
        let e = Encoder::new(EncoderConfig {
            width: 320,
            height: 240,
            fps_num: 25,
            ..Default::default()
        })
        .unwrap();
        let dsi = e.decoder_specific_info();
        assert_eq!(dsi, e.elementary_stream_header(), "两处必须是同一份字节");
        // 少了 VOS 那两级，`profile` / `level` 就报不出来。
        assert_eq!(
            &dsi[..4],
            &[0x00, 0x00, 0x01, 0xB0],
            "DSI 必须以 VOS 起始码开头"
        );
        assert_eq!(dsi[4], PROFILE_SIMPLE_LEVEL1);
        assert_eq!(
            &dsi[5..9],
            &[0x00, 0x00, 0x01, 0xB5],
            "接着是视觉对象起始码"
        );
        // 参考样本的 DSI 头 10 字节（`refv.mp4`）：
        //   `00 00 01 b0 01 00 00 01 b5 89 13`
        assert_eq!(
            &dsi[..11],
            &[0x00, 0x00, 0x01, 0xB0, 0x01, 0x00, 0x00, 0x01, 0xB5, 0x89, 0x13]
        );
    }
}
