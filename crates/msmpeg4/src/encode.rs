//! MS-MPEG-4 v3（MP43）**编码器**。
//!
//! 目前只做 **I 帧**（帧内）—— 那是「能出一条 ffmpeg 解得开的流」的最小完整切片。
//! P 帧（运动估计 + 帧间残差）还没做。
//!
//! 位流按 MS-MPEG-4 v3 的语法写：帧头 + 帧内宏块类型 + DC 一趟 + AC 的 RL 编码。
//! **熵编码必须严格按语法**（差一位 ffmpeg 就解不开 —— 这也是最硬的验证），
//! 但**量化器不必**：码流只需要「合法且质量好」，不必和别的编码器编出的字节相同。
//!
//! ## 两个「必须知道」的点
//!
//! - **DC 走单独一趟**：`level = round(F(0,0) / dc_scale)`，且要减去邻居预测
//!   （`dc_predict`，和我们解码端同一个函数）。AC 则走 RL 表。
//! - **level 的 8 位上限**：第三 escape 用 8 位有符号写 level，所以 |level| ≤ 127。
//!   `qscale` 会自动抬高来保证这一点（见 `pick_qscale`）。

use crate::bitwriter::BitWriter;
use crate::block::RlTable;
use crate::error::{Error, Result};
use crate::fdct::{fdct, fdct_i};
use crate::frame::{
    mc_chroma8, mc_luma16, mc_mb, mc_params, pred_motion, Frame, Planes, ZIGZAG,
};
use crate::idct::{idct_add, idct_put};
use crate::grid::Grid;
use crate::predict::{coded_block_predict, dc_predict};
use crate::tables;
use crate::vlc::Vlc;

/// `DC_MAX`：DC VLC 的 escape 符号（超过它就用 8 位明文）。
const DC_MAX: i32 = 119;
/// 第三 escape 的 level 是 8 位有符号，所以量化后的 level 必须落在这个范围里。
const MAX_ESCAPE_LEVEL: i32 = 127;
/// MV VLC 的真值个数；这个下标就是 escape（差分要用 6+6 位明文补）。
const MV_ESCAPE: usize = 1099;
/// 运动矢量差分相对预测值的可表示范围（模 64 回绕之前）。
const MV_DIFF_MAX: i32 = 31;

/// 一帧编码的配置。默认值取 V3 最常用的那几个下标。
#[derive(Debug, Clone, Copy)]
pub struct EncoderConfig {
    /// 目标 qscale（1..=31）。太小会被 `pick_qscale` 自动抬高。
    pub qscale: u8,
    /// 帧内亮度 RL 表下标（0..=2）。
    pub rl_table_index: u8,
    /// 帧内色度 RL 表下标（0..=2）。
    pub rl_chroma_table_index: u8,
    /// DC 表下标（0/1）。
    pub dc_table_index: u8,
    /// MV 表下标（0/1）。
    pub mv_table_index: u8,
    /// P 帧要不要写「skip 宏块」标志位（ffmpeg 的编码器恒写 1）。
    pub use_skip_mb_code: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            qscale: 8,
            rl_table_index: 0,
            rl_chroma_table_index: 0,
            dc_table_index: 0,
            mv_table_index: 1,
            use_skip_mb_code: true,
        }
    }
}

/// MP43 编码器。有状态：DC 预测与编码块图都要跨宏块累积（和编码器对称）。
pub struct Encoder {
    cfg: EncoderConfig,
    /// DC 预测器（存的是 `level * dc_scale`，与解码端同口径）。
    dc_luma: Grid<i16>,
    dc_cb: Grid<i16>,
    dc_cr: Grid<i16>,
    /// 编码块图（`val`，预测前的真值）。
    coded: Grid<u8>,
    /// MV 预测网格（P 帧用，边框 0）。
    mv_grid: Grid<(i32, i32)>,
    /// 上一帧的**重建**（P 帧运动补偿的参考）。`None` = 还没编过 I 帧。
    ref_planes: Option<Planes>,
    /// 上一帧实际用的 qscale（`pick_qscale` 抬高过的话会不一样）。
    pub last_qscale: u8,
}

impl Encoder {
    pub fn new(cfg: EncoderConfig) -> Self {
        Self {
            cfg,
            dc_luma: Grid::with_init(1, 1, 1024),
            dc_cb: Grid::with_init(1, 1, 1024),
            dc_cr: Grid::with_init(1, 1, 1024),
            coded: Grid::new(1, 1),
            mv_grid: Grid::new(1, 1),
            ref_planes: None,
            last_qscale: cfg.qscale,
        }
    }

    /// 把一帧编成 MP43 的**一个** I 帧位流。
    ///
    /// `frame` 是裁剪过的 YUV420P；内部会按宏块补齐（边缘复制）再编。
    pub fn encode_i_frame(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        let w = frame.width as usize;
        let h = frame.height as usize;
        if w == 0 || h == 0 || w % 2 != 0 || h % 2 != 0 {
            return Err(Error::Bad("尺寸必须是正的偶数（YUV420P）"));
        }
        let mb_w = w.div_ceil(16);
        let mb_h = h.div_ceil(16);
        let (pw, ph) = (mb_w * 16, mb_h * 16);

        // ---- 补齐到宏块对齐（边缘复制，和 MC 的边缘处理同思路）----
        let (y, u, v) = pad_planes(frame, pw, ph);

        // ---- 逐块正向 DCT（qscale 要等看完系数极值才能定，所以先存下来）----
        let mut coeffs: Vec<[i32; 64]> = Vec::with_capacity(mb_w * mb_h * 6);
        let mut max_ac = 0i32;
        for mby in 0..mb_h {
            for mbx in 0..mb_w {
                for n in 0..6usize {
                    let (plane, stride, px, py) = block_at(&y, &u, &v, pw, mbx, mby, n);
                    let mut c = [0i32; 64];
                    fdct(&plane[py * stride + px..], stride, &mut c);
                    for k in 1..64 {
                        max_ac = max_ac.max(c[k].abs());
                    }
                    coeffs.push(c);
                }
            }
        }

        // ---- 定 qscale：保证量化后的 |level| 不会超出 8 位 escape ----
        let qscale = pick_qscale(self.cfg.qscale, max_ac);
        self.last_qscale = qscale;
        let q = qscale as i32;
        let qmul = q << 1;
        let qadd = (q - 1) | 1;
        let y_dc_scale = tables::OLD_Y_DC_SCALE[qscale as usize] as i32;
        let c_dc_scale = tables::C_DC_SCALE[qscale as usize] as i32;

        // ---- 帧头（`ff_msmpeg4_encode_picture_header` 的 I 帧分支）----
        let mut wtr = BitWriter::new();
        wtr.put(0, 2); // pict_type - 1 = 0 → I
        wtr.put(qscale as u32, 5);
        // slice_code：slice_height == mb_height 时是 0x17（一个 slice 覆盖整帧）
        wtr.put(0x17, 5);
        code012(&mut wtr, self.cfg.rl_chroma_table_index);
        code012(&mut wtr, self.cfg.rl_table_index);
        wtr.put(self.cfg.dc_table_index as u32, 1);

        // ---- 表 ----
        let rl_luma = rl_table(self.cfg.rl_table_index as usize)?;
        let rl_chroma = rl_table(3 + self.cfg.rl_chroma_table_index as usize)?;
        let dc_vlc = Vlc::new(&tables::DC[self.cfg.dc_table_index as usize][0]);
        let dc_vlc_c = Vlc::new(&tables::DC[self.cfg.dc_table_index as usize][1]);
        let mb_intra_vlc = Vlc::new(&tables::MB_INTRA);

        self.dc_luma = Grid::with_init(2 * mb_w, 2 * mb_h, 1024);
        self.dc_cb = Grid::with_init(mb_w, mb_h, 1024);
        self.dc_cr = Grid::with_init(mb_w, mb_h, 1024);
        self.coded = Grid::new(2 * mb_w, 2 * mb_h);

        // 重建缓冲：P 帧运动补偿参考的是**重建后**的帧，不是原图 ——
        // 有损编码里解码端只能拿到重建值，参考它才不会漂移。
        let mut cur = Planes::new(pw, ph);

        // slice_height = mb_height → 只有第 0 行是片首行
        for mby in 0..mb_h {
            let first_slice_line = mby == 0;
            for mbx in 0..mb_w {
                let base = (mby * mb_w + mbx) * 6;

                // ---- 先量化，才知道 cbp ----
                let mut qb = [[0i32; 64]; 6];
                let mut cbp = 0u32;
                for n in 0..6usize {
                    let luma = n < 4;
                    let scale = if luma { y_dc_scale } else { c_dc_scale };
                    let c = &coeffs[base + n];
                    let mut blk = [0i32; 64];
                    // DC 单独：level = round(F(0,0) / dc_scale)
                    blk[0] = div_round(c[0], scale);
                    // AC：反量化是 `level*qmul ± qadd`，反过来取最接近的那个 level
                    let mut has_ac = false;
                    for k in 1..64 {
                        let l = quantize(c[k], qmul, qadd);
                        let l = l.clamp(-MAX_ESCAPE_LEVEL, MAX_ESCAPE_LEVEL);
                        blk[k] = l;
                        if l != 0 {
                            has_ac = true;
                        }
                    }
                    qb[n] = blk;
                    // cbp 的 bit n = 这个块**有 AC**（DC 无论如何都要传）
                    if has_ac {
                        cbp |= 1 << (5 - n);
                    }
                }

                // ---- MB 类型：帧内 + CBP 预测 ----
                let mut coded_cbp = cbp & 0x3;
                for i in 0..4usize {
                    let gx = 2 * mbx + (i & 1);
                    let gy = 2 * mby + (i >> 1);
                    let xy = self.coded.idx(gx, gy);
                    let a = self.coded.data[xy - 1];
                    let b = self.coded.data[xy - 1 - self.coded.w];
                    let c = self.coded.data[xy - self.coded.w];
                    let pred = coded_block_predict(a, b, c);
                    let val = (cbp >> (5 - i)) & 1;
                    self.coded.data[xy] = val as u8;
                    coded_cbp |= (val ^ pred as u32) << (5 - i);
                }
                let (code, len) = mb_intra_vlc
                    .code_len(coded_cbp as usize)
                    .ok_or(Error::Bad("CBP 超出帧内 MB 类型表"))?;
                wtr.put(code, len);
                wtr.put(0, 1); // ac_pred = 0（我们不做 AC 预测）

                // ---- 6 个块 ----
                for n in 0..6usize {
                    let luma = n < 4;
                    let scale = if luma { y_dc_scale } else { c_dc_scale };
                    let (dc_grid, dcv) = if luma {
                        (&mut self.dc_luma, &dc_vlc)
                    } else if n == 4 {
                        (&mut self.dc_cb, &dc_vlc_c)
                    } else {
                        (&mut self.dc_cr, &dc_vlc_c)
                    };
                    let (gx, gy) = if luma {
                        (2 * mbx + (n & 1), 2 * mby + (n >> 1))
                    } else {
                        (mbx, mby)
                    };
                    let xy = dc_grid.idx(gx, gy);

                    // DC 预测（与解码端同一个 dc_predict）
                    let mut a = dc_grid.data[xy - 1] as i32;
                    let mut b = dc_grid.data[xy - 1 - dc_grid.w] as i32;
                    let mut c = dc_grid.data[xy - dc_grid.w] as i32;
                    if first_slice_line && (n & 2) == 0 {
                        b = 1024;
                        c = 1024;
                    }
                    // ⚠️ 这里必须用**截断**除法（Rust 的 `/`），和解码端
                    // `(a + scale/2) / scale` 逐位一致；四舍五入会让预测值偏掉。
                    a = (a + scale / 2) / scale;
                    b = (b + scale / 2) / scale;
                    c = (c + scale / 2) / scale;
                    let (pred, _dir) = dc_predict(a, b, c);

                    let total = qb[n][0];
                    dc_grid.data[xy] = (total * scale) as i16;
                    write_dc(&mut wtr, dcv, total - pred);

                    // AC
                    let rl = if luma { &rl_luma } else { &rl_chroma };
                    write_ac(&mut wtr, rl, &qb[n], true, 0);

                    // ---- 重建（P 帧的参考）----
                    // 帧内反量化：DC 乘 dc_scale，AC 是 level*qmul ± qadd，然后 idct_put。
                    let (plane, stride, _px, _py) = recon_at(&mut cur, n, mbx, mby);
                    let mut rb = [0i16; 64];
                    rb[0] = (total * scale) as i16;
                    for k in 1..64 {
                        rb[k] = dequant(qb[n][k], qmul, qadd);
                    }
                    idct_put(&mut rb, plane, stride);
                }
            }
        }

        self.ref_planes = Some(cur);
        Ok(wtr.finish())
    }

    // ==================== P 帧 ====================

    /// 编一帧：第一帧是 I，之后都是 P。
    pub fn encode_frame(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        if self.ref_planes.is_none() {
            self.encode_i_frame(frame)
        } else {
            self.encode_p_frame(frame)
        }
    }

    /// 编一帧 **P 帧**（帧间）。必须先编过一帧 I。
    ///
    /// 帧间宏块一律按 inter 编（不做「这个宏块改用帧内」的判决）；
    /// 残差为零且运动矢量为零时退化成 skip。
    ///
    /// ⚠️ 半像素取整：P 帧用不用带舍入的插值由 `flipflop_rounding` 决定，而它只从
    /// I 帧尾部的 ext header 读。我们（与 ffmpeg 的编码器一样）**不写** ext header，
    /// 所以解码端恒得 `flipflop_rounding = 0` → P 帧恒用**带舍入**的插值。
    pub fn encode_p_frame(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        let w = frame.width as usize;
        let h = frame.height as usize;
        let mb_w = w.div_ceil(16);
        let mb_h = h.div_ceil(16);
        let (pw, ph) = (mb_w * 16, mb_h * 16);

        let refp = self
            .ref_planes
            .as_ref()
            .ok_or(Error::Bad("P 帧需要参考帧（先编一帧 I）"))?;
        if refp.pw != pw || refp.ph != ph {
            return Err(Error::Bad("参考帧尺寸与当前帧不一致"));
        }

        let (y, u, v) = pad_planes(frame, pw, ph);

        let q = self.cfg.qscale as i32;
        let qmul = q << 1;
        let qadd = (q - 1) | 1;

        let rl_inter = rl_table(3 + self.cfg.rl_table_index as usize)?;
        let mb_non_intra_vlc = Vlc::new(&tables::MB_NON_INTRA);
        let mv_tab = MvIndex::new(self.cfg.mv_table_index as usize);

        // P 帧恒用带舍入的插值，理由见上面的函数注释。
        let no_rounding = false;

        let mut cur = Planes::new(pw, ph);
        self.mv_grid = Grid::new(mb_w, mb_h);

        let mut wtr = BitWriter::new();
        wtr.put(1, 2); // pict_type - 1 = 1 → P
        wtr.put(self.cfg.qscale as u32, 5);
        wtr.put(u32::from(self.cfg.use_skip_mb_code), 1);
        code012(&mut wtr, self.cfg.rl_table_index);
        wtr.put(self.cfg.dc_table_index as u32, 1);
        wtr.put(self.cfg.mv_table_index as u32, 1);

        for mby in 0..mb_h {
            let first_slice_line = mby == 0;
            for mbx in 0..mb_w {
                // ---- 运动估计 + MC 预测 ----
                let pred = pred_motion(&self.mv_grid, mbx, mby, first_slice_line);
                let mv = motion_estimate(refp, &y, pw, mbx, mby, pred, no_rounding);
                let p = mc_params(mbx, mby, mv, pw, ph);
                let mut pred16 = [0u8; 256];
                mc_luma16(&mut pred16, 16, 0, 0, refp, &p, no_rounding);
                let mut predu = [0u8; 64];
                let mut predv = [0u8; 64];
                mc_chroma8(&mut predu, 8, 0, 0, refp, 0, &p, no_rounding);
                mc_chroma8(&mut predv, 8, 0, 0, refp, 1, &p, no_rounding);

                // ---- 残差 → 正向 DCT → 量化 → cbp ----
                let mut qb = [[0i32; 64]; 6];
                let mut cbp = 0u32;
                for n in 0..6usize {
                    let mut res = [0i32; 64];
                    if n < 4 {
                        let ox = mbx * 16 + (n & 1) * 8;
                        let oy = mby * 16 + (n >> 1) * 8;
                        for r in 0..8usize {
                            for c in 0..8usize {
                                let s = y[(oy + r) * pw + ox + c] as i32;
                                let d = pred16[((n >> 1) * 8 + r) * 16 + (n & 1) * 8 + c] as i32;
                                res[r * 8 + c] = s - d;
                            }
                        }
                    } else {
                        let cw = pw / 2;
                        let (plane, prd) = if n == 4 { (&u, &predu) } else { (&v, &predv) };
                        for r in 0..8usize {
                            for c in 0..8usize {
                                let s = plane[(mby * 8 + r) * cw + mbx * 8 + c] as i32;
                                res[r * 8 + c] = s - prd[r * 8 + c] as i32;
                            }
                        }
                    }
                    let mut coef = [0i32; 64];
                    fdct_i(&res, 8, &mut coef);
                    let mut blk = [0i32; 64];
                    let mut has = false;
                    for k in 0..64usize {
                        let l = quantize(coef[k], qmul, qadd)
                            .clamp(-MAX_ESCAPE_LEVEL, MAX_ESCAPE_LEVEL);
                        blk[k] = l;
                        if l != 0 {
                            has = true;
                        }
                    }
                    qb[n] = blk;
                    // 帧间的 cbp：**有任何系数**（含 DC）就置位 —— 帧间没有单独的 DC 一趟
                    if has {
                        cbp |= 1 << (5 - n);
                    }
                }

                // ---- skip：没有残差且不动 ----
                if self.cfg.use_skip_mb_code && cbp == 0 && mv == (0, 0) {
                    wtr.put(1, 1);
                    let g = self.mv_grid.idx(mbx, mby);
                    self.mv_grid.data[g] = (0, 0);
                    mc_mb(&mut cur, refp, mbx, mby, mv, no_rounding);
                    continue;
                }

                // ---- MB 类型 + 运动矢量 ----
                if self.cfg.use_skip_mb_code {
                    wtr.put(0, 1); // mb coded
                }
                let (code, len) = mb_non_intra_vlc
                    .code_len((cbp + 64) as usize)
                    .ok_or(Error::Bad("CBP 超出帧间 MB 类型表"))?;
                wtr.put(code, len);

                let g = self.mv_grid.idx(mbx, mby);
                self.mv_grid.data[g] = mv;
                mv_tab.write(&mut wtr, mv.0 - pred.0, mv.1 - pred.1)?;

                // ---- 6 个块（帧间：DC 也是普通系数，run 从 0 起）----
                for n in 0..6usize {
                    write_ac(&mut wtr, &rl_inter, &qb[n], false, 1);
                }

                // ---- 重建：先 MC 铺底，再叠回反量化后的残差 ----
                mc_mb(&mut cur, refp, mbx, mby, mv, no_rounding);
                for n in 0..6usize {
                    let (plane, stride, _px, _py) = recon_at(&mut cur, n, mbx, mby);
                    let mut rb = [0i16; 64];
                    for k in 0..64usize {
                        rb[k] = dequant(qb[n][k], qmul, qadd);
                    }
                    idct_add(&mut rb, plane, stride);
                }
            }
        }

        self.ref_planes = Some(cur);
        Ok(wtr.finish())
    }
}

/// 当前帧**重建**平面上第 `n` 个块的 `(平面切片, 行距, x, y)`。
fn recon_at(cur: &mut Planes, n: usize, mbx: usize, mby: usize) -> (&mut [u8], usize, usize, usize) {
    let pw = cur.pw;
    if n < 4 {
        let px = mbx * 16 + (n & 1) * 8;
        let py = mby * 16 + (n >> 1) * 8;
        let at = py * pw + px;
        (&mut cur.y[at..], pw, px, py)
    } else if n == 4 {
        let at = mby * 8 * (pw / 2) + mbx * 8;
        (&mut cur.u[at..], pw / 2, mbx * 8, mby * 8)
    } else {
        let at = mby * 8 * (pw / 2) + mbx * 8;
        (&mut cur.v[at..], pw / 2, mbx * 8, mby * 8)
    }
}

/// 反量化（帧内 AC 与帧间通用）：`level > 0 → level*qmul + qadd`，反之减。
fn dequant(level: i32, qmul: i32, qadd: i32) -> i16 {
    if level > 0 {
        (level * qmul + qadd) as i16
    } else if level < 0 {
        (level * qmul - qadd) as i16
    } else {
        0
    }
}

/// 运动矢量的反查表：`(dx+32) << 6 | (dy+32)` → 符号下标。
///
/// 解码表给的是「符号 → (x, y) 偏移（偏 +32）」，编码要反过来：把 1099 个真值
/// 铺进一张 64×64 的表，没铺到的格子就是 escape（差分用 6+6 位明文补）。
struct MvIndex {
    table: Vec<u16>,
    vlc: Vlc,
}

impl MvIndex {
    fn new(mv_table_index: usize) -> Self {
        let (code, len, xs, ys) = if mv_table_index == 0 {
            (
                &tables::MV0_CODE[..],
                &tables::MV0_LEN[..],
                &tables::MV0_X[..],
                &tables::MV0_Y[..],
            )
        } else {
            (
                &tables::MV1_CODE[..],
                &tables::MV1_LEN[..],
                &tables::MV1_X[..],
                &tables::MV1_Y[..],
            )
        };
        let mut table = vec![MV_ESCAPE as u16; 64 * 64];
        for i in 0..xs.len() {
            table[(xs[i] as usize) << 6 | ys[i] as usize] = i as u16;
        }
        let pairs: Vec<[u32; 2]> = code
            .iter()
            .zip(len.iter())
            .map(|(&c, &l)| [c as u32, l as u32])
            .collect();
        Self {
            table,
            vlc: Vlc::new(&pairs),
        }
    }

    /// 写一个运动矢量**差分**（相对预测值，半像素单位）。
    ///
    /// 先按这个格式的怪规矩回绕（不是真正的模 64），再加 32 落进 `[0,63]` 查表；
    /// 查不到就是 escape，后面明文写 6+6 位。
    fn write(&self, w: &mut BitWriter, dx: i32, dy: i32) -> Result<()> {
        let mut mx = dx;
        let mut my = dy;
        if mx <= -64 {
            mx += 64;
        } else if mx >= 64 {
            mx -= 64;
        }
        if my <= -64 {
            my += 64;
        } else if my >= 64 {
            my -= 64;
        }
        let mx = mx + 32;
        let my = my + 32;
        if !(0..64).contains(&mx) || !(0..64).contains(&my) {
            return Err(Error::Bad("运动矢量差分超出可表示范围"));
        }
        let code = self.table[(mx as usize) << 6 | my as usize] as usize;
        let (c, l) = self
            .vlc
            .code_len(code)
            .ok_or(Error::Bad("MV 符号下标越界"))?;
        w.put(c, l);
        if code == MV_ESCAPE {
            w.put(mx as u32, 6);
            w.put(my as u32, 6);
        }
        Ok(())
    }
}

/// 运动估计：整像素方形粗搜（围绕 (0,0) 与预测的整像素位置各 ±4）+ 半像素细化。
///
/// SAD 一律「先把预测块搬进临时缓冲再比」—— 这样估计用的插值与真正 MC 的
/// **完全一致**，残差才不会对得上号却解出别的东西。
/// 返回半像素单位的 MV，并保证它与预测的差能被 MV 编码表示。
fn motion_estimate(
    refp: &Planes,
    cur_y: &[u8],
    cstride: usize,
    mbx: usize,
    mby: usize,
    pred: (i32, i32),
    no_rounding: bool,
) -> (i32, i32) {
    let (pw, ph) = (refp.pw, refp.ph);
    let ox = mbx * 16;
    let oy = mby * 16;

    let mut cur = [0u8; 256];
    for r in 0..16usize {
        for c in 0..16usize {
            cur[r * 16 + c] = cur_y[(oy + r) * cstride + ox + c];
        }
    }
    let mut tmp = [0u8; 256];
    let mut sad_of = |mv: (i32, i32)| -> u64 {
        let p = mc_params(mbx, mby, mv, pw, ph);
        mc_luma16(&mut tmp, 16, 0, 0, refp, &p, no_rounding);
        let mut acc = 0u64;
        for i in 0..256usize {
            acc += (cur[i] as i32 - tmp[i] as i32).unsigned_abs() as u64;
        }
        acc
    };

    let mut best = (0i32, 0i32);
    let mut best_sad = sad_of(best);
    let mut seen = std::collections::HashSet::new();
    seen.insert(best);
    // 整像素粗搜：两个中心各 ±4（预测的位置通常就在附近，(0,0) 覆盖静止场景）
    for (cx, cy) in [(0i32, 0i32), (pred.0 >> 1, pred.1 >> 1)] {
        for dy in -4..=4i32 {
            for dx in -4..=4i32 {
                let cand = (cx + dx, cy + dy);
                if !seen.insert(cand) {
                    continue;
                }
                let s = sad_of(cand);
                if s < best_sad {
                    best_sad = s;
                    best = cand;
                }
            }
        }
    }
    // 半像素细化：最优整像素周围 ±1 半像素
    let base = best;
    for dy in -1..=1i32 {
        for dx in -1..=1i32 {
            let cand = (base.0 + dx, base.1 + dy);
            let s = sad_of(cand);
            if s < best_sad {
                best_sad = s;
                best = cand;
            }
        }
    }

    // 限制与预测的差，保证能被 MV 编码表示（模 64 回绕之前的可表示范围）
    let clamp = |v: i32, p: i32| -> i32 { (v - p).clamp(-MV_DIFF_MAX, MV_DIFF_MAX) + p };
    (clamp(best.0, pred.0), clamp(best.1, pred.1))
}

/// 补齐到宏块对齐的 YUV420P 平面（边缘复制）。
fn pad_planes(frame: &Frame, pw: usize, ph: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let pad = |src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize| {
        let mut out = vec![0u8; dw * dh];
        for r in 0..dh {
            let sr = r.min(sh - 1);
            for c in 0..dw {
                out[r * dw + c] = src[sr * sw + c.min(sw - 1)];
            }
        }
        out
    };
    let y = pad(&frame.y, w, h, pw, ph);
    let u = pad(&frame.u, w / 2, h / 2, pw / 2, ph / 2);
    let v = pad(&frame.v, w / 2, h / 2, pw / 2, ph / 2);
    (y, u, v)
}

/// 宏块里第 `n` 个块在平面上的 `(平面, 行距, x, y)`。
#[allow(clippy::type_complexity)]
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
        (y, pw, mbx * 16 + (n & 1) * 8, mby * 16 + (n >> 1) * 8)
    } else if n == 4 {
        (u, pw / 2, mbx * 8, mby * 8)
    } else {
        (v, pw / 2, mbx * 8, mby * 8)
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
fn quantize(f: i32, qmul: i32, qadd: i32) -> i32 {
    if f > 0 {
        div_round(f - qadd, qmul).max(0)
    } else if f < 0 {
        div_round(f + qadd, qmul).min(0)
    } else {
        0
    }
}

/// 保证量化后的 AC level 不超 8 位 escape：`|level| ≈ |F| / (2*qscale) ≤ 127`。
fn pick_qscale(requested: u8, max_ac: i32) -> u8 {
    let need = (max_ac as f64 / (2.0 * MAX_ESCAPE_LEVEL as f64)).ceil() as i64;
    let q = (requested as i64).max(need).clamp(1, 31);
    q as u8
}

/// `ff_msmpeg4_code012`：0 → `0`；1 → `10`；2 → `11`。
fn code012(w: &mut BitWriter, n: u8) {
    if n == 0 {
        w.put(0, 1);
    } else {
        w.put(1, 1);
        w.put(u32::from(n >= 2), 1);
    }
}

/// `msmpeg4_encode_dc`：符号 + 可选 escape + 符号位。
fn write_dc(w: &mut BitWriter, dc_vlc: &Vlc, level: i32) {
    let sign = u32::from(level < 0);
    let a = level.abs();
    let code = a.min(DC_MAX);
    if let Some((c, n)) = dc_vlc.code_len(code as usize) {
        w.put(c, n);
    }
    if code == DC_MAX {
        w.put(a as u32, 8);
    }
    if a != 0 {
        w.put(sign, 1);
    }
}

/// 一个块的 AC 部分：三种 escape 按语法逐位写。
fn write_ac(w: &mut BitWriter, rl: &RlTable, block: &[i32; 64], intra: bool, run_diff: i32) {
    let start = if intra { 1 } else { 0 };
    let last_index = match (start..64).rev().find(|&i| block[ZIGZAG[i] as usize] != 0) {
        Some(i) => i,
        None => return,
    };
    let esc = rl.escape_symbol();
    let mut last_non_zero = start as i32 - 1;

    for i in start..=last_index {
        let level = block[ZIGZAG[i] as usize];
        if level == 0 {
            continue;
        }
        let run = (i as i32 - last_non_zero - 1) as usize;
        let last = i == last_index;
        let sign = u32::from(level < 0);
        let lvl = level.abs();

        let code = rl.index_of(usize::from(last), run, lvl);
        if code != esc {
            if let Some((c, n)) = rl.code_len(code) {
                w.put(c, n);
            }
            w.put(sign, 1);
        } else {
            // 先写 escape 码
            if let Some((c, n)) = rl.code_len(esc) {
                w.put(c, n);
            }
            // 第一 escape：level 减掉该 (last,run) 的上界后还能在表里找到
            let level1 = lvl - rl.max_level_at(usize::from(last), run);
            let mut done = false;
            if level1 >= 1 {
                let c1 = rl.index_of(usize::from(last), run, level1);
                if c1 != esc {
                    w.put(1, 1);
                    if let Some((c, n)) = rl.code_len(c1) {
                        w.put(c, n);
                    }
                    w.put(sign, 1);
                    done = true;
                }
            }
            if !done {
                w.put(0, 1); // escape 的第一位 = 0
                // 第二 escape：run 减掉该 (last,level) 的上界后还能在表里找到
                let mut second = None;
                if lvl <= 63 {
                    let run1 = run as i32 - rl.max_run_at(usize::from(last), lvl) - run_diff;
                    if run1 >= 0 {
                        let c2 = rl.index_of(usize::from(last), run1 as usize, lvl);
                        if c2 != esc {
                            second = Some(c2);
                        }
                    }
                }
                match second {
                    Some(c2) => {
                        w.put(1, 1);
                        if let Some((c, n)) = rl.code_len(c2) {
                            w.put(c, n);
                        }
                        w.put(sign, 1);
                    }
                    None => {
                        // 第三 escape：last(1) + run(6) + level(8 位有符号)
                        w.put(0, 1);
                        w.put(u32::from(last), 1);
                        w.put(run as u32, 6);
                        w.put_signed(level, 8);
                    }
                }
            }
        }
        last_non_zero = i as i32;
    }
}

/// 和解码端同一张表（`ff_rl_table` 的 6 张）。
fn rl_table(index: usize) -> Result<RlTable> {
    Ok(match index {
        0 => RlTable::new(&tables::RL0_VLC, &tables::RL0_RUN, &tables::RL0_LEVEL, 85),
        1 => RlTable::new(&tables::RL2_VLC, &tables::RL2_RUN, &tables::RL2_LEVEL, 119),
        2 => RlTable::new(
            &tables::RL_INTRA_VLC,
            &tables::RL_INTRA_RUN,
            &tables::RL_INTRA_LEVEL,
            67,
        ),
        3 => RlTable::new(&tables::RL1_VLC, &tables::RL1_RUN, &tables::RL1_LEVEL, 81),
        4 => RlTable::new(&tables::RL4_VLC, &tables::RL4_RUN, &tables::RL4_LEVEL, 99),
        5 => RlTable::new(
            &tables::RL_INTER_VLC,
            &tables::RL_INTER_RUN,
            &tables::RL_INTER_LEVEL,
            58,
        ),
        _ => return Err(Error::Unsupported(format!("RL 表下标 {index}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_inverts_the_decoder_dequant() {
        // 解码：level>0 → level*qmul+qadd；level<0 → level*qmul-qadd；level==0 → 0
        for q in 1..=31i32 {
            let (qmul, qadd) = (q << 1, (q - 1) | 1);
            for f in -300..=300i32 {
                let l = quantize(f, qmul, qadd);
                let back = if l > 0 {
                    l * qmul + qadd
                } else if l < 0 {
                    l * qmul - qadd
                } else {
                    0
                };
                // 最优 level 的重建误差不该超过一个量化步长
                assert!(
                    (back - f).abs() <= qmul,
                    "q={q} f={f} level={l} 重建={back} 误差太大"
                );
            }
        }
    }

    #[test]
    fn pick_qscale_keeps_levels_in_range() {
        assert_eq!(pick_qscale(1, 0), 1);
        assert_eq!(pick_qscale(8, 100), 8);
        // |F|=254*8=2032 → 至少要 qscale=8 才能压到 127 以内
        assert_eq!(pick_qscale(1, 254 * 8), 8);
        assert_eq!(pick_qscale(1, i32::MAX), 31);
    }
}
