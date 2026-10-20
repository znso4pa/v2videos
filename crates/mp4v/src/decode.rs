//! MPEG-4 Part 2 **解码** —— I / P、H.263 量化、16×16 运动、渐进。
//!
//! ## 支持范围（先窄后宽）
//!
//! | 项 | 状态 |
//! |---|---|
//! | I / P 帧、矩形 VOL、`progressive_sequence = 1` | ✅ 与 ffmpeg **逐像素 0 差** |
//! | `quant_type = 0`（H.263 量化，ffmpeg 的默认） | ✅ |
//! | 16×16 运动矢量、半像素（`quarter_sample = 0`） | ✅ |
//! | `resync_marker_disable = 0`（**允许** marker 但实际不出现） | ✅ 本仓与 ffmpeg 的编码器都不插 |
//! | `mpeg_quant`（MPEG 矩阵量化） | ❌ 明确报 `Unsupported` |
//! | 4MV（`cbpc & 16`）、`dquant`、B / S 帧、隔行、GMC / sprite、数据分割 | ❌ 同上 |
//! | 流里**真的出现** resync marker | ❌ 不处理（宏块层会错位）—— ffmpeg **多线程**编码会按 slice 插，测试里用 `-threads 1` 避开 |
//!
//! 拒绝比猜好：这些分支每一个都「不报错、只出错」，猜出来的实现会把错画面
//! 当成功交出去。宁可明确说「还不支持」。
//!
//! ## P 帧漂移那件事（根因值得记）
//!
//! 第一版 P 帧能解完、不报错，但逐帧漂移：8 帧 I P P P I P P P 的不一致字节
//! `5161 / 7743 / 9876`，最大差 48，而且**每个 I 帧都清零、之后重新累积** ——
//! 典型的「参考帧被一点点污染」。
//!
//! 根因只有一行：**escape 里那个内层符号的 level 也要反量化**。
//! ffmpeg 的 `rl_vlc[qscale]` 表是按 qscale **预乘过**的（`ff_rl_init` 生成 32 档），
//! 所以 `GET_RL_VLC` 在 escape 分支里拿到的 `level` 已经是 `raw*qmul + qadd`；
//! 本实现用的是表里的原始 level，于是第一/第二 escape 的块比 ffmpeg 少一个 `qadd`。
//!
//! 为什么这么难看出来：**帧内 `qmul = 1`、`qadd = 0`，这一步是恒等** ——
//! 所以 I 帧全对、P 帧只有「用到 escape 的块」差一点。而 escape 只在
//! 系数幅度/游程超出普通码表时才出现（本例 300 个宏块里只有 5 个）。
//! 定位靠的是「把差异块反解出 DC」：我们的 DC=25、ffmpeg 的落在 31..36，
//! 而 25 = `1 + 3*8` 恰好少了 `qadd=3` —— 一眼就能对上是哪一条公式漏了。
//!
//! 另一个坑（测试侧的）：裸 `.m4v` 没有时间戳，ffmpeg 解到 raw 时会按**猜出来的
//! 帧率重采样、静默丢帧**（10 帧的流只吐 3 帧）。必须加 `-fps_mode passthrough`，
//! 否则「逐像素对账」只比了前几帧、照样全绿。
//!
//! ## 与 MP43（`v2v-msmpeg4`）的关系
//!
//! 复用的是**数值内核**：`BitReader` / `Vlc` / `RlTable` / `idct_put` / `idct_add` /
//! `mc_mb` / `Planes` / `ac_predict` / `dc_unscale`。不复用的是**语法**：
//!
//! - MCBPC / CBPY / MV 的**表不同**（MPEG-4 用 `ff_mvtab` 的 33 项，MP43 用 1100 项）；
//! - DC 预测的**平手方向相反**（MPEG-4 用严格 `<`，MP43 的 V3 用 `<=`）；
//! - **escape 的位模式不同**：MPEG-4 是 `0` → 第一、`10` → 第二、`11` → 第三；
//!   MP43 反过来（`1` → 第一、`01` → 第二、`00` → 第三）；
//! - 第二 escape 的 run 修正：MPEG-4 恒 `+1`，MP43 按版本（V2 是 0）。

use v2v_msmpeg4::{
    ac_predict, dc_unscale, idct_add, idct_put, mc_mb, AcVal, BitReader, Error, Frame, Planes,
    Result, RlTable, Vlc, ZIGZAG,
};

use crate::tables;

/// 视觉对象序列 / 视频对象 / 视频对象层 / 用户数据 / VOP 的起始码后缀。
const SC_VOS: u8 = 0xB0;
const SC_VO: u8 = 0xB5;
const SC_USER_DATA: u8 = 0xB2;
const SC_VOL: u8 = 0x20;
const SC_VOP: u8 = 0xB6;

/// `ff_mpeg4_dc_threshold`：VOP 头里那 3 位是**下标**。
const DC_THRESHOLD: [u32; 8] = [99, 13, 15, 17, 19, 21, 23, 0];

/// 视频对象层（VOL）里解码需要的那几个字段。
#[derive(Debug, Clone, Copy)]
pub struct Vol {
    pub width: u32,
    pub height: u32,
    /// `av_log2(fps_num - 1) + 1` —— VOP 头里 `vop_time_increment` 的位宽。
    pub time_increment_bits: u32,
    /// `quant_type`：`true` = MPEG 矩阵量化（**暂不支持**）。
    pub mpeg_quant: bool,
    /// `quarter_sample`：`true` = 四分之一像素（**暂不支持**）。
    pub quarter_sample: bool,
    /// `progressive_sequence`。
    pub progressive: bool,
    /// `vol_sprite_usage`（0 = 没有）。
    pub sprite: u32,
    pub data_partitioning: bool,
    /// `quant_precision`：4 或 5（`not_8_bit` 为真时是 4）。
    pub quant_precision: u32,
    /// `interlaced`（VOL 里的那位，取反后才是 `progressive_sequence`）。
    pub interlaced: bool,
}

impl Vol {
    /// 从 `esds` / AVI `strf` 的 extradata（一段以起始码开头的位流）里解析 VOL。
    pub fn parse(extra: &[u8]) -> Result<Self> {
        let (start, end) = find_start_code(extra, SC_VOL)
            .ok_or_else(|| Error::Bad("extradata 里没有 VOL 起始码（00 00 01 20）".into()))?;
        parse_vol(&extra[start..end])
    }
}

/// 一帧解码器。
pub struct Decoder {
    vol: Vol,
    /// 有没有拿到过 VOL。`new_from_stream()` 建出来的解码器靠**流里自带**的
    /// VOL 初始化（AVI 的 `strf` 里常常没有 extradata —— ffmpeg 把 VOL 写在
    /// 第一个样本的 VOP 前面）。
    have_vol: bool,
    prev: Option<Planes>,
    pw: usize,
    ph: usize,
    mb_w: usize,
    mb_h: usize,
}

impl Decoder {
    pub fn new(vol: Vol) -> Result<Self> {
        if vol.mpeg_quant {
            return Err(Error::Unsupported(
                "MPEG 矩阵量化（quant_type = 1）暂不支持".into(),
            ));
        }
        if vol.quarter_sample {
            return Err(Error::Unsupported("四分之一像素（qpel）暂不支持".into()));
        }
        if !vol.progressive {
            return Err(Error::Unsupported("隔行序列暂不支持".into()));
        }
        if vol.sprite != 0 {
            return Err(Error::Unsupported("sprite / GMC 暂不支持".into()));
        }
        if vol.data_partitioning {
            return Err(Error::Unsupported("数据分割暂不支持".into()));
        }
        let mb_w = (vol.width as usize).div_ceil(16);
        let mb_h = (vol.height as usize).div_ceil(16);
        Ok(Self {
            vol,
            have_vol: true,
            prev: None,
            pw: mb_w * 16,
            ph: mb_h * 16,
            mb_w,
            mb_h,
        })
    }

    /// 建一个**等流里自带 VOL** 的解码器（容器的 extradata 里没有序列头时用）。
    ///
    /// AVI 的 `strf` 常常是空的 40 字节 —— ffmpeg 把 VOL 写在第一个样本的 VOP 前面。
    /// 这种情况下只能先建解码器、等第一个样本到了再配置尺寸。
    pub fn new_from_stream() -> Self {
        Self {
            vol: Vol {
                width: 0,
                height: 0,
                time_increment_bits: 1,
                mpeg_quant: false,
                quarter_sample: false,
                progressive: true,
                sprite: 0,
                data_partitioning: false,
                quant_precision: 5,
                interlaced: false,
            },
            have_vol: false,
            prev: None,
            pw: 0,
            ph: 0,
            mb_w: 0,
            mb_h: 0,
        }
    }

    /// 解一个样本（可以带 VOL 头；`00 00 01 B6` 之前的都跳过）。
    pub fn decode(&mut self, data: &[u8]) -> Result<Frame> {
        let mut pos = 0usize;
        while pos + 4 <= data.len() {
            if data[pos] == 0 && data[pos + 1] == 0 && data[pos + 2] == 1 {
                let code = data[pos + 3];
                let start = pos + 4;
                let end = next_start_code(data, start).unwrap_or(data.len());
                match code {
                    SC_VOL => {
                        self.vol = parse_vol(&data[start..end])?;
                        let mb_w = (self.vol.width as usize).div_ceil(16);
                        let mb_h = (self.vol.height as usize).div_ceil(16);
                        self.mb_w = mb_w;
                        self.mb_h = mb_h;
                        self.pw = mb_w * 16;
                        self.ph = mb_h * 16;
                        self.have_vol = true;
                    }
                    SC_VOP => return self.decode_vop(&data[start..end]),
                    // 视觉对象序列 / 视觉对象 / 用户数据：都不影响解码，跳过。
                    SC_VOS | SC_VO | SC_USER_DATA => {}
                    _ => {}
                }
                pos = end;
            } else {
                pos += 1;
            }
        }
        Err(Error::Bad("样本里没有 VOP 起始码（00 00 01 B6）".into()))
    }

    fn decode_vop(&mut self, data: &[u8]) -> Result<Frame> {
        if !self.have_vol {
            return Err(Error::Bad("流里还没有 VOL（序列头）".into()));
        }
        let mut r = BitReader::new(data);
        let pict_type = r.u(2)?;
        if pict_type == 2 {
            return Err(Error::Unsupported("B 帧暂不支持".into()));
        }
        if pict_type == 3 {
            return Err(Error::Unsupported("S 帧（sprite）暂不支持".into()));
        }
        let is_p = pict_type == 1;

        // time_incr：一串 1，以一个 0 收尾。
        while r.bit()? != 0 {}
        marker(&mut r)?;
        let _time_increment = r.u(self.vol.time_increment_bits)?;
        marker(&mut r)?;
        if r.bit()? != 1 {
            return Err(Error::Bad("vop_coded = 0（这帧没有数据）".into()));
        }
        let no_rounding = if is_p { r.bit()? != 0 } else { false };
        // 矩形 VOL + 渐进 + 没有 complexity estimation → 这里没有别的字段。
        let intra_dc_threshold = DC_THRESHOLD[r.u(3)? as usize];
        let qscale = r.u(self.vol.quant_precision)?;
        if qscale == 0 {
            return Err(Error::Bad("qscale = 0".into()));
        }
        let f_code = if is_p {
            let f = r.u(3)?;
            if f == 0 {
                return Err(Error::Bad("f_code = 0".into()));
            }
            f
        } else {
            1
        };

        self.reconstruct(&mut r, is_p, qscale, f_code, no_rounding, intra_dc_threshold)
    }

    #[allow(clippy::too_many_arguments)]
    fn reconstruct(
        &mut self,
        r: &mut BitReader,
        is_p: bool,
        qscale: u32,
        f_code: u32,
        no_rounding: bool,
        intra_dc_threshold: u32,
    ) -> Result<Frame> {
        let t = Tables::get();
        let (mb_w, mb_h) = (self.mb_w, self.mb_h);
        let (pw, ph) = (self.pw, self.ph);

        if is_p && self.prev.is_none() {
            return Err(Error::Bad("P 帧需要参考帧（先解一帧 I）".into()));
        }

        let mut out = Planes::new(pw, ph);
        if is_p {
            out.y.copy_from_slice(&self.prev.as_ref().unwrap().y);
            out.u.copy_from_slice(&self.prev.as_ref().unwrap().u);
            out.v.copy_from_slice(&self.prev.as_ref().unwrap().v);
        }

        let q = qscale as usize;
        let y_dc_scale = tables::Y_DC_SCALE[q] as i32;
        let c_dc_scale = tables::C_DC_SCALE[q] as i32;
        let qmul = (qscale as i32) << 1;
        let qadd = ((qscale as i32) - 1) | 1;
        let use_intra_dc_vlc = qscale < intra_dc_threshold;

        // DC / AC 预测网格：带 1 格边框，DC 的初值是 **1024**（不是 0）。
        let mut dc_luma = Grid::with_init(2 * mb_w, 2 * mb_h, 1024i16);
        let mut dc_cb = Grid::with_init(mb_w, mb_h, 1024i16);
        let mut dc_cr = Grid::with_init(mb_w, mb_h, 1024i16);
        let mut ac_luma = Grid::<AcVal>::new(2 * mb_w, 2 * mb_h);
        let mut ac_cb = Grid::<AcVal>::new(mb_w, mb_h);
        let mut ac_cr = Grid::<AcVal>::new(mb_w, mb_h);
        // 运动矢量网格（每宏块一格，边框 0）。
        let mut mv_grid = Grid::<(i32, i32)>::new(mb_w, mb_h);

        for mby in 0..mb_h {
            for mbx in 0..mb_w {
                let cbp;
                let mut mv = (0i32, 0i32);
                let mut ac_pred = false;

                if is_p && r.bit()? != 0 {
                    // ---- 跳过宏块：MV = (0,0)、六块全不编码 ----
                    let g = mv_grid.idx(mbx, mby);
                    mv_grid.data[g] = (0, 0);
                    mc_mb(&mut out, self.prev.as_ref().unwrap(), mbx, mby, (0, 0), no_rounding);
                    continue;
                }

                let (mb_intra, cbpc) = if is_p {
                    // stuffing（cbpc == 20）要跳过重读 —— 上游是 `do { } while (cbpc == 20)`。
                    let mut cbpc = t.inter_mcbpc.decode(r)? as u32;
                    while cbpc == 20 {
                        cbpc = t.inter_mcbpc.decode(r)? as u32;
                    }
                    if cbpc & 8 != 0 {
                        return Err(Error::Unsupported("dquant 暂不支持".into()));
                    }
                    ((cbpc & 4) != 0, cbpc & 3)
                } else {
                    // ⚠️ 帧内的 **stuffing** 是 `cbpc == 8`（码字 `000000001`，9 位）——
                    // ffmpeg 那边是 `do { } while (cbpc == 8)`。不跳过它就会把一个
                    // 纯 stuffing 当成「cbpc=0 的普通宏块」，后面整条位流错位。
                    // 本仓编码器不插 stuffing，所以只在解**别人的**流时才会踩到。
                    let mut cbpc = t.intra_mcbpc.decode(r)? as u32;
                    while cbpc == 8 {
                        cbpc = t.intra_mcbpc.decode(r)? as u32;
                    }
                    if cbpc & 4 != 0 {
                        return Err(Error::Unsupported("dquant 暂不支持".into()));
                    }
                    (true, cbpc & 3)
                };

                if mb_intra {
                    ac_pred = r.bit()? != 0;
                    // 帧内的 CBPY **不取反**。
                    let cbpy = t.cbpy.decode(r)? as u32;
                    cbp = cbpc | (cbpy << 2);
                } else {
                    if cbpc & 16 != 0 {
                        return Err(Error::Unsupported("4MV（8×8 运动）暂不支持".into()));
                    }
                    // 帧间的 CBPY **要取反**（`^ 0x0F`）。
                    let cbpy = (t.cbpy.decode(r)? as u32) ^ 0x0F;
                    cbp = cbpc | (cbpy << 2);
                    let pred = pred_motion(&mv_grid, mbx, mby);
                    mv.0 = decode_motion(r, pred.0, f_code)?;
                    mv.1 = decode_motion(r, pred.1, f_code)?;
                    let g = mv_grid.idx(mbx, mby);
                    mv_grid.data[g] = mv;
                    mc_mb(&mut out, self.prev.as_ref().unwrap(), mbx, mby, mv, no_rounding);
                }

                for n in 0..6usize {
                    let coded = (cbp >> (5 - n)) & 1 != 0;
                    let luma = n < 4;
                    let scale = if luma { y_dc_scale } else { c_dc_scale };
                    let (dc_grid, ac_grid) = if luma {
                        (&mut dc_luma, &mut ac_luma)
                    } else if n == 4 {
                        (&mut dc_cb, &mut ac_cb)
                    } else {
                        (&mut dc_cr, &mut ac_cr)
                    };
                    let (gx, gy) = if luma {
                        (2 * mbx + (n & 1), 2 * mby + (n >> 1))
                    } else {
                        (mbx, mby)
                    };
                    let xy = dc_grid.idx(gx, gy);
                    let mut block = [0i16; 64];
                    let mut dir = 0usize;

                    if mb_intra {
                        // ---- DC 预测 ----
                        let (pred, d) = pred_dc(dc_grid, xy, mbx, mby, n, scale);
                        dir = d;
                        let mut level = 0i32;
                        if use_intra_dc_vlc {
                            // ⚠️ `dc` 的下标是「0 = 亮度、1 = 色度」，**不是** `bool as usize` ——
                            // `usize::from(true)` 是 1，会把亮度块喂给色度表
                            // （症状：DC 的 size 读成 6 而不是 7，整块错位）。
                            let size = t.dc[if luma { 0 } else { 1 }].decode(r)? as u32;
                            if size > 0 {
                                let v = r.u(size)? as i32;
                                // `get_xbits`：最高位是 0 表示负数，用「反码」还原。
                                level = if (v >> (size - 1)) == 0 {
                                    v - (1 << size) + 1
                                } else {
                                    v
                                };
                                if size > 8 {
                                    marker(r)?;
                                }
                            }
                        }
                        let total = level + pred;
                        let scaled = (total * scale) as i16;
                        dc_grid.data[xy] = scaled;
                        block[0] = scaled;
                    } else if !coded {
                        continue; // 帧间未编码块：不加残差（MC 已经铺过底了）
                    }

                    // ---- 系数 ----
                    let scan: &[u8; 64] = if mb_intra {
                        if ac_pred {
                            if dir == 0 {
                                &tables::V_SCAN
                            } else {
                                &tables::H_SCAN
                            }
                        } else {
                            &ZIGZAG
                        }
                    } else {
                        &ZIGZAG
                    };
                    if coded || !use_intra_dc_vlc {
                        let (qm, qa) = if mb_intra { (1, 0) } else { (qmul, qadd) };
                        let start = if mb_intra && use_intra_dc_vlc { 0 } else { -1 };
                        let rl = if mb_intra { &t.rl_intra } else { &t.rl_inter };
                        decode_coeffs(r, rl, scan, qm, qa, start, &mut block)?;
                    }

                    if mb_intra {
                        // AC 预测（**无论 coded 与否都要更新网格**）。
                        let nb_xy = if dir == 0 { xy - 1 } else { xy - ac_grid.w };
                        let neighbor = ac_grid.data[nb_xy];
                        let mut ac = [0i16; 16];
                        ac_predict(&mut block, &mut ac, &neighbor, dir, ac_pred);
                        ac_grid.data[xy] = ac;
                        // 帧内 AC 反量化（DC 已在上面乘过 scale）。
                        for i in 1..64usize {
                            let v = block[i] as i32;
                            if v != 0 {
                                block[i] = if v < 0 { v * qmul - qadd } else { v * qmul + qadd } as i16;
                            }
                        }
                    }
                    // ---- 落像素 ----
                    let (plane, stride) = plane_of(&mut out, n);
                    let ox = if luma { mbx * 16 + (n & 1) * 8 } else { mbx * 8 };
                    let oy = if luma { mby * 16 + (n >> 1) * 8 } else { mby * 8 };
                    let off = oy * stride + ox;
                    if mb_intra {
                        idct_put(&mut block, &mut plane[off..], stride);
                    } else {
                        idct_add(&mut block, &mut plane[off..], stride);
                    }
                }
            }
        }

        let _ = ph;
        self.prev = Some(out.clone());
        Ok(crop(out, self.vol.width as usize, self.vol.height as usize))
    }
}

/// 取第 `n` 个块所在的平面与行宽。
fn plane_of(p: &mut Planes, n: usize) -> (&mut [u8], usize) {
    match n {
        0..=3 => (&mut p.y, p.pw),
        4 => (&mut p.u, p.pw / 2),
        _ => (&mut p.v, p.pw / 2),
    }
}

/// 裁到真实宽高（`Planes` 是宏块对齐的）。
fn crop(p: Planes, w: usize, h: usize) -> Frame {
    let mut y = Vec::with_capacity(w * h);
    for r in 0..h {
        y.extend_from_slice(&p.y[r * p.pw..r * p.pw + w]);
    }
    let (cw, ch) = (w / 2, h / 2);
    let mut u = Vec::with_capacity(cw * ch);
    let mut v = Vec::with_capacity(cw * ch);
    for r in 0..ch {
        u.extend_from_slice(&p.u[r * (p.pw / 2)..r * (p.pw / 2) + cw]);
        v.extend_from_slice(&p.v[r * (p.pw / 2)..r * (p.pw / 2) + cw]);
    }
    Frame {
        width: w as u32,
        height: h as u32,
        y,
        u,
        v,
    }
}

/// 带 1 格边框的块级网格。
struct Grid<T> {
    w: usize,
    data: Vec<T>,
}

impl<T: Copy + Default> Grid<T> {
    fn new(w: usize, h: usize) -> Self {
        Self::with_init(w, h, T::default())
    }
    fn with_init(w: usize, h: usize, init: T) -> Self {
        Self {
            w: w + 2,
            data: vec![init; (w + 2) * (h + 2)],
        }
    }
    fn idx(&self, gx: usize, gy: usize) -> usize {
        (gy + 1) * self.w + gx + 1
    }
}

/// DC 预测（`ff_mpeg4_pred_dc` 的预测部分）。
///
/// ⚠️ 平手方向与 MP43 相反：这里是 **`|a-b| < |b-c|` 才取 C**（严格小于），
/// 平手取 A；MP43 的 V3 是 `<=`。差别只在「三个邻居恰好等距」时出现，
/// 而那种情况在帧首行、宏块边界上很常见 —— 写反了会整片偏一点、还查不出。
fn pred_dc(
    grid: &Grid<i16>,
    xy: usize,
    mbx: usize,
    mby: usize,
    n: usize,
    scale: i32,
) -> (i32, usize) {
    let mut a = grid.data[xy - 1] as i32;
    let mut b = grid.data[xy - 1 - grid.w] as i32;
    let mut c = grid.data[xy - grid.w] as i32;

    // 片首行（没有 resync marker 时就是第 0 行）。
    if mby == 0 && n != 3 {
        if n != 2 {
            b = 1024;
            c = 1024;
        }
        if n != 1 && mbx == 0 {
            b = 1024;
            a = 1024;
        }
    }
    if mbx == 0 && mby == 1 && (n == 0 || n == 4 || n == 5) {
        b = 1024;
    }

    let (pred, dir) = if (a - b).abs() < (b - c).abs() {
        (c, 1)
    } else {
        (a, 0)
    };
    (dc_unscale(pred, scale), dir)
}

/// H.263 的运动矢量预测（`ff_h263_pred_motion`）：左 / 上 / 右上取中值。
fn pred_motion(g: &Grid<(i32, i32)>, mbx: usize, mby: usize) -> (i32, i32) {
    let xy = g.idx(mbx, mby);
    let a = g.data[xy - 1];
    if mby == 0 {
        // 片首行：只有左邻可用。
        if mbx == 0 {
            (0, 0)
        } else {
            a
        }
    } else {
        let b = g.data[xy - g.w];
        let c = g.data[xy - g.w + 1];
        (
            mid(a.0, b.0, c.0),
            mid(a.1, b.1, c.1),
        )
    }
}

fn mid(a: i32, b: i32, c: i32) -> i32 {
    // ⚠️ 直接复用 msmpeg4 里那份（已对着 ffmpeg 逐像素验过）。
    // 这里原先自己写了一版 `a.max(b).min(c).max(...)` —— 它不是中值：
    // `(3,2,1)` 会返回 1 而不是 2。中值错了，P 帧的运动矢量预测就整片偏。
    v2v_msmpeg4::mid_pred(a, b, c)
}

/// 一个 MV 分量：`ff_mvtab` + 符号位 + `f_code - 1` 个低位，然后加预测、按
/// `5 + f_code` 位做模回绕。
fn decode_motion(r: &mut BitReader, pred: i32, f_code: u32) -> Result<i32> {
    let code = Tables::get().mv.decode(r)? as i32;
    if code == 0 {
        return Ok(pred);
    }
    let sign = r.bit()? != 0;
    let shift = f_code - 1;
    let mut val = code;
    if shift > 0 {
        val = (val - 1) << shift;
        val |= r.u(shift)? as i32;
        val += 1;
    }
    if sign {
        val = -val;
    }
    let v = val + pred;
    Ok(sign_extend(v, 5 + f_code))
}

fn sign_extend(x: i32, bits: u32) -> i32 {
    let m = 1i32 << (bits - 1);
    (x & (m - 1)) - (x & m)
}

/// 解一串系数。`start_i`：走了 DC 的 VLC 就是 0，否则 -1（DC 在 AC 流里）。
///
/// `qmul` / `qadd` 是**反量化参数**：帧内传 1/0（反量化在 AC 预测之后单独做），
/// 帧间传真值（边解边反量化）。
fn decode_coeffs(
    r: &mut BitReader,
    rl: &RlTable,
    scan: &[u8; 64],
    qmul: i32,
    qadd: i32,
    start_i: i32,
    block: &mut [i16; 64],
) -> Result<()> {
    let mut i = start_i;
    loop {
        let sym = rl.vlc.decode(r)?;
        if sym == rl.len() {
            // ---- escape：MPEG-4 是 `0` → 第一、`10` → 第二、`11` → 第三 ----
            if r.bit()? == 1 {
                if r.bit()? == 1 {
                    // 第三 escape：last(1) run(6) marker(1) level(12 有符号) marker(1)
                    let last = r.bit()? != 0;
                    let run = r.u(6)? as i32;
                    marker(r)?;
                    // 12 位**有符号**：2048..4095 表示 -2048..-1，必须**符号扩展**。
                    // 写成 `as i16` 只是截断到 16 位，负数会原样变成 2048..4095 的正数
                    // —— 系数炸到 3 万多，重建出来是一块纯白。实测踩过。
                    let l = sign_extend(r.u(12)? as i32, 12);
                    marker(r)?;
                    let level = if l > 0 { l * qmul + qadd } else { l * qmul - qadd };
                    i += run + 1;
                    if last {
                        i += 192;
                    }
                    if i > 62 {
                        i -= 192;
                        if i & !63 != 0 {
                            return Err(Error::Bad("ac 系数溢出（第三 escape）".into()));
                        }
                        block[scan[i as usize] as usize] = level as i16;
                        break;
                    }
                    block[scan[i as usize] as usize] = level as i16;
                    continue;
                }
                // 第二 escape：再解一个 RL 符号，run 补上该 (last, level) 的上界（+1）
                let sym2 = rl.vlc.decode(r)?;
                if sym2 == rl.len() {
                    return Err(Error::Bad("第二 escape 里又解到 escape".into()));
                }
                let (run2, lvl2, last2) = rl.symbol(sym2);
                // ⚠️ **内层符号的 level 也要反量化**：ffmpeg 的 `rl_vlc[qscale]` 表是按
                // qscale **预乘过**的（`ff_rl_init` 生成 32 档），所以 escape 分支里拿到的
                // `level` 已经是 `raw*qmul + qadd`。漏了这一步只有**帧间**会错
                // （帧内 qmul=1/qadd=0 是恒等）—— 症状正是「I 帧全对、P 帧只在用到
                // escape 的块上差一点，而且误差随参考帧累积」。
                let lvl2 = lvl2 * qmul + qadd;
                let lv = (lvl2 / qmul.max(1)).clamp(0, 63) as usize;
                // ⚠️ `run2` 是**表里的原值**（0 起），而 `i` 走的是「`run + 1`」那套。
                // 第二 escape 的 run 是 `max_run[last][level] + 1 + 表里的 run`，
                // 再按「run + 1」前进 —— 合起来是 `run2 + max_run + 2`。
                // 早先写成 `run2 + 1 + max_run`，**少 1**，位置一路往前偏。
                i += run2 as i32 + 2 + rl.max_run_at(usize::from(last2), lv as i32);
                if last2 {
                    i += 192;
                }
                let level = lvl2;
                let level = if r.bit()? != 0 { -level } else { level };
                if i > 62 {
                    i -= 192;
                    if i & !63 != 0 {
                        return Err(Error::Bad("ac 系数溢出（第二 escape）".into()));
                    }
                    block[scan[i as usize] as usize] = level as i16;
                    break;
                }
                block[scan[i as usize] as usize] = level as i16;
                continue;
            }
            // 第一 escape：run 照用，level 补上该 (last, run) 的上界
            let sym2 = rl.vlc.decode(r)?;
            if sym2 == rl.len() {
                return Err(Error::Bad("第一 escape 里又解到 escape".into()));
            }
            let (run2, lvl2, last2) = rl.symbol(sym2);
            // 第一 escape 的 run **不改**，但仍按「run + 1」前进（同普通符号）。
            i += run2 as i32 + 1;
            if last2 {
                i += 192;
            }
            // 同上：内层 level 已按 qscale 预乘过。
            let level = lvl2 * qmul + qadd + rl.max_level_at(usize::from(last2), run2 & 63) * qmul;
            let level = if r.bit()? != 0 { -level } else { level };
            if i > 62 {
                i -= 192;
                if i & !63 != 0 {
                    return Err(Error::Bad("ac 系数溢出（第一 escape）".into()));
                }
                block[scan[i as usize] as usize] = level as i16;
                break;
            }
            block[scan[i as usize] as usize] = level as i16;
            continue;
        }

        let (run, level, last) = rl.symbol(sym);
        let mut level = level * qmul + qadd;
        // `last` 在 run 上加 192，靠 `i > 62` 再减回来。
        i += run as i32 + 1 + if last { 192 } else { 0 };
        if r.bit()? != 0 {
            level = -level;
        }
        if i > 62 {
            i -= 192;
            if i & !63 != 0 {
                return Err(Error::Bad("ac 系数溢出".into()));
            }
            block[scan[i as usize] as usize] = level as i16;
            break;
        }
        block[scan[i as usize] as usize] = level as i16;
    }
    Ok(())
}

fn marker(r: &mut BitReader) -> Result<()> {
    if r.bit()? != 1 {
        return Err(Error::Bad("marker 位不是 1".into()));
    }
    Ok(())
}

/// 找某个起始码，返回**起始码之后**到下一个起始码之前的那段。
fn find_start_code(data: &[u8], code: u8) -> Option<(usize, usize)> {
    let mut pos = 0usize;
    while pos + 4 <= data.len() {
        if data[pos] == 0 && data[pos + 1] == 0 && data[pos + 2] == 1 {
            let start = pos + 4;
            let end = next_start_code(data, start).unwrap_or(data.len());
            if data[pos + 3] == code {
                return Some((start, end));
            }
            pos = end;
        } else {
            pos += 1;
        }
    }
    None
}

fn next_start_code(data: &[u8], from: usize) -> Option<usize> {
    let mut pos = from;
    while pos + 3 < data.len() {
        if data[pos] == 0 && data[pos + 1] == 0 && data[pos + 2] == 1 {
            return Some(pos);
        }
        pos += 1;
    }
    None
}

/// VOL 解析（`decode_vol_header` 的子集；不支持的组合在 `Decoder::new` 里拒绝）。
fn parse_vol(data: &[u8]) -> Result<Vol> {
    let mut r = BitReader::new(data);
    let _random_accessible = r.bit()?;
    let vo_type = r.u(8)?;
    if vo_type == 0x13 || vo_type == 0x12 {
        return Err(Error::Unsupported("Studio profile 暂不支持".into()));
    }
    let verid = if r.bit()? == 1 {
        let v = r.u(4)?;
        let _priority = r.u(3)?;
        v
    } else {
        1
    };
    let aspect = r.u(4)?;
    if aspect == 15 {
        let _par_w = r.u(8)?;
        let _par_h = r.u(8)?;
    }
    let mut low_delay = false;
    if r.bit()? == 1 {
        let chroma = r.u(2)?;
        if chroma != 1 {
            return Err(Error::Unsupported("只支持 4:2:0".into()));
        }
        low_delay = r.bit()? != 0;
        if r.bit()? == 1 {
            // vbv_parameters
            r.u(15)?;
            marker(&mut r)?;
            r.u(15)?;
            marker(&mut r)?;
            r.u(15)?;
            marker(&mut r)?;
            r.u(3)?;
            r.u(11)?;
            marker(&mut r)?;
            r.u(15)?;
            marker(&mut r)?;
        }
    }
    let _ = low_delay;
    let shape = r.u(2)?;
    if shape != 0 {
        return Err(Error::Unsupported("只支持矩形 VOL（shape = 0）".into()));
    }
    marker(&mut r)?;
    let fps_num = r.u(16)?;
    if fps_num == 0 {
        return Err(Error::Bad("vop_time_increment_resolution = 0".into()));
    }
    let time_increment_bits = (32 - (fps_num - 1).leading_zeros()).max(1);
    marker(&mut r)?;
    if r.bit()? == 1 {
        let _fixed = r.u(time_increment_bits)?;
    }
    marker(&mut r)?;
    let width = r.u(13)?;
    marker(&mut r)?;
    let height = r.u(13)?;
    marker(&mut r)?;
    if width == 0 || height == 0 {
        return Err(Error::Bad("VOL 的宽或高是 0".into()));
    }
    // ⚠️ 这一位是 **`interlaced`**（0 = 渐进），与 `progressive_sequence` 是取反关系。
    // 早先写成 `r.bit()? ^ 1 == 0` —— Rust 里 `^` 比 `==` 优先级**高**，
    // 实际算的是 `bit == 1`，把渐进判反了，于是连自己编的流都进不来。
    let interlaced = r.bit()? != 0;
    if interlaced {
        // 隔行的 VOL 后面还跟着 `top_field_first` / `alternate_scan` 两位。
        // 不支持就别硬着头皮往下读 —— 读歪了报出来的错会离根因很远。
        return Err(Error::Unsupported("隔行序列暂不支持".into()));
    }
    let progressive = true;
    let _obmc_disable = r.bit()?;
    let sprite = if verid == 1 { r.u(1)? } else { r.u(2)? };
    if sprite != 0 {
        return Err(Error::Unsupported("sprite / GMC 暂不支持".into()));
    }
    if verid != 1 && shape != 0 {
        let _ = r.u(1)?;
    }
    let not_8_bit = r.bit()? != 0;
    let quant_precision = if not_8_bit {
        let _qp = r.u(4)?;
        let _bpp = r.u(4)?;
        4
    } else {
        5
    };
    let mpeg_quant = r.bit()? != 0;
    if mpeg_quant {
        // 自定义矩阵（ffmpeg 的编码器只置 quant_type，不传矩阵）。
        if r.bit()? == 1 {
            for _ in 0..64 {
                if r.u(8)? == 0 {
                    break;
                }
            }
        }
        if r.bit()? == 1 {
            for _ in 0..64 {
                if r.u(8)? == 0 {
                    break;
                }
            }
        }
    }
    let complexity_disable = r.bit()? != 0;
    if !complexity_disable {
        return Err(Error::Unsupported("complexity estimation 暂不支持".into()));
    }
    // `resync_marker_disable = 0` 只表示「VOP 里**允许**出现 resync marker」，
    // **不是**「一定出现」—— 本仓编码器与 ffmpeg 都写 0、却都不插 marker。
    // 所以这一位不能当拒绝条件（早先写成了 `if !resync_disable { 拒绝 }`，
    // 结果连自己编的流都解不了）。真要出现 marker 时本解码器不处理，
    // 宏块层会错位 —— 见模块文档的能力表。
    let _resync_marker_disable = r.bit()?;
    let data_partitioning = r.bit()? != 0;
    Ok(Vol {
        width,
        height,
        time_increment_bits,
        mpeg_quant,
        quarter_sample: false,
        progressive,
        sprite,
        data_partitioning,
        quant_precision,
        interlaced: !progressive,
    })
}

/// 建一次就够的码表（`OnceLock`）。
struct Tables {
    intra_mcbpc: Vlc,
    inter_mcbpc: Vlc,
    cbpy: Vlc,
    dc: [Vlc; 2],
    mv: Vlc,
    rl_intra: RlTable,
    rl_inter: RlTable,
}

impl Tables {
    fn get() -> &'static Self {
        static T: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
        T.get_or_init(|| {
            let mcbpc = |code: &'static [u8], bits: &'static [u8]| {
                let pairs: Vec<[u32; 2]> = code
                    .iter()
                    .zip(bits.iter())
                    .map(|(&c, &l)| [u32::from(c), u32::from(l)])
                    .collect();
                Vlc::new(&pairs)
            };
            let pairs_u8 = |t: &'static [[u8; 2]]| {
                let p: Vec<[u32; 2]> = t
                    .iter()
                    .map(|&[c, l]| [u32::from(c), u32::from(l)])
                    .collect();
                Vlc::new(&p)
            };
            Tables {
                intra_mcbpc: mcbpc(&tables::INTRA_MCBPC_CODE, &tables::INTRA_MCBPC_BITS),
                inter_mcbpc: mcbpc(&tables::INTER_MCBPC_CODE, &tables::INTER_MCBPC_BITS),
                cbpy: pairs_u8(&tables::CBPY),
                dc: [pairs_u8(&tables::DC_LUM), pairs_u8(&tables::DC_CHROM)],
                mv: pairs_u8(&tables::MV_TAB),
                rl_intra: RlTable::new(
                    &tables::RL_INTRA_VLC,
                    &tables::RL_INTRA_RUN,
                    &tables::RL_INTRA_LEVEL,
                    67,
                ),
                rl_inter: RlTable::new(
                    &tables::RL_INTER_VLC,
                    &tables::RL_INTER_RUN,
                    &tables::RL_INTER_LEVEL,
                    58,
                ),
            }
        })
    }
}
