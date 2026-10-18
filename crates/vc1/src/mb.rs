//! VC-1 的**宏块层**：I 帧与 P 帧的宏块语法 + 系数解码 + 重建。
//!
//! 对照上游：`vc1_block.c` 的 `vc1_decode_i_blocks()` / `vc1_decode_p_blocks()`、
//! `vc1_decode_i_block()` / `vc1_decode_intra_block()` / `vc1_decode_p_block()`、
//! `vc1_decode_ac_coeff()`，以及 `GET_MVDATA` / `GET_MQUANT` 两个宏。
//!
//! ## 三处最容易写错、而且错了不会报错的地方
//!
//! 1. **DC 预测是「全量」不是「差分」**：`dc_val` 里存的是**预测后的完整 DC**，
//!    下一块拿它当预测值。所以 `block[0] = dcdiff * scale` 已经是完整系数，
//!    逆变换出来直接就是像素（I 帧 `put_pixels_clamped` 是**覆盖**，不是叠加）。
//! 2. **P 帧的帧内块多一个 `+128`**：I 帧走 `put_pixels_clamped`（块值当像素用），
//!    P 帧的帧内块走 `put_signed_pixels_clamped`（块值 + 128 再夹取）。
//!    同一份系数在两处的像素差 128 —— 这是上游的写法，不是笔误。
//! 3. **AC 预测的「左 / 上」是内存布局里的行 / 列**：`res_fasttx` 下
//!    `left_blk_sh = 0`、`top_blk_sh = 3`（转置布局），别按名字想当然。

use v2v_msmpeg4::{Error, Result, Vlc};

use crate::decoder::Ctx;
use crate::idct;
use crate::tables;

// ---- 前缀码（每张表建一次树，全局共用）----

fn vlc_u16(t: &[[u16; 2]]) -> Vlc {
    let v: Vec<[u32; 2]> = t.iter().map(|p| [p[0] as u32, p[1] as u32]).collect();
    Vlc::new(&v)
}

/// 宏块类型（I 帧的 CBP）：**就是** MS-MPEG-4 那张 `ff_msmp4_mb_i_vlc`。
fn vlc_mb_intra() -> &'static Vlc {
    static V: std::sync::OnceLock<Vlc> = std::sync::OnceLock::new();
    V.get_or_init(|| Vlc::new(&v2v_msmpeg4::tables::MB_INTRA))
}

/// DC 差分：`[dc_table_index][亮度/色度]`，同样是 MS-MPEG-4 那张 `ff_msmp4_dc_vlc`。
fn vlc_dc(table: usize, chroma: bool) -> &'static Vlc {
    static V: std::sync::OnceLock<Vec<Vlc>> = std::sync::OnceLock::new();
    let all = V.get_or_init(|| {
        let mut out = Vec::with_capacity(4);
        for t in 0..2 {
            for c in 0..2 {
                out.push(Vlc::new(&v2v_msmpeg4::tables::DC[t][c]));
            }
        }
        out
    });
    &all[table * 2 + chroma as usize]
}

/// AC 系数：8 个编码集各一张。
fn vlc_ac(codingset: usize) -> &'static Vlc {
    static V: std::sync::OnceLock<Vec<Vlc>> = std::sync::OnceLock::new();
    let all = V.get_or_init(|| {
        tables::AC_TABLES
            .iter()
            .map(|t| Vlc::new(t))
            .collect::<Vec<_>>()
    });
    &all[codingset]
}

fn vlc_p_cbpcy(tab: usize) -> &'static Vlc {
    static V: std::sync::OnceLock<Vec<Vlc>> = std::sync::OnceLock::new();
    &V.get_or_init(|| tables::P_CBPCY_VLC.iter().map(|t| vlc_u16(t)).collect())[tab]
}

fn vlc_p_mv_diff(tab: usize) -> &'static Vlc {
    static V: std::sync::OnceLock<Vec<Vlc>> = std::sync::OnceLock::new();
    &V.get_or_init(|| tables::P_MV_DIFF_VLC.iter().map(|t| vlc_u16(t)).collect())[tab]
}

fn vlc_ttmb(tab: usize) -> &'static Vlc {
    static V: std::sync::OnceLock<Vec<Vlc>> = std::sync::OnceLock::new();
    &V.get_or_init(|| tables::TTMB_VLC.iter().map(|t| vlc_u16(t)).collect())[tab]
}

fn vlc_ttblk(tab: usize) -> &'static Vlc {
    static V: std::sync::OnceLock<Vec<Vlc>> = std::sync::OnceLock::new();
    &V.get_or_init(|| tables::TTBLK_VLC.iter().map(|t| vlc_u16(t)).collect())[tab]
}

fn vlc_subblkpat(tab: usize) -> &'static Vlc {
    static V: std::sync::OnceLock<Vec<Vlc>> = std::sync::OnceLock::new();
    &V.get_or_init(|| tables::SUBBLKPAT_VLC.iter().map(|t| vlc_u16(t)).collect())[tab]
}

/// `vc1_i_pred_dc()` 里那张小表：`scale -> 1024/scale` 那一列「外边界值」。
/// 上游是函数内的局部静态表，照抄。
static DCPRED: [u16; 32] = [
    0, 1024, 512, 341, 256, 205, 171, 146, 128, 114, 102, 93, 85, 79, 73, 68, 64, 60, 57, 54, 51,
    49, 47, 45, 43, 41, 39, 38, 37, 35, 34, 33,
];

/// `offset_table[1]` 与 `size_table`：`GET_MVDATA` 里那两张本地小表。
static MV_OFFSET: [u8; 9] = [0, 1, 3, 7, 15, 31, 63, 127, 255];
static MV_SIZE: [u8; 6] = [0, 2, 3, 4, 5, 8];

impl Ctx<'_, '_> {
    // ================= I 帧 =================

    /// `vc1_decode_i_blocks()`。
    pub(crate) fn decode_i_blocks(&mut self) -> Result<()> {
        let y_idx = self.hdr.y_ac_table_index.unwrap_or(0);
        self.codingset = match y_idx {
            0 => {
                if self.hdr.pqindex <= 8 {
                    tables::CS_HIGH_RATE_INTRA as usize
                } else {
                    tables::CS_LOW_MOT_INTRA as usize
                }
            }
            1 => tables::CS_HIGH_MOT_INTRA as usize,
            _ => tables::CS_MID_RATE_INTRA as usize,
        };
        self.codingset2 = self.coding_set2();
        // I 帧的 DC 定标整帧一样（y 与 c 用同一张表）。
        self.y_dc_scale = tables::DC_SCALE[self.hdr.pq as usize] as i32;
        self.c_dc_scale = self.y_dc_scale;

        for mby in 0..self.mb_height {
            for mbx in 0..self.mb_width {
                *self.last_mb = (mbx, mby);
                *self.last_bits = self.r.bit_pos();
                self.decode_i_mb(mbx, mby)?;
                if self.loop_filter {
                    self.i_loop_filter(mbx, mby);
                }
            }
        }
        Ok(())
    }

    /// `c_ac_table_index` -> `codingset2`（I / P 帧同一段逻辑）。
    fn coding_set2(&self) -> usize {
        match self.hdr.c_ac_table_index {
            0 => {
                if self.hdr.pqindex <= 8 {
                    tables::CS_HIGH_RATE_INTER as usize
                } else {
                    tables::CS_LOW_MOT_INTER as usize
                }
            }
            1 => tables::CS_HIGH_MOT_INTER as usize,
            _ => tables::CS_MID_RATE_INTER as usize,
        }
    }

    fn decode_i_mb(&mut self, mbx: usize, mby: usize) -> Result<()> {
        let cbp = vlc_mb_intra().decode(&mut self.r)? as u32;
        let ac_pred = self.r.bit()? != 0;
        for k in 0..6usize {
            let mut val = (cbp >> (5 - k)) & 1;
            let bi = self.blk(mbx, mby, k);
            self.grids.mb_type[bi] = 1;
            if k < 4 {
                // 亮度块的 CBP 要**异或**邻居的「编码块预测」。
                let pred = self.coded_block_pred(mbx, mby, k);
                val ^= pred;
                self.grids.coded_block[bi] = val as u8;
            }
            let block = self.decode_i_block(mbx, mby, k, val != 0, ac_pred)?;
            self.put_block(mbx, mby, k, &block, false);
        }
        Ok(())
    }

    // ================= P 帧 =================

    /// `vc1_decode_p_blocks()`。
    pub(crate) fn decode_p_blocks(&mut self) -> Result<()> {
        // ⚠️ P 帧两个码表都从 `c_ac_table_index` 来（上游就是这样，不是笔误）。
        self.codingset = match self.hdr.c_ac_table_index {
            0 => {
                if self.hdr.pqindex <= 8 {
                    tables::CS_HIGH_RATE_INTRA as usize
                } else {
                    tables::CS_LOW_MOT_INTRA as usize
                }
            }
            1 => tables::CS_HIGH_MOT_INTRA as usize,
            _ => tables::CS_MID_RATE_INTRA as usize,
        };
        self.codingset2 = self.coding_set2();
        self.tt_index = (self.hdr.pq > 4) as usize + (self.hdr.pq > 12) as usize;

        for mby in 0..self.mb_height {
            for mbx in 0..self.mb_width {
                *self.last_mb = (mbx, mby);
                *self.last_bits = self.r.bit_pos();
                self.decode_p_mb(mbx, mby)?;
                if self.loop_filter {
                    self.p_loop_filter(mbx, mby);
                }
            }
        }
        Ok(())
    }

    /// `GET_MVDATA`：P 帧的「宏块类型」不是一个单独的 VLC —— 它**从 mv_diff 的下标推出来**。
    ///
    /// 返回 `(dmv_x, dmv_y, mb_intra, mb_has_coeffs)`。
    fn get_mvdata(&mut self) -> Result<(i32, i32, bool, bool)> {
        let mv_table_index = self.hdr.p.as_ref().unwrap().mv_table_index as usize;
        let qs = self.hdr.p.as_ref().unwrap().quarter_sample;
        let mut index = 1 + vlc_p_mv_diff(mv_table_index).decode(&mut self.r)? as i32;
        let mb_has_coeffs;
        if index > 36 {
            mb_has_coeffs = true;
            index -= 37;
        } else {
            mb_has_coeffs = false;
        }
        let mut mb_intra = false;
        let (mut dmv_x, mut dmv_y);
        if index == 0 {
            dmv_x = 0;
            dmv_y = 0;
        } else if index == 35 {
            dmv_x = self.r.u(self.k_x() - 1 + qs as u32)? as i32;
            dmv_y = self.r.u(self.k_y() - 1 + qs as u32)? as i32;
        } else if index == 36 {
            dmv_x = 0;
            dmv_y = 0;
            mb_intra = true;
        } else {
            let index1 = (index % 6) as usize;
            dmv_x = MV_OFFSET[index1] as i32;
            let mut val = MV_SIZE[index1] as i32 - (!qs && index1 == 5) as i32;
            if val > 0 {
                val = self.r.u(val as u32)? as i32;
                let sign = -(val & 1);
                dmv_x = (sign ^ ((val >> 1) + dmv_x)) - sign;
            }
            let index1 = (index / 6) as usize;
            dmv_y = MV_OFFSET[index1] as i32;
            let mut val = MV_SIZE[index1] as i32 - (!qs && index1 == 5) as i32;
            if val > 0 {
                val = self.r.u(val as u32)? as i32;
                let sign = -(val & 1);
                dmv_y = (sign ^ ((val >> 1) + dmv_y)) - sign;
            }
        }
        Ok((dmv_x, dmv_y, mb_intra, mb_has_coeffs))
    }

    fn k_x(&self) -> u32 {
        let mvrange = self.hdr.mvrange as u32;
        mvrange + 9 + (mvrange >> 1)
    }

    fn k_y(&self) -> u32 {
        self.hdr.mvrange as u32 + 8
    }

    /// `GET_MQUANT`：宏块级量化步长（`dquantfrm` 没开时就是帧级的 `pq`）。
    fn get_mquant(&mut self, mquant: i32, mbx: usize, mby: usize) -> Result<i32> {
        let Some(dq) = self.hdr.dquant.as_ref() else {
            return Ok(mquant);
        };
        let altpq = dq.altpq.unwrap_or(0) as i32;
        let mut mq = mquant;
        let mut edges = 0u32;
        if dq.dqprofile == tables::DQPROFILE_ALL_MBS {
            if dq.dqbilevel {
                mq = if self.r.bit()? != 0 { -altpq } else { self.hdr.pq as i32 };
            } else {
                let mqdiff = self.r.u(3)? as i32;
                if mqdiff != 7 {
                    mq = -(self.hdr.pq as i32) - mqdiff;
                } else {
                    mq = -(self.r.u(5)? as i32);
                }
            }
        }
        if dq.dqprofile == tables::DQPROFILE_SINGLE_EDGE {
            edges = 1 << dq.dqsbedge;
        } else if dq.dqprofile == tables::DQPROFILE_DOUBLE_EDGES {
            edges = (3 << dq.dqsbedge) % 15;
        } else if dq.dqprofile == tables::DQPROFILE_FOUR_EDGES {
            edges = 15;
        }
        if (edges & 1) != 0 && mbx == 0 {
            mq = -altpq;
        }
        if (edges & 2) != 0 && mby == 0 {
            mq = -altpq;
        }
        if (edges & 4) != 0 && mbx == self.mb_width - 1 {
            mq = -altpq;
        }
        if (edges & 8) != 0 && mby == self.mb_height - 1 {
            mq = -altpq;
        }
        if mq == 0 || mq > 31 || mq < -31 {
            // 上游在这里报错并强制成 1；本实现照做（不是「静默」—— 值本身就非法）。
            mq = 1;
        }
        Ok(mq)
    }

    /// `vc1_decode_p_mb()`（渐进式那一条路）。
    fn decode_p_mb(&mut self, mbx: usize, mby: usize) -> Result<()> {
        // ⚠️ 这里**不要** `self.hdr.p...clone()`：`PFrameHeader` 里带着两张位平面，
        // 一个宏块克隆一次就是 malloc + 几 KB memcpy。只要把标量取出来，
        // 位平面在**短作用域**里借一下（`self.hdr` 与 `self.r` 是不相交的字段，
        // 借用检查器允许同时存在）。
        let (fourmv, skipped) = {
            let p = self.hdr.p.as_ref().unwrap();
            let fourmv = match &p.mv_type_plane {
                Some(pl) if pl.raw => self.r.bit()? != 0,
                Some(pl) => pl.get(mbx, mby) != 0,
                None => false,
            };
            let skipped = if p.skip_plane.raw {
                self.r.bit()? != 0
            } else {
                p.skip_plane.get(mbx, mby) != 0
            };
            (fourmv, skipped)
        };
        let (cbptab, tt, ttmbf) = {
            let p = self.hdr.p.as_ref().unwrap();
            (p.cbptab, p.tt, p.ttmbf)
        };
        let mb_pos = self.mbidx(mbx, mby);

        let mut mquant = self.hdr.pq as i32;
        let mut block_cbp = 0u32;
        let mut block_tt = 0u32;
        let mut block_intra = 0u8;

        if !fourmv {
            if !skipped {
                let (dmv_x, dmv_y, mb_intra, mb_has_coeffs) = self.get_mvdata()?;
                if mb_intra {
                    // 帧内宏块：色度 MV 清 0（上游存的是一份独立的色度 MV 面）。
                    self.grids.luma_mv[mb_pos] = [0, 0];
                }
                self.pred_mv(mbx, mby, 0, dmv_x, dmv_y, true, mb_intra)?;

                let mut ac_pred = false;
                let cbp;
                if mb_intra && !mb_has_coeffs {
                    mquant = self.get_mquant(mquant, mbx, mby)?;
                    ac_pred = self.r.bit()? != 0;
                    cbp = 0u32;
                } else if mb_has_coeffs {
                    if mb_intra {
                        ac_pred = self.r.bit()? != 0;
                    }
                    cbp = vlc_p_cbpcy(cbptab as usize).decode(&mut self.r)? as u32;
                    mquant = self.get_mquant(mquant, mbx, mby)?;
                } else {
                    mquant = self.hdr.pq as i32;
                    cbp = 0;
                }
                self.mbp.qscale[mb_pos] = mquant;

                let mut ttmb: i32 = tt as i32;
                if !ttmbf && !mb_intra && mb_has_coeffs {
                    ttmb = vlc_ttmb(self.tt_index).decode(&mut self.r)? as i32;
                }
                if !mb_intra {
                    self.mc_1mv(mbx, mby);
                }

                let mut first_block = true;
                for i in 0..6usize {
                    self.set_dc(mbx, mby, i, 0);
                    let val = (cbp >> (5 - i)) & 1;
                    let bi = self.blk(mbx, mby, i);
                    self.grids.mb_type[bi] = mb_intra as u8;
                    if mb_intra {
                        self.set_avail(mbx, mby, i);
                        let block = self.decode_intra_block(mbx, mby, i, val != 0, mquant, ac_pred)?;
                        self.put_block(mbx, mby, i, &block, true);
                        block_cbp |= 0xF << (i << 2);
                        block_intra |= 1 << i;
                    } else if val != 0 {
                        let (pat, tt) =
                            self.decode_p_block(mbx, mby, i, mquant, ttmb, first_block)?;
                        block_cbp |= pat << (i << 2);
                        block_tt |= tt;
                        if !ttmbf && ttmb < 8 {
                            ttmb = -1;
                        }
                        first_block = false;
                    }
                }
            } else {
                // 跳过：整块 MV 预测 + 运动补偿，没有系数。
                for i in 0..6usize {
                    let bi = self.blk(mbx, mby, i);
                    self.grids.mb_type[bi] = 0;
                    self.set_dc(mbx, mby, i, 0);
                }
                self.mbp.qscale[mb_pos] = 0;
                self.pred_mv(mbx, mby, 0, 0, 0, true, false)?;
                self.mc_1mv(mbx, mby);
            }
        } else {
            // ---- 4MV ----
            if !skipped {
                let mut intra_count = 0i32;
                let mut coded_inter = false;
                let mut is_intra = [false; 6];
                let mut is_coded = [false; 6];
                let cbp = vlc_p_cbpcy(cbptab as usize).decode(&mut self.r)? as u32;
                for i in 0..6usize {
                    let val = (cbp >> (5 - i)) & 1;
                    self.set_dc(mbx, mby, i, 0);
                    if i < 4 {
                        let (mut dmv_x, mut dmv_y) = (0i32, 0i32);
                        let mut mb_intra = false;
                        let mut mb_has_coeffs = false;
                        if val != 0 {
                            let (x, y, intra, has) = self.get_mvdata()?;
                            dmv_x = x;
                            dmv_y = y;
                            mb_intra = intra;
                            mb_has_coeffs = has;
                        }
                        self.pred_mv(mbx, mby, i, dmv_x, dmv_y, false, mb_intra)?;
                        if !mb_intra {
                            self.mc_4mv_luma(mbx, mby, i);
                        }
                        intra_count += mb_intra as i32;
                        is_intra[i] = mb_intra;
                        is_coded[i] = mb_has_coeffs;
                    }
                    if i & 4 != 0 {
                        is_intra[i] = intra_count >= 3;
                        is_coded[i] = val != 0;
                    }
                    if i == 4 {
                        self.mc_4mv_chroma(mbx, mby);
                    }
                    let bi = self.blk(mbx, mby, i);
                    self.grids.mb_type[bi] = is_intra[i] as u8;
                    if !coded_inter {
                        coded_inter = !is_intra[i] && is_coded[i];
                    }
                }
                if intra_count == 0 && !coded_inter {
                    // 没有任何编码块：宏块层到此为止。
                    self.mbp.cbp[mb_pos] = 0;
                    self.mbp.ttblk[mb_pos] = 0;
                    self.mbp.is_intra[mb_pos] = 0;
                    return Ok(());
                }
                mquant = self.get_mquant(mquant, mbx, mby)?;
                self.mbp.qscale[mb_pos] = mquant;

                // 帧内块有没有可用邻居 -> 决定 ac_pred 这一位读不读。
                let mut intrapred = false;
                for i in 0..6usize {
                    if is_intra[i] {
                        let bi = self.blk(mbx, mby, i);
                        let wrap = self.block_wrap(i);
                        let a_ok = (mby != 0 || i == 2 || i == 3)
                            && self.grids.mb_type[bi - wrap] != 0;
                        let c_ok = (mbx != 0 || i == 1 || i == 3)
                            && self.grids.mb_type[bi - 1] != 0;
                        if a_ok || c_ok {
                            intrapred = true;
                            break;
                        }
                    }
                }
                let ac_pred = if intrapred { self.r.bit()? != 0 } else { false };

                let mut ttmb: i32 = tt as i32;
                if !ttmbf && coded_inter {
                    ttmb = vlc_ttmb(self.tt_index).decode(&mut self.r)? as i32;
                }
                let mut first_block = true;
                for i in 0..6usize {
                    if is_intra[i] {
                        self.set_avail(mbx, mby, i);
                        let block =
                            self.decode_intra_block(mbx, mby, i, is_coded[i], mquant, ac_pred)?;
                        self.put_block(mbx, mby, i, &block, true);
                        block_cbp |= 0xF << (i << 2);
                        block_intra |= 1 << i;
                    } else if is_coded[i] {
                        let (pat, tt) =
                            self.decode_p_block(mbx, mby, i, mquant, ttmb, first_block)?;
                        block_cbp |= pat << (i << 2);
                        block_tt |= tt;
                        if !ttmbf && ttmb < 8 {
                            ttmb = -1;
                        }
                        first_block = false;
                    }
                }
            } else {
                for i in 0..6usize {
                    let bi = self.blk(mbx, mby, i);
                    self.grids.mb_type[bi] = 0;
                    self.set_dc(mbx, mby, i, 0);
                }
                for i in 0..4usize {
                    self.pred_mv(mbx, mby, i, 0, 0, false, false)?;
                    self.mc_4mv_luma(mbx, mby, i);
                }
                self.mc_4mv_chroma(mbx, mby);
                self.mbp.qscale[mb_pos] = 0;
            }
        }

        self.mbp.cbp[mb_pos] = block_cbp;
        self.mbp.ttblk[mb_pos] = block_tt;
        self.mbp.is_intra[mb_pos] = block_intra;
        Ok(())
    }

    /// `s->block_wrap[n]`：亮度块是 b8 行距，色度块是宏块行距。
    pub(crate) fn block_wrap(&self, n: usize) -> usize {
        if n < 4 {
            self.b8_stride
        } else {
            self.mb_stride
        }
    }

    /// P 帧帧内块的邻居可用性（`v->a_avail` / `v->c_avail`）。
    fn set_avail(&mut self, mbx: usize, mby: usize, i: usize) {
        let mut a_avail = false;
        let mut c_avail = false;
        let bi = self.blk(mbx, mby, i);
        let wrap = self.block_wrap(i);
        if i == 2 || i == 3 || mby != 0 {
            a_avail = self.grids.mb_type[bi - wrap] != 0;
        }
        if i == 1 || i == 3 || mbx != 0 {
            c_avail = self.grids.mb_type[bi - 1] != 0;
        }
        self.a_avail = a_avail;
        self.c_avail = c_avail;
    }

    // ================= 块 =================

    /// `vc1_decode_ac_coeff()`：解一个 AC 系数（`(last, run, level)`）。
    fn decode_ac_coeff(&mut self, codingset: usize) -> Result<(bool, u32, i32)> {
        let last_sym = tables::AC_SIZES[codingset] as usize - 1;
        let index = vlc_ac(codingset).decode(&mut self.r)?;
        let (run, level, last, sign);
        if index != last_sym {
            run = tables::AC_INDEX_DECODE[codingset][index][0];
            level = tables::AC_INDEX_DECODE[codingset][index][1] as i32;
            last = index >= tables::AC_LAST_DECODE[codingset] as usize;
            sign = self.r.bit()?;
        } else {
            let escape = self.decode210()?;
            if escape != 2 {
                let index = vlc_ac(codingset).decode(&mut self.r)?;
                if index >= last_sym {
                    return Err(Error::Bad("vc1: AC escape 的第二个下标越界".into()));
                }
                let mut r0 = tables::AC_INDEX_DECODE[codingset][index][0];
                let mut l0 = tables::AC_INDEX_DECODE[codingset][index][1] as i32;
                let lst = index >= tables::AC_LAST_DECODE[codingset] as usize;
                if escape == 0 {
                    let d = if lst {
                        tables::AC_LAST_DELTA_LEVEL[codingset][r0 as usize]
                    } else {
                        tables::AC_DELTA_LEVEL[codingset][r0 as usize]
                    };
                    l0 += d as i32;
                } else {
                    let d = if lst {
                        tables::AC_LAST_DELTA_RUN[codingset][l0 as usize]
                    } else {
                        tables::AC_DELTA_RUN[codingset][l0 as usize]
                    };
                    r0 += d as u32 + 1;
                }
                run = r0;
                level = l0;
                last = lst;
                sign = self.r.bit()?;
            } else {
                last = self.r.bit()? != 0;
                if self.esc3_level_length == 0 {
                    if self.hdr.pq < 8 || self.hdr.dquant.is_some() {
                        // 表 59
                        self.esc3_level_length = self.r.u(3)?;
                        if self.esc3_level_length == 0 {
                            self.esc3_level_length = self.r.u(2)? + 8;
                        }
                    } else {
                        // 表 60
                        self.esc3_level_length = self.get_unary(1, 6)? + 2;
                    }
                    self.esc3_run_length = 3 + self.r.u(2)?;
                }
                run = self.r.u(self.esc3_run_length)?;
                sign = self.r.bit()?;
                level = self.r.u(self.esc3_level_length)? as i32;
            }
        }
        Ok((last, run, (level ^ -(sign as i32)) + sign as i32))
    }

    fn decode210(&mut self) -> Result<u32> {
        if self.r.bit()? != 0 {
            Ok(0)
        } else {
            Ok(2 - self.r.bit()?)
        }
    }

    fn decode012(&mut self) -> Result<u32> {
        if self.r.bit()? == 0 {
            Ok(0)
        } else {
            Ok(self.r.bit()? + 1)
        }
    }

    /// `get_unary(stop, len)`：读到 `stop` 或读满 `len` 位为止，返回**没读到 stop 的位数**。
    fn get_unary(&mut self, stop: u32, len: u32) -> Result<u32> {
        let mut i = 0;
        while i < len {
            if self.r.bit()? == stop {
                break;
            }
            i += 1;
        }
        Ok(i)
    }

    /// I 帧的帧内块（`vc1_decode_i_block()`）：DC 走 `vc1_i_pred_dc`，AC 扫描表按 `ac_pred` 选。
    fn decode_i_block(
        &mut self,
        mbx: usize,
        mby: usize,
        n: usize,
        coded: bool,
        ac_pred: bool,
    ) -> Result<[i16; 64]> {
        let mut block = [0i16; 64];
        let pq = self.hdr.pq as i32;
        let halfpq = self.hdr.halfpq as i32;
        let pquantizer = self.hdr.pquantizer;

        // ---- DC ----
        let mut dcdiff = vlc_dc(self.hdr.dc_table_index as usize, n >= 4).decode(&mut self.r)? as i32;
        if dcdiff != 0 {
            let m = if pq == 1 || pq == 2 { 3 - pq } else { 0 };
            if dcdiff == 119 {
                dcdiff = self.r.u((8 + m) as u32)? as i32;
            } else if m != 0 {
                dcdiff = (dcdiff << m) + self.r.u(m as u32)? as i32 - ((1 << m) - 1);
            }
            if self.r.bit()? != 0 {
                dcdiff = -dcdiff;
            }
        }
        let (pred, dir) = self.i_pred_dc(mbx, mby, n);
        dcdiff += pred;
        self.set_dc(mbx, mby, n, dcdiff as i16);
        let scale = if n < 4 { self.y_dc_scale } else { self.c_dc_scale };
        block[0] = (dcdiff * scale) as i16;

        let ac_scale = pq * 2 + halfpq;
        let (base, wrap16) = self.ac_index(mbx, mby, n);
        let mut i = coded as i32;

        if coded {
            let zz: &[u8; 64] = if ac_pred {
                if !dir {
                    &self.scans.zz_8x8[2]
                } else {
                    &self.scans.zz_8x8[3]
                }
            } else {
                &self.scans.zz_8x8[1]
            };
            // ⚠️ 色度块走的是 `codingset2`（上游 `(k < 4) ? codingset : codingset2`）——
            // 两套码表在有些帧里恰好解出同一串符号，所以这条错了会「大部分时候看着对」。
            let cs = if n < 4 { self.codingset } else { self.codingset2 };
            let mut last = false;
            while !last {
                let (l, skip, value) = self.decode_ac_coeff(cs)?;
                last = l;
                i += skip as i32;
                if i > 63 {
                    break;
                }
                block[zz[i as usize] as usize] = value as i16;
                i += 1;
            }
            if ac_pred {
                // 左：取邻居的 [1..8)；上：取邻居的 [8..16)。
                let nb = if dir { base - 16 } else { base - wrap16 };
                let sh = if dir { 0 } else { 3 };
                let ac = self.ac_plane(n);
                for k in 1..8usize {
                    let src = if dir { ac[nb + k] } else { ac[nb + 8 + k] };
                    block[k << sh] = (block[k << sh] as i32 + src as i32) as i16;
                }
            }
            // 存回本块（供后面的块预测）。
            {
                let ac = self.ac_plane(n);
                for k in 1..8usize {
                    ac[base + k] = block[k]; // left_blk_sh = 0
                    ac[base + 8 + k] = block[k * 8]; // top_blk_sh = 3
                }
            }
            // 反量化
            for k in 1..64usize {
                if block[k] != 0 {
                    block[k] = (block[k] as i32 * ac_scale) as i16;
                    if !pquantizer {
                        let add = if block[k] < 0 { -pq } else { pq };
                        block[k] = (block[k] as i32 + add) as i16;
                    }
                }
            }
        } else {
            let ac = self.ac_plane(n);
            for k in 0..16usize {
                ac[base + k] = 0;
            }
            if ac_pred {
                let sh = if dir { 0 } else { 3 };
                let (dst_off, src_off) = if dir {
                    (base, base - 16)
                } else {
                    (base + 8, base - wrap16 + 8)
                };
                for k in 0..8usize {
                    ac[dst_off + k] = ac[src_off + k];
                }
                for k in 1..8usize {
                    let v = (ac[src_off + k] as i32 * ac_scale) as i16;
                    block[k << sh] = v;
                    if !pquantizer && v != 0 {
                        let add = if v < 0 { -pq } else { pq };
                        block[k << sh] = (v as i32 + add) as i16;
                    }
                }
            }
        }
        Ok(block)
    }

    /// P 帧的帧内块（`vc1_decode_intra_block()`）：DC 走 `ff_vc1_pred_dc`（带跨 pq 换算），
    /// AC 恒用 `zz_8x8[0]`，而且 AC 预测也要按 `qscale` 换算。
    fn decode_intra_block(
        &mut self,
        mbx: usize,
        mby: usize,
        n: usize,
        coded: bool,
        mquant: i32,
        ac_pred: bool,
    ) -> Result<[i16; 64]> {
        let mut block = [0i16; 64];
        let halfpq = self.hdr.halfpq as i32;
        let pquantizer = self.hdr.pquantizer;
        let mut quant = mquant.abs();
        quant = quant.min(31); // av_clip_uintp2(quant, 5)
        self.y_dc_scale = tables::DC_SCALE[quant as usize] as i32;
        self.c_dc_scale = self.y_dc_scale;

        // ---- DC ----
        let mut dcdiff = vlc_dc(self.hdr.dc_table_index as usize, n >= 4).decode(&mut self.r)? as i32;
        if dcdiff != 0 {
            let m = if quant == 1 || quant == 2 { 3 - quant } else { 0 };
            if dcdiff == 119 {
                dcdiff = self.r.u((8 + m) as u32)? as i32;
            } else if m != 0 {
                dcdiff = (dcdiff << m) + self.r.u(m as u32)? as i32 - ((1 << m) - 1);
            }
            if self.r.bit()? != 0 {
                dcdiff = -dcdiff;
            }
        }
        let (pred, mut dir) = self.p_pred_dc(mbx, mby, n, quant);
        dcdiff += pred;
        self.set_dc(mbx, mby, n, dcdiff as i16);
        block[0] = (dcdiff * self.y_dc_scale) as i16;

        let mut use_pred = ac_pred;
        if !self.a_avail {
            dir = true;
        }
        if !self.c_avail {
            dir = false;
        }
        if !self.a_avail && !self.c_avail {
            use_pred = false;
        }
        let ac_scale = quant * 2 + if mquant < 0 { 0 } else { halfpq };
        let (base, wrap16) = self.ac_index(mbx, mby, n);

        // AC 预测的换算：两边量化步长不同时按 `dqscale` 缩放。
        let q1 = self.mbp.qscale[self.mbidx(mbx, mby)];
        let mut q2 = 0i32;
        if dir && self.c_avail && mbx > 0 {
            q2 = self.mbp.qscale[self.mbidx(mbx, mby) - 1];
        }
        if !dir && self.a_avail && mby > 0 {
            q2 = self.mbp.qscale[self.mbidx(mbx, mby) - self.mb_stride];
        }
        if (dir && n == 1) || (!dir && n == 2) || n == 3 {
            q2 = q1;
        }

        if coded {
            let zz = &self.scans.zz_8x8[0];
            let cs = if n < 4 { self.codingset } else { self.codingset2 };
            let mut i = 1i32;
            let mut last = false;
            while !last {
                let (l, skip, value) = self.decode_ac_coeff(cs)?;
                last = l;
                i += skip as i32;
                if i > 63 {
                    break;
                }
                block[zz[i as usize] as usize] = value as i16;
                i += 1;
            }
            if use_pred {
                let q1s = q1.abs() * 2 + if q1 < 0 { 0 } else { halfpq } - 1;
                if q1s < 1 {
                    return Err(Error::Bad("vc1: AC 预测的量化步长非法".into()));
                }
                let q2s = if q2 != 0 {
                    q2.abs() * 2 + if q2 < 0 { 0 } else { halfpq } - 1
                } else {
                    0
                };
                let nb = if dir { base - 16 } else { base - wrap16 };
                let sh = if dir { 0 } else { 3 };
                {
                    let ac = self.ac_plane(n);
                    for k in 1..8usize {
                        let src = if dir { ac[nb + k] } else { ac[nb + 8 + k] };
                        let v = if q2s != 0 && q1s != q2s {
                            (src as i32 * q2s * tables::DQSCALE[(q1s - 1) as usize] as i32
                                + 0x20000)
                                >> 18
                        } else {
                            src as i32
                        };
                        block[k << sh] = (block[k << sh] as i32 + v) as i16;
                    }
                }
            }
            {
                let ac = self.ac_plane(n);
                for k in 1..8usize {
                    ac[base + k] = block[k];
                    ac[base + 8 + k] = block[k * 8];
                }
            }
            for k in 1..64usize {
                if block[k] != 0 {
                    block[k] = (block[k] as i32 * ac_scale) as i16;
                    if !pquantizer {
                        let add = if block[k] < 0 { -quant } else { quant };
                        block[k] = (block[k] as i32 + add) as i16;
                    }
                }
            }
        } else {
            let ac = self.ac_plane(n);
            for k in 0..16usize {
                ac[base + k] = 0;
            }
            if use_pred {
                let (dst_off, src_off, sh) = if dir {
                    (base, base - 16, 0)
                } else {
                    (base + 8, base - wrap16 + 8, 3)
                };
                for k in 0..8usize {
                    ac[dst_off + k] = ac[src_off + k];
                }
                let q1s = q1.abs() * 2 + if q1 < 0 { 0 } else { halfpq } - 1;
                if q1s < 1 {
                    return Err(Error::Bad("vc1: AC 预测的量化步长非法".into()));
                }
                let q2s = if q2 != 0 {
                    q2.abs() * 2 + if q2 < 0 { 0 } else { halfpq } - 1
                } else {
                    0
                };
                for k in 1..8usize {
                    if q2s != 0 && q1s != q2s {
                        ac[dst_off + k] = ((ac[dst_off + k] as i32
                            * q2s
                            * tables::DQSCALE[(q1s - 1) as usize] as i32
                            + 0x20000)
                            >> 18) as i16;
                    }
                    let v = (ac[dst_off + k] as i32 * ac_scale) as i16;
                    block[k << sh] = v;
                    if !pquantizer && v != 0 {
                        let add = if v < 0 { -quant } else { quant };
                        block[k << sh] = (v as i32 + add) as i16;
                    }
                }
            }
        }
        Ok(block)
    }

    /// `vc1_decode_p_block()`：帧间块的残差（含 ABT 的四种变换）。
    ///
    /// 返回 `(pat, ttblk)`：`pat` 是「这个 8×8 里哪些 4×4 有系数」（环路滤波要用），
    /// `ttblk` 是实际的变换类型（4 位）。
    fn decode_p_block(
        &mut self,
        mbx: usize,
        mby: usize,
        n: usize,
        mquant: i32,
        ttmb: i32,
        first_block: bool,
    ) -> Result<(u32, u32)> {
        // ⚠️ 只取这一个标量。原来是 `self.hdr.p...clone()` —— 这个函数**每块**都进一次
        // （一个宏块 6 次，ABT 更多），克隆会把两张位平面一起拷，代价是 malloc + 几 KB memcpy。
        let ttmbf = self.hdr.p.as_ref().unwrap().ttmbf;
        let pquantizer = self.hdr.pquantizer;
        let halfpq = self.hdr.halfpq as i32;
        let mut block = [0i16; 64];
        let mut ttblk = (ttmb & 7) as u32;
        let mut subblkpat = 0u32;
        if ttmb == -1 {
            let idx = vlc_ttblk(self.tt_index).decode(&mut self.r)?;
            ttblk = tables::TTBLK_TO_TT[self.tt_index][idx] as u32;
        }
        if ttblk == tables::TT_4X4 as u32 {
            subblkpat = !(vlc_subblkpat(self.tt_index).decode(&mut self.r)? as u32 + 1);
        }
        if ttblk != tables::TT_8X8 as u32
            && ttblk != tables::TT_4X4 as u32
            && (ttmbf || (ttmb != -1 && (ttmb & 8) != 0 && !first_block)
                || (!self.seq.res_rtm_flag && !first_block))
        {
            subblkpat = self.decode012()?;
            if subblkpat != 0 {
                subblkpat ^= 3;
            }
            if ttblk == tables::TT_8X4_TOP as u32 || ttblk == tables::TT_8X4_BOTTOM as u32 {
                ttblk = tables::TT_8X4 as u32;
            }
            if ttblk == tables::TT_4X8_RIGHT as u32 || ttblk == tables::TT_4X8_LEFT as u32 {
                ttblk = tables::TT_4X8 as u32;
            }
        }
        let quant = mquant.abs();
        let scale = quant * 2 + if mquant < 0 { 0 } else { halfpq };

        // 8×4 / 4×8 的「上下 / 左右」要先化成通用的「两半」。
        if ttblk == tables::TT_8X4_TOP as u32 || ttblk == tables::TT_8X4_BOTTOM as u32 {
            subblkpat = 2 - (ttblk == tables::TT_8X4_TOP as u32) as u32;
            ttblk = tables::TT_8X4 as u32;
        }
        if ttblk == tables::TT_4X8_RIGHT as u32 || ttblk == tables::TT_4X8_LEFT as u32 {
            subblkpat = 2 - (ttblk == tables::TT_4X8_LEFT as u32) as u32;
            ttblk = tables::TT_4X8 as u32;
        }

        // 像素目标：亮度块按 2×2 摆，色度块各一个 8×8。
        let (plane, px, py) = self.block_origin(mbx, mby, n);
        let stride = self.plane_stride(plane);
        let pat;

        match ttblk {
            t if t == tables::TT_8X8 as u32 => {
                pat = 0xF;
                let mut i = 0i32;
                let mut last = false;
                while !last {
                    let (l, skip, value) = self.decode_ac_coeff(self.codingset2)?;
                    last = l;
                    i += skip as i32;
                    if i > 63 {
                        break;
                    }
                    let idx = self.scans.zz_8x8[0][i as usize] as usize;
                    i += 1;
                    block[idx] = (value * scale) as i16;
                    if !pquantizer {
                        let add = if block[idx] < 0 { -quant } else { quant };
                        block[idx] = (block[idx] as i32 + add) as i16;
                    }
                }
                let dst = self.plane_mut(plane);
                if i == 1 {
                    idct::inv_trans_8x8_dc(dst, stride, &block, px, py);
                } else {
                    idct::inv_trans_8x8(&mut block);
                    idct::add_pixels_clamped(&block, dst, stride, px, py);
                }
            }
            t if t == tables::TT_4X4 as u32 => {
                pat = !subblkpat & 0xF;
                for j in 0..4usize {
                    let mut last = subblkpat & (1 << (3 - j)) != 0;
                    let mut i = 0i32;
                    let off = (j & 1) * 4 + (j & 2) * 16;
                    while !last {
                        let (l, skip, value) = self.decode_ac_coeff(self.codingset2)?;
                        last = l;
                        i += skip as i32;
                        if i > 15 {
                            break;
                        }
                        let idx = self.scans.zz_4x4[i as usize] as usize;
                        i += 1;
                        block[idx + off] = (value * scale) as i16;
                        if !pquantizer {
                            let add = if block[idx + off] < 0 { -quant } else { quant };
                            block[idx + off] = (block[idx + off] as i32 + add) as i16;
                        }
                    }
                    if !(subblkpat & (1 << (3 - j)) != 0) {
                        let dst = self.plane_mut(plane);
                        let (ox, oy) = (px + (j & 1) * 4, py + (j & 2) * 2);
                        if i == 1 {
                            idct::inv_trans_4x4_dc(dst, stride, &block[off..], ox, oy);
                        } else {
                            idct::inv_trans_4x4(dst, stride, &mut block[off..], ox, oy);
                        }
                    }
                }
            }
            t if t == tables::TT_8X4 as u32 => {
                pat = !((subblkpat & 2) * 6 + (subblkpat & 1) * 3) & 0xF;
                for j in 0..2usize {
                    let mut last = subblkpat & (1 << (1 - j)) != 0;
                    let mut i = 0i32;
                    let off = j * 32;
                    while !last {
                        let (l, skip, value) = self.decode_ac_coeff(self.codingset2)?;
                        last = l;
                        i += skip as i32;
                        if i > 31 {
                            break;
                        }
                        let idx = self.scans.zz_8x4[i as usize] as usize + off;
                        i += 1;
                        block[idx] = (value * scale) as i16;
                        if !pquantizer {
                            let add = if block[idx] < 0 { -quant } else { quant };
                            block[idx] = (block[idx] as i32 + add) as i16;
                        }
                    }
                    if !(subblkpat & (1 << (1 - j)) != 0) {
                        let dst = self.plane_mut(plane);
                        let (ox, oy) = (px, py + j * 4);
                        if i == 1 {
                            idct::inv_trans_8x4_dc(dst, stride, &block[off..], ox, oy);
                        } else {
                            idct::inv_trans_8x4(dst, stride, &mut block[off..], ox, oy);
                        }
                    }
                }
            }
            _ => {
                // TT_4X8
                pat = !(subblkpat * 5) & 0xF;
                for j in 0..2usize {
                    let mut last = subblkpat & (1 << (1 - j)) != 0;
                    let mut i = 0i32;
                    let off = j * 4;
                    while !last {
                        let (l, skip, value) = self.decode_ac_coeff(self.codingset2)?;
                        last = l;
                        i += skip as i32;
                        if i > 31 {
                            break;
                        }
                        let idx = self.scans.zz_4x8[i as usize] as usize + off;
                        i += 1;
                        block[idx] = (value * scale) as i16;
                        if !pquantizer {
                            let add = if block[idx] < 0 { -quant } else { quant };
                            block[idx] = (block[idx] as i32 + add) as i16;
                        }
                    }
                    if !(subblkpat & (1 << (1 - j)) != 0) {
                        let dst = self.plane_mut(plane);
                        let (ox, oy) = (px + j * 4, py);
                        if i == 1 {
                            idct::inv_trans_4x8_dc(dst, stride, &block[off..], ox, oy);
                        } else {
                            idct::inv_trans_4x8(dst, stride, &mut block[off..], ox, oy);
                        }
                    }
                }
            }
        }
        Ok((pat, ttblk << (n * 4)))
    }

    /// 块 `n` 的像素目标：`(平面, x, y)`。
    pub(crate) fn block_origin(&self, mbx: usize, mby: usize, n: usize) -> (usize, usize, usize) {
        if n < 4 {
            (0, mbx * 16 + (n & 1) * 8, mby * 16 + (n & 2) * 4)
        } else {
            (n - 3, mbx * 8, mby * 8)
        }
    }

    /// 把解出来的块写进像素。
    ///
    /// `signed` 对应上游 `vc1_put_blocks_clamped()` 的 `put_signed` 参数：
    /// I 帧是 `false`（块值当像素）、P 帧的帧内块是 `true`（块值 + 128）。
    fn put_block(&mut self, mbx: usize, mby: usize, n: usize, block: &[i16; 64], signed: bool) {
        let mut b = *block;
        idct::inv_trans_8x8(&mut b);
        let (plane, px, py) = self.block_origin(mbx, mby, n);
        let stride = self.plane_stride(plane);
        let dst = self.plane_mut(plane);
        if signed {
            idct::put_signed_pixels_clamped(&b, dst, stride, px, py);
        } else {
            idct::put_pixels_clamped(&b, dst, stride, px, py);
        }
    }

    // ================= 预测网格 =================

    /// 当前块在 AC 网格上的基址与行距（`i16` 计）。
    fn ac_index(&self, mbx: usize, mby: usize, n: usize) -> (usize, usize) {
        if n < 4 {
            (self.blk(mbx, mby, n) * 16, self.b8_stride * 16)
        } else {
            // 色度区前面留一个块（见 `Grids::ac_chroma` 的注释）。
            (16 + self.cblk(mbx, mby, n - 4) * 16, self.mb_stride * 16)
        }
    }

    fn ac_plane(&mut self, n: usize) -> &mut [i16] {
        if n < 4 {
            &mut self.grids.ac_luma
        } else {
            &mut self.grids.ac_chroma
        }
    }

    /// `vc1_coded_block_pred()`：`pred = (b == c) ? a : c`（a=左、b=左上、c=上）。
    fn coded_block_pred(&self, mbx: usize, mby: usize, n: usize) -> u32 {
        let xy = self.blk(mbx, mby, n);
        let wrap = self.b8_stride;
        let a = self.grids.coded_block[xy - 1] as u32;
        let b = self.grids.coded_block[xy - 1 - wrap] as u32;
        let c = self.grids.coded_block[xy - wrap] as u32;
        if b == c {
            a
        } else {
            c
        }
    }

    /// `vc1_i_pred_dc()`：返回 `(预测值, 方向)`。方向 `true` = 从左（`left_blk_sh`）。
    fn i_pred_dc(&self, mbx: usize, mby: usize, n: usize) -> (i32, bool) {
        let scale = if n < 4 { self.y_dc_scale } else { self.c_dc_scale } as usize;
        let (mut a, mut b, mut c) = self.dc_abc(mbx, mby, n);
        // 本实现 `overlap` 恒为 false -> 走 `pq < 9 || !overlap` 那一支。
        if mby == 0 && !(n == 2 || n == 3) {
            b = DCPRED[scale] as i16;
            a = b;
        }
        if mbx == 0 && !(n == 1 || n == 3) {
            b = DCPRED[scale] as i16;
            c = b;
        }
        if (a as i32 - b as i32).abs() <= (b as i32 - c as i32).abs() {
            (c as i32, true)
        } else {
            (a as i32, false)
        }
    }

    /// `ff_vc1_pred_dc()`（P 帧的帧内块）：带跨量化步长的换算与可用性判断。
    fn p_pred_dc(&self, mbx: usize, mby: usize, n: usize, quant: i32) -> (i32, bool) {
        let mb_pos = self.mbidx(mbx, mby);
        let q1 = self.mbp.qscale[mb_pos].abs();
        let dqscale_index = tables::DC_SCALE[q1.min(31) as usize] as i32 - 1;
        if dqscale_index < 0 {
            return (0, true);
        }
        let (a0, b0, c0) = self.dc_abc(mbx, mby, n);
        let mut a = a0 as i32;
        let mut b = b0 as i32;
        let mut c = c0 as i32;
        let _ = quant;
        if self.c_avail && !(n == 1 || n == 3) {
            let q2 = self.mbp.qscale[mb_pos - 1].abs();
            if q2 != 0 && q2 != q1 {
                c = ((c as u32)
                    .wrapping_mul(tables::DC_SCALE[q2.min(31) as usize] as u32)
                    .wrapping_mul(tables::DQSCALE[dqscale_index as usize])
                    + 0x20000
                    >> 18) as i32;
            }
        }
        if self.a_avail && !(n == 2 || n == 3) {
            let q2 = self.mbp.qscale[mb_pos - self.mb_stride].abs();
            if q2 != 0 && q2 != q1 {
                a = ((a as u32)
                    .wrapping_mul(tables::DC_SCALE[q2.min(31) as usize] as u32)
                    .wrapping_mul(tables::DQSCALE[dqscale_index as usize])
                    + 0x20000
                    >> 18) as i32;
            }
        }
        if self.a_avail && self.c_avail && n != 3 {
            let mut off = mb_pos;
            if n != 1 {
                off -= 1;
            }
            if n != 2 {
                off -= self.mb_stride;
            }
            let q2 = self.mbp.qscale[off].abs();
            if q2 != 0 && q2 != q1 {
                b = ((b as u32)
                    .wrapping_mul(tables::DC_SCALE[q2.min(31) as usize] as u32)
                    .wrapping_mul(tables::DQSCALE[dqscale_index as usize])
                    + 0x20000
                    >> 18) as i32;
            }
        }
        if self.c_avail && (!self.a_avail || (a - b).abs() <= (b - c).abs()) {
            (c, true)
        } else if self.a_avail {
            (a, false)
        } else {
            (0, true)
        }
    }

    /// 邻居三元组 `(上 a, 左上 b, 左 c)`。越界时读 0（上游那几处**只在可用时才读**，
    /// 所以这里的 0 不会被用到）。
    fn dc_abc(&self, mbx: usize, mby: usize, n: usize) -> (i16, i16, i16) {
        let (base, wrap) = if n < 4 {
            (self.blk(mbx, mby, n), self.b8_stride)
        } else {
            (self.cblk(mbx, mby, n - 4), self.mb_stride)
        };
        let get = |off: isize| -> i16 {
            let idx = base as isize + off;
            if idx < 0 {
                return 0;
            }
            let g = if n < 4 {
                &self.grids.dc_luma
            } else {
                &self.grids.dc_chroma
            };
            *g.get(idx as usize).unwrap_or(&0)
        };
        (get(-(wrap as isize)), get(-(wrap as isize) - 1), get(-1))
    }

    fn set_dc(&mut self, mbx: usize, mby: usize, n: usize, v: i16) {
        if n < 4 {
            let idx = self.blk(mbx, mby, n);
            self.grids.dc_luma[idx] = v;
        } else {
            let idx = self.cblk(mbx, mby, n - 4);
            self.grids.dc_chroma[idx] = v;
        }
    }
}

/// P 帧的 `for (i = 0; i < 6; i++) s->dc_val[0][s->block_index[i]] = 0;`：
/// 帧间块会把 DC 预测值清零（**只清亮度那一份，色度不清** —— 上游也是这样）。
const _: () = ();

