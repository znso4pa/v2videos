//! MV 预测（`ff_vc1_pred_mv()`，渐进式那一条路）。
//!
//! 上游那个函数里有一大半是**隔行**（field / frame 两种交织模式）的分支，
//! 本实现只做渐进式，所以这里只剩：邻居取用、pullback、hybrid 预测、按 MV 范围取模。
//!
//! ⚠️ 三处容易漏：
//! - **`dmv` 要先按 `quarter_sample` 放大两倍**（半像素模式下差分是半像素单位）。
//! - **hybrid 预测会读位流**（`get_bits1`）—— 它不是纯函数，位游标会被它推进。
//! - **B 邻居的 `off` 逐块不同**（1MV 与 4MV 两套），而且 `!res_rtm_flag` 时
//!   首列还有一个特例偏移。

use v2v_msmpeg4::{mid_pred, Result};

use crate::decoder::Ctx;
use crate::header::MvMode;

impl Ctx<'_, '_> {
    /// `ff_vc1_pred_mv(v, n, dmv_x, dmv_y, mv1, ...)`。
    ///
    /// `mv1` = 这个宏块是「一个 MV 管四个块」（要把它复制到四个块上）；
    /// `mb_intra` = 当前块是帧内（MV 全部清 0 并返回）。
    pub(crate) fn pred_mv(
        &mut self,
        mbx: usize,
        mby: usize,
        n: usize,
        dmv_x: i32,
        dmv_y: i32,
        mv1: bool,
        mb_intra: bool,
    ) -> Result<()> {
        // ⚠️ 只取**标量**，不要 `hdr.p.clone()`：`PFrameHeader` 里带着两张位平面
        // （各一个 `Vec<u8>`），而这个函数**每个块**都进一次 —— 克隆一次就是一次
        // malloc + 几 KB memcpy，实测占掉解码时间的 10% 上下。
        let (qs, mixedmv_pic) = {
            let p = self.hdr.p.as_ref().unwrap();
            (
                p.quarter_sample,
                p.mv_mode == MvMode::MixedMv
                    || (p.mv_mode == MvMode::IntensityComp && p.mv_mode2 == Some(MvMode::MixedMv)),
            )
        };
        let (mut dmv_x, mut dmv_y) = (dmv_x, dmv_y);
        // 差分按**四分之一像素**为单位换算。
        if !qs {
            dmv_x *= 2;
            dmv_y *= 2;
        }
        let wrap = self.b8_stride as isize;
        let xy = self.blk(mbx, mby, n) as isize;

        if mb_intra {
            self.grids.mv[xy as usize] = [0, 0];
            if mv1 {
                for off in [1isize, wrap, wrap + 1] {
                    self.grids.mv[(xy + off) as usize] = [0, 0];
                }
                let mp = self.mbidx(mbx, mby);
                self.grids.luma_mv[mp] = [0, 0];
            }
            return Ok(());
        }

        // `mixedmv_pic` 在函数开头就算好了（见 `qs` 那段）。

        let a_valid = mby != 0 || (n == 2 || n == 3);
        let mut b_valid = a_valid;
        let c_valid = mbx != 0 || (n == 1 || n == 3);
        let off: isize;
        if mv1 {
            let _ = mixedmv_pic;
            off = if mbx == self.mb_width - 1 { -1 } else { 2 };
            b_valid = b_valid && self.mb_width > 1;
        } else {
            off = match n {
                0 => {
                    if mbx != 0 {
                        -1
                    } else if self.seq.res_rtm_flag {
                        1
                    } else {
                        2 * self.mb_width as isize - wrap - 1
                    }
                }
                1 => {
                    if mbx == self.mb_width - 1 {
                        -1
                    } else {
                        1
                    }
                }
                2 => 1,
                _ => -1,
            };
        }

        let mv_at = |i: isize| -> (i32, i32) {
            let m = self.grids.mv[i as usize];
            (m[0] as i32, m[1] as i32)
        };
        let (mut px, mut py);
        let fa = if a_valid { mv_at(xy - wrap) } else { (0, 0) };
        let fb = if b_valid { mv_at(xy - wrap + off) } else { (0, 0) };
        let fc = if c_valid { mv_at(xy - 1) } else { (0, 0) };
        if a_valid {
            px = fa.0;
            py = fa.1;
        } else if c_valid {
            px = fc.0;
            py = fc.1;
        } else if b_valid {
            px = fb.0;
            py = fb.1;
        } else {
            px = 0;
            py = 0;
        }

        // ⚠️ **两个以上邻居可用时改用中值**。上游那两个计数器（`num_samefield` /
        // `num_oppfield`）**不在 `field_mode` 的 if 里**，所以渐进式下它们就是
        // 「可用邻居的个数」—— 少了这一步，跨宏块的 MV 预测会整体偏（而且偏得不显眼）。
        if (a_valid as i32) + (b_valid as i32) + (c_valid as i32) > 1 {
            px = mid_pred(fa.0, fb.0, fc.0);
            py = mid_pred(fa.1, fb.1, fc.1);
        }

        // ---- pullback（8.3.5.3.4）----
        let big = if mv1 { -60 } else { -28 };
        let qx = (mbx as i32) << 6;
        let qx = qx + if n == 1 || n == 3 { 32 } else { 0 };
        let qy = (mby as i32) << 6;
        let qy = qy + if n == 2 || n == 3 { 32 } else { 0 };
        let x_max = ((self.mb_width as i32) << 6) - 4;
        let y_max = ((self.mb_height as i32) << 6) - 4;
        if qx + px < big {
            px = big - qx;
        }
        if qy + py < big {
            py = big - qy;
        }
        if qx + px > x_max {
            px = x_max - qx;
        }
        if qy + py > y_max {
            py = y_max - qy;
        }

        // ---- hybrid 预测（8.3.5.3.5）：这一支**要读位流** ----
        let hybridmv_thresh = 32;
        if a_valid && c_valid {
            let is_intra_a = self.grids.mb_type[(xy - wrap) as usize] != 0;
            let sum_a = if is_intra_a {
                px.abs() + py.abs()
            } else {
                (px - fa.0).abs() + (py - fa.1).abs()
            };
            if sum_a > hybridmv_thresh {
                if self.r.bit()? != 0 {
                    px = fa.0;
                    py = fa.1;
                } else {
                    px = fc.0;
                    py = fc.1;
                }
            } else {
                let is_intra_c = self.grids.mb_type[(xy - 1) as usize] != 0;
                let sum_c = if is_intra_c {
                    px.abs() + py.abs()
                } else {
                    (px - fc.0).abs() + (py - fc.1).abs()
                };
                if sum_c > hybridmv_thresh {
                    if self.r.bit()? != 0 {
                        px = fa.0;
                        py = fa.1;
                    } else {
                        px = fc.0;
                        py = fc.1;
                    }
                }
            }
        }

        // ---- 按 MV 范围取模后落盘（4.11）----
        let rx = self.hdr.range_x;
        let ry = self.hdr.range_y;
        let vx = ((px + dmv_x + rx) & ((rx << 1) - 1)) - rx;
        let vy = ((py + dmv_y + ry) & ((ry << 1) - 1)) - ry;
        self.grids.mv[xy as usize] = [vx as i16, vy as i16];
        if mv1 {
            // 1MV 宏块：四个块共用同一个 MV（后面的块与环路滤波都要读到）。
            for o in [1isize, wrap, wrap + 1] {
                self.grids.mv[(xy + o) as usize] = [vx as i16, vy as i16];
            }
        }
        Ok(())
    }

    /// 亮度 MV（供运动补偿与环路滤波读）。
    #[inline]
    pub(crate) fn mv_of(&self, mbx: usize, mby: usize, n: usize) -> (i32, i32) {
        let m = self.grids.mv[self.blk(mbx, mby, n)];
        (m[0] as i32, m[1] as i32)
    }
}
