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

use v2v_kernel::Result;
use v2v_kernel::mc::mid_pred;

use crate::decoder::Ctx;
use crate::header::MvMode;

/// B 帧的预测方向（上游 `BMV_TYPE_*`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BmvType {
    Forward = 0,
    Backward = 1,
    Interpolated = 2,
}

/// `scale_mv()`（`vc1_pred.h`）：把后向参考帧的 MV 按 `bfraction` 缩放到本帧。
///
/// ⚠️ 两处**必须照抄**：
/// - `inv` 是「换成另一侧」—— 减的是 `B_FRACTION_DEN`（256），不是取负；
/// - 半像素（`!qs`）那一支最后**乘 2**（半像素单位换成四分之一像素单位），
///   而且加的是 `255` 不是 `256`。写成 `value * n / 256` 会在负值上差 1。
fn scale_mv(value: i32, bfrac: i32, inv: bool, qs: bool) -> i32 {
    let mut n = bfrac;
    if inv {
        n -= 256;
    }
    if !qs {
        2 * ((value * n + 255) >> 9)
    } else {
        (value * n + 128) >> 8
    }
}

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

    /// `ff_vc1_pred_b_mv()`（渐进式）：B 帧的 MV 预测。
    ///
    /// 与 P 帧那条路的**三处**不同，少一处整帧就偏：
    /// 1. 两套 MV 面（前向 `mv` / 后向 `mv_bwd`），各有自己的空间预测链；
    /// 2. `direct` 时 MV 不是从位流读的，而是**后向参考帧的 MV 按 `bfraction` 缩放**
    ///    （上游读 `next_pic.motion_val[1]`），缩放之后再按 8.4.5.4 做 pullback；
    /// 3. **不做 hybrid 预测** —— 上游那两个 `if (0 && …)` 就是把它关了，
    ///    所以 B 帧这里一位都不用从位流读。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pred_b_mv(
        &mut self,
        mbx: usize,
        mby: usize,
        dmv_x: [i32; 2],
        dmv_y: [i32; 2],
        direct: bool,
        mvtype: BmvType,
        mb_intra: bool,
    ) -> Result<()> {
        let qs = self.quarter_sample();
        let bfrac = self.hdr.b.as_ref().map_or(0, |b| b.bfraction as i32);
        let wrap = self.b8_stride as isize;
        let xy = self.blk(mbx, mby, 0) as isize;
        let (rx, ry) = (self.hdr.range_x, self.hdr.range_y);

        if mb_intra {
            self.grids.mv[xy as usize] = [0, 0];
            self.grids.mv_bwd[xy as usize] = [0, 0];
            return Ok(());
        }

        let mut dmv_x = dmv_x;
        let mut dmv_y = dmv_y;
        if !qs {
            // 半像素模式：差分是半像素单位，MV 面里存的是四分之一像素。
            for i in 0..2 {
                dmv_x[i] *= 2;
                dmv_y[i] *= 2;
            }
        }

        // ---- direct 缩放：**每个宏块都算**，不只是 direct 的那些 ----
        //
        // ⚠️ 上游把这两份放在函数**开头**无条件算出来，然后
        //   - direct 宏块：直接用；
        //   - 非 direct 宏块：只覆盖**本宏块实际要用的那个方向**，另一个方向
        //     **保留这里的缩放值**；
        //   - 最后把两个方向**都**写进 MV 面（`vc1_pred.c` 第 885–888 行）。
        //
        // 漏掉「另一个方向也保留缩放值」这一步，后果非常隐蔽：该宏块本身像素
        // 是对的，但**右边 / 下边那个宏块**做空间预测时读到 0 而不是缩放值，
        // 于是从那里开始整片 MV 偏移 —— 症状是「坏宏块连成一片」，位游标却完全正常。
        let base = self.next_mv[xy as usize];
        let mut mvf = [
            scale_mv(base[0] as i32, bfrac, false, qs),
            scale_mv(base[1] as i32, bfrac, false, qs),
        ];
        let mut mvb = [
            scale_mv(base[0] as i32, bfrac, true, qs),
            scale_mv(base[1] as i32, bfrac, true, qs),
        ];
        // pullback（8.4.5.4）：注意这里用的是 `<<6` 与 `-60`，与下面非 direct 的
        // `<<5` / `-28` **不是同一套**（上游就是两套）。
        {
            let (qx, qy) = ((mbx as i32) << 6, (mby as i32) << 6);
            let (xmax, ymax) = (
                ((self.mb_width as i32) << 6) - 4,
                ((self.mb_height as i32) << 6) - 4,
            );
            mvf[0] = mvf[0].clamp(-60 - qx, xmax - qx);
            mvf[1] = mvf[1].clamp(-60 - qy, ymax - qy);
            mvb[0] = mvb[0].clamp(-60 - qx, xmax - qx);
            mvb[1] = mvb[1].clamp(-60 - qy, ymax - qy);
        }
        let want_fwd = matches!(mvtype, BmvType::Forward | BmvType::Interpolated);
        let want_bwd = matches!(mvtype, BmvType::Backward | BmvType::Interpolated);
        for dir in 0..2usize {
            if direct || (dir == 0 && !want_fwd) || (dir == 1 && !want_bwd) {
                continue;
            }
            let plane = if dir == 0 {
                &self.grids.mv
            } else {
                &self.grids.mv_bwd
            };
            let off = if mbx == self.mb_width - 1 { -2 } else { 2 };
            // ⚠️ 邻居只在「用得到」的时候才读：第一行（`mby == 0`）的 `A` / `B` 落在
            // 网格的**上边界之外**（b8 网格只留了一行一行距的余量，而这里是两行），
            // 读它就是负下标。上游那边有更大的 padding，所以它敢无条件读。
            let (mut px, mut py);
            if mby != 0 {
                let a = plane[(xy - wrap * 2) as usize];
                let b = plane[(xy - wrap * 2 + off) as usize];
                if self.mb_width == 1 {
                    px = a[0] as i32;
                    py = a[1] as i32;
                } else {
                    // 左边界：上游会先把 `C` 写 0 再用它，这里直接用 0。
                    let c = if mbx != 0 {
                        plane[(xy - 2) as usize]
                    } else {
                        [0i16; 2]
                    };
                    px = mid_pred(a[0] as i32, b[0] as i32, c[0] as i32);
                    py = mid_pred(a[1] as i32, b[1] as i32, c[1] as i32);
                }
            } else if mbx != 0 {
                let c = plane[(xy - 2) as usize];
                px = c[0] as i32;
                py = c[1] as i32;
            } else {
                px = 0;
                py = 0;
            }
            // pullback（8.3.5.3.4）：Simple/Main 用 `sh = 5`。
            let sh = 5;
            let (qx, qy) = ((mbx as i32) << sh, (mby as i32) << sh);
            let mv_lo = 4 - (1 << sh);
            let (xmax, ymax) = (
                ((self.mb_width as i32) << sh) - 4,
                ((self.mb_height as i32) << sh) - 4,
            );
            if qx + px < mv_lo {
                px = mv_lo - qx;
            }
            if qy + py < mv_lo {
                py = mv_lo - qy;
            }
            if qx + px > xmax {
                px = xmax - qx;
            }
            if qy + py > ymax {
                py = ymax - qy;
            }
            let vx = ((px + dmv_x[dir] + rx) & ((rx << 1) - 1)) - rx;
            let vy = ((py + dmv_y[dir] + ry) & ((ry << 1) - 1)) - ry;
            if dir == 0 {
                mvf = [vx, vy];
            } else {
                mvb = [vx, vy];
            }
        }
        self.grids.mv[xy as usize] = [mvf[0] as i16, mvf[1] as i16];
        self.grids.mv_bwd[xy as usize] = [mvb[0] as i16, mvb[1] as i16];
        Ok(())
    }
}
