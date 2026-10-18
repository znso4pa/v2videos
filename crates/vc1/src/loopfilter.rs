//! 环路滤波（`vc1_loopfilter.c` 的 `ff_vc1_i_loop_filter()` / `ff_vc1_p_loop_filter()`）。
//!
//! ## 为什么这里的「谁在什么时候被滤」看着别扭
//!
//! 上游的滤波**落后解码两行两列**：在解第 `(x, y)` 个宏块时，滤的是
//! `(x-1, y-1)` / `(x-1, y-2)` / `(x-2, y-2)` 那几个宏块的边。原因是滤波要读邻居
//! 已经**写进帧里**的像素，而写像素本身又落后一圈（`vc1_put_blocks_clamped`）。
//!
//! 本实现把像素**立刻**写进帧（不做 overlap，写像素与顺序无关），但滤波的调用点
//! 与判据**原样保留** —— 滤波之间是有先后依赖的（同一个宏块内「先竖后横」，
//! 交叉点上的像素会被两次滤波先后改到），顺序变了像素就不一样。
//!
//! ## 边的几何
//!
//! - **竖滤波**（`v_filter`）：滤一条**横向**的边 —— 边界在行 `y` 上方，
//!   管 `x .. x+len` 这一段；改的是第 `y-1` 与第 `y` 两行。
//! - **横滤波**（`h_filter`）：滤一条**纵向**的边 —— 边界在列 `x` 左边，
//!   管 `y .. y+len` 这一段；改的是第 `x-1` 与第 `x` 两列。
//!
//! 每个宏块自己负责：顶边、内部的第 8 行 / 第 8 列、左边；底边与右边由**下一个**
//! 宏块（或帧边界那几条特例）负责。这就是「落后两行两列」的由来。

use crate::decoder::Ctx;
use crate::tables;

const LEFT_EDGE: u32 = 1;
const RIGHT_EDGE: u32 = 2;
const TOP_EDGE: u32 = 4;
const BOTTOM_EDGE: u32 = 8;

/// `vc1_filter_line()`：一条边上的一个采样点。返回「其余三个点也该滤」。
fn filter_line(data: &mut [u8], off: isize, stride: isize, pq: i32) -> bool {
    let at = |k: isize| (off + k * stride) as usize;
    let s = |k: isize| data[at(k)] as i32;
    let mut a0 = (2 * (s(-2) - s(1)) - 5 * (s(-1) - s(0)) + 4) >> 3;
    let a0_sign = a0 >> 31;
    a0 = (a0 ^ a0_sign) - a0_sign; // 绝对值
    if a0 < pq {
        let a1 = ((2 * (s(-4) - s(-1)) - 5 * (s(-3) - s(-2)) + 4) >> 3).abs();
        let a2 = ((2 * (s(0) - s(3)) - 5 * (s(1) - s(2)) + 4) >> 3).abs();
        if a1 < a0 || a2 < a0 {
            let mut clip = s(-1) - s(0);
            let clip_sign = clip >> 31;
            clip = ((clip ^ clip_sign) - clip_sign) >> 1;
            if clip != 0 {
                let a3 = a1.min(a2);
                let mut d = 5 * (a3 - a0);
                let mut d_sign = d >> 31;
                d = ((d ^ d_sign) - d_sign) >> 3;
                d_sign ^= a0_sign;
                if d_sign ^ clip_sign == 0 {
                    // 只有同号才真的改像素（上游的 `if (d_sign ^ clip_sign) d = 0;`）。
                    d = d.min(clip);
                    d = (d ^ d_sign) - d_sign;
                    let (lo, hi) = (s(-1), s(0));
                    data[at(-1)] = (lo - d).clamp(0, 255) as u8;
                    data[at(0)] = (hi + d).clamp(0, 255) as u8;
                }
                return true;
            }
        }
    }
    false
}

/// `vc1_loop_filter()`：`len` 个像素宽/高的一条边，4 个一组。
fn loop_filter(data: &mut [u8], base: isize, step: isize, stride: isize, len: usize, pq: i32) {
    for i in (0..len as isize).step_by(4) {
        if filter_line(data, base + (i + 2) * step, stride, pq) {
            filter_line(data, base + i * step, stride, pq);
            filter_line(data, base + (i + 1) * step, stride, pq);
            filter_line(data, base + (i + 3) * step, stride, pq);
        }
    }
}

impl Ctx<'_, '_> {
    /// 竖滤波：边界在行 `y` 上方，管 `x .. x+len`。
    fn v_filter(&mut self, plane: usize, x: isize, y: isize, len: usize) {
        let pq = self.hdr.pq as i32;
        let stride = self.plane_stride(plane) as isize;
        let data = self.plane_mut(plane);
        loop_filter(data, y * stride + x, 1, stride, len, pq);
    }

    /// 横滤波：边界在列 `x` 左边，管 `y .. y+len`。
    fn h_filter(&mut self, plane: usize, x: isize, y: isize, len: usize) {
        let pq = self.hdr.pq as i32;
        let stride = self.plane_stride(plane) as isize;
        let data = self.plane_mut(plane);
        loop_filter(data, y * stride + x, stride, 1, len, pq);
    }

    /// 块 `block_num` 对应的 `(平面, 块内 x, 块内 y, 边长)`（像素计）。
    fn block_geom(&self, block_num: usize) -> (usize, usize, usize, usize) {
        if block_num > 3 {
            (block_num - 3, 0, 0, 8)
        } else {
            (0, (block_num & 1) * 8, (block_num & 2) * 4, 16)
        }
    }

    // ================= I 帧 =================

    /// `vc1_i_v_loop_filter()`：`(mx, my)` 是被滤宏块的坐标。
    fn i_v_filter(&mut self, block_num: usize, mx: isize, my: isize, flags: u32) {
        if block_num & 5 == 1 {
            return;
        }
        if flags & TOP_EDGE == 0 || block_num & 2 != 0 {
            let (plane, dx, dy, len) = self.block_geom(block_num);
            let scale = if plane == 0 { 16 } else { 8 } as isize;
            self.v_filter(plane, mx * scale + dx as isize, my * scale + dy as isize, len);
        }
    }

    /// `vc1_i_h_loop_filter()`。
    fn i_h_filter(&mut self, block_num: usize, mx: isize, my: isize, flags: u32) {
        if block_num & 2 != 0 {
            return;
        }
        if flags & LEFT_EDGE == 0 || block_num & 5 == 1 {
            let (plane, dx, dy, len) = self.block_geom(block_num);
            let scale = if plane == 0 { 16 } else { 8 } as isize;
            self.h_filter(plane, mx * scale + dx as isize, my * scale + dy as isize, len);
        }
    }

    /// `ff_vc1_i_loop_filter()`。
    pub(crate) fn i_loop_filter(&mut self, mbx: usize, mby: usize) {
        let (mw, mh) = (self.mb_width, self.mb_height);
        // ---- 上一行的竖滤波 ----
        if mby >= 1 {
            let flags = if mby == 1 { TOP_EDGE } else { 0 };
            if mbx >= 1 {
                for i in 0..6 {
                    self.i_v_filter(i, mbx as isize - 1, mby as isize - 1, flags);
                }
            }
            if mbx == mw - 1 {
                for i in 0..6 {
                    self.i_v_filter(i, mbx as isize, mby as isize - 1, flags);
                }
            }
        }
        // ---- 本行的竖滤波（最后一行才做）----
        if mby == mh - 1 {
            let flags = if mby == 0 {
                TOP_EDGE | BOTTOM_EDGE
            } else {
                BOTTOM_EDGE
            };
            if mbx >= 1 {
                for i in 0..6 {
                    self.i_v_filter(i, mbx as isize - 1, mby as isize, flags);
                }
            }
            if mbx == mw - 1 {
                for i in 0..6 {
                    self.i_v_filter(i, mbx as isize, mby as isize, flags);
                }
            }
        }
        // ---- 上上行的横滤波 ----
        if mby >= 2 {
            if mbx >= 1 {
                let flags = if mbx == 1 { LEFT_EDGE } else { 0 };
                for i in 0..6 {
                    self.i_h_filter(i, mbx as isize - 1, mby as isize - 2, flags);
                }
            }
            if mbx == mw - 1 {
                let flags = if mbx == 0 {
                    LEFT_EDGE | RIGHT_EDGE
                } else {
                    RIGHT_EDGE
                };
                for i in 0..6 {
                    self.i_h_filter(i, mbx as isize, mby as isize - 2, flags);
                }
            }
        }
        // ---- 最后两行的横滤波 ----
        if mby == mh - 1 {
            if mby >= 1 {
                if mbx >= 1 {
                    let flags = if mbx == 1 { LEFT_EDGE } else { 0 };
                    for i in 0..6 {
                        self.i_h_filter(i, mbx as isize - 1, mby as isize - 1, flags);
                    }
                }
                if mbx == mw - 1 {
                    let flags = if mbx == 0 {
                        LEFT_EDGE | RIGHT_EDGE
                    } else {
                        RIGHT_EDGE
                    };
                    for i in 0..6 {
                        self.i_h_filter(i, mbx as isize, mby as isize - 1, flags);
                    }
                }
            }
            if mbx >= 1 {
                let flags = if mbx == 1 { LEFT_EDGE } else { 0 };
                for i in 0..6 {
                    self.i_h_filter(i, mbx as isize - 1, mby as isize, flags);
                }
            }
            if mbx == mw - 1 {
                let flags = if mbx == 0 {
                    LEFT_EDGE | RIGHT_EDGE
                } else {
                    RIGHT_EDGE
                };
                for i in 0..6 {
                    self.i_h_filter(i, mbx as isize, mby as isize, flags);
                }
            }
        }
    }

    // ================= P 帧 =================

    /// 被滤宏块 `(mx, my)` 的块 `block_num` 的两侧信息。
    #[allow(clippy::type_complexity)]
    fn p_block_sides(
        &self,
        mx: usize,
        my: usize,
        block_num: usize,
    ) -> (u32, u32, bool, bool, [i16; 2], [i16; 2]) {
        let mb_pos = self.mbidx(mx, my);
        let stride = self.mb_stride;
        let cbp0 = self.mbp.cbp[mb_pos];
        let cbp1 = self.mbp.cbp[mb_pos + stride];
        let ii0 = self.mbp.is_intra[mb_pos];
        let ii1 = self.mbp.is_intra[mb_pos + stride];
        let top_cbp = cbp0 >> (block_num * 4);
        let top_is_intra = ii0 & (1 << block_num) != 0;
        let (bottom_is_intra, bottom_cbp) = if block_num > 3 {
            (ii1 & (1 << block_num) != 0, cbp1 >> (block_num * 4))
        } else if block_num < 2 {
            (
                ii0 & (1 << (block_num + 2)) != 0,
                cbp0 >> ((block_num + 2) * 4),
            )
        } else {
            (
                ii1 & (1 << (block_num - 2)) != 0,
                cbp1 >> ((block_num - 2) * 4),
            )
        };
        let (mv0, mv1) = if block_num > 3 {
            (
                self.grids.luma_mv[mb_pos],
                self.grids.luma_mv[mb_pos + stride],
            )
        } else {
            let b = self.blk(mx, my, block_num);
            (self.grids.mv[b], self.grids.mv[b + self.b8_stride])
        };
        (
            top_cbp,
            bottom_cbp,
            top_is_intra,
            bottom_is_intra,
            mv0,
            mv1,
        )
    }

    /// `vc1_p_v_loop_filter()`：`(mx, my)` 是被滤宏块。
    fn p_v_filter(&mut self, mx: usize, my: usize, block_num: usize, flags: u32) {
        let (plane, dx, dy, _) = self.block_geom(block_num);
        let scale = if plane == 0 { 16 } else { 8 };
        let x = (mx * scale + dx) as isize;
        let y = (my * scale + dy) as isize;
        let (top_cbp, bottom_cbp, top_intra, bottom_intra, mv0, mv1) =
            self.p_block_sides(mx, my, block_num);
        if flags & BOTTOM_EDGE == 0 || block_num < 2 {
            if top_intra || bottom_intra || mv0 != mv1 {
                self.v_filter(plane, x, y + 8, 8);
            } else {
                let idx = (top_cbp | (bottom_cbp >> 2)) & 3;
                if idx & 1 != 0 {
                    self.v_filter(plane, x + 4, y + 8, 4);
                }
                if idx & 2 != 0 {
                    self.v_filter(plane, x, y + 8, 4);
                }
            }
        }
        let tt = (self.mbp.ttblk[self.mbidx(mx, my)] >> (block_num * 4)) & 0xf;
        if tt == tables::TT_4X4 as u32 || tt == tables::TT_8X4 as u32 {
            if top_cbp & 5 != 0 {
                self.v_filter(plane, x + 4, y + 4, 4);
            }
            if top_cbp & 10 != 0 {
                self.v_filter(plane, x, y + 4, 4);
            }
        }
    }

    /// `vc1_p_h_loop_filter()`。
    fn p_h_filter(&mut self, mx: usize, my: usize, block_num: usize, flags: u32) {
        let (plane, dx, dy, _) = self.block_geom(block_num);
        let scale = if plane == 0 { 16 } else { 8 };
        let x = (mx * scale + dx) as isize;
        let y = (my * scale + dy) as isize;
        let mb_pos = self.mbidx(mx, my);
        let left_cbp = self.mbp.cbp[mb_pos] >> (block_num * 4);
        if flags & RIGHT_EDGE == 0 || block_num & 5 == 0 {
            let left_intra = self.mbp.is_intra[mb_pos] & (1 << block_num) != 0;
            let (right_intra, right_cbp) = if block_num > 3 {
                (
                    self.mbp.is_intra[mb_pos + 1] & (1 << block_num) != 0,
                    self.mbp.cbp[mb_pos + 1] >> (block_num * 4),
                )
            } else if block_num & 1 != 0 {
                (
                    self.mbp.is_intra[mb_pos + 1] & (1 << (block_num - 1)) != 0,
                    self.mbp.cbp[mb_pos + 1] >> ((block_num - 1) * 4),
                )
            } else {
                (
                    self.mbp.is_intra[mb_pos] & (1 << (block_num + 1)) != 0,
                    self.mbp.cbp[mb_pos] >> ((block_num + 1) * 4),
                )
            };
            let (mv0, mv1) = if block_num > 3 {
                (
                    self.grids.luma_mv[mb_pos],
                    self.grids.luma_mv[mb_pos + 1],
                )
            } else {
                let b = self.blk(mx, my, block_num);
                (self.grids.mv[b], self.grids.mv[b + 1])
            };
            if left_intra || right_intra || mv0 != mv1 {
                self.h_filter(plane, x + 8, y, 8);
            } else {
                let idx = (left_cbp | (right_cbp >> 1)) & 5;
                if idx & 1 != 0 {
                    self.h_filter(plane, x + 8, y + 4, 4);
                }
                if idx & 4 != 0 {
                    self.h_filter(plane, x + 8, y, 4);
                }
            }
        }
        let tt = (self.mbp.ttblk[mb_pos] >> (block_num * 4)) & 0xf;
        if tt == tables::TT_4X4 as u32 || tt == tables::TT_4X8 as u32 {
            if left_cbp & 3 != 0 {
                self.h_filter(plane, x + 4, y + 4, 4);
            }
            if left_cbp & 12 != 0 {
                self.h_filter(plane, x + 4, y, 4);
            }
        }
    }

    /// `ff_vc1_p_loop_filter()`。
    pub(crate) fn p_loop_filter(&mut self, mbx: usize, mby: usize) {
        let (mw, mh) = (self.mb_width, self.mb_height);
        // ---- 竖滤波 ----
        if mby >= 2 {
            let flags = if mby == 2 { TOP_EDGE } else { 0 };
            if mbx >= 1 {
                for i in 0..6 {
                    self.p_v_filter(mbx - 1, mby - 2, i, flags);
                }
            }
            if mbx == mw - 1 {
                for i in 0..6 {
                    self.p_v_filter(mbx, mby - 2, i, flags);
                }
            }
        }
        if mby == mh - 1 {
            if mbx >= 1 {
                if mby >= 1 {
                    let flags = if mby == 1 { TOP_EDGE } else { 0 };
                    for i in 0..6 {
                        self.p_v_filter(mbx - 1, mby - 1, i, flags);
                    }
                }
                let flags = if mby == 0 {
                    TOP_EDGE | BOTTOM_EDGE
                } else {
                    BOTTOM_EDGE
                };
                for i in 0..6 {
                    self.p_v_filter(mbx - 1, mby, i, flags);
                }
            }
            if mbx == mw - 1 {
                if mby >= 1 {
                    let flags = if mby == 1 { TOP_EDGE } else { 0 };
                    for i in 0..6 {
                        self.p_v_filter(mbx, mby - 1, i, flags);
                    }
                }
                let flags = if mby == 0 {
                    TOP_EDGE | BOTTOM_EDGE
                } else {
                    BOTTOM_EDGE
                };
                for i in 0..6 {
                    self.p_v_filter(mbx, mby, i, flags);
                }
            }
        }
        // ---- 横滤波 ----
        if mby >= 2 {
            if mbx >= 2 {
                let flags = if mbx == 2 { LEFT_EDGE } else { 0 };
                for i in 0..6 {
                    self.p_h_filter(mbx - 2, mby - 2, i, flags);
                }
            }
            if mbx == mw - 1 {
                if mbx >= 1 {
                    let flags = if mbx == 1 { LEFT_EDGE } else { 0 };
                    for i in 0..6 {
                        self.p_h_filter(mbx - 1, mby - 2, i, flags);
                    }
                }
                let flags = if mbx == 0 {
                    LEFT_EDGE | RIGHT_EDGE
                } else {
                    RIGHT_EDGE
                };
                for i in 0..6 {
                    self.p_h_filter(mbx, mby - 2, i, flags);
                }
            }
        }
        if mby == mh - 1 {
            if mby >= 1 {
                if mbx >= 2 {
                    let flags = if mbx == 2 { LEFT_EDGE } else { 0 };
                    for i in 0..6 {
                        self.p_h_filter(mbx - 2, mby - 1, i, flags);
                    }
                }
                if mbx == mw - 1 {
                    if mbx >= 1 {
                        let flags = if mbx == 1 { LEFT_EDGE } else { 0 };
                        for i in 0..6 {
                            self.p_h_filter(mbx - 1, mby - 1, i, flags);
                        }
                    }
                    let flags = if mbx == 0 {
                        LEFT_EDGE | RIGHT_EDGE
                    } else {
                        RIGHT_EDGE
                    };
                    for i in 0..6 {
                        self.p_h_filter(mbx, mby - 1, i, flags);
                    }
                }
            }
            if mbx >= 2 {
                let flags = if mbx == 2 { LEFT_EDGE } else { 0 };
                for i in 0..6 {
                    self.p_h_filter(mbx - 2, mby, i, flags);
                }
            }
            if mbx == mw - 1 {
                if mbx >= 1 {
                    let flags = if mbx == 1 { LEFT_EDGE } else { 0 };
                    for i in 0..6 {
                        self.p_h_filter(mbx - 1, mby, i, flags);
                    }
                }
                let flags = if mbx == 0 {
                    LEFT_EDGE | RIGHT_EDGE
                } else {
                    RIGHT_EDGE
                };
                for i in 0..6 {
                    self.p_h_filter(mbx, mby, i, flags);
                }
            }
        }
    }
}
