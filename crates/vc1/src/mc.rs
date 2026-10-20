//! 运动补偿（渐进式）：亮度四分之一 / 半像素插值 + 色度双线性。
//!
//! 对照上游：`vc1_mc.c` 的 `ff_vc1_mc_1mv()` / `ff_vc1_mc_4mv_luma()` /
//! `ff_vc1_mc_4mv_chroma()`（各自只留渐进式那一支），插值核来自 `vc1dsp.c` 的
//! `vc1_mspel_*`，色度来自 `h264chroma` 与 `vc1dsp` 的 `put_no_rnd_vc1_chroma_*`。
//!
//! ## 越界：永远先拷进本地缓冲
//!
//! 上游分两条路（够用就直接指参考帧、越界才 `emulated_edge_mc` 拷一份）。这里
//! **一律**拷进本地缓冲并做边缘复制 —— 两者在「没越界」时逐像素相同，而统一之后
//! 少一条只在边界上才走、因而最难验的路径。缓冲的布局：源坐标 `(src_x - pad, src_y - pad)`
//! 落在 `(pad, pad)`，`pad = mspel`（插值核要往左上多读 `mspel` 个像素）。
//!
//! ## 取整位 `rnd`
//!
//! `rnd` 是**逐帧翻转**的一位（位流里没有它，见 `HeaderState::rnd`）：它同时决定
//! 亮度插值的取整偏置与色度走哪张表（`rnd=0` 走 h264 的 `+32`，`rnd=1` 走 VC-1 的 `+28`）。
//! 少一处就是「一半的帧偏半个像素」。

use v2v_msmpeg4::mid_pred;

use crate::decoder::Ctx;

/// 本地缓冲的边长（够放 16×16 块 + mspel 的余量）。
///
/// ⚠️ 这两个常数**不能随便调大**：`let mut buf = [0u8; BUF * BUF]` 是**每个块**都要
/// 清零一次的栈数组，而 MC 每个宏块至少进一次（4MV 是四次）。24 是「够用且最小」那一档：
/// mspel 时 `base = BUF_STRIDE + 1`、`k = 19`，最远读到 `(1+19)*BUF_STRIDE + 19 = 499 < 576`。
const BUF: usize = 24;
#[doc(hidden)]
pub const BUF_STRIDE: usize = BUF;

/// `vc1_mspel_filter()`：单向插值（`stride` 是跨过边缘的方向）。
fn mspel_filter(src: &[u8], off: isize, stride: isize, mode: u32, r: i32) -> i32 {
    let s = |k: isize| src[(off + k * stride) as usize] as i32;
    match mode {
        0 => s(0),
        1 => (-4 * s(-1) + 53 * s(0) + 18 * s(1) - 3 * s(2) + 32 - r) >> 6,
        2 => (-1 * s(-1) + 9 * s(0) + 9 * s(1) - 1 * s(2) + 8 - r) >> 4,
        _ => (-3 * s(-1) + 18 * s(0) + 53 * s(1) - 4 * s(2) + 32 - r) >> 6,
    }
}

/// `OPNAME vc1_mspel_mc`：`W × W` 块的四分之一像素插值（`hmode` / `vmode` ∈ 0..4）。
///
/// `W` 走**常量泛型**而不是运行时参数：中间结果缓冲正好 `(W+3)*W` 个 i16，
/// 不必按最大尺寸（16）开一个再白清零一大半 —— 4MV 时每个宏块要进四次，
/// 那个 memset 是实打实的浪费。顺带每个 `W` 各自单态化，内层循环能完全展开。
#[allow(clippy::too_many_arguments)]
fn mspel_mc<const W: usize>(
    dst: &mut [u8],
    dstride: usize,
    dx: usize,
    dy: usize,
    src: &[u8],
    base: usize,
    hmode: u32,
    vmode: u32,
    rnd: i32,
    // 调用方按 `(W+3)*W` 开（常量泛型不能在数组长度里做算术），这里只切片。
    tmp: &mut [i16],
) {
    let w = W;
    let b = base as isize;
    if vmode != 0 {
        if hmode != 0 {
            // 两趟：先竖直（出中间结果），再水平。
            const SHIFT_VALUE: [i32; 4] = [0, 5, 1, 5];
            let shift = (SHIFT_VALUE[hmode as usize] + SHIFT_VALUE[vmode as usize]) >> 1;
            let r = (1 << (shift - 1)) + rnd - 1;
            // ⚠️ 中间结果**不夹取**（上游就是 `int16_t` 直接存）——夹了会在
            // 陡边处差 1，而那种差只出现在少数像素上，最难查。
            //
            // 缓冲**正好** `(W+3)*W`：由调用方按常量开，不为 8×8 白开 16×16 的。
            let tmp = &mut tmp[..(W + 3) * W];
            // 竖直那一趟：4 个抽头是「上下 4 行、同一列」。把**行偏移**提到内层循环
            // 外面，`vmode` 的系数选择也跟着提出来 —— 原来每像素都要重新 match 一次
            // （走的是 `mspel_filter16`，那个 `match mode` 在函数里）。
            for j in 0..w {
                // `src -= 1`：竖直那一趟整体左移一列（与上游一致）。
                let row0 = (b + (j as isize - 1) * BUF_STRIDE as isize - 1) as usize;
                #[cfg(target_arch = "aarch64")]
                crate::neon::vertical_pass(src, row0, w + 3, vmode, shift, r, &mut tmp[j * (w + 3)..]);
                #[cfg(not(target_arch = "aarch64"))]
                for i in 0..(w + 3) {
                    let o = row0 + i;
                    let (a, bb, c, d) = (
                        src[o] as i32,
                        src[o + BUF_STRIDE] as i32,
                        src[o + 2 * BUF_STRIDE] as i32,
                        src[o + 3 * BUF_STRIDE] as i32,
                    );
                    let v = match vmode {
                        1 => -4 * a + 53 * bb + 18 * c - 3 * d,
                        2 => -1 * a + 9 * bb + 9 * c - 1 * d,
                        _ => -3 * a + 18 * bb + 53 * c - 4 * d,
                    };
                    tmp[j * (w + 3) + i] = ((v + r) >> shift) as i16;
                }
            }
            let r2 = 64 - rnd;
            for j in 0..w {
                let t = j * (w + 3) + 1;
                let drow = (dy + j) * dstride + dx;
                #[cfg(target_arch = "aarch64")]
                crate::neon::horizontal_pass(tmp, t, w, hmode, r2, dst, drow);
                #[cfg(not(target_arch = "aarch64"))]
                for i in 0..w {
                    // ⚠️ 水平这一趟的系数**按 `hmode` 选**（1/4 位移那组系数只在
                    // `hmode == 1` 时才对）—— 写死成一组，平坦区就会整体偏亮/偏暗。
                    let o = t + i;
                    let (a, bb, c, d) = (
                        i32::from(tmp[o - 1]),
                        i32::from(tmp[o]),
                        i32::from(tmp[o + 1]),
                        i32::from(tmp[o + 2]),
                    );
                    let h = match hmode {
                        1 => -4 * a + 53 * bb + 18 * c - 3 * d,
                        2 => -1 * a + 9 * bb + 9 * c - 1 * d,
                        _ => -3 * a + 18 * bb + 53 * c - 4 * d,
                    };
                    dst[drow + i] = ((h + r2) >> 7).clamp(0, 255) as u8;
                }
            }
        } else {
            let r = 1 - rnd;
            for j in 0..w {
                let row0 = (b + (j * BUF_STRIDE) as isize) as usize;
                let drow = (dy + j) * dstride + dx;
                #[cfg(target_arch = "aarch64")]
                crate::neon::single_vertical(src, row0, w, vmode, r, dst, drow);
                #[cfg(not(target_arch = "aarch64"))]
                for i in 0..w {
                    let off = b + (j * BUF_STRIDE + i) as isize;
                    let v = mspel_filter(src, off, BUF_STRIDE as isize, vmode, r);
                    dst[drow + i] = v.clamp(0, 255) as u8;
                }
            }
        }
    } else {
        // 只有水平（含 hmode == 0 的整像素拷贝）。
        for j in 0..w {
            let row = (b + (j * BUF_STRIDE) as isize) as usize;
            let drow = (dy + j) * dstride + dx;
            #[cfg(target_arch = "aarch64")]
            if hmode != 0 {
                crate::neon::single_horizontal(src, row, w, hmode, rnd, dst, drow);
                continue;
            }
            for i in 0..w {
                let off = b + (j * BUF_STRIDE + i) as isize;
                let v = mspel_filter(src, off, 1, hmode, rnd);
                dst[drow + i] = v.clamp(0, 255) as u8;
            }
        }
    }
}

/// 半像素（hpel）插值：`dxy` 的位 0 = 水平、位 1 = 竖直。
#[allow(clippy::too_many_arguments)]
fn hpel_mc(
    dst: &mut [u8],
    dstride: usize,
    dx: usize,
    dy: usize,
    w: usize,
    src: &[u8],
    base: usize,
    dxy: usize,
    rnd: bool,
) {
    let hm = dxy & 1 != 0;
    let vm = dxy & 2 != 0;
    // ⚠️ **反的**：上游是 `if (!v->rnd) put_pixels_tab`（**带**取整）`else put_no_rnd_pixels_tab` ——
    // 也就是 `rnd == 0` 走 `rnd_avg`（`(a+b+1)>>1`）、`rnd == 1` 走截断。写反了就是
    // 「一半的帧整体偏 1」，而且只在半像素（hpel）那条路上现形。
    // 取整常数**只算一次**：`rnd` 是整帧不变量，原来把它写进闭包里，
    // 等于每个像素再分一次支。
    let r2 = if rnd { 0 } else { 1 };
    let r4 = if rnd { 1 } else { 2 };
    let avg2 = |a: i32, b: i32| (a + b + r2) >> 1;
    #[cfg_attr(target_arch = "aarch64", allow(unused_variables))]
    let avg4 = |a: i32, b: i32, c: i32, d: i32| (a + b + c + d + r4) >> 2;
    // `(hm, vm)` 是**循环不变量**，四条路各自一个紧循环 —— 写在最内层等于每个像素
    // 再分一次支；顺手把目的行的起点也算到行首（省掉每像素一次乘法）。
    match (hm, vm) {
        (false, false) => {
            for j in 0..w {
                let s = base + j * BUF_STRIDE;
                let d = (dy + j) * dstride + dx;
                dst[d..d + w].copy_from_slice(&src[s..s + w]);
            }
        }
        (true, false) => {
            for j in 0..w {
                let s = base + j * BUF_STRIDE;
                let d = (dy + j) * dstride + dx;
                #[cfg(target_arch = "aarch64")]
                crate::neon::hpel_avg2_row(src, s, dst, d, w, !rnd);
                #[cfg(not(target_arch = "aarch64"))]
                for i in 0..w {
                    dst[d + i] = avg2(src[s + i] as i32, src[s + i + 1] as i32) as u8;
                }
            }
        }
        (false, true) => {
            for j in 0..w {
                let s = base + j * BUF_STRIDE;
                let d = (dy + j) * dstride + dx;
                // 竖直方向的邻居跨行 —— 行拷到本地再走同一套横向内核最省事；
                // 但逐像素标量保留（这条在采样里占比小）。
                for i in 0..w {
                    dst[d + i] = avg2(src[s + i] as i32, src[s + i + BUF_STRIDE] as i32) as u8;
                }
            }
        }
        (true, true) => {
            for j in 0..w {
                let s = base + j * BUF_STRIDE;
                let d = (dy + j) * dstride + dx;
                #[cfg(target_arch = "aarch64")]
                crate::neon::hpel_avg4_row(src, s, dst, d, w, r4 as u32);
                #[cfg(not(target_arch = "aarch64"))]
                for i in 0..w {
                    dst[d + i] = avg4(
                        src[s + i] as i32,
                        src[s + i + 1] as i32,
                        src[s + i + BUF_STRIDE] as i32,
                        src[s + i + BUF_STRIDE + 1] as i32,
                    ) as u8;
                }
            }
        }
    }
}

/// 色度插值：`rnd=0` 走 h264（`+32`），`rnd=1` 走 VC-1 的 no-rnd（`+28`）。
#[allow(clippy::too_many_arguments)]
fn chroma_mc(
    dst: &mut [u8],
    dstride: usize,
    dx: usize,
    dy: usize,
    src: &[u8],
    base: usize,
    x: i32,
    y: i32,
    rnd: bool,
) {
    let a = (8 - x) * (8 - y);
    let b = x * (8 - y);
    let c = (8 - x) * y;
    let d = x * y;
    let bias = if rnd { 32 - 4 } else { 32 };
    #[cfg(target_arch = "aarch64")]
    {
        crate::neon::chroma_mc8(dst, dstride, dx, dy, src, base, a, b, c, d, bias as u32);
    }
    #[cfg(not(target_arch = "aarch64"))]
    for j in 0..8usize {
        for i in 0..8usize {
            let o = base + j * BUF_STRIDE + i;
            let s = a * src[o] as i32
                + b * src[o + 1] as i32
                + c * src[o + BUF_STRIDE] as i32
                + d * src[o + BUF_STRIDE + 1] as i32;
            dst[(dy + j) * dstride + dx + i] = ((s + bias) >> 6) as u8;
        }
    }
}

/// `median4()`（`mathops.h`）：四个数的中值（取中间两个的平均）。
fn median4(a: i32, b: i32, c: i32, d: i32) -> i32 {
    if a < b {
        if c < d {
            (b.min(d) + a.max(c)) / 2
        } else {
            (b.min(c) + a.max(d)) / 2
        }
    } else if c < d {
        (a.min(d) + b.max(c)) / 2
    } else {
        (a.min(c) + b.max(d)) / 2
    }
}

/// `get_chroma_mv()` 里那张「两个有效块」的配对表。
static CHROMA_INDEX2: [u8; 16] = [
    0, 0, 0, 0x01, 0, 0x02, 0x12, 0, 0, 0x03, 0x13, 0, 0x23, 0, 0, 0,
];

impl Ctx<'_, '_> {
    /// 亮度补偿：把缓冲里那 `k × k` 个参考像素按 `luts` 映射一遍。
    ///
    /// 逐行交替用两张表（`[0]` / `[1]`）—— 上游对**隔行**的两场各有一张表，
    /// 渐进式下退化成「按行的奇偶交替」，但交替这件事本身是语义的一部分，不能省。
    fn apply_ic(&self, plane: usize, buf: &mut [u8], k: usize, row_parity: i32) {
        let (lut0, lut1) = if plane == 0 {
            (&self.luts.last_y[0], &self.luts.last_y[1])
        } else {
            (&self.luts.last_uv[0], &self.luts.last_uv[1])
        };
        for j in 0..k {
            // 行 `j` 对应源行 `row_parity + j`（`row_parity = src_y - pad`）。
            let lut: &[u8; 256] = if (row_parity + j as i32) & 1 == 0 {
                lut0
            } else {
                lut1
            };
            for i in 0..k {
                // ⚠️ 与 `load_ref` **同一套下标**（缓冲的 (0,0) = 参考帧的 (src_x-pad, src_y-pad)）。
                let o = j * BUF_STRIDE + i;
                buf[o] = lut[buf[o] as usize];
            }
        }
    }

    /// 从参考帧取一块到本地缓冲（越界处边缘复制）。
    ///
    /// ⚠️ 缓冲的 **(0,0)** 对应参考帧的 `(src_x - pad, src_y - pad)`，而插值基准在
    /// `(pad, pad)` —— 与上游 `emulated_edge_mc` + `srcY += mspel*(1+linesize)` 同一套布局。
    /// 把源块放到 `(pad, pad)` 会让四分之一像素（`pad = 1`）整体错开一个像素：
    /// **平坦区看不出来，边缘才差 1** —— 最难查的那种。
    fn load_ref(&self, plane: usize, src_x: i32, src_y: i32, k: usize, pad: usize, buf: &mut [u8]) {
        let data = self.ref_plane(plane);
        let stride = self.plane_stride(plane);
        let (w, h) = if plane == 0 {
            (self.mb_width * 16, self.mb_height * 16)
        } else {
            (self.mb_width * 8, self.mb_height * 8)
        };
        let (x0, y0) = (src_x - pad as i32, src_y - pad as i32);
        // **快路**：整块都在画面内（绝大多数宏块都在）—— 逐行 `copy_from_slice`，
        // 省掉每像素两次 `clamp` 与一次边界检查。这是解码里调用次数最多的函数之一。
        if x0 >= 0 && y0 >= 0 && x0 + k as i32 <= w as i32 && y0 + k as i32 <= h as i32 {
            let (x0, y0) = (x0 as usize, y0 as usize);
            for j in 0..k {
                let s = (y0 + j) * stride + x0;
                buf[j * BUF_STRIDE..j * BUF_STRIDE + k].copy_from_slice(&data[s..s + k]);
            }
            return;
        }
        // 慢路：越界处边缘复制。
        for j in 0..k {
            let sy = (y0 + j as i32).clamp(0, h as i32 - 1) as usize;
            for i in 0..k {
                let sx = (x0 + i as i32).clamp(0, w as i32 - 1) as usize;
                buf[j * BUF_STRIDE + i] = data[sy * stride + sx];
            }
        }
    }

    /// `ff_vc1_mc_1mv()`（渐进式）。
    pub(crate) fn mc_1mv(&mut self, mbx: usize, mby: usize) {
        if !self.has_ref {
            return;
        }
        // ⚠️ 只取标量 —— `hdr.p.clone()` 会把两张位平面一起拷（每块一次 malloc + memcpy）。
        let (mx, my) = self.mv_of(mbx, mby, 0);
        let mb_pos = self.mbidx(mbx, mby);
        let mut uvmx = (mx + ((mx & 3) == 3) as i32) >> 1;
        let mut uvmy = (my + ((my & 3) == 3) as i32) >> 1;
        self.grids.luma_mv[mb_pos] = [uvmx as i16, uvmy as i16];

        // ---- 快路：整像素且无亮度补偿 ⇒ 输出就是参考块原样搬过来 ----
        //
        // 静态素材里「MV 预测出来恰好是 (0,0)」的宏块很多（跳过的宏块尤其如此）。
        // 走通用路径要 `load_ref`（19×19 逐像素 clamp）+ 256 次 `mspel_filter` 调用，
        // 而正确答案只是把参考块拷过来。`use_ic` 时必须绕开（那会改像素值）。
        if mx == 0 && my == 0 && !self.use_ic {
            let (dx, dy) = (mbx * 16, mby * 16);
            let ys = self.plane_stride(0);
            let cs = self.plane_stride(1);
            {
                let (src, dst) = (&self.ref_planes.y, &mut self.planes.y);
                for j in 0..16 {
                    let o = (dy + j) * ys + dx;
                    dst[o..o + 16].copy_from_slice(&src[o..o + 16]);
                }
            }
            let (cdx, cdy) = (dx / 2, dy / 2);
            for plane in [1usize, 2] {
                let (src, dst) = if plane == 1 {
                    (&self.ref_planes.u, &mut self.planes.u)
                } else {
                    (&self.ref_planes.v, &mut self.planes.v)
                };
                for j in 0..8 {
                    let o = (cdy + j) * cs + cdx;
                    dst[o..o + 8].copy_from_slice(&src[o..o + 8]);
                }
            }
            return;
        }
        if self.seq.fastuvmc {
            uvmx += if uvmx < 0 { uvmx & 1 } else { -(uvmx & 1) };
            uvmy += if uvmy < 0 { uvmy & 1 } else { -(uvmy & 1) };
        }
        let mspel = self.hdr.p.as_ref().unwrap().mspel as usize;
        let mut src_x = mbx as i32 * 16 + (mx >> 2);
        let mut src_y = mby as i32 * 16 + (my >> 2);
        let mut uvsrc_x = mbx as i32 * 8 + (uvmx >> 2);
        let mut uvsrc_y = mby as i32 * 8 + (uvmy >> 2);
        // Simple/Main profile 的裁剪范围（Advanced 才是另一套）。
        src_x = src_x.clamp(-16, (self.mb_width * 16) as i32);
        src_y = src_y.clamp(-16, (self.mb_height * 16) as i32);
        uvsrc_x = uvsrc_x.clamp(-8, (self.mb_width * 8) as i32);
        uvsrc_y = uvsrc_y.clamp(-8, (self.mb_height * 8) as i32);

        let mut buf = [0u8; BUF * BUF];
        let k = 17 + 2 * mspel;
        self.load_ref(0, src_x, src_y, k, mspel, &mut buf);
        if self.use_ic {
            self.apply_ic(0, &mut buf, k, src_y - mspel as i32);
        }
        let base = mspel * BUF_STRIDE + mspel;
        let stride = self.plane_stride(0);
        let (dx, dy) = (mbx * 16, mby * 16);
        let rnd = self.hdr.rnd as i32;
        let rnd_bit = self.hdr.rnd;
        {
            let dst = self.plane_mut(0);
            if mspel != 0 {
                let dxy = (((my & 3) << 2) | (mx & 3)) as usize;
                let mut tmp = [0i16; (16 + 3) * 16];
                mspel_mc::<16>(
                    dst,
                    stride,
                    dx,
                    dy,
                    &buf,
                    base,
                    (dxy & 3) as u32,
                    (dxy >> 2) as u32,
                    rnd,
                    &mut tmp,
                );
            } else {
                let dxy = ((my & 2) | ((mx & 2) >> 1)) as usize;
                hpel_mc(dst, stride, dx, dy, 16, &buf, base, dxy, rnd_bit);
            }
        }

        // 色度：永远是「四分之一像素双线性」。
        let cdx = uvmx & 3;
        let cdy = uvmy & 3;
        let cstride = self.plane_stride(1);
        let mut bufv = [0u8; BUF * BUF];
        self.load_ref(1, uvsrc_x, uvsrc_y, 9, 0, &mut buf);
        self.load_ref(2, uvsrc_x, uvsrc_y, 9, 0, &mut bufv);
        if self.use_ic {
            self.apply_ic(1, &mut buf, 9, uvsrc_y);
            self.apply_ic(2, &mut bufv, 9, uvsrc_y);
        }
        {
            let dst = self.plane_mut(1);
            chroma_mc(dst, cstride, dx / 2, dy / 2, &buf, 0, cdx << 1, cdy << 1, rnd_bit);
        }
        let dst = self.plane_mut(2);
        chroma_mc(dst, cstride, dx / 2, dy / 2, &bufv, 0, cdx << 1, cdy << 1, rnd_bit);
    }

    /// `ff_vc1_mc_4mv_luma()`（渐进式）。
    pub(crate) fn mc_4mv_luma(&mut self, mbx: usize, mby: usize, n: usize) {
        if !self.has_ref {
            return;
        }
        let (mx, my) = self.mv_of(mbx, mby, n);
        let mspel = self.hdr.p.as_ref().unwrap().mspel as usize;
        let (bx, by) = (
            mbx * 16 + (n & 1) * 8,
            mby * 16 + (n & 2) * 4,
        );
        // 快路：整像素且无亮度补偿 ⇒ 直接把参考块拷过来（与 `mc_1mv` 同一个理由）。
        if mx == 0 && my == 0 && !self.use_ic {
            let ys = self.plane_stride(0);
            let (src, dst) = (&self.ref_planes.y, &mut self.planes.y);
            for j in 0..8 {
                let o = (by + j) * ys + bx;
                dst[o..o + 8].copy_from_slice(&src[o..o + 8]);
            }
            return;
        }
        let mut src_x = bx as i32 + (mx >> 2);
        let mut src_y = by as i32 + (my >> 2);
        src_x = src_x.clamp(-16, (self.mb_width * 16) as i32);
        src_y = src_y.clamp(-16, (self.mb_height * 16) as i32);
        let mut buf = [0u8; BUF * BUF];
        let k = 9 + 2 * mspel;
        self.load_ref(0, src_x, src_y, k, mspel, &mut buf);
        if self.use_ic {
            self.apply_ic(0, &mut buf, k, src_y - mspel as i32);
        }
        let base = mspel * BUF_STRIDE + mspel;
        let stride = self.plane_stride(0);
        let dx = mbx * 16 + (n & 1) * 8;
        let dy = mby * 16 + (n & 2) * 4;
        let rnd = self.hdr.rnd as i32;
        let rnd_bit = self.hdr.rnd;
        let dst = self.plane_mut(0);
        if mspel != 0 {
            let dxy = (((my & 3) << 2) | (mx & 3)) as usize;
            let mut tmp = [0i16; (8 + 3) * 8];
            mspel_mc::<8>(
                dst,
                stride,
                dx,
                dy,
                &buf,
                base,
                (dxy & 3) as u32,
                (dxy >> 2) as u32,
                rnd,
                &mut tmp,
            );
        } else {
            let dxy = ((my & 2) | ((mx & 2) >> 1)) as usize;
            hpel_mc(dst, stride, dx, dy, 8, &buf, base, dxy, rnd_bit);
        }
    }

    /// `get_chroma_mv()`：由四个亮度块的 MV 推色度 MV（返回 `(mvx, mvy, 有效块数)`）。
    fn get_chroma_mv(&self, mbx: usize, mby: usize) -> (i32, i32, u32) {
        let b = [
            self.blk(mbx, mby, 0),
            self.blk(mbx, mby, 1),
            self.blk(mbx, mby, 2),
            self.blk(mbx, mby, 3),
        ];
        let mut idx = 0u32;
        for (k, &bi) in b.iter().enumerate() {
            // `!mb_type` = 「这个块不是帧内」= 有效。
            if self.grids.mb_type[bi] == 0 {
                idx |= 1 << k;
            }
        }
        let valid_count = idx.count_ones();
        let mv = |k: usize| {
            let m = self.grids.mv[b[k]];
            (m[0] as i32, m[1] as i32)
        };
        let (tx, ty) = match valid_count {
            4 => (
                median4(mv(0).0, mv(1).0, mv(2).0, mv(3).0),
                median4(mv(0).1, mv(1).1, mv(2).1, mv(3).1),
            ),
            3 => {
                let i = idx as usize;
                (
                    mid_pred(
                        mv((i > 0xd) as usize).0,
                        mv(1 + (i > 0xb) as usize).0,
                        mv(2 + (i > 0x7) as usize).0,
                    ),
                    mid_pred(
                        mv((i > 0xd) as usize).1,
                        mv(1 + (i > 0xb) as usize).1,
                        mv(2 + (i > 0x7) as usize).1,
                    ),
                )
            }
            2 => {
                let i = CHROMA_INDEX2[idx as usize] as usize;
                (
                    (mv(i >> 4).0 + mv(i & 0xf).0) / 2,
                    (mv(i >> 4).1 + mv(i & 0xf).1) / 2,
                )
            }
            _ => return (0, 0, 0),
        };
        (tx, ty, valid_count)
    }

    /// `ff_vc1_mc_4mv_chroma()`（渐进式）。
    pub(crate) fn mc_4mv_chroma(&mut self, mbx: usize, mby: usize) {
        if !self.has_ref {
            return;
        }
        let (tx, ty, valid_count) = self.get_chroma_mv(mbx, mby);
        let mb_pos = self.mbidx(mbx, mby);
        if valid_count == 0 {
            self.grids.luma_mv[mb_pos] = [0, 0];
            return;
        }
        let mut uvmx = (tx + ((tx & 3) == 3) as i32) >> 1;
        let mut uvmy = (ty + ((ty & 3) == 3) as i32) >> 1;
        self.grids.luma_mv[mb_pos] = [uvmx as i16, uvmy as i16];
        if self.seq.fastuvmc {
            uvmx += if uvmx < 0 { uvmx & 1 } else { -(uvmx & 1) };
            uvmy += if uvmy < 0 { uvmy & 1 } else { -(uvmy & 1) };
        }
        let mut uvsrc_x = mbx as i32 * 8 + (uvmx >> 2);
        let mut uvsrc_y = mby as i32 * 8 + (uvmy >> 2);
        uvsrc_x = uvsrc_x.clamp(-8, (self.mb_width * 8) as i32);
        uvsrc_y = uvsrc_y.clamp(-8, (self.mb_height * 8) as i32);
        let mut buf = [0u8; BUF * BUF];
        let mut bufv = [0u8; BUF * BUF];
        self.load_ref(1, uvsrc_x, uvsrc_y, 9, 0, &mut buf);
        self.load_ref(2, uvsrc_x, uvsrc_y, 9, 0, &mut bufv);
        if self.use_ic {
            self.apply_ic(1, &mut buf, 9, uvsrc_y);
            self.apply_ic(2, &mut bufv, 9, uvsrc_y);
        }
        let cstride = self.plane_stride(1);
        let (dx, dy) = (mbx * 8, mby * 8);
        let (cx, cy) = ((uvmx & 3) << 1, (uvmy & 3) << 1);
        let rnd_bit = self.hdr.rnd;
        {
            let dst = self.plane_mut(1);
            chroma_mc(dst, cstride, dx, dy, &buf, 0, cx, cy, rnd_bit);
        }
        let dst = self.plane_mut(2);
        chroma_mc(dst, cstride, dx, dy, &bufv, 0, cx, cy, rnd_bit);
    }
}
