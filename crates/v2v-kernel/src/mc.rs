//! 运动补偿与帧缓冲 —— H.263 / MPEG-4 系那一套（半像素两抽头）。
//!
//! ## 为什么在 `v2v-kernel` 而不是某个编解码器里
//!
//! 这一套是**格式无关**的：`v2v-msmpeg4`（MP43 编解码）、`v2v-mp4v`（MPEG-4 Part 2
//! 编解码）、`v2v-vc1`（只借 `Planes` 与 `mid_pred`）都用它。半像素搬运在
//! H.263 / MPEG-4 系里是**同一份**，复刻第二份才是真会跑偏的那份。
//!
//! ## 边界
//!
//! 收的是「给定参考帧 + 运动矢量，算出预测块」这件事本身，以及它需要的帧缓冲
//! （[`Frame`] / [`Planes`]）与中值预测（[`mid_pred`]）。
//!
//! **不收**：
//! - 运动矢量的**解码 / 编码**（`decode_motion` / MV VLC / 分片位）—— 那是语法，
//!   各家不同，留在各自的编解码器里；
//! - **WMV2 的 mspel**（四抽头 `(9(s₀+s₁) − (s₋₁+s₂) + 8) >> 4`、`hshift`、
//!   19×19 窗口）—— 只有 WMV2 用，留在 `v2v-msmpeg4` 的 `frame.rs`；
//! - 环路滤波（只有 WMV2 开）、DC 预测（按 `Version` 分支）。
//!
//! [`edge_emu_into`] 与 [`put_pixels`] 是**低层积木**，公开是因为 mspel 那条路
//! 要用它们；正常调用方请用 [`mc_mb`] / [`mc_luma16`] / [`mc_chroma8`]。

/// 解码出来的一帧（YUV420P，尺寸已裁到真实宽高）。
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// 宏块对齐、**未裁剪**的平面。运动补偿要在它上面做（会读到真实高度以外的行）。
///
/// ⚠️ 这是**共享**的：`v2v-mp4v` 的 P 帧编码、`v2v-vc1` 的解码都用它（连同下面那几个
/// MC 内核）。从解码器里提出来公开，是因为半像素搬运那一套在 H.263 / MPEG-4 系里
/// 是同一份 —— 复刻第二份才是真会跑偏的那份。
#[derive(Debug, Clone)]
pub struct Planes {
    pub pw: usize,
    pub ph: usize,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl Planes {
    /// 宏块对齐、清零的平面组。
    pub fn new(pw: usize, ph: usize) -> Self {
        Self {
            pw,
            ph,
            y: vec![0u8; pw * ph],
            u: vec![0u8; (pw / 2) * (ph / 2)],
            v: vec![0u8; (pw / 2) * (ph / 2)],
        }
    }
}

/// MPEG-4 的标准 zig-zag 扫描顺序（`ff_zigzag_direct`）。
/// 帧内（无 ac_pred）与**所有帧间块**都用它（`inter_scantable` 也是 zig-zag）。
pub const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// H.263 的中值预测（`mid_pred`）。
pub fn mid_pred(a: i32, b: i32, c: i32) -> i32 {
    let mut b = b;
    if a > b {
        if c > b {
            if c > a {
                b = a;
            } else {
                b = c;
            }
        }
    } else if b > c {
        if c > a {
            b = c;
        } else {
            b = a;
        }
    }
    b
}

/// 边缘复制：把 `bw × bh` 的区域填成
/// `src[clamp(src_y+y)][clamp(src_x+x)]`。
///
/// ⚠️ **不分配**：旧版每块一次 `vec![0u8; bw*bh]`（luma 一次 MC 就 289 字节），
/// 实测 `edge_emu` 一个人占整条链路的 **5.1%**（还带 malloc/free 的抖动）。
/// 现在由调用方给一块**栈上**的 scratch —— 同样的值，少一趟堆。
///
/// 逐像素的两次 `clamp` 也去掉了：x 方向的越界形状**整块共用**，所以先把
/// 「左侧贴边 / 中间连续 / 右侧贴边」三段算出来，每行只做一次切片拷贝 + 两次填充。
pub fn edge_emu_into(
    out: &mut [u8],
    src: &[u8],
    sstride: usize,
    src_x: i32,
    src_y: i32,
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) {
    debug_assert!(out.len() >= bw * bh);
    debug_assert!(w > 0 && h > 0);

    // x 方向三段（对每一行都一样）：
    //   [0, left)        贴左边界，取 row[0]
    //   [left, bw-right) 连续区间，从 row[src_x + left] 起整段拷
    //   [bw-right, bw)   贴右边界，取 row[w-1]
    let left = (-src_x).clamp(0, bw as i32) as usize;
    let right = (src_x + bw as i32 - w as i32).clamp(0, bw as i32) as usize;
    let right = right.min(bw - left);
    let mid = bw - left - right;

    for y in 0..bh {
        let sy = (src_y + y as i32).clamp(0, h as i32 - 1) as usize;
        let row = &src[sy * sstride..sy * sstride + w];
        let d = &mut out[y * bw..y * bw + bw];
        d[..left].fill(row[0]);
        if mid > 0 {
            let sx0 = (src_x + left as i32) as usize;
            debug_assert!(sx0 + mid <= w, "连续段越出参考帧");
            d[left..left + mid].copy_from_slice(&row[sx0..sx0 + mid]);
        }
        d[bw - right..].fill(row[w - 1]);
    }
}

/// 半像素双线性搬运（`put_pixels` 一族）。
///
/// `dxy` 的 bit0 = 水平半像素、bit1 = 垂直半像素。带舍入时：
/// x2/y2 用 `(a+b+1)>>1`，xy2 用 `(a+b+c+d+2)>>2`；不带舍入各减 1。
///
/// ⚠️ 四种相位**分开写**、按**行切片**取源：旧版把相位判断放在内层循环里、
/// 还用一个逐像素算地址的闭包（每像素 1~4 次带边界检查的索引）。
/// **算法一字未改**（相加、移位、舍入常数全照旧），所以输出逐字节不变。
pub fn put_pixels(
    dst: &mut [u8],
    dstride: usize,
    dx: usize,
    dy: usize,
    src: &[u8],
    sstride: usize,
    sx: usize,
    sy: usize,
    w: usize,
    h: usize,
    dxy: usize,
    no_rounding: bool,
) {
    let hx = dxy & 1 != 0;
    let hy = dxy & 2 != 0;
    let (r1, r2) = if no_rounding {
        (0u32, 1u32)
    } else {
        (1u32, 2u32)
    };
    // 横向半像素要多读一格（`src_x + w`）；纵向半像素才碰下一行。
    let sw = w + usize::from(hx);
    for y in 0..h {
        let base = (sy + y) * sstride + sx;
        let top = &src[base..base + sw];
        let dbase = (dy + y) * dstride + dx;
        let d = &mut dst[dbase..dbase + w];
        // ⚠️ 三个平均分支都写成 `zip`/`windows(2)` 的形态（「正好 w 个」由类型保证，
        // 内层是纯算术）—— 与 `sad_luma16`/`mspel8_v` 同一套写法。
        // 逐像素的公式与舍入常数一个字没动。
        match (hx, hy) {
            // ⚠️ 不用 `copy_from_slice`：w = 8/16 这种小尺寸它会变成一次真的
            // `memcpy` 调用（实测 `memmove` 占 MP43 解码的 8%）。`zip` 循环
            // 让 LLVM 内联成几条向量指令。
            (false, false) => {
                for (slot, v) in d.iter_mut().zip(top.iter()) {
                    *slot = *v;
                }
            }
            (true, false) => {
                for (slot, (a, b)) in d.iter_mut().zip(top.iter().zip(top[1..].iter())) {
                    *slot = ((*a as u32 + *b as u32 + r1) >> 1) as u8;
                }
            }
            (false, true) => {
                let bbase = (sy + y + 1) * sstride + sx;
                let bot = &src[bbase..bbase + sw];
                for (slot, (a, b)) in d.iter_mut().zip(top.iter().zip(bot.iter())) {
                    *slot = ((*a as u32 + *b as u32 + r1) >> 1) as u8;
                }
            }
            (true, true) => {
                let bbase = (sy + y + 1) * sstride + sx;
                let bot = &src[bbase..bbase + sw];
                for (slot, (t, b)) in d
                    .iter_mut()
                    .zip(top.windows(2).zip(bot.windows(2)))
                {
                    *slot = ((t[0] as u32 + t[1] as u32 + b[0] as u32 + b[1] as u32 + r2) >> 2)
                        as u8;
                }
            }
        }
    }
}

/// 一次运动补偿的参数：半像素相位、参考坐标、是否需要边缘复制。
pub struct McParams {
    pub dxy: usize,
    pub uvdxy: usize,
    pub src_x: i32,
    pub src_y: i32,
    pub need_emu: bool,
}

/// 算一个宏块的 MC 参数（H.263 / MPEG-4 的半像素一路）。
pub fn mc_params(mbx: usize, mby: usize, mv: (i32, i32), pw: usize, ph: usize) -> McParams {
    let (mx, my) = mv;
    let dxy = (((my & 1) << 1) | (mx & 1)) as usize;
    let src_x = (mbx as i32) * 16 + (mx >> 1);
    let src_y = (mby as i32) * 16 + (my >> 1);

    let h_edge = pw as i32;
    let v_edge = ph as i32;
    // 越界判据：`(unsigned)` 强制转换让负数必然触发（等价于「任一坐标为负就触发」）。
    let limit_x = (h_edge - (mx & 1) - 15).max(0);
    let limit_y = (v_edge - (my & 1) - 16 + 1).max(0);
    let need_emu = (src_x as u32) >= (limit_x as u32) || (src_y as u32) >= (limit_y as u32);
    let uvdxy = dxy | ((my as usize) & 2) | (((mx as usize) & 2) >> 1);
    McParams {
        dxy,
        uvdxy,
        src_x,
        src_y,
        need_emu,
    }
}

/// 16×16 亮度块的半像素搬运（写到 `dest` 的 `(dx,dy)` 处）。返回是否走了边缘复制。
pub fn mc_luma16(
    dest: &mut [u8],
    dstride: usize,
    dx: usize,
    dy: usize,
    refp: &Planes,
    p: &McParams,
    no_rounding: bool,
) -> bool {
    // 裁剪范围用**参考帧自己的**尺寸：它才是真正被索引的缓冲。
    // scratch 在**栈上**（17×17 = 289 字节）—— 旧版这里每块一次 `vec![]`。
    let mut ybuf = [0u8; 17 * 17];
    let (ysrc, ystride, yx, yy) = if p.need_emu {
        edge_emu_into(
            &mut ybuf,
            &refp.y,
            refp.pw,
            p.src_x,
            p.src_y,
            17,
            17,
            refp.pw,
            refp.ph,
        );
        (&ybuf[..], 17usize, 0usize, 0usize)
    } else {
        (&refp.y[..], refp.pw, p.src_x as usize, p.src_y as usize)
    };
    put_pixels(
        dest,
        dstride,
        dx,
        dy,
        ysrc,
        ystride,
        yx,
        yy,
        16,
        16,
        p.dxy,
        no_rounding,
    );
    p.need_emu
}

/// 半像素插值 + SAD **融合**：`cur`（16×16 的当前宏块）与 `mv` 处的参考块比。
///
/// ⚠️ 公式与 `put_pixels` **逐字相同**（同一套舍入常数），所以值与
/// 「先 `mc_luma16` 拷进缓冲、再逐像素求和」**逐位相同** —— 只是省掉一趟写 + 一趟读。
/// 这条是运动估计的内层：`v2v-mp4v` 与 `v2v-msmpeg4` 两个编码器都靠它
/// （同一份实现，不是各写一份）。
///
/// **越界（需要边缘复制）时不走融合那条路** —— 那种情况仍交给 `mc_luma16`
/// （它要按边界钳位取像素），`tmp` 就是给它用的 16×16 缓冲。
///
/// ⚠️ 切片长度必须跟着**相位**走：横向半像素（`hx`）才多读一格（`src_x + 16`），
/// 纵向半像素（`hy`）才碰下一行。`need_emu` 的边界正是按这个定的 ——
/// 无条件多切一格会在「最后一行 + 最后一列」越界 panic。
pub fn sad_luma16(
    cur: &[u8; 256],
    refp: &Planes,
    p: &McParams,
    no_rounding: bool,
    tmp: &mut [u8; 256],
) -> u64 {
    sad_luma16_capped(cur, refp, p, no_rounding, tmp, u32::MAX) as u64
}

/// 一行 16 个像素的绝对差之和。
///
/// ⚠️ 写成 **16 通道累加、最后收口**，不是 `zip` 逐字节 —— 后者 LLVM 只部分向量化。
/// 实测（`-C target-cpu=native`，真实行距）：现状 15.85 ns/次 → 重塑 9.51 ns/次（1.67×）。
/// 值完全相同（都是 16 个 `abs_diff` 的和，u16 通道不会溢出：单个差 ≤ 255，16 个 ≤ 4080）。
///
/// 内层从 `(a as i32 - b as i32).unsigned_abs() as u64` 换成 `u8::abs_diff` 累加到
/// `u32`（值完全相同，但少两次转换、更容易向量化）。
#[inline(always)]
fn sad_row16(cur: &[u8], refr: &[u8]) -> u32 {
    let mut lane = [0u16; 16];
    for i in 0..16 {
        lane[i] = cur[i].abs_diff(refr[i]) as u16;
    }
    let mut s = 0u32;
    for v in lane {
        s += v as u32;
    }
    s
}

/// 带**上限**的 SAD：累计值一到 `cap` 就停，直接返回 `cap`。
///
/// ⚠️ 提前退出**不改任何结果**：调用方只做 `s < best` 的比较，返回 `cap` 只会让这个
/// 候选「不赢」—— 与算完再比完全等价。**但绝不能返回小于真值的数**（那会让本该输的赢）。
/// 运动估计里一个宏块要评 150+ 个候选，绝大多数在第一行就注定输了。
#[inline]
pub fn sad_luma16_capped(
    cur: &[u8; 256],
    refp: &Planes,
    p: &McParams,
    no_rounding: bool,
    tmp: &mut [u8; 256],
    cap: u32,
) -> u32 {
    if p.need_emu {
        mc_luma16(tmp, 16, 0, 0, refp, p, no_rounding);
        // 分块（16 个一组）累加，**块末**才查上限 —— 逐像素早退会把向量化彻底挡掉：
        // 实测这一条形状 36.94 → 4.66 ns/次（7.9×），生成的正是 `uabd.16b` + `udot.4s`。
        // 返回值不变（两边都在「总和 ≥ cap」时返回 `cap`）。
        let mut acc = 0u32;
        for c in 0..16usize {
            acc += sad_row16(&cur[c * 16..c * 16 + 16], &tmp[c * 16..c * 16 + 16]);
            if acc >= cap {
                return cap;
            }
        }
        return acc;
    }
    let pw = refp.pw;
    let base = p.src_y as usize * pw + p.src_x as usize;
    // 非边界（绝大多数候选）：走共享 SIMD 内核 —— 一个 16×16 整块**一次调用**，
    // 半像素插值在寄存器里做，不再先写 `rv[16]` 临时数组再读一遍。
    // 提前退出的语义（每行查一次 cap、命中返回 cap）与旧实现逐位相同。
    crate::simd::sad_block16(cur, &refp.y, pw, base, p.dxy, no_rounding, cap)
}

/// 8×8 色度块的半像素搬运（`which`：0 = U，1 = V）。半像素相位由 `uvdxy` 表示。
pub fn mc_chroma8(
    dest: &mut [u8],
    dstride: usize,
    dx: usize,
    dy: usize,
    refp: &Planes,
    which: usize,
    p: &McParams,
    no_rounding: bool,
) {
    let cw = refp.pw / 2;
    let (ew, eh) = (refp.pw >> 1, refp.ph >> 1);
    let uvsrc_x = p.src_x >> 1;
    let uvsrc_y = p.src_y >> 1;
    let plane: &[u8] = if which == 0 { &refp.u } else { &refp.v };
    let mut buf = [0u8; 9 * 9];
    let (src, sstride, sx, sy) = if p.need_emu {
        edge_emu_into(&mut buf, plane, cw, uvsrc_x, uvsrc_y, 9, 9, ew, eh);
        (&buf[..], 9usize, 0usize, 0usize)
    } else {
        (plane, cw, uvsrc_x as usize, uvsrc_y as usize)
    };
    put_pixels(
        dest,
        dstride,
        dx,
        dy,
        src,
        sstride,
        sx,
        sy,
        8,
        8,
        p.uvdxy,
        no_rounding,
    );
}

/// 一个宏块的 16×16 运动补偿（`MV_TYPE_16X16` 一路）。返回 `(dxy, 是否走了边缘复制)`。
pub fn mc_mb(
    dest: &mut Planes,
    refp: &Planes,
    mbx: usize,
    mby: usize,
    mv: (i32, i32),
    no_rounding: bool,
) -> (usize, bool) {
    let p = mc_params(mbx, mby, mv, dest.pw, dest.ph);
    let emu = mc_luma16(
        &mut dest.y,
        dest.pw,
        mbx * 16,
        mby * 16,
        refp,
        &p,
        no_rounding,
    );
    mc_chroma8(
        &mut dest.u,
        dest.pw / 2,
        mbx * 8,
        mby * 8,
        refp,
        0,
        &p,
        no_rounding,
    );
    mc_chroma8(
        &mut dest.v,
        dest.pw / 2,
        mbx * 8,
        mby * 8,
        refp,
        1,
        &p,
        no_rounding,
    );
    (p.dxy, emu)
}
