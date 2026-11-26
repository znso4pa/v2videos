//! NEON 版 radix-2 FFT 的「级」（f64，2 lane）。
//!
//! ## 为什么这里值得写，而解码器那批内核不值得
//!
//! 判据是「调用粒度摊不摊得开向量宽度」。解码器上试过的那几个内核是**按 8×8 块**
//! 调的（半像素平均、IDCT 列通道），一次调用只干 64 个像素，调用开销 + 无法跨块
//! 流水就把向量宽度吃掉了 —— 实测端到端 0%（`sb_rust` vs `sb_idct`，0.22 对 0.22
//! ms/帧）。FFT 相反：**一次调用是一整段 256~2048 点的级**，每次都是几百个蝶形，
//! 摊得开。
//!
//! ## 为什么是 2 lane 而不是更多
//!
//! 变换是 f64。AArch64 的 `f64x2` 就是全部宽度 —— 想再宽只能降到 f32，而 f32
//! 会把 AAC / WMA / Vorbis 三条线的对账判据全部作废（那是改精度，不是改实现）。
//!
//! ## 逐位等价是怎么保证的
//!
//! 只用了 `vmulq_f64` / `vaddq_f64` / `vsubq_f64`，**一次 `vfmaq_f64` 都没有**。
//! 乘加融合会改变舍入，`a*b + c*d` 与 `fma(a, b, c*d)` 的末位可以不同。
//! 只要运算的种类、顺序、次数与标量版一字不差，结果就是**同一串 IEEE 运算**，
//! 逐位相同。`neon_matches_scalar` 比 `to_bits()` 钉住这一点。

use std::arch::aarch64::*;

use crate::fft::{bit_reverse_permute_pub, twiddles_for};

/// 与 [`crate::fft::fft_in_place_scalar`] **逐位相同**的 NEON 实现。
///
/// 位反转走共用实现：它是纯访存 + 交换，2 lane 的向量化没有意义（而且要处理
/// 任意配对，反而更慢）。
pub(crate) fn stages_neon(re: &mut [f64], im: &mut [f64]) {
    bit_reverse_permute_pub(re, im);
    let len = re.len();
    let mut size = 2usize;
    while size * 2 <= len {
        // 安全：`size` 是 2 的幂、`2*size <= len`，内核只访问 `base..base+2*size`。
        unsafe { fuse_two_stages_neon(re, im, size) };
        size <<= 2;
    }
    if size <= len {
        // 安全：`size <= len`，内核只访问 `base..base+size`。
        unsafe { radix2_stage_neon(re, im, size) };
    }
}

/// 把 `size` 与 `2·size` 两级融合成一趟（2 lane）。
///
/// # Safety
///
/// `size` 必须是 2 的幂、`size >= 2`、`2 * size <= re.len() == im.len()`。
/// 调用方（[`stages_neon`] 的循环）保证这一点；内核只访问
/// `base .. base + 2*size`，`base` 按 `2*size` 步进。
unsafe fn fuse_two_stages_neon(re: &mut [f64], im: &mut [f64], size: usize) {
    let block = 2 * size;
    let q = size / 2;
    let tw_inner = twiddles_for(size);
    let tw_outer = twiddles_for(block);
    let (outer_lo, outer_hi) = tw_outer.split_at(q);
    let len = re.len();

    // 裸指针一次取出来：循环里不再借 `re` / `im`，四分区互不重叠。
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let tws = tw_inner.as_ptr() as *const f64;
    let twl = outer_lo.as_ptr() as *const f64;
    let twh = outer_hi.as_ptr() as *const f64;

    let mut base = 0usize;
    while base < len {
        let r0 = rp.add(base);
        let r1 = r0.add(q);
        let r2 = r1.add(q);
        let r3 = r2.add(q);
        let i0 = ip.add(base);
        let i1 = i0.add(q);
        let i2 = i1.add(q);
        let i3 = i2.add(q);

        let mut k = 0usize;
        while k + 2 <= q {
            // `vld2q_f64` 正好把「(re, im) 相邻」的旋转因子表拆成两条 lane 向量。
            let ws = vld2q_f64(tws.add(2 * k));
            let wl = vld2q_f64(twl.add(2 * k));
            let wh = vld2q_f64(twh.add(2 * k));
            let (ws_re, ws_im) = (ws.0, ws.1);
            let (wl_re, wl_im) = (wl.0, wl.1);
            let (wh_re, wh_im) = (wh.0, wh.1);

            let r0v = vld1q_f64(r0.add(k));
            let i0v = vld1q_f64(i0.add(k));
            let r1v = vld1q_f64(r1.add(k));
            let i1v = vld1q_f64(i1.add(k));
            let r2v = vld1q_f64(r2.add(k));
            let i2v = vld1q_f64(i2.add(k));
            let r3v = vld1q_f64(r3.add(k));
            let i3v = vld1q_f64(i3.add(k));

            // 第一级：(Q0[k], Q1[k]) 与 (Q2[k], Q3[k])，同一个 W_size^k。
            let t0r = vsubq_f64(vmulq_f64(r1v, ws_re), vmulq_f64(i1v, ws_im));
            let t0i = vaddq_f64(vmulq_f64(r1v, ws_im), vmulq_f64(i1v, ws_re));
            let p0r = vaddq_f64(r0v, t0r);
            let p0i = vaddq_f64(i0v, t0i);
            let p1r = vsubq_f64(r0v, t0r);
            let p1i = vsubq_f64(i0v, t0i);

            let t2r = vsubq_f64(vmulq_f64(r3v, ws_re), vmulq_f64(i3v, ws_im));
            let t2i = vaddq_f64(vmulq_f64(r3v, ws_im), vmulq_f64(i3v, ws_re));
            let p2r = vaddq_f64(r2v, t2r);
            let p2i = vaddq_f64(i2v, t2i);
            let p3r = vsubq_f64(r2v, t2r);
            let p3i = vsubq_f64(i2v, t2i);

            // 第二级：(P0, P2) 用 W_block^k，(P1, P3) 用 W_block^{k+q}。
            let u0r = vsubq_f64(vmulq_f64(p2r, wl_re), vmulq_f64(p2i, wl_im));
            let u0i = vaddq_f64(vmulq_f64(p2r, wl_im), vmulq_f64(p2i, wl_re));
            vst1q_f64(r0.add(k), vaddq_f64(p0r, u0r));
            vst1q_f64(i0.add(k), vaddq_f64(p0i, u0i));
            vst1q_f64(r2.add(k), vsubq_f64(p0r, u0r));
            vst1q_f64(i2.add(k), vsubq_f64(p0i, u0i));

            let u1r = vsubq_f64(vmulq_f64(p3r, wh_re), vmulq_f64(p3i, wh_im));
            let u1i = vaddq_f64(vmulq_f64(p3r, wh_im), vmulq_f64(p3i, wh_re));
            vst1q_f64(r1.add(k), vaddq_f64(p1r, u1r));
            vst1q_f64(i1.add(k), vaddq_f64(p1i, u1i));
            vst1q_f64(r3.add(k), vsubq_f64(p1r, u1r));
            vst1q_f64(i3.add(k), vsubq_f64(p1i, u1i));

            k += 2;
        }

        // 尾巴：`q` 是 2 的幂，所以最多剩 1 个。表达式与标量版一字不差。
        while k < q {
            let (ws_re, ws_im) = tw_inner[k];
            let (wl_re, wl_im) = outer_lo[k];
            let (wh_re, wh_im) = outer_hi[k];
            let (r0k, i0k) = (*r0.add(k), *i0.add(k));
            let (r1k, i1k) = (*r1.add(k), *i1.add(k));
            let (r2k, i2k) = (*r2.add(k), *i2.add(k));
            let (r3k, i3k) = (*r3.add(k), *i3.add(k));

            let t0r = r1k * ws_re - i1k * ws_im;
            let t0i = r1k * ws_im + i1k * ws_re;
            let p0r = r0k + t0r;
            let p0i = i0k + t0i;
            let p1r = r0k - t0r;
            let p1i = i0k - t0i;

            let t2r = r3k * ws_re - i3k * ws_im;
            let t2i = r3k * ws_im + i3k * ws_re;
            let p2r = r2k + t2r;
            let p2i = i2k + t2i;
            let p3r = r2k - t2r;
            let p3i = i2k - t2i;

            let u0r = p2r * wl_re - p2i * wl_im;
            let u0i = p2r * wl_im + p2i * wl_re;
            *r0.add(k) = p0r + u0r;
            *i0.add(k) = p0i + u0i;
            *r2.add(k) = p0r - u0r;
            *i2.add(k) = p0i - u0i;

            let u1r = p3r * wh_re - p3i * wh_im;
            let u1i = p3r * wh_im + p3i * wh_re;
            *r1.add(k) = p1r + u1r;
            *i1.add(k) = p1i + u1i;
            *r3.add(k) = p1r - u1r;
            *i3.add(k) = p1i - u1i;

            k += 1;
        }

        base += block;
    }
}

/// 单级 radix-2 蝶形（2 lane）。
///
/// # Safety
///
/// `size` 必须是 2 的幂、`size >= 2`、`size <= re.len() == im.len()`；
/// 内核只访问 `base .. base + size`。
unsafe fn radix2_stage_neon(re: &mut [f64], im: &mut [f64], size: usize) {
    let half = size / 2;
    let tw = twiddles_for(size);
    let len = re.len();
    let rp = re.as_mut_ptr();
    let ip = im.as_mut_ptr();
    let twp = tw.as_ptr() as *const f64;

    let mut base = 0usize;
    while base < len {
        let a_re = rp.add(base);
        let b_re = a_re.add(half);
        let a_im = ip.add(base);
        let b_im = a_im.add(half);

        let mut k = 0usize;
        while k + 2 <= half {
            let w = vld2q_f64(twp.add(2 * k));
            let (w_re, w_im) = (w.0, w.1);
            let ar = vld1q_f64(a_re.add(k));
            let ai = vld1q_f64(a_im.add(k));
            let br = vld1q_f64(b_re.add(k));
            let bi = vld1q_f64(b_im.add(k));

            let tr = vsubq_f64(vmulq_f64(br, w_re), vmulq_f64(bi, w_im));
            let ti = vaddq_f64(vmulq_f64(br, w_im), vmulq_f64(bi, w_re));
            vst1q_f64(b_re.add(k), vsubq_f64(ar, tr));
            vst1q_f64(b_im.add(k), vsubq_f64(ai, ti));
            vst1q_f64(a_re.add(k), vaddq_f64(ar, tr));
            vst1q_f64(a_im.add(k), vaddq_f64(ai, ti));

            k += 2;
        }

        while k < half {
            let (w_re, w_im) = tw[k];
            let (ar, ai) = (*a_re.add(k), *a_im.add(k));
            let (br, bi) = (*b_re.add(k), *b_im.add(k));
            let tr = br * w_re - bi * w_im;
            let ti = br * w_im + bi * w_re;
            *b_re.add(k) = ar - tr;
            *b_im.add(k) = ai - ti;
            *a_re.add(k) = ar + tr;
            *a_im.add(k) = ai + ti;
            k += 1;
        }

        base += size;
    }
}
