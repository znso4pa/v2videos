//!  unsafe 内核。**每个内核都要有逐字节等价于安全路径的测试**（见 `tests/`）。
//!
//! 第一个上的是**半像素平均** —— 运动补偿里每个半像素块都要做
//! `dst[i] = (a[i] + b[i] + 1) >> 1`，是解码里最热的逐像素循环之一。
//! 选它打头是因为 `_mm_avg_epu8` / `vrhaddq_u8` 的语义**正好**就是这个式子，
//! 所以 SIMD 路径可以做到**逐字节相同**，不需要容差。

/// 本 crate 提供的内核（位掩码）。
pub const KERNEL_AVG: u32 = 1 << 0;
/// IDCT 列通道（`idct_cols`）。
pub const KERNEL_IDCT_COLS: u32 = 1 << 1;

/// 当前实际**会用到**的内核（位掩码）。
///
/// ⚠️ 关掉 `unsafe-kernels` feature 时这里是 0 —— 因为那时 `avg_rows` 走标量，
/// 内核根本没被调用。早先这里不看 feature、恒报可用，等于骗宿主。
pub fn available() -> u32 {
    #[cfg(feature = "unsafe-kernels")]
    {
        let mut mask = 0;
        if avg_available() {
            mask |= KERNEL_AVG;
        }
        if idct_cols_available() {
            mask |= KERNEL_IDCT_COLS;
        }
        mask
    }
    #[cfg(not(feature = "unsafe-kernels"))]
    {
        0
    }
}

/// 把本 crate 的内核装进解码器（进程内调一次即可）。
///
/// 目前只装 IDCT 列通道 —— `avg_rows` 没有装，因为实测运动补偿的平均
/// **已经由 LLVM 自动向量化成 `urhadd`**，手写反而没有收益（消融过：
/// 把整个平均删掉端到端也是 0）。
pub fn install() {
    #[cfg(all(feature = "unsafe-kernels", target_arch = "aarch64"))]
    v2v_mpeg12::decode::set_idct_cols(idct_cols);
}

#[cfg(feature = "unsafe-kernels")]
fn avg_available() -> bool {
    // x86_64 一定有 SSE2；AVX2 只是更快，没有也能跑。
    #[cfg(target_arch = "x86_64")]
    {
        true
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

// ---------------------------------------------------------------------------
// IDCT 列通道
// ---------------------------------------------------------------------------

#[cfg(feature = "unsafe-kernels")]
fn idct_cols_available() -> bool {
    cfg!(target_arch = "aarch64")
}

/// IDCT 列通道：分派到内核，没有内核就走 `v2v-mpeg12` 的标量参考实现。
///
/// ## 为什么只有列通道
///
/// 对固定频率 `i`，`b[i*8 + u]`（u = 0..7）在内存里是**连续的**，8 条 lane
/// 正好一个向量组 —— 天然可向量化。行通道相反：对固定 `u` 要跨 8 行取
/// （stride 8），得先转置，留标量更划算。
///
/// ## 逐位等价
///
/// 标量版里 `if c(4) != 0 { ... }` 那些分支只是「加 0 就跳过」的优化，
/// 加 0 本来就是空操作，所以向量版**无条件执行**，结果逐位相同。
/// `idct_cols_matches_scalar` 用穷举随机块钉住这一点。
pub fn idct_cols(b: &[i32; 64], out: &mut [i32; 64]) {
    #[cfg(all(feature = "unsafe-kernels", target_arch = "aarch64"))]
    {
        // 安全：`b`/`out` 都是 `&[i32; 64]`，内核只访问 `0..64`，不会越界；
        // NEON 是 AArch64 的基线，无需运行时检测。
        unsafe { idct_cols_neon(b, out) };
    }
    #[cfg(not(all(feature = "unsafe-kernels", target_arch = "aarch64")))]
    {
        v2v_mpeg12::decode::idct_cols_scalar(b, out);
    }
}

/// NEON 版 IDCT 列通道。**必须与 `idct_cols_scalar` 逐位相同。**
#[cfg(all(feature = "unsafe-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn idct_cols_neon(b: &[i32; 64], out: &mut [i32; 64]) {
    use std::arch::aarch64::*;
    const W1: u32 = 22725;
    const W2: u32 = 21407;
    const W3: u32 = 19266;
    const W4: u32 = 16383;
    const W5: u32 = 12873;
    const W6: u32 = 8867;
    const W7: u32 = 4520;

    // 8 条 lane（u = 0..7）拆成两个 4 lane 的向量。
    // 指针先取出来（裸指针是 Copy），闭包就不用可变借用 `out` 了。
    let bp = b.as_ptr();
    let op = out.as_mut_ptr();
    for half in 0..2usize {
        let u0 = half * 4;
        let ld = |i: usize| vld1q_u32(bp.add(i * 8 + u0) as *const u32);
        let (c0, c1, c2, c3) = (ld(0), ld(1), ld(2), ld(3));
        let (c4, c5, c6, c7) = (ld(4), ld(5), ld(6), ld(7));

        // DC 项的 +32 是 (1<<(COL_SHIFT-1))/W4 的整数商，不是 2^19/W4。
        let mut a0 = vmulq_n_u32(vaddq_u32(c0, vdupq_n_u32(32)), W4);
        let mut a1 = a0;
        let mut a2 = a0;
        let mut a3 = a0;
        a0 = vaddq_u32(a0, vmulq_n_u32(c2, W2));
        a1 = vaddq_u32(a1, vmulq_n_u32(c2, W6));
        a2 = vsubq_u32(a2, vmulq_n_u32(c2, W6));
        a3 = vsubq_u32(a3, vmulq_n_u32(c2, W2));

        let mut b0 = vaddq_u32(vmulq_n_u32(c1, W1), vmulq_n_u32(c3, W3));
        let mut b1 = vsubq_u32(vmulq_n_u32(c1, W3), vmulq_n_u32(c3, W7));
        let mut b2 = vsubq_u32(vmulq_n_u32(c1, W5), vmulq_n_u32(c3, W1));
        let mut b3 = vsubq_u32(vmulq_n_u32(c1, W7), vmulq_n_u32(c3, W5));

        a0 = vaddq_u32(a0, vmulq_n_u32(c4, W4));
        a1 = vsubq_u32(a1, vmulq_n_u32(c4, W4));
        a2 = vsubq_u32(a2, vmulq_n_u32(c4, W4));
        a3 = vaddq_u32(a3, vmulq_n_u32(c4, W4));

        b0 = vaddq_u32(b0, vmulq_n_u32(c5, W5));
        b1 = vsubq_u32(b1, vmulq_n_u32(c5, W1));
        b2 = vaddq_u32(b2, vmulq_n_u32(c5, W7));
        b3 = vaddq_u32(b3, vmulq_n_u32(c5, W3));

        a0 = vaddq_u32(a0, vmulq_n_u32(c6, W6));
        a1 = vsubq_u32(a1, vmulq_n_u32(c6, W2));
        a2 = vaddq_u32(a2, vmulq_n_u32(c6, W2));
        a3 = vsubq_u32(a3, vmulq_n_u32(c6, W6));

        b0 = vaddq_u32(b0, vmulq_n_u32(c7, W7));
        b1 = vsubq_u32(b1, vmulq_n_u32(c7, W5));
        b2 = vaddq_u32(b2, vmulq_n_u32(c7, W3));
        b3 = vsubq_u32(b3, vmulq_n_u32(c7, W1));

        // 结果：`(x as i32) >> 20`（算术右移，不是逻辑右移）。
        let st = |v: usize, x: uint32x4_t| {
            let s = vshrq_n_s32::<20>(vreinterpretq_s32_u32(x));
            vst1q_s32(op.add(v * 8 + u0), s);
        };
        st(0, vaddq_u32(a0, b0));
        st(1, vaddq_u32(a1, b1));
        st(2, vaddq_u32(a2, b2));
        st(3, vaddq_u32(a3, b3));
        st(4, vsubq_u32(a3, b3));
        st(5, vsubq_u32(a2, b2));
        st(6, vsubq_u32(a1, b1));
        st(7, vsubq_u32(a0, b0));
    }
}

/// 安全路径：`dst[i] = (a[i] + b[i] + 1) >> 1`。与 `v2v-mpeg12` 里
/// `predict_block` 的平均公式逐字节一致 —— 这是 oracle。
pub fn avg_rows_scalar(dst: &mut [u8], a: &[u8], b: &[u8]) {
    for i in 0..dst.len() {
        dst[i] = (((a[i] as u16) + (b[i] as u16) + 1) >> 1) as u8;
    }
}

/// 半像素平均。三个切片必须**等长**（不变量，调用方保证）。
///
/// `unsafe-kernels` feature 关掉时走标量 —— 这样「关掉 feature 就完全不碰
/// unsafe」才是真的（之前 feature 声明了却没人读，等于永远走 SIMD）。
pub fn avg_rows(dst: &mut [u8], a: &[u8], b: &[u8]) {
    assert_eq!(dst.len(), a.len(), "dst 与 a 不等长");
    assert_eq!(dst.len(), b.len(), "dst 与 b 不等长");
    #[cfg(feature = "unsafe-kernels")]
    {
        if avg_available() {
            #[cfg(target_arch = "x86_64")]
            {
                // 安全：切片等长已断言，内核只读写 `0..dst.len()`，不越界。
                unsafe { avg_rows_x86(dst, a, b) };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            {
                // 安全：同上，切片等长已断言，内核只读写 `0..dst.len()`。
                unsafe { avg_rows_neon(dst, a, b) };
                return;
            }
        }
    }
    avg_rows_scalar(dst, a, b);
}

#[cfg(all(feature = "unsafe-kernels", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn avg_rows_x86(dst: &mut [u8], a: &[u8], b: &[u8]) {
    use std::arch::x86_64::*;
    let n = dst.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let pd = dst.as_mut_ptr();
    let mut i = 0usize;
    // 一次 16 字节。`_mm_avg_epu8` 的语义就是 `(a + b + 1) >> 1`（无符号字节），
    // 与安全路径逐字节相同。
    while i + 16 <= n {
        let va = _mm_loadu_si128(pa.add(i) as *const __m128i);
        let vb = _mm_loadu_si128(pb.add(i) as *const __m128i);
        let v = _mm_avg_epu8(va, vb);
        _mm_storeu_si128(pd.add(i) as *mut __m128i, v);
        i += 16;
    }
    // 尾巴：标量收尾（不越界，因为循环条件保证了 i < n）。
    while i < n {
        *pd.add(i) = (((*pa.add(i) as u16) + (*pb.add(i) as u16) + 1) >> 1) as u8;
        i += 1;
    }
}

#[cfg(all(feature = "unsafe-kernels", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn avg_rows_neon(dst: &mut [u8], a: &[u8], b: &[u8]) {
    use std::arch::aarch64::*;
    let n = dst.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let pd = dst.as_mut_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        let va = vld1q_u8(pa.add(i));
        let vb = vld1q_u8(pb.add(i));
        vst1q_u8(pd.add(i), vrhaddq_u8(va, vb));
        i += 16;
    }
    while i < n {
        *pd.add(i) = (((*pa.add(i) as u16) + (*pb.add(i) as u16) + 1) >> 1) as u8;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 伪随机，但可复现（不引第三方 crate）。
    fn lcg(seed: &mut u64) -> u8 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*seed >> 33) as u8
    }

    #[test]
    fn avg_matches_scalar_byte_for_byte() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        for len in [0usize, 1, 7, 15, 16, 17, 31, 64, 720, 1024] {
            let a: Vec<u8> = (0..len).map(|_| lcg(&mut seed)).collect();
            let b: Vec<u8> = (0..len).map(|_| lcg(&mut seed)).collect();
            let mut want = vec![0u8; len];
            avg_rows_scalar(&mut want, &a, &b);
            let mut got = vec![0u8; len];
            avg_rows(&mut got, &a, &b);
            assert_eq!(got, want, "长度 {len} 不一致");
        }
    }

    #[test]
    fn avg_extremes() {
        // 0 和 255 的边界：(0+255+1)>>1 = 128，*(255+255+1)>>1 = 255。
        let a = vec![0u8, 255, 128, 1];
        let b = vec![255u8, 255, 127, 0];
        let mut got = vec![0u8; 4];
        avg_rows(&mut got, &a, &b);
        assert_eq!(got, vec![128, 255, 128, 1]);
    }

    /// IDCT 列通道内核必须与 `v2v-mpeg12` 的标量参考实现**逐位相同**。
    ///
    /// 输入取 i16 全范围（真实输入是行变换后截成 i16 的值），覆盖正负、
    /// 溢出回绕、以及「只有 DC」「全零」这些边界。整数 IDCT 错一位不是
    /// 「画质差一点」，而是整块偏 —— 而且会和参考实现的对账冲突。
    #[test]
    fn idct_cols_matches_scalar() {
        let mut seed = 0x0bad_c0de_dead_beefu64;
        let mut lcg = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as u32
        };

        let check = |b: &[i32; 64]| {
            let mut want = [0i32; 64];
            v2v_mpeg12::decode::idct_cols_scalar(b, &mut want);
            let mut got = [0i32; 64];
            idct_cols(b, &mut got);
            assert_eq!(got, want, "输入 {b:?}");
        };

        // 边界：全零、只有 DC（正/负）、DC + 一个高频
        check(&[0i32; 64]);
        for dc in [0i32, 1, -1, 1024, -1024, 32767, -32768] {
            let mut b = [0i32; 64];
            for v in 0..8 {
                b[v * 8] = dc;
            }
            check(&b);
        }
        // 随机：i16 全范围（含会触发 u32 回绕的大值）
        for _ in 0..500 {
            let mut b = [0i32; 64];
            for x in b.iter_mut() {
                *x = (lcg() as u16) as i16 as i32;
            }
            check(&b);
        }
        // 随机：稀疏（大部分为 0，接近真实块的形态）
        for _ in 0..500 {
            let mut b = [0i32; 64];
            for x in b.iter_mut() {
                if lcg() % 4 == 0 {
                    *x = ((lcg() as u16) as i16 as i32) / 8;
                }
            }
            check(&b);
        }
    }
}
