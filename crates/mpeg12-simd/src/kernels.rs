//!  unsafe 内核。**每个内核都要有逐字节等价于安全路径的测试**（见 `tests/`）。
//!
//! 第一个上的是**半像素平均** —— 运动补偿里每个半像素块都要做
//! `dst[i] = (a[i] + b[i] + 1) >> 1`，是解码里最热的逐像素循环之一。
//! 选它打头是因为 `_mm_avg_epu8` / `vrhaddq_u8` 的语义**正好**就是这个式子，
//! 所以 SIMD 路径可以做到**逐字节相同**，不需要容差。

/// 本 crate 提供的内核（位掩码）。
pub const KERNEL_AVG: u32 = 1 << 0;

/// 当前CPU 上实际可用的内核（运行时检测，不是编译期）。
pub fn available() -> u32 {
    let mut mask = 0;
    if avg_available() {
        mask |= KERNEL_AVG;
    }
    mask
}

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

/// 安全路径：`dst[i] = (a[i] + b[i] + 1) >> 1`。与 `v2v-mpeg12` 里
/// `predict_block` 的平均公式逐字节一致 —— 这是 oracle。
pub fn avg_rows_scalar(dst: &mut [u8], a: &[u8], b: &[u8]) {
    for i in 0..dst.len() {
        dst[i] = (((a[i] as u16) + (b[i] as u16) + 1) >> 1) as u8;
    }
}

/// 半像素平均。三个切片必须**等长**（不变量，调用方保证）。
pub fn avg_rows(dst: &mut [u8], a: &[u8], b: &[u8]) {
    assert_eq!(dst.len(), a.len(), "dst 与 a 不等长");
    assert_eq!(dst.len(), b.len(), "dst 与 b 不等长");
    if !avg_available() {
        avg_rows_scalar(dst, a, b);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        avg_rows_x86(dst, a, b)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        avg_rows_neon(dst, a, b)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        avg_rows_scalar(dst, a, b);
    }
}

#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "aarch64")]
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
}
