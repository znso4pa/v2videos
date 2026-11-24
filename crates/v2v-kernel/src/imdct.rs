//! MDCT 的共享逆变换。

use std::f64::consts::PI;

use crate::fft_in_place;

/// IMDCT 计划。先求 `u[j] = Σ c[k]·cos(π/n·(j+0.5)(k+0.5))`，
/// 然后按 DCT-IV 的偶对称与反周期性展开 `u[i+n/2]`，输出乘以调用方指定的定标系数。
///
/// 把偶数输入 `c[2k]` 和倒序奇数输入 `c[n-1-2k]` 打包为复数，
/// 预乘 `exp(-iπk/n)` 后做 `n/2` 点 FFT。第 `r` 项再乘
/// `exp(-iπ(r+1/4)/n)`，其真实部分为 `u[2r]`，虚部取负为 `u[n-1-2r]`。
pub struct ImdctPlan {
    scale: f64,
    pre: Vec<(f64, f64)>,
    post: Vec<(f64, f64)>,
    re: Vec<f64>,
    im: Vec<f64>,
}

impl ImdctPlan {
    /// `n` 为谱系数数量，必须是至少为 2 的二次幂。
    pub fn new(n: usize, scale: f64) -> Self {
        assert!(n >= 2 && n.is_power_of_two());
        let step = PI / n as f64;
        Self {
            scale,
            pre: (0..n / 2).map(|k| (step * k as f64).sin_cos()).collect(),
            post: (0..n / 2)
                .map(|r| (step * (r as f64 + 0.25)).sin_cos())
                .collect(),
            re: vec![0.0; n / 2],
            im: vec![0.0; n / 2],
        }
    }

    /// 对 `n` 个谱系数执行逆变换，覆盖输出的全部 `2n` 个样本。
    pub fn apply(&mut self, coefs: &[f32], out: &mut [f32]) {
        let n = coefs.len();
        assert_eq!(out.len(), 2 * n);
        assert_eq!(self.pre.len() * 2, n);
        let re = &mut self.re;
        let im = &mut self.im;
        for (k, &(sin, cos)) in self.pre.iter().enumerate() {
            let a = f64::from(coefs[2 * k]);
            let b = f64::from(coefs[n - 1 - 2 * k]);
            re[k] = a * cos + b * sin;
            im[k] = b * cos - a * sin;
        }
        fft_in_place(re, im);
        for (r, &(sin, cos)) in self.post.iter().enumerate() {
            let even = (self.scale * (re[r] * cos + im[r] * sin)) as f32;
            let odd = (self.scale * (re[r] * sin - im[r] * cos)) as f32;
            put_imdct_sample(out, n, 2 * r, even);
            put_imdct_sample(out, n, n - 1 - 2 * r, odd);
        }
    }
}

/// 每个 DCT-IV 输出出现两次，由跨越 `n` 的半块取反。
#[inline]
fn put_imdct_sample(out: &mut [f32], n: usize, j: usize, value: f32) {
    out[3 * n / 2 - 1 - j] = -value;
    if j < n / 2 {
        out[3 * n / 2 + j] = -value;
    } else {
        out[j - n / 2] = value;
    }
}
