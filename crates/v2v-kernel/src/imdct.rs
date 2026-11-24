//! 共享的 MDCT / IMDCT。

use std::f64::consts::PI;

use crate::fft_in_place;

/// DCT-IV：把偶数输入和倒序奇数输入打包成复数，预旋转后做 n/2 点 FFT。
/// 第 r 项后旋转的实部对应输出 2r，虚部取负对应输出 n-1-2r。
struct Rotations {
    pre: Vec<(f64, f64)>,
    post: Vec<(f64, f64)>,
}

impl Rotations {
    fn new(n: usize) -> Self {
        assert!(n >= 2 && n.is_power_of_two());
        let step = PI / n as f64;
        Self {
            pre: (0..n / 2).map(|k| (step * k as f64).sin_cos()).collect(),
            post: (0..n / 2)
                .map(|r| (step * (r as f64 + 0.25)).sin_cos())
                .collect(),
        }
    }

    fn transform(&self, input: impl Fn(usize) -> f64, re: &mut [f64], im: &mut [f64]) {
        let n = 2 * self.pre.len();
        assert_eq!(re.len(), n / 2);
        assert_eq!(im.len(), n / 2);
        for (k, &(sin, cos)) in self.pre.iter().enumerate() {
            let a = input(2 * k);
            let b = input(n - 1 - 2 * k);
            re[k] = a * cos + b * sin;
            im[k] = b * cos - a * sin;
        }
        fft_in_place(re, im);
    }

    fn pairs<'a>(&'a self, re: &'a [f64], im: &'a [f64]) -> impl Iterator<Item = (f64, f64)> + 'a {
        self.post
            .iter()
            .zip(re.iter().zip(im))
            .map(|(&(sin, cos), (&r, &i))| (r * cos + i * sin, r * sin - i * cos))
    }
}

/// 可复用的前向 MDCT 旋转表。定标由调用方指定。
pub struct MdctPlan {
    rotations: Rotations,
    scale: f64,
}

impl MdctPlan {
    /// `n` 为输出系数数量，必须是至少为 2 的二次幂。
    pub fn new(n: usize, scale: f64) -> Self {
        Self {
            rotations: Rotations::new(n),
            scale,
        }
    }

    /// 对 2n 个时域样本执行 MDCT。两个临时缓冲各为 n/2 点。
    pub fn apply(&self, input: &[f32], out: &mut [f32], re: &mut [f64], im: &mut [f64]) {
        let n = 2 * self.rotations.pre.len();
        assert_eq!(input.len(), 2 * n);
        assert_eq!(out.len(), n);
        // 相位偏移 n/2 后，DCT-IV 基函数按反射和反周期性折回 0..n。
        let fold = |j| {
            let reflected = f64::from(input[3 * n / 2 - 1 - j]);
            if j < n / 2 {
                -(reflected + f64::from(input[3 * n / 2 + j]))
            } else {
                f64::from(input[j - n / 2]) - reflected
            }
        };
        self.rotations.transform(fold, re, im);
        for (k, (even, odd)) in self.rotations.pairs(re, im).enumerate() {
            out[2 * k] = (self.scale * even) as f32;
            out[n - 1 - 2 * k] = (self.scale * odd) as f32;
        }
    }
}

/// IMDCT 先执行 DCT-IV，再按对称性展开，输出乘以调用方指定的定标系数。
pub struct ImdctPlan {
    scale: f64,
    rotations: Rotations,
    re: Vec<f64>,
    im: Vec<f64>,
}

impl ImdctPlan {
    /// `n` 为谱系数数量，必须是至少为 2 的二次幂。
    pub fn new(n: usize, scale: f64) -> Self {
        Self {
            scale,
            rotations: Rotations::new(n),
            re: vec![0.0; n / 2],
            im: vec![0.0; n / 2],
        }
    }

    /// 对 `n` 个谱系数执行逆变换，覆盖输出的全部 `2n` 个样本。
    pub fn apply(&mut self, coefs: &[f32], out: &mut [f32]) {
        let n = coefs.len();
        assert_eq!(out.len(), 2 * n);
        assert_eq!(self.rotations.pre.len() * 2, n);
        self.rotations
            .transform(|j| f64::from(coefs[j]), &mut self.re, &mut self.im);
        for (r, (even, odd)) in self.rotations.pairs(&self.re, &self.im).enumerate() {
            put_imdct_sample(out, n, 2 * r, (self.scale * even) as f32);
            put_imdct_sample(out, n, n - 1 - 2 * r, (self.scale * odd) as f32);
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
