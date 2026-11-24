//! ISO/IEC 11172-3 §2.4.3.2 与 §2.4.3.4 的混合合成滤波器组。

use std::{f64::consts::PI, sync::OnceLock};
use v2v_tables::mp3::SYNTHESIS_WINDOW;

struct Matrices {
    long: [[f64; 18]; 18],
    short: [[f64; 6]; 6],
    long_windows: [[f64; 36]; 3],
    short_window: [f64; 12],
    dct_weights: [f64; 31],
    alias: [(f64, f64); 8],
}

impl Matrices {
    fn get() -> &'static Self {
        static CACHE: OnceLock<Matrices> = OnceLock::new();
        CACHE.get_or_init(|| {
            let long_windows = std::array::from_fn(|mode| {
                std::array::from_fn(|i| {
                    let sine = (PI / 36.0 * (i as f64 + 0.5)).sin();
                    match (mode, i) {
                        (1, 18..=23) | (2, 12..=17) => 1.0,
                        (1, 24..=29) => (PI / 12.0 * (i as f64 - 17.5)).sin(),
                        (2, 6..=11) => (PI / 12.0 * (i as f64 - 5.5)).sin(),
                        (1, 30..) | (2, 0..=5) => 0.0,
                        _ => sine,
                    }
                })
            });
            let short_window = std::array::from_fn(|i| (PI / 12.0 * (i as f64 + 0.5)).sin());
            let mut dct_weights = [0.0; 31];
            for half in [1, 2, 4, 8, 16] {
                for i in 0..half {
                    let angle = PI * (2 * i + 1) as f64 / (4 * half) as f64;
                    dct_weights[half - 1 + i] = (2.0 * angle.cos()).recip();
                }
            }
            let alias = [
                -0.6_f64, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037,
            ]
            .map(|c| {
                let n = (1.0 + c * c).sqrt().recip();
                (n, c * n)
            });
            Self {
                long: imdct_basis(),
                short: imdct_basis(),
                long_windows,
                short_window,
                dct_weights,
                alias,
            }
        })
    }
}

pub(crate) struct Synthesis {
    overlap: [[f64; 18]; 32],
    history: [f64; 1024],
    head: usize,
}

impl Synthesis {
    pub(crate) fn new() -> Self {
        Self {
            overlap: [[0.0; 18]; 32],
            history: [0.0; 1024],
            head: 0,
        }
    }

    pub(crate) fn granule(
        &mut self,
        mut lines: [f64; 576],
        block_type: usize,
        long_subbands: usize,
        out: &mut Vec<f32>,
    ) {
        let m = Matrices::get();
        for subband in 1..long_subbands {
            let boundary = subband * 18;
            for (j, &(cs, ca)) in m.alias.iter().enumerate() {
                let a = lines[boundary - 1 - j];
                let b = lines[boundary + j];
                lines[boundary - 1 - j] = a * cs - b * ca;
                lines[boundary + j] = b * cs + a * ca;
            }
        }

        let mut slots = [[0.0; 32]; 18];
        for (sb, input) in lines.chunks_exact(18).enumerate() {
            let mut block = [0.0; 36];
            if input.iter().any(|&x| x != 0.0) {
                if block_type == 2 && sb >= long_subbands {
                    for window in 0..3 {
                        let short = std::array::from_fn(|k| input[3 * k + window]);
                        let start = 6 + window * 6;
                        imdct(
                            &short,
                            &m.short,
                            &m.short_window,
                            &mut block[start..start + 12],
                        );
                    }
                } else {
                    let mode = match block_type {
                        1 => 1,
                        3 => 2,
                        _ => 0,
                    };
                    imdct(
                        input.try_into().unwrap(),
                        &m.long,
                        &m.long_windows[mode],
                        &mut block,
                    );
                }
            }
            for i in 0..18 {
                let value = block[i] + self.overlap[sb][i];
                slots[i][sb] = if i & sb & 1 == 0 { value } else { -value };
                self.overlap[sb][i] = block[i + 18];
            }
        }

        for slot in &slots {
            self.head = (self.head + 960) % 1024;
            synthesis_vector(
                slot,
                &m.dct_weights,
                &mut self.history[self.head..self.head + 64],
            );
            for sample in 0..32 {
                let mut sum = 0.0;
                for bank in 0..8 {
                    let base = self.head + 128 * bank + sample;
                    sum +=
                        self.history[base % 1024] * f64::from(SYNTHESIS_WINDOW[64 * bank + sample]);
                    sum += self.history[(base + 96) % 1024]
                        * f64::from(SYNTHESIS_WINDOW[64 * bank + 32 + sample]);
                }
                // D 的表存成 ×65536，输出 i16 刻度需要 ×32768。
                out.push((sum * 0.5) as f32);
            }
        }
    }
}

/// IMDCT 的前半段反对称、后半段对称。每行的内积供一对时间位置复用。
#[inline]
fn imdct_pair(n: usize, row: usize) -> (usize, usize, f64) {
    if row < n / 2 {
        (row, n - 1 - row, -1.0)
    } else {
        (row + n / 2, 5 * n / 2 - 1 - row, 1.0)
    }
}

fn imdct_basis<const N: usize>() -> [[f64; N]; N] {
    std::array::from_fn(|row| {
        let (i, _, _) = imdct_pair(N, row);
        std::array::from_fn(|k| {
            (PI / (4 * N) as f64 * (2 * i + N + 1) as f64 * (2 * k + 1) as f64).cos()
        })
    })
}

#[inline]
fn imdct<const N: usize>(input: &[f64; N], basis: &[[f64; N]; N], window: &[f64], out: &mut [f64]) {
    for (row, coefficients) in basis.iter().enumerate() {
        let (i, j, sign) = imdct_pair(N, row);
        let raw: f64 = coefficients.iter().zip(input).map(|(a, b)| a * b).sum();
        out[i] += raw * window[i];
        out[j] += (raw * sign) * window[j];
    }
}

/// ISO 合成矩阵的第 i 行是 DCT-II 的第 (i + 16) 项。DCT 的第 32 项为 0，
/// 其余超出 0..31 的项由余弦的反对称性恢复。
fn synthesis_vector(slot: &[f64; 32], weights: &[f64; 31], out: &mut [f64]) {
    let coefficients = dct_32(slot, weights);
    out[..16].copy_from_slice(&coefficients[16..]);
    out[16] = 0.0;
    for (dest, &v) in out[17..48].iter_mut().zip(coefficients[1..].iter().rev()) {
        *dest = -v;
    }
    for (dest, &v) in out[48..64].iter_mut().zip(&coefficients[..16]) {
        *dest = -v;
    }
}

/// 将 DCT-II 拆成各一半长度的偶数项和奇数项。固定长度让编译器
/// 展开小变换，避免每个时隙都递归调用并重算边界。
macro_rules! dct_step {
    ($name:ident, $smaller:ident, $n:literal, $half:literal) => {
        #[inline]
        fn $name(input: &[f64; $n], weights: &[f64; 31]) -> [f64; $n] {
            let sums = std::array::from_fn(|i| input[i] + input[$n - 1 - i]);
            let differences =
                std::array::from_fn(|i| (input[i] - input[$n - 1 - i]) * weights[$half - 1 + i]);
            let even = $smaller(&sums, weights);
            let odd = $smaller(&differences, weights);
            let mut out = [0.0; $n];
            for (i, pair) in out.chunks_exact_mut(2).enumerate() {
                pair[0] = even[i];
                pair[1] = odd[i] + odd.get(i + 1).copied().unwrap_or(0.0);
            }
            out
        }
    };
}

#[inline]
fn dct_2(input: &[f64; 2], weights: &[f64; 31]) -> [f64; 2] {
    [input[0] + input[1], (input[0] - input[1]) * weights[0]]
}

dct_step!(dct_4, dct_2, 4, 2);
dct_step!(dct_8, dct_4, 8, 4);
dct_step!(dct_16, dct_8, 16, 8);
dct_step!(dct_32, dct_16, 32, 16);

#[cfg(test)]
mod tests {
    use super::*;

    fn check_imdct<const N: usize>(basis: &[[f64; N]; N], window: &[f64]) {
        for case in 0..=N {
            let input: [f64; N] = std::array::from_fn(|i| {
                if case == N {
                    (i as f64 * 0.71).sin()
                } else if case == i {
                    1.0
                } else {
                    0.0
                }
            });
            let mut out = vec![0.0; 2 * N];
            imdct(&input, basis, window, &mut out);
            for (i, computed) in out.into_iter().enumerate() {
                let expected = input
                    .iter()
                    .enumerate()
                    .map(|(k, &v)| {
                        let angle = PI * ((2 * i + N + 1) * (2 * k + 1)) as f64 / (4 * N) as f64;
                        (window[i] * angle.cos()) * v
                    })
                    .sum::<f64>();
                assert!(
                    (computed - expected).abs() < 2e-12,
                    "n={N}, case={case}, row={i}"
                );
            }
        }
    }

    #[test]
    fn folded_imdct_matches_the_direct_long_and_short_matrices() {
        let m = Matrices::get();
        for window in &m.long_windows {
            check_imdct(&m.long, window);
        }
        check_imdct(&m.short, &m.short_window);
    }

    #[test]
    fn fast_synthesis_matches_the_direct_iso_matrix() {
        let weights = &Matrices::get().dct_weights;
        let mut state = 0xbad5ee_du32;
        for case in 0..128 {
            let slot = std::array::from_fn(|i| {
                if case < 32 {
                    return if i == case { 1.0 } else { 0.0 };
                }
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f64::from(state) * (2.0 / f64::from(u32::MAX)) - 1.0
            });
            let mut actual = [0.0; 64];
            synthesis_vector(&slot, weights, &mut actual);

            for (i, computed) in actual.into_iter().enumerate() {
                let expected = slot
                    .iter()
                    .enumerate()
                    .map(|(k, &v)| v * (PI * ((i + 16) * (2 * k + 1)) as f64 / 64.0).cos())
                    .sum::<f64>();
                assert!((computed - expected).abs() < 2e-12, "case={case}, row={i}");
            }
        }
    }
}
