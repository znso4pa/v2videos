//! ISO/IEC 11172-3 §2.4.3.2 与 §2.4.3.4 的混合合成滤波器组。

use std::{f64::consts::PI, sync::OnceLock};
use v2v_tables::mp3::SYNTHESIS_WINDOW;

struct Matrices {
    long: [[[f64; 18]; 36]; 3],
    short: [[f64; 6]; 12],
    polyphase: [[f64; 32]; 64],
    alias: [(f64, f64); 8],
}

impl Matrices {
    fn get() -> &'static Self {
        static CACHE: OnceLock<Matrices> = OnceLock::new();
        CACHE.get_or_init(|| {
            let long = std::array::from_fn(|mode| {
                std::array::from_fn(|i| {
                    let sine = (PI / 36.0 * (i as f64 + 0.5)).sin();
                    let window = match (mode, i) {
                        (1, 18..=23) | (2, 12..=17) => 1.0,
                        (1, 24..=29) => (PI / 12.0 * (i as f64 - 17.5)).sin(),
                        (2, 6..=11) => (PI / 12.0 * (i as f64 - 5.5)).sin(),
                        (1, 30..) | (2, 0..=5) => 0.0,
                        _ => sine,
                    };
                    std::array::from_fn(|k| {
                        window * (PI / 72.0 * (2 * i + 19) as f64 * (2 * k + 1) as f64).cos()
                    })
                })
            });
            let short = std::array::from_fn(|i| {
                let window = (PI / 12.0 * (i as f64 + 0.5)).sin();
                std::array::from_fn(|k| {
                    window * (PI / 24.0 * (2 * i + 7) as f64 * (2 * k + 1) as f64).cos()
                })
            });
            let polyphase = std::array::from_fn(|i| {
                std::array::from_fn(|k| (PI / 64.0 * (i + 16) as f64 * (2 * k + 1) as f64).cos())
            });
            let alias = [
                -0.6_f64, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037,
            ]
            .map(|c| {
                let n = (1.0 + c * c).sqrt().recip();
                (n, c * n)
            });
            Self {
                long,
                short,
                polyphase,
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
            if block_type == 2 && sb >= long_subbands {
                for window in 0..3 {
                    for (i, row) in m.short.iter().enumerate() {
                        block[6 + window * 6 + i] += row
                            .iter()
                            .zip(input[window..].iter().step_by(3))
                            .map(|(a, b)| a * b)
                            .sum::<f64>();
                    }
                }
            } else {
                let mode = match block_type {
                    1 => 1,
                    3 => 2,
                    _ => 0,
                };
                for (y, row) in block.iter_mut().zip(&m.long[mode]) {
                    *y = row.iter().zip(input).map(|(a, b)| a * b).sum();
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
            for (y, row) in self.history[self.head..self.head + 64]
                .iter_mut()
                .zip(&m.polyphase)
            {
                *y = row.iter().zip(slot).map(|(a, b)| a * b).sum();
            }
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
