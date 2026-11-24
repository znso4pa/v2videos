use std::{
    f64::consts::{FRAC_1_SQRT_2, PI},
    sync::OnceLock,
};
use v2v_kernel::{BitReader, Vlc};
use v2v_tables::mp3::*;

use crate::{synth::Synthesis, unsupported, Error, Framer, Header, Pcm, Result};

/// MPEG-1、MPEG-2 和 MPEG-2.5 Layer III 的顺序解码器。
pub struct Decoder {
    framer: Framer,
    state: State,
}

impl Decoder {
    pub fn new(channels: usize, sample_rate: u32) -> Result<Self> {
        Ok(Self {
            framer: Framer::new(channels, sample_rate)?,
            state: State {
                reservoir: Vec::with_capacity(1024),
                main_data: Vec::new(),
                synthesis: std::array::from_fn(|_| Synthesis::new()),
            },
        })
    }

    /// 一次输出零到多个帧，每个帧的长度按 MPEG 版本确定。
    pub fn feed(&mut self, data: &[u8]) -> Result<Vec<Pcm>> {
        let mut out = Vec::new();
        self.framer.feed(data, |header, frame| {
            out.push(self.state.decode(header, frame)?);
            Ok(())
        })?;
        Ok(out)
    }

    pub fn finish(&self) -> Result<()> {
        self.framer.finish()
    }
}

struct State {
    reservoir: Vec<u8>,
    main_data: Vec<u8>,
    synthesis: [Synthesis; 2],
}

impl State {
    fn decode(&mut self, h: Header, frame: &[u8]) -> Result<Pcm> {
        let side_start = 4 + h.crc_len;
        let main_start = side_start + h.side_len;
        let mut side = BitReader::new(&frame[side_start..main_start]);
        let back = side.u(if h.mpeg1 { 9 } else { 8 })? as usize;
        let private = match (h.mpeg1, h.channels) {
            (true, 1) => 5,
            (true, _) => 3,
            (false, 1) => 1,
            (false, _) => 2,
        };
        side.skip(private)?;
        let mut scfsi = [0; 2];
        if h.mpeg1 {
            for mask in &mut scfsi[..h.channels] {
                *mask = side.u(4)?;
            }
        }

        // main_data_begin 从当前帧主数据起点往回数，历史只保留可寻址的 511 字节。
        let from = self
            .reservoir
            .len()
            .checked_sub(back)
            .ok_or(Error::Bad("MP3 reservoir underflow"))?;
        self.main_data.clear();
        self.main_data.extend_from_slice(&self.reservoir[from..]);
        self.main_data.extend_from_slice(&frame[main_start..]);
        self.reservoir.extend_from_slice(&frame[main_start..]);
        let keep_from = self.reservoir.len().saturating_sub(511);
        self.reservoir.drain(..keep_from);

        let granules = h.samples / 576;
        let mut main = BitReader::new(&self.main_data);
        let mut pcm: Pcm = (0..h.channels)
            .map(|_| Vec::with_capacity(h.samples))
            .collect();
        let mut scales = [[0; 39]; 2];
        for granule in 0..granules {
            let mut info = [Granule::default(); 2];
            let mut spectrum = [[0.0; 576]; 2];
            for ch in 0..h.channels {
                let mut g = Granule::read(&mut side, h)?;
                let end = main.bit_pos() + g.part_bits;
                if end > main.bit_pos() + main.bits_left() {
                    return Err(Error::Eof);
                }
                let mut part = Part {
                    bits: &mut main,
                    end,
                };
                let is_right = ch == 1 && h.stereo & 1 != 0;
                g.read_scales(
                    &mut part,
                    &mut scales[ch],
                    if granule == 0 { 0 } else { scfsi[ch] },
                    h,
                    is_right,
                )?;
                let ints = g.read_huffman(&mut part)?;
                part.bits.set_pos(end);
                g.requantize(h, &scales[ch], &ints, &mut spectrum[ch]);
                info[ch] = g;
            }
            if h.stereo != 0 {
                if info[0].block_type != info[1].block_type || info[0].mixed != info[1].mixed {
                    return Err(unsupported("joint stereo 的声道块类型不同"));
                }
                let [left, right] = &mut spectrum;
                info[1].stereo(h, &scales[1], left, right);
            }
            for ch in 0..h.channels {
                let g = &info[ch];
                let lines = g.reorder(h, &spectrum[ch]);
                let long_subbands = if g.block_type == 2 {
                    if g.mixed {
                        2
                    } else {
                        0
                    }
                } else {
                    32
                };
                self.synthesis[ch].granule(lines, g.block_type, long_subbands, &mut pcm[ch]);
            }
        }
        Ok(pcm)
    }
}

struct Part<'a, 'b> {
    bits: &'a mut BitReader<'b>,
    end: usize,
}

impl Part<'_, '_> {
    #[inline]
    fn read(&mut self, n: u32) -> Result<u32> {
        if self.bits.bit_pos() + n as usize > self.end {
            return Err(Error::Eof);
        }
        if n == 1 {
            self.bits.bit()
        } else {
            self.bits.u(n)
        }
    }

    fn vlc(&mut self, table: &Vlc) -> Result<usize> {
        let symbol = table.decode(self.bits)?;
        if self.bits.bit_pos() > self.end {
            return Err(Error::Eof);
        }
        Ok(symbol)
    }

    #[inline]
    fn signed(&mut self, value: usize, linbits: u8) -> Result<i16> {
        if value == 0 {
            return Ok(0);
        }
        let extra = if value == 15 && linbits != 0 {
            self.read(linbits.into())? as i16
        } else {
            0
        };
        let value = value as i16 + extra;
        Ok(if self.read(1)? == 0 { value } else { -value })
    }
}

#[derive(Clone, Copy)]
struct Granule {
    part_bits: usize,
    big_values: usize,
    gain: u32,
    compress: u32,
    block_type: usize,
    mixed: bool,
    tables: [usize; 3],
    regions: [usize; 2],
    subgain: [u32; 3],
    preflag: bool,
    sf_scale: u32,
    quad_table: usize,
    is_limit: [u8; 39],
}

impl Default for Granule {
    fn default() -> Self {
        Self {
            part_bits: 0,
            big_values: 0,
            gain: 0,
            compress: 0,
            block_type: 0,
            mixed: false,
            tables: [0; 3],
            regions: [0; 2],
            subgain: [0; 3],
            preflag: false,
            sf_scale: 0,
            quad_table: 0,
            is_limit: [0; 39],
        }
    }
}

impl Granule {
    fn read(r: &mut BitReader<'_>, h: Header) -> Result<Self> {
        let mut g = Self {
            part_bits: r.u(12)? as usize,
            big_values: r.u(9)? as usize,
            gain: r.u(8)?,
            compress: r.u(if h.mpeg1 { 4 } else { 9 })?,
            ..Self::default()
        };
        if g.big_values > 288 {
            return Err(Error::Bad("MP3 big_values"));
        }

        if r.bit()? != 0 {
            g.block_type = r.u(2)? as usize;
            g.mixed = r.bit()? != 0;
            if g.block_type == 0 || (g.mixed && g.block_type != 2) {
                return Err(Error::Bad("MP3 switched block type"));
            }
            if g.mixed && h.rate_index == 8 {
                return Err(unsupported("8 kHz mixed block"));
            }
            for table in &mut g.tables[..2] {
                *table = r.u(5)? as usize;
            }
            for gain in &mut g.subgain {
                *gain = r.u(3)?;
            }
            let split = match (g.block_type, h.rate_index) {
                (2, 8) => 72,
                (2, _) | (_, 0..=2) => 36,
                (_, 8) => 108,
                _ => 54,
            };
            g.regions = [split, 576];
        } else {
            for table in &mut g.tables {
                *table = r.u(5)? as usize;
            }
            let n = r.u(4)? as usize + 1;
            let m = r.u(3)? as usize + 1;
            let bands = &LONG_BANDS[h.rate_index];
            g.regions = [sum(&bands[..n]), sum(&bands[..(n + m).min(bands.len())])];
        }
        if h.mpeg1 {
            g.preflag = r.bit()? != 0;
        }
        g.sf_scale = r.bit()?;
        g.quad_table = r.bit()? as usize;
        Ok(g)
    }

    fn read_scales(
        &mut self,
        r: &mut Part<'_, '_>,
        sf: &mut [u8; 39],
        reuse: u32,
        h: Header,
        intensity: bool,
    ) -> Result<()> {
        if h.mpeg1 {
            let a = SLEN[0][self.compress as usize];
            let b = SLEN[1][self.compress as usize];
            if self.block_type == 2 {
                *sf = [0; 39];
                let first = if self.mixed { 17 } else { 18 };
                for (i, v) in sf[..first + 18].iter_mut().enumerate() {
                    *v = r.read(if i < first { a } else { b }.into())? as u8;
                }
            } else {
                let ranges = [(0, 6, a), (6, 11, a), (11, 16, b), (16, 21, b)];
                for (i, (start, end, n)) in ranges.into_iter().enumerate() {
                    if reuse & (8 >> i) == 0 {
                        for v in &mut sf[start..end] {
                            *v = r.read(n.into())? as u8;
                        }
                    }
                }
                sf[21..].fill(0);
            }
            self.is_limit.fill(7);
            return Ok(());
        }

        *sf = [0; 39];
        let c = self.compress;
        let (table, bits) = if intensity {
            match c / 2 {
                v @ 0..=179 => (3, [v / 36, v / 6 % 6, v % 6, 0]),
                180..=243 => {
                    let v = c / 2 - 180;
                    (4, [v / 16, v / 4 % 4, v % 4, 0])
                }
                _ => {
                    let v = c / 2 - 244;
                    (5, [v / 3, v % 3, 0, 0])
                }
            }
        } else {
            match c {
                0..=399 => (0, [c / 80, c / 16 % 5, c / 4 % 4, c % 4]),
                400..=499 => {
                    let v = c - 400;
                    (1, [v / 20, v / 4 % 5, v % 4, 0])
                }
                _ => {
                    let v = c - 500;
                    self.preflag = true;
                    (2, [v / 3, v % 3, 0, 0])
                }
            }
        };
        let layout = if self.block_type != 2 {
            0
        } else if self.mixed {
            2
        } else {
            1
        };
        let mut offset = 0;
        for (&count, n) in LSF_PARTITIONS[table * 3 + layout].iter().zip(bits) {
            for _ in 0..count {
                sf[offset] = r.read(n)? as u8;
                self.is_limit[offset] = ((1 << n) - 1) as u8;
                offset += 1;
            }
        }
        Ok(())
    }

    fn read_huffman(&self, r: &mut Part<'_, '_>) -> Result<[i16; 576]> {
        struct Books {
            pairs: [Vlc; 16],
            quads: [Vlc; 2],
        }
        static BOOKS: OnceLock<Books> = OnceLock::new();
        let books = BOOKS.get_or_init(|| Books {
            pairs: PAIRS.map(|v| Vlc::new(&v.iter().map(|&(c, n, _)| [c, n]).collect::<Vec<_>>())),
            quads: std::array::from_fn(|i| {
                Vlc::new(&std::array::from_fn::<_, 16, _>(|j| {
                    [QUAD_CODES[i][j].into(), QUAD_BITS[i][j].into()]
                }))
            }),
        });
        let mut out = [0; 576];
        let big_lines = 2 * self.big_values;
        // 区域边界以系数编号表示，按每对系数的起点选择码本。
        let [first, second] = self.regions.map(|end| (2 * end.div_ceil(2)).min(big_lines));
        for (table, range) in self
            .tables
            .into_iter()
            .zip([0..first, first..second, second..big_lines])
        {
            if table == 0 || range.is_empty() {
                continue;
            }
            let [book, linbits] = PAIR_MAP[table];
            if book == 0 {
                return Err(Error::Bad("MP3 reserved Huffman table"));
            }
            let book = book as usize;
            let vlc = &books.pairs[book];
            for pair in out[range].chunks_exact_mut(2) {
                let symbol = PAIRS[book][r.vlc(vlc)?].2;
                pair[0] = r.signed((symbol >> 4).into(), linbits)?;
                pair[1] = r.signed((symbol & 15).into(), linbits)?;
            }
        }

        for quad in out[big_lines..].chunks_exact_mut(4) {
            if r.bits.bit_pos() >= r.end {
                break;
            }
            // count1 结束没有独立标记，最后一个未读完的四元组是填充位。
            let Ok(code) = r.vlc(&books.quads[self.quad_table]) else {
                break;
            };
            let mut values = [0; 4];
            let mut complete = true;
            for (j, x) in values.iter_mut().enumerate() {
                match r.signed((code >> (3 - j)) & 1, 0) {
                    Ok(value) => *x = value,
                    Err(_) => {
                        complete = false;
                        break;
                    }
                }
            }
            if !complete {
                break;
            }
            quad.copy_from_slice(&values);
        }
        Ok(out)
    }

    fn bands(&self, h: Header) -> impl DoubleEndedIterator<Item = Band> {
        let (long, short) = match (self.block_type, self.mixed, h.mpeg1) {
            (2, true, true) => (8, 3),
            (2, true, false) => (6, 3),
            (2, false, _) => (0, 0),
            _ => (22, 13),
        };
        let long = (0..long).map(move |i| Band {
            index: i,
            len: LONG_BANDS[h.rate_index][i].into(),
            window: None,
        });
        let short = (short..13).flat_map(move |i| {
            (0..3).map(move |w| Band {
                index: i,
                len: SHORT_BANDS[h.rate_index][i].into(),
                window: Some(w),
            })
        });
        long.chain(short)
    }

    fn requantize(&self, h: Header, sf: &[u8; 39], ints: &[i16; 576], out: &mut [f64; 576]) {
        static POW: OnceLock<[f64; 8207]> = OnceLock::new();
        let pow = POW.get_or_init(|| std::array::from_fn(|i| (i as f64).powf(4.0 / 3.0)));
        // 全局增益与标度因子各 8 位，PRETAB 最大为 3，短窗增益 3 位。
        const MIN_EXP: i32 = -210 - 4 * (255 + 3) - 8 * 7;
        const MAX_EXP: i32 = 255 - 210;
        static GAINS: OnceLock<[f64; (MAX_EXP - MIN_EXP + 1) as usize]> = OnceLock::new();
        let gains = GAINS.get_or_init(|| {
            std::array::from_fn(|i| (f64::from(MIN_EXP + i as i32) * 0.25).exp2())
        });
        let mut start = 0;
        for (i, band) in self.bands(h).enumerate() {
            let end = start + band.len;
            let mut exponent = self.gain as i32 - 210;
            let pre = if self.preflag && band.window.is_none() {
                i32::from(PRETAB[1][band.index])
            } else {
                0
            };
            exponent -= (i32::from(sf[i]) + pre) * (2 * (1 + self.sf_scale as i32));
            if let Some(w) = band.window {
                exponent -= 8 * self.subgain[w] as i32;
            }
            let gain = gains[(exponent - MIN_EXP) as usize];
            for (y, &x) in out[start..end].iter_mut().zip(&ints[start..end]) {
                *y = gain * pow[x.unsigned_abs() as usize] * f64::from(x.signum());
            }
            start = end;
        }
    }

    fn stereo(&self, h: Header, sf: &[u8; 39], left: &mut [f64; 576], right: &mut [f64; 576]) {
        let intensity = h.stereo & 1 != 0;
        let ms = h.stereo & 2 != 0;
        let mut nonzero = [false; 4];
        let mut end = 576;
        let band_count = self.bands(h).count();
        for (i, band) in (0..band_count).rev().zip(self.bands(h).rev()) {
            let start = end - band.len;
            let l = &mut left[start..end];
            let r = &mut right[start..end];
            let lane = band.window.unwrap_or(3);
            nonzero[lane] |= r.iter().any(|&x| x != 0.0);
            if lane == 3 {
                nonzero[3] |= nonzero[..3].iter().any(|&x| x);
            }
            let sf_index = match (band.window, band.index) {
                (None, 21) => i - 1,
                (Some(_), 12) => i - 3,
                _ => i,
            };
            let position = sf[sf_index];
            if intensity && !nonzero[lane] && position < self.is_limit[sf_index] {
                let (a, b) = if h.mpeg1 {
                    let angle = f64::from(position) * PI / 12.0;
                    let (s, c) = angle.sin_cos();
                    (s / (s + c), c / (s + c))
                } else {
                    let gain = (-0.25
                        * f64::from(1 + (self.compress & 1))
                        * f64::from(position.div_ceil(2)))
                    .exp2();
                    if position & 1 == 0 {
                        (1.0, gain)
                    } else {
                        (gain, 1.0)
                    }
                };
                for (x, y) in l.iter_mut().zip(r) {
                    *y = *x * b;
                    *x *= a;
                }
            } else if ms {
                for (x, y) in l.iter_mut().zip(r) {
                    (*x, *y) = ((*x + *y) * FRAC_1_SQRT_2, (*x - *y) * FRAC_1_SQRT_2);
                }
            }
            end = start;
        }
    }

    fn reorder(&self, h: Header, spectrum: &[f64; 576]) -> [f64; 576] {
        if self.block_type != 2 {
            return *spectrum;
        }
        let mut out = [0.0; 576];
        let mut start = 0;
        for band in self.bands(h) {
            let end = start + band.len;
            if let Some(w) = band.window {
                let base = start - w * band.len;
                for (i, &x) in spectrum[start..end].iter().enumerate() {
                    out[base + 3 * i + w] = x;
                }
            } else {
                out[start..end].copy_from_slice(&spectrum[start..end]);
            }
            start = end;
        }
        out
    }
}

struct Band {
    index: usize,
    len: usize,
    window: Option<usize>,
}

fn sum(values: &[u8]) -> usize {
    values.iter().map(|&v| usize::from(v)).sum()
}
