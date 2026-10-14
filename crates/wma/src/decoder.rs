//! WMA v1/v2 帧解码 —— 逐段对着 `wma_decode_block` / `wma_decode_frame` / `wma_window`
//! 写。目标是**与 ffmpeg 的 PCM 逐样本对账**，所以算术顺序也尽量照着来。
//!
//! ## 一段一块（block），一块一帧（frame）
//!
//! 一帧 `frame_len` 个样本，由若干块拼成（`use_variable_block_len` 时块长可变，
//! 否则恒等于帧长）。每块：读头 → 解指数 → 解系数 → IMDCT → 加窗叠加进 `frame_out`。
//! 帧末把 `frame_out` 的前 `frame_len` 个样本吐出去，剩下的留作下一帧的重叠区。
//!
//! ## 几个「必须照抄」的地方
//!
//! - `mult = 10^(total_gain * 0.05) / max_exponent[ch] * mdct_norm`，
//!   `mdct_norm = 1 / (block_len / 2)`。
//! - 噪声编码时**高带不传系数**，只用噪声表按增益填；`noise_index` 是**跨块、跨帧**
//!   一直往前走的（不是每块重置）。
//! - LSP 指数用 `pow_m1_4` 那个位运算近似（**不是** `powf(x, -0.25)`）——
//!   它带查表插值，换成 `powf` 会有肉眼可见的偏差。

use crate::bitreader::BitReader;
use crate::params::WmaParams;
use crate::tables::*;
use crate::{Error, Result};

// FFT 走共享数值内核（AGENTS.md 不变量 3：不许复刻第二份）。
use v2v_msmpeg4::fft_in_place;

const NB_LSP_COEFS: usize = 10;
const LSP_POW_BITS: u32 = 7;
const NOISE_TAB_SIZE: usize = 8192;
const HIGH_BAND_MAX_SIZE: usize = 16;
const BLOCK_MIN_BITS: u32 = 7;
const BLOCK_MAX_SIZE: usize = 2048;

/// 一个 `(码长, 码字) → 符号` 的前缀码。
///
/// ## ⚠️ 这里曾经是「逐位读 + **线性扫全表**」
///
/// 旧版 `decode` 每一位都要 `entries.iter().find(..)` 扫一遍整张表：
///
/// ```text
///   for len in 1..=max_len {
///       code = (code << 1) | bit;
///       entries.iter().find(|(l, c, _)| l == len && c == code)   // ← 每次都扫全表
///   }
/// ```
///
/// 复杂度 `O(码长 × 项数)`，而 WMA 的系数表有 **666~1000+ 项** ——
/// `sample` 采样（整轨 208 秒）显示 `Vlc::decode` 一个人占了 WMA 解码的 **58%**
/// （349/605 个样本）。
///
/// 现在内部直接用 `v2v-msmpeg4` 的**二叉树** `Vlc`（`O(码长)`）——
/// 走树/建树的逻辑只有那一份，这里只做「表格式转换 + 位源适配」，
/// 不把树复刻第二遍（AGENTS.md 不变量 3）。
struct Vlc {
    inner: v2v_msmpeg4::Vlc,
}

impl Vlc {
    fn from_codes(codes: &[(u32, u8)]) -> Self {
        // 注：码字**不需要**按位反转（试过，反转后「VLC 里没有这个码」——MSB 优先是对的）。
        // 符号 = 数组下标（`Vlc::new` 的约定）。
        let table: Vec<[u32; 2]> = codes.iter().map(|&(c, l)| [c, u32::from(l)]).collect();
        Self {
            inner: v2v_msmpeg4::Vlc::new(&table),
        }
    }

    /// 由**码长**推规范 Huffman 码字 —— 严格照 ffmpeg 的 `ff_vlc_init_from_lengths`：
    ///
    /// ```c
    ///     j = code = 0;
    ///     for (i = 0; i < nb_codes; i++) {
    ///         len = *lens;
    ///         buf[j].code = code;          // 32 位定点：码字占高 len 位
    ///         code += 1U << (32 - len);
    ///     }
    /// ```
    ///
    /// ⚠️ **两个「不能想当然」的地方**（第一版两个都错了，位流直接错位）：
    /// 1. 是**按数组顺序**发码，**不是**「先按长度排序再发」——
    ///    `ff_wma_hgain_hufftab` 的长度列是无序的（10,10,10,10,9,9,8,8,…），
    ///    排序发码会给出完全不同的码字。
    /// 2. 累加器是 **32 位定点**（`1 << (32-len)`），不是「读一位加一位」的整数递增。
    ///
    /// ⚠️ 与 `from_codes` 不同：这里的**符号是表里的第一列**（`(sym, len)`），
    /// 不是数组下标 —— 所以要按 `sym` 铺进 `[[code, len]]` 那张表。
    fn from_lengths(lens: &[(i16, u8)]) -> Self {
        let max_sym = lens.iter().map(|&(s, _)| s.max(0) as usize).max().unwrap_or(0);
        let mut table = vec![[0u32; 2]; max_sym + 1];
        let mut code: u64 = 0;
        for &(sym, len) in lens {
            if len == 0 {
                continue;
            }
            let c = (code >> (32 - len as u32)) as u32;
            table[sym as usize] = [c, u32::from(len)];
            code += 1u64 << (32 - len as u32);
        }
        Self {
            inner: v2v_msmpeg4::Vlc::new(&table),
        }
    }

    /// 走树解一个符号（位流结束或未定义前缀都返回 `Bad`）。
    fn decode(&self, r: &mut BitReader) -> Result<u32> {
        self.inner
            .decode_bits(|| r.bit().ok().map(u32::from))
            .map(|sym| sym as u32)
            .ok_or_else(|| Error::Bad("VLC 里没有这个码"))
    }
}

/// 解码器。
pub struct WmaDecoder {
    params: WmaParams,

    /// 每种块长的指数频带划分（`exponent_bands[k]`）与个数。
    ///
    /// 目前解码路径只用到 `exponent_high_*`（高带），这两个是**完整的频带划分**，
    /// 留着给「指数 VLC 路径」和将来的编码器用 —— 别删。
    #[allow(dead_code)]
    exponent_bands: Vec<Vec<usize>>,
    #[allow(dead_code)]
    exponent_sizes: Vec<usize>,
    coefs_end: Vec<usize>,
    high_band_start: Vec<usize>,
    exponent_high_bands: Vec<Vec<usize>>,
    exponent_high_sizes: Vec<usize>,

    /// 每种块长的正弦窗。
    windows: Vec<Vec<f32>>,

    // LSP 用的表
    lsp_cos: Vec<f32>,
    lsp_pow_e: Vec<f32>,
    lsp_pow_m1: Vec<f32>,
    lsp_pow_m2: Vec<f32>,

    noise_table: Vec<f32>,
    noise_index: usize,

    /// 两套系数 VLC（立体声第二声道走另一套）。
    coef_vlc: [Vlc; 2],
    /// 指数 VLC（AAC scalefactor 表；v1 与 `use_exp_vlc=1` 用）。
    exp_vlc: Vlc,
    run_table: [Vec<u16>; 2],
    level_table: [Vec<f32>; 2],
    hgain_vlc: Vlc,

    // ---- 块/帧状态 ----
    reset_block_lengths: bool,
    prev_block_len_bits: u32,
    block_len_bits: u32,
    next_block_len_bits: u32,
    block_len: usize,
    block_pos: usize,

    ms_stereo: bool,
    channel_coded: [bool; 2],
    exponents: [Vec<f32>; 2],
    exponents_bsize: [u32; 2],
    exponents_initialized: [bool; 2],
    max_exponent: [f32; 2],
    coefs1: [Vec<f32>; 2],
    coefs: [Vec<f32>; 2],
    output: Vec<f32>,
    frame_out: [Vec<f32>; 2],

    /// IMDCT 的 FFT scratch（复用，避免每块都分配）。长度 `2 · BLOCK_MAX_SIZE`。
    imdct_re: Vec<f64>,
    imdct_im: Vec<f64>,

    // ---- 比特池（superframe 跨包搬运）----
    /// 上一包没用完、留给下一包接着解的那半帧的字节。
    last_superframe: Vec<u8>,
    /// `last_superframe` 开头要跳过的位数。
    last_bitoffset: usize,
    /// superframe 序号（4 位、16 回绕），用来校验包序。
    pub superframe_index: u32,
    /// 每声道上一块解出多少个系数（调试用）。
    pub last_coef_count: [usize; 2],
    /// 上一帧的 `total_gain`（调试用）。
    pub last_total_gain: i32,
}

impl WmaDecoder {
    pub fn new(params: WmaParams) -> Result<Self> {
        let nb = params.nb_block_sizes;
        let frame_len = params.frame_len;
        let frame_len_bits = params.frame_len_bits;
        let channels = params.channels;

        // ---- 指数频带（ff_wma_init 的那段）----
        let mut exponent_bands = vec![Vec::new(); nb];
        let mut exponent_sizes = vec![0usize; nb];
        let mut coefs_end = vec![0usize; nb];
        let mut high_band_start = vec![0usize; nb];
        let mut exponent_high_bands = vec![Vec::new(); nb];
        let mut exponent_high_sizes = vec![0usize; nb];

        for k in 0..nb {
            let block_len = frame_len >> k;

            // ---- 指数频带划分 ----
            // ⚠️ v1 与 v2 用的是**两条不同的公式**；v1 只有一条划分（写在 [0]）。
            // 这里只算「频带」这一件事，后面的 coefs_end / high_band 是两条路共用的
            // —— 第一版在 v1 分支里直接 `continue`，把 coefs_end 也跳过了，
            // 结果 `coefs_end - coefs_start` 下溢成一个天文数字（实测：18446744073709551613）。
            let bands: Vec<usize> = if params.version == 1 {
                let mut bands = Vec::new();
                let mut lpos = 0usize;
                for i in 0..25usize {
                    let a = CRITICAL_FREQS[i] as usize;
                    let b = params.sample_rate as usize;
                    let mut pos = ((block_len * 2 * a) + (b >> 1)) / b;
                    if pos > block_len {
                        pos = block_len;
                    }
                    bands.push(pos - lpos);
                    if pos >= block_len {
                        break;
                    }
                    lpos = pos;
                }
                bands
            } else {
                // v2 有「硬编码表」这条捷径：a = frame_len_bits - BLOCK_MIN_BITS - k < 3 时用表
                let a = frame_len_bits as i32 - BLOCK_MIN_BITS as i32 - k as i32;
                let table: Option<Vec<usize>> = if (0..3).contains(&a) {
                    exponent_band_table(params.sample_rate, a as usize)
                } else {
                    None
                };
                match table {
                    Some(t) => t,
                    None => {
                        let mut bands = Vec::new();
                        let mut lpos = 0usize;
                        for i in 0..25usize {
                            let a = CRITICAL_FREQS[i] as usize;
                            let b = params.sample_rate as usize;
                            let mut pos = ((block_len * 2 * a) + (b << 1)) / (4 * b);
                            pos <<= 2;
                            if pos > block_len {
                                pos = block_len;
                            }
                            if pos > lpos {
                                bands.push(pos - lpos);
                            }
                            if pos >= block_len {
                                break;
                            }
                            lpos = pos;
                        }
                        bands
                    }
                }
            };
            exponent_sizes[k] = bands.len();
            exponent_bands[k] = bands;

            coefs_end[k] = (frame_len - ((frame_len * 9) / 100)) >> k;
            high_band_start[k] = ((block_len * 2) as f32 * params.high_freq
                / params.sample_rate as f32
                + 0.5) as usize;

            // 高带：落在 [high_band_start, coefs_end) 里的频带
            let mut j = 0usize;
            let mut pos = 0usize;
            let mut hb = Vec::new();
            for i in 0..exponent_sizes[k] {
                let start = pos;
                pos += exponent_bands[k][i];
                let end = pos;
                let start = start.max(high_band_start[k]);
                let end = end.min(coefs_end[k]);
                if end > start {
                    hb.push(end - start);
                    j += 1;
                }
            }
            exponent_high_sizes[k] = j;
            exponent_high_bands[k] = hb;
        }

        // ---- 正弦窗（每种块长一份）----
        let mut windows = Vec::with_capacity(nb);
        for k in 0..nb {
            let bl = frame_len >> k;
            windows.push(
                (0..bl)
                    .map(|i| (std::f32::consts::PI * (i as f32 + 0.5) / (2.0 * bl as f32)).sin())
                    .collect(),
            );
        }

        // ---- LSP 表 ----
        let wdel = std::f32::consts::PI / frame_len as f32;
        let lsp_cos: Vec<f32> = (0..frame_len)
            .map(|i| 2.0 * (wdel * i as f32).cos())
            .collect();
        let lsp_pow_e: Vec<f32> = (0..256)
            .map(|i| ((i as i32 - 126) as f32 * -0.25).exp2())
            .collect();
        let mut lsp_pow_m1 = vec![0f32; 1 << LSP_POW_BITS];
        let mut lsp_pow_m2 = vec![0f32; 1 << LSP_POW_BITS];
        {
            let mut b = 1.0f32;
            for i in (0..(1usize << LSP_POW_BITS)).rev() {
                let m = (1usize << LSP_POW_BITS) + i;
                let a = (m as f32) * (0.5 / (1 << LSP_POW_BITS) as f32);
                let a = 1.0 / a.sqrt().sqrt();
                lsp_pow_m1[i] = 2.0 * a - b;
                lsp_pow_m2[i] = b - a;
                b = a;
            }
        }

        // ---- 噪声表 ----
        let noise_mult = if params.use_exp_vlc { 0.02f32 } else { 0.04f32 };
        let mut noise_table = vec![0f32; NOISE_TAB_SIZE];
        {
            let mut seed: u32 = 1;
            let norm = (1.0f32 / (1u64 << 31) as f32) * 3f32.sqrt() * noise_mult;
            for v in noise_table.iter_mut() {
                seed = seed.wrapping_mul(314159).wrapping_add(1);
                *v = (seed as i32) as f32 * norm;
            }
        }

        // ---- 系数 VLC：v2 且采样率 < 32000 → 用第 2 组（coef4/coef5）----
        let bps1 = {
            let bps = params.bit_rate as f32 / (channels as f32 * params.sample_rate as f32);
            if channels == 2 {
                bps * 1.6
            } else {
                bps
            }
        };
        let mut coef_vlc_table = 2usize;
        if params.sample_rate >= 32000 {
            if bps1 < 0.72 {
                coef_vlc_table = 0;
            } else if bps1 < 1.16 {
                coef_vlc_table = 1;
            }
        }
        let t0 = coef_vlc_table * 2;
        let (vlc0, run0, lev0) = build_coef_vlc(t0);
        let (vlc1, run1, lev1) = build_coef_vlc(t0 + 1);

        let hgain_vlc = Vlc::from_lengths(&HGAIN_TAB);
        let exp_vlc = Vlc::from_codes(&EXP_VLC);

        Ok(Self {
            params,
            exponent_bands,
            exponent_sizes,
            coefs_end,
            high_band_start,
            exponent_high_bands,
            exponent_high_sizes,
            windows,
            lsp_cos,
            lsp_pow_e,
            lsp_pow_m1,
            lsp_pow_m2,
            noise_table,
            noise_index: 0,
            coef_vlc: [vlc0, vlc1],
            exp_vlc,
            run_table: [run0, run1],
            level_table: [lev0, lev1],
            hgain_vlc,
            reset_block_lengths: true,
            prev_block_len_bits: frame_len_bits,
            block_len_bits: frame_len_bits,
            next_block_len_bits: frame_len_bits,
            block_len: frame_len,
            block_pos: 0,
            ms_stereo: false,
            channel_coded: [false; 2],
            exponents: [vec![0.0; BLOCK_MAX_SIZE], vec![0.0; BLOCK_MAX_SIZE]],
            exponents_bsize: [0; 2],
            exponents_initialized: [false; 2],
            max_exponent: [1.0; 2],
            coefs1: [vec![0.0; BLOCK_MAX_SIZE], vec![0.0; BLOCK_MAX_SIZE]],
            coefs: [vec![0.0; BLOCK_MAX_SIZE], vec![0.0; BLOCK_MAX_SIZE]],
            output: vec![0.0; BLOCK_MAX_SIZE * 2],
            frame_out: [vec![0.0; BLOCK_MAX_SIZE * 2], vec![0.0; BLOCK_MAX_SIZE * 2]],
            imdct_re: vec![0.0; BLOCK_MAX_SIZE * 2],
            imdct_im: vec![0.0; BLOCK_MAX_SIZE * 2],
            last_superframe: Vec::new(),
            last_bitoffset: 0,
            superframe_index: 0,
            last_coef_count: [0; 2],
            last_total_gain: 0,
        })
    }

    /// 解**一个 ASF 音频载荷**，返回它解出的若干帧。
    ///
    /// 开了比特池时，一个载荷 = 一个 superframe（可能含多帧，靠残留半帧跨包接续）；
    /// 没开时，一个载荷 = `len / block_align` 个**各自独立**的帧。
    ///
    /// 两条路都逐行对着 `wma_decode_superframe`：比特池的搬运、残留半帧的接续、
    /// `reset_block_lengths` 的时机，一个都不能少。
    pub fn decode_superframe(&mut self, payload: &[u8]) -> Result<Vec<Vec<Vec<f32>>>> {
        let p = self.params.clone();
        let mut frames: Vec<Vec<Vec<f32>>> = Vec::new();

        // 不用比特池：**每个 `block_align` 字节块解一帧**，不是整包一帧。
        //
        // ⚠️ 这是踩了很久的坑。`wma_decode_superframe` 里有这么两句：
        //
        // ```c
        // if (avctx->block_align) buf_size = avctx->block_align;   // 截到 block_align
        // ...
        // return buf_size;                                         // 返回已消费长度
        // ```
        //
        // 而通用解码层（`decode.c` 的 `decode_simple_internal`）认这个返回值：
        //
        // ```c
        // pkt->data += consumed;   // 剩下的字节留在 avci->in_pkt 里
        // pkt->size -= consumed;
        // ```
        //
        // 于是同一个 ASF 载荷会被**连着喂进解码器若干次**，每次 `block_align` 字节、
        // 各出一帧 —— 一个载荷出「载荷长度 ÷ `block_align`」帧。
        //
        // 曾经把载荷首字节读成「superframe 序号 14、帧数 3」，纯属巧合的误判
        // —— 比特池根本没开，那个字节就是普通帧数据的开头。判定实验：把同一批
        // 字节灌进 WAV 用两种 `block_align` 解，只有「截到 block_align」那种
        // 与 ASF 出来的 PCM **MD5 相同**。
        if !p.use_bit_reservoir {
            let ba = p.block_align;
            let mut off = 0;
            while payload.len() - off >= ba {
                let mut r = BitReader::new(&payload[off..off + ba]);
                frames.push(self.decode_frame(&mut r)?);
                off += ba;
            }
            // 末尾不足一块的余数：ffmpeg 会打 "Input packet size too small" 并丢掉。
            return Ok(frames);
        }

        let mut r = BitReader::new(payload);
        let index = r.get(4)?;
        let raw = r.get(4)?;
        self.superframe_index = index;
        // 上一包没有残留时，本包的总产出要减 1（那一帧本来该由上一包留下）
        let mut nb_frames = raw as i32 - i32::from(self.last_superframe.is_empty());
        if nb_frames <= 0 {
            // 攒位：把 `buf_size - 1` 字节接进残留缓冲（ffmpeg 就是从第 1 个字节起接）
            self.last_superframe.extend_from_slice(&payload[1..]);
            return Ok(frames);
        }

        let bit_offset = r.get(p.byte_offset_bits + 3)? as usize;
        let start = bit_offset + 4 + 4 + p.byte_offset_bits as usize + 3;
        if start > payload.len() * 8 {
            return Err(Error::Bad("bit_offset 超出包长"));
        }

        // 有残留：把本包开头 `bit_offset` 位接到残留后面，解出那一帧。
        //
        // ⚠️ 这 `bit_offset` 位是**从头之后接着读**的（ffmpeg 就在头解析完的位置上
        // `get_bits` 继续），**不是**从包的第 0 位读！第一版从第 0 位读，
        // 结果「残留帧」全错 —— 而帧号看着还对（帧数没少），只是那些帧的内容是错的，
        // 表现为「有的帧几乎正确、有的帧几乎静音」交替出现。
        if !self.last_superframe.is_empty() {
            let mut acc = std::mem::take(&mut self.last_superframe);
            let mut bitpos = acc.len() * 8;
            for _ in 0..bit_offset {
                let bit = r.bit()?;
                if bitpos % 8 == 0 {
                    acc.push(0);
                }
                let last = acc.len() - 1;
                acc[last] |= (bit as u8) << (7 - (bitpos % 8));
                bitpos += 1;
            }
            let mut rr = BitReader::new(&acc);
            rr.skip(self.last_bitoffset)?;
            frames.push(self.decode_frame(&mut rr)?);
            nb_frames -= 1;
        }

        // 本包新起的帧：从 `start` 位开始
        let mut rr = BitReader::new(&payload[start >> 3..]);
        rr.skip(start & 7)?;
        // ⚠️ `reset_block_lengths` 是**每个 superframe 重置一次**（ffmpeg 在
        // 解新帧之前置 1）—— 只在构造时置一次会让第二帧起少读一个字段，位流全错。
        self.reset_block_lengths = true;
        for _ in 0..nb_frames {
            frames.push(self.decode_frame(&mut rr)?);
        }

        // 存残留：这一包解完之后剩下的位
        let consumed = rr.bit_pos() + ((bit_offset + 4 + 4 + p.byte_offset_bits as usize + 3) & !7);
        self.last_bitoffset = consumed & 7;
        let byte = consumed >> 3;
        self.last_superframe = if byte <= payload.len() {
            payload[byte..].to_vec()
        } else {
            Vec::new()
        };
        Ok(frames)
    }

    /// 解一帧：返回**每个声道** `frame_len` 个样本。
    pub fn decode_frame(&mut self, r: &mut BitReader) -> Result<Vec<Vec<f32>>> {
        self.block_pos = 0;
        loop {
            let last = self.decode_block(r)?;
            if last {
                break;
            }
        }
        let fl = self.params.frame_len;
        let mut out = Vec::with_capacity(self.params.channels);
        for ch in 0..self.params.channels {
            out.push(self.frame_out[ch][..fl].to_vec());
            // 后半段留作下一帧的重叠区
            self.frame_out[ch].copy_within(fl..fl * 2, 0);
            for v in &mut self.frame_out[ch][fl..fl * 2] {
                *v = 0.0;
            }
        }
        Ok(out)
    }

    /// 解一块。返回 `true` = 这一块是帧的最后一块。
    fn decode_block(&mut self, r: &mut BitReader) -> Result<bool> {
        let channels = self.params.channels;
        let frame_len_bits = self.params.frame_len_bits;

        // ---- 块长（use_variable_block_len）----
        if self.params.use_variable_block_len {
            let n = crate::av_log2(self.params.nb_block_sizes as u32 - 1) + 1;
            if self.reset_block_lengths {
                self.reset_block_lengths = false;
                let v = r.get(n)?;
                if v >= self.params.nb_block_sizes as u32 {
                    return Err(Error::Bad("prev_block_len_bits 超范围"));
                }
                self.prev_block_len_bits = frame_len_bits - v;
                let v = r.get(n)?;
                if v >= self.params.nb_block_sizes as u32 {
                    return Err(Error::Bad("block_len_bits 超范围"));
                }
                self.block_len_bits = frame_len_bits - v;
            } else {
                self.prev_block_len_bits = self.block_len_bits;
                self.block_len_bits = self.next_block_len_bits;
            }
            let v = r.get(n)?;
            if v >= self.params.nb_block_sizes as u32 {
                return Err(Error::Bad("next_block_len_bits 超范围"));
            }
            self.next_block_len_bits = frame_len_bits - v;
        } else {
            self.next_block_len_bits = frame_len_bits;
            self.prev_block_len_bits = frame_len_bits;
            self.block_len_bits = frame_len_bits;
        }
        if frame_len_bits - self.block_len_bits >= self.params.nb_block_sizes as u32 {
            return Err(Error::Bad("block_len_bits 未初始化"));
        }
        self.block_len = 1usize << self.block_len_bits;
        if self.block_pos + self.block_len > self.params.frame_len {
            return Err(Error::Bad("frame_len 溢出"));
        }

        if channels == 2 {
            self.ms_stereo = r.bit()? != 0;
        }
        let mut any_coded = false;
        for ch in 0..channels {
            let a = r.bit()? != 0;
            self.channel_coded[ch] = a;
            any_coded |= a;
        }
        let bsize = (frame_len_bits - self.block_len_bits) as usize;

        // 一个声道都没编 → 直接进 MDCT/加窗（全零）
        if !any_coded {
            return self.finish_block(bsize, 0, [0; 2], false);
        }

        // ---- total_gain → coef_nb_bits ----
        let mut total_gain = 1i32;
        loop {
            if r.bits_left() < 7 {
                return Err(Error::Bad("total_gain 读越界"));
            }
            let a = r.get(7)? as i32;
            total_gain += a;
            if a != 127 {
                break;
            }
        }
        let coef_nb_bits = total_gain_to_bits(total_gain);
        self.last_total_gain = total_gain;

        let n = self.coefs_end[bsize] - self.params.coefs_start;
        let mut nb_coefs = [n; 2];

        // ---- 噪声编码：高带编码标志 + 增益 ----
        let mut high_band_coded = [[false; HIGH_BAND_MAX_SIZE]; 2];
        let mut high_band_values = [[0i32; HIGH_BAND_MAX_SIZE]; 2];
        if self.params.use_noise_coding {
            for ch in 0..channels {
                if self.channel_coded[ch] {
                    let n = self.exponent_high_sizes[bsize];
                    for i in 0..n {
                        let a = r.bit()? != 0;
                        high_band_coded[ch][i] = a;
                        if a {
                            nb_coefs[ch] -= self.exponent_high_bands[bsize][i];
                        }
                    }
                }
            }
            for ch in 0..channels {
                if self.channel_coded[ch] {
                    let n = self.exponent_high_sizes[bsize];
                    let mut val = i32::MIN;
                    for i in 0..n {
                        if high_band_coded[ch][i] {
                            if val == i32::MIN {
                                val = r.get(7)? as i32 - 19;
                            } else {
                                val += self.hgain_vlc.decode(r)? as i32 - 18;
                            }
                            high_band_values[ch][i] = val;
                        }
                    }
                }
            }
        }

        // ---- 指数 ----
        if self.block_len_bits == frame_len_bits || r.bit()? != 0 {
            for ch in 0..channels {
                if self.channel_coded[ch] {
                    if self.params.use_exp_vlc {
                        self.decode_exp_vlc(r, ch, bsize)?;
                    } else {
                        self.decode_exp_lsp(r, ch)?;
                    }
                    self.exponents_bsize[ch] = bsize as u32;
                    self.exponents_initialized[ch] = true;
                }
            }
        }
        for ch in 0..channels {
            if self.channel_coded[ch] && !self.exponents_initialized[ch] {
                return Err(Error::Bad("指数没初始化"));
            }
        }
        // ---- 系数（run-level）----
        for ch in 0..channels {
            if self.channel_coded[ch] {
                let tindex = usize::from(ch == 1 && self.ms_stereo);
                for v in self.coefs1[ch][..self.block_len].iter_mut() {
                    *v = 0.0;
                }
                self.run_level_decode(r, tindex, ch, 0, nb_coefs[ch], coef_nb_bits)?;
            }
            // ⚠️ v1 立体声：**每个声道**解完系数都要按字节对齐（wmadec.c 626 行，
            // 对齐在声道循环**里面**）。第一版放在循环外只对齐了一次，
            // 第二个声道的位流就错位了（实测：跑到第 427 个载荷才 `run-level 越界`，
            // 因为静音帧吃位少，错位要攒到内容帧才显形）。
            if self.params.version == 1 && channels >= 2 {
                let pad = (8 - (r.bit_pos() & 7)) & 7;
                if pad > 0 {
                    r.skip(pad)?;
                }
            }
        }

        // ---- 归一化系数 ----
        let n4 = (self.block_len / 2) as f32;
        let mut mdct_norm = 1.0f32 / n4;
        if self.params.version == 1 {
            mdct_norm *= n4.sqrt(); // v1 的归一化不一样（wmadec.c 634 行）
        }
        for ch in 0..channels {
            if !self.channel_coded[ch] {
                continue;
            }
            let mult = 10f32.powf(total_gain as f32 * 0.05) / self.max_exponent[ch] * mdct_norm;
            let esize = self.exponents_bsize[ch] as usize;
            let coefs_start = self.params.coefs_start;
            let coefs_end = self.coefs_end[bsize];
            let mut out = vec![0f32; self.block_len];
            let mut w = 0usize;
            if self.params.use_noise_coding {
                let mult1 = mult;
                // 极低频：噪声
                for i in 0..coefs_start {
                    out[w] = self.noise_table[self.noise_index]
                        * self.exponents[ch][(i << bsize) >> esize]
                        * mult1;
                    self.noise_index = (self.noise_index + 1) & (NOISE_TAB_SIZE - 1);
                    w += 1;
                }
                let n1 = self.exponent_high_sizes[bsize];
                // 高带功率
                let mut exp_power = [0f32; HIGH_BAND_MAX_SIZE];
                let mut last_high_band = 0usize;
                {
                    let mut idx = (self.high_band_start[bsize] << bsize) >> esize;
                    for j in 0..n1 {
                        let n = self.exponent_high_bands[bsize][j];
                        if high_band_coded[ch][j] {
                            let mut e2 = 0f32;
                            for i in 0..n {
                                let v = self.exponents[ch][idx + ((i << bsize) >> esize)];
                                e2 += v * v;
                            }
                            exp_power[j] = e2 / n as f32;
                            last_high_band = j;
                        }
                        idx += (n << bsize) >> esize;
                    }
                }
                // 主频带 + 高带
                let mut idx = (coefs_start << bsize) >> esize;
                let mut j = -1i32;
                while j < n1 as i32 {
                    let n = if j < 0 {
                        self.high_band_start[bsize] - coefs_start
                    } else {
                        self.exponent_high_bands[bsize][j as usize]
                    };
                    if j >= 0 && high_band_coded[ch][j as usize] {
                        let mut m1 = (exp_power[j as usize] / exp_power[last_high_band]).sqrt();
                        m1 *= 10f32.powf(high_band_values[ch][j as usize] as f32 * 0.05);
                        m1 /= self.max_exponent[ch] * 0.04;
                        m1 *= mdct_norm;
                        for i in 0..n {
                            let noise = self.noise_table[self.noise_index];
                            self.noise_index = (self.noise_index + 1) & (NOISE_TAB_SIZE - 1);
                            out[w] = noise * self.exponents[ch][idx + ((i << bsize) >> esize)] * m1;
                            w += 1;
                        }
                        idx += (n << bsize) >> esize;
                    } else {
                        for i in 0..n {
                            let noise = self.noise_table[self.noise_index];
                            self.noise_index = (self.noise_index + 1) & (NOISE_TAB_SIZE - 1);
                            out[w] = (self.coefs1[ch][w] + noise)
                                * self.exponents[ch][idx + ((i << bsize) >> esize)]
                                * mult;
                            w += 1;
                        }
                        idx += (n << bsize) >> esize;
                    }
                    j += 1;
                }
                // 极高频：噪声
                //
                // ⚠️ ffmpeg 这里是 `exponents[(-(1 << bsize)) >> esize]` —— **负索引**
                // （指针已经走到 coefs_end 附近，负偏移是回退一格取最后一个指数）。
                // Rust 里绝不能把负数强转成 usize：那是个天文数字，要么 panic 要么读错。
                let n = self.block_len - coefs_end;
                let base = (coefs_end << bsize) >> esize;
                let back = ((-(1i64 << bsize)) >> esize) as isize;
                let idx = (base as isize + back) as usize;
                let m1 = mult * self.exponents[ch][idx];
                for _ in 0..n {
                    out[w] = self.noise_table[self.noise_index] * m1;
                    self.noise_index = (self.noise_index + 1) & (NOISE_TAB_SIZE - 1);
                    w += 1;
                }
            } else {
                for _i in 0..coefs_start {
                    out[w] = 0.0;
                    w += 1;
                }
                for i in 0..nb_coefs[ch] {
                    out[w] = self.coefs1[ch][i] * self.exponents[ch][(i << bsize) >> esize] * mult;
                    w += 1;
                }
                let n = self.block_len - coefs_end;
                for _ in 0..n {
                    out[w] = 0.0;
                    w += 1;
                }
            }
            self.coefs[ch][..self.block_len].copy_from_slice(&out);
        }

        // ---- MS 立体声（在 MDCT 之前）----
        if self.ms_stereo && self.channel_coded[1] {
            if !self.channel_coded[0] {
                for v in self.coefs[0][..self.block_len].iter_mut() {
                    *v = 0.0;
                }
                self.channel_coded[0] = true;
            }
            for i in 0..self.block_len {
                let a = self.coefs[0][i];
                let b = self.coefs[1][i];
                self.coefs[0][i] = a + b;
                self.coefs[1][i] = a - b;
            }
        }

        self.finish_block(bsize, total_gain, nb_coefs, true)
    }

    /// IMDCT + 加窗 + 叠加进 `frame_out`；返回「是不是最后一块」。
    fn finish_block(
        &mut self,
        bsize: usize,
        _total_gain: i32,
        _nb_coefs: [usize; 2],
        _coded: bool,
    ) -> Result<bool> {
        let channels = self.params.channels;
        let frame_len = self.params.frame_len;
        let n4 = self.block_len / 2;

        for ch in 0..channels {
            if self.channel_coded[ch] {
                self.imdct(ch);
            } else if !(self.ms_stereo && ch == 1) {
                for v in self.output.iter_mut() {
                    *v = 0.0;
                }
            }
            let index = frame_len / 2 + self.block_pos - n4;
            self.window_add(ch, index, bsize);
        }

        self.block_pos += self.block_len;
        Ok(self.block_pos >= frame_len)
    }

    /// IMDCT：`block_len` 个系数 → `2 * block_len` 个样本（写进 `output`）。
    ///
    /// ⚠️ **整体取负**。教科书 IMDCT 用 `+cos(...)`，但 ffmpeg 的 MDCT 约定是它的
    /// 相反数 —— 实测：我们的输出与 ffmpeg 的 PCM 互相关正好是 **−1.0000**
    /// （完美反相、幅度与波形全对）。所以这里取负号，不是「调参」。
    ///
    /// 推导与代价见 `imdct_kernel`；这里只负责把解码器的缓冲递过去。
    fn imdct(&mut self, ch: usize) {
        let n = self.block_len;
        let m = 2 * n;
        imdct_kernel(
            &self.coefs[ch][..n],
            &mut self.output[..m],
            &mut self.imdct_re[..m],
            &mut self.imdct_im[..m],
        );
    }

    /// 加窗并把这一块叠加到 `frame_out[ch][index..]`。
    fn window_add(&mut self, ch: usize, index: usize, bsize: usize) {
        let bl = self.block_len;
        let mut out_i = index;
        let mut in_i = 0usize;

        // 左半
        if self.block_len_bits <= self.prev_block_len_bits {
            let win = &self.windows[bsize];
            for i in 0..bl {
                self.frame_out[ch][out_i + i] += self.output[in_i + i] * win[i];
            }
        } else {
            let blk = 1usize << self.prev_block_len_bits;
            let n = (bl - blk) / 2;
            let b2 = (self.params.frame_len_bits - self.prev_block_len_bits) as usize;
            let win = &self.windows[b2];
            for i in 0..blk {
                self.frame_out[ch][out_i + n + i] += self.output[in_i + n + i] * win[i];
            }
            for i in 0..n {
                self.frame_out[ch][out_i + n + blk + i] = self.output[in_i + n + blk + i];
            }
        }
        out_i += bl;
        in_i += bl;

        // 右半（窗反向）
        if self.block_len_bits <= self.next_block_len_bits {
            let win = &self.windows[bsize];
            for i in 0..bl {
                self.frame_out[ch][out_i + i] = self.output[in_i + i] * win[bl - 1 - i];
            }
        } else {
            let blk = 1usize << self.next_block_len_bits;
            let n = (bl - blk) / 2;
            let b2 = (self.params.frame_len_bits - self.next_block_len_bits) as usize;
            let win = &self.windows[b2];
            for i in 0..n {
                self.frame_out[ch][out_i + i] = self.output[in_i + i];
            }
            for i in 0..blk {
                self.frame_out[ch][out_i + n + i] = self.output[in_i + n + i] * win[blk - 1 - i];
            }
            for i in 0..n {
                self.frame_out[ch][out_i + n + blk + i] = 0.0;
            }
        }
    }

    /// **指数 VLC** 解码（`decode_exp_vlc`）—— WMA v1（以及 `use_exp_vlc = 1` 的 v2）走这条。
    ///
    /// 码表就是 **AAC 的 scalefactor 表**（ffmpeg 直接借它）。指数按频带填，
    /// 每个频带一个指数；`last_exp` 是**差分**累加的，初值 v1 从码流读 5 位 + 10。
    fn decode_exp_vlc(&mut self, r: &mut BitReader, ch: usize, bsize: usize) -> Result<()> {
        let mut last_exp: i32;
        let mut max_scale = 0f32;
        let mut pos = 0usize;
        let mut band = 0usize;
        let bands = self.exponent_bands[bsize].clone();
        let block_len = self.block_len;

        if self.params.version == 1 {
            last_exp = r.get(5)? as i32 + 10;
            let v = pow_tab(last_exp);
            max_scale = v;
            let n = bands[band];
            band += 1;
            for i in 0..n {
                self.exponents[ch][pos + i] = v;
            }
            pos += n;
        } else {
            last_exp = 36;
        }

        while pos < block_len {
            if band >= bands.len() {
                return Err(Error::Bad("指数频带用完了"));
            }
            let code = self.exp_vlc.decode(r)? as i32;
            last_exp += code - 60;
            if !(0..156).contains(&last_exp) {
                return Err(Error::Bad("指数超出 pow_tab 范围"));
            }
            let v = pow_tab(last_exp);
            if v > max_scale {
                max_scale = v;
            }
            let n = bands[band];
            band += 1;
            for i in 0..n {
                self.exponents[ch][pos + i] = v;
            }
            pos += n;
        }
        self.max_exponent[ch] = max_scale;
        Ok(())
    }

    /// LSP 指数解码（`decode_exp_lsp` + `wma_lsp_to_curve`）。
    fn decode_exp_lsp(&mut self, r: &mut BitReader, ch: usize) -> Result<()> {
        let mut lsp = [0f32; NB_LSP_COEFS];
        for i in 0..NB_LSP_COEFS {
            let bits = if i == 0 || i >= 8 { 3 } else { 4 };
            let val = r.get(bits)? as usize;
            lsp[i] = LSP_CODEBOOK[i][val];
        }
        let n = self.block_len;
        let mut val_max = 0f32;
        for i in 0..n {
            let w = self.lsp_cos[i];
            let mut p = 0.5f32;
            let mut q = 0.5f32;
            let mut j = 1usize;
            while j < NB_LSP_COEFS {
                q *= w - lsp[j - 1];
                p *= w - lsp[j];
                j += 2;
            }
            p *= p * (2.0 - w);
            q *= q * (2.0 + w);
            let v = self.pow_m1_4(p + q);
            if v > val_max {
                val_max = v;
            }
            self.exponents[ch][i] = v;
        }
        self.max_exponent[ch] = val_max;
        Ok(())
    }

    /// `pow_m1_4`：`x^-0.25` 的位运算近似（带查表插值）—— 必须照抄，换 `powf` 会偏。
    fn pow_m1_4(&self, x: f32) -> f32 {
        let u = x.to_bits();
        let e = (u >> 23) as usize;
        let m = ((u >> (23 - LSP_POW_BITS)) & ((1 << LSP_POW_BITS) - 1)) as usize;
        let t = f32::from_bits(((u << LSP_POW_BITS) & ((1 << 23) - 1)) | (127 << 23));
        let a = self.lsp_pow_m1[m];
        let b = self.lsp_pow_m2[m];
        self.lsp_pow_e[e] * (a + b * t)
    }

    /// 系数 run-level 解码（`ff_wma_run_level_decode`，version 0 分支）。
    fn run_level_decode(
        &mut self,
        r: &mut BitReader,
        tindex: usize,
        ch: usize,
        mut offset: usize,
        num_coefs: usize,
        coef_nb_bits: u32,
    ) -> Result<()> {
        let frame_len_bits = self.params.frame_len_bits;
        let coef_mask = self.block_len - 1;
        while offset < num_coefs {
            let code = self.coef_vlc[tindex].decode(r)? as usize;
            if code > 1 {
                offset += self.run_table[tindex][code] as usize;
                let sign = r.bit()? as i32 - 1;
                let mut lv = self.level_table[tindex][code];
                if sign == -1 {
                    lv = -lv;
                }
                self.coefs1[ch][offset & coef_mask] = lv;
            } else if code == 1 {
                break; // EOB
            } else {
                // escape
                let level = r.get(coef_nb_bits)? as i32;
                offset += r.get(frame_len_bits)? as usize;
                let sign = r.bit()? as i32 - 1;
                self.coefs1[ch][offset & coef_mask] = ((level ^ sign) - sign) as f32;
            }
            offset += 1;
        }
        self.last_coef_count[ch] = offset;
        if offset > num_coefs {
            return Err(Error::Bad("run-level 越界"));
        }
        Ok(())
    }
}

/// IMDCT 的内核（不含 `self`，便于单测直接对着「按定义算」对账）。
///
/// `coefs` 长 `n`（块长），`out` 长 `2n`，`re`/`im` 是长度 ≥ `2n` 的复用 scratch。
///
/// ## 定义（与第一版逐点等价，只换了算法）
///
/// ```text
///   out[i] = −Σ_{k=0}^{n-1} coefs[k]·cos( π/n ·(k+0.5)·(i + 0.5 + n/2) ),   i = 0..2n−1
/// ```
///
/// ## 为什么能走 FFT
///
/// **第一步：`2n` 个输出 = `n` 点 DCT-IV + 三段对称。**
/// 令 `u[j] = Σ_k coefs[k]·cos(π/n·(j+0.5)(k+0.5))`（`j = 0..n−1`，就是 DCT-IV）。
/// 把 `i + 0.5 + n/2` 写成 `(i + n/2) + 0.5`，立刻看出 `out[i] = −u(i + n/2)`
/// （把 `u` 看成 `j` 的连续函数）；而 `u` 有两条对称性，都是一行能验的：
///
/// ```text
///   u(−1−j) = u(j)        （余弦是偶函数）
///   u(j+2n) = −u(j)       （相位差 2π(k+1/2) —— 正好差一个 π）
/// ```
///
/// 用它们把 `i + n/2 ∈ [n/2, 5n/2)` 折回 `[0, n)`：
///
/// ```text
///   i ∈ [0, n/2)     : out[i] = −u[i + n/2]
///   i ∈ [n/2, 3n/2)  : out[i] = +u[3n/2 − 1 − i]
///   i ∈ [3n/2, 2n)   : out[i] = +u[i − 3n/2]
/// ```
///
/// **第二步：DCT-IV 走 FFT。** 把相位拆开（记 `m = 2n`）：
///
/// ```text
///   π/n·(j+0.5)(k+0.5) = πk(2j+1)/m + π(2j+1)/(4n)
///   ⇒  u[j] = Re{ e^{−iφ_j}·D[j] },   φ_j = π(2j+1)/(4n) = (j + 0.5)·π/m
///       D = FFT_m(b),   b[k] = coefs[k]·e^{−iπk/m}（k ≥ n 补零）
/// ```
///
/// `D[j] = Σ_k b[k]·e^{−2πijk/m}` 正好是 **`m` 点 FFT 的第 `j` 个 bin**（前 `n` 个）。
/// 两条相位（预旋转的 `−πk/m`、后旋转的 `−φ_j`）**都是线性的**，所以旋转因子都用
/// 切比雪夫递推生成 —— 一次变换只调 4 次 `sin`/`cos`，不建表。
///
/// ## 为什么值得换
///
/// 原来逐 `(k, i)` 是 **O(n²)**：`n = 2048` 时每块每声道 840 万次内层迭代
/// （1024 点块也有 210 万次）。实测它一个人占了整条转码链路的 **32%**
/// （`sample` 采样：`finish_block` 2007/6261，而 `finish_block` 的时间 99% 在它里面）。
/// 现在是一次 `m` 点 FFT + `O(n)` 旋转，同一份输出。
///
/// ⚠️ **数值上不是逐位相同**：旧版每加一项就把部分和舍进 f32（2048 项 → 误差按随机
/// 游走 ~3e-6 相对），新版在 f64 里累完只收一次（~1e-12）。落到 s16 的 PCM 上是个别
/// 样本 **±1** —— 对 ffmpeg 的对账容差是 4，实测没变（见 `TODO.md`）。
fn imdct_kernel(coefs: &[f32], out: &mut [f32], re: &mut [f64], im: &mut [f64]) {
    let n = coefs.len();
    let m = 2 * n;
    debug_assert_eq!(out.len(), m);
    debug_assert!(n.is_power_of_two(), "块长必须是 2 的幂（FFT 的要求）");
    let pi = std::f64::consts::PI;

    // ---- 预旋转 + 补零：b[k] = coefs[k]·e^{−iπk/m} ----
    {
        let step = pi / m as f64;
        let two_cos = 2.0 * step.cos();
        let mut prev = (-step).cos();
        let mut cur = 1.0f64; // cos(0)
        let mut prev_i = (-step).sin();
        let mut cur_i = 0.0f64; // sin(0)
        for k in 0..m {
            let c = if k < n { f64::from(coefs[k]) } else { 0.0 };
            re[k] = c * cur;
            im[k] = -c * cur_i;
            // cos(θ+step) = 2cos(step)cos(θ) − cos(θ−step)（sin 同理）
            let next = two_cos * cur - prev;
            prev = cur;
            cur = next;
            let next_i = two_cos * cur_i - prev_i;
            prev_i = cur_i;
            cur_i = next_i;
        }
    }

    fft_in_place(re, im);

    // ---- 后旋转：u[j] = Re{ e^{−iφ_j}·D[j] }，就地写回 `im` ----
    {
        let step = pi / m as f64;
        let base = step * 0.5; // φ_0
        let two_cos = 2.0 * step.cos();
        let mut prev = (base - step).cos();
        let mut cur = base.cos();
        let mut prev_i = (base - step).sin();
        let mut cur_i = base.sin();
        for j in 0..n {
            let (r, i2) = (re[j], im[j]);
            im[j] = r * cur + i2 * cur_i;
            let next = two_cos * cur - prev;
            prev = cur;
            cur = next;
            let next_i = two_cos * cur_i - prev_i;
            prev_i = cur_i;
            cur_i = next_i;
        }
    }

    // ---- 三段对称展开（`u` 现在在 `im[..n]`）----
    let u = &im[..n];
    for i in 0..n / 2 {
        out[i] = -(u[i + n / 2] as f32);
    }
    for i in n / 2..3 * n / 2 {
        out[i] = u[3 * n / 2 - 1 - i] as f32;
    }
    for i in 3 * n / 2..m {
        out[i] = u[i - 3 * n / 2] as f32;
    }
}

/// 指数值：`10^(exp / 16)`。
///
/// ⚠️ ffmpeg 的表是 `pow_tab[i] = 10^((i-60)/16)`（156 项，第 60 项 = 1.0），
/// 而代码里取的是 `ptab = pow_tab + 60` 再 `ptab[last_exp]` —— 也就是
/// **`10^(last_exp/16)`**。第一版把表的公式当成了调用公式，指数小了 55 倍，
/// `mult` 就大了 55 倍 —— v1 的 PCM 全错（幅度比乱、峰值高 21%）。
///
/// 用 f64 算完再收 f32，与 ffmpeg 的表生成方式一致。
fn pow_tab(exp: i32) -> f32 {
    10f64.powf(exp as f64 / 16.0) as f32
}

/// `ff_wma_total_gain_to_bits`。
fn total_gain_to_bits(total_gain: i32) -> u32 {
    if total_gain < 15 {
        13
    } else if total_gain < 32 {
        12
    } else if total_gain < 40 {
        11
    } else if total_gain < 45 {
        10
    } else {
        9
    }
}

/// 建一套系数 VLC（`init_coef_vlc`）。
fn build_coef_vlc(table: usize) -> (Vlc, Vec<u16>, Vec<f32>) {
    let codes: &[(u32, u8)] = match table {
        0 => &COEF0_CODES,
        1 => &COEF1_CODES,
        2 => &COEF2_CODES,
        3 => &COEF3_CODES,
        4 => &COEF4_CODES,
        _ => &COEF5_CODES,
    };
    let levels: &[u16] = match table {
        0 => &COEF0_LEVELS,
        1 => &COEF1_LEVELS,
        2 => &COEF2_LEVELS,
        3 => &COEF3_LEVELS,
        4 => &COEF4_LEVELS,
        _ => &COEF5_LEVELS,
    };
    let n = codes.len();
    let mut run_table = vec![0u16; n];
    let mut level_table = vec![0f32; n];
    let mut i = 2usize;
    let mut level = 1f32;
    let mut k = 0usize;
    while i < n && k < levels.len() {
        let l = levels[k] as usize;
        k += 1;
        for j in 0..l {
            if i >= n {
                break;
            }
            run_table[i] = j as u16;
            level_table[i] = level;
            i += 1;
        }
        level += 1.0;
    }
    (Vlc::from_codes(codes), run_table, level_table)
}

/// v2 的硬编码指数频带表（`exponent_band_*`，`a = frame_len_bits - 7 - k < 3` 时用）。
///
/// 每行第 0 个元素是**频带个数**（ffmpeg 里 `n = *table++`），后面才是宽度。
fn exponent_band_table(sample_rate: u32, a: usize) -> Option<Vec<usize>> {
    let row: &[u8; 25] = if sample_rate >= 44100 {
        &EXPONENT_BAND_44100[a]
    } else if sample_rate >= 32000 {
        &EXPONENT_BAND_32000[a]
    } else if sample_rate >= 22050 {
        &EXPONENT_BAND_22050[a]
    } else {
        return None;
    };
    let n = row[0] as usize;
    Some(row[1..1 + n].iter().map(|&v| v as usize).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 直接按定义算的 IMDCT —— oracle（O(n²)）。
    ///
    /// 与 `v2v-aac` 那边的 `fast_mdct_matches_the_definition` 是同一个套路：
    /// 换算法（O(n²) → O(n log n)）时，只有它能把「折叠 / 旋转 / 定标写错」当场抓住。
    fn direct_imdct(coefs: &[f32]) -> Vec<f32> {
        let n = coefs.len();
        let mut out = vec![0f32; 2 * n];
        for (i, o) in out.iter_mut().enumerate() {
            let mut s = 0f64;
            for (k, &c) in coefs.iter().enumerate() {
                let ang = std::f64::consts::PI / n as f64
                    * (k as f64 + 0.5)
                    * (i as f64 + 0.5 + n as f64 / 2.0);
                s -= f64::from(c) * ang.cos();
            }
            *o = s as f32;
        }
        out
    }

    #[test]
    fn fast_imdct_matches_the_definition() {
        // 覆盖 WMA 真会用的块长：`BLOCK_MIN_BITS = 7` → 128，到 `frame_len` 上限 2048。
        for &n in &[128usize, 512, 2048] {
            let mut state = 0x5eed_1234u32;
            let coefs: Vec<f32> = (0..n)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((state >> 16) as f32 / 32768.0 - 1.0) * 500.0
                })
                .collect();
            let want = direct_imdct(&coefs);

            let m = 2 * n;
            let mut got = vec![0f32; m];
            let mut re = vec![0f64; m];
            let mut im = vec![0f64; m];
            imdct_kernel(&coefs, &mut got, &mut re, &mut im);

            let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = got
                .iter()
                .zip(&want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            let rel = f64::from(worst) / f64::from(peak.max(1e-30));
            assert!(rel < 1e-6, "n={n}：与定义的最大相对偏差 {rel:.3e}");
        }
    }
}
