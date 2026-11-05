//! AAC-LC **解码**：位流解析（`parse_frame`）+ 合成出 PCM（`decode_frame`）。
//!
//! ## 为什么先做解析、不先做 DSP
//!
//! 解码器里最容易「整体对、个别错」的是**位流语法**：元素顺序、`common_window`
//! 只写一次、section 与 scalefactor「先把所有组写完」再写谱数据 —— 错一处整条流
//! 从第二个元素开始错位，而症状是「声音变成噪声」，看起来像 IMDCT 的锅。
//! 所以先把**结构**解出来，用我们自己的编码器当 oracle（它知道自己写了什么），
//! 再上谱数据的 Huffman 与 IMDCT。
//!
//! ## 现在能解什么
//!
//! - `AudioSpecificConfig`（5+4+4 + GASpecificConfig 3 位）；
//! - `raw_data_block`：`SCE` / `CPE`（含 `common_window` 与 M/S 掩码）/ `END`；
//! - `ics_info`：长窗与八个短窗、`max_sfb`、分组；
//! - `section_data`（4 位码本号 + 11 的 escape）、`scale_factor_data`（Huffman）；
//! - `tns_data`：长窗与**短窗**，四张系数量化表全支持（`coef_res` × `coef_compress`）；
//! - 谱数据：长窗与**短窗**（短窗按「组 → 带 → 组内窗口」读，落点见 `parse_spectral`）；
//! - 出 PCM：反量化 `|q|^(4/3) · 2^((sf−100)/4)` → M/S → TNS 逆滤波 →
//!   IMDCT + 窗 + 重叠相加（`synthesis`）。
//!
//! ## 还没做
//!
//! - PNS（噪声码本）、强度立体声、`pulse_data`、`gain_control`、SBR/PS。
//!
//! 遇到还没做的部分**明确报 `Unsupported`**，不猜、不静默跳过。

use crate::encode::{CB_DIM, CB_ESC, CB_MAXVAL, CB_RANGE, CB_UNSIGNED};
use crate::mdct::{LONG_LEN, SHORT_LEN};
use crate::synthesis::Synthesis;
use crate::{Error, Result};
use v2v_tables::aac as t;

/// 一个 TNS 滤波器（解码端只用得上「作用在哪个窗口 / 覆盖几条带 / 阶数 / 方向 / LPC」）。
///
/// `lpc` 是**量化值**推出来的（`tns_tmp2_map_*` 查表 → `lpc_from_ref`），
/// 与编码端用的是同一套值 —— 用原始反射系数会与位流里的对不上。
#[derive(Debug, Clone)]
pub struct TnsFilter {
    /// 作用在哪个窗口上。长窗恒 0；短窗 0..8（每个短窗各有一份独立的滤波器）。
    pub window: u8,
    /// 覆盖多少条频带（从**最高**的频带往下数）。
    pub length: u8,
    pub order: u8,
    /// `true` = 从高频往低频扫。
    pub direction: bool,
    pub lpc: Vec<f32>,
}

/// 一个通道的 ICS 头（不含谱数据）。
#[derive(Debug, Clone, Default)]
pub struct IcsInfo {
    pub global_gain: u8,
    /// 0 = 长窗；1 = 长起始；2 = 八个短窗；3 = 长终止。
    pub window_sequence: u8,
    /// 0 = 正弦窗；1 = KBD。
    pub window_shape: u8,
    pub max_sfb: u8,
    /// 八个短窗的分组（`scale_factor_grouping` 的 7 位，长窗为 0）。
    pub grouping: u8,
    /// 每个 sfb 的码本号（按分组展开；`0` = 零区，`13` = 噪声）。
    pub section_cb: Vec<u8>,
    /// 每个 sfb 的 scalefactor（按分组展开）。
    pub scalefactor: Vec<i16>,
    /// TNS 滤波器（空 = 这一帧没有 TNS）。
    pub tns: Vec<TnsFilter>,
}

/// 一个元素（SCE / CPE）。
#[derive(Debug, Clone)]
pub struct Element {
    /// 0 = SCE，1 = CPE。
    pub kind: u8,
    pub tag: u8,
    /// CPE 的 `common_window`；SCE 恒为 0。
    pub common_window: bool,
    /// `ms_mask_present`（只有 CPE 且 common_window 时有效）。
    pub ms_mask_present: u8,
    /// 逐 (组, sfb) 的 M/S 掩码，**只有 `ms_mask_present = 1` 时才非空**。
    ///
    /// `2` 是「所有带都用」（位流里不跟掩码），`0` 是「都不用」—— 两种情况都不需要
    /// 逐带记录，所以这里留空、由 `decode_frame` 按 `ms_mask_present` 分支。
    pub ms_mask: Vec<bool>,
    /// 每个通道一个（SCE 1 个，CPE 2 个）。
    pub channels: Vec<IcsInfo>,
    /// 谱数据开始的位偏移（M2 从这里接着解）。
    pub spectral_bit_pos: usize,
    /// 每个通道的量化谱系数（`frame_len` 个；M2 解出来的，尚未反量化）。
    pub spectra: Vec<Vec<i32>>,
}

/// 解出来的帧结构。
#[derive(Debug, Clone)]
pub struct FrameInfo {
    pub elements: Vec<Element>,
    /// 解析结束时用掉的位数。`len * 8 - bits_used < 8` 说明位流被**恰好**读完
    /// （只剩字节对齐的补零）—— 这是「语法没读偏」最强的间接判据。
    pub bits_used: usize,
}

/// AAC-LC 解码器。
pub struct Decoder {
    pub object_type: u8,
    pub sample_rate: u32,
    pub sample_rate_index: u8,
    pub channels: u8,
    /// 每帧的采样数（LC 恒 1024；`frame_length_flag = 1` 时是 960，暂不支持）。
    pub frame_len: usize,
    /// 每个声道的合成器（窗表 + 重叠历史）。`decode_frame` 才用得上。
    ola: Synthesis,
}

impl Decoder {
    /// 从 `esds` 的 AudioSpecificConfig 建解码器。
    pub fn new(asc: &[u8]) -> Result<Self> {
        if asc.len() < 2 {
            return Err(Error::Bad("AudioSpecificConfig 太短".into()));
        }
        let mut r = BitReader::new(asc);
        let object_type = r.u(5)? as u8;
        let sample_rate_index = r.u(4)? as u8;
        if sample_rate_index == 0xF {
            return Err(Error::Unsupported("显式采样率（index = 15）暂不支持".into()));
        }
        let channels = r.u(4)? as u8;
        if channels == 0 {
            return Err(Error::Unsupported("PCE 描述的声道布局暂不支持".into()));
        }
        // GASpecificConfig
        let frame_length_flag = r.u(1)?;
        if frame_length_flag != 0 {
            return Err(Error::Unsupported("960 样本帧暂不支持".into()));
        }
        let depends_on_core = r.u(1)?;
        let ext = r.u(1)?;
        if depends_on_core != 0 || ext != 0 {
            return Err(Error::Unsupported("core coder / extension 暂不支持".into()));
        }
        if object_type != 2 {
            return Err(Error::Unsupported(format!(
                "只做 AAC-LC（object_type = 2），收到 {object_type}"
            )));
        }
        let sample_rate = t::SAMPLE_RATES
            .get(sample_rate_index as usize)
            .copied()
            .ok_or_else(|| Error::Bad("采样率索引越界".into()))?;
        Ok(Self {
            object_type,
            sample_rate,
            sample_rate_index,
            channels,
            frame_len: 1024,
            ola: Synthesis::new(channels as usize),
        })
    }

    /// 解析一帧的**结构**（不碰谱数据）。
    pub fn parse_frame(&self, data: &[u8]) -> Result<FrameInfo> {
        let mut r = BitReader::new(data);
        let mut elements = Vec::new();
        loop {
            let id = r.u(3)? as u8;
            match id {
                0 => elements.push(self.parse_sce(&mut r)?),
                1 => elements.push(self.parse_cpe(&mut r)?),
                6 => self.skip_fill(&mut r)?,
                7 => break, // ID_END
                other => {
                    return Err(Error::Unsupported(format!(
                        "元素类型 {other}（CCE / LFE / DSE / PCE）暂不支持"
                    )))
                }
            }
        }
        Ok(FrameInfo {
            elements,
            bits_used: r.pos(),
        })
    }

    /// 解一帧到 PCM：每个声道 1024 个样本，`f32`、与编码器输入**同一个量级**
    /// （不是 ±1 归一化 —— 与 `wma` / `player` 的约定一致，宿主自己决定怎么用）。
    ///
    /// 顺序是规范定的，换一步都会「解得开但波形错」：
    ///
    /// ```text
    ///   Huffman → 反量化 → M/S → TNS 逆滤波 → IMDCT + 窗 + 重叠相加
    /// ```
    ///
    /// ⚠️ 第一帧交出的 1024 个样本是**编码器的 priming**（上一帧的重叠是全零），
    /// 与 ffmpeg 解同一文件时逐样本对得上，但**不等于原始波形**的起点 ——
    /// 拿它跟输入 PCM 比 SNR 要先跳过前 1024 个样本。
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<Vec<Vec<f32>>> {
        let info = self.parse_frame(data)?;
        let nch: usize = info.elements.iter().map(|e| e.channels.len()).sum();
        if nch != self.channels as usize {
            return Err(Error::Unsupported(format!(
                "这一帧有 {nch} 个声道，AudioSpecificConfig 说是 {}",
                self.channels
            )));
        }
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(nch);
        for e in &info.elements {
            let mut specs: Vec<Vec<f32>> = e
                .channels
                .iter()
                .zip(&e.spectra)
                .map(|(ics, s)| self.dequantize(ics, s))
                .collect();
            self.apply_ms(&mut specs, e);
            for (ics, spec) in e.channels.iter().zip(specs.iter_mut()) {
                self.apply_tns(ics, spec);
                let mut pcm = vec![0f32; self.frame_len];
                let ch = out.len();
                self.ola
                    .frame(ch, ics.window_sequence, ics.window_shape, spec, &mut pcm);
                out.push(pcm);
            }
        }
        Ok(out)
    }

    /// 丢掉重叠历史（seek 之后必须调）。
    pub fn reset(&mut self) {
        self.ola.reset();
    }

    /// 某个窗序列用的频带边界表（短窗 128 点、长窗 1024 点）。
    fn swb(&self, short: bool) -> &'static [u16] {
        let sr = self.sample_rate_index as usize;
        if short {
            &t::SWB_OFFSET_128[sr][..t::SWB_OFFSET_128_LEN[sr] as usize]
        } else {
            &t::SWB_OFFSET_1024[sr][..t::SWB_OFFSET_1024_LEN[sr] as usize]
        }
    }

    /// 反量化：`sign(q)·|q|^(4/3)·2^((sf−100)/4)`。
    ///
    /// 短窗的落点与 `parse_spectral` 一致（窗口主序 `[w*128 + k]`），
    /// 而位流里的带数据是「带主序」—— 那个重排已经在解析时做完了。
    fn dequantize(&self, ics: &IcsInfo, spec: &[i32]) -> Vec<f32> {
        let short = ics.window_sequence == 2;
        let stride = if short { SHORT_LEN } else { LONG_LEN };
        let swb = self.swb(short);
        let max_sfb = ics.max_sfb as usize;
        let mut out = vec![0f32; self.frame_len];
        let mut w0 = 0usize;
        for (g, &gl) in window_group_lens(ics).iter().enumerate() {
            for sfb in 0..max_sfb {
                let cb = ics.section_cb[g * max_sfb + sfb];
                if cb == 0 {
                    continue;
                }
                let sf = ics.scalefactor[g * max_sfb + sfb].clamp(0, 255) as u8;
                let gain = sf_gain(sf);
                let a = swb[sfb] as usize;
                let b = swb[sfb + 1] as usize;
                if b > stride || a >= b {
                    break;
                }
                for j in 0..gl {
                    let base = (w0 + j) * stride;
                    for k in a..b {
                        out[base + k] = dequant_one(spec[base + k]) * gain;
                    }
                }
            }
            w0 += gl;
        }
        out
    }

    /// M/S 反矩阵：`L = M + S`、`R = M − S`（编码端是 `M = (L+R)/2`、`S = (L−R)/2`）。
    ///
    /// 逐带掩码（`ms_mask_present = 1`）时**只对掩码为 1 的带**做 —— 漏掉这一条，
    /// 没做 M/S 的带会被「解」成 M+S，波形错但位流完全合法。
    fn apply_ms(&self, specs: &mut [Vec<f32>], e: &Element) {
        if specs.len() != 2 || e.ms_mask_present == 0 {
            return;
        }
        let ics = &e.channels[0];
        let short = ics.window_sequence == 2;
        let stride = if short { SHORT_LEN } else { LONG_LEN };
        let swb = self.swb(short);
        let max_sfb = ics.max_sfb as usize;
        let all = e.ms_mask_present == 2;
        let (left, right) = specs.split_at_mut(1);
        let (l, r) = (&mut left[0], &mut right[0]);
        let mut w0 = 0usize;
        for (g, &gl) in window_group_lens(ics).iter().enumerate() {
            for sfb in 0..max_sfb {
                let used = if all {
                    true
                } else {
                    e.ms_mask.get(g * max_sfb + sfb).copied().unwrap_or(false)
                };
                if !used {
                    continue;
                }
                let a = swb[sfb] as usize;
                let b = swb[sfb + 1] as usize;
                if b > stride || a >= b {
                    break;
                }
                for j in 0..gl {
                    let base = (w0 + j) * stride;
                    for k in a..b {
                        let m = l[base + k];
                        let s = r[base + k];
                        l[base + k] = m + s;
                        r[base + k] = m - s;
                    }
                }
            }
            w0 += gl;
        }
    }

    /// TNS 逆滤波（AR，用**已重建**的谱）。与编码端的 MA 滤波严格互逆 ——
    /// `tns::tests::ma_filter_is_inverted_by_ar_filter` 钉着这条。
    ///
    /// 短窗是**每个窗口各一份**滤波器（`TnsFilter::window`），作用在该窗口自己的
    /// 128 个系数上（`spec[w*128 .. w*128+128]`，与 `parse_spectral` 的落点一致）。
    /// 覆盖带数从各自窗口的**最高带**往下数 —— 所以每换一个窗口，`bottom` 都要
    /// 重置回 `num_swb`；忘了重置会让第二个短窗的 TNS 从「上一条滤波器的下界」
    /// 接着往下数，越走越低，最终落在几个低频带上（症状：只有短窗帧的中频发闷）。
    fn apply_tns(&self, ics: &IcsInfo, spec: &mut [f32]) {
        if ics.tns.is_empty() {
            return;
        }
        let short = ics.window_sequence == 2;
        let sr = self.sample_rate_index as usize;
        let stride = if short { SHORT_LEN } else { LONG_LEN };
        let swb = self.swb(short);
        let tns_max = if short {
            t::TNS_MAX_BANDS_128[sr]
        } else {
            t::TNS_MAX_BANDS_1024[sr]
        } as usize;
        let mmm = tns_max.min(ics.max_sfb as usize);
        let num_swb = swb.len() - 1;
        let mut bottom = num_swb;
        let mut cur_window = 0u8;
        for f in &ics.tns {
            if f.window != cur_window {
                cur_window = f.window;
                bottom = num_swb;
            }
            let top = bottom;
            bottom = top.saturating_sub(f.length as usize);
            // `bottom` 必须在 `order == 0` 之前推进：阶数为 0 的滤波器不作用，
            // 但它的 `length` 仍然占着频带，后面的滤波器要从它下面接着数。
            if f.order == 0 {
                continue;
            }
            let start = swb[bottom.min(mmm)] as usize;
            let end = swb[top.min(mmm)] as usize;
            if end <= start {
                continue;
            }
            let base = f.window as usize * stride;
            let size = end - start;
            let (start, inc): (isize, isize) = if f.direction {
                (end as isize - 1, -1)
            } else {
                (start as isize, 1)
            };
            for m in 0..size {
                let k = base + (start + m as isize * inc) as usize;
                for i in 1..=(f.order as usize).min(m) {
                    let j = base + (start + (m as isize - i as isize) * inc) as usize;
                    spec[k] -= f.lpc[i - 1] * spec[j];
                }
            }
        }
    }

    /// `ID_FIL`（填充元素）：**跳过**。
    ///
    /// ffmpeg 的编码器会用它在帧尾补齐字节对齐（CBR 下还用来填充到固定帧长），
    /// 所以「只解我们自己产的流」时看不到它，一碰真文件就会撞上。
    /// 载荷是扩展数据（SBR / DRM 之类），LC 用不上 —— 跳过就是全部语义。
    fn skip_fill(&self, r: &mut BitReader) -> Result<()> {
        let mut count = r.u(4)?;
        if count == 15 {
            // 转义：`count = 15 + esc − 1`。`esc = 0` 时结果是 14 —— 用
            // `saturating_sub` 而不是 `- 1`：解码器吃的是别人的字节，
            // 下溢 panic 会把整个转换流程带走（fuzz 判据是「允许 Err，禁止 panic」）。
            count = 15 + r.u(8)?.saturating_sub(1);
        }
        for _ in 0..count * 8 {
            r.u(1)?;
        }
        Ok(())
    }

    fn parse_sce(&self, r: &mut BitReader) -> Result<Element> {        let tag = r.u(4)? as u8;
        let ch = self.parse_ics(r, false)?;
        let sp = r.pos();
        let spec = self.parse_spectral(r, &ch)?;
        Ok(Element {
            kind: 0,
            tag,
            common_window: false,
            ms_mask_present: 0,
            ms_mask: Vec::new(),
            channels: vec![ch],
            spectral_bit_pos: sp,
            spectra: vec![spec],
        })
    }

    fn parse_cpe(&self, r: &mut BitReader) -> Result<Element> {


        let tag = r.u(4)? as u8;
        let common_window = r.u(1)? != 0;
        let mut channels = Vec::new();
        let mut spectra = Vec::new();
        let mut ms_mask_present = 0;
        let mut ms_mask: Vec<bool> = Vec::new();
        let sp;
        if common_window {
            // ⚠️ 顺序：`ics_info` → `ms_mask_present` → 两个通道各自的 `global_gain`。
            let shared = self.parse_ics_info(r)?;
            ms_mask_present = r.u(2)? as u8;
            // ⚠️ 语义容易记反（编码器那边有同样一条注释）：**2 = 全部频带都用 M/S**
            // （后面不跟掩码位），**1 = 后面跟逐带掩码**。记反了整条流错位，
            // 症状是「Number of bands exceeds limit」这种离根因很远的报错。
            match ms_mask_present {
                0 | 2 => {}
                1 => {
                    let bits = shared.max_sfb as usize * num_window_groups(&shared);
                    ms_mask.reserve(bits);
                    for _ in 0..bits {
                        ms_mask.push(r.u(1)? != 0);
                    }
                }
                _ => return Err(Error::Bad("ms_mask_present = 3 是保留值")),
            }
            let ch0 = self.parse_ics_body(r, &shared)?;
            sp = r.pos();
            let spec0 = self.parse_spectral(r, &ch0)?;
            let ch1 = self.parse_ics_body(r, &shared)?;
            let spec1 = self.parse_spectral(r, &ch1)?;
            channels.push(ch0);
            channels.push(ch1);
            spectra.push(spec0);
            spectra.push(spec1);
        } else {
            let ch0 = self.parse_ics(r, false)?;
            sp = r.pos();
            let spec0 = self.parse_spectral(r, &ch0)?;
            let ch1 = self.parse_ics(r, false)?;
            let spec1 = self.parse_spectral(r, &ch1)?;
            channels.push(ch0);
            channels.push(ch1);
            spectra.push(spec0);
            spectra.push(spec1);
        }
        Ok(Element {
            kind: 1,
            tag,
            common_window,
            ms_mask_present,
            ms_mask,
            channels,
            spectral_bit_pos: sp,
            spectra,
        })
    }

    /// `individual_channel_stream`（**不带** `common_window` 时）：
    /// **`global_gain` 在前、`ics_info` 在后**（与 common_window 那条相反 ——
    /// 那条是元素头先写 `ics_info`，每个声道只剩 `global_gain`）。
    /// 顺序写反了：解码端会先读 11 位的 ics_info 当增益，整条流从第一个元素就错位。
    fn parse_ics(&self, r: &mut BitReader, _write_ics: bool) -> Result<IcsInfo> {
        let global_gain = r.u(8)? as u8;
        let mut ics = self.parse_ics_info(r)?;
        ics.global_gain = global_gain;
        self.parse_sections(r, &mut ics)?;
        self.parse_scalefactors(r, &mut ics)?;
        if r.u(1)? != 0 {
            return Err(Error::Unsupported("pulse_data 暂不支持".into()));
        }
        if r.u(1)? != 0 {
            ics.tns = self.parse_tns(r, ics.window_sequence == 2)?;
        }
        if r.u(1)? != 0 {
            return Err(Error::Unsupported("gain_control 暂不支持".into()));
        }
        Ok(ics)
    }

    /// `common_window` 的第二个通道：不重读 ics_info，其余照常。
    fn parse_ics_body(&self, r: &mut BitReader, shared: &IcsInfo) -> Result<IcsInfo> {
        let global_gain = r.u(8)? as u8;
        let mut ics = shared.clone();
        ics.global_gain = global_gain;
        ics.section_cb.clear();
        ics.scalefactor.clear();
        self.parse_sections(r, &mut ics)?;
        self.parse_scalefactors(r, &mut ics)?;
        if r.u(1)? != 0 {
            return Err(Error::Unsupported("pulse_data 暂不支持".into()));
        }
        if r.u(1)? != 0 {
            ics.tns = self.parse_tns(r, ics.window_sequence == 2)?;
        }
        if r.u(1)? != 0 {
            return Err(Error::Unsupported("gain_control 暂不支持".into()));
        }
        Ok(ics)
    }

    fn parse_ics_info(&self, r: &mut BitReader) -> Result<IcsInfo> {
        r.u(1)?; // ics_reserved_bit
        let window_sequence = r.u(2)? as u8;
        let window_shape = r.u(1)? as u8;
        let (max_sfb, grouping) = if window_sequence == 2 {
            let max_sfb = r.u(4)? as u8;
            let grouping = r.u(7)? as u8;
            (max_sfb, grouping)
        } else {
            let max_sfb = r.u(6)? as u8;
            // predictor_data_present —— LC 里必须是 0
            if r.u(1)? != 0 {
                return Err(Error::Unsupported("预测器（Main profile）暂不支持".into()));
            }
            (max_sfb, 0)
        };
        Ok(IcsInfo {
            global_gain: 0,
            window_sequence,
            window_shape,
            max_sfb,
            grouping,
            section_cb: Vec::new(),
            scalefactor: Vec::new(),
            tns: Vec::new(),
        })
    }

    /// 谱数据。**长窗**落点是 `swb[sfb]..swb[sfb+1]`；**短窗**是「组 → 带 → 组内窗口」，
    /// 位流里一条带的数据是**组内各窗口这条带的拼接**，落点见下面的 scatter。
    ///
    /// ⚠️ 短窗的位流顺序是**带主序**（一条带里先放窗口 0、再窗口 1…），不是窗口主序。
    /// 而本函数的输出按**窗口主序**（`[w*128 + k]`）—— 因为后面 IMDCT 是逐窗口做的。
    /// 两个顺序混了不会报错，只会让重建出来的瞬态帧变成噪声。
    fn parse_spectral(&self, r: &mut BitReader, ics: &IcsInfo) -> Result<Vec<i32>> {
        let sr = self.sample_rate_index as usize;
        let mut out = vec![0i32; self.frame_len];
        if ics.window_sequence == 2 {
            let len = t::SWB_OFFSET_128_LEN[sr] as usize;
            let swb = &t::SWB_OFFSET_128[sr][..len];
            let max_sfb = ics.max_sfb as usize;
            let mut w0 = 0usize;
            let mut scratch: Vec<i32> = Vec::new();
            for (g, &gl) in window_group_lens(ics).iter().enumerate() {
                for sfb in 0..max_sfb {
                    let cb = ics.section_cb[g * max_sfb + sfb];
                    let a = swb[sfb] as usize;
                    let b = swb[sfb + 1] as usize;
                    if b > SHORT_LEN || a >= b {
                        break;
                    }
                    let width = b - a;
                    scratch.clear();
                    scratch.resize(width * gl, 0);
                    self.decode_band(r, cb, &mut scratch)?;
                    for j in 0..gl {
                        for k in 0..width {
                            out[(w0 + j) * SHORT_LEN + a + k] = scratch[j * width + k];
                        }
                    }
                }
                w0 += gl;
            }
            return Ok(out);
        }
        let len = t::SWB_OFFSET_1024_LEN[sr] as usize;
        let swb = &t::SWB_OFFSET_1024[sr][..len];
        for sfb in 0..ics.max_sfb as usize {
            let cb = ics.section_cb[sfb];
            let a = swb[sfb] as usize;
            let b = swb[sfb + 1] as usize;
            if b > self.frame_len || a >= b {
                break;
            }
            self.decode_band(r, cb, &mut out[a..b])?;
        }
        Ok(out)
    }

    /// `tns_data`：**每个窗口**先读 `n_filt`（长窗 2 位 / 短窗 1 位），有滤波器时读
    /// `coef_res`，再逐个读 `(length, order, direction, coef_compress, coefs)`。
    ///
    /// ⚠️ 系数是**四张表**，不是一个 `coef_res` 加一个 `coef_compress` 的简单组合。
    /// 位流里读的位数是 `coef_res + 3 − coef_compress`（4 / 3 / 3 / 2），
    /// **查的表也跟着换**（`tns_tmp2_map_<compress>_<res>`）。
    ///
    /// 四张表**不是**彼此的前缀：`_1_4` 的第 4~7 项是 `0.6737, 0.5264, 0.3612,
    /// 0.1837`，而 `_0_4` 同位置是 `-0.7431, -0.8660, -0.9511, -0.9945`。所以
    /// 「压缩 = 少读一位、仍查 4 位表」会得到一条**合法但错误**的滤波器：
    /// 不报错、不位移、位数也恰好对，只是 TNS 用了错的系数 —— 属于「看起来全对」
    /// 的那类错，只能靠和 ffmpeg 逐样本比对才发现。
    fn parse_tns(&self, r: &mut BitReader, short: bool) -> Result<Vec<TnsFilter>> {
        let windows: u8 = if short { 8 } else { 1 };
        // 长窗的 `order` 是 5 位（最大 31），但 LC 的 TNS 最多 12 阶；短窗 3 位
        // 天然封顶在 7。超上限就是位流有问题，报错而不是硬解出一条怪滤波器。
        let max_order: u8 = if short { 7 } else { 12 };
        let mut filters = Vec::new();
        for w in 0..windows {
            let n_filt = r.u(if short { 1 } else { 2 })? as usize;
            if n_filt == 0 {
                continue;
            }
            let res4 = r.u(1)? != 0;
            for _ in 0..n_filt {
                let length = r.u(if short { 4 } else { 6 })? as u8;
                let order = r.u(if short { 3 } else { 5 })? as u8;
                if order > max_order {
                    return Err(Error::Bad("TNS 阶数超出上限"));
                }
                if order == 0 {
                    // 阶数为 0 时后面**没有** direction / coef_compress / 系数位。
                    filters.push(TnsFilter {
                        window: w,
                        length,
                        order: 0,
                        direction: false,
                        lpc: Vec::new(),
                    });
                    continue;
                }
                let direction = r.u(1)? != 0;
                let compress = r.u(1)? != 0;
                let (table, bits): (&'static [f32], u32) = match (res4, compress) {
                    (true, false) => (&t::TNS_COEF_4BIT, 4),
                    (true, true) => (&t::TNS_COEF_4BIT_COMPRESSED, 3),
                    (false, false) => (&t::TNS_COEF_3BIT, 3),
                    (false, true) => (&t::TNS_COEF_3BIT_COMPRESSED, 2),
                };
                let mut vals = Vec::with_capacity(order as usize);
                for _ in 0..order {
                    vals.push(table[r.u(bits)? as usize]);
                }
                filters.push(TnsFilter {
                    window: w,
                    length,
                    order,
                    direction,
                    lpc: crate::tns::lpc_from_ref(&vals),
                });
            }
        }
        Ok(filters)
    }

    /// 一个频带的谱系数（Huffman + 符号位 + escape）。
    ///
    /// 输出的是**量化值 `q`**（`-(maxval..=maxval)` 或 `0..=maxval`），不是
    /// `CODEBOOK_VECTORS` 里那张表的值 —— 这两者**不一样**，见下面的 ⚠️。
    ///
    /// ⚠️ `CODEBOOK_VECTORS` 存的是 `trunc(|q|^(4/3))`（已经开过 4/3 次方的幅值，
    /// ffmpeg 的定点和浮点两条路共用它），**不是 `q`**。曾经直接把表值当 `q`
    /// 再开一次方，于是每个**非 escape** 系数都被多放大 1.5~2.7 倍：
    /// escape 的系数走的是另一条路（读变长整数、再开方），所以**低频带全对、
    /// 中高频带偏大 2.5 倍**。这种「只有一部分带错、错的还是固定倍数」的分布
    /// 最容易被误判成「频带表/定标」问题 —— 实际是**同一个数字的两种含义**。
    ///
    /// 所以 `q` 从索引的**位权数字**直接算（`idx` 是 `range` 进制拼起来的），
    /// 那张表只在测试里当 oracle 用（见 `codebook_vectors_match_the_power_of_q`）。
    fn decode_band(&self, r: &mut BitReader, cb: u8, out: &mut [i32]) -> Result<()> {
        if cb == 0 {
            out.fill(0);
            return Ok(());
        }
        let cb = cb as usize;
        let dim = CB_DIM[cb];
        let range = usize::from(CB_RANGE[cb]);
        let maxval = i32::from(CB_MAXVAL[cb]);
        let unsigned = CB_UNSIGNED[cb];
        let codes = &t::SPECTRAL_CODE[cb - 1];
        let lens = &t::SPECTRAL_BITS[cb - 1];
        let size = t::SPECTRAL_SIZE[cb - 1] as usize;
        let mut pos = 0usize;
        while pos < out.len() {
            let idx = decode_spectral(r, codes, lens, size)?;
            let mut vals = [0i32; 4];
            // `curidx = ((d0·range + d1)·range + d2)…`，反过来取数字。
            let mut t = idx;
            for j in (0..dim).rev() {
                let digit = (t % range) as i32;
                t /= range;
                vals[j] = if unsigned { digit } else { digit - maxval };
            }
            // 无符号码本：非零维逐个跟一个符号位。
            if unsigned {
                for v in vals.iter_mut().take(dim) {
                    if *v != 0 && r.u(1)? != 0 {
                        *v = -*v;
                    }
                }
            }
            // escape 码本：`q == maxval`（16）的那一维后面跟变长整数。
            if cb == CB_ESC {
                for v in vals.iter_mut().take(dim) {
                    if v.abs() == maxval {
                        let mag = read_escape(r)?;
                        *v = if *v < 0 { -mag } else { mag };
                    }
                }
            }
            for j in 0..dim {
                if pos + j < out.len() {
                    out[pos + j] = vals[j];
                }
            }
            pos += dim;
        }
        Ok(())
    }

    /// `section_data`：`sect_cb`（4 位）+ `sect_len_incr`（长窗 5 位 / 短窗 3 位），
    /// 反复直到覆盖 `max_sfb`。
    ///
    /// ⚠️ `sect_len_incr` **不是**「段长」而是「额外多少带」：段长 = 1 + 累计增量，
    /// 增量等于全 1（31 / 7）时还要再读一个。少读这一位，后面整条流错位。
    fn parse_sections(&self, r: &mut BitReader, ics: &mut IcsInfo) -> Result<()> {
        let short = ics.window_sequence == 2;
        let run_bits = if short { 3 } else { 5 };
        let max_run = (1u32 << run_bits) - 1;
        let groups = num_window_groups(ics);
        for _ in 0..groups {
            let mut sfb = 0usize;
            while sfb < ics.max_sfb as usize {
                let cb = r.u(4)? as u8;
                // 0 = 零区；1..=11 = 谱码本（**11 是 ESC，合法**）；
                // 12 = 保留；13 = PNS；14/15 = 强度立体声。
                match cb {
                    0..=11 => {}
                    12 => return Err(Error::Bad("码本号 12 是保留值")),
                    13 => return Err(Error::Unsupported("噪声码本（PNS）暂不支持".into())),
                    _ => return Err(Error::Unsupported("强度立体声（intensity）暂不支持".into())),
                }
                // `sect_len` = 各 `sect_len_incr` 之**和**（不是 1 + 和）——
                // 编码器的 `write_section` 就是这么写的，多算 1 会越过 max_sfb。
                let mut len = 0usize;
                loop {
                    let inc = r.u(run_bits)?;
                    len += inc as usize;
                    if inc != max_run {
                        break;
                    }
                }
                if len == 0 {
                    return Err(Error::Bad("section 长度为 0"));
                }
                if sfb + len > ics.max_sfb as usize {
                    return Err(Error::Bad("section 越过 max_sfb"));
                }
                for _ in 0..len {
                    ics.section_cb.push(cb);
                }
                sfb += len;
            }
        }
        Ok(())
    }

    /// `scale_factor_data`：Huffman 差分，起点是 `global_gain`；短窗时跨组延续。
    fn parse_scalefactors(&self, r: &mut BitReader, ics: &mut IcsInfo) -> Result<()> {
        let total = ics.section_cb.len();
        let mut prev = i32::from(ics.global_gain);
        for sfb in 0..total {
            let cb = ics.section_cb[sfb];
            if cb == 0 {
                ics.scalefactor.push(prev as i16);
                continue;
            }
            let idx = decode_scalefactor(r)? as i32;
            let sf = prev + (idx - 60);
            if !(0..=255).contains(&sf) {
                return Err(Error::Bad("scalefactor 越界"));
            }
            ics.scalefactor.push(sf as i16);
            prev = sf;
        }
        Ok(())
    }
}

/// 谱数据的 Huffman：在 `size` 个码字里找匹配的 `(code, len)`。
fn decode_spectral(r: &mut BitReader, codes: &[u16], lens: &[u8], size: usize) -> Result<usize> {
    let mut code = 0u32;
    for len in 1..=19u32 {
        code = (code << 1) | r.u(1)?;
        for i in 0..size {
            if u32::from(lens[i]) == len && u32::from(codes[i]) == code {
                return Ok(i);
            }
        }
    }
    Err(Error::Bad("谱数据的 Huffman 码无效"))
}

/// escape 序列：**一元前缀 + `len` 位尾数**，幅值 = `(1 << len) | 尾数`。
///
/// 前缀是「`len − 4` 个 1 再一个 0」（`len` 从 4 起）。尾数只有 **`len` 位**，
/// 而幅值本身要 `len + 1` 位才写得下 —— 那个**最高位的 1 是隐含的**，
/// 所以 `len = floor(log2(幅值))`、幅值必然落在 `[2^len, 2^(len+1))`。
///
/// ⚠️ 尾数**不是补码**。第一版按补码做了符号扩展（`v − 2^len`），
/// 后果很隐蔽：**消耗的位数一模一样，位流不前移**，只有幅值大的系数解错。
/// 症状是「高频带对、低频带全错」—— 因为小系数根本不进 escape。
/// 定标错的分布这么有规律时，先怀疑「同一个字段的两种解释」，别怀疑 IMDCT。
fn read_escape(r: &mut BitReader) -> Result<i32> {
    let mut ones = 0u32;
    while r.u(1)? != 0 {
        ones += 1;
        // 幅值上限是 13 位（`escape_coef` 封在 8191）→ `len ≤ 12` → 前缀 ≤ 8 个 1。
        if ones > 8 {
            return Err(Error::Bad("escape 前缀过长"));
        }
    }
    let len = ones + 4;
    Ok((r.u(len)? as i32) | (1 << len))
}

/// 短窗的分组数：`scale_factor_grouping` 的 7 位里，1 = 「与上一个窗同组」。
fn num_window_groups(ics: &IcsInfo) -> usize {
    if ics.window_sequence != 2 {
        return 1;
    }
    let ones = ics.grouping.count_ones() as usize;
    8 - ones
}

/// 每个组里有几个窗口（长窗恒为 `[1]`）。
///
/// 分组位是 7 位、位 j 对应窗口 j+1（**1 = 并进当前组**）。这个「1 = 同组」的极性
/// 与编码端 `write_ics_info` 是同一个约定 —— 记反了组数就从 1 变成 8，
/// 症状是 `section` 立刻越过 `max_sfb`。
fn window_group_lens(ics: &IcsInfo) -> Vec<usize> {
    if ics.window_sequence != 2 {
        return vec![1];
    }
    let mut lens = vec![1usize];
    for w in 1..8 {
        if (ics.grouping >> (7 - w)) & 1 == 1 {
            *lens.last_mut().expect("lens 至少有一项") += 1;
        } else {
            lens.push(1);
        }
    }
    lens
}

/// 反量化的定标：`sf_gain(sf) = 2^((sf − 100)/4)`。
///
/// ⚠️ 那个 `− 100` 不能漏。编码端 `iq()` 的文档里记着这条踩坑史：漏掉之后
/// 相对定标全对、只有整体电平错，错得像个固定倍数 —— 看起来像调参问题。
fn sf_gain(sf: u8) -> f32 {
    ((f64::from(sf) - 100.0) * 0.25).exp2() as f32
}

/// 一个量化系数的反量化：`sign(q)·|q|^(4/3)`（`sf` 的定标由调用方乘）。
fn dequant_one(q: i32) -> f32 {
    if q == 0 {
        return 0.0;
    }
    let m = (q.unsigned_abs() as f64).powf(4.0 / 3.0) as f32;
    if q < 0 {
        -m
    } else {
        m
    }
}

/// 标度因子 Huffman：`SCALEFACTOR_CODE/BITS`（121 项）→ 索引。
fn decode_scalefactor(r: &mut BitReader) -> Result<u32> {
    let mut code: u32 = 0;
    for len in 1..=19u32 {
        code = (code << 1) | r.u(1)?;
        for (i, (&c, &l)) in t::SCALEFACTOR_CODE
            .iter()
            .zip(t::SCALEFACTOR_BITS.iter())
            .enumerate()
        {
            if u32::from(l) == len && c == code {
                return Ok(i as u32);
            }
        }
    }
    Err(Error::Bad("scalefactor 的 Huffman 码无效".into()))
}

/// 位读取器（MSB 优先）。
struct BitReader<'a> {
    d: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(d: &'a [u8]) -> Self {
        Self { d, bit: 0 }
    }
    fn pos(&self) -> usize {
        self.bit
    }
    fn u(&mut self, n: u32) -> Result<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = self.bit / 8;
            if byte >= self.d.len() {
                return Err(Error::Bad("位流提前结束".into()));
            }
            let shift = 7 - (self.bit % 8);
            v = (v << 1) | u32::from((self.d[byte] >> shift) & 1);
            self.bit += 1;
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use v2v_kernel::BitWriter;

    #[test]
    fn scalefactor_huffman_round_trips() {
        for idx in 0..121usize {
            let mut w = BitWriter::new();
            w.put(t::SCALEFACTOR_CODE[idx], u32::from(t::SCALEFACTOR_BITS[idx]));
            while w.bit_pos() % 8 != 0 {
                w.put(0, 1);
            }
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            let got = decode_scalefactor(&mut r).unwrap() as usize;
            assert_eq!(got, idx, "索引 {idx} 往返成了 {got}");
        }
    }

    /// **`CODEBOOK_VECTORS` 的值是 `trunc(|q|^(4/3))`，不是 `q`。**
    ///
    /// 这条测试是给整条解码链立的界碑：那张表来自 ffmpeg 的 `codebook_vectorN`，
    /// 里面存的是**已经开过 4/3 次方的幅值**（定点和浮点两条路共用）。
    /// 曾经把表值当 `q` 再开一次方，后果是「非 escape 系数整体放大 1.5~2.7 倍」，
    /// 而 escape 系数走另一条路、完全正确 —— 于是症状是「低频带全对、
    /// 中高频带偏大 2.5 倍」，看着像频带表或定标问题，其实是**一个数字两种含义**。
    ///
    /// 所以现在 `q` 从索引的位权数字算，这张表退居 oracle：把每个码本、每个索引
    /// 的 `q` 都按 `trunc(|q|^(4/3))` 算一遍，必须与表逐项相同（escape 位是 64）。
    #[test]
    fn codebook_vectors_match_the_power_of_q() {
        for cb in 1usize..=11 {
            let dim = usize::from(CB_DIM[cb]);
            let range = usize::from(CB_RANGE[cb]);
            let maxval = i32::from(CB_MAXVAL[cb]);
            let unsigned = CB_UNSIGNED[cb];
            let size = t::SPECTRAL_SIZE[cb - 1] as usize;
            let vecs = t::CODEBOOK_VECTORS[cb - 1];
            for idx in 0..size {
                let mut vals = [0i32; 4];
                let mut v = idx;
                for j in (0..dim).rev() {
                    let digit = (v % range) as i32;
                    v /= range;
                    vals[j] = if unsigned { digit } else { digit - maxval };
                }
                for j in 0..dim {
                    let q = vals[j];
                    // escape 位在表里是 **64**（不是 `trunc(16^(4/3)) = 40`）——
                    // 那是个标记，不是幅值。
                    let want = if cb == CB_ESC && q.unsigned_abs() == maxval as u32 {
                        64
                    } else {
                        // ⚠️ 用 `cbrt(q^4)` 而不是 `powf(4/3)`：`8^(4/3)` 数学上正好是
                        // 16，而 `powf` 给 15.9999999…，截断就成了 15 —— 表里是 16。
                        // 立方根对完全立方数是精确的，这条测试才站得住。
                        let a = q.unsigned_abs() as f64;
                        (a * a * a * a).cbrt() as i32 * if q < 0 { -1 } else { 1 }
                    };
                    assert_eq!(
                        i32::from(vecs[idx * dim + j]),
                        want,
                        "码本 {cb} 索引 {idx} 第 {j} 维：表里是 {}，而 trunc(|{q}|^(4/3)) = {want}",
                        vecs[idx * dim + j]
                    );
                }
            }
        }
    }

    /// **TNS 的压缩系数表不是 4 位表的前缀。**
    ///
    /// 四张表 `tns_tmp2_map_<coef_compress>_<coef_res>` 很容易被当成「同一张表、
    /// 压缩只是少读一位」。它们不是：`_1_4` 的第 4~7 项是 `0.6737, 0.5264,
    /// 0.3612, 0.1837`，而 `_0_4` 同位置是 `-0.7431, -0.8660, -0.9511, -0.9945`。
    ///
    /// 这条测试钉住「不能拿 4 位表当压缩表用」。那种错的症状极隐蔽：读的位数恰好
    /// 对、位流不前移、也不报错，只是 TNS 逆滤波用了错的系数 —— 只有和 ffmpeg
    /// 逐样本比才看得见（本仓自己的编码器恒写 `coef_compress = 0`）。
    #[test]
    fn compressed_tns_table_is_not_a_prefix_of_the_4bit_one() {
        let full = &t::TNS_COEF_4BIT;
        let comp = &t::TNS_COEF_4BIT_COMPRESSED;
        assert_eq!(comp.len(), 8);
        assert_eq!(comp[..4], full[..4], "前 4 项确实相同");
        assert_ne!(
            comp[4..],
            full[4..8],
            "第 4~7 项必须与 4 位表不同，否则「少读一位」就够、不需要换表"
        );

        // 3 位那两张同理（前 2 项相同、之后分叉）。
        let full3 = &t::TNS_COEF_3BIT;
        let comp3 = &t::TNS_COEF_3BIT_COMPRESSED;
        assert_eq!(full3.len(), 8);
        assert_eq!(comp3.len(), 4);
        assert_eq!(comp3[..2], full3[..2]);
        assert_ne!(comp3[2..], full3[2..4]);
    }

    /// **频带编解码往返**：编码端 `write_band` 写出 → 解码端 `decode_band` 读回 →
    /// 与 `quantize_band` 的量化结果**逐个系数**比。
    ///
    /// 这是 Huffman / 符号位 / escape 三层的**唯一**判据。它不看位流字节，看的是
    /// 「读回来的量化值等于编码端量化出来的那一组吗」—— 因为这三层里任何一层错，
    /// 症状都是「位流合法、解出来是另一组系数」：
    ///
    /// - Huffman 索引错 → 系数值错，位数可能还对得上（**位流不前移**，最难查）；
    /// - 符号位错 → 整条带的正负反了；
    /// - escape 尾数按补码解（曾经就是这样）→ 只有幅值大的系数错，
    ///   表现为「低频带全错、高频带正常」。
    ///
    /// 覆盖 1..=11 全部码本、每个码本各自的 `maxval` 边界、以及带内系数个数
    /// 不是 `dim` 整数倍（末尾那一组要补零）的情形。
    #[test]
    fn band_codec_round_trips() {
        use crate::encode::{quantize_band, write_band};
        let dec = Decoder::new(&[0x12, 0x10]).expect("44100 / 单声道 / LC");
        let mut state = 0x5eed_1234u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        for cb in 1u8..=11 {
            for &sf in &[100u8, 120, 137, 160] {
                // 幅度按 sf 反着定标：`q34(sf)` 越大 → 输入要越小才不会全部封顶
                let q34 = 2f32.powf(-3.0 * (f32::from(sf) - 100.0) / 16.0);
                let scale = 1.0 / q34;
                for &len in &[4usize, 8, 12, 20] {
                    let band: Vec<f32> = (0..len)
                        .map(|_| {
                            // 混合「接近封顶的大值」和「小值」，两种都要走一遍
                            let r = (next() >> 16) as f32 / 65536.0;
                            let mag = if r > 0.5 {
                                scale * (4.0 + r * 12.0)
                            } else {
                                scale * (0.2 + r * 2.0)
                            };
                            if next() & 1 == 0 {
                                mag
                            } else {
                                -mag
                            }
                        })
                        .collect();
                    let mut w = BitWriter::new();
                    write_band(&mut w, &band, sf, cb).unwrap();
                    let bits = w.bit_pos();
                    while w.bit_pos() % 8 != 0 {
                        w.put(0, 1);
                    }
                    let bytes = w.finish();
                    let mut r = BitReader::new(&bytes);
                    let mut got = vec![0i32; len];
                    dec.decode_band(&mut r, cb, &mut got).unwrap();
                    assert_eq!(
                        r.pos() % 8,
                        bits % 8,
                        "cb={cb} sf={sf} len={len}：读回的位数与写出的不一致"
                    );
                    let want = quantize_band(&band, sf, cb);
                    assert_eq!(
                        got, want,
                        "cb={cb} sf={sf} len={len}：读回的系数与量化结果不一致"
                    );
                }
            }
        }
    }
}
