//! AAC-LC **解码**（M1：比特流解析）。
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
//! - `section_data`（4 位码本号 + 11 的 escape）、`scale_factor_data`（Huffman）。
//!
//! ## 还没做（M2 / M3）
//!
//! - **M2**：谱数据 Huffman（11 本码本，第 11 本带 escape）+ 反量化 `|q|^(4/3)`；
//! - **M3**：TNS 逆滤波、IMDCT、窗与重叠相加；短窗的 8 段 IMDCT。
//!
//! 遇到还没做的部分**明确报 `Unsupported`**，不猜、不静默跳过。

use crate::encode::{CB_DIM, CB_ESC, CB_UNSIGNED};
use crate::{Error, Result};
use v2v_tables::aac as t;

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

    fn parse_sce(&self, r: &mut BitReader) -> Result<Element> {
        let tag = r.u(4)? as u8;
        let ch = self.parse_ics(r, false)?;
        let sp = r.pos();
        let spec = self.parse_spectral(r, &ch)?;
        Ok(Element {
            kind: 0,
            tag,
            common_window: false,
            ms_mask_present: 0,
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
                    for _ in 0..bits {
                        r.u(1)?;
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
            return Err(Error::Unsupported("TNS 暂不支持（M3）".into()));
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
            return Err(Error::Unsupported("TNS 暂不支持（M3）".into()));
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
        })
    }

    /// 谱数据：按 `section_cb` 与 `swb` 逐带解（**长窗**；短窗留到 M3）。
    fn parse_spectral(&self, r: &mut BitReader, ics: &IcsInfo) -> Result<Vec<i32>> {
        if ics.window_sequence == 2 {
            return Err(Error::Unsupported("短窗谱数据暂不支持（M3）".into()));
        }
        let sr = self.sample_rate_index as usize;
        let len = t::SWB_OFFSET_1024_LEN[sr] as usize;
        let swb = &t::SWB_OFFSET_1024[sr][..len];
        let mut out = vec![0i32; self.frame_len];
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

    /// 一个频带的谱系数（Huffman + 符号位 + escape）。
    fn decode_band(&self, r: &mut BitReader, cb: u8, out: &mut [i32]) -> Result<()> {
        if cb == 0 {
            out.fill(0);
            return Ok(());
        }
        let cb = cb as usize;
        let dim = CB_DIM[cb];
        let unsigned = CB_UNSIGNED[cb];
        let codes = &t::SPECTRAL_CODE[cb - 1];
        let lens = &t::SPECTRAL_BITS[cb - 1];
        let size = t::SPECTRAL_SIZE[cb - 1] as usize;
        let vecs = t::CODEBOOK_VECTORS[cb - 1];
        let mut pos = 0usize;
        while pos < out.len() {
            let idx = decode_spectral(r, codes, lens, size)?;
            let mut vals = [0i32; 4];
            for j in 0..dim {
                vals[j] = i32::from(vecs[idx * dim + j]);
            }
            // 无符号码本：非零维逐个跟一个符号位。
            if unsigned {
                for v in vals.iter_mut().take(dim) {
                    if *v != 0 && r.u(1)? != 0 {
                        *v = -*v;
                    }
                }
            }
            // escape 码本：|值| == 64 的那一维后面跟变长整数。
            if cb == CB_ESC {
                for v in vals.iter_mut().take(dim) {
                    if v.abs() == 64 {
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

/// escape：`nbits-1` 个 1 再一个 0（`nbits = 前缀里 1 的个数 + 1`），
/// 然后 `len = nbits + 3` 位的有符号值。
fn read_escape(r: &mut BitReader) -> Result<i32> {
    let mut ones = 0u32;
    while r.u(1)? != 0 {
        ones += 1;
        if ones > 9 {
            return Err(Error::Bad("escape 前缀过长"));
        }
    }
    let len = ones + 4;
    let v = r.u(len)? as i32;
    let sign = 1i32 << (len - 1);
    Ok(if v & sign != 0 { v - (1 << len) } else { v })
}

/// 短窗的分组数：`scale_factor_grouping` 的 7 位里，1 = 「与上一个窗同组」。
fn num_window_groups(ics: &IcsInfo) -> usize {
    if ics.window_sequence != 2 {
        return 1;
    }
    let ones = ics.grouping.count_ones() as usize;
    8 - ones
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
