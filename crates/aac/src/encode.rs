//! AAC-LC 编码器（长窗、单/双声道、无 TNS/PNS/IS/MS）。
//!
//! ## 一帧的处理顺序
//!
//! ```text
//! 1. 攒够 2N 个样本（50% 重叠），开窗（正弦），MDCT → 1024 个谱系数
//! 2. 定全局增益 scf（"公共缩放"）：先把谱系数缩到合适量级
//! 3. 逐频带：穷举该带的 sf，选「量化失真 + λ·码字位数」最小的那个
//!    同时选码本（11 本里逐本比代价）
//! 4. 写位流：ICS 头 → 全局增益 → section_data → scalefactor_data → 谱系数
//! ```
//!
//! ## 定标（最要紧的一组数）
//!
//! 来自 `aac.h`：
//!
//! ```text
//!   SCALE_DIV_512 = 36    // sf 差 36 ⟺ 幅度差 512 倍
//!   SCALE_ONE_POS = 140   // sf = 140 ⟺ 缩放 = 1.0
//!   POW_SF2_ZERO  = 200   // pow2sf 表里 pow(2,0) 的下标
//! ```
//!
//! 于是 `Q(sf) = 2^((104 - sf) / 4)`、`IQ(sf) = 1/Q(sf) = 2^((sf - 104) / 4)`。
//! 量化：`q = round_down( pow(|c|, 3/4) · Q(sf)^(3/4) + 0.4054 )`，
//! 反量化：`|c| ≈ pow(q, 4/3) · IQ(sf)`。
//!
//! ⚠️ 量化里用的是 **`Q^(3/4)`**（`ff_aac_pow34sf_tab`），不是 `IQ^(3/4)` ——
//! 两者互为倒数，抄反了量化方向就整个反了。
//!
//! ## 为什么用「逐频带穷举 sf」
//!
//! ffmpeg 走的是心理声学模型 + 拉格朗日率失真优化，还要维护「频带间 sf 差分
//! 越界」的约束（`SCALE_MAX_DIFF = 60`）。那套东西的第一版必错，而且错了很难看出来。
//!
//! 这里改用**直接穷举**：每个频带在 `[0, 255]` 里试（实际是围绕一个估计值 ±24），
//! 代价函数是 `失真 + λ·位数`（λ 是位/能量单位的换算），取最小。
//! 好处是「谁更优」这件事完全不依赖任何模型 —— 与 MPEG-4 那边的
//! 「帧内/帧间两种都写完比位数」是同一个思路。

use v2v_msmpeg4::BitWriter;

use crate::mdct::{kbd_aac_window, mdct, sine_window, LONG_LEN, SHORT_LEN};
use crate::tables;
use crate::tns::{self, Tns};
use crate::Error;

/// AAC 的窗序列（`window_sequence`，位流里占 2 位）。
///
/// ⚠️ **这四个数值是格式定义，不能改**（`aac.h` 的 `enum WindowSequence`）：
/// `ONLY_LONG = 0`、`LONG_START = 1`、`EIGHT_SHORT = 2`、`LONG_STOP = 3`。
///
/// 这里踩过一次大坑：第一版把 `EIGHT_SHORT_SEQUENCE` 当成 **1**（其实 1 是
/// `LONG_START`）。后果是解码端按「长窗」读 `max_sfb`（6 位而不是 4 位），
/// 我们写的 `1110` + 分组头两位 `00` 被读成 **56**，报的是
/// `Number of scalefactor bands in group (56) exceeds limit (49)` ——
/// **报错指向频带数，根因却是一个枚举值**。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)] // `LongStart` / `LongStop` 的值要正确，但暂时还不发（见 TODO）
pub enum WindowSequence {
    OnlyLong = 0,
    LongStart = 1,
    EightShort = 2,
    LongStop = 3,
}

/// 一个**窗口组**的编码计划：虚拟谱 + 频带表 + 选出来的逐带 sf/码本。
///
/// 长窗只有一组（组内 1 个窗口，频带 = 1024 点的划分）；短窗按分组规则拆成
/// 若干组，每组的频带表 = `组内窗口数 × 128 点的划分`（见 `encode_channel`）。
///
/// `spec` 是**带主序**的：带 0 的（组内全部窗口）→ 带 1 的 → …，
/// 与解码端 `decode_spectrum_and_dequant` 的读取顺序一致
/// （它对一个组逐带逐窗口读：`cfo = coef + 128·窗口`）。
struct GroupPlan<'a> {
    spec: &'a [f32],
    swb: &'a [u16],
    band_sf: Vec<i32>,
    band_cb: Vec<u8>,
    band_zero: Vec<bool>,
}

/// Audio Object Type：AAC-LC。
pub const AOT_AAC_LC: u8 = 2;

/// 规范允许的最大 sf 差分（`SCALE_MAX_DIFF`）。
///
/// ⚠️ 这条是**硬约束**：频带之间 sf 的跨度超过 60，位流就是非法的。
/// 而且它作用在**相邻非零频带**之间（零频带不参与差分链）。
const SCALE_MAX_DIFF: i32 = 60;

/// 频谱码本的维数（4 元组还是 2 元组）。
///
/// 来源：`aaccoder.c` 里 `QUANTIZE_AND_ENCODE_BAND_COST_FUNC` 的 `BT_PAIR` 实参，
/// `dim = BT_PAIR ? 2 : 4`。
pub const CB_DIM: [usize; 12] = [0, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2];

/// 码本是否「无符号」（无符号的**不写符号位**）。
pub const CB_UNSIGNED: [bool; 12] = [
    false, false, false, true, true, false, false, true, true, true, true, true,
];

/// 码本里量化值的上限（`aac_cb_maxval`）。
pub const CB_MAXVAL: [u8; 12] = [0, 1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 16];

/// 码本索引的进制（`aac_cb_range`）：`curidx = curidx * range + q`。
pub const CB_RANGE: [u8; 12] = [0, 3, 3, 3, 3, 9, 9, 8, 8, 13, 13, 17];

/// 只有第 11 本是 escape 码本（`ESC_BT`）。
const CB_ESC: usize = 11;

/// 编码配置。
#[derive(Debug, Clone, Copy)]
pub struct EncoderConfig {
    pub channels: u8,
    pub sample_rate: u32,
    /// 目标平均码率（bit/s）。用来定 λ —— 量化失真与码字位数之间的换算。
    ///
    /// 这个编解码器**不做**码率控制（不做反馈、不做两级循环），
    /// 所以它是「目标」而不是「保证」：实际码率会在它附近浮动。
    pub bitrate: u32,
    /// 用 KBD 窗而不是正弦窗。默认正弦（不需要 Bessel 函数，AAC 里也常用）。
    pub kb_window: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            channels: 2,
            sample_rate: 44100,
            bitrate: 128_000,
            kb_window: false,
        }
    }
}

/// 一帧 PCM。交织的 f32，范围 `[-32768, 32767]`（与 `crates/wma` 的输出同口径）。
pub struct Frame {
    /// `channels` 条，每条 1024 个样本。
    pub planes: Vec<Vec<f32>>,
}

/// AudioSpecificConfig —— 放进 MP4 的 `esds`。
///
/// 5+4+4 位 + GASpecificConfig 的 3 位，共 2 字节（采样率是标准值时）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    pub object_type: u8,
    pub sample_rate_index: u8,
    pub channels: u8,
}

impl AudioSpecificConfig {
    /// 采样率的 4 位索引（`ff_aac_sample_rate_idx` 的逆查）。
    pub fn sample_rate_index(rate: u32) -> Option<u8> {
        tables::SAMPLE_RATES
            .iter()
            .position(|&r| r == rate)
            .map(|i| i as u8)
    }

    /// 序列化。标准采样率下正好 2 字节。
    pub fn to_bytes(self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.put(u32::from(self.object_type), 5);
        w.put(u32::from(self.sample_rate_index), 4);
        w.put(u32::from(self.channels), 4);
        w.put(0, 1); // frame_length_flag = 0 → 1024
        w.put(0, 1); // depends_on_core_coder = 0
        w.put(0, 1); // extension_flag = 0
        w.finish()
    }
}

/// AAC-LC 编码器。有状态：要跨帧维护 50% 重叠的那半块样本。
pub struct Encoder {
    cfg: EncoderConfig,
    /// 上一帧的后半段（`N` 个样本），与这一帧拼成 `2N`。
    overlap: Vec<Vec<f32>>,
    /// 长窗的开窗表（`2N` 长）。
    window: Vec<f32>,
    /// 采样率索引（TNS 的几张表按它索引）。
    sr_idx: usize,
    /// 已经编过多少帧 —— 用来交替「这一帧有没有上一帧的右半」。
    frame_index: u64,
    /// 每个频带的边界（含末尾），长 `num_sfb + 1`。
    swb: Vec<u16>,
    /// **短窗**（`EIGHT_SHORT_SEQUENCE`）的频带边界与开窗表。
    ///
    /// 短窗一帧 = 8 个 128 点的 MDCT；频带划分用 `SWB_OFFSET_128`。
    swb_short: Vec<u16>,
    window_short: Vec<f32>,
    /// 每声道「上一帧的攻击落在哪个子块」（`None` = 没有攻击）。
    ///
    /// 短窗的覆盖区是 `buf[448..1600]`（见 `encode_channel`），换算到**本帧样本**
    /// 就是 `[-576, +576)`：所以一帧里的攻击落在**前 ~576 个样本**（子块 0..=4）
    /// 时本帧要用短窗；落在**后 448 个样本**（子块 5..=7）时，攻击其实落在
    /// **下一帧**的窗口覆盖区里 —— 于是本帧发 `LONG_START`、下一帧发短窗。
    ///
    /// ⚠️ 这个「前/后」的划分是**不需要前瞻**就能决定 `LONG_START` 的关键：
    /// 「下一帧会不会是短窗」由**本帧自己的后段攻击**决定，而它是本帧已知的。
    attack_prev: Vec<Option<usize>>,

    /// 每帧的**位数预算**（`bitrate / 帧率 / 声道数`）。
    ///
    /// ⚠️ 用「预算 + 二分搜偏移」而不是「λ 权重」——
    /// λ 那条路我试过：`λ ≈ 1e-4` 而失真项 `≈ 1e9`，λ 被完全淹没、等于不起作用，
    /// 实测码率 432 kbps（目标是 128k，差 3.4 倍）。
    /// 位数预算是个**硬约束**，二分搜能保证逼近它 —— 见 `encode_channel` 的说明。
    bits_per_frame: u32,
    /// 已经编过多少帧**短窗**（`EIGHT_SHORT_SEQUENCE`）。
    ///
    /// 存在的理由：短窗是**由信号内容触发**的（瞬态攻击），不是配置项 ——
    /// 光看「ffmpeg 解得开」证明不了它真的被走过（长窗也解得开）。
    /// 测试用它断言「这段瞬态素材确实进了短窗那条路」。
    short_frames: u64,
}

impl Encoder {
    pub fn new(cfg: EncoderConfig) -> crate::Result<Self> {
        if cfg.channels == 0 || cfg.channels > 2 {
            return Err(Error::Unsupported(format!(
                "声道数 {}（目前只做 1 / 2）",
                cfg.channels
            )));
        }
        let sr_idx = AudioSpecificConfig::sample_rate_index(cfg.sample_rate).ok_or_else(|| {
            Error::Unsupported(format!("采样率 {} 不是 AAC 的标准采样率", cfg.sample_rate))
        })?;
        let len = tables::SWB_OFFSET_1024_LEN[sr_idx as usize] as usize;
        let swb = tables::SWB_OFFSET_1024[sr_idx as usize][..len].to_vec();
        let len_s = tables::SWB_OFFSET_128_LEN[sr_idx as usize] as usize;
        let swb_short = tables::SWB_OFFSET_128[sr_idx as usize][..len_s].to_vec();

        let window = if cfg.kb_window {
            // 长窗 alpha = 4.0（`aactab.c` 里的 `ff_kbd_window_init(.., 4.0, 1024)`）。
            // 注意 `kbd_aac_window` 会铺成 2N 长（MDCT 的输入长度）。
            kbd_aac_window(LONG_LEN, 4.0)
        } else {
            sine_window(LONG_LEN)
        };
        // 短窗与长窗**同系**：编码端写 `use_kb_window` 是每个 ics_info 一个位，
        // 简化起见长短窗永远同系（解码端只拿它决定用哪张窗表，不校验一致性）。
        let window_short = crate::mdct::short_window(cfg.kb_window);
        // 每帧的位数预算。`bitrate` 是**目标**不是保证 —— 编码器靠
        // 「二分搜整帧 sf 偏移」逼近它（见 `encode_channel`）。
        let frames_per_sec = f64::from(cfg.sample_rate) / LONG_LEN as f64;
        let bits_per_frame =
            (f64::from(cfg.bitrate) / frames_per_sec / f64::from(cfg.channels)).max(1.0);

        Ok(Self {
            overlap: vec![vec![0f32; LONG_LEN]; cfg.channels as usize],
            window,
            sr_idx: sr_idx as usize,
            frame_index: 0,
            swb,
            swb_short,
            window_short,
            attack_prev: vec![None; cfg.channels as usize],
            bits_per_frame: bits_per_frame as u32,
            short_frames: 0,
            cfg,
        })
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    /// 这个编码器对应的 `AudioSpecificConfig`（放进 `esds`）。
    pub fn audio_specific_config(&self) -> AudioSpecificConfig {
        AudioSpecificConfig {
            object_type: AOT_AAC_LC,
            sample_rate_index: AudioSpecificConfig::sample_rate_index(self.cfg.sample_rate)
                .expect("构造时已经验过"),
            channels: self.cfg.channels,
        }
    }

    /// 频带数（`max_sfb` 的上限）。
    pub fn num_sfb(&self) -> usize {
        self.swb.len() - 1
    }

    /// 已经编过多少帧**短窗**。短窗由信号内容触发（瞬态攻击），不是配置项，
    /// 所以测试要靠它确认「瞬态素材真的进了短窗那条路」。
    pub fn short_frame_count(&self) -> u64 {
        self.short_frames
    }

    /// 编一帧。
    ///
    /// `frame.planes[i]` 长度必须是 1024（或更短 —— 短了补 0，最后一帧用得上）。
    /// 返回一个裸 AAC 帧（不含 ADTS 头，MP4 里也不需要）。
    pub fn encode_frame(&mut self, frame: &Frame) -> crate::Result<Vec<u8>> {
        if frame.planes.len() != self.cfg.channels as usize {
            return Err(Error::Bad("声道数与配置不一致"));
        }
        for p in &frame.planes {
            if p.len() > LONG_LEN {
                return Err(Error::Bad("一帧超过 1024 个样本"));
            }
        }

        let mut w = BitWriter::new();
        let is_stereo = self.cfg.channels == 2;
        let mut any_short = false;

        // ---- 元素头 ----
        // SCE = 0，CPE = 1。
        w.put(if is_stereo { 1 } else { 0 }, 3);
        w.put(0, 4); // 元素实例标签（每帧都从 0 起）
        if is_stereo {
            // `common_window = 0`：两个通道各自带自己的 ICS。
            //
            // 走 common_window 能让两通道共用一套窗与频带划分，还能开 MS/IS ——
            // 但那条路要求两个通道的 ICS **完全一致**（`ics_info` 只写一份），
            // 并且要处理 ms_mask。先把 `common_window = 0` 这条窄路走通。
            w.put(0, 1);
        }

        for ch in 0..self.cfg.channels as usize {
            if self.encode_channel(&mut w, ch, &frame.planes[ch])? {
                any_short = true;
            }
        }
        if any_short {
            self.short_frames += 1;
        }

        // ---- 结尾的 END 元素 ----
        //
        // `ID_END = 7`（3 位）。解码端的主循环一轮一轮读 `id_syn_ele`，
        // 读到 7 才收工；没有它就会一直读到缓冲区耗尽
        // （ffmpeg 报 `Input buffer exhausted before END element found`）。
        w.put(7, 3);
        // 尾字节补 0 到字节边界（`ff_aac` 的 `align_put_bits` 语义）。
        // 位流是按字节存的，不补零最后那几个位会被读成下一帧的内容。
        while w.bit_pos() % 8 != 0 {
            w.put(0, 1);
        }

        self.frame_index += 1;
        Ok(w.finish())
    }

    /// 给**一个窗口组**选逐带的 sf 与码本（**不写位流**）。
    ///
    /// 「先选、后写」是刻意的：`global_gain` 是「第一个非零带的 sf」，而它写在
    /// 最前面 —— 先选完再写就不用做位级回填。
    ///
    /// `prev_sf` 是**跨组延续**的差分链（解码端的 `offset[0]` 不按组重置），
    /// 所以用 `&mut` 带进带出：上一组最后一个非零带的 sf 是下一组的差分基准。
    fn plan_group<'a>(
        &self,
        spec: &'a [f32],
        swb: &'a [u16],
        sf_floor: i32,
        sf_offset: i32,
        prev_sf: &mut Option<i32>,
    ) -> GroupPlan<'a> {
        let max_sfb = swb.len() - 1;
        // ⚠️ `band_sf` 就是要写进位流、且**与 `global_gain` 同域**（0..255）的那个值。
        // 解码端解出来的 `sfo = band_sf − 100` —— 那个 100 只在解码端出现一次。
        let mut band_sf = vec![0i32; max_sfb];
        let mut band_cb = vec![0u8; max_sfb];
        let mut band_zero = vec![false; max_sfb];
        // 逐带的掩蔽阈值（整帧算一次，与 sf_offset 无关 —— 阈值只看源信号）。
        let thresh = masking_thresholds(spec, swb, self.cfg.sample_rate, sf_floor);

        for sfb in 0..max_sfb {
            let lo = swb[sfb] as usize;
            let hi = swb[sfb + 1] as usize;
            let band = &spec[lo..hi];
            let th = thresh[sfb];

            // 搜索窗口 = 「整帧下界 + 偏移」∩「差分约束 ±60」∩ `0..=255`。
            //
            // ⚠️ 三个都必要：下界保证峰值不被裁；差分约束是位流硬要求；
            // `0..=255` 是 `sf` 自己的位宽。**偏移只抬下界** ——
            // 若把上界也压低，会挡住「该带本来想用更粗的 sf」的合法选择。
            let floor = (sf_floor + sf_offset).clamp(0, 255);
            let (lo_sf, hi_sf) = match *prev_sf {
                Some(p) => (
                    floor.max(p - SCALE_MAX_DIFF).clamp(0, 255),
                    (p + SCALE_MAX_DIFF).clamp(0, 255),
                ),
                None => (floor, 255),
            };
            if lo_sf > hi_sf {
                // 窗口空了（极少见）：退化成「贴着上一个带」，至少位流合法。
                let sf = prev_sf.unwrap_or(floor).clamp(0, 255);
                band_sf[sfb] = sf;
                band_cb[sfb] = 0;
                band_zero[sfb] = true;
                *prev_sf = None;
                continue;
            }

            // 逐带找「失真最小」的 sf（λ 项已去掉 —— 码率由外层二分控，
            // 代价函数只管保真）。
            let mut best: Option<(f32, u8, u8)> = None; // (代价, sf, cb)
            let step = 4.max((hi_sf - lo_sf) / 24);
            let mut cand: Vec<i32> = Vec::new();
            let mut sd = lo_sf;
            while sd <= hi_sf {
                cand.push(sd);
                sd += step;
            }
            for &sf in &cand {
                let (cost, cb) = band_cost(band, sf as u8, th);
                if best.is_none_or(|(bc, _, _)| cost < bc) {
                    best = Some((cost, sf as u8, cb));
                }
            }
            // 在最优附近再细扫 ±3（步长 1），避免粗扫步长漏掉最优点。
            if let Some((_, sf0, _)) = best {
                for sf in (i32::from(sf0) - 3).max(lo_sf)..=(i32::from(sf0) + 3).min(hi_sf) {
                    let (cost, cb) = band_cost(band, sf as u8, th);
                    if best.is_none_or(|(bc, _, _)| cost < bc) {
                        best = Some((cost, sf as u8, cb));
                    }
                }
            }

            let (_, sf, cb) = best.expect("候选集非空");
            band_sf[sfb] = i32::from(sf);
            band_cb[sfb] = cb;
            band_zero[sfb] = cb == 0 || band_is_zero(band, sf);
            if !band_zero[sfb] {
                *prev_sf = Some(i32::from(sf));
            }
        }

        GroupPlan {
            spec,
            swb,
            band_sf,
            band_cb,
            band_zero,
        }
    }

    /// 把**所有窗口组**的位流写全。
    ///
    /// ```text
    /// global_gain → ics_info → [各组的 section_data] → [各组的 scalefactor_data]
    ///             → pulse/tns/gain → [各组的谱系数]
    /// ```
    ///
    /// ⚠️ 顺序是规范定的：**section 与 scalefactor 都是「先把所有组写完」**，
    /// 谱数据再按组写 —— 不是「一组一段」地交错。写错了解码端会从第二组开始错位。
    fn write_pass(
        &self,
        seq: WindowSequence,
        groups: &[GroupPlan<'_>],
        group_lens: &[usize],
        tns_data: Option<&Tns>,
    ) -> crate::Result<(BitWriter, u32)> {
        // `global_gain` = 第一个非零带的 sf（全零帧用 100）。
        let global_gain: u8 = groups
            .iter()
            .flat_map(|g| g.band_sf.iter().zip(&g.band_zero))
            .find(|(_, z)| !**z)
            .map_or(100, |(sf, _)| (*sf).clamp(0, 255) as u8);

        let max_sfb = groups[0].band_sf.len();
        let run_bits = if seq == WindowSequence::EightShort { 3 } else { 5 };

        let mut w = BitWriter::new();
        w.put(u32::from(global_gain), 8);
        write_ics_info(&mut w, max_sfb as u8, seq, group_lens);
        for g in groups {
            write_sections(&mut w, &g.band_cb, &g.band_zero, run_bits);
        }
        let mut prev_sf = i32::from(global_gain);
        for g in groups {
            write_scalefactors(&mut w, &g.band_sf, &g.band_zero, &mut prev_sf);
        }

        w.put(0, 1); // pulse：不做
        write_tns(&mut w, tns_data);
        w.put(0, 1); // gain control：不做

        for g in groups {
            for sfb in 0..max_sfb {
                if g.band_zero[sfb] {
                    continue;
                }
                let lo = g.swb[sfb] as usize;
                let hi = g.swb[sfb + 1] as usize;
                write_band(
                    &mut w,
                    &g.spec[lo..hi],
                    g.band_sf[sfb].clamp(0, 255) as u8,
                    g.band_cb[sfb],
                )?;
            }
        }

        let bits = w.bit_pos() as u32;
        Ok((w, bits))
    }

    /// **编一「趟」**：给定整帧 `sf_offset`，给每个窗口组选 sf/码本、写出完整的
    /// 通道元素体，返回 `(位流, 位数)`。
    ///
    /// ## 为什么要有「趟」这个概念
    ///
    /// `bitrate` 是个**硬预算**，而 sf 与位数是单调关系：
    /// **偏移越大 → 量化越粗 → 位数越少、失真越大**。
    /// 于是外层（`encode_channel`）二分搜「位数刚好不超过预算的最小偏移」，
    /// 就是在「码率不超标」的前提下拿到最好的音质。
    ///
    /// ⚠️ 这条路的**每一趟都必须自己把位流写全**，不能只算位数 ——
    /// 因为选中的那一趟要直接作为输出。只算位数、最后再编一次的做法，
    /// 会让「算出来的位数」与「真正写出的位数」在两个不同的代码路径上，
    /// 一旦不一致（我踩过：`band_cost` 与 `write_band` 的 escape 计数写法不同），
    /// 预算就形同虚设。**同一条路径既算又写**是这里刻意选的。
    ///
    /// `group_specs` 每项是 `(该组的虚拟谱, 该组的频带表, 组内窗口数)`。
    /// 长窗只有一项（`self.swb` + `self.num_sfb()` 个带）；短窗按分组规则
    /// 拆成若干项，每项的频带表 = `组内窗口数 × 128 点的划分`。
    fn encode_pass(
        &self,
        seq: WindowSequence,
        group_specs: &[(&[f32], &[u16], usize)],
        sf_floor: i32,
        sf_offset: i32,
        tns_data: Option<&Tns>,
    ) -> crate::Result<(BitWriter, u32)> {
        let mut prev_sf: Option<i32> = None;
        let mut plans: Vec<GroupPlan<'_>> = Vec::with_capacity(group_specs.len());
        for &(spec, swb, _) in group_specs {
            plans.push(self.plan_group(spec, swb, sf_floor, sf_offset, &mut prev_sf));
        }
        let lens: Vec<usize> = group_specs.iter().map(|&(_, _, l)| l).collect();
        self.write_pass(seq, &plans, &lens, tns_data)
    }


    /// 一个通道元素（`single_channel_element`）的完整编码。
    ///
    /// ## 结构
    ///
    /// ```text
    /// 1. 攒 2N 个样本 → 开窗 → MDCT
    /// 2. 算什么都不裁的 sf 下界（sf_floor）
    /// 3. 写元素头
    /// 4. 二分搜「位数 ≤ 预算」的最小 sf 偏移，取那一趟的位流
    /// ```
    ///
    /// ## 为什么第 3 步和第 4 步是分开写的
    ///
    /// `encode_pass` 每次调用**都从零编出一条完整的通道元素位流**
    /// （从 `global_gain` 开始），不含元素头。所以这里的写法是：
    /// **先把元素头写进正式位流，再把二分选中的那一趟整段接上去**。
    ///
    /// 换句话说 `encode_pass` 的「趟」= 通道元素体，元素头归这里管。
    /// 这个分界是刻意的：元素头（`ID_SCE` + 标签）跟 `sf_offset` 无关，
    /// 让每趟都带一遍只是白算计数的次数。
    fn encode_channel(
        &mut self,
        w: &mut BitWriter,
        ch: usize,
        samples: &[f32],
    ) -> crate::Result<bool> {
        // ---- 1. 攒 2N 个样本 → 开窗 → MDCT ----
        let mut buf = vec![0f32; 2 * LONG_LEN];
        // 左半 = 上一帧的后半（AAC 是 50% 重叠）
        buf[..LONG_LEN].copy_from_slice(&self.overlap[ch]);
        // 右半 = 这一帧（不足补 0）
        let n = samples.len().min(LONG_LEN);
        buf[LONG_LEN..LONG_LEN + n].copy_from_slice(&samples[..n]);
        // 存下这一帧，作为下一帧的左半
        self.overlap[ch][..n].copy_from_slice(&samples[..n]);
        for v in &mut self.overlap[ch][n..] {
            *v = 0.0;
        }

        // ---- 窗序列 ----
        //
        // 攻击打在这一帧 → 本帧用短窗；**下一帧也要用短窗**（攻击的能量在
        // 50% 重叠区里会跨进下一帧的前半 —— 长窗会把攻击「摊平」成一帧的
        // pre-echo，这正是短窗要解决的东西）。
        let attack = detect_attack(samples);
        let prev_attack = self.attack_prev[ch];
        let short = attack.is_some() || prev_attack.is_some();
        self.attack_prev[ch] = attack;
        let seq = if short {
            WindowSequence::EightShort
        } else {
            WindowSequence::OnlyLong
        };

        // 短窗的**分组**：一个组内的窗口共用一套 sf/码本 —— 把攻击所在的窗口
        // 单独分出来，组里其余窗口才不会被攻击的尺度「带跑」。
        //
        // 分组规则照 3GPP TS26.403 5.4.1（ffmpeg 的 `aacpsy.c` 用同一张表）：
        // 表按**攻击所在的子块序号**（0..8，8 = 本帧没有攻击）查 8 位的掩码，
        // 位 i = 0 表示「窗口 i 另起一组」。
        //
        // ⚠️ 实测这一条值 5~7 dB：第一版「8 个窗口一组」在瞬态素材上
        // SNR 10.2 dB（ffmpeg 自己的 AAC 是 17.8 dB），分组后追上来一大截。
        let attack_pos = attack.or(prev_attack).unwrap_or(8);
        let group_lens: Vec<usize> = if short {
            group_lengths(WINDOW_GROUPING[attack_pos])
        } else {
            vec![1]
        };

        // 各组的虚拟谱（**带主序**）与频带表。长窗只有一组，与短窗共用同一套
        // 选 sf / 选码本 / 写位流的机制（见 `GroupPlan`）。
        let mut specs: Vec<Vec<f32>> = Vec::new();
        let mut swbs: Vec<Vec<u16>> = Vec::new();
        if short {
            // 8 个短窗：窗口 w 吃 `buf[448 + w·128 .. 448 + w·128 + 256]`
            // （相邻 50% 重叠），各产 128 个系数。
            //
            // ⚠️ **那个 448 不是可选的**：`EIGHT_SHORT_SEQUENCE` 的 8 个窗
            // 一共只覆盖 1152 个样本（7×128 + 256），在 2048 的窗里是**居中**的
            // —— 两端各留 448 个样本的零填充。ffmpeg 的合成端就是这么摆的
            // （`out + 448 + i*128`，见 `aacdec_dsp_template.c` 的
            // `vector_fmul_window(out + 448 + 0*128, ...)` 那一串）。
            //
            // 第一版按 `w·128` 铺（漏了 448），后果是**攻击点落在覆盖区之外**：
            // 帧内 512 处的攻击在 `buf[1024+512] = buf[1536]`，而窗口只铺到
            // `896+256 = 1152` —— 编码器「看见了攻击、选了短窗」，
            // 却把攻击本身漏在窗外，位流还完全合法（所以只有真素材才暴露）。
            const SHORT_PAD: usize = 448;
            let mut s8 = vec![0f32; LONG_LEN];
            for w in 0..8 {
                let off = SHORT_PAD + w * SHORT_LEN;
                let mut win = vec![0f32; 2 * SHORT_LEN];
                for (k, v) in win.iter_mut().enumerate() {
                    *v = buf[off + k] * self.window_short[k];
                }
                mdct(&win, &mut s8[w * SHORT_LEN..(w + 1) * SHORT_LEN]);
            }
            // 位流按「组 → 带 → 组内窗口」排（解码端 `decode_spectrum_and_dequant`
            // 对一个组逐带逐窗口读：`cfo = coef + 128·窗口`）。把每组、每条带、
            // 组内各窗口的系数**重排成带主序**：频带 b 的数据 = 组内各窗口这条带的
            // 拼接 —— 于是选 sf / 选码本 / 写位流的机制与长窗**完全共用**。
            let widths: Vec<usize> = self.swb_short[..self.swb_short.len() - 1]
                .iter()
                .zip(&self.swb_short[1..])
                .map(|(a, b)| (b - a) as usize)
                .collect();
            let mut w0 = 0usize;
            for &gl in &group_lens {
                let mut v: Vec<f32> = Vec::with_capacity(gl * SHORT_LEN);
                for (b, &width) in widths.iter().enumerate() {
                    let lo = self.swb_short[b] as usize;
                    for j in 0..gl {
                        let w = w0 + j;
                        v.extend_from_slice(&s8[w * SHORT_LEN + lo..w * SHORT_LEN + lo + width]);
                    }
                }
                debug_assert_eq!(v.len(), gl * SHORT_LEN);
                specs.push(v);
                // 该组的频带边界 = `组内窗口数 × 128 点的划分`（末项 = 128·gl）。
                swbs.push(
                    self.swb_short
                        .iter()
                        .map(|&o| o * gl as u16)
                        .collect::<Vec<u16>>(),
                );
                w0 += gl;
            }
            debug_assert_eq!(w0, 8);
        } else {
            for (b, wv) in buf.iter_mut().zip(self.window.iter()) {
                *b *= *wv;
            }
            let mut s = vec![0f32; LONG_LEN];
            mdct(&buf, &mut s);
            specs.push(s);
            swbs.push(self.swb.clone());
        }

        // ---- TNS（只做长窗）----
        //
        // 短窗已经在时间上把瞬态切开了，TNS 的收益主要落在**长窗**帧上
        // （音乐里绝大多数帧），所以先只做长窗 —— 短窗那条路的谱是
        // 「组→带→组内窗口」重排过的，要做也得先把布局还原回去。
        let tns_data = if short {
            None
        } else {
            let t = tns::search(&specs[0], &swbs[0], self.sr_idx, self.num_sfb());
            if let Some(t) = &t {
                tns::apply(&mut specs[0], t, &swbs[0], self.sr_idx, self.num_sfb());
            }
            t
        };

        // ---- 2. sf 的下界：不把峰值系数裁到 MAX_QUANT 之上 ----
        // 取各组的**最严**下界（任何一组被裁都是失真）。
        let sf_floor = specs
            .iter()
            .zip(&swbs)
            .map(|(s, t)| min_sf_for_peak(s, t))
            .max()
            .unwrap_or(0);

        // ⚠️ **这里不写元素头** —— `ID_SCE`/`ID_CPE` 与标签由 `encode_frame` 写。
        // 一开始在这里又写了一遍（3+4 位），整条位流错位 7 位，
        // ffmpeg 报的是「scalefactor bands 56 > 49」这种看着像频带表错的信息。
        // **症状与根因隔得很远**，所以这条注释留在这里。

        let group_specs: Vec<(&[f32], &[u16], usize)> = specs
            .iter()
            .zip(&swbs)
            .zip(&group_lens)
            .map(|((s, t), &l)| (s.as_slice(), t.as_slice(), l))
            .collect();

        // ---- 3. 二分搜「位数 ≤ 预算」的最小偏移 ----
        //
        // 偏移越大 → 量化越粗 → 位数越少、失真越大，所以「位数」关于偏移单调。
        // 二分出的这个最小偏移就是「码率不超标的前提下保真最好」的那一档。
        let budget = self.bits_per_frame;
        let mut lo = 0i32;
        // 上界 120：够粗了（偏移每 +4 让量化精度翻一倍；+120 相当于把每带
        // 的 sf 抬到几乎全零带）。再往上位数不再降（全零带位数是常数）。
        let mut hi = 120i32;
        let mut chosen: Option<BitWriter> = None;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let (pass, bits) = self.encode_pass(seq, &group_specs, sf_floor, mid, tns_data.as_ref())?;
            if bits <= budget {
                chosen = Some(pass);
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        // ⚠️ **一趟都不够省**时（预算小到连最粗的量化都塞不下）退化成
        // 「用偏移上界那一趟」：位数超预算，但**位流仍然合法**。
        // 让码率超标比「编不出东西」好 —— 而且这种情况只会在极端配置下出现。
        let pass = match chosen {
            Some(p) => p,
            None => self.encode_pass(seq, &group_specs, sf_floor, hi.max(0), tns_data.as_ref())?.0,
        };
        w.append(&pass);
        Ok(short)
    }
}

/// **瞬态攻击检测**：返回攻击落在哪个子块（`None` = 没有攻击）。
///
/// 把一帧 1024 个样本切成 8 段（每段 128，与短窗的时间分辨率一致），
/// 算各段能量；某一段的能量比前一段**暴涨**（≥8 倍）且绝对量够大，就算攻击，
/// 返回那一段的下标。位置要拿去查 `WINDOW_GROUPING`（分组表按攻击位置索引）。
///
/// ## 两个参数为什么这么定
///
/// - **比值 8**：太敏感会把正常音符的开头当攻击（短窗高频分辨率差、
///   稳态信号画质反而掉）；太钝又抓不到打击乐。8× ≈ 3 个二的幂，
///   是「下一子块比这一子块响得多」的下限。
/// - **绝对下限**：能量阈值取「每样本 RMS ≈ 90」的整段能量
///   （128·90² ≈ 1e6）—— 静音段之间 0 比 0 的「暴涨」不算数。
fn detect_attack(samples: &[f32]) -> Option<usize> {
    const SEG: usize = SHORT_LEN; // 128
    const RATIO: f64 = 8.0;
    const FLOOR: f64 = 1e6;

    let mut prev = 0f64;
    let mut have_prev = false;
    for (i, seg) in samples.chunks(SEG).enumerate() {
        let e: f64 = seg.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        if have_prev && e > FLOOR && e > RATIO * prev {
            return Some(i.min(8));
        }
        prev = e;
        have_prev = true;
    }
    None
}

/// 短窗的**窗口分组表**（3GPP TS26.403 5.4.1，与 ffmpeg 的 `aacpsy.c` 同一张）。
///
/// 按「攻击所在的子块序号」索引（0..8；8 = 本帧没有攻击，表里最后两项相同）。
/// 掩码的位 i = 0 表示**窗口 i 另起一组**；位 i = 1 表示并进当前组。
/// 一个组内的窗口共用一套 sf 与码本。
const WINDOW_GROUPING: [u8; 9] = [
    0xB6, 0x6C, 0xD8, 0xB2, 0x66, 0xC6, 0x96, 0x36, 0x36,
];

/// 把 `WINDOW_GROUPING` 的掩码翻成每组的窗口数（和 = 8）。
fn group_lengths(mask: u8) -> Vec<usize> {
    let mut lens = vec![1usize];
    for i in 1..8 {
        if (mask >> i) & 1 == 0 {
            lens.push(1);
        } else {
            *lens.last_mut().expect("至少一组") += 1;
        }
    }
    lens
}

/// 频谱里每个频带「不把最响的系数量化到超过 `MAX_QUANT`」所需的最小 sf。
///
/// 抄自 `rusty_aac/src/encode.rs::min_base`：
///
/// ```text
///   q = pow(max|x|, 0.75) · 2^(-0.1875·(sf − 100)) <= MAX_QUANT
///   ⟹  log2(maxp) − 0.1875·(sf − 100) <= log2(MAX_QUANT)
///   ⟹  sf >= 100 − log2(MAX_QUANT / maxp) / 0.1875
/// ```
///
/// 这就是**搜索的下界**：比它还小的 sf 会把系数裁到 `MAX_QUANT` 上、失真骤增。
/// 上界是 255（sf 域本身就是 8 位）。
///
/// ⚠️ 之前我一直试图「由峰值反推一个 global_gain」，那是错的方向 ——
/// `global_gain` 不是独立参数，它就是**第一个非零频带的 sf**
/// （`rusty_aac::global_gain`：`cbs.iter().position(..).map_or(100, |sfb| sf[sfb])`），
/// 而整帧的 sf 由「码率循环」搜出来。见 `min_sf_for_peak` 的调用处。
fn min_sf_for_peak(spec: &[f32], swb: &[u16]) -> i32 {
    const MAX_QUANT: f64 = 8191.0;
    let mut floor = 0i32;
    for sfb in 0..swb.len() - 1 {
        let band = &spec[swb[sfb] as usize..swb[sfb + 1] as usize];
        let maxp = band
            .iter()
            .fold(0f64, |m, &c| m.max(f64::from(c.abs()).powf(0.75)));
        if maxp <= 1e-9 {
            continue;
        }
        let min_sf = (100.0 - (MAX_QUANT / maxp).log2() / 0.1875).ceil() as i32;
        floor = floor.max(min_sf);
    }
    floor.clamp(0, 255)
}

/// 量化用的缩放因子 `Q34(sf)`，让
///
/// ```text
///   q = pow(|coef|, 3/4) · Q34(sf)
/// ```
///
/// 与解码端的反量化严格互逆。解码端是（`rusty_aac/src/dsp.rs` 与 ffmpeg 一致）：
///
/// ```text
///   dequant(q)  = sign(q) · |q|^(4/3)
///   sf_gain(sf) = 2^(0.25 · (sf − 100))
///   coef        = dequant(q) · sf_gain(sf)
/// ```
///
/// 反推编码端：`q = (|coef| / sf_gain)^(3/4) = |coef|^(3/4) · sf_gain^(-3/4)`，于是
///
/// ```text
///   Q34(sf) = 2^( -3·(sf − 100) / 16 )
///   IQ(sf)  = 1 / Q34^(4/3) = 2^( (sf − 100) / 4 )     ← 就是 sf_gain
/// ```
///
/// ## ⚠️ 那个 `- 100` 是踩过坑的
///
/// 我最初把 `sf_gain` 写成 `2^(sf/4)`（**少了 `-100` 那一项**）。
/// 后果很有迷惑性：**相对定标全对**（频带之间的高低关系正确、频谱形状也对），
/// **只有整体电平错**，而且错得像个固定倍数（实测输出只有输入的 0.52 倍、
/// 再调又变成 0.013 倍）—— 因为每挪动 `sf` 的搜索窗口，那个常数就换个值。
///
/// 找了很久：先怀疑 λ 权重、再怀疑 `global_gain` 与差分基准、又怀疑
/// `section_data` 的长度转义……真正的原因就是这一个 `- 100`。
/// **「形状对、整体差一个常数」时，先把定标公式逐字对着一个已知正确的实现抄一遍。**
fn q34(sf: u8) -> f32 {
    let e = -3.0 * (f32::from(sf) - 100.0) / 16.0;
    (e * std::f32::consts::LN_2).exp()
}

/// `IQ(sf) = 2^((sf − 100)/4)` —— 反量化用的缩放因子。
///
/// 就是解码端的 `sf_gain`。与 `q34` 的关系是 `IQ = Q34^(-4/3)`。
fn iq(sf: u8) -> f32 {
    let e = (f32::from(sf) - 100.0) / 4.0;
    (e * std::f32::consts::LN_2).exp()
}

/// 量化一个非负值：`min((int)(pow(a, 3/4) · q34 + 0.4054), maxval)`。
///
/// ⚠️ 是**截断**不是四舍五入 —— 那个 `+0.4054` 是 ffmpeg 调出来的偏置
/// （`ROUND_STANDARD`），用来补偿 `pow(·, 3/4)` 的均值偏移。
/// 换成 `round()` 会让所有系数偏大一点。
fn quant_one(a: f32, q34: f32, maxval: i32) -> i32 {
    let a34 = (a as f64).powf(0.75);
    // `rusty_aac::quantize_band_scalar`：`(pow * scale).round().min(MAX_QUANT)`。
    // 用 `round()` 而不是 ffmpeg 的 `+0.4054` 截断 —— 两者都对，
    // 但**必须与反量化成对**，混用会在低频小系数上积出可见的偏差。
    let q = (a34 * f64::from(q34)).round() as i32;
    q.min(maxval)
}

/// 反量化：`q^(4/3) · IQ(sf)` —— 与解码端的 `dequant(q) · sf_gain(sf)` 一致。
fn dequant_one(q: i32, sf: u8) -> f64 {
    (q as f64).powf(4.0 / 3.0) * f64::from(iq(sf))
}

/// 频带是不是「全零带」：按 `sf` 量化之后每个系数都是 0。
///
/// `sf` 就是量化参数本身（0..255 的 `sf` 域，与 `rusty_aac` 一致）。
fn band_is_zero(band: &[f32], sf: u8) -> bool {
    let q34v = q34(sf);
    band.iter().all(|&c| quant_one(c.abs(), q34v, 16) == 0)
}

/// 先算好每带的 Bark 中心频率 —— 「邻近带」是按 Bark 距离算的，
/// 不是按频带序号。低频的带很密（几十 Hz 一个）、高频很疏
/// （一个带跨几千 Hz），按序号算会把高频的掩蔽扩散弄得过宽。
fn bark_of(swb: &[u16], sample_rate: u32) -> Vec<f32> {
    swb.windows(2)
        .map(|w| {
            // 带的中心频率（Hz）。
            let fc = (f64::from(w[0]) + f64::from(w[1])) * 0.5 * f64::from(sample_rate)
                / (2.0 * LONG_LEN as f64);
            // Bark 尺度的常用近似（Traunmüller 1990）：
            //   z = 26.81·f / (1960 + f) − 0.53
            (26.81 * fc / (1960.0 + fc) - 0.53) as f32
        })
        .collect()
}

/// 逐带的**掩蔽阈值**（相对值，单位与谱系数的平方一致）。
///
/// ## 模型
///
/// ```text
///   扩展函数（掩蔽的扩散）：
///     低频掩蔽高频强、高频掩蔽低频弱 —— 这是听觉的实测特性
///       slope = 低频侧 27 dB/Bark、高频侧 −24 dB/Bark（Schroeder 等的经典值）
///     扩散量 = Σ_b 能量[b] · 10^(-slope·|zb − z| / 10)
///
///   阈值：
///     thresh[sfb] = 本带能量 + 扩散来的部分，再夹一个下限
/// ```
///
/// 为什么需要它：**逐带独立地「比原始平方误差」是错的度量** ——
/// 它把「一个响带旁边的高频误差」和「安静段里的误差」同等看待。
/// 前者听不见（被掩蔽了），后者很刺耳。按阈值归一化之后，
/// 代价函数就能把位数花在**听得见**的地方。
///
/// ⚠️ 这是个**简化模型**：不做 FFT 分析（直接用谱系数当掩蔽源）、
/// 不含绝对听阈（安静段的阈值会偏乐观）。但它抓住了主要矛盾 ——
/// 「强带掩蔽邻近带」。
fn masking_thresholds(spec: &[f32], swb: &[u16], sample_rate: u32, _sf_floor: i32) -> Vec<f32> {
    // `_sf_floor`（整帧的 sf 下界）在这里还用不上 —— 阈值只看源信号。
    // 留着签名是为了「将来把阈值抬到量化噪声下限」那条路不用改调用方。
    let n = swb.len() - 1;
    let bark = bark_of(swb, sample_rate);

    // 每带的能量（谱系数的平方和）。
    let energy: Vec<f64> = (0..n)
        .map(|sfb| {
            let (lo, hi) = (swb[sfb] as usize, swb[sfb + 1] as usize);
            spec[lo..hi]
                .iter()
                .map(|&c| f64::from(c) * f64::from(c))
                .sum()
        })
        .collect();

    // 全帧的能量尺度 —— 用来定「阈值下限」。没有下限的话，
    // 安静带的阈值趋近 0，代价会被放大到不合理的程度。
    let total: f64 = energy.iter().sum();
    let floor = (total / n as f64) * 1e-4 + 1e-12;

    (0..n)
        .map(|sfb| {
            let mut spread = 0f64;
            for (b, &e) in energy.iter().enumerate() {
                let dz = f64::from(bark[b] - bark[sfb]);
                // 扩展函数：dz > 0 表示「b 在 sfb 的高频侧」，
                // 那时用 24 dB/Bark（高频掩蔽低频弱）；
                // dz <= 0 表示 b 在低频侧，用 27 dB/Bark（低频掩蔽高频强）。
                let slope = if dz > 0.0 { -24.0 } else { 27.0 };
                spread += e * 10f64.powf(slope * dz.abs() / 10.0);
            }
            (0.5 * energy[sfb] + spread).max(floor) as f32
        })
        .collect()
}

/// 一个频带在给定 `sf` 下的 `(代价, 最优码本)`。
///
/// ## 代价是**掩蔽归一化**的失真
///
/// ```text
///   代价 = Σ (误差² / 阈值²)
/// ```
///
/// 除以阈值让「听不见的误差」几乎不花代价、而「安静段里的误差」被放大。
/// 阈值来自 `masking_thresholds`（邻带的掩蔽扩散）。
///
/// ## ⚠️ 这里**故意没有 λ · 位数那一项**
///
/// 码率由外层的「二分搜 sf 偏移」用**硬预算**控制。再在这里掺一个软权重
/// 只会互相干扰，而且 λ 的量级（1e-4）在失真（1e9）面前根本不起作用
/// —— 实测踩过，白忙一场（码率是目标的 3.4 倍）。
///
/// **两者是正交的分工**：外层管「花多少位」，这里管「位花在哪」。
fn band_cost(band: &[f32], sf: u8, threshold: f32) -> (f32, u8) {
    // 归一化因子：1/阈值²。夹一个下限防除 0（阈值已经在下游夹过，这里是二次保险）。
    let inv_th = 1.0 / threshold.max(1e-12);

    let mut best: Option<(f32, u8)> = None;
    for cb in 1..=11usize {
        let (_bits, dist) = band_code(band, sf, cb);
        // `band_code` 给的是原始平方误差；乘上 `1/阈值²` 就是归一化代价。
        let cost = dist * inv_th * inv_th;
        if best.is_none_or(|(bc, _)| cost < bc) {
            best = Some((cost, cb as u8));
        }
    }
    // cb == 0 是「全零带」的伪码本，作为兜底参与比较。
    let zero_dist: f64 = band.iter().map(|&c| f64::from(c) * f64::from(c)).sum();
    let zero_cost = zero_dist as f32 * inv_th * inv_th;
    match best {
        Some((c, cb)) if c <= zero_cost => (c, cb),
        _ => (zero_cost, 0),
    }
}

/// 用一个具体码本量化 + 编码一个频带，返回 `(位数, 平方失真)`。
///
/// 这是**唯一**决定「解出来是什么」的地方 —— 编码和失真必须用同一套量化值，
/// 否则「代价最小」选出来的那个 sf 解出来不是你以为的东西。
fn band_code(band: &[f32], sf: u8, cb: usize) -> (u32, f32) {
    let dim = CB_DIM[cb];
    let maxval = i32::from(CB_MAXVAL[cb]);
    let q34v = q34(sf);
    let unsigned = CB_UNSIGNED[cb];
    let bit_tab = &tables::SPECTRAL_BITS[cb - 1];
    let size = tables::SPECTRAL_SIZE[cb - 1] as usize;
    let vecs = tables::CODEBOOK_VECTORS[cb - 1];

    let mut bits = 0u32;
    let mut dist = 0f32;

    for chunk in band.chunks(dim) {
        // 量化这一组。
        let mut q = [0i32; 4];
        for (j, &c) in chunk.iter().enumerate() {
            let a = if unsigned { c.abs() } else { c };
            let m = quant_one(a.abs(), q34v, maxval);
            q[j] = if unsigned || a >= 0.0 { m } else { -m };
        }
        // 索引：`curidx = curidx * range + (q + off)`。
        let off = if unsigned { 0 } else { maxval };
        let mut curidx = 0usize;
        for j in 0..dim {
            curidx = curidx * usize::from(CB_RANGE[cb]) + (q[j] + off) as usize;
        }
        if curidx >= size {
            // 索引越界说明这一组量化超了码本能表示的范围 —— 用一个「必然不优」
            // 的极大代价把它排除掉（而不是 panic：穷举 sf 时会大量出现这种情况）。
            return (u32::MAX / 2, f32::MAX / 2.0);
        }

        let mut nbits = u32::from(bit_tab[curidx]);
        // 无符号码本要**逐维**写符号位（非零才写）。
        if unsigned {
            for j in 0..dim {
                if vecs[curidx * dim + j] != 0 {
                    nbits += 1;
                }
            }
        }
        // escape 码本：矢量里 `64` 的位置要额外写。
        if cb == CB_ESC {
            for j in 0..dim {
                if vecs[curidx * dim + j] == 64 {
                    let coef = escape_coef(chunk[j].abs(), q34v);
                    nbits += escape_bits(coef);
                }
            }
        }
        bits += nbits;

        // 失真：用码本矢量反量化后的值与原值比。
        for j in 0..dim {
            let v = f64::from(vecs[curidx * dim + j]);
            let rec = if v == 64.0 {
                // escape：真实值来自那 13 位。
                let coef = escape_coef(chunk[j].abs(), q34v);
                dequant_one(coef, sf)
            } else {
                let a = v.abs();
                let sign = if chunk[j] < 0.0 { -1.0 } else { 1.0 };
                a * sign * f64::from(iq(sf))
            };
            let d = f64::from(chunk[j]) - rec;
            dist += (d * d) as f32;
        }
    }
    (bits, dist)
}

/// escape 值的量化 —— 照 `aaccoder.c`：
/// `av_clip_uintp2(quant(|c|), 13)`。
///
/// ⚠️ `av_clip_uintp2(x, 13)` 的上限是 **`2^13 - 1 = 8191`，不是 8192**。
/// 差这一个值就会让 `len = av_log2(coef)` 从 12 跳到 13，
/// 解码端算出前导 1 有 9 个 > 8，直接
/// `error in spectral data, ESC overflow` —— 而且**只在幅度大的素材上发作**，
/// 纯正弦那种稀疏频谱测不出来。实测在真样本整段转换上撞到的就是这个。
fn escape_coef(a: f32, q34: f32) -> i32 {
    quant_one(a, q34, (1 << 13) - 1).min((1 << 13) - 1)
}

/// escape 序列占多少位（照 `aaccoder.c` 的 `put_bits` 数）。
///
/// `len = av_log2(coef)`；两个字段：`len - 4 + 1` 位的长度前缀 + `len` 位的值。
///
/// ⚠️ **没有 `len < 5` 的特例** —— ffmpeg 的编码端就是直算 `len - 4 + 1`。
/// 我一开始给它加了个「`len <= 4` 就只写一个 0 位」的分支，那是**错的**：
/// 解码端读到 0 个前导 1 会 `b += 4` 变成 4，然后去读 4 位低位，
/// 而编码端只写了 1 位 —— 整条位流从这里开始错位。
fn escape_bits(coef: i32) -> u32 {
    let len = av_log2(coef);
    debug_assert!(len >= 4, "escape 的 coef 至少 16（{coef}）");
    (len - 4 + 1) + len
}

/// 写一个 escape 序列。
fn write_escape(w: &mut BitWriter, coef: i32) {
    let len = av_log2(coef);
    debug_assert!(
        len >= 4 && len <= 12,
        "escape 的 len 必须在 4..=12（{len}）"
    );
    let nbits = len - 4 + 1;
    // 长度前缀：`nbits` 位、值是 `(1 << nbits) - 2`（一串 1 带个尾零）。
    w.put((1u32 << nbits) - 2, nbits);
    // 然后是 `len` 位的有符号值。
    w.put_signed(coef, len);
}

/// `av_log2`（`coef` 最高位的位置）。
fn av_log2(v: i32) -> u32 {
    debug_assert!(v > 0);
    31 - (v as u32).leading_zeros()
}

/// ICS 头：`ics_reserved(1) window_sequence(2) use_kb_window(1) max_sfb(6/4) …`。
///
/// ⚠️ **短窗没有 `predictor_data_present` 那一位** —— 它只写在长窗分支里，
/// 短窗分支的解码端是直接 `predictor_present = 0` 赋值（不读位）。
/// 一开始给短窗也写了这一位，整条位流后移 1 位：第一个通道的 `ics_info` 还读得对，
/// 但从 `section_data` 起全错位，第二个通道连 `window_sequence` 都读成了长窗 ——
/// 症状是「`number of scalefactor bands in group (56) exceeds limit (49)`」，
/// **看起来像频带表错了，其实是少写/多写了一位**。
fn write_ics_info(w: &mut BitWriter, max_sfb: u8, seq: WindowSequence, group_lens: &[usize]) {
    w.put(0, 1); // ics_reserved
    // ⚠️ `EIGHT_SHORT_SEQUENCE = 2`（不是 1 —— 1 是 `LONG_START`）。见 `WindowSequence`。
    w.put(seq as u32, 2);
    w.put(0, 1); // use_kb_window = 0（正弦窗；KBD 时两边的 cfg 都会改这里 —— 目前统一 0）
    if seq == WindowSequence::EightShort {
        // ⚠️ 短窗的 max_sfb 只有 **4 位**（长窗 6 位），后面跟 7 位的
        // **窗口分组**，每一位的语义是「**1 = 并进当前组、0 = 开新组**」。
        // 所以「一个组装下全部 8 个窗口」= 写 **7 个 1**。
        //
        // 第一版写成了 7 个 0（把语义记反了）—— 解码端于是认为有 **8 个组**，
        // 每组的 section 只描述 1 个频带；它接着把我们后面的 flag/ID_END 位
        // 当成分段数据读，报出 `Number of bands (6) exceeds limit (1)`
        // —— 症状是「频带数超了」，根因却是**分组位的极性**。
        w.put(u32::from(max_sfb), 4);
        // 分组位：7 位，位 j 对应「窗口 j+1」——**1 = 并进当前组、0 = 开新组**。
        // 所以「一个组装 8 个窗」= 7 个 1；组边界处写 0。
        let mut bits = [1u32; 7];
        let mut acc = 0usize;
        for &len in &group_lens[..group_lens.len().saturating_sub(1)] {
            acc += len;
            bits[acc - 1] = 0;
        }
        let mut packed = 0u32;
        for b in bits {
            packed = (packed << 1) | b;
        }
        w.put(packed, 7);
        // ⚠️ 到此为止 —— 短窗分支**没有** `predictor_data_present`
        // （解码端只在长窗分支里读它，见 `decode_ics_info`）。
    } else {
        w.put(u32::from(max_sfb), 6);
        w.put(0, 1); // predictor_data_present（LC 不允许预测，恒 0）
    }
}

/// `tns_data` 的写出（长窗：`n_filt` 2 位、`length` 6 位、`order` 5 位）。
///
/// ⚠️ `coef_compress` **恒写 0**：开了压缩，解码端会改用另一张量化表
/// （`tns_tmp2_map_1_4`），而编码端的「压缩」只是平移索引 —— 两端的系数值
/// 就对不上了。少一位不如少一个 bug（ffmpeg 的注释也说这条压缩并不无损）。
fn write_tns(w: &mut BitWriter, tns_data: Option<&Tns>) {
    match tns_data {
        None => w.put(0, 1), // tns_data_present = 0
        Some(t) => {
            w.put(1, 1); // tns_data_present
            w.put(t.filters.len() as u32, 2); // n_filt（长窗 2 位）
            w.put(1, 1); // coef_res = 1（4 位系数）
            for f in &t.filters {
                w.put(u32::from(f.length), 6);
                w.put(u32::from(f.order), 5);
                w.put(u32::from(f.direction), 1);
                w.put(0, 1); // coef_compress = 0
                for &c in &f.coef_idx {
                    w.put(u32::from(c), 4);
                }
            }
        }
    }
}

/// `section_data`：把**连续同码本**的频带并成一段，每段写 `(sect_cb, sect_len)`。
///
/// 全零带归到码本 0。长度用「转义 + 续接」编码：
/// 写 `run_bits` 位的值；等于满值就走续接（再把 `sect_len` 加上那个满值），
/// 所以任何长度都能用若干个满值 + 一个非满值表示。
fn write_sections(w: &mut BitWriter, band_cb: &[u8], band_zero: &[bool], run_bits: u32) {
    let n = band_cb.len();
    let mut i = 0usize;
    while i < n {
        let cb = if band_zero[i] { 0 } else { band_cb[i] };
        let mut len = 1usize;
        while i + len < n {
            let next = if band_zero[i + len] {
                0
            } else {
                band_cb[i + len]
            };
            if next != cb {
                break;
            }
            len += 1;
        }
        // ⚠️ 段长度**不能是转义值的整数倍** —— 那样末尾会写出一个 0，
        // 解码端的 `while (k < max_sfb)` 就转不动了。
        // 碰到整除就把最后一段砍掉 1 个频带，剩下那个自成一段（长度 1）。
        let run_esc = (1usize << run_bits) - 1;
        let mut this = len;
        if this % run_esc == 0 {
            this -= 1;
            write_section(w, cb, this, run_bits);
            // 多出来的那一个频带：如果它本来是同码本，单独成一段（长度 1）；
            // 长度 1 不是转义值的倍数，合法。
            let tail_cb = if band_zero[i + this] {
                0
            } else {
                band_cb[i + this]
            };
            write_section(w, tail_cb, 1, run_bits);
            i += len;
            continue;
        }
        write_section(w, cb, this, run_bits);
        i += len;
    }
}

/// 一段 `section` 的写出（长窗 5 位/转义 31，短窗 3 位/转义 7）。
///
/// 解码端是 `sect_end += sect_len_incr; .. while (sect_len_incr == 31)` ——
/// **每写一个值都会被加上去**，所以长于 31 的段用「若干个 31 + 一个余数」表示。
///
/// ⚠️ 余数**不能是 0**：解码端读到 0 会让 `sect_end` 不前进，
/// 而 `while (k < max_sfb)` 的循环条件仍成立 —— 死循环 / 位流错位。
/// 所以正好整除时要拆成 `31, ..., 31, 31`（把最后一个 31 拆成 `31` 与……不行），
/// 正确做法是让**最后一个**值落在 `1..=30`：`31 × n + r`，`r ∈ 1..=30`。
fn write_section(w: &mut BitWriter, cb: u8, len: usize, run_bits: u32) {
    // 长窗 `sect_len_incr` 用 5 位（转义值 31）；**短窗只用 3 位**（转义值 7）。
    // 「段长不能是转义值的整数倍」的约束对两种位宽同样成立。
    let run_esc = (1usize << run_bits) - 1;
    debug_assert!(len > 0, "段的长度不能是 0");
    w.put(u32::from(cb), 4);
    // 先写足够的 31，直到剩下的部分能用一个 1..=30 的数表示。
    let mut left = len;
    while left > run_esc {
        // 若 `left - 转义值` 会落在「转义值整数倍且余 0」上，就少写一个转义。
        if (left - run_esc) % run_esc == 0 {
            break;
        }
        w.put(run_esc as u32, run_bits);
        left -= run_esc;
    }
    if left > run_esc {
        // 走到这里说明 `left` 是 31 的整数倍 —— 拆成 (left - 1) 个 31 再加……
        // 不行，`left - 1` 不是 31 的倍数。改用「先写 30，再写剩下的 31」。
        // 但 30 不是转义值，会被当成段结束。所以唯一合法的拆法是
        // 「若干个 31 + 一个 1..=30」——而 31 的倍数永远做不到。
        //
        // 结论：**段的长度不能是 31 的整数倍**。调用方（`write_sections`）
        // 必须保证这一点 —— 见那里的处理。
        unreachable!("段长度不能是 31 的整数倍（write_sections 应当已经拆开）");
    }
    w.put(left as u32, run_bits);
}

/// `scalefactor_data`：相邻非零带的 sf **差分** + 60，查 `SCALE_DIFF_ZERO` 码表。
///
/// ⚠️ 差分链是**跳过零带**的：零带不参与，下一个非零带与**上一个非零带**做差。
/// 这条与 `encode_scale_factors` 一致（它只在 `!zeroes` 时更新 `off_sf`）。
fn write_scalefactors(w: &mut BitWriter, band_sf: &[i32], band_zero: &[bool], prev: &mut i32) {
    // ⚠️ 差分的**基准是 `global_gain` 本身**，不是 `global_gain - 100`。
    //
    // 解码端（`decode_scalefactors`，逐字）：
    //
    // ```c
    // int offset[3] = { global_gain, global_gain - NOISE_OFFSET, 0 };
    // ...
    // default:
    //     offset[0] += get_vlc2(...) - SCALE_DIFF_ZERO;   // SCALE_DIFF_ZERO = 60
    //     sce->sfo[...] = offset[0] - 100;                // ← 这里才减 100
    // ```
    //
    // 所以 **`sfo = global_gain + Σdiff - 100`**。要让 `sfo == band_sf`，
    // 差分链的起点（也就是第一条 diff 的"上一个值"）必须是 **`global_gain`**；
    // 写成 `global_gain - 100` 会让整条链整体偏 100（`2^(100/4) = 2^25` 倍）。
    // ⚠️ 差分链**跨组延续**（解码端的 `offset[0]` 不按组重置），所以 `prev`
    // 由调用方带进带出；`write_pass` 用 `global_gain` 初始化它。
    for sfb in 0..band_sf.len() {
        if band_zero[sfb] {
            continue;
        }
        let diff = band_sf[sfb] - *prev + 60;
        let idx = diff.clamp(0, 120) as usize;
        w.put(
            tables::SCALEFACTOR_CODE[idx],
            u32::from(tables::SCALEFACTOR_BITS[idx]),
        );
        *prev = band_sf[sfb];
    }
}

/// 一个频带的谱系数（Huffman + 符号位 + escape）。
fn write_band(w: &mut BitWriter, band: &[f32], sf: u8, cb: u8) -> crate::Result<()> {
    if cb == 0 {
        return Ok(());
    }
    let cb = cb as usize;
    let dim = CB_DIM[cb];
    let maxval = i32::from(CB_MAXVAL[cb]);
    let q34v = q34(sf);
    let unsigned = CB_UNSIGNED[cb];
    let idx_tab = &tables::SPECTRAL_CODE[cb - 1];
    let bit_tab = &tables::SPECTRAL_BITS[cb - 1];
    let size = tables::SPECTRAL_SIZE[cb - 1] as usize;
    let vecs = tables::CODEBOOK_VECTORS[cb - 1];

    for chunk in band.chunks(dim) {
        let mut q = [0i32; 4];
        for (j, &c) in chunk.iter().enumerate() {
            let m = quant_one(c.abs(), q34v, maxval);
            q[j] = if unsigned || c >= 0.0 { m } else { -m };
        }
        let off = if unsigned { 0 } else { maxval };
        let mut curidx = 0usize;
        for j in 0..dim {
            curidx = curidx * usize::from(CB_RANGE[cb]) + (q[j] + off) as usize;
        }
        if curidx >= size {
            return Err(Error::Bad("谱系数索引越界（fu 内不该发生）"));
        }
        w.put(u32::from(idx_tab[curidx]), u32::from(bit_tab[curidx]));
        // 无符号码本：逐维写非零系数的符号。
        if unsigned {
            for j in 0..dim {
                if vecs[curidx * dim + j] != 0 {
                    w.put(u32::from(chunk[j] < 0.0), 1);
                }
            }
        }
        // escape 码本。
        if cb == CB_ESC {
            for j in 0..dim {
                if vecs[curidx * dim + j] == 64 {
                    let coef = escape_coef(chunk[j].abs(), q34v);
                    write_escape(w, coef);
                }
            }
        }
    }
    Ok(())
}
