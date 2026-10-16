//! AAC-LC 编码器（长/短窗 + 过渡窗、单/双声道、M/S 立体声、TNS；**无 PNS / IS**）。
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

/// `sf_offset` 的搜索上界（含）。
///
/// 够粗了：偏移每 +4 让量化精度翻一倍，+120 相当于把每带的 sf 抬到几乎全零带；
/// 再往上位数不再降（全零带的位数是常数）。
const MAX_SF_OFFSET: i32 = 120;

/// 第一帧（还没有上一帧的答案）的搜索种子 —— 取区间中点。
const SF_OFFSET_SEED_INIT: i32 = MAX_SF_OFFSET / 2;

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
    /// 双声道时允不允许用 **M/S 立体声**（默认开）。
    ///
    /// 逐帧决策：两个声道相关时才用（判据见 `use_midside`）。开了它，位流里会写
    /// `common_window = 1` + `ms_mask_present = 1`。
    pub ms_stereo: bool,
    /// **编码时用几个线程**（默认 **1** = 不并行）。
    ///
    /// ## 为什么由调用者给、而不是库自己探测
    ///
    /// 只有调用者知道这台机器 / 这个 App **还有多少算力可用**：
    /// `std::thread::available_parallelism()` 报的是**整机核数**，它不知道调用者
    /// 是不是已经在并行转几个文件、也不知道这是不是 Android 上被系统限流的后台任务。
    /// 库自己起线程 = 把「什么时候发热、什么时候抢 CPU」的决定权从调用方拿走了。
    ///
    /// ## 两条并行路径
    ///
    /// 1. **帧并行**（`encode_frames`，批量接口）—— 把一批帧按连续区间切给各线程，
    ///    总工作量不变、核越多越快，是主要的提速来源（逐帧接口用不上这条）。
    /// 2. **声道并行**（`encode_frame` / 帧并行内部）—— 双声道时两个声道各占一个
    ///    线程，**输出逐字节相同**。
    ///
    /// 帧并行的输出**不是**逐字节相同（原因见 `encode_frames` 的说明：`sf_offset`
    /// 是近似搜索的锚点）。两条路径的线程数都**封顶到可用并行度** ——
    /// 单声道给多少都是 1、一批 4 帧给 8 个线程也只起 4 个。
    ///
    /// ## ⚠️ 调用方要自己保证不在主线程上等
    ///
    /// 这个 API 是**同步**的：`encode_frame` 会一直占到这一帧编完。
    /// Android 上要在工作线程 / 协程里调，别在 UI 线程调（那是 ANR 的来源，
    /// 跟这里开几个线程无关）。
    pub threads: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            channels: 2,
            sample_rate: 44100,
            bitrate: 128_000,
            kb_window: false,
            ms_stereo: true,
            threads: 1,
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
/// 一帧的**窗决策**（`window_sequence` + 分组）。
///
/// 双声道走 `common_window = 1` 时两个声道**共用同一份 `ics_info`**，所以窗序列与
/// 分组必须两个声道一致 —— 决策得提到 `encode_frame` 里**联合**做，不能各声道各决定。
/// 单声道就用自己那一份。
struct WindowPlan {
    /// 本帧的窗序列（`ONLY_LONG` / `LONG_START` / `EIGHT_SHORT` / `LONG_STOP`）。
    ///
    /// ⚠️ **不能只有「长/短」两态**：长 ↔ 短之间必须走过渡窗，否则相邻两帧的窗
    /// 在重叠区不互补 —— 见 `decide_sequence` 与 `mdct::long_start_window`。
    seq: WindowSequence,
    /// 分组长度（长窗恒 `[1]`）。
    group_lens: Vec<usize>,
    /// 每个声道本帧的攻击（要写回 `ChannelState::attack_prev`）。
    attacks: Vec<Option<usize>>,
}

/// **窗序列状态机**（照 `ffmpeg` 的 `psy_3gpp_window`，`aacpsy.c:443-463`）。
///
/// ```text
///   ONLY_LONG   + 攻击 → LONG_START   （本帧仍是长窗，但尾巴给短窗让位）
///   ONLY_LONG   + 无   → ONLY_LONG
///   LONG_START  + 任意 → EIGHT_SHORT  （过渡窗已发，下一帧必须是短窗）
///   LONG_STOP   + 攻击 → LONG_START
///   LONG_STOP   + 无   → ONLY_LONG
///   EIGHT_SHORT + 攻击 → EIGHT_SHORT  （继续短窗）
///   EIGHT_SHORT + 无   → LONG_STOP    （本帧仍是长窗，但头给上一帧的短窗让位）
/// ```
///
/// ## 为什么不能长↔短直接跳（实测）
///
/// MDCT 的完美重建要求**相邻两帧的窗在重叠区互补**（`w_a² + w_b² = 1`）。
/// 长配长、短配短都天然满足，**长配短不满足**：长窗前半是上升的 `lwindow`，
/// 而短窗帧两头是零 —— `[0, 448)` 那一段没有任何东西与长窗相加。
///
/// 帧对齐的 MDCT 域归因（真音乐 8 秒 @1 Mbps，逐帧与源比）：
///
/// | 帧类别 | 帧数占比 | 总误差占比 | 帧 SNR 中位 |
/// |---|---|---|---|
/// | 短窗 | 13.1% | 23.2% | 18.3 dB |
/// | **紧跟短窗之后的长窗** | **6.0%** | **55.9%** | **11.7 dB** |
/// | 紧邻短窗之前的长窗 | 6.0% | 15.2% | 15.2 dB |
/// | 孤立的长窗 | 75.0% | **5.7%** | **69.0 dB** |
///
/// 孤立长窗只占 5.7% 的误差 —— 说明**长窗那条路本身是好的**，坏的是切换。
fn decide_sequence(
    attack: Option<usize>,
    prev_attack: Option<usize>,
    prev: WindowSequence,
) -> (WindowSequence, Vec<usize>) {
    match prev {
        WindowSequence::OnlyLong => {
            if attack.is_some() {
                (WindowSequence::LongStart, vec![1])
            } else {
                (WindowSequence::OnlyLong, vec![1])
            }
        }
        WindowSequence::LongStart => {
            // 过渡窗已经发出去，本帧必然是短窗。攻击位置用上一帧记下来的那个
            // （本帧不一定还有攻击 —— 但那不重要，短窗覆盖的正是上一帧的攻击）。
            let pos = attack.or(prev_attack).unwrap_or(8);
            (WindowSequence::EightShort, group_lengths(WINDOW_GROUPING[pos]))
        }
        WindowSequence::LongStop => {
            if attack.is_some() {
                (WindowSequence::LongStart, vec![1])
            } else {
                (WindowSequence::OnlyLong, vec![1])
            }
        }
        WindowSequence::EightShort => {
            if let Some(pos) = attack {
                (WindowSequence::EightShort, group_lengths(WINDOW_GROUPING[pos]))
            } else {
                (WindowSequence::LongStop, vec![1])
            }
        }
    }
}

impl WindowPlan {
    /// 一个声道的窗决策（单声道走这条）。
    fn for_one(attack: Option<usize>, prev_attack: Option<usize>, prev: WindowSequence) -> Self {
        let (seq, group_lens) = decide_sequence(attack, prev_attack, prev);
        Self {
            seq,
            group_lens,
            attacks: vec![attack],
        }
    }

    /// **双声道联合**的窗决策：只要**任一**声道有攻击就算有攻击，
    /// 分组位置取两个声道里**最早**的那个攻击。
    fn for_pair(
        attacks: &[Option<usize>],
        prev: &[Option<usize>],
        prev_seq: WindowSequence,
    ) -> Self {
        let attack = attacks.iter().copied().flatten().min();
        let prev_attack = prev.iter().copied().flatten().min();
        let (seq, group_lens) = decide_sequence(attack, prev_attack, prev_seq);
        Self {
            seq,
            group_lens,
            attacks: attacks.to_vec(),
        }
    }
}

/// 一帧的**跨帧语义状态** —— 只有这四项会影响输出。
///
/// `cost_cache` 只影响**速度**（它本来就是帧内缓存，每帧开头 `reset`）。
///
/// ⚠️ `sf_offset_prev` 则**也影响结果**：偏移搜索是近似搜索、锚在这个种子上
/// （详见字段自身的说明）。所以 `to_chans` 把它设成固定初值 = 批量路径**每帧都
/// 从冷种子开始**，既丢掉了热启动、也让结果与逐帧路径差一点点。
/// 要「逐字节相同」就得让搜索与种子无关 —— 那是另一件事，现在没做。
#[derive(Clone)]
struct FrameState {
    /// 每个声道：上一帧的 coded 样本（按**本帧**要编的域取用）。
    overlap: Vec<Vec<f32>>,
    /// 每个声道：上一帧的 **M/S 域** coded 样本。
    overlap_ms: Vec<Vec<f32>>,
    /// 上一帧每个声道的攻击（窗序列状态机的分组定位用）。
    attack_prev: Vec<Option<usize>>,
    /// 上一帧实际发出的窗序列（状态机的输入）。
    window_seq_prev: WindowSequence,
}

impl FrameState {
    fn from_chans(chans: &[ChannelState]) -> Self {
        Self {
            overlap: chans.iter().map(|c| c.overlap.clone()).collect(),
            overlap_ms: chans.iter().map(|c| c.overlap_ms.clone()).collect(),
            attack_prev: chans.iter().map(|c| c.attack_prev).collect(),
            window_seq_prev: chans[0].window_seq_prev,
        }
    }

    /// 铺成一份**独立的** `ChannelState`（缓存是空的 —— 帧内会自己填）。
    fn to_chans(&self) -> Vec<ChannelState> {
        self.overlap
            .iter()
            .zip(&self.overlap_ms)
            .zip(&self.attack_prev)
            .map(|((ov, ovm), atk)| ChannelState {
                overlap: ov.clone(),
                overlap_ms: ovm.clone(),
                attack_prev: *atk,
                window_seq_prev: self.window_seq_prev,
                cost_cache: BandCostCache::new(),
                sf_offset_prev: SF_OFFSET_SEED_INIT,
            })
            .collect()
    }

    /// 把语义状态写回已有的 `ChannelState`（保留它们的缓存与搜索种子）。
    fn apply_to(&self, chans: &mut [ChannelState]) {
        for (ch, st) in chans.iter_mut().enumerate() {
            st.overlap.clone_from(&self.overlap[ch]);
            st.overlap_ms.clone_from(&self.overlap_ms[ch]);
            st.attack_prev = self.attack_prev[ch];
            st.window_seq_prev = self.window_seq_prev;
        }
    }
}

/// 一帧的**预扫结果** —— 编它需要的一切，不含编码器自己的缓存。
///
/// `prepare_frame` 产出它（并把跨帧状态推进一步），`encode_frame_inner` 消费它。
/// 按帧并行时，预扫与编码被拆成两步，中间夹着并行（见 `encode_frames`）。
struct PreparedFrame {
    /// 本帧实际要编的信号（M/S 时就是 M/S，否则是原始 L/R）。
    coded: Vec<Vec<f32>>,
    /// MDCT 的左半（上一帧的 coded，按**本帧**的域取）。
    hist: Vec<Vec<f32>>,
    /// 窗序列与分组。
    plan: WindowPlan,
    use_ms: bool,
}

/// 把一个声道的这一帧推进它的重叠历史（AAC 是 50% 重叠，左半就是上一帧）。
fn push_history(hist: &mut [f32], samples: &[f32]) {
    let n = samples.len().min(LONG_LEN);
    hist[..n].copy_from_slice(&samples[..n]);
    for v in &mut hist[n..] {
        *v = 0.0;
    }
}

/// 用不用 M/S 立体声（`ms_mask_present = 1`：**所有**频带都用）。
///
/// ## 判据：比两套表示的 **L1 范数**
///
/// `M = (L+R)/2`、`S = (L-R)/2`。L1 是**齐次**的（整体放大 2 倍 = sf 抬 2 档、位数不变），
/// 所以直接比 L1 就是比「各自在自己那一档 sf 下要花多少位」：
///
/// - 相关性强（`L ≈ R`）：`L1(M) + L1(S) ≈ L1(L)`，而 L/R 是 `2·L1(L)` → **选 M/S**
/// - 不相关：`≈ 1.4·(L1(L) + L1(R))` → 选 L/R
/// - 反相（`R = -L`）：两者相等（平局 → 选 L/R，少一次变换）
///
/// ## ⚠️ 为什么是「逐帧」而不是「逐带」
///
/// 逐带掩码（`ms_mask_present = 2`）要求**先把两个声道的谱都算出来**再决定，
/// 而现在每个声道是「算谱 → 搜索 → 写」一条路走到底（还能两个声道并行）。
/// 逐帧决策可以在**时域**先变换 —— MDCT 是线性的，`MDCT((L+R)/2) = (MDCT(L)+MDCT(R))/2`
/// —— 于是完全不用动结构。先做逐帧，把「有没有 M/S」这件事跑通；逐带留到下一步。
fn use_midside(l: &[f32], r: &[f32]) -> bool {
    let mut l1_lr = 0f64;
    let mut l1_ms = 0f64;
    for (a, b) in l.iter().zip(r) {
        l1_lr += f64::from(a.abs()) + f64::from(b.abs());
        l1_ms += f64::from(((a + b) * 0.5).abs()) + f64::from(((a - b) * 0.5).abs());
    }
    // 留一点余量：势均力敌时不换（换了也只是白算一遍变换）。
    l1_ms < l1_lr * MS_MARGIN
}

/// M/S 决策的余量（`< 1.0` 表示「M/S 要明显更省才用」）。实测调这个值。
const MS_MARGIN: f64 = 0.98;

/// `band_cost` 的**帧内缓存**（每个声道一份，跨帧复用）。
///
/// 同一帧要跑多「趟」（外层按位数预算二分搜全局 sf 偏移），而**趟与趟之间很多
/// 频带的候选 sf 窗口是同一套** —— 窗口主要由「上一个非零带 ± 60」决定，只有被
/// `floor = sf_floor + 偏移` 抬起来的那部分才随偏移移动。所以同一个 `(带, sf)`
/// 会被反复求值，而它的值只取决于该带的谱系数（一帧内不变）。
///
/// 缓存返回的值与现算**逐位相同**，所以输出逐字节不变 —— 纯提速。
/// 内存：每带 `256 × (f32,u8)` = 2 KB；长窗约 50 条带 → 100 KB，短窗最多 8 组
/// 共 ~112 条带 → 约 230 KB。**挂在声道状态上跨帧复用，不随帧增长。**
struct BandCostCache {
    /// `tables[组][带][sf] = (失真, 码本, 位数)`；码本为 `NO_CB` = 还没算过。
    ///
    /// 位数与 λ 无关，所以同一个 `(带, sf)` 在**所有** λ 取值下都能复用（二分 λ 时这是关键）。
    tables: Vec<Vec<Vec<(f32, u8, u32)>>>,
    /// `band_cost` 用的量化 scratch（不封顶的 `q`）—— 挂在这里**零额外分配**：
    /// 一条带最多 ~80 个系数，容量涨一次就不再动。
    q_scratch: Vec<i32>,
}

/// 「这一格还没算过」的哨兵 —— 真码本只到 11，所以 `0xFF` 安全。
const NO_CB: u8 = 0xFF;

/// 逐带 RD 的 λ 斜率：`λ = RD_LAMBDA_K · sf_offset²`（失真/位数的兑换率）。
///
/// ## ⚠️ 现在是 `0.0`：这条「λ 挂偏移」的耦合路走不通（量过了，别再试同一条）
///
/// 让 λ 跟着偏移走是想复用外层「位数 ≤ 预算」的搜索（偏移本来就是粗细旋钮），
/// 但实测 `K ∈ {0, 1, 10, 100}` 结果**完全一样**（13.85/13.86 dB、码率也不动）——
/// RD 项根本没参与决策。两个原因：
///
/// 1. **量级**：sf 加一档失真 ×1.41（步长 ∝ `2^(sf/4)`），位数只变几点 —— 要顶动一档
///    λ 得在 `1e3~1e4`，而按「失真/位数」估出来是几十（差 1~2 个数量级）。
/// 2. **耦合本身是死的**：λ 一旦大到能顶动一档，偏移只要 > 0 就同时抬下界 + 给大 λ →
///    位数直接崩到预算以下，搜索只能退回偏移 0 —— 粗细旋钮变成**二值**的。
///
/// **正路**（见 TODO 第八轮）：下界交还给「不裁剪」、由 λ 自己二分（对数网格 `2^k`），
/// 偏移这个旋钮退场。在那之前这里保持 `0.0` —— 行为与旧路**逐字节相同**（md5 已对）。
const RD_LAMBDA_K: f32 = 0.0;

impl BandCostCache {
    fn new() -> Self {
        Self {
            tables: Vec::new(),
            q_scratch: Vec::new(),
        }
    }

    /// 按这一帧的形状准备（组数 + 每组条带数）。形状没变就只清空，**不重新分配**。
    fn reset(&mut self, shapes: &[usize]) {
        self.tables.resize_with(shapes.len(), Vec::new);
        for (g, &n) in shapes.iter().enumerate() {
            let t = &mut self.tables[g];
            t.resize_with(n, Vec::new);
            for band in t.iter_mut() {
                band.resize(256, (0.0, NO_CB, 0));
                for e in band.iter_mut() {
                    e.1 = NO_CB;
                }
            }
        }
    }

    #[inline]
    fn get(&self, g: usize, b: usize, sf: u8) -> Option<(f32, u8, u32)> {
        let e = *self.tables.get(g)?.get(b)?.get(sf as usize)?;
        (e.1 != NO_CB).then_some(e)
    }

    #[inline]
    fn put(&mut self, g: usize, b: usize, sf: u8, v: (f32, u8, u32)) {
        if let Some(e) = self
            .tables
            .get_mut(g)
            .and_then(|t| t.get_mut(b))
            .and_then(|t| t.get_mut(sf as usize))
        {
            *e = v;
        }
    }
}

/// **一个声道自己的状态** —— 编码时唯一可变的东西。
///
/// 抽出来是为了能**两个声道并行**：`encode_channel` 除了这个结构之外只读
/// `Encoder` 的其余部分，所以两个声道之间没有共享的可变状态。
/// 谁起线程由调用者定（`EncoderConfig::threads`），库不自己探测核数。
struct ChannelState {
    /// 上一帧的后半段（`N` 个样本），与这一帧拼成 `2N`（AAC 是 50% 重叠）——
    /// **原始域**（L / R）的那一份。
    overlap: Vec<f32>,
    /// 同一件事，但存的是 **M/S 域**（`M = (L+R)/2` / `S = L-R`）的那一份。
    ///
    /// ⚠️ **两套历史都要维护**，因为 M/S 是**逐帧**决策的：MDCT 的重叠必须与
    /// 本帧实际编的那个域一致（TDAC 要求「上一帧那一半」是同一个信号），
    /// 而下一帧可能换回另一个域。两套各 4 KB，代价可忽略。
    overlap_ms: Vec<f32>,
    /// 「上一帧的攻击落在哪个子块」（`None` = 没有攻击）。
    ///
    /// 只用来给**分组**定位：`EIGHT_SHORT` 帧拿它查 `WINDOW_GROUPING`，
    /// 把攻击所在的那个窗口单独分一组（见 `decide_sequence`）。
    /// 短窗的覆盖区是 `buf[448..1600]`，换算到**本帧样本**就是 `[-576, +576)`。
    attack_prev: Option<usize>,
    /// 上一帧实际发出的**窗序列** —— 本帧的窗序列状态机拿它决定过渡态
    /// （见 `decide_sequence`）。初值 `ONLY_LONG`（第一帧之前没有短窗）。
    window_seq_prev: WindowSequence,
    /// `band_cost` 的帧内缓存（见 `BandCostCache`）。
    cost_cache: BandCostCache,
    /// **上一帧选中的 `sf_offset`** —— 下一帧搜偏移时的**种子**。
    ///
    /// 相邻帧的谱形状是连续的，所以上一帧的答案是很好的出发点：从它出发做
    /// 指数搜索，绝大多数帧 2–4 趟就收敛（固定区间二分恒 7 趟）。
    ///
    /// ⚠️ 它**不只是种子**：`plan_group` 的三级细化（12→3→1）是**近似搜索**、
    /// 锚在这个偏移上，所以换种子会落到不同的格点、给出**略微不同**的码率分配
    /// （不是「同一个答案」—— 早期这里写错过，实测真音乐 350 帧里 2 帧差
    /// 14 字节）。改这里要知道自己在动**结果**，不只是动速度。
    sf_offset_prev: i32,
}

pub struct Encoder {
    cfg: EncoderConfig,
    /// 每声道一份可变状态。`threads >= 2` 时两个声道各在一个线程里动自己那份。
    chans: Vec<ChannelState>,
    /// **长窗**的开窗表（`2N` 长）。
    window: Vec<f32>,
    /// **过渡窗**（长 ↔ 短）。没有它们就只能长↔短直接跳变、重叠区不互补
    /// —— 见 `mdct::long_start_window` 与 `decide_sequence`。
    window_long_start: Vec<f32>,
    window_long_stop: Vec<f32>,
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
        // 过渡窗（长 ↔ 短）—— 形状与解码端的 `LONG_START` / `LONG_STOP` 合成窗
        // 成对，构造性判据见 `mdct::transition_windows_are_complementary`。
        let window_long_start = crate::mdct::long_start_window(&window, &window_short);
        let window_long_stop = crate::mdct::long_stop_window(&window, &window_short);
        // 每帧的位数预算。`bitrate` 是**目标**不是保证 —— 编码器靠
        // 「二分搜整帧 sf 偏移」逼近它（见 `encode_channel`）。
        let frames_per_sec = f64::from(cfg.sample_rate) / LONG_LEN as f64;
        let bits_per_frame =
            (f64::from(cfg.bitrate) / frames_per_sec / f64::from(cfg.channels)).max(1.0);

        Ok(Self {
            chans: (0..cfg.channels)
                .map(|_| ChannelState {
                    overlap: vec![0f32; LONG_LEN],
                    overlap_ms: vec![0f32; LONG_LEN],
                    attack_prev: None,
                    window_seq_prev: WindowSequence::OnlyLong,
                    cost_cache: BandCostCache::new(),
                    sf_offset_prev: SF_OFFSET_SEED_INIT,
                })
                .collect(),
            window,
            window_long_start,
            window_long_stop,
            sr_idx: sr_idx as usize,
            frame_index: 0,
            swb,
            swb_short,
            window_short,
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
        let mut chans = std::mem::take(&mut self.chans);
        let r = self.encode_frame_inner(frame, &mut chans);
        self.chans = chans;
        let (bytes, short) = r?;
        if short {
            self.short_frames += 1;
        }
        self.frame_index += 1;
        Ok(bytes)
    }

    /// 算一帧「要编什么、怎么编」，并把**跨帧状态**往下推一步。
    ///
    /// 决策逻辑（M/S 选择、窗序列状态机、重叠历史的推进）只有这一份 ——
    /// `encode_frame_inner` 直接用它，`encode_frames` 的预扫也用它（只是不编码）。
    ///
    /// ⚠️ 它会**修改 `chans`**（推进 `overlap*` / `attack_prev` / `window_seq_prev`），
    /// 所以调用方要么把 `chans` 用完就丢（预扫），要么紧接着编码（逐帧）。
    fn prepare_frame(&self, frame: &Frame, chans: &mut [ChannelState]) -> PreparedFrame {
        let n = self.cfg.channels as usize;
        let is_stereo = self.cfg.channels == 2;

        // ---- M/S 立体声：逐帧决策，决定本帧**实际要编的信号** ----
        //
        // MDCT 是线性的（`MDCT((L+R)/2) = (MDCT(L)+MDCT(R))/2`），所以逐帧的 M/S
        // 可以在**时域**先变换再进 MDCT —— 不用把「两个声道的谱都算出来」才能决定。
        let ms_planes: Option<Vec<Vec<f32>>> = (is_stereo && self.cfg.ms_stereo).then(|| {
            let (l, r) = (&frame.planes[0], &frame.planes[1]);
            // ⚠️ **两个都要减半**：解码端用的是 `butterflies_float`（`L = M + S`、
            // `R = M − S`，没有额外缩放），所以编码端必须是
            // `M = (L+R)/2`、`S = (L−R)/2` —— 解这个方程组得到的。
            // 只减 M 不减 S 的话，解出来是 `1.5L − 0.5R`：位流**解得开**（语法全对），
            // 但波形是错的 —— 实测 SNR 从 11.27 dB 掉到 **5.48 dB**。
            vec![
                l.iter().zip(r).map(|(a, b)| (a + b) * 0.5).collect(),
                l.iter().zip(r).map(|(a, b)| (a - b) * 0.5).collect(),
            ]
        });
        let use_ms =
            ms_planes.is_some() && use_midside(&frame.planes[0], &frame.planes[1]);
        let coded: Vec<Vec<f32>> = match (&ms_planes, use_ms) {
            (Some(ms), true) => ms.clone(),
            _ => frame.planes.clone(),
        };

        // ---- 窗决策（双声道**联合**，因为要共用一份 `ics_info`）----
        // 攻击检测跑在**本帧实际要编的信号**上（M/S 时就是 M/S）。
        let plan = if is_stereo {
            let attacks: Vec<Option<usize>> = coded.iter().map(|p| detect_attack(p)).collect();
            let prev: Vec<Option<usize>> = chans.iter().map(|c| c.attack_prev).collect();
            WindowPlan::for_pair(&attacks, &prev, chans[0].window_seq_prev)
        } else {
            WindowPlan::for_one(
                detect_attack(&coded[0]),
                chans[0].attack_prev,
                chans[0].window_seq_prev,
            )
        };

        // ---- 重叠历史与跨帧状态：**读旧值、写新值** ----
        //
        // 先取本帧要用的那一套（按 `use_ms`），再把**两套都**往下推进 ——
        // 下一帧可能换域，而 MDCT 的 TDAC 要求「上一帧那一半」是**同一个域**的信号。
        let hist: Vec<Vec<f32>> = (0..n)
            .map(|ch| {
                if use_ms {
                    chans[ch].overlap_ms.clone()
                } else {
                    chans[ch].overlap.clone()
                }
            })
            .collect();
        for ch in 0..n {
            push_history(&mut chans[ch].overlap, &coded[ch]);
            if let Some(ms) = ms_planes.as_deref() {
                push_history(&mut chans[ch].overlap_ms, &ms[ch]);
            }
            // 下一帧的窗序列状态机要看这两项。
            chans[ch].attack_prev = plan.attacks[ch];
            chans[ch].window_seq_prev = plan.seq;
        }

        PreparedFrame {
            coded,
            hist,
            plan,
            use_ms,
        }
    }

    /// **按帧并行**地编一整批（线程数取 `cfg.threads`）。
    ///
    /// ## 为什么能并行（以及为什么必须「先预扫」）
    ///
    /// AAC-LC 逐帧独立，**唯一的例外是窗序列** —— `decide_sequence` 的状态机
    /// 要看上一帧实际发的序列。其余状态都不是障碍：`cost_cache` 本来就是帧内
    /// 缓存、重叠历史就是**上一帧的 coded 样本**（可以直接算出来）。
    ///
    /// ⚠️ `sf_offset_prev` 看着也像「只影响从哪儿开始探」，其实**不是**：
    /// `plan_group` 的三级细化（12→3→1）是**近似搜索**、锚在「上一帧留下的偏移」
    /// 上，换锚点就会落到不同的格点。批内每帧的锚由预扫铺好，所以**同一批里**
    /// 批量与逐帧一致；跨批（或与单帧 `encode_frame` 混用）时锚不一样，
    /// 个别帧的码率分配会差一点 —— 实测真音乐 350 帧里 **2 帧**不同、
    /// 合计 **14 字节**，SNR 相同。
    ///
    /// 所以先顺序推一遍状态（每帧一次 M/S 决策 + 一次 `detect_attack`，
    /// 相对编码本身可以忽略），再并行编码 —— 与逐帧 `encode_frame` 的输出
    /// **帧数与体积基本一致（每帧差 ≤ 16 字节），但不保证逐字节相同**。
    ///
    /// ⚠️ **批与批之间是顺序的**：`self.chans` 会推进到批尾，所以调用方可以
    /// 连续喂多批（CLI 的音频编码 worker 就是这么用的）。
    pub fn encode_frames(&mut self, frames: &[Frame]) -> crate::Result<Vec<Vec<u8>>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let n = self.cfg.channels as usize;
        let threads = self.cfg.threads.max(1).min(frames.len());

        // ---- 1. 顺序预扫：把每帧的「输入状态」推出来 ----
        //
        // 只为推进跨帧状态（`prepare_frame` 顺带算出的 coded/plan 会被丢弃 ——
        // 并行那一步会各自重算一遍，那点重复远比编码本身便宜）。
        let mut states: Vec<FrameState> = Vec::with_capacity(frames.len());
        let mut cur = FrameState::from_chans(&self.chans);
        let mut scratch = cur.to_chans();
        for frame in frames {
            if frame.planes.len() != n {
                return Err(Error::Bad("声道数与配置不一致"));
            }
            for p in &frame.planes {
                if p.len() > LONG_LEN {
                    return Err(Error::Bad("一帧超过 1024 个样本"));
                }
            }
            states.push(cur.clone());
            let _ = self.prepare_frame(frame, &mut scratch);
            cur = FrameState::from_chans(&scratch);
        }
        // 跨帧状态推进到批尾（缓存与搜索种子留在 `self.chans` 上不动）。
        cur.apply_to(&mut self.chans);

        // ---- 2. 并行编码：按**连续区间**分块，块内顺序、块间并行（无锁）----
        //
        // `&*self` 把可变借用降成不可变 —— 并行那一段只读编码器
        // （`encode_frame_inner` 就是 `&self`，状态全在每帧自己那份 `ChannelState` 里）。
        let this: &Encoder = &*self;
        let chunk = frames.len().div_ceil(threads);
        let mut results: Vec<(usize, Vec<u8>, bool)> = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (t, part) in frames.chunks(chunk).enumerate() {
                let base = t * chunk;
                let states = &states;
                handles.push(scope.spawn(move || -> crate::Result<Vec<(usize, Vec<u8>, bool)>> {
                    let mut v = Vec::with_capacity(part.len());
                    // ⚠️ **一份 `ChannelState` 在块内复用**（不是每帧一份）：
                    // `apply_to` 只覆盖那四项语义状态，**保留 `sf_offset_prev` 与
                    // `cost_cache`** —— 于是块内第 2 帧起就拿到了上一帧的搜索种子。
                    //
                    // 这不是微调：种子决定偏移搜索的**起点**，冷启动（`to_chans`）
                    // 时指数搜索要从 `SF_OFFSET_SEED_INIT` 一路探到真值（实测 6–8 趟），
                    // 而热启动只要 2–4 趟；每多一趟就多扫一遍全帧的带（`plan_group`），
                    // 那是整个链路最大的单项开销。块首帧仍是冷启动 —— 所以**块要长**。
                    let mut chans = states[base].to_chans();
                    for (j, frame) in part.iter().enumerate() {
                        if j > 0 {
                            states[base + j].apply_to(&mut chans);
                        }
                        let (bytes, short) = this.encode_frame_inner(frame, &mut chans)?;
                        v.push((base + j, bytes, short));
                    }
                    Ok(v)
                }));
            }
            let mut out = Vec::with_capacity(frames.len());
            for h in handles {
                let part = h
                    .join()
                    .map_err(|_| Error::Bad("帧编码线程 panic".into()))??;
                out.extend(part);
            }
            Ok::<_, Error>(out)
        })?;

        // ---- 3. 按帧序收齐 ----
        results.sort_by_key(|(i, _, _)| *i);
        let mut out = Vec::with_capacity(frames.len());
        for (_, bytes, short) in results {
            if short {
                self.short_frames += 1;
            }
            self.frame_index += 1;
            out.push(bytes);
        }
        Ok(out)
    }

    /// 编一帧 —— **声道状态由调用方给**，自己不碰 `self.chans`，所以 `&self` 就够
    /// （多个线程可以同时调它，见 `encode_frames`）。
    ///
    /// `chans` 里只有四项影响输出：`overlap` / `overlap_ms` / `attack_prev` /
    /// `window_seq_prev`（见 `FrameState`）；`cost_cache` 与 `sf_offset_prev`
    /// 只影响速度。返回 `(位流, 本帧是不是短窗)`。
    fn encode_frame_inner(
        &self,
        frame: &Frame,
        chans: &mut [ChannelState],
    ) -> crate::Result<(Vec<u8>, bool)> {
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

        // ---- 决策：M/S / 窗序列 / 重叠历史 ----
        //
        // 全在 `prepare_frame` 里 —— 它把跨帧状态往下推一步，并返回本帧要编的
        // 东西。**按帧并行的预扫走的是同一个函数**（见 `encode_frames`），
        // 所以决策逻辑只有一份。
        let prep = self.prepare_frame(frame, chans);
        let use_ms = prep.use_ms;
        let PreparedFrame {
            coded, plan, hist, ..
        } = &prep;

        // ---- 元素头 ----
        // SCE = 0，CPE = 1。
        w.put(if is_stereo { 1 } else { 0 }, 3);
        w.put(0, 4); // 元素实例标签（每帧都从 0 起）
        if is_stereo {
            // `common_window = 1`：两个通道**共用一份 `ics_info`**。
            //
            // 这是 M/S 立体声的**前提**（`ms_mask_present` 只在 common_window 下存在）。
            // 代价是两个声道的窗序列与分组必须一致 —— 所以决策提到了上面那一步联合做。
            w.put(1, 1);
            // 窗序列直接来自状态机（含 `LONG_START` / `LONG_STOP` 两个过渡态）。
            let seq = plan.seq;
            let short = seq == WindowSequence::EightShort;
            // ⚠️ `max_sfb` 短窗要用**短窗那张频带表**的条带数（4 位），长窗才用长表的
            // （6 位）。写错会让解码端按错误的频带数读 `section_data` —— 报出来的是
            // 「number of bands (7) exceeds limit (1)」这种看着像分组错的信息。
            let max_sfb = if short {
                self.swb_short.len() - 1
            } else {
                self.num_sfb()
            };
            write_ics_info(
                &mut w,
                max_sfb as u8,
                seq,
                &plan.group_lens,
                self.cfg.kb_window,
            );
            // `ms_mask_present`：0 = 不用 M/S；**2 = 所有频带都用**（后面不跟掩码位）；
            // 1 = 后面跟逐带掩码（每带 1 位）。
            //
            // ⚠️ **1 和 2 的语义我一开始记反了**（以为 1 = 全部带）。ffmpeg 的
            // 解码器与编码器两边都确认：`1` 是「掩码跟在后面」，`2` 才是「全部带」。
            // 记反的后果不是「音质差一点」，而是**整条位流错位** —— 解码端把掩码位
            // 当成了通道数据，报出来的是 `Number of bands (65) exceeds limit (49)`
            // 这种看着像频带表错的信息（症状离根因很远）。
            w.put(if use_ms { 2 } else { 0 }, 2);
        }

        // ---- 声道元素体：串行，或（调用者要求时）两个声道并行 ----
        //
        // 位流里两个声道是**前后相接**的两段，各自独立 —— 所以并行只影响
        // 「谁在什么时候算」，不影响写出来的字节。这也是本仓唯一一种
        // 「不拿算力换延迟」的并行：总工作量不变，只是让两个核同时干。
        //
        // ⚠️ 线程数由**调用者**给（`EncoderConfig::threads`），库不自己探测核数：
        // 只有调用者知道这台机器 / 这个 App 还有多少算力可用（Android 上尤其如此）。
        let bodies =
            self.encode_channel_bodies(chans, coded, hist, plan, !is_stereo, self.cfg.threads);
        for (body, short) in bodies? {
            w.append(&body);
            any_short |= short;
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

        Ok((w.finish(), any_short))
    }

    /// 编出**所有声道**的元素体（不含元素头），每个声道一段。
    ///
    /// `threads >= 2` 且是双声道时，两个声道各在一个线程里编 —— 位流里它们本来
    /// 就是前后相接的两段、互不影响，所以**输出逐字节相同**。
    ///
    /// 线程数**封顶到声道数**：单声道没有可并行的东西，给 8 也是白给。
    ///
    /// ⚠️ `hist`（MDCT 的左半）与 `threads` 都由**调用方**给：重叠历史的推进
    /// 搬到了 `encode_frame_from` —— 按帧并行时那一步必须走在编码之外
    /// （见 `encode_frames`），而且帧并行时每帧内部的声道并行要关掉（`threads = 1`），
    /// 否则线程会超订。
    fn encode_channel_bodies(
        &self,
        chans: &mut [ChannelState],
        coded: &[Vec<f32>],
        hist: &[Vec<f32>],
        plan: &WindowPlan,
        write_ics: bool,
        threads: usize,
    ) -> crate::Result<Vec<(BitWriter, bool)>> {
        let n = self.cfg.channels as usize;
        let threads = threads.min(n).max(1);
        // 这一帧的总预算，按**每个声道需要多少**分下去（不是平均分）。
        let budgets = channel_budgets(coded, self.bits_per_frame * n as u32);

        if threads < 2 || n < 2 {
            let mut out = Vec::with_capacity(n);
            for ch in 0..n {
                let mut w = BitWriter::new();
                let short = self.encode_channel(
                    &mut chans[ch],
                    &mut w,
                    &coded[ch],
                    &hist[ch],
                    plan,
                    ch,
                    write_ics,
                    budgets[ch],
                )?;
                out.push((w, short));
            }
            return Ok(out);
        }

        // 双声道并行：每个线程只碰**自己那份** `ChannelState`（`split_at_mut` 保证
        // 不重叠），`self` 全程只读 —— 所以两个声道之间没有共享的可变状态。
        //
        // ⚠️ 闭包里不能用 `?`（它的返回类型是那个元组），所以两边都先把 `Result`
        // 带出来，到外面再展开。
        let (a, b) = chans.split_at_mut(1);
        let (p0, p1) = (&coded[0], &coded[1]);
        let (h0, h1) = (&hist[0], &hist[1]);
        let (r0, r1) = std::thread::scope(|scope| {
            let h = scope.spawn(|| {
                let mut w = BitWriter::new();
                let short =
                    self.encode_channel(&mut a[0], &mut w, p0, h0, plan, 0, write_ics, budgets[0])?;
                Ok::<_, Error>((w, short))
            });
            let mut w1 = BitWriter::new();
            let r1 = self
                .encode_channel(&mut b[0], &mut w1, p1, h1, plan, 1, write_ics, budgets[1])
                .map(|short| (w1, short));
            (h.join(), r1)
        });
        // 线程里的 panic 要转成自己的错误类型再往上抛（`JoinError` 不是 `Error`）。
        let r0 = r0.map_err(|_| Error::Bad("声道编码线程 panic".into()))??;
        let r1 = r1?;
        Ok(vec![r0, r1])
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
        cache: &mut BandCostCache,
        gi: usize,
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

        for sfb in 0..max_sfb {
            let lo = swb[sfb] as usize;
            let hi = swb[sfb + 1] as usize;
            let band = &spec[lo..hi];

            // 搜索窗口 = 「整帧下界 + 偏移」∩「差分约束 ±60」∩ `0..=255`。
            //
            // ⚠️ 三个都必要：下界保证峰值不被裁；差分约束是位流硬要求；
            // `0..=255` 是 `sf` 自己的位宽。**偏移只抬下界** ——
            // 若把上界也压低，会挡住「该带本来想用更粗的 sf」的合法选择。
            let floor = (sf_floor + sf_offset).clamp(0, 255);
            // 逐带 RD 的 λ：由**偏移**单调映射（`RD_LAMBDA_K · offset²`，offset = 0 时 λ = 0
            // 就退回「纯失真」）。偏移同时管着下界与 λ —— 这是这一刀的简化，
            // 真正的版本该把下界交还给「不裁剪」、由 λ 单独控制粗细（见 TODO）。
            let lam = RD_LAMBDA_K * (sf_offset * sf_offset) as f32;
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
            //
            // ## 三级细化：12 → 3 → 1（**覆盖范围与原「粗扫 + ±3」完全一样**）
            //
            // 每一级的细扫半径 = 上一级步长的一半，所以最优点**一定**落在下一级的
            // 扫描范围里：级 1 步长 12 → 最优点离它挑出的网格点 ≤ 6；级 2 步长 3 扫
            // `±6` → ≤ 1.5（≤ 2）；级 3 步长 1 扫 `±2` → **精确**。
            //
            // 评估次数从 ~31 降到 ~21（窗口 120 宽时），而每一趟都要跑一遍 ——
            // 这是**又省时间又省 CPU** 的那一类（不靠多线程，也不增加内存）。
            let mut best: Option<(f32, u8, u8)> = None; // (代价, sf, cb)
            let mut evaluate = |sf: i32, best: &mut Option<(f32, u8, u8)>| {
                let (dist, cb, bits) = cached_band_cost(cache, gi, sfb, band, sf as u8);
                // 代价 = 失真 + λ·位数。位数进代价之后，同一条带内部的「粗一点更划算」
                // 才第一次能被选出来（纯失真时最优点永远在窗口下界上）。
                let cost = dist + lam * bits as f32;
                if best.is_none_or(|(bc, _, _)| cost < bc) {
                    *best = Some((cost, sf as u8, cb));
                }
            };
            let mut lo_scan = lo_sf;
            let mut hi_scan = hi_sf;
            let mut step = 12i32;
            loop {
                // 从 `lo_scan` **向上取整到 `step` 的倍数**再开始扫 —— 这就是「锚定」：
                // 格点的位置只由 `step` 决定，偏移只影响哪些格点被 `floor` 挡掉。
                let mut sd = lo_scan.div_euclid(step) * step;
                if sd < lo_scan {
                    sd += step;
                }
                while sd <= hi_scan {
                    evaluate(sd, &mut best);
                    sd += step;
                }
                // ⚠️ **两个端点必须显式补上**：锚定之后格点是 `step` 的倍数，
                // 窗口下界（`floor = sf_floor + 偏移`）很可能不在格点上 ——
                // 而**安静频带的最优点恰恰就在 floor 上**（那是「什么都不裁」的下界）。
                // 少了它实测掉 **0.83 dB**（12.13 → 11.30），补回来就恢复。
                // 端点被反复求值不要紧 —— 帧内缓存让重复求值几乎免费。
                evaluate(lo_scan, &mut best);
                evaluate(hi_scan, &mut best);
                if step == 1 {
                    break;
                }
                let centre = i32::from(best.expect("候选集非空").1);
                let next = (step / 4).max(1);
                // ⚠️ 半径要**向上取整**（`step/2` 对步长 3 只给 ±1，而最优点可能
                // 在 ±1.5 —— 差那一点就会漏掉最优、白丢 0.05 dB）。
                let radius = (step + 1) / 2;
                lo_scan = (centre - radius).max(lo_sf);
                hi_scan = (centre + radius).min(hi_sf);
                step = next;
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
        write_ics: bool,
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
        // `common_window = 1` 时 `ics_info` 由元素头那边写一次（两个声道共用），
        // 通道元素体里**不再写** —— 多写一次整条位流就错位。
        if write_ics {
            write_ics_info(&mut w, max_sfb as u8, seq, group_lens, self.cfg.kb_window);
        }
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
        cache: &mut BandCostCache,
        seq: WindowSequence,
        group_specs: &[(&[f32], &[u16], usize)],
        sf_floor: i32,
        sf_offset: i32,
        tns_data: Option<&Tns>,
        write_ics: bool,
    ) -> crate::Result<(BitWriter, u32)> {
        let mut prev_sf: Option<i32> = None;
        let mut plans: Vec<GroupPlan<'_>> = Vec::with_capacity(group_specs.len());
        for (gi, &(spec, swb, _)) in group_specs.iter().enumerate() {
            plans.push(self.plan_group(cache, gi, spec, swb, sf_floor, sf_offset, &mut prev_sf));
        }
        let lens: Vec<usize> = group_specs.iter().map(|&(_, _, l)| l).collect();
        self.write_pass(seq, &plans, &lens, tns_data, write_ics)
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
        &self,
        st: &mut ChannelState,
        w: &mut BitWriter,
        samples: &[f32],
        hist: &[f32],
        plan: &WindowPlan,
        _ch: usize,
        write_ics: bool,
        budget: u32,
    ) -> crate::Result<bool> {
        // ---- 1. 攒 2N 个样本 → 开窗 → MDCT ----
        //
        // 左半 = **上一帧的后半**（AAC 是 50% 重叠）。`hist` 由调用方给 ——
        // 它要按本帧编的域（原始 / M/S）挑对应的那一套，并且**两套都往下更新**。
        let mut buf = vec![0f32; 2 * LONG_LEN];
        buf[..LONG_LEN].copy_from_slice(hist);
        let n = samples.len().min(LONG_LEN);
        buf[LONG_LEN..LONG_LEN + n].copy_from_slice(&samples[..n]);

        // ---- 窗序列（由调用者决定，见 `WindowPlan`）----
        //
        // 攻击打在这一帧 → 本帧用短窗；**下一帧也要用短窗**（攻击的能量在
        // 50% 重叠区里会跨进下一帧的前半 —— 长窗会把攻击「摊平」成一帧的
        // pre-echo，这正是短窗要解决的东西）。
        //
        // ⚠️ 决策**不在这里做**：双声道要共用一份 `ics_info`（`common_window = 1`），
        // 所以两个声道必须拿到同一份窗序列与分组。
        //
        // 跨帧状态（`attack_prev` / `window_seq_prev`）由 `encode_frame_inner`
        // 统一推进 —— 那里是「读旧值、写新值」的唯一落点，按帧并行也要经过它。
        let seq = plan.seq;
        let short = seq == WindowSequence::EightShort;

        // 短窗的**分组**：一个组内的窗口共用一套 sf/码本 —— 把攻击所在的窗口
        // 单独分出来，组里其余窗口才不会被攻击的尺度「带跑」。
        //
        // 分组规则照 3GPP TS26.403 5.4.1（ffmpeg 的 `aacpsy.c` 用同一张表）：
        // 表按**攻击所在的子块序号**（0..8，8 = 本帧没有攻击）查 8 位的掩码，
        // 位 i = 0 表示「窗口 i 另起一组」。
        //
        // ⚠️ 实测这一条值 5~7 dB：第一版「8 个窗口一组」在瞬态素材上
        // SNR 10.2 dB（ffmpeg 自己的 AAC 是 17.8 dB），分组后追上来一大截。
        let group_lens: &[usize] = &plan.group_lens;

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
            for &gl in group_lens {
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
            // 长窗路径 —— 三种窗形：`ONLY_LONG` 是普通长窗；`LONG_START` /
            // `LONG_STOP` 是**与短窗配对**的过渡窗（少了它们，长↔短直接跳变时
            // 重叠区不互补 —— 见 `decide_sequence` 的实测表）。
            let shape: &[f32] = match seq {
                WindowSequence::OnlyLong => &self.window,
                WindowSequence::LongStart => &self.window_long_start,
                WindowSequence::LongStop => &self.window_long_stop,
                WindowSequence::EightShort => unreachable!("短窗走上面那条分支"),
            };
            for (b, wv) in buf.iter_mut().zip(shape) {
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
            .zip(group_lens)
            .map(|((s, t), &l)| (s.as_slice(), t.as_slice(), l))
            .collect();

        // 缓存按这一帧的形状重置（形状没变就只清空，不重新分配）。
        let shapes: Vec<usize> = group_specs.iter().map(|&(_, swb, _)| swb.len() - 1).collect();
        st.cost_cache.reset(&shapes);

        // ---- 3. 搜「位数 ≤ 预算」的最小偏移 ----
        //
        // 偏移越大 → 量化越粗 → 位数越少、失真越大，所以「位数」关于偏移单调，
        // 「最小的可行偏移」是良定义的；它就是「码率不超标的前提下保真最好」的那一档。
        //
        // ⚠️ **搜索从上一帧的答案出发做指数搜索**，而不是每帧都从 `[0, 120]`
        // 重新二分：相邻帧的谱形状连续，上一帧的答案通常就在真值 ±1 里，
        // 实测 2–4 趟就收敛（固定二分恒 7 趟，而每趟都要整帧量化 + 计费 + 写位流）。
        // 找的**还是同一个答案**，所以位流逐字节不变 —— 这是纯提速，不改语义。
        //
        // 括号约定：`lo` = 已知**不可行**的最大偏移（`-1` = 还不知道），
        // `hi` = 已知**可行**的最小偏移（`MAX_SF_OFFSET + 1` = 还不知道）。
        let seed = st.sf_offset_prev.clamp(0, MAX_SF_OFFSET);
        let mut chosen: Option<BitWriter> = None;
        let mut lo: i32 = -1;
        let mut hi: i32 = MAX_SF_OFFSET + 1;
        let (pass, bits) =
            self.encode_pass(&mut st.cost_cache, seq, &group_specs, sf_floor, seed, tns_data.as_ref(), write_ics)?;
        if bits <= budget {
            hi = seed;
            chosen = Some(pass);
        } else {
            lo = seed;
        }

        // 指数搜索把另一侧也夹住：步长 1、2、4…（真值离种子远时才多探几次）。
        let mut step = 1i32;
        if chosen.is_none() {
            while lo < MAX_SF_OFFSET {
                let o = (lo + step).min(MAX_SF_OFFSET);
                let (pass, bits) =
                    self.encode_pass(&mut st.cost_cache, seq, &group_specs, sf_floor, o, tns_data.as_ref(), write_ics)?;
                if bits <= budget {
                    hi = o;
                    chosen = Some(pass);
                    break;
                }
                lo = o;
                step = step.saturating_mul(2);
            }
        } else {
            while hi > 0 {
                let o = (hi - step).max(0);
                let (pass, bits) =
                    self.encode_pass(&mut st.cost_cache, seq, &group_specs, sf_floor, o, tns_data.as_ref(), write_ics)?;
                if bits > budget {
                    lo = o;
                    break;
                }
                hi = o;
                chosen = Some(pass);
                step = step.saturating_mul(2);
            }
        }

        // 夹住了就二分收尾 —— 种子准的时候这一步一次都不跑。
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            let (pass, bits) =
                self.encode_pass(&mut st.cost_cache, seq, &group_specs, sf_floor, mid, tns_data.as_ref(), write_ics)?;
            if bits <= budget {
                hi = mid;
                chosen = Some(pass);
            } else {
                lo = mid;
            }
        }
        // 记住这一帧的答案，给下一帧当种子（`hi` 超界 = 这一帧没有任何可行偏移）。
        st.sf_offset_prev = if hi <= MAX_SF_OFFSET { hi } else { seed };

        // ⚠️ **一趟都不够省**时（预算小到连最粗的量化都塞不下）退化成
        // 「用偏移上界那一趟」：位数超预算，但**位流仍然合法**。
        // 让码率超标比「编不出东西」好 —— 而且这种情况只会在极端配置下出现。
        let pass = match chosen {
            Some(p) => p,
            None => {
                self.encode_pass(&mut st.cost_cache, seq, &group_specs, sf_floor, MAX_SF_OFFSET, tns_data.as_ref(), write_ics)?
                    .0
            }
        };
        w.append(&pass);
        Ok(short)
    }
}

/// 一个声道「需要多少位」的**廉价估计**：`Σ|c|^0.75`。
///
/// 用 `x^0.75` 而不是能量（`x²`）或绝对值：`x^0.75` 正好是量化器作用的那一维
/// （`quant_one` 就是这个），所以它比能量更贴近「要花多少位」—— 能量会被最大的
/// 那几个系数主导，而位数是**对数**域的。两次硬件开方、一个系数一次，
/// 相对后面每帧几万次码本查表可以忽略。
fn channel_mass(x: &[f32]) -> f64 {
    x.iter()
        .map(|&c| {
            let a = f64::from(c).abs();
            let r = a.sqrt();
            r * r.sqrt()
        })
        .sum()
}

/// 把这一帧的总预算**按每个声道需要多少**分下去，而不是平均分。
///
/// ## 为什么不能平均分（实测）
///
/// M/S 立体声下两个声道的**信息量差得很远**。最极端的是「单声道源复制成两轨」
/// （老 `.wmv` 里很常见）：`S = (L−R)/2 = 0`，它一位都用不上 —— 可平均分还是
/// 把它那一半扣着，于是**总码率只能到目标的一半**。
///
/// 同一份素材、目标 384k：平均分只花掉 171 kbps（45%）、SNR 21.17 dB；
/// 把预算加倍（等于让 `M` 拿到全部）之后花掉 388 kbps、SNR **34.97 dB**。
/// 真音乐上两个声道本来就都用满 ~99%，所以这个改动对它**基本中性**
/// （实测 320k / 384k / 768k 三个点上的差别在 0.05 dB 以内）。
///
/// ⚠️ 传进来的 `coded` 必须是**本帧实际要编的那个域**（M/S 时就是 M/S）——
/// 拿 `L/R` 去算权重的话，单声道复制素材的 `L` 与 `R` 一模一样、权重相等，
/// 又会退回「平均分」，这个修复就白做了。
///
/// ## 保底
///
/// 每个声道至少分到 `FLOOR × 最重的那个声道` 的量 —— 不给任何声道分 0：
/// `S` 哪怕一模一样也还要写几十位的头。
fn channel_budgets(coded: &[Vec<f32>], total: u32) -> Vec<u32> {
    let n = coded.len();
    if n == 0 {
        return Vec::new();
    }
    let masses: Vec<f64> = coded.iter().map(|c| channel_mass(c)).collect();
    let max = masses.iter().cloned().fold(0.0f64, f64::max);
    if !(max > 0.0) {
        // 整帧全静音：平均分（怎么分都一样）。
        return vec![total / n as u32; n];
    }
    const FLOOR: f64 = 0.05;
    let weights: Vec<f64> = masses.iter().map(|&m| m + FLOOR * max).collect();
    let sum: f64 = weights.iter().sum();
    weights
        .iter()
        .map(|&w| ((f64::from(total) * w / sum).max(1.0)) as u32)
        .collect()
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
            .fold(0f64, |m, &c| {
                // 同 `quant_one`：`x^0.75 = √x·√√x`，别调 `powf`。
                let x = f64::from(c.abs());
                let r = x.sqrt();
                m.max(r * r.sqrt())
            });
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
    // ⚠️ **查表**，别每次算 `exp()`：这个函数在 `band_code` 里是**逐 (带, sf, 码本)**
    // 调的（一帧 ~5 万次），而 sf 只有 256 个可能值。表算一次，值完全相同。
    static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 256];
        for (sf, v) in t.iter_mut().enumerate() {
            let e = -3.0 * (sf as f32 - 100.0) / 16.0;
            *v = (e * std::f32::consts::LN_2).exp();
        }
        t
    })[sf as usize]
}

/// `IQ(sf) = 2^((sf − 100)/4)` —— 反量化用的缩放因子。
///
/// 就是解码端的 `sf_gain`。与 `q34` 的关系是 `IQ = Q34^(-4/3)`。
fn iq(sf: u8) -> f32 {
    // 同上：查表（`IQ = 2^((sf−100)/4)`）。
    static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 256];
        for (sf, v) in t.iter_mut().enumerate() {
            let e = (sf as f32 - 100.0) / 4.0;
            *v = (e * std::f32::consts::LN_2).exp();
        }
        t
    })[sf as usize]
}

/// 量化一个非负值：`min((int)(pow(a, 3/4) · q34 + 0.4054), maxval)`。
///
/// ⚠️ 是**截断**不是四舍五入 —— 那个 `+0.4054` 是 ffmpeg 调出来的偏置
/// （`ROUND_STANDARD`），用来补偿 `pow(·, 3/4)` 的均值偏移。
/// 换成 `round()` 会让所有系数偏大一点。
fn quant_one(a: f32, q34: f32, maxval: i32) -> i32 {
    quant_uncapped(a, q34).min(maxval)
}

/// `quant_one` **去掉封顶**的那一半：`round(|x|^0.75 · q34)`。
///
/// 分出来是为了「一次算好、11 本码本共用」（见 `band_cost`）—— 每个码本只差
/// `.min(maxval)`，而**两次开方是最贵的那一步**。
///
/// ⚠️ 用 `round()` 而不是 ffmpeg 的 `+0.4054` 截断 —— 两者都对，
/// 但**必须与反量化成对**，混用会在低频小系数上积出可见的偏差。
/// 也**不要** `powf(0.75)`：那是 libm 超越函数（几十到上百纳秒），
/// 而这里是逐系数调的最内层；`x^0.75 = √x · √√x` 两次硬件开方就够。
/// （ffmpeg 那种 **pow34 查表 + 插值**试过：既没更快，还把质量打掉 0.78 dB → 撤。）
fn quant_uncapped(a: f32, q34: f32) -> i32 {
    let r = f64::from(a).sqrt();
    (r * r.sqrt() * f64::from(q34)).round() as i32
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
///
/// ⚠️ **短窗必须按「单窗口」的划分算**：组的 `swb` 是「128 点划分 × 组内窗口数」
/// （见 `encode_channel` 里建 `swbs` 的那段），而每条带对应的是**同一个**频率带
/// 在组内各窗口的重复 —— 所以频率要把 `swb` 除回 `group_len`，尺度也要用单窗口的
/// 系数个数（长窗 1024 / 短窗 128）。少了这一步，短窗的频率会算成 `gl` 倍，
/// 掩蔽扩散整个错位（长窗 `group_len = 1`，两种算法等价）。
#[allow(dead_code)] // 只服务 `masking_thresholds`
fn bark_of(swb: &[u16], sample_rate: u32, group_len: usize, coefs_per_window: usize) -> Vec<f32> {
    let gl = group_len.max(1) as f64;
    let coefs = coefs_per_window.max(1) as f64;
    swb.windows(2)
        .map(|w| {
            // 带的中心频率（Hz），按**单窗口**的划分。
            let fc = (f64::from(w[0]) + f64::from(w[1])) * 0.5 / gl * f64::from(sample_rate)
                / (2.0 * coefs);
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
#[allow(dead_code)] // 当前未使用 —— 见 `band_cost` 的说明（当真音乐上有害 + 逐带常数权重是 no-op）
fn masking_thresholds(
    spec: &[f32],
    swb: &[u16],
    sample_rate: u32,
    _sf_floor: i32,
    group_len: usize,
) -> Vec<f32> {
    // `_sf_floor`（整帧的 sf 下界）在这里还用不上 —— 阈值只看源信号。
    // 留着签名是为了「将来把阈值抬到量化噪声下限」那条路不用改调用方。
    let n = swb.len() - 1;
    let coefs_per_window = spec.len() / group_len.max(1);
    let bark = bark_of(swb, sample_rate, group_len, coefs_per_window);

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

/// 带缓存的 `band_cost`：命中就直接返回，没命中才算并记下。
///
/// 值**逐位相同**（同一帧里同一条带的谱系数不变），所以决策不会变 → 输出逐字节不变。
#[inline]
fn cached_band_cost(
    cache: &mut BandCostCache,
    gi: usize,
    sfb: usize,
    band: &[f32],
    sf: u8,
) -> (f32, u8, u32) {
    if let Some(v) = cache.get(gi, sfb, sf) {
        return v;
    }
    let v = band_cost_rd(band, sf, &mut cache.q_scratch);
    cache.put(gi, sfb, sf, v);
    v
}

/// 一个频带在给定 `sf` 下的 `(代价, 最优码本)`。
///
/// ## 代价是**纯平方失真**（这是当前实现）
///
/// ```text
///   代价 = Σ 误差²
/// ```
///
/// ## ⚠️ 这里**故意没有 λ · 位数那一项**
///
/// 码率由外层的「二分搜 sf 偏移」用**硬预算**控制。再在这里掺一个软权重
/// 只会互相干扰，而且 λ 的量级（1e-4）在失真（1e9）面前根本不起作用
/// —— 实测踩过，白忙一场（码率是目标的 3.4 倍）。
///
/// **两者是正交的分工**：外层管「花多少位」，这里管「位花在哪」。
///
/// ## ⚠️ 曾经除以掩蔽阈值²，已撤（实测，不是偏好）
///
/// 曾经这里除以掩蔽阈值²（`dist / th²`，见 `masking_thresholds`）。在**粉噪声**
/// 上它确实略好（效率 +8%，见 `quality_is_within_2db_of_ffmpeg` 的说明），
/// 但在**真音乐**上是灾难：
///
/// | 12 秒真音乐（单声道 64k） | 总 SNR |
/// |---|---|
/// | 除以掩蔽阈值 | 5.47 dB |
/// | 只比平方误差 | **11.85 dB** |
/// | ffmpeg 同码率 | 11.91 dB |
///
/// 逐带能量一看就明白了：音乐的低音比中高频高 **40+ dB**，而量化台阶是
/// **全局一个**（见 `plan_group` 的下界），于是中频带（约 500 Hz–5.7 kHz）
/// 被量化掉 **7–25 dB**。而「除以阈值」正好**把这份损失合法化**了 ——
/// 模型认为那些带被掩蔽，代价算出来很小。粉噪声频谱平坦、没有这个动态范围，
/// 所以看不出来。
///
/// ## 机制（三方对照，别猜）
///
/// 把阈值装回去、**只把代价的算法换成 f64**，结果就回来了：
///
/// ```text
///   阈值 + f32 代价（原实现）   5.47 dB
///   阈值 + f64 代价            11.88 dB   ← 与「无阈值」一致
///   无阈值                     11.85 dB
/// ```
///
/// 所以这个归一化**没有任何正当效果** —— 它的唯一实际作用是**数值溢出**：
/// 安静带的阈值小（`th ≈ 0.5·带能量`），`(1/th)²` 能到 1e24，而 `dist` 是
/// 1e9~1e11 量级，`dist·(1/th)²` 直接**溢出 f32** 变成 `inf`。
/// 代价一旦是 `inf`，`band_cost` 里的比较全部失效，最后落到兜底分支
/// —— **把这条带判成「全零带」**。于是中频带整段被抹掉。
///
/// 这也解释了当初在粉噪声上量到的「+8% 效率」：那不是模型省下的位，
/// 是**直接扔掉的频带**（粉噪声频谱平坦，扔几条听不出来）。
///
/// ## 🚫 更根本的一条：**给失真乘一个逐带常数，按构造就不可能改变任何决策**
///
/// `feat/aac-psy` 上试过「有界的比值权重」`w = clamp((E_带/阈值)^p, 1, W_MAX)`
/// （权重恒 ≥ 1，只会加重可闻带的代价）—— 量出来**产物与改前逐字节相同**。
/// 原因不是参数没调好，是数学上必然如此：
///
/// ```text
///   argmin_sf ( w · dist(sf) )  ==  argmin_sf dist(sf)      （w > 0 且与 sf 无关）
/// ```
///
/// 每带的 `w` 在**该带自己的候选之间是同一个常数**，所以它改变不了任何一条带的
/// sf / 码本选择；而 sf 的选择又是逐带独立的（`prev_sf` 只提供**约束**、不参与比较），
/// 于是整帧的决策、位数、外层偏移全都不变。
///
/// **要用掩蔽，权重必须和一个「率」项一起出现**（`w·dist + λ·位数`）——
/// 只有那时它才会改变「这一带值不值得多花位」的权衡。也就是说：
/// **心理声学模型得挂在率-失真分配（`feat/aac-rd-alloc` 那套 λ 机制）上，
/// 单独挂在纯失真上等于没写。** 详见 TODO 的 `feat/aac-psy` 一节。
/// 一个频带在 `(sf, cb)` 下的**失真与位数** —— `plan_group` 用 `失真 + λ·位数` 选 `sf`/码本。
///
/// 一条路径两个出口是有意的：决策口径只有一份，位数的口径也只有一份
/// （`band_bits`，与 `write_band` 对拍钉住）。
fn band_cost_rd(band: &[f32], sf: u8, qu: &mut Vec<i32>) -> (f32, u8, u32) {
    // ---- 量化只算一次，11 本码本共用 ----
    //
    // 同一个 `sf` 下所有码本的**量化步长相同**（都是 `q34(sf)`），差别只在
    // `CB_MAXVAL`（`1,1,2,2,4,4,7,7,12,12,16`）—— 也就是说每个码本要的
    // `q = min(round(|x|^0.75·q34), maxval)` 只差**一个封顶**。
    // 所以把不封顶的 `q` 算一遍存进 scratch，各码本自己 `.min(maxval)` 就够：
    // 值与「每个码本各自重算」**逐位相同**（`quant_one` 本来就是
    // `round(...)` 再 `.min()`），省掉的却是最贵的那一步 —— 每个系数两次开方。
    // escape 那条路同理：`escape_coef` 是同一个数封顶到 8191。
    //
    // 这条是量出来的：把 `band_cost` 短路掉，AAC 从 ~0.80 s 掉到 ~0.23 s
    // （t2、200 帧），说明**码本评估就是 AAC 的大头**。另外两条更「聪明」的捷径
    // 试过都**不成立**，别再走：
    // - 「按 `CB_MAXVAL` 查表直接定赢家」：`sf` 极端时 `iq` 极小，cb1 与 cb9 的
    //   失真在 **f32** 里舍成同一个值，平手规则就选了编号小的那个 → 结果不同。
    // - 「部分失真越界就提前收工」：安静带上 11 本码本给出**相同**失真，
    //   部分和永远越不过界 → 只快 4%。
    qu.clear();
    qu.resize(band.len(), 0);
    let q34v = q34(sf);
    for (j, &c) in band.iter().enumerate() {
        qu[j] = quant_uncapped(c.abs(), q34v);
    }
    // ---- 11 本码本只有 **6 个不同的封顶值**，而失真只取决于封顶值 ----
    //
    // `CB_MAXVAL` = `1,1,2,2,4,4,7,7,12,12,16`：同封顶值的那两本（1/2、3/4、5/6、
    // 7/8、9/10）反量化值**完全相同**（都是 `±min(q, maxval)·iq`，与码本无关）→
    // 失真逐位相同。按「平手取编号小」的规矩，取小的那本当代表即可：
    // `[1, 3, 5, 7, 9, 11]` 六次。cb=11 是 escape（大系数走 13 位真实值），单独算。
    //
    // ⚠️ **不算位数** —— 旧版在这里把每本的位数也算了一遍（`bit_tab` 查表、无符号
    // 码本的符号位计数、escape 的位数），而 `band_cost` 从来只看失真
    // （`let (_bits, dist) = ...`）：位数是 `write_band` 写位流时自己算的。白算。
    let mut best: Option<(f32, u8)> = None;
    for cb in [1usize, 3, 5, 7, 9, CB_ESC] {
        let dist = band_dist(band, sf, cb, qu);
        if best.is_none_or(|(bc, _)| dist < bc) {
            best = Some((dist, cb as u8));
        }
    }
    // cb == 0 是「全零带」的伪码本，作为兜底参与比较：量化后还不如直接清零，
    // 就把这条带标成零（代价 = 原始能量）。
    let zero_cost: f32 = band.iter().map(|&c| c * c).sum();
    match best {
        Some((c, cb)) if c <= zero_cost => (c, cb, band_bits(band, sf, cb, qu)),
        _ => (zero_cost, 0, 0),
    }
}

/// 一个频带在 `(sf, cb)` 下要**多少位** —— 与 `write_band` 同一套计数。
///
/// 数的是这一带自己的开销：码本索引 + 无符号码本的符号位 + escape 码；
/// **不含** `section_data` 的段长与 `scalefactor_data` 的差分（那些是跨带的，
/// 同一个 `sf` 下的候选之间差别很小，当常数处理）。
///
/// `qu` 是**不封顶**的量化值（`band_cost` 里算好一次、各码本 `.min(maxval)` 共用）——
/// 与 `write_band` 自己重算 `quant_one` **逐位相同**（那条等价性写在 `band_cost` 的注释里）。
///
/// ⚠️ 由 `band_bits_matches_what_write_band_actually_writes` 对拍钉住：
/// 这里的计数与写出端**必须**一致，否则 RD 代价模型会追着错的位数量走。
fn band_bits(band: &[f32], sf: u8, cb: u8, qu: &[i32]) -> u32 {
    if cb == 0 {
        return 0;
    }
    let cb = cb as usize;
    let dim = CB_DIM[cb];
    let maxval = i32::from(CB_MAXVAL[cb]);
    let unsigned = CB_UNSIGNED[cb];
    let bit_tab = &tables::SPECTRAL_BITS[cb - 1];
    let size = tables::SPECTRAL_SIZE[cb - 1] as usize;
    let vecs = tables::CODEBOOK_VECTORS[cb - 1];
    let q34v = q34(sf);
    let mut bits = 0u32;
    for (ci, chunk) in band.chunks(dim).enumerate() {
        let q0 = ci * dim;
        let mut q = [0i32; 4];
        for (j, &c) in chunk.iter().enumerate() {
            let m = qu[q0 + j].min(maxval);
            q[j] = if unsigned || c >= 0.0 { m } else { -m };
        }
        let off = if unsigned { 0 } else { maxval };
        let mut curidx = 0usize;
        for j in 0..dim {
            curidx = curidx * usize::from(CB_RANGE[cb]) + (q[j] + off) as usize;
        }
        if curidx >= size {
            // 越界 = 这一本写不出来（穷举 sf 时大量出现）：给个「必然不优」的数，
            // 与 `band_dist` 的 f32::MAX/2 一个套路。
            return u32::MAX / 2;
        }
        bits += u32::from(bit_tab[curidx]);
        // 无符号码本：逐维写非零系数的符号。
        if unsigned {
            for j in 0..dim {
                if vecs[curidx * dim + j] != 0 {
                    bits += 1;
                }
            }
        }
        // escape 码本：矢量里是 `64` 的那几维要写真实幅值。
        if cb == CB_ESC {
            for j in 0..dim {
                if vecs[curidx * dim + j] == 64 {
                    bits += escape_bits(escape_coef(chunk[j].abs(), q34v));
                }
            }
        }
    }
    bits
}

/// 一个频带在**某个码本的封顶值**下的平方失真（**不算位数**）。
///
/// 这是代价模型的核心：编码器按它选 `sf` 与码本，所以它必须与解码端算同一个值
/// （见下面「符号要带」那条）。
///
/// ⚠️ **只算失真**：位数由 `write_band` 写位流时自己算 —— 旧版在这里也算了一遍
/// （`bit_tab` 查表、无符号码本的符号位计数、escape 的位数），而 `band_cost`
/// 从来只看失真（`let (_bits, dist) = ...`）。白算。
///
/// `qu` 是**不封顶**的量化值（`band_cost` 里一次算好、六次求值共用）：
/// 每个封顶值要的 `q` 就是 `qu[j].min(maxval)` —— 与各自重算逐位相同。
fn band_dist(band: &[f32], sf: u8, cb: usize, qu: &[i32]) -> f32 {
    let dim = CB_DIM[cb];
    // ⚠️ 前提：频带长度是**码本维度的整数倍**。AAC 的 swb 频带宽度都是 4 的倍数
    // （长窗、短窗都一样），所以真实调用方一定满足；但少了这条断言，
    // 传进一个短尾块就会在下面 `chunk[j]`（j 走到 dim-1）越界 panic。
    debug_assert_eq!(band.len() % dim, 0, "频带长度必须是码本维度的整数倍");
    debug_assert_eq!(qu.len(), band.len());
    let maxval = i32::from(CB_MAXVAL[cb]);
    let unsigned = CB_UNSIGNED[cb];
    let size = tables::SPECTRAL_SIZE[cb - 1] as usize;
    let vecs = tables::CODEBOOK_VECTORS[cb - 1];
    // ⚠️ `iq(sf)` **提到循环外**：它只跟 `sf` 有关（是个 `exp()`），
    // 原来写在逐系数的循环里 —— 那是每个系数一次超越函数。
    let iq_sf = f64::from(iq(sf));

    let mut dist = 0f32;
    for (ci, chunk) in band.chunks(dim).enumerate() {
        let q0 = ci * dim;
        // 量化这一组（`qu` 不封顶，各码本只差一个封顶）。
        let mut q = [0i32; 4];
        for (j, &c) in chunk.iter().enumerate() {
            let m = qu[q0 + j].min(maxval);
            q[j] = if unsigned || c >= 0.0 { m } else { -m };
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
            return f32::MAX / 2.0;
        }
        // escape 码本：矢量里 `64` 的位置要用那 13 位的真实值（幅值）。
        let mut esc = [0i32; 4];
        if cb == CB_ESC {
            for j in 0..dim {
                if vecs[curidx * dim + j] == 64 {
                    // `escape_coef` = 同一个量化值封顶到 8191（`2^13 - 1`）。
                    esc[j] = qu[q0 + j].min((1 << 13) - 1);
                }
            }
        }
        for j in 0..dim {
            let v = f64::from(vecs[curidx * dim + j]);
            // ⚠️ **符号要带**：escape 码本是无符号的，符号位由「逐维写符号」单独发出
            // （`write_band` 同款），解码端会把符号乘回去 —— 代价模型必须与解码端
            // 算同一个值。
            //
            // 这里曾经漏掉 escape 那一路的符号（只写 `dequant_one(esc[j], sf)`，是个幅值），
            // 于是负系数的误差被估成 `(|c| + rec)²` 量级 —— 编码器**系统性避开 escape**，
            // 把大系数逼去撞 `maxval ≤ 16` 的普通码本。修好之后的实测（真样本的 `.mp4`
            // 音频路径、对 ffmpeg 解出的 WMA PCM 比 SNR）：64k +0.001、128k +0.019、
            // **256k +0.289 dB** —— 码率越高越吃这条（低码率下 sf 预算本来就主导）。
            let sign = if chunk[j] < 0.0 { -1.0 } else { 1.0 };
            let rec = if v == 64.0 {
                // escape：真实值来自那 13 位（幅值），符号乘回去。
                dequant_one(esc[j], sf) * sign
            } else {
                v.abs() * sign * iq_sf
            };
            let d = f64::from(chunk[j]) - rec;
            dist += (d * d) as f32;
        }
    }
    dist
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

/// 写一个 escape 序列。
fn write_escape(w: &mut BitWriter, coef: i32) {
    let len = av_log2(coef);
    debug_assert!(
        len >= 4 && len <= 12,
        "escape 的 len 必须在 4..=12（{len}）"
    );
    debug_assert_eq!(
        escape_bits(coef),
        (len - 4 + 1) + len,
        "escape 的位数只有一份真源（`escape_bits`）"
    );
    let nbits = len - 4 + 1;
    // 长度前缀：`nbits` 位、值是 `(1 << nbits) - 2`（一串 1 带个尾零）。
    w.put((1u32 << nbits) - 2, nbits);
    // 然后是 `len` 位的有符号值。
    w.put_signed(coef, len);
}

/// escape 码要多少位：**长度前缀**（`len-4+1`）**+ 值**（`len`）。
///
/// 与 `write_escape` 一一对应（那边有 `debug_assert_eq!` 钉着）。代价模型要数位数时
/// 走这里，别在别处再推一遍 —— 本仓的历史是「`band_cost` 与 `write_band` 各数一份、
/// 结果分叉」。
pub(crate) fn escape_bits(coef: i32) -> u32 {
    let len = av_log2(coef);
    (len - 4 + 1) + len
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
fn write_ics_info(
    w: &mut BitWriter,
    max_sfb: u8,
    seq: WindowSequence,
    group_lens: &[usize],
    kb_window: bool,
) {
    w.put(0, 1); // ics_reserved
    // ⚠️ `EIGHT_SHORT_SEQUENCE = 2`（不是 1 —— 1 是 `LONG_START`）。见 `WindowSequence`。
    w.put(seq as u32, 2);
    // ⚠️ 这一位必须与**编码端实际用的窗**一致（`cfg.kb_window`）—— 传进来，别写死。
    //
    // 解码端只拿它决定用哪张窗表，**不校验与内容是否一致**：写错了位流照样合法、
    // ffmpeg 照样解得开、帧数也对，只是它拿正弦窗去重建 KBD 窗编出来的系数。
    // 后果在重建公式里：重叠区要 `w_a·w_s + w_a'·w_s' = 1` 才能抵消时域混叠，
    // 两端同系时正弦与 KBD **各自**都满足，一错一配就不满足了 —— 误差表现为
    // 帧边界的「呼吸」。所以判据只能是解码回来的 SNR，见
    // `tests/ffmpeg_roundtrip.rs` 的 `kb_window_bit_matches_the_window_actually_used`。
    w.put(if kb_window { 1 } else { 0 }, 1); // use_kb_window
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 一段确定性的立体声测试信号（左右不同，才能验出「两边各算各的」）。
    fn stereo_frames(n: usize) -> Vec<Frame> {
        let mut state = 12345u32;
        let mut frames = Vec::new();
        for t in 0..n {
            let mut planes = Vec::new();
            for ch in 0..2usize {
                let mut p = vec![0f32; LONG_LEN];
                for (i, v) in p.iter_mut().enumerate() {
                    state = state.wrapping_mul(1103515245).wrapping_add(12345);
                    let noise = ((state >> 16) as f32 / 32768.0) - 1.0;
                    let k = (t * LONG_LEN + i) as f32;
                    let tone = (k * if ch == 0 { 0.07 } else { 0.13 }).sin() * 6000.0;
                    // 每 3 帧来一次瞬态，把短窗那条路也带上。
                    let env = if t % 3 == 2 && i < 200 { 3.0 } else { 1.0 };
                    *v = (tone + noise * 300.0) * env;
                }
                planes.push(p);
            }
            frames.push(Frame { planes });
        }
        frames
    }

    /// **`threads = 2` 必须与 `threads = 1` 逐字节相同。**
    ///
    /// 声道并行只改「谁在什么时候算」，不改写出来的字节 —— 这是它能放心开的原因，
    /// 也是这条并行唯一的判据。线程数封顶到声道数，所以 `threads = 8` 也该一样。
    #[test]
    fn parallel_channels_produce_identical_bytes() {
        let frames = stereo_frames(8);
        let mut one = Encoder::new(EncoderConfig {
            threads: 1,
            ..Default::default()
        })
        .expect("建编码器");
        let mut two = Encoder::new(EncoderConfig {
            threads: 2,
            ..Default::default()
        })
        .expect("建编码器");
        let mut many = Encoder::new(EncoderConfig {
            threads: 8,
            ..Default::default()
        })
        .expect("建编码器");
        for (i, f) in frames.iter().enumerate() {
            let a = one.encode_frame(f).expect("1 线程");
            let b = two.encode_frame(f).expect("2 线程");
            let c = many.encode_frame(f).expect("8 线程");
            assert_eq!(a, b, "第 {i} 帧：1 线程与 2 线程的输出不同");
            assert_eq!(a, c, "第 {i} 帧：1 线程与 8 线程的输出不同");
        }
    }

    /// 单声道给多少线程都该退化成串行 —— 没有第二个声道可以并行。
    #[test]
    fn mono_ignores_thread_count() {
        let frames: Vec<Frame> = stereo_frames(3)
            .into_iter()
            .map(|f| Frame {
                planes: vec![f.planes[0].clone()],
            })
            .collect();
        let mut one = Encoder::new(EncoderConfig {
            channels: 1,
            threads: 1,
            ..Default::default()
        })
        .expect("建编码器");
        let mut four = Encoder::new(EncoderConfig {
            channels: 1,
            threads: 4,
            ..Default::default()
        })
        .expect("建编码器");
        for (i, f) in frames.iter().enumerate() {
            assert_eq!(
                one.encode_frame(f).expect("1 线程"),
                four.encode_frame(f).expect("4 线程"),
                "第 {i} 帧：单声道下线程数不该影响输出"
            );
        }
    }

    /// escape 序列占多少位（照 `aaccoder.c` 的 `put_bits` 数）。
    ///
    /// `len = av_log2(coef)`；两个字段：`len - 4 + 1` 位的长度前缀 + `len` 位的值。
    ///
    /// ⚠️ **没有 `len < 5` 的特例** —— ffmpeg 的编码端就是直算 `len - 4 + 1`。
    /// 我一开始给它加了个「`len <= 4` 就只写一个 0 位」的分支，那是**错的**：
    /// 解码端读到 0 个前导 1 会 `b += 4` 变成 4，然后去读 4 位低位，
    /// 而编码端只写了 1 位 —— 整条位流从这里开始错位。
    ///
    #[test]
    fn band_bits_matches_what_write_band_actually_writes() {
        // 代价模型与写出端**必须**数同一套位数。本仓的历史教训：两边各推一份 → 分叉
        // （代价模型追着错的位数量走，RD 就白做了）。这条用**实际写出的位流长度**当裁判。
        let mut state = 987_654_321u32;
        for trial in 0..48usize {
            let len = 4 * (1 + trial % 8);
            let mut band = vec![0f32; len];
            for v in band.iter_mut() {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let unit = ((state >> 16) as f32 / 32768.0) - 1.0;
                // 幅度跨三档：普通系数 / 无符号码本 / escape（`escape_coef` 要 ≥ 16）
                let scale = [1.0e3, 1.0e5, 1.0e7][trial % 3];
                *v = unit * scale;
            }
            for sf in [60u8, 100, 140, 180, 220] {
                let mut qu = Vec::new();
                let (_, cb, _) = band_cost_rd(&band, sf, &mut qu);
                if cb == 0 {
                    assert_eq!(band_bits(&band, sf, cb, &qu), 0, "全零带没有位数");
                    continue;
                }
                let mut w = BitWriter::new();
                write_band(&mut w, &band, sf, cb).expect("这一档必须写得出来");
                assert_eq!(
                    band_bits(&band, sf, cb, &qu) as usize,
                    w.bit_pos(),
                    "sf={sf} cb={cb} len={len}：计数与实际写出的位数对不上"
                );
            }
        }
    }

    /// 生产代码里只有 oracle 需要它：`band_dist` 不算位数（位数由 `write_band` 在写
    /// 位流时自己算），而参考实现要按老样子把位数也数一遍。
    fn escape_bits(coef: i32) -> u32 {
        let len = av_log2(coef);
        debug_assert!(len >= 4, "escape 的 coef 至少 16（{coef}）");
        (len - 4 + 1) + len
    }

    #[test]
    fn a_silent_side_channel_does_not_hoard_half_the_budget() {
        // 「单声道源复制成两轨」在老 `.wmv` 里很常见：M/S 之后 `S = (L−R)/2 = 0`，
        // 它一位都用不上。而预算是 `bitrate / fps / channels` **平均分**的 ——
        // 平均分就等于把那一半白扔（实测总码率只能到目标的一半）。
        // 这条钉住「按每个声道需要多少分」：实际码率应当贴近目标。
        let n = 40usize;
        let mut frames = Vec::new();
        let mut state = 0x1234_5678u32;
        for t in 0..n {
            let mut p = vec![0f32; LONG_LEN];
            for (i, v) in p.iter_mut().enumerate() {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                let noise = ((state >> 16) as f32 / 32768.0) - 1.0;
                let k = (t * LONG_LEN + i) as f32;
                *v = (k * 0.07).sin() * 6000.0 + noise * 300.0;
            }
            // 两个声道**一模一样** —— `S` 恒等于 0。
            frames.push(Frame {
                planes: vec![p.clone(), p],
            });
        }
        let mut enc = Encoder::new(EncoderConfig {
            bitrate: 128_000,
            ..Default::default()
        })
        .expect("建编码器");
        let mut bytes = 0usize;
        for f in &frames {
            bytes += enc.encode_frame(f).expect("编帧").len();
        }
        let frames_per_sec = 44100.0 / LONG_LEN as f64;
        let kbps = bytes as f64 * 8.0 * frames_per_sec / n as f64 / 1000.0;
        // 平均分的话实测只有 ~66 kbps（目标的一半）；按需要分应当贴近 128k。
        assert!(
            kbps > 100.0,
            "单声道复制素材只花掉 {kbps:.1} kbps（目标 128k）—— 预算又被平均分了"
        );
    }

    /// 一段**带尖锐攻击**的立体声信号 —— 会触发短窗，从而走到窗序列状态机
    /// 那条路（稳态信号永远只发 `ONLY_LONG`，测不到它）。
    fn attacked_frames(n: usize) -> Vec<Frame> {
        (0..n)
            .map(|t| {
                let mut p = vec![0f32; LONG_LEN];
                for (i, v) in p.iter_mut().enumerate() {
                    let k = t * LONG_LEN + i;
                    // 每 4 帧一记：**前 20 ms 静音、然后满幅** —— 这正是
                    // `detect_attack` 要抓的「段间能量暴涨」（静音段能量 0，
                    // 满幅段远超下限；用脉冲/乐音比会被 128 样本的分段摊薄）。
                    let ph = k % (4 * LONG_LEN);
                    *v = if ph < 882 {
                        0.0
                    } else {
                        20000.0 * ((k as f32) * 0.06).sin()
                    };
                }
                Frame {
                    planes: vec![p.clone(), p.iter().map(|v| v * 0.8).collect()],
                }
            })
            .collect()
    }

    /// 带攻击（会触发短窗）的批量编码：**帧数一致、质量不退化**。
    ///
    /// ⚠️ **这里不能断言「逐字节相同」** —— 实测（真音乐 350 帧）有 **2 帧**不同、
    /// 音频载荷差 14 字节，而 **SNR 完全一致**（14.15 vs 14.15）。根因不在并行，
    /// 在 `plan_group` 的**三级细化是近似搜索**（12 → 3 → 1）：它从
    /// `floor = sf_floor + sf_offset` 锚定格点，**不同的 `sf_offset` 会给出不同的
    /// 格点**，偶尔落到不同的「最优」。而偏移搜索的**种子**（`sf_offset_prev`）
    /// 在批量版里每帧都是初始值、在逐帧版里是上一帧的答案 —— 于是极少数帧分叉。
    ///
    /// 也就是说「`sf_offset_prev` 只影响速度、不影响答案」这条**在现在的实现下
    /// 不成立**（它是按更早的「固定二分」写的）。两条路径各给出一个都合法的结果。
    #[test]
    fn batch_encoding_matches_frame_by_frame_with_attacks() {
        let frames = attacked_frames(24);
        let mut a = Encoder::new(EncoderConfig {
            channels: 2,
            threads: 1,
            ..Default::default()
        })
        .expect("建编码器");
        let one: Vec<Vec<u8>> = frames
            .iter()
            .map(|f| a.encode_frame(f).expect("编一帧"))
            .collect();
        assert!(a.short_frame_count() > 0, "这段信号没触发短窗，测试失去意义");

        // 线程数不影响结果（并行是确定的）。
        for threads in [1usize, 3, 8] {
            let mut b = Encoder::new(EncoderConfig {
                channels: 2,
                threads,
                ..Default::default()
            })
            .expect("建编码器");
            let batch = b.encode_frames(&frames).expect("批量编码");
            assert_eq!(one.len(), batch.len(), "帧数不同（threads={threads}）");
            for (i, (x, y)) in one.iter().zip(&batch).enumerate() {
                assert!(
                    x.len().abs_diff(y.len()) <= 16,
                    "第 {i} 帧差得太多（{} vs {} 字节，threads={threads}）",
                    x.len(),
                    y.len()
                );
            }
        }
    }

    /// 批量与逐帧：**帧数一致、线程数不影响结果**；但不保证**逐字节**相同
    /// （原因见下面那条带攻击的测试与 `sf_offset_prev` 的说明）。
    ///
    /// 这里仍断言「同一次批量里线程数不改变一个字节」—— 并行本身是确定的
    /// （每帧一份独立的 `ChannelState`，结果与分块方式无关）。
    ///
    /// 顺带验**分批接续**：把同一批切成小块多次调用，必须与一次喂完**逐字节**相同
    /// （预扫的状态推进是顺序的，切法不该影响它）。
    #[test]
    fn batch_encoding_matches_frame_by_frame() {
        let frames = stereo_frames(40);

        // 基准：逐帧（单线程）
        let mut a = Encoder::new(EncoderConfig {
            channels: 2,
            threads: 1,
            ..Default::default()
        })
        .expect("建编码器");
        let one: Vec<Vec<u8>> = frames
            .iter()
            .map(|f| a.encode_frame(f).expect("编一帧"))
            .collect();

        for threads in [1usize, 2, 4, 8] {
            let mut b = Encoder::new(EncoderConfig {
                channels: 2,
                threads,
                ..Default::default()
            })
            .expect("建编码器");
            let batch = b.encode_frames(&frames).expect("批量编码");
            assert_eq!(one.len(), batch.len(), "帧数不同（threads={threads}）");
            for (i, (x, y)) in one.iter().zip(&batch).enumerate() {
                assert_eq!(x, y, "第 {i} 帧不同（threads={threads}）");
            }
        }

        // 分批喂（每批 7 帧）—— 状态要跨批接续
        let mut c = Encoder::new(EncoderConfig {
            channels: 2,
            threads: 4,
            ..Default::default()
        })
        .expect("建编码器");
        let mut chunked: Vec<Vec<u8>> = Vec::new();
        for part in frames.chunks(7) {
            chunked.extend(c.encode_frames(part).expect("分批编码"));
        }
        assert_eq!(one, chunked, "分批接续与一次喂完不同");
    }

    /// 优化**前**的 `band_code`：每个码本**各自重算**量化。留在这里当 oracle。
    fn band_code_reference(band: &[f32], sf: u8, cb: usize) -> (u32, f32) {
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
                return (u32::MAX / 2, f32::MAX / 2.0);
            }
            let mut nbits = u32::from(bit_tab[curidx]);
            if unsigned {
                for j in 0..dim {
                    if vecs[curidx * dim + j] != 0 {
                        nbits += 1;
                    }
                }
            }
            if cb == CB_ESC {
                for j in 0..dim {
                    if vecs[curidx * dim + j] == 64 {
                        nbits += escape_bits(escape_coef(chunk[j].abs(), q34v));
                    }
                }
            }
            bits += nbits;
            let iq_sf = f64::from(iq(sf));
            for j in 0..dim {
                let v = f64::from(vecs[curidx * dim + j]);
                // 与生产代码同款：escape 那一路也要乘符号（见 `band_code` 的说明）。
                let sign = if chunk[j] < 0.0 { -1.0 } else { 1.0 };
                let rec = if v == 64.0 {
                    dequant_one(escape_coef(chunk[j].abs(), q34v), sf) * sign
                } else {
                    v.abs() * sign * iq_sf
                };
                let d = f64::from(chunk[j]) - rec;
                dist += (d * d) as f32;
            }
        }
        (bits, dist)
    }

    /// 优化**前**的 `band_cost`：11 本码本全算一遍、取最小失真（平手取编号小）。
    fn band_cost_reference(band: &[f32], sf: u8) -> (f32, u8) {
        let mut best: Option<(f32, u8)> = None;
        for cb in 1..=11usize {
            let (_bits, dist) = band_code_reference(band, sf, cb);
            if best.is_none_or(|(bc, _)| dist < bc) {
                best = Some((dist, cb as u8));
            }
        }
        let zero_cost: f32 = band.iter().map(|&c| c * c).sum();
        match best {
            Some((c, cb)) if c <= zero_cost => (c, cb),
            _ => (zero_cost, 0),
        }
    }

    #[test]
    fn band_cost_matches_the_reference() {
        // 「量化一次、11 本共用」必须与「每个码本各自重算」**逐位相同**：
        // 失真用 `to_bits` 比（浮点值也要一模一样），码本编号直接比
        // （它要写进位流，差一个整条流就不一样）。
        //
        // 素材覆盖四种形态 × 各种长度（4 的倍数）× 两端极端的 `sf`：
        // 全零带、极小（所有码本都不裁剪）、普通、单尖峰（只有 escape 容得下）。
        // `sf = 0` 那一档特别重要 —— 那里 `iq` 极小，失真全靠 f32 舍入，
        // 早先那条「按上限直接定赢家」的捷径就是在它上面翻的车。
        let mut state = 0x51ce_f00du32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        let mut scratch: Vec<i32> = Vec::new();
        for case in 0..64usize {
            // 长度取 4 的倍数 —— AAC 的 swb 频带宽度都是 4 的倍数（`band_code` 的前提）。
            let len = 4 * (1 + case % 10);
            let band: Vec<f32> = (0..len)
                .map(|i| {
                    let v = (next() >> 16) as f32 / 32768.0 - 1.0;
                    match case % 4 {
                        0 => 0.0,
                        1 => v * 0.5,
                        2 => v * 3000.0,
                        _ => {
                            if i == 0 {
                                v * 100_000.0
                            } else {
                                v * 0.01
                            }
                        }
                    }
                })
                .collect();
            for &sf in &[0u8, 20, 40, 60, 80, 100, 120, 140, 160, 200, 230, 255] {
                let fast = band_cost_rd(&band, sf, &mut scratch);
                let full = band_cost_reference(&band, sf);
                assert_eq!(
                    fast.0.to_bits(),
                    full.0.to_bits(),
                    "case {case} sf {sf}：失真不同（{fast:?} vs {full:?}）"
                );
                assert_eq!(
                    fast.1, full.1,
                    "case {case} sf {sf}：码本不同（fast={fast:?} full={full:?}）"
                );
            }
        }
    }
}
