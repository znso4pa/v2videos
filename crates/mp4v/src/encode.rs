//! MPEG-4 Part 2（`mp4v`）编码器 —— **I 帧 + P 帧**。
//!
//! 不做 B 帧、不做码率控制、不做 dquant、不做 8×8 运动补偿 —— 整帧一个 qscale、
//! 宏块级 16×16 的 MV。目标是「`.wmv → .mp4` 的体积别离谱」，不是做到 ffmpeg 的压缩率。
//!
//! ## 与 MP43 编码器的关系：**地基复用，语法各写**
//!
//! | 能复用（`v2v_msmpeg4`） | 不能复用 |
//! |---|---|
//! | `fdct` / `fdct_i`（正向 DCT，定标一致） | 宏块类型表：MP43 是 `MB_INTRA`/`MB_NON_INTRA`，这里是 H.263 的 `ff_h263_{intra,inter}_MCBPC` |
//! | `BitWriter`（MSB 优先） | DC 熵编码：MP43 是「符号 + 8 位 escape」，这里是「**位长** VLC」 |
//! | `RlTable` / `Vlc`（前缀码容器） | AC escape：MP43 有三种、含义不同；这里是统一表挑最短 |
//! | `Frame`、`Error`/`Result`、`ZIGZAG` | DC/AC **预测**：tie-break 判据不同（MP43 是 `<=`，这里是 `<`） |
//! | `tables::RL_INTRA_*`（同一张表） | DC 缩放表：MP43 用 `OLD_Y_DC_SCALE`，这里用 `ff_mpeg4_y_dc_scale_table` |
//! | `Planes` / `mc_params` / `mc_luma16` / `mc_chroma8` / `mc_mb`（半像素 MC） | **运动矢量 VLC**：MP43 是 1100 项的 `ff_msmp4_mv_table`，这里是 33 项的 `ff_mvtab` |
//! | `idct_put` / `idct_add`（重建） | 帧间 RL 表的**取用方式**：MP43 按 qscale 分档，这里用第 0 档（恒等） |
//!
//! ## I 帧的处理顺序（每一步的顺序都是语法的一部分，不能换）
//!
//! ```text
//! 1. 6 个块各做正向 DCT → 量化（DC 用 dc_scale；AC 按 `quant_type`：
//!    H.263 平坦步长 / MPEG 逐系数矩阵，见 `Quant`）
//! 2. **先**把 6 个块的 DC 预测全部算完（`ff_mpeg4_pred_dc`）—— 它会**就地更新** DC 网格，
//!    而块 1/2/3 要读块 0 刚写进去的值，所以必须按 n = 0,1,2,3,4,5 顺序走
//! 3. AC 预测（可选，见下）—— 同样按 n 顺序走，因为块 1 要读块 0 的 ac_val
//! 4. 由**扫描序里最后一个非零系数**定 cbp
//! 5. 写 MCBPC（I 表）→ ac_pred 位 → CBPY → 6 个块（每块：DC 再 AC）
//! ```
//!
//! ## P 帧的处理顺序
//!
//! ```text
//! 1. MV 预测（`ff_h263_pred_motion` 的 16×16 一路：中值预测 + 首行的特例）
//! 2. 运动估计 → MV（差分夹在 ±31，配 f_code = 1）
//! 3. 半像素运动补偿 → 预测宏块；残差 = 当前 − 预测
//! 4. 残差 → 正向 DCT → 量化。**帧间没有独立的 DC 一趟**：64 个系数一视同仁
//! 5. cbp：6 个块**任一系数非零**（含 DC）就置位
//! 6. 三条候选比位数：**skip**（MV=0 且无残差，只花 1 位）/ **帧间** / **帧内**
//! 7. 写：`mb_coded` → inter MCBPC → CBPY（**要 `^0xF`**）→ MV → 6 个块（只有 AC）
//! ```
//!
//! **帧内宏块在 P 帧里的写法不同**：`mb_coded = 0` 之后接的是
//! `inter_MCBPC[cbpc + 4]`（而不是 `intra_MCBPC`），后面才是 `ac_pred` 位与
//! **不取反**的 CBPY。顺序和 I 帧不一样，抄错了位流就整体错位。
//!
//! ## `ac_pred` 是逐宏块试出来的，不是拍脑袋开的
//!
//! `ac_pred` 是**每宏块一个位**。开了之后，本块首行/首列会减掉邻居的首行/首列再编码。
//! 邻居和本块相似时省码字，不相似时反而费码字。ffmpeg 用一个近似的码率模型打分；
//! 这里**直接两种都写完、比位数**（`BitWriter::bit_pos`），选短的 —— 小宏块、算得准，
//! 而且不需要维护一套码率模型。
//!
//! 关键性质：`ac_val`（留给邻居的首行/首列）在**两种情况下都是「预测前」的值**
//! （解码端加完预测后正好还原成预测前的值），所以试两种**不用回滚网格**，
//! 这让「比位数」变得很便宜。
//!
//! ## 帧内/帧间的判决也是比位数
//!
//! 每条候选都先写进**临时** `BitWriter`，比 `bit_pos()`，赢的那条整段 `append` 进正式位流。
//! 好处是「比较时的位数」和「实际写出的位数」是同一个数 —— 不需要码率模型，也不会
//! 出现「模型说便宜、实际更贵」。
//!
//! 代价是帧内候选会**顺手改掉 DC / AC 网格**（DC 预测本来就要写网格）。所以：
//! - 试帧内之前先**拍快照**（只拍本宏块会写的那 6 个槽）；
//! - 帧内赢了 → 保留网格；帧间赢了 → 还原快照（也就是 1024 / 0，与解码端的
//!   `ff_clean_intra_table_entries` 一致）。
//!
//! ## DC / AC 网格**每帧重建**
//!
//! ffmpeg 的 `dc_val` / `ac_val` 是跨帧复用的（`mpegvideo.c` 只在初始化时填 1024）。
//! 但一帧里每个帧内格子都是「先写后读」、每个帧间格子都被清成 1024 / 0，
//! 所以每帧重建与跨帧复用**结果相同** —— 重建更省心，也就没有跨帧脏状态可错。

use v2v_msmpeg4::{
    fdct, fdct_i, idct_add, idct_put, mc_chroma8, mc_luma16, mc_mb, mc_params, mid_pred,
    sad_luma16, BitWriter, Error, Frame, Planes, Result, RlTable, ZIGZAG,
};

use crate::header::{
    time_increment_bits, write_user_data, write_visual_object_header, write_vol, write_vop_header,
    VolParams, VopKind, VopParams, PROFILE_SIMPLE_LEVEL1,
};
use crate::tables;

/// 越界的 DC 邻居值。ffmpeg 把整块 `dc_val_base` 填成 1024
/// （`mpegvideo.c:607`），所以帧外读到的是 1024 而不是 0。
const DC_NONE: i32 = 1024;

/// 半像素舍入方式。0 = 四舍五入。**必须与 `mc_*` 的 `no_rounding` 参数一致**，
/// 否则编码器的重建会与解码器差半个像素，误差逐帧累积。
const NO_ROUNDING: bool = false;

/// P 帧的 `f_code`。见 `VopParams::f_code` 的说明 —— 取 1 时 MV 差分限在 ±31，
/// 我们的运动估计就按这个夹。
const F_CODE: u8 = 1;

/// MV 差分（半像素单位）的可表示范围上限，配 `F_CODE = 1`。
///
/// `ff_mvtab` 有 33 项，`code = |d|`（`f_code = 1` 时没有低位），最大 `|d| = 32`；
/// 但解码端把**重建后的绝对 MV** 也 `sign_extend(.., 6)`，`+32` 会绕成 `-32`，
/// 所以实际安全上限是 **31**（与 `msmpeg4` 的 MP43 编码器同口径）。
const MV_DIFF_MAX: i32 = 31;

/// `sign_extend(val, 6)` 的值域：绝对 MV 必须落在这里面。
/// 帧内试编的**预判倍数**：帧间最优 SAD × 它 仍小于「帧内块的 AC 能量」时，
/// 跳过帧内试编（那一趟要 6 次正向 DCT + 量化 + **两遍**熵编码，是每宏块最贵的一段）。
///
/// 道理：帧内编的是**原始块**（MPEG-4 帧内没有方向预测），代价与「块减去自身均值之后的
/// 能量」强相关；帧间的代价与残差 SAD 强相关 —— 两者都是同一量纲的绝对差之和，
/// 所以直接比。**倍数越大越保守（跳得越少）**，取值见 TODO.md 的 A/B。
const INTRA_SKIP_RATIO: u64 = 2;

const MV_MIN: i32 = -32;
const MV_MAX: i32 = 31;

/// 运动估计的整像素搜索半径（两个中心各搜这么大，再各做一圈半像素细化）。
const ME_RANGE: i32 = 8;

/// 一个块留给邻居的 16 个值：`[1..8]` 是**首列**（给右边用），`[9..16]` 是**首行**（给下面用）。
/// `[0]` 与 `[8]` 是 DC 的槽位，格式里不参与 AC 预测，恒 0（与 ffmpeg 的 `ac_val` 布局一致）。
type AcVal = [i16; 16];

/// MPEG-4 的 qscale 上限。
const QSCALE_MAX: u8 = 31;

/// 编码配置。
#[derive(Debug, Clone, Copy)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    /// 帧率分子 —— 写进 VOL 的 `vop_time_increment_resolution`，也决定 `time_increment_bits`。
    pub fps_num: u32,
    /// 整帧固定的量化步长（1..=31）。越大越糊、文件越小。
    ///
    /// 和 MP43 不同，这里**没有**「量化后 level 必须塞进 8 位 escape」的约束 ——
    /// MPEG-4 的 ESC3 是 12 位，所以 qscale 可以放心取小。
    pub qscale: u8,
    /// **目标码率（bit/s）**。`Some` 时开**跨帧反馈式码率控制**（逐帧调 qscale，见模块文档）；
    /// `None` 时整段用固定的 `qscale`（旧行为）。
    ///
    /// 开着的时候 `qscale` 只当**起步值**用 —— 控制器会按实际花费把它拉走。
    pub bitrate: Option<u32>,
    /// 要不要逐宏块试 AC 预测（见模块文档）。默认开。
    pub ac_pred: bool,
    /// 要不要对**帧间块**做系数级 RD（trellis，见 `trellis_levels`）。
    ///
    /// 换了什么：量化不再是「逐系数取最近电平」，而是在 `{0, L, L∓1}` 里挑
    /// `失真 + λ·位数` 最小的那个 —— 编码代价（run / escape / 符号）第一次参与决策。
    ///
    /// ⚠️ **默认关，因为实测是净亏**（数字见 TODO）：
    /// - 固定 qscale 下它确实省位（−18%）而且是个公平交易（PSNR −1.6 dB）；
    /// - 但接到我们这套**帧级**码控上，四档 λ 全是「体积更大 + 画质更差」
    ///   （最好那档仍是 PSNR −0.5 dB / XPSNR −0.75，体积还 +4%）。
    ///
    /// 两条能解释实测的原因：① 我们**没有 GOP 级比特分配** —— I 帧不受 trellis 影响，
    /// P 帧省下的位换不成整段更细的 q，反而让帧级码控把 q 一路压低、I 帧把体积撑回去；
    /// ② λ 的「正确值」是帧级码控的边际失真-比特斜率，它不是只跟 `q²` 走的常数。
    /// **要用它，得先有分配那一层**（那也正是感知加权生效的前提）。
    pub trellis: bool,
    /// VOL 的 `quant_type`（见 `Quant`）。默认**开** —— 实测同码率下
    /// PSNR 持平而 SSIM / XPSNR / VMAF 都更高（默认矩阵就是对比敏感度曲线：
    /// 高频粗量化，人眼看不出来）。关掉退回 H.263 的平坦步长。
    pub mpeg_quant: bool,
    /// **周期 I 帧间隔**（帧数）。`0` = 只首帧是 I（默认，与「全 I 帧」行为一致）。
    ///
    /// 开着的意义：P 帧只参考前一帧，误差会累积（漂移）；周期 I 帧还能让播放器 seek。
    /// 转换工具按秒数换算后传进来。
    pub keyframe_interval: usize,
    /// VOL 里的 `low_delay`。恒 1 更省事（没有 B 帧）。
    pub low_delay: bool,
    /// 视觉对象序列头的 profile_and_level_indication。
    pub profile_and_level: u8,
    /// 用户数据（`00 00 01 B2` 之后）。空串表示不写。
    pub user_data: &'static str,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            fps_num: 25,
            qscale: 4,
            bitrate: None,
            ac_pred: true,
            mpeg_quant: true,
            trellis: false,
            keyframe_interval: 0,
            low_delay: true,
            profile_and_level: PROFILE_SIMPLE_LEVEL1,
            user_data: "v2video",
        }
    }
}

/// MPEG-4 Part 2 编码器。有状态：P 帧的参考帧与 MV 网格都要跨帧留着。
pub struct Encoder {
    cfg: EncoderConfig,
    mb_w: usize,
    mb_h: usize,
    time_increment_bits: u32,
    /// 已经编过多少帧 —— 决定 VOP 头里的 `time_mod` 与回绕位。
    frame_index: u64,
    /// 上一次时间基回绕的次数（`frame_index / fps_num` 的旧值）。
    last_time_base: u64,
    /// 上一帧的**重建**（P 帧运动补偿的参考）。`None` = 还没编过 I 帧。
    ///
    /// ⚠️ 存的是**重建**而不是原始帧 —— 解码器手里只有重建，编码器要是拿原始帧当参考，
    /// 误差就会一帧一帧地分叉出去。
    reference: Option<Planes>,
    /// 距上一个 I 帧编了多少帧（周期 I 帧用）。
    frames_since_i: usize,
    /// **上一帧是不是关键帧（I 帧）** —— 封装器写载荷的关键帧位要用。
    pub last_key_frame: bool,
    /// **上一帧里走了帧内路径的宏块数**（P 帧判决的结果；I 帧恒为全部）。
    /// 测试与调用方用它确认「画面切换时帧内真的接管了」。
    pub intra_mbs_last: usize,
    /// 上一帧**被预判挡掉**的帧内试编次数（见 `INTRA_SKIP_RATIO`）——
    /// 与 `intra_mbs_last` 一起看就知道预判挡得狠不狠。
    pub intra_skips_last: usize,
    /// **本帧要用的 qscale**。码率控制关着时恒等于 `cfg.qscale`，开着时逐帧变。
    cur_qscale: u8,
    /// 码率控制器的状态。`cfg.bitrate` 是 `None` 时也是 `None`。
    rc: Option<RateCtl>,
    /// **参考帧的 MV 场**（每宏块一个，光栅序）—— 运动估计的**时间预测**。
    ///
    /// 菱形搜索从三个起点各走一遍：`(0,0)`（静止）、`pred`（邻居中值）、
    /// 以及**同位宏块在上一帧的 MV**（运动的连续性）。
    /// 空 `Vec` = 还没有上一帧（第一帧 P 帧），那时只走前两个起点。
    mv_prev: Vec<(i32, i32)>,
}

/// 单趟**跨帧反馈**式码率控制的状态。
///
/// 思路只有两条：
///
/// 1. **虚拟缓冲**记「累计超支/结余」（`buffer`）。它负责长期不跑偏。
/// 2. **位数大致与 qscale 成反比**（`bits ≈ 内容复杂度 / q`），所以「想让下一帧花
///    `wanted` 位，就把 q 乘上 `本帧位数 / wanted`」。它负责短期跟得上。
///
/// 三个防抖动的夹子（都是踩过 AAC 那边「一帧跳到底」的坑之后加的）：
/// 每帧 q 最多动 `RC_MAX_STEP` 档、缓冲偏差摊到 `RC_PULL_FRAMES` 帧里纠、
/// 「该花多少位」有下限。**宁可慢慢收敛，也不要一帧把 q 顶到 31 再荡回来。**
struct RateCtl {
    /// 目标每帧位数 = `bitrate / fps`。
    target_bits: f64,
    /// 虚拟缓冲占用（位数）。**正 = 已超支**，负 = 还有结余。
    buffer: f64,
    /// 缓冲容量（位数）—— 也是控制器「能容忍多大的累计偏差」。
    buffer_size: f64,
    /// 已编码帧数与累计位数（收尾时自证「有没有落在目标附近」）。
    frames: u64,
    total_bits: u64,
    /// **上一帧**的复杂度估计 `X = 位数 × q`（因为 `bits ∝ X/q`）。
    ///
    /// ⚠️ 这里**故意不按帧类型分开**：分开那一版量过（见 TODO 第五轮）——
    /// 它等于把「I 帧该花多少位」变成「跟 P 帧一样多」，反而把 I 帧量化粗了，
    /// 285k 上 PSNR −0.40 dB。I 帧的优待要**显式**给（见 `RC_I_SHARE`），不是靠分开记。
    x_last: Option<f64>,
    /// 最近一次给出的 q（当「每帧最多动几档」的起点）。
    last_q: u8,
}

/// 每帧 qscale 最多动几档 —— 防一帧跳到底。
const RC_MAX_STEP: f64 = 2.0;
/// 缓冲偏差摊到几帧里纠回来（越大越温和）。
const RC_PULL_FRAMES: f64 = 8.0;
/// 「下一帧该花多少位」相对目标的下限（防止 wanted 掉到 0 或负数）。
const RC_MIN_WANTED: f64 = 0.25;
/// 缓冲容量 = 几帧的预算。
const RC_BUFFER_FRAMES: f64 = 8.0;
/// **I 帧的比特份额倍数**（`1.0` = 与 P 帧相同的每帧预算）。
///
/// 为什么把它拎出来当旋钮：I 帧是**参考帧**，它的质量顺着预测链传给后面每一帧，
/// 而帧级码控只认「这一帧花了多少位」、不认「这些位后面被谁用」。
///
/// ⚠️ **实测 `1.0` 就是最好**（等体积插值比，数字见 TODO 第六轮）：
/// - `0.5`（让 I 帧更粗）明显更差 —— 方向对；
/// - `2 / 4 / 8` 画质只涨一点点，但那是靠**超码率**换的；拉到**等体积**再比，三项全输
///   （≈ PSNR −0.27 dB、SSIM −0.007、XPSNR −0.24）。
/// 原因：我们是 **I+P、没有 B 帧** —— 每个 P 帧同样是后面 11 帧的参考，
/// 「I 帧特别值」的前提在这套结构里不成立。等真做了 B 帧（B 不被参考）再回来看这个旋钮。
const RC_I_SHARE: f64 = 1.0;

impl RateCtl {
    fn new(bitrate: u32, fps: u32, q0: u8) -> Self {
        let target_bits = f64::from(bitrate) / f64::from(fps);
        Self {
            target_bits,
            buffer: 0.0,
            buffer_size: target_bits * RC_BUFFER_FRAMES,
            frames: 0,
            total_bits: 0,
            x_last: None,
            last_q: q0,
        }
    }

    /// 给**下一帧**（类型 `is_i`）定 q。
    ///
    /// `q = X_last / wanted_t`（`X = 位数 × q` ⇒ 想花 `wanted` 位就该用 `X/wanted`）。
    /// `wanted_t` 里带一个**按帧类型的份额**：I 帧是参考帧，它多花一位比 P 帧多花一位
    /// 「更值」（后面整条预测链都吃它的质量）—— 这是「传播价值」的最便宜近似。
    fn q_for(&mut self, is_i: bool) -> u8 {
        let wanted = (self.target_bits - self.buffer / RC_PULL_FRAMES)
            .clamp(self.target_bits * RC_MIN_WANTED, self.target_bits * 4.0);
        let wanted = if is_i { wanted * RC_I_SHARE } else { wanted };
        let want_q = match self.x_last {
            Some(x) => x / wanted,
            None => f64::from(self.last_q),
        };
        let step = (want_q - f64::from(self.last_q)).clamp(-RC_MAX_STEP, RC_MAX_STEP);
        self.last_q = (f64::from(self.last_q) + step).round().clamp(1.0, QSCALE_MAX as f64) as u8;
        self.last_q
    }

    /// 编完一帧：记账（缓冲 + 这一帧的复杂度）。
    fn note_bits(&mut self, bits: u64, q: u8) {
        let bits = bits as f64;
        self.frames += 1;
        self.total_bits += bits as u64;
        self.buffer =
            (self.buffer + bits - self.target_bits).clamp(-self.buffer_size, self.buffer_size);
        self.x_last = Some(bits * f64::from(q));
    }
}

impl Encoder {
    pub fn new(cfg: EncoderConfig) -> Result<Self> {
        if cfg.width == 0
            || cfg.height == 0
            || !cfg.width.is_multiple_of(2)
            || !cfg.height.is_multiple_of(2)
        {
            return Err(Error::Bad("尺寸必须是正的偶数（YUV420P）"));
        }
        if !(1..=QSCALE_MAX).contains(&cfg.qscale) {
            return Err(Error::Bad("qscale 必须在 1..=31"));
        }
        if cfg.fps_num == 0 {
            return Err(Error::Bad("帧率分子不能是 0"));
        }
        if cfg.bitrate == Some(0) {
            return Err(Error::Bad("目标码率不能是 0"));
        }
        Ok(Self {
            mb_w: (cfg.width as usize).div_ceil(16),
            mb_h: (cfg.height as usize).div_ceil(16),
            time_increment_bits: time_increment_bits(cfg.fps_num),
            frame_index: 0,
            last_time_base: 0,
            reference: None,
            frames_since_i: 0,
            last_key_frame: false,
            intra_mbs_last: 0,
            intra_skips_last: 0,
            cur_qscale: cfg.qscale,
            rc: cfg.bitrate.map(|b| RateCtl::new(b, cfg.fps_num, cfg.qscale)),
            mv_prev: Vec::new(),
            cfg,
        })
    }

    /// 码率控制开着时：`(已编码帧数, 累计位数)`；关着时 `None`。
    ///
    /// 调用方（CLI / 测试）用它自证「实际码率有没有落在目标附近」——
    /// 这是码率控制唯一值得看的判据。
    pub fn rate_control_stats(&self) -> Option<(u64, u64)> {
        self.rc.as_ref().map(|rc| (rc.frames, rc.total_bits))
    }

    /// 本帧实际用的 qscale（码率控制开着时会逐帧变，调试/测试用）。
    pub fn current_qscale(&self) -> u8 {
        self.cur_qscale
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    /// 宏块数 `(宽, 高)`。
    pub fn mb_count(&self) -> (usize, usize) {
        (self.mb_w, self.mb_h)
    }

    /// 塞进 `esds` 的 `DecoderSpecificInfo`。
    ///
    /// **就是基本流前缀那一份字节**（VOS + 视觉对象 + VOL + 用户数据），不是只有 VOL ——
    /// 少了 VOS 那两级，`profile_and_level_indication` 就丢了，`ffprobe` 只能报
    /// `profile=unknown`（参考样本报的是 `Simple Profile` / `level 1`）。
    /// ffmpeg 的 `esds` 也是这么放的：它的 DSI 以 `00 00 01 B0 01` 开头。
    pub fn decoder_specific_info(&self) -> Vec<u8> {
        self.elementary_stream_header()
    }

    /// 整条基本流的前缀：VOS + 视觉对象 + VOL + 用户数据。
    ///
    /// 只在流的最前面写一次。每个 I 帧前面也可以重写一遍 VOL（让流能任意位置起播），
    /// 但那样每一帧都多 40 多字节；全 I 帧的流里由调用方按需决定（见 `encode_i_frame_with_vol`）。
    pub fn elementary_stream_header(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        write_visual_object_header(&mut w, self.cfg.profile_and_level);
        self.write_vol_into(&mut w);
        if !self.cfg.user_data.is_empty() {
            write_user_data(&mut w, self.cfg.user_data);
        }
        w.finish()
    }

    fn write_vol_into(&self, w: &mut BitWriter) {
        write_vol(
            w,
            &VolParams {
                width: self.cfg.width,
                height: self.cfg.height,
                fps_num: self.cfg.fps_num,
                low_delay: self.cfg.low_delay,
                mpeg_quant: self.cfg.mpeg_quant,
            },
        );
    }

    /// 编一个 I 帧（只含 VOP 头 + 宏块，不含 VOL）。
    pub fn encode_i_frame(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        self.encode_i_frame_with_vol(frame, false)
    }

    /// 编一帧：按 `keyframe_interval` 决定 I 还是 P。
    ///
    /// 首帧恒为 I，之后每 `keyframe_interval` 帧插一个 I。
    ///
    /// ⚠️ `keyframe_interval == 0` 是**关掉 GOP** —— 首帧 I、其余**全是 P**，
    /// 与 `v2v_msmpeg4` 的语义一致。注意这**不是**「全 I 帧」；想要全 I 帧就别调这个函数、
    /// 直接调 `encode_i_frame`。
    pub fn encode_frame(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        // ⚠️ 先把「距上个 I 帧多远」记到**当前帧**上，再判要不要 I ——
        // 反过来写会整整多一帧（12 帧 / GOP 5 会变成 0、6 两个关键帧，而不是 0、5、10）。
        // 这个顺序与 `msmpeg4::Encoder::encode_frame` 一致，两边的 GOP 才对得上。
        if self.reference.is_some() {
            self.frames_since_i += 1;
        }
        let force_i = self.reference.is_none()
            || (self.cfg.keyframe_interval > 0
                && self.frames_since_i >= self.cfg.keyframe_interval);
        if force_i {
            self.encode_i_frame_with_vol(frame, false)
        } else {
            self.encode_p_frame(frame)
        }
    }

    /// 同 `encode_i_frame`，但可以要求在 VOP 前重写一遍 VOL。
    pub fn encode_i_frame_with_vol(&mut self, frame: &Frame, with_vol: bool) -> Result<Vec<u8>> {
        if frame.width != self.cfg.width || frame.height != self.cfg.height {
            return Err(Error::Bad("帧尺寸与配置不一致"));
        }

        let (mb_w, mb_h) = (self.mb_w, self.mb_h);
        let (pw, ph) = (mb_w * 16, mb_h * 16);
        let (py, pu, pv) = pad_planes(frame, pw, ph);

        let qscale = self.take_qscale(true);
        let quant = Quant::new(qscale, self.cfg.mpeg_quant);
        let y_dc_scale = i32::from(tables::Y_DC_SCALE[qscale as usize]);
        let c_dc_scale = i32::from(tables::C_DC_SCALE[qscale as usize]);

        let rl = intra_rl_table();

        let mut w = BitWriter::new();
        if with_vol {
            self.write_vol_into(&mut w);
        }
        let time_base = self.frame_index / u64::from(self.cfg.fps_num);
        write_vop_header(
            &mut w,
            &VopParams {
                kind: VopKind::I,
                time_mod: (self.frame_index % u64::from(self.cfg.fps_num)) as u32,
                time_incr: (time_base - self.last_time_base) as u32,
                time_increment_bits: self.time_increment_bits,
                qscale,
                no_rounding: NO_ROUNDING,
                f_code: F_CODE,
            },
        );
        self.last_time_base = time_base;
        self.frame_index += 1;
        // 单独调 `encode_i_frame` 也要把计数清掉，不然之后接 `encode_frame` 会把
        // 间隔算多。`encode_frame` 自己也会在编完 I 时清，这里重复一次是幂等的。
        self.frames_since_i = 0;
        self.last_key_frame = true;

        // DC / AC 网格：每帧重建（见模块文档）。
        let mut dc = DcGrid::new(mb_w, mb_h, y_dc_scale, c_dc_scale);
        let mut ac = AcGrid::new(mb_w, mb_h);
        // 重建缓冲：**下一帧的 P 要拿它当参考**，所以即使这帧是 I 也要重建。
        let mut cur = Planes::new(pw, ph);

        let mut blocks = [[0i32; 64]; 6];
        for mby in 0..mb_h {
            for mbx in 0..mb_w {
                // ---- 1. 正向 DCT + 量化 ----
                for (n, slot) in blocks.iter_mut().enumerate() {
                    let (plane, stride, px, py) = block_at(&py, &pu, &pv, pw, mbx, mby, n);
                    let mut coeff = [0i32; 64];
                    fdct(&plane[py * stride + px..], stride, &mut coeff);
                    quantize_block(&mut coeff, n, &quant, y_dc_scale, c_dc_scale);
                    *slot = coeff;
                }

                // ---- 2. DC 预测（顺序敏感：会就地更新网格）----
                let mut dc_diff = [0i32; 6];
                let mut dir = [0usize; 6];
                for n in 0..6 {
                    let (pred_level, d) = dc.predict(n, mbx, mby);
                    dc_diff[n] = blocks[n][0] - pred_level;
                    dir[n] = d;
                    dc.update(blocks[n][0], n, mbx, mby);
                }

                // ---- 3+4+5. AC 预测（两种都试）+ 定 cbp + 写宏块 ----
                let (best, _) = intra_try(
                    MbFlavour::IntraI,
                    mbx,
                    mby,
                    &blocks,
                    &dc_diff,
                    &dir,
                    self.cfg.ac_pred,
                    &rl,
                    &mut ac,
                )?;
                w.append(&best);

                // ---- 6. 重建（AC 预测不影响重建 —— 解码端会把预测加回来）----
                reconstruct_intra(
                    &mut cur, mbx, mby, &blocks, &quant, y_dc_scale, c_dc_scale,
                );
            }
        }

        self.intra_mbs_last = mb_w * mb_h;
        self.reference = Some(cur);
        Ok(self.note_frame(w.finish(), qscale))
    }

    /// 编一个 **P 帧**（只含 VOP 头 + 宏块）。
    ///
    /// 前提：之前编过至少一个 I 帧（参考帧由 `encode_i_frame*` 建起来）。
    pub fn encode_p_frame(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        if frame.width != self.cfg.width || frame.height != self.cfg.height {
            return Err(Error::Bad("帧尺寸与配置不一致"));
        }
        // 参考帧**取出来**：既绕开借用检查，也顺带保证「编完一定换成新的」。
        let refp = self
            .reference
            .take()
            .ok_or(Error::Bad("P 帧之前必须先编一个 I 帧"))?;

        let (mb_w, mb_h) = (self.mb_w, self.mb_h);
        let (pw, ph) = (mb_w * 16, mb_h * 16);
        let (sy, su, sv) = pad_planes(frame, pw, ph);

        let qscale = self.take_qscale(false);
        let quant = Quant::new(qscale, self.cfg.mpeg_quant);
        let y_dc_scale = i32::from(tables::Y_DC_SCALE[qscale as usize]);
        let c_dc_scale = i32::from(tables::C_DC_SCALE[qscale as usize]);

        let rl_intra = intra_rl_table();
        let rl_inter = inter_rl_table();

        let mut w = BitWriter::new();
        let time_base = self.frame_index / u64::from(self.cfg.fps_num);
        write_vop_header(
            &mut w,
            &VopParams {
                kind: VopKind::P,
                time_mod: (self.frame_index % u64::from(self.cfg.fps_num)) as u32,
                time_incr: (time_base - self.last_time_base) as u32,
                time_increment_bits: self.time_increment_bits,
                qscale,
                no_rounding: NO_ROUNDING,
                f_code: F_CODE,
            },
        );
        self.last_time_base = time_base;
        self.frame_index += 1;
        // 计数归 `encode_frame` 管（它在判定前 +1）。这里不再动 ——
        // 两处都加会变成每帧 +2，GOP 直接减半。
        self.last_key_frame = false;

        // 时间预测要用**参考帧**的 MV 场 —— 取出来（绕开借用），编完再写回本帧的。
        let mv_prev = std::mem::take(&mut self.mv_prev);
        let mut st = PState {
            refp: &refp,
            cur: Planes::new(pw, ph),
            sy: &sy,
            su: &su,
            sv: &sv,
            pw,
            mb_w,
            mb_h,
            quant,
            y_dc_scale,
            c_dc_scale,
            use_ac_pred: self.cfg.ac_pred,
            trellis: self.cfg.trellis,
            dc: DcGrid::new(mb_w, mb_h, y_dc_scale, c_dc_scale),
            ac: AcGrid::new(mb_w, mb_h),
            mv: vec![(0, 0); mb_w * mb_h],
            mv_prev: &mv_prev,
            intra_mbs: 0,
            intra_skips: 0,
            rl_intra: &rl_intra,
            rl_inter: &rl_inter,
            me_cache: MeSadCache::new(),
        };
        for mby in 0..mb_h {
            for mbx in 0..mb_w {
                st.mb(&mut w, mbx, mby)?;
            }
        }

        self.intra_mbs_last = st.intra_mbs;
        self.intra_skips_last = st.intra_skips;
        self.reference = Some(st.cur);
        // 本帧的 MV 场留给下一帧当**时间预测**的起点。
        self.mv_prev = std::mem::take(&mut st.mv);
        Ok(self.note_frame(w.finish(), qscale))
    }

    /// 本帧该用哪个 q：开码控时问控制器（要按本帧类型给份额），关着就用固定值。
    fn take_qscale(&mut self, is_i: bool) -> u8 {
        if let Some(rc) = self.rc.as_mut() {
            self.cur_qscale = rc.q_for(is_i);
        }
        self.cur_qscale
    }

    /// 收下一个编好的帧：记账（位数 + 复杂度）。
    ///
    /// 关着码率控制时是个**直通**（原样返回），所以调用点不必分支。
    fn note_frame(&mut self, bytes: Vec<u8>, q: u8) -> Vec<u8> {
        if let Some(rc) = self.rc.as_mut() {
            // `* 8`：按位算 —— 预算本来就是位。
            rc.note_bits(bytes.len() as u64 * 8, q);
        }
        bytes
    }
}

/// 帧内 RL 表：`ff_mpeg4_rl_intra`，`n = 102`（下标 102 是 escape）、`last = 67`。
fn intra_rl_table() -> RlTable {
    RlTable::new(
        &tables::RL_INTRA_VLC,
        &tables::RL_INTRA_RUN,
        &tables::RL_INTRA_LEVEL,
        67,
    )
}

/// 帧间 RL 表：`ff_h263_rl_inter`（就是 `ff_rl_table[5]`），`n = 102`、`last = 58`。
///
/// 注意取的是**第 0 档**（恒等）—— ffmpeg 的 `ff_rl_init` 会按 qscale 预乘出 1..31 档，
/// 解码端帧间取第 `qscale` 档；那是「解出来即最终系数」的便利做法。
/// **编码器要的是量化后的原始 level**，所以用恒等的第 0 档，反量化自己另算一趟。
fn inter_rl_table() -> RlTable {
    RlTable::new(
        &tables::RL_INTER_VLC,
        &tables::RL_INTER_RUN,
        &tables::RL_INTER_LEVEL,
        58,
    )
}

// ───────────────────────── 宏块 / 块 写出 ─────────────────────────

/// 帧内宏块走哪条语法分支（帧间走 `write_inter_mb`，不在这里）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MbFlavour {
    /// I 帧里的帧内宏块：`intra_MCBPC`，CBPY **不**取反。
    IntraI,
    /// P 帧里的帧内宏块：`mb_coded = 0` + `inter_MCBPC[cbpc + 4]`，CBPY **不**取反。
    IntraP,
}

/// 写一个宏块（帧内两路 + 帧间）。`use_ac_pred` 只对帧内有效。
///
/// 帧内：`ac_val` 网格无论走哪条路都要更新，而且存的是**预测前**的值 ——
/// 解码端加完预测正好还原成它，所以两种写法留下的是同一份网格，
/// 试两种**不用回滚**（这是「比位数」能做得这么便宜的原因）。
///
/// 帧间：`blocks` 是**量化后的残差**，没有独立的 DC 一趟，扫描序恒 zig-zag。
#[allow(clippy::too_many_arguments)]
fn write_mb(
    w: &mut BitWriter,
    flavour: MbFlavour,
    mbx: usize,
    mby: usize,
    blocks: &[[i32; 64]; 6],
    dc_diff: &[i32; 6],
    dir: &[usize; 6],
    use_ac_pred: bool,
    rl_intra: &RlTable,
    ac: &mut AcGrid,
) -> Result<()> {
    // ---- AC 预测：按 n = 0..5 走（块 1 要读块 0 刚存进去的值）----
    let mut work = *blocks;
    let mut scan: [&[u8; 64]; 6] = [&ZIGZAG; 6];
    for n in 0..6 {
        let cell = cell_of(n, mbx, mby);
        let nb = ac.neighbor(n, dir[n], (cell.0 as usize, cell.1 as usize));
        // 存回预测前的首行/首列 —— 必须在读**下一个块**之前做。
        store_ac_val(ac, n, (cell.0 as usize, cell.1 as usize), &blocks[n]);
        if use_ac_pred {
            apply_ac_pred(&mut work[n], &nb, dir[n]);
        }
        scan[n] = if !use_ac_pred {
            &ZIGZAG
        } else if dir[n] == 0 {
            &tables::V_SCAN
        } else {
            &tables::H_SCAN
        };
    }

    // ---- cbp：由**该扫描序下**最后一个非零系数决定 ----
    let mut cbp = 0usize;
    for n in 0..6 {
        if has_ac(&work[n], scan[n]) {
            cbp |= 1 << (5 - n);
        }
    }

    // ---- 宏块头 ----
    match flavour {
        MbFlavour::IntraI => write_intra_mcbpc(w, cbp & 3)?,
        MbFlavour::IntraP => {
            w.put(0, 1); // mb coded
            write_inter_mcbpc(w, (cbp & 3) + 4)?;
        }
    }
    w.put(u32::from(use_ac_pred), 1);
    // 帧内的 CBPY **不取反**（只有帧间才 `^0xF`）。
    write_cbpy(w, cbp >> 2)?;

    // ---- 6 个块：每块先 DC 再 AC ----
    for n in 0..6 {
        write_dc(w, dc_diff[n], n < 4)?;
        write_ac(w, rl_intra, &work[n], scan[n], true)?;
    }
    Ok(())
}

/// 写一个 **P 帧里的帧间宏块**（不含 skip 那一位 —— 调用方先写 `mb_coded`）。
///
/// 顺序：`inter_MCBPC` → CBPY（取反）→ MV → 6 个块的 AC。
/// **帧间没有 DC 这一趟**，6 个块都是「整块 64 个系数」。
fn write_inter_mb(
    w: &mut BitWriter,
    cbp: usize,
    mv: (i32, i32),
    pred: (i32, i32),
    blocks: &[[i32; 64]; 6],
    rl_inter: &RlTable,
) -> Result<()> {
    write_inter_mcbpc(w, cbp & 3)?;
    write_cbpy(w, (cbp >> 2) ^ 0xF)?;
    write_motion(w, mv.0 - pred.0)?;
    write_motion(w, mv.1 - pred.1)?;
    for b in blocks {
        write_ac(w, rl_inter, b, &ZIGZAG, false)?;
    }
    Ok(())
}

/// 帧间宏块的 cbp：**任一系数非零**（含 DC）就置位 —— 帧间没有单独的 DC 一趟。
/// 帧内试编的便宜代理：4 个亮度块「减去自身均值之后的绝对差之和」。
///
/// 与运动估计的 SAD 同量纲（都是绝对差之和），所以能直接比 —— 见 `INTRA_SKIP_RATIO`。
/// 代价约 512 次加法/减法，而它要替代的那一趟是 6 次 DCT + 量化 + 两遍熵编码。
fn intra_luma_ac(y: &[u8], stride: usize, mbx: usize, mby: usize) -> u64 {
    let mut acc = 0u64;
    for n in 0..4usize {
        let ox = mbx * 16 + (n & 1) * 8;
        let oy = mby * 16 + (n >> 1) * 8;
        let mut sum = 0u32;
        for r in 0..8usize {
            let row = &y[(oy + r) * stride + ox..][..8];
            for &p in row {
                sum += u32::from(p);
            }
        }
        let mean = (sum / 64) as i32;
        for r in 0..8usize {
            let row = &y[(oy + r) * stride + ox..][..8];
            for &p in row {
                acc += (i32::from(p) - mean).unsigned_abs() as u64;
            }
        }
    }
    acc
}

fn inter_cbp(blocks: &[[i32; 64]; 6]) -> usize {
    let mut cbp = 0usize;
    for (n, b) in blocks.iter().enumerate() {
        if b.iter().any(|&v| v != 0) {
            cbp |= 1 << (5 - n);
        }
    }
    cbp
}

/// 帧内宏块：AC 预测的两种写法都写一遍，返回**位流更短的那条**与它的 `ac_pred` 值。
///
/// `ac` 网格两次调用留下的是同一份（存的是「预测前」的值），所以**不用回滚**。
fn intra_try(
    flavour: MbFlavour,
    mbx: usize,
    mby: usize,
    blocks: &[[i32; 64]; 6],
    dc_diff: &[i32; 6],
    dir: &[usize; 6],
    use_ac_pred: bool,
    rl_intra: &RlTable,
    ac: &mut AcGrid,
) -> Result<(BitWriter, bool)> {
    let mut plain = BitWriter::new();
    write_mb(
        &mut plain, flavour, mbx, mby, blocks, dc_diff, dir, false, rl_intra, ac,
    )?;
    if !use_ac_pred {
        return Ok((plain, false));
    }
    let mut predicted = BitWriter::new();
    write_mb(
        &mut predicted,
        flavour,
        mbx,
        mby,
        blocks,
        dc_diff,
        dir,
        true,
        rl_intra,
        ac,
    )?;
    if predicted.bit_pos() < plain.bit_pos() {
        Ok((predicted, true))
    } else {
        Ok((plain, false))
    }
}

// ───────────────────────── P 帧的逐宏块决策 ─────────────────────────

/// P 帧编码的每帧状态。
///
/// 单独抽一个结构是为了把「哪些量在宏块之间累积」说清楚 ——
/// DC / AC 网格、MV 网格、重建缓冲，三样都是逐宏块推进、跨宏块可见的。
/// 运动估计的**候选 SAD 记忆表**（直接映射 + 「代」标记）。
///
/// 同一个候选会被反复求值：菱形每走一步，新邻域与旧邻域重叠一半；三个起点各走一遍
/// 又会互相覆盖；最后还有 ±2 补搜与半像素细化。而 SAD 只取决于 `(dx, dy)` 与参考帧，
/// 所以记忆一次就够 —— **值逐位相同，只是不再重算**。
///
/// 「代」标记是拿来省掉清表的：每宏块 `gen += 1`，命中要求 `(dx, dy, gen)` 三者都对。
/// 槽被别的候选占了（冲突）就当没命中，重算 —— 结果不变，只是少省一点。
struct MeSadCache {
    slots: Vec<(i32, i32, u32, u64)>,
    gen: u32,
}

impl MeSadCache {
    /// 256 槽（4 KB）：一个宏块真正访问的候选约一百个，命中率够高。
    fn new() -> Self {
        Self {
            slots: vec![(0, 0, 0, 0); 256],
            gen: 0,
        }
    }

    #[inline]
    fn slot(&self, dx: i32, dy: i32) -> usize {
        let h = (dx as u32).wrapping_mul(0x9E37_79B1) ^ (dy as u32).wrapping_mul(0x85EB_CA77);
        (h as usize) & (self.slots.len() - 1)
    }

    #[inline]
    fn get(&self, dx: i32, dy: i32) -> Option<u64> {
        let s = self.slots[self.slot(dx, dy)];
        (s.0 == dx && s.1 == dy && s.2 == self.gen).then_some(s.3)
    }

    #[inline]
    fn put(&mut self, dx: i32, dy: i32, sad: u64) {
        let i = self.slot(dx, dy);
        self.slots[i] = (dx, dy, self.gen, sad);
    }

    /// 换一代（每个宏块调一次）—— 比清 256 个槽便宜得多。
    #[inline]
    fn next_mb(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            // 回绕了：把标记清干净再从头数（几十万帧才碰一次，不值得更聪明）。
            for s in self.slots.iter_mut() {
                s.2 = 0;
            }
            self.gen = 1;
        }
    }
}

struct PState<'a> {
    refp: &'a Planes,
    /// 本帧的重建 —— 编完就是下一帧的参考。
    cur: Planes,
    sy: &'a [u8],
    su: &'a [u8],
    sv: &'a [u8],
    pw: usize,
    mb_w: usize,
    mb_h: usize,
    /// 本帧的量化规则（含逐系数步长）。
    quant: Quant,
    y_dc_scale: i32,
    c_dc_scale: i32,
    use_ac_pred: bool,
    /// 帧间块的系数级 RD 开关（见 `EncoderConfig::trellis`）。
    trellis: bool,
    dc: DcGrid,
    ac: AcGrid,
    /// 每宏块一个 MV（只做 16×16 模式，亮度那 2×2 个格子填的是同一个值）。
    mv: Vec<(i32, i32)>,
    /// 参考帧的 MV 场（时间预测的起点）。长度对不上就当作没有。
    mv_prev: &'a [(i32, i32)],
    intra_mbs: usize,
    /// 被预判挡掉的帧内试编次数（见 `INTRA_SKIP_RATIO`）。
    intra_skips: usize,
    rl_intra: &'a RlTable,
    rl_inter: &'a RlTable,
    /// 运动估计的候选 SAD 记忆表（见 `MeSadCache`）。
    me_cache: MeSadCache,
}

impl PState<'_> {
    /// H.263 的中值预测（`ff_h263_pred_motion` 的 16×16 一路）。
    ///
    /// 我们只做 16×16 的 MV，亮度那 2×2 个格子填的是同一个值 —— 于是：
    /// `A` = 左宏块、`B` = 上宏块、`C` = **右上**宏块，取三者的中值。
    ///
    /// ⚠️ **首行的特例不能省**：`first_slice_line`（`mby == 0`）时没有上一行，
    /// 此时预测值是 `mbx == 0 ? (0,0) : 左邻居`，**不是中值**。
    fn mv_predict(&self, mbx: usize, mby: usize) -> (i32, i32) {
        let at = |x: i64, y: i64| -> (i32, i32) {
            if x < 0 || y < 0 || x >= self.mb_w as i64 || y >= self.mb_h as i64 {
                // 帧外恒 (0,0)：ffmpeg 每帧开始把整张 motion_val 清零。
                (0, 0)
            } else {
                self.mv[y as usize * self.mb_w + x as usize]
            }
        };
        let a = at(mbx as i64 - 1, mby as i64);
        if mby == 0 {
            return if mbx == 0 { (0, 0) } else { a };
        }
        let b = at(mbx as i64, mby as i64 - 1);
        let c = at(mbx as i64 + 1, mby as i64 - 1);
        (mid_pred(a.0, b.0, c.0), mid_pred(a.1, b.1, c.1))
    }

    /// 运动估计：**三个起点 + 菱形搜索（LDSP）+ 全搜 ±2 收尾**，最后半像素细化。
    ///
    /// 起点是 `(0,0)`（静止）、`pred`（邻居中值，运动场景）与**同位宏块上一帧的 MV**
    /// （运动的连续性）。结果再夹进 `clamp_mv` 的可表示集合 —— 夹完的 MV 一定能被
    /// 解码端原样还原。
    ///
    /// ## 为什么不是全搜
    ///
    /// 全搜是 `17×17 × 2 个中心 = 578` 个候选，每个候选都要搬 16×16 个像素再求和，
    /// 实测占了视频侧 **94%** 的时间（全 I 帧 500 帧 0.59 s vs GOP12 10.07 s）。
    /// 菱形通常 20–40 个候选就落在同一个（或更好的）SAD 上。
    ///
    /// ## ⚠️ 这会改变输出
    ///
    /// MV 不同 → 位流不同。所以它不是「同语义更快」：质量必须重新对账
    /// （**固定 qscale** 下比 PSNR 与体积 —— 开码率控制时不同的 MV 场会让控制器
    /// 落到不同的 qscale，比出来的差不干净）。三个起点就是为了把这一档差补回来。
    fn motion_estimate(&mut self, mbx: usize, mby: usize, pred: (i32, i32)) -> ((i32, i32), u64) {
        let (pw, ph) = (self.pw, self.refp.ph);
        let refp = self.refp;
        // 复制成局部量：闭包要**可变借** `me_cache`，就不能再借 `self` 的别的字段。
        let sy = self.sy;
        let mut cur = [0u8; 256];
        // 逐行拷贝（每行 16 字节）—— 值不变，少一层带边界检查的内层循环。
        for r in 0..16usize {
            let o = (mby * 16 + r) * pw + mbx * 16;
            cur[r * 16..r * 16 + 16].copy_from_slice(&sy[o..o + 16]);
        }
        let mut tmp = [0u8; 256];
        self.me_cache.next_mb();
        let cache = &mut self.me_cache;
        let mut sad_of = |mv: (i32, i32)| -> u64 {
            // 同一个候选会被反复求值（菱形每走一步新邻域与旧邻域重叠一半、三个起点
            // 各走一遍、±2 补搜、半像素细化）—— 记忆一次就够：SAD 只取决于
            // `(dx, dy)` 与参考帧，**值逐位相同**，只是不再重算。
            if let Some(v) = cache.get(mv.0, mv.1) {
                return v;
            }
            // 插值与 SAD 的融合版住在共享内核里（`sad_luma16`）—— 与 MP43 编码器
            // 用的是同一份，不是各写一份。
            let p = mc_params(mbx, mby, mv, pw, ph);
            let acc = sad_luma16(&cur, refp, &p, NO_ROUNDING, &mut tmp);
            cache.put(mv.0, mv.1, acc);
            acc
        };

        // ---- 三个起点**各走一遍菱形**，取最好的结果 ----
        //
        // 只从「三个起点里最好的那个」出发是不够的：SAD 面在强边缘 + 细线上是**多峰**的，
        // 从一个起点出发的菱形会落在离它最近的那个盆地里。三个起点各走一遍
        // （约 3×30 个候选，仍比全搜的 578 少一个数量级）才把「漏掉真最优」压下去。
        let temporal = self.mv_prev.get(mby * self.mb_w + mbx).copied();
        let mut starts = [(0i32, 0i32); 3];
        starts[0] = (0, 0);
        starts[1] = pred;
        let mut n_starts = 2;
        if let Some(t) = temporal {
            starts[n_starts] = t;
            n_starts += 1;
        }

        // 搜索窗 = 三个起点各自的 ±`ME_RANGE`。
        let in_window = |c: (i32, i32)| -> bool {
            let near =
                |cx: i32, cy: i32| (c.0 - cx).abs() <= ME_RANGE && (c.1 - cy).abs() <= ME_RANGE;
            starts[..n_starts].iter().any(|&(sx, sy)| near(sx, sy))
        };

        // 8 个点（±2 与对角）。菱形只覆盖 `(±2,0)/(0,±2)/(±1,±1)`，
        // 所以每个起点走完还要补一圈全搜 ±2。
        const LDSP: [(i32, i32); 8] = [
            (0, -2),
            (1, -1),
            (2, 0),
            (1, 1),
            (0, 2),
            (-1, 1),
            (-2, 0),
            (-1, -1),
        ];
        let mut best = (0i32, 0i32);
        let mut best_sad = u64::MAX;
        for &start in &starts[..n_starts] {
            let mut cur_best = start;
            let mut cur_sad = sad_of(start);
            // 大菱形走到收敛（硬上限只是防御：窗口有限，正常几个来回就收敛）。
            for _ in 0..32 {
                let mut improved = false;
                for &(dx, dy) in &LDSP {
                    let cand = (cur_best.0 + dx, cur_best.1 + dy);
                    if !in_window(cand) {
                        continue;
                    }
                    let s = sad_of(cand);
                    if s < cur_sad {
                        cur_sad = s;
                        cur_best = cand;
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
            if cur_sad < best_sad {
                best_sad = cur_sad;
                best = cur_best;
            }
            if best_sad == 0 {
                return (clamp_mv(best, pred), 0);
            }
        }

        // 半像素细化：最优整像素周围 ±1 个半像素。
        let base = best;
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                let cand = (base.0 + dx, base.1 + dy);
                let s = sad_of(cand);
                if s < best_sad {
                    best_sad = s;
                    best = cand;
                }
            }
        }
        // 最优 SAD 一并带出去 —— 帧内试编的预判要用（见 `INTRA_SKIP_RATIO`）。
        (clamp_mv(best, pred), best_sad)
    }

    /// 帧内候选：把当前帧的**原始像素**当 I 帧宏块编一遍。
    ///
    /// ⚠️ 它会**就地更新 DC 网格**（DC 预测本来就要写网格）—— 调用方必须先用
    /// `snap` 拍快照，输了再 `restore`。
    fn intra_candidate(
        &mut self,
        mbx: usize,
        mby: usize,
    ) -> Result<([[i32; 64]; 6], [i32; 6], [usize; 6], bool, BitWriter)> {
        let mut blocks = [[0i32; 64]; 6];
        for (n, slot) in blocks.iter_mut().enumerate() {
            let (plane, stride, px, py) = block_at(self.sy, self.su, self.sv, self.pw, mbx, mby, n);
            let mut coeff = [0i32; 64];
            fdct(&plane[py * stride + px..], stride, &mut coeff);
            quantize_block(
                &mut coeff,
                n,
                &self.quant,
                self.y_dc_scale,
                self.c_dc_scale,
            );
            *slot = coeff;
        }
        let mut dc_diff = [0i32; 6];
        let mut dir = [0usize; 6];
        for n in 0..6 {
            let (pred_level, d) = self.dc.predict(n, mbx, mby);
            dc_diff[n] = blocks[n][0] - pred_level;
            dir[n] = d;
            self.dc.update(blocks[n][0], n, mbx, mby);
        }
        let (best, ac_pred) = intra_try(
            MbFlavour::IntraP,
            mbx,
            mby,
            &blocks,
            &dc_diff,
            &dir,
            self.use_ac_pred,
            self.rl_intra,
            &mut self.ac,
        )?;
        Ok((blocks, dc_diff, dir, ac_pred, best))
    }

    /// 一个 P 帧宏块：三条候选（skip / 帧间 / 帧内）比位数，赢的写进位流并重建。
    fn mb(&mut self, w: &mut BitWriter, mbx: usize, mby: usize) -> Result<()> {
        // ---------- 帧间候选 ----------
        let pred = self.mv_predict(mbx, mby);
        let (mv, inter_sad) = self.motion_estimate(mbx, mby, pred);
        let p = mc_params(mbx, mby, mv, self.pw, self.refp.ph);
        let mut pred16 = [0u8; 256];
        mc_luma16(&mut pred16, 16, 0, 0, self.refp, &p, NO_ROUNDING);
        let mut predu = [0u8; 64];
        let mut predv = [0u8; 64];
        mc_chroma8(&mut predu, 8, 0, 0, self.refp, 0, &p, NO_ROUNDING);
        mc_chroma8(&mut predv, 8, 0, 0, self.refp, 1, &p, NO_ROUNDING);

        let mut qb = [[0i32; 64]; 6];
        for (n, slot) in qb.iter_mut().enumerate() {
            let mut res = [0i32; 64];
            if n < 4 {
                let ox = mbx * 16 + (n & 1) * 8;
                let oy = mby * 16 + (n >> 1) * 8;
                for r in 0..8usize {
                    for c in 0..8usize {
                        res[r * 8 + c] = self.sy[(oy + r) * self.pw + ox + c] as i32
                            - pred16[((n >> 1) * 8 + r) * 16 + (n & 1) * 8 + c] as i32;
                    }
                }
            } else {
                let cw = self.pw / 2;
                let (plane, prd) = if n == 4 {
                    (self.su, &predu)
                } else {
                    (self.sv, &predv)
                };
                for r in 0..8usize {
                    for c in 0..8usize {
                        res[r * 8 + c] =
                            plane[(mby * 8 + r) * cw + mbx * 8 + c] as i32 - prd[r * 8 + c] as i32;
                    }
                }
            }
            let mut coef = [0i32; 64];
            fdct_i(&res, 8, &mut coef);
            for (k, v) in coef.iter().enumerate() {
                slot[k] = self.quant.quantize(k, *v, false);
            }
            if self.trellis {
                trellis_levels(&self.quant, self.rl_inter, &coef, slot, self.quant.lambda());
                // trellis 会在基线电平上 ∓1，可能把刚好在边界上的值顶出去 —— 夹回来。
                for l in slot.iter_mut() {
                    *l = (*l).clamp(-2047, 2047);
                }
            }
        }
        let cbp_i = inter_cbp(&qb);
        let skip = mv == (0, 0) && cbp_i == 0;

        let mut wi = BitWriter::new();
        if skip {
            // skip 宏块：一个 `1` 位搞定，MV 恒 (0,0)、没有残差。
            wi.put(1, 1);
        } else {
            wi.put(0, 1); // mb coded
            write_inter_mb(&mut wi, cbp_i, mv, pred, &qb, self.rl_inter)?;
        }

        // ---------- 帧内候选（会改 DC / AC 网格，先拍快照）----------
        //
        // ⚠️ **`skip` 时整段不做**：这时 `wi` 只有那一个 `1` 位，而帧内候选至少要写
        // 一个宏块类型（`wi2.bit_pos() >= 1`），所以 `wi2.bit_pos() < wi.bit_pos()`
        // **恒不成立** —— 帧内不可能赢。这不是启发式，是 `bit_pos()` 上的恒等式。
        //
        // 省下的是 6 个块的 `fdct` + 量化 + 写位流（外加 DC/AC 预测），
        // 而「跳过」的代价只是少写一次 DC/AC 网格 —— 那本来就只在帧内赢时才有用，
        // 帧间分支里那两句 `restore` 正是它的逆操作。输出**逐字节不变**。
        let intra = if skip {
            None
        } else if inter_sad * INTRA_SKIP_RATIO < intra_luma_ac(self.sy, self.pw, mbx, mby) {
            // 帧间明显更好 —— 省掉帧内那一趟试编（网格也不用动，与 `skip` 那条同形）。
            self.intra_skips += 1;
            None
        } else {
            let dc_snap = self.dc.snap(mbx, mby);
            let ac_snap = self.ac.snap(mbx, mby);
            let (ib, _dc_diff, _dir, _ac_pred, wi2) = self.intra_candidate(mbx, mby)?;
            Some((dc_snap, ac_snap, ib, wi2))
        };

        // ---------- 定胜负 ----------
        let intra_wins = intra
            .as_ref()
            .is_some_and(|(_, _, _, wi2)| wi2.bit_pos() < wi.bit_pos());
        if intra_wins {
            let (_, _, ib, wi2) = intra.expect("`intra_wins` 已经保证它是 `Some`");
            // 帧内赢了：DC / AC 网格保留（`intra_candidate` 已经写进去了）。
            w.append(&wi2);
            reconstruct_intra(
                &mut self.cur,
                mbx,
                mby,
                &ib,
                &self.quant,
                self.y_dc_scale,
                self.c_dc_scale,
            );
            // 帧内宏块在 MV 网格上留 (0,0) —— 与解码端一致（它不给帧内宏块写 motion_val）。
            self.mv[mby * self.mb_w + mbx] = (0, 0);
            self.intra_mbs += 1;
        } else {
            // 帧间赢了：还原网格（= 清成 1024 / 0，等价于解码端的
            // `ff_clean_intra_table_entries`）。`skip` 时网格根本没被动过，不用还原。
            if let Some((dc_snap, ac_snap, _, _)) = &intra {
                self.dc.restore(mbx, mby, dc_snap);
                self.ac.restore(mbx, mby, ac_snap);
            }
            w.append(&wi);
            if skip {
                self.mv[mby * self.mb_w + mbx] = (0, 0);
                mc_mb(&mut self.cur, self.refp, mbx, mby, (0, 0), NO_ROUNDING);
            } else {
                self.mv[mby * self.mb_w + mbx] = mv;
                mc_mb(&mut self.cur, self.refp, mbx, mby, mv, NO_ROUNDING);
                reconstruct_inter(&mut self.cur, mbx, mby, &qb, &self.quant);
            }
        }
        Ok(())
    }
}

/// 把 MV 夹进「解码端能原样还原」的集合：
/// 既要 `MV_MIN <= mv <= MV_MAX`（**绝对** MV 会走 `sign_extend(.., 6)`），
/// 又要 `|mv - pred| <= MV_DIFF_MAX`（**差分**走 `ff_mvtab`）。
///
/// 两个区间一定相交 —— `pred` 本身就在里面，所以 `clamp` 不会 panic。
fn clamp_mv(mv: (i32, i32), pred: (i32, i32)) -> (i32, i32) {
    let one = |v: i32, p: i32| -> i32 {
        let lo = MV_MIN.max(p - MV_DIFF_MAX);
        let hi = MV_MAX.min(p + MV_DIFF_MAX);
        debug_assert!(lo <= hi, "MV 可行区间空了（pred={p}）");
        v.clamp(lo, hi)
    };
    (one(mv.0, pred.0), one(mv.1, pred.1))
}

// ───────────────────────── 重建 ─────────────────────────

/// 宏块里第 `n` 个块在重建缓冲上的 `(平面切片, 行距)` —— 切片从块的左上角开始。
fn recon_at(cur: &mut Planes, n: usize, mbx: usize, mby: usize) -> (&mut [u8], usize) {
    let pw = cur.pw;
    let cw = pw / 2;
    if n < 4 {
        let px = mbx * 16 + (n & 1) * 8;
        let py = mby * 16 + (n >> 1) * 8;
        (&mut cur.y[py * pw + px..], pw)
    } else if n == 4 {
        (&mut cur.u[mby * 8 * cw + mbx * 8..], cw)
    } else {
        (&mut cur.v[mby * 8 * cw + mbx * 8..], cw)
    }
}

/// 量化/反量化的规则 —— 对应 VOL 里的 `quant_type`。
///
/// - **H.263**（`quant_type = 0`）：平坦步长，`qmul = 2q`、`qadd = (q-1)|1`，
///   反量化是 `level*qmul ± qadd`（与 MP43 同一个式子）。
/// - **MPEG**（`quant_type = 1`）：逐系数乘**规范默认矩阵**
///   （`tables::DEFAULT_INTRA_MATRIX` / `DEFAULT_INTER_MATRIX`，自然序）：
///   帧内 `(|level|·2q·W) >> 4`、帧间 `((2|level|+1)·2q·W) >> 5`。
///   两者都是「先取绝对值、截断（算术右移）、最后补符号」——
///   与 ffmpeg 的 `dct_unquantize_mpeg2_*` 逐位一致。
///
/// 帧内 DC 两种量化都用 `dc_scale`（解码端 `block[0] *= dc_scale`），由调用方单独处理。
///
/// ## ⚠️ 帧间的**失配控制**（MPEG 量化独有）
///
/// 把 64 个反量化值求和（初值 −1），是奇数就把**下标 63** 的最低位翻一下。
/// 解码端的 `dct_unquantize_mpeg2_inter_c` 无条件做这一步，我们少做一次，
/// 重建就每块差一位 —— P 帧会一帧帧累积成漂移。
///
/// ## 为什么这一项值得做（实测，见 TODO）
///
/// 默认矩阵就是一条对比敏感度曲线（高频步长最大到低频的 5 倍），
/// 同码率下 PSNR 持平而 SSIM / XPSNR / VMAF 都更高。它属于「权重在**单元内部**
/// 逐频率变化」那一类 —— 不受「逐块常数加权是 no-op」那条结论的限制。
pub(crate) struct Quant {
    mpeg: bool,
    /// 本帧的 `qscale`（只为算 trellis 的 λ）。
    q: i32,
    /// H.263 的两个常数（MPEG 时不用）。
    qmul: i32,
    qadd: i32,
    /// H.263 那条的倒数：`ceil(2^32/qmul)`（`qmul = 2q ≤ 62`，表按它直接索引）。
    inv_qmul: u32,
    /// 逐系数步长 `2q·W[k]`（自然序）。MPEG 时用。
    intra_step: [i32; 64],
    inter_step: [i32; 64],
    /// MPEG 两条路的倒数：`ceil(2^48 / 除数)` —— 除数逐系数不同（`8·step` / `2·step`），
    /// 所以是两张 64 项的表，在 `new` 里按帧算一次。
    intra_inv: [u64; 64],
    inter_inv: [u64; 64],
}

/// 帧内量化偏置，单位是 1/8 个量化步长。
///
/// `4` = 四舍五入到最近的合法电平 —— 与 ffmpeg 的**非 trellis** 路一致
/// （`dct_quantize_c` 对 `mpeg_quant` 把 bias 覆盖成 `1<<(QMAT_SHIFT-1)`，也就是半个步长；
/// 它那个 `intra_quant_bias = 3/8` 只服务 trellis 路）。调到更小 = 更偏死区：
/// 小系数更容易被判成 0，省下的位由码控花到别处 —— 我们没有 trellis 替我们做这笔账，
/// 所以这是个真旋钮（扫描的数字见 TODO）。
const INTRA_BIAS_EIGHTHS: i32 = 4;

/// trellis 的 λ 旋钮，千分之一为单位（`λ = 它 · q² / 1000`）。
///
/// 尺度怎么来的：零掉一个「离判决边界半个步长」的系数，省下 `几步` 位、付出
/// `(步长/2)²` 的失真 —— 而重建步长 ≈ `q·W/8`，所以 λ 的量级就是 `(qW/16)²/位数`。
/// **它不是推出来的，是标定出来的**（扫描见 TODO）：本仓在另一条线上踩过 λ 差 90 倍。
const TRELLIS_LAMBDA_MILLI: i64 = 2800;

/// 允不允许 trellis 把系数**砍到 0**。关掉就只剩「挪一档」。
/// 加这个开关是为了把「省位」和「漂移」分开量：丢残差会让预测环里的误差累积。
const TRELLIS_ALLOW_ZERO: bool = true;

impl Quant {
    fn new(qscale: u8, mpeg: bool) -> Self {
        let q = i32::from(qscale);
        let mut intra_step = [0i32; 64];
        let mut inter_step = [0i32; 64];
        let mut intra_inv = [0u64; 64];
        let mut inter_inv = [0u64; 64];
        for k in 0..64 {
            intra_step[k] = 2 * q * i32::from(tables::DEFAULT_INTRA_MATRIX[k]);
            inter_step[k] = 2 * q * i32::from(tables::DEFAULT_INTER_MATRIX[k]);
            // 除数与 `quantize` 里用的一模一样：帧内 `8·step`、帧间 `2·step`。
            intra_inv[k] = recip48((8 * intra_step[k]) as u64);
            inter_inv[k] = recip48((2 * inter_step[k]) as u64);
        }
        let qmul = q << 1;
        Self {
            mpeg,
            q,
            qmul,
            qadd: (q - 1) | 1,
            inv_qmul: INV_QMUL[qmul as usize],
            intra_step,
            inter_step,
            intra_inv,
            inter_inv,
        }
    }

    /// trellis 的 λ（失真换位数的兑换率）：`TRELLIS_LAMBDA_MILLI · q² / 1000`。
    fn lambda(&self) -> i64 {
        TRELLIS_LAMBDA_MILLI * i64::from(self.q) * i64::from(self.q) / 1000
    }

    fn step(&self, k: usize, intra: bool) -> i32 {
        if intra {
            self.intra_step[k]
        } else {
            self.inter_step[k]
        }
    }

    /// 反量化一个 AC（帧间含下标 0）系数。`k` 是**自然序**下标。
    fn dequant(&self, k: usize, level: i32, intra: bool) -> i32 {
        if level == 0 {
            return 0;
        }
        if !self.mpeg {
            return if level > 0 {
                level * self.qmul + self.qadd
            } else {
                level * self.qmul - self.qadd
            };
        }
        let step = self.step(k, intra);
        let v = if intra {
            (level.abs() * step) >> 4
        } else {
            ((2 * level.abs() + 1) * step) >> 5
        };
        if level < 0 {
            -v
        } else {
            v
        }
    }

    /// `dequant` 的逆：取让重建值离 `f` 最近的 `level`。
    ///
    /// ⚠️ **除法全部换成了倒数乘法**：三条路各有一个运行期除数（`qmul` / `8·step` / `2·step`），
    /// 而 `sdiv` 一条要几十周期、这里每个 AC 系数都要走一次（实测占整条 `.mp4` 链路的 7.5%）。
    /// 除数在一帧里是常数（逐系数那两条各 64 个值），所以 `new` 里算一次倒数即可。
    /// **值与原式逐位相同** —— 见 `quantize_matches_division` 测试（与 `div_round` 对拍）。
    #[inline(always)]
    fn quantize(&self, k: usize, f: i32, intra: bool) -> i32 {
        if f == 0 {
            return 0;
        }
        if !self.mpeg {
            // 与旧路逐位相同（原 `quantize_ac`）：先按 `a` 的符号取绝对值再还原。
            let a = if f > 0 { f - self.qadd } else { f + self.qadd };
            let m = if a >= 0 { a as u32 } else { (-(a as i64)) as u32 };
            let n = m + (self.qmul as u32 / 2);
            let q = (((n as u64) * (self.inv_qmul as u64)) >> 32) as i32;
            let r = if a >= 0 { q } else { -q };
            let v = if f > 0 { r.max(0) } else { r.min(0) };
            // ⚠️ 这条**提前 return**，别漏了夹取（见下面 `lim` 的说明）。
            let lim = if intra { 1023 } else { 2047 };
            return v.clamp(-lim, lim);
        }
        let step = self.step(k, intra);
        let v = if intra {
            // 帧内：重建电平在 `step/16` 的整数倍上。偏置 = `INTRA_BIAS_EIGHTHS`/8 个步长
            // （`4` 就是四舍五入；写成 8 倍只是为了让更小的偏置也能取整数）。
            let n = (f.unsigned_abs() as u64) * 128 + (INTRA_BIAS_EIGHTHS * step) as u64;
            ((n * self.intra_inv[k]) >> 48) as i32
        } else {
            // 帧间：重建电平在 `step/32` 的**奇数**倍上（`(2|level|+1)`），
            // 所以最近的合法电平是 `round((|f|·32/step − 1)/2)`，
            // 也就是下面这个整数式（`step` 那一项就是那个 −1）。
            // `div_round(N, 2·step)` 的 N = `|f|·32 − step`，而它自己还要 `+ b/2 = + step`
            // —— 两项正好抵消，所以分子就是 `|f|·32`。
            // 原式那条 `N ≤ 0 ⇒ 结果 ≤ 0 ⇒ .max(0) = 0` 也自动成立（`|f|·32 ≤ step` 时商为 0）。
            let num = (f.unsigned_abs() as u64) * 32;
            ((num * self.inter_inv[k]) >> 48) as i32
        };
        let v = if f < 0 { -v } else { v };
        // ⚠️ **电平必须夹进位流装得下的范围**，而且**重建要用同一个值** ——
        // ESC3 的 level 字段是 12 位有符号（`& 0xfff`），超了会**静默回绕**：
        // 编码器按原值重建、解码器读到的是回绕后的值，两边就此分叉。
        // 帧间直接写电平 ⇒ ±2047；帧内写的是「电平 − 预测」，而预测取自邻居**同样被夹过**
        // 的电平 ⇒ 两边各 ±1023 就够保证差分 ≤ 2046。
        let lim = if intra { 1023 } else { 2047 };
        v.clamp(-lim, lim)
    }

    /// 帧间块的失配控制（只有 `quant_type = 1` 做，见结构体文档）。
    fn mismatch_control(&self, rb: &mut [i16; 64]) {
        if !self.mpeg {
            return;
        }
        let mut sum = -1i32;
        for v in rb.iter() {
            sum += i32::from(*v);
        }
        rb[63] ^= (sum & 1) as i16;
    }
}

/// 帧内宏块的重建：反量化后 `idct_put` **覆盖**。
///
/// ⚠️ 用的是**预测前**的 `blocks` —— 解码端会把 AC 预测加回来，还原出的正是它。
fn reconstruct_intra(
    cur: &mut Planes,
    mbx: usize,
    mby: usize,
    blocks: &[[i32; 64]; 6],
    quant: &Quant,
    y_dc_scale: i32,
    c_dc_scale: i32,
) {
    for n in 0..6 {
        let scale = if n < 4 { y_dc_scale } else { c_dc_scale };
        let mut rb = [0i16; 64];
        rb[0] = (blocks[n][0] * scale) as i16;
        for k in 1..64 {
            rb[k] = quant.dequant(k, blocks[n][k], true) as i16;
        }
        let (plane, stride) = recon_at(cur, n, mbx, mby);
        idct_put(&mut rb, plane, stride);
    }
}

/// 帧间宏块的重建：`cur` 里已经铺好了 MC 预测，这里只把反量化后的残差叠上去。
fn reconstruct_inter(
    cur: &mut Planes,
    mbx: usize,
    mby: usize,
    blocks: &[[i32; 64]; 6],
    quant: &Quant,
) {
    for n in 0..6 {
        let mut rb = [0i16; 64];
        for (k, slot) in rb.iter_mut().enumerate() {
            *slot = quant.dequant(k, blocks[n][k], false) as i16;
        }
        quant.mismatch_control(&mut rb);
        let (plane, stride) = recon_at(cur, n, mbx, mby);
        idct_add(&mut rb, plane, stride);
    }
}

/// 块在扫描序 `scan` 下有没有 AC 系数（扫描序下标 >= 1）。
fn has_ac(block: &[i32; 64], scan: &[u8; 64]) -> bool {
    (1..64).any(|i| block[scan[i] as usize] != 0)
}

// ───────────────────────── 量化 ─────────────────────────

/// 量化一个块的 64 个自然序系数（就地）。
///
/// - **DC**（下标 0）：`round(F(0,0) / dc_scale)`。
///   解码端 `block[0] *= dc_scale` 再进 IDCT（`mpegvideo.c:210`），
///   而我们的 `fdct` 给的是 `F(0,0) = 8 × 均值`，所以 `level = 8×均值 / dc_scale`
///   正好还原 —— `dc_scale` 最小是 8，`level` 落在 `[0, 255]`。
///   **两种 `quant_type` 都一样**（MPEG 量化的帧内 DC 也走 `dc_scale`）。
/// - **AC**：按 `quant` 的规则取 `level`（H.263 平坦步长 / MPEG 逐系数矩阵）。
///   两种都不必管 escape 上限：MPEG-4 的 ESC3 是 12 位。
fn quantize_block(
    coeff: &mut [i32; 64],
    n: usize,
    quant: &Quant,
    y_dc_scale: i32,
    c_dc_scale: i32,
) {
    let dc_scale = if n < 4 { y_dc_scale } else { c_dc_scale };
    coeff[0] = div_round(coeff[0], dc_scale);
    for (k, c) in coeff.iter_mut().enumerate().skip(1) {
        *c = quant.quantize(k, *c, true);
    }
}

/// 取整除法（`b > 0`，`a` 可负）—— 四舍五入、远离零。
///
/// 只在 DC 那条路（每块一次）与测试里用；AC 那条走下面的倒数乘法。
fn div_round(a: i32, b: i32) -> i32 {
    debug_assert!(b > 0);
    if a >= 0 {
        (a + b / 2) / b
    } else {
        -((-a + b / 2) / b)
    }
}

/// `ceil(2^32 / b)` —— 「除以常数」那套魔数。
///
/// 依据：`(n * ceil(2^32/b)) >> 32 == n / b` 在 `b ∈ [1,128]`、`n < 2^20` 上**逐值精确**
/// （穷举验证过；见 `ac_quantization_matches_division` 测试）。范围够用：
/// `qmul = 2*qscale ≤ 62`，而 `n = |a| + qmul/2`，`|a| ≤ |DCT 系数| + qadd < 2^13`。
const fn recip(b: u32) -> u32 {
    (((1u64 << 32) + b as u64 - 1) / b as u64) as u32
}

/// `ceil(2^48 / b)` —— MPEG 那两条路用（除数大得多，2^32 那档的精度余量不够）。
///
/// 依据同 `recip`：`(n · ceil(2^S/b)) >> S == n / b` 只要 `n·b < 2^S`。
/// 实测范围：`n ≤ |f|·128 + 偏置 < 2^19`、`b = 8·step ≤ 2^15` ⇒ `n·b < 2^34 ≪ 2^48`。
const fn recip48(b: u64) -> u64 {
    ((1u64 << 48) + b - 1) / b
}

/// `qmul` 只取偶数 `2*qscale`（`qscale ∈ 1..=31`）—— 表按 `qmul` 直接索引。
static INV_QMUL: [u32; 64] = {
    let mut t = [0u32; 64];
    let mut i = 1;
    while i < 64 {
        t[i] = recip(i as u32);
        i += 1;
    }
    t
};
// ───────────────────────── DC 预测 ─────────────────────────

/// DC 预测网格。存的是 `level * dc_scale`（与 ffmpeg 的 `dc_val` 同口径），
/// 并且**截到 `[0, 2047]`** —— 解码端就是这么截的（`ff_mpeg4_pred_dc`），
/// 不跟着截，预测值会在极端块上分叉。
///
/// ⚠️ **Cb 与 Cr 必须是两张独立的表**（ffmpeg 是 `dc_val[1]` / `dc_val[2]`）。
/// 两者格位完全相同（都是 `(mbx, mby)`），合一张表就会让 Cb 的预测值串到 Cr 上 ——
/// 位流照样能编出来、单看也「像那么回事」，但解码端对不上。
struct DcGrid {
    /// 亮度：`2*mb_w × 2*mb_h` 个 8×8 块。
    luma: Vec<i32>,
    /// 色度：`[0] = Cb`、`[1] = Cr`，各 `mb_w × mb_h`。
    chroma: [Vec<i32>; 2],
    mb_w: usize,
    y_scale: i32,
    c_scale: i32,
}

impl DcGrid {
    fn new(mb_w: usize, mb_h: usize, y_scale: i32, c_scale: i32) -> Self {
        Self {
            luma: vec![DC_NONE; 2 * mb_w * 2 * mb_h],
            chroma: [vec![DC_NONE; mb_w * mb_h], vec![DC_NONE; mb_w * mb_h]],
            mb_w,
            y_scale,
            c_scale,
        }
    }

    /// 读一个格子的 DC 值；越界返回 `DC_NONE`。
    fn at(&self, n: usize, col: i32, row: i32) -> i32 {
        if col < 0 || row < 0 {
            return DC_NONE;
        }
        let (c, r) = (col as usize, row as usize);
        if n < 4 {
            let w = 2 * self.mb_w;
            if c >= w || r >= self.luma.len() / w {
                return DC_NONE;
            }
            self.luma[r * w + c]
        } else {
            let p = &self.chroma[n - 4];
            if c >= self.mb_w || r >= p.len() / self.mb_w {
                return DC_NONE;
            }
            p[r * self.mb_w + c]
        }
    }

    /// 算预测**水平**（不是系数）与方向：`0` = 取左（A）、`1` = 取上（C）。
    ///
    /// 判据是 `|a-b| < |b-c|` → 取上。注意 MP43 那边是 `<=`，差一个等号，
    /// 会让部分宏块的预测方向反过来 —— 两边的代码不能互相抄。
    fn predict(&self, n: usize, mbx: usize, mby: usize) -> (i32, usize) {
        let (col, row) = cell_of(n, mbx, mby);
        let a = self.at(n, col - 1, row);
        let b = self.at(n, col - 1, row - 1);
        let c = self.at(n, col, row - 1);
        let (pred, dir) = if (a - b).abs() < (b - c).abs() {
            (c, 1)
        } else {
            (a, 0)
        };
        let scale = if n < 4 { self.y_scale } else { self.c_scale };
        // 网格里存的是 level*scale，这里换回 level。`+scale/2` 是四舍五入。
        ((pred + (scale >> 1)) / scale, dir)
    }

    /// 把本块的**最终** DC level 写回网格（解码端就是这么存的）。
    fn update(&mut self, level: i32, n: usize, mbx: usize, mby: usize) {
        let scale = if n < 4 { self.y_scale } else { self.c_scale };
        let v = (level * scale).clamp(0, 2047);
        self.set(n, mbx, mby, v);
    }

    /// 直接写一个格子（`snap` / `restore` 用）。
    fn set(&mut self, n: usize, mbx: usize, mby: usize, v: i32) {
        let (col, row) = cell_of(n, mbx, mby);
        let (c, r) = (col as usize, row as usize);
        if n < 4 {
            self.luma[r * (2 * self.mb_w) + c] = v;
        } else {
            self.chroma[n - 4][r * self.mb_w + c] = v;
        }
    }

    /// 读本宏块自己的 6 个格子（亮度 4 + 色度 2）。
    fn snap(&self, mbx: usize, mby: usize) -> [i32; 6] {
        let mut out = [DC_NONE; 6];
        for (n, slot) in out.iter_mut().enumerate() {
            let (col, row) = cell_of(n, mbx, mby);
            *slot = self.at(n, col, row);
        }
        out
    }

    /// 还原 `snap` 拍下的 6 个格子。
    ///
    /// 因为每个格子一帧里只被写一次，快照的值必然是初值 —— 也就是说「还原」等价于
    /// 「清成 1024」，正是解码端 `ff_clean_intra_table_entries` 对**帧间宏块**做的事。
    fn restore(&mut self, mbx: usize, mby: usize, snap: &[i32; 6]) {
        for (n, &v) in snap.iter().enumerate() {
            self.set(n, mbx, mby, v);
        }
    }
}

/// 块 `n` 在 DC 网格里的格位 `(列, 行)`。
fn cell_of(n: usize, mbx: usize, mby: usize) -> (i32, i32) {
    if n < 4 {
        ((2 * mbx + (n & 1)) as i32, (2 * mby + (n >> 1)) as i32)
    } else {
        (mbx as i32, mby as i32)
    }
}

// ───────────────────────── AC 预测 ─────────────────────────

/// AC 预测网格：每个块留 16 个 i16 给右边/下面的邻居。
///
/// 与 `DcGrid` 同理，**Cb / Cr 必须分开**（ffmpeg 是 `ac_val[1]` / `ac_val[2]`）。
struct AcGrid {
    luma: Vec<AcVal>,
    chroma: [Vec<AcVal>; 2],
    mb_w: usize,
}

impl AcGrid {
    fn new(mb_w: usize, mb_h: usize) -> Self {
        Self {
            luma: vec![[0i16; 16]; 2 * mb_w * 2 * mb_h],
            chroma: [vec![[0i16; 16]; mb_w * mb_h], vec![[0i16; 16]; mb_w * mb_h]],
            mb_w,
        }
    }

    /// 取块 `n` 的邻居留给它的值。
    ///
    /// 越界（左邻居在帧外 / 上邻居在帧外）返回全 0 —— ffmpeg 的 `ac_val` 是
    /// 零初始化的，帧外格子从没被写过，读出来就是 0。
    fn neighbor(&self, n: usize, dir: usize, cell: (usize, usize)) -> AcVal {
        let (col, row) = cell;
        if dir == 0 {
            if col == 0 {
                return [0i16; 16];
            }
            self.get(n, col - 1, row)
        } else {
            if row == 0 {
                return [0i16; 16];
            }
            self.get(n, col, row - 1)
        }
    }

    fn get(&self, n: usize, col: usize, row: usize) -> AcVal {
        if n < 4 {
            self.luma[row * (2 * self.mb_w) + col]
        } else {
            self.chroma[n - 4][row * self.mb_w + col]
        }
    }

    /// 存回本块的首列/首行（**预测前**的值）。
    fn put(&mut self, n: usize, col: usize, row: usize, v: AcVal) {
        if n < 4 {
            self.luma[row * (2 * self.mb_w) + col] = v;
        } else {
            self.chroma[n - 4][row * self.mb_w + col] = v;
        }
    }

    /// 读本宏块自己的 6 个格子（亮度 4 + 色度 2）。
    fn snap(&self, mbx: usize, mby: usize) -> [AcVal; 6] {
        let mut out = [[0i16; 16]; 6];
        for (n, slot) in out.iter_mut().enumerate() {
            let (col, row) = cell_of(n, mbx, mby);
            *slot = self.get(n, col as usize, row as usize);
        }
        out
    }

    /// 还原 `snap` 拍下的 6 个格子（= 清成全 0，与解码端对帧间宏块做的一致）。
    fn restore(&mut self, mbx: usize, mby: usize, snap: &[AcVal; 6]) {
        for (n, &v) in snap.iter().enumerate() {
            let (col, row) = cell_of(n, mbx, mby);
            self.put(n, col as usize, row as usize, v);
        }
    }
}

/// `ff_mpeg4_pred_ac` 的编码侧：把邻居的首行/首列**减**掉。
///
/// - `dir == 0`（取左）：减邻居的**首列**，落在本块的首列（`block[i*8]`）。
/// - `dir == 1`（取上）：减邻居的**首行**，落在本块的首行（`block[i]`）。
///
/// `i` 从 1 起 —— 下标 0 是 DC，不参与。
fn apply_ac_pred(block: &mut [i32; 64], nb: &AcVal, dir: usize) {
    if dir == 0 {
        for i in 1..8 {
            block[i * 8] -= i32::from(nb[i]);
        }
    } else {
        for i in 1..8 {
            block[i] -= i32::from(nb[8 + i]);
        }
    }
}

/// 把本块的首列/首行存回网格（存**预测前**的值）。
fn store_ac_val(ac: &mut AcGrid, n: usize, cell: (usize, usize), block: &[i32; 64]) {
    let mut v = [0i16; 16];
    for i in 1..8 {
        v[i] = block[i * 8] as i16; // 首列 → 给右边
        v[8 + i] = block[i] as i16; // 首行 → 给下面
    }
    ac.put(n, cell.0, cell.1, v);
}

// ───────────────────────── 熵编码 ─────────────────────────

/// 一个块的 AC：MPEG-4 的 RL 前缀码 + 三种 escape。
///
/// `intra` 决定起点：帧内从扫描序**第 1 个**开始（下标 0 是 DC，走单独的 DC VLC）；
/// 帧间从**第 0 个**开始（帧间没有 DC 这一趟，DC 就是普通 AC 系数）。
///
/// 与 MP43 的 `write_ac` **不是一回事**：MP43 的三种 escape 含义不同
/// （第一 escape 补 level、第二 escape 补 run、第三 escape 明文 1+6+8 位）。
/// 这里 ffmpeg 是预先算好一张「统一表」在 ESC0/ESC1/ESC2/ESC3 里挑最短的 ——
/// 我们**在写的时候现算**同样的挑选，省掉那张 8192 项的表。
fn write_ac(
    w: &mut BitWriter,
    rl: &RlTable,
    block: &[i32; 64],
    scan: &[u8; 64],
    intra: bool,
) -> Result<()> {
    let start = if intra { 1 } else { 0 };
    let Some(last_index) = (start..64).rev().find(|&i| block[scan[i] as usize] != 0) else {
        return Ok(());
    };
    // `last_non_zero` 初值 = 「起点之前那一个」：帧内是 0（DC），帧间是 -1。
    let mut last_non_zero: i32 = start as i32 - 1;
    for i in start..=last_index {
        let level = block[scan[i] as usize];
        if level == 0 {
            continue;
        }
        let run = (i as i32 - last_non_zero - 1) as usize;
        write_rl(w, rl, i == last_index, run, level)?;
        last_non_zero = i as i32;
    }
    Ok(())
}

/// 一个 `(last, run, level)` 的最省写法。
///
/// 四种候选（`init_uni_mpeg4_rl_tab` 的逻辑，现算而不是查表）：
///
/// | | 位流 | 长度 |
/// |---|---|---|
/// | ESC0 | `code` + sign | `len(code)+1` |
/// | ESC1 | `esc` `0` `code'` + sign，其中 `level' = level - max_level[last][run]` | `len(esc)+1+len(code')+1` |
/// | ESC2 | `esc` `10` `code''` + sign，其中 `run' = run - max_run[last][level] - 1` | `len(esc)+2+len(code'')+1` |
/// | ESC3 | `esc` `11` last(1) run(6) 1 level(12) 1 | `len(esc)+23` |
///
/// 严格取最短的（并列时取靠前的候选，与 ffmpeg 的 `len < best` 一致）。
fn write_rl(w: &mut BitWriter, rl: &RlTable, last: bool, run: usize, level: i32) -> Result<()> {
    let (code, len) = best_rl(rl, last, run, level)?;
    debug_assert!(len <= 32, "一次最多写 32 位，ESC3 是 {len} 位");
    w.put(code, len);
    Ok(())
}

/// 一个 `(last, run, level)` 的**最省写法**：`(code, len)`。
///
/// ⚠️ 位数的唯一真源 —— 写位流（`write_rl`）与代价模型（`rl_cost`）都走这里。
/// 各算一份必然分叉（本仓踩过：`band_cost` 与 `write_band` 的 escape 计数不一样）。
fn best_rl(rl: &RlTable, last: bool, run: usize, level: i32) -> Result<(u32, u32)> {
    let esc = rl.escape_symbol();
    let (esc_code, esc_len) = rl
        .code_len(esc)
        .ok_or(Error::Bad("RL 表里没有 escape 码"))?;
    let li = usize::from(last);
    let lvl = level.abs();
    let sign = u32::from(level < 0);

    // 候选：`(code, len)` —— 取最短。
    let mut best: Option<(u32, u32)> = None;
    let better = |code: u32, len: u32, best: &mut Option<(u32, u32)>| {
        if best.is_none_or(|(_, bl)| len < bl) {
            *best = Some((code, len));
        }
    };

    // ESC0
    let c0 = rl.index_of(li, run, lvl);
    if c0 != esc {
        if let Some((code, len)) = rl.code_len(c0) {
            better(code << 1 | sign, len + 1, &mut best);
        }
    }
    // ESC1
    let level1 = lvl - rl.max_level_at(li, run);
    if level1 > 0 {
        let c1 = rl.index_of(li, run, level1);
        if c1 != esc {
            if let Some((code, len)) = rl.code_len(c1) {
                let bits = ((esc_code << 1) << len | code) << 1 | sign;
                better(bits, esc_len + 1 + len + 1, &mut best);
            }
        }
    }
    // ESC2
    let run1 = run as i32 - rl.max_run_at(li, lvl) - 1;
    if run1 >= 0 {
        let c2 = rl.index_of(li, run1 as usize, lvl);
        if c2 != esc {
            if let Some((code, len)) = rl.code_len(c2) {
                let bits = ((esc_code << 2 | 0b10) << len | code) << 1 | sign;
                better(bits, esc_len + 2 + len + 1, &mut best);
            }
        }
    }
    // ESC3：`esc` `11` last run(6) 1 level(12) 1 —— 12 位有符号，用补码低 12 位。
    {
        let bits = ((((esc_code << 2 | 0b11) << 1 | u32::from(last)) << 6 | run as u32) << 1 | 1)
            << 12
            | ((level as u32) & 0xfff);
        let bits = bits << 1 | 1;
        better(bits, esc_len + 23, &mut best);
    }

    best.ok_or(Error::Bad("AC 系数写不出来"))
}

/// 一个 `(last, run, level)` 要花多少位（`best_rl` 的长度那一半）。
fn rl_cost(rl: &RlTable, last: bool, run: usize, level: i32) -> u32 {
    best_rl(rl, last, run, level).map_or(32, |(_, len)| len)
}

/// 系数级 RD（trellis）：在**扫描序**上做一遍 DP，逐系数在 `{L, L∓1}` 里挑
/// `失真 + λ·位数` 最小的那个。
///
/// ## 为什么值得做
///
/// 只做「逐系数取最近电平」时，一个系数的编码代价（run / escape / 符号）**完全不参与决策**：
/// 花 20 位才写得出来、却只换回一点失真的系数照写不误。DP 把失真和位数放在同一个尺度上比 ——
/// 这也正是**感知加权唯一能生效的那一层**（权重必须和率项同时出现，见 TODO 的 5109 §1）。
///
/// ## 设计取舍（都是有意的收缩，别当成 bug）
///
/// - **只用于帧间块**：帧内块的扫描序随 AC 预测方向变（`H_SCAN` / `V_SCAN`），
///   要接进来得连 `intra_try` 的候选一起改 —— 那一层先不动。
/// - **候选只有 `{L, L∓1}`，且不新增非零系数**（基线为 0 的位置保持 0）：
///   收益的大头是「把不划算的系数砍掉」，提电平那一半先不做。
/// - **`last` 按基线算**：真把基线的「最后一个非零」砍掉时，代价模型会差 1–2 位
///   （写出端自己算得对，只有这里的估计偏一点）。
/// - **λ 的尺度是坑**：`dist` 是 DCT 域的平方误差（量级 1e6~1e8），`bits` 是个位数，
///   而且步长变粗时失真单位跟着 `q²` 涨 —— 所以 λ 必须带着 `q²` 走
///   （本仓在另一条线上踩过「λ 比实测最优大 90 倍」）。
fn trellis_levels(
    quant: &Quant,
    rl: &RlTable,
    coeff: &[i32; 64],
    levels: &mut [i32; 64],
    lambda: i64,
) {
    const INF: i64 = i64::MAX / 4;
    // 基线（扫描序）+ 代价模型里的 `last`。
    let mut base = [0i32; 64];
    for (i, b) in base.iter_mut().enumerate() {
        *b = levels[ZIGZAG[i] as usize];
    }
    let last_index = (0..64).rev().find(|&i| base[i] != 0).unwrap_or(0);

    // 状态 = 最后一个非零系数的扫描序下标 + 1（0 = 还没有）；代价表滚动用。
    let mut cost = [[INF; 65]; 2];
    // `choice[出状态][i] = (候选下标, 入状态)` —— 2 字节，回溯靠它，
    // 不能只存候选：`level != 0` 时同一个出状态来自哪个入状态，决定了 run、也就决定了位数。
    let mut choice = [[(0u8, 0u8); 64]; 65];
    cost[0][0] = 0;
    for i in 0..64 {
        let (cur, nxt) = (i & 1, (i & 1) ^ 1);
        cost[nxt] = [INF; 65];
        let l0 = base[i];
        // 候选：**0** 与 `L∓1 / L`。0 必须在里面 —— 「砍掉不划算的系数」是收益最大的一类，
        // 而只给 `{L, L∓1}` 时 `|L| >= 2` 的系数根本够不到 0（这条是单测逼出来的）。
        let (mut cands, n_cands) = if l0 == 0 {
            ([0i32; 4], 1usize)
        } else {
            let s = l0.signum();
            let mut c = [0i32; 4];
            let mut n = 0usize;
            if TRELLIS_ALLOW_ZERO {
                c[0] = 0;
                n = 1;
            }
            for cand in [l0 - s, l0, l0 + s] {
                if cand != 0 && !c[..n].contains(&cand) {
                    c[n] = cand;
                    n += 1;
                }
            }
            (c, n)
        };
        for c in cands.iter_mut() {
            *c = (*c).clamp(-2047, 2047); // ESC3 的 level 是 12 位有符号
        }
        let f = i64::from(coeff[ZIGZAG[i] as usize]);
        for prev in 0..65usize {
            if cost[cur][prev] >= INF {
                continue;
            }
            for (ci, &l) in cands.iter().take(n_cands).enumerate() {
                let dq = i64::from(quant.dequant(ZIGZAG[i] as usize, l, false));
                let mut c = cost[cur][prev] + (f - dq) * (f - dq);
                let nprev = if l != 0 { i + 1 } else { prev };
                if l != 0 {
                    let run = (i as i32 - (prev as i32 - 1) - 1).max(0) as usize;
                    c += lambda * i64::from(rl_cost(rl, i == last_index, run, l));
                }
                if c < cost[nxt][nprev] {
                    cost[nxt][nprev] = c;
                    choice[nprev][i] = (ci as u8, prev as u8);
                }
            }
        }
    }

    // 回溯（`i = 63` 时 `nxt == 0`，所以终局代价在 `cost[0]`）。
    let mut st = (0..65usize).min_by_key(|&p| cost[0][p]).unwrap_or(0);
    let mut out = [0i32; 64];
    for i in (0..64).rev() {
        let (ci, in_st) = choice[st][i];
        let l0 = base[i];
        out[i] = if l0 == 0 {
            0
        } else {
            let s = l0.signum();
            let mut c = [0i32; 4];
            let mut n = 1usize;
            for cand in [l0 - s, l0, l0 + s] {
                if cand != 0 && !c[..n].contains(&cand) {
                    c[n] = cand;
                    n += 1;
                }
            }
            let _ = n;
            c[ci as usize].clamp(-2047, 2047)
        };
        st = in_st as usize;
    }
    for (i, &l) in out.iter().enumerate() {
        levels[ZIGZAG[i] as usize] = l;
    }
}

/// `mpeg4_encode_dc`：VLC 的**符号是 DC 差的位长**，后面跟位长个位。
///
/// 值的写法是 `get_xbits` 语义：非负照写，负数是 `(-|v|) ^ ((1<<size)-1)`。
/// `size > 8` 时后面还有一个 marker 位（值为 1）。
fn write_dc(w: &mut BitWriter, level: i32, luma: bool) -> Result<()> {
    if !(-2047..=2047).contains(&level) {
        return Err(Error::Bad("DC 差超出 DC 表能表示的范围"));
    }
    let size = bit_length(level.unsigned_abs()) as usize;
    let tab = if luma {
        &tables::DC_LUM
    } else {
        &tables::DC_CHROM
    };
    let (code, len) = tab
        .get(size)
        .map(|e| (e[0], e[1]))
        .ok_or(Error::Bad("DC 位长超出 DC 表"))?;
    w.put(u32::from(code), u32::from(len));
    if size > 0 {
        let payload = if level < 0 {
            ((-level) as u32) ^ ((1u32 << size) - 1)
        } else {
            level as u32
        };
        w.put(payload, size as u32);
        if size > 8 {
            w.put(1, 1); // marker
        }
    }
    Ok(())
}

/// `|v|` 的位长（`v == 0` → 0）。
fn bit_length(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        32 - v.leading_zeros()
    }
}

/// I 宏块（以及 P 帧里的帧内宏块）的 MCBPC：低 2 位是色度 CBP，
/// 第 3 位（值 4）表示「本宏块带 dquant」。不做 dquant，所以只用 0..=3。
fn write_intra_mcbpc(w: &mut BitWriter, cbpc: usize) -> Result<()> {
    let code = *tables::INTRA_MCBPC_CODE
        .get(cbpc)
        .ok_or(Error::Bad("cbpc 越界"))?;
    let bits = *tables::INTRA_MCBPC_BITS
        .get(cbpc)
        .ok_or(Error::Bad("cbpc 越界"))?;
    w.put(u32::from(code), u32::from(bits));
    Ok(())
}

/// P 帧的 28 项 MCBPC。索引的含义见 `tables.rs` 里那段注释：
/// `0..3` 帧间、`4..7` 帧内、`8..15` 带 dquant、`16..19` 8×8 MV、`20..23` stuffing。
fn write_inter_mcbpc(w: &mut BitWriter, cbpc: usize) -> Result<()> {
    let code = *tables::INTER_MCBPC_CODE
        .get(cbpc)
        .ok_or(Error::Bad("P 帧 cbpc 越界"))?;
    let bits = *tables::INTER_MCBPC_BITS
        .get(cbpc)
        .ok_or(Error::Bad("P 帧 cbpc 越界"))?;
    // stuffing 那 4 项码长是 0 —— 写进去等于没写，位流会静默错位。
    if bits == 0 {
        return Err(Error::Bad("P 帧 cbpc 落到了 stuffing 项"));
    }
    w.put(u32::from(code), u32::from(bits));
    Ok(())
}

/// CBPY：`ff_h263_cbpy_tab[cbpy] = (code, len)`。
///
/// ⚠️ **取不取反由帧类型决定**，这是最容易抄错的一处：
/// - MPEG-4 的 **I 帧** 与 **P 帧里的帧内宏块**：直取；
/// - MPEG-4 的 **帧间宏块**：`cbpy ^ 0xF`（`mpeg4_decode_mb` 帧间分支就是 `get_vlc2(...) ^ 0x0F`）。
///
/// 取反在调用方做（`write_mb` / `write_inter_mb`），这里只管查表。
fn write_cbpy(w: &mut BitWriter, cbpy: usize) -> Result<()> {
    let e = tables::CBPY.get(cbpy).ok_or(Error::Bad("cbpy 越界"))?;
    w.put(u32::from(e[0]), u32::from(e[1]));
    Ok(())
}

/// 一个 MV 分量（`ff_h263_encode_motion`）。
///
/// 位流形状：`ff_mvtab[code]` + 符号位 + `bit_size` 个低位；`val == 0` 时只写一个 `1` 位。
/// `code` 是**绝对值的量化档**而不是矢量本身 ——
/// 解码端 `val = (code - 1) << bit_size` 再补低位、读符号，所以 `code = (|d| - 1) >> bit_size + 1`。
///
/// 我们恒用 `f_code = 1` → `bit_size = 0`，于是 `code = |d|`、不写低位。
fn write_motion(w: &mut BitWriter, d: i32) -> Result<()> {
    if d == 0 {
        // 零矢量就是 `ff_mvtab[0]`：一个 `1` 位，后面没有符号位也没有低位。
        w.put(1, 1);
        return Ok(());
    }
    let bit_size = u32::from(F_CODE) - 1;
    let range = 1i32 << bit_size;
    // 模 `2^(6 + bit_size)` 回绕 —— 与解码端对**重建后的绝对 MV** 做的那次
    // `sign_extend(val, 5 + f_code)` 是同一个模数（5 + f_code == 6 + bit_size）。
    let d = sign_extend(d, 6 + bit_size);
    let sign = u32::from(d < 0);
    let mag = d.unsigned_abs() - 1;
    let code = (mag >> bit_size) + 1;
    let low = mag & (range as u32 - 1);
    let (c, len) = tables::MV_TAB
        .get(code as usize)
        .map(|e| (e[0], e[1]))
        .ok_or(Error::Bad("MV 超出 ff_mvtab 能表示的范围"))?;
    w.put((u32::from(c) << 1) | sign, u32::from(len) + 1);
    if bit_size > 0 {
        w.put(low, bit_size);
    }
    Ok(())
}

/// 按 `bits` 位做符号扩展（`ffmpeg` 的 `sign_extend`）。
fn sign_extend(x: i32, bits: u32) -> i32 {
    let m = 1i32 << (bits - 1);
    ((x & ((1 << bits) - 1)) ^ m) - m
}

// ───────────────────────── 像素搬运 ─────────────────────────

/// 把帧补齐到宏块对齐（边缘复制）。返回 `(y, u, v)` 三块平面。
fn pad_planes(frame: &Frame, pw: usize, ph: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let (cw, ch) = (w / 2, h / 2);
    let (pcw, pch) = (pw / 2, ph / 2);

    let mut y = vec![0u8; pw * ph];
    for r in 0..ph {
        let src = &frame.y[r.min(h - 1) * w..][..w];
        let dst = &mut y[r * pw..][..pw];
        dst[..w].copy_from_slice(src);
        // 右边补齐：边缘复制。
        dst[w..].fill(src[w - 1]);
    }

    let pad_chroma = |src: &[u8]| {
        let mut out = vec![0u8; pcw * pch];
        for r in 0..pch {
            let s = &src[r.min(ch - 1) * cw..][..cw];
            let d = &mut out[r * pcw..][..pcw];
            d[..cw].copy_from_slice(s);
            d[cw..].fill(s[cw - 1]);
        }
        out
    };
    (y, pad_chroma(&frame.u), pad_chroma(&frame.v))
}

/// 取宏块 `(mbx, mby)` 里第 `n` 个 8×8 块的平面、行距与左上角坐标。
fn block_at<'a>(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    pw: usize,
    mbx: usize,
    mby: usize,
    n: usize,
) -> (&'a [u8], usize, usize, usize) {
    if n < 4 {
        let px = mbx * 16 + (n & 1) * 8;
        let py = mby * 16 + (n >> 1) * 8;
        (y, pw, px, py)
    } else if n == 4 {
        (u, pw / 2, mbx * 8, mby * 8)
    } else {
        (v, pw / 2, mbx * 8, mby * 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(w: u32, h: u32, luma: u8, chroma: u8) -> Frame {
        Frame {
            width: w,
            height: h,
            y: vec![luma; (w * h) as usize],
            u: vec![chroma; (w * h / 4) as usize],
            v: vec![chroma; (w * h / 4) as usize],
        }
    }

    #[test]
    fn bit_length_is_the_index_of_the_top_set_bit_plus_one() {
        assert_eq!(bit_length(0), 0);
        assert_eq!(bit_length(1), 1);
        assert_eq!(bit_length(2), 2);
        assert_eq!(bit_length(3), 2);
        assert_eq!(bit_length(255), 8);
        assert_eq!(bit_length(256), 9);
    }

    #[test]
    /// **写出去的电平必须装得进 ESC3 的 12 位字段**。
    ///
    /// ESC3 写的是 `(level as u32) & 0xfff` —— 超了**静默回绕**，而编码器的重建用的是
    /// 原值 ⇒ 编解码从那个系数起分叉（解码器还会报 `|level| overflow in 3. esc`）。
    /// 所以 `quantize` 自己就把电平夹住：
    ///
    /// - **帧间**直接写电平 ⇒ 12 位有符号，±2047；
    /// - **帧内**写的是「电平 − 预测」，而预测取自邻居**同样被夹过**的电平 ⇒
    ///   两边各 ±1023 就够保证差分 ≤ 2046。
    ///
    /// ⚠️ 只喂**域内**系数：`quantize` 的倒数乘法要求 `|f| < 2^13`（见 `recip` 的说明），
    /// 超出会溢出 —— debug 下直接 panic，那是另一回事，不在这条的射程里。
    /// 正常内容到不了夹取（真样本上它是**逐字节的 no-op**，见 TODO.md），
    /// 所以末尾有一条「必须真被触发」的反空转断言。
    #[test]
    fn quantized_levels_fit_the_escape_field() {
        let mut clamped = 0usize;
        for qscale in [1u8, 2, 3, 8, 31] {
            for mpeg in [false, true] {
                let q = Quant::new(qscale, mpeg);
                for &f in &[8191i32, -8191, 4096, -2048, 1024, -513] {
                    for k in [1usize, 2, 5, 17, 63] {
                        let inter = q.quantize(k, f, false);
                        assert!(
                            inter.abs() <= 2047,
                            "帧间电平 {inter} 装不进 12 位（q{qscale} mpeg{mpeg} k{k} f{f}）"
                        );
                        let intra = q.quantize(k, f, true);
                        assert!(
                            intra.abs() <= 1023,
                            "帧内电平 {intra} 会让差分越界（q{qscale} mpeg{mpeg} k{k} f{f}）"
                        );
                        // 两个被夹过的帧内电平之差也必须装得下。
                        let diff = intra - (-1023);
                        assert!(diff.abs() <= 2047, "差分 {diff} 越界");
                        if inter.abs() == 2047 || intra.abs() == 1023 {
                            clamped += 1;
                        }
                    }
                }
            }
        }
        assert!(clamped > 0, "夹取一次都没触发 —— 这条测试是空的");
    }

    #[test]
    /// 三条路的倒数乘法都必须与原式**逐值相同** —— 与 `div_round` 对拍。
    #[test]
    fn quantize_matches_division() {
        for qscale in 1..=31u8 {
            for mpeg in [false, true] {
                let q = Quant::new(qscale, mpeg);
                for k in [0usize, 1, 3, 17, 40, 63] {
                    for intra in [true, false] {
                        for f in (-9000..=9000i32).step_by(11) {
                            let want = if f == 0 {
                                0
                            } else if !mpeg {
                                if f > 0 {
                                    div_round(f - q.qadd, q.qmul).max(0)
                                } else {
                                    div_round(f + q.qadd, q.qmul).min(0)
                                }
                            } else {
                                let step = q.step(k, intra);
                                let v = if intra {
                                    (f.abs() * 128 + INTRA_BIAS_EIGHTHS * step) / (8 * step)
                                } else {
                                    div_round(f.abs() * 32 - step, 2 * step).max(0)
                                };
                                if f < 0 { -v } else { v }
                            };
                            // `quantize` 末尾还有一道「夹进 12 位可表示范围」的策略
                            // （见 `quantized_levels_fit_the_escape_field`）——
                            // 这条测的是**算术**，所以把它也算进来，别让两边互相打架。
                            let lim = if intra { 1023 } else { 2047 };
                            let want = want.clamp(-lim, lim);
                            assert_eq!(q.quantize(k, f, intra), want, "q{qscale} mpeg{mpeg} k{k} intra{intra} f{f}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn quantize_ac_inverts_the_h263_dequant() {
        // 解码端：`level > 0 ? level*qmul + qadd : level*qmul - qadd`
        let quant = Quant::new(4, false);
        let (qmul, qadd) = (2 * 4, (4 - 1) | 1);
        for level in [-20i32, -3, -1, 0, 1, 3, 20] {
            let f = if level > 0 {
                level * qmul + qadd
            } else if level < 0 {
                level * qmul - qadd
            } else {
                0
            };
            assert_eq!(quant.quantize(3, f, true), level, "level {level} 没还原");
        }
        // 落在两个 level 之间时取最近的
        assert_eq!(quant.quantize(3, qmul + qadd - 1, true), 1);
        assert_eq!(quant.quantize(3, qmul + qadd - 4, true), 1); // 还在上半区
    }

    #[test]
    fn mpeg_quant_round_trips_through_its_own_dequant() {
        // 两套规则各自的「量化 → 反量化」自洽性：重建值必须落在半个步长之内。
        // 顺带印证帧间是**奇数倍**电平（`2|level|+1`）、帧内是整数倍。
        for qscale in [1u8, 4, 12, 31] {
            let quant = Quant::new(qscale, true);
            for k in [0usize, 1, 5, 20, 63] {
                for f in [-900i32, -137, -8, 0, 7, 138, 901] {
                    for intra in [true, false] {
                        let level = quant.quantize(k, f, intra);
                        let back = quant.dequant(k, level, intra);
                        let step = quant.step(k, intra);
                        // 帧内步长 `step/16`、帧间 `step/32`，取宽松的一侧比较
                        let tol = step / 16 + 1;
                        assert!(
                            (back - f).abs() <= tol,
                            "q={qscale} k={k} intra={intra} f={f} level={level} back={back}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mpeg_inter_reconstruction_levels_are_odd_multiples() {
        // 帧间：`dequant(L) == ((2L+1)·step) >> 5`，L=0 是唯一的例外
        // （解码端 `if (level)` 跳过零 → 0 就是 0，**不是** `step>>5`）。
        // 电平落在奇数倍上意味着死区是「半个电平」，抄错会整块偏半步。
        let quant = Quant::new(4, true);
        for k in [0usize, 1, 40] {
            let step = quant.step(k, false);
            assert_eq!(quant.dequant(k, 0, false), 0, "level==0 就是 0");
            for level in 1..6i32 {
                assert_eq!(
                    quant.dequant(k, level, false),
                    ((2 * level + 1) * step) >> 5
                );
                assert_eq!(
                    quant.dequant(k, -level, false),
                    -(((2 * level + 1) * step) >> 5)
                );
            }
        }
    }

    #[test]
    fn mpeg_intra_dequant_is_matrix_scaled() {
        // 帧内：`(|level|·2q·W) >> 4` —— 高频的步长比低频大（W 从 8 涨到 45）。
        let quant = Quant::new(6, true);
        for level in [1i32, -1, 7, -7] {
            for k in [1usize, 63] {
                let step = quant.step(k, true);
                let want = (level.abs() * step) >> 4;
                let want = if level < 0 { -want } else { want };
                assert_eq!(quant.dequant(k, level, true), want, "k={k} level={level}");
            }
        }
        // 高频步长必须比低频粗 —— 这就是「感知」那一半
        assert!(quant.step(63, true) > quant.step(1, true));
        assert!(quant.step(63, false) > quant.step(0, false));
    }

    /// 按 trellis 自己的模型算一个 level 排布的代价（失真 + λ·位数）——只服务下面那条对拍。
    fn model_cost(
        quant: &Quant,
        rl: &RlTable,
        coeff: &[i32; 64],
        levels: &[i32; 64],
        lambda: i64,
    ) -> i64 {
        let last_index = (0..64)
            .rev()
            .find(|&i| levels[ZIGZAG[i] as usize] != 0)
            .unwrap_or(0);
        let mut cost = 0i64;
        let mut prev: i32 = -1;
        for i in 0..64 {
            let k = ZIGZAG[i] as usize;
            let l = levels[k];
            let f = i64::from(coeff[k]);
            let dq = i64::from(quant.dequant(k, l, false));
            cost += (f - dq) * (f - dq);
            if l != 0 {
                let run = (i as i32 - prev - 1).max(0) as usize;
                cost += lambda * i64::from(rl_cost(rl, i == last_index, run, l));
                prev = i as i32;
            }
        }
        cost
    }

    #[test]
    fn trellis_dp_matches_brute_force_on_a_small_support() {
        // 非零系数少的时候可以把候选组合全枚举一遍，DP 必须取到同一个最小值 ——
        // 这条钉的是状态转移与回溯（出状态存候选、入状态用来还原 run）。
        let quant = Quant::new(6, true);
        let rl = inter_rl_table();
        let lambda = quant.lambda();
        let mut coeff = [0i32; 64];
        for (scan_i, v) in [(1usize, 260i32), (5, -170), (30, 90)] {
            coeff[ZIGZAG[scan_i] as usize] = v;
        }
        let mut base = [0i32; 64];
        for k in 0..64 {
            base[k] = quant.quantize(k, coeff[k], false);
        }
        let mut got = base;
        trellis_levels(&quant, &rl, &coeff, &mut got, lambda);

        let positions: Vec<usize> = (0..64)
            .filter(|&i| base[ZIGZAG[i] as usize] != 0)
            .collect();
        assert_eq!(positions.len(), 3, "前提：基线只有 3 个非零");
        let mut best = i64::MAX;
        for mask in 0..64usize {
            let mut cand = base;
            let mut m = mask;
            for &i in &positions {
                let k = ZIGZAG[i] as usize;
                let s = base[k].signum();
                let mut set = [0i32; 4];
                let mut n = 1usize;
                for cand_l in [base[k] - s, base[k], base[k] + s] {
                    if cand_l != 0 && !set[..n].contains(&cand_l) {
                        set[n] = cand_l;
                        n += 1;
                    }
                }
                cand[k] = set[m % n].clamp(-2047, 2047);
                m /= n;
            }
            best = best.min(model_cost(&quant, &rl, &coeff, &cand, lambda));
        }
        assert_eq!(
            model_cost(&quant, &rl, &coeff, &got, lambda),
            best,
            "DP 没取到暴力枚举的最小值"
        );
    }


    #[test]
    fn trellis_drops_a_coefficient_whose_bits_are_expensive() {
        // 孤零零一个高频系数：写它要 escape（20+ 位），换回的失真却很小 —— λ 够大时该砍掉。
        //
        // ⚠️ 这条用**显式的大 λ**：砍到 0 的失真代价是 `f²`（整格），比「挪一档」的
        // `(重建步长)²` 大得多，所以「该不该砍」用的 λ 比「该挪哪一档」的 λ 高一个量级。
        // 默认 λ 靠扫描标定（见 TODO），不在这条测试里钉。
        let quant = Quant::new(6, true);
        let rl = inter_rl_table();
        let k = ZIGZAG[58] as usize;
        let mut coeff = [0i32; 64];
        coeff[k] = 30; // 该位置 step = 360，level 1 的门槛是 22.5 —— 30 刚好过线
        let mut base = [0i32; 64];
        base[k] = quant.quantize(k, coeff[k], false);
        assert_eq!(base[k].abs(), 1, "前提：基线是 ±1（这样 0 才在候选集里）");
        let mut out = base;
        trellis_levels(&quant, &rl, &coeff, &mut out, 400);
        assert_eq!(out[k], 0, "λ 够大时，昂贵的高频小系数该被砍掉");
        assert!(
            model_cost(&quant, &rl, &coeff, &out, 400)
                <= model_cost(&quant, &rl, &coeff, &base, 400)
        );
        // 反过来：λ = 0（只看失真）必须留着它 —— 这半条是「砍」的对照
        let mut kept = base;
        trellis_levels(&quant, &rl, &coeff, &mut kept, 0);
        assert!(kept[k].abs() >= 1, "λ=0 时不该砍");
    }

    #[test]
    fn trellis_never_moves_a_level_by_more_than_one() {
        // 收缩的边界：只在 {L, L∓1} 里挑，且不许新增非零、不许变号、不越 ESC3。
        let quant = Quant::new(9, true);
        let rl = inter_rl_table();
        let mut seed = 0x1234_5678u32;
        for _ in 0..40 {
            let mut coeff = [0i32; 64];
            for v in coeff.iter_mut() {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                *v = ((seed >> 16) as i32 % 2001) - 1000;
            }
            let mut base = [0i32; 64];
            for k in 0..64 {
                base[k] = quant.quantize(k, coeff[k], false);
            }
            let mut out = base;
            trellis_levels(&quant, &rl, &coeff, &mut out, quant.lambda());
            for k in 0..64 {
                if base[k] == 0 {
                    assert_eq!(out[k], 0, "不许新增非零系数");
                } else if out[k] != 0 {
                    assert!(
                        (out[k] - base[k]).abs() <= 1,
                        "只许动一档（或直接归零）：k={k} base={} out={}",
                        base[k],
                        out[k]
                    );
                    assert_eq!(out[k].signum(), base[k].signum(), "不许变号");
                    assert!(out[k].abs() <= 2047, "不许越 ESC3 的 12 位");
                }
            }
        }
    }

    #[test]
    fn mismatch_control_flips_only_the_lsb_of_the_last_coefficient() {
        // 解码端 `dct_unquantize_mpeg2_inter_c`：`sum` **初值 −1**，累加 64 个反量化值，
        // `sum & 1` 时翻下标 63 的最低位 —— 换成「64 个值之和」的说法就是：
        // 和为**偶数**时翻（−1 把奇偶反了过来）。不做这一步，重建每块差一位、P 帧累积漂移。
        let off = Quant::new(6, false);
        let on = Quant::new(6, true);

        // ① 和为偶数（2 + 4 = 6）→ 翻；H.263 不翻
        let mut even = [0i16; 64];
        even[1] = 2;
        even[63] = 4;
        let mut a = even;
        on.mismatch_control(&mut a);
        assert_eq!(a[63], even[63] ^ 1, "和为偶数要翻最低位");
        assert_eq!(&a[..63], &even[..63], "只许动下标 63");
        let mut b = even;
        off.mismatch_control(&mut b);
        assert_eq!(b, even, "H.263 没有失配控制");

        // ② 和为奇数（2 + 15 = 17）→ 不翻
        let mut odd = [0i16; 64];
        odd[1] = 2;
        odd[2] = 15;
        let mut c = odd;
        on.mismatch_control(&mut c);
        assert_eq!(c, odd, "和为奇数不动");
    }

    #[test]
    fn dc_quantize_rounds_to_nearest() {
        // dc_scale = 8：level = round(F/8)，而 F = 8×均值 → level == 均值。
        for mean in [0i32, 1, 127, 128, 200, 255] {
            assert_eq!(div_round(8 * mean, 8), mean);
        }
    }

    #[test]
    fn dc_prediction_tie_goes_to_left_not_top() {
        // MPEG-4 是 `|a-b| < |b-c|` → 取上；相等时取**左**。
        // 这和 MP43 的 `<=`（相等时取上）差一个等号，是最容易抄错的地方。
        let g = DcGrid::new(1, 1, 8, 8);
        let (_, dir) = g.predict(0, 0, 0);
        // 越界时 a = b = c = 1024 → |0| < |0| 为假 → 取左
        assert_eq!(dir, 0, "三者相等时应当取左");
    }

    #[test]
    fn dc_prediction_reads_left_and_top_neighbours() {
        let mut g = DcGrid::new(2, 1, 8, 8);
        // 手工放两个块：宏块 0 的亮度块 0 = level 100，亮度块 1 = level 200。
        // 网格存的是 `level * dc_scale`。
        g.update(100, 0, 0, 0);
        g.update(200, 1, 0, 0);
        // 宏块 1 的块 0：a = 左邻居（宏块 0 的块 1）= 1600，b = 越界 = 1024，c = 越界 = 1024。
        // |a-b| = 576，|b-c| = 0 → 576 < 0 为假 → 取**左**。
        let (pred, dir) = g.predict(0, 1, 0);
        assert_eq!(dir, 0, "差得远的邻居反而应当被选上（判据只看两个差谁更小）");
        assert_eq!(pred, 200, "1600/8 = 200");
    }

    #[test]
    fn dc_prediction_prefers_the_closer_of_left_and_top() {
        // 判据是 `|a-b| < |b-c|` → 取上（c）。所以「取上」要的是 **a 与 b 相近、
        // c 与 b 差得远**，而不是「c 和本块相近」—— 这是最容易理解反的一处。
        //
        // 预测宏块 (1,1) 的亮度块 0（格位 (2,2)）：
        //   a = 格位 (1,2) = 宏块 (0,1) 的块 1
        //   b = 格位 (1,1) = 宏块 (0,0) 的块 3
        //   c = 格位 (2,1) = 宏块 (1,0) 的块 2
        let mut g = DcGrid::new(2, 2, 8, 8);
        g.update(50, 1, 0, 1); // a
        g.update(50, 3, 0, 0); // b
        g.update(10, 2, 1, 0); // c
        let (pred, dir) = g.predict(0, 1, 1);
        assert_eq!(dir, 1, "a 与 b 相等、c 离得远 → 取上");
        assert_eq!(pred, 10, "取上就用 c 的 level");
    }

    #[test]
    fn ac_val_layout_matches_the_prediction_loops() {
        // 首列落在 `[1..8]`、首行落在 `[9..16]` —— 和 `apply_ac_pred` 的读法必须对上。
        let mut ac = AcGrid::new(1, 1);
        let mut block = [0i32; 64];
        for i in 1..8 {
            block[i * 8] = i as i32 * 3; // 首列
            block[i] = i as i32 * 5; // 首行
        }
        store_ac_val(&mut ac, 0, (0, 0), &block);
        let v = ac.get(0, 0, 0);
        for i in 1..8 {
            assert_eq!(v[i], (i as i32 * 3) as i16, "首列第 {i} 项");
            assert_eq!(v[8 + i], (i as i32 * 5) as i16, "首行第 {i} 项");
        }
    }

    #[test]
    fn out_of_frame_ac_neighbours_are_zero() {
        let ac = AcGrid::new(2, 2);
        assert_eq!(ac.neighbor(0, 0, (0, 0)), [0i16; 16], "左边越界");
        assert_eq!(ac.neighbor(0, 1, (0, 0)), [0i16; 16], "上边越界");
    }

    /// Cb 与 Cr 的格位相同（都是 `(mbx, mby)`），所以**必须各有一张表**。
    ///
    /// 曾经把它们合在一张里：编出来的位流照样「像那么回事」，ffmpeg 却在第 3 个宏块
    /// 起报 `Error at MB: 2`、PSNR 掉到 12 dB。根因就是 Cb 的预测值串到了 Cr 上。
    #[test]
    fn cb_and_cr_never_share_a_dc_cell() {
        let mut g = DcGrid::new(2, 1, 8, 8);
        g.update(90, 4, 0, 0); // Cb 宏块 0 = level 90
        g.update(170, 5, 0, 0); // Cr 宏块 0 = level 170
        assert_eq!(g.predict(4, 1, 0).0, 90, "Cb 读到的是 Cb");
        assert_eq!(g.predict(5, 1, 0).0, 170, "Cr 读到的是 Cr");
    }

    #[test]
    fn cb_and_cr_never_share_an_ac_cell() {
        let mut ac = AcGrid::new(2, 1);
        let mut cb = [0i32; 64];
        let mut cr = [0i32; 64];
        for i in 1..8 {
            cb[i * 8] = 7; // Cb 首列
            cr[i * 8] = -9; // Cr 首列
        }
        store_ac_val(&mut ac, 4, (0, 0), &cb);
        store_ac_val(&mut ac, 5, (0, 0), &cr);
        assert_eq!(ac.neighbor(4, 0, (1, 0))[1], 7, "Cb 的首列");
        assert_eq!(ac.neighbor(5, 0, (1, 0))[1], -9, "Cr 的首列");
    }

    #[test]
    fn applying_ac_prediction_subtracts_the_right_edge() {
        let mut block = [0i32; 64];
        for i in 1..8 {
            block[i * 8] = 10;
            block[i] = 20;
        }
        let mut nb = [0i16; 16];
        for i in 1..8 {
            nb[i] = i as i16;
            nb[8 + i] = 100 + i as i16;
        }
        let mut left = block;
        apply_ac_pred(&mut left, &nb, 0);
        for i in 1..8 {
            assert_eq!(left[i * 8], 10 - i as i32, "取左：减首列");
            assert_eq!(left[i], 20, "取左：首行不动");
        }
        let mut top = block;
        apply_ac_pred(&mut top, &nb, 1);
        for i in 1..8 {
            assert_eq!(top[i * 8], 10, "取上：首列不动");
            assert_eq!(top[i], 20 - (100 + i as i32), "取上：减首行");
        }
    }

    #[test]
    fn write_dc_matches_the_bit_length_vlc() {
        // VLC 的**符号是位长**，所以总位数 = VLC 长度 + 位长（+ size > 8 时的一个 marker）。
        // 亮度表：`DC_LUM[0..4] = (3,3) (3,2) (2,2) (2,3) (1,3)`，之后一路 `(1, size+1)`。
        for (level, bits) in [
            (0i32, 3u32), // size 0 → VLC 3 位
            (1, 3),       // size 1 → VLC 2 位 + 1
            (-1, 3),
            (3, 4),    // size 2 → VLC 2 位 + 2
            (255, 15), // size 8 → VLC 7 位 + 8（size 不 > 8，没有 marker）
            (256, 18), // size 9 → VLC 8 位 + 9 + marker
            (511, 18),
            (512, 20), // size 10 → VLC 9 位 + 10 + marker
        ] {
            let mut w = BitWriter::new();
            write_dc(&mut w, level, true).unwrap();
            assert_eq!(w.bit_pos() as u32, bits, "亮度 level {level}");
        }
        // 色度表短一点：`DC_CHROM[0] = (3,2)`。
        let mut w = BitWriter::new();
        write_dc(&mut w, 0, false).unwrap();
        assert_eq!(w.bit_pos(), 2, "色度 level 0");
    }

    #[test]
    fn write_dc_uses_xbits_negation() {
        // -1 在 size 1 下写成 0；-2 在 size 2 下写成 1；-3 在 size 2 下写成 0；3 照写。
        for (level, size, payload) in [(-1i32, 1u32, 0u32), (-2, 2, 1), (-3, 2, 0), (3, 2, 3)] {
            let mut w = BitWriter::new();
            write_dc(&mut w, level, true).unwrap();
            let nbits = w.bit_pos() as u32;
            let bytes = w.finish();
            let mut all = 0u32;
            for &x in &bytes {
                all = all << 8 | u32::from(x);
            }
            let total = bytes.len() as u32 * 8;
            // 载荷是「已写位」里的最后 size 位。
            let got = (all >> (total - nbits)) & ((1 << size) - 1);
            assert_eq!(got, payload, "level {level} 的载荷");
        }
    }

    #[test]
    fn escape3_costs_the_escape_code_plus_twenty_three_bits() {
        // 帧内表的 escape 码是 `0000011`（7 位，`ff_mpeg4_intra_vlc[102] = (3, 7)`），
        // 所以 ESC3 一共 7 + 2 + 1 + 6 + 1 + 12 + 1 = 30 位。
        // 用一个大 level 逼出 ESC3（12 位能表示 ±2047，但表里没有这么大的 level）。
        let rl = intra_rl_table();
        let mut w = BitWriter::new();
        write_rl(&mut w, &rl, true, 30, 2000).unwrap();
        assert_eq!(w.bit_pos(), 30);
    }

    #[test]
    fn short_levels_avoid_the_escape() {
        // (last=0, run=0, level=1) 在表里，应当只花 ESC0 的位数。
        let rl = intra_rl_table();
        let mut w = BitWriter::new();
        write_rl(&mut w, &rl, false, 0, 1).unwrap();
        assert!(w.bit_pos() <= 4, "最常用的系数不该走 escape");
    }

    #[test]
    fn a_flat_frame_produces_only_dc_and_a_zero_cbp() {
        // 纯色帧：6 个块的 AC 全 0 → cbp = 0 → 每块只有 DC。
        // 参考样本的第一帧正是这样（`cbpc=0`、`cbpy=0`）。
        let mut e = Encoder::new(EncoderConfig {
            width: 16,
            height: 16,
            ..Default::default()
        })
        .unwrap();
        let data = e.encode_i_frame(&flat_frame(16, 16, 128, 128)).unwrap();
        // VOP 起始码必须在
        assert!(data.windows(4).any(|x| x == [0x00, 0x00, 0x01, 0xB6]));
        // 只有 VOP 头 + 1 个宏块，应该很短
        assert!(data.len() < 32, "纯色宏块不该有 {} 字节", data.len());
    }

    #[test]
    fn encoder_rejects_bad_dimensions() {
        let cfg = EncoderConfig {
            width: 15,
            height: 16,
            ..Default::default()
        };
        assert!(Encoder::new(cfg).is_err());
        let cfg = EncoderConfig {
            width: 16,
            height: 0,
            ..Default::default()
        };
        assert!(Encoder::new(cfg).is_err());
    }

    #[test]
    fn decoder_specific_info_carries_the_vos_header() {
        let e = Encoder::new(EncoderConfig {
            width: 320,
            height: 240,
            fps_num: 25,
            ..Default::default()
        })
        .unwrap();
        let dsi = e.decoder_specific_info();
        assert_eq!(dsi, e.elementary_stream_header(), "两处必须是同一份字节");
        // 少了 VOS 那两级，`profile` / `level` 就报不出来。
        assert_eq!(
            &dsi[..4],
            &[0x00, 0x00, 0x01, 0xB0],
            "DSI 必须以 VOS 起始码开头"
        );
        assert_eq!(dsi[4], PROFILE_SIMPLE_LEVEL1);
        assert_eq!(
            &dsi[5..9],
            &[0x00, 0x00, 0x01, 0xB5],
            "接着是视觉对象起始码"
        );
        // ffmpeg 产出的参考 MP4 里，DSI 头 10 字节是：
        //   `00 00 01 b0 01 00 00 01 b5 89 13`
        assert_eq!(
            &dsi[..11],
            &[0x00, 0x00, 0x01, 0xB0, 0x01, 0x00, 0x00, 0x01, 0xB5, 0x89, 0x13]
        );
    }
}
