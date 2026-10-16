//! 纯 Rust 的 **AAC-LC 音频编码**。
//!
//! 存在的理由很直接：`.wmv → .mp4` 那条路上音频只能塞未压缩 PCM（`sowt`），
//! 而真样本整段里**音频载荷比视频载荷还大** —— 音频成了体积的大头。
//! 补上 AAC 编码器才有可能把 `.mp4` 压到合理大小。
//!
//! ## 与 WMA 解码那边的对称性
//!
//! | WMA 解码（已有） | AAC 编码（本 crate） |
//! |---|---|
//! | IMDCT + 重叠相加 | MDCT + 50% 重叠 |
//! | 反量化（指数 → 系数） | 量化（`pow(·, 3/4)` + sf） |
//! | 6 张 RL 表的 Huffman 解码 | 11 本 Huffman 码本的频谱编码 |
//! | 位读取器 | `BitWriter`（复用 `v2v_msmpeg4` 的，MSB 优先语义一致） |
//!
//! ## 现状
//!
//! - ✅ MDCT / IMDCT / 正弦窗 / KBD 窗（`mdct`）
//! - ✅ 码表（`tables`，由 `tools/gen_aac_tables.py` 生成）
//! - ✅ 量化（`pow(·, 3/4)` + 全局增益 + 逐频带 sf）
//! - ✅ Huffman 编码（码本选择 + escape）
//! - ✅ ICS / SCE / CPE 语法 + AudioSpecificConfig
//! - ✅ 短窗（`EIGHT_SHORT_SEQUENCE`，瞬态）+ 窗口分组（3GPP TS26.403 5.4.1）
//!   + `LONG_START` / `LONG_STOP` 过渡窗（状态机见 `plan_seq`、开窗表见 `mdct`）
//! - ✅ 逐带率-失真分配（每带按 `失真 + λ·位数` 选 sf / 码本，λ 从上一帧的答案出发搜）
//! - ⏳ PNS / IS / LTP
//! - ⏳ 心理声学模型（`masking_thresholds` 算好了但**没接线** —— 两套接法
//!   （线性加权 / 阈值饱和）实测都是净亏、已撤，见 `encode.rs` 与 `TODO.md`）
//!
//! ## 窄路径的选择
//!
//! 与 WMA 解码、MPEG-4 编码那两条路一个思路：**先啃通一条能端到端跑、
//! 每步都能手算验证的窄路**，再叠复杂度。
//!
//! 第一阶段只做 `ONLY_LONG_SEQUENCE` + 单声道 / 立体声 + 无 TNS / PNS / MS；
//! 之后逐项补上，**现在已经有了**：
//!
//! - ✅ 长 / 短窗 + **过渡窗**（`LONG_START` / `LONG_STOP`）—— 瞬态检测驱动窗序列状态机
//! - ✅ 短窗**分组**（攻击所在的那个窗口单独一组）
//! - ✅ **M/S 立体声**（逐帧决策，`common_window = 1`）
//! - ✅ **TNS**（长窗）
//! - ✅ 全局增益用固定点搜索，逐频带 sf 用穷举 + 三级细化
//!
//! 还没做：
//!
//! - ❌ **PNS**（感知噪声替换）—— 对噪声类信号省码率，但对音乐增益有限
//! - ❌ **IS**（强度立体声）—— 高码率下不如 M/S，低码率才有意义
//! - ❌ **LTP / 预测**（那是 AAC-LC 之外的对象类型）
//! - ❌ 非标准采样率（`AudioSpecificConfig` 只认规范里那 13 个）
//!
//! 这样出来的码流是**合法**的 AAC（ffmpeg 解得开）。

mod encode;
pub mod mdct;
pub mod tables;
mod tns;

pub use encode::{AudioSpecificConfig, Encoder, EncoderConfig, Frame, AOT_AAC_LC};
// 错误类型复用 msmpeg4 的（`mp4v` 也是这么做的）—— 它本来就是「编解码器的通用错误」。
pub use v2v_msmpeg4::{Error, Result};
