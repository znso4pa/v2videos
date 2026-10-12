//! 纯 Rust 的 **AAC-LC 音频编码**。
//!
//! 存在的理由很直接：`.wmv → .mp4` 那条路上音频只能塞未压缩 PCM（`sowt`），
//! 而真样本整段里 **PCM 占了 36.7 MB、MPEG-4 视频才 7.3 MB** —— 音频成了体积的大头。
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
//! - ⏳ 短窗（`EIGHT_SHORT_SEQUENCE`，瞬态）
//! - ⏳ TNS / PNS / IS / MS / LTP
//! - ⏳ 心理声学模型（现在用「逐频带穷举 sf 使失真最小」的朴素替代）
//!
//! ## 窄路径的选择
//!
//! 与 WMA 解码、MPEG-4 编码那两条路一个思路：**先啃通一条能端到端跑、
//! 每步都能手算验证的窄路**，再叠复杂度。所以第一阶段只做：
//!
//! - AAC-LC、`ONLY_LONG_SEQUENCE`（不做瞬态检测、不做短窗）
//! - 单声道 / 立体声（不做 IS / MS）
//! - 无 TNS / PNS / LTP / 预测
//! - 全局增益用固定点搜索，逐频带 sf 用穷举
//!
//! 这样出来的码流是**合法**的 AAC（ffmpeg 解得开），只是压缩率不如 ffmpeg 的原生编码器。

mod encode;
pub mod mdct;
pub mod tables;

pub use encode::{AudioSpecificConfig, Encoder, EncoderConfig, Frame, AOT_AAC_LC};
// 错误类型复用 msmpeg4 的（`mp4v` 也是这么做的）—— 它本来就是「编解码器的通用错误」。
pub use v2v_msmpeg4::{Error, Result};
