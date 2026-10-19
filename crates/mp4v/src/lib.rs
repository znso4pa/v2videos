//! 纯 Rust 的 MPEG-4 Part 2（`mp4v`）视频编码 —— **I 帧 + P 帧**（含运动估计与码率控制）。
//!
//! 存在的理由很具体：`.wmv` 里的视频是 MP43，而 **MP4 / 3GP 容器不收 MP43**
//! （`mp4` / `3gp` / `3g2` 三个 muxer 都报 `codec not currently supported in container`；
//! `mov` 收但是单向街 —— 写出来的东西 ffmpeg 自己读不回来）。
//! 所以 `.wmv → .mp4` 这条路上**必须有一个真正的 MPEG-4 编码器**，
//! 而不是「换个容器装同一份码流」。
//!
//! 音频那边不用写编码器：PCM（`sowt`）塞进 `isom` 品牌的文件里 ffmpeg 读得好好的，
//! 见 `crates/mp4` 的模块文档。
//!
//! ## 现状
//!
//! - ✅ VOL / VOP / 视觉对象头的位级写出（`header`）
//! - ✅ I 宏块：正向 DCT → 量化 → DC 预测 → AC 预测（逐宏块试优）→ 熵编码
//! - ✅ P 宏块：运动估计（三起点菱形 + 时间预测 + 半像素细化）、skip、中值 MV 预测、
//!   **逐宏块帧内/帧间判决**（两条路各写一份临时位流比位数）
//! - ✅ 可选**码率控制**：给了目标码率就逐帧调 qscale（`--video-bitrate`）
//! - ✅ 输出能进 `esds` 的 `DecoderSpecificInfo`，播放器不解码就能读出宽高/帧率
//! - ⏳ B 帧、8×8 运动补偿、dquant、resync marker、数据分割
//!
//! ## 有 P 帧意味着什么
//!
//! 要有参考帧 → 要重建（编码端自己解一遍）、要运动估计、要帧内/帧间判决；
//! 码率控制是可选开关，不开就是固定 qscale（此时逐帧输出与「单独编这一帧」一致，
//! 这也是按 I 帧分块并行能保持逐字节不变的前提）。
//!
//! ## 与 MP43 编码器（`v2v_msmpeg4::Encoder`）的关系
//!
//! 两者**语法不同、表不同、预测的 tie-break 都不同**，只有 DCT / 位写入器 /
//! 前缀码容器 / 一张 RL 表能复用（见 `encode` 模块文档的对照表）。
//! 所以这是一个独立 crate，而不是往 `msmpeg4` 里塞第二个编码器。

mod decode;
mod encode;
mod header;
pub mod tables;

pub use decode::{Decoder, Vol};
pub use encode::{Encoder, EncoderConfig};
// 错误类型直接用 `msmpeg4` 的 —— 它本来就是「编解码器的通用错误」，
// 而且 `mp4v` 已经依赖那个 crate 拿 DCT / 位写入器 / RL 表了。
pub use header::{
    time_increment_bits, VolParams, VopParams, PROFILE_SIMPLE_LEVEL1, USER_DATA_STARTCODE,
    VISUAL_OBJ_STARTCODE, VOP_STARTCODE, VOS_STARTCODE,
};
pub use v2v_msmpeg4::{Error, Result};

/// 宏块尺寸（MPEG-4 Part 2 固定 16×16）。
pub const MB_SIZE: u32 = 16;
