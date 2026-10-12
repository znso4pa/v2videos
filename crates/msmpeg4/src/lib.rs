//! 纯 Rust 的 Microsoft MPEG-4（MP41/MP42/MP43、WMV1/WMV2）视频解码。
//!
//! 目标：把 ASF 解封装吐出的裸视频帧解成 YUV420P。无 C、无 FFI、零依赖。
//!
//! ## 现状
//!
//! - ✅ 位读取器（MSB 优先）
//! - ✅ 帧头解析（`parse_frame_header`）—— MP43 无起始码，逐帧从位流开头读
//! - ✅ 反量化后的 8×8 IDCT（H.263 / MPEG-4 `simple_idct`），含叠加版 `idct_add`
//! - ✅ 系数（AC）解码：RL 前缀码 + 三种 escape + 反量化
//! - ✅ 预测内核：DC 预测 / AC 预测 / 编码块图预测
//! - ✅ I 帧解码：MB 类型 + DC/AC + IDCT → YUV420P
//! - ✅ P 帧解码：skip / 帧间 CBP / MV（差分 + H.263 中值预测）/ 半像素 MC / 残差叠加
//! - ✅ **I 帧编码**（`Encoder`）：正向 DCT + 量化 + DC/RL 熵编码 —— 编出来的流
//!   **ffmpeg 解得开**，且两个解码器逐像素一致
//! - ⏳ P 帧编码（运动估计）/ ASF 写
//!
//! 实现按 H.263 / MPEG-4 Part 2 的语法写，**不链 ffmpeg、不引依赖** ——
//! ffmpeg 只当**交叉验证的 oracle**（测试里解同一条流、逐像素对账），不进产物。

mod bitreader;
mod block;
mod error;
mod frame;
mod grid;
mod header;
mod idct;
mod predict;
#[cfg(test)]
mod testbit;
pub mod tables;
mod vlc;

pub use bitreader::BitReader;
pub use error::{Error, Result};
pub use header::{decode012, parse_frame_header, FrameHeader, PictureType};
pub use block::RlTable;
pub use frame::{decode_i_frame, AcPredRead, Decoder, Frame, Stats, ZIGZAG};
pub use idct::{idct_add, idct_put};
pub use predict::{ac_predict, coded_block_predict, dc_predict, AcVal};
pub use vlc::Vlc;

/// 宏块尺寸（MPEG-4 / H.263 系固定 16×16）。
pub const MB_SIZE: u32 = 16;

/// 由像素尺寸算宏块数。
pub fn mb_count(width: u32, height: u32) -> (u32, u32) {
    (
        width.div_ceil(MB_SIZE),
        height.div_ceil(MB_SIZE),
    )
}
