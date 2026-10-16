//! 纯 Rust 的 Microsoft MPEG-4 视频解码 —— **目前只做 MP43（v3）**。
//!
//! ⚠️ 文档以前写成「MP41/MP42/MP43、WMV1/WMV2 都支持」，那是**夸大**：
//! WMV1/WMV2（v1/v2）的帧头与宏块语法不同，还没实现（实测一个 WMV1 样本会解不动，
//! 对账测试里显式 SKIP 掉了）。见 TODO。
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
//! - ✅ I 帧编码（`Encoder`）：正向 DCT + 量化 + DC/RL 熵编码 —— 编出来的流
//!   **ffmpeg 解得开**，且两个解码器逐像素一致
//! - ✅ P 帧编码：运动估计（整像素粗搜 + 半像素细化）+ 帧间残差 + skip；
//!   **帧内/帧间判决** —— 画面切换时帧内宏块自动接管
//!
//! ASF 封装**不在这个 crate 里**，见 `v2v-asf`（本 crate 只管裸 MP43 帧）。
//! 编出来的也只是 **MP43（v3）**，不是 WMV9/VC-1 —— 原因见 README 的能力矩阵。
//!
//! 实现按 H.263 / MPEG-4 Part 2 的语法写，**不链 ffmpeg、不引依赖** ——
//! ffmpeg 只当**交叉验证的 oracle**（测试里解同一条流、逐像素对账），不进产物。

mod bitreader;
pub mod bitwriter;
mod block;
mod encode;
mod error;
mod fdct;
mod fft;
mod frame;
mod grid;
mod header;
mod idct;
mod predict;
pub mod tables;
#[cfg(test)]
mod testbit;
mod version;
mod vlc;

pub use bitreader::BitReader;
pub use bitwriter::BitWriter;
pub use block::{AcCtx, RlTable};
pub use encode::{Encoder, EncoderConfig};
pub use error::{Error, Result};
pub use fdct::{fdct, fdct_i};
pub use fft::fft_in_place;
pub use frame::{
    decode_i_frame, mc_chroma8, mc_luma16, mc_mb, mc_mb_mspel, mc_params, mid_pred, sad_luma16,
    AcPredRead, Decoder, Frame, McParams, Planes, Stats, ZIGZAG,
};
pub use header::{
    decode012, parse_frame_header, parse_wmv2_frame, FrameHeader, HeaderState, PictureType,
    Wmv2Flags, Wmv2Frame, Wmv2Header, II_BITRATE, MBAC_BITRATE,
};
pub use version::Version;
pub use idct::{idct_add, idct_put};
pub use predict::{ac_predict, coded_block_predict, dc_predict, AcVal};
pub use vlc::Vlc;

/// 宏块尺寸（MPEG-4 / H.263 系固定 16×16）。
pub const MB_SIZE: u32 = 16;

/// 由像素尺寸算宏块数。
pub fn mb_count(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(MB_SIZE), height.div_ceil(MB_SIZE))
}
