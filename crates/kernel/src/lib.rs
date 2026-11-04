//! 编解码器共用的信号处理内核。
//!
//! ## 为什么单独一个 crate
//!
//! 这些东西原来全住在 `v2v-msmpeg4` 里：位读写、VLC、FDCT / IDCT、FFT、SAD、SIMD。
//! 而 aac / wma / mpeg12 / mp4v / vc1 都要用 —— 于是代价有三层：
//!
//! 1. **语义被挂错门牌**：`v2v_msmpeg4::fft_in_place` 这个名字，写 AAC 的人看了
//!    会愣三秒，而且 `mpeg12` 号称「纯安全」却依赖着全仓唯一有 NEON `unsafe` 的 crate；
//! 2. **编译耦合**：改一行 MS-MPEG-4 的帧头解析，AAC / WMA / MPEG-1/2 全跟着重编；
//! 3. **边界说不清**：没人能回答「哪些代码是共用的」，因为答案散在 `pub use` 里。
//!
//! ## 收什么、不收什么
//!
//! 判断标准只有一条：**把任意一个编解码器整个删掉，本 crate 里不该有函数变成死代码。**
//!
//! - 收：位流读写、VLC、RL 表容器、8×8 DCT/IDCT（含 ABT 的 4:8 / 8:4）、FFT、
//!   帧缓冲（`Frame` / `Planes`）、运动补偿内核、SAD、宏块层的预测内核、
//!   NEON、以及编解码器共用的 `Error` / `Result`。
//! - 不收：WMV2 那套 `wmv2_idct_*` 与 mspel 运动补偿（只有 WMV2 用，留在
//!   `v2v-msmpeg4`）、帧头解析、宏块语法、运动矢量的解码/编码、环路滤波，
//!   以及各家自己的码表（`crates/*/src/tables.rs`）。
//!
//! ## unsafe
//!
//! crate 根上 `#![deny(unsafe_code)]`，只有 [`simd`] 显式放开。「别处不许写 unsafe」
//! 由编译器保证，不靠人记。
//!
//! 默认**不**开 SIMD 的 `unsafe` 路径这件事还没做（见 TODO 的 C 步）：现在只要
//! 依赖本 crate，就依赖了那 12 处 `unsafe`。
#![deny(unsafe_code)]

pub mod bitreader;
pub mod bitwriter;
pub mod error;
pub mod fdct;
pub mod fft;
pub mod idct;
pub mod mc;
pub mod predict;
pub mod rl;
pub mod simd;
pub mod vlc;

pub use bitreader::BitReader;
pub use bitwriter::BitWriter;
pub use error::{Error, Result};
pub use fdct::{fdct, fdct_i, fdct_i_float, fdct_llm};
pub use fft::fft_in_place;
pub use idct::{idct48_add, idct84_add, idct_add, idct_put};
pub use mc::{
    edge_emu_into, mc_chroma8, mc_luma16, mc_mb, mc_params, mid_pred, put_pixels, sad_luma16,
    sad_luma16_capped, Frame, McParams, Planes, ZIGZAG,
};
pub use predict::{ac_predict, coded_block_predict, dc_unscale, AcVal};
pub use rl::RlTable;
pub use vlc::Vlc;
