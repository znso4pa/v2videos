//! 从 FFmpeg 数据文件生成的码表 —— 本仓**唯一**的 LGPL 派生数据。
//!
//! 为什么单独一个 crate：这些表是 LGPL-2.1-or-later（派生自 FFmpeg 的数据文件），
//! 其余代码是照公开规范自研的、可以按 MIT 分发。把派生数据收在一个 crate 里，
//! 「要不要带 LGPL」就变成**一个依赖的有无**，而不是散在六个编解码器里；
//! 将来换数据源（规范附录 / 宽松许可实现）也只改这一个 crate。
//!
//! 每个模块的文件头都写着它的生成器；`tools/gen_notice.py` 与 `NOTICE` 一一对应。
//! **不要手改**：改生成器再重新生成，并用「与旧表逐值相同」当迁移判据。
//!
//! 这些表是 H.263 / MPEG-4 Part 2 / MPEG-4 Audio / WMA / VC-1 / MPEG-1/2 / IntraX8
//! 的**格式定义**（编解码器规格的一部分），不是某个实现的私有数据。

//! 每个模块一个 feature（见 `Cargo.toml`）：编解码器 crate 只开自己那一份，
//! 「只编 WMV3 的 `.so`」就不会把 WMA / AAC 的表链进去。

#[cfg(feature = "aac")]
pub mod aac;
#[cfg(feature = "intrax8")]
pub mod intrax8;
#[cfg(feature = "mpeg12")]
pub mod mpeg12;
#[cfg(feature = "mp4v")]
pub mod mp4v;
#[cfg(feature = "msmpeg4")]
pub mod msmpeg4;
#[cfg(feature = "vc1")]
pub mod vc1;
#[cfg(feature = "wma")]
pub mod wma;
