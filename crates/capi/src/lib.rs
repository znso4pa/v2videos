//! 全仓的 C ABI —— 一层一组符号。
//!
//! | 层 | 符号 | 覆盖 |
//! |---|---|---|
//! | 公共 | `v2v_abi_version` / `v2v_last_error` / `v2v_last_error_kind` | 版本、错误通道 |
//! | 容器 | `v2v_probe` / `v2v_demux_*` | ASF、AVI |
//! | 解码 | `v2v_vdec_*` / `v2v_adec_*` | VC-1、MS-MPEG-4 家族、MPEG-4 Part 2、WMA |
//! | 编码 | `v2v_venc_*` / `v2v_aenc_*` | MP43、MPEG-4 Part 2、AAC |
//! | 封装 | `v2v_mux_*` | ASF、MP4 |
//! | 播放 | `v2v_player_*` | 顺时播放（解封装 + 解码） |
//! | 转码 | `v2v_transcode` / `v2v_hw_supported` | 硬件编码（`hw` feature） |
//!
//! ## 为什么是**一个** crate 而不是每个 crate 一个 `.so`
//!
//! 一个版本号、一个错误通道、一张派发表（`codec_id → 构造器` 只在这里 `match`）。
//! 代价是全量 `.so` 体积大 —— 那是用 feature 裁剪解决的（见 `Cargo.toml`），
//! 不是拆成多个 `.so` 解决的：拆开之后「哪个 `.so` 是哪个 ABI 版本」会立刻变成
//! 调用方要自己维护的事。
//!
//! ## 全局约定（每个符号都遵守，头文件里也写了一遍）
//!
//! 1. **panic 不穿透**：每个入口都在 [`common::guard`] 里，兜底返回错误码。
//! 2. **谁分配谁释放**：库交给调用方的指针，要么「活到下次同类调用」，
//!    要么配一个 `v2v_*_free`。调用方**永远不 `free`** 库给的内存。
//! 3. **出参结构体首字段是 `size`**：调用方填 `sizeof`，库按 `min` 写。
//! 4. **句柄只能在一个线程用**：要并发就开多个句柄（各句柄互不影响）。
//!
//! ## 一条不做的事
//!
//! 这里**不做**颜色转换 / 缩放 / 重采样。像素一律 YUV420P、音频一律每声道一条
//! `f32`，`stride` 语义写死在头文件里 —— 转换是调用方渲染层的事，
//! 在 ABI 里做一层只会多一次全帧拷贝，还多一份要维护的语义。

pub mod common;
pub mod demux;
pub mod player;

pub use common::{
    arg_err, cstr, guard, guard_ptr, rc, set_err, slice, write_sized, IntoAbi, ABI_MAJOR, ABI_MINOR,
    EOS, ERR_ARG, ERR_FAILED, ERR_UNSUPPORTED, KIND_BAD, KIND_IO, KIND_UNSUPPORTED, OK,
};
