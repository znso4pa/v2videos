//! VC-1（WMV3）解码。
//!
//! **范围**：Simple / Main profile、**渐进**（非隔行）、**I / P 帧**。
//! Advanced profile（序列头在码流里、带起始码）、B 帧、隔行、sprite 帧都不在范围内 ——
//! 遇到就报 `Unsupported` 或 `Bad`，不静默出错。
//!
//! **走到哪一步了**：目前只到**头** —— 序列头（容器 extradata 里的 `STRUCT_C`）、
//! 帧头、以及帧头里的**位平面**（逐宏块跳过标志那一类整帧位图）。
//! 宏块层、逆变换（8×8 + ABT）、运动补偿、环路滤波是后面的切片，还没做。
//! 所以这一层现在的作用是「把每帧的头部字段解对、把游标停在宏块层的起点」，
//! 而它自己已经被真样本 + `ffmpeg` 的 `-debug pict` 日志钉住了。
//!
//! 与上游的对照：`vc1.c` 的 `ff_vc1_decode_sequence_header()` / `bitplane_decoding()`
//! / `ff_vc1_parse_frame_header()`。

mod bitplane;
mod header;
pub mod idct;
pub mod tables;

pub use bitplane::{Bitplane, Imode};
pub use header::{
    Dquant, FrameHeader, HeaderState, MvMode, PFrameHeader, PictType, Profile, QuantizerMode,
    SeqHeader,
};

// 位读取器 / 前缀码 / 错误类型都复用 msmpeg4 那一套 —— 与 mp4v 一样，
// 不另立一份（复刻出来的第二份才是真会跑偏的那份）。
pub use v2v_msmpeg4::{Error, Result};
