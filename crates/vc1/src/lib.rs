//! VC-1（WMV3）解码。
//!
//! **范围**：Simple / Main profile、**渐进**（非隔行）、**I / P 帧**。
//! Advanced profile（序列头在码流里、带起始码）、B 帧、隔行、sprite 帧都不在范围内 ——
//! 遇到就报 `Unsupported` 或 `Bad`，不静默出错。
//!
//! **走到哪一步了**：**整条链路通了** —— 序列头（容器 extradata 里的 `STRUCT_C`）→
//! 帧头（含位平面）→ 宏块层（I / P）→ 逆变换（8×8 + ABT 四种）→ 运动补偿
//! （四分之一 / 半像素）→ 环路滤波 → YUV420P。
//!
//! 还没做的：`res_fasttx=0`（走 MS-MPEG-4 simple_idct 的那一档）、overlap smoothing、
//! 多分辨率、`rangered`、重同步标记、B 帧、隔行、Advanced profile ——
//! `Decoder::new` 会直接拒掉其中影响重建的那几个，不静默出错。
//!
//! 与上游的对照：`vc1.c`（序列头 / 帧头 / 位平面）、`vc1_block.c`（宏块层）、
//! `vc1_mc.c`（运动补偿）、`vc1_pred.c`（MV 预测）、`vc1_loopfilter.c`（环路滤波）、
//! `vc1dsp.c`（变换与插值核）、`vc1dec.c`（帧级组装）。

mod bitplane;
mod decoder;
mod header;
pub mod idct;
mod loopfilter;
mod mb;
mod mc;
mod mv;
pub mod tables;

pub use bitplane::{Bitplane, Imode};
pub use decoder::{Decoder, Frame};
pub use header::{
    Dquant, FrameHeader, HeaderState, MvMode, PFrameHeader, PictType, Profile, QuantizerMode,
    SeqHeader,
};

// 位读取器 / 前缀码 / 错误类型都复用 msmpeg4 那一套 —— 与 mp4v 一样，
// 不另立一份（复刻出来的第二份才是真会跑偏的那份）。
pub use v2v_msmpeg4::{Error, Result};
