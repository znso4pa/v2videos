//! VC-1（WMV3）解码需要的 VLC / 量化表 / 枚举常量。
//!
//! **本文件由 `tools/gen_vc1_tables.py` 生成，不要手改。**
//!
//! ## 表本身是什么
//!
//! VC-1（SMPTE 421M）规范里的 VLC、位平面编码与量化定义 —— 编解码器格式的一部分，
//! 不是某个实现的私有数据。生成器只做「读出来 + 排版」，不做任何推导。
//!
//! ## 为什么枚举也在里面
//!
//! 表里有几处是**符号名**而不是数字（`ff_vc1_ttfrm_to_tt = { TT_8X8, TT_8X4, … }`、
//! `ff_vc1_mv_pmode_table = { MV_PMODE_* }`）。生成器把枚举从源文件读出来翻译，
//! 于是「枚举顺序」也只有一份真源 —— 手抄数值等于把顺序抄了一遍。
//!
//! ## 许可
//!
//! 本文件的数据取自 FFmpeg 7.1（LGPL-2.1-or-later）的这几个文件，
//! 以及它们的版权声明（由生成器从源文件原样读出）：
//!
//! ```text
//! `vc1data.c`
//!     copyright (c) 2011 Mashiat Sarker Shakkhar
//!     copyright (c) 2006 Konstantin Shishkov
//! `vc1_vlc_data.h`
//!     copyright (c) 2011 Mashiat Sarker Shakkhar
//!     copyright (c) 2006 Konstantin Shishkov
//! `vc1.h`
//!     Copyright (c) 2006-2007 Konstantin Shishkov
//! `vc1_common.h`
//!     Copyright (c) 2006-2007 Konstantin Shishkov
//! ```
//!
//! 与本仓的许可一致，整个仓库统一按 LGPL-2.1-or-later 使用。
//! 详见 README 的「许可证」一节与根目录的 `NOTICE`。
//!
//! 换一个数据源（规范附录、或别的宽松许可实现）即可整体替换 —— 生成器已就位，
//! 且表本身与实现无关。

#![allow(dead_code)]

// ---- `enum`（源：`libavcodec/vc1_common.h`）

pub const PROFILE_SIMPLE   : u8 = 0;
pub const PROFILE_MAIN     : u8 = 1;
pub const PROFILE_COMPLEX  : u8 = 2;
pub const PROFILE_ADVANCED : u8 = 3;

// ---- `enum`（源：`libavcodec/vc1.h`）

pub const QUANT_FRAME_IMPLICIT : u8 = 0;
pub const QUANT_FRAME_EXPLICIT : u8 = 1;
pub const QUANT_NON_UNIFORM    : u8 = 2;
pub const QUANT_UNIFORM        : u8 = 3;

// ---- `enum`（源：`libavcodec/vc1.h`）

pub const DQPROFILE_FOUR_EDGES   : u8 = 0;
pub const DQPROFILE_DOUBLE_EDGES : u8 = 1;
pub const DQPROFILE_SINGLE_EDGE  : u8 = 2;
pub const DQPROFILE_ALL_MBS      : u8 = 3;

// ---- `enum`（源：`libavcodec/vc1.h`）

pub const MV_PMODE_1MV_HPEL_BILIN : u8 = 0;
pub const MV_PMODE_1MV            : u8 = 1;
pub const MV_PMODE_1MV_HPEL       : u8 = 2;
pub const MV_PMODE_MIXED_MV       : u8 = 3;
pub const MV_PMODE_INTENSITY_COMP : u8 = 4;

// ---- `enum`（源：`libavcodec/vc1.h`）

pub const TT_8X8        : u8 = 0;
pub const TT_8X4_BOTTOM : u8 = 1;
pub const TT_8X4_TOP    : u8 = 2;
pub const TT_8X4        : u8 = 3;
pub const TT_4X8_RIGHT  : u8 = 4;
pub const TT_4X8_LEFT   : u8 = 5;
pub const TT_4X8        : u8 = 6;
pub const TT_4X4        : u8 = 7;

// ---- `enum`（源：`libavcodec/vc1.h`）

pub const IMODE_RAW     : u8 = 0;
pub const IMODE_NORM2   : u8 = 1;
pub const IMODE_DIFF2   : u8 = 2;
pub const IMODE_NORM6   : u8 = 3;
pub const IMODE_DIFF6   : u8 = 4;
pub const IMODE_ROWSKIP : u8 = 5;
pub const IMODE_COLSKIP : u8 = 6;

// ---- 位平面（bitplane）的 VLC。`[code, len]`，下标即符号。

pub static IMODE_VLC: [[u32; 2]; 7] = [
    [0, 4],
    [2, 2],
    [1, 3],
    [3, 2],
    [1, 4],
    [2, 3],
    [3, 3],
];

pub static NORM2_VLC: [[u32; 2]; 4] = [
    [0, 1],
    [4, 3],
    [5, 3],
    [3, 2],
];

pub static NORM6_VLC: [[u32; 2]; 64] = [
    [1, 1],
    [2, 4],
    [3, 4],
    [0, 8],
    [4, 4],
    [1, 8],
    [2, 8],
    [71, 10],
    [5, 4],
    [3, 8],
    [4, 8],
    [75, 10],
    [5, 8],
    [77, 10],
    [78, 10],
    [782, 13],
    [6, 4],
    [6, 8],
    [7, 8],
    [83, 10],
    [8, 8],
    [85, 10],
    [86, 10],
    [781, 13],
    [9, 8],
    [89, 10],
    [90, 10],
    [780, 13],
    [92, 10],
    [779, 13],
    [778, 13],
    [55, 9],
    [7, 4],
    [10, 8],
    [11, 8],
    [67, 10],
    [12, 8],
    [69, 10],
    [70, 10],
    [777, 13],
    [13, 8],
    [73, 10],
    [74, 10],
    [776, 13],
    [76, 10],
    [775, 13],
    [774, 13],
    [54, 9],
    [14, 8],
    [81, 10],
    [82, 10],
    [773, 13],
    [84, 10],
    [772, 13],
    [771, 13],
    [53, 9],
    [88, 10],
    [770, 13],
    [769, 13],
    [52, 9],
    [768, 13],
    [51, 9],
    [50, 9],
    [7, 6],
];

// ---- 量化：`pqindex` -> `pq`。三行分别是隐式 / 显式均匀 / 显式非均匀。

pub static PQUANT: [[u8; 32]; 3] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 31],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    [0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 31],
];

// ---- P 帧的帧级 MV 模式：`[lowquant][unary 值]` -> `MV_PMODE_*`。

pub static MV_PMODE_TABLE: [[u8; 5]; 2] = [
    [0, 1, 2, 4, 3],
    [1, 3, 2, 4, 0],
];

pub static MV_PMODE_TABLE2: [[u8; 4]; 2] = [
    [0, 1, 2, 3],
    [1, 3, 2, 0],
];

// ---- 帧头 `ttfrm`（2 bit）-> `TT_*`。

pub static TTFRM_TO_TT: [u8; 4] = [
    0, 3, 6, 7,
];

