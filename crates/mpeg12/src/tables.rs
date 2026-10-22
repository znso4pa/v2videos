//! MPEG-1 的码表与量化矩阵。**本文件由 `tools/gen_mpeg12_tables.py` 生成，不要手改。**
//!
//! 数据从一份已知正确的实现里机械抽取，抽取后由本 crate 的 `table_tests`
//! 做前缀无关 / Kraft 自检 —— 手抄这种表最容易出「整体对、个别错」，
//! 而那种错只表现为画面局部花掉，反推代价极高。

/// 默认帧内量化矩阵（8×8，行优先）。
pub const INTRA_MATRIX: [u8; 64] = [
    8, 16, 19, 22, 26, 27, 29, 34,
    16, 16, 22, 24, 27, 29, 34, 37,
    19, 22, 26, 27, 29, 34, 34, 38,
    22, 22, 26, 27, 29, 34, 37, 40,
    22, 26, 27, 29, 32, 35, 40, 48,
    26, 27, 29, 32, 35, 40, 48, 58,
    26, 27, 29, 34, 38, 46, 56, 69,
    27, 29, 35, 38, 46, 56, 69, 83,
];

/// 默认帧间量化矩阵（8×8，行优先）。P 帧残差用它反量化。
pub const NON_INTRA_MATRIX: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
];

/// 一条前缀码：`bits` 位码 `code`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Code {
    pub code: u16,
    pub bits: u8,
}

/// 亮度块 DC 尺寸码，下标即 `dct_dc_size`。
pub const DC_LUM: [Code; 12] = [
    Code { code: 0x0004, bits: 3 },
    Code { code: 0x0000, bits: 2 },
    Code { code: 0x0001, bits: 2 },
    Code { code: 0x0005, bits: 3 },
    Code { code: 0x0006, bits: 3 },
    Code { code: 0x000e, bits: 4 },
    Code { code: 0x001e, bits: 5 },
    Code { code: 0x003e, bits: 6 },
    Code { code: 0x007e, bits: 7 },
    Code { code: 0x00fe, bits: 8 },
    Code { code: 0x01fe, bits: 9 },
    Code { code: 0x01ff, bits: 9 },
];

/// 色度块 DC 尺寸码，下标即 `dct_dc_size`。
pub const DC_CHROMA: [Code; 12] = [
    Code { code: 0x0000, bits: 2 },
    Code { code: 0x0001, bits: 2 },
    Code { code: 0x0002, bits: 2 },
    Code { code: 0x0006, bits: 3 },
    Code { code: 0x000e, bits: 4 },
    Code { code: 0x001e, bits: 5 },
    Code { code: 0x003e, bits: 6 },
    Code { code: 0x007e, bits: 7 },
    Code { code: 0x00fe, bits: 8 },
    Code { code: 0x01fe, bits: 9 },
    Code { code: 0x03fe, bits: 10 },
    Code { code: 0x03ff, bits: 10 },
];

/// 宏块地址增量码，下标 + 1 即增量值。
pub const MB_ADDR_INCR: [Code; 36] = [
    Code { code: 0x0001, bits: 1 },
    Code { code: 0x0003, bits: 3 },
    Code { code: 0x0002, bits: 3 },
    Code { code: 0x0003, bits: 4 },
    Code { code: 0x0002, bits: 4 },
    Code { code: 0x0003, bits: 5 },
    Code { code: 0x0002, bits: 5 },
    Code { code: 0x0007, bits: 7 },
    Code { code: 0x0006, bits: 7 },
    Code { code: 0x000b, bits: 8 },
    Code { code: 0x000a, bits: 8 },
    Code { code: 0x0009, bits: 8 },
    Code { code: 0x0008, bits: 8 },
    Code { code: 0x0007, bits: 8 },
    Code { code: 0x0006, bits: 8 },
    Code { code: 0x0017, bits: 10 },
    Code { code: 0x0016, bits: 10 },
    Code { code: 0x0015, bits: 10 },
    Code { code: 0x0014, bits: 10 },
    Code { code: 0x0013, bits: 10 },
    Code { code: 0x0012, bits: 10 },
    Code { code: 0x0023, bits: 11 },
    Code { code: 0x0022, bits: 11 },
    Code { code: 0x0021, bits: 11 },
    Code { code: 0x0020, bits: 11 },
    Code { code: 0x001f, bits: 11 },
    Code { code: 0x001e, bits: 11 },
    Code { code: 0x001d, bits: 11 },
    Code { code: 0x001c, bits: 11 },
    Code { code: 0x001b, bits: 11 },
    Code { code: 0x001a, bits: 11 },
    Code { code: 0x0019, bits: 11 },
    Code { code: 0x0018, bits: 11 },
    Code { code: 0x0008, bits: 11 },
    Code { code: 0x000f, bits: 11 },
    Code { code: 0x0000, bits: 8 },
];

/// `coded_block_pattern` 码，下标即 CBP。
pub const MB_PATTERN: [Code; 64] = [
    Code { code: 0x0001, bits: 9 },
    Code { code: 0x000b, bits: 5 },
    Code { code: 0x0009, bits: 5 },
    Code { code: 0x000d, bits: 6 },
    Code { code: 0x000d, bits: 4 },
    Code { code: 0x0017, bits: 7 },
    Code { code: 0x0013, bits: 7 },
    Code { code: 0x001f, bits: 8 },
    Code { code: 0x000c, bits: 4 },
    Code { code: 0x0016, bits: 7 },
    Code { code: 0x0012, bits: 7 },
    Code { code: 0x001e, bits: 8 },
    Code { code: 0x0013, bits: 5 },
    Code { code: 0x001b, bits: 8 },
    Code { code: 0x0017, bits: 8 },
    Code { code: 0x0013, bits: 8 },
    Code { code: 0x000b, bits: 4 },
    Code { code: 0x0015, bits: 7 },
    Code { code: 0x0011, bits: 7 },
    Code { code: 0x001d, bits: 8 },
    Code { code: 0x0011, bits: 5 },
    Code { code: 0x0019, bits: 8 },
    Code { code: 0x0015, bits: 8 },
    Code { code: 0x0011, bits: 8 },
    Code { code: 0x000f, bits: 6 },
    Code { code: 0x000f, bits: 8 },
    Code { code: 0x000d, bits: 8 },
    Code { code: 0x0003, bits: 9 },
    Code { code: 0x000f, bits: 5 },
    Code { code: 0x000b, bits: 8 },
    Code { code: 0x0007, bits: 8 },
    Code { code: 0x0007, bits: 9 },
    Code { code: 0x000a, bits: 4 },
    Code { code: 0x0014, bits: 7 },
    Code { code: 0x0010, bits: 7 },
    Code { code: 0x001c, bits: 8 },
    Code { code: 0x000e, bits: 6 },
    Code { code: 0x000e, bits: 8 },
    Code { code: 0x000c, bits: 8 },
    Code { code: 0x0002, bits: 9 },
    Code { code: 0x0010, bits: 5 },
    Code { code: 0x0018, bits: 8 },
    Code { code: 0x0014, bits: 8 },
    Code { code: 0x0010, bits: 8 },
    Code { code: 0x000e, bits: 5 },
    Code { code: 0x000a, bits: 8 },
    Code { code: 0x0006, bits: 8 },
    Code { code: 0x0006, bits: 9 },
    Code { code: 0x0012, bits: 5 },
    Code { code: 0x001a, bits: 8 },
    Code { code: 0x0016, bits: 8 },
    Code { code: 0x0012, bits: 8 },
    Code { code: 0x000d, bits: 5 },
    Code { code: 0x0009, bits: 8 },
    Code { code: 0x0005, bits: 8 },
    Code { code: 0x0005, bits: 9 },
    Code { code: 0x000c, bits: 5 },
    Code { code: 0x0008, bits: 8 },
    Code { code: 0x0004, bits: 8 },
    Code { code: 0x0004, bits: 9 },
    Code { code: 0x0007, bits: 3 },
    Code { code: 0x000a, bits: 5 },
    Code { code: 0x0008, bits: 5 },
    Code { code: 0x000c, bits: 6 },
];

/// 运动矢量码，下标即 `|motion_code|`（0..=16）。
pub const MV_CODE: [Code; 17] = [
    Code { code: 0x0001, bits: 1 },
    Code { code: 0x0001, bits: 2 },
    Code { code: 0x0001, bits: 3 },
    Code { code: 0x0001, bits: 4 },
    Code { code: 0x0003, bits: 6 },
    Code { code: 0x0005, bits: 7 },
    Code { code: 0x0004, bits: 7 },
    Code { code: 0x0003, bits: 7 },
    Code { code: 0x000b, bits: 9 },
    Code { code: 0x000a, bits: 9 },
    Code { code: 0x0009, bits: 9 },
    Code { code: 0x0011, bits: 10 },
    Code { code: 0x0010, bits: 10 },
    Code { code: 0x000f, bits: 10 },
    Code { code: 0x000e, bits: 10 },
    Code { code: 0x000d, bits: 10 },
    Code { code: 0x000c, bits: 10 },
];

/// 一条 run/level 码字。level 的符号是码字之后的**独立 1 位**。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RlEntry {
    pub code: u16,
    pub bits: u8,
    pub run: u8,
    pub level: i16,
}

/// `(run, level)` 到码字的表。
pub const MPEG1_RL: [RlEntry; 111] = [
    RlEntry { code: 0x0003, bits: 2, run: 0, level: 1 },
    RlEntry { code: 0x0004, bits: 4, run: 0, level: 2 },
    RlEntry { code: 0x0005, bits: 5, run: 0, level: 3 },
    RlEntry { code: 0x0006, bits: 7, run: 0, level: 4 },
    RlEntry { code: 0x0026, bits: 8, run: 0, level: 5 },
    RlEntry { code: 0x0021, bits: 8, run: 0, level: 6 },
    RlEntry { code: 0x000a, bits: 10, run: 0, level: 7 },
    RlEntry { code: 0x001d, bits: 12, run: 0, level: 8 },
    RlEntry { code: 0x0018, bits: 12, run: 0, level: 9 },
    RlEntry { code: 0x0013, bits: 12, run: 0, level: 10 },
    RlEntry { code: 0x0010, bits: 12, run: 0, level: 11 },
    RlEntry { code: 0x001a, bits: 13, run: 0, level: 12 },
    RlEntry { code: 0x0019, bits: 13, run: 0, level: 13 },
    RlEntry { code: 0x0018, bits: 13, run: 0, level: 14 },
    RlEntry { code: 0x0017, bits: 13, run: 0, level: 15 },
    RlEntry { code: 0x001f, bits: 14, run: 0, level: 16 },
    RlEntry { code: 0x001e, bits: 14, run: 0, level: 17 },
    RlEntry { code: 0x001d, bits: 14, run: 0, level: 18 },
    RlEntry { code: 0x001c, bits: 14, run: 0, level: 19 },
    RlEntry { code: 0x001b, bits: 14, run: 0, level: 20 },
    RlEntry { code: 0x001a, bits: 14, run: 0, level: 21 },
    RlEntry { code: 0x0019, bits: 14, run: 0, level: 22 },
    RlEntry { code: 0x0018, bits: 14, run: 0, level: 23 },
    RlEntry { code: 0x0017, bits: 14, run: 0, level: 24 },
    RlEntry { code: 0x0016, bits: 14, run: 0, level: 25 },
    RlEntry { code: 0x0015, bits: 14, run: 0, level: 26 },
    RlEntry { code: 0x0014, bits: 14, run: 0, level: 27 },
    RlEntry { code: 0x0013, bits: 14, run: 0, level: 28 },
    RlEntry { code: 0x0012, bits: 14, run: 0, level: 29 },
    RlEntry { code: 0x0011, bits: 14, run: 0, level: 30 },
    RlEntry { code: 0x0010, bits: 14, run: 0, level: 31 },
    RlEntry { code: 0x0018, bits: 15, run: 0, level: 32 },
    RlEntry { code: 0x0017, bits: 15, run: 0, level: 33 },
    RlEntry { code: 0x0016, bits: 15, run: 0, level: 34 },
    RlEntry { code: 0x0015, bits: 15, run: 0, level: 35 },
    RlEntry { code: 0x0014, bits: 15, run: 0, level: 36 },
    RlEntry { code: 0x0013, bits: 15, run: 0, level: 37 },
    RlEntry { code: 0x0012, bits: 15, run: 0, level: 38 },
    RlEntry { code: 0x0011, bits: 15, run: 0, level: 39 },
    RlEntry { code: 0x0010, bits: 15, run: 0, level: 40 },
    RlEntry { code: 0x0003, bits: 3, run: 1, level: 1 },
    RlEntry { code: 0x0006, bits: 6, run: 1, level: 2 },
    RlEntry { code: 0x0025, bits: 8, run: 1, level: 3 },
    RlEntry { code: 0x000c, bits: 10, run: 1, level: 4 },
    RlEntry { code: 0x001b, bits: 12, run: 1, level: 5 },
    RlEntry { code: 0x0016, bits: 13, run: 1, level: 6 },
    RlEntry { code: 0x0015, bits: 13, run: 1, level: 7 },
    RlEntry { code: 0x001f, bits: 15, run: 1, level: 8 },
    RlEntry { code: 0x001e, bits: 15, run: 1, level: 9 },
    RlEntry { code: 0x001d, bits: 15, run: 1, level: 10 },
    RlEntry { code: 0x001c, bits: 15, run: 1, level: 11 },
    RlEntry { code: 0x001b, bits: 15, run: 1, level: 12 },
    RlEntry { code: 0x001a, bits: 15, run: 1, level: 13 },
    RlEntry { code: 0x0019, bits: 15, run: 1, level: 14 },
    RlEntry { code: 0x0013, bits: 16, run: 1, level: 15 },
    RlEntry { code: 0x0012, bits: 16, run: 1, level: 16 },
    RlEntry { code: 0x0011, bits: 16, run: 1, level: 17 },
    RlEntry { code: 0x0010, bits: 16, run: 1, level: 18 },
    RlEntry { code: 0x0005, bits: 4, run: 2, level: 1 },
    RlEntry { code: 0x0004, bits: 7, run: 2, level: 2 },
    RlEntry { code: 0x000b, bits: 10, run: 2, level: 3 },
    RlEntry { code: 0x0014, bits: 12, run: 2, level: 4 },
    RlEntry { code: 0x0014, bits: 13, run: 2, level: 5 },
    RlEntry { code: 0x0007, bits: 5, run: 3, level: 1 },
    RlEntry { code: 0x0024, bits: 8, run: 3, level: 2 },
    RlEntry { code: 0x001c, bits: 12, run: 3, level: 3 },
    RlEntry { code: 0x0013, bits: 13, run: 3, level: 4 },
    RlEntry { code: 0x0006, bits: 5, run: 4, level: 1 },
    RlEntry { code: 0x000f, bits: 10, run: 4, level: 2 },
    RlEntry { code: 0x0012, bits: 12, run: 4, level: 3 },
    RlEntry { code: 0x0007, bits: 6, run: 5, level: 1 },
    RlEntry { code: 0x0009, bits: 10, run: 5, level: 2 },
    RlEntry { code: 0x0012, bits: 13, run: 5, level: 3 },
    RlEntry { code: 0x0005, bits: 6, run: 6, level: 1 },
    RlEntry { code: 0x001e, bits: 12, run: 6, level: 2 },
    RlEntry { code: 0x0014, bits: 16, run: 6, level: 3 },
    RlEntry { code: 0x0004, bits: 6, run: 7, level: 1 },
    RlEntry { code: 0x0015, bits: 12, run: 7, level: 2 },
    RlEntry { code: 0x0007, bits: 7, run: 8, level: 1 },
    RlEntry { code: 0x0011, bits: 12, run: 8, level: 2 },
    RlEntry { code: 0x0005, bits: 7, run: 9, level: 1 },
    RlEntry { code: 0x0011, bits: 13, run: 9, level: 2 },
    RlEntry { code: 0x0027, bits: 8, run: 10, level: 1 },
    RlEntry { code: 0x0010, bits: 13, run: 10, level: 2 },
    RlEntry { code: 0x0023, bits: 8, run: 11, level: 1 },
    RlEntry { code: 0x001a, bits: 16, run: 11, level: 2 },
    RlEntry { code: 0x0022, bits: 8, run: 12, level: 1 },
    RlEntry { code: 0x0019, bits: 16, run: 12, level: 2 },
    RlEntry { code: 0x0020, bits: 8, run: 13, level: 1 },
    RlEntry { code: 0x0018, bits: 16, run: 13, level: 2 },
    RlEntry { code: 0x000e, bits: 10, run: 14, level: 1 },
    RlEntry { code: 0x0017, bits: 16, run: 14, level: 2 },
    RlEntry { code: 0x000d, bits: 10, run: 15, level: 1 },
    RlEntry { code: 0x0016, bits: 16, run: 15, level: 2 },
    RlEntry { code: 0x0008, bits: 10, run: 16, level: 1 },
    RlEntry { code: 0x0015, bits: 16, run: 16, level: 2 },
    RlEntry { code: 0x001f, bits: 12, run: 17, level: 1 },
    RlEntry { code: 0x001a, bits: 12, run: 18, level: 1 },
    RlEntry { code: 0x0019, bits: 12, run: 19, level: 1 },
    RlEntry { code: 0x0017, bits: 12, run: 20, level: 1 },
    RlEntry { code: 0x0016, bits: 12, run: 21, level: 1 },
    RlEntry { code: 0x001f, bits: 13, run: 22, level: 1 },
    RlEntry { code: 0x001e, bits: 13, run: 23, level: 1 },
    RlEntry { code: 0x001d, bits: 13, run: 24, level: 1 },
    RlEntry { code: 0x001c, bits: 13, run: 25, level: 1 },
    RlEntry { code: 0x001b, bits: 13, run: 26, level: 1 },
    RlEntry { code: 0x001f, bits: 16, run: 27, level: 1 },
    RlEntry { code: 0x001e, bits: 16, run: 28, level: 1 },
    RlEntry { code: 0x001d, bits: 16, run: 29, level: 1 },
    RlEntry { code: 0x001c, bits: 16, run: 30, level: 1 },
    RlEntry { code: 0x001b, bits: 16, run: 31, level: 1 },
];

/// 块结束码 `10`。
pub const EOB_CODE: u16 = 0b10;
pub const EOB_BITS: u8 = 2;
/// 逃逸码 `000001`，后跟定长的 run / level。
pub const ESCAPE_CODE: u16 = 0b000001;
pub const ESCAPE_BITS: u8 = 6;

/// P 帧宏块的运动来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Motion {
    /// 帧内宏块，不做运动补偿。
    None,
    /// 前向预测但运动矢量为 0（不读矢量位）。
    Zero,
    /// 前向预测，运动矢量从码流里读。
    Forward,
}

/// P 帧的一条 `macroblock_type`：码字 + 语义。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PMbType {
    pub code: u16,
    pub bits: u8,
    pub intra: bool,
    /// 后面跟 `coded_block_pattern`（决定哪些块有残差）。
    pub coded: bool,
    pub motion: Motion,
    pub quant: bool,
}

/// P 帧 `macroblock_type` 表（下标即 VLC 符号）。
pub const P_MB_TYPE: [PMbType; 7] = [
    PMbType { code: 0x0003, bits: 5, intra: true, coded: false, motion: Motion::None, quant: false },
    PMbType { code: 0x0001, bits: 2, intra: false, coded: true, motion: Motion::Zero, quant: false },
    PMbType { code: 0x0001, bits: 3, intra: false, coded: false, motion: Motion::Forward, quant: false },
    PMbType { code: 0x0001, bits: 1, intra: false, coded: true, motion: Motion::Forward, quant: false },
    PMbType { code: 0x0001, bits: 6, intra: true, coded: false, motion: Motion::None, quant: true },
    PMbType { code: 0x0001, bits: 5, intra: false, coded: true, motion: Motion::Zero, quant: true },
    PMbType { code: 0x0002, bits: 5, intra: false, coded: true, motion: Motion::Forward, quant: true },
];
