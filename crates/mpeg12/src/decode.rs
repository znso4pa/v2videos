//! MPEG-1 帧内（I）/ 前向（P）/ 双向（B）帧重建。
//!
//! slice → 宏块 → 块 → 反量化 → IDCT；P 帧加前向运动补偿，B 帧加前向/后向/双向补偿。
//! **重排序**（解码序 ≠ 显示序）不在这里 —— 由调用方按 `temporal_reference` 处理。
//!
//! 每个失败点都带上「在哪一步、第几位」，因为位对齐错一位的表现只是「后面全错」，
//! 没有位偏移就无从下手。

use crate::tables::{
    Code, DC_CHROMA, DC_LUM, EOB_BITS, EOB_CODE, ESCAPE_BITS, ESCAPE_CODE, INTRA_MATRIX,
    MB_ADDR_INCR, MB_PATTERN, MPEG1_RL, MV_CODE, NON_INTRA_MATRIX, P_MB_TYPE, B_MB_TYPE, Motion,
    ALT_SCAN, MPEG2_RL, NON_LINEAR_QSCALE, RlEntry,
};
use crate::{BitReader, PictureHeader, PictureType, SequenceHeader};


/// 解码出来的一帧（I420）。
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// 解码失败的位置。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError {
    pub stage: &'static str,
    pub bit: usize,
    pub detail: i64,
}

impl DecodeError {
    fn new(stage: &'static str, bit: usize, detail: i64) -> Self {
        Self { stage, bit, detail }
    }
}

/// 按最长前缀在 `table` 里找码，命中后消耗对应位数。
fn decode_vlc(r: &mut BitReader, table: &[Code]) -> Option<usize> {
    let max_bits = table.iter().map(|c| c.bits).max().unwrap_or(0) as u32;
    for len in 1..=max_bits {
        if r.bits_left() < len as usize {
            return None;
        }
        let value = r.peek_u(len);
        for (index, code) in table.iter().enumerate() {
            if code.bits as u32 == len && code.code as u32 == value {
                r.skip(len).ok()?;
                return Some(index);
            }
        }
    }
    None
}

/// 在 run/level 表里按最长前缀找码。
fn decode_rl(r: &mut BitReader, table: &[RlEntry]) -> Option<usize> {
    let max_bits = table.iter().map(|e| e.bits).max().unwrap_or(0) as u32;
    for len in 1..=max_bits {
        if r.bits_left() < len as usize {
            return None;
        }
        let value = r.peek_u(len);
        for (index, entry) in table.iter().enumerate() {
            if entry.bits as u32 == len && entry.code as u32 == value {
                r.skip(len).ok()?;
                return Some(index);
            }
        }
    }
    None
}

/// P 帧 `macroblock_type`：码表是 `PMbType`，单独走一遍最长前缀。
fn decode_p_mb_type(r: &mut BitReader) -> Option<usize> {
    let max_bits = P_MB_TYPE.iter().map(|c| c.bits).max().unwrap_or(0) as u32;
    for len in 1..=max_bits {
        if r.bits_left() < len as usize {
            return None;
        }
        let value = r.peek_u(len);
        for (index, entry) in P_MB_TYPE.iter().enumerate() {
            if entry.bits as u32 == len && entry.code as u32 == value {
                r.skip(len).ok()?;
                return Some(index);
            }
        }
    }
    None
}

/// B 帧 `macroblock_type`。
fn decode_b_mb_type(r: &mut BitReader) -> Option<usize> {
    let max_bits = B_MB_TYPE.iter().map(|c| c.bits).max().unwrap_or(0) as u32;
    for len in 1..=max_bits {
        if r.bits_left() < len as usize {
            return None;
        }
        let value = r.peek_u(len);
        for (index, entry) in B_MB_TYPE.iter().enumerate() {
            if entry.bits as u32 == len && entry.code as u32 == value {
                r.skip(len).ok()?;
                return Some(index);
            }
        }
    }
    None
}

/// zigzag 扫描顺序 → 光栅下标。
///
/// 码流里的系数按 zigzag 顺序排列，而量化矩阵与系数数组都是光栅顺序，
/// 所以两边都要过这张表。算法生成（按反对角线来回扫），不是手写常量。
pub const ZIGZAG: [usize; 64] = build_zigzag();

const fn build_zigzag() -> [usize; 64] {
    let mut table = [0usize; 64];
    let mut k = 0;
    let mut diagonal = 0;
    while diagonal < 15 {
        let mut step = 0;
        while step < 8 {
            // 偶数对角线从下往上，奇数从上往下 —— 这就是 zigzag 的定义。
            let (row, col) = if diagonal % 2 == 0 {
                (diagonal as isize - step, step)
            } else {
                (step, diagonal as isize - step)
            };
            if row >= 0 && row < 8 && col >= 0 && col < 8 && k < 64 {
                table[k] = (row as usize) * 8 + col as usize;
                k += 1;
            }
            step += 1;
        }
        diagonal += 1;
    }
    table
}

/// 反量化参数：MPEG-1 / MPEG-2、线性 / 非线性量化只在这里分叉。
///
/// MPEG-2 的量化器在反量化前被 `<<1`（线性）或查非线性表，所以它的移位比
/// MPEG-1 多一位，且**没有 oddification**（`2|level|+1` 已经把奇偶钉死）。
#[derive(Clone, Copy)]
struct Quant {
    q: i32,
    intra_shift: u32,
    inter_shift: u32,
    oddify: bool,
    dc_scale: i32,
    /// MPEG-2 的逃逸 level 是 12 位补码；MPEG-1 是 8 位 + 哨兵。
    mpeg2: bool,
}

impl Quant {
    fn new(scale: i32, mpeg2: bool, non_linear: bool, intra_dc_precision: u8) -> Self {
        if mpeg2 {
            let q = if non_linear {
                NON_LINEAR_QSCALE[scale as usize] as i32
            } else {
                scale << 1
            };
            Self {
                q,
                intra_shift: 4,
                inter_shift: 5,
                oddify: false,
                dc_scale: 8 >> intra_dc_precision,
                mpeg2: true,
            }
        } else {
            Self {
                q: scale,
                intra_shift: 3,
                inter_shift: 4,
                oddify: true,
                dc_scale: 8,
                mpeg2: false,
            }
        }
    }
}

/// 帧内反量化：DC 乘 `dc_scale`，交流 `(|level|·q·W) >> intra_shift`。
fn dequantise_intra(quantised: &[i32; 64], quant: Quant) -> [i32; 64] {
    let mut out = [0i32; 64];
    out[0] = quantised[0] * quant.dc_scale;
    // MPEG-2 的 mismatch control：把块内系数和的奇偶钉住。
    let mut mismatch = out[0] ^ 1;
    for i in 1..64 {
        let level = quantised[i];
        if level == 0 {
            continue;
        }
        let mut magnitude = (level.abs() * quant.q * INTRA_MATRIX[i] as i32) >> quant.intra_shift;
        if quant.oddify {
            magnitude = (magnitude - 1) | 1;
        }
        out[i] = if level < 0 { -magnitude } else { magnitude };
        mismatch ^= out[i];
    }
    if quant.mpeg2 {
        out[63] ^= mismatch & 1;
    }
    out
}

/// 帧间反量化：`((2|level|+1)·q·W) >> inter_shift`，DC 无特例。
fn dequantise_inter(quantised: &[i32; 64], quant: Quant) -> [i32; 64] {
    let mut out = [0i32; 64];
    let mut mismatch = 1;
    for i in 0..64 {
        let level = quantised[i];
        if level == 0 {
            continue;
        }
        let mut magnitude =
            ((level.abs() * 2 + 1) * quant.q * NON_INTRA_MATRIX[i] as i32) >> quant.inter_shift;
        if quant.oddify {
            magnitude = (magnitude - 1) | 1;
        }
        out[i] = if level < 0 { -magnitude } else { magnitude };
        mismatch ^= out[i];
    }
    if quant.mpeg2 {
        out[63] ^= mismatch & 1;
    }
    out
}

/// 余弦表（编译期算好，避免每帧重复求值）。
const COS: [[f64; 8]; 8] = build_cos();

const fn cos_approx(x: f64) -> f64 {
    // **必须先做周期归约**：泰勒展开只在 0 附近收敛，而这里最大的参数是
    // `15 × 7 × π / 16 ≈ 20.6` 弧度。不归约的话，误差随参数增大而爆炸 ——
    // 低频基函数（参数小）仍然准，高频全错，看起来就像「只有高频块花掉」。
    let pi = core::f64::consts::PI;
    let two_pi = 2.0 * pi;
    let mut x = x;
    while x > pi {
        x -= two_pi;
    }
    while x < -pi {
        x += two_pi;
    }
    let x2 = x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut n = 1;
    while n < 12 {
        term = -term * x2 / (((2 * n - 1) * (2 * n)) as f64);
        sum += term;
        n += 1;
    }
    sum
}

const fn build_cos() -> [[f64; 8]; 8] {
    let mut table = [[0f64; 8]; 8];
    let mut x = 0;
    while x < 8 {
        let mut u = 0;
        while u < 8 {
            table[x][u] = cos_approx(((2 * x + 1) * u) as f64 * core::f64::consts::PI / 16.0);
            u += 1;
        }
        x += 1;
    }
    table
}

/// 8×8 逆变换，返回**残差**（未加 128、未夹取）。
fn idct_residual(block: &[i32; 64]) -> [i32; 64] {
    let mut tmp = [[0f64; 8]; 8];
    for v in 0..8 {
        for x in 0..8 {
            let mut sum = 0f64;
            for u in 0..8 {
                let cu = if u == 0 { core::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
                sum += cu * block[v * 8 + u] as f64 * COS[x][u];
            }
            tmp[v][x] = sum / 2.0;
        }
    }
    let mut out = [0i32; 64];
    for x in 0..8 {
        for y in 0..8 {
            let mut sum = 0f64;
            for v in 0..8 {
                let cv = if v == 0 { core::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
                sum += cv * tmp[v][x] * COS[y][v];
            }
            out[y * 8 + x] = (sum / 2.0).round() as i32;
        }
    }
    out
}

/// 把残差（+ 一个偏移）写到平面，夹到 0..=255。帧内块偏移 128。
fn store_block(residual: &[i32; 64], offset: i32, out: &mut [u8], stride: usize, at: usize) {
    for y in 0..8 {
        for x in 0..8 {
            out[at + y * stride + x] = (residual[y * 8 + x] + offset).clamp(0, 255) as u8;
        }
    }
}

/// 预测块 + 残差，夹到 0..=255。帧间块用。
fn store_pred(pred: &[i32; 64], residual: &[i32; 64], out: &mut [u8], stride: usize, at: usize) {
    for y in 0..8 {
        for x in 0..8 {
            out[at + y * stride + x] = (pred[y * 8 + x] + residual[y * 8 + x]).clamp(0, 255) as u8;
        }
    }
}

/// 把帧内残差写到宏块的某个块（按帧 / 场 DCT 的几何）。
#[allow(clippy::too_many_arguments)]
fn store_mb_block(
    frame: &mut Frame,
    block: usize,
    column: usize,
    mb_row: usize,
    width: usize,
    cw: usize,
    interlaced_dct: bool,
    residual: &[i32; 64],
    offset: i32,
    chroma_format: u8,
) {
    let (chroma, origin_x, origin_y, stride, _) =
        block_geometry(block, column, mb_row, width, cw, interlaced_dct, chroma_format);
    let plane_width = if chroma { cw } else { width };
    let plane = match block {
        0..=3 => &mut frame.y,
        4 | 6 | 8 | 10 => &mut frame.u,
        _ => &mut frame.v,
    };
    store_block(residual, offset, plane, stride, origin_y * plane_width + origin_x);
}

/// 取预测 + 叠残差，写到宏块的某个块（按帧 / 场 DCT 的几何）。
#[allow(clippy::too_many_arguments)]
fn store_pred_block(
    frame: &mut Frame,
    reference: &Frame,
    block: usize,
    column: usize,
    mb_row: usize,
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    interlaced_dct: bool,
    mv_x: i32,
    mv_y: i32,
    residual: &[i32; 64],
    chroma_format: u8,
) {
    let (chroma, origin_x, origin_y, stride, row_step) =
        block_geometry(block, column, mb_row, width, cw, interlaced_dct, chroma_format);
    let plane_width = if chroma { cw } else { width };
    let (rw, rh) = if chroma { (cw, ch) } else { (width, height) };
    let ref_plane = match block {
        0..=3 => &reference.y,
        4 | 6 | 8 | 10 => &reference.u,
        _ => &reference.v,
    };
    let pred = predict_block(
        ref_plane, plane_width, rw, rh, origin_x, origin_y, mv_x, mv_y, chroma, row_step, 1,
        chroma_format,
    );
    let plane = match block {
        0..=3 => &mut frame.y,
        4 | 6 | 8 | 10 => &mut frame.u,
        _ => &mut frame.v,
    };
    store_pred(&pred, residual, plane, stride, origin_y * plane_width + origin_x);
}

/// 解一个帧内块。
fn decode_intra_block(
    r: &mut BitReader,
    quant: Quant,
    dc_table: &[Code],
    dc_predictor: &mut i32,
    rl: &[RlEntry],
    scan: &[usize; 64],
    eob: (u16, u8),
) -> Result<[i32; 64], DecodeError> {
    let mut coeffs = [0i32; 64];

    let size = decode_vlc(r, dc_table)
        .ok_or_else(|| DecodeError::new("dc_size", r.bit_pos(), 0))?;
    if size > 0 {
        let raw = r
            .u(size as u32)
            .map_err(|_| DecodeError::new("dc_diff", r.bit_pos(), size as i64))? as i32;
        let half = 1i32 << (size - 1);
        let diff = if raw & half != 0 { raw } else { raw - (half << 1) + 1 };
        // 帧内 DC 是**差分**编码：要累加到预测值上，否则整块偏掉。
        *dc_predictor += diff;
    }
    coeffs[0] = *dc_predictor;

    let mut index = 1usize;
    while index < 64 {
        if r.bits_left() == 0 {
            break;
        }
        if r.bits_left() >= eob.1 as usize && r.peek_u(eob.1 as u32) == eob.0 as u32 {
            r.skip(eob.1 as u32)
                .map_err(|_| DecodeError::new("eob", r.bit_pos(), 0))?;
            break;
        }
        if r.bits_left() >= ESCAPE_BITS as usize
            && r.peek_u(ESCAPE_BITS as u32) == ESCAPE_CODE as u32
        {
            r.skip(ESCAPE_BITS as u32)
                .map_err(|_| DecodeError::new("escape", r.bit_pos(), 0))?;
            let run = r
                .u(6)
                .map_err(|_| DecodeError::new("escape_run", r.bit_pos(), 0))? as usize;
            // MPEG-2 的逃逸 level 是 **12 位补码**；MPEG-1 是 8 位 + 哨兵。
            let level = if quant.mpeg2 {
                let raw = r
                    .u(12)
                    .map_err(|_| DecodeError::new("escape_level12", r.bit_pos(), 0))?
                    as i32;
                if raw & 0x800 != 0 { raw - 4096 } else { raw }
            } else {
                // 8 位补码，然后两个**哨兵值**各有专门含义（照参考实现）：
                //   -128 → 再读 8 位，值减 256（覆盖 -256..-1）
                //      0 → 再读 8 位，值即 level（覆盖 1..255）
                let raw = r
                    .u(8)
                    .map_err(|_| DecodeError::new("escape_level", r.bit_pos(), 0))? as i32;
                let mut level = if raw & 0x80 != 0 { raw - 256 } else { raw };
                if level == -128 {
                    let extra = r
                        .u(8)
                        .map_err(|_| DecodeError::new("escape_neg", r.bit_pos(), 0))? as i32;
                    level = extra - 256;
                } else if level == 0 {
                    level = r
                        .u(8)
                        .map_err(|_| DecodeError::new("escape_pos", r.bit_pos(), 0))?
                        as i32;
                }
                level
            };
            index += run;
            if index < 64 {
                coeffs[scan[index]] = level;
            }
            index += 1;
            continue;
        }

        let entry =
            decode_rl(r, rl).ok_or_else(|| DecodeError::new("rl", r.bit_pos(), index as i64))?;
        let run = rl[entry].run as usize;
        let level = rl[entry].level as i32;
        let negative = r
            .u(1)
            .map_err(|_| DecodeError::new("rl_sign", r.bit_pos(), 0))?
            == 1;
        index += run;
        if index < 64 {
            coeffs[scan[index]] = if negative { -level } else { level };
        }
        index += 1;
    }

    // 块被 64 个系数填满时，EOB 仍然在码流里 —— 不消费它，后面全部错位。
    if index >= 64
        && r.bits_left() >= eob.1 as usize
        && r.peek_u(eob.1 as u32) == eob.0 as u32
    {
        r.skip(eob.1 as u32)
            .map_err(|_| DecodeError::new("eob_full", r.bit_pos(), 0))?;
    }

    Ok(dequantise_intra(&coeffs, quant))
}

/// 解一个帧间块（P/B 帧残差）。没有 DC 预测，系数从扫描第 0 位开始。
fn decode_inter_block(
    r: &mut BitReader,
    quant: Quant,
    rl: &[RlEntry],
    scan: &[usize; 64],
    eob: (u16, u8),
    field_picture: bool,
) -> Result<[i32; 64], DecodeError> {
    let mut coeffs = [0i32; 64];
    let mut index = 0usize;

    // 首系数快速路径（照参考实现）：首码以 '1' 开头时就是 (run 0, level 1)，
    // 第 2 位当符号，**只吃 2 位**。走通用 RL 会多吃 1 位，从第二个色度块起就错位。
    if r.bits_left() >= 2 && r.peek_u(1) == 1 {
        let negative = (r.peek_u(2) & 1) == 1;
        r.skip(2).ok();
        coeffs[scan[0]] = if negative { -1 } else { 1 };
        index = 1;
        // 场图的参考实现用 `GET_CACHE <= 0xBFFFFFFF`（下两位不是 `11`）判 EOB；
        // 帧图用显式的 `10`。两者不能混用，否则整段位流会偏。
        let eob_hit = if field_picture {
            r.bits_left() >= 2 && r.peek_u(2) != 0b11
        } else {
            r.bits_left() >= eob.1 as usize && r.peek_u(eob.1 as u32) == eob.0 as u32
        };
        if eob_hit {
            r.skip(2)
                .map_err(|_| DecodeError::new("eob_fast", r.bit_pos(), 0))?;
            return Ok(dequantise_inter(&coeffs, quant));
        }
    }

    while index < 64 {
        if r.bits_left() == 0 {
            break;
        }
        if r.bits_left() >= eob.1 as usize && r.peek_u(eob.1 as u32) == eob.0 as u32 {
            r.skip(eob.1 as u32)
                .map_err(|_| DecodeError::new("eob_inter", r.bit_pos(), 0))?;
            break;
        }
        if r.bits_left() >= ESCAPE_BITS as usize
            && r.peek_u(ESCAPE_BITS as u32) == ESCAPE_CODE as u32
        {
            r.skip(ESCAPE_BITS as u32)
                .map_err(|_| DecodeError::new("escape_inter", r.bit_pos(), 0))?;
            let run = r
                .u(6)
                .map_err(|_| DecodeError::new("escape_run_inter", r.bit_pos(), 0))?
                as usize;
            let level = if quant.mpeg2 {
                let raw = r
                    .u(12)
                    .map_err(|_| DecodeError::new("escape_level12_inter", r.bit_pos(), 0))?
                    as i32;
                if raw & 0x800 != 0 { raw - 4096 } else { raw }
            } else {
                let raw = r
                    .u(8)
                    .map_err(|_| DecodeError::new("escape_level_inter", r.bit_pos(), 0))?
                    as i32;
                let mut level = if raw & 0x80 != 0 { raw - 256 } else { raw };
                if level == -128 {
                    let extra = r
                        .u(8)
                        .map_err(|_| DecodeError::new("escape_neg_inter", r.bit_pos(), 0))?
                        as i32;
                    level = extra - 256;
                } else if level == 0 {
                    level = r
                        .u(8)
                        .map_err(|_| DecodeError::new("escape_pos_inter", r.bit_pos(), 0))?
                        as i32;
                }
                level
            };
            index += run;
            if index < 64 {
                coeffs[scan[index]] = level;
            }
            index += 1;
            continue;
        }

        let entry = decode_rl(r, rl)
            .ok_or_else(|| DecodeError::new("rl_inter", r.bit_pos(), index as i64))?;
        let run = rl[entry].run as usize;
        let level = rl[entry].level as i32;
        let negative = r
            .u(1)
            .map_err(|_| DecodeError::new("rl_sign_inter", r.bit_pos(), 0))?
            == 1;
        index += run;
        if index < 64 {
            coeffs[scan[index]] = if negative { -level } else { level };
        }
        index += 1;
    }

    if index >= 64
        && r.bits_left() >= eob.1 as usize
        && r.peek_u(eob.1 as u32) == eob.0 as u32
    {
        r.skip(eob.1 as u32)
            .map_err(|_| DecodeError::new("eob_full_inter", r.bit_pos(), 0))?;
    }

    Ok(dequantise_inter(&coeffs, quant))
}

/// 把 `val` 当 `bits` 位补码解释（MPEG 运动矢量的模运算）。
fn sign_extend(val: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (val << shift) >> shift
}

/// 解一个运动矢量分量（`motion_code` + 残差 + 预测），返回**半像素**单位。
/// 读 `macroblock_address_increment`：返回**要从当前位置前进多少个宏块**；
/// `None` = 这个 slice 到此结束。
///
/// ⚠️ 表里有三个**不是增量**的符号（下标 33 / 34 / 35）：
/// - 33 = `macroblock_escape`：**再加 33** 后继续读下一个增量（一帧里连续跳过的
///   宏块可能超过 33 个）；
/// - 34 = stuffing：跳过，继续读；
/// - 35 = 片结束（后面还会跟一串 0 位）。
///
/// 把它们当成「增量 34 / 35 / 36」会让位游标整体错位 —— 症状是解到某一片时
/// `coded_block_pattern` 的 VLC 突然解不出来。实测 MPEG-1 的 B 帧末行必现
/// （那里正好是一长串跳过宏块 + escape）。
fn decode_mb_increment(r: &mut BitReader) -> Option<isize> {
    let mut total = 0isize;
    loop {
        match decode_vlc(r, &MB_ADDR_INCR)? {
            33 => total += 33,
            34 => {}
            35 => return None,
            code => return Some(total + code as isize + 1),
        }
    }
}

fn decode_motion(r: &mut BitReader, fcode: u32, pred: i32) -> Option<i32> {
    let code = decode_vlc(r, &MV_CODE)? as i32;
    if code == 0 {
        return Some(pred);
    }
    let sign = r.u(1).ok()? == 1;
    let shift = fcode.saturating_sub(1);
    let mut val = code;
    if shift > 0 {
        val = (val - 1) << shift;
        val |= r.u(shift).ok()? as i32;
        val += 1;
    }
    if sign {
        val = -val;
    }
    Some(sign_extend(val + pred, 5 + shift))
}

/// 色度平面的尺寸。
fn chroma_dims(chroma_format: u8, width: usize, height: usize) -> (usize, usize) {
    match chroma_format {
        2 => (width.div_ceil(2), height),
        3 => (width, height),
        _ => (width.div_ceil(2), height.div_ceil(2)),
    }
}

/// 一个宏块的块数：4:2:0 = 6，4:2:2 = 8，4:4:4 = 12。
fn mb_block_count(chroma_format: u8) -> usize {
    4 + (1 << chroma_format)
}

/// 解一个 dual prime 的差分矢量（`dmv`：先 1 位，非零再 1 位定 ±1）。
fn decode_dmv(r: &mut BitReader) -> Option<i32> {
    if r.u(1).ok()? == 1 {
        Some(1 - ((r.u(1).ok()? as i32) << 1))
    } else {
        Some(0)
    }
}

/// 宏块内第 `block` 个块的几何：`(是否色度, 原点 x, 原点 y, 写步长, 取行步长)`。
///
/// 场 DCT（`interlaced_dct=1`）时亮度块隔行写 / 取：块 0/1 在偶行、2/3 在奇行，
/// 行步长 2；色度在 4:2:0 下始终按帧 DCT 摆放。
fn block_geometry(
    block: usize,
    column: usize,
    mb_row: usize,
    width: usize,
    cw: usize,
    interlaced_dct: bool,
    chroma_format: u8,
) -> (bool, usize, usize, usize, i32) {
    match block {
        0..=3 => {
            let (bx, by) = (block % 2, block / 2);
            let origin_x = column * 16 + bx * 8;
            if interlaced_dct {
                (false, origin_x, mb_row * 16 + by, 2 * width, 2)
            } else {
                (false, origin_x, mb_row * 16 + by * 8, width, 1)
            }
        }
        _ => {
            // 4:2:0：一个 Cb / Cr；4:2:2：上下两个；4:4:4：2×2 四个。
            let (bx, by) = match chroma_format {
                2 => (0, if block >= 6 { 1 } else { 0 }),
                3 => (
                    if block >= 8 { 1 } else { 0 },
                    if (block % 4) >= 2 { 1 } else { 0 },
                ),
                _ => (0, 0),
            };
            let chroma_mb_w = if chroma_format == 3 { 16 } else { 8 };
            let origin_x = column * chroma_mb_w + bx * 8;
            if chroma_format == 1 {
                // 4:2:0 的色度始终帧基（参考实现也只用 uvlinesize）。
                (true, origin_x, mb_row * 8, cw, 1)
            } else if interlaced_dct {
                (true, origin_x, mb_row * 16 + by, 2 * cw, 2)
            } else {
                (true, origin_x, mb_row * 16 + by * 8, cw, 1)
            }
        }
    }
}

/// 从参考帧按运动矢量取一个 8×8 预测块（半像素双线性）。
///
/// 亮度：整数部分 `mv >> 1`、小数位 `mv & 1`。
/// 色度：先把矢量按 `mv / 2`（向零截断）缩到色度采样，再 `>> 1` 取整数、
/// `& 1` 取小数 —— 这两步不能合成 `mv >> 2`，负矢量会差一格。
fn predict_block(
    reference: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    mv_x: i32,
    mv_y: i32,
    chroma: bool,
    row_step: i32,
    mv_y_scale: i32,
    chroma_format: u8,
) -> [i32; 64] {
    let (mx, my) = if chroma {
        let (hs, vs) = match chroma_format {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        (mv_x / hs, mv_y / vs)
    } else {
        (mv_x, mv_y)
    };
    let ix = mx >> 1;
    let iy = (my >> 1) * mv_y_scale;
    let fx = mx & 1;
    let fy = my & 1;
    let mut out = [0i32; 64];
    for y in 0..8 {
        for x in 0..8 {
            let sx = origin_x as i32 + ix + x as i32;
            let sy = origin_y as i32 + row_step * y as i32 + iy;
            let at = |dx: i32, dy: i32| -> i32 {
                let px = (sx + dx).clamp(0, width as i32 - 1) as usize;
                let py = (sy + dy).clamp(0, height as i32 - 1) as usize;
                reference[py * stride + px] as i32
            };
            let a = at(0, 0);
            out[y * 8 + x] = match (fx, fy) {
                (0, 0) => a,
                (1, 0) => (a + at(1, 0) + 1) >> 1,
                (0, 1) => (a + at(0, row_step) + 1) >> 1,
                _ => (a + at(1, 0) + at(0, row_step) + at(1, row_step) + 2) >> 2,
            };
        }
    }
    out
}

/// 每幅图不变的块级配置：RL 表、扫描顺序、EOB 码。
struct BlockTables<'a> {
    intra_rl: &'a [RlEntry],
    inter_rl: &'a [RlEntry],
    scan: &'a [usize; 64],
    intra_eob: (u16, u8),
    inter_eob: (u16, u8),
}

fn block_tables(ext: Option<&crate::PictureCodingExtension>) -> BlockTables<'static> {
    match ext {
        Some(ext) => BlockTables {
            // `ff_mpeg2_vlc_table` 只服务 `intra_vlc_format=1` 的**帧内**块；
            // MPEG-2 的帧间块仍用 MPEG-1 的 RL 表（参考实现里 `mpeg2_decode_block_non_intra`
            // 就是 `ff_mpeg1_rl_vlc`）。
            intra_rl: if ext.intra_vlc_format { &MPEG2_RL } else { &MPEG1_RL },
            inter_rl: &MPEG1_RL,
            scan: if ext.alternate_scan { &ALT_SCAN } else { &ZIGZAG },
            // **EOB 码不同**：MPEG-2 的 RL 表是 `0110`（4 位），MPEG-1 是 `10`（2 位）。
            intra_eob: if ext.intra_vlc_format {
                (0b0110, 4)
            } else {
                (EOB_CODE, EOB_BITS)
            },
            inter_eob: (EOB_CODE, EOB_BITS),
        },
        None => BlockTables {
            intra_rl: &MPEG1_RL,
            inter_rl: &MPEG1_RL,
            scan: &ZIGZAG,
            intra_eob: (EOB_CODE, EOB_BITS),
            inter_eob: (EOB_CODE, EOB_BITS),
        },
    }
}

/// 在图像数据里找 picture coding extension（`00 00 01 B5`，id = 8）。
fn picture_extension(data: &[u8]) -> Option<crate::PictureCodingExtension> {
    crate::split_units(data)
        .iter()
        .find(|(code, _)| *code == 0xB5)
        .and_then(|(_, payload)| crate::PictureCodingExtension::parse(payload))
}

/// MPEG-2 扩展头校验：只支持 progressive 帧图、4:2:0、`frame_pred_frame_dct`。
fn mpeg2_extension(
    sequence: &SequenceHeader,
    data: &[u8],
) -> Result<Option<crate::PictureCodingExtension>, DecodeError> {
    if !sequence.mpeg2_stream() {
        return Ok(None);
    }
    let ext = picture_extension(data)
        .ok_or_else(|| DecodeError::new("missing_picture_extension", 0, 0))?;
    Ok(Some(ext))
}

/// 解一帧 I 图像。
pub fn decode_i_frame(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
) -> Result<Frame, DecodeError> {
    if !matches!(picture.picture_type, PictureType::I) {
        return Err(DecodeError::new("picture_type", 0, 0));
    }
    let ext = mpeg2_extension(sequence, data)?;
    let tables = block_tables(ext.as_ref());
    let non_linear = ext.as_ref().map_or(false, |e| e.q_scale_type);
    let precision = ext.as_ref().map_or(0, |e| e.intra_dc_precision);
    let frame_picture = ext.as_ref().map_or(true, |e| e.picture_structure == 3);
    let read_interlaced_dct =
        frame_picture && ext.as_ref().map_or(false, |e| !e.frame_pred_frame_dct);

    let width = sequence.width as usize;
    let height = sequence.height as usize;
    let chroma_format = sequence.chroma_format;
    let (cw, ch) = chroma_dims(chroma_format, width, height);
    let mb_blocks = mb_block_count(chroma_format);
    let mut frame = Frame {
        width: sequence.width,
        height: sequence.height,
        y: vec![0; width * height],
        u: vec![128; cw * ch],
        v: vec![128; cw * ch],
    };

    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    let mut decoded_slices = 0usize;

    for (code, slice) in crate::split_units(data) {
        if !(0x01..=0xAF).contains(&code) {
            continue;
        }
        let row = (code as usize) - 1;
        if row >= mb_h {
            continue;
        }
        decoded_slices += 1;

        let mut r = BitReader::new(slice);
        let mut scale = r
            .u(5)
            .map_err(|_| DecodeError::new("slice_scale", r.bit_pos(), code as i64))? as i32;
        if scale == 0 {
            return Err(DecodeError::new("slice_scale_zero", r.bit_pos(), code as i64));
        }
        // extra_bit_slice
        while r
            .u(1)
            .map_err(|_| DecodeError::new("extra_bit_slice", r.bit_pos(), code as i64))?
            == 1
        {
            r.u(8)
                .map_err(|_| DecodeError::new("extra_info_slice", r.bit_pos(), code as i64))?;
        }

        // 帧内 DC 预测值在 slice 起始重置，亮度与两个色度各自独立。
        let mut dc_predictor = [0i32; 3];
        // slice 不等于一个宏块行：9 个 slice 覆盖 15 行时，后面的 slice 会跨行。
        // 所以按**光栅序号**推进，而不是按行号定位。
        let mut mb_index = (row * mb_w) as isize - 1;
        let total_mb = mb_w * mb_h;
        loop {
            let Some(increment) = decode_mb_increment(&mut r) else {
                break;
            };
            mb_index += increment;
            if mb_index < 0 || mb_index as usize >= total_mb {
                break;
            }
            let column = mb_index as usize % mb_w;
            let mb_row = mb_index as usize / mb_w;

            // I 图像：'1' = 帧内；'01' = 帧内 + 新量化器。
            let first = r
                .u(1)
                .map_err(|_| DecodeError::new("mb_type", r.bit_pos(), column as i64))?;
            let quantised_mb = if first == 1 {
                false
            } else if r
                .u(1)
                .map_err(|_| DecodeError::new("mb_type2", r.bit_pos(), column as i64))?
                == 1
            {
                true
            } else {
                return Err(DecodeError::new("mb_type_00", r.bit_pos(), column as i64));
            };
            // MPEG-2 隔行：帧内宏块在量化器之前有一个 interlaced_dct 位。
            let interlaced_dct = if read_interlaced_dct {
                r.u(1)
                    .map_err(|_| DecodeError::new("interlaced_dct", r.bit_pos(), column as i64))?
                    == 1
            } else {
                false
            };
            if quantised_mb {
                scale = r
                    .u(5)
                    .map_err(|_| DecodeError::new("mb_scale", r.bit_pos(), column as i64))?
                    as i32;
                if scale == 0 {
                    return Err(DecodeError::new("mb_scale_zero", r.bit_pos(), column as i64));
                }
            }

            let quant = Quant::new(scale, sequence.mpeg2_stream(), non_linear, precision);
            for block in 0..mb_blocks {
                let dc_table = if block < 4 { &DC_LUM } else { &DC_CHROMA };
                let component = if block < 4 { 0 } else { (block - 4) % 2 + 1 };
                let coeffs = decode_intra_block(
                    &mut r,
                    quant,
                    dc_table,
                    &mut dc_predictor[component],
                    tables.intra_rl,
                    tables.scan,
                    tables.intra_eob,
                )?;
                let residual = idct_residual(&coeffs);
                store_mb_block(
                    &mut frame, block, column, mb_row, width, cw, interlaced_dct, &residual, 128,
                    chroma_format,
                );
            }
        }
    }

    if decoded_slices == 0 {
        return Err(DecodeError::new("no_slices", 0, 0));
    }
    Ok(frame)
}

/// 把场运动的预测写进整帧：亮度每个场两个 8×8、色度每个场 8×4，都隔行写。
#[allow(clippy::too_many_arguments)]
fn write_field_prediction(
    frame: &mut Frame,
    reference: &Frame,
    column: usize,
    mb_row: usize,
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    field_select: [u8; 2],
    mv: [[i32; 2]; 2],
    chroma_format: u8,
) {
    // 亮度：场 j 的两个 8×8 左右块。源取 `field_select[j]` 场，目标写第 j 场（偶/奇行）。
    for j in 0..2 {
        for bx in 0..2 {
            let origin_x = column * 16 + bx * 8;
            let src_y = mb_row * 16 + field_select[j] as usize;
            let dst_y = mb_row * 16 + j;
            let pred = predict_block(
                &reference.y, width, width, height, origin_x, src_y, mv[j][0], mv[j][1],
                false, 2, 2, chroma_format,
            );
            store_block(&pred, 0, &mut frame.y, 2 * width, dst_y * width + origin_x);
        }
    }
    // 色度：4:2:0 每个场 8×4；4:2:2 / 4:4:4 每个场 8×8。
    let chroma_rows = if chroma_format == 1 { 4 } else { 8 };
    let chroma_mb_h = if chroma_format == 1 { 8 } else { 16 };
    let chroma_mb_w = if chroma_format == 3 { 16 } else { 8 };
    for j in 0..2 {
        let origin_x = column * chroma_mb_w;
        let src_y = mb_row * chroma_mb_h + field_select[j] as usize;
        let dst_y = mb_row * chroma_mb_h + j;
        let pred_u = predict_block(
            &reference.u, cw, cw, ch, origin_x, src_y, mv[j][0], mv[j][1], true, 2, 2,
            chroma_format,
        );
        let pred_v = predict_block(
            &reference.v, cw, cw, ch, origin_x, src_y, mv[j][0], mv[j][1], true, 2, 2,
            chroma_format,
        );
        for y in 0..chroma_rows {
            for x in 0..8 {
                let idx = dst_y * cw + origin_x + y * (2 * cw) + x;
                frame.u[idx] = pred_u[y * 8 + x].clamp(0, 255) as u8;
                frame.v[idx] = pred_v[y * 8 + x].clamp(0, 255) as u8;
            }
        }
    }
}

/// 写一个**场图**宏块的预测：`field_select` 选参考场，目标写半高场。
#[allow(clippy::too_many_arguments)]
fn write_field_picture_mb(
    frame: &mut Frame,
    same: &Frame,
    other: &Frame,
    column: usize,
    mb_row: usize,
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    chroma_format: u8,
    field_select: u8,
    mv: [i32; 2],
) {
    let reference = if field_select == 1 { other } else { same };
    for block in 0..mb_block_count(chroma_format) {
        let (chroma, origin_x, origin_y, stride, _) =
            block_geometry(block, column, mb_row, width, cw, false, chroma_format);
        let plane_width = if chroma { cw } else { width };
        let (rw, rh) = if chroma { (cw, ch) } else { (width, height) };
        let ref_plane = match block {
            0..=3 => &reference.y,
            4 | 6 | 8 | 10 => &reference.u,
            _ => &reference.v,
        };
        let pred = predict_block(
            ref_plane, plane_width, rw, rh, origin_x, origin_y, mv[0], mv[1], chroma, 1, 1,
            chroma_format,
        );
        let out = match block {
            0..=3 => &mut frame.y,
            4 | 6 | 8 | 10 => &mut frame.u,
            _ => &mut frame.v,
        };
        store_block(&pred, 0, out, stride, origin_y * plane_width + origin_x);
    }
}

/// 写一个 dual prime 宏块的预测：同场 `mv0`、异场 `mv2`，两路平均。
#[allow(clippy::too_many_arguments)]
fn write_dmv_prediction(
    frame: &mut Frame,
    same: &Frame,
    other: &Frame,
    column: usize,
    mb_row: usize,
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    chroma_format: u8,
    mv0: [i32; 2],
    mv2: [i32; 2],
) {
    for block in 0..mb_block_count(chroma_format) {
        let (chroma, origin_x, origin_y, stride, _) =
            block_geometry(block, column, mb_row, width, cw, false, chroma_format);
        let plane_width = if chroma { cw } else { width };
        let (rw, rh) = if chroma { (cw, ch) } else { (width, height) };
        let (sp, op): (&[u8], &[u8]) = if chroma {
            let is_u = matches!(block, 4 | 6 | 8 | 10);
            (
                if is_u { &same.u } else { &same.v },
                if is_u { &other.u } else { &other.v },
            )
        } else {
            (&same.y, &other.y)
        };
        let p0 = predict_block(
            sp, plane_width, rw, rh, origin_x, origin_y, mv0[0], mv0[1], chroma, 1, 1,
            chroma_format,
        );
        let p2 = predict_block(
            op, plane_width, rw, rh, origin_x, origin_y, mv2[0], mv2[1], chroma, 1, 1,
            chroma_format,
        );
        let mut pred = [0i32; 64];
        for i in 0..64 {
            pred[i] = (p0[i] + p2[i] + 1) >> 1;
        }
        let out = match block {
            0..=3 => &mut frame.y,
            4 | 6 | 8 | 10 => &mut frame.u,
            _ => &mut frame.v,
        };
        store_block(&pred, 0, out, stride, origin_y * plane_width + origin_x);
    }
}

/// 把残差加到已写好的预测上（场运动用）。
#[allow(clippy::too_many_arguments)]
fn add_mb_block(
    frame: &mut Frame,
    block: usize,
    column: usize,
    mb_row: usize,
    width: usize,
    cw: usize,
    interlaced_dct: bool,
    residual: &[i32; 64],
    chroma_format: u8,
) {
    let (chroma, origin_x, origin_y, stride, _) =
        block_geometry(block, column, mb_row, width, cw, interlaced_dct, chroma_format);
    let plane_width = if chroma { cw } else { width };
    let plane = match block {
        0..=3 => &mut frame.y,
        4 | 6 | 8 | 10 => &mut frame.u,
        _ => &mut frame.v,
    };
    let at = origin_y * plane_width + origin_x;
    for y in 0..8 {
        for x in 0..8 {
            let idx = at + y * stride + x;
            plane[idx] = (plane[idx] as i32 + residual[y * 8 + x]).clamp(0, 255) as u8;
        }
    }
}

/// 解一帧 P 图像：以 `reference` 为底做前向运动补偿 + 残差。
pub fn decode_p_frame(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    reference: &Frame,
    other: Option<&Frame>,
) -> Result<Frame, DecodeError> {
    if !matches!(picture.picture_type, PictureType::P) {
        return Err(DecodeError::new("picture_type", 0, 0));
    }
    let ext = mpeg2_extension(sequence, data)?;
    let tables = block_tables(ext.as_ref());
    let non_linear = ext.as_ref().map_or(false, |e| e.q_scale_type);
    let precision = ext.as_ref().map_or(0, |e| e.intra_dc_precision);
    let (picture_structure, fpfd) = match ext.as_ref() {
        Some(e) => (e.picture_structure, e.frame_pred_frame_dct),
        None => (3, true),
    };
    let frame_picture = picture_structure == 3;
    let read_motion_type = !(frame_picture && fpfd);
    let read_interlaced_dct = frame_picture && !fpfd;
    if reference.width != sequence.width || reference.height != sequence.height {
        return Err(DecodeError::new("reference_size", 0, 0));
    }

    let width = sequence.width as usize;
    let height = sequence.height as usize;
    let chroma_format = sequence.chroma_format;
    let (cw, ch) = chroma_dims(chroma_format, width, height);
    let mb_blocks = mb_block_count(chroma_format);
    // 以参考帧为底：**被跳过的宏块天然就是「零运动矢量、无残差」的预测**，
    // 不需要单独处理 —— 后面解到的宏块再覆盖它。
    let mut frame = Frame {
        width: sequence.width,
        height: sequence.height,
        y: reference.y.clone(),
        u: reference.u.clone(),
        v: reference.v.clone(),
    };

    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    let total_mb = mb_w * mb_h;
    let (fcode_h, fcode_v) = match ext.as_ref() {
        Some(e) => (e.f_code[0][0], e.f_code[0][1]),
        None => (picture.forward_f_code, picture.forward_f_code),
    };
    let mut decoded_slices = 0usize;

    for (code, slice) in crate::split_units(data) {
        if !(0x01..=0xAF).contains(&code) {
            continue;
        }
        let row = (code as usize) - 1;
        if row >= mb_h {
            continue;
        }
        decoded_slices += 1;

        let mut r = BitReader::new(slice);
        let mut scale = r
            .u(5)
            .map_err(|_| DecodeError::new("slice_scale", r.bit_pos(), code as i64))? as i32;
        if scale == 0 {
            return Err(DecodeError::new("slice_scale_zero", r.bit_pos(), code as i64));
        }
        while r
            .u(1)
            .map_err(|_| DecodeError::new("extra_bit_slice", r.bit_pos(), code as i64))?
            == 1
        {
            r.u(8)
                .map_err(|_| DecodeError::new("extra_info_slice", r.bit_pos(), code as i64))?;
        }

        // 帧内 DC 预测器：每个 slice 从 0 起；遇到帧间宏块重置（对应参考实现的
        // `last_dc = 128`，与我们的 0 基准等价）。
        let mut dc_predictor = [0i32; 3];
        // 运动矢量预测器：每个 slice 从 0 起；按场分开存（帧运动时两场同值）。
        let mut last_mv_f = [[0i32; 2]; 2];
        let mut mb_index = (row * mb_w) as isize - 1;
        loop {
            let Some(increment) = decode_mb_increment(&mut r) else {
                break;
            };
            let previous = mb_index;
            mb_index += increment;
            if mb_index < 0 || mb_index as usize >= total_mb {
                break;
            }
            // 跳过的宏块：已经是参考帧原位拷贝（帧初始化时克隆好了），
            // 只需把运动矢量预测器归零。
            //
            // ⚠️ **帧内 DC 预测器也要归零**：跳过宏块按定义是「非帧内」，
            // 而规范要求每个非帧内宏块之后 DC 预测器清零。漏了这一步，
            // 下一个帧内宏块的 DC 会带着上一次的值 → 整块恒定偏移
            // （亮度 4 块共用一路预测器，所以亮度整块偏、色度不受影响）。
            if mb_index > previous + 1 {
                last_mv_f = [[0, 0], [0, 0]];
                dc_predictor = [0i32; 3];
            }
            let column = mb_index as usize % mb_w;
            let mb_row = mb_index as usize / mb_w;

            // slice 末尾的填充位可能凑出一个无效的 mb_type —— 当作本 slice 结束。
            let Some(entry) = decode_p_mb_type(&mut r) else { break; };
            let mb = P_MB_TYPE[entry];

            // MPEG-2 隔行：帧内读 interlaced_dct；零矢量只读 interlaced_dct；
            // 非零矢量先读 motion_type 再读 interlaced_dct。
            let (motion_type, interlaced_dct) = if mb.intra {
                let dct = if read_interlaced_dct {
                    r.u(1)
                        .map_err(|_| DecodeError::new("interlaced_dct", r.bit_pos(), column as i64))?
                        == 1
                } else {
                    false
                };
                (2u32, dct)
            } else if read_motion_type {
                match mb.motion {
                    Motion::Zero => {
                        let dct = r
                            .u(1)
                            .map_err(|_| DecodeError::new("interlaced_dct", r.bit_pos(), column as i64))?
                            == 1;
                        (2u32, dct)
                    }
                    Motion::Forward => {
                        let mt = r.u(2).map_err(|_| {
                            DecodeError::new("motion_type", r.bit_pos(), column as i64)
                        })?;
                        let dct = if read_interlaced_dct && mb.coded {
                            r.u(1)
                                .map_err(|_| {
                                    DecodeError::new("interlaced_dct", r.bit_pos(), column as i64)
                                })?
                                == 1
                        } else {
                            false
                        };
                        (mt, dct)
                    }
                    Motion::None => (2, false),
                }
            } else {
                (2, false)
            };
            if !mb.intra && motion_type != 2 && motion_type != 1 && motion_type != 3 {
                return Err(DecodeError::new(
                    "motion_type_unsupported",
                    r.bit_pos(),
                    motion_type as i64,
                ));
            }

            if mb.quant {
                scale = r
                    .u(5)
                    .map_err(|_| DecodeError::new("mb_scale", r.bit_pos(), column as i64))?
                    as i32;
                if scale == 0 {
                    return Err(DecodeError::new("mb_scale_zero", r.bit_pos(), column as i64));
                }
            }
            let quant = Quant::new(scale, sequence.mpeg2_stream(), non_linear, precision);

            if mb.intra {
                last_mv_f = [[0, 0], [0, 0]];
                for block in 0..mb_blocks {
                    let dc_table = if block < 4 { &DC_LUM } else { &DC_CHROMA };
                    let component = if block < 4 { 0 } else { (block - 4) % 2 + 1 };
                    let coeffs = decode_intra_block(
                        &mut r,
                        quant,
                        dc_table,
                        &mut dc_predictor[component],
                        tables.intra_rl,
                        tables.scan,
                        tables.intra_eob,
                    )?;
                    let residual = idct_residual(&coeffs);
                    store_mb_block(
                        &mut frame, block, column, mb_row, width, cw, interlaced_dct, &residual,
                        128, chroma_format,
                    );
                }
                continue;
            }

            // 帧间：先运动矢量，再 cbp，再各块残差。
            let mut field_motion = false;
            let mut dmv = false;
            let mut field_select = [0u8; 2];
            let mut mv_f = [[0i32; 2]; 2];
            match mb.motion {
                Motion::Forward => {
                    if frame_picture {
                        // 帧图：mt=1 场运动（field_select + 每场一对矢量）；否则帧运动。
                        if motion_type == 1 {
                            field_motion = true;
                            for j in 0..2 {
                                field_select[j] = r
                                    .u(1)
                                    .map_err(|_| {
                                        DecodeError::new("field_select", r.bit_pos(), column as i64)
                                    })? as u8;
                                let x = decode_motion(&mut r, fcode_h, last_mv_f[j][0]).ok_or_else(
                                    || DecodeError::new("mv_x", r.bit_pos(), column as i64),
                                )?;
                                last_mv_f[j][0] = x;
                                let y = decode_motion(&mut r, fcode_v, last_mv_f[j][1] >> 1)
                                    .ok_or_else(|| {
                                        DecodeError::new("mv_y", r.bit_pos(), column as i64)
                                    })?;
                                last_mv_f[j][1] = 2 * y;
                                mv_f[j] = [x, y];
                            }
                        } else {
                            let x = decode_motion(&mut r, fcode_h, last_mv_f[0][0])
                                .ok_or_else(|| DecodeError::new("mv_x", r.bit_pos(), column as i64))?;
                            let y = decode_motion(&mut r, fcode_v, last_mv_f[0][1])
                                .ok_or_else(|| DecodeError::new("mv_y", r.bit_pos(), column as i64))?;
                            last_mv_f[0] = [x, y];
                            last_mv_f[1] = [x, y];
                            mv_f[0] = [x, y];
                            mv_f[1] = [x, y];
                        }
                    } else {
                        // 场图：mt=1 = field_select + 一对矢量；mt=2 = 一对矢量；mt=3 = dual prime。
                        if motion_type == 1 {
                            field_motion = true;
                            field_select[0] = r
                                .u(1)
                                .map_err(|_| {
                                    DecodeError::new("field_select", r.bit_pos(), column as i64)
                                })? as u8;
                            let x = decode_motion(&mut r, fcode_h, last_mv_f[0][0])
                                .ok_or_else(|| DecodeError::new("mv_x", r.bit_pos(), column as i64))?;
                            last_mv_f[0][0] = x;
                            let y = decode_motion(&mut r, fcode_v, last_mv_f[0][1])
                                .ok_or_else(|| DecodeError::new("mv_y", r.bit_pos(), column as i64))?;
                            last_mv_f[0][1] = y;
                            last_mv_f[1] = [x, y];
                            mv_f[0] = [x, y];
                            mv_f[1] = [x, y];
                        } else if motion_type == 3 {
                            let mx = decode_motion(&mut r, fcode_h, last_mv_f[0][0])
                                .ok_or_else(|| DecodeError::new("mv_x", r.bit_pos(), column as i64))?;
                            last_mv_f[0][0] = mx;
                            last_mv_f[1][0] = mx;
                            let dmx = decode_dmv(&mut r)
                                .ok_or_else(|| DecodeError::new("dmv_x", r.bit_pos(), column as i64))?;
                            let my = decode_motion(&mut r, fcode_v, last_mv_f[0][1])
                                .ok_or_else(|| DecodeError::new("mv_y", r.bit_pos(), column as i64))?;
                            last_mv_f[0][1] = my;
                            last_mv_f[1][1] = my;
                            let dmy = decode_dmv(&mut r)
                                .ok_or_else(|| DecodeError::new("dmv_y", r.bit_pos(), column as i64))?;
                            let m = if picture_structure == 1 { -1 } else { 1 };
                            mv_f[0] = [mx, my];
                            mv_f[1] = [
                                ((mx + (mx > 0) as i32) >> 1) + dmx,
                                ((my + (my > 0) as i32) >> 1) + dmy + m,
                            ];
                            dmv = true;
                        } else {
                            let x = decode_motion(&mut r, fcode_h, last_mv_f[0][0])
                                .ok_or_else(|| DecodeError::new("mv_x", r.bit_pos(), column as i64))?;
                            let y = decode_motion(&mut r, fcode_v, last_mv_f[0][1])
                                .ok_or_else(|| DecodeError::new("mv_y", r.bit_pos(), column as i64))?;
                            last_mv_f[0] = [x, y];
                            last_mv_f[1] = [x, y];
                            mv_f[0] = [x, y];
                            mv_f[1] = [x, y];
                        }
                    }
                    if picture.full_pel_forward {
                        // full_pel 的矢量是整像素，换成半像素单位再补偿。
                        for j in 0..2 {
                            mv_f[j][0] *= 2;
                            mv_f[j][1] *= 2;
                        }
                    }
                }
                Motion::Zero | Motion::None => {
                    last_mv_f = [[0, 0], [0, 0]];
                }
            }
            // 帧间宏块之后，帧内 DC 预测器归零。
            dc_predictor = [0i32; 3];

            let cbp = if mb.coded {
                let base = decode_vlc(&mut r, &MB_PATTERN)
                    .ok_or_else(|| DecodeError::new("cbp", r.bit_pos(), column as i64))?
                    as u32;
                if mb_blocks > 6 {
                    let extra = r
                        .u((mb_blocks - 6) as u32)
                        .map_err(|_| DecodeError::new("cbp_ext", r.bit_pos(), column as i64))?
                        as u32;
                    (base << (mb_blocks - 6)) | extra
                } else {
                    base
                }
            } else {
                0
            };

            if dmv {
                // dual prime：同场 `reference`、异场 `other`，两路平均。
                let other_ref = other.unwrap_or(reference);
                write_dmv_prediction(
                    &mut frame, reference, other_ref, column, mb_row, width, height, cw, ch,
                    chroma_format, mv_f[0], mv_f[1],
                );
                for block in 0..mb_blocks {
                    let coded = mb.coded && (cbp & (1 << (mb_blocks - 1 - block))) != 0;
                    if coded {
                        let coeffs =
                            decode_inter_block(&mut r, quant, tables.inter_rl, tables.scan, tables.inter_eob, !frame_picture)?;
                        let residual = idct_residual(&coeffs);
                        add_mb_block(
                            &mut frame, block, column, mb_row, width, cw, interlaced_dct,
                            &residual, chroma_format,
                        );
                    }
                }
            } else if field_motion {
                if frame_picture {
                    // 帧图：两场预测写进整帧。
                    write_field_prediction(
                        &mut frame, reference, column, mb_row, width, height, cw, ch,
                        field_select, mv_f, chroma_format,
                    );
                } else {
                    // 场图：`field_select` 选参考场，写半高场。
                    write_field_picture_mb(
                        &mut frame,
                        reference,
                        other.unwrap_or(reference),
                        column,
                        mb_row,
                        width,
                        height,
                        cw,
                        ch,
                        chroma_format,
                        field_select[0],
                        mv_f[0],
                    );
                }
                for block in 0..mb_blocks {
                    let coded = mb.coded && (cbp & (1 << (mb_blocks - 1 - block))) != 0;
                    if coded {
                        let coeffs =
                            decode_inter_block(&mut r, quant, tables.inter_rl, tables.scan, tables.inter_eob, !frame_picture)?;
                        let residual = idct_residual(&coeffs);
                        add_mb_block(
                            &mut frame, block, column, mb_row, width, cw, interlaced_dct,
                            &residual, chroma_format,
                        );
                    }
                }
            } else {
                for block in 0..mb_blocks {
                    let coded = mb.coded && (cbp & (1 << (mb_blocks - 1 - block))) != 0;
                    let residual = if coded {
                        let coeffs =
                            decode_inter_block(&mut r, quant, tables.inter_rl, tables.scan, tables.inter_eob, !frame_picture)?;
                        idct_residual(&coeffs)
                    } else {
                        [0i32; 64]
                    };
                    store_pred_block(
                        &mut frame, reference, block, column, mb_row, width, height, cw, ch,
                        interlaced_dct, mv_f[0][0], mv_f[0][1], &residual, chroma_format,
                    );
                }
            }
        }
    }

    if decoded_slices == 0 {
        return Err(DecodeError::new("no_slices", 0, 0));
    }
    Ok(frame)
}

/// 写一个 B 帧块：按前向 / 后向 / 双向取预测，再叠残差。
#[allow(clippy::too_many_arguments)]
fn write_b_block(
    frame: &mut Frame,
    forward: &Frame,
    backward: &Frame,
    block: usize,
    column: usize,
    mb_row: usize,
    fwd: bool,
    bwd: bool,
    mv_f: (i32, i32),
    mv_b: (i32, i32),
    residual: &[i32; 64],
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    interlaced_dct: bool,
    chroma_format: u8,
) {
    let (chroma, origin_x, origin_y, stride, row_step) =
        block_geometry(block, column, mb_row, width, cw, interlaced_dct, chroma_format);
    let plane_width = if chroma { cw } else { width };
    let (rw, rh) = if chroma { (cw, ch) } else { (width, height) };
    let (f_plane, b_plane): (&[u8], &[u8]) = if chroma {
        let is_u = matches!(block, 4 | 6 | 8 | 10);
        let f = if is_u { &forward.u } else { &forward.v };
        let b = if is_u { &backward.u } else { &backward.v };
        (f, b)
    } else {
        (&forward.y, &backward.y)
    };
    let pf = if fwd {
        Some(predict_block(
            f_plane, plane_width, rw, rh, origin_x, origin_y, mv_f.0, mv_f.1, chroma, row_step, 1,
            chroma_format,
        ))
    } else {
        None
    };
    let pb = if bwd {
        Some(predict_block(
            b_plane, plane_width, rw, rh, origin_x, origin_y, mv_b.0, mv_b.1, chroma, row_step, 1,
            chroma_format,
        ))
    } else {
        None
    };
    let mut pred = [0i32; 64];
    for i in 0..64 {
        pred[i] = match (&pf, &pb) {
            // 双向预测取两路平均（四舍五入）。
            (Some(a), Some(b)) => (a[i] + b[i] + 1) >> 1,
            (Some(a), None) => a[i],
            (None, Some(b)) => b[i],
            (None, None) => 0,
        };
    }
    let out_plane = match block {
        0..=3 => &mut frame.y,
        4 | 6 | 8 | 10 => &mut frame.u,
        _ => &mut frame.v,
    };
    store_pred(&pred, residual, out_plane, stride, origin_y * plane_width + origin_x);
}

/// 一个只有预测、没有残差的 B 帧宏块（跳过宏块）。
#[allow(clippy::too_many_arguments)]
fn write_predicted_mb(
    frame: &mut Frame,
    forward: &Frame,
    backward: &Frame,
    column: usize,
    mb_row: usize,
    fwd: bool,
    bwd: bool,
    mv_f: (i32, i32),
    mv_b: (i32, i32),
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    interlaced_dct: bool,
    chroma_format: u8,
    mb_blocks: usize,
) {
    let zero = [0i32; 64];
    for block in 0..mb_blocks {
        write_b_block(
            frame, forward, backward, block, column, mb_row, fwd, bwd, mv_f, mv_b, &zero, width,
            height, cw, ch, interlaced_dct, chroma_format,
        );
    }
}

/// 合并前向 / 后向两路预测（双向取平均）。
fn combine_pred(pf: &Option<[i32; 64]>, pb: &Option<[i32; 64]>) -> [i32; 64] {
    let mut out = [0i32; 64];
    for i in 0..64 {
        out[i] = match (pf, pb) {
            (Some(a), Some(b)) => (a[i] + b[i] + 1) >> 1,
            (Some(a), None) => a[i],
            (None, Some(b)) => b[i],
            (None, None) => 0,
        };
    }
    out
}

/// 写一个 B 帧场运动宏块的预测（两场各自前向 / 后向 / 双向）。
#[allow(clippy::too_many_arguments)]
fn write_b_field_prediction(
    frame: &mut Frame,
    forward: &Frame,
    backward: &Frame,
    column: usize,
    mb_row: usize,
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    fwd: bool,
    bwd: bool,
    fs_f: [u8; 2],
    mv_f: [[i32; 2]; 2],
    fs_b: [u8; 2],
    mv_b: [[i32; 2]; 2],
    chroma_format: u8,
) {
    for j in 0..2 {
        // 亮度：场 j 的两个 8×8。
        for bx in 0..2 {
            let origin_x = column * 16 + bx * 8;
            let dst_y = mb_row * 16 + j;
            let pf = if fwd {
                let src_y = mb_row * 16 + fs_f[j] as usize;
                Some(predict_block(
                    &forward.y, width, width, height, origin_x, src_y, mv_f[j][0], mv_f[j][1],
                    false, 2, 2, chroma_format,
                ))
            } else {
                None
            };
            let pb = if bwd {
                let src_y = mb_row * 16 + fs_b[j] as usize;
                Some(predict_block(
                    &backward.y, width, width, height, origin_x, src_y, mv_b[j][0], mv_b[j][1],
                    false, 2, 2, chroma_format,
                ))
            } else {
                None
            };
            let pred = combine_pred(&pf, &pb);
            store_block(&pred, 0, &mut frame.y, 2 * width, dst_y * width + origin_x);
        }
        // 色度：4:2:0 每个场 8×4；4:2:2 / 4:4:4 每个场 8×8。
        let chroma_rows = if chroma_format == 1 { 4 } else { 8 };
        let chroma_mb_h = if chroma_format == 1 { 8 } else { 16 };
        let chroma_mb_w = if chroma_format == 3 { 16 } else { 8 };
        let origin_x = column * chroma_mb_w;
        let dst_y = mb_row * chroma_mb_h + j;
        let pred = |plane_f: &[u8], plane_b: &[u8]| -> [i32; 64] {
            let pf = if fwd {
                let src_y = mb_row * chroma_mb_h + fs_f[j] as usize;
                Some(predict_block(
                    plane_f, cw, cw, ch, origin_x, src_y, mv_f[j][0], mv_f[j][1], true, 2, 2,
                    chroma_format,
                ))
            } else {
                None
            };
            let pb = if bwd {
                let src_y = mb_row * chroma_mb_h + fs_b[j] as usize;
                Some(predict_block(
                    plane_b, cw, cw, ch, origin_x, src_y, mv_b[j][0], mv_b[j][1], true, 2, 2,
                    chroma_format,
                ))
            } else {
                None
            };
            combine_pred(&pf, &pb)
        };
        let pu = pred(&forward.u, &backward.u);
        let pv = pred(&forward.v, &backward.v);
        for y in 0..chroma_rows {
            for x in 0..8 {
                let idx = dst_y * cw + origin_x + y * (2 * cw) + x;
                frame.u[idx] = pu[y * 8 + x].clamp(0, 255) as u8;
                frame.v[idx] = pv[y * 8 + x].clamp(0, 255) as u8;
            }
        }
    }
}

/// B 帧「上一个已编码宏块」的完整状态（跳过宏块沿用）。
#[derive(Clone, Copy)]
struct BPrev {
    kind: crate::tables::BMbType,
    field_motion: bool,
    fs_f: [u8; 2],
    mv_f: [[i32; 2]; 2],
    fs_b: [u8; 2],
    mv_b: [[i32; 2]; 2],
}

/// 解一帧 B 图像：前向 / 后向 / 双向运动补偿 + 残差。
///
/// 参考帧由调用方按**解码序**维护：`forward` 是前一个 I/P，`backward` 是紧随其后的
/// I/P（在码流里它比 B 帧先出现）。
pub fn decode_b_frame(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    forward: &Frame,
    backward: &Frame,
) -> Result<Frame, DecodeError> {
    if !matches!(picture.picture_type, PictureType::B) {
        return Err(DecodeError::new("picture_type", 0, 0));
    }
    let ext = mpeg2_extension(sequence, data)?;
    let tables = block_tables(ext.as_ref());
    let non_linear = ext.as_ref().map_or(false, |e| e.q_scale_type);
    let precision = ext.as_ref().map_or(0, |e| e.intra_dc_precision);
    let (frame_picture, fpfd) = match ext.as_ref() {
        Some(e) => (e.picture_structure == 3, e.frame_pred_frame_dct),
        None => (true, true),
    };
    let read_motion_type = !(frame_picture && fpfd);
    let read_interlaced_dct = frame_picture && !fpfd;
    if forward.width != sequence.width
        || forward.height != sequence.height
        || backward.width != sequence.width
        || backward.height != sequence.height
    {
        return Err(DecodeError::new("reference_size", 0, 0));
    }

    let width = sequence.width as usize;
    let height = sequence.height as usize;
    let chroma_format = sequence.chroma_format;
    let (cw, ch) = chroma_dims(chroma_format, width, height);
    let mb_blocks = mb_block_count(chroma_format);
    let mut frame = Frame {
        width: sequence.width,
        height: sequence.height,
        y: forward.y.clone(),
        u: forward.u.clone(),
        v: forward.v.clone(),
    };

    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    let total_mb = mb_w * mb_h;
    let fcode_f = picture.forward_f_code;
    let fcode_b = picture.backward_f_code;
    let (fcode_fh, fcode_fv, fcode_bh, fcode_bv) = match ext.as_ref() {
        Some(e) => (e.f_code[0][0], e.f_code[0][1], e.f_code[1][0], e.f_code[1][1]),
        None => (fcode_f, fcode_f, fcode_b, fcode_b),
    };
    let mut decoded_slices = 0usize;

    for (code, slice) in crate::split_units(data) {
        if !(0x01..=0xAF).contains(&code) {
            continue;
        }
        let row = (code as usize) - 1;
        if row >= mb_h {
            continue;
        }
        decoded_slices += 1;

        let mut r = BitReader::new(slice);
        let mut scale = r
            .u(5)
            .map_err(|_| DecodeError::new("slice_scale", r.bit_pos(), code as i64))? as i32;
        if scale == 0 {
            return Err(DecodeError::new("slice_scale_zero", r.bit_pos(), code as i64));
        }
        while r
            .u(1)
            .map_err(|_| DecodeError::new("extra_bit_slice", r.bit_pos(), code as i64))?
            == 1
        {
            r.u(8)
                .map_err(|_| DecodeError::new("extra_info_slice", r.bit_pos(), code as i64))?;
        }

        let mut dc_predictor = [0i32; 3];
        // 前向 / 后向的运动矢量预测器，各自按场独立。
        let mut last_mv_f = [[0i32; 2]; 2];
        let mut last_mv_b = [[0i32; 2]; 2];
        // 上一个已编码宏块：跳过宏块沿用它（类型 + 两路矢量 + 场信息）。
        let mut prev: Option<BPrev> = None;
        let mut mb_index = (row * mb_w) as isize - 1;
        loop {
            let Some(increment) = decode_mb_increment(&mut r) else {
                break;
            };
            let previous = mb_index;
            mb_index += increment;
            if mb_index < 0 || mb_index as usize >= total_mb {
                break;
            }

            // 跳过的宏块：沿用上一个已编码宏块的类型与矢量，无残差。
            for skipped in (previous + 1)..mb_index {
                let Some(p) = prev else {
                    return Err(DecodeError::new("b_skip_no_prev", r.bit_pos(), skipped as i64));
                };
                if p.kind.intra {
                    return Err(DecodeError::new(
                        "b_skip_after_intra",
                        r.bit_pos(),
                        skipped as i64,
                    ));
                }
                // 跳过宏块 = 非帧内，DC 预测器要清零（同 P 帧那条，见那边的注释）。
                dc_predictor = [0i32; 3];
                let col = skipped as usize % mb_w;
                let row_s = skipped as usize / mb_w;
                if p.field_motion {
                    write_b_field_prediction(
                        &mut frame, forward, backward, col, row_s, width, height, cw, ch,
                        p.kind.forward, p.kind.backward, p.fs_f, p.mv_f, p.fs_b, p.mv_b,
                        chroma_format,
                    );
                } else {
                    write_predicted_mb(
                        &mut frame, forward, backward, col, row_s, p.kind.forward,
                        p.kind.backward, (p.mv_f[0][0], p.mv_f[0][1]),
                        (p.mv_b[0][0], p.mv_b[0][1]), width, height, cw, ch, false,
                        chroma_format, mb_blocks,
                    );
                }
            }

            let column = mb_index as usize % mb_w;
            let mb_row = mb_index as usize / mb_w;

            let Some(entry) = decode_b_mb_type(&mut r) else { break; };
            let mb = B_MB_TYPE[entry];

            // MPEG-2 隔行：与 P 帧同样的 motion_type / interlaced_dct 位置。
            let (motion_type, interlaced_dct) = if mb.intra {
                let dct = if read_interlaced_dct {
                    r.u(1)
                        .map_err(|_| DecodeError::new("interlaced_dct", r.bit_pos(), column as i64))?
                        == 1
                } else {
                    false
                };
                (2u32, dct)
            } else if read_motion_type {
                let mt = r
                    .u(2)
                    .map_err(|_| DecodeError::new("motion_type", r.bit_pos(), column as i64))?;
                let dct = if read_interlaced_dct && mb.coded {
                    r.u(1)
                        .map_err(|_| DecodeError::new("interlaced_dct", r.bit_pos(), column as i64))?
                        == 1
                } else {
                    false
                };
                (mt, dct)
            } else {
                (2, false)
            };
            if !mb.intra && motion_type != 2 && motion_type != 1 {
                return Err(DecodeError::new(
                    "motion_type_unsupported",
                    r.bit_pos(),
                    motion_type as i64,
                ));
            }

            if mb.quant {
                scale = r
                    .u(5)
                    .map_err(|_| DecodeError::new("mb_scale", r.bit_pos(), column as i64))?
                    as i32;
                if scale == 0 {
                    return Err(DecodeError::new("mb_scale_zero", r.bit_pos(), column as i64));
                }
            }
            let quant = Quant::new(scale, sequence.mpeg2_stream(), non_linear, precision);

            if mb.intra {
                last_mv_f = [[0, 0], [0, 0]];
                last_mv_b = [[0, 0], [0, 0]];
                for block in 0..mb_blocks {
                    let dc_table = if block < 4 { &DC_LUM } else { &DC_CHROMA };
                    let component = if block < 4 { 0 } else { (block - 4) % 2 + 1 };
                    let coeffs = decode_intra_block(
                        &mut r,
                        quant,
                        dc_table,
                        &mut dc_predictor[component],
                        tables.intra_rl,
                        tables.scan,
                        tables.intra_eob,
                    )?;
                    let residual = idct_residual(&coeffs);
                    store_mb_block(
                        &mut frame, block, column, mb_row, width, cw, interlaced_dct, &residual,
                        128, chroma_format,
                    );
                }
                prev = Some(BPrev {
                    kind: mb,
                    field_motion: false,
                    fs_f: [0, 0],
                    mv_f: [[0, 0], [0, 0]],
                    fs_b: [0, 0],
                    mv_b: [[0, 0], [0, 0]],
                });
                continue;
            }

            // 运动矢量：前向、后向各自独立解码，各自累加预测。
            let field_motion = motion_type == 1;
            let mut fs_f = [0u8; 2];
            let mut fs_b = [0u8; 2];
            let mut mv_f = [[0i32; 2]; 2];
            let mut mv_b = [[0i32; 2]; 2];
            if mb.forward {
                if field_motion {
                    for j in 0..2 {
                        fs_f[j] = r
                            .u(1)
                            .map_err(|_| {
                                DecodeError::new("field_select", r.bit_pos(), column as i64)
                            })? as u8;
                        let x = decode_motion(&mut r, fcode_fh, last_mv_f[j][0]).ok_or_else(
                            || DecodeError::new("mv_fx", r.bit_pos(), column as i64),
                        )?;
                        last_mv_f[j][0] = x;
                        let y = decode_motion(&mut r, fcode_fv, last_mv_f[j][1] >> 1)
                            .ok_or_else(|| {
                                DecodeError::new("mv_fy", r.bit_pos(), column as i64)
                            })?;
                        last_mv_f[j][1] = 2 * y;
                        mv_f[j] = [x, y];
                    }
                } else {
                    let x = decode_motion(&mut r, fcode_fh, last_mv_f[0][0])
                        .ok_or_else(|| DecodeError::new("mv_fx", r.bit_pos(), column as i64))?;
                    let y = decode_motion(&mut r, fcode_fv, last_mv_f[0][1])
                        .ok_or_else(|| DecodeError::new("mv_fy", r.bit_pos(), column as i64))?;
                    last_mv_f[0] = [x, y];
                    last_mv_f[1] = [x, y];
                    mv_f[0] = [x, y];
                    mv_f[1] = [x, y];
                }
                if picture.full_pel_forward {
                    for j in 0..2 {
                        mv_f[j][0] *= 2;
                        mv_f[j][1] *= 2;
                    }
                }
            }
            if mb.backward {
                if field_motion {
                    for j in 0..2 {
                        fs_b[j] = r
                            .u(1)
                            .map_err(|_| {
                                DecodeError::new("field_select", r.bit_pos(), column as i64)
                            })? as u8;
                        let x = decode_motion(&mut r, fcode_bh, last_mv_b[j][0]).ok_or_else(
                            || DecodeError::new("mv_bx", r.bit_pos(), column as i64),
                        )?;
                        last_mv_b[j][0] = x;
                        let y = decode_motion(&mut r, fcode_bv, last_mv_b[j][1] >> 1)
                            .ok_or_else(|| {
                                DecodeError::new("mv_by", r.bit_pos(), column as i64)
                            })?;
                        last_mv_b[j][1] = 2 * y;
                        mv_b[j] = [x, y];
                    }
                } else {
                    let x = decode_motion(&mut r, fcode_bh, last_mv_b[0][0])
                        .ok_or_else(|| DecodeError::new("mv_bx", r.bit_pos(), column as i64))?;
                    let y = decode_motion(&mut r, fcode_bv, last_mv_b[0][1])
                        .ok_or_else(|| DecodeError::new("mv_by", r.bit_pos(), column as i64))?;
                    last_mv_b[0] = [x, y];
                    last_mv_b[1] = [x, y];
                    mv_b[0] = [x, y];
                    mv_b[1] = [x, y];
                }
                if picture.full_pel_backward {
                    for j in 0..2 {
                        mv_b[j][0] *= 2;
                        mv_b[j][1] *= 2;
                    }
                }
            }
            dc_predictor = [0i32; 3];

            let cbp = if mb.coded {
                let base = decode_vlc(&mut r, &MB_PATTERN)
                    .ok_or_else(|| DecodeError::new("cbp", r.bit_pos(), column as i64))?
                    as u32;
                if mb_blocks > 6 {
                    let extra = r
                        .u((mb_blocks - 6) as u32)
                        .map_err(|_| DecodeError::new("cbp_ext", r.bit_pos(), column as i64))?
                        as u32;
                    (base << (mb_blocks - 6)) | extra
                } else {
                    base
                }
            } else {
                0
            };

            if field_motion {
                write_b_field_prediction(
                    &mut frame, forward, backward, column, mb_row, width, height, cw, ch,
                    mb.forward, mb.backward, fs_f, mv_f, fs_b, mv_b, chroma_format,
                );
                for block in 0..mb_blocks {
                    let coded = mb.coded && (cbp & (1 << (mb_blocks - 1 - block))) != 0;
                    if coded {
                        let coeffs =
                            decode_inter_block(&mut r, quant, tables.inter_rl, tables.scan, tables.inter_eob, !frame_picture)?;
                        let residual = idct_residual(&coeffs);
                        add_mb_block(
                            &mut frame, block, column, mb_row, width, cw, interlaced_dct,
                            &residual, chroma_format,
                        );
                    }
                }
            } else {
                for block in 0..mb_blocks {
                    let coded = mb.coded && (cbp & (1 << (mb_blocks - 1 - block))) != 0;
                    let residual = if coded {
                        let coeffs =
                            decode_inter_block(&mut r, quant, tables.inter_rl, tables.scan, tables.inter_eob, !frame_picture)?;
                        idct_residual(&coeffs)
                    } else {
                        [0i32; 64]
                    };
                    write_b_block(
                        &mut frame, forward, backward, block, column, mb_row, mb.forward,
                        mb.backward, (mv_f[0][0], mv_f[0][1]), (mv_b[0][0], mv_b[0][1]),
                        &residual, width, height, cw, ch, interlaced_dct, chroma_format,
                    );
                }
            }
            prev = Some(BPrev {
                kind: mb,
                field_motion,
                fs_f,
                mv_f,
                fs_b,
                mv_b,
            });
        }
    }

    if decoded_slices == 0 {
        return Err(DecodeError::new("no_slices", 0, 0));
    }
    Ok(frame)
}

/// 按图像类型分派。P 帧给 `forward`；B 帧给 `forward` 与 `backward`。
pub fn decode_frame(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    forward: Option<&Frame>,
    backward: Option<&Frame>,
) -> Result<Frame, DecodeError> {
    if let Some(ext) = mpeg2_extension(sequence, data)? {
        if ext.picture_structure != 3 {
            return Err(DecodeError::new("field_picture_in_frame_path", 0, 0));
        }
    }
    match picture.picture_type {
        PictureType::I => decode_i_frame(sequence, picture, data),
        PictureType::P => {
            let forward =
                forward.ok_or_else(|| DecodeError::new("missing_reference", 0, 0))?;
            decode_p_frame(sequence, picture, data, forward, None)
        }
        PictureType::B => {
            let forward =
                forward.ok_or_else(|| DecodeError::new("missing_reference", 0, 0))?;
            let backward =
                backward.ok_or_else(|| DecodeError::new("missing_backward_reference", 0, 0))?;
            decode_b_frame(sequence, picture, data, forward, backward)
        }
        _ => Err(DecodeError::new("unsupported_picture_type", 0, 0)),
    }
}

/// 解一个**场**（`picture_structure = 1/2`）：半高画面。
///
/// 参考是**上一场**（调用方按同奇偶维护）。输出是半高的 `Frame`，
/// 上下两场由调用方交织成整帧。
pub fn decode_field(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    same: Option<&Frame>,
    other: Option<&Frame>,
) -> Result<Frame, DecodeError> {
    let ext = mpeg2_extension(sequence, data)?
        .ok_or_else(|| DecodeError::new("missing_picture_extension", 0, 0))?;
    if ext.picture_structure == 3 {
        return Err(DecodeError::new("not_a_field", 0, 0));
    }
    let mut field_seq = sequence.clone();
    field_seq.height /= 2;
    match picture.picture_type {
        PictureType::I => decode_i_frame(&field_seq, picture, data),
        PictureType::P => {
            let same = same.ok_or_else(|| DecodeError::new("missing_reference", 0, 0))?;
            decode_p_frame(&field_seq, picture, data, same, other)
        }
        PictureType::B => {
            let same = same.ok_or_else(|| DecodeError::new("missing_reference", 0, 0))?;
            let other =
                other.ok_or_else(|| DecodeError::new("missing_backward_reference", 0, 0))?;
            decode_b_frame(&field_seq, picture, data, same, other)
        }
        _ => Err(DecodeError::new("unsupported_picture_type", 0, 0)),
    }
}
