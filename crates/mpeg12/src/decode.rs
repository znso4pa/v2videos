//! MPEG-1 帧内（I）/ 前向预测（P）帧重建。
//!
//! 只做 I / P：slice → 宏块 → 块 → 反量化 → IDCT，P 帧再加运动补偿。
//! B 帧要双向预测与重排序，不在这里。
//!
//! 每个失败点都带上「在哪一步、第几位」，因为位对齐错一位的表现只是「后面全错」，
//! 没有位偏移就无从下手。

use crate::tables::{
    Code, DC_CHROMA, DC_LUM, EOB_BITS, EOB_CODE, ESCAPE_BITS, ESCAPE_CODE, INTRA_MATRIX,
    MB_ADDR_INCR, MB_PATTERN, MPEG1_RL, MV_CODE, NON_INTRA_MATRIX, P_MB_TYPE, Motion,
};
use crate::{BitReader, PictureHeader, PictureType, SequenceHeader};


/// 解码出来的一帧（I420）。
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
fn decode_rl(r: &mut BitReader) -> Option<usize> {
    let max_bits = MPEG1_RL.iter().map(|e| e.bits).max().unwrap_or(0) as u32;
    for len in 1..=max_bits {
        if r.bits_left() < len as usize {
            return None;
        }
        let value = r.peek_u(len);
        for (index, entry) in MPEG1_RL.iter().enumerate() {
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

/// 反量化：帧内 DC 是 `8 × diff`（不乘 scale），交流是 `(2·level·matrix·scale)/16`。
fn dequantise_intra(quantised: &[i32; 64], scale: i32) -> [i32; 64] {
    let mut out = [0i32; 64];
    out[0] = quantised[0] * 8;
    for i in 1..64 {
        let level = quantised[i];
        if level == 0 {
            continue;
        }
        // 交流系数按 `(|level| * scale * W) >> 3` 反量化，然后**强制奇数**
        // （MPEG-1 的 oddification：`(v - 1) | 1`）。少了这一步，偶数结果会差 1，
        // 高频块经过 IDCT 之后误差被放大成可见的错。
        let magnitude = (level.abs() * scale * INTRA_MATRIX[i] as i32) >> 3;
        let odd = (magnitude - 1) | 1;
        out[i] = if level < 0 { -odd } else { odd };
    }
    out
}

/// 帧间反量化：`((2|level|+1) · scale · W) >> 4`，然后强制奇数。
///
/// 帧间块没有 DC 特例 —— 第 0 个系数和其余一样走这条公式。`2|level|+1`
/// 是 MPEG-1 非帧内反量化的定义，`(v-1)|1` 再把它钉成奇数。
fn dequantise_inter(quantised: &[i32; 64], scale: i32) -> [i32; 64] {
    let mut out = [0i32; 64];
    for i in 0..64 {
        let level = quantised[i];
        if level == 0 {
            continue;
        }
        let magnitude = ((level.abs() * 2 + 1) * scale * NON_INTRA_MATRIX[i] as i32) >> 4;
        let odd = (magnitude - 1) | 1;
        out[i] = if level < 0 { -odd } else { odd };
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

/// 解一个帧内块。
fn decode_intra_block(
    r: &mut BitReader,
    scale: i32,
    dc_table: &[Code],
    dc_predictor: &mut i32,
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
        if r.bits_left() >= EOB_BITS as usize && r.peek_u(EOB_BITS as u32) == EOB_CODE as u32 {
            r.skip(EOB_BITS as u32)
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
            // 逃逸的 level 是**8 位补码**，不是「8 位 + 一个符号位」——
            // 多读那一位会让后面全部错位。
            let raw = r
                .u(8)
                .map_err(|_| DecodeError::new("escape_level", r.bit_pos(), 0))? as i32;
            // 8 位补码，然后两个**哨兵值**各有专门含义（照参考实现）：
            //   -128 → 再读 8 位，值减 256（覆盖 -256..-1）
            //      0 → 再读 8 位，值即 level（覆盖 1..255）
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
            index += run;
            if index < 64 {
                coeffs[ZIGZAG[index]] = level;
            }
            index += 1;
            continue;
        }

        let entry =
            decode_rl(r).ok_or_else(|| DecodeError::new("rl", r.bit_pos(), index as i64))?;
        let run = MPEG1_RL[entry].run as usize;
        let level = MPEG1_RL[entry].level as i32;
        let negative = r
            .u(1)
            .map_err(|_| DecodeError::new("rl_sign", r.bit_pos(), 0))?
            == 1;
        index += run;
        if index < 64 {
            coeffs[ZIGZAG[index]] = if negative { -level } else { level };
        }
        index += 1;
    }

    // 块被 64 个系数填满时，EOB 仍然在码流里 —— 不消费它，后面全部错位。
    if index >= 64
        && r.bits_left() >= EOB_BITS as usize
        && r.peek_u(EOB_BITS as u32) == EOB_CODE as u32
    {
        r.skip(EOB_BITS as u32)
            .map_err(|_| DecodeError::new("eob_full", r.bit_pos(), 0))?;
    }

    Ok(dequantise_intra(&coeffs, scale))
}

/// 解一个帧间块（P 帧残差）。没有 DC 预测，系数从 zigzag 第 0 位开始。
fn decode_inter_block(r: &mut BitReader, scale: i32) -> Result<[i32; 64], DecodeError> {
    let mut coeffs = [0i32; 64];
    let mut index = 0usize;

    // 首系数快速路径（照参考实现）：首码以 '1' 开头时就是 (run 0, level 1)，
    // 第 2 位当符号，**只吃 2 位**。走通用 RL 会多吃 1 位，从第二个色度块起就错位。
    if r.bits_left() >= 2 && r.peek_u(1) == 1 {
        let negative = (r.peek_u(2) & 1) == 1;
        r.skip(2).ok();
        coeffs[ZIGZAG[0]] = if negative { -1 } else { 1 };
        index = 1;
        if r.bits_left() >= EOB_BITS as usize && r.peek_u(EOB_BITS as u32) == EOB_CODE as u32 {
            r.skip(EOB_BITS as u32)
                .map_err(|_| DecodeError::new("eob_fast", r.bit_pos(), 0))?;
            return Ok(dequantise_inter(&coeffs, scale));
        }
    }

    while index < 64 {
        if r.bits_left() == 0 {
            break;
        }
        if r.bits_left() >= EOB_BITS as usize && r.peek_u(EOB_BITS as u32) == EOB_CODE as u32 {
            r.skip(EOB_BITS as u32)
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
            index += run;
            if index < 64 {
                coeffs[ZIGZAG[index]] = level;
            }
            index += 1;
            continue;
        }

        let entry = decode_rl(r)
            .ok_or_else(|| DecodeError::new("rl_inter", r.bit_pos(), index as i64))?;
        let run = MPEG1_RL[entry].run as usize;
        let level = MPEG1_RL[entry].level as i32;
        let negative = r
            .u(1)
            .map_err(|_| DecodeError::new("rl_sign_inter", r.bit_pos(), 0))?
            == 1;
        index += run;
        if index < 64 {
            coeffs[ZIGZAG[index]] = if negative { -level } else { level };
        }
        index += 1;
    }

    if index >= 64
        && r.bits_left() >= EOB_BITS as usize
        && r.peek_u(EOB_BITS as u32) == EOB_CODE as u32
    {
        r.skip(EOB_BITS as u32)
            .map_err(|_| DecodeError::new("eob_full_inter", r.bit_pos(), 0))?;
    }

    Ok(dequantise_inter(&coeffs, scale))
}

/// 把 `val` 当 `bits` 位补码解释（MPEG 运动矢量的模运算）。
fn sign_extend(val: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (val << shift) >> shift
}

/// 解一个运动矢量分量（`motion_code` + 残差 + 预测），返回**半像素**单位。
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

/// 宏块内第 `block` 个块在平面上的原点。
///
/// 0..=3 是亮度 8×8 子块（宏块内 2×2 排列），4 / 5 是 Cb / Cr。
fn block_origin(block: usize, column: usize, mb_row: usize) -> (bool, usize, usize) {
    match block {
        0..=3 => {
            let (bx, by) = (block % 2, block / 2);
            (false, column * 16 + bx * 8, mb_row * 16 + by * 8)
        }
        _ => (true, column * 8, mb_row * 8),
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
) -> [i32; 64] {
    let (mx, my) = if chroma { (mv_x / 2, mv_y / 2) } else { (mv_x, mv_y) };
    let ix = mx >> 1;
    let iy = my >> 1;
    let fx = mx & 1;
    let fy = my & 1;
    let mut out = [0i32; 64];
    for y in 0..8 {
        for x in 0..8 {
            let sx = origin_x as i32 + ix + x as i32;
            let sy = origin_y as i32 + iy + y as i32;
            let at = |dx: i32, dy: i32| -> i32 {
                let px = (sx + dx).clamp(0, width as i32 - 1) as usize;
                let py = (sy + dy).clamp(0, height as i32 - 1) as usize;
                reference[py * stride + px] as i32
            };
            let a = at(0, 0);
            out[y * 8 + x] = match (fx, fy) {
                (0, 0) => a,
                (1, 0) => (a + at(1, 0) + 1) >> 1,
                (0, 1) => (a + at(0, 1) + 1) >> 1,
                _ => (a + at(1, 0) + at(0, 1) + at(1, 1) + 2) >> 2,
            };
        }
    }
    out
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
    // 目前只做 MPEG-1。MPEG-2 的 macroblock_type 是另一套码表（还多了帧/场预测的
    // 类型），沿用这套会解出**错帧**而不是报错 —— 所以宁可明确拒绝。
    if sequence.mpeg2_stream() {
        return Err(DecodeError::new("mpeg2_unsupported", 0, 0));
    }

    let width = sequence.width as usize;
    let height = sequence.height as usize;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
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
            let Some(increment) = decode_vlc(&mut r, &MB_ADDR_INCR) else {
                break;
            };
            mb_index += increment as isize + 1;
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
            if quantised_mb {
                scale = r
                    .u(5)
                    .map_err(|_| DecodeError::new("mb_scale", r.bit_pos(), column as i64))?
                    as i32;
                if scale == 0 {
                    return Err(DecodeError::new("mb_scale_zero", r.bit_pos(), column as i64));
                }
            }

            for block in 0..6 {
                let dc_table = if block < 4 { &DC_LUM } else { &DC_CHROMA };
                let component = if block < 4 { 0 } else { block - 3 };
                let coeffs =
                    decode_intra_block(&mut r, scale, dc_table, &mut dc_predictor[component])?;
                let residual = idct_residual(&coeffs);
                let (_, origin_x, origin_y) = block_origin(block, column, mb_row);
                let (plane, stride) = match block {
                    0..=3 => (&mut frame.y, width),
                    4 => (&mut frame.u, cw),
                    _ => (&mut frame.v, cw),
                };
                store_block(&residual, 128, plane, stride, origin_y * stride + origin_x);
            }
        }
    }

    if decoded_slices == 0 {
        return Err(DecodeError::new("no_slices", 0, 0));
    }
    Ok(frame)
}

/// 解一帧 P 图像：以 `reference` 为底做前向运动补偿 + 残差。
pub fn decode_p_frame(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    reference: &Frame,
) -> Result<Frame, DecodeError> {
    if !matches!(picture.picture_type, PictureType::P) {
        return Err(DecodeError::new("picture_type", 0, 0));
    }
    if sequence.mpeg2_stream() {
        return Err(DecodeError::new("mpeg2_unsupported", 0, 0));
    }
    if reference.width != sequence.width || reference.height != sequence.height {
        return Err(DecodeError::new("reference_size", 0, 0));
    }

    let width = sequence.width as usize;
    let height = sequence.height as usize;
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
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
    let fcode = picture.forward_f_code;
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
        // 运动矢量预测器：每个 slice 从 0 起；帧内 / 零矢量 / 跳过宏块都归零。
        let mut last_mv = (0i32, 0i32);
        let mut mb_index = (row * mb_w) as isize - 1;
        loop {
            let Some(increment) = decode_vlc(&mut r, &MB_ADDR_INCR) else {
                break;
            };
            let previous = mb_index;
            mb_index += increment as isize + 1;
            if mb_index < 0 || mb_index as usize >= total_mb {
                break;
            }
            // 跳过的宏块：已经是参考帧原位拷贝（帧初始化时克隆好了），
            // 只需把运动矢量预测器归零。
            if mb_index > previous + 1 {
                last_mv = (0, 0);
            }
            let column = mb_index as usize % mb_w;
            let mb_row = mb_index as usize / mb_w;

            // slice 末尾的填充位可能凑出一个无效的 mb_type —— 当作本 slice 结束。
            let Some(entry) = decode_p_mb_type(&mut r) else { break; };
            let mb = P_MB_TYPE[entry];

            if mb.quant {
                scale = r
                    .u(5)
                    .map_err(|_| DecodeError::new("mb_scale", r.bit_pos(), column as i64))?
                    as i32;
                if scale == 0 {
                    return Err(DecodeError::new("mb_scale_zero", r.bit_pos(), column as i64));
                }
            }

            if mb.intra {
                last_mv = (0, 0);
                for block in 0..6 {
                    let dc_table = if block < 4 { &DC_LUM } else { &DC_CHROMA };
                    let component = if block < 4 { 0 } else { block - 3 };
                    let coeffs = decode_intra_block(
                        &mut r,
                        scale,
                        dc_table,
                        &mut dc_predictor[component],
                    )?;
                    let residual = idct_residual(&coeffs);
                    let (_, origin_x, origin_y) = block_origin(block, column, mb_row);
                    let (plane, stride) = match block {
                        0..=3 => (&mut frame.y, width),
                        4 => (&mut frame.u, cw),
                        _ => (&mut frame.v, cw),
                    };
                    store_block(&residual, 128, plane, stride, origin_y * stride + origin_x);
                }
                continue;
            }

            // 帧间：先运动矢量，再 cbp，再各块残差。
            let (mut mv_x, mut mv_y) = (0i32, 0i32);
            match mb.motion {
                Motion::Forward => {
                    let decoded_x = decode_motion(&mut r, fcode, last_mv.0)
                        .ok_or_else(|| DecodeError::new("mv_x", r.bit_pos(), column as i64))?;
                    let decoded_y = decode_motion(&mut r, fcode, last_mv.1)
                        .ok_or_else(|| DecodeError::new("mv_y", r.bit_pos(), column as i64))?;
                    last_mv = (decoded_x, decoded_y);
                    mv_x = decoded_x;
                    mv_y = decoded_y;
                    if picture.full_pel_forward {
                        // full_pel 的矢量是整像素，换成半像素单位再补偿。
                        mv_x *= 2;
                        mv_y *= 2;
                    }
                }
                Motion::Zero | Motion::None => {
                    last_mv = (0, 0);
                }
            }
            // 帧间宏块之后，帧内 DC 预测器归零。
            dc_predictor = [0i32; 3];

            let cbp = if mb.coded {
                decode_vlc(&mut r, &MB_PATTERN)
                    .ok_or_else(|| DecodeError::new("cbp", r.bit_pos(), column as i64))?
                    as u32
            } else {
                0
            };

            for block in 0..6 {
                let coded = mb.coded && (cbp & (1 << (5 - block))) != 0;
                let (chroma, origin_x, origin_y) = block_origin(block, column, mb_row);
                let residual = if coded {
                    let coeffs = decode_inter_block(&mut r, scale)?;
                    idct_residual(&coeffs)
                } else {
                    [0i32; 64]
                };
                match block {
                    0..=3 => {
                        let pred = predict_block(
                            &reference.y, width, width, height, origin_x, origin_y, mv_x, mv_y,
                            chroma,
                        );
                        store_pred(
                            &pred,
                            &residual,
                            &mut frame.y,
                            width,
                            origin_y * width + origin_x,
                        );
                    }
                    4 => {
                        let pred = predict_block(
                            &reference.u, cw, cw, ch, origin_x, origin_y, mv_x, mv_y, chroma,
                        );
                        store_pred(&pred, &residual, &mut frame.u, cw, origin_y * cw + origin_x);
                    }
                    _ => {
                        let pred = predict_block(
                            &reference.v, cw, cw, ch, origin_x, origin_y, mv_x, mv_y, chroma,
                        );
                        store_pred(&pred, &residual, &mut frame.v, cw, origin_y * cw + origin_x);
                    }
                }
            }
        }
    }

    if decoded_slices == 0 {
        return Err(DecodeError::new("no_slices", 0, 0));
    }
    Ok(frame)
}

/// 按图像类型分派。P 帧必须给参考帧。
pub fn decode_frame(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    reference: Option<&Frame>,
) -> Result<Frame, DecodeError> {
    match picture.picture_type {
        PictureType::I => decode_i_frame(sequence, picture, data),
        PictureType::P => {
            let reference =
                reference.ok_or_else(|| DecodeError::new("missing_reference", 0, 0))?;
            decode_p_frame(sequence, picture, data, reference)
        }
        _ => Err(DecodeError::new("unsupported_picture_type", 0, 0)),
    }
}
