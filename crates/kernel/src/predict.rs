//! 宏块层的预测内核 —— **与具体编解码器无关**的那一半。
//!
//! 收：AC 预测（把邻居块的首行/首列加到本块上）、编码块图的预测、以及 DC 反量化的
//! 那一步缩放。这几件事在 H.263 / MPEG-4 系里是同一份，`v2v-msmpeg4` 与 `v2v-mp4v`
//! 都调它。
//!
//! **不收**：
//! - **DC 预测的方向判定**（`dc_predict`）—— 它按 `Version` 分支（MP43 用 `<=`、
//!   WMV1/2 用 `<`），留在 `v2v-msmpeg4`；
//! - 运动矢量的预测（`pred_motion` / `wmv2_pred_motion`）—— 那是语法，各家不同，
//!   在 `v2v-kernel::mc` 里只有中值预测 [`mid_pred`](crate::mc::mid_pred) 这一块
//!   纯算术。
//!
//! ## 关于扫描序
//!
//! 这里的系数一律按**自然序**（`block[row*8+col]`）存取，不绕扫描置换表 ——
//! 调用方负责把扫描序解出来的系数摆回自然序。换 IDCT 实现时要重新确认这条假设。

pub fn coded_block_predict(a: u8, b: u8, c: u8) -> u8 {
    if b == c {
        a
    } else {
        c
    }
}

/// 一个块留给邻居用的 16 个值：`[0..8]` 是**首列**（给右边用），`[8..16]` 是**首行**（给下面用）。
pub type AcVal = [i16; 16];

/// MPEG-4 的 AC 预测：把邻居块的首行/首列加到本块上，再把本块的首列/首行存回去。
///
/// - `dir == 0`：从**左**邻居预测，加的是邻居的**首列**，落到本块的首列。
/// - `dir == 1`：从**上**邻居预测，加的是邻居的**首行**，落到本块的首行。
///
/// 只在 `ac_pred` 为真时相加；**存回去总是做**（否则下一个邻居拿到的就是没更新的值）。
pub fn ac_predict(block: &mut [i16; 64], ac: &mut AcVal, neighbor: &AcVal, dir: usize, ac_pred: bool) {
    if ac_pred {
        if dir == 0 {
            for i in 1..8 {
                block[i * 8] = block[i * 8].wrapping_add(neighbor[i]);
            }
        } else {
            for i in 1..8 {
                block[i] = block[i].wrapping_add(neighbor[8 + i]);
            }
        }
    }

    // 首列（给右边）
    for i in 1..8 {
        ac[i] = block[i * 8];
    }
    // 首行（给下面）
    for i in 1..8 {
        ac[8 + i] = block[i];
    }
}


/// DC 预测器里的那一步「除以 scale」（`ff_msmpeg4_pred_dc` 的后半段）。
///
/// ⚠️ 它**分两条路**，而且结果对负数不一样：
/// - `scale == 8`（低 qscale 那一档）：`(a + 4) / 8` —— C 的 `/`，**向零取整**；
/// - 其余：`FASTDIV(a + scale/2, scale)` —— 那是 `((uint64_t)a * ff_inverse[b]) >> 32`，
///   `a` 先**符号扩展**到 64 位，于是对负数等价于**向下取整**。
///
/// 差 1 的 DC 就是「整块偏 1」，而且只在负 DC 上冒出来 —— 低 qscale 的样本根本碰不到。
/// `ff_inverse[b] = ceil(2^32 / b)`（经典的「除以常数」魔数）。
#[inline]
fn fastdiv(v: i32, scale: i32) -> i32 {
    let inv = ((1u64 << 32) + scale as u64 - 1) / scale as u64;
    ((v as i64 as u64).wrapping_mul(inv) >> 32) as u32 as i32
}
#[inline]
pub fn dc_unscale(v: i32, scale: i32) -> i32 {
    if scale == 8 {
        // 常见档位走这条：C 的 `/`，**向零取整**。
        (v + 4) / 8
    } else {
        // 其余走 `FASTDIV(a + scale/2, scale)` —— 见上面的说明。
        fastdiv(v + (scale >> 1), scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coded_block_predict_prefers_a_when_b_equals_c() {
        assert_eq!(coded_block_predict(1, 0, 0), 1);
        assert_eq!(coded_block_predict(0, 1, 0), 0); // b != c → c
        assert_eq!(coded_block_predict(1, 1, 1), 1);
        assert_eq!(coded_block_predict(1, 0, 1), 1); // b == c → a
    }

    #[test]
    fn ac_predict_left_adds_the_neighbors_first_column() {
        let mut block = [0i16; 64];
        let mut neighbor: AcVal = [0; 16];
        for i in 0..8 {
            neighbor[i] = (i as i16) * 10; // 首列
        }
        let mut ac: AcVal = [0; 16];

        ac_predict(&mut block, &mut ac, &neighbor, 0, true);

        // 本块首列（i*8）拿到邻居首列的值；i 从 1 起，所以 [0] 不动。
        assert_eq!(block[0], 0);
        for i in 1..8 {
            assert_eq!(block[i * 8], (i as i16) * 10, "第 {i} 行首列");
        }
        // 首行没被碰
        for i in 0..8 {
            assert_eq!(block[i], 0);
        }
        // 存回去：ac 的首列 = 本块首列
        for i in 1..8 {
            assert_eq!(ac[i], block[i * 8]);
        }
    }

    #[test]
    fn ac_predict_top_adds_the_neighbors_first_row() {
        let mut block = [0i16; 64];
        let mut neighbor: AcVal = [0; 16];
        for i in 0..8 {
            neighbor[8 + i] = (i as i16) + 1; // 首行
        }
        let mut ac: AcVal = [0; 16];

        ac_predict(&mut block, &mut ac, &neighbor, 1, true);

        assert_eq!(block[0], 0);
        for i in 1..8 {
            assert_eq!(block[i], (i as i16) + 1, "第 {i} 列首行");
        }
        for i in 0..8 {
            assert_eq!(ac[8 + i], block[i]);
        }
    }

    #[test]
    fn ac_predict_still_copies_back_when_prediction_is_off() {
        let mut block = [0i16; 64];
        block[8] = 7; // 首列第 1 项
        block[1] = 9; // 首行第 1 项
        let neighbor: AcVal = [0; 16];
        let mut ac: AcVal = [0; 16];

        ac_predict(&mut block, &mut ac, &neighbor, 0, false);

        assert_eq!(ac[1], 7, "不做预测也要把首列存回去");
        assert_eq!(ac[8 + 1], 9, "不做预测也要把首行存回去");
    }
}
