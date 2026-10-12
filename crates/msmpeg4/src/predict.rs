//! 宏块层的三个**预测**内核 —— 全是「看着简单、写错很难发现」的地方。
//!
//! 三个内核，按 MS-MPEG-4 v3（H.263 / MPEG-4 Part 2 语法）的定义实现：
//! - **DC 预测**：从邻居的 DC 值取预测（`dc_predict`）
//! - **CBP 预测**：编码块图从邻居推（`coded_block_predict`）
//! - **AC 预测**：把邻居块的首行/首列加到本块上（`ac_predict`）
//!
//! ## 关于扫描序
//!
//! 这里的系数一律按**自然序**（`block[row*8+col]`）存取，不绕扫描置换表 ——
//! 调用方负责把扫描序解出来的系数摆回自然序。换 IDCT 实现时要重新确认这条假设。

/// MP43 的 DC 预测。返回 `(预测值, 方向)`，方向 `0 = 取左(A)`、`1 = 取上(C)`。
///
/// 邻块（缩放后的 DC 值）：
/// ```text
///   B C
///   A X   ← X 是当前块
/// ```
///
/// ⚠️ **这里的比较是 `<=`，标准 MPEG-4 的对应判定是 `<`** —— 这个差别会让部分宏块的
/// 预测方向反转。判据与 ffmpeg 交叉验证过（`tests/` 里逐像素对账）。
pub fn dc_predict(a: i32, b: i32, c: i32) -> (i32, usize) {
    if (a - b).abs() <= (b - c).abs() {
        (c, 1)
    } else {
        (a, 0)
    }
}

/// 已编码块图（coded block pattern）的预测：`b == c` 取 `a`，否则取 `c`。
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_predict_tie_goes_to_top() {
        // 相等时取 C（方向 1）—— 这是 MS-MPEG-4 的 `<=`，不是 MPEG-4 的 `<`。
        assert_eq!(dc_predict(10, 10, 10), (10, 1));
        // |a-b| = 0 <= |b-c| = 5 → 取 C
        assert_eq!(dc_predict(0, 0, 5), (5, 1));
        // |a-b| = 5 > |b-c| = 0 → 取 A
        assert_eq!(dc_predict(5, 0, 0), (5, 0));
        // |a-b| = 2 <= |b-c| = 3 → 取 C
        assert_eq!(dc_predict(8, 10, 7), (7, 1));
    }

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
