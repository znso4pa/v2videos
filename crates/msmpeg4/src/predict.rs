//! **DC 预测** —— 唯一按版本分支的那个预测内核，所以留在本 crate。
//!
//! CBP 预测、AC 预测与 DC 反量化缩放都是格式无关的，在 `v2v-kernel::predict`。
//!
//! 这里只剩 `dc_predict`：它的比较运算符**按版本不同**（MP43 用 `<=`、WMV1/2 用 `<`），
//! 平手方向是反的 —— 见函数上的说明，那是本仓最容易写错的一处。

use crate::version::Version;

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
/// ⚠️ **比较运算符按版本不同**，而且 ffmpeg 自己用大写字标了
/// 「they did not choose the same test as MPEG-4. **This is very important !**」：
///
/// - **V1 / V2 / V3**：`abs(a−b) <= abs(b−c)` → 平手取 `c`、`dir = 1`
/// - **WMV1 / WMV2**：`abs(a−b) <  abs(b−c)` → 平手取 `a`、`dir = 0`
///
/// 平手在 **P 帧里的帧内宏块**上**必然发生**：邻居是帧间宏块、它们的 DC 网格槽
/// 被清成 1024，于是 `a == b == c`。而 `dir` 还会**决定 AC 预测用哪张扫描表** ——
/// 取错就是整块系数落错位置（症状：位流位置与符号个数全对、只有值不对）。
pub fn dc_predict(a: i32, b: i32, c: i32, version: Version) -> (i32, usize) {
    let (ab, bc) = ((a - b).abs(), (b - c).abs());
    let take_c = if version.is_wmv() { ab < bc } else { ab <= bc };
    if take_c {
        (c, 1)
    } else {
        (a, 0)
    }
}

/// 已编码块图（coded block pattern）的预测：`b == c` 取 `a`，否则取 `c`。
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_predict_tie_goes_to_top_for_v1_v2_v3() {
        // V1/V2/V3 用 `<=`：相等时取 C（方向 1）。
        assert_eq!(dc_predict(10, 10, 10, Version::V3), (10, 1));
        // |a-b| = 0 <= |b-c| = 5 → 取 C
        assert_eq!(dc_predict(0, 0, 5, Version::V3), (5, 1));
        // |a-b| = 5 > |b-c| = 0 → 取 A
        assert_eq!(dc_predict(5, 0, 0, Version::V3), (5, 0));
        // |a-b| = 2 <= |b-c| = 3 → 取 C
        assert_eq!(dc_predict(8, 10, 7, Version::V3), (7, 1));
    }

    /// ⚠️ **WMV1/WMV2 的平手方向是反的** —— 用 `<` 而不是 `<=`。
    ///
    /// 这条曾经漏掉，症状很阴：位流位置、符号个数、宏块结构全对，只有**值**不对。
    /// 因为平手在 **P 帧里的帧内宏块**上必然发生（邻居是帧间宏块、DC 槽被清成
    /// 1024 → `a == b == c`），而方向还决定 AC 预测用哪张扫描表 —— 取错就整块落错位。
    #[test]
    fn dc_predict_tie_goes_to_left_for_wmv1_wmv2() {
        // ffmpeg 原话：「they did not choose the same test as MPEG-4.
        // This is very important !」
        assert_eq!(dc_predict(10, 10, 10, Version::Wmv1), (10, 0));
        assert_eq!(dc_predict(10, 10, 10, Version::Wmv2), (10, 0));
        // 非平手时两个版本一致。
        assert_eq!(dc_predict(0, 0, 5, Version::Wmv1), (5, 1));
        assert_eq!(dc_predict(5, 0, 0, Version::Wmv1), (5, 0));
        assert_eq!(dc_predict(8, 10, 7, Version::Wmv1), (7, 1));
    }
}
