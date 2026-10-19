//! MS-MPEG-4 **v1 / v2** 独有的那点东西 —— 别的版本不走这里。
//!
//! 两个版本与 V3（MP43）的差别集中在三处，而且每一处都「差一个位整帧就错」：
//!
//! | | V1 / V2 | V3（MP43） |
//! |---|---|---|
//! | DC 的 VLC | H.263 那套 512 项，符号值以 **256 为中心** | `DC[dc_table_index][...]`，120 项 + `DC_MAX` 转义 |
//! | DC 缩放 | `ff_mpeg1_dc_scale_table`（**恒 8**） | `ff_mpeg4_y_dc_scale_table` |
//! | 运动矢量 | `H263_MV` + `f_code = 1` + **±64 回绕** | `MV0` / `MV1`，分片位 + 另一种回绕 |
//!
//! 这里只放「表」这一层；宏块层的分支在 `frame.rs`（它需要 `Decoder` 的状态）。

use crate::vlc::Vlc;

/// `ff_mpeg4_DCtab_lum`：`size` → `(code, len)`。
///
/// 与 `v2v-mp4v::tables::DC_LUM` 是同一份数据（格式定义，不是谁的私有表）。
/// 两个 crate 各存一份是因为**依赖方向**：`mp4v` 依赖 `msmpeg4`，不能反向引用。
static MPEG4_DC_LUM: [[u8; 2]; 13] = [
    [3, 3],
    [3, 2],
    [2, 2],
    [2, 3],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
    [1, 7],
    [1, 8],
    [1, 9],
    [1, 10],
    [1, 11],
];

/// `ff_mpeg4_DCtab_chrom`。
static MPEG4_DC_CHROM: [[u8; 2]; 13] = [
    [3, 2],
    [2, 2],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
    [1, 7],
    [1, 8],
    [1, 9],
    [1, 10],
    [1, 11],
    [1, 12],
];

/// 造 `v2_dc_lum_vlc` / `v2_dc_chroma_vlc`（返回 `[亮度, 色度]`）。
///
/// 这两张表**不是静态数据**：上游 `init_h263_dc_for_msmpeg4()` 现场从 MPEG-4 的
/// `DCtab` 算 512 项（`level ∈ -256..=255`，**下标 = level + 256**）。所以解出来的
/// 符号值必须 **减 256** 才是 level —— 漏了这一步就是「整帧亮度偏一大截」。
///
/// 与 H.263 原版的区别是**码字取反**（`code ^= (1 << len) - 1`，上游注释写着
/// "M$ does not like compatibility"）；负数用「反码」而不是「符号位 + 绝对值」：
/// `l = (-level) ^ ((1 << size) - 1)`。
pub fn dc_tables() -> [Vlc; 2] {
    [build(&MPEG4_DC_LUM), build(&MPEG4_DC_CHROM)]
}

fn build(tab: &[[u8; 2]; 13]) -> Vlc {
    let mut pairs = vec![[0u32; 2]; 512];
    for level in -256i32..256 {
        let size = 32 - level.unsigned_abs().leading_zeros();
        let mag = if level < 0 {
            (-level) ^ ((1 << size) - 1)
        } else {
            level
        };
        let (mut code, mut len) = (u32::from(tab[size as usize][0]), u32::from(tab[size as usize][1]));
        // 取反：M$ 的 v1/v2 与 H.263 唯一的「不兼容」之处。
        code ^= (1 << len) - 1;
        if size > 0 {
            code = (code << size) | mag as u32;
            len += size;
            if size > 8 {
                code = (code << 1) | 1;
                len += 1;
            }
        }
        pairs[(level + 256) as usize] = [code, len];
    }
    Vlc::new(&pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 512 项全都有码、且**两两不冲突**（真前缀码）。
    ///
    /// 手算过三个：`level = 0` → `0b100`（3 位）、`level = 1` → `0b001`、
    /// `level = -1` → `0b000` —— 与上游 `init_h263_dc_for_msmpeg4` 一致
    /// （取反那一步是关键：不取反的话 `level = 0` 会是 `0b011`）。
    #[test]
    fn tables_are_complete_prefix_codes() {
        for tab in dc_tables() {
            let mut codes: Vec<(u32, u32, i32)> = Vec::new();
            for level in -256i32..256 {
                let (code, len) = tab
                    .code_len((level + 256) as usize)
                    .expect("每个 level 都该有码");
                assert!((1..=21).contains(&len), "level={level} len={len}");
                codes.push((code, len, level));
            }
            for (i, &(ca, la, va)) in codes.iter().enumerate() {
                for (j, &(cb, lb, vb)) in codes.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    if la <= lb {
                        assert_ne!(cb >> (lb - la), ca, "level {va} 与 {vb} 前缀冲突");
                    }
                }
            }
        }
    }
}
