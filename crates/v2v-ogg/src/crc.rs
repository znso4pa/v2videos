//! Ogg 页的 CRC-32。
//!
//! **这个 CRC 和 zlib 那个不是一回事**，两者互换会静默算出别的数：
//!
//! | | Ogg 页校验 | zlib / PNG / Ethernet |
//! |---|---|---|
//! | 多项式 | `0x04C11DB7` | `0x04C11DB7`（同一个） |
//! | 位序 | **不反射**（MSB first） | 反射（LSB first） |
//! | 初值 | `0` | `0xFFFFFFFF` |
//! | 末尾异或 | **无** | `0xFFFFFFFF` |
//!
//! 校验范围是**整页**：27 字节页头 + 段表 + 页体，计算时把页头里
//! `crc` 那 4 个字节当 0（规范写法是「先清零再算」，不是「跳过」——
//! 少算 4 个字节的结果不一样）。
//!
//! 表是 `const fn` 在编译期算出来的，运行时零初始化。

/// 生成多项式（不含最高位）。
const POLY: u32 = 0x04C1_1DB7;

const fn make_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = (i as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ POLY
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static TABLE: [u32; 256] = make_table();

/// 增量计算用：把 `data` 续到已有的 `crc` 后面。
pub fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    let mut c = crc;
    for &b in data {
        let idx = ((c >> 24) as u8 ^ b) as usize;
        c = (c << 8) ^ TABLE[idx];
    }
    c
}

/// 从头算一段数据的 CRC。
pub fn crc32(data: &[u8]) -> u32 {
    crc32_update(0, data)
}

/// 算**一整页**的 CRC：`page` 是页的原始字节，`crc_offset` 是 `crc` 字段在其中的偏移。
///
/// 之所以不要求调用方先把字段清零再传进来：那是「先破坏再恢复」，
/// 中间任何一步出错都会留下一个被改过的页。这里按规范在计算时把那 4 字节当 0。
pub fn crc32_page(page: &[u8], crc_offset: usize) -> u32 {
    debug_assert!(crc_offset + 4 <= page.len());
    let head = crc32_update(0, &page[..crc_offset]);
    let head = crc32_update(head, &[0, 0, 0, 0]);
    crc32_update(head, &page[crc_offset + 4..])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一条独立算出来的基准：`"OggS"` 这个四字节串在本多项式下的值
    /// （另写一份 Python 逐位实现算出来对过的，不是从本模块抄的）。
    ///
    /// 顺手记下「为什么不能直接换 `crc32fast`」：zlib 口径算同一个串得
    /// `0x7775B7FC`，本模块是 `0x5FB0A94F` —— 两个数不一样，正是这条测试存在的理由。
    #[test]
    fn msb_first_polynomial_is_not_zlib() {
        assert_eq!(crc32(b"OggS"), 0x5FB0_A94F);
    }

    #[test]
    fn empty_input_keeps_the_initial_value() {
        assert_eq!(crc32(b""), 0);
    }

    /// 增量算和一次算完必须一样 —— 页是分「头 / 段表 / 体」三段拼起来算的，
    /// 这条保证拼接点不出错。
    #[test]
    fn incremental_matches_one_shot() {
        let data: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        let mut inc = 0;
        for chunk in data.chunks(7) {
            inc = crc32_update(inc, chunk);
        }
        assert_eq!(inc, crc32(&data));
    }

    #[test]
    fn page_crc_treats_the_field_as_zero() {
        // 一个假的「页」：前 4 字节随便，中间 4 字节是 crc 字段，后面随便。
        let mut page = vec![1u8, 2, 3, 4, 0xAA, 0xBB, 0xCC, 0xDD, 5, 6, 7];
        let expected = crc32_page(&page, 4);
        // 把那 4 字节改成别的值，结果不该变。
        page[4..8].copy_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        assert_eq!(crc32_page(&page, 4), expected);
        // 改成全 0 再整段算，应当一致。
        page[4..8].copy_from_slice(&[0, 0, 0, 0]);
        assert_eq!(crc32(&page), expected);
    }
}
