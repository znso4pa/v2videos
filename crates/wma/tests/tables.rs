//! `tables.rs` 的自检 —— 生成器已经把过关（Kraft 和），这里再在 Rust 侧钉一遍，
//! 免得生成器或表文件被改动后没人发现。

use v2v_wma::tables::*;

fn kraft(lens: &[u8]) -> f64 {
    lens.iter().map(|&n| 2f64.powi(-(n as i32))).sum()
}

#[test]
fn coefficient_tables_are_complete_prefix_codes() {
    // 6 张表的项数（对着 wmadata.h 的数组长度）
    let tables: [(&str, &[(u32, u8)], usize); 6] = [
        ("coef0", &COEF0_CODES, 666),
        ("coef1", &COEF1_CODES, 555),
        ("coef2", &COEF2_CODES, 1336),
        ("coef3", &COEF3_CODES, 1072),
        ("coef4", &COEF4_CODES, 476),
        ("coef5", &COEF5_CODES, 435),
    ];
    for (name, t, want) in tables {
        assert_eq!(t.len(), want, "{name} 项数");
        let lens: Vec<u8> = t.iter().map(|&(_, l)| l).collect();
        let k = kraft(&lens);
        assert!(
            (k - 1.0).abs() < 1e-9,
            "{name} 的 Kraft 和 = {k}（前缀码必须 = 1）"
        );
        assert!(lens.iter().all(|&l| l > 0 && l <= 32), "{name} 有非法码长");
    }
    assert_eq!(COEF0_LEVELS.len(), 60);
    assert_eq!(COEF1_LEVELS.len(), 40);
    assert_eq!(COEF2_LEVELS.len(), 340);
    assert_eq!(COEF3_LEVELS.len(), 180);
    assert_eq!(COEF4_LEVELS.len(), 70);
    assert_eq!(COEF5_LEVELS.len(), 40);
}

#[test]
fn hgain_is_a_prefix_code() {
    let lens: Vec<u8> = HGAIN_TAB.iter().map(|&(_, l)| l).collect();
    assert_eq!(lens.len(), 37);
    let k = kraft(&lens);
    assert!((k - 1.0).abs() < 1e-9, "hgain 的 Kraft 和 = {k}");
}

#[test]
fn lsp_codebook_short_rows_are_zero_padded() {
    // 源头是 C 的 `[10][16]`，**第 0 行与第 8 行起只写了 8 个值**（其余补 0），
    // 而 `decode_exp_lsp` 对 i==0 与 i>=8 只取 3 位（0..7）—— 正好只用前 8 列。
    // 这条断言把这个「短行」事实钉住：补零补错了，LSP 解出来就是错的。
    // 源头每行的值个数：`[8, 16, 16, 16, 16, 16, 16, 16, 8, 8]`
    assert_eq!(LSP_CODEBOOK.len(), 10);
    assert_eq!(LSP_CODEBOOK[0][0], 1.98732877, "第 0 行首值");
    for i in [0usize, 8, 9] {
        for k in 8..16 {
            assert_eq!(LSP_CODEBOOK[i][k], 0.0, "第 {i} 行第 {k} 列应为补零");
        }
    }
    // 第 1..7 行是满 16 个值
    for i in 1..8 {
        assert_ne!(LSP_CODEBOOK[i][15], 0.0, "第 {i} 行第 15 列不该是 0");
    }
}

#[test]
fn critical_freqs_match_the_source() {
    assert_eq!(CRITICAL_FREQS.len(), 25);
    assert_eq!(CRITICAL_FREQS[0], 100);
    assert_eq!(CRITICAL_FREQS[24], 24500);
    // 单调递增
    for w in CRITICAL_FREQS.windows(2) {
        assert!(w[0] < w[1], "critical freq 应递增");
    }
}
