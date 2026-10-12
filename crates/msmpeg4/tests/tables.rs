//! 生成表（`tables.rs`）的自检。
//!
//! 这些表是从 FFmpeg 源码机械抽取的，最怕**抄错一个数字** —— 抄错了不会立刻炸，
//! 只会让个别块解错，很难定位。所以这里做三件事：
//!
//! 1. **条目数**对得上（尺寸错了立刻红）；
//! 2. **前缀无关性**：没有哪个码是另一个码的前缀（真前缀码的必要条件）；
//! 3. **Kraft 和 == 1**：完整前缀码的充要条件 —— 这条能抓住「多一个/少一个码」
//!    以及「某个 len 写错」这类错误。
//!
//! 再对几个已知项做**点值抽查**，挡住整体错位。

use v2v_msmpeg4::tables as t;

/// 返回 `(是否前缀无关, Kraft 和是否为 1)`。
fn check_prefix_code(table: &[[u32; 2]]) -> (bool, bool) {
    let used: Vec<(u32, u8)> = table
        .iter()
        .filter(|e| e[1] != 0)
        .map(|e| (e[0], e[1] as u8))
        .collect();

    // 前缀无关：把每个码按 (code, len) 归一化后两两比较。
    let mut prefix_free = true;
    'outer: for (i, &(ci, li)) in used.iter().enumerate() {
        for &(cj, lj) in used.iter().skip(i + 1) {
            let (short, ls, long, ll) = if li <= lj {
                (ci, li, cj, lj)
            } else {
                (cj, lj, ci, li)
            };
            if (long >> (ll - ls)) == short {
                prefix_free = false;
                break 'outer;
            }
        }
    }

    // Kraft：Σ 2^-len == 1。用整数避免浮点误差。
    let max = used.iter().map(|&(_, l)| l).max().unwrap_or(0) as u32;
    let sum: u64 = used.iter().map(|&(_, l)| 1u64 << (max - l as u32)).sum();
    let kraft_ok = sum == (1u64 << max);

    (prefix_free, kraft_ok)
}

/// MV 表的 `(code, len)` 是两张分开的数组，拼成 `[[u32; 2]]` 好复用同一套检查。
fn mv_pairs(code: &[u16], len: &[u8]) -> Vec<[u32; 2]> {
    assert_eq!(code.len(), len.len());
    code.iter().zip(len).map(|(&c, &l)| [c as u32, l as u32]).collect()
}

fn assert_vlc(name: &str, table: &[[u32; 2]], expect_entries: usize) {
    assert_eq!(table.len(), expect_entries, "{name} 条目数不对");
    let (pf, kraft) = check_prefix_code(table);
    assert!(pf, "{name} 不是前缀无关码");
    assert!(kraft, "{name} 的 Kraft 和 != 1（码不完整或长度写错）");
}

#[test]
fn table_sizes_and_prefix_codes_are_valid() {
    assert_vlc("MB_NON_INTRA", &t::MB_NON_INTRA, 128);
    assert_vlc("MB_INTRA", &t::MB_INTRA, 64);
    assert_vlc("INTER_INTRA", &t::INTER_INTRA, 4);
    assert_vlc("RL0_VLC", &t::RL0_VLC, 133);
    assert_vlc("RL1_VLC", &t::RL1_VLC, 149);
    assert_vlc("RL2_VLC", &t::RL2_VLC, 186);
    assert_vlc("RL4_VLC", &t::RL4_VLC, 169);

    assert_eq!(t::RL0_RUN.len(), 132);
    assert_eq!(t::RL0_LEVEL.len(), 132);
    assert_eq!(t::RL4_RUN.len(), 168);
    // 运动矢量表（每张 1100 个码 = 1099 个真值 + 1 个 escape）。
    // ⚠️ 这两张表**样本走不到第 0 张**：FFmpeg 的编码器把 `mv_table_index` 硬编码成 1
    // （`msmpeg4enc.c:234`），所以表 0 只能靠这里的自检兜底。
    assert_eq!(t::MV0_CODE.len(), 1100);
    assert_eq!(t::MV0_LEN.len(), 1100);
    assert_eq!(t::MV1_CODE.len(), 1100);
    assert_eq!(t::MV1_LEN.len(), 1100);
    assert_eq!(t::MV0_X.len(), 1099);
    assert_eq!(t::MV0_Y.len(), 1099);
    assert_eq!(t::MV1_X.len(), 1099);
    assert_eq!(t::MV1_Y.len(), 1099);
    assert_vlc("MV0", &mv_pairs(&t::MV0_CODE, &t::MV0_LEN), 1100);
    assert_vlc("MV1", &mv_pairs(&t::MV1_CODE, &t::MV1_LEN), 1100);
    // 偏移值必须落在 `[0, 63]`（解码时要减 32 再按 64 回绕）
    assert!(t::MV0_X.iter().chain(&t::MV0_Y).all(|&v| v < 64));
    assert!(t::MV1_X.iter().chain(&t::MV1_Y).all(|&v| v < 64));
    assert_eq!(t::SCAN.len(), 4);
    assert!(t::SCAN.iter().all(|r| r.len() == 64));
    assert_eq!(t::Y_DC_SCALE.len(), 32);
}

#[test]
fn spot_checks_match_the_source() {
    // 取自 FFmpeg 的 `msmpeg4data.c` / `msmpeg4_vc1_data.c` 头几项 ——
    // 挡住「整体错位一格」这类错误。
    assert_eq!(t::MB_NON_INTRA[0], [0x40, 7]);
    assert_eq!(t::MB_NON_INTRA[1], [0x13c9, 13]);
    assert_eq!(t::MB_INTRA[0], [0x1, 1]);
    assert_eq!(t::MB_INTRA[1], [0x17, 6]);
    assert_eq!(t::RL0_VLC[0], [0x1, 2]);
    assert_eq!(t::RL0_VLC[1], [0x6, 3]);
    assert_eq!(t::INTER_INTRA[0], [0, 1]);
    assert_eq!(t::INTER_INTRA[1], [2, 2]);
    // WMV1 的扫描顺序**不是**标准 zig-zag —— 实测第一行是 0,8,1,2,9,16,24,17。
    assert_eq!(&t::SCAN[0][..8], &[0, 8, 1, 2, 9, 16, 24, 17]);
    // MV 表头几项（`msmpeg4data.c` 的 table0_mv_code / _bits / _mvx / _mvy）
    assert_eq!(&t::MV0_CODE[..4], &[0x0001, 0x0003, 0x0005, 0x0007]);
    assert_eq!(&t::MV0_LEN[..4], &[1, 4, 4, 4]);
    assert_eq!(&t::MV0_X[..4], &[32, 32, 31, 32]);
    assert_eq!(&t::MV0_Y[..4], &[32, 31, 32, 33]);
}

#[test]
fn every_generated_table_decodes_to_its_own_index() {
    // MV 表是两张分开的数组，先拼出来
    let mv0 = mv_pairs(&t::MV0_CODE, &t::MV0_LEN);
    let mv1 = mv_pairs(&t::MV1_CODE, &t::MV1_LEN);
    // 用解码器把每个码喂回去，必须得到它自己的下标 —— 证明 Vlc 建树与表一致。
    for (name, table) in [
        ("MB_NON_INTRA", &t::MB_NON_INTRA[..]),
        ("MB_INTRA", &t::MB_INTRA[..]),
        ("RL0_VLC", &t::RL0_VLC[..]),
        ("RL1_VLC", &t::RL1_VLC[..]),
        ("RL2_VLC", &t::RL2_VLC[..]),
        ("RL4_VLC", &t::RL4_VLC[..]),
        ("MV0", &mv0[..]),
        ("MV1", &mv1[..]),
    ] {
        let v = v2v_msmpeg4::Vlc::new(table);
        for (i, &[code, len]) in table.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let nbytes = (len as usize).div_ceil(8) + 1;
            let mut bytes = vec![0u8; nbytes];
            for b in 0..len {
                if (code >> (len - 1 - b)) & 1 == 1 {
                    bytes[(b / 8) as usize] |= 1 << (7 - (b % 8));
                }
            }
            let mut r = v2v_msmpeg4::BitReader::new(&bytes);
            assert_eq!(
                v.decode(&mut r).unwrap(),
                i,
                "{name} 第 {i} 项（code={code:#x} len={len}）解码得到别的东西"
            );
        }
    }
}
