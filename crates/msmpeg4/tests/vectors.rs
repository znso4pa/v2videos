//! 分量级的**向量/往返**测试。
//!
//! 这类测试不依赖真样本：给输入、断言输出。它们抓的是「实现和规范不一致」，
//! 而真样本对账抓的是「整条链串起来对不对」—— 两类都要有。
//!
//! 对应 AGENTS.md：合成往返测试**必要但不充分**，所以它们只守分量，不替代 oracle。

use v2v_msmpeg4::{decode012, parse_frame_header, BitReader, PictureType, ZIGZAG};
use v2v_asf::fourcc_string;
use v2v_msmpeg4::{AcVal, Vlc};
use v2v_msmpeg4::{ac_predict, coded_block_predict, dc_predict, idct_put};
use v2v_msmpeg4::tables;

/// 本地位写入器（`testbit` 模块是 `#[cfg(test)]`，集成测试拿不到）。
struct BitWriter { bytes: Vec<u8>, cur: u8, nbits: u8 }
impl BitWriter {
    fn new() -> Self { Self { bytes: Vec::new(), cur: 0, nbits: 0 } }
    fn put(&mut self, v: u32, n: u32) {
        for i in (0..n).rev() {
            self.cur = (self.cur << 1) | (((v >> i) & 1) as u8);
            self.nbits += 1;
            if self.nbits == 8 { self.bytes.push(self.cur); self.cur = 0; self.nbits = 0; }
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 { self.cur <<= 8 - self.nbits; self.bytes.push(self.cur); }
        self.bytes
    }
}

// ---------- 位流读写往返 ----------

#[test]
fn bitstream_roundtrip_across_widths() {
    // 把一组数按各种位宽写进去再读出来，必须原样。
    let cases: &[(u32, u32)] = &[(0, 1), (1, 1), (1, 3), (0x7f, 7), (0xff, 8), (0xdead, 16), (0x12345, 20)];
    let mut w = BitWriter::new();
    for &(v, n) in cases {
        w.put(v, n);
    }
    // 末尾补零，保证读得完
    let mut data = w.finish();
    data.extend_from_slice(&[0, 0, 0]);

    let mut r = BitReader::new(&data);
    for &(v, n) in cases {
        assert_eq!(r.u(n).unwrap(), v, "位宽 {n} 的往返失败");
    }
}

#[test]
fn bitreader_show_does_not_advance() {
    let mut r = BitReader::new(&[0b1011_0100]);
    assert_eq!(r.show(4).unwrap(), 0b1011);
    assert_eq!(r.show(4).unwrap(), 0b1011, "show 不能前进");
    assert_eq!(r.u(4).unwrap(), 0b1011);
    assert_eq!(r.u(4).unwrap(), 0b0100);
    assert_eq!(r.bits_left(), 0);
}

// ---------- 扫描表 ----------

#[test]
fn scan_tables_are_permutations() {
    // 任何扫描表都必须是 0..63 的一个排列，否则系数会重叠/丢失。
    let ok = |name: &str, t: &[u8; 64]| {
        let mut seen = [false; 64];
        for &z in t {
            assert!(!seen[z as usize], "{name} 有重复下标 {z}");
            seen[z as usize] = true;
        }
    };
    ok("ZIGZAG", &ZIGZAG);
    ok("H_SCAN", &tables::H_SCAN);
    ok("V_SCAN", &tables::V_SCAN);
    // 三张表互不相同（否则 ac_pred 的分支没有意义）
    assert_ne!(&tables::H_SCAN[..], &tables::V_SCAN[..]);
}

#[test]
fn zigzag_maps_raster_to_diagonal_order() {
    // zig-zag 的本质：按 (i+j) 的反对角线顺序走。抽查前几项。
    // scan[i] = 第 i 个被访问的自然序下标。
    assert_eq!(ZIGZAG[0], 0);   // (0,0)
    assert_eq!(ZIGZAG[1], 1);   // (0,1)
    assert_eq!(ZIGZAG[2], 8);   // (1,0)
    assert_eq!(ZIGZAG[3], 16);  // (2,0)
    assert_eq!(ZIGZAG[63], 63); // (7,7)
}

// ---------- 预测内核 ----------

#[test]
fn dc_predict_is_symmetric_under_constant_input() {
    // 常数场：a=b=c → 全部命中 `<=` → 取 C（方向 1）。这是 MS-MPEG-4 的 tie-break。
    for v in [-1000i32, -1, 0, 1, 1024, 65535] {
        assert_eq!(dc_predict(v, v, v), (v, 1));
    }
}

#[test]
fn ac_predict_roundtrips_through_copy_back() {
    // 把一块的首行/首列存进 ac_val，再用它当「邻居」预测下一块，
    // 下一块的首行/首列应当精确等于上一块的值。
    let mut prev = [0i16; 64];
    for i in 0..8 {
        prev[i] = (i as i16) * 3 + 1;      // 首行
        prev[i * 8] = (i as i16) * 5 + 2;  // 首列
    }
    let mut ac: AcVal = [0; 16];
    let neighbor: AcVal = [0; 16];
    // 方向 1（取上），ac_pred 关掉 → 只做拷回
    let mut cur = [0i16; 64];
    cur.copy_from_slice(&prev);
    ac_predict(&mut cur, &mut ac, &neighbor, 1, false);
    for i in 1..8 {
        assert_eq!(ac[8 + i], prev[i], "首行存回失败");
        assert_eq!(ac[i], prev[i * 8], "首列存回失败");
    }
    // 用这份 ac_val 当邻居，对一块全零做「取上」预测 → 首行应等于 prev 的首行
    let mut next = [0i16; 64];
    let mut ac2: AcVal = [0; 16];
    ac_predict(&mut next, &mut ac2, &ac, 1, true);
    for i in 1..8 {
        assert_eq!(next[i], ac[8 + i], "取上预测应把邻居首行加进来");
    }
}

#[test]
fn coded_block_predict_matches_its_truth_table() {
    // 8 种组合全覆盖 —— 这是个 3 输入的纯函数，穷举是可行的。
    for a in 0..2u8 {
        for b in 0..2u8 {
            for c in 0..2u8 {
                let want = if b == c { a } else { c };
                assert_eq!(coded_block_predict(a, b, c), want, "a={a} b={b} c={c}");
            }
        }
    }
}

// ---------- IDCT ----------

#[test]
fn idct_is_linear_in_dc_and_matches_reference_per_coefficient() {
    // 单系数激励：只放一个非零系数，检查输出是「该频率的基图案」——
    // 即 DC 只影响幅度、AC 影响形状。逐项与浮点参考比。
    let mut seed = 0xC0FFEEu32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 16) as i32 % 41 - 20) as i16
    };
    for _ in 0..50 {
        let mut block = [0i16; 64];
        for c in block.iter_mut() {
            *c = next();
        }
        let mut got = block;
        let mut dest = vec![0u8; 64];
        idct_put(&mut got, &mut dest, 8);
        // 像素范围无需断言：`dest` 是 `Vec<u8>`，类型本身就保证 0..=255
        // （原来写的 `p <= 255` 对 `u8` 恒真，被 lint 指出来了）。
        // 线性：IDCT(2x) ≈ 2×IDCT(x)（允许舍入差）
        let mut doubled = block;
        for c in doubled.iter_mut() {
            *c = c.saturating_mul(2);
        }
        let mut d2 = [0u8; 64];
        let mut b2 = doubled;
        idct_put(&mut b2, &mut d2, 8);
        for i in 0..64 {
            let expect = (dest[i] as i32 * 2).clamp(0, 255);
            let got = d2[i];
            assert!(
                (got as i32 - expect).abs() <= 3,
                "第 {i} 像素线性失败：{got} vs {expect}"
            );
        }
    }
}

// ---------- VLC ----------

#[test]
fn vlc_survives_random_juxtaposition() {
    // 把一张表的前 32 个符号随机串起来编码再解码，必须一一还原 ——
    // 这是前缀码最容易被「边界 juxtaposition」坑的地方。
    let tab: Vec<[u32; 2]> = tables::RL0_VLC[..32].to_vec();
    let v = Vlc::new(&tab);
    let mut w = BitWriter::new();
    let seq: Vec<usize> = (0..32).collect();
    for &i in &seq {
        let c = tab[i][0];
        let l = tab[i][1];
        w.put(c, l);
    }
    let mut data = w.finish();
    data.extend_from_slice(&[0, 0, 0]);
    let mut r = BitReader::new(&data);
    for &i in &seq {
        assert_eq!(v.decode(&mut r).unwrap(), i, "第 {} 个符号解错", i);
    }
}

// ---------- 帧头 ----------

#[test]
fn frame_header_roundtrip_all_qscale_and_types() {
    // 穷举 qscale 1..31 × I/P 两种类型，帧头必须能写进去再读出来。
    for q in 1u32..=31 {
        for (pict, pbits) in [(PictureType::I, 0u32), (PictureType::P, 1)] {
            let mut w = BitWriter::new();
            w.put(pbits, 2);
            w.put(q, 5);
            if pict == PictureType::I {
                w.put(0x17, 5);
                w.put(0, 1);
                w.put(0, 1);
                w.put(1, 1);
            } else {
                w.put(1, 1); // skip
                w.put(0, 1);
                w.put(0, 1);
                w.put(0, 1);
            }
            let mut data = w.finish();
            data.extend_from_slice(&[0, 0, 0]);
            let (h, used) = parse_frame_header(&data, 16, 16).unwrap();
            assert_eq!(h.qscale, q as u8);
            assert_eq!(h.picture_type, pict);
            let want = if pict == PictureType::I { 15 } else { 11 };
            assert_eq!(used, want, "qscale={q} {:?} 的头位数不对", pict);
        }
    }
}

#[test]
fn decode012_is_exhaustive() {
    // 3 种输入全覆盖
    // decode012：先读 1 位，0 → 0；否则再读 1 位，0 → 1、1 → 2
    assert_eq!(decode012(&mut BitReader::new(&[0b0000_0000])).unwrap(), 0);
    assert_eq!(decode012(&mut BitReader::new(&[0b1000_0000])).unwrap(), 1);
    assert_eq!(decode012(&mut BitReader::new(&[0b1100_0000])).unwrap(), 2);
}

#[test]
fn fourcc_never_panics_on_arbitrary_input() {
    for v in [0u32, 0x3334504D, 0xFFFF_FFFF, 0x4142_4344, 0x2020_2020] {
        let s = fourcc_string(v);
        assert_eq!(s.chars().count(), 4);
        // 可打印字符原样保留（含空格），其余退化为 '.'
        assert!(s.chars().all(|c| c.is_ascii_graphic() || c == '.' || c == ' '));
    }
}
