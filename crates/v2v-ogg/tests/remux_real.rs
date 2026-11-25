//! 真样本**重封装**：`PageReader` → `OggMuxer::write_page` → 与原文件逐字节比。
//!
//! ## 为什么判据是「逐字节相同」而不是「能播」
//!
//! 页写入器只有两件事可能错：**段表**（包边界）和 **CRC**。两者都不会让文件
//! 「打不开」—— 段表错了播放器只是把两个包粘成一个（听着像爆音），CRC 错了
//! 绝大多数实现**默认不校验**（我们的 `PageReader` 默认也不校验，得显式打开）。
//! 所以「能播」什么也证明不了。
//!
//! 原样搬页的时候，写出来的字节**必须**与原文件一模一样：页头字段、段表、页体、
//! CRC 全都在我们手里重算一遍。任何一处算错，这个断言就红。
//!
//! 这也顺带证明了**我们的 CRC 算法**是对的（CRC 是唯一一处「自己算、自己验」
//! 会同时错的环节 —— 但这里比对的是**别人写的** CRC，所以它是独立 oracle）。
//!
//! 需要 `V2V_OGG_SAMPLE`；不给就跳过。

use std::path::Path;

use v2v_ogg::{OggMuxer, PageReader};

fn sample_path() -> Option<std::path::PathBuf> {
    let p = std::env::var_os("V2V_OGG_SAMPLE")?;
    let p = std::path::PathBuf::from(p);
    if !p.is_file() {
        eprintln!("SKIP: V2V_OGG_SAMPLE 指向的文件不存在");
        return None;
    }
    Some(p)
}

fn unique_temp(stem: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{}_{n}.ogg", std::process::id()))
}

/// 把整份文件按页搬一遍。
fn remux(orig: &[u8]) -> Vec<u8> {
    let mut r = PageReader::new(orig);
    r.set_verify_crc(true);
    let mut m: Option<OggMuxer> = None;
    let mut pages = 0usize;
    let mut packets = 0usize;
    while let Some(p) = r.next_page().expect("读页") {
        // 第一个页的 `serial` 决定整份文件的 `serial` —— Ogg 里同一条逻辑流
        // 的页必须同一个 serial（多流复用才有第二个）。
        let mux = m.get_or_insert_with(|| OggMuxer::new(p.serial));
        packets += p.packet_count();
        mux.write_page(&p.segments, &p.body, p.granule, p.flags.continued())
            .expect("写页");
        pages += 1;
    }
    assert_eq!(r.crc_errors(), 0, "原文件的 CRC 就不对 —— 样本坏了，别拿它当判据");
    eprintln!("搬了 {pages} 页 / {packets} 个包");
    m.expect("一个页都没读到").finish()
}

#[test]
fn remux_is_byte_identical_to_the_original() {
    let Some(input) = sample_path() else { return };
    let orig = std::fs::read(&input).expect("读样本");
    let ours = remux(&orig);

    let out = unique_temp("v2v_ogg_remux");
    std::fs::write(&out, &ours).expect("写重封装文件");
    eprintln!(
        "{}（{} 字节）→ {}（{} 字节）",
        input.display(),
        orig.len(),
        out.display(),
        ours.len()
    );

    // ---- 判据 1：逐字节相同 ----
    //
    // 原样搬页时这必须成立。`EOS` 是唯一的例外：原文件如果没标，我们会补在最后
    // 一页上（规范要求有），那种样本下长度会一样但有一个字节不同。
    if orig != ours {
        let at = orig
            .iter()
            .zip(&ours)
            .position(|(a, b)| a != b)
            .unwrap_or(orig.len().min(ours.len()));
        // 说清楚「差在哪个字节」，否则一个 500 KB 的比对失败等于什么都没说。
        let page = orig[..at.min(orig.len())]
            .windows(4)
            .rposition(|w| w == b"OggS")
            .map(|p| at - p)
            .unwrap_or(at);
        panic!(
            "重封装产物与原文件不同：长度 {} vs {}，首个差异在第 {at} 字节\
             （页内偏移 {page}）",
            orig.len(),
            ours.len()
        );
    }

    // ---- 判据 2：再读一遍，页数 / 包数 / granule 都对得上 ----
    let mut a = PageReader::new(&orig[..]);
    let mut b = PageReader::new(&ours[..]);
    a.set_verify_crc(true);
    b.set_verify_crc(true);
    let mut n = 0usize;
    loop {
        match (a.next_page().expect("读原文件"), b.next_page().expect("读产物")) {
            (Some(pa), Some(pb)) => {
                assert_eq!(pa.serial, pb.serial, "第 {n} 页的 serial");
                assert_eq!(pa.sequence, pb.sequence, "第 {n} 页的 sequence");
                assert_eq!(pa.granule, pb.granule, "第 {n} 页的 granule");
                assert_eq!(pa.segments, pb.segments, "第 {n} 页的段表");
                assert_eq!(pa.body, pb.body, "第 {n} 页的页体");
                n += 1;
            }
            (None, None) => break,
            (x, y) => panic!(
                "页数不同：原文件第 {n} 页是 {}，产物是 {}",
                if x.is_some() { "有" } else { "没有" },
                if y.is_some() { "有" } else { "没有" }
            ),
        }
    }
    assert!(n > 0, "一页都没读到");
    assert_eq!(b.crc_errors(), 0, "产物的 CRC 必须全对");

    let _ = std::fs::remove_file(&out);
}

/// 只搬**中间一段**页：产物必须是原文件的一个连续子串（页级无损剪切的基础）。
///
/// 这条不要求 granule 重新标定 —— 剪切要改的是 granule 的**基准**，
/// 那是 `trim` 那一层的事。这里只验「页写入器不会把跨页续接的包弄坏」。
#[test]
fn remux_of_a_page_window_is_a_contiguous_slice() {
    let Some(input) = sample_path() else { return };
    let orig = std::fs::read(&input).expect("读样本");

    let mut r = PageReader::new(&orig[..]);
    r.set_verify_crc(true);
    let mut m: Option<OggMuxer> = None;
    let mut kept = 0usize;
    let mut first_at: Option<usize> = None;
    let mut last_end = 0usize;
    let mut at = 0usize;
    let mut n = 0usize;
    while let Some(p) = r.next_page().expect("读页") {
        let len = p.byte_len();
        // 取中间 1/4 到 1/2 的页（跳过 BOS 那几个头页）。
        n += 1;
        if n >= 8 && n < 24 {
            let mux = m.get_or_insert_with(|| OggMuxer::new(p.serial));
            mux.write_page(&p.segments, &p.body, p.granule, p.flags.continued())
                .expect("写页");
            first_at.get_or_insert(at);
            last_end = at + len;
            kept += 1;
        }
        at += len;
    }
    if kept == 0 {
        eprintln!("SKIP: 样本页数太少");
        return;
    }
    let ours = m.expect("没写页").finish();
    let (start, end) = (first_at.unwrap(), last_end);
    let expect = &orig[start..end];
    // 页内容必须一模一样（页头里的 sequence 我们是从 0 重编的，所以前 6 字节
    // 与 CRC 会不同 —— 比的是**页体**，那是真正搬过去的东西）。
    let mut a = PageReader::new(expect);
    let mut b = PageReader::new(&ours[..]);
    let mut k = 0usize;
    loop {
        match (a.next_page().expect("读原文件片段"), b.next_page().expect("读产物")) {
            (Some(pa), Some(pb)) => {
                assert_eq!(pa.segments, pb.segments, "第 {k} 页的段表");
                assert_eq!(pa.body, pb.body, "第 {k} 页的页体");
                assert_eq!(pa.granule, pb.granule, "第 {k} 页的 granule");
                k += 1;
            }
            (None, None) => break,
            _ => panic!("页数不同（第 {k} 页）"),
        }
    }
    assert_eq!(k, kept, "搬了多少页就要写出多少页");
    let _ = Path::new(&input);
}
