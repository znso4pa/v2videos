//! AVI 解封装的对账测试。
//!
//! 两层：
//! 1. **手写的最小 AVI**（不依赖外部工具、永远能跑）—— 验 `hdrl` / `movi` / `idx1`
//!    的解析、时间戳换算、关键帧标志。
//! 2. **真样本的「同流异容器」对账**（`#[ignore]`，需要样本 + ffmpeg）——
//!    同一条 MP43 码流分别放进 `.wmv` 与 `.avi`，两个解封装器吐出的**帧字节必须
//!    一模一样**。这是容器解析正确性的硬判据：容器只该改变「怎么装」，
//!    不该改变「装的是什么」。

use std::io::Cursor;
use std::path::Path;
use std::process::Command;

use v2v_avi::{AviDemuxer, StreamKind};

// ==================== 手写 AVI 的工具 ====================

fn u32le(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn u16le(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn cc(v: &mut Vec<u8>, s: &[u8; 4]) {
    v.extend_from_slice(s);
}

fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    cc(&mut v, id);
    u32le(&mut v, body.len() as u32);
    v.extend_from_slice(body);
    if body.len() % 2 == 1 {
        v.push(0);
    }
    v
}

fn list(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut inner = Vec::new();
    cc(&mut inner, kind);
    inner.extend_from_slice(body);
    chunk(b"LIST", &inner)
}

/// 一个 16×16、MP43、30fps、3 帧的最小 AVI；帧 0 是关键帧。
fn handmade_avi() -> Vec<u8> {
    // avih（56 字节）
    let mut avih = Vec::new();
    u32le(&mut avih, 33333); // 微秒/帧
    u32le(&mut avih, 0); // 最大字节率
    u32le(&mut avih, 0); // padding
    u32le(&mut avih, 0x10); // 标志：HASINDEX
    u32le(&mut avih, 3); // 总帧数
    u32le(&mut avih, 0); // 初始帧
    u32le(&mut avih, 1); // 流数
    u32le(&mut avih, 0); // 建议缓冲
    u32le(&mut avih, 16); // 宽
    u32le(&mut avih, 16); // 高
    for _ in 0..4 {
        u32le(&mut avih, 0);
    }

    // strh（56 字节）
    let mut strh = Vec::new();
    cc(&mut strh, b"vids");
    cc(&mut strh, b"MP43");
    u32le(&mut strh, 0); // 标志
    u16le(&mut strh, 0); // 优先级
    u16le(&mut strh, 0); // 语言
    u32le(&mut strh, 0); // 初始帧
    u32le(&mut strh, 1); // scale
    u32le(&mut strh, 30); // rate → 30fps
    u32le(&mut strh, 0); // start
    u32le(&mut strh, 3); // length
    u32le(&mut strh, 0); // 建议缓冲
    u32le(&mut strh, 0xFFFF_FFFF); // quality
    u32le(&mut strh, 0); // sample size
    for _ in 0..4 {
        u16le(&mut strh, 0); // rcFrame
    }

    // strf：BITMAPINFOHEADER（40 字节）
    let mut strf = Vec::new();
    u32le(&mut strf, 40);
    u32le(&mut strf, 16);
    u32le(&mut strf, 16);
    u16le(&mut strf, 1);
    u16le(&mut strf, 24);
    cc(&mut strf, b"MP43");
    u32le(&mut strf, 16 * 16 * 3);
    u32le(&mut strf, 0);
    u32le(&mut strf, 0);
    u32le(&mut strf, 0);
    u32le(&mut strf, 0);

    let strl = list(
        b"strl",
        &[chunk(b"strh", &strh), chunk(b"strf", &strf)].concat(),
    );
    let hdrl = list(b"hdrl", &[chunk(b"avih", &avih), strl].concat());

    // movi：三帧（第三帧长度是奇数，验补齐）
    let f0: Vec<u8> = vec![1, 2, 3];
    let f1: Vec<u8> = vec![4, 5, 6, 7];
    let f2: Vec<u8> = vec![8, 9];
    let movi_body = [
        chunk(b"00dc", &f0),
        chunk(b"00dc", &f1),
        chunk(b"00dc", &f2),
    ]
    .concat();
    let movi = list(b"movi", &movi_body);

    // idx1：偏移相对 movi **数据**起点
    let mut idx1 = Vec::new();
    let entries = [
        (f0.len() as u32, 0x10u32, 0u32),           // 关键帧
        (f1.len() as u32, 0u32, 8 + 3 + 1),         // 非关键帧
        (f2.len() as u32, 0u32, 8 + 3 + 1 + 8 + 4), // 非关键帧
    ];
    for (size, flags, off) in entries {
        cc(&mut idx1, b"00dc");
        u32le(&mut idx1, flags);
        u32le(&mut idx1, off);
        u32le(&mut idx1, size);
    }
    let idx1 = chunk(b"idx1", &idx1);

    let body = [hdrl, movi, idx1].concat();
    let mut out = Vec::new();
    cc(&mut out, b"RIFF");
    u32le(&mut out, (body.len() + 4) as u32);
    cc(&mut out, b"AVI ");
    out.extend_from_slice(&body);
    out
}

#[test]
fn parses_a_handmade_avi() {
    let file = handmade_avi();
    let mut d = AviDemuxer::new(Cursor::new(&file)).expect("解析");
    assert!(d.has_index(), "有 idx1 就该认出来");

    assert_eq!(d.streams().len(), 1);
    let s = &d.streams()[0];
    assert_eq!(s.kind, StreamKind::Video);
    let vf = s.video.as_ref().expect("视频格式");
    assert_eq!(vf.width, 16);
    assert_eq!(vf.height, 16);
    assert_eq!(v2v_avi::fourcc_string(vf.fourcc), "MP43");

    let mut got = Vec::new();
    while let Some(sm) = d.next_sample().expect("读样本") {
        got.push(sm);
    }
    assert_eq!(got.len(), 3);
    assert_eq!(got[0].data, vec![1, 2, 3]);
    assert_eq!(got[1].data, vec![4, 5, 6, 7]);
    assert_eq!(got[2].data, vec![8, 9]);
    // 30fps → 0 / 33 / 66 ms
    assert_eq!(
        got.iter().map(|s| s.timestamp_ms).collect::<Vec<_>>(),
        vec![0, 33, 66]
    );
    // 关键帧标志来自 idx1
    assert_eq!(
        got.iter().map(|s| s.key_frame).collect::<Vec<_>>(),
        vec![true, false, false]
    );
}

#[test]
fn rejects_non_avi() {
    let junk = b"RIFF\x00\x00\x00\x00WAVEfmt ".to_vec();
    match AviDemuxer::new(Cursor::new(junk)) {
        Err(v2v_avi::Error::Bad(_)) => {}
        Err(e) => panic!("期望 Bad，得到 {e}"),
        Ok(_) => panic!("WAVE 不是 AVI，不该解成功"),
    }
    let short = b"RIFF".to_vec();
    assert!(AviDemuxer::new(Cursor::new(short)).is_err());
}

// ==================== 真样本：同流异容器 ====================

/// **同一条 MP43 码流，两个容器**：ffmpeg 把样本无损转进 AVI（`-c:v copy`），
/// 我们分别用 AVI / ASF 解封装器拆开 —— 帧字节必须**逐字节相同**。
///
/// ```text
/// V2V_ASF_SAMPLE=<样本> cargo test --release -p v2v-avi -- --ignored --nocapture
/// ```
#[test]
#[ignore = "需要真样本与 ffmpeg"]
fn same_stream_in_avi_and_wmv_matches() {
    let src = match std::env::var("V2V_ASF_SAMPLE") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SKIP: V2V_ASF_SAMPLE 未设置");
            return;
        }
    };
    if !Path::new(&src).is_file() {
        eprintln!("SKIP: 样本不存在");
        return;
    }
    let dir = std::env::temp_dir().join(format!("v2v_avi_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let avi_path = dir.join("same.avi");

    let ok = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-i",
            &src,
            "-t",
            "2",
            "-c:v",
            "copy",
            "-an",
            avi_path.to_str().unwrap(),
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("SKIP: 没有可用的 ffmpeg");
        return;
    }

    // ffmpeg 写出的这个 AVI 声明 1000fps（保留源的 1/1000 时基），
    // 所以里面大部分帧槽是空的，只有少数真有数据。
    //
    // ⚠️ **不要硬编码帧数**。这个数取决于样本本身的时长与帧率
    // （2 秒 / 25fps 只有 51 帧，但 4 秒 / 50fps 就有 200 帧）——
    // 写死 `n = 100` 会对低帧率样本直接失败，看着像 AVI 解封装坏了、
    // 其实是**测试的前提**不成立（实测踩过）。
    //
    // 也不能「AVI 读尽、ASF 读尽」然后比总数：上面的 ffmpeg 命令带 `-t 2`，
    // **只截了 2 秒**装进 AVI，而 ASF 侧是整条流 —— 总数当然对不上。
    // 正解是**两边各读尽**，然后比**较短的**那个长度（见下面的 `n`）。
    let n = usize::MAX;

    // ---- AVI 侧 ----
    let mut ad = AviDemuxer::new(std::fs::File::open(&avi_path).expect("开 AVI")).expect("解 AVI");
    let (anum, aw, ah, afourcc) = {
        let v = ad
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Video)
            .expect("AVI 视频轨");
        let vf = v.video.as_ref().expect("AVI 视频格式");
        (v.number, vf.width, vf.height, vf.fourcc)
    };
    assert_eq!(v2v_avi::fourcc_string(afourcc), "MP43");
    let mut avi_frames = Vec::new();
    while avi_frames.len() < n {
        match ad.next_sample().expect("读 AVI") {
            Some(s) if s.stream == anum => avi_frames.push(s),
            Some(_) => {}
            None => break,
        }
    }

    // ---- ASF 侧（同一条码流）----
    let mut sd =
        v2v_asf::AsfDemuxer::new(std::fs::File::open(&src).expect("开 WMV")).expect("解 WMV");
    let (snum, sw, sh) = {
        let v = sd
            .streams()
            .iter()
            .find(|s| s.kind == v2v_asf::StreamKind::Video)
            .expect("WMV 视频轨");
        let vf = v.video.as_ref().expect("WMV 视频格式");
        (v.number, vf.width, vf.height)
    };
    let mut asf_frames = Vec::new();
    // AVI 那边被 `-t 2` 截过，所以它更短 —— 读到**它的长度**就够，
    // 不用把整条 ASF 流读尽（整条读完要白等很久）。
    while asf_frames.len() < avi_frames.len().max(1) {
        match sd.next_sample().expect("读 WMV") {
            Some(s) if s.stream == snum => asf_frames.push(s),
            Some(_) => {}
            None => break,
        }
    }

    eprintln!(
        "AVI {} 帧（{aw}×{ah}）/ ASF {} 帧（{sw}×{sh}）",
        avi_frames.len(),
        asf_frames.len()
    );
    assert_eq!(aw, sw);
    assert_eq!(ah, sh);

    // 取两边的**较小长度**来逐帧比 —— AVI 那边被 `-t 2` 截过，
    // 所以它通常更短，用它的长度去比 ASF 的**前缀**。
    // 关键断言是「同一序号的帧字节完全相同」，长度只决定比多少帧。
    let n = avi_frames.len().min(asf_frames.len());
    assert!(n > 0, "两边的视频帧都是空的");

    // 帧字节逐字节相同 + 关键帧标志一致
    let mut keys_avi = 0;
    let mut keys_asf = 0;
    for i in 0..n {
        assert_eq!(
            avi_frames[i].data, asf_frames[i].data,
            "第 {i} 帧：AVI 与 WMV 里的码流字节不一致"
        );
        assert_eq!(
            avi_frames[i].key_frame, asf_frames[i].key_frame,
            "第 {i} 帧：关键帧标志不一致"
        );
        if avi_frames[i].key_frame {
            keys_avi += 1;
        }
        if asf_frames[i].key_frame {
            keys_asf += 1;
        }
    }
    eprintln!("前 {n} 帧字节一致；关键帧 AVI {keys_avi} / ASF {keys_asf}");
    assert!(keys_avi > 0, "前 {n} 帧里一个关键帧都没有，可疑");
    let _ = std::fs::remove_dir_all(&dir);
}

/// **空槽语义**：AVI 允许「有帧槽但没数据」的块（ffmpeg 的封装器就会这么写）。
/// 这种块**不产样本**，但**占时间轴** —— 后面的帧时间戳不能因此前移。
#[test]
fn empty_chunks_are_skipped_but_still_advance_time() {
    let mut avih = Vec::new();
    u32le(&mut avih, 1000);
    for _ in 0..6 {
        u32le(&mut avih, 0);
    }
    u32le(&mut avih, 1); // 流数
    u32le(&mut avih, 0);
    u32le(&mut avih, 16);
    u32le(&mut avih, 16);
    for _ in 0..4 {
        u32le(&mut avih, 0);
    }
    let mut strh = Vec::new();
    cc(&mut strh, b"vids");
    cc(&mut strh, b"MP43");
    u32le(&mut strh, 0); // flags
    u16le(&mut strh, 0); // priority
    u16le(&mut strh, 0); // language
    u32le(&mut strh, 0); // initial frames
    u32le(&mut strh, 1); // scale
    u32le(&mut strh, 1000); // rate → 1000fps（每毫秒一个槽）
    for _ in 0..3 {
        u32le(&mut strh, 0);
    }
    u32le(&mut strh, 0xFFFF_FFFF);
    u32le(&mut strh, 0);
    for _ in 0..4 {
        u16le(&mut strh, 0);
    }
    let mut strf = Vec::new();
    u32le(&mut strf, 40);
    u32le(&mut strf, 16);
    u32le(&mut strf, 16);
    u16le(&mut strf, 1);
    u16le(&mut strf, 24);
    cc(&mut strf, b"MP43");
    for _ in 0..5 {
        u32le(&mut strf, 0);
    }
    let strl = list(
        b"strl",
        &[chunk(b"strh", &strh), chunk(b"strf", &strf)].concat(),
    );
    let hdrl = list(b"hdrl", &[chunk(b"avih", &avih), strl].concat());

    // 槽 0 有数据、槽 1 空、槽 2 有数据 → 只应吐 2 个样本，时间戳 0 与 2 ms
    let movi = list(
        b"movi",
        &[
            chunk(b"00dc", &[1, 2, 3]),
            chunk(b"00dc", &[]),
            chunk(b"00dc", &[9]),
        ]
        .concat(),
    );
    let body = [hdrl, movi].concat();
    let mut file = Vec::new();
    cc(&mut file, b"RIFF");
    u32le(&mut file, (body.len() + 4) as u32);
    cc(&mut file, b"AVI ");
    file.extend_from_slice(&body);

    let mut d = AviDemuxer::new(Cursor::new(&file)).expect("解析");
    let mut got = Vec::new();
    while let Some(sm) = d.next_sample().expect("读") {
        got.push(sm);
    }
    assert_eq!(got.len(), 2, "空槽不该产样本");
    assert_eq!(got[0].data, vec![1, 2, 3]);
    assert_eq!(got[0].timestamp_ms, 0);
    assert_eq!(got[1].data, vec![9]);
    assert_eq!(got[1].timestamp_ms, 2, "空槽必须占时间轴");
}
