//! MP4 封装器**真样本对账**。
//!
//! ## 参考产物怎么来
//!
//! ```bash
//! # 纯视频参考：把一个带 MP43 视频轨的样本转成 MPEG-4 Part 2（mp4v）的 MP4
//! ffmpeg -i <样本.wmv> -map 0:v -c:v mpeg4 -an -f mp4 <参考.mp4>
//! # 裸 PCM（s16le）—— 喂给我们的 sowt 轨
//! ffmpeg -i <样本.wmv> -map 0:a -c:a pcm_s16le -f s16le <裸 PCM>
//! ```
//!
//! ## 怎么跑
//!
//! ```bash
//! V2V_MP4_REF=<参考.mp4> V2V_MP4_OUT=/tmp/ours_v.mp4 \
//! cargo test -p v2v-mp4 --test real_sample -- --ignored --nocapture
//!
//! V2V_MP4_REF=<参考.mp4> V2V_MP4_PCM=<裸 PCM> V2V_MP4_OUT_AV=/tmp/ours_av.mp4 \
//! cargo test -p v2v-mp4 --test real_sample -- --ignored --nocapture
//! ```
//!
//! ## 为什么这样对账
//!
//! 封装器没有「参考实现」可比 —— 它的正确性判据只有一条：
//! **同一个码流换个封装，解码结果必须一模一样**。
//! 所以流程是：
//!
//! 1. 读参考 MP4 的**样本表**（`stsz` / `stsc` / `stco` / `stss` / `stts`）
//!    和 `esds` 里的 DSI；
//! 2. 按展开后的偏移 + `stsz` 从参考文件的 `mdat` 里把每个样本**原样抠出来**；
//! 3. 用 `Mp4Muxer` 重新封一遍；
//! 4. **比表**：`stsz` / `stts` / `stss` 必须逐项相同，偏移只能差一个常量
//!    （参考文件在 `ftyp` 后面多写了一个 8 字节 `free`，我们没有）；
//! 5. **比解码**：两边各跑一遍 `ffmpeg -f framemd5`，逐帧 MD5 必须相同。
//!
//! 第 5 步是硬判据。第 4 步只是让失败时的诊断更省事 ——
//! 表错了但解码还对，说明表没被用到；解码错了但表对，说明错在别处。
//!
//! 音频那一路更省事：PCM 的判据可以**精确到字节** —— 喂进去多少采样，
//! `ffmpeg -f s16le` 解出来必须一模一样，不需要任何容差。
//!
//! ## 为什么不用 `-f m4v` 抽基本流
//!
//! 试过：`-f m4v` 的输出**不是** MP4 样本的简单拼接（同一份输入，
//! `m4v` 比 `stsz` 之和大 50 万字节 —— 基本流里每个关键帧前都重复了
//! 一整套 VOS/VO/VOL，而 MP4 把这些放进了 `esds`）。从参考 MP4 自己
//! 的 `stco`/`stsz` 里抠样本，才保证「喂进去的就是它装过的」。

use std::path::{Path, PathBuf};
use std::process::Command;

use v2v_mp4::{AudioCodec, AudioTrack, Mp4Muxer, VideoTrack};

// ==================== 一个够用的只读 ISOBMFF 遍历器 ====================
//
// 只在测试里用。刻意不复用 `crates/mp4/src/writer.rs` —— 那边是**写**，
// 这边是**读**；共用会让「两边一起错」变得难以察觉。

fn be32(d: &[u8]) -> u32 {
    u32::from_be_bytes(d[0..4].try_into().unwrap())
}

fn be16(d: &[u8]) -> u16 {
    u16::from_be_bytes(d[0..2].try_into().unwrap())
}

/// 列出 `[start, end)` 里的所有 box，返回 `(type, body_start, body_end)`。
fn boxes(d: &[u8], start: usize, end: usize) -> Vec<([u8; 4], usize, usize)> {
    let mut v = Vec::new();
    let mut p = start;
    while p + 8 <= end {
        let sz = be32(&d[p..]) as usize;
        let mut typ = [0u8; 4];
        typ.copy_from_slice(&d[p + 4..p + 8]);
        if sz < 8 || p + sz > end {
            break;
        }
        v.push((typ, p + 8, p + sz));
        p += sz;
    }
    v
}

/// 递归找第一个叫 `name` 的 box。
///
/// 只往**纯容器** box 里钻（`moov`/`trak`/`mdia`/`minf`/`stbl`）。
/// 不钻 `stsd` —— 否则会顺着 sample entry 找到里面嵌套的 `esds`，
/// 那就不是「stbl 的 esds」了。
fn find(d: &[u8], name: &[u8; 4], start: usize, end: usize) -> Option<(usize, usize)> {
    for (t, s, e) in boxes(d, start, end) {
        if &t == name {
            return Some((s, e));
        }
        if matches!(&t, b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl") {
            if let Some(r) = find(d, name, s, e) {
                return Some(r);
            }
        }
    }
    None
}

/// `stsd` 的第一个 sample entry：`(4CC, body 起点, body 终点)`。
fn first_sample_entry(d: &[u8], stsd: (usize, usize)) -> ([u8; 4], usize, usize) {
    let b = stsd.0;
    assert_eq!(be32(&d[b + 4..]), 1, "参考文件应该只有一个 sample entry");
    let ep = b + 8;
    let sz = be32(&d[ep..]) as usize;
    let mut typ = [0u8; 4];
    typ.copy_from_slice(&d[ep + 4..ep + 8]);
    assert!(sz >= 8, "sample entry 太小");
    (typ, ep + 8, ep + sz)
}

/// 读一个 MPEG-4 描述符的变长长度：每字节 7 位，MSB 是「还有下一字节」。
/// 最多 4 字节 —— 与 ffmpeg 的 `ff_mp4_read_descr_len` 一致。
fn descr_len(d: &[u8], p: &mut usize) -> usize {
    let mut len = 0usize;
    for _ in 0..4 {
        let c = d[*p];
        *p += 1;
        len = (len << 7) | (c & 0x7f) as usize;
        if c & 0x80 == 0 {
            break;
        }
    }
    len
}

/// 从 `esds` 里挖出 `DecoderSpecificInfo`（tag `0x05`）的字节。
fn dsi_from_esds(d: &[u8], esds: (usize, usize)) -> Vec<u8> {
    let mut p = esds.0 + 4; // version/flags
    assert_eq!(d[p], 0x03, "第一个描述符应是 ES_Descriptor");
    p += 1;
    let _ = descr_len(d, &mut p);
    p += 2; // ES_ID
    let flags = d[p];
    p += 1;
    if flags & 0x80 != 0 {
        p += 2; // dependsOn_ES_ID
    }
    if flags & 0x40 != 0 {
        let n = d[p] as usize; // URL 是定长 8 位长度
        p += 1 + n;
    }
    if flags & 0x20 != 0 {
        p += 2; // OCR_ES_Id
    }
    assert_eq!(d[p], 0x04, "第二个描述符应是 DecoderConfigDescriptor");
    p += 1;
    let _ = descr_len(d, &mut p);
    p += 1 + 1 + 3 + 4 + 4; // objectType / streamType / bufferSize / max / avg
    assert_eq!(d[p], 0x05, "第三个描述符应是 DecoderSpecificInfo");
    p += 1;
    let n = descr_len(d, &mut p);
    d[p..p + n].to_vec()
}

// ---------- 样本表 ----------

fn stsz_sizes(d: &[u8], r: (usize, usize)) -> Vec<u32> {
    let b = r.0;
    let def = be32(&d[b + 4..]);
    let n = be32(&d[b + 8..]) as usize;
    (0..n)
        .map(|i| if def != 0 { def } else { be32(&d[b + 12 + 4 * i..]) })
        .collect()
}

fn stco_offsets(d: &[u8], r: (usize, usize)) -> Vec<u32> {
    let b = r.0;
    let n = be32(&d[b + 4..]) as usize;
    (0..n).map(|i| be32(&d[b + 8 + 4 * i..])).collect()
}

/// 同步样本号（**1 起**）。`stss` 缺席 = 全部都是同步点。
fn stss_keys(d: &[u8], r: (usize, usize)) -> Vec<u32> {
    let b = r.0;
    let n = be32(&d[b + 4..]) as usize;
    (0..n).map(|i| be32(&d[b + 8 + 4 * i..])).collect()
}

/// 把 `stts` 的游程展开成「每个样本一个时长」。
fn stts_durations(d: &[u8], r: (usize, usize)) -> Vec<u32> {
    let b = r.0;
    let n = be32(&d[b + 4..]) as usize;
    let mut out = Vec::new();
    for i in 0..n {
        let cnt = be32(&d[b + 8 + 8 * i..]) as usize;
        let delta = be32(&d[b + 12 + 8 * i..]);
        for _ in 0..cnt {
            out.push(delta);
        }
    }
    out
}

/// 把 `stco`（逐 chunk 偏移）+ `stsc`（chunk→样本数）+ `stsz` 展开成
/// **逐样本的绝对偏移**。
///
/// 参考文件里 ffmpeg 会把同一条轨的连续样本合并进一个 chunk（这个样本里
/// 75 个视频样本只占 1 个 chunk），所以 `stco` 的项数不等于样本数，
/// 必须按 `stsc` 展开才能和我们「一样本一 chunk」的表逐项比。
fn stsc_expand(d: &[u8], stsc: (usize, usize), stco: (usize, usize), sizes: &[u32]) -> Vec<u32> {
    let b = stsc.0;
    let n = be32(&d[b + 4..]) as usize;
    let entries: Vec<(u32, u32)> = (0..n)
        .map(|i| (be32(&d[b + 8 + 12 * i..]), be32(&d[b + 12 + 12 * i..])))
        .collect();
    let chunk_offs = stco_offsets(d, stco);

    let mut out = Vec::with_capacity(sizes.len());
    let mut si = 0usize;
    for (ci, &base) in chunk_offs.iter().enumerate() {
        let chunk_no = ci as u32 + 1;
        // 适用项 = first_chunk <= chunk_no 的最后一项
        let per = entries
            .iter()
            .rev()
            .find(|(fc, _)| *fc <= chunk_no)
            .map(|(_, spc)| *spc)
            .expect("stsc 第一项应从 chunk 1 开始");
        let mut within = 0u32;
        for _ in 0..per {
            assert!(si < sizes.len(), "stsc 展开出的样本数多于 stsz");
            out.push(base + within);
            within += sizes[si];
            si += 1;
        }
    }
    assert_eq!(si, sizes.len(), "stsc 展开出的样本数少于 stsz");
    out
}

/// `mdhd` 的时间基。
fn mdhd_timescale(d: &[u8], r: (usize, usize)) -> u32 {
    let b = r.0;
    assert_eq!(d[b], 0, "只处理 version 0 的 mdhd");
    be32(&d[b + 12..])
}

// ==================== 参考文件里那条视频轨的全部信息 ====================

struct RefTrack {
    width: u16,
    height: u16,
    timescale: u32,
    dsi: Vec<u8>,
    samples: Vec<Vec<u8>>,
    /// 逐样本的绝对偏移（已按 `stsc` 展开）。
    sample_offsets: Vec<u32>,
    durations: Vec<u32>,
    keys: Vec<bool>,
}

fn read_ref_video(path: &Path) -> RefTrack {
    let d = std::fs::read(path).expect("读参考 MP4");
    let moov = find(&d, b"moov", 0, d.len()).expect("有 moov");
    let trak = find(&d, b"trak", moov.0, moov.1).expect("有 trak");
    let stbl = find(&d, b"stbl", trak.0, trak.1).expect("有 stbl");

    let stsd = find(&d, b"stsd", stbl.0, stbl.1).expect("有 stsd");
    let (fourcc, sb, se) = first_sample_entry(&d, stsd);
    assert_eq!(&fourcc, b"mp4v", "参考文件应是 MPEG-4 Part 2");

    // VisualSampleEntry：6 reserved + 2 data_ref + 2 pre_defined + 2 reserved
    // + 12 pre_defined[3] + 2 width + 2 height
    let width = be16(&d[sb + 24..]);
    let height = be16(&d[sb + 26..]);

    // 子 box 从 78 字节固定头之后开始
    let esds = boxes(&d, sb + 78, se)
        .into_iter()
        .find(|(t, _, _)| t == b"esds")
        .map(|(_, s, e)| (s, e))
        .expect("mp4v entry 里有 esds");
    let dsi = dsi_from_esds(&d, esds);

    let mdhd = find(&d, b"mdhd", trak.0, trak.1).expect("有 mdhd");
    let timescale = mdhd_timescale(&d, mdhd);

    let stsz = find(&d, b"stsz", stbl.0, stbl.1).expect("有 stsz");
    let stco = find(&d, b"stco", stbl.0, stbl.1).expect("有 stco");
    let stsc = find(&d, b"stsc", stbl.0, stbl.1).expect("有 stsc");
    let stts = find(&d, b"stts", stbl.0, stbl.1).expect("有 stts");
    let sizes = stsz_sizes(&d, stsz);
    let offsets = stsc_expand(&d, stsc, stco, &sizes);
    let durations = stts_durations(&d, stts);
    assert_eq!(sizes.len(), offsets.len(), "stsz 与 stsc 展开后应等长");
    assert_eq!(sizes.len(), durations.len(), "stsz 与 stts 展开后应等长");

    let keys: Vec<bool> = match find(&d, b"stss", stbl.0, stbl.1) {
        Some(r) => {
            let set: Vec<u32> = stss_keys(&d, r);
            (1..=sizes.len() as u32).map(|i| set.contains(&i)).collect()
        }
        None => vec![true; sizes.len()],
    };

    // 从 mdat 里按 (offset, size) 原样抠样本
    let mdat = find(&d, b"mdat", 0, d.len()).expect("有 mdat");
    let samples: Vec<Vec<u8>> = offsets
        .iter()
        .zip(&sizes)
        .map(|(&o, &s)| {
            let o = o as usize;
            assert!(
                o >= mdat.0 && o + s as usize <= mdat.1,
                "样本 @{o}+{s} 落在 mdat 之外"
            );
            d[o..o + s as usize].to_vec()
        })
        .collect();

    RefTrack { width, height, timescale, dsi, samples, sample_offsets: offsets, durations, keys }
}

// ==================== ffmpeg 对账 ====================

/// 跑一遍 ffmpeg 并把 stdout 收回来。
fn ffmpeg_capture(path: &Path, args: &[&str]) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-v", "error", "-i"])
        .arg(path)
        .args(args)
        .output()
        .expect("跑不动 ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg 读不了 {}：{}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// `ffmpeg -f framemd5` 的数据行（去掉 `#` 开头的注释/表头）。
fn framemd5(path: &Path, map: &str) -> String {
    let raw = ffmpeg_capture(path, &["-map", map, "-f", "framemd5", "-"]);
    String::from_utf8_lossy(&raw)
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
#[ignore = "需要真样本与 ffmpeg"]
fn remuxes_a_real_mpeg4_track() {
    let src = match std::env::var("V2V_MP4_REF") {
        Ok(p) if Path::new(&p).is_file() => PathBuf::from(p),
        _ => {
            eprintln!("SKIP: V2V_MP4_REF 未设置或文件不存在");
            return;
        }
    };
    let out = PathBuf::from(
        std::env::var("V2V_MP4_OUT").unwrap_or_else(|_| "/tmp/v2v_ours.mp4".into()),
    );

    // ---- 1. 读参考 ----
    let r = read_ref_video(&src);
    eprintln!(
        "参考：{}×{}，时间基 {}，{} 个样本，{} 个关键帧，DSI {} 字节",
        r.width,
        r.height,
        r.timescale,
        r.samples.len(),
        r.keys.iter().filter(|k| **k).count(),
        r.dsi.len()
    );
    assert!(!r.samples.is_empty(), "参考文件里没有样本");

    // ---- 2. 用我们的封装器重封 ----
    let mut m = Mp4Muxer::new(VideoTrack {
        width: r.width,
        height: r.height,
        timescale: r.timescale,
        decoder_specific_info: r.dsi.clone(),
        avg_bitrate: 0,
    });
    for ((s, &dur), &key) in r.samples.iter().zip(&r.durations).zip(&r.keys) {
        m.write_video(s, key, dur).expect("写样本");
    }
    let ours = m.finish();
    std::fs::write(&out, &ours).expect("写我们的 MP4");
    eprintln!("写出 {}（{} 字节）", out.display(), ours.len());

    // ---- 3. 比表 ----
    let moov = find(&ours, b"moov", 0, ours.len()).expect("我们有 moov");
    let trak = find(&ours, b"trak", moov.0, moov.1).expect("我们有 trak");
    let stbl = find(&ours, b"stbl", trak.0, trak.1).expect("我们有 stbl");

    let our_sizes = stsz_sizes(&ours, find(&ours, b"stsz", stbl.0, stbl.1).unwrap());
    let our_offs = stco_offsets(&ours, find(&ours, b"stco", stbl.0, stbl.1).unwrap());
    let our_durs = stts_durations(&ours, find(&ours, b"stts", stbl.0, stbl.1).unwrap());

    assert_eq!(
        our_sizes,
        r.samples.iter().map(|s| s.len() as u32).collect::<Vec<_>>(),
        "stsz 必须逐项相同"
    );
    assert_eq!(our_durs, r.durations, "stts 展开后必须逐项相同");

    let ref_offs = &r.sample_offsets;
    // 参考文件在 ftyp 后多写了一个 8 字节 free，所以我们的偏移整体前移一个常量
    let delta = our_offs[0] as i64 - ref_offs[0] as i64;
    for (i, (&o, &ro)) in our_offs.iter().zip(ref_offs).enumerate() {
        assert_eq!(
            o as i64 - ro as i64,
            delta,
            "第 {i} 个样本的偏移不是同一个常量差（{o} vs {ro}）"
        );
    }
    eprintln!("偏移常量差 = {delta} 字节（参考文件的 free box）");

    // 关键帧集合（我们的 stss 是按 `key` 写的，参考的是原始表）
    let our_keys: Vec<bool> = match find(&ours, b"stss", stbl.0, stbl.1) {
        Some(r2) => {
            let set: Vec<u32> = stss_keys(&ours, r2);
            (1..=our_sizes.len() as u32).map(|i| set.contains(&i)).collect()
        }
        None => vec![true; our_sizes.len()],
    };
    assert_eq!(our_keys, r.keys, "关键帧集合必须相同");

    // ---- 4. 让 ffmpeg 解码两边，逐帧比 MD5 ----
    let a = framemd5(&src, "0:v");
    let b = framemd5(&out, "0:v");
    let la: Vec<&str> = a.lines().collect();
    let lb: Vec<&str> = b.lines().collect();
    assert_eq!(la.len(), lb.len(), "解码出的帧数不同：{} vs {}", la.len(), lb.len());
    assert_eq!(a, b, "逐帧 MD5 不同");
    eprintln!("ffmpeg 逐帧 MD5 一致，共 {} 帧", la.len());
}

/// **A/V 双轨**对账：视频走 `mp4v`，音频走**未压缩 PCM**（`sowt`）。
///
/// ```text
/// V2V_MP4_REF=<纯视频参考 .mp4> \
/// V2V_MP4_PCM=<裸 s16le PCM（44.1 kHz 立体声）> \
/// V2V_MP4_OUT=/tmp/ours_av.mp4 \
/// cargo test -p v2v-mp4 --test real_sample -- --ignored --nocapture
/// ```
///
/// 音频这一路没有「参考实现」可对 —— 但 PCM 的好处是**判据可以精确到字节**：
/// 喂进去多少采样，解出来必须一模一样。所以：
///
/// 1. 视频照上一个测试那样从参考里抠样本；
/// 2. 音频按「一个样本 = 一个采样点」切（`stsz` 的 default size = 声道数 × 2）；
/// 3. **按时间交错**写入（每个视频帧后面补够它时长对应的音频采样点），
///    这样两条轨的样本在 `mdat` 里交错，正好考验 chunk 切分；
/// 4. 解码回来：视频逐帧 MD5 要一致，音频字节要**完全相等**。
#[test]
#[ignore = "需要真样本与 ffmpeg"]
fn muxes_video_plus_pcm_audio_and_keeps_both_intact() {
    let (src, pcm_path) = match (std::env::var("V2V_MP4_REF"), std::env::var("V2V_MP4_PCM")) {
        (Ok(a), Ok(b)) if Path::new(&a).is_file() && Path::new(&b).is_file() => {
            (PathBuf::from(a), PathBuf::from(b))
        }
        _ => {
            eprintln!("SKIP: V2V_MP4_REF / V2V_MP4_PCM 未设置或文件不存在");
            return;
        }
    };
    let out = PathBuf::from(
        std::env::var("V2V_MP4_OUT_AV").unwrap_or_else(|_| "/tmp/v2v_ours_av.mp4".into()),
    );

    const RATE: u32 = 44100;
    const CHANNELS: u16 = 2;
    const POINT: usize = CHANNELS as usize * 2; // s16le 一个采样点 4 字节

    let r = read_ref_video(&src);
    let pcm = std::fs::read(&pcm_path).expect("读 PCM");
    assert_eq!(pcm.len() % POINT, 0, "PCM 长度应是采样点的整数倍");
    let points = pcm.len() / POINT;
    eprintln!("视频 {} 帧；音频 {} 个采样点（{:.2} s）", r.samples.len(), points, points as f64 / RATE as f64);

    // ---- 按时间交错写 ----
    let mut m = Mp4Muxer::with_audio(
        VideoTrack {
            width: r.width,
            height: r.height,
            timescale: r.timescale,
            decoder_specific_info: r.dsi.clone(),
            avg_bitrate: 0,
        },
        AudioTrack {
            channels: CHANNELS,
            sample_rate: RATE,
            timescale: RATE,
            codec: AudioCodec::Pcm { bits: 16, big_endian: false },
            avg_bitrate: 0,
            // PCM 没有编码器延迟。
            priming: 0,
        },
    );

    let mut done = 0usize; // 已写入的音频采样点
    let mut acc = 0u64; // 累计「视频已覆盖的 tick × 采样率」—— 整数累加，避免漂移
    for ((s, &dur), &key) in r.samples.iter().zip(&r.durations).zip(&r.keys) {
        m.write_video(s, key, dur).expect("写视频");
        acc += dur as u64 * RATE as u64;
        // 到现在为止，音频应该已经写到第几个采样点
        let want = (acc / r.timescale as u64) as usize;
        let n = want.saturating_sub(done).min(points - done);
        for i in 0..n {
            let p = (done + i) * POINT;
            m.write_audio(&pcm[p..p + POINT], 1).expect("写音频");
        }
        done += n;
    }
    // 视频放完还有剩的音频就补齐
    for i in done..points {
        let p = i * POINT;
        m.write_audio(&pcm[p..p + POINT], 1).expect("写尾部音频");
    }
    assert_eq!(done, points, "音频采样点应全部写完");

    let ours = m.finish();
    std::fs::write(&out, &ours).expect("写我们的 MP4");
    eprintln!("写出 {}（{} 字节）", out.display(), ours.len());

    // ---- 视频：逐帧 MD5 ----
    let a = framemd5(&src, "0:v");
    let b = framemd5(&out, "0:v");
    assert_eq!(a.lines().count(), b.lines().count(), "视频帧数不同");
    assert_eq!(a, b, "视频逐帧 MD5 不同");

    // ---- 音频：字节必须完全相等 ----
    let back = ffmpeg_capture(&out, &["-map", "0:a", "-f", "s16le", "-"]);
    assert_eq!(
        back.len(),
        pcm.len(),
        "解回来的 PCM 长度不同：{} vs {}",
        back.len(),
        pcm.len()
    );
    assert!(back == pcm, "解回来的 PCM 字节不同");

    eprintln!(
        "视频 {} 帧 MD5 一致；音频 {} 字节原样往返",
        a.lines().count(),
        back.len()
    );
}
