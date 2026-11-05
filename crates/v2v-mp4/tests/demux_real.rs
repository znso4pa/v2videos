//! MP4 解封装对 **`ffprobe` 的包列表**（真样本 + 独立 oracle）。
//!
//! 判据（AGENTS.md「真样本 + 独立 oracle 才算验证过」）：
//! - 每条流的包数一致；
//! - 每个包的**大小**一致（读错偏移会立刻暴露）；
//! - PTS 的**增量**一致（±1 ms）—— 不比起始值，因为 AAC 的 `edts` 编辑列表会整体
//!   平移音频时间轴，而我们**明确不解析 `edts`**（见 `demux.rs` 顶部「不做的」）；
//! - 关键帧位一致。
//!
//! 也不比**顺序**：ffprobe 按文件存储顺序列包，我们按 DTS 合并 —— 两者都合法。
//!
//! 样本按环境变量给：`V2V_MP4_SAMPLE=/path/x.mp4 cargo test -p v2v-mp4 --test demux_real -- --nocapture`
//! 不给就 `SKIP` 并通过（样本一律 `.gitignore`，见 AGENTS.md）。

use std::process::Command;

use v2v_mp4::Mp4Demuxer;

fn probe_packets(path: &str) -> Vec<(i64, i64, i64, bool)> {
    // 输出：stream_index,pts_time,size,flags
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "packet=stream_index,pts_time,size,flags",
            "-of",
            "csv=p=0",
            path,
        ])
        .output()
        .expect("跑不了 ffprobe");
    assert!(out.status.success(), "ffprobe 失败");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.trim().split(',').collect();
            if f.len() < 4 {
                return None;
            }
            let si: i64 = f[0].parse().ok()?;
            let pts: f64 = f[1].parse().ok()?;
            let size: i64 = f[2].parse().ok()?;
            let key = f[3].contains('K');
            Some((si, (pts * 1000.0).round() as i64, size, key))
        })
        .collect()
}

#[test]
fn packets_match_ffprobe() {
    let Ok(path) = std::env::var("V2V_MP4_SAMPLE") else {
        println!("SKIP: 没设 V2V_MP4_SAMPLE");
        return;
    };
    if !std::path::Path::new(&path).is_file() {
        println!("SKIP: 样本不存在 {path}");
        return;
    }

    let expected = probe_packets(&path);
    assert!(!expected.is_empty(), "ffprobe 没给出包");

    let f = std::fs::File::open(&path).unwrap();
    let mut d = Mp4Demuxer::new(std::io::BufReader::new(f)).unwrap();

    // 我们的 stream number 是 1 起的轨道序号；ffprobe 的 stream_index 是 0 起的
    // 文件内序号。两者在「只有一轨」时差 1；多轨时按出现顺序对应，这里按顺序映射。
    let nums: Vec<u8> = d.streams().iter().map(|s| s.number).collect();
    let idx_of = |n: u8| nums.iter().position(|&x| x == n).unwrap() as i64;

    let mut got = Vec::new();
    while let Some(s) = d.next_sample().unwrap() {
        got.push((idx_of(s.stream), s.timestamp_ms, s.data.len() as i64, s.key_frame));
    }

    let by_stream = |v: &[(i64, i64, i64, bool)]| {
        let mut m: std::collections::BTreeMap<i64, Vec<(i64, i64, bool)>> = Default::default();
        for &(si, pts, size, key) in v {
            m.entry(si).or_default().push((pts, size, key));
        }
        for list in m.values_mut() {
            list.sort_by_key(|x| x.0);
        }
        m
    };
    let e = by_stream(&expected);
    let g = by_stream(&got);

    assert_eq!(e.len(), g.len(), "流数不一致");
    let mut bad = 0;
    for (si, elist) in &e {
        let glist = g.get(si).unwrap_or_else(|| panic!("缺流 {si}"));
        assert_eq!(elist.len(), glist.len(), "流 {si} 的包数不一致");
        for (i, (ep, gp)) in elist.iter().zip(glist.iter()).enumerate() {
            // 大小与关键帧位必须**精确**相同
            if ep.1 != gp.1 || ep.2 != gp.2 {
                if bad < 5 {
                    println!("流 {si} 第 {i} 包：ffprobe size={} key={} / 我们 size={} key={}",
                        ep.1, ep.2, gp.1, gp.2);
                }
                bad += 1;
            }
            // PTS 比增量（跳过第一个）
            if i > 0 {
                let ed = ep.0 - elist[i - 1].0;
                let gd = gp.0 - glist[i - 1].0;
                if (ed - gd).abs() > 1 {
                    if bad < 5 {
                        println!("流 {si} 第 {i} 包 PTS 增量：ffprobe {ed}ms / 我们 {gd}ms");
                    }
                    bad += 1;
                }
            }
        }
    }
    assert_eq!(bad, 0, "{bad} 处不一致");
    println!("对账通过：{} 条流 / {} 个包，大小与时间增量与 ffprobe 一致",
        e.len(), expected.len());
}

/// 视频的 `extra`（mp4v 是 VOL 序列、avc1 是 avcC）必须与 ffprobe 的 extradata 一致。
///
/// 这条是「解码能不能跑起来」的前置：extra 错一个字节，解码器就会报
/// `invalid vlc code` 这类看起来像「解码器有 bug」的错误。
#[test]
fn video_extradata_matches_ffprobe() {
    let Ok(path) = std::env::var("V2V_MP4_SAMPLE") else {
        println!("SKIP: 没设 V2V_MP4_SAMPLE");
        return;
    };
    if !std::path::Path::new(&path).is_file() {
        println!("SKIP: 样本不存在 {path}");
        return;
    }
    let out = Command::new("ffprobe")
        .args([
            "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=extradata", "-show_data",
            "-of", "default=nw=1",
            &path,
        ])
        .output()
        .expect("跑不了 ffprobe");
    let text = String::from_utf8_lossy(&out.stdout);
    let mut want = Vec::new();
    for line in text.lines() {
        let Some((_, rest)) = line.split_once(": ") else { continue };
        // "0000 01b0 0100 ..." —— 每 4 个 hex 一组，组间空格
        for tok in rest.split_whitespace() {
            if tok.len() != 4 || !tok.chars().all(|c| c.is_ascii_hexdigit()) {
                continue;
            }
            for i in (0..4).step_by(2) {
                want.push(u8::from_str_radix(&tok[i..i + 2], 16).unwrap());
            }
        }
    }
    assert!(!want.is_empty(), "ffprobe 没给出 extradata");

    let f = std::fs::File::open(&path).unwrap();
    let d = Mp4Demuxer::new(std::io::BufReader::new(f)).unwrap();
    let v = d.streams().iter().find(|s| s.video.is_some()).unwrap();
    let got = &v.video.as_ref().unwrap().extra;

    assert_eq!(
        got, &want,
        "extra 与 ffprobe 不一致\n我们: {:02x?}\n期望: {:02x?}",
        &got[..got.len().min(48)],
        &want[..want.len().min(48)]
    );
    println!("extradata 一致（{} 字节）", got.len());
}
