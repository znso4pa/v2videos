//! 真样本 + 独立 oracle（`ffprobe`）。
//!
//! 按 AGENTS.md：**自己造、自己读的往返测试必要但不充分** —— 它和它编码的假设会一起错。
//! 这里把解封装结果对着 `ffprobe` 逐项比：流属性、包数、关键帧数、首末时间戳。
//!
//! 样本不入库（版权 + 体积）。设 `V2V_ASF_SAMPLE=<路径>` 才会真跑；缺样本或缺 `ffprobe`
//! 就打 `SKIP …` 并通过 —— 新 clone 的 `cargo test` 不该因为缺一个 120 MB 的样本就红。

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

use v2v_asf::{fourcc_string, AsfDemuxer, StreamKind};

fn have_ffprobe() -> bool {
    Command::new("ffprobe")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn ffprobe(args: &[&str], path: &str) -> String {
    let out = Command::new("ffprobe")
        .args(args)
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(out.status.success(), "ffprobe failed: {args:?}");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// `-of default=noprint_wrappers=1` 的输出，按 `[STREAM]` 切块，每块是一组 `k=v`。
fn parse_ffprobe_streams(text: &str) -> Vec<HashMap<String, String>> {
    let mut blocks = Vec::new();
    let mut cur: Option<HashMap<String, String>> = None;
    for line in text.lines() {
        let line = line.trim();
        if line == "[STREAM]" {
            cur = Some(HashMap::new());
        } else if line == "[/STREAM]" {
            if let Some(b) = cur.take() {
                blocks.push(b);
            }
        } else if let Some(b) = cur.as_mut() {
            if let Some((k, v)) = line.split_once('=') {
                b.insert(k.to_string(), v.to_string());
            }
        }
    }
    blocks
}

fn count_packets(path: &str, stream: &str) -> (u64, u64) {
    let text = ffprobe(
        &[
            "-v",
            "error",
            "-select_streams",
            stream,
            "-show_entries",
            "packet=flags",
            "-of",
            "csv=p=0",
        ],
        path,
    );
    let mut total = 0u64;
    let mut key = 0u64;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        total += 1;
        if line.contains('K') {
            key += 1;
        }
    }
    (total, key)
}

/// 常见 ASF 音频 wFormatTag → ffprobe 的 codec_name。表里没有就跳过名字这一项检查。
fn audio_codec_name(tag: u16) -> Option<&'static str> {
    Some(match tag {
        0x0001 => "pcm_s16le",
        0x0055 => "mp3",
        0x0160 => "wmav1",
        0x0161 => "wmav2",
        0x0162 => "wmapro",
        0x0163 => "wmalossless",
        _ => return None,
    })
}

#[test]
fn matches_ffprobe_on_a_real_sample() {
    let path = match std::env::var("V2V_ASF_SAMPLE") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SKIP matches_ffprobe_on_a_real_sample: V2V_ASF_SAMPLE 未设置");
            return;
        }
    };
    if !Path::new(&path).is_file() {
        eprintln!("SKIP matches_ffprobe_on_a_real_sample: 样本不存在 {path}");
        return;
    }
    if !have_ffprobe() {
        eprintln!("SKIP matches_ffprobe_on_a_real_sample: 没有 ffprobe");
        return;
    }

    // ---- 我们这边 ----
    let mut d = AsfDemuxer::new(BufReader::new(File::open(&path).expect("open sample")))
        .expect("parse asf header");
    let kinds: HashMap<u8, StreamKind> = d.streams().iter().map(|s| (s.number, s.kind)).collect();

    let mut v_packets = 0u64;
    let mut v_keys = 0u64;
    let mut a_packets = 0u64;
    let mut v_first = i64::MAX;
    let mut v_last = i64::MIN;
    while let Some(s) = d.next_sample().expect("demux") {
        match kinds.get(&s.stream) {
            Some(StreamKind::Video) => {
                v_packets += 1;
                if s.key_frame {
                    v_keys += 1;
                }
                v_first = v_first.min(s.timestamp_ms);
                v_last = v_last.max(s.timestamp_ms);
            }
            Some(StreamKind::Audio) => a_packets += 1,
            _ => {}
        }
    }

    // ---- ffprobe 那边 ----
    let streams = parse_ffprobe_streams(&ffprobe(
        &[
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,codec_tag_string,width,height,channels,sample_rate",
            "-of",
            "default",
        ],
        &path,
    ));

    let probe_video = streams
        .iter()
        .find(|s| s.get("codec_type").map(String::as_str) == Some("video"))
        .expect("ffprobe 报告的视频流");
    let probe_audio = streams
        .iter()
        .find(|s| s.get("codec_type").map(String::as_str) == Some("audio"))
        .expect("ffprobe 报告的音频流");

    // 流数一致。
    assert_eq!(
        streams.len(),
        kinds.len(),
        "流数量不一致：我们 {}，ffprobe {}",
        kinds.len(),
        streams.len()
    );

    // 视频：FourCC + 宽高。
    let ours_video = d
        .streams()
        .iter()
        .find_map(|s| s.video.as_ref())
        .expect("我们的视频流");
    assert_eq!(
        fourcc_string(ours_video.fourcc),
        probe_video["codec_tag_string"],
        "视频 FourCC 不一致"
    );
    assert_eq!(
        ours_video.width.to_string(),
        probe_video["width"],
        "视频宽不一致"
    );
    assert_eq!(
        ours_video.height.to_string(),
        probe_video["height"],
        "视频高不一致"
    );

    // 音频：声道 + 采样率 + （表内）codec 名。
    let ours_audio = d
        .streams()
        .iter()
        .find_map(|s| s.audio.as_ref())
        .expect("我们的音频流");
    assert_eq!(
        ours_audio.channels.to_string(),
        probe_audio["channels"],
        "音频声道不一致"
    );
    assert_eq!(
        ours_audio.sample_rate.to_string(),
        probe_audio["sample_rate"],
        "音频采样率不一致"
    );
    if let Some(name) = audio_codec_name(ours_audio.format_tag) {
        assert_eq!(name, probe_audio["codec_name"], "音频 codec 名不一致");
    }

    // 包数 / 关键帧数。
    let (pv, pk) = count_packets(&path, "v");
    let (pa, _) = count_packets(&path, "a");
    assert_eq!(v_packets, pv, "视频包数不一致");
    assert_eq!(v_keys, pk, "视频关键帧数不一致");
    assert_eq!(a_packets, pa, "音频包数不一致");

    // 首末时间戳（毫秒）。
    let pts = ffprobe(
        &[
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
        ],
        &path,
    );
    let times: Vec<f64> = pts
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let (first, last) = (times[0], times[times.len() - 1]);
    assert!(
        (v_first as f64 / 1000.0 - first).abs() < 0.001,
        "首帧 PTS 不一致：我们 {} ms，ffprobe {} s",
        v_first,
        first
    );
    assert!(
        (v_last as f64 / 1000.0 - last).abs() < 0.001,
        "末帧 PTS 不一致：我们 {} ms，ffprobe {} s",
        v_last,
        last
    );

    eprintln!(
        "对账通过：video {v_packets} 包 / {v_keys} 关键帧，audio {a_packets} 包，PTS {v_first}..{v_last} ms"
    );
}
