//! ASF **封装 ↔ 解封装往返**：这是全仓唯一能做到「封装器与解封装器互证」的容器。
//!
//! ## 为什么只有 ASF 能做这件事
//!
//! `AsfMuxer` 与 `AsfDemuxer` 都是**纯内存**的（前者吐 `Vec<u8>`，后者吃
//! `impl Read`），所以可以「随机造一份 → 封 → 解 → 逐样本比字节」闭环。
//! MP4 那边只有封装器（仓里唯一的 ISOBMFF **读取器**在手搓的测试里），
//! AVI 只有解封装器 —— 都凑不出这个环。
//!
//! ## 这一条为什么比「没报错」强
//!
//! 往返断言的是 **`assert_eq!(解出来的 data, 封进去的 data)`** ——
//! 不是「跑通了」。容器只该改变「怎么装」，不该改变「装的是什么」，
//! 所以任何一处长度/偏移/分片算错都会在这里变成字节不等。
//!
//! 刻意覆盖三条最容易错的路：
//!
//! | 路 | 为什么容易错 |
//! |---|---|
//! | **样本大于包容量** | 走 `frag_len1` 分片：同一帧切成多段，媒体对象号相同、偏移递增 —— 解封装端要按偏移重组 |
//! | **跨多秒** | Simple Index 的槽位要按秒推进，索引与数据包要对得上 |
//! | **音频直通** | 两条流交替写，媒体对象号各自计数（视频每帧 +1、音频独立） |
//! | **单帧文件** | 只有一帧、没有索引槽、`maximum_packet` 等统计量退化的边界 |
//!
//! 随机序列由**固定种子**驱动（splitmix64，与模糊测试共用一份）——
//! 失败能原样复现，而不是「跑十次红一次」。

#[path = "../../msmpeg4/tests/common/mutate.rs"]
mod mutate;

use std::io::Cursor;

use mutate::Rng;
use v2v_asf::{AsfDemuxer, AsfMuxer, AudioFormat, Sample, VideoStreamInfo};
use v2v_container::StreamKind;

const W: u32 = 32;
const H: u32 = 32;

/// 一次往返的输入：流配置 + 样本序列。
struct Case {
    name: String,
    video: VideoStreamInfo,
    audio: Option<AudioFormat>,
    /// 逐样本 `(是视频, 数据, 关键帧, 时间戳 ms)` —— 按**写入顺序**（即解码序）。
    samples: Vec<(bool, Vec<u8>, bool, u32)>,
}

fn wma_audio() -> AudioFormat {
    AudioFormat {
        format_tag: 0x0161, // WMA v2
        channels: 2,
        sample_rate: 44100,
        avg_bytes_per_sec: 16000,
        block_align: 2048,
        bits_per_sample: 16,
        extra: vec![0u8; 10],
    }
}

/// 随机造一局。
///
/// `shape` 决定这一局专攻哪条路 —— 随机是**在形状之内**随机，
/// 不然「样本大于包容量」这种边界会被均匀分布稀释掉，几千轮也撞不到几次。
fn random_case(rng: &mut Rng, shape: usize, n: usize) -> Case {
    // 视频载荷大小的分布按形状给：
    //   0 = 单帧；1 = 一律小于包容量（走多载荷打包）；2 = 一律大于包容量（走分片）
    let pick_len = |rng: &mut Rng| -> usize {
        match shape {
            0 => 1 + rng.below(50),
            1 => 1 + rng.below(200),          // 远小于 PACKET_SIZE
            _ => 3000 + rng.below(3000),      // 大于 PACKET_SIZE → 必然分片
        }
    };
    let with_audio = shape != 0 && rng.below(2) == 0;
    let video = VideoStreamInfo {
        width: W,
        height: H,
        fourcc: u32::from_le_bytes(*b"MP43"),
    };

    let mut samples = Vec::with_capacity(n);
    let mut ts = 0u32;
    for i in 0..n {
        let len = pick_len(rng);
        // 载荷内容**逐字节可辨认**：任何一处错位/丢字节都会在 assert_eq! 里现形。
        // 用 `i` 与位置混合，避免「全 0 载荷」这种看不出错位的退化情形。
        let data: Vec<u8> = (0..len)
            .map(|k| (i as u8).wrapping_mul(31).wrapping_add((k % 251) as u8))
            .collect();
        samples.push((true, data, i % 5 == 0, ts));
        ts += 40;
        if with_audio && i % 3 == 0 {
            let alen = 2048;
            let adata: Vec<u8> = (0..alen).map(|k| (k as u8).wrapping_add(i as u8)).collect();
            samples.push((false, adata, true, ts));
        }
    }
    Case {
        name: format!("形状 {shape} / {n} 个样本 / 音频 {}", if with_audio { "有" } else { "无" }),
        video,
        audio: with_audio.then(wma_audio),
        samples,
    }
}

/// 封一份，再解回来，逐样本比。
fn roundtrip(case: &Case) -> Result<Vec<Sample>, String> {
    let mut m = match case.audio.clone() {
        Some(a) => AsfMuxer::with_audio(case.video.clone(), Some(a)),
        None => AsfMuxer::new(case.video.clone()),
    };
    for (is_video, data, key, ts) in &case.samples {
        let r = if *is_video {
            m.write_video(data, *key, *ts)
        } else {
            m.write_audio(data, *ts)
        };
        r.map_err(|e| format!("写载荷失败（ts={ts} len={}）：{e}", data.len()))?;
    }
    let bytes = m.finish();

    let mut d = AsfDemuxer::new(Cursor::new(&bytes)).map_err(|e| format!("解封装失败：{e}"))?;
    let mut got = Vec::new();
    loop {
        match d.next_sample() {
            Ok(Some(s)) => got.push(s),
            Ok(None) => break,
            Err(e) => return Err(format!("读样本失败（已读 {} 个）：{e}", got.len())),
        }
        if got.len() > case.samples.len() + 8 {
            return Err(format!(
                "解出来的样本数（>{}）超过封进去的 {} 个 —— 解封装器在重复吐样本",
                got.len(),
                case.samples.len()
            ));
        }
    }
    Ok(got)
}

/// 逐样本比字节 —— **这条才是判据**。
fn check(case: &Case, got: &[Sample]) {
    let want: Vec<&(bool, Vec<u8>, bool, u32)> =
        case.samples.iter().filter(|(is_v, _, _, _)| *is_v).collect();
    let want_audio = case.samples.iter().filter(|(is_v, _, _, _)| !*is_v).count();

    let got_v: Vec<&Sample> = got.iter().filter(|s| s.stream == 1).collect();
    let got_a: Vec<&Sample> = got.iter().filter(|s| s.stream != 1).collect();

    assert_eq!(
        got_v.len(),
        want.len(),
        "【{}】视频样本数对不上：封进去 {} 个，解出来 {} 个",
        case.name,
        want.len(),
        got_v.len()
    );
    assert_eq!(
        got_a.len(),
        want_audio,
        "【{}】音频样本数对不上：封进去 {want_audio} 个，解出来 {} 个",
        case.name,
        got_a.len()
    );

    for (i, (s, w)) in got_v.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            s.data,
            w.1,
            "【{}】第 {i} 个视频样本**字节不等**（封进去 {} 字节，解出来 {} 字节）",
            case.name,
            w.1.len(),
            s.data.len()
        );
        assert_eq!(
            s.key_frame, w.2,
            "【{}】第 {i} 个视频样本的关键帧标志丢了",
            case.name
        );
        assert_eq!(
            s.timestamp_ms, w.3 as i64,
            "【{}】第 {i} 个视频样本的时间戳对不上",
            case.name
        );
    }
}

/// **随机序列往返**：四种形状各跑若干局，逐样本比字节。
#[test]
fn random_streams_survive_a_roundtrip_byte_for_byte() {
    let rounds = mutate::rounds(24);
    let mut rng = Rng::new(0xA5F0_5EED_0000_0001);
    let mut total_samples = 0usize;
    for shape in 0..3usize {
        for r in 0..rounds {
            let n = 1 + rng.below(14);
            let case = random_case(&mut rng, shape, n);
            let got = roundtrip(&case).unwrap_or_else(|e| panic!("【{}】{e}", case.name));
            check(&case, &got);
            total_samples += case.samples.len();
            let _ = r;
        }
    }
    eprintln!(
        "往返：{} 局（3 种形状 × {rounds} 轮），共 {} 个样本逐字节相同",
        rounds * 3,
        total_samples
    );
    assert!(total_samples > 0, "一局都没跑起来");
}

/// **单帧文件** —— 边界：只有一帧、没有索引槽、统计量都退化。
#[test]
fn a_single_frame_file_roundtrips() {
    let case = Case {
        name: "单帧".to_string(),
        video: VideoStreamInfo {
            width: W,
            height: H,
            fourcc: u32::from_le_bytes(*b"MP43"),
        },
        audio: None,
        samples: vec![(true, vec![0x11, 0x22, 0x33], true, 0)],
    };
    let got = roundtrip(&case).expect("单帧往返");
    check(&case, &got);
    assert_eq!(got.len(), 1);
}

/// **样本远大于包容量** —— 强制走 `frag_len1` 分片那条路。
///
/// 这一条单独拎出来是因为它最容易错、又最不容易随机撞到：
/// 一帧被切成多段写进不同数据包，解封装端要按「媒体对象号相同、偏移递增」
/// 重组。偏移差一个字节，比出来的就是错位的数据。
#[test]
fn a_frame_larger_than_a_packet_roundtrips() {
    for len in [3000usize, 6000, 20_000] {
        let data: Vec<u8> = (0..len).map(|k| (k % 253) as u8).collect();
        let case = Case {
            name: format!("分片 {len} 字节"),
            video: VideoStreamInfo {
                width: W,
                height: H,
                fourcc: u32::from_le_bytes(*b"MP43"),
            },
            audio: None,
            samples: vec![(true, data, true, 0), (true, vec![7u8; 40], false, 40)],
        };
        let got = roundtrip(&case).unwrap_or_else(|e| panic!("【{}】{e}", case.name));
        check(&case, &got);
    }
}

/// **跨多秒** —— 走 Simple Index：每 1 秒一个 seek 槽，索引要与数据包对得上。
#[test]
fn a_multi_second_file_roundtrips() {
    // 25fps、每帧 40ms：5 秒 = 125 帧。
    let samples: Vec<(bool, Vec<u8>, bool, u32)> = (0..125)
        .map(|i| {
            let data: Vec<u8> = (0..80).map(|k| (i as u8).wrapping_add(k as u8)).collect();
            (true, data, i % 25 == 0, i * 40)
        })
        .collect();
    let case = Case {
        name: "跨 5 秒".to_string(),
        video: VideoStreamInfo {
            width: W,
            height: H,
            fourcc: u32::from_le_bytes(*b"MP43"),
        },
        audio: None,
        samples,
    };
    let got = roundtrip(&case).expect("多秒往返");
    check(&case, &got);
    assert_eq!(got.len(), 125);
}

/// **音频直通**：两条流交替写，媒体对象号各自计数。
#[test]
fn audio_passthrough_roundtrips() {
    let mut samples = Vec::new();
    let mut ts = 0u32;
    for i in 0..10u32 {
        samples.push((true, vec![0xA0u8 + i as u8; 120], i % 3 == 0, ts));
        samples.push((false, vec![0xB0u8 + i as u8; 2048], true, ts));
        ts += 40;
    }
    let case = Case {
        name: "音视频交替".to_string(),
        video: VideoStreamInfo {
            width: W,
            height: H,
            fourcc: u32::from_le_bytes(*b"MP43"),
        },
        audio: Some(wma_audio()),
        samples,
    };
    let got = roundtrip(&case).expect("音视频往返");
    check(&case, &got);

    // 样本字节对了还不够 —— **流的类型**也要对：视频恒是流 1 且被认成视频，
    // 音频是另一条且被认成音频。只比字节的话，「两条流都被当成视频」
    // 这种错也能蒙混过去。
    let bytes = {
        let mut m = AsfMuxer::with_audio(case.video.clone(), case.audio.clone());
        for (is_v, data, key, ts) in &case.samples {
            if *is_v {
                m.write_video(data, *key, *ts).expect("写视频");
            } else {
                m.write_audio(data, *ts).expect("写音频");
            }
        }
        m.finish()
    };
    let d = AsfDemuxer::new(Cursor::new(&bytes)).expect("解封装");
    let kinds: Vec<(u8, StreamKind)> = d
        .streams()
        .iter()
        .map(|s| (s.number, s.kind))
        .collect();
    eprintln!("流：{kinds:?}");
    assert_eq!(kinds.len(), 2, "应当正好两条流，实际 {kinds:?}");
    assert_eq!(kinds[0], (1, StreamKind::Video), "第 1 条该是流 1 / 视频");
    assert_eq!(kinds[1].1, StreamKind::Audio, "第 2 条该被认成音频");
}
