//! 调用序列模糊测试 —— **MP4 封装器**。
//!
//! ## 为什么 MP4 不能像 ASF 那样做往返
//!
//! `crates/mp4/src/lib.rs` 只有 **mux + writer**，没有解封装器 ——
//! 仓里唯一的 ISOBMFF **读取器**住在 `tests/real_sample.rs` 里，是手搓的。
//! 所以拿不到「封 → 解 → 比字节」那个闭环，只能换一种判据：
//!
//! **随机调用序列驱动 muxer（宿主参数也随机），然后检查两件事**：
//!
//! 1. `finish()` **不许 panic** —— 允许 `Err`，但崩了整个转换就没了；
//! 2. 产出的字节里，**顶层 box 链必须自洽**：`ftyp` 打头、`moov` 收尾、
//!    每个 box 的长度字段正好铺满文件、不重不漏。
//!
//! ## 宿主参数刻意往坏里给
//!
//! `0 duration` / 空 `data` / 超大 `data` / **写没声明过的流**
//! （没建音频轨却调 `write_audio`）—— 这些是调用方（CLI / Android）真会犯的错，
//! 而它们最容易撞出「除以零」「下溢」「空切片取元素」这一类。
//!
//! ## 为什么不复用 `writer.rs`
//!
//! `real_sample.rs` 里已经记过这条理由：**共用会让两边一起错变得难以察觉**。
//! 读取器要独立于写入器，否则「写入器写错 + 读取器按同样的错读」会互相掩护。
//! 所以这里手搓一份最小读取器（三十行）。
//!
//! ## 跑法
//!
//! ```bash
//! cargo test -p v2v-mp4 --test fuzz
//! V2V_FUZZ_ROUNDS=200000 cargo test -p v2v-mp4 --release --test fuzz -- --ignored --nocapture
//! ```

#[path = "../../msmpeg4/tests/common/mutate.rs"]
mod mutate;

use mutate::Rng;
use v2v_mp4::{AudioCodec, AudioTrack, Error, Mp4Muxer, VideoTrack};

// ---------------------------------------------------------------- 独立读取器

/// 走一遍**顶层** box 链，返回 `(类型, 总长度)` 列表。
///
/// ⚠️ 这份读取器是**故意独立**于 `src/writer.rs` 的（见模块头那条理由）。
/// 它只做最保守的事：按长度字段一步一步走，走不通就报错。
fn walk_top_level(b: &[u8]) -> Result<Vec<([u8; 4], usize)>, String> {
    let mut out = Vec::new();
    let mut p = 0usize;
    while p < b.len() {
        if p + 8 > b.len() {
            return Err(format!("位置 {p} 只剩 {} 字节，装不下 box 头", b.len() - p));
        }
        let size = u32::from_be_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]) as usize;
        let kind = [b[p + 4], b[p + 5], b[p + 6], b[p + 7]];
        if size == 0 {
            // `size == 0` 表示「一直到文件尾」—— 只允许出现在最后一个 box 上。
            out.push((kind, b.len() - p));
            p = b.len();
            break;
        }
        if size < 8 {
            return Err(format!(
                "位置 {p} 的 box `{}` 长度 {size} < 8（连头都装不下）",
                show(kind)
            ));
        }
        if p + size > b.len() {
            return Err(format!(
                "位置 {p} 的 box `{}` 长度 {size} 越过了文件尾（还剩 {} 字节）",
                show(kind),
                b.len() - p
            ));
        }
        out.push((kind, size));
        p += size;
    }
    if p != b.len() {
        return Err(format!("顶层 box 链走到 {p}，文件长 {} —— 没铺满", b.len()));
    }
    Ok(out)
}

fn show(k: [u8; 4]) -> String {
    k.iter()
        .map(|&c| if (0x20..0x7f).contains(&c) { c as char } else { '.' })
        .collect()
}

// ---------------------------------------------------------------- 造参数

fn video_track() -> VideoTrack {
    VideoTrack {
        width: 32,
        height: 32,
        timescale: 12800,
        // VOL 头（`00 00 01 B0` 起）—— 内容对封装器无意义，只要非空。
        decoder_specific_info: vec![0x00, 0x00, 0x01, 0xB0, 0x01, 0x00, 0x00, 0x01],
        avg_bitrate: 500_000,
    }
}

fn audio_track() -> AudioTrack {
    AudioTrack {
        channels: 2,
        sample_rate: 44100,
        timescale: 44100,
        codec: AudioCodec::Pcm {
            bits: 16,
            big_endian: false,
        },
        avg_bitrate: 0,
        priming: 0,
    }
}

/// 一次调用的参数 —— 好坏都有。
enum Call {
    Video { len: usize, key: bool, duration: u32 },
    Audio { len: usize, duration: u32 },
}

/// 按 rng 造一次调用。**刻意有相当比例是坏的**：
/// 空数据（1/6）、时长为 0（1/6）、超大载荷（1/8）。
fn random_call(rng: &mut Rng, is_audio_track: bool) -> Call {
    let roll = rng.below(12);
    let len = match roll {
        0 | 1 => 0,                      // 空样本
        2 => 1 + rng.below(3),           // 极小
        3 => 200_000,                    // 超大（远超一个包）
        _ => 1 + rng.below(300),
    };
    let duration = if roll == 4 || roll == 5 {
        0 // 时长为 0
    } else {
        1 + rng.below(512) as u32
    };
    let audio = if is_audio_track { rng.below(3) == 0 } else { rng.below(4) == 0 };
    if audio {
        Call::Audio { len, duration }
    } else {
        Call::Video {
            len,
            key: rng.below(4) == 0,
            duration,
        }
    }
}

/// 跑一局：随机调用序列 → `finish()` → 检查 box 链。
///
/// 返回 `(成功写入的样本数, 报错次数, 产出字节数)`。
fn one_case(seed: u64, with_audio: bool, calls: usize) -> (usize, usize, usize) {
    let mut rng = Rng::new(seed);
    let mut m = if with_audio {
        Mp4Muxer::with_audio(video_track(), audio_track())
    } else {
        Mp4Muxer::new(video_track())
    };
    let mut ok = 0usize;
    let mut err = 0usize;
    for i in 0..calls {
        let call = random_call(&mut rng, with_audio);
        // ⚠️ 先打日志再调用（房规）—— 这里坐标就是「第几次调用 + 参数」。
        let desc = match &call {
            Call::Video {
                len,
                key,
                duration,
            } => format!("第 {i} 次：write_video(len={len} key={key} dur={duration})"),
            Call::Audio { len, duration } => {
                format!("第 {i} 次：write_audio(len={len} dur={duration})")
            }
        };
        eprintln!("[mp4 序列 种子 {seed:#x} 音频 {with_audio}] {desc}");
        let data = vec![0x5Au8; match &call {
            Call::Video { len, .. } | Call::Audio { len, .. } => *len,
        }];
        let r = match call {
            Call::Video {
                key, duration, ..
            } => m.write_video(&data, key, duration),
            Call::Audio { duration, .. } => m.write_audio(&data, duration),
        };
        match r {
            Ok(()) => ok += 1,
            Err(_) => err += 1,
        }
    }
    let bytes = m.finish();
    let boxes = walk_top_level(&bytes)
        .unwrap_or_else(|e| panic!("【种子 {seed:#x} 音频 {with_audio}】顶层 box 链不自洽：{e}"));

    // ---- box 链的判据 ----
    assert!(
        !boxes.is_empty(),
        "【种子 {seed:#x}】产出为空 —— 连 ftyp 都没有"
    );
    assert_eq!(
        show(boxes[0].0),
        "ftyp",
        "【种子 {seed:#x}】第一个 box 该是 ftyp，实际 {}",
        show(boxes[0].0)
    );
    assert_eq!(
        show(boxes[boxes.len() - 1].0),
        "moov",
        "【种子 {seed:#x}】最后一个 box 该是 moov（非分片 MP4 的 moov 在尾部），实际 {}",
        show(boxes[boxes.len() - 1].0)
    );
    let has_mdat = boxes.iter().any(|(k, _)| show(*k) == "mdat");
    assert!(has_mdat, "【种子 {seed:#x}】没有 mdat");
    // 长度字段之和必须正好等于文件长度（`walk_top_level` 已经保证铺满，
    // 这里再独立核一遍总和 —— 两处算错才可能同时通过）。
    let sum: usize = boxes.iter().map(|(_, n)| *n).sum();
    assert_eq!(
        sum,
        bytes.len(),
        "【种子 {seed:#x}】box 长度之和 {sum} ≠ 文件长度 {}",
        bytes.len()
    );
    (ok, err, bytes.len())
}

/// 轻局：CI / 日常。
#[test]
fn random_call_sequences_never_panic() {
    let rounds = mutate::rounds(60);
    let (mut ok, mut err) = (0usize, 0usize);
    for r in 0..rounds {
        let with_audio = r % 2 == 0;
        let calls = 1 + (r * 7) % 40;
        let (o, e, _) = one_case(0x1111_0000 + r as u64, with_audio, calls);
        ok += o;
        err += e;
    }
    eprintln!("[mp4 序列] {rounds} 局：写入成功 {ok} 次，被挡下 {err} 次");

    // ---- 覆盖断言 ----
    // 「没 panic」单独不构成证据：如果所有调用都被挡下，等于什么都没写。
    assert!(ok > 0, "所有调用都被挡下了 —— 没测到真正的写入路径");
    // 反过来也要：一次都没挡下说明「坏参数」根本没造出来，边界没测到。
    assert!(
        err > 0,
        "一次都没被挡下 —— 坏参数（空样本 / 时长 0 / 无此轨）没造出来，边界没测到"
    );
}

/// 重局：本机深挖用。
///
/// ```bash
/// V2V_FUZZ_ROUNDS=200000 cargo test -p v2v-mp4 --release --test fuzz -- --ignored --nocapture
/// ```
#[test]
#[ignore = "重局：本机深挖用，轮数由 V2V_FUZZ_ROUNDS 给（默认 2 万）"]
fn heavy_random_call_sequences_never_panic() {
    let rounds = mutate::rounds(mutate::HEAVY_ROUNDS);
    eprintln!("重局：{rounds} 局");
    for r in 0..rounds {
        one_case(0x2222_0000 + r as u64, r % 2 == 0, 1 + (r * 13) % 80);
    }
    eprintln!("{rounds} 局全部通过");
}

/// **一个样本都没写**就 `finish()` —— 最容易漏的边界。
///
/// 时长求和、`moov` 里的 `stts`/`stsz` 在「零样本」时都会退化，
/// 是最容易出「除以零」或「空切片取元素」的地方。
#[test]
fn finishing_without_any_sample_is_fine() {
    for with_audio in [false, true] {
        let m = if with_audio {
            Mp4Muxer::with_audio(video_track(), audio_track())
        } else {
            Mp4Muxer::new(video_track())
        };
        let bytes = m.finish();
        let boxes = walk_top_level(&bytes)
            .unwrap_or_else(|e| panic!("零样本的 box 链不自洽（音频 {with_audio}）：{e}"));
        assert_eq!(show(boxes[0].0), "ftyp");
        assert_eq!(show(boxes[boxes.len() - 1].0), "moov");
        assert!(bytes.len() > 100, "零样本也该产出一个结构完整的文件");
    }
}

/// **写没声明过的流**必须报错，不能默默吞掉。
///
/// 这条单独拎出来是因为它最阴：调用方以为写进去了，实际没写 ——
/// 产出文件「看起来正常」，但音频轨是空的。所以要钉住「报错」这个行为本身。
#[test]
fn writing_an_undeclared_track_is_an_error() {
    let mut m = Mp4Muxer::new(video_track()); // 只有视频轨
    let e = m.write_audio(&[1, 2, 3, 4], 1024);
    assert!(matches!(e, Err(Error::Bad(_))), "没建音频轨却写音频，该报错：{e:?}");
    // 报错之后 muxer 还得能用（不能把内部状态弄坏）。
    m.write_video(&[0xAA; 64], true, 512).expect("视频轨还能写");
    let bytes = m.finish();
    walk_top_level(&bytes).expect("报错之后产出仍要自洽");
}
