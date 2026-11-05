//! 无损剪切 / 拼接的判据：**输出解出的样本与源对应区间逐字节相同**。
//!
//! 为什么不用 PSNR：PSNR 只能证明「看起来一样」，而这条路的价值恰恰是「真的没动过」。
//! 逐字节比较同时钉住了三件事 —— 载荷没被重新编码、顺序没乱、时间戳按预期重定。
//!
//! 造样本用**任意字节**当载荷：这条路不解码，载荷是什么都不影响。这样测试不依赖
//! 编码器，也不需要 ffmpeg（真样本那条另说，见文件末尾）。

use std::path::{Path, PathBuf};

use v2v_asf::{AsfDemuxer, AsfMuxer, AudioFormat, Sample, StreamKind, VideoStreamInfo};
use v2v_cli::{concat, trim};

const MP43: u32 = u32::from_le_bytes(*b"MP43");

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("v2v_lossless_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("建临时目录");
    d
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

/// 一帧视频载荷。每个下标给一段**可辨认**的字节 —— 逐字节比对才有意义
/// （全零载荷会让「搬错了帧」也通过）。
fn payload(i: usize) -> Vec<u8> {
    (0..(60 + i)).map(|k| (i.wrapping_mul(37) + k) as u8).collect()
}

/// 造一个 ASF。`frames` 是 `(载荷, 是否关键帧, pts_ms)`。
fn make_asf(
    path: &Path,
    w: u32,
    h: u32,
    fourcc: u32,
    extra: &[u8],
    frames: &[(Vec<u8>, bool, u32)],
    audio: Option<&[(Vec<u8>, u32)]>,
) {
    let mut m = AsfMuxer::with_audio(
        VideoStreamInfo {
            width: w,
            height: h,
            fourcc,
        },
        audio.map(|_| wma_audio()),
    );
    m.set_video_extra(extra.to_vec());
    // 视频与音频按时间戳交织写入 —— 与真实文件的包序一致，剪切才会走到「交织中找起点」。
    let mut v = frames.iter().peekable();
    let mut a = audio.unwrap_or(&[]).iter().peekable();
    loop {
        let take_v = match (v.peek(), a.peek()) {
            (None, None) => break,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some(x), Some(y)) => x.2 <= y.1,
        };
        if take_v {
            let (data, key, pts) = v.next().unwrap();
            m.write_video(data, *key, *pts).expect("写视频");
        } else {
            let (data, pts) = a.next().unwrap();
            m.write_audio(data, *pts).expect("写音频");
        }
    }
    let bytes = m.finish();
    std::fs::write(path, bytes).expect("落盘");
}

fn demux(path: &Path) -> Vec<Sample> {
    let mut d = AsfDemuxer::new(std::fs::File::open(path).expect("开文件")).expect("解封装");
    let mut out = Vec::new();
    while let Some(s) = d.next_sample().expect("读样本") {
        out.push(s);
    }
    out
}

fn video_of(path: &Path) -> (u8, Vec<u8>) {
    let d = AsfDemuxer::new(std::fs::File::open(path).expect("开文件")).expect("解封装");
    let v = d
        .streams()
        .iter()
        .find(|s| s.kind == StreamKind::Video)
        .expect("视频轨");
    (v.number, v.video.as_ref().expect("视频格式").extra.clone())
}

/// 30 帧、每帧 40 ms、关键帧在 0 / 400 / 800 ms。
fn frames_30() -> Vec<(Vec<u8>, bool, u32)> {
    (0..30)
        .map(|i| (payload(i), i % 10 == 0, i as u32 * 40))
        .collect()
}

// ==================== 剪切 ====================

/// **主判据**：从关键帧起剪出来的视频载荷与源**逐字节相同**，时间戳从 0 重定。
#[test]
fn trim_keeps_video_payloads_byte_identical() {
    let dir = temp_dir("trim_bytes");
    let src = dir.join("in.wmv");
    let out = dir.join("out.wmv");
    let frames = frames_30();
    make_asf(&src, 32, 32, MP43, b"", &frames, None);

    // 400 ms 正好是第 10 帧（关键帧）。
    let r = trim(&src, &out, 400, Some(800)).expect("剪切");
    assert_eq!(r.actual_start_ms, 400, "起点不该偏移");
    assert_eq!(r.video_samples, 10, "400..800 ms 应该有 10 帧");

    let got: Vec<_> = demux(&out)
        .into_iter()
        .filter(|s| s.stream == 1)
        .collect();
    assert_eq!(got.len(), 10);
    for (k, s) in got.iter().enumerate() {
        assert_eq!(
            s.data,
            frames[10 + k].0,
            "第 {k} 帧的载荷与源的第 {} 帧不同 —— 要么搬错了帧，要么被重编码了",
            10 + k
        );
        assert_eq!(
            s.timestamp_ms,
            k as i64 * 40,
            "第 {k} 帧的时间戳没有从 0 重定"
        );
        assert_eq!(s.key_frame, frames[10 + k].1, "第 {k} 帧的关键帧标志不对");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// 起点不在关键帧上：**向后**对齐，并如实报出实际起点。
#[test]
fn trim_snaps_forward_to_a_keyframe() {
    let dir = temp_dir("trim_snap");
    let src = dir.join("in.wmv");
    let out = dir.join("out.wmv");
    let frames = frames_30();
    make_asf(&src, 32, 32, MP43, b"", &frames, None);

    // 500 ms 落在第 12 帧（不是关键帧），后面最近的关键帧是 800 ms。
    let r = trim(&src, &out, 500, None).expect("剪切");
    assert_eq!(r.actual_start_ms, 800, "应当向后对齐到 800 ms 那个关键帧");
    let got: Vec<_> = demux(&out).into_iter().filter(|s| s.stream == 1).collect();
    assert_eq!(got[0].data, frames[20].0, "首帧应当是源的第 20 帧");
    assert!(got[0].key_frame, "剪出来的第一帧必须是关键帧");
    let _ = std::fs::remove_dir_all(&dir);
}

/// 编解码器私有数据必须原样带走。
///
/// 没有它，WMV2 的能力位、VC-1 的 `STRUCT_C` 就丢了 —— 产物「结构能读、画面解不出来」。
#[test]
fn trim_carries_codec_private_data() {
    let dir = temp_dir("trim_extra");
    let src = dir.join("in.wmv");
    let out = dir.join("out.wmv");
    // 一段不规整的字节（不是全零 —— 全零看不出「长度对但内容错」）。
    let extra: Vec<u8> = (0..37u8).map(|b| b.wrapping_mul(7)).collect();
    make_asf(&src, 32, 32, MP43, &extra, &frames_30(), None);
    assert_eq!(video_of(&src).1, extra, "造样本这一步就该把私有数据写进去");

    trim(&src, &out, 0, Some(400)).expect("剪切");
    assert_eq!(video_of(&out).1, extra, "产物丢了编解码器私有数据");
    let _ = std::fs::remove_dir_all(&dir);
}

/// 音频按**时间戳**裁，且载荷逐字节相同。
#[test]
fn trim_audio_follows_the_timestamp_window() {
    let dir = temp_dir("trim_audio");
    let src = dir.join("in.wmv");
    let out = dir.join("out.wmv");
    let frames = frames_30();
    // 每 100 ms 一个音频载荷，共 12 个（0..1100 ms）。
    let audio: Vec<(Vec<u8>, u32)> = (0..12)
        .map(|i| {
            let n = 300 + i;
            (vec![(i as u8).wrapping_add(0x80); n], i as u32 * 100)
        })
        .collect();
    make_asf(&src, 32, 32, MP43, b"", &frames, Some(&audio));

    let r = trim(&src, &out, 400, Some(800)).expect("剪切");
    let got: Vec<_> = demux(&out).into_iter().filter(|s| s.stream == 2).collect();
    assert_eq!(
        got.len(),
        r.audio_samples,
        "报告里的音频样本数与文件里的不一致"
    );
    assert!(!got.is_empty(), "400..800 ms 之间应该有音频");
    for s in &got {
        assert!(
            (0..400).contains(&s.timestamp_ms),
            "音频时间戳 {} ms 落在区间外（应当已重定到 0 起）",
            s.timestamp_ms
        );
        // 载荷必须与源里同一时刻的那个一模一样。
        let src_pts = s.timestamp_ms as u32 + 400;
        let want = audio
            .iter()
            .find(|(_, p)| *p == src_pts)
            .map(|(d, _)| d)
            .expect("源里应当有这个时间戳的音频");
        assert_eq!(&s.data, want, "音频载荷被改动了");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// 起点之后没有关键帧 → 明确报错，不产出空文件。
#[test]
fn trim_errors_when_no_keyframe_after_start() {
    let dir = temp_dir("trim_nokey");
    let src = dir.join("in.wmv");
    let out = dir.join("out.wmv");
    make_asf(&src, 32, 32, MP43, b"", &frames_30(), None);
    let err = trim(&src, &out, 100_000, None).expect_err("起点之后没有关键帧，应当报错");
    assert!(
        err.to_string().contains("关键帧"),
        "错误信息要让人知道是「没有关键帧」：{err}"
    );
    assert!(!out.exists(), "报错时不该留下文件");
    let _ = std::fs::remove_dir_all(&dir);
}

// ==================== 拼接 ====================

/// **主判据**：拼接产物的视频载荷 = 各段载荷首尾相接，逐字节相同。
#[test]
fn concat_is_a_byte_identical_concatenation() {
    let dir = temp_dir("concat_bytes");
    let a = dir.join("a.wmv");
    let b = dir.join("b.wmv");
    let out = dir.join("ab.wmv");
    // 两段都是 5 帧、关键帧只在第 0 帧（拼接要求每段首帧是关键帧）。
    let fa: Vec<_> = (0..5)
        .map(|i| (payload(i), i == 0, i as u32 * 40))
        .collect();
    let fb: Vec<_> = (0..4)
        .map(|i| (payload(100 + i), i == 0, i as u32 * 40))
        .collect();
    make_asf(&a, 32, 32, MP43, b"", &fa, None);
    make_asf(&b, 32, 32, MP43, b"", &fb, None);

    let r = concat(&[a, b], &out).expect("拼接");
    assert_eq!(r.segments, 2);
    assert_eq!(r.video_samples, 9, "5 + 4 帧");

    let got: Vec<_> = demux(&out).into_iter().filter(|s| s.stream == 1).collect();
    assert_eq!(got.len(), 9);
    for (k, s) in got.iter().enumerate() {
        let want = if k < 5 { &fa[k].0 } else { &fb[k - 5].0 };
        assert_eq!(&s.data, want, "第 {k} 帧的载荷不对");
    }
    // 时间戳必须严格递增 —— 接缝处重合会让播放器丢掉一段。
    for w in got.windows(2) {
        assert!(
            w[1].timestamp_ms > w[0].timestamp_ms,
            "时间戳没有严格递增：{} → {}",
            w[0].timestamp_ms,
            w[1].timestamp_ms
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// 第二段的尺寸不同 → 明确报错。
#[test]
fn concat_rejects_mismatched_geometry() {
    let dir = temp_dir("concat_geo");
    let a = dir.join("a.wmv");
    let b = dir.join("b.wmv");
    let f: Vec<_> = (0..3).map(|i| (payload(i), i == 0, i as u32 * 40)).collect();
    make_asf(&a, 32, 32, MP43, b"", &f, None);
    make_asf(&b, 48, 32, MP43, b"", &f, None);
    let err = concat(&[a, b], &dir.join("ab.wmv")).expect_err("尺寸不同应当报错");
    assert!(err.to_string().contains("尺寸"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// 私有数据不同 → 明确报错（同一 FourCC 也可能参数不同）。
#[test]
fn concat_rejects_mismatched_codec_private_data() {
    let dir = temp_dir("concat_extra");
    let a = dir.join("a.wmv");
    let b = dir.join("b.wmv");
    let f: Vec<_> = (0..3).map(|i| (payload(i), i == 0, i as u32 * 40)).collect();
    make_asf(&a, 32, 32, MP43, b"AAAA", &f, None);
    make_asf(&b, 32, 32, MP43, b"BBBB", &f, None);
    let err = concat(&[a, b], &dir.join("ab.wmv")).expect_err("私有数据不同应当报错");
    assert!(err.to_string().contains("私有数据"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// 第一段有音频、第二段没有 → 明确报错（一个音频流属性描述不了两种）。
#[test]
fn concat_rejects_audio_presence_mismatch() {
    let dir = temp_dir("concat_audio");
    let a = dir.join("a.wmv");
    let b = dir.join("b.wmv");
    let f: Vec<_> = (0..3).map(|i| (payload(i), i == 0, i as u32 * 40)).collect();
    let audio: Vec<(Vec<u8>, u32)> = vec![(vec![1u8; 200], 0), (vec![2u8; 200], 100)];
    make_asf(&a, 32, 32, MP43, b"", &f, Some(&audio));
    make_asf(&b, 32, 32, MP43, b"", &f, None);
    let err = concat(&[a, b], &dir.join("ab.wmv")).expect_err("音频有无不一致应当报错");
    assert!(err.to_string().contains("音频"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// 某段的第一个视频样本不是关键帧 → 明确报错。
///
/// 这一条是「宁可拒绝也不产出坏文件」的具体体现：从 P 帧起解，接缝之后一直到
/// 下一个关键帧全是花的，而帧数、时长、文件结构全都正常 —— 不逐帧看根本发现不了。
#[test]
fn concat_rejects_a_segment_that_does_not_start_on_a_keyframe() {
    let dir = temp_dir("concat_nokey");
    let a = dir.join("a.wmv");
    let b = dir.join("b.wmv");
    let fa: Vec<_> = (0..3).map(|i| (payload(i), i == 0, i as u32 * 40)).collect();
    // 第 0 帧不是关键帧。
    let fb: Vec<_> = (0..3).map(|i| (payload(50 + i), false, i as u32 * 40)).collect();
    make_asf(&a, 32, 32, MP43, b"", &fa, None);
    make_asf(&b, 32, 32, MP43, b"", &fb, None);
    let err = concat(&[a, b], &dir.join("ab.wmv")).expect_err("首帧不是关键帧应当报错");
    assert!(err.to_string().contains("关键帧"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// 只给一个文件 → 明确报错（那件事叫剪切）。
#[test]
fn concat_needs_at_least_two_inputs() {
    let dir = temp_dir("concat_one");
    let a = dir.join("a.wmv");
    make_asf(&a, 32, 32, MP43, b"", &frames_30(), None);
    let err = concat(&[a], &dir.join("o.wmv")).expect_err("一个文件不该当拼接");
    assert!(err.to_string().contains("至少"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

// ==================== 真样本（默认跳过）====================

/// 真 VC-1 样本端到端：剪切 → 交 **ffmpeg** 解码 → 与源对应区间的画面**逐帧逐字节**比。
///
/// ```text
/// V2V_ASF_SAMPLE=<样本> cargo test -p v2v-cli --test lossless -- --ignored --nocapture
/// ```
///
/// 为什么这条必须走 ffmpeg：合成样本证明的是「我们搬对了自己的载荷」，证明不了
/// 「搬出来的东西还是标准 ASF」。而私有数据（VC-1 的序列头）到底有没有搬对，
/// 只有外部解码器能回答 —— 少了它 ffmpeg 会直接解不出画面。
#[test]
#[ignore = "需要真样本与 ffmpeg"]
fn trim_real_sample_decodes_identically_in_ffmpeg() {
    let src = match std::env::var("V2V_ASF_SAMPLE") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("SKIP: V2V_ASF_SAMPLE 未设置");
            return;
        }
    };
    if std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| !s.success())
        .unwrap_or(true)
    {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }
    let dir = temp_dir("trim_real");
    let out = dir.join("cut.wmv");
    let (start_s, end_s) = (10u64, 15u64);
    let r = trim(&src, &out, start_s * 1000, Some(end_s * 1000)).expect("剪切");
    eprintln!(
        "剪切：{} 帧 / {} 音频载荷，区间 {}~{} ms",
        r.video_samples, r.audio_samples, r.actual_start_ms, r.end_ms
    );

    // 源对应区间（用**实际起点**，不是请求值 —— 起点可能被对齐过）。
    //
    // ⚠️ 终点用**请求的** `end_s`，不用 `r.end_ms`：后者是「最后一个写进去的帧的时间戳」，
    // 是含的；拿它当半开区间的右端会把最后一帧排除掉，于是「我们的产物多一帧」。
    // 实测就是这么红的：150 帧对 149 帧。
    let raw = |args: &[&str]| -> Vec<u8> {
        let o = std::process::Command::new("ffmpeg")
            .args(args)
            .output()
            .expect("跑 ffmpeg");
        assert!(o.status.success(), "ffmpeg 失败：{}", String::from_utf8_lossy(&o.stderr));
        o.stdout
    };
    let want = raw(&[
        "-v", "error",
        "-ss", &format!("{:.3}", r.actual_start_ms as f64 / 1000.0),
        "-t", &format!("{:.3}", (end_s * 1000 - r.actual_start_ms) as f64 / 1000.0),
        "-i", src.to_str().unwrap(),
        "-map", "0:v", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-",
    ]);
    let got = raw(&[
        "-v", "error", "-i", out.to_str().unwrap(),
        "-map", "0:v", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-",
    ]);
    assert_eq!(
        got.len(),
        want.len(),
        "解出来的画面字节数不同（{} vs {}）—— 帧数或尺寸对不上",
        got.len(),
        want.len()
    );
    assert_eq!(got, want, "画面不是逐字节相同 —— 这就不是「无损」了");

    let _ = std::fs::remove_dir_all(&dir);
}
