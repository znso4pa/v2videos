//! 扫描器的判据：**说对了什么**、**坏文件不崩**、**一千个文件够快**。
//!
//! 这三条对应 TODO 里 C 段的三个要求。第一条用合成样本钉（容器 / 编码 / 分辨率 /
//! 时长都能自己造），第二条用截断与改字节，第三条数时间。
//!
//! 为什么不拿「能不能真转」当判据：扫描器**只读头部**，它给出的结论本来就是
//! 「容器声称的编解码器我们有解码器」。拿「转得出来」去要求它，等于要求它解码 ——
//! 那就不是扫描器了。

use std::path::{Path, PathBuf};
use std::time::Instant;

use v2v_asf::{AsfDemuxer, AsfMuxer, AudioFormat, StreamKind, VideoStreamInfo};
use v2v_cli::scan::{format_table, format_tsv, scan, Verdict};

const MP43: u32 = u32::from_le_bytes(*b"MP43");
const FLV1: u32 = u32::from_le_bytes(*b"FLV1");

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("v2v_scan_{}_{}", std::process::id(), tag));
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

fn payload(i: usize) -> Vec<u8> {
    (0..(60 + i)).map(|k| (i.wrapping_mul(37) + k) as u8).collect()
}

/// 造一个 ASF。`frames` 是 `(载荷, 是否关键帧, pts_ms)`。
fn make_asf(
    path: &Path,
    w: u32,
    h: u32,
    fourcc: u32,
    frames: &[(Vec<u8>, bool, u32)],
    audio: Option<AudioFormat>,
) {
    let has_audio = audio.is_some();
    let mut m = AsfMuxer::with_audio(
        VideoStreamInfo {
            width: w,
            height: h,
            fourcc,
        },
        audio,
    );
    for (i, (data, key, pts)) in frames.iter().enumerate() {
        m.write_video(data, *key, *pts).expect("写视频");
        // 音频与视频交替写 —— 真实文件就是交错的，扫描器要能在交织里认出两条流。
        if has_audio && i % 2 == 0 {
            m.write_audio(&payload(i), *pts).expect("写音频");
        }
    }
    std::fs::write(path, m.finish()).expect("落盘");
}

/// 10 帧、每帧 40 ms 的「视频」。
fn frames_10() -> Vec<(Vec<u8>, bool, u32)> {
    (0..10)
        .map(|i| (payload(i), i == 0, i as u32 * 40))
        .collect()
}

#[test]
fn asf_reports_container_codec_size_and_duration() {
    let d = temp_dir("asf");
    let p = d.join("a.wmv");
    make_asf(&p, 640, 360, MP43, &frames_10(), Some(wma_audio()));

    let entries = scan(&[p.clone()]).expect("扫");
    assert_eq!(entries.len(), 1);
    let e = &entries[0];

    assert_eq!(e.container, Some("ASF"));
    let v = e.video.as_ref().expect("要有视频流");
    assert_eq!(v.codec, "MP43");
    assert_eq!((v.width, v.height), (640, 360));
    let a = e.audio.as_ref().expect("要有音频流");
    assert_eq!(a.codec, "WMA2");
    assert_eq!((a.channels, a.sample_rate), (2, 44100));
    assert_eq!(e.bytes, std::fs::metadata(&p).unwrap().len());

    // 时长必须是**容器自己说的那个数**，不是扫描器自己算的 —— 所以跟独立打开的
    // 解封装器对账，而不是写死一个毫秒数（那是在钉封装器的公式）。
    let direct = AsfDemuxer::new(std::io::BufReader::new(std::fs::File::open(&p).unwrap()))
        .expect("直接打开")
        .info()
        .duration_ms();
    assert_eq!(e.duration_ms, Some(direct));
    // 顺带确认这个数不是 0 / 不是天文数字：10 帧 × 40 ms 的文件怎么算都在半秒上下。
    let dur = e.duration_ms.unwrap();
    assert!((360..=600).contains(&dur), "10 帧的时长不对劲：{dur} ms");

    assert_eq!(
        e.verdict,
        Verdict::Convertible {
            targets: vec!["wmv", "mp4"]
        }
    );
}

/// 视频能解、音频不能：**两条路都还得 `--no-audio`**，所以必须带上提示，
/// 否则用户照着 TARGET 去转，会撞上一句「音频是 0x0055」。
#[test]
fn unsupported_audio_keeps_targets_but_warns() {
    let d = temp_dir("mp3");
    let p = d.join("a.wmv");
    let mp3 = AudioFormat {
        format_tag: 0x0055,
        ..wma_audio()
    };
    make_asf(&p, 320, 240, MP43, &frames_10(), Some(mp3));

    let e = &scan(&[p]).expect("扫")[0];
    assert_eq!(
        e.verdict,
        Verdict::Convertible {
            targets: vec!["wmv", "mp4"]
        }
    );
    assert!(
        e.note.as_deref().unwrap_or("").contains("--no-audio"),
        "得提示只能 `--no-audio`，实际 note = {:?}",
        e.note
    );
}

/// AAC 只能去 `.mp4`（那条路是**原样搬帧**，`.wmv` 只直通 WMA）。
#[test]
fn aac_audio_only_offers_mp4() {
    let d = temp_dir("aac");
    let p = d.join("a.wmv");
    let aac = AudioFormat {
        format_tag: 0x00FF,
        extra: vec![0x12, 0x10],
        ..wma_audio()
    };
    make_asf(&p, 320, 240, MP43, &frames_10(), Some(aac));

    let e = &scan(&[p]).expect("扫")[0];
    assert_eq!(
        e.verdict,
        Verdict::Convertible {
            targets: vec!["mp4"]
        }
    );
}

/// 视频编解码器不认识 → `unsupported`，且**说清是哪一个**。
#[test]
fn unsupported_video_codec_names_the_codec() {
    let d = temp_dir("flv");
    let p = d.join("a.wmv");
    make_asf(&p, 320, 240, FLV1, &frames_10(), None);

    let e = &scan(&[p]).expect("扫")[0];
    assert!(!e.verdict.is_convertible());
    let reason = match &e.verdict {
        Verdict::Unsupported(s) => s.clone(),
        other => panic!("应该是 unsupported，实际 {other:?}"),
    };
    assert!(reason.contains("FLV1"), "原因里要带上 FourCC：{reason}");
    // 容器与尺寸仍然要报出来 —— 这恰恰是「认出来了但解不了」的信息量。
    assert_eq!(e.container, Some("ASF"));
    assert_eq!(e.video.as_ref().map(|v| v.width), Some(320));
}

/// 空文件 / 乱字节 / 截断 / 改字节：**一律不 panic**，认不出的报 `unknown`。
#[test]
fn broken_files_are_unknown_and_never_panic() {
    let d = temp_dir("broken");
    let good = d.join("good.wmv");
    make_asf(&good, 320, 240, MP43, &frames_10(), None);
    let bytes = std::fs::read(&good).unwrap();

    let empty = d.join("empty.wmv");
    std::fs::write(&empty, b"").unwrap();
    let junk = d.join("junk.wmv");
    std::fs::write(&junk, [0xA5u8; 64]).unwrap();

    // 截断到各种长度（含「刚过魔数」「刚过头部」这些最容易越界的位置）。
    for n in [4usize, 16, 30, 64, bytes.len() / 2] {
        let p = d.join(format!("trunc{n}.wmv"));
        std::fs::write(&p, &bytes[..n.min(bytes.len())]).unwrap();
    }
    // 改字节：把头部各处翻成 0xFF（长度字段、GUID 都会被打坏）。
    for at in [0usize, 8, 16, 24, 32, 40, 48, 56] {
        let mut b = bytes.clone();
        if at < b.len() {
            b[at] = 0xFF;
        }
        let p = d.join(format!("flip{at}.wmv"));
        std::fs::write(&p, &b).unwrap();
    }

    // 不 panic 就是主要判据（panic 会让整个测试失败）；再确认结论是自洽的。
    let entries = scan(&[d.clone()]).expect("扫");
    assert!(entries.len() >= 15, "应该扫到全部文件：{}", entries.len());
    for e in &entries {
        match &e.verdict {
            Verdict::Convertible { .. } => assert!(e.video.is_some(), "能转就该有视频流"),
            Verdict::Unsupported(s) | Verdict::Unknown(s) => {
                assert!(!s.is_empty(), "{} 没给原因", e.path.display());
            }
        }
    }
    // 空文件必须明确报 unknown（读不到任何魔数）。
    let e = entries
        .iter()
        .find(|e| e.path == empty)
        .expect("空文件要在结果里");
    assert!(matches!(&e.verdict, Verdict::Unknown(_)), "空文件应报 unknown");
}

#[test]
fn walk_is_recursive_sorted_and_skips_symlinks() {
    let d = temp_dir("walk");
    std::fs::create_dir_all(d.join("sub/deep")).unwrap();
    let body = {
        let tmp = d.join("tpl.wmv");
        make_asf(&tmp, 160, 120, MP43, &frames_10(), None);
        let b = std::fs::read(&tmp).unwrap();
        // 模板文件本身要删掉，否则它会出现在遍历结果里。
        std::fs::remove_file(&tmp).unwrap();
        b
    };
    std::fs::write(d.join("b.wmv"), &body).unwrap();
    std::fs::write(d.join("sub/a.wmv"), &body).unwrap();
    std::fs::write(d.join("sub/deep/c.wmv"), &body).unwrap();

    let entries = scan(&[d.clone()]).expect("扫");
    let names: Vec<String> = entries
        .iter()
        .map(|e| {
            e.path
                .strip_prefix(&d)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    // 排序是扫描器给的保证：同样的输入永远同样的输出（`read_dir` 的顺序不保证）。
    assert_eq!(names, vec!["b.wmv", "sub/a.wmv", "sub/deep/c.wmv"]);

    // 符号链接在**目录遍历**里要跳过：否则同一个文件会被数两遍，
    // 指回上层的链接还会让遍历转圈。
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(d.join("b.wmv"), d.join("link.wmv")).unwrap();
        std::os::unix::fs::symlink(&d, d.join("loop")).unwrap();
        let again = scan(&[d.clone()]).expect("扫");
        assert_eq!(again.len(), 3, "符号链接不该被数进来");
        // 命令行上**直接点名**的符号链接照扫 —— 那是用户的明确意图。
        let one = scan(&[d.join("link.wmv")]).expect("扫");
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].container, Some("ASF"));
    }
}

/// 一千个文件 < 2 s（TODO 里 C 段的第三条判据）。
///
/// 这个门限挡的是「扫描器偷偷把整个文件读了一遍」这类退化：只读头部的实现
/// 在一千个小文件上是几十毫秒级，差两个数量级才可能超时。
#[test]
fn scanning_a_thousand_files_stays_under_two_seconds() {
    let d = temp_dir("many");
    let good = d.join("tpl.wmv");
    make_asf(&good, 320, 240, MP43, &frames_10(), Some(wma_audio()));
    let body = std::fs::read(&good).unwrap();
    std::fs::remove_file(&good).unwrap();

    for i in 0..1000 {
        let p = d.join(format!("f{i:04}.wmv"));
        if i % 4 == 0 {
            std::fs::write(&p, &body).unwrap();
        } else {
            // 一半以上是认不出的东西 —— 真实目录里大部分文件都不是视频。
            std::fs::write(&p, format!("not a video {i}").as_bytes()).unwrap();
        }
    }

    let t = Instant::now();
    let entries = scan(&[d.clone()]).expect("扫");
    let dt = t.elapsed();
    assert_eq!(entries.len(), 1000);
    let convertible = entries.iter().filter(|e| e.verdict.is_convertible()).count();
    assert_eq!(convertible, 250, "四分之一是能转的");
    assert!(
        dt.as_secs_f64() < 2.0,
        "1000 个文件用了 {:.2} s（门限 2 s）—— 是不是把整个文件读进去了？",
        dt.as_secs_f64()
    );
}

/// 我们自己封的 MP4 也要认出来（`mp4v` + `mvhd` 时长）。
#[test]
fn mp4_reports_mp4v_and_mvhd_duration() {
    use v2v_mp4::{Mp4Muxer, VideoCodec, VideoTrack};

    let d = temp_dir("mp4");
    let p = d.join("a.mp4");
    let mut m = Mp4Muxer::new(VideoTrack {
        width: 320,
        height: 240,
        timescale: 12800,
        codec: VideoCodec::Mp4v {
            decoder_specific_info: vec![0, 0, 1, 0xb0, 0x01],
        },
        avg_bitrate: 300_000,
    });
    // 10 帧 × 512 tick @12800 = 每帧 40 ms，共 400 ms。
    for i in 0..10 {
        m.write_video(&payload(i), i == 0, 512).unwrap();
    }
    std::fs::write(&p, m.finish()).unwrap();

    let e = &scan(&[p]).expect("扫")[0];
    assert_eq!(e.container, Some("MP4"));
    let v = e.video.as_ref().expect("要有视频流");
    assert_eq!(v.codec, "mp4v");
    assert_eq!((v.width, v.height), (320, 240));
    // `mvhd` 的时间基是 1000（毫秒），所以这里的数应当是**精确**的 400。
    assert_eq!(e.duration_ms, Some(400));
    assert!(e.verdict.is_convertible());
}

/// ffmpeg 产的 AVI：容器 / 编码 / 尺寸 / 时长都要对（没有 ffmpeg 就跳过）。
#[test]
fn avi_from_ffmpeg_reports_the_right_things() {
    if !has_ffmpeg() {
        eprintln!("SKIP: 没有可用的 ffmpeg");
        return;
    }
    let d = temp_dir("avi");
    let p = d.join("a.avi");
    let ok = std::process::Command::new("ffmpeg")
        .args([
            "-v", "error", "-y", "-f", "lavfi", "-i",
            "testsrc=size=320x240:rate=25:duration=1", "-c:v", "mpeg4", "-q:v", "5",
        ])
        .arg(&p)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("SKIP: ffmpeg 造样本失败");
        return;
    }

    let e = &scan(&[p]).expect("扫")[0];
    assert_eq!(e.container, Some("AVI"));
    let v = e.video.as_ref().expect("要有视频流");
    assert_eq!(v.codec, "FMP4", "ffmpeg 在 AVI 里写的是 FMP4");
    assert_eq!((v.width, v.height), (320, 240));
    // 25 fps × 1 s：`avih` 的「帧数 × 帧时长」应当给 1000 ms（±1 帧）。
    let dur = e.duration_ms.expect("avi 该给时长");
    assert!((960..=1040).contains(&dur), "1 秒的样本报了 {dur} ms");
    assert!(e.verdict.is_convertible());
}

/// 真样本（默认 `#[ignore]`）：扫描结果必须与**直接打开解封装器**读到的一致。
///
/// 跑法：`V2V_SCAN_DIR=testing cargo test --release -p v2v-cli -- --ignored --nocapture`
#[test]
#[ignore = "需要真样本目录（V2V_SCAN_DIR）"]
fn real_samples_match_the_container() {
    let dir = std::env::var("V2V_SCAN_DIR").unwrap_or_else(|_| "testing".into());
    let entries = scan(&[PathBuf::from(&dir)]).expect("扫");
    let mut checked = 0;
    for e in &entries {
        if e.container != Some("ASF") {
            continue;
        }
        let dm = AsfDemuxer::new(std::io::BufReader::new(
            std::fs::File::open(&e.path).unwrap(),
        ))
        .expect("直接打开");
        let want = dm
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Video)
            .and_then(|s| s.video.clone())
            .expect("真样本有视频流");
        let got = e.video.as_ref().expect("扫描结果要有视频流");
        assert_eq!(got.codec, v2v_asf::fourcc_string(want.fourcc));
        assert_eq!((got.width, got.height), (want.width, want.height));
        assert_eq!(e.duration_ms, Some(dm.info().duration_ms()));
        checked += 1;
    }
    assert!(checked > 0, "{dir} 里没有 ASF 样本？");
    eprintln!("对账了 {checked} 个真样本");
}

/// 两种输出都不能因为「一条能转的都没有」就吐出半张表 / 半个表头。
#[test]
fn empty_result_renders_nothing() {
    let d = temp_dir("empty");
    let p = d.join("junk.bin");
    std::fs::write(&p, b"nope").unwrap();
    let entries = scan(&[p]).expect("扫");
    assert_eq!(format_table(&entries, false), "");
    // TSV 留表头（管道那头要能无条件按列取），但只有表头。
    assert_eq!(format_tsv(&entries, false).lines().count(), 1);
}

fn has_ffmpeg() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
