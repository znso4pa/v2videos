//! **纯音频 MP4** 对 `ffmpeg` 的判据（独立 oracle）。
//!
//! 为什么单开一条：`.ogg → .mp4` 的地基就是「只有一条音轨的 MP4」，
//! 而本仓的封装器原本只有 `new` / `with_audio` 两个入口、**都要求视频轨**。
//! `Mp4Muxer::audio_only` 补上了这个口子 —— 但「我们写出来了」不等于
//! 「别人读得懂」，所以要拿 ffmpeg 当 oracle。
//!
//! 判据分两层，因为「格式被接受」和「表填对了」是两件事：
//!
//! 1. **格式级**：ffprobe 认这个文件 —— 只有 1 条流、`codec_type=audio`、
//!    `codec_name=pcm_s16le`、声道数 / 采样率 / 时长都对。
//!    这条挡的是「`moov` 里只有一个 `trak`」「轨号与 `mvhd.next_track_ID`
//!    对不上」「`mvhd.volume` 之类没填」这类**结构性**错误 ——
//!    它们会让播放器直接不认这个文件。
//! 2. **样本级**：`ffmpeg -f s16le` 解出来的字节与写进去的**逐字节相同**。
//!    这条挡的是 `stsz` / `stts` / `stco` / `stsc` 填错 ——
//!    那些错误 ffprobe **看不出来**，只会让音频变调、错位或截断。
//!
//! 输入是本地合成的确定性信号（两个声道用**不同周期**的伪随机序列，
//! 周期都大于总长），所以：
//! - 不需要外部样本，这个测试**永远跑**（`real_sample.rs` 那种要环境变量）；
//! - 任何**错位**（丢一个采样点、重复一段、左右声道互换）都会被抓到，
//!   而单纯的正弦波在这些情况下常常「听着还行」。
//!
//! ## 踩过的坑：别用「静音」或「正弦」当测试信号
//!
//! 静音乘任何系数还是 0，正弦对错位不敏感 —— 两者都能让写坏的封装器
//! 通过测试（本仓在这上面被骗过两次，见 AGENTS.md）。

use std::process::Command;

use v2v_mp4::{AudioCodec, AudioTrack, Mp4Muxer};

const CHANNELS: u16 = 2;
const SAMPLE_RATE: u32 = 44_100;
/// 采样点数 = 1 秒。
const POINTS: usize = SAMPLE_RATE as usize;

/// 合成确定性 PCM（S16LE，交错）。
///
/// 左右声道用**不同周期**的线性同余序列：周期 65536 > 44100，
/// 所以整段里不会自己重复，任何整体错位都会让字节对不上。
fn synth_pcm() -> Vec<u8> {
    let mut out = Vec::with_capacity(POINTS * CHANNELS as usize * 2);
    let mut l: u16 = 0x1234;
    let mut r: u16 = 0x9abc;
    for _ in 0..POINTS {
        // 两种不同的 LCG 步进，避免两声道同步变化掩盖「声道错位」。
        l = l.wrapping_mul(1103).wrapping_add(12345);
        r = r.wrapping_mul(2087).wrapping_add(54321);
        out.extend_from_slice(&l.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
    }
    out
}

fn pcm_track() -> AudioTrack {
    AudioTrack {
        channels: CHANNELS,
        sample_rate: SAMPLE_RATE,
        timescale: SAMPLE_RATE,
        codec: AudioCodec::Pcm {
            bits: 16,
            big_endian: false,
        },
        avg_bitrate: 0,
        // PCM 没有编码器预热。
        priming: 0,
        presentation_samples: None,
    }
}

fn write_file(bytes: &[u8], name: &str) -> std::path::PathBuf {
    let path = std::path::PathBuf::from(format!("/tmp/{name}"));
    std::fs::write(&path, bytes).expect("写 MP4");
    path
}

#[test]
fn audio_only_pcm_is_read_by_ffmpeg() {
    let pcm = synth_pcm();
    let point_bytes = CHANNELS as usize * 2;

    // ---- 1. 按文档的口径写：一个采样点 = 一个样本 ----
    let mut m = Mp4Muxer::audio_only(pcm_track());
    for chunk in pcm.chunks_exact(point_bytes) {
        m.write_audio(chunk, 1).expect("写音频样本");
    }
    let file = m.finish();
    let path = write_file(&file, "v2v_audio_only.mp4");
    eprintln!("写出 {}（{} 字节）", path.display(), file.len());

    // ---- 2. 格式级：ffprobe 认不认 ----
    //
    // 按 `key=value` 取，**不按字段顺序**：ffprobe 的 `csv` 输出顺序是它自己
    // 定的（实测 `codec_name,codec_type,sample_rate,channels`），拿它当
    // 期望值等于把「ffprobe 的排版」写进判据 —— 换个版本就红。
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries"])
        .arg("stream=codec_name,codec_type,channels,sample_rate")
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(&path)
        .output()
        .expect("跑不了 ffprobe");
    assert!(
        probe.status.success(),
        "ffprobe 失败：{}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let probe_out = String::from_utf8_lossy(&probe.stdout);
    eprintln!("ffprobe stream:\n{probe_out}");
    let get = |key: &str| -> String {
        probe_out
            .lines()
            .find_map(|l| l.strip_prefix(key)?.strip_prefix('='))
            .unwrap_or_else(|| panic!("ffprobe 输出里没有 {key}：{probe_out}"))
            .trim()
            .to_string()
    };
    assert_eq!(get("codec_name"), "pcm_s16le", "编解码器不对");
    assert_eq!(get("codec_type"), "audio", "这条流应该是音频");
    assert_eq!(get("channels"), "2", "声道数不对");
    assert_eq!(get("sample_rate"), "44100", "采样率不对");

    // 流数量必须正好是 1 —— 纯音频文件里不该有第二条流。
    let count = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "format=nb_streams"])
        .args(["-of", "csv=p=0"])
        .arg(&path)
        .output()
        .expect("跑不了 ffprobe");
    assert_eq!(
        String::from_utf8_lossy(&count.stdout).trim(),
        "1",
        "纯音频 MP4 应该只有 1 条流"
    );

    // 时长：`mvhd` / `tkhd` 都按毫秒写，1 秒的 PCM 应该报 1.000 秒。
    let dur = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "format=duration"])
        .args(["-of", "csv=p=0"])
        .arg(&path)
        .output()
        .expect("跑不了 ffprobe");
    let dur: f64 = String::from_utf8_lossy(&dur.stdout)
        .trim()
        .parse()
        .expect("时长不是数字");
    assert!(
        (dur - 1.0).abs() < 0.01,
        "时长应该是 1.000 秒，ffprobe 报 {dur}"
    );

    // ---- 3. 样本级：解出来必须逐字节相同 ----
    let dec = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-f", "s16le", "-acodec", "pcm_s16le", "-"])
        .output()
        .expect("跑不了 ffmpeg");
    assert!(
        dec.status.success(),
        "ffmpeg 解码失败：{}",
        String::from_utf8_lossy(&dec.stderr)
    );
    assert_eq!(
        dec.stdout.len(),
        pcm.len(),
        "解出来的字节数不对（{} vs {}）—— `stsz` / `stts` 填错了",
        dec.stdout.len(),
        pcm.len()
    );
    // 第一个不同的位置 —— 报出来比「assert_eq! 两个 176400 字节的 Vec」有用得多。
    if let Some(i) = dec.stdout.iter().zip(&pcm).position(|(a, b)| a != b) {
        let point = i / point_bytes;
        let chan = (i % point_bytes) / 2;
        panic!(
            "第 {i} 字节（采样点 {point}、声道 {chan}）不同：{:02x} vs {:02x}",
            dec.stdout[i], pcm[i]
        );
    }
}

/// 同一个文件里，**样本数不是 1 秒整数倍**时也不能丢尾。
///
/// 真实场景里末页的采样点数几乎总是零头（`.ogg` 的末页 granule 就是），
/// 所以这条不能只测「正好 1 秒」。
#[test]
fn audio_only_pcm_keeps_ragged_tail() {
    const POINTS: usize = 1000;
    let point_bytes = CHANNELS as usize * 2;
    let pcm: Vec<u8> = (0..POINTS * point_bytes).map(|i| (i * 37) as u8).collect();

    let mut m = Mp4Muxer::audio_only(pcm_track());
    for chunk in pcm.chunks_exact(point_bytes) {
        m.write_audio(chunk, 1).expect("写音频样本");
    }
    let path = write_file(&m.finish(), "v2v_audio_only_tail.mp4");

    let dec = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-f", "s16le", "-acodec", "pcm_s16le", "-"])
        .output()
        .expect("跑不了 ffmpeg");
    assert!(dec.status.success(), "ffmpeg 解码失败");
    assert_eq!(
        dec.stdout, pcm,
        "{} 个采样点（{} 字节）必须一个不少",
        POINTS,
        pcm.len()
    );
}
