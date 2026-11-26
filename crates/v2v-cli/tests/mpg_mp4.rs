//! `.mpg`（MPEG program stream）→ `.mp4` 的端到端判据。
//!
//! ## 这条要钉什么
//!
//! MPEG-1/2 的**解码序 ≠ 显示序**，而 CLI 的转码管线是「一个样本 → 若干帧」的
//! 形状。两者接起来有两类错，它们**都不会报错**：
//!
//! 1. **最后一个 GOP 被丢掉** —— 没有下一个 I 帧来触发摊平，`finish()` 忘了调就
//!    整块消失。`.mp4` 照样能播、时长照样「看着差不多」，只是短了几十帧。
//! 2. **帧序错** —— B 帧被按解码序交出去。PSNR 会掉到 10~12 dB，但「能不能解 /
//!    宽高 / 帧率」全都正常。
//!
//! 所以判据分两条，一条管帧数、一条管画面：
//!
//! - **帧数**：与 `MpgPlayer::picture_count()`（picture 个数 = 显示帧数）**必须相等**。
//!   这是精确判据，不含阈值。
//! - **画面**：ffmpeg 把产物解回来，与源按帧号对齐算 PSNR，**Y 分量 ≥ 30 dB**。
//!
//! ⚠️ **30 dB 不是质量门槛，是「灾难检测」**：qscale=6 的重编码实测在 41~46 dB
//! （720×576 隔行 / 640×360 渐进 / 带 B 帧的 500 帧都量过），而帧序错或裁剪错会
//! 直接掉到 12 dB 以下。中间那一大段没有信息量，所以不设更细的阈值 ——
//! 拿 PSNR 当质量判据是另一件事，它依赖素材（见 `ogg_mp4.rs` 的同类说明）。
//!
//! 需要 `V2V_MPEG12_SAMPLE` 与 `ffmpeg` / `ffprobe`；缺任一个就跳过。

use std::path::PathBuf;
use std::process::{Command, Stdio};

use v2v_cli::{ConvertOptions, Mp4Audio, OutputFormat, convert_file};

fn sample_path() -> Option<PathBuf> {
    let p = std::env::var_os("V2V_MPEG12_SAMPLE")?;
    let p = PathBuf::from(p);
    if !p.is_file() {
        eprintln!("SKIP: V2V_MPEG12_SAMPLE 指向的文件不存在");
        return None;
    }
    Some(p)
}

fn has_tool(name: &str) -> bool {
    Command::new(name)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn unique_temp(stem: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("{stem}.{}", std::process::id()));
    p
}

#[test]
fn mpg_to_mp4_keeps_every_frame_and_the_right_picture() {
    let Some(input) = sample_path() else { return };
    if !has_tool("ffmpeg") || !has_tool("ffprobe") {
        eprintln!("SKIP: PATH 上没有 ffmpeg / ffprobe");
        return;
    }

    // 参考帧数：picture 个数就是显示帧数（每张 picture 都显示一次）。
    let pictures = {
        let p = v2v_player::mpeg::MpgPlayer::open(&input).expect("打开");
        p.picture_count()
    };

    let out = unique_temp("v2v_mpg_mp4").with_extension("mp4");
    let opts = ConvertOptions {
        format: OutputFormat::Mp4,
        // MPEG-PS 的音轨是 MP2 / AC-3，本仓没有解码器 —— 这条路只转视频。
        // 不加这个开关 `convert_file` 会明确报错（那条判据见下一个测试）。
        copy_audio: false,
        audio: Mp4Audio::Aac,
        ..Default::default()
    };
    let rep = convert_file(&input, &out, &opts).expect("转码");
    assert_eq!(rep.frames, pictures, "转出来的帧数与 picture 个数不一致");

    // ---- 帧数：ffprobe 从产物里数（不看我们自己的报告）----
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,codec_name",
            "-of",
            "csv=p=0",
        ])
        .arg(&out)
        .output()
        .expect("跑 ffprobe");
    let text = String::from_utf8_lossy(&probe.stdout);
    let text = text.trim();
    // `-show_entries` 的输出顺序跟字段在结构体里的位置走，别假定 —— 取最后那个
    // 纯数字的字段（`codec_name` 一定不是数字，`nb_read_frames` 一定是）。
    let frames: usize = text
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .next_back()
        .unwrap_or_else(|| panic!("ffprobe 没给帧数：{text:?}"));
    assert_eq!(
        frames, pictures,
        "产物里有 {frames} 帧，源有 {pictures} 张 picture —— 最后一个 GOP 丢了？"
    );

    // ---- 画面：ffmpeg 把产物解回来，与源按帧号对齐 ----
    //
    // `setpts=N/TB` 两边都改成「按帧号」的时间轴：源是 PS（90 kHz）、产物是 MP4
    // （timescale 1000），不对齐的话 psnr 滤镜会按时间戳配对，一错就是整段错位。
    // ⚠️ `-v info` 是必须的：`psnr` 滤镜把统计打在 **INFO** 级别，
    // 用 `-v error` 会连它一起筛掉，只剩空日志 —— 那时人会以为是滤镜没跑。
    let psnr = Command::new("ffmpeg")
        .args(["-v", "info", "-i"])
        .arg(&out)
        .arg("-i")
        .arg(&input)
        .args([
            "-lavfi",
            "[0:v]setpts=N/TB[a];[1:v]setpts=N/TB[b];[a][b]psnr",
            "-f",
            "null",
            "-",
        ])
        .output()
        .expect("跑 ffmpeg");
    let log = String::from_utf8_lossy(&psnr.stderr);
    let y = log
        .split("PSNR y:")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or_else(|| panic!("ffmpeg 没给 PSNR：{log}"));
    assert!(
        y >= 30.0,
        "PSNR Y = {y:.2} dB，低于 30 —— 帧序或裁剪错了（正常是 41~46 dB）"
    );

    let _ = std::fs::remove_file(&out);
}

/// 源里有音频、而我们解不了时必须**明确报错**，不能悄悄转出一个没声音的文件。
///
/// 判据是「错误信息里出现 `--no-audio`」—— 用户照着做就能过，这是这条报错存在的
/// 唯一意义。用 `V2V_MPEG12_SAMPLE_WITH_AUDIO` 指定一个带 MP2 音轨的 `.mpg`。
#[test]
fn mpg_with_mp2_audio_is_refused_with_a_way_out() {
    let Some(p) = std::env::var_os("V2V_MPEG12_SAMPLE_WITH_AUDIO") else {
        eprintln!("SKIP: 没设 V2V_MPEG12_SAMPLE_WITH_AUDIO");
        return;
    };
    let input = PathBuf::from(p);
    if !input.is_file() {
        eprintln!("SKIP: V2V_MPEG12_SAMPLE_WITH_AUDIO 指向的文件不存在");
        return;
    }
    let out = unique_temp("v2v_mpg_audio").with_extension("mp4");
    let opts = ConvertOptions {
        format: OutputFormat::Mp4,
        ..Default::default()
    };
    let err = convert_file(&input, &out, &opts).expect_err("有 MP2 音轨时必须拒绝");
    let msg = err.to_string();
    assert!(
        msg.contains("--no-audio"),
        "报错里得给出路（`--no-audio`），实际是：{msg}"
    );
}

/// 只有视频的 `.mpg` 不加 `--no-audio` 也应该能转 —— 别把「没音轨」和
/// 「有音轨但解不了」混成一种情况。
#[test]
fn mpg_without_audio_converts_by_default() {
    let Some(input) = sample_path() else { return };
    let out = unique_temp("v2v_mpg_noaudio").with_extension("mp4");
    let opts = ConvertOptions {
        format: OutputFormat::Mp4,
        ..Default::default()
    };
    convert_file(&input, &out, &opts).expect("没有音轨的 .mpg 默认就该能转");
    let _ = std::fs::remove_file(&out);
}
