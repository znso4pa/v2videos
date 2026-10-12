//! `pcm2aac` —— 把裸 PCM 编成 ADTS `.aac`。
//!
//! `v2v-aac` 是库，正常入口是 `v2v convert --mp4`。但**量音质**（对着 ffmpeg 比
//! SNR）需要一个能直接把 PCM 编成 AAC 的独立工具 —— 从 `.wmv` 一路转过去太绕，
//! 而且中间夹了 WMA 解码，基准就不干净了。
//!
//! ```text
//!   pcm2aac <in.pcm> <out.aac> [选项]
//!
//!   --rate N       采样率（默认 44100）
//!   --channels N   声道数（默认 2）
//!   --bitrate N    目标码率 bit/s（默认 128000）
//!   --kb           用 KBD 窗（默认正弦窗）
//! ```
//!
//! 输入的 PCM 是**小端 16 位交织**（`ffmpeg -f s16le` 的格式），
//! 与 `v2v-aac` 期望的「每声道一条、f32、±32768 量级」差一步拆声道 + 转 f32。
//!
//! 用法示例（配 ffmpeg 量产参考）：
//!
//! ```bash
//! ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=20" \
//!        -ac 2 -f s16le /tmp/clean.pcm -y
//! cargo run --release -p v2v-aac --example pcm2aac -- /tmp/clean.pcm /tmp/ours.aac
//! ffmpeg -i /tmp/ours.aac -f s16le /tmp/ours.pcm -y        # 再比 SNR
//! ```

use std::process::ExitCode;

use v2v_aac::{Encoder, EncoderConfig, Frame};
use v2v_msmpeg4::BitWriter;

/// 一帧 AAC 的采样点数（长窗）。
const FRAME_LEN: usize = 1024;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("pcm2aac: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (mut input, mut output) = (None, None);
    let mut sample_rate = 44_100u32;
    let mut channels = 2u8;
    let mut bitrate = 128_000u32;
    let mut kb_window = false;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--rate" => sample_rate = next(&mut args, "--rate")?,
            "--channels" => channels = next(&mut args, "--channels")?,
            "--bitrate" => bitrate = next(&mut args, "--bitrate")?,
            "--kb" => kb_window = true,
            "-h" | "--help" => {
                println!(
                    "{}",
                    include_str!("pcm2aac.rs")
                        .lines()
                        .take(22)
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                return Ok(());
            }
            _ if input.is_none() => input = Some(a),
            _ if output.is_none() => output = Some(a),
            _ => return Err(format!("多余的参数：{a}")),
        }
    }
    let (input, output) =
        match (input, output) {
            (Some(i), Some(o)) => (i, o),
            _ => return Err(
                "用法：pcm2aac <in.pcm> <out.aac> [--rate N] [--channels N] [--bitrate N] [--kb]"
                    .into(),
            ),
        };

    let raw = std::fs::read(&input).map_err(|e| format!("读 {input}：{e}"))?;
    if raw.len() % (2 * channels as usize) != 0 {
        return Err(format!(
            "PCM 长度 {} 不是「2 字节 × {} 声道」的整数倍",
            raw.len(),
            channels
        ));
    }
    let total = raw.len() / 2 / channels as usize;

    let mut enc = Encoder::new(EncoderConfig {
        channels,
        sample_rate,
        bitrate,
        kb_window,
    })
    .map_err(|e| format!("建编码器：{e}"))?;

    // 逐帧编：每帧 1024 个采样点（最后一帧补 0）。
    let mut out = Vec::with_capacity(raw.len() / 8);
    let mut pos = 0usize;
    let mut frames = 0usize;
    while pos < total {
        let n = (total - pos).min(FRAME_LEN);
        let planes: Vec<Vec<f32>> = (0..channels as usize)
            .map(|ch| {
                (0..n)
                    .map(|i| {
                        let o = ((pos + i) * channels as usize + ch) * 2;
                        f32::from(i16::from_le_bytes([raw[o], raw[o + 1]]))
                    })
                    .collect()
            })
            .collect();
        let frame = enc
            .encode_frame(&Frame { planes })
            .map_err(|e| format!("编第 {frames} 帧：{e}"))?;
        out.extend_from_slice(&adts_header(frame.len(), channels, sample_rate));
        out.extend_from_slice(&frame);
        pos += FRAME_LEN;
        frames += 1;
    }

    std::fs::write(&output, &out).map_err(|e| format!("写 {output}：{e}"))?;
    let kbps = out.len() as f64 * 8.0 / (total as f64 / f64::from(sample_rate)) / 1000.0;
    println!(
        "{input} → {output}：{frames} 帧 / {} 字节（{total} 个采样点，{kbps:.0} kbps）",
        out.len()
    );
    Ok(())
}

fn next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    args.next()
        .ok_or_else(|| format!("{flag} 后面要跟一个值"))?
        .parse()
        .map_err(|_| format!("{flag} 的值解析不了"))
}

/// ADTS 头（7 字节，无 CRC）。
///
/// ⚠️ `aac_frame_length` 是**含这 7 字节头**的整帧长度，且跨字节 3/4/5，
/// 手拼位运算极易错（我在测试里踩过：写成载荷长度 + `>>3` 只取低 8 位，
/// 于是 13 位的字段只剩高 5 位）。所以用 `BitWriter` 逐字段写。
fn adts_header(payload: usize, channels: u8, sample_rate: u32) -> [u8; 7] {
    let full = 7 + payload;
    assert!(full < (1 << 13), "ADTS 的 frame_length 只有 13 位");
    let sr_idx = v2v_aac::AudioSpecificConfig::sample_rate_index(sample_rate)
        .unwrap_or_else(|| panic!("{sample_rate} 不是 AAC 的标准采样率"));

    let mut w = BitWriter::new();
    w.put(0xFFF, 12); // syncword
    w.put(0, 1); // MPEG-4
    w.put(0, 2); // layer
    w.put(1, 1); // protection_absent（无 CRC）
    w.put(1, 2); // profile_objecttype：AAC-LC 是 AOT 2，ADTS 里写 AOT-1 = 1
    w.put(u32::from(sr_idx), 4); // sampling_frequency_index
    w.put(0, 1); // private_bit
    w.put(u32::from(channels), 3); // channel_configuration
    w.put(0, 1); // original_copy
    w.put(0, 1); // home
    w.put(0, 1); // copyright_identification_bit
    w.put(0, 1); // copyright_identification_start
    w.put(full as u32, 13); // aac_frame_length
    w.put(0x7FF, 11); // adts_buffer_fullness（VBR）
    w.put(0, 2); // number_of_raw_data_blocks_in_frame - 1
    let v = w.finish();
    assert_eq!(v.len(), 7, "ADTS 头应当是 7 字节");
    v.try_into().unwrap()
}
