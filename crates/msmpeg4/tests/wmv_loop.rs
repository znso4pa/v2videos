//! **WMV 闭环**：我们的编码器 → 我们的 ASF 封装 → `ffprobe` 验字段 + `ffmpeg`
//! 逐像素解 → 我们的解封装 + 解码器复原。
//!
//! 这是「格式转换」目标的端到端判据：整条链上没有任何一段依赖 ffmpeg 的
//! **实现**（它只当 oracle），产出的 `.wmv` 是标准的。
//!
//! 用法：要真跑 ffmpeg/ffprobe 对账，机器上得有它们（没有就 SKIP）。

mod common;

use common::{compare, ffmpeg_frames, unique_temp};
use v2v_asf::{AsfDemuxer, AsfMuxer, StreamKind, VideoStreamInfo};
use v2v_msmpeg4::{Decoder, Encoder, EncoderConfig, Frame};

/// 合成一段有运动的序列（纹理背景 + 平移方块），与 `encode_p.rs` 同思路。
fn moving_sequence(w: u32, h: u32, n: usize) -> Vec<Frame> {
    let (wu, hu) = (w as usize, h as usize);
    let mut out = Vec::new();
    for i in 0..n {
        let mut y = vec![0u8; wu * hu];
        for r in 0..hu {
            for c in 0..wu {
                let v = 100 + (((c + i * 3) / 8) % 4) as i32 * 30 + ((r / 16) % 2) as i32 * 10;
                y[r * wu + c] = v.clamp(16, 235) as u8;
            }
        }
        let ox = 40 + i * 6;
        let oy = 30 + i * 3;
        for r in 0..40usize {
            for c in 0..40usize {
                let yy = oy + r;
                let xx = ox + c;
                if yy < hu && xx < wu {
                    y[yy * wu + xx] = if (r / 4 + c / 4) % 2 == 0 { 30 } else { 220 };
                }
            }
        }
        let (cw, ch) = (wu / 2, hu / 2);
        let mut u = vec![128u8; cw * ch];
        let mut v = vec![128u8; cw * ch];
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = (128 + (c / 8) as i32 * 6 - (i as i32) * 3).clamp(16, 240) as u8;
                v[r * cw + c] = (128 - (r / 8) as i32 * 5 + (i as i32) * 2).clamp(16, 240) as u8;
            }
        }
        out.push(Frame { width: w, height: h, y, u, v });
    }
    out
}

#[test]
fn encode_mux_probe_decode_full_loop() {
    let (w, h) = (320u32, 240u32);
    let frames = moving_sequence(w, h, 6);

    // ---- ① 编码：I + P×5 ----
    let mut enc = Encoder::new(EncoderConfig {
        qscale: 8,
        ..Default::default()
    });
    let mut streams = Vec::new();
    for f in &frames {
        streams.push(enc.encode_frame(f).expect("编码"));
    }

    // ---- ② 封装成 .wmv ----
    let mut mux = AsfMuxer::new(VideoStreamInfo {
        width: w,
        height: h,
        fourcc: u32::from_le_bytes(*b"MP43"),
    });
    for (i, s) in streams.iter().enumerate() {
        mux.write_video(s, i == 0, (i as u32) * 33).expect("封装");
    }
    let wmv = mux.finish();
    let path = unique_temp("v2v_wmv_loop", "wmv");
    std::fs::write(&path, &wmv).expect("写文件");

    // ---- ③ ffprobe 验字段 ----
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_packets",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,nb_read_packets,nb_read_frames",
            "-of",
            "default",
            &path.to_string_lossy(),
        ])
        .output();
    if let Ok(out) = &probe {
        let text = String::from_utf8_lossy(&out.stdout);
        eprintln!("ffprobe:\n{text}");
        if text.contains("msmpeg4v3") {
            assert!(text.contains("msmpeg4v3"), "fourcc 没被认成 msmpeg4v3");
            assert!(text.contains(&format!("width={w}")));
            assert!(text.contains(&format!("height={h}")));
            assert!(text.contains("nb_read_frames=6"), "帧数不对");
        }
    }

    // ---- ④ 我们自己的解封装：原样读回 ----
    let mut d = AsfDemuxer::new(std::fs::File::open(&path).expect("打开")).expect("解封装");
    assert_eq!(d.streams().len(), 1);
    assert_eq!(d.streams()[0].kind, StreamKind::Video);
    let mut samples = Vec::new();
    while let Some(s) = d.next_sample().expect("读样本") {
        samples.push(s);
    }
    assert_eq!(samples.len(), 6, "解封装出的样本数应等于帧数");
    for (i, s) in samples.iter().enumerate() {
        assert_eq!(s.key_frame, i == 0, "帧 {i} 关键帧标志不对");
        assert_eq!(s.data, streams[i], "帧 {i} 字节不一致（封装丢数据）");
    }

    // ---- ⑤ 两个解码器逐位对账：我们的 vs ffmpeg 的 ----
    let ff = match ffmpeg_frames(path.to_str().unwrap(), 6, w, h) {
        Some(v) if v.len() == 6 => v,
        _ => {
            eprintln!("SKIP: 没有可用的 ffmpeg");
            let _ = std::fs::remove_file(&path);
            return;
        }
    };
    let _ = std::fs::remove_file(&path);

    let mut dec = Decoder::new();
    for (i, s) in streams.iter().enumerate() {
        let ours = dec.decode(s, w, h).expect("解码");
        let (worst, _at, _mean, _) = compare(&ours.y, &ours.u, &ours.v, &ff[i], w, h);
        assert_eq!(worst, 0, "帧 {i}：ffmpeg 与我们的解码不一致（最大差 {worst}）");
    }
}

use std::process::Command;
