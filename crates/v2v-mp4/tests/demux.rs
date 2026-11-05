//! 封装 → 解封装 往返：`Mp4Muxer` 写出来的东西，`Mp4Demuxer` 要能原样读回来。
//!
//! 这条测试**不依赖真样本**，所以 CI 里也真跑（不是 `SKIP`）。它验的是：
//! 流数 / 宽高 / fourcc / 样本数 / 关键帧位 / 样本字节 —— 往返一圈都对得上。

use std::io::Cursor;

use v2v_mp4::{AudioCodec, AudioTrack, Mp4Demuxer, Mp4Muxer, VideoCodec, VideoTrack};

fn sample_file() -> Vec<u8> {
    let vt = VideoTrack {
        width: 320,
        height: 240,
        timescale: 1000,
        codec: VideoCodec::Mp4v {
            decoder_specific_info: vec![0, 0, 1, 0xB0],
        },
        avg_bitrate: 500_000,
    };
    let at = AudioTrack {
        channels: 2,
        sample_rate: 44100,
        timescale: 44100,
        codec: AudioCodec::Es {
            object_type: 0x40,
            decoder_specific_info: vec![0x12, 0x10],
        },
        avg_bitrate: 128_000,
        priming: 0,
    };
    let mut m = Mp4Muxer::with_audio(vt, at);
    for i in 0..5u8 {
        // 每帧内容不同，这样「读错偏移」会被字节比对抓住。
        m.write_video(&[i; 100], i == 0, 40).unwrap();
        m.write_audio(&[0xA0 | i; 20], 1024).unwrap();
    }
    m.finish()
}

#[test]
fn roundtrip_streams_and_samples() {
    let bytes = sample_file();
    let mut r = Cursor::new(bytes);
    let mut d = Mp4Demuxer::new(&mut r).unwrap();

    assert_eq!(d.streams().len(), 2, "应当有视频 + 音频两条流");
    let v = d
        .streams()
        .iter()
        .find(|s| s.video.is_some())
        .expect("没有视频流");
    let vf = v.video.as_ref().unwrap();
    assert_eq!((vf.width, vf.height), (320, 240));
    assert_eq!(vf.fourcc, u32::from_le_bytes(*b"mp4v"), "fourcc 按小端约定存");

    let a = d
        .streams()
        .iter()
        .find(|s| s.audio.is_some())
        .expect("没有音频流");
    let af = a.audio.as_ref().unwrap();
    assert_eq!(af.channels, 2);
    assert_eq!(af.sample_rate, 44100);
    assert_eq!(af.format_tag, 0x00FF, "AAC 的内部约定编号");

    // 后面要可变借用 `d`，先把编号拷出来（借用到 `streams()` 为止）。
    let (vnum, anum) = (v.number, a.number);

    let mut total = 0;
    let mut vids = 0;
    let mut auds = 0;
    let mut first_video_key = None;
    while let Some(s) = d.next_sample().unwrap() {
        total += 1;
        if s.stream == vnum {
            vids += 1;
            if first_video_key.is_none() {
                first_video_key = Some(s.key_frame);
                assert_eq!(s.data.len(), 100);
                assert!(s.data.iter().all(|&b| b == 0), "第一帧应当是 i=0 的内容");
            }
        } else if s.stream == anum {
            auds += 1;
            assert_eq!(s.data.len(), 20);
            if auds == 1 {
                assert!(s.data.iter().all(|&b| b == 0xA0), "第一段音频应当是 i=0");
            }
        }
    }
    assert_eq!((vids, auds), (5, 5), "两轨各 5 个样本");
    assert_eq!(total, 10);
    assert_eq!(first_video_key, Some(true), "第一帧应当是关键帧");
}

#[test]
fn truncated_file_is_an_error_not_a_panic() {
    let bytes = sample_file();
    // 砍掉尾部（moov 在尾部）→ 必须报错，不能 panic、不能返回半截数据。
    let cut = &bytes[..bytes.len() / 2];
    let mut r = Cursor::new(cut.to_vec());
    match Mp4Demuxer::new(&mut r) {
        Err(_) => {}
        Ok(_) => panic!("截断的文件不该被当成有效 MP4"),
    }
}
