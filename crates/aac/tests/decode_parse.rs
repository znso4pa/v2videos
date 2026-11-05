//! AAC-LC **比特流解析**（M1）：拿我们自己的编码器当 oracle。
//!
//! 编码器知道自己写了什么（几个元素、`max_sfb`、每带的码本与 scalefactor），
//! 解码器解析出来的结构必须与它一致。这条测试**不依赖真样本**，CI 里也真跑。

use v2v_aac::{Decoder, Encoder, EncoderConfig, Frame};

#[test]
fn parses_own_mono_frame() {
    let cfg = EncoderConfig {
        channels: 1,
        sample_rate: 44_100,
        ..Default::default()
    };
    let mut enc = Encoder::new(cfg).unwrap();
    let asc = enc.audio_specific_config().to_bytes();
    let dec = Decoder::new(&asc).unwrap();
    assert_eq!(dec.sample_rate, 44_100);
    assert_eq!(dec.channels, 1);
    assert_eq!(dec.frame_len, 1024);

    let n = 1024usize;
    let frame = Frame {
        planes: vec![(0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.5)
            .collect()],
    };
    let bytes = enc.encode_frame(&frame).unwrap();
    let info = dec.parse_frame(&bytes).unwrap();

    assert_eq!(info.elements.len(), 1, "一帧应当只有一个元素");
    let e = &info.elements[0];
    assert_eq!(e.kind, 0, "单声道应当是 SCE");
    assert_eq!(e.channels.len(), 1);
    let ch = &e.channels[0];
    assert_eq!(ch.max_sfb as usize, enc.num_sfb(), "max_sfb 应当等于编码器的频带数");
    assert_eq!(ch.section_cb.len(), ch.max_sfb as usize);
    assert_eq!(ch.scalefactor.len(), ch.max_sfb as usize);
    assert!(ch.section_cb.iter().all(|&cb| cb <= 11), "码本号越界");
    assert!(e.spectral_bit_pos < bytes.len() * 8);
}

/// M1 的**结构性边界**：CPE 的第二个声道在第一个声道的**谱数据**之后，
/// 而谱数据的位长要解完 Huffman 才知道（M2）。所以这里钉住「明确报错」——
/// **不能**静默解出半截数据。
#[test]
fn cpe_is_explicitly_unsupported() {
    let cfg = EncoderConfig {
        channels: 2,
        sample_rate: 44_100,
        ..Default::default()
    };
    let mut enc = Encoder::new(cfg).unwrap();
    let asc = enc.audio_specific_config().to_bytes();
    let dec = Decoder::new(&asc).unwrap();
    let n = 1024usize;
    let frame = Frame {
        planes: vec![
            (0..n).map(|i| (i as f32 * 0.05).sin() * 0.4).collect(),
            (0..n).map(|i| (i as f32 * 0.05).sin() * 0.35).collect(),
        ],
    };
    let bytes = enc.encode_frame(&frame).unwrap();
    let e = dec.parse_frame(&bytes).unwrap_err();
    assert!(
        format!("{e}").contains("M2"),
        "CPE 的错误信息要点明「需要 M2」：{e}"
    );
}

#[test]
fn rejects_non_lc_asc() {
    // object_type = 5（SBR）→ 明确报 Unsupported
    let mut asc = vec![0u8; 2];
    asc[0] = 5 << 3; // object_type 高 5 位
    assert!(Decoder::new(&asc).is_err());
}

#[test]
fn truncated_frame_is_an_error_not_a_panic() {
    let cfg = EncoderConfig {
        channels: 1,
        sample_rate: 44_100,
        ..Default::default()
    };
    let mut enc = Encoder::new(cfg).unwrap();
    let asc = enc.audio_specific_config().to_bytes();
    let dec = Decoder::new(&asc).unwrap();
    let n = 1024usize;
    let frame = Frame {
        planes: vec![(0..n)
            .map(|i| (i as f32 * 0.05).sin() * 0.4)
            .collect::<Vec<_>>()],
    };
    let bytes = enc.encode_frame(&frame).unwrap();
    let cut = &bytes[..bytes.len() / 3];
    assert!(dec.parse_frame(cut).is_err(), "截断的帧必须报错");
}
