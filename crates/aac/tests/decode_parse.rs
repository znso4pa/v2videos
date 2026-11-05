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

/// 双声道：M2 之后两个声道都能解（第二个声道的位置依赖第一个声道的谱数据长度）。
#[test]
fn parses_own_stereo_frame() {
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
    let info = dec.parse_frame(&bytes).unwrap();
    assert_eq!(info.elements.len(), 1);
    let e = &info.elements[0];
    assert_eq!(e.kind, 1, "双声道应当是 CPE");
    assert_eq!(e.channels.len(), 2);
    assert_eq!(e.spectra.len(), 2, "两个声道各一份谱");
    for s in &e.spectra {
        assert_eq!(s.len(), 1024);
    }
    // 位流被恰好读完：只剩字节对齐的补零。
    let used = info.bits_used;
    let total = bytes.len() * 8;
    assert!(
        used <= total && total - used < 8,
        "位流没读准：用了 {used} 位 / 共 {total} 位"
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

/// 多种信号各解一遍：静音 / 正弦 / 噪声 / 瞬态（瞬态会触发短窗，报 Unsupported 也算对）。
///
/// 判据是「**位流被恰好读完**」：任何一处 Huffman 读错，位游标就会漂，
/// 要么解析失败，要么 `bits_used` 与总位数差出 8 位以上。
#[test]
fn parses_many_signal_types() {
    let cfg = EncoderConfig {
        channels: 1,
        sample_rate: 44_100,
        ..Default::default()
    };
    let mut enc = Encoder::new(cfg).unwrap();
    let asc = enc.audio_specific_config().to_bytes();
    let dec = Decoder::new(&asc).unwrap();
    let n = 1024usize;

    let mut state = 12345u32;
    let mut ok = 0;
    for t in 0..8usize {
        let plane: Vec<f32> = (0..n)
            .map(|i| match t % 4 {
                0 => 0.0, // 静音：全零带
                1 => (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.5,
                2 => {
                    state = state.wrapping_mul(1103515245).wrapping_add(12345);
                    ((state >> 16) as f32 / 32768.0 - 1.0) * 0.2
                }
                _ => {
                    // 瞬态：前 100 个样本很强 → 编码器会走短窗
                    if i < 100 {
                        ((i as f32) * 0.9).sin() * 0.9
                    } else {
                        0.0
                    }
                }
            })
            .collect();
        let bytes = enc.encode_frame(&Frame { planes: vec![plane] }).unwrap();
        match dec.parse_frame(&bytes) {
            Ok(info) => {
                let total = bytes.len() * 8;
                assert!(
                    info.bits_used <= total && total - info.bits_used < 8,
                    "第 {t} 帧位游标漂了：用了 {} / 共 {total}",
                    info.bits_used
                );
                ok += 1;
            }
            Err(e) => {
                // 短窗还没做（M3）：明确报 Unsupported 可以接受，其它错误不行。
                assert!(
                    format!("{e}").contains("短窗"),
                    "第 {t} 帧不该出这个错：{e}"
                );
            }
        }
    }
    assert!(ok >= 4, "至少应当有一半的帧能解析（实际 {ok}）");
}
