//! **收尾那一帧补零，不能把上一块的后半段吃掉。**
//!
//! ## 这条判据钉的是什么
//!
//! AAC 是 MDCT + 50% 重叠：第 `k` 块的重建 = `tail(帧 k) + head(帧 k+1)`。
//! 收尾时（`v2v-cli` 的 `flush_aac`）要再推一帧**全零**，把最后一块从编码器里
//! 挤出来 —— 而那一帧走的是 **L/R 域**（`use_midside` 对全零帧返回 false）。
//!
//! 如果编码器把「原始域历史」（`ChannelState::overlap`）写成了 M/S，这一帧的
//! MDCT 左半就是错的，于是它交出的 `head` 不再是「最后一块的后半段」而是
//! **零的淡入** —— 重建出来的最后一块只剩 `tail` 的淡出，听起来是一段淡出到静音。
//!
//! 症状很窄：只错**一个声道**、只错**最后一块**。原因是 M/S 帧把 `S` 灌进了
//! 右声道的 L/R 历史，而 `L == R` 的素材上 `S ≡ 0` —— 历史被清零。
//!
//! 实测（2026-11-26）：`Ogg → MP4(AAC)` 的端到端测试上，末尾 2048 个交织样本的
//! SNR 从 70 dB 掉到 **3.4 dB**，而左声道完好（只查左声道就完全看不见）。
//!
//! ## 为什么不拿 ffmpeg 当裁判
//!
//! 这个错**在编码器里** —— 位流本身就是错的。两个解码器解同一个错的位流会得到
//! 同一个错结果，所以「我们对 ffmpeg」的判据（`decode_pcm.rs`）恒绿。只有
//! 「解回来对**原始输入**」才看得见。
//!
//! ```bash
//! cargo test -p v2v-aac --test ms_flush_tail -- --nocapture
//! ```
//!
//! 不需要外部工具。

use v2v_aac::{Decoder, Encoder, EncoderConfig, Frame};

/// 一帧的采样点数（AAC 长窗）。
const N: usize = 1024;

/// 单声道内容装在立体声里（`L == R`）—— `S ≡ 0`，M/S 帧会把右声道的 L/R 历史清零。
///
/// 这不是编出来的特例：真素材里大量存在（手机录音、单声道源转立体声、
/// 两边共用一个话筒），而**它恰好是这个 bug 唯一会显形的形状**。
fn mono_into_stereo(frames: usize) -> Vec<Frame> {
    (0..frames)
        .map(|t| {
            let p: Vec<f32> = (0..N)
                .map(|i| {
                    let k = (t * N + i) as f32;
                    (k * 0.0627).sin() * 6000.0
                })
                .collect();
            Frame {
                planes: vec![p.clone(), p],
            }
        })
        .collect()
}

/// 每个 `(帧, 声道)` 的 SNR。
///
/// 块序：`block k = tail(帧 k) + head(帧 k+1)`。所以 `input[k]` 由 `帧 k` 与
/// `帧 k+1` 一起重建 —— 喂了 `F` 个内容帧 + 1 个补零帧，`input[0..F]` **全都**
/// 能完整重建（最后一块要的正是那个补零帧）。
///
/// ⚠️ 第一版把最后一块当成「重建不出来」跳过了 —— 而它恰好是唯一被这个 bug
/// 打中的那一块（补零帧的 `head` 错，吃到的是**上一块**的后半段）。判据挑错了
/// 区间，bug 改回去它照样绿。
fn snr(a: &[f32], b: &[f32]) -> f64 {
    let (mut se, mut sp) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        se += (x - y) * (x - y);
        sp += y * y;
    }
    if se == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sp / se).log10()
}

#[test]
fn the_flush_frame_does_not_eat_the_last_block() {
    const FRAMES: usize = 6;
    let input = mono_into_stereo(FRAMES);
    let mut enc = Encoder::new(EncoderConfig::default()).expect("建编码器");
    let asc = enc.audio_specific_config().to_bytes();

    let mut streams = Vec::new();
    for f in &input {
        streams.push(enc.encode_frame(f).expect("编一帧"));
    }
    // 收尾：推一帧全零，把最后一块挤出来（与 `flush_aac` 第二步同一个动作）。
    let zeros = Frame {
        planes: vec![vec![0f32; N]; 2],
    };
    streams.push(enc.encode_frame(&zeros).expect("编补零帧"));

    let mut dec = Decoder::new(&asc).expect("建解码器");
    let mut out = vec![Vec::new(); dec.channels as usize];
    for s in &streams {
        let pcm = dec.decode_frame(s).expect("解一帧");
        assert_eq!(pcm.len(), out.len());
        for (o, p) in out.iter_mut().zip(pcm) {
            o.extend_from_slice(&p);
        }
    }
    assert_eq!(out[0].len(), (FRAMES + 1) * N, "解码出来的块数不对");

    // `input[k]` 对应解码出来的第 `k+1` 块；补零帧把最后一块也补齐了。
    let mut worst = f64::INFINITY;
    for k in 0..FRAMES {
        for ch in 0..2 {
            let got = &out[ch][(k + 1) * N..(k + 2) * N];
            let want = &input[k].planes[ch];
            let s = snr(got, want);
            println!("第 {k} 块 第 {ch} 声道：SNR = {s:.1} dB");
            worst = worst.min(s);
        }
    }
    // 128 kbps 下的纯正弦，正常在 40 dB 以上；出错时（淡出到静音）是负的。
    // 阈值取 20 dB —— 只判「这块还在不在」，不当成音质判据。
    assert!(
        worst > 20.0,
        "最差的一块只有 {worst:.1} dB —— 收尾补零帧把上一块的后半段吃掉了？\
         （`ChannelState::overlap` 是不是被写成了 M/S）"
    );
}
