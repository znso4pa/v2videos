//! **「从任意 I 帧起解 == 整条链解到那里」吗？（MS-MPEG-4 版）**
//!
//! 这是 `v2v convert --threads > 1` 对 **MP43 输入**开并行解码的前提：每条 GOP 用一条
//! **全新的**解码器、从它的 I 帧独立解，像素必须与串行链完全一致。
//!
//! 与 `v2v-vc1` 那份 `gop_start` 不同，这里**不依赖真样本**：用本仓自己的 MP43 编码器
//! 造一条带运动的流（6 条 GOP、每条里的 P 帧都有真实残差），再对同一条流做
//! 「串行链」与「逐 I 帧新起」的对账。它验证的是**解码器的跨帧状态**在 I 帧处自洽：
//! `flipflop_rounding` 与 `HeaderState::bit_rate` 都只从 **I 帧的 ext header** 里来，
//! 所以每条 GOP 的 I 帧自己就会把它们带对。
//!
//! ⚠️ 这条测试过了**不等于** WMV1/WMV2 也能并行解码：
//! - WMV2 挂着 `IntraX8.prediction_table`（上游只在初始化清一次、**跨帧累积**），
//!   新解码器是全零表，j_type 流会解出不同的像素；
//! - WMV1 的 ext header 在**每帧**头里、`bit_rate` 有可能不在 I 帧上出现（`br == 0`），
//!   没有真样本之前不下结论。

use v2v_msmpeg4::{
    decode_i_frame, Decoder, Encoder, EncoderConfig, Frame, Version,
};

/// 一帧 64×48 的 YUV420P：一块 16×16 的方块从左上往右下平移，背景是渐变。
fn make_frame(w: u32, h: u32, t: usize) -> Frame {
    let (yw, yh) = (w as usize, h as usize);
    let (cw, ch) = (yw / 2, yh / 2);
    let mut y = vec![0u8; yw * yh];
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    let ox = (t * 3) % (yw - 20);
    let oy = (t * 2) % (yh - 20);
    for r in 0..yh {
        for c in 0..yw {
            // 渐变背景 + 平移方块（边缘陡，P 帧必须有真残差）。
            let bg = ((c * 255 / yw) as i32 + (r * 255 / yh) as i32) / 2;
            let in_box =
                c >= ox && c < ox + 20 && r >= oy && r < oy + 20;
            y[r * yw + c] = if in_box { 235 } else { bg as u8 };
        }
    }
    Frame {
        width: w,
        height: h,
        y,
        u,
        v,
    }
}

#[test]
fn fresh_decoder_at_i_frame_matches_serial_chain() {
    let (w, h) = (64u32, 48u32);
    let n_frames = 90usize;

    // ---- 用本仓的 MP43 编码器造一条流（6 条 GOP，P 帧有真实残差与运动）----
    let mut enc = Encoder::new(EncoderConfig {
        qscale: 5,
        keyframe_interval: 15,
        ..EncoderConfig::default()
    });
    let mut payloads: Vec<(Vec<u8>, bool)> = Vec::new();
    for t in 0..n_frames {
        let f = make_frame(w, h, t);
        let data = enc.encode_frame(&f).expect("编码");
        payloads.push((data, enc.last_key_frame));
    }
    let i_frames = payloads.iter().filter(|(_, k)| *k).count();
    assert!(i_frames >= 5, "底本要有足够多的 I 帧，实测 {i_frames}");

    // ---- 串行链：全部解一遍，I 帧处留一份像素 ----
    let mut serial = Decoder::with_version(Version::V3);
    let mut snapshots: Vec<Option<Frame>> = (0..n_frames).map(|_| None).collect();
    for (i, (data, _)) in payloads.iter().enumerate() {
        let f = serial.decode(data, w, h).expect("串行解码");
        if payloads[i].1 {
            snapshots[i] = Some(f);
        }
    }

    // ---- 逐 I 帧新起解码器，与串行链在同一帧的像素逐字节比 ----
    let mut checked = 0usize;
    let mut bad: Vec<usize> = Vec::new();
    for (i, (data, _)) in payloads.iter().enumerate() {
        if !payloads[i].1 {
            continue;
        }
        let mut fresh = Decoder::with_version(Version::V3);
        let f = fresh.decode(data, w, h).expect("从 I 帧起解");
        let want = snapshots[i].as_ref().expect("串行链快照");
        let diff = f
            .y
            .iter()
            .zip(&want.y)
            .chain(f.u.iter().zip(&want.u))
            .chain(f.v.iter().zip(&want.v))
            .filter(|(a, b)| a != b)
            .count();
        if diff != 0 {
            bad.push(i);
        }
        checked += 1;
    }
    assert!(
        bad.is_empty(),
        "从 I 帧起解与整条链不一致（帧号）：{bad:?}（查了 {checked} 条）\n\
         说明 MS-MPEG-4 有跨 GOP 的状态没补上 —— 并行解码之前必须先处理它"
    );
    assert!(checked >= 5);

    // `decode_i_frame` 是「只解这一帧」的入口，语义上等价于新解码器解 I 帧 ——
    // 拿它再对一遍，防止将来有人把它改成有状态的。
    let (data, _) = &payloads[0];
    let via_free = decode_i_frame(data, w, h).expect("自由函数解 I 帧");
    assert_eq!(via_free.y, snapshots[0].as_ref().unwrap().y);
}
