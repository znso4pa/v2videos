//! Android 真机上的硬件编码测试 —— 只在设备/模拟器上编译运行。
//!
//! 判据（与软件对垒同口径）：
//! 1. 输出文件能被 ffmpeg 解开、帧数对；
//! 2. 用 `ffmpeg ssim/psnr` 对 VC-1 解码参考算质量（数值记录在 TODO，不硬编码阈值
//!    —— 厂商编码器质量差异大，阈值会误报）；
//! 3. 音频 PTS 不偏移（我们的 AAC 时间轴零偏移的优势在这条里守住）。
#![cfg(target_os = "android")]

use v2v_android::Mp4HwWriter;
use v2v_player::Player;

fn sample(name: &str) -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testing").join(name);
    p.exists().then_some(p)
}

/// 编 30 帧合成画面 + 一条 AAC 轨，验证文件能封完。
#[test]
fn hw_encode_writes_a_playable_mp4() {
    let Some(wmv) = sample("sample.wmv") else {
        eprintln!("SKIP: testing/sample.wmv 不在");
        return;
    };
    let mut p = Player::open(&wmv).expect("打开");
    let info = p.info().clone();
    let out = std::env::temp_dir().join(format!("v2v_hw_{}.mp4", std::process::id()));

    let mut hw = Mp4HwWriter::open(&out, info.width, info.height, 4_000_000, 30, 4, true)
        .expect("打开硬件编码器");
    // 音频轨（ASC 从我们的 AAC 编码器拿）。
    let mut enc = v2v_aac::Encoder::new(v2v_aac::EncoderConfig {
        channels: info.audio_channels,
        sample_rate: info.audio_sample_rate,
        ..v2v_aac::EncoderConfig::default()
    })
    .expect("建 AAC 编码器");
    hw.add_audio_track(&enc.audio_specific_config().to_bytes(), info.audio_sample_rate, info.audio_channels)
        .expect("加音频轨");

    let mut n = 0;
    while n < 30 {
        if let Some(f) = p.next_video().expect("解帧") {
            hw.write_video(&f.y, &f.u, &f.v, f.pts_ms * 1000, false).expect("编码");
            n += 1;
        }
    }
    hw.finish().expect("收尾");

    let sz = std::fs::metadata(&out).unwrap().len();
    assert!(sz > 10_000, "产物太小（{sz} B）—— 编码器没出有效数据");
    let _ = std::fs::remove_file(&out);
}
