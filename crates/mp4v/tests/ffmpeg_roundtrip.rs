//! MPEG-4 Part 2 编码器**对账 ffmpeg**。
//!
//! ## 判据是什么
//!
//! 编码器没有「参考位流」可比 —— 同一帧可以编出无数种合法的位流。
//! 唯一有意义的判据是 **ffmpeg 解得开，且解出来的画面接近原图**：
//!
//! 1. `ffmpeg -i ours.m4v -f rawvideo` **退出码 0**，stderr 里没有 `error` / `corrupt`；
//! 2. 解出来的尺寸、帧数对；
//! 3. 逐像素 PSNR 够高（阈值见 `PSNR_MIN_DB`）；
//! 4. **同一个 qscale 下和 ffmpeg 自己的 mpeg4 编码器比一比** —— 不是要求追平，
//!    而是要求「在同一量级」：差 10 dB 说明我们的编码器有实质性的错（不是调优问题）。
//!
//! 第 4 条是这套测试里最有价值的一条。只测「能解开」会漏掉一整类错：
//! 预测方向反了、DC 缩放表用错、量化公式反了 —— 这些都能解开，只是画质崩。
//!
//! ## 为什么要跑两遍（裸流 + MP4）
//!
//! 两条路会踩不同的坑：
//!
//! - **裸基本流**（`.m4v`）：VOL 在流里，解码器自己找头。测的是**位流本身**。
//! - **MP4**：VOL 搬进 `esds` 的 DSI，流里只剩 VOP。测的是**VOL 那份拷贝
//!   与流里的一致**（`decoder_specific_info()` 与 `write_vol_into()` 是同一个函数，
//!   但这条测试会把「不小心分叉」钉死）。
//!
//! ## 怎么跑
//!
//! ```bash
//! cargo test -p v2v-mp4v --test ffmpeg_roundtrip -- --nocapture
//! ```
//!
//! 需要 `ffmpeg` / `ffprobe` 在 `PATH` 上。

use std::io::Write;
use std::process::{Command, Stdio};

use v2v_mp4v::{Encoder, EncoderConfig};
use v2v_msmpeg4::Frame;

const W: u32 = 320;
const H: u32 = 240;
const FPS: u32 = 25;
const FRAMES: usize = 12;

/// 逐像素 PSNR 的下限。qscale = 4 的 MPEG-4 在 320×240 上应当远好于这个数。
const PSNR_MIN_DB: f64 = 30.0;

/// 与 ffmpeg 自己的编码器相比，允许的最大差距（dB）。
///
/// 我们只做全 I 帧、不做 dquant、不做自适应量化，所以必然差一些；
/// 但「差一点点」和「差 10 dB」是两回事 —— 后者说明有实质性的错。
const PSNR_GAP_MAX_DB: f64 = 6.0;

// ==================== 测试素材 ====================
//
// 刻意**三种结构都有**，因为每种结构考验编码器的不同部分：
//
// - 平滑斜向渐变 → AC 预测应当赢（邻居首行/首列很像）
// - 移动的亮方块 → 强边缘，AC 预测应当输（邻居完全不同）
// - 细竖条纹 → 高频，考验 RL escape 与扫描序
//
// 只喂平滑渐变会漏掉 escape 的错；只喂噪声会让「能不能解开」都成问题。

fn test_frame(t: usize) -> Frame {
    let (w, h) = (W as usize, H as usize);
    let mut y = vec![0u8; w * h];
    // 移动的方块
    let cx = (w / 2 + t * 4) % w;
    let cy = (h / 3 + t * 3) % h;
    for j in 0..h {
        for i in 0..w {
            // 平滑斜向渐变（0..200），再叠一点时间变化
            let mut v =
                (i as f64 / w as f64) * 200.0 + (j as f64 / h as f64) * 55.0 + t as f64 * 2.0;
            // 强边缘：一个 48×48 的亮方块
            if i.abs_diff(cx) < 24 && j.abs_diff(cy) < 24 {
                v = 235.0;
            }
            // 高频：每 37 像素一根细黑线，随时间平移
            if (i + t * 3) % 37 == 0 {
                v = 16.0;
            }
            y[j * w + i] = v.clamp(0.0, 255.0) as u8;
        }
    }

    let (cw, ch) = (w / 2, h / 2);
    let mut u = vec![0u8; cw * ch];
    let mut v_plane = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            let dx = i as f64 - (cw as f64 / 2.0 + t as f64 * 2.0);
            let dy = j as f64 - (ch as f64 / 2.0);
            let r = (dx * dx + dy * dy).sqrt();
            u[j * cw + i] = if r < 40.0 { 200 } else { 90 };
            v_plane[j * cw + i] = if r < 60.0 { 60 } else { 170 };
        }
    }

    Frame {
        width: W,
        height: H,
        y,
        u,
        v: v_plane,
    }
}

/// 把一帧 YUV420P 写成 `ffmpeg` 认的裸帧字节。
fn frame_bytes(f: &Frame) -> Vec<u8> {
    let mut b = f.y.clone();
    b.extend_from_slice(&f.u);
    b.extend_from_slice(&f.v);
    b
}

// ==================== ffmpeg 帮手 ====================

fn ffmpeg(args: &[&str], stdin_data: Option<&[u8]>) -> (bool, Vec<u8>, String) {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(args)
        .stdin(if stdin_data.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("起不来 ffmpeg（在 PATH 上吗？）");
    if let Some(d) = stdin_data {
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(d)
            .expect("喂不进 stdin");
    }
    let out = child.wait_with_output().expect("ffmpeg 没跑完");
    (
        out.status.success(),
        out.stdout,
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// 用 ffmpeg 把一个视频文件解成裸 YUV420P。
fn decode_to_yuv(path: &str) -> (Vec<u8>, String) {
    let (ok, data, log) = ffmpeg(
        &[
            "-v", "warning", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-",
        ],
        None,
    );
    assert!(ok, "ffmpeg 解不开 {path}：\n{log}");
    (data, log)
}

/// 算两段裸 YUV420P 的 PSNR（dB）。长度不一致直接判失败。
fn psnr(a: &[u8], b: &[u8], frames: usize) -> f64 {
    assert_eq!(a.len(), b.len(), "解码出来的字节数不一样");
    assert_eq!(a.len(), frames * (W * H * 3 / 2) as usize, "帧数不对");
    let mut se = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = f64::from(*x) - f64::from(*y);
        se += d * d;
    }
    let mse = se / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

// ==================== 测试 ====================

/// 把我们的编码结果写成一个裸 `.m4v` 基本流。
fn encode_raw_es(cfg: EncoderConfig, frames: &[Frame]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut e = Encoder::new(cfg).expect("建编码器");
    let mut es = e.elementary_stream_header();
    let mut samples = Vec::new();
    for f in frames {
        let s = e.encode_i_frame(f).expect("编一帧");
        es.extend_from_slice(&s);
        samples.push(s);
    }
    (es, samples)
}

#[test]
fn ffmpeg_decodes_our_elementary_stream() {
    let frames: Vec<Frame> = (0..FRAMES).map(test_frame).collect();
    let cfg = EncoderConfig {
        width: W,
        height: H,
        fps_num: FPS,
        ..Default::default()
    };
    let (es, _) = encode_raw_es(cfg, &frames);

    let path = std::env::temp_dir().join("v2v_mp4v_ours.m4v");
    std::fs::write(&path, &es).expect("写基本流");
    let path = path.to_str().unwrap();

    let (got, log) = decode_to_yuv(path);
    for bad in ["error", "corrupt", "invalid", "damaged"] {
        assert!(
            !log.to_lowercase().contains(bad),
            "ffmpeg 报了「{bad}」—— 位流有问题：\n{log}"
        );
    }

    let mut src = Vec::new();
    for f in &frames {
        src.extend_from_slice(&frame_bytes(f));
    }
    let p = psnr(&src, &got, FRAMES);
    println!("[裸基本流] qscale = {}，PSNR = {p:.2} dB", cfg.qscale);
    assert!(
        p >= PSNR_MIN_DB,
        "PSNR 只有 {p:.2} dB（下限 {PSNR_MIN_DB}）—— 位流能解开但画质崩了，多半是预测/量化错了"
    );
}

#[test]
fn quality_is_in_the_same_league_as_ffmpeg() {
    let frames: Vec<Frame> = (0..FRAMES).map(test_frame).collect();
    let cfg = EncoderConfig {
        width: W,
        height: H,
        fps_num: FPS,
        ..Default::default()
    };
    let (es, _) = encode_raw_es(cfg, &frames);

    let ours_path = std::env::temp_dir().join("v2v_mp4v_league_ours.m4v");
    std::fs::write(&ours_path, &es).expect("写基本流");
    let ours_path = ours_path.to_str().unwrap();

    // 同一份原始帧喂给 ffmpeg 自己的 mpeg4 编码器，用同一个 qscale。
    let mut raw = Vec::new();
    for f in &frames {
        raw.extend_from_slice(&frame_bytes(f));
    }
    let ref_path = std::env::temp_dir().join("v2v_mp4v_league_ref.m4v");
    let ref_path = ref_path.to_str().unwrap().to_string();
    let (ok, _, log) = ffmpeg(
        &[
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            &format!("{W}x{H}"),
            "-r",
            &FPS.to_string(),
            "-i",
            "-",
            "-c:v",
            "mpeg4",
            "-qscale:v",
            &cfg.qscale.to_string(),
            "-y",
            &ref_path,
        ],
        Some(&raw),
    );
    assert!(ok, "ffmpeg 自己的 mpeg4 编码器没跑起来：\n{log}");

    let (ours, _) = decode_to_yuv(ours_path);
    let (theirs, _) = decode_to_yuv(&ref_path);
    let mut src = Vec::new();
    for f in &frames {
        src.extend_from_slice(&frame_bytes(f));
    }
    let p_ours = psnr(&src, &ours, FRAMES);
    let p_theirs = psnr(&src, &theirs, FRAMES);
    println!("[同 qscale 对比] 我们 {p_ours:.2} dB / ffmpeg {p_theirs:.2} dB");
    assert!(
        p_theirs - p_ours <= PSNR_GAP_MAX_DB,
        "比 ffmpeg 差了 {:.2} dB（上限 {PSNR_GAP_MAX_DB}）—— 这不是调优能解释的差距",
        p_theirs - p_ours
    );
}

#[test]
fn vol_survives_the_trip_into_an_mp4_esds() {
    use v2v_mp4::{Mp4Muxer, VideoTrack};

    let frames: Vec<Frame> = (0..FRAMES).map(test_frame).collect();
    let cfg = EncoderConfig {
        width: W,
        height: H,
        fps_num: FPS,
        ..Default::default()
    };
    let mut e = Encoder::new(cfg).expect("建编码器");
    let dsi = e.decoder_specific_info();

    let mut mux = Mp4Muxer::new(VideoTrack {
        width: W as u16,
        height: H as u16,
        timescale: FPS * 512,
        decoder_specific_info: dsi.clone(),
        avg_bitrate: 0,
    });
    let dur = 512; // timescale = fps*512 → 每帧 512 个 tick
    for f in &frames {
        let s = e.encode_i_frame(f).expect("编一帧");
        mux.write_video(&s, true, dur).expect("写样本");
    }
    let mp4 = mux.finish();

    let path = std::env::temp_dir().join("v2v_mp4v_ours.mp4");
    std::fs::write(&path, &mp4).expect("写 mp4");
    let path = path.to_str().unwrap();

    // ffprobe 从 esds 里读出的参数必须和我们对得上 —— 这证明 DSI 里的 VOL 是完整的。
    let (ok, out, log) = ffmpeg_probe(&[
        "-v",
        "error",
        "-select_streams",
        "v",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,profile,level",
        "-of",
        "default=nw=1",
        path,
    ]);
    assert!(ok, "ffprobe 读不了 {path}：\n{log}");
    println!("[MP4] ffprobe:\n{out}");
    assert!(out.contains("codec_name=mpeg4"), "codec 不对：\n{out}");
    assert!(out.contains(&format!("width={W}")), "宽度不对：\n{out}");
    assert!(out.contains(&format!("height={H}")), "高度不对：\n{out}");
    assert!(out.contains("pix_fmt=yuv420p"), "像素格式不对：\n{out}");
    // `profile` 是从 DSI 里的 **VOS 头**读出来的 —— 只放 VOL 的话这里会是 `unknown`。
    assert!(
        out.contains("profile=Simple Profile"),
        "profile 不对（DSI 里缺 VOS 头？）：\n{out}"
    );
    assert!(out.contains("level=1"), "level 不对：\n{out}");

    let (got, log) = decode_to_yuv(path);
    assert!(!log.to_lowercase().contains("error"), "解码报了错：\n{log}");
    let mut src = Vec::new();
    for f in &frames {
        src.extend_from_slice(&frame_bytes(f));
    }
    let p = psnr(&src, &got, FRAMES);
    println!("[MP4 封装] PSNR = {p:.2} dB");
    assert!(p >= PSNR_MIN_DB, "MP4 这条路上画质崩了：{p:.2} dB");
}

fn ffprobe_probe(args: &[&str]) -> (bool, String, String) {
    let out = Command::new("ffprobe")
        .args(args)
        .output()
        .expect("起不来 ffprobe（在 PATH 上吗？）");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// `ffmpeg -i` 的 probe 分支走的是同一个 `ffprobe` 库，直接复用它省一层壳。
fn ffmpeg_probe(args: &[&str]) -> (bool, String, String) {
    ffprobe_probe(args)
}
