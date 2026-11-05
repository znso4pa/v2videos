//! **A1 判据：VideoToolbox 硬件解码 vs ffmpeg 软解，逐帧 PSNR。**
//!
//! ## 实测结果（先看这里，再看判据）
//!
//! Apple Silicon + macOS 26 + ffmpeg 9.0.1：**逐字节相同，PSNR = ∞**。
//! 720p / 1080p、合成画面 / 真内容、各 60~90 帧，一帧不差。
//!
//! 这不是测量出了错，而是 H.264 的应有结果：反变换、环路滤波、帧内预测
//! 在标准里都是**精确整数运算**（不像某些编码格式允许近似），所以任何两个
//! 符合标准的解码器必须逐字节一致。硬件实现只是更快，不是「更粗糙」。
//!
//! 原来 TODO 里写的「硬件解码有舍入，不能要求 0 差」是**错的**，已改。
//!
//! ## 那为什么阈值还是 45 dB
//!
//! 因为断言写「必须逐字节相同」会把一个**实现差异**当成 bug：别的 macOS
//! 版本、别的驱动、别的 GPU 上完全可能出现 ±1 LSB（标准允许的最后一个
//! 自由度是色度上采样，虽然这里不涉及）。45 dB ≈ 每样本平均误差 0.45
//! 灰度级，是「不影响任何下游」的下限。
//!
//! 它仍然**紧得够用**：真错的时候会掉到 20–30 dB 那一档。实测（把 Y 平面
//! 的取址错开一行）：**31.69 dB**，测试当场红。相邻帧错位则是 11–27 dB。
//! 45 dB 把「坏了」和「舍入不同」分得很干净 —— 谁想调低它，先解释为什么。
//!
//! ## 位流来自 ffmpeg，不是我们自己
//!
//! A3 M3 的教训（见 MEMORY.md）：自编自解只能证明「两头一致地错」。
//! 这里位流一律由 **ffmpeg 的 x264** 编码，我们只负责解 —— 参考画面也来自
//! ffmpeg 的**软件**解码器（`-c:v h264`，不是 videotoolbox）。
//!
//! 链路：ffmpeg 编 MP4 → **我们的 `Mp4Demuxer`** 取 avcC + 样本 →
//! **VideoToolbox 硬解** → 对 ffmpeg 软解的 yuv420p 算 PSNR。
//!
//! `testsrc2` 两个分辨率常跑（合成画面、几秒钟）；`sample.wmv` 真内容标
//! `#[ignore]`（要解 144 MB 的源，`--ignored` 时跑）。

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

use v2v_android::vt::VideoToolboxDecoder;
use v2v_mp4::Mp4Demuxer;

/// 硬件解码的判据。见文件头：45 dB 不是放宽，是这个被测物的正确口径。
const MIN_PSNR_DB: f64 = 45.0;

fn has_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn sample(name: &str) -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testing")
        .join(name);
    p.exists().then_some(p)
}

/// 临时文件路径（每次调用给不同的 `n`，避免并行测试互相覆盖）。
/// 必须带 `.mp4` 后缀：ffmpeg 靠扩展名选封装器。
fn tmp(stem: &str, n: u32) -> PathBuf {
    std::env::temp_dir().join(format!("{stem}_{}_{n}.mp4", std::process::id()))
}

fn ffmpeg(args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("ffmpeg").args(args).output().ok()?;
    if !out.status.success() {
        eprintln!("ffmpeg 失败：{}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    Some(out.stdout)
}

/// 用 ffmpeg 从 `src_args` 编一段 H.264 MP4（x264，带 B 帧）。
///
/// B 帧是刻意留着的：它会逼出「喂进去的样本 ≠ 出来的帧」这种情形，
/// 正好验 `flush` 有没有把在途帧排干净。
fn encode_h264(path: &Path, src_args: &[&str], size: &str, frames: u32) {
    let scale = format!("scale={size}");
    let count = frames.to_string();
    let mut args = vec!["-v", "error", "-y"];
    args.extend_from_slice(src_args);
    args.extend_from_slice(&[
        "-vf",
        &scale,
        "-frames:v",
        &count,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-bf",
        "3",
        "-g",
        "30",
        path.to_str().unwrap(),
    ]);
    ffmpeg(&args).expect("ffmpeg 编码失败");
}

/// ffmpeg **软件**解码到裸 yuv420p（参考画面）。显式 `-c:v h264`：
/// 不加这一项在某些机器上会走 videotoolbox，那就变成「自己跟自己比」了。
fn soft_decode(path: &Path) -> Vec<u8> {
    ffmpeg(&[
        "-v",
        "error",
        "-c:v",
        "h264",
        "-i",
        path.to_str().unwrap(),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-",
    ])
    .expect("ffmpeg 软解失败")
}

/// 用**我们的** MP4 解封装取 avcC 与样本（解码顺序）。
///
/// 刻意不复用 ffmpeg 的输出：解封装这一层也在被测范围里 —— 长度前缀宽度、
/// 样本边界、PTS 都在这里定下来。
fn demux(path: &Path) -> (Vec<u8>, Vec<(Vec<u8>, i64)>) {
    let f = std::fs::File::open(path).expect("打开 MP4 失败");
    let mut d = Mp4Demuxer::new(f).expect("我们的 MP4 解封装失败");
    let video = d
        .streams()
        .iter()
        .find(|s| s.video.is_some())
        .expect("没有视频轨");
    let avcc = video.video.as_ref().unwrap().extra.clone();
    let number = video.number;
    assert!(!avcc.is_empty(), "avcC 是空的（样本描述里没有 SPS/PPS）");

    let mut samples = Vec::new();
    while let Some(s) = d.next_sample().expect("读样本失败") {
        if s.stream == number {
            samples.push((s.data, s.timestamp_ms));
        }
    }
    assert!(!samples.is_empty(), "解封装没解出样本");
    (avcc, samples)
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "两个平面长度不一致：{} vs {}", a.len(), b.len());
    let mut se = 0u64;
    for i in 0..a.len() {
        let d = i32::from(a[i]) - i32::from(b[i]);
        se += (d * d) as u64;
    }
    let mse = se as f64 / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// 解码一段 MP4，返回 (帧, Y PSNR 最差值, UV PSNR 最差值)。
///
/// 逐帧比而不是整段比：整段会把「只有某几帧错」摊薄成一个看起来还行的数
/// —— 而那种错正是最需要被抓到的（A3 M3 的 1585 LSB 就只在 LONG_STOP 帧上）。
fn decode_and_compare(path: &Path, w: u32, h: u32) -> (usize, usize, f64, f64) {
    let (avcc, samples) = demux(path);
    let mut dec = VideoToolboxDecoder::open(&avcc).expect("打开 VideoToolbox 解码会话失败");

    let mut frames = Vec::new();
    for (data, pts_ms) in &samples {
        frames.extend(dec.feed(data, pts_ms * 1000).expect("喂样本失败"));
    }
    frames.extend(dec.flush().expect("flush 失败"));

    // VT 正常情况下就按显示顺序吐帧；这里排一次是为了让「顺序不同」和
    // 「画面不同」两种错能分开报 —— 前者会在帧数/尺寸断言上先炸。
    frames.sort_by_key(|f| f.pts_us);

    let refs = soft_decode(path);
    let (wu, hu) = (w as usize, h as usize);
    let ylen = wu * hu;
    let clen = (wu / 2) * (hu / 2);
    let fsize = ylen + 2 * clen;
    let nref = refs.len() / fsize;

    // 帧数：允许 ±1（解码器在流的首尾对「未输出帧」的处理没有强制约定）。
    let diff = frames.len().abs_diff(nref);
    assert!(
        diff <= 1,
        "帧数不一致：我们 {} 帧，ffmpeg {} 帧（差 {diff}）—— 差得多说明样本没喂全或 flush 没排干净",
        frames.len(),
        nref
    );

    let n = frames.len().min(nref);

    // **尺子自检**：参考画面的相邻两帧必须明显不同。内容若近乎静止，
    // 「逐字节一致」就证明不了任何事 —— A3 M3 的教训：静音区间里比 PCM，
    // 连续两次被骗过。这一条保证「错一帧」一定会被下面的 PSNR 抓到。
    if nref >= 2 {
        let p01 = psnr(&refs[..ylen], &refs[fsize..fsize + ylen]);
        eprintln!("尺子自检：参考画面相邻两帧 Y PSNR {p01:.2} dB");
        assert!(
            p01.is_finite() && p01 < 40.0,
            "参考画面相邻帧几乎相同（{p01:.2} dB）—— 这段内容验不出帧错位，换素材"
        );
    }
    let (mut min_y, mut min_uv) = (f64::INFINITY, f64::INFINITY);
    for i in 0..n {
        let f = &frames[i];
        assert_eq!(
            (f.width, f.height),
            (w, h),
            "第 {i} 帧尺寸不对（clean aperture 没裁到显示尺寸？）"
        );
        assert_eq!(f.y.len(), ylen, "第 {i} 帧 Y 平面长度不对");
        assert_eq!(f.u.len(), clen, "第 {i} 帧 U 平面长度不对");
        assert_eq!(f.v.len(), clen, "第 {i} 帧 V 平面长度不对");

        let base = i * fsize;
        let py = psnr(&f.y, &refs[base..base + ylen]);
        let pu = psnr(&f.u, &refs[base + ylen..base + ylen + clen]);
        let pv = psnr(&f.v, &refs[base + ylen + clen..base + fsize]);
        if py < min_y {
            min_y = py;
        }
        let uv = pu.min(pv);
        if uv < min_uv {
            min_uv = uv;
        }
        assert!(
            py >= MIN_PSNR_DB,
            "第 {i} 帧（pts {} us）Y 平面 PSNR {py:.2} dB < {MIN_PSNR_DB} dB —— 这是真错，不是舍入",
            f.pts_us
        );
        assert!(
            uv >= MIN_PSNR_DB,
            "第 {i} 帧（pts {} us）色度 PSNR {uv:.2} dB < {MIN_PSNR_DB} dB",
            f.pts_us
        );
    }
    (frames.len(), nref, min_y, min_uv)
}

/// 合成画面 → 720p H.264 → 我们解。轻量，常跑。
#[test]
fn decodes_ffmpeg_720p_like_ffmpeg_soft() {
    if !has_ffmpeg() {
        eprintln!("跳过：没有 ffmpeg");
        return;
    }
    let path = tmp("v2v_vtdec_720p.mp4", 0);
    encode_h264(
        &path,
        &["-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=2"],
        "1280x720",
        60,
    );

    let (ours, theirs, min_y, min_uv) = decode_and_compare(&path, 1280, 720);
    eprintln!("720p：我们 {ours} 帧 / ffmpeg {theirs} 帧，最差 Y {min_y:.2} dB，最差色度 {min_uv:.2} dB");

    let _ = std::fs::remove_file(&path);
}

/// 1080p：**出来的必须是 1080，不是 1088**。
///
/// H.264 的宏块高度是 16 的倍数，1080 的流一定按 1088 编码、靠 SPS 里的
/// `frame_cropping` 声明显示高度。少了这一步就是「画面看着完全对、PSNR
/// 只有 30 dB」—— 逐帧比对整体错位 8 行，看起来像解码器坏了。
///
/// ⚠️ 实测这条路径**不由我们的裁剪代码保证**：VT 自己就把 `frame_cropping`
/// 折进了 `CVPixelBuffer` 的尺寸（拿到的是 1920x1080，不是 1088）。
/// `pack_yuv420` 里的 clean aperture 分支是为「别的版本/驱动给 1088」留的
/// 兜底，在本机跑不到。所以这条测试验的是**契约**（出来是显示尺寸），
/// 不是「裁剪代码被执行过」—— 后者在本机没法验，别假装验了。
#[test]
fn decodes_ffmpeg_1080p_cropped_to_display_size() {
    if !has_ffmpeg() {
        eprintln!("跳过：没有 ffmpeg");
        return;
    }
    let path = tmp("v2v_vtdec_1080p.mp4", 0);
    encode_h264(
        &path,
        &["-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=2"],
        "1920x1080",
        60,
    );

    let (ours, theirs, min_y, min_uv) = decode_and_compare(&path, 1920, 1080);
    eprintln!(
        "1080p：我们 {ours} 帧 / ffmpeg {theirs} 帧，最差 Y {min_y:.2} dB，最差色度 {min_uv:.2} dB"
    );

    let _ = std::fs::remove_file(&path);
}

/// 真样本对账：`sample.wmv`（真内容）→ ffmpeg 编成 720p 与 1080p H.264 → 我们解。
///
/// 合成画面（`testsrc2`）是平的，很多解码错误在平画面上看不出来；真内容有
/// 细节、有运动，PSNR 掉下来才说明问题。标 `#[ignore]`：要解 144 MB 的源。
///
/// **取 `-ss 10` 那 3 秒**：这段片的开头是接近静止的标题画面（实测相邻帧
/// 47.9 dB），在那上面「逐字节一致」什么都证明不了。10 s 处是全片运动最猛的
/// 一段（相邻帧 11.4 dB），帧错位一定被抓到。
#[test]
#[ignore]
fn decodes_real_content_at_720p_and_1080p() {
    if !has_ffmpeg() {
        eprintln!("跳过：没有 ffmpeg");
        return;
    }
    let Some(src) = sample("sample.wmv") else {
        eprintln!("跳过：没有 testing/sample.wmv");
        return;
    };

    for (i, (size, w, h)) in [("1280x720", 1280u32, 720u32), ("1920x1080", 1920, 1080)]
        .into_iter()
        .enumerate()
    {
        let path = tmp("v2v_vtdec_real.mp4", i as u32);
        encode_h264(
            &path,
            &["-ss", "10", "-i", src.to_str().unwrap(), "-t", "3", "-an"],
            size,
            90,
        );

        let (ours, theirs, min_y, min_uv) = decode_and_compare(&path, w, h);
        eprintln!(
            "真内容 {size}：我们 {ours} 帧 / ffmpeg {theirs} 帧，最差 Y {min_y:.2} dB，最差色度 {min_uv:.2} dB"
        );

        let _ = std::fs::remove_file(&path);
    }
}

/// 会话确实落在硬件解码器上。
///
/// 单独一条：PSNR 全绿但实际在跑软解，会让「A1 = 硬件解码」这个结论
/// 从根上不成立 —— 而那种情况在输出上看不出任何异常。
#[test]
fn session_reports_hardware_decoder() {
    if !has_ffmpeg() {
        eprintln!("跳过：没有 ffmpeg");
        return;
    }
    let path = tmp("v2v_vtdec_hw.mp4", 0);
    encode_h264(
        &path,
        &["-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=1"],
        "1280x720",
        30,
    );

    let (avcc, samples) = demux(&path);
    let mut dec = VideoToolboxDecoder::open(&avcc).expect("打开解码会话失败");
    // 会话刚建好时属性可能还没定（VT 在第一个样本后才决定走哪条路）。
    dec.feed(&samples[0].0, samples[0].1 * 1000).expect("喂样本失败");
    let hw = dec.is_hardware();
    eprintln!("is_hardware = {hw:?}");
    assert_eq!(
        hw,
        Some(true),
        "会话没有落在硬件解码器上（Some(false) = 软件回退，None = 系统查不到该属性）"
    );

    let _ = std::fs::remove_file(&path);
}
