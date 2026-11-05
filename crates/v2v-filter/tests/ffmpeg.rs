//! 对 ffmpeg 同名滤镜的对账（`scale` / `format`）。
//!
//! 判据是 **PSNR 门限**而不是「逐字节相同」—— 理由必须写清楚，否则以后有人会把
//! 门限一路放宽到「能过」：
//!
//! - 缩放是**重采样**，两个实现只要抽头数、相位、定点精度、边界策略里有一样不同，
//!   输出就不可能逐字节一致。swscale 的 bilinear 用 15 位中间精度、色度走它自己
//!   的滤波路径；我们用 8 位小数的 u16 中间平面、色度与亮度同一个核。**都不是错**。
//! - 所以这里问的是「**没有实质差异**」：PSNR 高到「差异只在舍入量级」。
//!   作为参照，错半个像素（相位错）会掉到 25 dB 上下，错一个抽头会掉到 20 dB 以下。
//!
//! 没有 ffmpeg 就 `SKIP`（CI 的 macOS job 没装 ffmpeg；Linux job 装了）。

use std::io::Write;
use std::process::{Command, Stdio};

use v2v_filter::{resize, Algo};
use v2v_kernel::mc::Frame;

const SW: usize = 320;
const SH: usize = 240;

/// ffmpeg 在不在。不在就打 `SKIP` 并让测试通过（仓里的统一做法）。
fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// 跑一次 ffmpeg：`input` 为 `Some` 时从 stdin 喂，返回 stdout 的原始字节。
fn ffmpeg(args: &[&str], input: Option<&[u8]>) -> Vec<u8> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-loglevel", "error"]).args(args);
    if input.is_some() {
        cmd.stdin(Stdio::piped());
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("起不来 ffmpeg");
    if let Some(data) = input {
        // 数据量很小（一帧 rawvideo），写满管道不会死锁。
        child
            .stdin
            .take()
            .expect("stdin")
            .write_all(data)
            .expect("喂 ffmpeg");
    }
    let out = child.wait_with_output().expect("等 ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg 失败：{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// 造一帧源：`testsrc2`（有渐变、有细节、有彩色块），比纯色或棋盘更有区分度。
fn source_frame() -> Frame {
    let raw = ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=size={SW}x{SH}:rate=1"),
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ],
        None,
    );
    let yn = SW * SH;
    let cn = (SW / 2) * (SH / 2);
    assert_eq!(raw.len(), yn + 2 * cn, "源帧长度不对");
    Frame {
        width: SW as u32,
        height: SH as u32,
        y: raw[..yn].to_vec(),
        u: raw[yn..yn + cn].to_vec(),
        v: raw[yn + cn..].to_vec(),
    }
}

/// 4:2:0 的打包顺序（Y 平面 + U 平面 + V 平面），与 rawvideo 的布局一致。
fn pack(f: &Frame) -> Vec<u8> {
    let mut out = Vec::with_capacity(f.y.len() + f.u.len() + f.v.len());
    out.extend_from_slice(&f.y);
    out.extend_from_slice(&f.u);
    out.extend_from_slice(&f.v);
    out
}

/// 平面级 PSNR。
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "比的两个平面长度不同");
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// 把三个平面的 PSNR 打出来（`--nocapture` 才看得见），并返回亮度那个。
fn report(tag: &str, ours: &Frame, theirs: &[u8]) -> f64 {
    let yn = SW * SH * 4;
    let cn = (SW * 2 / 2) * (SH * 2 / 2);
    let (y, rest) = theirs.split_at(yn);
    let (u, v) = rest.split_at(cn);
    let py = psnr(&ours.y, y);
    let pu = psnr(&ours.u, u);
    let pv = psnr(&ours.v, v);
    eprintln!("{tag}: Y {py:.2} dB / U {pu:.2} dB / V {pv:.2} dB");
    py
}

/// 双线性 2× 放大对 ffmpeg 的 `scale=...:flags=bilinear`。
///
/// 门限 40 dB：这个量级意味着差异只在「最后一位舍入」，而不是滤波形状不同。
/// 作为对照，把相位挪半个像素会掉到 25 dB 上下（见 `scale.rs` 的相位判据）。
#[test]
fn bilinear_matches_ffmpeg_scale() {
    if !have_ffmpeg() {
        eprintln!("SKIP：没有 ffmpeg");
        return;
    }
    let src = source_frame();
    let theirs = ffmpeg(
        &[
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            &format!("{SW}x{SH}"),
            "-i",
            "-",
            "-vf",
            &format!("scale={}:{}:flags=bilinear", SW * 2, SH * 2),
            "-f",
            "rawvideo",
            "-",
        ],
        Some(&pack(&src)),
    );
    let ours = resize(&src, (SW * 2) as u32, (SH * 2) as u32, Algo::Bilinear).unwrap();
    let py = report("bilinear 2x", &ours, &theirs);
    assert!(py >= 40.0, "亮度 PSNR {py:.2} dB 低于 40 dB");
}

/// 双三次对 ffmpeg 的 `flags=bicubic`，**并把参数对齐到 Catmull-Rom**。
///
/// ffmpeg 默认 `B = 0, C = 0.6`，我们是 `B = 0, C = 0.5` —— 不比参数就是比两种算法，
/// 掉到 30 dB 也说明不了谁错。所以显式传 `param0=0:param1=0.5`。
///
/// 注意 swscale 的 bicubic 在 4:2:0 上对色度另有处理，U/V 的 PSNR 会比 Y 低一截，
/// 这里只卡亮度。
#[test]
fn bicubic_matches_ffmpeg_scale_with_same_taps() {
    if !have_ffmpeg() {
        eprintln!("SKIP：没有 ffmpeg");
        return;
    }
    let src = source_frame();
    let theirs = ffmpeg(
        &[
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            &format!("{SW}x{SH}"),
            "-i",
            "-",
            "-vf",
            &format!(
                "scale={}:{}:flags=bicubic:param0=0:param1=0.5",
                SW * 2,
                SH * 2
            ),
            "-f",
            "rawvideo",
            "-",
        ],
        Some(&pack(&src)),
    );
    let ours = resize(&src, (SW * 2) as u32, (SH * 2) as u32, Algo::Bicubic).unwrap();
    let py = report("bicubic 2x (B=0,C=0.5)", &ours, &theirs);
    assert!(py >= 40.0, "亮度 PSNR {py:.2} dB 低于 40 dB");
}

/// 缩小（2:1）对 ffmpeg 的双线性。缩小时边界策略的影响比放大更明显，
/// 这条能抓「边缘补黑而不是 clamp」那类错（补黑会在四边压出一圈暗带）。
#[test]
fn bilinear_downscale_matches_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("SKIP：没有 ffmpeg");
        return;
    }
    let src = source_frame();
    let (dw, dh) = (SW / 2, SH / 2);
    let theirs = ffmpeg(
        &[
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            &format!("{SW}x{SH}"),
            "-i",
            "-",
            "-vf",
            &format!("scale={dw}:{dh}:flags=bilinear"),
            "-f",
            "rawvideo",
            "-",
        ],
        Some(&pack(&src)),
    );
    let ours = resize(&src, dw as u32, dh as u32, Algo::Bilinear).unwrap();
    let yn = dw * dh;
    let cn = (dw / 2) * (dh / 2);
    let py = psnr(&ours.y, &theirs[..yn]);
    eprintln!(
        "bilinear 1/2: Y {py:.2} dB / U {:.2} dB / V {:.2} dB",
        psnr(&ours.u, &theirs[yn..yn + cn]),
        psnr(&ours.v, &theirs[yn + cn..])
    );
    assert!(py >= 40.0, "亮度 PSNR {py:.2} dB 低于 40 dB");
}
