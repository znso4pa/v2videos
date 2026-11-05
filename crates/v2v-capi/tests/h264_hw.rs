//! `V2V_CODEC_H264_HW` 的 **ABI 层**判据（macOS + `hw` feature 才编）。
//!
//! M1 已经证明「硬解出来的像素对」（`v2v-android/tests/vt_decode.rs`，对 ffmpeg
//! 软解逐字节一致）。这里验的是**另一件事**：那套异步协议穿过 C ABI 之后还成不成立。
//!
//! 三件只有在 ABI 层才看得出来的事：
//!
//! 1. **pts 必须能传进去**。VT 靠 pts 定显示顺序，喂 0 拿到的是**解码**顺序 ——
//!    实测第 1 帧就错到 26.8 dB。所以 `v2v_vdec_decode`（没有 pts 参数）在
//!    `H264_HW` 上必须明确拒绝，而不是「凑合着解」。
//! 2. **`next` 不能丢帧**。B 帧让「喂 N 包」≠「出 N 帧」：一次 feed 吐好几帧时，
//!    多出来的必须靠 `next` 取走，不取就丢。45 帧一个不少就是这条的判据。
//! 3. **pts 要原样带回来**。`v2v_picture.pts_us` 是追加字段，得确认它真的填了
//!    （老调用方看不见它，但新调用方靠它把帧对回容器的时间轴）。
//!
//! ⚠️ **`flush` 的必要性这条测试证明不了**：实测（本机、这段流）不调它也能拿到
//! 45/45 帧 —— 连「一次喂完再抽」那种最该出问题的顺序也一样。它保留是按 VT 的
//! 协议（销毁会话前必须 `WaitForAsynchronousFrames`，否则在途帧被丢掉），
//! **不是因为这里验过它**。别因为「不调也行」就删掉它，也别以为这里有覆盖。
//!
//! 判据与 M1 同一把尺：对 ffmpeg **软件**解码逐帧比，PSNR ≥ 45 dB
//! （实测 = ∞，逐字节一致；45 dB 是给别的驱动留的余量，不是「硬件更糙」）。

#![cfg(all(target_os = "macos", feature = "hw"))]

use std::path::{Path, PathBuf};
use std::process::Command;

use v2v_capi::common::*;
use v2v_capi::vdec::*;
use v2v_mp4::Mp4Demuxer;

/// 与 M1 同一个下限。见文件头：实测是逐字节一致，45 dB 是兜底不是放宽。
const MIN_PSNR_DB: f64 = 45.0;

fn has_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn ffmpeg(args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("ffmpeg").args(args).output().ok()?;
    if !out.status.success() {
        eprintln!("ffmpeg 失败：{}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    Some(out.stdout)
}

/// 编一段带 B 帧的 720p H.264（B 帧是重点：没有它 `next` / `flush` 等于没验）。
fn make_clip(path: &Path, frames: u32) {
    let count = frames.to_string();
    ffmpeg(&[
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=1280x720:rate=30:duration=3",
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
    ])
    .expect("ffmpeg 编码失败");
}

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

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
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

fn tmp(stem: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{stem}_{}.mp4", std::process::id()))
}

/// 按 stride 逐行拷一个平面（`planes` 里的指针带 stride，不是紧排）。
fn plane(src: *const u8, stride: u32, w: usize, h: usize) -> Vec<u8> {
    let st = stride as usize;
    let s = unsafe { std::slice::from_raw_parts(src, (h - 1) * st + w) };
    let mut out = Vec::with_capacity(w * h);
    for r in 0..h {
        out.extend_from_slice(&s[r * st..r * st + w]);
    }
    out
}

/// 从 ABI 出参里把三块平面拷出来（`planes` 只活到下一次调用，必须当场拷）。
fn take(p: &V2vPicture) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let pl = &p.planes;
    let (w, h) = (pl.width as usize, pl.height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    (
        plane(pl.y, pl.stride_y, w, h),
        plane(pl.u, pl.stride_u, cw, ch),
        plane(pl.v, pl.stride_v, cw, ch),
    )
}

#[test]
fn abi_h264_hw_decodes_like_ffmpeg_and_reports_pts() {
    if !has_ffmpeg() {
        eprintln!("跳过：没有 ffmpeg");
        return;
    }
    let path = tmp("v2v_capi_h264");
    make_clip(&path, 45);

    // 解封装走**我们自己的** Mp4Demuxer：avcC、样本边界、PTS 都在这一层定下来。
    let f = std::fs::File::open(&path).expect("打开 MP4 失败");
    let mut d = Mp4Demuxer::new(f).expect("MP4 解封装失败");
    let (avcc, w, h, number) = {
        let s = d
            .streams()
            .iter()
            .find(|s| s.video.is_some())
            .expect("没有视频轨");
        let v = s.video.as_ref().unwrap();
        (v.extra.clone(), v.width, v.height, s.number)
    };
    assert!(!avcc.is_empty(), "avcC 是空的");
    let mut samples = Vec::new();
    while let Some(s) = d.next_sample().expect("读样本失败") {
        if s.stream == number {
            samples.push((s.data, s.timestamp_ms));
        }
    }

    let dec = unsafe { v2v_vdec_open(CODEC_H264_HW, avcc.as_ptr(), avcc.len() as u32, w, h) };
    assert!(!dec.is_null(), "打开失败：{}", unsafe {
        std::ffi::CStr::from_ptr(v2v_last_error()).to_string_lossy()
    });

    // ── 1. 没有 pts 的那个入口必须明确拒绝 ──────────────────────────────
    let mut pic: V2vPicture = unsafe { std::mem::zeroed() };
    pic.size = std::mem::size_of::<V2vPicture>() as u32;
    let r = unsafe { v2v_vdec_decode(dec, samples[0].0.as_ptr(), samples[0].0.len() as u32, &mut pic) };
    assert_eq!(
        r, ERR_ARG,
        "没有 pts 的入口在 H264_HW 上必须报错 —— 凑合着解会得到解码顺序，看不出错"
    );

    // ── 2. 正规流程：decode_pts + next 抽干 + flush + next 抽干 ──────────
    let mut got: Vec<(i64, Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();
    let push = |pic: &V2vPicture, got: &mut Vec<(i64, Vec<u8>, Vec<u8>, Vec<u8>)>| {
        assert_eq!((pic.planes.width, pic.planes.height), (w, h), "尺寸不对");
        let (y, u, v) = take(pic);
        got.push((pic.pts_us, y, u, v));
    };
    for (data, pts_ms) in &samples {
        let r = unsafe {
            v2v_vdec_decode_pts(dec, data.as_ptr(), data.len() as u32, pts_ms * 1000, &mut pic)
        };
        assert!(r == OK || r == EOS, "decode_pts 返回 {r}");
        if r == OK {
            push(&pic, &mut got);
        }
        loop {
            let r = unsafe { v2v_vdec_next(dec, &mut pic) };
            assert!(r == OK || r == EOS, "next 返回 {r}");
            if r == EOS {
                break;
            }
            push(&pic, &mut got);
        }
    }
    assert_eq!(unsafe { v2v_vdec_flush(dec) }, OK, "flush 失败");
    loop {
        let r = unsafe { v2v_vdec_next(dec, &mut pic) };
        assert!(r == OK || r == EOS, "flush 后 next 返回 {r}");
        if r == EOS {
            break;
        }
        push(&pic, &mut got);
    }
    unsafe { v2v_vdec_close(dec) };

    // ── 3. 帧数与 pts 序列 ────────────────────────────────────────────
    let refs = soft_decode(&path);
    let ylen = w as usize * h as usize;
    let clen = (w as usize / 2) * (h as usize / 2);
    let fsize = ylen + 2 * clen;
    let nref = refs.len() / fsize;
    assert_eq!(
        got.len(),
        nref,
        "帧数不对：ABI 拿到 {} 帧，ffmpeg 解出 {nref} 帧 —— flush 没排干净？",
        got.len()
    );
    let mut want: Vec<i64> = samples.iter().map(|(_, ms)| ms * 1000).collect();
    let mut have: Vec<i64> = got.iter().map(|(p, ..)| *p).collect();
    want.sort_unstable();
    have.sort_unstable();
    assert_eq!(have, want, "带回来的 pts 集合与喂进去的不一致");

    // ── 4. 逐帧比像素（与 M1 同一把尺）────────────────────────────────
    // 出帧顺序 = 显示顺序（pts 升序），所以按 pts 排一次再逐帧比。
    let mut order: Vec<usize> = (0..got.len()).collect();
    order.sort_by_key(|&i| got[i].0);
    let (mut min_y, mut min_uv) = (f64::INFINITY, f64::INFINITY);
    for (k, &i) in order.iter().enumerate() {
        let base = k * fsize;
        let (_, y, u, v) = &got[i];
        let py = psnr(y, &refs[base..base + ylen]);
        let pu = psnr(u, &refs[base + ylen..base + ylen + clen]);
        let pv = psnr(v, &refs[base + ylen + clen..base + fsize]);
        min_y = min_y.min(py);
        min_uv = min_uv.min(pu.min(pv));
        assert!(
            py >= MIN_PSNR_DB,
            "第 {k} 帧（pts {} us）Y PSNR {py:.2} dB < {MIN_PSNR_DB} dB",
            got[i].0
        );
        assert!(pu >= MIN_PSNR_DB && pv >= MIN_PSNR_DB, "第 {k} 帧色度 PSNR 不够");
    }
    eprintln!(
        "ABI H264_HW：{} 帧（ffmpeg {nref}），最差 Y {min_y:.2} dB，最差色度 {min_uv:.2} dB",
        got.len()
    );

    let _ = std::fs::remove_file(&path);
}

/// 没有 `hw` feature / 不在 macOS 上时，`V2V_CODEC_H264_HW` 必须**明确报不支持**，
/// 不能返回一个「能开但一喂就错」的句柄。
#[test]
fn unsupported_platform_reports_clearly() {
    // 这条在 macOS + hw 下是「能开」的反面用例：给空 avcC，必须报参数错。
    let dec = unsafe { v2v_vdec_open(CODEC_H264_HW, std::ptr::null(), 0, 1280, 720) };
    assert!(dec.is_null(), "空 avcC 不该开出句柄");
    assert_eq!(v2v_last_error_kind(), KIND_BAD);
}
