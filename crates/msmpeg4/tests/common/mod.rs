//! 集成测试共用的小工具（`tests/common/mod.rs` 不会被当成独立测试目标）。
//!
//! 这里放的是「和 ffmpeg 对账」的那一套：拿样本、取帧、跑 ffmpeg、逐像素比。
//! 编码器和解码器的测试都用它。

#![allow(dead_code)] // 共用工具箱：每个测试目标只用到其中一部分

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

use std::sync::atomic::{AtomicUsize, Ordering};

use v2v_asf::{AsfDemuxer, StreamKind};

/// 进程内唯一的临时文件路径。
///
/// **必须唯一**：`cargo test` 默认多线程跑，同一个测试二进制里的两个测试如果共用
/// 一个临时路径，一个在写、另一个的 ffmpeg 在读 —— 会解出垃圾（踩过一次）。
pub fn unique_temp(prefix: &str, ext: &str) -> std::path::PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{}_{n}.{ext}", std::process::id()))
}

/// 把**一帧裸 MP43** 包成一个最小 AVI，好让 ffmpeg 解它。
///
/// 为什么需要这个：手头 .wmv 里的帧数据是**跨 ASF 包分片**的，没法在文件里直接改一位；
/// 而 AVI 是 RIFF，单帧就是一个 `00dc` 块 —— 想喂什么就喂什么（补丁帧、手写帧都行）。
pub fn wrap_single_frame_avi(frame: &[u8], w: u32, h: u32) -> Vec<u8> {
    wrap_avi(&[frame], w, h)
}

/// 把**若干帧裸 MP43** 包成一个最小 AVI（I 在前、P 在后，按顺序）。
pub fn wrap_avi(frames: &[&[u8]], w: u32, h: u32) -> Vec<u8> {
    fn u32le(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn u16le(v: &mut Vec<u8>, x: u16) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn fourcc(v: &mut Vec<u8>, s: &[u8; 4]) {
        v.extend_from_slice(s);
    }
    fn chunk(v: &mut Vec<u8>, id: &[u8; 4], body: &[u8]) {
        fourcc(v, id);
        u32le(v, body.len() as u32);
        v.extend_from_slice(body);
        if body.len() % 2 == 1 {
            v.push(0); // RIFF 块要偶数长度
        }
    }
    fn list(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut inner = Vec::new();
        fourcc(&mut inner, kind);
        inner.extend_from_slice(body);
        let mut out = Vec::new();
        fourcc(&mut out, b"LIST");
        u32le(&mut out, inner.len() as u32);
        out.extend_from_slice(&inner);
        out
    }

    // ---- strf：BITMAPINFOHEADER（40 字节）----
    let mut strf = Vec::new();
    u32le(&mut strf, 40);
    u32le(&mut strf, w);
    u32le(&mut strf, h);
    u16le(&mut strf, 1); // planes
    u16le(&mut strf, 24); // bitcount
    fourcc(&mut strf, b"MP43"); // 就是 MS-MPEG-4 v3 的 AVI fourcc
    u32le(&mut strf, w * h * 3);
    u32le(&mut strf, 0);
    u32le(&mut strf, 0);
    u32le(&mut strf, 0);
    u32le(&mut strf, 0);
    assert_eq!(strf.len(), 40);

    // ---- strh：AVIStreamHeader（56 字节）----
    let mut strh = Vec::new();
    fourcc(&mut strh, b"vids");
    fourcc(&mut strh, b"MP43");
    u32le(&mut strh, 0); // flags
    u16le(&mut strh, 0); // priority
    u16le(&mut strh, 0); // language
    u32le(&mut strh, 0); // initial frames
    u32le(&mut strh, 1); // scale
    u32le(&mut strh, 60); // rate → 60 fps
    u32le(&mut strh, 0); // start
    u32le(&mut strh, frames.len() as u32); // length
    u32le(&mut strh, frames.iter().map(|f| f.len()).max().unwrap_or(0) as u32);
    u32le(&mut strh, 0); // quality
    u32le(&mut strh, 0); // sample size
    u16le(&mut strh, 0);
    u16le(&mut strh, 0);
    u16le(&mut strh, w as u16);
    u16le(&mut strh, h as u16);
    assert_eq!(strh.len(), 56);

    // ---- avih：MainAVIHeader（56 字节）----
    let mut avih = Vec::new();
    u32le(&mut avih, 1_000_000 / 60); // 微秒/帧
    u32le(&mut avih, 0);
    u32le(&mut avih, 0);
    u32le(&mut avih, 0x10); // AVIF_HASINDEX
    u32le(&mut avih, frames.len() as u32); // 总帧数
    u32le(&mut avih, 0);
    u32le(&mut avih, 1); // 流数
    u32le(&mut avih, frames.iter().map(|f| f.len()).max().unwrap_or(0) as u32);
    u32le(&mut avih, w);
    u32le(&mut avih, h);
    for _ in 0..4 {
        u32le(&mut avih, 0);
    }
    assert_eq!(avih.len(), 56);

    // ---- 组装 ----
    let mut strl_body = Vec::new();
    chunk(&mut strl_body, b"strh", &strh);
    chunk(&mut strl_body, b"strf", &strf);

    let mut hdrl_body = Vec::new();
    chunk(&mut hdrl_body, b"avih", &avih);
    hdrl_body.extend_from_slice(&list(b"strl", &strl_body));

    let mut movi_body = Vec::new();
    for f in frames {
        chunk(&mut movi_body, b"00dc", f);
    }

    let mut idx1 = Vec::new();
    let mut off = 4usize; // 'movi' fourcc 之后
    for f in frames {
        fourcc(&mut idx1, b"00dc");
        u32le(&mut idx1, 4); // AVIIF_KEYFRAME
        u32le(&mut idx1, off as u32);
        u32le(&mut idx1, f.len() as u32);
        off += 8 + f.len() + (f.len() % 2);
    }

    let mut body = Vec::new();
    body.extend_from_slice(&list(b"hdrl", &hdrl_body));
    body.extend_from_slice(&list(b"movi", &movi_body));
    chunk(&mut body, b"idx1", &idx1);

    let mut out = Vec::new();
    fourcc(&mut out, b"RIFF");
    u32le(&mut out, (body.len() + 4) as u32);
    fourcc(&mut out, b"AVI ");
    out.extend_from_slice(&body);
    out
}
/// 样本路径；没设 / 文件不存在 → `None`（调用方打 SKIP）。
pub fn sample() -> Option<String> {
    let p = std::env::var("V2V_ASF_SAMPLE").ok()?;
    Path::new(&p).is_file().then_some(p)
}

/// **多个**样本：`V2V_ASF_SAMPLE` 里用 `:` 分隔多个路径。
///
/// 存在的理由：一个样本只能覆盖它自己走过的路径 —— 实测过好几次
/// 「换一个真样本就崩」（压缩载荷、1000fps 的 AVI、混装载荷…）。
/// 让同一条对账测试跑多个样本，等于把「只对一个样本调通」这个坑堵上。
/// 空项与不存在的路径会被跳过（缺样本照样 SKIP）。
pub fn sample_list() -> Vec<String> {
    let Ok(v) = std::env::var("V2V_ASF_SAMPLE") else {
        return Vec::new();
    };
    v.split(':')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && Path::new(s).is_file())
        .map(|s| s.to_string())
        .collect()
}


/// yuv420p 一帧的字节数。
pub fn yuv420p_frame_len(w: u32, h: u32) -> usize {
    let (wi, hi) = (w as usize, h as usize);
    wi * hi + 2 * (wi / 2) * (hi / 2)
}


/// 视频流的 `(宽, 高, fourcc)`（拿不到就 `None`）。
pub fn video_fourcc(path: &str) -> Option<(u32, u32, u32)> {
    let d = AsfDemuxer::new(BufReader::new(File::open(path).ok()?)).ok()?;
    let v = d.streams().iter().find(|s| s.kind == StreamKind::Video)?;
    let vf = v.video.as_ref()?;
    Some((vf.width, vf.height, vf.fourcc))
}

/// 按**解码顺序**取出全部视频帧（不过滤关键帧），连同视频尺寸。
pub fn all_video_frames(path: &str) -> Option<(Vec<Vec<u8>>, u32, u32)> {
    first_video_frames(path, usize::MAX)
}

/// 同上，但只取前 `n` 帧（只关心开头时省内存/时间）。
pub fn first_video_frames(path: &str, n: usize) -> Option<(Vec<Vec<u8>>, u32, u32)> {
    let mut d = AsfDemuxer::new(BufReader::new(File::open(path).ok()?)).ok()?;
    let (vnum, w, h) = {
        let v = d.streams().iter().find(|s| s.kind == StreamKind::Video)?;
        let vf = v.video.as_ref()?;
        (v.number, vf.width, vf.height)
    };
    let mut out = Vec::new();
    while out.len() < n {
        match d.next_sample().ok()? {
            Some(s) if s.stream == vnum => out.push(s.data),
            Some(_) => {}
            None => break,
        }
    }
    (!out.is_empty()).then_some((out, w, h))
}


/// 用 ffmpeg 解出前 `count` 帧的 yuv420p 原始像素。
///
/// 固定 `-idct 2`（C 版 `simple_idct`），且**必须放在 `-i` 之前** —— 放后面 ffmpeg
/// 不报错、不警告、直接静默忽略，会让人误以为「已钉在 C 版」。
/// `-fps_mode passthrough` 同理必加，否则帧号与样本对不上。
pub fn ffmpeg_frames(path: &str, count: usize, w: u32, h: u32) -> Option<Vec<Vec<u8>>> {
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-idct",
            "2",
            "-i",
            path,
            "-fps_mode",
            "passthrough",
            "-frames:v",
            &count.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let fsz = yuv420p_frame_len(w, h);
    if fsz == 0 || out.stdout.len() < fsz {
        return None;
    }
    Some(out.stdout.chunks_exact(fsz).map(|c| c.to_vec()).collect())
}


/// 与参考帧逐像素比，返回 `(最大差, 最大差下标, 平均差, 参考 Y 取值种数)`。
pub fn compare(y: &[u8], u: &[u8], v: &[u8], ref_yuv: &[u8], w: u32, h: u32) -> (i32, usize, f64, usize) {
    let (wi, hi) = (w as usize, h as usize);
    let (cw, ch) = (wi / 2, hi / 2);
    let (ysz, csz) = (wi * hi, cw * ch);
    assert_eq!(ref_yuv.len(), ysz + 2 * csz, "参考帧大小不对");
    let (ry, ru, rv) = (
        &ref_yuv[..ysz],
        &ref_yuv[ysz..ysz + csz],
        &ref_yuv[ysz + csz..],
    );

    let mut worst = 0i32;
    let mut worst_at = 0usize;
    let mut sum = 0i64;
    let mut idx = 0usize;
    for (name, a, b) in [("Y", y, ry), ("U", u, ru), ("V", v, rv)] {
        assert_eq!(a.len(), b.len(), "{name} 平面大小不一致");
        for (&x, &r) in a.iter().zip(b.iter()) {
            let d = (x as i32 - r as i32).abs();
            sum += d as i64;
            if d > worst {
                worst = d;
                worst_at = idx;
            }
            idx += 1;
        }
    }
    let mut levels = [false; 256];
    for &r in ry {
        levels[r as usize] = true;
    }
    (
        worst,
        worst_at,
        sum as f64 / idx as f64,
        levels.iter().filter(|&&b| b).count(),
    )
}
