//! 真样本上验证 I 帧解码：与 ffmpeg 解出的 I 帧**逐像素比**。
//!
//! 这是解码器「立住」的判据 —— 前面每个核（表 / VLC / 系数 / 预测 / IDCT）单独测都过了，
//! 但只有整帧对得上，才能说明它们串起来是对的。
//!
//! 设 `V2V_ASF_SAMPLE=<路径>` 才真跑；缺样本或缺 ffmpeg 打 `SKIP` 并通过。

mod common;

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

use v2v_asf::{AsfDemuxer, StreamKind};

/// 对账时最多取几个 I 帧：够覆盖首帧 + 若干内容帧，又不至于让 oracle 测试太慢。
const MAX_I_FRAMES: usize = 6;

/// 样本路径；没设 / 文件不存在 → `None`（调用方打 SKIP）。
fn sample() -> Option<String> {
    let p = std::env::var("V2V_ASF_SAMPLE").ok()?;
    Path::new(&p).is_file().then_some(p)
}

/// yuv420p 一帧的字节数。
fn yuv420p_frame_len(w: u32, h: u32) -> usize {
    let (wi, hi) = (w as usize, h as usize);
    wi * hi + 2 * (wi / 2) * (hi / 2)
}

/// 解出样本里**全部**关键帧（I 帧），连同视频尺寸。
fn all_i_frames(path: &str) -> Option<(Vec<Vec<u8>>, u32, u32)> {
    let mut d = AsfDemuxer::new(BufReader::new(File::open(path).ok()?)).ok()?;
    let (vnum, w, h) = {
        let v = d.streams().iter().find(|s| s.kind == StreamKind::Video)?;
        let vf = v.video.as_ref()?;
        (v.number, vf.width, vf.height)
    };
    let mut out = Vec::new();
    while let Some(s) = d.next_sample().ok()? {
        if s.stream == vnum && s.key_frame {
            out.push(s.data);
        }
    }
    (!out.is_empty()).then_some((out, w, h))
}

/// 用 ffmpeg 解出**前 `count` 个 I 帧**的 yuv420p 原始像素（一次调用，按 `w`/`h` 切帧）。
///
/// 固定 `-idct 2`（C 版 `simple_idct`）：aarch64 上 ffmpeg 默认走 NEON 版
/// （`FF_IDCT_PERM_PARTTRANS`），两者有 ±1 的舍入差。本仓的 IDCT 对齐的是 **C 版**，
/// 参考帧也必须钉在 C 版上，否则会拿别人的舍入误差当自己的 bug。
///
/// ⚠️ `-idct` 是**解码器选项，必须放在 `-i` 之前**。放后面 ffmpeg 不报错、不警告，
/// 直接静默忽略（输出与默认 NEON 逐字节相同）—— 会让人误以为「已钉在 C 版」而实际上
/// 一直在跟 NEON 比，凭空多出 ±1。
///
/// `-fps_mode passthrough` 必加：否则 ffmpeg 会按 CFR 复制帧，帧号就和 ASF 样本对不上了。
fn ffmpeg_i_frames(path: &str, count: usize, w: u32, h: u32) -> Option<Vec<Vec<u8>>> {
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            // 解码器选项 —— 必须在 -i 之前，否则被静默忽略（见上方文档）。
            "-idct",
            "2",
            "-i",
            path,
            "-vf",
            "select=eq(pict_type\\,I)",
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
    Some(
        out.stdout
            .chunks_exact(fsz)
            .map(|c| c.to_vec())
            .collect(),
    )
}

/// 一帧的对账结果。
struct Diff {
    /// 最大逐像素差。
    worst: i32,
    /// 最大差所在的下标（Y → U → V 连续编号）。
    worst_at: usize,
    /// 平均逐像素差。
    mean: f64,
    /// 参考帧 Y 平面的取值个数（1 = 纯色）。
    ref_levels: usize,
}

/// 把我们的三个平面和 ffmpeg 的 yuv420p 原始帧逐像素比。
fn compare(y: &[u8], u: &[u8], v: &[u8], ref_yuv: &[u8], w: u32, h: u32) -> Diff {
    let (wi, hi) = (w as usize, h as usize);
    let (cw, ch) = (wi / 2, hi / 2);
    let ysz = wi * hi;
    let csz = cw * ch;
    assert_eq!(
        ref_yuv.len(),
        ysz + 2 * csz,
        "ffmpeg 参考帧大小不对（{} vs {}）—— 尺寸可能没对齐",
        ref_yuv.len(),
        ysz + 2 * csz
    );

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
    let ref_levels = levels.iter().filter(|&&b| b).count();

    Diff {
        worst,
        worst_at,
        mean: sum as f64 / idx as f64,
        ref_levels,
    }
}

/// 冒烟：第一个 I 帧能解出来，平面尺寸对，且**不 panic、不越界**。
///
/// 注意这里**不用「有方差」当判据** —— 本样本的第一帧是**纯色**的（渐入灰），
/// 方差判据会误杀正确的解码。逐像素的对账在 `first_i_frame_matches_ffmpeg` /
/// `coded_i_frames_match_ffmpeg` 里做。
#[test]
fn decodes_the_first_i_frame_without_error() {
    let path = match sample() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: V2V_ASF_SAMPLE 未设置或样本不存在");
            return;
        }
    };
    let (i_frames, w, h) = match all_i_frames(&path) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 没找到 I 帧");
            return;
        }
    };
    let (version, extra) = match common::version_and_extra(&path) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 认不出样本视频轨的 FourCC（本对账只覆盖 MS-MPEG-4 家族 + WMV1/2）");
            return;
        }
    };
    let f = common::decode_one_frame(&i_frames[0], w, h, version, &extra).expect("解 I 帧");
    assert_eq!(f.y.len(), (w as usize) * (h as usize));
    assert_eq!(f.u.len(), (w as usize / 2) * (h as usize / 2));
    assert_eq!(f.v.len(), (w as usize / 2) * (h as usize / 2));
    let min = *f.y.iter().min().unwrap();
    let max = *f.y.iter().max().unwrap();
    eprintln!("I 帧冒烟通过：Y 范围 {min}..{max}（纯色帧属正常）");
}

/// 首帧（纯色）与 ffmpeg **逐像素相同**（最大差 0）。
///
/// ⚠️ 局限：首帧只有直流系数非零，这条只能证明**直流通路 + 位流对齐**是对的，
/// **证明不了 AC 通路**。AC 由 `coded_i_frames_match_ffmpeg` 覆盖 —— 这正是当年
/// 只测首帧时漏掉 AC 反量化 bug 的原因。
#[test]
fn first_i_frame_matches_ffmpeg() {
    let path = match sample() {
        Some(p) => p,
        None => {
            eprintln!("SKIP first_i_frame_matches_ffmpeg: V2V_ASF_SAMPLE 未设置或样本不存在");
            return;
        }
    };
    let (i_frames, w, h) = match all_i_frames(&path) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 没找到 I 帧");
            return;
        }
    };
    let (version, extra) = match common::version_and_extra(&path) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 认不出样本视频轨的 FourCC（本对账只覆盖 MS-MPEG-4 家族 + WMV1/2）");
            return;
        }
    };
    let refs = match ffmpeg_i_frames(&path, 1, w, h) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 没有可用的 ffmpeg");
            return;
        }
    };

    let ours = common::decode_one_frame(&i_frames[0], w, h, version, &extra).expect("解 I 帧");
    let d = compare(&ours.y, &ours.u, &ours.v, &refs[0], w, h);
    eprintln!(
        "首帧对账：最大差 {}（第 {} 个像素），平均差 {:.3}，参考 Y 取值 {} 种",
        d.worst, d.worst_at, d.mean, d.ref_levels
    );
    assert_eq!(
        d.worst, 0,
        "与 ffmpeg 不一致：最大差 {}（第 {} 个像素），平均差 {:.3}",
        d.worst, d.worst_at, d.mean
    );
}

/// **内容帧**（非纯色）必须与 ffmpeg 逐像素一致 —— 这条才是真正钉住 **AC 通路**的回归测试。
///
/// 为什么要单独一条：首帧是纯色（渐入灰），只有直流系数非零。曾经因为只测首帧，
/// 「帧内 AC 反量化」与「亮度 DC 缩放表选错」两个 bug 一直没被发现 ——
/// 前者让带 AC 的块整体偏小，后者让 DC 预测整体差一大截，但首帧都看不出来。
///
/// 容忍度：最大差 **0**（逐位一致）。修好后实测 I#0…I#5 与 `-idct 2`（C 版）全部
/// 「最大差 0、平均差 0.000」。
/// 作对照：漏掉 AC 反量化时最大差是 55、平均差 34；DC 缩放表取错时平均差 34.02 ——
/// 这条能稳稳抓住。
/// 若将来换 ffmpeg 版本导致 C 版 `simple_idct` 的舍入变了，把 0 放宽到 1（并在此注明）。
#[test]
fn coded_i_frames_match_ffmpeg() {
    let path = match sample() {
        Some(p) => p,
        None => {
            eprintln!("SKIP coded_i_frames_match_ffmpeg: V2V_ASF_SAMPLE 未设置或样本不存在");
            return;
        }
    };
    let (i_frames, w, h) = match all_i_frames(&path) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 没找到 I 帧");
            return;
        }
    };
    let (version, extra) = match common::version_and_extra(&path) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 认不出样本视频轨的 FourCC（本对账只覆盖 MS-MPEG-4 家族 + WMV1/2）");
            return;
        }
    };
    let want = i_frames.len().min(MAX_I_FRAMES);
    if want < 2 {
        eprintln!("SKIP: 样本里只有 {} 个 I 帧，覆盖不到内容帧", i_frames.len());
        return;
    }
    let refs = match ffmpeg_i_frames(&path, want, w, h) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 没有可用的 ffmpeg");
            return;
        }
    };
    if refs.len() < want {
        eprintln!("SKIP: ffmpeg 只解出 {} 个 I 帧（要 {want}）", refs.len());
        return;
    }

    let mut coded_seen = 0usize;
    for n in 1..want {
        let ours = common::decode_one_frame(&i_frames[n], w, h, version, &extra).expect("解 I 帧");
        let d = compare(&ours.y, &ours.u, &ours.v, &refs[n], w, h);
        eprintln!(
            "I#{n} 对账：最大差 {}（第 {} 个像素），平均差 {:.3}，参考 Y 取值 {} 种",
            d.worst, d.worst_at, d.mean, d.ref_levels
        );
        if d.ref_levels > 1 {
            coded_seen += 1;
        }
        assert_eq!(
            d.worst, 0,
            "I#{n} 与 ffmpeg 不一致：最大差 {}（第 {} 个像素），平均差 {:.3}",
            d.worst, d.worst_at, d.mean
        );
    }
    assert!(
        coded_seen > 0,
        "取到的 I 帧全是纯色，这条测试失去意义（样本或取帧逻辑不对）"
    );
}
