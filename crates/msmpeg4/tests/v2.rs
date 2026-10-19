//! **MP42（MS-MPEG-4 v2）** 的逐像素对账：拿 ffmpeg 当 oracle，整帧整帧比。
//!
//! 为什么单独一个文件：V1/V2 的语法与 V3（MP43）差在**三处表**上（DC 的 512 项 VLC、
//! DC 缩放恒 8、H.263 的运动矢量），而这三处都不报错、只出错 —— 只有整帧对得上才算数。
//!
//! 设 `V2V_MP42_SAMPLE=<某个 .avi>` 才真跑；缺样本或缺 ffmpeg 打 `SKIP` 并通过。
//!
//! 造样本（MP42 只有 ffmpeg 的 `msmpeg4v2` 编码器能造，**MP41 没有编码器**，
//! 所以 V1 这条分支至今只有代码、没有对账样本）：
//!
//! ```text
//! ffmpeg -f lavfi -i testsrc2=size=352x288:rate=25 -t 1.6 \
//!        -c:v msmpeg4v2 -q:v 6 -g 12 /tmp/mp42.avi
//! V2V_MP42_SAMPLE=/tmp/mp42.avi cargo test -p v2v-msmpeg4 --test v2
//! ```
//!
//! ⚠️ **qscale 要扫几个值**：V1/V2 的 DC 缩放是「恒 8」（`ff_mpeg1_dc_scale_table`），
//! 而 V3 是 `ff_mpeg4_y_dc_scale_table`。两张表在 **qscale <= 4 时一模一样**，
//! 从 q=5 起才分叉（10 vs 8）。只测低 qscale 的样本，拿错表也看不出来。

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

use v2v_avi::AviDemuxer;
use v2v_msmpeg4::{Decoder, Version};

/// 样本路径；没设 / 文件不存在 → `None`（调用方打 SKIP）。
fn sample() -> Option<String> {
    let p = std::env::var("V2V_MP42_SAMPLE").ok()?;
    Path::new(&p).is_file().then_some(p)
}

/// 用 ffmpeg 解出全部帧的 yuv420p 原始像素。
///
/// 固定 `-idct 2`（C 版 `simple_idct`）：aarch64 上 ffmpeg 默认走 NEON 版，
/// 两者有 **±1** 的舍入差 —— 本仓的 IDCT 对齐的是 C 版。不加这个参数就会
/// 把别人的舍入差当自己的 bug（实测：320×240 的样本每帧 2~6 个像素差 1）。
///
/// ⚠️ `-idct` 是**解码器选项，必须放在 `-i` 之前**；放后面会被静默忽略。
fn ffmpeg_frames(path: &str) -> Option<Vec<u8>> {
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-idct",
            "2",
            "-i",
            path,
            // 不按 CFR 复制帧，否则帧号与样本对不上。
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-",
        ])
        .output()
        .ok()?;
    if !out.status.success() || out.stdout.is_empty() {
        return None;
    }
    Some(out.stdout)
}

#[test]
fn mp42_matches_ffmpeg_frame_by_frame() {
    let Some(path) = sample() else {
        eprintln!("SKIP：没给 V2V_MP42_SAMPLE");
        return;
    };
    let Some(oracle) = ffmpeg_frames(&path) else {
        eprintln!("SKIP：ffmpeg 解不出来");
        return;
    };

    let mut d = AviDemuxer::new(BufReader::new(File::open(&path).unwrap())).unwrap();
    let vnum = d
        .streams()
        .iter()
        .find(|s| s.video.is_some())
        .map(|s| s.number)
        .expect("没有视频流");
    let vf = d
        .streams()
        .iter()
        .find(|s| s.number == vnum)
        .and_then(|s| s.video.clone())
        .unwrap();
    assert_eq!(
        vf.fourcc,
        u32::from_le_bytes(*b"MP42"),
        "这个测试只吃 MP42 样本"
    );
    let (w, h) = (vf.width as usize, vf.height as usize);
    let fsz = w * h + 2 * (w / 2) * (h / 2);

    let mut dec = Decoder::with_version(Version::V2);
    let mut i = 0usize;
    while let Some(s) = d.next_sample().unwrap() {
        if s.stream != vnum {
            continue;
        }
        let frame = dec
            .decode(&s.data, vf.width, vf.height)
            .unwrap_or_else(|e| panic!("帧 {i} 解失败：{e}"));
        assert!(
            (i + 1) * fsz <= oracle.len(),
            "ffmpeg 只给了 {} 帧，我们解到第 {i} 帧",
            oracle.len() / fsz
        );
        let r = &oracle[i * fsz..(i + 1) * fsz];
        let (ry, ru, rv) = (
            &r[..w * h],
            &r[w * h..w * h + (w / 2) * (h / 2)],
            &r[w * h + (w / 2) * (h / 2)..],
        );
        assert_eq!(frame.y.len(), w * h, "帧 {i} 的 Y 尺寸");
        assert_eq!(frame.u.len(), (w / 2) * (h / 2), "帧 {i} 的 U 尺寸");
        assert_eq!(frame.v.len(), (w / 2) * (h / 2), "帧 {i} 的 V 尺寸");
        for (name, ours, theirs) in [("Y", &frame.y, ry), ("U", &frame.u, ru), ("V", &frame.v, rv)]
        {
            if let Some(p) = ours.iter().zip(theirs).position(|(a, b)| a != b) {
                let n = ours.iter().zip(theirs).filter(|(a, b)| a != b).count();
                panic!(
                    "帧 {i} 的 {name} 平面不一致：首个差异在 {p}（我们 {} / ffmpeg {}），共 {n} 个像素",
                    ours[p], theirs[p]
                );
            }
        }
        i += 1;
    }
    assert!(i > 0, "一帧都没解出来");
    assert_eq!(
        i,
        oracle.len() / fsz,
        "帧数不一致（我们 {i}，ffmpeg {}）",
        oracle.len() / fsz
    );
    eprintln!("{i} 帧逐像素一致（Y/U/V 全比）");
}
