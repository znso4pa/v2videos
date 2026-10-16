//! **WMV2（WMV8）解码**：真样本逐帧对 `ffmpeg` 比像素。
//!
//! 为什么单开一个文件、而不是并进 `pframe.rs`：WMV2 有两样别人没有的东西 ——
//! 帧头是**另一套语法**（1 位的 `pict_type`、整帧 skip 试探），而且解码器要
//! **容器的 extradata**（能力位：`mspel` / `abt` / `j_type` / `top_left_mv` …）才建得起来。
//!
//! 用法（缺样本 / 缺 ffmpeg 一律 `SKIP` 并通过）：
//!
//! ```text
//! V2V_ASF_SAMPLE='/path/a.wmv:/path/b.asf' cargo test --test wmv2 -- --nocapture
//! ```
//!
//! `V2V_MAX_FRAMES` 改「每个样本比几帧」（默认 24）。`V2V_MB_MAP=1` 时对有差异的帧
//! 打一张「逐宏块、分平面」的误差地图 —— 判断「整片错」还是「零星错」用。
//!
//! ⚠️ **本仓不做错误隐藏**：首帧不是能解析的 I 帧、或中途遇到 IntraX8（`j_type`，
//! 那是另一个内帧编解码器）时，我们**明确报错**，而 ffmpeg 会照样硬解。
//! 这两条都打出来并停下，不当成「对不上」。

mod common;

use common::{all_video_frames, compare, ffmpeg_frames, sample_list, video_extra};
use v2v_msmpeg4::{Decoder, Error, Version, Wmv2Flags, Wmv2Frame};

/// 每个样本比几帧。`V2V_MAX_FRAMES=0` = **整段**（别写成「比 0 帧」——
/// 那样 `want` 会是 0，ffmpeg 那条路起不来，于是打出一句**误导的**
/// 「SKIP: 没有可用的 ffmpeg」，看着像环境缺 ffmpeg）。
fn max_frames() -> usize {
    match std::env::var("V2V_MAX_FRAMES").ok().and_then(|v| v.parse().ok()) {
        Some(0) | None => usize::MAX,
        Some(n) => n,
    }
}

#[test]
fn wmv2_frames_match_ffmpeg() {
    let list = sample_list();
    if list.is_empty() {
        eprintln!("SKIP: V2V_ASF_SAMPLE 未设置或样本不存在");
        return;
    }
    for path in &list {
        eprintln!("==== 样本：{path} ====");
        check(path);
    }
}

fn check(path: &str) {
    let Some((_, _, fourcc)) = common::video_fourcc(path) else {
        eprintln!("SKIP: 读不到视频轨");
        return;
    };
    if Version::from_fourcc(fourcc) != Some(Version::Wmv2) {
        eprintln!(
            "SKIP: 视频是 {}，不是 WMV2（这条对账只覆盖 WMV2）",
            v2v_asf::fourcc_string(fourcc)
        );
        return;
    }
    // 能力位在容器的 extradata 里 —— 没有它连帧头都读不出来。
    let extra = video_extra(path).unwrap_or_default();
    let flags = match Wmv2Flags::from_extradata(&extra) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("SKIP: extradata 读不出能力位（{e}）");
            return;
        }
    };
    eprintln!("能力位：{flags:?}");
    let mut dec = match Decoder::with_wmv2(&extra) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: extradata 建不起 WMV2 解码器（{e}）");
            return;
        }
    };
    let (samples, w, h) = match all_video_frames(path) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 没找到视频帧");
            return;
        }
    };
    let want = samples.len().min(max_frames());
    if want == 0 {
        eprintln!("SKIP: 这一份没有视频帧");
        return;
    }
    let refs = match ffmpeg_frames(path, want, w, h) {
        Some(v) => v,
        None => {
            // 区分「没装 ffmpeg」与「ffmpeg 解得出来但我们解得不一样」——
            // 后者是**真问题**，别拿 SKIP 盖过去。
            if common::has_ffmpeg() {
                panic!("{path}: ffmpeg 起得来但没给出 {want} 帧参考（帧序/帧数对不上？）");
            }
            eprintln!("SKIP: 没有可用的 ffmpeg");
            return;
        }
    };
    assert_eq!(
        refs.len(),
        want,
        "帧数不一致：ffmpeg {}，我们 {} —— 帧序/帧数对不上，先查解封装",
        refs.len(),
        want
    );

    let (mb_w, mb_h) = (w.div_ceil(16), h.div_ceil(16));
    let mut no_rounding = false;
    let mut results = Vec::new();
    for i in 0..want {
        // 帧头单独解一遍打出来 —— 与 `ffmpeg -debug pict` 那几行是同一组字段，
        // 对不上时能一眼看出是「帧头读错」还是「MB 层读错」。
        match v2v_msmpeg4::parse_wmv2_frame(&samples[i], mb_w, mb_h, &flags, no_rounding) {
            Ok((Wmv2Frame::Frame(h), used)) => {
                no_rounding = h.no_rounding;
                eprintln!(
                    "帧 {i:3} 头: {:?} q={} rl={} rlc={} dc={} mv={} mbrl={} mspel={} per_mb_abt={} abt_type={} cbp_idx={} skip={} 用了 {used} 位",
                    h.picture_type,
                    h.qscale,
                    h.rl_table_index,
                    h.rl_chroma_table_index,
                    h.dc_table_index,
                    h.mv_table_index,
                    h.per_mb_rl_table as u8,
                    h.mspel as u8,
                    h.per_mb_abt as u8,
                    h.abt_type,
                    h.cbp_table_index,
                    h.mb_skip.iter().filter(|&&s| s).count(),
                );
            }
            Ok((Wmv2Frame::Skipped, _)) => eprintln!("帧 {i:3} 头: 整帧 skip"),
            Err(e) => eprintln!("帧 {i:3} 头: 解不出来（{e}）"),
        }
        let f = match dec.decode(&samples[i], w, h) {
            Ok(f) => f,
            // IntraX8 是另一个内帧编解码器 —— 明确说「没做」，别伪装成「对不上」。
            Err(Error::Unsupported(m)) => {
                eprintln!("第 {i} 帧碰到本实现不做的语法，就此停下：{m}");
                eprintln!("（已对完 {i} 帧；ffmpeg 会照样硬解，所以这不是「不一致」）");
                break;
            }
            Err(e) => panic!("第 {i} 帧解码失败：{e}"),
        };
        let (worst, at, mean, levels) = compare(&f.y, &f.u, &f.v, &refs[i], w, h);
        eprintln!(
            "帧 {i:3}: 最大差 {worst:3}（第 {at} 个像素）平均差 {mean:.4} 参考 Y 取值 {levels} 种"
        );
        if worst != 0 && std::env::var("V2V_MB_MAP").is_ok() {
            mb_map(&f, &refs[i], w, h);
        }
        results.push(worst);
    }

    if results.is_empty() {
        eprintln!("SKIP: 一帧都没比（首帧就走不通）");
        return;
    }
    // 通路覆盖：这几条只要非零，就说明对应的分支**真的被样本走到了** ——
    // 「全绿」和「每条路都测过」是两回事（AGENTS.md 里那条）。
    eprintln!("统计：{:?}", dec.stats());
    let bad: Vec<usize> = results
        .iter()
        .enumerate()
        .filter(|(_, &w)| w != 0)
        .map(|(i, _)| i)
        .collect();
    eprintln!(
        "小结：比了 {} 帧，最大差不为 0 的有 {} 帧",
        results.len(),
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "这些帧与 ffmpeg 不一致：{bad:?}（逐帧最大差 {results:?}）"
    );
}

/// 逐宏块、**分平面**打印最大差 —— 用来判断「整片错」还是「零星错」，
/// 以及错在亮度还是色度（这两者的根因完全不同）。
fn mb_map(f: &v2v_msmpeg4::Frame, r: &[u8], w: u32, h: u32) {
    let (wi, hi) = (w as usize, h as usize);
    let (cw, ch) = (wi / 2, hi / 2);
    let (ysz, csz) = (wi * hi, cw * ch);
    let planes: [(&str, &[u8], &[u8], usize, usize, usize); 3] = [
        ("Y", &f.y, &r[..ysz], wi, hi, 16),
        ("U", &f.u, &r[ysz..ysz + csz], cw, ch, 8),
        ("V", &f.v, &r[ysz + csz..], cw, ch, 8),
    ];
    for (name, a, b, pw, ph, bs) in planes {
        let (mx, my) = (pw.div_ceil(bs), ph.div_ceil(bs));
        let mut bad = Vec::new();
        for m in 0..my {
            for n in 0..mx {
                let mut worst = 0i32;
                let mut cnt = 0usize;
                for y in m * bs..((m + 1) * bs).min(ph) {
                    for x in n * bs..((n + 1) * bs).min(pw) {
                        let d = (a[y * pw + x] as i32 - b[y * pw + x] as i32).abs();
                        if d != 0 {
                            cnt += 1;
                        }
                        worst = worst.max(d);
                    }
                }
                if worst != 0 {
                    bad.push((n, m, worst, cnt));
                }
            }
        }
        eprintln!("  {name} 平面：{} 个宏块有差异（共 {}）", bad.len(), mx * my);
        for (n, m, worst, cnt) in bad.iter().take(10) {
            eprintln!("    MB({n},{m}) 最大差 {worst} 差异像素 {cnt}");
        }
    }
}
