//! 真样本上验证解码链路：整帧与 ffmpeg 逐像素比。
//!
//! I 帧的测试（`iframe.rs`）证明不了帧间通路 —— 那里没有运动补偿、没有帧间反量化、
//! 没有 `run_diff`。这里的 `p_frames_match_ffmpeg` 才是帧间的判据。
//!
//! `every_frame_of_the_whole_sample_matches_ffmpeg` 把**整段样本**逐帧对完，
//! 它默认 `#[ignore]`（debug 下太慢），要跑就：
//!
//! ```text
//! V2V_ASF_SAMPLE=<样本> cargo test --release --test pframe -- --ignored --nocapture
//! ```
//!
//! 设 `V2V_ASF_SAMPLE=<路径>` 才真跑；缺样本或缺 ffmpeg 打 `SKIP` 并通过。

use std::io::{BufReader, Read};
use std::process::{Child, ChildStdout, Command, Stdio};

mod common;

use common::{all_video_frames, compare, ffmpeg_frames, first_video_frames, sample, sample_list, unique_temp, yuv420p_frame_len, wrap_single_frame_avi};
use v2v_msmpeg4::{decode_i_frame, parse_frame_header, Decoder, PictureType, Version};

/// 对**一个**样本做逐帧对账（`V2V_ASF_SAMPLE` 里 `:` 分隔的每个样本都跑一遍）。
#[test]
fn p_frames_match_ffmpeg() {
    let list = sample_list();
    if list.is_empty() {
        eprintln!("SKIP: V2V_ASF_SAMPLE 未设置或样本不存在");
        return;
    }
    for path in &list {
        eprintln!("==== 样本：{path} ====");
        check_p_frames(path);
    }
}

fn check_p_frames(path: &str) {
    // 按容器的 FourCC 选**版本** —— 帧头与系数解码都按版本分叉，
    // 拿 V3 去解 WMV1 只会得到「对不上」这种假失败。
    // 认不出的 codec 直接跳过，别把「没实现」伪装成「对不上」。
    let version = match common::video_fourcc(path).and_then(|(_, _, fc)| Version::from_fourcc(fc)) {
        Some(v) => v,
        None => {
            let name = common::video_fourcc(path)
                .map(|(_, _, fc)| v2v_asf::fourcc_string(fc))
                .unwrap_or_else(|| "?".into());
            eprintln!("SKIP: 视频是 {name}，不是 MS-MPEG-4 家族（本对账只覆盖 MP41/42/43 + WMV1/2）");
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
    let want = samples.len().min(MAX_FRAMES);
    let refs = match ffmpeg_frames(path, want, w, h) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 没有可用的 ffmpeg");
            return;
        }
    };
    assert_eq!(
        refs.len(),
        samples.len().min(MAX_FRAMES),
        "帧数不一致：ffmpeg {}，我们 {} —— 帧序/帧数对不上，先查解封装",
        refs.len(),
        want
    );

    // 逐帧解码。**不提前 assert** —— 一次跑完把所有帧的差异都打出来，好定位。
    let mut dec = Decoder::with_version(version);
    let mut results = Vec::new();
    let mut decoded = Vec::new();
    for i in 0..want {
        let f = dec.decode(&samples[i], w, h).expect("解码");
        let (worst, at, mean, levels) = compare(&f.y, &f.u, &f.v, &refs[i], w, h);
        eprintln!("帧 {i:2}: 最大差 {worst:3}（第 {at} 个像素）平均差 {mean:.4} 参考 Y 取值 {levels} 种");
        results.push(worst);
        decoded.push(f);
    }

    // 有错的帧：打一张「每个宏块的最差差」地图，好判断是整片错还是零星错。
    if std::env::var("V2V_PFRAME_MAP").is_ok() {
        for i in 0..want {
            if results[i] == 0 {
                continue;
            }
            eprintln!("---- 帧 {i} 的宏块误差地图 ----");
            let ours = &decoded[i];
            let (wi, hi) = (w as usize, h as usize);
            let (mw, mh) = (wi.div_ceil(16), hi.div_ceil(16));
            let mut bad = Vec::new();
            let mut per_row = vec![0usize; mh];
            for my in 0..mh {
                for mx in 0..mw {
                    let mut m = 0i32;
                    let mut c = 0usize;
                    for y in my * 16..((my + 1) * 16).min(hi) {
                        for x in mx * 16..((mx + 1) * 16).min(wi) {
                            let idx = y * wi + x;
                            let dv = (ours.y[idx] as i32 - refs[i][idx] as i32).abs();
                            if dv != 0 {
                                c += 1;
                            }
                            m = m.max(dv);
                        }
                    }
                    if m != 0 {
                        per_row[my] += 1;
                        bad.push((mx, my, m, c));
                    }
                }
            }
            eprintln!("共 {} 个宏块有差异（总 {}）", bad.len(), mw * mh);
            let rows: Vec<String> = per_row
                .iter()
                .enumerate()
                .filter(|(_, &c)| c > 0)
                .map(|(y, c)| format!("行{y}:{c}"))
                .collect();
            eprintln!("按宏块行汇总：{}", rows.join(" "));
            for b in bad.iter().take(12) {
                eprintln!("  MB({},{}) 最大差 {} 差异像素 {}", b.0, b.1, b.2, b.3);
            }
        }
    }

    let bad: Vec<usize> = results
        .iter()
        .enumerate()
        .filter(|(_, &w)| w != 0)
        .map(|(i, _)| i)
        .collect();
    assert!(
        bad.is_empty(),
        "这些帧与 ffmpeg 不一致：{bad:?}（逐帧最大差 {results:?}）"
    );
}


/// 最多比几帧（够覆盖 I/P 交替、`no_rounding` 翻转、以及带运动的内容帧）。
const MAX_FRAMES: usize = 24;

/// 流式读 ffmpeg 解出的帧。
///
/// 整段样本的原始像素是几十 GB 量级，不能一次性吃进内存 ——
/// 所以让 ffmpeg 往管道里吐，我们一帧一帧地读、比、丢。
struct FfmpegStream {
    child: Child,
    out: BufReader<ChildStdout>,
    fsz: usize,
}

impl FfmpegStream {
    fn start(path: &str, w: u32, h: u32) -> Option<Self> {
        let mut child = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                // 解码器选项，必须在 -i 之前（放后面会被静默忽略）
                "-idct",
                "2",
                "-i",
                path,
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let out = BufReader::new(child.stdout.take()?);
        Some(Self {
            child,
            out,
            fsz: yuv420p_frame_len(w, h),
        })
    }

    /// 读下一帧到 `buf`；`false` 表示流已结束。
    fn next_into(&mut self, buf: &mut Vec<u8>) -> bool {
        buf.clear();
        buf.resize(self.fsz, 0);
        self.out.read_exact(buf).is_ok()
    }
}

impl Drop for FfmpegStream {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// **整段样本逐帧对账** —— 这才是「解码链路全通」的判据。
///
/// 默认 `#[ignore]`（debug 下整段太慢）。要跑：
/// `V2V_ASF_SAMPLE=<样本> cargo test --release --test pframe -- --ignored --nocapture`
///
/// 用 `V2V_MAX_FRAMES=n` 可以只跑前 n 帧（迭代时省时间）。
#[test]
#[ignore = "整段样本耗时长；用 cargo test --release -- --ignored 跑"]
fn every_frame_of_the_whole_sample_matches_ffmpeg() {
    let path = match sample() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: V2V_ASF_SAMPLE 未设置或样本不存在");
            return;
        }
    };
    let (samples, w, h) = match all_video_frames(&path) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 没找到视频帧");
            return;
        }
    };
    let limit = std::env::var("V2V_MAX_FRAMES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    let want = samples.len().min(limit);

    let mut ff = match FfmpegStream::start(&path, w, h) {
        Some(s) => s,
        None => {
            eprintln!("SKIP: 没有可用的 ffmpeg");
            return;
        }
    };

    let mut dec = Decoder::new();
    let mut refbuf: Vec<u8> = Vec::new();
    let mut failures: Vec<(usize, i32, f64)> = Vec::new();
    let mut worst_overall = 0i32;
    let t0 = std::time::Instant::now();

    for i in 0..want {
        let f = match dec.decode(&samples[i], w, h) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("帧 {i}: 解码报错 {e}");
                failures.push((i, -1, 0.0));
                continue;
            }
        };
        if !ff.next_into(&mut refbuf) {
            eprintln!("帧 {i}: ffmpeg 提前结束（只有 {i} 帧）—— 帧序对不上，先查解封装");
            break;
        }
        let (worst, at, mean, levels) = compare(&f.y, &f.u, &f.v, &refbuf, w, h);
        worst_overall = worst_overall.max(worst);
        if worst != 0 {
            eprintln!("帧 {i:5}: 最大差 {worst:3}（第 {at} 个像素）平均差 {mean:.4} 参考 Y 取值 {levels} 种");
            failures.push((i, worst, mean));
        }
        // 每 500 帧报一次进度
        if (i + 1) % 500 == 0 {
            eprintln!(
                "… 已对 {} / {want} 帧，{:.1}s，累计失败 {} 帧",
                i + 1,
                t0.elapsed().as_secs_f64(),
                failures.len()
            );
        }
    }

    // ffmpeg 那边不该还有帧剩下（帧数不一致说明解封装漏帧）。
    // 只在「整段都跑」时才是硬判据；用 V2V_MAX_FRAMES 截断时剩下的本来就该有。
    let mut extra = 0usize;
    while ff.next_into(&mut refbuf) {
        extra += 1;
    }
    let full_run = want == samples.len();

    eprintln!(
        "对账完成：{} 帧（样本共 {} 帧），{:.1}s，最大差 {}，不一致 {} 帧，ffmpeg {} {extra} 帧",
        want,
        samples.len(),
        t0.elapsed().as_secs_f64(),
        worst_overall,
        failures.len(),
        if full_run { "多出" } else { "剩余" },
    );
    if !failures.is_empty() {
        let head: Vec<String> = failures
            .iter()
            .take(20)
            .map(|(i, w, m)| format!("帧{i}(最大差{w}/平均{m:.3})"))
            .collect();
        panic!("{} 帧不一致，前几个：{}", failures.len(), head.join(" "));
    }
    if full_run {
        // 「我们和 ffmpeg 的帧数一致」这件事由 `extra == 0` 覆盖（上面已经数过 ffmpeg 那边
        // 还剩几帧）。**不要**在这里写死样本的帧数 —— 那是「从某个样本量出来的数字」，
        // 等于把夹具出处写进公开历史，而且换个样本就会红。
        assert_eq!(extra, 0, "ffmpeg 比我们多 {extra} 帧 —— 解封装漏帧了");
    }

    // ---- 通路覆盖 ----
    // 「测试全绿」不等于「每条路都测过」：一条从没进去过的分支，和一条写对的分支
    // 在测试里长得一模一样。所以把走过哪些分支打出来，并把关键的那几条钉成断言。
    let st = dec.stats();
    eprintln!(
        "通路覆盖：\n  \
         I 帧 {} / P 帧 {}\n  \
         帧内宏块 {}（其中出现在 P 帧里的 {}）\n  \
         帧间：有残差 {} / 只有 MC {} / skip {}\n  \
         ac_pred 块 {}\n  \
         MV 相位：零 {} / 水平半 {} / 垂直半 {} / 双向半 {}\n  \
         边缘复制(MC) {}\n  \
         帧间 RL 表（下标 0/1/2 → 表 3/4/5）用到 {:?} 帧\n  \
         帧内 RL 表（下标 0/1/2）用到 {:?} 帧\n  \
         用到第 1 张 MV 表的帧 {}",
        st.frames_i,
        st.frames_p,
        st.mb_intra,
        st.mb_intra_in_p,
        st.mb_inter_coded,
        st.mb_inter_empty,
        st.mb_skip,
        st.blocks_ac_pred,
        st.mv_zero,
        st.mv_half_x,
        st.mv_half_y,
        st.mv_half_xy,
        st.mc_edge_emu,
        st.frames_rl_inter,
        st.frames_rl_intra,
        st.frames_mv_table_1,
    );

    assert!(st.frames_i > 0 && st.frames_p > 0, "I/P 帧都要有");
    assert!(st.mb_intra > 0, "帧内宏块一条都没走到");
    assert!(st.mb_inter_coded > 0, "帧间带残差的宏块一条都没走到");
    assert!(st.mb_skip > 0, "skip 宏块一条都没走到");
    assert!(
        st.mv_zero > 0 && (st.mv_half_x + st.mv_half_y + st.mv_half_xy) > 0,
        "半像素 MC 没被走到 —— 样本里居然没有非整数 MV，覆盖不完整"
    );
    // 下面这几条**只报告不失败** —— 走不到是样本的性质，不是代码的错。
    for (name, hit) in [
        ("P 帧里的帧内宏块（DC 预测的「清 1024」只在它身上起作用）", st.mb_intra_in_p > 0),
        ("AC 预测（ac_pred）", st.blocks_ac_pred > 0),
        ("MC 的边缘复制", st.mc_edge_emu > 0),
        ("垂直/双向半像素相位", st.mv_half_y + st.mv_half_xy > 0),
        ("第 1 张 MV 表", st.frames_mv_table_1 > 0),
        ("帧间 RL 表 4（rl_table_index = 1）", st.frames_rl_inter[1] > 0),
        ("帧间 RL 表 5（rl_table_index = 2）", st.frames_rl_inter[2] > 0),
    ] {
        if !hit {
            eprintln!("⚠️ 这条通路本样本没走到：{name}");
        }
    }
}

/// 喂**截断 / 垃圾**数据不能 panic。
///
/// 解码器解不出来可以报错，但绝不能崩 —— 真样本难免有损坏帧，而 panic 会把整个
/// 转换流程带走。这里用真样本的前两帧当底本，做截断与随机字节两种破坏。
#[test]
fn never_panics_on_truncated_or_garbage_frames() {
    let path = match sample() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: V2V_ASF_SAMPLE 未设置或样本不存在");
            return;
        }
    };
    let (samples, w, h) = match all_video_frames(&path) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 没找到视频帧");
            return;
        }
    };
    if samples.len() < 3 {
        eprintln!("SKIP: 样本太短");
        return;
    }

    let mut seed = 0xC0FFEEu32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 16) as u8
    };

    // 每个破坏样本都从一个干净的解码器开始（各自先解一帧真 I 帧拿到参考帧），
    // 免得前面一帧的垃圾状态影响后面 —— 我们要测的是「不 panic」，不是错误传播。
    for k in 0..120usize {
        let mut dec = Decoder::new();
        dec.decode(&samples[0], w, h).expect("首帧应当解得出来");

        // ① 截断到各种长度
        let p = &samples[1];
        let n = (k * 7919) % (p.len() + 1);
        let _ = dec.decode(&p[..n], w, h);

        // ② 随机字节
        let mut g = vec![0u8; 1 + (k * 13) % 128];
        for b in g.iter_mut() {
            *b = rnd();
        }
        let _ = dec.decode(&g, w, h);

        // ③ 真帧头 + 垃圾尾巴
        let mut mix = p[..p.len().min(8)].to_vec();
        mix.extend((0..64).map(|_| rnd()));
        let _ = dec.decode(&mix, w, h);

        // ④ 尺寸被改小（宏块数变了，越界风险最高）
        let _ = dec.decode(p, 16, 16);
        let _ = dec.decode(p, w, 16);
    }
    eprintln!("截断/垃圾输入 120 组：全部没 panic");
}

/// Y 平面里出现了多少种不同的值（用来判断一帧是不是「纯色」）。
fn distinct_levels(y: &[u8]) -> usize {
    let mut seen = [false; 256];
    for &p in y {
        seen[p as usize] = true;
    }
    seen.iter().filter(|&&b| b).count()
}

/// **AC 预测**那条路，手头的样本走不到 —— FFmpeg 自己的 MSMPEG4 编码器永远写
/// `ac_pred = 0`（`msmpeg4enc.c` 里两处 `put_bits(&s->pb, 1, 0); /* no AC prediction yet */`）。
/// 所以这里把真流里的这一位**翻成 1**，再让 ffmpeg 解**同一条**补丁流来对账。
///
/// 翻位为什么是安全的：`ac_pred` 只影响「用哪张扫描表」和「要不要把邻块的首行/首列加进来」，
/// **不改变任何码字的比特数**，所以翻完之后两个解码器仍然同步，逐像素比较才有意义。
#[test]
fn ac_prediction_path_matches_ffmpeg_when_forced_on() {
    let path = match sample() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: V2V_ASF_SAMPLE 未设置或样本不存在");
            return;
        }
    };
    let (samples, w, h) = match first_video_frames(&path, 400) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 没找到视频帧");
            return;
        }
    };
    let (mw, mh) = (w.div_ceil(16), h.div_ceil(16));

    for (idx, f) in samples.iter().enumerate() {
        // 只挑 **I 帧**（能单独解）且**有内容**的 —— 纯色帧 CBP 全 0、邻块 AC 也全 0，
        // 翻了位看不出变化，测不到东西。
        let Ok((hdr, _)) = parse_frame_header(f, mw, mh, v2v_msmpeg4::Version::V3, &mut Default::default()) else {
            continue;
        };
        if hdr.picture_type != PictureType::I {
            continue;
        }
        let Ok(base) = decode_i_frame(f, w, h) else {
            continue;
        };
        if distinct_levels(&base.y) <= 64 {
            continue;
        }

        // 找一个可翻的 `ac_pred` 位：优先「CBP 非 0 且不在首行首列」的宏块
        //（首行首列的邻块是零边框，加了等于没加）。
        let mut probe = Decoder::new();
        probe.trace_ac_pred(true);
        if probe.decode(f, w, h).is_err() {
            continue;
        }
        let reads = probe.ac_pred_reads();
        let pick = reads
            .iter()
            .find(|r| r.coded && !r.value && r.mbx >= 1 && r.mby >= 1)
            .or_else(|| reads.iter().find(|r| !r.value && r.mbx >= 1 && r.mby >= 1))
            .copied();
        let Some(target) = pick else { continue };

        let byte = target.bit_pos / 8;
        let mask = 1u8 << (7 - (target.bit_pos % 8));

        // ---- 我们解补丁帧 ----
        let mut patched = f.clone();
        patched[byte] ^= mask;
        let mut dec = Decoder::new();
        dec.trace_ac_pred(true);
        let ours = match dec.decode(&patched, w, h) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("帧 {idx}: 补丁帧我们解不了（{e}），换一帧");
                continue;
            }
        };
        let after = dec
            .ac_pred_reads()
            .iter()
            .find(|r| r.mbx == target.mbx && r.mby == target.mby)
            .copied();
        assert!(
            after.is_some_and(|r| r.value),
            "翻位后 MB({},{}) 的 ac_pred 应当读到 1，实际 {after:?}",
            target.mbx,
            target.mby
        );
        assert_ne!(
            base.y, ours.y,
            "把 MB({},{}) 的 ac_pred 翻成 1 竟然没改变输出 —— 这条测试失去意义",
            target.mbx, target.mby
        );

        // ---- ffmpeg 解同一条补丁帧 ----
        let avi = wrap_single_frame_avi(&patched, w, h);
        let tmp = unique_temp("v2v_ac_pred_patched", "avi");
        std::fs::write(&tmp, &avi).expect("写临时 AVI");
        let refs = ffmpeg_frames(tmp.to_str().unwrap(), 1, w, h);
        let _ = std::fs::remove_file(&tmp);
        let Some(refs) = refs.filter(|v| !v.is_empty()) else {
            eprintln!("帧 {idx}: ffmpeg 解不了补丁帧，换一帧");
            continue;
        };

        let (worst, at, mean, _) = compare(&ours.y, &ours.u, &ours.v, &refs[0], w, h);
        eprintln!(
            "AC 预测（强制打开）：帧 {idx}，MB({},{})，帧内位偏移 {}，对账最大差 {worst}，平均差 {mean:.4}",
            target.mbx, target.mby, target.bit_pos
        );
        assert_eq!(
            worst, 0,
            "强制打开 AC 预测后与 ffmpeg 不一致：最大差 {worst}（第 {at} 个像素），平均差 {mean:.4}"
        );
        return;
    }
    eprintln!("SKIP: 前 400 帧里没找到可用来验证 AC 预测的 I 帧");
}
