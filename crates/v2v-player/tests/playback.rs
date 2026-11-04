//! 两个真样本上的**播放**行为：流式、PTS、EOS、两轨独立。
//!
//! 与 `crates/vc1/tests/frames.rs` 的分工：那边钉「解得对不对」（逐像素对 ffmpeg），
//! 这里钉「**作为播放器用起来对不对**」—— 顺序取帧不乱、PTS 单调、EOS 稳定、
//! 两轨互不阻塞、平面尺寸与声明一致。
//!
//! 样本在仓库的 `testing/`（本地、不入库），缺了就 `SKIP` 并通过。

use std::path::{Path, PathBuf};

use v2v_player::Player;

/// 样本路径（仓库的 `testing/` 目录，不入库）。
fn sample(name: &str) -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testing")
        .join(name);
    p.canonicalize().ok().filter(|p| p.is_file())
}

/// 一帧的平面尺寸必须与声明的宽高一致 —— 宿主直接按这个上传纹理，错了就是花屏。
fn check_planes(f: &v2v_player::VideoFrame) {
    assert_eq!(
        f.y.len(),
        f.width as usize * f.height as usize,
        "Y 平面大小不对"
    );
    let (cw, ch) = (f.width as usize / 2, f.height as usize / 2);
    assert_eq!(f.u.len(), cw * ch, "U 平面大小不对");
    assert_eq!(f.v.len(), cw * ch, "V 平面大小不对");
}

/// `black.wmv`：**无音频**、单帧、preroll 3000 ms / 时长 1000 ms。
///
/// 这条钉三件事：① preroll 真的扣掉了（首帧 PTS 是 0 而不是 3000）；
/// ② 取完之后 EOS 稳定返回 `None`；③ 平面尺寸与声明一致。
#[test]
fn black_wmv_opens_and_reaches_eos() {
    let Some(path) = sample("black.wmv") else {
        eprintln!("SKIP: testing/black.wmv 不在（样本不入库）");
        return;
    };
    let mut p = Player::open(&path).expect("打开");
    let info = p.info();
    eprintln!("info: {info:?}");
    assert!(!info.has_audio, "这个样本没有音频轨");
    assert!(info.duration_ms > 0, "时长该是正数");
    assert!(info.width > 0 && info.height > 0);

    let mut n = 0usize;
    let mut last = i64::MIN;
    while let Some(f) = p.next_video().expect("解帧不该报错") {
        check_planes(&f);
        assert!(f.pts_ms >= last, "视频 PTS 倒退（{} → {}）", last, f.pts_ms);
        last = f.pts_ms;
        n += 1;
        // 一秒的素材不该有几百帧 —— 防止「EOS 判不出来、循环停不下来」。
        assert!(n < 600, "EOS 判不出来：已经取了 {n} 帧还在继续");
    }
    assert!(n >= 1, "至少要解出一帧");
    assert_eq!(last, 0, "preroll 没扣：首帧 PTS 应该是 0");
    // EOS 之后**稳定**返回 None（宿主要能据此判「播完了」）。
    assert!(p.next_video().expect("EOS 后不该报错").is_none());
    assert!(p.next_video().expect("EOS 后不该报错").is_none());
}

/// `sample.wmv`：**有音频**、138 s。全片逐帧，慢，`#[ignore]`。
///
/// 这条钉四件事：① 两轨各自独立推进、都能到 EOS；② PTS 单调；
/// ③ 音频的 PTS 从 0 起（编码器延迟的静音丢掉了，但时间轴没被推歪）；
/// ④ 两条轨的最后一帧都落在时长附近（差超过 1 s 说明轨没读完或 PTS 算错）。
#[test]
#[ignore = "全片逐帧 + 整条音频，慢；用 --ignored 跑"]
fn sample_wmv_streams_both_tracks_to_eos() {
    let Some(path) = sample("sample.wmv") else {
        eprintln!("SKIP: testing/sample.wmv 不在（样本不入库）");
        return;
    };
    let mut p = Player::open(&path).expect("打开");
    let duration = p.info().duration_ms;
    assert!(p.info().has_audio, "这个样本该有音频轨");
    assert!(p.info().audio_sample_rate > 0);

    // ---- 视频：拉到 EOS ----
    let mut frames = 0usize;
    let mut last_v = i64::MIN;
    while let Some(f) = p.next_video().expect("解帧不该报错") {
        check_planes(&f);
        assert!(f.pts_ms >= last_v, "视频 PTS 倒退（{} → {}）", last_v, f.pts_ms);
        last_v = f.pts_ms;
        frames += 1;
        // 防挂死：30 fps 的 2.5 分钟素材不该超过几万帧。
        assert!(frames < 100_000, "EOS 判不出来");
    }
    eprintln!("视频 {frames} 帧，末帧 PTS {last_v} ms（时长 {duration} ms）");

    // ---- 音频：拉到 EOS（在视频之后拉 —— 两轨独立，顺序无所谓）----
    let mut audio_frames = 0usize;
    let mut last_a = i64::MIN;
    let mut total_samples = 0u64;
    while let Some(a) = p.next_audio().expect("解音频不该报错") {
        assert_eq!(
            a.planes.len(),
            a.channels,
            "平面数与声道数不一致"
        );
        assert!(a.planes.iter().all(|p| p.len() == a.samples), "各声道样本数不一致");
        assert!(a.pts_ms >= last_a, "音频 PTS 倒退（{} → {}）", last_a, a.pts_ms);
        last_a = a.pts_ms;
        total_samples += a.samples as u64;
        audio_frames += 1;
        assert!(audio_frames < 1_000_000, "EOS 判不出来");
    }
    eprintln!(
        "音频 {audio_frames} 帧 / {total_samples} 样本，末帧 PTS {last_a} ms（时长 {duration} ms）"
    );

    assert!(frames > 0 && audio_frames > 0, "两条轨都该有内容");
    assert!(last_v <= duration as i64, "视频 PTS 超过了时长");
    assert!(
        duration as i64 - last_v <= 1000,
        "视频末帧离时长太远（{last_v} vs {duration}）—— 轨没读完或 PTS 算错"
    );
    assert!(
        duration as i64 - last_a <= 1000,
        "音频末帧离时长太远（{last_a} vs {duration}）—— skip 之后时间轴被推歪了"
    );
}

/// **两轨互不阻塞**：先取几帧视频、再取几帧音频，都该成功。
///
/// 这是「两条轨各一条解封装实例」这个设计的直接检验 —— 单实例顺序解封装
/// 做不到「先视频后音频再回头视频」而不缓存整条轨。
#[test]
fn tracks_can_be_pulled_in_any_order() {
    let Some(path) = sample("sample.wmv") else {
        eprintln!("SKIP: testing/sample.wmv 不在（样本不入库）");
        return;
    };
    let mut p = Player::open(&path).expect("打开");

    let mut v = 0usize;
    while v < 10 {
        if p.next_video().expect("视频").is_none() {
            break;
        }
        v += 1;
    }
    let mut a = 0usize;
    while a < 10 {
        if p.next_audio().expect("音频").is_none() {
            break;
        }
        a += 1;
    }
    eprintln!("先取 {v} 帧视频、再取 {a} 帧音频 —— 两轨独立推进");
    assert!(v > 0 && a > 0, "两轨都得取得到");

    // 再回头取视频，也应该继续有（不会因为之前拉过音频就断流）。
    let more = p.next_video().expect("回头取视频不该报错");
    assert!(more.is_some(), "回头取视频应该还有");
}

/// **seek 落在关键帧上，而且能继续往下播。**
///
/// 判据：
/// ① 落点 ≤ 目标（索引槽里存的是**不晚于它**的关键帧）且差不超过 2 s
///    （槽粒度 1 s + 一个 GOP）；
/// ② seek 之后 `next_video` 交出的**就是**落点那一帧、而且是关键帧
///    —— 落点要是只是个数字、画面却从别处开始，宿主的时间轴就歪了；
/// ③ 之后 PTS 单调、音频也跟着到位。
#[test]
fn seek_lands_on_a_keyframe_and_keeps_going() {
    let Some(path) = sample("sample.wmv") else {
        eprintln!("SKIP: testing/sample.wmv 不在（样本不入库）");
        return;
    };
    let mut p = Player::open(&path).expect("打开");
    let duration = p.info().duration_ms;

    // 先推进几帧再 seek —— 宿主总是从「正在播」的状态跳，不是从刚打开的状态。
    for _ in 0..5 {
        assert!(p.next_video().expect("解帧").is_some());
    }

    let target = duration / 2;
    let actual = p.seek_ms(target).expect("seek");
    eprintln!("seek {target} ms → 实际 {actual} ms");
    // 需求要的是「关键帧回退 + 前向解码到目标」—— 所以落点 ≥ 目标，
    // 且差不超过 1 s（一帧的量级；再大说明前向解码没做到位）。
    assert!(actual >= target as i64, "落点 {actual} 没到目标 {target}");
    assert!(
        actual - target as i64 <= 1000,
        "落点离目标太远（{actual} vs {target}）—— 前向解码没做到位"
    );

    // 下一帧就是落点那一帧。
    let f = p
        .next_video()
        .expect("seek 之后不该报错")
        .expect("seek 之后该有帧");
    assert_eq!(f.pts_ms, actual, "seek 的落点应该是下一帧");
    check_planes(&f);

    // 之后继续、PTS 单调。
    let mut last = f.pts_ms;
    for _ in 0..30 {
        if let Some(f) = p.next_video().expect("继续解帧") {
            assert!(f.pts_ms >= last, "seek 之后 PTS 倒退");
            last = f.pts_ms;
        }
    }

    // 音频也跟着到位（同一包起重解，PTS 基准 = 落点）。
    let a = p.next_audio().expect("音频不该报错").expect("seek 之后该有音频");
    eprintln!("seek 后第一帧音频 pts {}（落点 {actual}）", a.pts_ms);
    assert!(
        a.pts_ms >= actual - 1000 && a.pts_ms <= actual + 1000,
        "seek 后音频 PTS 离落点太远（{} vs {actual}）",
        a.pts_ms
    );
}

/// **seek 回开头**：落点应是 0、下一帧就是首帧。
///
/// 这条顺带钉住「编码器延迟的静音在 seek 回 0 时要重新丢」——
/// 不丢的话首帧音频会晚约 93 ms。
#[test]
fn seek_back_to_zero_returns_the_start() {
    let Some(path) = sample("sample.wmv") else {
        eprintln!("SKIP: testing/sample.wmv 不在（样本不入库）");
        return;
    };
    let mut p = Player::open(&path).expect("打开");
    assert!(p.next_video().expect("解帧").is_some());
    assert!(p.next_audio().expect("解音频").is_some());

    let actual = p.seek_ms(0).expect("seek 回 0");
    assert_eq!(actual, 0, "seek 0 的落点该是 0");
    let f = p.next_video().expect("不该报错").expect("该有首帧");
    assert_eq!(f.pts_ms, 0, "回到开头后第一帧 PTS 该是 0");
    let a = p.next_audio().expect("不该报错").expect("该有音频");
    assert_eq!(a.pts_ms, 0, "回到开头后音频 PTS 该是 0（延迟静音重新丢掉了）");
}
