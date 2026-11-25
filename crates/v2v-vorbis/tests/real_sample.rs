//! 真样本对账 **ffmpeg**。
//!
//! ## 为什么这条测试必须存在
//!
//! 单元测试用的是**自己造的**码流（规范例子、合成窗、朴素 MDCT）。它们能证明
//! 「公式抄对了」，但证明不了「整条链能解别人的产物」—— 这两件事差得很远：
//! 码本的码字分配、floor 的 `step2`、residue 的 `partition` 边界、耦合的符号，
//! 任何一处理解偏了，自造码流照样自洽，真文件却一塌糊涂。
//!
//! 所以判据是**别人编、我们解**：拿 ffmpeg 解同一个 `.ogg` 的 PCM 当真值。
//!
//! ## 判据
//!
//! 1. **长度对得上**（差不超过一页的量级）；
//! 2. **对齐后 SNR > 60 dB**（实测在 90 dB 上下）；
//! 3. **比对区间不是静音** —— 静音乘任何系数都还是 0，「逐样本一致」在静音里
//!    毫无信息量。这条单独断言，因为踩过两次。
//!
//! Vorbis 不是整数精确变换（IMDCT 是浮点），所以**不能**要求逐字节一致；
//! 但 60 dB 这个门槛足够挡住「窗/耦合/定标错了」那类问题（那种错是 0~20 dB）。
//!
//! ## 怎么跑
//!
//! ```bash
//! V2V_VORBIS_SAMPLE=/path/to/file.ogg \
//!   cargo test -p v2v-vorbis --test real_sample -- --nocapture
//! ```
//!
//! 没设环境变量就跳过（真样本不进仓库）。需要 `ffmpeg` 在 `PATH` 上。

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use v2v_ogg::PacketReader;
use v2v_vorbis::Decoder;

fn sample_path() -> Option<PathBuf> {
    std::env::var_os("V2V_VORBIS_SAMPLE").map(PathBuf::from)
}

fn has_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// 进程内唯一的临时路径。`cargo test` 默认多线程，共用一个路径会读到半截数据。
fn unique_temp(stem: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{}_{n}", std::process::id()))
}

/// 我们自己的解码：拆包 → 解 → 逐声道拼起来。
struct Ours {
    planes: Vec<Vec<f32>>,
    packets: usize,
    bad_packets: usize,
    sample_rate: u32,
    blocksize_0: usize,
    blocksize_1: usize,
    /// 每个包输出长度的直方图 —— 用来证明「短块 / hybrid 窗那条路真的被走过」。
    /// `Np/4 + N/4` 只有三种取值：短短 / 混合 / 长长。
    hist: std::collections::BTreeMap<usize, usize>,
}

fn decode_ours(path: &Path) -> Ours {
    let f = File::open(path).expect("打开样本");
    let mut r = PacketReader::new(BufReader::new(f));

    let mut id: Option<Vec<u8>> = None;
    let mut setup: Option<Vec<u8>> = None;
    let mut dec: Option<Decoder> = None;
    let mut planes: Vec<Vec<f32>> = Vec::new();
    let (mut packets, mut bad_packets) = (0usize, 0usize);
    let (mut sample_rate, mut blocksize_0, mut blocksize_1) = (0u32, 0usize, 0usize);
    let mut hist = std::collections::BTreeMap::new();

    while let Some(p) = r.next_packet().expect("拆包") {
        // 三个头包按 `\xNNvorbis` 的包类型位区分；音频包的包类型位是 0。
        if p.data.starts_with(b"\x01vorbis") {
            id = Some(p.data);
            continue;
        }
        if p.data.starts_with(b"\x03vorbis") {
            continue;
        }
        if p.data.starts_with(b"\x05vorbis") {
            setup = Some(p.data);
            continue;
        }
        if dec.is_none() {
            let d = Decoder::new(
                id.as_deref().expect("样本里没有识别头"),
                setup.as_deref().expect("样本里没有 setup 头"),
            )
            .expect("建解码器");
            sample_rate = d.sample_rate();
            blocksize_0 = d.info().blocksize_0;
            blocksize_1 = d.info().blocksize_1;
            planes = vec![Vec::new(); d.channels()];
            dec = Some(d);
        }
        let d = dec.as_mut().unwrap();
        match d.decode(&p.data) {
            Ok(out) => {
                if let Some(first) = out.first() {
                    *hist.entry(first.len()).or_insert(0usize) += 1;
                }
                for (dst, src) in planes.iter_mut().zip(&out) {
                    dst.extend_from_slice(src);
                }
                packets += 1;
            }
            // 单个坏包不该让整条对账失败 —— 但要计数，多了就说明理解错了。
            Err(e) => {
                bad_packets += 1;
                eprintln!("第 {packets} 个音频包解不开：{e}");
            }
        }
    }
    assert!(packets > 0, "一个音频包都没解出来");
    Ours { planes, packets, bad_packets, sample_rate, blocksize_0, blocksize_1, hist }
}

/// ffmpeg 解同一份文件，返回逐声道 f32（±1 归一化，和我们同一口径）。
fn decode_ffmpeg(path: &Path, channels: usize) -> Vec<Vec<f32>> {
    let raw = unique_temp("v2v_vorbis_ref.f32");
    let _ = std::fs::remove_file(&raw);
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            path.to_str().unwrap(),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-y",
            raw.to_str().unwrap(),
        ])
        .output()
        .expect("跑 ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg 解不开这个样本：{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&raw).expect("读 ffmpeg 的 PCM");
    let n = bytes.len() / 4 / channels;
    let mut planes = vec![Vec::with_capacity(n); channels];
    for i in 0..n {
        for (c, p) in planes.iter_mut().enumerate() {
            let o = (i * channels + c) * 4;
            p.push(f32::from_le_bytes([
                bytes[o],
                bytes[o + 1],
                bytes[o + 2],
                bytes[o + 3],
            ]));
        }
    }
    planes
}

/// 在 `[lo, hi)` 这段参考信号上，找让 `ours[..]` 对齐最好的偏移。
fn best_offset(ours: &[f32], reference: &[f32], max_shift: isize) -> isize {
    // 测量窗口放在参考信号的中段（避开两端的 priming / 收尾渐变）。
    let len = 4096usize.min(reference.len() / 3);
    let start = (reference.len() / 2).saturating_sub(len / 2);
    let mut best = (f64::MAX, 0isize);
    for sh in -max_shift..=max_shift {
        let mut se = 0f64;
        for j in 0..len {
            let oi = start as isize + j as isize + sh;
            if oi < 0 || oi as usize >= ours.len() {
                se = f64::MAX;
                break;
            }
            let d = f64::from(ours[oi as usize]) - f64::from(reference[start + j]);
            se += d * d;
        }
        if se < best.0 {
            best = (se, sh);
        }
    }
    best.1
}

#[test]
fn decodes_a_real_ogg_and_matches_ffmpeg() {
    let Some(path) = sample_path() else {
        eprintln!("SKIP: 没设 V2V_VORBIS_SAMPLE");
        return;
    };
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }

    let ours = decode_ours(&path);
    let ch = ours.planes.len();
    let reference = decode_ffmpeg(&path, ch);
    let our_len = ours.planes[0].len();
    let ref_len = reference[0].len();
    println!(
        "{}：{} Hz / {} 声道；解了 {} 个音频包（坏 {}），{} 个样本；ffmpeg 解出 {} 个样本",
        path.display(),
        ours.sample_rate,
        ch,
        ours.packets,
        ours.bad_packets,
        our_len,
        ref_len
    );
    assert_eq!(ours.bad_packets, 0, "有包解不开 —— 见上面的报错");

    // ---- 覆盖：短块 / hybrid 窗那条路真的被走过吗 ----
    //
    // 「自测全绿 ≠ 路径被覆盖」。本仓踩过这个坑：编码器只产长块，于是解码器的
    // 短窗 / 起止窗 / 窗形切换三条路径从没被执行过，四类 bug 全藏在那里。
    // 这里用输出长度的分布反推块序列（`Np/4 + N/4` 只有三种取值）。
    let short = ours.blocksize_0 / 4 * 2; // 短短：bs0/4 + bs0/4
    let long = ours.blocksize_1 / 4 * 2; // 长长
    let mixed = ours.blocksize_0 / 4 + ours.blocksize_1 / 4;
    println!(
        "块序列分布（bs0={} / bs1={}）：{short} 短短 ×{:?}，{mixed} 混合 ×{:?}，{long} 长长 ×{:?}",
        ours.blocksize_0,
        ours.blocksize_1,
        ours.hist.get(&short),
        ours.hist.get(&mixed),
        ours.hist.get(&long)
    );
    if ours.blocksize_0 != ours.blocksize_1 {
        assert!(
            ours.hist.get(&short).copied().unwrap_or(0) > 0,
            "这份样本一个短块都没有 —— 短窗路径没被验到"
        );
        assert!(
            ours.hist.get(&mixed).copied().unwrap_or(0) > 0,
            "这份样本没有长短切换 —— hybrid 窗路径没被验到"
        );
    }

    // ---- 判据 1：长度 ----
    //
    // 差一两个块是**正常**的：ffmpeg 会按 `granule_position` 裁掉尾部的
    // 补齐样本，我们的解码器只按块输出、不做裁剪。
    let diff = our_len.abs_diff(ref_len);
    assert!(
        diff <= 8192,
        "长度差 {diff}（我们 {our_len}，ffmpeg {ref_len}）—— 超过一页的量级"
    );

    // ---- 判据 3（先查）：比对区间不是静音 ----
    let peak = reference[0]
        .iter()
        .fold(0f32, |m, v| m.max(v.abs()));
    assert!(
        peak > 0.01,
        "参考信号几乎全是静音（峰值 {peak}）—— 这样的样本什么也证明不了"
    );

    for c in 0..ch {
        let sh = best_offset(&ours.planes[c], &reference[c], 2048);
        // 对齐之后，只在两边都有的区间里比。
        let lo = 0usize;
        let hi = our_len.min(ref_len);
        let (mut se, mut sp, mut worst) = (0f64, 0f64, 0f32);
        let (mut n, mut nonzero) = (0usize, 0usize);
        for i in lo..hi {
            let oi = i as isize + sh;
            if oi < 0 || oi as usize >= our_len {
                continue;
            }
            let a = f64::from(ours.planes[c][oi as usize]);
            let b = f64::from(reference[c][i]);
            se += (a - b) * (a - b);
            sp += b * b;
            worst = worst.max((a - b).abs() as f32);
            n += 1;
            if b.abs() > 0.01 {
                nonzero += 1;
            }
        }
        assert!(n > 100_000, "只比了 {n} 个样本，覆盖太窄");
        let snr = 10.0 * (sp / se.max(1e-30)).log10();
        println!(
            "声道 {c}：对齐偏移 {sh}，比了 {n} 个样本（其中 {nonzero} 个非静音），\
             SNR {snr:.1} dB，最大偏差 {worst:.2e}"
        );
        assert!(
            nonzero * 2 > n,
            "比对区间里只有 {nonzero}/{n} 个非静音样本 —— 判据落在静音上没有意义"
        );
        assert!(
            snr > 60.0,
            "声道 {c} 的 SNR 只有 {snr:.1} dB —— 窗 / 耦合 / 定标有系统性问题"
        );
    }
}
