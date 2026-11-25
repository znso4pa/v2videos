//! `.ogg → .mp4` 的端到端判据（**纯音频 MP4**）。
//!
//! ## 为什么这条要单独存在
//!
//! `.ogg` 没有视频轨，而本仓的 `.mp4` 路以前**必须**有一条视频轨（要写 VOL、定帧率）。
//! `Mp4Muxer::audio_only` 补上了「只有一个 `trak` 的 MP4」，`convert_audio_to_mp4`
//! 补上了那条流水线。`crates/v2v-mp4/tests/mux_audio_only.rs` 已经证明「ffmpeg 认这种文件、
//! 且 PCM 逐字节还原」，但它绕过了真正要用的那条路：`open_source` 按魔数认容器、
//! `classify_audio` 派发、`audio_total_samples` 裁尾、AAC 的 1024 点攒帧。
//!
//! ## 判据分两条，各自配一个**干净**的 oracle
//!
//! 1. **`--pcm` 路线**：MP4 里那条 `sowt` 轨解出来的 PCM，与 `--wav` 的输出
//!    **逐字节相同**。WAV 那条路已经由 `ogg_wav.rs` 对着 ffmpeg 钉住（≤1 LSB），
//!    所以这条断言真正验的是**封装**：`stsz` / `stts` / `stco` / `stsc` 填错、
//!    采样点多写或少写、`elst` 把时间轴推歪，都会让两者对不上。
//! 2. **AAC 路线**：ffprobe 认成「一条 `aac` 音轨、没有视频轨」，并且解码出来的波形
//!    与源**在 lag 0 对齐** —— 这一条专抓 `priming` 声明错。漏声明就整体晚 23 ms，
//!    而「能不能解 / 时长 / 帧数」这些判据**全都不会报警**。
//!
//! ⚠️ **不拿 SNR 当 AAC 的判据**：AAC 是有损重编码，SNR 取决于素材（实测同一份
//! 128 kbps 输出，正弦上 40+ dB、粉噪声上 24 dB）。拿它做阈值就是埋一个「换首歌
//! 就红」的测试。**对齐**才是内容无关的。
//!
//! 需要 `V2V_OGG_SAMPLE` 与 `ffmpeg` / `ffprobe`；缺任一个就跳过。

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use v2v_cli::{convert_file, ConvertOptions, Mp4Audio, OutputFormat};

fn sample_path() -> Option<PathBuf> {
    let p = std::env::var_os("V2V_OGG_SAMPLE")?;
    let p = PathBuf::from(p);
    if !p.is_file() {
        eprintln!("SKIP: V2V_OGG_SAMPLE 指向的文件不存在");
        return None;
    }
    Some(p)
}

fn has_tool(name: &str) -> bool {
    Command::new(name)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn unique_temp(stem: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{}_{n}", std::process::id()))
}

/// `ffprobe` 的 `key=value` 输出 → 取某个键。**不按字段顺序**（那是 ffprobe 的排版，
/// 不是判据）。
fn probe(path: &Path, entries: &str) -> Vec<(String, String)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", entries])
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(path)
        .output()
        .expect("跑不了 ffprobe");
    assert!(
        out.status.success(),
        "ffprobe 失败：{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let (k, v) = l.split_once('=')?;
            Some((k.trim().to_string(), v.trim().to_string()))
        })
        .collect()
}

fn get<'a>(kv: &'a [(String, String)], key: &str) -> &'a str {
    kv.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("ffprobe 输出里没有 {key}：{kv:?}"))
}

/// 让 ffmpeg 把一个文件解成 s16le 的交织 PCM。
fn ffmpeg_pcm(path: &Path) -> Vec<i16> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "s16le", "-acodec", "pcm_s16le", "-"])
        .output()
        .expect("跑不了 ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg 解不开 {}：{}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// 归一化互相关（同一位置的点积 / 能量）。用来判「有没有整体时移」。
fn corr(a: &[i16], b: &[i16], lag: isize, n: usize) -> f64 {
    let (mut dot, mut ea, mut eb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let j = i as isize + lag;
        if j < 0 || j as usize >= b.len() {
            continue;
        }
        let (x, y) = (f64::from(a[i]), f64::from(b[j as usize]));
        dot += x * y;
        ea += x * x;
        eb += y * y;
    }
    if ea == 0.0 || eb == 0.0 {
        return 0.0;
    }
    dot / (ea.sqrt() * eb.sqrt())
}

/// 两路 PCM 的 SNR（dB）。**只在「同一段内容」上比** —— 用来判「尾巴有没有被截掉」，
/// 不当成 AAC 的质量判据（见文件头）。
fn snr_db(a: &[i16], b: &[i16]) -> f64 {
    let (mut se, mut sp) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        se += (x - y) * (x - y);
        sp += x * x;
    }
    if se == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sp / se).log10()
}

/// 跑一次转换，返回 (报告, 输出路径)。
fn convert(input: &Path, audio: Mp4Audio, stem: &str) -> (v2v_cli::ConvertReport, PathBuf) {
    let dir = unique_temp(stem);
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("out.mp4");
    let opts = ConvertOptions {
        format: OutputFormat::Mp4,
        audio,
        ..Default::default()
    };
    let rep = convert_file(input, &out, &opts).expect("转码");
    (rep, out)
}

#[test]
fn ogg_to_mp4_pcm_is_bit_exact_with_the_wav_route() {
    let Some(input) = sample_path() else { return };
    if !has_tool("ffmpeg") || !has_tool("ffprobe") {
        eprintln!("SKIP: PATH 上没有 ffmpeg / ffprobe");
        return;
    }

    // 基准：`--wav` 那条路（它自己由 `ogg_wav.rs` 对着 ffmpeg 钉着）。
    let wav_dir = unique_temp("v2v_ogg_mp4_wav");
    std::fs::create_dir_all(&wav_dir).unwrap();
    let wav = wav_dir.join("ref.wav");
    let wopts = ConvertOptions {
        format: OutputFormat::Wav,
        ..Default::default()
    };
    let wrep = convert_file(&input, &wav, &wopts).expect("抽 WAV");
    let reference = ffmpeg_pcm(&wav);
    // ⚠️ 单位：`audio_payloads` 数的是**采样点**，ffmpeg 吐的是**交织样本**
    // （一个采样点 = 声道数个 i16）。这两个数差一个声道倍数 —— 别拿它们直接比。
    let channels: usize = get(&probe(&wav, "stream=channels"), "channels")
        .parse()
        .expect("声道数不是数字");
    assert_eq!(
        reference.len() as u64,
        wrep.audio_payloads as u64 * channels as u64,
        "WAV 的交织样本数与报告对不上（尺子先自检）"
    );

    let (rep, mp4) = convert(&input, Mp4Audio::Pcm, "v2v_ogg_mp4_pcm");
    println!(
        "{} → {}：{} 个采样点，{} 字节",
        input.display(),
        mp4.display(),
        rep.audio_payloads,
        rep.bytes_out
    );
    assert_eq!(rep.frames, 0, "纯音频 MP4 不该有视频帧");

    // ---- 长度必须正好等于容器声明的总采样数（末页 granule）----
    //
    // Vorbis 每块输出 `Np/4 + N/4` 个采样点，**最后一块会越过真实长度**
    // （实测 ffmpeg 编的样本多出 640 个点）。不按 granule 截尾，输出就会比源长
    // 一丁点 —— 不报错，只是「尾巴多了几十毫秒的残响」。
    //
    // ⚠️ 有些样本的最后一块正好落在 granule 上（比如 libvorbis 编的这个），
    // 那种样本上这条断言是**空过**：没有溢出就没有可裁的东西。所以别指望它
    // 在所有样本上都能抓到这个 bug —— 它保证的是「绝对长度对」。
    let total: u64 = get(&probe(&input, "stream=duration_ts"), "duration_ts")
        .parse()
        .expect("duration_ts 不是数字");
    assert_eq!(
        rep.audio_payloads as u64, total,
        "写出的采样点数与容器声明的总数不同（{} vs {}）—— 截尾没生效",
        rep.audio_payloads, total
    );

    // ---- 格式级 ----
    let s = probe(&mp4, "stream=codec_name,codec_type,channels,sample_rate");
    assert_eq!(get(&s, "codec_name"), "pcm_s16le", "编解码器不对");
    assert_eq!(get(&s, "codec_type"), "audio", "这条流应该是音频");
    let f = probe(&mp4, "format=nb_streams");
    assert_eq!(get(&f, "nb_streams"), "1", "纯音频 MP4 应该只有 1 条流");

    // ---- 样本级：与 WAV 逐字节相同 ----
    let ours = ffmpeg_pcm(&mp4);
    assert_eq!(
        ours.len(),
        reference.len(),
        "采样点数与 WAV 不同（我们 {}，WAV {}）—— 截尾没生效，或者 stsz/stts 填错了",
        ours.len(),
        reference.len()
    );
    if let Some(i) = ours.iter().zip(&reference).position(|(a, b)| a != b) {
        panic!(
            "第 {i} 个采样点不同（{} vs {}）—— 封装把样本顺序或偏移搞坏了",
            ours[i], reference[i]
        );
    }

    if !keep_temp() {
        let _ = std::fs::remove_dir_all(mp4.parent().unwrap());
        let _ = std::fs::remove_dir_all(&wav_dir);
    }
}

/// 临时目录要不要留。
///
/// 默认删掉；`V2V_KEEP_TEMP=1` 时留着 —— 判据红的时候，光看 `assert` 的
/// 那句话往往不够（「末尾 SNR 3.4 dB」不能告诉你**哪一段**坏了），
/// 得把产物捞出来自己量。
fn keep_temp() -> bool {
    std::env::var_os("V2V_KEEP_TEMP").is_some()
}

#[test]
fn ogg_to_mp4_aac_is_one_track_and_aligned_at_lag_zero() {
    let Some(input) = sample_path() else { return };
    if !has_tool("ffmpeg") || !has_tool("ffprobe") {
        eprintln!("SKIP: PATH 上没有 ffmpeg / ffprobe");
        return;
    }

    let (rep, mp4) = convert(&input, Mp4Audio::Aac, "v2v_ogg_mp4_aac");
    println!(
        "{} → {}：{} 个采样点，{} 字节",
        input.display(),
        mp4.display(),
        rep.audio_payloads,
        rep.bytes_out
    );

    // ---- 格式级：一条 aac 音轨，没有视频 ----
    let s = probe(&mp4, "stream=codec_name,codec_type,channels,sample_rate");
    assert_eq!(get(&s, "codec_name"), "aac", "编解码器不对");
    assert_eq!(get(&s, "codec_type"), "audio");
    let f = probe(&mp4, "format=nb_streams");
    assert_eq!(get(&f, "nb_streams"), "1", "纯音频 MP4 应该只有 1 条流");

    // 声道数 / 采样率必须与源一致 —— 重采样了就是错的（我们根本没做重采样）。
    let src = probe(&input, "stream=channels,sample_rate");
    assert_eq!(get(&s, "channels"), get(&src, "channels"), "声道数变了");
    assert_eq!(
        get(&s, "sample_rate"),
        get(&src, "sample_rate"),
        "采样率变了"
    );

    // ---- 时长：AAC 一帧 1024 点，尾巴最多补一帧 ----
    let dur: f64 = get(&probe(&mp4, "format=duration"), "duration")
        .parse()
        .expect("时长不是数字");
    let src_dur: f64 = get(&probe(&input, "format=duration"), "duration")
        .parse()
        .expect("时长不是数字");
    assert!(
        (dur - src_dur).abs() < 0.05,
        "时长差得太多：{dur} vs 源 {src_dur}"
    );

    // ---- 对齐：lag 0 必须显著胜出 ----
    //
    // 这是**内容无关**的判据（不看 SNR）。priming 漏声明 → 整体平移 1024 个点，
    // 那时 lag ±1024 的相关性会跟 lag 0 一样高甚至更高。
    let ours = ffmpeg_pcm(&mp4);
    let reference = ffmpeg_pcm(&input);
    let n = 16_384.min(ours.len()).min(reference.len());
    assert!(n > 4096, "样本太短，判不了对齐");
    let lags = [-2048isize, -1024, 0, 1024, 2048];
    let scores: Vec<(isize, f64)> = lags.iter().map(|&l| (l, corr(&ours, &reference, l, n))).collect();
    println!(
        "相关系数：{}",
        scores
            .iter()
            .map(|(l, c)| format!("lag {l} = {c:.3}"))
            .collect::<Vec<_>>()
            .join("  ")
    );
    let best = scores
        .iter()
        .cloned()
        .fold((0isize, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
    assert_eq!(
        best.0, 0,
        "最佳对齐在 lag {} —— 音频整体平移了，`priming` 声明错了？",
        best.0
    );
    assert!(
        best.1 > 0.5,
        "lag 0 的相关系数只有 {:.3} —— 解出来的东西和源几乎无关",
        best.1
    );

    // ---- 长度与尾巴：**一个真样本都不能少** ----
    //
    // ⚠️ 这里曾经少 384 个真样本。AAC 编码器是 MDCT + 50% 重叠，有一帧前瞻
    // （`output[k] = input[k-1]`）：喂完最后一个输入块之后不额外推一帧，
    // 它永远出不来。长度正好是 1024 的整数倍时丢满 1024 个（23 ms），
    // 而「帧数 / 时长 / 能不能解」这些判据**全都不报警**。
    //
    // 容差 48 个采样点（≈1 ms）：`elst.segment_duration` 的单位是影片时间轴的
    // 毫秒（1 ms = 44.1 个采样点），精度到不了采样点 —— 这是**尺子**的上限，
    // 不是被测物的。再往上收紧就得把 `mvhd` 的时间基调细（会改动所有产物）。
    let diff = ours.len() as i64 - reference.len() as i64;
    assert!(
        diff.abs() <= 48,
        "解码出来 {} 个采样点，源 {} 个（差 {diff}，超过 1 ms）",
        ours.len(),
        reference.len()
    );
    // 尾巴必须是**真内容**，不是补零 —— 长度对了但尾巴是静音同样是错的。
    let n = ours.len().min(reference.len());
    let tail = 2048.min(n);
    let tail_snr = snr_db(&ours[n - tail..], &reference[n - tail..]);
    println!("末尾 {tail} 个样本的 SNR = {tail_snr:.1} dB");
    assert!(
        tail_snr > 10.0,
        "末尾的 SNR 只有 {tail_snr:.1} dB —— 尾巴被截断或补成静音了"
    );

    if !keep_temp() {
        let _ = std::fs::remove_dir_all(mp4.parent().unwrap());
    }
}
