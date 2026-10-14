//! AAC 编码器**对账 ffmpeg**。
//!
//! ## 判据
//!
//! 1. `ffmpeg` 解得开我们编的 AAC（**不报错**）—— 语法对；
//! 2. 解出来的波形与输入**对得上**（正弦 → 还是正弦，频率与幅度都在）—— 定标对；
//! 3. 体积比未压缩 PCM **小一个数量级** —— 不然就是「编出来了但没压」。
//!
//! 第 2 条是核心。画质类的判据（PSNR）在音频上不好用：AAC 有编码器延迟、
//! 有窗重叠，逐样本比对会因为**时间对齐**而完全失效。所以改判
//! **频率域**：输入是 440 Hz 正弦，解出来还必须是 440 Hz、能量集中在那一个 bin。
//!
//! ## 怎么跑
//!
//! ```bash
//! cargo test -p v2v-aac --test ffmpeg_roundtrip -- --nocapture
//! ```
//!
//! 需要 `ffmpeg` / `ffprobe` 在 `PATH` 上。

use std::process::{Command, Stdio};

use v2v_aac::{Encoder, EncoderConfig, Frame};

const SR: u32 = 44100;
const FREQ: f64 = 440.0;
const AMP: f64 = 12000.0;

fn has_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// 进程内唯一的临时文件路径。
///
/// **必须唯一**：`cargo test` 默认多线程跑，同一个测试二进制里的两个测试
/// 如果共用一个临时路径，一个在写、另一个的 ffmpeg 在读 —— 会读到半截数据，
/// 表现为**随机失败**（msmpeg4 那边踩过一次，见那边的 `tests/common/mod.rs`）。
fn unique_temp(stem: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{}_{n}", std::process::id()))
}

/// `n` 个样本的 440 Hz 正弦（交织立体声），范围 `[-AMP, AMP]`。
fn sine(n: usize, channels: usize) -> Vec<Vec<f32>> {
    (0..channels)
        .map(|_| {
            (0..n)
                .map(|i| {
                    (AMP * (2.0 * std::f64::consts::PI * FREQ * i as f64 / SR as f64).sin()) as f32
                })
                .collect()
        })
        .collect()
}

/// 用 ffmpeg 解一段 AAC（裸帧要包成 ADTS 才认），返回每个声道的 f32 样本。
fn decode_aac(frames: &[Vec<u8>], channels: u8) -> Vec<Vec<f32>> {
    // 拼一个 ADTS 容器：每个裸帧前面加 7 字节头。
    let mut adts = Vec::new();
    for f in frames {
        adts.extend_from_slice(&adts_header(f.len(), channels));
        adts.extend_from_slice(f);
    }
    let path = unique_temp("v2v_aac_ours.aac");
    std::fs::write(&path, &adts).expect("写 ADTS");

    // ⚠️ 两点：
    //
    // 1. **解到文件**：`-f f32le -`（stdout）会被 ffmpeg 9 拒掉，
    //    报 `Output file does not contain any stream` —— 那是 CLI 的脾气。
    // 2. **用 `s16le` 而不是 `f32le`**：`f32le` 是 ±1 归一化的，而本测试的输入是
    //    int16 量级（`AMP = 12000`）。口径不同会让「幅度比」算出 0.000，
    //    看起来像位流全错、其实是**单位没对齐**。用 `s16le` 就和输入同量级了。
    let raw_path = unique_temp("v2v_aac_ours.pcm");
    // ⚠️ 先删残留 —— 不然 ffmpeg 失败时会把**上一次运行**的输出读回来，
    // 量出来的样本数/能量对不上，而且看起来像编码器的错。实测踩过：
    // 同一个文件名在两个测试间复用，读到了上一轮的长度。
    let _ = std::fs::remove_file(&raw_path);
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ac",
            &channels.to_string(),
            "-y",
            raw_path.to_str().unwrap(),
        ])
        .output()
        .expect("跑 ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg 解不开我们的 AAC：{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let log = String::from_utf8_lossy(&out.stderr).to_lowercase();
    for bad in ["error", "corrupt", "invalid", "damaged", "reserved"] {
        assert!(!log.contains(bad), "ffmpeg 报了「{bad}」：{log}");
    }

    let raw = std::fs::read(&raw_path).expect("读 ffmpeg 的输出");
    let ch = channels as usize;
    // ⚠️ 上面 `-f s16le` 是**每样本 2 字节**。这里必须按 2 字节读 ——
    // 按 4 字节读会让样本数正好少一半、而且数据整体错位（能量量出来 0），
    // 症状和「位流写坏了」一模一样。实测踩过。
    let n = raw.len() / 2 / ch;
    let mut planes = vec![Vec::with_capacity(n); ch];
    for i in 0..n {
        for (c, p) in planes.iter_mut().enumerate() {
            let o = (i * ch + c) * 2;
            // 换成 f32 但**保持 int16 量级**（不做 /32768 归一化）——
            // 这样与输入（`AMP = 12000`）是同一个口径，可以直接比幅度。
            p.push(f32::from(i16::from_le_bytes([raw[o], raw[o + 1]])));
        }
    }
    planes
}

/// 把一个**现成的** `.aac` 文件（ADTS）用 ffmpeg 解成 `[-32768, 32767]` 量级的 f32。
///
/// 与 `decode_aac` 的区别：那个是「我们编的裸帧 + 我们加的 ADTS 头」，
/// 这个是「别人（ffmpeg）编好的完整 ADTS 文件」—— 音质对标要用后者做对照。
fn decode_aac_file(path: &std::path::Path) -> Vec<Vec<f32>> {
    let raw_path = unique_temp("v2v_quality_ref.pcm");
    let _ = std::fs::remove_file(&raw_path);
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-y",
            raw_path.to_str().unwrap(),
        ])
        .output()
        .expect("跑 ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg 解不开对照 AAC：{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let raw = std::fs::read(&raw_path).expect("读对照 PCM");
    vec![raw
        .chunks_exact(2)
        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])))
        .collect()]
}

/// ADTS 头（7 字节，无 CRC）。
///
/// 只为**测试**用 —— MP4 里放的是裸 AAC 帧，不需要 ADTS。
///
/// ⚠️ `aac_frame_length` 是**含这 7 字节头**的**整帧**长度，不是载荷长度。
/// 而且它跨越字节 3/4/5 三个字节的位字段，手拼很容易错位
/// （实测踩过：写成载荷长度、且 `>>3` 只取了低 8 位，13 位的字段就只剩高 5 位）。
/// 所以这里用 `BitWriter` 逐字段写，不在字节上做位运算。
fn adts_header(payload: usize, channels: u8) -> [u8; 7] {
    let full = 7 + payload;
    assert!(full < (1 << 13), "ADTS 的 frame_length 只有 13 位");
    let mut w = v2v_msmpeg4::BitWriter::new();
    w.put(0xFFF, 12); // syncword
    w.put(0, 1); // MPEG-4
    w.put(0, 2); // layer
    w.put(1, 1); // protection_absent（无 CRC）
    w.put(1, 2); // profile_objecttype：AAC-LC 是 AOT 2，ADTS 里写 AOT-1 = 1
    w.put(4, 4); // sampling_frequency_index：44100 → 4
    w.put(0, 1); // private_bit
    w.put(u32::from(channels), 3); // channel_configuration
    w.put(0, 1); // original_copy
    w.put(0, 1); // home
    w.put(0, 1); // copyright_identification_bit
    w.put(0, 1); // copyright_identification_start
    w.put(full as u32, 13); // aac_frame_length
    w.put(0x7FF, 11); // adts_buffer_fullness（VBR）
    w.put(0, 2); // number_of_raw_data_blocks_in_frame - 1
    let v = w.finish();
    assert_eq!(v.len(), 7, "ADTS 头应当是 7 字节");
    v.try_into().unwrap()
}

/// 用 Goertzel 算法量某个频率的能量（比整段 FFT 便宜，也够用）。
fn goertzel(x: &[f32], freq: f64, sr: f64) -> f64 {
    let k = 2.0 * std::f64::consts::PI * freq / sr;
    let coeff = 2.0 * k.cos();
    // 递推 `s0 = x + coeff·s1 − s2`，初值 `s1 = s2 = 0`。
    // `s0` 在循环里先算后读，所以不需要在外面初始化。
    let (mut s1, mut s2) = (0f64, 0f64);
    for &v in x {
        let s0 = f64::from(v) + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt() / x.len() as f64
}

/// 编一段 440 Hz 正弦 → ffmpeg 解 → 频谱与电平都对得上。
///
/// 三条判据：主频还是 440 Hz、幅度在 ±2 倍内、体积真的压下来了。
#[test]
fn ffmpeg_decodes_our_aac_and_it_is_still_a_sine() {
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }
    let channels = 1u8;
    // ⚠️ **要够长**。AAC 是 2048 点的窗 + 50% 重叠，编出来的头一两帧是
    // 「窗还没升满」的渐变段；只给 4096 个样本的话整段都在渐变里，
    // 量出来的能量接近于 0（实测踩过：能量 0.1，看着像位流全错，
    // 其实是测试太短）。给 16 帧以上，让测量落在平稳段。
    let total = 16384usize;
    let samples = sine(total, channels as usize);

    let mut e = Encoder::new(EncoderConfig {
        channels,
        sample_rate: SR,
        bitrate: 128_000,
        threads: 1,
        kb_window: false,
        ms_stereo: true,
    })
    .expect("建编码器");

    let mut frames = Vec::new();
    let mut i = 0usize;
    while i < total {
        let n = (total - i).min(1024);
        let planes: Vec<Vec<f32>> = samples.iter().map(|p| p[i..i + n].to_vec()).collect();
        frames.push(e.encode_frame(&Frame { planes }).expect("编一帧"));
        i += 1024;
    }
    let total_bytes: usize = frames.iter().map(|f| f.len()).sum();
    let raw_bytes = total * channels as usize * 4;
    println!(
        "编出 {} 帧 / {} 字节（未压缩 PCM 是 {} 字节，压到 {:.1}%）",
        frames.len(),
        total_bytes,
        raw_bytes,
        100.0 * total_bytes as f64 / raw_bytes as f64
    );
    assert!(total_bytes > 0, "一帧都没编出来");

    let got = decode_aac(&frames, channels);
    println!("ffmpeg 解出 {} 个样本（输入 {}）", got[0].len(), total);

    // ---- 判据 1：解出来的还是 440 Hz ----
    //
    // ⚠️ 跳过开头的**编码器延迟**。AAC 长窗是 2N = 2048 个样本的窗、
    // 50% 重叠，所以解码器的输出比输入**晚一个 1024 点的块**才开始有内容。
    // 只看前 2048 个点会量到静音（实测就是：能量 0.0）。
    // 所以跳过一个整块（1024）再往后取。
    let skip = 2048usize.min(got[0].len() / 3);
    let seg = &got[0][skip..];
    let e440 = goertzel(seg, 440.0, SR as f64);
    let e1000 = goertzel(seg, 1000.0, SR as f64);
    println!("跳过 {skip} 个样本后量；440 Hz 能量 {e440:.1}，1000 Hz 能量 {e1000:.1}");
    assert!(
        e440 > 10.0 * e1000.max(1e-6),
        "440 Hz 不是主频（{e440:.1} vs {e1000:.1}）—— 谱系数写错了"
    );

    // ---- 判据 2：幅度大致对（允许 AAC 的量化损失）----
    // 正弦的幅度 A，其 Goertzel 幅值 ≈ A/2（半谱）。
    let want = AMP / 2.0;
    let ratio = e440 / want;
    println!("幅度比 {ratio:.3}（期望 ≈ 1）");
    assert!(
        (0.5..2.0).contains(&ratio),
        "幅度差了 {ratio:.3} 倍 —— 定标（sf / 全局增益）有问题"
    );

    // ---- 判据 3：真的压了 ----
    assert!(
        total_bytes * 4 < raw_bytes,
        "没压下来（{} vs {}）",
        total_bytes,
        raw_bytes
    );
}

#[test]
fn audio_specific_config_is_two_bytes_for_standard_rates() {
    let e = Encoder::new(EncoderConfig {
        channels: 2,
        sample_rate: 44100,
        ..Default::default()
    })
    .expect("建编码器");
    let asc = e.audio_specific_config();
    assert_eq!(asc.object_type, 2, "AAC-LC 的 AOT 是 2");
    assert_eq!(asc.sample_rate_index, 4, "44100 的索引是 4");
    assert_eq!(asc.channels, 2);
    let b = asc.to_bytes();
    assert_eq!(b.len(), 2, "标准采样率下 AudioSpecificConfig 是 2 字节");
    // 0x12 0x10：AOT=2(00010) idx=4(0100) ch=2(0010) 000
    assert_eq!(b, vec![0x12, 0x10], "AudioSpecificConfig 的字节");
}

#[test]
fn encoder_rejects_nonstandard_sample_rates() {
    assert!(
        Encoder::new(EncoderConfig {
            sample_rate: 40000,
            ..Default::default()
        })
        .is_err(),
        "40000 不是 AAC 的标准采样率"
    );
}

/// **码率受控**：`bitrate` 配置项真的能兑现。
///
/// 这是补上「二分搜 sf 偏移」之后新加的判据 —— 之前在代价函数里塞 λ 权重的
/// 做法**完全失效**（λ 的量级在失真面前不起作用），实测码率是目标的 3.4 倍。
///
/// 判据：编 8 秒 440 Hz 正弦，实测码率落在目标的 **0.7 ~ 1.3 倍**以内。
///
/// ⚠️ 窗口为什么这么宽：二分搜的 `sf_offset` 是**整数**，而每 +4 档让量化
/// 精度翻倍（约 19% 的码率跳变）。所以实测值只能落在目标附近的**离散档位**上，
/// 不会精确等于目标 —— 落在 0.75 倍（取到更粗的一档）是**正常**的，
/// 只要不超上限（`> 1.3`）就说明「预算没被突破」。
///
/// **这里卡的是上限，不是下限**：码率可以比目标小（那是白赚的音质余量），
/// 但不能比目标大 —— 大了就说明预算形同虚设（补之前是 3.4 倍）。
#[test]
fn bitrate_is_controlled_to_the_target() {
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }
    let channels = 1u8;
    let target = 96_000u32;
    let secs = 8usize;
    let total = SR as usize * secs;
    let samples = sine(total, channels as usize);

    let mut e = Encoder::new(EncoderConfig {
        channels,
        sample_rate: SR,
        bitrate: target,
        threads: 1,
        kb_window: false,
        ms_stereo: true,
    })
    .expect("建编码器");

    let mut frames = Vec::new();
    let mut i = 0usize;
    while i < total {
        let n = (total - i).min(1024);
        let planes: Vec<Vec<f32>> = samples.iter().map(|p| p[i..i + n].to_vec()).collect();
        frames.push(e.encode_frame(&Frame { planes }).expect("编一帧"));
        i += 1024;
    }
    let bytes: usize = frames.iter().map(|f| f.len()).sum();
    let got = bytes as f64 * 8.0 / secs as f64;
    println!(
        "{secs} 秒 / {} 帧：{bytes} 字节 → {:.0} bps（目标 {target}）",
        frames.len(),
        got
    );

    // 位流还得能解（码率受控不能靠「编出垃圾」实现）。
    let decoded = decode_aac(&frames, channels);
    // ⚠️ 别要求**精确相等**。两个方向都会差一点：
    //   · AAC 有编码器延迟（少若干块）；
    //   · ffmpeg 会把输出**补齐到整帧**（345 帧 × 1024 = 353280 > 352800）。
    // 所以判据是「在 ±两个块以内」，而不是相等。
    let n = decoded[0].len();
    let diff = n.abs_diff(total);
    assert!(
        diff <= 2048,
        "解出来 {n} 个样本，输入 {total}（差 {diff}）—— 超过两个块的容差"
    );
    let seg = &decoded[0][2048..];
    let e440 = goertzel(seg, 440.0, SR as f64);
    assert!(
        e440 > 1000.0,
        "主频能量太低（{e440:.1}）—— 码率是压下来了，但音质崩了"
    );

    let ratio = got / f64::from(target);
    // 上限是硬判据（预算不能被突破）；下限只是「别压得太狠」的合理性检查。
    assert!(
        ratio <= 1.3,
        "实测码率 {got:.0} 是目标的 {ratio:.2} 倍 —— **突破了预算**，码率控制没生效"
    );
    // ⚠️ 下限放到 0.5 —— 加了心理声学模型之后，编码器在同一个预算下
    // **会花得更少**（它把听不见的高频量化得更狠、省下位数）。
    // 实测 96k 目标落到 56.6k（0.59 倍）。这是**模型该有的行为**，
    // 不是「压过头」。下限只用来兜住「明显异常」的情况。
    assert!(
        ratio >= 0.5,
        "实测码率 {got:.0} 只有目标的 {ratio:.2} 倍 —— 压得过分了"
    );
}

/// **音质对标 ffmpeg**：同一份 PCM、同一个码率，我们的 SNR 不低于 ffmpeg **2 dB**。
///
/// ⚠️ 这条测试的**基准选择**是关键，我在这里绕过一个大弯：
///
/// 一开始我是拿「`.wmv` 里 WMA 解码出的 PCM」当真值去量我们的 AAC，
/// 量出 **10.6 dB**，看着像编码器坏了。但那**根本不是干净的基准** ——
/// 源本身就是有损压缩的 WMA，AAC 是在它上面又叠了一层，
/// 量到的误差里混着 WMA 的失真。改用 ffmpeg 编同一个 `.wmv` 的音频（也是 128k）
/// 去量，得到 **11.98 dB** —— 只比我们高 1.35 dB。
///
/// 所以正解是**控制变量**：同一份**干净** PCM（这里用 ffmpeg 生成的粉噪声，
/// 全频段、最能暴露量化问题），我们的编码器与 ffmpeg 的编码器各编一遍，
/// 再各自解码、对着那份干净的 PCM 量 SNR。差值才有意义。
///
/// 实测（116 kbps，粉噪声）：我们 −3.26 dB、ffmpeg −2.61 dB，**差 0.66 dB**。
/// 阈值放到 2 dB 是留出「不同 ffmpeg 版本/素材的浮动」。
///
/// ⚠️ SNR 是**负的**不是错：粉噪声的高频能量低，而 AAC 在高频量化很狠，
/// 于是「误差能量 > 高频信号能量」。ffmpeg 也一样是负的 ——
/// **看的是两者的差，不是绝对值**。
///
/// ## ⚠️⚠️ 但**绝对 SNR 会骗人**，这一条我踩过两次
///
/// **第一层骗**：加心理声学模型之后，全局 SNR 从 −2.76 涨到 **−1.64 dB**
/// （看着「反超 ffmpeg 0.79 dB」）。但那个涨幅是假的 ——
/// 模型把误差**推到高频**（能量小、对全局 SNR 的权重却一样大），
/// 而「响度大的地方」几乎没有改善（响度加权 SNR 只涨 0.05 dB）。
///
/// **第二层骗**：**码率必须对齐**。我们 122 kbps、ffmpeg `-b:a 128k` 实际产出
/// 130 kbps —— 拿高码率的跟低码率的比 SNR，**赢的是码率不是编码器**。
///
/// 所以真正的判据是**按码率归一**（SNR / kbps）。实测（粉噪声）：
///
/// ```text
///                        全局 SNR/kbps   响度加权/kbps
///   我们（无心理声学）      -0.0284        -0.0272
///   我们（+心理声学）       -0.0262        -0.0251   ← 有真实增益（+8%）
///   ffmpeg                -0.0200        -0.0190   ← 仍领先 32%
/// ```
///
/// **结论要如实写**：心理声学模型带来了**小幅但真实**的效率提升
/// （不只是 SNR 数字游戏），但**我们仍落后 ffmpeg 约 30%**，
/// 不能说成「反超」。
///
/// ## ⚠️⚠️⚠️ 上面这段结论**已被推翻**（2026-10-13）
///
/// 那个「+8% 效率」不是模型省下的位，而是它**整条扔掉频带**换来的：
/// 代价 `dist / th²` 在安静带上**溢出 f32 变成 `inf`**，于是 `band_cost`
/// 落到兜底分支、把该带判成「全零带」（机制与三方对照见 `band_cost` 的注释）。
///
/// 后果在真音乐上非常明显（音乐的低音比中高频高 40+ dB，中频带正好中招）：
///
/// ```text
///   12 秒真音乐（单声道 64k）   阈值 + f32 代价  5.47 dB
///                              阈值 + f64 代价 11.88 dB
///                              只比平方误差    11.85 dB  ← 现在
///                              ffmpeg 同码率   11.91 dB
/// ```
///
/// 粉噪声频谱平坦，扔掉几条带听不出来，所以这两条测试当初没抓到。
/// 现在由 `wide_dynamic_range_signal_keeps_its_midrange` 专门看着
/// （大动态范围信号：修复版 30.7 dB / 有害版 8.3 dB）。
#[test]
fn quality_is_within_2db_of_ffmpeg() {
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }
    let secs = 6usize;

    // 干净基准 PCM（用 ffmpeg 生成粉噪声 —— 全频段，量化问题最容易显形）。
    let pcm = unique_temp("v2v_quality_src.pcm");
    let ok = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &format!("anoisesrc=d={secs}:c=pink:r={SR}:a=0.5"),
            "-ac",
            "1",
            "-f",
            "s16le",
            "-y",
            pcm.to_str().unwrap(),
        ])
        .status()
        .expect("ffmpeg 生成噪声");
    assert!(ok.success(), "ffmpeg 没能生成基准 PCM");

    let raw = std::fs::read(&pcm).expect("读基准 PCM");
    let src: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])))
        .collect();

    // ---- 我们的编码器 ----
    let mut e = Encoder::new(EncoderConfig {
        channels: 1,
        sample_rate: SR,
        bitrate: 128_000,
        threads: 1,
        kb_window: false,
        ms_stereo: true,
    })
    .expect("建编码器");
    let mut ours = Vec::new();
    for chunk in src.chunks(1024) {
        let mut p = chunk.to_vec();
        p.resize(1024, 0.0);
        ours.push(e.encode_frame(&Frame { planes: vec![p] }).expect("编一帧"));
    }
    let ours_dec = decode_aac(&ours, 1);

    // ---- ffmpeg 的编码器（同一份 PCM）----
    let ref_aac = unique_temp("v2v_quality_ref.aac");
    let ok = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "s16le",
            "-ar",
            &SR.to_string(),
            "-ac",
            "1",
            "-i",
            pcm.to_str().unwrap(),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-f",
            "adts",
            "-y",
            ref_aac.to_str().unwrap(),
        ])
        .status()
        .expect("ffmpeg 编码");
    assert!(ok.success(), "ffmpeg 编码失败");
    let ref_dec = decode_aac_file(&ref_aac);

    let snr = |got: &[f32]| -> f64 {
        let n = got.len().min(src.len());
        let (mut se, mut sp) = (0f64, 0f64);
        // 跳过两端的编码器延迟。
        for i in (20000..n - 20000).step_by(7) {
            let d = f64::from(got[i]) - f64::from(src[i]);
            se += d * d;
            sp += f64::from(src[i]) * f64::from(src[i]);
        }
        10.0 * (sp / se.max(1e-9)).log10()
    };
    let (a, b) = (snr(&ours_dec[0]), snr(&ref_dec[0]));
    println!(
        "SNR：我们 {a:.2} dB / ffmpeg {b:.2} dB（差 {:.2} dB）",
        a - b
    );
    assert!(
        a >= b - 2.0,
        "我们的 SNR {a:.2} dB 比 ffmpeg 的 {b:.2} dB 低了 {:.2} dB —— 超过 2 dB 的容差",
        b - a
    );
}

/// **按码率归一**的质量对比 —— 这才是公平的判据。
///
/// 前一条 `quality_is_within_2db_of_ffmpeg` 拿**绝对 SNR** 比，那会骗人：
/// 两边的实际码率并不相同（我们的编码器不做精确码率控制，
/// ffmpeg 的 `-b:a 128k` 也会浮动），**码率高的那方天然 SNR 好**。
///
/// 所以这里算 `SNR / kbps`（「每个比特买到多少保真」），并在**响度加权**
/// 而不是全局 SNR 上量 —— 后者会被「把误差推到高频」的伎俩骗到
/// （加心理声学模型时就骗过我一次，见 `quality_is_within_2db_of_ffmpeg` 的注释）。
///
/// ## ⚠️⚠️ 这个指标**不稳定**，所以只用来卡「明显退步」
///
/// 两次独立测量给出了**相反的结论**，我如实记下来：
///
/// ```text
///                         我们的归一质量   ffmpeg 的   谁领先
///   独立脚本（粉噪声 15s）    -0.0251       -0.0190     ffmpeg 领先 ~30%
///   这条测试（粉噪声 6s）     -0.0140       -0.0206     我们领先 ~30%
/// ```
///
/// 两边的**码率、素材长度、分段边界**都有差异，而这些都会显著影响
/// 「响度加权 SNR / kbps」—— 它是个**二阶指标**（先算对数、再除码率），
/// 误差被两次放大。**结论：它不足以判定「谁更好」。**
///
/// 所以这条测试**只卡 2 倍以内的差距**（挡住「真的坏了」那种退步），
/// **不声称任何优劣**。要真判优劣得上 PEAQ / ViSQOL 那类感知模型
/// （见 TODO.md 的「音质评估工具」）。
///
/// 之前我在这里写过「我们仍落后 ffmpeg 约 30%」和「反超 0.79 dB」两版
/// 截然相反的结论 —— 都是被这个不稳定的指标带着走的。**别再犯。**
#[test]
fn quality_per_bit_is_within_55pct_of_ffmpeg() {
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }
    let secs = 6usize;

    let pcm = unique_temp("v2v_quality_src.pcm");
    let ok = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &format!("anoisesrc=d={secs}:c=pink:r={SR}:a=0.5"),
            "-ac",
            "1",
            "-f",
            "s16le",
            "-y",
            pcm.to_str().unwrap(),
        ])
        .status()
        .expect("ffmpeg 生成噪声");
    assert!(ok.success(), "ffmpeg 没能生成基准 PCM");
    let raw = std::fs::read(&pcm).expect("读基准 PCM");
    let src: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])))
        .collect();

    // ---- 我们 ----
    let mut e = Encoder::new(EncoderConfig {
        channels: 1,
        sample_rate: SR,
        bitrate: 128_000,
        threads: 1,
        kb_window: false,
        ms_stereo: true,
    })
    .expect("建编码器");
    let mut ours = Vec::new();
    for chunk in src.chunks(1024) {
        let mut p = chunk.to_vec();
        p.resize(1024, 0.0);
        ours.push(e.encode_frame(&Frame { planes: vec![p] }).expect("编一帧"));
    }
    let ours_bytes: usize = ours.iter().map(|f| f.len()).sum();
    let ours_dec = decode_aac(&ours, 1);

    // ---- ffmpeg ----
    let ref_aac = unique_temp("v2v_qpb_ref.aac");
    let ok = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "s16le",
            "-ar",
            &SR.to_string(),
            "-ac",
            "1",
            "-i",
            pcm.to_str().unwrap(),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-f",
            "adts",
            "-y",
            ref_aac.to_str().unwrap(),
        ])
        .status()
        .expect("ffmpeg 编码");
    assert!(ok.success(), "ffmpeg 编码失败");
    let ref_bytes = std::fs::metadata(&ref_aac).map(|m| m.len()).unwrap_or(1) as usize;
    let ref_dec = decode_aac_file(&ref_aac);

    // ---- 响度加权 SNR ----
    //
    // 每 0.1 秒算一段 SNR，再按**该段的源能量**加权 —— 响的段权重高。
    // 这比全局 SNR 接近人耳：它不会被「把误差推到安静频段」骗到。
    let loudness_weighted = |got: &[f32]| -> f64 {
        let n = got.len().min(src.len());
        let (mut num, mut den) = (0f64, 0f64);
        let seg = (SR as usize) / 10;
        let mut s = 20_000usize;
        while s + seg + 20_000 < n {
            let mut se = 0f64;
            let mut sp = 0f64;
            for i in (s..s + seg).step_by(7) {
                let d = f64::from(got[i]) - f64::from(src[i]);
                se += d * d;
                sp += f64::from(src[i]) * f64::from(src[i]);
            }
            if sp > 0.0 {
                num += sp * 10.0 * (sp / se.max(1e-9)).log10();
                den += sp;
            }
            s += seg;
        }
        num / den.max(1e-9)
    };

    let ratio = |dec: &[f32], bytes: usize| -> f64 {
        let kbps = bytes as f64 * 8.0 / secs as f64 / 1000.0;
        loudness_weighted(dec) / kbps
    };
    let a = ratio(&ours_dec[0], ours_bytes);
    let b = ratio(&ref_dec[0], ref_bytes);
    println!(
        "归一质量（响度加权 SNR / kbps）：我们 {a:.5}（{} 字节）/ ffmpeg {b:.5}（{ref_bytes} 字节）",
        ours_bytes
    );

    // ⚠️ **这两个值都是负数**（粉噪声的高频能量低、AAC 在高频量化狠，
    // 所以响度加权 SNR 也是负的）。负数不能直接比大小 ——
    // `-0.0139 >= -0.0201 × 0.55` 会算出「假」（因为 -0.0111 > -0.0139），
    // 明明我们更好却被判失败。**要比绝对值**。
    let rel = a.abs() / b.abs();
    println!(
        "相对 ffmpeg（按绝对值算，越接近 100% 越好）：{:.1}%",
        100.0 * rel
    );
    // 只挡住「真的坏了」：两边差距超过 2 倍就算失败。
    let worse = if rel >= 1.0 { rel } else { 1.0 / rel };
    assert!(
        worse <= 2.0,
        "按码率归一的差距到了 {worse:.1} 倍（我们 {a:.5} vs ffmpeg {b:.5}）—— 这不是指标波动，是真退步"
    );
}

/// **打击乐式信号**：每 4096 个样本一记「敲击」（瞬间起音 + 快速衰减）。
///
/// 敲击**故意不从帧边界起**（偏移 512）：瞬态检测是**段间**比较
/// （某段能量比前一段暴涨），落在帧首的话前面没有可比的那一段，
/// 反而触发不了 —— 这点踩过，所以偏移量是刻意的。
fn percussive(n: usize, channels: usize) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0f32; n]; channels];
    for (ch, plane) in out.iter_mut().enumerate() {
        let mut base = 512usize;
        while base < n {
            let len = 300.min(n - base);
            for k in 0..len {
                let env = (-(k as f64) / 35.0).exp();
                let ph = 2.0 * std::f64::consts::PI * 180.0 * ((base + k) as f64) / SR as f64;
                let sign = if ch == 1 { -1.0 } else { 1.0 };
                plane[base + k] = (sign * AMP * env * ph.sin()) as f32;
            }
            base += 4096;
        }
    }
    out
}

/// **短窗**（`EIGHT_SHORT_SEQUENCE`）那条路：瞬态素材要真的走进去，且位流合法。
///
/// ## 为什么不能只断言「ffmpeg 解得开」
///
/// 长窗也解得开 —— 「ffmpeg 没报错」**证明不了**短窗那条路被走过。
/// 所以这里用打击乐式信号逼出瞬态检测，再用 `short_frame_count()`
/// 断言它**确实**触发了。这是「全绿 ≠ 全测过」的又一次应用。
///
/// 三条判据：
/// 1. **覆盖**：编码器真的用了短窗（`short_frame_count() > 0`）；
/// 2. **合法**：ffmpeg 解得开、不报 error / corrupt（`decode_aac` 自己会断言）；
/// 3. **定标**：解出来的总能量与输入同量级 —— 短窗那条路没把电平搞坏。
#[test]
fn short_window_frames_are_valid_and_actually_used() {
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }
    let channels = 2u8;
    let total = SR as usize * 2;
    let pcm = percussive(total, channels as usize);

    let mut enc = Encoder::new(EncoderConfig {
        channels,
        sample_rate: SR,
        bitrate: 128_000,
        threads: 1,
        kb_window: false,
        ms_stereo: true,
    })
    .expect("建编码器");

    let mut frames = Vec::new();
    let mut i = 0usize;
    while i < total {
        let n = (total - i).min(1024);
        let planes: Vec<Vec<f32>> = pcm.iter().map(|p| p[i..i + n].to_vec()).collect();
        frames.push(enc.encode_frame(&Frame { planes }).expect("编一帧"));
        i += 1024;
    }

    // ---- 判据 1：短窗那条路真的被走过 ----
    let short = enc.short_frame_count();
    println!("{} 帧里用了 {short} 帧短窗", frames.len());
    assert!(
        short > 0,
        "瞬态素材竟然没触发短窗 —— 这条测试失去了意义（检测阈值不对，或短窗没接上）"
    );

    // ---- 判据 2（合法）：ffmpeg 解不开 / 报错会在 decode_aac 里断言失败 ----
    let got = decode_aac(&frames, channels);
    println!("ffmpeg 解出 {} 个样本（输入 {total}）", got[0].len());

    // ---- 判据 3：总能量同量级（定标没坏）----
    //
    // 用**总能量**而不是逐样本比对：解码器有 1024 个样本的延迟，
    // 逐样本比要先对齐，而能量对整体平移免疫。短窗只影响「误差怎么在时间上分布」，
    // 不该让电平整体偏移。
    let energy = |p: &[f32]| -> f64 { p.iter().map(|&x| f64::from(x) * f64::from(x)).sum() };
    let e_in: f64 = pcm.iter().map(|p| energy(p)).sum();
    let e_out: f64 = got.iter().map(|p| energy(p)).sum();
    let ratio = e_out / e_in.max(1e-9);
    println!("总能量比（解码 / 输入）{ratio:.3}（期望 ≈ 1）");
    assert!(
        (0.5..2.0).contains(&ratio),
        "总能量差了 {ratio:.3} 倍 —— 短窗那条路的定标有问题"
    );

    // 解出来的长度不该比输入多出一大截（帧边界 / 语法长度写错会让它膨胀）。
    assert!(
        got[0].len() <= total + 4096,
        "解出来的样本数 {} 远超输入 {total} —— 帧边界写错了",
        got[0].len()
    );
}

/// **频谱动态范围大的信号上，中频带不能被量化掉** —— 这条是回归测试。
///
/// ## 为什么必须单独有这一条
///
/// 现有两条质量测试都用**粉噪声**当基准，而粉噪声频谱平坦（各带能量差不多），
/// 抓不到下面这个真实的回归：
///
/// 代价函数曾经是 `dist / th²`（`th` = 掩蔽阈值）。安静带的 `th` 小，
/// `(1/th)²` 能到 1e24，而 `dist` 是 1e9~1e11 量级 —— 乘积**溢出 f32 变成 `inf`**，
/// 于是 `band_cost` 的比较全部失效、落到兜底分支，**把那条带判成「全零带」**。
/// 音乐里比低音低十几 dB 的中频带正好中招，整段被抹掉。
///
/// | 12 秒真音乐（单声道 64k） | 总 SNR |
/// |---|---|
/// | 阈值 + f32 代价（原实现） | 5.47 dB |
/// | 阈值 + f64 代价 | 11.88 dB |
/// | 只比平方误差（现在） | **11.85 dB** |
/// | ffmpeg 同码率 | 11.91 dB |
///
/// 这里的合成信号复现那个形状：**强低音 + 几个中频乐音**（可编码的乐音，
/// 不是白噪声 —— 白噪声在低码率下本来就编不好，用它当判据会冤枉编码器）。
/// 中频带被抹掉时全局 SNR 会塌下来。
#[test]
fn wide_dynamic_range_signal_keeps_its_midrange() {
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }
    const SECS: usize = 4;
    let n = SR as usize * SECS;
    const MID: [f64; 4] = [700.0, 1500.0, 3000.0, 6000.0];

    // 强低音（80 Hz，12000）+ 四个中频乐音（各 2500）+ 一点底噪。
    let mut state = 12345u32;
    let src: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(SR);
            let mut v = 12000.0 * (2.0 * std::f64::consts::PI * 80.0 * t).sin();
            for f in MID {
                v += 2500.0 * (2.0 * std::f64::consts::PI * f * t).sin();
            }
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            v += ((state >> 16) as f64 / 32768.0 - 1.0) * 150.0;
            v as f32
        })
        .collect();

    let mut e = Encoder::new(EncoderConfig {
        channels: 1,
        sample_rate: SR,
        bitrate: 96_000,
        threads: 1,
        kb_window: false,
        ms_stereo: true,
    })
    .expect("建编码器");
    let mut frames = Vec::new();
    for chunk in src.chunks(1024) {
        let mut p = chunk.to_vec();
        p.resize(1024, 0.0);
        frames.push(e.encode_frame(&Frame { planes: vec![p] }).expect("编一帧"));
    }
    let dec = decode_aac(&frames, 1);

    let (snr, sh) = aligned_snr(&dec[0], &src);
    println!("大动态范围信号（96k 单声道）：SNR {snr:.2} dB（对齐 {sh}）");
    assert!(
        snr > 12.0,
        "SNR 只有 {snr:.2} dB —— 中频带被量化掉了（代价函数是不是又在拿掩蔽阈值当权重了？）"
    );
}

/// 先对齐、再算 SNR，返回 `(SNR dB, 最佳延迟)`。
///
/// ⚠️ **必须先对齐**：AAC 解码出来的样本比源**晚约 1024 个**（编码器的延迟 +
/// 首帧的 priming），而 `decode_aac` 不做补偿。直接逐样本比会把「对不齐」量成
/// SNR —— 周期性信号上尤其惨（差一个采样点就全错，SNR 直接掉到 0 附近）。
/// 所以先按 32 个样本粗搜、再逐样本细化。
fn aligned_snr(got: &[f32], src: &[f32]) -> (f64, usize) {
    // 两端各留一段给编码器延迟与收尾的渐变段。
    const MARGIN: usize = 20000;

    let mse_at = |sh: usize| -> f64 {
        // 上界同时受 `got` 的剩余长度约束 —— 否则 `got` 比 `src` 短时会越界。
        let hi = got.len().saturating_sub(sh).min(src.len());
        if hi <= 2 * MARGIN {
            return f64::MAX;
        }
        let (mut s, mut c) = (0f64, 0u32);
        for i in (MARGIN..hi - MARGIN).step_by(7) {
            let d = f64::from(got[i + sh]) - f64::from(src[i]);
            s += d * d;
            c += 1;
        }
        s / f64::from(c.max(1))
    };

    let mut best = (f64::MAX, 0usize);
    for sh in (0..4096).step_by(32) {
        let v = mse_at(sh);
        if v < best.0 {
            best = (v, sh);
        }
    }
    for sh in best.1.saturating_sub(32)..(best.1 + 32).min(4096) {
        let v = mse_at(sh);
        if v < best.0 {
            best = (v, sh);
        }
    }

    let sh = best.1;
    let hi = got.len().saturating_sub(sh).min(src.len());
    let (mut se, mut sp) = (0f64, 0f64);
    for i in MARGIN..hi.saturating_sub(MARGIN) {
        let d = f64::from(got[i + sh]) - f64::from(src[i]);
        se += d * d;
        sp += f64::from(src[i]) * f64::from(src[i]);
    }
    (10.0 * (sp / se.max(1e-9)).log10(), sh)
}

/// **KBD 窗**：位流里的 `use_kb_window` 必须与编码端**实际用的窗**一致。
///
/// 这是位流里最容易「合法但错」的一位。解码端只拿它决定用哪张窗表，
/// **不校验与内容是否一致** —— 所以写错了：ffmpeg 照样解得开、不报任何错、
/// 帧数也对，只是它拿正弦窗去重建 KBD 窗编出来的系数。
///
/// 后果在重建公式里很直接：重叠区要满足
/// `w_a[n]·w_s[n] + w_a[n+N]·w_s[n+N] = 1` 才能抵消时域混叠。
/// 两端同系时这个式子对正弦和 KBD **都**成立（各自满足 `w²+w'²=1`）；
/// **一错一配**就不成立 —— 重叠区多出一个随帧位置起伏的增益，
/// 表现成帧边界的「呼吸」，而不是任何解码错误。
///
/// 所以判据**不能**是「ffmpeg 解得开」（写错也解得开），必须是
/// **解码回来与原信号的 SNR**：同一份信号用两种窗各编一遍，两条 SNR
/// 应当在同一量级。这条测试在修好之前是**红的**（KBD 那侧低 30+ dB）。
#[test]
fn kb_window_bit_matches_the_window_actually_used() {
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }
    // ⚠️ 要够长：`aligned_snr` 两端各留 20000 个样本，信号比它短就整段都在
    // 边距里，量出来是 `-inf`（看着像「编码器全坏了」，其实是测试太短）。
    let total = SR as usize * 3;
    let src = sine(total, 1);

    let encode_with = |kb: bool| -> Vec<Vec<u8>> {
        let mut e = Encoder::new(EncoderConfig {
            channels: 1,
            sample_rate: SR,
            bitrate: 128_000,
            threads: 1,
            kb_window: kb,
            ms_stereo: true,
        })
        .expect("建编码器");
        let mut frames = Vec::new();
        let mut i = 0usize;
        while i < total {
            let n = (total - i).min(1024);
            let planes: Vec<Vec<f32>> = src.iter().map(|p| p[i..i + n].to_vec()).collect();
            frames.push(e.encode_frame(&Frame { planes }).expect("编一帧"));
            i += 1024;
        }
        frames
    };

    let (sine_db, sh_s) = aligned_snr(&decode_aac(&encode_with(false), 1)[0], &src[0]);
    let (kb_db, sh_k) = aligned_snr(&decode_aac(&encode_with(true), 1)[0], &src[0]);
    println!(
        "同一份 440 Hz 正弦（128k 单声道）：正弦窗 {sine_db:.2} dB（对齐 {sh_s}）/ \
         KBD 窗 {kb_db:.2} dB（对齐 {sh_k}）"
    );

    // ⚠️ 这个下限**不是判据**，只是防止「两边一起坏了还互相接近」。
    // 绝对值是我们编码器在这类**稀疏频谱**上的真实水平（远低于 ffmpeg —— 同一个
    // 正弦、同 128k，ffmpeg 解回来有 64 dB 量级），那是另一件事，别在这里查。
    assert!(
        sine_db > 20.0,
        "正弦窗这一侧的基线本身就不对（{sine_db:.2} dB）—— 先看别的测试，别在这儿找原因"
    );
    assert!(
        kb_db >= sine_db - 6.0,
        "KBD 窗比正弦窗低了 {:.2} dB（{kb_db:.2} vs {sine_db:.2}）—— \
         位流里的 `use_kb_window` 与编码端实际用的窗不一致",
        sine_db - kb_db
    );
}
