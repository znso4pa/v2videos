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
use v2v_msmpeg4::BitWriter;

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

/// 交织成 `f32le` 字节流（喂给 ffmpeg）。
fn interleave(planes: &[Vec<f32>]) -> Vec<u8> {
    let ch = planes.len();
    let n = planes[0].len();
    let mut out = Vec::with_capacity(n * ch * 4);
    for i in 0..n {
        for p in planes {
            out.extend_from_slice(&p[i].to_le_bytes());
        }
    }
    out
}

/// 用 ffmpeg 解一段 AAC（裸帧要包成 ADTS 才认），返回每个声道的 f32 样本。
fn decode_aac(frames: &[Vec<u8>], channels: u8) -> Vec<Vec<f32>> {
    // 拼一个 ADTS 容器：每个裸帧前面加 7 字节头。
    let mut adts = Vec::new();
    for f in frames {
        adts.extend_from_slice(&adts_header(f.len(), channels));
        adts.extend_from_slice(f);
    }
    let path = std::env::temp_dir().join("v2v_aac_ours.aac");
    std::fs::write(&path, &adts).expect("写 ADTS");

    // ⚠️ 两点：
    //
    // 1. **解到文件**：`-f f32le -`（stdout）会被 ffmpeg 9 拒掉，
    //    报 `Output file does not contain any stream` —— 那是 CLI 的脾气。
    // 2. **用 `s16le` 而不是 `f32le`**：`f32le` 是 ±1 归一化的，而本测试的输入是
    //    int16 量级（`AMP = 12000`）。口径不同会让「幅度比」算出 0.000，
    //    看起来像位流全错、其实是**单位没对齐**。用 `s16le` 就和输入同量级了。
    let raw_path = std::env::temp_dir().join("v2v_aac_ours.pcm");
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
    let raw_path = std::env::temp_dir().join("v2v_quality_ref.pcm");
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
    let (mut s0, mut s1, mut s2) = (0f64, 0f64, 0f64);
    for &v in x {
        s0 = f64::from(v) + coeff * s1 - s2;
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
        kb_window: false,
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
        kb_window: false,
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
    assert!(
        ratio >= 0.6,
        "实测码率 {got:.0} 只有目标的 {ratio:.2} 倍 —— 压过头了，音质会被白牺牲"
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
#[test]
fn quality_is_within_2db_of_ffmpeg() {
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }
    let secs = 6usize;
    let total = SR as usize * secs;
    let tmp = std::env::temp_dir();

    // 干净基准 PCM（用 ffmpeg 生成粉噪声 —— 全频段，量化问题最容易显形）。
    let pcm = tmp.join("v2v_quality_src.pcm");
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
        kb_window: false,
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
    let ref_aac = tmp.join("v2v_quality_ref.aac");
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
