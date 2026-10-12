//! 转码工具的集成测试：**编一段 → 转 → 验输出**。
//!
//! 判据三层：
//! 1. 转出来的文件我们自己的解封装 + 解码器能读（结构合法）；
//! 2. 校验 PSNR 在合理范围（内容确实搬过去了，不是黑帧）；
//! 3. 命令行入口真能批量跑（拿 `CARGO_BIN_EXE_v2v` 跑真二进制）。

use std::path::{Path, PathBuf};
use std::process::Command;

use v2v_asf::{AsfDemuxer, AsfMuxer, StreamKind, VideoStreamInfo};
use v2v_cli::{convert_file, ConvertOptions, MP43};
use v2v_msmpeg4::{Decoder, Encoder, EncoderConfig, Frame};

/// 进程内唯一的临时目录。
fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("v2v_cli_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("建临时目录");
    d
}

/// 合成一段有运动的序列。
fn moving_sequence(w: u32, h: u32, n: usize) -> Vec<Frame> {
    let (wu, hu) = (w as usize, h as usize);
    let mut out = Vec::new();
    for i in 0..n {
        let mut y = vec![0u8; wu * hu];
        for r in 0..hu {
            for c in 0..wu {
                let v = 90 + (((c + i * 4) / 8) % 4) as i32 * 28 + ((r / 16) % 2) as i32 * 12;
                y[r * wu + c] = v.clamp(16, 235) as u8;
            }
        }
        let ox = 30 + i * 7;
        let oy = 24 + i * 4;
        for r in 0..36usize {
            for c in 0..36usize {
                let (yy, xx) = (oy + r, ox + c);
                if yy < hu && xx < wu {
                    y[yy * wu + xx] = if (r / 4 + c / 4) % 2 == 0 { 25 } else { 225 };
                }
            }
        }
        let (cw, ch) = (wu / 2, hu / 2);
        let mut u = vec![128u8; cw * ch];
        let mut v = vec![128u8; cw * ch];
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = (128 + (c / 8) as i32 * 5 - i as i32 * 2).clamp(16, 240) as u8;
                v[r * cw + c] = (128 - (r / 8) as i32 * 4 + i as i32 * 2).clamp(16, 240) as u8;
            }
        }
        out.push(Frame { width: w, height: h, y, u, v });
    }
    out
}

/// 用我们的编码器 + 封装器造一个输入 `.wmv`。
fn make_wmv(path: &Path, frames: &[Frame], fourcc: u32, qscale: u8) {
    let (w, h) = (frames[0].width, frames[0].height);
    let mut enc = Encoder::new(EncoderConfig {
        qscale,
        ..Default::default()
    });
    let mut mux = AsfMuxer::new(VideoStreamInfo {
        width: w,
        height: h,
        fourcc,
    });
    for (i, f) in frames.iter().enumerate() {
        let s = enc.encode_frame(f).expect("编码");
        mux.write_video(&s, i == 0, (i as u32) * 33).expect("封装");
    }
    std::fs::write(path, mux.finish()).expect("写输入文件");
}

#[test]
fn converts_and_output_is_readable() {
    let dir = temp_dir("basic");
    let (w, h) = (320u32, 240u32);
    let frames = moving_sequence(w, h, 8);
    let input = dir.join("in.wmv");
    let output = dir.join("out.wmv");
    make_wmv(&input, &frames, MP43, 4);

    let opts = ConvertOptions {
        qscale: 6,
        keyframe_interval: 4, // 8 帧 → 关键帧在 0 与 4
        verify: true,
        max_frames: None,
        copy_audio: true,
    };
    let rep = convert_file(&input, &output, &opts).expect("转换");

    eprintln!("报告：{rep:?}");
    assert_eq!(rep.frames, 8);
    assert_eq!(rep.key_frames, 2, "GOP=4 时 8 帧应有 2 个关键帧");
    assert!(output.is_file(), "输出文件没生成");
    let psnr = rep.psnr_db.expect("校验开着就该有 PSNR");
    assert!(psnr > 25.0, "转出来的画质太差（PSNR {psnr:.2} dB）");

    // ---- 输出能被我们自己的解封装读回，且关键帧位置对 ----
    let mut d = AsfDemuxer::new(std::fs::File::open(&output).expect("开输出")).expect("解封装输出");
    let s = &d.streams()[0];
    assert_eq!(s.kind, StreamKind::Video);
    assert_eq!(s.video.as_ref().unwrap().width, w);
    assert_eq!(s.video.as_ref().unwrap().height, h);
    let mut keys = Vec::new();
    let mut n = 0usize;
    while let Some(sm) = d.next_sample().expect("读样本") {
        if sm.key_frame {
            keys.push(n);
        }
        n += 1;
    }
    assert_eq!(n, 8, "帧数不对");
    assert_eq!(keys, vec![0, 4], "关键帧位置不对");

    // ---- 输出能被解码器解（内容不是垃圾）----
    let mut d = AsfDemuxer::new(std::fs::File::open(&output).expect("开输出")).expect("解封装");
    let mut dec = Decoder::new();
    let mut decoded = 0usize;
    while let Some(sm) = d.next_sample().expect("读样本") {
        let f = dec.decode(&sm.data, w, h).expect("解码输出");
        assert_eq!(f.y.len(), (w * h) as usize);
        decoded += 1;
    }
    assert_eq!(decoded, 8);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rejects_non_mp43_input() {
    let dir = temp_dir("reject");
    let frames = moving_sequence(160, 120, 2);
    // 故意标成 WMV1：转码器必须明确报「不支持」，不能静默按 MP43 解。
    let input = dir.join("wmv1.wmv");
    make_wmv(&input, &frames, u32::from_le_bytes(*b"WMV1"), 8);
    let out = dir.join("out.wmv");
    let err = convert_file(&input, &out, &ConvertOptions::default()).expect_err("应当报不支持");
    let msg = err.to_string();
    eprintln!("报错：{msg}");
    assert!(msg.contains("WMV1"), "报错里应点名实际格式：{msg}");
    assert!(!out.exists(), "报错时不该留下输出文件");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_batch_converts_two_files() {
    let dir = temp_dir("batch");
    let frames = moving_sequence(192, 144, 4);
    let a = dir.join("a.wmv");
    let b = dir.join("b.wmv");
    make_wmv(&a, &frames, MP43, 4);
    make_wmv(&b, &frames, MP43, 8);
    let out_dir = dir.join("out");

    let exe = env!("CARGO_BIN_EXE_v2v");
    let status = Command::new(exe)
        .args([
            "convert",
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            "-o",
            out_dir.to_str().unwrap(),
            "-q",
            "7",
            "--no-verify",
        ])
        .status()
        .expect("跑 v2v 二进制");
    assert!(status.success(), "批量转换应当成功：{status}");

    for name in ["a.wmv", "b.wmv"] {
        let p = out_dir.join(name);
        assert!(p.is_file(), "{} 没生成", p.display());
        let mut d = AsfDemuxer::new(std::fs::File::open(&p).expect("开")).expect("解封装");
        let mut n = 0usize;
        while d.next_sample().expect("读").is_some() {
            n += 1;
        }
        assert_eq!(n, 4, "{} 帧数不对", p.display());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_reports_failure_and_exit_code() {
    let dir = temp_dir("badinput");
    let bad = dir.join("bad.wmv");
    std::fs::write(&bad, b"this is not an asf file at all, just text.........").expect("写垃圾");
    let exe = env!("CARGO_BIN_EXE_v2v");
    let out = Command::new(exe)
        .args(["convert", bad.to_str().unwrap(), "--no-verify"])
        .output()
        .expect("跑 v2v");
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(!out.status.success(), "坏输入应当以非零退出码结束");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("失败"),
        "应当有失败摘要"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// 取 ffmpeg 解出的裸 YUV420P（每帧 `w*h*3/2` 字节）。
fn ffmpeg_yuv(path: &Path, count: usize, w: u32, h: u32) -> Option<Vec<u8>> {
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-idct",
            "2",
            "-i",
            path.to_str()?,
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
    // 只取前 `count` 帧（多解的丢掉，保证调用方的步长假设成立）。
    let want = count * (w * h * 3 / 2) as usize;
    let mut v = out.stdout;
    v.truncate(want);
    Some(v)
}

/// **真样本端到端**（默认 `#[ignore]`）：真 `.wmv` 转 N 帧 → 输出交 ffmpeg 解 →
/// 与我们的解码器**逐位**比。这是「转出来的文件是标准 WMV」的最终判据。
///
/// ```text
/// V2V_ASF_SAMPLE=<样本> cargo test --release -p v2v-cli -- --ignored --nocapture
/// ```
#[test]
#[ignore = "需要真样本与 ffmpeg，1080p 下较慢"]
fn converts_real_sample_and_ffmpeg_agrees() {
    let src = match std::env::var("V2V_ASF_SAMPLE") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SKIP: V2V_ASF_SAMPLE 未设置");
            return;
        }
    };
    if !Path::new(&src).is_file() {
        eprintln!("SKIP: 样本不存在");
        return;
    }
    let dir = temp_dir("real");
    // 换个中性文件名 —— 样本原名不该出现在临时产物里。
    let input = dir.join("in.wmv");
    std::fs::copy(&src, &input).expect("拷样本");
    let output = dir.join("out.wmv");

    let n = 12usize;
    let opts = ConvertOptions {
        qscale: 6,
        keyframe_interval: 0,
        verify: true,
        max_frames: Some(n),
        copy_audio: true,
    };
    let rep = convert_file(&input, &output, &opts).expect("转真样本");
    eprintln!("真样本：{rep:?}");
    assert_eq!(rep.frames, n);
    assert!(rep.psnr_db.unwrap() > 30.0, "真内容画质太差");

    // ---- 我们解输出 ----
    let mut d = AsfDemuxer::new(std::fs::File::open(&output).expect("开")).expect("解输出");
    let (vnum, w, h) = {
        let v = d
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Video)
            .expect("视频轨");
        let vf = v.video.as_ref().expect("视频格式");
        (v.number, vf.width, vf.height)
    };
    let mut dec = Decoder::new();
    let mut ours = Vec::new();
    while let Some(s) = d.next_sample().expect("读") {
        if s.stream == vnum {
            ours.push(dec.decode(&s.data, w, h).expect("解"));
        }
    }
    assert_eq!(ours.len(), n, "输出帧数不对");

    // ---- ffmpeg 解输出，逐帧逐位比 ----
    let raw = match ffmpeg_yuv(&output, n, w, h) {
        Some(v) => v,
        None => {
            eprintln!("SKIP: 没有可用的 ffmpeg");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
    };
    let fs = (w * h * 3 / 2) as usize;
    assert_eq!(raw.len(), fs * n, "ffmpeg 解出的字节数不对");
    for i in 0..n {
        let r = &raw[i * fs..(i + 1) * fs];
        let (wi, hi) = (w as usize, h as usize);
        let (ysz, csz) = (wi * hi, wi / 2 * (hi / 2));
        for (name, a, b) in [
            ("Y", &ours[i].y[..], &r[..ysz]),
            ("U", &ours[i].u[..], &r[ysz..ysz + csz]),
            ("V", &ours[i].v[..], &r[ysz + csz..]),
        ] {
            for (k, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x, y,
                    "帧 {i} {name} 第 {k} 个像素不一致：我们 {x}，ffmpeg {y}"
                );
            }
        }
    }
    eprintln!("真样本闭环：{n} 帧逐位一致（{w}×{h}）");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------- AVI 输入 ----------

/// 把若干帧裸 MP43 包成一个最小 AVI（自包含：不依赖 ffmpeg）。
/// 结构：`RIFF/AVI ` + `LIST hdrl`（avih + strl{strh, strf}）+ `LIST movi`（`00dc` 块）+ `idx1`。
fn wrap_avi(frames: &[&[u8]], w: u32, h: u32, key_first: bool) -> Vec<u8> {
    fn u32le(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn u16le(v: &mut Vec<u8>, x: u16) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(id);
        u32le(&mut v, body.len() as u32);
        v.extend_from_slice(body);
        if body.len() % 2 == 1 {
            v.push(0);
        }
        v
    }
    fn list(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut inner = Vec::new();
        inner.extend_from_slice(kind);
        inner.extend_from_slice(body);
        chunk(b"LIST", &inner)
    }

    let mut avih = Vec::new();
    u32le(&mut avih, 33333);
    u32le(&mut avih, 0);
    u32le(&mut avih, 0);
    u32le(&mut avih, 0x10);
    u32le(&mut avih, frames.len() as u32);
    u32le(&mut avih, 0);
    u32le(&mut avih, 1);
    u32le(&mut avih, 0);
    u32le(&mut avih, w);
    u32le(&mut avih, h);
    for _ in 0..4 {
        u32le(&mut avih, 0);
    }

    let mut strh = Vec::new();
    strh.extend_from_slice(b"vids");
    strh.extend_from_slice(b"MP43");
    u32le(&mut strh, 0);
    u16le(&mut strh, 0);
    u16le(&mut strh, 0);
    u32le(&mut strh, 0);
    u32le(&mut strh, 1); // scale
    u32le(&mut strh, 30); // rate
    u32le(&mut strh, 0);
    u32le(&mut strh, frames.len() as u32);
    u32le(&mut strh, 0);
    u32le(&mut strh, 0xFFFF_FFFF);
    u32le(&mut strh, 0);
    for _ in 0..4 {
        u16le(&mut strh, 0);
    }

    let mut strf = Vec::new();
    u32le(&mut strf, 40);
    u32le(&mut strf, w);
    u32le(&mut strf, h);
    u16le(&mut strf, 1);
    u16le(&mut strf, 24);
    strf.extend_from_slice(b"MP43");
    u32le(&mut strf, w * h * 3);
    for _ in 0..4 {
        u32le(&mut strf, 0);
    }

    let strl = list(b"strl", &[chunk(b"strh", &strh), chunk(b"strf", &strf)].concat());
    let hdrl = list(b"hdrl", &[chunk(b"avih", &avih), strl].concat());

    let mut movi_body = Vec::new();
    let mut offsets = Vec::new();
    for f in frames {
        offsets.push(movi_body.len() as u32);
        movi_body.extend_from_slice(&chunk(b"00dc", f));
    }
    let movi = list(b"movi", &movi_body);

    let mut idx1 = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        idx1.extend_from_slice(b"00dc");
        u32le(&mut idx1, if i == 0 && key_first { 0x10 } else { 0 });
        // 偏移相对 `movi` 四字（= 列表数据体 - 4），与 ffmpeg 的写法一致
        u32le(&mut idx1, offsets[i] + 4);
        u32le(&mut idx1, f.len() as u32);
    }
    let idx1 = chunk(b"idx1", &idx1);

    let body = [hdrl, movi, idx1].concat();
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    u32le(&mut out, (body.len() + 4) as u32);
    out.extend_from_slice(b"AVI ");
    out.extend_from_slice(&body);
    out
}

/// **AVI 输入**：把我们自己编的 MP43 帧包成 AVI，转成 WMV —— 输出要能被自己读回、
/// 帧数与内容都对得上。容器按**魔数**自动认（不看扩展名）。
#[test]
fn converts_avi_input() {
    let dir = temp_dir("avi_in");
    let (w, h) = (160u32, 128u32);
    let frames = moving_sequence(w, h, 5);

    // 先编出裸 MP43 帧
    let mut enc = Encoder::new(EncoderConfig {
        qscale: 4,
        ..Default::default()
    });
    let mut streams = Vec::new();
    for f in &frames {
        streams.push(enc.encode_frame(f).expect("编码"));
    }

    // 包成 AVI，故意用 `.bin` 后缀 —— 逼着它按魔数认容器
    let avi = wrap_avi(
        &streams.iter().map(|s| s.as_slice()).collect::<Vec<_>>(),
        w,
        h,
        true,
    );
    let input = dir.join("input.bin");
    std::fs::write(&input, &avi).expect("写 AVI");

    let output = dir.join("out.wmv");
    let opts = ConvertOptions {
        qscale: 6,
        keyframe_interval: 0,
        verify: true,
        max_frames: None,
        copy_audio: true,
    };
    let rep = convert_file(&input, &output, &opts).expect("转 AVI");
    eprintln!("AVI → WMV：{rep:?}");
    assert_eq!(rep.frames, 5, "AVI 里的帧数不对");
    assert_eq!(rep.key_frames, 1, "只有首帧是关键帧");
    assert!(rep.psnr_db.unwrap() > 25.0, "画质太差");

    // 输出仍是我们自己的 ASF，能读回
    let mut d = AsfDemuxer::new(std::fs::File::open(&output).expect("开")).expect("解输出");
    let mut n = 0usize;
    while d.next_sample().expect("读").is_some() {
        n += 1;
    }
    assert_eq!(n, 5);
    let _ = std::fs::remove_dir_all(&dir);
}

/// **音频直通**（真样本，`#[ignore]`）：转出来的文件要带音频，且音频载荷**逐字节**等于源的。
///
/// ```text
/// V2V_AUDIO_SAMPLE=<带 WMA 音频 + MP43 视频的 .wmv/.asf> cargo test --release -p v2v-cli -- --ignored --nocapture
/// ```
#[test]
#[ignore = "需要真样本（带 WMA 音轨 + MP43 视频）"]
fn audio_is_passed_through_bit_exactly() {
    let src = match std::env::var("V2V_AUDIO_SAMPLE") {
        Ok(p) if Path::new(&p).is_file() => p,
        _ => {
            eprintln!("SKIP: V2V_AUDIO_SAMPLE 未设置或文件不存在");
            return;
        }
    };
    let dir = temp_dir("audio");
    let input = dir.join("in.wmv");
    std::fs::copy(&src, &input).expect("拷样本");
    let output = dir.join("out.wmv");

    // 只转 60 帧（够验证直通，且省时间）
    let opts = ConvertOptions {
        qscale: 6,
        keyframe_interval: 0,
        verify: false,
        max_frames: Some(60),
        copy_audio: true,
    };
    let rep = convert_file(&input, &output, &opts).expect("转换");
    eprintln!("报告：{rep:?}");
    assert_eq!(rep.frames, 60);
    assert!(rep.audio_payloads > 0, "一个音频载荷都没搬");

    // ---- 输出里得有两条流，音频格式与源一致 ----
    let mut d = AsfDemuxer::new(std::fs::File::open(&output).expect("开")).expect("解输出");
    let out_audio = d
        .streams()
        .iter()
        .find(|s| s.kind == StreamKind::Audio)
        .expect("输出没有音频轨");
    let out_af = out_audio.audio.clone().expect("音频格式");
    let out_num = out_audio.number;

    let mut src_d = AsfDemuxer::new(std::fs::File::open(&input).expect("开")).expect("解源");
    let src_audio = src_d
        .streams()
        .iter()
        .find(|s| s.kind == StreamKind::Audio)
        .expect("源没有音频轨");
    let src_af = src_audio.audio.clone().expect("源音频格式");
    let src_num = src_audio.number;
    assert_eq!(out_af.format_tag, src_af.format_tag);
    assert_eq!(out_af.sample_rate, src_af.sample_rate);
    assert_eq!(out_af.channels, src_af.channels);
    assert_eq!(out_af.block_align, src_af.block_align);

    // ---- 音频载荷逐字节比（重叠的那一段）----
    let mut out_payloads = Vec::new();
    while let Some(s) = d.next_sample().expect("读输出") {
        if s.stream == out_num {
            out_payloads.push(s.data);
        }
    }
    let mut src_payloads = Vec::new();
    while let Some(s) = src_d.next_sample().expect("读源") {
        if s.stream == src_num {
            src_payloads.push(s.data);
        }
    }
    eprintln!(
        "音频载荷：输出 {} 个 / 源 {} 个",
        out_payloads.len(),
        src_payloads.len()
    );
    assert!(
        out_payloads.len() <= src_payloads.len(),
        "输出的音频载荷比源还多？"
    );
    let n = out_payloads.len().saturating_sub(1); // 最后一个可能被截断
    for i in 0..n {
        assert_eq!(
            out_payloads[i], src_payloads[i],
            "第 {i} 个音频载荷不一致 —— 直通没做到逐字节相同"
        );
    }
    eprintln!("前 {n} 个音频载荷逐字节一致 ✓");
    let _ = std::fs::remove_dir_all(&dir);
}
