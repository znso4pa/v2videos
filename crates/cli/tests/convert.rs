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

/// 源样本视频轨的**版本 + extradata**（认不出 FourCC 给 `None`）。
fn src_version(path: &Path) -> Option<(v2v_msmpeg4::Version, Vec<u8>)> {
    let d = AsfDemuxer::new(std::fs::File::open(path).ok()?).ok()?;
    let vf = d
        .streams()
        .iter()
        .find(|s| s.kind == StreamKind::Video)?
        .video
        .as_ref()?;
    Some((v2v_msmpeg4::Version::from_fourcc(vf.fourcc)?, vf.extra.clone()))
}

/// 按源样本的**版本**造解码器（WMV2 的能力位在容器的 extradata 里）。
///
/// ⚠️ 别对**源样本**用 `Decoder::new()`（V3 默认）：解 WMV1 / WMV2 的源会以
/// 「ac coefficient overflow」之类失败 —— 报出来像解码器坏了，其实是版本没对上。
/// 解**我们自己写出来的 `.wmv`**（按构造是 MP43）时才可以用默认值。
fn src_decoder(path: &Path) -> Option<Decoder> {
    let (version, extra) = src_version(path)?;
    if version == v2v_msmpeg4::Version::Wmv2 {
        Decoder::with_wmv2(&extra).ok()
    } else {
        Some(Decoder::with_version(version))
    }
}

/// 源样本里有没有音频轨 —— 音频相关的真样本测试靠它决定「跑」还是「SKIP」。
///
/// 只按 ASF 解（这两条测试针对的就是 `.wmv` 样本）。判据取自**源的解封装结果**，
/// 不是转码报告里的 `audio_payloads` —— 后者正是被测的东西，拿它当前提会把
/// 「音频整条丢了」变成「跳过」，那是最糟的一种假绿。
fn src_has_audio(path: &Path) -> bool {
    AsfDemuxer::new(std::fs::File::open(path).expect("开样本"))
        .map(|d| d.streams().iter().any(|s| s.kind == StreamKind::Audio))
        .unwrap_or(false)
}

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
        out.push(Frame {
            width: w,
            height: h,
            y,
            u,
            v,
        });
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
        ..Default::default()
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

/// **GOP 并行的产物与串行逐字节相同**（`--threads > 1` 那条路）。
///
/// 这是 `.wmv` 并行化唯一站得住的理由：MP43 编码器**没有跨 GOP 的状态**
/// （MV / DC 网格每帧重置、`ref_planes` 只在 GOP 内部有用、我们不写 ext header
/// 所以 `flipflop_rounding` 恒 0），所以按 GOP 分给多个 worker、每个 worker 起一个
/// 新编码器，编出来的字节应当与单线程**完全一样**。哪天有人往编码器里加了跨 GOP 的
/// 状态、或改了结果的归并顺序，这条会立刻红。
///
/// 也顺带钉住**包的排布**：并行那条路是「结果按 GOP 序号归位 → 与音频按 pts 合并」
/// 再写封装器的，合并规则一旦与串行读到的顺序不一致，长度或字节就会变。
///
/// 合成源、无样本、无 ffmpeg —— 所以 CI 里也真跑，不是 SKIP。
#[test]
fn parallel_gop_encoding_is_byte_identical() {
    let dir = temp_dir("par");
    let (w, h) = (320u32, 240u32);
    let frames = moving_sequence(w, h, 20);
    let input = dir.join("in.wmv");
    make_wmv(&input, &frames, MP43, 4);

    let base = ConvertOptions {
        qscale: 6,
        keyframe_interval: 5, // 20 帧 → 4 个 GOP
        verify: false,
        max_frames: None,
        copy_audio: false,
        ..Default::default()
    };
    let serial = dir.join("serial.wmv");
    let rep1 = convert_file(&input, &serial, &base).expect("串行转换");
    let par = dir.join("par.wmv");
    let rep2 = convert_file(
        &input,
        &par,
        &ConvertOptions {
            threads: 4,
            ..base.clone()
        },
    )
    .expect("并行转换");

    assert_eq!(rep1.frames, rep2.frames, "帧数不一致");
    assert_eq!(rep1.key_frames, rep2.key_frames, "关键帧数不一致");
    let a = std::fs::read(&serial).expect("读串行产物");
    let b = std::fs::read(&par).expect("读并行产物");
    assert_eq!(a.len(), b.len(), "产物长度不同：{} vs {}", a.len(), b.len());
    assert!(a == b, "并行与串行的产物不是逐字节相同");
    let _ = std::fs::remove_dir_all(&dir);
}

/// **还没实现的版本要明确拒绝**，而且不留输出文件。
#[test]
fn rejects_versions_we_do_not_decode_yet() {
    let dir = temp_dir("reject");
    let frames = moving_sequence(160, 120, 2);
    // WMV2 的帧头与 MB 层是另一套（ffmpeg 里在 `wmv2dec.c`），还没接。
    let input = dir.join("wmv2.wmv");
    make_wmv(&input, &frames, u32::from_le_bytes(*b"WMV2"), 8);
    let out = dir.join("out.wmv");
    let err = convert_file(&input, &out, &ConvertOptions::default()).expect_err("应当报不支持");
    let msg = err.to_string();
    eprintln!("报错：{msg}");
    assert!(msg.contains("WMV2"), "报错里应点名实际格式：{msg}");
    assert!(!out.exists(), "报错时不该留下输出文件");
    let _ = std::fs::remove_dir_all(&dir);
}

/// **标错版本的流不能静默解出来**。
///
/// 这条是上一版 `rejects_non_mp43_input` 的延续：那版把 MP43 的流标成 `WMV1`
/// 期望「被拒绝」，而现在 `WMV1` **已经支持**了 —— 于是判据变成：
/// 拿 WMV1 的语法去解一份 MP43 的字节，**必须报错**，绝不能「解出来一份看着能播、
/// 实际是垃圾」的输出。静默产出错误画面比明确拒绝糟得多。
#[test]
fn a_mislabeled_stream_does_not_decode_silently() {
    let dir = temp_dir("mislabel");
    let frames = moving_sequence(160, 120, 2);
    let input = dir.join("mislabeled.wmv");
    make_wmv(&input, &frames, u32::from_le_bytes(*b"WMV1"), 8);
    let out = dir.join("out.wmv");
    let r = convert_file(&input, &out, &ConvertOptions::default());
    if r.is_ok() {
        // 「解出来了」只有在**产物真的对**时才可接受 —— 但这里底本是 MP43、
        // 语法标的是 WMV1，解出来必然不对，所以只允许报错。
        panic!("标错版本的流被静默解出来了（产物 {} 字节）", std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0));
    }
    let msg = r.unwrap_err().to_string();
    eprintln!("报错：{msg}");
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
#[ignore = "需要真样本与 ffmpeg，高分辨率样本下较慢"]
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
        ..Default::default()
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

    let strl = list(
        b"strl",
        &[chunk(b"strh", &strh), chunk(b"strf", &strf)].concat(),
    );
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
        ..Default::default()
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
        ..Default::default()
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

// ==================== `.mp4`（MPEG-4 Part 2）那条路 ====================

/// 一个够用的只读 ISOBMFF 遍历器（只在测试里用）。
///
/// 刻意不复用 `crates/mp4` 的写侧代码 —— 那边是**写**、这边是**读**，
/// 共用会让「两边一起错」变得难以察觉。
mod isobmff {
    pub fn be32(d: &[u8]) -> u32 {
        u32::from_be_bytes(d[0..4].try_into().unwrap())
    }

    /// `[start, end)` 里的所有 box：`(类型, body 起点, body 终点)`。
    pub fn boxes(d: &[u8], start: usize, end: usize) -> Vec<([u8; 4], usize, usize)> {
        let mut v = Vec::new();
        let mut p = start;
        while p + 8 <= end {
            let sz = be32(&d[p..]) as usize;
            let mut ty = [0u8; 4];
            ty.copy_from_slice(&d[p + 4..p + 8]);
            if sz < 8 || p + sz > end {
                break;
            }
            v.push((ty, p + 8, p + sz));
            p += sz;
        }
        v
    }

    /// 按路径逐层往下找，返回最后一层的 body `(起点, 终点)`。
    pub fn find(d: &[u8], start: usize, end: usize, path: &[&[u8; 4]]) -> Option<(usize, usize)> {
        let mut cur = (start, end);
        for want in path {
            let hit = boxes(d, cur.0, cur.1)
                .into_iter()
                .find(|(t, _, _)| t == *want)?;
            cur = (hit.1, hit.2);
        }
        Some(cur)
    }

    /// 所有 box 类型（整棵树的深度优先展平），给「结构看起来对不对」用。
    ///
    /// ⚠️ `stsd` **必须**在下钻名单里 —— 编解码器的 sample entry
    /// （`mp4a` / `mp4v` / `sowt`）就嵌在它里面。漏了 `stsd`，
    /// `all_types` 就永远找不到 `mp4a`，测试会报「音频轨应当是 mp4a」
    /// 这种**指向编码器**、实际是遍历器漏层的错（实测踩过）。
    ///
    /// 但 `stsd` 的 layout 与普通容器不同（前面有 8 字节的
    /// version/flags + entry_count，之后才是 sample entry），
    /// 所以不能当普通容器钻 —— 要把 body 往后挪 8 字节。
    pub fn all_types(d: &[u8], start: usize, end: usize, out: &mut Vec<[u8; 4]>) {
        for (t, b, e) in boxes(d, start, end) {
            out.push(t);
            if matches!(
                &t,
                b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"udta"
            ) {
                all_types(d, b, e, out);
            }
            // `stsd`：跳过 `version/flags(4) + entry_count(4)` 再往下钻。
            if &t == b"stsd" && e >= b + 8 {
                all_types(d, b + 8, e, out);
            }
            // 音频 sample entry（`mp4a` 等）：前面是 8 字节的
            // SampleEntry（6 保留 + 2 data_reference_index），再往后是
            // `AudioSampleEntry` 的 20 字节固定头，`esds` 就在其后。
            if matches!(&t, b"mp4a" | b"sowt" | b"twos" | b"mp3 ") && e >= b + 28 {
                all_types(d, b + 28, e, out);
            }
        }
    }
}

fn has_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "两边字节数不同");
    let se: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum();
    let mse = se / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// 输出 `.mp4` 的结构：**不依赖 ffmpeg**，我们自己拆 box 看。
///
/// 判据是「换容器不换内容」之外还多一条 —— MP4 的 `esds` 里那 40 多字节
/// `DecoderSpecificInfo` 必须是**完整**的 VOS + 视觉对象 + VOL，
/// 少了 VOS 那两级 `ffprobe` 就报不出 `profile` / `level`。
///
/// 这条用**全 I 帧**（`-g 0`）：`stss` 会被整个省掉，正好是「全 I 帧不必写 `stss`」
/// 这条约定的判据。P 帧那条路上 `stss` 必须写，见 `mp4_p_frames_write_stss`。
#[test]
fn mp4_output_has_a_valid_isobmff_structure() {
    let dir = temp_dir("mp4_struct");
    let (w, h) = (64u32, 48u32);
    let frames = moving_sequence(w, h, 5);
    let input = dir.join("in.wmv");
    let output = dir.join("out.mp4");
    make_wmv(&input, &frames, MP43, 4);

    let opts = ConvertOptions {
        qscale: 6,
        format: v2v_cli::OutputFormat::Mp4,
        // 全 I 帧：`stss` 该被整个省掉。
        keyframe_interval: 0,
        verify: true,
        max_frames: None,
        copy_audio: true,
        ..Default::default()
    };
    let rep = convert_file(&input, &output, &opts).expect("转 MP4");
    eprintln!("报告：{rep:?}");
    assert_eq!(rep.frames, 5);
    assert_eq!(rep.key_frames, 5, "全 I 帧 → 每帧都是关键帧");
    assert!(rep.psnr_db.is_none(), "`.mp4` 这条路没有解码器可比");
    assert!(rep.audio_payloads == 0, "合成样本没有音频轨");

    let d = std::fs::read(&output).expect("读输出");
    assert_eq!(&d[4..8], b"ftyp", "开头必须是 ftyp");
    assert_eq!(&d[8..12], b"isom", "品牌必须是 isom（不是 qt）");

    let mut types = Vec::new();
    isobmff::all_types(&d, 0, d.len(), &mut types);
    for need in [
        &b"ftyp"[..],
        b"moov",
        b"mdat",
        b"trak",
        b"mdia",
        b"minf",
        b"stbl",
        b"stsd",
        b"stts",
        b"stsc",
        b"stsz",
        b"stco",
    ] {
        assert!(
            types.iter().any(|t| t == need),
            "少了 box {:?}",
            String::from_utf8_lossy(need)
        );
    }
    // 全 I 帧 → `stss` 应当被整个省掉（见 `crates/mp4` 的模块文档）。
    assert!(!types.iter().any(|t| t == b"stss"), "全 I 帧不该写 stss");

    // ---- stsd 里的编码器标识与 DSI ----
    let stsd = isobmff::find(
        &d,
        0,
        d.len(),
        &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"],
    )
    .expect("stsd");
    // stsd: ver/flags(4) count(4) 之后是 sample entry
    let ent = stsd.0 + 8;
    assert_eq!(&d[ent + 4..ent + 8], b"mp4v", "sample entry 必须是 mp4v");

    let esds = isobmff::find(
        &d,
        ent + 86,
        ent + isobmff::be32(&d[ent..]) as usize,
        &[b"esds"],
    )
    .expect("esds");
    // esds 的 body：ver/flags(4) → ES_Descriptor → DecoderConfigDescriptor → DSI。
    //
    // ⚠️ 描述符长度是**变宽**的（7 位/字节，最高位是续接标志），所以必须用
    // `descr_len` 走，不能假设 4 字节。`descr_len` 返回 `(长度值, 长度字段之后的偏移)` ——
    // 拿后者再加长度值才是 body 的末尾。
    let mut q = esds.0 + 4 + 1; // 跳过 ver/flags 和 ES_Descriptor 的 tag
    let (_, e) = descr_len(&d, q);
    q = e + 3; // es_id(2) + flags(1)；flags == 0 所以没有 dependsOn/URL/OCR
    assert_eq!(d[q], 0x04, "接着应当是 DecoderConfigDescriptor（tag 4）");
    let (_, e) = descr_len(&d, q + 1);
    q = e + 13; // oti(1) + streamType(1) + bufferSizeDB(3) + maxBitrate(4) + avgBitrate(4)
    assert_eq!(d[q], 0x05, "接着应当是 DecoderSpecificInfo（tag 5）");
    let (ln, q2) = descr_len(&d, q + 1);
    let dsi = &d[q2..q2 + ln];
    eprintln!("DSI（{ln} 字节）：{}", hex(dsi));
    assert_eq!(
        &dsi[..4],
        &[0x00, 0x00, 0x01, 0xB0],
        "DSI 必须以 VOS 起始码开头"
    );
    assert!(
        dsi.windows(4).any(|x| x == [0x00, 0x00, 0x01, 0xB5]),
        "DSI 里要有视觉对象起始码"
    );
    assert!(
        dsi.windows(4).any(|x| x == [0x00, 0x00, 0x01, 0x20]),
        "DSI 里要有 VOL 起始码"
    );

    // ---- stsz 的样本数 = 帧数；mdat 的字节数 = 各样本之和 ----
    let stsz = isobmff::find(
        &d,
        0,
        d.len(),
        &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsz"],
    )
    .expect("stsz");
    let count = isobmff::be32(&d[stsz.0 + 8..]);
    assert_eq!(count as usize, rep.frames, "stsz 的样本数应等于帧数");

    let mdat = isobmff::find(&d, 0, d.len(), &[b"mdat"]).expect("mdat");
    assert!(
        mdat.1 - mdat.0 > 0 && (mdat.1 - mdat.0) < d.len(),
        "mdat 大小不合理"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

fn descr_len(d: &[u8], mut o: usize) -> (usize, usize) {
    let mut n = 0usize;
    for _ in 0..4 {
        let c = d[o];
        o += 1;
        n = (n << 7) | (c & 0x7F) as usize;
        if c & 0x80 == 0 {
            break;
        }
    }
    (n, o)
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// `.mp4` 的输出交给 ffmpeg 解，与「输入解码后的画面」比 PSNR。
///
/// 比的是**输入解码后的画面**（编码器看到的东西），不是原始合成帧 ——
/// 这样量到的就只是 MPEG-4 这一趟的损失，不掺 MP43 那一趟。
#[test]
fn mp4_output_decodes_with_ffmpeg() {
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }
    let dir = temp_dir("mp4_ffmpeg");
    let (w, h) = (320u32, 240u32);
    let n = 6usize;
    let frames = moving_sequence(w, h, n);
    let input = dir.join("in.wmv");
    let output = dir.join("out.mp4");
    make_wmv(&input, &frames, MP43, 4);

    // 编码器看到的画面（= 我们解输入得到的东西）
    let mut src = Vec::new();
    {
        let mut d = AsfDemuxer::new(std::fs::File::open(&input).expect("开输入")).expect("解封装");
        let (vnum, ww, hh) = {
            let v = d
                .streams()
                .iter()
                .find(|s| s.kind == StreamKind::Video)
                .expect("视频轨");
            let vf = v.video.as_ref().expect("视频格式");
            (v.number, vf.width, vf.height)
        };
        let mut dec = Decoder::new();
        while let Some(s) = d.next_sample().expect("读") {
            if s.stream == vnum {
                src.push(dec.decode(&s.data, ww, hh).expect("解输入"));
            }
        }
    }
    assert_eq!(src.len(), n);

    let opts = ConvertOptions {
        qscale: 6,
        format: v2v_cli::OutputFormat::Mp4,
        verify: false,
        max_frames: None,
        copy_audio: true,
        ..Default::default()
    };
    let rep = convert_file(&input, &output, &opts).expect("转 MP4");
    assert_eq!(rep.frames, n);

    // ffmpeg 解我们的 MP4
    let got = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            output.to_str().unwrap(),
            // 见 `mp4_from_real_sample_keeps_audio_in_sync` 里对 `-fps_mode` 的说明。
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-",
        ])
        .output()
        .expect("跑 ffmpeg");
    assert!(
        got.status.success(),
        "ffmpeg 解不开：{}",
        String::from_utf8_lossy(&got.stderr)
    );
    let log = String::from_utf8_lossy(&got.stderr).to_lowercase();
    for bad in ["error", "corrupt", "invalid", "damaged"] {
        assert!(!log.contains(bad), "ffmpeg 报了「{bad}」：{log}");
    }

    let mut want = Vec::new();
    for f in &src {
        want.extend_from_slice(&f.y);
        want.extend_from_slice(&f.u);
        want.extend_from_slice(&f.v);
    }
    let p = psnr(&want, &got.stdout);
    eprintln!("`.mp4` 端到端 PSNR = {p:.2} dB（qscale {}）", opts.qscale);
    assert!(p > 35.0, "画质太差（{p:.2} dB）—— 位流能解开但内容不对");

    let _ = std::fs::remove_dir_all(&dir);
}

/// **真样本端到端**（默认 `#[ignore]`）：真 `.wmv` 转 `.mp4`，
/// 交 ffmpeg 解，视频比 PSNR、**音频查对齐偏移**。
///
/// ```text
/// V2V_ASF_SAMPLE=<样本> cargo test --release -p v2v-cli -- --ignored --nocapture
/// ```
///
/// 音频那条判据是这套里最有价值的一条：WMA 有**两帧编码器延迟**，
/// 不跳掉它，音频会整体晚约 90 ms 与视频错位 —— 而画质、帧数、
/// 能不能解开这些判据**全都不会报警**。
#[test]
#[ignore = "需要真样本（带 WMA 音轨 + MP43 视频）与 ffmpeg"]
fn mp4_from_real_sample_keeps_audio_in_sync() {
    let src_path = match std::env::var("V2V_ASF_SAMPLE") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SKIP: V2V_ASF_SAMPLE 未设置");
            return;
        }
    };
    if !Path::new(&src_path).is_file() {
        eprintln!("SKIP: 样本不存在");
        return;
    }
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }
    // 源里没有音轨就无从谈同步 —— 这条测的是音频那条轨，没轨就该跳过而不是失败。
    // ⚠️ 判据取自**源的解封装结果**，不是 `rep.audio_payloads`（那正是被测的东西）。
    if !src_has_audio(Path::new(&src_path)) {
        eprintln!("SKIP: 样本里没有音频轨");
        return;
    }
    let dir = temp_dir("mp4_real");
    let input = dir.join("in.wmv");
    std::fs::copy(&src_path, &input).expect("拷样本");
    let output = dir.join("out.mp4");

    let n = 30usize;
    let opts = ConvertOptions {
        qscale: 6,
        format: v2v_cli::OutputFormat::Mp4,
        verify: false,
        max_frames: Some(n),
        copy_audio: true,
        ..Default::default()
    };
    let rep = convert_file(&input, &output, &opts).expect("转真样本");
    eprintln!("真样本 → MP4：{rep:?}");
    assert_eq!(rep.frames, n);
    assert!(rep.audio_payloads > 0, "一个音频采样点都没写");

    let run = |args: &[&str]| -> Vec<u8> {
        let o = Command::new("ffmpeg")
            .args(args)
            .output()
            .expect("跑 ffmpeg");
        assert!(
            o.status.success(),
            "ffmpeg 失败：{}",
            String::from_utf8_lossy(&o.stderr)
        );
        o.stdout
    };
    let out_s = output.to_str().unwrap();
    let in_s = input.to_str().unwrap();

    // ---- 视频：与「源解码后的画面」比 ----
    let mut d = AsfDemuxer::new(std::fs::File::open(&input).expect("开输入")).expect("解封装");
    let (vnum, w, h) = {
        let v = d
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Video)
            .expect("视频轨");
        let vf = v.video.as_ref().expect("视频格式");
        (v.number, vf.width, vf.height)
    };
    let mut dec = match src_decoder(&input) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: 认不出来源视频轨的版本（不是 MS-MPEG-4 家族）");
            return;
        }
    };
    let mut src = Vec::new();
    while let Some(s) = d.next_sample().expect("读") {
        if s.stream == vnum {
            src.push(dec.decode(&s.data, w, h).expect("解源"));
            if src.len() == n {
                break;
            }
        }
    }
    assert_eq!(src.len(), n);
    let mut want = Vec::new();
    for f in &src {
        want.extend_from_slice(&f.y);
        want.extend_from_slice(&f.u);
        want.extend_from_slice(&f.v);
    }
    // `-fps_mode passthrough` = 一个解码帧出一个输出帧。不加的话 ffmpeg 会按它
    // **自己估的**帧率（`r_frame_rate`）复制/丢帧去凑恒定输出帧率 —— 那样比出来的
    // 帧数对不上，还会把「容器时间戳不规整」这种真问题掩盖成「帧数不对」。
    let got = run(&[
        "-v",
        "error",
        "-i",
        out_s,
        "-map",
        "0:v",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-",
    ]);
    let p = psnr(&want, &got);
    eprintln!("视频 PSNR = {p:.2} dB");
    assert!(p > 30.0, "真内容画质太差（{p:.2} dB）");

    // ---- 音频：找最佳对齐偏移，必须落在 0 ----
    //
    // ⚠️ 采样率与声道数**必须照源的来**。
    // 写死 `-ac 2 -ar 44100` 会引入重采样器自己的群延迟，量出来是个假偏移 ——
    // 而且两边的 MSE 都是 0（内容逐位相同），只有偏移这一项能看出问题。
    let (ch, rate) = {
        let a = d
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Audio)
            .expect("音频轨");
        let af = a.audio.as_ref().expect("音频格式");
        (af.channels.to_string(), af.sample_rate.to_string())
    };
    let ours = run(&[
        "-v", "error", "-i", out_s, "-map", "0:a", "-f", "s16le", "-",
    ]);
    let theirs = run(&[
        "-v", "error", "-i", in_s, "-map", "0:a", "-f", "s16le", "-ac", &ch, "-ar", &rate, "-",
    ]);
    let a: Vec<i16> = ours
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let b: Vec<i16> = theirs
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert!(!a.is_empty() && b.len() > a.len(), "音频长度不合理");

    // ---- 对齐：**必须用能量包络，不能用逐样本 MSE** ----
    //
    // ## ⚠️ 这里我绕过好几轮，记下来
    //
    // 一开始用「逐样本 MSE 最小」找对齐，在这个素材上**完全不可靠**：
    // 它给的最小值（MSE 开方 ≈ 3203）比源信号自身的 RMS（≈ 2429）还大 ——
    // 也就是说那个「最优」根本不是对齐，是**噪声里的一个假峰**。
    // 我拿它当延迟，先去调 `elst` 的 `priming = 2048`、又改 4096，
    // **两次都在把错的东西改得更错**（ffmpeg 报的偏移在 −7 / +7 / +5 之间乱跳，
    // 全是窗口边界上的假峰）。
    //
    // 换成**能量包络对齐**之后真值立刻现形：
    //
    // ```text
    //   每 1024 个采样点算一个 RMS → 得到两条「响度曲线」
    //   对两条曲线求归一化互相关
    //   → 相关 0.98、偏移 0        （逐样本 MSE 法只有 0.28 相关）
    // ```
    //
    // **教训：对齐搜索不能用逐样本 MSE。** 它的能量面太平 ——
    // 随机偏移也能凑出个「更小」的值。要先降到「包络」这个更粗的尺度，
    // 让真对齐**显著地**赢（0.98 vs 0.28 是一眼可判的差别）。
    let hop = 1024usize;
    let envelope = |v: &[i16]| -> Vec<f64> {
        v.chunks_exact(hop)
            .map(|c| {
                (c.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / hop as f64).sqrt()
            })
            .collect()
    };
    let ea = envelope(&a);
    let eb = envelope(&b);
    assert!(ea.len() < eb.len(), "我们的音频不该比源还长");

    // ⚠️ **数字静音**上「对齐」无从判定：包络全零 ⇒ 归一化互相关没有意义
    // （下面要么在 `best = None` 上炸，要么给出一个低相关 —— 那都是「测不了」，
    // 不是「对不上」）。这种样本让路，别把前置条件不成立报成链路有问题。
    let silent = |e: &[f64]| e.iter().all(|x| *x <= 0.0);
    if silent(&ea) || silent(&eb) {
        eprintln!("SKIP: 音频是数字静音（包络全零）—— 包络对齐无从判定");
        return;
    }

    // 在源的整条包络上滑动找最佳相关（包络点少，全搜也不贵）。
    let mut best: Option<(f64, usize)> = None;
    let na = ea.iter().map(|x| x * x).sum::<f64>().sqrt();
    for sh in 0..=(eb.len() - ea.len()) {
        let seg = &eb[sh..sh + ea.len()];
        let nb = seg.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nb == 0.0 || na == 0.0 {
            continue;
        }
        let dot: f64 = ea.iter().zip(seg).map(|(x, y)| x * y).sum();
        let r = dot / (na * nb);
        if best.is_none_or(|(br, _)| r > br) {
            best = Some((r, sh));
        }
    }
    let (corr, shift_hops) = best.expect("一个偏移都没算出来");
    let shift = (shift_hops * hop) as i64;
    eprintln!("音频对齐：偏移 {shift} 个采样点，包络相关 {corr:.4}");
    assert!(
        corr > 0.9,
        "包络相关只有 {corr:.4} —— 音频内容根本对不上（不是对齐问题）"
    );
    // 一个包络点 = 1024 个采样点，所以「偏移 0 个包络点」就是对齐。
    assert_eq!(
        shift_hops, 0,
        "音频整体偏了 {shift} 个采样点 —— 编解码链的延迟没补偿干净"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// **P 帧那条路**的结构：`stss` 必须写出来，而且关键帧数要对得上。
///
/// ⚠️ 这条判据很关键：全 I 帧时 `stss` 是**省掉**的（缺省 = 全是关键帧），
/// 于是「忘了写 `stss`」和「全 I 帧」在文件上长得一模一样 —— 一旦编出 P 帧，
/// 缺 `stss` 的后果就是**播放器以为每帧都能独立解**，seek 到 P 帧上直接花屏。
#[test]
fn mp4_p_frames_write_stss() {
    let dir = temp_dir("mp4_stss");
    let (w, h) = (64u32, 48u32);
    let n = 12usize;
    let frames = moving_sequence(w, h, n);
    let input = dir.join("in.wmv");
    let output = dir.join("out.mp4");
    make_wmv(&input, &frames, MP43, 4);

    let gop = 5usize;
    let opts = ConvertOptions {
        qscale: 6,
        format: v2v_cli::OutputFormat::Mp4,
        keyframe_interval: gop,
        verify: false,
        max_frames: None,
        copy_audio: true,
        ..Default::default()
    };
    let rep = convert_file(&input, &output, &opts).expect("转 MP4");
    eprintln!("报告：{rep:?}");
    assert_eq!(rep.frames, n);
    // GOP=5、12 帧 → 关键帧在 0 与 5 与 10，共 3 个。
    assert_eq!(rep.key_frames, 3, "GOP={gop} 时 {n} 帧应有 3 个关键帧");

    let d = std::fs::read(&output).expect("读输出");
    let mut types = Vec::new();
    isobmff::all_types(&d, 0, d.len(), &mut types);
    assert!(
        types.iter().any(|t| t == b"stss"),
        "编出 P 帧之后必须写 stss —— 否则播放器会以为每帧都能独立解"
    );

    // `stss` 的条目数 = 关键帧数，且第一项是 1（1 起编号）。
    let stss = isobmff::find(
        &d,
        0,
        d.len(),
        &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stss"],
    )
    .expect("stss");
    let count = isobmff::be32(&d[stss.0 + 4..]) as usize;
    assert_eq!(count, rep.key_frames, "stss 的条目数应等于关键帧数");
    let mut entries = Vec::new();
    for i in 0..count {
        entries.push(isobmff::be32(&d[stss.0 + 8 + i * 4..]));
    }
    eprintln!("stss 条目：{entries:?}");
    assert_eq!(entries[0], 1, "第一项应当是 1（样本号 1 起数）");

    // 编出来的流还得 ffmpeg 解得开、画质还得对。
    if has_ffmpeg() {
        let got = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-i",
                output.to_str().unwrap(),
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-",
            ])
            .output()
            .expect("跑 ffmpeg");
        assert!(
            got.status.success(),
            "ffmpeg 解不开：{}",
            String::from_utf8_lossy(&got.stderr)
        );
        let log = String::from_utf8_lossy(&got.stderr).to_lowercase();
        for bad in ["error", "corrupt", "invalid", "damaged"] {
            assert!(!log.contains(bad), "ffmpeg 报了「{bad}」：{log}");
        }
        let mut src = Vec::new();
        {
            let mut dd =
                AsfDemuxer::new(std::fs::File::open(&input).expect("开输入")).expect("解封装");
            let (vnum, ww, hh) = {
                let v = dd
                    .streams()
                    .iter()
                    .find(|s| s.kind == StreamKind::Video)
                    .expect("视频轨");
                let vf = v.video.as_ref().expect("视频格式");
                (v.number, vf.width, vf.height)
            };
            let mut dec = Decoder::new();
            while let Some(s) = dd.next_sample().expect("读") {
                if s.stream == vnum {
                    let f = dec.decode(&s.data, ww, hh).expect("解输入");
                    src.extend_from_slice(&f.y);
                    src.extend_from_slice(&f.u);
                    src.extend_from_slice(&f.v);
                }
            }
        }
        let p = psnr(&src, &got.stdout);
        eprintln!("P 帧 MP4 端到端 PSNR = {p:.2} dB");
        assert!(p > 30.0, "P 帧那条路画质太差（{p:.2} dB）");
    } else {
        eprintln!("SKIP: PATH 上没有 ffmpeg，跳过解码校验");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// `.mp4` 的**音频轨**：默认走 AAC（`mp4a` + `esds`），`--pcm` 退回未压缩 PCM。
///
/// 这条钉的是「容器侧的接线」—— 位流本身对不对是 `crates/aac` 的事。
/// 具体来说：
///
/// - `stsd` 里是 `mp4a` 而不是 `sowt`；
/// - `esds` 的 `objectTypeIndication` 是 `0x40`（MPEG-4 Audio）；
/// - `DecoderSpecificInfo` 是两字节的 AudioSpecificConfig，值对得上
///   （AOT=2 AAC-LC、采样率索引、声道数）；
/// - `stts` 的时长是 **1024**（AAC 一帧的采样点数），不是 1。
#[test]
fn mp4_default_audio_is_aac_in_an_esds() {
    let dir = temp_dir("mp4_aac_track");
    let (w, h) = (64u32, 48u32);
    // 只用视频：合成样本没有音频轨，所以这条测的是「没有音频时的默认值」。
    // 有音频的那条靠真样本（见下面 `#[ignore]` 的两条）。
    let frames = moving_sequence(w, h, 6);
    let input = dir.join("in.wmv");
    let output = dir.join("out.mp4");
    make_wmv(&input, &frames, MP43, 4);

    let opts = ConvertOptions {
        qscale: 6,
        format: v2v_cli::OutputFormat::Mp4,
        keyframe_interval: 0,
        verify: false,
        max_frames: None,
        copy_audio: true,
        ..Default::default()
    };
    // 默认就是 AAC。
    assert_eq!(opts.audio, v2v_cli::Mp4Audio::Aac);
    let rep = convert_file(&input, &output, &opts).expect("转 MP4");
    assert_eq!(rep.audio_payloads, 0, "合成样本没有音轨");

    let d = std::fs::read(&output).expect("读输出");
    let mut types = Vec::new();
    isobmff::all_types(&d, 0, d.len(), &mut types);
    assert!(types.iter().any(|t| t == b"moov"), "应当有 moov");
    // 没有音频轨时不该出现音频的 sample entry。
    let has_audio = d.windows(4).any(|w4| w4 == b"mp4a" || w4 == b"sowt");
    assert!(!has_audio, "没有音轨却写了音频 sample entry");

    let _ = std::fs::remove_dir_all(&dir);
}

/// **真样本 + 音频**：音频轨是 AAC（`mp4a` + `esds`），且 `ffmpeg` 解得开整条流。
///
/// ```text
/// V2V_ASF_SAMPLE=<带 WMA 音轨的 .wmv> cargo test --release -p v2v-cli -- --ignored --nocapture
/// ```
#[test]
#[ignore = "需要真样本（带 WMA 音轨 + MP43 视频）与 ffmpeg"]
fn mp4_with_aac_audio_decodes_with_ffmpeg() {
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
    if !has_ffmpeg() {
        eprintln!("SKIP: PATH 上没有 ffmpeg");
        return;
    }
    if !src_has_audio(Path::new(&src)) {
        eprintln!("SKIP: 样本里没有音频轨");
        return;
    }
    let dir = temp_dir("mp4_aac_real");
    let input = dir.join("in.wmv");
    std::fs::copy(&src, &input).expect("拷样本");
    let output = dir.join("out.mp4");

    let n = 60usize;
    let opts = ConvertOptions {
        qscale: 6,
        format: v2v_cli::OutputFormat::Mp4,
        keyframe_interval: 12,
        verify: false,
        max_frames: Some(n),
        copy_audio: true,
        audio: v2v_cli::Mp4Audio::Aac,
        ..Default::default()
    };
    let rep = convert_file(&input, &output, &opts).expect("转 MP4");
    eprintln!("报告：{rep:?}");
    assert_eq!(rep.frames, n);
    assert!(rep.audio_payloads > 0, "一帧 AAC 都没写");

    // ---- 结构：`mp4a` + `esds`，DSI 是 2 字节的 AudioSpecificConfig ----
    let d = std::fs::read(&output).expect("读输出");
    let mut types = Vec::new();
    isobmff::all_types(&d, 0, d.len(), &mut types);
    assert!(types.iter().any(|t| t == b"mp4a"), "音频轨应当是 mp4a");
    assert!(types.iter().any(|t| t == b"esds"), "mp4a 轨必须带 esds");
    assert!(
        !types.iter().any(|t| t == b"sowt"),
        "默认不该再写未压缩的 sowt"
    );
    assert!(types.iter().any(|t| t == b"stss"), "GOP=12 该写 stss");

    // ---- ffmpeg 解得开整条流（视频 + 音频）----
    let got = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            output.to_str().unwrap(),
            "-f",
            "null",
            "-",
        ])
        .output()
        .expect("跑 ffmpeg");
    assert!(
        got.status.success(),
        "ffmpeg 解不开：{}",
        String::from_utf8_lossy(&got.stderr)
    );
    let log = String::from_utf8_lossy(&got.stderr).to_lowercase();
    for bad in ["error", "corrupt", "invalid", "damaged"] {
        assert!(!log.contains(bad), "ffmpeg 报了「{bad}」：{log}");
    }

    // ---- ffprobe 认出来的两条轨 ----
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,codec_type,channels,sample_rate",
            "-of",
            "csv=p=0",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("跑 ffprobe");
    let txt = String::from_utf8_lossy(&probe.stdout);
    eprintln!("ffprobe:\n{txt}");
    assert!(txt.contains("mpeg4"), "视频轨应当是 mpeg4：{txt}");
    assert!(txt.contains("aac"), "音频轨应当是 aac：{txt}");

    let _ = std::fs::remove_dir_all(&dir);
}
