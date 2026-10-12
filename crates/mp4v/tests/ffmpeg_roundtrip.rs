//! MPEG-4 Part 2 编码器**对账 ffmpeg**。
//!
//! ## 判据是什么
//!
//! 编码器没有「参考位流」可比 —— 同一帧可以编出无数种合法的位流。
//! 唯一有意义的判据是 **ffmpeg 解得开，且解出来的画面接近原图**：
//!
//! 1. `ffmpeg -i ours.m4v -f rawvideo` **退出码 0**，stderr 里没有 `error` / `corrupt`；
//! 2. 解出来的尺寸、帧数对；
//! 3. 逐像素 PSNR 够高（阈值见 `PSNR_MIN_DB`）；
//! 4. **同一个 qscale 下和 ffmpeg 自己的 mpeg4 编码器比一比** —— 不是要求追平，
//!    而是要求「在同一量级」：差 10 dB 说明我们的编码器有实质性的错（不是调优问题）。
//!
//! 第 4 条是这套测试里最有价值的一条。只测「能解开」会漏掉一整类错：
//! 预测方向反了、DC 缩放表用错、量化公式反了 —— 这些都能解开，只是画质崩。
//!
//! ## 为什么要跑两遍（裸流 + MP4）
//!
//! 两条路会踩不同的坑：
//!
//! - **裸基本流**（`.m4v`）：VOL 在流里，解码器自己找头。测的是**位流本身**。
//! - **MP4**：VOL 搬进 `esds` 的 DSI，流里只剩 VOP。测的是**VOL 那份拷贝
//!   与流里的一致**（`decoder_specific_info()` 与 `write_vol_into()` 是同一个函数，
//!   但这条测试会把「不小心分叉」钉死）。
//!
//! ## 怎么跑
//!
//! ```bash
//! cargo test -p v2v-mp4v --test ffmpeg_roundtrip -- --nocapture
//! ```
//!
//! 需要 `ffmpeg` / `ffprobe` 在 `PATH` 上。

use std::io::Write;
use std::process::{Command, Stdio};

use v2v_mp4v::{Encoder, EncoderConfig};
use v2v_msmpeg4::Frame;

const W: u32 = 320;
const H: u32 = 240;
const FPS: u32 = 25;
const FRAMES: usize = 12;

/// 逐像素 PSNR 的下限。qscale = 4 的 MPEG-4 在 320×240 上应当远好于这个数。
const PSNR_MIN_DB: f64 = 30.0;

/// 与 ffmpeg 自己的编码器相比，允许的最大差距（dB）。
///
/// 我们只做全 I 帧、不做 dquant、不做自适应量化，所以必然差一些；
/// 但「差一点点」和「差 10 dB」是两回事 —— 后者说明有实质性的错。
const PSNR_GAP_MAX_DB: f64 = 6.0;

// ==================== 测试素材 ====================
//
// 刻意**三种结构都有**，因为每种结构考验编码器的不同部分：
//
// - 平滑斜向渐变 → AC 预测应当赢（邻居首行/首列很像）
// - 移动的亮方块 → 强边缘，AC 预测应当输（邻居完全不同）
// - 细竖条纹 → 高频，考验 RL escape 与扫描序
//
// 只喂平滑渐变会漏掉 escape 的错；只喂噪声会让「能不能解开」都成问题。

fn test_frame(t: usize) -> Frame {
    let (w, h) = (W as usize, H as usize);
    let mut y = vec![0u8; w * h];
    // 移动的方块
    let cx = (w / 2 + t * 4) % w;
    let cy = (h / 3 + t * 3) % h;
    for j in 0..h {
        for i in 0..w {
            // 平滑斜向渐变（0..200），再叠一点时间变化
            let mut v =
                (i as f64 / w as f64) * 200.0 + (j as f64 / h as f64) * 55.0 + t as f64 * 2.0;
            // 强边缘：一个 48×48 的亮方块
            if i.abs_diff(cx) < 24 && j.abs_diff(cy) < 24 {
                v = 235.0;
            }
            // 高频：每 37 像素一根细黑线，随时间平移
            if (i + t * 3) % 37 == 0 {
                v = 16.0;
            }
            y[j * w + i] = v.clamp(0.0, 255.0) as u8;
        }
    }

    let (cw, ch) = (w / 2, h / 2);
    let mut u = vec![0u8; cw * ch];
    let mut v_plane = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            let dx = i as f64 - (cw as f64 / 2.0 + t as f64 * 2.0);
            let dy = j as f64 - (ch as f64 / 2.0);
            let r = (dx * dx + dy * dy).sqrt();
            u[j * cw + i] = if r < 40.0 { 200 } else { 90 };
            v_plane[j * cw + i] = if r < 60.0 { 60 } else { 170 };
        }
    }

    Frame {
        width: W,
        height: H,
        y,
        u,
        v: v_plane,
    }
}

/// 把一帧 YUV420P 写成 `ffmpeg` 认的裸帧字节。
fn frame_bytes(f: &Frame) -> Vec<u8> {
    let mut b = f.y.clone();
    b.extend_from_slice(&f.u);
    b.extend_from_slice(&f.v);
    b
}

// ==================== ffmpeg 帮手 ====================

fn ffmpeg(args: &[&str], stdin_data: Option<&[u8]>) -> (bool, Vec<u8>, String) {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(args)
        .stdin(if stdin_data.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("起不来 ffmpeg（在 PATH 上吗？）");
    if let Some(d) = stdin_data {
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(d)
            .expect("喂不进 stdin");
    }
    let out = child.wait_with_output().expect("ffmpeg 没跑完");
    (
        out.status.success(),
        out.stdout,
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// 用 ffmpeg 把一个视频文件解成裸 YUV420P。
fn decode_to_yuv(path: &str) -> (Vec<u8>, String) {
    let (ok, data, log) = ffmpeg(
        &[
            "-v", "warning", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-",
        ],
        None,
    );
    assert!(ok, "ffmpeg 解不开 {path}：\n{log}");
    (data, log)
}

/// 算两段裸 YUV420P 的 PSNR（dB）。长度不一致直接判失败。
fn psnr(a: &[u8], b: &[u8], frames: usize) -> f64 {
    assert_eq!(a.len(), b.len(), "解码出来的字节数不一样");
    assert_eq!(a.len(), frames * (W * H * 3 / 2) as usize, "帧数不对");
    let mut se = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = f64::from(*x) - f64::from(*y);
        se += d * d;
    }
    let mse = se / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

// ==================== 测试 ====================

/// 把我们的编码结果写成一个裸 `.m4v` 基本流。
///
/// ⚠️ `cfg.keyframe_interval == 0` 时走 **`encode_i_frame`（全 I 帧）**，
/// 而不是 `encode_frame` —— 后者在 `mp4v` 里的语义是「无 GOP」（首帧 I、其余全 P），
/// 是个容易写错的坑（第一个 P 帧会因为没有参考帧而报错）。
/// 非 0 时才用 `encode_frame`，按 GOP 真编 P 帧。
fn encode_raw_es(cfg: EncoderConfig, frames: &[Frame]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut e = Encoder::new(cfg).expect("建编码器");
    let mut es = e.elementary_stream_header();
    let mut samples = Vec::new();
    for f in frames {
        let s = if cfg.keyframe_interval == 0 {
            e.encode_i_frame(f).expect("编一帧")
        } else {
            e.encode_frame(f).expect("编一帧")
        };
        es.extend_from_slice(&s);
        samples.push(s);
    }
    (es, samples)
}

#[test]
fn ffmpeg_decodes_our_elementary_stream() {
    let frames: Vec<Frame> = (0..FRAMES).map(test_frame).collect();
    let cfg = EncoderConfig {
        width: W,
        height: H,
        fps_num: FPS,
        ..Default::default()
    };
    let (es, _) = encode_raw_es(cfg, &frames);

    let path = std::env::temp_dir().join("v2v_mp4v_ours.m4v");
    std::fs::write(&path, &es).expect("写基本流");
    let path = path.to_str().unwrap();

    let (got, log) = decode_to_yuv(path);
    for bad in ["error", "corrupt", "invalid", "damaged"] {
        assert!(
            !log.to_lowercase().contains(bad),
            "ffmpeg 报了「{bad}」—— 位流有问题：\n{log}"
        );
    }

    let mut src = Vec::new();
    for f in &frames {
        src.extend_from_slice(&frame_bytes(f));
    }
    let p = psnr(&src, &got, FRAMES);
    println!("[裸基本流] qscale = {}，PSNR = {p:.2} dB", cfg.qscale);
    assert!(
        p >= PSNR_MIN_DB,
        "PSNR 只有 {p:.2} dB（下限 {PSNR_MIN_DB}）—— 位流能解开但画质崩了，多半是预测/量化错了"
    );
}

#[test]
fn quality_is_in_the_same_league_as_ffmpeg() {
    let frames: Vec<Frame> = (0..FRAMES).map(test_frame).collect();
    let cfg = EncoderConfig {
        width: W,
        height: H,
        fps_num: FPS,
        ..Default::default()
    };
    let (es, _) = encode_raw_es(cfg, &frames);

    let ours_path = std::env::temp_dir().join("v2v_mp4v_league_ours.m4v");
    std::fs::write(&ours_path, &es).expect("写基本流");
    let ours_path = ours_path.to_str().unwrap();

    // 同一份原始帧喂给 ffmpeg 自己的 mpeg4 编码器，用同一个 qscale。
    let mut raw = Vec::new();
    for f in &frames {
        raw.extend_from_slice(&frame_bytes(f));
    }
    let ref_path = std::env::temp_dir().join("v2v_mp4v_league_ref.m4v");
    let ref_path = ref_path.to_str().unwrap().to_string();
    let (ok, _, log) = ffmpeg(
        &[
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            &format!("{W}x{H}"),
            "-r",
            &FPS.to_string(),
            "-i",
            "-",
            "-c:v",
            "mpeg4",
            "-qscale:v",
            &cfg.qscale.to_string(),
            "-y",
            &ref_path,
        ],
        Some(&raw),
    );
    assert!(ok, "ffmpeg 自己的 mpeg4 编码器没跑起来：\n{log}");

    let (ours, _) = decode_to_yuv(ours_path);
    let (theirs, _) = decode_to_yuv(&ref_path);
    let mut src = Vec::new();
    for f in &frames {
        src.extend_from_slice(&frame_bytes(f));
    }
    let p_ours = psnr(&src, &ours, FRAMES);
    let p_theirs = psnr(&src, &theirs, FRAMES);
    println!("[同 qscale 对比] 我们 {p_ours:.2} dB / ffmpeg {p_theirs:.2} dB");
    assert!(
        p_theirs - p_ours <= PSNR_GAP_MAX_DB,
        "比 ffmpeg 差了 {:.2} dB（上限 {PSNR_GAP_MAX_DB}）—— 这不是调优能解释的差距",
        p_theirs - p_ours
    );
}

#[test]
fn vol_survives_the_trip_into_an_mp4_esds() {
    use v2v_mp4::{Mp4Muxer, VideoTrack};

    let frames: Vec<Frame> = (0..FRAMES).map(test_frame).collect();
    let cfg = EncoderConfig {
        width: W,
        height: H,
        fps_num: FPS,
        ..Default::default()
    };
    let mut e = Encoder::new(cfg).expect("建编码器");
    let dsi = e.decoder_specific_info();

    let mut mux = Mp4Muxer::new(VideoTrack {
        width: W as u16,
        height: H as u16,
        timescale: FPS * 512,
        decoder_specific_info: dsi.clone(),
        avg_bitrate: 0,
    });
    let dur = 512; // timescale = fps*512 → 每帧 512 个 tick
    for f in &frames {
        let s = e.encode_i_frame(f).expect("编一帧");
        mux.write_video(&s, true, dur).expect("写样本");
    }
    let mp4 = mux.finish();

    let path = std::env::temp_dir().join("v2v_mp4v_ours.mp4");
    std::fs::write(&path, &mp4).expect("写 mp4");
    let path = path.to_str().unwrap();

    // ffprobe 从 esds 里读出的参数必须和我们对得上 —— 这证明 DSI 里的 VOL 是完整的。
    let (ok, out, log) = ffmpeg_probe(&[
        "-v",
        "error",
        "-select_streams",
        "v",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,profile,level",
        "-of",
        "default=nw=1",
        path,
    ]);
    assert!(ok, "ffprobe 读不了 {path}：\n{log}");
    println!("[MP4] ffprobe:\n{out}");
    assert!(out.contains("codec_name=mpeg4"), "codec 不对：\n{out}");
    assert!(out.contains(&format!("width={W}")), "宽度不对：\n{out}");
    assert!(out.contains(&format!("height={H}")), "高度不对：\n{out}");
    assert!(out.contains("pix_fmt=yuv420p"), "像素格式不对：\n{out}");
    // `profile` 是从 DSI 里的 **VOS 头**读出来的 —— 只放 VOL 的话这里会是 `unknown`。
    assert!(
        out.contains("profile=Simple Profile"),
        "profile 不对（DSI 里缺 VOS 头？）：\n{out}"
    );
    assert!(out.contains("level=1"), "level 不对：\n{out}");

    let (got, log) = decode_to_yuv(path);
    assert!(!log.to_lowercase().contains("error"), "解码报了错：\n{log}");
    let mut src = Vec::new();
    for f in &frames {
        src.extend_from_slice(&frame_bytes(f));
    }
    let p = psnr(&src, &got, FRAMES);
    println!("[MP4 封装] PSNR = {p:.2} dB");
    assert!(p >= PSNR_MIN_DB, "MP4 这条路上画质崩了：{p:.2} dB");
}

fn ffprobe_probe(args: &[&str]) -> (bool, String, String) {
    let out = Command::new("ffprobe")
        .args(args)
        .output()
        .expect("起不来 ffprobe（在 PATH 上吗？）");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// `ffmpeg -i` 的 probe 分支走的是同一个 `ffprobe` 库，直接复用它省一层壳。
fn ffmpeg_probe(args: &[&str]) -> (bool, String, String) {
    ffprobe_probe(args)
}

/// 一段**低运动**序列：静态背景 + 缓慢平移的方块。
///
/// 真实视频里绝大多数帧是这样的，而 `test_frame` 那种「斜向渐变 + 每帧平移 3 像素的
/// 细条纹」在 qscale=4 下几乎不可预测 —— 实测最佳整帧平移的 MSE（1100）比「压根不动」
/// （1008）还差，也就是说相邻帧之间**没有可用的运动相关性**。
/// 所以体积那条判据要在这段上量，不然量的是「这段素材本来就压不动」。
fn low_motion_frame(t: usize) -> Frame {
    let (w, h) = (W as usize, H as usize);
    let mut y = vec![0u8; w * h];
    // 静态背景：平滑的横向渐变
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = (60 + (i * 120 / w)) as u8;
        }
    }
    // 一个缓慢右移的方块（每帧 2 像素 → MV (4,0) 半像素单位）
    let ox = (w / 4 + t * 2).min(w - 48);
    let oy = h / 3;
    for r in 0..48usize {
        for c in 0..48usize {
            y[(oy + r) * w + ox + c] = if (r / 8 + c / 8) % 2 == 0 { 40 } else { 230 };
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    Frame {
        width: W,
        height: H,
        y,
        u: vec![110; cw * ch],
        v: vec![150; cw * ch],
    }
}

// ==================== P 帧 ====================

/// 一个 **I+P 序列**要同时满足三件事，缺一不可：
///
/// 1. **ffmpeg 解得开、不报错** —— 位流语法对；
/// 2. **画质没崩** —— MV / 残差 / 重建这一套全对（错一处画质会断崖式下跌）；
/// 3. **体积真的小了** —— 不然就是「P 帧编出来了但等于没用」。
///
/// 第 3 条尤其重要：前两条在「P 帧全部退化成帧内宏块」时**照样能过**。
/// 所以要顺带断言「帧间宏块确实占了多数」。
#[test]
fn p_frames_cut_the_bitrate_without_breaking_the_stream() {
    let frames: Vec<Frame> = (0..FRAMES).map(test_frame).collect();

    // 全 I 帧（基准）
    let all_i = EncoderConfig {
        width: W,
        height: H,
        fps_num: FPS,
        keyframe_interval: 0,
        ..Default::default()
    };
    let (es_i, samples_i) = encode_raw_es(all_i, &frames);

    // GOP = 250 → 只有首帧是 I，其余全是 P
    let with_p = EncoderConfig {
        keyframe_interval: 250,
        ..all_i
    };
    let (es_p, samples_p) = encode_raw_es(with_p, &frames);

    // ---- 1. ffmpeg 解得开 ----
    let path_i = std::env::temp_dir().join("v2v_mp4v_alli.m4v");
    std::fs::write(&path_i, &es_i).expect("写");
    let path_p = std::env::temp_dir().join("v2v_mp4v_withp.m4v");
    std::fs::write(&path_p, &es_p).expect("写");

    let (got_i, log_i) = decode_to_yuv(path_i.to_str().unwrap());
    let (got_p, log_p) = decode_to_yuv(path_p.to_str().unwrap());
    for (tag, log) in [("全 I", &log_i), ("I+P", &log_p)] {
        for bad in ["error", "corrupt", "invalid", "damaged"] {
            assert!(
                !log.to_lowercase().contains(bad),
                "{tag} 的流里 ffmpeg 报了「{bad}」：\n{log}"
            );
        }
    }

    // ---- 2. 画质 ----
    let mut src = Vec::new();
    for f in &frames {
        src.extend_from_slice(&frame_bytes(f));
    }
    let p_i = psnr(&src, &got_i, FRAMES);
    let p_p = psnr(&src, &got_p, FRAMES);
    println!("[全 I] PSNR {p_i:.2} dB，基本流 {} 字节", es_i.len());
    println!("[I+P] PSNR {p_p:.2} dB，基本流 {} 字节", es_p.len());
    // P 帧会累积漂移，比全 I 差一点是正常的 —— 但不能崩。
    assert!(
        p_p >= PSNR_MIN_DB,
        "I+P 的画质崩了（{p_p:.2} dB）—— MV / 残差 / 重建里有一处错"
    );
    assert!(
        p_i - p_p <= 6.0,
        "P 帧把画质拖低了 {:.2} dB，太多了",
        p_i - p_p
    );

    // ---- 3. 体积 ----
    // 逐帧比：首帧 I 两边一样，从第 2 帧起 P 应当明显小。
    let first_p = samples_p[1].len();
    let first_i = samples_i[1].len();
    println!("[第 2 帧] 全 I {first_i} 字节 → P {first_p} 字节");
    assert!(
        es_p.len() < es_i.len(),
        "P 帧没把体积压下来（{} vs {}）",
        es_p.len(),
        es_i.len()
    );
    // ⚠️ **在这段素材上只能要求「有节省」，不能要求省多少** ——
    // 它是斜向渐变 + 每帧平移 3 像素的细条纹，相邻帧之间几乎没有可用的运动相关性
    // （实测最佳整帧平移的 MSE 比「压根不动」还差）。体积的判据放在下面那条
    // 低运动序列上量才有意义。
    assert!(
        es_p.len() * 10 < es_i.len() * 9,
        "P 帧一点没省（{} vs {}）",
        es_p.len(),
        es_i.len()
    );

    let _ = std::fs::remove_file(&path_i);
    let _ = std::fs::remove_file(&path_p);
}

/// **压缩率**这条判据要在有运动相关性的素材上量。
///
/// 低运动序列（静态背景 + 每帧移 2 像素的方块）是真实视频的常态：
/// 绝大部分宏块一个 skip 位（1 bit）就过去了。这里要求 P 帧把体积压到 **1/4 以下**。
///
/// 这条测试是「P 帧真的在干活」的硬判据 —— 前面那些（ffmpeg 解得开、画质没崩）
/// 在「P 帧全部退化成帧内宏块」时**照样能过**。
#[test]
fn p_frames_compress_a_low_motion_sequence() {
    let frames: Vec<Frame> = (0..FRAMES).map(low_motion_frame).collect();

    let base = EncoderConfig {
        width: W,
        height: H,
        fps_num: FPS,
        ..Default::default()
    };
    let (es_i, _) = encode_raw_es(
        EncoderConfig {
            keyframe_interval: 0,
            ..base
        },
        &frames,
    );
    let (es_p, samples_p) = encode_raw_es(
        EncoderConfig {
            keyframe_interval: 250,
            ..base
        },
        &frames,
    );

    // ffmpeg 也得解得开、画质也得对。
    let path = std::env::temp_dir().join("v2v_mp4v_lowmotion.m4v");
    std::fs::write(&path, &es_p).expect("写");
    let (got, log) = decode_to_yuv(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    for bad in ["error", "corrupt", "invalid", "damaged"] {
        assert!(
            !log.to_lowercase().contains(bad),
            "ffmpeg 报错：
{log}"
        );
    }
    let mut src = Vec::new();
    for f in &frames {
        src.extend_from_slice(&frame_bytes(f));
    }
    let p = psnr(&src, &got, FRAMES);
    let ratio = es_p.len() as f64 / es_i.len() as f64;
    println!(
        "[低运动] 全 I {} 字节 → I+P {} 字节（{:.0}%），PSNR {p:.2} dB",
        es_i.len(),
        es_p.len(),
        ratio * 100.0
    );
    println!(
        "[低运动] 第 2 帧：全 I {} 字节 → P {} 字节",
        es_i.len() / FRAMES,
        samples_p[1].len()
    );
    assert!(p >= PSNR_MIN_DB, "低运动序列的画质崩了（{p:.2} dB）");
    assert!(
        ratio < 0.25,
        "P 帧只压到 {:.0}% —— 低运动序列该压到 25% 以下",
        ratio * 100.0
    );
    // 低运动素材里第 2 帧该是「大部分 skip + 少量帧间宏块」。剩下的那点字节主要来自
    // **跟着方块走的那几个宏块**（残差 + MV）—— 方块有强边缘，qscale=4 下残差不算小。
    let per_i = es_i.len() / FRAMES;
    assert!(
        samples_p[1].len() * 3 < per_i,
        "第 2 帧还有 {} 字节（全 I 是 {per_i}）—— 低运动序列该小得多",
        samples_p[1].len()
    );
}

/// P 帧里「帧间宏块用上了没有、MV 有没有真的动」—— 单看体积容易漏掉这层。
///
/// 判据：`intra_mbs_last`（上一帧走帧内路径的宏块数）在 P 帧里必须**远小于**总数。
#[test]
fn p_frames_actually_use_inter_macroblocks() {
    let frames: Vec<Frame> = (0..FRAMES).map(test_frame).collect();
    let mut e = Encoder::new(EncoderConfig {
        width: W,
        height: H,
        fps_num: FPS,
        keyframe_interval: 250,
        ..Default::default()
    })
    .expect("建编码器");

    let total = e.mb_count().0 * e.mb_count().1;
    e.encode_frame(&frames[0]).expect("I 帧");
    assert!(e.last_key_frame, "首帧必须是关键帧");
    assert_eq!(e.intra_mbs_last, total, "I 帧里全是帧内宏块");

    for (i, f) in frames.iter().enumerate().skip(1) {
        e.encode_frame(f).expect("P 帧");
        assert!(!e.last_key_frame, "第 {i} 帧不该是关键帧");
        let intra = e.intra_mbs_last;
        println!("第 {i} 帧：帧内宏块 {intra}/{total}");
        assert!(
            intra * 4 < total,
            "第 {i} 帧有 {intra}/{total} 个宏块走了帧内 —— 帧间路径没生效"
        );
    }
}

/// **逐帧、逐位**比对：把我们的 I+P 流交给 ffmpeg 解，同时用**自己的解码器**解一遍。
///
/// 问题是 `v2v-mp4v` 只有编码器、没有解码器 —— 所以这条判据反过来用：
/// 让 ffmpeg 解我们的流，再让 ffmpeg 解「编码器看到的画面」，比 PSNR。
/// 上一条测试已经做了这件事。这里补的是**逐帧**版本的诊断输出，方便定位出问题的是哪一帧。
#[test]
fn every_p_frame_stays_within_tolerance() {
    let frames: Vec<Frame> = (0..FRAMES).map(test_frame).collect();
    let (es, _) = encode_raw_es(
        EncoderConfig {
            width: W,
            height: H,
            fps_num: FPS,
            keyframe_interval: 250,
            ..Default::default()
        },
        &frames,
    );
    let path = std::env::temp_dir().join("v2v_mp4v_perframe.m4v");
    std::fs::write(&path, &es).expect("写");
    let (got, _) = decode_to_yuv(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);

    let one = (W * H * 3 / 2) as usize;
    let mut worst = f64::INFINITY;
    for (i, f) in frames.iter().enumerate() {
        let a = frame_bytes(f);
        let b = &got[i * one..(i + 1) * one];
        let p = psnr(&a, b, 1);
        println!("第 {i:2} 帧 PSNR {p:.2} dB");
        worst = worst.min(p);
    }
    println!("最差 {worst:.2} dB");
    // P 帧漂移会让最差的那帧比平均低，但不该低到难看。
    assert!(worst >= 28.0, "有一帧画质崩了（最差 {worst:.2} dB）");
}
