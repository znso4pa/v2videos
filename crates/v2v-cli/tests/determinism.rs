//! B3：**确定性契约** —— 同输入 + 同参数 → **字节相同**的输出。
//!
//! ## 为什么这条值得单独钉
//!
//! 「编码器没有随机数、没有多线程，所以当然是确定的」是**推断**，不是事实。真实
//! 的不确定性来源有一堆，而且**每一个都不报错**：
//!
//! - 往容器里写创建时间 / 修改时间（每次运行都不同）；
//! - 遍历 `HashMap`（Rust 的默认哈希种子**每个进程都不一样**）；
//! - 并行调度：谁先算完谁先写，GOP 之间的字节顺序就变了；
//! - 浮点 / 迭代到收敛（本仓没有，但换个人写就可能引入）；
//! - 时间戳按「当前时间」而不是「源的时间」算。
//!
//! 这些都不会让产物「打不开」，只会让**同一份输入每次跑出不同的文件** —— 于是
//! 校验和没法比对、增量构建的缓存全部失效、复现一个 bug 要靠运气。
//!
//! ## 判据
//!
//! **直接比字节**，不比 sha256：sha256 只是「字节相同」的压缩写法，而直接比能报出
//! 「第几个字节不同」，诊断价值高得多（不用再为了定位去算一次哈希）。
//!
//! ## 「两次相同」这条判据有两个窟窿（都实测踩过）
//!
//! **一、秒级的时间戳抓不到。** 把 `mvhd` 的 `creation_time` 改成
//! `SystemTime::now()` 的秒数，同一进程内连续两次转换几乎总落在同一秒 ——
//! 5 次里 4 次是绿的（只有恰好跨过秒边界才红）。跨进程也一样，两次子进程启动
//! 加起来不到一秒。所以另加**直接**判据：问「那个字段是不是 0」，而不是
//! 「两次是否相同」（见 `assert_asf_no_timestamp` 与 `creation_times`）。
//!
//! **二、路径可能根本没走到。** 并行那条路的并行度上限是 GOP 数，默认
//! `keyframe_interval = 250` 时 12 帧只有 1 个 GOP，`--threads 2` 会安静地退回
//! 串行 —— 测试全绿，但它只是「把串行跑了两遍」。所以合成源必须设小 GOP，
//! 而且必须**带音频**（音视频归并的错只在有音频时暴露）。
//!
//! 两条合起来的教训：**判据要落在「出问题的那个东西」上**，并且**先确认那条路
//! 真的被走到**。全绿不等于对。
//!
//! ## 跨 `debug` / `release`
//!
//! 测试只能在**一个** profile 里跑，所以「跨 profile 相同」不能自动化。这里的做法是
//! 测试守住「同 profile 内两次相同」，跨 profile 由人手动跑一次并记在 `TODO.md` 里
//! （`cargo test --release -p v2v-cli --test determinism`）。
//! 若哪天 debug 与 release 不同，说明有浮点或迭代顺序依赖 —— 那是**要查的 bug**，
//! 不是「允许的差异」。

use std::path::{Path, PathBuf};

use v2v_asf::{AsfDemuxer, AsfMuxer, AudioFormat, VideoStreamInfo};
use v2v_cli::{convert_file, trim, ConvertOptions, OutputFormat};
use v2v_kernel::mc::Frame;
use v2v_msmpeg4::{Encoder, EncoderConfig};

const W: u32 = 160;
const H: u32 = 120;
const FRAMES: usize = 12;
const GOP: usize = 6;
const MP43: u32 = u32::from_le_bytes(*b"MP43");

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("v2v_det_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("建临时目录");
    d
}

/// 一帧**有运动、有细节**的画面。
///
/// 纯色或静止画面会让编码器一路走 skip 分支，那些「真的在算」的路径根本没被执行 ——
/// 那样的确定性测试是装饰。
fn frame(i: usize) -> Frame {
    let (w, h) = (W as usize, H as usize);
    let mut y = vec![0u8; w * h];
    for yy in 0..h {
        for xx in 0..w {
            // 一个每帧移动的方块（制造整像素运动）+ 斜纹（给 DCT 一点活干）。
            let bx = (xx + i * 5) % w;
            let by = (yy + i * 3) % h;
            let in_box = bx < 32 && by < 24;
            y[yy * w + xx] = if in_box {
                210
            } else {
                (40 + (xx * 3 + yy * 5) % 60) as u8
            };
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    Frame {
        width: W,
        height: H,
        y,
        u: vec![120; cw * ch],
        v: vec![136; cw * ch],
    }
}

/// 造一个**真能解码**的 ASF 源（视频 + 音频直通）。
///
/// 不能用无损测试那种「任意字节当载荷」的样本：这条路要真的解码再编码。
/// 自己编自己解在这条判据里没有循环论证的问题 —— 问的是「两次是否相同」，
/// 不是「解对没解对」。
///
/// ## 音频为什么要带，而且 pts 要**故意落后**视频
///
/// 并行那条路是按「写入序号」把视频帧与音频样本归并回去的。如果源里两者的
/// 写入顺序**恰好等于 pts 顺序**，那么按 pts 归并也一样对 —— 判据就抓不到错。
/// 真实 ASF 里视频包常含**多帧**、音频包更细，于是「先写进去的样本 pts 反而
/// 更大」；这里用一个 pts 落后 20 ms 的音频流复现那个形状。
/// （实测过：不带音频时，`.wmv` 的并行路径即便按 pts 归并也和串行**逐字节相同**，
/// 所以这条测试之前是假绿。）
fn make_source(path: &Path) {
    let mut enc = Encoder::new(EncoderConfig {
        qscale: 6,
        keyframe_interval: GOP,
        ..Default::default()
    });
    let mut m = AsfMuxer::with_audio(
        VideoStreamInfo {
            width: W,
            height: H,
            fourcc: MP43,
        },
        Some(AudioFormat {
            format_tag: 0x0161, // WMA v2
            channels: 2,
            sample_rate: 44_100,
            avg_bytes_per_sec: 16_000,
            block_align: 1024,
            bits_per_sample: 16,
            extra: vec![0; 4],
        }),
    );
    // 音频是**直通**的（转换时原样搬），所以载荷内容无所谓，只要长度稳定。
    let audio = [0u8; 1024];
    for i in 0..FRAMES {
        let vpts = (i * 40) as u32;
        let data = enc.encode_frame(&frame(i)).expect("编码");
        m.write_video(&data, i % GOP == 0, vpts).expect("写视频");
        m.write_audio(&audio, vpts.saturating_sub(20))
            .expect("写音频");
    }
    std::fs::write(path, m.finish()).expect("落盘");
}

fn convert_once(src: &Path, out: &Path, opts: &ConvertOptions) {
    let _ = std::fs::remove_file(out);
    convert_file(src, out, opts).expect("转换失败");
}

fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("读 {} 失败：{e}", p.display()))
}

/// 字节级比对；不同时报出**第几个字节**，省得再去算一次哈希。
fn assert_same(a: &[u8], b: &[u8], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}：长度不同 {} vs {}", a.len(), b.len());
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!(
            "{what}：第 {i} 字节起不同（{} vs {}），共 {} 字节",
            a[i],
            b[i],
            a.len()
        );
    }
}

/// 断言 `.wmv` 产物的**创建时间字段是 0**。
///
/// 「两次运行字节相同」抓不到这个：写当前时间通常是**秒级**的，同进程（甚至跨进程）
/// 连续两次转换几乎总落在同一秒。实测把 `AsfMuxer` 的创建时间改成
/// `SystemTime::now()` 的秒数后，整个测试文件 **6 条一条都没红**。
/// 所以必须直接问「那个字段是什么」，而不是问「两次是否相同」。
fn assert_asf_no_timestamp(path: &Path, what: &str) {
    let f = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("打开 {} 失败：{e}", path.display()));
    let d = AsfDemuxer::new(f)
        .unwrap_or_else(|e| panic!("解析 {} 失败：{e}", path.display()));
    assert_eq!(
        d.info().creation_time,
        0,
        "{what}：ASF 创建时间不是 0（产物里混进了运行时刻）"
    );
}

/// `.wmv`（MP43 in ASF）：同一输入跑两次，产物必须逐字节相同。
#[test]
fn wmv_output_is_byte_identical_across_runs() {
    let d = temp_dir("wmv");
    let src = d.join("src.wmv");
    make_source(&src);

    let opts = ConvertOptions {
        qscale: 6,
        format: OutputFormat::Asf,
        verify: false, // 只比产物字节，不必再解回来算 PSNR
        ..Default::default()
    };
    let a = d.join("a.wmv");
    let b = d.join("b.wmv");
    convert_once(&src, &a, &opts);
    convert_once(&src, &b, &opts);
    assert_same(&read(&a), &read(&b), ".wmv 两次运行");
    assert_asf_no_timestamp(&a, ".wmv");
}

/// 从 MP4 字节里取出某类 box 的 `creation_time`（version 0 的 4 字节）。
///
/// 结构：`size(4) type(4) version(1) flags(3) creation_time(4)`。
///
/// 只认 `size` 看着像 box 长度的命中 —— 否则样本数据里恰好出现 `mvhd` 四个字母
/// 就会被当成 box。本仓只写 version 0，所以不用管 64 位版本。
fn creation_times(mp4: &[u8], kind: &[u8; 4]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 4; // 从 4 起：`type` 不可能是文件的头四个字节
    while i + 12 <= mp4.len() {
        if &mp4[i..i + 4] == kind {
            let size = u32::from_be_bytes([mp4[i - 4], mp4[i - 3], mp4[i - 2], mp4[i - 1]]);
            if size as usize > 8 && size as usize <= mp4.len() {
                out.push(u32::from_be_bytes([
                    mp4[i + 8],
                    mp4[i + 9],
                    mp4[i + 10],
                    mp4[i + 11],
                ]));
            }
        }
        i += 1;
    }
    out
}

/// `.mp4`（mp4v in MP4）：同一输入跑两次，产物必须逐字节相同。
///
/// 光比两次字节**不够**：真实的不确定性里最常见的是「往 `mvhd` 写当前时间」，
/// 而它是**秒级**的 —— 同一进程内连续两次转换大多落在同一秒。实测把
/// `creation_time` 改成 `SystemTime::now()` 的秒数后，这条判据 5 次里 4 次是绿的
/// （只有恰好跨过秒边界才红）。所以这里再钉一条**直接**的判据：那几个字段就是 0。
#[test]
fn mp4_output_is_byte_identical_across_runs() {
    let d = temp_dir("mp4");
    let src = d.join("src.wmv");
    make_source(&src);

    let opts = ConvertOptions {
        qscale: 6,
        format: OutputFormat::Mp4,
        verify: false,
        ..Default::default()
    };
    let a = d.join("a.mp4");
    let b = d.join("b.mp4");
    convert_once(&src, &a, &opts);
    convert_once(&src, &b, &opts);
    let (ba, bb) = (read(&a), read(&b));
    assert_same(&ba, &bb, ".mp4 两次运行");

    // 直接判据：时间字段必须是 0，不靠「两次恰好落在同一秒」。
    for kind in [b"mvhd", b"tkhd", b"mdhd"] {
        let name = String::from_utf8_lossy(kind).into_owned();
        let ts = creation_times(&ba, kind);
        assert!(!ts.is_empty(), "没找到 `{name}` box —— 这条判据失效了");
        assert!(
            ts.iter().all(|&t| t == 0),
            "`{name}` 的 creation_time 非 0：{ts:?}（产物里混进了运行时刻）"
        );
    }
}

/// **串行与 GOP 并行在 `.wmv` 上必须给出同一份字节。**
///
/// 这是 `--threads` 文档里**明确承诺过**的一条，所以要有判据守着：并行那条路是
/// 「每个 GOP 一个 worker、按顺序拼回去」，只要有任何一处依赖「谁先算完」，输出就
/// 会随线程数变 —— 而它不会报错，用户只会以为两次转出来的画质设置不一样。
///
/// 用 `.wmv` 而不是 `.mp4`，是因为 `.mp4` **不承诺**这件事：GOP 边界会重置运动
/// 估计与码控，`--threads 2` 的产物与串行差几字节到几十 KB（见 `--threads` 的帮助
/// 文本）。把「不承诺的东西」钉成必须相同，就是在给一个会随参数漂的实现立碑。
#[test]
fn parallel_matches_serial() {
    let d = temp_dir("par");
    let src = d.join("src.wmv");
    make_source(&src);

    let base = ConvertOptions {
        qscale: 6,
        format: OutputFormat::Asf,
        // ⚠️ **必须设小 GOP**：默认 `keyframe_interval = 250`，12 帧的源只有 **1 个
        // GOP** —— 而并行度的上限就是 GOP 数，`threads: 2` 会安静地退回串行，这条
        // 测试就成了「把串行跑了两遍」（实测过：不设它，测试全绿但并行路径没走）。
        keyframe_interval: GOP,
        verify: false,
        ..Default::default()
    };
    let one = d.join("t1.wmv");
    let two = d.join("t2.wmv");
    convert_once(
        &src,
        &one,
        &ConvertOptions {
            threads: 1,
            ..base.clone()
        },
    );
    convert_once(
        &src,
        &two,
        &ConvertOptions {
            threads: 2,
            ..base
        },
    );
    assert_same(&read(&one), &read(&two), "--threads 1 与 --threads 2");
}

/// **跨进程**：真的起两次 `v2v` 子进程，产物必须逐字节相同。
///
/// 上面几条都在**同一个进程**里跑两次 —— 这对最常见的那类不确定性是**不够的**：
/// 「往容器里写当前时间」通常是**秒级**的，同进程内两次转换几乎总落在同一秒，
/// 判据会假绿。只有跨进程（中间隔着进程启动、CLI 解析、读输入）才真的把时间推
/// 过去，也才顺带覆盖「进程级初始化留下的状态」。
#[test]
fn separate_processes_produce_identical_bytes() {
    let d = temp_dir("proc");
    let src = d.join("src.wmv");
    make_source(&src);

    // 两个输出目录 → 两份产物文件名不同、内容必须相同（`output_path_for` 只在
    // 与输入同路径时才插 `.conv`，这里路径不同，所以名字都是 `src.wmv`）。
    let bin = env!("CARGO_BIN_EXE_v2v");
    let mut outs = Vec::new();
    for tag in ["pa", "pb"] {
        let out_dir = d.join(tag);
        std::fs::create_dir_all(&out_dir).expect("建输出目录");
        let st = std::process::Command::new(bin)
            .arg("convert")
            .args(["-q", "6", "--no-verify", "-o"])
            .arg(&out_dir)
            .arg(&src)
            .output()
            .expect("起 v2v 子进程");
        assert!(
            st.status.success(),
            "子进程 {tag} 失败（{}）：{}",
            st.status,
            String::from_utf8_lossy(&st.stderr)
        );
        outs.push(out_dir.join("src.wmv"));
    }
    assert_same(&read(&outs[0]), &read(&outs[1]), "两个进程的产物");
    for (i, p) in outs.iter().enumerate() {
        assert_asf_no_timestamp(p, &format!("子进程 {i} 的产物"));
    }
}

/// 无损剪切也必须是确定的（只搬样本，本不该有任何随机性 —— 但「本不该」要验）。
///
/// 区间取 `[0, 240)`：源的关键帧在 0 / 240，`end_ms` 是**半开**的，所以正好剪下
/// 前 6 帧。别用 `start=40, end=240` —— 起点会向后对齐到关键帧 240，跟半开右端
/// 撞在一起变成空区间（那是「对齐行为」该单独测的事，不是这条）。
#[test]
fn trim_is_byte_identical_across_runs() {
    let d = temp_dir("trim");
    let src = d.join("src.wmv");
    make_source(&src);

    let a = d.join("cut_a.wmv");
    let b = d.join("cut_b.wmv");
    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
    let ra = trim(&src, &a, 0, Some(240)).expect("剪切 a");
    let rb = trim(&src, &b, 0, Some(240)).expect("剪切 b");
    // 空产物之间也「字节相同」，所以必须钉住真的搬了东西。
    assert_eq!(ra.video_samples, GOP, "剪出来的视频样本数不对");
    assert_eq!(ra.video_samples, rb.video_samples, "两次剪出的样本数不同");
    assert_same(&read(&a), &read(&b), "trim 两次运行");
}

/// 同一进程内**连续多次**转换：排除「进程内累积状态」这类不确定性
/// （例如某个全局计数器、`OnceCell` 缓存）。
#[test]
fn repeated_runs_in_one_process_stay_identical() {
    let d = temp_dir("rep");
    let src = d.join("src.wmv");
    make_source(&src);

    let opts = ConvertOptions {
        qscale: 6,
        format: OutputFormat::Asf,
        verify: false,
        ..Default::default()
    };
    let first = d.join("first.wmv");
    convert_once(&src, &first, &opts);
    let want = read(&first);
    for i in 0..3 {
        let out = d.join(format!("again{i}.wmv"));
        convert_once(&src, &out, &opts);
        assert_same(&want, &read(&out), &format!("第 {i} 次重跑"));
    }
}
