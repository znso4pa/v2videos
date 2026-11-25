//! Ogg 解封装**真样本**对账。
//!
//! ```text
//! V2V_OGG_REF_SAMPLE=/path/to/ref.ogg cargo test -p v2v-ogg --test real_sample -- --nocapture
//! ```
//!
//! ⚠️ 这里的**期望值是写死在一份具体样本上的**（88 个包 / 2001 ms / 首包 172 字节），
//! 所以它读 `V2V_OGG_REF_SAMPLE` 而不是共用的 `V2V_OGG_SAMPLE` —— 见
//! `ref_sample_path()` 的说明。
//!
//! ## 对账基准从哪来
//!
//! 期望值全部来自 `ffprobe -show_packets`（同一份样本），不是从本实现反推的：
//!
//! - 包大小：`packet=size`
//! - 时间戳：`packet=pts`（44.1 kHz、常量长块下每包 1024 采样）
//! - 总时长：`format=duration` / 末页 `granule_position`
//!
//! ## 一个必须说清的差异：extradata 长度
//!
//! `ffprobe` 报 `extradata_size = 3308`，本实现是 **3336**（30 + 59 + 3247，
//! 三个头包原样拼接）。**差的 28 字节不是丢了数据** —— ffmpeg 会把三个头
//! **重新序列化**成 Xiph lacing 格式，并且顺手把注释头里的 vendor 串从
//! `Lavf63.1.101` + `encoder=Lavc63.1.101 vorbis` 改写成更短的
//! `Lavc63.1.101`、丢掉 `encoder` 标签。那是 ffmpeg 自己的封装选择，
//! 我们不该跟着改文件内容（下游解码器要的是**原字节**）。
//! 这里用「三个头包各自的魔数在拼接结果里的位置」来钉住这件事。

use std::fs::File;

use v2v_ogg::{duration_ms, OggDemuxer, PacketTiming, StreamKind, VORBIS_FORMAT_TAG};

/// 测试用的「包时长」：这个样本全是常量长块（2048），推进量恒为 1024。
///
/// 真实的 Vorbis 实现要从 setup 头里读模式表（本仓还没有 Vorbis 解码器，
/// 见 README 的能力矩阵），这里只为了把**容器**这一层单独验干净。
struct ConstantLongBlock;

impl PacketTiming for ConstantLongBlock {
    fn advance(&self, _prev: &[u8], _cur: &[u8]) -> Option<u32> {
        Some(1024)
    }
}

/// 这条测试的**期望值是写死在某一份具体样本上的**（88 个包、2001 ms、首包 172 字节…），
/// 所以它读的是**另一个**环境变量 `V2V_OGG_REF_SAMPLE`，不是 `V2V_OGG_SAMPLE`。
///
/// ⚠️ 为什么必须分开（2026-11-26 踩的）：`V2V_OGG_SAMPLE` 被 `ogg_wav.rs` /
/// `ogg_mp4.rs` / `remux_real.rs` / `demux_bench.rs` 共用，那几条**对样本无要求**。
/// 用同一份 600 s 的样本跑全仓测试时，这条就红在「25841 个包 vs 88 个」——
/// 看着像解封装坏了，其实只是**换了素材**。判据挑的样本是判据的一部分。
fn ref_sample_path() -> Option<String> {
    match std::env::var("V2V_OGG_REF_SAMPLE") {
        Ok(p) if std::path::Path::new(&p).is_file() => Some(p),
        _ => {
            eprintln!("SKIP: V2V_OGG_REF_SAMPLE 未设置或文件不存在");
            None
        }
    }
}

#[test]
fn demuxes_every_packet_of_the_real_sample() {
    let Some(path) = ref_sample_path() else { return };
    let file = File::open(&path).unwrap();

    // 从第一页起就校验 CRC —— 真样本必须全过，顺带把 CRC 实现钉在真实数据上。
    let mut d = OggDemuxer::open_with(file, true).unwrap();
    d.set_timing(Box::new(ConstantLongBlock));

    assert_eq!(d.streams().len(), 1, "单音频流");
    let s = &d.streams()[0];
    assert_eq!(s.kind, StreamKind::Audio);
    let a = s.audio.as_ref().expect("音频流要有格式");
    assert_eq!(a.format_tag, VORBIS_FORMAT_TAG);
    assert_eq!(a.channels, 2);
    assert_eq!(a.sample_rate, 44100);

    // 三个头包原样拼接：按长度字段拆开，长度分别是 30 / 59 / 3247。
    // 用解析器拆而不是靠搜魔数 —— 注释头里是自由文本，理论上可以含 `\x05vorbis`。
    let (id_pkt, cm_pkt, setup_pkt) =
        v2v_ogg::split_vorbis_headers(&a.extra).expect("三个头包要拆得开");
    assert_eq!((id_pkt.len(), cm_pkt.len(), setup_pkt.len()), (30, 59, 3247));
    assert_eq!(&id_pkt[..7], b"\x01vorbis");
    assert_eq!(&cm_pkt[..7], b"\x03vorbis");
    assert_eq!(&setup_pkt[..7], b"\x05vorbis");
    assert_eq!(a.extra.len(), 30 + 59 + 3247);
    assert_eq!(&a.extra[..7], b"\x01vorbis");
    assert_eq!(&a.extra[30..37], b"\x03vorbis");
    assert_eq!(&a.extra[89..96], b"\x05vorbis");

    let id = d.vorbis_id().expect("Vorbis 识别头");
    // 这份样本是 **ffmpeg 原生 vorbis 编码器**产的，它的块大小字节是 0xBB ——
    // 短块和长块**都是 2048**，也就是说它从不切换块大小。
    // 记在这里是因为它决定了这份样本能验什么、不能验什么：
    // **短窗、窗形切换、块大小过渡这几条路径它一条都走不到**
    // （和 AAC 那次「本仓编码器只产 ONLY_LONG，于是短窗三条路径从没被执行」同一个坑）。
    assert_eq!(id.short_block(), 2048);
    assert_eq!(id.long_block(), 2048);

    let stream_number = d.streams()[0].number;
    let mut sizes = Vec::new();
    let mut pts_ms = Vec::new();
    while let Some(smp) = d.next_sample().unwrap() {
        assert_eq!(smp.stream, stream_number);
        assert!(smp.key_frame);
        sizes.push(smp.data.len());
        pts_ms.push(smp.timestamp_ms);
    }

    // 88 个音频包（ffprobe: nb_read_packets 同口径），首尾大小对得上。
    assert_eq!(sizes.len(), 88);
    assert_eq!(sizes[0], 172);
    assert_eq!(sizes[87], 206);
    assert_eq!(sizes.iter().sum::<usize>(), 11598 - (30 + 59 + 3247));

    // 时间戳：第 1 包 -1024 采样，之后每包 +1024。
    // 换成毫秒是 -23 / 0 / 23 / 46 …（四舍五入，与 ffmpeg 的 av_rescale 同口径）。
    assert_eq!(pts_ms[0], -23, "首包 pts");
    assert_eq!(pts_ms[1], 0);
    assert_eq!(pts_ms[2], 23);
    assert_eq!(pts_ms[3], 46);
    for (i, &ms) in pts_ms.iter().enumerate() {
        let expected_samples = (i as i64 - 1) * 1024;
        let expected_ms = {
            let num = expected_samples * 1000;
            if num >= 0 {
                (num + 22050) / 44100
            } else {
                -((-num + 22050) / 44100)
            }
        };
        assert_eq!(ms, expected_ms, "第 {i} 包时间戳");
    }

    // 末包 pts = 88064 采样（ffprobe 同值）→ 输出结束在 89088。
    // 但末页 granule 是 88256：编码器把最后 832 个采样标成可丢弃
    // （ffprobe 报 `Skip Samples 0 832`）。两者相差正好 832，这条把
    // 「累加器」和「权威总长」两个口径钉开 —— 它们**本来就不该相等**。
    assert_eq!(d.samples_produced(), 89088);
    assert_eq!(d.total_samples(), Some(88256));
    assert_eq!(d.samples_produced() - d.total_samples().unwrap() as i64, 832);

    assert_eq!(d.crc_errors(), 0, "真样本不该有 CRC 错");
    assert_eq!(d.timing_misses(), 0);
}

#[test]
fn duration_matches_ffprobe() {
    let Some(path) = ref_sample_path() else { return };
    let mut f = File::open(&path).unwrap();
    let ms = duration_ms(&mut f).unwrap().expect("末页 granule");
    // ffprobe: duration=2.001270（88256 采样 @ 44100）
    assert_eq!(ms, 2001, "总时长");
}

#[test]
fn non_ogg_input_is_rejected() {
    let bad = b"RIFF\x00\x00\x00\x00AVI LIST".to_vec();
    let e = OggDemuxer::open(&bad[..]).err().expect("应当报错");
    let s = e.to_string();
    assert!(s.contains("Ogg"), "{s}");
}

/// 与具体编码器无关的**结构不变量**：任何 Ogg 文件都得满足。
///
/// 用 `V2V_OGG_SAMPLE_EXTRA` 喂额外的样本（`:` 分隔），比如另一种编码器产的
/// 同内容文件 —— 那份的块大小会真的切换，包长分布完全不同，正好用来
/// 验「包重组不是只对一种封装方式成立」。
///
/// 这里**不装 timing**，所以时间戳恒为 0：这一条只管包层。
#[test]
fn packet_layer_invariants_hold_on_any_sample() {
    let mut paths: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("V2V_OGG_SAMPLE") {
        if std::path::Path::new(&p).is_file() {
            paths.push(p);
        }
    }
    if let Ok(list) = std::env::var("V2V_OGG_SAMPLE_EXTRA") {
        paths.extend(list.split(':').filter(|p| std::path::Path::new(p).is_file()).map(String::from));
    }
    if paths.is_empty() {
        eprintln!("SKIP: V2V_OGG_SAMPLE / V2V_OGG_SAMPLE_EXTRA 都未设置");
        return;
    }

    for path in &paths {
        let file = File::open(path).unwrap();
        let mut d = OggDemuxer::open_with(file, true).unwrap();
        let rate = d.sample_rate();
        assert!(rate > 0, "{path}: 采样率为 0");

        let mut n = 0usize;
        let mut bytes = 0usize;
        while let Some(s) = d.next_sample().unwrap() {
            assert!(!s.data.is_empty(), "{path}: 第 {n} 个包是空的");
            n += 1;
            bytes += s.data.len();
        }
        assert!(n > 0, "{path}: 一个音频包都没有");
        assert_eq!(d.crc_errors(), 0, "{path}: CRC 错");
        assert_eq!(d.discarded_partials(), 0, "{path}: 有半截包被丢");
        assert_eq!(d.sequence_gaps(), 0, "{path}: 页序号有跳变");

        let total = d.total_samples().expect("末页 granule");
        assert!(total > 0, "{path}: 总采样数为 0");
        // 总长不可能超过「包数 × 长块」—— 一包最多推进一个长块。
        // 超过说明包重组把某一页算了两遍，或者 granule 读错了符号。
        let long = d.vorbis_id().map(|i| i.long_block()).unwrap_or(2048) as u64;
        assert!(
            total <= n as u64 * long,
            "{path}: 总长 {total} 采样 > {n} 包 × 长块 {long}"
        );
        eprintln!("{path}: {n} 包 / {bytes} 字节 / 总长 {total} 采样 / {rate} Hz");
    }
}
