//! 帧头的对账测试：**真样本 + `ffmpeg -debug pict`**。
//!
//! 判据不是「没报错」，而是逐帧对齐三样东西：
//! `pict_type`、以及两张位平面的 `(imode, invert)`。
//!
//! 为什么这三样够硬：位平面紧跟在帧头的标量之后 —— 标量少读/多读**一位**，
//! 位平面立刻变成别的 imode 或直接解不出来。所以「位平面全对」等价于
//! 「帧头解到宏块层起点时，位游标一位不差」。
//!
//! ⚠️ 日志里**每帧的行序是「先位平面、后 `pict_type`」**（`pict_type` 那行在
//! `vc1_decode_frame()` 里、帧头解完之后才打），夹具已按这个顺序攒帧。
//! 反了会整体错开一帧 —— 而且因为相邻帧常常同类型，看起来「差不多对」。

mod common;

use v2v_vc1::{FrameHeader, HeaderState, PictType, SeqHeader};

/// 逐帧比帧头。
#[test]
fn frame_headers_match_ffmpeg_pict_log() {
    let samples = common::sample_list();
    if samples.is_empty() {
        eprintln!("SKIP: 没设 V2V_ASF_SAMPLE");
        return;
    }
    if !common::has_ffmpeg() {
        eprintln!("SKIP: 没装 ffmpeg");
        return;
    }

    let mut checked = 0usize;
    let mut total_frames = 0usize;
    for path in &samples {
        let Some(track) = common::video_track(path) else {
            eprintln!("SKIP {path}: 取不到视频轨");
            continue;
        };
        if track.fourcc != common::FOURCC_WMV3 {
            eprintln!("SKIP {path}: 不是 WMV3（fourcc {:#x}）", track.fourcc);
            continue;
        }
        let seq = SeqHeader::parse(&track.extra)
            .unwrap_or_else(|e| panic!("{path}: 序列头解析失败: {e}"));
        // 宏块网格来自**容器**（Simple/Main 的序列头里没有尺寸）。
        let mb_w = ((track.encoded_width + 15) / 16) as usize;
        let mb_h = ((track.encoded_height + 15) / 16) as usize;

        let frames = common::all_video_frames(path, track.number)
            .unwrap_or_else(|| panic!("{path}: 取不到视频帧"));
        let (profile, log) = common::ffmpeg_pict_log(path)
            .unwrap_or_else(|| panic!("{path}: 拿不到 ffmpeg 的 pict 日志"));

        // 序列头的第一个字段就有现成的 oracle。
        assert_eq!(
            seq.profile as u8, profile,
            "{path}: profile 与 ffmpeg 不一致"
        );
        // ⚠️ **先逐帧比字段、后比帧数**：帧数一差就提前退出的话，字段对不对就看不见了。
        // （帧数本身可能因为「ffmpeg 多解了一帧」这类日志/收尾差异而不等，
        // 那是另一回事，不该把字段的对账挡住。）
        let n = frames.len().min(log.len());
        let mut state = HeaderState::default();
        for (i, (data, want)) in frames.iter().zip(log.iter()).take(n).enumerate() {
            let h = FrameHeader::parse(data, &seq, mb_w, mb_h, &mut state)
                .unwrap_or_else(|e| panic!("{path}: 第 {i} 帧帧头解析失败: {e}"));

            let got = match h.pict_type {
                PictType::I => 'I',
                PictType::P => 'P',
            };
            assert_eq!(got, want.pict_type, "{path}: 第 {i} 帧 pict_type");

            if let Some(p) = &h.p {
                match (want.mv_type_plane, &p.mv_type_plane) {
                    (Some((imode, invert)), Some(plane)) => {
                        assert_eq!(
                            plane.imode as u8, imode,
                            "{path}: 第 {i} 帧 MV 类型位平面的 imode"
                        );
                        assert_eq!(
                            plane.invert as u8, invert,
                            "{path}: 第 {i} 帧 MV 类型位平面的 invert"
                        );
                    }
                    (None, None) => {}
                    (Some(_), None) => panic!("{path}: 第 {i} 帧 ffmpeg 有 MV 类型位平面，我们没解"),
                    (None, Some(_)) => panic!("{path}: 第 {i} 帧 ffmpeg 没有 MV 类型位平面，我们解了一张"),
                }
                let (imode, invert) = want
                    .skip_plane
                    .unwrap_or_else(|| panic!("{path}: 第 {i} 帧 ffmpeg 没打跳过位平面"));
                assert_eq!(
                    p.skip_plane.imode as u8, imode,
                    "{path}: 第 {i} 帧跳过位平面的 imode"
                );
                assert_eq!(
                    p.skip_plane.invert as u8, invert,
                    "{path}: 第 {i} 帧跳过位平面的 invert"
                );
            } else {
                assert!(
                    want.mv_type_plane.is_none() && want.skip_plane.is_none(),
                    "{path}: 第 {i} 帧是 I 帧，ffmpeg 却打了位平面行"
                );
            }
        }
        assert_eq!(
            frames.len(),
            log.len(),
            "{path}: 帧数不一致（我们 {} / ffmpeg {}，已比过前 {n} 帧）",
            frames.len(),
            log.len()
        );
        eprintln!(
            "{path}: {} 帧逐帧对齐（{}×{}，宏块网格 {}×{}，profile {:?}）",
            frames.len(),
            track.encoded_width,
            track.encoded_height,
            mb_w,
            mb_h,
            seq.profile
        );
        checked += 1;
        total_frames += frames.len();
    }
    assert!(
        checked > 0,
        "给了样本但没有一个是 WMV3 —— 没有真的对账过（V2V_ASF_SAMPLE 里没有 WMV3 样本？）"
    );
    eprintln!("合计 {checked} 个样本 / {total_frames} 帧逐帧对齐");
}
