//! 真样本上验证帧头解析。
//!
//! 关键交叉校验：**ASF 里标了关键帧的帧，必须是 I 帧；反之亦然。**
//! 这两个信息来自完全独立的解析路径（容器层 vs 编码层），能对上才说明帧头解析是对的。
//!
//! 设 `V2V_ASF_SAMPLE=<路径>` 才真跑；缺样本打 `SKIP` 并通过。

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use v2v_asf::{AsfDemuxer, StreamKind};
use v2v_msmpeg4::{mb_count, parse_frame_header, PictureType};

#[test]
fn frame_headers_agree_with_the_container() {
    let path = match std::env::var("V2V_ASF_SAMPLE") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SKIP frame_headers_agree_with_the_container: V2V_ASF_SAMPLE 未设置");
            return;
        }
    };
    if !Path::new(&path).is_file() {
        eprintln!("SKIP frame_headers_agree_with_the_container: 样本不存在");
        return;
    }

    let mut d = AsfDemuxer::new(BufReader::new(File::open(&path).expect("open"))).expect("asf");
    let (video_number, width, height, fourcc) = {
        let video = d
            .streams()
            .iter()
            .find(|s| s.kind == StreamKind::Video)
            .expect("视频流");
        let vf = video.video.as_ref().expect("视频格式");
        (
            video.number,
            vf.width,
            vf.height,
            v2v_asf::fourcc_string(vf.fourcc),
        )
    };
    // 这个测试只针对 MS-MPEG-4 v3（MP43）。
    assert_eq!(fourcc, "MP43", "这个测试只对 MP43 有意义");
    let (mb_w, mb_h) = mb_count(width, height);

    let mut i_frames = 0u64;
    let mut p_frames = 0u64;
    let mut mismatches = 0u64;
    let mut bad_qscale = 0u64;

    while let Some(s) = d.next_sample().expect("demux") {
        if s.stream != video_number {
            continue;
        }
        let (h, used) = parse_frame_header(&s.data, mb_w, mb_h).expect("帧头必须能解析");
        assert!(used > 0 && used <= 8 * s.data.len());
        if !(1..=31).contains(&h.qscale) {
            bad_qscale += 1;
        }
        match h.picture_type {
            PictureType::I => i_frames += 1,
            PictureType::P => p_frames += 1,
        }
        let is_i = h.picture_type == PictureType::I;
        if is_i != s.key_frame {
            mismatches += 1;
        }
    }

    assert_eq!(bad_qscale, 0, "有帧的 qscale 落在 1..=31 之外");
    assert!(i_frames > 0 && p_frames > 0, "应当同时有 I 帧和 P 帧");
    assert_eq!(
        mismatches, 0,
        "I 帧与容器关键帧不一致：{mismatches} 帧（I={i_frames} P={p_frames}）"
    );

    eprintln!("帧头对账通过：I={i_frames} P={p_frames}，与容器关键帧完全一致");
}
