//! 对真样本的验证：拆出的视频 elementary stream 必须与 ffmpeg 抽出的**逐字节相同**。
//!
//! ```bash
//! MPEGPS_TEST_FILE=<clip>.mpg MPEGPS_TEST_ES=<clip>.m1v \
//!   cargo test -p mpegps --test real_stream -- --ignored
//! ```

use v2v_mpegps::PsDemuxer;

#[test]
#[ignore = "needs MPEGPS_TEST_FILE (a .mpg) and MPEGPS_TEST_ES (the same video as .m1v)"]
fn extracted_video_stream_matches_ffmpeg() {
    let ps = std::env::var("MPEGPS_TEST_FILE").expect("set MPEGPS_TEST_FILE");
    let es = std::env::var("MPEGPS_TEST_ES").expect("set MPEGPS_TEST_ES");
    let file = std::fs::read(&ps).expect("read the program stream");
    let expected = std::fs::read(&es).expect("read the elementary stream");

    let mut demux = PsDemuxer::new(file);
    let mut video = Vec::new();
    let mut packets = 0usize;
    let mut with_pts = 0usize;
    while let Some(packet) = demux.next_packet() {
        if !packet.is_video() {
            continue;
        }
        packets += 1;
        if packet.pts_ms.is_some() {
            with_pts += 1;
        }
        video.extend_from_slice(&demux.data()[packet.range.clone()]);
    }

    assert!(packets > 0, "no video packets found");
    assert_eq!(
        video.len(),
        expected.len(),
        "{packets} video packets, {} with a PTS: extracted {} bytes, ffmpeg {}",
        with_pts,
        video.len(),
        expected.len()
    );
    if let Some(at) = video.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!("first difference at byte {at}: {:02x?} vs {:02x?}", video[at], expected[at]);
    }
    assert!(video.starts_with(&[0, 0, 1, 0xB3]), "video should start with a sequence header");
}
