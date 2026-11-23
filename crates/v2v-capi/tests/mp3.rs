#![cfg(any(feature = "mp3", feature = "mp3-ffi"))]

use std::ptr::null;
use v2v_capi::{adec::*, EOS, OK};

#[test]
fn mp3_feed_accepts_fragmented_frames_and_preserves_sample_clock() {
    // MPEG-1 Layer III，128 kbit/s，44100 Hz 双声道。空的主数据解成静音。
    let mut frame = [0_u8; 417];
    frame[..4].copy_from_slice(&[0xff, 0xfb, 0x90, 0x00]);
    let cfg = V2vAdecCfg {
        size: size_of::<V2vAdecCfg>() as u32,
        format_tag: 0x0055,
        channels: 2,
        sample_rate: 44100,
        avg_bytes_per_sec: 16000,
        block_align: 1,
        extra: null(),
        extra_len: 0,
        drop_priming: 1,
    };
    let mut pcm = V2vAdecFrame {
        size: size_of::<V2vAdecFrame>() as u32,
        planes: null(),
        channels: 0,
        samples: 0,
        sample_rate: 0,
        pts_ms: 0,
    };
    unsafe {
        let h = v2v_adec_open(&cfg);
        assert!(!h.is_null());
        assert_eq!(v2v_adec_feed(h, frame.as_ptr(), 3), OK);
        assert_eq!(v2v_adec_next(h, &mut pcm), EOS);

        let mut rest = frame[3..].to_vec();
        rest.extend_from_slice(&frame);
        assert_eq!(v2v_adec_feed(h, rest.as_ptr(), rest.len() as u32), OK);
        for pts in [0, 26] {
            assert_eq!(v2v_adec_next(h, &mut pcm), OK);
            assert_eq!(
                (pcm.pts_ms, pcm.sample_rate, pcm.samples, pcm.channels),
                (pts, 44100, 1152, 2)
            );
            for ch in 0..2 {
                let plane = std::slice::from_raw_parts(*pcm.planes.add(ch), pcm.samples as usize);
                assert!(plane.iter().all(|&v| v == 0.0));
            }
        }
        assert_eq!(v2v_adec_next(h, &mut pcm), EOS);
        v2v_adec_close(h);
    }
}
