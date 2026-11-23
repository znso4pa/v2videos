use std::{fs, path::Path, process::Command};

fn pcm(path: &Path) -> Vec<i16> {
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-threads", "1", "-i"])
        .arg(path)
        .args(["-map", "0:a:0", "-f", "s16le", "-"])
        .output()
        .expect("ffmpeg");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
        .stdout
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

#[test]
fn mp3_in_avi_matches_ffmpeg_pcm() {
    let Some(paths) = std::env::var_os("V2V_MP3_AVI_SAMPLE") else {
        eprintln!("SKIP：请设 V2V_MP3_AVI_SAMPLE 指向带 MP3 音轨的 AVI");
        return;
    };
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("SKIP：找不到 ffmpeg");
        return;
    }
    let dir = std::env::temp_dir().join(format!("v2v_mp3_{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();

    for src in std::env::split_paths(&paths) {
        assert!(
            src.is_file(),
            "V2V_MP3_AVI_SAMPLE 不是文件：{}",
            src.display()
        );
        let output = Command::new(env!("CARGO_BIN_EXE_v2v"))
            .args(["convert", "--mp4", "--pcm", "--threads", "1", "-o"])
            .arg(&dir)
            .arg(&src)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let out = dir.join(src.file_stem().unwrap()).with_extension("mp4");
        let expected = pcm(&src);
        let actual = pcm(&out);
        fs::remove_file(&out).unwrap();

        let mut demux = v2v_avi::AviDemuxer::new(fs::File::open(&src).unwrap()).unwrap();
        let (stream, channels) = demux
            .streams()
            .iter()
            .find_map(|s| {
                s.audio.as_ref().map(|a| {
                    assert_eq!(a.format_tag, v2v_mp3::FORMAT_TAG);
                    (s.number, a.channels as usize)
                })
            })
            .unwrap();
        let mut points = 0;
        let mut framer = None;
        while let Some(s) = demux.next_sample().unwrap() {
            if s.stream != stream || s.data.is_empty() {
                continue;
            }
            // 帧长度由音频头确定。只读帧头，独立于两个解码后端。
            let framer = framer.get_or_insert_with(|| {
                let h = v2v_mp3::Header::parse(&s.data).unwrap();
                v2v_mp3::Framer::new(h.channels, h.sample_rate).unwrap()
            });
            framer
                .feed(&s.data, |h, _| {
                    points += h.samples;
                    Ok(())
                })
                .unwrap();
        }
        framer.unwrap().finish().unwrap();
        assert_eq!(expected.len(), points * channels);
        // v2v 的 MP4 输出会裁掉比视频更长的音频尾巴。样本应选音视频近似等长的 AVI。
        assert!(
            actual.len() >= expected.len() * 99 / 100,
            "PCM 长度：{} / {}",
            actual.len(),
            expected.len()
        );
        assert!(actual.len() <= expected.len());
        let max_diff = actual
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap();
        eprintln!(
            "PCM: {} / {} 个值，最大误差 {max_diff} LSB",
            actual.len(),
            expected.len()
        );
        assert!(max_diff <= 2, "PCM 最大误差 {max_diff} LSB");
    }
    fs::remove_dir(&dir).unwrap();
}
