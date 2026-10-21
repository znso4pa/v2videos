//! 解码基准：把一个 WMV 从头到尾解完，量速度；可选把产物落盘供对账。
//!
//! ```text
//! cargo run --release -p v2v-player --example bench -- testing/sample.wmv
//! cargo run --release -p v2v-player --example bench -- testing/sample.wmv \
//!     --video-only --yuv /tmp/ours.yuv --pcm /tmp/ours.pcm
//! cargo run --release -p v2v-player --example bench -- testing/sample.wmv --audio-only
//! ```
//!
//! 输出：帧数 / 样本数、耗时、视频 fps、相对时长的倍速（>1 = 快于实时）。
//! `--audio-only` 只走音频解码（量 WMA 自己的速度，别被视频解码掩盖）。

use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use v2v_player::Player;

struct Args {
    path: PathBuf,
    video_only: bool,
    audio_only: bool,
    yuv: Option<PathBuf>,
    pcm: Option<PathBuf>,
}

fn parse_args() -> Option<Args> {
    let mut it = std::env::args().skip(1);
    let path = PathBuf::from(it.next()?);
    let mut a = Args {
        path,
        video_only: false,
        audio_only: false,
        yuv: None,
        pcm: None,
    };
    while let Some(s) = it.next() {
        match s.as_str() {
            "--video-only" => a.video_only = true,
            "--audio-only" => a.audio_only = true,
            "--yuv" => a.yuv = Some(PathBuf::from(it.next()?)),
            "--pcm" => a.pcm = Some(PathBuf::from(it.next()?)),
            _ => return None,
        }
    }
    Some(a)
}

fn main() -> std::io::Result<()> {
    let Some(args) = parse_args() else {
        eprintln!(
            "用法: bench <file.wmv> [--video-only|--audio-only] [--yuv out.yuv] [--pcm out.pcm]"
        );
        std::process::exit(2);
    };

    let mut p = Player::open(&args.path).expect("打开失败");
    let info = p.info().clone();
    println!(
        "{}  {}x{}  {:?}  时长 {} ms  音频 {} Hz × {} 声",
        args.path.display(),
        info.width,
        info.height,
        info.video_fourcc,
        info.duration_ms,
        info.audio_sample_rate,
        info.audio_channels,
    );

    // 4 MiB 缓冲：YUV 一帧就 1.38 MB，默认 8 KB 会把落盘变成纯 syscall 开销
    // （实测全轨模式因此多花 ~8 s，让解码数字失真）。
    let mut yuv = args
        .yuv
        .map(|p| std::io::BufWriter::with_capacity(4 << 20, std::fs::File::create(p).unwrap()));
    let mut pcm = args
        .pcm
        .map(|p| std::io::BufWriter::with_capacity(4 << 20, std::fs::File::create(p).unwrap()));

    let mut vframes = 0u64;
    let mut asamples = 0u64;
    let mut aframes = 0u64;
    let t = Instant::now();
    loop {
        let mut got_video = false;
        let mut got_audio = false;
        if !args.audio_only {
            if let Some(f) = p.next_video().expect("视频解帧出错") {
                vframes += 1;
                got_video = true;
                if let Some(w) = &mut yuv {
                    w.write_all(&f.y)?;
                    w.write_all(&f.u)?;
                    w.write_all(&f.v)?;
                }
            }
        }
        if !args.video_only {
            if let Some(f) = p.next_audio().expect("音频解帧出错") {
                aframes += 1;
                asamples += f.samples as u64;
                got_audio = true;
                if let Some(w) = &mut pcm {
                    // 交织成 f32le（对比 ffmpeg 的 `-f f32le` 口径，ffmpeg 是 ±1，
                    // 我们的解码器输出 ±32768，对账时除回去）。
                    let ch = f.channels;
                    for i in 0..f.samples {
                        for plane in &f.planes {
                            w.write_all(&plane[i].to_le_bytes())?;
                        }
                        let _ = ch;
                    }
                }
            }
        }
        // 结束条件按「这一趟解到了什么」判：`--audio-only` 时视频侧永远是 false。
        if !got_video && (args.video_only || !got_audio) {
            break;
        }
        if args.audio_only && !got_audio {
            break;
        }
    }
    let dt = t.elapsed();

    let dur_s = info.duration_ms as f64 / 1000.0;
    let dec_s = dt.as_secs_f64();
    let a_dur_s = if info.audio_sample_rate > 0 {
        asamples as f64 / f64::from(info.audio_sample_rate)
    } else {
        0.0
    };
    println!(
        "视频 {} 帧 | 音频 {} 帧 / {} 样本 | 解码 {:.2} s | 视频 {:.1} fps | 倍速 {:.2}×",
        vframes,
        aframes,
        asamples,
        dec_s,
        vframes as f64 / dec_s,
        dur_s / dec_s,
    );
    if args.audio_only {
        println!(
            "音频时长 {:.2} s | 音频解码 {:.2} s | 音频倍速 {:.2}×",
            a_dur_s,
            dec_s,
            a_dur_s / dec_s
        );
    }
    Ok(())
}
