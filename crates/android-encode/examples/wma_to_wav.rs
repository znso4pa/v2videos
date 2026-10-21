//! 把 WMV/ASF 的 WMA 音轨抽成 WAV（PCM16LE、交织）。
//!
//! 用法：`cargo run -p v2v-android --example wma_to_wav -- <in.wmv> <out.wav>`
//!
//! 纯 Rust 解码，不解视频、不重编码。给「自带 WMA 解码器慢到没法用」的引擎
//! 当替换音频路径用。

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <in.wmv> <out.wav>", args[0]);
        std::process::exit(2);
    }
    let in_p = Path::new(&args[1]);

    let has = v2v_android::wma_wav::has_wma_audio(in_p).expect("探测音轨");
    if !has {
        eprintln!("这个文件没有可解的 WMA 音轨");
        std::process::exit(1);
    }

    let bytes = v2v_android::wma_wav::wma_to_wav_bytes(in_p).expect("解码 WMA");
    std::fs::write(&args[2], &bytes).expect("写 WAV");

    // 44 字节头 + 数据；立体声 16 bit = 4 字节/帧。
    let secs = (bytes.len().saturating_sub(44)) as f64 / 176_400.0;
    println!("wrote {} bytes (~{:.1} s audio)", bytes.len(), secs);
}
