//! 临时：把整条音频轨解成 f32 落盘，用来做「改动前后逐字节对账」。
//!
//! ```text
//! cargo run --release -p v2v-player --example pcm_dump -- <file.wmv> <out.bin>
//! ```
//!
//! 写的是 `f32::to_bits()` 的原始字节，不是 i16 —— 定标/裁剪那一步会掩盖
//! 浮点末位的差异，而我们要看的正是末位。

use std::path::Path;

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    let (Some(path), Some(out)) = (args.next(), args.next()) else {
        eprintln!("用法: pcm_dump <file> <out.bin>");
        return std::process::ExitCode::FAILURE;
    };
    let mut player = match v2v_player::Player::open(Path::new(&path)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("打开失败：{e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let mut bytes = Vec::new();
    let mut frames = 0usize;
    loop {
        match player.next_audio() {
            Ok(Some(a)) => {
                frames += 1;
                for plane in &a.planes {
                    for v in plane {
                        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("解码失败：{e}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }
    if let Err(e) = std::fs::write(&out, &bytes) {
        eprintln!("写失败：{e}");
        return std::process::ExitCode::FAILURE;
    }
    eprintln!("{frames} 个音频帧 / {} 字节 → {out}", bytes.len());
    std::process::ExitCode::SUCCESS
}
