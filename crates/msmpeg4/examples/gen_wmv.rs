//! 生成一个小的测试 .wmv（编码 + 封装），给 ffprobe/ffmpeg 手动检查用。
use std::io::Write;
use v2v_asf::{AsfMuxer, VideoStreamInfo};
use v2v_msmpeg4::{Encoder, EncoderConfig, Frame};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args().nth(1).unwrap_or("/tmp/loop.wmv".into());
    let (w, h) = (320u32, 240u32);
    let (wu, hu) = (w as usize, h as usize);
    let mut enc = Encoder::new(EncoderConfig { qscale: 8, ..Default::default() });
    let mut mux = AsfMuxer::new(VideoStreamInfo { width: w, height: h, fourcc: u32::from_le_bytes(*b"MP43") });
    for i in 0..6usize {
        let mut y = vec![0u8; wu * hu];
        for r in 0..hu { for c in 0..wu {
            let v = 100 + (((c + i * 3) / 8) % 4) as i32 * 30 + ((r / 16) % 2) as i32 * 10;
            y[r * wu + c] = v.clamp(16, 235) as u8;
        }}
        let ox = 40 + i * 6; let oy = 30 + i * 3;
        for r in 0..40usize { for c in 0..40usize {
            let yy = oy + r; let xx = ox + c;
            if yy < hu && xx < wu { y[yy * wu + xx] = if (r / 4 + c / 4) % 2 == 0 { 30 } else { 220 }; }
        }}
        let (cw, ch) = (wu / 2, hu / 2);
        let u = vec![128u8; cw * ch];
        let mut v = vec![128u8; cw * ch];
        for r in 0..ch { for c in 0..cw { v[r * cw + c] = (128 - (r / 8) as i32 * 5 + (i as i32) * 2).clamp(16, 240) as u8; }}
        let f = Frame { width: w, height: h, y, u, v };
        let s = enc.encode_frame(&f)?;
        mux.write_video(&s, i == 0, (i as u32) * 33)?;
    }
    std::fs::File::create(&out)?.write_all(&mux.finish())?;
    println!("written {out}");
    Ok(())
}
