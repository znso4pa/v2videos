//! WMA **真样本**对账：拿 ASF 里那条音频轨的每个载荷，跑 superframe 头解析。
//!
//! ```text
//! V2V_WMA_SAMPLE=<带 WMA 音轨的 .wmv/.asf> cargo test -p v2v-wma -- --nocapture
//! ```
//!
//! ## 这一段能验什么、不能验什么
//!
//! **能验**：参数推导（帧长 / `byte_offset_bits` / 噪声编码开关）、superframe 头的
//! 位序与字段位置、序号是否逐包递增、`bit_offset` 是否落在包内。
//! 还能验一条**跨层一致性**：推出来的帧数 × 帧长 ÷ 采样率 ≈ 音频时长
//! （由「文件里音频字节数 ÷ 码率」算出）—— 两边独立算出来的秒数必须接近。
//!
//! **不能验**：精确帧数。superframe 末尾会把「没用完的位」留给下一包接着解，
//! 残留长度取决于帧内实际消耗了多少位 —— 那要等帧解码器。

use std::fs::File;
use std::io::BufReader;

use v2v_asf::{AsfDemuxer, StreamKind};
use v2v_wma::{SuperframeState, WmaParams};

#[test]
fn parses_every_audio_payload_of_the_real_sample() {
    let path = match std::env::var("V2V_WMA_SAMPLE") {
        Ok(p) if std::path::Path::new(&p).is_file() => p,
        _ => {
            eprintln!("SKIP: V2V_WMA_SAMPLE 未设置或文件不存在");
            return;
        }
    };

    let mut d = AsfDemuxer::new(BufReader::new(File::open(&path).expect("开文件"))).expect("解封装");
    let audio = d
        .streams()
        .iter()
        .find(|s| s.kind == StreamKind::Audio)
        .expect("音频轨");
    let af = audio.audio.as_ref().expect("音频格式");
    let num = audio.number;

    // 码率：WAVEFORMATEX 里没有 bit_rate 字段，用 avg_bytes_per_sec × 8（与 ffprobe 同口径）
    let bit_rate = af.avg_bytes_per_sec * 8;
    let params = WmaParams::new(
        2, // 0x0161
        af.channels as usize,
        af.sample_rate,
        bit_rate,
        af.block_align as usize,
        &af.extra,
    )
    .expect("推参数");
    eprintln!("参数：{params:?}");

    let mut st = SuperframeState::default();
    let mut payloads = 0usize;
    let mut frames = 0i64;
    let mut bytes = 0usize;
    let mut last_index: Option<u32> = None;
    let mut max_bit_offset = 0u32;

    loop {
        let s = match d.next_sample() {
            Ok(Some(s)) => s,
            Ok(None) => break,
            Err(e) => panic!("读样本失败（已读 {payloads} 个音频载荷）：{e}"),
        };
        if s.stream != num {
            continue;
        }
        payloads += 1;
        bytes += s.data.len();
        assert_eq!(
            s.data.len(),
            params.block_align,
            "载荷长度应等于 block_align"
        );

        let h = st.parse(&params, &s.data).expect("superframe 头").expect("有帧");
        // 序号逐包 +1（4 位字段，16 回绕 —— 实测不是 8 回绕）
        if let Some(prev) = last_index {
            assert_eq!(h.index, (prev + 1) % 16, "superframe 序号不连续");
        }
        last_index = Some(h.index);
        // bit_offset 必须落在包内，且第一个完整帧的起点不能越界
        assert!(
            h.frames_start_bit <= s.data.len() * 8,
            "bit_offset 越界：{} > {}",
            h.frames_start_bit,
            s.data.len() * 8
        );
        max_bit_offset = max_bit_offset.max(h.bit_offset);
        assert!((1..=15).contains(&h.nb_frames), "帧数离谱：{}", h.nb_frames);
        frames += h.nb_frames as i64;
    }

    let seconds_by_frames = frames as f64 * params.frame_len as f64 / params.sample_rate as f64;
    let seconds_by_bytes = bytes as f64 * 8.0 / bit_rate as f64;
    eprintln!(
        "载荷 {payloads} 个 / {bytes} 字节；推出帧数 {frames}；\
         bit_offset 最大 {max_bit_offset}\n\
         按时长算：{seconds_by_frames:.2} s（帧数×帧长/采样率） vs \
         {seconds_by_bytes:.2} s（字节数×8/码率）"
    );

    assert!(payloads > 0, "一个音频载荷都没读到");
    assert!(frames > 0, "推出 0 帧");
    // 两条独立路径算出来的时长必须接近（允许 5%：帧的边界会切在包中间）。
    // 实测这个样本：494 帧 → 22.94 s vs 23.40 s（比值 0.98），
    // 而 ffmpeg 解出来的 PCM 正好是 493 帧 —— 三个独立数字互相印证。
    let ratio = seconds_by_frames / seconds_by_bytes;
    assert!(
        (0.95..1.05).contains(&ratio),
        "帧数与字节数算出的时长对不上：{seconds_by_frames:.2} s vs {seconds_by_bytes:.2} s"
    );
}
