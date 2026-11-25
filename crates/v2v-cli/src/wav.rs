//! 44 字节 RIFF/WAVE 头（`fmt ` PCM + `data`），16 bit 小端、声道交织。
//!
//! 与 `crates/v2v-android/src/wma_wav.rs` 里那份**逐字段相同** —— 两处各有一份是
//! 因为它们的宿主不同（一个给 Android 的 JNI 层出 `Vec<u8>`，一个给 CLI 落盘），
//! 谁依赖谁都不对。布局是规范死的，改一处就要改另一处；两边的测试都在。
//!
//! **只写不读**：本仓没有 WAV 解封装（输入侧不需要，`.wav` 不是转换的目标源）。

/// 44 字节头 + 交织 PCM 拼成一个完整的 `.wav`。
pub fn wrap_pcm16(channels: u16, sample_rate: u32, pcm: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(&header(channels, sample_rate, pcm.len() as u32));
    out.extend_from_slice(pcm);
    out
}

fn header(channels: u16, sample_rate: u32, data_size: u32) -> [u8; 44] {
    const BYTES_PER_SAMPLE: u16 = 2;
    let block_align = channels.saturating_mul(BYTES_PER_SAMPLE);
    let byte_rate = sample_rate.saturating_mul(u32::from(block_align));

    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&(36u32.saturating_add(data_size)).to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");

    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes()); // fmt chunk 长度
    h[20..22].copy_from_slice(&1u16.to_le_bytes()); // audio_format = PCM
    h[22..24].copy_from_slice(&channels.to_le_bytes());
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    h[32..34].copy_from_slice(&block_align.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits_per_sample

    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_size.to_le_bytes());
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16_at(h: &[u8], i: usize) -> u16 {
        u16::from_le_bytes([h[i], h[i + 1]])
    }
    fn u32_at(h: &[u8], i: usize) -> u32 {
        u32::from_le_bytes([h[i], h[i + 1], h[i + 2], h[i + 3]])
    }

    #[test]
    fn header_layout_is_the_standard_one() {
        let h = header(2, 48_000, 1000);
        assert_eq!(&h[0..4], b"RIFF");
        assert_eq!(u32_at(&h, 4), 36 + 1000);
        assert_eq!(&h[8..12], b"WAVE");
        assert_eq!(&h[12..16], b"fmt ");
        assert_eq!(u32_at(&h, 16), 16);
        assert_eq!(u16_at(&h, 20), 1);
        assert_eq!(u16_at(&h, 22), 2);
        assert_eq!(u32_at(&h, 24), 48_000);
        assert_eq!(u32_at(&h, 28), 48_000 * 4);
        assert_eq!(u16_at(&h, 32), 4);
        assert_eq!(u16_at(&h, 34), 16);
        assert_eq!(&h[36..40], b"data");
        assert_eq!(u32_at(&h, 40), 1000);
    }

    /// 单声道也要对：`block_align` / `byte_rate` 都随声道数变，写死 2/4 会错。
    #[test]
    fn mono_uses_half_the_block_align() {
        let h = header(1, 44_100, 0);
        assert_eq!(u16_at(&h, 32), 2);
        assert_eq!(u32_at(&h, 28), 44_100 * 2);
    }

    #[test]
    fn wrap_pcm16_prefixes_exactly_44_bytes() {
        let w = wrap_pcm16(2, 8000, &[1, 2, 3, 4]);
        assert_eq!(w.len(), 48);
        assert_eq!(&w[44..], &[1, 2, 3, 4]);
        assert_eq!(u32_at(&w, 40), 4);
    }
}
