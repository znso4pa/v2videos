//! ASF **封装**（muxer）：把裸基本流样本写回 `.wmv`。
//!
//! 目前只做**单视频流**（MP43 闭环够用）；音频与多流待做。
//!
//! ## 与 ffmpeg muxer 的字节级对应
//!
//! 布局逐字段对照 `ffmpeg` 的 `asfenc.c`（`asf_write_header1` /
//! `put_payload_parsing_info` / `put_payload_header` / `put_frame`）：
//!
//! - **Header Object**：File Properties + Stream Properties + Header Extension
//!   （空的）。Codec Comment / 内容描述是可选对象，不写。
//! - **数据包**：每包恰好 `PACKET_SIZE`（3200）字节；EC 头 `0x82` + 2 字节 EC 数据；
//!   **单载荷**（不做多载荷打包 —— 合法 ASF，只是每包少塞几个 payload）；
//!   载荷头 15 字节 = 流号 | 关键帧位 + 媒体对象号 + 对象内偏移 + 8 字节
//!   replicated data（对象总长 + 显示时间）。
//! - **帧大于包容量时分片**：同一帧的分片共用媒体对象号、偏移递增 ——
//!   解封装端按这个重组（我们的解包器与 ffmpeg 都这么做）。
//! - **padding**：包内剩余空间按 ASF 规范写 padding 长度字段（<256 用 1 字节、
//!   否则 2 字节）；`chunk == capacity - 1` 时缩 1 字节 —— 与 ffmpeg 同款
//!   off-by-one 回避（padding 字段本身会吃掉 1 字节）。
//!
//! ## 已知边界
//!
//! - 全部在内存里缓冲（`finish()` 返回整个文件）。流式写出待做 —— 对批量转换
//!   的体积没影响（WMV 本来就要整个头部已知包数才能填），只影响内存峰值。
//! - 不写 Simple Index：播放器 seek 会退化到逐包扫。待做。

use crate::error::{Error, Result};
use crate::guid::{self, Guid};
use crate::types::AudioFormat;

/// 数据包大小。ffmpeg 的 muxer 默认 3200。
const PACKET_SIZE: usize = 3200;
/// preroll（毫秒）。载荷的显示时间与包的发送时间都按它偏移 —— 与 ffmpeg 一致。
const PREROLL_MS: u32 = 3100;
/// 单载荷的载荷头长度：流号(1) + 对象号(1) + 偏移(4) + replicated 长度(1) + replicated(8)。
const PAYLOAD_HEADER_SIZE: usize = 15;
/// 包头最小长度：EC(1+2) + 长度类型(1) + 属性(1) + 发送时间(4) + 时长(2)。
const PACKET_HEADER_MIN_SIZE: usize = 11;
/// 单载荷包里载荷的最大长度。
const PAYLOAD_CAPACITY: usize = PACKET_SIZE - PACKET_HEADER_MIN_SIZE - PAYLOAD_HEADER_SIZE;

/// 视频流的封装参数。
#[derive(Debug, Clone)]
pub struct VideoStreamInfo {
    pub width: u32,
    pub height: u32,
    /// FourCC，如 `'MP43'`（`u32::from_le_bytes(*b"MP43")`）。
    pub fourcc: u32,
}

/// ASF 封装器（单视频流）。
///
/// 用法：`write_video()` 逐帧喂（**解码序**，I 帧在前），最后 `finish()` 拿整个文件。
pub struct AsfMuxer {
    file_id: Guid,
    video: VideoStreamInfo,
    /// 音频流（有就一起复用；目前只做**直通**，不重编码）。
    audio: Option<AudioFormat>,
    /// 视频的媒体对象号（1 起，每帧 +1）。
    seq: u8,
    /// 音频的媒体对象号（独立计数）。
    seq_audio: u8,
    packets: Vec<Vec<u8>>,
    /// 已见到的最大显示时间（毫秒，不含 preroll）。
    max_pts_ms: u32,
}

impl AsfMuxer {
    /// 只复用视频。
    pub fn new(video: VideoStreamInfo) -> Self {
        Self::with_audio(video, None)
    }

    /// 复用视频 + 音频。`audio` 是源文件的 WAVEFORMATEX（**原样**写进流属性，
    /// 所以音频必须是**直通**：写进去的载荷要和它匹配）。
    pub fn with_audio(video: VideoStreamInfo, audio: Option<AudioFormat>) -> Self {
        Self {
            file_id: [0u8; 16],
            video,
            audio,
            seq: 1,
            seq_audio: 1,
            packets: Vec::new(),
            max_pts_ms: 0,
        }
    }

    /// 写一个音频载荷（**直通**：调用方给的就是源的载荷字节）。
    ///
    /// 音频流的号是 2（视频是 1）。音频载荷一般不大，整块塞进一个包；
    /// 超过单包容量的会自动分片（与视频同一条路径）。
    pub fn write_audio(&mut self, data: &[u8], pts_ms: u32) -> Result<()> {
        if self.audio.is_none() {
            return Err(Error::Bad("没声明音频流，却要写音频载荷"));
        }
        if data.is_empty() {
            return Ok(());
        }
        self.max_pts_ms = self.max_pts_ms.max(pts_ms);
        let mut offset = 0usize;
        while offset < data.len() {
            let mut chunk = (data.len() - offset).min(PAYLOAD_CAPACITY);
            if chunk == PAYLOAD_CAPACITY - 1 {
                chunk -= 1;
            }
            let pad = PACKET_SIZE - PACKET_HEADER_MIN_SIZE - PAYLOAD_HEADER_SIZE - chunk;
            let mut p = Vec::with_capacity(PACKET_SIZE);
            p.push(0x82);
            p.push(0);
            p.push(0);
            match pad {
                0 => p.push(0x00),
                1..=255 => p.push(0x08),
                _ => p.push(0x10),
            }
            p.push(0x5D);
            match pad {
                1..=255 => p.push((pad - 1) as u8),
                _ if pad >= 256 => p.extend_from_slice(&((pad - 2) as u16).to_le_bytes()),
                _ => {}
            }
            p.extend_from_slice(&pts_ms.to_le_bytes());
            p.extend_from_slice(&0u16.to_le_bytes());
            self.payload(&mut p, data, offset, chunk, false, pts_ms, 2, self.seq_audio)?;
            p.resize(PACKET_SIZE, 0);
            self.packets.push(p);
            offset += chunk;
        }
        self.seq_audio = self.seq_audio.wrapping_add(1);
        Ok(())
    }

    /// 写一帧视频。`pts_ms` 是显示时间（毫秒，**不含** preroll —— 封装时统一加）。
    pub fn write_video(&mut self, data: &[u8], key_frame: bool, pts_ms: u32) -> Result<()> {
        if data.is_empty() {
            return Ok(()); // 空样本没有媒体对象可写
        }
        self.max_pts_ms = self.max_pts_ms.max(pts_ms);

        let mut offset = 0usize;
        while offset < data.len() {
            let mut chunk = (data.len() - offset).min(PAYLOAD_CAPACITY);
            // ffmpeg 的 off-by-one 回避：chunk 占满 `capacity - 1` 时 padding 字段
            // 会把账挤爆，缩 1 字节。
            if chunk == PAYLOAD_CAPACITY - 1 {
                chunk -= 1;
            }
            let pad = PACKET_SIZE - PACKET_HEADER_MIN_SIZE - PAYLOAD_HEADER_SIZE - chunk;

            // ---- 包头（布局逐字段对应 put_payload_parsing_info）----
            let mut p = Vec::with_capacity(PACKET_SIZE);
            p.push(0x82); // 错误纠正存在 + EC 数据 2 字节
            p.push(0);
            p.push(0);
            // 长度类型标志：无序列 / 无包长；有 padding 才写 padding 字段
            match pad {
                0 => p.push(0x00),
                1..=255 => p.push(0x08), // padding 长度 = 1 字节
                _ => p.push(0x10),       // padding 长度 = 2 字节
            }
            p.push(0x5D); // 属性标志：对象号 1 字节 / 偏移 4 字节 / replicated 1 字节 / 流号 1 字节
            match pad {
                1..=255 => p.push((pad - 1) as u8),
                _ if pad >= 256 => p.extend_from_slice(&((pad - 2) as u16).to_le_bytes()),
                _ => {}
            }
            p.extend_from_slice(&pts_ms.to_le_bytes()); // 发送时间（不含 preroll）
            p.extend_from_slice(&0u16.to_le_bytes()); // 时长：单载荷恒 0

            // ---- 载荷 ----
            self.payload(&mut p, data, offset, chunk, key_frame, pts_ms, 1, self.seq)?;
            debug_assert!(p.len() <= PACKET_SIZE);
            p.resize(PACKET_SIZE, 0); // padding 补齐
            self.packets.push(p);

            offset += chunk;
        }
        self.seq = self.seq.wrapping_add(1);
        Ok(())
    }

    /// 写一个载荷（头 + 数据）到包缓冲 `p`。
    #[allow(clippy::too_many_arguments)]
    fn payload(
        &self,
        p: &mut Vec<u8>,
        data: &[u8],
        offset: usize,
        chunk: usize,
        key_frame: bool,
        pts_ms: u32,
        stream: u8,
        seq: u8,
    ) -> Result<()> {
        p.push((if key_frame { 0x80 } else { 0x00 }) | stream);
        p.push(seq);
        p.extend_from_slice(&(offset as u32).to_le_bytes());
        p.push(0x08); // replicated data 长度（恒 8：对象总长 + 显示时间）
        p.extend_from_slice(&(data.len() as u32).to_le_bytes());
        p.extend_from_slice(&(pts_ms.wrapping_add(PREROLL_MS)).to_le_bytes());
        p.extend_from_slice(&data[offset..offset + chunk]);
        Ok(())
    }

    /// 收尾：拼出整个文件。
    pub fn finish(self) -> Vec<u8> {
        let nb_packets = self.packets.len() as u64;
        let data_size = 50 + nb_packets * PACKET_SIZE as u64;

        // ---- File Properties ----
        let send_dur = (self.max_pts_ms as u64 + PREROLL_MS as u64) * 10_000;
        let play_dur = self.max_pts_ms as u64 * 10_000;
        let fp_body = {
            let mut b = Vec::with_capacity(80);
            b.extend_from_slice(&self.file_id);
            let file_size = 0u64; // 下面统一算，先占位
            let _ = file_size;
            b.extend_from_slice(&0u64.to_le_bytes()); // 文件大小，见下
            b.extend_from_slice(&0u64.to_le_bytes()); // 创建时间
            b.extend_from_slice(&nb_packets.to_le_bytes());
            b.extend_from_slice(&send_dur.to_le_bytes());
            b.extend_from_slice(&play_dur.to_le_bytes());
            b.extend_from_slice(&(PREROLL_MS as u64).to_le_bytes());
            b.extend_from_slice(&2u32.to_le_bytes()); // 标志：非广播
            b.extend_from_slice(&(PACKET_SIZE as u32).to_le_bytes());
            b.extend_from_slice(&(PACKET_SIZE as u32).to_le_bytes());
            b.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // 最大码率：未知
            b
        };

        // ---- Stream Properties（视频）----
        let sp_body = {
            let mut b = Vec::with_capacity(78 + 51);
            b.extend_from_slice(&guid::VIDEO_MEDIA);
            b.extend_from_slice(&video_conceal_none());
            b.extend_from_slice(&0u64.to_le_bytes()); // 时间偏移
            b.extend_from_slice(&51u32.to_le_bytes()); // type-specific 长度
            b.extend_from_slice(&0u32.to_le_bytes()); // 纠错数据长度
            b.extend_from_slice(&1u16.to_le_bytes()); // 流号
            b.extend_from_slice(&0u32.to_le_bytes()); // 保留
            // type-specific：宽/高/标志/格式长度 + BITMAPINFOHEADER(40)
            b.extend_from_slice(&self.video.width.to_le_bytes());
            b.extend_from_slice(&self.video.height.to_le_bytes());
            b.push(2); // 标志（与 ffmpeg muxer 同值）
            b.extend_from_slice(&40u16.to_le_bytes());
            b.extend_from_slice(&40u32.to_le_bytes()); // biSize
            b.extend_from_slice(&(self.video.width as i32).to_le_bytes());
            b.extend_from_slice(&(self.video.height as i32).to_le_bytes());
            b.extend_from_slice(&1u16.to_le_bytes()); // planes
            b.extend_from_slice(&24u16.to_le_bytes()); // bpp
            b.extend_from_slice(&self.video.fourcc.to_le_bytes()); // 压缩类型
            b.extend_from_slice(&((self.video.width * self.video.height * 3) as u32).to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes()); // xppm
            b.extend_from_slice(&0u32.to_le_bytes()); // yppm
            b.extend_from_slice(&0u32.to_le_bytes()); // clr used
            b.extend_from_slice(&0u32.to_le_bytes()); // clr important
            b
        };

        // ---- Header Extension（空）----
        let he_body = {
            let mut b = Vec::with_capacity(22);
            b.extend_from_slice(&head2());
            b.extend_from_slice(&6u16.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes()); // 扩展数据长度
            b
        };

        // ---- Stream Properties（音频，可选）----
        // 子对象数要用到它，所以先算。
        let sp_audio: Option<Vec<u8>> = self.audio.as_ref().map(|a| {
            let mut wfx = Vec::with_capacity(18 + a.extra.len());
            wfx.extend_from_slice(&a.format_tag.to_le_bytes());
            wfx.extend_from_slice(&a.channels.to_le_bytes());
            wfx.extend_from_slice(&a.sample_rate.to_le_bytes());
            wfx.extend_from_slice(&a.avg_bytes_per_sec.to_le_bytes());
            wfx.extend_from_slice(&a.block_align.to_le_bytes());
            wfx.extend_from_slice(&a.bits_per_sample.to_le_bytes());
            wfx.extend_from_slice(&(a.extra.len() as u16).to_le_bytes());
            wfx.extend_from_slice(&a.extra);

            let mut b = Vec::with_capacity(64 + wfx.len());
            b.extend_from_slice(&audio_media());
            b.extend_from_slice(&audio_conceal_spread());
            b.extend_from_slice(&0u64.to_le_bytes()); // 时间偏移
            b.extend_from_slice(&(wfx.len() as u32).to_le_bytes());
            b.extend_from_slice(&8u32.to_le_bytes()); // 纠错数据长度
            b.extend_from_slice(&2u16.to_le_bytes()); // 流号 2
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(&wfx);
            // 纠错数据（与 ffmpeg 的写法一致：0x01 + block_align×2 + 0x01 + 0x00）
            b.push(0x01);
            b.extend_from_slice(&a.block_align.to_le_bytes());
            b.extend_from_slice(&a.block_align.to_le_bytes());
            b.extend_from_slice(&1u16.to_le_bytes());
            b.push(0x00);
            b
        });

        let mut out = Vec::with_capacity(200 + data_size as usize);
        out.extend_from_slice(&guid::HEADER);
        let header_size_pos = out.len();
        out.extend_from_slice(&0u64.to_le_bytes()); // 头长度，下面回填
        out.extend_from_slice(&(3u32 + u32::from(sp_audio.is_some())).to_le_bytes()); // 子对象数
        out.push(1);
        out.push(2);
        let header_body_start = out.len();
        out.extend_from_slice(&object(guid::FILE_PROPERTIES, &fp_body));
        out.extend_from_slice(&object(guid::STREAM_PROPERTIES, &sp_body));
        if let Some(a) = &sp_audio {
            out.extend_from_slice(&object(guid::STREAM_PROPERTIES, a));
        }
        out.extend_from_slice(&object(guid::HEADER_EXTENSION, &he_body));
        let header_size = (out.len() - header_body_start + 30) as u64;
        out[header_size_pos..header_size_pos + 8]
            .copy_from_slice(&header_size.to_le_bytes());

        // ---- Data Object ----
        out.extend_from_slice(&guid::DATA);
        out.extend_from_slice(&data_size.to_le_bytes());
        out.extend_from_slice(&self.file_id);
        out.extend_from_slice(&nb_packets.to_le_bytes());
        out.push(1);
        out.push(1);

        // ---- 数据包 ----
        for p in &self.packets {
            out.extend_from_slice(p);
        }

        // 回填 File Properties 里的文件总大小（= 当前总长）。
        let file_size = out.len() as u64;
        let fp_at = header_body_start + 24; // object 头 24 字节之后
        out[fp_at + 16..fp_at + 24].copy_from_slice(&file_size.to_le_bytes());

        out
    }
}

/// 一个 ASF 对象：GUID + 总长（含头）+ 主体。
fn object(g: Guid, body: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(24 + body.len());
    v.extend_from_slice(&g);
    v.extend_from_slice(&((body.len() + 24) as u64).to_le_bytes());
    v.extend_from_slice(body);
    v
}

/// `ff_asf_video_conceal_none`（解封装用不到，封装要用）。
fn video_conceal_none() -> Guid {
    [
        0x00, 0x57, 0xFB, 0x20, 0x55, 0x5B, 0xCF, 0x11, 0xA8, 0xFD, 0x00, 0x80, 0x5F, 0x5C, 0x44,
        0x2B,
    ]
}

/// `ff_asf_audio_media`（音频流的类型 GUID）。
fn audio_media() -> Guid {
    [
        0x40, 0x9E, 0x69, 0xF8, 0x4D, 0x5B, 0xCF, 0x11, 0xA8, 0xFD, 0x00, 0x80, 0x5F, 0x5C, 0x44,
        0x2B,
    ]
}

/// `ff_asf_audio_conceal_spread`（音频流的纠错类型 GUID）。
fn audio_conceal_spread() -> Guid {
    [
        0x50, 0xCD, 0xC3, 0xBF, 0x8F, 0x61, 0xCF, 0x11, 0x8B, 0xB2, 0x00, 0xAA, 0x00, 0xB4, 0xE2,
        0x20,
    ]
}

/// `ff_asf_head2_guid`（Header Extension 的保留字段）。
fn head2() -> Guid {
    [
        0x11, 0xD2, 0xD3, 0xAB, 0xBA, 0xA9, 0xCF, 0x11, 0x8E, 0xE6, 0x00, 0xC0, 0x0C, 0x20, 0x53,
        0x65,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AsfDemuxer;
    use std::io::Cursor;

    #[test]
    fn roundtrips_through_our_own_demuxer() {
        let mut m = AsfMuxer::new(VideoStreamInfo {
            width: 64,
            height: 48,
            fourcc: u32::from_le_bytes(*b"MP43"),
        });
        // 帧 0：大帧（跨包分片）；帧 1：小帧（单包 + padding）；帧 2：恰好 capacity-1 附近
        let big: Vec<u8> = (0..9000).map(|i| (i * 7) as u8).collect();
        let small = vec![42u8; 100];
        let mid: Vec<u8> = (0..3172).map(|i| (i * 13) as u8).collect();
        m.write_video(&big, true, 0).unwrap();
        m.write_video(&small, false, 40).unwrap();
        m.write_video(&mid, false, 80).unwrap();
        let file = m.finish();

        let mut d = AsfDemuxer::new(Cursor::new(&file)).expect("解封装自己写的文件");
        assert_eq!(d.streams().len(), 1);
        let s = &d.streams()[0];
        assert_eq!(s.kind, crate::StreamKind::Video);
        let vf = s.video.as_ref().expect("视频格式");
        assert_eq!(vf.width, 64);
        assert_eq!(vf.height, 48);
        assert_eq!(vf.fourcc, u32::from_le_bytes(*b"MP43"));

        let mut got = Vec::new();
        while let Some(s) = d.next_sample().unwrap() {
            got.push(s);
        }
        assert_eq!(got.len(), 3);
        assert!(got[0].key_frame);
        assert!(!got[1].key_frame);
        assert_eq!(got[0].data, big);
        assert_eq!(got[1].data, small);
        assert_eq!(got[2].data, mid);
    }

    #[test]
    fn packet_count_matches_header_and_each_packet_is_full_size() {
        let mut m = AsfMuxer::new(VideoStreamInfo {
            width: 32,
            height: 32,
            fourcc: u32::from_le_bytes(*b"MP43"),
        });
        let f0: Vec<u8> = (0..9000).map(|i| (i * 3) as u8).collect(); // 3 包
        let f1: Vec<u8> = vec![1u8; 10]; // 1 包
        m.write_video(&f0, true, 0).unwrap();
        m.write_video(&f1, false, 33).unwrap();
        let file = m.finish();

        // Data Object 头里的包数 = 实际包数；每包恰好 PACKET_SIZE。
        let mut d = AsfDemuxer::new(Cursor::new(&file)).expect("解封装");
        assert_eq!(d.info().packet_count, 4, "9000→3 包 + 10→1 包");
        // 文件总大小应与 File Properties 里写的一致
        assert_eq!(d.info().file_size, file.len() as u64);
        let mut n = 0u64;
        while d.next_sample().unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, 2);
    }
}
