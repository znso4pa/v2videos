//! MPEG-1 / MPEG-2（`.mpg` / `.dat` 那种 program stream）的**顺时播放**。
//!
//! 与 [`crate::Player`]（ASF + WMV3 + WMA）并列，**不共用**实现：容器、编解码器、
//! 有没有音频都不一样，硬塞进一个结构只会让两边都变糊。
//!
//! ## 为什么单独一个 feature
//!
//! 宿主打 `.so` 时要的是**最小体积**：只吃 WMV 的宿主不该背上 MPEG-1/2 的码表。
//! 所以这一整块挂在 `mpeg12` feature 下，默认**不开**：
//!
//! ```toml
//! v2v-player = { path = "...", default-features = false, features = ["mpeg12"] }
//! ```
//!
//! ## 音频
//!
//! **不解**。MPEG-1 的音轨是 MP2（MPEG-1 Audio Layer II），本仓没有它的解码器；
//! 宿主想要声音得自己接。这里只出视频帧，`has_audio` 恒为 false —— 不假装有。
//!
//! ## 显示序
//!
//! 解码序 ≠ 显示序（B 帧要等后面的参考帧）。这里按 **GOP 缓冲**：解到下一个 I 帧
//! 就把上一个 GOP 按 `temporal_reference` 排好一次交出去。GOP 通常十几帧，
//! 1080p 也就几十 MB，比「整片进内存」稳得多。

use std::collections::VecDeque;
use std::path::Path;

use v2v_mpeg12::{PictureType, SequenceHeader, decode, split_pictures, split_units};
use v2v_mpegps::PsDemuxer;

use crate::{Error, Result, VideoFrame};

/// 一个 GOP 的缓冲上限：超过就当作「没有 I 帧的坏流」，明确报错而不是吃满内存。
const MAX_GOP_FRAMES: usize = 600;

/// MPEG-1/2 顺时播放器（**只有视频**）。
pub struct MpgPlayer {
    sequence: SequenceHeader,
    /// 还没解的 picture（解码序）。
    pictures: VecDeque<(v2v_mpeg12::PictureHeader, Vec<u8>)>,
    /// 解码序的两路参考帧。
    last_ip: Option<decode::Frame>,
    prev_ip: Option<decode::Frame>,
    /// 当前 GOP 已解、待按显示序交出的帧。
    pending: Vec<(u32, decode::Frame)>,
    /// 这一批里哪些 tr 是 I 帧（交出时要带 `key_frame`）。
    pending_key: std::collections::BTreeMap<u32, bool>,
    /// 已交出多少帧（= 显示序的下标，也是 PTS 的基准）。
    emitted: u64,
    /// `pending` 里第一帧的显示序号起点。
    gop_base: u64,
    /// 已经摊平、正在按显示序吐出的那一段有多长（吐完再把 `gop_base` 往前推）。
    flushing: Option<u64>,
    frame_ms: f64,
    width: u32,
    height: u32,
}

impl MpgPlayer {
    /// 打开一个 MPEG program stream。整文件读进内存 —— PS 没有索引，
    /// 逐包扫一遍才知道每帧在哪，边扫边解反而更慢（实测素材都是几十 MB 级）。
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| Error::Bad(format!("读不了文件：{e}")))?;
        Self::from_bytes(&bytes)
    }

    /// 同上，但从内存里的字节开始（测试用）。
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut demux = PsDemuxer::new(bytes.to_vec());
        let mut es = Vec::new();
        while let Some(packet) = demux.next_packet() {
            if packet.is_video() {
                es.extend_from_slice(&demux.data()[packet.range.clone()]);
            }
        }
        if es.is_empty() {
            return Err(Error::Bad("program stream 里没有视频".into()));
        }

        let units = split_units(&es);
        let mut sequence = units
            .iter()
            .find(|(code, _)| *code == 0xB3)
            .and_then(|(_, payload)| SequenceHeader::parse(payload))
            .ok_or_else(|| Error::Bad("没有 sequence header（不是 MPEG-1/2 视频？）".into()))?;
        sequence.mpeg2 = v2v_mpeg12::has_sequence_extension(&es);
        if sequence.mpeg2 {
            if let Some(ext) = units
                .iter()
                .find(|(code, _)| *code == 0xB5)
                .and_then(|(_, payload)| v2v_mpeg12::SequenceExtension::parse(payload))
            {
                sequence.apply_sequence_extension(&ext);
            }
        }

        let pictures: VecDeque<(_, Vec<u8>)> = split_pictures(&es)
            .into_iter()
            .map(|(header, data)| (header, data.to_vec()))
            .collect();
        if pictures.is_empty() {
            return Err(Error::Bad("没有 picture start code".into()));
        }
        let frame_ms = if sequence.frame_rate_milli > 0 {
            1000.0 / (sequence.frame_rate_milli as f64 / 1000.0)
        } else {
            1000.0 / 30.0
        };
        let (width, height) = (sequence.width, sequence.height);
        Ok(Self {
            sequence,
            pictures,
            last_ip: None,
            prev_ip: None,
            pending: Vec::new(),
            pending_key: std::collections::BTreeMap::new(),
            emitted: 0,
            gop_base: 0,
            flushing: None,
            frame_ms,
            width,
            height,
        })
    }

    /// 画面尺寸与帧率（毫秒/帧）。
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn frame_ms(&self) -> f64 {
        self.frame_ms
    }

    /// 视频总帧数（picture 个数，含 B 帧）。
    pub fn picture_count(&self) -> usize {
        self.pictures.len()
    }

    /// 取下一帧（**显示序**）。`None` = 放完了。
    pub fn next_video(&mut self) -> Result<Option<VideoFrame>> {
        loop {
            if let Some(frame) = self.take_pending() {
                return Ok(Some(frame));
            }
            // 缓冲空了 —— 解下一个 GOP。
            if self.pictures.is_empty() {
                // 收尾：把最后一段按显示序吐出去。
                if !self.pending.is_empty() {
                    self.flush_gop();
                    continue;
                }
                return Ok(None);
            }
            self.decode_next_picture()?;
        }
    }

    /// 缓冲里按显示序取一帧。
    fn take_pending(&mut self) -> Option<VideoFrame> {
        // ⚠️ **只有摊平过的那一段才能交**：解码序里 B 帧排在它的参考帧后面，
        // 一边解一边交就会把 P 帧先交出去（实测第 1 帧就差 229）。
        if self.flushing.is_none() {
            return None;
        }
        if self.pending.is_empty() {
            // 这一段吐完了 —— 把显示序的起点往前推，等下一段。
            if let Some(len) = self.flushing.take() {
                self.gop_base += len;
                self.pending_key.clear(); // 这一段吐完了，关键帧表跟着作废
            }
            return None;
        }
        self.pending.sort_by_key(|(tr, _)| *tr);
        let (tr, frame) = self.pending.remove(0);
        let pts_ms = ((self.gop_base + u64::from(tr)) as f64 * self.frame_ms).round() as i64;
        self.emitted += 1;
        Some(VideoFrame {
            pts_ms,
            // I 帧就是这里的关键帧（seek 的落点候选）。
            key_frame: self.pending_key.get(&tr).copied().unwrap_or(false),
            width: self.width,
            height: self.height,
            y: frame.y,
            u: frame.u,
            v: frame.v,
        })
    }

    /// 把当前 GOP 按显示序摊平（新的 I 帧到来 / EOS 时调）。
    ///
    /// ⚠️ **不能在这里推 `gop_base`**：这些帧还没交出去，它们的 PTS 用的就是
    /// 当前这个 base。推到 `take_pending` 吐完最后一帧时再做。
    fn flush_gop(&mut self) {
        self.pending.sort_by_key(|(tr, _)| *tr);
        let n = self
            .pending
            .last()
            .map(|(tr, _)| u64::from(*tr) + 1)
            .unwrap_or(0)
            .max(self.pending.len() as u64);
        self.flushing = Some(n);
    }

    /// 解一张 picture（解码序）。遇到新的 I 帧就把上一段摊平。
    fn decode_next_picture(&mut self) -> Result<()> {
        // 新 GOP 的起点：**先摊平上一段、把这张 I 帧留在队里**（别 pop，否则丢帧）。
        if let Some((header, _)) = self.pictures.front() {
            if matches!(header.picture_type, PictureType::I) && !self.pending.is_empty() {
                self.flush_gop();
                return Ok(());
            }
        }
        let (header, data) = self.pictures.pop_front().expect("调用方查过非空");
        let kind = header.picture_type;
        let tr = header.temporal_reference;
        if matches!(kind, PictureType::D) {
            return Ok(());
        }
        let forward = if matches!(kind, PictureType::B) {
            self.prev_ip.as_ref()
        } else {
            self.last_ip.as_ref()
        };
        let backward = self.last_ip.as_ref();
        let frame = decode::decode_frame(&self.sequence, header, &data, forward, backward)
            .map_err(|e| Error::Bad(format!("MPEG 解码失败：{e:?}")))?;
        if matches!(kind, PictureType::I | PictureType::P) {
            self.prev_ip = self.last_ip.take();
            self.last_ip = Some(frame.clone());
        }
        if matches!(kind, PictureType::I) {
            self.pending_key.insert(tr, true);
        }
        self.pending.push((tr, frame));
        if self.pending.len() > MAX_GOP_FRAMES {
            return Err(Error::Bad("一个 GOP 超过 600 帧，疑似没有 I 帧的坏流".into()));
        }
        Ok(())
    }
}
