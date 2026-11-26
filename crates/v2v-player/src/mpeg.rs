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
//! 需要在支持的平台上启用块内 C/NEON 预测时，改用 `mpeg12-c` feature。
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

use v2v_mpeg12::picture::{DecodedFrame, PictureDecoder};
use v2v_mpeg12::{PictureHeader, SequenceHeader, decode, split_pictures, split_units};
use v2v_mpegps::PsDemuxer;

use crate::{Error, Result, VideoFrame};

/// MPEG-1/2 顺时播放器（**只有视频**）。
pub struct MpgPlayer {
    /// 解码序 → 显示序的状态机。**与 CLI 的 `.mpg` → `.mp4` 转码链路共用同一份**
    /// （`v2v_mpeg12::picture::PictureDecoder`）—— 同一份码流走两条路必须出同样的
    /// 像素和同样的 PTS，所以不能各写一份。
    dec: PictureDecoder,
    /// 还没喂进去的 picture（解码序）。
    pictures: VecDeque<(PictureHeader, Vec<u8>)>,
    /// 已经解好、还没交出去的帧（显示序）。
    ready: VecDeque<DecodedFrame>,
    /// 码流收过尾了（`finish` 调过一次）。
    finished: bool,
    width: u32,
    height: u32,
}

impl MpgPlayer {
    /// 打开一个 MPEG program stream。整文件读进内存 —— PS 没有索引，
    /// 逐包扫一遍才知道每帧在哪，边扫边解反而更慢（实测素材都是几十 MB 级）。
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_threads(path, 1)
    }

    /// 同 [`MpgPlayer::open`]，但用 `threads` 个线程做**切片级**并行解码。
    ///
    /// 并行只在「一帧里有多个 slice」时有效 —— MPEG-1/2 的编码器通常每个宏块行
    /// 一个 slice（实测 720×576 是 36 个），所以基本都吃得到。产物与单线程
    /// **逐字节相同**（`mpeg12` 的 `tests/mt.rs` 钉着这条）。
    ///
    /// ⚠️ 默认是 1：线程数不是越多越好。实测 720×576 上 2 线程 1.16×、
    /// 4 线程 1.72×、**8 线程回落到 1.62×** —— 每帧的 worker 填底色与行合并
    /// 是纯内存流量，核一多就撞带宽。要开就开 4 左右。
    pub fn open_with_threads(path: &Path, threads: usize) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| Error::Bad(format!("读不了文件：{e}")))?;
        Self::from_bytes_with_threads(&bytes, threads)
    }

    /// 同上，但从内存里的字节开始（测试用）。
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_with_threads(bytes, 1)
    }

    /// 同 [`MpgPlayer::from_bytes`]，带线程数（见 [`MpgPlayer::open_with_threads`]）。
    pub fn from_bytes_with_threads(bytes: &[u8], threads: usize) -> Result<Self> {
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
        let (width, height) = (sequence.width, sequence.height);
        #[cfg(feature = "mpeg12-c")]
        v2v_mpeg12_c::install();
        Ok(Self {
            dec: PictureDecoder::new(sequence, threads),
            pictures,
            ready: VecDeque::new(),
            finished: false,
            width,
            height,
        })
    }

    /// 当前块内运动补偿后端的标识，供性能报告记录。
    pub fn motion_compensation_backend(&self) -> &'static str {
        decode::mc8_backend()
    }

    /// 画面尺寸与帧率（毫秒/帧）。
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn frame_ms(&self) -> f64 {
        self.dec.frame_ms()
    }

    /// 视频总帧数（picture 个数，含 B 帧）。
    pub fn picture_count(&self) -> usize {
        self.pictures.len()
    }

    /// 取下一帧（**显示序**）。`None` = 放完了。
    pub fn next_video(&mut self) -> Result<Option<VideoFrame>> {
        loop {
            if let Some(f) = self.ready.pop_front() {
                return Ok(Some(VideoFrame {
                    pts_ms: f.pts_ms,
                    key_frame: f.key_frame,
                    width: self.width,
                    height: self.height,
                    y: f.frame.y,
                    u: f.frame.u,
                    v: f.frame.v,
                }));
            }
            // 手里没有现成的帧 —— 喂一张 picture 进去，或者收尾。
            let Some((header, data)) = self.pictures.pop_front() else {
                if self.finished {
                    return Ok(None);
                }
                self.finished = true;
                let got = self
                    .dec
                    .finish()
                    .map_err(|e| Error::Bad(format!("MPEG 解码失败：{e:?}")))?;
                self.ready.extend(got);
                continue;
            };
            let got = self
                .dec
                .push(header, &data)
                .map_err(|e| Error::Bad(format!("MPEG 解码失败：{e:?}")))?;
            self.ready.extend(got);
        }
    }
}
