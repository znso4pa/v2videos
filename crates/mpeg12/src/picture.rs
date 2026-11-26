//! 解码序 → 显示序：把一张张 picture 喂进去，按**显示序**吐出帧。
//!
//! 为什么单独抽出来：这件事有**两个**消费者 ——
//!
//! 1. 播放器（`v2v-player` 的 `MpgPlayer`）：解封装 PS，然后一帧一帧往外给；
//! 2. 转码链路的 CLI（`.mpg` → `.mp4`）：一个 picture 一个样本喂进来。
//!
//! 两者都要「按 `temporal_reference` 重排 + 维护前向/后向参考帧 + 一段一段摊平」，
//! 而且**必须完全一致**（同一份码流走两条路要出同样的像素和同样的 PTS）。
//! 所以只有这一份实现。
//!
//! ## 为什么必须「一段一段」而不是「一帧进一帧出」
//!
//! 解码序里 B 帧排在它的参考帧**后面**。一边解一边交就会先把 P 帧交出去
//! （实测第 1 帧就差 229）。所以攒到一个新 I 帧到来（或码流结束）才把这一段
//! 按 `temporal_reference` 排序摊平，之后才允许交帧。

use std::collections::BTreeMap;

use crate::decode::{self, DecodeError, Frame, MtDecoder};
use crate::{PictureHeader, PictureType, SequenceHeader};

/// 一个 GOP 最多多少帧 —— 防御「没有 I 帧的坏流」把内存吃光。
const MAX_GOP_FRAMES: usize = 600;

/// 解好、已经按显示序排好的一帧。
pub struct DecodedFrame {
    /// 显示时间（毫秒）：`(gop 起点 + temporal_reference) × 帧长`。
    pub pts_ms: i64,
    /// 是不是 I 帧（seek 的落点候选）。
    pub key_frame: bool,
    /// **已经裁到可见高度** —— 平面的 16 对齐尾巴去掉了。
    ///
    /// 内部缓冲要留那几行（后续帧的宏块可能引用它们，见 `decode::Frame` 的说明），
    /// 但交出去的这一处必须裁 —— 消费者历来假设 `y.len() == width * height`。
    pub frame: Frame,
}

/// 解码序进、显示序出的状态机。见模块文档。
pub struct PictureDecoder {
    sequence: SequenceHeader,
    frame_ms: f64,
    /// 最近一个 I/P 帧（B 帧的后向参考）。
    last_ip: Option<Frame>,
    /// 再往前一个 I/P 帧（B 帧的前向参考）。
    prev_ip: Option<Frame>,
    /// 当前这一段（还没摊平）的帧，按 `temporal_reference` 配对。
    pending: Vec<(u32, Frame)>,
    /// 这一段里哪些 `temporal_reference` 是 I 帧。
    pending_key: BTreeMap<u32, bool>,
    /// 已经交出去的帧数（显示序），用作下一段的 PTS 起点。
    gop_base: u64,
    /// `Some(n)` = 这一段已经摊平，还剩 `n` 个显示位置没吐完。
    flushing: Option<u64>,
    /// 切片级并行；`None` = 单线程。
    mt: Option<MtDecoder>,
}

impl PictureDecoder {
    /// `threads > 1` 时切片级并行解码（产物与单线程逐字节相同）。
    pub fn new(sequence: SequenceHeader, threads: usize) -> Self {
        let frame_ms = if sequence.frame_rate_milli > 0 {
            1000.0 / (sequence.frame_rate_milli as f64 / 1000.0)
        } else {
            1000.0 / 30.0
        };
        Self {
            sequence,
            frame_ms,
            last_ip: None,
            prev_ip: None,
            pending: Vec::new(),
            pending_key: BTreeMap::new(),
            gop_base: 0,
            flushing: None,
            mt: (threads > 1).then(|| MtDecoder::new(threads)),
        }
    }

    pub fn sequence(&self) -> &SequenceHeader {
        &self.sequence
    }

    /// 毫秒/帧。
    pub fn frame_ms(&self) -> f64 {
        self.frame_ms
    }

    /// 喂一张 picture（**解码序**），返回现在就可以按显示序交出去的帧。
    ///
    /// 多数时候返回空；遇到新的 I 帧（或 `finish`）会把上一段整段摊平交出来，
    /// 所以是 `Vec` 而不是 `Option`。
    pub fn push(
        &mut self,
        header: PictureHeader,
        data: &[u8],
    ) -> Result<Vec<DecodedFrame>, DecodeError> {
        let mut out = Vec::new();
        // 新 GOP 的起点：**先把上一段摊平交出去**，再解这张 I 帧。
        if matches!(header.picture_type, PictureType::I) && !self.pending.is_empty() {
            self.flush_gop();
            self.drain(&mut out);
        }

        let kind = header.picture_type;
        let tr = header.temporal_reference;
        // D 帧不显示（MPEG-1 的「只用于预测」帧），跳过。
        if matches!(kind, PictureType::D) {
            return Ok(out);
        }
        let forward = if matches!(kind, PictureType::B) {
            self.prev_ip.as_ref()
        } else {
            self.last_ip.as_ref()
        };
        let backward = self.last_ip.as_ref();
        let frame = match &mut self.mt {
            Some(mt) => mt.decode(&self.sequence, header, data, forward, backward),
            None => decode::decode_frame(&self.sequence, header, data, forward, backward),
        }?;
        if matches!(kind, PictureType::I | PictureType::P) {
            self.prev_ip = self.last_ip.take();
            self.last_ip = Some(frame.clone());
        }
        if matches!(kind, PictureType::I) {
            self.pending_key.insert(tr, true);
        }
        self.pending.push((tr, frame));
        if self.pending.len() > MAX_GOP_FRAMES {
            return Err(DecodeError {
                stage: "gop_too_long",
                bit: 0,
                detail: MAX_GOP_FRAMES as i64,
            });
        }
        self.drain(&mut out);
        Ok(out)
    }

    /// 码流结束：把最后一段摊平交出来。
    pub fn finish(&mut self) -> Result<Vec<DecodedFrame>, DecodeError> {
        let mut out = Vec::new();
        if !self.pending.is_empty() {
            self.flush_gop();
            self.drain(&mut out);
        }
        Ok(out)
    }

    /// 把当前这一段按显示序摊平（新的 I 帧到来 / EOS 时调）。
    ///
    /// ⚠️ **不能在这里推 `gop_base`**：这些帧还没交出去，它们的 PTS 用的就是
    /// 当前这个 base。推到 `drain` 吐完最后一帧时再做。
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

    /// 把已经摊平的那一段按显示序搬进 `out`（`flushing` 为 `None` 时什么也不做）。
    fn drain(&mut self, out: &mut Vec<DecodedFrame>) {
        if self.flushing.is_none() {
            return;
        }
        // 摊平之后 `pending` 不再变动，排序一次就够 —— 原来每交一帧排一次。
        self.pending.sort_by_key(|(tr, _)| *tr);
        for (tr, mut frame) in std::mem::take(&mut self.pending) {
            let pts_ms = ((self.gop_base + u64::from(tr)) as f64 * self.frame_ms).round() as i64;
            let key_frame = self.pending_key.get(&tr).copied().unwrap_or(false);
            frame.truncate_visible(self.sequence.chroma_format);
            out.push(DecodedFrame {
                pts_ms,
                key_frame,
                frame,
            });
        }
        // 这一段吐完了 —— 把显示序的起点往前推，关键帧表跟着作废。
        self.gop_base += self.flushing.take().unwrap_or(0);
        self.pending_key.clear();
    }
}
