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

/// 「把一帧并行解出来」的策略。
///
/// ## 为什么是 trait 而不是直接调 `MtDecoder`
///
/// 因为**更好的并行实现住在别的 crate 里**（`v2v-mpeg12-simd` 的常驻线程池
/// 带 `unsafe`，而本 crate 的「零 unsafe」是拿 `cargo tree` 机器守着的）。
/// 本 crate 不能反向依赖它，所以让上层把实现**注入**进来 —— 与
/// [`crate::decode::set_idct_cols`] 同一个思路，只是这个实现带状态
/// （worker 要跨帧复用），所以是 `&mut self` 的 trait 而不是函数指针。
///
/// ## 实现方必须保证
///
/// 产物与 [`crate::decode::decode_frame`] **逐字节相同**。`out` 的初始内容由
/// [`PictureDecoder`] 负责（它按 `split_frame_groups` 的覆盖范围填好底色），
/// 实现方只管把自己那段行区间解出来、**再合并回 `out`**。
pub trait FrameParallel {
    /// `groups` 由 [`crate::decode::split_frame_groups`] 给出（唯一的切法）。
    /// 返回的 `Err` 原样上报，不吞。
    fn decode_into(
        &mut self,
        sequence: &SequenceHeader,
        picture: PictureHeader,
        forward: Option<&Frame>,
        backward: Option<&Frame>,
        groups: Vec<(usize, usize, Vec<u8>)>,
        out: &mut Frame,
    ) -> Result<(), DecodeError>;

    /// 并行度（`--threads` 报给用户的数字就是它）。
    fn width(&self) -> usize;
}

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
    ///
    /// 默认实现是 `MtDecoder`（每帧 `scope + spawn`）；上层可以用
    /// [`PictureDecoder::set_parallel`] 换成常驻池。
    par: Option<Box<dyn FrameParallel>>,
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
            par: (threads > 1).then(|| {
                Box::new(MtDecoder::new(threads)) as Box<dyn FrameParallel>
            }),
        }
    }

    /// 换一个并行实现（默认是 `MtDecoder`）。
    ///
    /// 上层用它把**带 unsafe 的常驻线程池**注入进来 —— 本 crate 自己不碰
    /// unsafe，也不依赖那个 crate。
    pub fn set_parallel(&mut self, p: Box<dyn FrameParallel>) {
        self.par = Some(p);
    }

    /// 当前的并行度（没有并行就是 1）。
    pub fn parallel_width(&self) -> usize {
        self.par.as_ref().map_or(1, |p| p.width())
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
        // 并行只在「切得开」时才走：切法是唯一的（`split_frame_groups`），
        // 它说 `None`（片太少 / 帧太矮 / 线程 1）就老老实实单线程。
        let groups = self
            .par
            .as_ref()
            .and_then(|p| decode::split_frame_groups(&self.sequence, data, p.width()));
        let frame = match (groups, &mut self.par) {
            (Some(groups), Some(p)) => {
                let mut out = decode::blank_frame_pub(&self.sequence);
                // ⚠️ `out` 的初始内容由**这里**负责（并行实现只管自己那几行）：
                // 没有被任何组覆盖的行保持「参考帧原位 / I 帧底色」。
                let mb_h = (self.sequence.height as usize).div_ceil(16);
                let first_start = groups.first().map(|g| g.0).unwrap_or(0);
                let last_end = groups.last().map(|g| g.1).unwrap_or(0);
                let intra = matches!(kind, PictureType::I);
                for (r0, r1) in [(0usize, first_start), (last_end, mb_h)] {
                    if r0 >= r1 {
                        continue;
                    }
                    match (intra, forward) {
                        (true, _) | (false, None) => decode::fill_blank_rows_pub(
                            &mut out,
                            r0,
                            r1,
                            self.sequence.chroma_format,
                        ),
                        (false, Some(f)) => decode::copy_rows_pub(
                            &mut out,
                            f,
                            r0,
                            r1,
                            self.sequence.chroma_format,
                        ),
                    }
                }
                p.decode_into(&self.sequence, header, forward, backward, groups, &mut out)?;
                out
            }
            _ => decode::decode_frame(&self.sequence, header, data, forward, backward)?,
        };
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
