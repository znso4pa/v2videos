//! 一条「每块一格、带 1 格边框」的状态网格。
//!
//! 编解码都要用：DC 预测、AC 预测、编码块图、运动矢量，都是这种形状 ——
//! 带一圈边框就省掉了到处判边界。`idx(gx, gy)` 里 `gx`/`gy` 可以是「宏块坐标」，
//! 取左邻就是 `xy - 1`、上邻 `xy - w`、右上 `xy - w + 1`。
//!
//! 边框的值由调用方定：DC 预测要 **1024**（不是 0），运动矢量要 0。

/// 一条块级的 DC/AC 预测状态网格（带 1 格边框，方便取左/上邻居）。
pub(crate) struct Grid<T> {
    pub(crate) w: usize,
    pub(crate) data: Vec<T>,
}

impl<T: Copy + Default> Grid<T> {
    pub(crate) fn new(w: usize, h: usize) -> Self {
        Self::with_init(w, h, T::default())
    }
    pub(crate) fn with_init(w: usize, h: usize, init: T) -> Self {
        Self {
            w: w + 2,
            data: vec![init; (w + 2) * (h + 2)],
        }
    }
    pub(crate) fn idx(&self, gx: usize, gy: usize) -> usize {
        (gy + 1) * self.w + (gx + 1)
    }
}
