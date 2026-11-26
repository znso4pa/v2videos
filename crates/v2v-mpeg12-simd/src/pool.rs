//! 常驻线程池：**跨帧复用 worker**，省掉每帧一次的 `spawn` / `join`。
//!
//! ## 为什么需要 unsafe
//!
//! 解码一帧要交给 worker 两样**借来的**东西：`&[u8]`（自己那段码流）和
//! `&mut Frame`（自己那块输出缓冲）。`std::thread::scope` 能把这两个借用的
//! 生命周期绑在作用域上，代价是**每帧新建一次线程** —— 实测 4 线程
//! `scope + spawn + join` 就要 **40.7 µs/帧**，占 4 线程总耗时（约 383 µs）的
//! **10%**；8 线程更贵（72 µs）。
//!
//! 想让 worker 跨帧活着，就得让它持有一个**比借用更长的**东西。唯一干净的说法
//! 是把借用降级成裸指针、在池内部重建生命周期 —— 那正是本模块那几处 unsafe。
//!
//! ## 安全性论证（改动前请重读）
//!
//! 全部依赖四条不变量，缺一条就会 UB：
//!
//! 1. **每个 worker 只拿到自己那块缓冲的裸指针**，来源是 `workers[i].buf`，
//!    而一个任务只对应一个 `i`。所以「多个 `&mut` 指向同一块内存」不会发生。
//! 2. **借用期覆盖整个调用**：`decode_into` 在返回前把 `inflight` 里的结果**全部
//!    收齐**。任何指针在被 worker 解开之后，指向的对象都在 `decode_into` 的栈帧里
//!    （`sequence` / `forward` / `backward` / `keep` 里的码流），比指针活得久。
//! 3. **一个 worker 一次只跑一个任务**：派发是 `send`，worker 取到就干活，干完
//!    才回来 `recv` 下一个。没有「同一个槽被写两次」的窗口。
//! 4. **出错也要收齐**：任何一个 worker 报错，仍然继续 `recv` 到 `remaining == 0`
//!    —— 提前返回会让还没干完的 worker 写一块已经失效的栈内存。
//!
//! 违反其中任何一条（比如两个任务指向同一缓冲、或者出错就提前 return）都会
//! **静默出错**，而且症状是「偶尔坏一帧」。所以 `tests/pool.rs` 除了逐字节
//! 对账，还有一条专门跑「池只建一次、连续解多帧」的用例。
//!
//! ## 为什么不去改 `MtDecoder`
//!
//! `v2v-mpeg12` 的文档写着「本 crate 零 unsafe，依赖闭包里也没有」，那句是拿
//! `cargo tree` 机器守着的，也是它当 **oracle** 的立身之本。把指针塞进去就等于
//! 把唯一那份「不用怀疑的实现」污染掉。所以那边只导出安全的行区间助手
//! （`copy_rows_pub` / `fill_blank_rows_pub` / `blank_frame_pub` /
//! `plane_dims`），裸指针全留在这里。

#![allow(unsafe_code)]

use std::sync::mpsc;

use v2v_mpeg12::decode::{
    blank_frame_pub, copy_rows_pub, decode_frame_into, fill_blank_rows_pub, plane_dims,
};
use v2v_mpeg12::decode::{DecodeError, Frame};
use v2v_mpeg12::{PictureHeader, PictureType, SequenceHeader};

/// 造一个 [`DecodeError`]。
///
/// `DecodeError::new` 在 `v2v-mpeg12` 里是私有的（它只给自己内部造），
/// 所以这边按它的字段直接构造 —— 字段是 `pub` 的。
fn bad(stage: &'static str, bit: usize, detail: i64) -> DecodeError {
    DecodeError { stage, bit, detail }
}

/// 一次派发给 worker 的任务。**指针以 `usize` 存**，安全性见模块文档。
struct Job {
    /// `&SequenceHeader`。
    sequence: usize,
    /// `&[u8]` 的 `(ptr, len)`。
    data: (usize, usize),
    /// `Option<&Frame>`（前向参考）。
    forward: Option<usize>,
    /// `Option<&Frame>`（后向参考）。
    backward: Option<usize>,
    /// 自己那块缓冲的 `*mut Frame`。
    buf: usize,
    picture: PictureHeader,
    /// 自己负责的宏块行区间 `[rs, re)`。
    rows: (usize, usize),
    /// 填底色而不是拷参考帧（I 帧 / 没有前向参考）。
    intra: bool,
    chroma_format: u8,
}

enum Cmd {
    Work(Job),
    Stop,
}

struct Worker {
    tx: mpsc::Sender<Cmd>,
    handle: Option<std::thread::JoinHandle<()>>,
    /// 跨帧复用的输出缓冲。
    buf: Frame,
}

/// 常驻线程池。`threads` 个 worker，跨帧复用。
pub struct Pool {
    workers: Vec<Worker>,
    rx: mpsc::Receiver<Result<usize, DecodeError>>,
    /// 开 worker 时用的分辨率；换了就重建缓冲。
    dims: (usize, usize),
}

impl Pool {
    /// 建一个有 `threads` 个 worker 的池。
    ///
    /// ## ⚠️ 只该用在 `threads >= 2` 的地方
    ///
    /// 池省的是「每帧 `spawn`/`join`」。单线程下这笔钱本来就不该花 ——
    /// `MtDecoder` 在 `threads == 1` 时**连组都不切**（省掉扫码流 + 拼装的
    /// ~28 µs/帧），而池仍然要切组才能派发。实测 `threads=1` 时池比
    /// `MtDecoder` **慢 6.6%**。调用方（`MpgPlayer`）按 `threads > 1` 判断，
    /// 别无条件注入。
    pub fn new(threads: usize, sequence: &SequenceHeader) -> Self {
        let threads = threads.max(1);
        let (tx, rx) = mpsc::channel();
        let dims = (
            sequence.width as usize,
            (sequence.height as usize).div_ceil(16) * 16,
        );
        let template = blank_frame_pub(sequence);
        let mut workers = Vec::with_capacity(threads);
        for _ in 0..threads {
            let (wtx, wrx) = mpsc::channel::<Cmd>();
            let out = tx.clone();
            let handle = std::thread::Builder::new()
                .name("v2v-mpeg12-pool".into())
                .spawn(move || worker_loop(wrx, out))
                .expect("起 MPEG 解码 worker");
            workers.push(Worker {
                tx: wtx,
                handle: Some(handle),
                // 每个 worker 先按当前分辨率开一块（之后跨帧复用）。
                buf: Frame {
                    width: template.width,
                    height: template.height,
                    y: vec![0u8; template.y.len()],
                    u: vec![128u8; template.u.len()],
                    v: vec![128u8; template.v.len()],
                },
            });
        }
        Self {
            workers,
            rx,
            dims,
        }
    }

    /// 池里并行几个 worker。
    pub fn width(&self) -> usize {
        self.workers.len()
    }

    /// 解一帧，把各 worker 的结果合并进 `out`。
    ///
    /// `groups` 是「(起始宏块行, 结束宏块行, 该组重组的码流)」—— 与
    /// `v2v_mpeg12::decode::MtDecoder` 切出来的形状**完全一样**，所以两条路能直接
    /// 逐字节对账（`tests/pool.rs` 就是干这个的）。
    ///
    /// `out` 的初始内容由调用方负责，语义同 [`decode_frame_into`]：没有被任何组
    /// 覆盖的行保持原样（调用方拿 `blank_frame_pub` 起手就会得到正确的底色）。
    pub fn decode_into(
        &mut self,
        sequence: &SequenceHeader,
        picture: PictureHeader,
        forward: Option<&Frame>,
        backward: Option<&Frame>,
        groups: Vec<(usize, usize, Vec<u8>)>,
        out: &mut Frame,
    ) -> Result<(), DecodeError> {
        if groups.is_empty() {
            return Err(bad("no_slices", 0, 0));
        }
        // 分辨率变了就把每个 worker 的缓冲重开（换片时一次，之后不再走）。
        let want = (
            sequence.width as usize,
            (sequence.height as usize).div_ceil(16) * 16,
        );
        if self.dims != want {
            let (w, ph, cw, ch) = {
                let t = blank_frame_pub(sequence);
                let (w, ph, cw, ch) = plane_dims(&t, sequence.chroma_format);
                (w, ph, cw, ch)
            };
            for wk in &mut self.workers {
                wk.buf = Frame {
                    width: sequence.width,
                    height: sequence.height,
                    y: vec![0u8; w * ph],
                    u: vec![128u8; cw * ch],
                    v: vec![128u8; cw * ch],
                };
            }
            self.dims = want;
        }

        let chroma_format = sequence.chroma_format;
        let intra = matches!(picture.picture_type, PictureType::I) || forward.is_none();
        let seq_ptr = sequence as *const SequenceHeader as usize;
        let fwd_ptr = forward.map(|f| f as *const Frame as usize);
        let bwd_ptr = backward.map(|f| f as *const Frame as usize);
        let mut groups = groups.into_iter();

        // 派发期间**续命**：worker 拿的是 `raw.as_ptr()`，得有个 owned 的东西保证
        // 那些字节活到收齐结果之后。
        let mut keep: Vec<Vec<u8>> = Vec::with_capacity(self.workers.len());
        // (worker 下标, 起始行, 结束行) —— 合并时按它来。
        let mut plan: Vec<(usize, usize, usize)> = Vec::with_capacity(self.workers.len());

        for (i, wk) in self.workers.iter_mut().enumerate() {
            let Some((rs, re, raw)) = groups.next() else {
                break;
            };
            let job = Job {
                sequence: seq_ptr,
                data: (raw.as_ptr() as usize, raw.len()),
                forward: fwd_ptr,
                backward: bwd_ptr,
                buf: &mut wk.buf as *mut Frame as usize,
                picture: picture.clone(),
                rows: (rs, re),
                intra,
                chroma_format,
            };
            plan.push((i, rs, re));
            // ⚠️ 顺序不能反：先把自己那块码流留着（`keep`），再派发。
            // worker 拿到的是指针，`keep` 才是它的生命周期依据。
            keep.push(raw);
            if wk.tx.send(Cmd::Work(job)).is_err() {
                // worker 已经不在（panic 在 worker 里）。此时**还没有任务在飞**
                // 之外的风险 —— 但已经派发的必须收完，见下面的收尾。
                break;
            }
        }
        let dispatched = plan.len();

        // ---- 收结果：**必须等齐**（不变量 2 / 4）----
        let mut first_err: Option<DecodeError> = None;
        let mut got = 0usize;
        while got < dispatched {
            match self.rx.recv() {
                Ok(Ok(_)) => got += 1,
                Ok(Err(e)) => {
                    got += 1;
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
                // 所有 Tx 都没了 —— 池不可用（worker 全死了）。
                Err(_) => {
                    return Err(bad("pool_disconnected", 0, got as i64));
                }
            }
        }
        // 到这里所有 worker 都干完并交了结果，裸指针全部失效。
        drop(keep);

        if let Some(e) = first_err {
            return Err(e);
        }

        // ---- 合并：按 plan 把每个 worker 的缓冲拷回 `out` ----
        for (i, rs, re) in plan {
            copy_rows_pub(out, &self.workers[i].buf, rs, re, chroma_format);
        }
        Ok(())
    }

    fn stop(&mut self) {
        for wk in &mut self.workers {
            let _ = wk.tx.send(Cmd::Stop);
        }
        for wk in &mut self.workers {
            if let Some(h) = wk.handle.take() {
                let _ = h.join();
            }
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.stop();
    }
}

impl v2v_mpeg12::picture::FrameParallel for Pool {
    fn decode_into(
        &mut self,
        sequence: &SequenceHeader,
        picture: PictureHeader,
        forward: Option<&Frame>,
        backward: Option<&Frame>,
        groups: Vec<(usize, usize, Vec<u8>)>,
        out: &mut Frame,
    ) -> Result<(), DecodeError> {
        Pool::decode_into(self, sequence, picture, forward, backward, groups, out)
    }

    fn width(&self) -> usize {
        Pool::width(self)
    }
}

fn worker_loop(rx: mpsc::Receiver<Cmd>, out: mpsc::Sender<Result<usize, DecodeError>>) {
    while let Ok(cmd) = rx.recv() {
        let Cmd::Work(job) = cmd else { return };
        // SAFETY: 四条不变量见模块文档。
        // ① `buf` 是 `&mut Frame` 转来的，且每个 worker 只拿到自己那一个；
        // ② `sequence` / `forward` / `backward` / 码流由调用方保证活到收齐结果；
        // ③ 一个 worker 一次只跑一个任务；④ 调用方出错也收齐。
        let res = unsafe {
            let sequence = &*(job.sequence as *const SequenceHeader);
            let data = std::slice::from_raw_parts(job.data.0 as *const u8, job.data.1);
            let forward = job.forward.map(|p| &*(p as *const Frame));
            let backward = job.backward.map(|p| &*(p as *const Frame));
            let buf = &mut *(job.buf as *mut Frame);
            // ① 把自己那段行区间填成底色（跳过宏块要的是「参考帧原位」）。
            if job.intra {
                fill_blank_rows_pub(buf, job.rows.0, job.rows.1, job.chroma_format);
            } else if let Some(f) = forward {
                copy_rows_pub(buf, f, job.rows.0, job.rows.1, job.chroma_format);
            }
            // ② 解自己那几个 slice，写进自己那块缓冲。
            decode_frame_into(sequence, job.picture, data, forward, backward, buf)
                .map(|()| job.rows.0)
        };
        if out.send(res).is_err() {
            return;
        }
    }
}
