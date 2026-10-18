//! VC-1 的**帧级**组装：帧头 → 宏块层 → 重建 → 环路滤波 → 输出。
//!
//! 这一层是「把前面几个核串起来」的地方，和 `v2v-msmpeg4` 的 `frame.rs` 同角色。
//! 上游对应 `vc1dec.c` 的 `vc1_decode_frame()`（渐进式那一条路）+ `vc1_block.c` 的
//! `vc1_decode_i_blocks()` / `vc1_decode_p_blocks()`。
//!
//! ## 状态分成三档，别混
//!
//! | 档 | 例子 | 生命周期 |
//! |---|---|---|
//! | 逐块网格 | `dc_val` / `ac_val` / `coded_block` / `mv` | **跨帧保留**（上游就是这样：DC/AC 预测值不从零开始） |
//! | 逐宏块面 | `cbp` / `is_intra` / `ttblk` / `luma_mv` / `qscale` | 每帧全写一遍 |
//! | 参考帧 | 上一帧的像素 | 帧间预测读它 |
//!
//! ⚠️ 第一档**不能**每帧清零：I 帧第一块之外的所有 DC 预测都建立在「邻居刚写进去的值」
//! 上，而跨帧边界处那些值是上一帧留下的 —— 清零会和上游分叉（像素 0 差就是判据）。
//!
//! ## 重建：I 帧覆盖、P 帧里帧间块自己写、帧内块最后统一写
//!
//! 上游的「写像素」是**落后解码一圈**的（`vc1_put_blocks_clamped`，因为 overlap 滤波
//! 要等邻居的块），这里不需要那套轮转：本实现不做 overlap（样本 `overlap=false`），
//! 写像素是纯覆盖、与顺序无关。但**环路滤波**仍然按上游的时序在同一个位置调用
//! （它依赖「哪个宏块已经写好了」），所以 `decode_i_blocks` / `decode_p_blocks`
//! 里那两行 `loop_filter` 不能挪。

use v2v_msmpeg4::{BitReader, Error, Planes, Result};

use crate::header::{FrameHeader, HeaderState, PictType, SeqHeader};
use crate::tables;

/// 解码出来的一帧（YUV420P，已裁到显示尺寸）。
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// 跨帧保留的逐块网格（见文件头那张表）。
pub(crate) struct Grids {
    /// 亮度 DC 预测值：b8 网格（`[mb_height*2] × [mb_width*2]`，多留一行一列）。
    pub dc_luma: Vec<i16>,
    /// 色度 DC 预测值：`[平面][宏块]`（两个平面拼在一起，行距 `mb_stride`）。
    pub dc_chroma: Vec<i16>,
    /// 亮度 AC 预测值：每块 16 个 `i16`（`[0..8)` 左、`[8..16)` 上），b8 网格。
    pub ac_luma: Vec<i16>,
    /// 色度 AC 预测值：`[平面][宏块][16]`。
    ///
    /// ⚠️ **前面多留一个块**（16 个 `i16`）：色度块「从左预测」时读的是 `base - 16`，
    /// 而第一个色度块的 `base` 就是 0 —— 不留余量会**下溢 panic**（调试构建）。
    /// 上游那边靠「色度区紧跟亮度区」天然有得读（读到的是亮度区的尾巴），这里读 0。
    pub ac_chroma: Vec<i16>,
    /// 编码块图预测：逐 8×8 块，b8 网格。
    pub coded_block: Vec<u8>,
    /// 逐 8×8 块「是不是帧内」（`v->mb_type[0]`）：b8 网格。
    pub mb_type: Vec<u8>,
    /// 亮度 MV：b8 网格。
    pub mv: Vec<[i16; 2]>,
    /// 逐宏块的**色度** MV（上游叫 `luma_mv`，名字骗人）：宏块网格。
    pub luma_mv: Vec<[i16; 2]>,
}

/// 逐宏块、每帧重建一遍的面。
pub(crate) struct MbPlanes {
    /// 本宏块的量化步长（有符号：负 = 半步长不适用），`ff_vc1_pred_dc` 与 AC 预测要它。
    pub qscale: Vec<i32>,
    /// 逐块 CBP（4 位一块，6 块共 24 位）—— 环路滤波用。
    pub cbp: Vec<u32>,
    /// 逐块「是不是帧内」（位掩码，6 位）—— 环路滤波用。
    pub is_intra: Vec<u8>,
    /// 逐块变换类型（4 位一块）—— 环路滤波用。
    pub ttblk: Vec<u32>,
}

/// 亮度补偿（intensity compensation）的查找表。
///
/// 对照上游 `vc1.c` 的 `INIT_LUT` / `rotate_luts()`：P 帧可以声明「参考帧要先按
/// `lumscale`/`lumshift` 映射一遍再拿来做运动补偿」。表是**逐帧轮转**的
/// （`last` / `next` 换位，`curr` 归恒等），而且新表是**串**在旧表上算出来的
/// （`chain`）—— 因为参考帧自己可能已经被映射过一次了。
pub(crate) struct Luts {
    pub last_y: [[u8; 256]; 2],
    pub last_uv: [[u8; 256]; 2],
    pub next_y: [[u8; 256]; 2],
    pub next_uv: [[u8; 256]; 2],
    pub last_use_ic: bool,
    pub next_use_ic: bool,
}

/// `INIT_LUT`：`chain` = 在**已有**的表上再串一层。
fn init_lut(
    lumscale: u8,
    lumshift: u8,
    luty: &mut [u8; 256],
    lutuv: &mut [u8; 256],
    chain: bool,
) {
    let (scale, shift) = if lumscale == 0 {
        let mut shift = (255 - lumshift as i32 * 2) * 64;
        if lumshift > 31 {
            shift += 128 << 6;
        }
        (-64i32, shift)
    } else {
        let scale = lumscale as i32 + 32;
        let shift = if lumshift > 31 {
            (lumshift as i32 - 64) * 64
        } else {
            (lumshift as i32) << 6
        };
        (scale, shift)
    };
    for i in 0..256usize {
        let iy = if chain { luty[i] } else { i as u8 } as i32;
        let iu = if chain { lutuv[i] } else { i as u8 } as i32;
        luty[i] = ((scale * iy + shift + 32) >> 6).clamp(0, 255) as u8;
        lutuv[i] = ((scale * (iu - 128) + 128 * 64 + 32) >> 6).clamp(0, 255) as u8;
    }
}

impl Luts {
    fn new() -> Self {
        Self {
            last_y: [[0; 256]; 2],
            last_uv: [[0; 256]; 2],
            next_y: [[0; 256]; 2],
            next_uv: [[0; 256]; 2],
            last_use_ic: false,
            next_use_ic: false,
        }
    }

    /// `rotate_luts()`：`last` 与 `next` 换位，然后把 `curr`（= `next`）写成恒等。
    fn rotate(&mut self) {
        std::mem::swap(&mut self.last_y, &mut self.next_y);
        std::mem::swap(&mut self.last_uv, &mut self.next_uv);
        std::mem::swap(&mut self.last_use_ic, &mut self.next_use_ic);
        init_lut(32, 0, &mut self.next_y[0], &mut self.next_uv[0], false);
        init_lut(32, 0, &mut self.next_y[1], &mut self.next_uv[1], false);
        self.next_use_ic = false;
    }

    /// 本帧声明了亮度补偿：造表（串在旧表上）。
    fn set_ic(&mut self, lumscale: u8, lumshift: u8) {
        self.last_use_ic = true;
        init_lut(lumscale, lumshift, &mut self.last_y[0], &mut self.last_uv[0], true);
        init_lut(lumscale, lumshift, &mut self.last_y[1], &mut self.last_uv[1], true);
    }
}

/// 扫描表。
///
/// `res_fasttx`（本实现只支持这一档）用的是**转置**过的 8×8 扫描表 —— 因为配套的
/// VC-1 整数变换工作在转置布局上。8×4 / 4×8 那两张**不**转置（上游也只转置 8×8）。
pub(crate) struct Scans {
    pub zz_8x8: [[u8; 64]; 4],
    pub zz_8x4: [u8; 32],
    pub zz_4x8: [u8; 32],
    pub zz_4x4: [u8; 16],
}

fn transpose(x: u8) -> u8 {
    (x >> 3) | ((x & 7) << 3)
}

impl Scans {
    fn new() -> Self {
        let mut zz_8x8 = [[0u8; 64]; 4];
        for t in 0..4 {
            for i in 0..64 {
                zz_8x8[t][i] = transpose(v2v_msmpeg4::tables::SCAN[t][i]);
            }
        }
        let mut zz_8x4 = [0u8; 32];
        let mut zz_4x8 = [0u8; 32];
        zz_8x4.copy_from_slice(&v2v_msmpeg4::tables::ABT_SCAN_A[..32]);
        zz_4x8.copy_from_slice(&v2v_msmpeg4::tables::ABT_SCAN_B[..32]);
        Self {
            zz_8x8,
            zz_8x4,
            zz_4x8,
            zz_4x4: tables::SCAN_4X4,
        }
    }
}

/// VC-1 解码器（一帧进、一帧出）。
///
/// 尺寸来自**容器**：Simple/Main profile 的序列头里没有宽高。
pub struct Decoder {
    seq: SeqHeader,
    width: u32,
    height: u32,
    mb_width: usize,
    mb_height: usize,
    state: HeaderState,
    scans: Scans,
    /// 当前帧（宏块对齐、**未裁剪** —— 运动补偿会读到显示高度以外的行）。
    planes: Planes,
    /// 参考帧（同尺寸，未裁剪）。
    ref_planes: Planes,
    has_ref: bool,
    grids: Grids,
    mbp: MbPlanes,
    luts: Luts,
    /// 最后开始解的那个宏块与当时的位游标 —— 出错时用来定位
    /// （错误类型里塞不下动态信息，所以放在这儿）。
    last_mb: (usize, usize),
    last_bits: usize,
}

impl Decoder {
    /// `width` / `height` 是**显示**尺寸（容器给的）；宏块网格由它向上取整到 16 得到。
    pub fn new(seq: SeqHeader, width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::Bad("vc1: 宽高为 0"));
        }
        // 本实现只做 `res_fasttx` 那一档（VC-1 整数变换 + 转置扫描表）。
        // 另一档走的是 MS-MPEG-4 的 simple_idct，本仓没有可验证的样本 —— 与其
        // 写一条没验过的路，不如明确拒绝（仓规：不静默出错）。
        if !seq.res_fasttx {
            return Err(Error::Unsupported(
                "vc1: res_fasttx=0（走 MS-MPEG-4 simple_idct 的那一档）还没做".into(),
            ));
        }
        if seq.overlap {
            return Err(Error::Unsupported("vc1: overlap smoothing 还没做".into()));
        }
        let mb_width = (width as usize).div_ceil(16);
        let mb_height = (height as usize).div_ceil(16);
        let mb_stride = mb_width + 1;
        let b8_stride = mb_width * 2 + 1;
        let b8_rows = mb_height * 2 + 2;
        let mb_rows = mb_height + 2;
        // 亮度区（b8 网格）+ 色度两个平面各自的区域（与上游 `block_index[4]/[5]` 同构）。
        let b8_len = b8_stride * b8_rows
            + b8_stride
            + 2 * mb_stride * (mb_height + 1)
            + mb_stride;
        let mb_len = mb_stride * mb_rows;
        Ok(Self {
            seq,
            width,
            height,
            mb_width,
            mb_height,
            state: HeaderState::default(),
            scans: Scans::new(),
            planes: Planes::new(mb_width * 16, mb_height * 16),
            ref_planes: Planes::new(mb_width * 16, mb_height * 16),
            has_ref: false,
            grids: Grids {
                // ⚠️ DC 网格的**初值**是 1024（上游 `dc_val_base` 分配时就填这个），
                // 不是 0：I 帧第一块的预测就落在这上面。
                dc_luma: vec![1024; b8_len],
                dc_chroma: vec![1024; 2 * mb_len],
                ac_luma: vec![0; 16 * b8_len],
                ac_chroma: vec![0; 16 + 16 * 2 * mb_len],
                coded_block: vec![0; b8_len],
                mb_type: vec![0; b8_len],
                mv: vec![[0, 0]; b8_len],
                luma_mv: vec![[0, 0]; mb_len],
            },
            mbp: MbPlanes {
                qscale: vec![0; mb_len],
                cbp: vec![0; mb_len],
                is_intra: vec![0; mb_len],
                ttblk: vec![0; mb_len],
            },
            luts: Luts::new(),
            last_mb: (0, 0),
            last_bits: 0,
        })
    }

    /// 解一帧。`data` 是裸帧数据（ASF 样本里的一个 sample），帧头在最前面。
    ///
    /// 返回**用掉的位数** —— 判据之一就是它落在帧尾（`bits <= data.len() * 8` 且
    /// 通常落在最后一个字节里）。
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<usize> {
        let mut r = BitReader::new(data);
        let hdr = FrameHeader::parse_from(&mut r, &self.seq, self.mb_width, self.mb_height, &mut self.state)?;
        let mb_stride = self.mb_width + 1;
        let b8_stride = self.mb_width * 2 + 1;

        // 亮度补偿：每帧先轮转表；本帧声明了补偿就现造一张（串在旧表上）。
        self.luts.rotate();
        if let Some(p) = &hdr.p {
            if p.mv_mode == crate::header::MvMode::IntensityComp {
                self.luts.set_ic(p.lumscale, p.lumshift);
            }
        }

        let bits_used;
        {
            let mut ctx = Ctx {
                seq: self.seq,
                hdr: &hdr,
                mb_width: self.mb_width,
                mb_height: self.mb_height,
                mb_stride,
                b8_stride,
                b8_off: b8_stride + 1,
                scans: &self.scans,
                planes: &mut self.planes,
                ref_planes: &self.ref_planes,
                has_ref: self.has_ref,
                grids: &mut self.grids,
                mbp: &mut self.mbp,
                luts: &self.luts,
                use_ic: self.luts.last_use_ic,
                last_mb: &mut self.last_mb,
                last_bits: &mut self.last_bits,
                r,
                codingset: 0,
                codingset2: 0,
                y_dc_scale: 0,
                c_dc_scale: 0,
                tt_index: 0,
                esc3_level_length: 0,
                esc3_run_length: 0,
                a_avail: false,
                c_avail: false,
                loop_filter: self.seq.loop_filter,
            };
            match hdr.pict_type {
                PictType::I => ctx.decode_i_blocks()?,
                PictType::P => ctx.decode_p_blocks()?,
            }
            bits_used = ctx.r.bit_pos();
        }

        // 本帧成为下一帧的参考。交换比整帧 memcpy 便宜，且两边都写满（每个宏块都会
        // 覆盖自己那 16×16 / 8×8 像素），不需要先清空。
        std::mem::swap(&mut self.planes, &mut self.ref_planes);
        self.has_ref = true;
        Ok(bits_used)
    }

    /// 最近解出的那一帧（裁到显示尺寸）。
    pub fn frame(&self) -> Frame {
        let pw = self.mb_width * 16;
        let (w, h) = (self.width as usize, self.height as usize);
        let src = &self.ref_planes;
        let mut y = Vec::with_capacity(w * h);
        for r in 0..h {
            y.extend_from_slice(&src.y[r * pw..r * pw + w]);
        }
        let cw = pw / 2;
        let (cw_vis, ch_vis) = (w.div_ceil(2), h.div_ceil(2));
        let mut u = Vec::with_capacity(cw_vis * ch_vis);
        let mut v = Vec::with_capacity(cw_vis * ch_vis);
        for r in 0..ch_vis {
            u.extend_from_slice(&src.u[r * cw..r * cw + cw_vis]);
            v.extend_from_slice(&src.v[r * cw..r * cw + cw_vis]);
        }
        Frame {
            width: self.width,
            height: self.height,
            y,
            u,
            v,
        }
    }

    /// 最近解出的那一帧的**宏块对齐、未裁剪**平面（对账测试用：ffmpeg 吐的也是这个尺寸）。
    pub fn planes(&self) -> &Planes {
        &self.ref_planes
    }

    /// **解到一半**的那一帧的工作平面 —— 只在 `decode_frame` 返回 `Err` 之后有意义。
    ///
    /// 用途：定位位流从哪个宏块开始跑偏（错的宏块只影响它自己和后面，前面的像素还是对的）。
    pub fn partial_planes(&self) -> &Planes {
        &self.planes
    }

    /// 上一帧是不是已经解出来了（第一帧必须是 I 帧）。
    pub fn has_reference(&self) -> bool {
        self.has_ref
    }

    /// 最近一次 `decode_frame` 最后开始解的那个宏块 `(x, y)` 与当时的位游标 —— 出错时用来定位。
    pub fn last_mb(&self) -> (usize, usize) {
        self.last_mb
    }

    /// 见 [`Decoder::last_mb`]。
    pub fn last_mb_bits(&self) -> usize {
        self.last_bits
    }
}

/// 一帧解码过程中的全部可变状态。
///
/// 分成 `&mut` 借来的几块（平面 / 网格 / 位流），是为了让「读参考帧、写当前帧」
/// 这种同时借用能在 Rust 里成立 —— 它们本来就是不相交的字段。
pub(crate) struct Ctx<'a, 'b> {
    pub seq: SeqHeader,
    pub hdr: &'b FrameHeader,
    pub mb_width: usize,
    pub mb_height: usize,
    pub mb_stride: usize,
    pub b8_stride: usize,
    /// b8 网格的基址偏移（一行一列）—— 于是「左邻 / 上邻」不会落到负数下标上。
    pub b8_off: usize,
    pub scans: &'b Scans,
    pub planes: &'a mut Planes,
    pub ref_planes: &'a Planes,
    pub has_ref: bool,
    pub grids: &'a mut Grids,
    pub mbp: &'a mut MbPlanes,
    pub luts: &'a Luts,
    /// 本帧的运动补偿要不要先把参考帧按 `luts.last_*` 映射一遍。
    pub use_ic: bool,
    pub last_mb: &'a mut (usize, usize),
    pub last_bits: &'a mut usize,
    pub r: BitReader<'b>,
    // ---- 帧级派生量（帧头解完就定下来）----
    pub codingset: usize,
    pub codingset2: usize,
    pub y_dc_scale: i32,
    pub c_dc_scale: i32,
    pub tt_index: usize,
    /// ESC3 的位宽：**逐帧复用**（解出来一次，后面的块接着用）。
    pub esc3_level_length: u32,
    pub esc3_run_length: u32,
    /// 逐块 DC 预测的邻居可用性（P 帧的帧内块要）。
    pub a_avail: bool,
    pub c_avail: bool,
    pub loop_filter: bool,
}

impl Ctx<'_, '_> {
    /// 宏块 `(x, y)` 的块 `n`（0..4 亮度、4 色度 U、5 色度 V）在 b8 网格上的下标。
    ///
    /// 与上游 `s->block_index[]` 同构：亮度块是 2×2 的 8×8 网格（`+1` 是右邻、
    /// `+b8_stride` 是下一行块），色度块各占一格。
    #[inline]
    pub(crate) fn blk(&self, mbx: usize, mby: usize, n: usize) -> usize {
        if n < 4 {
            self.b8_off
                + self.b8_stride * (2 * mby)
                + 2 * mbx
                + (n & 1)
                + (n >> 1) * self.b8_stride
        } else {
            // ⚠️ 色度块在**亮度区之后**的独立区域里（上游 `block_index[4]/[5]` 就是
            // `b8_stride*mb_height*2 + …`）。少了这一段偏移，色度的下标会落进亮度区、
            // 把邻居的「是不是帧内」冲掉 —— 症状是**隔很远**的宏块 MV 预测偏掉。
            self.b8_off
                + self.b8_stride * (2 * self.mb_height)
                + self.mb_stride * (mby + 1)
                + mbx
                - 1
                + (n - 4) * self.mb_stride * (self.mb_height + 1)
        }
    }

    /// 宏块 `(x, y)` 的**色度**块（U / V 各一格）在色度网格上的下标（按宏块坐标直接算）。
    #[inline]
    pub(crate) fn cblk(&self, mbx: usize, mby: usize, plane: usize) -> usize {
        plane * self.mb_stride * (self.mb_height + 2) + mby * self.mb_stride + mbx
    }

    /// 当前宏块 `(x, y)` 在逐宏块面上的下标。
    #[inline]
    pub(crate) fn mbidx(&self, mbx: usize, mby: usize) -> usize {
        mby * self.mb_stride + mbx
    }

    #[inline]
    pub(crate) fn plane_stride(&self, plane: usize) -> usize {
        if plane == 0 {
            self.mb_width * 16
        } else {
            self.mb_width * 8
        }
    }

    /// 当前帧某个平面的像素（可写）。
    #[inline]
    pub(crate) fn plane_mut(&mut self, plane: usize) -> &mut [u8] {
        match plane {
            0 => &mut self.planes.y,
            1 => &mut self.planes.u,
            _ => &mut self.planes.v,
        }
    }

    /// 参考帧某个平面的像素。
    #[inline]
    pub(crate) fn ref_plane(&self, plane: usize) -> &[u8] {
        match plane {
            0 => &self.ref_planes.y,
            1 => &self.ref_planes.u,
            _ => &self.ref_planes.v,
        }
    }
}
