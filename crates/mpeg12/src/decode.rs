//! MPEG-1 帧内（I）/ 前向（P）/ 双向（B）帧重建。
//!
//! slice → 宏块 → 块 → 反量化 → IDCT；P 帧加前向运动补偿，B 帧加前向/后向/双向补偿。
//! **重排序**（解码序 ≠ 显示序）不在这里 —— 由调用方按 `temporal_reference` 处理。
//!
//! 每个失败点都带上「在哪一步、第几位」，因为位对齐错一位的表现只是「后面全错」，
//! 没有位偏移就无从下手。

use crate::tables::{
    Code, DC_CHROMA, DC_LUM, EOB_BITS, EOB_CODE, ESCAPE_BITS, ESCAPE_CODE, INTRA_MATRIX,
    MB_ADDR_INCR, MB_PATTERN, MPEG1_RL, MV_CODE, NON_INTRA_MATRIX, P_MB_TYPE, B_MB_TYPE, Motion,
    ALT_SCAN, MPEG2_RL, NON_LINEAR_QSCALE, RlEntry,
};
use crate::{BitReader, PictureHeader, PictureType, SequenceHeader};


/// 解码出来的一帧（I420）。
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// 极简热点剖析（**只在设了 `V2V_PROF` 时开**，否则零开销）。
///
/// 目的是回答「下一个 unsafe 内核该上哪儿」—— 不凭感觉。四个桶：
/// 码流/反量化、逆变换、运动补偿预测、其它（总时间减前三）。
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub const BITS: usize = 0; // 位读取 + VLC（**含**嵌套的 DEQUANT —— 见下）
    pub const IDCT: usize = 1;
    pub const MC: usize = 2;
    pub const DEQUANT: usize = 3; // 反量化（被 BITS 嵌套计时，报告时单列）
    pub const PARSE: usize = 4; // slice/宏块头、运动矢量、cbp 等循环开销（含嵌套的 BITS/MC/STORE）
    pub const STORE: usize = 5; // 残差/预测夹到 0..255 写回平面（被 PARSE 嵌套）
    pub const BUCKETS: usize = 6;

    static NS: [AtomicU64; BUCKETS] =
        [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    static ON: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        ON.load(Ordering::Relaxed) != 0
    }

    /// 在 main 里调一次（读环境变量）。
    pub fn init_from_env() {
        let on = std::env::var("V2V_PROF").is_ok() as u64;
        ON.store(on, Ordering::Relaxed);
    }

    /// RAII 计时器：drop 时把耗时加进桶。
    pub struct Guard(usize, Instant);
    impl Guard {
        pub fn new(bucket: usize) -> Self {
            Guard(bucket, Instant::now())
        }
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            NS[self.0].fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    pub fn report() {
        if !enabled() {
            return;
        }
        let bits = NS[BITS].load(Ordering::Relaxed) as f64 / 1e9;
        let idct = NS[IDCT].load(Ordering::Relaxed) as f64 / 1e9;
        let mc = NS[MC].load(Ordering::Relaxed) as f64 / 1e9;
        let deq = NS[DEQUANT].load(Ordering::Relaxed) as f64 / 1e9;
        let parse = NS[PARSE].load(Ordering::Relaxed) as f64 / 1e9;
        let store = NS[STORE].load(Ordering::Relaxed) as f64 / 1e9;
        // ⚠️ 嵌套关系：BITS 包住 DEQUANT；PARSE 包住 BITS/IDCT/MC/STORE；STORE 包住自身。
        // 所以「纯解析开销」= PARSE - BITS - IDCT - MC - STORE；「纯 VLC」= BITS - DEQUANT。
        let vlc = bits - deq;
        let parse_only = parse - bits - idct - mc - store;
        eprintln!(
            "[prof] VLC {vlc:.3}s | 反量化 {deq:.3}s | 逆变换 {idct:.3}s | 运动补偿 {mc:.3}s | 写回 {store:.3}s | 解析开销 {parse_only:.3}s"
        );
    }
}

const LUT_BITS: u32 = 16;

/// 本地位缓存：一个 64 位窗口，左对齐，够连着服务好几个符号。
///
/// ## 为什么需要它
///
/// `BitReader::peek_u` 每个符号都要重新做一次「8 字节 unaligned load +
/// `from_be_bytes` + 移位」。系数解码的内层循环一个块最多 64 个符号，
/// 那个装载占了相当一部分时间。参考实现的做法是 `OPEN_READER`/`UPDATE_CACHE`
/// —— 在**寄存器**里维持一个位缓存，每消费 30~50 位才重新装一次。
///
/// ## 语义必须与 `peek_u` 逐位相同
///
/// - `peek(n)` 返回的位与 `BitReader::peek_u(n)` **完全一致**（含尾部补 0）；
/// - `bits_left()` 报的是**流里真实剩下的位数**，与缓存装了多少无关 ——
///   调用方靠它决定什么时候落回慢路。
///
/// 只在系数解码的内层循环里用；慢路（escape / 尾部）会把游标交还给
/// `BitReader` 再重新装缓存，所以 `BitReader` 始终是唯一的事实来源。
struct BitCache<'a> {
    data: &'a [u8],
    /// 下一个要读的位的绝对位置。
    pos: usize,
    /// 从 `pos` 起的位，**左对齐**（最高位就是 `pos` 处的位）；无效低位恒为 0。
    cache: u64,
    /// `cache` 里可信的位数（0..=64）。
    valid: u32,
}

impl<'a> BitCache<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        let mut c = Self {
            data,
            pos,
            cache: 0,
            valid: 0,
        };
        c.refill();
        c
    }

    /// 读 `at` 位位置开始的 64 位（越界补 0，与 `peek_u` 的慢路一致）。
    #[inline]
    fn read64(&self, at: usize) -> u64 {
        let byte = at >> 3;
        match self.data.get(byte..byte + 8) {
            Some(s) => u64::from_be_bytes(s.try_into().unwrap()),
            None => {
                let mut w = 0u64;
                for i in 0..8usize {
                    w = (w << 8) | *self.data.get(byte + i).unwrap_or(&0) as u64;
                }
                w
            }
        }
    }

    /// 把缓存补满（至少 57 位，够任何一次 `peek(17)`）。
    #[inline]
    fn refill(&mut self) {
        if self.valid >= 32 {
            return;
        }
        let keep = self.valid as usize;
        let at = self.pos + keep;
        let off = (at & 7) as u32;
        // `w << off` 把 `at` 处的位顶到最高位，于是它正好接在已有的 keep 位后面。
        let w = self.read64(at) << off;
        self.cache |= w >> keep;
        self.valid = 64.min(keep as u32 + 64 - off);
    }

    /// 偷看 `n` 位（`n <= 32`）。语义与 `BitReader::peek_u` 相同。
    #[inline]
    fn peek(&mut self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        if self.valid < n {
            self.refill();
        }
        (self.cache >> (64 - n)) as u32
    }

    /// 前进 `n` 位（调用方保证刚 `peek` 过、位数够）。
    #[inline]
    fn advance(&mut self, n: u32) {
        debug_assert!(self.valid >= n);
        self.cache <<= n;
        self.valid -= n;
        self.pos += n as usize;
    }

    /// 流里真实剩下的位数（与缓存装了多少无关）。
    #[inline]
    fn bits_left(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.pos)
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }
}

/// 解码失败的位置。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError {
    pub stage: &'static str,
    pub bit: usize,
    pub detail: i64,
}
impl DecodeError {
    fn new(stage: &'static str, bit: usize, detail: i64) -> Self {
        Self { stage, bit, detail }
    }
}
/// RL 表的查找表：**一个 u32 装下整个符号**，查表之后不再碰 RL 表。
///
/// 布局（位域）：
///
/// | 位 | 含义 |
/// |---|------|
/// | 0-1 | kind：0 = 普通符号、1 = EOB、2 = escape、3 = 长码 / 不该走表 |
/// | 2-6 | run（0..31） |
/// | 7-12 | level 的**幅度**（1..40；符号位在码里单独 1 位） |
/// | 13-17 | 码长（1..16） |
///
/// 为什么把 EOB / escape 也编进表：主循环每个符号原本要做
/// 「EOB 检查 + escape 检查 + RL 查表 + 符号位读取」四次独立取位，
/// 现在一次 16 位窗口 + 一次查表全部解决。level 恒为正（符号在码里），
/// run <= 31，所以 18 位就够 —— 65536 格 x 4 字节 = 256 KB，L2 放得下。
struct RlLut {
    slots: Vec<u32>,
}

const RL_SYM: u32 = 0;
const RL_EOB: u32 = 1;
const RL_ESC: u32 = 2;
const RL_LONG: u32 = 3;

fn build_rl_lut(table: &[RlEntry], eob: (u16, u8)) -> RlLut {
    let mut slots = vec![RL_LONG; 1usize << LUT_BITS];
    // 按**码长**从短到长填（和线性扫描的尝试顺序一致：短码永远赢）。
    for len in 1..=LUT_BITS {
        for entry in table {
            if entry.bits as u32 != len {
                continue;
            }
            let shift = LUT_BITS - len;
            let base = (entry.code as usize) << shift;
            let slot =
                RL_SYM | ((entry.run as u32) << 2) | ((entry.level as u32) << 7) | (len << 13);
            for s in slots.iter_mut().skip(base).take(1usize << shift) {
                if *s == RL_LONG {
                    *s = slot;
                }
            }
        }
    }
    // EOB / escape **后填并覆盖**：调用方的检查顺序就是 EOB 先、escape 次、RL 最后，
    // 所以同一段前缀上它们必须赢过表里的普通符号。
    for (code, bits, kind) in [
        (eob.0, eob.1 as u32, RL_EOB),
        (ESCAPE_CODE, ESCAPE_BITS as u32, RL_ESC),
    ] {
        if bits > LUT_BITS {
            continue;
        }
        let shift = LUT_BITS - bits;
        let base = (code as usize) << shift;
        let slot = kind | (bits << 13);
        for s in slots.iter_mut().skip(base).take(1usize << shift) {
            *s = slot;
        }
    }
    RlLut { slots }
}

/// 帧间 RL 查找表（MPEG-1 表 + 2 位 EOB）。**进程内只建一次** ——
/// 表来自 `const`，本来就可以共享（曾经每张图建一次，白烧两千多万次写入）。
fn inter_lut() -> &'static RlLut {
    static LUT: std::sync::OnceLock<RlLut> = std::sync::OnceLock::new();
    LUT.get_or_init(|| build_rl_lut(&MPEG1_RL, (EOB_CODE, EOB_BITS)))
}

/// 帧内 RL 查找表（`intra_vlc_format` 决定用 MPEG-1 还是 MPEG-2 的表 ——
/// 两者的 EOB 也不同：MPEG-1 是 `10`，MPEG-2 的 intra_vlc 表是 `0110`）。
fn intra_lut_for(mpeg2_table: bool) -> &'static RlLut {
    static M1: std::sync::OnceLock<RlLut> = std::sync::OnceLock::new();
    static M2: std::sync::OnceLock<RlLut> = std::sync::OnceLock::new();
    if mpeg2_table {
        M2.get_or_init(|| build_rl_lut(&MPEG2_RL, (0b0110, 4)))
    } else {
        M1.get_or_init(|| build_rl_lut(&MPEG1_RL, (EOB_CODE, EOB_BITS)))
    }
}

/// 通用前缀码查找表：一个 16 位窗口 → `(码长, 下标)`，一次查表解决。
///
/// 位域（`u32`）：
///
/// | 位 | 含义 |
/// |---|------|
/// | 0-4 | 码长（1..=16；0 = 该前缀不对应任何码） |
/// | 5-15 | 表下标 |
///
/// 为什么需要它：结构码（`macroblock_address_increment`、`motion_code`、
/// `coded_block_pattern`、`dct_dc_size`、`macroblock_type`）原本走的是
/// 「按码长从短到长 × 遍历整张表」的线性扫描 —— 解一个 MBA 增量最坏要做
/// 11 × 36 ≈ 400 次比较，而它**每个宏块至少调用一次**。剖析里这类结构码
/// 合计占了 34% 的解码时间，是当时最大的一块。
struct VlcLut {
    slots: Vec<u32>,
    /// 原始表，**只给尾部（不足 16 位）退化用**。
    codes: Vec<Code>,
}

/// 空槽：该前缀不对应任何码。
const VLC_INVALID: u32 = 0;

impl VlcLut {
    fn from_codes(codes: &[Code]) -> Self {
        Self::build(codes.len(), |i| (codes[i].code, codes[i].bits))
    }

    fn from_pairs(n: usize, at: impl Fn(usize) -> (u16, u8)) -> Self {
        Self::build(n, at)
    }

    fn build(n: usize, at: impl Fn(usize) -> (u16, u8)) -> Self {
        let mut slots = vec![VLC_INVALID; 1usize << LUT_BITS];
        let mut codes = Vec::with_capacity(n);
        for i in 0..n {
            let (code, bits) = at(i);
            codes.push(Code { code, bits });
        }
        // 按**码长从短到长**填，且「先填者赢」—— 与线性扫描的尝试顺序完全一致。
        for len in 1..=LUT_BITS as u8 {
            for (index, code) in codes.iter().enumerate() {
                if code.bits != len {
                    continue;
                }
                let shift = LUT_BITS - len as u32;
                let base = (code.code as usize) << shift;
                let slot = len as u32 | ((index as u32) << 5);
                for s in slots.iter_mut().skip(base).take(1usize << shift) {
                    if *s == VLC_INVALID {
                        *s = slot;
                    }
                }
            }
        }
        Self { slots, codes }
    }
}

/// 一次查表解一个前缀码。
#[inline]
fn decode_lut(r: &mut BitReader, lut: &VlcLut) -> Option<usize> {
    if r.bits_left() >= LUT_BITS as usize {
        let w = r.peek_u(LUT_BITS) as usize;
        let slot = lut.slots[w];
        let bits = slot & 31;
        if bits == 0 {
            return None;
        }
        r.advance(bits);
        return Some((slot >> 5) as usize);
    }
    // 尾部不足一个 16 位窗口：退化回线性扫描。语义与旧实现逐位一致，
    // 且只在每个 slice 的最后两个字节才会走到。
    let max_bits = lut.codes.iter().map(|c| c.bits).max().unwrap_or(0) as u32;
    for len in 1..=max_bits {
        if r.bits_left() < len as usize {
            return None;
        }
        let value = r.peek_u(len);
        for (index, code) in lut.codes.iter().enumerate() {
            if code.bits as u32 == len && code.code as u32 == value {
                r.skip(len).ok()?;
                return Some(index);
            }
        }
    }
    None
}

/// 结构码表各自**一张**查找表，进程内只建一次。
///
/// ⚠️ 必须「一张表一个 `OnceLock`」。曾经在 `decode_vlc` 内部用单个
/// `OnceLock` 缓存，结果第一张表（MBA 增量）的 LUT 被复用给了后面所有表 ——
/// 解出来的符号全错，而且症状离根因极远（表现为某个 slice 突然解不动）。
fn mb_addr_lut() -> &'static VlcLut {
    static L: std::sync::OnceLock<VlcLut> = std::sync::OnceLock::new();
    L.get_or_init(|| VlcLut::from_codes(&MB_ADDR_INCR))
}

fn mv_code_lut() -> &'static VlcLut {
    static L: std::sync::OnceLock<VlcLut> = std::sync::OnceLock::new();
    L.get_or_init(|| VlcLut::from_codes(&MV_CODE))
}

fn mb_pattern_lut() -> &'static VlcLut {
    static L: std::sync::OnceLock<VlcLut> = std::sync::OnceLock::new();
    L.get_or_init(|| VlcLut::from_codes(&MB_PATTERN))
}

fn dc_lum_lut() -> &'static VlcLut {
    static L: std::sync::OnceLock<VlcLut> = std::sync::OnceLock::new();
    L.get_or_init(|| VlcLut::from_codes(&DC_LUM))
}

fn dc_chroma_lut() -> &'static VlcLut {
    static L: std::sync::OnceLock<VlcLut> = std::sync::OnceLock::new();
    L.get_or_init(|| VlcLut::from_codes(&DC_CHROMA))
}

fn p_mb_type_lut() -> &'static VlcLut {
    static L: std::sync::OnceLock<VlcLut> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        VlcLut::from_pairs(P_MB_TYPE.len(), |i| (P_MB_TYPE[i].code, P_MB_TYPE[i].bits))
    })
}

fn b_mb_type_lut() -> &'static VlcLut {
    static L: std::sync::OnceLock<VlcLut> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        VlcLut::from_pairs(B_MB_TYPE.len(), |i| (B_MB_TYPE[i].code, B_MB_TYPE[i].bits))
    })
}
/// 在 run/level 表里按最长前缀找码。
fn decode_rl(r: &mut BitReader, table: &[RlEntry]) -> Option<usize> {
    let max_bits = table.iter().map(|e| e.bits).max().unwrap_or(0) as u32;
    for len in 1..=max_bits {
        if r.bits_left() < len as usize {
            return None;
        }
        let value = r.peek_u(len);
        for (index, entry) in table.iter().enumerate() {
            if entry.bits as u32 == len && entry.code as u32 == value {
                r.skip(len).ok()?;
                return Some(index);
            }
        }
    }
    None
}

/// P 帧 `macroblock_type`。
fn decode_p_mb_type(r: &mut BitReader) -> Option<usize> {
    decode_lut(r, p_mb_type_lut())
}

/// B 帧 `macroblock_type`。
fn decode_b_mb_type(r: &mut BitReader) -> Option<usize> {
    decode_lut(r, b_mb_type_lut())
}

/// zigzag 扫描顺序 → 光栅下标。
///
/// 码流里的系数按 zigzag 顺序排列，而量化矩阵与系数数组都是光栅顺序，
/// 所以两边都要过这张表。算法生成（按反对角线来回扫），不是手写常量。
pub const ZIGZAG: [usize; 64] = build_zigzag();

const fn build_zigzag() -> [usize; 64] {
    let mut table = [0usize; 64];
    let mut k = 0;
    let mut diagonal = 0;
    while diagonal < 15 {
        let mut step = 0;
        while step < 8 {
            // 偶数对角线从下往上，奇数从上往下 —— 这就是 zigzag 的定义。
            let (row, col) = if diagonal % 2 == 0 {
                (diagonal as isize - step, step)
            } else {
                (step, diagonal as isize - step)
            };
            if row >= 0 && row < 8 && col >= 0 && col < 8 && k < 64 {
                table[k] = (row as usize) * 8 + col as usize;
                k += 1;
            }
            step += 1;
        }
        diagonal += 1;
    }
    table
}

/// 反量化参数：MPEG-1 / MPEG-2、线性 / 非线性量化只在这里分叉。
///
/// MPEG-2 的量化器在反量化前被 `<<1`（线性）或查非线性表，所以它的移位比
/// MPEG-1 多一位，且**没有 oddification**（`2|level|+1` 已经把奇偶钉死）。
#[derive(Clone, Copy)]
struct Quant {
    q: i32,
    intra_shift: u32,
    inter_shift: u32,
    oddify: bool,
    dc_scale: i32,
    /// MPEG-2 的逃逸 level 是 12 位补码；MPEG-1 是 8 位 + 哨兵。
    mpeg2: bool,
}

impl Quant {
    fn new(scale: i32, mpeg2: bool, non_linear: bool, intra_dc_precision: u8) -> Self {
        if mpeg2 {
            let q = if non_linear {
                NON_LINEAR_QSCALE[scale as usize] as i32
            } else {
                scale << 1
            };
            Self {
                q,
                intra_shift: 4,
                inter_shift: 5,
                oddify: false,
                dc_scale: 8 >> intra_dc_precision,
                mpeg2: true,
            }
        } else {
            Self {
                q: scale,
                intra_shift: 3,
                inter_shift: 4,
                oddify: true,
                dc_scale: 8,
                mpeg2: false,
            }
        }
    }
}

/// 帧内反量化：DC 乘 `dc_scale`，交流 `(|level|·q·W) >> intra_shift`。
///
/// ⚠️ **只作测试参考**：生产路径已经把反量化融进 `decode_intra_block`
/// （同帧间的做法）。这条留着给 `fused_dequant_matches_array_dequant` 当 oracle。
#[cfg(test)]
fn dequantise_intra(quantised: &[i32; 64], quant: Quant) -> [i32; 64] {
    let mut out = [0i32; 64];
    out[0] = quantised[0] * quant.dc_scale;
    // MPEG-2 的 mismatch control：把块内系数和的奇偶钉住。
    let mut mismatch = out[0] ^ 1;
    for i in 1..64 {
        let level = quantised[i];
        if level == 0 {
            continue;
        }
        out[i] = dequant_intra_one(level, i, quant);
        mismatch ^= out[i];
    }
    if quant.mpeg2 {
        out[63] ^= mismatch & 1;
    }
    out
}

/// 反量化**一个**帧内交流系数（公式与 `dequantise_intra` 逐位相同）。
#[inline]
fn dequant_intra_one(level: i32, raster: usize, quant: Quant) -> i32 {
    let mut magnitude = (level.abs() * quant.q * INTRA_MATRIX[raster] as i32) >> quant.intra_shift;
    if quant.oddify {
        magnitude = (magnitude - 1) | 1;
    }
    if level < 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// 帧内版的「解到一个系数就就地反量化」。
#[inline]
fn put_intra_coeff(
    coeffs: &mut [i32; 64],
    mismatch: &mut i32,
    idx: usize,
    level: i32,
    scan: &[usize; 64],
    quant: Quant,
) {
    if idx < 64 {
        let v = dequant_intra_one(level, scan[idx], quant);
        coeffs[scan[idx]] = v;
        *mismatch ^= v;
    }
}

/// 帧间反量化：`((2|level|+1)·q·W) >> inter_shift`，DC 无特例。
///
/// ⚠️ **只作测试参考**：生产路径已经把反量化融进 `decode_inter_block`
/// （解到系数就地乘，省掉整趟 64 元素 pass）。这条留着给
/// `fused_dequant_matches_array_dequant` 当 oracle。
#[cfg(test)]
fn dequantise_inter(quantised: &[i32; 64], quant: Quant) -> [i32; 64] {
    let mut out = [0i32; 64];
    let mut mismatch = 1;
    for i in 0..64 {
        let level = quantised[i];
        if level == 0 {
            continue;
        }
        out[i] = dequant_inter_one(level, i, quant);
        mismatch ^= out[i];
    }
    if quant.mpeg2 {
        out[63] ^= mismatch & 1;
    }
    out
}

/// 反量化**一个**帧间系数（公式与 `dequantise_inter` 逐位相同，只是拆出来）。
#[inline]
fn dequant_inter_one(level: i32, raster: usize, quant: Quant) -> i32 {
    let mut magnitude =
        ((level.abs() * 2 + 1) * quant.q * NON_INTRA_MATRIX[raster] as i32) >> quant.inter_shift;
    if quant.oddify {
        magnitude = (magnitude - 1) | 1;
    }
    if level < 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// 融合路径里「解到一个系数就就地反量化」。
///
/// `idx` 是 **zigzag 序**，矩阵下标要用 `scan[idx]` 换回光栅序。
/// `idx >= 64` 的系数会被丢掉，**且不计入 `mismatch`** —— 与参考实现一致
/// （那里这些位置保持 0，`mismatch ^= 0` 是空操作）。
#[inline]
fn put_inter_coeff(
    coeffs: &mut [i32; 64],
    mismatch: &mut i32,
    idx: usize,
    level: i32,
    scan: &[usize; 64],
    quant: Quant,
) {
    if idx < 64 {
        let v = dequant_inter_one(level, scan[idx], quant);
        coeffs[scan[idx]] = v;
        *mismatch ^= v;
    }
}

/// 融合路径的收尾：MPEG-2 mismatch control 的最后一步。
#[inline]
fn finish_mismatch(mut coeffs: [i32; 64], mismatch: i32, quant: Quant) -> [i32; 64] {
    if quant.mpeg2 {
        coeffs[63] ^= mismatch & 1;
    }
    coeffs
}

/// 8×8 逆 DCT（残差块用）：**逐位复刻 ffmpeg `simple_idct` 的 8 位路径**。
///
/// ## 为什么换掉原来的浮点版
///
/// 原来是 f64 两天通道 + `.round()`：慢（占解码 28%），而且和 ffmpeg 的整数 IDCT
/// 有 ±2~3 的舍入差 —— 对账里「最差 2/3」全是它。移植 ffmpeg 的整数 IDCT 之后，
/// 两边**逐位相同**（判据：I 帧对 ffmpeg 的最差像素差从 2 掉到 0）。
///
/// ## 必须原样保留的三个细节（改了就不是同一个 IDCT）
///
/// 1. **DC-only 捷径**：一行里只有 DC 时，8 个输出都 = `(dc * 8) as i16`
///    （`& 0xffff` 后按 int16 读回）。不走通用路径 —— 两者的舍入不同。
/// 2. **列通道的 DC 取整**：`a0 = W4 * (dc + 32)`，那个 32 是
///    `(1 << (COL_SHIFT-1)) / W4 = 524288/16383` 的整数商（≈ 但不是 2^19/W4）。
/// 3. **无符号回绕运算**：ffmpeg 的累加器是 `unsigned`，负数系数先转无符号再乘。
///    Rust 侧用 `wrapping_*` 逐个对应；行/列结果都截回 i16 再进下一步。
///
/// 输入是**反量化后的系数**（光栅序，`block[v*8+u]`，v = 垂直频率）；
/// 系数按 int16 截断（ffmpeg 的反量化就写进 int16 缓冲区）。
/// 输出是**未裁剪**的残差 —— 裁剪由调用方在「预测 + 残差」之后做，
/// 与 ffmpeg 的 `idct_add` 语义一致。
fn idct_residual(block: &[i32; 64]) -> [i32; 64] {
    let _g = prof::enabled().then(|| prof::Guard::new(prof::IDCT));
    const W1: u32 = 22725;
    const W2: u32 = 21407;
    const W3: u32 = 19266;
    const W4: u32 = 16383;
    const W5: u32 = 12873;
    const W6: u32 = 8867;
    const W7: u32 = 4520;
    const ROW_SHIFT: u32 = 11;

    // 输入截成 int16（与 ffmpeg 的反量化缓冲区同宽）。
    let mut b = [0i32; 64];
    for (dst, src) in b.iter_mut().zip(block) {
        *dst = *src as i16 as i32;
    }

    // ---- 行通道（水平频率 u）----
    for v in 0..8 {
        let r = &mut b[v * 8..v * 8 + 8];
        let u = |i: usize| r[i] as u32; // 负数系数转无符号（与 C 的 (SUINT) 相同）
        // ① DC-only 捷径：1..7 全为 0 时，8 个输出都是 (dc*8) as i16。
        if r[1] == 0 && r[2] == 0 && r[3] == 0 && r[4] == 0 && r[5] == 0 && r[6] == 0 && r[7] == 0
        {
            let flat = (r[0].wrapping_mul(8)) as i16 as i32;
            r.fill(flat);
            continue;
        }
        let mut a0 = W4.wrapping_mul(u(0)).wrapping_add(1 << (ROW_SHIFT - 1));
        let mut a1 = a0;
        let mut a2 = a0;
        let mut a3 = a0;
        a0 = a0.wrapping_add(W2.wrapping_mul(u(2)));
        a1 = a1.wrapping_add(W6.wrapping_mul(u(2)));
        a2 = a2.wrapping_sub(W6.wrapping_mul(u(2)));
        a3 = a3.wrapping_sub(W2.wrapping_mul(u(2)));

        let mut b0 = W1.wrapping_mul(u(1));
        b0 = b0.wrapping_add(W3.wrapping_mul(u(3)));
        let mut b1 = W3.wrapping_mul(u(1));
        b1 = b1.wrapping_sub(W7.wrapping_mul(u(3)));
        let mut b2 = W5.wrapping_mul(u(1));
        b2 = b2.wrapping_sub(W1.wrapping_mul(u(3)));
        let mut b3 = W7.wrapping_mul(u(1));
        b3 = b3.wrapping_sub(W5.wrapping_mul(u(3)));

        // 4..7 全 0 时跳过（加 0 是空操作，纯优化）。
        if r[4] != 0 || r[5] != 0 || r[6] != 0 || r[7] != 0 {
            a0 = a0.wrapping_add(W4.wrapping_mul(u(4)).wrapping_add(W6.wrapping_mul(u(6))));
            a1 = a1.wrapping_sub(W4.wrapping_mul(u(4)).wrapping_add(W2.wrapping_mul(u(6))));
            a2 = a2.wrapping_sub(W4.wrapping_mul(u(4)).wrapping_sub(W2.wrapping_mul(u(6))));
            a3 = a3.wrapping_add(W4.wrapping_mul(u(4)).wrapping_sub(W6.wrapping_mul(u(6))));

            b0 = b0.wrapping_add(W5.wrapping_mul(u(5)).wrapping_add(W7.wrapping_mul(u(7))));
            b1 = b1.wrapping_sub(W1.wrapping_mul(u(5)).wrapping_add(W5.wrapping_mul(u(7))));
            b2 = b2.wrapping_add(W7.wrapping_mul(u(5)).wrapping_add(W3.wrapping_mul(u(7))));
            b3 = b3.wrapping_add(W3.wrapping_mul(u(5)).wrapping_sub(W1.wrapping_mul(u(7))));
        }

        r[0] = ((a0.wrapping_add(b0) as i32) >> ROW_SHIFT) as i16 as i32;
        r[7] = ((a0.wrapping_sub(b0) as i32) >> ROW_SHIFT) as i16 as i32;
        r[1] = ((a1.wrapping_add(b1) as i32) >> ROW_SHIFT) as i16 as i32;
        r[6] = ((a1.wrapping_sub(b1) as i32) >> ROW_SHIFT) as i16 as i32;
        r[2] = ((a2.wrapping_add(b2) as i32) >> ROW_SHIFT) as i16 as i32;
        r[5] = ((a2.wrapping_sub(b2) as i32) >> ROW_SHIFT) as i16 as i32;
        r[3] = ((a3.wrapping_add(b3) as i32) >> ROW_SHIFT) as i16 as i32;
        r[4] = ((a3.wrapping_sub(b3) as i32) >> ROW_SHIFT) as i16 as i32;
    }

    // ---- 列通道（垂直频率 v）----
    let mut out = [0i32; 64];
    // 列通道可以整个换掉（unsafe/FFI 加速）。默认标量；`v2v-mpeg12-simd`
    // 可以装一个 NEON/SSE2 版本进来 —— **约定：必须与标量版逐位相同**。
    // 每块一次原子读，相对 64 个系数的变换可以忽略。
    let cols = IDCT_COLS.get().copied().unwrap_or(idct_cols_scalar);
    cols(&b, &mut out);
    out
}

/// 列通道内核的签名：`b` 是行变换后的中间块（i16 值存在 i32 里），
/// 结果写到 `out`（光栅序，未裁剪）。
pub type IdctCols = fn(&[i32; 64], &mut [i32; 64]);

/// 可替换的列通道内核。默认没装 = 走标量。
static IDCT_COLS: std::sync::OnceLock<IdctCols> = std::sync::OnceLock::new();

/// 装一个列通道内核（只生效一次；重复调用忽略）。
///
/// ⚠️ 装进来的实现**必须与 `idct_cols_scalar` 逐位相同** —— 这是解码器
/// 对账「逐字节一致」的地基。`v2v-mpeg12-simd` 里有等价的穷举测试。
pub fn set_idct_cols(f: IdctCols) {
    let _ = IDCT_COLS.set(f);
}

/// 列通道的标量实现（参考实现 / oracle）。
///
/// 注意 `c(0) + 32` 里的 32：它是 `(1 << (COL_SHIFT-1)) / W4 = 524288/16383`
/// 的**整数商**，不是 `2^19 / W4`。换成后者就不是同一个 IDCT 了。
pub fn idct_cols_scalar(b: &[i32; 64], out: &mut [i32; 64]) {
    const W1: u32 = 22725;
    const W2: u32 = 21407;
    const W3: u32 = 19266;
    const W4: u32 = 16383;
    const W5: u32 = 12873;
    const W6: u32 = 8867;
    const W7: u32 = 4520;
    const COL_SHIFT: u32 = 20;

    for u in 0..8 {
        let c = |i: usize| b[i * 8 + u] as u32;
        let mut a0 = W4.wrapping_mul(c(0).wrapping_add(32));
        let mut a1 = a0;
        let mut a2 = a0;
        let mut a3 = a0;
        a0 = a0.wrapping_add(W2.wrapping_mul(c(2)));
        a1 = a1.wrapping_add(W6.wrapping_mul(c(2)));
        a2 = a2.wrapping_sub(W6.wrapping_mul(c(2)));
        a3 = a3.wrapping_sub(W2.wrapping_mul(c(2)));

        let mut b0 = W1.wrapping_mul(c(1));
        b0 = b0.wrapping_add(W3.wrapping_mul(c(3)));
        let mut b1 = W3.wrapping_mul(c(1));
        b1 = b1.wrapping_sub(W7.wrapping_mul(c(3)));
        let mut b2 = W5.wrapping_mul(c(1));
        b2 = b2.wrapping_sub(W1.wrapping_mul(c(3)));
        let mut b3 = W7.wrapping_mul(c(1));
        b3 = b3.wrapping_sub(W5.wrapping_mul(c(3)));

        if c(4) != 0 {
            a0 = a0.wrapping_add(W4.wrapping_mul(c(4)));
            a1 = a1.wrapping_sub(W4.wrapping_mul(c(4)));
            a2 = a2.wrapping_sub(W4.wrapping_mul(c(4)));
            a3 = a3.wrapping_add(W4.wrapping_mul(c(4)));
        }
        if c(5) != 0 {
            b0 = b0.wrapping_add(W5.wrapping_mul(c(5)));
            b1 = b1.wrapping_sub(W1.wrapping_mul(c(5)));
            b2 = b2.wrapping_add(W7.wrapping_mul(c(5)));
            b3 = b3.wrapping_add(W3.wrapping_mul(c(5)));
        }
        if c(6) != 0 {
            a0 = a0.wrapping_add(W6.wrapping_mul(c(6)));
            a1 = a1.wrapping_sub(W2.wrapping_mul(c(6)));
            a2 = a2.wrapping_add(W2.wrapping_mul(c(6)));
            a3 = a3.wrapping_sub(W6.wrapping_mul(c(6)));
        }
        if c(7) != 0 {
            b0 = b0.wrapping_add(W7.wrapping_mul(c(7)));
            b1 = b1.wrapping_sub(W5.wrapping_mul(c(7)));
            b2 = b2.wrapping_add(W3.wrapping_mul(c(7)));
            b3 = b3.wrapping_sub(W1.wrapping_mul(c(7)));
        }

        out[0 * 8 + u] = (a0.wrapping_add(b0) as i32) >> COL_SHIFT;
        out[1 * 8 + u] = (a1.wrapping_add(b1) as i32) >> COL_SHIFT;
        out[2 * 8 + u] = (a2.wrapping_add(b2) as i32) >> COL_SHIFT;
        out[3 * 8 + u] = (a3.wrapping_add(b3) as i32) >> COL_SHIFT;
        out[4 * 8 + u] = (a3.wrapping_sub(b3) as i32) >> COL_SHIFT;
        out[5 * 8 + u] = (a2.wrapping_sub(b2) as i32) >> COL_SHIFT;
        out[6 * 8 + u] = (a1.wrapping_sub(b1) as i32) >> COL_SHIFT;
        out[7 * 8 + u] = (a0.wrapping_sub(b0) as i32) >> COL_SHIFT;
    }
}

/// 把残差（+ 一个偏移）写到平面，夹到 0..=255。帧内块偏移 128。
fn store_block(residual: &[i32; 64], offset: i32, out: &mut [u8], stride: usize, at: usize) {
    let _g = prof::enabled().then(|| prof::Guard::new(prof::STORE));
    for y in 0..8 {
        // 一行先攒成 `[u8; 8]` 再整块写：直接逐像素下标的话，LLVM 既证明不了
        // 边界（每像素一次检查），也没法把 8 次单字节写合成一次 8 字节写。
        let mut row = [0u8; 8];
        for x in 0..8 {
            row[x] = (residual[y * 8 + x] + offset).clamp(0, 255) as u8;
        }
        let base = at + y * stride;
        out[base..base + 8].copy_from_slice(&row);
    }
}

/// 预测块 + 残差，夹到 0..=255。帧间块用。
///
/// 预测是 u8、残差是 i32：预测本来就在 0..=255，用 i32 存只是白烧 4 倍流量。
fn store_pred(pred: &[u8; 64], residual: &[i32; 64], out: &mut [u8], stride: usize, at: usize) {
    let _g = prof::enabled().then(|| prof::Guard::new(prof::STORE));
    for y in 0..8 {
        // 同 `store_block`：一行攒成 8 字节再整块写。
        let mut row = [0u8; 8];
        for x in 0..8 {
            row[x] = (pred[y * 8 + x] as i32 + residual[y * 8 + x]).clamp(0, 255) as u8;
        }
        let base = at + y * stride;
        out[base..base + 8].copy_from_slice(&row);
    }
}

/// 只写预测（没有残差）：预测已经是 0..=255，不用再夹。
fn store_pixels(pred: &[u8; 64], out: &mut [u8], stride: usize, at: usize) {
    let _g = prof::enabled().then(|| prof::Guard::new(prof::STORE));
    for y in 0..8 {
        out[at + y * stride..at + y * stride + 8].copy_from_slice(&pred[y * 8..y * 8 + 8]);
    }
}

/// 把帧内残差写到宏块的某个块（按帧 / 场 DCT 的几何）。
#[allow(clippy::too_many_arguments)]
fn store_mb_block(
    frame: &mut Frame,
    block: usize,
    column: usize,
    mb_row: usize,
    width: usize,
    cw: usize,
    interlaced_dct: bool,
    residual: &[i32; 64],
    offset: i32,
    chroma_format: u8,
) {
    let (chroma, origin_x, origin_y, stride, _) =
        block_geometry(block, column, mb_row, width, cw, interlaced_dct, chroma_format);
    let plane_width = if chroma { cw } else { width };
    let plane = match block {
        0..=3 => &mut frame.y,
        4 | 6 | 8 | 10 => &mut frame.u,
        _ => &mut frame.v,
    };
    store_block(residual, offset, plane, stride, origin_y * plane_width + origin_x);
}

/// 取预测 + 叠残差，写到宏块的某个块（按帧 / 场 DCT 的几何）。
#[allow(clippy::too_many_arguments)]
fn store_pred_block(
    frame: &mut Frame,
    reference: &Frame,
    block: usize,
    column: usize,
    mb_row: usize,
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    interlaced_dct: bool,
    mv_x: i32,
    mv_y: i32,
    residual: &[i32; 64],
    chroma_format: u8,
) {
    let (chroma, origin_x, origin_y, stride, row_step) =
        block_geometry(block, column, mb_row, width, cw, interlaced_dct, chroma_format);
    let plane_width = if chroma { cw } else { width };
    let (rw, rh) = if chroma { (cw, ch) } else { (width, height) };
    let ref_plane = match block {
        0..=3 => &reference.y,
        4 | 6 | 8 | 10 => &reference.u,
        _ => &reference.v,
    };
    let pred = predict_block(
        ref_plane, plane_width, rw, rh, origin_x, origin_y, mv_x, mv_y, chroma, row_step, 1,
        chroma_format,
    );
    let plane = match block {
        0..=3 => &mut frame.y,
        4 | 6 | 8 | 10 => &mut frame.u,
        _ => &mut frame.v,
    };
    store_pred(&pred, residual, plane, stride, origin_y * plane_width + origin_x);
}

/// 解一个帧内块。
fn decode_intra_block(
    r: &mut BitReader,
    quant: Quant,
    lut: &RlLut,
    dc_lut: &VlcLut,
    dc_predictor: &mut i32,
    rl: &[RlEntry],
    scan: &[usize; 64],
    eob: (u16, u8),
) -> Result<[i32; 64], DecodeError> {
    let _g = prof::enabled().then(|| prof::Guard::new(prof::BITS));
    let mut coeffs = [0i32; 64];

    let size = decode_lut(r, dc_lut)
        .ok_or_else(|| DecodeError::new("dc_size", r.bit_pos(), 0))?;
    if size > 0 {
        let raw = r
            .u(size as u32)
            .map_err(|_| DecodeError::new("dc_diff", r.bit_pos(), size as i64))? as i32;
        let half = 1i32 << (size - 1);
        let diff = if raw & half != 0 { raw } else { raw - (half << 1) + 1 };
        // 帧内 DC 是**差分**编码：要累加到预测值上，否则整块偏掉。
        *dc_predictor += diff;
    }
    coeffs[0] = *dc_predictor * quant.dc_scale;
    // 反量化融进解码（同帧间）：DC 先单独做，AC 解到就乘。
    let mut mismatch = coeffs[0] ^ 1;

    let mut index = 1usize;
    // 内层循环改用本地位缓存（一次装 64 位，够连着服务好几个符号）。
    // 慢路（escape / 尾部）会把游标交还给 `r`，所以 `r` 始终是事实来源。
    let mut c = BitCache::new(r.data(), r.bit_pos());
    while index < 64 {
        let left = c.bits_left();
        if left == 0 {
            break;
        }
        // ── 快路：一次 17 位窗口同时解决 EOB / escape / RL / 符号位 ──
        // 窗口 = 16 位码前缀 + 紧接着的 1 位符号，所以**16 位的长码也走这条路**
        // （早先只用 16 位窗口，长码全掉到慢路上去 —— 高码率素材里长码很常见）。
        // escape 和「一段不足 17 位」才落回下面的原路径（原路径本来就是判据）。
        if left >= LUT_BITS as usize + 1 {
            let w = c.peek(LUT_BITS + 1);
            let slot = lut.slots[(w >> 1) as usize];
            let bits = slot >> 13;
            match slot & 3 {
                RL_SYM => {
                    let run = ((slot >> 2) & 31) as usize;
                    let level = ((slot >> 7) & 63) as i32;
                    // 窗口是 17 位：码占高 16 位里的前 `bits` 位，符号位在
                    // 第 `16 - bits` 位（不是最低位 —— 只有 bits == 16 时才是）。
                    let negative = (w >> (LUT_BITS - bits)) & 1 == 1;
                    c.advance(bits + 1);
                    index += run;
                    if index < 64 {
                        put_intra_coeff(&mut coeffs, &mut mismatch, index, if negative { -level } else { level }, scan, quant);
                    }
                    index += 1;
                    continue;
                }
                RL_EOB => {
                    c.advance(bits);
                    break;
                }
                _ => {}
            }
        }
        // ── 慢路：把游标交还给 `r`（escape / 尾部要用 `r.u` / `r.skip`）──
        r.set_pos(c.pos());
        if r.bits_left() >= eob.1 as usize && r.peek_u(eob.1 as u32) == eob.0 as u32 {
            r.skip(eob.1 as u32)
                .map_err(|_| DecodeError::new("eob", r.bit_pos(), 0))?;
            c = BitCache::new(r.data(), r.bit_pos());
            break;
        }
        if r.bits_left() >= ESCAPE_BITS as usize
            && r.peek_u(ESCAPE_BITS as u32) == ESCAPE_CODE as u32
        {
            r.skip(ESCAPE_BITS as u32)
                .map_err(|_| DecodeError::new("escape", r.bit_pos(), 0))?;
            let run = r
                .u(6)
                .map_err(|_| DecodeError::new("escape_run", r.bit_pos(), 0))? as usize;
            // MPEG-2 的逃逸 level 是 **12 位补码**；MPEG-1 是 8 位 + 哨兵。
            let level = if quant.mpeg2 {
                let raw = r
                    .u(12)
                    .map_err(|_| DecodeError::new("escape_level12", r.bit_pos(), 0))?
                    as i32;
                if raw & 0x800 != 0 { raw - 4096 } else { raw }
            } else {
                // 8 位补码，然后两个**哨兵值**各有专门含义（照参考实现）：
                //   -128 → 再读 8 位，值减 256（覆盖 -256..-1）
                //      0 → 再读 8 位，值即 level（覆盖 1..255）
                let raw = r
                    .u(8)
                    .map_err(|_| DecodeError::new("escape_level", r.bit_pos(), 0))? as i32;
                let mut level = if raw & 0x80 != 0 { raw - 256 } else { raw };
                if level == -128 {
                    let extra = r
                        .u(8)
                        .map_err(|_| DecodeError::new("escape_neg", r.bit_pos(), 0))? as i32;
                    level = extra - 256;
                } else if level == 0 {
                    level = r
                        .u(8)
                        .map_err(|_| DecodeError::new("escape_pos", r.bit_pos(), 0))?
                        as i32;
                }
                level
            };
            index += run;
            if index < 64 {
                put_intra_coeff(&mut coeffs, &mut mismatch, index, level, scan, quant);
            }
            index += 1;
            c = BitCache::new(r.data(), r.bit_pos());
            continue;
        }

        let entry =
            decode_rl(r, rl)
                .ok_or_else(|| DecodeError::new("rl", r.bit_pos(), index as i64))?;
        let run = rl[entry].run as usize;
        let level = rl[entry].level as i32;
        let negative = r
            .u(1)
            .map_err(|_| DecodeError::new("rl_sign", r.bit_pos(), 0))?
            == 1;
        index += run;
        if index < 64 {
            put_intra_coeff(&mut coeffs, &mut mismatch, index, if negative { -level } else { level }, scan, quant);
        }
        index += 1;
        c = BitCache::new(r.data(), r.bit_pos());
    }
    r.set_pos(c.pos());

    // 块被 64 个系数填满时，EOB 仍然在码流里 —— 不消费它，后面全部错位。
    if index >= 64
        && r.bits_left() >= eob.1 as usize
        && r.peek_u(eob.1 as u32) == eob.0 as u32
    {
        r.skip(eob.1 as u32)
            .map_err(|_| DecodeError::new("eob_full", r.bit_pos(), 0))?;
    }

    Ok(finish_mismatch(coeffs, mismatch, quant))
}

/// 解一个帧间块（P/B 帧残差）。没有 DC 预测，系数从扫描第 0 位开始。
fn decode_inter_block(
    r: &mut BitReader,
    quant: Quant,
    lut: &RlLut,
    rl: &[RlEntry],
    scan: &[usize; 64],
    eob: (u16, u8),
    field_picture: bool,
) -> Result<[i32; 64], DecodeError> {
    let _g = prof::enabled().then(|| prof::Guard::new(prof::BITS));
    let mut coeffs = [0i32; 64];
    let mut index = 0usize;
    // 反量化**融进解码**：解到一个系数就地乘，省掉原来那趟 64 元素 pass。
    // `mismatch` 是 MPEG-2 mismatch control 的累加器，等价于参考实现的
    // `mismatch ^= out[i]`（每个非零系数只写一次，且 `index` 单调递增）。
    let mut mismatch = 1i32;

    // 首系数快速路径（照参考实现）：首码以 '1' 开头时就是 (run 0, level 1)，
    // 第 2 位当符号，**只吃 2 位**。走通用 RL 会多吃 1 位，从第二个色度块起就错位。
    if r.bits_left() >= 2 && r.peek_u(1) == 1 {
        let negative = (r.peek_u(2) & 1) == 1;
        r.skip(2).ok();
        put_inter_coeff(
            &mut coeffs,
            &mut mismatch,
            0,
            if negative { -1 } else { 1 },
            scan,
            quant,
        );
        index = 1;
        // 场图的参考实现用 `GET_CACHE <= 0xBFFFFFFF`（下两位不是 `11`）判 EOB；
        // 帧图用显式的 `10`。两者不能混用，否则整段位流会偏。
        let eob_hit = if field_picture {
            r.bits_left() >= 2 && r.peek_u(2) != 0b11
        } else {
            r.bits_left() >= eob.1 as usize && r.peek_u(eob.1 as u32) == eob.0 as u32
        };
        if eob_hit {
            r.skip(2)
                .map_err(|_| DecodeError::new("eob_fast", r.bit_pos(), 0))?;
            return Ok(finish_mismatch(coeffs, mismatch, quant));
        }
    }

    // 内层循环改用本地位缓存（一次装 64 位，够连着服务好几个符号）。
    // 慢路（escape / 尾部）会把游标交还给 `r`，所以 `r` 始终是事实来源。
    let mut c = BitCache::new(r.data(), r.bit_pos());
    while index < 64 {
        let left = c.bits_left();
        if left == 0 {
            break;
        }
        // ── 快路：一次 17 位窗口同时解决 EOB / escape / RL / 符号位 ──
        // 窗口 = 16 位码前缀 + 紧接着的 1 位符号，所以**16 位的长码也走这条路**
        // （早先只用 16 位窗口，长码全掉到慢路上去 —— 高码率素材里长码很常见）。
        // escape 和「一段不足 17 位」才落回下面的原路径（原路径本来就是判据）。
        if left >= LUT_BITS as usize + 1 {
            let w = c.peek(LUT_BITS + 1);
            let slot = lut.slots[(w >> 1) as usize];
            let bits = slot >> 13;
            match slot & 3 {
                RL_SYM => {
                    let run = ((slot >> 2) & 31) as usize;
                    let level = ((slot >> 7) & 63) as i32;
                    // 窗口是 17 位：码占高 16 位里的前 `bits` 位，符号位在
                    // 第 `16 - bits` 位（不是最低位 —— 只有 bits == 16 时才是）。
                    let negative = (w >> (LUT_BITS - bits)) & 1 == 1;
                    c.advance(bits + 1);
                    index += run;
                    if index < 64 {
                        put_inter_coeff(&mut coeffs, &mut mismatch, index, if negative { -level } else { level }, scan, quant);
                    }
                    index += 1;
                    continue;
                }
                RL_EOB => {
                    c.advance(bits);
                    break;
                }
                _ => {}
            }
        }
        // ── 慢路：把游标交还给 `r`（escape / 尾部要用 `r.u` / `r.skip`）──
        r.set_pos(c.pos());
        if r.bits_left() >= eob.1 as usize && r.peek_u(eob.1 as u32) == eob.0 as u32 {
            r.skip(eob.1 as u32)
                .map_err(|_| DecodeError::new("eob_inter", r.bit_pos(), 0))?;
            c = BitCache::new(r.data(), r.bit_pos());
            break;
        }
        if r.bits_left() >= ESCAPE_BITS as usize
            && r.peek_u(ESCAPE_BITS as u32) == ESCAPE_CODE as u32
        {
            r.skip(ESCAPE_BITS as u32)
                .map_err(|_| DecodeError::new("escape_inter", r.bit_pos(), 0))?;
            let run = r
                .u(6)
                .map_err(|_| DecodeError::new("escape_run_inter", r.bit_pos(), 0))?
                as usize;
            let level = if quant.mpeg2 {
                let raw = r
                    .u(12)
                    .map_err(|_| DecodeError::new("escape_level12_inter", r.bit_pos(), 0))?
                    as i32;
                if raw & 0x800 != 0 { raw - 4096 } else { raw }
            } else {
                let raw = r
                    .u(8)
                    .map_err(|_| DecodeError::new("escape_level_inter", r.bit_pos(), 0))?
                    as i32;
                let mut level = if raw & 0x80 != 0 { raw - 256 } else { raw };
                if level == -128 {
                    let extra = r
                        .u(8)
                        .map_err(|_| DecodeError::new("escape_neg_inter", r.bit_pos(), 0))?
                        as i32;
                    level = extra - 256;
                } else if level == 0 {
                    level = r
                        .u(8)
                        .map_err(|_| DecodeError::new("escape_pos_inter", r.bit_pos(), 0))?
                        as i32;
                }
                level
            };
            index += run;
            if index < 64 {
                put_inter_coeff(&mut coeffs, &mut mismatch, index, level, scan, quant);
            }
            index += 1;
            c = BitCache::new(r.data(), r.bit_pos());
            continue;
        }

        let entry = decode_rl(r, rl)
            .ok_or_else(|| DecodeError::new("rl_inter", r.bit_pos(), index as i64))?;
        let run = rl[entry].run as usize;
        let level = rl[entry].level as i32;
        let negative = r
            .u(1)
            .map_err(|_| DecodeError::new("rl_sign_inter", r.bit_pos(), 0))?
            == 1;
        index += run;
        if index < 64 {
            put_inter_coeff(&mut coeffs, &mut mismatch, index, if negative { -level } else { level }, scan, quant);
        }
        index += 1;
        c = BitCache::new(r.data(), r.bit_pos());
    }
    r.set_pos(c.pos());

    if index >= 64
        && r.bits_left() >= eob.1 as usize
        && r.peek_u(eob.1 as u32) == eob.0 as u32
    {
        r.skip(eob.1 as u32)
            .map_err(|_| DecodeError::new("eob_full_inter", r.bit_pos(), 0))?;
    }

    Ok(finish_mismatch(coeffs, mismatch, quant))
}

/// 把 `val` 当 `bits` 位补码解释（MPEG 运动矢量的模运算）。
fn sign_extend(val: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (val << shift) >> shift
}

/// 解一个运动矢量分量（`motion_code` + 残差 + 预测），返回**半像素**单位。
/// 读 `macroblock_address_increment`：返回**要从当前位置前进多少个宏块**；
/// `None` = 这个 slice 到此结束。
///
/// ⚠️ 表里有三个**不是增量**的符号（下标 33 / 34 / 35）：
/// - 33 = `macroblock_escape`：**再加 33** 后继续读下一个增量（一帧里连续跳过的
///   宏块可能超过 33 个）；
/// - 34 = stuffing：跳过，继续读；
/// - 35 = 片结束（后面还会跟一串 0 位）。
///
/// 把它们当成「增量 34 / 35 / 36」会让位游标整体错位 —— 症状是解到某一片时
/// `coded_block_pattern` 的 VLC 突然解不出来。实测 MPEG-1 的 B 帧末行必现
/// （那里正好是一长串跳过宏块 + escape）。
fn decode_mb_increment(r: &mut BitReader) -> Option<isize> {
    let mut total = 0isize;
    loop {
        match decode_lut(r, mb_addr_lut())? {
            33 => total += 33,
            34 => {}
            35 => return None,
            code => return Some(total + code as isize + 1),
        }
    }
}

fn decode_motion(r: &mut BitReader, fcode: u32, pred: i32) -> Option<i32> {
    let code = decode_lut(r, mv_code_lut())? as i32;
    if code == 0 {
        return Some(pred);
    }
    let sign = r.u(1).ok()? == 1;
    let shift = fcode.saturating_sub(1);
    let mut val = code;
    if shift > 0 {
        val = (val - 1) << shift;
        val |= r.u(shift).ok()? as i32;
        val += 1;
    }
    if sign {
        val = -val;
    }
    Some(sign_extend(val + pred, 5 + shift))
}

/// 色度平面的尺寸。
fn chroma_dims(chroma_format: u8, width: usize, height: usize) -> (usize, usize) {
    match chroma_format {
        2 => (width.div_ceil(2), height),
        3 => (width, height),
        _ => (width.div_ceil(2), height.div_ceil(2)),
    }
}

/// 一个宏块的块数：4:2:0 = 6，4:2:2 = 8，4:4:4 = 12。
fn mb_block_count(chroma_format: u8) -> usize {
    4 + (1 << chroma_format)
}

/// 解一个 dual prime 的差分矢量（`dmv`：先 1 位，非零再 1 位定 ±1）。
fn decode_dmv(r: &mut BitReader) -> Option<i32> {
    if r.u(1).ok()? == 1 {
        Some(1 - ((r.u(1).ok()? as i32) << 1))
    } else {
        Some(0)
    }
}

/// 宏块内第 `block` 个块的几何：`(是否色度, 原点 x, 原点 y, 写步长, 取行步长)`。
///
/// 场 DCT（`interlaced_dct=1`）时亮度块隔行写 / 取：块 0/1 在偶行、2/3 在奇行，
/// 行步长 2；色度在 4:2:0 下始终按帧 DCT 摆放。
fn block_geometry(
    block: usize,
    column: usize,
    mb_row: usize,
    width: usize,
    cw: usize,
    interlaced_dct: bool,
    chroma_format: u8,
) -> (bool, usize, usize, usize, i32) {
    match block {
        0..=3 => {
            let (bx, by) = (block % 2, block / 2);
            let origin_x = column * 16 + bx * 8;
            if interlaced_dct {
                (false, origin_x, mb_row * 16 + by, 2 * width, 2)
            } else {
                (false, origin_x, mb_row * 16 + by * 8, width, 1)
            }
        }
        _ => {
            // 4:2:0：一个 Cb / Cr；4:2:2：上下两个；4:4:4：2×2 四个。
            let (bx, by) = match chroma_format {
                2 => (0, if block >= 6 { 1 } else { 0 }),
                3 => (
                    if block >= 8 { 1 } else { 0 },
                    if (block % 4) >= 2 { 1 } else { 0 },
                ),
                _ => (0, 0),
            };
            let chroma_mb_w = if chroma_format == 3 { 16 } else { 8 };
            let origin_x = column * chroma_mb_w + bx * 8;
            if chroma_format == 1 {
                // 4:2:0 的色度始终帧基（参考实现也只用 uvlinesize）。
                (true, origin_x, mb_row * 8, cw, 1)
            } else if interlaced_dct {
                (true, origin_x, mb_row * 16 + by, 2 * cw, 2)
            } else {
                (true, origin_x, mb_row * 16 + by * 8, cw, 1)
            }
        }
    }
}

/// 从参考帧按运动矢量取一个 8×8 预测块（半像素双线性）。
///
/// 亮度：整数部分 `mv >> 1`、小数位 `mv & 1`。
/// 色度：先把矢量按 `mv / 2`（向零截断）缩到色度采样，再 `>> 1` 取整数、
/// `& 1` 取小数 —— 这两步不能合成 `mv >> 2`，负矢量会差一格。
fn predict_block(
    reference: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    mv_x: i32,
    mv_y: i32,
    chroma: bool,
    row_step: i32,
    mv_y_scale: i32,
    chroma_format: u8,
) -> [u8; 64] {
    let _g = prof::enabled().then(|| prof::Guard::new(prof::MC));
    let (mx, my) = if chroma {
        let (hs, vs) = match chroma_format {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        (mv_x / hs, mv_y / vs)
    } else {
        (mv_x, mv_y)
    };
    let ix = mx >> 1;
    let iy = (my >> 1) * mv_y_scale;
    let fx = mx & 1;
    let fy = my & 1;
    // 预测值本身就是 0..=255 的像素（u8 的平均还在 0..=255 内），所以存 u8 ——
    // 存 i32 要多 4 倍内存流量，而且挡住 u8 的向量化平均。
    let mut out = [0u8; 64];
    let sx = origin_x as i32 + ix;
    let sy = origin_y as i32 + row_step * 0 + iy; // 块左上角的参考坐标
    // 四个抽头在这一块上覆盖的包围盒。整块都在帧内 → 一个像素都不用 clamp
    //（clamp 只对帧边界有意义，而边界块是极少数）。
    let last_y = 7 * row_step + fy * row_step;
    let (ylo, yhi) = if last_y >= 0 {
        (sy, sy + last_y)
    } else {
        (sy + last_y, sy)
    };
    let interior = sx >= 0
        && sx + 7 + fx < width as i32
        && ylo >= 0
        && yhi < height as i32
        && sy >= 0
        && sy + 7 * row_step >= 0;
    if interior {
        // ── 快路：行首地址一次算好，内层直接下标 ──
        let row0 = (sy as usize) * stride + sx as usize;
        let row1 = ((sy + row_step) as usize) * stride + sx as usize;
        match (fx, fy) {
            (0, 0) => {
                for y in 0..8 {
                    let base = row0 + y * row_step as usize * stride;
                    for x in 0..8 {
                        out[y * 8 + x] = reference[base + x];
                    }
                }
            }
            (1, 0) => {
                for y in 0..8 {
                    let base = row0 + y * row_step as usize * stride;
                    for x in 0..8 {
                        let a = reference[base + x] as u16;
                        let b = reference[base + x + 1] as u16;
                        out[y * 8 + x] = ((a + b + 1) >> 1) as u8;
                    }
                }
            }
            (0, 1) => {
                for y in 0..8 {
                    let a0 = row0 + y * row_step as usize * stride;
                    let b0 = row1 + y * row_step as usize * stride;
                    for x in 0..8 {
                        let a = reference[a0 + x] as u16;
                        let b = reference[b0 + x] as u16;
                        out[y * 8 + x] = ((a + b + 1) >> 1) as u8;
                    }
                }
            }
            _ => {
                for y in 0..8 {
                    let a0 = row0 + y * row_step as usize * stride;
                    let b0 = row1 + y * row_step as usize * stride;
                    for x in 0..8 {
                        let a = reference[a0 + x] as u16;
                        let b = reference[a0 + x + 1] as u16;
                        let c = reference[b0 + x] as u16;
                        let d = reference[b0 + x + 1] as u16;
                        out[y * 8 + x] = ((a + b + c + d + 2) >> 2) as u8;
                    }
                }
            }
        }
        return out;
    }
    // ── 慢路：贴着帧边界的块，逐像素 clamp（和原来逐位相同）──
    for y in 0..8 {
        for x in 0..8 {
            let sx = origin_x as i32 + ix + x as i32;
            let sy = origin_y as i32 + row_step * y as i32 + iy;
            let at = |dx: i32, dy: i32| -> u16 {
                let px = (sx + dx).clamp(0, width as i32 - 1) as usize;
                let py = (sy + dy).clamp(0, height as i32 - 1) as usize;
                reference[py * stride + px] as u16
            };
            let a = at(0, 0);
            out[y * 8 + x] = match (fx, fy) {
                (0, 0) => a as u8,
                (1, 0) => ((a + at(1, 0) + 1) >> 1) as u8,
                (0, 1) => ((a + at(0, row_step) + 1) >> 1) as u8,
                _ => ((a + at(1, 0) + at(0, row_step) + at(1, row_step) + 2) >> 2) as u8,
            };
        }
    }
    out
}

/// 每幅图不变的块级配置：RL 表、扫描顺序、EOB 码。
struct BlockTables<'a> {
    intra_rl: &'a [RlEntry],
    inter_rl: &'a [RlEntry],
    /// 上面两张 RL 表的查找表（进程内建一次，见 `inter_lut` / `intra_lut_for`）。
    intra_lut: &'static RlLut,
    inter_lut: &'static RlLut,
    scan: &'a [usize; 64],
    intra_eob: (u16, u8),
    inter_eob: (u16, u8),
}

fn block_tables(ext: Option<&crate::PictureCodingExtension>) -> BlockTables<'static> {
    match ext {
        Some(ext) => BlockTables {
            // `ff_mpeg2_vlc_table` 只服务 `intra_vlc_format=1` 的**帧内**块；
            // MPEG-2 的帧间块仍用 MPEG-1 的 RL 表（参考实现里 `mpeg2_decode_block_non_intra`
            // 就是 `ff_mpeg1_rl_vlc`）。
            intra_rl: if ext.intra_vlc_format { &MPEG2_RL } else { &MPEG1_RL },
            inter_rl: &MPEG1_RL,
            intra_lut: intra_lut_for(ext.intra_vlc_format),
            inter_lut: inter_lut(),
            scan: if ext.alternate_scan { &ALT_SCAN } else { &ZIGZAG },
            // **EOB 码不同**：MPEG-2 的 RL 表是 `0110`（4 位），MPEG-1 是 `10`（2 位）。
            intra_eob: if ext.intra_vlc_format {
                (0b0110, 4)
            } else {
                (EOB_CODE, EOB_BITS)
            },
            inter_eob: (EOB_CODE, EOB_BITS),
        },
        None => BlockTables {
            intra_rl: &MPEG1_RL,
            inter_rl: &MPEG1_RL,
            intra_lut: intra_lut_for(false),
            inter_lut: inter_lut(),
            scan: &ZIGZAG,
            intra_eob: (EOB_CODE, EOB_BITS),
            inter_eob: (EOB_CODE, EOB_BITS),
        },
    }
}

/// 在图像数据里找 picture coding extension（`00 00 01 B5`，id = 8）。
fn picture_extension(data: &[u8]) -> Option<crate::PictureCodingExtension> {
    crate::split_units(data)
        .iter()
        .find(|(code, _)| *code == 0xB5)
        .and_then(|(_, payload)| crate::PictureCodingExtension::parse(payload))
}

/// MPEG-2 扩展头校验：只支持 progressive 帧图、4:2:0、`frame_pred_frame_dct`。
fn mpeg2_extension(
    sequence: &SequenceHeader,
    data: &[u8],
) -> Result<Option<crate::PictureCodingExtension>, DecodeError> {
    if !sequence.mpeg2_stream() {
        return Ok(None);
    }
    let ext = picture_extension(data)
        .ok_or_else(|| DecodeError::new("missing_picture_extension", 0, 0))?;
    Ok(Some(ext))
}

/// 解一帧 I 图像。
/// 解一个 I 帧，写进调用方给的 `out`。
///
/// ⚠️ **不建帧、不清空**：底色由调用方负责（单线程走 `decode_i_frame` 的包装，
/// 多线程走 `MtDecoder` —— 那边只预填 worker 负责的那几行，别处是复用的旧数据）。
/// 写入只允许发生在自己那些 slice 覆盖的行上。
fn decode_i_frame_into(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    out: &mut Frame,
) -> Result<(), DecodeError> {
    if !matches!(picture.picture_type, PictureType::I) {
        return Err(DecodeError::new("picture_type", 0, 0));
    }
    let ext = mpeg2_extension(sequence, data)?;
    let tables = block_tables(ext.as_ref());
    let non_linear = ext.as_ref().map_or(false, |e| e.q_scale_type);
    let precision = ext.as_ref().map_or(0, |e| e.intra_dc_precision);
    let frame_picture = ext.as_ref().map_or(true, |e| e.picture_structure == 3);
    let read_interlaced_dct =
        frame_picture && ext.as_ref().map_or(false, |e| !e.frame_pred_frame_dct);

    let width = sequence.width as usize;
    let height = sequence.height as usize;
    let chroma_format = sequence.chroma_format;
    let (cw, _ch) = chroma_dims(chroma_format, width, height);
    let mb_blocks = mb_block_count(chroma_format);
    let mut frame = out;

    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    let mut decoded_slices = 0usize;

    for (code, slice) in crate::split_units(data) {
        if !(0x01..=0xAF).contains(&code) {
            continue;
        }
        let row = (code as usize) - 1;
        if row >= mb_h {
            continue;
        }
        decoded_slices += 1;

        let mut r = BitReader::new(slice);
        let mut scale = r
            .u(5)
            .map_err(|_| DecodeError::new("slice_scale", r.bit_pos(), code as i64))? as i32;
        if scale == 0 {
            return Err(DecodeError::new("slice_scale_zero", r.bit_pos(), code as i64));
        }
        // extra_bit_slice
        while r
            .u(1)
            .map_err(|_| DecodeError::new("extra_bit_slice", r.bit_pos(), code as i64))?
            == 1
        {
            r.u(8)
                .map_err(|_| DecodeError::new("extra_info_slice", r.bit_pos(), code as i64))?;
        }

        // 帧内 DC 预测值在 slice 起始重置，亮度与两个色度各自独立。
        let mut dc_predictor = [0i32; 3];
        // slice 不等于一个宏块行：9 个 slice 覆盖 15 行时，后面的 slice 会跨行。
        // 所以按**光栅序号**推进，而不是按行号定位。
        let mut mb_index = (row * mb_w) as isize - 1;
        let total_mb = mb_w * mb_h;
        loop {
            let _g = prof::enabled().then(|| prof::Guard::new(prof::PARSE));
            let Some(increment) = decode_mb_increment(&mut r) else {
                break;
            };
            mb_index += increment;
            if mb_index < 0 || mb_index as usize >= total_mb {
                break;
            }
            let column = mb_index as usize % mb_w;
            let mb_row = mb_index as usize / mb_w;

            // I 图像：'1' = 帧内；'01' = 帧内 + 新量化器。
            let first = r
                .u(1)
                .map_err(|_| DecodeError::new("mb_type", r.bit_pos(), column as i64))?;
            let quantised_mb = if first == 1 {
                false
            } else if r
                .u(1)
                .map_err(|_| DecodeError::new("mb_type2", r.bit_pos(), column as i64))?
                == 1
            {
                true
            } else {
                return Err(DecodeError::new("mb_type_00", r.bit_pos(), column as i64));
            };
            // MPEG-2 隔行：帧内宏块在量化器之前有一个 interlaced_dct 位。
            let interlaced_dct = if read_interlaced_dct {
                r.u(1)
                    .map_err(|_| DecodeError::new("interlaced_dct", r.bit_pos(), column as i64))?
                    == 1
            } else {
                false
            };
            if quantised_mb {
                scale = r
                    .u(5)
                    .map_err(|_| DecodeError::new("mb_scale", r.bit_pos(), column as i64))?
                    as i32;
                if scale == 0 {
                    return Err(DecodeError::new("mb_scale_zero", r.bit_pos(), column as i64));
                }
            }

            let quant = Quant::new(scale, sequence.mpeg2_stream(), non_linear, precision);
            for block in 0..mb_blocks {
                let dc_lut = if block < 4 { dc_lum_lut() } else { dc_chroma_lut() };
                let component = if block < 4 { 0 } else { (block - 4) % 2 + 1 };
                let coeffs = decode_intra_block(
                    &mut r,
                    quant,
                    &tables.intra_lut,
                    dc_lut,
                    &mut dc_predictor[component],
                    tables.intra_rl,
                    tables.scan,
                    tables.intra_eob,
                )?;
                let residual = idct_residual(&coeffs);
                store_mb_block(
                    &mut frame, block, column, mb_row, width, cw, interlaced_dct, &residual, 128,
                    chroma_format,
                );
            }
        }
    }

    if decoded_slices == 0 {
        return Err(DecodeError::new("no_slices", 0, 0));
    }
    Ok(())
}

/// 解一个 I 帧（自己建帧缓冲）。多线程 / 复用缓冲的路径走 `decode_i_frame_into`。
pub fn decode_i_frame(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
) -> Result<Frame, DecodeError> {
    let mut frame = base_frame(sequence, picture.picture_type, None);
    decode_i_frame_into(sequence, picture, data, &mut frame)?;
    Ok(frame)
}

/// 解一个 P 帧（自己建帧缓冲，底色 = 参考帧）。
pub fn decode_p_frame(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    reference: &Frame,
    other: Option<&Frame>,
) -> Result<Frame, DecodeError> {
    let mut frame = base_frame(sequence, picture.picture_type, Some(reference));
    decode_p_frame_into(sequence, picture, data, reference, other, &mut frame)?;
    Ok(frame)
}

/// 解一个 B 帧（自己建帧缓冲，底色 = 前向参考帧）。
pub fn decode_b_frame(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    forward: &Frame,
    backward: &Frame,
) -> Result<Frame, DecodeError> {
    let mut frame = base_frame(sequence, picture.picture_type, Some(forward));
    decode_b_frame_into(sequence, picture, data, forward, backward, &mut frame)?;
    Ok(frame)
}

/// 把场运动的预测写进整帧：亮度每个场两个 8×8、色度每个场 8×4，都隔行写。
#[allow(clippy::too_many_arguments)]
fn write_field_prediction(
    frame: &mut Frame,
    reference: &Frame,
    column: usize,
    mb_row: usize,
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    field_select: [u8; 2],
    mv: [[i32; 2]; 2],
    chroma_format: u8,
) {
    // 亮度：场 j 的两个 8×8 左右块。源取 `field_select[j]` 场，目标写第 j 场（偶/奇行）。
    for j in 0..2 {
        for bx in 0..2 {
            let origin_x = column * 16 + bx * 8;
            let src_y = mb_row * 16 + field_select[j] as usize;
            let dst_y = mb_row * 16 + j;
            let pred = predict_block(
                &reference.y, width, width, height, origin_x, src_y, mv[j][0], mv[j][1],
                false, 2, 2, chroma_format,
            );
            store_pixels(&pred, &mut frame.y, 2 * width, dst_y * width + origin_x);
        }
    }
    // 色度：4:2:0 每个场 8×4；4:2:2 / 4:4:4 每个场 8×8。
    let chroma_rows = if chroma_format == 1 { 4 } else { 8 };
    let chroma_mb_h = if chroma_format == 1 { 8 } else { 16 };
    let chroma_mb_w = if chroma_format == 3 { 16 } else { 8 };
    for j in 0..2 {
        let origin_x = column * chroma_mb_w;
        let src_y = mb_row * chroma_mb_h + field_select[j] as usize;
        let dst_y = mb_row * chroma_mb_h + j;
        let pred_u = predict_block(
            &reference.u, cw, cw, ch, origin_x, src_y, mv[j][0], mv[j][1], true, 2, 2,
            chroma_format,
        );
        let pred_v = predict_block(
            &reference.v, cw, cw, ch, origin_x, src_y, mv[j][0], mv[j][1], true, 2, 2,
            chroma_format,
        );
        for y in 0..chroma_rows {
            for x in 0..8 {
                let idx = dst_y * cw + origin_x + y * (2 * cw) + x;
                frame.u[idx] = pred_u[y * 8 + x];
                frame.v[idx] = pred_v[y * 8 + x];
            }
        }
    }
}

/// 写一个**场图**宏块的预测：`field_select` 选参考场，目标写半高场。
#[allow(clippy::too_many_arguments)]
fn write_field_picture_mb(
    frame: &mut Frame,
    same: &Frame,
    other: &Frame,
    column: usize,
    mb_row: usize,
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    chroma_format: u8,
    field_select: u8,
    mv: [i32; 2],
) {
    let reference = if field_select == 1 { other } else { same };
    for block in 0..mb_block_count(chroma_format) {
        let (chroma, origin_x, origin_y, stride, _) =
            block_geometry(block, column, mb_row, width, cw, false, chroma_format);
        let plane_width = if chroma { cw } else { width };
        let (rw, rh) = if chroma { (cw, ch) } else { (width, height) };
        let ref_plane = match block {
            0..=3 => &reference.y,
            4 | 6 | 8 | 10 => &reference.u,
            _ => &reference.v,
        };
        let pred = predict_block(
            ref_plane, plane_width, rw, rh, origin_x, origin_y, mv[0], mv[1], chroma, 1, 1,
            chroma_format,
        );
        let out = match block {
            0..=3 => &mut frame.y,
            4 | 6 | 8 | 10 => &mut frame.u,
            _ => &mut frame.v,
        };
        store_pixels(&pred, out, stride, origin_y * plane_width + origin_x);
    }
}

/// 写一个 dual prime 宏块的预测：同场 `mv0`、异场 `mv2`，两路平均。
#[allow(clippy::too_many_arguments)]
fn write_dmv_prediction(
    frame: &mut Frame,
    same: &Frame,
    other: &Frame,
    column: usize,
    mb_row: usize,
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    chroma_format: u8,
    mv0: [i32; 2],
    mv2: [i32; 2],
) {
    for block in 0..mb_block_count(chroma_format) {
        let (chroma, origin_x, origin_y, stride, _) =
            block_geometry(block, column, mb_row, width, cw, false, chroma_format);
        let plane_width = if chroma { cw } else { width };
        let (rw, rh) = if chroma { (cw, ch) } else { (width, height) };
        let (sp, op): (&[u8], &[u8]) = if chroma {
            let is_u = matches!(block, 4 | 6 | 8 | 10);
            (
                if is_u { &same.u } else { &same.v },
                if is_u { &other.u } else { &other.v },
            )
        } else {
            (&same.y, &other.y)
        };
        let p0 = predict_block(
            sp, plane_width, rw, rh, origin_x, origin_y, mv0[0], mv0[1], chroma, 1, 1,
            chroma_format,
        );
        let p2 = predict_block(
            op, plane_width, rw, rh, origin_x, origin_y, mv2[0], mv2[1], chroma, 1, 1,
            chroma_format,
        );
        let mut pred = [0u8; 64];
        for i in 0..64 {
            pred[i] = ((p0[i] as u16 + p2[i] as u16 + 1) >> 1) as u8;
        }
        let out = match block {
            0..=3 => &mut frame.y,
            4 | 6 | 8 | 10 => &mut frame.u,
            _ => &mut frame.v,
        };
        store_pixels(&pred, out, stride, origin_y * plane_width + origin_x);
    }
}

/// 把残差加到已写好的预测上（场运动用）。
#[allow(clippy::too_many_arguments)]
fn add_mb_block(
    frame: &mut Frame,
    block: usize,
    column: usize,
    mb_row: usize,
    width: usize,
    cw: usize,
    interlaced_dct: bool,
    residual: &[i32; 64],
    chroma_format: u8,
) {
    let (chroma, origin_x, origin_y, stride, _) =
        block_geometry(block, column, mb_row, width, cw, interlaced_dct, chroma_format);
    let plane_width = if chroma { cw } else { width };
    let plane = match block {
        0..=3 => &mut frame.y,
        4 | 6 | 8 | 10 => &mut frame.u,
        _ => &mut frame.v,
    };
    let at = origin_y * plane_width + origin_x;
    for y in 0..8 {
        for x in 0..8 {
            let idx = at + y * stride + x;
            plane[idx] = (plane[idx] as i32 + residual[y * 8 + x]).clamp(0, 255) as u8;
        }
    }
}

/// 解一帧 P 图像：以 `reference` 为底做前向运动补偿 + 残差。
/// 解一个 P 帧，写进调用方给的 `out`（底色 = 参考帧，见 `decode_i_frame_into` 的说明）。
fn decode_p_frame_into(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    reference: &Frame,
    other: Option<&Frame>,
    out: &mut Frame,
) -> Result<(), DecodeError> {
    if !matches!(picture.picture_type, PictureType::P) {
        return Err(DecodeError::new("picture_type", 0, 0));
    }
    let ext = mpeg2_extension(sequence, data)?;
    let tables = block_tables(ext.as_ref());
    let non_linear = ext.as_ref().map_or(false, |e| e.q_scale_type);
    let precision = ext.as_ref().map_or(0, |e| e.intra_dc_precision);
    let (picture_structure, fpfd) = match ext.as_ref() {
        Some(e) => (e.picture_structure, e.frame_pred_frame_dct),
        None => (3, true),
    };
    let frame_picture = picture_structure == 3;
    let read_motion_type = !(frame_picture && fpfd);
    let read_interlaced_dct = frame_picture && !fpfd;
    if reference.width != sequence.width || reference.height != sequence.height {
        return Err(DecodeError::new("reference_size", 0, 0));
    }

    let width = sequence.width as usize;
    let height = sequence.height as usize;
    let chroma_format = sequence.chroma_format;
    let (cw, ch) = chroma_dims(chroma_format, width, height);
    let mb_blocks = mb_block_count(chroma_format);
    let mut frame = out;

    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    let total_mb = mb_w * mb_h;
    let (fcode_h, fcode_v) = match ext.as_ref() {
        Some(e) => (e.f_code[0][0], e.f_code[0][1]),
        None => (picture.forward_f_code, picture.forward_f_code),
    };
    let mut decoded_slices = 0usize;

    for (code, slice) in crate::split_units(data) {
        if !(0x01..=0xAF).contains(&code) {
            continue;
        }
        let row = (code as usize) - 1;
        if row >= mb_h {
            continue;
        }
        decoded_slices += 1;

        let mut r = BitReader::new(slice);
        let mut scale = r
            .u(5)
            .map_err(|_| DecodeError::new("slice_scale", r.bit_pos(), code as i64))? as i32;
        if scale == 0 {
            return Err(DecodeError::new("slice_scale_zero", r.bit_pos(), code as i64));
        }
        while r
            .u(1)
            .map_err(|_| DecodeError::new("extra_bit_slice", r.bit_pos(), code as i64))?
            == 1
        {
            r.u(8)
                .map_err(|_| DecodeError::new("extra_info_slice", r.bit_pos(), code as i64))?;
        }

        // 帧内 DC 预测器：每个 slice 从 0 起；遇到帧间宏块重置（对应参考实现的
        // `last_dc = 128`，与我们的 0 基准等价）。
        let mut dc_predictor = [0i32; 3];
        // 运动矢量预测器：每个 slice 从 0 起；按场分开存（帧运动时两场同值）。
        let mut last_mv_f = [[0i32; 2]; 2];
        let mut mb_index = (row * mb_w) as isize - 1;
        loop {
            let _g = prof::enabled().then(|| prof::Guard::new(prof::PARSE));
            let Some(increment) = decode_mb_increment(&mut r) else {
                break;
            };
            let previous = mb_index;
            mb_index += increment;
            if mb_index < 0 || mb_index as usize >= total_mb {
                break;
            }
            // 跳过的宏块：已经是参考帧原位拷贝（帧初始化时克隆好了），
            // 只需把运动矢量预测器归零。
            //
            // ⚠️ **帧内 DC 预测器也要归零**：跳过宏块按定义是「非帧内」，
            // 而规范要求每个非帧内宏块之后 DC 预测器清零。漏了这一步，
            // 下一个帧内宏块的 DC 会带着上一次的值 → 整块恒定偏移
            // （亮度 4 块共用一路预测器，所以亮度整块偏、色度不受影响）。
            if mb_index > previous + 1 {
                last_mv_f = [[0, 0], [0, 0]];
                dc_predictor = [0i32; 3];
            }
            let column = mb_index as usize % mb_w;
            let mb_row = mb_index as usize / mb_w;

            // slice 末尾的填充位可能凑出一个无效的 mb_type —— 当作本 slice 结束。
            let Some(entry) = decode_p_mb_type(&mut r) else { break; };
            let mb = P_MB_TYPE[entry];

            // MPEG-2 隔行：帧内读 interlaced_dct；零矢量只读 interlaced_dct；
            // 非零矢量先读 motion_type 再读 interlaced_dct。
            let (motion_type, interlaced_dct) = if mb.intra {
                let dct = if read_interlaced_dct {
                    r.u(1)
                        .map_err(|_| DecodeError::new("interlaced_dct", r.bit_pos(), column as i64))?
                        == 1
                } else {
                    false
                };
                (2u32, dct)
            } else if read_motion_type {
                match mb.motion {
                    Motion::Zero => {
                        let dct = r
                            .u(1)
                            .map_err(|_| DecodeError::new("interlaced_dct", r.bit_pos(), column as i64))?
                            == 1;
                        (2u32, dct)
                    }
                    Motion::Forward => {
                        let mt = r.u(2).map_err(|_| {
                            DecodeError::new("motion_type", r.bit_pos(), column as i64)
                        })?;
                        let dct = if read_interlaced_dct && mb.coded {
                            r.u(1)
                                .map_err(|_| {
                                    DecodeError::new("interlaced_dct", r.bit_pos(), column as i64)
                                })?
                                == 1
                        } else {
                            false
                        };
                        (mt, dct)
                    }
                    Motion::None => (2, false),
                }
            } else {
                (2, false)
            };
            if !mb.intra && motion_type != 2 && motion_type != 1 && motion_type != 3 {
                return Err(DecodeError::new(
                    "motion_type_unsupported",
                    r.bit_pos(),
                    motion_type as i64,
                ));
            }

            if mb.quant {
                scale = r
                    .u(5)
                    .map_err(|_| DecodeError::new("mb_scale", r.bit_pos(), column as i64))?
                    as i32;
                if scale == 0 {
                    return Err(DecodeError::new("mb_scale_zero", r.bit_pos(), column as i64));
                }
            }
            let quant = Quant::new(scale, sequence.mpeg2_stream(), non_linear, precision);

            if mb.intra {
                last_mv_f = [[0, 0], [0, 0]];
                for block in 0..mb_blocks {
                    let dc_lut = if block < 4 { dc_lum_lut() } else { dc_chroma_lut() };
                    let component = if block < 4 { 0 } else { (block - 4) % 2 + 1 };
                    let coeffs = decode_intra_block(
                        &mut r,
                        quant,
                        &tables.intra_lut,
                        dc_lut,
                        &mut dc_predictor[component],
                        tables.intra_rl,
                        tables.scan,
                        tables.intra_eob,
                    )?;
                    let residual = idct_residual(&coeffs);
                    store_mb_block(
                        &mut frame, block, column, mb_row, width, cw, interlaced_dct, &residual,
                        128, chroma_format,
                    );
                }
                continue;
            }

            // 帧间：先运动矢量，再 cbp，再各块残差。
            let mut field_motion = false;
            let mut dmv = false;
            let mut field_select = [0u8; 2];
            let mut mv_f = [[0i32; 2]; 2];
            match mb.motion {
                Motion::Forward => {
                    if frame_picture {
                        // 帧图：mt=1 场运动（field_select + 每场一对矢量）；否则帧运动。
                        if motion_type == 1 {
                            field_motion = true;
                            for j in 0..2 {
                                field_select[j] = r
                                    .u(1)
                                    .map_err(|_| {
                                        DecodeError::new("field_select", r.bit_pos(), column as i64)
                                    })? as u8;
                                let x = decode_motion(&mut r, fcode_h, last_mv_f[j][0]).ok_or_else(
                                    || DecodeError::new("mv_x", r.bit_pos(), column as i64),
                                )?;
                                last_mv_f[j][0] = x;
                                let y = decode_motion(&mut r, fcode_v, last_mv_f[j][1] >> 1)
                                    .ok_or_else(|| {
                                        DecodeError::new("mv_y", r.bit_pos(), column as i64)
                                    })?;
                                last_mv_f[j][1] = 2 * y;
                                mv_f[j] = [x, y];
                            }
                        } else {
                            let x = decode_motion(&mut r, fcode_h, last_mv_f[0][0])
                                .ok_or_else(|| DecodeError::new("mv_x", r.bit_pos(), column as i64))?;
                            let y = decode_motion(&mut r, fcode_v, last_mv_f[0][1])
                                .ok_or_else(|| DecodeError::new("mv_y", r.bit_pos(), column as i64))?;
                            last_mv_f[0] = [x, y];
                            last_mv_f[1] = [x, y];
                            mv_f[0] = [x, y];
                            mv_f[1] = [x, y];
                        }
                    } else {
                        // 场图：mt=1 = field_select + 一对矢量；mt=2 = 一对矢量；mt=3 = dual prime。
                        if motion_type == 1 {
                            field_motion = true;
                            field_select[0] = r
                                .u(1)
                                .map_err(|_| {
                                    DecodeError::new("field_select", r.bit_pos(), column as i64)
                                })? as u8;
                            let x = decode_motion(&mut r, fcode_h, last_mv_f[0][0])
                                .ok_or_else(|| DecodeError::new("mv_x", r.bit_pos(), column as i64))?;
                            last_mv_f[0][0] = x;
                            let y = decode_motion(&mut r, fcode_v, last_mv_f[0][1])
                                .ok_or_else(|| DecodeError::new("mv_y", r.bit_pos(), column as i64))?;
                            last_mv_f[0][1] = y;
                            last_mv_f[1] = [x, y];
                            mv_f[0] = [x, y];
                            mv_f[1] = [x, y];
                        } else if motion_type == 3 {
                            let mx = decode_motion(&mut r, fcode_h, last_mv_f[0][0])
                                .ok_or_else(|| DecodeError::new("mv_x", r.bit_pos(), column as i64))?;
                            last_mv_f[0][0] = mx;
                            last_mv_f[1][0] = mx;
                            let dmx = decode_dmv(&mut r)
                                .ok_or_else(|| DecodeError::new("dmv_x", r.bit_pos(), column as i64))?;
                            let my = decode_motion(&mut r, fcode_v, last_mv_f[0][1])
                                .ok_or_else(|| DecodeError::new("mv_y", r.bit_pos(), column as i64))?;
                            last_mv_f[0][1] = my;
                            last_mv_f[1][1] = my;
                            let dmy = decode_dmv(&mut r)
                                .ok_or_else(|| DecodeError::new("dmv_y", r.bit_pos(), column as i64))?;
                            let m = if picture_structure == 1 { -1 } else { 1 };
                            mv_f[0] = [mx, my];
                            mv_f[1] = [
                                ((mx + (mx > 0) as i32) >> 1) + dmx,
                                ((my + (my > 0) as i32) >> 1) + dmy + m,
                            ];
                            dmv = true;
                        } else {
                            let x = decode_motion(&mut r, fcode_h, last_mv_f[0][0])
                                .ok_or_else(|| DecodeError::new("mv_x", r.bit_pos(), column as i64))?;
                            let y = decode_motion(&mut r, fcode_v, last_mv_f[0][1])
                                .ok_or_else(|| DecodeError::new("mv_y", r.bit_pos(), column as i64))?;
                            last_mv_f[0] = [x, y];
                            last_mv_f[1] = [x, y];
                            mv_f[0] = [x, y];
                            mv_f[1] = [x, y];
                        }
                    }
                    if picture.full_pel_forward {
                        // full_pel 的矢量是整像素，换成半像素单位再补偿。
                        for j in 0..2 {
                            mv_f[j][0] *= 2;
                            mv_f[j][1] *= 2;
                        }
                    }
                }
                Motion::Zero | Motion::None => {
                    last_mv_f = [[0, 0], [0, 0]];
                }
            }
            // 帧间宏块之后，帧内 DC 预测器归零。
            dc_predictor = [0i32; 3];

            let cbp = if mb.coded {
                let base = decode_lut(&mut r, mb_pattern_lut())
                    .ok_or_else(|| DecodeError::new("cbp", r.bit_pos(), column as i64))?
                    as u32;
                if mb_blocks > 6 {
                    let extra = r
                        .u((mb_blocks - 6) as u32)
                        .map_err(|_| DecodeError::new("cbp_ext", r.bit_pos(), column as i64))?
                        as u32;
                    (base << (mb_blocks - 6)) | extra
                } else {
                    base
                }
            } else {
                0
            };

            if dmv {
                // dual prime：同场 `reference`、异场 `other`，两路平均。
                let other_ref = other.unwrap_or(reference);
                write_dmv_prediction(
                    &mut frame, reference, other_ref, column, mb_row, width, height, cw, ch,
                    chroma_format, mv_f[0], mv_f[1],
                );
                for block in 0..mb_blocks {
                    let coded = mb.coded && (cbp & (1 << (mb_blocks - 1 - block))) != 0;
                    if coded {
                        let coeffs = decode_inter_block(
                            &mut r, quant, &tables.inter_lut, tables.inter_rl, tables.scan,
                            tables.inter_eob, !frame_picture,
                        )?;
                        let residual = idct_residual(&coeffs);
                        add_mb_block(
                            &mut frame, block, column, mb_row, width, cw, interlaced_dct,
                            &residual, chroma_format,
                        );
                    }
                }
            } else if field_motion {
                if frame_picture {
                    // 帧图：两场预测写进整帧。
                    write_field_prediction(
                        &mut frame, reference, column, mb_row, width, height, cw, ch,
                        field_select, mv_f, chroma_format,
                    );
                } else {
                    // 场图：`field_select` 选参考场，写半高场。
                    write_field_picture_mb(
                        &mut frame,
                        reference,
                        other.unwrap_or(reference),
                        column,
                        mb_row,
                        width,
                        height,
                        cw,
                        ch,
                        chroma_format,
                        field_select[0],
                        mv_f[0],
                    );
                }
                for block in 0..mb_blocks {
                    let coded = mb.coded && (cbp & (1 << (mb_blocks - 1 - block))) != 0;
                    if coded {
                        let coeffs = decode_inter_block(
                            &mut r, quant, &tables.inter_lut, tables.inter_rl, tables.scan,
                            tables.inter_eob, !frame_picture,
                        )?;
                        let residual = idct_residual(&coeffs);
                        add_mb_block(
                            &mut frame, block, column, mb_row, width, cw, interlaced_dct,
                            &residual, chroma_format,
                        );
                    }
                }
            } else {
                for block in 0..mb_blocks {
                    let coded = mb.coded && (cbp & (1 << (mb_blocks - 1 - block))) != 0;
                    let residual = if coded {
                        let coeffs = decode_inter_block(
                            &mut r, quant, &tables.inter_lut, tables.inter_rl, tables.scan,
                            tables.inter_eob, !frame_picture,
                        )?;
                        idct_residual(&coeffs)
                    } else {
                        [0i32; 64]
                    };
                    store_pred_block(
                        &mut frame, reference, block, column, mb_row, width, height, cw, ch,
                        interlaced_dct, mv_f[0][0], mv_f[0][1], &residual, chroma_format,
                    );
                }
            }
        }
    }

    if decoded_slices == 0 {
        return Err(DecodeError::new("no_slices", 0, 0));
    }
    Ok(())
}

/// 写一个 B 帧块：按前向 / 后向 / 双向取预测，再叠残差。
#[allow(clippy::too_many_arguments)]
fn write_b_block(
    frame: &mut Frame,
    forward: &Frame,
    backward: &Frame,
    block: usize,
    column: usize,
    mb_row: usize,
    fwd: bool,
    bwd: bool,
    mv_f: (i32, i32),
    mv_b: (i32, i32),
    residual: &[i32; 64],
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    interlaced_dct: bool,
    chroma_format: u8,
) {
    let (chroma, origin_x, origin_y, stride, row_step) =
        block_geometry(block, column, mb_row, width, cw, interlaced_dct, chroma_format);
    let plane_width = if chroma { cw } else { width };
    let (rw, rh) = if chroma { (cw, ch) } else { (width, height) };
    let (f_plane, b_plane): (&[u8], &[u8]) = if chroma {
        let is_u = matches!(block, 4 | 6 | 8 | 10);
        let f = if is_u { &forward.u } else { &forward.v };
        let b = if is_u { &backward.u } else { &backward.v };
        (f, b)
    } else {
        (&forward.y, &backward.y)
    };
    let pf = if fwd {
        Some(predict_block(
            f_plane, plane_width, rw, rh, origin_x, origin_y, mv_f.0, mv_f.1, chroma, row_step, 1,
            chroma_format,
        ))
    } else {
        None
    };
    let pb = if bwd {
        Some(predict_block(
            b_plane, plane_width, rw, rh, origin_x, origin_y, mv_b.0, mv_b.1, chroma, row_step, 1,
            chroma_format,
        ))
    } else {
        None
    };
    let pred = combine_pred(&pf, &pb);
    let out_plane = match block {
        0..=3 => &mut frame.y,
        4 | 6 | 8 | 10 => &mut frame.u,
        _ => &mut frame.v,
    };
    store_pred(&pred, residual, out_plane, stride, origin_y * plane_width + origin_x);
}

/// 一个只有预测、没有残差的 B 帧宏块（跳过宏块）。
#[allow(clippy::too_many_arguments)]
fn write_predicted_mb(
    frame: &mut Frame,
    forward: &Frame,
    backward: &Frame,
    column: usize,
    mb_row: usize,
    fwd: bool,
    bwd: bool,
    mv_f: (i32, i32),
    mv_b: (i32, i32),
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    interlaced_dct: bool,
    chroma_format: u8,
    mb_blocks: usize,
) {
    let zero = [0i32; 64];
    for block in 0..mb_blocks {
        write_b_block(
            frame, forward, backward, block, column, mb_row, fwd, bwd, mv_f, mv_b, &zero, width,
            height, cw, ch, interlaced_dct, chroma_format,
        );
    }
}

/// 合并前向 / 后向两路预测（双向取平均）。
fn combine_pred(pf: &Option<[u8; 64]>, pb: &Option<[u8; 64]>) -> [u8; 64] {
    let mut out = [0u8; 64];
    for i in 0..64 {
        out[i] = match (pf, pb) {
            (Some(a), Some(b)) => ((a[i] as u16 + b[i] as u16 + 1) >> 1) as u8,
            (Some(a), None) => a[i],
            (None, Some(b)) => b[i],
            (None, None) => 0,
        };
    }
    out
}

/// 写一个 B 帧场运动宏块的预测（两场各自前向 / 后向 / 双向）。
#[allow(clippy::too_many_arguments)]
fn write_b_field_prediction(
    frame: &mut Frame,
    forward: &Frame,
    backward: &Frame,
    column: usize,
    mb_row: usize,
    width: usize,
    height: usize,
    cw: usize,
    ch: usize,
    fwd: bool,
    bwd: bool,
    fs_f: [u8; 2],
    mv_f: [[i32; 2]; 2],
    fs_b: [u8; 2],
    mv_b: [[i32; 2]; 2],
    chroma_format: u8,
) {
    for j in 0..2 {
        // 亮度：场 j 的两个 8×8。
        for bx in 0..2 {
            let origin_x = column * 16 + bx * 8;
            let dst_y = mb_row * 16 + j;
            let pf = if fwd {
                let src_y = mb_row * 16 + fs_f[j] as usize;
                Some(predict_block(
                    &forward.y, width, width, height, origin_x, src_y, mv_f[j][0], mv_f[j][1],
                    false, 2, 2, chroma_format,
                ))
            } else {
                None
            };
            let pb = if bwd {
                let src_y = mb_row * 16 + fs_b[j] as usize;
                Some(predict_block(
                    &backward.y, width, width, height, origin_x, src_y, mv_b[j][0], mv_b[j][1],
                    false, 2, 2, chroma_format,
                ))
            } else {
                None
            };
            let pred = combine_pred(&pf, &pb);
            store_pixels(&pred, &mut frame.y, 2 * width, dst_y * width + origin_x);
        }
        // 色度：4:2:0 每个场 8×4；4:2:2 / 4:4:4 每个场 8×8。
        let chroma_rows = if chroma_format == 1 { 4 } else { 8 };
        let chroma_mb_h = if chroma_format == 1 { 8 } else { 16 };
        let chroma_mb_w = if chroma_format == 3 { 16 } else { 8 };
        let origin_x = column * chroma_mb_w;
        let dst_y = mb_row * chroma_mb_h + j;
        let pred = |plane_f: &[u8], plane_b: &[u8]| -> [u8; 64] {
            let pf = if fwd {
                let src_y = mb_row * chroma_mb_h + fs_f[j] as usize;
                Some(predict_block(
                    plane_f, cw, cw, ch, origin_x, src_y, mv_f[j][0], mv_f[j][1], true, 2, 2,
                    chroma_format,
                ))
            } else {
                None
            };
            let pb = if bwd {
                let src_y = mb_row * chroma_mb_h + fs_b[j] as usize;
                Some(predict_block(
                    plane_b, cw, cw, ch, origin_x, src_y, mv_b[j][0], mv_b[j][1], true, 2, 2,
                    chroma_format,
                ))
            } else {
                None
            };
            combine_pred(&pf, &pb)
        };
        let pu = pred(&forward.u, &backward.u);
        let pv = pred(&forward.v, &backward.v);
        for y in 0..chroma_rows {
            for x in 0..8 {
                let idx = dst_y * cw + origin_x + y * (2 * cw) + x;
                frame.u[idx] = pu[y * 8 + x].clamp(0, 255) as u8;
                frame.v[idx] = pv[y * 8 + x].clamp(0, 255) as u8;
            }
        }
    }
}

/// B 帧「上一个已编码宏块」的完整状态（跳过宏块沿用）。
///
/// 只存**矢量**，不存运动类型：跳过宏块的运动类型由图像结构决定，
/// 见 `decode_b_frame_into` 里跳过宏块那段注释。
#[derive(Clone, Copy)]
struct BPrev {
    kind: crate::tables::BMbType,
    fs_f: [u8; 2],
    mv_f: [[i32; 2]; 2],
    fs_b: [u8; 2],
    mv_b: [[i32; 2]; 2],
}

/// 解一帧 B 图像：前向 / 后向 / 双向运动补偿 + 残差。
///
/// 参考帧由调用方按**解码序**维护：`forward` 是前一个 I/P，`backward` 是紧随其后的
/// I/P（在码流里它比 B 帧先出现）。
/// 解一个 B 帧，写进调用方给的 `out`（底色 = 前向参考帧）。
fn decode_b_frame_into(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    forward: &Frame,
    backward: &Frame,
    out: &mut Frame,
) -> Result<(), DecodeError> {
    if !matches!(picture.picture_type, PictureType::B) {
        return Err(DecodeError::new("picture_type", 0, 0));
    }
    let ext = mpeg2_extension(sequence, data)?;
    let tables = block_tables(ext.as_ref());
    let non_linear = ext.as_ref().map_or(false, |e| e.q_scale_type);
    let precision = ext.as_ref().map_or(0, |e| e.intra_dc_precision);
    let (frame_picture, fpfd) = match ext.as_ref() {
        Some(e) => (e.picture_structure == 3, e.frame_pred_frame_dct),
        None => (true, true),
    };
    let read_motion_type = !(frame_picture && fpfd);
    let read_interlaced_dct = frame_picture && !fpfd;
    if forward.width != sequence.width
        || forward.height != sequence.height
        || backward.width != sequence.width
        || backward.height != sequence.height
    {
        return Err(DecodeError::new("reference_size", 0, 0));
    }

    let width = sequence.width as usize;
    let height = sequence.height as usize;
    let chroma_format = sequence.chroma_format;
    let (cw, ch) = chroma_dims(chroma_format, width, height);
    let mb_blocks = mb_block_count(chroma_format);
    let mut frame = out;

    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    let total_mb = mb_w * mb_h;
    let fcode_f = picture.forward_f_code;
    let fcode_b = picture.backward_f_code;
    let (fcode_fh, fcode_fv, fcode_bh, fcode_bv) = match ext.as_ref() {
        Some(e) => (e.f_code[0][0], e.f_code[0][1], e.f_code[1][0], e.f_code[1][1]),
        None => (fcode_f, fcode_f, fcode_b, fcode_b),
    };
    let mut decoded_slices = 0usize;

    for (code, slice) in crate::split_units(data) {
        if !(0x01..=0xAF).contains(&code) {
            continue;
        }
        let row = (code as usize) - 1;
        if row >= mb_h {
            continue;
        }
        decoded_slices += 1;

        let mut r = BitReader::new(slice);
        let mut scale = r
            .u(5)
            .map_err(|_| DecodeError::new("slice_scale", r.bit_pos(), code as i64))? as i32;
        if scale == 0 {
            return Err(DecodeError::new("slice_scale_zero", r.bit_pos(), code as i64));
        }
        while r
            .u(1)
            .map_err(|_| DecodeError::new("extra_bit_slice", r.bit_pos(), code as i64))?
            == 1
        {
            r.u(8)
                .map_err(|_| DecodeError::new("extra_info_slice", r.bit_pos(), code as i64))?;
        }

        let mut dc_predictor = [0i32; 3];
        // 前向 / 后向的运动矢量预测器，各自按场独立。
        let mut last_mv_f = [[0i32; 2]; 2];
        let mut last_mv_b = [[0i32; 2]; 2];
        // 上一个已编码宏块：跳过宏块沿用它（类型 + 两路矢量 + 场信息）。
        let mut prev: Option<BPrev> = None;
        let mut mb_index = (row * mb_w) as isize - 1;
        loop {
            let _g = prof::enabled().then(|| prof::Guard::new(prof::PARSE));
            let Some(increment) = decode_mb_increment(&mut r) else {
                break;
            };
            let previous = mb_index;
            mb_index += increment;
            if mb_index < 0 || mb_index as usize >= total_mb {
                break;
            }

            // 跳过的宏块：沿用上一个已编码宏块的类型与矢量，无残差。
            for skipped in (previous + 1)..mb_index {
                let Some(p) = prev else {
                    return Err(DecodeError::new("b_skip_no_prev", r.bit_pos(), skipped as i64));
                };
                if p.kind.intra {
                    return Err(DecodeError::new(
                        "b_skip_after_intra",
                        r.bit_pos(),
                        skipped as i64,
                    ));
                }
                // 跳过宏块 = 非帧内，DC 预测器要清零（同 P 帧那条，见那边的注释）。
                dc_predictor = [0i32; 3];
                let col = skipped as usize % mb_w;
                let row_s = skipped as usize / mb_w;
                // ⚠️ 跳过宏块复用上一块的**矢量与方向**，但**运动类型不照抄**：
                // 参考实现按图像结构定死 —— 帧图一律 16×16 帧运动（只用第 0 场的
                // 那对矢量），场图才是场运动。
                //
                // 照抄 `field_motion` 会让帧图里的跳过宏块按场运动取像素
                // （行步长 2、垂直矢量按场单位翻倍），实测隔行 B 帧每个画面
                // 都有一个宏块的**底场**错 90 —— 顶场对、底场错，就是这个。
                if frame_picture {
                    write_predicted_mb(
                        &mut frame, forward, backward, col, row_s, p.kind.forward,
                        p.kind.backward, (p.mv_f[0][0], p.mv_f[0][1]),
                        (p.mv_b[0][0], p.mv_b[0][1]), width, height, cw, ch, false,
                        chroma_format, mb_blocks,
                    );
                } else {
                    write_b_field_prediction(
                        &mut frame, forward, backward, col, row_s, width, height, cw, ch,
                        p.kind.forward, p.kind.backward, p.fs_f, p.mv_f, p.fs_b, p.mv_b,
                        chroma_format,
                    );
                }
            }

            let column = mb_index as usize % mb_w;
            let mb_row = mb_index as usize / mb_w;

            let Some(entry) = decode_b_mb_type(&mut r) else { break; };
            let mb = B_MB_TYPE[entry];

            // MPEG-2 隔行：与 P 帧同样的 motion_type / interlaced_dct 位置。
            let (motion_type, interlaced_dct) = if mb.intra {
                let dct = if read_interlaced_dct {
                    r.u(1)
                        .map_err(|_| DecodeError::new("interlaced_dct", r.bit_pos(), column as i64))?
                        == 1
                } else {
                    false
                };
                (2u32, dct)
            } else if read_motion_type {
                let mt = r
                    .u(2)
                    .map_err(|_| DecodeError::new("motion_type", r.bit_pos(), column as i64))?;
                let dct = if read_interlaced_dct && mb.coded {
                    r.u(1)
                        .map_err(|_| DecodeError::new("interlaced_dct", r.bit_pos(), column as i64))?
                        == 1
                } else {
                    false
                };
                (mt, dct)
            } else {
                (2, false)
            };
            if !mb.intra && motion_type != 2 && motion_type != 1 {
                return Err(DecodeError::new(
                    "motion_type_unsupported",
                    r.bit_pos(),
                    motion_type as i64,
                ));
            }

            if mb.quant {
                scale = r
                    .u(5)
                    .map_err(|_| DecodeError::new("mb_scale", r.bit_pos(), column as i64))?
                    as i32;
                if scale == 0 {
                    return Err(DecodeError::new("mb_scale_zero", r.bit_pos(), column as i64));
                }
            }
            let quant = Quant::new(scale, sequence.mpeg2_stream(), non_linear, precision);

            if mb.intra {
                last_mv_f = [[0, 0], [0, 0]];
                last_mv_b = [[0, 0], [0, 0]];
                for block in 0..mb_blocks {
                    let dc_lut = if block < 4 { dc_lum_lut() } else { dc_chroma_lut() };
                    let component = if block < 4 { 0 } else { (block - 4) % 2 + 1 };
                    let coeffs = decode_intra_block(
                        &mut r,
                        quant,
                        &tables.intra_lut,
                        dc_lut,
                        &mut dc_predictor[component],
                        tables.intra_rl,
                        tables.scan,
                        tables.intra_eob,
                    )?;
                    let residual = idct_residual(&coeffs);
                    store_mb_block(
                        &mut frame, block, column, mb_row, width, cw, interlaced_dct, &residual,
                        128, chroma_format,
                    );
                }
                prev = Some(BPrev {
                    kind: mb,
                    fs_f: [0, 0],
                    mv_f: [[0, 0], [0, 0]],
                    fs_b: [0, 0],
                    mv_b: [[0, 0], [0, 0]],
                });
                continue;
            }

            // 运动矢量：前向、后向各自独立解码，各自累加预测。
            let field_motion = motion_type == 1;
            let mut fs_f = [0u8; 2];
            let mut fs_b = [0u8; 2];
            let mut mv_f = [[0i32; 2]; 2];
            let mut mv_b = [[0i32; 2]; 2];
            if mb.forward {
                if field_motion {
                    for j in 0..2 {
                        fs_f[j] = r
                            .u(1)
                            .map_err(|_| {
                                DecodeError::new("field_select", r.bit_pos(), column as i64)
                            })? as u8;
                        let x = decode_motion(&mut r, fcode_fh, last_mv_f[j][0]).ok_or_else(
                            || DecodeError::new("mv_fx", r.bit_pos(), column as i64),
                        )?;
                        last_mv_f[j][0] = x;
                        let y = decode_motion(&mut r, fcode_fv, last_mv_f[j][1] >> 1)
                            .ok_or_else(|| {
                                DecodeError::new("mv_fy", r.bit_pos(), column as i64)
                            })?;
                        last_mv_f[j][1] = 2 * y;
                        mv_f[j] = [x, y];
                    }
                } else {
                    let x = decode_motion(&mut r, fcode_fh, last_mv_f[0][0])
                        .ok_or_else(|| DecodeError::new("mv_fx", r.bit_pos(), column as i64))?;
                    let y = decode_motion(&mut r, fcode_fv, last_mv_f[0][1])
                        .ok_or_else(|| DecodeError::new("mv_fy", r.bit_pos(), column as i64))?;
                    last_mv_f[0] = [x, y];
                    last_mv_f[1] = [x, y];
                    mv_f[0] = [x, y];
                    mv_f[1] = [x, y];
                }
                if picture.full_pel_forward {
                    for j in 0..2 {
                        mv_f[j][0] *= 2;
                        mv_f[j][1] *= 2;
                    }
                }
            }
            if mb.backward {
                if field_motion {
                    for j in 0..2 {
                        fs_b[j] = r
                            .u(1)
                            .map_err(|_| {
                                DecodeError::new("field_select", r.bit_pos(), column as i64)
                            })? as u8;
                        let x = decode_motion(&mut r, fcode_bh, last_mv_b[j][0]).ok_or_else(
                            || DecodeError::new("mv_bx", r.bit_pos(), column as i64),
                        )?;
                        last_mv_b[j][0] = x;
                        let y = decode_motion(&mut r, fcode_bv, last_mv_b[j][1] >> 1)
                            .ok_or_else(|| {
                                DecodeError::new("mv_by", r.bit_pos(), column as i64)
                            })?;
                        last_mv_b[j][1] = 2 * y;
                        mv_b[j] = [x, y];
                    }
                } else {
                    let x = decode_motion(&mut r, fcode_bh, last_mv_b[0][0])
                        .ok_or_else(|| DecodeError::new("mv_bx", r.bit_pos(), column as i64))?;
                    let y = decode_motion(&mut r, fcode_bv, last_mv_b[0][1])
                        .ok_or_else(|| DecodeError::new("mv_by", r.bit_pos(), column as i64))?;
                    last_mv_b[0] = [x, y];
                    last_mv_b[1] = [x, y];
                    mv_b[0] = [x, y];
                    mv_b[1] = [x, y];
                }
                if picture.full_pel_backward {
                    for j in 0..2 {
                        mv_b[j][0] *= 2;
                        mv_b[j][1] *= 2;
                    }
                }
            }
            dc_predictor = [0i32; 3];

            let cbp = if mb.coded {
                let base = decode_lut(&mut r, mb_pattern_lut())
                    .ok_or_else(|| DecodeError::new("cbp", r.bit_pos(), column as i64))?
                    as u32;
                if mb_blocks > 6 {
                    let extra = r
                        .u((mb_blocks - 6) as u32)
                        .map_err(|_| DecodeError::new("cbp_ext", r.bit_pos(), column as i64))?
                        as u32;
                    (base << (mb_blocks - 6)) | extra
                } else {
                    base
                }
            } else {
                0
            };

            if field_motion {
                write_b_field_prediction(
                    &mut frame, forward, backward, column, mb_row, width, height, cw, ch,
                    mb.forward, mb.backward, fs_f, mv_f, fs_b, mv_b, chroma_format,
                );
                for block in 0..mb_blocks {
                    let coded = mb.coded && (cbp & (1 << (mb_blocks - 1 - block))) != 0;
                    if coded {
                        let coeffs = decode_inter_block(
                            &mut r, quant, &tables.inter_lut, tables.inter_rl, tables.scan,
                            tables.inter_eob, !frame_picture,
                        )?;
                        let residual = idct_residual(&coeffs);
                        add_mb_block(
                            &mut frame, block, column, mb_row, width, cw, interlaced_dct,
                            &residual, chroma_format,
                        );
                    }
                }
            } else {
                for block in 0..mb_blocks {
                    let coded = mb.coded && (cbp & (1 << (mb_blocks - 1 - block))) != 0;
                    let residual = if coded {
                        let coeffs = decode_inter_block(
                            &mut r, quant, &tables.inter_lut, tables.inter_rl, tables.scan,
                            tables.inter_eob, !frame_picture,
                        )?;
                        idct_residual(&coeffs)
                    } else {
                        [0i32; 64]
                    };
                    write_b_block(
                        &mut frame, forward, backward, block, column, mb_row, mb.forward,
                        mb.backward, (mv_f[0][0], mv_f[0][1]), (mv_b[0][0], mv_b[0][1]),
                        &residual, width, height, cw, ch, interlaced_dct, chroma_format,
                    );
                }
            }
            prev = Some(BPrev {
                kind: mb,
                fs_f,
                mv_f,
                fs_b,
                mv_b,
            });
        }
    }

    if decoded_slices == 0 {
        return Err(DecodeError::new("no_slices", 0, 0));
    }
    Ok(())
}

/// 一帧的「底色」：I 帧 = 黑 + 中性色度；P / B 帧 = 参考帧的拷贝。
///
/// 只有**没有被任何 slice 覆盖的行**会留着底色；被覆盖的行由解码写入。
/// 跳过宏块（`macroblock_address_increment > 1`）的语义是「参考帧原位」，
/// 所以多线程路径下每个 worker 必须**先把自己那段行区间填成底色**再解
/// （见 `MtDecoder::decode`）。
fn base_frame(sequence: &SequenceHeader, picture_type: PictureType, forward: Option<&Frame>) -> Frame {
    let width = sequence.width as usize;
    let height = sequence.height as usize;
    let (cw, ch) = chroma_dims(sequence.chroma_format, width, height);
    let (y, u, v) = match (picture_type, forward) {
        (PictureType::I, _) | (_, None) => (
            vec![0u8; width * height],
            vec![128u8; cw * ch],
            vec![128u8; cw * ch],
        ),
        (_, Some(f)) => (f.y.clone(), f.u.clone(), f.v.clone()),
    };
    Frame {
        width: sequence.width,
        height: sequence.height,
        y,
        u,
        v,
    }
}

/// 一段**宏块行** `[rs, re)` 在三个平面上对应的字节区间。
///
/// ⚠️ 行号要先乘平面宽度才是字节偏移 —— 直接把宏块行号当下标用，会把
/// 「第 20 个宏块行」当成「第 20 个字节」，整片错位。
/// 色度的行数按色度格式缩放：4:2:0 减半，4:2:2 / 4:4:4 与亮度相同。
struct RowSpan<'a> {
    y: &'a mut [u8],
    u: &'a mut [u8],
    v: &'a mut [u8],
}

/// 把 `src` 的宏块行 `[r0, r1)` 拷进 `dst` 的同一段行（三个平面各一份）。
fn copy_rows(dst: &mut Frame, src: &Frame, r0: usize, r1: usize, chroma_format: u8) {
    let height = dst.height as usize;
    let lw = dst.y.len() / height.max(1);
    let ch = match chroma_format {
        2 | 3 => height,
        _ => height.div_ceil(2),
    };
    let cw = dst.u.len() / ch.max(1);
    let shift = if chroma_format == 1 { 1 } else { 0 };
    let (y0, y1) = (r0 * 16 * lw, r1 * 16 * lw);
    let (c0, c1) = ((r0 << (4 - shift)) * cw, (r1 << (4 - shift)) * cw);
    dst.y[y0..y1].copy_from_slice(&src.y[y0..y1]);
    dst.u[c0..c1].copy_from_slice(&src.u[c0..c1]);
    dst.v[c0..c1].copy_from_slice(&src.v[c0..c1]);
}

/// 把 `dst` 的宏块行 `[r0, r1)` 填成 I 帧的底色（亮度 0、色度 128）。
fn fill_blank_rows(dst: &mut Frame, r0: usize, r1: usize, chroma_format: u8) {
    let span = row_span(dst, r0, r1, chroma_format);
    span.y.fill(0);
    span.u.fill(128);
    span.v.fill(128);
}

fn row_span(frame: &mut Frame, rs: usize, re: usize, chroma_format: u8) -> RowSpan<'_> {
    let height = frame.height as usize;
    let (ylen, ulen) = (frame.y.len(), frame.u.len());
    let lw = ylen / height.max(1); // 亮度平面一行多少字节
    let ch = match chroma_format {
        2 | 3 => height,
        _ => height.div_ceil(2),
    };
    let cw = ulen / ch.max(1);
    // 4:2:0 的色度行数是亮度的一半；4:2:2 / 4:4:4 相同。
    let shift = if chroma_format == 1 { 1 } else { 0 };
    let (c0, c1) = ((rs << (4 - shift)) * cw, (re << (4 - shift)) * cw);
    let (y0, y1) = ((rs * 16 * lw).min(ylen), (re * 16 * lw).min(ylen));
    let (c0, c1) = (c0.min(ulen), c1.min(ulen));
    RowSpan {
        y: &mut frame.y[y0..y1],
        u: &mut frame.u[c0..c1],
        v: &mut frame.v[c0..c1],
    }
}

/// 多线程解码器：**按 slice 的行区间切给多个线程**，各自解自己那几行，最后按行合并。
///
/// ## 为什么能按 slice 切
///
/// slice 从 `slice_vertical_position` 指定的**宏块行**开始，宏块按光栅序连续排到
/// 下一个 slice —— 每片覆盖一段**互不重叠的整行**；slice 内的状态（DC / MV
/// 预测器、量化尺度）都以 slice 为单位复位，**片间无依赖**。
///
/// ## 不为线程单独写解码逻辑
///
/// 每个线程拿「图像头 + 自己那几个 slice」重组的码流，直接调
/// `decode_frame_into` 写进**自己复用的帧缓冲**，主线程再把各行拷回结果。
///
/// ## 每个 worker 要先填底色
///
/// 跳过宏块的语义是「参考帧原位」，所以 worker 必须把自己那段行区间先填成底色
/// （P/B 从参考帧拷，I 帧填 0/128）再解；缓冲区其余部分是上一帧的旧数据，**不会
/// 被合并，也不该被读**。
///
/// ## 代价
///
/// 每帧两次全帧大小的行拷贝（worker 填底色 + 主线程合并），换来解码时间按线程数
/// 下降。worker 缓冲跨帧复用，不重新分配。
pub struct MtDecoder {
    threads: usize,
    /// 每个 worker 复用的帧缓冲（跨帧不释放）。
    bufs: Vec<Frame>,
}

impl MtDecoder {
    pub fn new(threads: usize) -> Self {
        MtDecoder {
            threads: threads.max(1),
            bufs: Vec::new(),
        }
    }

    /// 解一帧。`forward` / `backward` 的语义同 `decode_frame`。
    pub fn decode(
        &mut self,
        sequence: &SequenceHeader,
        picture: PictureHeader,
        data: &[u8],
        forward: Option<&Frame>,
        backward: Option<&Frame>,
    ) -> Result<Frame, DecodeError> {
        // 收集这一帧的所有 slice（连同起始码，方便重组码流）。
        let units = crate::split_units(data);
        let mut slices: Vec<(u8, Vec<u8>)> = Vec::new();
        // ⚠️ **图像头 / 扩展要一起带上**：MPEG-2 的 picture coding extension 就在图像头
        // 后面、第一个 slice 前面，`mpeg2_extension()` 是从 `data` 里扫它的。只把 slice
        // 拼给线程，扩展就丢了 → MPEG-2 会按 MPEG-1 的表解，直接报错。
        let mut head: &[u8] = &[];
        for (code, payload) in &units {
            if (0x01..=0xAF).contains(code) {
                if head.is_empty() {
                    let off = (payload.as_ptr() as usize)
                        .wrapping_sub(data.as_ptr() as usize)
                        .saturating_sub(4) // 去掉这个 slice 自己的起始码（下面会重新拼）
                        .min(data.len());
                    head = &data[..off];
                }
                let mut raw = Vec::with_capacity(payload.len() + 4);
                raw.extend_from_slice(&[0, 0, 1, *code]);
                raw.extend_from_slice(payload);
                slices.push((*code, raw));
            }
        }
        let mb_h = (sequence.height as usize).div_ceil(16);
        // 单线程 / 只有一片 / 帧太小 —— 直接走原路径，别为线程开销买单。
        if self.threads <= 1 || slices.len() <= 1 || mb_h < 4 {
            return decode_frame(sequence, picture, data, forward, backward);
        }

        // 按**行区间**切组，而不是按「每组几片」—— 常见流只有 9 片（每片 4 行），
        // 按片数切最多用满 5 个线程；按行区间切能把 36 行均分给 8 个线程。
        // 一个 slice 覆盖到下一个 slice 的起始行为止，所以切点必须落在**片边界**上
        // （否则某一组会缺行）。做法：按行边界扫，到了下一个线程的行区间就切。
        let mut groups: Vec<(usize, usize, Vec<u8>)> = Vec::new(); // (起始行, 结束行, 码流)
        let mut cur: Vec<(u8, Vec<u8>)> = Vec::new();
        let flush = |cur: &mut Vec<(u8, Vec<u8>)>, groups: &mut Vec<(usize, usize, Vec<u8>)>| {
            if cur.is_empty() {
                return;
            }
            let first = (cur[0].0 as usize).saturating_sub(1);
            let last = (cur[cur.len() - 1].0 as usize).saturating_sub(1) + 1;
            let raw_len: usize = cur.iter().map(|(_, r)| r.len()).sum();
            let mut with_head = Vec::with_capacity(head.len() + raw_len);
            with_head.extend_from_slice(head);
            for (_, r) in cur.drain(..) {
                with_head.extend_from_slice(&r);
            }
            groups.push((first, last, with_head));
        };
        // 第 i 组覆盖的行区间是 [i*mb_h/threads, (i+1)*mb_h/threads)；遇到某个 slice
        // 的起始行已经越过这个上界、并且还没用完线程额度，就切一刀。
        let threads = self.threads.min(slices.len());
        let mut group_index = 0usize;
        let mut next_boundary = mb_h.div_ceil(threads).max(1);
        for s in slices {
            let row = (s.0 as usize).saturating_sub(1);
            if row >= next_boundary && group_index + 1 < threads {
                flush(&mut cur, &mut groups);
                group_index += 1;
                next_boundary =
                    ((group_index + 1) * mb_h).div_ceil(threads).max(next_boundary + 1);
            }
            cur.push(s);
        }
        flush(&mut cur, &mut groups);

        // ⚠️ **组的结束行 = 下一组的起始行**（最后一组到帧底）。一个 slice 覆盖到
        // 下一个 slice 的起始行为止；写成「本组最后一个 slice 的起始行 + 1」会把
        // 最后一个 slice 之后的行漏掉 —— I 帧那些行没人写，就留着初值。
        for i in 0..groups.len() {
            groups[i].1 = if i + 1 < groups.len() {
                groups[i + 1].0
            } else {
                mb_h
            };
        }

        // worker 缓冲：按需分配 / 尺寸变了就重建（跨帧复用，不每帧申请）。
        let want = groups.len();
        if self.bufs.len() < want
            || self
                .bufs
                .first()
                .is_some_and(|f| f.width != sequence.width || f.height != sequence.height)
        {
            self.bufs.clear();
            for _ in 0..want {
                self.bufs.push(Frame {
                    width: sequence.width,
                    height: sequence.height,
                    y: Vec::new(),
                    u: Vec::new(),
                    v: Vec::new(),
                });
            }
        }
        for buf in &mut self.bufs[..want] {
            if buf.y.len() != (sequence.width as usize) * (sequence.height as usize) {
                let (w, h) = (sequence.width as usize, sequence.height as usize);
                let (cw, ch) = chroma_dims(sequence.chroma_format, w, h);
                buf.y.resize(w * h, 0);
                buf.u.resize(cw * ch, 128);
                buf.v.resize(cw * ch, 128);
            }
        }

        // 结果帧：只需要「没有被任何组覆盖的行」有底色（被覆盖的行由合并写满）。
        let mut out = base_frame(sequence, picture.picture_type, forward);
        let first_start = groups.first().map(|g| g.0).unwrap_or(0);
        let last_end = groups.last().map(|g| g.1).unwrap_or(0);
        let intra = matches!(picture.picture_type, PictureType::I);
        for (r0, r1) in [(0usize, first_start), (last_end, mb_h)] {
            if r0 >= r1 {
                continue;
            }
            match (intra, forward) {
                (true, _) | (false, None) => {
                    fill_blank_rows(&mut out, r0, r1, sequence.chroma_format)
                }
                (false, Some(f)) => copy_rows(&mut out, f, r0, r1, sequence.chroma_format),
            }
        }

        let chroma_format = sequence.chroma_format;
        let mut results: Vec<(usize, usize, Result<(), DecodeError>)> = Vec::new();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for ((rs, re, raw), buf) in groups.into_iter().zip(self.bufs.iter_mut()) {
                let f = forward;
                let b = backward;
                let pic = picture.clone();
                handles.push((
                    rs,
                    re,
                    scope.spawn(move || {
                        // ① 把自己那段行区间填成底色（跳过宏块需要「参考帧原位」）。
                        match (intra, f) {
                            (true, _) | (false, None) => {
                                fill_blank_rows(buf, rs, re, chroma_format)
                            }
                            (false, Some(fwd)) => copy_rows(buf, fwd, rs, re, chroma_format),
                        }
                        // ② 解自己那几个 slice，写进同一块缓冲。
                        decode_frame_into(sequence, pic, &raw, f, b, buf)
                    }),
                ));
            }
            for (rs, re, h) in handles {
                results.push((
                    rs,
                    re,
                    h.join()
                        .unwrap_or_else(|_| Err(DecodeError::new("worker panic", 0, 0))),
                ));
            }
        });

        // ③ 按行合并回结果帧（每组一段，互不重叠）。任何一个 worker 失败就整体失败 ——
        // 半帧结果比没有结果更坏。
        for (i, (rs, re, res)) in results.into_iter().enumerate() {
            res?;
            let buf = &self.bufs[i];
            copy_rows(&mut out, buf, rs, re, chroma_format);
        }
        Ok(out)
    }
}

/// 按图像类型分派。P 帧给 `forward`；B 帧给 `forward` 与 `backward`。
pub fn decode_frame(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    forward: Option<&Frame>,
    backward: Option<&Frame>,
) -> Result<Frame, DecodeError> {
    let mut frame = base_frame(sequence, picture.picture_type, forward);
    decode_frame_into(sequence, picture, data, forward, backward, &mut frame)?;
    Ok(frame)
}

/// 同 `decode_frame`，但写进调用方给的 `out`（**不会清空它**）。
///
/// ⚠️ 调用方必须先把 `out` 填成底色（`base_frame` 干的事）：被跳过的宏块
/// 直接读缓冲区里已有的内容当预测。多线程路径只填 worker 负责的那几行。
pub fn decode_frame_into(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    forward: Option<&Frame>,
    backward: Option<&Frame>,
    out: &mut Frame,
) -> Result<(), DecodeError> {
    if let Some(ext) = mpeg2_extension(sequence, data)? {
        if ext.picture_structure != 3 {
            return Err(DecodeError::new("field_picture_in_frame_path", 0, 0));
        }
    }
    match picture.picture_type {
        PictureType::I => decode_i_frame_into(sequence, picture, data, out),
        PictureType::P => {
            let forward =
                forward.ok_or_else(|| DecodeError::new("missing_reference", 0, 0))?;
            decode_p_frame_into(sequence, picture, data, forward, None, out)
        }
        PictureType::B => {
            let forward =
                forward.ok_or_else(|| DecodeError::new("missing_reference", 0, 0))?;
            let backward =
                backward.ok_or_else(|| DecodeError::new("missing_backward_reference", 0, 0))?;
            decode_b_frame_into(sequence, picture, data, forward, backward, out)
        }
        _ => Err(DecodeError::new("unsupported_picture_type", 0, 0)),
    }
}

/// 一次性多线程解码（方便用；**每帧都新建 `MtDecoder`**，缓冲不跨帧复用）。
///
/// 连续解很多帧请直接用 [`MtDecoder`] —— 那边的 worker 缓冲是复用的。
pub fn decode_frame_mt(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    forward: Option<&Frame>,
    backward: Option<&Frame>,
    threads: usize,
) -> Result<Frame, DecodeError> {
    MtDecoder::new(threads).decode(sequence, picture, data, forward, backward)
}

/// 解一个**场**（`picture_structure = 1/2`）：半高画面。
///
/// 参考是**上一场**（调用方按同奇偶维护）。输出是半高的 `Frame`，
/// 上下两场由调用方交织成整帧。
pub fn decode_field(
    sequence: &SequenceHeader,
    picture: PictureHeader,
    data: &[u8],
    same: Option<&Frame>,
    other: Option<&Frame>,
) -> Result<Frame, DecodeError> {
    let ext = mpeg2_extension(sequence, data)?
        .ok_or_else(|| DecodeError::new("missing_picture_extension", 0, 0))?;
    if ext.picture_structure == 3 {
        return Err(DecodeError::new("not_a_field", 0, 0));
    }
    let mut field_seq = sequence.clone();
    field_seq.height /= 2;
    match picture.picture_type {
        PictureType::I => decode_i_frame(&field_seq, picture, data),
        PictureType::P => {
            let same = same.ok_or_else(|| DecodeError::new("missing_reference", 0, 0))?;
            decode_p_frame(&field_seq, picture, data, same, other)
        }
        PictureType::B => {
            let same = same.ok_or_else(|| DecodeError::new("missing_reference", 0, 0))?;
            let other =
                other.ok_or_else(|| DecodeError::new("missing_backward_reference", 0, 0))?;
            decode_b_frame(&field_seq, picture, data, same, other)
        }
        _ => Err(DecodeError::new("unsupported_picture_type", 0, 0)),
    }
}

#[cfg(test)]
mod lut_tests {
    use super::*;
    use crate::tables::{MPEG1_RL, MPEG2_RL};

    /// 参考语义：**调用方原来的检查顺序** —— EOB 先、escape 次、RL 线性扫描最后。
    /// 快路（RlLut）必须在每一个前缀上和它给出一致的结论。
    fn reference(prefix: u32, table: &[RlEntry], eob: (u16, u8)) -> (u32, u32, u32) {
        // 返回 (kind, a, b)：kind 0 = 普通符号（a = 下标），1 = EOB，2 = escape，3 = 没有。
        let top = |n: u32| prefix >> (LUT_BITS - n);
        if top(eob.1 as u32) == eob.0 as u32 {
            return (1, 0, eob.1 as u32);
        }
        if top(ESCAPE_BITS as u32) == ESCAPE_CODE as u32 {
            return (2, 0, ESCAPE_BITS as u32);
        }
        for len in 1..=LUT_BITS {
            let v = top(len);
            for (index, entry) in table.iter().enumerate() {
                if entry.bits as u32 == len && entry.code as u32 == v {
                    return (0, index as u32, len);
                }
            }
        }
        (3, 0, 0)
    }

    /// 两张表（MPEG-1 / MPEG-2 intra_vlc）× 全部 65536 个 16 位前缀：
    /// 快路查出来的 (kind, run, level, bits) 必须和参考语义逐项相同。
    #[test]
    fn rl_lut_agrees_with_reference_order() {
        for (name, table, eob) in [
            ("MPEG1_RL/inter", &MPEG1_RL, (EOB_CODE, EOB_BITS)),
            ("MPEG2_RL/intra_vlc", &MPEG2_RL, (0b0110u16, 4u8)),
        ] {
            let lut = build_rl_lut(table, eob);
            for prefix in 0u32..(1 << LUT_BITS) {
                let (kind, a, bits) = reference(prefix, table, eob);
                let slot = lut.slots[prefix as usize];
                let got_kind = slot & 3;
                let got_bits = slot >> 13;
                assert_eq!(got_kind, kind, "{name}: 前缀 {prefix:016b} 的 kind 不一致");
                assert_eq!(got_bits, bits, "{name}: 前缀 {prefix:016b} 的码长不一致");
                if kind == RL_SYM {
                    let entry = &table[a as usize];
                    assert_eq!((slot >> 2) & 31, entry.run as u32, "{name}: run 不一致");
                    assert_eq!((slot >> 7) & 63, entry.level as u32, "{name}: level 不一致");
                }
            }
        }
    }

    /// 结构码表的通用查找表：**每张表 × 全部 65536 个 16 位前缀**，
    /// 查表结论必须与线性扫描（改造前的实现）逐项一致 —— 包括「解不出码」。
    ///
    /// 这是快路的判据：结构码错了不是「画质差一点」，而是位游标整体错位，
    /// 症状离根因极远（曾经就因为这个，某个 slice 突然整片解不动）。
    #[test]
    fn vlc_lut_agrees_with_linear_scan() {
        /// 改造前的实现，原样保留为参考。
        fn scan(prefix: u32, codes: &[Code]) -> Option<(usize, u32)> {
            let max_bits = codes.iter().map(|c| c.bits).max().unwrap_or(0) as u32;
            for len in 1..=max_bits {
                let v = prefix >> (LUT_BITS - len);
                for (index, code) in codes.iter().enumerate() {
                    if code.bits as u32 == len && code.code as u32 == v {
                        return Some((index, len));
                    }
                }
            }
            None
        }

        fn as_codes(n: usize, at: impl Fn(usize) -> (u16, u8)) -> Vec<Code> {
            (0..n).map(|i| {
                let (code, bits) = at(i);
                Code { code, bits }
            }).collect()
        }

        let tables: [(&str, &[Code]); 5] = [
            ("MB_ADDR_INCR", &MB_ADDR_INCR),
            ("MV_CODE", &MV_CODE),
            ("MB_PATTERN", &MB_PATTERN),
            ("DC_LUM", &DC_LUM),
            ("DC_CHROMA", &DC_CHROMA),
        ];
        let mb_types: [(&str, Vec<Code>); 2] = [
            (
                "P_MB_TYPE",
                as_codes(P_MB_TYPE.len(), |i| (P_MB_TYPE[i].code, P_MB_TYPE[i].bits)),
            ),
            (
                "B_MB_TYPE",
                as_codes(B_MB_TYPE.len(), |i| (B_MB_TYPE[i].code, B_MB_TYPE[i].bits)),
            ),
        ];

        for (name, codes) in tables
            .iter()
            .map(|(n, c)| (*n, *c))
            .chain(mb_types.iter().map(|(n, c)| (*n, c.as_slice())))
        {
            let lut = VlcLut::from_codes(codes);
            for prefix in 0u32..(1 << LUT_BITS) {
                let want = scan(prefix, codes);
                let slot = lut.slots[prefix as usize];
                let got = if slot == VLC_INVALID {
                    None
                } else {
                    Some(((slot >> 5) as usize, slot & 31))
                };
                assert_eq!(got, want, "{name}: 前缀 {prefix:016b} 结论不一致");
            }
        }
    }

    /// 位缓存必须与 `BitReader` **逐位**一致。
    ///
    /// 判据：随机数据 × 随机起点 × 随机 peek/advance 序列 —— 每一步的
    /// `peek(n)` 与 `peek_u(n)` 相同、`bits_left()` 相同、最终位置相同。
    /// 位缓存错一位不是「画质差一点」，而是整帧错位，症状离根因极远。
    #[test]
    fn bit_cache_matches_bit_reader() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..300 {
            let len = 8 + (rnd() % 72) as usize;
            let data: Vec<u8> = (0..len).map(|_| (rnd() & 0xFF) as u8).collect();
            // 起点随机（覆盖非字节对齐）
            let start = (rnd() % (len as u64 * 8 - 32)) as usize;
            let mut r = BitReader::new(&data);
            r.skip(start as u32).unwrap();
            let mut c = BitCache::new(&data, start);
            assert_eq!(c.bits_left(), r.bits_left(), "case {case} 起点 {start}");
            for step in 0..80 {
                let n = 1 + (rnd() % 17) as u32;
                if r.bits_left() < n as usize {
                    break;
                }
                assert_eq!(
                    c.peek(n),
                    r.peek_u(n),
                    "case {case} step {step} n {n} @ {}",
                    r.bit_pos()
                );
                c.advance(n);
                r.advance(n);
                assert_eq!(c.pos(), r.bit_pos(), "case {case} step {step} 位置");
                assert_eq!(c.bits_left(), r.bits_left(), "case {case} step {step} 余量");
            }
        }
    }

    /// 融合路径（`put_inter_coeff` + `finish_mismatch`）必须与参考实现
    /// `dequantise_inter` **逐位相同**。
    ///
    /// 融合改变了三件容易错的事：矩阵下标要用 `scan[idx]`（光栅序）而不是
    /// `idx`（zigzag 序）；`mismatch` 要增量累积；`idx >= 64` 的系数既不能写
    /// 也不能计入 `mismatch`。这条测试把三种量化器组合都过一遍。
    #[test]
    fn fused_dequant_matches_array_dequant() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let cases = [
            (false, false, 1i32),
            (false, false, 8),
            (false, false, 31),
            (true, false, 1),
            (true, false, 8),
            (true, false, 31),
            (true, true, 1),
            (true, true, 16),
            (true, true, 31),
        ];
        let scan: [usize; 64] = ZIGZAG;
        for (mpeg2, non_linear, scale) in cases {
            let quant = Quant::new(scale, mpeg2, non_linear, 0);
            for _ in 0..20 {
                let mut quantised = [0i32; 64];
                for i in 0..64 {
                    if rnd() % 3 == 0 {
                        let mag = 1 + (rnd() % 40) as i32;
                        quantised[scan[i]] = if rnd() & 1 == 0 { mag } else { -mag };
                    }
                }
                let want = dequantise_inter(&quantised, quant);
                let mut got = [0i32; 64];
                let mut mismatch = 1i32;
                for i in 0..64 {
                    let level = quantised[scan[i]];
                    if level != 0 {
                        put_inter_coeff(&mut got, &mut mismatch, i, level, &scan, quant);
                    }
                }
                let got = finish_mismatch(got, mismatch, quant);
                assert_eq!(
                    got, want,
                    "inter mpeg2={mpeg2} non_linear={non_linear} scale={scale}"
                );

                // 帧内：DC 单独先做（乘 dc_scale），AC 从 index 1 起
                let want_i = dequantise_intra(&quantised, quant);
                let mut got_i = [0i32; 64];
                got_i[0] = quantised[0] * quant.dc_scale;
                let mut mismatch_i = got_i[0] ^ 1;
                for i in 1..64 {
                    let level = quantised[scan[i]];
                    if level != 0 {
                        put_intra_coeff(&mut got_i, &mut mismatch_i, i, level, &scan, quant);
                    }
                }
                let got_i = finish_mismatch(got_i, mismatch_i, quant);
                assert_eq!(
                    got_i, want_i,
                    "intra mpeg2={mpeg2} non_linear={non_linear} scale={scale}"
                );
            }
        }
    }

    /// 纯 DC 块（只有一个 DC 系数）必须出**平地**，值约等于 dc / 8
    /// （IDCT 的归一化：u=v=0 时 s = S[0][0] / 8）。
    ///
    /// 这条同时钉住 DC-only 捷径：那一行的 8 个中间值都是 `(dc*8) as i16`，
    /// 再经列通道的 `W4*(x+32) >> 20`。若捷径失效（走了通用路径），
    /// 舍入会不同 —— 表现为平地上出现 ±1 的纹。
    #[test]
    fn idct_dc_only_block_is_flat() {
        for dc in [0i32, 1, -1, 8, -8, 64, -64, 512, -512, 4095, -4095] {
            let mut block = [0i32; 64];
            block[0] = dc;
            let out = idct_residual(&block);
            let flat = out[0];
            assert!(
                out.iter().all(|&v| v == flat),
                "dc={dc} 不是平地：{:?}",
                &out[..8]
            );
            // 容差 1：`>> 20` 的取整。
            assert!(
                (flat - dc / 8).abs() <= 1,
                "dc={dc} 的平地值 {flat} 偏离 dc/8={}",
                dc / 8
            );
        }
    }

    /// 全零块必须出全零（线性 + 捷径都不能凭空造出值）。
    #[test]
    fn idct_zero_is_zero() {
        assert_eq!(idct_residual(&[0i32; 64]), [0i32; 64]);
    }
}
