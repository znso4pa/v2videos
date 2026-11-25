//! floor：频谱包络。规范第 6 章（type 0）与第 7 章（type 1）。
//!
//! **只做 type 1。** 现实里的编码器（libvorbis / ffmpeg 原生编码器）产出的都是
//! floor 1；floor 0 是 LSP 表示，曲线公式在规范里是图片，本实现没做 ——
//! 遇到 floor 0 的流在 **setup 阶段就明确报 `Unsupported`**，不假装能解。
//!
//! floor 1 的曲线计算有两条容易写错的地方：
//!
//! 1. **第 1 步（振幅预测）和第 2 步（画线）用的点集不一样**：差值为 0 的点在
//!    预测里要用（它把线钉在那儿），画线时却要**跳过**。跳过是为了让折线更平滑。
//! 2. **整数除法要向零取整**（`render_line` 的 `base`），不是向下取整 ——
//!    Rust 的 `/` 对 `i32` 就是向零取整，照抄即可，但换成移位就会错。

use crate::bit::BitReader;
use crate::codebook::Codebook;
use crate::error::{Error, Result};
use crate::ilog;

/// `{256, 128, 86, 64}` —— 由 `floor1_multiplier` 选。
const RANGES: [u32; 4] = [256, 128, 86, 64];

/// floor 1 的静态反 dB 表：`table[i] = 10^((i - 255) * 7 / 256)`。
///
/// 规范 10.1 给的是 256 个具体数值。它们不是抄来的数据，而是**一个公式的采样**：
/// 相邻两项相差 `7/256`（≈0.547 dB）的幅度比，第 255 项正好是 1.0（0 dB），
/// 第 0 项 `10^(-255*7/256) = 1.0649863e-07` 与规范印的第一个数一致（见测试）。
/// 这里按公式算，省掉一张手抄表 —— 抄错一位不会报错，只会让某段频谱偏一点。
pub fn inverse_db_table() -> [f32; 256] {
    let mut t = [0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        *v = 10f64.powf((i as f64 - 255.0) * 7.0 / 256.0) as f32;
    }
    t
}

/// floor 1 的配置。
pub struct Floor1 {
    partition_class: Vec<u8>,
    class_dimensions: Vec<u8>,
    class_subclasses: Vec<u8>,
    class_masterbooks: Vec<u8>,
    /// `subclass_books[class][j]`，`-1` 表示这个位置不编码（值为 0）。
    subclass_books: Vec<Vec<i16>>,
    multiplier: i32,
    /// 展开顺序（list order）的 X 列表。
    x_list: Vec<u16>,
    /// 按 X 升序排好的下标（曲线合成第 2 步要排序）。
    sorted: Vec<usize>,
    /// 每个下标在 list order 里的低 / 高邻居（预测要用）。
    low: Vec<usize>,
    high: Vec<usize>,
    /// 反 dB 表（规范 10.1）。**在 setup 时算一次**。
    ///
    /// 原来是每次 `curve()` 现算：256 次 `powf`，而 `curve()` 是**每包每声道**
    /// 一次 —— 一份 4 分钟的真样本要算 675 万次 `powf`。表是常量，没理由重算。
    inverse_db: [f32; 256],
}

/// 一个包的 floor 1 解码结果（曲线合成第 1 步之后）。
///
/// **每声道一个、逐包复用**：`decode_into` 会按需 `resize` 再整段重写
/// （`y` 的每个下标在 `decode_inner` 里都会被赋值，`step2` 也是），
/// 所以不需要清零。原来每包每声道分配两个 `Vec`（65 × 4 B + 65 × 1 B），
/// 真样本上是 10 万次分配 + 10 万次释放。
pub struct Floor1Packet {
    y: Vec<i32>,
    step2: Vec<bool>,
}

impl Floor1Packet {
    pub fn new() -> Self {
        Self { y: Vec::new(), step2: Vec::new() }
    }
}

impl Default for Floor1Packet {
    fn default() -> Self {
        Self::new()
    }
}

impl Floor1 {
    /// 读 floor 1 配置。规范 7.2.2。
    pub fn read(bit: &mut BitReader, codebooks: &[Codebook]) -> Result<Self> {
        let partitions = bit.read(5)? as usize;
        let mut partition_class = Vec::with_capacity(partitions);
        let mut maximum_class: i32 = -1;
        for _ in 0..partitions {
            let c = bit.read(4)? as u8;
            maximum_class = maximum_class.max(c as i32);
            partition_class.push(c);
        }

        let classes = (maximum_class + 1) as usize;
        let mut class_dimensions = vec![0u8; classes];
        let mut class_subclasses = vec![0u8; classes];
        let mut class_masterbooks = vec![0u8; classes];
        let mut subclass_books = vec![Vec::new(); classes];
        for i in 0..classes {
            class_dimensions[i] = bit.read(3)? as u8 + 1;
            let subclasses = bit.read(2)? as u8;
            class_subclasses[i] = subclasses;
            if subclasses > 0 {
                let book = bit.read(8)?;
                if book as usize >= codebooks.len() {
                    return Err(Error::Bad(format!(
                        "floor1 的 masterbook {book} 越界（共 {} 个码本）",
                        codebooks.len()
                    )));
                }
                class_masterbooks[i] = book as u8;
            }
            let n = 1usize << subclasses;
            let mut books = Vec::with_capacity(n);
            for _ in 0..n {
                let book = bit.read(8)? as i32 - 1; // 0 表示「没有」，读到的值减一
                if book >= codebooks.len() as i32 {
                    return Err(Error::Bad(format!(
                        "floor1 的 subclass book {book} 越界（共 {} 个码本）",
                        codebooks.len()
                    )));
                }
                books.push(book as i16);
            }
            subclass_books[i] = books;
        }

        let multiplier = bit.read(2)? as i32 + 1;
        let rangebits = bit.read(4)?;

        let mut x_list = vec![0u16, (1u32 << rangebits) as u16];
        for &c in &partition_class {
            for _ in 0..class_dimensions[c as usize] {
                let x = bit.read(rangebits)?;
                if x_list.len() >= 65 {
                    return Err(Error::Bad(
                        "floor1 的 X 列表超过 65 项（规范上限）".into(),
                    ));
                }
                x_list.push(x as u16);
            }
        }
        // X 必须两两不同：重复会让 low/high 邻居的定义失去意义。
        {
            let mut seen = x_list.clone();
            seen.sort_unstable();
            if seen.windows(2).any(|w| w[0] == w[1]) {
                return Err(Error::Bad("floor1 的 X 列表有重复值".into()));
            }
        }

        // 排序置换：曲线合成第 2 步按 X 升序遍历。
        let mut sorted: Vec<usize> = (0..x_list.len()).collect();
        sorted.sort_by_key(|&i| x_list[i]);

        // 低 / 高邻居：都在**list order** 里找（规范 9.2.4 / 9.2.5）。
        let values = x_list.len();
        let mut low = vec![0usize; values];
        let mut high = vec![0usize; values];
        for i in 2..values {
            let mut l = 0usize;
            let mut h = 1usize;
            for j in 0..i {
                if x_list[j] < x_list[i] && x_list[j] > x_list[l] {
                    l = j;
                }
                if x_list[j] > x_list[i] && x_list[j] < x_list[h] {
                    h = j;
                }
            }
            low[i] = l;
            high[i] = h;
        }

        Ok(Self {
            partition_class,
            class_dimensions,
            class_subclasses,
            class_masterbooks,
            subclass_books,
            multiplier,
            x_list,
            sorted,
            low,
            high,
            inverse_db: inverse_db_table(),
        })
    }

    pub fn multiplier(&self) -> i32 {
        self.multiplier
    }

    /// 解一个包的 floor，写进 `pkt`（**复用**，不分配）。
    ///
    /// 返回 `false` 表示这个声道本帧「未使用」（全零）。
    /// 规范 7.2.3：包尾读完算**正常**，按 `nonzero` 未置位处理。
    ///
    /// 复用是安全的：`decode_inner` 会把 `pkt.y` / `pkt.step2` 的**每个**下标
    /// 都赋值（`y[0]`、`y[1]` 显式写，其余在 2..values 的循环里写，最后整段
    /// `clamp` 一遍；`step2[0]`、`step2[1]` 显式置真，其余每个 `i` 都设
    /// 真或假），所以残留内容不会漏出来。
    pub fn decode_into(
        &self,
        bit: &mut BitReader,
        codebooks: &[Codebook],
        pkt: &mut Floor1Packet,
    ) -> Result<bool> {
        let values = self.x_list.len();
        if pkt.y.len() != values {
            pkt.y.resize(values, 0);
            pkt.step2.resize(values, false);
        }
        match self.decode_inner(bit, codebooks, &mut pkt.y, &mut pkt.step2) {
            Ok(()) => Ok(true),
            Err(Error::Eop) => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn decode_inner(
        &self,
        bit: &mut BitReader,
        codebooks: &[Codebook],
        y: &mut [i32],
        step2: &mut [bool],
    ) -> Result<()> {
        if !bit.read_bool()? {
            return Err(Error::Eop); // 「未使用」，与「读到包尾」走同一条出口
        }
        let range = RANGES[(self.multiplier - 1) as usize];
        let bits = ilog(range - 1);
        y[0] = bit.read(bits)? as i32;
        y[1] = bit.read(bits)? as i32;

        let mut offset = 2usize;
        for &c in &self.partition_class {
            let c = c as usize;
            let cdim = self.class_dimensions[c] as usize;
            let cbits = self.class_subclasses[c];
            let csub = (1u32 << cbits) - 1;
            let mut cval = if cbits > 0 {
                codebooks[self.class_masterbooks[c] as usize].decode_scalar(bit)? as u32
            } else {
                0
            };
            for _ in 0..cdim {
                let book = self.subclass_books[c][(cval & csub) as usize];
                cval >>= cbits;
                y[offset] = if book >= 0 {
                    codebooks[book as usize].decode_scalar(bit)? as i32
                } else {
                    0
                };
                offset += 1;
            }
        }
        debug_assert_eq!(offset, self.x_list.len());

        // 第 1 步：把「总是非负」的差值还原成 ±，沿已有的线做预测。
        step2[0] = true;
        step2[1] = true;
        for i in 2..self.x_list.len() {
            let lo = self.low[i];
            let hi = self.high[i];
            let predicted = render_point(
                self.x_list[lo] as i32,
                y[lo],
                self.x_list[hi] as i32,
                y[hi],
                self.x_list[i] as i32,
            );
            let val = y[i];
            let highroom = range as i32 - predicted;
            let lowroom = predicted;
            let room = if highroom < lowroom { highroom * 2 } else { lowroom * 2 };
            if val != 0 {
                step2[lo] = true;
                step2[hi] = true;
                step2[i] = true;
                y[i] = if val >= room {
                    if highroom > lowroom {
                        val - lowroom + predicted
                    } else {
                        predicted - val + highroom - 1
                    }
                } else if val & 1 == 1 {
                    predicted - (val + 1) / 2
                } else {
                    predicted + val / 2
                };
            } else {
                step2[i] = false;
                y[i] = predicted;
            }
        }
        // 规范建议夹到 [0, range)：合法的 setup 不会越界，坏 setup 也不该让查表崩。
        for v in y.iter_mut() {
            *v = (*v).clamp(0, range as i32 - 1);
        }
        Ok(())
    }

    /// 曲线合成（第 2 步 + 反 dB 查表），写满 `out`（长度 = 块大小 / 2）。
    ///
    /// 这条给「只要曲线」的调用方（以及测试）用；解码主路径走 `curve_mul`。
    pub fn curve(&self, pkt: &Floor1Packet, out: &mut [f32]) {
        self.render::<false>(pkt, &[], out);
    }

    /// 曲线 × residue：**一趟写出来**，`out[x] = table[y] * spectra[x]`。
    ///
    /// 原来分两趟：`curve()` 写满一遍 `out`，然后 `for (c, s) in curve.zip(spectra)`
    /// 再整条读一遍、乘、写一遍 —— 每包每声道多 4 KiB 读 + 4 KiB 写。
    /// 真样本上那是 5 万 × 8 KiB ≈ 400 MB 的额外流量。
    pub fn curve_mul(&self, pkt: &Floor1Packet, spectra: &[f32], out: &mut [f32]) {
        self.render::<true>(pkt, spectra, out);
    }

    fn render<const SCALED: bool>(&self, pkt: &Floor1Packet, spec: &[f32], out: &mut [f32]) {
        let table = &self.inverse_db;
        let n = out.len();
        let mult = self.multiplier;

        let mut lx = 0usize;
        let mut ly = pkt.y[self.sorted[0]] * mult;
        let mut hx = 0usize;
        let mut hy = ly;

        for &idx in self.sorted.iter().skip(1) {
            if !pkt.step2[idx] {
                continue;
            }
            hy = pkt.y[idx] * mult;
            hx = self.x_list[idx] as usize;
            render_line::<SCALED>(lx, ly, hx, hy, table, spec, out);
            lx = hx;
            ly = hy;
        }
        if hx < n {
            render_line::<SCALED>(hx, hy, n, hy, table, spec, out);
        }
    }
}

/// 规范 9.2.6：在线段上求 `X` 处的 `Y`（整数算法）。
fn render_point(x0: i32, y0: i32, x1: i32, y1: i32, x: i32) -> i32 {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let ady = dy.abs();
    let err = ady * (x - x0);
    let off = err / adx; // 向零取整
    if dy < 0 {
        y0 - off
    } else {
        y0 + off
    }
}

/// 规范 9.2.7：把 `[x0, x1)` 之间的整数折线写进 `out`。
///
/// 注意端点：`x0` 写、`x1` **不写**（下一次调用从 `x1` 开始写）。
/// 把「反 dB 表的整数索引」钳到 `[0, 255]`。
///
/// 与原来的 `(*v as usize).min(255)` **逐值等价**：Rust 的浮点→整数转换是
/// 饱和的，负数变 0；而这里的 `y` 是整数，`clamp` 就是同一个意思。
#[inline]
fn db_index(y: i32) -> usize {
    y.clamp(0, 255) as usize
}

/// 画一条从 `(x0,y0)` 到 `(x1,y1)` 的线（`[x0, x1)` 端点语义），**直接写反 dB 表的值**。
///
/// 原来写的是整数 `y`，再由 `curve()` 走一遍整条曲线查表 —— 那是两遍 1024 个样本。
/// 合成一遍之后少一遍全量读写（真样本上约 4%）。
///
/// 表作为参数传进来，是为了让「线的几何」还能被单独测：测试传一张恒等表，
/// 断言就还是原来那些整数。
#[inline]
fn render_line<const SCALED: bool>(
    x0: usize,
    y0: i32,
    x1: usize,
    y1: i32,
    table: &[f32; 256],
    spec: &[f32],
    out: &mut [f32],
) {
    let n = out.len();
    if x0 >= n {
        return;
    }
    let x1 = x1.min(n);
    if x1 <= x0 {
        return;
    }
    let dy = y1 - y0;
    let adx = (x1 - x0) as i32;
    let ady = dy.abs();
    let base = dy / adx;
    // 原版的 `sy = base ± 1`，也就是「走一步 = `base`，再额外 ±1」。把那个 ±1 拆出来，
    // 内层就不需要分支了 —— 见下面的 `bump`。
    let sign = if dy < 0 { -1 } else { 1 };
    let ady = ady - base.abs() * adx;

    let mut y = y0;
    let mut err = 0i32;
    // `SCALED` 是编译期常量，`if SCALED` 在单态化后会被消掉，内层不留分支。
    out[x0] = if SCALED {
        table[db_index(y)] * spec[x0]
    } else {
        table[db_index(y)]
    };
    for x in (x0 + 1)..x1 {
        err += ady;
        // `err >= adx` 的成立概率就是 `ady / adx`，对音频频谱来说是**随机**的，
        // 分支预测器在这儿一直在赔钱。换成 0/1 的 `bump`：两条路径合并成
        // 「`err -= adx·bump`、`y += base + sign·bump`」，算术结果与原来**逐位相同**
        // —— 由 `render_line_matches_the_branchy_version` 钉住。
        let bump = (err >= adx) as i32;
        err -= adx * bump;
        y += base + sign * bump;
        let v = table[db_index(y)];
        out[x] = if SCALED { v * spec[x] } else { v };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 反 dB 表的两个端点必须与规范 10.1 印出来的数一致 —— 这是「公式等价于那张表」
    /// 的判据，不是「看起来差不多」。
    #[test]
    fn inverse_db_table_matches_the_published_values() {
        let t = inverse_db_table();
        assert_eq!(t[255], 1.0);
        // 规范印的：1.0649863e-07 / 0.9389798 / 0.00033677814
        assert!((t[0] as f64 - 1.0649863e-07).abs() / 1.0649863e-07 < 1e-5, "{}", t[0]);
        assert!((t[254] as f64 - 0.9389798).abs() < 1e-6, "{}", t[254]);
        assert!((t[128] as f64 - 0.00033677814).abs() / 0.00033677814 < 1e-5, "{}", t[128]);
        // 单调递增
        assert!(t.windows(2).all(|w| w[0] < w[1]));
    }

    /// 规范 7.2.4 的例子：X = 0,16,32,...,128（list order 0,128,64,32,96,16,48,80,112），
    /// Y 依次 110,20,-5,-45,0,-25,-10,30,-10。
    ///
    /// 这里只验 `render_point`：它决定「预测值」，错了整条曲线会歪。
    #[test]
    fn render_point_uses_integer_line() {
        // (0,110) → (128,20) 在 X=64 处：dy=-90, off = 90*64/128 = 45 → 110-45 = 65
        assert_eq!(render_point(0, 110, 128, 20, 64), 65);
        // 正向：(0,0) → (100,50) 在 X=40 处 = 20
        assert_eq!(render_point(0, 0, 100, 50, 40), 20);
    }

    /// 无分支版的 oracle：原来那个 `if err >= adx { err -= adx; y += sy } else { y += base }`。
    ///
    /// 保留它是因为「把分支换成 `bump`」这种改写**看不出错**：差一位只会让某段曲线
    /// 整体平移一格，听上去像轻微的频谱倾斜，不会报任何错。所以判据必须是逐位。
    fn render_line_branchy(
        x0: usize,
        y0: i32,
        x1: usize,
        y1: i32,
        table: &[f32; 256],
        out: &mut [f32],
    ) {
        let n = out.len();
        if x0 >= n {
            return;
        }
        let x1 = x1.min(n);
        if x1 <= x0 {
            return;
        }
        let dy = y1 - y0;
        let adx = (x1 - x0) as i32;
        let ady = dy.abs();
        let base = dy / adx;
        let mut y = y0;
        let mut err = 0i32;
        let sy = if dy < 0 { base - 1 } else { base + 1 };
        let ady = ady - base.abs() * adx;
        out[x0] = table[db_index(y)];
        for x in (x0 + 1)..x1 {
            err += ady;
            if err >= adx {
                err -= adx;
                y += sy;
            } else {
                y += base;
            }
            out[x] = table[db_index(y)];
        }
    }

    /// 无分支改写必须**逐位**等价 —— 覆盖正斜率、负斜率、水平线、
    /// 以及 `ady % adx == 0`（`bump` 永远不触发）与 `ady > adx`（每步都触发）两端。
    #[test]
    fn render_line_matches_the_branchy_version() {
        let table = inverse_db_table();
        let cases: &[(usize, i32, usize, i32)] = &[
            (0, 0, 8, 0),        // 水平线：bump 永不触发
            (0, 0, 8, 8),        // 斜率恰为 1：ady = 0
            (0, 8, 8, 0),        // 负斜率、ady = 0
            (0, -40, 4, 900),    // 两端都要钳位
            (0, 300, 7, -300),   // 钳位 + 负斜率
            (3, 17, 129, 200),   // 奇数长度、ady 与 adx 互质
            (0, 255, 1000, 0),   // 长线
            (0, 0, 33, 7),       // 缓升
            (0, 200, 33, 3),     // 陡降
            (5, 100, 5, 200),    // 空区间（x1 == x0）
        ];
        for &(x0, y0, x1, y1) in cases {
            let mut want = vec![f32::NAN; 64];
            let mut got = vec![f32::NAN; 64];
            render_line_branchy(x0, y0, x1, y1, &table, &mut want);
            render_line::<false>(x0, y0, x1, y1, &table, &[], &mut got);
            for (i, (a, b)) in want.iter().zip(&got).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "({x0},{y0})→({x1},{y1}) 第 {i} 个样本不是逐位相同"
                );
            }
        }
    }

    /// 手工搭一个 floor 1：X 列表 = {0, 8, 16}，邻居关系按 list order 排好。
    /// 反 dB 表换成恒等表 —— 断言于是只验「线的几何」，不掺表值。
    fn test_floor() -> Floor1 {
        let mut ident = [0f32; 256];
        for (i, v) in ident.iter_mut().enumerate() {
            *v = i as f32;
        }
        Floor1 {
            partition_class: Vec::new(),
            class_dimensions: Vec::new(),
            class_subclasses: Vec::new(),
            class_masterbooks: Vec::new(),
            subclass_books: Vec::new(),
            multiplier: 1,
            x_list: vec![0, 8, 16],
            sorted: vec![0, 1, 2],
            low: vec![0, 0, 1],
            high: vec![1, 2, 2],
            inverse_db: ident,
        }
    }

    /// 几组覆盖不同路径的 `(y, step2)`。
    fn test_packets() -> Vec<Floor1Packet> {
        let mut out = Vec::new();
        for step2 in [
            vec![true, true, true],
            vec![true, true, false],
            // 只有 0 号点参与：整条线是一条水平线，靠末尾那段补齐。
            vec![true, false, false],
        ] {
            for y in [
                vec![10, 20, 30],
                vec![0, 0, 0],
                vec![255, 0, 255],
                vec![3, 200, 7],
            ] {
                out.push(Floor1Packet { y: y.clone(), step2: step2.clone() });
            }
        }
        out
    }

    /// `curve()` 必须**写满** `out` 的每一个元素 —— 这是 `Decoder::decode` 里
    /// 省掉 `curve.fill(0.0)` 的前提。
    ///
    /// 判据用 NaN 预填：漏写的位置会留下 NaN，而 NaN 乘任何 residue 还是 NaN，
    /// 最后在 PCM 里变成静音或爆音 —— 属于「不会报错」的那一类。
    #[test]
    fn curve_writes_every_sample() {
        let f = test_floor();
        for pkt in test_packets() {
            let mut out = vec![f32::NAN; 16];
            f.curve(&pkt, &mut out);
            assert!(
                out.iter().all(|v| !v.is_nan()),
                "y={:?} step2={:?}：curve 没写满，漏下的位置是 NaN",
                pkt.y,
                pkt.step2
            );
        }
    }

    /// `curve_mul` 必须与「先 `curve()`、再逐点乘 residue」**逐位相同**。
    ///
    /// 这是融合的唯一判据：融合写错了不会报任何错，只会让整条频谱偏一个倍数
    /// 或错一位 —— 听上去是「音量不对」或者「高频发毛」，比崩溃难查得多。
    #[test]
    fn curve_mul_matches_curve_then_multiply() {
        let f = test_floor();
        for pkt in test_packets() {
            // residue 里带上 0、负数和 1 —— 三种都能暴露「乘错位置」。
            let spec: Vec<f32> =
                (0..16).map(|i| (i as f32 * 0.37).sin() * 1.7 - 0.2).collect();
            let mut want = vec![f32::NAN; 16];
            f.curve(&pkt, &mut want);
            for (w, s) in want.iter_mut().zip(&spec) {
                *w *= s;
            }
            let mut got = vec![f32::NAN; 16];
            f.curve_mul(&pkt, &spec, &mut got);
            for (i, (a, b)) in want.iter().zip(&got).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "y={:?} 第 {i} 个样本：融合版 {b} vs 两趟版 {a}",
                    pkt.y
                );
            }
        }
    }

    /// `render_line` 写 `[x0, x1)`：端点归属错一位，整条曲线就整体平移。
    #[test]
    fn render_line_endpoints() {
        // 恒等表：让断言还是「线的几何」本身，而不是反 dB 表的数值。
        let mut ident = [0f32; 256];
        for (i, v) in ident.iter_mut().enumerate() {
            *v = i as f32;
        }
        let mut out = vec![0f32; 8];
        render_line::<false>(2, 10, 6, 10, &ident, &[], &mut out);
        assert_eq!(out, vec![0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0]);

        // 斜线：(0,0) → (4,4)：0,1,2,3 写在 0..4
        let mut out = vec![0f32; 8];
        render_line::<false>(0, 0, 4, 4, &ident, &[], &mut out);
        assert_eq!(out[..5], [0.0, 1.0, 2.0, 3.0, 0.0]);

        // 钳位：负值落到 0、超过 255 落到 255。
        //
        // (0,-40) → (4,900)：dy=940, adx=4 → base=235, ady=940-235·4=0，
        // 所以 y 依次是 -40 / 195 / 430 / 665 → 索引 0 / 195 / 255 / 255。
        // 这条钉的是「原来的 `y as usize` + `min(255)`」被换成 `clamp` 之后仍然等价：
        // 少了下界的钳位，负值会绕成一个巨大的 usize。
        let mut out = vec![0f32; 8];
        render_line::<false>(0, -40, 4, 900, &ident, &[], &mut out);
        assert_eq!(out[..4], [0.0, 195.0, 255.0, 255.0]);
    }
}
