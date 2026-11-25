//! Vorbis 解码的性能基准（`#[ignore]`，样本走 `V2V_VORBIS_SAMPLE`）。
//!
//! ## 为什么要有它
//!
//! 性能改动**不出现在产物字节里** —— 所以光看「测试全绿」什么也证明不了。
//! 这个基准同时给两个数：
//!
//! - **耗时**：整文件解码的中位数（多次取最快，压掉调度噪声）；
//! - **摘要**：把每包输出的 f32 位模式滚成 FNV-1a。**优化前后这个数必须一模一样**，
//!   否则不是「更快」，是「改了行为」。
//!
//! 两者缺一不可：只有耗时，可能拿错的结果换速度；只有摘要，看不出快没快。

use std::io::Read;
use std::path::Path;
use std::time::Instant;

use v2v_vorbis::Decoder;

/// FNV-1a over the raw f32 bit patterns.
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Digest(0xcbf2_9ce4_8422_2325)
    }
    fn feed(&mut self, samples: &[f32]) {
        for s in samples {
            for b in s.to_bits().to_le_bytes() {
                self.0 ^= u64::from(b);
                self.0 = self.0.wrapping_mul(0x100_0000_01b3);
            }
        }
    }
}

fn sample_path() -> Option<std::path::PathBuf> {
    match std::env::var("V2V_VORBIS_SAMPLE") {
        Ok(p) if Path::new(&p).is_file() => Some(p.into()),
        Ok(p) => panic!("V2V_VORBIS_SAMPLE 指向的文件不存在：{p}"),
        Err(_) => None,
    }
}

/// 一次完整解码：返回 (摘要, 总样本数, 包数, 声道数, 采样率)。
///
/// 只量**解码**：拆包用的是 `Cursor`（零 syscall），所以这里的时间就是
/// 「拿到包之后要花多少」。
///
/// ⚠️ `hash = false` 时**不跑全量摘要**。这不是偷懒：全量 FNV 要对每个 f32
/// 做 4 次乘法（这份样本 5 290 万个样本 → 2.1 亿次乘加），实测占基准自身耗时的
/// **37%** —— 它把每一项的占比都稀释了（`fft_in_place` 因此看起来只有 17%）。
/// 摘要只在预热轮算一次，计时轮只碰长度和每个声道的第一个样本（足够让编译器
/// 不能把解码当死代码），开销可以忽略。
///
/// ⚠️ 采样率/声道数**从识别头里读**，不许硬编码：这里原来写死 48 000，
/// 而手边那份样本是 44 100 —— 于是报出来的「551.25 s / ×898」是错的，
/// 真实是 600 s / ×977。硬编码的具体数值迟早炸，而且炸得像被测物的问题。
fn decode_once(bytes: &[u8], hash: bool) -> (u64, usize, usize, usize, u32) {
    let mut r = v2v_ogg::PacketReader::new(std::io::Cursor::new(bytes));
    let mut id: Option<Vec<u8>> = None;
    let mut dec: Option<Decoder> = None;
    let mut digest = Digest::new();
    let (mut samples, mut packets) = (0usize, 0usize);
    let (mut channels, mut rate) = (0usize, 0u32);
    let mut probe = 0u32;

    while let Some(p) = r.next_packet().expect("拆包") {
        // 三个头包按 `\xNNvorbis` 的包类型位区分；音频包的包类型位是 0。
        if p.data.starts_with(b"\x01vorbis") {
            // 识别头：`\x01vorbis` + 版本(4) + 声道(1) + 采样率(4, LE) + ...
            assert!(p.data.len() >= 16, "识别头只有 {} 字节", p.data.len());
            channels = p.data[11] as usize;
            rate = u32::from_le_bytes([p.data[12], p.data[13], p.data[14], p.data[15]]);
            id = Some(p.data);
            continue;
        }
        if p.data.starts_with(b"\x03vorbis") {
            continue;
        }
        if p.data.starts_with(b"\x05vorbis") {
            let ident = id.as_deref().expect("setup 之前没见到识别头");
            dec = Some(Decoder::new(ident, &p.data).expect("建解码器"));
            continue;
        }
        let Some(d) = dec.as_mut() else { continue };
        // 走 `decode_planes`（零分配）—— `decode` 是它的拷贝版，会给基准
        // 凭空加上「每包把 4 KiB × 声道数再复制一遍」的开销。
        if let Ok(Some(planes)) = d.decode_planes(&p.data) {
            for plane in planes {
                if hash {
                    digest.feed(plane);
                } else if let Some(&v) = plane.first() {
                    probe = probe.rotate_left(5) ^ v.to_bits();
                }
                samples += plane.len();
            }
            packets += 1;
        }
    }
    std::hint::black_box(probe);
    assert!(channels > 0 && rate > 0, "识别头没解析出声道/采样率");
    (digest.0, samples, packets, channels, rate)
}

#[test]
#[ignore = "性能基准：cargo test -p v2v-vorbis --release --test decode_bench -- --ignored --nocapture"]
fn decode_throughput() {
    let Some(path) = sample_path() else {
        eprintln!("SKIP：没设 V2V_VORBIS_SAMPLE");
        return;
    };
    let mut bytes = Vec::new();
    std::fs::File::open(&path)
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();

    // 先跑一次拿摘要（也是预热）。摘要只在**这一轮**算 —— 见 `decode_once` 的注释。
    let (digest, samples, packets, channels, rate) = decode_once(&bytes, true);

    const ROUNDS: usize = 7;
    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let t = Instant::now();
        let (d, _, _, _, _) = decode_once(&bytes, false);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        // 计时轮没有摘要可比（摘要只算一次），但 `decode_once` 里的 `black_box`
        // 保证这一轮真的跑过解码，不会被编译器当死代码删掉。
        std::hint::black_box(d);
        best = best.min(ms);
    }

    let per_channel = samples / channels;
    let seconds = per_channel as f64 / f64::from(rate);
    // 「总样本数对不对得上」是比「快不快」更靠前的一关：少解一半也能很好看。
    let declared = v2v_ogg::duration_ms(&mut std::io::Cursor::new(&bytes))
        .ok()
        .flatten()
        .map(|ms| ms as f64 / 1e3);
    if let Some(declared) = declared {
        let drift = (seconds - declared).abs();
        assert!(
            drift < 0.5,
            "解出 {seconds:.2} s，而容器声明 {declared:.2} s —— 差 {drift:.2} s，不是同一段音频"
        );
    }
    eprintln!(
        "decode: {best:.1} ms / {packets} 包 / {per_channel} 样本/声道 / {seconds:.2} s 音频（{rate} Hz × {channels}）"
    );
    eprintln!(
        "        实时倍速 ×{:.1}，每秒音频 {:.1} ms，每包 {:.1} µs",
        seconds * 1e3 / best,
        best / seconds,
        best * 1e3 / packets as f64
    );
    eprintln!("        摘要 {digest:016x}");
}

/// 单独量 IMDCT。
///
/// 这条是为了**归因**：总耗时里到底有多少是变换本身，多少是熵解码。
/// 判据不是「快不快」，是「它占多少」—— 占得少就别再往那儿使劲。
///
/// `REPS` 是 26384 —— 手边那份样本（25841 包 × 2 声道）的量级。**它只是个量级**，
/// 不是精确包数：换样本时不用改，因为这条要的是「每次多少 µs」，不是总数。
#[test]
#[ignore = "归因用：cargo test -p v2v-vorbis --release --test decode_bench imdct_cost -- --ignored --nocapture"]
fn imdct_cost() {
    use v2v_kernel::ImdctPlan;
    let n = 1024usize;
    let coefs: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32) * 1e-3).collect();
    let mut out = vec![0f32; 2 * n];
    let mut plan = ImdctPlan::new(n, 1.0);
    const REPS: usize = 26384;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..REPS {
            plan.apply(&coefs, &mut out);
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    std::hint::black_box(&out);
    eprintln!("imdct: {REPS} 次 1024 点 = {best:.1} ms（每次 {:.2} µs）", best * 1e3 / REPS as f64);
}

/// 码本的码长分布 —— 用来决定快表宽度（`codebook::FAST_BITS`）。
///
/// 判据不是「平均码长」，而是**有多少条目短于快表宽度**：只有这些能命中快路径。
#[test]
#[ignore = "诊断：cargo test -p v2v-vorbis --release --test decode_bench codeword_lengths -- --ignored --nocapture"]
fn codeword_lengths() {
    let Some(path) = sample_path() else { return };
    let bytes = std::fs::read(&path).unwrap();
    let mut r = v2v_ogg::PacketReader::new(std::io::Cursor::new(&bytes));
    let mut id: Option<Vec<u8>> = None;
    while let Some(p) = r.next_packet().expect("拆包") {
        if p.data.starts_with(b"\x01vorbis") {
            id = Some(p.data);
        } else if p.data.starts_with(b"\x05vorbis") {
            let setup = v2v_vorbis::Setup::read(&p.data, 2).expect("setup");
            let mut total = 0usize;
            let mut le10 = 0usize;
            let mut le12 = 0usize;
            for (i, cb) in setup.codebooks.iter().enumerate() {
                let lens = cb.codeword_lengths();
                let used: Vec<u8> = lens.iter().copied().filter(|&l| l > 0).collect();
                if used.is_empty() {
                    continue;
                }
                let n = used.len();
                let n10 = used.iter().filter(|&&l| l <= 10).count();
                let n12 = used.iter().filter(|&&l| l <= 12).count();
                total += n;
                le10 += n10;
                le12 += n12;
                let max = *used.iter().max().unwrap();
                if i < 12 || n10 * 10 < n * 9 {
                    eprintln!(
                        "码本 {i:3}: dim={:2} entries={:6} 用到的={n:6} max_len={max:2} ≤10={n10:6} ≤12={n12:6}",
                        cb.dimensions, cb.entry_count()
                    );
                }
            }
            eprintln!("合计：{total} 个条目，≤10 位 {le10}（{:.1}%），≤12 位 {le12}（{:.1}%）",
                100.0 * le10 as f64 / total as f64, 100.0 * le12 as f64 / total as f64);
            let _ = id;
            return;
        }
    }
}
