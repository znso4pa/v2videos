//! 变异模糊测试 —— **WMA 解码器**。
//!
//! ## 和 MP43 那局的关键区别：**没有合法底本**
//!
//! 仓里没有 WMA 编码器（只有解码器），所以造不出「一帧合法的 WMA」当底本 ——
//! 这一局只能喂**合成的字节**。那就要当心一件事：
//!
//! ⚠️ **纯随机字节的模糊测试会假绿**。随机位流几乎必然在头几个 VLC 就
//! `invalid vlc code`，于是「几千轮全绿」其实只测了最外面那一层。
//!
//! 所以这里用**一小组结构性的底本**（全 0 / 全 1 / 周期图案 / 固定伪随机），
//! 每个都系统性变异一遍。全 0 与周期图案**能一路解到系数层**（解出来是静音，
//! 但结构合法），所以变异撞到的是深处的分支，而不是「第一字节就失败」。
//!
//! 覆盖断言因此分两层：
//! - 硬断言：至少撞出 N 种错误 + 有若干轮**完全没报错**（说明真的解进去了）
//! - 报告：哪些底本能走通、走到了哪
//!
//! ## 跑法
//!
//! ```bash
//! cargo test -p v2v-wma --test fuzz
//! V2V_FUZZ_ROUNDS=200000 cargo test -p v2v-wma --release --test fuzz -- --ignored --nocapture
//! ```

#[path = "../../msmpeg4/tests/common/mutate.rs"]
mod mutate;

use mutate::{Rng, Report};
use v2v_wma::{Error, WmaDecoder, WmaParams};

/// 底本长度 = `block_align` —— `decode_superframe` 按 `block_align` 切块。
///
/// 取 256（真实 WMA v2 常见 2048 或 4096）：够装下一帧的指数 + 若干系数块，
/// 而系统性全扫（每个截断偏移 + 每位翻转 + 三种替换）也只要 ~1500 轮。
const BLOCK_ALIGN: usize = 256;

/// 造参数 —— 纯内存，不需要真样本。
///
/// `extradata` 给空：`WmaParams::new` 对短 extradata 是**容忍**的
/// （读不到就当 0，于是 `use_variable_block_len = false`、`nb_block_sizes = 1`），
/// 走的正是「定长块」这条更简单的路。
fn params() -> WmaParams {
    WmaParams::new(
        2,           // version：v2
        2,           // channels
        44100,       // sample_rate
        128_000,     // bit_rate
        BLOCK_ALIGN, // block_align
        &[],         // extradata
    )
    .expect("内存造参数不该失败")
}

/// 底本语料：**结构性**的几种字节图案。
///
/// 每种都在撞不同的东西：全 0 撞「全零位流的 VLC 解码」、全 1 撞「所有位都是 1」、
/// 周期图案撞「字段之间错位」、固定伪随机撞「无结构的位流」。
fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    let mut rng = Rng::new(0xC0FF_EE00_1234_5678);
    let pseudo: Vec<u8> = (0..BLOCK_ALIGN).map(|_| rng.byte()).collect();
    vec![
        ("全 0", vec![0x00; BLOCK_ALIGN]),
        ("全 1", vec![0xFF; BLOCK_ALIGN]),
        ("周期 0x55", (0..BLOCK_ALIGN).map(|i| if i % 2 == 0 { 0x55 } else { 0xAA }).collect()),
        ("固定伪随机", pseudo),
    ]
}

/// 错误分类：`Bad` 带的是 `&'static str`（没有数字），所以「见过几种」
/// 直接等于「撞到几个不同的报错点」。`Unsupported` / `Io` 只取变体名。
fn classify(e: &Error) -> String {
    match e {
        Error::Bad(m) => format!("Bad: {m}"),
        Error::Unsupported(_) => "Unsupported（本实现不支持）".to_string(),
        Error::Io(_) => "Io（底层 IO）".to_string(),
    }
}

/// 一局的**真实覆盖** —— 比「没报错」强得多的判据。
///
/// ⚠️ 为什么非要有这个：`decode_superframe` 对**攒位**的包会返回 `Ok(vec![])`，
/// 对垃圾载荷也可能整包不产帧。于是「几千轮没报错」可能其实是
/// 「几千轮什么都没解」—— 那正是这个文件最想避免的假绿。
/// 所以这里数**真的产出了多少帧**。
#[derive(Default)]
struct Out {
    /// 产出了至少一帧的轮数。
    rounds_with_frames: usize,
    /// 累计产出的音频帧数。
    frames: usize,
    /// 累计产出的采样点数（帧 × frame_len × 声道）。
    samples: usize,
}

impl Out {
    fn add(&mut self, got: &[Vec<Vec<f32>>]) {
        if !got.is_empty() {
            self.rounds_with_frames += 1;
        }
        for frame in got {
            self.frames += 1;
            for ch in frame {
                self.samples += ch.len();
            }
        }
    }
    fn merge(&mut self, o: &Out) {
        self.rounds_with_frames += o.rounds_with_frames;
        self.frames += o.frames;
        self.samples += o.samples;
    }
}

/// 对**一个**底本跑完整一轮（系统性 + 随机），返回报告与真实覆盖。
fn fuzz_one(tag: &'static str, base: &[u8], extra: usize) -> (Report<String>, Out) {
    let mut muts = mutate::systematic(base.len(), 1);
    let mut rng = Rng::new(0xF0F0_0F0F_1234_5678);
    muts.extend(mutate::random(&mut rng, base.len(), extra));

    let p = params();
    let mut out = Out::default();
    let rep = mutate::drive(tag, base, &muts, classify, |data| {
        // 每轮新建解码器：`decode_superframe` 是有状态的（噪声编码的随机种子、
        // 位池位置都跨帧延续），共用会让「一个坐标 = 一次复现」不成立。
        let mut dec = WmaDecoder::new(p.clone()).expect("建解码器");
        match dec.decode_superframe(data) {
            Ok(got) => {
                out.add(&got);
                Ok(())
            }
            Err(e) => Err(e),
        }
    });
    (rep, out)
}

/// 轻局：CI / 日常。四个底本各跑一遍系统性变异 + 几百轮随机。
#[test]
fn mutated_payloads_never_panic() {
    let extra = mutate::rounds(mutate::LIGHT_ROUNDS);
    let mut total = 0usize;
    let mut ok = 0usize;
    let mut kinds = std::collections::BTreeSet::new();
    let mut all = Out::default();
    for (name, base) in corpus() {
        let tag: &'static str = Box::leak(format!("wma 模糊/{name}").into_boxed_str());
        let (rep, out) = fuzz_one(tag, &base, extra);
        rep.print();
        eprintln!(
            "[{tag}]    → 真的解出帧的轮数 {}（共 {} 帧 / {} 个采样点）",
            out.rounds_with_frames, out.frames, out.samples
        );
        total += rep.total;
        ok += rep.ok;
        all.merge(&out);
        kinds.extend(rep.errs.keys().cloned());
    }
    eprintln!(
        "[wma 模糊] 合计 {total} 轮，没报错 {ok} 轮，撞出 {} 种不同错误；其中 {} 轮真的解出了帧（共 {} 帧）",
        kinds.len(),
        all.rounds_with_frames,
        all.frames
    );

    // ---- 覆盖率断言 ----
    assert!(
        kinds.len() >= 3,
        "四个底本加起来只撞出 {} 种错误 —— 变异没有探到解码器的不同失败路径",
        kinds.len()
    );
    // ⚠️ 这条是**这一局唯一能证明「不是假绿」的断言**：如果每一轮都在
    // 第一个 VLC 就失败，`ok` 会是 0，那说明底本根本没解进去。
    // ⚠️ 这条才是这一局**唯一能证明「不是假绿」的断言**。
    // `ok > 0` 不够：`decode_superframe` 对攒位的包返回 `Ok(vec![])`，
    // 「没报错」完全可能等于「什么都没解」。所以要求**真的产出了帧**。
    assert!(
        all.frames > 0 && all.rounds_with_frames > 0,
        "{} 轮里一轮都没真的解出帧 —— 底本没有结构性，这个模糊测试是假的",
        total
    );
}

/// 重局：本机深挖用。
///
/// ```bash
/// V2V_FUZZ_ROUNDS=200000 cargo test -p v2v-wma --release --test fuzz -- --ignored --nocapture
/// ```
#[test]
#[ignore = "重局：本机深挖用，轮数由 V2V_FUZZ_ROUNDS 给（默认 2 万）"]
fn heavy_mutation_never_panics() {
    let extra = mutate::rounds(mutate::HEAVY_ROUNDS);
    eprintln!("重局：每个底本 {extra} 轮随机变异");
    let mut kinds = std::collections::BTreeSet::new();
    let mut all = Out::default();
    for (name, base) in corpus() {
        let tag: &'static str = Box::leak(format!("wma 重局/{name}").into_boxed_str());
        let (rep, out) = fuzz_one(tag, &base, extra);
        rep.print();
        all.merge(&out);
        kinds.extend(rep.errs.keys().cloned());
    }
    assert!(kinds.len() >= 3, "只撞出 {} 种错误", kinds.len());
    assert!(all.frames > 0, "没有一轮真的解出帧");
}

/// 直接喂 `decode_frame`（比 `decode_superframe` 更深一层的入口）。
///
/// `decode_superframe` 会先做超帧头的解析与位池定位；这一局跳过那些，
/// 把变异体**直接**塞进「解一帧」那一步 —— 两条路的失败模式不一样，
/// 都要覆盖。
#[test]
fn mutated_bitstreams_never_panic_in_decode_frame() {
    let p = params();
    let mut kinds = std::collections::BTreeSet::new();
    let mut total = 0usize;
    let mut ok = 0usize;
    for (name, base) in corpus() {
        let tag: &'static str = Box::leak(format!("wma decode_frame/{name}").into_boxed_str());
        let mut muts = mutate::systematic(base.len(), 4); // stride 4：这一局只做抽样
        let mut rng = Rng::new(0x1234_ABCD_5678_EF01);
        muts.extend(mutate::random(&mut rng, base.len(), mutate::rounds(200)));
        let rep = mutate::drive(tag, &base, &muts, classify, |data| {
            let mut dec = WmaDecoder::new(p.clone()).expect("建解码器");
            let mut r = v2v_wma::BitReader::new(data);
            dec.decode_frame(&mut r).map(|_| ())
        });
        rep.print();
        total += rep.total;
        ok += rep.ok;
        kinds.extend(rep.errs.keys().cloned());
    }
    eprintln!("[wma decode_frame] 合计 {total} 轮，没报错 {ok} 轮，{} 种错误", kinds.len());
    assert!(kinds.len() >= 2, "只撞出 {} 种错误", kinds.len());
}
