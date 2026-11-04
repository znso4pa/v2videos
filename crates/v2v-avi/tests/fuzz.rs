//! 变异模糊测试 —— **AVI 解封装器**。
//!
//! ## 为什么这一局特别值得做
//!
//! `v2v-avi` 是容器层**最薄**的一块：整个 crate 不到一千行，而且只有对账测试
//! （`oracle.rs`，其中一半还要真样本 + ffmpeg）。也就是说 ——
//! **它的输入校验几乎没有被「坏数据」压过**。模糊测试在这里的期望收益最高。
//!
//! ## 底本
//!
//! 手搓的最小 AVI（`tests/common/mod.rs::build_avi`），纯内存、零依赖。
//! 底本**两种都要**：带 `idx1` 与不带 —— 解封装器读索引与不读索引是两条不同的路。
//!
//! ## 跑法
//!
//! ```bash
//! cargo test -p v2v-avi --test fuzz
//! V2V_FUZZ_ROUNDS=200000 cargo test -p v2v-avi --release --test fuzz -- --ignored --nocapture
//! ```

mod common;

#[path = "../../msmpeg4/tests/common/mutate.rs"]
mod mutate;

use std::io::Cursor;

use common::build_avi;
use mutate::{Rng, Report};
use v2v_avi::{AviDemuxer, Error};

/// 底本尺寸 —— 解封装不看像素，随便取个合法值。
const W: u32 = 16;
const H: u32 = 16;

/// 造一份底本。
///
/// 帧长刻意**长短不一、且有奇数长度**：`chunk` 的奇数补齐、`idx1` 的偏移累加
/// 都靠这个才有东西可撞。第 0 帧最大（模拟关键帧）。
fn seed(with_index: bool) -> Vec<u8> {
    let frames: Vec<Vec<u8>> = vec![
        vec![0x5A; 137], // 奇数长度
        vec![0xA5; 40],
        vec![0x33; 9], // 奇数长度
        vec![0xCC; 64],
        vec![0x0F; 1], // 极短
    ];
    build_avi(&frames, W, H, 25, with_index)
}

/// 错误分类：AVI 的 `Bad` 带 `&'static str`（没有数字），
/// 所以「见过几种」直接等于「撞到几个不同的报错点」。
fn classify(e: &Error) -> String {
    match e {
        Error::Bad(m) => format!("Bad: {m}"),
        Error::Unsupported(_) => "Unsupported（本实现不支持）".to_string(),
        Error::Io(_) => "Io（底层 IO）".to_string(),
    }
}

/// 解一份字节，返回读到的样本数（同时把样本内容丢掉 —— 本局只看「崩没崩」）。
fn demux_all(data: &[u8]) -> Result<usize, Error> {
    let mut d = AviDemuxer::new(Cursor::new(data))?;
    let mut n = 0usize;
    while d.next_sample()?.is_some() {
        n += 1;
        if n > 100_000 {
            break; // 防呆，不让测试挂死
        }
    }
    Ok(n)
}

/// 底本自己必须能被**完整**解出来。
fn check_seed(data: &[u8], want: usize) {
    let n = demux_all(data).expect("底本必须解得开（底本坏了，不是被测代码的问题）");
    eprintln!("底本：{} 字节，解出 {} 个样本", data.len(), n);
    assert_eq!(n, want, "底本本该解出 {want} 个样本，实际 {n}");
}

fn fuzz_one(tag: &'static str, with_index: bool, extra: usize) -> (Report<String>, usize) {
    let base = seed(with_index);
    check_seed(&base, 5);

    let mut muts = mutate::systematic(base.len(), 4);
    let mut rng = Rng::new(if with_index {
        0xA71_0001
    } else {
        0xA71_0002
    });
    muts.extend(mutate::random(&mut rng, base.len(), extra));

    let mut samples = 0usize;
    let rep = mutate::drive(tag, &base, &muts, classify, |data| match demux_all(data) {
        Ok(n) => {
            samples += n;
            Ok(())
        }
        Err(e) => Err(e),
    });
    (rep, samples)
}

/// 轻局：CI / 日常。带索引与不带索引各跑一遍。
#[test]
fn mutated_avi_files_never_panic() {
    let extra = mutate::rounds(mutate::LIGHT_ROUNDS);
    let mut total = 0usize;
    let mut samples = 0usize;
    let mut kinds = std::collections::BTreeSet::new();
    for (name, with_index) in [("带 idx1", true), ("不带 idx1", false)] {
        let tag: &'static str = Box::leak(format!("avi 模糊/{name}").into_boxed_str());
        let (rep, n) = fuzz_one(tag, with_index, extra);
        rep.print();
        eprintln!("[{tag}]    变异体一共解出 {n} 个样本");
        total += rep.total;
        samples += n;
        kinds.extend(rep.errs.keys().cloned());
    }
    eprintln!(
        "[avi 模糊] 合计 {total} 轮，撞出 {} 种不同错误，变异体一共解出 {samples} 个样本",
        kinds.len()
    );

    assert!(
        kinds.len() >= 3,
        "两种底本加起来只撞出 {} 种错误 —— 变异没有探到解封装器的不同失败路径",
        kinds.len()
    );
    // 「不是假绿」的判据：变异体里得**真的**有能解出样本的。
    assert!(
        samples > 0,
        "所有变异体一个样本都没解出来 —— 变异太粗暴，没测到深的分支"
    );
}

/// **回归**：`LIST hdrl` 的长度字段小于 4 时不许崩。
///
/// 这是本模糊测试**第一天就撞出来的真 bug**（不是设计出来的例子）：
/// `lib.rs` 里写的是 `read_vec(reader, (sz - 4) as usize)`，而 `sz` 是从文件里
/// 读来的。把 `LIST hdrl` 的长度字段改成 0：
///
/// - **debug**：`sz - 4` 直接下溢 panic（`attempt to subtract with overflow`）
/// - **release**：溢出检查关着，`sz - 4` 绕成 2^64-4，`vec![0u8; 那个数]`
///   **分配失败直接 abort** —— 连 catch 的机会都没有，比 panic 还狠
///
/// 复现坐标（当时日志的最后一行）：
/// `[avi 模糊/带 idx1] 轮 658/2678 替换 第 16 字节 → 0x00（606 字节）`
///
/// 修法是挡住 `sz < 4` 并返回 `Err`。这条测试钉住「挡住」这件事 ——
/// 它不看具体错误消息，只看**不许 panic**（消息以后可能改，panic 不允许）。
#[test]
fn a_list_shorter_than_its_kind_field_is_an_error_not_a_panic() {
    let mut data = seed(true);
    // 定位顶层那个 `LIST hdrl` 的长度字段：`RIFF`(4) + 长度(4) + `AVI `(4) + `LIST`(4) = 16。
    assert_eq!(&data[12..16], b"LIST", "底本第 12 字节该是 LIST");
    assert_eq!(&data[20..24], b"hdrl", "底本第 20 字节该是 hdrl");
    data[16..20].copy_from_slice(&0u32.to_le_bytes()); // 长度 = 0（< 4）
    let r = demux_all(&data);
    assert!(r.is_err(), "长度字段为 0 的 LIST 应当报错，实际 {r:?}");
    eprintln!("挡住了一个真实的崩溃：{r:?}");

    // 顺带把 1..4 也扫一遍 —— 边界值是这类下溢最爱躲的地方。
    for sz in 0u32..4 {
        let mut d = seed(true);
        d[16..20].copy_from_slice(&sz.to_le_bytes());
        let r = demux_all(&d);
        assert!(r.is_err(), "长度字段 {sz} 应当报错，实际 {r:?}");
    }
}

/// 重局：本机深挖用。
///
/// ```bash
/// V2V_FUZZ_ROUNDS=200000 cargo test -p v2v-avi --release --test fuzz -- --ignored --nocapture
/// ```
#[test]
#[ignore = "重局：本机深挖用，轮数由 V2V_FUZZ_ROUNDS 给（默认 2 万）"]
fn heavy_mutation_never_panics() {
    let extra = mutate::rounds(mutate::HEAVY_ROUNDS);
    eprintln!("重局：每种底本 {extra} 轮随机变异");
    let mut kinds = std::collections::BTreeSet::new();
    for (name, with_index) in [("带 idx1", true), ("不带 idx1", false)] {
        let tag: &'static str = Box::leak(format!("avi 重局/{name}").into_boxed_str());
        let (rep, n) = fuzz_one(tag, with_index, extra);
        rep.print();
        eprintln!("[{tag}]    变异体一共解出 {n} 个样本");
        kinds.extend(rep.errs.keys().cloned());
    }
    assert!(kinds.len() >= 3, "只撞出 {} 种错误", kinds.len());
}
