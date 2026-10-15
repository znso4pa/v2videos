//! 变异模糊测试 —— **ASF 解封装器**。
//!
//! ## 底本从哪来：自己封装一份（纯内存）
//!
//! ASF 是全仓**唯一**能「封装器造底本、解封装器吃底本」的容器 ——
//! `AsfMuxer::finish()` 直接吐 `Vec<u8>`，不需要文件、不需要 ffmpeg、不需要真样本。
//! 所以这一局的底本是**结构完全合法**的一份 ASF，变异撞到的是解封装器里
//! 真正深的分支（对象表、数据包、Simple Index），而不是「第一字节就不是 ASF」。
//!
//! ## 覆盖断言
//!
//! 硬断言（三条都要成立，否则这个模糊测试是假的）：
//! 1. 至少撞出 3 种不同的错误；
//! 2. **有若干轮真的解出了样本**（变异没破坏结构）；
//! 3. 底本自己必须能被完整解出来 —— 底本坏了就什么都没测。
//!
//! ## 跑法
//!
//! ```bash
//! cargo test -p v2v-asf --test fuzz
//! V2V_FUZZ_ROUNDS=200000 cargo test -p v2v-asf --release --test fuzz -- --ignored --nocapture
//! ```

#[path = "../../msmpeg4/tests/common/mutate.rs"]
mod mutate;

use std::io::Cursor;

use mutate::{Rng, Report};
use v2v_asf::{AsfDemuxer, AsfMuxer, Error, VideoStreamInfo};

/// 底本用 16×16 的 `MP43` —— 载荷内容不影响解封装，所以随便填。
const W: u32 = 16;
const H: u32 = 16;

/// 造一份**合法**的 ASF（纯内存）。
///
/// 刻意做得「不平凡」：多帧、含关键帧与非关键帧、时间戳递增 ——
/// 这样 Simple Index、多载荷打包、时间戳字段都在底本里，变异才有东西可撞。
fn seed() -> Vec<u8> {
    let mut m = AsfMuxer::new(VideoStreamInfo {
        width: W,
        height: H,
        fourcc: u32::from_le_bytes(*b"MP43"),
    });
    // 关键帧给大一点（真实 I 帧就比 P 帧大），非关键帧小一点。
    let big = vec![0x5Au8; 400];
    let small = vec![0xA5u8; 60];
    for i in 0..12u32 {
        let key = i % 4 == 0;
        let data = if key { &big } else { &small };
        m.write_video(data, key, i * 40).expect("写视频载荷");
    }
    m.finish()
}

/// 错误分类：`Bad` 带的是 `&'static str`（没有数字），所以「见过几种」
/// 直接等于「撞到几个不同的报错点」。
fn classify(e: &Error) -> String {
    match e {
        Error::Bad(m) => format!("Bad: {m}"),
        Error::Eof => "Eof（数据提前结束）".to_string(),
        Error::Unsupported(_) => "Unsupported（本实现不支持）".to_string(),
        Error::Io(_) => "Io（底层 IO）".to_string(),
    }
}

/// 解一份字节，返回读到的样本数。
///
/// `next_sample` 可能返回 `Err`，那就直接抛出去 —— 本局只关心「崩没崩」。
fn demux_all(data: &[u8]) -> Result<usize, Error> {
    let mut d = AsfDemuxer::new(Cursor::new(data))?;
    let mut n = 0usize;
    while d.next_sample()?.is_some() {
        n += 1;
        // 防呆：底本只有十几帧，读出一万个样本说明解封装器失控了
        // （真的失控会先在别处炸，这里只是不让测试挂死）。
        if n > 10_000 {
            break;
        }
    }
    Ok(n)
}

/// 底本自己必须能被**完整**解出来 —— 否则后面几百轮全是噪声。
fn check_seed(data: &[u8]) {
    let n = demux_all(data).expect("底本必须解得开（底本坏了，不是被测代码的问题）");
    eprintln!("底本：{} 字节，解出 {} 个样本", data.len(), n);
    assert_eq!(n, 12, "底本本该解出 12 个样本，实际 {n} —— 底本或解封装器坏了");
}

fn fuzz(tag: &'static str, extra: usize) -> (Report<String>, usize) {
    let base = seed();
    check_seed(&base);

    let mut muts = mutate::systematic(base.len(), 4);
    let mut rng = Rng::new(0xA5F0_1234_5678_9ABC);
    muts.extend(mutate::random(&mut rng, base.len(), extra));

    let mut samples = 0usize;
    let rep = mutate::drive(tag, &base, &muts, classify, |data| {
        match demux_all(data) {
            Ok(n) => {
                samples += n;
                Ok(())
            }
            Err(e) => Err(e),
        }
    });
    (rep, samples)
}

/// 轻局：CI / 日常。
#[test]
fn mutated_asf_files_never_panic() {
    let (rep, samples) = fuzz("asf 模糊", mutate::rounds(mutate::LIGHT_ROUNDS));
    rep.print();
    eprintln!("[asf 模糊] 变异体一共解出 {samples} 个样本");

    assert!(
        rep.kinds() >= 3,
        "只撞出 {} 种错误 —— 变异没有探到解封装器的不同失败路径",
        rep.kinds()
    );
    // ⚠️ 这条是「不是假绿」的判据：变异体里得**真的**有能解出样本的，
    // 否则「几千轮全报错」只说明变异把文件全破坏了。
    assert!(
        samples > 0,
        "所有变异体一个样本都没解出来 —— 变异太粗暴，这个模糊测试没测到深的分支"
    );
}

/// 重局：本机深挖用。
///
/// ```bash
/// V2V_FUZZ_ROUNDS=200000 cargo test -p v2v-asf --release --test fuzz -- --ignored --nocapture
/// ```
#[test]
#[ignore = "重局：本机深挖用，轮数由 V2V_FUZZ_ROUNDS 给（默认 2 万）"]
fn heavy_mutation_never_panics() {
    let n = mutate::rounds(mutate::HEAVY_ROUNDS);
    eprintln!("重局：{n} 轮随机变异");
    let (rep, samples) = fuzz("asf 重局", n);
    rep.print();
    eprintln!("[asf 重局] 变异体一共解出 {samples} 个样本");
    assert!(rep.kinds() >= 3, "只撞出 {} 种错误", rep.kinds());
}
