//! 变异模糊测试 —— **MP43 解码器**。
//!
//! ## 为什么要有这个
//!
//! 解码器吃的是**别人的字节**。真样本难免有损坏帧，而一条 `panic` 会把整个转换
//! 流程带走（CLI 里那是「整批任务崩掉」）。所以判据不是「能解开合法的流」，
//! 而是「**喂什么都不能崩**」—— 允许 `Err`，禁止 `panic`。
//!
//! ## 种子从哪来
//!
//! 全部**内存生成**，零文件、零 ffmpeg、零真样本 —— 所以它在 CI 里也能跑：
//!
//! 1. `Encoder::encode_i_frame` 编一帧合法的 **I 帧**（当参考帧）
//! 2. `Encoder::encode_p_frame` 编一帧合法的 **P 帧**（当变异的底本）
//!
//! ⚠️ 两帧都刻意造成**半像素位移**的关系：P 帧就是 I 帧的水平半像素插值。
//! 这样运动估计的最优点落在半像素上，**半像素 MC 那条路才真的被走到** ——
//! 否则覆盖率断言只是「我把 I/P 帧喂进去了」这种自我安慰。
//!
//! ## 为什么每轮都新建 `Decoder` 而不是共用一个
//!
//! 用户口径是「必须用持久 decoder，否则只会走 I 帧路径」。这里满足的是那条**实质**：
//! 每轮都是「先喂合法 I 帧建立参考帧、再喂变异的 P 帧」，所以 P 帧路径每轮都真的进。
//!
//! 而**不**跨轮共用同一个 `Decoder` 是刻意的：解码器有状态（参考帧、DC/AC 网格、
//! `no_rounding` 的翻转位）。共用一个的话，第 900 轮的失败会依赖前面 899 轮留下的
//! 状态 —— 日志里那四个坐标就不再是完整的复现信息了。隔离让「一个坐标 = 一次复现」
//! 成立，这比多覆盖一点更有价值。
//!
//! ## 跑法
//!
//! ```bash
//! cargo test -p v2v-msmpeg4 --test fuzz            # 轻局（几百轮，秒级）
//! V2V_FUZZ_ROUNDS=200000 cargo test -p v2v-msmpeg4 --release --test fuzz -- --ignored --nocapture
//! ```

mod common;

use common::mutate::{self, Rng};
use v2v_msmpeg4::{Decoder, Encoder, EncoderConfig, Error, Frame};

/// 16×16 —— 一个宏块。够小，所以「逐字节、逐位」的全扫也很快。
const W: u32 = 16;
const H: u32 = 16;

/// 底本固定的种子 —— 换种子会换掉全部变异体，失败就没法复现了。
const SEED: u64 = 0x5EED_1234_ABCD_0001;

/// 造一帧**高对比、有高频**的 YUV420P。
///
/// ⚠️ 底本的质量直接决定这个模糊测试是不是**假绿**。踩过的坑：
///
/// - **纯色**：编出来只有 DC，宏块全是 skip，变异撞不到几条分支。
/// - **平缓渐变**：编出来的 I 帧很小；更糟的是它的半像素插值几乎等于原图，
///   运动补偿之后残差被量化直接抹平 → P 帧退化成**一个 skip 宏块**（实测编出来
///   **只有 3 字节**），P 帧里除了帧头什么都没走到。
///
/// 所以这里用「2 像素周期的棋盘（强高频）+ 斜向斜坡（保证半像素插值与原图**明显**
/// 不同）」，亮度拉满 40..200 的对比度。
fn textured() -> Frame {
    let (w, h) = (W as usize, H as usize);
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            let checker = if ((col / 2) + (row / 2)) % 2 == 0 { 150.0 } else { 40.0 };
            let ramp = (col as f32 * 3.0 + row as f32 * 1.5) % 60.0;
            y[row * w + col] = (checker + ramp).clamp(0.0, 255.0) as u8;
        }
    }
    let mut u = vec![0u8; (w / 2) * (h / 2)];
    let mut v = vec![0u8; (w / 2) * (h / 2)];
    for row in 0..h / 2 {
        for col in 0..w / 2 {
            u[row * (w / 2) + col] = (60.0 + col as f32 * 12.0).clamp(0.0, 255.0) as u8;
            v[row * (w / 2) + col] = (200.0 - row as f32 * 12.0).clamp(0.0, 255.0) as u8;
        }
    }
    Frame {
        width: W,
        height: H,
        y,
        u,
        v,
    }
}

/// **水平半像素插值** —— 把 `src` 变成它自己右移半个像素的样子。
///
/// `dst[x] = (src[x] + src[x+1]) / 2`。这正是解码端半像素 MC 用的插值，
/// 所以拿它当 P 帧，运动估计的最优点就在半像素上。
fn half_pel_shift(src: &Frame) -> Frame {
    let (w, h) = (W as usize, H as usize);
    let lerp = |plane: &[u8], pw: usize, ph: usize| -> Vec<u8> {
        let mut out = vec![0u8; pw * ph];
        for row in 0..ph {
            for col in 0..pw {
                let a = plane[row * pw + col] as u32;
                let b = plane[row * pw + (col + 1).min(pw - 1)] as u32;
                out[row * pw + col] = ((a + b) / 2) as u8;
            }
        }
        out
    };
    Frame {
        width: W,
        height: H,
        y: lerp(&src.y, w, h),
        u: lerp(&src.u, w / 2, h / 2),
        v: lerp(&src.v, w / 2, h / 2),
    }
}

/// 编一对底本：合法 I 帧 + 合法 P 帧（P 是 I 的半像素位移 **+ 一处直流扰动**）。
///
/// 加那处扰动的理由见 `perturb` —— 不加的话 P 帧会退化成单个 skip 宏块，
/// 模糊测试就变成「只测了帧头」的假绿。
fn seeds() -> (Vec<u8>, Vec<u8>) {
    let a = textured();
    let b = perturb(&half_pel_shift(&a));
    let cfg = EncoderConfig::default();
    let mut enc = Encoder::new(cfg);
    let i = enc.encode_i_frame(&a).expect("编 I 帧（底本，不该失败）");
    let p = enc.encode_p_frame(&b).expect("编 P 帧（底本，不该失败）");
    (i, p)
}

/// 给左上角 8×8 加一个直流偏移。
///
/// ⚠️ **非加不可**：只做半像素插值的话运动补偿几乎完美，残差被量化抹平，
/// 宏块退化成 skip（实测编出来只有 **3 字节**）—— 那样 P 帧里除了帧头
/// 一条分支都没走到，模糊测试是假绿的。加一个 32 的直流偏移，残差活过量化，
/// 宏块变成「帧间带残差」，那才是 P 帧解码的主干路径。
///
/// 偏移量刻意选小（32）：大到会影响运动估计的选点，就不是「半像素位移」了。
fn perturb(src: &Frame) -> Frame {
    let w = W as usize;
    let mut y = src.y.clone();
    for row in 0..8 {
        for col in 0..8 {
            y[row * w + col] = (y[row * w + col] as i32 + 32).clamp(0, 255) as u8;
        }
    }
    Frame {
        width: W,
        height: H,
        y,
        u: src.u.clone(),
        v: src.v.clone(),
    }
}

/// 底本**自己**走了哪条路 —— 并把关键的几条钉成断言。
///
/// 这是这个文件里最容易被跳过、却最要紧的一步：**底本退化的话，整个模糊测试是假的**。
/// 踩过一次 —— 平缓渐变的底本编出来的 P 帧只有 3 字节（整帧一个 skip 宏块），
/// 于是「几百轮全绿」其实只测了帧头。所以底本自己必须先被解一遍、通路先被验一遍。
fn check_seed_richness(iframe: &[u8], pframe: &[u8]) {
    let mut dec = Decoder::new();
    dec.decode(iframe, W, H).expect("底本 I 帧必须解得开");
    dec.decode(pframe, W, H).expect("底本 P 帧必须解得开");
    let st = dec.stats();
    eprintln!(
        "底本通路：I 帧 {} / P 帧 {}\n           帧内宏块 {}（P 帧里的 {}）\n           帧间：带残差 {} / 只有 MC {} / skip {}\n           MV 相位：零 {} / 水平半 {} / 垂直半 {} / 双向半 {}",
        st.frames_i,
        st.frames_p,
        st.mb_intra,
        st.mb_intra_in_p,
        st.mb_inter_coded,
        st.mb_inter_empty,
        st.mb_skip,
        st.mv_zero,
        st.mv_half_x,
        st.mv_half_y,
        st.mv_half_xy,
    );
    assert!(st.frames_i > 0 && st.frames_p > 0, "底本要 I、P 各一帧");
    assert!(st.mb_intra > 0, "底本 I 帧的宏块没走到帧内路径");
    assert!(
        st.mb_inter_coded + st.mb_inter_empty > 0,
        "底本 P 帧退化成了 skip 宏块（编出来 {} 字节）—— 变异只会撞到帧头，这个模糊测试是假的",
        pframe.len()
    );
    assert!(
        st.mv_half_x + st.mv_half_y + st.mv_half_xy > 0,
        "底本 P 帧没走到半像素 MC —— 种子没满足设计意图（P 帧本该是 I 帧的半像素位移）"
    );
}

/// 错误分类：用**具体的消息**，不是变体名。
///
/// `Bad` 带的是 `&'static str`（消息里没有数字），所以「见过几种」直接等于
/// 「撞到几个不同的报错点」—— 是真实覆盖，不是被数字撑虚的计数。
/// `Unsupported` 的消息可能带参数，只取变体名。
fn classify(e: &Error) -> String {
    match e {
        Error::Bad(m) => format!("Bad: {m}"),
        Error::Eof => "Eof（位流提前结束）".to_string(),
        Error::Unsupported(_) => "Unsupported（本实现不支持）".to_string(),
    }
}

/// 把 `Stats` 累加进一个总量（`Stats` 没有 `Add`，手写一遍字段）。
fn acc(total: &mut v2v_msmpeg4::Stats, s: &v2v_msmpeg4::Stats) {
    total.frames_i += s.frames_i;
    total.frames_p += s.frames_p;
    total.mb_intra += s.mb_intra;
    total.mb_intra_in_p += s.mb_intra_in_p;
    total.mb_inter_coded += s.mb_inter_coded;
    total.mb_inter_empty += s.mb_inter_empty;
    total.mb_skip += s.mb_skip;
    total.blocks_ac_pred += s.blocks_ac_pred;
    total.mv_zero += s.mv_zero;
    total.mv_half_x += s.mv_half_x;
    total.mv_half_y += s.mv_half_y;
    total.mv_half_xy += s.mv_half_xy;
    total.mc_edge_emu += s.mc_edge_emu;
    total.frames_mv_table_1 += s.frames_mv_table_1;
    for i in 0..3 {
        total.frames_rl_inter[i] += s.frames_rl_inter[i];
        total.frames_rl_intra[i] += s.frames_rl_intra[i];
    }
}

/// 跑一轮模糊测试，返回覆盖报告 + 解码器统计。
///
/// `extra_rounds` 是**随机**变异（区间拼接等）的轮数；系统性那一层
/// （每个截断偏移、每一位翻转、每个字节的三种替换）永远全跑。
fn fuzz(tag: &'static str, extra_rounds: usize) -> (mutate::Report<String>, v2v_msmpeg4::Stats) {
    let (iframe, pframe) = seeds();
    eprintln!("底本：I 帧 {} 字节，P 帧 {} 字节", iframe.len(), pframe.len());
    check_seed_richness(&iframe, &pframe);

    // 系统性那一层：截断的**每一个**偏移 + 逐字节逐位的翻转/替换。
    // 16×16 的 P 帧只有几百字节，全扫很便宜 —— 而「位流在哪个位置结束」
    // 是解码器最容易写错的一维，没有理由抽样。
    let mut muts = mutate::systematic(pframe.len(), 1);
    // 再撒一把随机的（主要是区间拼接：让字段之间自相矛盾）。
    let mut rng = Rng::new(SEED);
    muts.extend(mutate::random(&mut rng, pframe.len(), extra_rounds));

    let mut total = v2v_msmpeg4::Stats::default();
    let rep = mutate::drive(tag, &pframe, &muts, classify, |data| {
        let mut dec = Decoder::new();
        // 先喂合法 I 帧建立参考帧 —— 没有这一步，P 帧路径永远进不去
        // （`decode` 会直接返回「P 帧需要参考帧」）。
        dec.decode(&iframe, W, H)
            .expect("合法 I 帧必须解得开（底本坏了，不是被测代码的问题）");
        // `decode` 成功时给的是 `Frame`；这里只关心「崩没崩」，把值丢掉。
        let r = dec.decode(data, W, H).map(|_| ());
        acc(&mut total, &dec.stats());
        r
    });
    (rep, total)
}

/// 轻局：CI / 日常。系统性全扫 + 几百轮随机。
#[test]
fn mutated_frames_never_panic() {
    let (rep, st) = fuzz("mp43 模糊", mutate::rounds(mutate::LIGHT_ROUNDS));
    rep.print();

    // ---- 覆盖率断言 ----
    // 「测试没红」不等于「查过了」。一个每轮都在第 0 字节就 `Err` 的模糊测试
    // 也会绿，而且绿得毫无价值。所以这里把「撞到了多少条不同的错误通路」钉住。
    assert!(
        rep.kinds() >= 3,
        "只撞出 {} 种错误 —— 变异没有真的探到解码器的不同失败路径，这个模糊测试是假的",
        rep.kinds()
    );
    assert!(
        rep.total > 0 && rep.shortest < 8,
        "截断没有截到很短（最短变异体 {} 字节）—— 系统性那一层没跑起来",
        rep.shortest
    );

    // ---- 解码器通路覆盖 ----
    eprintln!(
        "通路覆盖：\n  \
         I 帧 {} / P 帧 {}\n  \
         帧内宏块 {}（其中出现在 P 帧里的 {}）\n  \
         帧间：有残差 {} / 只有 MC {} / skip {}\n  \
         MV 相位：零 {} / 水平半 {} / 垂直半 {} / 双向半 {}\n  \
         边缘复制(MC) {}",
        st.frames_i,
        st.frames_p,
        st.mb_intra,
        st.mb_intra_in_p,
        st.mb_inter_coded,
        st.mb_inter_empty,
        st.mb_skip,
        st.mv_zero,
        st.mv_half_x,
        st.mv_half_y,
        st.mv_half_xy,
        st.mc_edge_emu,
    );

    assert!(st.frames_p > 0, "P 帧路径一轮都没走到 —— 底本或喂法有问题");
    assert!(
        st.mb_inter_coded + st.mb_inter_empty + st.mb_skip > 0,
        "帧间宏块一条都没走到 —— P 帧只解析了帧头就失败了，等于没测 P 帧"
    );

    // 下面这些**只报告不失败**：撞不到是变异分布的性质，不是代码的错。
    for (name, hit) in [
        (
            "半像素 MC（底本刻意做成半像素位移，本该撞到）",
            st.mv_half_x + st.mv_half_y + st.mv_half_xy > 0,
        ),
        ("MC 的边缘复制", st.mc_edge_emu > 0),
        ("P 帧里的帧内宏块", st.mb_intra_in_p > 0),
        ("AC 预测（ac_pred）", st.blocks_ac_pred > 0),
    ] {
        if !hit {
            eprintln!("⚠️ 这条通路本局没走到：{name}");
        }
    }
}

/// 重局：本机深挖用。轮数由 `V2V_FUZZ_ROUNDS` 给。
///
/// ```bash
/// V2V_FUZZ_ROUNDS=200000 cargo test -p v2v-msmpeg4 --release --test fuzz -- --ignored --nocapture
/// ```
#[test]
#[ignore = "重局：本机深挖用，轮数由 V2V_FUZZ_ROUNDS 给（默认 2 万）"]
fn heavy_mutation_never_panics() {
    let n = mutate::rounds(mutate::HEAVY_ROUNDS);
    eprintln!("重局：{n} 轮随机变异");
    let (rep, _st) = fuzz("mp43 重局", n);
    rep.print();
    assert!(rep.kinds() >= 3, "只撞出 {} 种错误", rep.kinds());
}

/// **I 帧**也要单独模糊一遍 —— 它走的是另一条路（DC 预测 + 帧内 RL 表）。
///
/// 这一局不需要参考帧，所以每轮就是「新建解码器 → 喂变异体」。
#[test]
fn mutated_i_frames_never_panic() {
    let (iframe, _) = seeds();
    let mut muts = mutate::systematic(iframe.len(), 1);
    let mut rng = Rng::new(SEED ^ 0x1F1F);
    muts.extend(mutate::random(&mut rng, iframe.len(), mutate::rounds(mutate::LIGHT_ROUNDS)));

    let mut total = v2v_msmpeg4::Stats::default();
    let rep = mutate::drive("mp43 I 帧模糊", &iframe, &muts, classify, |data| {
        let mut dec = Decoder::new();
        let r = dec.decode(data, W, H).map(|_| ());
        acc(&mut total, &dec.stats());
        r
    });
    rep.print();

    assert!(rep.kinds() >= 3, "I 帧只撞出 {} 种错误", rep.kinds());
    assert!(total.frames_i > 0, "I 帧路径一轮都没走到");
    assert!(
        total.mb_intra > 0,
        "帧内宏块一条都没走到 —— I 帧只解析了帧头就失败了"
    );
}
