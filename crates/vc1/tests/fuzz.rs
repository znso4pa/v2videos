//! 变异模糊测试：**允许 `Err`，禁止 `panic`**。
//!
//! 零样本、零 `ffmpeg`、纯内存 —— 所以它在 CI 里也真跑，不是「SKIP 了」。
//! 帧头吃的同样是**别人的字节**（容器里那一段），一条 panic 会把整条转换流程带走。
//!
//! ⚠️ 两件按仓规必须先做的事（都在别的局上踩过）：
//! - **底本质量断言**：底本自己先解一遍、还要断言它**走到了深处**
//!   （混合 MV 的位平面 / 非 RAW 的跳过平面）。否则「几千轮全绿」可能只测了帧头前几位。
//! - **「没报错」不等于「解进去了」**：所以这里数的是「解出 `Ok` 的轮数」，
//!   不是「没 panic 的轮数」。
//!
//! 轮数由 `V2V_FUZZ_ROUNDS` 控制（不设就用这里的默认）。

use v2v_vc1::{FrameHeader, HeaderState, Imode, PictType, SeqHeader};

/// 一轮里的变异次数上限。
const MAX_MUTATIONS: usize = 3;

/// 便宜、确定、无依赖的 PRNG（xorshift64）。
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// 攒位用的写入器（测试里要把帧头**写出来**，好有个像样的底本）。
struct BitSink {
    bits: Vec<u8>,
}

impl BitSink {
    fn new() -> Self {
        Self { bits: Vec::new() }
    }

    fn put(&mut self, v: u32, n: u32) {
        for i in (0..n).rev() {
            self.bits.push(((v >> i) & 1) as u8);
        }
    }

    fn bytes(self) -> Vec<u8> {
        let mut bits = self.bits;
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        bits.chunks(8)
            .map(|c| c.iter().fold(0u8, |a, b| (a << 1) | b))
            .collect()
    }
}

/// 一份 Main profile 的序列头（`extended_mv` / `vstransform` 都开，无 B 帧、无隔行）。
fn seq_bytes() -> Vec<u8> {
    let mut s = BitSink::new();
    s.put(1, 2); // profile = Main
    s.put(0, 1); // res_y411
    s.put(0, 1); // res_sprite
    s.put(0, 3); // frmrtq_postproc
    s.put(0, 5); // bitrtq_postproc
    s.put(1, 1); // loop_filter
    s.put(0, 1); // res_x8
    s.put(0, 1); // multires
    s.put(1, 1); // res_fasttx
    s.put(0, 1); // fastuvmc
    s.put(1, 1); // extended_mv
    s.put(0, 2); // dquant
    s.put(1, 1); // vstransform
    s.put(0, 1); // res_transtab
    s.put(0, 1); // overlap
    s.put(0, 1); // resync_marker
    s.put(0, 1); // rangered
    s.put(0, 3); // max_b_frames
    s.put(0, 2); // quantizer_mode = 隐式
    s.put(0, 1); // finterpflag
    s.put(1, 1); // res_rtm_flag
    s.bytes()
}

/// 一份**能解出来**的 P 帧帧头底本：帧级混合 MV（要解「4MV 吗」的位平面）
/// + 列跳（非 RAW）的跳过位平面。
fn frame_bytes() -> Vec<u8> {
    let mut s = BitSink::new();
    s.put(0, 2); // framecnt
    s.put(1, 1); // pict_type = P
    s.put(8, 5); // pqindex -> pq = 8
    s.put(0, 1); // halfpq
    s.put(0, 1); // mvrange（unary stop=0）
    s.put(0b01, 2); // mv_mode（unary stop=1）= 1 -> MIXED_MV
    // 「4MV 吗」位平面（2×2 网格）：invert=0, imode=NORM2(10), 两码 11 11
    s.put(0, 1);
    s.put(0b10, 2);
    s.put(0b11, 2);
    s.put(0b11, 2);
    // 跳过位平面（2×2）：invert=0, imode=COLSKIP(011), 两列都整列清零
    s.put(0, 1);
    s.put(0b011, 3);
    s.put(0, 2);
    s.put(0, 2); // mv_table_index
    s.put(0, 2); // cbptab
    s.put(0, 1); // ttmbf = 0 -> TT_8X8
    s.put(0, 1); // c_ac（decode012）
    s.put(0, 1); // dc
    s.bytes()
}

fn heavy_rounds() -> usize {
    std::env::var("V2V_FUZZ_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000)
}

/// 底本质量断言：底本必须**解得出来**，而且**走到深处**。
#[test]
fn baseline_is_not_degenerate() {
    let seq = SeqHeader::parse(&seq_bytes()).expect("底本序列头该能解");
    assert_eq!(seq.profile, v2v_vc1::Profile::Main);
    assert!(seq.extended_mv && seq.vstransform);

    let mut st = HeaderState::default();
    let f = FrameHeader::parse(&frame_bytes(), &seq, 2, 2, &mut st).expect("底本帧头该能解");
    assert_eq!(f.pict_type, PictType::P);
    let p = f.p.as_ref().expect("P 帧该有 P 段");
    assert!(p.mv_type_plane.is_some(), "底本该走到「混合 MV」那条路");
    assert_eq!(p.skip_plane.imode, Imode::ColSkip, "底本该是非 RAW 的跳过平面");
    assert!(!p.skip_plane.raw);
}

/// 变异帧头：**允许 `Err`，禁止 `panic`**。返回「仍解出 `Ok`」的轮数。
fn fuzz_frames(n: usize) -> usize {
    let seq = SeqHeader::parse(&seq_bytes()).unwrap();
    let base = frame_bytes();
    let mut rng = Rng(0x243F_6A88_85A3_08D3);
    let mut ok = 0usize;

    for _ in 0..n {
        let mut data = base.clone();
        for _ in 0..1 + rng.below(MAX_MUTATIONS) {
            match rng.below(3) {
                // 翻一位
                0 => {
                    let i = rng.below(data.len());
                    data[i] ^= 1 << rng.below(8);
                }
                // 整字节替换
                1 => {
                    let i = rng.below(data.len());
                    data[i] = rng.next() as u8;
                }
                // 截断（提前结束）
                _ => {
                    let k = rng.below(data.len());
                    data.truncate(k.max(1));
                }
            }
        }
        // 网格尺寸也一起变 —— 位平面是按它撒点的。
        let (w, h) = (1 + rng.below(8), 1 + rng.below(8));
        let mut st = HeaderState::default();
        if FrameHeader::parse(&data, &seq, w, h, &mut st).is_ok() {
            ok += 1;
        }
    }
    ok
}

/// 变异序列头（extradata）：**允许 `Err`，禁止 `panic`**。
fn fuzz_seq(n: usize) -> usize {
    let base = seq_bytes();
    let mut rng = Rng(0x13198A2E03707344);
    let mut ok = 0usize;

    for _ in 0..n {
        let mut data = base.clone();
        for _ in 0..1 + rng.below(MAX_MUTATIONS) {
            let i = rng.below(data.len());
            data[i] = rng.next() as u8;
        }
        // 长度也变（extradata 短了要报 `Eof` 而不是 panic）。
        if rng.below(4) == 0 {
            let k = rng.below(data.len() + 1);
            data.truncate(k);
        }
        if SeqHeader::parse(&data).is_ok() {
            ok += 1;
        }
    }
    ok
}

#[test]
fn mutated_frame_headers_never_panic() {
    let n = 20_000;
    let ok = fuzz_frames(n);
    eprintln!(
        "帧头变异：{n} 轮，其中 {ok} 轮仍解出 `Ok`（{:.0}%）",
        100.0 * ok as f64 / n as f64
    );
    // 「没 panic」不是判据 —— 判据是**真的解进去了**：`Ok` 太少说明底本太脆，
    // 那这一局只是空转（实测 ~16%）。
    assert!(ok > n / 20, "变异后能解出来的太少（{ok}/{n}）—— 这一局没测到实处");
}

#[test]
fn mutated_seq_headers_never_panic() {
    let n = 20_000;
    let ok = fuzz_seq(n);
    eprintln!(
        "序列头变异：{n} 轮，其中 {ok} 轮仍解出 `Ok`（{:.0}%）",
        100.0 * ok as f64 / n as f64
    );
    assert!(ok > 0, "变异后一个都没解出来（{n} 轮）—— 这一局没测到实处");
}

/// 重局：本机深挖用。轮数由 `V2V_FUZZ_ROUNDS` 给。
///
/// ```text
/// V2V_FUZZ_ROUNDS=200000 cargo test -p v2v-vc1 --release --test fuzz -- --ignored --nocapture
/// ```
#[test]
#[ignore = "重局：本机深挖用，轮数由 V2V_FUZZ_ROUNDS 给（默认 20 万）"]
fn heavy_round() {
    let n = heavy_rounds();
    let (a, b) = (fuzz_frames(n), fuzz_seq(n));
    eprintln!("重局 {n} 轮：帧头仍解出 {a}，序列头仍解出 {b}");
}
