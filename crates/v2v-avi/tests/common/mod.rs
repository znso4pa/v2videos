//! AVI 集成测试的共用件：RIFF 原语 + 一个**参数化**的最小 AVI 构造器。
//!
//! ## 为什么要有这个模块
//!
//! 仓里原本有三处各写各的「手搓 AVI」（`avi/tests/oracle.rs` 两处、
//! `cli/tests/convert.rs` 一处、`msmpeg4/tests/common/mod.rs` 一处）——
//! 它们**不是**重复：每一处都是为一条特定断言定制的（奇数长度帧的补齐、
//! 空槽不产样本、`idx1` 偏移的精确值……），合并反而会把断言弄糊。
//!
//! 但这个模块要解决的是另一件事：模糊测试需要的是**参数化**的底本
//! （帧数 / 尺寸 / 有没有 `idx1` 都要能调），而上面三处都是写死的。
//! 所以这里放的是**原语 + 一个通用构造器**，`fuzz.rs` 用它，
//! `oracle.rs` 用它的原语（自己那 4 个函数删掉了，布局代码不动）。

#![allow(dead_code)] // 共用工具箱：每个测试目标只用到其中一部分

pub fn u32le(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

pub fn u16le(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_le_bytes());
}

pub fn cc(v: &mut Vec<u8>, s: &[u8; 4]) {
    v.extend_from_slice(s);
}

/// 一个 RIFF 块：`id` + 长度 + 内容 + **奇数长度补一个 0**。
///
/// ⚠️ 那个补齐是 RIFF 的硬规矩，而且**长度字段写的是补齐前**的值 ——
/// 写错会让后面所有块的偏移整体错位。
pub fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    cc(&mut v, id);
    u32le(&mut v, body.len() as u32);
    v.extend_from_slice(body);
    if body.len() % 2 == 1 {
        v.push(0);
    }
    v
}

/// `LIST <kind> <body>`。
pub fn list(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut inner = Vec::new();
    cc(&mut inner, kind);
    inner.extend_from_slice(body);
    chunk(b"LIST", &inner)
}

/// 一份最小但**结构完整**的 AVI：`RIFF/AVI ` + `LIST hdrl` + `LIST movi`（+ 可选 `idx1`）。
///
/// - `frames`：逐帧的裸视频载荷（`00dc` 块）。第 0 帧标关键帧。
/// - `with_index`：要不要写 `idx1`（`avih` 的 HASINDEX 标志跟着一起设）。
///   **两种都要有**：解封装器读索引与不读索引是两条不同的路。
/// - `fps`：只影响 `avih` 的微秒/帧与 `strh` 的 rate/scale。
pub fn build_avi(frames: &[Vec<u8>], w: u32, h: u32, fps: u32, with_index: bool) -> Vec<u8> {
    let n = frames.len() as u32;

    // avih（56 字节）
    let mut avih = Vec::new();
    u32le(&mut avih, 1_000_000 / fps.max(1)); // 微秒/帧
    u32le(&mut avih, 0); // 最大字节率
    u32le(&mut avih, 0); // padding
    u32le(&mut avih, if with_index { 0x10 } else { 0 }); // 标志：HASINDEX
    u32le(&mut avih, n); // 总帧数
    u32le(&mut avih, 0); // 初始帧
    u32le(&mut avih, 1); // 流数
    u32le(&mut avih, 0); // 建议缓冲
    u32le(&mut avih, w);
    u32le(&mut avih, h);
    for _ in 0..4 {
        u32le(&mut avih, 0);
    }

    // strh（56 字节）
    let mut strh = Vec::new();
    cc(&mut strh, b"vids");
    cc(&mut strh, b"MP43");
    u32le(&mut strh, 0); // 标志
    u16le(&mut strh, 0); // 优先级
    u16le(&mut strh, 0); // 语言
    u32le(&mut strh, 0); // 初始帧
    u32le(&mut strh, 1); // scale
    u32le(&mut strh, fps); // rate
    u32le(&mut strh, 0); // start
    u32le(&mut strh, n); // length
    u32le(&mut strh, 0); // 建议缓冲
    u32le(&mut strh, 0xFFFF_FFFF); // quality
    u32le(&mut strh, 0); // sample size
    for _ in 0..4 {
        u16le(&mut strh, 0); // rcFrame
    }

    // strf：BITMAPINFOHEADER（40 字节）
    let mut strf = Vec::new();
    u32le(&mut strf, 40);
    u32le(&mut strf, w);
    u32le(&mut strf, h);
    u16le(&mut strf, 1);
    u16le(&mut strf, 24);
    cc(&mut strf, b"MP43");
    for _ in 0..5 {
        u32le(&mut strf, 0);
    }

    let strl = list(
        b"strl",
        &[chunk(b"strh", &strh), chunk(b"strf", &strf)].concat(),
    );
    let hdrl = list(b"hdrl", &[chunk(b"avih", &avih), strl].concat());

    // movi：逐帧一个 `00dc` 块。**偏移按块在 movi 数据区里的位置累加** ——
    // 注意 `chunk` 会给奇数长度的块补一个字节，所以步长不是 `8 + len`。
    let mut movi_body = Vec::new();
    let mut offsets = Vec::with_capacity(frames.len());
    for f in frames {
        offsets.push(movi_body.len() as u32);
        movi_body.extend_from_slice(&chunk(b"00dc", f));
    }
    let movi = list(b"movi", &movi_body);

    let body = if with_index {
        let mut idx1 = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            cc(&mut idx1, b"00dc");
            // 第 0 帧标关键帧（`AVIIF_KEYFRAME`）。
            u32le(&mut idx1, if i == 0 { 0x10 } else { 0 });
            u32le(&mut idx1, offsets[i]);
            u32le(&mut idx1, f.len() as u32);
        }
        [hdrl, movi, chunk(b"idx1", &idx1)].concat()
    } else {
        [hdrl, movi].concat()
    };

    let mut out = Vec::new();
    cc(&mut out, b"RIFF");
    u32le(&mut out, (body.len() + 4) as u32);
    cc(&mut out, b"AVI ");
    out.extend_from_slice(&body);
    out
}
