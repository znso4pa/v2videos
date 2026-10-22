#!/usr/bin/env python3
"""从 FFmpeg 的数据文件生成 `crates/mpeg12/src/tables.rs`。

用法：
`python3 tools/gen_mpeg12_tables.py <ffmpeg 源码根>/libavcodec/mpeg12data.c <ffmpeg 源码根>/libavcodec/mpeg12.c`

- `mpeg12data.c`：量化矩阵、DC / RL / 宏块地址 / 模式 / 运动矢量码表。
- `mpeg12.c`：P 帧 `macroblock_type` 码表与语义（`table_mb_ptype` / `ptype2mb_type`）。

**为什么不手写**：这几张表加起来两百多条，手抄最容易出「整体对、个别错」，
而那种错只表现为画面局部花掉，反推代价极高。生成 + 自检（前缀无关、Kraft 和）
比手抄可靠得多。
"""

import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "libavcodec/mpeg12data.c"
SRC2 = sys.argv[2] if len(sys.argv) > 2 else "libavcodec/mpeg12.c"
OUT = "crates/mpeg12/src/tables.rs"


def body(src, name):
    i = src.index(name)
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[j : k + 1]
    raise SystemExit(f"unbalanced braces around {name}")


def ints(text):
    return [int(x, 0) for x in re.findall(r"-?(?:0x[0-9a-fA-F]+|\d+)", text)]


def pairs(text):
    return [(int(c, 0), int(b)) for c, b in
            re.findall(r"\{\s*(0x[0-9a-fA-F]+|\d+)\s*,\s*(\d+)\s*\}", text)]


def mb_ptype_semantics(text):
    """把 `ptype2mb_type` 的每个 C 条目翻成 `(intra, coded, motion, quant)`。

    不读宏的数值 —— 只认宏名。宏的位值会随实现变，语义不会。
    """
    out = []
    for item in text.strip().strip("{}").split(","):
        item = item.strip()
        if not item:
            continue
        if "MB_TYPE_ZERO_MV" in item:
            motion = "Zero"
        elif "MB_TYPE_FORWARD_MV" in item:
            motion = "Forward"
        else:
            motion = "None"
        out.append((
            "MB_TYPE_INTRA" in item,
            "MB_TYPE_CBP" in item,
            motion,
            "MB_TYPE_QUANT" in item,
        ))
    return out


def mb_btype_semantics(text):
    """把 `btype2mb_type` 的每个条目翻成 `(intra, coded, forward, backward, quant)`。

    `MB_TYPE_BIDIR_MV` 同时意味着前向与后向。
    """
    out = []
    for item in text.strip().strip("{}").split(","):
        item = item.strip()
        if not item:
            continue
        bidir = "MB_TYPE_BIDIR_MV" in item
        out.append((
            "MB_TYPE_INTRA" in item,
            "MB_TYPE_CBP" in item,
            "MB_TYPE_FORWARD_MV" in item or bidir,
            "MB_TYPE_BACKWARD_MV" in item or bidir,
            "MB_TYPE_QUANT" in item,
        ))
    return out


def main():
    src = open(SRC).read()
    src2 = open(SRC2).read()

    intra = ints(body(src, "ff_mpeg1_default_intra_matrix"))[:64]
    non_intra = ints(body(src, "ff_mpeg1_default_non_intra_matrix"))[:64]
    dc_lum = list(zip(ints(body(src, "ff_mpeg12_vlc_dc_lum_code")),
                      ints(body(src, "ff_mpeg12_vlc_dc_lum_bits"))))
    dc_chr = list(zip(ints(body(src, "ff_mpeg12_vlc_dc_chroma_code")),
                      ints(body(src, "ff_mpeg12_vlc_dc_chroma_bits"))))
    mb_addr = pairs(body(src, "ff_mpeg12_mbAddrIncrTable"))
    mb_pat = pairs(body(src, "ff_mpeg12_mbPatTable"))
    mv = pairs(body(src, "ff_mpeg12_mbMotionVectorTable"))
    rl_vlc = pairs(body(src, "ff_mpeg1_vlc_table"))
    level = ints(body(src, "ff_mpeg12_level"))
    run = ints(body(src, "ff_mpeg12_run"))

    ptype = pairs(body(src2, "table_mb_ptype"))
    psem = mb_ptype_semantics(body(src2, "ptype2mb_type"))
    if len(ptype) != len(psem):
        raise SystemExit(f"P mb_type: {len(ptype)} codes vs {len(psem)} semantics")
    btype = pairs(body(src2, "table_mb_btype"))
    bsem = mb_btype_semantics(body(src2, "btype2mb_type"))
    if len(btype) != len(bsem):
        raise SystemExit(f"B mb_type: {len(btype)} codes vs {len(bsem)} semantics")

    out = []
    w = out.append
    w("//! MPEG-1 的码表与量化矩阵。**本文件由 `tools/gen_mpeg12_tables.py` 生成，不要手改。**")
    w("//!")
    w("//! 数据从一份已知正确的实现里机械抽取，抽取后由本 crate 的 `table_tests`")
    w("//! 做前缀无关 / Kraft 自检 —— 手抄这种表最容易出「整体对、个别错」，")
    w("//! 而那种错只表现为画面局部花掉，反推代价极高。")
    w("")
    w("/// 默认帧内量化矩阵（8×8，行优先）。")
    w("pub const INTRA_MATRIX: [u8; 64] = [")
    for i in range(0, 64, 8):
        w("    " + ", ".join(str(v) for v in intra[i:i + 8]) + ",")
    w("];")
    w("")
    w("/// 默认帧间量化矩阵（8×8，行优先）。P 帧残差用它反量化。")
    w("pub const NON_INTRA_MATRIX: [u8; 64] = [")
    for i in range(0, 64, 8):
        w("    " + ", ".join(str(v) for v in non_intra[i:i + 8]) + ",")
    w("];")
    w("")
    w("/// 一条前缀码：`bits` 位码 `code`。")
    w("#[derive(Debug, Clone, Copy, PartialEq, Eq)]")
    w("pub struct Code {")
    w("    pub code: u16,")
    w("    pub bits: u8,")
    w("}")
    w("")

    def emit_code_table(name, doc, entries):
        w(f"/// {doc}")
        w(f"pub const {name}: [Code; {len(entries)}] = [")
        for code, bits in entries:
            w(f"    Code {{ code: 0x{code:04x}, bits: {bits} }},")
        w("];")
        w("")

    emit_code_table("DC_LUM", "亮度块 DC 尺寸码，下标即 `dct_dc_size`。", dc_lum)
    emit_code_table("DC_CHROMA", "色度块 DC 尺寸码，下标即 `dct_dc_size`。", dc_chr)
    emit_code_table("MB_ADDR_INCR", "宏块地址增量码，下标 + 1 即增量值。", mb_addr)
    emit_code_table("MB_PATTERN", "`coded_block_pattern` 码，下标即 CBP。", mb_pat)
    emit_code_table("MV_CODE", "运动矢量码，下标即 `|motion_code|`（0..=16）。", mv)

    w("/// 一条 run/level 码字。level 的符号是码字之后的**独立 1 位**。")
    w("#[derive(Debug, Clone, Copy, PartialEq, Eq)]")
    w("pub struct RlEntry {")
    w("    pub code: u16,")
    w("    pub bits: u8,")
    w("    pub run: u8,")
    w("    pub level: i16,")
    w("}")
    w("")
    w(f"/// `(run, level)` 到码字的表。")
    w(f"pub const MPEG1_RL: [RlEntry; {len(level)}] = [")
    for i in range(len(level)):
        code, bits = rl_vlc[i]
        w(f"    RlEntry {{ code: 0x{code:04x}, bits: {bits}, run: {run[i]}, level: {level[i]} }},")
    w("];")
    w("")
    w("/// 块结束码 `10`。")
    w("pub const EOB_CODE: u16 = 0b10;")
    w("pub const EOB_BITS: u8 = 2;")
    w("/// 逃逸码 `000001`，后跟定长的 run / level。")
    w("pub const ESCAPE_CODE: u16 = 0b000001;")
    w("pub const ESCAPE_BITS: u8 = 6;")
    w("")

    w("/// P 帧宏块的运动来源。")
    w("#[derive(Debug, Clone, Copy, PartialEq, Eq)]")
    w("pub enum Motion {")
    w("    /// 帧内宏块，不做运动补偿。")
    w("    None,")
    w("    /// 前向预测但运动矢量为 0（不读矢量位）。")
    w("    Zero,")
    w("    /// 前向预测，运动矢量从码流里读。")
    w("    Forward,")
    w("}")
    w("")
    w("/// P 帧的一条 `macroblock_type`：码字 + 语义。")
    w("#[derive(Debug, Clone, Copy, PartialEq, Eq)]")
    w("pub struct PMbType {")
    w("    pub code: u16,")
    w("    pub bits: u8,")
    w("    pub intra: bool,")
    w("    /// 后面跟 `coded_block_pattern`（决定哪些块有残差）。")
    w("    pub coded: bool,")
    w("    pub motion: Motion,")
    w("    pub quant: bool,")
    w("}")
    w("")
    w(f"/// P 帧 `macroblock_type` 表（下标即 VLC 符号）。")
    w(f"pub const P_MB_TYPE: [PMbType; {len(ptype)}] = [")
    for (code, bits), (intra_f, coded, motion, quant) in zip(ptype, psem):
        w(f"    PMbType {{ code: 0x{code:04x}, bits: {bits}, intra: {str(intra_f).lower()}, "
          f"coded: {str(coded).lower()}, motion: Motion::{motion}, quant: {str(quant).lower()} }},")
    w("];")
    w("")
    w("/// B 帧的一条 `macroblock_type`：前向 / 后向 / 双向预测。")
    w("#[derive(Debug, Clone, Copy, PartialEq, Eq)]")
    w("pub struct BMbType {")
    w("    pub code: u16,")
    w("    pub bits: u8,")
    w("    pub intra: bool,")
    w("    pub coded: bool,")
    w("    pub forward: bool,")
    w("    pub backward: bool,")
    w("    pub quant: bool,")
    w("}")
    w("")
    w("/// B 帧 `macroblock_type` 表（下标即 VLC 符号）。")
    w(f"pub const B_MB_TYPE: [BMbType; {len(btype)}] = [")
    for (code, bits), (intra_f, coded, fwd, bwd, quant) in zip(btype, bsem):
        w(f"    BMbType {{ code: 0x{code:04x}, bits: {bits}, intra: {str(intra_f).lower()}, "
          f"coded: {str(coded).lower()}, forward: {str(fwd).lower()}, backward: {str(bwd).lower()}, "
          f"quant: {str(quant).lower()} }},")
    w("];")

    open(OUT, "w").write("\n".join(out) + "\n")
    print(f"wrote {OUT}: {len(level)} rl, {len(mb_addr)} mb_addr, {len(mb_pat)} mb_pat, "
          f"{len(mv)} mv, {len(ptype)} p_mb_type, {len(btype)} b_mb_type")


if __name__ == "__main__":
    main()
