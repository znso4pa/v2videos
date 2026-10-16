#!/usr/bin/env python3
"""把 IntraX8（`j_type` 那一套内帧子解码器）需要的表读出来，排版成 Rust 常量。

**IntraX8 是什么**：WMV2 与 VC-1 共用的一个**独立内帧子解码器**（8×8 块级，
不是 16×16 宏块）。WMV2 的 extradata 里 `j_type` 置位时，**每个 I 帧**都走它 ——
所以没有它，那类流解到第一个 I 帧就得停。

**为什么是生成而不是手抄**：光 AC 码表就有 `2×2×8 = 32` 张 × 77 个码字
（再加 DC 的 16 张 × 34、方向表的 6 张 × 12）—— 手抄必错，而且错一个码字
只会让**某些**块解出噪声。

**表本身是什么**：IntraX8 的 VLC、量化矩阵、预测权重与方向查表 —— 编解码器格式的一部分，
不是某个实现的私有数据。生成器只做「读出来 + 排版」，不做任何推导。

**许可（重要）**：读的是 FFmpeg 的数据文件（**LGPL-2.1-or-later**），
所以产出的 `intrax8_tables.rs` 按 LGPL-2.1-or-later 使用，
整个仓库统一按 LGPL-2.1-or-later（见 README 的「许可证」一节）。

用法：
    python3 tools/gen_intrax8_tables.py [ffmpeg_src_dir]

写 `crates/msmpeg4/src/intrax8_tables.rs`（路径相对仓库根，自动定位）。
表来自 `libavcodec/intrax8huf.h`、`libavcodec/intrax8.c`、`libavcodec/intrax8dsp.c`。

默认源目录 `/tmp/ffsize/ffmpeg-7.1`。
⚠️ **默认源目录是作者的本地路径**，第三方 clone 里不存在。
自己准备一份 FFmpeg 7.1 源码（<https://ffmpeg.org/download.html>），把路径作为第一个参数传进来。
"""

import os
import re
import sys

from ffmpeg_src import copyright_lines

DEFAULT_SRC = "/tmp/ffsize/ffmpeg-7.1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

args = [a for a in sys.argv[1:] if not a.startswith("-")]
SRC = args[0] if args else DEFAULT_SRC

SOURCES = ["intrax8huf.h", "intrax8.c", "intrax8dsp.c"]


def read(path):
    with open(f"{SRC}/libavcodec/{path}") as f:
        return f.read()


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    return re.sub(r"//[^\n]*", " ", s)


def body(text, name):
    """取出 `NAME[...] = { ... };` 的花括号内容（**已剥掉注释**）。"""
    m = re.search(re.escape(name) + r"\s*\[[^=]*=\s*\{", text)
    if not m:
        raise SystemExit(f"找不到 {name}")
    i = m.end() - 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return strip_comments(text[i + 1 : j])
    raise SystemExit(f"{name} 括号不配对")


def nums(s):
    return [int(x, 0) for x in re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", s)]


def flat(text, name):
    return nums(body(text, name))


def pairs(text, name):
    """所有最内层 `{ a, b }` -> [(a, b), ...]，**按出现顺序**。"""
    return [(nums(g)[0], nums(g)[1]) for g in re.findall(r"\{\s*([^{}]*?)\s*\}", body(text, name))]


def code_pairs(text, name, per_table):
    """IntraX8 的码表是 `{符号, 码长}` —— **不是** `{码字, 码长}`。

    上游用 `ff_vlc_init_tables_from_lengths()` 建表：只给长度，码字是**规范构造**
    出来的（`vlc.c` 里一个 32 位定点累加器：按表内顺序，每个码字取累加器的高
    `len` 位，然后 `code += 1 << (32 - len)`）。所以这里要把长度**翻译成码字** ——
    直接把第一列当码字用会得到一张「能建起来、但解出来全错」的表。

    ⚠️ **码字按表内顺序算，输出按符号排序** —— 这两件事必须分开：
    规范码字是按**表内顺序**累加出来的（上游 `vlc.c` 的循环就是照表走），
    但这些表里**符号顺序 ≠ 表内顺序**（`x8_ac_quant_table[1][1][7]` 的第 71 项
    符号是 58）。落盘时按符号排好，Rust 侧才能继续用「下标即符号」的 `Vlc::new`
    —— 不然整张表会「能建起来、解出来全错」（第一版就是这么错的）。

    `per_table` 是**一张表**有几个码字（12 / 34 / 77）：整块里有多张表，
    Kraft 和与「符号是 0..n-1 的排列」这两条自检都得**逐表**做（整块做会得到 4.0）。
    """
    entries = pairs(text, name)
    if len(entries) % per_table:
        raise SystemExit(f"{name}: 共 {len(entries)} 项，不是 {per_table} 的整数倍")
    out = []
    for start in range(0, len(entries), per_table):
        chunk = entries[start : start + per_table]
        code = 0
        kraft = 0.0
        by_sym = [None] * per_table
        for sym, length in chunk:
            if length <= 0:
                raise SystemExit(f"{name}: 第 {start} 张表里有非正的码长")
            if not 0 <= sym < per_table:
                raise SystemExit(f"{name}: 第 {start} 张表里符号 {sym} 越界")
            # 上游自己的检查：累加器的低位必须为 0（否则它会报 Invalid VLC）。
            if code & ((1 << (32 - length)) - 1):
                raise SystemExit(f"{name}: 第 {start} 张表累加器低位不为 0")
            by_sym[sym] = (code >> (32 - length), length)
            kraft += 2.0**-length
            code += 1 << (32 - length)
        if abs(kraft - 1.0) > 1e-9:
            raise SystemExit(f"{name}: 第 {start} 张表的 Kraft 和是 {kraft}，不是 1")
        if any(e is None for e in by_sym):
            raise SystemExit(f"{name}: 第 {start} 张表的符号不是 0..{per_table - 1} 的排列")
        out.extend(by_sym)
    return out


def rows(text, name):
    """所有最内层 `{ ... }`，每组返回其全部数字（给数值矩阵用）。"""
    return [nums(g) for g in re.findall(r"\{\s*([^{}]*?)\s*\}", body(text, name))]


# ---- `ac_decode_table` 那 27 项是**宏拼出来的表达式**，不是数字。
# 手抄一遍等于把「哪个 offset 配哪个 extra 位数」抄错一遍，所以让生成器把宏拆开算。
_AC_MACROS = {
    "extra_run": 0xFF << 8,
    "extra_level": 0x00 << 8,
}


def eval_ac_entry(expr):
    """`extra_bits(3) | extra_run | run_offset(16) | level_offset(0)` -> u32。"""
    val = 0
    for term in expr.split("|"):
        term = term.strip()
        if not term:
            continue
        m = re.fullmatch(r"extra_bits\((\d+)\)", term)
        if m:
            val |= int(m.group(1))
            continue
        m = re.fullmatch(r"run_offset\((\d+)\)", term)
        if m:
            val |= int(m.group(1)) << 16
            continue
        m = re.fullmatch(r"level_offset\((\d+)\)", term)
        if m:
            val |= int(m.group(1)) << 24
            continue
        if term in _AC_MACROS:
            val |= _AC_MACROS[term]
            continue
        raise SystemExit(f"认不出的表达式项：{term!r}（上游改了宏？）")
    return val


class Out:
    def __init__(self):
        self.buf = []

    def w(self, s):
        self.buf.append(s)

    def pairs(self, name, data, stride, per_line=1):
        """`[[u32; 2]; N]`，**平铺**（N 张 × stride 项），配一个 stride 常量。"""
        self.w(f"pub static {name}: [[u32; 2]; {len(data)}] = [\n")
        for i in range(0, len(data), per_line):
            chunk = data[i : i + per_line]
            self.w("    " + " ".join(f"[{a}, {b}]," for a, b in chunk) + "\n")
        self.w("];\n\n")
        self.w(f"/// 每张表的码字数（`{name}` 的 stride）。\n")
        self.w(f"pub const {name}_STRIDE: usize = {stride};\n\n")

    def flat(self, name, ty, data, per_line=16):
        self.w(f"pub static {name}: [{ty}; {len(data)}] = [\n")
        for i in range(0, len(data), per_line):
            self.w("    " + ", ".join(str(v) for v in data[i : i + per_line]) + ",\n")
        self.w("];\n\n")

    def matrix(self, name, ty, data):
        self.w(f"pub static {name}: [[{ty}; {len(data[0])}]; {len(data)}] = [\n")
        for row in data:
            self.w("    [" + ", ".join(str(v) for v in row) + "],\n")
        self.w("];\n\n")

    def text(self):
        return "".join(self.buf)


def header():
    out = Out()
    out.w(
        """//! IntraX8（WMV2 / VC-1 共用的内帧子解码器）需要的表。
//!
//! **本文件由 `tools/gen_intrax8_tables.py` 生成，不要手改。**
//!
//! ## 表本身是什么
//!
//! IntraX8 的 VLC（AC / DC / 方向）、量化矩阵、预测权重与方向查表 ——
//! 编解码器格式的一部分，不是某个实现的私有数据。生成器只做「读出来 + 排版」。
//!
//! ## 码表的排布
//!
//! `[code, len]`，**下标即符号**。上游的源表是 `{符号, 码长}`（用
//! `ff_vlc_init_tables_from_lengths` 建，码字是规范构造出来的），生成器把长度
//! **翻译成码字**后再落盘 —— 直接把第一列当码字用会得到一张「能建起来、但解出来全错」的表。
//! 多维的几张**平铺**成一维，配一个 `*_STRIDE` 常量：调用方切一片
//! `&TABLE[i * S..][..S]` 就能交给 VLC 构造器，不用跟四维数组打交道。
//!
//! ## 许可
//!
//! 本文件的数据取自 FFmpeg 7.1（LGPL-2.1-or-later）的这几个文件，
//! 以及它们的版权声明（由生成器从源文件原样读出）：
//!
//! ```text
"""
    )
    for name in SOURCES:
        lines = copyright_lines(f"{SRC}/libavcodec/{name}")
        out.w(f"//! `{name}`\n")
        if lines:
            for line in lines:
                out.w(f"//!     {line}\n")
        else:
            out.w("//!     （该文件头没有版权行）\n")
    out.w(
        """//! ```
//!
//! 与本仓的许可一致，整个仓库统一按 LGPL-2.1-or-later 使用。
//! 详见 README 的「许可证」一节与根目录的 `NOTICE`。

#![allow(dead_code)]

"""
    )
    return out.text()


def gen():
    huf = read("intrax8huf.h")
    c8 = read("intrax8.c")
    dsp = read("intrax8dsp.c")

    out = Out()
    out.w(header())

    # ---- 码表：上游的维度是 `[quant<13][intra/inter][select][77]` 这类，
    # 平铺后**行主序**，与 C 的 `table[i][j][k]` 逐层展开一致。
    out.w("// ---- 方向（orient）码表：`[quant<13][select]`。\n\n")
    out.pairs("X8_ORIENT_LOW", code_pairs(huf, "x8_orient_lowquant_table", 12), 12)
    out.pairs("X8_ORIENT_HIGH", code_pairs(huf, "x8_orient_highquant_table", 12), 12)

    out.w("// ---- DC 码表：`[quant<13][select]`，每张 34 个码字。\n\n")
    out.pairs("X8_DC_QUANT", code_pairs(huf, "x8_dc_quant_table", 34), 34)

    out.w("// ---- AC 码表：`[quant<13][intra/inter][select]`，每张 77 个码字。\n\n")
    out.pairs("X8_AC_QUANT", code_pairs(huf, "x8_ac_quant_table", 77), 77)

    # ---- 数值表。
    out.w("// ---- 空间预测的权重（`spatial_compensation_0` 用）。\n\n")
    out.flat("X8_ZERO_PREDICTION_WEIGHTS", "u16", flat(dsp, "zero_prediction_weights"))

    out.w("// ---- 可选的量化矩阵（帧头一位决定用不用）。\n\n")
    out.flat("X8_QUANT_TABLE", "i16", flat(c8, "quant_table"))

    out.w("// ---- AC 码字 46..72 的「额外位数 / 加在 run 还是 level 上 / 偏移」。\n\n")
    # ⚠️ 这里**不能**先 `strip_comments`：那 27 项的 `/* 46 */` 注释正是用来认行的
    # （剥掉就一条都找不到）。所以拿原文切，逐行认 `/* n */ 表达式,`。
    m = re.search(r"ac_decode_table\[\]\s*=\s*\{(.*?)\n\};", c8, re.S)
    if not m:
        raise SystemExit("找不到 ac_decode_table")
    entries = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("/*"):
            continue
        expr = line.split("*/", 1)[1].rstrip(",").strip()
        entries.append(eval_ac_entry(expr))
    if len(entries) != 27:
        raise SystemExit(f"ac_decode_table 应有 27 项，读出 {len(entries)} 项")
    out.flat("X8_AC_DECODE_TABLE", "u32", entries, per_line=3)

    out.w("// ---- 码字 73/74 的 (run, level) 混合表。\n\n")
    out.flat("X8_CRAZY_MIX_RUNLEVEL", "u8", flat(c8, "crazy_mix_runlevel"))

    out.w("// ---- DC 码字的额外位数偏移。\n\n")
    out.flat("X8_DC_INDEX_OFFSET", "u8", flat(c8, "dc_index_offset"))

    out.w("// ---- 方向预测的查表：`prediction_table[edges 推断出的 orient][原始 orient]`。\n\n")
    out.matrix("X8_PREDICTION_TABLE", "u8", rows(c8, "prediction_table"))

    return out.text()


def write(rel, text):
    path = os.path.join(ROOT, rel)
    old = None
    if os.path.exists(path):
        with open(path) as f:
            old = f.read()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    state = "未变" if old == text else ("更新" if old is not None else "新建")
    print(f"{state}  {rel}  ({len(text)} 字节, {text.count(chr(10))} 行)")


if not os.path.isdir(f"{SRC}/libavcodec"):
    raise SystemExit(f"源目录不对：{SRC}/libavcodec 不存在（用 `python3 tools/gen_intrax8_tables.py <ffmpeg_src>`）")

write("crates/msmpeg4/src/intrax8_tables.rs", gen())
