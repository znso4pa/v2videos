#!/usr/bin/env python3
"""把 H.263 / MPEG-4 Part 2 需要的 VLC / 反量化表读出来，排版成 Rust 常量。

**为什么是生成而不是手抄**：这些表有几千个数字，手抄必错；生成可复现，也把出处钉在一处。

**表本身是什么**：H.263 / MPEG-4 Part 2 的 VLC 与反量化定义 —— 编解码器格式的一部分，
不是某个实现的私有数据。生成器只做「读出来 + 排版」，不做任何推导。

**许可（重要）**：目前这份生成器读的是 FFmpeg 的数据文件（**LGPL-2.1-or-later**），
所以产出的 `tables.rs` 按 LGPL-2.1-or-later 使用，
整个仓库统一按 LGPL-2.1-or-later（见 README 的「许可证」一节）。
换一个数据源（规范附录、或别的宽松许可实现）即可整体替换 —— 生成器就是为此留的；
输出文件头会记下源文件与版本，方便日后核对。

用法：
    python3 tools/gen_tables.py [ffmpeg_src_dir]

同时写两个文件（路径相对仓库根，自动定位，不依赖当前工作目录）：

    crates/msmpeg4/src/tables.rs    MS-MPEG-4（MP43）用
    crates/mp4v/src/tables.rs       MPEG-4 Part 2 用

默认源目录 `/tmp/ffsize/ffmpeg-7.1`。
"""

import os
import re
import sys

from ffmpeg_src import copyright_lines

DEFAULT_SRC = "/tmp/ffsize/ffmpeg-7.1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

args = [a for a in sys.argv[1:] if not a.startswith("-")]
SRC = args[0] if args else DEFAULT_SRC

# 每个产出**自己**衍生自哪几个文件 —— 写进头注释，也是版权声明的依据。
MSMPEG4_SOURCES = ["msmpeg4data.c", "msmpeg4_vc1_data.c"]
MP4V_SOURCES = ["h263data.c", "mpeg4data.h", "mpegvideodata.c"]


def read(path):
    with open(f"{SRC}/libavcodec/{path}") as f:
        return f.read()


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


def strip_comments(s):
    """剥掉 `//` 行注释与 `/* */` 块注释。

    非剥不可：`ff_mpeg4_y_dc_scale_table` 这类表的**注释里也写着数字**
    （`//  0  1  2 ... 31` 标出索引），不剥就会把注释里的 32 个数当成表项，
    表长直接翻倍。这是个只会在特定表上发作的坑，所以放在 `body()` 里统一处理。
    """
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    return re.sub(r"//[^\n]*", " ", s)


def nums(s):
    return [int(x, 0) for x in re.findall(r"-?0x[0-9a-fA-F]+|-?\d+", s)]


def pairs(text, name):
    """`{ a, b }, ...` -> [(a, b), ...]"""
    out = []
    for g in re.findall(r"\{\s*([^{}]*?)\s*\}", body(text, name)):
        n = nums(g)
        out.append((n[0], n[1]))
    return out


def flat(text, name):
    return nums(body(text, name))


def pairs_nested(text, name):
    """所有最内层 `{a, b}`，按出现顺序返回。"""
    return [(nums(g)[0], nums(g)[1]) for g in re.findall(r"\{\s*([^{}]*?)\s*\}", body(text, name))]


def rows(text, name):
    """所有最内层 `{ ... }`，每组返回其全部数字（给 `[N][64]` 这类表）。"""
    return [nums(g) for g in re.findall(r"\{\s*([^{}]*?)\s*\}", body(text, name))]


class Out:
    """攒文本，最后一次性写盘 —— 免得生成到一半失败留下半个文件。"""

    def __init__(self):
        self.buf = []

    def w(self, s):
        self.buf.append(s)

    def pairs(self, name, data, ty="u32"):
        self.w(f"pub static {name}: [[{ty}; 2]; {len(data)}] = [\n")
        for a, b in data:
            self.w(f"    [{a}, {b}],\n")
        self.w("];\n\n")

    def flat(self, name, ty, data, per_line=16):
        self.w(f"pub static {name}: [{ty}; {len(data)}] = [\n")
        for i in range(0, len(data), per_line):
            self.w("    " + ", ".join(str(v) for v in data[i : i + per_line]) + ",\n")
        self.w("];\n\n")

    def text(self):
        return "".join(self.buf)


def attribution(sources):
    """源文件清单 + 各自文件头里的版权行 —— **原样读出，不手抄**（见 `tools/ffmpeg_src.py`）。"""
    out = []
    for name in sources:
        out.append(f"//! `{name}`")
        lines = copyright_lines(f"{SRC}/libavcodec/{name}")
        out.extend(f"//!     {line}" for line in lines)
        if not lines:
            out.append("//!     （该文件头没有版权行）")
    return "".join(line + "\n" for line in out)


def header(title, sources, extra=""):
    return (
        f"//! {title}\n"
        "//!\n"
        "//! **本文件由 `tools/gen_tables.py` 生成，不要手改。**\n"
        "//!\n"
        "//! ## 表本身是什么\n"
        "//!\n"
        "//! H.263 / MPEG-4 Part 2 的 VLC 与反量化定义 —— 编解码器格式的一部分，\n"
        "//! 不是某个实现的私有数据。生成器只做「读出来 + 排版」，不做任何推导。\n"
        "//!\n"
        "//! ## 许可\n"
        "//!\n"
        "//! 本文件的数据取自 FFmpeg 7.1（LGPL-2.1-or-later）的这几个文件，\n"
        "//! 以及它们的版权声明（由生成器从源文件原样读出）：\n"
        "//!\n"
        "//! ```text\n"
        + attribution(sources)
        + "//! ```\n"
        "//!\n"
        "//! 与本仓的许可一致，整个仓库统一按 LGPL-2.1-or-later 使用。\n"
        "//! 详见 README 的「许可证」一节与根目录的 `NOTICE`。\n"
        "//!\n"
        "//! 换一个数据源（规范附录、或别的宽松许可实现）即可整体替换 —— 生成器已就位，\n"
        "//! 且表本身与实现无关。\n"
        + extra
        + "\n#![allow(dead_code, clippy::all)]\n\n"
    )


def gen_msmpeg4():
    d = read("msmpeg4data.c")
    v = read("msmpeg4_vc1_data.c")

    o = Out()
    o.w(
        header(
            "MS-MPEG-4（MP43）解码需要的 VLC / 反量化表。",
            MSMPEG4_SOURCES,
            "//!\n"
            "//! 这些表是 **MP43 自用**的：宏块类型表 `MB_INTRA` / `MB_NON_INTRA`、\n"
            "//! 三套 escape 的 RL 表、以及 WMV1 的 DC 缩放表。**MPEG-4 Part 2 不能用**，\n"
            "//! 那边见 `crates/mp4v/src/tables.rs`。\n",
        )
    )

    # 交织的 (code, len) 表
    o.pairs("MB_NON_INTRA", pairs(d, "ff_table_mb_non_intra"))
    o.pairs("MB_INTRA", pairs(v, "ff_msmp4_mb_i_table"))
    o.pairs("INTER_INTRA", pairs(d, "ff_table_inter_intra"))

    # 系数 RL 表（vlc 是 (code,len)，run/level 是 int8）
    for tag in ("0", "1", "2", "4"):
        o.pairs(f"RL{tag}_VLC", pairs(d, f"table{tag}_vlc"))
        o.flat(f"RL{tag}_RUN", "i8", flat(d, f"table{tag}_run"))
        o.flat(f"RL{tag}_LEVEL", "i8", flat(d, f"table{tag}_level"))

    # MPEG-4 的通用 RL 表（ff_rl_table[2] 与 [5]）—— 在 mpeg4data.h 里
    md = read("mpeg4data.h")
    o.pairs("RL_INTRA_VLC", pairs(md, "ff_mpeg4_intra_vlc"))
    o.flat("RL_INTRA_RUN", "i8", flat(md, "ff_mpeg4_intra_run"))
    o.flat("RL_INTRA_LEVEL", "i8", flat(md, "ff_mpeg4_intra_level"))

    # 帧间通用 RL 表 `ff_rl_table[5]` == `ff_h263_rl_inter`（n=102, last=58）——
    # 在 h263data.c 里。P 帧的 rl_table_index=2 会落到这张表上。
    hd = read("h263data.c")
    o.pairs("RL_INTER_VLC", pairs(hd, "ff_inter_vlc"))
    o.flat("RL_INTER_RUN", "i8", flat(hd, "ff_inter_run"))
    o.flat("RL_INTER_LEVEL", "i8", flat(hd, "ff_inter_level"))

    # MPEG-4/H.263 的水平/垂直**交替扫描**（ac_pred 为真时用，不是 zig-zag）
    mvd = read("mpegvideodata.c")
    o.flat("H_SCAN", "u8", flat(mvd, "ff_alternate_horizontal_scan"))
    o.flat("V_SCAN", "u8", flat(mvd, "ff_alternate_vertical_scan"))

    # 运动矢量表。每张表 1100 个码（1099 个真值 + 1 个 escape），
    # escape 之后明文读 6+6 位；其余码用 `mvx/mvy` 查偏移（各 1099 项）。
    o.flat("MV0_CODE", "u16", flat(d, "table0_mv_code"))
    o.flat("MV0_LEN", "u8", flat(d, "table0_mv_bits"))
    o.flat("MV1_CODE", "u16", flat(d, "table1_mv_code"))
    o.flat("MV1_LEN", "u8", flat(d, "table1_mv_bits"))
    o.flat("MV0_X", "u8", flat(d, "table0_mvx"))
    o.flat("MV0_Y", "u8", flat(d, "table0_mvy"))
    o.flat("MV1_X", "u8", flat(d, "table1_mvx"))
    o.flat("MV1_Y", "u8", flat(d, "table1_mvy"))

    # DC 表：[table_index][luma/chroma] -> 120 个 (code,len)
    dc = pairs_nested(v, "ff_msmp4_dc_tables")
    assert len(dc) == 480, f"DC 表应有 480 项，得到 {len(dc)}"
    o.w("pub static DC: [[[[u32; 2]; 120]; 2]; 2] = [\n")
    k = 0
    for _ti in range(2):
        o.w("  [\n")
        for _lc in range(2):
            o.w("    [\n")
            for _ in range(120):
                a, b = dc[k]
                k += 1
                o.w(f"      [{a}, {b}],\n")
            o.w("    ],\n")
        o.w("  ],\n")
    o.w("];\n\n")

    # 扫描表
    sc = rows(v, "ff_wmv1_scantable")
    assert all(len(r) == 64 for r in sc), "扫描表每行应为 64 项"
    o.w(f"pub static SCAN: [[u8; 64]; {len(sc)}] = [\n")
    for r in sc:
        o.w("    [" + ", ".join(str(x) for x in r) + "],\n")
    o.w("];\n\n")

    # 反量化：DC 缩放表
    o.flat("Y_DC_SCALE", "u8", flat(d, "ff_wmv1_y_dc_scale_table"))
    o.flat("C_DC_SCALE", "u8", flat(d, "ff_wmv1_c_dc_scale_table"))
    o.flat("OLD_Y_DC_SCALE", "u8", flat(d, "ff_old_ff_y_dc_scale_table"))

    return o.text()


def gen_mp4v():
    """MPEG-4 Part 2（`mp4v`）编解码需要的表。

    刻意**自成一份**，不引 `msmpeg4::tables`：RL_INTRA / H_SCAN / V_SCAN 两边都要，
    但它们本来就是「MPEG-4 Part 2 的表」，让 msmpeg4 反过来借更别扭；
    生成器只有一个源，重复生成不会跑偏。
    """
    hd = read("h263data.c")
    md = read("mpeg4data.h")
    mvd = read("mpegvideodata.c")

    o = Out()
    o.w(
        header(
            "MPEG-4 Part 2（`mp4v`）编解码需要的表。",
            MP4V_SOURCES,
            "//!\n"
            "//! 与 `crates/msmpeg4/src/tables.rs` 的区别：那边是 MP43 的私有表\n"
            "//! （`MB_INTRA` / `MB_NON_INTRA`、三套 escape 的 RL 表、WMV1 的 DC 缩放），\n"
            "//! **这里一张都不能用**。MPEG-4 Part 2 的宏块语法是纯 H.263 的，\n"
            "//! DC 走 `ff_mpeg4_DCtab_*` 的「位长 VLC」，DC 缩放走 `ff_mpeg4_*_dc_scale_table`。\n",
        )
    )

    # ── 宏块类型 ────────────────────────────────────────────────────────────
    # I 帧：`cbpc` 3 位 → 0..8，其中 8 = stuffing（码长 9，解码端循环跳过）。
    # 低 2 位是色度 CBP，第 3 位（值 4）表示「本宏块带 dquant」。
    o.flat("INTRA_MCBPC_CODE", "u8", flat(hd, "ff_h263_intra_MCBPC_code"), 9)
    o.flat("INTRA_MCBPC_BITS", "u8", flat(hd, "ff_h263_intra_MCBPC_bits"), 9)

    # P 帧的 28 项，分 7 组各 4 项（低 2 位都是色度 CBP）：
    #   0..3 inter / 4..7 intra / 8..11 interQ / 12..15 intraQ
    #   16..19 inter4v / 20..23 stuffing（**码长 0**）/ 24..27 inter4v+Q
    # 解码端 `do { cbpc = get_vlc2(...) } while (cbpc == 20)` —— stuffing 靠循环跳过。
    o.flat("INTER_MCBPC_CODE", "u8", flat(hd, "ff_h263_inter_MCBPC_code"), 28)
    o.flat("INTER_MCBPC_BITS", "u8", flat(hd, "ff_h263_inter_MCBPC_bits"), 28)

    # CBPY：`{code, len}`（注意顺序 —— 跟 MCBPC 的「两张平行数组」不同）。
    # ⚠️ 索引是**取反后**的亮度 4 块 CBP：I 帧直取，**P 帧要 `cbpy ^ 0xF`**
    # （`mpeg4_decode_mb` 的帧间分支就是 `get_vlc2(...) ^ 0x0F`）。
    o.pairs("CBPY", pairs(hd, "ff_h263_cbpy_tab"), "u8")

    # ── 运动矢量 ───────────────────────────────────────────────────────────
    # `ff_mvtab[code] = (码, 码长)`，`code` 是「绝对值的量化档」而不是矢量本身：
    # 解码端 `val = (code - 1) << (f_code - 1)` 再补 `f_code - 1` 个低位、读一个符号位。
    # `code == 0` 是**零矢量**（只写一个 `1` 位，不读符号也不读低位）。
    o.pairs("MV_TAB", pairs(hd, "ff_mvtab"), "u8")

    # ── DC ─────────────────────────────────────────────────────────────────
    # `DC_LUM[size] = (code, len)`，`size` 是 DC 差的**位长**（不是值本身）。
    # 值写在 VLC 后面，用 `get_xbits` 的语义（负数是 `(-|v|) ^ ((1<<size)-1)`），
    # 且 `size > 8` 时后面还有一个 marker 位。
    o.pairs("DC_LUM", pairs(md, "ff_mpeg4_DCtab_lum"), "u8")
    o.pairs("DC_CHROM", pairs(md, "ff_mpeg4_DCtab_chrom"), "u8")

    # DC 反量化缩放表 —— **不是** MP43 的 `OLD_Y_DC_SCALE`。
    o.flat("Y_DC_SCALE", "u8", flat(md, "ff_mpeg4_y_dc_scale_table"), 32)
    o.flat("C_DC_SCALE", "u8", flat(md, "ff_mpeg4_c_dc_scale_table"), 32)

    # ── 系数 RL ────────────────────────────────────────────────────────────
    # 帧内：`ff_mpeg4_rl_intra`（n=103, last=67）。帧内 DC 单独走上面的 DC 表，
    # 所以这张表从**扫描序第 1 个**开始用。
    o.pairs("RL_INTRA_VLC", pairs(md, "ff_mpeg4_intra_vlc"))
    o.flat("RL_INTRA_RUN", "i8", flat(md, "ff_mpeg4_intra_run"))
    o.flat("RL_INTRA_LEVEL", "i8", flat(md, "ff_mpeg4_intra_level"))

    # 帧间：`ff_h263_rl_inter`（n=102, last=58），P 帧用。
    o.pairs("RL_INTER_VLC", pairs(hd, "ff_inter_vlc"))
    o.flat("RL_INTER_RUN", "i8", flat(hd, "ff_inter_run"))
    o.flat("RL_INTER_LEVEL", "i8", flat(hd, "ff_inter_level"))

    # ── 交替扫描 ───────────────────────────────────────────────────────────
    # `ac_pred == 1` 时不用 zig-zag，而是「按预测方向沿行/列扫」。
    o.flat("H_SCAN", "u8", flat(mvd, "ff_alternate_horizontal_scan"), 16)
    o.flat("V_SCAN", "u8", flat(mvd, "ff_alternate_vertical_scan"), 16)

    return o.text()


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
    raise SystemExit(f"源目录不对：{SRC}/libavcodec 不存在（用 `python3 tools/gen_tables.py <ffmpeg_src>`）")

write("crates/msmpeg4/src/tables.rs", gen_msmpeg4())
write("crates/mp4v/src/tables.rs", gen_mp4v())
