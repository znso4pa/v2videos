#!/usr/bin/env python3
"""把 MS-MPEG-4（MP43）需要的 VLC / 反量化表读出来，排版成 Rust 常量。

**为什么是生成而不是手抄**：这些表有几千个数字，手抄必错；生成可复现，也把出处钉在一处。

**表本身是什么**：H.263 / MPEG-4 Part 2 的 VLC 与反量化定义 —— 编解码器格式的一部分，
不是某个实现的私有数据。生成器只做「读出来 + 排版」，不做任何推导。

**许可（重要）**：目前这份生成器读的是 FFmpeg 的数据文件（**LGPL-2.1-or-later**），
所以产出的 `crates/msmpeg4/src/tables.rs` 按 LGPL-2.1-or-later 使用，
整个仓库统一按 LGPL-2.1-or-later（见 README 的「许可证」一节）。
换一个数据源（规范附录、或别的宽松许可实现）即可整体替换 —— 生成器就是为此留的；
输出文件头会记下源文件与版本，方便日后核对。

用法：python3 tools/gen_tables.py <ffmpeg_src_dir> > crates/msmpeg4/src/tables.rs
"""

import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ffsize/ffmpeg-7.1"


def read(path):
    with open(f"{SRC}/libavcodec/{path}") as f:
        return f.read()


def body(text, name):
    """取出 `NAME[...] = { ... };` 的花括号内容。"""
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
                return text[i + 1 : j]
    raise SystemExit(f"{name} 括号不配对")


def nums(s):
    return [int(x, 0) for x in re.findall(r"0x[0-9a-fA-F]+|\d+", s)]


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


def emit_pairs(f, name, data):
    f.write(f"pub static {name}: [[u32; 2]; {len(data)}] = [\n")
    for a, b in data:
        f.write(f"    [{a}, {b}],\n")
    f.write("];\n\n")


def emit_flat(f, name, ty, data):
    f.write(f"pub static {name}: [{ty}; {len(data)}] = [\n")
    for i in range(0, len(data), 16):
        f.write("    " + ", ".join(str(v) for v in data[i : i + 16]) + ",\n")
    f.write("];\n\n")


def main():
    d = read("msmpeg4data.c")
    v = read("msmpeg4_vc1_data.c")

    f = sys.stdout
    f.write(
        "//! MS-MPEG-4（MP43）解码需要的 VLC / 反量化表。\n"
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
        "//! 生成器目前读的是 FFmpeg 7.1 的 `libavcodec/msmpeg4data.c` 与\n"
        "//! `libavcodec/msmpeg4_vc1_data.c`（LGPL-2.1-or-later）—— 与本仓的许可一致，\n"
        "//! 整个仓库统一按 LGPL-2.1-or-later 使用。详见 README 的「许可证」一节。\n"
        "//!\n"
        "//! 换一个数据源（规范附录、或别的宽松许可实现）即可整体替换 —— 生成器已就位，\n"
        "//! 且表本身与实现无关。\n\n"
        "#![allow(dead_code, clippy::all)]\n\n"
    )

    # 交织的 (code, len) 表
    emit_pairs(f, "MB_NON_INTRA", pairs(d, "ff_table_mb_non_intra"))
    emit_pairs(f, "MB_INTRA", pairs(v, "ff_msmp4_mb_i_table"))
    emit_pairs(f, "INTER_INTRA", pairs(d, "ff_table_inter_intra"))

    # 系数 RL 表（vlc 是 (code,len)，run/level 是 int8）
    for tag in ("0", "1", "2", "4"):
        emit_pairs(f, f"RL{tag}_VLC", pairs(d, f"table{tag}_vlc"))
        emit_flat(f, f"RL{tag}_RUN", "i8", flat(d, f"table{tag}_run"))
        emit_flat(f, f"RL{tag}_LEVEL", "i8", flat(d, f"table{tag}_level"))

    # MPEG-4 的通用 RL 表（ff_rl_table[2] 与 [5]）—— 在 mpeg4data.h 里
    md = read("mpeg4data.h")
    emit_pairs(f, "RL_INTRA_VLC", pairs(md, "ff_mpeg4_intra_vlc"))
    emit_flat(f, "RL_INTRA_RUN", "i8", flat(md, "ff_mpeg4_intra_run"))
    emit_flat(f, "RL_INTRA_LEVEL", "i8", flat(md, "ff_mpeg4_intra_level"))

    # 帧间通用 RL 表 `ff_rl_table[5]` == `ff_h263_rl_inter`（n=102, last=58）——
    # 在 h263data.c 里。P 帧的 rl_table_index=2 会落到这张表上。
    hd = read("h263data.c")
    emit_pairs(f, "RL_INTER_VLC", pairs(hd, "ff_inter_vlc"))
    emit_flat(f, "RL_INTER_RUN", "i8", flat(hd, "ff_inter_run"))
    emit_flat(f, "RL_INTER_LEVEL", "i8", flat(hd, "ff_inter_level"))

    # MPEG-4/H.263 的水平/垂直**交替扫描**（ac_pred 为真时用，不是 zig-zag）
    mvd = read("mpegvideodata.c")
    emit_flat(f, "H_SCAN", "u8", flat(mvd, "ff_alternate_horizontal_scan"))
    emit_flat(f, "V_SCAN", "u8", flat(mvd, "ff_alternate_vertical_scan"))

    # 运动矢量表。每张表 1100 个码（1099 个真值 + 1 个 escape），
    # escape 之后明文读 6+6 位；其余码用 `mvx/mvy` 查偏移（各 1099 项）。
    emit_flat(f, "MV0_CODE", "u16", flat(d, "table0_mv_code"))
    emit_flat(f, "MV0_LEN", "u8", flat(d, "table0_mv_bits"))
    emit_flat(f, "MV1_CODE", "u16", flat(d, "table1_mv_code"))
    emit_flat(f, "MV1_LEN", "u8", flat(d, "table1_mv_bits"))
    emit_flat(f, "MV0_X", "u8", flat(d, "table0_mvx"))
    emit_flat(f, "MV0_Y", "u8", flat(d, "table0_mvy"))
    emit_flat(f, "MV1_X", "u8", flat(d, "table1_mvx"))
    emit_flat(f, "MV1_Y", "u8", flat(d, "table1_mvy"))

    # DC 表：[table_index][luma/chroma] -> 120 个 (code,len)
    dc = pairs_nested(v, "ff_msmp4_dc_tables")
    assert len(dc) == 480, f"DC 表应有 480 项，得到 {len(dc)}"
    f.write("pub static DC: [[[[u32; 2]; 120]; 2]; 2] = [\n")
    k = 0
    for ti in range(2):
        f.write("  [\n")
        for _lc in range(2):
            f.write("    [\n")
            for _ in range(120):
                a, b = dc[k]
                k += 1
                f.write(f"      [{a}, {b}],\n")
            f.write("    ],\n")
        f.write("  ],\n")
    f.write("];\n\n")

    # 扫描表
    sc = rows(v, "ff_wmv1_scantable")
    assert all(len(r) == 64 for r in sc), "扫描表每行应为 64 项"
    f.write(f"pub static SCAN: [[u8; 64]; {len(sc)}] = [\n")
    for r in sc:
        f.write("    [" + ", ".join(str(x) for x in r) + "],\n")
    f.write("];\n\n")

    # 反量化：DC 缩放表
    emit_flat(f, "Y_DC_SCALE", "u8", flat(d, "ff_wmv1_y_dc_scale_table"))
    emit_flat(f, "C_DC_SCALE", "u8", flat(d, "ff_wmv1_c_dc_scale_table"))
    emit_flat(f, "OLD_Y_DC_SCALE", "u8", flat(d, "ff_old_ff_y_dc_scale_table"))


main()
