#!/usr/bin/env python3
"""把 VC-1（WMV3）解码需要的 VLC / 量化表读出来，排版成 Rust 常量。

**为什么是生成而不是手抄**：位平面与宏块层的 VLC 有几十张、几千个码字，手抄必错，
而且错一个码字只会让**某些**素材解出噪声 —— 是那种「大多数时候对」的错。

**表本身是什么**：VC-1（SMPTE 421M）规范里的 VLC 与量化定义 —— 编解码器格式的一部分，
不是某个实现的私有数据。生成器只做「读出来 + 排版」，不做任何推导。

**为什么连枚举一起生成**：`ff_vc1_ttfrm_to_tt` 是**符号名**表（`{ TT_8X8, TT_8X4, … }`），
`ff_vc1_mv_pmode_table` 也是（`MV_PMODE_*`）。手抄这些名字的**数值**等于把「枚举顺序」
抄了一遍 —— 上游调一次顺序就全错，而且错得无声无息。所以生成器把枚举从源文件里读出来，
`tables.rs` 里的常量与 C 侧同名同值，表里的名字由生成器现场翻译（认不出的名字直接报错）。

**许可（重要）**：读的是 FFmpeg 的数据文件（**LGPL-2.1-or-later**），
所以产出的 `tables.rs` 按 LGPL-2.1-or-later 使用，
整个仓库统一按 LGPL-2.1-or-later（见 README 的「许可证」一节）。
换一个数据源（规范附录、或别的宽松许可实现）即可整体替换 —— 生成器就是为此留的；
输出文件头会记下源文件与版本，方便日后核对。

用法：
    python3 tools/gen_vc1_tables.py [ffmpeg_src_dir]

写 `crates/vc1/src/tables.rs`（路径相对仓库根，自动定位，不依赖当前工作目录）。
表来自 `libavcodec/vc1data.c` 与 `libavcodec/vc1_vlc_data.h`，枚举来自
`libavcodec/vc1.h` 与 `libavcodec/vc1_common.h`。

默认源目录 `/tmp/ffsize/ffmpeg-7.1`。
⚠️ **默认源目录 `/tmp/ffsize/ffmpeg-7.1` 是作者的本地路径**，第三方 clone 里不存在。
自己准备一份 FFmpeg 7.1 源码（`https://ffmpeg.org/download.html`），把路径作为第一个参数传进来。
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
# 前两个是**数据表**，后两个是**枚举常量**（`ff_vc1_ttfrm_to_tt` 那类表里是符号名，
# 数值要从枚举翻出来 —— 那份枚举顺序同样是上游的著作）。
VC1_SOURCES = ["vc1data.c", "vc1_vlc_data.h", "vc1.h", "vc1_common.h"]

# 要落到 Rust 的枚举。名字必须在源文件里找得到 —— 找不到就是上游改了名，
# 那时候**要有人来看一眼**，不能默默跳过。
WANTED_ENUMS = ["Profile", "QuantMode", "DQProfile", "MVModes", "TransformTypes", "Imode"]


def read(path):
    with open(f"{SRC}/libavcodec/{path}") as f:
        return f.read()


def strip_comments(s):
    """剥掉 `//` 行注释与 `/* */` 块注释。

    非剥不可：这些表的**注释里也写着数字/名字**（`// 0 1 2 … 31` 标出索引、
    `TT_8X4, // both halves` 标出语义），不剥就会把注释内容当成表项。
    """
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


def rows(text, name):
    """所有最内层 `{ ... }`，每组返回其全部数字（给 `[N][M]` 这类表）。"""
    return [nums(g) for g in re.findall(r"\{\s*([^{}]*?)\s*\}", body(text, name))]


def sym_rows(text, name):
    """所有最内层 `{ ... }`，每组返回**原样的逗号分隔 token**。

    给**符号名**表用（`MV_PMODE_*` 那种）。⚠️ 这里**绝不能**走 `nums()`：
    `MV_PMODE_1MV` 里就带个 `1`，被当成数字抠出来会得到一张「全是 1」的表 ——
    合法、能编译、无声地全错。（第一版就是这么错的。）
    """
    return [
        [t.strip() for t in g.split(",") if t.strip()]
        for g in re.findall(r"\{\s*([^{}]*?)\s*\}", body(text, name))
    ]


def sym_flat(text, name):
    """一维的符号名表：`NAME[N] = { A, B, ... };` -> ["A", "B", ...]。"""
    m = re.search(re.escape(name) + r"\s*\[[^=]*=\s*\{([^{}]*)\}", strip_comments(text))
    if not m:
        raise SystemExit(f"找不到 {name}")
    return [t.strip() for t in m.group(1).split(",") if t.strip()]


def parse_enums(text):
    """`enum [Name] { A, B = 0x10, C };` -> {Name: [(sym, val), …]}。

    只处理**扁平的**枚举（这里的几个都是）；带花括号初始化或嵌套的会在这里暴露 ——
    与其猜，不如报错。
    """
    out = {}
    for m in re.finditer(r"enum\s+(\w+)?\s*\{(.*?)\}", text, re.S):
        name = m.group(1)
        if not name:
            continue
        entries = []
        val = 0
        for part in strip_comments(m.group(2)).split(","):
            part = part.strip()
            if not part:
                continue
            if "{" in part or "(" in part:
                raise SystemExit(f"枚举 {name} 里出现了非扁平写法：{part!r}")
            if "=" in part:
                k, v = part.split("=", 1)
                val = int(v.strip(), 0)
                entries.append((k.strip(), val))
            else:
                entries.append((part, val))
            val += 1
        out[name] = entries
    return out


class Out:
    """攒文本，最后一次性写盘 —— 免得生成到一半失败留下半个文件。"""

    def __init__(self):
        self.buf = []

    def w(self, s):
        self.buf.append(s)

    def enum(self, src_file, entries):
        self.w(f"// ---- `enum`（源：`libavcodec/{src_file}`）\n\n")
        width = max(len(k) for k, _ in entries)
        for k, v in entries:
            self.w(f"pub const {k:<{width}} : u8 = {v};\n")
        self.w("\n")

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

    def matrix(self, name, ty, data):
        self.w(f"pub static {name}: [[{ty}; {len(data[0])}]; {len(data)}] = [\n")
        for row in data:
            self.w("    [" + ", ".join(str(v) for v in row) + "],\n")
        self.w("];\n\n")

    def text(self):
        return "".join(self.buf)


def header():
    """文件头：出处、许可、版权声明（版权行从源文件现读，见 `ffmpeg_src.py`）。"""
    out = Out()
    out.w(
        """//! VC-1（WMV3）解码需要的 VLC / 量化表 / 枚举常量。
//!
//! **本文件由 `tools/gen_vc1_tables.py` 生成，不要手改。**
//!
//! ## 表本身是什么
//!
//! VC-1（SMPTE 421M）规范里的 VLC、位平面编码与量化定义 —— 编解码器格式的一部分，
//! 不是某个实现的私有数据。生成器只做「读出来 + 排版」，不做任何推导。
//!
//! ## 为什么枚举也在里面
//!
//! 表里有几处是**符号名**而不是数字（`ff_vc1_ttfrm_to_tt = { TT_8X8, TT_8X4, … }`、
//! `ff_vc1_mv_pmode_table = { MV_PMODE_* }`）。生成器把枚举从源文件读出来翻译，
//! 于是「枚举顺序」也只有一份真源 —— 手抄数值等于把顺序抄了一遍。
//!
//! ## 许可
//!
//! 本文件的数据取自 FFmpeg 7.1（LGPL-2.1-or-later）的这几个文件，
//! 以及它们的版权声明（由生成器从源文件原样读出）：
//!
//! ```text
"""
    )
    for name in VC1_SOURCES:
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
//!
//! 换一个数据源（规范附录、或别的宽松许可实现）即可整体替换 —— 生成器已就位，
//! 且表本身与实现无关。

#![allow(dead_code)]

"""
    )
    return out.text()


def gen():
    data_c = read("vc1data.c")
    vlc_h = read("vc1_vlc_data.h")
    enums = {}
    for f in VC1_SOURCES:
        for k, v in parse_enums(read(f)).items():
            enums[k] = (f, v)

    sym = {k: val for _, entries in enums.values() for k, val in entries}

    def resolve(name):
        if name not in sym:
            raise SystemExit(f"表里出现认不出的符号名 {name}（上游改了枚举？）")
        return sym[name]

    out = Out()
    out.w(header())

    for want in WANTED_ENUMS:
        if want not in enums:
            raise SystemExit(f"源文件里找不到枚举 {want}")
        out.enum(enums[want][0], enums[want][1])

    # ---- 位平面：imode 与 norm2 / norm6 的码表。
    #
    # 符号**就是下标**（`VLC_INIT_STATIC_TABLE` 只吃 bits/codes，没有单独的 syms 数组），
    # 所以 `[code, len]` 的排布顺序即符号顺序。`IMODE_*` 的数值见上面的枚举。
    out.w("// ---- 位平面（bitplane）的 VLC。`[code, len]`，下标即符号。\n\n")
    for name, codes, bits in [
        ("IMODE_VLC", "vc1_imode_codes", "vc1_imode_bits"),
        ("NORM2_VLC", "vc1_norm2_codes", "vc1_norm2_bits"),
        ("NORM6_VLC", "vc1_norm6_codes", "vc1_norm6_bits"),
    ]:
        c = flat(vlc_h, codes)
        b = flat(vlc_h, bits)
        if len(c) != len(b):
            raise SystemExit(f"{name}: codes/bits 长度不一致（{len(c)} vs {len(b)}）")
        out.pairs(name, list(zip(c, b)))

    # ---- 量化。
    out.w("// ---- 量化：`pqindex` -> `pq`。三行分别是隐式 / 显式均匀 / 显式非均匀。\n\n")
    out.matrix("PQUANT", "u8", rows(data_c, "ff_vc1_pquant_table"))

    # ---- 帧级运动补偿模式。
    out.w("// ---- P 帧的帧级 MV 模式：`[lowquant][unary 值]` -> `MV_PMODE_*`。\n\n")
    for name, src in [("MV_PMODE_TABLE", "ff_vc1_mv_pmode_table"), ("MV_PMODE_TABLE2", "ff_vc1_mv_pmode_table2")]:
        out.matrix(name, "u8", [[resolve(t) for t in row] for row in sym_rows(data_c, src)])

    # ---- 变换类型：帧头的 2 bit -> `TT_*`。
    out.w("// ---- 帧头 `ttfrm`（2 bit）-> `TT_*`。\n\n")
    out.flat("TTFRM_TO_TT", "u8", [resolve(n) for n in sym_flat(data_c, "ff_vc1_ttfrm_to_tt")])

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
    raise SystemExit(f"源目录不对：{SRC}/libavcodec 不存在（用 `python3 tools/gen_vc1_tables.py <ffmpeg_src>`）")

write("crates/vc1/src/tables.rs", gen())
