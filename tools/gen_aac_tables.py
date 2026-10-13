#!/usr/bin/env python3
"""把 AAC-LC 编码需要的 Huffman 码表 / 频带表读出来，排版成 Rust 常量。

**为什么是生成而不是手抄**：光频谱码字就有 11 本 × 81 个，手抄必错，而且错一个码字
只会让某些素材解出噪声 —— 是那种「大多数时候对」的错。

**表本身是什么**：MPEG-4 Audio（AAC）规范里的 Huffman 码表与标度因子频带边界 ——
编解码器格式的一部分，不是某个实现的私有数据。生成器只做「读出来 + 排版」。

**许可**：与 `gen_tables.py` 一致 —— 读的是 FFmpeg 的数据文件（LGPL-2.1-or-later），
所以产出的 `tables.rs` 按 LGPL-2.1-or-later 使用。详见 README 的「许可证」一节。

用法：
    python3 tools/gen_aac_tables.py [ffmpeg_src_dir]

写 `crates/aac/src/tables.rs`。表来自 `libavcodec/aactab.c` 与
`libavcodec/aacenctab.h`。
"""

import os
import re
import sys

DEFAULT_SRC = "/tmp/ffsize/ffmpeg-7.1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

args = [a for a in sys.argv[1:] if not a.startswith("-")]
SRC = args[0] if args else DEFAULT_SRC


def read(path):
    with open(f"{SRC}/libavcodec/{path}") as f:
        return f.read()


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    return re.sub(r"//[^\n]*", " ", s)


def body(text, name):
    """取出 `NAME[...] = { ... };` 的花括号内容（已剥注释）。

    ⚠️ 有的表是 `static const DECLARE_ALIGNED(16, float, name)[N] = {` ——
    名字被 `DECLARE_ALIGNED` 宏包在参数里，所以要在名字后面允许 `)`。
    """
    m = re.search(
        re.escape(name) + r"\s*\)?\s*\[[^=]*=\s*\{",
        text,
    )
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
    """抽数字。十六进制与十进制都要，浮点返回 float。

    ⚠️ 十六进制里会有 `e`/`E`（`0x1ec`），不能拿 `"e" in tok` 判浮点 ——
    先判 `0x` 前缀。
    """
    out = []
    for tok in re.findall(r"-?0x[0-9a-fA-F]+|-?\d+\.?\d*(?:[eE][-+]?\d+)?", s):
        if tok.lstrip("-").startswith("0x"):
            out.append(int(tok, 0))
        elif "." in tok or "e" in tok.lower():
            out.append(float(tok))
        else:
            out.append(int(tok))
    return out


def rows(text, name):
    """所有最内层 `{ ... }`，每组返回其全部数字。"""
    return [nums(g) for g in re.findall(r"\{\s*([^{}]*?)\s*\}", body(text, name))]


class Out:
    def __init__(self):
        self.buf = []

    def w(self, s):
        self.buf.append(s)

    def ints(self, name, ty, data, per_line=12):
        self.w(f"pub static {name}: [{ty}; {len(data)}] = [\n")
        for i in range(0, len(data), per_line):
            self.w("    " + ", ".join(str(v) for v in data[i : i + per_line]) + ",\n")
        self.w("];\n\n")

    def table2(self, name, ty, data, per_line=None):
        """`[[ty; N]; M]`，`data` 是 M 个等长行。"""
        m = len(data)
        n = len(data[0])
        assert all(len(r) == n for r in data), f"{name} 各行长度不一致"
        self.w(f"pub static {name}: [[{ty}; {n}]; {m}] = [\n")
        for r in data:
            self.w("    [" + ", ".join(str(v) for v in r) + "],\n")
        self.w("];\n\n")

    def text(self):
        return "".join(self.buf)


HEADER = """//! AAC-LC 编码需要的 Huffman 码表与频带表。
//!
//! **本文件由 `tools/gen_aac_tables.py` 生成，不要手改。**
//!
//! ## 表本身是什么
//!
//! MPEG-4 Audio（AAC）规范里的 Huffman 码表与标度因子频带边界 ——
//! 编解码器格式的一部分，不是某个实现的私有数据。生成器只做「读出来 + 排版」。
//!
//! ## 许可
//!
//! 生成器读的是 FFmpeg 7.1 的 `libavcodec/aactab.c` 与 `aacenctab.h`
//! （LGPL-2.1-or-later）—— 与本仓的许可一致，整个仓库统一按 LGPL-2.1-or-later 使用。
//! 详见 README 的「许可证」一节。
//!
//! ## 与 `crates/msmpeg4` / `crates/mp4v` 的表的关系
//!
//! **没关系**。AAC 是音频，那两边是视频；码表体系完全不同，一张都不能借。
//!
//! #![allow(dead_code, clippy::all)]

"""


def gen():
    at = read("aactab.c")
    et = read("aacenctab.h")

    def block_rows(text, var):
        """`var[...] = { 子表, 子表, ... }` 里每个子表（`{...}`）的全部数字。"""
        return [nums(g) for g in re.findall(r"\{\s*([^{}]*?)\s*\}", body(text, var))]

    o = Out()
    o.w(HEADER)

    # ── 频谱 Huffman ────────────────────────────────────────────────────────
    # 频谱 Huffman：11 本码本，每本最多 289 个码字（`ff_aac_spectral_sizes`）。
    #
    # ⚠️ `ff_aac_spectral_codes` 是**指针数组**（`codes1, codes2, ...`），
    # 真正的表在 `static const uint16_t codesN[]` 里。所以要先按名字取子表。
    sizes = nums(body(at, "ff_aac_spectral_sizes"))
    o.ints("SPECTRAL_SIZE", "u16", sizes)

    def sub_tables(names, ty):
        return [nums(body(at, n)) for n in names]

    code_names = [f"codes{i}" for i in range(1, 12)]
    bit_names = [f"bits{i}" for i in range(1, 12)]
    codes = sub_tables(code_names, "u16")
    bits = sub_tables(bit_names, "u8")
    for cb, (c, b, n) in enumerate(zip(codes, bits, sizes)):
        assert len(c) >= n, f"codes{cb+1} 只有 {len(c)} 项，规格说 {n}"
        assert len(b) >= n, f"bits{cb+1} 只有 {len(b)} 项，规格说 {n}"
    maxn = max(sizes)
    o.table2(
        "SPECTRAL_CODE",
        "u16",
        [c[:n] + [0] * (maxn - n) for c, n in zip(codes, sizes)],
        per_line=None,
    )
    o.table2(
        "SPECTRAL_BITS",
        "u8",
        [b[:n] + [0] * (maxn - n) for b, n in zip(bits, sizes)],
        per_line=None,
    )

    # 码字对应的量化矢量。`64.0` 是 escape 标记（不是真的 64）。
    #
    # `ff_aac_codebook_vectors[11]` 也是指针数组，指向**已经展开好**的
    # `codebook_vectorN`。
    #
    # 维数来自 `aaccoder.c` 里 `QUANTIZE_AND_ENCODE_BAND_COST_FUNC` 的实参 ——
    # 宏签名是 `(NAME, BT_ZERO, BT_UNSIGNED, BT_PAIR, ...)`，而
    # `dim = BT_PAIR ? 2 : 4`。所以 **CB5/CB6（SPAIR）是 2 维**，不是 4 维
    # （它们的矢量表只有 162 = 81×2 项，这一点就能验证）。
    #
    # ⚠️ 每本码本的维数不同（4 或 2），所以**不能**摊平成一张等宽的表 ——
    # 那样 2 维的会被补成 4 维，索引就全错位了。这里按「每本一个 `Vec`」存：
    # Rust 侧用 `Vec<&'static [i8]>` 按码本取，索引是 `curidx * dim + j`。
    dims = [4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2]
    names = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10]
    vecs = [nums(body(at, f"codebook_vector{n}")) for n in names]
    o.w("pub static CODEBOOK_VECTORS: [&[i8]; 11] = [\n")
    for cb, v in enumerate(vecs):
        want = sizes[cb] * dims[cb]
        assert len(v) >= want, f"codebook_vector{names[cb]} 只有 {len(v)} 项，需要 {want}"
        vals = ", ".join(str(int(x)) for x in v[:want])
        o.w(f"    &[{vals}],\n")
    o.w("];\n\n")
    o.ints("CB_DIM", "u8", dims)
    o.ints("CB_VECTOR_NAMES", "u8", names)

    # 标度因子差分码表（`diff + 60`，0..120）。
    o.ints("SCALEFACTOR_CODE", "u32", nums(body(at, "ff_aac_scalefactor_code")))
    o.ints("SCALEFACTOR_BITS", "u8", nums(body(at, "ff_aac_scalefactor_bits")))

    # ── 频带边界 ────────────────────────────────────────────────────────────
    # `ff_swb_offset_1024[sampling_index]` 指向该采样率下的边界表（含末尾的 1024）。
    # 13 个采样率索引（`ff_aac_sample_rate_idx`）：96/88.2/64/48/44.1/32/24/22.05/16/12/11.025/8/7.35 kHz。
    # ⚠️ `ff_swb_offset_1024` 也是指针数组，真正的表按采样率命名
    # (`swb_offset_1024_96` 等)。13 个采样率索引只对应 7 张表（相邻的会共用，
    # 比如 96/88.2 kHz 共用、48/44.1 共用）。这里按 `ff_swb_offset_1024` 的
    # 指针顺序逐一取。
    ptr_names = re.findall(
        r"(swb_offset_1024_\d+)",
        body(at, "ff_swb_offset_1024"),
    )
    # `body()` 只给了花括号内容，但这是个「名字列表」而不是数字 —— 直接从原文抓。
    m = re.search(r"ff_swb_offset_1024\[\]\s*=\s*\{(.*?)\};", at, re.S)
    ptr_names = re.findall(r"swb_offset_1024_(\d+)", m.group(1))
    assert len(ptr_names) == 13, f"ff_swb_offset_1024 应有 13 个指针，得到 {len(ptr_names)}"
    swb = [nums(body(at, f"swb_offset_1024_{n}")) for n in ptr_names]
    # 补齐成等长（各自末尾补 0），实际长度另存一份。
    lens = [len(r) for r in swb]
    maxlen = max(lens)
    padded = [r + [0] * (maxlen - len(r)) for r in swb]
    o.ints("SWB_OFFSET_1024_LEN", "u16", lens)
    o.table2("SWB_OFFSET_1024", "u16", padded)

    # ── 短窗（EIGHT_SHORT_SEQUENCE）的频带边界 ─────────────────────────────
    # 同一套指针表套路：`ff_swb_offset_128[sampling_index]`（含末尾的 128）+
    # `ff_aac_num_swb_128[sampling_index]`。短窗一条频带只有 128 个系数里的几格，
    # 而**分组内的每个窗口共用一套 sf/码本** —— 所以编码端把「一个组的频带 b」
    # 看成「8 个窗口的这条带拼接」，位流按 带→组内窗口 的顺序写（解码端
    # `decode_spectrum_and_dequant` 就是这么读的：cfo = coef + 128·窗口）。
    m128 = re.search(r"ff_swb_offset_128\[\]\s*=\s*\{(.*?)\};", at, re.S)
    # ⚠️ 指针表的**注释**里也提到表名（"swb_offset_128_64 is a duplicate..."），
    # 不剥注释会把注释里的名字也当成指针 —— 13 个指针抓成 15 个。
    body128 = re.sub(r"/\*.*?\*/|//[^\n]*", " ", m128.group(1), flags=re.S)
    ptr128 = re.findall(r"swb_offset_128_(\d+)", body128)
    assert len(ptr128) == 13, f"ff_swb_offset_128 应有 13 个指针，得到 {len(ptr128)}"
    swb128 = [nums(body(at, f"swb_offset_128_{n}")) for n in ptr128]
    # 各行末尾必须是 128（块的系数总数），这是频带表的自检。
    for i, r in enumerate(swb128):
        assert r and r[-1] == 128, f"swb_offset_128 #{i} 末尾不是 128：{r[-3:]}"
    lens128 = [len(r) for r in swb128]
    maxlen128 = max(lens128)
    padded128 = [r + [0] * (maxlen128 - len(r)) for r in swb128]
    o.ints("NUM_SWB_128", "u8", nums(body(at, "ff_aac_num_swb_128")))
    # 自检：num_swb 与边界表长度 - 1 一致（边界表含 0 和 128）。
    nswb = nums(body(at, "ff_aac_num_swb_128"))
    for i, l in enumerate(lens128):
        assert nswb[i] == l - 1, f"num_swb_128[{i}]={nswb[i]} 与边界表长度 {l} 不符"
    o.ints("SWB_OFFSET_128_LEN", "u16", lens128)
    o.table2("SWB_OFFSET_128", "u16", padded128)

    # ── 码本参数 ────────────────────────────────────────────────────────────
    # `aac_cb_maxval`：码本能表示的量化值的上限（0..16）；
    # `aac_cb_range`：索引码本时的进制（`curidx = curidx * range + q`）。
    o.ints("CB_MAXVAL", "u8", nums(body(et, "aac_cb_maxval")))
    o.ints("CB_RANGE", "u8", nums(body(et, "aac_cb_range")))

    # ── TNS（时域噪声整形）─────────────────────────────────────────────────
    # `tns_tmp2_map_0_4`：4 位系数量化表（16 项）。编码端量化与解码端重建**用同一张**，
    # 所以必须一致 —— 它也是「为什么不能开系数压缩」的原因：开了压缩，解码端会改用
    # `tns_tmp2_map_1_4`（另一张表），两端的值就对不上了。
    tns_map = [float(x) for x in re.findall(r"-?\d+\.\d+", body(at, "tns_tmp2_map_0_4"))]
    assert len(tns_map) == 16, f"tns_tmp2_map_0_4 项数 {len(tns_map)}"
    o.w("/// `tns_tmp2_map_0_4`：TNS 的 4 位系数量化表（编码端量化 / 解码端重建共用）。\n")
    o.w("pub static TNS_COEF_4BIT: [f32; 16] = [\n")
    for v in tns_map:
        o.w(f"    {v},\n")
    o.w("];\n\n")
    o.ints("TNS_MAX_BANDS_1024", "u8", nums(body(at, "ff_tns_max_bands_1024")))
    o.ints("TNS_MIN_SFB_LONG", "u8", nums(body(et, "tns_min_sfb_long"))[:13])

    # 采样率索引 → 采样率（用于 AudioSpecificConfig 的 4 位字段）。
    o.ints(
        "SAMPLE_RATES",
        "u32",
        [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350],
    )

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
    raise SystemExit(f"源目录不对：{SRC}/libavcodec 不存在")

write("crates/aac/src/tables.rs", gen())
