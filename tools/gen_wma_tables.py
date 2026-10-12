#!/usr/bin/env python3
"""从 ffmpeg 的 `wmadata.h` / `wma_freqs.c` 抽出 WMA 解码要用的表，生成 Rust 文件。

用法：

```bash
python3 tools/gen_wma_tables.py \
    --src /tmp/ffsize/ffmpeg-7.1/libavcodec \
    --out crates/wma/src/tables.rs
```

## 抽哪些、为什么

| 表 | 用途 |
|---|---|
| `ff_wma_lsp_codebook[10][16]` | LSP 指数编码（`decode_exp_lsp`） |
| `coefN_huffcodes / coefN_huffbits / levelsN`（N=0..5） | 系数 run-level 解码（`ff_wma_run_level_decode`），6 张表 |
| `ff_wma_hgain_huffcodes / bits` | 噪声编码的高带增益 VLC |
| `ff_wma_huffcodes / ff_wma_huffbits` | 指数 VLC（`use_exp_vlc=1` 的样本用；本样本用 LSP 路径） |
| `ff_wma_critical_freqs[25]` | 位分配（`bit_alloc`）与指数频带划分 |

**不抽**（运行时算得出，且更不容易抄错）：`pow_tab`（`10^((i-60)/16)`）、
噪声表、窗、指数频带表。

## 自检

生成完会对每个 Huffman 表验 **Kraft 和**（`Σ 2^-len == 1`，前缀码的必要条件）
与**项数**，不对就报错退出 —— 抽错表是最容易发生、也最难查的错。
"""

import argparse
import re
import sys
from pathlib import Path

NB_LSP_COEFS = 10


def strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def grab_array(src: str, name: str) -> list[int]:
    """抓一个 C 数组的字面量（支持多维，按行序展平）。"""
    m = re.search(rf"\b{name}\s*(\[[^=]*?)?=\s*\{{", src)
    if not m:
        raise SystemExit(f"找不到数组 {name}")
    i = m.end() - 1
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                body = src[i + 1 : j]
                break
    else:
        raise SystemExit(f"数组 {name} 没闭合")
    # ⚠️ 码字是十六进制（0x00258），得按 C 的字面量语法解析 ——
    # 用「只抓十进制」的正则会把一个十六进制数拆成两个，项数直接翻倍。
    return [int(x, 0) for x in re.findall(r"0[xX][0-9a-fA-F]+|-?\d+", body)]


def grab_floats(src: str, name: str) -> list[float]:
    m = re.search(rf"\b{name}\s*(\[[^=]*?)?=\s*\{{", src)
    if not m:
        raise SystemExit(f"找不到数组 {name}")
    i = m.end() - 1
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                body = src[i + 1 : j]
                break
    else:
        raise SystemExit(f"数组 {name} 没闭合")
    return [float(x) for x in re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", body)]


def grab_int_rows(src: str, name: str) -> list[list[int]]:
    """抓一个二维整数数组的**每一行**（行内项数可以不满 —— C 会补 0）。"""
    m = re.search(rf"\b{name}\s*(\[[^=]*?)?=\s*\{{", src)
    if not m:
        raise SystemExit(f"找不到数组 {name}")
    i = m.end() - 1
    depth = 0
    end = None
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    body = src[i + 1 : end]
    rows = []
    depth = 0
    start = None
    for j, ch in enumerate(body):
        if ch == "{":
            if depth == 0:
                start = j + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                rows.append(
                    [int(x, 0) for x in re.findall(r"0[xX][0-9a-fA-F]+|-?\d+", body[start:j])]
                )
    return rows


def grab_float_rows(src: str, name: str) -> list[list[float]]:
    """抓一个二维浮点数组的**每一行**（行内项数可以不满 —— C 会补 0）。"""
    m = re.search(rf"\b{name}\s*(\[[^=]*?)?=\s*\{{", src)
    if not m:
        raise SystemExit(f"找不到数组 {name}")
    i = m.end() - 1
    depth = 0
    end = None
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    if end is None:
        raise SystemExit(f"数组 {name} 没闭合")
    body = src[i + 1 : end]
    rows = []
    depth = 0
    start = None
    for j, ch in enumerate(body):
        if ch == "{":
            if depth == 0:
                start = j + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                rows.append(
                    [float(x) for x in re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", body[start:j])]
                )
    return rows


def kraft_sum(lens: list[int]) -> float:
    return sum(2.0 ** (-n) for n in lens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/ffsize/ffmpeg-7.1/libavcodec", help="ffmpeg libavcodec 目录")
    ap.add_argument("--out", default="crates/wma/src/tables.rs")
    args = ap.parse_args()

    src_dir = Path(args.src)
    wmadata = strip_comments((src_dir / "wmadata.h").read_text())
    wma_freqs = strip_comments((src_dir / "wma_freqs.c").read_text())

    out = []
    out.append("//! WMA 解码用的码表 —— **由 tools/gen_wma_tables.py 从 ffmpeg 的数据文件生成**。")
    out.append("//!")
    out.append("//! 别手改这个文件：改生成器再重新生成。")
    out.append("//! 出处与理由见 `tools/gen_wma_tables.py` 的文档（含 Kraft 和自检）。")
    out.append("")
    out.append("#![allow(clippy::all)]")
    out.append("")

    # ---- LSP 码本 ----
    # ⚠️ 它是 `[10][16]` 但**有的行只写了 8 个值**（C 会把剩下的补 0）——
    # 而 `decode_exp_lsp` 对 i==0 与 i>=8 只取 3 位（0..7），正好只用前 8 列。
    # 所以必须**按行**解析再补零，不能把整个字面量展平。
    lsp_rows = grab_float_rows(wmadata, "ff_wma_lsp_codebook")
    assert len(lsp_rows) == NB_LSP_COEFS, f"lsp_codebook 行数 {len(lsp_rows)}"
    lsp = []
    for row in lsp_rows:
        assert len(row) <= 16, f"lsp 行有 {len(row)} 个值"
        lsp.extend(row + [0.0] * (16 - len(row)))
    assert len(lsp) == NB_LSP_COEFS * 16
    out.append(f"/// `ff_wma_lsp_codebook[{NB_LSP_COEFS}][16]`（LSP 指数编码）。")
    out.append(f"pub static LSP_CODEBOOK: [[f32; 16]; {NB_LSP_COEFS}] = [")
    for i in range(NB_LSP_COEFS):
        row = ", ".join(f"{v:.8}" for v in lsp[i * 16 : (i + 1) * 16])
        out.append(f"    [{row}],")
    out.append("];")
    out.append("")

    # ---- 6 组系数 VLC ----
    counts = []
    for n in range(6):
        codes = grab_array(wmadata, f"coef{n}_huffcodes")
        bits = grab_array(wmadata, f"coef{n}_huffbits")
        levels = grab_array(wmadata, f"levels{n}")
        assert len(codes) == len(bits), f"coef{n}：码字数 {len(codes)} != 长度数 {len(bits)}"
        ks = kraft_sum(bits)
        if abs(ks - 1.0) > 1e-9:
            raise SystemExit(f"coef{n} 的 Kraft 和 = {ks}（应为 1）—— 抽错了")
        counts.append((n, len(bits), len(levels)))
        out.append(f"/// 系数 VLC 表 {n}：`(code, len)` {len(bits)} 项 + `levels` {len(levels)} 项。")
        out.append(f"pub static COEF{n}_CODES: [(u32, u8); {len(bits)}] = [")
        for c, b in zip(codes, bits):
            out.append(f"    ({c}, {b}),")
        out.append("];")
        out.append(f"pub static COEF{n}_LEVELS: [u16; {len(levels)}] = [")
        for i in range(0, len(levels), 12):
            out.append("    " + ", ".join(str(v) for v in levels[i : i + 12]) + ",")
        out.append("];")
        out.append("")

    # ---- hgain（噪声编码的高带增益）----
    # 它是一张 `[37][2]` 的 `(符号, 码长)` 表 —— 码字由 ffmpeg 按**规范 Huffman**
    # 现推（`ff_vlc_init_from_lengths`），所以这里也只存长度，运行时自己推。
    hgain = grab_array(wmadata, "ff_wma_hgain_hufftab")
    assert len(hgain) % 2 == 0, "hgain 表应是 (符号, 码长) 成对"
    hsym = hgain[0::2]
    hlen = hgain[1::2]
    assert len(hsym) == 37, f"hgain 项数 {len(hsym)}"
    ks = kraft_sum(hlen)
    if abs(ks - 1.0) > 1e-9:
        raise SystemExit(f"hgain 的 Kraft 和 = {ks}（应为 1）—— 抽错了")
    out.append("/// `ff_wma_hgain_hufftab[37][2]`：噪声编码的高带增益 `(符号, 码长)`。")
    out.append("///")
    out.append("/// 符号要**减去 18** 才是增益增量（ffmpeg 的 `ff_vlc_init_from_lengths` 传的 offset）。")
    out.append(f"pub static HGAIN_TAB: [(i16, u8); {len(hsym)}] = [")
    for a, b in zip(hsym, hlen):
        out.append(f"    ({a}, {b}),")
    out.append("];")
    out.append("")
    out.append("// ⚠️ 指数 VLC（`use_exp_vlc = 1` 的样本用）走的是 **AAC 的 scalefactor 表**")
    out.append("// （`ff_aac_scalefactor_code` / `_bits`），还没抽。本样本 `use_exp_vlc = 0`，")
    out.append("// 走 LSP 路径，所以先不做；要用的时候在生成器里加一段。")
    out.append("")

    # ---- 指数频带表（v2 的捷径，a < 3 时用）----
    for tag, name in [
        ("44100", "exponent_band_44100"),
        ("32000", "exponent_band_32000"),
        ("22050", "exponent_band_22050"),
    ]:
        # ⚠️ 又是「短行」：每行**首元素是数量**（代码里 `n = *table++`），后面才是频带，
        # 剩下的位置 C 补 0。所以要按行抽、补到 25 列。
        rows = grab_int_rows(wmadata, name)
        assert len(rows) == 3, f"{name} 行数 {len(rows)}"
        out.append(f"/// `{name}[3][25]`：v2 在 `frame_len_bits - 7 - k < 3` 时用的指数频带表。")
        out.append("///")
        out.append("/// **每行第 0 个元素是频带个数**（ffmpeg 里 `n = *table++`），后面才是频带宽度。")
        out.append(f"pub static EXPONENT_BAND_{tag}: [[u8; 25]; 3] = [")
        for row in rows:
            assert len(row) <= 25, f"{name} 某行有 {len(row)} 个值"
            padded = row + [0] * (25 - len(row))
            out.append("    [" + ", ".join(str(v) for v in padded) + "],")
        out.append("];")
        out.append("")

    # ---- critical freqs ----
    crit = grab_array(wma_freqs, "ff_wma_critical_freqs")
    assert len(crit) == 25, f"critical_freqs 项数 {len(crit)}"
    out.append("/// `ff_wma_critical_freqs[25]`（位分配与指数频带划分）。")
    out.append("pub static CRITICAL_FREQS: [u16; 25] = [")
    out.append("    " + ", ".join(str(v) for v in crit) + ",")
    out.append("];")
    out.append("")

    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"写出 {args.out}")
    for n, nc, nl in counts:
        print(f"  coef{n}: 码字 {nc} 项 / levels {nl} 项（Kraft 和 = 1 ✓）")
    print(f"  lsp_codebook: {len(lsp)} 项")
    print(f"  critical_freqs: {len(crit)} 项")
    return 0


if __name__ == "__main__":
    sys.exit(main())
