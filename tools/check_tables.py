#!/usr/bin/env python3
"""用**第二个独立来源**核对 `crates/msmpeg4/src/tables.rs` 里的码表。

## 为什么要有这个

我们的表目前是从 **FFmpeg 的 C 数组**抽出来的（LGPL）。但同一批表还有另一个
**完全独立的出处**：`OxideAV/oxideav-msmpeg4`（**MIT**）—— 它的表是从微软的
`mpg4c32.dll` **干净室提取**出来的（它仓库 `build.rs` 里写了 DLL 的 SHA-256 与
每个表在 DLL 里的文件偏移）。

两个出处对上了，说明：
1. 我们表里的数字是对的（两个独立提取过程互相印证）；
2. **换数据源不会改变内容** —— 这是把仓库从 LGPL 挪回 MIT 的前提。

## 用法

```bash
# 抓 OxideAV 的表（只读网络，不写仓库）
python3 tools/check_tables.py --fetch

# 与我们的 tables.rs 逐项比
python3 tools/check_tables.py

# 指定缓存目录 / 表文件
python3 tools/check_tables.py --dir /tmp/oxtab --tables crates/msmpeg4/src/tables.rs
```

退出码：全部一致 0；有差异 1；抓取失败 2。

## 已知的列义坑

`region_05eed0.csv`（帧内 CBPCY）的列名**是历史误标**：它的 `symbol_dec` 存的是
**码长**、`bit_length` 存的是**码字**（OxideAV 自己的 `build.rs` 里也这么写）。
所以这个表要按「码字在 bit_length 列」读。别的 CSV 是正常的
`symbol_index,file_offset_hex,code_dec,code_bin,bit_length`。
"""

import argparse
import csv
import re
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/OxideAV/oxideav-msmpeg4/HEAD/tables"

# 我们的表 → OxideAV 的 CSV。（还没映射完的表列在 MISSING 里。）
# (我们的表名, CSV 文件, 码字列, 码长列, 期望项数, 码长取「本行 + N」行)
#
# 最后那个 `len_shift` 是为 `region_05eed0.csv` 准备的：它的列名历史误标
# （码字在 `bit_length` 列），而且**长度列整体错位一行** —— 第 i 项的长度写在
# 第 i+1 行的 `symbol_dec` 里（所以它有 65 行、最后一行只有长度）。
# 这不是猜的：按这个读法 64 项逐项对上，见 tools/check_tables.py 的注释。
MAPPING = [
    ("MB_NON_INTRA", "region_05eac8_mcbpcy.csv", "code_dec", "bit_length", 128, 0),
    ("MV0", "region_05bfc0_mvvlc.csv", "code_dec", "bit_length", 1100, 0),
    ("MV1", "region_0594b8_mvvlc.csv", "code_dec", "bit_length", 1100, 0),
    ("MB_INTRA", "region_05eed0.csv", "bit_length", "symbol_dec", 64, 1),
]

# 还没映射到 OxideAV 的表（要挪许可就得先把这些补齐）。
MISSING = [
    "RL0/RL1/RL2/RL4（VLC + run + level）",
    "RL_INTRA / RL_INTER（MPEG-4 通用表）",
    "DC（2×2×120）",
    "SCAN（WMV1 扫描）",
    "MV0_X/MV0_Y、MV1_X/MV1_Y（MV LUT）",
    "H_SCAN / V_SCAN（MPEG-4 交替扫描，标准表）",
    "Y_DC_SCALE / C_DC_SCALE / OLD_Y_DC_SCALE",
    "INTER_INTRA",
]


def fetch(dir_: Path, names) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for n in names:
        dst = dir_ / n
        if dst.exists():
            print(f"  已有 {n}")
            continue
        url = f"{BASE}/{n}"
        print(f"  抓 {url}")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                dst.write_bytes(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"    失败：{e}", file=sys.stderr)
            raise


def parse_static(name: str, path: Path):
    s = path.read_text()
    m = re.search(rf"pub static {name}: .*?= \[(.*?)\n\];", s, re.S)
    if not m:
        return None
    body = m.group(1)
    if body.lstrip().startswith("["):
        rows = re.findall(r"\[(.*?)\]", body, re.S)
        return [[int(x) for x in re.findall(r"-?\d+", r)] for r in rows]
    return [int(x) for x in re.findall(r"-?\d+", body)]


def main() -> int:
    ap = argparse.ArgumentParser(description="用 OxideAV（MIT）的表核对我们的 tables.rs")
    ap.add_argument("--dir", default="/tmp/v2v_ox_tables", help="CSV 缓存目录")
    ap.add_argument("--tables", default="crates/msmpeg4/src/tables.rs", help="我们的表文件")
    ap.add_argument("--fetch", action="store_true", help="先抓 CSV")
    args = ap.parse_args()

    dir_ = Path(args.dir)
    if args.fetch:
        try:
            fetch(dir_, sorted({m[1] for m in MAPPING}))
        except Exception:  # noqa: BLE001
            return 2

    tables = Path(args.tables)
    if not tables.is_file():
        print(f"找不到 {tables}", file=sys.stderr)
        return 2

    bad_total = 0
    for name, csv_name, ck, lk, expect, len_shift in MAPPING:
        p = dir_ / csv_name
        if not p.is_file():
            print(f"{name}: 缺 {csv_name}（先跑 --fetch）", file=sys.stderr)
            return 2
        rows = list(csv.DictReader(p.open()))
        if len(rows) < expect + len_shift:
            print(f"{name}: CSV 行数不足（{len(rows)}）", file=sys.stderr)
            return 2
        ours = parse_static(name, tables)
        if name.startswith("MV"):
            codes = parse_static(f"{name}_CODE", tables)
            lens = parse_static(f"{name}_LEN", tables)
            ours = list(zip(codes, lens))
        n = min(len(ours), len(rows))
        bad = [
            (i, ours[i], (int(rows[i][ck]), int(rows[i + len_shift][lk])))
            for i in range(n)
            if ours[i][0] != int(rows[i][ck])
            or ours[i][1] != int(rows[i + len_shift][lk])
        ]
        status = "✓ 完全一致" if not bad else f"✗ {len(bad)} 项不一致（例 {bad[:3]}）"
        print(f"{name}: {n}/{expect} 项  {status}")
        if len(ours) != expect:
            print(f"  ⚠ 项数不对：我们 {len(ours)}、期望 {expect}")
            bad_total += 1
        bad_total += len(bad)

    print()
    print("还没映射到 OxideAV 的表（换许可前要补齐）：")
    for m in MISSING:
        print(f"  - {m}")
    print()
    if bad_total == 0:
        print("结论：已映射的表与第二独立来源**逐项一致**。")
        return 0
    print(f"结论：有 {bad_total} 处不一致 —— 先查清楚再谈换许可。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
