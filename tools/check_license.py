#!/usr/bin/env python3
"""许可证一致性检查 —— 把「按 crate 分两种许可证」从人的记忆变成机器约束。

检查四件事：

1. `crates/*` 的 `license` 与约定一致：六个含 FFmpeg 派生码表的 crate 是
   `LGPL-2.1-or-later`，其余是 `MIT`（名单见下，与 `NOTICE` 由同一个生成器维护）。
2. 每个 crate 都被这两张名单覆盖 —— 新增 crate 时忘了定许可证会在这里红。
3. `LICENSE` 与 `LICENSE-MIT` 都存在。
4. **每一个**「本文件由 `tools/gen_*.py` 生成」的源文件都出现在 `NOTICE` 里
   （LGPL-2.1 §4(a) 要求列出上游版权声明；漏一个就是许可瑕疵，而它不影响编译）。

为什么需要这个脚本：`license` 字段写错、或者新增一张生成表却没进 NOTICE，
**编译、测试、clippy 全都不会有反应**。这类错只在法务意义上存在。

用法：

```bash
python3 tools/check_license.py
```
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 tools/gen_notice.py 的 LGPL_CRATES / MIT_CRATES 必须一致。
# 两边都列是**故意的**：这里读的是「期望值」，生成器读的是「写进 NOTICE 的值」，
# 两边同时改错才会一起骗过人。
LGPL_CRATES = ["aac", "mpeg12", "mp4v", "msmpeg4", "vc1", "wma"]
MIT_CRATES = [
    "android-encode",
    "asf",
    "avi",
    "bench",
    "capi",
    "cli",
    "container",
    "kernel",
    "mpeg12-simd",
    "mpegps",
    "mp4",
    "player",
]

LGPL = "LGPL-2.1-or-later"
MIT = "MIT"

RE_LICENSE = re.compile(r'^license\s*=\s*"([^"]+)"', re.M)
RE_INHERIT = re.compile(r"^license\.workspace\s*=\s*true", re.M)
RE_GENERATED = re.compile(r"本文件由 `tools/gen_[a-z0-9_]+\.py` 生成")

errors = []


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def workspace_license():
    """`[workspace.package]` 里的 license —— `license.workspace = true` 继承的就是它。"""
    text = read(os.path.join(ROOT, "Cargo.toml"))
    m = re.search(r"\[workspace\.package\](.*?)(?:\n\[|\Z)", text, re.S)
    if not m:
        errors.append("根 Cargo.toml 里找不到 [workspace.package]")
        return None
    lm = RE_LICENSE.search(m.group(1))
    if not lm:
        errors.append("[workspace.package] 里没有 license 字段")
        return None
    return lm.group(1)


def crate_license(name, inherited):
    path = os.path.join(ROOT, "crates", name, "Cargo.toml")
    if not os.path.exists(path):
        return None
    text = read(path)
    m = RE_LICENSE.search(text)
    if m:
        return m.group(1)
    if RE_INHERIT.search(text):
        return inherited
    return None


def check_crates():
    inherited = workspace_license()
    crates_dir = os.path.join(ROOT, "crates")
    present = sorted(d for d in os.listdir(crates_dir) if os.path.isdir(os.path.join(crates_dir, d)))

    expected = {**{c: LGPL for c in LGPL_CRATES}, **{c: MIT for c in MIT_CRATES}}

    for name in present:
        if name not in expected:
            errors.append(f"crates/{name}: 不在 LGPL/MIT 名单里 —— 新 crate 必须定许可证（改本脚本与 NOTICE 生成器）")
            continue
        got = crate_license(name, inherited)
        if got != expected[name]:
            errors.append(f"crates/{name}: license = {got!r}，应为 {expected[name]!r}")

    for name in expected:
        if name not in present:
            errors.append(f"名单里的 crates/{name} 不存在（改名或删除了？）")


def check_license_files():
    for name in ("LICENSE", "LICENSE-MIT"):
        if not os.path.exists(os.path.join(ROOT, name)):
            errors.append(f"缺 {name}")


def check_notice_covers_generated():
    """每张生成表的文件都必须出现在 NOTICE 里。"""
    notice = read(os.path.join(ROOT, "NOTICE"))
    crates_dir = os.path.join(ROOT, "crates")
    for name in sorted(os.listdir(crates_dir)):
        src = os.path.join(crates_dir, name, "src")
        if not os.path.isdir(src):
            continue
        for f in sorted(os.listdir(src)):
            if not f.endswith(".rs"):
                continue
            head = "\n".join(read(os.path.join(src, f)).splitlines()[:12])
            if not RE_GENERATED.search(head):
                continue
            rel = f"crates/{name}/src/{f}"
            if rel not in notice:
                errors.append(f"{rel}: 是生成文件，但 NOTICE 里没有它（改 tools/gen_notice.py 的 OUTPUTS 后重新生成）")


def main():
    check_crates()
    check_license_files()
    check_notice_covers_generated()

    if errors:
        print("许可证检查失败：")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"许可证检查通过：{len(LGPL_CRATES)} 个 LGPL / {len(MIT_CRATES)} 个 MIT，NOTICE 覆盖全部生成表")
    return 0


if __name__ == "__main__":
    sys.exit(main())
