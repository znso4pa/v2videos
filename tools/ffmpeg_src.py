"""从 FFmpeg 源文件里**原样抽出**版权行 —— 三个表生成器共用。

## 为什么要抽，而不是写死

这些是**别人的著作权声明**（Bellard / Niedermayer / Shimon / Gavrilov /
Pehlivanov / The FFmpeg Project …）。写死有两个问题：

1. **会抄错**。一个字母、一个年份区间都可能错，而版权声明抄错比不写更糟 ——
   它把「谁的作品」说错了。
2. **会过期**。上游加一个作者，生成的文件不会知道。

所以生成器每次重跑都从源文件里**现读**：产出文件里的声明永远与它抄的那份数据
同源。这也是 LGPL-2.1 §4(a) 要求的「保留版权声明」最省事的做法 ——
只要重跑生成器，声明就跟着数据一起更新。

## 只认什么

文件开头第一个注释块里，形如

    copyright (c) 2001 Fabrice Bellard
    Copyright (c) 2005-2006 Oded Shimon ( ods15 ods15 dyndns org )
    msmpeg4v1 & v2 stuff by Michael Niedermayer <michaelni@gmx.at>

的行。**不认**同一块里的描述行（`AAC data`、`H.263+ support`）——
那些不是署名。没有版权行的文件（纯表文件，例如 `h263data.c`）返回空列表，
调用方照实写「该文件头没有版权行」，**不编造**。
"""

import re

# 版权行，或「…… by 某某」的署名行。
#
# ⚠️ `by` 后面必须是**大写开头**的名字：`\bby\s+\S` 会把
# `License as published by the Free Software Foundation; either` 也吃进来 ——
# 那是许可条款的正文，不是署名。`by Michael Niedermayer` 才是。
_ATTRIB = re.compile(r"(?i:copyright)|\bby\s+[A-Z]")


def copyright_lines(path):
    """读 `path` 开头第一个 `/* ... */`，返回里面原样的版权/署名行。

    返回值已剥掉注释的 ` * ` 装饰与首尾空白，顺序与源文件一致。
    读不到文件或开头不是块注释时返回 `[]`（调用方负责如实呈现）。
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(8192)
    except OSError:
        return []
    m = re.match(r"\s*/\*(.*?)\*/", head, re.S)
    if not m:
        return []
    out = []
    for line in m.group(1).splitlines():
        s = line.strip().lstrip("*").strip()
        if s and _ATTRIB.search(s):
            out.append(s)
    return out
