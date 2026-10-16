#!/usr/bin/env python3
"""量两段 PCM 的 SNR —— **先搜对齐再算**，并如实报出偏移。

## 为什么要有这个

AAC（以及任何有编码器延迟的编码器）解码出来的波形和源**对不上头**：
ffmpeg 的 AAC 输出相对源带一段**编码器延迟**（本仓量到过 +1024 这种值，
换个容器 / 封装还会变），我们那条轨的偏移是 0。
不搜对齐直接逐样本比，量出来是负数 —— 那不说明编码器差，只说明没对齐。
所以这个工具**每次都把找到的偏移打出来**（`current 的样本 i+sh ≈ reference 的样本 i`），
换素材时先看它。

对齐还有一个方向性陷阱：**只朝一个方向搜**会得到一个「贴在窗口边界」的偏移
（和视频那边 `tools/psnr.py` 的边界陷阱同源）。所以这里是**双向**搜：
粗搜 32 样本步长 → 在最优点附近逐样本细化，窗口边界上的最优点会明确警告。

## 用法

```bash
# 我们自己编的 AAC 解出来的 PCM vs 源 PCM
ffmpeg -v error -i ours.aac -f s16le -ar 44100 -ac 2 ours.pcm
python3 tools/snr.py ours.pcm clean.pcm --rate 44100 --channels 2

# 两声道分别报（想定位「只有一路坏」时用）
python3 tools/snr.py ours.pcm clean.pcm --per-channel
```

## 口径（照抄，别改）

- 输入是 **s16le 交织**（`ffmpeg -f s16le` 的格式），`--rate` / `--channels` 必须和它一致。
- 两端各留 `--margin`（默认 20000）个样本不算 —— 编码器延迟与收尾渐变段不属于编码质量。
- **不要拿 `-n` 截过的产物比**：音频截断会让两边对不上位置。
- 报的数：**全局 SNR**（所有声道所有采样点一起算）、以及逐帧（1024 点）的
  **最差 / 中位**（均值会被少数极差帧主导，逐帧看才看得见）。
- 纯 Python 也能跑；有 numpy 时快很多，结果一致。

退出码：0 正常；1 参数/读文件出错。
"""

import argparse
import math
import sys

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

FRAME = 1024  # AAC 长窗一帧的采样点数（逐帧统计用）


def load(path, channels):
    if _np is not None:
        a = _np.fromfile(path, dtype="<i2").astype(_np.float64)
        if a.size == 0:
            sys.exit(f"{path} 读不出样本")
        return a.reshape(-1, channels)
    # pragma: no cover - 纯 Python 兜底
    import array

    a = array.array("h")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if len(a) == 0:
        sys.exit(f"{path} 读不出样本")
    rows = len(a) // channels
    return [[float(a[r * channels + c]) for c in range(channels)] for r in range(rows)]


def snr_db(se, sp):
    if se <= 0:
        return float("inf")
    if sp <= 0:
        return float("-inf")
    return 10.0 * math.log10(sp / se)


def aligned(got, src, margin, max_shift):
    """搜「`got[i+sh] ≈ src[i]`」的最优 sh（双向）。返回 (sh, se, sp, n)。"""
    nf = len(src)
    ns = len(got)

    def at(sh, step):
        # 两端各留 margin；对 i 按 step 抽样
        hi = min(ns - sh, nf)
        if hi <= 2 * margin:
            return (float("inf"), 0.0, 0, 0)
        se = sp = 0.0
        cnt = 0
        for i in range(margin, hi - margin, step):
            g = got[i + sh]
            s = src[i]
            for c in range(len(s)):
                d = g[c] - s[c]
                se += d * d
                sp += s[c] * s[c]
            cnt += 1
        return (se / max(cnt, 1), se, sp, cnt)

    best = (float("inf"), 0)
    for sh in range(-max_shift, max_shift + 1, 32):
        v = at(sh, 7)[0]
        if v < best[0]:
            best = (v, sh)
    coarse = best[1]
    for sh in range(coarse - 32, coarse + 33):
        if sh < -max_shift or sh > max_shift:
            continue
        v = at(sh, 2)[0]
        if v < best[0]:
            best = (v, sh)
    sh = best[1]
    _, se, sp, cnt = at(sh, 1)
    return sh, se, sp, cnt * len(src[0])


def per_frame(got, src, sh, margin):
    """逐帧（1024 点）SNR 的 (最差, 中位)。"""
    nf, ns = len(src), len(got)
    hi = min(ns - sh, nf)
    vals = []
    i = margin
    while i + FRAME + margin < hi:
        se = sp = 0.0
        for k in range(i, i + FRAME):
            s = src[k]
            g = got[k + sh]
            for c in range(len(s)):
                d = g[c] - s[c]
                se += d * d
                sp += s[c] * s[c]
        if sp > 1.0:
            vals.append(snr_db(se, sp))
        i += FRAME
    vals.sort()
    if not vals:
        return (float("nan"), float("nan"))
    return (vals[0], vals[len(vals) // 2])


def main():
    ap = argparse.ArgumentParser(description="对齐后量 PCM 的 SNR（见文件头）")
    ap.add_argument("current", help="待量的 PCM（s16le 交织）")
    ap.add_argument("reference", help="参考 PCM（s16le 交织）")
    ap.add_argument("--rate", type=int, default=44100)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--margin", type=int, default=20000, help="两端各丢多少样本（默认 20000）")
    ap.add_argument("--max-shift", type=int, default=4096, help="对齐搜索半径（默认 4096）")
    ap.add_argument("--per-channel", action="store_true", help="再报每声道")
    args = ap.parse_args()

    got = load(args.current, args.channels)
    src = load(args.reference, args.channels)
    print(
        f"{args.rate} Hz / {args.channels} 声道：current {len(got)} 点（{len(got)/args.rate:.2f} s）"
        f" / reference {len(src)} 点（{len(src)/args.rate:.2f} s）"
    )

    sh, se, sp, n = aligned(got, src, args.margin, args.max_shift)
    at_edge = abs(sh) >= args.max_shift - 32
    print(f"\n对齐偏移：current 的样本 i+{sh} ≈ reference 的样本 i" + ("（0）" if sh == 0 else ""))
    if at_edge:
        print(f"  ⚠️ 最优点贴在搜索窗口边界（|{sh}| ≥ {args.max_shift - 32}）—— 真实偏移可能在窗外，")
        print(f"     加大 --max-shift 再量；现在的 SNR 不可信。")
    print(f"全局 SNR = {snr_db(se, sp):.2f} dB（{n} 个采样点，两端各丢 {args.margin}）")

    worst, med = per_frame(got, src, sh, args.margin)
    print(f"逐帧（{FRAME} 点）：最差 {worst:.2f} dB / 中位 {med:.2f} dB")

    if args.per_channel:
        for c in range(args.channels):
            se = sp = 0.0
            hi = min(len(got) - sh, len(src))
            for i in range(args.margin, hi - args.margin):
                s = src[i][c]
                d = got[i + sh][c] - s
                se += d * d
                sp += s * s
            print(f"  声道 {c}：{snr_db(se, sp):.2f} dB")


if __name__ == "__main__":
    main()
