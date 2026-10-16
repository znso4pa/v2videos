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
- **不要拿 `-n` 截过的产物比**：音频截断会让两边对不上位置。要比短一点，
  就先把**同一份**干净 PCM 切好再喂给两个编码器（两边拿到的是同一段）。
- 报的数：**全局 SNR**（所有声道所有采样点一起算）、以及逐帧（1024 点）的
  **最差 / 中位**（均值会被少数极差帧主导，逐帧看才看得见）。

有 numpy 时用向量化切片算，没有就退回纯 Python（结果一样，慢）。

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
        if a.size % channels:
            sys.exit(f"{path} 的样本数不是声道数的整数倍 —— --channels 对吗？")
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


def ssd(cur, ref, sh, lo, hi, step, ch=None):
    """`Σ(cur[i+sh] − ref[i])²` 与 `Σ ref[i]²`，i 取 lo..hi 里每隔 step 个。

    只此一处「怎么算」，两种实现（numpy 切片 / 纯 Python 循环）——
    算法不复制第二份。`ch` 给定时只看那个声道。
    """
    hi = min(hi, len(ref), len(cur) - sh)
    if hi <= lo:
        return 0.0, 0.0, 0
    if _np is not None:
        c = slice(lo + sh, hi + sh, step)
        r = slice(lo, hi, step)
        a = cur[c] if ch is None else cur[c, ch]
        b = ref[r] if ch is None else ref[r, ch]
        d = a - b
        return float((d * d).sum()), float((b * b).sum()), b.size
    se = sp = 0.0
    cnt = 0
    for i in range(lo, hi, step):
        a = cur[i + sh]
        b = ref[i]
        if ch is None:
            cols = range(len(b))
        else:
            cols = (ch,)
        for j in cols:
            d = a[j] - b[j]
            se += d * d
            sp += b[j] * b[j]
            cnt += 1
    return se, sp, cnt


def snr_db(se, sp, n):
    if n == 0:
        return float("nan")
    if se <= 0:
        return float("inf")
    if sp <= 0:
        return float("-inf")
    return 10.0 * math.log10((sp / n) / (se / n))


def aligned(cur, ref, margin, max_shift):
    """搜「`cur[i+sh] ≈ ref[i]`」的最优 sh（**双向**粗搜 → 逐样本细化）。

    返回 (sh, se, sp, n)。最优点贴在 `±max_shift` 上要当心（调用方会警告）。
    """
    def mse(sh, step):
        se, _, n = ssd(cur, ref, sh, margin, len(ref) - margin, step)
        return se / max(n, 1)

    best = (float("inf"), 0)
    for sh in range(-max_shift, max_shift + 1, 32):
        v = mse(sh, 7)
        if v < best[0]:
            best = (v, sh)
    coarse = best[1]
    for sh in range(coarse - 32, coarse + 33):
        if -max_shift <= sh <= max_shift:
            v = mse(sh, 2)
            if v < best[0]:
                best = (v, sh)
    sh = best[1]
    se, sp, n = ssd(cur, ref, sh, margin, len(ref) - margin, 1)
    return sh, se, sp, n


def per_frame(cur, ref, sh, margin):
    """逐帧（`FRAME` 点）SNR 的 (最差, 中位)。"""
    vals = []
    i = margin
    hi = min(len(ref), len(cur) - sh) - margin
    while i + FRAME < hi:
        se, sp, n = ssd(cur, ref, sh, i, i + FRAME, 1)
        if sp > 1.0:
            vals.append(snr_db(se, sp, n))
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

    cur = load(args.current, args.channels)
    ref = load(args.reference, args.channels)
    print(
        f"{args.rate} Hz / {args.channels} 声道：current {len(cur)} 点（{len(cur)/args.rate:.2f} s）"
        f" / reference {len(ref)} 点（{len(ref)/args.rate:.2f} s）"
    )

    sh, se, sp, n = aligned(cur, ref, args.margin, args.max_shift)
    print(f"\n对齐偏移：current 的样本 i+{sh} ≈ reference 的样本 i")
    if abs(sh) >= args.max_shift - 32:
        print(f"  ⚠️ 最优点贴在搜索窗口边界（|{sh}| ≥ {args.max_shift - 32}）—— 真实偏移可能在窗外，")
        print(f"     加大 --max-shift 再量；现在的 SNR 不可信。")
    if n == 0:
        sys.exit("可比区间是空的 —— 文件太短或 --margin 太大")
    print(f"全局 SNR = {snr_db(se, sp, n):.2f} dB（{n} 个采样点，两端各丢 {args.margin}）")

    worst, med = per_frame(cur, ref, sh, args.margin)
    print(f"逐帧（{FRAME} 点）：最差 {worst:.2f} dB / 中位 {med:.2f} dB")

    if args.per_channel:
        for c in range(args.channels):
            se, sp, n = ssd(cur, ref, sh, args.margin, len(ref) - args.margin, 1, ch=c)
            print(f"  声道 {c}：{snr_db(se, sp, n):.2f} dB")


if __name__ == "__main__":
    main()
