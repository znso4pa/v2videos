#!/usr/bin/env python3
"""按**帧号**对齐量 PSNR —— 对抗 `ffmpeg -lavfi psnr` 的错位陷阱。

## 为什么要有这个

`ffmpeg` 的 `psnr` 滤镜按**时间戳**配对两路视频。我们的产物里
`fps` / `timebase` 是从源时间戳推出来的，参考件（比如 ffmpeg 自己编的同一段）
未必声明得一模一样 —— 于是两边差上几帧，滤镜照样「比完给个数」，只是那个数
被系统性拉低了。**差几帧就足以把「打平」读成「落后好几 dB」** ——
本仓真在这上面栽过（一次错位把结论引偏了两轮，数字记在本地 `TODO.md`）。

这个工具绕开时间戳：两边都按**帧号**取图（`-f rawvideo` 按输出顺序），
比第 i 帧 vs 第 i 帧，并用 `--scan` 把 `k = -K..K` 的偏移全扫一遍。
两条判读规矩：

- **峰值不在 `k=0`** ⇒ 没对齐，先解决对齐再看数字；
- **峰值贴在扫描窗口边界** ⇒ 真实偏移可能在窗外，**加大 `--scan`** ——
  窗内那个「极值」是边界造出来的（本仓在别处撞过同一形状的坑）。

## 用法

```bash
# 我们产出 vs 参考（默认比 100 帧、扫 ±4 帧）
python3 tools/psnr.py ours.mp4 ref.wmv --frames 147

# 换参考流（多轨容器）、加大扫描半径
python3 tools/psnr.py ours.mp4 ref.wmv --frames 147 --ref-map 0:v:1 --scan 8

# 没有 numpy 也能跑（纯 Python 慢一些，结果一样）
python3 tools/psnr.py a.mp4 b.mp4 --frames 30
```

## 口径

- 只比 **Y**（luma）—— 画质判据用它；色度两边都会被各自的编码器狠狠量化，
  拿色度 PSNR 说事容易自欺。
- 报三个数：**聚合**（所有帧的平方误差一起算，本仓的判据口径）、
  **逐帧均值**（两者差得多说明「少数极差帧主导了均值」）、**最差 5 帧**。
- 对齐后建议同时看 **k=0 与 k=±1** 的差：差得越小，说明对帧号的敏感度越低。

退出码：0 正常；1 参数/外部命令出错。
"""

import argparse
import math
import shutil
import subprocess
import sys

try:  # numpy 只用来加速；没有它结果一样（慢几倍）
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


def ffmpeg_probe(path):
    """返回 (宽, 高, 帧数) —— 帧数可能是 None（容器没写）。"""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        sys.exit(f"ffprobe 读不了 {path}：{out.stderr.strip()}")
    first = out.stdout.strip().splitlines()[0].split(",")
    w, h = int(first[0]), int(first[1])
    nb = int(first[2]) if len(first) > 2 and first[2].isdigit() else None
    return w, h, nb


def luma_frames(path, n, stream_map=None, label=""):
    """按**帧号**取 n 帧 luma（`-f rawvideo -pix_fmt gray`），返回 bytes 列表。

    `-fps_mode passthrough`：别让 ffmpeg 按声明的帧率补/丢帧 ——
    我们要的就是「第 i 个解码帧」。
    """
    cmd = ["ffmpeg", "-v", "error", "-threads", "1"]
    if stream_map:
        cmd += ["-map", stream_map]
    cmd += ["-i", path, "-fps_mode", "passthrough", "-frames:v", str(n)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "gray", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        sys.exit(f"ffmpeg 取 {label or path} 的帧失败：{r.stderr.decode()[:400]}")
    if not r.stdout:
        sys.exit(f"{label or path}：一帧都没取到（--frames 太大？）")
    return r.stdout


def psnr_of(a, b, shape):
    """聚合 PSNR（对给定的一批帧）。a/b 是 bytes，shape=(帧数, 每帧像素数)。"""
    if _np is not None:
        x = _np.frombuffer(a, dtype=_np.uint8).astype(_np.int16)
        y = _np.frombuffer(b, dtype=_np.uint8).astype(_np.int16)
        mse = float(((x - y) ** 2).mean())
    else:  # pragma: no cover - 纯 Python 兜底
        se = 0
        for pa, pb in zip(a, b):
            d = pa - pb
            se += d * d
        mse = se / (shape[0] * shape[1])
    return math.log10(255.0 * 255.0 / max(mse, 1e-12)) * 10.0


def per_frame_psnr(cur, ref, n, px):
    out = []
    for i in range(n):
        a = cur[i * px : (i + 1) * px]
        b = ref[i * px : (i + 1) * px]
        if not a or not b:
            break
        out.append(psnr_of(a, b, (1, px)))
    return out


def main():
    ap = argparse.ArgumentParser(description="按帧号对齐量 PSNR-Y（见文件头）")
    ap.add_argument("current", help="待量的产物")
    ap.add_argument("reference", help="参考（真值）")
    ap.add_argument("--frames", type=int, default=100, help="比多少帧（默认 100）")
    ap.add_argument("--scan", type=int, default=4, help="偏移扫描半径 k=-K..K（默认 4）")
    ap.add_argument("--cur-map", help="current 的 -map（默认第一路视频）")
    ap.add_argument("--ref-map", help="reference 的 -map（默认第一路视频）")
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("PATH 上没有 ffmpeg / ffprobe")

    w1, h1, _ = ffmpeg_probe(args.current)
    w2, h2, _ = ffmpeg_probe(args.reference)
    if (w1, h1) != (w2, h2):
        sys.exit(f"分辨率不同：{w1}x{h1} vs {w2}x{h2} —— 先归一化再比")
    px = w1 * h1

    n = args.frames
    keep = (n + abs(args.scan)) if args.scan > 0 else n
    cur = luma_frames(args.current, keep, args.cur_map, "current")
    ref = luma_frames(args.reference, keep, args.ref_map, "reference")
    n_cur, n_ref = len(cur) // px, len(ref) // px
    print(f"{w1}x{h1}：current {n_cur} 帧 / reference {n_ref} 帧（各取 {keep} 帧上限）")
    if min(n_cur, n_ref) < 2:
        sys.exit("取到的帧太少，没法比")

    # ---- 偏移扫描：峰值不在 0 就是没对齐，先解决对齐再看数字 ----
    if args.scan > 0:
        print(f"\n偏移扫描（current 第 i 帧 vs reference 第 i+k 帧；k=0 才对得上）：")
        best = (None, -1e9)
        for k in range(-args.scan, args.scan + 1):
            if k >= 0:
                m = min(n_cur, n_ref - k)
                a, b = cur[: m * px], ref[k * px : (k + m) * px]
            else:
                m = min(n_cur + k, n_ref)
                a, b = cur[-k * px : (-k + m) * px], ref[: m * px]
            if m < 1:
                continue
            v = psnr_of(a, b, (m, px))
            if v > best[1]:
                best = (k, v)
            print(f"  k={k:+d}  PSNR-Y = {v:6.2f} dB  (n={m})")
        if best[0] != 0:
            print(f"  ⚠️ 峰值在 k={best[0]:+d}（{best[1]:.2f} dB）—— **没对齐**，")
            print(f"     下面的 k=0 数字是错的，别拿它当判据。")
        if abs(best[0]) == args.scan:
            print(f"  ⚠️ 峰值贴在上面的扫描窗口边界（±{args.scan}）—— 真实偏移可能在窗外，")
            print(f"     加大 --scan 重扫；窗内这个「极值」是边界造出来的。")

    # ---- k=0 的正式数字 ----
    m = min(n, n_cur, n_ref)
    cur0, ref0 = cur[: m * px], ref[: m * px]
    agg = psnr_of(cur0, ref0, (m, px))
    per = per_frame_psnr(cur0, ref0, m, px)
    mean = sum(per) / len(per)
    worst = sorted(range(len(per)), key=lambda i: per[i])[:5]
    print(f"\n按帧号对齐（k=0，比 {m} 帧）：")
    print(f"  聚合 PSNR-Y = {agg:.2f} dB；逐帧均值 = {mean:.2f} dB")
    print(f"  最差 5 帧：" + "、".join(f"#{i} {per[i]:.1f}" for i in worst))
    if abs(agg - mean) > 1.0:
        print(f"  ⚠️ 聚合与逐帧均值差 {abs(agg - mean):.2f} dB —— 少数极差帧主导了聚合，逐帧看。")


if __name__ == "__main__":
    main()
