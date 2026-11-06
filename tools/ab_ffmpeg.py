#!/usr/bin/env python3
"""对着 ffmpeg 做 A/B：同输入、同 qscale、同 GOP，比**速度**与**产物质量**。

为什么需要它：`v2v bench` 只量「解码速度」与「transcode 的绝对耗时」，
**不量产物质量**，也没有 ffmpeg 那一侧的转码对照。而「我们和 ffmpeg 比到底怎么样」
这个问题，只有把「同一份源 → 两条转码链 → 各自解回来对源算 PSNR/SSIM」放在
**同一次运行、同一台机器、背靠背**里量，数字才有意义（见 `bench/README.md` 的噪声一节）。

## 口径（改之前先想清楚）

- **同输入**：两侧吃同一个源文件。输入编码不同不影响结论，因为它对两侧的解码成本
  是同一份。
- **同 qscale / 同 GOP**：`v2v convert -q Q -g G` vs `ffmpeg -c:v mpeg4 -q:v Q -bf 0 -g G`。
  `-bf 0` 是必须的 —— 本仓编码器没有 B 帧，放开 B 帧等于换了个编码器在比。
- **不比音频**：两侧都 `--no-audio` / `-an`。AAC 的质量口径（PEAQ/SNR）与视频不是
  一把尺子，混在一张表里会让人误读。
- **速度**：`reps` 次取**最快**（机器抖动只会让它变慢，不会变快）。我们的
  `--no-verify` 必须给 —— 否则计时里含一次「解回来对源算 PSNR」，那是验证不是转码。
  ffmpeg 那边钉 `-threads 1`（解码与编码都钉）：它的 VC-1 解码默认开帧级多线程，
  不钉住就会把「它用了 4 个核」算进「我们慢几倍」里。
- **质量必须按帧号配对**：`ffmpeg -lavfi psnr` 是按**时间戳**配对的，我们的 `.mp4`
  声明的 fps 与源未必一致 —— 实测同一份产物，滤镜口径 23.66 dB、按帧号 39.57 dB。
  所以这里先把三条流解成**裸 yuv**（没有时间戳），再让滤镜比。
- **质量**：产物与**源**比（不是与对方的产物比）。两侧各自对源算，谁高谁低才有意义。

## 用法

    tools/ab_ffmpeg.py /tmp/corpus/*.wmv
    tools/ab_ffmpeg.py --qscale 4 --gop 12 --reps 3 --out /tmp/ab.tsv /tmp/corpus/*.wmv

需要 PATH 上有 `ffmpeg` / `ffprobe`，以及 `--v2v` 指的 release 版 v2v（debug 的数字没有意义）。
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def timed(cmd, reps):
    """跑 `reps` 次，返回最快的一次秒数。失败抛异常并带上 stderr。"""
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if p.returncode != 0:
            raise RuntimeError(f"命令失败：{' '.join(cmd)}\n{p.stderr[-800:]}")
        best = dt if best is None else min(best, dt)
    return best


def frames_and_size(path):
    """帧数与字节数。帧数用 ffprobe 数包（`nb_frames` 在 mp4v 里常是 N/A）。"""
    size = os.path.getsize(path)
    p = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "default=nw=1:nk=1", path,
    ])
    try:
        n = int(p.stdout.strip())
    except ValueError:
        n = 0
    return n, size


def probe_size(path):
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", path])
    w, h = p.stdout.strip().split(",")[:2]
    return int(w), int(h)


def decode_to_yuv(src, dst, w, h, frames):
    """解成**裸 yuv420p**，按帧号排（raw 没有时间戳，天然按顺序）。

    ⚠️ 这一步是必须的：`ffmpeg -lavfi psnr` 是**按时间戳配对**两路视频的。
    我们的 `.mp4` 声明的 fps / timebase 是从源时间戳推出来的，与源未必一致 ——
    差几帧照样「比完给个数」，只是那个数被系统性拉低。实测同一份产物：
    滤镜口径 **23.66 dB**，按帧号对齐 **39.57 dB**（差 16 dB！）。
    所以这里先落成裸 yuv，再让 psnr/ssim 滤镜吃两个**没有时间戳**的输入。
    """
    p = run(["ffmpeg", "-v", "error", "-i", src, "-an",
             "-frames:v", str(frames),
             "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-y", dst])
    if p.returncode != 0:
        raise RuntimeError(f"解码失败：{src}\n{p.stderr[-500:]}")


def quality(a, b, w, h):
    """对**源**算 PSNR-Y / PSNR-avg / SSIM-All（两个裸 yuv，按帧号配对）。"""
    def run_filter(name):
        return run([
            "ffmpeg", "-v", "info",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-i", a,
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-i", b,
            "-lavfi", f"[0:v][1:v]{name}", "-f", "null", "-",
        ])

    psnr = run_filter("psnr")
    ssim = run_filter("ssim")
    m = re.search(r"average:([0-9.]+|inf)", psnr.stderr)
    y = re.search(r"y:([0-9.]+|inf)", psnr.stderr)
    s = re.search(r"All:([0-9.]+)", ssim.stderr)
    return (
        float("inf") if (y and y.group(1) == "inf") else (float(y.group(1)) if y else float("nan")),
        float("inf") if (m and m.group(1) == "inf") else (float(m.group(1)) if m else float("nan")),
        float(s.group(1)) if s else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser(description="v2video vs ffmpeg：速度与产物质量 A/B")
    ap.add_argument("samples", nargs="+", help="源文件（两侧吃同一份）")
    ap.add_argument("--v2v", default="./target/release/v2v", help="release 版 v2v 路径")
    ap.add_argument("--qscale", type=int, default=6, help="两侧同 qscale（默认 6）")
    ap.add_argument("--gop", type=int, default=12, help="两侧同 GOP（默认 12）")
    ap.add_argument("--reps", type=int, default=3, help="速度取几次里最快（默认 3）")
    ap.add_argument("--out", help="把结果写成 TSV 到这个路径")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"PATH 上没有 {tool}")
    if not os.access(args.v2v, os.X_OK):
        sys.exit(f"{args.v2v} 不可执行 —— 先 cargo build --release -p v2v-cli")

    rows = []
    print(f"# qscale={args.qscale} gop={args.gop} reps={args.reps}  "
          f"（两侧都 --no-audio / -an；质量对**源**算）")
    print(f"{'样本':<16} {'帧':>4} {'我们 ms':>9} {'ffmpeg ms':>10} {'倍数':>6} "
          f"{'我们 KB':>8} {'ff KB':>8} {'体积比':>7} "
          f"{'我们 PSNR-Y':>11} {'ff PSNR-Y':>10} {'我们 SSIM':>10} {'ff SSIM':>9}")

    with tempfile.TemporaryDirectory(prefix="v2v_ab_") as tmp:
        for src in args.samples:
            stem = os.path.splitext(os.path.basename(src))[0]
            ours = os.path.join(tmp, f"{stem}_ours.mp4")
            theirs = os.path.join(tmp, f"{stem}_ff.mp4")

            # 我们的：--no-audio 去掉音频；--no-verify 去掉「解回来对源算 PSNR」那一步
            ours_cmd = [args.v2v, "convert", "--mp4", "--no-audio", "--no-verify",
                        "-q", str(args.qscale), "-g", str(args.gop), "-o", tmp, src]
            # 产物名由 CLI 决定（<stem>.mp4），可能与我们想的同名 —— 先清掉再跑
            produced = os.path.join(tmp, f"{stem}.mp4")
            for f in (produced, ours):
                if os.path.exists(f):
                    os.remove(f)
            t_ours = timed(ours_cmd, args.reps)
            os.rename(produced, ours)

            # `-threads 1`：ffmpeg 的 VC-1 解码默认开帧级多线程，不钉住它，
            # 「我们慢 5 倍」里有一半是「它用了 4 个核」—— 那是比机器，不是比实现。
            ff_cmd = ["ffmpeg", "-v", "error", "-threads", "1", "-i", src, "-an",
                      "-c:v", "mpeg4", "-q:v", str(args.qscale), "-bf", "0",
                      "-g", str(args.gop), "-threads", "1", "-y", theirs]
            t_ff = timed(ff_cmd, args.reps)

            n_ours, sz_ours = frames_and_size(ours)
            n_ff, sz_ff = frames_and_size(theirs)
            if n_ours != n_ff:
                print(f"⚠️ {stem}: 帧数不一致（我们 {n_ours} / ffmpeg {n_ff}）", file=sys.stderr)

            # 按帧号对齐：三条流都解成同样帧数的裸 yuv，再让滤镜比
            n = min(n_ours, n_ff)
            w, h = probe_size(ours)
            ref = os.path.join(tmp, f"{stem}_ref.yuv")
            ours_yuv = os.path.join(tmp, f"{stem}_ours.yuv")
            ff_yuv = os.path.join(tmp, f"{stem}_ff.yuv")
            decode_to_yuv(src, ref, w, h, n)
            decode_to_yuv(ours, ours_yuv, w, h, n)
            decode_to_yuv(theirs, ff_yuv, w, h, n)
            p_y, p_avg, s_all = quality(ours_yuv, ref, w, h)
            f_y, f_avg, fs_all = quality(ff_yuv, ref, w, h)

            rows.append(dict(
                sample=stem, frames=n_ours,
                ours_ms=t_ours * 1000, ff_ms=t_ff * 1000, ratio=t_ours / t_ff,
                ours_bytes=sz_ours, ff_bytes=sz_ff, size_ratio=sz_ours / sz_ff,
                ours_psnr_y=p_y, ours_psnr_avg=p_avg, ours_ssim=s_all,
                ff_psnr_y=f_y, ff_psnr_avg=f_avg, ff_ssim=fs_all,
            ))
            print(f"{stem:<16} {n_ours:>4} {t_ours*1000:>9.0f} {t_ff*1000:>10.0f} "
                  f"{t_ours/t_ff:>5.2f}x {sz_ours/1024:>8.0f} {sz_ff/1024:>8.0f} "
                  f"{sz_ours/sz_ff:>6.2f}x {p_y:>11.2f} {f_y:>10.2f} "
                  f"{s_all:>10.4f} {fs_all:>9.4f}")

    if args.out:
        # 头部与 `bench/baseline/*.tsv` 同风格：**机器指纹 + 口径 + 负载**。
        # 少了这几行，三个月后没人说得清这堆毫秒还算不算数（bench/README.md 的由来）。
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip() or "unknown"
        load = os.getloadavg()
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# v2v-ab 1\n")
            f.write(f"# cpu\t{cpu}\n")
            f.write(f"# arch\t{os.uname().machine}\n")
            f.write(f"# cores\t{os.cpu_count()}\n")
            f.write(f"# profile\trelease\n")
            f.write(f"# tool\ttools/ab_ffmpeg.py\n")
            f.write(f"# load\t{load[0]:.2f} {load[1]:.2f} {load[2]:.2f}\n")
            f.write(f"# settings\tqscale={args.qscale} gop={args.gop} reps={args.reps} "
                    f"threads=1 both, audio=off, psnr=frame-aligned\n")
            for i, s_ in enumerate(args.samples):
                f.write(f"# sample.{i}\t{os.path.basename(s_)}\n")
            f.write("# sample\tframes\tours_ms\tff_ms\tratio\tours_bytes\tff_bytes\t"
                    "size_ratio\tours_psnr_y\tours_psnr_avg\tours_ssim\t"
                    "ff_psnr_y\tff_psnr_avg\tff_ssim\n")
            for r in rows:
                f.write(
                    f"{r['sample']}\t{r['frames']}\t{r['ours_ms']:.0f}\t{r['ff_ms']:.0f}\t"
                    f"{r['ratio']:.3f}\t{r['ours_bytes']}\t{r['ff_bytes']}\t"
                    f"{r['size_ratio']:.3f}\t{r['ours_psnr_y']:.3f}\t{r['ours_psnr_avg']:.3f}\t"
                    f"{r['ours_ssim']:.5f}\t{r['ff_psnr_y']:.3f}\t{r['ff_psnr_avg']:.3f}\t"
                    f"{r['ff_ssim']:.5f}\n"
                )
        print(f"\n写好了：{args.out}")


if __name__ == "__main__":
    main()
