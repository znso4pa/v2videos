#!/usr/bin/env python3
"""生成根目录的 `NOTICE` —— 第三方来源与版权声明。

**为什么是生成的**：里面那些版权行是**别人的著作权声明**，手抄一个字母都可能错，
而且上游会变。这里和三个表生成器用**同一个**抽取函数（`tools/ffmpeg_src.py`），
所以 `NOTICE` 里写的、和 `crates/*/src/tables.rs` 头里写的，永远是同一批字符串。

用法：

```bash
python3 tools/gen_notice.py [ffmpeg_src_dir]
```

默认源目录 `/tmp/ffsize/ffmpeg-7.1`（与三个表生成器一致）。
⚠️ **默认源目录 `/tmp/ffsize/ffmpeg-7.1` 是作者的本地路径**，第三方 clone 里不存在。
自己准备一份 FFmpeg 7.1 源码（`https://ffmpeg.org/download.html`），然后：
- 前三个脚本：把路径作为**第一个参数**传进去；
- `gen_wma_tables.py`：用 `--src <ffmpeg>/libavcodec`。

"""

import os
import sys

from ffmpeg_src import copyright_lines

DEFAULT_SRC = "/tmp/ffsize/ffmpeg-7.1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

args = [a for a in sys.argv[1:] if not a.startswith("-")]
SRC = args[0] if args else DEFAULT_SRC

FFMPEG_VERSION = "7.1"

# (生成器, 产出, [上游文件...]) —— 与各生成器里自己的来源清单一致。
OUTPUTS = [
    ("tools/gen_tables.py", "crates/msmpeg4/src/tables.rs", ["msmpeg4data.c", "msmpeg4_vc1_data.c"]),
    ("tools/gen_tables.py", "crates/mp4v/src/tables.rs", ["h263data.c", "mpeg4data.h", "mpegvideodata.c"]),
    ("tools/gen_aac_tables.py", "crates/aac/src/tables.rs", ["aactab.c", "aacenctab.h"]),
    ("tools/gen_wma_tables.py", "crates/wma/src/tables.rs", ["wmadata.h", "wma_freqs.c", "aactab.c"]),
    ("tools/gen_vc1_tables.py", "crates/vc1/src/tables.rs", ["vc1data.c", "vc1_vlc_data.h", "vc1.h", "vc1_common.h"]),
]


def main():
    out = []
    w = out.append

    w("v2video —— 第三方代码与数据来源")
    w("=" * 34)
    w("")
    w("本仓以 **LGPL-2.1-or-later** 发布（全文见 [LICENSE](LICENSE)）。")
    w("本文件列出**不是本仓原创**的部分、它们的来源、以及上游的版权声明 ——")
    w("依 LGPL-2.1 §4(a)「保留版权声明与许可声明」的要求。")
    w("")
    w("⚠️ **本文件由 `tools/gen_notice.py` 生成，不要手改。**")
    w("里面的版权行是生成器从上游源文件**原样读出**的（见 `tools/ffmpeg_src.py`），")
    w("与 `crates/*/src/tables.rs` 文件头里写的是同一批字符串。")
    w("")
    w("")
    w("一、从 FFmpeg 的数据文件读出的码表")
    w("-" * 34)
    w("")
    w("下面这些 Rust 源文件是**生成的**（文件头都标了「不要手改」），")
    w(f"数据由 `tools/` 下的生成器从 FFmpeg {FFMPEG_VERSION} 的 `libavcodec/` 读出：")
    w("")
    for gen, dst, _ in OUTPUTS:
        w(f"    {dst}")
        w(f"        ← {gen}")
    w("")
    w("这些表是 H.263 / MPEG-4 Part 2 / MPEG-4 Audio / WMA / VC-1 的**格式定义**")
    w("（编解码器规格的一部分），不是某个实现的私有数据；")
    w("生成器只做「读出来 + 排版」，不做任何推导。")
    w("换一个数据源（规范附录，或别的宽松许可实现）即可整体替换 —— 生成器就是为此留的。")
    w("")
    w("上游文件与它们的版权声明：")
    w("")

    seen = set()
    for _, _, files in OUTPUTS:
        for name in files:
            if name in seen:
                continue
            seen.add(name)
            w(f"    libavcodec/{name}")
            lines = copyright_lines(os.path.join(SRC, "libavcodec", name))
            for line in lines:
                w(f"        {line}")
            if not lines:
                w("        （该文件头没有版权行 —— 它是纯数据文件，只有 LGPL 许可块）")
            w("")

    w("")
    w("二、许可证")
    w("-" * 34)
    w("")
    w("FFmpeg 以 **GNU Lesser General Public License, version 2.1 or later** 发布。")
    w("LGPL-2.1 全文见本仓的 [LICENSE](LICENSE) —— 与本仓用的是同一份文本。")
    w("")
    w("    FFmpeg 项目主页：https://ffmpeg.org/")
    w("    源码下载：      https://ffmpeg.org/download.html")
    w(f"    本仓生成表时读的是 FFmpeg {FFMPEG_VERSION}。")
    w("")
    w("")
    w("三、其余部分")
    w("-" * 34)
    w("")
    w("除上面列出的生成码表之外，本仓的代码是自研的，没有链接任何第三方库")
    w("（`Cargo.lock` 里只有本仓自己的包）。编解码器的**算法**来自公开规范")
    w("（ITU-T H.263、ISO/IEC 14496-2、ISO/IEC 14496-3、ASF 规范），")
    w("不是从任何实现逐行翻译的。")
    w("")

    text = "\n".join(out).rstrip() + "\n"
    path = os.path.join(ROOT, "NOTICE")
    old = open(path).read() if os.path.exists(path) else None
    with open(path, "w") as f:
        f.write(text)
    state = "未变" if old == text else ("更新" if old is not None else "新建")
    print(f"{state}  NOTICE  ({len(text)} 字节, {text.count(chr(10))} 行)")


if __name__ == "__main__":
    main()
