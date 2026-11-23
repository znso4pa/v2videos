#!/usr/bin/env python3
"""Generate Layer III data from FFmpeg 7.1. Pass the FFmpeg source root."""
import re
import sys
from pathlib import Path

from ffmpeg_src import copyright_lines

root = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1]) / "libavcodec"
common = source / "mpegaudiodec_common.c"
dsp = source / "mpegaudiodsp_data.c"


def values(path, name):
    body = re.search(r"\b" + name + r"\[.*?=\s*\{(.*?)\};", path.read_text(), re.S)[1]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    return [int(v, 0) for v in re.findall(r"-?(?:0x[0-9a-fA-F]+|\d+)", body)]


out = [
    "//! 本文件由 `tools/gen_mp3_tables.py` 生成；请勿手改。",
    "//! 来源：FFmpeg 7.1 libavcodec/mpegaudiodec_common.c、mpegaudiodsp_data.c。",
    "//! SPDX-License-Identifier: LGPL-2.1-or-later",
]
for path in (common, dsp):
    out.extend(f"//! {line}" for line in copyright_lines(str(path)))
out.append("")


def array(name, ty, values, width=None):
    size = len(values)
    if width is None:
        out.append(f"pub const {name}: [{ty}; {size}] = [")
        for i in range(0, size, 16):
            out.append("    " + ", ".join(map(str, values[i:i + 16])) + ",")
    else:
        assert size % width == 0
        out.append(f"pub const {name}: [[{ty}; {width}]; {size // width}] = [")
        for i in range(0, size, width):
            out.append("    [" + ", ".join(map(str, values[i:i + width])) + "],")
    out.extend(["];", ""])


lengths = values(common, "mpa_hufflens")
symbols = values(common, "mpa_huffsymbols")
offset = 0
for table, last in enumerate(values(common, "mpa_huff_sizes_minus_one"), 1):
    code = 0
    out.append(f"const PAIRS_{table}: &[(u32, u32, u8)] = &[")
    for length, symbol in zip(lengths[offset:offset + last + 1], symbols[offset:offset + last + 1]):
        assert code % (1 << (32 - length)) == 0
        out.append(f"    ({code >> (32 - length)}, {length}, 0x{symbol:02x}),")
        code += 1 << (32 - length)
    assert code == 1 << 32
    out.extend(["];", ""])
    offset += last + 1
assert offset == len(lengths) == len(symbols)
out.extend(["pub const PAIRS: [&[(u32, u32, u8)]; 16] = [", "    &[],"])
out.extend(f"    PAIRS_{i}," for i in range(1, 16))
out.extend(["];", ""])
for name, c_name, width in [
    ("PAIR_MAP", "ff_mpa_huff_data", 2),
    ("QUAD_CODES", "mpa_quad_codes", 16),
    ("QUAD_BITS", "mpa_quad_bits", 16),
    ("LONG_BANDS", "ff_band_size_long", 22),
    ("SHORT_BANDS", "ff_band_size_short", 13),
    ("SLEN", "ff_slen_table", 16),
    ("PRETAB", "ff_mpa_pretab", 22),
    ("LSF_PARTITIONS", "ff_lsf_nsf_table", 4),
]:
    array(name, "u8", values(common, c_name), width)
window = values(dsp, "ff_mpa_enwindow")
assert len(window) == 257
window += [window[512 - i] * (1 if i % 64 == 0 else -1) for i in range(257, 512)]
array("SYNTHESIS_WINDOW", "i32", window)
(root / "crates/tables/src/mp3.rs").write_text("\n".join(out), encoding="utf-8")
