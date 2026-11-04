#!/bin/sh
# 出两个产物：min（默认 feature）与 full（全部 feature）。
#
# 为什么是两次构建：cargo 一个包只能出一个 `cdylib`，第二次编出来的会覆盖第一次的
# `libv2v_capi.*`。想同时要「瘦」和「全」两版，只能在两次之间把产物拷走改名
# （这条实测确认过，不是假设）。
#
# 用法：
#   crates/v2v-capi/build-products.sh [--out DIR] [--debug]
#
# 产物：
#   DIR/libv2v_capi_min.<ext>    解码 / 编码 / 封装 / 播放 —— 宿主默认该用这个
#   DIR/libv2v_capi_full.<ext>   再加 hw（硬件转码）与 mpeg12（MPEG-1/2）
#
# 两个产物**主版本相同**就同 ABI：符号名、返回码、结构体布局都一样，
# full 只是多出 `v2v_mpeg_*` / `v2v_ps_*` / `v2v_transcode` 那一族。
set -eu

out=dist
profile=release
while [ $# -gt 0 ]; do
    case "$1" in
        --out) out=$2; shift 2 ;;
        --debug) profile=debug; shift ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "未知参数：$1（用 --help 看用法）" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/../.."

case "$(uname -s)" in
    Darwin) ext=dylib; name=libv2v_capi.dylib ;;
    MINGW*|MSYS*|CYGWIN*) ext=dll; name=v2v_capi.dll ;;
    *) ext=so; name=libv2v_capi.so ;;
esac

target_dir=${CARGO_TARGET_DIR:-target}
if [ "$profile" = release ]; then
    profile_flag=--release
else
    profile_flag=
fi

# $1 = 产物后缀，其余 = feature 列表（可以没有）
build() {
    suffix=$1
    shift
    if [ $# -gt 0 ]; then
        # shellcheck disable=SC2086
        cargo build -p v2v-capi $profile_flag --features "$*"
    else
        # shellcheck disable=SC2086
        cargo build -p v2v-capi $profile_flag
    fi
    src="$target_dir/$profile/$name"
    if [ ! -f "$src" ]; then
        echo "编完了却没找到 $src" >&2
        exit 1
    fi
    mkdir -p "$out"
    dst="$out/libv2v_capi_$suffix.$ext"
    cp "$src" "$dst"
    echo "$dst"
}

build min
build full hw,mpeg12
