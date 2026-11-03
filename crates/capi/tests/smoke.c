/*
 * 纯 C 的 ABI 冒烟：dlopen 打开 libv2v_capi，按 v2v.h 里写的那套约定走一遍。
 *
 * 为什么值得单独写：Rust 集成测试调的是 `v2v_capi::` 里的函数，**绕过了头文件** ——
 * 头文件把参数写反了它照样绿。只有 C 编译器能回答「头文件自洽吗」，
 * 而 dlopen 顺带回答了「符号真的在 .so 里吗、签名真的是这个吗」。
 *
 * 用法：
 *     smoke <libv2v_capi 的路径> [样本路径]
 *
 *   给了样本就多走一段 open → info → 读帧 → seek → close；
 *   没给就只验「不需要文件」的那几条：版本、probe、错误通道、size 契约。
 *
 * 退出码：0 全过；否则是第一个失败点的行号（便于定位）。
 * 每过一条打一行 `ok  <名字>`，失败打 `FAIL 第 N 行: ...`。
 */

#if !defined(_POSIX_C_SOURCE)
/* -std=c11 会打开 __STRICT_ANSI__，glibc 默认就不声明 dlopen 了。
 * 这个宏必须在任何 #include 之前定义。 */
#define _POSIX_C_SOURCE 200809L
#endif

#include <dlfcn.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "v2v.h"

static void *lib;

static void fail(int line, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "FAIL 第 %d 行: ", line);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(line);
}

#define CHECK(cond, ...) \
    do { \
        if (!(cond)) fail(__LINE__, __VA_ARGS__); \
    } while (0)

static void *need(const char *name) {
    void *p = dlsym(lib, name);
    if (!p) {
        fprintf(stderr, "FAIL dlsym(%s): %s\n", name, dlerror());
        exit(2);
    }
    return p;
}

/* 函数指针直接从**头文件里的声明**取类型 —— 于是「头文件写错了」会在这里
 * 编不过，而不是等到运行期参数错位才发现。 */
#define DECL(fn) static __typeof__(fn) *p_##fn
#define LOAD(fn) \
    do { \
        void *s_ = need(#fn); \
        memcpy(&p_##fn, &s_, sizeof(p_##fn)); \
    } while (0)

DECL(v2v_abi_version);
DECL(v2v_last_error);
DECL(v2v_last_error_kind);
DECL(v2v_probe);
DECL(v2v_player_open);
DECL(v2v_player_info);
DECL(v2v_player_read_video);
DECL(v2v_player_seek_ms);
DECL(v2v_player_position_ms);
DECL(v2v_player_close);
DECL(v2v_demux_open);
DECL(v2v_demux_stream_count);
DECL(v2v_demux_stream);
DECL(v2v_demux_next);
DECL(v2v_demux_close);

/* ---- 不需要样本的那几条 ---- */

static void check_version(void) {
    uint32_t v = p_v2v_abi_version();
    uint32_t major = v >> 16;
    uint32_t minor = v & 0xFFFFu;
    CHECK(major == V2V_ABI_MAJOR, "库的主版本 %u != 头文件的 %u", major, V2V_ABI_MAJOR);
    printf("ok  abi_version = %u.%u\n", major, minor);
}

static void check_probe(void) {
    /* ASF 头 GUID 的前 16 字节 */
    static const uint8_t asf[16] = {0x30, 0x26, 0xB2, 0x75, 0x8E, 0x66, 0xCF, 0x11,
                                    0xA6, 0xD9, 0x00, 0xAA, 0x00, 0x62, 0xCE, 0x6C};
    static const uint8_t ps[4] = {0x00, 0x00, 0x01, 0xBA};
    static const uint8_t junk[8] = {0};

    CHECK(p_v2v_probe(asf, (uint32_t)sizeof asf) == V2V_CONTAINER_ASF, "ASF 头没认出来");
    CHECK(p_v2v_probe(ps, (uint32_t)sizeof ps) == V2V_CONTAINER_MPEG_PS, "PS 起始码没认出来");
    CHECK(p_v2v_probe(junk, (uint32_t)sizeof junk) == V2V_CONTAINER_UNKNOWN, "垃圾字节不该认成容器");
    CHECK(p_v2v_probe(NULL, 0) == V2V_CONTAINER_UNKNOWN, "空指针不该崩，该返回 UNKNOWN");
    printf("ok  probe\n");
}

static void check_error_channel(void) {
    CHECK(p_v2v_player_open(NULL) == NULL, "空路径不该打开成功");
    const char *msg = p_v2v_last_error();
    CHECK(msg != NULL, "失败时 last_error 不该是空指针");
    CHECK(msg[0] != '\0', "失败时 last_error 不该是空串");
    CHECK(p_v2v_last_error_kind() == V2V_KIND_BAD, "空指针该算 KIND_BAD，实际 %d",
          p_v2v_last_error_kind());
    printf("ok  错误通道（%s）\n", msg);

    /* 空句柄上的每个入口都必须是 V2V_ERR_ARG，不是段错误。 */
    v2v_info info;
    memset(&info, 0, sizeof info);
    V2V_INIT(info);
    CHECK(p_v2v_player_info(NULL, &info) == V2V_ERR_ARG, "空句柄该返回 V2V_ERR_ARG");
    CHECK(p_v2v_demux_stream_count(NULL) == 0, "空句柄的流数该是 0");
}

/* ---- 需要样本的那几条 ---- */

static void check_player(const char *path) {
    v2v_player *pl = p_v2v_player_open(path);
    CHECK(pl != NULL, "打开 %s 失败: %s", path, p_v2v_last_error());

    v2v_info info;
    memset(&info, 0xAA, sizeof info);
    V2V_INIT(info);
    CHECK(p_v2v_player_info(pl, &info) == V2V_OK, "info 失败: %s", p_v2v_last_error());
    CHECK(info.width > 0 && info.height > 0, "宽高是 0");
    CHECK(info.video_fourcc != 0, "video_fourcc 是 0");

    /* size 契约：填 0 / 填太小必须报错，不能「返回 OK 却没写」。 */
    v2v_info bad;
    memset(&bad, 0xAA, sizeof bad);
    bad.size = 0;
    CHECK(p_v2v_player_info(pl, &bad) == V2V_ERR_ARG, "size = 0 必须返回 V2V_ERR_ARG");
    bad.size = 2;
    CHECK(p_v2v_player_info(pl, &bad) == V2V_ERR_ARG, "size = 2 必须返回 V2V_ERR_ARG");

    /* 截断写：只给到 has_audio 之前，后面的字节一个都不许碰。 */
    {
        unsigned char buf[sizeof(v2v_info)];
        size_t keep = offsetof(v2v_info, has_audio);
        memset(buf, 0xAA, sizeof buf);
        *(uint32_t *)buf = (uint32_t)keep;
        CHECK(p_v2v_player_info(pl, (v2v_info *)(void *)buf) == V2V_OK, "截断写该成功");
        for (size_t i = keep; i < sizeof buf; i++) {
            CHECK(buf[i] == 0xAA, "第 %zu 字节被越界写了", i);
        }
    }

    int got = 0;
    long long last = -9223372036854775807LL - 1;
    for (int i = 0; i < 10; i++) {
        v2v_frame f;
        memset(&f, 0xAA, sizeof f);
        V2V_INIT(f);
        int32_t rc = p_v2v_player_read_video(pl, &f);
        if (rc == V2V_EOS) break;
        CHECK(rc == V2V_OK, "读视频失败: %s", p_v2v_last_error());
        CHECK(f.planes.y != NULL && f.planes.u != NULL && f.planes.v != NULL, "平面是空指针");
        CHECK(f.planes.stride_y == f.planes.width, "stride_y(%u) != width(%u)",
              f.planes.stride_y, f.planes.width);
        CHECK(f.planes.stride_u == (f.planes.width + 1) / 2, "stride_u 不是 ceil(width/2)");
        CHECK((long long)f.pts_ms >= last, "时间戳回退：%lld < %lld", (long long)f.pts_ms, last);
        last = (long long)f.pts_ms;
        got++;
    }
    CHECK(got > 0, "一帧都没读到");

    int64_t landed = p_v2v_player_seek_ms(pl, 0);
    CHECK(landed >= 0, "seek 到 0 失败");
    v2v_frame after;
    memset(&after, 0xAA, sizeof after);
    V2V_INIT(after);
    CHECK(p_v2v_player_read_video(pl, &after) == V2V_OK, "seek 之后读不到帧: %s",
          p_v2v_last_error());
    CHECK((long long)after.pts_ms >= (long long)landed, "落点之后的帧比落点还早");
    CHECK(p_v2v_player_position_ms(pl) >= 0, "position_ms 是负的");

    p_v2v_player_close(pl);
    printf("ok  player（%d 帧，%ux%u）\n", got, info.width, info.height);
}

static void check_demux(const char *path) {
    v2v_demux *d = p_v2v_demux_open(path);
    CHECK(d != NULL, "demux 打开失败: %s", p_v2v_last_error());

    uint32_t n = p_v2v_demux_stream_count(d);
    CHECK(n > 0, "一条流都没有");
    int video = 0;
    for (uint32_t i = 0; i < n; i++) {
        v2v_stream_info si;
        memset(&si, 0xAA, sizeof si);
        V2V_INIT(si);
        CHECK(p_v2v_demux_stream(d, i, &si) == V2V_OK, "取第 %u 条流失败: %s", i,
              p_v2v_last_error());
        if (si.kind == V2V_STREAM_VIDEO) {
            video++;
            CHECK(si.width > 0 && si.height > 0, "视频流宽高是 0");
        }
    }
    CHECK(video > 0, "没有视频流");

    int got = 0;
    for (int i = 0; i < 20; i++) {
        v2v_sample s;
        memset(&s, 0xAA, sizeof s);
        V2V_INIT(s);
        int32_t rc = p_v2v_demux_next(d, &s);
        if (rc == V2V_EOS) break;
        CHECK(rc == V2V_OK, "取样本失败: %s", p_v2v_last_error());
        CHECK(s.data != NULL && s.data_len > 0, "样本是空的");
        got++;
    }
    CHECK(got > 0, "一个样本都没取到");

    p_v2v_demux_close(d);
    printf("ok  demux（%u 条流 / %d 个样本）\n", n, got);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <libv2v_capi 路径> [样本路径]\n", argv[0]);
        return 1;
    }

    lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen(%s) 失败: %s\n", argv[1], dlerror());
        return 1;
    }

    LOAD(v2v_abi_version);
    LOAD(v2v_last_error);
    LOAD(v2v_last_error_kind);
    LOAD(v2v_probe);
    LOAD(v2v_player_open);
    LOAD(v2v_player_info);
    LOAD(v2v_player_read_video);
    LOAD(v2v_player_seek_ms);
    LOAD(v2v_player_position_ms);
    LOAD(v2v_player_close);
    LOAD(v2v_demux_open);
    LOAD(v2v_demux_stream_count);
    LOAD(v2v_demux_stream);
    LOAD(v2v_demux_next);
    LOAD(v2v_demux_close);

    check_version();
    check_probe();
    check_error_channel();

    if (argc >= 3) {
        check_player(argv[2]);
        check_demux(argv[2]);
    } else {
        printf("（没给样本，跳过 open / 读帧 / seek / demux 那几段）\n");
    }

    dlclose(lib);
    printf("PASS\n");
    return 0;
}
