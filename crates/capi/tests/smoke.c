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
DECL(v2v_vdec_open);
DECL(v2v_vdec_decode);
DECL(v2v_vdec_close);
DECL(v2v_venc_open);
DECL(v2v_venc_encode);
DECL(v2v_venc_header);
DECL(v2v_venc_close);
DECL(v2v_adec_open);
DECL(v2v_adec_feed);
DECL(v2v_adec_next);
DECL(v2v_adec_close);
DECL(v2v_aenc_open);
DECL(v2v_aenc_encode);
DECL(v2v_aenc_asc);
DECL(v2v_aenc_close);
DECL(v2v_mux_open);
DECL(v2v_mux_write_video);
DECL(v2v_mux_write_audio);
DECL(v2v_mux_finish);
DECL(v2v_mux_close);

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

/* FourCC 在文件里是四个字节，按小端读成一个 u32。 */
static uint32_t fourcc(char a, char b, char c, char d) {
    return (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) | ((uint32_t)(uint8_t)c << 16) |
           ((uint32_t)(uint8_t)d << 24);
}

/* 容器里的 fourcc → ABI 的 codec_id。认不出返回 -1。 */
static int codec_of(uint32_t f) {
    if (f == fourcc('W', 'M', 'V', '3')) return V2V_CODEC_WMV3;
    if (f == fourcc('M', 'P', '4', '3')) return V2V_CODEC_MP43;
    if (f == fourcc('M', 'P', '4', '2')) return V2V_CODEC_MP42;
    if (f == fourcc('W', 'M', 'V', '1')) return V2V_CODEC_WMV1;
    if (f == fourcc('W', 'M', 'V', '2')) return V2V_CODEC_WMV2;
    if (f == fourcc('F', 'M', 'P', '4') || f == fourcc('D', 'I', 'V', 'X') ||
        f == fourcc('X', 'V', 'I', 'D'))
        return V2V_CODEC_MP4V;
    return -1;
}

/* WMA 解码：喂容器里的音频载荷、逐个取帧。
 *
 * 这一段验的是 `v2v_adec_cfg` 与 `v2v_adec_frame` 的字段顺序 —— 尤其是
 * `planes`（`const float *const *`）那种「指针层数写错就崩」的字段。
 * 不解成 WAV、不算 SNR：形状测试只问「取到了没有、声道与采样率对不对」。 */
static void check_audio_decode(const char *sample_path) {
    v2v_demux *d = p_v2v_demux_open(sample_path);
    CHECK(d != NULL, "audio: demux 打开失败: %s", p_v2v_last_error());

    uint32_t n = p_v2v_demux_stream_count(d);
    v2v_stream_info as;
    int found = 0;
    memset(&as, 0, sizeof as);
    for (uint32_t i = 0; i < n; i++) {
        v2v_stream_info si;
        memset(&si, 0xAA, sizeof si);
        V2V_INIT(si);
        CHECK(p_v2v_demux_stream(d, i, &si) == V2V_OK, "audio: 取第 %u 条流失败", i);
        if (si.kind == V2V_STREAM_AUDIO) {
            as = si;
            found = 1;
            break;
        }
    }
    if (!found) {
        printf("（样本里没有音频流，跳过 WMA 解码那段）\n");
        p_v2v_demux_close(d);
        return;
    }

    v2v_adec_cfg ac;
    memset(&ac, 0, sizeof ac);
    V2V_INIT(ac);
    ac.format_tag = as.format_tag;
    ac.channels = as.channels;
    ac.sample_rate = as.sample_rate;
    ac.avg_bytes_per_sec = as.avg_bytes_per_sec;
    ac.block_align = as.block_align;
    ac.extra = as.extra;
    ac.extra_len = as.extra_len;
    ac.drop_priming = 1; /* 与 ffmpeg 一致：丢掉编码器开头的静音 */
    v2v_adec *dec = p_v2v_adec_open(&ac);
    CHECK(dec != NULL, "audio: 建解码器失败: %s", p_v2v_last_error());

    int frames = 0;
    long long samples = 0;
    int fed = 0;
    while (fed < 200) {
        v2v_sample s;
        memset(&s, 0, sizeof s);
        V2V_INIT(s);
        int32_t rc = p_v2v_demux_next(d, &s);
        if (rc == V2V_EOS) break;
        CHECK(rc == V2V_OK, "audio: 取样本失败: %s", p_v2v_last_error());
        if (s.stream != as.number) continue;
        fed++;
        CHECK(p_v2v_adec_feed(dec, s.data, s.data_len) == V2V_OK, "audio: 喂载荷失败: %s",
              p_v2v_last_error());
        for (;;) {
            v2v_adec_frame f;
            memset(&f, 0, sizeof f);
            V2V_INIT(f);
            int32_t r = p_v2v_adec_next(dec, &f);
            if (r == V2V_EOS) break; /* 队列空了，**不是**流结束 */
            CHECK(r == V2V_OK, "audio: 取帧失败: %s", p_v2v_last_error());
            CHECK(f.planes != NULL, "audio: planes 是空指针");
            CHECK(f.planes[0] != NULL, "audio: 第一条声道是空指针");
            CHECK(f.channels == as.channels, "audio: 声道数变了（%u != %u）", f.channels,
                  as.channels);
            CHECK(f.sample_rate == as.sample_rate, "audio: 采样率变了（%u != %u）", f.sample_rate,
                  as.sample_rate);
            CHECK(f.samples > 0, "audio: 空帧");
            frames++;
            samples += (long long)f.samples;
        }
    }
    CHECK(frames > 0, "audio: 一帧都没解出来");
    p_v2v_adec_close(dec);
    p_v2v_demux_close(d);
    printf("ok  audio-decode（%d 帧 / %lld 个采样点 / %u ch @ %u Hz）\n", frames, samples,
           as.channels, as.sample_rate);
}

/* demux → 解码 → 编码 → 封装，产物写进 `out_path`（视频 + 音频双轨）。
 *
 * 这一段的价值不只是「链路能跑」：它让 C 编译器去核对
 * `v2v_stream_info` / `v2v_picture` / `v2v_venc_cfg` / `v2v_packet` /
 * `v2v_mux_video` / `v2v_aenc_cfg` / `v2v_mux_audio` 这几个结构体的**字段顺序**。
 * Rust 侧测试用的是同一份 Rust 结构体，字段顺序写反了照样全绿 ——
 * 只有从 C 填一遍才暴露。
 *
 * 只取前 `frames` 帧：这是形状测试，不是「转一个完整文件」。 */
static void check_chain(const char *sample_path, const char *out_path, int frames) {
    v2v_demux *d = p_v2v_demux_open(sample_path);
    CHECK(d != NULL, "chain: demux 打开失败: %s", p_v2v_last_error());

    uint32_t n = p_v2v_demux_stream_count(d);
    v2v_stream_info vs;
    int found = 0;
    memset(&vs, 0, sizeof vs);
    for (uint32_t i = 0; i < n; i++) {
        v2v_stream_info si;
        memset(&si, 0xAA, sizeof si);
        V2V_INIT(si);
        CHECK(p_v2v_demux_stream(d, i, &si) == V2V_OK, "chain: 取第 %u 条流失败", i);
        if (si.kind == V2V_STREAM_VIDEO) {
            vs = si;
            found = 1;
            break;
        }
    }
    CHECK(found, "chain: 没有视频流");
    int codec = codec_of(vs.fourcc);
    CHECK(codec >= 0, "chain: 认不出 fourcc 0x%08X", vs.fourcc);

    v2v_vdec *dec =
        p_v2v_vdec_open((uint32_t)codec, vs.extra, vs.extra_len, vs.width, vs.height);
    CHECK(dec != NULL, "chain: 建解码器失败: %s", p_v2v_last_error());

    v2v_venc_cfg vc;
    memset(&vc, 0, sizeof vc);
    V2V_INIT(vc);
    vc.width = vs.width;
    vc.height = vs.height;
    vc.qscale = 4;
    vc.fps_num = 25;
    vc.ac_pred = 1;
    vc.mpeg_quant = 0; /* 本仓解码器只做 quant_type = 0 */
    v2v_venc *enc = p_v2v_venc_open(V2V_CODEC_MP4V, &vc);
    CHECK(enc != NULL, "chain: 建编码器失败: %s", p_v2v_last_error());

    uint8_t dsi[512];
    int32_t dsi_len = p_v2v_venc_header(enc, dsi, (uint32_t)sizeof dsi);
    CHECK(dsi_len > 0, "chain: 取 VOL 失败: %s", p_v2v_last_error());

    /* ---- 音频：合成一段正弦，编成 AAC ----
     *
     * 故意**不**用刚解出来的 WMA：那要把解码结果攒够 1024 个采样点再喂 AAC
     * （一个 WMA 载荷吐出来的帧长不固定），在 smoke 里写环形缓冲不值得。
     * 这一段要验的是 `v2v_aenc_cfg` / `v2v_mux_audio` 的**字段顺序**（后者 15 个
     * 字段、两组语义叠在一起，是最容易写错的），以及「双轨能一起封装」——
     * 不是音画同步。 */
    uint32_t a_ch = 2;
    uint32_t a_sr = 44100;
    for (uint32_t i = 0; i < n; i++) {
        v2v_stream_info si;
        memset(&si, 0xAA, sizeof si);
        V2V_INIT(si);
        CHECK(p_v2v_demux_stream(d, i, &si) == V2V_OK, "chain: 取第 %u 条流失败", i);
        /* 只取标量：`si.extra` 的借用会被下一次 `v2v_demux_stream` 作废。 */
        if (si.kind == V2V_STREAM_AUDIO && si.channels >= 1 && si.channels <= 2 &&
            si.sample_rate >= 8000) {
            a_ch = si.channels;
            a_sr = si.sample_rate;
            break;
        }
    }

    v2v_aenc_cfg xc;
    memset(&xc, 0, sizeof xc);
    V2V_INIT(xc);
    xc.channels = a_ch;
    xc.sample_rate = a_sr;
    xc.bitrate = 128000;
    xc.ms_stereo = 1;
    xc.threads = 1;
    v2v_aenc *aenc = p_v2v_aenc_open(&xc);
    CHECK(aenc != NULL, "chain: 建 AAC 编码器失败: %s", p_v2v_last_error());

    uint8_t asc[64];
    int32_t asc_len = p_v2v_aenc_asc(aenc, asc, (uint32_t)sizeof asc);
    CHECK(asc_len > 0, "chain: 取 ASC 失败: %s", p_v2v_last_error());

    v2v_mux_video mv;
    memset(&mv, 0, sizeof mv);
    V2V_INIT(mv);
    mv.width = vs.width;
    mv.height = vs.height;
    mv.timescale = 1000; /* tick 就是毫秒，下面 duration 直接给 ms */
    mv.codec = V2V_MUX_VCODEC_MP4V;
    mv.decoder_specific_info = dsi;
    mv.dsi_len = (uint32_t)dsi_len;

    v2v_mux_audio ma;
    memset(&ma, 0, sizeof ma);
    V2V_INIT(ma);
    ma.timescale = a_sr; /* AAC 的时间基用采样率，别用毫秒 */
    ma.object_type = 0x40; /* AAC */
    ma.avg_bitrate = 128000;
    ma.asc = asc;
    ma.asc_len = (uint32_t)asc_len;
    ma.priming = V2V_AAC_FRAME_SAMPLES; /* AAC 的编码器延迟 */

    v2v_mux *mux = p_v2v_mux_open(V2V_MUX_MP4, out_path, &mv, &ma);
    CHECK(mux != NULL, "chain: 开封装器失败: %s", p_v2v_last_error());

    int got = 0;
    int64_t last_pts = -1;
    while (got < frames) {
        v2v_sample s;
        memset(&s, 0, sizeof s);
        V2V_INIT(s);
        int32_t rc = p_v2v_demux_next(d, &s);
        if (rc == V2V_EOS) break;
        CHECK(rc == V2V_OK, "chain: 取样本失败: %s", p_v2v_last_error());
        if (s.stream != vs.number) continue;

        v2v_picture pic;
        memset(&pic, 0, sizeof pic);
        V2V_INIT(pic);
        CHECK(p_v2v_vdec_decode(dec, s.data, s.data_len, &pic) == V2V_OK,
              "chain: 解码失败: %s", p_v2v_last_error());

        v2v_packet pkt;
        memset(&pkt, 0, sizeof pkt);
        V2V_INIT(pkt);
        CHECK(p_v2v_venc_encode(enc, &pic.planes, got == 0, &pkt) == V2V_OK,
              "chain: 编码失败: %s", p_v2v_last_error());

        int64_t dur = last_pts < 0 ? 40 : (s.timestamp_ms - last_pts);
        if (dur <= 0) dur = 40;
        CHECK(p_v2v_mux_write_video(mux, pkt.data, pkt.len, pkt.key_frame, s.timestamp_ms,
                                    (uint32_t)dur) == V2V_OK,
              "chain: 写封装失败: %s", p_v2v_last_error());
        last_pts = s.timestamp_ms;

        /* 一帧视频配一帧 AAC（内容无关，见上面那段说明）。 */
        float pcm[2][V2V_AAC_FRAME_SAMPLES];
        for (int i = 0; i < V2V_AAC_FRAME_SAMPLES; i++) {
            float v = 0.1f * (float)(got + 1);
            pcm[0][i] = v;
            pcm[1][i] = -v;
        }
        const float *planes[2] = {pcm[0], pcm[1]};
        v2v_packet apkt;
        memset(&apkt, 0, sizeof apkt);
        V2V_INIT(apkt);
        CHECK(p_v2v_aenc_encode(aenc, planes, V2V_AAC_FRAME_SAMPLES, &apkt) == V2V_OK,
              "chain: AAC 编码失败: %s", p_v2v_last_error());
        int64_t apts = (int64_t)got * V2V_AAC_FRAME_SAMPLES * 1000 / (int64_t)a_sr;
        CHECK(p_v2v_mux_write_audio(mux, apkt.data, apkt.len, apts, V2V_AAC_FRAME_SAMPLES) ==
                  V2V_OK,
              "chain: 写音频失败: %s", p_v2v_last_error());

        got++;
    }
    CHECK(got > 0, "chain: 一帧都没走通");

    CHECK(p_v2v_mux_finish(mux) == V2V_OK, "chain: finish 失败: %s", p_v2v_last_error());
    p_v2v_mux_close(mux);
    p_v2v_aenc_close(aenc);
    p_v2v_venc_close(enc);
    p_v2v_vdec_close(dec);
    p_v2v_demux_close(d);
    printf("ok  chain（%d 帧 %ux%u → %s）\n", got, vs.width, vs.height, out_path);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <libv2v_capi 路径> [样本路径] [输出.mp4]\n", argv[0]);
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
    LOAD(v2v_vdec_open);
    LOAD(v2v_vdec_decode);
    LOAD(v2v_vdec_close);
    LOAD(v2v_venc_open);
    LOAD(v2v_venc_encode);
    LOAD(v2v_venc_header);
    LOAD(v2v_venc_close);
    LOAD(v2v_adec_open);
    LOAD(v2v_adec_feed);
    LOAD(v2v_adec_next);
    LOAD(v2v_adec_close);
    LOAD(v2v_aenc_open);
    LOAD(v2v_aenc_encode);
    LOAD(v2v_aenc_asc);
    LOAD(v2v_aenc_close);
    LOAD(v2v_mux_open);
    LOAD(v2v_mux_write_video);
    LOAD(v2v_mux_write_audio);
    LOAD(v2v_mux_finish);
    LOAD(v2v_mux_close);

    check_version();
    check_probe();
    check_error_channel();

    if (argc >= 3) {
        check_player(argv[2]);
        check_demux(argv[2]);
        check_audio_decode(argv[2]);
        if (argc >= 4) {
            check_chain(argv[2], argv[3], 4);
        }
    } else {
        printf("（没给样本，跳过 open / 读帧 / seek / demux / 音频 / 转码那几段）\n");
    }

    dlclose(lib);
    printf("PASS\n");
    return 0;
}
