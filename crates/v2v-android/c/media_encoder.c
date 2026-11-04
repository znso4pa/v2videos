/*
 * media_encoder.c —— Android 硬件编码 shim：MediaCodec（H.264）+ MediaMuxer（MP4）。
 *
 * 这是整个 v2video 仓库里 **唯一碰 Android NDK 的 C 文件**（unsafe 的另一半）。
 * Rust 侧（src/ffi.rs）只声明下面这几个 `ame_*` 函数；所有 NDK 句柄、
 * 缓冲区生命周期、错误处理都收在这一个文件里，不外漏。
 *
 * 数据面（零拷贝做不到，但不做多余拷贝）：
 *   Rust（VC-1 解码出的 YUV420P 三平面）
 *     → ame_write_video 拷进 MediaCodec 的 input buffer（一次，整帧）
 *     → 硬件 H.264 编码
 *     → drain 里把 output buffer 的下标直接交给 AMediaMuxer_writeSampleData
 *       （MediaMuxer 按下标读 codec 的缓冲，不用我们再拷出来）
 *
 * 返回值约定：0 = 成功；-1 = 失败（ame_last_error 拿原因）。
 */
#include "media_encoder.h"

#include <fcntl.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>
#include <media/NdkMediaMuxer.h>

/* COLOR_FormatYUV420Planar —— 编码器最广为接受的平面格式（严格行对齐 = width）。
 * 新平台更愿意用 Flexible（0x7F888888），但那要按各厂商的 stride 摆平面，
 * 先用 19 这种「字节含义明确」的；遇到不吃它的设备再在 Rust 侧加参数。 */
#define COLOR_FMT_YUV420_PLANAR 19

#define MIME_AVC "video/avc"
#define MIME_AAC "audio/mp4a-latm"

struct ame_ctx {
    AMediaCodec *codec;
    AMediaMuxer *muxer;
    int fd;

    int w, h;
    int y_size, uv_size; /* w*h 与 w*h/4 */
    int has_audio;       /* 打开时声明的音频轨：muxer 要等音频 track 加完才 start */

    ssize_t video_track; /* muxer 里的下标，-1 = 还没加 */
    ssize_t audio_track;
    int muxer_started;

    uint8_t *csd; /* codec config（SPS/PPS），加 track 时塞进格式 */
    size_t csd_len;

    int eos_seen;
    char err[256];
};

static void set_err(struct ame_ctx *c, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
static void set_err(struct ame_ctx *c, const char *fmt, ...) {
    if (!c) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(c->err, sizeof(c->err), fmt, ap);
    va_end(ap);
}

const char *ame_last_error(struct ame_ctx *c) {
    return c ? c->err : "null ctx";
}

static media_status_t try_start_muxer(struct ame_ctx *c) {
    if (c->muxer_started || c->video_track < 0) return AMEDIA_OK;
    /* 声明过音频轨的，必须等音频 track 也加完 —— MediaMuxer 不允许 start 之后再加。 */
    if (c->has_audio && c->audio_track < 0) return AMEDIA_OK;
    media_status_t st = AMediaMuxer_start(c->muxer);
    if (st != AMEDIA_OK) {
        set_err(c, "AMediaMuxer_start: %d", (int)st);
        return st;
    }
    c->muxer_started = 1;
    return AMEDIA_OK;
}

/* 把 codec 的输出吸干（编码结果交 MediaMuxer、收 csd、认 EOS）。
 * 非阻塞（timeout 0）：调用方每次喂完一帧调一次，公平轮转。 */
static int drain(struct ame_ctx *c) {
    for (;;) {
        AMediaCodecBufferInfo info;
        memset(&info, 0, sizeof(info));
        ssize_t idx =
            AMediaCodec_dequeueOutputBuffer(c->codec, &info, 0 /* 不等 */);
        if (idx == AMEDIACODEC_INFO_TRY_AGAIN_LATER) return 0;
        if (idx == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
            /* 有些设备在这里才给 csd（格式变化时机因厂商而异）。 */
            if (c->video_track < 0) {
                AMediaFormat *fmt = AMediaCodec_getOutputFormat(c->codec);
                if (!fmt) {
                    set_err(c, "getOutputFormat 返回空");
                    return -1;
                }
                if (!c->csd) {
                    void *p = NULL;
                    size_t n = 0;
                    if (AMediaFormat_getBuffer(fmt, "csd-0", &p, &n) && p && n) {
                        c->csd = malloc(n);
                        memcpy(c->csd, p, n);
                        c->csd_len = n;
                    }
                }
                AMediaFormat_setString(fmt, "mime", MIME_AVC);
                if (c->csd)
                    AMediaFormat_setBuffer(fmt, "csd-0", c->csd, c->csd_len);
                c->video_track = AMediaMuxer_addTrack(c->muxer, fmt);
                AMediaFormat_delete(fmt);
                if (c->video_track < 0) {
                    set_err(c, "addTrack(视频) 失败");
                    return -1;
                }
                if (try_start_muxer(c) != AMEDIA_OK) return -1;
            }
            continue;
        }
        if (idx < 0) {
            set_err(c, "dequeueOutputBuffer: %zd", idx);
            return -1;
        }

        size_t cap = 0;
        uint8_t *buf = AMediaCodec_getOutputBuffer(c->codec, (size_t)idx, &cap);
        if (!buf || (size_t)info.size > cap) {
            set_err(c, "output buffer 读不出（cap %zu < size %d）", cap, info.size);
            return -1;
        }

        if (info.flags & AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG) {
            /* SPS/PPS：先存着，加 track 时用；不进 MP4 的样本流。 */
            free(c->csd);
            c->csd = malloc((size_t)info.size);
            memcpy(c->csd, buf + info.offset, (size_t)info.size);
            c->csd_len = (size_t)info.size;
            AMediaCodec_releaseOutputBuffer(c->codec, (size_t)idx, 0);
            continue;
        }

        if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
            c->eos_seen = 1;
            AMediaCodec_releaseOutputBuffer(c->codec, (size_t)idx, 0);
            return 0;
        }

        if (info.size > 0) {
            /* NDK 的 writeSampleData 收**指针**（不是 Java 版的缓冲下标）——
             * 所以必须在 release 之前把数据交出去。 */
            if (try_start_muxer(c) != AMEDIA_OK) return -1;
            if (c->muxer_started) {
                media_status_t st = AMediaMuxer_writeSampleData(
                    c->muxer, (size_t)c->video_track, buf + info.offset, &info);
                if (st != AMEDIA_OK) {
                    AMediaCodec_releaseOutputBuffer(c->codec, (size_t)idx, 0);
                    set_err(c, "writeSampleData(视频): %d", (int)st);
                    return -1;
                }
            }
            /* muxer 还没 start（csd 没到）时这帧只能丢：丢帧是时间轴上的洞，
             * 阻塞是整个编码卡死 —— 实测正常设备 csd 在第一帧前就到了。 */
        }
        AMediaCodec_releaseOutputBuffer(c->codec, (size_t)idx, 0);
    }
}

struct ame_ctx *ame_open(const char *path, int w, int h, int bitrate, int fps,
                        int i_frame_interval_s, int has_audio) {
    struct ame_ctx *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->video_track = -1;
    c->audio_track = -1;
    c->w = w;
    c->h = h;
    c->has_audio = has_audio;
    c->y_size = w * h;
    c->uv_size = w * h / 4;

    c->fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (c->fd < 0) {
        set_err(c, "open(%s) 失败", path);
        ame_free(c);
        return NULL;
    }
    c->muxer = AMediaMuxer_new(c->fd, AMEDIAMUXER_OUTPUT_FORMAT_MPEG_4);
    if (!c->muxer) {
        set_err(c, "AMediaMuxer_new 失败");
        ame_free(c);
        return NULL;
    }

    c->codec = AMediaCodec_createEncoderByType(MIME_AVC);
    if (!c->codec) {
        set_err(c, "设备上没有 H.264 编码器");
        ame_free(c);
        return NULL;
    }

    AMediaFormat *fmt = AMediaFormat_new();
    if (!fmt) {
        set_err(c, "AMediaFormat_new 失败");
        ame_free(c);
        return NULL;
    }
    AMediaFormat_setString(fmt, "mime", MIME_AVC);
    AMediaFormat_setInt32(fmt, "width", w);
    AMediaFormat_setInt32(fmt, "height", h);
    AMediaFormat_setInt32(fmt, "bitrate", bitrate);
    AMediaFormat_setInt32(fmt, "color-format", COLOR_FMT_YUV420_PLANAR);
    AMediaFormat_setInt32(fmt, "frame-rate", fps);
    AMediaFormat_setInt32(fmt, "i-frame-interval", i_frame_interval_s);
    media_status_t st = AMediaCodec_configure(c->codec, fmt, NULL, NULL,
                                              AMEDIACODEC_CONFIGURE_FLAG_ENCODE);
    AMediaFormat_delete(fmt);
    if (st != AMEDIA_OK) {
        set_err(c, "编码器 configure: %d（多半是这台设备不吃 COLOR_FormatYUV420Planar）", (int)st);
        ame_free(c);
        return NULL;
    }
    if (AMediaCodec_start(c->codec) != AMEDIA_OK) {
        set_err(c, "编码器 start 失败");
        ame_free(c);
        return NULL;
    }
    return c;
}

int ame_add_audio_track(struct ame_ctx *c, const uint8_t *asc, int asc_len,
                        int sample_rate, int channels) {
    if (!c) return -1;
    if (c->audio_track >= 0) {
        set_err(c, "音频轨已加过");
        return -1;
    }
    AMediaFormat *fmt = AMediaFormat_new();
    if (!fmt) {
        set_err(c, "AMediaFormat_new 失败");
        return -1;
    }
    AMediaFormat_setString(fmt, "mime", MIME_AAC);
    AMediaFormat_setInt32(fmt, "sample-rate", sample_rate);
    AMediaFormat_setInt32(fmt, "channel-count", channels);
    AMediaFormat_setBuffer(fmt, "csd-0", asc, (size_t)asc_len);
    c->audio_track = AMediaMuxer_addTrack(c->muxer, fmt);
    AMediaFormat_delete(fmt);
    if (c->audio_track < 0) {
        set_err(c, "addTrack(音频) 失败");
        return -1;
    }
    if (try_start_muxer(c) != AMEDIA_OK) return -1;
    return 0;
}

/* 取一个 input buffer。
 *
 * `AMediaCodec_dequeueInputBuffer` 在「暂时没有空闲缓冲」时返回 -1
 * （= AMEDIACODEC_INFO_TRY_AGAIN_LATER），这是**正常现象**：软件编码器、
 * 低端 SoC、或系统繁忙时都可能出现。只等 10 ms 就判失败，会让转码在这类
 * 设备上直接放弃（模拟器的软件 AAC 编码器就是这么暴露的）—— 所以在 5 s
 * 内重试，超时才当错误。 */
static ssize_t dequeue_input_wait(struct ame_ctx *c, const char *what) {
    for (int waited = 0; waited < 5000; waited += 10) {
        ssize_t idx = AMediaCodec_dequeueInputBuffer(c->codec, 10 /* ms */);
        if (idx >= 0) return idx;
        if (idx != AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
            set_err(c, "%s dequeueInputBuffer: %zd", what, idx);
            return -1;
        }
    }
    set_err(c, "%s 5 s 内拿不到 input buffer（编码器卡住）", what);
    return -1;
}

int ame_write_video(struct ame_ctx *c, const uint8_t *y, const uint8_t *u,
                    const uint8_t *v, int64_t pts_us, int force_keyframe) {
    if (!c) return -1;
    ssize_t idx = dequeue_input_wait(c, "视频");
    if (idx < 0) return -1;
    size_t cap = 0;
    uint8_t *buf = AMediaCodec_getInputBuffer(c->codec, (size_t)idx, &cap);
    if (!buf || (size_t)(c->y_size + 2 * c->uv_size) > cap) {
        set_err(c, "input buffer 容量 %zu < 需要 %d", cap, c->y_size + 2 * c->uv_size);
        return -1;
    }
    /* YUV420P 三平面 → COLOR_FormatYUV420Planar（stride = width，紧排）。
     * 三个 memcpy：Y、U、V 各一次 —— 没有更省的路径（解码出的平面本来就是分开的）。 */
    memcpy(buf, y, (size_t)c->y_size);
    memcpy(buf + c->y_size, u, (size_t)c->uv_size);
    memcpy(buf + c->y_size + c->uv_size, v, (size_t)c->uv_size);

    media_status_t st = AMediaCodec_queueInputBuffer(
        c->codec, (size_t)idx, 0, (size_t)(c->y_size + 2 * c->uv_size),
        (uint64_t)pts_us, 0);
    if (st != AMEDIA_OK) {
        set_err(c, "queueInputBuffer: %d", (int)st);
        return -1;
    }
    if (force_keyframe) {
        /* API 26+（最低 API 也是 26，见 build.rs）。失败就静默忽略 —— GOP 由 i-frame-interval 兜底。 */
        AMediaFormat *p = AMediaFormat_new();
        if (p) {
            AMediaFormat_setInt32(p, "request-sync", 0);
            AMediaCodec_setParameters(c->codec, p);
            AMediaFormat_delete(p);
        }
    }
    return drain(c);
}

int ame_write_audio(struct ame_ctx *c, const uint8_t *data, int len, int64_t pts_us) {
    if (!c) return -1;
    ssize_t idx = dequeue_input_wait(c, "音频");
    if (idx < 0) return -1;
    size_t cap = 0;
    uint8_t *buf = AMediaCodec_getInputBuffer(c->codec, (size_t)idx, &cap);
    if (!buf || (size_t)len > cap) {
        set_err(c, "音频 input buffer 容量不足（%zu < %d）", cap, len);
        return -1;
    }
    memcpy(buf, data, (size_t)len);
    media_status_t st = AMediaCodec_queueInputBuffer(c->codec, (size_t)idx, 0,
                                                     (size_t)len, (uint64_t)pts_us, 0);
    if (st != AMEDIA_OK) {
        set_err(c, "音频 queueInputBuffer: %d", (int)st);
        return -1;
    }
    return drain(c);
}

int ame_finish(struct ame_ctx *c) {
    if (!c) return -1;
    ssize_t idx = AMediaCodec_dequeueInputBuffer(c->codec, 100000 /* EOS 可以等久点 */);
    if (idx >= 0) {
        AMediaCodec_queueInputBuffer(c->codec, (size_t)idx, 0, 0, 0,
                                     AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
    }
    /* EOS 之后的输出可能还要几帧才吐完；设个上限防死等。 */
    for (int i = 0; i < 10000 && !c->eos_seen; i++) {
        if (drain(c) != 0) return -1;
    }
    AMediaCodec_stop(c->codec);
    if (c->muxer_started) {
        media_status_t st = AMediaMuxer_stop(c->muxer);
        if (st != AMEDIA_OK) {
            set_err(c, "AMediaMuxer_stop: %d", (int)st);
            return -1;
        }
    }
    close(c->fd);
    c->fd = -1;
    return 0;
}

void ame_free(struct ame_ctx *c) {
    if (!c) return;
    if (c->codec) AMediaCodec_delete(c->codec);
    if (c->muxer) AMediaMuxer_delete(c->muxer);
    if (c->fd >= 0) close(c->fd);
    free(c->csd);
    free(c);
}
