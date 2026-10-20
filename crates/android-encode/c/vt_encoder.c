/*
 * vt_encoder.c —— macOS 硬件编码 shim：VideoToolbox（H.264）。
 *
 * 与 media_encoder.c（Android / MediaCodec）并列的第二个后端 —— 同一个
 * 「速度与质量优先，rs + C/C++」的决定：Apple Silicon 的 VT 是**真硬件**
 * 编码器，质量与速度都远超我们的标量 mpeg4，且只能从 C 侧调。
 *
 * 与 MediaCodec 后端不同：这里**只产样本，不封装** —— 输出 H.264 的
 * 「长度前缀 NALU」样本 + avcC（SPS/PPS），MP4 封装交给 Rust 侧的
 * v2v-mp4（复用我们自己的 muxer，保证两套后端产物同构）。
 *
 * 约定：0 = 成功；-1 = 失败（vt_last_error 拿原因）；vt_next 返回 1 = 暂无更多。
 * vt_next 交出的缓冲在下一次 vt_next 前有效。
 */
#include "vt_encoder.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include <CoreFoundation/CoreFoundation.h>
#include <CoreMedia/CoreMedia.h>
#include <VideoToolbox/VideoToolbox.h>

/* 输入像素格式：三平面 YUV420（Y / U / V），与我们的解码器输出同构。 */
#define PIX_FMT kCVPixelFormatType_420YpCbCr8Planar

struct sample {
    uint8_t *data;
    size_t len;
    int64_t pts_us;
    int key;
    struct sample *next;
};

struct vt_ctx {
    VTCompressionSessionRef session;
    int w, h;
    pthread_mutex_t lock;

    struct sample *head, *tail; /* 编码结果队列（顺序 = 输入顺序） */
    uint8_t *avcc;              /* AVCDecoderConfigurationRecord（SPS/PPS） */
    size_t avcc_len;

    /* vt_next 交出的缓冲（静态持有，下次调用时回收 —— 单线程使用）。 */
    uint8_t *hold;
    size_t hold_cap;

    char err[256];
};

static void set_err(struct vt_ctx *c, const char *msg) {
    if (c) snprintf(c->err, sizeof(c->err), "%s", msg);
}

const char *vt_last_error(struct vt_ctx *c) {
    return c ? c->err : "null ctx";
}

static void queue_push(struct vt_ctx *c, uint8_t *data, size_t len, int64_t pts_us,
                       int key) {
    struct sample *s = malloc(sizeof(*s));
    *s = (struct sample){data, len, pts_us, key, NULL};
    pthread_mutex_lock(&c->lock);
    if (c->tail)
        c->tail->next = s;
    else
        c->head = s;
    c->tail = s;
    pthread_mutex_unlock(&c->lock);
}

/* 从输出样本的 format description 里抠 avcC（SPS/PPS 都在里面）。
 * 一拿到就存住 —— 之后每帧都带同一份，不重复取。 */
static void maybe_grab_avcc(struct vt_ctx *c, CMSampleBufferRef sbuf) {
    if (c->avcc) return;
    CMFormatDescriptionRef desc = CMSampleBufferGetFormatDescription(sbuf);
    if (!desc) return;
    CFDictionaryRef exts = CMFormatDescriptionGetExtensions(desc);
    if (!exts) return;
    CFDictionaryRef atoms = CFDictionaryGetValue(
        exts, kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms);
    if (!atoms) return;
    CFDataRef avcc = CFDictionaryGetValue(atoms, CFSTR("avcC"));
    if (!avcc) return;
    size_t n = CFDataGetLength(avcc);
    uint8_t *copy = malloc(n);
    memcpy(copy, CFDataGetBytePtr(avcc), n);
    pthread_mutex_lock(&c->lock);
    if (!c->avcc) { /* 回调可能与外部并发，先到先得 */
        c->avcc = copy;
        c->avcc_len = n;
        copy = NULL;
    }
    pthread_mutex_unlock(&c->lock);
    free(copy);
}

/* 编码完成回调（可能跑在 VT 的内部线程上）。
 * `sourceFrameRefCon` 是我们传进去的 pts（微秒）。 */
static void compression_cb(void *opaque, void *sourceFrameRefCon, OSStatus status,
                           VTEncodeInfoFlags flags, CMSampleBufferRef sbuf) {
    struct vt_ctx *c = opaque;
    if (!c) return;
    if (status != noErr || !sbuf) {
        set_err(c, "编码回调报错");
        return;
    }
    maybe_grab_avcc(c, sbuf);

    CMBlockBufferRef block = CMSampleBufferGetDataBuffer(sbuf);
    if (!block) return;
    size_t len = CMBlockBufferGetDataLength(block);
    if (len == 0) return;
    uint8_t *data = malloc(len);
    /* 块可能不连续 —— 统一拷平（每帧几十 KB 的量级，不是热点）。 */
    if (CMBlockBufferCopyDataBytes(block, 0, len, data) != kCMBlockBufferNoErr) {
        free(data);
        set_err(c, "样本数据拷贝失败");
        return;
    }
    /* 同步样本判定：有 `NotSync` 附件且为真的不是关键帧。 */
    int key = 1;
    CFBooleanRef not_sync =
        CMGetAttachment(sbuf, kCMSampleAttachmentKey_NotSync, NULL);
    if (not_sync == kCFBooleanTrue) key = 0;
    if (flags & kVTEncodeInfo_FrameDropped) key = 0;

    queue_push(c, data, len, (int64_t)(intptr_t)sourceFrameRefCon, key);
}

struct vt_ctx *vt_open(int w, int h, int bitrate, int fps, int gop_seconds) {
    struct vt_ctx *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->w = w;
    c->h = h;
    pthread_mutex_init(&c->lock, NULL);

    /* 输入像素缓冲的约束（VT 据此建池，行距可能比 width 大）。 */
    CFMutableDictionaryRef buf_attrs = CFDictionaryCreateMutable(
        NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    int32_t w32 = w, h32 = h, fmt = PIX_FMT;
    CFNumberRef nw = CFNumberCreate(NULL, kCFNumberSInt32Type, &w32);
    CFNumberRef nh = CFNumberCreate(NULL, kCFNumberSInt32Type, &h32);
    CFNumberRef nf = CFNumberCreate(NULL, kCFNumberSInt32Type, &fmt);
    CFDictionarySetValue(buf_attrs, kCVPixelBufferWidthKey, nw);
    CFDictionarySetValue(buf_attrs, kCVPixelBufferHeightKey, nh);
    CFDictionarySetValue(buf_attrs, kCVPixelBufferPixelFormatTypeKey, nf);
    CFRelease(nw);
    CFRelease(nh);
    CFRelease(nf);

    OSStatus st = VTCompressionSessionCreate(
        /* allocator */ NULL, w, h, kCMVideoCodecType_H264,
        /* encoderSpecification */ NULL, buf_attrs,
        /* compressedDataAllocator */ NULL, compression_cb, c, &c->session);
    CFRelease(buf_attrs);
    if (st != noErr || !c->session) {
        set_err(c, "VTCompressionSessionCreate 失败");
        vt_free(c);
        return NULL;
    }
    /* ⚠️ **不要缓存 `VTCompressionSessionGetPixelBufferPool` 的返回值**：
     * 那是 Get 语义（不持有），而且 session 会在内部**重建**池 —— 缓存的指针
     * 过一会儿就悬空。实测：窗口调大（在飞帧变多）后 `vt_encode` 里
     * `CVPixelBufferPoolCreatePixelBuffer` 直接 SIGSEGV（崩溃栈停在 objc_msgSend）。
     * 改成每次用时现取（见 `vt_encode`）。 */

    /* 质量优先：RealTime = false（离线模式，编码器可以多花时间换质量）；
     * 不许重排帧（无 B 帧）—— 输出顺序 = 输入顺序，封装的时间轴才是直的。 */
    VTSessionSetProperty(c->session, kVTCompressionPropertyKey_RealTime,
                         kCFBooleanFalse);
    VTSessionSetProperty(c->session, kVTCompressionPropertyKey_AllowFrameReordering,
                         kCFBooleanFalse);
    int32_t br = bitrate;
    CFNumberRef nbr = CFNumberCreate(NULL, kCFNumberSInt32Type, &br);
    VTSessionSetProperty(c->session, kVTCompressionPropertyKey_AverageBitRate, nbr);
    CFRelease(nbr);
    double fps_d = fps;
    CFNumberRef nfps = CFNumberCreate(NULL, kCFNumberDoubleType, &fps_d);
    VTSessionSetProperty(c->session, kVTCompressionPropertyKey_ExpectedFrameRate, nfps);
    CFRelease(nfps);
    /* macOS 的 SDK 没有 GopDuration（那是 iOS 的名字）——
     * 用「最大关键帧间隔」的两种等价写法：帧数 + 秒数都设。 */
    int32_t gop_frames = gop_seconds * fps;
    CFNumberRef ngf = CFNumberCreate(NULL, kCFNumberSInt32Type, &gop_frames);
    VTSessionSetProperty(c->session, kVTCompressionPropertyKey_MaxKeyFrameInterval, ngf);
    CFRelease(ngf);
    double gop = gop_seconds;
    CFNumberRef ngop = CFNumberCreate(NULL, kCFNumberDoubleType, &gop);
    VTSessionSetProperty(c->session, kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration,
                         ngop);
    CFRelease(ngop);
    VTSessionSetProperty(c->session, kVTCompressionPropertyKey_ProfileLevel,
                         kVTProfileLevel_H264_Main_AutoLevel);

    if (VTCompressionSessionPrepareToEncodeFrames(c->session) != noErr) {
        set_err(c, "PrepareToEncode 失败");
        vt_free(c);
        return NULL;
    }
    return c;
}

int vt_encode(struct vt_ctx *c, const uint8_t *y, const uint8_t *u, const uint8_t *v,
              int64_t pts_us, int force_keyframe) {
    if (!c || !c->session) return -1;
    CVPixelBufferRef pix = NULL;
    CVPixelBufferPoolRef pool = VTCompressionSessionGetPixelBufferPool(c->session);
    if (!pool || CVPixelBufferPoolCreatePixelBuffer(NULL, pool, &pix) != kCVReturnSuccess ||
        !pix) {
        set_err(c, "取不到 pixel buffer");
        return -1;
    }
    /* 池给的格式必须就是三平面 —— 实测抓过一次「格式不符 → 平面 2 地址为空 →
     * memmove 崩」的案例（VT 有时会换成 BiPlanar）。宁可显式拒绝。 */
    OSType got = CVPixelBufferGetPixelFormatType(pix);
    int nplanes = CVPixelBufferGetPlaneCount(pix);
    if (got != PIX_FMT && got != '420v' && got != '420f') {
        char fcc[8] = {0};
        memcpy(fcc, &got, 4);
        char msg[128];
        snprintf(msg, sizeof(msg), "pixel buffer 格式不支持：'%s'（plane %d）", fcc, nplanes);
        set_err(c, msg);
        CVPixelBufferRelease(pix);
        return -1;
    }
    if (CVPixelBufferLockBaseAddress(pix, 0) != kCVReturnSuccess) {
        set_err(c, "锁 pixel buffer 失败");
        CVPixelBufferRelease(pix);
        return -1;
    }
    if (nplanes == 3) {
        /* 420Planar（Y / U / V 三平面）：逐行拷，池的行距可能比 width 大。 */
        const uint8_t *planes[3] = {y, u, v};
        for (size_t p = 0; p < 3; p++) {
            uint8_t *dst = CVPixelBufferGetBaseAddressOfPlane(pix, p);
            size_t stride = CVPixelBufferGetBytesPerRowOfPlane(pix, p);
            int pw = p == 0 ? c->w : c->w / 2;
            int ph = p == 0 ? c->h : c->h / 2;
            for (int row = 0; row < ph; row++) {
                memcpy(dst + row * stride, planes[p] + (size_t)row * pw, (size_t)pw);
            }
        }
    } else {
        /* NV12（'420v' / '420f'，两平面）：VT 在 Apple Silicon 上就爱给这个。
         * Y 原样逐行；U/V 要**交织**进第二平面的 UV 行 —— 两个内存布局的
         * 转换躲不掉，用两根指针一次扫完。 */
        uint8_t *dy = CVPixelBufferGetBaseAddressOfPlane(pix, 0);
        size_t sy = CVPixelBufferGetBytesPerRowOfPlane(pix, 0);
        for (int row = 0; row < c->h; row++) {
            memcpy(dy + row * sy, y + (size_t)row * c->w, (size_t)c->w);
        }
        uint8_t *duv = CVPixelBufferGetBaseAddressOfPlane(pix, 1);
        size_t suv = CVPixelBufferGetBytesPerRowOfPlane(pix, 1);
        for (int row = 0; row < c->h / 2; row++) {
            const uint8_t *pu = u + (size_t)row * (c->w / 2);
            const uint8_t *pv = v + (size_t)row * (c->w / 2);
            uint8_t *dst = duv + row * suv;
            for (int i = 0; i < c->w / 2; i++) {
                dst[2 * i] = pu[i];
                dst[2 * i + 1] = pv[i];
            }
        }
    }
    CVPixelBufferUnlockBaseAddress(pix, 0);

    CFMutableDictionaryRef props = NULL;
    if (force_keyframe) {
        props = CFDictionaryCreateMutable(NULL, 1, &kCFTypeDictionaryKeyCallBacks,
                                          &kCFTypeDictionaryValueCallBacks);
        CFDictionarySetValue(props, kVTEncodeFrameOptionKey_ForceKeyFrame,
                             kCFBooleanTrue);
    }
    CMTime pts = CMTimeMake(pts_us, 1000 * 1000); /* 微秒时间基 */
    /* 结果一律走回调（EncodeFrame 的末参只是 flags 出口，不是输出句柄）。 */
    VTEncodeInfoFlags flags_out = 0;
    OSStatus st = VTCompressionSessionEncodeFrame(
        c->session, pix, pts, kCMTimeInvalid, props, (void *)(intptr_t)pts_us,
        &flags_out);
    if (props) CFRelease(props);
    CVPixelBufferRelease(pix);
    if (st != noErr) {
        set_err(c, "EncodeFrame 失败");
        return -1;
    }
    return 0;
}

int vt_next(struct vt_ctx *c, uint8_t **data, int *len, int64_t *pts_us, int *key) {
    if (!c) return -1;
    pthread_mutex_lock(&c->lock);
    struct sample *s = c->head;
    if (s) {
        c->head = s->next;
        if (!c->head) c->tail = NULL;
    }
    pthread_mutex_unlock(&c->lock);
    if (!s) return 1; /* 暂无更多 */
    if (c->hold_cap < s->len) {
        free(c->hold);
        c->hold = malloc(s->len);
        c->hold_cap = s->len;
    }
    memcpy(c->hold, s->data, s->len);
    int64_t pts = s->pts_us;
    int k = s->key;
    size_t n = s->len;
    free(s->data);
    free(s);
    *data = c->hold;
    *len = (int)n;
    *pts_us = pts;
    *key = k;
    return 0;
}

int vt_avcc(struct vt_ctx *c, uint8_t **data, int *len) {
    if (!c) return -1;
    if (!c->avcc) return 1; /* 还没有（一帧都没编出来） */
    *data = c->avcc;
    *len = (int)c->avcc_len;
    return 0;
}

int vt_finish(struct vt_ctx *c) {
    if (!c || !c->session) return -1;
    if (VTCompressionSessionCompleteFrames(c->session, kCMTimeInvalid) != noErr) {
        set_err(c, "CompleteFrames 失败");
        return -1;
    }
    return 0;
}

void vt_free(struct vt_ctx *c) {
    if (!c) return;
    if (c->session) {
        VTCompressionSessionInvalidate(c->session);
        CFRelease(c->session);
    }
    struct sample *s = c->head;
    while (s) {
        struct sample *nx = s->next;
        free(s->data);
        free(s);
        s = nx;
    }
    free(c->avcc);
    free(c->hold);
    pthread_mutex_destroy(&c->lock);
    free(c);
}
