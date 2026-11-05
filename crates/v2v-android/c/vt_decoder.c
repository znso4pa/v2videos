/*
 * vt_decoder.c —— macOS 硬件解码 shim：VideoToolbox（H.264）。
 *
 * 与 vt_encoder.c 对称的第二个方向。输入口径刻意选**avcC + 长度前缀 NALU**：
 * 那正是我们 MP4 解封装吐出来的样本格式，也是我们自己硬编产出的格式 ——
 * 解码器因此不需要再写一层 Annex-B 转换。
 *
 * 约定：0 = 成功；-1 = 失败（vt_dec_last_error 拿原因）；vt_dec_next 返回 1 = 暂无更多。
 * vt_dec_next 交出的缓冲在下一次 vt_dec_next 前有效。
 */
#include "vt_decoder.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CoreFoundation/CoreFoundation.h>
#include <CoreMedia/CoreMedia.h>
#include <CoreVideo/CoreVideo.h>
#include <VideoToolbox/VideoToolbox.h>

/* 想要的输出像素格式：三平面 YUV420（Y / U / V）。
 * ⚠️ 这是**请求**不是保证 —— 硬件解码器可能不认，改给 NV12（'420v'）。
 * 实测（Apple Silicon / macOS 26）本机给的是 3 平面；2 平面那条路是兜底，
 * 在本机跑不到（见 pack_yuv420）。 */
#define PIX_FMT kCVPixelFormatType_420YpCbCr8Planar

struct frame {
    uint8_t *data; /* Y 平面起点；U、V 依次紧随其后（紧排） */
    int w, h;
    int64_t pts_us;
    struct frame *next;
};

struct vt_dec_ctx {
    VTDecompressionSessionRef session;
    CMVideoFormatDescriptionRef fmt;
    int nal_len; /* 长度前缀的字节数（avcC 的 lengthSizeMinusOne + 1） */
    pthread_mutex_t lock;

    struct frame *head, *tail; /* 解码结果队列（顺序 = 显示顺序） */

    /* vt_dec_next 交出的缓冲（静态持有，下次调用时回收 —— 单线程使用）。 */
    uint8_t *hold;
    size_t hold_cap;
    int hold_w, hold_h;

    char err[256];
};

static void set_err(struct vt_dec_ctx *c, const char *msg) {
    if (c) snprintf(c->err, sizeof(c->err), "%s", msg);
}

const char *vt_dec_last_error(struct vt_dec_ctx *c) {
    return c ? c->err : "null ctx";
}

/* ── 从 avcC 里抠出 SPS / PPS 与长度前缀宽度 ──────────────────────────────
 * AVCDecoderConfigurationRecord 的布局（ISO/IEC 14496-15 §5.2.4.1）：
 *   [0]    configurationVersion = 1
 *   [1..3] profile / compat / level
 *   [4]    0b111111xx，低 2 位 = lengthSizeMinusOne
 *   [5]    0b111xxxxx，低 5 位 = numOfSequenceParameterSets
 *   然后每个 SPS：[2 字节长度][数据]…
 *   再 [1 字节 numOfPictureParameterSets]，每个 PPS：[2 字节长度][数据]…
 * 指针指回 avcc 自己的内存 —— 调用方必须让 avcc 活着（本文件里拷了一份存住）。 */
static int parse_avcc(const uint8_t *avcc, int len, const uint8_t *ps[], size_t psl[],
                      int *ps_count, int *nal_len) {
    if (!avcc || len < 7) return -1;
    if (avcc[0] != 1) return -1;
    *nal_len = (avcc[4] & 0x03) + 1;
    int n = 0;
    int pos = 6;
    int nsps = avcc[5] & 0x1f;
    for (int i = 0; i < nsps; i++) {
        if (pos + 2 > len) return -1;
        int l = (avcc[pos] << 8) | avcc[pos + 1];
        pos += 2;
        if (l <= 0 || pos + l > len) return -1;
        if (n >= 2) return -1;
        ps[n] = avcc + pos;
        psl[n] = (size_t)l;
        n++;
        pos += l;
    }
    if (pos + 1 > len) return -1;
    int npps = avcc[pos];
    pos++;
    for (int i = 0; i < npps; i++) {
        if (pos + 2 > len) return -1;
        int l = (avcc[pos] << 8) | avcc[pos + 1];
        pos += 2;
        if (l <= 0 || pos + l > len) return -1;
        if (n >= 2) return -1;
        ps[n] = avcc + pos;
        psl[n] = (size_t)l;
        n++;
        pos += l;
    }
    if (n < 2) return -1; /* 至少要有 SPS + PPS */
    *ps_count = n;
    return 0;
}

/* ── 把 CVPixelBuffer 拷成紧排 YUV420P，按 clean aperture 裁到显示尺寸 ─────
 * 返回 malloc 的缓冲（调用方 free），尺寸经 out_w / out_h 带回。 */
static uint8_t *pack_yuv420(CVPixelBufferRef pix, int *out_w, int *out_h, char *err,
                            size_t errlen) {
    const int cw = (int)CVPixelBufferGetWidth(pix);
    const int ch = (int)CVPixelBufferGetHeight(pix);
    const OSType pf = CVPixelBufferGetPixelFormatType(pix);
    const int nplanes = (int)CVPixelBufferGetPlaneCount(pix);

    /* 显示尺寸：优先用 clean aperture。
     * ⚠️ 实测（Apple Silicon / macOS 26）VT **已经**把 SPS 的 frame_cropping
     * 折进了 buffer 尺寸（1080 的流拿到的就是 1920x1080，不是 1088），
     * 所以下面这段在本机跑不到。留着是为了别的版本/驱动给 1088 的情况：
     * 少裁这 8 行黑边会让逐帧比对整体错位，症状是「画面看着对、PSNR 只有
     * 30 dB」—— 看起来像解码器坏了，其实是没裁。 */
    int dw = cw, dh = ch, ox = 0, oy = 0;
    CFDictionaryRef ca = (CFDictionaryRef)CMGetAttachment(
        (CMAttachmentBearerRef)pix, kCVImageBufferCleanApertureKey, NULL);
    if (ca) {
        int32_t v = 0, ho = 0, vo = 0;
        CFNumberRef n;
        if ((n = (CFNumberRef)CFDictionaryGetValue(
                 ca, kCVImageBufferCleanApertureWidthKey)) &&
            CFNumberGetValue(n, kCFNumberSInt32Type, &v))
            dw = v;
        if ((n = (CFNumberRef)CFDictionaryGetValue(
                 ca, kCVImageBufferCleanApertureHeightKey)) &&
            CFNumberGetValue(n, kCFNumberSInt32Type, &v))
            dh = v;
        /* 偏移是「干净窗口中心相对图像中心」的量。 */
        if ((n = (CFNumberRef)CFDictionaryGetValue(
                 ca, kCVImageBufferCleanApertureHorizontalOffsetKey)) &&
            CFNumberGetValue(n, kCFNumberSInt32Type, &ho)) {
        }
        if ((n = (CFNumberRef)CFDictionaryGetValue(
                 ca, kCVImageBufferCleanApertureVerticalOffsetKey)) &&
            CFNumberGetValue(n, kCFNumberSInt32Type, &vo)) {
        }
        ox = (cw - dw) / 2 + ho;
        oy = (ch - dh) / 2 + vo;
    }
    if (dw <= 0 || dh <= 0 || dw > cw || dh > ch) {
        dw = cw;
        dh = ch;
        ox = 0;
        oy = 0;
    }
    /* 420 要求偶数；裁剪原点也必须落在偶数格上，否则色度会串一行。 */
    dw &= ~1;
    dh &= ~1;
    ox &= ~1;
    oy &= ~1;
    if (ox < 0) ox = 0;
    if (oy < 0) oy = 0;
    if (ox + dw > cw) ox = (cw - dw) & ~1;
    if (oy + dh > ch) oy = (ch - dh) & ~1;

    const size_t ylen = (size_t)dw * dh;
    const size_t clen = (size_t)(dw / 2) * (dh / 2);
    uint8_t *out = malloc(ylen + 2 * clen);
    if (!out) {
        snprintf(err, errlen, "分配 %zu 字节失败", ylen + 2 * clen);
        return NULL;
    }
    uint8_t *dy = out, *du = out + ylen, *dv = du + clen;

    if (CVPixelBufferLockBaseAddress(pix, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) {
        free(out);
        snprintf(err, errlen, "锁 pixel buffer 失败");
        return NULL;
    }
    if (nplanes == 3) {
        /* 420Planar：Y / U / V 各一平面。 */
        const uint8_t *sy = CVPixelBufferGetBaseAddressOfPlane(pix, 0);
        const uint8_t *su = CVPixelBufferGetBaseAddressOfPlane(pix, 1);
        const uint8_t *sv = CVPixelBufferGetBaseAddressOfPlane(pix, 2);
        const size_t sty = CVPixelBufferGetBytesPerRowOfPlane(pix, 0);
        const size_t stu = CVPixelBufferGetBytesPerRowOfPlane(pix, 1);
        const size_t stv = CVPixelBufferGetBytesPerRowOfPlane(pix, 2);
        for (int r = 0; r < dh; r++)
            memcpy(dy + (size_t)r * dw, sy + (size_t)(oy + r) * sty + ox, (size_t)dw);
        for (int r = 0; r < dh / 2; r++) {
            memcpy(du + (size_t)r * (dw / 2), su + (size_t)(oy / 2 + r) * stu + ox / 2,
                   (size_t)(dw / 2));
            memcpy(dv + (size_t)r * (dw / 2), sv + (size_t)(oy / 2 + r) * stv + ox / 2,
                   (size_t)(dw / 2));
        }
    } else if (nplanes == 2) {
        /* NV12（'420v' 视频范围 / '420f' 全范围）：Y 一平面，UV 交织一平面。
         * 本机跑不到（VT 给的是 3 平面），留着当兜底 —— 所以这条分支
         * **没有测试覆盖**，改动它要格外小心。 */
        const uint8_t *sy = CVPixelBufferGetBaseAddressOfPlane(pix, 0);
        const uint8_t *suv = CVPixelBufferGetBaseAddressOfPlane(pix, 1);
        const size_t sty = CVPixelBufferGetBytesPerRowOfPlane(pix, 0);
        const size_t suv_st = CVPixelBufferGetBytesPerRowOfPlane(pix, 1);
        for (int r = 0; r < dh; r++)
            memcpy(dy + (size_t)r * dw, sy + (size_t)(oy + r) * sty + ox, (size_t)dw);
        for (int r = 0; r < dh / 2; r++) {
            const uint8_t *src = suv + (size_t)(oy / 2 + r) * suv_st + ox;
            uint8_t *pu = du + (size_t)r * (dw / 2);
            uint8_t *pv = dv + (size_t)r * (dw / 2);
            for (int i = 0; i < dw / 2; i++) {
                pu[i] = src[2 * i];
                pv[i] = src[2 * i + 1];
            }
        }
    } else {
        CVPixelBufferUnlockBaseAddress(pix, kCVPixelBufferLock_ReadOnly);
        free(out);
        snprintf(err, errlen, "pixel buffer 平面数不支持：%d（格式 '%.4s'）", nplanes,
                 (const char *)&pf);
        return NULL;
    }
    CVPixelBufferUnlockBaseAddress(pix, kCVPixelBufferLock_ReadOnly);

    *out_w = dw;
    *out_h = dh;
    return out;
}

/* 解码完成回调（可能跑在 VT 的内部线程上）。
 * `sourceFrameRefCon` 是我们传进去的 pts（微秒）。 */
static void decomp_cb(void *decompressionOutputRefCon, void *sourceFrameRefCon,
                      OSStatus status, VTDecodeInfoFlags infoFlags,
                      CVImageBufferRef imageBuffer, CMTime presentationTimeStamp,
                      CMTime presentationDuration) {
    (void)presentationTimeStamp;
    (void)presentationDuration;
    struct vt_dec_ctx *c = decompressionOutputRefCon;
    if (!c) return;
    if (status != noErr || !imageBuffer) {
        set_err(c, "解码回调报错");
        return;
    }
    if (infoFlags & kVTDecodeInfo_FrameDropped) return;

    int w = 0, h = 0;
    char msg[128] = {0};
    uint8_t *data = pack_yuv420(imageBuffer, &w, &h, msg, sizeof(msg));
    if (!data) {
        set_err(c, msg);
        return;
    }
    struct frame *f = malloc(sizeof(*f));
    if (!f) {
        free(data);
        set_err(c, "分配帧节点失败");
        return;
    }
    *f = (struct frame){data, w, h, (int64_t)(intptr_t)sourceFrameRefCon, NULL};
    pthread_mutex_lock(&c->lock);
    if (c->tail)
        c->tail->next = f;
    else
        c->head = f;
    c->tail = f;
    pthread_mutex_unlock(&c->lock);
}

struct vt_dec_ctx *vt_dec_open(const uint8_t *avcc, int avcc_len) {
    struct vt_dec_ctx *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    pthread_mutex_init(&c->lock, NULL);

    /* avcC 要活到会话销毁：SPS/PPS 指针指进这份拷贝。 */
    uint8_t *copy = malloc((size_t)avcc_len);
    if (!copy) {
        set_err(c, "分配 avcC 失败");
        vt_dec_free(c);
        return NULL;
    }
    memcpy(copy, avcc, (size_t)avcc_len);

    const uint8_t *ps[2] = {NULL, NULL};
    size_t psl[2] = {0, 0};
    int ps_count = 0;
    if (parse_avcc(copy, avcc_len, ps, psl, &ps_count, &c->nal_len) != 0) {
        free(copy);
        set_err(c, "avcC 解析失败（缺 SPS/PPS 或版本不是 1）");
        vt_dec_free(c);
        return NULL;
    }
    OSStatus st = CMVideoFormatDescriptionCreateFromH264ParameterSets(
        NULL, ps_count, ps, psl, c->nal_len, &c->fmt);
    free(copy);
    if (st != noErr || !c->fmt) {
        set_err(c, "CMVideoFormatDescriptionCreateFromH264ParameterSets 失败");
        vt_dec_free(c);
        return NULL;
    }

    /* 目标像素缓冲的属性。IOSurface 那一项不能省：硬件解码要直接写进
     * IOSurface 支撑的缓冲，不给这一项 VT 会退到软件路径。 */
    CFMutableDictionaryRef dst = CFDictionaryCreateMutable(
        NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    int32_t fmt = PIX_FMT;
    CFNumberRef nf = CFNumberCreate(NULL, kCFNumberSInt32Type, &fmt);
    CFDictionarySetValue(dst, kCVPixelBufferPixelFormatTypeKey, nf);
    CFRelease(nf);
    CFMutableDictionaryRef empty = CFDictionaryCreateMutable(
        NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(dst, kCVPixelBufferIOSurfacePropertiesKey, empty);
    CFRelease(empty);

    /* 「能上硬件就上」——但不强求：强求会在没有硬解的设备上直接失败，
     * 而我们真正想量的是「同一份位流两个解码器差多少」。 */
    CFMutableDictionaryRef spec = CFDictionaryCreateMutable(
        NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(spec, CFSTR("EnableHardwareAcceleratedVideoDecoder"), kCFBooleanTrue);

    /* 回调是「记录」不是两个散参数：refCon 塞本 ctx，帧的 pts 另走
     * DecodeFrame 的 sourceFrameRefCon（两者互不干扰）。 */
    VTDecompressionOutputCallbackRecord rec;
    rec.decompressionOutputCallback = decomp_cb;
    rec.decompressionOutputRefCon = c;
    st = VTDecompressionSessionCreate(NULL, c->fmt, spec, dst, &rec, &c->session);
    CFRelease(spec);
    CFRelease(dst);
    if (st != noErr || !c->session) {
        set_err(c, "VTDecompressionSessionCreate 失败");
        vt_dec_free(c);
        return NULL;
    }
    return c;
}

int vt_dec_feed(struct vt_dec_ctx *c, const uint8_t *data, int len, int64_t pts_us) {
    if (!c || !c->session) return -1;
    if (len <= 0) return 0;

    /* CMBlockBuffer 要接管这块内存（kCFAllocatorMalloc 会在释放时调 free），
     * 所以必须自己拷一份 —— 调用方的样本随时可能被复用。 */
    uint8_t *copy = malloc((size_t)len);
    if (!copy) {
        set_err(c, "分配样本失败");
        return -1;
    }
    memcpy(copy, data, (size_t)len);

    CMBlockBufferRef block = NULL;
    OSStatus st = CMBlockBufferCreateWithMemoryBlock(
        kCFAllocatorDefault, copy, (size_t)len, kCFAllocatorMalloc, NULL, 0, (size_t)len, 0,
        &block);
    if (st != kCMBlockBufferNoErr || !block) {
        free(copy);
        set_err(c, "CMBlockBufferCreateWithMemoryBlock 失败");
        return -1;
    }
    CMSampleBufferRef sbuf = NULL;
    CMSampleTimingInfo timing;
    timing.duration = kCMTimeInvalid;
    timing.presentationTimeStamp = CMTimeMake(pts_us, 1000 * 1000); /* 微秒时间基 */
    timing.decodeTimeStamp = kCMTimeInvalid;
    st = CMSampleBufferCreateReady(kCFAllocatorDefault, block, c->fmt, 1, 1, &timing, 0, NULL,
                                   &sbuf);
    CFRelease(block);
    if (st != noErr || !sbuf) {
        set_err(c, "CMSampleBufferCreateReady 失败");
        return -1;
    }
    VTDecodeInfoFlags flags_out = 0;
    st = VTDecompressionSessionDecodeFrame(c->session, sbuf, 0, (void *)(intptr_t)pts_us,
                                           &flags_out);
    CFRelease(sbuf);
    if (st != noErr) {
        char msg[96];
        snprintf(msg, sizeof(msg), "DecodeFrame 失败（OSStatus %d）", (int)st);
        set_err(c, msg);
        return -1;
    }
    return 0;
}

int vt_dec_next(struct vt_dec_ctx *c, const uint8_t **y, const uint8_t **u,
                const uint8_t **v, int *w, int *h, int64_t *pts_us) {
    if (!c) return -1;
    pthread_mutex_lock(&c->lock);
    struct frame *f = c->head;
    if (f) {
        c->head = f->next;
        if (!c->head) c->tail = NULL;
    }
    pthread_mutex_unlock(&c->lock);
    if (!f) return 1; /* 暂无更多 */

    const size_t need = (size_t)f->w * f->h * 3 / 2;
    if (c->hold_cap < need) {
        free(c->hold);
        c->hold = malloc(need);
        c->hold_cap = need;
    }
    memcpy(c->hold, f->data, need);
    c->hold_w = f->w;
    c->hold_h = f->h;
    *y = c->hold;
    *u = c->hold + (size_t)f->w * f->h;
    *v = *u + (size_t)(f->w / 2) * (f->h / 2);
    *w = f->w;
    *h = f->h;
    *pts_us = f->pts_us;
    free(f->data);
    free(f);
    return 0;
}

int vt_dec_flush(struct vt_dec_ctx *c) {
    if (!c || !c->session) return -1;
    /* 两件事分开做：先标记「不再有延迟帧」，再等异步解码收尾。
     * 只等不标记的话，带 B 帧的流会一直留着最后几帧不出。 */
    if (VTDecompressionSessionFinishDelayedFrames(c->session) != noErr) {
        set_err(c, "FinishDelayedFrames 失败");
        return -1;
    }
    if (VTDecompressionSessionWaitForAsynchronousFrames(c->session) != noErr) {
        set_err(c, "WaitForAsynchronousFrames 失败");
        return -1;
    }
    return 0;
}

int vt_dec_is_hardware(struct vt_dec_ctx *c) {
    if (!c || !c->session) return -1;
    CFBooleanRef b = NULL;
    OSStatus st = VTSessionCopyProperty(
        (VTSessionRef)c->session,
        kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder, kCFAllocatorDefault,
        &b);
    if (st != noErr || !b) return -1;
    int r = CFBooleanGetValue(b) ? 1 : 0;
    CFRelease(b);
    return r;
}

void vt_dec_free(struct vt_dec_ctx *c) {
    if (!c) return;
    if (c->session) {
        /* Invalidate 会先排空并停掉回调线程 —— 必须在释放队列之前，
         * 否则回调可能正往一个已经 free 的队列里塞节点。 */
        VTDecompressionSessionInvalidate(c->session);
        CFRelease(c->session);
    }
    if (c->fmt) CFRelease(c->fmt);
    struct frame *f = c->head;
    while (f) {
        struct frame *nx = f->next;
        free(f->data);
        free(f);
        f = nx;
    }
    free(c->hold);
    pthread_mutex_destroy(&c->lock);
    free(c);
}
