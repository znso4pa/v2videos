/*
 * vt_encoder.h —— macOS VideoToolbox 硬件编码 shim 的 C ABI
 * （Rust 在 src/vt.rs 里声明同一份）。
 *
 * 约定：0 = 成功；-1 = 失败（vt_last_error 拿原因）；vt_next 返回 1 = 暂无更多。
 * 时间戳一律微秒。
 */
#ifndef V2V_VT_ENCODER_H
#define V2V_VT_ENCODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vt_ctx vt_ctx;

/* 打开硬件 H.264 编码器。bitrate：bit/s；fps：整数帧率；
 * gop_seconds：GOP 秒数（VT 的原生单位就是秒）。 */
struct vt_ctx *vt_open(int w, int h, int bitrate, int fps, int gop_seconds);

/* 喂一帧 YUV420P（三平面紧排，行距 = width；VT 内部按自己的行距搬）。
 * force_keyframe：非 0 强制这一帧是 IDR。结果经回调进队列，用 vt_next 取。 */
int vt_encode(struct vt_ctx *c, const uint8_t *y, const uint8_t *u, const uint8_t *v,
              int64_t pts_us, int force_keyframe);

/* 取一个编码结果（长度前缀 NALU，非 Annex-B）。返回 0 = 有，1 = 暂无更多。
 * 缓冲在下一次 vt_next 前有效。 */
int vt_next(struct vt_ctx *c, uint8_t **data, int *len, int64_t *pts_us, int *key);

/* 取 avcC（AVCDecoderConfigurationRecord，含 SPS/PPS）。
 * 返回 0 = 有；1 = 还没有（一帧都没编出来）。缓冲随 ctx 存活。 */
int vt_avcc(struct vt_ctx *c, uint8_t **data, int *len);

/* 发 EOS、吸干所有在途样本（结果进队列，用 vt_next 取完）。 */
int vt_finish(struct vt_ctx *c);

const char *vt_last_error(struct vt_ctx *c);

void vt_free(struct vt_ctx *c);

#ifdef __cplusplus
}
#endif

#endif /* V2V_VT_ENCODER_H */
