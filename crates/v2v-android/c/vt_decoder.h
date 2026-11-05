/*
 * vt_decoder.h —— macOS VideoToolbox 硬件解码 shim 的 C ABI
 * （Rust 在 src/vt.rs 里声明同一份）。
 *
 * 与 vt_encoder.h 对称：0 = 成功；-1 = 失败（vt_dec_last_error 拿原因）；
 * vt_dec_next 返回 1 = 暂无更多。时间戳一律微秒。
 *
 * 输入口径与我们的 MP4 解封装一致：**avcC + 长度前缀 NALU**（不是 Annex-B）。
 * 输出口径与我们的解码器一致：YUV420P 三平面、紧排、行距 = 宽。
 */
#ifndef V2V_VT_DECODER_H
#define V2V_VT_DECODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vt_dec_ctx vt_dec_ctx;

/* 用 avcC（AVCDecoderConfigurationRecord，含 SPS/PPS）打开解码会话。
 * 长度前缀的宽度取自 avcC 里的 lengthSizeMinusOne。 */
struct vt_dec_ctx *vt_dec_open(const uint8_t *avcc, int avcc_len);

/* 喂一个样本（长度前缀 NALU）。结果经回调进队列，用 vt_dec_next 取。 */
int vt_dec_feed(struct vt_dec_ctx *c, const uint8_t *data, int len, int64_t pts_us);

/* 取一个解出来的帧。y/u/v 指向**同一个**缓冲里的三个平面（紧排，行距 = 宽），
 * 尺寸 = 显示尺寸（不是编码尺寸：1080 的流按 1088 编码，靠 SPS 的
 * frame_cropping 声明显示高度）。返回 0 = 有；1 = 暂无更多；-1 = 错。
 * 缓冲在下一次 vt_dec_next 前有效。 */
int vt_dec_next(struct vt_dec_ctx *c, const uint8_t **y, const uint8_t **u,
                const uint8_t **v, int *w, int *h, int64_t *pts_us);

/* 发 EOS、等所有在途帧出来（结果进队列，用 vt_dec_next 取完）。 */
int vt_dec_flush(struct vt_dec_ctx *c);

/* 1 = 确实在用硬件解码器；0 = 软件回退；-1 = 查不到（属性不存在 / 会话已坏）。 */
int vt_dec_is_hardware(struct vt_dec_ctx *c);

const char *vt_dec_last_error(struct vt_dec_ctx *c);

void vt_dec_free(struct vt_dec_ctx *c);

#ifdef __cplusplus
}
#endif

#endif /* V2V_VT_DECODER_H */
