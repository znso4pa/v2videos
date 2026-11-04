/*
 * media_encoder.h —— Android 硬件编码 shim 的 C ABI（Rust 在 src/ffi.rs 里声明同一份）。
 *
 * 约定：返回 0 = 成功；-1 = 失败（ame_last_error 拿原因）。
 * 时间戳一律是微秒（MediaCodec / MediaMuxer 的原生单位）。
 */
#ifndef V2V_MEDIA_ENCODER_H
#define V2V_MEDIA_ENCODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ame_ctx ame_ctx;

/* 打开编码器 + MP4 封装器。
 * path：输出文件；bitrate：视频目标码率（bit/s）；fps：整数帧率；
 * i_frame_interval_s：GOP 秒数（MediaCodec 的原生单位就是秒）；
 * has_audio：会不会加音频轨 —— 声明过的，MediaMuxer 会等音频 track 加完才 start。 */
struct ame_ctx *ame_open(const char *path, int w, int h, int bitrate, int fps,
                         int i_frame_interval_s, int has_audio);

/* 加音频轨。asc：AudioSpecificConfig（2 字节），MediaMuxer 要它当 csd-0。
 * 必须在第一帧音频样本之前、可以在视频编码中途调用。 */
int ame_add_audio_track(struct ame_ctx *c, const uint8_t *asc, int asc_len,
                        int sample_rate, int channels);

/* 喂一帧 YUV420P（三平面紧排，stride = width）。pts_us：微秒。
 * force_keyframe：非 0 时尽量让这一帧是 IDR（API 26+，失败则静默忽略）。 */
int ame_write_video(struct ame_ctx *c, const uint8_t *y, const uint8_t *u,
                    const uint8_t *v, int64_t pts_us, int force_keyframe);

/* 喂一段裸 AAC 帧（无 ADTS 头）。 */
int ame_write_audio(struct ame_ctx *c, const uint8_t *data, int len, int64_t pts_us);

/* 发 EOS、吸干编码器、封 MP4（写 moov）。之后句柄只能 ame_free。 */
int ame_finish(struct ame_ctx *c);

/* 最后一次错误的描述；ctx 为空返回常量字符串。 */
const char *ame_last_error(struct ame_ctx *c);

void ame_free(struct ame_ctx *c);

#ifdef __cplusplus
}
#endif

#endif /* V2V_MEDIA_ENCODER_H */
