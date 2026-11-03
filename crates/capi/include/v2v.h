/*
 * v2v.h —— v2video 全仓的 C ABI。
 *
 * 一份头文件，一层一组符号：容器 / 解码 / 编码 / 封装 / 播放 / 转码。
 * 只用 C11 标准头，不需要 Rust 工具链就能编过。
 *
 * ============================ 全局约定 ============================
 *
 * 1) 返回码
 *      V2V_OK            0   成功
 *      V2V_ERR_ARG       1   调用方参数不对（空指针 / 非 UTF-8 / size 太小）
 *      V2V_ERR_UNSUPPORTED 2 结构合法，但本实现不支持
 *      V2V_ERR_FAILED    3   运行期失败（IO / 数据坏 / 内部 panic）
 *      V2V_EOS           4   已经到头（不是错误，只用于「取下一个」）
 *    失败时用 v2v_last_error() 拿消息、v2v_last_error_kind() 拿种类。
 *
 * 2) 错误通道是**每线程**的
 *    v2v_last_error() 返回的指针活到「本线程的下一次 ABI 调用」。
 *    别跨线程读，别存起来。成功时返回空串（不是空指针）。
 *
 * 3) 所有权：谁分配谁释放
 *    库交给你的指针，要么「活到下次同类调用」（借用，见各函数注释），
 *    要么配一个 v2v_*_free。**永远不要 free 库给的内存。**
 *
 * 4) 出参结构体首字段是 size
 *    调用方先填 `size = sizeof(结构体)`，库按 min(size, 库自己的大小) 写。
 *    于是库加字段对老调用方安全，调用方比库新也安全。
 *    不填（size = 0）会返回 V2V_ERR_ARG。
 *
 * 5) 句柄（v2v_* 的 typedef 都是不透明指针）
 *    一个句柄**只能在一个线程用**（内部不是 Sync）。要并发就开多个句柄 ——
 *    它们互不影响，同一个文件开两个也完全可以。
 *
 * 6) 像素与采样
 *    像素一律 YUV420P 三平面分开；**stride 恒等于 width**（我们没有 padding，
 *    每行紧排）。音频一律每声道一条 float，刻度同 int16（不是 [-1, 1)），
 *    并且**可能超出 ±32768** —— 转 int16 的调用方必须自己 clamp。
 *
 * 7) 版本
 *    v2v_abi_version() 返回 (major << 16) | minor。major 不同就别继续。
 */

#ifndef V2V_H
#define V2V_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define V2V_ABI_MAJOR 1
#define V2V_ABI_MINOR 0

/* ---- 返回码 ---- */
#define V2V_OK 0
#define V2V_ERR_ARG 1
#define V2V_ERR_UNSUPPORTED 2
#define V2V_ERR_FAILED 3
#define V2V_EOS 4

/* ---- v2v_last_error_kind() ---- */
#define V2V_KIND_IO 0
#define V2V_KIND_BAD 1
#define V2V_KIND_UNSUPPORTED 2

/* ==================================================================
 * 公共
 * ================================================================== */

/* (major << 16) | minor。open 之前比一次。 */
uint32_t v2v_abi_version(void);

/* 本线程最后一次失败的消息（UTF-8，NUL 结尾）。活到本线程的下一次 ABI 调用。 */
const char *v2v_last_error(void);

/* 本线程最后一次失败的种类。 */
int32_t v2v_last_error_kind(void);

/* 出参结构体统一用这个初始化（填 size）。忘了填会拿到 V2V_ERR_ARG。 */
#define V2V_INIT(s) ((s).size = (uint32_t)sizeof(s))

/* ==================================================================
 * 播放（L5）—— 打开一个文件，按自己的节奏交替拉视频帧与音频帧
 *
 * 这一层已经把「一个载荷里装着好几帧音频」「编码器延迟的静音」
 * 「seek 之后解码器重建」处理掉了。想自己交织的话用下面的解封装 + 解码层。
 *
 * 两条轨**互不阻塞**：先拉谁、拉多少都行；seek 会把两轨一起挪。
 * ================================================================== */

typedef struct v2v_player v2v_player;

/* v2v_info.flags 的位。
 *
 * HAS_B_FRAMES       码流声明了 B 帧。本实现只做 I / P —— 置位即「播不了」，
 *                    不是「会慢一点」。请在 open 之后先看一眼再决定播不播。
 * INTERLACED         恒为 0：Main profile 的序列头不携带隔行信息（FCM 在每帧
 *                    的帧头里），open 时判断不出来。隔行码流本实现也不支持。
 * ADVANCED_PROFILE   恒为 0：Advanced 的序列头在码流里而不是 extradata 里，
 *                    open 阶段就已经返回 V2V_ERR_UNSUPPORTED 了。 */
#define V2V_INFO_HAS_B_FRAMES 0x1u
#define V2V_INFO_INTERLACED 0x2u
#define V2V_INFO_ADVANCED_PROFILE 0x4u

/* YUV420P 三平面。
 *
 * stride 恒等于 width（色度是 ceil(width/2)）—— 我们没有 padding，每行紧排。
 * 三个指针活到本句柄的下一次 read_video / seek_ms / close。 */
typedef struct v2v_planes {
    const uint8_t *y;
    const uint8_t *u;
    const uint8_t *v;
    uint32_t width;
    uint32_t height;
    uint32_t stride_y;
    uint32_t stride_u;
    uint32_t stride_v;
} v2v_planes;

/* 一帧视频。pts_ms 是容器时间戳（已扣 preroll）。 */
typedef struct v2v_frame {
    uint32_t size;
    int64_t pts_ms;
    int32_t key_frame; /* 0 / 1；seek 的落点候选就是它 */
    v2v_planes planes;
} v2v_frame;

/* 一帧音频。
 *
 * 刻度同 int16（不是 [-1, 1)），并且**可能超出 ±32768** —— 转 int16 时必须
 * 自己 clamp。pts_ms 是按已交出的样本数推出来的，不是容器时间戳。
 * planes 指向 channels 个指针，每条 samples 个样本，活到下次 read_audio。 */
typedef struct v2v_audio_frame {
    uint32_t size;
    int64_t pts_ms;
    const float *const *planes;
    uint32_t channels;
    uint32_t samples;
    uint32_t sample_rate;
} v2v_audio_frame;

/* 流的静态信息。audio_channels == 0 表示没有音频轨。 */
typedef struct v2v_info {
    uint32_t size;
    uint64_t duration_ms;
    uint32_t width;
    uint32_t height;
    uint32_t video_fourcc; /* 如 'WMV3' 的字节小端 */
    uint32_t audio_sample_rate;
    uint32_t audio_channels;
    uint32_t has_audio;
    uint32_t flags; /* V2V_INFO_* */
} v2v_info;

/* 打开一个文件。失败返回 NULL（原因看 v2v_last_error）。 */
v2v_player *v2v_player_open(const char *path);

/* 关闭并释放。传 NULL 是空操作。 */
void v2v_player_close(v2v_player *h);

/* 取静态信息。 */
int32_t v2v_player_info(v2v_player *h, v2v_info *out);

/* 取下一帧视频。V2V_EOS = 视频轨到头。
 * 出参里的指针活到本句柄的下一次 read_video / seek_ms / close。 */
int32_t v2v_player_read_video(v2v_player *h, v2v_frame *out);

/* 取下一帧音频。V2V_EOS = 音频轨到头（或本来就没有音频）。 */
int32_t v2v_player_read_audio(v2v_player *h, v2v_audio_frame *out);

/* seek 到 ms，返回实际落点（关键帧时间戳，<= ms，最多差一个 GOP）；失败 -1。
 * 落点之后下一次 read_video 交出的就是落点那一帧。 */
int64_t v2v_player_seek_ms(v2v_player *h, uint64_t ms);

/* 已交出的最后一帧视频的时间戳（ms）。没取过帧时是 0。 */
int64_t v2v_player_position_ms(const v2v_player *h);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* V2V_H */
