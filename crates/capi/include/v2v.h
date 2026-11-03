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

/* YUV420P 三平面 —— 解码 / 编码 / 播放三层共用。
 *
 * stride 恒等于 width（色度是 ceil(width/2)）—— 我们没有 padding，每行紧排，
 * 所以想留一帧就是逐行 memcpy。
 * 三个指针都是**借用**的：活到本句柄的下一次同类调用。 */
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

/* ==================================================================
 * 容器（L0 嗅探 / L1 解封装）
 *
 * ASF 与 AVI 共用**同一套符号**：它们的动词本来就一样（开、列流、取下一个样本），
 * 吐出来的也是同一个样本结构。加新容器 = 加一个后端，这套符号不动。
 *
 * 样本的 data 是**借用**的：活到下一次 v2v_demux_next / close。
 * 典型用法是「拿到就解」，所以这里刻意不做「交所有权 + free」。
 * ================================================================== */

#define V2V_CONTAINER_UNKNOWN 0
#define V2V_CONTAINER_ASF 1
#define V2V_CONTAINER_AVI 2
/* MPEG-1/2 program stream。认得出，但**不在这套符号里**（形状不同，见 v2v_mpeg_*）。 */
#define V2V_CONTAINER_MPEG_PS 3

#define V2V_STREAM_AUDIO 0
#define V2V_STREAM_VIDEO 1
#define V2V_STREAM_COMMAND 2
#define V2V_STREAM_JFIF 3
#define V2V_STREAM_UNKNOWN 4

typedef struct v2v_demux v2v_demux;

/* 一条流的静态信息。视频字段与音频字段在同一块里，用不到的为 0。
 *
 * extra 是编解码器私有数据（VC-1 的序列头 / WMA 的 params 都在这里），
 * 借用，活到下一次 v2v_demux_stream / close。 */
typedef struct v2v_stream_info {
    uint32_t size;
    uint32_t number; /* 容器内的流号，只在同一容器内可比 */
    uint32_t kind;   /* V2V_STREAM_* */
    uint64_t time_offset_ms;

    /* 视频 */
    uint32_t width;
    uint32_t height;
    uint32_t encoded_width;
    uint32_t encoded_height;
    uint32_t fourcc;
    uint32_t bits_per_pixel;

    /* 音频 */
    uint32_t format_tag; /* 0x0161 = WMA v2, 0x0160 = WMA v1, 0x0001 = PCM */
    uint32_t sample_rate;
    uint32_t avg_bytes_per_sec;
    uint32_t channels;
    uint32_t block_align;
    uint32_t bits_per_sample;

    const uint8_t *extra;
    uint32_t extra_len;
} v2v_stream_info;

/* 一个已重组的样本（视频一帧 / 音频一帧）。 */
typedef struct v2v_sample {
    uint32_t size;
    uint32_t stream; /* 与 v2v_stream_info.number 对应 */
    int32_t key_frame;
    int64_t timestamp_ms; /* ASF 已扣 preroll；AVI 从 0 起 */
    const uint8_t *data;  /* 借用，活到下一次 next / close */
    uint32_t data_len;
} v2v_sample;

/* 索引（只有 ASF 有）。槽 i = 「seek 到第 i 槽该从哪个包起读」。
 * 用法：slot = ms / interval_ms; 然后 v2v_demux_seek_packet(packets[slot])。 */
typedef struct v2v_index {
    uint32_t size;
    uint32_t interval_ms;
    uint32_t count;
    const uint32_t *packets; /* 借用 */
} v2v_index;

/* 嗅探容器种类。认不出来返回 V2V_CONTAINER_UNKNOWN（不是错误）。 */
int32_t v2v_probe(const uint8_t *head, uint32_t len);

/* 打开（自动嗅探）。失败返回 NULL。 */
v2v_demux *v2v_demux_open(const char *path);

void v2v_demux_close(v2v_demux *h);

/* 有几条流。 */
uint32_t v2v_demux_stream_count(const v2v_demux *h);

/* 取第 i 条流的信息。越界返回 V2V_ERR_ARG。 */
int32_t v2v_demux_stream(v2v_demux *h, uint32_t i, v2v_stream_info *out);

/* 取下一个样本（所有轨混在一起，按容器里的顺序）。V2V_EOS = 到头。 */
int32_t v2v_demux_next(v2v_demux *h, v2v_sample *out);

/* 把读游标挪到第 packet 个包。AVI 没有包级索引 → V2V_ERR_UNSUPPORTED。 */
int32_t v2v_demux_seek_packet(v2v_demux *h, uint64_t packet);

/* 取索引。V2V_EOS = 这个文件没有索引。 */
int32_t v2v_demux_index(v2v_demux *h, v2v_index *out);

/* ==================================================================
 * 视频解码（L2）
 *
 * 一包字节进，一帧 YUV 出。解码器是**有状态**的（P 帧参考前一帧）：
 * 必须按显示顺序喂、不能跳帧、不能并发用。
 * 要「从某个关键帧重来」就关掉重开一个句柄 —— 内部的参考帧没法半途清零。
 *
 * 不做错误隐藏：解不出来就返回错误，不会给一帧「大概是这样」的画面。
 * ================================================================== */

#define V2V_CODEC_WMV3 0 /* VC-1 */
#define V2V_CODEC_MP43 1 /* MS-MPEG-4 v3 */
#define V2V_CODEC_MP42 2 /* MS-MPEG-4 v2 */
#define V2V_CODEC_WMV1 3 /* WMV7 */
#define V2V_CODEC_WMV2 4 /* WMV8，必须给 extradata */
#define V2V_CODEC_MP4V 5 /* MPEG-4 Part 2；extra 为空则按「VOL 在码流里」 */

typedef struct v2v_vdec v2v_vdec;

typedef struct v2v_picture {
    uint32_t size;
    v2v_planes planes;
} v2v_picture;

/* 建解码器。extra 用 v2v_stream_info.extra 原样传；width/height 用**显示尺寸**。
 * 失败返回 NULL。 */
v2v_vdec *v2v_vdec_open(uint32_t codec, const uint8_t *extra, uint32_t extra_len,
                        uint32_t width, uint32_t height);

/* 解一帧。data 是容器里的一个样本（裸帧数据，帧头在最前面）。 */
int32_t v2v_vdec_decode(v2v_vdec *h, const uint8_t *data, uint32_t len, v2v_picture *out);

void v2v_vdec_close(v2v_vdec *h);

/* ==================================================================
 * 音频解码（L2）：WMA v1 / v2
 *
 * 为什么是「喂一个载荷 + 逐个取帧」两件事：ASF 的一个音频载荷里常装着
 * 好几个 WMA 帧（superframe），解码器一次吐一串。硬做成「一次调用 = 一帧」
 * 会让调用方要么丢帧、要么自己攒。
 *
 * 注意：v2v_adec_next 返回 V2V_EOS 只表示**队列空了**（该喂下一个载荷），
 * 不是流结束。
 * ================================================================== */

typedef struct v2v_adec v2v_adec;

typedef struct v2v_adec_cfg {
    uint32_t size;
    uint32_t format_tag; /* 0x0160 = WMA v1, 0x0161 = WMA v2 */
    uint32_t channels;
    uint32_t sample_rate;
    uint32_t avg_bytes_per_sec;
    uint32_t block_align;
    const uint8_t *extra;
    uint32_t extra_len;
    /* 1 = 丢掉编码器开头的静音（frame_len * 2 个采样点），与 ffmpeg 一致；
     * 不丢的话音频整体晚约 90 ms。建议开。 */
    uint32_t drop_priming;
} v2v_adec_cfg;

typedef struct v2v_adec_frame {
    uint32_t size;
    const float *const *planes; /* 借用，活到下次 next / close */
    uint32_t channels;
    uint32_t samples; /* 每声道 */
    uint32_t sample_rate;
    int64_t pts_ms; /* 按已交出的样本数推出来的 */
} v2v_adec_frame;

v2v_adec *v2v_adec_open(const v2v_adec_cfg *cfg);

void v2v_adec_close(v2v_adec *h);

/* 喂一个载荷（容器里的一个音频样本），解出来的帧进队列。 */
int32_t v2v_adec_feed(v2v_adec *h, const uint8_t *data, uint32_t len);

/* 取一帧。V2V_EOS = 队列空了（不是流结束）。 */
int32_t v2v_adec_next(v2v_adec *h, v2v_adec_frame *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* V2V_H */
