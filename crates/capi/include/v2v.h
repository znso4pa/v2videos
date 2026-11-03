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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* V2V_H */
