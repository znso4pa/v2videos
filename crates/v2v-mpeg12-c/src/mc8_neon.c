/* SPDX-License-Identifier: MIT */
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

/* Rust checks the complete input span. dst points to a separate 8×8 block. */
__attribute__((visibility("hidden")))
void v2v_mpeg12_mc8_neon(const uint8_t *src, size_t step, uint8_t phase, uint8_t *dst) {
    switch (phase) {
    case 0:
        for (size_t row = 0; row < 8; ++row)
            vst1_u8(dst + row * 8, vld1_u8(src + row * step));
        break;
    case 1:
        for (size_t row = 0; row < 8; ++row) {
            const uint8_t *top = src + row * step;
            vst1_u8(dst + row * 8, vrhadd_u8(vld1_u8(top), vld1_u8(top + 1)));
        }
        break;
    case 2:
        for (size_t row = 0; row < 8; ++row) {
            const uint8_t *top = src + row * step;
            vst1_u8(dst + row * 8, vrhadd_u8(vld1_u8(top), vld1_u8(top + step)));
        }
        break;
    case 3:
        for (size_t row = 0; row < 8; ++row) {
            const uint8_t *top = src + row * step;
            const uint8_t *bottom = top + step;
            uint16x8_t sum = vaddl_u8(vld1_u8(top), vld1_u8(top + 1));
            sum = vaddw_u8(sum, vld1_u8(bottom));
            sum = vaddw_u8(sum, vld1_u8(bottom + 1));
            vst1_u8(dst + row * 8, vrshrn_n_u16(sum, 2));
        }
        break;
    }
}
