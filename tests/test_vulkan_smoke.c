/* tests/test_vulkan_smoke.c -- Vulkan backend smoke test (Fase 0..2).
 *
 * Verifies the core lifecycle, tensor alloc/fill/read/write/copy, command
 * buffer round-trips, and the Fase 1 elementwise kernels (add/add3).  Fase 2
 * adds roundtrips for the kernel core: swiglu, argmax, sort_i32_rows_asc,
 * RMS norm (plain/weight, single and multi-row), rope_tail, matmul
 * Q8_0/F16/F32, and Q8_0 token embeddings -- each checked against a CPU
 * reference over a wrapped model buffer (VK_EXT_external_memory_host).
 *
 * Links against the three Vulkan TUs (ds4_vulkan.o, ds4_vulkan_compat.o,
 * ds4_vulkan_unavailable.o). */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "../vulkan/shaders/iq2_tables_host.h"

static int failures = 0;

/* The streaming cache load-layer surface is ROCm/Apple-gated in ds4_gpu.h,
 * but the Vulkan backend provides it as the Fase 6 all-resident policy.
 * Prototypes here so the C test can exercise them. */
int ds4_gpu_stream_expert_cache_load_layer(
        const ds4_gpu_stream_expert_table *table);
int ds4_gpu_stream_expert_cache_seed_from_layer_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_tokens,
        uint32_t                           n_seed_tokens,
        uint32_t                           n_selected);
int ds4_gpu_stream_expert_cache_finish_pending_batch(void);
int ds4_gpu_stream_expert_cache_release_layer_cache(void);
int ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts);

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", msg); \
            failures++; \
        } else { \
            printf("ok: %s\n", msg); \
        } \
    } while (0)

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t f;
    if (exp == 0u) {
        if (man == 0u) {
            f = sign;
        } else {
            uint32_t m = man;
            int e = -14;
            while ((m & 0x400u) == 0u) { m <<= 1; e--; }
            f = sign | (uint32_t)(e + 127) << 23 | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 31u) {
        f = sign | 0x7f800000u | (man << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

static uint16_t f32_to_f16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t man = bits & 0x7fffffu;
    if (exp >= 31) return (uint16_t)(sign | 0x7bffu);
    if (exp <= 0) {
        const uint32_t mant = man | 0x800000u;
        const int shift = 1 - exp;
        return (uint16_t)(sign | (uint16_t)((mant >> (uint32_t)shift) >> 13));
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

/* Quantize `n` floats into a 34-byte Q8_0 block (same math as the shader and
 * CUDA lrintf: d = amax/127, q = round-half-to-even(x/d) clamped). */
static void quant_q8_block(uint8_t *dst, const float *src, uint32_t n) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < n; i++) amax = fmaxf(amax, fabsf(src[i]));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const uint16_t scale = f32_to_f16(d);
    memcpy(dst, &scale, 2);
    for (uint32_t i = 0; i < 32u; i++) {
        int v = 0;
        if (i < n) {
            v = (int)nearbyintf(src[i] * id);   /* round-half-to-even */
            if (v > 127) v = 127;
            if (v < -128) v = -128;
        }
        dst[2 + i] = (uint8_t)(int8_t)v;
    }
}

static float q8_dequant(const uint8_t *block, uint32_t lane) {
    uint16_t scale;
    memcpy(&scale, block, 2);
    return f16_to_f32(scale) * (float)(int8_t)block[2 + lane];
}

/* --- CPU references (match the shader math) ------------------------------ */

static void cpu_swiglu(float *out, const float *gate, const float *up,
                       uint32_t n, float clamp, float weight) {
    for (uint32_t i = 0; i < n; i++) {
        float g = gate[i];
        float u = up[i];
        if (clamp > 1.0e-6f) {
            g = fminf(g, clamp);
            u = fminf(fmaxf(u, -clamp), clamp);
        }
        out[i] = (g / (1.0f + expf(-g))) * u * weight;
    }
}

static void cpu_rms_norm(float *out, const float *x, const float *w,
                         uint32_t n, uint32_t rows, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < n; i++) sum += x[r * n + i] * x[r * n + i];
        const float scale = 1.0f / sqrtf(sum / (float)n + eps);
        for (uint32_t i = 0; i < n; i++) {
            out[r * n + i] = x[r * n + i] * scale *
                             (w ? w[i] : 1.0f);
        }
    }
}

static void cpu_rope_tail(float *x, uint32_t n_tok, uint32_t n_head,
                          uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
                          uint32_t n_ctx_orig, int inverse, float freq_base,
                          float freq_scale, float ext_factor,
                          float attn_factor, float beta_fast,
                          float beta_slow) {
    const uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot *
                       logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) /
                       denom);
        corr1 = ceilf((float)n_rot *
                      logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) /
                      denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t h = 0; h < n_head; h++) {
            float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
            for (uint32_t pair = 0; pair < n_rot / 2u; pair++) {
                const uint32_t i = pair * 2u;
                float theta_extrap =
                    (float)(pos0 + t) * powf(freq_base, -((float)i) / (float)n_rot);
                float theta_interp = freq_scale * theta_extrap;
                float theta = theta_interp;
                float mscale = attn_factor;
                if (ext_factor != 0.0f) {
                    const int i0 = (int)i;
                    const float y = ((float)(i0 / 2) - corr0) /
                                    fmaxf(0.001f, corr1 - corr0);
                    const float ramp =
                        (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
                    theta = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
                    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
                }
                const float c = cosf(theta) * mscale;
                float s = sinf(theta) * mscale;
                if (inverse) s = -s;
                const float x0 = tail[i];
                const float x1 = tail[i + 1];
                tail[i] = x0 * c - x1 * s;
                tail[i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

static void cpu_matmul_q8(float *out, const uint8_t *w, uint32_t out_dim,
                          uint32_t in_dim, const float *x, uint32_t n_tok) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            float acc = 0.0f;
            for (uint32_t b = 0; b < blocks; b++) {
                const uint32_t i0 = b * 32u;
                const uint32_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
                const uint8_t *blk = w + (uint64_t)(row * blocks + b) * 34u;
                uint16_t scale;
                memcpy(&scale, blk, 2);
                const float sw = f16_to_f32(scale);
                float amax = 0.0f;
                for (uint32_t i = 0; i < bn; i++) {
                    amax = fmaxf(amax, fabsf(x[tok * in_dim + i0 + i]));
                }
                const float d = amax / 127.0f;
                const float id = d != 0.0f ? 1.0f / d : 0.0f;
                int dot = 0;
                for (uint32_t i = 0; i < bn; i++) {
                    int q = (int)nearbyintf(x[tok * in_dim + i0 + i] * id);
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;
                    dot += (int)(int8_t)blk[2 + i] * q;
                }
                acc += sw * d * (float)dot;
            }
            out[tok * out_dim + row] = acc;
        }
    }
}

/* Partial-K Q8_0 reference: weight rows span full_blocks; only the
 * [block_start, block_start+slice_blocks) slice is used, against a k_cnt
 * element activation row (x). */
static void cpu_matmul_q8_kslice(float *out, const uint8_t *w, uint32_t out_dim,
                                 uint32_t full_blocks, uint32_t block_start,
                                 uint32_t slice_blocks, uint32_t k_cnt,
                                 const float *x, uint32_t n_tok) {
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            float acc = 0.0f;
            for (uint32_t sb = 0; sb < slice_blocks; sb++) {
                const uint32_t wblock = row * full_blocks + block_start + sb;
                const uint32_t i0 = sb * 32u;
                const uint32_t bn = k_cnt - i0 < 32u ? k_cnt - i0 : 32u;
                const uint8_t *blk = w + (uint64_t)wblock * 34u;
                uint16_t scale;
                memcpy(&scale, blk, 2);
                const float sw = f16_to_f32(scale);
                float amax = 0.0f;
                for (uint32_t i = 0; i < bn; i++) {
                    amax = fmaxf(amax, fabsf(x[tok * k_cnt + i0 + i]));
                }
                const float d = amax / 127.0f;
                const float id = d != 0.0f ? 1.0f / d : 0.0f;
                int dot = 0;
                for (uint32_t i = 0; i < bn; i++) {
                    int q = (int)nearbyintf(x[tok * k_cnt + i0 + i] * id);
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;
                    dot += (int)(int8_t)blk[2 + i] * q;
                }
                acc += sw * d * (float)dot;
            }
            out[tok * out_dim + row] = acc;
        }
    }
}

static void cpu_matmul_f16(float *out, const uint16_t *w, uint32_t out_dim,
                           uint32_t in_dim, const float *x, uint32_t n_tok) {
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < in_dim; k++) {
                acc += f16_to_f32(w[row * in_dim + k]) * x[tok * in_dim + k];
            }
            out[tok * out_dim + row] = acc;
        }
    }
}

static void cpu_matmul_f32(float *out, const float *w, uint32_t out_dim,
                           uint32_t in_dim, const float *x, uint32_t n_tok) {
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < in_dim; k++) {
                acc += w[row * in_dim + k] * x[tok * in_dim + k];
            }
            out[tok * out_dim + row] = acc;
        }
    }
}

/* Dense Q4_K matmul: out[tok][row] = dot(dequant_q4k(w[row]), x[tok]). */
static void cpu_matmul_q4k(float *out, const uint8_t *w, uint32_t out_dim,
                           uint32_t in_dim, const float *x, uint32_t n_tok) {
    const uint32_t blocks = in_dim / 256u;
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            const uint8_t *wrow = w + (uint64_t)row * blocks * 144u;
            float acc = 0.0f;
            for (uint32_t b = 0; b < blocks; b++) {
                const uint8_t *blk = wrow + (uint64_t)b * 144u;
                uint16_t dh, dmh;
                memcpy(&dh, blk, 2);
                memcpy(&dmh, blk + 2, 2);
                const float d = f16_to_f32(dh);
                const float dmin = f16_to_f32(dmh);
                for (uint32_t j = 0; j < 8u; j++) {
                    uint32_t sc, m;
                    if (j < 4u) {
                        sc = blk[4u + j] & 63u;
                        m = blk[4u + j + 4u] & 63u;
                    } else {
                        const uint32_t a = blk[4u + j + 4u];
                        const uint32_t bb = blk[4u + j - 4u];
                        const uint32_t c = blk[4u + j];
                        sc = (a & 0xfu) | ((bb >> 6u) << 4u);
                        m = (a >> 4u) | ((c >> 6u) << 4u);
                    }
                    const float scale = d * (float)sc;
                    const float minv = dmin * (float)m;
                    const uint32_t byte_off = (j >> 1u) * 32u;
                    const uint32_t shift = (j & 1u) * 4u;
                    for (uint32_t l = 0; l < 32u; l++) {
                        const uint32_t qb = blk[16u + byte_off + l];
                        const uint32_t q = (qb >> shift) & 0xfu;
                        acc += (scale * (float)q - minv) *
                               x[tok * in_dim + b * 256u + j * 32u + l];
                    }
                }
            }
            out[tok * out_dim + row] = acc;
        }
    }
}

/* Dense Q4_0 matmul: out[tok][row] = dot(dequant_q4_0(w[row]), x[tok]).
 * Elements 0..15 are the low nibbles of qs[0..15], 16..31 the high. */
static void cpu_matmul_q4_0(float *out, const uint8_t *w, uint32_t out_dim,
                            uint32_t in_dim, const float *x, uint32_t n_tok) {
    const uint32_t blocks = in_dim / 32u;
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            const uint8_t *wrow = w + (uint64_t)row * blocks * 18u;
            float acc = 0.0f;
            for (uint32_t b = 0; b < blocks; b++) {
                const uint8_t *blk = wrow + (uint64_t)b * 18u;
                uint16_t dh;
                memcpy(&dh, blk, 2);
                const float d = f16_to_f32(dh);
                for (uint32_t j = 0; j < 16u; j++) {
                    const uint8_t packed = blk[2u + j];
                    acc += d * ((float)(packed & 0xfu) - 8.0f) *
                           x[tok * in_dim + b * 32u + j];
                    acc += d * ((float)(packed >> 4u) - 8.0f) *
                           x[tok * in_dim + b * 32u + j + 16u];
                }
            }
            out[tok * out_dim + row] = acc;
        }
    }
}

/* --- Fase 4: MoE CPU references (match the shader math) ------------------ */

static float cpu_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

/* Q8_0 dot of one weight row (Q8_0 blocks, in_dim elements) against an
 * f32 activation row at x_off.  row_off is the byte offset of the row's
 * first block in the model buffer. */
static float cpu_dot_q8_row(const uint8_t *model, uint64_t row_off,
                            uint32_t in_dim, const float *x, uint32_t x_off) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint32_t i0 = b * 32u;
        const uint32_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const uint8_t *blk = model + row_off + (uint64_t)b * 34u;
        uint16_t scale;
        memcpy(&scale, blk, 2);
        const float sw = f16_to_f32(scale);
        float amax = 0.0f;
        for (uint32_t i = 0; i < bn; i++) {
            amax = fmaxf(amax, fabsf(x[x_off + i0 + i]));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        int dot = 0;
        for (uint32_t i = 0; i < bn; i++) {
            int q = (int)nearbyintf(x[x_off + i0 + i] * id);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            dot += (int)(int8_t)blk[2 + i] * q;
        }
        acc += sw * d * (float)dot;
    }
    return acc;
}

/* --- IQ2_XXS / Q2_K CPU references (routed MoE, Fase 6) ------------------ */

/* Dequant an IQ2_XXS row (66-byte 256-value blocks) and dot with x.
 * Mirrors dequantize_row_iq2_xxs; the lookup tables come from
 * iq2_tables_host.h (generated from cuda/mmq/ggml-common.h). */
static float cpu_dot_iq2_row(const uint8_t *model, uint64_t row_off,
                             uint32_t in_dim, const float *x,
                             uint32_t x_off) {
    const uint32_t blocks = in_dim / 256u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *blk = model + row_off + (uint64_t)b * 66u;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = f16_to_f32(dh);
        for (uint32_t sub = 0; sub < 8u; sub++) {
            uint32_t aux32[2];
            memcpy(aux32, blk + 2 + 8u * sub, 8);
            const uint8_t *aux8 = (const uint8_t *)aux32;
            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
            for (uint32_t l = 0; l < 4u; l++) {
                const uint64_t g = cpu_iq2xxs_grid[aux8[l]];
                const uint8_t signs =
                    cpu_ksigns_iq2xs[(aux32[1] >> (7u * l)) & 127u];
                const uint32_t e0 = sub * 32u + l * 8u;
                for (uint32_t j = 0; j < 8u; j++) {
                    const int8_t gv = (int8_t)((g >> (j * 8u)) & 0xffu);
                    float w = db * (float)gv;
                    if (signs & (1u << j)) w = -w;
                    acc += w * x[x_off + b * 256u + e0 + j];
                }
            }
        }
    }
    return acc;
}

/* Dequant a Q2_K row (84-byte 256-value blocks) and dot with x.
 * Mirrors dequantize_row_q2_K. */
static float cpu_dot_q2k_row(const uint8_t *model, uint64_t row_off,
                             uint32_t in_dim, const float *x,
                             uint32_t x_off) {
    const uint32_t blocks = in_dim / 256u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *blk = model + row_off + (uint64_t)b * 84u;
        uint16_t dh, dhmin;
        memcpy(&dh, blk + 80, 2);
        memcpy(&dhmin, blk + 82, 2);
        const float d = f16_to_f32(dh);
        const float dmin = f16_to_f32(dhmin);
        for (uint32_t e = 0; e < 256u; e++) {
            const uint32_t half = e >> 7;
            const uint32_t p = e & 127u;
            const uint32_t j = p >> 5;
            const uint32_t l = p & 31u;
            const uint32_t sc_idx =
                half * 8u + j * 2u + (l >= 16u ? 1u : 0u);
            const uint8_t sc = blk[sc_idx];
            const float dl = d * (float)(sc & 0xfu);
            const float ml = dmin * (float)(sc >> 4u);
            const uint8_t qb = blk[16u + half * 32u + l];
            const float w = dl * (float)((qb >> (2u * j)) & 3u) - ml;
            acc += w * x[x_off + b * 256u + e];
        }
    }
    return acc;
}

/* Dequant a Q4_K row (144-byte 256-value blocks) and dot with x.
 * Mirrors ds4_vec_dot_q4_K_f32 / q4_k_get_scale_min. */
static float cpu_dot_q4k_row(const uint8_t *model, uint64_t row_off,
                             uint32_t in_dim, const float *x, uint32_t x_off) {
    const uint32_t blocks = in_dim / 256u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *blk = model + row_off + (uint64_t)b * 144u;
        uint16_t dh, dmh;
        memcpy(&dh, blk, 2);
        memcpy(&dmh, blk + 2, 2);
        const float d = f16_to_f32(dh);
        const float dmin = f16_to_f32(dmh);
        for (uint32_t j = 0; j < 8u; j++) {
            uint32_t sc, m;
            if (j < 4u) {
                sc = blk[4u + j] & 63u;
                m = blk[4u + j + 4u] & 63u;
            } else {
                const uint32_t a = blk[4u + j + 4u];
                const uint32_t bb = blk[4u + j - 4u];
                const uint32_t c = blk[4u + j];
                sc = (a & 0xfu) | ((bb >> 6u) << 4u);
                m = (a >> 4u) | ((c >> 6u) << 4u);
            }
            const float scale = d * (float)sc;
            const float minv = dmin * (float)m;
            const uint32_t byte_off = (j >> 1u) * 32u;
            const uint32_t shift = (j & 1u) * 4u;
            for (uint32_t l = 0; l < 32u; l++) {
                const uint32_t qb = blk[16u + byte_off + l];
                const uint32_t q = (qb >> shift) & 0xfu;
                acc += (scale * (float)q - minv) *
                       x[x_off + b * 256u + j * 32u + l];
            }
        }
    }
    return acc;
}

/* Routed MoE reference for Q4_K gate/up/down experts. */
static void cpu_routed_moe_q4k(
        float *out, float *gate, float *up, float *mid,
        const uint8_t *model, uint64_t gate_off, uint64_t up_off,
        uint64_t down_off, uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes, uint64_t down_expert_bytes,
        uint64_t down_row_bytes, uint32_t in_dim, uint32_t mid_dim,
        uint32_t out_dim, const int32_t *selected, const float *weights,
        const float *x, uint32_t n_tokens, uint32_t n_expert,
        float clamp) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t slot = 0; slot < n_expert; slot++) {
            const int32_t expert = selected[t * n_expert + slot];
            for (uint32_t r = 0; r < mid_dim; r++) {
                const uint64_t gbase =
                    (uint64_t)expert * gate_expert_bytes +
                    (uint64_t)r * gate_row_bytes;
                const float g = cpu_dot_q4k_row(model, gate_off + gbase,
                                                in_dim, x, t * in_dim);
                const float u = cpu_dot_q4k_row(model, up_off + gbase,
                                                in_dim, x, t * in_dim);
                float gc = g, uc = u;
                if (clamp > 1.0e-6f) {
                    gc = fminf(gc, clamp);
                    uc = fminf(fmaxf(uc, -clamp), clamp);
                }
                const uint64_t idx =
                    ((uint64_t)t * n_expert + slot) * mid_dim + r;
                gate[idx] = gc;
                up[idx] = uc;
                mid[idx] = (gc / (1.0f + expf(-gc))) * uc *
                           weights[t * n_expert + slot];
            }
        }
        for (uint32_t r = 0; r < out_dim; r++) {
            float acc = 0.0f;
            for (uint32_t slot = 0; slot < n_expert; slot++) {
                const int32_t expert = selected[t * n_expert + slot];
                const uint64_t dbase = (uint64_t)expert * down_expert_bytes +
                                       (uint64_t)r * down_row_bytes;
                acc += cpu_dot_q4k_row(model, down_off + dbase, mid_dim,
                                       mid + (uint64_t)t * n_expert * mid_dim,
                                       slot * mid_dim);
            }
            out[t * out_dim + r] = acc;
        }
    }
}

/* Dequant an MXFP4 row (17-byte 32-value blocks) and dot with x.
 * Mirrors ds4_vec_dot_mxfp4_f32 / ds4_e8m0_to_f32. */
static float cpu_dot_mxfp4_row(const uint8_t *model, uint64_t row_off,
                               uint32_t in_dim, const float *x, uint32_t x_off) {
    static const float vals[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    const uint32_t blocks = in_dim / 32u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *blk = model + row_off + (uint64_t)b * 17u;
        const uint32_t e = blk[0];
        const uint32_t bits = (e == 0u) ? 0x00400000u : (e << 23u);
        float d;
        memcpy(&d, &bits, 4);
        for (uint32_t j = 0; j < 16u; j++) {
            const uint8_t q = blk[1u + j];
            acc += d * vals[q & 0xfu] * x[x_off + b * 32u + j];
            acc += d * vals[q >> 4] * x[x_off + b * 32u + j + 16u];
        }
    }
    return acc;
}

/* Routed MoE reference for MXFP4 gate/up/down experts. */
static void cpu_routed_moe_mxfp4(
        float *out, float *gate, float *up, float *mid,
        const uint8_t *model, uint64_t gate_off, uint64_t up_off,
        uint64_t down_off, uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes, uint64_t down_expert_bytes,
        uint64_t down_row_bytes, uint32_t in_dim, uint32_t mid_dim,
        uint32_t out_dim, const int32_t *selected, const float *weights,
        const float *x, uint32_t n_tokens, uint32_t n_expert,
        float clamp) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t slot = 0; slot < n_expert; slot++) {
            const int32_t expert = selected[t * n_expert + slot];
            for (uint32_t r = 0; r < mid_dim; r++) {
                const uint64_t gbase =
                    (uint64_t)expert * gate_expert_bytes +
                    (uint64_t)r * gate_row_bytes;
                const float g = cpu_dot_mxfp4_row(model, gate_off + gbase,
                                                  in_dim, x, t * in_dim);
                const float u = cpu_dot_mxfp4_row(model, up_off + gbase,
                                                  in_dim, x, t * in_dim);
                float gc = g, uc = u;
                if (clamp > 1.0e-6f) {
                    gc = fminf(gc, clamp);
                    uc = fminf(fmaxf(uc, -clamp), clamp);
                }
                const uint64_t idx =
                    ((uint64_t)t * n_expert + slot) * mid_dim + r;
                gate[idx] = gc;
                up[idx] = uc;
                mid[idx] = (gc / (1.0f + expf(-gc))) * uc *
                           weights[t * n_expert + slot];
            }
        }
        for (uint32_t r = 0; r < out_dim; r++) {
            float acc = 0.0f;
            for (uint32_t slot = 0; slot < n_expert; slot++) {
                const int32_t expert = selected[t * n_expert + slot];
                const uint64_t dbase = (uint64_t)expert * down_expert_bytes +
                                       (uint64_t)r * down_row_bytes;
                acc += cpu_dot_mxfp4_row(model, down_off + dbase, mid_dim,
                                         mid + (uint64_t)t * n_expert * mid_dim,
                                         slot * mid_dim);
            }
            out[t * out_dim + r] = acc;
        }
    }
}

/* Routed MoE reference for IQ2_XXS gate/up + Q2_K down. */
static void cpu_routed_moe_iq2q2(
        float *out, float *gate, float *up, float *mid,
        const uint8_t *model, uint64_t gate_off, uint64_t up_off,
        uint64_t down_off, uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes, uint64_t down_expert_bytes,
        uint64_t down_row_bytes, uint32_t in_dim, uint32_t mid_dim,
        uint32_t out_dim, const int32_t *selected, const float *weights,
        const float *x, uint32_t n_tokens, uint32_t n_expert,
        float clamp) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t slot = 0; slot < n_expert; slot++) {
            const int32_t expert = selected[t * n_expert + slot];
            for (uint32_t r = 0; r < mid_dim; r++) {
                const uint64_t gbase =
                    (uint64_t)expert * gate_expert_bytes +
                    (uint64_t)r * gate_row_bytes;
                const float g = cpu_dot_iq2_row(model, gate_off + gbase,
                                                in_dim, x, t * in_dim);
                const float u = cpu_dot_iq2_row(model, up_off + gbase,
                                                in_dim, x, t * in_dim);
                float gc = g, uc = u;
                if (clamp > 1.0e-6f) {
                    gc = fminf(gc, clamp);
                    uc = fminf(fmaxf(uc, -clamp), clamp);
                }
                const uint64_t idx =
                    ((uint64_t)t * n_expert + slot) * mid_dim + r;
                gate[idx] = gc;
                up[idx] = uc;
                mid[idx] = (gc / (1.0f + expf(-gc))) * uc *
                           weights[t * n_expert + slot];
            }
        }
        for (uint32_t r = 0; r < out_dim; r++) {
            float acc = 0.0f;
            for (uint32_t slot = 0; slot < n_expert; slot++) {
                const int32_t expert = selected[t * n_expert + slot];
                const uint64_t dbase = (uint64_t)expert * down_expert_bytes +
                                       (uint64_t)r * down_row_bytes;
                acc += cpu_dot_q2k_row(model, down_off + dbase, mid_dim,
                                       mid + (uint64_t)t * n_expert * mid_dim,
                                       slot * mid_dim);
            }
            out[t * out_dim + r] = acc;
        }
    }
}

static void cpu_shared_gate_up_swiglu(float *gate_out, float *up_out,
                                      float *mid_out,
                                      const uint8_t *model,
                                      uint64_t gate_off, uint64_t up_off,
                                      uint32_t in_dim, uint32_t out_dim,
                                      const float *x, uint32_t n_tok,
                                      float clamp) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            const float g = cpu_dot_q8_row(model, gate_off + (uint64_t)r * row_bytes,
                                           in_dim, x, t * in_dim);
            const float u = cpu_dot_q8_row(model, up_off + (uint64_t)r * row_bytes,
                                           in_dim, x, t * in_dim);
            float gc = g, uc = u;
            if (clamp > 1.0e-6f) {
                gc = fminf(gc, clamp);
                uc = fminf(fmaxf(uc, -clamp), clamp);
            }
            if (gate_out) gate_out[t * out_dim + r] = g;
            if (up_out) up_out[t * out_dim + r] = u;
            if (mid_out) mid_out[t * out_dim + r] =
                (gc / (1.0f + expf(-gc))) * uc;
        }
    }
}

static void cpu_router_select(int32_t *sel_out, float *w_out, float *prob_out,
                              const float *logits, const float *bias,
                              const int32_t *hash, uint32_t hash_rows,
                              const int32_t *tokens, int32_t tok_scalar,
                              uint32_t n_tokens, int has_bias, int hash_mode,
                              float scale) {
    const uint32_t n_expert = 256u, n_used = 6u;
    for (uint32_t t = 0; t < n_tokens; t++) {
        float prob[256];
        for (uint32_t i = 0; i < n_expert; i++) {
            prob[i] = sqrtf(cpu_softplus(logits[t * n_expert + i]));
            if (prob_out) prob_out[t * n_expert + i] = prob[i];
        }
        int32_t sel[6];
        if (hash_mode) {
            int32_t tok = tokens ? tokens[t] : tok_scalar;
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            for (uint32_t j = 0; j < n_used; j++) {
                sel[j] = hash[(uint64_t)tok * n_used + j];
            }
        } else {
            for (uint32_t j = 0; j < n_used; j++) sel[j] = -1;
            for (uint32_t e = 0; e < n_expert; e++) {
                const float score =
                    prob[e] + (has_bias ? bias[e] : 0.0f);
                for (uint32_t j = 0; j < n_used; j++) {
                    const int32_t cur = sel[j];
                    const float cur_score = cur >= 0
                        ? prob[(uint32_t)cur] +
                              (has_bias ? bias[(uint32_t)cur] : 0.0f)
                        : 0.0f;
                    if (cur < 0 || score > cur_score) {
                        for (uint32_t k = n_used - 1u; k > j; k--) {
                            sel[k] = sel[k - 1];
                        }
                        sel[j] = (int32_t)e;
                        break;
                    }
                }
            }
        }
        float sum = 0.0f;
        for (uint32_t j = 0; j < n_used; j++) {
            const int32_t e = sel[j];
            const float v = (e >= 0 && (uint32_t)e < n_expert)
                                ? prob[(uint32_t)e] : 0.0f;
            sel_out[t * n_used + j] = e;
            w_out[t * n_used + j] = v;
            sum += v;
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        for (uint32_t j = 0; j < n_used; j++) {
            w_out[t * n_used + j] = w_out[t * n_used + j] / sum * scale;
        }
    }
}

/* Routed MoE reference: gate/up/mid weighted SwiGLU per (token, slot),
 * then the per-expert down projections summed into out. */
static void cpu_routed_moe(float *out, float *gate, float *up, float *mid,
                           const uint8_t *model,
                           uint64_t gate_off, uint64_t up_off,
                           uint64_t down_off,
                           uint64_t gate_expert_bytes,
                           uint64_t gate_row_bytes,
                           uint64_t down_expert_bytes,
                           uint64_t down_row_bytes,
                           uint32_t in_dim, uint32_t mid_dim,
                           uint32_t out_dim, const int32_t *selected,
                           const float *weights, const float *x,
                           uint32_t n_tokens, uint32_t n_expert,
                           float clamp) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t slot = 0; slot < n_expert; slot++) {
            const int32_t expert = selected[t * n_expert + slot];
            for (uint32_t r = 0; r < mid_dim; r++) {
                const uint64_t gbase =
                    (uint64_t)expert * gate_expert_bytes + (uint64_t)r * gate_row_bytes;
                const uint64_t ubase =
                    (uint64_t)expert * gate_expert_bytes + (uint64_t)r * gate_row_bytes;
                const float g = cpu_dot_q8_row(model, gate_off + gbase,
                                               in_dim, x, t * in_dim);
                const float u = cpu_dot_q8_row(model, up_off + ubase,
                                               in_dim, x, t * in_dim);
                float gc = g, uc = u;
                if (clamp > 1.0e-6f) {
                    gc = fminf(gc, clamp);
                    uc = fminf(fmaxf(uc, -clamp), clamp);
                }
                const uint64_t idx = ((uint64_t)t * n_expert + slot) * mid_dim + r;
                gate[idx] = gc;
                up[idx] = uc;
                mid[idx] = (gc / (1.0f + expf(-gc))) * uc *
                           weights[t * n_expert + slot];
            }
        }
        for (uint32_t r = 0; r < out_dim; r++) {
            float acc = 0.0f;
            for (uint32_t slot = 0; slot < n_expert; slot++) {
                const int32_t expert = selected[t * n_expert + slot];
                const uint64_t dbase = (uint64_t)expert * down_expert_bytes +
                                       (uint64_t)r * down_row_bytes;
                acc += cpu_dot_q8_row(model, down_off + dbase, mid_dim,
                                      mid + (uint64_t)t * n_expert * mid_dim,
                                      slot * mid_dim);
            }
            out[t * out_dim + r] = acc;
        }
    }
}

static int cmp_i32(const void *a, const void *b) {
    const int32_t x = *(const int32_t *)a;
    const int32_t y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

/* --- Fase 3 CPU references ------------------------------------------------ */

/* E4M3FN representable values (parity with dsv4_e4m3fn_value_dev). */
static float e4m3fn_value(int i) {
    const int exp = (i >> 3) & 15;
    const int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

static float e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (e4m3fn_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - e4m3fn_value(best));
        const float nd = fabsf(ax - e4m3fn_value(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) {
            best++;
        }
    }
    return sign * e4m3fn_value(best);
}

/* In-place FP8 KV quantize of the nope part (blocks of 64). */
static void cpu_fp8_kv_quantize(float *x, uint32_t n_tok, uint32_t head_dim,
                                uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t row = 0; row < n_tok; row++) {
        float *xr = x + (uint64_t)row * head_dim;
        for (uint32_t off = 0; off < n_nope; off += 64u) {
            float amax = 0.0f;
            for (uint32_t i = 0; i < 64u && off + i < n_nope; i++) {
                amax = fmaxf(amax, fabsf(xr[off + i]));
            }
            const float scale = exp2f(ceilf(log2f(fmaxf(amax, 1.0e-4f) / 448.0f)));
            for (uint32_t i = 0; i < 64u && off + i < n_nope; i++) {
                const float v = xr[off + i];
                const float q = e4m3fn_dequant(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
                xr[off + i] = q;
            }
        }
    }
}

/* Round-to-nearest-even f32 -> f16 (matches HLSL f32tof16 / CUDA __float2half,
 * unlike the truncating f32_to_f16 used to build model weights). */
static uint16_t f32_to_f16_rne(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t man = bits & 0x7fffffu;
    if (exp >= 31) return (uint16_t)(sign | 0x7bffu);
    if (exp <= 0) {
        if (exp < -10) return sign;   /* underflow to zero */
        const uint32_t mant = man | 0x800000u;
        const int shift = 1 - exp;
        const uint32_t m = mant >> (uint32_t)shift;
        const uint32_t rem = (mant >> (uint32_t)shift) & 0x1fffu;
        const uint32_t halfway = 0x1000u;
        uint16_t h = (uint16_t)(sign | (uint16_t)(m >> 13));
        if (rem > halfway || (rem == halfway && (h & 1u))) h++;
        return h;
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    const uint32_t rem = man & 0x1fffu;
    const uint32_t halfway = 0x1000u;
    if (rem > halfway || (rem == halfway && (h & 1u))) h++;
    return h;
}

/* Store raw rows with an f16 round trip. */
static void cpu_store_raw_kv(float *raw, const float *kv, uint32_t raw_cap,
                             uint32_t pos0, uint32_t n_tokens,
                             uint32_t head_dim) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t d = 0; d < head_dim; d++) {
            const uint32_t row = (pos0 + t) % raw_cap;
            raw[(uint64_t)row * head_dim + d] = f16_to_f32(f32_to_f16_rne(kv[(uint64_t)t * head_dim + d]));
        }
    }
}

/* Single-row FP8 KV quantize + f16 store (kv modified in place). */
static void cpu_kv_fp8_store_raw(float *kv, float *raw, uint32_t raw_cap,
                                 uint32_t raw_row, uint32_t head_dim,
                                 uint32_t n_rot) {
    cpu_fp8_kv_quantize(kv, 1, head_dim, n_rot);
    cpu_store_raw_kv(raw, kv, raw_cap, raw_row % raw_cap, 1, head_dim);
}

/* Attention decode over a raw ring-buffer span plus compressed rows.  One
 * (token, head) block; n_tokens is handled as a batch of independent tokens
 * sharing one raw cache (the decode mixed-batch form). */
static void cpu_attn_decode(float *heads, const float *sinks, const float *q,
                            const float *raw_kv, uint32_t n_raw,
                            uint32_t raw_cap, uint32_t raw_start,
                            const float *comp_kv, uint32_t n_comp,
                            const float *comp_mask, uint32_t use_mask,
                            uint32_t n_tokens, uint32_t pos0, uint32_t window,
                            uint32_t ratio, uint32_t n_head,
                            uint32_t head_dim) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const bool single_all = (n_tokens == 1u && ratio == 0u);
        const uint32_t qpos = pos0 + t;
        const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
        uint32_t visible_comp =
            single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
        if (visible_comp > n_comp) visible_comp = n_comp;
        uint32_t raw_count = 0, raw_first_idx = 0;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count = n_raw < 256u ? n_raw : 256u;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
            float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
            float scores[512];
            float local_max = sinks[h];
            for (uint32_t r = 0; r < raw_count; r++) {
                const uint32_t row = (raw_start + raw_first_idx + r) % raw_cap;
                const float *kvrow = raw_kv + (uint64_t)row * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                scores[r] = dot * scale;
                local_max = fmaxf(local_max, scores[r]);
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                const float add = use_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                float s = -INFINITY;
                if (add > -1.0e20f) {
                    const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                    s = dot * scale + add;
                }
                scores[raw_count + c] = s;
                local_max = fmaxf(local_max, s);
            }
            float den = expf(sinks[h] - local_max);
            for (uint32_t i = 0; i < raw_count + visible_comp; i++) {
                scores[i] = expf(scores[i] - local_max);
                den += scores[i];
            }
            for (uint32_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (uint32_t r = 0; r < raw_count; r++) {
                    const uint32_t row = (raw_start + raw_first_idx + r) % raw_cap;
                    acc += raw_kv[(uint64_t)row * head_dim + d] * scores[r];
                }
                for (uint32_t c = 0; c < visible_comp; c++) {
                    acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
                }
                oh[d] = acc / den;
            }
        }
    }
}

/* Indexed mixed decode reference (mirrors attention_indexed_mixed_kernel in
 * ds4_cuda.cu): raw ring span plus the compressed rows selected by the
 * per-token topk, compacted stably to those < visible_comp. */
static void cpu_attn_indexed_decode(float *heads, const float *sinks,
                                    const float *q, const float *raw_kv,
                                    uint32_t n_raw, uint32_t raw_cap,
                                    uint32_t raw_start, const float *comp_kv,
                                    uint32_t n_comp, const uint32_t *topk,
                                    uint32_t top_k, uint32_t n_tokens,
                                    uint32_t pos0, uint32_t window,
                                    uint32_t ratio, uint32_t n_head,
                                    uint32_t head_dim) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t qpos = pos0 + t;
        const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
        uint32_t visible_comp = n_comp;
        if (ratio != 0u) {
            visible_comp = (qpos + 1u) / ratio;
            if (visible_comp > n_comp) visible_comp = n_comp;
        }
        uint32_t raw_count = 0, raw_first_idx = 0;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        uint32_t comp_rows[512];
        uint32_t comp_count = 0;
        for (uint32_t k = 0; k < top_k && comp_count < 512u; k++) {
            const int c = (int)topk[(uint64_t)t * top_k + k];
            if (c >= 0 && (uint32_t)c < visible_comp) {
                comp_rows[comp_count++] = (uint32_t)c;
            }
        }
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
            float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
            float scores[768];
            float local_max = sinks[h];
            for (uint32_t r = 0; r < raw_count; r++) {
                const uint32_t row = (raw_start + raw_first_idx + r) % raw_cap;
                const float *kvrow = raw_kv + (uint64_t)row * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                scores[r] = dot * scale;
                local_max = fmaxf(local_max, scores[r]);
            }
            for (uint32_t c = 0; c < comp_count; c++) {
                const float *kvrow = comp_kv + (uint64_t)comp_rows[c] * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                scores[raw_count + c] = dot * scale;
                local_max = fmaxf(local_max, scores[raw_count + c]);
            }
            float den = expf(sinks[h] - local_max);
            const uint32_t n_score = raw_count + comp_count;
            for (uint32_t i = 0; i < n_score; i++) {
                scores[i] = expf(scores[i] - local_max);
                den += scores[i];
            }
            for (uint32_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (uint32_t r = 0; r < raw_count; r++) {
                    const uint32_t row = (raw_start + raw_first_idx + r) % raw_cap;
                    acc += raw_kv[(uint64_t)row * head_dim + d] * scores[r];
                }
                for (uint32_t c = 0; c < comp_count; c++) {
                    acc += comp_kv[(uint64_t)comp_rows[c] * head_dim + d] *
                           scores[raw_count + c];
                }
                oh[d] = acc / den;
            }
        }
    }
}

/* Causal prefill: raw rows read directly (rows [raw_start, t]), plus optional
 * compressed rows. */
static void cpu_attn_prefill(float *heads, const float *sinks, const float *q,
                             const float *raw_kv, const float *comp_kv,
                             uint32_t n_comp, const float *comp_mask,
                             uint32_t use_mask, uint32_t n_tokens,
                             uint32_t window, uint32_t ratio, uint32_t n_head,
                             uint32_t head_dim) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t raw_start =
            (window != 0u && t + 1u > window) ? t + 1u - window : 0u;
        const uint32_t raw_count = t + 1u - raw_start;
        uint32_t visible_comp = (t + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
            float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
            float scores[512];
            float local_max = sinks[h];
            for (uint32_t r = 0; r < raw_count; r++) {
                const float *kvrow = raw_kv + (uint64_t)(raw_start + r) * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                scores[r] = dot * scale;
                local_max = fmaxf(local_max, scores[r]);
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                const float add = use_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                float s = -INFINITY;
                if (add > -1.0e20f) {
                    const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                    s = dot * scale + add;
                }
                scores[raw_count + c] = s;
                local_max = fmaxf(local_max, s);
            }
            float den = expf(sinks[h] - local_max);
            for (uint32_t i = 0; i < raw_count + visible_comp; i++) {
                scores[i] = expf(scores[i] - local_max);
                den += scores[i];
            }
            for (uint32_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (uint32_t r = 0; r < raw_count; r++) {
                    acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
                }
                for (uint32_t c = 0; c < visible_comp; c++) {
                    acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
                }
                oh[d] = acc / den;
            }
        }
    }
}

/* Low-rank grouped attention output A: low[t][g*rank+r] = sum_d
 * heads[t][g*group_dim+d] * out_a_q8[(g*rank+r)][d]. */
static void cpu_attn_output_low(float *low, const uint8_t *out_a,
                                uint32_t n_groups_total, uint32_t group0,
                                uint32_t group_cnt, uint64_t group_dim,
                                uint64_t rank, const float *heads,
                                uint32_t n_rows) {
    const uint32_t blocks_a = (uint32_t)((group_dim + 31u) / 32u);
    for (uint32_t t = 0; t < n_rows; t++) {
        for (uint32_t gl = 0; gl < group_cnt; gl++) {
            const uint32_t g = group0 + gl;
            for (uint32_t r = 0; r < rank; r++) {
                float acc = 0.0f;
                for (uint32_t b = 0; b < blocks_a; b++) {
                    const uint32_t i0 = b * 32u;
                    const uint32_t bn = (uint32_t)(group_dim - i0 < 32u ? group_dim - i0 : 32u);
                    const uint32_t wrow = g * (uint32_t)rank + r;
                    float amax = 0.0f;
                    for (uint32_t i = 0; i < bn; i++) {
                        amax = fmaxf(amax, fabsf(heads[(uint64_t)t * n_groups_total * group_dim +
                                                       (uint64_t)g * group_dim + i0 + i]));
                    }
                    const float d = amax / 127.0f;
                    const float id = d != 0.0f ? 1.0f / d : 0.0f;
                    int dot = 0;
                    for (uint32_t i = 0; i < bn; i++) {
                        int q = (int)nearbyintf(heads[(uint64_t)t * n_groups_total * group_dim +
                                                      (uint64_t)g * group_dim + i0 + i] * id);
                        if (q > 127) q = 127;
                        if (q < -128) q = -128;
                        const uint8_t *blk = out_a + (uint64_t)(wrow * blocks_a + b) * 34u;
                        dot += (int)(int8_t)blk[2 + i] * q;
                    }
                    const uint8_t *blk = out_a + (uint64_t)(wrow * blocks_a + b) * 34u;
                    uint16_t sw;
                    memcpy(&sw, blk, 2);
                    acc += f16_to_f32(sw) * d * (float)dot;
                }
                low[(uint64_t)t * (group_cnt * rank) + (uint64_t)gl * rank + r] = acc;
            }
        }
    }
}

/* Low-rank grouped attention output A in Q4_K: low[t][g*rank+r] = sum_d
 * heads[t][g*group_dim+d] * out_a_q4k[(g*rank+r)][d].  Activation is raw f32
 * (no inline quantization). */
static void cpu_attn_output_low_q4k(float *low, const uint8_t *out_a,
                                    uint32_t n_groups_total, uint32_t group0,
                                    uint32_t group_cnt, uint64_t group_dim,
                                    uint64_t rank, const float *heads,
                                    uint32_t n_rows) {
    const uint32_t blocks_a = (uint32_t)(group_dim / 256u);
    for (uint32_t t = 0; t < n_rows; t++) {
        for (uint32_t gl = 0; gl < group_cnt; gl++) {
            const uint32_t g = group0 + gl;
            for (uint32_t r = 0; r < rank; r++) {
                const uint32_t wrow = gl * (uint32_t)rank + r;
                float acc = 0.0f;
                for (uint32_t b = 0; b < blocks_a; b++) {
                    const uint8_t *blk = out_a + (uint64_t)(wrow * blocks_a + b) * 144u;
                    uint16_t dh, dmh;
                    memcpy(&dh, blk, 2);
                    memcpy(&dmh, blk + 2, 2);
                    const float d = f16_to_f32(dh);
                    const float dmin = f16_to_f32(dmh);
                    for (uint32_t j = 0; j < 8u; j++) {
                        uint32_t sc, m;
                        if (j < 4u) {
                            sc = blk[4u + j] & 63u;
                            m = blk[4u + j + 4u] & 63u;
                        } else {
                            const uint32_t a = blk[4u + j + 4u];
                            const uint32_t bb = blk[4u + j - 4u];
                            const uint32_t c = blk[4u + j];
                            sc = (a & 0xfu) | ((bb >> 6u) << 4u);
                            m = (a >> 4u) | ((c >> 6u) << 4u);
                        }
                        const float scale = d * (float)sc;
                        const float minv = dmin * (float)m;
                        const uint32_t byte_off = (j >> 1u) * 32u;
                        const uint32_t shift = (j & 1u) * 4u;
                        for (uint32_t l = 0; l < 32u; l++) {
                            const uint32_t qb = blk[16u + byte_off + l];
                            const uint32_t q = (qb >> shift) & 0xfu;
                            acc += (scale * (float)q - minv) *
                                   heads[(uint64_t)t * n_groups_total * group_dim +
                                         (uint64_t)g * group_dim + b * 256u +
                                         j * 32u + l];
                        }
                    }
                }
                low[(uint64_t)t * (group_cnt * rank) + (uint64_t)gl * rank + r] = acc;
            }
        }
    }
}

/* --- model buffer layout (must stay 4-aligned) ---------------------------- */

#define OFF_Q8 0u                    /* matmul Q8_0 weights */
#define OFF_F16 8704u                /* matmul F16 weights */
#define OFF_F32_NORM 25088u          /* RMS norm F32 weights (128) */
#define OFF_F32_MM 25600u            /* matmul F32 weights */
#define OFF_EMB 33792u               /* token embedding Q8_0 */
#define OFF_SINKS 51200u             /* attention sinks F32 (64) */
#define OFF_OUT_A 51456u             /* attention output A Q8_0 (4x8x64) */
#define OFF_OUT_B 53632u             /* attention output B Q8_0 (16x32) */
#define OFF_MOE_GATE 54176u          /* routed MoE gate experts Q8_0 (8x32x64) */
#define OFF_MOE_UP 71584u            /* routed MoE up experts Q8_0 (8x32x64) */
#define OFF_MOE_DOWN 88992u          /* routed MoE down experts Q8_0 (8x32x32) */
#define OFF_ROUTER_BIAS 97696u       /* router bias F32 (256) */
#define OFF_ROUTER_HASH 98720u       /* router hash int32 (16 rows x 6) */
#define OFF_HC_SCALE 99104u          /* HC sinkhorn scale F32 (3) */
#define OFF_HC_BASE 99116u           /* HC sinkhorn base F32 (24) */
#define OFF_HC_NORM 99212u           /* HC norm weight F32 (32) */
#define OFF_OUT_HC_SCALE 99340u      /* output HC scale F32 (1) */
#define OFF_OUT_HC_BASE 99344u       /* output HC base F32 (4) */
#define OFF_APE 99360u               /* compressor ape F32 (64) */
#define OFF_DS1 99616u               /* dspark markov w1 row Q8_0 (2 blocks) */
#define OFF_DS2 99684u               /* dspark markov w2 Q8_0 (64 x 2 blocks) */
#define OFF_EMB_F16 104036u          /* token embedding F16 (16 x 64) */
#define OFF_COMP_KV 106084u          /* compressor kv F16 weights (256 x 4096) */
#define OFF_COMP_SC 2203236u         /* compressor gate F16 weights (256 x 4096) */
#define OFF_COMP_APE 4300388u        /* compressor ape F32 (ratio4 x 256) */
#define MODEL_BYTES 4304484u
#define MODEL_PADDED 4304896u        /* next multiple of 4096 */

static uint8_t *g_model = NULL;

static void build_model(void) {
    /* Matmul Q8_0 weights: out_dim=32, in_dim=256. */
    const uint32_t mm_out = 32u, mm_in = 256u;
    const uint32_t mm_blocks = mm_in / 32u;
    float row[32];
    for (uint32_t r = 0; r < mm_out; r++) {
        for (uint32_t b = 0; b < mm_blocks; b++) {
            for (uint32_t i = 0; i < 32u; i++) {
                row[i] = 0.05f * (float)(int32_t)((r * 7u + b * 3u + i) % 63) - 1.5f;
            }
            quant_q8_block(g_model + OFF_Q8 + (uint64_t)(r * mm_blocks + b) * 34u,
                           row, 32);
        }
    }
    /* Matmul F16 weights: out_dim=32, in_dim=256. */
    for (uint32_t r = 0; r < mm_out; r++) {
        for (uint32_t k = 0; k < mm_in; k++) {
            const float v = 0.02f * (float)(r + 1) *
                            (float)(int32_t)((k * 13u) % 17) - 1.2f;
            uint16_t h = f32_to_f16(v);
            memcpy(g_model + OFF_F16 + (uint64_t)(r * mm_in + k) * 2u, &h, 2);
        }
    }
    /* RMS norm F32 weights (n=128). */
    for (uint32_t i = 0; i < 128u; i++) {
        float v = 0.5f + 0.01f * (float)(i % 11);
        memcpy(g_model + OFF_F32_NORM + (uint64_t)i * 4u, &v, 4);
    }
    /* Matmul F32 weights: out_dim=16, in_dim=128. */
    const uint32_t f32_out = 16u, f32_in = 128u;
    for (uint32_t r = 0; r < f32_out; r++) {
        for (uint32_t k = 0; k < f32_in; k++) {
            float v = 0.005f * (float)(r * f32_in + k) - 2.0f;
            memcpy(g_model + OFF_F32_MM + (uint64_t)(r * f32_in + k) * 4u, &v, 4);
        }
    }
    /* Token embedding Q8_0: n_vocab=64, n_embd=256. */
    const uint32_t n_vocab = 64u, n_embd = 256u;
    for (uint32_t tok = 0; tok < n_vocab; tok++) {
        for (uint32_t b = 0; b < n_embd / 32u; b++) {
            for (uint32_t i = 0; i < 32u; i++) {
                row[i] = 0.02f * (float)(int32_t)((tok * 5u + b * 11u + i) % 41) - 0.4f;
            }
            quant_q8_block(g_model + OFF_EMB +
                               (uint64_t)(tok * (n_embd / 32u) + b) * 34u,
                           row, 32);
        }
    }
    /* Attention sinks (n=64). */
    for (uint32_t h = 0; h < 64u; h++) {
        float v = 0.1f + 0.01f * (float)(h % 7);
        memcpy(g_model + OFF_SINKS + (uint64_t)h * 4u, &v, 4);
    }
    /* Attention output A: n_groups=4, rank=8, group_dim=64 (Q8_0). */
    {
        const uint32_t ng = 4u, rank = 8u, gdim = 64u;
        const uint32_t gb = gdim / 32u;
        for (uint32_t g = 0; g < ng; g++) {
            for (uint32_t r = 0; r < rank; r++) {
                for (uint32_t b = 0; b < gb; b++) {
                    for (uint32_t i = 0; i < 32u; i++) {
                        row[i] = 0.04f * (float)(int32_t)((g * 17u + r * 5u + b * 3u + i) % 61) - 1.2f;
                    }
                    quant_q8_block(g_model + OFF_OUT_A +
                                       (uint64_t)((g * rank + r) * gb + b) * 34u,
                                   row, 32);
                }
            }
        }
    }
    /* Attention output B: out_dim=16, low_dim=32 (Q8_0). */
    {
        const uint32_t od = 16u, ld = 32u;
        const uint32_t lb = ld / 32u;
        for (uint32_t r = 0; r < od; r++) {
            for (uint32_t b = 0; b < lb; b++) {
                for (uint32_t i = 0; i < 32u; i++) {
                    row[i] = 0.03f * (float)(int32_t)((r * 9u + b * 5u + i) % 53) - 0.8f;
                }
                quant_q8_block(g_model + OFF_OUT_B +
                                   (uint64_t)(r * lb + b) * 34u,
                               row, 32);
            }
        }
    }
    /* Routed MoE experts: n_total_expert=8, expert_in_dim=64 (2 blocks),
     * expert_mid_dim=32 (1 block), out_dim=32.  gate/up are [8][32][64],
     * down is [8][32][32], all Q8_0. */
    {
        const uint32_t n_exp = 8u, mid_dim = 32u, in_dim = 64u;
        const uint32_t in_b = in_dim / 32u;
        for (uint32_t e = 0; e < n_exp; e++) {
            for (uint32_t r = 0; r < mid_dim; r++) {
                for (uint32_t b = 0; b < in_b; b++) {
                    for (uint32_t i = 0; i < 32u; i++) {
                        row[i] = 0.05f * (float)(int32_t)((e * 13u + r * 7u + b * 3u + i) % 59) - 1.4f;
                    }
                    quant_q8_block(g_model + OFF_MOE_GATE +
                                       (uint64_t)((e * mid_dim + r) * in_b + b) * 34u,
                                   row, 32);
                    for (uint32_t i = 0; i < 32u; i++) {
                        row[i] = 0.04f * (float)(int32_t)((e * 17u + r * 5u + b * 11u + i) % 47) - 1.1f;
                    }
                    quant_q8_block(g_model + OFF_MOE_UP +
                                       (uint64_t)((e * mid_dim + r) * in_b + b) * 34u,
                                   row, 32);
                }
            }
        }
        for (uint32_t e = 0; e < n_exp; e++) {
            for (uint32_t r = 0; r < 32u; r++) {
                for (uint32_t i = 0; i < 32u; i++) {
                    row[i] = 0.06f * (float)(int32_t)((e * 7u + r * 11u + i) % 43) - 1.3f;
                }
                quant_q8_block(g_model + OFF_MOE_DOWN +
                                   (uint64_t)(e * 32u + r) * 34u,
                               row, 32);
            }
        }
    }
    /* Router bias (256 f32) and hash table (16 rows x 6 int32). */
    for (uint32_t i = 0; i < 256u; i++) {
        float v = 0.02f * (float)(int32_t)(i % 17) - 0.16f;
        memcpy(g_model + OFF_ROUTER_BIAS + (uint64_t)i * 4u, &v, 4);
    }
    for (uint32_t r = 0; r < 16u; r++) {
        for (uint32_t j = 0; j < 6u; j++) {
            const int32_t v = (int32_t)((r * 11u + j * 5u) % 8);
            memcpy(g_model + OFF_ROUTER_HASH + (uint64_t)(r * 6u + j) * 4u, &v, 4);
        }
    }
    /* Token embedding F16 (n_vocab=16, n_embd=64) for the HC embedding. */
    {
        const uint32_t n_vocab = 16u, n_embd = 64u;
        for (uint32_t tok = 0; tok < n_vocab; tok++) {
            for (uint32_t d = 0; d < n_embd; d++) {
                const float v = 0.03f * (float)(int32_t)((tok * 13u + d * 7u) % 29) - 0.5f;
                uint16_t h = f32_to_f16(v);
                memcpy(g_model + OFF_EMB_F16 + (uint64_t)(tok * n_embd + d) * 2u,
                       &h, 2);
            }
        }
    }
    /* HC sinkhorn scale (pre/post/comb) + base (24). */
    const float hc_scale[3] = { 0.7f, 0.9f, 0.5f };
    memcpy(g_model + OFF_HC_SCALE, hc_scale, sizeof(hc_scale));
    for (uint32_t i = 0; i < 24u; i++) {
        float v = 0.05f * (float)(int32_t)((i * 5u) % 21) - 0.5f;
        memcpy(g_model + OFF_HC_BASE + (uint64_t)i * 4u, &v, 4);
    }
    /* HC norm weight (n_embd=32) and output HC scale/base. */
    for (uint32_t i = 0; i < 32u; i++) {
        float v = 0.5f + 0.02f * (float)(i % 7);
        memcpy(g_model + OFF_HC_NORM + (uint64_t)i * 4u, &v, 4);
    }
    const float ohc_scale = 0.8f;
    memcpy(g_model + OFF_OUT_HC_SCALE, &ohc_scale, sizeof(ohc_scale));
    for (uint32_t i = 0; i < 4u; i++) {
        float v = 0.1f * (float)(i + 1) - 0.2f;
        memcpy(g_model + OFF_OUT_HC_BASE + (uint64_t)i * 4u, &v, 4);
    }
    /* Compressor ape (width=16, ratio=4 -> 64 f32). */
    for (uint32_t i = 0; i < 64u; i++) {
        float v = 0.05f * (float)(int32_t)((i * 7u) % 23) - 0.4f;
        memcpy(g_model + OFF_APE + (uint64_t)i * 4u, &v, 4);
    }
    /* DSpark markov w1 row (rank=64 -> 2 q8 blocks) and w2 (64 x 2 blocks). */
    for (uint32_t b = 0; b < 2u; b++) {
        float row[32];
        for (uint32_t i = 0; i < 32u; i++) {
            row[i] = 0.05f * (float)(int32_t)((b * 13u + i * 3u) % 61) - 1.4f;
        }
        quant_q8_block(g_model + OFF_DS1 + (uint64_t)b * 34u, row, 32);
    }
    for (uint32_t r = 0; r < 64u; r++) {
        for (uint32_t b = 0; b < 2u; b++) {
            float row[32];
            for (uint32_t i = 0; i < 32u; i++) {
                row[i] = 0.03f * (float)(int32_t)((r * 17u + b * 11u + i * 5u) % 71) -
                         1.0f;
            }
            quant_q8_block(g_model + OFF_DS2 + (uint64_t)(r * 2u + b) * 34u, row, 32);
        }
    }
    /* F16 compressor pair weights: width=256, in_dim=4096 (the production
     * shape for ds4_gpu_matmul_f16_pair_compressor_store_tensor). */
    {
        const uint32_t comp_in = 4096u, comp_width = 256u;
        for (uint32_t r = 0; r < comp_width; r++) {
            for (uint32_t k = 0; k < comp_in; k++) {
                float v = 0.001f * (float)(int32_t)((r * 3u + k * 7u) % 37) - 0.02f;
                uint16_t h = f32_to_f16(v);
                memcpy(g_model + OFF_COMP_KV + (uint64_t)(r * comp_in + k) * 2u,
                       &h, 2);
                v = 0.001f * (float)(int32_t)((r * 5u + k * 3u) % 41) - 0.02f;
                h = f32_to_f16(v);
                memcpy(g_model + OFF_COMP_SC + (uint64_t)(r * comp_in + k) * 2u,
                       &h, 2);
            }
        }
        for (uint32_t i = 0; i < 4u * comp_width; i++) {
            float v = 0.005f * (float)(int32_t)((i * 11u) % 29) - 0.06f;
            memcpy(g_model + OFF_COMP_APE + (uint64_t)i * 4u, &v, 4);
        }
    }
}

static int check_close(const float *got, const float *ref, uint32_t n,
                       float tol, const char *msg) {
    int ok = 1;
    for (uint32_t i = 0; i < n; i++) {
        const float err = fabsf(got[i] - ref[i]);
        const float rel = fabsf(ref[i]) > 1e-3f ? err / fabsf(ref[i]) : err;
        if (rel > tol) {
            fprintf(stderr,
                    "FAIL: %s element %u got %.6f ref %.6f (rel_err %.3g tol %.3g)\n",
                    msg, i, got[i], ref[i], rel, tol);
            ok = 0;
            break;
        }
    }
    if (ok) printf("ok: %s\n", msg);
    else failures++;
    return ok;
}

/* --- Phase 5 CPU references (match the shader math) ---------------------- */

static void cpu_hc4_split_one(float *out, const float *mix,
                              const float *scale, const float *base,
                              uint32_t sinkhorn_iters, float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) {
            c[r * 4 + col] = c[r * 4 + col] / s + epsv;
        }
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}

static void cpu_hc_weighted_sum(float *out, const float *residual,
                                const float *w, uint32_t n_embd,
                                uint32_t n_hc, uint32_t n_tokens,
                                uint32_t stride) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = 0.0f;
            for (uint32_t h = 0; h < n_hc; h++) {
                acc += residual[(uint64_t)t * n_hc * n_embd + h * n_embd + d] *
                       w[(uint64_t)t * stride + h];
            }
            out[(uint64_t)t * n_embd + d] = acc;
        }
    }
}

/* Generic HC expand; post/comb are read with strides (tensor weights use
 * stride n_hc / n_hc*n_hc, split weights use the 24-float mix stride). */
static void cpu_hc_expand(float *out_hc, const float *block_out,
                          const float *block_add, const float *residual_hc,
                          const float *post, const float *comb,
                          uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens,
                          uint32_t post_stride, uint32_t comb_stride,
                          uint32_t has_add, uint32_t has_add2,
                          const float *block_add2) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t dst = 0; dst < n_hc; dst++) {
            for (uint32_t d = 0; d < n_embd; d++) {
                float bv = block_out[(uint64_t)t * n_embd + d];
                if (has_add) bv += block_add[(uint64_t)t * n_embd + d];
                if (has_add2) bv += block_add2[(uint64_t)t * n_embd + d];
                float acc = bv * post[(uint64_t)t * post_stride + dst];
                for (uint32_t src = 0; src < n_hc; src++) {
                    float comb_v = comb[(uint64_t)t * comb_stride + dst +
                                        (uint64_t)src * n_hc];
                    float res_v =
                        residual_hc[(uint64_t)t * n_hc * n_embd +
                                    (uint64_t)src * n_embd + d];
                    acc += comb_v * res_v;
                }
                out_hc[(uint64_t)t * n_hc * n_embd +
                       (uint64_t)dst * n_embd + d] = acc;
            }
        }
    }
}

static void cpu_output_hc_weights(float *out, const float *pre,
                                  const float *scale, const float *base,
                                  uint32_t n_hc, uint32_t n_tokens,
                                  float epsv) {
    for (uint32_t gid = 0; gid < n_tokens * n_hc; gid++) {
        uint32_t h = gid % n_hc;
        float z = pre[gid] * scale[0] + base[h];
        out[gid] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
}

static void cpu_directional_steering(float *x, const float *dir,
                                     uint32_t layer, uint32_t width,
                                     uint32_t rows, float scale) {
    const float *dl = dir + (uint64_t)layer * width;
    for (uint32_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        float *xr = x + (uint64_t)r * width;
        for (uint32_t i = 0; i < width; i++) sum += xr[i] * dl[i];
        const float coeff = scale * sum;
        for (uint32_t i = 0; i < width; i++) xr[i] -= coeff * dl[i];
    }
}

static void cpu_indexer_scores(float *scores, const float *q,
                               const float *weights, const float *comp,
                               uint32_t n_comp, uint32_t n_tokens,
                               uint32_t pos0, uint32_t n_head,
                               uint32_t head_dim, uint32_t ratio,
                               float scale, uint32_t causal) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t c = 0; c < n_comp; c++) {
            if (causal) {
                uint32_t n_visible = (pos0 + t + 1u) / ratio;
                if (c >= n_visible) {
                    scores[(uint64_t)t * n_comp + c] = -INFINITY;
                    continue;
                }
            }
            float total = 0.0f;
            for (uint32_t h = 0; h < n_head; h++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += q[((uint64_t)t * n_head + h) * head_dim + d] *
                           comp[(uint64_t)c * head_dim + d];
                }
                total += fmaxf(dot, 0.0f) * weights[(uint64_t)t * n_head + h];
            }
            scores[(uint64_t)t * n_comp + c] = total * scale;
        }
    }
}

static void cpu_indexer_topk(uint32_t *sel, const float *scores,
                             uint32_t n_comp, uint32_t n_tokens,
                             uint32_t top_k) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *row = scores + (uint64_t)t * n_comp;
        uint32_t *dst = sel + (uint64_t)t * top_k;
        for (uint32_t k = 0; k < top_k; k++) dst[k] = 0;
        for (uint32_t c = 0; c < n_comp; c++) {
            float v = row[c];
            for (uint32_t k = 0; k < top_k; k++) {
                if (k >= c || v > row[dst[k]]) {
                    for (uint32_t j = top_k - 1; j > k; j--) dst[j] = dst[j - 1];
                    dst[k] = c;
                    break;
                }
            }
        }
    }
}

static void cpu_topk_mask(float *mask, const uint32_t *topk,
                          uint32_t n_comp, uint32_t n_tokens,
                          uint32_t top_k) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t c = 0; c < n_comp; c++) {
            float v = -INFINITY;
            for (uint32_t k = 0; k < top_k; k++) {
                if (topk[(uint64_t)t * top_k + k] == c) { v = 0.0f; break; }
            }
            mask[(uint64_t)t * n_comp + c] = v;
        }
    }
}

static void cpu_indexer_top1_value(uint32_t *sel, float *values,
                                   const float *scores, uint32_t n_comp,
                                   uint32_t n_tokens, uint32_t index_offset) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        float best_v = -INFINITY;
        uint32_t best_i = 0;
        for (uint32_t i = 0; i < n_comp; i++) {
            const float v = scores[(uint64_t)t * n_comp + i];
            const uint32_t gi = index_offset + i;
            const uint32_t bgi = index_offset + best_i;
            if (v > best_v || (v == best_v && gi < bgi)) {
                best_v = v;
                best_i = i;
            }
        }
        sel[t] = index_offset + best_i;
        values[t] = best_v;
    }
}

/* E2M1FN dequantize (test-local; parity with the shader helper). */
static float dsv4_e2m1fn_dequant_cpu(float x) {
    const float vals[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - vals[0]);
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - vals[i]);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 &&
                                 (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * vals[best];
}

static void cpu_hadamard_fp4(float *x, uint32_t n_rows) {
    for (uint32_t row = 0; row < n_rows; row++) {
        float vals[128];
        float *xr = x + (uint64_t)row * 128;
        for (uint32_t i = 0; i < 128; i++) vals[i] = xr[i];
        for (uint32_t stride = 1; stride < 128; stride <<= 1) {
            for (uint32_t i = 0; i < 128; i += 2 * stride) {
                for (uint32_t k = 0; k < stride; k++) {
                    float a = vals[i + k];
                    float b = vals[i + stride + k];
                    vals[i + k] = a + b;
                    vals[i + stride + k] = a - b;
                }
            }
        }
        for (uint32_t i = 0; i < 128; i++) {
            float v = vals[i] * 0.08838834764831845f;
            float amax = 0.0f;
            uint32_t b0 = (i / 32u) * 32u;
            for (uint32_t k = b0; k < b0 + 32u; k++) {
                amax = fmaxf(amax, fabsf(vals[k] * 0.08838834764831845f));
            }
            amax = fmaxf(amax, 7.052966104933725e-38f);
            float scale = exp2f(ceilf(log2f(amax / 6.0f)));
            float cl = fminf(6.0f, fmaxf(-6.0f, v / scale));
            xr[i] = dsv4_e2m1fn_dequant_cpu(cl) * scale;
        }
    }
}

int main(void) {
    printf("=== Vulkan backend smoke test ===\n");

    CHECK(ds4_gpu_init() != 0, "ds4_gpu_init");
    CHECK(g_n_gpus == 1, "g_n_gpus == 1");

    /* --- basic tensor lifecycle (Fase 0) --- */
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(16u * 1024u);
    CHECK(t != NULL, "tensor_alloc(16 KiB)");
    CHECK(ds4_gpu_tensor_bytes(t) == 16u * 1024u, "tensor_bytes");
    if (t) {
        float src[16];
        for (int i = 0; i < 16; i++) src[i] = (float)i * 0.5f;
        CHECK(ds4_gpu_tensor_write(t, 0, src, sizeof(src)) != 0,
              "tensor_write(16 floats)");
        float dst[16];
        CHECK(ds4_gpu_tensor_read(t, 0, dst, sizeof(dst)) != 0,
              "tensor_read(16 floats)");
        int same = 1;
        for (int i = 0; i < 16; i++) {
            if (dst[i] != src[i]) { same = 0; break; }
        }
        CHECK(same, "readback matches written data");
        CHECK(ds4_gpu_tensor_fill_f32(t, 3.5f, 16) != 0, "fill_f32(16)");
        float one;
        CHECK(ds4_gpu_tensor_read(t, 0, &one, sizeof(one)) != 0,
              "tensor_read(1 float)");
        CHECK(one == 3.5f, "fill_f32 value correct");
    }

    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(64u);
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(64u);
    if (a && b) {
        uint8_t data[64];
        for (int i = 0; i < 64; i++) data[i] = (uint8_t)(i + 1);
        CHECK(ds4_gpu_tensor_write(a, 0, data, 64) != 0, "write a");
        CHECK(ds4_gpu_tensor_copy(b, 0, a, 0, 64) != 0, "tensor_copy");
        uint8_t out[64];
        CHECK(ds4_gpu_tensor_read(b, 0, out, 64) != 0, "read b");
        CHECK(memcmp(data, out, 64) == 0, "copy round-trip matches");
    }
    if (a) ds4_gpu_tensor_free(a);
    if (b) ds4_gpu_tensor_free(b);

    CHECK(ds4_gpu_begin_commands() != 0, "begin_commands");
    CHECK(ds4_gpu_commands_active() != 0, "commands_active");
    CHECK(ds4_gpu_flush_encoder() != 0, "flush_encoder");
    CHECK(ds4_gpu_flush_commands() != 0, "flush_commands");
    CHECK(ds4_gpu_end_commands() != 0, "end_commands");
    CHECK(ds4_gpu_synchronize() != 0, "synchronize");

    /* --- add (Fase 1) --- */
    {
        const uint32_t n = 1024u;
        ds4_gpu_tensor *a = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        ds4_gpu_tensor *b = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        CHECK(a && b && out, "add: alloc a/b/out");
        if (a && b && out) {
            float *av = (float *)malloc(n * sizeof(float));
            float *bv = (float *)malloc(n * sizeof(float));
            float *ov = (float *)malloc(n * sizeof(float));
            for (uint32_t i = 0; i < n; i++) {
                av[i] = (float)i;
                bv[i] = (float)(n - i);
            }
            CHECK(ds4_gpu_tensor_write(a, 0, av, (uint64_t)n * sizeof(float)) != 0,
                  "add: write a (H2D)");
            CHECK(ds4_gpu_tensor_write(b, 0, bv, (uint64_t)n * sizeof(float)) != 0,
                  "add: write b (H2D)");
            CHECK(ds4_gpu_begin_commands() != 0, "add: begin_commands");
            CHECK(ds4_gpu_add_tensor(out, a, b, n) != 0, "add: add_tensor");
            CHECK(ds4_gpu_end_commands() != 0, "add: end_commands");
            CHECK(ds4_gpu_synchronize() != 0, "add: synchronize");
            CHECK(ds4_gpu_tensor_read(out, 0, ov, (uint64_t)n * sizeof(float)) != 0,
                  "add: read out (D2H)");
            int ok = 1;
            for (uint32_t i = 0; i < n; i++) {
                if (ov[i] != (float)n) { ok = 0; break; }
            }
            CHECK(ok, "add: out == a + b");
            free(av); free(bv); free(ov);
        }
        if (a) ds4_gpu_tensor_free(a);
        if (b) ds4_gpu_tensor_free(b);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* --- add3 (Fase 1) --- */
    {
        const uint32_t n = 4096u;
        ds4_gpu_tensor *a = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        ds4_gpu_tensor *b = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        ds4_gpu_tensor *c = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        CHECK(a && b && c && out, "add3: alloc a/b/c/out");
        if (a && b && c && out) {
            float *av = (float *)malloc(n * sizeof(float));
            float *bv = (float *)malloc(n * sizeof(float));
            float *cv = (float *)malloc(n * sizeof(float));
            float *ov = (float *)malloc(n * sizeof(float));
            for (uint32_t i = 0; i < n; i++) { av[i] = 1.0f; bv[i] = 2.0f; cv[i] = 3.0f; }
            CHECK(ds4_gpu_tensor_write(a, 0, av, (uint64_t)n * sizeof(float)) != 0,
                  "add3: write a");
            CHECK(ds4_gpu_tensor_write(b, 0, bv, (uint64_t)n * sizeof(float)) != 0,
                  "add3: write b");
            CHECK(ds4_gpu_tensor_write(c, 0, cv, (uint64_t)n * sizeof(float)) != 0,
                  "add3: write c");
            CHECK(ds4_gpu_add3_tensor(out, a, b, c, n) != 0,
                  "add3: add3_tensor (one-shot)");
            CHECK(ds4_gpu_synchronize() != 0, "add3: synchronize");
            CHECK(ds4_gpu_tensor_read(out, 0, ov, (uint64_t)n * sizeof(float)) != 0,
                  "add3: read out");
            int ok = 1;
            for (uint32_t i = 0; i < n; i++) if (ov[i] != 6.0f) { ok = 0; break; }
            CHECK(ok, "add3: out == a + b + c");
            free(av); free(bv); free(cv); free(ov);
        }
        if (a) ds4_gpu_tensor_free(a);
        if (b) ds4_gpu_tensor_free(b);
        if (c) ds4_gpu_tensor_free(c);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* --- view + tier shims (Fase 0) --- */
    if (t) {
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(t, 8u, 8u);
        CHECK(view != NULL && ds4_gpu_tensor_bytes(view) == 8u,
              "tensor_view(offset 8, 8 bytes)");
        if (view) ds4_gpu_tensor_free(view);
    }
    ds4_gpu_tensor tier = {0};
    CHECK(ds4_gpu_tensor_alloc_on(&tier, 0, 128u) == 0, "tensor_alloc_on(tier 0)");
    ds4_gpu_tensor_free_in_place(&tier);

    /* --- model wrapper + Fase 2 kernel roundtrips --- */
    /* VK_EXT_external_memory_host requires the host pointer to be aligned to
     * minImportedHostPointerAlignment (4096) and the size to be a multiple
     * of it. */
    {
        void *mp = NULL;
        if (posix_memalign(&mp, 4096, MODEL_PADDED) == 0) {
            g_model = (uint8_t *)mp;
        }
    }
    CHECK(g_model != NULL, "model: alloc (posix_memalign 4096)");
    memset(g_model, 0, MODEL_PADDED);
    build_model();
    CHECK(ds4_gpu_set_model_map(g_model, MODEL_PADDED) != 0,
          "model: set_model_map (host import)");

    /* swiglu */
    {
        const uint32_t n = 2048u;
        const float clamp = 12.0f, weight = 1.5f;
        ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        CHECK(gate && up && out, "swiglu: alloc");
        if (gate && up && out) {
            float *g = (float *)malloc(n * sizeof(float));
            float *u = (float *)malloc(n * sizeof(float));
            float *o = (float *)malloc(n * sizeof(float));
            float *ref = (float *)malloc(n * sizeof(float));
            for (uint32_t i = 0; i < n; i++) {
                g[i] = 0.1f * (float)(int32_t)(i % 100) - 3.0f;
                u[i] = 0.05f * (float)(int32_t)((i * 7u) % 100) - 1.0f;
            }
            cpu_swiglu(ref, g, u, n, clamp, weight);
            ds4_gpu_tensor_write(gate, 0, g, (uint64_t)n * sizeof(float));
            ds4_gpu_tensor_write(up, 0, u, (uint64_t)n * sizeof(float));
            CHECK(ds4_gpu_swiglu_tensor(out, gate, up, n, clamp, weight) != 0,
                  "swiglu: swiglu_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "swiglu: synchronize");
            ds4_gpu_tensor_read(out, 0, o, (uint64_t)n * sizeof(float));
            check_close(o, ref, n, 1e-4f, "swiglu: out matches CPU");
            free(g); free(u); free(o); free(ref);
        }
        if (gate) ds4_gpu_tensor_free(gate);
        if (up) ds4_gpu_tensor_free(up);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* argmax */
    {
        const uint32_t n = 4096u;
        ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(int32_t));
        CHECK(logits && out, "argmax: alloc");
        if (logits && out) {
            float *l = (float *)malloc(n * sizeof(float));
            for (uint32_t i = 0; i < n; i++) l[i] = sinf((float)i) * 5.0f;
            l[42] = 100.0f;          /* unique max */
            l[777] = 100.0f;         /* tie with 42 -> 42 wins */
            uint32_t ref = 42u;
            ds4_gpu_tensor_write(logits, 0, l, (uint64_t)n * sizeof(float));
            CHECK(ds4_gpu_argmax_tensor(out, logits, n) != 0,
                  "argmax: argmax_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "argmax: synchronize");
            int32_t got = -1;
            ds4_gpu_tensor_read(out, 0, &got, sizeof(got));
            CHECK((uint32_t)got == ref, "argmax: index == 42 (tie low)");
            free(l);
        }
        if (logits) ds4_gpu_tensor_free(logits);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* sort_i32_rows_asc */
    {
        const uint32_t w = 128u, rows = 3u;
        const uint64_t bytes = (uint64_t)w * rows * sizeof(int32_t);
        ds4_gpu_tensor *src = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *dst = ds4_gpu_tensor_alloc(bytes);
        CHECK(src && dst, "sort: alloc");
        if (src && dst) {
            int32_t *s = (int32_t *)malloc(bytes);
            int32_t *d = (int32_t *)malloc(bytes);
            int32_t *ref = (int32_t *)malloc(bytes);
            for (uint32_t i = 0; i < w * rows; i++) {
                s[i] = (int32_t)((i * 2654435761u) >> 16) % 1000 - 500;
            }
            for (uint32_t r = 0; r < rows; r++) {
                memcpy(ref + r * w, s + r * w, w * sizeof(int32_t));
                qsort(ref + r * w, w, sizeof(int32_t), cmp_i32);
            }
            ds4_gpu_tensor_write(src, 0, s, bytes);
            CHECK(ds4_gpu_sort_i32_rows_asc_tensor(dst, src, w, rows) != 0,
                  "sort: sort_i32_rows_asc");
            CHECK(ds4_gpu_synchronize() != 0, "sort: synchronize");
            ds4_gpu_tensor_read(dst, 0, d, bytes);
            int ok = memcmp(d, ref, bytes) == 0;
            if (!ok) {
                fprintf(stderr, "FAIL: sort: row 0 got %d expected %d\n",
                        d[0], ref[0]);
                failures++;
            } else {
                printf("ok: sort: rows sorted ascending\n");
            }
            free(s); free(d); free(ref);
        }
        if (src) ds4_gpu_tensor_free(src);
        if (dst) ds4_gpu_tensor_free(dst);
    }

    /* rms_norm_plain / rows */
    {
        const uint32_t n = 512u, rows = 3u;
        const float eps = 1e-5f;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n * rows * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n * rows * sizeof(float));
        CHECK(x && out, "rms_norm: alloc");
        if (x && out) {
            float *xv = (float *)malloc(n * rows * sizeof(float));
            float *ov = (float *)malloc(n * rows * sizeof(float));
            float *ref = (float *)malloc(n * rows * sizeof(float));
            for (uint32_t i = 0; i < n * rows; i++) {
                xv[i] = 0.03f * (float)(int32_t)(i % 200) - 2.0f;
            }
            cpu_rms_norm(ref, xv, NULL, n, rows, eps);
            ds4_gpu_tensor_write(x, 0, xv, n * rows * sizeof(float));
            CHECK(ds4_gpu_rms_norm_plain_rows_tensor(out, x, n, rows, eps) != 0,
                  "rms_norm: plain_rows");
            CHECK(ds4_gpu_synchronize() != 0, "rms_norm: sync");
            ds4_gpu_tensor_read(out, 0, ov, n * rows * sizeof(float));
            check_close(ov, ref, n * rows, 1e-4f,
                        "rms_norm: plain_rows matches CPU");
            free(xv); free(ov); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* rms_norm_weight / weight_rows (f32 weights from model) */
    {
        const uint32_t n = 128u, rows = 2u;
        const float eps = 1e-5f;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n * rows * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n * rows * sizeof(float));
        CHECK(x && out, "rms_norm_weight: alloc");
        if (x && out) {
            float *xv = (float *)malloc(n * rows * sizeof(float));
            float *ov = (float *)malloc(n * rows * sizeof(float));
            float *ref = (float *)malloc(n * rows * sizeof(float));
            float wf[128];
            for (uint32_t i = 0; i < n; i++) memcpy(&wf[i], g_model + OFF_F32_NORM + i * 4u, 4);
            for (uint32_t i = 0; i < n * rows; i++) {
                xv[i] = 0.02f * (float)(int32_t)(i % 300) - 1.5f;
            }
            cpu_rms_norm(ref, xv, wf, n, rows, eps);
            ds4_gpu_tensor_write(x, 0, xv, n * rows * sizeof(float));
            CHECK(ds4_gpu_rms_norm_weight_rows_tensor(
                      out, x, g_model, MODEL_PADDED, OFF_F32_NORM, n, rows, eps) != 0,
                  "rms_norm: weight_rows");
            CHECK(ds4_gpu_synchronize() != 0, "rms_norm_weight: sync");
            ds4_gpu_tensor_read(out, 0, ov, n * rows * sizeof(float));
            check_close(ov, ref, n * rows, 1e-4f,
                        "rms_norm: weight_rows matches CPU");
            free(xv); free(ov); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* rope_tail */
    {
        const uint32_t n_tok = 2u, n_head = 3u, head_dim = 64u, n_rot = 32u;
        const uint32_t total = n_tok * n_head * head_dim;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)total * sizeof(float));
        CHECK(x != NULL, "rope: alloc");
        if (x) {
            float *xv = (float *)malloc(total * sizeof(float));
            float *ov = (float *)malloc(total * sizeof(float));
            float *ref = (float *)malloc(total * sizeof(float));
            for (uint32_t i = 0; i < total; i++) xv[i] = 0.1f * (float)(i % 50) - 1.0f;
            memcpy(ref, xv, total * sizeof(float));
            cpu_rope_tail(ref, n_tok, n_head, head_dim, n_rot, 7u, 4096u, 0,
                          10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            ds4_gpu_tensor_write(x, 0, xv, total * sizeof(float));
            CHECK(ds4_gpu_rope_tail_tensor(x, n_tok, n_head, head_dim, n_rot,
                                           7u, 4096u, false, 10000.0f, 1.0f,
                                           0.0f, 1.0f, 32.0f, 1.0f) != 0,
                  "rope: rope_tail_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "rope: sync");
            ds4_gpu_tensor_read(x, 0, ov, total * sizeof(float));
            /* Cross-vendor cos/sin precision: the NVIDIA driver's transcendentals
             * differ from host libm by ~1.3e-5 relative on the tail angles, so
             * the 1e-5 tolerance is too tight for this test. */
            check_close(ov, ref, total, 2e-5f, "rope: tail matches CPU");
            free(xv); free(ov); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
    }

    /* matmul_q8_0 (decode, n_tok=1) + (prefill, n_tok=4) */
    {
        const uint32_t in_dim = 256u, out_dim = 32u;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)4 * in_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)4 * out_dim * sizeof(float));
        CHECK(x && out, "matmul_q8: alloc");
        if (x && out) {
            float *xv = (float *)malloc(4 * in_dim * sizeof(float));
            float *ov = (float *)malloc(4 * out_dim * sizeof(float));
            float *ref = (float *)malloc(4 * out_dim * sizeof(float));
            for (uint32_t i = 0; i < 4 * in_dim; i++) {
                xv[i] = 0.01f * (float)(int32_t)(i % 400) - 1.0f;
            }
            ds4_gpu_tensor_write(x, 0, xv, 4 * in_dim * sizeof(float));
            CHECK(ds4_gpu_matmul_q8_0_tensor(out, g_model, MODEL_PADDED, OFF_Q8,
                                             in_dim, out_dim, x, 4u) != 0,
                  "matmul_q8: q8_0_tensor (n_tok=4)");
            CHECK(ds4_gpu_synchronize() != 0, "matmul_q8: sync");
            ds4_gpu_tensor_read(out, 0, ov, 4 * out_dim * sizeof(float));
            cpu_matmul_q8(ref, g_model + OFF_Q8, out_dim, in_dim, xv, 4u);
            check_close(ov, ref, 4 * out_dim, 0.01f,
                        "matmul_q8: out matches CPU");
            free(xv); free(ov); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* matmul_f16 */
    {
        const uint32_t in_dim = 256u, out_dim = 32u, n_tok = 2u;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
        CHECK(x && out, "matmul_f16: alloc");
        if (x && out) {
            float *xv = (float *)malloc(n_tok * in_dim * sizeof(float));
            float *ov = (float *)malloc(n_tok * out_dim * sizeof(float));
            float *ref = (float *)malloc(n_tok * out_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tok * in_dim; i++) {
                xv[i] = 0.01f * (float)(int32_t)(i % 350) - 0.8f;
            }
            ds4_gpu_tensor_write(x, 0, xv, n_tok * in_dim * sizeof(float));
            CHECK(ds4_gpu_matmul_f16_tensor(out, g_model, MODEL_PADDED, OFF_F16,
                                            in_dim, out_dim, x, n_tok) != 0,
                  "matmul_f16: f16_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "matmul_f16: sync");
            ds4_gpu_tensor_read(out, 0, ov, n_tok * out_dim * sizeof(float));
            cpu_matmul_f16(ref, (const uint16_t *)(g_model + OFF_F16), out_dim,
                           in_dim, xv, n_tok);
            check_close(ov, ref, n_tok * out_dim, 0.01f,
                        "matmul_f16: out matches CPU");
            free(xv); free(ov); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* matmul_f16_pair_compressor_store (decode fusion) */
    {
        const uint32_t in_dim = 4096u, width = 256u, ratio = 4u, pos = 9u;
        const uint32_t state_rows = 2u * ratio;   /* 8 */
        const uint32_t stn = state_rows * width;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
        ds4_gpu_tensor *out_kv = ds4_gpu_tensor_alloc((uint64_t)width * sizeof(float));
        ds4_gpu_tensor *out_sc = ds4_gpu_tensor_alloc((uint64_t)width * sizeof(float));
        ds4_gpu_tensor *st_kv = ds4_gpu_tensor_alloc((uint64_t)stn * sizeof(float));
        ds4_gpu_tensor *st_sc = ds4_gpu_tensor_alloc((uint64_t)stn * sizeof(float));
        CHECK(x && out_kv && out_sc && st_kv && st_sc,
              "matmul_f16_comp: alloc");
        if (x && out_kv && out_sc && st_kv && st_sc) {
            float *xv = (float *)malloc((uint64_t)in_dim * sizeof(float));
            float *okv = (float *)malloc((uint64_t)width * sizeof(float));
            float *osc = (float *)malloc((uint64_t)width * sizeof(float));
            float *skv = (float *)malloc((uint64_t)stn * sizeof(float));
            float *ssc = (float *)malloc((uint64_t)stn * sizeof(float));
            float *rk = (float *)malloc((uint64_t)width * sizeof(float));
            float *rs = (float *)malloc((uint64_t)width * sizeof(float));
            const float *ape = (const float *)(g_model + OFF_COMP_APE);
            for (uint32_t i = 0; i < in_dim; i++) {
                xv[i] = 0.001f * (float)(int32_t)(i % 353) - 0.9f;
            }
            ds4_gpu_tensor_write(x, 0, xv, (uint64_t)in_dim * sizeof(float));
            memset(skv, 0, (uint64_t)stn * sizeof(float));
            memset(ssc, 0, (uint64_t)stn * sizeof(float));
            ds4_gpu_tensor_write(st_kv, 0, skv, (uint64_t)stn * sizeof(float));
            ds4_gpu_tensor_write(st_sc, 0, ssc, (uint64_t)stn * sizeof(float));
            const int fused = ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                    out_kv, out_sc, st_kv, st_sc, g_model, MODEL_PADDED,
                    OFF_COMP_KV, OFF_COMP_SC, OFF_COMP_APE, 0u, in_dim,
                    width, x, ratio, pos);
            CHECK(fused != 0, "matmul_f16_comp: fused store");
            CHECK(ds4_gpu_synchronize() != 0, "matmul_f16_comp: sync");
            ds4_gpu_tensor_read(out_kv, 0, okv, (uint64_t)width * sizeof(float));
            ds4_gpu_tensor_read(out_sc, 0, osc, (uint64_t)width * sizeof(float));
            ds4_gpu_tensor_read(st_kv, 0, skv, (uint64_t)stn * sizeof(float));
            ds4_gpu_tensor_read(st_sc, 0, ssc, (uint64_t)stn * sizeof(float));
            cpu_matmul_f16(rk, (const uint16_t *)(g_model + OFF_COMP_KV),
                           width, in_dim, xv, 1);
            cpu_matmul_f16(rs, (const uint16_t *)(g_model + OFF_COMP_SC),
                           width, in_dim, xv, 1);
            check_close(okv, rk, width, 0.02f,
                        "matmul_f16_comp: out_kv matches CPU");
            check_close(osc, rs, width, 0.02f,
                        "matmul_f16_comp: out_score matches CPU");
            const uint32_t pos_mod = pos % ratio;
            const uint32_t dst_row = ratio + pos_mod;
            int st_ok = 1;
            for (uint32_t j = 0; j < width; j++) {
                if (fabsf(skv[dst_row * width + j] - rk[j]) > 1e-3f ||
                    fabsf(ssc[dst_row * width + j] -
                          (rs[j] + ape[pos_mod * width + j])) > 1e-3f) {
                    st_ok = 0;
                }
            }
            CHECK(st_ok, "matmul_f16_comp: state row matches CPU");
            free(xv); free(okv); free(osc); free(skv); free(ssc);
            free(rk); free(rs);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (out_kv) ds4_gpu_tensor_free(out_kv);
        if (out_sc) ds4_gpu_tensor_free(out_sc);
        if (st_kv) ds4_gpu_tensor_free(st_kv);
        if (st_sc) ds4_gpu_tensor_free(st_sc);
    }

    /* matmul_f32 */
    {
        const uint32_t in_dim = 128u, out_dim = 16u, n_tok = 2u;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
        CHECK(x && out, "matmul_f32: alloc");
        if (x && out) {
            float *xv = (float *)malloc(n_tok * in_dim * sizeof(float));
            float *ov = (float *)malloc(n_tok * out_dim * sizeof(float));
            float *ref = (float *)malloc(n_tok * out_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tok * in_dim; i++) {
                xv[i] = 0.01f * (float)(int32_t)(i % 250) - 0.7f;
            }
            ds4_gpu_tensor_write(x, 0, xv, n_tok * in_dim * sizeof(float));
            CHECK(ds4_gpu_matmul_f32_tensor(out, g_model, MODEL_PADDED, OFF_F32_MM,
                                            in_dim, out_dim, x, n_tok) != 0,
                  "matmul_f32: f32_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "matmul_f32: sync");
            ds4_gpu_tensor_read(out, 0, ov, n_tok * out_dim * sizeof(float));
            cpu_matmul_f32(ref, (const float *)(g_model + OFF_F32_MM), out_dim,
                           in_dim, xv, n_tok);
            check_close(ov, ref, n_tok * out_dim, 0.01f,
                        "matmul_f32: out matches CPU");
            free(xv); free(ov); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* matmul_q8_0 pair (two projections sharing x) */
    {
        const uint32_t in_dim = 256u, out_dim = 32u, n_tok = 2u;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
        ds4_gpu_tensor *o0 = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
        ds4_gpu_tensor *o1 = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
        CHECK(x && o0 && o1, "pair: alloc");
        if (x && o0 && o1) {
            float *xv = (float *)malloc(n_tok * in_dim * sizeof(float));
            float *v0 = (float *)malloc(n_tok * out_dim * sizeof(float));
            float *v1 = (float *)malloc(n_tok * out_dim * sizeof(float));
            float *r0 = (float *)malloc(n_tok * out_dim * sizeof(float));
            float *r1 = (float *)malloc(n_tok * out_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tok * in_dim; i++) {
                xv[i] = 0.01f * (float)(int32_t)(i % 333) - 0.9f;
            }
            ds4_gpu_tensor_write(x, 0, xv, n_tok * in_dim * sizeof(float));
            CHECK(ds4_gpu_matmul_q8_0_pair_tensor(
                      o0, o1, g_model, MODEL_PADDED, OFF_Q8, OFF_Q8,
                      in_dim, out_dim, out_dim, x, n_tok) != 0,
                  "pair: q8_0_pair_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "pair: sync");
            ds4_gpu_tensor_read(o0, 0, v0, n_tok * out_dim * sizeof(float));
            ds4_gpu_tensor_read(o1, 0, v1, n_tok * out_dim * sizeof(float));
            cpu_matmul_q8(r0, g_model + OFF_Q8, out_dim, in_dim, xv, n_tok);
            cpu_matmul_q8(r1, g_model + OFF_Q8, out_dim, in_dim, xv, n_tok);
            check_close(v0, r0, n_tok * out_dim, 0.01f, "pair: out0 matches CPU");
            check_close(v1, r1, n_tok * out_dim, 0.01f, "pair: out1 matches CPU");
            free(xv); free(v0); free(v1); free(r0); free(r1);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (o0) ds4_gpu_tensor_free(o0);
        if (o1) ds4_gpu_tensor_free(o1);
    }

    /* matmul_q8_0 kslice (partial K, single token via x view) */
    {
        const uint32_t full_in = 256u, out_dim = 32u, k_off = 64u, k_cnt = 128u;
        const uint32_t x_elem_off = 32u;   /* slice starts at x[32] */
        const uint32_t x_len = x_elem_off + k_cnt;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)x_len * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
        CHECK(x && out, "kslice: alloc");
        if (x && out) {
            float *xv = (float *)malloc(x_len * sizeof(float));
            float *ov = (float *)malloc(out_dim * sizeof(float));
            float *ref = (float *)malloc(out_dim * sizeof(float));
            for (uint32_t i = 0; i < x_len; i++) {
                xv[i] = 0.01f * (float)(int32_t)((i * 7u) % 200) - 0.6f;
            }
            ds4_gpu_tensor_write(x, 0, xv, x_len * sizeof(float));
            CHECK(ds4_gpu_matmul_q8_0_kslice_tensor(
                      out, g_model, MODEL_PADDED, OFF_Q8, full_in, k_off,
                      k_cnt, out_dim, x, x_elem_off) != 0,
                  "kslice: kslice_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "kslice: sync");
            ds4_gpu_tensor_read(out, 0, ov, out_dim * sizeof(float));
            cpu_matmul_q8_kslice(ref, g_model + OFF_Q8, out_dim, full_in / 32u,
                                 k_off / 32u, k_cnt / 32u, k_cnt,
                                 xv + x_elem_off, 1u);
            check_close(ov, ref, out_dim, 0.01f, "kslice: out matches CPU");
            free(xv); free(ov); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* matmul_q8_0 f16_out (f16 activations output) */
    {
        const uint32_t in_dim = 256u, out_dim = 32u, n_tok = 2u;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
        ds4_gpu_tensor *oh = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(uint16_t));
        CHECK(x && oh, "f16_out: alloc");
        if (x && oh) {
            float *xv = (float *)malloc(n_tok * in_dim * sizeof(float));
            uint16_t *hv = (uint16_t *)malloc(n_tok * out_dim * sizeof(uint16_t));
            float *fv = (float *)malloc(n_tok * out_dim * sizeof(float));
            float *ref = (float *)malloc(n_tok * out_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tok * in_dim; i++) {
                xv[i] = 0.01f * (float)(int32_t)(i % 300) - 0.7f;
            }
            ds4_gpu_tensor_write(x, 0, xv, n_tok * in_dim * sizeof(float));
            CHECK(ds4_gpu_matmul_q8_0_f16_out_tensor(
                      oh, g_model, MODEL_PADDED, OFF_Q8, in_dim, out_dim, x,
                      n_tok) != 0,
                  "f16_out: q8_0_f16_out_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "f16_out: sync");
            ds4_gpu_tensor_read(oh, 0, hv, n_tok * out_dim * sizeof(uint16_t));
            for (uint32_t i = 0; i < n_tok * out_dim; i++) fv[i] = f16_to_f32(hv[i]);
            cpu_matmul_q8(ref, g_model + OFF_Q8, out_dim, in_dim, xv, n_tok);
            check_close(fv, ref, n_tok * out_dim, 0.01f,
                        "f16_out: output matches CPU (f16)");
            free(xv); free(hv); free(fv); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (oh) ds4_gpu_tensor_free(oh);
    }

    /* matmul_q8_0 top1 (fused matmul + argmax) */
    {
        const uint32_t in_dim = 256u, out_dim = 32u, index_offset = 100u;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
        ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc(sizeof(uint32_t));
        ds4_gpu_tensor *val = ds4_gpu_tensor_alloc(sizeof(float));
        CHECK(x && sel && val, "top1: alloc");
        if (x && sel && val) {
            float *xv = (float *)malloc(in_dim * sizeof(float));
            for (uint32_t i = 0; i < in_dim; i++) {
                xv[i] = 0.01f * (float)(int32_t)((i * 3u) % 150) - 0.5f;
            }
            ds4_gpu_tensor_write(x, 0, xv, in_dim * sizeof(float));
            CHECK(ds4_gpu_matmul_q8_0_top1_tensor(
                      sel, val, g_model, MODEL_PADDED, OFF_Q8, in_dim,
                      out_dim, x, index_offset) != 0,
                  "top1: q8_0_top1_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "top1: sync");
            uint32_t got_sel = 0;
            float got_val = 0.0f;
            ds4_gpu_tensor_read(sel, 0, &got_sel, sizeof(got_sel));
            ds4_gpu_tensor_read(val, 0, &got_val, sizeof(got_val));
            float *ref = (float *)malloc(out_dim * sizeof(float));
            cpu_matmul_q8(ref, g_model + OFF_Q8, out_dim, in_dim, xv, 1u);
            uint32_t best = 0;
            float bv = ref[0];
            for (uint32_t r = 1; r < out_dim; r++) {
                if (ref[r] > bv) { bv = ref[r]; best = r; }
            }
            CHECK(got_sel == best + index_offset, "top1: selected == argmax + offset");
            CHECK(fabsf(got_val - bv) <= 0.01f * (fabsf(bv) > 1.0f ? fabsf(bv) : 1.0f),
                  "top1: value matches argmax");
            free(xv); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (sel) ds4_gpu_tensor_free(sel);
        if (val) ds4_gpu_tensor_free(val);
    }

    /* GPU copy inside a command scope (vkCmdCopyBuffer path) */
    {
        const uint64_t bytes = 4096u;
        ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(bytes);
        CHECK(a && b, "gpu_copy: alloc");
        if (a && b) {
            uint8_t *av = (uint8_t *)malloc(bytes);
            uint8_t *bv = (uint8_t *)malloc(bytes);
            for (uint32_t i = 0; i < bytes; i++) av[i] = (uint8_t)(i & 0xff);
            ds4_gpu_tensor_write(a, 0, av, bytes);
            CHECK(ds4_gpu_begin_commands() != 0, "gpu_copy: begin_commands");
            CHECK(ds4_gpu_tensor_copy(b, 0, a, 0, bytes) != 0,
                  "gpu_copy: tensor_copy in scope");
            CHECK(ds4_gpu_end_commands() != 0, "gpu_copy: end_commands");
            CHECK(ds4_gpu_synchronize() != 0, "gpu_copy: synchronize");
            ds4_gpu_tensor_read(b, 0, bv, bytes);
            CHECK(memcmp(av, bv, bytes) == 0, "gpu_copy: round-trip matches");
            free(av); free(bv);
        }
        if (a) ds4_gpu_tensor_free(a);
        if (b) ds4_gpu_tensor_free(b);
    }

    /* matmul_f16 rms_fold (folded RMS norm into F16 projection) */
    {
        const uint32_t in_dim = 256u, out_dim = 32u, n_tok = 2u;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
        CHECK(x && out, "rms_fold: alloc");
        if (x && out) {
            float *xv = (float *)malloc(n_tok * in_dim * sizeof(float));
            float *ov = (float *)malloc(n_tok * out_dim * sizeof(float));
            float *ref = (float *)malloc(n_tok * out_dim * sizeof(float));
            const float eps = 1e-5f;
            for (uint32_t i = 0; i < n_tok * in_dim; i++) {
                xv[i] = 0.02f * (float)(int32_t)(i % 220) - 1.0f;
            }
            ds4_gpu_tensor_write(x, 0, xv, n_tok * in_dim * sizeof(float));
            CHECK(ds4_gpu_matmul_f16_rms_fold_tensor(
                      out, g_model, MODEL_PADDED, OFF_F16, in_dim, out_dim, x,
                      n_tok, eps) != 0,
                  "rms_fold: f16_rms_fold_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "rms_fold: sync");
            ds4_gpu_tensor_read(out, 0, ov, n_tok * out_dim * sizeof(float));
            float *norm = (float *)malloc(n_tok * in_dim * sizeof(float));
            cpu_rms_norm(norm, xv, NULL, in_dim, n_tok, eps);
            cpu_matmul_f16(ref, (const uint16_t *)(g_model + OFF_F16), out_dim,
                           in_dim, norm, n_tok);
            check_close(ov, ref, n_tok * out_dim, 0.01f,
                        "rms_fold: out matches CPU (norm-fold)");
            free(xv); free(ov); free(ref); free(norm);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* embed_token_q8_0 + embed_tokens_q8_0 */
    {
        const uint32_t n_vocab = 64u, n_embd = 256u, n_tokens = 3u;
        ds4_gpu_tensor *out1 = ds4_gpu_tensor_alloc((uint64_t)n_embd * sizeof(float));
        ds4_gpu_tensor *tok = ds4_gpu_tensor_alloc((uint64_t)n_tokens * sizeof(int32_t));
        ds4_gpu_tensor *outn = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_embd * sizeof(float));
        CHECK(out1 && tok && outn, "embed: alloc");
        if (out1 && tok && outn) {
            float *o1 = (float *)malloc(n_embd * sizeof(float));
            float *on = (float *)malloc(n_tokens * n_embd * sizeof(float));
            float *ref1 = (float *)malloc(n_embd * sizeof(float));
            float *refn = (float *)malloc(n_tokens * n_embd * sizeof(float));
            const uint32_t row_blocks = n_embd / 32u;
            int32_t toks[3] = { 5, 33, 61 };
            ds4_gpu_tensor_write(tok, 0, toks, sizeof(toks));
            CHECK(ds4_gpu_embed_token_q8_0_tensor(out1, g_model, MODEL_PADDED,
                                                  OFF_EMB, n_vocab, 5u, n_embd) != 0,
                  "embed: token_q8_0");
            CHECK(ds4_gpu_embed_tokens_q8_0_tensor(outn, tok, g_model, MODEL_PADDED,
                                                   OFF_EMB, n_vocab, n_tokens,
                                                   n_embd) != 0,
                  "embed: tokens_q8_0");
            CHECK(ds4_gpu_synchronize() != 0, "embed: sync");
            ds4_gpu_tensor_read(out1, 0, o1, n_embd * sizeof(float));
            ds4_gpu_tensor_read(outn, 0, on, n_tokens * n_embd * sizeof(float));
            int ok1 = 1, okn = 1;
            for (uint32_t d = 0; d < n_embd; d++) {
                ref1[d] = q8_dequant(g_model + OFF_EMB +
                                         (uint64_t)(5u * row_blocks + d / 32u) * 34u,
                                     d & 31u);
                if (fabsf(o1[d] - ref1[d]) > 1e-4f) { ok1 = 0; break; }
            }
            CHECK(ok1, "embed: token_q8_0 matches CPU");
            for (uint32_t t2 = 0; t2 < n_tokens; t2++) {
                for (uint32_t d = 0; d < n_embd; d++) {
                    refn[t2 * n_embd + d] =
                        q8_dequant(g_model + OFF_EMB +
                                       (uint64_t)((uint32_t)toks[t2] * row_blocks +
                                                  d / 32u) * 34u,
                                   d & 31u);
                    if (fabsf(on[t2 * n_embd + d] - refn[t2 * n_embd + d]) >
                        1e-4f) { okn = 0; break; }
                }
            }
            CHECK(okn, "embed: tokens_q8_0 matches CPU");
            free(o1); free(on); free(ref1); free(refn);
        }
        if (out1) ds4_gpu_tensor_free(out1);
        if (tok) ds4_gpu_tensor_free(tok);
        if (outn) ds4_gpu_tensor_free(outn);
    }

    /* embed_token_hc / embed_tokens_hc (f16 token_embd broadcast to n_hc) */
    {
        const uint32_t n_vocab = 16u, n_embd = 64u, n_hc = 4u, n_tokens = 3u;
        ds4_gpu_tensor *oh = ds4_gpu_tensor_alloc(
                (uint64_t)n_hc * n_embd * sizeof(float));
        ds4_gpu_tensor *tok = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * sizeof(int32_t));
        ds4_gpu_tensor *on = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * n_hc * n_embd * sizeof(float));
        CHECK(oh && tok && on, "embed_hc: alloc");
        if (oh && tok && on) {
            float *o1 = (float *)malloc(n_hc * n_embd * sizeof(float));
            float *onv = (float *)malloc(n_tokens * n_hc * n_embd * sizeof(float));
            float *ref = (float *)malloc(n_hc * n_embd * sizeof(float));
            const int32_t toks[3] = { 2, 9, 15 };
            const uint16_t *w = (const uint16_t *)(g_model + OFF_EMB_F16);
            ds4_gpu_tensor_write(tok, 0, toks, sizeof(toks));
            CHECK(ds4_gpu_embed_token_hc_tensor(oh, g_model, MODEL_PADDED,
                                                OFF_EMB_F16, n_vocab, 2u,
                                                n_embd, n_hc) != 0,
                  "embed_hc: token_hc");
            CHECK(ds4_gpu_embed_tokens_hc_tensor(on, tok, g_model, MODEL_PADDED,
                                                 OFF_EMB_F16, n_vocab, n_tokens,
                                                 n_embd, n_hc) != 0,
                  "embed_hc: tokens_hc");
            CHECK(ds4_gpu_synchronize() != 0, "embed_hc: sync");
            ds4_gpu_tensor_read(oh, 0, o1, n_hc * n_embd * sizeof(float));
            ds4_gpu_tensor_read(on, 0, onv,
                                n_tokens * n_hc * n_embd * sizeof(float));
            int ok1 = 1, okn = 1;
            for (uint32_t h = 0; h < n_hc && ok1; h++) {
                for (uint32_t d = 0; d < n_embd; d++) {
                    ref[h * n_embd + d] = f16_to_f32(w[2u * n_embd + d]);
                    if (fabsf(o1[h * n_embd + d] - ref[h * n_embd + d]) > 1e-4f) {
                        ok1 = 0; break;
                    }
                }
            }
            CHECK(ok1, "embed_hc: token_hc matches CPU");
            for (uint32_t t2 = 0; t2 < n_tokens && okn; t2++) {
                for (uint32_t h = 0; h < n_hc && okn; h++) {
                    for (uint32_t d = 0; d < n_embd; d++) {
                        const float exp = f16_to_f32(
                                w[(uint32_t)toks[t2] * n_embd + d]);
                        if (fabsf(onv[((uint64_t)t2 * n_hc + h) * n_embd + d] -
                                  exp) > 1e-4f) { okn = 0; break; }
                    }
                }
            }
            CHECK(okn, "embed_hc: tokens_hc matches CPU");
            free(o1); free(onv); free(ref);
        }
        if (oh) ds4_gpu_tensor_free(oh);
        if (tok) ds4_gpu_tensor_free(tok);
        if (on) ds4_gpu_tensor_free(on);
    }

    /* --- Fase 3: FP8 KV quantize + raw stores ----------------------------- */
    {
        const uint32_t head_dim = 128u, n_rot = 32u, n_tok = 3u;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * head_dim * sizeof(float));
        CHECK(x != NULL, "fp8_kv: alloc");
        if (x) {
            float *xv = (float *)malloc((uint64_t)n_tok * head_dim * sizeof(float));
            float *ref = (float *)malloc((uint64_t)n_tok * head_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tok * head_dim; i++) {
                xv[i] = 0.5f * sinf((float)i * 0.7f) * (float)(1 + (i % 5));
            }
            memcpy(ref, xv, (uint64_t)n_tok * head_dim * sizeof(float));
            cpu_fp8_kv_quantize(ref, n_tok, head_dim, n_rot);
            ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tok * head_dim * sizeof(float));
            CHECK(ds4_gpu_dsv4_fp8_kv_quantize_tensor(x, n_tok, head_dim, n_rot) != 0,
                  "fp8_kv: quantize_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "fp8_kv: sync");
            ds4_gpu_tensor_read(x, 0, xv, (uint64_t)n_tok * head_dim * sizeof(float));
            check_close(xv, ref, n_tok * head_dim, 1e-4f,
                        "fp8_kv: quantize matches CPU");
            free(xv); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
    }

    /* store_raw_kv (single) + store_raw_kv_batch */
    {
        const uint32_t head_dim = 64u, raw_cap = 8u, n_tokens = 3u;
        ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)n_tokens * head_dim * sizeof(float));
        ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc((uint64_t)raw_cap * head_dim * sizeof(float));
        CHECK(kv && raw, "store_raw_kv: alloc");
        if (kv && raw) {
            float *kvv = (float *)malloc((uint64_t)n_tokens * head_dim * sizeof(float));
            float *rawv = (float *)malloc((uint64_t)raw_cap * head_dim * sizeof(float));
            float *ref = (float *)malloc((uint64_t)raw_cap * head_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tokens * head_dim; i++) {
                kvv[i] = 0.4f * (float)(int32_t)(i % 40) - 3.0f;
            }
            memset(rawv, 0, (uint64_t)raw_cap * head_dim * sizeof(float));
            memset(ref, 0, (uint64_t)raw_cap * head_dim * sizeof(float));
            ds4_gpu_tensor_write(kv, 0, kvv, (uint64_t)n_tokens * head_dim * sizeof(float));
            ds4_gpu_tensor_write(raw, 0, rawv, (uint64_t)raw_cap * head_dim * sizeof(float));
            CHECK(ds4_gpu_store_raw_kv_batch_tensor(raw, kv, raw_cap, 5u, n_tokens,
                                                    head_dim) != 0,
                  "store_raw_kv: batch_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "store_raw_kv: sync");
            ds4_gpu_tensor_read(raw, 0, rawv, (uint64_t)raw_cap * head_dim * sizeof(float));
            cpu_store_raw_kv(ref, kvv, raw_cap, 5u, n_tokens, head_dim);
            check_close(rawv, ref, raw_cap * head_dim, 1e-4f,
                        "store_raw_kv: batch matches CPU");
            /* single-row store: pos0 = row index */
            memset(rawv, 0, (uint64_t)raw_cap * head_dim * sizeof(float));
            ds4_gpu_tensor_write(raw, 0, rawv, (uint64_t)raw_cap * head_dim * sizeof(float));
            CHECK(ds4_gpu_store_raw_kv_tensor(raw, kv, raw_cap, 2u, head_dim) != 0,
                  "store_raw_kv: single_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "store_raw_kv: sync single");
            ds4_gpu_tensor_read(raw, 0, rawv, (uint64_t)raw_cap * head_dim * sizeof(float));
            memset(ref, 0, (uint64_t)raw_cap * head_dim * sizeof(float));
            cpu_store_raw_kv(ref, kvv, raw_cap, 2u, 1u, head_dim);
            check_close(rawv, ref, raw_cap * head_dim, 1e-4f,
                        "store_raw_kv: single matches CPU");
            free(kvv); free(rawv); free(ref);
        }
        if (kv) ds4_gpu_tensor_free(kv);
        if (raw) ds4_gpu_tensor_free(raw);
    }

    /* kv_fp8_store_raw (quantize + f16 store fused) */
    {
        const uint32_t head_dim = 128u, n_rot = 32u, raw_cap = 8u, raw_row = 3u;
        ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)head_dim * sizeof(float));
        ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc((uint64_t)raw_cap * head_dim * sizeof(float));
        CHECK(kv && raw, "kv_fp8_store_raw: alloc");
        if (kv && raw) {
            float *kvv = (float *)malloc((uint64_t)head_dim * sizeof(float));
            float *rawv = (float *)malloc((uint64_t)raw_cap * head_dim * sizeof(float));
            float *ref_kv = (float *)malloc((uint64_t)head_dim * sizeof(float));
            float *ref_raw = (float *)malloc((uint64_t)raw_cap * head_dim * sizeof(float));
            for (uint32_t i = 0; i < head_dim; i++) {
                kvv[i] = 0.6f * sinf((float)i) * (float)(1 + (i % 7));
            }
            memcpy(ref_kv, kvv, (uint64_t)head_dim * sizeof(float));
            memset(ref_raw, 0, (uint64_t)raw_cap * head_dim * sizeof(float));
            cpu_kv_fp8_store_raw(ref_kv, ref_raw, raw_cap, raw_row, head_dim, n_rot);
            ds4_gpu_tensor_write(kv, 0, kvv, (uint64_t)head_dim * sizeof(float));
            CHECK(ds4_gpu_kv_fp8_store_raw_tensor(kv, raw, raw_cap, raw_row, head_dim,
                                                  n_rot) != 0,
                  "kv_fp8_store_raw: store_tensor");
            CHECK(ds4_gpu_synchronize() != 0, "kv_fp8_store_raw: sync");
            ds4_gpu_tensor_read(kv, 0, kvv, (uint64_t)head_dim * sizeof(float));
            ds4_gpu_tensor_read(raw, 0, rawv, (uint64_t)raw_cap * head_dim * sizeof(float));
            check_close(kvv, ref_kv, head_dim, 1e-4f,
                        "kv_fp8_store_raw: kv quantized matches CPU");
            check_close(rawv, ref_raw, raw_cap * head_dim, 1e-4f,
                        "kv_fp8_store_raw: raw matches CPU");
            free(kvv); free(rawv); free(ref_kv); free(ref_raw);
        }
        if (kv) ds4_gpu_tensor_free(kv);
        if (raw) ds4_gpu_tensor_free(raw);
    }

    /* --- Fase 3: attention decode (single token, heads) -------------------- */
    {
        const uint32_t n_head = 4u, head_dim = 32u, raw_cap = 16u;
        const uint32_t n_raw = 8u, raw_start = 5u, n_comp = 3u;
        ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)n_head * head_dim * sizeof(float));
        ds4_gpu_tensor *raw_kv = ds4_gpu_tensor_alloc((uint64_t)raw_cap * head_dim * sizeof(float));
        ds4_gpu_tensor *comp_kv = ds4_gpu_tensor_alloc((uint64_t)n_comp * head_dim * sizeof(float));
        ds4_gpu_tensor *comp_mask = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
        ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc((uint64_t)n_head * head_dim * sizeof(float));
        CHECK(q && raw_kv && comp_kv && comp_mask && heads, "attn_decode: alloc");
        if (q && raw_kv && comp_kv && comp_mask && heads) {
            float *qv = (float *)malloc((uint64_t)n_head * head_dim * sizeof(float));
            float *rv = (float *)malloc((uint64_t)raw_cap * head_dim * sizeof(float));
            float *cv = (float *)malloc((uint64_t)n_comp * head_dim * sizeof(float));
            float *mv = (float *)malloc((uint64_t)n_comp * sizeof(float));
            float *hv = (float *)malloc((uint64_t)n_head * head_dim * sizeof(float));
            float *ref = (float *)malloc((uint64_t)n_head * head_dim * sizeof(float));
            for (uint32_t i = 0; i < n_head * head_dim; i++) qv[i] = 0.3f * sinf((float)i);
            for (uint32_t i = 0; i < raw_cap * head_dim; i++) rv[i] = 0.2f * cosf((float)i * 0.5f);
            for (uint32_t i = 0; i < n_comp * head_dim; i++) cv[i] = 0.25f * sinf((float)i * 0.3f);
            for (uint32_t i = 0; i < n_comp; i++) mv[i] = (float)(i == 1 ? -1.0e30f : 0.5f * i);
            ds4_gpu_tensor_write(q, 0, qv, (uint64_t)n_head * head_dim * sizeof(float));
            ds4_gpu_tensor_write(raw_kv, 0, rv, (uint64_t)raw_cap * head_dim * sizeof(float));
            ds4_gpu_tensor_write(comp_kv, 0, cv, (uint64_t)n_comp * head_dim * sizeof(float));
            ds4_gpu_tensor_write(comp_mask, 0, mv, (uint64_t)n_comp * sizeof(float));
            cpu_attn_decode(ref, (const float *)(g_model + OFF_SINKS), qv, rv,
                            n_raw, raw_cap, raw_start, cv, n_comp, mv, 1,
                            1u, 0u, 0u, 0u, n_head, head_dim);
            CHECK(ds4_gpu_attention_decode_heads_tensor(
                      heads, g_model, MODEL_PADDED, OFF_SINKS, q, raw_kv,
                      n_raw, raw_cap, raw_start, comp_kv, 0, n_comp, comp_mask,
                      1, n_head, head_dim) != 0,
                  "attn_decode: decode_heads");
            CHECK(ds4_gpu_synchronize() != 0, "attn_decode: sync");
            ds4_gpu_tensor_read(heads, 0, hv, (uint64_t)n_head * head_dim * sizeof(float));
            check_close(hv, ref, n_head * head_dim, 1e-3f,
                        "attn_decode: heads match CPU");
            /* unmasked variant */
            cpu_attn_decode(ref, (const float *)(g_model + OFF_SINKS), qv, rv,
                            n_raw, raw_cap, raw_start, cv, n_comp, NULL, 0,
                            1u, 0u, 0u, 0u, n_head, head_dim);
            CHECK(ds4_gpu_attention_decode_heads_tensor(
                      heads, g_model, MODEL_PADDED, OFF_SINKS, q, raw_kv,
                      n_raw, raw_cap, raw_start, comp_kv, 0, n_comp, NULL,
                      0, n_head, head_dim) != 0,
                  "attn_decode: decode_heads unmasked");
            CHECK(ds4_gpu_synchronize() != 0, "attn_decode: sync unmasked");
            ds4_gpu_tensor_read(heads, 0, hv, (uint64_t)n_head * head_dim * sizeof(float));
            check_close(hv, ref, n_head * head_dim, 1e-3f,
                        "attn_decode: heads unmasked match CPU");
            free(qv); free(rv); free(cv); free(mv); free(hv); free(ref);
        }
        if (q) ds4_gpu_tensor_free(q);
        if (raw_kv) ds4_gpu_tensor_free(raw_kv);
        if (comp_kv) ds4_gpu_tensor_free(comp_kv);
        if (comp_mask) ds4_gpu_tensor_free(comp_mask);
        if (heads) ds4_gpu_tensor_free(heads);
    }

    /* --- Fase 3: attention indexed mixed decode (topk-selected comp rows) - */
    {
        const uint32_t n_head = 4u, head_dim = 32u, raw_cap = 16u;
        const uint32_t n_raw = 8u, raw_start = 5u, n_comp = 6u;
        const uint32_t top_k = 4u, window = 4u, ratio = 2u;
        const uint32_t n_tokens = 2u, pos0 = 4u;
        ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
        ds4_gpu_tensor *raw_kv = ds4_gpu_tensor_alloc(
                (uint64_t)raw_cap * head_dim * sizeof(float));
        ds4_gpu_tensor *comp_kv = ds4_gpu_tensor_alloc(
                (uint64_t)n_comp * head_dim * sizeof(float));
        ds4_gpu_tensor *topk = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * top_k * sizeof(uint32_t));
        ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
        CHECK(q && raw_kv && comp_kv && topk && heads, "attn_indexed_decode: alloc");
        if (q && raw_kv && comp_kv && topk && heads) {
            float *qv = (float *)malloc(
                    (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            float *rv = (float *)malloc((uint64_t)raw_cap * head_dim * sizeof(float));
            float *cv = (float *)malloc((uint64_t)n_comp * head_dim * sizeof(float));
            uint32_t *tv = (uint32_t *)malloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
            float *hv = (float *)malloc(
                    (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            float *ref = (float *)malloc(
                    (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tokens * n_head * head_dim; i++)
                qv[i] = 0.3f * sinf((float)i);
            for (uint32_t i = 0; i < raw_cap * head_dim; i++)
                rv[i] = 0.2f * cosf((float)i * 0.5f);
            for (uint32_t i = 0; i < n_comp * head_dim; i++)
                cv[i] = 0.25f * sinf((float)i * 0.3f);
            /* token 0: topk {1,5,2,0} -> visible_comp=2 -> comp_rows {1,0}
             * token 1: topk {3,1,0,2} -> visible_comp=3 -> comp_rows {1,0,2} */
            const uint32_t tk[2][4] = {{1, 5, 2, 0}, {3, 1, 0, 2}};
            for (uint32_t t = 0; t < n_tokens; t++)
                for (uint32_t k = 0; k < top_k; k++)
                    tv[t * top_k + k] = tk[t][k];
            ds4_gpu_tensor_write(q, 0, qv,
                                 (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            ds4_gpu_tensor_write(raw_kv, 0, rv, (uint64_t)raw_cap * head_dim * sizeof(float));
            ds4_gpu_tensor_write(comp_kv, 0, cv, (uint64_t)n_comp * head_dim * sizeof(float));
            ds4_gpu_tensor_write(topk, 0, tv, (uint64_t)n_tokens * top_k * sizeof(uint32_t));
            cpu_attn_indexed_decode(ref, (const float *)(g_model + OFF_SINKS), qv,
                                    rv, n_raw, raw_cap, raw_start, cv, n_comp,
                                    tv, top_k, n_tokens, pos0, window, ratio,
                                    n_head, head_dim);
            CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                      heads, g_model, MODEL_PADDED, OFF_SINKS, q, raw_kv,
                      comp_kv, 0, topk, n_tokens, pos0, n_raw, raw_cap,
                      raw_start, n_comp, top_k, window, ratio, n_head,
                      head_dim) != 0,
                  "attn_indexed_decode: indexed_mixed_batch");
            CHECK(ds4_gpu_synchronize() != 0, "attn_indexed_decode: sync");
            ds4_gpu_tensor_read(heads, 0, hv,
                                (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            check_close(hv, ref, n_tokens * n_head * head_dim, 1e-3f,
                        "attn_indexed_decode: heads match CPU");
            free(qv); free(rv); free(cv); free(tv); free(hv); free(ref);
        }
        if (q) ds4_gpu_tensor_free(q);
        if (raw_kv) ds4_gpu_tensor_free(raw_kv);
        if (comp_kv) ds4_gpu_tensor_free(comp_kv);
        if (topk) ds4_gpu_tensor_free(topk);
        if (heads) ds4_gpu_tensor_free(heads);
    }

    /* --- Fase 3: attention decode raw batch (n_tokens) --------------------- */
    {
        const uint32_t n_head = 3u, head_dim = 32u, raw_cap = 24u;
        const uint32_t n_tokens = 4u, pos0 = 3u, n_raw = 20u, raw_start = 2u;
        const uint32_t window = 8u;
        ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_head * head_dim * sizeof(float));
        ds4_gpu_tensor *raw_kv = ds4_gpu_tensor_alloc((uint64_t)raw_cap * head_dim * sizeof(float));
        ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_head * head_dim * sizeof(float));
        CHECK(q && raw_kv && heads, "attn_batch: alloc");
        if (q && raw_kv && heads) {
            float *qv = (float *)malloc((uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            float *rv = (float *)malloc((uint64_t)raw_cap * head_dim * sizeof(float));
            float *hv = (float *)malloc((uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            float *ref = (float *)malloc((uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tokens * n_head * head_dim; i++) qv[i] = 0.2f * cosf((float)i);
            for (uint32_t i = 0; i < raw_cap * head_dim; i++) rv[i] = 0.3f * sinf((float)i * 0.4f);
            ds4_gpu_tensor_write(q, 0, qv, (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            ds4_gpu_tensor_write(raw_kv, 0, rv, (uint64_t)raw_cap * head_dim * sizeof(float));
            cpu_attn_decode(ref, (const float *)(g_model + OFF_SINKS), qv, rv,
                            n_raw, raw_cap, raw_start, NULL, 0, NULL, 0,
                            n_tokens, pos0, window, 1u, n_head, head_dim);
            CHECK(ds4_gpu_attention_decode_raw_batch_heads_tensor(
                      heads, g_model, MODEL_PADDED, OFF_SINKS, q, raw_kv,
                      n_tokens, pos0, n_raw, raw_cap, raw_start, window,
                      n_head, head_dim) != 0,
                  "attn_batch: raw_batch_heads");
            CHECK(ds4_gpu_synchronize() != 0, "attn_batch: sync");
            ds4_gpu_tensor_read(heads, 0, hv, (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            check_close(hv, ref, n_tokens * n_head * head_dim, 1e-3f,
                        "attn_batch: raw batch heads match CPU");
            free(qv); free(rv); free(hv); free(ref);
        }
        if (q) ds4_gpu_tensor_free(q);
        if (raw_kv) ds4_gpu_tensor_free(raw_kv);
        if (heads) ds4_gpu_tensor_free(heads);
    }

    /* --- Fase 3: attention prefill raw + static/masked mixed --------------- */
    {
        const uint32_t n_head = 3u, head_dim = 32u, n_tokens = 6u;
        const uint32_t window = 5u, n_comp = 4u, ratio = 2u;
        ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_head * head_dim * sizeof(float));
        ds4_gpu_tensor *raw_kv = ds4_gpu_tensor_alloc((uint64_t)n_tokens * head_dim * sizeof(float));
        ds4_gpu_tensor *comp_kv = ds4_gpu_tensor_alloc((uint64_t)n_comp * head_dim * sizeof(float));
        ds4_gpu_tensor *comp_mask = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_comp * sizeof(float));
        ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_head * head_dim * sizeof(float));
        CHECK(q && raw_kv && comp_kv && comp_mask && heads, "attn_prefill: alloc");
        if (q && raw_kv && comp_kv && comp_mask && heads) {
            float *qv = (float *)malloc((uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            float *rv = (float *)malloc((uint64_t)n_tokens * head_dim * sizeof(float));
            float *cv = (float *)malloc((uint64_t)n_comp * head_dim * sizeof(float));
            float *mv = (float *)malloc((uint64_t)n_tokens * n_comp * sizeof(float));
            float *hv = (float *)malloc((uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            float *ref = (float *)malloc((uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tokens * n_head * head_dim; i++) qv[i] = 0.2f * sinf((float)i * 0.3f);
            for (uint32_t i = 0; i < n_tokens * head_dim; i++) rv[i] = 0.3f * cosf((float)i * 0.7f);
            for (uint32_t i = 0; i < n_comp * head_dim; i++) cv[i] = 0.25f * sinf((float)i * 0.5f);
            for (uint32_t i = 0; i < n_tokens * n_comp; i++) {
                mv[i] = (i % n_comp == 1u) ? -1.0e30f : 0.4f * (float)(i % n_comp);
            }
            ds4_gpu_tensor_write(q, 0, qv, (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            ds4_gpu_tensor_write(raw_kv, 0, rv, (uint64_t)n_tokens * head_dim * sizeof(float));
            ds4_gpu_tensor_write(comp_kv, 0, cv, (uint64_t)n_comp * head_dim * sizeof(float));
            ds4_gpu_tensor_write(comp_mask, 0, mv, (uint64_t)n_tokens * n_comp * sizeof(float));

            /* prefill raw */
            cpu_attn_prefill(ref, (const float *)(g_model + OFF_SINKS), qv, rv,
                             NULL, 0, NULL, 0, n_tokens, window, 1u, n_head,
                             head_dim);
            CHECK(ds4_gpu_attention_prefill_raw_heads_tensor(
                      heads, g_model, MODEL_PADDED, OFF_SINKS, q, raw_kv,
                      n_tokens, window, n_head, head_dim) != 0,
                  "attn_prefill: raw_heads");
            CHECK(ds4_gpu_synchronize() != 0, "attn_prefill: sync raw");
            ds4_gpu_tensor_read(heads, 0, hv, (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            check_close(hv, ref, n_tokens * n_head * head_dim, 1e-3f,
                        "attn_prefill: raw heads match CPU");

            /* static mixed */
            cpu_attn_prefill(ref, (const float *)(g_model + OFF_SINKS), qv, rv,
                             cv, n_comp, NULL, 0, n_tokens, window, ratio,
                             n_head, head_dim);
            CHECK(ds4_gpu_attention_prefill_static_mixed_heads_tensor(
                      heads, g_model, MODEL_PADDED, OFF_SINKS, q, raw_kv, comp_kv,
                      0, n_tokens, n_comp, window, ratio, n_head, head_dim) != 0,
                  "attn_prefill: static_mixed");
            CHECK(ds4_gpu_synchronize() != 0, "attn_prefill: sync static");
            ds4_gpu_tensor_read(heads, 0, hv, (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            check_close(hv, ref, n_tokens * n_head * head_dim, 1e-3f,
                        "attn_prefill: static mixed heads match CPU");

            /* masked mixed */
            cpu_attn_prefill(ref, (const float *)(g_model + OFF_SINKS), qv, rv,
                             cv, n_comp, mv, 1, n_tokens, window, ratio,
                             n_head, head_dim);
            CHECK(ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
                      heads, g_model, MODEL_PADDED, OFF_SINKS, q, raw_kv, comp_kv,
                      0, comp_mask, n_tokens, n_comp, window, ratio, n_head,
                      head_dim) != 0,
                  "attn_prefill: masked_mixed");
            CHECK(ds4_gpu_synchronize() != 0, "attn_prefill: sync masked");
            ds4_gpu_tensor_read(heads, 0, hv, (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
            check_close(hv, ref, n_tokens * n_head * head_dim, 1e-3f,
                        "attn_prefill: masked mixed heads match CPU");
            free(qv); free(rv); free(cv); free(mv); free(hv); free(ref);
        }
        if (q) ds4_gpu_tensor_free(q);
        if (raw_kv) ds4_gpu_tensor_free(raw_kv);
        if (comp_kv) ds4_gpu_tensor_free(comp_kv);
        if (comp_mask) ds4_gpu_tensor_free(comp_mask);
        if (heads) ds4_gpu_tensor_free(heads);
    }

    /* --- Fase 3: attention output projections ------------------------------ */
    {
        const uint32_t n_groups = 4u, n_rows = 3u;
        const uint64_t group_dim = 64u, rank = 8u;
        const uint32_t low_dim = n_groups * (uint32_t)rank;
        ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc((uint64_t)n_rows * n_groups * group_dim * sizeof(float));
        ds4_gpu_tensor *low = ds4_gpu_tensor_alloc((uint64_t)n_rows * low_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_rows * 16u * sizeof(float));
        CHECK(heads && low && out, "attn_out: alloc");
        if (heads && low && out) {
            float *hvv = (float *)malloc((uint64_t)n_rows * n_groups * group_dim * sizeof(float));
            float *lvv = (float *)malloc((uint64_t)n_rows * low_dim * sizeof(float));
            float *ov = (float *)malloc((uint64_t)n_rows * 16u * sizeof(float));
            float *ref_low = (float *)malloc((uint64_t)n_rows * low_dim * sizeof(float));
            float *ref_out = (float *)malloc((uint64_t)n_rows * 16u * sizeof(float));
            for (uint32_t i = 0; i < n_rows * n_groups * group_dim; i++) {
                hvv[i] = 0.1f * (float)(int32_t)((i * 7u) % 200) - 1.0f;
            }
            ds4_gpu_tensor_write(heads, 0, hvv, (uint64_t)n_rows * n_groups * group_dim * sizeof(float));
            cpu_attn_output_low(ref_low, g_model + OFF_OUT_A, n_groups, 0, n_groups,
                                group_dim, rank, hvv, n_rows);
            CHECK(ds4_gpu_attention_output_low_q8_rows_exact_tensor(
                      low, g_model, MODEL_PADDED, OFF_OUT_A, group_dim, rank,
                      n_groups, 0u, n_groups, heads, n_rows) != 0,
                  "attn_out: low_q8_rows_exact");
            CHECK(ds4_gpu_synchronize() != 0, "attn_out: sync low");
            ds4_gpu_tensor_read(low, 0, lvv, (uint64_t)n_rows * low_dim * sizeof(float));
            check_close(lvv, ref_low, n_rows * low_dim, 0.01f,
                        "attn_out: low q8 matches CPU");

            /* full batch output (low + out_b matmul) */
            cpu_matmul_q8(ref_out, g_model + OFF_OUT_B, 16u, low_dim, ref_low, n_rows);
            CHECK(ds4_gpu_attention_output_q8_batch_tensor(
                      out, low, NULL, NULL, g_model, MODEL_PADDED, OFF_OUT_A,
                      OFF_OUT_B, group_dim, rank, n_groups, 16u, heads,
                      n_rows) != 0,
                  "attn_out: q8_batch");
            CHECK(ds4_gpu_synchronize() != 0, "attn_out: sync batch");
            ds4_gpu_tensor_read(out, 0, ov, (uint64_t)n_rows * 16u * sizeof(float));
            check_close(ov, ref_out, n_rows * 16u, 0.01f,
                        "attn_out: q8 batch matches CPU");

            /* single-token low (the decode path) */
            float *ref_low1 = (float *)malloc((uint64_t)low_dim * sizeof(float));
            cpu_attn_output_low(ref_low1, g_model + OFF_OUT_A, n_groups, 0, n_groups,
                                group_dim, rank, hvv, 1u);
            CHECK(ds4_gpu_attention_output_low_q8_tensor(
                      low, g_model, MODEL_PADDED, OFF_OUT_A, group_dim, rank,
                      n_groups, heads) != 0,
                  "attn_out: low_q8 single");
            CHECK(ds4_gpu_synchronize() != 0, "attn_out: sync low single");
            ds4_gpu_tensor_read(low, 0, lvv, (uint64_t)low_dim * sizeof(float));
            check_close(lvv, ref_low1, low_dim, 0.01f,
                        "attn_out: low q8 single matches CPU");
            free(ref_low1);
            free(hvv); free(lvv); free(ov); free(ref_low); free(ref_out);
        }
        if (heads) ds4_gpu_tensor_free(heads);
        if (low) ds4_gpu_tensor_free(low);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* --- Fase 4: shared expert gate/up/swiglu ------------------------------ */
    {
        const uint32_t in_dim = 64u, out_dim = 32u, n_tok = 3u;
        const float clamp = 5.0f;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
        ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
        ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
        ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
        CHECK(x && gate && up && mid, "moe_shared: alloc");
        if (x && gate && up && mid) {
            float *xv = (float *)malloc((uint64_t)n_tok * in_dim * sizeof(float));
            float *gv = (float *)malloc((uint64_t)n_tok * out_dim * sizeof(float));
            float *uv = (float *)malloc((uint64_t)n_tok * out_dim * sizeof(float));
            float *mv = (float *)malloc((uint64_t)n_tok * out_dim * sizeof(float));
            float *g_ref = (float *)malloc((uint64_t)n_tok * out_dim * sizeof(float));
            float *u_ref = (float *)malloc((uint64_t)n_tok * out_dim * sizeof(float));
            float *m_ref = (float *)malloc((uint64_t)n_tok * out_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tok * in_dim; i++) {
                xv[i] = 0.3f * sinf((float)i * 0.11f) + 0.2f * (float)(i % 5);
            }
            ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tok * in_dim * sizeof(float));

            /* single token */
            cpu_shared_gate_up_swiglu(g_ref, u_ref, m_ref, g_model, OFF_MOE_GATE,
                                      OFF_MOE_UP, in_dim, out_dim, xv, 1u, clamp);
            CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                      gate, up, mid, g_model, MODEL_PADDED, OFF_MOE_GATE,
                      OFF_MOE_UP, in_dim, out_dim, x, clamp) != 0,
                  "moe_shared: gate_up_swiglu single");
            CHECK(ds4_gpu_synchronize() != 0, "moe_shared: sync single");
            ds4_gpu_tensor_read(gate, 0, gv, out_dim * sizeof(float));
            ds4_gpu_tensor_read(up, 0, uv, out_dim * sizeof(float));
            ds4_gpu_tensor_read(mid, 0, mv, out_dim * sizeof(float));
            check_close(gv, g_ref, out_dim, 0.01f, "moe_shared: gate matches CPU");
            check_close(uv, u_ref, out_dim, 0.01f, "moe_shared: up matches CPU");
            check_close(mv, m_ref, out_dim, 0.01f, "moe_shared: mid matches CPU");

            /* rows */
            cpu_shared_gate_up_swiglu(g_ref, u_ref, m_ref, g_model, OFF_MOE_GATE,
                                      OFF_MOE_UP, in_dim, out_dim, xv, n_tok, clamp);
            CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
                      gate, up, mid, g_model, MODEL_PADDED, OFF_MOE_GATE,
                      OFF_MOE_UP, in_dim, out_dim, x, n_tok, clamp) != 0,
                  "moe_shared: gate_up_swiglu rows");
            CHECK(ds4_gpu_synchronize() != 0, "moe_shared: sync rows");
            ds4_gpu_tensor_read(gate, 0, gv, (uint64_t)n_tok * out_dim * sizeof(float));
            ds4_gpu_tensor_read(up, 0, uv, (uint64_t)n_tok * out_dim * sizeof(float));
            ds4_gpu_tensor_read(mid, 0, mv, (uint64_t)n_tok * out_dim * sizeof(float));
            check_close(gv, g_ref, n_tok * out_dim, 0.01f, "moe_shared: rows gate matches CPU");
            check_close(uv, u_ref, n_tok * out_dim, 0.01f, "moe_shared: rows up matches CPU");
            check_close(mv, m_ref, n_tok * out_dim, 0.01f, "moe_shared: rows mid matches CPU");

            /* shared_mid (scratch gate/up inside the backend) */
            CHECK(ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                      mid, g_model, MODEL_PADDED, OFF_MOE_GATE, OFF_MOE_UP,
                      in_dim, out_dim, x, clamp) != 0,
                  "moe_shared: mid_swiglu");
            CHECK(ds4_gpu_synchronize() != 0, "moe_shared: sync mid");
            ds4_gpu_tensor_read(mid, 0, mv, out_dim * sizeof(float));
            check_close(mv, m_ref, out_dim, 0.01f, "moe_shared: mid_swiglu matches CPU");
            free(xv); free(gv); free(uv); free(mv);
            free(g_ref); free(u_ref); free(m_ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (gate) ds4_gpu_tensor_free(gate);
        if (up) ds4_gpu_tensor_free(up);
        if (mid) ds4_gpu_tensor_free(mid);
    }

    /* --- Fase 4: router select --------------------------------------------- */
    {
        const uint32_t n_tokens = 4u;
        const uint32_t hash_rows = 16u;
        ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc((uint64_t)n_tokens * 256u * sizeof(float));
        ds4_gpu_tensor *tokens = ds4_gpu_tensor_alloc((uint64_t)n_tokens * sizeof(int32_t));
        ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * 6u * sizeof(int32_t));
        ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)n_tokens * 6u * sizeof(float));
        ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc((uint64_t)n_tokens * 256u * sizeof(float));
        CHECK(logits && tokens && selected && weights && probs, "router: alloc");
        if (logits && tokens && selected && weights && probs) {
            float *lv = (float *)malloc((uint64_t)n_tokens * 256u * sizeof(float));
            int32_t *tv = (int32_t *)malloc((uint64_t)n_tokens * sizeof(int32_t));
            int32_t *sv = (int32_t *)malloc((uint64_t)n_tokens * 6u * sizeof(int32_t));
            float *wv = (float *)malloc((uint64_t)n_tokens * 6u * sizeof(float));
            float *pv = (float *)malloc((uint64_t)n_tokens * 256u * sizeof(float));
            int32_t *s_ref = (int32_t *)malloc((uint64_t)n_tokens * 6u * sizeof(int32_t));
            float *w_ref = (float *)malloc((uint64_t)n_tokens * 6u * sizeof(float));
            float *p_ref = (float *)malloc((uint64_t)n_tokens * 256u * sizeof(float));
            for (uint32_t i = 0; i < n_tokens * 256u; i++) {
                lv[i] = 0.02f * (float)(int32_t)((i * 7u) % 250) - 1.5f;
            }
            for (uint32_t i = 0; i < n_tokens; i++) tv[i] = (int32_t)(i * 3u + 1u);
            ds4_gpu_tensor_write(logits, 0, lv, (uint64_t)n_tokens * 256u * sizeof(float));
            ds4_gpu_tensor_write(tokens, 0, tv, (uint64_t)n_tokens * sizeof(int32_t));
            const float *bias = (const float *)(g_model + OFF_ROUTER_BIAS);
            const int32_t *hash = (const int32_t *)(g_model + OFF_ROUTER_HASH);

            /* batch, no bias, topk */
            cpu_router_select(s_ref, w_ref, p_ref, lv, NULL, hash, hash_rows,
                              tv, 0, n_tokens, 0, 0, 1.5f);
            CHECK(ds4_gpu_router_select_batch_tensor(
                      selected, weights, probs, g_model, MODEL_PADDED, 0, 0,
                      hash_rows, 0, 0, false, false, logits, tokens,
                      256u, 6u, 1.5f, n_tokens) != 0,
                  "router: batch topk");
            CHECK(ds4_gpu_synchronize() != 0, "router: sync batch");
            ds4_gpu_tensor_read(selected, 0, sv, (uint64_t)n_tokens * 6u * sizeof(int32_t));
            ds4_gpu_tensor_read(weights, 0, wv, (uint64_t)n_tokens * 6u * sizeof(float));
            ds4_gpu_tensor_read(probs, 0, pv, (uint64_t)n_tokens * 256u * sizeof(float));
            CHECK(memcmp(sv, s_ref, (uint64_t)n_tokens * 6u * sizeof(int32_t)) == 0,
                  "router: selected indices match CPU");
            check_close(pv, p_ref, n_tokens * 256u, 1e-3f, "router: probs match CPU");
            check_close(wv, w_ref, n_tokens * 6u, 1e-3f, "router: weights match CPU");

            /* batch, with bias */
            cpu_router_select(s_ref, w_ref, p_ref, lv, bias, hash, hash_rows,
                              tv, 0, n_tokens, 1, 0, 1.5f);
            CHECK(ds4_gpu_router_select_batch_tensor(
                      selected, weights, probs, g_model, MODEL_PADDED,
                      OFF_ROUTER_BIAS, 0, hash_rows, 0, 0, true, false,
                      logits, tokens, 256u, 6u, 1.5f, n_tokens) != 0,
                  "router: batch bias");
            CHECK(ds4_gpu_synchronize() != 0, "router: sync batch bias");
            ds4_gpu_tensor_read(selected, 0, sv, (uint64_t)n_tokens * 6u * sizeof(int32_t));
            ds4_gpu_tensor_read(weights, 0, wv, (uint64_t)n_tokens * 6u * sizeof(float));
            CHECK(memcmp(sv, s_ref, (uint64_t)n_tokens * 6u * sizeof(int32_t)) == 0,
                  "router: bias selected indices match CPU");
            check_close(wv, w_ref, n_tokens * 6u, 1e-3f, "router: bias weights match CPU");

            /* batch, hash mode */
            cpu_router_select(s_ref, w_ref, p_ref, lv, NULL, hash, hash_rows,
                              tv, 0, n_tokens, 0, 1, 1.5f);
            CHECK(ds4_gpu_router_select_batch_tensor(
                      selected, weights, probs, g_model, MODEL_PADDED, 0,
                      OFF_ROUTER_HASH, hash_rows, 0, 0, false, true,
                      logits, tokens, 256u, 6u, 1.5f, n_tokens) != 0,
                  "router: batch hash");
            CHECK(ds4_gpu_synchronize() != 0, "router: sync batch hash");
            ds4_gpu_tensor_read(selected, 0, sv, (uint64_t)n_tokens * 6u * sizeof(int32_t));
            ds4_gpu_tensor_read(weights, 0, wv, (uint64_t)n_tokens * 6u * sizeof(float));
            CHECK(memcmp(sv, s_ref, (uint64_t)n_tokens * 6u * sizeof(int32_t)) == 0,
                  "router: hash selected indices match CPU");
            check_close(wv, w_ref, n_tokens * 6u, 1e-3f, "router: hash weights match CPU");

            /* single token (decode), no tokens tensor */
            cpu_router_select(s_ref, w_ref, p_ref, lv, bias, hash, hash_rows,
                              NULL, 2, 1, 1, 0, 1.5f);
            CHECK(ds4_gpu_router_select_tensor(
                      selected, weights, probs, g_model, MODEL_PADDED,
                      OFF_ROUTER_BIAS, 0, hash_rows, 2, 256u, 6u, 1.5f,
                      0, 0, true, false, logits) != 0,
                  "router: single bias");
            CHECK(ds4_gpu_synchronize() != 0, "router: sync single");
            ds4_gpu_tensor_read(selected, 0, sv, 6u * sizeof(int32_t));
            ds4_gpu_tensor_read(weights, 0, wv, 6u * sizeof(float));
            CHECK(memcmp(sv, s_ref, 6u * sizeof(int32_t)) == 0,
                  "router: single selected indices match CPU");
            check_close(wv, w_ref, 6u, 1e-3f, "router: single weights match CPU");
            free(lv); free(tv); free(sv); free(wv); free(pv);
            free(s_ref); free(w_ref); free(p_ref);
        }
        if (logits) ds4_gpu_tensor_free(logits);
        if (tokens) ds4_gpu_tensor_free(tokens);
        if (selected) ds4_gpu_tensor_free(selected);
        if (weights) ds4_gpu_tensor_free(weights);
        if (probs) ds4_gpu_tensor_free(probs);
    }

    /* --- Fase 4: routed MoE (Q8_0 experts) --------------------------------- */
    {
        const uint32_t n_tokens = 2u, n_expert = 3u, n_total = 8u;
        const uint32_t in_dim = 64u, mid_dim = 32u, out_dim = 32u;
        const uint32_t in_b = in_dim / 32u;
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * in_b * 34u;
        const uint64_t gate_row_bytes = (uint64_t)in_b * 34u;
        const uint64_t down_expert_bytes = (uint64_t)out_dim * 34u;
        const uint64_t down_row_bytes = 34u;
        const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
        const float clamp = 5.0f;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * in_dim * sizeof(float));
        ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(pair_count * sizeof(int32_t));
        ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
        ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(pair_count * out_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
        CHECK(x && selected && weights && gate && up && mid && down && out,
              "routed_moe: alloc");
        if (x && selected && weights && gate && up && mid && down && out) {
            float *xv = (float *)malloc((uint64_t)n_tokens * in_dim * sizeof(float));
            int32_t *sv = (int32_t *)malloc(pair_count * sizeof(int32_t));
            float *wv = (float *)malloc(pair_count * sizeof(float));
            float *gv = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *uv = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *mv = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *dv = (float *)malloc(pair_count * out_dim * sizeof(float));
            float *ov = (float *)malloc((uint64_t)n_tokens * out_dim * sizeof(float));
            float *g_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *u_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *m_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *o_ref = (float *)malloc((uint64_t)n_tokens * out_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tokens * in_dim; i++) {
                xv[i] = 0.2f * cosf((float)i * 0.17f) + 0.3f * (float)(i % 7) - 0.6f;
            }
            for (uint32_t i = 0; i < pair_count; i++) {
                sv[i] = (int32_t)((i * 5u + 1u) % n_total);
                wv[i] = 0.3f + 0.1f * (float)(i % 5);
            }
            ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tokens * in_dim * sizeof(float));
            ds4_gpu_tensor_write(selected, 0, sv, pair_count * sizeof(int32_t));
            ds4_gpu_tensor_write(weights, 0, wv, pair_count * sizeof(float));

            cpu_routed_moe(o_ref, g_ref, u_ref, m_ref, g_model, OFF_MOE_GATE,
                           OFF_MOE_UP, OFF_MOE_DOWN, gate_expert_bytes,
                           gate_row_bytes, down_expert_bytes, down_row_bytes,
                           in_dim, mid_dim, out_dim, sv, wv, xv,
                           n_tokens, n_expert, clamp);

            /* batch */
            CHECK(ds4_gpu_routed_moe_batch_tensor(
                      out, gate, up, mid, down, g_model, MODEL_PADDED,
                      OFF_MOE_GATE, OFF_MOE_UP, OFF_MOE_DOWN, 8u, 8u,
                      gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                      down_row_bytes, in_dim, mid_dim, out_dim, selected,
                      weights, n_total, n_expert, clamp, x, 0u, n_tokens,
                      NULL, true) != 0,
                  "routed_moe: batch q8");
            CHECK(ds4_gpu_synchronize() != 0, "routed_moe: sync batch");
            ds4_gpu_tensor_read(gate, 0, gv, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(up, 0, uv, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(mid, 0, mv, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(down, 0, dv, pair_count * out_dim * sizeof(float));
            ds4_gpu_tensor_read(out, 0, ov, (uint64_t)n_tokens * out_dim * sizeof(float));
            check_close(gv, g_ref, pair_count * mid_dim, 0.01f, "routed_moe: gate matches CPU");
            check_close(uv, u_ref, pair_count * mid_dim, 0.01f, "routed_moe: up matches CPU");
            check_close(mv, m_ref, pair_count * mid_dim, 0.01f, "routed_moe: mid matches CPU");
            check_close(ov, o_ref, n_tokens * out_dim, 0.01f, "routed_moe: out matches CPU");

            /* single token (decode path) */
            float *o_ref1 = (float *)malloc((uint64_t)out_dim * sizeof(float));
            cpu_routed_moe(o_ref1, g_ref, u_ref, m_ref, g_model, OFF_MOE_GATE,
                           OFF_MOE_UP, OFF_MOE_DOWN, gate_expert_bytes,
                           gate_row_bytes, down_expert_bytes, down_row_bytes,
                           in_dim, mid_dim, out_dim, sv, wv, xv, 1u, n_expert,
                           clamp);
            CHECK(ds4_gpu_routed_moe_one_tensor(
                      out, gate, up, mid, down, g_model, MODEL_PADDED,
                      OFF_MOE_GATE, OFF_MOE_UP, OFF_MOE_DOWN, 8u, 8u,
                      gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                      down_row_bytes, in_dim, mid_dim, out_dim, selected,
                      weights, n_total, n_expert, clamp, x, NULL, 0u, true) != 0,
                  "routed_moe: one q8");
            CHECK(ds4_gpu_synchronize() != 0, "routed_moe: sync one");
            ds4_gpu_tensor_read(out, 0, ov, (uint64_t)out_dim * sizeof(float));
            check_close(ov, o_ref1, out_dim, 0.01f, "routed_moe: one out matches CPU");
            free(o_ref1);
            free(xv); free(sv); free(wv); free(gv); free(uv); free(mv);
            free(dv); free(ov); free(g_ref); free(u_ref); free(m_ref); free(o_ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (selected) ds4_gpu_tensor_free(selected);
        if (weights) ds4_gpu_tensor_free(weights);
        if (gate) ds4_gpu_tensor_free(gate);
        if (up) ds4_gpu_tensor_free(up);
        if (mid) ds4_gpu_tensor_free(mid);
        if (down) ds4_gpu_tensor_free(down);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* --- Fase 6: routed MoE (IQ2_XXS gate/up + Q2_K down experts) -------- */
    {
        const uint32_t n_tokens = 2u, n_expert = 3u, n_total = 4u;
        const uint32_t in_dim = 256u, mid_dim = 256u, out_dim = 64u;
        const uint64_t gate_row_bytes = 66u;              /* 1 IQ2_XXS block */
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
        const uint64_t down_row_bytes = 84u;              /* 1 Q2_K block */
        const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
        const uint64_t gate_region = (uint64_t)n_total * gate_expert_bytes;
        const uint64_t down_region = (uint64_t)n_total * down_expert_bytes;
        const uint64_t up_off = gate_region;
        const uint64_t down_off = 2u * gate_region;
        const uint64_t iq2_model_size =
            (down_off + down_region + 4095u) & ~(uint64_t)4095u;
        const float clamp = 5.0f;
        const uint64_t pair_count = (uint64_t)n_tokens * n_expert;

        uint8_t *iq2_model = NULL;
        if (posix_memalign((void **)&iq2_model, 4096, iq2_model_size) == 0) {
            memset(iq2_model, 0, iq2_model_size);
        }
        CHECK(iq2_model != NULL, "moe_iq2: model alloc");
        if (iq2_model) {
            /* Random IQ2_XXS gate/up blocks and Q2_K down blocks.  For IQ2 any
             * byte pattern is valid (grid index / sign bits / scale nibble);
             * d ~ uniform(0.05, 0.2).  Q2_K: random scales/qs, d/dmin ~ 0.02..0.1. */
            srand(0x5eedu);
            for (uint64_t r = 0; r < (uint64_t)n_total * mid_dim; r++) {
                for (uint32_t which = 0; which < 2u; which++) {
                    uint8_t *blk = iq2_model +
                        (which ? gate_region : 0u) + r * gate_row_bytes;
                    uint16_t dh = f32_to_f16(
                            0.05f + 0.15f * (float)(rand() % 1000) / 1000.0f);
                    memcpy(blk, &dh, 2);
                    for (uint32_t i = 0; i < 64u; i++) {
                        blk[2u + i] = (uint8_t)(rand() & 0xffu);
                    }
                }
            }
            for (uint64_t r = 0; r < (uint64_t)n_total * out_dim; r++) {
                uint8_t *blk = iq2_model + down_off + r * down_row_bytes;
                for (uint32_t i = 0; i < 16u; i++) {
                    blk[i] = (uint8_t)(rand() & 0xffu);
                }
                for (uint32_t i = 0; i < 64u; i++) {
                    blk[16u + i] = (uint8_t)(rand() & 0xffu);
                }
                uint16_t dh = f32_to_f16(
                        0.02f + 0.08f * (float)(rand() % 1000) / 1000.0f);
                uint16_t dm = f32_to_f16(
                        0.02f + 0.08f * (float)(rand() % 1000) / 1000.0f);
                memcpy(blk + 80, &dh, 2);
                memcpy(blk + 82, &dm, 2);
            }
        }

        CHECK(ds4_gpu_set_model_map(iq2_model, iq2_model_size) != 0,
              "moe_iq2: set_model_map");
        CHECK(ds4_gpu_synchronize() != 0, "moe_iq2: sync after map");

        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * in_dim * sizeof(float));
        ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(pair_count * sizeof(int32_t));
        ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
        ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(pair_count * out_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
        CHECK(x && selected && weights && gate && up && mid && down && out,
              "moe_iq2: alloc");
        if (iq2_model && x && selected && weights && gate && up && mid && down && out) {
            float *xv = (float *)malloc((uint64_t)n_tokens * in_dim * sizeof(float));
            int32_t *sv = (int32_t *)malloc(pair_count * sizeof(int32_t));
            float *wv = (float *)malloc(pair_count * sizeof(float));
            float *gv = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *uv = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *mv = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *ov = (float *)malloc((uint64_t)n_tokens * out_dim * sizeof(float));
            float *g_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *u_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *m_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *o_ref = (float *)malloc((uint64_t)n_tokens * out_dim * sizeof(float));
            CHECK(xv && sv && wv && gv && uv && mv && ov &&
                  g_ref && u_ref && m_ref && o_ref, "moe_iq2: ref alloc");
            for (uint32_t i = 0; i < n_tokens * in_dim; i++) {
                xv[i] = 0.2f * cosf((float)i * 0.31f) +
                        0.3f * (float)(i % 7) - 0.5f;
            }
            for (uint32_t i = 0; i < pair_count; i++) {
                sv[i] = (int32_t)((i * 5u + 1u) % n_total);
                wv[i] = 0.3f + 0.1f * (float)(i % 5);
            }
            ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tokens * in_dim * sizeof(float));
            ds4_gpu_tensor_write(selected, 0, sv, pair_count * sizeof(int32_t));
            ds4_gpu_tensor_write(weights, 0, wv, pair_count * sizeof(float));

            cpu_routed_moe_iq2q2(o_ref, g_ref, u_ref, m_ref, iq2_model,
                                 0u, up_off, down_off, gate_expert_bytes,
                                 gate_row_bytes, down_expert_bytes,
                                 down_row_bytes, in_dim, mid_dim, out_dim,
                                 sv, wv, xv, n_tokens, n_expert, clamp);

            CHECK(ds4_gpu_routed_moe_batch_tensor(
                      out, gate, up, mid, down, iq2_model, iq2_model_size,
                      0u, up_off, down_off, 16u, 10u,
                      gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                      down_row_bytes, in_dim, mid_dim, out_dim, selected,
                      weights, n_total, n_expert, clamp, x, 0u, n_tokens,
                      NULL, true) != 0,
                  "routed_moe: batch iq2/q2k");
            CHECK(ds4_gpu_synchronize() != 0, "routed_moe: sync iq2 batch");
            ds4_gpu_tensor_read(gate, 0, gv, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(up, 0, uv, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(mid, 0, mv, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(out, 0, ov, (uint64_t)n_tokens * out_dim * sizeof(float));
            check_close(gv, g_ref, pair_count * mid_dim, 0.01f,
                        "routed_moe: iq2 gate matches CPU");
            check_close(uv, u_ref, pair_count * mid_dim, 0.01f,
                        "routed_moe: iq2 up matches CPU");
            check_close(mv, m_ref, pair_count * mid_dim, 0.01f,
                        "routed_moe: iq2 mid matches CPU");
            check_close(ov, o_ref, n_tokens * out_dim, 0.01f,
                        "routed_moe: iq2 out matches CPU");

            float *o_ref1 = (float *)malloc((uint64_t)out_dim * sizeof(float));
            cpu_routed_moe_iq2q2(o_ref1, g_ref, u_ref, m_ref, iq2_model,
                                 0u, up_off, down_off, gate_expert_bytes,
                                 gate_row_bytes, down_expert_bytes,
                                 down_row_bytes, in_dim, mid_dim, out_dim,
                                 sv, wv, xv, 1u, n_expert, clamp);
            CHECK(ds4_gpu_routed_moe_one_tensor(
                      out, gate, up, mid, down, iq2_model, iq2_model_size,
                      0u, up_off, down_off, 16u, 10u,
                      gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                      down_row_bytes, in_dim, mid_dim, out_dim, selected,
                      weights, n_total, n_expert, clamp, x, NULL, 0u, true) != 0,
                  "routed_moe: one iq2/q2k");
            CHECK(ds4_gpu_synchronize() != 0, "routed_moe: sync iq2 one");
            ds4_gpu_tensor_read(out, 0, ov, (uint64_t)out_dim * sizeof(float));
            check_close(ov, o_ref1, out_dim, 0.01f,
                        "routed_moe: iq2 one out matches CPU");

            /* Non-IQ2/Q2K combos must still fail cleanly. */
            CHECK(ds4_gpu_routed_moe_batch_tensor(
                      out, gate, up, mid, down, iq2_model, iq2_model_size,
                      0u, up_off, down_off, 16u, 16u,
                      gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                      down_row_bytes, in_dim, mid_dim, out_dim, selected,
                      weights, n_total, n_expert, clamp, x, 0u, n_tokens,
                      NULL, true) == 0,
                  "routed_moe: IQ2_XXS+IQ2_XXS rejected");

            free(o_ref1);
            free(xv); free(sv); free(wv); free(gv); free(uv); free(mv);
            free(ov); free(g_ref); free(u_ref); free(m_ref); free(o_ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (selected) ds4_gpu_tensor_free(selected);
        if (weights) ds4_gpu_tensor_free(weights);
        if (gate) ds4_gpu_tensor_free(gate);
        if (up) ds4_gpu_tensor_free(up);
        if (mid) ds4_gpu_tensor_free(mid);
        if (down) ds4_gpu_tensor_free(down);
        if (out) ds4_gpu_tensor_free(out);

        /* Restore the main model wrapper for the remaining tests. */
        CHECK(ds4_gpu_set_model_map(g_model, MODEL_PADDED) != 0,
              "moe_iq2: restore full model map");
        if (iq2_model) free(iq2_model);
    }

    /* --- Fase 6b: routed MoE (Q4_K gate/up/down experts) ------------------ */
    {
        const uint32_t n_tokens = 2u, n_expert = 3u, n_total = 4u;
        const uint32_t in_dim = 256u, mid_dim = 256u, out_dim = 64u;
        const uint64_t gate_row_bytes = 144u;             /* 1 Q4_K block */
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
        const uint64_t down_row_bytes = 144u;             /* 1 Q4_K block */
        const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
        const uint64_t gate_region = (uint64_t)n_total * gate_expert_bytes;
        const uint64_t down_region = (uint64_t)n_total * down_expert_bytes;
        const uint64_t up_off = gate_region;
        const uint64_t down_off = 2u * gate_region;
        const uint64_t q4k_model_size =
            (down_off + down_region + 4095u) & ~(uint64_t)4095u;
        const float clamp = 5.0f;
        const uint64_t pair_count = (uint64_t)n_tokens * n_expert;

        uint8_t *q4k_model = NULL;
        if (posix_memalign((void **)&q4k_model, 4096, q4k_model_size) == 0) {
            memset(q4k_model, 0, q4k_model_size);
        }
        CHECK(q4k_model != NULL, "moe_q4k: model alloc");
        if (q4k_model) {
            /* Random Q4_K blocks: any byte pattern is a valid block; d/dmin
             * ~ 0.02..0.1, random scale/qs bytes. */
            srand(0x4a4bu);
            for (uint64_t r = 0; r < (uint64_t)n_total * mid_dim; r++) {
                for (uint32_t which = 0; which < 2u; which++) {
                    uint8_t *blk = q4k_model +
                        (which ? gate_region : 0u) + r * gate_row_bytes;
                    uint16_t dh = f32_to_f16(
                            0.02f + 0.08f * (float)(rand() % 1000) / 1000.0f);
                    uint16_t dm = f32_to_f16(
                            0.02f + 0.08f * (float)(rand() % 1000) / 1000.0f);
                    memcpy(blk, &dh, 2);
                    memcpy(blk + 2, &dm, 2);
                    for (uint32_t i = 4u; i < 144u; i++) {
                        blk[i] = (uint8_t)(rand() & 0xffu);
                    }
                }
            }
            for (uint64_t r = 0; r < (uint64_t)n_total * out_dim; r++) {
                uint8_t *blk = q4k_model + down_off + r * down_row_bytes;
                uint16_t dh = f32_to_f16(
                        0.02f + 0.08f * (float)(rand() % 1000) / 1000.0f);
                uint16_t dm = f32_to_f16(
                        0.02f + 0.08f * (float)(rand() % 1000) / 1000.0f);
                memcpy(blk, &dh, 2);
                memcpy(blk + 2, &dm, 2);
                for (uint32_t i = 4u; i < 144u; i++) {
                    blk[i] = (uint8_t)(rand() & 0xffu);
                }
            }
        }

        CHECK(ds4_gpu_set_model_map(q4k_model, q4k_model_size) != 0,
              "moe_q4k: set_model_map");
        CHECK(ds4_gpu_synchronize() != 0, "moe_q4k: sync after map");

        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * in_dim * sizeof(float));
        ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(pair_count * sizeof(int32_t));
        ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
        ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(pair_count * out_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
        CHECK(x && selected && weights && gate && up && mid && down && out,
              "moe_q4k: alloc");
        if (q4k_model && x && selected && weights && gate && up && mid && down && out) {
            float *xv = (float *)malloc((uint64_t)n_tokens * in_dim * sizeof(float));
            int32_t *sv = (int32_t *)malloc(pair_count * sizeof(int32_t));
            float *wv = (float *)malloc(pair_count * sizeof(float));
            float *gv = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *uv = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *mv = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *ov = (float *)malloc((uint64_t)n_tokens * out_dim * sizeof(float));
            float *g_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *u_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *m_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *o_ref = (float *)malloc((uint64_t)n_tokens * out_dim * sizeof(float));
            CHECK(xv && sv && wv && gv && uv && mv && ov &&
                  g_ref && u_ref && m_ref && o_ref, "moe_q4k: ref alloc");
            for (uint32_t i = 0; i < n_tokens * in_dim; i++) {
                xv[i] = 0.2f * cosf((float)i * 0.31f) +
                        0.3f * (float)(i % 7) - 0.5f;
            }
            for (uint32_t i = 0; i < pair_count; i++) {
                sv[i] = (int32_t)((i * 5u + 1u) % n_total);
                wv[i] = 0.3f + 0.1f * (float)(i % 5);
            }
            ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tokens * in_dim * sizeof(float));
            ds4_gpu_tensor_write(selected, 0, sv, pair_count * sizeof(int32_t));
            ds4_gpu_tensor_write(weights, 0, wv, pair_count * sizeof(float));

            cpu_routed_moe_q4k(o_ref, g_ref, u_ref, m_ref, q4k_model,
                               0u, up_off, down_off, gate_expert_bytes,
                               gate_row_bytes, down_expert_bytes,
                               down_row_bytes, in_dim, mid_dim, out_dim,
                               sv, wv, xv, n_tokens, n_expert, clamp);

            CHECK(ds4_gpu_routed_moe_batch_tensor(
                      out, gate, up, mid, down, q4k_model, q4k_model_size,
                      0u, up_off, down_off, 12u, 12u,
                      gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                      down_row_bytes, in_dim, mid_dim, out_dim, selected,
                      weights, n_total, n_expert, clamp, x, 0u, n_tokens,
                      NULL, true) != 0,
                  "routed_moe: batch q4k");
            CHECK(ds4_gpu_synchronize() != 0, "routed_moe: sync q4k batch");
            ds4_gpu_tensor_read(gate, 0, gv, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(up, 0, uv, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(mid, 0, mv, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(out, 0, ov, (uint64_t)n_tokens * out_dim * sizeof(float));
            check_close(gv, g_ref, pair_count * mid_dim, 0.01f,
                        "routed_moe: q4k gate matches CPU");
            check_close(uv, u_ref, pair_count * mid_dim, 0.01f,
                        "routed_moe: q4k up matches CPU");
            check_close(mv, m_ref, pair_count * mid_dim, 0.01f,
                        "routed_moe: q4k mid matches CPU");
            check_close(ov, o_ref, n_tokens * out_dim, 0.01f,
                        "routed_moe: q4k out matches CPU");

            float *o_ref1 = (float *)malloc((uint64_t)out_dim * sizeof(float));
            cpu_routed_moe_q4k(o_ref1, g_ref, u_ref, m_ref, q4k_model,
                               0u, up_off, down_off, gate_expert_bytes,
                               gate_row_bytes, down_expert_bytes,
                               down_row_bytes, in_dim, mid_dim, out_dim,
                               sv, wv, xv, 1u, n_expert, clamp);
            CHECK(ds4_gpu_routed_moe_one_tensor(
                      out, gate, up, mid, down, q4k_model, q4k_model_size,
                      0u, up_off, down_off, 12u, 12u,
                      gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                      down_row_bytes, in_dim, mid_dim, out_dim, selected,
                      weights, n_total, n_expert, clamp, x, NULL, 0u, true) != 0,
                  "routed_moe: one q4k");
            CHECK(ds4_gpu_synchronize() != 0, "routed_moe: sync q4k one");
            ds4_gpu_tensor_read(out, 0, ov, (uint64_t)out_dim * sizeof(float));
            check_close(ov, o_ref1, out_dim, 0.01f,
                        "routed_moe: q4k one out matches CPU");

            /* Non-Q4_K combos must still fail cleanly. */
            CHECK(ds4_gpu_routed_moe_batch_tensor(
                      out, gate, up, mid, down, q4k_model, q4k_model_size,
                      0u, up_off, down_off, 12u, 10u,
                      gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                      down_row_bytes, in_dim, mid_dim, out_dim, selected,
                      weights, n_total, n_expert, clamp, x, 0u, n_tokens,
                      NULL, true) == 0,
                  "routed_moe: Q4_K+Q2_K rejected");

            free(o_ref1);
            free(xv); free(sv); free(wv); free(gv); free(uv); free(mv);
            free(ov); free(g_ref); free(u_ref); free(m_ref); free(o_ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (selected) ds4_gpu_tensor_free(selected);
        if (weights) ds4_gpu_tensor_free(weights);
        if (gate) ds4_gpu_tensor_free(gate);
        if (up) ds4_gpu_tensor_free(up);
        if (mid) ds4_gpu_tensor_free(mid);
        if (down) ds4_gpu_tensor_free(down);
        if (out) ds4_gpu_tensor_free(out);

        CHECK(ds4_gpu_set_model_map(g_model, MODEL_PADDED) != 0,
              "moe_q4k: restore full model map");
        if (q4k_model) free(q4k_model);
    }

    /* --- Fase 6c: routed MoE (MXFP4 gate/up/down experts) ----------------- */
    {
        const uint32_t n_tokens = 2u, n_expert = 3u, n_total = 4u;
        const uint32_t in_dim = 256u, mid_dim = 256u, out_dim = 64u;
        const uint64_t gate_row_bytes = (in_dim / 32u) * 17u;  /* 8 blocks */
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
        const uint64_t down_row_bytes = (in_dim / 32u) * 17u;
        const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
        const uint64_t gate_region = (uint64_t)n_total * gate_expert_bytes;
        const uint64_t down_region = (uint64_t)n_total * down_expert_bytes;
        const uint64_t up_off = gate_region;
        const uint64_t down_off = 2u * gate_region;
        const uint64_t mxfp4_model_size =
            (down_off + down_region + 4095u) & ~(uint64_t)4095u;
        const float clamp = 5.0f;
        const uint64_t pair_count = (uint64_t)n_tokens * n_expert;

        uint8_t *mxfp4_model = NULL;
        if (posix_memalign((void **)&mxfp4_model, 4096, mxfp4_model_size) == 0) {
            memset(mxfp4_model, 0, mxfp4_model_size);
        }
        CHECK(mxfp4_model != NULL, "moe_mxfp4: model alloc");
        if (mxfp4_model) {
            /* Random MXFP4 blocks: e ~ 124..130 (scale 2^-3..2^3), random
             * nibbles (any E2M1 code is valid). */
            srand(0x3f9au);
            for (uint64_t r = 0; r < (uint64_t)n_total * mid_dim; r++) {
                for (uint32_t which = 0; which < 2u; which++) {
                    uint8_t *blk = mxfp4_model +
                        (which ? gate_region : 0u) + r * gate_row_bytes;
                    for (uint64_t off = 0; off < gate_row_bytes; off += 17u) {
                        blk[off] = (uint8_t)(124u + (uint32_t)(rand() % 7));
                        for (uint32_t i = 1u; i < 17u; i++) {
                            blk[off + i] = (uint8_t)(rand() & 0xffu);
                        }
                    }
                }
            }
            for (uint64_t r = 0; r < (uint64_t)n_total * out_dim; r++) {
                uint8_t *blk = mxfp4_model + down_off + r * down_row_bytes;
                for (uint64_t off = 0; off < down_row_bytes; off += 17u) {
                    blk[off] = (uint8_t)(124u + (uint32_t)(rand() % 7));
                    for (uint32_t i = 1u; i < 17u; i++) {
                        blk[off + i] = (uint8_t)(rand() & 0xffu);
                    }
                }
            }
        }

        CHECK(ds4_gpu_set_model_map(mxfp4_model, mxfp4_model_size) != 0,
              "moe_mxfp4: set_model_map");
        CHECK(ds4_gpu_synchronize() != 0, "moe_mxfp4: sync after map");

        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * in_dim * sizeof(float));
        ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(pair_count * sizeof(int32_t));
        ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
        ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(pair_count * out_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
        CHECK(x && selected && weights && gate && up && mid && down && out,
              "moe_mxfp4: alloc");
        if (mxfp4_model && x && selected && weights && gate && up && mid && down && out) {
            float *xv = (float *)malloc((uint64_t)n_tokens * in_dim * sizeof(float));
            int32_t *sv = (int32_t *)malloc(pair_count * sizeof(int32_t));
            float *wv = (float *)malloc(pair_count * sizeof(float));
            float *gv = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *uv = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *mv = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *ov = (float *)malloc((uint64_t)n_tokens * out_dim * sizeof(float));
            float *g_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *u_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *m_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *o_ref = (float *)malloc((uint64_t)n_tokens * out_dim * sizeof(float));
            CHECK(xv && sv && wv && gv && uv && mv && ov &&
                  g_ref && u_ref && m_ref && o_ref, "moe_mxfp4: ref alloc");
            for (uint32_t i = 0; i < n_tokens * in_dim; i++) {
                xv[i] = 0.2f * cosf((float)i * 0.31f) +
                        0.3f * (float)(i % 7) - 0.5f;
            }
            for (uint32_t i = 0; i < pair_count; i++) {
                sv[i] = (int32_t)((i * 5u + 1u) % n_total);
                wv[i] = 0.3f + 0.1f * (float)(i % 5);
            }
            ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tokens * in_dim * sizeof(float));
            ds4_gpu_tensor_write(selected, 0, sv, pair_count * sizeof(int32_t));
            ds4_gpu_tensor_write(weights, 0, wv, pair_count * sizeof(float));

            cpu_routed_moe_mxfp4(o_ref, g_ref, u_ref, m_ref, mxfp4_model,
                                 0u, up_off, down_off, gate_expert_bytes,
                                 gate_row_bytes, down_expert_bytes,
                                 down_row_bytes, in_dim, mid_dim, out_dim,
                                 sv, wv, xv, n_tokens, n_expert, clamp);

            CHECK(ds4_gpu_routed_moe_batch_tensor(
                      out, gate, up, mid, down, mxfp4_model, mxfp4_model_size,
                      0u, up_off, down_off, 39u, 39u,
                      gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                      down_row_bytes, in_dim, mid_dim, out_dim, selected,
                      weights, n_total, n_expert, clamp, x, 0u, n_tokens,
                      NULL, true) != 0,
                  "routed_moe: batch mxfp4");
            CHECK(ds4_gpu_synchronize() != 0, "routed_moe: sync mxfp4 batch");
            ds4_gpu_tensor_read(gate, 0, gv, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(up, 0, uv, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(mid, 0, mv, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(out, 0, ov, (uint64_t)n_tokens * out_dim * sizeof(float));
            check_close(gv, g_ref, pair_count * mid_dim, 0.01f,
                        "routed_moe: mxfp4 gate matches CPU");
            check_close(uv, u_ref, pair_count * mid_dim, 0.01f,
                        "routed_moe: mxfp4 up matches CPU");
            check_close(mv, m_ref, pair_count * mid_dim, 0.01f,
                        "routed_moe: mxfp4 mid matches CPU");
            check_close(ov, o_ref, n_tokens * out_dim, 0.01f,
                        "routed_moe: mxfp4 out matches CPU");

            float *o_ref1 = (float *)malloc((uint64_t)out_dim * sizeof(float));
            cpu_routed_moe_mxfp4(o_ref1, g_ref, u_ref, m_ref, mxfp4_model,
                                 0u, up_off, down_off, gate_expert_bytes,
                                 gate_row_bytes, down_expert_bytes,
                                 down_row_bytes, in_dim, mid_dim, out_dim,
                                 sv, wv, xv, 1u, n_expert, clamp);
            CHECK(ds4_gpu_routed_moe_one_tensor(
                      out, gate, up, mid, down, mxfp4_model, mxfp4_model_size,
                      0u, up_off, down_off, 39u, 39u,
                      gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                      down_row_bytes, in_dim, mid_dim, out_dim, selected,
                      weights, n_total, n_expert, clamp, x, NULL, 0u, true) != 0,
                  "routed_moe: one mxfp4");
            CHECK(ds4_gpu_synchronize() != 0, "routed_moe: sync mxfp4 one");
            ds4_gpu_tensor_read(out, 0, ov, (uint64_t)out_dim * sizeof(float));
            check_close(ov, o_ref1, out_dim, 0.01f,
                        "routed_moe: mxfp4 one out matches CPU");

            /* Mixed pairs must still fail cleanly. */
            CHECK(ds4_gpu_routed_moe_batch_tensor(
                      out, gate, up, mid, down, mxfp4_model, mxfp4_model_size,
                      0u, up_off, down_off, 39u, 12u,
                      gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                      down_row_bytes, in_dim, mid_dim, out_dim, selected,
                      weights, n_total, n_expert, clamp, x, 0u, n_tokens,
                      NULL, true) == 0,
                  "routed_moe: MXFP4+Q4_K rejected");

            free(o_ref1);
            free(xv); free(sv); free(wv); free(gv); free(uv); free(mv);
            free(ov); free(g_ref); free(u_ref); free(m_ref); free(o_ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (selected) ds4_gpu_tensor_free(selected);
        if (weights) ds4_gpu_tensor_free(weights);
        if (gate) ds4_gpu_tensor_free(gate);
        if (up) ds4_gpu_tensor_free(up);
        if (mid) ds4_gpu_tensor_free(mid);
        if (down) ds4_gpu_tensor_free(down);
        if (out) ds4_gpu_tensor_free(out);

        CHECK(ds4_gpu_set_model_map(g_model, MODEL_PADDED) != 0,
              "moe_mxfp4: restore full model map");
        if (mxfp4_model) free(mxfp4_model);
    }

    /* --- Fase 3b: attention output with out_a Q4_K ------------------------ */
    {
        const uint32_t n_groups = 2u, rank = 16u, out_dim = 16u, n_rows = 3u;
        const uint64_t group_dim = 256u;
        const uint32_t low_dim = n_groups * rank;
        const uint32_t out_a_rows = n_groups * rank;
        const uint64_t out_a_bytes = (uint64_t)out_a_rows * 144u;
        const uint64_t out_b_off = out_a_bytes;
        const uint64_t out_b_bytes = (uint64_t)out_dim * 34u;
        const uint64_t q4k_size =
            (out_b_off + out_b_bytes + 4095u) & ~(uint64_t)4095u;

        uint8_t *q4k_attn = NULL;
        if (posix_memalign((void **)&q4k_attn, 4096, q4k_size) == 0) {
            memset(q4k_attn, 0, q4k_size);
        }
        CHECK(q4k_attn != NULL, "attn_out_q4k: model alloc");
        if (q4k_attn) {
            srand(0xa771u);
            for (uint32_t r = 0; r < out_a_rows; r++) {
                uint8_t *blk = q4k_attn + (uint64_t)r * 144u;
                uint16_t dh = f32_to_f16(
                        0.02f + 0.08f * (float)(rand() % 1000) / 1000.0f);
                uint16_t dm = f32_to_f16(
                        0.02f + 0.08f * (float)(rand() % 1000) / 1000.0f);
                memcpy(blk, &dh, 2);
                memcpy(blk + 2, &dm, 2);
                for (uint32_t i = 4u; i < 144u; i++) {
                    blk[i] = (uint8_t)(rand() & 0xffu);
                }
            }
            for (uint32_t r = 0; r < out_dim; r++) {
                float row[32];
                for (uint32_t i = 0; i < 32u; i++) {
                    row[i] = 0.03f * (float)(int32_t)((r * 9u + i * 5u) % 53) - 0.8f;
                }
                quant_q8_block(q4k_attn + out_b_off + (uint64_t)r * 34u, row, 32);
            }
        }

        CHECK(ds4_gpu_set_model_map(q4k_attn, q4k_size) != 0,
              "attn_out_q4k: set_model_map");
        CHECK(ds4_gpu_synchronize() != 0, "attn_out_q4k: sync after map");

        ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc((uint64_t)n_rows * n_groups * group_dim * sizeof(float));
        ds4_gpu_tensor *low = ds4_gpu_tensor_alloc((uint64_t)n_rows * low_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_rows * out_dim * sizeof(float));
        CHECK(heads && low && out, "attn_out_q4k: alloc");
        if (q4k_attn && heads && low && out) {
            float *hvv = (float *)malloc((uint64_t)n_rows * n_groups * group_dim * sizeof(float));
            float *lvv = (float *)malloc((uint64_t)n_rows * low_dim * sizeof(float));
            float *ov = (float *)malloc((uint64_t)n_rows * out_dim * sizeof(float));
            float *ref_low = (float *)malloc((uint64_t)n_rows * low_dim * sizeof(float));
            float *ref_out = (float *)malloc((uint64_t)n_rows * out_dim * sizeof(float));
            for (uint32_t i = 0; i < n_rows * n_groups * group_dim; i++) {
                hvv[i] = 0.1f * (float)(int32_t)((i * 7u) % 200) - 1.0f;
            }
            ds4_gpu_tensor_write(heads, 0, hvv,
                                 (uint64_t)n_rows * n_groups * group_dim * sizeof(float));

            /* single-token group slice (decode path) */
            float *ref_low1 = (float *)malloc((uint64_t)low_dim * sizeof(float));
            cpu_attn_output_low_q4k(ref_low1, q4k_attn, n_groups, 0, n_groups,
                                    group_dim, rank, hvv, 1u);
            CHECK(ds4_gpu_attention_output_low_q4_K_slice_tensor(
                      low, q4k_attn, q4k_size, 0u, group_dim, rank, 0u, n_groups,
                      heads) != 0,
                  "attn_out_q4k: low slice");
            CHECK(ds4_gpu_synchronize() != 0, "attn_out_q4k: sync low slice");
            ds4_gpu_tensor_read(low, 0, lvv, (uint64_t)low_dim * sizeof(float));
            check_close(lvv, ref_low1, low_dim, 0.01f,
                        "attn_out_q4k: low slice matches CPU");
            free(ref_low1);

            /* full batch output (low Q4_K + out_b Q8_0 matmul) */
            cpu_attn_output_low_q4k(ref_low, q4k_attn, n_groups, 0, n_groups,
                                    group_dim, rank, hvv, n_rows);
            cpu_matmul_q8(ref_out, q4k_attn + out_b_off, out_dim, low_dim,
                          ref_low, n_rows);
            CHECK(ds4_gpu_attention_output_q4_K_batch_tensor(
                      out, low, NULL, NULL, q4k_attn, q4k_size, 0u, out_b_off,
                      8u, group_dim, rank, n_groups, out_dim, heads,
                      n_rows) != 0,
                  "attn_out_q4k: batch");
            CHECK(ds4_gpu_synchronize() != 0, "attn_out_q4k: sync batch");
            ds4_gpu_tensor_read(out, 0, ov, (uint64_t)n_rows * out_dim * sizeof(float));
            check_close(ov, ref_out, n_rows * out_dim, 0.01f,
                        "attn_out_q4k: batch matches CPU");

            free(hvv); free(lvv); free(ov); free(ref_low); free(ref_out);
        }
        if (heads) ds4_gpu_tensor_free(heads);
        if (low) ds4_gpu_tensor_free(low);
        if (out) ds4_gpu_tensor_free(out);

        CHECK(ds4_gpu_set_model_map(g_model, MODEL_PADDED) != 0,
              "attn_out_q4k: restore full model map");
        if (q4k_attn) free(q4k_attn);
    }

    /* --- Fase 3: dense quant matmul (Q4_K + Q4_0) ------------------------- */
    {
        const uint32_t out_dim = 32u, in_dim = 256u, n_tok = 2u;
        const uint64_t q4k_row_bytes = (in_dim / 256u) * 144u;
        const uint64_t q4_0_row_bytes = (in_dim / 32u) * 18u;
        const uint64_t q4k_off = 0u;
        const uint64_t q4k_bytes = (uint64_t)out_dim * q4k_row_bytes;
        const uint64_t q4_0_off = q4k_bytes;
        const uint64_t q4_0_bytes = (uint64_t)out_dim * q4_0_row_bytes;
        const uint64_t model_size =
            (q4_0_off + q4_0_bytes + 4095u) & ~(uint64_t)4095u;

        uint8_t *dm_model = NULL;
        if (posix_memalign((void **)&dm_model, 4096, model_size) == 0) {
            memset(dm_model, 0, model_size);
        }
        CHECK(dm_model != NULL, "matmul_quant: model alloc");
        if (dm_model) {
            srand(0x9d31u);
            for (uint32_t r = 0; r < out_dim; r++) {
                uint8_t *blk = dm_model + q4k_off + (uint64_t)r * q4k_row_bytes;
                uint16_t dh = f32_to_f16(
                        0.02f + 0.08f * (float)(rand() % 1000) / 1000.0f);
                uint16_t dm = f32_to_f16(
                        0.02f + 0.08f * (float)(rand() % 1000) / 1000.0f);
                memcpy(blk, &dh, 2);
                memcpy(blk + 2, &dm, 2);
                for (uint32_t i = 4u; i < 144u; i++) {
                    blk[i] = (uint8_t)(rand() & 0xffu);
                }
            }
            for (uint32_t r = 0; r < out_dim; r++) {
                uint8_t *rowp = dm_model + q4_0_off + (uint64_t)r * q4_0_row_bytes;
                for (uint32_t b = 0; b < in_dim / 32u; b++) {
                    uint8_t *blk = rowp + (uint64_t)b * 18u;
                    uint16_t dh = f32_to_f16(
                            0.02f + 0.08f * (float)(rand() % 1000) / 1000.0f);
                    memcpy(blk, &dh, 2);
                    for (uint32_t i = 0; i < 16u; i++) {
                        blk[2u + i] = (uint8_t)(rand() & 0xffu);
                    }
                }
            }
        }
        CHECK(ds4_gpu_set_model_map(dm_model, model_size) != 0,
              "matmul_quant: set_model_map");
        CHECK(ds4_gpu_synchronize() != 0, "matmul_quant: sync after map");

        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
        CHECK(x && out, "matmul_quant: alloc");
        if (dm_model && x && out) {
            float *xv = (float *)malloc((uint64_t)n_tok * in_dim * sizeof(float));
            float *ov = (float *)malloc((uint64_t)n_tok * out_dim * sizeof(float));
            float *ref = (float *)malloc((uint64_t)n_tok * out_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tok * in_dim; i++) {
                xv[i] = 0.2f * cosf((float)i * 0.31f) +
                        0.3f * (float)(i % 7) - 0.5f;
            }
            ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tok * in_dim * sizeof(float));

            /* Q4_K dense (type 12) */
            cpu_matmul_q4k(ref, dm_model + q4k_off, out_dim, in_dim, xv, n_tok);
            CHECK(ds4_gpu_matmul_quant_tensor(out, dm_model, model_size, q4k_off,
                                              12u, in_dim, out_dim, x, n_tok) != 0,
                  "matmul_quant: q4k");
            CHECK(ds4_gpu_synchronize() != 0, "matmul_quant: sync q4k");
            ds4_gpu_tensor_read(out, 0, ov, (uint64_t)n_tok * out_dim * sizeof(float));
            check_close(ov, ref, n_tok * out_dim, 0.01f,
                        "matmul_quant: q4k matches CPU");

            /* Q4_0 dense (type 2) */
            cpu_matmul_q4_0(ref, dm_model + q4_0_off, out_dim, in_dim, xv, n_tok);
            CHECK(ds4_gpu_matmul_quant_tensor(out, dm_model, model_size, q4_0_off,
                                              2u, in_dim, out_dim, x, n_tok) != 0,
                  "matmul_quant: q4_0");
            CHECK(ds4_gpu_synchronize() != 0, "matmul_quant: sync q4_0");
            ds4_gpu_tensor_read(out, 0, ov, (uint64_t)n_tok * out_dim * sizeof(float));
            check_close(ov, ref, n_tok * out_dim, 0.01f,
                        "matmul_quant: q4_0 matches CPU");

            free(xv); free(ov); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (out) ds4_gpu_tensor_free(out);
        CHECK(ds4_gpu_set_model_map(g_model, MODEL_PADDED) != 0,
              "matmul_quant: restore full model map");
        if (dm_model) free(dm_model);
    }

    /* --- Fase 5: head RMS norm (+ rope tail) --------------------------- */
    {
        const uint32_t hn_tok = 2u, hn_head = 4u, hn_dim = 32u, hn_nrot = 16u;
        const float eps = 1e-5f;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(
                (uint64_t)hn_tok * hn_head * hn_dim * sizeof(float));
        CHECK(x != NULL, "head_rms: alloc");
        if (x) {
            const uint32_t n = hn_tok * hn_head * hn_dim;
            float *xv = (float *)malloc(n * sizeof(float));
            float *ref = (float *)malloc(n * sizeof(float));
            for (uint32_t i = 0; i < n; i++) {
                xv[i] = 0.05f * (float)(int32_t)((i * 3u) % 79) - 1.7f;
            }
            ds4_gpu_tensor_write(x, 0, xv, n * sizeof(float));
            memcpy(ref, xv, n * sizeof(float));
            CHECK(ds4_gpu_head_rms_norm_tensor(x, hn_tok, hn_head, hn_dim, eps) != 0,
                  "head_rms: head_rms_norm");
            CHECK(ds4_gpu_synchronize() != 0, "head_rms: sync");
            cpu_rms_norm(ref, ref, NULL, hn_dim, hn_tok * hn_head, eps);
            ds4_gpu_tensor_read(x, 0, xv, n * sizeof(float));
            check_close(xv, ref, n, 1e-4f, "head_rms: out matches CPU");

            memcpy(xv, ref, n * sizeof(float));   /* restart from normalized */
            ds4_gpu_tensor_write(x, 0, xv, n * sizeof(float));
            CHECK(ds4_gpu_head_rms_norm_rope_tail_tensor(
                      x, hn_tok, hn_head, hn_dim, hn_nrot, 3u, 2048u, false,
                      10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, eps) != 0,
                  "head_rms: rope_tail");
            CHECK(ds4_gpu_synchronize() != 0, "head_rms: rope sync");
            cpu_rope_tail(ref, hn_tok, hn_head, hn_dim, hn_nrot, 3u, 2048u,
                          0, 10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
            ds4_gpu_tensor_read(x, 0, xv, n * sizeof(float));
            check_close(xv, ref, n, 1e-4f, "head_rms: rope_tail matches CPU");
            free(xv); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
    }

    /* --- Fase 5: HC elementwise ----------------------------------------- */
    {
        const uint32_t n_embd = 32u, n_hc = 4u, n_tokens = 2u;
        const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;

        /* repeat_hc (single row) */
        {
            ds4_gpu_tensor *row = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
                    (uint64_t)n_embd * n_hc * sizeof(float));
            CHECK(row && out, "repeat_hc: alloc");
            if (row && out) {
                float *rv = (float *)malloc(n_embd * sizeof(float));
                float *ov = (float *)malloc(n_embd * n_hc * sizeof(float));
                float *ref = (float *)malloc(n_embd * n_hc * sizeof(float));
                for (uint32_t i = 0; i < n_embd; i++) rv[i] = 0.1f * (float)i - 1.5f;
                for (uint32_t i = 0; i < n_embd * n_hc; i++) ref[i] = rv[i % n_embd];
                ds4_gpu_tensor_write(row, 0, rv, n_embd * sizeof(float));
                CHECK(ds4_gpu_repeat_hc_tensor(out, row, n_embd, n_hc) != 0,
                      "repeat_hc: repeat_hc");
                CHECK(ds4_gpu_synchronize() != 0, "repeat_hc: sync");
                ds4_gpu_tensor_read(out, 0, ov, n_embd * n_hc * sizeof(float));
                check_close(ov, ref, n_embd * n_hc, 1e-6f,
                            "repeat_hc: out matches CPU");
                free(rv); free(ov); free(ref);
            }
            if (row) ds4_gpu_tensor_free(row);
            if (out) ds4_gpu_tensor_free(out);
        }

        /* repeat_hc_rows */
        {
            ds4_gpu_tensor *rows = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_embd * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_embd * n_hc * sizeof(float));
            CHECK(rows && out, "repeat_hc_rows: alloc");
            if (rows && out) {
                const uint32_t ne = n_tokens * n_embd;
                float *rv = (float *)malloc(ne * sizeof(float));
                float *ov = (float *)malloc((uint64_t)ne * n_hc * sizeof(float));
                float *ref = (float *)malloc((uint64_t)ne * n_hc * sizeof(float));
                for (uint32_t i = 0; i < ne; i++) rv[i] = 0.05f * (float)i - 1.2f;
                for (uint32_t t = 0; t < n_tokens; t++)
                    for (uint32_t h = 0; h < n_hc; h++)
                        for (uint32_t d = 0; d < n_embd; d++)
                            ref[(t * n_hc + h) * n_embd + d] = rv[t * n_embd + d];
                ds4_gpu_tensor_write(rows, 0, rv, ne * sizeof(float));
                CHECK(ds4_gpu_repeat_hc_rows_tensor(out, rows, n_tokens, n_embd,
                                                    n_hc) != 0,
                      "repeat_hc_rows: repeat");
                CHECK(ds4_gpu_synchronize() != 0, "repeat_hc_rows: sync");
                ds4_gpu_tensor_read(out, 0, ov, (uint64_t)ne * n_hc * sizeof(float));
                check_close(ov, ref, ne * n_hc, 1e-6f,
                            "repeat_hc_rows: out matches CPU");
                free(rv); free(ov); free(ref);
            }
            if (rows) ds4_gpu_tensor_free(rows);
            if (out) ds4_gpu_tensor_free(out);
        }

        /* hc_weighted_sum (tensor weights) */
        {
            ds4_gpu_tensor *res = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_hc * n_embd * sizeof(float));
            ds4_gpu_tensor *w = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_hc * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_embd * sizeof(float));
            CHECK(res && w && out, "hc_weighted_sum: alloc");
            if (res && w && out) {
                const uint32_t rn = n_tokens * n_hc * n_embd;
                const uint32_t wn = n_tokens * n_hc;
                float *rv = (float *)malloc(rn * sizeof(float));
                float *wv = (float *)malloc(wn * sizeof(float));
                float *ov = (float *)malloc(n_tokens * n_embd * sizeof(float));
                float *ref = (float *)malloc(n_tokens * n_embd * sizeof(float));
                for (uint32_t i = 0; i < rn; i++) rv[i] = 0.03f * (float)(i % 101) - 1.5f;
                for (uint32_t i = 0; i < wn; i++) wv[i] = 0.2f + 0.1f * (float)(i % 5);
                cpu_hc_weighted_sum(ref, rv, wv, n_embd, n_hc, n_tokens, n_hc);
                ds4_gpu_tensor_write(res, 0, rv, rn * sizeof(float));
                ds4_gpu_tensor_write(w, 0, wv, wn * sizeof(float));
                CHECK(ds4_gpu_hc_weighted_sum_tensor(out, res, w, n_embd, n_hc) != 0,
                      "hc_weighted_sum: kernel");
                CHECK(ds4_gpu_synchronize() != 0, "hc_weighted_sum: sync");
                ds4_gpu_tensor_read(out, 0, ov, n_tokens * n_embd * sizeof(float));
                check_close(ov, ref, n_tokens * n_embd, 1e-5f,
                            "hc_weighted_sum: out matches CPU");
                free(rv); free(wv); free(ov); free(ref);
            }
            if (res) ds4_gpu_tensor_free(res);
            if (w) ds4_gpu_tensor_free(w);
            if (out) ds4_gpu_tensor_free(out);
        }

        /* hc_weighted_sum_split (weights in the split layout) */
        {
            ds4_gpu_tensor *res = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_hc * n_embd * sizeof(float));
            ds4_gpu_tensor *split = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * mix_hc * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_embd * sizeof(float));
            CHECK(res && split && out, "hc_weighted_sum_split: alloc");
            if (res && split && out) {
                const uint32_t rn = n_tokens * n_hc * n_embd;
                float *rv = (float *)malloc(rn * sizeof(float));
                float *sv = (float *)malloc((uint64_t)n_tokens * mix_hc * sizeof(float));
                float *ov = (float *)malloc(n_tokens * n_embd * sizeof(float));
                float *ref = (float *)malloc(n_tokens * n_embd * sizeof(float));
                for (uint32_t i = 0; i < rn; i++) rv[i] = 0.02f * (float)(i % 97) - 1.0f;
                for (uint32_t i = 0; i < n_tokens * mix_hc; i++) {
                    sv[i] = 0.15f + 0.05f * (float)(i % 9);
                }
                cpu_hc_weighted_sum(ref, rv, sv, n_embd, n_hc, n_tokens, mix_hc);
                ds4_gpu_tensor_write(res, 0, rv, rn * sizeof(float));
                ds4_gpu_tensor_write(split, 0, sv, (uint64_t)n_tokens * mix_hc * sizeof(float));
                CHECK(ds4_gpu_hc_weighted_sum_split_tensor(out, res, split, n_embd,
                                                           n_hc) != 0,
                      "hc_weighted_sum_split: kernel");
                CHECK(ds4_gpu_synchronize() != 0, "hc_weighted_sum_split: sync");
                ds4_gpu_tensor_read(out, 0, ov, n_tokens * n_embd * sizeof(float));
                check_close(ov, ref, n_tokens * n_embd, 1e-5f,
                            "hc_weighted_sum_split: out matches CPU");
                free(rv); free(sv); free(ov); free(ref);
            }
            if (res) ds4_gpu_tensor_free(res);
            if (split) ds4_gpu_tensor_free(split);
            if (out) ds4_gpu_tensor_free(out);
        }

        /* hc_expand and hc_expand_add (post/comb tensors) */
        {
            ds4_gpu_tensor *bo = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_embd * sizeof(float));
            ds4_gpu_tensor *ba = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_embd * sizeof(float));
            ds4_gpu_tensor *res = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_hc * n_embd * sizeof(float));
            ds4_gpu_tensor *post = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_hc * sizeof(float));
            ds4_gpu_tensor *comb = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_hc * n_hc * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_hc * n_embd * sizeof(float));
            CHECK(bo && ba && res && post && comb && out, "hc_expand: alloc");
            if (bo && ba && res && post && comb && out) {
                const uint32_t rn = n_tokens * n_hc * n_embd;
                float *bv = (float *)malloc(n_tokens * n_embd * sizeof(float));
                float *av = (float *)malloc(n_tokens * n_embd * sizeof(float));
                float *rv = (float *)malloc(rn * sizeof(float));
                float *pv = (float *)malloc(n_tokens * n_hc * sizeof(float));
                float *cv = (float *)malloc(n_tokens * n_hc * n_hc * sizeof(float));
                float *ov = (float *)malloc(rn * sizeof(float));
                float *ref = (float *)malloc(rn * sizeof(float));
                for (uint32_t i = 0; i < n_tokens * n_embd; i++) {
                    bv[i] = 0.04f * (float)(i % 81) - 1.3f;
                    av[i] = 0.02f * (float)(i % 57) - 0.7f;
                }
                for (uint32_t i = 0; i < rn; i++) rv[i] = 0.03f * (float)(i % 93) - 1.1f;
                for (uint32_t i = 0; i < n_tokens * n_hc; i++) pv[i] = 0.3f + 0.1f * (float)(i % 6);
                for (uint32_t i = 0; i < n_tokens * n_hc * n_hc; i++) {
                    cv[i] = 0.05f + 0.03f * (float)(i % 17);
                }
                ds4_gpu_tensor_write(bo, 0, bv, n_tokens * n_embd * sizeof(float));
                ds4_gpu_tensor_write(ba, 0, av, n_tokens * n_embd * sizeof(float));
                ds4_gpu_tensor_write(res, 0, rv, rn * sizeof(float));
                ds4_gpu_tensor_write(post, 0, pv, n_tokens * n_hc * sizeof(float));
                ds4_gpu_tensor_write(comb, 0, cv, n_tokens * n_hc * n_hc * sizeof(float));
                cpu_hc_expand(ref, bv, bv, rv, pv, cv, n_embd, n_hc, n_tokens,
                              n_hc, n_hc * n_hc, 0u, 0u, NULL);
                CHECK(ds4_gpu_hc_expand_tensor(out, bo, res, post, comb, n_embd,
                                               n_hc) != 0,
                      "hc_expand: kernel");
                CHECK(ds4_gpu_synchronize() != 0, "hc_expand: sync");
                ds4_gpu_tensor_read(out, 0, ov, rn * sizeof(float));
                check_close(ov, ref, rn, 1e-5f, "hc_expand: out matches CPU");

                cpu_hc_expand(ref, bv, av, rv, pv, cv, n_embd, n_hc, n_tokens,
                              n_hc, n_hc * n_hc, 1u, 0u, NULL);
                CHECK(ds4_gpu_hc_expand_add_tensor(out, bo, ba, res, post, comb,
                                                   n_embd, n_hc) != 0,
                      "hc_expand_add: kernel");
                CHECK(ds4_gpu_synchronize() != 0, "hc_expand_add: sync");
                ds4_gpu_tensor_read(out, 0, ov, rn * sizeof(float));
                check_close(ov, ref, rn, 1e-5f, "hc_expand_add: out matches CPU");
                free(bv); free(av); free(rv); free(pv); free(cv); free(ov); free(ref);
            }
            if (bo) ds4_gpu_tensor_free(bo);
            if (ba) ds4_gpu_tensor_free(ba);
            if (res) ds4_gpu_tensor_free(res);
            if (post) ds4_gpu_tensor_free(post);
            if (comb) ds4_gpu_tensor_free(comb);
            if (out) ds4_gpu_tensor_free(out);
        }

        /* hc_split_sinkhorn */
        {
            const uint32_t n_rows = 2u, iters = 4u;
            const float epsv = 1e-6f;
            ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(n_rows * 24u * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n_rows * 24u * sizeof(float));
            CHECK(mix && out, "hc_sinkhorn: alloc");
            if (mix && out) {
                float *mv = (float *)malloc(n_rows * 24u * sizeof(float));
                float *ov = (float *)malloc(n_rows * 24u * sizeof(float));
                float *ref = (float *)malloc(n_rows * 24u * sizeof(float));
                const float *scale = (const float *)(g_model + OFF_HC_SCALE);
                const float *base = (const float *)(g_model + OFF_HC_BASE);
                for (uint32_t i = 0; i < n_rows * 24u; i++) {
                    mv[i] = 0.1f * (float)(int32_t)((i * 7u) % 31) - 1.4f;
                }
                for (uint32_t r = 0; r < n_rows; r++) {
                    cpu_hc4_split_one(ref + (uint64_t)r * 24u, mv + (uint64_t)r * 24u,
                                      scale, base, iters, epsv);
                }
                ds4_gpu_tensor_write(mix, 0, mv, n_rows * 24u * sizeof(float));
                CHECK(ds4_gpu_hc_split_sinkhorn_tensor(
                          out, mix, g_model, MODEL_PADDED, OFF_HC_SCALE,
                          OFF_HC_BASE, n_hc, iters, epsv) != 0,
                      "hc_sinkhorn: kernel");
                CHECK(ds4_gpu_synchronize() != 0, "hc_sinkhorn: sync");
                ds4_gpu_tensor_read(out, 0, ov, n_rows * 24u * sizeof(float));
                check_close(ov, ref, n_rows * 24u, 1e-4f,
                            "hc_sinkhorn: out matches CPU");
                free(mv); free(ov); free(ref);
            }
            if (mix) ds4_gpu_tensor_free(mix);
            if (out) ds4_gpu_tensor_free(out);
        }

        /* hc_split_weighted_sum (+ norm) */
        {
            const uint32_t iters = 3u;
            const float epsv = 1e-6f, norm_eps = 1e-5f;
            ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * mix_hc * sizeof(float));
            ds4_gpu_tensor *res = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_hc * n_embd * sizeof(float));
            ds4_gpu_tensor *split = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * mix_hc * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_embd * sizeof(float));
            ds4_gpu_tensor *norm_out = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_embd * sizeof(float));
            CHECK(mix && res && split && out && norm_out,
                  "hc_split_weighted_sum: alloc");
            if (mix && res && split && out && norm_out) {
                const uint32_t rn = n_tokens * n_hc * n_embd;
                float *mv = (float *)malloc((uint64_t)n_tokens * mix_hc * sizeof(float));
                float *rv = (float *)malloc(rn * sizeof(float));
                float *sv = (float *)malloc((uint64_t)n_tokens * mix_hc * sizeof(float));
                float *ov = (float *)malloc(n_tokens * n_embd * sizeof(float));
                float *nv = (float *)malloc(n_tokens * n_embd * sizeof(float));
                float *sref = (float *)malloc((uint64_t)n_tokens * mix_hc * sizeof(float));
                float *ref = (float *)malloc(n_tokens * n_embd * sizeof(float));
                float *nref = (float *)malloc(n_tokens * n_embd * sizeof(float));
                const float *scale = (const float *)(g_model + OFF_HC_SCALE);
                const float *base = (const float *)(g_model + OFF_HC_BASE);
                for (uint32_t i = 0; i < n_tokens * mix_hc; i++) {
                    mv[i] = 0.08f * (float)(int32_t)((i * 9u) % 29) - 1.0f;
                }
                for (uint32_t i = 0; i < rn; i++) rv[i] = 0.04f * (float)(i % 71) - 1.2f;
                for (uint32_t t = 0; t < n_tokens; t++) {
                    cpu_hc4_split_one(sref + (uint64_t)t * mix_hc, mv + (uint64_t)t * mix_hc,
                                      scale, base, iters, epsv);
                }
                cpu_hc_weighted_sum(ref, rv, sref, n_embd, n_hc, n_tokens, mix_hc);
                cpu_rms_norm(nref, ref, (const float *)(g_model + OFF_HC_NORM),
                             n_embd, n_tokens, norm_eps);
                ds4_gpu_tensor_write(mix, 0, mv, (uint64_t)n_tokens * mix_hc * sizeof(float));
                ds4_gpu_tensor_write(res, 0, rv, rn * sizeof(float));
                CHECK(ds4_gpu_hc_split_weighted_sum_tensor(
                          out, split, mix, res, g_model, MODEL_PADDED,
                          OFF_HC_SCALE, OFF_HC_BASE, n_embd, n_hc, iters,
                          epsv) != 0,
                      "hc_split_weighted_sum: kernel");
                CHECK(ds4_gpu_synchronize() != 0, "hc_split_weighted_sum: sync");
                ds4_gpu_tensor_read(split, 0, sv,
                                    (uint64_t)n_tokens * mix_hc * sizeof(float));
                if (memcmp(sv, sref, (uint64_t)n_tokens * mix_hc * sizeof(float)) != 0) {
                    fprintf(stderr, "hc_split_weighted_sum: split[0] got %.6f ref %.6f\n",
                            sv[0], sref[0]);
                }
                ds4_gpu_tensor_read(out, 0, ov, n_tokens * n_embd * sizeof(float));
                check_close(ov, ref, n_tokens * n_embd, 1e-4f,
                            "hc_split_weighted_sum: out matches CPU");

                CHECK(ds4_gpu_hc_split_weighted_sum_norm_tensor(
                          out, norm_out, split, mix, res, g_model, MODEL_PADDED,
                          OFF_HC_SCALE, OFF_HC_BASE, OFF_HC_NORM, n_embd, n_hc,
                          iters, epsv, norm_eps) != 0,
                      "hc_split_weighted_sum_norm: kernel");
                CHECK(ds4_gpu_synchronize() != 0, "hc_split_weighted_sum_norm: sync");
                ds4_gpu_tensor_read(norm_out, 0, nv, n_tokens * n_embd * sizeof(float));
                check_close(nv, nref, n_tokens * n_embd, 1e-4f,
                            "hc_split_weighted_sum_norm: out matches CPU");
                free(mv); free(rv); free(sv); free(ov); free(nv);
                free(sref); free(ref); free(nref);
            }
            if (mix) ds4_gpu_tensor_free(mix);
            if (res) ds4_gpu_tensor_free(res);
            if (split) ds4_gpu_tensor_free(split);
            if (out) ds4_gpu_tensor_free(out);
            if (norm_out) ds4_gpu_tensor_free(norm_out);
        }

        /* hc_expand_split + hc_expand_add_split (split layout) */
        {
            ds4_gpu_tensor *bo = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_embd * sizeof(float));
            ds4_gpu_tensor *ba = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_embd * sizeof(float));
            ds4_gpu_tensor *res = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_hc * n_embd * sizeof(float));
            ds4_gpu_tensor *split = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * mix_hc * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_hc * n_embd * sizeof(float));
            CHECK(bo && ba && res && split && out, "hc_expand_split: alloc");
            if (bo && ba && res && split && out) {
                const uint32_t rn = n_tokens * n_hc * n_embd;
                float *bv = (float *)malloc(n_tokens * n_embd * sizeof(float));
                float *av = (float *)malloc(n_tokens * n_embd * sizeof(float));
                float *rv = (float *)malloc(rn * sizeof(float));
                float *sv = (float *)malloc((uint64_t)n_tokens * mix_hc * sizeof(float));
                float *ov = (float *)malloc(rn * sizeof(float));
                float *ref = (float *)malloc(rn * sizeof(float));
                for (uint32_t i = 0; i < n_tokens * n_embd; i++) {
                    bv[i] = 0.05f * (float)(i % 61) - 1.2f;
                    av[i] = 0.03f * (float)(i % 43) - 0.8f;
                }
                for (uint32_t i = 0; i < rn; i++) rv[i] = 0.02f * (float)(i % 87) - 0.9f;
                for (uint32_t t = 0; t < n_tokens; t++) {
                    for (uint32_t i = 0; i < mix_hc; i++) {
                        sv[t * mix_hc + i] = 0.1f + 0.05f * (float)((t * 13u + i) % 11);
                    }
                }
                ds4_gpu_tensor_write(bo, 0, bv, n_tokens * n_embd * sizeof(float));
                ds4_gpu_tensor_write(ba, 0, av, n_tokens * n_embd * sizeof(float));
                ds4_gpu_tensor_write(res, 0, rv, rn * sizeof(float));
                ds4_gpu_tensor_write(split, 0, sv, (uint64_t)n_tokens * mix_hc * sizeof(float));
                /* post = split + n_hc, comb = split + 2*n_hc (stride mix_hc) */
                cpu_hc_expand(ref, bv, bv, rv, sv + n_hc, sv + 2u * n_hc, n_embd,
                              n_hc, n_tokens, mix_hc, mix_hc, 0u, 0u, NULL);
                CHECK(ds4_gpu_hc_expand_split_tensor(out, bo, res, split, n_embd,
                                                     n_hc) != 0,
                      "hc_expand_split: kernel");
                CHECK(ds4_gpu_synchronize() != 0, "hc_expand_split: sync");
                ds4_gpu_tensor_read(out, 0, ov, rn * sizeof(float));
                check_close(ov, ref, rn, 1e-5f, "hc_expand_split: out matches CPU");

                cpu_hc_expand(ref, bv, av, rv, sv + n_hc, sv + 2u * n_hc, n_embd,
                              n_hc, n_tokens, mix_hc, mix_hc, 1u, 0u, NULL);
                CHECK(ds4_gpu_hc_expand_add_split_tensor(out, bo, ba, res, split,
                                                         n_embd, n_hc) != 0,
                      "hc_expand_add_split: kernel");
                CHECK(ds4_gpu_synchronize() != 0, "hc_expand_add_split: sync");
                ds4_gpu_tensor_read(out, 0, ov, rn * sizeof(float));
                check_close(ov, ref, rn, 1e-5f,
                            "hc_expand_add_split: out matches CPU");
                free(bv); free(av); free(rv); free(sv); free(ov); free(ref);
            }
            if (bo) ds4_gpu_tensor_free(bo);
            if (ba) ds4_gpu_tensor_free(ba);
            if (res) ds4_gpu_tensor_free(res);
            if (split) ds4_gpu_tensor_free(split);
            if (out) ds4_gpu_tensor_free(out);
        }

        /* output_hc_weights */
        {
            ds4_gpu_tensor *pre = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_hc * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
                    (uint64_t)n_tokens * n_hc * sizeof(float));
            CHECK(pre && out, "output_hc_weights: alloc");
            if (pre && out) {
                const float epsv = 1e-6f;
                float *pv = (float *)malloc(n_tokens * n_hc * sizeof(float));
                float *ov = (float *)malloc(n_tokens * n_hc * sizeof(float));
                float *ref = (float *)malloc(n_tokens * n_hc * sizeof(float));
                for (uint32_t i = 0; i < n_tokens * n_hc; i++) {
                    pv[i] = 0.2f * (float)(i % 13) - 1.2f;
                }
                cpu_output_hc_weights(
                        ref, pv, (const float *)(g_model + OFF_OUT_HC_SCALE),
                        (const float *)(g_model + OFF_OUT_HC_BASE), n_hc,
                        n_tokens, epsv);
                ds4_gpu_tensor_write(pre, 0, pv, n_tokens * n_hc * sizeof(float));
                CHECK(ds4_gpu_output_hc_weights_tensor(
                          out, pre, g_model, MODEL_PADDED, OFF_OUT_HC_SCALE,
                          OFF_OUT_HC_BASE, n_hc, epsv) != 0,
                      "output_hc_weights: kernel");
                CHECK(ds4_gpu_synchronize() != 0, "output_hc_weights: sync");
                ds4_gpu_tensor_read(out, 0, ov, n_tokens * n_hc * sizeof(float));
                check_close(ov, ref, n_tokens * n_hc, 1e-5f,
                            "output_hc_weights: out matches CPU");
                free(pv); free(ov); free(ref);
            }
            if (pre) ds4_gpu_tensor_free(pre);
            if (out) ds4_gpu_tensor_free(out);
        }

        /* matmul_q8_0_hc_expand (composition) */
        {
            const uint32_t mm_in = 256u, mm_out = 32u;   /* n_embd == mm_out */
            ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(mm_in * sizeof(float));
            ds4_gpu_tensor *bo = ds4_gpu_tensor_alloc(mm_out * sizeof(float));
            ds4_gpu_tensor *res = ds4_gpu_tensor_alloc(
                    (uint64_t)n_hc * mm_out * sizeof(float));
            ds4_gpu_tensor *split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
                    (uint64_t)n_hc * mm_out * sizeof(float));
            CHECK(x && bo && res && split && out, "matmul_hc: alloc");
            if (x && bo && res && split && out) {
                float *xv = (float *)malloc(mm_in * sizeof(float));
                float *rv = (float *)malloc((uint64_t)n_hc * mm_out * sizeof(float));
                float *sv = (float *)malloc(mix_hc * sizeof(float));
                float *ov = (float *)malloc((uint64_t)n_hc * mm_out * sizeof(float));
                float *ref = (float *)malloc((uint64_t)n_hc * mm_out * sizeof(float));
                for (uint32_t i = 0; i < mm_in; i++) {
                    xv[i] = 0.01f * (float)(int32_t)((i * 5u) % 191) - 0.7f;
                }
                for (uint32_t i = 0; i < n_hc * mm_out; i++) {
                    rv[i] = 0.03f * (float)(i % 73) - 1.0f;
                }
                for (uint32_t i = 0; i < mix_hc; i++) {
                    sv[i] = 0.2f + 0.04f * (float)(i % 13);
                }
                ds4_gpu_tensor_write(x, 0, xv, mm_in * sizeof(float));
                ds4_gpu_tensor_write(res, 0, rv, (uint64_t)n_hc * mm_out * sizeof(float));
                ds4_gpu_tensor_write(split, 0, sv, mix_hc * sizeof(float));
                float *mm = (float *)malloc(mm_out * sizeof(float));
                cpu_matmul_q8(mm, g_model + OFF_Q8, mm_out, mm_in, xv, 1u);
                cpu_hc_expand(ref, mm, mm, rv, sv + n_hc, sv + 2u * n_hc, mm_out,
                              n_hc, 1u, mix_hc, mix_hc, 0u, 0u, NULL);
                free(mm);
                CHECK(ds4_gpu_matmul_q8_0_hc_expand_tensor(
                          out, bo, g_model, MODEL_PADDED, OFF_Q8, mm_in, mm_out,
                          x, res, split, mm_out, n_hc) != 0,
                      "matmul_hc: kernel");
                CHECK(ds4_gpu_synchronize() != 0, "matmul_hc: sync");
                ds4_gpu_tensor_read(out, 0, ov, (uint64_t)n_hc * mm_out * sizeof(float));
                check_close(ov, ref, n_hc * mm_out, 0.01f,
                            "matmul_hc: out matches CPU");
                free(xv); free(rv); free(sv); free(ov); free(ref);
            }
            if (x) ds4_gpu_tensor_free(x);
            if (bo) ds4_gpu_tensor_free(bo);
            if (res) ds4_gpu_tensor_free(res);
            if (split) ds4_gpu_tensor_free(split);
            if (out) ds4_gpu_tensor_free(out);
        }
    }

    /* --- Fase 5: directional steering ---------------------------------- */
    {
        const uint32_t width = 32u, rows = 4u, layer = 2u;
        const float scale = 0.5f;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(
                (uint64_t)rows * width * sizeof(float));
        ds4_gpu_tensor *dir = ds4_gpu_tensor_alloc(
                (uint64_t)(layer + 1u) * width * sizeof(float));
        CHECK(x && dir, "directional: alloc");
        if (x && dir) {
            const uint32_t xn = rows * width, dn = (layer + 1u) * width;
            float *xv = (float *)malloc(xn * sizeof(float));
            float *dv = (float *)malloc(dn * sizeof(float));
            float *ref = (float *)malloc(xn * sizeof(float));
            for (uint32_t i = 0; i < xn; i++) xv[i] = 0.05f * (float)(i % 63) - 1.4f;
            for (uint32_t i = 0; i < dn; i++) dv[i] = 0.04f * (float)(i % 47) - 0.9f;
            memcpy(ref, xv, xn * sizeof(float));
            cpu_directional_steering(ref, dv, layer, width, rows, scale);
            ds4_gpu_tensor_write(x, 0, xv, xn * sizeof(float));
            ds4_gpu_tensor_write(dir, 0, dv, dn * sizeof(float));
            CHECK(ds4_gpu_directional_steering_project_tensor(x, dir, layer, width,
                                                              rows, scale) != 0,
                  "directional: kernel");
            CHECK(ds4_gpu_synchronize() != 0, "directional: sync");
            ds4_gpu_tensor_read(x, 0, xv, xn * sizeof(float));
            check_close(xv, ref, xn, 1e-5f, "directional: out matches CPU");
            free(xv); free(dv); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (dir) ds4_gpu_tensor_free(dir);
    }

    /* --- Fase 5: indexer ------------------------------------------------ */
    {
        const uint32_t n_comp = 16u, n_head = 4u, head_dim = 8u, n_tokens = 4u;
        const uint32_t ratio = 2u;
        const float scale = 0.7f;
        ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * n_head * head_dim * sizeof(float));
        ds4_gpu_tensor *w = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * n_head * sizeof(float));
        ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(
                (uint64_t)n_comp * head_dim * sizeof(float));
        ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * n_comp * sizeof(float));
        CHECK(q && w && comp && scores, "indexer: alloc");
        if (q && w && comp && scores) {
            const uint32_t qn = n_tokens * n_head * head_dim;
            float *qv = (float *)malloc(qn * sizeof(float));
            float *wv = (float *)malloc(n_tokens * n_head * sizeof(float));
            float *cv = (float *)malloc(n_comp * head_dim * sizeof(float));
            float *sv = (float *)malloc(n_tokens * n_comp * sizeof(float));
            float *ref = (float *)malloc(n_tokens * n_comp * sizeof(float));
            for (uint32_t i = 0; i < qn; i++) qv[i] = 0.06f * (float)(i % 41) - 1.2f;
            for (uint32_t i = 0; i < n_tokens * n_head; i++) {
                wv[i] = 0.4f + 0.1f * (float)(i % 6);
            }
            for (uint32_t i = 0; i < n_comp * head_dim; i++) {
                cv[i] = 0.05f * (float)(i % 53) - 1.1f;
            }
            ds4_gpu_tensor_write(q, 0, qv, qn * sizeof(float));
            ds4_gpu_tensor_write(w, 0, wv, n_tokens * n_head * sizeof(float));
            ds4_gpu_tensor_write(comp, 0, cv, n_comp * head_dim * sizeof(float));

            /* score_one (non-causal) */
            cpu_indexer_scores(ref, qv, wv, cv, n_comp, 1u, 0u, n_head,
                               head_dim, 1u, scale, 0u);
            CHECK(ds4_gpu_indexer_score_one_tensor(scores, q, w, comp, n_comp,
                                                   n_head, head_dim, scale) != 0,
                  "indexer: score_one");
            CHECK(ds4_gpu_synchronize() != 0, "indexer: score_one sync");
            ds4_gpu_tensor_read(scores, 0, sv, n_comp * sizeof(float));
            check_close(sv, ref, n_comp, 1e-4f, "indexer: score_one matches CPU");

            /* scores_prefill (causal, pos0=0) */
            cpu_indexer_scores(ref, qv, wv, cv, n_comp, n_tokens, 0u, n_head,
                               head_dim, ratio, scale, 1u);
            CHECK(ds4_gpu_indexer_scores_prefill_tensor(
                      scores, q, w, comp, n_comp, n_tokens, n_head, head_dim,
                      ratio, scale) != 0,
                  "indexer: scores_prefill");
            CHECK(ds4_gpu_synchronize() != 0, "indexer: prefill sync");
            ds4_gpu_tensor_read(scores, 0, sv, n_tokens * n_comp * sizeof(float));
            check_close(sv, ref, n_tokens * n_comp, 1e-4f,
                        "indexer: scores_prefill matches CPU");

            /* scores_decode_batch (causal, pos0=5) */
            cpu_indexer_scores(ref, qv, wv, cv, n_comp, n_tokens, 5u, n_head,
                               head_dim, ratio, scale, 1u);
            CHECK(ds4_gpu_indexer_scores_decode_batch_tensor(
                      scores, q, w, comp, n_comp, n_tokens, 5u, n_head, head_dim,
                      ratio, scale) != 0,
                  "indexer: scores_decode_batch");
            CHECK(ds4_gpu_synchronize() != 0, "indexer: decode sync");
            ds4_gpu_tensor_read(scores, 0, sv, n_tokens * n_comp * sizeof(float));
            check_close(sv, ref, n_tokens * n_comp, 1e-4f,
                        "indexer: scores_decode_batch matches CPU");

            free(qv); free(wv); free(cv); free(sv); free(ref);
        }
        if (q) ds4_gpu_tensor_free(q);
        if (w) ds4_gpu_tensor_free(w);
        if (comp) ds4_gpu_tensor_free(comp);
        if (scores) ds4_gpu_tensor_free(scores);
    }

    /* indexer topk / top1_value / topk_mask / dsv4_indexer_qat / dspark */
    {
        const uint32_t n_comp = 16u, n_tokens = 4u, top_k = 4u;
        ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * n_comp * sizeof(float));
        ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * top_k * sizeof(uint32_t));
        CHECK(scores && sel, "indexer_topk: alloc");
        if (scores && sel) {
            float *sv = (float *)malloc(n_tokens * n_comp * sizeof(float));
            uint32_t *srv = (uint32_t *)malloc(n_tokens * top_k * sizeof(uint32_t));
            uint32_t *ref = (uint32_t *)malloc(n_tokens * top_k * sizeof(uint32_t));
            for (uint32_t i = 0; i < n_tokens * n_comp; i++) {
                sv[i] = 0.05f * (float)((i * 7u) % 97) - 2.0f;
            }
            ds4_gpu_tensor_write(scores, 0, sv, n_tokens * n_comp * sizeof(float));
            cpu_indexer_topk(ref, sv, n_comp, n_tokens, top_k);
            CHECK(ds4_gpu_indexer_topk_tensor(sel, scores, n_comp, n_tokens,
                                              top_k) != 0,
                  "indexer_topk: kernel");
            CHECK(ds4_gpu_synchronize() != 0, "indexer_topk: sync");
            ds4_gpu_tensor_read(sel, 0, srv, n_tokens * top_k * sizeof(uint32_t));
            CHECK(memcmp(srv, ref, n_tokens * top_k * sizeof(uint32_t)) == 0,
                  "indexer_topk: selected matches CPU");
            free(sv); free(srv); free(ref);
        }
        if (scores) ds4_gpu_tensor_free(scores);
        if (sel) ds4_gpu_tensor_free(sel);
    }
    {
        const uint32_t n_comp = 16u, n_tokens = 4u, index_offset = 100u;
        ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * n_comp * sizeof(float));
        ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc(n_tokens * sizeof(uint32_t));
        ds4_gpu_tensor *val = ds4_gpu_tensor_alloc(n_tokens * sizeof(float));
        CHECK(scores && sel && val, "indexer_top1_value: alloc");
        if (scores && sel && val) {
            float *sv = (float *)malloc(n_tokens * n_comp * sizeof(float));
            uint32_t *srv = (uint32_t *)malloc(n_tokens * sizeof(uint32_t));
            float *vv = (float *)malloc(n_tokens * sizeof(float));
            uint32_t *sref = (uint32_t *)malloc(n_tokens * sizeof(uint32_t));
            float *vref = (float *)malloc(n_tokens * sizeof(float));
            for (uint32_t i = 0; i < n_tokens * n_comp; i++) {
                sv[i] = 0.03f * (float)((i * 11u) % 83) - 1.5f;
            }
            ds4_gpu_tensor_write(scores, 0, sv, n_tokens * n_comp * sizeof(float));
            cpu_indexer_top1_value(sref, vref, sv, n_comp, n_tokens, index_offset);
            CHECK(ds4_gpu_indexer_top1_value_tensor(sel, val, scores, n_comp,
                                                    n_tokens, index_offset) != 0,
                  "indexer_top1_value: kernel");
            CHECK(ds4_gpu_synchronize() != 0, "indexer_top1_value: sync");
            ds4_gpu_tensor_read(sel, 0, srv, n_tokens * sizeof(uint32_t));
            ds4_gpu_tensor_read(val, 0, vv, n_tokens * sizeof(float));
            CHECK(memcmp(srv, sref, n_tokens * sizeof(uint32_t)) == 0,
                  "indexer_top1_value: selected matches CPU");
            check_close(vv, vref, n_tokens, 1e-6f,
                        "indexer_top1_value: values match CPU");
            free(sv); free(srv); free(vv); free(sref); free(vref);
        }
        if (scores) ds4_gpu_tensor_free(scores);
        if (sel) ds4_gpu_tensor_free(sel);
        if (val) ds4_gpu_tensor_free(val);
    }
    {
        const uint32_t n_comp = 16u, n_tokens = 4u, top_k = 4u;
        ds4_gpu_tensor *topk = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * top_k * sizeof(uint32_t));
        ds4_gpu_tensor *mask = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * n_comp * sizeof(float));
        CHECK(topk && mask, "topk_mask: alloc");
        if (topk && mask) {
            uint32_t *tv = (uint32_t *)malloc(n_tokens * top_k * sizeof(uint32_t));
            float *mv = (float *)malloc(n_tokens * n_comp * sizeof(float));
            float *ref = (float *)malloc(n_tokens * n_comp * sizeof(float));
            for (uint32_t i = 0; i < n_tokens * top_k; i++) {
                tv[i] = (uint32_t)((i * 7u + 3u) % n_comp);
            }
            ds4_gpu_tensor_write(topk, 0, tv, n_tokens * top_k * sizeof(uint32_t));
            cpu_topk_mask(ref, tv, n_comp, n_tokens, top_k);
            CHECK(ds4_gpu_dsv4_topk_mask_tensor(mask, topk, n_comp, n_tokens,
                                                top_k) != 0,
                  "topk_mask: kernel");
            CHECK(ds4_gpu_synchronize() != 0, "topk_mask: sync");
            ds4_gpu_tensor_read(mask, 0, mv, n_tokens * n_comp * sizeof(float));
            check_close(mv, ref, n_tokens * n_comp, 1e-6f,
                        "topk_mask: matches CPU");
            free(tv); free(mv); free(ref);
        }
        if (topk) ds4_gpu_tensor_free(topk);
        if (mask) ds4_gpu_tensor_free(mask);
    }
    {
        const uint32_t n_rows = 4u, head_dim = 128u;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(
                (uint64_t)n_rows * head_dim * sizeof(float));
        CHECK(x != NULL, "indexer_qat: alloc");
        if (x) {
            float *xv = (float *)malloc(n_rows * head_dim * sizeof(float));
            float *ref = (float *)malloc(n_rows * head_dim * sizeof(float));
            for (uint32_t i = 0; i < n_rows * head_dim; i++) {
                xv[i] = 0.02f * (float)(i % 201) - 1.8f;
            }
            memcpy(ref, xv, n_rows * head_dim * sizeof(float));
            ds4_gpu_tensor_write(x, 0, xv, n_rows * head_dim * sizeof(float));
            CHECK(ds4_gpu_dsv4_indexer_qat_tensor(x, n_rows, head_dim) != 0,
                  "indexer_qat: kernel");
            CHECK(ds4_gpu_synchronize() != 0, "indexer_qat: sync");
            cpu_hadamard_fp4(ref, n_rows);
            ds4_gpu_tensor_read(x, 0, xv, n_rows * head_dim * sizeof(float));
            check_close(xv, ref, n_rows * head_dim, 1e-4f,
                        "indexer_qat: out matches CPU");
            free(xv); free(ref);
        }
        if (x) ds4_gpu_tensor_free(x);
    }
    {
        const uint32_t vocab = 64u, rank = 64u, prev_token = 0u;
        const uint32_t rank_blocks = rank / 32u;
        ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(vocab * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(uint64_t));
        CHECK(logits && out, "dspark: alloc");
        if (logits && out) {
            float *lv = (float *)malloc(vocab * sizeof(float));
            for (uint32_t i = 0; i < vocab; i++) {
                lv[i] = 0.05f * (float)(i % 61) - 1.5f;
            }
            ds4_gpu_tensor_write(logits, 0, lv, vocab * sizeof(float));
            /* CPU reference: compute best over logits + w2·w1[prev] */
            const uint8_t *w1 = g_model + OFF_DS1;
            const uint8_t *w2 = g_model + OFF_DS2;
            float state[64];
            for (uint32_t b = 0; b < rank_blocks; b++) {
                uint16_t sw;
                memcpy(&sw, w1 + (uint64_t)b * 34u, 2);
                const float d = f16_to_f32(sw);
                for (uint32_t k = 0; k < 32u; k++) {
                    state[b * 32u + k] = d * (float)(int8_t)w1[(uint64_t)b * 34u + 2u + k];
                }
            }
            float best_v = -INFINITY;
            uint32_t best_i = 0;
            for (uint32_t i = 0; i < vocab; i++) {
                float acc = 0.0f;
                for (uint32_t b = 0; b < rank_blocks; b++) {
                    const uint8_t *blk = w2 + (uint64_t)i * rank_blocks * 34u +
                                         (uint64_t)b * 34u;
                    uint16_t sw;
                    memcpy(&sw, blk, 2);
                    const float d = f16_to_f32(sw);
                    float s = 0.0f;
                    for (uint32_t k = 0; k < 32u; k++) {
                        s += (float)(int8_t)blk[2u + k] * state[b * 32u + k];
                    }
                    acc += d * s;
                }
                const float v = lv[i] + acc;
                if (v > best_v || (v == best_v && i < best_i)) {
                    best_v = v;
                    best_i = i;
                }
            }
            uint32_t bits;
            memcpy(&bits, &best_v, sizeof(bits));
            const uint32_t fkey = (bits & 0x80000000u) ? ~bits
                                                       : (bits | 0x80000000u);
            const uint64_t ref_key = ((uint64_t)fkey << 32) | (uint32_t)(~best_i);
            CHECK(ds4_gpu_dspark_markov_argmax_tensor(
                      out, logits, g_model, MODEL_PADDED, OFF_DS1, OFF_DS2,
                      prev_token, vocab, rank) != 0,
                  "dspark: kernel");
            CHECK(ds4_gpu_synchronize() != 0, "dspark: sync");
            uint64_t got_key = 0;
            ds4_gpu_tensor_read(out, 0, &got_key, sizeof(uint64_t));
            CHECK(got_key == ref_key, "dspark: key matches CPU");
            if (got_key != ref_key) {
                fprintf(stderr, "dspark: got %016llx ref %016llx\n",
                        (unsigned long long)got_key, (unsigned long long)ref_key);
            }
            free(lv);
        }
        if (logits) ds4_gpu_tensor_free(logits);
        if (out) ds4_gpu_tensor_free(out);
    }

    /* --- Fase 5: compressor --------------------------------------------- */
    {
        const uint32_t head_dim = 8u, ratio = 4u, n_tokens = 8u, pos0 = 2u;
        const uint32_t coff = 2u, width = coff * head_dim;
        const uint32_t state_rows = 8u, n_comp = n_tokens / ratio;
        const uint32_t ape_type = 0u;   /* f32 */
        ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * width * sizeof(float));
        ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * width * sizeof(float));
        ds4_gpu_tensor *st_kv = ds4_gpu_tensor_alloc(
                (uint64_t)state_rows * width * sizeof(float));
        ds4_gpu_tensor *st_sc = ds4_gpu_tensor_alloc(
                (uint64_t)state_rows * width * sizeof(float));
        ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(
                (uint64_t)n_comp * head_dim * sizeof(float));
        CHECK(kv && sc && st_kv && st_sc && comp, "compressor: alloc");
        if (kv && sc && st_kv && st_sc && comp) {
            const uint32_t kvn = n_tokens * width;
            const uint32_t stn = state_rows * width;
            float *kvv = (float *)malloc(kvn * sizeof(float));
            float *scv = (float *)malloc(kvn * sizeof(float));
            float *stv = (float *)malloc(stn * sizeof(float));
            float *stsv = (float *)malloc(stn * sizeof(float));
            float *cv = (float *)malloc(n_comp * head_dim * sizeof(float));
            float *ref = (float *)malloc(n_comp * head_dim * sizeof(float));
            const float *ape = (const float *)(g_model + OFF_APE);
            for (uint32_t i = 0; i < kvn; i++) {
                kvv[i] = 0.04f * (float)(i % 89) - 1.6f;
                scv[i] = 0.03f * (float)(i % 71) - 1.0f;
            }
            ds4_gpu_tensor_write(kv, 0, kvv, kvn * sizeof(float));
            ds4_gpu_tensor_write(sc, 0, scv, kvn * sizeof(float));

            /* store_batch */
            memset(stv, 0, stn * sizeof(float));
            memset(stsv, 0, stn * sizeof(float));
            ds4_gpu_tensor_write(st_kv, 0, stv, stn * sizeof(float));
            ds4_gpu_tensor_write(st_sc, 0, stsv, stn * sizeof(float));
            for (uint32_t t = 0; t < n_tokens; t++) {
                const uint32_t pos_mod = (pos0 + t) % ratio;
                const uint32_t dst_row = ratio + pos_mod;
                for (uint32_t j = 0; j < width; j++) {
                    stv[dst_row * width + j] = kvv[t * width + j];
                    stsv[dst_row * width + j] =
                        scv[t * width + j] + ape[pos_mod * width + j];
                }
            }
            CHECK(ds4_gpu_compressor_store_batch_tensor(
                      kv, sc, st_kv, st_sc, g_model, MODEL_PADDED, OFF_APE,
                      ape_type, head_dim, ratio, pos0, n_tokens) != 0,
                  "compressor: store_batch");
            CHECK(ds4_gpu_synchronize() != 0, "compressor: store_batch sync");
            ds4_gpu_tensor_read(st_kv, 0, stv, stn * sizeof(float));
            ds4_gpu_tensor_read(st_sc, 0, stsv, stn * sizeof(float));
            /* reference copy of state */
            float *stk_ref = (float *)malloc(stn * sizeof(float));
            float *sts_ref = (float *)malloc(stn * sizeof(float));
            memset(stk_ref, 0, stn * sizeof(float));
            memset(sts_ref, 0, stn * sizeof(float));
            for (uint32_t t = 0; t < n_tokens; t++) {
                const uint32_t pos_mod = (pos0 + t) % ratio;
                const uint32_t dst_row = ratio + pos_mod;
                for (uint32_t j = 0; j < width; j++) {
                    stk_ref[dst_row * width + j] = kvv[t * width + j];
                    sts_ref[dst_row * width + j] =
                        scv[t * width + j] + ape[pos_mod * width + j];
                }
            }
            check_close(stv, stk_ref, stn, 1e-5f, "compressor: store kv matches CPU");
            check_close(stsv, sts_ref, stn, 1e-5f, "compressor: store score matches CPU");

            /* prefill (pos0=2, n_tokens=8, ratio=4) */
            const uint32_t cutoff = n_comp * ratio;   /* 8, rem=0 */
            memset(stk_ref, 0, stn * sizeof(float));
            for (uint32_t i = 0; i < stn; i++) sts_ref[i] = -INFINITY;
            if (cutoff >= ratio) {
                const uint32_t prev_start = cutoff - ratio;
                for (uint32_t r = 0; r < ratio; r++) {
                    const uint32_t src = prev_start + r;
                    const uint32_t phase = (pos0 + src) % ratio;
                    for (uint32_t j = 0; j < width; j++) {
                        stk_ref[r * width + j] = kvv[src * width + j];
                        sts_ref[r * width + j] =
                            scv[src * width + j] + ape[phase * width + j];
                    }
                }
            }
            /* pool reference */
            for (uint32_t c = 0; c < n_comp; c++) {
                for (uint32_t d = 0; d < head_dim; d++) {
                    float vals[8], scores[8];
                    float max_s = -INFINITY;
                    uint32_t n_cand = 0;
                    if (c > 0) {
                        uint32_t base = (c - 1u) * ratio;
                        for (uint32_t r = 0; r < 4u; r++) {
                            uint32_t t = base + r;
                            float apev = ape[((pos0 + t) % ratio) * width + d];
                            vals[n_cand] = kvv[t * width + d];
                            scores[n_cand] = scv[t * width + d] + apev;
                            max_s = fmaxf(max_s, scores[n_cand++]);
                        }
                    }
                    uint32_t base = c * ratio;
                    for (uint32_t r = 0; r < 4u; r++) {
                        uint32_t t = base + r;
                        float apev = ape[((pos0 + t) % ratio) * width + head_dim + d];
                        vals[n_cand] = kvv[t * width + head_dim + d];
                        scores[n_cand] = scv[t * width + head_dim + d] + apev;
                        max_s = fmaxf(max_s, scores[n_cand++]);
                    }
                    float den = 0.0f, acc = 0.0f;
                    for (uint32_t i = 0; i < n_cand; i++) {
                        float wgt = expf(scores[i] - max_s);
                        den += wgt;
                        acc += vals[i] * wgt;
                    }
                    ref[c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
                }
            }
            CHECK(ds4_gpu_compressor_prefill_tensor(
                      comp, st_kv, st_sc, kv, sc, g_model, MODEL_PADDED, OFF_APE,
                      ape_type, OFF_HC_NORM, 0u, head_dim, ratio, pos0, n_tokens,
                      0u, 2048u, false, 10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                      1e-5f) != 0,
                  "compressor: prefill");
            CHECK(ds4_gpu_synchronize() != 0, "compressor: prefill sync");
            cpu_rms_norm(ref, ref, (const float *)(g_model + OFF_HC_NORM),
                         head_dim, n_comp, 1e-5f);
            ds4_gpu_tensor_read(comp, 0, cv, n_comp * head_dim * sizeof(float));
            check_close(cv, ref, n_comp * head_dim, 1e-4f,
                        "compressor: prefill comp matches CPU");
            free(stk_ref); free(sts_ref);

            /* update (emit at pos=3) */
            {
                const uint32_t pos = 3u, comp_row = 0u;
                /* fresh state via store at pos */
                float *stk2 = (float *)malloc(stn * sizeof(float));
                float *sts2 = (float *)malloc(stn * sizeof(float));
                memset(stk2, 0, stn * sizeof(float));
                memset(sts2, 0, stn * sizeof(float));
                for (uint32_t t = 0; t < 1u; t++) {
                    const uint32_t pos_mod = (pos + t) % ratio;
                    const uint32_t dst_row = ratio + pos_mod;
                    for (uint32_t j = 0; j < width; j++) {
                        stk2[dst_row * width + j] = kvv[t * width + j];
                        sts2[dst_row * width + j] =
                            scv[t * width + j] + ape[pos_mod * width + j];
                    }
                }
                ds4_gpu_tensor_write(st_kv, 0, stk2, stn * sizeof(float));
                ds4_gpu_tensor_write(st_sc, 0, sts2, stn * sizeof(float));
                /* update with state_already_stored=true */
                CHECK(ds4_gpu_compressor_update_tensor(
                          kv, sc, st_kv, st_sc, comp, g_model, MODEL_PADDED,
                          OFF_APE, ape_type, OFF_HC_NORM, 0u, head_dim, ratio,
                          pos, comp_row, 0u, 2048u, 10000.0f, 1.0f, 0.0f, 1.0f,
                          1.0f, 1.0f, 1e-5f, true, true, false) != 0,
                      "compressor: update");
                CHECK(ds4_gpu_synchronize() != 0, "compressor: update sync");
                /* reference: pool rows 0-3 first half + rows 4-7 second half */
                float *uref = (float *)malloc(head_dim * sizeof(float));
                for (uint32_t d = 0; d < head_dim; d++) {
                    float vals[8], scores[8];
                    float max_s = -INFINITY;
                    uint32_t n_cand = 0;
                    for (uint32_t r = 0; r < 4u; r++) {
                        vals[n_cand] = stk2[r * width + d];
                        scores[n_cand] = sts2[r * width + d];
                        max_s = fmaxf(max_s, scores[n_cand++]);
                    }
                    for (uint32_t r = 0; r < 4u; r++) {
                        vals[n_cand] = stk2[(ratio + r) * width + head_dim + d];
                        scores[n_cand] = sts2[(ratio + r) * width + head_dim + d];
                        max_s = fmaxf(max_s, scores[n_cand++]);
                    }
                    float den = 0.0f, acc = 0.0f;
                    for (uint32_t i = 0; i < n_cand; i++) {
                        float wgt = expf(scores[i] - max_s);
                        den += wgt;
                        acc += vals[i] * wgt;
                    }
                    uref[d] = den != 0.0f ? acc / den : 0.0f;
                }
                /* then rms_norm_weight (n_rot=0) */
                cpu_rms_norm(uref, uref, (const float *)(g_model + OFF_HC_NORM),
                             head_dim, 1u, 1e-5f);
                ds4_gpu_tensor_read(comp, 0, cv, n_comp * head_dim * sizeof(float));
                check_close(cv, uref, head_dim, 1e-4f,
                            "compressor: update comp row matches CPU");
                free(stk2); free(sts2); free(uref);
            }
            free(kvv); free(scv); free(stv); free(stsv); free(cv); free(ref);
        }
        if (kv) ds4_gpu_tensor_free(kv);
        if (sc) ds4_gpu_tensor_free(sc);
        if (st_kv) ds4_gpu_tensor_free(st_kv);
        if (st_sc) ds4_gpu_tensor_free(st_sc);
        if (comp) ds4_gpu_tensor_free(comp);
    }

    /* compressor ratio4 replay + state */
    {
        const uint32_t head_dim = 8u, ratio = 4u, n_tokens = 8u, pos0 = 0u;
        const uint32_t width = 2u * head_dim, state_rows = 8u;
        const uint32_t n_comp = n_tokens / ratio;
        const uint32_t ape_type = 0u;
        ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * width * sizeof(float));
        ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(
                (uint64_t)n_tokens * width * sizeof(float));
        ds4_gpu_tensor *st_kv = ds4_gpu_tensor_alloc(
                (uint64_t)state_rows * width * sizeof(float));
        ds4_gpu_tensor *st_sc = ds4_gpu_tensor_alloc(
                (uint64_t)state_rows * width * sizeof(float));
        ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(
                (uint64_t)n_comp * head_dim * sizeof(float));
        CHECK(kv && sc && st_kv && st_sc && comp, "compressor ratio4: alloc");
        if (kv && sc && st_kv && st_sc && comp) {
            const uint32_t kvn = n_tokens * width;
            const uint32_t stn = state_rows * width;
            float *kvv = (float *)malloc(kvn * sizeof(float));
            float *scv = (float *)malloc(kvn * sizeof(float));
            float *cv = (float *)malloc(n_comp * head_dim * sizeof(float));
            float *ref = (float *)malloc(n_comp * head_dim * sizeof(float));
            float *stk2 = (float *)malloc(stn * sizeof(float));
            float *sts2 = (float *)malloc(stn * sizeof(float));
            const float *ape = (const float *)(g_model + OFF_APE);
            for (uint32_t i = 0; i < kvn; i++) {
                kvv[i] = 0.03f * (float)(i % 97) - 1.3f;
                scv[i] = 0.02f * (float)(i % 59) - 0.8f;
            }
            ds4_gpu_tensor_write(kv, 0, kvv, kvn * sizeof(float));
            ds4_gpu_tensor_write(sc, 0, scv, kvn * sizeof(float));

            /* prefill_ratio4_replay: state rows 0-3 hold the previous group.
             * Build them as kv rows (n_tokens - ratio .. n_tokens - 1) */
            for (uint32_t r = 0; r < 4u; r++) {
                const uint32_t src = n_tokens - ratio + r;
                const uint32_t phase = (pos0 + src) % ratio;
                for (uint32_t j = 0; j < width; j++) {
                    stk2[r * width + j] = kvv[src * width + j];
                    sts2[r * width + j] = scv[src * width + j] +
                                          ape[phase * width + j];
                }
            }
            ds4_gpu_tensor_write(st_kv, 0, stk2, stn * sizeof(float));
            ds4_gpu_tensor_write(st_sc, 0, sts2, stn * sizeof(float));

            /* pool with replay=1: c==0 uses state rows 0-3; other c uses kv */
            for (uint32_t c = 0; c < n_comp; c++) {
                for (uint32_t d = 0; d < head_dim; d++) {
                    float vals[8], scores[8];
                    float max_s = -INFINITY;
                    uint32_t n_cand = 0;
                    if (c == 0) {
                        for (uint32_t r = 0; r < 4u; r++) {
                            vals[n_cand] = stk2[r * width + d];
                            scores[n_cand] = sts2[r * width + d];
                            max_s = fmaxf(max_s, scores[n_cand++]);
                        }
                    } else {
                        uint32_t base = (c - 1u) * ratio;
                        for (uint32_t r = 0; r < 4u; r++) {
                            uint32_t t = base + r;
                            float apev = ape[((pos0 + t) % ratio) * width + d];
                            vals[n_cand] = kvv[t * width + d];
                            scores[n_cand] = scv[t * width + d] + apev;
                            max_s = fmaxf(max_s, scores[n_cand++]);
                        }
                    }
                    uint32_t base = c * ratio;
                    for (uint32_t r = 0; r < 4u; r++) {
                        uint32_t t = base + r;
                        float apev = ape[((pos0 + t) % ratio) * width + head_dim + d];
                        vals[n_cand] = kvv[t * width + head_dim + d];
                        scores[n_cand] = scv[t * width + head_dim + d] + apev;
                        max_s = fmaxf(max_s, scores[n_cand++]);
                    }
                    float den = 0.0f, acc = 0.0f;
                    for (uint32_t i = 0; i < n_cand; i++) {
                        float wgt = expf(scores[i] - max_s);
                        den += wgt;
                        acc += vals[i] * wgt;
                    }
                    ref[c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
                }
            }
            cpu_rms_norm(ref, ref, (const float *)(g_model + OFF_HC_NORM),
                         head_dim, n_comp, 1e-5f);
            CHECK(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                      comp, st_kv, st_sc, kv, sc, g_model, MODEL_PADDED, OFF_APE,
                      ape_type, OFF_HC_NORM, 0u, head_dim, pos0, n_tokens, 0u,
                      2048u, false, 10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                      1e-5f) != 0,
                  "compressor: prefill_ratio4_replay");
            CHECK(ds4_gpu_synchronize() != 0, "compressor: replay sync");
            ds4_gpu_tensor_read(comp, 0, cv, n_comp * head_dim * sizeof(float));
            check_close(cv, ref, n_comp * head_dim, 1e-4f,
                        "compressor: replay comp matches CPU");

            /* prefill_state_ratio4: state from the last 4 kv rows */
            memset(stk2, 0, stn * sizeof(float));
            for (uint32_t i = 0; i < stn; i++) sts2[i] = -INFINITY;
            for (uint32_t r = 0; r < 4u; r++) {
                const uint32_t src = 0u + r;
                const uint32_t phase = (pos0 + src) % ratio;
                for (uint32_t j = 0; j < width; j++) {
                    stk2[r * width + j] = kvv[src * width + j];
                    sts2[r * width + j] = scv[src * width + j] +
                                          ape[phase * width + j];
                }
            }
            ds4_gpu_tensor_write(st_kv, 0, stk2, stn * sizeof(float));
            ds4_gpu_tensor_write(st_sc, 0, sts2, stn * sizeof(float));
            CHECK(ds4_gpu_compressor_prefill_state_ratio4_tensor(
                      st_kv, st_sc, kv, sc, g_model, MODEL_PADDED, OFF_APE,
                      ape_type, head_dim, pos0) != 0,
                  "compressor: prefill_state_ratio4");
            CHECK(ds4_gpu_synchronize() != 0, "compressor: state sync");
            ds4_gpu_tensor_read(st_kv, 0, stk2, stn * sizeof(float));
            ds4_gpu_tensor_read(st_sc, 0, sts2, stn * sizeof(float));
            float *skr = (float *)malloc(stn * sizeof(float));
            float *ssr = (float *)malloc(stn * sizeof(float));
            memset(skr, 0, stn * sizeof(float));
            for (uint32_t i = 0; i < stn; i++) ssr[i] = -INFINITY;
            for (uint32_t r = 0; r < 4u; r++) {
                const uint32_t phase = (pos0 + r) % ratio;
                for (uint32_t j = 0; j < width; j++) {
                    skr[r * width + j] = kvv[r * width + j];
                    ssr[r * width + j] = scv[r * width + j] +
                                         ape[phase * width + j];
                }
            }
            check_close(stk2, skr, stn, 1e-5f, "compressor: state kv matches CPU");
            check_close(sts2, ssr, stn, 1e-5f, "compressor: state score matches CPU");
            free(skr); free(ssr);
            free(kvv); free(scv); free(cv); free(ref); free(stk2); free(sts2);
        }
        if (kv) ds4_gpu_tensor_free(kv);
        if (sc) ds4_gpu_tensor_free(sc);
        if (st_kv) ds4_gpu_tensor_free(st_kv);
        if (st_sc) ds4_gpu_tensor_free(st_sc);
        if (comp) ds4_gpu_tensor_free(comp);
    }

    /* --- Fase 6: fused Q/KV RMS norm (+ KV RoPE tail) -------------------- */
    {
        const uint32_t rows = 2u, q_n = 64u, kv_n_head = 1u, kv_head_dim = 128u;
        const uint32_t kv_n = kv_n_head * kv_head_dim;
        const uint32_t n_rot = 32u;
        const uint32_t pos0 = 3u, n_ctx_orig = 4096u;
        const float freq_base = 10000.0f, freq_scale = 1.0f;
        const float ext_factor = 0.0f, attn_factor = 1.0f;
        const float beta_fast = 32.0f, beta_slow = 1.0f;
        const float eps = 1e-5f;
        ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)rows * q_n * sizeof(float));
        ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)rows * kv_n * sizeof(float));
        ds4_gpu_tensor *qo = ds4_gpu_tensor_alloc((uint64_t)rows * q_n * sizeof(float));
        ds4_gpu_tensor *kvo = ds4_gpu_tensor_alloc((uint64_t)rows * kv_n * sizeof(float));
        CHECK(q && kv && qo && kvo, "qkv_rms: alloc");
        if (q && kv && qo && kvo) {
            float *qv = (float *)malloc((uint64_t)rows * q_n * sizeof(float));
            float *kvv = (float *)malloc((uint64_t)rows * kv_n * sizeof(float));
            float *qov = (float *)malloc((uint64_t)rows * q_n * sizeof(float));
            float *kvov = (float *)malloc((uint64_t)rows * kv_n * sizeof(float));
            float *q_ref = (float *)malloc((uint64_t)rows * q_n * sizeof(float));
            float *kv_ref = (float *)malloc((uint64_t)rows * kv_n * sizeof(float));
            for (uint32_t i = 0; i < rows * q_n; i++) {
                qv[i] = 0.03f * (float)(int32_t)((i * 5u) % 131) - 1.3f;
            }
            for (uint32_t i = 0; i < rows * kv_n; i++) {
                kvv[i] = 0.02f * (float)(int32_t)((i * 7u) % 173) - 0.9f;
            }
            ds4_gpu_tensor_write(q, 0, qv, (uint64_t)rows * q_n * sizeof(float));
            ds4_gpu_tensor_write(kv, 0, kvv, (uint64_t)rows * kv_n * sizeof(float));
            /* Weights: q uses OFF_F32_NORM, kv uses OFF_F32_NORM + 64. */
            const uint64_t q_w = OFF_F32_NORM;
            const uint64_t kv_w = OFF_F32_NORM + (uint64_t)q_n * sizeof(float);

            /* rows (no rope) */
            cpu_rms_norm(q_ref, qv, (const float *)(g_model + q_w),
                         q_n, rows, eps);
            cpu_rms_norm(kv_ref, kvv, (const float *)(g_model + kv_w),
                         kv_n, rows, eps);
            CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                      qo, q, g_model, MODEL_PADDED, q_w, q_n,
                      kvo, kv, kv_w, kv_n, rows, eps) != 0,
                  "qkv_rms: rows tensor");
            CHECK(ds4_gpu_synchronize() != 0, "qkv_rms: rows sync");
            ds4_gpu_tensor_read(qo, 0, qov, (uint64_t)rows * q_n * sizeof(float));
            ds4_gpu_tensor_read(kvo, 0, kvov, (uint64_t)rows * kv_n * sizeof(float));
            check_close(qov, q_ref, rows * q_n, 1e-4f, "qkv_rms: q matches CPU");
            check_close(kvov, kv_ref, rows * kv_n, 1e-4f, "qkv_rms: kv matches CPU");

            /* rows + KV rope tail */
            cpu_rms_norm(q_ref, qv, (const float *)(g_model + q_w),
                         q_n, rows, eps);
            cpu_rms_norm(kv_ref, kvv, (const float *)(g_model + kv_w),
                         kv_n, rows, eps);
            cpu_rope_tail(kv_ref, rows, kv_n_head, kv_head_dim, n_rot, pos0,
                          n_ctx_orig, 0, freq_base, freq_scale, ext_factor,
                          attn_factor, beta_fast, beta_slow);
            CHECK(ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                      qo, q, g_model, MODEL_PADDED, q_w, q_n,
                      kvo, kv, kv_w, kv_n, rows, kv_n_head, kv_head_dim,
                      n_rot, pos0, n_ctx_orig, false, freq_base, freq_scale,
                      ext_factor, attn_factor, beta_fast, beta_slow, eps) != 0,
                  "qkv_rms: rows+rope tensor");
            CHECK(ds4_gpu_synchronize() != 0, "qkv_rms: rows+rope sync");
            ds4_gpu_tensor_read(qo, 0, qov, (uint64_t)rows * q_n * sizeof(float));
            ds4_gpu_tensor_read(kvo, 0, kvov, (uint64_t)rows * kv_n * sizeof(float));
            check_close(qov, q_ref, rows * q_n, 1e-4f, "qkv_rms: q+rope matches CPU");
            check_close(kvov, kv_ref, rows * kv_n, 1e-4f, "qkv_rms: kv+rope matches CPU");

            free(qv); free(kvv); free(qov); free(kvov);
            free(q_ref); free(kv_ref);
        }
        if (q) ds4_gpu_tensor_free(q);
        if (kv) ds4_gpu_tensor_free(kv);
        if (qo) ds4_gpu_tensor_free(qo);
        if (kvo) ds4_gpu_tensor_free(kvo);
    }

    /* --- Fase 6 step 2: staging pool re-pointing (set_model_map_spans) --- */
    {
        /* Re-point the model wrapper at a sub-window and re-run weight-backed
         * kernels: exercises the host upload (memcpy into the staging pool)
         * plus the absolute->relative offset translation by g_model_base. */
        const uint64_t f32_off = OFF_F32_MM;
        const uint64_t f32_bytes = (uint64_t)16u * 128u * 4u;   /* 16x128 F32 */
        const uint64_t q8_off = OFF_Q8;
        const uint64_t f16_off = OFF_F16;
        const uint64_t f16_bytes = (uint64_t)32u * 256u * 2u;   /* 32x256 F16 */

        /* Single span starting at a non-zero base. */
        const uint64_t off1[1] = { f32_off };
        const uint64_t sz1[1] = { f32_bytes };
        CHECK(ds4_gpu_set_model_map_spans(g_model, MODEL_PADDED, off1, sz1, 1,
                                          0) != 0,
              "staging: set_model_map_spans single span (base != 0)");
        {
            const uint32_t in_dim = 128u, out_dim = 16u, n_tok = 1u;
            ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(n_tok * in_dim * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
            CHECK(x && out, "staging: alloc f32");
            if (x && out) {
                float *xv = (float *)malloc(n_tok * in_dim * sizeof(float));
                float *ov = (float *)malloc(n_tok * out_dim * sizeof(float));
                float *ref = (float *)malloc(n_tok * out_dim * sizeof(float));
                for (uint32_t i = 0; i < n_tok * in_dim; i++) {
                    xv[i] = 0.01f * (float)(int32_t)(i % 250) - 0.7f;
                }
                ds4_gpu_tensor_write(x, 0, xv, n_tok * in_dim * sizeof(float));
                CHECK(ds4_gpu_matmul_f32_tensor(out, g_model, MODEL_PADDED,
                                                f32_off, in_dim, out_dim,
                                                x, n_tok) != 0,
                      "staging: matmul_f32 on span");
                CHECK(ds4_gpu_synchronize() != 0, "staging: sync f32");
                ds4_gpu_tensor_read(out, 0, ov, n_tok * out_dim * sizeof(float));
                cpu_matmul_f32(ref, (const float *)(g_model + f32_off),
                               out_dim, in_dim, xv, n_tok);
                check_close(ov, ref, n_tok * out_dim, 0.01f,
                            "staging: f32 span matches CPU");
                free(xv); free(ov); free(ref);
            }
            if (x) ds4_gpu_tensor_free(x);
            if (out) ds4_gpu_tensor_free(out);
        }

        /* Union of two spans (F16 + F32 regions): base = min(OFF_F16, ...). */
        const uint64_t off2[2] = { f16_off, f32_off };
        const uint64_t sz2[2] = { f16_bytes, f32_bytes };
        CHECK(ds4_gpu_set_model_map_spans(g_model, MODEL_PADDED, off2, sz2, 2,
                                          0) != 0,
              "staging: set_model_map_spans union of two spans");
        {
            const uint32_t in_dim = 128u, out_dim = 16u, n_tok = 1u;
            ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(n_tok * in_dim * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
            CHECK(x && out, "staging: alloc union");
            if (x && out) {
                float *xv = (float *)malloc(n_tok * in_dim * sizeof(float));
                float *ov = (float *)malloc(n_tok * out_dim * sizeof(float));
                float *ref = (float *)malloc(n_tok * out_dim * sizeof(float));
                for (uint32_t i = 0; i < n_tok * in_dim; i++) {
                    xv[i] = 0.01f * (float)(int32_t)(i % 250) - 0.7f;
                }
                ds4_gpu_tensor_write(x, 0, xv, n_tok * in_dim * sizeof(float));
                CHECK(ds4_gpu_matmul_f32_tensor(out, g_model, MODEL_PADDED,
                                                f32_off, in_dim, out_dim,
                                                x, n_tok) != 0,
                      "staging: matmul_f32 on union");
                CHECK(ds4_gpu_synchronize() != 0, "staging: sync union f32");
                ds4_gpu_tensor_read(out, 0, ov, n_tok * out_dim * sizeof(float));
                cpu_matmul_f32(ref, (const float *)(g_model + f32_off),
                               out_dim, in_dim, xv, n_tok);
                check_close(ov, ref, n_tok * out_dim, 0.01f,
                            "staging: f32 union matches CPU");
                free(xv); free(ov); free(ref);
            }
            if (x) ds4_gpu_tensor_free(x);
            if (out) ds4_gpu_tensor_free(out);
        }
        {
            const uint32_t in_dim = 256u, out_dim = 32u, n_tok = 1u;
            ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(n_tok * in_dim * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
            CHECK(x && out, "staging: alloc f16");
            if (x && out) {
                float *xv = (float *)malloc(n_tok * in_dim * sizeof(float));
                float *ov = (float *)malloc(n_tok * out_dim * sizeof(float));
                float *ref = (float *)malloc(n_tok * out_dim * sizeof(float));
                for (uint32_t i = 0; i < n_tok * in_dim; i++) {
                    xv[i] = 0.01f * (float)(int32_t)(i % 350) - 0.8f;
                }
                ds4_gpu_tensor_write(x, 0, xv, n_tok * in_dim * sizeof(float));
                CHECK(ds4_gpu_matmul_f16_tensor(out, g_model, MODEL_PADDED,
                                                f16_off, in_dim, out_dim,
                                                x, n_tok) != 0,
                      "staging: matmul_f16 on union");
                CHECK(ds4_gpu_synchronize() != 0, "staging: sync union f16");
                ds4_gpu_tensor_read(out, 0, ov, n_tok * out_dim * sizeof(float));
                cpu_matmul_f16(ref, (const uint16_t *)(g_model + f16_off),
                               out_dim, in_dim, xv, n_tok);
                check_close(ov, ref, n_tok * out_dim, 0.01f,
                            "staging: f16 union matches CPU");
                free(xv); free(ov); free(ref);
            }
            if (x) ds4_gpu_tensor_free(x);
            if (out) ds4_gpu_tensor_free(out);
        }

        /* A kernel bound outside the staged window must fail cleanly (OFF_Q8
         * lies below the current base). */
        {
            ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(256u * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(32u * sizeof(float));
            CHECK(x && out, "staging: alloc out-of-window");
            if (x && out) {
                float *xv = (float *)malloc(256u * sizeof(float));
                for (uint32_t i = 0; i < 256u; i++) xv[i] = 0.001f;
                ds4_gpu_tensor_write(x, 0, xv, 256u * sizeof(float));
                free(xv);
                CHECK(ds4_gpu_matmul_q8_0_tensor(out, g_model, MODEL_PADDED,
                                                 q8_off, 256u, 32u, x, 1u) == 0,
                      "staging: out-of-window weights fail cleanly");
            }
            if (x) ds4_gpu_tensor_free(x);
            if (out) ds4_gpu_tensor_free(out);
        }

        /* Restore the full model for the streaming block below. */
        CHECK(ds4_gpu_set_model_map(g_model, MODEL_PADDED) != 0,
              "staging: restore full model map");
        CHECK(ds4_gpu_synchronize() != 0, "staging: sync restore");
    }

    /* --- Fase 6: streaming expert cache (all-resident policy) ------------ */
    {
        const uint64_t gate_expert_bytes = 32u * 2u * 34u;  /* 8x32x64 Q8_0 */
        const uint64_t down_expert_bytes = 32u * 1u * 34u;  /* 8x32x32 Q8_0 */
        const uint32_t budget = 64u;
        ds4_gpu_set_streaming_expert_cache_budget(budget);
        ds4_gpu_set_streaming_expert_cache_expert_bytes(
                gate_expert_bytes + down_expert_bytes);
        CHECK(ds4_gpu_stream_expert_cache_configured_count() == budget,
              "stream: configured_count reflects budget");
        CHECK(ds4_gpu_stream_expert_cache_current_count() == budget,
              "stream: current_count reflects all-resident budget");
        CHECK(ds4_gpu_stream_expert_cache_budget_for_expert_size(
                  gate_expert_bytes, down_expert_bytes) >= budget,
              "stream: budget_for_expert_size reports at least the configured budget");
        CHECK(ds4_gpu_stream_expert_cache_budget_for_expert_size(0, 0) == 0,
              "stream: budget_for_expert_size rejects zero sizes");

        const ds4_gpu_stream_expert_table table = {
            .model_map = g_model,
            .model_size = MODEL_PADDED,
            .layer = 0,
            .n_total_expert = 8,
            .gate_offset = OFF_MOE_GATE,
            .up_offset = OFF_MOE_UP,
            .down_offset = OFF_MOE_DOWN,
            .gate_expert_bytes = gate_expert_bytes,
            .down_expert_bytes = down_expert_bytes,
        };
        int32_t ids[6] = { 0, 3, 1, 5, 2, 7 };
        uint32_t pri[6] = { 6, 5, 4, 3, 2, 1 };
        int32_t batch_ids[24];
        for (uint32_t i = 0; i < 24; i++) {
            batch_ids[i] = (int32_t)((i * 5u + 1u) % 8u);
        }
        CHECK(ds4_gpu_stream_expert_cache_seed_selected(&table, ids, 6) != 0,
              "stream: seed_selected reports all-resident success");
        CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, ids, 6) != 0,
              "stream: begin_selected_load keeps GPU MoE path");
        CHECK(ds4_gpu_stream_expert_cache_prepare_selected_batch(
                  &table, batch_ids, 4, 6) != 0,
              "stream: prepare_selected_batch seeds the layer pool");
        CHECK(ds4_gpu_stream_expert_cache_seed_experts(&table, ids, pri, 6) != 0,
              "stream: seed_experts succeeds");
        CHECK(ds4_gpu_stream_expert_cache_load_layer(&table) != 0,
              "stream: load_layer succeeds");
        CHECK(ds4_gpu_stream_expert_cache_seed_from_layer_selected(
                  &table, NULL, 1, 1, 6) != 0,
              "stream: seed_from_layer_selected succeeds");
        CHECK(ds4_gpu_stream_expert_cache_finish_pending_batch() != 0,
              "stream: finish_pending_batch succeeds");
        CHECK(ds4_gpu_stream_expert_cache_release_layer_cache() != 0,
              "stream: release_layer_cache succeeds");
        CHECK(ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(
                  &table, ids, pri, 6) != 0,
              "stream: seed_experts_gpu_copy succeeds");
        CHECK(ds4_gpu_stream_expert_cache_seed_selected(NULL, ids, 6) == 0,
              "stream: NULL table rejected");
    }

    /* --- Fase 6 step 3: device-local expert pool ------------------------- */
    {
        const uint32_t n_tokens = 2u, n_expert = 3u, n_total = 8u;
        const uint32_t in_dim = 64u, mid_dim = 32u, out_dim = 32u;
        const uint32_t in_b = in_dim / 32u;
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * in_b * 34u;
        const uint64_t gate_row_bytes = (uint64_t)in_b * 34u;
        const uint64_t down_expert_bytes = (uint64_t)out_dim * 34u;
        const uint64_t down_row_bytes = 34u;
        const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
        const float clamp = 5.0f;
        int32_t ids[6];
        for (uint32_t i = 0; i < 6; i++) ids[i] = (int32_t)((i * 5u + 1u) % n_total);

        const ds4_gpu_stream_expert_table ptable = {
            .model_map = g_model,
            .model_size = MODEL_PADDED,
            .layer = 0,
            .n_total_expert = n_total,
            .gate_offset = OFF_MOE_GATE,
            .up_offset = OFF_MOE_UP,
            .down_offset = OFF_MOE_DOWN,
            .gate_expert_bytes = gate_expert_bytes,
            .down_expert_bytes = down_expert_bytes,
        };

        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * in_dim * sizeof(float));
        ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(pair_count * sizeof(int32_t));
        ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
        ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(pair_count * mid_dim * sizeof(float));
        ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(pair_count * out_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
        CHECK(x && selected && weights && gate && up && mid && down && out,
              "pool: alloc");
        if (x && selected && weights && gate && up && mid && down && out) {
            float *xv = (float *)malloc((uint64_t)n_tokens * in_dim * sizeof(float));
            int32_t *sv = (int32_t *)malloc(pair_count * sizeof(int32_t));
            float *wv = (float *)malloc(pair_count * sizeof(float));
            float *gg = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *gu = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *gm = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *go = (float *)malloc((uint64_t)n_tokens * out_dim * sizeof(float));
            float *g_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *u_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *m_ref = (float *)malloc(pair_count * mid_dim * sizeof(float));
            float *o_ref = (float *)malloc((uint64_t)n_tokens * out_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tokens * in_dim; i++) {
                xv[i] = 0.2f * cosf((float)i * 0.17f) + 0.3f * (float)(i % 7) - 0.6f;
            }
            for (uint32_t i = 0; i < pair_count; i++) {
                sv[i] = (int32_t)((i * 5u + 1u) % n_total);
                wv[i] = 0.3f + 0.1f * (float)(i % 5);
            }
            ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tokens * in_dim * sizeof(float));
            ds4_gpu_tensor_write(selected, 0, sv, pair_count * sizeof(int32_t));
            ds4_gpu_tensor_write(weights, 0, wv, pair_count * sizeof(float));

            cpu_routed_moe(o_ref, g_ref, u_ref, m_ref, g_model, OFF_MOE_GATE,
                           OFF_MOE_UP, OFF_MOE_DOWN, gate_expert_bytes,
                           gate_row_bytes, down_expert_bytes, down_row_bytes,
                           in_dim, mid_dim, out_dim, sv, wv, xv,
                           n_tokens, n_expert, clamp);

            /* Streaming mode forces the MoE to serve weights from the pool. */
            ds4_gpu_set_ssd_streaming(1);
            CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(
                      &ptable, ids, 6) != 0,
                  "pool: begin_selected_load seeds the layer pool");
            CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(
                      &ptable, ids, 6) != 0,
                  "pool: re-seed (cache hit) keeps success");

            /* Drop the expert blobs from the staged windows so the MoE cannot
             * fall back to the model range: only the pool can serve them. */
            uint64_t spans_off[1] = { 0u };
            uint64_t spans_sz[1] = { OFF_MOE_GATE };
            CHECK(ds4_gpu_set_model_map_spans(g_model, MODEL_PADDED, spans_off,
                                              spans_sz, 1, spans_sz[0]) != 0,
                  "pool: re-map without expert windows");
            CHECK(ds4_gpu_synchronize() != 0, "pool: sync after re-map");

            CHECK(ds4_gpu_routed_moe_batch_tensor(
                      out, gate, up, mid, down, g_model, MODEL_PADDED,
                      OFF_MOE_GATE, OFF_MOE_UP, OFF_MOE_DOWN, 8u, 8u,
                      gate_expert_bytes, gate_row_bytes, down_expert_bytes,
                      down_row_bytes, in_dim, mid_dim, out_dim, selected,
                      weights, n_total, n_expert, clamp, x, 0u, n_tokens,
                      NULL, true) != 0,
                  "pool: routed_moe batch from pool");
            CHECK(ds4_gpu_synchronize() != 0, "pool: sync batch");
            ds4_gpu_tensor_read(gate, 0, gg, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(up, 0, gu, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(mid, 0, gm, pair_count * mid_dim * sizeof(float));
            ds4_gpu_tensor_read(out, 0, go, (uint64_t)n_tokens * out_dim * sizeof(float));
            check_close(gg, g_ref, pair_count * mid_dim, 0.01f,
                        "pool: gate matches CPU");
            check_close(gu, u_ref, pair_count * mid_dim, 0.01f,
                        "pool: up matches CPU");
            check_close(gm, m_ref, pair_count * mid_dim, 0.01f,
                        "pool: mid matches CPU");
            check_close(go, o_ref, n_tokens * out_dim, 0.01f,
                        "pool: out matches CPU");

            /* Restore the full model and the non-streaming MoE path. */
            CHECK(ds4_gpu_set_model_map(g_model, MODEL_PADDED) != 0,
                  "pool: restore full model map");
            ds4_gpu_set_ssd_streaming(0);
            ds4_gpu_stream_expert_cache_release_resident();
            CHECK(ds4_gpu_synchronize() != 0, "pool: sync restore");

            free(xv); free(sv); free(wv); free(gg); free(gu); free(gm); free(go);
            free(g_ref); free(u_ref); free(m_ref); free(o_ref);
        }
        if (x) ds4_gpu_tensor_free(x);
        if (selected) ds4_gpu_tensor_free(selected);
        if (weights) ds4_gpu_tensor_free(weights);
        if (gate) ds4_gpu_tensor_free(gate);
        if (up) ds4_gpu_tensor_free(up);
        if (mid) ds4_gpu_tensor_free(mid);
        if (down) ds4_gpu_tensor_free(down);
        if (out) ds4_gpu_tensor_free(out);
    }

    if (t) ds4_gpu_tensor_free(t);
    ds4_gpu_cleanup();
    if (g_model) free(g_model);

    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
