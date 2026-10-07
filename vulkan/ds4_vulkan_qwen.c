/* vulkan/ds4_vulkan_qwen.c -- Qwen3.8-Flash-Next (qwen4exp) Vulkan backend.
 *
 * Per-family TU: the Qwen4 single-session ds4_gpu_qwen4_* entry points.  It
 * records dispatches through the shared core surface exposed by
 * ds4_vulkan_internal.h (bindings, push-constant mirror, pipe table, launcher
 * and bind/scratch helpers); everything else stays private to ds4_vulkan.c.
 *
 * P1 milestone 1 is single-session (CUDA parity): decode T=1, prefill, MTP.
 * The batch surface (_rows/_rows2/_grouped/batch_mm_q8) is Metal-only and out
 * of scope.  Numeric target: f32 activations, deterministic groupshared
 * reductions, parity with tests/test_qwen4_kernels.c.
 *
 * This file is compiled as C++17 with C linkage (ds4_gpu.h wraps the API in
 * extern "C"), matching ds4_vulkan.c. */

#include <vulkan/vulkan.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_vulkan_internal.h"

/* GGUF type ids (ds4.c): 0 F32, 1 F16, 2 Q4_0, 8 Q8_0, 12 Q4_K, 30 BF16. */

/* Qwen rope table (ds4_gpu_qwen4_set_rope).  Kept host-side and uploaded
 * lazily to a small tensor that attn_prep / block_key bind.  When no table is
 * set the shaders derive frequencies from `base` (plain rope). */
static float g_qwen4_rope_freq[32];
static uint32_t g_qwen4_rope_pairs = 0;
static float g_qwen4_rope_scale = 1.0f;
static int g_qwen4_rope_set = 0;
/* One table per logical tier: attention runs on whichever tier owns the layer,
 * so a single global tensor would be bound on the wrong device in multi-GPU. */
static ds4_gpu_tensor *g_qwen4_rope_tensor[DS4_MAX_GPUS] = {0};

void ds4_gpu_qwen4_set_rope(const float *freq, uint32_t n_pairs, float mscale) {
    g_qwen4_rope_set = freq != NULL;
    g_qwen4_rope_scale = freq ? mscale : 1.0f;
    memset(g_qwen4_rope_freq, 0, sizeof(g_qwen4_rope_freq));
    g_qwen4_rope_pairs = freq ? (n_pairs < 32u ? n_pairs : 32u) : 0u;
    if (freq) memcpy(g_qwen4_rope_freq, freq, g_qwen4_rope_pairs * sizeof(float));
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        if (g_qwen4_rope_tensor[t]) {
            ds4_gpu_tensor_free(g_qwen4_rope_tensor[t]);
            g_qwen4_rope_tensor[t] = NULL;
        }
    }
}

/* The Vulkan backend implements every fused decode entry point (q8_pair,
 * gdn_front, multi_gemv, hc_combine_norm), matching the CUDA reference. */
int ds4_gpu_qwen4_decode_fusions_enabled(void) {
    const char *env = getenv("DS4_QWEN4_DECODE_FUSIONS");
    if (env && env[0] && strcmp(env, "0") == 0) return 0;
    return 1;
}

static const ds4_gpu_tensor *qwen4_rope_table(void) {
    if (!g_qwen4_rope_set) return NULL;
    int tier = ds4_vulkan_current_tier();
    if (tier < 0 || tier >= DS4_MAX_GPUS) tier = 0;
    if (!g_qwen4_rope_tensor[tier]) {
        g_qwen4_rope_tensor[tier] = ds4_gpu_tensor_alloc(32u * sizeof(float));
        if (g_qwen4_rope_tensor[tier]) {
            (void)ds4_gpu_tensor_write(g_qwen4_rope_tensor[tier], 0,
                                       g_qwen4_rope_freq,
                                       32u * sizeof(float));
        }
    }
    return g_qwen4_rope_tensor[tier];
}

/* Model range check: in-window and inside the registered map. */
static int qwen4_range_ok(uint64_t off, uint64_t bytes, uint64_t model_size) {
    if (off > model_size || bytes > model_size - off) return 0;
    return vulkan_model_range_ok(off, bytes);
}

/* Exact byte length of one weight row for a dense type, or 0 when the type is
 * not handled by the dense path. */
static uint64_t qwen4_dense_row_bytes(uint32_t type, uint32_t in_dim) {
    switch (type) {
    case 0u:  /* F32 */
        return (uint64_t)in_dim * 4u;
    case 1u:  /* F16 */
    case 30u: /* BF16 */
        return (uint64_t)in_dim * 2u;
    case 8u:  /* Q8_0: 34-byte block / 32 values */
        if (in_dim % 32u != 0u) return 0;
        return ((uint64_t)in_dim / 32u) * 34u;
    case 2u:  /* Q4_0: 18-byte block / 32 values */
        if (in_dim % 32u != 0u) return 0;
        return ((uint64_t)in_dim / 32u) * 18u;
    case 12u: /* Q4_K: 144-byte block / 256 values */
        if (in_dim % 256u != 0u) return 0;
        return ((uint64_t)in_dim / 256u) * 144u;
    default:
        return 0;
    }
}

/* Pipeline + push-constant block count for a dense type. */
static VkPipeline qwen4_dense_pipe(uint32_t type, uint32_t in_dim,
                                   uint32_t *blocks) {
    switch (type) {
    case 0u:
        *blocks = (in_dim + 31u) / 32u;
        return g_pipes[DS4_PIPE_MATMUL_F32];
    case 1u:
        *blocks = (in_dim + 31u) / 32u;
        return g_pipes[DS4_PIPE_MATMUL_F16];
    case 30u:
        *blocks = (in_dim + 31u) / 32u;
        return g_pipes[DS4_PIPE_MATMUL_BF16];
    case 8u:
        *blocks = in_dim / 32u;
        return g_pipes[DS4_PIPE_MATMUL_Q8_0_F32];
    case 2u:
        *blocks = in_dim / 32u;
        return g_pipes[DS4_PIPE_MATMUL_Q4_0];
    case 12u:
        *blocks = in_dim / 256u;
        return g_pipes[DS4_PIPE_MATMUL_Q4K];
    default:
        *blocks = 0u;
        return VK_NULL_HANDLE;
    }
}

/* out[tok][row] = dot(w[row], x[tok]) with raw f32 activations.  `wbind` is
 * the already-resolved weight bind (model window or GPU tensor). */
static int qwen4_dense_bind(ds4_gpu_tensor *out, struct ds4_vk_bind wbind,
                            uint32_t type, uint32_t n_tokens, uint32_t in_dim,
                            uint32_t out_rows, const ds4_gpu_tensor *x) {
    if (!out || !x || wbind.buffer == VK_NULL_HANDLE) return 0;
    if (n_tokens == 0u || in_dim == 0u || out_rows == 0u) return 0;
    if (x->bytes < (uint64_t)n_tokens * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_rows * sizeof(float)) {
        return 0;
    }
    uint32_t blocks = 0u;
    VkPipeline pipe = qwen4_dense_pipe(type, in_dim, &blocks);
    if (pipe == VK_NULL_HANDLE) return 0;

    struct ds4_vk_params p = {};
    p.in_dim = in_dim;
    p.out_dim = out_rows;
    p.rows = n_tokens;
    p.blocks = blocks;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    binds[nb++] = wbind;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    return vulkan_dispatch(pipe, &p, sizeof(p), binds, nb, out_rows, n_tokens,
                           1u);
}

int ds4_gpu_qwen4_dense_mm_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
                                  const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint32_t weight_type,
                                  uint32_t n_tokens, uint32_t in_dim,
                                  uint32_t out_rows) {
    if (!out || !x || !model_map) return 0;
    if (n_tokens == 0u || in_dim == 0u || out_rows == 0u) return 0;
    const uint64_t row = qwen4_dense_row_bytes(weight_type, in_dim);
    if (row == 0u || out_rows > UINT64_MAX / row) return 0;
    const uint64_t weight_bytes = (uint64_t)out_rows * row;
    if (weight_offset > model_size ||
        weight_bytes > model_size - weight_offset) {
        return 0;
    }
    if (!vulkan_model_range_ok(weight_offset, weight_bytes)) return 0;
    const struct ds4_vk_bind wbind =
        vulkan_bind_model(DS4_VK_BINDING_W, weight_offset, weight_bytes);
    return qwen4_dense_bind(out, wbind, weight_type, n_tokens, in_dim, out_rows,
                            x);
}

int ds4_gpu_qwen4_matmul_q8_0_tensor(ds4_gpu_tensor *out,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t weight_offset, uint64_t in_dim,
                                     uint64_t out_dim,
                                     const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    return ds4_gpu_qwen4_dense_mm_tensor(
        out, x, model_map, model_size, weight_offset, 8u, (uint32_t)n_tok,
        (uint32_t)in_dim, (uint32_t)out_dim);
}

/* Single-token Q8_0 matvec whose weight rows live in a GPU tensor (the
 * gathered MTP draft head); same math as the model-range path. */
int ds4_gpu_qwen4_matmul_q8_0_weights_tensor(ds4_gpu_tensor *out,
                                             const ds4_gpu_tensor *w,
                                             uint32_t in_dim, uint32_t out_dim,
                                             const ds4_gpu_tensor *x) {
    if (!out || !w || !x) return 0;
    if (in_dim == 0u || out_dim == 0u || in_dim % 32u != 0u) return 0;
    const uint64_t row = ((uint64_t)in_dim / 32u) * 34u;
    if (out_dim > UINT64_MAX / row) return 0;
    if (w->bytes < (uint64_t)out_dim * row) return 0;
    const struct ds4_vk_bind wbind =
        vulkan_bind_tensor(DS4_VK_BINDING_W, w);
    return qwen4_dense_bind(out, wbind, 8u, 1u, in_dim, out_dim, x);
}

/* Up to four projections of x in one entry point; weight types 0 f32, 1 f16,
 * 2 q4_0, 8 q8_0, 12 q4_K, 30 bf16. */
int ds4_gpu_qwen4_multi_gemv_tensor(const ds4_gpu_tensor *x, uint32_t n_tokens,
                                    uint32_t in_dim, uint32_t n_out,
                                    ds4_gpu_tensor *const *outs,
                                    const void *model_map, uint64_t model_size,
                                    const uint64_t *offsets,
                                    const uint32_t *types,
                                    const uint32_t *out_rows) {
    if (!x || !outs || !offsets || !types || !out_rows) return 0;
    if (n_out == 0u || n_out > 4u) return 0;
    for (uint32_t i = 0; i < n_out; i++) {
        if (!ds4_gpu_qwen4_dense_mm_tensor(outs[i], x, model_map, model_size,
                                           offsets[i], types[i], n_tokens,
                                           in_dim, out_rows[i])) {
            return 0;
        }
    }
    return 1;
}

/* Concatenated Q8_0 projection: two matvecs sharing x.  CUDA parity requires
 * even output widths. */
int ds4_gpu_qwen4_q8_pair_tensor(ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
                                 const void *model_map, uint64_t model_size,
                                 uint64_t weight0_offset,
                                 uint64_t weight1_offset, uint64_t in_dim,
                                 uint64_t out0_dim, uint64_t out1_dim,
                                 const ds4_gpu_tensor *x, uint64_t n_tok) {
    if ((out0_dim & 1u) != 0u || (out1_dim & 1u) != 0u) return 0;
    if (!ds4_gpu_qwen4_matmul_q8_0_tensor(out0, model_map, model_size,
                                          weight0_offset, in_dim, out0_dim, x,
                                          n_tok)) {
        return 0;
    }
    return ds4_gpu_qwen4_matmul_q8_0_tensor(out1, model_map, model_size,
                                            weight1_offset, in_dim, out1_dim, x,
                                            n_tok);
}

/* --- P2: gated delta-net (GDN) -------------------------------------------
 *
 * Single-session parity with the CUDA reference (ds4_qwen4_cuda.cuh):
 * conv_stream, gdn_prep, gdn_scan, gdn_out, and gdn_front (conv + alpha/beta
 * dense projections + prep).  All activations are f32.  The scan is
 * sequential over tokens (parallel over state rows) and its groupshared
 * reduction is independent of the chunk length, so split/batched prefill is
 * byte-exact. */

/* Depthwise causal conv (K in 2..4) over `history`, optionally SiLU, writing
 * back into x and, at token snap_t/snap2_t, into the snapshot tensors. */
static int qwen4_conv_dispatch(ds4_gpu_tensor *x, ds4_gpu_tensor *history,
                               uint64_t model_size, uint64_t conv_off,
                               uint32_t T, uint32_t C, uint32_t K,
                               bool activate, ds4_gpu_tensor *snap,
                               uint32_t snap_t, ds4_gpu_tensor *snap2,
                               uint32_t snap2_t) {
    if (!x || !history) return 0;
    if (T == 0u || C == 0u || K < 2u || K > 4u) return 0;
    if (x->bytes < (uint64_t)T * C * sizeof(float) ||
        history->bytes < (uint64_t)(K - 1u) * C * sizeof(float)) {
        return 0;
    }
    const uint64_t wbytes = (uint64_t)C * K * sizeof(float);
    if (conv_off > model_size || wbytes > model_size - conv_off) return 0;
    if (!vulkan_model_range_ok(conv_off, wbytes)) return 0;
    const uint64_t hbytes = (uint64_t)(K - 1u) * C * sizeof(float);
    if (snap && snap->bytes < hbytes) return 0;
    if (snap2 && snap2->bytes < hbytes) return 0;

    struct ds4_vk_params p = {};
    p.n = C;
    p.rows = T;
    p.in_dim = K;
    p.flags = activate ? 1u : 0u;
    p.index = snap ? snap_t : 0xffffffffu;
    p.aux = snap2 ? snap2_t : 0xffffffffu;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, x);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, history);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, conv_off, wbytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3,
                                     snap ? snap : history);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT4,
                                     snap2 ? snap2 : history);
    return vulkan_dispatch(g_pipes[DS4_PIPE_CONV_STREAM], &p, sizeof(p), binds,
                           nb, (C + 255u) / 256u, 1u, 1u);
}

int ds4_gpu_qwen4_conv_stream_tensor(ds4_gpu_tensor *x, ds4_gpu_tensor *state,
                                     const void *model_map, uint64_t model_size,
                                     uint64_t weight_offset, uint32_t n_tokens,
                                     uint32_t n_channels, uint32_t conv_kernel,
                                     bool apply_silu) {
    if (!model_map) return 0;
    return qwen4_conv_dispatch(x, state, model_size, weight_offset, n_tokens,
                               n_channels, conv_kernel, apply_silu, NULL, 0u,
                               NULL, 0u);
}

int ds4_gpu_qwen4_gdn_prep_tensor(ds4_gpu_tensor *qkv, ds4_gpu_tensor *a,
                                  ds4_gpu_tensor *b, const void *model_map,
                                  uint64_t model_size, uint64_t ssm_a_offset,
                                  uint64_t dt_bias_offset, uint32_t n_tokens,
                                  uint32_t n_k_head, uint32_t n_v_head,
                                  uint32_t head_dim) {
    if (!qkv || !a || !b || !model_map) return 0;
    if (n_tokens == 0u || n_k_head == 0u || n_v_head == 0u) return 0;
    if (head_dim < 32u || head_dim > 128u || (head_dim & 31u) != 0u) return 0;
    const uint64_t qkv_bytes = (uint64_t)n_tokens *
                               (2u * n_k_head + n_v_head) * head_dim *
                               sizeof(float);
    const uint64_t ab_bytes = (uint64_t)n_tokens * n_v_head * sizeof(float);
    if (qkv->bytes < qkv_bytes || a->bytes < ab_bytes || b->bytes < ab_bytes) {
        return 0;
    }
    const uint64_t wbytes = (uint64_t)n_v_head * sizeof(float);
    if (ssm_a_offset > model_size || wbytes > model_size - ssm_a_offset) {
        return 0;
    }
    if (dt_bias_offset > model_size || wbytes > model_size - dt_bias_offset) {
        return 0;
    }
    if (!vulkan_model_range_ok(ssm_a_offset, wbytes) ||
        !vulkan_model_range_ok(dt_bias_offset, wbytes)) {
        return 0;
    }

    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_tokens;
    p.in_dim = n_k_head;
    p.out_dim = n_v_head;
    p.eps = 1e-6f;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, qkv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, a);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, b);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, ssm_a_offset, wbytes);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_C, dt_bias_offset, wbytes);
    return vulkan_dispatch(g_pipes[DS4_PIPE_GDN_PREP], &p, sizeof(p), binds, nb,
                           n_k_head, n_tokens, 1u);
}

int ds4_gpu_qwen4_gdn_scan_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *state,
                                  const ds4_gpu_tensor *qkv,
                                  const ds4_gpu_tensor *a,
                                  const ds4_gpu_tensor *b, uint32_t n_tokens,
                                  uint32_t n_k_head, uint32_t n_v_head,
                                  uint32_t head_dim,
                                  ds4_gpu_tensor *snap_state, uint32_t snap_tok,
                                  ds4_gpu_tensor *snap2_state,
                                  uint32_t snap2_tok) {
    if (!out || !state || !qkv || !a || !b) return 0;
    if (n_tokens == 0u || n_k_head == 0u || n_v_head == 0u) return 0;
    if (n_v_head % n_k_head != 0u) return 0;
    if (head_dim < 32u || head_dim > 128u || (head_dim & 31u) != 0u) return 0;
    const uint32_t rows = n_tokens > 8u ? 4u : 1u;
    if (head_dim % (4u * rows) != 0u) return 0;
    const uint64_t state_bytes = (uint64_t)n_v_head * head_dim * head_dim * 4u;
    const uint64_t out_bytes = (uint64_t)n_tokens * n_v_head * head_dim * 4u;
    const uint64_t qkv_bytes = (uint64_t)n_tokens *
                               (2u * n_k_head + n_v_head) * head_dim * 4u;
    const uint64_t ab_bytes = (uint64_t)n_tokens * n_v_head * 4u;
    if (out->bytes < out_bytes || state->bytes < state_bytes ||
        qkv->bytes < qkv_bytes || a->bytes < ab_bytes || b->bytes < ab_bytes) {
        return 0;
    }
    if (snap_state && snap_state->bytes < state_bytes) return 0;
    if (snap2_state && snap2_state->bytes < state_bytes) return 0;

    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_tokens;
    p.in_dim = n_k_head;
    p.out_dim = n_v_head;
    p.index = snap_state ? snap_tok : 0xffffffffu;
    p.aux = snap2_state ? snap2_tok : 0xffffffffu;
    p.rsvd2 = rows;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, state);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, qkv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, a);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, b);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3,
                                     snap_state ? snap_state : state);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT4,
                                     snap2_state ? snap2_state : state);
    return vulkan_dispatch(g_pipes[DS4_PIPE_GDN_SCAN], &p, sizeof(p), binds, nb,
                           head_dim / (4u * rows), n_v_head, 1u);
}

int ds4_gpu_qwen4_gdn_out_tensor(ds4_gpu_tensor *o, const ds4_gpu_tensor *z,
                                 const void *model_map, uint64_t model_size,
                                 uint64_t weight_offset, uint32_t n_tokens,
                                 uint32_t n_head, uint32_t head_dim, float eps) {
    if (!o || !z || !model_map) return 0;
    if (n_tokens == 0u || n_head == 0u) return 0;
    if (head_dim < 32u || head_dim > 128u || (head_dim & 31u) != 0u) return 0;
    const uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    if (o->bytes < n * 4u || z->bytes < n * 4u) return 0;
    const uint64_t wbytes = (uint64_t)head_dim * sizeof(float);
    if (weight_offset > model_size || wbytes > model_size - weight_offset) {
        return 0;
    }
    if (!vulkan_model_range_ok(weight_offset, wbytes)) return 0;

    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_tokens;
    p.in_dim = n_head;
    p.eps = eps;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, o);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, z);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, weight_offset, wbytes);
    return vulkan_dispatch(g_pipes[DS4_PIPE_GDN_OUT], &p, sizeof(p), binds, nb,
                           n_head, n_tokens, 1u);
}

int ds4_gpu_qwen4_gdn_front_tensor(
    ds4_gpu_tensor *qkv, ds4_gpu_tensor *state, const ds4_gpu_tensor *mixed,
    ds4_gpu_tensor *ga, ds4_gpu_tensor *gb, const void *model_map,
    uint64_t model_size, uint64_t conv_offset, uint64_t alpha_offset,
    uint64_t beta_offset, uint64_t ssm_a_offset, uint64_t dt_bias_offset,
    uint32_t weight_type, uint32_t n_tokens, uint32_t n_k_head,
    uint32_t n_v_head, uint32_t head_dim, uint32_t conv_kernel,
    uint32_t in_dim, ds4_gpu_tensor *snap_state, uint32_t snap_tok,
    ds4_gpu_tensor *snap2_state, uint32_t snap2_tok) {
    if (!qkv || !state || !mixed || !ga || !gb || !model_map) return 0;
    if (n_tokens == 0u || n_k_head == 0u || n_v_head == 0u || in_dim == 0u) {
        return 0;
    }
    if (head_dim < 32u || head_dim > 128u || (head_dim & 31u) != 0u) return 0;
    if (conv_kernel < 2u || conv_kernel > 4u) return 0;
    const uint64_t C = (uint64_t)(2u * n_k_head + n_v_head) * head_dim;
    if (C > UINT32_MAX) return 0;

    if (!ds4_gpu_qwen4_dense_mm_tensor(ga, mixed, model_map, model_size,
                                       alpha_offset, weight_type, n_tokens,
                                       in_dim, n_v_head)) {
        return 0;
    }
    if (!ds4_gpu_qwen4_dense_mm_tensor(gb, mixed, model_map, model_size,
                                       beta_offset, weight_type, n_tokens,
                                       in_dim, n_v_head)) {
        return 0;
    }
    if (!qwen4_conv_dispatch(qkv, state, model_size, conv_offset, n_tokens,
                             (uint32_t)C, conv_kernel, true, snap_state, snap_tok,
                             snap2_state, snap2_tok)) {
        return 0;
    }
    return ds4_gpu_qwen4_gdn_prep_tensor(qkv, ga, gb, model_map, model_size,
                                         ssm_a_offset, dt_bias_offset, n_tokens,
                                         n_k_head, n_v_head, head_dim);
}

/* --- P3: QSA indexer ------------------------------------------------------
 *
 * score[t][b] = sum_h relu(q_h . key_b) over the indexer heads, with the
 * incomplete/future blocks scored -3e38; the optional tile_max scratch holds
 * per-8-block maxima as uint keys; the exact 4-pass radix select and the token
 * expansion match the CUDA reference (ds4_qwen4_cuda.cuh). */

int ds4_gpu_qwen4_idx_score_tensor(ds4_gpu_tensor *score,
                                   ds4_gpu_tensor *tile_max,
                                   const ds4_gpu_tensor *iq,
                                   const ds4_gpu_tensor *block_key,
                                   uint32_t n_tokens, uint32_t n_blocks,
                                   uint32_t n_idx_head, uint32_t idx_dim,
                                   uint32_t pos0, uint32_t ratio) {
    if (!score || !iq || !block_key) return 0;
    if (n_tokens == 0u || n_blocks == 0u || n_idx_head == 0u ||
        idx_dim == 0u || ratio == 0u) {
        return 0;
    }
    if (score->bytes < (uint64_t)n_tokens * n_blocks * 4u ||
        iq->bytes < (uint64_t)n_tokens * n_idx_head * idx_dim * 4u ||
        block_key->bytes < (uint64_t)n_blocks * idx_dim * 2u) {
        return 0;
    }
    const uint32_t nt = (n_blocks + 7u) / 8u;
    if (tile_max && tile_max->bytes < (uint64_t)n_tokens * nt * 4u) return 0;

    struct ds4_vk_params p = {};
    p.n = n_blocks;
    p.rows = n_tokens;
    p.in_dim = n_idx_head;
    p.out_dim = idx_dim;
    p.pos0 = pos0;
    p.ratio = ratio;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, score);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, iq);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, block_key);
    if (!vulkan_dispatch(g_pipes[DS4_PIPE_IDX_SCORE], &p, sizeof(p), binds, nb,
                         n_blocks, n_tokens, 1u)) {
        return 0;
    }
    if (tile_max) {
        struct ds4_vk_params pt = {};
        pt.n = n_blocks;
        pt.rows = n_tokens;
        pt.blocks = nt;
        struct ds4_vk_bind tb[DS4_VK_MAX_BINDS];
        uint32_t tnb = 0;
        tb[tnb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, tile_max);
        tb[tnb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, score);
        if (!vulkan_dispatch(g_pipes[DS4_PIPE_IDX_TILE_MAX], &pt, sizeof(pt),
                             tb, tnb, (nt + 255u) / 256u, n_tokens, 1u)) {
            return 0;
        }
    }
    return 1;
}

int ds4_gpu_qwen4_idx_select_tensor(ds4_gpu_tensor *sel,
                                    const ds4_gpu_tensor *score,
                                    const ds4_gpu_tensor *tile_max,
                                    uint32_t n_blocks, uint32_t n_tokens,
                                    uint32_t top_k) {
    (void)tile_max; /* CUDA/Vulkan select is always the exact full-row radix */
    if (!sel || !score) return 0;
    if (n_tokens == 0u || top_k == 0u || top_k > n_blocks) return 0;
    if (sel->bytes < (uint64_t)n_tokens * top_k * 4u ||
        score->bytes < (uint64_t)n_tokens * n_blocks * 4u) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = n_blocks;
    p.out_dim = top_k;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, sel);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, score);
    return vulkan_dispatch(g_pipes[DS4_PIPE_IDX_SELECT], &p, sizeof(p), binds,
                           nb, n_tokens, 1u, 1u);
}

int ds4_gpu_qwen4_idx_expand_tensor(ds4_gpu_tensor *sel_tokens,
                                    ds4_gpu_tensor *n_sel,
                                    const ds4_gpu_tensor *sel_blocks,
                                    uint32_t n_tokens, uint32_t n_sel_blocks,
                                    uint32_t ratio, uint32_t pos0,
                                    uint32_t sel_stride) {
    if (!sel_tokens || !n_sel || !sel_blocks) return 0;
    if (n_tokens == 0u || n_sel_blocks == 0u || ratio == 0u) return 0;
    if ((uint64_t)n_sel_blocks * ratio + ratio - 1u > sel_stride) return 0;
    if (sel_tokens->bytes < (uint64_t)n_tokens * sel_stride * 4u ||
        n_sel->bytes < (uint64_t)n_tokens * 4u ||
        sel_blocks->bytes < (uint64_t)n_tokens * n_sel_blocks * 4u) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = n_sel_blocks;
    p.in_dim = sel_stride;
    p.ratio = ratio;
    p.pos0 = pos0;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, sel_tokens);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, n_sel);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, sel_blocks);
    return vulkan_dispatch(g_pipes[DS4_PIPE_IDX_EXPAND], &p, sizeof(p), binds,
                           nb, n_tokens, 1u, 1u);
}

/* --- P3: full attention --------------------------------------------------
 *
 * attn_prep normalises q/k (rope + per-head gamma), copies the attention gate
 * and fills the f16 k/v caches, the indexer q, the raw indexer key; block_key
 * builds the per-block indexer keys; attn_decode is the scalar online-softmax
 * attention (with optional split-KV partials) and attn_merge reconnects them.
 * The batch (_rows) surface is Metal-only. */

int ds4_gpu_qwen4_attn_prep_tensor(
    ds4_gpu_tensor *q_out, ds4_gpu_tensor *gate_out, ds4_gpu_tensor *k_cache,
    ds4_gpu_tensor *v_cache, ds4_gpu_tensor *iq_out, ds4_gpu_tensor *ik_cache,
    const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kproj,
    const ds4_gpu_tensor *vproj, const ds4_gpu_tensor *iq,
    const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3, const void *model_map,
    uint64_t model_size, uint64_t g_q_offset, uint64_t g_k_offset,
    uint64_t g_iq_offset, uint32_t n_tokens, uint32_t n_head,
    uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot, uint32_t n_idx_head,
    uint32_t idx_dim, uint32_t pos0, uint32_t cache_cap, float rope_base,
    float eps) {
    if (!q_out || !gate_out || !k_cache || !v_cache || !iq_out || !ik_cache ||
        !qg || !kproj || !vproj || !iq || !ik || !pos3 || !model_map) {
        return 0;
    }
    if (n_tokens == 0u || n_head == 0u || n_head_kv == 0u || n_head % n_head_kv ||
        n_idx_head == 0u) {
        return 0;
    }
    if (head_dim < 32u || head_dim > 256u || (head_dim & 31u) != 0u) return 0;
    if (idx_dim < 32u || idx_dim > 128u || (idx_dim & 31u) != 0u) return 0;
    if (n_rot > 64u || n_rot > head_dim || n_rot > idx_dim || (n_rot & 1u) != 0u) {
        return 0;
    }
    if ((uint64_t)pos0 + n_tokens > cache_cap) return 0;
    const uint64_t qb = (uint64_t)n_tokens * n_head * head_dim * 4u;
    const uint64_t kb = (uint64_t)n_tokens * n_head_kv * head_dim * 4u;
    const uint64_t ib = (uint64_t)n_tokens * n_idx_head * idx_dim * 4u;
    if (q_out->bytes < qb || gate_out->bytes < qb || qg->bytes < qb * 2u ||
        kproj->bytes < kb || vproj->bytes < kb || iq->bytes < ib ||
        iq_out->bytes < ib || ik->bytes < (uint64_t)n_tokens * idx_dim * 4u ||
        ik_cache->bytes < (uint64_t)cache_cap * idx_dim * 4u ||
        k_cache->bytes < (uint64_t)cache_cap * n_head_kv * head_dim * 2u ||
        v_cache->bytes < (uint64_t)cache_cap * n_head_kv * head_dim * 2u ||
        pos3->bytes < (uint64_t)cache_cap * 16u) {
        return 0;
    }
    const uint64_t gqb = (uint64_t)head_dim * 4u, gib = (uint64_t)idx_dim * 4u;
    if (!qwen4_range_ok(g_q_offset, gqb, model_size) ||
        !qwen4_range_ok(g_k_offset, gqb, model_size) ||
        !qwen4_range_ok(g_iq_offset, gib, model_size)) {
        return 0;
    }
    const ds4_gpu_tensor *rope = qwen4_rope_table();

    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_tokens;
    p.in_dim = n_head;
    p.out_dim = n_head_kv;
    p.aux = n_idx_head;
    p.ratio = idx_dim;
    p.pos0 = pos0;
    p.eps = eps;
    p.n_rot = n_rot;
    p.freq_base = rope_base;
    p.freq_scale = g_qwen4_rope_scale;
    p.flags = g_qwen4_rope_set ? 2u : 0u;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, q_out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, gate_out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, k_cache);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT4, v_cache);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT5, iq_out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_TBL, ik_cache);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, qg);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, kproj);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, vproj);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_W, iq);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_X0, ik);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_X1, pos3);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_X2, g_q_offset, gqb);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_X3, g_k_offset, gqb);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_X4, g_iq_offset, gib);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_X5, rope ? rope : pos3);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_ATTN_PREP], &p, sizeof(p),
                           binds, nb, n_head + n_head_kv + n_idx_head + 1u,
                           n_tokens, 1u);
}

int ds4_gpu_qwen4_idx_block_key_tensor(
    ds4_gpu_tensor *block_key, const ds4_gpu_tensor *ik_cache,
    const ds4_gpu_tensor *pos3, const void *model_map, uint64_t model_size,
    uint64_t g_ik_offset, uint32_t block0, uint32_t n_blocks, uint32_t ratio,
    uint32_t idx_dim, uint32_t n_rot, float rope_base, float eps) {
    if (!block_key || !ik_cache || !pos3 || !model_map) return 0;
    if (n_blocks == 0u || ratio == 0u) return 0;
    if (idx_dim < 32u || idx_dim > 128u || (idx_dim & 31u) != 0u) return 0;
    if (n_rot > idx_dim || n_rot > 64u || (n_rot & 1u) != 0u) return 0;
    const uint64_t rows = (uint64_t)block0 + n_blocks;
    if (block_key->bytes < rows * idx_dim * 2u ||
        ik_cache->bytes < rows * ratio * idx_dim * 4u ||
        pos3->bytes < rows * ratio * 16u) {
        return 0;
    }
    const uint64_t wb = (uint64_t)idx_dim * 4u;
    if (!qwen4_range_ok(g_ik_offset, wb, model_size)) return 0;
    const ds4_gpu_tensor *rope = qwen4_rope_table();

    struct ds4_vk_params p = {};
    p.n = idx_dim;
    p.rows = n_blocks;
    p.in_dim = block0;
    p.ratio = ratio;
    p.eps = eps;
    p.n_rot = n_rot;
    p.freq_base = rope_base;
    p.freq_scale = g_qwen4_rope_scale;
    p.flags = g_qwen4_rope_set ? 2u : 0u;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, block_key);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, ik_cache);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, pos3);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_C, g_ik_offset, wb);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_W, rope ? rope : pos3);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_BLOCK_KEY], &p, sizeof(p),
                           binds, nb, n_blocks, 1u, 1u);
}

int ds4_gpu_qwen4_attn_decode_tensor(
    ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate,
    const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
    const ds4_gpu_tensor *sel_tokens, const ds4_gpu_tensor *n_sel,
    ds4_gpu_tensor *part, uint32_t n_tokens, uint32_t n_head,
    uint32_t n_head_kv, uint32_t head_dim, uint32_t pos0, bool use_sel,
    uint32_t sel_stride, float scale) {
    if (!out || !q || !gate || !k_cache || !v_cache) return 0;
    if (n_tokens == 0u || n_head == 0u || n_head_kv == 0u || n_head % n_head_kv) {
        return 0;
    }
    if (head_dim != 32u && head_dim != 128u && head_dim != 256u) return 0;
    const uint64_t n = (uint64_t)n_tokens * n_head * head_dim * 4u;
    const uint64_t cb = ((uint64_t)pos0 + n_tokens) * n_head_kv * head_dim * 2u;
    if (out->bytes < n || q->bytes < n || gate->bytes < n ||
        k_cache->bytes < cb || v_cache->bytes < cb) {
        return 0;
    }
    if (use_sel) {
        if (!sel_stride || !sel_tokens || !n_sel) return 0;
        if (sel_tokens->bytes < (uint64_t)n_tokens * sel_stride * 4u ||
            n_sel->bytes < (uint64_t)n_tokens * 4u) {
            return 0;
        }
    }
    const uint32_t keys = use_sel ? sel_stride : pos0 + n_tokens;
    uint32_t splits = 1u;
    if (part) {
        splits = (keys + 31u) / 32u;
        if (splits > 64u) splits = 64u;
        if (splits == 0u) splits = 1u;
        if (part->bytes <
            (uint64_t)n_tokens * n_head * splits * (head_dim + 2u) * 4u) {
            return 0;
        }
    }

    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_tokens;
    p.in_dim = n_head;
    p.out_dim = n_head_kv;
    p.pos0 = pos0;
    p.ratio = sel_stride;
    p.blocks = splits;
    p.weight = scale;
    p.flags = use_sel ? 1u : 0u;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2,
                                     part ? part : k_cache);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, q);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, gate);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, k_cache);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_W, v_cache);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_X0,
                                     use_sel ? sel_tokens : k_cache);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_X1,
                                     use_sel ? n_sel : k_cache);
    if (!vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_ATTN_DECODE], &p, sizeof(p),
                         binds, nb, n_head, n_tokens, splits)) {
        return 0;
    }
    if (splits > 1u) {
        struct ds4_vk_params pm = {};
        pm.n = head_dim;
        pm.in_dim = n_head;
        pm.blocks = splits;
        struct ds4_vk_bind mb[DS4_VK_MAX_BINDS];
        uint32_t mnb = 0;
        mb[mnb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
        mb[mnb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, part);
        mb[mnb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, gate);
        if (!vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_ATTN_MERGE], &pm,
                             sizeof(pm), mb, mnb, n_head, n_tokens, 1u)) {
            return 0;
        }
    }
    return 1;
}

uint64_t ds4_gpu_qwen4_attn_part_floats(uint32_t n_tokens, uint32_t n_head,
                                        uint32_t head_dim) {
    return (uint64_t)n_tokens * n_head * 64u * (head_dim + 2u);
}

/* --- P1: MoE router ------------------------------------------------------
 *
 * Softmax over NE expert logits, strict top-NS (ties -> lower id),
 * renormalize, and the shared-expert gate dot.  Distinct from the DeepSeek
 * router (sqrt(softplus) + bias/hash + insertion sort). */

int ds4_gpu_qwen4_router_topk_tensor(
    ds4_gpu_tensor *sel, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits,
    const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size,
    uint64_t gate_offset, uint32_t gate_type, uint32_t gate_in_dim,
    ds4_gpu_tensor *shared_gate, uint32_t n_tokens, uint32_t n_expert,
    uint32_t n_used) {
    if (!sel || !weights || !logits) return 0;
    if (n_tokens == 0u || n_expert == 0u || n_expert > 512u || n_used == 0u ||
        n_used > n_expert || n_used > 32u) {
        return 0;
    }
    if (sel->bytes < (uint64_t)n_tokens * n_used * 4u ||
        weights->bytes < (uint64_t)n_tokens * n_used * 4u ||
        logits->bytes < (uint64_t)n_tokens * n_expert * 4u) {
        return 0;
    }
    uint64_t gate_bytes = 0u;
    struct ds4_vk_bind gbind = vulkan_bind_tensor(DS4_VK_BINDING_B, logits);
    if (gate_in_dim != 0u) {
        if (!model_map || !x || !shared_gate) return 0;
        if (x->bytes < (uint64_t)n_tokens * gate_in_dim * 4u ||
            shared_gate->bytes < (uint64_t)n_tokens * 4u) {
            return 0;
        }
        gate_bytes = qwen4_dense_row_bytes(gate_type, gate_in_dim);
        if (gate_bytes == 0u) return 0;
        if (!qwen4_range_ok(gate_offset, gate_bytes, model_size)) return 0;
        gbind = vulkan_bind_model(DS4_VK_BINDING_B, gate_offset, gate_bytes);
    }

    struct ds4_vk_params p = {};
    p.in_dim = n_expert;
    p.out_dim = n_used;
    p.rows = n_tokens;
    p.index = gate_in_dim;
    p.rsvd3 = gate_type;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, logits);
    binds[nb++] = gbind;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C,
                                     x ? x : logits);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, sel);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, weights);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3,
                                     shared_gate ? shared_gate : sel);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_ROUTER_TOPK], &p, sizeof(p),
                           binds, nb, n_tokens, 1u, 1u);
}

/* Exact byte length of one expert weight row (supports the MoE quant set:
 * f32/f16/bf16/q4_0/q8_0/mxfp4/q2_K/q4_K/iq2_xxs).  Q2_K rows round the
 * logical length up to a whole 256-value block, as the GGUF layout does. */
static uint64_t qwen4_moe_row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
    case 0u:
        return n * 4u;
    case 1u:
    case 30u:
        return n * 2u;
    case 2u:
        return (n % 32u) ? 0u : n / 32u * 18u;
    case 8u:
        return (n % 32u) ? 0u : n / 32u * 34u;
    case 39u:
        return (n % 32u) ? 0u : n / 32u * 17u;
    case 10u:
        return (n % 256u) ? 0u : n / 256u * 84u;
    case 12u:
        return (n % 256u) ? 0u : n / 256u * 144u;
    case 16u:
        return (n % 256u) ? 0u : n / 256u * 66u;
    default:
        return 0u;
    }
}

static uint64_t qwen4_moe_expert_row_bytes(uint32_t type, uint32_t k) {
    if (type == 10u && (k % 32u) != 0u) return 0u;
    return qwen4_moe_row_bytes(type, type == 10u ? ((uint64_t)k + 255u) / 256u * 256u
                                                 : (uint64_t)k);
}

int ds4_gpu_qwen4_moe_mid_tensor(
    ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *selected,
    const void *model_map, uint64_t model_size, uint64_t gate_offset,
    uint64_t up_offset, uint32_t weight_type, uint32_t n_total_expert,
    uint32_t n_tokens, uint32_t n_slots, uint32_t in_dim, uint32_t ff_dim,
    uint64_t shared_gate_offset, uint64_t shared_up_offset,
    uint32_t shared_type) {
    if (!mid || !x || !selected || !model_map) return 0;
    if (n_tokens == 0u || n_total_expert == 0u || n_slots == 0u ||
        in_dim == 0u || ff_dim == 0u) {
        return 0;
    }
    const bool has_shared = (shared_type != UINT32_MAX);
    const uint32_t n_out = n_slots + (has_shared ? 1u : 0u);
    if (mid->bytes < (uint64_t)n_tokens * n_out * ff_dim * 4u ||
        x->bytes < (uint64_t)n_tokens * in_dim * 4u ||
        selected->bytes < (uint64_t)n_tokens * n_slots * 4u) {
        return 0;
    }
    const uint64_t rb = qwen4_moe_expert_row_bytes(weight_type, in_dim);
    if (rb == 0u || ff_dim > UINT64_MAX / rb) return 0;
    const uint64_t expert_bytes = (uint64_t)n_total_expert * rb * ff_dim;
    if (expert_bytes > UINT32_MAX) return 0;
    /* Pool first: when the expert pool serves this layer the expert blobs are
     * not mapped into the model windows, so the range check would fail. */
    struct ds4_vk_bind gbind = {};
    struct ds4_vk_bind ubind = {};
    struct ds4_vk_bind selb = vulkan_bind_tensor(DS4_VK_BINDING_OUT, selected);
    struct ds4_vk_bind tblb = {};
    const int pool = vulkan_qwen_moe_pool_binds(0u, rb * ff_dim, n_tokens,
                                                n_slots, selected, &gbind,
                                                &ubind, &selb, &tblb);
    if (!pool) {
        if (!qwen4_range_ok(gate_offset, expert_bytes, model_size) ||
            !qwen4_range_ok(up_offset, expert_bytes, model_size)) {
            return 0;
        }
        gbind = vulkan_bind_model(DS4_VK_BINDING_B, gate_offset, expert_bytes);
        ubind = vulkan_bind_model(DS4_VK_BINDING_C, up_offset, expert_bytes);
    }
    uint64_t srb = 0u;
    if (has_shared) {
        srb = qwen4_moe_row_bytes(shared_type, in_dim);
        if (srb == 0u) return 0;
        const uint64_t shb = srb * ff_dim;
        if (!qwen4_range_ok(shared_gate_offset, shb, model_size) ||
            !qwen4_range_ok(shared_up_offset, shb, model_size)) {
            return 0;
        }
    }

    struct ds4_vk_params p = {};
    p.n = in_dim;
    p.rows = n_tokens;
    p.in_dim = ff_dim;
    p.out_dim = n_slots;
    p.aux = (uint32_t)expert_bytes;
    p.ratio = (uint32_t)rb;
    p.blocks = has_shared ? shared_type : UINT32_MAX;
    p.index = (uint32_t)srb;
    p.rsvd3 = weight_type;
    p.flags = has_shared ? 2u : 0u;
    if (pool) p.flags |= 1u;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    binds[nb++] = gbind;
    binds[nb++] = ubind;
    binds[nb++] = selb;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT5, mid);
    if (has_shared) {
        binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_X0, shared_gate_offset,
                                        srb * ff_dim);
        binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_X1, shared_up_offset,
                                        srb * ff_dim);
    } else {
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_X0, x);
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_X1, x);
    }
    if (pool) binds[nb++] = tblb;
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_MOE_MID], &p, sizeof(p),
                           binds, nb, ff_dim, n_out, n_tokens);
}

int ds4_gpu_qwen4_moe_down_tensor(
    ds4_gpu_tensor *part, const ds4_gpu_tensor *mid,
    const ds4_gpu_tensor *selected, const void *model_map, uint64_t model_size,
    uint64_t down_offset, uint32_t weight_type, uint32_t n_total_expert,
    uint32_t n_tokens, uint32_t n_slots, uint32_t ff_dim, uint32_t out_dim,
    uint64_t shared_down_offset, uint32_t shared_type) {
    if (!part || !mid || !selected || !model_map) return 0;
    if (n_tokens == 0u || n_total_expert == 0u || n_slots == 0u ||
        ff_dim == 0u || out_dim == 0u) {
        return 0;
    }
    const bool has_shared = (shared_type != UINT32_MAX);
    const uint32_t n_out = n_slots + (has_shared ? 1u : 0u);
    if (part->bytes < (uint64_t)n_tokens * n_out * out_dim * 4u ||
        mid->bytes < (uint64_t)n_tokens * n_out * ff_dim * 4u ||
        selected->bytes < (uint64_t)n_tokens * n_slots * 4u) {
        return 0;
    }
    const uint64_t rb = qwen4_moe_expert_row_bytes(weight_type, ff_dim);
    if (rb == 0u || out_dim > UINT64_MAX / rb) return 0;
    const uint64_t expert_bytes = (uint64_t)n_total_expert * rb * out_dim;
    if (expert_bytes > UINT32_MAX) return 0;
    struct ds4_vk_bind wbind = {};
    struct ds4_vk_bind wbind2 = {};
    struct ds4_vk_bind selb = vulkan_bind_tensor(DS4_VK_BINDING_OUT, selected);
    struct ds4_vk_bind tblb = {};
    const int pool = vulkan_qwen_moe_pool_binds(1u, rb * out_dim, n_tokens,
                                                n_slots, selected, &wbind,
                                                &wbind2, &selb, &tblb);
    if (!pool) {
        if (!qwen4_range_ok(down_offset, expert_bytes, model_size)) return 0;
        wbind = vulkan_bind_model(DS4_VK_BINDING_B, down_offset, expert_bytes);
        wbind2 = wbind;
    }
    uint64_t srb = 0u;
    if (has_shared) {
        srb = qwen4_moe_row_bytes(shared_type, ff_dim);
        if (srb == 0u) return 0;
        if (!qwen4_range_ok(shared_down_offset, srb * out_dim, model_size)) {
            return 0;
        }
    }

    struct ds4_vk_params p = {};
    p.n = ff_dim;
    p.rows = n_tokens;
    p.in_dim = out_dim;
    p.out_dim = n_slots;
    p.aux = (uint32_t)expert_bytes;
    p.ratio = (uint32_t)rb;
    p.blocks = has_shared ? shared_type : UINT32_MAX;
    p.index = (uint32_t)srb;
    p.rsvd3 = weight_type;
    p.flags = has_shared ? 2u : 0u;
    if (pool) p.flags |= 1u;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, mid);
    binds[nb++] = wbind;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, mid);
    binds[nb++] = selb;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT5, part);
    if (has_shared) {
        binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_X0, shared_down_offset,
                                        srb * out_dim);
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_X1, mid);
    } else {
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_X0, mid);
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_X1, mid);
    }
    if (pool) binds[nb++] = tblb;
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_MOE_DOWN], &p, sizeof(p),
                           binds, nb, out_dim, n_out, n_tokens);
}

int ds4_gpu_qwen4_moe_reduce_tensor(
    ds4_gpu_tensor *out, const ds4_gpu_tensor *part,
    const ds4_gpu_tensor *weights, const ds4_gpu_tensor *shared_gate,
    const ds4_gpu_tensor *shared, ds4_gpu_tensor *R, const ds4_gpu_tensor *inj,
    uint32_t n_tokens, uint32_t n_slots, uint32_t part_stride, uint32_t dim,
    uint32_t n_hc) {
    if (!out || !part || !weights) return 0;
    if (n_tokens == 0u || n_slots == 0u || dim == 0u || n_hc > 4u) return 0;
    if (part_stride < n_slots + ((shared_gate && !shared) ? 1u : 0u)) return 0;
    if (out->bytes < (uint64_t)n_tokens * dim * 4u ||
        part->bytes < (uint64_t)n_tokens * part_stride * dim * 4u ||
        weights->bytes < (uint64_t)n_tokens * n_slots * 4u) {
        return 0;
    }
    if (shared_gate && shared_gate->bytes < (uint64_t)n_tokens * 4u) return 0;
    if (shared && shared->bytes < (uint64_t)n_tokens * dim * 4u) return 0;
    if (n_hc) {
        if (!R || !inj) return 0;
        if (R->bytes < (uint64_t)n_tokens * n_hc * dim * 4u ||
            inj->bytes < (uint64_t)n_tokens * n_hc * n_hc * 8u * 4u) {
            return 0;
        }
    }

    struct ds4_vk_params p = {};
    p.rows = n_tokens;
    p.out_dim = dim;
    p.in_dim = n_slots;
    p.ratio = part_stride;
    p.index = n_hc;
    p.flags = (shared_gate ? 1u : 0u) | (shared ? 2u : 0u);
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, part);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, weights);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C,
                                     shared_gate ? shared_gate : weights);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_W,
                                     shared ? shared : part);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, n_hc ? R : out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, n_hc ? inj : out);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_MOE_REDUCE], &p, sizeof(p),
                           binds, nb, (dim + 255u) / 256u, n_tokens, 1u);
}

int ds4_gpu_qwen4_moe_build_lists_tensor(
    ds4_gpu_tensor *lists, ds4_gpu_tensor *counts,
    const ds4_gpu_tensor *selected, uint32_t n_tokens, uint32_t n_slots,
    uint32_t n_expert, uint32_t list_cap) {
    if (!lists || !counts || !selected) return 0;
    if (n_tokens == 0u || n_slots == 0u || n_expert == 0u || n_expert > 512u ||
        list_cap < n_tokens) {
        return 0;
    }
    const uint64_t pairs = (uint64_t)n_tokens * n_slots;
    if (pairs > INT32_MAX) return 0;
    if (lists->bytes < (uint64_t)n_expert * list_cap * 4u ||
        counts->bytes < (uint64_t)n_expert * 4u ||
        selected->bytes < pairs * 4u) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.index = n_expert;
    p.rows = (uint32_t)pairs;
    p.out_dim = list_cap;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_W, selected);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT4, lists);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT5, counts);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_MOE_BUILD_LISTS], &p,
                           sizeof(p), binds, nb, 1u, 1u, 1u);
}

int ds4_gpu_qwen4_moe_mm_mid_tensor(
    ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *lists,
    const ds4_gpu_tensor *counts, const void *model_map, uint64_t model_size,
    uint64_t gate_offset, uint64_t up_offset, uint32_t weight_type,
    uint32_t n_expert, uint32_t n_tokens, uint32_t n_slots, uint32_t n_out,
    uint32_t in_dim, uint32_t ff_dim, uint32_t list_cap) {
    if (!mid || !x || !lists || !counts || !model_map) return 0;
    if (n_tokens == 0u || n_expert == 0u || n_slots == 0u || n_out < n_slots ||
        in_dim == 0u || ff_dim == 0u || list_cap < n_tokens) {
        return 0;
    }
    if (mid->bytes < (uint64_t)n_tokens * n_out * ff_dim * 4u ||
        x->bytes < (uint64_t)n_tokens * in_dim * 4u ||
        lists->bytes < (uint64_t)n_expert * list_cap * 4u ||
        counts->bytes < (uint64_t)n_expert * 4u) {
        return 0;
    }
    const uint64_t rb = qwen4_moe_expert_row_bytes(weight_type, in_dim);
    if (rb == 0u || ff_dim > UINT64_MAX / rb) return 0;
    const uint64_t expert_bytes = (uint64_t)n_expert * rb * ff_dim;
    if (expert_bytes > UINT32_MAX) return 0;
    if (!qwen4_range_ok(gate_offset, expert_bytes, model_size) ||
        !qwen4_range_ok(up_offset, expert_bytes, model_size)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = in_dim;
    p.rows = n_tokens;
    p.in_dim = ff_dim;
    p.out_dim = n_out;
    p.index = n_slots;
    p.ratio = (uint32_t)rb;
    p.blocks = list_cap;
    p.rsvd3 = weight_type;
    p.aux = (uint32_t)expert_bytes;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_B, gate_offset, expert_bytes);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_C, up_offset, expert_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_W, lists);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_X0, counts);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, mid);
    uint32_t gz = (n_tokens + 15u) / 16u;
    if (gz > 8u) gz = 8u;
    p.rsvd2 = gz;
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_MOE_MM_MID], &p, sizeof(p),
                           binds, nb, (ff_dim + 15u) / 16u, n_expert, gz);
}

int ds4_gpu_qwen4_moe_mm_down_tensor(
    ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *lists,
    const ds4_gpu_tensor *counts, const void *model_map, uint64_t model_size,
    uint64_t down_offset, uint32_t weight_type, uint32_t n_expert,
    uint32_t n_tokens, uint32_t n_slots, uint32_t n_out, uint32_t ff_dim,
    uint32_t out_dim, uint32_t list_cap) {
    if (!part || !mid || !lists || !counts || !model_map) return 0;
    if (n_tokens == 0u || n_expert == 0u || n_slots == 0u || n_out < n_slots ||
        ff_dim == 0u || out_dim == 0u || list_cap < n_tokens) {
        return 0;
    }
    if (part->bytes < (uint64_t)n_tokens * n_out * out_dim * 4u ||
        mid->bytes < (uint64_t)n_tokens * n_out * ff_dim * 4u ||
        lists->bytes < (uint64_t)n_expert * list_cap * 4u ||
        counts->bytes < (uint64_t)n_expert * 4u) {
        return 0;
    }
    const uint64_t rb = qwen4_moe_expert_row_bytes(weight_type, ff_dim);
    if (rb == 0u || out_dim > UINT64_MAX / rb) return 0;
    const uint64_t expert_bytes = (uint64_t)n_expert * rb * out_dim;
    if (expert_bytes > UINT32_MAX) return 0;
    if (!qwen4_range_ok(down_offset, expert_bytes, model_size)) return 0;
    struct ds4_vk_params p = {};
    p.n = ff_dim;
    p.rows = n_tokens;
    p.in_dim = out_dim;
    p.out_dim = n_out;
    p.index = n_slots;
    p.ratio = (uint32_t)rb;
    p.blocks = list_cap;
    p.rsvd3 = weight_type;
    p.aux = (uint32_t)expert_bytes;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, mid);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_B, down_offset, expert_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, mid);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_W, lists);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_X0, counts);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, part);
    uint32_t gz = (n_tokens + 15u) / 16u;
    if (gz > 8u) gz = 8u;
    p.rsvd2 = gz;
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_MOE_MM_DOWN], &p, sizeof(p),
                           binds, nb, (out_dim + 15u) / 16u, n_expert, gz);
}


/* --- P4: hyper-connections ----------------------------------------------- */

int ds4_gpu_qwen4_hc_norm_tensor(
    ds4_gpu_tensor *xn, ds4_gpu_tensor *inj_part, const ds4_gpu_tensor *R,
    const void *model_map, uint64_t model_size, uint64_t gamma_offset,
    uint64_t inject_offset, uint32_t weight_type, uint32_t n_tokens,
    uint32_t n_embd, uint32_t n_hc, uint32_t n_inject, float eps) {
    if (!xn || !R || !model_map) return 0;
    if (n_tokens == 0u || n_embd == 0u || n_hc == 0u || n_hc > 8u ||
        n_inject > 4u) {
        return 0;
    }
    const uint64_t n = (uint64_t)n_tokens * n_embd * n_hc;
    if (xn->bytes < n * 4u || R->bytes < n * 4u) return 0;
    if (n_inject && (!inj_part || inj_part->bytes <
                     (uint64_t)n_tokens * n_hc * 8u * n_inject * 4u)) {
        return 0;
    }
    const uint64_t dim = (uint64_t)n_embd * n_hc;
    const uint64_t gb = dim * 4u;
    if (!qwen4_range_ok(gamma_offset, gb, model_size)) return 0;
    uint64_t wb = gb;
    if (n_inject) {
        const uint64_t rb = qwen4_dense_row_bytes(weight_type, (uint32_t)dim);
        if (rb == 0u) return 0;
        wb = rb * n_inject;
        if (!qwen4_range_ok(inject_offset, wb, model_size)) return 0;
    }
    struct ds4_vk_params p = {};
    p.rows = n_tokens;
    p.in_dim = n_embd;
    p.out_dim = n_hc;
    p.index = n_inject;
    p.rsvd3 = weight_type;
    p.eps = eps;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, R);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_B, gamma_offset, gb);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_X0,
                                    n_inject ? inject_offset : gamma_offset, wb);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, xn);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2,
                                     n_inject ? inj_part : xn);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_HC_NORM], &p, sizeof(p),
                           binds, nb, n_hc * 8u, n_tokens, 1u);
}

int ds4_gpu_qwen4_hc_gate_mix_tensor(
    ds4_gpu_tensor *mixed, const ds4_gpu_tensor *xn, const ds4_gpu_tensor *lo,
    const void *model_map, uint64_t model_size, uint64_t up_offset,
    uint32_t weight_type, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc,
    uint32_t n_rank) {
    if (!mixed || !xn || !lo || !model_map) return 0;
    if (n_tokens == 0u || n_embd == 0u || n_hc == 0u || n_hc > 4u ||
        n_rank == 0u) {
        return 0;
    }
    if (mixed->bytes < (uint64_t)n_tokens * n_embd * 4u ||
        xn->bytes < (uint64_t)n_tokens * n_embd * n_hc * 4u ||
        lo->bytes < (uint64_t)n_tokens * n_rank * 4u) {
        return 0;
    }
    const uint64_t rb = qwen4_dense_row_bytes(weight_type, n_rank);
    if (rb == 0u) return 0;
    const uint64_t wb = rb * (uint64_t)n_embd * n_hc;
    if (!qwen4_range_ok(up_offset, wb, model_size)) return 0;
    struct ds4_vk_params p = {};
    p.rows = n_tokens;
    p.in_dim = n_embd;
    p.out_dim = n_hc;
    p.index = n_rank;
    p.rsvd3 = weight_type;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, xn);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_B, up_offset, wb);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, lo);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, mixed);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_HC_GATE_MIX], &p, sizeof(p),
                           binds, nb, (n_embd + 255u) / 256u, n_tokens, 1u);
}

int ds4_gpu_qwen4_hc_combine_tensor(ds4_gpu_tensor *R,
                                    const ds4_gpu_tensor *out,
                                    const ds4_gpu_tensor *inj, uint32_t n_tokens,
                                    uint32_t n_embd, uint32_t n_hc) {
    if (!R || !out || !inj) return 0;
    if (n_tokens == 0u || n_embd == 0u || n_hc == 0u || n_hc > 4u) return 0;
    if (R->bytes < (uint64_t)n_tokens * n_embd * n_hc * 4u ||
        out->bytes < (uint64_t)n_tokens * n_embd * 4u ||
        inj->bytes < (uint64_t)n_tokens * n_hc * n_hc * 8u * 4u) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.rows = n_tokens;
    p.in_dim = n_embd;
    p.out_dim = n_hc;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, inj);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, R);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_HC_COMBINE], &p, sizeof(p),
                           binds, nb, (n_embd + 255u) / 256u, n_tokens, 1u);
}

int ds4_gpu_qwen4_hc_combine_norm_tensor(
    ds4_gpu_tensor *next, const ds4_gpu_tensor *blk,
    const ds4_gpu_tensor *oldinj, ds4_gpu_tensor *xn, ds4_gpu_tensor *inj,
    const ds4_gpu_tensor *R, const void *model_map, uint64_t model_size,
    uint64_t gamma_offset, uint64_t inject_offset, uint32_t weight_type,
    uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc, uint32_t n_inject,
    float eps) {
    if (!next || !blk || !oldinj || !xn || !inj || !R) return 0;
    if (next == R || inj == oldinj) return 0;
    const uint64_t bytes = (uint64_t)n_tokens * n_embd * n_hc * 4u;
    if (next->bytes < bytes || R->bytes < bytes) return 0;
    if (!ds4_gpu_tensor_copy(next, 0, R, 0, bytes)) return 0;
    if (!ds4_gpu_qwen4_hc_combine_tensor(next, blk, oldinj, n_tokens, n_embd,
                                         n_hc)) {
        return 0;
    }
    return ds4_gpu_qwen4_hc_norm_tensor(xn, inj, next, model_map, model_size,
                                        gamma_offset, inject_offset,
                                        weight_type, n_tokens, n_embd, n_hc,
                                        n_inject, eps);
}

int ds4_gpu_qwen4_hc_lo_act_tensor(ds4_gpu_tensor *lo_act,
                                   const ds4_gpu_tensor *lo, uint32_t n_tokens,
                                   uint32_t n_hc, uint32_t n_rank) {
    if (!lo_act || !lo) return 0;
    const uint64_t n = (uint64_t)n_tokens * n_rank;
    if (n == 0u || n_hc == 0u) return 0;
    if (lo_act->bytes < n * 4u || lo->bytes < n * 4u) return 0;
    struct ds4_vk_params p = {};
    p.n = (uint32_t)n;
    p.index = n_hc;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, lo);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, lo_act);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_HC_LO], &p, sizeof(p), binds,
                           nb, (uint32_t)((n + 255u) / 256u), 1u, 1u);
}

int ds4_gpu_qwen4_hc_mix_rows_tensor(ds4_gpu_tensor *mixed,
                                     const ds4_gpu_tensor *u,
                                     const ds4_gpu_tensor *xn, uint32_t n_tokens,
                                     uint32_t n_embd, uint32_t n_hc) {
    if (!mixed || !u || !xn) return 0;
    if (n_tokens == 0u || n_embd == 0u || n_hc == 0u || n_hc > 4u) return 0;
    if (mixed->bytes < (uint64_t)n_tokens * n_embd * 4u ||
        u->bytes < (uint64_t)n_tokens * n_hc * n_embd * 4u ||
        xn->bytes < (uint64_t)n_tokens * n_hc * n_embd * 4u) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.rows = n_tokens;
    p.in_dim = n_embd;
    p.out_dim = n_hc;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, u);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, xn);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, mixed);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_HC_MIX_ROWS], &p, sizeof(p),
                           binds, nb, (n_embd + 255u) / 256u, n_tokens, 1u);
}

/* --- P4: PLE (n-gram) ----------------------------------------------------- */
int ds4_gpu_qwen4_ple_gate_tensor(
    ds4_gpu_tensor *gated, ds4_gpu_tensor *normed, const ds4_gpu_tensor *R,
    const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
    const void *model_map, uint64_t model_size, uint64_t g_key_offset,
    uint64_t g_query_offset, uint64_t g_conv_offset, uint32_t n_tokens,
    uint32_t n_embd, uint32_t n_hc, float eps) {
    if (!gated || !normed || !R || !key || !value || !model_map) return 0;
    if (n_tokens == 0u || n_embd == 0u || n_hc == 0u || n_hc > 4u) return 0;
    const uint64_t bytes = (uint64_t)n_tokens * n_embd * n_hc * 4u;
    if (gated->bytes < bytes || normed->bytes < bytes || R->bytes < bytes ||
        key->bytes < bytes || value->bytes < (uint64_t)n_tokens * n_embd * 4u) {
        return 0;
    }
    const uint64_t wb = (uint64_t)n_embd * n_hc * 4u;
    if (!qwen4_range_ok(g_key_offset, wb, model_size) ||
        !qwen4_range_ok(g_query_offset, wb, model_size) ||
        !qwen4_range_ok(g_conv_offset, wb, model_size)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.rows = n_tokens;
    p.in_dim = n_embd;
    p.out_dim = n_hc;
    p.eps = eps;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, R);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, key);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, value);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_X0, g_key_offset, wb);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_X1, g_query_offset, wb);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_X2, g_conv_offset, wb);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, gated);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, normed);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_PLE_GATE], &p, sizeof(p),
                           binds, nb, n_hc, n_tokens, 1u);
}

int ds4_gpu_qwen4_ple_conv_tensor(
    ds4_gpu_tensor *R, const ds4_gpu_tensor *gated,
    const ds4_gpu_tensor *normed, ds4_gpu_tensor *history,
    const void *model_map, uint64_t model_size, uint64_t weight_offset,
    uint32_t weight_type, uint32_t n_tokens, uint32_t n_channels,
    uint32_t conv_kernel, uint32_t dilation, ds4_gpu_tensor *snap_history,
    uint32_t snap_tok, ds4_gpu_tensor *snap2_history, uint32_t snap2_tok) {
    if (!R || !gated || !normed || !history || !model_map) return 0;
    if (n_tokens == 0u || n_channels == 0u || conv_kernel < 2u ||
        conv_kernel > 4u || dilation == 0u || dilation > 3u ||
        (weight_type != 0u && weight_type != 1u)) {
        return 0;
    }
    const uint64_t n = (uint64_t)n_tokens * n_channels * 4u;
    const uint64_t hb = (uint64_t)(conv_kernel - 1u) * dilation * n_channels * 4u;
    if (R->bytes < n || gated->bytes < n || normed->bytes < n ||
        history->bytes < hb || (snap_history && snap_history->bytes < hb) ||
        (snap2_history && snap2_history->bytes < hb)) {
        return 0;
    }
    const uint64_t rb = qwen4_dense_row_bytes(weight_type, n_channels * conv_kernel);
    if (rb == 0u || !qwen4_range_ok(weight_offset, rb, model_size)) return 0;
    struct ds4_vk_params p = {};
    p.n = n_channels;
    p.rows = n_tokens;
    p.in_dim = conv_kernel;
    p.index = dilation;
    p.rsvd3 = weight_type;
    p.aux = snap_history ? snap_tok : UINT32_MAX;
    p.blocks = snap2_history ? snap2_tok : UINT32_MAX;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, gated);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, normed);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, weight_offset, rb);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, R);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, history);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3,
                                     snap_history ? snap_history : history);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT4,
                                     snap2_history ? snap2_history : history);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_PLE_CONV], &p, sizeof(p),
                           binds, nb, (n_channels + 255u) / 256u, 1u, 1u);
}

/* --- P5: multi-token predictor (nextn / MTP) ---------------------------- */

/* Two-pass argmax over n_vocab F32 logits: pass 1 reduces 4096-value chunks
 * into `scratch` (uint2 per chunk), pass 2 merges the winners into out[0].
 * NaN is skipped, +-Inf kept, ties break toward the lower index (matches the
 * host sample_argmax and the Metal kernel_qwen4_argmax). */
int ds4_gpu_qwen4_argmax_tensor(ds4_gpu_tensor *out_idx, ds4_gpu_tensor *scratch,
                                const ds4_gpu_tensor *logits, uint32_t n_vocab) {
    if (!out_idx || !scratch || !logits) return 0;
    if (n_vocab == 0u || n_vocab > 0x7fffffffu) return 0;
    const uint32_t chunks = (n_vocab + 4095u) / 4096u;
    if (out_idx->bytes < sizeof(int32_t)) return 0;
    if (scratch->bytes < (uint64_t)chunks * 8u) return 0;
    if (logits->bytes < (uint64_t)n_vocab * sizeof(float)) return 0;
    struct ds4_vk_params p = {};
    p.n = n_vocab;
    struct ds4_vk_bind binds[3];
    binds[0] = vulkan_bind_tensor(DS4_VK_BINDING_A, logits);
    binds[1] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out_idx);
    binds[2] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, scratch);
    if (!vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_ARGMAX], &p, sizeof(p), binds, 3,
                         chunks, 1u, 1u)) {
        return 0;
    }
    p.n = chunks;
    p.flags = 1u;
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_ARGMAX], &p, sizeof(p), binds, 3,
                           1u, 1u, 1u);
}

/* Stage the MTP concat rows: row 0 = [rms(e)*g_e | 0], row 1+s =
 * [0 | R_s/rms(R)*g_h_s], one RMS over all hc streams.  One threadgroup per
 * row. */
int ds4_gpu_qwen4_mtp_stage_tensor(
    ds4_gpu_tensor *cat, const ds4_gpu_tensor *e, const ds4_gpu_tensor *R,
    const void *model_map, uint64_t model_size, uint64_t g_e_offset,
    uint64_t g_h_offset, uint32_t n_embd, uint32_t n_hc, float eps) {
    if (!cat || !e || !R || !model_map) return 0;
    if (n_embd == 0u || n_hc == 0u || n_hc > 8u) return 0;
    const uint64_t eb = (uint64_t)n_embd * sizeof(float);
    const uint64_t hb = (uint64_t)n_hc * eb;
    if (e->bytes < eb || R->bytes < hb) return 0;
    if (cat->bytes < (uint64_t)(n_hc + 1u) * 2u * eb) return 0;
    if (!qwen4_range_ok(g_e_offset, eb, model_size) ||
        !qwen4_range_ok(g_h_offset, hb, model_size)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = n_embd;
    p.index = n_hc;
    p.eps = eps;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, e);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, R);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_C, g_e_offset, eb);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, g_h_offset, hb);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, cat);
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_MTP_STAGE], &p, sizeof(p), binds,
                           nb, n_hc + 1u, 1u, 1u);
}

/* R_out[s][d] = proj[0][d] + proj[1+s][d]. */
int ds4_gpu_qwen4_mtp_combine_tensor(
    ds4_gpu_tensor *R_out, const ds4_gpu_tensor *proj, uint32_t n_embd,
    uint32_t n_hc) {
    if (!R_out || !proj) return 0;
    if (n_embd == 0u || n_hc == 0u || n_hc > 8u) return 0;
    const uint64_t eb = (uint64_t)n_embd * sizeof(float);
    if (R_out->bytes < (uint64_t)n_hc * eb) return 0;
    if (proj->bytes < (uint64_t)(n_hc + 1u) * eb) return 0;
    struct ds4_vk_params p = {};
    p.n = n_embd;
    p.index = n_hc;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, proj);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, R_out);
    const uint32_t n = n_embd * n_hc;
    return vulkan_dispatch(g_pipes[DS4_PIPE_QWEN4_MTP_COMBINE], &p, sizeof(p),
                           binds, nb, (n + 255u) / 256u, 1u, 1u);
}
