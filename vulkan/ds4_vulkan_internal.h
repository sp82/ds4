/* vulkan/ds4_vulkan_internal.h -- shared surface of the Vulkan backend.
 *
 * The Vulkan backend is split into translation units:
 *   - ds4_vulkan.c            core lifecycle, staging, pool, dispatch, and the
 *                             DeepSeek-family ds4_gpu_* entry points;
 *   - ds4_vulkan_qwen.c       Qwen3.8-Flash-Next (qwen4exp) entry points;
 *   - ds4_vulkan_compat.c     single-GPU compatibility / multi-GPU shims;
 *   - ds4_vulkan_unavailable.c loud stubs for the unimplemented surface.
 *
 * This header carries ONLY the minimum the per-family TUs need to record
 * dispatches: the descriptor bindings, the push-constant mirror, the pipeline
 * enum, the active pipeline table, and the launcher/bind/scratch helpers.
 * Everything else stays private to ds4_vulkan.c.  Add to it sparingly; keep
 * the core self-contained. */

#ifndef DS4_VULKAN_INTERNAL_H
#define DS4_VULKAN_INTERNAL_H

#include <vulkan/vulkan.h>

#include <stdint.h>

#include "ds4_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Descriptor bindings shared with the HLSL (see shaders/common.hlsl): SRVs on
 * bindings 0..3 (t0..t3), the UAVs on bindings 4..8 (u0..u4), plus the MoE
 * expert->slot table on 9. */
#define DS4_VK_BINDING_A 0u
#define DS4_VK_BINDING_B 1u
#define DS4_VK_BINDING_C 2u
#define DS4_VK_BINDING_W 3u   /* model weight buffer (ByteAddressBuffer) */
#define DS4_VK_BINDING_OUT 4u
#define DS4_VK_BINDING_OUT2 5u
#define DS4_VK_BINDING_OUT3 6u
#define DS4_VK_BINDING_OUT4 7u
#define DS4_VK_BINDING_OUT5 8u
#define DS4_VK_BINDING_TBL 9u  /* on-device expert->slot table (MoE pool) */
/* Extra storage buffers (bindings 10..15).  They can be either SRVs
 * (register(t10..t15)) or UAVs (register(u6..u11)) depending on the shader;
 * attn_prep needs six read-write outputs plus ten read-only inputs. */
#define DS4_VK_BINDING_X0 10u
#define DS4_VK_BINDING_X1 11u
#define DS4_VK_BINDING_X2 12u
#define DS4_VK_BINDING_X3 13u
#define DS4_VK_BINDING_X4 14u
#define DS4_VK_BINDING_X5 15u
#define DS4_VK_MAX_BINDS 16u

/* Fixed push constant block.  Layout must match DS4Params in
 * shaders/common.hlsl (24 x uint32 = 96 bytes). */
struct ds4_vk_params {
    uint32_t n;         /* element count / head_dim / n_embd / row_width */
    uint32_t rows;      /* n_tok / n_rows / n_heads * n_tok */
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    int32_t  inverse;
    float    eps;
    float    clamp;
    float    weight;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t blocks;
    uint32_t index;
    uint32_t aux;
    uint32_t ratio;
    uint32_t flags;
    uint32_t rsvd2;
    uint32_t rsvd3;
};

/* One descriptor write: a storage-buffer range on a binding. */
struct ds4_vk_bind {
    uint32_t     binding;
    VkBuffer     buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
};

/* Pipeline table.  Values are append-only (the enum index is the stable pipe
 * id used by telemetry); never reorder.  DS4_VK_PIPE_COUNT must equal the
 * number of rows in the pipes[] table in ds4_vulkan.c. */
enum ds4_vk_pipe {
    DS4_PIPE_ADD = 0,
    DS4_PIPE_ADD3,
    DS4_PIPE_SWIGLU,
    DS4_PIPE_ARGMAX,
    DS4_PIPE_SORT_I32_ROWS_ASC,
    DS4_PIPE_RMS_NORM_PLAIN,
    DS4_PIPE_RMS_NORM_WEIGHT,
    DS4_PIPE_ROPE_TAIL,
    DS4_PIPE_MATMUL_Q8_0,
    DS4_PIPE_MATMUL_Q8_0_PREQ,
    DS4_PIPE_MATMUL_Q8_0_KSLICE,
    DS4_PIPE_QUANTIZE_Q8_0,
    DS4_PIPE_MATMUL_Q8_0_TOP1,
    DS4_PIPE_F32_TO_F16,
    DS4_PIPE_MATMUL_F16,
    DS4_PIPE_MATMUL_F16_PAIR_COMPRESSOR_STORE,
    DS4_PIPE_MATMUL_F32,
    DS4_PIPE_EMBED_TOKEN_Q8_0,
    DS4_PIPE_EMBED_TOKENS_Q8_0,
    DS4_PIPE_EMBED_TOKEN_HC,
    DS4_PIPE_EMBED_TOKENS_HC,
    DS4_PIPE_FP8_KV_QUANTIZE,
    DS4_PIPE_STORE_RAW_KV,
    DS4_PIPE_KV_FP8_STORE_RAW,
    DS4_PIPE_ATTN_DECODE,
    DS4_PIPE_ATTN_DECODE_INDEXED,
    DS4_PIPE_ATTN_PREFILL,
    DS4_PIPE_ATTN_OUTPUT_LOW_Q8,
    DS4_PIPE_ROUTER_SELECT,
    DS4_PIPE_MOE_GATE_UP_MID_Q8,
    DS4_PIPE_MOE_DOWN_Q8,
    DS4_PIPE_MOE_SUM,
    DS4_PIPE_MOE_GATE_UP_MID_IQ2XXS,
    DS4_PIPE_MOE_DOWN_Q2K,
    DS4_PIPE_FILL_F32,
    DS4_PIPE_HEAD_RMS_NORM,
    DS4_PIPE_HEAD_RMS_NORM_ROPE_TAIL,
    DS4_PIPE_HC_SPLIT_SINKHORN,
    DS4_PIPE_HC_WEIGHTED_SUM,
    DS4_PIPE_HC_EXPAND,
    DS4_PIPE_HC_EXPAND4,
    DS4_PIPE_HC_SPLIT_WEIGHTED_SUM_FUSED,
    DS4_PIPE_OUTPUT_HC_WEIGHTS,
    DS4_PIPE_REPEAT_HC,
    DS4_PIPE_HC_EXPAND4_HALF,
    DS4_PIPE_HC_EXPAND4_ADD_HALF,
    DS4_PIPE_INDEXER_SCORES,
    DS4_PIPE_DSV4_INDEXER_QAT,
    DS4_PIPE_INDEXER_TOPK,
    DS4_PIPE_INDEXER_TOP1_VALUE,
    DS4_PIPE_TOPK_MASK,
    DS4_PIPE_DSPARK_MARKOV_ARGMAX,
    DS4_PIPE_COMPRESSOR_STORE,
    DS4_PIPE_COMPRESSOR_SET_ROWS,
    DS4_PIPE_COMPRESSOR_PREFILL_POOL,
    DS4_PIPE_COMPRESSOR_UPDATE_POOL,
    DS4_PIPE_COMPRESSOR_SHIFT_RATIO4,
    DS4_PIPE_DIRECTIONAL_STEERING_PROJECT,
    DS4_PIPE_MATMUL_Q8_0_PREQ_V2,
    DS4_PIPE_MATMUL_Q8_0_PREQ_V3,
    DS4_PIPE_MOE_GATE_UP_MID_IQ2XXS_V2,
    DS4_PIPE_MOE_DOWN_Q2K_V2,
    DS4_PIPE_MOE_GATE_UP_MID_Q4K,
    DS4_PIPE_MOE_DOWN_Q4K,
    DS4_PIPE_ATTN_OUTPUT_LOW_Q4K,
    DS4_PIPE_MOE_GATE_UP_MID_MXFP4,
    DS4_PIPE_MOE_DOWN_MXFP4,
    DS4_PIPE_MOE_GATE_UP_MID_MXFP4_V2,
    DS4_PIPE_MOE_DOWN_MXFP4_V2,
    DS4_PIPE_MOE_GROUP,
    DS4_PIPE_MATMUL_Q4K,
    DS4_PIPE_MATMUL_Q4_0,
    DS4_PIPE_ATTN_OUTPUT_LOW_Q8_V2,
    DS4_PIPE_ATTN_OUTPUT_LOW_Q8_V3,
    /* Qwen3.8-Flash-Next (append-only from here). */
    DS4_PIPE_MATMUL_Q8_0_F32,
    DS4_PIPE_MATMUL_BF16,
    DS4_PIPE_CONV_STREAM,
    DS4_PIPE_GDN_PREP,
    DS4_PIPE_GDN_SCAN,
    DS4_PIPE_GDN_OUT,
    DS4_PIPE_IDX_SCORE,
    DS4_PIPE_IDX_TILE_MAX,
    DS4_PIPE_IDX_SELECT,
    DS4_PIPE_IDX_EXPAND,
    DS4_PIPE_QWEN4_ATTN_PREP,
    DS4_PIPE_QWEN4_BLOCK_KEY,
    DS4_PIPE_QWEN4_ATTN_DECODE,
    DS4_PIPE_QWEN4_ATTN_MERGE,
    DS4_PIPE_QWEN4_ROUTER_TOPK,
    DS4_PIPE_QWEN4_MOE_MID,
    DS4_PIPE_QWEN4_MOE_DOWN,
    DS4_PIPE_QWEN4_MOE_REDUCE,
    DS4_PIPE_QWEN4_MOE_BUILD_LISTS,
    DS4_PIPE_QWEN4_MOE_MM_MID,
    DS4_PIPE_QWEN4_MOE_MM_DOWN,
};
#define DS4_VK_PIPE_COUNT 95

/* Active pipeline table (swapped per logical device tier by the context
 * save/load path). */
extern VkPipeline g_pipes[DS4_VK_PIPE_COUNT];

/* General compute launcher: allocates a descriptor set, applies `n_binds`
 * storage-buffer writes, binds pipeline + push constants + set, dispatches
 * (gx, gy, gz) threadgroups, and inserts a SHADER_WRITE->SHADER_READ barrier.
 * Works both inside an open command scope and as a one-shot. */
int vulkan_dispatch(VkPipeline pipeline,
                    const void *params, uint32_t params_size,
                    const struct ds4_vk_bind *binds, uint32_t n_binds,
                    uint32_t gx, uint32_t gy, uint32_t gz);

/* Descriptor binds.  vulkan_bind_tensor binds a whole tensor; _at binds a
 * sub-range; vulkan_bind_model resolves a [offset, offset+bytes) range of the
 * staged model wrapper (weight tensors). */
struct ds4_vk_bind vulkan_bind_tensor(uint32_t binding,
                                      const ds4_gpu_tensor *t);
struct ds4_vk_bind vulkan_bind_tensor_at(uint32_t binding,
                                         const ds4_gpu_tensor *t,
                                         uint64_t off, uint64_t bytes);
struct ds4_vk_bind vulkan_bind_model(uint32_t binding,
                                     uint64_t offset, uint64_t bytes);
int vulkan_model_range_ok(uint64_t offset, uint64_t bytes);

/* Persistent scratch tensors, grown on demand (reused across dispatches). */
ds4_gpu_tensor *vulkan_scratch_a(uint64_t bytes);
ds4_gpu_tensor *vulkan_scratch_b(uint64_t bytes);
ds4_gpu_tensor *vulkan_scratch_c(uint64_t bytes);
ds4_gpu_tensor *vulkan_scratch_d(uint64_t bytes);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* DS4_VULKAN_INTERNAL_H */
