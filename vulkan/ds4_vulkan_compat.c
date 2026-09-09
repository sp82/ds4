/* vulkan/ds4_vulkan_compat.c -- single-GPU compatibility surface for the
 * Vulkan backend (Fase 0).
 *
 * Mirrors ds4_rocm_compat.cu: enforces one GPU per process, provides the
 * tier-indexed tensor allocation shims the engine expects, the runtime
 * feature setters, and the auto-probe used by --gpu-vram auto.  The real
 * compute kernels live in ds4_vulkan.c (or are stubbed there). */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_gpu_args.h"

#define DS4_VULKAN_LOG_PREFIX "ds4: Vulkan "

ds4_gpu_ctx g_gpu[DS4_MAX_GPUS] = {};
int g_n_gpus = 1;
int g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS] = {{1}};

extern "C" void ds4_vulkan_tensor_release_device(ds4_gpu_tensor *t);
extern "C" void ds4_vulkan_set_ssd_streaming(int enabled);
extern "C" uint64_t ds4_vulkan_probe_vram_bytes(void);
extern "C" void ds4_vulkan_set_stream_budget(uint32_t experts);
extern "C" void ds4_vulkan_set_expert_bytes(uint64_t bytes);
extern "C" void ds4_vulkan_set_stream_layer_count(uint32_t layers);
extern "C" uint32_t ds4_vulkan_effective_budget(void);
extern "C" int ds4_vulkan_stream_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *ids, uint32_t n_selected);
extern "C" int ds4_vulkan_stream_seed_selected_async(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *ids, uint32_t n_selected);
extern "C" void ds4_vulkan_pool_commit_pending(void);
extern "C" int ds4_vulkan_stream_seed_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *ids, uint32_t n_tokens, uint32_t n_selected);
extern "C" int ds4_vulkan_stream_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *ids, const uint32_t *priorities, uint32_t n_experts);
extern "C" void ds4_vulkan_stream_pool_reset(void);
extern "C" void ds4_vulkan_stream_pool_reset_hotness(void);
extern "C" void ds4_vulkan_telemetry_snapshot(ds4_gpu_expert_telemetry *t);

static int vulkan_tier_valid(int tier) {
    return tier == 0 && g_n_gpus == 1;
}

/* --- multi-GPU init (single-GPU enforcement) ---------------------------- */

extern "C" int ds4_gpu_init_multi(const ds4_gpu_config *cfg) {
    if (!cfg || cfg->n_gpus != 1) {
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "supports one GPU per process for now\n");
        return 0;
    }
    g_gpu[0].device_id = cfg->device_indices[0];
    g_n_gpus = 1;
    return ds4_gpu_init();
}

extern "C" int ds4_gpu_set_current_device(int tier) {
    if (!vulkan_tier_valid(tier)) return 1;
    return 0;
}

extern "C" int ds4_gpu_set_current_device_fenced(int tier) {
    return ds4_gpu_set_current_device(tier);
}

/* --- tier-indexed tensor allocation -------------------------------------- */

extern "C" int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int tier,
                                       uint64_t bytes) {
    if (!t) return 1;
    if (!vulkan_tier_valid(tier)) return 2;
    ds4_gpu_tensor *alloced = ds4_gpu_tensor_alloc(bytes);
    if (!alloced) return 3;
    *t = *alloced;
    free(alloced);
    return 0;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier,
                                                       uint64_t bytes) {
    if (!vulkan_tier_valid(tier)) return NULL;
    return ds4_gpu_tensor_alloc(bytes);
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int tier,
                                                           uint64_t bytes) {
    if (!vulkan_tier_valid(tier)) return NULL;
    return ds4_gpu_tensor_alloc_managed(bytes);
}

extern "C" void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *t) {
    if (!t) return;
    ds4_vulkan_tensor_release_device(t);
    memset(t, 0, sizeof(*t));
}

extern "C" int ds4_gpu_tensor_device(const ds4_gpu_tensor *t) {
    return t ? 0 : -1;
}

extern "C" int ds4_gpu_tensor_copy_async(ds4_gpu_tensor *dst,
                                         const ds4_gpu_tensor *src,
                                         uint64_t bytes) {
    if (!dst || !src || bytes > dst->bytes || bytes > src->bytes) return 0;
    if (bytes == 0) return 1;
    return ds4_gpu_tensor_copy(dst, 0, src, 0, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                                        const ds4_gpu_tensor *src,
                                        uint64_t bytes) {
    return ds4_gpu_tensor_copy(dst, 0, src, 0, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *dst,
                                                const ds4_gpu_tensor *src,
                                                uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                                const ds4_gpu_tensor *src,
                                                uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src,
                                        int dst_tier) {
    return src && vulkan_tier_valid(dst_tier);
}

extern "C" int ds4_gpu_tensor_wait_xdev_default(const ds4_gpu_tensor *src,
                                                int dst_tier) {
    return ds4_gpu_tensor_wait_xdev(src, dst_tier);
}

extern "C" uint64_t ds4_gpu_tier_free_vram(int tier) {
    if (!vulkan_tier_valid(tier)) return 0;
    /* Fase 0 does not track budgets; report a generous ceiling so the
     * placement planner does not spill to CPU.  A future allocator tracks
     * VkDeviceMemory usage. */
    return (uint64_t)16ull * 1024ull * 1024ull * 1024ull;
}

/* --- auto probe (--gpu-vram auto) ---------------------------------------- */

extern "C" int ds4_gpu_args_probe_auto_cuda(
        const int *device_filter, int filter_len, ds4_gpu_config *out,
        size_t safety_margin_bytes, char *errbuf, size_t errbuflen) {
    (void)device_filter;
    (void)filter_len;
    if (!out) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen, "internal: NULL out");
        }
        return 1;
    }
    /* Report a single device with the real device-local heap size. */
    memset(out, 0, sizeof(*out));
    out->device_indices[0] = 0;
    out->vram_bytes[0] = ds4_vulkan_probe_vram_bytes();
    if (out->vram_bytes[0] == 0) {
        out->vram_bytes[0] = 16ull * 1024ull * 1024ull * 1024ull;
    }
    out->n_gpus = 1;
    out->safety_margin_bytes = safety_margin_bytes;
    return 0;
}

/* --- runtime feature setters ---------------------------------------------- */

static int g_vulkan_q8_cache_suppressed = 0;

extern "C" int ds4_gpu_q8_cache_suppressed(void) {
    return g_vulkan_q8_cache_suppressed;
}

extern "C" void ds4_gpu_set_q8_cache_suppressed(int suppressed) {
    g_vulkan_q8_cache_suppressed = suppressed != 0;
}

extern "C" int ds4_gpu_set_decode_fast_attention(int enabled) {
    (void)enabled;
    return 0;
}

extern "C" int ds4_gpu_set_decode_score_vec4(int enabled) {
    (void)enabled;
    return 0;
}

extern "C" int ds4_gpu_set_decode_pipeline_fast_lookup(int enabled) {
    (void)enabled;
    return 0;
}

extern "C" int ds4_gpu_set_decode_attn_rope_fuse(int enabled) {
    (void)enabled;
    return 0;
}

extern "C" int ds4_gpu_decode_attn_rope_fuse_available(void) {
    return 0;
}

extern "C" int ds4_gpu_decode_attn_rope_fuse_used(void) {
    return 0;
}

extern "C" int ds4_gpu_kv_rope_fp8_fuse_available(void) {
    return 0;
}

/* GLM is OUT OF SCOPE for the Vulkan backend (see ds4_vulkan_unavailable.c).
 * These setters are no-ops kept only for contract completeness. */
extern "C" void ds4_gpu_set_glm_model(bool enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_set_glm_streaming_prefill_full_layer(bool enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_set_glm_mtp_verify_mode(bool enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_set_quality(bool quality) {
    (void)quality;
}

extern "C" void ds4_gpu_set_ssd_streaming(bool enabled) {
    ds4_vulkan_set_ssd_streaming(enabled ? 1 : 0);
}

extern "C" void ds4_gpu_enable_q8_dequant_gemm(void) {
}

extern "C" int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes,
                                                   uint64_t context_bytes) {
    (void)kv_cache_bytes;
    (void)context_bytes;
    return 0;
}

extern "C" int ds4_gpu_pro_q4_expert_table_auto_available(void) {
    return 0;
}

extern "C" void ds4_gpu_model_residency_skip(int skip) {
    (void)skip;
}

extern "C" void ds4_gpu_print_memory_report(const char *label) {
    (void)label;
}

extern "C" void ds4_gpu_release_q8_f16_cache(void) {
}

extern "C" void ds4_gpu_release_zero_prefix_prefill_mask_cache(void) {
}

extern "C" void ds4_gpu_test_set_flags(uint32_t flags) {
    (void)flags;
}

/* --- streaming expert cache queries --------------------------------------- */

static uint32_t g_vulkan_stream_budget = 0;
static uint64_t g_vulkan_expert_bytes = 0;

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    g_vulkan_stream_budget = experts;
    ds4_vulkan_set_stream_budget(experts);
}

extern "C" void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    g_vulkan_expert_bytes = bytes;
    ds4_vulkan_set_expert_bytes(bytes);
}

extern "C" void ds4_gpu_set_streaming_expert_cache_layer_count(uint32_t layers) {
    ds4_vulkan_set_stream_layer_count(layers);
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    return g_vulkan_stream_budget;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return g_vulkan_stream_budget;
}

/* Fase 6: the streaming expert cache stages the hot routed experts of every
 * layer into the persistent device-local expert pool (ds4_vulkan.c) instead
 * of re-staging the full expert layer per token.  The seed/load entry points
 * below route the engine's per-(token,layer) selections into the pool; the
 * routed MoE kernels bind the pool (vulkan_pool_moe_binds) when it is active.
 * A seed whose selection is already resident writes nothing (cross-token
 * reuse), so cache hits cost no device synchronization. */
static int vulkan_stream_table_valid(
        const ds4_gpu_stream_expert_table *table) {
    return table != NULL &&
           table->n_total_expert != 0 &&
           table->gate_expert_bytes != 0 &&
           table->down_expert_bytes != 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    if (gate_expert_bytes == 0 || down_expert_bytes == 0) return 0;
    const uint32_t eff = ds4_vulkan_effective_budget();
    const uint32_t configured = ds4_gpu_stream_expert_cache_configured_count();
    return eff > configured ? eff : configured;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!vulkan_stream_table_valid(table)) return 0;
    return ds4_vulkan_stream_seed_selected(table, selected_ids, n_selected);
}

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!vulkan_stream_table_valid(table)) return 0;
    return ds4_vulkan_stream_seed_selected(table, selected_ids, n_selected);
}

/* Fase 7: async variant used by the engine's async load worker (the store
 * runs on the dedicated worker command buffer; the main thread waits the
 * fence via ds4_vulkan_pool_commit_pending before the MoE dispatch). */
extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load_async(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!vulkan_stream_table_valid(table)) return 0;
    return ds4_vulkan_stream_seed_selected_async(table, selected_ids,
                                                 n_selected);
}

extern "C" int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected) {
    if (!vulkan_stream_table_valid(table)) return 0;
    return ds4_vulkan_stream_seed_batch(table, selected_ids, n_tokens,
                                        n_selected);
}

extern "C" int ds4_gpu_stream_expert_cache_load_layer(
        const ds4_gpu_stream_expert_table *table) {
    if (!vulkan_stream_table_valid(table)) return 0;
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_from_layer_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_tokens,
        uint32_t                           n_seed_tokens,
        uint32_t                           n_selected) {
    (void)selected;
    (void)n_tokens;
    (void)n_seed_tokens;
    (void)n_selected;
    if (!vulkan_stream_table_valid(table)) return 0;
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_finish_pending_batch(void) {
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_release_layer_cache(void) {
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts) {
    if (!vulkan_stream_table_valid(table)) return 0;
    return ds4_vulkan_stream_seed_experts(table, expert_ids,
                                          expert_priorities, n_experts);
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts) {
    return ds4_gpu_stream_expert_cache_seed_experts(table, expert_ids,
                                                    expert_priorities,
                                                    n_experts);
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) {
    return 0;
}

/* Fase 7: signal/wait/commit/tensor_read_after_selected_event are
 * implemented in ds4_vulkan.c (fence-based readback for the async expert
 * load worker; Metal/CUDA pattern).  note_service_thread is a no-op: the
 * worker only touches the dedicated worker command buffer, never g_cmd. */
extern "C" void ds4_gpu_stream_expert_cache_note_service_thread(void) {
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
    ds4_vulkan_stream_pool_reset_hotness();
}

extern "C" void ds4_gpu_stream_expert_cache_release_resident(void) {
    ds4_vulkan_stream_pool_reset();
}

/* Aggregate pool telemetry snapshot (per-layer counters, per-phase totals,
 * occupancy gauges).  Backend-internal; the engine reads deltas between two
 * snapshots to report a generation/response (--vulkan-stats). */
extern "C" int ds4_gpu_stream_expert_cache_telemetry_snapshot(
        ds4_gpu_expert_telemetry *t) {
    if (!t) return 0;
    ds4_vulkan_telemetry_snapshot(t);
    return 1;
}
