/* vulkan/ds4_vulkan_unavailable.c -- stubs for the Vulkan backend surface
 * that is not implemented yet.
 *
 * The shared graph engine references the full ds4_gpu_* contract even when
 * the Vulkan backend only implements the core lifecycle and tensor paths so
 * far.  These C-linkage stubs keep everything linkable.  The stubs are MUTE
 * (they return 0 / do nothing): the set of unimplemented entry points is
 * reported once at backend init by ds4_vulkan_report_unavailable() (called
 * from ds4_gpu_init).  Each stub registers its own name at static-init time,
 * so the report stays in sync with the actual stub list automatically.
 *
 * Pattern: mirror ds4_rocm_unavailable.cu.  Functions accept any argument
 * list (they never inspect their arguments). */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

namespace {
std::vector<const char *> &vulkan_unavailable_registry() {
    static std::vector<const char *> names;
    return names;
}
struct VulkanUnavailableRegistrar {
    explicit VulkanUnavailableRegistrar(const char *name) {
        vulkan_unavailable_registry().push_back(name);
    }
};
}  /* namespace */

#define VULKAN_UNAVAILABLE_INT(name) \
    namespace { VulkanUnavailableRegistrar vulkan_unavail_reg_##name(#name); } \
    extern "C" int name(...) { return 0; }

#define VULKAN_UNAVAILABLE_VOID(name) \
    namespace { VulkanUnavailableRegistrar vulkan_unavail_reg_##name(#name); } \
    extern "C" void name(...) {}

VULKAN_UNAVAILABLE_INT(ds4_gpu_attention_prefill_raw_heads_range_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_attention_prefill_static_mixed_heads_range_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_device_cache_support_tensors)
VULKAN_UNAVAILABLE_INT(ds4_gpu_device_cache_tensors)
VULKAN_UNAVAILABLE_INT(ds4_gpu_moe_handoff_pack_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_register_model_map_no_copy)
VULKAN_UNAVAILABLE_INT(ds4_gpu_register_support_map)
VULKAN_UNAVAILABLE_INT(ds4_gpu_rope_tail_decode_rows_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_routed_moe_batch_owned_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_routed_moe_one_owned_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_routed_moe_owned_packed_combine_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_routed_moe_owned_slots_combine_rows_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_routed_moe_owned_slots_combine_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor)

/* The streaming expert cache surface (seed/begin/prepare/load/seed_experts)
 * is implemented in ds4_vulkan_compat.c as the Fase 6 all-resident policy:
 * the model wrapper VkBuffer makes every expert always addressable, so those
 * entry points are bookkeeping that reports success.  GLM is out of scope. */

/* Multi-GPU / TP surface: Fase 0 is single-GPU.  The simple same-device
 * copies (ds4_gpu_tensor_copy_xdev* and wait_xdev) live in
 * ds4_vulkan_compat.c as real single-GPU implementations; the grouped and
 * reduction forms are unreachable on a single device. */
VULKAN_UNAVAILABLE_INT(ds4_gpu_tensor_copy_xdev3)
VULKAN_UNAVAILABLE_INT(ds4_gpu_tensor_copy_xdev3_default_dst)
VULKAN_UNAVAILABLE_INT(ds4_gpu_add_xdev_tensor)

namespace { VulkanUnavailableRegistrar vulkan_unavail_reg_ds4_gpu_tp_big_gate_kick(
    "ds4_gpu_tp_big_gate_kick"); }
extern "C" uint64_t ds4_gpu_tp_big_gate_kick(...) { return 0; }
VULKAN_UNAVAILABLE_INT(ds4_gpu_tp_big_gate_wait)

/* Retired/fused readback plumbing is unused on Vulkan for now. */
VULKAN_UNAVAILABLE_INT(ds4_gpu_add_rms_norm_weight_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_attention_noncausal_raw_batch_heads_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_attention_output_q8_batch_f16_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_attention_output_q8_tp_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_dsv4_comp_row_finalize_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_flash_kv_stage_f16_tensor)
/* GLM (DS4_MODEL_FAMILY_GLM_DSA) is OUT OF SCOPE for the Vulkan backend:
 * Vulkan supports only the DeepSeek V4 families (Flash/Pro).  These stubs
 * exist solely so the shared graph engine links; if a GLM model is ever
 * routed to Vulkan they fail loudly and the engine falls back to its CPU
 * path.  Do not implement GLM kernels here. */
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_flash_staged_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_flash_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_full_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_indexed_batch_lora_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_indexed_batch_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_indexed_batch_typed_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_indexed_decode_split_group8_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_indexed_decode_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_indexed_decode_typed_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_build_kv_cache_flash_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_build_kv_cache_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_fill_selected_range_batch_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_fill_selected_range_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_indexer_rope_tail_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_indexer_score_one_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_indexer_scores_batch_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_k_b_project_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_k_b_project_typed_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_kv_lora_rms_norm_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_qk_lowrank_q8_0_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_qk_lowrank_typed_batch_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_qk_lowrank_typed_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_qkv_norm_store_compact_kv_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_rope_tail_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_routed_moe_batch_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_routed_moe_one_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_router_select_batch_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_router_select_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_store_compact_kv_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_store_indexer_k_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_value_project_q8_0_batch_heads_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_value_project_typed_batch_heads_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_hc_rms_norm_mix_f16_available)
VULKAN_UNAVAILABLE_INT(ds4_gpu_hc_rms_norm_mix_f16_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_hc_rms_scale_project_f16_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_lookup_cache_strict)
VULKAN_UNAVAILABLE_INT(ds4_gpu_matmul_f16_quad_compressor_store_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_pack_slot_rows_f32_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_qkv_pair_quad_compressor_store_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_router_project_select_fused_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_router_shared_gate_up_q8_0_tensor)

/* Tensor-parallel gates are Metal-only; stubs keep shared graph code
 * linkable (TP option validation rejects non-Metal backends). */
VULKAN_UNAVAILABLE_INT(ds4_gpu_tp_batch_gate_encode)
VULKAN_UNAVAILABLE_INT(ds4_gpu_tp_big_gate_encode)
VULKAN_UNAVAILABLE_INT(ds4_gpu_tp_failed)
VULKAN_UNAVAILABLE_INT(ds4_gpu_tp_gate_encode)
VULKAN_UNAVAILABLE_INT(ds4_gpu_tp_init)
VULKAN_UNAVAILABLE_VOID(ds4_gpu_tp_keepalive_pause)
VULKAN_UNAVAILABLE_VOID(ds4_gpu_tp_set_attn_head_split)
VULKAN_UNAVAILABLE_VOID(ds4_gpu_tp_set_batch_exchange)
VULKAN_UNAVAILABLE_VOID(ds4_gpu_tp_set_big_exchange)
VULKAN_UNAVAILABLE_VOID(ds4_gpu_tp_shutdown)
VULKAN_UNAVAILABLE_VOID(ds4_gpu_tp_suspend_expert_sharding)

/* Decode-island graph capture is CUDA-only; Vulkan decodes eagerly.
 * ds4_gpu_decode_graphs_invalidate is implemented (silently) in
 * ds4_vulkan.c because the engine calls it unconditionally at teardown. */
VULKAN_UNAVAILABLE_INT(ds4_gpu_decode_graphs_supported)
VULKAN_UNAVAILABLE_INT(ds4_gpu_decode_graph_begin)
VULKAN_UNAVAILABLE_INT(ds4_gpu_decode_graph_end)
VULKAN_UNAVAILABLE_VOID(ds4_gpu_decode_graph_abort)

/* Upstream Vision-Exp / GLM 5.3 multimodal and DSpark GPU surface.  All are
 * unreachable on the Vulkan runtime for the DS4 model family (the engine
 * gates them behind backend==CUDA/Metal, the GLM_DSA model family, or a loaded
 * vision model, none of which the Vulkan backend serves), so they stay loud
 * stubs instead of real no-ops: any future wiring bug is caught by the
 * "Vulkan unavailable" message at the call site. */
VULKAN_UNAVAILABLE_INT(ds4_gpu_device_is_spark)
VULKAN_UNAVAILABLE_INT(ds4_gpu_set_aux_model_map_range)
VULKAN_UNAVAILABLE_INT(ds4_gpu_matmul_q4_K_pair_decode_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_attention_visual_mixed_batch_heads_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_router_select_batch_visual_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_deepseek4_vision_encode)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm53_embedding_bf16)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm53_matmul_bf16)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm53_kda_decode)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm53_kda_prefill)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm53_vision_encode)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm53_scatter_image_hc)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm53_expand_pool_selection_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm53_indexer_pool_update_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm53_indexer_scores_batch_tensor)
VULKAN_UNAVAILABLE_INT(ds4_gpu_glm_attention_dense_compact_lora_causal_tensor)

/* Print, once at backend init, a categorized summary of the ds4_gpu_* entry
 * points that are silent stubs (not implemented / out of scope).  The total is
 * the exact number of registered stubs; the per-category split is heuristic
 * (prefix/substring).  Set DS4_VULKAN_STUBS_VERBOSE=1 to list every name. */
extern "C" void ds4_vulkan_report_unavailable(void) {
    static bool reported = false;
    if (reported) return;
    reported = true;
    const std::vector<const char *> &names = vulkan_unavailable_registry();
    size_t n_glm = 0, n_vision = 0, n_mgpu = 0, n_graph = 0, n_other = 0;
    for (size_t i = 0; i < names.size(); i++) {
        const char *nm = names[i];
        if (strncmp(nm, "ds4_gpu_glm", 11) == 0) {
            n_glm++;
        } else if (strstr(nm, "vision") != NULL || strstr(nm, "visual") != NULL ||
                   strstr(nm, "scatter_image") != NULL) {
            n_vision++;
        } else if (strncmp(nm, "ds4_gpu_tp_", 11) == 0 ||
                   strstr(nm, "_owned_") != NULL || strstr(nm, "xdev") != NULL ||
                   strstr(nm, "tp_tensor") != NULL) {
            n_mgpu++;
        } else if (strncmp(nm, "ds4_gpu_decode_graph", 20) == 0) {
            n_graph++;
        } else {
            n_other++;
        }
    }
    fprintf(stderr,
            "ds4: Vulkan backend: %zu ds4_gpu_* entry points are silent stubs "
            "(not implemented / out of scope):\n",
            names.size());
    if (n_glm) {
        fprintf(stderr, "ds4:   - GLM 5.2/5.3 + KDA (out of scope): %zu\n", n_glm);
    }
    if (n_vision) {
        fprintf(stderr, "ds4:   - Vision-Exp / multimodal (out of scope): %zu\n", n_vision);
    }
    if (n_mgpu) {
        fprintf(stderr, "ds4:   - multi-GPU / TP / expert-parallel (Fase 8): %zu\n", n_mgpu);
    }
    if (n_graph) {
        fprintf(stderr, "ds4:   - decode graph capture (CUDA-only): %zu\n", n_graph);
    }
    if (n_other) {
        fprintf(stderr, "ds4:   - fused Metal-only / plumbing / other: %zu\n", n_other);
    }
    if (getenv("DS4_VULKAN_STUBS_VERBOSE") != NULL) {
        for (size_t i = 0; i < names.size(); i++) {
            fprintf(stderr, "ds4:     %s\n", names[i]);
        }
    }
}

