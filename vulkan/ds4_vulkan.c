/* vulkan/ds4_vulkan.c -- Vulkan GPU backend for DS4.
 *
 * Fase 2: kernel core.  The model file is wrapped as a single VkBuffer
 * through VK_EXT_external_memory_host (the analog of Metal's
 * ds4_gpu_wrap_model_range), so weight-backed kernels bind the model buffer
 * directly at the weight tensor's byte offset.  Tensor representation is
 * VkBuffer + VkDeviceMemory (host visible/coherent) with persistent host
 * mapping (Fase 1).  All compute kernels are compiled from HLSL through dxc
 * (see shaders/gen_shaders.py and ds4_vulkan_shaders.inc).
 *
 * This file is compiled as C++17 (the Vulkan C API is C, but the shared
 * ds4_gpu.h contract wraps everything in extern "C" so linkage is C).
 *
 * Tensor layout: each ds4_gpu_tensor points (via tensor->ptr) at a heap
 * allocated ds4_vulkan_tensor handle that carries the VkBuffer,
 * VkDeviceMemory, byte offset and size.  Views share the buffer/memory and
 * only adjust offset/bytes.  Host reads/writes go through a persistent host
 * mapping of the host-visible memory (host-coherent, so no explicit flush).
 * Compute kernels bind the VkBuffer at the tensor's offset through a
 * descriptor set updated per dispatch.
 */

#include <vulkan/vulkan.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <atomic>

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_vulkan_mgpu.h"

#include "shaders/ds4_vulkan_shaders.inc"

/* Defined in ds4_vulkan_unavailable.c: prints once, at backend init, the
 * categorized list of ds4_gpu_* entry points that are silent stubs. */
extern "C" void ds4_vulkan_report_unavailable(void);

#define DS4_VULKAN_LOG_PREFIX "ds4: Vulkan "

/* Descriptor bindings shared with the HLSL (see common.hlsl): SRVs on
 * bindings 0..3 (t0..t3), the UAVs on bindings 4..7 (u0..u3). */
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
#define DS4_VK_MAX_BINDS 10u

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

/* --- runtime state ----------------------------------------------------- */

static VkInstance        g_instance  = VK_NULL_HANDLE;
static VkPhysicalDevice  g_phys      = VK_NULL_HANDLE;
static VkDevice          g_device    = VK_NULL_HANDLE;
static VkQueue           g_queue     = VK_NULL_HANDLE;
static uint32_t          g_queue_family = 0;

/* GPU timestamp scope timing (DS4_VULKAN_DEBUG_GPU_TS).  Debug-only: writes a
 * device timestamp at the first dispatch of a main compute scope and one right
 * before the scope's vkEndCommandBuffer, then reports the delta.  This measures
 * the scope's own GPU time, excluding any queue work queued ahead (e.g. the
 * expert-store copy) that inflates the host-side fwait.  It requires the async
 * overlap worker to be off (DS4_METAL_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP)
 * so the single query slot is never touched by the worker thread. */
static VkQueryPool       g_ts_pool = VK_NULL_HANDLE;
static float             g_ts_period = 0.0f;
static int               g_ts_on = 0;
static int               g_ts_started = 0;
static int               g_ts_router_written = 0;
static int               g_ts_verbose = 0;
static uint32_t          g_ts_valid_bits = 0;
/* Verbose mode (DS4_VULKAN_DEBUG_GPU_TS_VERBOSE): timestamp every dispatch of
 * a scope and report the per-dispatch GPU time with the pipe name. */
static uint32_t          g_ts_pipe[96];
static uint32_t          g_ts_n = 0;
static char              g_pipe_names[96][24];
static VkCommandPool     g_cmd_pool  = VK_NULL_HANDLE;
/* Compute-scope command buffers, double-buffered (Fase 7 flush early-submit):
 * a scope can be submitted while the sibling CB records the next one, so the
 * GPU overlaps the shared-expert compute with the worker's expert store
 * without a device drain.  g_cmd_fence[i] tracks CB i in flight: VK_NULL_HANDLE
 * = free, a real fence = submitted scope, g_readback_fence = the router scope
 * submitted by signal_selected_readback_ready (the readback owner resets it;
 * an acquire only waits it, never resets it, to not race the worker's wait). */
static VkCommandBuffer   g_cmd[2]    = { VK_NULL_HANDLE, VK_NULL_HANDLE };
static VkFence           g_cmd_fence[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
/* Persistent per-CB fence, created once and reused; g_cmd_fence[i] is only the
 * in-flight marker (this fence, or g_readback_fence for a readback scope). */
static VkFence           g_cb_fence[2]  = { VK_NULL_HANDLE, VK_NULL_HANDLE };
static int               g_cmd_i     = 0;   /* CB currently recording */
static bool              g_commands_active = false;
/* Debug-only (DS4_VULKAN_DEBUG_SUBMIT): wall time at which the current scope
 * started recording, so the submit line can report the host-side encode time
 * (end_commands - begin) separately from the GPU fence wait. */
static double            g_dbg_scope_t0_ms = 0.0;

/* Fase 7 (overlap store esperti <-> compute GPU): the engine's async load
 * worker (ds4.c metal_graph_selected_async_load_worker_main, already used by
 * Metal/CUDA) reads the router selection and stores the missing experts in a
 * background thread while the main thread keeps encoding.  The worker must
 * not touch g_cmd (the main may be recording a scope), so the readback and
 * the pool copy use a dedicated command buffer + fence on the main queue.
 * The main queue FIFO orders the copy after the previous MoE scope (slot
 * reuse safety) and the main waits the fence (vulkan_pool_commit_pending)
 * right before dispatching the MoE that reads the pool. */
static pthread_mutex_t   g_worker_mutex = PTHREAD_MUTEX_INITIALIZER;
static VkCommandPool     g_worker_pool  = VK_NULL_HANDLE;
static VkCommandBuffer   g_worker_cb   = VK_NULL_HANDLE;
static VkFence           g_worker_fence = VK_NULL_HANDLE;
static int               g_worker_fence_pending = 0;
/* VK_KHR_shader_integer_dot_product available (hardware packed int8 dot). */
static int               g_has_int_dot = 0;
/* attn_output_low variant chosen by the init autotuner: -1 = not yet, 0/1/2 =
 * v1/v2/v3.  See vulkan_autotune_attn_out. */
static int               g_attn_out_autotuned = -1;
/* Fence signaling the router-scope submission that produced the selection
 * read back by the worker (signal_selected_readback_ready / wait). */
static VkFence           g_readback_fence = VK_NULL_HANDLE;
static int               g_readback_fence_pending = 0;

/* Per-scope debug counters (DS4_VULKAN_DEBUG_SUBMIT): accumulate the dispatch
 * count and the storage-buffer bind footprint recorded into the open scope,
 * printed when end_commands submits it. */
static uint64_t g_scope_bind_bytes = 0;
static uint32_t g_scope_bind_count = 0;
static uint32_t g_scope_dispatch_count = 0;

static VkPhysicalDeviceProperties    g_props;
/* Physical-device index chosen at init (forced or auto-picked).  Printed at
 * startup so a test session can confirm which GPU ran without inferring. */
static uint32_t                      g_device_index = 0;
/* VK_NV_device_diagnostic_checkpoints debug (DS4_VULKAN_CHECKPOINTS): insert a
 * marker per dispatch; after a device lost the queue's last checkpoint names
 * the faulting kernel (no serialization, unlike timestamps). */
static PFN_vkCmdSetCheckpointNV          g_pfnCmdSetCheckpointNV = NULL;
static PFN_vkGetQueueCheckpointDataNV    g_pfnGetQueueCheckpointDataNV = NULL;
static int                               g_dbg_checkpoints = -1;
static VkPhysicalDeviceMemoryProperties g_mem_props;
static uint32_t g_host_visible_mem_type = UINT32_MAX;
static uint32_t g_device_local_mem_type = UINT32_MAX; /* pure VRAM (no map) */
static uint32_t g_uma_mem_type = UINT32_MAX; /* DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT */
static int g_is_uma = 0; /* true when the only device-local heap is also host-visible */
static uint64_t g_host_pointer_align = 4096;   /* staging-window alignment */

/* Optional VK_KHR_buffer_device_address proc (debug: prints window VAs so a
 * GPUVM fault address can be mapped to a staged window). */
static PFN_vkGetBufferDeviceAddressKHR g_vk_bda = NULL;

/* Command-scope helpers used by the staging upload/download paths before
 * their definitions below. */
static VkCommandBuffer vulkan_dispatch_begin(void);
static void vulkan_trace(const char *fmt, ...);
static int vulkan_submit_one_shot(void);
static int vulkan_compute_init(void);
static int vulkan_cb_submit(void);

/* Compute pipeline state (created in vulkan_compute_init, called from
 * ds4_gpu_init; never lazily -- see progress.md gotcha 8). */
static VkDescriptorPool     g_desc_pool    = VK_NULL_HANDLE;
static VkDescriptorSetLayout g_desc_layout = VK_NULL_HANDLE;
static VkPipelineLayout     g_pipe_layout  = VK_NULL_HANDLE;
/* Descriptor sets allocated since the last pool reset.  The double-buffered
 * scopes no longer drain per layer, so during long read-free passes (the
 * decode-style prefill) the pool would otherwise exhaust (~950 sets/layer-
 * token vs 8192 max): a high-water drain in vulkan_cb_acquire bounds it. */
static uint32_t            g_desc_sets_allocated = 0;
#define DS4_VK_DESC_SET_HWM 6144u

/* Host reads of device-written memory are only coherent after the submitted
 * scopes complete.  Track whether any work is in flight so ds4_gpu_tensor_read
 * can drain before copying (the engine relies on reads reflecting the latest
 * GPU writes, e.g. the router's selected ids and the decode logits). */
static int g_vulkan_device_dirty = 0;

/* Serialize vkDeviceWaitIdle + the descriptor-pool reset: the Fase 7 async
 * expert-load worker also calls vulkan_device_wait (staging grow), and a
 * concurrent vkResetDescriptorPool from two threads corrupts RADV internal
 * state (double free / context lost). */
static pthread_mutex_t g_device_wait_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Serialize every use of g_queue: the Fase 7 async expert-load worker submits
 * on g_queue while the main thread also submits and drains it.  Vulkan
 * requires external synchronization of the queue; without it NVIDIA loses the
 * device.  (Defined here so vulkan_device_wait can take it too.) */
static pthread_mutex_t g_queue_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Serialize g_readback_fence: the main thread resets+submits it (signal) while
 * the async worker waits it (wait); Vulkan requires external synchronization
 * of VkFence. */
static pthread_mutex_t g_readback_mutex = PTHREAD_MUTEX_INITIALIZER;

static void vulkan_device_wait_impl(int reset_pool) {
    if (g_device == VK_NULL_HANDLE) return;
    pthread_mutex_lock(&g_device_wait_mutex);
    /* vkDeviceWaitIdle drains the queue; the following vkResetDescriptorPool
     * frees descriptors a still-in-flight submit could reference.  Both must
     * exclude the worker's queue submits. */
    pthread_mutex_lock(&g_queue_mutex);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    vkDeviceWaitIdle(g_device);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double ms = (double)(t1.tv_sec - t0.tv_sec) * 1000.0 +
                      (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
    if (ms > 1000.0 && getenv("DS4_VULKAN_DEBUG_BINDS") != NULL) {
        fprintf(stderr, "ds4: Vulkan debug: device wait %.0f ms\n", ms);
    }
    g_vulkan_device_dirty = 0;
    /* Every set allocated for a completed scope is safe to reclaim.  With the
     * static decode map the model windows are no longer re-staged per layer,
     * so this is the per-layer reset point for the descriptor pool (Fase 6
     * step 3; the pool is only otherwise reset at synchronize).  The async
     * worker must NOT reset the pool the main thread allocates from. */
    if (reset_pool && g_desc_pool != VK_NULL_HANDLE && !g_commands_active) {
        vkResetDescriptorPool(g_device, g_desc_pool, 0);
        g_desc_sets_allocated = 0;
    }
    pthread_mutex_unlock(&g_queue_mutex);
    pthread_mutex_unlock(&g_device_wait_mutex);
}

/* Full wait: drain + reclaim the descriptor pool (main thread).  The async
 * worker uses vulkan_device_wait_impl(0) to drain without touching the pool. */
static void vulkan_device_wait(void) { vulkan_device_wait_impl(1); }

#define DS4_VK_PIPE_COUNT 74
static VkPipeline     g_pipes[DS4_VK_PIPE_COUNT];
static VkShaderModule g_mods[DS4_VK_PIPE_COUNT];

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
};

/* Whole-model wrapper (Fase 6 step 2 staging pool).  The windows requested by
 * ds4_gpu_set_model_map / set_model_map_range / set_model_map_spans are copied
 * host-side into per-window host-visible tensors; weight-backed kernels bind
 * the window that covers a tensor's byte offset.  The old
 * VK_EXT_external_memory_host zero-copy import is gone: it cannot import a
 * file-backed mmap on a discrete GPU (RX 6900 XT / RADV, progress.md §3h).
 * A single contiguous window is not enough: this model's per-layer tensors
 * are spread across the whole interleaved GGUF file, so the staging keeps a
 * set of windows (one per requested span, coalesced on small gaps). */

/* Largest single staged window.  A per-layer span (the largest SSD-streaming
 * request) is well below this; a whole-model request (e.g. the MTP support
 * model's full-range map) is not staged at all. */
#define DS4_VK_MAX_STAGED_WINDOW (8ull * 1024ull * 1024ull * 1024ull)
/* Coalesce spans closer than this into one window (the HC scale/base pair and
 * a layer's small static tensors sit adjacent in the GGUF). */
#define DS4_VK_WINDOW_MERGE_GAP (256ull * 1024ull)
/* A decode-time static map (all 43 layers' non-expert tensors, spread across
 * the interleaved GGUF) coalesces into far more than the old per-layer count;
 * the cap is generous so accumulation-style maps stay fully staged. */
#define DS4_VK_MAX_MODEL_WINDOWS 4096

struct ds4_vk_model_window {
    VkBuffer         buffer;
    ds4_gpu_tensor  *tensor;   /* owns buffer+memory+host map; NULL when the
                                * window shares a batched buffer owned by the
                                * first window of the set */
    uint64_t         base;     /* absolute model-file offset */
    uint64_t         size;     /* staged bytes */
    uint64_t         buf_offset; /* offset inside the shared window buffer */
};

struct ds4_vk_model_span {
    uint64_t lo;
    uint64_t hi;
};

static struct ds4_vk_model_window g_model_windows[DS4_VK_MAX_MODEL_WINDOWS];
static uint32_t g_model_window_count = 0;

/* Fase 8e: persistent per-device weight cache.
 *
 * ds4_gpu_device_cache_tensors registers the model-file ranges a tier owns;
 * they are staged once into device-local VRAM and stay resident for the whole
 * session (unlike g_model_windows, which the per-layer SSD-streaming path
 * destroys and re-uploads per token).  vulkan_model_window_for resolves weight
 * binds against this cache before the transient windows, so a layer whose
 * weights are cached needs no per-token re-staging at all.  Each logical tier
 * keeps its own set (swapped by vulkan_ctx_save/load like every other
 * per-device global). */
static struct ds4_vk_model_window g_weight_cache[DS4_VK_MAX_MODEL_WINDOWS];
static uint32_t g_weight_cache_count = 0;
static int      g_weight_cache_ready = 0;

/* Fase 6 step 3: device-local (persistent host-visible) expert pool.
 *
 * The decode keeps the hot routed experts of every layer in a persistent
 * per-layer buffer instead of re-staging the whole ~1.8 GB expert layer per
 * token.  The pool is seeded through the streaming expert cache entry points
 * (seed_selected / begin_selected_load / prepare_selected_batch, routed from
 * ds4_vulkan_compat.c); the routed MoE kernels then bind the pool and a
 * per-layer remap scratch (expert id -> pool slot) instead of the model
 * windows, so the shaders keep their existing expert*expert_bytes addressing.
 *
 * Per-layer tensor layout:
 *   [ gate slots ][ up slots ][ down slots ][ remap scratch (int32) ]
 * Slot i holds the i-th resident expert; the remap scratch carries, per
 * (token, slot), the pool slot index of the selected expert.
 *
 * Writes happen host-side between command scopes.  To stay correct when the
 * GPU may still read a slot (same layer of a previous token), any actual
 * write is preceded by vkDeviceWaitIdle; a seed whose selection is unchanged
 * writes nothing and waits for nothing (the cross-token reuse win).  Overlap
 * of the loads with compute is a Fase 8c target. */

#define DS4_VK_POOL_MAX_LAYERS 512
#define DS4_VK_POOL_MIN_SLOTS 6u
#define DS4_VK_POOL_MAX_SLOTS 256u
#define DS4_VK_POOL_SEL_CAP 4096u   /* max (token,slot) pairs per MoE call */
#define DS4_VK_POOL_TABLE_ENTRIES 384u /* max expert id per layer */

/* Hotness (SPECS_HOTNESS): route hotness is a per-(layer,expert) counter that
 * survives eviction.  The pool stores it per layer (n_experts entries); the
 * engine-level table is indexed by layer slot in this TU.  Decay every
 * DS4_VK_POOL_HOTNESS_DECAY_TOKENS routed seeds (a decode token routes one
 * seed per layer, so this tracks Metal's decode-token cadence). */
#define DS4_VK_POOL_HOTNESS_DECAY_TOKENS 16u

struct ds4_vk_pool_layer {
    int              active;
    ds4_gpu_tensor  *tensor;   /* owner: gate|up|down slots (device-local) */
    ds4_gpu_tensor  *meta;     /* on-device expert->slot table (host-visible) */
    uint64_t         gate_expert_bytes;
    uint64_t         down_expert_bytes;
    uint32_t         n_slots;   /* slot capacity */
    uint32_t         n_used;    /* resident slots */
    int32_t          slots[DS4_VK_POOL_MAX_SLOTS];   /* expert id per slot */
    uint32_t         slot_age[DS4_VK_POOL_MAX_SLOTS];/* LRU last-use age */
    uint32_t         age;                            /* monotonic use counter */
    int32_t          table_host[DS4_VK_POOL_TABLE_ENTRIES]; /* mirror; -1 = absent */
    uint8_t          ever_loaded[DS4_VK_POOL_TABLE_ENTRIES]; /* stored since pool setup */
    uint32_t         route_hotness[DS4_VK_POOL_TABLE_ENTRIES]; /* per-expert LFU ticks */
    uint32_t         route_seed_count;  /* routed seeds seen (approx. tokens) */
    uint32_t         route_last_decay;  /* decay watermark in seed_count */
    /* Recency pin (DS4_VULKAN_POOL_PIN_TOKENS): epoch of the last routed seed
     * that requested each expert, so the previous tokens' working set can be
     * protected from eviction.  0 = never requested. */
    uint32_t         seed_epoch;
    uint32_t         last_used[DS4_VK_POOL_TABLE_ENTRIES];
};

/* Bisect gate: DS4_VULKAN_HOTNESS_OFF=1 restores the legacy LRU-only eviction
 * and disables the route-hotness notes, for A/B measurements on the server.
 * Cached at first call (like the other DS4_VULKAN_* toggles). */
static int vulkan_pool_hotness_off(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = getenv("DS4_VULKAN_HOTNESS_OFF") != NULL;
    }
    return cached;
}

/* Decay the layer's route hotness (halve all entries) when the routed-seed
 * counter has advanced by DS4_VK_POOL_HOTNESS_DECAY_TOKENS since the last
 * decay, mirroring Metal (ds4_metal.m maybe_decay_route_hotness). */
static void vulkan_pool_hotness_maybe_decay(struct ds4_vk_pool_layer *l) {
    while (l->route_seed_count - l->route_last_decay >=
           DS4_VK_POOL_HOTNESS_DECAY_TOKENS) {
        for (uint32_t i = 0; i < DS4_VK_POOL_TABLE_ENTRIES; i++) {
            l->route_hotness[i] >>= 1;
        }
        l->route_last_decay += DS4_VK_POOL_HOTNESS_DECAY_TOKENS;
    }
}

/* Add `amount` to the route hotness of expert e (saturating), mirroring
 * Metal's note_route_hotness. */
static void vulkan_pool_hotness_note(struct ds4_vk_pool_layer *l, int32_t e,
                                     uint32_t amount) {
    if (!l || e < 0 || (uint32_t)e >= DS4_VK_POOL_TABLE_ENTRIES ||
        amount == 0) {
        return;
    }
    uint32_t *hotness = &l->route_hotness[(uint32_t)e];
    if (*hotness > UINT32_MAX - amount) {
        *hotness = UINT32_MAX;
    } else {
        *hotness += amount;
    }
}

/* Note a routed selection (all selected experts, hit or miss): +1 per expert,
 * then advance the layer seed counter and decay as needed. */
static void vulkan_pool_hotness_note_selected(struct ds4_vk_pool_layer *l,
                                              const int32_t *ids,
                                              uint32_t n_ids) {
    if (!l || !ids || n_ids == 0 || vulkan_pool_hotness_off()) return;
    if (l->route_seed_count != UINT32_MAX) {
        l->route_seed_count++;
    }
    vulkan_pool_hotness_maybe_decay(l);
    for (uint32_t i = 0; i < n_ids; i++) {
        vulkan_pool_hotness_note(l, ids[i], 1u);
    }
}

/* Hotlist seed: give each seeded expert an initial retention advantage from
 * its priority (Metal ds4_metal.m:16630).  Falls back to 1. */
static void vulkan_pool_hotness_seed_priorities(struct ds4_vk_pool_layer *l,
                                                const int32_t *ids,
                                                const uint32_t *priorities,
                                                uint32_t n_experts) {
    if (!l || !ids || vulkan_pool_hotness_off()) return;
    for (uint32_t i = 0; i < n_experts; i++) {
        const uint32_t priority = priorities ? priorities[i] : 0u;
        vulkan_pool_hotness_note(l, ids[i], priority != 0 ? priority : 1u);
    }
}

static void vulkan_pool_hotness_reset(struct ds4_vk_pool_layer *l) {
    if (!l) return;
    memset(l->route_hotness, 0, sizeof(l->route_hotness));
    l->route_seed_count = 0;
    l->route_last_decay = 0;
}

static struct ds4_vk_pool_layer g_pool_layers[DS4_VK_POOL_MAX_LAYERS];
static uint32_t g_pool_slots_per_layer = DS4_VK_POOL_MIN_SLOTS;
static uint32_t g_pool_budget = 0;
static uint32_t g_pool_layer_count = 0;   /* routed-expert layers in the model */
static uint64_t g_pool_per_expert_bytes = 0;
static uint64_t g_pool_total_bytes = 0;
static uint64_t g_pool_max_bytes_fixed = 0;   /* frozen at pool_configure */
static int g_pool_ready = 0;   /* budget-derived slot count configured */

/* Telemetry of the streaming expert pool (feeding the engine's --vulkan-stats
 * report).  Counters are cumulative per (layer, phase) so the writer needs no
 * lock: a layer pool is only ever touched by one thread at a time (the main
 * thread for sync seeds and MoE, or the async load worker for that layer's
 * store), and a layer is never seeded by two threads concurrently.  The
 * values are atomics only so that a snapshot read is safe when the worker is
 * still running.  Phases follow the seed entry point: routed decode seeds,
 * prefill batch seeds, and the one-shot hotlist preload. */
struct ds4_vk_pool_tel {
    std::atomic<uint64_t> requests[DS4_GPU_EXPERT_PHASES];
    std::atomic<uint64_t> hits[DS4_GPU_EXPERT_PHASES];
    std::atomic<uint64_t> misses[DS4_GPU_EXPERT_PHASES];
    std::atomic<uint64_t> loaded_bytes[DS4_GPU_EXPERT_PHASES];
    std::atomic<uint64_t> reload_misses[DS4_GPU_EXPERT_PHASES];
    std::atomic<uint64_t> evictions;
};
static struct ds4_vk_pool_tel g_pool_tel[DS4_VK_POOL_MAX_LAYERS];

/* Wall time spent stalling decode on expert loads: the sync seeds' device
 * wait before a store and the main thread's wait for the async worker copy
 * fence.  Only the main thread writes this (the worker submits without
 * waiting), but it stays atomic for safe snapshot reads. */
static std::atomic<uint64_t> g_pool_tel_wait_us(0);

/* Debug-only breakdown of the expert pool store (DS4_VULKAN_DEBUG_POOL_TIME):
 * host copy time (memcpy from the mmap: page faults + RAM copy) vs the GPU
 * submit/wait.  Answers whether the store is disk/fault-bound or submit-bound. */
static std::atomic<uint64_t> g_dbg_store_copy_us(0);
static std::atomic<uint64_t> g_dbg_store_submit_us(0);
static std::atomic<uint64_t> g_dbg_store_bytes(0);
static std::atomic<uint64_t> g_dbg_store_calls(0);
static std::atomic<int> g_dbg_store_time_reg(0);

static void vulkan_store_time_report(void) {
    const uint64_t copy = g_dbg_store_copy_us.load();
    const uint64_t sub = g_dbg_store_submit_us.load();
    const uint64_t bytes = g_dbg_store_bytes.load();
    const uint64_t calls = g_dbg_store_calls.load();
    const double copy_ms = (double)copy / 1000.0;
    const double sub_ms = (double)sub / 1000.0;
    const double mib = (double)bytes / (1024.0 * 1024.0);
    fprintf(stderr,
            "ds4: vulkan store-time: calls=%llu bytes=%.1f MiB "
            "copy=%.1f ms (%.1f MiB/s) submit=%.1f ms total=%.1f ms\n",
            (unsigned long long)calls, mib, copy_ms,
            copy_ms > 0.0 ? mib / (copy_ms / 1000.0) : 0.0,
            sub_ms, copy_ms + sub_ms);
}

/* Count the reloads among the experts being loaded now: an expert whose
 * weights were stored in this pool before (ever_loaded) and evicted since is
 * a reload, the signature of a pool that is too small for the routed set. */
static uint32_t vulkan_pool_tel_reloads(const struct ds4_vk_pool_layer *l,
                                        const int32_t *missing,
                                        uint32_t n_missing) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < n_missing; i++) {
        const int32_t e = missing[i];
        if (e >= 0 && (uint32_t)e < DS4_VK_POOL_TABLE_ENTRIES &&
            l->ever_loaded[(uint32_t)e]) {
            n++;
        }
    }
    return n;
}

static uint64_t vulkan_pool_effective_budget_bytes(void);
static uint64_t vulkan_pool_auto_budget_bytes(void);

/* Total bytes currently allocated in the device-local heap (weights, static
 * decode windows, expert pool).  Lets the pool budget target a fixed amount of
 * free VRAM instead of a blunt fraction of the heap, so the same code leaves
 * a consistent reserve on any GPU size (16/24/32 GiB...). */
static uint64_t g_device_local_bytes = 0;

/* Hard cap on the whole pool.  The base is the decode working set (~1.8 GB);
 * a generous CLI budget or the auto VRAM-derived budget raises it so the pool
 * can actually use the free device-local memory (Fase 6 step 4e). */
static uint64_t vulkan_pool_max_bytes(void) {
    /* Frozen at pool_configure (see there).  Recomputing it from the live
     * device-local usage made the cap shrink while the pool was still growing
     * (decode/prefill allocate more device-local memory as they run), which
     * tripped "pool cap hit" and aborted the seed mid-prefill. */
    if (g_pool_max_bytes_fixed != 0) return g_pool_max_bytes_fixed;
    uint64_t cap = 3ull * 1024ull * 1024ull * 1024ull;
    const uint64_t eff = vulkan_pool_effective_budget_bytes();
    if (eff > cap) cap = eff;
    return cap;
}

/* Device-local heap size of the selected device (the pool lives there). */
static uint64_t vulkan_pool_heap_bytes(void) {
    if (g_mem_props.memoryTypeCount == 0) return 0;
    const uint32_t mt = g_is_uma ? g_uma_mem_type : g_device_local_mem_type;
    if (mt == UINT32_MAX || mt >= g_mem_props.memoryTypeCount) return 0;
    const uint32_t heap = g_mem_props.memoryTypes[mt].heapIndex;
    if (heap >= g_mem_props.memoryHeapCount) return 0;
    return (uint64_t)g_mem_props.memoryHeaps[heap].size;
}

/* Auto pool budget from the device-local heap: what is left after the static
 * decode map (and any other device-local tensors) is in place, minus a reserve
 * for the driver + KV headroom + the targeted free VRAM.  Because the static
 * decode map is staged BEFORE the pool seeds (engine), the tracked usage at
 * pool-configure time already includes the ~8 GiB of non-expert weights, so
 * this targets a consistent free-VRAM reserve on any GPU size instead of a
 * blunt fraction of the heap.  The pool's OWN bytes (g_pool_total_bytes) are
 * excluded from the tracked usage so the budget stays constant while the pool
 * grows.  Tunables: DS4_VULKAN_POOL_RESERVE_GB (default 0.5), and legacy
 * DS4_VULKAN_POOL_AUTO_FRACTION / DS4_VULKAN_POOL_AUTO_GB (clamped). */
static uint64_t vulkan_pool_auto_budget_bytes(void) {
    const uint64_t heap = vulkan_pool_heap_bytes();
    if (heap == 0) return 0;
    double reserve_gb = 0.4;
    const char *renv = getenv("DS4_VULKAN_POOL_RESERVE_GB");
    if (renv && renv[0]) {
        const double v = atof(renv);
        if (v >= 0.0) reserve_gb = v;
    }
    const uint64_t reserve =
        (uint64_t)(reserve_gb * 1024.0 * 1024.0 * 1024.0);
    const uint64_t used = g_device_local_bytes >= g_pool_total_bytes ?
                          g_device_local_bytes - g_pool_total_bytes : 0;
    if (used + reserve >= heap) return 0;
    uint64_t bytes = heap - used - reserve;
    const char *fenv = getenv("DS4_VULKAN_POOL_AUTO_FRACTION");
    if (fenv && fenv[0]) {
        const double v = atof(fenv);
        if (v > 0.0 && v < 1.0) {
            const uint64_t fb = (uint64_t)((double)heap * v);
            if (fb < bytes) bytes = fb;
        }
    }
    const char *genv = getenv("DS4_VULKAN_POOL_AUTO_GB");
    if (genv && genv[0]) {
        const double gb = atof(genv);
        if (gb > 0.0) {
            const uint64_t gb_bytes =
                (uint64_t)(gb * 1024.0 * 1024.0 * 1024.0);
            if (gb_bytes < bytes) bytes = gb_bytes;
        }
    }
    return bytes;
}

/* Bytes the pool may hold: the CLI budget if set, at least the auto
 * VRAM-derived budget so the free device-local memory is actually used. */
static uint64_t vulkan_pool_effective_budget_bytes(void) {
    uint64_t budget = 0;
    if (g_pool_budget != 0 && g_pool_per_expert_bytes != 0) {
        budget = (uint64_t)g_pool_budget * g_pool_per_expert_bytes;
    }
    const uint64_t auto_bytes = vulkan_pool_auto_budget_bytes();
    if (auto_bytes > budget) budget = auto_bytes;
    return budget;
}

/* Persistent scratch tensors for the Q8_0 prequant path (and helpers that
 * need a temporary buffer).  Grown on demand, freed at cleanup. */
static ds4_gpu_tensor *g_scratch_a = NULL;
static ds4_gpu_tensor *g_scratch_b = NULL;
static ds4_gpu_tensor *g_scratch_c = NULL;
static ds4_gpu_tensor *g_scratch_d = NULL;

/* Persistent staging buffer for batched expert-pool stores (one submit per
 * layer seed instead of a one-shot submit per gate/up/down range). */
static ds4_gpu_tensor *g_pool_staging = NULL;

/* Persistent scratch holding the expert-grouped pair order for the routed MoE
 * (see moe_group.hlsl); grows to the largest pair count seen. */
static ds4_gpu_tensor *g_moe_order = NULL;

/* --- per-device context (M4, SPECS_MGPU.md) ---------------------------- */

/* Every device-dependent handle the single-device backend kept in a global.
 * The compute/staging/dispatch code keeps using the globals; switching tiers
 * saves them into the leaving slot and loads them from the entering slot, so
 * that (large) code is unchanged.  Vulkan objects (pipelines, command pools,
 * memories, fences) are per-VkDevice, hence per tier. */
struct ds4_vk_dev_ctx {
    VkPhysicalDevice phys;
    VkDevice         device;
    VkQueue          queue;
    uint32_t         queue_family;
    uint32_t         device_index;
    VkPhysicalDeviceProperties       props;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t host_visible_mem_type;
    uint32_t device_local_mem_type;
    uint32_t uma_mem_type;
    int      is_uma;
    uint64_t host_pointer_align;
    uint32_t ts_valid_bits;
    PFN_vkCmdSetCheckpointNV       pfn_cmd_checkpoint;
    PFN_vkGetQueueCheckpointDataNV pfn_get_checkpoint;
    int      dbg_checkpoints;
    PFN_vkGetBufferDeviceAddressKHR vk_bda;
    int      has_int_dot;
    int      attn_out_autotuned;
    VkCommandPool   cmd_pool;
    VkCommandBuffer cmd[2];
    VkFence         cmd_fence[2];
    VkFence         cb_fence[2];
    int             cmd_i;
    bool            commands_active;
    VkCommandPool   worker_pool;
    VkCommandBuffer worker_cb;
    VkFence         worker_fence;
    int             worker_fence_pending;
    VkFence         readback_fence;
    int             readback_fence_pending;
    VkQueryPool     ts_pool;
    float           ts_period;
    int             ts_on;
    int             ts_started;
    int             ts_router_written;
    int             ts_verbose;
    VkDescriptorPool      desc_pool;
    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout      pipe_layout;
    uint32_t              desc_sets_allocated;
    int                   vulkan_device_dirty;
    VkPipeline     pipes[DS4_VK_PIPE_COUNT];
    VkShaderModule mods[DS4_VK_PIPE_COUNT];
    struct ds4_vk_model_window model_windows[DS4_VK_MAX_MODEL_WINDOWS];
    uint32_t model_window_count;
    struct ds4_vk_model_window weight_cache[DS4_VK_MAX_MODEL_WINDOWS];
    uint32_t weight_cache_count;
    int      weight_cache_ready;
    int      dev_mode;   /* 0 static (weights fully resident), 1 dynamic */
    uint64_t device_local_bytes;
    ds4_gpu_tensor *scratch_a;
    ds4_gpu_tensor *scratch_b;
    ds4_gpu_tensor *scratch_c;
    ds4_gpu_tensor *scratch_d;
    ds4_gpu_tensor *pool_staging;
    ds4_gpu_tensor *moe_order;
    struct ds4_vk_pool_layer pool_layers[DS4_VK_POOL_MAX_LAYERS];
    uint32_t pool_slots_per_layer;
    uint32_t pool_budget;
    uint32_t pool_layer_count;
    uint64_t pool_per_expert_bytes;
    uint64_t        pool_total_bytes;
    uint64_t        pool_max_bytes_fixed;
    int      pool_ready;
};

static struct ds4_vk_dev_ctx g_vk_ctx[DS4_MAX_GPUS];
static int g_vk_ctx_count = 0;
static int g_vk_ctx_active = -1;
static int g_dev_mode = 0;   /* dev_mode of the active tier (see ds4_vk_dev_ctx) */

/* True once ds4_gpu_init_multi brought up more than one device.  The engine
 * uses this to force the non-static per-layer window path (the shared expert
 * pool is single-device) across the tiers. */
static int g_vulkan_multi_tier = 0;
extern "C" int ds4_vulkan_multi_tier_active(void) {
    return g_vulkan_multi_tier;
}

static void vulkan_ctx_save(struct ds4_vk_dev_ctx *c) {
    memset(c, 0, sizeof(*c));
    c->phys = g_phys;
    c->device = g_device;
    c->queue = g_queue;
    c->queue_family = g_queue_family;
    c->device_index = g_device_index;
    c->props = g_props;
    c->mem_props = g_mem_props;
    c->host_visible_mem_type = g_host_visible_mem_type;
    c->device_local_mem_type = g_device_local_mem_type;
    c->uma_mem_type = g_uma_mem_type;
    c->is_uma = g_is_uma;
    c->host_pointer_align = g_host_pointer_align;
    c->ts_valid_bits = g_ts_valid_bits;
    c->pfn_cmd_checkpoint = g_pfnCmdSetCheckpointNV;
    c->pfn_get_checkpoint = g_pfnGetQueueCheckpointDataNV;
    c->dbg_checkpoints = g_dbg_checkpoints;
    c->vk_bda = g_vk_bda;
    c->has_int_dot = g_has_int_dot;
    c->attn_out_autotuned = g_attn_out_autotuned;
    c->cmd_pool = g_cmd_pool;
    memcpy(c->cmd, g_cmd, sizeof(g_cmd));
    memcpy(c->cmd_fence, g_cmd_fence, sizeof(g_cmd_fence));
    memcpy(c->cb_fence, g_cb_fence, sizeof(g_cb_fence));
    c->cmd_i = g_cmd_i;
    c->commands_active = g_commands_active;
    c->worker_pool = g_worker_pool;
    c->worker_cb = g_worker_cb;
    c->worker_fence = g_worker_fence;
    c->worker_fence_pending = g_worker_fence_pending;
    c->readback_fence = g_readback_fence;
    c->readback_fence_pending = g_readback_fence_pending;
    c->ts_pool = g_ts_pool;
    c->ts_period = g_ts_period;
    c->ts_on = g_ts_on;
    c->ts_started = g_ts_started;
    c->ts_router_written = g_ts_router_written;
    c->ts_verbose = g_ts_verbose;
    c->desc_pool = g_desc_pool;
    c->desc_layout = g_desc_layout;
    c->pipe_layout = g_pipe_layout;
    c->desc_sets_allocated = g_desc_sets_allocated;
    c->vulkan_device_dirty = g_vulkan_device_dirty;
    memcpy(c->pipes, g_pipes, sizeof(g_pipes));
    memcpy(c->mods, g_mods, sizeof(g_mods));
    memcpy(c->model_windows, g_model_windows, sizeof(g_model_windows));
    c->model_window_count = g_model_window_count;
    memcpy(c->weight_cache, g_weight_cache, sizeof(g_weight_cache));
    c->weight_cache_count = g_weight_cache_count;
    c->weight_cache_ready = g_weight_cache_ready;
    c->dev_mode = g_dev_mode;
    c->device_local_bytes = g_device_local_bytes;
    c->scratch_a = g_scratch_a;
    c->scratch_b = g_scratch_b;
    c->scratch_c = g_scratch_c;
    c->scratch_d = g_scratch_d;
    c->pool_staging = g_pool_staging;
    c->moe_order = g_moe_order;
    memcpy(c->pool_layers, g_pool_layers, sizeof(g_pool_layers));
    c->pool_slots_per_layer = g_pool_slots_per_layer;
    c->pool_budget = g_pool_budget;
    c->pool_layer_count = g_pool_layer_count;
    c->pool_per_expert_bytes = g_pool_per_expert_bytes;
    c->pool_total_bytes = g_pool_total_bytes;
    c->pool_max_bytes_fixed = g_pool_max_bytes_fixed;
    c->pool_ready = g_pool_ready;
}

static void vulkan_ctx_load(const struct ds4_vk_dev_ctx *c) {
    g_phys = c->phys;
    g_device = c->device;
    g_queue = c->queue;
    g_queue_family = c->queue_family;
    g_device_index = c->device_index;
    g_props = c->props;
    g_mem_props = c->mem_props;
    g_host_visible_mem_type = c->host_visible_mem_type;
    g_device_local_mem_type = c->device_local_mem_type;
    g_uma_mem_type = c->uma_mem_type;
    g_is_uma = c->is_uma;
    g_host_pointer_align = c->host_pointer_align;
    g_ts_valid_bits = c->ts_valid_bits;
    g_pfnCmdSetCheckpointNV = c->pfn_cmd_checkpoint;
    g_pfnGetQueueCheckpointDataNV = c->pfn_get_checkpoint;
    g_dbg_checkpoints = c->dbg_checkpoints;
    g_vk_bda = c->vk_bda;
    g_has_int_dot = c->has_int_dot;
    g_attn_out_autotuned = c->attn_out_autotuned;
    g_cmd_pool = c->cmd_pool;
    memcpy(g_cmd, c->cmd, sizeof(g_cmd));
    memcpy(g_cmd_fence, c->cmd_fence, sizeof(g_cmd_fence));
    memcpy(g_cb_fence, c->cb_fence, sizeof(g_cb_fence));
    g_cmd_i = c->cmd_i;
    g_commands_active = c->commands_active;
    g_worker_pool = c->worker_pool;
    g_worker_cb = c->worker_cb;
    g_worker_fence = c->worker_fence;
    g_worker_fence_pending = c->worker_fence_pending;
    g_readback_fence = c->readback_fence;
    g_readback_fence_pending = c->readback_fence_pending;
    g_ts_pool = c->ts_pool;
    g_ts_period = c->ts_period;
    g_ts_on = c->ts_on;
    g_ts_started = c->ts_started;
    g_ts_router_written = c->ts_router_written;
    g_ts_verbose = c->ts_verbose;
    g_desc_pool = c->desc_pool;
    g_desc_layout = c->desc_layout;
    g_pipe_layout = c->pipe_layout;
    g_desc_sets_allocated = c->desc_sets_allocated;
    g_vulkan_device_dirty = c->vulkan_device_dirty;
    memcpy(g_pipes, c->pipes, sizeof(g_pipes));
    memcpy(g_mods, c->mods, sizeof(g_mods));
    memcpy(g_model_windows, c->model_windows, sizeof(g_model_windows));
    g_model_window_count = c->model_window_count;
    memcpy(g_weight_cache, c->weight_cache, sizeof(g_weight_cache));
    g_weight_cache_count = c->weight_cache_count;
    g_weight_cache_ready = c->weight_cache_ready;
    g_dev_mode = c->dev_mode;
    g_device_local_bytes = c->device_local_bytes;
    g_scratch_a = c->scratch_a;
    g_scratch_b = c->scratch_b;
    g_scratch_c = c->scratch_c;
    g_scratch_d = c->scratch_d;
    g_pool_staging = c->pool_staging;
    g_moe_order = c->moe_order;
    memcpy(g_pool_layers, c->pool_layers, sizeof(g_pool_layers));
    g_pool_slots_per_layer = c->pool_slots_per_layer;
    g_pool_budget = c->pool_budget;
    g_pool_layer_count = c->pool_layer_count;
    g_pool_per_expert_bytes = c->pool_per_expert_bytes;
    g_pool_total_bytes = c->pool_total_bytes;
    g_pool_max_bytes_fixed = c->pool_max_bytes_fixed;
    g_pool_ready = c->pool_ready;
}

/* Select the logical tier that subsequent ds4_gpu_* calls operate on. */
extern "C" int ds4_vulkan_set_current_device(int tier) {
    if (tier < 0 || tier >= g_vk_ctx_count) return 1;
    if (tier == g_vk_ctx_active) return 0;
    vulkan_trace("tier.switch from=%d to=%d", g_vk_ctx_active, tier);
    if (g_commands_active) {
        /* A tier switch must happen between command scopes: a scope left open
         * on the leaving tier would be saved into its context and never
         * submitted, corrupting the next use of that tier.  Submit it first
         * (loudly, never silently), then let the next begin_commands record on
         * the sibling CB. */
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "tier %d->%d: open command scope at switch; submitting it\n",
                g_vk_ctx_active, tier);
        if (!vulkan_cb_submit()) {
            g_commands_active = false;
            return 1;
        }
        g_commands_active = false;
        g_cmd_i = 1 - g_cmd_i;
    }
    if (g_vk_ctx_active >= 0) vulkan_ctx_save(&g_vk_ctx[g_vk_ctx_active]);
    vulkan_ctx_load(&g_vk_ctx[tier]);
    g_vk_ctx_active = tier;
    return 0;
}

/* Per-tensor device handle.  tensor->ptr points at one of these (heap). */
struct ds4_vulkan_tensor {
    VkBuffer         buffer;
    VkDeviceMemory   memory;
    VkDeviceSize     offset;      /* byte offset of this tensor within buffer */
    uint64_t         bytes;
    int              owner;       /* owns buffer+memory */
    int              device_local; /* memory lives in the device-local heap */
    int              tier;         /* logical device tier that owns the buffer */
    unsigned char   *host_map;    /* persistent mapping of memory */
};

/* --- helpers ----------------------------------------------------------- */

static void vulkan_log_vk(VkResult rc, const char *what) {
    if (rc != VK_SUCCESS) {
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX "%s failed: %d\n", what, (int)rc);
    }
}

static struct ds4_vulkan_tensor *vulkan_tensor_handle(
        const ds4_gpu_tensor *t) {
    return t ? (struct ds4_vulkan_tensor *)t->ptr : NULL;
}

static double vulkan_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* Unified per-token trace (DS4_TRACE).  Timestamp is CLOCK_MONOTONIC in
 * seconds, the same clock the engine's now_sec() uses, so engine and backend
 * events interleave in one timeline.  Writes to stderr from whatever thread
 * calls it (the async expert-load worker included). */
static void vulkan_trace(const char *fmt, ...) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("DS4_TRACE");
        on = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    if (!on) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "TRACE t=%.3f gpu ", vulkan_now_ms() / 1000.0);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

/* Wall-clock ms since epoch, to correlate dispatch timestamps with the
 * amdgpu fault timestamps in dmesg (CLOCK_REALTIME). */
static double vulkan_wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int vulkan_device_compute_queue(VkPhysicalDevice phys,
                                       uint32_t *family_out) {
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, NULL);
    if (family_count == 0) return 0;
    VkQueueFamilyProperties *families = (VkQueueFamilyProperties *)calloc(
            family_count, sizeof(*families));
    if (!families) return 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, families);
    uint32_t compute_family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; i++) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_family = i;
            break;
        }
    }
    free(families);
    if (compute_family == UINT32_MAX) return 0;
    if (family_out) *family_out = compute_family;
    return 1;
}

/* Probe the real PCIe transfer bandwidth (H2D + D2H) of a physical device by
 * creating a minimal device and timing a vkCmdCopyBuffer in both directions.
 * Returns the slowest direction's bandwidth in GB/s (the bottleneck for SSD
 * streaming), or 0.0 on failure.  Mirrors vulkan/tools/vkbench: the sysfs
 * current_link_width reports the max link width, not the negotiated one, and
 * is misleading on the 4x RX 6900 XT server (3 cards report "16x" but run at
 * ~0.8 GB/s = x1, measured).  A device with no compute queue probes 0. */
/* Measured host<->device bandwidth cache, keyed by physical device handle.
 * Filled by ds4_vulkan_probe_devices() and reused by vulkan_pick_device() so
 * the throwaway-device probe runs at most once per GPU. */
struct vulkan_bw_cache_entry { VkPhysicalDevice phys; double bw; };
static struct vulkan_bw_cache_entry g_bw_cache[DS4_VK_MAX_DEVICES];
static int g_bw_cache_n = 0;

static double vulkan_probe_transfer_bw_uncached(VkPhysicalDevice phys) {
    uint32_t qfamily = UINT32_MAX;
    if (!vulkan_device_compute_queue(phys, &qfamily)) return 0.0;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qfamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &dci, NULL, &dev) != VK_SUCCESS) return 0.0;

    VkQueue queue;
    vkGetDeviceQueue(dev, qfamily, 0, &queue);

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint32_t dev_type = UINT32_MAX, host_type = UINT32_MAX, host_fallback = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if (dev_type == UINT32_MAX && (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) dev_type = i;
        if (host_fallback == UINT32_MAX && (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            host_fallback = i;
        if (host_type == UINT32_MAX && (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) host_type = i;
    }
    if (host_type == UINT32_MAX) host_type = host_fallback;

    const VkDeviceSize size = 16u * 1024u * 1024u;
    VkBuffer dst = VK_NULL_HANDLE, stg = VK_NULL_HANDLE;
    VkDeviceMemory dmem = VK_NULL_HANDLE, smem = VK_NULL_HANDLE;
    VkCommandPool cpool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    double bw = 0.0;

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    do {
        if (dev_type == UINT32_MAX || host_type == UINT32_MAX) break;
        if (vkCreateBuffer(dev, &bci, NULL, &dst) != VK_SUCCESS ||
            vkCreateBuffer(dev, &bci, NULL, &stg) != VK_SUCCESS) break;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev, dst, &req);
        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = dev_type;
        if (vkAllocateMemory(dev, &mai, NULL, &dmem) != VK_SUCCESS) break;
        mai.memoryTypeIndex = host_type;
        if (vkAllocateMemory(dev, &mai, NULL, &smem) != VK_SUCCESS) break;
        if (vkBindBufferMemory(dev, dst, dmem, 0) != VK_SUCCESS ||
            vkBindBufferMemory(dev, stg, smem, 0) != VK_SUCCESS) break;
        void *map = NULL;
        if (vkMapMemory(dev, smem, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) break;
        for (uint64_t i = 0; i < size; i += 4096) ((unsigned char *)map)[i] = 1;
        vkUnmapMemory(dev, smem);

        VkCommandPoolCreateInfo cpci = {};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = qfamily;
        if (vkCreateCommandPool(dev, &cpci, NULL, &cpool) != VK_SUCCESS) break;
        VkCommandBufferAllocateInfo cai = {};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = cpool;
        cai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(dev, &cai, &cb) != VK_SUCCESS) break;
        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(dev, &fci, NULL, &fence) != VK_SUCCESS) break;

        VkBufferCopy bc = {};
        bc.srcOffset = 0;
        bc.dstOffset = 0;
        bc.size = size;
        double best = 0.0;
        for (int dir = 0; dir < 2; dir++) {
            best = 0.0;
            for (int it = 0; it < 3; it++) {
                VkCommandBufferBeginInfo bi = {};
                bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(cb, &bi);
                if (dir == 0) vkCmdCopyBuffer(cb, stg, dst, 1, &bc); /* H2D */
                else          vkCmdCopyBuffer(cb, dst, stg, 1, &bc); /* D2H */
                vkEndCommandBuffer(cb);
                double t0 = vulkan_now_ms();
                VkSubmitInfo si = {};
                si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                si.commandBufferCount = 1;
                si.pCommandBuffers = &cb;
                if (vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS) break;
                vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
                vkResetFences(dev, 1, &fence);
                double dt = (vulkan_now_ms() - t0) / 1000.0;
                if (dt > 0 && (best == 0.0 || dt < best)) best = dt;
            }
            if (best == 0.0) break;
            double dir_bw = (double)size / best / 1e9;
            if (dir == 0 || dir_bw < bw) bw = dir_bw;
        }
    } while (0);

    if (fence) vkDestroyFence(dev, fence, NULL);
    if (cpool) vkDestroyCommandPool(dev, cpool, NULL);
    if (stg) vkDestroyBuffer(dev, stg, NULL);
    if (dst) vkDestroyBuffer(dev, dst, NULL);
    if (smem) vkFreeMemory(dev, smem, NULL);
    if (dmem) vkFreeMemory(dev, dmem, NULL);
    vkDestroyDevice(dev, NULL);
    return bw;
}

static double vulkan_probe_transfer_bw(VkPhysicalDevice phys) {
    for (int i = 0; i < g_bw_cache_n; i++) {
        if (g_bw_cache[i].phys == phys) return g_bw_cache[i].bw;
    }
    const double bw = vulkan_probe_transfer_bw_uncached(phys);
    if (g_bw_cache_n < DS4_VK_MAX_DEVICES) {
        g_bw_cache[g_bw_cache_n].phys = phys;
        g_bw_cache[g_bw_cache_n].bw = bw;
        g_bw_cache_n++;
    }
    return bw;
}

/* PCI bus/device/function of a physical device (VK_EXT_pci_bus_info), so the
 * Vulkan index printed at startup can be mapped to the real slot.  Best-effort:
 * returns 0 (and zeros) when the extension is unavailable. */
static int vulkan_query_bdf_inst(VkInstance inst, VkPhysicalDevice phys,
                                 uint32_t *dom, uint32_t *bus,
                                 uint32_t *dev, uint32_t *fn) {
    if (dom) *dom = 0;
    if (bus) *bus = 0;
    if (dev) *dev = 0;
    if (fn) *fn = 0;
    if (!phys || inst == VK_NULL_HANDLE) return 0;
    PFN_vkGetPhysicalDeviceProperties2 p2 =
        (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
            inst, "vkGetPhysicalDeviceProperties2");
    if (!p2) {
        p2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
            inst, "vkGetPhysicalDeviceProperties2KHR");
    }
    if (!p2) return 0;
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(phys, NULL, &n, NULL);
    if (n == 0) return 0;
    VkExtensionProperties *e = (VkExtensionProperties *)malloc(sizeof(*e) * n);
    if (!e) return 0;
    vkEnumerateDeviceExtensionProperties(phys, NULL, &n, e);
    int has = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (strcmp(e[i].extensionName, "VK_EXT_pci_bus_info") == 0) {
            has = 1;
            break;
        }
    }
    free(e);
    if (!has) return 0;
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;
    VkPhysicalDeviceProperties2 props2;
    memset(&props2, 0, sizeof(props2));
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &pci;
    p2(phys, &props2);
    if (dom) *dom = pci.pciDomain;
    if (bus) *bus = pci.pciBus;
    if (dev) *dev = pci.pciDevice;
    if (fn) *fn = pci.pciFunction;
    return 1;
}

static int vulkan_query_bdf(VkPhysicalDevice phys, uint32_t *dom, uint32_t *bus,
                            uint32_t *dev, uint32_t *fn) {
    return vulkan_query_bdf_inst(g_instance, phys, dom, bus, dev, fn);
}

static int vulkan_pick_device_ex(int forced_override) {
    uint32_t count = 0;
    VkResult rc = vkEnumeratePhysicalDevices(g_instance, &count, NULL);
    if (rc != VK_SUCCESS || count == 0) {
        if (rc == VK_SUCCESS) rc = VK_ERROR_INITIALIZATION_FAILED;
        vulkan_log_vk(rc, "vkEnumeratePhysicalDevices");
        return 0;
    }
    VkPhysicalDevice *devices = (VkPhysicalDevice *)calloc(count, sizeof(*devices));
    if (!devices) return 0;
    rc = vkEnumeratePhysicalDevices(g_instance, &count, devices);
    if (rc != VK_SUCCESS) {
        free(devices);
        vulkan_log_vk(rc, "vkEnumeratePhysicalDevices");
        return 0;
    }
    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    /* An optional DS4_VULKAN_DEVICE_INDEX forces a specific physical device
     * (testing).  Otherwise score each usable device by the measured PCIe
     * transfer bandwidth (SSD streaming is host->GPU, and the real link speed
     * is the bottleneck -- sysfs reports the max link width, not the
     * negotiated one, so it is misleading on the 4x RX 6900 XT server).  Pick
     * the fastest device; >= 10 GB/s is a healthy x16 link, so stop probing
     * early.  llvmpipe (CPU) is skipped: it would silently fall back to
     * software rendering.  Fall back to the first usable device if no probe
     * succeeds. */
    const char *dev_env = getenv("DS4_VULKAN_DEVICE_INDEX");
    int forced = forced_override;
    if (forced < 0 && dev_env) {
        char *end = NULL;
        long v = strtol(dev_env, &end, 10);
        if (end != dev_env && v >= 0 && v < (long)count) forced = (int)v;
    }
    uint32_t chosen_idx = 0;
    if (forced >= 0) {
        chosen = devices[forced];
        chosen_idx = (uint32_t)forced;
    } else {
        VkPhysicalDevice first_usable = VK_NULL_HANDLE;
        uint32_t first_usable_idx = 0;
        double best_bw = 0.0;
        for (uint32_t i = 0; i < count; i++) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[i], &props);
            if (props.apiVersion < VK_API_VERSION_1_0) continue;
            if (first_usable == VK_NULL_HANDLE) {
                first_usable = devices[i];
                first_usable_idx = i;
            }
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ||
                props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
                continue;
            }
            double bw = vulkan_probe_transfer_bw(devices[i]);
            if (bw > best_bw) {
                best_bw = bw;
                chosen = devices[i];
                chosen_idx = i;
            }
            if (bw >= 10.0) break;
        }
        if (chosen == VK_NULL_HANDLE) {
            chosen = first_usable;
            chosen_idx = first_usable_idx;
        }
        if (getenv("DS4_VULKAN_DEBUG_BINDS") != NULL) {
            fprintf(stderr, "ds4: Vulkan debug: picked device index %u (bw %.1f GB/s)\n",
                    (unsigned)chosen_idx, best_bw);
        }
    }
    free(devices);
    if (chosen == VK_NULL_HANDLE) return 0;
    g_device_index = chosen_idx;

    g_phys = chosen;
    vkGetPhysicalDeviceProperties(g_phys, &g_props);
    vkGetPhysicalDeviceMemoryProperties(g_phys, &g_mem_props);

    /* Prefer a host-visible AND host-coherent type so host reads/writes of
     * the shared memory need no explicit flush/invalidate after compute
     * kernels.  Fall back to plain host-visible. */
    g_host_visible_mem_type = UINT32_MAX;
    uint32_t fallback = UINT32_MAX;
    for (uint32_t i = 0; i < g_mem_props.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags f = g_mem_props.memoryTypes[i].propertyFlags;
        if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        if (fallback == UINT32_MAX) fallback = i;
        if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
            g_host_visible_mem_type = i;
            break;
        }
    }
    if (g_host_visible_mem_type == UINT32_MAX) {
        g_host_visible_mem_type = fallback;
    }

    /* Fase 6 step 4a: pick the device-local tiers.  Prefer a pure
     * DEVICE_LOCAL type (discrete GPU VRAM, no host map) for weight
     * residency; on UMA/APU the DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT
     * type IS the device-local heap and no staging/copy is needed.
     *
     * UMA is detected by heap size, not by mere type existence: on a discrete
     * GPU the DEVICE_LOCAL|HOST_VISIBLE type is the small BAR carveout
     * (~256 MB) riding on a separate host heap, while on an APU it is the
     * dominant system-memory heap.  Only when the UMA type's heap is at
     * least as large as the biggest pure device-local heap do we treat the
     * device as UMA (allocating weights there with a direct host map). */
    g_device_local_mem_type = UINT32_MAX;
    g_uma_mem_type = UINT32_MAX;
    for (uint32_t i = 0; i < g_mem_props.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags f = g_mem_props.memoryTypes[i].propertyFlags;
        if (!(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            if (g_uma_mem_type == UINT32_MAX) g_uma_mem_type = i;
            continue;
        }
        if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            if (g_device_local_mem_type == UINT32_MAX) g_device_local_mem_type = i;
        }
    }
    g_is_uma = 0;
    if (g_uma_mem_type != UINT32_MAX) {
        const uint32_t uheap = g_mem_props.memoryTypes[g_uma_mem_type].heapIndex;
        const VkDeviceSize uma_size =
            g_mem_props.memoryHeaps[uheap].size;
        VkDeviceSize dl_size = 0;
        if (g_device_local_mem_type != UINT32_MAX) {
            const uint32_t dheap =
                g_mem_props.memoryTypes[g_device_local_mem_type].heapIndex;
            dl_size = g_mem_props.memoryHeaps[dheap].size;
        }
        /* APU: the shared heap is the largest device-local heap. */
        g_is_uma = (g_device_local_mem_type == UINT32_MAX ||
                    uma_size >= dl_size);
    }
    if (getenv("DS4_VULKAN_DEBUG_BINDS") != NULL) {
        fprintf(stderr,
                "ds4: Vulkan debug: mem types host=%u device_local=%u uma=%u is_uma=%d uma_heap=%.1fGiB dl_heap=%.1fGiB\n",
                g_host_visible_mem_type, g_device_local_mem_type,
                g_uma_mem_type, g_is_uma,
                g_uma_mem_type != UINT32_MAX
                    ? (double)g_mem_props.memoryHeaps[
                          g_mem_props.memoryTypes[g_uma_mem_type].heapIndex].size /
                          1073741824.0
                    : 0.0,
                g_device_local_mem_type != UINT32_MAX
                    ? (double)g_mem_props.memoryHeaps[
                          g_mem_props.memoryTypes[g_device_local_mem_type].heapIndex].size /
                          1073741824.0
                    : 0.0);
    }
    return 1;
}

static int vulkan_pick_device(void) {
    return vulkan_pick_device_ex(-1);
}

static VkResult vulkan_create_device(void) {
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &family_count, NULL);
    VkQueueFamilyProperties *families = (VkQueueFamilyProperties *)calloc(
            family_count, sizeof(*families));
    if (!families) return VK_ERROR_OUT_OF_HOST_MEMORY;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &family_count, families);
    int32_t compute_family = -1;
    for (uint32_t i = 0; i < family_count; i++) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_family = (int32_t)i;
            break;
        }
    }
    if (compute_family >= 0) {
        g_ts_valid_bits = families[compute_family].timestampValidBits;
    }
    free(families);
    if (compute_family < 0) return VK_ERROR_INITIALIZATION_FAILED;
    g_queue_family = (uint32_t)compute_family;

    /* Device extensions: VK_KHR_buffer_device_address (when present) lets the
     * debug env DS4_VULKAN_DEBUG_BINDS print the VA of each allocation, so a
     * GPUVM fault address can be mapped to a buffer in the same run.  Gated
     * behind the debug env so the default VA layout is the production one
     * (with BDA the driver packs buffers into a dense VA range, which masks
     * the over-read fault we are chasing). */
    const char *bda_ext = VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME;
    const char *intdot_ext =
        VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
    const char *chkpt_ext =
        VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME;
    const char *dev_exts[3];
    uint32_t n_dev_exts = 0;
    const int want_bda = getenv("DS4_VULKAN_DEBUG_BINDS") != NULL;
    const int want_chkpt = getenv("DS4_VULKAN_CHECKPOINTS") != NULL;
    VkPhysicalDeviceFeatures2 feat2 = {};
    VkPhysicalDeviceDescriptorIndexingFeatures desc_index = {};
    VkPhysicalDeviceShaderIntegerDotProductFeatures intdot_feat = {};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda_feat = {};
    VkPhysicalDeviceFeatures dev_feats = {};
    vkGetPhysicalDeviceFeatures(g_phys, &dev_feats);
    if (getenv("DS4_VULKAN_DEBUG_ROBUST") != NULL) {
        /* Clamp out-of-bounds shader accesses instead of faulting: a
         * diagnostic to confirm an OOB write and to let the decode proceed. */
        dev_feats.robustBufferAccess = VK_TRUE;
    }
    /* Probe device extensions once: VK_KHR_shader_integer_dot_product (always;
     * drives the Q8 preq v3 hardware-dot selection) and
     * VK_KHR_buffer_device_address (debug env only, see the comment above). */
    {
        uint32_t dext_n = 0;
        if (vkEnumerateDeviceExtensionProperties(g_phys, NULL, &dext_n, NULL) ==
            VK_SUCCESS && dext_n > 0) {
            VkExtensionProperties *dexts = (VkExtensionProperties *)calloc(
                    dext_n, sizeof(*dexts));
            if (dexts) {
                vkEnumerateDeviceExtensionProperties(g_phys, NULL, &dext_n, dexts);
                for (uint32_t i = 0; i < dext_n; i++) {
                    if (strcmp(dexts[i].extensionName, intdot_ext) == 0) {
                        dev_exts[n_dev_exts++] = intdot_ext;
                        g_has_int_dot = 1;
                    } else if (want_bda &&
                               strcmp(dexts[i].extensionName, bda_ext) == 0) {
                        dev_exts[n_dev_exts++] = bda_ext;
                    } else if (want_chkpt &&
                               strcmp(dexts[i].extensionName, chkpt_ext) == 0) {
                        dev_exts[n_dev_exts++] = chkpt_ext;
                        g_dbg_checkpoints = 1;
                    }
                }
                free(dexts);
            }
        }
    }
    int have_bda = 0;
    for (uint32_t i = 0; i < n_dev_exts; i++) {
        if (dev_exts[i] == bda_ext) have_bda = 1;
    }
    if (g_has_int_dot) {
        intdot_feat.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
        intdot_feat.shaderIntegerDotProduct = VK_TRUE;
    }
    if (have_bda) {
        bda_feat.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        bda_feat.bufferDeviceAddress = VK_TRUE;
    }
    /* The shared descriptor set layout marks every binding PARTIALLY_BOUND so
     * kernels may leave unused bindings unwritten.  That flag is only legal
     * when the feature is enabled at device creation; without it the driver
     * (NVIDIA in particular) ignores it and a dispatch with an unwritten
     * descriptor faults the GPU (device lost).  Chain: desc_index -> intdot
     * -> bda. */
    desc_index.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    desc_index.descriptorBindingPartiallyBound = VK_TRUE;
    void *chain = NULL;
    if (have_bda) {
        bda_feat.pNext = chain;
        chain = &bda_feat;
    }
    if (g_has_int_dot) {
        intdot_feat.pNext = chain;
        chain = &intdot_feat;
    }
    desc_index.pNext = chain;
    feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feat2.features = dev_feats;
    feat2.pNext = &desc_index;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = g_queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = n_dev_exts;
    dci.ppEnabledExtensionNames = n_dev_exts ? dev_exts : NULL;
    dci.pNext = &feat2;

    VkResult rc = vkCreateDevice(g_phys, &dci, NULL, &g_device);
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkCreateDevice");
        return rc;
    }
    vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);
    if (n_dev_exts > 0) {
        g_vk_bda = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(
                g_device, "vkGetBufferDeviceAddressKHR");
    }
    if (g_dbg_checkpoints == 1) {
        g_pfnCmdSetCheckpointNV = (PFN_vkCmdSetCheckpointNV)vkGetDeviceProcAddr(
                g_device, "vkCmdSetCheckpointNV");
        g_pfnGetQueueCheckpointDataNV =
            (PFN_vkGetQueueCheckpointDataNV)vkGetDeviceProcAddr(
                g_device, "vkGetQueueCheckpointDataNV");
        if (!g_pfnCmdSetCheckpointNV || !g_pfnGetQueueCheckpointDataNV) {
            g_dbg_checkpoints = 0;
        }
    }

    VkCommandPoolCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = g_queue_family;
    rc = vkCreateCommandPool(g_device, &cpci, NULL, &g_cmd_pool);
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkCreateCommandPool");
        return rc;
    }
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = g_cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 2;   /* double-buffered compute scopes */
    rc = vkAllocateCommandBuffers(g_device, &cbai, g_cmd);
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkAllocateCommandBuffers");
        return rc;
    }
    g_cmd_i = 0;
    g_cmd_fence[0] = VK_NULL_HANDLE;
    g_cmd_fence[1] = VK_NULL_HANDLE;
    return VK_SUCCESS;
}

/* --- compute pipeline ---------------------------------------------------- */

static VkShaderModule vulkan_create_shader_module(
        const uint32_t *code, size_t code_len) {
    VkShaderModuleCreateInfo smci = {};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code_len;
    smci.pCode = code;
    VkShaderModule mod = VK_NULL_HANDLE;
    VkResult rc = vkCreateShaderModule(g_device, &smci, NULL, &mod);
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkCreateShaderModule");
        return VK_NULL_HANDLE;
    }
    return mod;
}

static VkPipeline vulkan_create_compute_pipeline(
        VkShaderModule mod, VkPipelineLayout layout, const char *entry) {
    VkPipelineShaderStageCreateInfo ss = {};
    ss.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ss.module = mod;
    ss.pName = entry;

    VkComputePipelineCreateInfo cp = {};
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage = ss;
    cp.layout = layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult rc = vkCreateComputePipelines(g_device, VK_NULL_HANDLE, 1,
                                           &cp, NULL, &pipeline);
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkCreateComputePipelines");
        return VK_NULL_HANDLE;
    }
    return pipeline;
}

/* Create descriptor pool, set layout, pipeline layout and all compute
 * pipelines.  Idempotent; called from ds4_gpu_init.  Each pipeline maps to
 * one HLSL entry point (see gen_shaders.py and the files under
 * vulkan/shaders). */
static int vulkan_compute_init(void) {
    if (g_desc_layout != VK_NULL_HANDLE) return 1;
    if (g_device == VK_NULL_HANDLE) return 0;

    /* Storage-buffer bindings shared by every kernel: t0..t3 on bindings
     * 0..3 (a/b/c/model), u0..u3 on bindings 4..7 (out/out2/out3/out4).
     * Binding 3 is the model weight buffer.  PARTIALLY_BOUND lets kernels
     * that use only a subset of the bindings skip updating the rest. */
    const VkDescriptorSetLayoutBinding bindings[] = {
        { DS4_VK_BINDING_A, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
          VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { DS4_VK_BINDING_B, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
          VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { DS4_VK_BINDING_C, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
          VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { DS4_VK_BINDING_W, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
          VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { DS4_VK_BINDING_OUT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
          VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { DS4_VK_BINDING_OUT2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
          VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { DS4_VK_BINDING_OUT3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
          VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { DS4_VK_BINDING_OUT4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
          VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { DS4_VK_BINDING_OUT5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
          VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { DS4_VK_BINDING_TBL, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
          VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };
    const uint32_t n_bindings =
        (uint32_t)(sizeof(bindings) / sizeof(bindings[0]));
    VkDescriptorBindingFlags binding_flags[sizeof(bindings) /
                                           sizeof(bindings[0])];
    for (uint32_t bi = 0; bi < n_bindings; bi++) {
        binding_flags[bi] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    }
    const VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        NULL, n_bindings, binding_flags,
    };
    VkDescriptorSetLayoutCreateInfo dslci = {};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.pNext = &flags_info;
    dslci.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
    dslci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(g_device, &dslci, NULL,
                                    &g_desc_layout) != VK_SUCCESS) {
        vulkan_log_vk(VK_ERROR_INITIALIZATION_FAILED,
                      "vkCreateDescriptorSetLayout");
        return 0;
    }

    const VkPushConstantRange pcr = {
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(struct ds4_vk_params) };

    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &g_desc_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(g_device, &plci, NULL,
                               &g_pipe_layout) != VK_SUCCESS) {
        vulkan_log_vk(VK_ERROR_INITIALIZATION_FAILED,
                      "vkCreatePipelineLayout");
        return 0;
    }

    /* Descriptor pool: generous fixed count so an entire un-reset command
     * scope (a full 43-layer prefill pass records thousands of dispatches)
     * fits without exhausting.  Reset on drains and at ds4_gpu_synchronize. */
    VkDescriptorPoolSize pool_size = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128u * 1024u };
    VkDescriptorPoolCreateInfo dpci = {};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 8192;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(g_device, &dpci, NULL,
                               &g_desc_pool) != VK_SUCCESS) {
        vulkan_log_vk(VK_ERROR_INITIALIZATION_FAILED,
                      "vkCreateDescriptorPool");
        return 0;
    }

    /* Shader modules + pipelines: (module inc, entry point). */
    struct { const uint32_t *code; size_t len; const char *entry; } pipes[] = {
        { ds4_spv_unary_add, ds4_spv_unary_add_len, "add" },
        { ds4_spv_unary_add3, ds4_spv_unary_add3_len, "add3" },
        { ds4_spv_unary_swiglu, ds4_spv_unary_swiglu_len, "swiglu" },
        { ds4_spv_argmax, ds4_spv_argmax_len, "argmax" },
        { ds4_spv_sort_i32_rows_asc, ds4_spv_sort_i32_rows_asc_len,
          "sort_i32_rows_asc" },
        { ds4_spv_rms_norm_plain, ds4_spv_rms_norm_plain_len,
          "rms_norm_plain" },
        { ds4_spv_rms_norm_weight, ds4_spv_rms_norm_weight_len,
          "rms_norm_weight" },
        { ds4_spv_rope_tail, ds4_spv_rope_tail_len, "rope_tail" },
        { ds4_spv_matmul_q8_0, ds4_spv_matmul_q8_0_len, "matmul_q8_0" },
        { ds4_spv_matmul_q8_0_preq, ds4_spv_matmul_q8_0_preq_len,
          "matmul_q8_0_preq" },
        { ds4_spv_matmul_q8_0_kslice, ds4_spv_matmul_q8_0_kslice_len,
          "matmul_q8_0_kslice" },
        { ds4_spv_quantize_q8_0, ds4_spv_quantize_q8_0_len, "quantize_q8_0" },
        { ds4_spv_top1, ds4_spv_top1_len, "matmul_q8_0_top1" },
        { ds4_spv_f32_to_f16, ds4_spv_f32_to_f16_len, "f32_to_f16" },
        { ds4_spv_matmul_f16, ds4_spv_matmul_f16_len, "matmul_f16" },
        { ds4_spv_matmul_f16_pair_compressor_store,
          ds4_spv_matmul_f16_pair_compressor_store_len,
          "matmul_f16_pair_compressor_store" },
        { ds4_spv_matmul_f32, ds4_spv_matmul_f32_len, "matmul_f32" },
        { ds4_spv_embed_token_q8_0, ds4_spv_embed_token_q8_0_len,
          "embed_token_q8_0" },
        { ds4_spv_embed_tokens_q8_0, ds4_spv_embed_tokens_q8_0_len,
          "embed_tokens_q8_0" },
        { ds4_spv_embed_token_hc, ds4_spv_embed_token_hc_len,
          "embed_token_hc" },
        { ds4_spv_embed_tokens_hc, ds4_spv_embed_tokens_hc_len,
          "embed_tokens_hc" },
        { ds4_spv_fp8_kv_quantize, ds4_spv_fp8_kv_quantize_len,
          "fp8_kv_quantize" },
        { ds4_spv_store_raw_kv, ds4_spv_store_raw_kv_len, "store_raw_kv" },
        { ds4_spv_kv_fp8_store_raw, ds4_spv_kv_fp8_store_raw_len,
          "kv_fp8_store_raw" },
        { ds4_spv_attn_decode, ds4_spv_attn_decode_len, "attn_decode" },
        { ds4_spv_attn_indexed_decode, ds4_spv_attn_indexed_decode_len,
          "attn_indexed_decode" },
        { ds4_spv_attn_prefill, ds4_spv_attn_prefill_len, "attn_prefill" },
        { ds4_spv_attn_output_low_q8, ds4_spv_attn_output_low_q8_len,
          "attn_output_low_q8" },
        { ds4_spv_router_select, ds4_spv_router_select_len,
          "router_select" },
        { ds4_spv_moe_gate_up_mid_q8, ds4_spv_moe_gate_up_mid_q8_len,
          "moe_gate_up_mid_q8" },
        { ds4_spv_moe_down_q8, ds4_spv_moe_down_q8_len, "moe_down_q8" },
        { ds4_spv_moe_sum, ds4_spv_moe_sum_len, "moe_sum" },
        { ds4_spv_moe_gate_up_mid_iq2xxs, ds4_spv_moe_gate_up_mid_iq2xxs_len,
          "moe_gate_up_mid_iq2xxs" },
        { ds4_spv_moe_down_q2k, ds4_spv_moe_down_q2k_len, "moe_down_q2k" },
        { ds4_spv_fill_f32, ds4_spv_fill_f32_len, "fill_f32" },
        { ds4_spv_head_rms_norm, ds4_spv_head_rms_norm_len,
          "head_rms_norm" },
        { ds4_spv_head_rms_norm_rope_tail, ds4_spv_head_rms_norm_rope_tail_len,
          "head_rms_norm_rope_tail" },
        { ds4_spv_hc_split_sinkhorn, ds4_spv_hc_split_sinkhorn_len,
          "hc_split_sinkhorn" },
        { ds4_spv_hc_weighted_sum, ds4_spv_hc_weighted_sum_len,
          "hc_weighted_sum" },
        { ds4_spv_hc_expand, ds4_spv_hc_expand_len, "hc_expand" },
        { ds4_spv_hc_expand4, ds4_spv_hc_expand4_len, "hc_expand4" },
        { ds4_spv_hc_split_weighted_sum_fused,
          ds4_spv_hc_split_weighted_sum_fused_len,
          "hc_split_weighted_sum_fused" },
        { ds4_spv_output_hc_weights, ds4_spv_output_hc_weights_len,
          "output_hc_weights" },
        { ds4_spv_repeat_hc, ds4_spv_repeat_hc_len, "repeat_hc" },
        { ds4_spv_hc_expand4_half, ds4_spv_hc_expand4_half_len,
          "hc_expand4_half" },
        { ds4_spv_hc_expand4_add_half, ds4_spv_hc_expand4_add_half_len,
          "hc_expand4_add_half" },
        { ds4_spv_indexer_scores, ds4_spv_indexer_scores_len,
          "indexer_scores" },
        { ds4_spv_dsv4_indexer_qat, ds4_spv_dsv4_indexer_qat_len,
          "dsv4_indexer_qat" },
        { ds4_spv_indexer_topk, ds4_spv_indexer_topk_len, "indexer_topk" },
        { ds4_spv_indexer_top1_value, ds4_spv_indexer_top1_value_len,
          "indexer_top1_value" },
        { ds4_spv_topk_mask, ds4_spv_topk_mask_len, "topk_mask" },
        { ds4_spv_dspark_markov_argmax, ds4_spv_dspark_markov_argmax_len,
          "dspark_markov_argmax" },
        { ds4_spv_compressor_store, ds4_spv_compressor_store_len,
          "compressor_store" },
        { ds4_spv_compressor_set_rows, ds4_spv_compressor_set_rows_len,
          "compressor_set_rows" },
        { ds4_spv_compressor_prefill_pool, ds4_spv_compressor_prefill_pool_len,
          "compressor_prefill_pool" },
        { ds4_spv_compressor_update_pool, ds4_spv_compressor_update_pool_len,
          "compressor_update_pool" },
        { ds4_spv_compressor_shift_ratio4, ds4_spv_compressor_shift_ratio4_len,
          "compressor_shift_ratio4" },
        { ds4_spv_directional_steering_project,
          ds4_spv_directional_steering_project_len,
          "directional_steering_project" },
        { ds4_spv_matmul_q8_0_preq_v2, ds4_spv_matmul_q8_0_preq_v2_len,
          "matmul_q8_0_preq_v2" },
        { ds4_spv_matmul_q8_0_preq_v3, ds4_spv_matmul_q8_0_preq_v3_len,
          "matmul_q8_0_preq_v3" },
        { ds4_spv_moe_gate_up_mid_iq2xxs_v2,
          ds4_spv_moe_gate_up_mid_iq2xxs_v2_len,
          "moe_gate_up_mid_iq2xxs_v2" },
        { ds4_spv_moe_down_q2k_v2, ds4_spv_moe_down_q2k_v2_len,
          "moe_down_q2k_v2" },
        { ds4_spv_moe_gate_up_mid_q4k, ds4_spv_moe_gate_up_mid_q4k_len,
          "moe_gate_up_mid_q4k" },
        { ds4_spv_moe_down_q4k, ds4_spv_moe_down_q4k_len, "moe_down_q4k" },
        { ds4_spv_attn_output_low_q4k, ds4_spv_attn_output_low_q4k_len,
          "attn_output_low_q4k" },
        { ds4_spv_moe_gate_up_mid_mxfp4, ds4_spv_moe_gate_up_mid_mxfp4_len,
          "moe_gate_up_mid_mxfp4" },
        { ds4_spv_moe_down_mxfp4, ds4_spv_moe_down_mxfp4_len,
          "moe_down_mxfp4" },
        { ds4_spv_moe_gate_up_mid_mxfp4_v2,
          ds4_spv_moe_gate_up_mid_mxfp4_v2_len,
          "moe_gate_up_mid_mxfp4_v2" },
        { ds4_spv_moe_down_mxfp4_v2, ds4_spv_moe_down_mxfp4_v2_len,
          "moe_down_mxfp4_v2" },
        { ds4_spv_moe_group, ds4_spv_moe_group_len, "moe_group" },
        { ds4_spv_matmul_q4k, ds4_spv_matmul_q4k_len, "matmul_q4k" },
        { ds4_spv_matmul_q4_0, ds4_spv_matmul_q4_0_len, "matmul_q4_0" },
        { ds4_spv_attn_output_low_q8_v2, ds4_spv_attn_output_low_q8_v2_len,
          "attn_output_low_q8_v2" },
        { ds4_spv_attn_output_low_q8_v3, ds4_spv_attn_output_low_q8_v3_len,
          "attn_output_low_q8_v3" },
    };
    for (uint32_t i = 0; i < DS4_VK_PIPE_COUNT; i++) {
        snprintf(g_pipe_names[i], sizeof(g_pipe_names[i]), "%s",
                 pipes[i].entry);
        /* v3 uses OpSDot and can only be created when the device exposes
         * VK_KHR_shader_integer_dot_product; the selector never returns it
         * otherwise. */
        if (i == DS4_PIPE_MATMUL_Q8_0_PREQ_V3 && !g_has_int_dot) {
            g_mods[i] = VK_NULL_HANDLE;
            g_pipes[i] = VK_NULL_HANDLE;
            continue;
        }
        if (i == DS4_PIPE_ATTN_OUTPUT_LOW_Q8_V2 && !g_has_int_dot) {
            g_mods[i] = VK_NULL_HANDLE;
            g_pipes[i] = VK_NULL_HANDLE;
            continue;
        }
        if (i == DS4_PIPE_ATTN_OUTPUT_LOW_Q8_V3 && !g_has_int_dot) {
            g_mods[i] = VK_NULL_HANDLE;
            g_pipes[i] = VK_NULL_HANDLE;
            continue;
        }
        g_mods[i] = vulkan_create_shader_module(pipes[i].code, pipes[i].len);
        if (!g_mods[i]) return 0;
        g_pipes[i] = vulkan_create_compute_pipeline(g_mods[i], g_pipe_layout,
                                                    pipes[i].entry);
        if (!g_pipes[i]) return 0;
    }

    /* Debug GPU timestamps: create the query pool once if requested, the
     * overlap worker is off, and the queue family supports timestamps. */
    if (getenv("DS4_VULKAN_DEBUG_GPU_TS") != NULL &&
        getenv("DS4_VULKAN_DEBUG_SUBMIT") != NULL &&
        getenv("DS4_METAL_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP") != NULL &&
        g_ts_valid_bits > 0 && g_device != VK_NULL_HANDLE) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(g_phys, &props);
        g_ts_period = props.limits.timestampPeriod;
        VkQueryPoolCreateInfo qpci = {};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 128;
        if (g_ts_period > 0.0f &&
            vkCreateQueryPool(g_device, &qpci, NULL, &g_ts_pool) == VK_SUCCESS) {
            g_ts_on = 1;
            g_ts_verbose =
                getenv("DS4_VULKAN_DEBUG_GPU_TS_VERBOSE") != NULL;
            fprintf(stderr,
                    "ds4: GPU timestamp scope timing ON (period=%.1f ns%s)\n",
                    (double)g_ts_period, g_ts_verbose ? ", verbose" : "");
        }
    }

    return 1;
}

static void vulkan_compute_cleanup(void) {
    if (g_device == VK_NULL_HANDLE) return;
    for (uint32_t i = 0; i < DS4_VK_PIPE_COUNT; i++) {
        if (g_pipes[i]) {
            vkDestroyPipeline(g_device, g_pipes[i], NULL);
            g_pipes[i] = VK_NULL_HANDLE;
        }
        if (g_mods[i]) {
            vkDestroyShaderModule(g_device, g_mods[i], NULL);
            g_mods[i] = VK_NULL_HANDLE;
        }
    }
    if (g_pipe_layout) { vkDestroyPipelineLayout(g_device, g_pipe_layout, NULL); g_pipe_layout = VK_NULL_HANDLE; }
    if (g_desc_layout) { vkDestroyDescriptorSetLayout(g_device, g_desc_layout, NULL); g_desc_layout = VK_NULL_HANDLE; }
    if (g_desc_pool) { vkDestroyDescriptorPool(g_device, g_desc_pool, NULL); g_desc_pool = VK_NULL_HANDLE; }
    if (g_ts_pool) { vkDestroyQueryPool(g_device, g_ts_pool, NULL); g_ts_pool = VK_NULL_HANDLE; }
    g_ts_on = 0;
    g_ts_started = 0;
}

/* --- public contract: lifecycle ---------------------------------------- */

static void vulkan_autotune_moe_rows(void);
static void vulkan_autotune_attn_out(void);

int ds4_gpu_init(void) {
    if (g_device != VK_NULL_HANDLE) return 1; /* already initialized */

    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "ds4";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_0;

    /* Instance extensions: VK_KHR_get_physical_device_properties2 (when
     * present) lets the startup log report the PCI BDF via
     * VK_EXT_pci_bus_info.  Nothing else is required (Vulkan 1.0 core). */
    const char *inst_exts[1];
    uint32_t n_inst_exts = 0;
    {
        uint32_t n = 0;
        vkEnumerateInstanceExtensionProperties(NULL, &n, NULL);
        if (n > 0) {
            VkExtensionProperties *e =
                (VkExtensionProperties *)malloc(sizeof(*e) * n);
            if (e) {
                vkEnumerateInstanceExtensionProperties(NULL, &n, e);
                for (uint32_t i = 0; i < n; i++) {
                    if (strcmp(e[i].extensionName,
                               "VK_KHR_get_physical_device_properties2") == 0) {
                        inst_exts[n_inst_exts++] =
                            "VK_KHR_get_physical_device_properties2";
                        break;
                    }
                }
                free(e);
            }
        }
    }

    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = n_inst_exts;
    ici.ppEnabledExtensionNames = n_inst_exts ? inst_exts : NULL;

    VkResult rc = vkCreateInstance(&ici, NULL, &g_instance);
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkCreateInstance");
        return 0;
    }
    /* Multi-GPU discovery/classification (SPECS_MGPU.md M1): measure the real
     * host<->device PCIe bandwidth of every usable device and mark fast/slow.
     * Cached for the pick below (and for the future multi-device planner). */
    {
        ds4_vk_dev_info dinfo[DS4_VK_MAX_DEVICES];
        const int nd = ds4_vulkan_probe_devices(dinfo, DS4_VK_MAX_DEVICES);
        if (nd > 1) {
            for (int i = 0; i < nd; i++) {
                fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                        "device[%u] %s bdf=%s bw=%.1f GB/s vram=%.1f GiB %s\n",
                        dinfo[i].vk_index, dinfo[i].name, dinfo[i].bdf,
                        dinfo[i].bw_gbps,
                        (double)dinfo[i].vram_bytes / (1024.0 * 1024.0 * 1024.0),
                        !dinfo[i].usable ? "unusable"
                                         : (dinfo[i].is_fast ? "FAST" : "SLOW"));
            }
        }
    }
    if (!vulkan_pick_device()) {
        vkDestroyInstance(g_instance, NULL);
        g_instance = VK_NULL_HANDLE;
        return 0;
    }
    rc = vulkan_create_device();
    if (rc != VK_SUCCESS) {
        vkDestroyInstance(g_instance, NULL);
        g_instance = VK_NULL_HANDLE;
        return 0;
    }

    uint32_t bdf_dom = 0, bdf_bus = 0, bdf_dev = 0, bdf_fn = 0;
    const int have_bdf = vulkan_query_bdf(g_phys, &bdf_dom, &bdf_bus,
                                          &bdf_dev, &bdf_fn);
    char bdf_str[32];
    if (have_bdf) {
        snprintf(bdf_str, sizeof(bdf_str), "%04x:%02x:%02x.%x",
                 bdf_dom, bdf_bus, bdf_dev, bdf_fn);
    } else {
        snprintf(bdf_str, sizeof(bdf_str), "n/a");
    }
    fprintf(stderr, DS4_VULKAN_LOG_PREFIX
            "initialized: device[%u] %s bdf=%s (api 0x%08x)\n",
            g_device_index, g_props.deviceName, bdf_str,
            (unsigned)g_props.apiVersion);
    if (getenv("DS4_VULKAN_INFO_ONLY") != NULL) {
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "device index %u selected; exiting (DS4_VULKAN_INFO_ONLY)\n",
                g_device_index);
        exit(0);
    }
    ds4_vulkan_report_unavailable();
    g_gpu[0].device_id = 0;
    g_gpu[0].stream = NULL;
    g_gpu[0].cublas = NULL;
    g_n_gpus = 1;
    if (!vulkan_compute_init()) {
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "compute pipeline initialization failed\n");
        ds4_gpu_cleanup();
        return 0;
    }
    vulkan_autotune_moe_rows();
    vulkan_autotune_attn_out();
    vulkan_ctx_save(&g_vk_ctx[0]);
    g_vk_ctx_count = 1;
    g_vk_ctx_active = 0;
    return 1;
}

/* Create the shared VkInstance (idempotent) and log the M1 classification of
 * every physical device.  Shared by ds4_gpu_init (single) and the multi-device
 * init below. */
static int vulkan_instance_init(void) {
    if (g_instance != VK_NULL_HANDLE) return 1;
    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "ds4";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_0;

    const char *inst_exts[1];
    uint32_t n_inst_exts = 0;
    {
        uint32_t n = 0;
        vkEnumerateInstanceExtensionProperties(NULL, &n, NULL);
        if (n > 0) {
            VkExtensionProperties *e = (VkExtensionProperties *)malloc(sizeof(*e) * n);
            if (e) {
                vkEnumerateInstanceExtensionProperties(NULL, &n, e);
                for (uint32_t i = 0; i < n; i++) {
                    if (strcmp(e[i].extensionName,
                               "VK_KHR_get_physical_device_properties2") == 0) {
                        inst_exts[n_inst_exts++] =
                            "VK_KHR_get_physical_device_properties2";
                        break;
                    }
                }
                free(e);
            }
        }
    }
    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = n_inst_exts;
    ici.ppEnabledExtensionNames = n_inst_exts ? inst_exts : NULL;
    VkResult rc = vkCreateInstance(&ici, NULL, &g_instance);
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkCreateInstance");
        return 0;
    }
    ds4_vk_dev_info dinfo[DS4_VK_MAX_DEVICES];
    const int nd = ds4_vulkan_probe_devices(dinfo, DS4_VK_MAX_DEVICES);
    if (nd > 1) {
        for (int i = 0; i < nd; i++) {
            fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                    "device[%u] %s bdf=%s bw=%.1f GB/s vram=%.1f GiB %s\n",
                    dinfo[i].vk_index, dinfo[i].name, dinfo[i].bdf,
                    dinfo[i].bw_gbps,
                    (double)dinfo[i].vram_bytes / (1024.0 * 1024.0 * 1024.0),
                    !dinfo[i].usable ? "unusable"
                                     : (dinfo[i].is_fast ? "FAST" : "SLOW"));
        }
    }
    return 1;
}

/* Zero the device-dependent globals so the next device starts from a clean
 * slate (pipeline creation is gated on g_desc_layout == NULL). */
static void vulkan_ctx_clear(void) {
    struct ds4_vk_dev_ctx zero;
    memset(&zero, 0, sizeof(zero));
    zero.host_pointer_align = 4096;
    vulkan_ctx_load(&zero);
}

/* Multi-device init: one VkDevice per logical tier (the engine's placement
 * order), each with its own pipelines, command pool, scratch and pool state.
 * SPECS_MGPU.md M4. */
extern "C" int ds4_vulkan_init_multi(const ds4_gpu_config *cfg) {
    if (!cfg || cfg->n_gpus <= 1) return ds4_gpu_init();
    if (cfg->n_gpus > DS4_MAX_GPUS) return 0;
    if (!vulkan_instance_init()) return 0;

    g_vk_ctx_count = 0;
    g_vk_ctx_active = -1;
    g_n_gpus = 0;
    for (int tier = 0; tier < cfg->n_gpus; tier++) {
        const int phys_idx = cfg->device_indices[tier];
        vulkan_ctx_clear();
        /* Tag this tier as active for the whole bring-up so tensors the
         * autotune allocates/frees carry the right tier.  Leaving it at -1
         * made every such tensor default to tier 0; once g_vk_ctx_count > 1
         * (tier >= 2) freeing one switched the active context to tier 0, and
         * the closing ctx_save then wrote tier 0's state into this tier. */
        g_vk_ctx_active = tier;
        if (!vulkan_pick_device_ex(phys_idx)) {
            fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                    "tier %d: device index %d unavailable\n", tier, phys_idx);
            return 0;
        }
        if (vulkan_create_device() != VK_SUCCESS) return 0;
        uint32_t dom = 0, bus = 0, dev = 0, fn = 0;
        char bdf[32];
        if (vulkan_query_bdf(g_phys, &dom, &bus, &dev, &fn)) {
            snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.%x", dom, bus, dev, fn);
        } else {
            snprintf(bdf, sizeof(bdf), "n/a");
        }
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "tier %d: device[%u] %s bdf=%s (%.1f GiB)\n",
                tier, g_device_index, g_props.deviceName, bdf,
                (double)cfg->vram_bytes[tier] / (1024.0 * 1024.0 * 1024.0));
        if (!vulkan_compute_init()) {
            fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                    "tier %d: compute init failed\n", tier);
            return 0;
        }
        vulkan_autotune_moe_rows();
        vulkan_autotune_attn_out();
        g_dev_mode = cfg->dev_mode[tier];
        vulkan_ctx_save(&g_vk_ctx[tier]);
        g_vk_ctx_count = tier + 1;
        g_gpu[tier].device_id = phys_idx;
        g_gpu[tier].budget_bytes = cfg->vram_bytes[tier];
        g_n_gpus = tier + 1;
    }
    g_vk_ctx_active = -1;
    if (ds4_vulkan_set_current_device(0) != 0) return 0;
    g_vulkan_multi_tier = 1;
    return 1;
}

extern "C" void ds4_vulkan_stream_pool_reset(void);

void ds4_gpu_cleanup(void) {
    if (g_device != VK_NULL_HANDLE) {
        vulkan_device_wait();
        vulkan_compute_cleanup();
        ds4_vulkan_stream_pool_reset();
        if (g_scratch_a) { ds4_gpu_tensor_free(g_scratch_a); g_scratch_a = NULL; }
        if (g_scratch_b) { ds4_gpu_tensor_free(g_scratch_b); g_scratch_b = NULL; }
        if (g_scratch_c) { ds4_gpu_tensor_free(g_scratch_c); g_scratch_c = NULL; }
        if (g_scratch_d) { ds4_gpu_tensor_free(g_scratch_d); g_scratch_d = NULL; }
        if (g_pool_staging) { ds4_gpu_tensor_free(g_pool_staging); g_pool_staging = NULL; }
        if (g_moe_order) { ds4_gpu_tensor_free(g_moe_order); g_moe_order = NULL; }
        for (uint32_t i = 0; i < g_model_window_count; i++) {
            ds4_gpu_tensor_free(g_model_windows[i].tensor);
            g_model_windows[i].tensor = NULL;
            g_model_windows[i].buffer = VK_NULL_HANDLE;
        }
        g_model_window_count = 0;
        for (int i = 0; i < 2; i++) {
            if (g_cb_fence[i] != VK_NULL_HANDLE) {
                vkDestroyFence(g_device, g_cb_fence[i], NULL);
                g_cb_fence[i] = VK_NULL_HANDLE;
            }
            g_cmd_fence[i] = VK_NULL_HANDLE;
        }
        if (g_readback_fence != VK_NULL_HANDLE) {
            vkDestroyFence(g_device, g_readback_fence, NULL);
            g_readback_fence = VK_NULL_HANDLE;
        }
        if (g_worker_fence != VK_NULL_HANDLE) {
            vkDestroyFence(g_device, g_worker_fence, NULL);
            g_worker_fence = VK_NULL_HANDLE;
        }
        if (g_worker_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(g_device, g_worker_pool, NULL);
            g_worker_pool = VK_NULL_HANDLE;
        }
        vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
        vkDestroyDevice(g_device, NULL);
    }
    if (g_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(g_instance, NULL);
    }
    g_instance = VK_NULL_HANDLE;
    g_phys = VK_NULL_HANDLE;
    g_device = VK_NULL_HANDLE;
    g_queue = VK_NULL_HANDLE;
    g_cmd_pool = VK_NULL_HANDLE;
    g_cmd[0] = VK_NULL_HANDLE;
    g_cmd[1] = VK_NULL_HANDLE;
    g_worker_cb = VK_NULL_HANDLE;
    g_cmd_i = 0;
    g_commands_active = false;
    g_readback_fence_pending = 0;
    g_worker_fence_pending = 0;
    g_host_visible_mem_type = UINT32_MAX;
    g_model_window_count = 0;
}

/* Exposed to ds4_vulkan_compat.c for in-place tensor free. */
extern "C" void *ds4_vulkan_device_handle(void) {
    return (void *)g_device;
}

/* Release the device resources owned by a tensor (buffer, memory, mapping)
 * without freeing the ds4_gpu_tensor struct itself.  Used by
 * ds4_gpu_tensor_free_in_place from the compat layer.  Only owner tensors
 * destroy the shared buffer/memory; views leave them untouched. */
extern "C" void ds4_vulkan_tensor_release_device(ds4_gpu_tensor *t) {
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(t);
    if (!h || !h->owner) return;
    const int prev = g_vk_ctx_active;
    if (g_vk_ctx_count > 1 && h->tier != g_vk_ctx_active)
        ds4_vulkan_set_current_device(h->tier);
    if (h->buffer && g_device != VK_NULL_HANDLE) {
        if (h->host_map) vkUnmapMemory(g_device, h->memory);
        vkDestroyBuffer(g_device, h->buffer, NULL);
        if (h->memory) vkFreeMemory(g_device, h->memory, NULL);
        if (h->device_local) {
            if (h->bytes <= g_device_local_bytes) {
                g_device_local_bytes -= h->bytes;
            } else {
                g_device_local_bytes = 0;
            }
        }
        h->buffer = VK_NULL_HANDLE;
        h->memory = VK_NULL_HANDLE;
        h->host_map = NULL;
        h->offset = 0;
        h->bytes = 0;
        h->owner = 0;
    }
    /* Never leave the backend's selected tier changed by a free: doing so
     * silently desynced it from the engine's active tier (multi-GPU). */
    if (prev >= 0 && prev != g_vk_ctx_active)
        (void)ds4_vulkan_set_current_device(prev);
}

/* --- public contract: tensors ------------------------------------------ */

static ds4_gpu_tensor *vulkan_tensor_new(void) {
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    /* Stamp the logical tier the tensor is allocated on, mirroring the CUDA
     * backend where device_id is the tier.  The engine compares it against
     * the layer's home tier (placement[il+1]); leaving it 0 made
     * ds4_gpu_tensor_device() lie on multi-tier Vulkan. */
    if (t) t->device_id = g_vk_ctx_active >= 0 ? g_vk_ctx_active : 0;
    return t;
}

static struct ds4_vulkan_tensor *vulkan_handle_new(void) {
    struct ds4_vulkan_tensor *h = (struct ds4_vulkan_tensor *)calloc(1,
            sizeof(struct ds4_vulkan_tensor));
    if (h) h->tier = g_vk_ctx_active >= 0 ? g_vk_ctx_active : 0;
    return h;
}

static void vulkan_handle_free(struct ds4_vulkan_tensor *h) {
    free(h);
}

/* Create a VkBuffer of `bytes` in the given memory type and optionally map
 * it.  Returns the buffer handle or VK_NULL_HANDLE; out_mem and out_map
 * receive the memory handle and the persistent host mapping (out_map is
 * NULL when the memory type is not host-visible). */
static VkBuffer vulkan_create_buffer(uint64_t bytes, uint32_t mem_type,
                                     int want_map, VkDeviceMemory *out_mem,
                                     unsigned char **out_map) {
    if (g_device == VK_NULL_HANDLE || bytes == 0) return VK_NULL_HANDLE;

    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(g_device, &bci, NULL, &buffer) != VK_SUCCESS) {
        vulkan_log_vk(VK_ERROR_INITIALIZATION_FAILED, "vkCreateBuffer");
        return VK_NULL_HANDLE;
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_device, buffer, &req);

    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mem_type;
    if (vkAllocateMemory(g_device, &mai, NULL, out_mem) != VK_SUCCESS) {
        vulkan_log_vk(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vkAllocateMemory");
        vkDestroyBuffer(g_device, buffer, NULL);
        return VK_NULL_HANDLE;
    }
    if (vkBindBufferMemory(g_device, buffer, *out_mem, 0) != VK_SUCCESS) {
        vulkan_log_vk(VK_ERROR_INITIALIZATION_FAILED, "vkBindBufferMemory");
        vkFreeMemory(g_device, *out_mem, NULL);
        vkDestroyBuffer(g_device, buffer, NULL);
        return VK_NULL_HANDLE;
    }
    if (out_map) *out_map = NULL;
    if (want_map) {
        const VkMemoryPropertyFlags f =
            g_mem_props.memoryTypes[mem_type].propertyFlags;
        if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            if (vkMapMemory(g_device, *out_mem, 0, VK_WHOLE_SIZE, 0,
                            (void **)out_map) != VK_SUCCESS) {
                vulkan_log_vk(VK_ERROR_MEMORY_MAP_FAILED, "vkMapMemory");
                vkFreeMemory(g_device, *out_mem, NULL);
                vkDestroyBuffer(g_device, buffer, NULL);
                return VK_NULL_HANDLE;
            }
        }
    }
    return buffer;
}

/* Host-visible + host-coherent tensor buffer, mapped (legacy default tier:
 * scratch, activations, user tensors). */
static VkBuffer vulkan_create_host_buffer(
        uint64_t bytes, VkDeviceMemory *out_mem,
        unsigned char **out_map) {
    const uint32_t mem_type = (g_host_visible_mem_type != UINT32_MAX) ?
                              g_host_visible_mem_type : 0u;
    return vulkan_create_buffer(bytes, mem_type, 1, out_mem, out_map);
}

/* Shared tensor construction from a created buffer. */
static ds4_gpu_tensor *vulkan_tensor_wrap_buffer(VkBuffer buffer,
                                                 VkDeviceMemory mem,
                                                 unsigned char *map,
                                                 uint64_t bytes) {
    struct ds4_vulkan_tensor *h = vulkan_handle_new();
    if (!h) {
        if (map) vkUnmapMemory(g_device, mem);
        vkFreeMemory(g_device, mem, NULL);
        vkDestroyBuffer(g_device, buffer, NULL);
        return NULL;
    }
    h->buffer = buffer;
    h->memory = mem;
    h->offset = 0;
    h->bytes = bytes;
    h->owner = 1;
    h->host_map = map;
    /* Debug: zero host-visible allocations so any read of uninitialized memory
     * (a multi-GPU-only nondeterminism suspect) becomes deterministic. */
    if (map && getenv("DS4_VULKAN_ZERO_ALLOC") != NULL) memset(map, 0, bytes);

    ds4_gpu_tensor *t = vulkan_tensor_new();
    if (!t) {
        if (map) vkUnmapMemory(g_device, mem);
        vkFreeMemory(g_device, mem, NULL);
        vkDestroyBuffer(g_device, buffer, NULL);
        vulkan_handle_free(h);
        return NULL;
    }
    t->ptr = h;
    t->bytes = bytes;
    t->owner = 1;
    if (getenv("DS4_VULKAN_DEBUG_BINDS") != NULL && g_vk_bda != NULL) {
        static uint64_t dbg_alloc_id = 0;
        VkBufferDeviceAddressInfo bdai = {};
        bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bdai.buffer = buffer;
        const uint64_t va = g_vk_bda(g_device, &bdai);
        Dl_info di = {};
        const char *caller = "?";
        if (dladdr(__builtin_return_address(0), &di) != 0 && di.dli_sname) {
            caller = di.dli_sname;
        }
        fprintf(stderr, "ds4: Vulkan debug: alloc[%llu] bytes=%llu va=0x%llx..0x%llx caller=%s map=%s\n",
                (unsigned long long)dbg_alloc_id++, (unsigned long long)bytes,
                (unsigned long long)va, (unsigned long long)(va + bytes),
                caller, map ? "host" : "device");
    }
    return t;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (g_device == VK_NULL_HANDLE) return NULL;
    if (bytes == 0) bytes = 1;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    unsigned char *map = NULL;
    VkBuffer buffer = vulkan_create_host_buffer(bytes, &mem, &map);
    if (buffer == VK_NULL_HANDLE) return NULL;
    return vulkan_tensor_wrap_buffer(buffer, mem, map, bytes);
}

/* Allocate a host-visible tensor on a specific logical tier and restore the
 * previously active tier.  Used by the engine's multi-tier allocations
 * (SPECS_MGPU.md M4). */
extern "C" ds4_gpu_tensor *ds4_vulkan_tensor_alloc_ptr_on(int tier,
                                                          uint64_t bytes) {
    if (tier < 0 || tier >= g_vk_ctx_count) return NULL;
    const int prev = g_vk_ctx_active;
    if (ds4_vulkan_set_current_device(tier) != 0) return NULL;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    struct ds4_vulkan_tensor *h = t ? vulkan_tensor_handle(t) : NULL;
    if (h) h->tier = tier;
    if (t) t->device_id = tier;
    if (prev >= 0 && prev != tier) ds4_vulkan_set_current_device(prev);
    return t;
}

/* MANAGED-memory tensor on a specific logical tier, restoring the previously
 * active tier.  Mirrors ds4_vulkan_tensor_alloc_ptr_on; stamps the tier on the
 * tensor (and its device_id) so a later free/read resolves the right device. */
extern "C" ds4_gpu_tensor *ds4_vulkan_tensor_alloc_managed_on(int tier,
                                                              uint64_t bytes) {
    if (tier < 0 || tier >= g_vk_ctx_count) return NULL;
    const int prev = g_vk_ctx_active;
    if (ds4_vulkan_set_current_device(tier) != 0) return NULL;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc_managed(bytes);
    struct ds4_vulkan_tensor *h = t ? vulkan_tensor_handle(t) : NULL;
    if (h) h->tier = tier;
    if (t) t->device_id = tier;
    if (prev >= 0 && prev != tier) ds4_vulkan_set_current_device(prev);
    return t;
}

/* DEVICE-LOCAL (VRAM) tensor on a specific logical tier, restoring the
 * previously active tier.  The buffer has no host map, so the GPU reads and
 * writes it on-chip (fast, coherent) instead of over PCIe/GTT.  Cross-tier
 * handoffs go through ds4_gpu_tensor_copy's staging path. */
ds4_gpu_tensor *ds4_gpu_tensor_alloc_device_local(uint64_t bytes);
extern "C" ds4_gpu_tensor *ds4_vulkan_tensor_alloc_device_local_on(int tier,
                                                                   uint64_t bytes) {
    if (tier < 0 || tier >= g_vk_ctx_count) return NULL;
    const int prev = g_vk_ctx_active;
    if (ds4_vulkan_set_current_device(tier) != 0) return NULL;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc_device_local(bytes);
    struct ds4_vulkan_tensor *h = t ? vulkan_tensor_handle(t) : NULL;
    if (h) h->tier = tier;
    if (t) t->device_id = tier;
    if (prev >= 0 && prev != tier) ds4_vulkan_set_current_device(prev);
    return t;
}

/* Fase 6 step 4a: device-local tensor (VRAM).  On UMA/APU the only
 * device-local heap is also host-visible, so the tensor keeps a host map and
 * uploads are plain memcpys; on a discrete GPU the buffer is pure
 * DEVICE_LOCAL with no host map and data must be staged with
 * vulkan_upload_to_tensor(). */
ds4_gpu_tensor *ds4_gpu_tensor_alloc_device_local(uint64_t bytes) {
    if (g_device == VK_NULL_HANDLE) return NULL;
    if (bytes == 0) bytes = 1;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    unsigned char *map = NULL;
    uint32_t mem_type;
    if (g_is_uma) {
        mem_type = g_uma_mem_type;
    } else {
        mem_type = (g_device_local_mem_type != UINT32_MAX) ?
                   g_device_local_mem_type : 0u;
    }
    VkBuffer buffer = vulkan_create_buffer(bytes, mem_type, g_is_uma,
                                           &mem, &map);
    if (buffer == VK_NULL_HANDLE) return NULL;
    ds4_gpu_tensor *t = vulkan_tensor_wrap_buffer(buffer, mem, map, bytes);
    struct ds4_vulkan_tensor *h = t ? vulkan_tensor_handle(t) : NULL;
    if (h) {
        h->device_local = 1;
        g_device_local_bytes += bytes;
    }
    return t;
}

/* Host -> tensor upload.  Direct memcpy when the tensor has a host map
 * (host-visible tier or UMA); otherwise a one-shot staging copy
 * (HOST_STAGING -> DEVICE_LOCAL).  The staging path requires that no command
 * scope is open (model window staging and pool seeds run between scopes);
 * appending to an open scope would need a deferred-free of the staging
 * buffer. */
static void vulkan_check_tensor_device(const char *op,
                                       const struct ds4_vulkan_tensor *h);

static int vulkan_upload_to_tensor(ds4_gpu_tensor *t, uint64_t offset,
                                   const void *data, uint64_t bytes) {
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(t);
    if (!h || !data || bytes == 0) return 0;
    if (offset > h->bytes || bytes > h->bytes - offset) return 0;
    vulkan_check_tensor_device("upload", h);
    if (h->host_map) {
        memcpy(h->host_map + h->offset + offset, data, bytes);
        return 1;
    }
    if (g_commands_active) return 0;   /* staging path is one-shot only */
    if (!vulkan_compute_init()) return 0;
    ds4_gpu_tensor *st = ds4_gpu_tensor_alloc(bytes);
    if (!st) return 0;
    struct ds4_vulkan_tensor *sh = vulkan_tensor_handle(st);
    if (!sh || !sh->host_map) {
        ds4_gpu_tensor_free(st);
        return 0;
    }
    memcpy(sh->host_map, data, bytes);
    VkCommandBuffer cb = vulkan_dispatch_begin();
    if (!cb) {
        ds4_gpu_tensor_free(st);
        return 0;
    }
    VkBufferCopy region = { sh->offset, h->offset + offset, bytes };
    vkCmdCopyBuffer(cb, sh->buffer, h->buffer, 1, &region);
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &mb, 0, NULL, 0, NULL);
    if (!vulkan_submit_one_shot()) {
        ds4_gpu_tensor_free(st);
        return 0;
    }
    ds4_gpu_tensor_free(st);
    return 1;
}

/* Tensor -> host download.  Direct memcpy when mapped, otherwise a one-shot
 * staging readback (DEVICE_LOCAL -> HOST_STAGING). */
static int vulkan_download_from_tensor(const ds4_gpu_tensor *t, uint64_t offset,
                                       void *data, uint64_t bytes) {
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(t);
    if (!h || !data || bytes == 0) return 0;
    if (offset > h->bytes || bytes > h->bytes - offset) return 0;
    vulkan_check_tensor_device("download", h);
    if (h->host_map) {
        if (g_commands_active) {
            /* An open (unsubmitted) scope may hold the kernel that wrote this
             * tensor: submit + drain before reading, then reopen so the caller
             * keeps recording.  A bare dirty check is 0 here (nothing is
             * submitted yet), so reading without the flush would return stale
             * data; never do that silently.  (This used to be gated behind
             * DS4_VULKAN_READ_FLUSH.) */
            if (!ds4_gpu_flush_commands()) return 0;
            if (!ds4_gpu_synchronize()) return 0;
        } else if (g_vulkan_device_dirty) {
            vulkan_device_wait();
        }
        memcpy(data, h->host_map + h->offset + offset, bytes);
        return 1;
    }
    if (g_commands_active) return 0;
    if (!vulkan_compute_init()) return 0;
    ds4_gpu_tensor *st = ds4_gpu_tensor_alloc(bytes);
    if (!st) return 0;
    struct ds4_vulkan_tensor *sh = vulkan_tensor_handle(st);
    if (!sh || !sh->host_map) {
        ds4_gpu_tensor_free(st);
        return 0;
    }
    VkCommandBuffer cb = vulkan_dispatch_begin();
    if (!cb) {
        ds4_gpu_tensor_free(st);
        return 0;
    }
    VkBufferCopy region = { h->offset + offset, sh->offset, bytes };
    vkCmdCopyBuffer(cb, h->buffer, sh->buffer, 1, &region);
    if (!vulkan_submit_one_shot()) {
        ds4_gpu_tensor_free(st);
        return 0;
    }
    memcpy(data, sh->host_map, bytes);
    ds4_gpu_tensor_free(st);
    return 1;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    return ds4_gpu_tensor_alloc(bytes);
}

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset,
                                    uint64_t bytes) {
    struct ds4_vulkan_tensor *bh = vulkan_tensor_handle(base);
    if (!bh || offset + bytes > bh->bytes) return NULL;
    struct ds4_vulkan_tensor *h = vulkan_handle_new();
    if (!h) return NULL;
    *h = *bh; /* shares buffer, memory, host_map */
    h->offset = bh->offset + offset;
    h->bytes = bytes;
    h->owner = 0;
    ds4_gpu_tensor *t = vulkan_tensor_new();
    if (!t) {
        vulkan_handle_free(h);
        return NULL;
    }
    t->ptr = h;
    t->bytes = bytes;
    t->owner = 0;
    t->device_id = base->device_id;
    return t;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(tensor);
    if (h) {
        const int prev = g_vk_ctx_active;
        if (g_vk_ctx_count > 1 && h->tier != g_vk_ctx_active)
            ds4_vulkan_set_current_device(h->tier);
        if (h->owner && h->buffer && g_device != VK_NULL_HANDLE) {
            if (getenv("DS4_VULKAN_DEBUG_FREE") != NULL && g_vulkan_device_dirty) {
                void *ra = __builtin_return_address(0);
                Dl_info di = {};
                const char *caller = "?";
                if (dladdr(ra, &di) != 0 && di.dli_sname) {
                    caller = di.dli_sname;
                }
                fprintf(stderr,
                        "ds4: Vulkan debug: FREE owner buffer bytes=%llu while device dirty (in-flight) caller=%s ra=%p\n",
                        (unsigned long long)h->bytes, caller, ra);
            }
            if (h->host_map) vkUnmapMemory(g_device, h->memory);
            vkDestroyBuffer(g_device, h->buffer, NULL);
            if (h->memory) vkFreeMemory(g_device, h->memory, NULL);
        }
        vulkan_handle_free(h);
        /* A free must never change the backend's selected tier. */
        if (prev >= 0 && prev != g_vk_ctx_active)
            (void)ds4_vulkan_set_current_device(prev);
    }
    free(tensor);
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(tensor);
    if (!h || !h->host_map) return NULL;   /* device-local: no host view */
    return (void *)(h->host_map + h->offset);
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset,
                         const void *data, uint64_t bytes) {
    return vulkan_upload_to_tensor(tensor, offset, data, bytes);
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset,
                        void *data, uint64_t bytes) {
    return vulkan_download_from_tensor(tensor, offset, data, bytes);
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(tensor);
    if (!h || count * sizeof(float) > h->bytes) return 0;
    if (h->host_map) {
        float *f = (float *)(h->host_map + h->offset);
        for (uint64_t i = 0; i < count; i++) f[i] = value;
        return 1;
    }
    /* device-local: fill a host staging buffer and upload it */
    float *tmp = (float *)malloc((size_t)count * sizeof(float));
    if (!tmp) return 0;
    for (uint64_t i = 0; i < count; i++) tmp[i] = value;
    const int ok = vulkan_upload_to_tensor(tensor, 0, tmp,
                                           (size_t)count * sizeof(float));
    free(tmp);
    return ok;
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                        const ds4_gpu_tensor *src, uint64_t src_offset,
                        uint64_t bytes) {
    struct ds4_vulkan_tensor *dh = vulkan_tensor_handle(dst);
    struct ds4_vulkan_tensor *sh = vulkan_tensor_handle(src);
    if (!dh || !sh || dst_offset + bytes > dh->bytes ||
        src_offset + bytes > sh->bytes) return 0;
    if (bytes == 0) return 1;

    /* GPU copy path: only inside an open command scope and only when the
     * regions do not overlap (vkCmdCopyBuffer semantics require disjoint).
     * The one-shot path stays a synchronous host memmove so standalone
     * copies keep their blocking semantics. */
    if (g_commands_active && g_device != VK_NULL_HANDLE &&
        dh->buffer && sh->buffer && dh->tier == sh->tier) {
        const uint64_t d_start = dh->offset + dst_offset;
        const uint64_t s_start = sh->offset + src_offset;
        const uint64_t d_end = d_start + bytes;
        const uint64_t s_end = s_start + bytes;
        const int disjoint = d_end <= s_start || s_end <= d_start;
        if (disjoint) {
            if (!vulkan_compute_init()) return 0;
    VkCommandBuffer cb = vulkan_dispatch_begin();
    if (!cb) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL") != NULL)
            fprintf(stderr, "ds4: dispatch fail: no command buffer (commands_active=%d)\n",
                    g_commands_active);
        return 0;
    }
            VkBufferCopy region = { s_start, d_start, bytes };
            vkCmdCopyBuffer(cb, sh->buffer, dh->buffer, 1, &region);
            VkMemoryBarrier mb = {};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &mb, 0, NULL, 0, NULL);
            return 1;
        }
    }
    /* Host path: only when both sides are mapped (host-visible or UMA).
     * Otherwise fall back to upload/download staging. */
    if (dh->host_map && sh->host_map) {
        /* Multi-tier: a cross-device copy (the decode's per-layer hidden-state
         * bounce between tiers) must not read the source mapping while the
         * source device's last submitted commands may still be writing it, nor
         * overwrite the destination mapping while the destination device's
         * in-flight commands (e.g. the previous token's output head reading
         * cur_hc) may still read it.  Drain BOTH devices before the memmove. */
        if (g_vk_ctx_count > 1 && dh->tier != sh->tier &&
            sh->tier >= 0 && sh->tier < g_vk_ctx_count) {
            const int prev = g_vk_ctx_active;
            if (ds4_vulkan_set_current_device(sh->tier) == 0) {
                vulkan_device_wait();               /* drain source */
                if (prev >= 0 && prev != sh->tier)
                    (void)ds4_vulkan_set_current_device(prev);
            }
            if (g_device != VK_NULL_HANDLE) vulkan_device_wait();  /* drain dst */
        }
        memmove(dh->host_map + dh->offset + dst_offset,
                sh->host_map + sh->offset + src_offset, bytes);
        return 1;
    }
    if (sh->host_map) {
        return vulkan_upload_to_tensor(dst, dst_offset,
                                       sh->host_map + sh->offset + src_offset,
                                       bytes);
    }
    if (dh->host_map) {
        return vulkan_download_from_tensor(src, src_offset,
                                           dh->host_map + dh->offset + dst_offset,
                                           bytes);
    }
    /* device -> device: stage through a temporary host buffer.  Cross-tier
     * copies must run the download on the SOURCE device and the upload on the
     * DESTINATION device (each tier owns its own buffers/queue), so switch the
     * active device around them and restore it after. */
    const int prev_dev = g_vk_ctx_active;
    if (g_vk_ctx_count > 1 && sh->tier >= 0 && sh->tier < g_vk_ctx_count &&
        sh->tier != g_vk_ctx_active) {
        (void)ds4_vulkan_set_current_device(sh->tier);
    }
    void *tmp = malloc((size_t)bytes);
    if (!tmp) {
        if (prev_dev >= 0 && prev_dev != g_vk_ctx_active)
            (void)ds4_vulkan_set_current_device(prev_dev);
        return 0;
    }
    int ok = vulkan_download_from_tensor(src, src_offset, tmp, bytes);
    if (g_vk_ctx_count > 1 && dh->tier >= 0 && dh->tier < g_vk_ctx_count &&
        dh->tier != g_vk_ctx_active) {
        (void)ds4_vulkan_set_current_device(dh->tier);
    }
    if (ok) ok = vulkan_upload_to_tensor(dst, dst_offset, tmp, bytes);
    free(tmp);
    if (prev_dev >= 0 && prev_dev != g_vk_ctx_active)
        (void)ds4_vulkan_set_current_device(prev_dev);
    return ok;
}

int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count) {
    struct ds4_vulkan_tensor *dh = vulkan_tensor_handle(dst);
    struct ds4_vulkan_tensor *sh = vulkan_tensor_handle(src);
    if (!dh || !sh || !dh->host_map || !sh->host_map) return 0;
    const float *s = (const float *)(sh->host_map + sh->offset + src_offset);
    uint16_t *d = (uint16_t *)(dh->host_map + dh->offset + dst_offset);
    for (uint64_t i = 0; i < count; i++) {
        float v = s[i];
        uint32_t bits = 0;
        memcpy(&bits, &v, sizeof(bits));
        uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
        int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
        uint32_t man = bits & 0x7fffffu;
        if (exp >= 31) {
            d[i] = (uint16_t)(sign | 0x7bffu);
        } else if (exp <= 0) {
            uint32_t mant = man | 0x800000u;
            int shift = 1 - exp;
            d[i] = (uint16_t)(sign | (mant >> (uint32_t)shift) >> 13);
        } else {
            d[i] = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
        }
    }
    return 1;
}

/* --- compute dispatch ---------------------------------------------------- */

/* Every submission to g_queue goes through this helper.  The lock covers only
 * the submit, not GPU execution, so the async overlap is preserved. */
/* After a device lost, report the last checkpoint markers the GPU reached
 * (DS4_VULKAN_CHECKPOINTS) so the faulting dispatch is identifiable. */
static void vulkan_dbg_dump_checkpoints(void) {
    if (g_dbg_checkpoints != 1 || !g_pfnGetQueueCheckpointDataNV) return;
    uint32_t n = 0;
    g_pfnGetQueueCheckpointDataNV(g_queue, &n, NULL);
    if (n == 0) {
        fprintf(stderr, "ds4: checkpoints: none recorded\n");
        return;
    }
    if (n > 64) n = 64;
    VkCheckpointDataNV data[64];
    for (uint32_t i = 0; i < n; i++) {
        data[i].sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV;
        data[i].pNext = NULL;
    }
    g_pfnGetQueueCheckpointDataNV(g_queue, &n, data);
    for (uint32_t i = 0; i < n; i++) {
        fprintf(stderr, "ds4: GPU last checkpoint pipe=%u\n",
                (uint32_t)(uintptr_t)data[i].pCheckpointMarker);
    }
}

static VkResult vulkan_queue_submit(uint32_t count, const VkSubmitInfo *si,
                                    VkFence fence) {
    pthread_mutex_lock(&g_queue_mutex);
    const VkResult rc = vkQueueSubmit(g_queue, count, si, fence);
    pthread_mutex_unlock(&g_queue_mutex);
    if (rc != VK_SUCCESS) vulkan_dbg_dump_checkpoints();
    return rc;
}

/* One-shot submit of the current compute CB (used when the caller dispatches
 * without an open command scope): end, submit, wait, reset. */
static int vulkan_submit_one_shot(void) {
    if (vkEndCommandBuffer(g_cmd[g_cmd_i]) != VK_SUCCESS) return 0;
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd[g_cmd_i];
    VkResult rc = vulkan_queue_submit(1, &si, VK_NULL_HANDLE);
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkQueueSubmit");
        return 0;
    }
    g_vulkan_device_dirty = 1;
    vulkan_device_wait();
    vkResetCommandBuffer(g_cmd[g_cmd_i], 0);
    return 1;
}

/* Begin recording the current compute CB.  Waits only the fence of the CB
 * being reused: the compute scopes are double-buffered, so an in-flight
 * sibling scope is NOT drained here (same queue FIFO orders it before this
 * scope; the static decode map is staged once, no per-layer re-stage race).
 * With switch_ok the caller (begin_commands/flush) may move to the sibling
 * CB to keep recording while the previous scope still runs.  The readback
 * fence may sit in the slot as the in-flight marker of a signal-submitted
 * scope: wait it but never reset it (the readback owner resets it at the
 * next signal; a concurrent worker wait must not race a reset). */
static VkCommandBuffer vulkan_cb_acquire(int switch_ok) {
    if (g_device == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    if (g_cmd_fence[g_cmd_i] != VK_NULL_HANDLE && switch_ok) {
        g_cmd_i = 1 - g_cmd_i;
    }
    if (g_cmd_fence[g_cmd_i] != VK_NULL_HANDLE) {
        const VkFence f = g_cmd_fence[g_cmd_i];
        vkWaitForFences(g_device, 1, &f, VK_TRUE, UINT64_MAX);
        /* Clear the in-flight marker; the CB's own persistent fence stays in
         * g_cb_fence[] and is reset+reused by the next vulkan_cb_submit (the
         * readback fence is a global owned by the signal). */
        g_cmd_fence[g_cmd_i] = VK_NULL_HANDLE;
    }
    /* Descriptor-pool bound: no scope is open here, so a rare drain resets
     * the pool (all in-flight sets are reclaimable) before it can exhaust
     * during long read-free passes (decode-style prefill). */
    if (g_desc_sets_allocated >= DS4_VK_DESC_SET_HWM) {
        vulkan_device_wait();
    }
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(g_cmd[g_cmd_i], &bi) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return g_cmd[g_cmd_i];
}

/* --- GPU timestamp scope timing (see globals) --------------------------- */

static void vulkan_ts_begin(VkCommandBuffer cb) {
    if (!g_ts_on || !g_commands_active || g_ts_started || !cb) return;
    vkCmdResetQueryPool(cb, g_ts_pool, 0, 128);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g_ts_pool, 0);
    g_ts_started = 1;
    g_ts_router_written = 0;
    g_ts_n = 0;
}

static void vulkan_ts_end(VkCommandBuffer cb) {
    if (!g_ts_on || !g_ts_started || !cb || g_ts_verbose) return;
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g_ts_pool, 1);
}

static void vulkan_ts_report(const char *tag) {
    if (!g_ts_on || !g_ts_started) return;
    if (g_ts_verbose) {
        const uint32_t n = g_ts_n;
        uint64_t r[128] = {0};
        const VkResult rc =
            vkGetQueryPoolResults(g_device, g_ts_pool, 0, 1u + 2u * n,
                                   (1u + 2u * n) * sizeof(uint64_t), r,
                                   sizeof(uint64_t),
                                   VK_QUERY_RESULT_64_BIT |
                                   VK_QUERY_RESULT_WAIT_BIT);
        if (rc == VK_SUCCESS && n > 0) {
            const double scope =
                (double)(r[2 * n] - r[0]) * (double)g_ts_period / 1e6;
            fprintf(stderr, "ds4: gpu-ts %s: %.3f ms (%u disp)\n", tag, scope,
                    n);
            for (uint32_t i = 0; i < n; i++) {
                const double dt = (double)(r[2 + 2 * i] - r[1 + 2 * i]) *
                                  (double)g_ts_period / 1e6;
                const uint32_t p = g_ts_pipe[i];
                fprintf(stderr, "ds4: gpu-pipe %-28s %.4f ms\n",
                        p < 96 ? g_pipe_names[p] : "?", dt);
            }
        }
        g_ts_started = 0;
        g_ts_n = 0;
        return;
    }
    uint64_t r[4] = {0, 0, 0, 0};
    const uint32_t n = g_ts_router_written ? 4u : 2u;
    const VkResult rc =
        vkGetQueryPoolResults(g_device, g_ts_pool, 0, n, n * sizeof(uint64_t),
                              r, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT |
                              VK_QUERY_RESULT_WAIT_BIT);
    if (rc == VK_SUCCESS) {
        const double scope = (double)(r[1] - r[0]) * (double)g_ts_period / 1e6;
        fprintf(stderr, "ds4: gpu-ts %s: %.3f ms", tag, scope);
        if (n == 4u) {
            const double router =
                (double)(r[3] - r[2]) * (double)g_ts_period / 1e6;
            fprintf(stderr, " (router=%.3f ms)", router);
        }
        fprintf(stderr, "\n");
    }
    g_ts_started = 0;
    g_ts_router_written = 0;
}

/* End + submit the currently recording CB with its in-flight fence (the
 * GPU starts it immediately; the fence marks the CB busy until it is
 * reused by vulkan_cb_acquire). */
static int vulkan_cb_submit(void) {
    vulkan_ts_end(g_cmd[g_cmd_i]);
    if (vkEndCommandBuffer(g_cmd[g_cmd_i]) != VK_SUCCESS) return 0;
    if (g_cb_fence[g_cmd_i] == VK_NULL_HANDLE) {
        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(g_device, &fci, NULL, &g_cb_fence[g_cmd_i]) != VK_SUCCESS) {
            return 0;
        }
    } else if (vkResetFences(g_device, 1, &g_cb_fence[g_cmd_i]) != VK_SUCCESS) {
        return 0;
    }
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd[g_cmd_i];
    const VkResult rc = vulkan_queue_submit(1, &si, g_cb_fence[g_cmd_i]);
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkQueueSubmit(scope)");
        return 0;
    }
    g_cmd_fence[g_cmd_i] = g_cb_fence[g_cmd_i];
    g_vulkan_device_dirty = 1;
    return 1;
}

/* Ensure a command buffer is recording.  If the caller opened a command
 * scope (begin_commands), records into it; otherwise starts a one-shot. */
static VkCommandBuffer vulkan_dispatch_begin(void) {
    if (g_device == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    if (g_commands_active) return g_cmd[g_cmd_i];
    /* One-shot: wait only the CB being reused.  A signal-submitted scope
     * (ds4_gpu_signal_selected_readback_ready) leaves its CB in flight with
     * the readback fence in the slot; re-recording it before the GPU is done
     * would fault (GPUVM / context lost), so the acquire waits it here. */
    return vulkan_cb_acquire(0);
}

static int vulkan_dispatch_end(int oneshot) {
    if (!oneshot) return 1;
    return vulkan_submit_one_shot();
}

/* --- Fase 7: async readback + worker-safe download/store ---------------- */

/* Ensure the dedicated worker command buffer (never g_cmd) and its fence.
 * Called under g_worker_mutex. */
static int vulkan_worker_cb_ensure(void) {
    if (g_worker_cb != VK_NULL_HANDLE && g_worker_fence != VK_NULL_HANDLE) return 1;
    if (g_device == VK_NULL_HANDLE) return 0;
    if (g_worker_cb == VK_NULL_HANDLE) {
        /* Dedicated pool: the main thread records g_cmd[] from g_cmd_pool
         * concurrently, and a command pool must not be used from two threads
         * without external synchronization (NVIDIA faults with Xid 13). */
        if (g_worker_pool == VK_NULL_HANDLE) {
            VkCommandPoolCreateInfo cpci = {};
            cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            cpci.queueFamilyIndex = g_queue_family;
            if (vkCreateCommandPool(g_device, &cpci, NULL,
                                    &g_worker_pool) != VK_SUCCESS) {
                return 0;
            }
        }
        VkCommandBufferAllocateInfo cbai = {};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = g_worker_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(g_device, &cbai, &g_worker_cb) != VK_SUCCESS) {
            return 0;
        }
    }
    if (g_worker_fence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(g_device, &fci, NULL, &g_worker_fence) != VK_SUCCESS) {
            return 0;
        }
    }
    return 1;
}

/* Record `regions` (device -> device copies) into the worker CB and submit
 * them on the main queue with g_worker_fence (not waited here).  The main
 * queue FIFO orders this copy after every previously submitted scope (the
 * previous token's MoE readers of the same pool slots) and the caller
 * (vulkan_pool_commit_pending) waits the fence before the MoE that consumes
 * the data.  Host-visible sources must be fully written by the host before
 * this submit (the worker memcpy happens before). */
static int vulkan_worker_copy_submit_multi(VkBuffer src, VkBuffer dst,
                                           const VkBufferCopy *regions,
                                           uint32_t nr) {
    if (nr == 0) return 1;
    pthread_mutex_lock(&g_worker_mutex);
    if (!vulkan_worker_cb_ensure()) {
        pthread_mutex_unlock(&g_worker_mutex);
        return 0;
    }
    /* The worker CB is single-buffered: a previous submission of it may still
     * be executing.  Wait its fence before re-recording (Vulkan forbids
     * modifying a command buffer while it is in use; NVIDIA faults with Xid
     * 13).  The main thread's own wait on the same fence is idempotent. */
    if (g_worker_fence_pending) {
        vkWaitForFences(g_device, 1, &g_worker_fence, VK_TRUE, UINT64_MAX);
    }
    if (vkResetCommandBuffer(g_worker_cb, 0) != VK_SUCCESS ||
        vkResetFences(g_device, 1, &g_worker_fence) != VK_SUCCESS) {
        pthread_mutex_unlock(&g_worker_mutex);
        return 0;
    }
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(g_worker_cb, &bi) != VK_SUCCESS) {
        pthread_mutex_unlock(&g_worker_mutex);
        return 0;
    }
    vkCmdCopyBuffer(g_worker_cb, src, dst, nr, regions);
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(g_worker_cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &mb, 0, NULL, 0, NULL);
    if (vkEndCommandBuffer(g_worker_cb) != VK_SUCCESS) {
        pthread_mutex_unlock(&g_worker_mutex);
        return 0;
    }
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_worker_cb;
    const VkResult rc = vulkan_queue_submit(1, &si, g_worker_fence);
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkQueueSubmit(worker copy)");
        pthread_mutex_unlock(&g_worker_mutex);
        return 0;
    }
    g_worker_fence_pending = 1;
    g_vulkan_device_dirty = 1;
    pthread_mutex_unlock(&g_worker_mutex);
    return 1;
}

/* Wait (only) the worker copy fence, if a store/readback submit is pending.
 * Called by the main thread right before a dispatch that consumes the pool.
 * This is a per-submit wait, not a device wait: the GPU can interleave the
 * main thread's compute submissions in between. */
static int vulkan_worker_copy_wait(void) {
    if (g_device == VK_NULL_HANDLE) return 1;
    if (!g_worker_fence_pending) return 1;
    pthread_mutex_lock(&g_worker_mutex);
    const VkResult rc = vkWaitForFences(g_device, 1, &g_worker_fence,
                                        VK_TRUE, UINT64_MAX);
    g_worker_fence_pending = 0;
    pthread_mutex_unlock(&g_worker_mutex);
    return rc == VK_SUCCESS ? 1 : 0;
}

/* Worker-safe device->host download (never touches g_cmd). */
static int vulkan_worker_download(const ds4_gpu_tensor *t, uint64_t offset,
                                  void *data, uint64_t bytes) {
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(t);
    if (!h || !data || bytes == 0) return 0;
    if (offset > h->bytes || bytes > h->bytes - offset) return 0;
    if (h->host_map) {
        if (g_vulkan_device_dirty) vulkan_device_wait();
        memcpy(data, h->host_map + h->offset + offset, bytes);
        return 1;
    }
    if (g_device == VK_NULL_HANDLE) return 0;
    ds4_gpu_tensor *st = ds4_gpu_tensor_alloc(bytes);
    if (!st) return 0;
    struct ds4_vulkan_tensor *sh = vulkan_tensor_handle(st);
    if (!sh || !sh->host_map || !sh->buffer || !h->buffer) {
        ds4_gpu_tensor_free(st);
        return 0;
    }
    VkBufferCopy region = { h->offset + offset, sh->offset, bytes };
    const int ok = vulkan_worker_copy_submit_multi(h->buffer, sh->buffer,
                                                   &region, 1);
    if (ok) {
        pthread_mutex_lock(&g_worker_mutex);
        const VkResult rc = g_worker_fence_pending
            ? vkWaitForFences(g_device, 1, &g_worker_fence, VK_TRUE, UINT64_MAX)
            : VK_SUCCESS;
        g_worker_fence_pending = 0;
        pthread_mutex_unlock(&g_worker_mutex);
        if (rc == VK_SUCCESS) memcpy(data, sh->host_map, bytes);
    }
    ds4_gpu_tensor_free(st);
    return ok;
}

/* The readback fence signals when the submission that produced the selection
 * (the router scope) has completed on the GPU.  signal runs in the main
 * thread at a clean command-scope boundary: it ends the open scope and
 * submits it directly with the fence (no separate empty submit), so the
 * worker knows the selection is readable exactly when the fence signals. */
extern "C" int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value) {
    if (event_value) *event_value = 1;
    if (g_device == VK_NULL_HANDLE) return 1;
    if (!g_commands_active) return 1;

    /* The readback fence doubles as the in-flight marker of this CB (the
     * worker waits the fence; vulkan_cb_acquire waits it too when the CB is
     * reused but never resets it, so the waits cannot race). */
    pthread_mutex_lock(&g_readback_mutex);
    int ret = 1;
    do {
    if (getenv("DS4_VULKAN_DEBUG_SUBMIT") != NULL) {
        const double e0 = vulkan_now_ms();
        vulkan_ts_end(g_cmd[g_cmd_i]);
        if (vkEndCommandBuffer(g_cmd[g_cmd_i]) != VK_SUCCESS) {
            g_commands_active = false;
            ret = 0;
            break;
        }
        const double e1 = vulkan_now_ms();
        if (g_readback_fence == VK_NULL_HANDLE) {
            VkFenceCreateInfo fci = {};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (vkCreateFence(g_device, &fci, NULL, &g_readback_fence) != VK_SUCCESS) {
                ret = 0;
                break;
            }
        } else if (vkResetFences(g_device, 1, &g_readback_fence) != VK_SUCCESS) {
            ret = 0;
            break;
        }
        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &g_cmd[g_cmd_i];
        const double tsub = vulkan_now_ms();
        const VkResult rc = vulkan_queue_submit(1, &si, g_readback_fence);
        double wq = 0.0;
        if (rc == VK_SUCCESS) {
            /* Measurement only (DEBUG_SUBMIT): wait the attention+router scope
             * so the print attributes its GPU time. The worker and the next
             * signal still own the fence; waiting does not reset it, so the
             * async path is unaffected outside the debug run. */
            if (vkWaitForFences(g_device, 1, &g_readback_fence, VK_TRUE,
                                UINT64_MAX) == VK_SUCCESS) {
                wq = vulkan_now_ms() - tsub;
            }
            vulkan_ts_report("attn+router");
        }
        const double e3 = vulkan_now_ms();
        g_commands_active = false;
        fprintf(stderr, "ds4: Vulkan debug submit: host=%.3f ms end=%.3f ms queue=%.3f ms fwait=%.3f ms total=%.3f ms ndisp=%u binds=%u bytes=%llu\n",
                g_dbg_scope_t0_ms > 0.0 ? e0 - g_dbg_scope_t0_ms : 0.0,
                e1 - e0, 0.0, wq, e3 - e0, g_scope_dispatch_count,
                g_scope_bind_count, (unsigned long long)g_scope_bind_bytes);
        g_scope_bind_bytes = 0;
        g_scope_bind_count = 0;
        g_scope_dispatch_count = 0;
        if (rc != VK_SUCCESS) {
            vulkan_log_vk(rc, "vkQueueSubmit(readback scope)");
            ret = 0;
            break;
        }
        g_cmd_fence[g_cmd_i] = g_readback_fence;
        g_readback_fence_pending = 1;
        g_vulkan_device_dirty = 1;
        break;
    }

    if (vkEndCommandBuffer(g_cmd[g_cmd_i]) != VK_SUCCESS) {
        g_commands_active = false;
        ret = 0;
        break;
    }
    if (g_readback_fence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(g_device, &fci, NULL, &g_readback_fence) != VK_SUCCESS) {
            g_commands_active = false;
            ret = 0;
            break;
        }
    } else if (vkResetFences(g_device, 1, &g_readback_fence) != VK_SUCCESS) {
        g_commands_active = false;
        ret = 0;
        break;
    }
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd[g_cmd_i];
    const VkResult rc = vulkan_queue_submit(1, &si, g_readback_fence);
    g_commands_active = false;
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkQueueSubmit(readback scope)");
        ret = 0;
        break;
    }
    g_cmd_fence[g_cmd_i] = g_readback_fence;
    g_readback_fence_pending = 1;
    g_vulkan_device_dirty = 1;
} while (0);
    pthread_mutex_unlock(&g_readback_mutex);
    return ret;
}

extern "C" int ds4_gpu_wait_selected_readback_ready(uint64_t event_value,
                                                    const char *label) {
    (void)event_value;
    (void)label;
    if (g_device == VK_NULL_HANDLE) return 1;
    pthread_mutex_lock(&g_readback_mutex);
    int ret = 1;
    if (g_readback_fence_pending) {
        const VkResult rc = vkWaitForFences(g_device, 1, &g_readback_fence,
                                            VK_TRUE, UINT64_MAX);
        g_readback_fence_pending = 0;
        ret = rc == VK_SUCCESS ? 1 : 0;
    }
    pthread_mutex_unlock(&g_readback_mutex);
    return ret;
}

extern "C" int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value,
                                                         const char *label) {
    return ds4_gpu_wait_selected_readback_ready(event_value, label);
}

extern "C" int ds4_gpu_tensor_read_after_selected_event(
        const ds4_gpu_tensor *tensor, uint64_t offset, void *data,
        uint64_t bytes, uint64_t event_value, const char *label) {
    (void)event_value;
    (void)label;
    if (ds4_gpu_wait_selected_readback_ready(event_value, label) == 0) return 0;
    return vulkan_worker_download(tensor, offset, data, bytes);
}

/* One descriptor write: a storage-buffer range on a binding. */
struct ds4_vk_bind {
    uint32_t     binding;
    VkBuffer     buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
};

/* Tier-ownership invariant: a VkBuffer belongs to one VkDevice, so binding a
 * tensor whose tier differs from the selected device is undefined.  Report it
 * loudly (rate-limited) — a silent mismatch is exactly the multi-GPU
 * corruption class we are eliminating. */
static uint64_t g_tier_mismatch_reports = 0;
static void vulkan_check_tensor_device(const char *op,
                                       const struct ds4_vulkan_tensor *h) {
    if (!h || g_vk_ctx_count <= 1) return;
    if (h->tier < 0 || h->tier == g_vk_ctx_active) return;
    if (g_tier_mismatch_reports++ < 128) {
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "TIER MISMATCH: %s tensor_tier=%d selected_tier=%d bytes=%llu\n",
                op, h->tier, g_vk_ctx_active, (unsigned long long)h->bytes);
    }
}

static struct ds4_vk_bind vulkan_bind_tensor(uint32_t binding,
                                             const ds4_gpu_tensor *t) {
    struct ds4_vk_bind b = {};
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(t);
    vulkan_check_tensor_device("bind_tensor", h);
    if (h) {
        b.binding = binding;
        b.buffer = h->buffer;
        b.offset = h->offset;
        b.range = h->bytes;
    }
    return b;
}

/* Bind a byte sub-range [off, off+bytes) of a tensor (used for the per-layer
 * expert pool regions, which share one tensor). */
static struct ds4_vk_bind vulkan_bind_tensor_at(uint32_t binding,
                                                const ds4_gpu_tensor *t,
                                                uint64_t off, uint64_t bytes) {
    struct ds4_vk_bind b = {};
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(t);
    vulkan_check_tensor_device("bind_tensor_at", h);
    if (h && off <= h->bytes && bytes <= h->bytes - off) {
        b.binding = binding;
        b.buffer = h->buffer;
        b.offset = h->offset + off;
        b.range = bytes;
    }
    return b;
}

/* Same coverage rule as vulkan_model_window_for, over the persistent cache. */
static struct ds4_vk_model_window *vulkan_weight_cache_for(
        uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return NULL;
    for (uint32_t i = 0; i < g_weight_cache_count; i++) {
        struct ds4_vk_model_window *w = &g_weight_cache[i];
        if (offset < w->base) continue;
        const uint64_t rel = offset - w->base;
        if (rel <= w->size && bytes <= w->size - rel) {
            return w;
        }
    }
    return NULL;
}

/* Find the staged window covering [offset, offset+bytes).  NULL when no
 * window covers the whole range.  Uses offset - base (never base+size -
 * offset) so an offset beyond the window end cannot underflow.  The
 * persistent per-device weight cache is consulted first: a cached range is
 * resident for the whole session and never needs a transient window. */
static struct ds4_vk_model_window *vulkan_model_window_for(
        uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return NULL;
    struct ds4_vk_model_window *cached = vulkan_weight_cache_for(offset, bytes);
    if (cached) return cached;
    for (uint32_t i = 0; i < g_model_window_count; i++) {
        struct ds4_vk_model_window *w = &g_model_windows[i];
        if (offset < w->base) continue;
        const uint64_t rel = offset - w->base;
        if (rel <= w->size && bytes <= w->size - rel) {
            return w;
        }
    }
    return NULL;
}

static struct ds4_vk_bind vulkan_bind_model(uint32_t binding,
                                            uint64_t offset, uint64_t bytes) {
    struct ds4_vk_bind b = {};
    struct ds4_vk_model_window *w = vulkan_model_window_for(offset, bytes);
    if (w) {
        b.binding = binding;
        b.buffer = w->buffer;
        b.offset = w->buf_offset + (offset - w->base);
        b.range = bytes;
    }
    return b;
}

/* True when the model wrapper has a staged window covering [offset, bytes). */
static int vulkan_model_range_ok(uint64_t offset, uint64_t bytes) {
    return vulkan_model_window_for(offset, bytes) != NULL;
}

/* Return a persistent scratch tensor of at least `bytes`, growing it if
 * needed.  NULL on allocation failure.
 *
 * The scratch buffers are reused across dispatches and across layers.  When
 * one needs to grow, the old buffer must NOT be freed while an earlier submit
 * may still read it (the decode overlaps command scopes: layer N's MoE submit
 * is still in flight while layer N+1 records).  Settle the device before
 * freeing, otherwise the in-flight shader reads/writes an unmapped buffer and
 * the GPU reports a GPUVM fault (RW=WRITE, one past the old buffer end). */
/* Free a scratch buffer that may still be referenced by the OPEN command
 * scope.  vulkan_device_wait() only drains submitted work, so submit the
 * current scope first; otherwise the GPU reads/writes the freed buffer when
 * the open scope executes (device lost / Xid).  Scratch growth is rare, so
 * the extra submit is cheap. */
static void vulkan_scratch_retire(ds4_gpu_tensor **slot) {
    if (!*slot) return;
    if (g_commands_active) {
        (void)ds4_gpu_flush_commands();
    }
    vulkan_device_wait();
    ds4_gpu_tensor_free(*slot);
    *slot = NULL;
}

static ds4_gpu_tensor *vulkan_scratch_a(uint64_t bytes) {
    if (g_scratch_a && g_scratch_a->bytes >= bytes) return g_scratch_a;
    vulkan_scratch_retire(&g_scratch_a);
    g_scratch_a = ds4_gpu_tensor_alloc(bytes);
    return g_scratch_a;
}

static ds4_gpu_tensor *vulkan_scratch_b(uint64_t bytes) {
    if (g_scratch_b && g_scratch_b->bytes >= bytes) return g_scratch_b;
    vulkan_scratch_retire(&g_scratch_b);
    g_scratch_b = ds4_gpu_tensor_alloc(bytes);
    return g_scratch_b;
}

static ds4_gpu_tensor *vulkan_scratch_c(uint64_t bytes) {
    if (g_scratch_c && g_scratch_c->bytes >= bytes) return g_scratch_c;
    vulkan_scratch_retire(&g_scratch_c);
    g_scratch_c = ds4_gpu_tensor_alloc(bytes);
    return g_scratch_c;
}

static ds4_gpu_tensor *vulkan_scratch_d(uint64_t bytes) {
    if (g_scratch_d && g_scratch_d->bytes >= bytes) return g_scratch_d;
    vulkan_scratch_retire(&g_scratch_d);
    g_scratch_d = ds4_gpu_tensor_alloc(bytes);
    return g_scratch_d;
}

/* General compute launcher: allocates a descriptor set, applies `n_binds`
 * storage-buffer writes, binds pipeline + push constants + set, dispatches
 * (gx, gy, gz) threadgroups, and inserts a SHADER_WRITE->SHADER_READ memory
 * barrier.  Works both inside an open command scope and as a one-shot. */
static int vulkan_dispatch(VkPipeline pipeline,
                           const void *params, uint32_t params_size,
                           const struct ds4_vk_bind *binds, uint32_t n_binds,
                           uint32_t gx, uint32_t gy, uint32_t gz) {
    if (g_device == VK_NULL_HANDLE) return 0;
    if (n_binds > DS4_VK_MAX_BINDS) return 0;
    if (!vulkan_compute_init()) return 0;

    if (getenv("DS4_VULKAN_DEBUG_SUBMIT") != NULL) {
        g_scope_dispatch_count++;
        g_scope_bind_count += n_binds;
        for (uint32_t bi = 0; bi < n_binds; bi++) {
            g_scope_bind_bytes += binds[bi].range;
        }
    }

    /* Debug bisection (DS4_VULKAN_SKIP_PIPE=<i,j,...>): skip a dispatch so a
     * GPUVM fault can be attributed to a specific kernel. */
    const char *skip_env = getenv("DS4_VULKAN_SKIP_PIPE");
    if (skip_env != NULL) {
        int pipe_idx = -1;
        for (int pi = 0; pi < DS4_VK_PIPE_COUNT; pi++) {
            if (g_pipes[pi] == pipeline) { pipe_idx = pi; break; }
        }
        for (const char *s = skip_env; *s;) {
            while (*s == ',' || *s == ' ') s++;
            if (!*s) break;
            const char *e = s;
            while (*e && *e != ',') e++;
            char tok[32];
            const size_t tl = (size_t)(e - s) < sizeof(tok) - 1 ?
                              (size_t)(e - s) : sizeof(tok) - 1;
            memcpy(tok, s, tl);
            tok[tl] = 0;
            if (pipe_idx == atoi(tok)) {
                fprintf(stderr, "ds4: Vulkan debug: SKIP pipe=%d (bisect)\n",
                        pipe_idx);
                return 1;   /* no-op success so the encode keeps going */
            }
            s = e;
        }
    }

    VkCommandBuffer cb = vulkan_dispatch_begin();
    if (!cb) return 0;
    const int oneshot = !g_commands_active;
    vulkan_ts_begin(cb);

    const int dbg_disp = getenv("DS4_VULKAN_DEBUG_DISPATCH_TIME") != NULL;
    const double d0 = dbg_disp ? vulkan_now_ms() : 0.0;

    /* Fresh descriptor set per dispatch; pool is reset at synchronize. */
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dsai = {};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = g_desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &g_desc_layout;
    if (vkAllocateDescriptorSets(g_device, &dsai, &set) != VK_SUCCESS) {
        vulkan_log_vk(VK_ERROR_OUT_OF_POOL_MEMORY,
                      "vkAllocateDescriptorSets");
        return 0;
    }
    g_desc_sets_allocated++;
    const double d1 = dbg_disp ? vulkan_now_ms() : 0.0;

    VkWriteDescriptorSet writes[DS4_VK_MAX_BINDS];
    VkDescriptorBufferInfo infos[DS4_VK_MAX_BINDS];
    for (uint32_t i = 0; i < n_binds; i++) {
        if (binds[i].buffer == VK_NULL_HANDLE &&
            getenv("DS4_VULKAN_DEBUG_BINDS") != NULL) {
            fprintf(stderr, "ds4: Vulkan debug: dispatch bind %u has NULL buffer\n",
                    binds[i].binding);
        }
        infos[i].buffer = binds[i].buffer;
        infos[i].offset = binds[i].offset;
        infos[i].range = binds[i].range;
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = binds[i].binding;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(g_device, n_binds, writes, 0, NULL);
    const double d2 = dbg_disp ? vulkan_now_ms() : 0.0;

    /* Verbose debug timing: timestamp every dispatch of the scope. */
    const int ts_each =
        (g_ts_on && g_ts_verbose && g_ts_started && g_ts_n < 62u);
    if (ts_each) {
        int pi = -1;
        for (int k = 0; k < DS4_VK_PIPE_COUNT; k++) {
            if (g_pipes[k] == pipeline) { pi = k; break; }
        }
        g_ts_pipe[g_ts_n] = (uint32_t)(pi < 0 ? 0 : pi);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            g_ts_pool, 1u + 2u * g_ts_n);
    }
    const int ts_router =
        (!g_ts_verbose && g_ts_on &&
         pipeline == g_pipes[DS4_PIPE_ROUTER_SELECT]);
    if (ts_router) {
        g_ts_router_written = 1;
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            g_ts_pool, 2);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cb, g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, params_size, params);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            g_pipe_layout, 0, 1, &set, 0, NULL);
    if (g_dbg_checkpoints == 1 && g_pfnCmdSetCheckpointNV) {
        int dbg_pipe = -1;
        for (int pi = 0; pi < DS4_VK_PIPE_COUNT; pi++) {
            if (g_pipes[pi] == pipeline) { dbg_pipe = pi; break; }
        }
        g_pfnCmdSetCheckpointNV(cb, (const void *)(uintptr_t)(uint32_t)dbg_pipe);
    }
    vkCmdDispatch(cb, gx, gy, gz);
    if (ts_each) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            g_ts_pool, 2u + 2u * g_ts_n);
        g_ts_n++;
    }
    if (ts_router)
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            g_ts_pool, 3);
    if (getenv("DS4_VULKAN_DEBUG_DISPATCH") != NULL) {
        int pipe_idx = -1;
        for (int pi = 0; pi < DS4_VK_PIPE_COUNT; pi++) {
            if (g_pipes[pi] == pipeline) { pipe_idx = pi; break; }
        }
        fprintf(stderr, "ds4: Vulkan debug dispatch: w=%llu t=%llu pipe=%d gx=%u gy=%u gz=%u active=%d\n",
                (unsigned long long)vulkan_wall_ms(),
                (unsigned long long)vulkan_now_ms(), pipe_idx, gx, gy, gz,
                g_commands_active ? 1 : 0);
    }
    const double d3 = dbg_disp ? vulkan_now_ms() : 0.0;

    /* Order following dispatches (and host reads after synchronize) against
     * this kernel's writes. */
    if (getenv("DS4_VULKAN_FULL_BARRIER") != NULL) {
        /* Bisect: the compute->compute barrier may be insufficient for
         * host-visible (GTT) memory on this driver.  Test a full memory
         * barrier that orders every read/write against every command stage. */
        VkMemoryBarrier fmb = {};
        fmb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        fmb.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        fmb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                             1, &fmb, 0, NULL, 0, NULL);
    } else {
        VkMemoryBarrier mb = {};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &mb, 0, NULL, 0, NULL);
    }
    if (dbg_disp) {
        fprintf(stderr, "ds4: Vulkan debug disp-time: alloc=%.3f upd=%.3f rec=%.3f bar=%.3f total=%.3f ms nb=%u\n",
                d1 - d0, d2 - d1, d3 - d2, vulkan_now_ms() - d3,
                vulkan_now_ms() - d0, n_binds);
    }

    return vulkan_dispatch_end(oneshot);
}

/* --- public contract: command lifecycle --------------------------------- */

int ds4_gpu_begin_commands(void) {
    if (g_device == VK_NULL_HANDLE) return 0;
    if (getenv("DS4_VULKAN_DEBUG_ONESHOT") != NULL) {
        /* Submit+wait after every dispatch so a GPUVM fault is attributed to
         * a single kernel (the dispatch log line right before the fault). */
        return 1;
    }
    if (g_commands_active) return 1;
    /* Double-buffered scope CBs (Fase 7 flush early-submit): wait only the CB
     * being reused; an in-flight sibling scope is NOT drained here (same
     * queue FIFO orders it before this scope).  The static decode map is
     * staged once so successive per-layer scopes overlap without re-stage
     * races; window re-staging and scratch growth still settle before
     * freeing (vulkan_destroy_model_wrapper / vulkan_scratch). */
    if (getenv("DS4_VULKAN_DEBUG_DRAIN") != NULL) {
        const double d0 = vulkan_now_ms();
        if (vulkan_cb_acquire(1) == VK_NULL_HANDLE) return 0;
        const double d1 = vulkan_now_ms();
        fprintf(stderr, "ds4: Vulkan debug: acquire begin_commands %.1f ms\n",
                d1 - d0);
    } else if (vulkan_cb_acquire(1) == VK_NULL_HANDLE) {
        return 0;
    }
    g_commands_active = true;
    if (getenv("DS4_VULKAN_DEBUG_SUBMIT") != NULL) g_dbg_scope_t0_ms = vulkan_now_ms();
    return 1;
}

int ds4_gpu_flush_encoder(void) {
    return g_commands_active ? 1 : 0;
}

/* Submit the open scope (if any) so the GPU can start it while the host keeps
 * encoding into a fresh scope.  Metal's flush_encoder has the same semantic;
 * the async expert-load overlap (Fase 7) relies on it to run the shared-expert
 * compute while the worker stores the routed experts.  When nothing is open a
 * flush is a no-op success.  On Vulkan the reopen moves to the sibling CB:
 * NO device drain, so the just-submitted scope overlaps the encoding of the
 * next one (the only waits are the per-CB fences at CB reuse). */
int ds4_gpu_flush_commands(void) {
    if (g_device == VK_NULL_HANDLE) return 0;
    if (!g_commands_active) return 1;
    if (!vulkan_cb_submit()) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL"))
            fprintf(stderr, "ds4: flush: cb_submit failed\n");
        g_commands_active = false;
        return 0;
    }
    g_commands_active = false;
    g_cmd_i = 1 - g_cmd_i;
    if (vulkan_cb_acquire(0) == VK_NULL_HANDLE) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL"))
            fprintf(stderr, "ds4: flush: cb_acquire failed\n");
        return 0;
    }
    g_commands_active = true;
    if (getenv("DS4_VULKAN_DEBUG_SUBMIT") != NULL) g_dbg_scope_t0_ms = vulkan_now_ms();
    return 1;
}

int ds4_gpu_commands_active(void) {
    return g_commands_active ? 1 : 0;
}

int ds4_gpu_end_commands(void) {
    if (getenv("DS4_VULKAN_DEBUG_ONESHOT") != NULL) return 1;
    if (!g_commands_active) return 1;
    if (getenv("DS4_VULKAN_DEBUG_SUBMIT") != NULL) {
        const double e0 = vulkan_now_ms();
        if (!vulkan_cb_submit()) {
            g_commands_active = false;
            return 0;
        }
        const double e1 = vulkan_now_ms();
        VkFence fence = g_cmd_fence[g_cmd_i];
        double wq = 0.0;
        if (fence != VK_NULL_HANDLE) {
            vkWaitForFences(g_device, 1, &fence, VK_TRUE, UINT64_MAX);
            wq = vulkan_now_ms() - e1;
            /* Clear the in-flight marker only, exactly like vulkan_cb_acquire.
             * The CB's persistent fence stays in g_cb_fence[] and is
             * reset+reused by the next vulkan_cb_submit; the readback fence
             * (which may occupy this slot for a signal-submitted scope) is a
             * global owned/reset by the signal path. Destroying either here
             * left a dangling handle and crashed the next reset. */
            g_cmd_fence[g_cmd_i] = VK_NULL_HANDLE;
        }
        if (g_ts_on) {
            /* Tag the scope with the active device so per-layer GPU time can
             * be aggregated per tier (static vs dynamic). */
            char ts_tag[32];
            snprintf(ts_tag, sizeof(ts_tag), "scope tier=%d", g_vk_ctx_active);
            vulkan_ts_report(ts_tag);
        } else {
            vulkan_ts_report("scope");
        }
        const double e3 = vulkan_now_ms();
        g_commands_active = false;
        fprintf(stderr, "ds4: Vulkan debug submit: host=%.3f ms end=%.3f ms queue=%.3f ms fwait=%.3f ms total=%.3f ms ndisp=%u binds=%u bytes=%llu\n",
                g_dbg_scope_t0_ms > 0.0 ? e0 - g_dbg_scope_t0_ms : 0.0,
                e1 - e0, 0.0, wq, e3 - e0, g_scope_dispatch_count,
                g_scope_bind_count, (unsigned long long)g_scope_bind_bytes);
        g_scope_bind_bytes = 0;
        g_scope_bind_count = 0;
        g_scope_dispatch_count = 0;
        return 1;
    }
    g_commands_active = false;
    if (getenv("DS4_VULKAN_DEBUG_DISPATCH") != NULL) {
        const double e0 = vulkan_now_ms();
        const int ok = vulkan_cb_submit();
        fprintf(stderr, "ds4: Vulkan debug: submit w=%llu rc=%d\n",
                (unsigned long long)vulkan_wall_ms(), (int)ok);
        (void)e0;
        return ok;
    }
    return vulkan_cb_submit();
}

int ds4_gpu_synchronize(void) {
    if (g_device == VK_NULL_HANDLE) return 0;
    vulkan_device_wait();
    /* Reset only when no scope is open: a recorded-but-unsubmitted CB still
     * references the pool's descriptor sets, and freeing them here would
     * leave the submit reading undefined sets.  (Matches the guard in
     * vulkan_device_wait_impl.) */
    if (g_desc_pool != VK_NULL_HANDLE && !g_commands_active) {
        vkResetDescriptorPool(g_device, g_desc_pool, 0);
        g_desc_sets_allocated = 0;
    }
    return 1;
}

/* Multi-tier correctness: the per-token logits readback only waits the head
 * tier's device, so another tier's still-in-flight scope (the previous
 * token's tail layers) can race the next token's embedding write into the
 * same double-buffered cur_hc slot on that tier.  Drain EVERY tier and
 * restore the previously active one. */
extern "C" void ds4_vulkan_synchronize_all_tiers(void) {
    if (getenv("DS4_TRACE") != NULL) {
        vulkan_trace("sync.all_tiers n=%d", g_vk_ctx_count);
    }
    if (g_vk_ctx_count <= 1) {
        if (g_device != VK_NULL_HANDLE) vulkan_device_wait();
        return;
    }
    const int prev = g_vk_ctx_active;
    for (int t = 0; t < g_vk_ctx_count; t++) {
        if (ds4_vulkan_set_current_device(t) == 0) vulkan_device_wait();
    }
    const int back = (prev >= 0 && prev < g_vk_ctx_count) ? prev : 0;
    (void)ds4_vulkan_set_current_device(back);
}

/* --- public contract: model map ------------------------------------------- */

/* The model windows requested by set_model_map / set_model_map_range /
 * set_model_map_spans are staged host-side into per-window tensors
 * (Fase 6 step 2 staging pool; vulkan_set_span_windows stages each span),
 * so weight-backed kernels bind the covering window at the tensor's byte
 * offset -- no device-side copy.  When a staging allocation fails, the
 * affected window is skipped and those kernels fail cleanly.  The host
 * pointer is still exported through lookup_cache for the host-visible paths
 * (non-streaming only). */

static const void *g_vulkan_model_map = NULL;
static uint64_t g_vulkan_model_size = 0;
static int g_vulkan_model_fd = -1;
static int g_vulkan_ssd_streaming = 0;

extern "C" void ds4_vulkan_set_ssd_streaming(int enabled) {
    g_vulkan_ssd_streaming = enabled ? 1 : 0;
}

/* Query the device-local heap size of the best physical device without a
 * full backend init (used by --gpu-vram auto, which runs during CLI parsing).
 * Creates a throwaway instance, prefers a discrete GPU, and returns the
 * largest DEVICE_LOCAL heap; 0 on failure. */
/* Largest DEVICE_LOCAL heap of a physical device (0 when none). */
static uint64_t vulkan_phys_vram_bytes(VkPhysicalDevice phys) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem);
    uint64_t vram = 0;
    for (uint32_t i = 0; i < mem.memoryHeapCount; i++) {
        if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            mem.memoryHeaps[i].size > vram) {
            vram = mem.memoryHeaps[i].size;
        }
    }
    return vram;
}

/* Enumerate and classify every physical device (see ds4_vulkan_mgpu.h).
 * Creates a throwaway instance when g_instance is not yet up. */
extern "C" int ds4_vulkan_probe_devices(ds4_vk_dev_info *out, int max_devices) {
    if (!out || max_devices <= 0) return 0;
    VkInstance inst = g_instance;
    int own_instance = 0;
    if (inst == VK_NULL_HANDLE) {
        VkApplicationInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.apiVersion = VK_API_VERSION_1_0;
        VkInstanceCreateInfo ici = {};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &ai;
        if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) return 0;
        own_instance = 1;
    }

    /* Fast/slow classification threshold (GB/s).  5.0 so a PCIe 4.0 x4 link
     * (~6-7 GB/s real) counts as FAST: a bifurcated 4x4x4x4 topology then
     * classifies all four GPUs as dynamic (dense resident + per-GPU hot expert
     * pool) instead of forcing an all-static plan that cannot fit the model.
     * Override with DS4_VULKAN_FAST_LINK_GBPS. */
    double fast_threshold = 5.0;
    const char *thr_env = getenv("DS4_VULKAN_FAST_LINK_GBPS");
    if (thr_env && *thr_env) {
        const double v = atof(thr_env);
        if (v > 0.0) fast_threshold = v;
    }

    uint32_t count = 0;
    int n = 0;
    if (vkEnumeratePhysicalDevices(inst, &count, NULL) == VK_SUCCESS && count) {
        VkPhysicalDevice *devs = (VkPhysicalDevice *)calloc(count, sizeof(*devs));
        if (devs &&
            vkEnumeratePhysicalDevices(inst, &count, devs) == VK_SUCCESS) {
            for (uint32_t i = 0; i < count && n < max_devices; i++) {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(devs[i], &props);
                ds4_vk_dev_info *d = &out[n];
                memset(d, 0, sizeof(*d));
                d->vk_index = i;
                snprintf(d->name, sizeof(d->name), "%s", props.deviceName);
                uint32_t qfamily = UINT32_MAX;
                const int has_queue =
                    vulkan_device_compute_queue(devs[i], &qfamily);
                const int is_cpu =
                    props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ||
                    props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU;
                d->usable = has_queue && !is_cpu;
                uint32_t dom = 0, bus = 0, dev = 0, fn = 0;
                if (vulkan_query_bdf_inst(inst, devs[i], &dom, &bus, &dev, &fn)) {
                    snprintf(d->bdf, sizeof(d->bdf), "%04x:%02x:%02x.%x",
                             dom, bus, dev, fn);
                } else {
                    snprintf(d->bdf, sizeof(d->bdf), "n/a");
                }
                d->vram_bytes = vulkan_phys_vram_bytes(devs[i]);
                d->bw_gbps = d->usable ? vulkan_probe_transfer_bw(devs[i]) : 0.0;
                d->is_fast = d->usable && d->bw_gbps >= fast_threshold;
                n++;
            }
        }
        free(devs);
    }
    if (own_instance) vkDestroyInstance(inst, NULL);
    return n;
}

extern "C" uint64_t ds4_vulkan_probe_vram_bytes(void) {
    VkApplicationInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) return 0;

    uint32_t count = 0;
    uint64_t vram = 0;
    if (vkEnumeratePhysicalDevices(inst, &count, NULL) == VK_SUCCESS &&
        count > 0) {
        VkPhysicalDevice *devs = (VkPhysicalDevice *)calloc(
                count, sizeof(*devs));
        if (devs &&
            vkEnumeratePhysicalDevices(inst, &count, devs) == VK_SUCCESS) {
            VkPhysicalDevice chosen = VK_NULL_HANDLE;
            for (int pass = 0; pass < 2 && chosen == VK_NULL_HANDLE; pass++) {
                for (uint32_t i = 0; i < count; i++) {
                    VkPhysicalDeviceProperties props;
                    vkGetPhysicalDeviceProperties(devs[i], &props);
                    if (props.apiVersion < VK_API_VERSION_1_0) continue;
                    if (pass == 0 &&
                        props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                        continue;
                    }
                    chosen = devs[i];
                    break;
                }
            }
            if (chosen != VK_NULL_HANDLE) {
                VkPhysicalDeviceMemoryProperties mem;
                vkGetPhysicalDeviceMemoryProperties(chosen, &mem);
                for (uint32_t i = 0; i < mem.memoryHeapCount; i++) {
                    if ((mem.memoryHeaps[i].flags &
                         VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
                        mem.memoryHeaps[i].size > vram) {
                        vram = mem.memoryHeaps[i].size;
                    }
                }
            }
        }
        free(devs);
    }
    vkDestroyInstance(inst, NULL);
    return vram;
}

/* Drop every staged model window.  The windows' buffers may be read by an
 * in-flight submit (SSD streaming re-points between command scopes), so
 * settle the GPU first, then reclaim the descriptor sets that the completed
 * scopes accumulated (the pool is only otherwise reset at synchronize). */
static void vulkan_destroy_model_wrapper(void) {
    vulkan_device_wait();
    for (uint32_t i = 0; i < g_model_window_count; i++) {
        ds4_gpu_tensor_free(g_model_windows[i].tensor);
        g_model_windows[i].tensor = NULL;
        g_model_windows[i].buffer = VK_NULL_HANDLE;
    }
    g_model_window_count = 0;
    if (g_desc_pool != VK_NULL_HANDLE && !g_commands_active) {
        vkResetDescriptorPool(g_device, g_desc_pool, 0);
        g_desc_sets_allocated = 0;
    }
}

/* Stage one model window [host_ptr + base, host_ptr + base + size) into a
 * device-local (VRAM) tensor (Fase 6 step 4b: the static decode map is
 * uploaded once and read from VRAM on every token, never re-read from PCIe).
 * On UMA the upload is a direct memcpy; on a discrete GPU it is a one-shot
 * staging copy.  Returns 0 on allocation failure (the caller keeps the
 * previous windows); returns 1 when the window is too large to stage (the
 * streaming path covers it per layer instead). */
static int vulkan_add_model_window(const void *host_ptr, uint64_t base,
                                   uint64_t size) {
    if (g_device == VK_NULL_HANDLE || host_ptr == NULL || size == 0) return 0;
    if (size > DS4_VK_MAX_STAGED_WINDOW) return 1;
    if (g_model_window_count >= DS4_VK_MAX_MODEL_WINDOWS) return 0;

    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc_device_local(size);
    if (!t) return 0;
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(t);
    if (!h) {
        ds4_gpu_tensor_free(t);
        return 0;
    }
    if (!vulkan_upload_to_tensor(t, 0, (const uint8_t *)host_ptr + base,
                                 size)) {
        ds4_gpu_tensor_free(t);
        return 0;
    }

    struct ds4_vk_model_window *w = &g_model_windows[g_model_window_count];
    w->buffer = h->buffer;
    w->tensor = t;
    w->base = base;
    w->size = size;
    w->buf_offset = 0;
    g_model_window_count++;
    if (getenv("DS4_VULKAN_DEBUG_BINDS") != NULL && g_vk_bda != NULL) {
        VkBufferDeviceAddressInfo bdai = {};
        bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bdai.buffer = h->buffer;
        const uint64_t va = g_vk_bda(g_device, &bdai);
        fprintf(stderr, "ds4: Vulkan debug: window[%u] base=%llu size=%llu va=0x%llx..0x%llx\n",
                g_model_window_count - 1u, (unsigned long long)base,
                (unsigned long long)size,
                (unsigned long long)va, (unsigned long long)(va + size));
    }
    return 1;
}

/* Clear the windows and stage a single window [base, base+size). */
static void vulkan_set_single_window(const void *host_ptr, uint64_t base,
                                     uint64_t size) {
    vulkan_destroy_model_wrapper();
    vulkan_add_model_window(host_ptr, base, size);
}

/* Clear the windows and stage each requested span (aligned to the window
 * alignment) as its own window, coalescing spans whose gap is small.  A
 * whole-model span set (spread across the interleaved file) becomes many
 * windows instead of one impossibly large union.
 *
 * Fase 6 step 4b: the windows are device-local (VRAM) and the whole set is
 * uploaded with ONE staging buffer and ONE transfer submit (not one wait per
 * window), so re-staging the static decode map or a prefill layer costs one
 * device wait + one transfer, not hundreds. */
/* Split [lo,hi) into the sub-ranges NOT covered by the persistent per-device
 * weight cache, so a decode layer re-stages only tensors that are not already
 * resident.  The cache holds per-tensor windows (coalesced up to ~64 MiB),
 * while a decode span is the whole layer's dense region (hundreds of MiB), so
 * the old whole-span coverage check (vulkan_weight_cache_for) never matched
 * and re-uploaded the entire dense region on every layer of every token.
 * Returns the number of gaps written to out (capped at cap). */
#define DS4_VK_SPAN_GAP_MAX 512
static uint32_t vulkan_subtract_cached_range(uint64_t lo, uint64_t hi,
                                             struct ds4_vk_model_span *out,
                                             uint32_t cap) {
    uint32_t n = 0;
    uint64_t cur = lo;
    while (cur < hi && n < cap) {
        int covered = 0;
        for (uint32_t i = 0; i < g_weight_cache_count; i++) {
            const struct ds4_vk_model_window *w = &g_weight_cache[i];
            if (w->size == 0 || w->base > cur) continue;
            if (cur - w->base >= w->size) continue;   /* ends at/before cur */
            const uint64_t wend = w->base + w->size;
            cur = wend > cur ? wend : hi;
            covered = 1;
            break;
        }
        if (covered) continue;
        uint64_t next = hi;
        for (uint32_t i = 0; i < g_weight_cache_count; i++) {
            const uint64_t b = g_weight_cache[i].base;
            if (g_weight_cache[i].size != 0 && b > cur && b < next) next = b;
        }
        out[n].lo = cur;
        out[n].hi = next;
        n++;
        cur = next;
    }
    return n;
}

static void vulkan_set_span_windows(const void *host_ptr, uint64_t model_size,
                                    const uint64_t *offsets,
                                    const uint64_t *sizes, uint32_t count) {
    struct ds4_vk_model_span spans[DS4_VK_MAX_MODEL_WINDOWS];
    struct ds4_vk_model_span gaps[DS4_VK_SPAN_GAP_MAX];
    uint32_t n = 0;
    for (uint32_t i = 0; i < count && n < DS4_VK_MAX_MODEL_WINDOWS; i++) {
        if (sizes[i] == 0 || offsets[i] > model_size ||
            sizes[i] > model_size - offsets[i]) {
            continue;
        }
        /* A span already resident in the persistent per-device weight cache
         * needs no transient window; stage only the sub-ranges not covered by
         * the cache (per-tensor), so the fixed dense part stays resident and
         * is never re-uploaded. */
        const uint64_t end = offsets[i] + sizes[i];
        const uint32_t ng = vulkan_subtract_cached_range(offsets[i], end, gaps,
                                                         DS4_VK_SPAN_GAP_MAX);
        for (uint32_t g = 0; g < ng && n < DS4_VK_MAX_MODEL_WINDOWS; g++) {
            spans[n].lo = gaps[g].lo & ~(g_host_pointer_align - 1);
            spans[n].hi = (gaps[g].hi + g_host_pointer_align - 1) &
                          ~(g_host_pointer_align - 1);
            n++;
        }
    }
    {
        uint64_t req_bytes = 0, gap_bytes = 0;
        for (uint32_t i = 0; i < count; i++) req_bytes += sizes[i];
        for (uint32_t i = 0; i < n; i++) gap_bytes += spans[i].hi - spans[i].lo;
        vulkan_trace("span req=%u req_bytes=%llu gap_spans=%u gap_bytes=%llu",
                     count, (unsigned long long)req_bytes, n,
                     (unsigned long long)gap_bytes);
    }
    if (n == 0) {
        if (g_model_window_count != 0) vulkan_destroy_model_wrapper();
        return;
    }
    /* Sort by lo, then coalesce spans closer than the merge gap. */
    for (uint32_t i = 1; i < n; i++) {
        for (uint32_t j = i; j > 0 && spans[j - 1].lo > spans[j].lo; j--) {
            struct ds4_vk_model_span t = spans[j - 1];
            spans[j - 1] = spans[j];
            spans[j] = t;
        }
    }
    uint32_t m = 0;
    for (uint32_t i = 1; i < n; i++) {
        const uint64_t gap = spans[i].lo - spans[m].hi;
        const uint64_t merged_hi =
            spans[i].hi > spans[m].hi ? spans[i].hi : spans[m].hi;
        const uint64_t merged_size = merged_hi - spans[m].lo;
        if (gap <= DS4_VK_WINDOW_MERGE_GAP &&
            merged_size <= DS4_VK_MAX_STAGED_WINDOW) {
            if (spans[i].hi > spans[m].hi) spans[m].hi = spans[i].hi;
        } else {
            m++;
            spans[m] = spans[i];
        }
    }
    uint32_t windows = m + 1;

    vulkan_destroy_model_wrapper();

    /* Batched upload: one device-local buffer covering every window, one
     * staging buffer, one transfer submit.  Falls back to per-window staging
     * when the total exceeds the staged-window cap (whole-model maps). */
    uint64_t total = 0;
    for (uint32_t i = 0; i < windows; i++) {
        total += spans[i].hi - spans[i].lo;
    }
    if (g_device == VK_NULL_HANDLE || g_commands_active ||
        total == 0 || total > DS4_VK_MAX_STAGED_WINDOW ||
        !vulkan_compute_init()) {
        for (uint32_t i = 0; i < windows; i++) {
            if (!vulkan_add_model_window(host_ptr, spans[i].lo,
                                         spans[i].hi - spans[i].lo)) {
                break;   /* keep the windows staged so far */
            }
        }
    } else {
        ds4_gpu_tensor *big = ds4_gpu_tensor_alloc_device_local(total);
        struct ds4_vulkan_tensor *bh = big ? vulkan_tensor_handle(big) : NULL;
        ds4_gpu_tensor *st = NULL;
        if (bh) st = ds4_gpu_tensor_alloc(total);
        struct ds4_vulkan_tensor *sh = st ? vulkan_tensor_handle(st) : NULL;
        if (bh && sh && sh->host_map) {
            uint64_t off = 0;
            unsigned char *smap = sh->host_map;
            for (uint32_t i = 0; i < windows; i++) {
                const uint64_t sz = spans[i].hi - spans[i].lo;
                memcpy(smap + off, (const uint8_t *)host_ptr + spans[i].lo,
                       sz);
                off += sz;
            }
            VkCommandBuffer cb = vulkan_dispatch_begin();
            if (cb) {
                VkBufferCopy regions[DS4_VK_MAX_MODEL_WINDOWS];
                off = 0;
                for (uint32_t i = 0; i < windows; i++) {
                    const uint64_t sz = spans[i].hi - spans[i].lo;
                    regions[i].srcOffset = sh->offset + off;
                    regions[i].dstOffset = bh->offset + off;
                    regions[i].size = sz;
                    off += sz;
                }
                vkCmdCopyBuffer(cb, sh->buffer, bh->buffer, windows, regions);
                VkMemoryBarrier mb = {};
                mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, &mb, 0, NULL, 0, NULL);
                if (vulkan_submit_one_shot()) {
                    off = 0;
                    for (uint32_t i = 0; i < windows; i++) {
                        const uint64_t sz = spans[i].hi - spans[i].lo;
                        struct ds4_vk_model_window *w =
                            &g_model_windows[g_model_window_count++];
                        w->buffer = bh->buffer;
                        w->tensor = (i == 0) ? big : NULL; /* owner: first */
                        w->base = spans[i].lo;
                        w->size = sz;
                        w->buf_offset = off;
                        off += sz;
                    }
                }
            }
            ds4_gpu_tensor_free(st);
        } else {
            if (big) ds4_gpu_tensor_free(big);
            if (st) ds4_gpu_tensor_free(st);
            for (uint32_t i = 0; i < windows; i++) {
                if (!vulkan_add_model_window(host_ptr, spans[i].lo,
                                             spans[i].hi - spans[i].lo)) {
                    break;
                }
            }
        }
    }
    if (getenv("DS4_VULKAN_DEBUG_BINDS") != NULL) {
        uint64_t total2 = 0, maxw = 0;
        for (uint32_t i = 0; i < g_model_window_count; i++) {
            total2 += g_model_windows[i].size;
            if (g_model_windows[i].size > maxw) maxw = g_model_windows[i].size;
        }
        fprintf(stderr, "ds4: Vulkan debug: staged %u/%u windows (%.2f GiB, max %.2f GiB)\n",
                g_model_window_count, windows,
                (double)total2 / 1073741824.0,
                (double)maxw / 1073741824.0);
    }
}

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    g_vulkan_model_map = model_map;
    g_vulkan_model_size = model_size;
    if (model_map == NULL || model_size == 0) {
        vulkan_destroy_model_wrapper();
        return 1;
    }
    vulkan_set_single_window(model_map, 0, model_size);
    return 1;
}

int ds4_gpu_set_model_fd(int fd) {
    g_vulkan_model_fd = fd;
    return 1;
}

int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    g_vulkan_model_fd = fd;
    g_vulkan_model_map = model_map;
    return 1;
}

/* Multi-tier startup registers the host mmap WITHOUT staging any window (the
 * per-tier selective caches are not implemented on Vulkan; weights are staged
 * on demand by the model-window path per active tier). */
int ds4_gpu_register_model_map_no_copy(const void *model_map,
                                       uint64_t model_size) {
    g_vulkan_model_map = model_map;
    g_vulkan_model_size = model_size;
    return 1;
}

int ds4_gpu_build_derived_artifacts(const void *model_map, uint64_t model_size,
                                    const char *model_path) {
    (void)model_map; (void)model_size; (void)model_path;
    return 1;
}

int ds4_gpu_model_range_replaced(const void *model_map, uint64_t offset,
                                 uint64_t bytes) {
    (void)model_map; (void)offset; (void)bytes;
    return 1;
}

int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size,
                                uint64_t map_offset, uint64_t map_size,
                                uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (map_offset >= model_size || map_size == 0 ||
        map_size > model_size - map_offset) {
        return ds4_gpu_set_model_map(model_map, model_size);
    }
    const uint64_t base = map_offset & ~(g_host_pointer_align - 1);
    const uint64_t end = (map_offset + map_size + g_host_pointer_align - 1) &
                         ~(g_host_pointer_align - 1);
    vulkan_set_single_window(model_map, base, end - base);
    return 1;
}

int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size,
                                const uint64_t *offsets, const uint64_t *sizes,
                                uint32_t count, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (model_map == NULL || offsets == NULL || sizes == NULL || count == 0) {
        return ds4_gpu_set_model_map(model_map, model_size);
    }
    /* SSD streaming re-points the staged windows per layer (decode) or per
     * span set.  A single contiguous window over the requested spans would
     * cover the whole interleaved GGUF (this model's per-layer tensors are
     * spread across the file), so each span is staged as its own window and
     * weight binds resolve to the covering window. */
    vulkan_set_span_windows(model_map, model_size, offsets, sizes, count);
    return 1;
}

/* Stage one model-file range into device-local VRAM and keep it resident for
 * the whole session.  Returns 1 on success (or when already cached), 0 on
 * allocation/upload failure. */
static int vulkan_weight_cache_add(const void *host_ptr, uint64_t base,
                                   uint64_t size) {
    if (g_device == VK_NULL_HANDLE || host_ptr == NULL || size == 0) return 0;
    if (g_weight_cache_count >= DS4_VK_MAX_MODEL_WINDOWS) return 0;
    if (vulkan_weight_cache_for(base, size) != NULL) return 1;

    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc_device_local(size);
    if (!t) return 0;
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(t);
    if (!h || !vulkan_upload_to_tensor(t, 0,
                                       (const uint8_t *)host_ptr + base,
                                       size)) {
        ds4_gpu_tensor_free(t);
        return 0;
    }
    struct ds4_vk_model_window *w = &g_weight_cache[g_weight_cache_count];
    w->buffer = h->buffer;
    w->tensor = t;
    w->base = base;
    w->size = size;
    w->buf_offset = 0;
    g_weight_cache_count++;
    return 1;
}

/* Multi-GPU (Fase 8e): the engine hands each tier the model-file ranges it
 * owns (dense tensors everywhere; the routed experts too on the static
 * tiers).  Stage them into device-local VRAM once: decode weight binds then
 * resolve from the persistent cache and no longer re-upload per token.  The
 * device_id is the Vulkan physical-device index recorded in
 * g_vk_ctx[].device_index.  Returns 0 on success. */
int ds4_gpu_device_cache_tensors(int device_id,
                                 const ds4_tensor_range *ranges,
                                 int n_ranges) {
    if (ranges == NULL || n_ranges <= 0) return 0;
    if (g_vulkan_model_map == NULL || g_vulkan_model_size == 0) {
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "device_cache_tensors: model map not registered\n");
        return 1;
    }
    int tier = -1;
    for (int i = 0; i < g_vk_ctx_count; i++) {
        if (g_vk_ctx[i].device_index == (uint32_t)device_id) {
            tier = i;
            break;
        }
    }
    if (tier < 0) {
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "device_cache_tensors: no tier for device %d (have %d: [",
                device_id, g_vk_ctx_count);
        for (int i = 0; i < g_vk_ctx_count; i++) {
            fprintf(stderr, "%s%u", i ? "," : "",
                    (unsigned)g_vk_ctx[i].device_index);
        }
        fprintf(stderr, "])\n");
        return 1;
    }
    const int prev = g_vk_ctx_active;
    if (ds4_vulkan_set_current_device(tier) != 0) return 1;

    uint64_t total = 0;
    for (int i = 0; i < n_ranges; i++) {
        const ds4_tensor_range *r = &ranges[i];
        if (r->bytes == 0) continue;
        if (r->source_offset > g_vulkan_model_size ||
            r->bytes > g_vulkan_model_size - r->source_offset) {
            fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                    "device_cache_tensors: tier %d range [%llu,+%llu) out of "
                    "model bounds (%llu)\n",
                    tier, (unsigned long long)r->source_offset,
                    (unsigned long long)r->bytes,
                    (unsigned long long)g_vulkan_model_size);
            if (prev >= 0) (void)ds4_vulkan_set_current_device(prev);
            return 1;
        }
        if (!vulkan_weight_cache_add(g_vulkan_model_map, r->source_offset,
                                     r->bytes)) {
            fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                    "device_cache_tensors: tier %d range [%llu,+%llu) staging "
                    "failed (%.2f GiB in cache so far)\n",
                    tier, (unsigned long long)r->source_offset,
                    (unsigned long long)r->bytes,
                    (double)total / 1073741824.0);
            if (prev >= 0) (void)ds4_vulkan_set_current_device(prev);
            return 1;
        }
        total += r->bytes;
    }
    /* A static tier owns every weight of its layers, so decode never needs a
     * transient window there; record that so the engine can skip per-layer
     * restaging.  A dynamic tier only caches its dense tensors (experts are
     * streamed per layer), so it stays non-resident. */
    if (g_dev_mode == 0) g_weight_cache_ready = 1;
    fprintf(stderr, DS4_VULKAN_LOG_PREFIX
            "tier %d weight cache: %.2f GiB resident (%u entries, mode=%s)\n",
            tier, (double)total / 1073741824.0, g_weight_cache_count,
            g_dev_mode == 0 ? "static" : "dynamic");

    if (prev >= 0) (void)ds4_vulkan_set_current_device(prev);
    return 0;
}

/* --- Fase 8e: parallel per-tier weight-cache staging ---------------------
 *
 * Each logical tier owns an independent VkDevice/queue, so the four PCIe
 * links can be filled concurrently instead of one after another.  These
 * workers touch only their own ds4_vk_dev_ctx (device, queue, mem types) and
 * never the shared backend globals, so they run in parallel safely; the
 * results are merged into the per-tier cache on the main thread after join.
 * The raw VkBuffer/VkDeviceMemory are owned by the cache (tensor == NULL) and
 * released by vkDestroyDevice at shutdown. */

struct vulkan_par_range {
    uint64_t base;
    uint64_t size;
    VkBuffer buffer;
};

struct vulkan_par_job {
    struct ds4_vk_dev_ctx *ctx;
    int                    tier;
    const void            *host_ptr;
    const ds4_tensor_range *ranges;
    int                    n_ranges;
    int                    ok;
    double                 host_ms;
    double                 gpu_ms;
    struct vulkan_par_range out[DS4_VK_MAX_MODEL_WINDOWS];
    uint32_t               n_out;
};

static VkBuffer vulkan_par_buffer(VkDevice dev,
                                  const VkPhysicalDeviceMemoryProperties *mp,
                                  uint32_t mem_type, uint64_t bytes,
                                  VkDeviceMemory *out_mem,
                                  unsigned char **out_map) {
    if (mem_type == UINT32_MAX) return VK_NULL_HANDLE;
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf = VK_NULL_HANDLE;
    if (vkCreateBuffer(dev, &bci, NULL, &buf) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, buf, &req);
    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mem_type;
    if (vkAllocateMemory(dev, &mai, NULL, out_mem) != VK_SUCCESS) {
        vkDestroyBuffer(dev, buf, NULL);
        return VK_NULL_HANDLE;
    }
    if (vkBindBufferMemory(dev, buf, *out_mem, 0) != VK_SUCCESS) {
        vkFreeMemory(dev, *out_mem, NULL);
        vkDestroyBuffer(dev, buf, NULL);
        return VK_NULL_HANDLE;
    }
    if (out_map) {
        *out_map = NULL;
        if ((mp->memoryTypes[mem_type].propertyFlags &
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            vkMapMemory(dev, *out_mem, 0, VK_WHOLE_SIZE, 0,
                        (void **)out_map) != VK_SUCCESS) {
            vkFreeMemory(dev, *out_mem, NULL);
            vkDestroyBuffer(dev, buf, NULL);
            return VK_NULL_HANDLE;
        }
    }
    return buf;
}

static void *vulkan_par_job_main(void *arg) {
    struct vulkan_par_job *j = (struct vulkan_par_job *)arg;
    struct ds4_vk_dev_ctx *c = j->ctx;
    VkDevice dev = c->device;
    const uint32_t dmem = c->is_uma ? c->uma_mem_type
                                    : c->device_local_mem_type;
    const uint32_t hmem = c->host_visible_mem_type != UINT32_MAX
                          ? c->host_visible_mem_type : 0u;
    j->ok = 1;

    VkCommandPoolCreateInfo pci = {};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = c->queue_family;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(dev, &pci, NULL, &pool) != VK_SUCCESS) {
        j->ok = 0;
        return NULL;
    }
    VkCommandBufferAllocateInfo cbi = {};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    int setup_ok = vkAllocateCommandBuffers(dev, &cbi, &cb) == VK_SUCCESS;
    if (setup_ok) {
        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        setup_ok = vkCreateFence(dev, &fci, NULL, &fence) == VK_SUCCESS;
    }

    for (int i = 0; i < j->n_ranges && j->ok && setup_ok; i++) {
        const ds4_tensor_range *r = &j->ranges[i];
        if (r->bytes == 0) continue;
        VkDeviceMemory dm = VK_NULL_HANDLE;
        VkBuffer db = vulkan_par_buffer(dev, &c->mem_props, dmem, r->bytes,
                                        &dm, NULL);
        VkDeviceMemory sm = VK_NULL_HANDLE;
        unsigned char *smap = NULL;
        VkBuffer sb = db ? vulkan_par_buffer(dev, &c->mem_props, hmem,
                                             r->bytes, &sm, &smap)
                         : VK_NULL_HANDLE;
        if (!db || !sb || !smap) {
            if (sb) { vkDestroyBuffer(dev, sb, NULL);
                      vkFreeMemory(dev, sm, NULL); }
            if (db) { vkDestroyBuffer(dev, db, NULL);
                      vkFreeMemory(dev, dm, NULL); }
            j->ok = 0;
            break;
        }
        struct timespec th0, th1, tg0, tg1;
        clock_gettime(CLOCK_MONOTONIC, &th0);
        memcpy(smap, (const unsigned char *)j->host_ptr + r->source_offset,
               r->bytes);
        clock_gettime(CLOCK_MONOTONIC, &th1);
        j->host_ms += (double)(th1.tv_sec - th0.tv_sec) * 1000.0 +
                      (double)(th1.tv_nsec - th0.tv_nsec) / 1.0e6;
        clock_gettime(CLOCK_MONOTONIC, &tg0);

        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VkBufferCopy region = { 0, 0, r->bytes };
        VkMemoryBarrier mb = {};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        int ok = vkResetCommandBuffer(cb, 0) == VK_SUCCESS &&
                 vkBeginCommandBuffer(cb, &bi) == VK_SUCCESS;
        if (ok) {
            vkCmdCopyBuffer(cb, sb, db, 1, &region);
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &mb, 0, NULL, 0, NULL);
            ok = vkEndCommandBuffer(cb) == VK_SUCCESS;
        }
        if (ok) {
            vkResetFences(dev, 1, &fence);
            ok = vkQueueSubmit(c->queue, 1, &si, fence) == VK_SUCCESS &&
                 vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX) ==
                     VK_SUCCESS;
        }
        clock_gettime(CLOCK_MONOTONIC, &tg1);
        j->gpu_ms += (double)(tg1.tv_sec - tg0.tv_sec) * 1000.0 +
                     (double)(tg1.tv_nsec - tg0.tv_nsec) / 1.0e6;
        vkUnmapMemory(dev, sm);
        vkDestroyBuffer(dev, sb, NULL);
        vkFreeMemory(dev, sm, NULL);
        if (!ok) {
            vkDestroyBuffer(dev, db, NULL);
            vkFreeMemory(dev, dm, NULL);
            j->ok = 0;
            break;
        }
        j->out[j->n_out].base = r->source_offset;
        j->out[j->n_out].size = r->bytes;
        j->out[j->n_out].buffer = db;
        j->n_out++;
    }

    if (!setup_ok) j->ok = 0;
    if (fence) vkDestroyFence(dev, fence, NULL);
    if (cb) vkFreeCommandBuffers(dev, pool, 1, &cb);
    if (pool) vkDestroyCommandPool(dev, pool, NULL);
    return NULL;
}

/* Stage every tier's weight ranges in parallel (one worker thread per
 * device), then merge the results into the per-tier persistent cache.  The
 * device_id -> tier mapping is the same as ds4_gpu_device_cache_tensors. */
extern "C" int ds4_vulkan_device_cache_tensors_multi(
        int n, const int *device_ids,
        const ds4_tensor_range *const *ranges, const int *counts) {
    if (n <= 0) return 0;
    if (g_vulkan_model_map == NULL || g_vulkan_model_size == 0) {
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "device_cache_tensors_multi: model map not registered\n");
        return 1;
    }
    struct vulkan_par_job *jobs =
        (struct vulkan_par_job *)calloc((size_t)n, sizeof(*jobs));
    pthread_t *th = (pthread_t *)calloc((size_t)n, sizeof(*th));
    int *started = (int *)calloc((size_t)n, sizeof(*started));
    if (!jobs || !th || !started) {
        free(jobs); free(th); free(started);
        return 1;
    }

    int rc = 0;
    int launched = 0;
    const int sequential = getenv("DS4_VULKAN_LOAD_SEQUENTIAL") != NULL;
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int i = 0; i < n; i++) {
        int tier = -1;
        for (int t = 0; t < g_vk_ctx_count; t++) {
            if (g_vk_ctx[t].device_index == (uint32_t)device_ids[i]) {
                tier = t;
                break;
            }
        }
        if (tier < 0) {
            fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                    "device_cache_tensors_multi: no tier for device %d\n",
                    device_ids[i]);
            rc = 1;
            break;
        }
        jobs[i].ctx = &g_vk_ctx[tier];
        jobs[i].tier = tier;
        jobs[i].host_ptr = g_vulkan_model_map;
        jobs[i].ranges = ranges[i];
        jobs[i].n_ranges = counts[i];
        if (sequential) {
            vulkan_par_job_main(&jobs[i]);
            started[i] = 1;
            launched++;
            continue;
        }
        if (pthread_create(&th[i], NULL, vulkan_par_job_main, &jobs[i]) != 0) {
            fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                    "device_cache_tensors_multi: thread %d failed to start\n",
                    i);
            rc = 1;
            break;
        }
        started[i] = 1;
        launched++;
    }
    if (!sequential) {
        for (int i = 0; i < launched; i++) pthread_join(th[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    const double load_ms = (double)(ts1.tv_sec - ts0.tv_sec) * 1000.0 +
                           (double)(ts1.tv_nsec - ts0.tv_nsec) / 1.0e6;
    fprintf(stderr, DS4_VULKAN_LOG_PREFIX
            "parallel weight cache load: %.1f ms across %d tiers (parallel)\n",
            load_ms, launched);

    for (int i = 0; i < launched && rc == 0; i++) {
        struct ds4_vk_dev_ctx *c = jobs[i].ctx;
        if (!jobs[i].ok) {
            fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                    "device_cache_tensors_multi: tier %d staging failed\n",
                    jobs[i].tier);
            rc = 1;
            break;
        }
        uint64_t total = 0;
        for (uint32_t k = 0; k < jobs[i].n_out; k++) {
            if (c->weight_cache_count >= DS4_VK_MAX_MODEL_WINDOWS) {
                rc = 1;
                break;
            }
            struct ds4_vk_model_window *w =
                &c->weight_cache[c->weight_cache_count++];
            w->buffer = jobs[i].out[k].buffer;
            w->tensor = NULL;   /* raw buffer, freed by vkDestroyDevice */
            w->base = jobs[i].out[k].base;
            w->size = jobs[i].out[k].size;
            w->buf_offset = 0;
            c->device_local_bytes += jobs[i].out[k].size;
            total += jobs[i].out[k].size;
        }
        if (c->dev_mode == 0) c->weight_cache_ready = 1;
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "tier %d weight cache: %.2f GiB resident (%u entries, mode=%s, "
                "parallel) host=%.1f ms gpu=%.1f ms\n",
                jobs[i].tier, (double)total / 1073741824.0,
                c->weight_cache_count,
                c->dev_mode == 0 ? "static" : "dynamic",
                jobs[i].host_ms, jobs[i].gpu_ms);
    }

    /* The active tier's globals were loaded before this merge; reload them so
     * weight binds see the freshly cached ranges. */
    if (g_vk_ctx_active >= 0 && g_vk_ctx_active < g_vk_ctx_count) {
        vulkan_ctx_load(&g_vk_ctx[g_vk_ctx_active]);
    }
    free(jobs);
    free(th);
    free(started);
    return rc;
}

/* Debug: per-tier device-local accounting.  Called by the engine only under
 * DS4_VULKAN_DEBUG_MEM, e.g. right after the multi-tier graph session is
 * allocated, to see how much VRAM each GPU is really holding (cache + graph +
 * pool) versus its budget. */
extern "C" void ds4_vulkan_debug_mem_report(const char *tag) {
    if (getenv("DS4_VULKAN_DEBUG_MEM") == NULL) return;
    for (int t = 0; t < g_vk_ctx_count; t++) {
        const struct ds4_vk_dev_ctx *c = &g_vk_ctx[t];
        const uint32_t mt = c->is_uma ? c->uma_mem_type
                                      : c->device_local_mem_type;
        uint64_t heap = 0;
        if (mt != UINT32_MAX && c->mem_props.memoryTypeCount != 0) {
            const uint32_t h = c->mem_props.memoryTypes[mt].heapIndex;
            if (h < c->mem_props.memoryHeapCount) {
                heap = (uint64_t)c->mem_props.memoryHeaps[h].size;
            }
        }
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "mem[%s] tier %d device %u: device_local=%.2f GiB "
                "pool=%.2f GiB heap=%.2f GiB slack=%.2f GiB\n",
                tag ? tag : "?", t, c->device_index,
                (double)c->device_local_bytes / 1073741824.0,
                (double)c->pool_total_bytes / 1073741824.0,
                (double)heap / 1073741824.0,
                (double)(heap > c->device_local_bytes
                         ? heap - c->device_local_bytes : 0) /
                    1073741824.0);
    }
}

int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size,
                              uint64_t offset, uint64_t bytes,
                              const char *label) {
    (void)model_map; (void)model_size; (void)offset; (void)bytes; (void)label;
    return 1;
}

int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size,
                               uint64_t offset, uint64_t bytes,
                               uint64_t in_dim, uint64_t out_dim,
                               const char *label) {
    (void)model_map; (void)model_size; (void)offset; (void)bytes;
    (void)in_dim; (void)out_dim; (void)label;
    return 1;
}

int ds4_gpu_lookup_cache(uint64_t source_offset, uint64_t bytes,
                         int *out_device_id, void **out_device_ptr) {
    (void)bytes;
    if (out_device_id) *out_device_id = 0;
    if (out_device_ptr) *out_device_ptr = NULL;
    if (g_vulkan_ssd_streaming) {
        /* Streaming mode: a host-pointer fallback would be consumed as a
         * device pointer by kernels; refuse it (ROCm parity). */
        return 0;
    }
    if (g_vulkan_model_map && source_offset < g_vulkan_model_size) {
        if (out_device_ptr) {
            *out_device_ptr = (void *)((const uint8_t *)g_vulkan_model_map +
                                       source_offset);
        }
        return 1;
    }
    return 0;
}

int ds4_gpu_lookup_cache_device(uint64_t source_offset, uint64_t bytes) {
    (void)source_offset; (void)bytes;
    return 0;
}

int ds4_gpu_preload_q4_expert_tables(const void *model_map, uint64_t model_size,
                                     uint64_t gate_offset, uint64_t up_offset,
                                     uint64_t down_offset,
                                     uint64_t gate_expert_bytes,
                                     uint64_t down_expert_bytes,
                                     uint32_t n_total_expert) {
    (void)model_map; (void)model_size; (void)gate_offset; (void)up_offset;
    (void)down_offset; (void)gate_expert_bytes; (void)down_expert_bytes;
    (void)n_total_expert;
    return 1;
}

/* --- public contract: elementwise and core kernels ----------------------- */

static int vulkan_dispatch_flat(
        VkPipeline pipeline,
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
        uint32_t n) {
    if (!out || !a || !b || n == 0) return 0;
    if (out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float) ||
        (c && c->bytes < (uint64_t)n * sizeof(float))) return 0;
    struct ds4_vk_params p = {};
    p.n = n;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, a);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, b);
    if (c) binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, c);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    const uint32_t groups = (n + 255u) / 256u;
    return vulkan_dispatch(pipeline, &p, sizeof(p), binds, nb, groups, 1, 1);
}

int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a,
                       const ds4_gpu_tensor *b, uint32_t n) {
    return vulkan_dispatch_flat(g_pipes[DS4_PIPE_ADD], out, a, b, NULL, n);
}

int ds4_gpu_add3_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a,
                        const ds4_gpu_tensor *b, const ds4_gpu_tensor *c,
                        uint32_t n) {
    return vulkan_dispatch_flat(g_pipes[DS4_PIPE_ADD3], out, a, b, c, n);
}

int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
                          const ds4_gpu_tensor *up, uint32_t n,
                          float clamp, float weight) {
    if (!out || !gate || !up || n == 0) return 0;
    if (out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
    struct ds4_vk_params p = {};
    p.n = n;
    p.clamp = clamp;
    p.weight = weight;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, gate);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, up);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    const uint32_t groups = (n + 255u) / 256u;
    return vulkan_dispatch(g_pipes[DS4_PIPE_SWIGLU], &p, sizeof(p), binds, nb,
                           groups, 1, 1);
}

/* GPU argmax over n_vocab F32 logits.  Tie-break: lower index wins. */
int ds4_gpu_argmax_tensor(ds4_gpu_tensor *out_idx,
                          const ds4_gpu_tensor *logits, uint32_t n_vocab) {
    if (!out_idx || !logits || n_vocab == 0) return 0;
    if (out_idx->bytes < sizeof(int32_t) ||
        logits->bytes < (uint64_t)n_vocab * sizeof(float)) return 0;
    struct ds4_vk_params p = {};
    p.n = n_vocab;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, logits);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out_idx);
    return vulkan_dispatch(g_pipes[DS4_PIPE_ARGMAX], &p, sizeof(p), binds, nb,
                           1, 1, 1);
}

/* Ascending bitonic sort of int32 rows.  row_width must be a power of two
 * (<= 2048, the shader threadgroup scratch cap). */
int ds4_gpu_sort_i32_rows_asc_tensor(
        ds4_gpu_tensor *dst, const ds4_gpu_tensor *src,
        uint32_t row_width, uint32_t n_rows) {
    if (!dst || !src || row_width == 0 || n_rows == 0) return 0;
    if ((row_width & (row_width - 1u)) != 0) return 0;
    if (row_width > 2048u) return 0;
    const uint64_t bytes = (uint64_t)row_width * n_rows * sizeof(int32_t);
    if (dst->bytes < bytes || src->bytes < bytes) return 0;
    struct ds4_vk_params p = {};
    p.n = row_width;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, src);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, dst);
    return vulkan_dispatch(g_pipes[DS4_PIPE_SORT_I32_ROWS_ASC], &p,
                           sizeof(p), binds, nb, 1, n_rows, 1);
}

/* --- RMS norm -------------------------------------------------------------- */

static int vulkan_rms_norm_common(VkPipeline pipe, ds4_gpu_tensor *out,
                                  const ds4_gpu_tensor *x, uint32_t n,
                                  uint32_t rows, float eps,
                                  uint64_t weight_offset,
                                  const void *model_map) {
    if (!out || !x || n == 0 || rows == 0) return 0;
    if (out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    struct ds4_vk_params p = {};
    p.n = n;
    p.rows = rows;
    p.eps = eps;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    if (weight_offset != (uint64_t)-1) {
        if (!model_map || !vulkan_model_range_ok(weight_offset,
                                                 (uint64_t)n * sizeof(float))) {
            return 0;
        }
        binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, weight_offset,
                                        (uint64_t)n * sizeof(float));
    }
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    return vulkan_dispatch(pipe, &p, sizeof(p), binds, nb, 1, rows, 1);
}

int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out,
                                  const ds4_gpu_tensor *x, uint32_t n,
                                  float eps) {
    return vulkan_rms_norm_common(g_pipes[DS4_PIPE_RMS_NORM_PLAIN], out, x,
                                  n, 1, eps, (uint64_t)-1, NULL);
}

int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out,
                                       const ds4_gpu_tensor *x, uint32_t n,
                                       uint32_t rows, float eps) {
    return vulkan_rms_norm_common(g_pipes[DS4_PIPE_RMS_NORM_PLAIN], out, x,
                                  n, rows, eps, (uint64_t)-1, NULL);
}

int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out,
                                   const ds4_gpu_tensor *x,
                                   const void *model_map,
                                   uint64_t model_size, uint64_t weight_offset,
                                   uint32_t n, float eps) {
    (void)model_size;
    return vulkan_rms_norm_common(g_pipes[DS4_PIPE_RMS_NORM_WEIGHT], out, x,
                                  n, 1, eps, weight_offset, model_map);
}

int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out,
                                        const ds4_gpu_tensor *x,
                                        const void *model_map,
                                        uint64_t model_size,
                                        uint64_t weight_offset,
                                        uint32_t n, uint32_t rows, float eps) {
    (void)model_size;
    return vulkan_rms_norm_common(g_pipes[DS4_PIPE_RMS_NORM_WEIGHT], out, x,
                                  n, rows, eps, weight_offset, model_map);
}

/* --- RoPE ------------------------------------------------------------------ */

/* pos_stride is the per-token position increment (1 for normal row layouts;
 * the compressor's compressed rows use stride == ratio). */
static int vulkan_rope_tail(ds4_gpu_tensor *x, uint32_t n_tok,
                            uint32_t n_head, uint32_t head_dim,
                            uint32_t n_rot, uint32_t pos0,
                            uint32_t pos_stride, uint32_t n_ctx_orig,
                            bool inverse, float freq_base, float freq_scale,
                            float ext_factor, float attn_factor,
                            float beta_fast, float beta_slow) {
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || n_rot == 0) {
        return 0;
    }
    if (n_rot > head_dim || (n_rot & 1u) != 0) return 0;
    if (x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_tok;
    p.index = n_head;
    p.n_rot = n_rot;
    p.pos0 = pos0;
    p.aux = pos_stride ? pos_stride : 1u;
    p.n_ctx_orig = n_ctx_orig;
    p.inverse = inverse ? 1 : 0;
    p.freq_base = freq_base;
    p.freq_scale = freq_scale;
    p.ext_factor = ext_factor;
    p.attn_factor = attn_factor;
    p.beta_fast = beta_fast;
    p.beta_slow = beta_slow;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, x);
    const uint64_t pairs = (uint64_t)n_tok * n_head * (n_rot / 2u);
    const uint32_t groups = (uint32_t)((pairs + 255u) / 256u);
    return vulkan_dispatch(g_pipes[DS4_PIPE_ROPE_TAIL], &p, sizeof(p),
                           binds, nb, groups, 1, 1);
}

int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok,
                             uint32_t n_head, uint32_t head_dim,
                             uint32_t n_rot, uint32_t pos0,
                             uint32_t n_ctx_orig, bool inverse,
                             float freq_base, float freq_scale,
                             float ext_factor, float attn_factor,
                             float beta_fast, float beta_slow) {
    return vulkan_rope_tail(x, n_tok, n_head, head_dim, n_rot, pos0, 1u,
                            n_ctx_orig, inverse, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
}

/* Fused Q/KV RMS norm over `rows` rows (composition of two
 * rms_norm_weight_rows dispatches; the CUDA fused kernel is equivalent). */
int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    return ds4_gpu_rms_norm_weight_rows_tensor(
                   q_out, q, model_map, model_size, q_weight_offset,
                   q_n, rows, eps) &&
           ds4_gpu_rms_norm_weight_rows_tensor(
                   kv_out, kv, model_map, model_size, kv_weight_offset,
                   kv_n, rows, eps);
}

/* Fused Q/KV RMS norm + KV RoPE tail (composition: two
 * rms_norm_weight_rows then the shared rope_tail over the KV heads at
 * position pos0 + row, matching dsv4_qkv_rms_norm_rows_kv_rope_kernel). */
int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        uint32_t                kv_n_head,
        uint32_t                kv_head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        bool                    inverse,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps) {
    if (!q_out || !q || !kv_out || !kv || !model_map || rows == 0 ||
        kv_n_head == 0 || kv_head_dim == 0 || n_rot == 0) {
        return 0;
    }
    if (kv_n != kv_n_head * kv_head_dim || n_rot > kv_head_dim) return 0;
    if (!ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                q_out, q, model_map, model_size, q_weight_offset, q_n,
                kv_out, kv, kv_weight_offset, kv_n, rows, eps)) {
        return 0;
    }
    return ds4_gpu_rope_tail_tensor(kv_out, rows, kv_n_head, kv_head_dim,
                                    n_rot, pos0, n_ctx_orig, inverse,
                                    freq_base, freq_scale, ext_factor,
                                    attn_factor, beta_fast, beta_slow);
}

/* --- matmul ---------------------------------------------------------------- */

/* Shared launcher for the weight-backed matmul kernels.  weight_bytes is
 * the exact byte length of the weight tensor. */
static int vulkan_matmul_common(VkPipeline pipe, ds4_gpu_tensor *out,
                                const void *model_map, uint64_t model_size,
                                uint64_t weight_offset, uint64_t weight_bytes,
                                uint64_t in_dim, uint64_t out_dim,
                                const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    if (in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size ||
        weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    if (!vulkan_model_range_ok(weight_offset, weight_bytes)) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.in_dim = (uint32_t)in_dim;
    p.out_dim = (uint32_t)out_dim;
    p.rows = (uint32_t)n_tok;
    p.blocks = (uint32_t)((in_dim + 31u) / 32u);
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, weight_offset,
                                    weight_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    return vulkan_dispatch(pipe, &p, sizeof(p), binds, nb,
                           (uint32_t)out_dim, (uint32_t)n_tok, 1);
}

/* Fase 7: ponte verso l'autotuner (SPECS_AUTOTUNE §2.4).  La variante v2 del
 * matmul Q8 preq (256 thread attivi + letture vettorizzate) è il default:
 * vince su 780M e 6900 XT (kbench: 0.386 vs 0.543 ms e 0.124 vs 0.290 ms).
 * DS4_VULKAN_FORCE_VARIANT è una lista separata da virgole di coppie
 * <slot>:<indice> (es. "Q8_PREQ:0,MOE_IQ2:1,MOE_DOWN:1") per bisect/debug.
 * Valutato per dispatch: costo trascurabile, e kbench misura v1 e v2 nello
 * stesso processo.  Con l'autotuner la scelta diventa una tabella misurata a
 * init (g_pipes[slot] = variante vincente per-device). */
static int vulkan_force_variant(const char *slot, const char *idx) {
    const char *v = getenv("DS4_VULKAN_FORCE_VARIANT");
    if (!v || !v[0]) return 0;
    const size_t sl = strlen(slot), il = strlen(idx);
    const char *p = v;
    for (;;) {
        const char *c = strchr(p, ',');
        const size_t n = c ? (size_t)(c - p) : strlen(p);
        if (n == sl + 1 + il && strncmp(p, slot, sl) == 0 &&
            p[sl] == ':' && strncmp(p + sl + 1, idx, il) == 0) {
            return 1;
        }
        if (!c) break;
        p = c + 1;
    }
    return 0;
}

static VkPipeline vulkan_pipe_q8_preq(void) {
    if (vulkan_force_variant("Q8_PREQ", "0")) {
        return g_pipes[DS4_PIPE_MATMUL_Q8_0_PREQ];
    }
    if (vulkan_force_variant("Q8_PREQ", "1")) {
        return g_pipes[DS4_PIPE_MATMUL_Q8_0_PREQ_V2];
    }
    if (g_has_int_dot && vulkan_force_variant("Q8_PREQ", "2")) {
        return g_pipes[DS4_PIPE_MATMUL_Q8_0_PREQ_V3];
    }
    /* Default: the hardware packed-dot v3 when available, else v2. */
    if (g_has_int_dot) {
        return g_pipes[DS4_PIPE_MATMUL_Q8_0_PREQ_V3];
    }
    return g_pipes[DS4_PIPE_MATMUL_Q8_0_PREQ_V2];
}

/* attn_output_low variant selection.  v3 (activation reuse) and v2 (DP4A)
 * need VK_KHR_shader_integer_dot_product; v3 additionally needs blocks<=128
 * and rank%8==0.  The init autotuner (vulkan_autotune_attn_out) pins the
 * fastest per device in g_attn_out_autotuned; DS4_VULKAN_FORCE_VARIANT=
 * ATTN_OUT_LOW:0|1|2 overrides for bisect.  Defaults: best supported. */
static int vulkan_attn_out_variant(uint32_t blocks, uint64_t rank) {
    const int v3_ok = g_has_int_dot && blocks <= 128u && (rank % 8u) == 0u &&
                      g_pipes[DS4_PIPE_ATTN_OUTPUT_LOW_Q8_V3] != VK_NULL_HANDLE;
    const int v2_ok = g_has_int_dot &&
                      g_pipes[DS4_PIPE_ATTN_OUTPUT_LOW_Q8_V2] != VK_NULL_HANDLE;
    int want = -1;
    if (vulkan_force_variant("ATTN_OUT_LOW", "0")) want = 0;
    else if (vulkan_force_variant("ATTN_OUT_LOW", "1")) want = 1;
    else if (vulkan_force_variant("ATTN_OUT_LOW", "2")) want = 2;
    else if (g_attn_out_autotuned >= 0) want = g_attn_out_autotuned;
    if (want == 2 && v3_ok) return 2;
    if (want == 1 && v2_ok) return 1;
    if (want == 0) return 0;
    if (v3_ok) return 2;
    if (v2_ok) return 1;
    return 0;
}

static VkPipeline vulkan_pipe_attn_output_low_variant(int v) {
    if (v == 2) return g_pipes[DS4_PIPE_ATTN_OUTPUT_LOW_Q8_V3];
    if (v == 1) return g_pipes[DS4_PIPE_ATTN_OUTPUT_LOW_Q8_V2];
    return g_pipes[DS4_PIPE_ATTN_OUTPUT_LOW_Q8];
}

/* Variante MoE IQ2_XXS gate/up/mid e Q2_K down (Fase 7).  Default v2 (dequant
 * vettorizzato + 256 lane attive): misura e2e scope MoE 2.47 -> 1.03 ms
 * mediana, smoke 396/396 su 780M e 6900 XT.  DS4_VULKAN_FORCE_VARIANT=
 * MOE_IQ2:0/1 e MOE_DOWN:0/1 per forzare.  Stesso pattern del Q8 preq
 * (ponte autotuner). */
static VkPipeline vulkan_pipe_moe_gate_iq2(void) {
    if (vulkan_force_variant("MOE_IQ2", "0")) {
        return g_pipes[DS4_PIPE_MOE_GATE_UP_MID_IQ2XXS];
    }
    return g_pipes[DS4_PIPE_MOE_GATE_UP_MID_IQ2XXS_V2];
}

static VkPipeline vulkan_pipe_moe_down_q2k(void) {
    if (vulkan_force_variant("MOE_DOWN", "0")) {
        return g_pipes[DS4_PIPE_MOE_DOWN_Q2K];
    }
    return g_pipes[DS4_PIPE_MOE_DOWN_Q2K_V2];
}

/* MXFP4 routed MoE: default v2 (split lanes so all 256 are active at the real
 * block counts).  MOE_MXFP4:0 forces the one-lane-per-block v1. */
static VkPipeline vulkan_pipe_moe_mxfp4_gate(void) {
    if (vulkan_force_variant("MOE_MXFP4", "0")) {
        return g_pipes[DS4_PIPE_MOE_GATE_UP_MID_MXFP4];
    }
    return g_pipes[DS4_PIPE_MOE_GATE_UP_MID_MXFP4_V2];
}

static VkPipeline vulkan_pipe_moe_mxfp4_down(void) {
    if (vulkan_force_variant("MOE_MXFP4", "0")) {
        return g_pipes[DS4_PIPE_MOE_DOWN_MXFP4];
    }
    return g_pipes[DS4_PIPE_MOE_DOWN_MXFP4_V2];
}

/* Prequantized Q8_0 matmul: quantize x into scratch int8 blocks + scales,
 * then a DOT4-style matmul over the prequantized activations.  The inline
 * fused kernel (DS4_PIPE_MATMUL_Q8_0) is kept as the fallback. */
static int vulkan_matmul_q8_0_preq(ds4_gpu_tensor *out,
                                   uint64_t weight_offset,
                                   uint64_t weight_bytes,
                                   uint32_t in_dim, uint32_t out_dim,
                                   uint32_t blocks,
                                   const ds4_gpu_tensor *x, uint32_t n_tok) {
    ds4_gpu_tensor *xq = vulkan_scratch_a((uint64_t)n_tok * blocks * 32u);
    ds4_gpu_tensor *xs = vulkan_scratch_b((uint64_t)n_tok * blocks * 4u);
    if (!xq || !xs) return 0;

    struct ds4_vk_params p = {};
    p.in_dim = in_dim;
    p.blocks = blocks;
    p.rows = n_tok;
    struct ds4_vk_bind qbinds[DS4_VK_MAX_BINDS];
    uint32_t qnb = 0;
    qbinds[qnb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    qbinds[qnb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, xq);
    qbinds[qnb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, xs);
    if (!vulkan_dispatch(g_pipes[DS4_PIPE_QUANTIZE_Q8_0], &p, sizeof(p),
                         qbinds, qnb, blocks, n_tok, 1)) {
        return 0;
    }

    p.out_dim = out_dim;
    struct ds4_vk_bind mbinds[DS4_VK_MAX_BINDS];
    uint32_t mnb = 0;
    mbinds[mnb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, xq);
    mbinds[mnb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, xs);
    mbinds[mnb++] = vulkan_bind_model(DS4_VK_BINDING_W, weight_offset,
                                      weight_bytes);
    mbinds[mnb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    return vulkan_dispatch(vulkan_pipe_q8_preq(), &p, sizeof(p),
                           mbinds, mnb, out_dim, n_tok, 1);
}

int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map,
                               uint64_t model_size, uint64_t weight_offset,
                               uint64_t in_dim, uint64_t out_dim,
                               const ds4_gpu_tensor *x, uint64_t n_tok) {
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (blocks == 0 || out_dim > UINT64_MAX / (blocks * 34u)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (!out || !x || !model_map) return 0;
    if (in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size ||
        weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    if (!vulkan_model_range_ok(weight_offset, weight_bytes)) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    if (vulkan_matmul_q8_0_preq(out, weight_offset, weight_bytes,
                                (uint32_t)in_dim, (uint32_t)out_dim,
                                (uint32_t)blocks, x, (uint32_t)n_tok)) {
        return 1;
    }
    return vulkan_matmul_common(g_pipes[DS4_PIPE_MATMUL_Q8_0], out, model_map,
                                model_size, weight_offset, weight_bytes,
                                in_dim, out_dim, x, n_tok);
}

int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map,
                              uint64_t model_size, uint64_t weight_offset,
                              uint64_t in_dim, uint64_t out_dim,
                              const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (in_dim == 0 || out_dim > UINT64_MAX / in_dim) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    return vulkan_matmul_common(g_pipes[DS4_PIPE_MATMUL_F16], out, model_map,
                                model_size, weight_offset, weight_bytes,
                                in_dim, out_dim, x, n_tok);
}

int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map,
                              uint64_t model_size, uint64_t weight_offset,
                              uint64_t in_dim, uint64_t out_dim,
                              const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (in_dim == 0 || out_dim > UINT64_MAX / in_dim) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(float);
    return vulkan_matmul_common(g_pipes[DS4_PIPE_MATMUL_F32], out, model_map,
                                model_size, weight_offset, weight_bytes,
                                in_dim, out_dim, x, n_tok);
}

/* Typed quant matmul: only Q8_0 (type 8) is implemented; anything else
 * returns 0 so the caller falls back to its CPU path. */
/* Dense Q4_K / Q4_0 matmul: out[tok][row] = dot(dequant(w[row]), x[tok]) with
 * a raw f32 activation.  Weight rows are blocks * block_bytes wide. */
static int vulkan_matmul_quant_dense(VkPipeline pipe, ds4_gpu_tensor *out,
                                     const void *model_map, uint64_t model_size,
                                     uint64_t weight_offset, uint32_t blocks,
                                     uint32_t block_bytes, uint64_t in_dim,
                                     uint64_t out_dim, const ds4_gpu_tensor *x,
                                     uint64_t n_tok) {
    if (!out || !x || !model_map || blocks == 0 || block_bytes == 0 ||
        in_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)blocks * block_bytes;
    if (out_dim > UINT64_MAX / row_bytes) return 0;
    const uint64_t weight_bytes = out_dim * row_bytes;
    if (weight_offset > model_size ||
        weight_bytes > model_size - weight_offset) {
        return 0;
    }
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) {
        return 0;
    }
    if (!vulkan_model_range_ok(weight_offset, weight_bytes)) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.in_dim = (uint32_t)in_dim;
    p.out_dim = (uint32_t)out_dim;
    p.rows = (uint32_t)n_tok;
    p.blocks = blocks;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, weight_offset,
                                    weight_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    return vulkan_dispatch(pipe, &p, sizeof(p), binds, nb,
                           (uint32_t)out_dim, (uint32_t)n_tok, 1);
}

int ds4_gpu_matmul_quant_tensor(ds4_gpu_tensor *out, const void *model_map,
                                uint64_t model_size, uint64_t weight_offset,
                                uint32_t weight_type, uint64_t in_dim,
                                uint64_t out_dim, const ds4_gpu_tensor *x,
                                uint64_t n_tok) {
    if (weight_type == 8u) {   /* DS4_TENSOR_Q8_0 */
        return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                          weight_offset, in_dim, out_dim, x,
                                          n_tok);
    }
    if (weight_type == 12u) {  /* DS4_TENSOR_Q4_K */
        if (in_dim % 256u != 0) return 0;
        return vulkan_matmul_quant_dense(
                g_pipes[DS4_PIPE_MATMUL_Q4K], out, model_map, model_size,
                weight_offset, (uint32_t)(in_dim / 256u), 144u, in_dim,
                out_dim, x, n_tok);
    }
    if (weight_type == 2u) {   /* DS4_TENSOR_Q4_0 */
        if (in_dim % 32u != 0) return 0;
        return vulkan_matmul_quant_dense(
                g_pipes[DS4_PIPE_MATMUL_Q4_0], out, model_map, model_size,
                weight_offset, (uint32_t)(in_dim / 32u), 18u, in_dim,
                out_dim, x, n_tok);
    }
    return 0;
}

/* --- matmul variants (Q8_0) ------------------------------------------------- */

/* Fused pair: two Q8_0 projections sharing x (out0 = W0@x, out1 = W1@x). */
int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t weight0_offset, uint64_t weight1_offset,
        uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map) return 0;
    if (!ds4_gpu_matmul_q8_0_tensor(out0, model_map, model_size,
                                    weight0_offset, in_dim, out0_dim, x,
                                    n_tok)) {
        return 0;
    }
    return ds4_gpu_matmul_q8_0_tensor(out1, model_map, model_size,
                                      weight1_offset, in_dim, out1_dim, x,
                                      n_tok);
}

/* Decode variants share the Q8_0 matmul semantics (the CUDA/Metal names pick
 * different launch shapes / reduction orders, not different math). */
int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                      weight_offset, in_dim, out_dim, x,
                                      n_tok);
}

int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                      weight_offset, in_dim, out_dim, x,
                                      n_tok);
}

int ds4_gpu_matmul_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                      weight_offset, in_dim, out_dim, x,
                                      n_tok);
}

int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                      weight_offset, in_dim, out_dim, x,
                                      (uint64_t)n_rows);
}

int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t weight0_offset, uint64_t weight1_offset,
        uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
        const ds4_gpu_tensor *x, uint32_t n_rows) {
    if (!out0 || !out1 || !x || !model_map) return 0;
    if (!ds4_gpu_matmul_q8_0_tensor(out0, model_map, model_size,
                                    weight0_offset, in_dim, out0_dim, x,
                                    (uint64_t)n_rows)) {
        return 0;
    }
    return ds4_gpu_matmul_q8_0_tensor(out1, model_map, model_size,
                                      weight1_offset, in_dim, out1_dim, x,
                                      (uint64_t)n_rows);
}

/* Q8_0 matmul with FP16 output: run the standard Q8_0 matmul into a scratch
 * f32 buffer, then convert f32->f16 with the packed conversion kernel (two
 * halves per aligned word, so no cross-block RMW race). */
int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor *out_h, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out_h || !x || !model_map) return 0;
    if (in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size ||
        out_dim > UINT64_MAX / ((in_dim + 31u) / 32u * 34u)) return 0;
    const uint64_t weight_bytes =
        out_dim * ((in_dim + 31u) / 32u) * 34u;
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < n_tok * in_dim * sizeof(float) ||
        out_h->bytes < n_tok * out_dim * sizeof(uint16_t)) return 0;
    if (n_tok * out_dim > UINT64_MAX / sizeof(float)) return 0;

    ds4_gpu_tensor *tmp = vulkan_scratch_a(n_tok * out_dim * sizeof(float));
    if (!tmp) return 0;
    if (!ds4_gpu_matmul_q8_0_tensor(tmp, model_map, model_size, weight_offset,
                                    in_dim, out_dim, x, n_tok)) {
        return 0;
    }
    const uint64_t count = n_tok * out_dim;
    if (count > UINT32_MAX) return 0;
    struct ds4_vk_params p = {};
    p.n = (uint32_t)count;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, tmp);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out_h);
    const uint32_t groups = (uint32_t)((count + 1u) / 2u + 255u) / 256u;
    return vulkan_dispatch(g_pipes[DS4_PIPE_F32_TO_F16], &p, sizeof(p),
                           binds, nb, groups, 1, 1);
}

/* Partial-K Q8_0 matmul (kslice).  weight rows span in_dim blocks; only the
 * [k_off, k_off+k_cnt) slice participates.  k_off/k_cnt multiples of 32. */
int ds4_gpu_matmul_q8_0_kslice_rows_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        uint64_t in_start, uint64_t in_count, const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    if (in_dim == 0 || out_dim == 0 || in_count == 0 || n_tok == 0) return 0;
    if ((in_start & 31u) != 0 || (in_count & 31u) != 0 ||
        in_start > in_dim || in_count > in_dim - in_start) return 0;
    const uint64_t full_blocks = (in_dim + 31u) / 32u;
    if (weight_offset > model_size ||
        out_dim > UINT64_MAX / (full_blocks * 34u)) return 0;
    const uint64_t weight_bytes = out_dim * full_blocks * 34u;
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < n_tok * in_count * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    if (!vulkan_model_range_ok(weight_offset, weight_bytes)) return 0;
    if (out_dim > UINT32_MAX || in_count > UINT32_MAX || n_tok > UINT32_MAX ||
        in_start / 32u > UINT32_MAX) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.in_dim = (uint32_t)in_count;
    p.out_dim = (uint32_t)out_dim;
    p.rows = (uint32_t)n_tok;
    p.blocks = (uint32_t)full_blocks;
    p.index = (uint32_t)(in_start / 32u);
    p.aux = (uint32_t)(in_count / 32u);
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, weight_offset,
                                    weight_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    return vulkan_dispatch(g_pipes[DS4_PIPE_MATMUL_Q8_0_KSLICE], &p,
                           sizeof(p), binds, nb, (uint32_t)out_dim,
                           (uint32_t)n_tok, 1);
}

int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    if (!x || x_elem_off > x->bytes / sizeof(float) ||
        k_cnt > x->bytes / sizeof(float) - x_elem_off) {
        return 0;
    }
    ds4_gpu_tensor *view = ds4_gpu_tensor_view(x, x_elem_off * sizeof(float),
                                               k_cnt * sizeof(float));
    if (!view) return 0;
    const int rc = ds4_gpu_matmul_q8_0_kslice_rows_tensor(
            out, model_map, model_size, weight_offset, full_in_dim, out_dim,
            k_off, k_cnt, view, 1u);
    ds4_gpu_tensor_free(view);
    return rc;
}

/* Fused Q8_0 matmul + top-1 over the output rows (single token).  Writes
 * values[0] and selected[0] = argmax + index_offset (ties: lower index). */
int ds4_gpu_matmul_q8_0_top1_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *values,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint32_t index_offset) {
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (blocks == 0 || out_dim > UINT64_MAX / (blocks * 34u)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (!selected || !values || !x || !model_map) return 0;
    if (in_dim == 0 || out_dim == 0) return 0;
    if (weight_offset > model_size ||
        weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < in_dim * sizeof(float) ||
        selected->bytes < sizeof(uint32_t) ||
        values->bytes < sizeof(float)) return 0;
    if (!vulkan_model_range_ok(weight_offset, weight_bytes)) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX) return 0;
    struct ds4_vk_params p = {};
    p.in_dim = (uint32_t)in_dim;
    p.out_dim = (uint32_t)out_dim;
    p.rows = 1u;
    p.blocks = (uint32_t)blocks;
    p.index = index_offset;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, weight_offset,
                                    weight_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, values);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, selected);
    return vulkan_dispatch(g_pipes[DS4_PIPE_MATMUL_Q8_0_TOP1], &p, sizeof(p),
                           binds, nb, 1, 1, 1);
}

/* --- matmul variants (F16) -------------------------------------------------- */

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out_a, ds4_gpu_tensor *out_b,
        const void *model_map, uint64_t model_size,
        uint64_t weight_a_offset, uint64_t weight_b_offset,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out_a || !out_b || !x || !model_map) return 0;
    if (!ds4_gpu_matmul_f16_tensor(out_a, model_map, model_size,
                                   weight_a_offset, in_dim, out_dim, x,
                                   n_tok)) {
        return 0;
    }
    return ds4_gpu_matmul_f16_tensor(out_b, model_map, model_size,
                                     weight_b_offset, in_dim, out_dim, x,
                                     n_tok);
}

/* Optional decode fusion (Metal parity): the paired F16 compressor
 * projection and the recurrent compressor-state store run as one dispatch.
 * Returns 1 when the fused kernel ran, 0 when the caller should use the
 * separate matmul_f16_pair_tensor + compressor_store path, -1 on error. */
int ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        ds4_gpu_tensor *out_kv, ds4_gpu_tensor *out_score,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        const void *model_map, uint64_t model_size,
        uint64_t weight_kv_offset, uint64_t weight_score_offset,
        uint64_t ape_offset, uint32_t ape_type, uint64_t in_dim,
        uint32_t width, const ds4_gpu_tensor *x, uint32_t ratio,
        uint32_t pos) {
    if (!out_kv || !out_score || !state_kv || !state_score || !model_map ||
        !x || in_dim == 0 || width == 0 || ratio == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return -1;
    }
    if (in_dim != 4096u ||
        (width != 256u && width != 512u && width != 1024u) ||
        (ratio != 4u && ratio != 128u)) {
        return 0;
    }
    const uint32_t state_rows = ratio == 4u ? 2u * ratio : ratio;
    const uint64_t weight_bytes = in_dim * width * sizeof(uint16_t);
    const uint64_t out_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (weight_kv_offset > model_size ||
        weight_bytes > model_size - weight_kv_offset ||
        weight_score_offset > model_size ||
        weight_bytes > model_size - weight_score_offset ||
        ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out_kv->bytes < out_bytes || out_score->bytes < out_bytes ||
        state_kv->bytes < state_bytes ||
        state_score->bytes < state_bytes) {
        return -1;
    }
    if (in_dim > UINT32_MAX || width > UINT32_MAX) return -1;
    if (!vulkan_model_range_ok(weight_kv_offset, weight_bytes) ||
        !vulkan_model_range_ok(weight_score_offset, weight_bytes) ||
        !vulkan_model_range_ok(ape_offset, ape_bytes)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.in_dim = (uint32_t)in_dim;
    p.out_dim = width;
    p.rows = 1;
    p.ratio = ratio;
    p.pos0 = pos;
    p.index = ape_type;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_B, weight_kv_offset,
                                    weight_bytes);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_C, weight_score_offset,
                                    weight_bytes);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, ape_offset, ape_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out_kv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, out_score);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, state_kv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT4, state_score);
    if (!vulkan_dispatch(g_pipes[DS4_PIPE_MATMUL_F16_PAIR_COMPRESSOR_STORE],
                         &p, sizeof(p), binds, nb, width, 1, 1)) {
        return -1;
    }
    return 1;
}

/* Exact multi-row form of the 4096x256 F16 router projection. */
int ds4_gpu_matmul_f16_router_rows_exact_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, const ds4_gpu_tensor *x, uint32_t n_rows) {
    if (!out || !x || !model_map || n_rows == 0) return 0;
    return ds4_gpu_matmul_f16_tensor(out, model_map, model_size,
                                     weight_offset, 4096u, 256u, x,
                                     (uint64_t)n_rows);
}

/* Batch path: fold an input RMS normalization into the F16 projection.
 * Computed as rms_norm_plain into a scratch f32 buffer, then the F16 matmul
 * (the fold only changes the activation values, not the projection math). */
int ds4_gpu_matmul_f16_rms_fold_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok, float norm_eps) {
    if (!out || !x || !model_map) return 0;
    if (in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size ||
        out_dim > UINT64_MAX / in_dim) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    ds4_gpu_tensor *norm = vulkan_scratch_a(
            n_tok * in_dim * sizeof(float));
    if (!norm) return 0;
    if (!ds4_gpu_rms_norm_plain_rows_tensor(norm, x, (uint32_t)in_dim,
                                            (uint32_t)n_tok, norm_eps)) {
        return 0;
    }
    return ds4_gpu_matmul_f16_tensor(out, model_map, model_size,
                                     weight_offset, in_dim, out_dim, norm,
                                     n_tok);
}

int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t full_in_dim,
        uint64_t k_off, uint64_t k_cnt, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t x_elem_off) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_matmul_q8_0_kslice_tensor(
            out, model_map, model_size, weight_offset, full_in_dim, k_off,
            k_cnt, out_dim, x, x_elem_off);
}

int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t in_dim,
        uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return ds4_gpu_matmul_quant_tensor(out, model_map, model_size,
                                       weight_offset, weight_type, in_dim,
                                       out_dim, x, n_tok);
}

int ds4_gpu_matmul_quant_rows_scalar_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t in_dim,
        uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return ds4_gpu_matmul_quant_tensor(out, model_map, model_size,
                                       weight_offset, weight_type, in_dim,
                                       out_dim, x, n_tok);
}

/* --- embeddings ------------------------------------------------------------ */

static int vulkan_embed_common(VkPipeline pipe, ds4_gpu_tensor *out,
                               const ds4_gpu_tensor *tokens,
                               const void *model_map, uint64_t model_size,
                               uint64_t weight_offset, uint32_t n_vocab,
                               uint32_t n_tokens, uint32_t n_embd,
                               uint32_t token) {
    if (!out || !model_map || n_embd == 0 || (n_embd & 31u) != 0u) return 0;
    if (token >= n_vocab) return 0;
    const uint64_t row_bytes = ((uint64_t)n_embd / 32u) * 34u;
    if (weight_offset > model_size ||
        (uint64_t)n_vocab * row_bytes > model_size - weight_offset) return 0;
    if (out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float)) return 0;
    if (tokens && tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t)) {
        return 0;
    }
    if (!vulkan_model_range_ok(weight_offset, (uint64_t)n_vocab * row_bytes)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = n_embd;
    p.rows = n_tokens;
    p.blocks = (n_embd + 31u) / 32u;
    p.index = token;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    if (tokens) binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, tokens);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, weight_offset,
                                    (uint64_t)n_vocab * row_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    const uint64_t total = (uint64_t)n_tokens * n_embd;
    const uint32_t groups = (uint32_t)((total + 255u) / 256u);
    return vulkan_dispatch(pipe, &p, sizeof(p), binds, nb, groups, 1, 1);
}

int ds4_gpu_embed_token_q8_0_tensor(ds4_gpu_tensor *out,
                                    const void *model_map,
                                    uint64_t model_size,
                                    uint64_t weight_offset,
                                    uint32_t n_vocab, uint32_t token,
                                    uint32_t n_embd) {
    return vulkan_embed_common(g_pipes[DS4_PIPE_EMBED_TOKEN_Q8_0], out, NULL,
                               model_map, model_size, weight_offset, n_vocab,
                               1, n_embd, token);
}

int ds4_gpu_embed_tokens_q8_0_tensor(ds4_gpu_tensor *out,
                                     const ds4_gpu_tensor *tokens,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t weight_offset,
                                     uint32_t n_vocab, uint32_t n_tokens,
                                     uint32_t n_embd) {
    return vulkan_embed_common(g_pipes[DS4_PIPE_EMBED_TOKENS_Q8_0], out,
                               tokens, model_map, model_size, weight_offset,
                               n_vocab, n_tokens, n_embd, 0);
}

int ds4_gpu_embed_token_quant_tensor(ds4_gpu_tensor *out,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t weight_offset,
                                     uint32_t weight_type, uint32_t n_vocab,
                                     uint32_t token, uint32_t n_embd) {
    if (weight_type != 8u) return 0;   /* DS4_TENSOR_Q8_0 */
    return ds4_gpu_embed_token_q8_0_tensor(out, model_map, model_size,
                                           weight_offset, n_vocab, token,
                                           n_embd);
}

int ds4_gpu_embed_tokens_quant_tensor(ds4_gpu_tensor *out,
                                      const ds4_gpu_tensor *tokens,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t weight_offset,
                                      uint32_t weight_type, uint32_t n_vocab,
                                      uint32_t n_tokens, uint32_t n_embd) {
    if (weight_type != 8u) return 0;   /* DS4_TENSOR_Q8_0 */
    return ds4_gpu_embed_tokens_q8_0_tensor(out, tokens, model_map, model_size,
                                            weight_offset, n_vocab, n_tokens,
                                            n_embd);
}

/* --- HC embeddings (f16 token_embd broadcast to n_hc copies) --------------- */

static int vulkan_embed_hc_common(VkPipeline pipe, ds4_gpu_tensor *out,
                                  const ds4_gpu_tensor *tokens,
                                  const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint32_t n_vocab,
                                  uint32_t n_tokens, uint32_t n_embd,
                                  uint32_t n_hc, uint32_t token) {
    if (!out || !model_map || n_embd == 0 || n_hc == 0) return 0;
    if (token >= n_vocab) return 0;
    const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        return 0;
    }
    if (out->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float)) {
        return 0;
    }
    if (tokens && tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t)) {
        return 0;
    }
    if (!vulkan_model_range_ok(weight_offset, weight_bytes)) return 0;
    struct ds4_vk_params p = {};
    p.n = n_embd;
    p.rows = n_tokens;
    p.aux = n_hc;
    p.blocks = n_vocab;
    p.index = token;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    if (tokens) binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, tokens);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, weight_offset,
                                    weight_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    const uint64_t total = (uint64_t)n_tokens * n_hc * n_embd;
    const uint32_t groups = (uint32_t)((total + 255u) / 256u);
    return vulkan_dispatch(pipe, &p, sizeof(p), binds, nb, groups, 1, 1);
}

int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc,
                                  const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint32_t n_vocab,
                                  uint32_t token, uint32_t n_embd,
                                  uint32_t n_hc) {
    return vulkan_embed_hc_common(g_pipes[DS4_PIPE_EMBED_TOKEN_HC], out_hc,
                                  NULL, model_map, model_size, weight_offset,
                                  n_vocab, 1, n_embd, n_hc, token);
}

int ds4_gpu_embed_tokens_hc_tensor(ds4_gpu_tensor       *out_hc,
                                   const ds4_gpu_tensor *tokens,
                                   const void *model_map, uint64_t model_size,
                                   uint64_t weight_offset, uint32_t n_vocab,
                                   uint32_t n_tokens, uint32_t n_embd,
                                   uint32_t n_hc) {
    return vulkan_embed_hc_common(g_pipes[DS4_PIPE_EMBED_TOKENS_HC], out_hc,
                                  tokens, model_map, model_size, weight_offset,
                                  n_vocab, n_tokens, n_embd, n_hc, 0);
}

/* --- Fase 3: FP8 KV quantize / raw store ---------------------------------- */

/* In-place FP8 KV quantize: only the nope part (head_dim - n_rot) is
 * quantized in 64-element blocks (E4M3FN), the rope tail stays f32. */
int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok,
                                        uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_rot > head_dim ||
        x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_tok;
    p.n_rot = n_rot;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, x);
    return vulkan_dispatch(g_pipes[DS4_PIPE_FP8_KV_QUANTIZE], &p, sizeof(p),
                           binds, nb, n_tok, 1, 1);
}

/* Store KV rows into the raw attention cache with an f16 round trip:
 * raw[(pos0 + t) % raw_cap][d] = half2float(float2half(kv[t][d])). */
static int vulkan_store_raw_kv_common(ds4_gpu_tensor *raw_cache,
                                      const ds4_gpu_tensor *kv,
                                      uint32_t raw_cap, uint32_t pos0,
                                      uint32_t n_tokens,
                                      uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_tokens;
    p.pos0 = pos0;
    p.aux = raw_cap;
    if (getenv("DS4_VULKAN_DEBUG_KV") != NULL) {
        fprintf(stderr,
                "ds4: Vulkan debug: store_raw_kv raw_cap=%u pos0=%u n_tokens=%u head_dim=%u\n",
                raw_cap, pos0, n_tokens, head_dim);
    }
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, kv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, raw_cache);
    const uint64_t total = (uint64_t)n_tokens * head_dim;
    const uint32_t groups = (uint32_t)((total + 255u) / 256u);
    return vulkan_dispatch(g_pipes[DS4_PIPE_STORE_RAW_KV], &p, sizeof(p),
                           binds, nb, groups, 1, 1);
}

int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache,
                                const ds4_gpu_tensor *kv, uint32_t raw_cap,
                                uint32_t row, uint32_t head_dim) {
    return vulkan_store_raw_kv_common(raw_cache, kv, raw_cap, row, 1,
                                      head_dim);
}

int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache,
                                      const ds4_gpu_tensor *kv,
                                      uint32_t raw_cap, uint32_t pos0,
                                      uint32_t n_tokens, uint32_t head_dim) {
    return vulkan_store_raw_kv_common(raw_cache, kv, raw_cap, pos0, n_tokens,
                                      head_dim);
}

/* Fused FP8 KV quantize + raw store for one row (the decode finalizer).
 * The nope part is E4M3FN-quantized in place; the full row is written to the
 * raw cache with an f16 round trip. */
static int vulkan_kv_fp8_store_raw_one(ds4_gpu_tensor *kv,
                                       ds4_gpu_tensor *raw_cache,
                                       uint32_t raw_cap, uint32_t raw_row,
                                       uint32_t head_dim, uint32_t n_rot) {
    if (!kv || !raw_cache || raw_cap == 0 || n_rot > head_dim ||
        kv->bytes < (uint64_t)head_dim * sizeof(float) ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.n_rot = n_rot;
    p.index = raw_row % raw_cap;
    p.aux = raw_cap;
    if (getenv("DS4_VULKAN_DEBUG_KV") != NULL) {
        fprintf(stderr,
                "ds4: Vulkan debug: kv_fp8_store_raw raw_cap=%u raw_row=%u head_dim=%u n_rot=%u\n",
                raw_cap, raw_row, head_dim, n_rot);
    }
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, kv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, raw_cache);
    return vulkan_dispatch(g_pipes[DS4_PIPE_KV_FP8_STORE_RAW], &p, sizeof(p),
                           binds, nb, 1, 1, 1);
}

int ds4_gpu_kv_fp8_store_raw_tensor(ds4_gpu_tensor *kv,
                                    ds4_gpu_tensor *raw_cache,
                                    uint32_t raw_cap, uint32_t raw_row,
                                    uint32_t head_dim, uint32_t n_rot) {
    return vulkan_kv_fp8_store_raw_one(kv, raw_cache, raw_cap, raw_row,
                                       head_dim, n_rot);
}

/* Multi-session form: kv rows are contiguous, each row is written to its own
 * session-private raw cache (different VkBuffer per row), so this is one
 * one-row dispatch per session. */
int ds4_gpu_kv_fp8_store_raw_decode_rows_tensor(
        ds4_gpu_tensor *kv, ds4_gpu_tensor *const *raw_caches,
        const uint32_t *raw_caps, const uint32_t *raw_rows, uint32_t n_rows,
        uint32_t head_dim, uint32_t n_rot) {
    if (!kv || !raw_caches || !raw_caps || !raw_rows || n_rows == 0 ||
        n_rows > DS4_GPU_ATTENTION_DECODE_BATCH_MAX ||
        kv->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    for (uint32_t i = 0; i < n_rows; i++) {
        const ds4_gpu_tensor *raw = raw_caches[i];
        if (!raw || raw_caps[i] == 0 || raw_rows[i] >= raw_caps[i] ||
            raw->bytes < (uint64_t)raw_caps[i] * head_dim * sizeof(float)) {
            return 0;
        }
        ds4_gpu_tensor *kview =
            ds4_gpu_tensor_view(kv, (uint64_t)i * head_dim * sizeof(float),
                                (uint64_t)head_dim * sizeof(float));
        if (!kview) return 0;
        const int rc = vulkan_kv_fp8_store_raw_one(
                kview, (ds4_gpu_tensor *)raw, raw_caps[i], raw_rows[i],
                head_dim, n_rot);
        ds4_gpu_tensor_free(kview);
        if (!rc) return 0;
    }
    return 1;
}

/* --- Fase 3: attention decode / prefill ---------------------------------- */

/* Shared attention decode launcher.  Computes per (token, head) block the
 * attention over a raw ring-buffer span plus optional compressed rows, using
 * a global score scratch (binding OUT3) so large contexts never exceed the
 * groupshared limit.  raw_cap may be 0 when raw_kv is a plain (non-ring)
 * buffer (prefill), where n_raw rows are read from rows [raw_start, ...). */
static int vulkan_attn_decode_common(VkPipeline pipe, ds4_gpu_tensor *heads,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t sinks_offset,
                                     const ds4_gpu_tensor *q,
                                     const ds4_gpu_tensor *raw_kv,
                                     uint32_t n_raw, uint32_t raw_cap,
                                     uint32_t raw_start,
                                     const ds4_gpu_tensor *comp_kv,
                                     uint32_t n_comp,
                                     const ds4_gpu_tensor *comp_mask,
                                     uint32_t use_mask, uint32_t n_tokens,
                                     uint32_t pos0, uint32_t window,
                                     uint32_t ratio, uint32_t n_head,
                                     uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_head == 0 ||
        head_dim == 0 || n_tokens == 0 || n_raw == 0 ||
        (raw_cap != 0 && raw_cap < n_raw) ||
        (raw_cap != 0 && raw_start >= raw_cap) ||
        (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)(raw_cap ? raw_cap : n_raw) * head_dim *
                            sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim *
                                        sizeof(float)) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp *
                                             sizeof(float))) {
        return 0;
    }
    if (!vulkan_model_range_ok(sinks_offset,
                               (uint64_t)n_head * sizeof(float))) {
        return 0;
    }
    /* Score scratch: one slab of (max_raw + n_comp) floats per (token, head).
     * raw rows are capped at 256 like the CUDA fallback kernel. */
    const uint32_t raw_max = n_raw < 256u ? n_raw : 256u;
    const uint32_t slab = raw_max + n_comp;
    if (slab == 0 || slab > 0x4000000u) return 0;
    ds4_gpu_tensor *scratch = vulkan_scratch_a(
            (uint64_t)n_tokens * n_head * slab * sizeof(float));
    if (!scratch) return 0;

    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_head;
    p.in_dim = n_tokens;
    p.out_dim = n_raw;
    p.n_rot = raw_cap;
    p.pos0 = pos0;
    p.index = raw_start;
    p.blocks = n_comp;
    p.aux = window;
    p.ratio = ratio;
    p.flags = use_mask;
    p.rsvd2 = slab;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, q);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, raw_kv);
    if (n_comp) binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, comp_kv);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, sinks_offset,
                                    (uint64_t)n_head * sizeof(float));
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, heads);
    if (use_mask) {
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, comp_mask);
    }
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, scratch);
    return vulkan_dispatch(pipe, &p, sizeof(p), binds, nb, n_tokens, n_head,
                           1);
}

/* Single-token decode (heads).  raw_kv is a ring buffer of raw_cap rows,
 * the current row's windowed span starts at raw_start. */
int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, uint32_t n_raw, uint32_t raw_cap,
        uint32_t raw_start, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, uint32_t n_comp,
        const ds4_gpu_tensor *comp_mask, uint32_t use_mask, uint32_t n_head,
        uint32_t head_dim) {
    if (comp_kv_f16) return 0;
    return vulkan_attn_decode_common(g_pipes[DS4_PIPE_ATTN_DECODE], heads,
                                     model_map, model_size, sinks_offset, q,
                                     raw_kv, n_raw, raw_cap, raw_start,
                                     comp_kv, n_comp, comp_mask, use_mask, 1u,
                                     0u, 0u, 0u, n_head, head_dim);
}

int ds4_gpu_attention_decode_heads_rope_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, uint32_t n_raw, uint32_t raw_cap,
        uint32_t raw_start, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, uint32_t n_comp,
        const ds4_gpu_tensor *comp_mask, uint32_t use_mask, uint32_t n_head,
        uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow,
        int *fused_inv_rope) {
    (void)n_rot; (void)pos0; (void)n_ctx_orig; (void)freq_base;
    (void)freq_scale; (void)ext_factor; (void)attn_factor; (void)beta_fast;
    (void)beta_slow;
    if (fused_inv_rope) *fused_inv_rope = 0;
    /* The standalone RoPE kernel handles rotation before decode; this entry
     * is the plain decode when the fused inverse-rope path is not in use. */
    return ds4_gpu_attention_decode_heads_tensor(
            heads, model_map, model_size, sinks_offset, q, raw_kv, n_raw,
            raw_cap, raw_start, comp_kv, comp_kv_f16, n_comp, comp_mask,
            use_mask, n_head, head_dim);
}

/* Batched decode over contiguous q/head rows and a shared raw ring buffer.
 * raw_cap != 0 selects the ring form (raw_start..), otherwise raw_kv holds
 * n_raw rows read directly from index raw_start. */
int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t window, uint32_t n_head, uint32_t head_dim) {
    return vulkan_attn_decode_common(g_pipes[DS4_PIPE_ATTN_DECODE], heads,
                                     model_map, model_size, sinks_offset, q,
                                     raw_kv, n_raw, raw_cap, raw_start, NULL,
                                     0, NULL, 0, n_tokens, pos0, window, 1u,
                                     n_head, head_dim);
}

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, const ds4_gpu_tensor *comp_mask,
        uint32_t use_comp_mask, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head,
        uint32_t head_dim) {
    if (comp_kv_f16) return 0;
    return vulkan_attn_decode_common(g_pipes[DS4_PIPE_ATTN_DECODE], heads,
                                     model_map, model_size, sinks_offset, q,
                                     raw_kv, n_raw, raw_cap, raw_start,
                                     comp_kv, n_comp, comp_mask, use_comp_mask,
                                     n_tokens, pos0, window, ratio, n_head,
                                     head_dim);
}

/* Indexed mixed decode launcher: per (token, head) block over the raw ring
 * span plus the compressed rows selected by the indexer topk (see
 * attn_indexed_decode in attention.hlsl).  Bindings: q (A), raw_kv (B),
 * comp_kv (C), sinks (W), heads (OUT), scores scratch (OUT3), topk (OUT4). */
static int vulkan_attn_decode_indexed_common(
        VkPipeline pipe, ds4_gpu_tensor *heads, const void *model_map,
        uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, uint32_t n_raw, uint32_t raw_cap,
        uint32_t raw_start, const ds4_gpu_tensor *comp_kv, uint32_t n_comp,
        const ds4_gpu_tensor *topk, uint32_t top_k, uint32_t n_tokens,
        uint32_t pos0, uint32_t window, uint32_t ratio, uint32_t n_head,
        uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !comp_kv || !topk || !model_map ||
        n_head == 0 || head_dim == 0 || n_tokens == 0 || n_raw == 0 ||
        (raw_cap != 0 && (raw_cap < n_raw || raw_start >= raw_cap)) ||
        n_comp == 0 || top_k == 0 || top_k > 512u ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)(raw_cap ? raw_cap : n_raw) * head_dim *
                            sizeof(float) ||
        comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    if (!vulkan_model_range_ok(sinks_offset,
                               (uint64_t)n_head * sizeof(float))) {
        return 0;
    }
    /* Score scratch: (raw_max + comp_max) floats per (token, head).  comp_max
     * is capped at 512 like the CUDA fallback kernel's shared comp_rows. */
    const uint32_t raw_max = n_raw < 256u ? n_raw : 256u;
    const uint32_t comp_max = top_k < 512u ? top_k : 512u;
    const uint32_t slab = raw_max + comp_max;
    if (slab == 0 || slab > 0x4000000u) return 0;
    ds4_gpu_tensor *scratch = vulkan_scratch_a(
            (uint64_t)n_tokens * n_head * slab * sizeof(float));
    if (!scratch) return 0;

    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_head;
    p.in_dim = n_tokens;
    p.out_dim = n_raw;
    p.n_rot = raw_cap;
    p.pos0 = pos0;
    p.index = raw_start;
    p.blocks = n_comp;
    p.aux = window;
    p.ratio = ratio;
    p.flags = top_k;
    p.rsvd2 = slab;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, q);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, raw_kv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, comp_kv);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, sinks_offset,
                                    (uint64_t)n_head * sizeof(float));
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, heads);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, scratch);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT4, topk);
    return vulkan_dispatch(pipe, &p, sizeof(p), binds, nb, n_tokens, n_head,
                           1);
}

/* Indexed mixed attention over raw ring + topk-selected compressed rows.
 * Used by the decode and batched prefill paths whenever the indexer produces
 * a compressed-row selection (comp_kv is FP32 on the Vulkan backend). */
int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, const ds4_gpu_tensor *topk, uint32_t n_tokens,
        uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t top_k, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    if (comp_kv_f16) return 0;
    return vulkan_attn_decode_indexed_common(
            g_pipes[DS4_PIPE_ATTN_DECODE_INDEXED], heads, model_map,
            model_size, sinks_offset, q, raw_kv, n_raw, raw_cap, raw_start,
            comp_kv, n_comp, topk, top_k, n_tokens, pos0, window, ratio,
            n_head, head_dim);
}

/* Shared prefill launcher: causal attention over raw rows (read directly from
 * raw_kv, not a ring) plus optional compressed rows with an optional per
 * token comp mask. */
static int vulkan_attn_prefill_common(ds4_gpu_tensor *heads,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t sinks_offset,
                                      const ds4_gpu_tensor *q,
                                      const ds4_gpu_tensor *raw_kv,
                                      const ds4_gpu_tensor *comp_kv,
                                      uint32_t n_comp,
                                      const ds4_gpu_tensor *comp_mask,
                                      uint32_t use_comp_mask,
                                      uint32_t n_tokens, uint32_t window,
                                      uint32_t ratio, uint32_t n_head,
                                      uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        ratio == 0 || n_head == 0 || head_dim == 0 ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim *
                                        sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp *
                                                 sizeof(float))) {
        return 0;
    }
    if (!vulkan_model_range_ok(sinks_offset,
                               (uint64_t)n_head * sizeof(float))) {
        return 0;
    }
    const uint32_t raw_max = window != 0 ? window : n_tokens;
    const uint32_t slab = raw_max + n_comp;
    if (slab == 0 || slab > 0x4000000u) return 0;
    ds4_gpu_tensor *scratch = vulkan_scratch_a(
            (uint64_t)n_tokens * n_head * slab * sizeof(float));
    if (!scratch) return 0;

    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_head;
    p.in_dim = n_tokens;
    p.out_dim = n_comp;
    p.index = window;
    p.aux = ratio;
    p.flags = use_comp_mask;
    p.rsvd2 = slab;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, q);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, raw_kv);
    if (n_comp) binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, comp_kv);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, sinks_offset,
                                    (uint64_t)n_head * sizeof(float));
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, heads);
    if (use_comp_mask) {
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, comp_mask);
    }
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, scratch);
    return vulkan_dispatch(g_pipes[DS4_PIPE_ATTN_PREFILL], &p, sizeof(p),
                           binds, nb, n_tokens, n_head, 1);
}

int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window,
        uint32_t n_head, uint32_t head_dim) {
    return vulkan_attn_prefill_common(heads, model_map, model_size,
                                      sinks_offset, q, raw_kv, NULL, 0, NULL,
                                      0, n_tokens, window, 1u, n_head,
                                      head_dim);
}

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, uint32_t n_tokens, uint32_t n_comp,
        uint32_t window, uint32_t ratio, uint32_t n_head,
        uint32_t head_dim) {
    if (comp_kv_f16) return 0;
    return vulkan_attn_prefill_common(heads, model_map, model_size,
                                      sinks_offset, q, raw_kv, comp_kv,
                                      n_comp, NULL, 0, n_tokens, window,
                                      ratio, n_head, head_dim);
}

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, const ds4_gpu_tensor *comp_mask,
        uint32_t n_tokens, uint32_t n_comp, uint32_t window,
        uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    if (comp_kv_f16) return 0;
    return vulkan_attn_prefill_common(heads, model_map, model_size,
                                      sinks_offset, q, raw_kv, comp_kv,
                                      n_comp, comp_mask, 1, n_tokens, window,
                                      ratio, n_head, head_dim);
}

/* Multi-session decode over private per-session KV caches.  Each row's raw
 * cache is a separate VkBuffer (the row table carries the tensor handle, as
 * the engine stores tensor->ptr), so this is one decode per row. */
int ds4_gpu_attention_decode_rows_rope_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_attention_decode_row *rows, uint32_t n_rows,
        uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow) {
    (void)n_rot; (void)n_ctx_orig; (void)freq_base; (void)freq_scale;
    (void)ext_factor; (void)attn_factor; (void)beta_fast; (void)beta_slow;
    if (!heads || !q || !rows || !model_map || n_rows == 0 ||
        n_rows > DS4_GPU_ATTENTION_DECODE_BATCH_MAX || n_head == 0 ||
        head_dim == 0 || sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_rows * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_rows * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    if (!vulkan_model_range_ok(sinks_offset,
                               (uint64_t)n_head * sizeof(float))) {
        return 0;
    }
    for (uint32_t i = 0; i < n_rows; i++) {
        const ds4_gpu_attention_decode_row r = rows[i];
        if (r.raw_kv == 0u || r.n_raw == 0u || r.raw_cap < r.n_raw ||
            r.raw_start >= r.raw_cap) {
            return 0;
        }
        struct ds4_vulkan_tensor *rawh =
            (struct ds4_vulkan_tensor *)(uintptr_t)r.raw_kv;
        if (!rawh || rawh->buffer == VK_NULL_HANDLE) return 0;
        ds4_gpu_tensor raw_t = {};
        raw_t.ptr = rawh;
        raw_t.bytes = rawh->bytes;
        raw_t.owner = 0;
        raw_t.device_id = 0;

        const struct ds4_vulkan_tensor *comph = NULL;
        ds4_gpu_tensor comp_t = {};
        if (r.n_comp != 0u && r.comp_kv != 0u) {
            comph = (struct ds4_vulkan_tensor *)(uintptr_t)r.comp_kv;
            if (!comph || comph->buffer == VK_NULL_HANDLE) return 0;
            comp_t.ptr = (void *)comph;
            comp_t.bytes = comph->bytes;
            comp_t.owner = 0;
            comp_t.device_id = 0;
        }

        ds4_gpu_tensor *qview = ds4_gpu_tensor_view(
                q, (uint64_t)i * n_head * head_dim * sizeof(float),
                (uint64_t)n_head * head_dim * sizeof(float));
        ds4_gpu_tensor *hview = ds4_gpu_tensor_view(
                heads, (uint64_t)i * n_head * head_dim * sizeof(float),
                (uint64_t)n_head * head_dim * sizeof(float));
        if (!qview || !hview) {
            if (qview) ds4_gpu_tensor_free(qview);
            if (hview) ds4_gpu_tensor_free(hview);
            return 0;
        }
        const int rc = vulkan_attn_decode_common(
                g_pipes[DS4_PIPE_ATTN_DECODE], hview, model_map, model_size,
                sinks_offset, qview, &raw_t, r.n_raw, r.raw_cap, r.raw_start,
                r.n_comp ? &comp_t : NULL, r.n_comp, NULL, 0, 1u, r.pos, 0u,
                0u, n_head, head_dim);
        ds4_gpu_tensor_free(qview);
        ds4_gpu_tensor_free(hview);
        if (!rc) return 0;
    }
    return 1;
}

/* --- Fase 3: attention output projections -------------------------------- */

/* Low-rank grouped projection (out_a Q8_0): low[t][g*rank + r] =
 * sum_d heads[t][g*group_dim + d] * out_a[(g*rank+r)][d].  Shared by the
 * full batch form and the group-slice rows_exact form.  The model buffer is
 * bound at out_a's byte offset (group0 picks the slice inside it). */
static int vulkan_attn_output_low_q8(ds4_gpu_tensor *low,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t out_a_offset,
                                     uint64_t group_dim, uint64_t rank,
                                     uint32_t n_groups_total,
                                     uint32_t group0, uint32_t group_cnt,
                                     const ds4_gpu_tensor *heads,
                                     uint32_t n_rows) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 ||
        n_groups_total == 0 || group_cnt == 0 ||
        group0 > n_groups_total || group_cnt > n_groups_total - group0 ||
        n_rows == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)group_cnt * rank;
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    const uint64_t row_a_bytes = blocks_a * 34u;
    const uint64_t a_offset =
        out_a_offset + (uint64_t)group0 * rank * row_a_bytes;
    const uint64_t out_a_bytes = low_dim * row_a_bytes;
    if (a_offset < out_a_offset || a_offset > model_size ||
        out_a_bytes > model_size - a_offset ||
        heads->bytes < (uint64_t)n_rows * n_groups_total * group_dim *
                           sizeof(float) ||
        low->bytes < (uint64_t)n_rows * low_dim * sizeof(float)) {
        return 0;
    }
    if (!vulkan_model_range_ok(a_offset, out_a_bytes)) return 0;
    if (group_dim > UINT32_MAX || rank > UINT32_MAX || low_dim > UINT32_MAX) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = (uint32_t)group_dim;
    p.out_dim = (uint32_t)rank;
    p.blocks = (uint32_t)blocks_a;
    p.index = n_groups_total;
    p.aux = group0;
    p.n_rot = group_cnt;
    p.rows = n_rows;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, heads);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, a_offset, out_a_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, low);
    if (low_dim > UINT32_MAX || n_rows > UINT32_MAX) return 0;
    const int var = vulkan_attn_out_variant((uint32_t)blocks_a, rank);
    if (var == 2) {
        /* v3: one workgroup per AO_RT=8 rows. */
        const uint32_t gx = (uint32_t)((low_dim + 7u) / 8u);
        return vulkan_dispatch(vulkan_pipe_attn_output_low_variant(2), &p,
                               sizeof(p), binds, nb, gx, n_rows, 1);
    }
    return vulkan_dispatch(vulkan_pipe_attn_output_low_variant(var), &p,
                           sizeof(p), binds, nb, (uint32_t)low_dim, n_rows, 1);
}

/* Full attention output: low = heads @ out_a^T (grouped Q8_0), then
 * out = low @ out_b^T (Q8_0 matmul over the concatenated low dim). */
int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, ds4_gpu_tensor *group_tmp,
        ds4_gpu_tensor *low_tmp, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset, uint64_t group_dim,
        uint64_t rank, uint32_t n_groups, uint64_t out_dim,
        const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    (void)group_tmp; (void)low_tmp;
    if (!out || !low || !heads || !model_map || n_groups == 0 ||
        out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    if (!vulkan_attn_output_low_q8(low, model_map, model_size, out_a_offset,
                                   group_dim, rank, n_groups, 0u, n_groups,
                                   heads, n_tokens)) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                      out_b_offset, low_dim, out_dim, low,
                                      n_tokens);
}

int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups, const ds4_gpu_tensor *heads) {
    return vulkan_attn_output_low_q8(low, model_map, model_size, out_a_offset,
                                     group_dim, rank, n_groups, 0u, n_groups,
                                     heads, 1u);
}

int ds4_gpu_attention_output_low_q8_rows_exact_tensor(
        ds4_gpu_tensor       *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups_total, uint32_t group0, uint32_t group_cnt,
        const ds4_gpu_tensor *heads, uint32_t n_rows) {
    return vulkan_attn_output_low_q8(low, model_map, model_size, out_a_offset,
                                     group_dim, rank, n_groups_total, group0,
                                     group_cnt, heads, n_rows);
}

/* Low-rank grouped projection (out_a Q4_K): same math as the Q8_0 helper but
 * the weight rows are Q4_K (144-byte 256-value blocks) and the activation is
 * used raw (no inline Q8 quantization).  group_dim must be a multiple of 256.
 * The model buffer is bound at the group0 slice of out_a and the kernel uses
 * the local group index, so the bound window stays small. */
static int vulkan_attn_output_low_q4k(ds4_gpu_tensor *low,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t out_a_offset,
                                      uint64_t group_dim, uint64_t rank,
                                      uint32_t n_groups_total,
                                      uint32_t group0, uint32_t group_cnt,
                                      const ds4_gpu_tensor *heads,
                                      uint32_t n_rows) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 ||
        n_groups_total == 0 || group_cnt == 0 ||
        group0 > n_groups_total || group_cnt > n_groups_total - group0 ||
        n_rows == 0 || (group_dim % 256u) != 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)group_cnt * rank;
    const uint64_t blocks_a = group_dim / 256u;
    const uint64_t row_a_bytes = blocks_a * 144u;
    const uint64_t a_offset =
        out_a_offset + (uint64_t)group0 * rank * row_a_bytes;
    const uint64_t out_a_bytes = low_dim * row_a_bytes;
    if (a_offset < out_a_offset || a_offset > model_size ||
        out_a_bytes > model_size - a_offset ||
        heads->bytes < (uint64_t)n_rows * n_groups_total * group_dim *
                           sizeof(float) ||
        low->bytes < (uint64_t)n_rows * low_dim * sizeof(float)) {
        return 0;
    }
    if (!vulkan_model_range_ok(a_offset, out_a_bytes)) return 0;
    if (group_dim > UINT32_MAX || rank > UINT32_MAX || low_dim > UINT32_MAX ||
        n_groups_total > UINT32_MAX || group0 > UINT32_MAX ||
        group_cnt > UINT32_MAX) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = (uint32_t)group_dim;
    p.out_dim = (uint32_t)rank;
    p.blocks = (uint32_t)blocks_a;
    p.index = n_groups_total;
    p.aux = group0;
    p.n_rot = group_cnt;
    p.rows = n_rows;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, heads);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, a_offset, out_a_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, low);
    return vulkan_dispatch(g_pipes[DS4_PIPE_ATTN_OUTPUT_LOW_Q4K], &p, sizeof(p),
                           binds, nb, (uint32_t)low_dim, n_rows, 1);
}

int ds4_gpu_attention_output_low_q4_K_slice_tensor(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t group0, uint32_t group_cnt, const ds4_gpu_tensor *heads) {
    /* Single-token group slice (decode).  The engine calls this with the full
     * heads row and group0 == 0 on the non-TP path. */
    return vulkan_attn_output_low_q4k(low, model_map, model_size, out_a_offset,
                                      group_dim, rank, group0 + group_cnt,
                                      group0, group_cnt, heads, 1u);
}

/* Full attention output with out_a Q4_K: low = heads @ out_a^T (grouped
 * Q4_K), then out = low @ out_b^T (typed matmul over the concatenated low
 * dim; out_b_type == Q8_0 today).  Falls back to the engine per-token path
 * when out_b is another quant (return 0). */
int ds4_gpu_attention_output_q4_K_batch_tensor(
        ds4_gpu_tensor       *out, ds4_gpu_tensor *low, ds4_gpu_tensor *group_tmp,
        ds4_gpu_tensor       *low_tmp, const void *model_map, uint64_t model_size,
        uint64_t              out_a_offset, uint64_t out_b_offset,
        uint32_t              out_b_type, uint64_t group_dim, uint64_t rank,
        uint32_t              n_groups, uint64_t out_dim,
        const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    (void)group_tmp; (void)low_tmp;
    if (!out || !low || !heads || !model_map || n_groups == 0 ||
        out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    if (!vulkan_attn_output_low_q4k(low, model_map, model_size, out_a_offset,
                                    group_dim, rank, n_groups, 0u, n_groups,
                                    heads, n_tokens)) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    return ds4_gpu_matmul_quant_tensor(out, model_map, model_size,
                                       out_b_offset, out_b_type, low_dim,
                                       out_dim, low, n_tokens);
}

/* --- Fase 4: MoE routed / shared ---------------------------------------- */

/* Fused HC expand for the split layout (n_hc == 4); defined with the Fase 5
 * kernels below.  Used by shared_down_hc_expand* to fold the routed add. */
static int vulkan_hc_expand4_common(VkPipeline pipe, ds4_gpu_tensor *out_hc,
                                    const ds4_gpu_tensor *block_out,
                                    const ds4_gpu_tensor *block_add,
                                    const ds4_gpu_tensor *block_add2,
                                    const ds4_gpu_tensor *residual_hc,
                                    const ds4_gpu_tensor *split,
                                    uint32_t n_embd, uint32_t n_hc,
                                    uint32_t flags);

/* Shared-expert gate/up + SwiGLU as a composition of the Fase 2 kernels:
 * the Q8_0 pair projection (gate/up sharing x) followed by the clamped
 * SwiGLU.  Matches the CUDA shared path (matmul_q8_0_pair + swiglu). */
static int vulkan_shared_gate_up_swiglu_common(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok, float clamp) {
    if (!gate || !up || !mid || !x || !model_map ||
        in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        out_dim > UINT32_MAX / ((in_dim + 31u) / 32u * 34u)) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight_bytes = out_dim * row_bytes;
    if (gate_offset > model_size || weight_bytes > model_size - gate_offset ||
        up_offset > model_size || weight_bytes > model_size - up_offset) {
        return 0;
    }
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        gate->bytes < n_tok * out_dim * sizeof(float) ||
        up->bytes < n_tok * out_dim * sizeof(float) ||
        mid->bytes < n_tok * out_dim * sizeof(float)) {
        return 0;
    }
    if (!ds4_gpu_matmul_q8_0_pair_tensor(
            gate, up, model_map, model_size, gate_offset, up_offset,
            in_dim, out_dim, out_dim, x, n_tok)) {
        return 0;
    }
    return ds4_gpu_swiglu_tensor(mid, gate, up,
                                 (uint32_t)(out_dim * n_tok), clamp, 1.0f);
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, float clamp) {
    return vulkan_shared_gate_up_swiglu_common(
            gate, up, mid, model_map, model_size, gate_offset, up_offset,
            in_dim, out_dim, x, 1u, clamp);
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok, float clamp) {
    return vulkan_shared_gate_up_swiglu_common(
            gate, up, mid, model_map, model_size, gate_offset, up_offset,
            in_dim, out_dim, x, n_tok, clamp);
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, float clamp) {
    return vulkan_shared_gate_up_swiglu_common(
            gate, up, mid, model_map, model_size, gate_offset, up_offset,
            in_dim, out_dim, x, 1u, clamp);
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok, float clamp) {
    return vulkan_shared_gate_up_swiglu_common(
            gate, up, mid, model_map, model_size, gate_offset, up_offset,
            in_dim, out_dim, x, n_tok, clamp);
}

/* Shared mid SwiGLU: gate/up Q8_0 projections into persistent scratch, then
 * the clamped SwiGLU into mid (scratch_c/d are not touched by the matmul's
 * prequant scratch_a/b path). */
int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor *mid, const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, float clamp) {
    if (!mid || !x || !model_map || in_dim == 0 || out_dim == 0) return 0;
    if (out_dim > UINT32_MAX / sizeof(float)) return 0;
    ds4_gpu_tensor *sg = vulkan_scratch_c((uint64_t)out_dim * sizeof(float));
    ds4_gpu_tensor *su = vulkan_scratch_d((uint64_t)out_dim * sizeof(float));
    if (!sg || !su) return 0;
    return vulkan_shared_gate_up_swiglu_common(
            sg, su, mid, model_map, model_size, gate_offset, up_offset,
            in_dim, out_dim, x, 1u, clamp);
}

/* Multi-GPU decode-exact variant: on the single-device Vulkan backend the
 * expert_split path is unsupported (expert_split > 1), otherwise this is the
 * plain shared mid SwiGLU. */
int ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
        ds4_gpu_tensor *mid, const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, float clamp,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *prequant,
        uint32_t expert_split, bool home_rank) {
    if (selected && (selected->bytes < 6u * sizeof(int32_t) ||
                     expert_split == 0u)) {
        return 0;
    }
    if (selected && expert_split > 1u) return 0;   /* no expert split yet */
    (void)prequant; (void)home_rank;
    return ds4_gpu_shared_mid_swiglu_q8_0_tensor(
            mid, model_map, model_size, gate_offset, up_offset,
            in_dim, out_dim, x, clamp);
}

/* Shared-expert down projection: Q8_0 matmul into shared_out, then the HC
 * expand+add+split.  The HC kernels land in Fase 5; until then these return
 * 0 (engine falls back to the non-fused path) once the matmul is issued. */
int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor *out_hc, ds4_gpu_tensor *shared_out,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *shared_mid, const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !shared_out || !shared_mid || !routed_out ||
        !residual_hc || !split || !model_map) return 0;
    if (!ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                    weight_offset, in_dim, out_dim,
                                    shared_mid, 1)) {
        return 0;
    }
    return ds4_gpu_hc_expand_add_split_tensor(
            out_hc, shared_out, routed_out, residual_hc, split, n_embd, n_hc);
}

int ds4_gpu_shared_down_hc_expand_add_q8_0_tensor(
        ds4_gpu_tensor *out_hc, ds4_gpu_tensor *shared_out,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *shared_mid, const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *routed_add, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !shared_out || !shared_mid || !routed_out || !routed_add ||
        !residual_hc || !split || !model_map) return 0;
    if (!ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                    weight_offset, in_dim, out_dim,
                                    shared_mid, 1)) {
        return 0;
    }
    /* Fused expand with both routed_out (add) and routed_add (add2). */
    return vulkan_hc_expand4_common(
            g_pipes[DS4_PIPE_HC_EXPAND4], out_hc, shared_out, routed_out,
            routed_add, residual_hc, split, n_embd, n_hc, 3u);
}

int ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor(
        ds4_gpu_tensor *out_hc, ds4_gpu_tensor *shared_out,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *shared_mid, const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_packed, const ds4_gpu_tensor *selected,
        uint32_t expert_split, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)shared_out; (void)model_map; (void)model_size;
    (void)weight_offset; (void)in_dim; (void)out_dim; (void)shared_mid;
    (void)home_slots; (void)peer_packed; (void)selected; (void)expert_split;
    (void)residual_hc; (void)split; (void)n_embd; (void)n_hc;
    /* Owned (multi-GPU slot) combine is not supported on a single device. */
    return 0;
}

/* --- Fase 4: router select ----------------------------------------------- */

/* Shared router top-6 selection (one dispatch, single token).  The model
 * bias/hash regions are bound at their offsets; the token travels as a
 * push-constant (hash mode needs it when no tokens tensor is present). */
int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size,
        uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows,
        uint32_t token, uint32_t n_expert, uint32_t n_expert_used,
        float expert_weight_scale, uint32_t n_expert_groups,
        uint32_t n_group_used, bool has_bias, bool hash_mode,
        const ds4_gpu_tensor *logits) {
    if (!selected || !weights || !probs || !logits || !model_map) return 0;
    if (n_expert != 256u || n_expert_used != 6u ||
        n_expert_groups > 1u || n_group_used > 0u ||
        fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    if (logits->bytes < 256u * sizeof(float) ||
        probs->bytes < 256u * sizeof(float) ||
        selected->bytes < 6u * sizeof(int32_t) ||
        weights->bytes < 6u * sizeof(float)) return 0;
    const int use_bias = has_bias && !hash_mode;
    if (use_bias && (bias_offset > model_size ||
                     model_size - bias_offset < 256u * sizeof(float))) {
        return 0;
    }
    if (hash_mode) {
        const uint64_t hash_bytes =
            (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) {
            return 0;
        }
    }
    struct ds4_vk_params p = {};
    p.in_dim = n_expert;
    p.out_dim = n_expert_used;
    p.rows = 1u;
    p.index = hash_rows;
    p.pos0 = token;
    p.weight = expert_weight_scale;
    p.flags = (use_bias ? 1u : 0u) | (hash_mode ? 2u : 0u);
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, logits);
    if (use_bias) {
        binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_B, bias_offset,
                                        256u * sizeof(float));
    }
    if (hash_mode) {
        binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_C, hash_offset,
                                        (uint64_t)hash_rows * 6u *
                                            sizeof(int32_t));
    }
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, probs);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, weights);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, selected);
    return vulkan_dispatch(g_pipes[DS4_PIPE_ROUTER_SELECT], &p, sizeof(p),
                           binds, nb, 1, 1, 1);
}

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size,
        uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows,
        uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias,
        bool hash_mode, const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens, uint32_t n_expert,
        uint32_t n_expert_used, float expert_weight_scale,
        uint32_t n_tokens) {
    if (n_expert != 256u || n_expert_used != 6u ||
        fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map ||
        n_tokens == 0 || n_expert_groups > 1u || n_group_used > 0u ||
        logits->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * 6u * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * 6u * sizeof(float) ||
        tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t)) {
        return 0;
    }
    const int use_bias = has_bias && !hash_mode;
    if (use_bias && (bias_offset > model_size ||
                     model_size - bias_offset < 256u * sizeof(float))) {
        return 0;
    }
    if (hash_mode) {
        const uint64_t hash_bytes =
            (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) {
            return 0;
        }
    }
    struct ds4_vk_params p = {};
    p.in_dim = n_expert;
    p.out_dim = n_expert_used;
    p.rows = n_tokens;
    p.index = hash_rows;
    p.pos0 = 0u;
    p.weight = expert_weight_scale;
    p.flags = (use_bias ? 1u : 0u) | (hash_mode ? 2u : 0u) | 4u;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, logits);
    if (use_bias) {
        binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_B, bias_offset,
                                        256u * sizeof(float));
    }
    if (hash_mode) {
        binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_C, hash_offset,
                                        (uint64_t)hash_rows * 6u *
                                            sizeof(int32_t));
    }
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_W, tokens);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, probs);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, weights);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, selected);
    return vulkan_dispatch(g_pipes[DS4_PIPE_ROUTER_SELECT], &p, sizeof(p),
                           binds, nb, n_tokens, 1, 1);
}

int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected,
                                             uint32_t n_selected) {
    (void)selected; (void)n_selected;
    return 1;
}

/* --- Fase 6 step 3: device-local expert pool ------------------------------ */

static uint64_t vulkan_pool_gate_base(const struct ds4_vk_pool_layer *l) {
    (void)l;
    return 0;
}

static uint64_t vulkan_pool_up_base(const struct ds4_vk_pool_layer *l) {
    return (uint64_t)l->n_slots * l->gate_expert_bytes;
}

static uint64_t vulkan_pool_down_base(const struct ds4_vk_pool_layer *l) {
    return 2ull * (uint64_t)l->n_slots * l->gate_expert_bytes;
}

static uint64_t vulkan_pool_layer_bytes(const struct ds4_vk_pool_layer *l) {
    return vulkan_pool_down_base(l) +
           (uint64_t)l->n_slots * l->down_expert_bytes;
}

/* The per-layer on-device expert->slot table lives in a separate host-visible
 * tensor (l->meta), updated only when the resident set changes (eviction).
 * It replaces the old per-token remap scratch, whose device-local upload
 * staged a submit every layer even on cache hits. */
static uint64_t vulkan_pool_meta_bytes(void) {
    return (uint64_t)DS4_VK_POOL_TABLE_ENTRIES * sizeof(int32_t);
}

/* Derive the per-layer slot count from the effective expert budget and the
 * model's routed-expert layer count.  The engine reports the layer count via
 * ds4_gpu_set_streaming_expert_cache_layer_count; without it, fall back to
 * the observed maximum layer index seen so far (plus one). */
static void vulkan_pool_configure(void) {
    if (g_pool_ready) return;
    uint32_t layers = g_pool_layer_count;
    if (layers == 0) {
        uint32_t max_seen = 0;
        for (uint32_t i = 0; i < DS4_VK_POOL_MAX_LAYERS; i++) {
            if (g_pool_layers[i].active) max_seen = i + 1;
        }
        layers = max_seen > 0 ? max_seen : DS4_VK_POOL_MAX_LAYERS;
    }
    if (layers < 1) layers = 1;
    g_pool_slots_per_layer = DS4_VK_POOL_MIN_SLOTS;
    const uint64_t eff = vulkan_pool_effective_budget_bytes();
    if (eff != 0 && g_pool_per_expert_bytes != 0) {
        const uint64_t per_layer =
            eff / ((uint64_t)layers * g_pool_per_expert_bytes);
        if (per_layer < DS4_VK_POOL_MIN_SLOTS) {
            g_pool_slots_per_layer = DS4_VK_POOL_MIN_SLOTS;
        } else if (per_layer > DS4_VK_POOL_MAX_SLOTS) {
            g_pool_slots_per_layer = DS4_VK_POOL_MAX_SLOTS;
        } else {
            g_pool_slots_per_layer = (uint32_t)per_layer;
        }
    }
    {
        uint64_t cap = 3ull * 1024ull * 1024ull * 1024ull;
        if (eff > cap) cap = eff;
        g_pool_max_bytes_fixed = cap;
    }
    if (getenv("DS4_VULKAN_DEBUG_BINDS") != NULL) {
        fprintf(stderr,
                "ds4: Vulkan debug: pool layers=%u eff_budget=%.2f GiB "
                "per_expert=%.2f MiB slots_per_layer=%u max_bytes=%.2f GiB "
                "device_local=%.2f GiB\n",
                layers,
                (double)eff / 1073741824.0,
                (double)g_pool_per_expert_bytes / 1048576.0,
                g_pool_slots_per_layer,
                (double)vulkan_pool_max_bytes() / 1073741824.0,
                (double)g_device_local_bytes / 1073741824.0);
    }
    g_pool_ready = 1;
}

extern "C" void ds4_vulkan_set_stream_budget(uint32_t experts) {
    g_pool_budget = experts;
    g_pool_ready = 0;
}

extern "C" void ds4_vulkan_set_expert_bytes(uint64_t bytes) {
    g_pool_per_expert_bytes = bytes;
}

extern "C" void ds4_vulkan_set_stream_layer_count(uint32_t layers) {
    g_pool_layer_count = layers;
    g_pool_ready = 0;
}

static struct ds4_vk_pool_layer *vulkan_pool_layer(uint32_t il) {
    return il < DS4_VK_POOL_MAX_LAYERS ? &g_pool_layers[il] : NULL;
}

/* Ensure the layer pool tensor exists with the requested layout, reallocating
 * (preserving resident slot data) when the capacity must grow.  Returns 1 on
 * success.  A reallocation loses the residency map (rare: sizes are uniform
 * on the target model). */
static int vulkan_pool_layer_ensure(struct ds4_vk_pool_layer *l,
                                    uint64_t gate_expert_bytes,
                                    uint64_t down_expert_bytes,
                                    uint32_t sel_count,
                                    uint32_t n_slots_needed,
                                    int allow_realloc) {
    vulkan_pool_configure();
    uint32_t n_slots = g_pool_slots_per_layer;
    if (n_slots_needed > n_slots) {
        if (n_slots_needed > DS4_VK_POOL_MAX_SLOTS) return 0;
        n_slots = n_slots_needed;
    }
    /* Never shrink an active layer: the hotlist seed may have sized this
     * layer's pool above the budget-derived slot count (the first
     * `preload_count` hotlist entries distribute unevenly, up to ~34 per
     * layer), and re-sizing down here would reset residency and force a full
     * re-store on every token. */
    if (l->active && l->n_slots > n_slots) n_slots = l->n_slots;
    if (sel_count < DS4_VK_POOL_MIN_SLOTS) sel_count = DS4_VK_POOL_MIN_SLOTS;
    if (sel_count > DS4_VK_POOL_SEL_CAP) sel_count = DS4_VK_POOL_SEL_CAP;
    if (l->active && l->tensor && l->meta &&
        l->gate_expert_bytes == gate_expert_bytes &&
        l->down_expert_bytes == down_expert_bytes &&
        l->n_slots == n_slots) {
        return 1;
    }

    /* Fase 7 async seed: the worker thread must never reallocate the pool
     * tensor -- the main thread may be binding it for a dispatch (a freed
     * VkBuffer -> GPUVM fault).  When the layer is not already sized, the
     * async seed fails and the caller (overlap finish) retries the store
     * synchronously on the main thread at a clean scope boundary. */
    if (!allow_realloc) return 0;

    const uint64_t old_bytes = l->tensor ? vulkan_pool_layer_bytes(l) : 0;
    const uint64_t new_bytes = (uint64_t)n_slots * 2ull * gate_expert_bytes +
                               (uint64_t)n_slots * down_expert_bytes;
    if (new_bytes > old_bytes &&
        g_pool_total_bytes - old_bytes + new_bytes > vulkan_pool_max_bytes()) {
        if (getenv("DS4_VULKAN_DEBUG_BINDS") != NULL) {
            fprintf(stderr,
                    "ds4: Vulkan debug: pool cap hit layer=%u total=%.2f GiB "
                    "need=%.2f GiB max=%.2f GiB\n",
                    (unsigned)(l - g_pool_layers),
                    (double)g_pool_total_bytes / 1073741824.0,
                    (double)new_bytes / 1073741824.0,
                    (double)vulkan_pool_max_bytes() / 1073741824.0);
        }
        return 0;
    }

    if (l->tensor) {
        vulkan_device_wait();
        ds4_gpu_tensor_free(l->tensor);
        g_pool_total_bytes -= old_bytes;
        l->tensor = NULL;
    }
    for (uint32_t i = 0; i < DS4_VK_POOL_MAX_SLOTS; i++) {
        l->slots[i] = -1;
        l->slot_age[i] = 0;
    }
    for (uint32_t i = 0; i < DS4_VK_POOL_TABLE_ENTRIES; i++) {
        l->table_host[i] = -1;
    }
    l->gate_expert_bytes = gate_expert_bytes;
    l->down_expert_bytes = down_expert_bytes;
    l->n_slots = n_slots;
    l->n_used = 0;
    l->age = 0;
    l->tensor = ds4_gpu_tensor_alloc_device_local(vulkan_pool_layer_bytes(l));
    if (!l->tensor) return 0;
    g_pool_total_bytes += vulkan_pool_layer_bytes(l);
    if (!l->meta) {
        l->meta = ds4_gpu_tensor_alloc(vulkan_pool_meta_bytes());
        if (!l->meta) {
            ds4_gpu_tensor_free(l->tensor);
            l->tensor = NULL;
            g_pool_total_bytes -= vulkan_pool_layer_bytes(l);
            return 0;
        }
        if (!vulkan_upload_to_tensor(l->meta, 0, l->table_host,
                                     vulkan_pool_meta_bytes())) {
            ds4_gpu_tensor_free(l->meta);
            l->meta = NULL;
            ds4_gpu_tensor_free(l->tensor);
            l->tensor = NULL;
            g_pool_total_bytes -= vulkan_pool_layer_bytes(l);
            return 0;
        }
    }
    l->active = 1;
    if (getenv("DS4_VULKAN_DEBUG_BINDS") != NULL && g_vk_bda != NULL) {
        struct ds4_vulkan_tensor *h = vulkan_tensor_handle(l->tensor);
        VkBufferDeviceAddressInfo bdai = {};
        bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bdai.buffer = h->buffer;
        const uint64_t va = g_vk_bda(g_device, &bdai);
        fprintf(stderr, "ds4: Vulkan debug: pool layer %u bytes=%llu va=0x%llx..0x%llx\n",
                (unsigned)(l - g_pool_layers), (unsigned long long)new_bytes,
                (unsigned long long)va,
                (unsigned long long)(va + vulkan_pool_layer_bytes(l)));
    }
    return 1;
}

/* Slot holding expert e; -1 when not resident. */
static int vulkan_pool_slot_for(const struct ds4_vk_pool_layer *l, int32_t e) {
    for (uint32_t i = 0; i < l->n_slots; i++) {
        if (l->slots[i] == e) return (int)i;
    }
    return -1;
}

/* True when expert e is among the ids being seeded (so it must not be
 * evicted while the current selection is being loaded). */
static int vulkan_pool_id_requested(const int32_t *ids, uint32_t n_ids,
                                    int32_t e) {
    for (uint32_t i = 0; i < n_ids; i++) {
        if (ids[i] == e) return 1;
    }
    return 0;
}

/* Recency pin: an expert requested within the last N routed seeds for this
 * layer (N tokens) is protected from eviction while an unpinned victim exists.
 * Default 3, DS4_VULKAN_POOL_PIN_TOKENS=0 disables. */
static uint32_t vulkan_pool_pin_tokens(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("DS4_VULKAN_POOL_PIN_TOKENS");
        if (v && *v) {
            const long x = strtol(v, NULL, 10);
            cached = x > 0 ? (int)x : 0;
        } else {
            cached = 3;
        }
    }
    return (uint32_t)cached;
}

static int vulkan_pool_id_pinned(const struct ds4_vk_pool_layer *l, int32_t e) {
    const uint32_t pin = vulkan_pool_pin_tokens();
    if (!l || !pin || e < 0 || (uint32_t)e >= DS4_VK_POOL_TABLE_ENTRIES) {
        return 0;
    }
    const uint32_t last = l->last_used[(uint32_t)e];
    if (last == 0) return 0;
    return l->seed_epoch - last < pin;
}

/* Copy expert e's gate/up/down from the model map into pool slot `slot`.
 * NOTE: this is the batched-only store; the actual upload happens in
 * vulkan_pool_store_batch below so a layer seed issues ONE staging copy +
 * submit instead of 3 one-shot submits per expert (~1 ms each). */
static int vulkan_pool_slot_mark(struct ds4_vk_pool_layer *l,
                                 uint32_t slot, int32_t e) {
    if (e < 0 || (uint64_t)e >= DS4_VK_POOL_TABLE_ENTRIES) return 0;
    l->slots[slot] = e;
    l->table_host[(uint32_t)e] = (int32_t)slot;
    return 1;
}

/* Reserve a slot for expert e WITHOUT uploading weights (the caller batches
 * the uploads), evicting the least-hot slot (LFU) that is not part of the
 * current selection; ties break on least-recently-used (LRU slot_age),
 * mirroring Metal's prune (lowest route_hotness, then last_used).  Returns
 * the slot index or -1. */
static int vulkan_pool_slot_reserve_only(struct ds4_vk_pool_layer *l,
                                         const int32_t *ids, uint32_t n_ids,
                                         int32_t e) {
    const int found = vulkan_pool_slot_for(l, e);
    if (found >= 0) return found;
    if (l->n_used < l->n_slots) {
        const uint32_t slot = l->n_used++;
        if (!vulkan_pool_slot_mark(l, slot, e)) return -1;
        return (int)slot;
    }
    uint32_t victim = UINT32_MAX;
    int victim_pass = -1;
    /* Pass 0 respects the recency pin (last N tokens); pass 1 falls back to
     * ignoring it when every candidate is pinned. */
    for (int pass = 0; pass < 2 && victim == UINT32_MAX; pass++) {
        const int respect_pin = (pass == 0 && vulkan_pool_pin_tokens() != 0);
        if (vulkan_pool_hotness_off()) {
            /* Legacy LRU-only (bisect DS4_VULKAN_HOTNESS_OFF). */
            uint32_t min_age = UINT32_MAX;
            for (uint32_t i = 0; i < l->n_slots; i++) {
                if (vulkan_pool_id_requested(ids, n_ids, l->slots[i])) continue;
                if (respect_pin &&
                    vulkan_pool_id_pinned(l, l->slots[i])) continue;
                if (l->slot_age[i] < min_age) {
                    min_age = l->slot_age[i];
                    victim = i;
                }
            }
        } else {
            /* LFU victim: lowest route hotness, tiebreak LRU slot_age (Metal
             * prune semantics). */
            uint32_t lowest_hotness = UINT32_MAX;
            uint32_t oldest_age = UINT32_MAX;
            for (uint32_t i = 0; i < l->n_slots; i++) {
                if (vulkan_pool_id_requested(ids, n_ids, l->slots[i])) continue;
                if (respect_pin &&
                    vulkan_pool_id_pinned(l, l->slots[i])) continue;
                const int32_t slot_expert = l->slots[i];
                uint32_t hotness = 0;
                if (slot_expert >= 0 &&
                    (uint32_t)slot_expert < DS4_VK_POOL_TABLE_ENTRIES) {
                    hotness = l->route_hotness[(uint32_t)slot_expert];
                }
                if (hotness < lowest_hotness ||
                    (hotness == lowest_hotness &&
                     l->slot_age[i] < oldest_age)) {
                    lowest_hotness = hotness;
                    oldest_age = l->slot_age[i];
                    victim = i;
                }
            }
        }
        if (victim != UINT32_MAX) victim_pass = pass;
    }
    if (victim == UINT32_MAX) return -1;
    /* A resident expert is evicted to make room: count the churn (thrash
     * signal when the evicted expert is reloaded soon after). */
    g_pool_tel[l - g_pool_layers].evictions.fetch_add(1,
            std::memory_order_relaxed);
    const int32_t old_e = l->slots[victim];
    {
        const uint32_t vage = (old_e >= 0)
            ? l->seed_epoch - l->last_used[(uint32_t)old_e] : 0u;
        vulkan_trace("pool.evict layer=%u victim=%d age=%u pinned=%d new=%d "
                     "pass=%d hot=%u",
                     (unsigned)(l - g_pool_layers), old_e, vage,
                     (old_e >= 0) ? vulkan_pool_id_pinned(l, old_e) : 0,
                     e, victim_pass,
                     (old_e >= 0) ? l->route_hotness[(uint32_t)old_e] : 0u);
    }
    if (old_e >= 0 && (uint32_t)old_e < DS4_VK_POOL_TABLE_ENTRIES) {
        l->table_host[(uint32_t)old_e] = -1;
    }
    if (!vulkan_pool_slot_mark(l, victim, e)) return -1;
    return (int)victim;
}

/* model-part (SPECS_MODEL_PART.md): pick the sparse-mirror source for a
 * hosted expert.  The part file mirrors the model at the same byte offsets, so
 * a hosted expert is read from part_map at the SAME offset.  Returns
 * model_map when no part file covers this expert. */
static const uint8_t *vulkan_expert_src(const ds4_gpu_stream_expert_table *t,
                                        const uint64_t *mask, int32_t e) {
    if (t->part_map && mask && e >= 0) {
        const uint32_t ue = (uint32_t)e;
        if (t->part_mask_words > ue / 64u &&
            ((mask[ue / 64u] >> (ue % 64u)) & UINT64_C(1))) {
            return (const uint8_t *)t->part_map;
        }
    }
    return (const uint8_t *)t->model_map;
}

/* Upload the weights of the given (already reserved) experts into their pool
 * slots in ONE staging copy + ONE submit/wait.  Each expert occupies a
 * contiguous per-expert slab (gate|up|down), so the whole batch is one
 * memcpy into the staging buffer followed by one vkCmdCopyBuffer with one
 * region per slab.  On UMA the pool tensor is host-mapped and the copies are
 * plain memcpys (no staging submit at all).  The caller has already drained. */
static int vulkan_pool_store_batch(struct ds4_vk_pool_layer *l,
                                   const ds4_gpu_stream_expert_table *table,
                                   const int32_t *missing, uint32_t n_missing,
                                   int async) {
    if (n_missing == 0) return 1;
    const uint8_t *src = (const uint8_t *)table->model_map;
    if (!src || !l->tensor || n_missing > DS4_VK_POOL_MAX_SLOTS) return 0;
    const uint64_t per = 2ull * l->gate_expert_bytes + l->down_expert_bytes;
    if (per == 0 || n_missing > UINT64_MAX / per) return 0;
    const uint64_t total = (uint64_t)n_missing * per;

    const bool dbg_time = getenv("DS4_VULKAN_DEBUG_POOL_TIME") != NULL;
    double dbg_copy0 = 0.0, dbg_sub0 = 0.0;
    if (dbg_time) {
        if (!g_dbg_store_time_reg.exchange(1)) atexit(vulkan_store_time_report);
        g_dbg_store_calls.fetch_add(1, std::memory_order_relaxed);
        g_dbg_store_bytes.fetch_add(total, std::memory_order_relaxed);
        dbg_copy0 = vulkan_now_ms();
    }

    struct ds4_vulkan_tensor *ph = vulkan_tensor_handle(l->tensor);
    if (!ph) return 0;
    if (ph->host_map) {
        for (uint32_t i = 0; i < n_missing; i++) {
            const int32_t e = missing[i];
            const int slot = vulkan_pool_slot_for(l, e);
            if (slot < 0 || e < 0) return 0;
            const uint64_t eg = (uint64_t)e;
            const uint8_t *gsrc = vulkan_expert_src(table, table->part_mask_gate, e);
            const uint8_t *usrc = vulkan_expert_src(table, table->part_mask_up, e);
            const uint8_t *dsrc = vulkan_expert_src(table, table->part_mask_down, e);
            memcpy(ph->host_map + ph->offset + vulkan_pool_gate_base(l) +
                       (uint64_t)slot * l->gate_expert_bytes,
                   gsrc + table->gate_offset + eg * l->gate_expert_bytes,
                   (size_t)l->gate_expert_bytes);
            memcpy(ph->host_map + ph->offset + vulkan_pool_up_base(l) +
                       (uint64_t)slot * l->gate_expert_bytes,
                   usrc + table->up_offset + eg * l->gate_expert_bytes,
                   (size_t)l->gate_expert_bytes);
            memcpy(ph->host_map + ph->offset + vulkan_pool_down_base(l) +
                       (uint64_t)slot * l->down_expert_bytes,
                   dsrc + table->down_offset + eg * l->down_expert_bytes,
                   (size_t)l->down_expert_bytes);
        }
        if (dbg_time)
            g_dbg_store_copy_us.fetch_add(
                (uint64_t)((vulkan_now_ms() - dbg_copy0) * 1000.0),
                std::memory_order_relaxed);
        return 1;
    }

    if (g_pool_staging && g_pool_staging->bytes < total) {
        vulkan_device_wait_impl(0);
        ds4_gpu_tensor_free(g_pool_staging);
        g_pool_staging = NULL;
    }
    if (!g_pool_staging) {
        g_pool_staging = ds4_gpu_tensor_alloc(total);
        if (!g_pool_staging) return 0;
    }
    struct ds4_vulkan_tensor *sh = vulkan_tensor_handle(g_pool_staging);
    if (!sh || !sh->host_map) return 0;

    VkBufferCopy regions[3 * DS4_VK_POOL_MAX_SLOTS];
    uint32_t nr = 0;
    uint64_t s_off = 0;
    if (dbg_time) dbg_copy0 = vulkan_now_ms();
    for (uint32_t i = 0; i < n_missing; i++) {
        const int32_t e = missing[i];
        const int slot = vulkan_pool_slot_for(l, e);
        if (slot < 0 || e < 0 || nr + 3 > 3 * DS4_VK_POOL_MAX_SLOTS) return 0;
        const uint64_t eg = (uint64_t)e;
        const uint8_t *gsrc = vulkan_expert_src(table, table->part_mask_gate, e);
        const uint8_t *usrc = vulkan_expert_src(table, table->part_mask_up, e);
        const uint8_t *dsrc = vulkan_expert_src(table, table->part_mask_down, e);
        /* gate */
        memcpy(sh->host_map + sh->offset + s_off,
               gsrc + table->gate_offset + eg * l->gate_expert_bytes,
               (size_t)l->gate_expert_bytes);
        regions[nr].srcOffset = sh->offset + s_off;
        regions[nr].dstOffset = ph->offset + vulkan_pool_gate_base(l) +
                                (uint64_t)slot * l->gate_expert_bytes;
        regions[nr].size = l->gate_expert_bytes;
        nr++;
        s_off += l->gate_expert_bytes;
        /* up */
        memcpy(sh->host_map + sh->offset + s_off,
               usrc + table->up_offset + eg * l->gate_expert_bytes,
               (size_t)l->gate_expert_bytes);
        regions[nr].srcOffset = sh->offset + s_off;
        regions[nr].dstOffset = ph->offset + vulkan_pool_up_base(l) +
                                (uint64_t)slot * l->gate_expert_bytes;
        regions[nr].size = l->gate_expert_bytes;
        nr++;
        s_off += l->gate_expert_bytes;
        /* down */
        memcpy(sh->host_map + sh->offset + s_off,
               dsrc + table->down_offset + eg * l->down_expert_bytes,
               (size_t)l->down_expert_bytes);
        regions[nr].srcOffset = sh->offset + s_off;
        regions[nr].dstOffset = ph->offset + vulkan_pool_down_base(l) +
                                (uint64_t)slot * l->down_expert_bytes;
        regions[nr].size = l->down_expert_bytes;
        nr++;
        s_off += l->down_expert_bytes;
    }
    if (dbg_time) {
        g_dbg_store_copy_us.fetch_add(
            (uint64_t)((vulkan_now_ms() - dbg_copy0) * 1000.0),
            std::memory_order_relaxed);
        dbg_sub0 = vulkan_now_ms();
    }
    if (!vulkan_compute_init()) return 0;
    if (async) {
        const int src_ok = vulkan_worker_copy_submit_multi(sh->buffer,
                                                           ph->buffer, regions,
                                                           nr);
        if (dbg_time)
            g_dbg_store_submit_us.fetch_add(
                (uint64_t)((vulkan_now_ms() - dbg_sub0) * 1000.0),
                std::memory_order_relaxed);
        return src_ok ? 1 : 0;
    }
    VkCommandBuffer cb = vulkan_dispatch_begin();
    if (!cb) return 0;
    vkCmdCopyBuffer(cb, sh->buffer, ph->buffer, nr, regions);
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &mb, 0, NULL, 0, NULL);
    const int sub_ok = vulkan_submit_one_shot();
    if (dbg_time)
        g_dbg_store_submit_us.fetch_add(
            (uint64_t)((vulkan_now_ms() - dbg_sub0) * 1000.0),
            std::memory_order_relaxed);
    return sub_ok;
}

/* Seed the layer pool with the given selection and keep the on-device
 * expert->slot table current.  Writes (pool data and/or the table) happen
 * only when the resident set changes (a requested expert is missing); an
 * all-hit selection writes nothing (no per-token staging round-trip).
 * When `async` is set (Fase 7 worker path) the store runs on the dedicated
 * worker command buffer and returns before the GPU copy finishes: the main
 * thread waits the fence (vulkan_pool_commit_pending) right before the MoE
 * that consumes the pool, and the main queue FIFO orders the copy after the
 * previous token's readers of the same slots.  Returns the pair count, or -1
 * on failure. */
static int vulkan_pool_seed_remap(struct ds4_vk_pool_layer *l,
                                  const ds4_gpu_stream_expert_table *table,
                                  const int32_t *ids, uint32_t n_ids,
                                  int32_t *remap_out, int async,
                                  int phase) {
    if (!l || !table || !ids || !remap_out || n_ids == 0) return -1;
    if (n_ids > DS4_VK_POOL_SEL_CAP || !l->tensor || !l->meta) return -1;

    /* New routed seed (approximately a new token for this layer): advance the
     * recency epoch and stamp the current selection so it stays pinned for the
     * next DS4_VULKAN_POOL_PIN_TOKENS seeds. */
    l->seed_epoch++;
    for (uint32_t i = 0; i < n_ids; i++) {
        const int32_t e = ids[i];
        if (e >= 0 && (uint32_t)e < DS4_VK_POOL_TABLE_ENTRIES) {
            l->last_used[(uint32_t)e] = l->seed_epoch;
        }
    }

    /* Collect the experts that are not resident yet. */
    int32_t missing[DS4_VK_POOL_MAX_SLOTS];
    uint32_t n_missing = 0;
    for (uint32_t i = 0; i < n_ids && n_missing < DS4_VK_POOL_MAX_SLOTS; i++) {
        if (vulkan_pool_slot_for(l, ids[i]) < 0) {
            missing[n_missing++] = ids[i];
        }
    }
    if (getenv("DS4_VULKAN_DEBUG_POOL_HIT") != NULL) {
        fprintf(stderr, "ds4: Vulkan debug pool seed: layer=%u n_ids=%u miss=%u\n",
                (unsigned)(l - g_pool_layers), n_ids, n_missing);
    }
    /* Drain BEFORE writing the pool (a previous token's MoE may still read
     * it); the caller's read of the selection has already drained, so this
     * is normally ~0.  Skipped on the async worker path: the worker copy is
     * submitted to the main queue AFTER the previous MoE scope, so the FIFO
     * order already guarantees the readers drained. */
    if (n_missing != 0 && !async) {
        const double w0 = vulkan_now_ms();
        vulkan_device_wait();
        if (phase >= 0 && phase < DS4_GPU_EXPERT_PHASES) {
            g_pool_tel_wait_us.fetch_add(
                (uint64_t)((vulkan_now_ms() - w0) * 1000.0),
                std::memory_order_relaxed);
        }
    }

    const double p0 = getenv("DS4_VULKAN_DEBUG_POOL_HIT") != NULL ?
                      vulkan_now_ms() : 0.0;
    for (uint32_t i = 0; i < n_ids; i++) {
        const int slot = vulkan_pool_slot_reserve_only(l, ids, n_ids, ids[i]);
        if (slot < 0) return -1;
        remap_out[i] = (int32_t)slot;
        l->slot_age[(uint32_t)slot] = l->age++;
    }

    if (n_missing != 0) {
        if (!vulkan_pool_store_batch(l, table, missing, n_missing, async)) {
            return -1;
        }
        /* The expert->slot table changed; upload it (host-visible memcpy). */
        if (!vulkan_upload_to_tensor(l->meta, 0, l->table_host,
                                     vulkan_pool_meta_bytes())) {
            return -1;
        }
    }
    if (getenv("DS4_VULKAN_DEBUG_POOL_HIT") != NULL) {
        fprintf(stderr, "ds4: Vulkan debug pool seed: store_time=%.2f ms\n",
                vulkan_now_ms() - p0);
    }
    /* Count the seed once the resident set is stable (a failed store above
     * would return -1 and the caller retries, so nothing is double counted).
     * An all-hit seed still counts requests+hits: that is the cross-token
     * reuse the pool exists for. */
    if (phase >= 0 && phase < DS4_GPU_EXPERT_PHASES) {
        struct ds4_vk_pool_tel *t = &g_pool_tel[l - g_pool_layers];
        t->requests[phase].fetch_add(n_ids, std::memory_order_relaxed);
        if (n_missing == 0) {
            t->hits[phase].fetch_add(n_ids, std::memory_order_relaxed);
        } else {
            t->hits[phase].fetch_add(n_ids - n_missing,
                                     std::memory_order_relaxed);
            t->misses[phase].fetch_add(n_missing, std::memory_order_relaxed);
            const uint32_t reloads = vulkan_pool_tel_reloads(l, missing,
                                                             n_missing);
            if (reloads != 0) {
                t->reload_misses[phase].fetch_add(reloads,
                                                  std::memory_order_relaxed);
            }
            const uint64_t per = 2ull * l->gate_expert_bytes +
                                 l->down_expert_bytes;
            t->loaded_bytes[phase].fetch_add((uint64_t)n_missing * per,
                                             std::memory_order_relaxed);
            for (uint32_t i = 0; i < n_missing; i++) {
                const int32_t e = missing[i];
                if (e >= 0 && (uint32_t)e < DS4_VK_POOL_TABLE_ENTRIES) {
                    l->ever_loaded[(uint32_t)e] = 1;
                }
            }
        }
    }
    if (n_missing != 0) {
        vulkan_trace("pool.seed layer=%u req=%u hit=%u miss=%u resident=%u/%u "
                     "loaded_bytes=%llu async=%d",
                     (unsigned)(l - g_pool_layers), n_ids, n_ids - n_missing,
                     n_missing, l->n_used, l->n_slots,
                     (unsigned long long)((uint64_t)n_missing *
                         (2ull * l->gate_expert_bytes + l->down_expert_bytes)),
                     async);
    } else {
        vulkan_trace("pool.seed layer=%u req=%u hit=%u miss=0 resident=%u/%u allhit",
                     (unsigned)(l - g_pool_layers), n_ids, n_ids,
                     l->n_used, l->n_slots);
    }
    return (int)n_ids;
}

/* Wait any pending worker store/readback copy (per-submit fence, no device
 * wait).  The main thread calls this right before a dispatch that reads the
 * pool (routed MoE). */
extern "C" void ds4_vulkan_pool_commit_pending(void) {
    const double w0 = vulkan_now_ms();
    vulkan_worker_copy_wait();
    g_pool_tel_wait_us.fetch_add(
        (uint64_t)((vulkan_now_ms() - w0) * 1000.0),
        std::memory_order_relaxed);
}

/* Fase 8e: true when this layer's routed experts are already resident in the
 * persistent per-device weight cache (a static multi-tier tier).  There the
 * MoE binds the cache directly, so the streaming pool must NOT be activated:
 * its loads read from the model mmap and would hit the disk/page cache,
 * defeating the VRAM residency.  On a dynamic tier the experts are not cached,
 * so this returns false and the pool works as in the single-GPU path. */
static int vulkan_weight_cache_covers_experts(
        const ds4_gpu_stream_expert_table *table) {
    if (!table || table->n_total_expert == 0 ||
        table->gate_expert_bytes == 0 || table->down_expert_bytes == 0) {
        return 0;
    }
    const uint64_t gate_region =
        (uint64_t)table->n_total_expert * table->gate_expert_bytes;
    const uint64_t down_region =
        (uint64_t)table->n_total_expert * table->down_expert_bytes;
    return vulkan_weight_cache_for(table->gate_offset, gate_region) != NULL &&
           vulkan_weight_cache_for(table->up_offset, gate_region) != NULL &&
           vulkan_weight_cache_for(table->down_offset, down_region) != NULL;
}

/* --- Fase 8e: layer -> logical tier ownership -----------------------------
 *
 * The engine registers its placement once (which tier owns each model layer).
 * A pool/seed operation then names its target by LAYER — the tier is resolved
 * from this immutable table, so no operation depends on the mutable "selected
 * device".  Single-tier leaves it empty and falls back to the selected tier. */
static int      g_layer_tier[DS4_VK_POOL_MAX_LAYERS];
static uint32_t g_layer_tier_n = 0;

extern "C" void ds4_vulkan_set_layer_placement(const int *layer_tier,
                                               uint32_t n_layers) {
    if (n_layers > DS4_VK_POOL_MAX_LAYERS) n_layers = DS4_VK_POOL_MAX_LAYERS;
    for (uint32_t i = 0; i < n_layers; i++) g_layer_tier[i] = layer_tier ? layer_tier[i] : 0;
    g_layer_tier_n = layer_tier ? n_layers : 0;
}

static int vulkan_layer_tier(uint32_t layer) {
    if (layer < g_layer_tier_n) return g_layer_tier[layer];
    return g_vk_ctx_active;
}

/* Report — never silently — when an operation that targets `layer` runs while
 * a different tier is selected: that is the multi-tier ownership invariant,
 * and a violation means the operation would hit the wrong device.  Log only
 * for now; promoted to fail-fast once the validation runs are clean. */
static void vulkan_require_layer_tier(uint32_t layer, const char *op) {
    const int want = vulkan_layer_tier(layer);
    if (want < 0 || g_vk_ctx_active < 0) return;
    if (want == g_vk_ctx_active) return;
    fprintf(stderr, DS4_VULKAN_LOG_PREFIX
            "TIER MISMATCH: %s layer=%u owner_tier=%d selected_tier=%d\n",
            op, layer, want, g_vk_ctx_active);
}

extern "C" int ds4_vulkan_stream_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *ids, uint32_t n_selected) {
    if (!table || !ids || n_selected == 0 ||
        table->model_map == NULL ||
        table->gate_expert_bytes == 0 || table->down_expert_bytes == 0) {
        return 0;
    }
    vulkan_require_layer_tier(table->layer, "seed_selected_async");
    if (vulkan_weight_cache_covers_experts(table)) return 1;
    struct ds4_vk_pool_layer *l = vulkan_pool_layer(table->layer);
    if (!l) return 0;
    if (!vulkan_pool_layer_ensure(l, table->gate_expert_bytes,
                                  table->down_expert_bytes, n_selected,
                                  n_selected, 1)) {
        return 0;
    }
    vulkan_pool_hotness_note_selected(l, ids, n_selected);
    int32_t remap[DS4_VK_POOL_SEL_CAP];
    return vulkan_pool_seed_remap(l, table, ids, n_selected, remap, 0,
                                  DS4_GPU_EXPERT_PHASE_DECODE) >= 0;
}

/* Async seed (Fase 7 worker path): same as ds4_vulkan_stream_seed_selected
 * but the store is submitted on the worker command buffer and returns before
 * the GPU copy completes.  The main thread waits the fence via
 * ds4_vulkan_pool_commit_pending() right before the MoE dispatch. */
extern "C" int ds4_vulkan_stream_seed_selected_async(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *ids, uint32_t n_selected) {
    /* Debug/bisect: fall back to the synchronous store (the worker waits the
     * copy inline) to isolate the async copy from the readback/overlap. */
    if (getenv("DS4_VULKAN_ASYNC_STORE_OFF") != NULL) {
        return ds4_vulkan_stream_seed_selected(table, ids, n_selected);
    }
    if (!table || !ids || n_selected == 0 ||
        table->model_map == NULL ||
        table->gate_expert_bytes == 0 || table->down_expert_bytes == 0) {
        return 0;
    }
    if (vulkan_weight_cache_covers_experts(table)) return 1;
    struct ds4_vk_pool_layer *l = vulkan_pool_layer(table->layer);
    if (!l) return 0;
    /* Never reallocate here: the worker runs concurrently with the main's MoE
     * dispatch (which binds l->tensor); if the layer needs to grow, fail and
     * let the overlap finish retry the store synchronously on the main thread. */
    if (!vulkan_pool_layer_ensure(l, table->gate_expert_bytes,
                                  table->down_expert_bytes, n_selected,
                                  n_selected, 0)) {
        return 0;
    }
    vulkan_pool_hotness_note_selected(l, ids, n_selected);
    int32_t remap[DS4_VK_POOL_SEL_CAP];
    return vulkan_pool_seed_remap(l, table, ids, n_selected, remap, 1,
                                  DS4_GPU_EXPERT_PHASE_DECODE) >= 0;
}

extern "C" int ds4_vulkan_stream_seed_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *ids, uint32_t n_tokens, uint32_t n_selected) {
    if (!table || !ids || n_tokens == 0 || n_selected == 0 ||
        (uint64_t)n_tokens * n_selected > DS4_VK_POOL_SEL_CAP ||
        table->model_map == NULL ||
        table->gate_expert_bytes == 0 || table->down_expert_bytes == 0) {
        return 0;
    }
    vulkan_require_layer_tier(table->layer, "seed_batch");
    if (vulkan_weight_cache_covers_experts(table)) return 1;
    const uint32_t n_ids = (uint32_t)((uint64_t)n_tokens * n_selected);
    uint32_t distinct = 0;
    int32_t seen[DS4_VK_POOL_MAX_SLOTS];
    for (uint32_t i = 0; i < n_ids && distinct < DS4_VK_POOL_MAX_SLOTS; i++) {
        int dup = 0;
        for (uint32_t j = 0; j < distinct; j++) {
            if (seen[j] == ids[i]) { dup = 1; break; }
        }
        if (!dup) seen[distinct++] = ids[i];
    }
    if (distinct > DS4_VK_POOL_MAX_SLOTS) return 0;
    struct ds4_vk_pool_layer *l = vulkan_pool_layer(table->layer);
    if (!l) return 0;
    if (!vulkan_pool_layer_ensure(l, table->gate_expert_bytes,
                                  table->down_expert_bytes, n_ids,
                                  distinct, 1)) {
        return 0;
    }
    vulkan_pool_hotness_note_selected(l, seen, distinct);
    int32_t remap[DS4_VK_POOL_SEL_CAP];
    if (vulkan_pool_seed_remap(l, table, ids, n_ids, remap, 0,
                               DS4_GPU_EXPERT_PHASE_PREFILL) < 0) return 0;
    return 1;
}

extern "C" int ds4_vulkan_stream_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *ids, const uint32_t *priorities,
        uint32_t n_experts) {
    if (!table || !ids || n_experts == 0 || table->model_map == NULL ||
        table->gate_expert_bytes == 0 || table->down_expert_bytes == 0) {
        return 0;
    }
    vulkan_require_layer_tier(table->layer, "seed_experts");
    if (vulkan_weight_cache_covers_experts(table)) return 1;
    struct ds4_vk_pool_layer *l = vulkan_pool_layer(table->layer);
    if (!l) return 0;
    /* The hotlist may carry more candidates per layer than the budget-derived
     * pool capacity; seed only the first (highest-priority) slots.  Growing
     * the pool per layer beyond the budget would hit the pool cap and abort
     * the prefill (and shrinking it back in the decode seed would reset the
     * residency). */
    vulkan_pool_configure();
    uint32_t n = n_experts;
    if (g_pool_slots_per_layer != 0 && n > g_pool_slots_per_layer) {
        n = g_pool_slots_per_layer;
    }
    if (n == 0) return 1;
    if (!vulkan_pool_layer_ensure(l, table->gate_expert_bytes,
                                  table->down_expert_bytes,
                                  DS4_VK_POOL_MIN_SLOTS, n, 1)) {
        return 0;
    }
    /* Use the hotlist priorities (Metal ds4_metal.m:16630): a higher priority
     * gives a stronger retention advantage against the LFU eviction.  The
     * routed +1 notes on top of this during decode. */
    vulkan_pool_hotness_seed_priorities(l, ids, priorities, n);
    int32_t remap[DS4_VK_POOL_SEL_CAP];
    if (vulkan_pool_seed_remap(l, table, ids, n, remap, 0,
                               DS4_GPU_EXPERT_PHASE_HOTLIST) < 0) return 0;
    return 1;
}

/* Build the MoE binds against the layer pool when it is active for the
 * requested sizes.  Returns 1 and fills gate_b/up_b/down_b/sel_b/tbl_b; the
 * caller falls back to the model windows otherwise.  sel_b binds the router's
 * selected-expert ids; the on-device expert->slot table (tbl_b) translates
 * them to pool slots inside the shader (set via params.flags bit 0). */
static int vulkan_pool_moe_binds(uint32_t layer_index,
                                 uint64_t gate_expert_bytes,
                                 uint64_t down_expert_bytes,
                                 uint32_t n_tokens, uint32_t n_expert,
                                 const ds4_gpu_tensor *selected,
                                 struct ds4_vk_bind *gate_b,
                                 struct ds4_vk_bind *up_b,
                                 struct ds4_vk_bind *down_b,
                                 struct ds4_vk_bind *sel_b,
                                 struct ds4_vk_bind *tbl_b) {
    if (!g_vulkan_ssd_streaming) return 0;
    struct ds4_vk_pool_layer *l = vulkan_pool_layer(layer_index);
    if (!l || !l->active || !l->tensor || !l->meta || l->n_used == 0) return 0;
    if (l->gate_expert_bytes != gate_expert_bytes ||
        l->down_expert_bytes != down_expert_bytes) return 0;
    const uint64_t n_pair = (uint64_t)n_tokens * n_expert;
    if (n_pair == 0 || n_pair > DS4_VK_POOL_SEL_CAP) return 0;
    *gate_b = vulkan_bind_tensor_at(DS4_VK_BINDING_B, l->tensor,
                                    vulkan_pool_gate_base(l),
                                    (uint64_t)l->n_slots * l->gate_expert_bytes);
    *up_b = vulkan_bind_tensor_at(DS4_VK_BINDING_C, l->tensor,
                                  vulkan_pool_up_base(l),
                                  (uint64_t)l->n_slots * l->gate_expert_bytes);
    *down_b = vulkan_bind_tensor_at(DS4_VK_BINDING_B, l->tensor,
                                    vulkan_pool_down_base(l),
                                    (uint64_t)l->n_slots * l->down_expert_bytes);
    *sel_b = vulkan_bind_tensor(DS4_VK_BINDING_OUT, selected);
    *tbl_b = vulkan_bind_tensor(DS4_VK_BINDING_TBL, l->meta);
    return 1;
}

/* Per-tier pool layer array: the live globals for the selected tier, the
 * saved snapshot inside the context for every other tier.  Immutable-tier
 * views let telemetry/reset cover ALL tiers without switching devices. */
static struct ds4_vk_pool_layer *vulkan_ctx_pool_layers(int t) {
    if (t == g_vk_ctx_active) return g_pool_layers;
    return g_vk_ctx[t].pool_layers;
}

extern "C" void ds4_vulkan_stream_pool_reset(void) {
    /* Reset EVERY tier's pool, not just the selected one: with N dynamic
     * tiers each device owns its own pool and all of them must be released. */
    const int prev = g_vk_ctx_active;
    const int n = g_vk_ctx_count > 1 ? g_vk_ctx_count : 1;
    for (int t = 0; t < n; t++) {
        if (n > 1 && ds4_vulkan_set_current_device(t) != 0) continue;
        vulkan_device_wait();
        if (getenv("DS4_VULKAN_DEBUG_BINDS") != NULL) {
            fprintf(stderr, "ds4: Vulkan debug: pool reset tier=%d total=%.2f GiB device_local=%.2f GiB\n",
                    t, (double)g_pool_total_bytes / 1073741824.0,
                    (double)g_device_local_bytes / 1073741824.0);
        }
        for (uint32_t i = 0; i < DS4_VK_POOL_MAX_LAYERS; i++) {
            struct ds4_vk_pool_layer *l = &g_pool_layers[i];
            if (l->tensor) ds4_gpu_tensor_free(l->tensor);
            if (l->meta) ds4_gpu_tensor_free(l->meta);
            memset(l, 0, sizeof(*l));
        }
        g_pool_total_bytes = 0;
        g_pool_ready = 0;
        g_pool_budget = 0;
        g_pool_layer_count = 0;
        g_pool_per_expert_bytes = 0;
    }
    if (n > 1 && prev >= 0 && prev != g_vk_ctx_active)
        (void)ds4_vulkan_set_current_device(prev);
}

/* Clear the route hotness of every layer (session restart / new decode),
 * without dropping the resident weights: mirrors Metal's
 * ds4_gpu_stream_expert_cache_reset_route_hotness.  Hotness is host-side, so
 * this covers all tiers without any device switch. */
extern "C" void ds4_vulkan_stream_pool_reset_hotness(void) {
    for (int t = 0; t < g_vk_ctx_count; t++) {
        struct ds4_vk_pool_layer *pl = vulkan_ctx_pool_layers(t);
        for (uint32_t i = 0; i < DS4_VK_POOL_MAX_LAYERS; i++) {
            struct ds4_vk_pool_layer *l = &pl[i];
            if (!l->active) continue;
            vulkan_pool_hotness_reset(l);
        }
    }
}

/* Effective expert budget (bytes / per-expert bytes) exposed to the compat
 * layer so the engine's hotlist/prefill preload scales with the pool size. */
extern "C" uint32_t ds4_vulkan_effective_budget(void) {
    const uint64_t eff = vulkan_pool_effective_budget_bytes();
    if (eff == 0 || g_pool_per_expert_bytes == 0) return 0;
    uint64_t n = eff / g_pool_per_expert_bytes;
    if (n > UINT32_MAX) n = UINT32_MAX;
    return (uint32_t)n;
}

/* Decode-island CUDA graph capture is CUDA-only; Vulkan decodes eagerly, so
 * there are never captured graphs whose baked buffer addresses could dangle.
 * The engine still calls this at graph-runtime teardown on every backend; a
 * silent no-op avoids the unavailable-stub noise at the end of every run. */
extern "C" void ds4_gpu_decode_graphs_invalidate(void) {}

/* Aggregate the per-layer telemetry counters into a snapshot for the engine's
 * --vulkan-stats report.  Reads are atomic per counter; gauges (n_used, slot
 * capacity, allocated bytes) are read without a lock, which is fine because
 * the engine snapshots between tokens when no async store is running. */
extern "C" void ds4_vulkan_telemetry_snapshot(ds4_gpu_expert_telemetry *t) {
    if (!t) return;
    memset(t, 0, sizeof(*t));
    uint64_t evictions = 0;
    uint64_t resident_bytes = 0, budget_bytes = 0;
    int ready = 0;
    /* Aggregate EVERY tier's pool: with N dynamic tiers the report must cover
     * all of them, not just the selected one.  The per-layer counters are
     * global (g_pool_tel[layer]); the layer set comes from each tier's
     * context (live globals for the selected tier). */
    for (int ct = 0; ct < g_vk_ctx_count; ct++) {
        const bool active = (ct == g_vk_ctx_active);
        const struct ds4_vk_pool_layer *pl = vulkan_ctx_pool_layers(ct);
        resident_bytes += active ? g_pool_total_bytes
                                 : g_vk_ctx[ct].pool_total_bytes;
        budget_bytes += active ? g_pool_max_bytes_fixed
                               : g_vk_ctx[ct].pool_max_bytes_fixed;
        if (active ? g_pool_ready : g_vk_ctx[ct].pool_ready) ready = 1;
        for (uint32_t i = 0; i < DS4_VK_POOL_MAX_LAYERS; i++) {
            const struct ds4_vk_pool_layer *l = &pl[i];
            const struct ds4_vk_pool_tel *tel = &g_pool_tel[i];
            if (!l->active) continue;
            t->layers_active++;
            t->resident_experts += l->n_used;
            t->pool_slots += l->n_slots;
            if (i < DS4_GPU_EXPERT_TELE_LAYERS) {
                uint64_t layer_req = 0, layer_miss = 0, layer_bytes = 0;
                for (int p = 0; p < DS4_GPU_EXPERT_PHASES; p++) {
                    const uint64_t req =
                        tel->requests[p].load(std::memory_order_relaxed);
                    const uint64_t miss =
                        tel->misses[p].load(std::memory_order_relaxed);
                    t->requests[p] += req;
                    t->hits[p] += tel->hits[p].load(std::memory_order_relaxed);
                    t->misses[p] += miss;
                    t->loaded_bytes[p] +=
                        tel->loaded_bytes[p].load(std::memory_order_relaxed);
                    t->reload_misses[p] +=
                        tel->reload_misses[p].load(std::memory_order_relaxed);
                    layer_req += req;
                    layer_miss += miss;
                    layer_bytes +=
                        tel->loaded_bytes[p].load(std::memory_order_relaxed);
                }
                t->layer_requests[i] = layer_req;
                t->layer_misses[i] = layer_miss;
                t->layer_loaded_bytes[i] = layer_bytes;
            }
            evictions += tel->evictions.load(std::memory_order_relaxed);
        }
    }
    t->evictions = evictions;
    t->wait_us = g_pool_tel_wait_us.load(std::memory_order_relaxed);
    t->resident_bytes = resident_bytes;
    t->budget_bytes = budget_bytes;
    t->ready = (uint8_t)ready;
}

/* --- Fase 4/6: routed MoE (Q8_0 and IQ2_XXS+Q2_K experts) ---------------- */

/* Routed MoE for Q8_0 expert weights (gate_type/down_type == 8), IQ2_XXS
 * gate/up with Q2_K down (gate_type == 16, down_type == 10), Q4_K experts
 * (gate_type == down_type == 12), or MXFP4 experts (gate_type == down_type
 * == 39); any other quant returns 0 so the engine falls back to its CPU
 * path.  Three dispatches:
 * gate/up/mid (weighted SwiGLU), down projection into the down scratch, then
 * the expert sum into out.  The model buffer is bound at the base of each
 * expert tensor with a range covering all n_total_expert experts; the shaders
 * address rows as expert*expert_bytes + row*row_bytes (+ 256-value super
 * block offsets for the IQ2_XXS/Q2_K path; rejected when the region would
 * exceed the 32-bit shader addressing). */
/* Autotuned rows-per-workgroup for the routed-MoE kernels (SPECS_AUTOTUNE.md
 * §2.5): 0 = not yet autotuned (default 8).  DS4_VULKAN_MOE_ROWS overrides. */
static uint32_t g_moe_rows = 0;

static uint32_t vulkan_moe_rows_per_group(void) {
    const char *re = getenv("DS4_VULKAN_MOE_ROWS");
    if (re && *re) {
        long v = strtol(re, NULL, 10);
        if (v >= 1 && v <= 64) return (uint32_t)v;
    }
    return g_moe_rows ? g_moe_rows : 8u;
}

/* Ensure the routed-MoE pair-order scratch holds at least `pairs` uint32. */
static ds4_gpu_tensor *vulkan_moe_order_ensure(uint64_t pairs) {
    const uint64_t bytes = pairs * sizeof(uint32_t);
    if (g_moe_order && g_moe_order->bytes >= bytes) return g_moe_order;
    if (g_moe_order) {
        ds4_gpu_tensor_free(g_moe_order);
        g_moe_order = NULL;
    }
    g_moe_order = ds4_gpu_tensor_alloc(bytes);
    return g_moe_order;
}

static int vulkan_routed_moe_launch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *down,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t n_tokens, uint32_t layer_index) {
    vulkan_trace("moe.begin layer=%u ntok=%u tier=%d", layer_index, n_tokens,
                 g_vk_ctx_active);
    if (getenv("DS4_VULKAN_DEBUG_FAIL"))
        fprintf(stderr, "ds4: MoE launch layer=%u ntok=%u gate=%u down=%u indim=%u mid=%u out=%u\n",
                layer_index, n_tokens, gate_type, down_type,
                expert_in_dim, expert_mid_dim, out_dim);
    if (!out || !gate || !up || !mid || !down || !model_map || !selected ||
        !weights || !x || n_tokens == 0 || n_total_expert == 0 ||
        n_expert == 0 || expert_in_dim == 0 || expert_mid_dim == 0 ||
        out_dim == 0) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL"))
            fprintf(stderr, "ds4: MoE fail: bad args layer=%u\n", layer_index);
        return 0;
    }
    /* Fase 7: wait any pending worker pool store (async expert load) before
     * this dispatch reads the pool.  Per-submit fence, not a device wait. */
    ds4_vulkan_pool_commit_pending();
    const int q8_path = (gate_type == 8u && down_type == 8u);
    const int iq2_path = (gate_type == 16u && down_type == 10u);
    const int q4_path = (gate_type == 12u && down_type == 12u);
    const int mxfp4_path = (gate_type == 39u && down_type == 39u);
    if (!q8_path && !iq2_path && !q4_path && !mxfp4_path) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL"))
            fprintf(stderr, "ds4: MoE fail: quant gate=%u down=%u layer=%u\n",
                    gate_type, down_type, layer_index);
        return 0;   /* Q8_0, IQ2_XXS+Q2_K, Q4_K or MXFP4 experts */
    }
    /* The IQ2_XXS/Q2_K and Q4_K kernels dequant 256-value super-blocks; MXFP4
     * uses 32-value blocks (handled by the generic /32 path below). */
    if ((iq2_path || q4_path) &&
        (expert_in_dim % 256u != 0 || expert_mid_dim % 256u != 0)) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL"))
            fprintf(stderr, "ds4: MoE fail: iq2/q4 divisibility layer=%u\n", layer_index);
        return 0;
    }
    if (mxfp4_path &&
        (expert_in_dim % 32u != 0 || expert_mid_dim % 32u != 0)) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL"))
            fprintf(stderr, "ds4: MoE fail: mxfp4 divisibility layer=%u\n", layer_index);
        return 0;
    }
    const uint64_t gate_region = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_region = (uint64_t)n_total_expert * down_expert_bytes;
    const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
    if (gate_region > UINT32_MAX || down_region > UINT32_MAX ||
        pair_count > UINT32_MAX || expert_in_dim > UINT32_MAX ||
        expert_mid_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL"))
            fprintf(stderr, "ds4: MoE fail: region size layer=%u gate_region=%llu down_region=%llu\n",
                    layer_index, (unsigned long long)gate_region,
                    (unsigned long long)down_region);
        return 0;
    }
    if (x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < pair_count * sizeof(int32_t) ||
        weights->bytes < pair_count * sizeof(float) ||
        gate->bytes < pair_count * expert_mid_dim * sizeof(float) ||
        up->bytes < pair_count * expert_mid_dim * sizeof(float) ||
        mid->bytes < pair_count * expert_mid_dim * sizeof(float) ||
        down->bytes < pair_count * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL"))
            fprintf(stderr, "ds4: MoE fail: tensor size layer=%u pair=%llu xb=%llu selb=%llu wb=%llu gateb=%llu\n",
                    layer_index, (unsigned long long)pair_count,
                    (unsigned long long)x->bytes,
                    (unsigned long long)selected->bytes,
                    (unsigned long long)weights->bytes,
                    (unsigned long long)gate->bytes);
        return 0;
    }

    /* Fase 6 step 3: the expert pool (when active for this layer) is the
     * source for the routed weights; the MoE reads the router's selected
     * expert ids and translates them through the on-device expert->slot table
     * (params.flags bit 0).  Otherwise bind the model windows covering the
     * full expert blobs. */
    struct ds4_vk_bind gate_b, up_b, down_b, sel_b, tbl_b;
    const int pool_mode = vulkan_pool_moe_binds(layer_index,
                                                gate_expert_bytes,
                                                down_expert_bytes,
                                                n_tokens, n_expert,
                                                selected,
                                                &gate_b, &up_b, &down_b,
                                                &sel_b, &tbl_b);
    if (!pool_mode) {
        if (gate_offset > model_size || gate_region > model_size - gate_offset ||
            up_offset > model_size || gate_region > model_size - up_offset ||
            down_offset > model_size || down_region > model_size - down_offset) {
            if (getenv("DS4_VULKAN_DEBUG_FAIL"))
                fprintf(stderr, "ds4: MoE fail: region bounds layer=%u\n", layer_index);
            return 0;
        }
        if (!vulkan_model_range_ok(gate_offset, gate_region) ||
            !vulkan_model_range_ok(up_offset, gate_region) ||
            !vulkan_model_range_ok(down_offset, down_region)) {
            if (getenv("DS4_VULKAN_DEBUG_FAIL"))
                fprintf(stderr, "ds4: MoE fail: window not covering layer=%u gate=%llu up=%llu down=%llu\n",
                        layer_index,
                        (unsigned long long)gate_offset,
                        (unsigned long long)up_offset,
                        (unsigned long long)down_offset);
            return 0;
        }
        gate_b = vulkan_bind_model(DS4_VK_BINDING_B, gate_offset, gate_region);
        up_b = vulkan_bind_model(DS4_VK_BINDING_C, up_offset, gate_region);
        down_b = vulkan_bind_model(DS4_VK_BINDING_B, down_offset, down_region);
        sel_b = vulkan_bind_tensor(DS4_VK_BINDING_OUT, selected);
    }

    struct ds4_vk_params p = {};
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    if (pool_mode) p.flags |= 1u;

    /* All routed-MoE gate/down kernels honour params.rsvd2 = rows per
     * workgroup, so the prefill grid can be collapsed on every path
     * (see SPECS_AUTOTUNE.md §2.5). */
    const uint32_t moe_R = vulkan_moe_rows_per_group();
    p.rsvd2 = moe_R;

    /* Opt-in expert grouping (DS4_VULKAN_MOE_GROUP): reorder the (token,
     * expert) pairs so the MXFP4 kernels walk them in expert order.  The
     * default keeps the raw pair order: the batch kernel is bound by the
     * MXFP4 dequant, not by weight bandwidth, so grouping only helps when the
     * expert set far exceeds L2. */
    ds4_gpu_tensor *order = NULL;
    if (mxfp4_path && getenv("DS4_VULKAN_MOE_GROUP") != NULL) {
        order = vulkan_moe_order_ensure(pair_count);
        if (!order) return 0;
        struct ds4_vk_params gp = {};
        gp.index = n_total_expert;
        gp.rows = (uint32_t)pair_count;
        gp.flags = pool_mode ? 1u : 0u;
        struct ds4_vk_bind gb[3];
        uint32_t gnb = 0;
        gb[gnb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, selected);
        if (pool_mode) {
            struct ds4_vk_bind gtb = tbl_b;
            gtb.binding = DS4_VK_BINDING_B;
            gb[gnb++] = gtb;
        }
        gb[gnb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, order);
        if (!vulkan_dispatch(g_pipes[DS4_PIPE_MOE_GROUP], &gp, sizeof(gp),
                             gb, gnb, 1, 1, 1)) {
            return 0;
        }
        p.flags |= 2u;
    }

    /* 1. gate/up/mid: grid (expert_mid_dim, pair_count). */
    p.in_dim = expert_in_dim;
    p.out_dim = expert_mid_dim;
    p.rows = n_tokens;
    p.index = n_expert;
    p.aux = (uint32_t)gate_expert_bytes;
    p.ratio = (uint32_t)gate_row_bytes;
    p.blocks = (iq2_path || q4_path) ? (expert_in_dim / 256u)
                                     : ((expert_in_dim + 31u) / 32u);
    p.clamp = clamp;
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    binds[nb++] = gate_b;
    binds[nb++] = up_b;
    binds[nb++] = sel_b;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, weights);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, gate);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT4, up);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT5, mid);
    if (pool_mode) binds[nb++] = tbl_b;
    if (order) binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_W, order);
    const VkPipeline gate_pipe = mxfp4_path ?
        vulkan_pipe_moe_mxfp4_gate() :
        (q4_path ? g_pipes[DS4_PIPE_MOE_GATE_UP_MID_Q4K] :
         (iq2_path ? vulkan_pipe_moe_gate_iq2()
                   : g_pipes[DS4_PIPE_MOE_GATE_UP_MID_Q8]));
    const uint32_t gate_gx = (expert_mid_dim + moe_R - 1u) / moe_R;
    if (!vulkan_dispatch(gate_pipe, &p, sizeof(p),
                         binds, nb, gate_gx, (uint32_t)pair_count, 1)) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL"))
            fprintf(stderr, "ds4: MoE fail: gate dispatch layer=%u pool=%d\n",
                    layer_index, pool_mode);
        return 0;
    }

    /* 2. down: grid (out_dim, pair_count). */
    p.in_dim = expert_mid_dim;
    p.out_dim = out_dim;
    p.aux = (uint32_t)down_expert_bytes;
    p.ratio = (uint32_t)down_row_bytes;
    p.blocks = (iq2_path || q4_path) ? (expert_mid_dim / 256u)
                                     : ((expert_mid_dim + 31u) / 32u);
    nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, mid);
    binds[nb++] = down_b;
    binds[nb++] = sel_b;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT5, down);
    if (pool_mode) binds[nb++] = tbl_b;
    if (order) binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_W, order);
    const VkPipeline down_pipe = mxfp4_path ?
        vulkan_pipe_moe_mxfp4_down() :
        (q4_path ? g_pipes[DS4_PIPE_MOE_DOWN_Q4K] :
         (iq2_path ? vulkan_pipe_moe_down_q2k()
                   : g_pipes[DS4_PIPE_MOE_DOWN_Q8]));
    const uint32_t down_gx = (out_dim + moe_R - 1u) / moe_R;
    if (!vulkan_dispatch(down_pipe, &p, sizeof(p),
                         binds, nb, down_gx, (uint32_t)pair_count, 1)) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL"))
            fprintf(stderr, "ds4: MoE fail: down dispatch layer=%u pool=%d\n",
                    layer_index, pool_mode);
        return 0;
    }

    /* 3. expert sum: out[tok][row] = sum_slot down[(tok*n_expert+slot)][row]. */
    p.out_dim = out_dim;
    p.rows = n_tokens;
    p.index = n_expert;
    nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, down);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT5, out);
    const uint64_t n = (uint64_t)n_tokens * out_dim;
    const uint32_t groups = (uint32_t)((n + 255u) / 256u);
    if (!vulkan_dispatch(g_pipes[DS4_PIPE_MOE_SUM], &p, sizeof(p),
                         binds, nb, groups, 1, 1)) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL"))
            fprintf(stderr, "ds4: MoE fail: sum dispatch layer=%u\n", layer_index);
        return 0;
    }
    vulkan_trace("moe.end layer=%u ok=1", layer_index);
    return 1;
}

int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *experts,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *add_in,
        uint32_t layer_index, bool force_resident) {
    (void)layer_index; (void)force_resident;
    if (add_in) {
        fprintf(stderr, "ds4: Vulkan routed MoE addend fold is Metal-only\n");
        return 0;
    }
    return vulkan_routed_moe_launch(
            out, gate, up, mid, experts, model_map, model_size,
            gate_offset, up_offset, down_offset, gate_type, down_type,
            gate_expert_bytes, gate_row_bytes, down_expert_bytes,
            down_row_bytes, expert_in_dim, expert_mid_dim, out_dim,
            selected, weights, n_total_expert, n_expert, clamp, x, 1u,
            layer_index);
}

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *experts,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens,
        bool *mid_is_f16, bool force_resident) {
    (void)layer_index; (void)force_resident;
    if (mid_is_f16) *mid_is_f16 = false;
    return vulkan_routed_moe_launch(
            out, gate, up, mid, experts, model_map, model_size,
            gate_offset, up_offset, down_offset, gate_type, down_type,
            gate_expert_bytes, gate_row_bytes, down_expert_bytes,
            down_row_bytes, expert_in_dim, expert_mid_dim, out_dim,
            selected, weights, n_total_expert, n_expert, clamp, x, n_tokens,
            layer_index);
}

/* SPECS_AUTOTUNE.md §2.5: pick R (rows-per-workgroup) for the routed-MoE
 * kernels by timing a small prefill-shaped MoE dispatch at init, before the
 * model is loaded.  R is a per-device dispatch property, so one probe on the
 * MXFP4 layout serves every quant path.  Cheap (~0.2 s worst case) and it
 * leaves the runtime exactly as the explicit DS4_VULKAN_MOE_ROWS override
 * would.  Skipped when DS4_VULKAN_MOE_ROWS is set or DS4_VULKAN_AUTOTUNE_OFF=1;
 * DS4_VULKAN_AUTOTUNE_PROFILE=1 logs each measurement. */
/* Benchmark the attn_output_low variants at init and pin the fastest per
 * device in g_attn_out_autotuned.  Uses a synthetic Q8 weight map and an
 * activation scratch (timing is shape-bound, data-independent), like
 * vulkan_autotune_moe_rows.  Skips when int-dot is unavailable (only v1) or a
 * forced variant is set. */
static void vulkan_autotune_attn_out(void) {
    if (getenv("DS4_VULKAN_AUTOTUNE_OFF")) return;
    if (getenv("DS4_VULKAN_FORCE_VARIANT")) return;
    if (!g_has_int_dot) return;
    if (g_pipes[DS4_PIPE_ATTN_OUTPUT_LOW_Q8_V2] == VK_NULL_HANDLE) return;

    /* DS4 Flash attention output shape. */
    const uint64_t group_dim = 4096u, rank = 1024u;
    const uint32_t n_groups = 8u;
    const uint64_t blocks = group_dim / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t map_size =
        ((uint64_t)n_groups * rank * row_bytes + 4095u) & ~(uint64_t)4095u;

    uint8_t *map = NULL;
    ds4_gpu_tensor *low = NULL, *heads = NULL;
    float *hv = NULL;
    if (posix_memalign((void **)&map, 4096, (size_t)map_size) == 0) {
        for (uint64_t i = 0; i < map_size; i++) {
            map[i] = (uint8_t)(i * 131u + 7u);
        }
        low = ds4_gpu_tensor_alloc((uint64_t)n_groups * rank * 4u);
        heads = ds4_gpu_tensor_alloc((uint64_t)n_groups * group_dim * 4u);
        if (low && heads) {
            hv = (float *)malloc((size_t)n_groups * group_dim * 4u);
        }
    }
    if (map && low && heads && hv) {
        for (uint64_t i = 0; i < (uint64_t)n_groups * group_dim; i++) {
            hv[i] = 0.2f * (float)(i % 13u) - 0.5f;
        }
        ds4_gpu_set_model_map(map, map_size);
        ds4_gpu_tensor_write(heads, 0, hv,
                             (uint64_t)n_groups * group_dim * 4u);

        int best = 0;
        double best_ms = 0.0;
        for (int v = 0; v < 3; v++) {
            if (v == 1 &&
                g_pipes[DS4_PIPE_ATTN_OUTPUT_LOW_Q8_V2] == VK_NULL_HANDLE)
                continue;
            if (v == 2 &&
                (g_pipes[DS4_PIPE_ATTN_OUTPUT_LOW_Q8_V3] == VK_NULL_HANDLE ||
                 blocks > 128u || (rank % 8u) != 0u))
                continue;
            g_attn_out_autotuned = v;
            /* Warm-up primes pipeline/descriptor state. */
            ds4_gpu_begin_commands();
            (void)vulkan_attn_output_low_q8(low, map, map_size, 0u, group_dim,
                                            rank, n_groups, 0u, n_groups,
                                            heads, 1u);
            ds4_gpu_end_commands();
            ds4_gpu_synchronize();
            const int iters = 30;
            ds4_gpu_begin_commands();
            const double t0 = vulkan_now_ms();
            for (int it = 0; it < iters; it++) {
                (void)vulkan_attn_output_low_q8(low, map, map_size, 0u,
                                                group_dim, rank, n_groups, 0u,
                                                n_groups, heads, 1u);
            }
            ds4_gpu_end_commands();
            ds4_gpu_synchronize();
            const double ms = (vulkan_now_ms() - t0) / (double)iters;
            if (getenv("DS4_VULKAN_AUTOTUNE_PROFILE")) {
                fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                        "autotune: ATTN_OUT v%d -> %.3f ms\n", v, ms);
            }
            if (best_ms == 0.0 || ms < best_ms) {
                best_ms = ms;
                best = v;
            }
        }
        g_attn_out_autotuned = best;
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "autotune: attn_output_low variant = v%d (%.3f ms, "
                "gdim=%llu rank=%llu)\n",
                best, best_ms, (unsigned long long)group_dim,
                (unsigned long long)rank);
    }

    free(hv);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_set_model_map(NULL, 0);
    free(map);
}

static void vulkan_autotune_moe_rows(void) {
    if (getenv("DS4_VULKAN_AUTOTUNE_OFF")) return;
    {
        const char *re = getenv("DS4_VULKAN_MOE_ROWS");
        if (re && *re) return;
    }
    /* MXFP4 (type 39) layout: 8 experts, 16 tokens x 6 used. */
    const uint32_t n_tok = 16u, n_used = 6u, n_total = 8u;
    const uint32_t in_dim = 4096u, mid_dim = 2048u, out_dim = 4096u;
    const uint64_t gate_row = (in_dim / 32u) * 17u;
    const uint64_t down_row = (mid_dim / 32u) * 17u;
    const uint64_t gate_expert = (uint64_t)mid_dim * gate_row;
    const uint64_t down_expert = (uint64_t)out_dim * down_row;
    const uint64_t gate_region = (uint64_t)n_total * gate_expert;
    const uint64_t down_region = (uint64_t)n_total * down_expert;
    const uint64_t up_off = gate_region;
    const uint64_t down_off = 2u * gate_region;
    const uint64_t map_size =
        (down_off + down_region + 4095u) & ~(uint64_t)4095u;
    const uint64_t pairs = (uint64_t)n_tok * n_used;

    uint8_t *map = NULL;
    ds4_gpu_tensor *x = NULL, *sel = NULL, *w = NULL, *gate = NULL;
    ds4_gpu_tensor *up = NULL, *mid = NULL, *down = NULL, *out = NULL;
    float *xv = NULL;
    int32_t *sv = NULL;
    float *wv = NULL;

    if (posix_memalign((void **)&map, 4096, (size_t)map_size) == 0) {
        for (uint64_t i = 0; i < map_size; i++)
            map[i] = (uint8_t)(i * 131u + 7u);
        x = ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * 4u);
        sel = ds4_gpu_tensor_alloc(pairs * 4u);
        w = ds4_gpu_tensor_alloc(pairs * 4u);
        gate = ds4_gpu_tensor_alloc(pairs * mid_dim * 4u);
        up = ds4_gpu_tensor_alloc(pairs * mid_dim * 4u);
        mid = ds4_gpu_tensor_alloc(pairs * mid_dim * 4u);
        down = ds4_gpu_tensor_alloc(pairs * out_dim * 4u);
        out = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * 4u);
    } else {
        map = NULL;
    }
    if (x && sel && w && gate && up && mid && down && out) {
        xv = (float *)malloc((size_t)n_tok * in_dim * 4u);
        sv = (int32_t *)malloc((size_t)pairs * 4u);
        wv = (float *)malloc((size_t)pairs * 4u);
    }
    if (map && xv && sv && wv) {
        for (uint32_t i = 0; i < n_tok * in_dim; i++)
            xv[i] = 0.2f * (float)(i % 13u) - 0.5f;
        for (uint64_t i = 0; i < pairs; i++) {
            sv[i] = (int32_t)((i * 5u + 1u) % n_total);
            wv[i] = 0.3f + 0.1f * (float)(i % 5u);
        }
        ds4_gpu_set_model_map(map, map_size);
        ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tok * in_dim * 4u);
        ds4_gpu_tensor_write(sel, 0, sv, pairs * 4u);
        ds4_gpu_tensor_write(w, 0, wv, pairs * 4u);

        static const uint32_t cand[] = { 1u, 4u, 8u };
        const int n_cand = (int)(sizeof(cand) / sizeof(cand[0]));
        const int max_iters = 1024;   /* stay under the descriptor-set HWM */
        const int profile = getenv("DS4_VULKAN_AUTOTUNE_PROFILE") != NULL;
        double target_ms = 1000.0;    /* whole probe budget, GPU-independent */
        {
            const char *tm = getenv("DS4_VULKAN_AUTOTUNE_MS");
            if (tm && *tm) {
                double v = atof(tm);
                if (v >= 10.0 && v <= 10000.0) target_ms = v;
            }
        }
        const double per_cand_ms = target_ms / (double)n_cand;
        auto do_moe = [&]() -> int {
            return ds4_gpu_routed_moe_batch_tensor(
                    out, gate, up, mid, down, map, map_size,
                    0u, up_off, down_off, 39u, 39u,
                    gate_expert, gate_row, down_expert, down_row,
                    in_dim, mid_dim, out_dim, sel, w,
                    n_total, n_used, 5.0f, x, 0u, n_tok, NULL, true);
        };
        /* Warm-up: prime the pipeline/descriptor state so the calibration is
         * representative and not dominated by the first-dispatch cost. */
        g_moe_rows = cand[0];
        ds4_gpu_begin_commands();
        (void)do_moe();
        ds4_gpu_end_commands();
        ds4_gpu_synchronize();

        uint32_t best = 8u;
        double best_ms = 0.0;
        for (int ci = 0; ci < n_cand; ci++) {
            g_moe_rows = cand[ci];
            /* Calibrate with a couple of dispatches, then size the timed pass
             * so each candidate is measured for ~per_cand_ms (GPU-independent
             * probe cost and enough iterations to suppress clock/DPM noise). */
            const int calib = 2;
            ds4_gpu_begin_commands();
            const double c0 = vulkan_now_ms();
            for (int it = 0; it < calib; it++) (void)do_moe();
            ds4_gpu_end_commands();
            ds4_gpu_synchronize();
            const double est = (vulkan_now_ms() - c0) / (double)calib;
            int iters = max_iters;
            if (est > 0.0) {
                iters = (int)(per_cand_ms / est + 0.5);
                if (iters < 1) iters = 1;
                if (iters > max_iters) iters = max_iters;
            }
            int ok = 1;
            ds4_gpu_begin_commands();
            const double t0 = vulkan_now_ms();
            for (int it = 0; it < iters; it++) {
                if (!do_moe()) {
                    ok = 0;
                    break;
                }
            }
            ds4_gpu_end_commands();
            ds4_gpu_synchronize();
            const double ms = (vulkan_now_ms() - t0) / (double)iters;
            if (profile)
                fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                        "autotune: MOE_ROWS R=%u -> %.3f ms @16 tok "
                        "(%d iters, calib %.3f ms)\n",
                        cand[ci], ms, iters, est);
            if (ok && (best_ms == 0.0 || ms < best_ms)) {
                best_ms = ms;
                best = cand[ci];
            }
        }
        g_moe_rows = best;
        fprintf(stderr, DS4_VULKAN_LOG_PREFIX
                "autotune: MoE rows/workgroup = %u (%.3f ms @16 tok)\n",
                best, best_ms);
    }

    free(xv); free(sv); free(wv);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(sel); ds4_gpu_tensor_free(w);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(out);
    ds4_gpu_set_model_map(NULL, 0);
    free(map);
}




/* --- Fase 5: HC (hyper-connection) ---------------------------------------- */

/* Bind a model range that spans two regions (scale + base), returning the
 * byte offsets of each region relative to the bind start.  Mirrors CUDA's
 * pair of cuda_resolve_weight_ptr calls with one descriptor. */
static int vulkan_bind_model_span(uint64_t off1, uint64_t size1,
                                  uint64_t off2, uint64_t size2,
                                  uint64_t *rel1, uint64_t *rel2) {
    const uint64_t start = off1 < off2 ? off1 : off2;
    const uint64_t end1 = off1 + size1;
    const uint64_t end2 = off2 + size2;
    const uint64_t end = end1 > end2 ? end1 : end2;
    /* Both regions must sit in one staged window: the HC scale/base tensors
     * are adjacent in the GGUF and land in the same coalesced window. */
    if (!vulkan_model_window_for(start, end - start)) return 0;
    *rel1 = off1 - start;
    *rel2 = off2 - start;
    return 1;
}

int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok,
                                 uint32_t n_head, uint32_t head_dim,
                                 float eps) {
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0) return 0;
    if (x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_tok;
    p.index = n_head;
    p.eps = eps;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, x);
    return vulkan_dispatch(g_pipes[DS4_PIPE_HEAD_RMS_NORM], &p, sizeof(p),
                           binds, nb, n_tok * n_head, 1, 1);
}

int ds4_gpu_head_rms_norm_rope_tail_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head,
        uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
        uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow,
        float eps) {
    if (!x || n_rot > head_dim || (n_rot & 1u) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_tok;
    p.index = n_head;
    p.n_rot = n_rot;
    p.pos0 = pos0;
    p.n_ctx_orig = n_ctx_orig;
    p.inverse = inverse ? 1 : 0;
    p.eps = eps;
    p.freq_base = freq_base;
    p.freq_scale = freq_scale;
    p.ext_factor = ext_factor;
    p.attn_factor = attn_factor;
    p.beta_fast = beta_fast;
    p.beta_slow = beta_slow;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, x);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, x);
    return vulkan_dispatch(g_pipes[DS4_PIPE_HEAD_RMS_NORM_ROPE_TAIL], &p,
                           sizeof(p), binds, nb, n_tok * n_head, 1, 1);
}

int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row,
                             uint32_t n_embd, uint32_t n_hc) {
    if (!out || !row || n_embd == 0 || n_hc == 0 ||
        row->bytes < (uint64_t)n_embd * sizeof(float) ||
        out->bytes < (uint64_t)n_embd * n_hc * sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = n_embd;
    p.rows = 1;
    p.index = n_hc;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, row);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    const uint64_t n = (uint64_t)n_embd * n_hc;
    const uint32_t groups = (uint32_t)((n + 255u) / 256u);
    return vulkan_dispatch(g_pipes[DS4_PIPE_REPEAT_HC], &p, sizeof(p),
                           binds, nb, groups, 1, 1);
}

int ds4_gpu_repeat_hc_rows_tensor(ds4_gpu_tensor *out,
                                  const ds4_gpu_tensor *rows,
                                  uint32_t n_tokens, uint32_t n_embd,
                                  uint32_t n_hc) {
    if (!out || !rows || n_tokens == 0 || n_embd == 0 || n_hc == 0 ||
        rows->bytes < (uint64_t)n_tokens * n_embd * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * n_embd * n_hc * sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = n_embd;
    p.rows = n_tokens;
    p.index = n_hc;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, rows);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    const uint64_t n = (uint64_t)n_tokens * n_embd * n_hc;
    const uint32_t groups = (uint32_t)((n + 255u) / 256u);
    return vulkan_dispatch(g_pipes[DS4_PIPE_REPEAT_HC], &p, sizeof(p),
                           binds, nb, groups, 1, 1);
}

/* Shared launcher for the flat HC weighted sum (tensor weights or split
 * weights; they differ only in the stride). */
static int vulkan_hc_weighted_sum_common(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc,
        uint32_t weight_stride) {
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0) {
        return 0;
    }
    const uint64_t out_bytes = (uint64_t)n_embd * sizeof(float);
    if (out->bytes < out_bytes || out->bytes % out_bytes != 0) return 0;
    const uint32_t n_tokens = (uint32_t)(out->bytes / out_bytes);
    if (residual_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd *
                                 sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * weight_stride * sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = n_embd;
    p.rows = n_tokens;
    p.index = n_hc;
    p.ratio = weight_stride;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, residual_hc);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, weights);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    const uint32_t groups = (uint32_t)((n + 255u) / 256u);
    return vulkan_dispatch(g_pipes[DS4_PIPE_HC_WEIGHTED_SUM], &p, sizeof(p),
                           binds, nb, groups, 1, 1);
}

int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out,
                                   const ds4_gpu_tensor *residual_hc,
                                   const ds4_gpu_tensor *weights,
                                   uint32_t n_embd, uint32_t n_hc) {
    return vulkan_hc_weighted_sum_common(out, residual_hc, weights, n_embd,
                                         n_hc, n_hc);
}

int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out,
                                         const ds4_gpu_tensor *residual_hc,
                                         const ds4_gpu_tensor *split,
                                         uint32_t n_embd, uint32_t n_hc) {
    const uint32_t stride = 2u * n_hc + n_hc * n_hc;
    return vulkan_hc_weighted_sum_common(out, residual_hc, split, n_embd,
                                         n_hc, stride);
}

/* Generic HC expand with post/comb as separate tensors.  flags&1 = has_add. */
static int vulkan_hc_expand_common(
        VkPipeline pipe, ds4_gpu_tensor *out_hc,
        const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc,
        uint32_t post_stride, uint32_t comb_stride, uint32_t flags) {
    if (!out_hc || !block_out || !residual_hc || !post || !comb ||
        n_embd == 0 || n_hc == 0) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out_hc->bytes < row_bytes || out_hc->bytes % row_bytes != 0) return 0;
    const uint32_t n_tokens = (uint32_t)(out_hc->bytes / row_bytes);
    if (block_out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float) ||
        residual_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd *
                                 sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = n_embd;
    p.rows = n_tokens;
    p.index = n_hc;
    p.aux = post_stride;
    p.blocks = comb_stride;
    p.flags = flags;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, block_out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B,
                                     (flags & 1u) ? block_add : block_out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, residual_hc);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out_hc);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, post);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, comb);
    const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const uint32_t groups = (uint32_t)((n_elem + 255u) / 256u);
    return vulkan_dispatch(pipe, &p, sizeof(p), binds, nb, groups, 1, 1);
}

int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc,
                             const ds4_gpu_tensor *block_out,
                             const ds4_gpu_tensor *residual_hc,
                             const ds4_gpu_tensor *post,
                             const ds4_gpu_tensor *comb, uint32_t n_embd,
                             uint32_t n_hc) {
    return vulkan_hc_expand_common(g_pipes[DS4_PIPE_HC_EXPAND], out_hc,
                                   block_out, block_out, residual_hc, post,
                                   comb, n_embd, n_hc, n_hc, n_hc * n_hc, 0u);
}

int ds4_gpu_hc_expand_add_tensor(ds4_gpu_tensor *out_hc,
                                 const ds4_gpu_tensor *block_out,
                                 const ds4_gpu_tensor *block_add,
                                 const ds4_gpu_tensor *residual_hc,
                                 const ds4_gpu_tensor *post,
                                 const ds4_gpu_tensor *comb,
                                 uint32_t n_embd, uint32_t n_hc) {
    return vulkan_hc_expand_common(g_pipes[DS4_PIPE_HC_EXPAND], out_hc,
                                   block_out, block_add, residual_hc, post,
                                   comb, n_embd, n_hc, n_hc, n_hc * n_hc, 1u);
}

/* Fused HC expand for the split layout (post = split+4, comb = split+8),
 * n_hc == 4.  flags: bit0 has_add, bit1 has_add2. */
static int vulkan_hc_expand4_common(
        VkPipeline pipe, ds4_gpu_tensor *out_hc,
        const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *block_add2, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc,
        uint32_t flags) {
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 ||
        n_hc != 4) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out_hc->bytes < row_bytes || out_hc->bytes % row_bytes != 0) return 0;
    const uint32_t n_tokens = (uint32_t)(out_hc->bytes / row_bytes);
    const uint64_t mix_bytes = 24u * sizeof(float);
    if (split->bytes < (uint64_t)n_tokens * mix_bytes ||
        residual_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd *
                                 sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = n_embd;
    p.rows = n_tokens;
    p.index = n_hc;
    p.flags = flags;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, block_out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B,
                                     (flags & 1u) ? block_add : block_out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C,
                                     (flags & 2u) ? block_add2 : block_out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out_hc);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, residual_hc);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, split);
    const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const uint32_t groups = (uint32_t)((n_elem + 255u) / 256u);
    return vulkan_dispatch(pipe, &p, sizeof(p), binds, nb, groups, 1, 1);
}

int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc,
                                   const ds4_gpu_tensor *block_out,
                                   const ds4_gpu_tensor *residual_hc,
                                   const ds4_gpu_tensor *split,
                                   uint32_t n_embd, uint32_t n_hc) {
    return vulkan_hc_expand4_common(g_pipes[DS4_PIPE_HC_EXPAND4], out_hc,
                                    block_out, block_out, block_out,
                                    residual_hc, split, n_embd, n_hc, 0u);
}

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    return vulkan_hc_expand4_common(g_pipes[DS4_PIPE_HC_EXPAND4], out_hc,
                                    block_out, block_add, block_out,
                                    residual_hc, split, n_embd, n_hc, 1u);
}

/* f16 block_out variants (attn_out_f16 / shared_down_f16 paths). */
static int vulkan_hc_expand_half_common(
        VkPipeline pipe, ds4_gpu_tensor *out_hc,
        const ds4_gpu_tensor *block_h, const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc, int has_add) {
    if (!out_hc || !block_h || !residual_hc || !split || n_embd == 0 ||
        n_hc != 4) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out_hc->bytes < row_bytes || out_hc->bytes % row_bytes != 0) return 0;
    const uint32_t n_tokens = (uint32_t)(out_hc->bytes / row_bytes);
    struct ds4_vk_params p = {};
    p.n = n_embd;
    p.rows = n_tokens;
    p.index = n_hc;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    if (has_add) {
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, block_h);
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, block_add_h);
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, residual_hc);
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out_hc);
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, split);
    } else {
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, block_h);
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, residual_hc);
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, split);
        binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out_hc);
    }
    const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const uint32_t groups = (uint32_t)((n_elem + 255u) / 256u);
    return vulkan_dispatch(pipe, &p, sizeof(p), binds, nb, groups, 1, 1);
}

int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    return vulkan_hc_expand_half_common(g_pipes[DS4_PIPE_HC_EXPAND4_HALF],
                                        out_hc, block_out_h, NULL,
                                        residual_hc, split, n_embd, n_hc, 0);
}

int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    return vulkan_hc_expand_half_common(g_pipes[DS4_PIPE_HC_EXPAND4_ADD_HALF],
                                        out_hc, block_out, block_add_h,
                                        residual_hc, split, n_embd, n_hc, 1);
}

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *mix,
        const void *model_map, uint64_t model_size, uint64_t scale_offset,
        uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters,
        float eps) {
    (void)model_map;
    if (!out || !mix || n_hc != 4) return 0;
    const uint64_t mix_bytes = 24ull * sizeof(float);
    if (scale_offset > model_size ||
        model_size - scale_offset < 3ull * sizeof(float) ||
        base_offset > model_size || model_size - base_offset < mix_bytes ||
        mix->bytes < mix_bytes || out->bytes < mix_bytes) {
        return 0;
    }
    uint64_t rel_scale = 0, rel_base = 0;
    if (!vulkan_bind_model_span(scale_offset, 3ull * sizeof(float),
                                base_offset, mix_bytes, &rel_scale,
                                &rel_base)) {
        return 0;
    }
    const uint64_t start = scale_offset < base_offset ? scale_offset
                                                      : base_offset;
    const uint64_t end1 = scale_offset + 3ull * sizeof(float);
    const uint64_t end2 = base_offset + mix_bytes;
    const uint64_t span = (end1 > end2 ? end1 : end2) - start;
    const uint64_t n_rows = mix->bytes / mix_bytes;
    struct ds4_vk_params p = {};
    p.n = n_hc;
    p.rows = (uint32_t)n_rows;
    p.aux = sinkhorn_iters;
    p.eps = eps;
    p.blocks = (uint32_t)rel_scale;
    p.ratio = (uint32_t)rel_base;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, mix);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, start, span);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    const uint32_t groups = (uint32_t)((n_rows + 255u) / 256u);
    return vulkan_dispatch(g_pipes[DS4_PIPE_HC_SPLIT_SINKHORN], &p, sizeof(p),
                           binds, nb, groups, 1, 1);
}

int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *split,
        const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual_hc,
        const void *model_map, uint64_t model_size, uint64_t scale_offset,
        uint64_t base_offset, uint32_t n_embd, uint32_t n_hc,
        uint32_t sinkhorn_iters, float eps) {
    (void)model_map;
    if (!out || !split || !mix || !residual_hc || n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        scale_offset > model_size ||
        3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL") != NULL)
            fprintf(stderr, "ds4: DBG hc_split bad sizes out=%llu mixb=%llu\n",
                    (unsigned long long)out->bytes,
                    (unsigned long long)mix_bytes);
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL") != NULL)
            fprintf(stderr, "ds4: DBG hc_split row sizes rows=%llu mix=%llu split=%llu resid=%llu\n",
                    (unsigned long long)n_rows,
                    (unsigned long long)mix->bytes,
                    (unsigned long long)split->bytes,
                    (unsigned long long)residual_hc->bytes);
        return 0;
    }
    uint64_t rel_scale = 0, rel_base = 0;
    if (!vulkan_bind_model_span(scale_offset, 3ull * sizeof(float),
                                base_offset, mix_bytes, &rel_scale,
                                &rel_base)) {
        if (getenv("DS4_VULKAN_DEBUG_FAIL") != NULL)
            fprintf(stderr, "ds4: DBG hc_split bind span FAIL scale=%llu base=%llu span=%llu\n",
                    (unsigned long long)scale_offset,
                    (unsigned long long)base_offset,
                    (unsigned long long)mix_bytes);
        return 0;
    }
    const uint64_t start = scale_offset < base_offset ? scale_offset
                                                      : base_offset;
    const uint64_t end1 = scale_offset + 3ull * sizeof(float);
    const uint64_t end2 = base_offset + mix_bytes;
    const uint64_t span = (end1 > end2 ? end1 : end2) - start;
    struct ds4_vk_params p = {};
    p.n = n_embd;
    p.rows = (uint32_t)n_rows;
    p.index = n_hc;
    p.aux = sinkhorn_iters;
    p.eps = eps;
    p.blocks = (uint32_t)rel_scale;
    p.ratio = (uint32_t)rel_base;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, mix);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, residual_hc);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, start, span);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, split);
    const int hc_split_rc = vulkan_dispatch(
            g_pipes[DS4_PIPE_HC_SPLIT_WEIGHTED_SUM_FUSED], &p,
            sizeof(p), binds, nb, (uint32_t)n_rows, 1, 1);
    if (hc_split_rc == 0 && getenv("DS4_VULKAN_DEBUG_FAIL") != NULL)
        fprintf(stderr, "ds4: DBG hc_split dispatch FAIL\n");
    return hc_split_rc;
}

int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *norm_out,
        ds4_gpu_tensor *split, const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc, const void *model_map,
        uint64_t model_size, uint64_t scale_offset, uint64_t base_offset,
        uint64_t norm_weight_offset, uint32_t n_embd, uint32_t n_hc,
        uint32_t sinkhorn_iters, float eps, float norm_eps) {
    if (!ds4_gpu_hc_split_weighted_sum_tensor(
            out, split, mix, residual_hc, model_map, model_size, scale_offset,
            base_offset, n_embd, n_hc, sinkhorn_iters, eps)) {
        return 0;
    }
    const uint32_t n_rows =
        (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    return ds4_gpu_rms_norm_weight_rows_tensor(
            norm_out, out, model_map, model_size, norm_weight_offset, n_embd,
            n_rows, norm_eps);
}

int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *pre,
        const void *model_map, uint64_t model_size, uint64_t scale_offset,
        uint64_t base_offset, uint32_t n_hc, float eps) {
    (void)model_map;
    if (!out || !pre || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0 || out->bytes < row_bytes || out->bytes % row_bytes != 0 ||
        pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    uint64_t rel_scale = 0, rel_base = 0;
    if (!vulkan_bind_model_span(scale_offset, sizeof(float), base_offset,
                                row_bytes, &rel_scale, &rel_base)) {
        return 0;
    }
    const uint64_t start = scale_offset < base_offset ? scale_offset
                                                      : base_offset;
    const uint64_t end1 = scale_offset + sizeof(float);
    const uint64_t end2 = base_offset + row_bytes;
    const uint64_t span = (end1 > end2 ? end1 : end2) - start;
    struct ds4_vk_params p = {};
    p.n = n_hc;
    p.rows = (uint32_t)n_tokens;
    p.eps = eps;
    p.blocks = (uint32_t)rel_scale;
    p.aux = (uint32_t)rel_base;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, pre);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, start, span);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    const uint64_t n = n_tokens * n_hc;
    const uint32_t groups = (uint32_t)((n + 255u) / 256u);
    return vulkan_dispatch(g_pipes[DS4_PIPE_OUTPUT_HC_WEIGHTS], &p, sizeof(p),
                           binds, nb, groups, 1, 1);
}

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor *x, const ds4_gpu_tensor *directions, uint32_t layer,
        uint32_t width, uint32_t rows, float scale) {
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) {
        return 0;
    }
    const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
    const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
    if (x->bytes < x_bytes || directions->bytes < dir_bytes) return 0;
    struct ds4_vk_params p = {};
    p.n = width;
    p.rows = rows;
    p.index = layer;
    p.weight = scale;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, directions);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, x);
    return vulkan_dispatch(g_pipes[DS4_PIPE_DIRECTIONAL_STEERING_PROJECT], &p,
                           sizeof(p), binds, nb, rows, 1, 1);
}

/* --- Fase 5: indexer ------------------------------------------------------ */

static int vulkan_indexer_scores_launch(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0, uint32_t n_head,
        uint32_t head_dim, uint32_t ratio, float scale, uint32_t causal) {
    if (!scores || !q || !weights || !index_comp || n_comp == 0 ||
        n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
        return 0;
    }
    if (causal && ratio == 0) return 0;
    struct ds4_vk_params p = {};
    p.n = n_comp;
    p.rows = n_tokens;
    p.index = pos0;
    p.aux = n_head;
    p.blocks = head_dim;
    p.ratio = ratio;
    p.weight = scale;
    p.flags = causal;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, q);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, weights);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_C, index_comp);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, scores);
    return vulkan_dispatch(g_pipes[DS4_PIPE_INDEXER_SCORES], &p, sizeof(p),
                           binds, nb, n_comp, n_tokens, 1);
}

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_head, uint32_t head_dim, float scale) {
    return vulkan_indexer_scores_launch(scores, q, weights, index_comp,
                                        n_comp, 1, 0, n_head, head_dim, 1,
                                        scale, 0);
}

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t n_head,
        uint32_t head_dim, uint32_t ratio, float scale) {
    return vulkan_indexer_scores_launch(scores, q, weights, index_comp,
                                        n_comp, n_tokens, 0, n_head, head_dim,
                                        ratio, scale, 1);
}

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0, uint32_t n_head,
        uint32_t head_dim, uint32_t ratio, float scale) {
    return vulkan_indexer_scores_launch(scores, q, weights, index_comp,
                                        n_comp, n_tokens, pos0, n_head,
                                        head_dim, ratio, scale, 1);
}

int ds4_gpu_indexer_topk_tensor(ds4_gpu_tensor *selected,
                                const ds4_gpu_tensor *scores,
                                uint32_t n_comp, uint32_t n_tokens,
                                uint32_t top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = n_comp;
    p.rows = n_tokens;
    p.index = top_k;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, scores);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, selected);
    return vulkan_dispatch(g_pipes[DS4_PIPE_INDEXER_TOPK], &p, sizeof(p),
                           binds, nb, n_tokens, 1, 1);
}

int ds4_gpu_indexer_top1_value_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *values,
        const ds4_gpu_tensor *scores, uint32_t n_comp, uint32_t n_tokens,
        uint32_t index_offset) {
    if (!selected || !values || !scores || n_comp == 0 || n_tokens == 0 ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * sizeof(uint32_t) ||
        values->bytes < (uint64_t)n_tokens * sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = n_comp;
    p.rows = n_tokens;
    p.index = index_offset;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, scores);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, selected);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, values);
    return vulkan_dispatch(g_pipes[DS4_PIPE_INDEXER_TOP1_VALUE], &p, sizeof(p),
                           binds, nb, n_tokens, 1, 1);
}

int ds4_gpu_dsv4_topk_mask_tensor(ds4_gpu_tensor *mask,
                                  const ds4_gpu_tensor *topk,
                                  uint32_t n_comp, uint32_t n_tokens,
                                  uint32_t top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = n_comp;
    p.rows = n_tokens;
    p.index = top_k;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, topk);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, mask);
    const uint64_t n = (uint64_t)n_tokens * n_comp;
    const uint32_t groups = (uint32_t)((n + 255u) / 256u);
    return vulkan_dispatch(g_pipes[DS4_PIPE_TOPK_MASK], &p, sizeof(p),
                           binds, nb, groups, 1, 1);
}

int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows,
                                    uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_rows;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, x);
    return vulkan_dispatch(g_pipes[DS4_PIPE_DSV4_INDEXER_QAT], &p, sizeof(p),
                           binds, nb, n_rows, 1, 1);
}

int ds4_gpu_dspark_markov_argmax_tensor(
        ds4_gpu_tensor *out_idx, const ds4_gpu_tensor *logits_row,
        const void *model_map, uint64_t model_size, uint64_t w1_offset,
        uint64_t w2_offset, uint32_t prev_token, uint32_t vocab,
        uint32_t rank) {
    (void)model_map;
    if (!out_idx || !logits_row || vocab == 0 || rank == 0 ||
        (rank & 31u) != 0u || rank > 256u ||
        out_idx->bytes < sizeof(unsigned long long) ||
        logits_row->bytes < (uint64_t)vocab * sizeof(float)) {
        return 0;
    }
    const uint32_t rank_blocks = rank / 32u;
    const uint64_t row_bytes = (uint64_t)rank_blocks * 34u;
    const uint64_t w1_row_off = w1_offset + (uint64_t)prev_token * row_bytes;
    const uint64_t w2_bytes = (uint64_t)vocab * row_bytes;
    if (w1_offset > model_size || w1_row_off > model_size ||
        row_bytes > model_size - w1_row_off ||
        w2_offset > model_size || w2_bytes > model_size - w2_offset) {
        return 0;
    }
    uint64_t rel_w1 = 0, rel_w2 = 0;
    if (!vulkan_bind_model_span(w1_row_off, row_bytes, w2_offset, w2_bytes,
                                &rel_w1, &rel_w2)) {
        return 0;
    }
    const uint64_t start = w1_row_off < w2_offset ? w1_row_off : w2_offset;
    const uint64_t end1 = w1_row_off + row_bytes;
    const uint64_t end2 = w2_offset + w2_bytes;
    const uint64_t span = (end1 > end2 ? end1 : end2) - start;
    struct ds4_vk_params p = {};
    p.n = vocab;
    p.blocks = rank_blocks;
    p.aux = (uint32_t)rel_w1;
    p.ratio = (uint32_t)rel_w2;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, logits_row);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, start, span);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out_idx);
    return vulkan_dispatch(g_pipes[DS4_PIPE_DSPARK_MARKOV_ARGMAX], &p,
                           sizeof(p), binds, nb, 1, 1, 1);
}

/* --- Fase 5: compressor --------------------------------------------------- */

static int vulkan_fill_f32(ds4_gpu_tensor *out, uint64_t count, float value) {
    if (!out || count == 0 || out->bytes < count * sizeof(float)) return 0;
    struct ds4_vk_params p = {};
    p.n = (uint32_t)count;
    p.weight = value;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, out);
    const uint32_t groups = (uint32_t)((count + 255u) / 256u);
    return vulkan_dispatch(g_pipes[DS4_PIPE_FILL_F32], &p, sizeof(p),
                           binds, nb, groups, 1, 1);
}

int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        const void *model_map, uint64_t model_size, uint64_t ape_offset,
        uint32_t ape_type, uint32_t head_dim, uint32_t ratio, uint32_t pos0,
        uint32_t n_tokens) {
    (void)model_map;
    if (!kv || !sc || !state_kv || !state_score || head_dim == 0 ||
        ratio == 0 || n_tokens == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.rows = n_tokens;
    p.ratio = ratio;
    p.pos0 = pos0;
    p.index = ape_type;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, kv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, sc);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, ape_offset, ape_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, state_kv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, state_score);
    const uint64_t n = (uint64_t)n_tokens * width;
    const uint32_t groups = (uint32_t)((n + 255u) / 256u);
    return vulkan_dispatch(g_pipes[DS4_PIPE_COMPRESSOR_STORE], &p, sizeof(p),
                           binds, nb, groups, 1, 1);
}

/* Low-level dispatches shared by the compressor prefill variants. */
static int vulkan_compressor_set_rows(
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc,
        const void *model_map, uint64_t model_size, uint64_t ape_offset,
        uint32_t ape_type, uint32_t width, uint32_t ratio, uint32_t pos0,
        uint32_t src0, uint32_t dst0, uint32_t rows) {
    (void)model_map;
    (void)model_size;
    const uint64_t ape_bytes =
        (uint64_t)width * ratio * (ape_type == 1u ? 2u : 4u);
    struct ds4_vk_params p = {};
    p.n = width;
    p.rows = rows;
    p.ratio = ratio;
    p.pos0 = pos0;
    p.index = ape_type;
    p.aux = src0;
    p.blocks = dst0;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, kv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, sc);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, ape_offset, ape_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, state_kv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, state_score);
    const uint64_t n = (uint64_t)rows * width;
    const uint32_t groups = (uint32_t)((n + 255u) / 256u);
    return vulkan_dispatch(g_pipes[DS4_PIPE_COMPRESSOR_SET_ROWS], &p,
                           sizeof(p), binds, nb, groups, 1, 1);
}

static int vulkan_compressor_prefill_pool(
        ds4_gpu_tensor *comp_cache, const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc, ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score, const void *model_map,
        uint64_t model_size, uint64_t ape_offset, uint32_t ape_type,
        uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_comp,
        uint32_t replay) {
    (void)model_map;
    (void)model_size;
    const uint64_t ape_bytes =
        (uint64_t)(ratio == 4u ? 2u : 1u) * head_dim * ratio *
        (ape_type == 1u ? 2u : 4u);
    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.index = ratio;
    p.pos0 = pos0;
    p.aux = ape_type;
    p.flags = replay ? 1u : 0u;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, kv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_B, sc);
    binds[nb++] = vulkan_bind_model(DS4_VK_BINDING_W, ape_offset, ape_bytes);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, state_kv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, state_score);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, comp_cache);
    const uint32_t gx = (head_dim + 255u) / 256u;
    return vulkan_dispatch(g_pipes[DS4_PIPE_COMPRESSOR_PREFILL_POOL], &p,
                           sizeof(p), binds, nb, gx, n_comp, 1);
}

int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor *comp_cache, ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score, const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc, const void *model_map, uint64_t model_size,
        uint64_t ape_offset, uint32_t ape_type, uint64_t norm_offset,
        uint32_t norm_type, uint32_t head_dim, uint32_t ratio, uint32_t pos0,
        uint32_t n_tokens, uint32_t n_rot, uint32_t n_ctx_orig,
        bool quantize_fp8, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow,
        float rms_eps) {
    (void)model_map;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (n_comp && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    const uint64_t state_n = (uint64_t)state_rows * width;
    if (!vulkan_fill_f32(state_kv, state_n, 0.0f)) return 0;
    if (!vulkan_fill_f32(state_score, state_n, -INFINITY)) return 0;

    if (ratio == 4u) {
        if (cutoff >= ratio) {
            const uint32_t prev_start = cutoff - ratio;
            if (!vulkan_compressor_set_rows(
                    state_kv, state_score, kv, sc, model_map, model_size,
                    ape_offset, ape_type, width, ratio, pos0, prev_start, 0u,
                    ratio)) {
                return 0;
            }
        }
        if (rem != 0) {
            if (!vulkan_compressor_set_rows(
                    state_kv, state_score, kv, sc, model_map, model_size,
                    ape_offset, ape_type, width, ratio, pos0, cutoff, ratio,
                    rem)) {
                return 0;
            }
        }
    } else if (rem != 0) {
        if (!vulkan_compressor_set_rows(
                state_kv, state_score, kv, sc, model_map, model_size,
                ape_offset, ape_type, width, ratio, pos0, cutoff, 0u, rem)) {
            return 0;
        }
    }
    if (n_comp != 0) {
        if (!vulkan_compressor_prefill_pool(
                comp_cache, kv, sc, state_kv, state_score, model_map,
                model_size, ape_offset, ape_type, head_dim, ratio, pos0,
                n_comp, 0)) {
            return 0;
        }
        if (!ds4_gpu_rms_norm_weight_rows_tensor(
                comp_cache, comp_cache, model_map, model_size, norm_offset,
                head_dim, n_comp, rms_eps)) {
            return 0;
        }
        if (n_rot != 0) {
            if (!vulkan_rope_tail(comp_cache, n_comp, 1, head_dim, n_rot,
                                  pos0, ratio, n_ctx_orig, false, freq_base,
                                  freq_scale, ext_factor, attn_factor,
                                  beta_fast, beta_slow)) {
                return 0;
            }
        }
        if (quantize_fp8 &&
            !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim,
                                                 n_rot)) {
            return 0;
        }
    }
    return 1;
}

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur, const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        ds4_gpu_tensor *comp_cache, const void *model_map, uint64_t model_size,
        uint64_t ape_offset, uint32_t ape_type, uint64_t norm_offset,
        uint32_t norm_type, uint32_t head_dim, uint32_t ratio, uint32_t pos,
        uint32_t comp_row, uint32_t n_rot, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, float rms_eps,
        bool state_already_stored, bool decode_one_token,
        bool defer_finalize) {
    (void)decode_one_token;
    (void)defer_finalize;
    (void)model_map;
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        head_dim == 0 || ratio == 0 || n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv_cur->bytes < kv_bytes || sc_cur->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (emit && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    if (!state_already_stored) {
        if (!ds4_gpu_compressor_store_batch_tensor(
                kv_cur, sc_cur, state_kv, state_score, model_map, model_size,
                ape_offset, ape_type, head_dim, ratio, pos, 1)) {
            return 0;
        }
    }
    if (!emit) return 1;
    ds4_gpu_tensor *row_view = ds4_gpu_tensor_view(
            comp_cache, (uint64_t)comp_row * head_dim * sizeof(float),
            (uint64_t)head_dim * sizeof(float));
    if (!row_view) return 0;
    struct ds4_vk_params p = {};
    p.n = head_dim;
    p.index = ratio;
    struct ds4_vk_bind binds[DS4_VK_MAX_BINDS];
    uint32_t nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, state_kv);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, state_score);
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT3, row_view);
    int ok = vulkan_dispatch(g_pipes[DS4_PIPE_COMPRESSOR_UPDATE_POOL], &p,
                             sizeof(p), binds, nb,
                             (head_dim + 255u) / 256u, 1, 1);
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(
                row_view, row_view, model_map, model_size, norm_offset,
                head_dim, 1, rms_eps);
    }
    if (ok && n_rot != 0) {
        ok = ds4_gpu_rope_tail_tensor(row_view, 1, 1, head_dim, n_rot,
                                      pos + 1u - ratio, n_ctx_orig, false,
                                      freq_base, freq_scale, ext_factor,
                                      attn_factor, beta_fast, beta_slow);
    }
    ds4_gpu_tensor_free(row_view);
    if (ok && ratio == 4u) {
        struct ds4_vk_params sp = {};
        sp.n = width;
        struct ds4_vk_bind sb[DS4_VK_MAX_BINDS];
        uint32_t snb = 0;
        sb[snb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT, state_kv);
        sb[snb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT2, state_score);
        ok = vulkan_dispatch(g_pipes[DS4_PIPE_COMPRESSOR_SHIFT_RATIO4], &sp,
                             sizeof(sp), sb, snb, (4ull * width + 255u) / 256u,
                             1, 1);
    }
    return ok;
}

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor *comp_cache, ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score, const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc, const void *model_map, uint64_t model_size,
        uint64_t ape_offset, uint32_t ape_type, uint64_t norm_offset,
        uint32_t norm_type, uint32_t head_dim, uint32_t pos0,
        uint32_t n_tokens, uint32_t n_rot, uint32_t n_ctx_orig,
        bool quantize_fp8, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow,
        float rms_eps) {
    (void)model_map;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 ||
        (pos0 & 3u) != 0 || n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        comp_cache->bytes < comp_bytes) {
        return 0;
    }
    if (!vulkan_compressor_prefill_pool(
            comp_cache, kv, sc, state_kv, state_score, model_map, model_size,
            ape_offset, ape_type, head_dim, ratio, pos0, n_comp, 1)) {
        return 0;
    }
    if (!ds4_gpu_rms_norm_weight_rows_tensor(
            comp_cache, comp_cache, model_map, model_size, norm_offset,
            head_dim, n_comp, rms_eps)) {
        return 0;
    }
    if (n_rot != 0) {
        if (!vulkan_rope_tail(comp_cache, n_comp, 1, head_dim, n_rot, pos0,
                              ratio, n_ctx_orig, false, freq_base, freq_scale,
                              ext_factor, attn_factor, beta_fast, beta_slow)) {
            return 0;
        }
    }
    if (quantize_fp8 &&
        !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim,
                                             n_rot)) {
        return 0;
    }
    const uint64_t state_n = (uint64_t)state_rows * width;
    if (!vulkan_fill_f32(state_kv, state_n, 0.0f)) return 0;
    if (!vulkan_fill_f32(state_score, state_n, -INFINITY)) return 0;
    const uint32_t prev_start = n_tokens - ratio;
    return vulkan_compressor_set_rows(
            state_kv, state_score, kv, sc, model_map, model_size, ape_offset,
            ape_type, width, ratio, pos0, prev_start, 0u, ratio);
}

int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        const ds4_gpu_tensor *kv_tail, const ds4_gpu_tensor *sc_tail,
        const void *model_map, uint64_t model_size, uint64_t ape_offset,
        uint32_t ape_type, uint32_t head_dim, uint32_t pos0) {
    (void)model_map;
    if (!state_kv || !state_score || !kv_tail || !sc_tail || head_dim == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const uint64_t state_n = (uint64_t)state_rows * width;
    if (!vulkan_fill_f32(state_kv, state_n, 0.0f)) return 0;
    if (!vulkan_fill_f32(state_score, state_n, -INFINITY)) return 0;
    return vulkan_compressor_set_rows(
            state_kv, state_score, kv_tail, sc_tail, model_map, model_size,
            ape_offset, ape_type, width, ratio, pos0, 0u, 0u, ratio);
}

/* --- Fase 5: matmul + HC expand (composition, CUDA non-fused fallback) --- */

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor *out_hc, ds4_gpu_tensor *block_out,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || out_dim != n_embd) {
        return 0;
    }
    if (!ds4_gpu_matmul_q8_0_tensor(block_out, model_map, model_size,
                                    weight_offset, in_dim, out_dim, x, 1)) {
        return 0;
    }
    return ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc,
                                          split, n_embd, n_hc);
}

int ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
        ds4_gpu_tensor *out_hc, ds4_gpu_tensor *block_out,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim, uint64_t in_start,
        uint64_t in_count, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !x || !block_add || !residual_hc || !split ||
        !model_map || in_dim == 0 || out_dim == 0 || in_count == 0 ||
        n_embd == 0 || n_hc == 0 || out_dim != (uint64_t)n_embd) {
        return 0;
    }
    if ((in_start % 32u) != 0 || (in_count % 32u) != 0 ||
        in_start > in_dim || in_count > in_dim - in_start) {
        return 0;
    }
    if (!ds4_gpu_matmul_q8_0_kslice_tensor(
            block_out, model_map, model_size, weight_offset, in_dim,
            in_start, in_count, out_dim, x, 0)) {
        return 0;
    }
    return ds4_gpu_hc_expand_add_split_tensor(out_hc, block_out, block_add,
                                              residual_hc, split, n_embd,
                                              n_hc);
}
