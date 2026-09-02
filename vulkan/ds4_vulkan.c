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

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include "shaders/ds4_vulkan_shaders.inc"

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
static int               g_cmd_i     = 0;   /* CB currently recording */
static bool              g_commands_active = false;

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
static VkCommandBuffer   g_worker_cb   = VK_NULL_HANDLE;
static VkFence           g_worker_fence = VK_NULL_HANDLE;
static int               g_worker_fence_pending = 0;
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
static int vulkan_submit_one_shot(void);
static int vulkan_compute_init(void);

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

static void vulkan_device_wait(void) {
    if (g_device != VK_NULL_HANDLE) {
        pthread_mutex_lock(&g_device_wait_mutex);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        vkDeviceWaitIdle(g_device);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms = (double)(t1.tv_sec - t0.tv_sec) * 1000.0 +
                          (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
        if (ms > 1000.0 && getenv("DS4_VULKAN_DEBUG_BINDS") != NULL) {
            fprintf(stderr, "ds4: Vulkan debug: device wait %.0f ms\n", ms);
        }
    }
    g_vulkan_device_dirty = 0;
    /* Every set allocated for a completed scope is safe to reclaim.  With the
     * static decode map the model windows are no longer re-staged per layer,
     * so this is the per-layer reset point for the descriptor pool (Fase 6
     * step 3; the pool is only otherwise reset at synchronize). */
    if (g_desc_pool != VK_NULL_HANDLE && !g_commands_active) {
        vkResetDescriptorPool(g_device, g_desc_pool, 0);
        g_desc_sets_allocated = 0;
    }
    pthread_mutex_unlock(&g_device_wait_mutex);
}

#define DS4_VK_PIPE_COUNT 61
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
    DS4_PIPE_MOE_GATE_UP_MID_IQ2XXS_V2,
    DS4_PIPE_MOE_DOWN_Q2K_V2,
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
};

static struct ds4_vk_pool_layer g_pool_layers[DS4_VK_POOL_MAX_LAYERS];
static uint32_t g_pool_slots_per_layer = DS4_VK_POOL_MIN_SLOTS;
static uint32_t g_pool_budget = 0;
static uint32_t g_pool_layer_count = 0;   /* routed-expert layers in the model */
static uint64_t g_pool_per_expert_bytes = 0;
static uint64_t g_pool_total_bytes = 0;
static int g_pool_ready = 0;   /* budget-derived slot count configured */

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

/* Per-tensor device handle.  tensor->ptr points at one of these (heap). */
struct ds4_vulkan_tensor {
    VkBuffer         buffer;
    VkDeviceMemory   memory;
    VkDeviceSize     offset;      /* byte offset of this tensor within buffer */
    uint64_t         bytes;
    int              owner;       /* owns buffer+memory */
    int              device_local; /* memory lives in the device-local heap */
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
static double vulkan_probe_transfer_bw(VkPhysicalDevice phys) {
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

static int vulkan_pick_device(void) {
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
    int forced = -1;
    if (dev_env) {
        char *end = NULL;
        long v = strtol(dev_env, &end, 10);
        if (end != dev_env && v >= 0 && v < (long)count) forced = (int)v;
    }
    if (forced >= 0) {
        chosen = devices[forced];
    } else {
        VkPhysicalDevice first_usable = VK_NULL_HANDLE;
        uint32_t first_usable_idx = 0;
        double best_bw = 0.0;
        uint32_t chosen_idx = 0;
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
    const char *dev_exts[1];
    uint32_t n_dev_exts = 0;
    VkPhysicalDeviceFeatures2 feat2 = {};
    VkPhysicalDeviceBufferDeviceAddressFeatures bda_feat = {};
    VkPhysicalDeviceFeatures dev_feats = {};
    vkGetPhysicalDeviceFeatures(g_phys, &dev_feats);
    if (getenv("DS4_VULKAN_DEBUG_ROBUST") != NULL) {
        /* Clamp out-of-bounds shader accesses instead of faulting: a
         * diagnostic to confirm an OOB write and to let the decode proceed. */
        dev_feats.robustBufferAccess = VK_TRUE;
    }
    if (getenv("DS4_VULKAN_DEBUG_BINDS") != NULL) {
        uint32_t dext_n = 0;
        if (vkEnumerateDeviceExtensionProperties(g_phys, NULL, &dext_n, NULL) ==
            VK_SUCCESS && dext_n > 0) {
            VkExtensionProperties *dexts = (VkExtensionProperties *)calloc(
                    dext_n, sizeof(*dexts));
            if (dexts) {
                vkEnumerateDeviceExtensionProperties(g_phys, NULL, &dext_n, dexts);
                for (uint32_t i = 0; i < dext_n; i++) {
                    if (strcmp(dexts[i].extensionName, bda_ext) == 0) {
                        dev_exts[n_dev_exts++] = bda_ext;
                        break;
                    }
                }
                free(dexts);
            }
        }
        if (n_dev_exts > 0) {
            bda_feat.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
            bda_feat.bufferDeviceAddress = VK_TRUE;
        }
    }
    feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feat2.features = dev_feats;
    if (n_dev_exts > 0) {
        feat2.pNext = &bda_feat;
    }

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
    if (n_dev_exts > 0 || dev_feats.robustBufferAccess) dci.pNext = &feat2;

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
    const VkDescriptorBindingFlags flags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    const VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        NULL, sizeof(bindings) / sizeof(bindings[0]), &flags,
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
        { ds4_spv_moe_gate_up_mid_iq2xxs_v2,
          ds4_spv_moe_gate_up_mid_iq2xxs_v2_len,
          "moe_gate_up_mid_iq2xxs_v2" },
        { ds4_spv_moe_down_q2k_v2, ds4_spv_moe_down_q2k_v2_len,
          "moe_down_q2k_v2" },
    };
    for (uint32_t i = 0; i < DS4_VK_PIPE_COUNT; i++) {
        g_mods[i] = vulkan_create_shader_module(pipes[i].code, pipes[i].len);
        if (!g_mods[i]) return 0;
        g_pipes[i] = vulkan_create_compute_pipeline(g_mods[i], g_pipe_layout,
                                                    pipes[i].entry);
        if (!g_pipes[i]) return 0;
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
}

/* --- public contract: lifecycle ---------------------------------------- */

int ds4_gpu_init(void) {
    if (g_device != VK_NULL_HANDLE) return 1; /* already initialized */

    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "ds4";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_0;

    /* Instance extensions: none required (Vulkan 1.0 core). */

    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 0;
    ici.ppEnabledExtensionNames = NULL;

    VkResult rc = vkCreateInstance(&ici, NULL, &g_instance);
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkCreateInstance");
        return 0;
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

    fprintf(stderr, DS4_VULKAN_LOG_PREFIX "initialized: %s (api 0x%08x)\n",
            g_props.deviceName, (unsigned)g_props.apiVersion);
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
        for (uint32_t i = 0; i < g_model_window_count; i++) {
            ds4_gpu_tensor_free(g_model_windows[i].tensor);
            g_model_windows[i].tensor = NULL;
            g_model_windows[i].buffer = VK_NULL_HANDLE;
        }
        g_model_window_count = 0;
        for (int i = 0; i < 2; i++) {
            if (g_cmd_fence[i] != VK_NULL_HANDLE &&
                g_cmd_fence[i] != g_readback_fence) {
                vkDestroyFence(g_device, g_cmd_fence[i], NULL);
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
}

/* --- public contract: tensors ------------------------------------------ */

static ds4_gpu_tensor *vulkan_tensor_new(void) {
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (t) t->device_id = 0;
    return t;
}

static struct ds4_vulkan_tensor *vulkan_handle_new(void) {
    return (struct ds4_vulkan_tensor *)calloc(1,
            sizeof(struct ds4_vulkan_tensor));
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
static int vulkan_upload_to_tensor(ds4_gpu_tensor *t, uint64_t offset,
                                   const void *data, uint64_t bytes) {
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(t);
    if (!h || !data || bytes == 0) return 0;
    if (offset > h->bytes || bytes > h->bytes - offset) return 0;
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
    if (h->host_map) {
        if (g_vulkan_device_dirty) vulkan_device_wait();
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
        dh->buffer && sh->buffer) {
        const uint64_t d_start = dh->offset + dst_offset;
        const uint64_t s_start = sh->offset + src_offset;
        const uint64_t d_end = d_start + bytes;
        const uint64_t s_end = s_start + bytes;
        const int disjoint = d_end <= s_start || s_end <= d_start;
        if (disjoint) {
            if (!vulkan_compute_init()) return 0;
            VkCommandBuffer cb = vulkan_dispatch_begin();
            if (!cb) return 0;
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
    /* device -> device: stage through a temporary host buffer */
    void *tmp = malloc((size_t)bytes);
    if (!tmp) return 0;
    int ok = vulkan_download_from_tensor(src, src_offset, tmp, bytes);
    if (ok) ok = vulkan_upload_to_tensor(dst, dst_offset, tmp, bytes);
    free(tmp);
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

/* One-shot submit of the current compute CB (used when the caller dispatches
 * without an open command scope): end, submit, wait, reset. */
static int vulkan_submit_one_shot(void) {
    if (vkEndCommandBuffer(g_cmd[g_cmd_i]) != VK_SUCCESS) return 0;
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd[g_cmd_i];
    VkResult rc = vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE);
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
        if (f != g_readback_fence) vkResetFences(g_device, 1, &f);
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

/* End + submit the currently recording CB with its in-flight fence (the
 * GPU starts it immediately; the fence marks the CB busy until it is
 * reused by vulkan_cb_acquire). */
static int vulkan_cb_submit(void) {
    if (vkEndCommandBuffer(g_cmd[g_cmd_i]) != VK_SUCCESS) return 0;
    if (g_cmd_fence[g_cmd_i] == VK_NULL_HANDLE) {
        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(g_device, &fci, NULL, &g_cmd_fence[g_cmd_i]) != VK_SUCCESS) {
            return 0;
        }
    } else if (vkResetFences(g_device, 1, &g_cmd_fence[g_cmd_i]) != VK_SUCCESS) {
        return 0;
    }
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd[g_cmd_i];
    const VkResult rc = vkQueueSubmit(g_queue, 1, &si, g_cmd_fence[g_cmd_i]);
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkQueueSubmit(scope)");
        return 0;
    }
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
        if (g_cmd_pool == VK_NULL_HANDLE) return 0;
        VkCommandBufferAllocateInfo cbai = {};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = g_cmd_pool;
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
    const VkResult rc = vkQueueSubmit(g_queue, 1, &si, g_worker_fence);
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
    if (getenv("DS4_VULKAN_DEBUG_SUBMIT") != NULL) {
        const double e0 = vulkan_now_ms();
        if (vkEndCommandBuffer(g_cmd[g_cmd_i]) != VK_SUCCESS) {
            g_commands_active = false;
            return 0;
        }
        const double e1 = vulkan_now_ms();
        if (g_readback_fence == VK_NULL_HANDLE) {
            VkFenceCreateInfo fci = {};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (vkCreateFence(g_device, &fci, NULL, &g_readback_fence) != VK_SUCCESS) {
                return 0;
            }
        } else if (vkResetFences(g_device, 1, &g_readback_fence) != VK_SUCCESS) {
            return 0;
        }
        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &g_cmd[g_cmd_i];
        const VkResult rc = vkQueueSubmit(g_queue, 1, &si, g_readback_fence);
        double wq = 0.0;
        if (rc == VK_SUCCESS) {
            wq = 0.0;
        }
        const double e3 = vulkan_now_ms();
        g_commands_active = false;
        fprintf(stderr, "ds4: Vulkan debug submit: end=%.3f ms queue=%.3f ms fwait=%.3f ms total=%.3f ms ndisp=%u binds=%u bytes=%llu\n",
                e1 - e0, 0.0, wq, e3 - e0, g_scope_dispatch_count,
                g_scope_bind_count, (unsigned long long)g_scope_bind_bytes);
        g_scope_bind_bytes = 0;
        g_scope_bind_count = 0;
        g_scope_dispatch_count = 0;
        if (rc != VK_SUCCESS) {
            vulkan_log_vk(rc, "vkQueueSubmit(readback scope)");
            return 0;
        }
        g_cmd_fence[g_cmd_i] = g_readback_fence;
        g_readback_fence_pending = 1;
        g_vulkan_device_dirty = 1;
        return 1;
    }

    if (vkEndCommandBuffer(g_cmd[g_cmd_i]) != VK_SUCCESS) {
        g_commands_active = false;
        return 0;
    }
    if (g_readback_fence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(g_device, &fci, NULL, &g_readback_fence) != VK_SUCCESS) {
            g_commands_active = false;
            return 0;
        }
    } else if (vkResetFences(g_device, 1, &g_readback_fence) != VK_SUCCESS) {
        g_commands_active = false;
        return 0;
    }
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd[g_cmd_i];
    const VkResult rc = vkQueueSubmit(g_queue, 1, &si, g_readback_fence);
    g_commands_active = false;
    if (rc != VK_SUCCESS) {
        vulkan_log_vk(rc, "vkQueueSubmit(readback scope)");
        return 0;
    }
    g_cmd_fence[g_cmd_i] = g_readback_fence;
    g_readback_fence_pending = 1;
    g_vulkan_device_dirty = 1;
    return 1;
}

extern "C" int ds4_gpu_wait_selected_readback_ready(uint64_t event_value,
                                                    const char *label) {
    (void)event_value;
    (void)label;
    if (g_device == VK_NULL_HANDLE) return 1;
    if (!g_readback_fence_pending) return 1;
    const VkResult rc = vkWaitForFences(g_device, 1, &g_readback_fence,
                                        VK_TRUE, UINT64_MAX);
    g_readback_fence_pending = 0;
    return rc == VK_SUCCESS ? 1 : 0;
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

static struct ds4_vk_bind vulkan_bind_tensor(uint32_t binding,
                                             const ds4_gpu_tensor *t) {
    struct ds4_vk_bind b = {};
    struct ds4_vulkan_tensor *h = vulkan_tensor_handle(t);
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
    if (h && off <= h->bytes && bytes <= h->bytes - off) {
        b.binding = binding;
        b.buffer = h->buffer;
        b.offset = h->offset + off;
        b.range = bytes;
    }
    return b;
}

/* Find the staged window covering [offset, offset+bytes).  NULL when no
 * window covers the whole range.  Uses offset - base (never base+size -
 * offset) so an offset beyond the window end cannot underflow. */
static struct ds4_vk_model_window *vulkan_model_window_for(
        uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return NULL;
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
static ds4_gpu_tensor *vulkan_scratch_a(uint64_t bytes) {
    if (g_scratch_a && g_scratch_a->bytes >= bytes) return g_scratch_a;
    vulkan_device_wait();
    if (g_scratch_a) ds4_gpu_tensor_free(g_scratch_a);
    g_scratch_a = ds4_gpu_tensor_alloc(bytes);
    return g_scratch_a;
}

static ds4_gpu_tensor *vulkan_scratch_b(uint64_t bytes) {
    if (g_scratch_b && g_scratch_b->bytes >= bytes) return g_scratch_b;
    vulkan_device_wait();
    if (g_scratch_b) ds4_gpu_tensor_free(g_scratch_b);
    g_scratch_b = ds4_gpu_tensor_alloc(bytes);
    return g_scratch_b;
}

static ds4_gpu_tensor *vulkan_scratch_c(uint64_t bytes) {
    if (g_scratch_c && g_scratch_c->bytes >= bytes) return g_scratch_c;
    vulkan_device_wait();
    if (g_scratch_c) ds4_gpu_tensor_free(g_scratch_c);
    g_scratch_c = ds4_gpu_tensor_alloc(bytes);
    return g_scratch_c;
}

static ds4_gpu_tensor *vulkan_scratch_d(uint64_t bytes) {
    if (g_scratch_d && g_scratch_d->bytes >= bytes) return g_scratch_d;
    vulkan_device_wait();
    if (g_scratch_d) ds4_gpu_tensor_free(g_scratch_d);
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

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cb, g_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, params_size, params);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            g_pipe_layout, 0, 1, &set, 0, NULL);
    vkCmdDispatch(cb, gx, gy, gz);
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
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &mb, 0, NULL, 0, NULL);
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
 * next one (the only waits are the per-CB fences at CB reuse).  Env
 * DS4_VULKAN_FLUSH_END_BEGIN=0 keeps the legacy no-op flush for bisect. */
int ds4_gpu_flush_commands(void) {
    if (g_device == VK_NULL_HANDLE) return 0;
    if (!g_commands_active) return 1;
    const char *feb = getenv("DS4_VULKAN_FLUSH_END_BEGIN");
    if (feb != NULL && feb[0] == '0') return 1;
    if (!vulkan_cb_submit()) {
        g_commands_active = false;
        return 0;
    }
    g_commands_active = false;
    g_cmd_i = 1 - g_cmd_i;
    if (vulkan_cb_acquire(0) == VK_NULL_HANDLE) return 0;
    g_commands_active = true;
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
            /* The scope is complete: drop the fence so the next acquire does
             * not wait a (reset) never-signaled fence. */
            vkDestroyFence(g_device, fence, NULL);
            g_cmd_fence[g_cmd_i] = VK_NULL_HANDLE;
        }
        const double e3 = vulkan_now_ms();
        g_commands_active = false;
        fprintf(stderr, "ds4: Vulkan debug submit: end=%.3f ms queue=%.3f ms fwait=%.3f ms total=%.3f ms ndisp=%u binds=%u bytes=%llu\n",
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
    if (g_desc_pool != VK_NULL_HANDLE) {
        vkResetDescriptorPool(g_device, g_desc_pool, 0);
        g_desc_sets_allocated = 0;
    }
    return 1;
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
static void vulkan_set_span_windows(const void *host_ptr, uint64_t model_size,
                                    const uint64_t *offsets,
                                    const uint64_t *sizes, uint32_t count) {
    struct ds4_vk_model_span spans[DS4_VK_MAX_MODEL_WINDOWS];
    uint32_t n = 0;
    for (uint32_t i = 0; i < count && n < DS4_VK_MAX_MODEL_WINDOWS; i++) {
        if (sizes[i] == 0 || offsets[i] > model_size ||
            sizes[i] > model_size - offsets[i]) {
            continue;
        }
        spans[n].lo = offsets[i] & ~(g_host_pointer_align - 1);
        const uint64_t end = offsets[i] + sizes[i];
        spans[n].hi = (end + g_host_pointer_align - 1) &
                      ~(g_host_pointer_align - 1);
        n++;
    }
    if (n == 0) {
        vulkan_destroy_model_wrapper();
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
    return g_pipes[DS4_PIPE_MATMUL_Q8_0_PREQ_V2];
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
int ds4_gpu_matmul_quant_tensor(ds4_gpu_tensor *out, const void *model_map,
                                uint64_t model_size, uint64_t weight_offset,
                                uint32_t weight_type, uint64_t in_dim,
                                uint64_t out_dim, const ds4_gpu_tensor *x,
                                uint64_t n_tok) {
    if (weight_type != 8u) return 0;   /* DS4_TENSOR_Q8_0 */
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                      weight_offset, in_dim, out_dim, x,
                                      n_tok);
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
    return vulkan_dispatch(g_pipes[DS4_PIPE_ATTN_OUTPUT_LOW_Q8], &p, sizeof(p),
                           binds, nb, (uint32_t)low_dim, n_rows, 1);
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
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups_total, uint32_t group0, uint32_t group_cnt,
        const ds4_gpu_tensor *heads, uint32_t n_rows) {
    return vulkan_attn_output_low_q8(low, model_map, model_size, out_a_offset,
                                     group_dim, rank, n_groups_total, group0,
                                     group_cnt, heads, n_rows);
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
 * the uploads), evicting the least-recently-used slot that is not part of
 * the current selection.  Returns the slot index or -1. */
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
    uint32_t min_age = UINT32_MAX;
    for (uint32_t i = 0; i < l->n_slots; i++) {
        if (vulkan_pool_id_requested(ids, n_ids, l->slots[i])) continue;
        if (l->slot_age[i] < min_age) {
            min_age = l->slot_age[i];
            victim = i;
        }
    }
    if (victim == UINT32_MAX) return -1;
    const int32_t old_e = l->slots[victim];
    if (old_e >= 0 && (uint32_t)old_e < DS4_VK_POOL_TABLE_ENTRIES) {
        l->table_host[(uint32_t)old_e] = -1;
    }
    if (!vulkan_pool_slot_mark(l, victim, e)) return -1;
    return (int)victim;
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

    struct ds4_vulkan_tensor *ph = vulkan_tensor_handle(l->tensor);
    if (!ph) return 0;
    if (ph->host_map) {
        for (uint32_t i = 0; i < n_missing; i++) {
            const int32_t e = missing[i];
            const int slot = vulkan_pool_slot_for(l, e);
            if (slot < 0 || e < 0) return 0;
            const uint64_t eg = (uint64_t)e;
            memcpy(ph->host_map + ph->offset + vulkan_pool_gate_base(l) +
                       (uint64_t)slot * l->gate_expert_bytes,
                   src + table->gate_offset + eg * l->gate_expert_bytes,
                   (size_t)l->gate_expert_bytes);
            memcpy(ph->host_map + ph->offset + vulkan_pool_up_base(l) +
                       (uint64_t)slot * l->gate_expert_bytes,
                   src + table->up_offset + eg * l->gate_expert_bytes,
                   (size_t)l->gate_expert_bytes);
            memcpy(ph->host_map + ph->offset + vulkan_pool_down_base(l) +
                       (uint64_t)slot * l->down_expert_bytes,
                   src + table->down_offset + eg * l->down_expert_bytes,
                   (size_t)l->down_expert_bytes);
        }
        return 1;
    }

    if (g_pool_staging && g_pool_staging->bytes < total) {
        vulkan_device_wait();
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
    for (uint32_t i = 0; i < n_missing; i++) {
        const int32_t e = missing[i];
        const int slot = vulkan_pool_slot_for(l, e);
        if (slot < 0 || e < 0 || nr + 3 > 3 * DS4_VK_POOL_MAX_SLOTS) return 0;
        const uint64_t eg = (uint64_t)e;
        /* gate */
        memcpy(sh->host_map + sh->offset + s_off,
               src + table->gate_offset + eg * l->gate_expert_bytes,
               (size_t)l->gate_expert_bytes);
        regions[nr].srcOffset = sh->offset + s_off;
        regions[nr].dstOffset = ph->offset + vulkan_pool_gate_base(l) +
                                (uint64_t)slot * l->gate_expert_bytes;
        regions[nr].size = l->gate_expert_bytes;
        nr++;
        s_off += l->gate_expert_bytes;
        /* up */
        memcpy(sh->host_map + sh->offset + s_off,
               src + table->up_offset + eg * l->gate_expert_bytes,
               (size_t)l->gate_expert_bytes);
        regions[nr].srcOffset = sh->offset + s_off;
        regions[nr].dstOffset = ph->offset + vulkan_pool_up_base(l) +
                                (uint64_t)slot * l->gate_expert_bytes;
        regions[nr].size = l->gate_expert_bytes;
        nr++;
        s_off += l->gate_expert_bytes;
        /* down */
        memcpy(sh->host_map + sh->offset + s_off,
               src + table->down_offset + eg * l->down_expert_bytes,
               (size_t)l->down_expert_bytes);
        regions[nr].srcOffset = sh->offset + s_off;
        regions[nr].dstOffset = ph->offset + vulkan_pool_down_base(l) +
                                (uint64_t)slot * l->down_expert_bytes;
        regions[nr].size = l->down_expert_bytes;
        nr++;
        s_off += l->down_expert_bytes;
    }
    if (!vulkan_compute_init()) return 0;
    if (async) {
        if (!vulkan_worker_copy_submit_multi(sh->buffer, ph->buffer, regions,
                                             nr)) {
            return 0;
        }
        return 1;
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
    return vulkan_submit_one_shot();
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
                                  int32_t *remap_out, int async) {
    if (!l || !table || !ids || !remap_out || n_ids == 0) return -1;
    if (n_ids > DS4_VK_POOL_SEL_CAP || !l->tensor || !l->meta) return -1;

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
    if (n_missing != 0 && !async) vulkan_device_wait();

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
    return (int)n_ids;
}

/* Wait any pending worker store/readback copy (per-submit fence, no device
 * wait).  The main thread calls this right before a dispatch that reads the
 * pool (routed MoE). */
extern "C" void ds4_vulkan_pool_commit_pending(void) {
    vulkan_worker_copy_wait();
}

extern "C" int ds4_vulkan_stream_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *ids, uint32_t n_selected) {
    if (!table || !ids || n_selected == 0 ||
        table->model_map == NULL ||
        table->gate_expert_bytes == 0 || table->down_expert_bytes == 0) {
        return 0;
    }
    struct ds4_vk_pool_layer *l = vulkan_pool_layer(table->layer);
    if (!l) return 0;
    if (!vulkan_pool_layer_ensure(l, table->gate_expert_bytes,
                                  table->down_expert_bytes, n_selected,
                                  n_selected, 1)) {
        return 0;
    }
    int32_t remap[DS4_VK_POOL_SEL_CAP];
    return vulkan_pool_seed_remap(l, table, ids, n_selected, remap, 0) >= 0;
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
    int32_t remap[DS4_VK_POOL_SEL_CAP];
    return vulkan_pool_seed_remap(l, table, ids, n_selected, remap, 1) >= 0;
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
    int32_t remap[DS4_VK_POOL_SEL_CAP];
    if (vulkan_pool_seed_remap(l, table, ids, n_ids, remap, 0) < 0) return 0;
    return 1;
}

extern "C" int ds4_vulkan_stream_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *ids, uint32_t n_experts) {
    if (!table || !ids || n_experts == 0 || table->model_map == NULL ||
        table->gate_expert_bytes == 0 || table->down_expert_bytes == 0) {
        return 0;
    }
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
    int32_t remap[DS4_VK_POOL_SEL_CAP];
    if (vulkan_pool_seed_remap(l, table, ids, n, remap, 0) < 0) return 0;
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

extern "C" void ds4_vulkan_stream_pool_reset(void) {
    vulkan_device_wait();
    if (getenv("DS4_VULKAN_DEBUG_BINDS") != NULL) {
        fprintf(stderr, "ds4: Vulkan debug: pool reset total=%.2f GiB device_local=%.2f GiB\n",
                (double)g_pool_total_bytes / 1073741824.0,
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

/* Effective expert budget (bytes / per-expert bytes) exposed to the compat
 * layer so the engine's hotlist/prefill preload scales with the pool size. */
extern "C" uint32_t ds4_vulkan_effective_budget(void) {
    const uint64_t eff = vulkan_pool_effective_budget_bytes();
    if (eff == 0 || g_pool_per_expert_bytes == 0) return 0;
    uint64_t n = eff / g_pool_per_expert_bytes;
    if (n > UINT32_MAX) n = UINT32_MAX;
    return (uint32_t)n;
}

/* --- Fase 4/6: routed MoE (Q8_0 and IQ2_XXS+Q2_K experts) ---------------- */

/* Routed MoE for Q8_0 expert weights (gate_type/down_type == 8) or IQ2_XXS
 * gate/up with Q2_K down (gate_type == 16, down_type == 10); any other quant
 * returns 0 so the engine falls back to its CPU path.  Three dispatches:
 * gate/up/mid (weighted SwiGLU), down projection into the down scratch, then
 * the expert sum into out.  The model buffer is bound at the base of each
 * expert tensor with a range covering all n_total_expert experts; the shaders
 * address rows as expert*expert_bytes + row*row_bytes (+ 256-value super
 * block offsets for the IQ2_XXS/Q2_K path; rejected when the region would
 * exceed the 32-bit shader addressing). */
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
    if (!out || !gate || !up || !mid || !down || !model_map || !selected ||
        !weights || !x || n_tokens == 0 || n_total_expert == 0 ||
        n_expert == 0 || expert_in_dim == 0 || expert_mid_dim == 0 ||
        out_dim == 0) {
        return 0;
    }
    /* Fase 7: wait any pending worker pool store (async expert load) before
     * this dispatch reads the pool.  Per-submit fence, not a device wait. */
    ds4_vulkan_pool_commit_pending();
    const int q8_path = (gate_type == 8u && down_type == 8u);
    const int iq2_path = (gate_type == 16u && down_type == 10u);
    if (!q8_path && !iq2_path) return 0;   /* Q8_0 or IQ2_XXS+Q2_K experts */
    /* The IQ2_XXS/Q2_K kernels dequant 256-value super-blocks. */
    if (iq2_path &&
        (expert_in_dim % 256u != 0 || expert_mid_dim % 256u != 0)) {
        return 0;
    }
    const uint64_t gate_region = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_region = (uint64_t)n_total_expert * down_expert_bytes;
    const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
    if (gate_region > UINT32_MAX || down_region > UINT32_MAX ||
        pair_count > UINT32_MAX || expert_in_dim > UINT32_MAX ||
        expert_mid_dim > UINT32_MAX || out_dim > UINT32_MAX) {
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
            return 0;
        }
        if (!vulkan_model_range_ok(gate_offset, gate_region) ||
            !vulkan_model_range_ok(up_offset, gate_region) ||
            !vulkan_model_range_ok(down_offset, down_region)) {
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

    /* 1. gate/up/mid: grid (expert_mid_dim, pair_count). */
    p.in_dim = expert_in_dim;
    p.out_dim = expert_mid_dim;
    p.rows = n_tokens;
    p.index = n_expert;
    p.aux = (uint32_t)gate_expert_bytes;
    p.ratio = (uint32_t)gate_row_bytes;
    p.blocks = iq2_path ? (expert_in_dim / 256u) : ((expert_in_dim + 31u) / 32u);
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
    const VkPipeline gate_pipe = iq2_path ?
        vulkan_pipe_moe_gate_iq2() :
        g_pipes[DS4_PIPE_MOE_GATE_UP_MID_Q8];
    if (!vulkan_dispatch(gate_pipe, &p, sizeof(p),
                         binds, nb, expert_mid_dim, (uint32_t)pair_count, 1)) {
        return 0;
    }

    /* 2. down: grid (out_dim, pair_count). */
    p.in_dim = expert_mid_dim;
    p.out_dim = out_dim;
    p.aux = (uint32_t)down_expert_bytes;
    p.ratio = (uint32_t)down_row_bytes;
    p.blocks = iq2_path ? (expert_mid_dim / 256u) : ((expert_mid_dim + 31u) / 32u);
    nb = 0;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_A, mid);
    binds[nb++] = down_b;
    binds[nb++] = sel_b;
    binds[nb++] = vulkan_bind_tensor(DS4_VK_BINDING_OUT5, down);
    if (pool_mode) binds[nb++] = tbl_b;
    const VkPipeline down_pipe = iq2_path ?
        vulkan_pipe_moe_down_q2k() : g_pipes[DS4_PIPE_MOE_DOWN_Q8];
    if (!vulkan_dispatch(down_pipe, &p, sizeof(p),
                         binds, nb, out_dim, (uint32_t)pair_count, 1)) {
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
    return vulkan_dispatch(g_pipes[DS4_PIPE_MOE_SUM], &p, sizeof(p),
                           binds, nb, groups, 1, 1);
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
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
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
    return vulkan_dispatch(g_pipes[DS4_PIPE_HC_SPLIT_WEIGHTED_SUM_FUSED], &p,
                           sizeof(p), binds, nb, (uint32_t)n_rows, 1, 1);
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
