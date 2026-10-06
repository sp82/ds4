/* vulkan/tools/mgpu_probe/mgpu_probe.c -- multi-GPU capability probe.
 *
 * Standalone (links only Vulkan): enumerates the physical devices, reports
 * per-device VRAM, measures device-local bandwidth four ways -- pure read
 * (compute kernel), pure write (compute kernel), pure write (vkCmdFillBuffer,
 * DMA cross-check), and read+write (D2D copy) -- plus host<->device bandwidth,
 * and tests cross-device (P2P) transfer via
 * VK_KHR_external_memory_fd (dma-buf / opaque fd) + optional external
 * semaphores.
 *
 * Bandwidth semantics: the DRAM bus has ONE total rate (spec ~512 GB/s on an
 * RX 6900 XT).  A D2D copy consumes 2 bus bytes per copied byte, so its
 * per-direction rate is ~half the bus and its r+w sum is bounded by the bus
 * (~87% efficient is normal).  Pure read/pure write are the per-direction
 * rates and approach the full bus rate.  Do not compare the copy's "r=" or
 * "w=" value to the spec number: compare its r+w sum (or use pure read/write).
 *
 * Purpose: characterize the hardware BEFORE implementing the native Vulkan
 * multi-GPU backend (SPEC.md §8e / progress.md §1b), so placement and the
 * P2P-vs-host-bounce decision are made on measured facts.
 *
 * Build:  make mgpu-probe
 * Run:    ./mgpu-probe [--mb N] [--devices 0,1,2,3] [--iters N] [--warmup N] [--p2p-fresh]
 * Env:    DS4_VULKAN_DEVICE_INDEX is ignored (this tool wants every device).
 *         DS4_MGPU_PROBE_CPU=1 also include CPU/llvmpipe devices.
 *         DS4_MGPU_PROBE_IMPORT_DEBUG=1 prints the imported buffer's memory
 *         type; DS4_MGPU_PROBE_IMPORT_MEMTYPE=<idx> forces it (diagnostics:
 *         e.g. tell a placement problem from a real cross-vendor P2P limit).
 *         DS4_MGPU_PROBE_COPY_ON_SRC=1 owns the shared buffer on the
 *         destination and runs the copy on the source (workaround for a
 *         driver that cannot export an importable VRAM buffer, e.g. NVIDIA).
 *
 * Output: one block per device + a P2P matrix (ordered pairs) + a host-bounce
 * matrix (A->host->B, the fallback path when direct P2P is unavailable, e.g.
 * cross-vendor).  Host-bounce is derived as the harmonic mean of the two
 * measured one-way host<->device rates.  Any unsupported step is reported,
 * never fatal.
 *
 * --warmup N runs N untimed copies (submitted and waited) before each timed
 * measurement so freshly-ramped devices (e.g. AMD after runtime-suspend)
 * report a stable value instead of a different one every run.  AMD cards need
 * a fairly large warmup (~40+ copies) to boost the memory clock; the default
 * is 50 so the device-local number is stable out of the box.
 *
 * Defaults: --mb 256, --iters 5, --warmup 50.  mb is clamped to >= 256 MiB: a
 * smaller working set fits in the Infinity Cache (128 MiB on Navi 21) and would
 * measure cache bandwidth (~1.7 TB/s), not DRAM (~450 GB/s r+w on a 6900 XT).
 *
 * Robustness: every P2P pair is content-verified (the destination must contain
 * the marker written into the source), compared against the host-bounce
 * estimate, and flagged "suspect" when it exceeds the PCIe bandwidth of either
 * endpoint.  A successful submit alone is NOT proof that the copy crossed the
 * PCIe link: the imported dma-buf may have been migrated/cached locally.  The
 * per-iteration min/max exposes a costly first touch.  --p2p-fresh re-creates
 * the exported/imported buffer for every timed copy to defeat such warmup
 * migration (slower, opt-in).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <vulkan/vulkan.h>
#include "read_spv.h"
#include "write_spv.h"

#define LOG "mgpu-probe: "

/* Infinity Cache on Navi 21 is 128 MiB: a copy working set smaller than ~2x
 * that fits in cache and measures cache bandwidth, not DRAM.  Enforce a floor
 * so a small --mb can never report a cache-inflated number. */
#define MIN_MB 256u

/* Marker pattern written into the P2P source so the copy can be verified. */
#define P2P_FILL 0xA5A5A5A5u

/* P2P result sentinels stored in the matrix (positive = GB/s). */
#define P2P_N_A   (-2)   /* external memory dma-buf unsupported */
#define P2P_NOFD  (-3)   /* vkGetMemoryFdKHR failed */
#define P2P_ALLOC (-4)   /* alloc/import failed */
#define P2P_FAIL  (-1)   /* submit failed or verification mismatch */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int has_ext(const VkExtensionProperties *v, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (strcmp(v[i].extensionName, name) == 0) return 1;
    return 0;
}

typedef struct {
    VkPhysicalDevice phys;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem;
    VkDevice dev;
    uint32_t qfam;
    VkQueue queue;
    VkCommandPool pool;
    VkCommandBuffer cb;
    int ext_mem_fd;
    int ext_sem_fd;
    int dma_buf;
    int mem_budget;
    uint64_t vram_bytes;
    uint32_t pci_dom, pci_bus, pci_dev, pci_fn;
    int ok;
    /* pure-read bandwidth bench (compute kernel from read_spv.h) */
    VkShaderModule rb_mod;
    VkDescriptorSetLayout rb_dsl;
    VkPipelineLayout rb_pl;
    VkPipeline rb_pipe;
    VkDescriptorPool rb_pool;
    VkDescriptorSet rb_set;
    int rb_ok;
    /* pure-write bandwidth bench (compute kernel from write_spv.h) */
    VkShaderModule wb_mod;
    VkPipeline wb_pipe;
    int wb_ok;
} Dev;

static PFN_vkGetMemoryFdKHR pGetMemoryFdKHR;
static PFN_vkGetMemoryFdPropertiesKHR pGetMemoryFdPropertiesKHR;
static PFN_vkGetPhysicalDeviceExternalBufferProperties
    pGetPhysicalDeviceExternalBufferProperties;

static int find_mem_type(const Dev *d, uint32_t bits, VkMemoryPropertyFlags want,
                         uint32_t *out) {
    for (uint32_t i = 0; i < d->mem.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        if ((d->mem.memoryTypes[i].propertyFlags & want) == want) {
            *out = i;
            return 1;
        }
    }
    return 0;
}

/* Allocate a buffer.  want_export: make it exportable (dedicated alloc). */
static int alloc_buf(const Dev *d, uint64_t bytes, int want_export,
                     int host_visible, VkBufferUsageFlags usage,
                     VkBuffer *buf_out, VkDeviceMemory *mem_out) {
    VkBufferCreateInfo bci;
    memset(&bci, 0, sizeof(bci));
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf = VK_NULL_HANDLE;
    if (vkCreateBuffer(d->dev, &bci, NULL, &buf) != VK_SUCCESS) return 0;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(d->dev, buf, &req);

    VkMemoryPropertyFlags want = host_visible
        ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t mt = 0;
    if (!find_mem_type(d, req.memoryTypeBits, want, &mt)) {
        /* fall back to any type with the wanted visibility, then any */
        if (!find_mem_type(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &mt) &&
            !find_mem_type(d, req.memoryTypeBits, 0, &mt)) {
            vkDestroyBuffer(d->dev, buf, NULL);
            return 0;
        }
    }

    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;
    VkMemoryDedicatedAllocateInfo ded;
    memset(&ded, 0, sizeof(ded));
    ded.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    ded.buffer = buf;
    VkExportMemoryAllocateInfo exp;
    memset(&exp, 0, sizeof(exp));
    exp.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    VkExternalMemoryHandleTypeFlagBits ht = d->dma_buf
        ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
        : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    exp.handleTypes = ht;
    if (want_export) {
        ai.pNext = &exp;
        exp.pNext = &ded;
    }
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(d->dev, &ai, NULL, &mem) != VK_SUCCESS) {
        /* retry without dedicated/export */
        ai.pNext = NULL;
        if (vkAllocateMemory(d->dev, &ai, NULL, &mem) != VK_SUCCESS) {
            vkDestroyBuffer(d->dev, buf, NULL);
            return 0;
        }
    }
    if (vkBindBufferMemory(d->dev, buf, mem, 0) != VK_SUCCESS) {
        vkFreeMemory(d->dev, mem, NULL);
        vkDestroyBuffer(d->dev, buf, NULL);
        return 0;
    }
    *buf_out = buf;
    *mem_out = mem;
    return 1;
}

/* Import a dma-buf/opaque fd as a buffer on device d. */
static int import_buf2(Dev *d, int fd, uint64_t bytes, VkBuffer *buf_out,
                       VkDeviceMemory *mem_out) {
    if (!pGetMemoryFdPropertiesKHR) return 0;
    VkExternalMemoryBufferCreateInfo ext;
    memset(&ext, 0, sizeof(ext));
    ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext.handleTypes = d->dma_buf
        ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
        : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkBufferCreateInfo bci;
    memset(&bci, 0, sizeof(bci));
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext = &ext;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf = VK_NULL_HANDLE;
    if (vkCreateBuffer(d->dev, &bci, NULL, &buf) != VK_SUCCESS) return 0;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(d->dev, buf, &req);
    VkMemoryFdPropertiesKHR fdp;
    memset(&fdp, 0, sizeof(fdp));
    fdp.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
    const VkExternalMemoryHandleTypeFlagBits ht = d->dma_buf
        ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
        : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    if (pGetMemoryFdPropertiesKHR(d->dev, ht, fd, &fdp) != VK_SUCCESS) {
        vkDestroyBuffer(d->dev, buf, NULL);
        return 0;
    }
    uint32_t bits = req.memoryTypeBits & fdp.memoryTypeBits;
    uint32_t mt = 0;
    /* Diagnostic override: DS4_MGPU_PROBE_IMPORT_MEMTYPE=<idx> forces the
     * imported buffer into a specific memory type (e.g. GTT instead of VRAM),
     * to tell a placement problem from a real import/P2P limitation. */
    const char *force_mt = getenv("DS4_MGPU_PROBE_IMPORT_MEMTYPE");
    if (force_mt && *force_mt && (bits & (1u << (uint32_t)atoi(force_mt)))) {
        mt = (uint32_t)atoi(force_mt);
    } else if (!find_mem_type(d, bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &mt) &&
        !find_mem_type(d, bits, 0, &mt)) {
        vkDestroyBuffer(d->dev, buf, NULL);
        return 0;
    }
    if (getenv("DS4_MGPU_PROBE_IMPORT_DEBUG")) {
        printf(LOG "  import fd=%d typeBits=0x%x chose mt=%u flags=0x%x heap=%u\n",
               fd, bits, mt, d->mem.memoryTypes[mt].propertyFlags,
               d->mem.memoryTypes[mt].heapIndex);
    }
    VkImportMemoryFdInfoKHR imp;
    memset(&imp, 0, sizeof(imp));
    imp.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    imp.handleType = ht;
    imp.fd = fd;
    VkMemoryDedicatedAllocateInfo ded;
    memset(&ded, 0, sizeof(ded));
    ded.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    ded.buffer = buf;
    imp.pNext = &ded;
    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = &imp;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(d->dev, &ai, NULL, &mem) != VK_SUCCESS) {
        ai.pNext = &imp;
        imp.pNext = NULL;
        if (vkAllocateMemory(d->dev, &ai, NULL, &mem) != VK_SUCCESS) {
            vkDestroyBuffer(d->dev, buf, NULL);
            return 0;
        }
    }
    if (vkBindBufferMemory(d->dev, buf, mem, 0) != VK_SUCCESS) {
        vkFreeMemory(d->dev, mem, NULL);
        vkDestroyBuffer(d->dev, buf, NULL);
        return 0;
    }
    *buf_out = buf;
    *mem_out = mem;
    return 1;
}

/* Run iters copies src->dst on the device and return total ms.
 * `warmup` copies are actually submitted and waited before the timed loop so
 * the GPU clocks/power state settle first: otherwise a freshly-ramped device
 * (e.g. AMD after runtime-suspend) reports a different value each run.
 * Returns -1 if any submit fails (so a failed P2P/copy is reported, instead of
 * silently producing an absurd bandwidth from a broken queue). */
static double copy_bench(const Dev *d, VkBuffer src, VkBuffer dst, uint64_t bytes,
                         int iters, int warmup) {
    for (int i = 0; i < warmup; i++) {
        vkResetCommandBuffer(d->cb, 0);
        VkCommandBufferBeginInfo bi;
        memset(&bi, 0, sizeof(bi));
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(d->cb, &bi);
        VkBufferCopy region = {0, 0, bytes};
        vkCmdCopyBuffer(d->cb, src, dst, 1, &region);
        vkEndCommandBuffer(d->cb);
        VkSubmitInfo si;
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d->cb;
        if (vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) return -1;
    }
    vkQueueWaitIdle(d->queue);
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        vkResetCommandBuffer(d->cb, 0);
        VkCommandBufferBeginInfo bi;
        memset(&bi, 0, sizeof(bi));
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(d->cb, &bi);
        VkBufferCopy region = {0, 0, bytes};
        vkCmdCopyBuffer(d->cb, src, dst, 1, &region);
        vkEndCommandBuffer(d->cb);
        VkSubmitInfo si;
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d->cb;
        if (vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) return -1;
    }
    vkQueueWaitIdle(d->queue);
    return now_ms() - t0;
}

/* Fill a buffer (forces allocation/population on the owning device). */
static void fill_buf(const Dev *d, VkBuffer buf, uint64_t bytes) {
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(d->cb, 0);
    vkBeginCommandBuffer(d->cb, &bi);
    vkCmdFillBuffer(d->cb, buf, 0, bytes, P2P_FILL);
    vkEndCommandBuffer(d->cb);
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &d->cb;
    vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(d->queue);
}

/* Clamp a requested transfer size: never below MIN_MB (a smaller buffer fits in
 * cache and would measure cache, not DRAM), and never above vram/8 (keep the
 * allocation sane). */
static uint64_t clamp_bytes_mb(uint64_t mb, uint64_t vram_bytes) {
    if (mb < MIN_MB) mb = MIN_MB;
    uint64_t bytes = mb * 1024ull * 1024ull;
    if (vram_bytes && bytes > vram_bytes / 8) bytes = vram_bytes / 8;
    return bytes;
}

/* Like copy_bench, but each timed copy is waited behind a fence, so a costly
 * first iteration (first touch of an imported buffer -> migration/fault) shows
 * up in *out_min / *out_max instead of being hidden by the batched queue.
 * Returns total timed ms, or -1 on submit failure. */
static double copy_bench_stats(const Dev *d, VkBuffer src, VkBuffer dst, uint64_t bytes,
                               int iters, int warmup, double *out_min, double *out_max) {
    for (int i = 0; i < warmup; i++) {
        vkResetCommandBuffer(d->cb, 0);
        VkCommandBufferBeginInfo bi;
        memset(&bi, 0, sizeof(bi));
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(d->cb, &bi);
        VkBufferCopy region = {0, 0, bytes};
        vkCmdCopyBuffer(d->cb, src, dst, 1, &region);
        vkEndCommandBuffer(d->cb);
        VkSubmitInfo si;
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d->cb;
        if (vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) return -1;
    }
    vkQueueWaitIdle(d->queue);

    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(d->dev, &fci, NULL, &fence) != VK_SUCCESS) return -1;

    double total = 0.0, mn = 1e30, mx = 0.0;
    for (int i = 0; i < iters; i++) {
        vkResetFences(d->dev, 1, &fence);
        vkResetCommandBuffer(d->cb, 0);
        VkCommandBufferBeginInfo bi;
        memset(&bi, 0, sizeof(bi));
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(d->cb, &bi);
        VkBufferCopy region = {0, 0, bytes};
        vkCmdCopyBuffer(d->cb, src, dst, 1, &region);
        vkEndCommandBuffer(d->cb);
        VkSubmitInfo si;
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d->cb;
        double t0 = now_ms();
        if (vkQueueSubmit(d->queue, 1, &si, fence) != VK_SUCCESS) {
            vkDestroyFence(d->dev, fence, NULL);
            return -1;
        }
        vkWaitForFences(d->dev, 1, &fence, VK_TRUE, UINT64_MAX);
        double dt = now_ms() - t0;
        total += dt;
        if (dt < mn) mn = dt;
        if (dt > mx) mx = dt;
    }
    vkDestroyFence(d->dev, fence, NULL);
    if (out_min) *out_min = mn;
    if (out_max) *out_max = mx;
    return total;
}

/* Verify that the first and last 4 KiB of a device buffer contain the marker
 * pattern written by fill_buf() on the source device.  Proves bytes really
 * moved, though not (by itself) that they crossed the PCIe link. */
static int verify_buf(const Dev *d, VkBuffer buf, uint64_t bytes) {
    uint64_t chunk = 4096;
    if (bytes < chunk * 2) chunk = bytes / 2;
    if (chunk < 4 || (bytes % 4) != 0) return 0;
    VkBuffer hb = VK_NULL_HANDLE;
    VkDeviceMemory hm = VK_NULL_HANDLE;
    if (!alloc_buf(d, chunk * 2, 0, 1,
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   &hb, &hm)) return 0;
    int ok = 0;
    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(d->cb, 0);
    if (vkBeginCommandBuffer(d->cb, &bi) == VK_SUCCESS) {
        VkBufferCopy c0 = {0, 0, chunk};
        VkBufferCopy c1 = {bytes - chunk, chunk, chunk};
        vkCmdCopyBuffer(d->cb, buf, hb, 1, &c0);
        vkCmdCopyBuffer(d->cb, buf, hb, 1, &c1);
        vkEndCommandBuffer(d->cb);
        VkSubmitInfo si;
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d->cb;
        if (vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS) {
            vkQueueWaitIdle(d->queue);
            void *p = NULL;
            if (vkMapMemory(d->dev, hm, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS) {
                const uint32_t *w = (const uint32_t *)p;
                uint64_t words = chunk / 4;
                ok = 1;
                for (uint64_t k = 0; k < words; k++) {
                    if (w[k] != P2P_FILL || w[words + k] != P2P_FILL) { ok = 0; break; }
                }
                vkUnmapMemory(d->dev, hm);
            }
        }
    }
    vkDestroyBuffer(d->dev, hb, NULL);
    vkFreeMemory(d->dev, hm, NULL);
    return ok;
}

/* ---- Pure read/write bandwidth benches ----
 *
 * read_bench: a compute kernel (read_spv.h) reads the whole working set and
 *   writes one uint per 256-thread group (~0.01% of the read traffic).
 *   Measures PURE read DRAM bandwidth.
 * fill_bench: vkCmdFillBuffer over the whole working set (copy engine,
 *   write-only).  Measures PURE write DRAM bandwidth.
 * Both use the same warmup-then-batched-timed structure as copy_bench so a
 * freshly-ramped device reports a stable value.  Return total timed ms,
 * or -1 on submit/init failure. */
#define RB_ROW_W    (1u << 20)   /* threads per dispatch row (== ROW_W in hlsl) */
/* uint4 elements each thread reads in the grid-stride loop.  Small values
 * launch huge numbers of short-lived threads and measure block-dispatch
 * throughput instead of DRAM (e.g. elem=8 -> ~117 GB/s on a 6900 XT); 256
 * keeps threads resident long enough to reach the DRAM bus (~500 GB/s).
 * Rounded up to a multiple of 8 (the kernel's inner unroll). */
static uint32_t rb_elem(void) {
    const char *e = getenv("DS4_MGPU_READ_ELEM");
    uint32_t v = e ? (uint32_t)atoi(e) : 256u;
    if (v < 8u) v = 8u;
    return (v / 8u) * 8u;
}

/* Elements per thread for the pure-WRITE kernel.  Stores are posted (no
 * latency to hide) and need enough resident threads to keep the write path
 * busy; 128 is the stable sweet spot on a 6900 XT (~482 GB/s) while 160..256
 * make it erratic (~270-400).  Rounded up to a multiple of 8. */
static uint32_t wb_elem(void) {
    const char *e = getenv("DS4_MGPU_WRITE_ELEM");
    uint32_t v = e ? (uint32_t)atoi(e) : 128u;
    if (v < 8u) v = 8u;
    return (v / 8u) * 8u;
}

static int init_read_bench(Dev *d) {
    VkShaderModuleCreateInfo mci;
    memset(&mci, 0, sizeof(mci));
    mci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mci.codeSize = read_bench_spv_len;
    mci.pCode = (const uint32_t *)read_bench_spv;
    if (vkCreateShaderModule(d->dev, &mci, NULL, &d->rb_mod) != VK_SUCCESS)
        return 0;
    VkDescriptorSetLayoutBinding bl[2];
    memset(&bl[0], 0, sizeof(bl[0]));
    bl[0].binding = 0;
    bl[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bl[0].descriptorCount = 1;
    bl[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    memset(&bl[1], 0, sizeof(bl[1]));
    bl[1].binding = 2;   /* u0 maps to binding 2 (dxc -fvk-u-shift 2 0) */
    bl[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bl[1].descriptorCount = 1;
    bl[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dli;
    memset(&dli, 0, sizeof(dli));
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 2;
    dli.pBindings = bl;
    if (vkCreateDescriptorSetLayout(d->dev, &dli, NULL, &d->rb_dsl) != VK_SUCCESS)
        return 0;
    VkPipelineLayoutCreateInfo pli;
    memset(&pli, 0, sizeof(pli));
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &d->rb_dsl;
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(d->dev, &pli, NULL, &d->rb_pl) != VK_SUCCESS)
        return 0;
    VkPipelineShaderStageCreateInfo ssc;
    memset(&ssc, 0, sizeof(ssc));
    ssc.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssc.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssc.module = d->rb_mod;
    ssc.pName = "main";
    VkComputePipelineCreateInfo cpc;
    memset(&cpc, 0, sizeof(cpc));
    cpc.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpc.stage = ssc;
    cpc.layout = d->rb_pl;
    if (vkCreateComputePipelines(d->dev, VK_NULL_HANDLE, 1, &cpc, NULL,
                                 &d->rb_pipe) != VK_SUCCESS)
        return 0;
    VkDescriptorPoolSize psize;
    memset(&psize, 0, sizeof(psize));
    psize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psize.descriptorCount = 2;
    VkDescriptorPoolCreateInfo pdc;
    memset(&pdc, 0, sizeof(pdc));
    pdc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pdc.maxSets = 1;
    pdc.poolSizeCount = 1;
    pdc.pPoolSizes = &psize;
    if (vkCreateDescriptorPool(d->dev, &pdc, NULL, &d->rb_pool) != VK_SUCCESS)
        return 0;
    VkDescriptorSetAllocateInfo sal;
    memset(&sal, 0, sizeof(sal));
    sal.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    sal.descriptorPool = d->rb_pool;
    sal.descriptorSetCount = 1;
    sal.pSetLayouts = &d->rb_dsl;
    if (vkAllocateDescriptorSets(d->dev, &sal, &d->rb_set) != VK_SUCCESS)
        return 0;
    return 1;
}

static double read_bench(const Dev *d, uint64_t bytes, int iters, int warmup) {
    if (!d->rb_ok) return -1;
    uint32_t elem = rb_elem();
    uint64_t total_elems = (bytes / 16u / elem) * elem;  /* multiple of elem */
    if (total_elems < (uint64_t)elem * 256u) return -1;
    uint64_t rbytes = total_elems * 16u;
    uint64_t ttotal = total_elems / elem;                /* total threads */
    if (ttotal == 0 || ttotal > 0xffffffffull) return -1;
    VkBuffer buf = VK_NULL_HANDLE, obuf = VK_NULL_HANDLE;
    VkDeviceMemory m = VK_NULL_HANDLE, om = VK_NULL_HANDLE;
    const VkBufferUsageFlags sto = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    uint64_t ogbytes = (ttotal / 256u + 1u) * 4u;        /* one uint per group */
    if (!alloc_buf(d, rbytes, 0, 0, sto, &buf, &m) ||
        !alloc_buf(d, ogbytes, 0, 0, sto, &obuf, &om))
        return -1;
    VkDescriptorBufferInfo bi0, bi1;
    memset(&bi0, 0, sizeof(bi0));
    bi0.buffer = buf;
    bi0.range = rbytes;
    memset(&bi1, 0, sizeof(bi1));
    bi1.buffer = obuf;
    bi1.range = ogbytes;
    VkWriteDescriptorSet ws[2];
    memset(&ws[0], 0, sizeof(ws[0]));
    ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[0].dstSet = d->rb_set;
    ws[0].dstBinding = 0;
    ws[0].descriptorCount = 1;
    ws[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ws[0].pBufferInfo = &bi0;
    memset(&ws[1], 0, sizeof(ws[1]));
    ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[1].dstSet = d->rb_set;
    ws[1].dstBinding = 2;
    ws[1].descriptorCount = 1;
    ws[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ws[1].pBufferInfo = &bi1;
    vkUpdateDescriptorSets(d->dev, 2, ws, 0, NULL);
    struct { uint32_t n_total; uint32_t elem; uint32_t stride; uint32_t pad; } pc =
        { (uint32_t)ttotal, elem, (uint32_t)ttotal, 0 };
    uint32_t dx = (uint32_t)(ttotal < RB_ROW_W ? ttotal : RB_ROW_W);
    uint32_t dy = (uint32_t)((ttotal + RB_ROW_W - 1u) / RB_ROW_W);
    double total_ms = 0.0;
    for (int i = 0; i < warmup; i++) {
        vkResetCommandBuffer(d->cb, 0);
        VkCommandBufferBeginInfo bi;
        memset(&bi, 0, sizeof(bi));
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(d->cb, &bi) != VK_SUCCESS) goto fail;
        vkCmdBindPipeline(d->cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->rb_pipe);
        vkCmdBindDescriptorSets(d->cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->rb_pl,
                                0, 1, &d->rb_set, 0, NULL);
        vkCmdPushConstants(d->cb, d->rb_pl, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(d->cb, dx, dy, 1);
        vkEndCommandBuffer(d->cb);
        VkSubmitInfo si;
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d->cb;
        if (vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
            goto fail;
    }
    vkQueueWaitIdle(d->queue);
    for (int i = 0; i < iters; i++) {
        double t0 = now_ms();
        vkResetCommandBuffer(d->cb, 0);
        VkCommandBufferBeginInfo bi;
        memset(&bi, 0, sizeof(bi));
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(d->cb, &bi) != VK_SUCCESS) goto fail;
        vkCmdBindPipeline(d->cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->rb_pipe);
        vkCmdBindDescriptorSets(d->cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->rb_pl,
                                0, 1, &d->rb_set, 0, NULL);
        vkCmdPushConstants(d->cb, d->rb_pl, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(d->cb, dx, dy, 1);
        vkEndCommandBuffer(d->cb);
        VkSubmitInfo si;
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d->cb;
        if (vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
            goto fail;
        if (i == iters - 1) vkQueueWaitIdle(d->queue);
        total_ms += now_ms() - t0;
    }
    vkDestroyBuffer(d->dev, obuf, NULL);
    vkFreeMemory(d->dev, om, NULL);
    vkDestroyBuffer(d->dev, buf, NULL);
    vkFreeMemory(d->dev, m, NULL);
    return total_ms;
fail:
    if (obuf) vkDestroyBuffer(d->dev, obuf, NULL);
    if (om) vkFreeMemory(d->dev, om, NULL);
    if (buf) vkDestroyBuffer(d->dev, buf, NULL);
    if (m) vkFreeMemory(d->dev, m, NULL);
    return -1;
}

/* Pure-write bandwidth: same grid-stride shape as read_bench, but the kernel
 * (write_spv.h) only stores.  Shares the read bench's descriptor set, layout
 * and pool (binding 0 is bound but unused; the kernel writes binding 2), so
 * it requires rb_ok. */
static int init_write_bench(Dev *d) {
    if (!d->rb_ok) return 0;
    VkShaderModuleCreateInfo mci;
    memset(&mci, 0, sizeof(mci));
    mci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mci.codeSize = write_bench_spv_len;
    mci.pCode = (const uint32_t *)write_bench_spv;
    if (vkCreateShaderModule(d->dev, &mci, NULL, &d->wb_mod) != VK_SUCCESS)
        return 0;
    VkPipelineShaderStageCreateInfo ssc;
    memset(&ssc, 0, sizeof(ssc));
    ssc.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssc.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssc.module = d->wb_mod;
    ssc.pName = "main";
    VkComputePipelineCreateInfo cpc;
    memset(&cpc, 0, sizeof(cpc));
    cpc.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpc.stage = ssc;
    cpc.layout = d->rb_pl;
    if (vkCreateComputePipelines(d->dev, VK_NULL_HANDLE, 1, &cpc, NULL,
                                 &d->wb_pipe) != VK_SUCCESS)
        return 0;
    return 1;
}

static double write_bench(const Dev *d, uint64_t bytes, int iters, int warmup) {
    if (!d->wb_ok) return -1;
    uint32_t elem = wb_elem();
    uint64_t total_elems = (bytes / 16u / elem) * elem;
    if (total_elems < (uint64_t)elem * 256u) return -1;
    uint64_t wbytes = total_elems * 16u;
    uint64_t ttotal = total_elems / elem;
    if (ttotal == 0 || ttotal > 0xffffffffull) return -1;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory m = VK_NULL_HANDLE;
    if (!alloc_buf(d, wbytes, 0, 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf, &m))
        return -1;
    VkDescriptorBufferInfo dbi;
    memset(&dbi, 0, sizeof(dbi));
    dbi.buffer = buf;
    dbi.range = wbytes;
    VkWriteDescriptorSet ws[2];
    memset(&ws[0], 0, sizeof(ws[0]));
    ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[0].dstSet = d->rb_set;
    ws[0].dstBinding = 0;
    ws[0].descriptorCount = 1;
    ws[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ws[0].pBufferInfo = &dbi;
    memset(&ws[1], 0, sizeof(ws[1]));
    ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[1].dstSet = d->rb_set;
    ws[1].dstBinding = 2;
    ws[1].descriptorCount = 1;
    ws[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ws[1].pBufferInfo = &dbi;
    vkUpdateDescriptorSets(d->dev, 2, ws, 0, NULL);
    struct { uint32_t n_total; uint32_t elem; uint32_t stride; uint32_t seed; } pc =
        { (uint32_t)ttotal, elem, (uint32_t)ttotal, 1u };
    uint32_t dx = (uint32_t)(ttotal < RB_ROW_W ? ttotal : RB_ROW_W);
    uint32_t dy = (uint32_t)((ttotal + RB_ROW_W - 1u) / RB_ROW_W);
    double total_ms = 0.0;
    for (int i = 0; i < warmup; i++) {
        vkResetCommandBuffer(d->cb, 0);
        VkCommandBufferBeginInfo cbi;
        memset(&cbi, 0, sizeof(cbi));
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(d->cb, &cbi) != VK_SUCCESS) goto fail;
        vkCmdBindPipeline(d->cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->wb_pipe);
        vkCmdBindDescriptorSets(d->cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->rb_pl,
                                0, 1, &d->rb_set, 0, NULL);
        vkCmdPushConstants(d->cb, d->rb_pl, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(d->cb, dx, dy, 1);
        vkEndCommandBuffer(d->cb);
        VkSubmitInfo si;
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d->cb;
        if (vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
            goto fail;
    }
    vkQueueWaitIdle(d->queue);
    for (int i = 0; i < iters; i++) {
        double t0 = now_ms();
        vkResetCommandBuffer(d->cb, 0);
        VkCommandBufferBeginInfo cbi;
        memset(&cbi, 0, sizeof(cbi));
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(d->cb, &cbi) != VK_SUCCESS) goto fail;
        vkCmdBindPipeline(d->cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->wb_pipe);
        vkCmdBindDescriptorSets(d->cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->rb_pl,
                                0, 1, &d->rb_set, 0, NULL);
        vkCmdPushConstants(d->cb, d->rb_pl, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(d->cb, dx, dy, 1);
        vkEndCommandBuffer(d->cb);
        VkSubmitInfo si;
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d->cb;
        if (vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
            goto fail;
        if (i == iters - 1) vkQueueWaitIdle(d->queue);
        total_ms += now_ms() - t0;
    }
    vkDestroyBuffer(d->dev, buf, NULL);
    vkFreeMemory(d->dev, m, NULL);
    return total_ms;
fail:
    if (buf) vkDestroyBuffer(d->dev, buf, NULL);
    if (m) vkFreeMemory(d->dev, m, NULL);
    return -1;
}

static double fill_bench(const Dev *d, VkBuffer dst, uint64_t bytes,
                         int iters, int warmup) {
    for (int i = 0; i < warmup; i++) {
        vkResetCommandBuffer(d->cb, 0);
        VkCommandBufferBeginInfo bi;
        memset(&bi, 0, sizeof(bi));
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(d->cb, &bi) != VK_SUCCESS) return -1;
        vkCmdFillBuffer(d->cb, dst, 0, bytes, P2P_FILL);
        vkEndCommandBuffer(d->cb);
        VkSubmitInfo si;
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d->cb;
        if (vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
            return -1;
    }
    vkQueueWaitIdle(d->queue);
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        vkResetCommandBuffer(d->cb, 0);
        VkCommandBufferBeginInfo bi;
        memset(&bi, 0, sizeof(bi));
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(d->cb, &bi) != VK_SUCCESS) return -1;
        vkCmdFillBuffer(d->cb, dst, 0, bytes, P2P_FILL);
        vkEndCommandBuffer(d->cb);
        VkSubmitInfo si;
        memset(&si, 0, sizeof(si));
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &d->cb;
        if (vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
            return -1;
    }
    vkQueueWaitIdle(d->queue);
    return now_ms() - t0;
}

/* Print the P2P matrix exactly as the probe measures it.  Shared by the test
 * section and the summary so the two can never diverge. */
static void print_p2p_matrix(const Dev *devs, int nd, const double p2p[16][16]) {
    printf(LOG "=== P2P transfer matrix (rows=source, cols=dest, GB/s) ===\n");
    printf(LOG "     ");
    for (int j = 0; j < nd; j++) printf("%7d", j);
    printf("\n");
    for (int i = 0; i < nd; i++) {
        printf(LOG "%3d: ", i);
        for (int j = 0; j < nd; j++) {
            if (i == j || !devs[i].ok || !devs[j].ok) { printf("      -"); continue; }
            double v = p2p[i][j];
            if (v > 0)                     printf("%7.1f", v);
            else if (v == (double)P2P_N_A)   printf("   n/a ");
            else if (v == (double)P2P_NOFD)  printf("  nofd ");
            else if (v == (double)P2P_ALLOC) printf("  alloc");
            else                             printf("  fail ");
        }
        printf("\n");
    }
    printf(LOG "n/a = external memory dma-buf unsupported; fail = import/copy/verify failed; "
           "- = same/absent device\n");
}

/* Effective "host-bounce" throughput A -> host -> B, derived from the two
 * measured one-way host<->device rates: A reads its device memory and writes
 * host (device->host), then B reads host and writes its device memory
 * (host->device).  The two PCIe legs are serial, so the combined rate is the
 * harmonic mean 1/(1/rA + 1/rB).  This is what ds4 falls back to when
 * cross-vendor P2P is unavailable.  A direct cross-device measurement is
 * avoided because a P2P failure can leave the device queue in a bad state
 * (silently yielding ~0ms legs). */
static double host_bounce_gbs(double d2h_rate, double h2d_rate) {
    if (d2h_rate <= 0 || h2d_rate <= 0) return -1;
    return 1.0 / (1.0 / d2h_rate + 1.0 / h2d_rate);
}

static void print_mem_heaps(const Dev *d) {
    for (uint32_t i = 0; i < d->mem.memoryHeapCount; i++) {
        int local = (d->mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        printf(LOG "  heap[%u] %s %.2f GiB\n", i, local ? "DEVICE_LOCAL" : "host      ",
               (double)d->mem.memoryHeaps[i].size / (1024.0 * 1024.0 * 1024.0));
    }
}

static int init_device(Dev *d) {
    uint32_t extn = 0;
    vkEnumerateDeviceExtensionProperties(d->phys, NULL, &extn, NULL);
    VkExtensionProperties *exts = (VkExtensionProperties *)malloc(sizeof(*exts) * (extn ? extn : 1));
    vkEnumerateDeviceExtensionProperties(d->phys, NULL, &extn, exts);
    d->ext_mem_fd = has_ext(exts, extn, "VK_KHR_external_memory_fd");
    d->ext_sem_fd = has_ext(exts, extn, "VK_KHR_external_semaphore_fd");
    d->dma_buf = has_ext(exts, extn, "VK_EXT_external_memory_dma_buf");
    d->mem_budget = has_ext(exts, extn, "VK_EXT_memory_budget");
    free(exts);

    /* External-buffer support for the device. */
    if (pGetPhysicalDeviceExternalBufferProperties) {
        VkPhysicalDeviceExternalBufferInfo bi;
        memset(&bi, 0, sizeof(bi));
        bi.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
        bi.handleType = d->dma_buf ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
                                   : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkExternalBufferProperties bp;
        memset(&bp, 0, sizeof(bp));
        bp.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
        pGetPhysicalDeviceExternalBufferProperties(d->phys, &bi, &bp);
        printf(LOG "  external buffer: importable=%d exportable=%d handleType=%s\n",
               (bp.externalMemoryProperties.externalMemoryFeatures &
                VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0,
               (bp.externalMemoryProperties.externalMemoryFeatures &
                VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0,
               d->dma_buf ? "DMA_BUF" : "OPAQUE_FD");
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci;
    memset(&qci, 0, sizeof(qci));
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = d->qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    const char *dexts[4];
    uint32_t nd = 0;
    if (d->ext_mem_fd) dexts[nd++] = "VK_KHR_external_memory_fd";
    if (d->ext_sem_fd) dexts[nd++] = "VK_KHR_external_semaphore_fd";
    if (d->dma_buf) dexts[nd++] = "VK_EXT_external_memory_dma_buf";

    VkDeviceCreateInfo dci;
    memset(&dci, 0, sizeof(dci));
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nd;
    dci.ppEnabledExtensionNames = dexts;
    if (vkCreateDevice(d->phys, &dci, NULL, &d->dev) != VK_SUCCESS) {
        printf(LOG "  vkCreateDevice failed (can't measure)\n");
        return 0;
    }
    vkGetDeviceQueue(d->dev, d->qfam, 0, &d->queue);
    VkCommandPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = d->qfam;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(d->dev, &pci, NULL, &d->pool) != VK_SUCCESS) return 0;
    VkCommandBufferAllocateInfo cba;
    memset(&cba, 0, sizeof(cba));
    cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = d->pool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(d->dev, &cba, &d->cb) != VK_SUCCESS) return 0;
    return 1;
}

/* PCI bus/device/function (needs VK_EXT_pci_bus_info) so the Vulkan index
 * can be mapped to the real slot (lspci/sysfs link width). */
static void query_pci(VkPhysicalDevice phys, uint32_t *dom, uint32_t *bus,
                      uint32_t *dev, uint32_t *fn) {
    *dom = *bus = *dev = *fn = 0;
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(phys, NULL, &n, NULL);
    VkExtensionProperties *e = (VkExtensionProperties *)malloc(sizeof(*e) * (n ? n : 1));
    vkEnumerateDeviceExtensionProperties(phys, NULL, &n, e);
    const int ok = has_ext(e, n, "VK_EXT_pci_bus_info");
    free(e);
    if (!ok) return;
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;
    VkPhysicalDeviceProperties2 p2;
    memset(&p2, 0, sizeof(p2));
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &pci;
    vkGetPhysicalDeviceProperties2(phys, &p2);
    *dom = pci.pciDomain;
    *bus = pci.pciBus;
    *dev = pci.pciDevice;
    *fn = pci.pciFunction;
}

int main(int argc, char **argv) {
    uint64_t mb = 256;
    int iters = 5;
    int warmup = 50;
    const char *devfilter = NULL;
    int p2p_fresh = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mb") && i + 1 < argc) mb = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--devices") && i + 1 < argc) devfilter = argv[++i];
        else if (!strcmp(argv[i], "--p2p-fresh")) p2p_fresh = 1;
    }
    if (mb < MIN_MB) {
        printf(LOG "--mb %llu < %u MiB: clamped (a smaller buffer fits Infinity "
               "Cache and would measure cache, not DRAM)\n",
               (unsigned long long)mb, MIN_MB);
    }

    VkApplicationInfo app;
    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "mgpu-probe";
    app.apiVersion = VK_API_VERSION_1_1;

    /* Enable only the instance extensions actually present (portability). */
    uint32_t iextn = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &iextn, NULL);
    VkExtensionProperties *iextp =
        (VkExtensionProperties *)malloc(sizeof(*iextp) * (iextn ? iextn : 1));
    vkEnumerateInstanceExtensionProperties(NULL, &iextn, iextp);
    static const char *want[] = {
        "VK_KHR_external_memory_capabilities",
        "VK_KHR_external_semaphore_capabilities",
        "VK_KHR_get_physical_device_properties2",
    };
    const char *iexts[4];
    uint32_t ni = 0;
    for (unsigned w = 0; w < sizeof(want) / sizeof(want[0]); w++)
        if (has_ext(iextp, iextn, want[w])) iexts[ni++] = want[w];
    free(iextp);

    VkInstanceCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = ni;
    ici.ppEnabledExtensionNames = iexts;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
        printf(LOG "vkCreateInstance failed\n");
        return 1;
    }
    pGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetInstanceProcAddr(inst, "vkGetMemoryFdKHR");
    pGetMemoryFdPropertiesKHR =
        (PFN_vkGetMemoryFdPropertiesKHR)vkGetInstanceProcAddr(inst, "vkGetMemoryFdPropertiesKHR");
    pGetPhysicalDeviceExternalBufferProperties =
        (PFN_vkGetPhysicalDeviceExternalBufferProperties)
            vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceExternalBufferProperties");

    uint32_t np = 0;
    vkEnumeratePhysicalDevices(inst, &np, NULL);
    VkPhysicalDevice *phys = (VkPhysicalDevice *)malloc(sizeof(*phys) * (np ? np : 1));
    vkEnumeratePhysicalDevices(inst, &np, phys);
    printf(LOG "found %u physical device(s)\n", np);

    Dev devs[16];
    memset(devs, 0, sizeof(devs));
    double dev_local_gbs[16] = {0};     /* device-local copy (r+w) GB/s */
    double dev_read_gbs[16] = {0};      /* pure-read GB/s (compute kernel) */
    double dev_write_gbs[16] = {0};     /* pure-write GB/s (compute kernel) */
    double dev_fill_gbs[16] = {0};      /* pure-write GB/s (vkCmdFillBuffer) */
    double host_d2h[16] = {0};          /* device->host one-way GB/s */
    double host_h2d[16] = {0};          /* host->device one-way GB/s */
    double p2p_gbs[16][16];             /* >0 measured; sentinel codes otherwise; 0 same */
    char p2p_note[16][16][160];
    double p2p_min[16][16], p2p_max[16][16];
    memset(p2p_note, 0, sizeof(p2p_note));
    int nd = 0;
    const int include_cpu = getenv("DS4_MGPU_PROBE_CPU") != NULL;
    const int copy_on_src = getenv("DS4_MGPU_PROBE_COPY_ON_SRC") != NULL;
    for (uint32_t i = 0; i < np && nd < 16; i++) {
        VkPhysicalDeviceProperties pprops;
        vkGetPhysicalDeviceProperties(phys[i], &pprops);
        if (!include_cpu && pprops.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
            continue;
        if (devfilter && *devfilter) {
            char buf[64];
            snprintf(buf, sizeof(buf), ",%u,", i);
            char tmp[80];
            snprintf(tmp, sizeof(tmp), ",%s,", devfilter);
            if (!strstr(tmp, buf)) continue;
        }
        Dev *d = &devs[nd];
        d->phys = phys[i];
        vkGetPhysicalDeviceProperties(d->phys, &d->props);
        vkGetPhysicalDeviceMemoryProperties(d->phys, &d->mem);
        d->vram_bytes = 0;
        for (uint32_t h = 0; h < d->mem.memoryHeapCount; h++)
            if (d->mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                d->vram_bytes += d->mem.memoryHeaps[h].size;
        /* pick a compute queue family */
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d->phys, &qn, NULL);
        VkQueueFamilyProperties *q = (VkQueueFamilyProperties *)malloc(sizeof(*q) * (qn ? qn : 1));
        vkGetPhysicalDeviceQueueFamilyProperties(d->phys, &qn, q);
        d->qfam = 0;
        for (uint32_t k = 0; k < qn; k++)
            if (q[k].queueFlags & VK_QUEUE_COMPUTE_BIT) { d->qfam = k; break; }
        free(q);
        uint32_t pd = 0, pb = 0, pdev = 0, pfn = 0;
        query_pci(d->phys, &pd, &pb, &pdev, &pfn);
        d->pci_dom = pd; d->pci_bus = pb; d->pci_dev = pdev; d->pci_fn = pfn;
        printf("\n" LOG "device[%d] (plat=%u) bdf=%04x:%02x:%02x.%x: %s\n", nd, i,
               pd, pb, pdev, pfn, d->props.deviceName);
        printf(LOG "  type=%d api=%u.%u.%u vram=%.2f GiB\n", d->props.deviceType,
               VK_VERSION_MAJOR(d->props.apiVersion), VK_VERSION_MINOR(d->props.apiVersion),
               VK_VERSION_PATCH(d->props.apiVersion),
               (double)d->vram_bytes / (1024.0 * 1024.0 * 1024.0));
        print_mem_heaps(d);
        d->ok = init_device(d);
        d->rb_ok = d->ok && init_read_bench(d);
        d->wb_ok = d->rb_ok && init_write_bench(d);
        nd++;
    }

    /* Per-device bandwidth + VRAM budget.
     * Bandwidth semantics: the DRAM bus has ONE total rate (spec ~512 GB/s on
     * an RX 6900 XT), shared by reads and writes.  A D2D copy consumes 1 bus
     * byte read + 1 written per copied byte, so its r+w sum is bounded by the
     * bus (~87% is normal DMA efficiency) and its per-direction "r="/"w=" is
     * ~half the bus -- do NOT compare that to the spec number.  A pure write
     * (fill) or a pure read alone can approach the full bus rate. */
    printf(LOG "note: copy r+w is bounded by the TOTAL DRAM bus (read+write share it);\n");
    printf(LOG "      compare the bus to the spec via pure read/pure write or copy r+w, not copy r=/w=\n");
    /* Per-device bandwidth + VRAM budget. */
    for (int i = 0; i < nd; i++) {
        Dev *d = &devs[i];
        if (!d->ok) { printf("\n" LOG "device[%d]: no device\n", i); continue; }
        uint64_t bytes = clamp_bytes_mb(mb, d->vram_bytes);
        VkBuffer a = VK_NULL_HANDLE, b = VK_NULL_HANDLE;
        VkDeviceMemory ma = VK_NULL_HANDLE, mbh = VK_NULL_HANDLE;
        VkBuffer ha = VK_NULL_HANDLE;
        VkDeviceMemory mha = VK_NULL_HANDLE;
        const VkBufferUsageFlags xfer =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (!alloc_buf(d, bytes, 0, 0, xfer, &a, &ma) ||
            !alloc_buf(d, bytes, 0, 0, xfer, &b, &mbh)) {
            printf("\n" LOG "device[%d]: device-local alloc failed\n", i);
            continue;
        }
        double ms = copy_bench(d, a, b, bytes, iters, warmup);
        double gbs = ms < 0 ? 0 : (double)bytes * 2.0 * iters / (ms * 1.0e6);
        dev_local_gbs[i] = gbs;
        printf("\n" LOG "device[%d] %s: VRAM %.2f GiB, copy r=%.1f GB/s w=%.1f GB/s (r+w %.1f, %llu MiB x%d)\n",
               i, d->props.deviceName,
               (double)d->vram_bytes / (1024.0 * 1024.0 * 1024.0),
               gbs / 2.0, gbs / 2.0, gbs, (unsigned long long)(bytes >> 20), iters);
        {
            double rms = read_bench(d, bytes, iters, warmup);
            double wms = write_bench(d, bytes, iters, warmup);
            double fms = fill_bench(d, b, bytes, iters, warmup);
            dev_read_gbs[i] = rms < 0 ? 0 : (double)bytes * iters / (rms * 1.0e6);
            dev_write_gbs[i] = wms < 0 ? 0 : (double)bytes * iters / (wms * 1.0e6);
            dev_fill_gbs[i] = fms < 0 ? 0 : (double)bytes * iters / (fms * 1.0e6);
            printf(LOG "  pure read (shader) %.1f GB/s | pure write (shader) %.1f GB/s "
                   "| fill (DMA) %.1f GB/s\n",
                   dev_read_gbs[i], dev_write_gbs[i], dev_fill_gbs[i]);
        }
        if (alloc_buf(d, bytes, 0, 1, xfer, &ha, &mha)) {
            double t1 = copy_bench(d, a, ha, bytes, iters, warmup);   /* device -> host */
            double t2 = copy_bench(d, ha, a, bytes, iters, warmup);   /* host -> device */
            host_d2h[i] = t1 < 0 ? 0 : (double)bytes * iters / (t1 * 1.0e6);
            host_h2d[i] = t2 < 0 ? 0 : (double)bytes * iters / (t2 * 1.0e6);
            printf(LOG "  host<->device: device->host %.1f GB/s, host->device %.1f GB/s\n",
                   host_d2h[i], host_h2d[i]);
            vkDestroyBuffer(d->dev, ha, NULL);
            vkFreeMemory(d->dev, mha, NULL);
        }
        vkDestroyBuffer(d->dev, a, NULL);
        vkFreeMemory(d->dev, ma, NULL);
        vkDestroyBuffer(d->dev, b, NULL);
        vkFreeMemory(d->dev, mbh, NULL);
        if (d->wb_ok) {
            vkDestroyPipeline(d->dev, d->wb_pipe, NULL);
            vkDestroyShaderModule(d->dev, d->wb_mod, NULL);
            d->wb_ok = 0;
        }
        if (d->rb_ok) {
            vkDestroyPipeline(d->dev, d->rb_pipe, NULL);
            vkDestroyPipelineLayout(d->dev, d->rb_pl, NULL);
            vkDestroyDescriptorSetLayout(d->dev, d->rb_dsl, NULL);
            vkDestroyDescriptorPool(d->dev, d->rb_pool, NULL);
            vkDestroyShaderModule(d->dev, d->rb_mod, NULL);
            d->rb_ok = 0;
        }
    }

    /* P2P matrix.  Measure every ordered pair, content-verify the copy, and
     * compare against the host-bounce estimate / each endpoint's PCIe rate so
     * an import/migration artifact is flagged instead of trusted. */
    for (int i = 0; i < nd; i++)
        for (int j = 0; j < nd; j++) p2p_gbs[i][j] = (i == j) ? 0 : P2P_FAIL;
    printf("\n" LOG "measuring %d P2P pair(s)...\n", nd * (nd - 1));
    if (copy_on_src) printf(LOG "  (copy-on-src workaround enabled)\n");
    for (int i = 0; i < nd; i++) {
        for (int j = 0; j < nd; j++) {
            if (i == j || !devs[i].ok || !devs[j].ok) continue;
            Dev *A = &devs[i];
            Dev *B = &devs[j];
            if (!A->ext_mem_fd || !A->dma_buf || !pGetMemoryFdKHR) {
                p2p_gbs[i][j] = P2P_N_A;
                continue;
            }
            uint64_t bytes = clamp_bytes_mb(mb, A->vram_bytes);
            VkBuffer dstB = VK_NULL_HANDLE;
            VkDeviceMemory mDstB = VK_NULL_HANDLE;
            /* DS4_MGPU_PROBE_COPY_ON_SRC: own the shared buffer on B, import it
             * on A and run the copy on A (source).  Workaround for drivers that
             * cannot provide an sg_table for their own export (NVIDIA bug: an
             * NVIDIA-exported VRAM dma-buf imported on AMD fails at CS submit). */
            if (!alloc_buf(B, bytes, copy_on_src ? 1 : 0, 0,
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           &dstB, &mDstB)) {
                p2p_gbs[i][j] = P2P_ALLOC;
                continue;
            }
            VkBuffer impDstA = VK_NULL_HANDLE;
            VkDeviceMemory mImpDstA = VK_NULL_HANDLE;
            double total_ms = 0.0, mn = 1e30, mx = 0.0;
            int failed = 0, verified = 0, reason = P2P_FAIL;
            if (copy_on_src) {
                int fd_dst = -1;
                VkMemoryGetFdInfoKHR gfd;
                memset(&gfd, 0, sizeof(gfd));
                gfd.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
                gfd.memory = mDstB;
                gfd.handleType = B->dma_buf
                    ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
                    : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
                if (pGetMemoryFdKHR(B->dev, &gfd, &fd_dst) != VK_SUCCESS) {
                    reason = P2P_NOFD;
                    failed = 1;
                } else if (!import_buf2(A, fd_dst, bytes, &impDstA, &mImpDstA)) {
                    reason = P2P_ALLOC;
                    failed = 1;
                }
            }
            /* --p2p-fresh: a brand-new exported/imported source per copy, so a
             * warmup migration of the imported BO cannot inflate the result. */
            int reps = p2p_fresh ? (iters + warmup) : 1;
            for (int rep = 0; rep < reps && !failed; rep++) {
                int do_timed = !p2p_fresh || rep >= warmup;
                VkBuffer srcA = VK_NULL_HANDLE;
                VkDeviceMemory mSrcA = VK_NULL_HANDLE;
                if (!alloc_buf(A, bytes, copy_on_src ? 0 : 1, 0,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               &srcA, &mSrcA)) {
                    reason = P2P_ALLOC; failed = 1; break;
                }
                fill_buf(A, srcA, bytes);
                VkBuffer impB = VK_NULL_HANDLE;
                VkDeviceMemory mImpB = VK_NULL_HANDLE;
                if (!copy_on_src) {
                    int fd = -1;
                    VkMemoryGetFdInfoKHR gfi;
                    memset(&gfi, 0, sizeof(gfi));
                    gfi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
                    gfi.memory = mSrcA;
                    gfi.handleType = A->dma_buf
                        ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
                        : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
                    if (pGetMemoryFdKHR(A->dev, &gfi, &fd) != VK_SUCCESS) {
                        reason = P2P_NOFD;
                        vkDestroyBuffer(A->dev, srcA, NULL);
                        vkFreeMemory(A->dev, mSrcA, NULL);
                        failed = 1;
                        break;
                    }
                    if (!import_buf2(B, fd, bytes, &impB, &mImpB)) {
                        reason = P2P_ALLOC;
                        vkDestroyBuffer(A->dev, srcA, NULL);
                        vkFreeMemory(A->dev, mSrcA, NULL);
                        failed = 1;
                        break;
                    }
                }
                double tmn = 0.0, tmx = 0.0;
                double ms = copy_on_src
                    ? copy_bench_stats(A, srcA, impDstA, bytes,
                                       p2p_fresh ? 1 : iters, p2p_fresh ? 0 : warmup, &tmn, &tmx)
                    : copy_bench_stats(B, impB, dstB, bytes,
                                       p2p_fresh ? 1 : iters, p2p_fresh ? 0 : warmup, &tmn, &tmx);
                if (ms < 0) {
                    failed = 1;
                } else if (do_timed) {
                    total_ms += ms;
                    if (tmn < mn) mn = tmn;
                    if (tmx > mx) mx = tmx;
                }
                verified = verify_buf(B, dstB, bytes);
                if (impB) { vkDestroyBuffer(B->dev, impB, NULL); vkFreeMemory(B->dev, mImpB, NULL); }
                vkDestroyBuffer(A->dev, srcA, NULL);
                vkFreeMemory(A->dev, mSrcA, NULL);
            }
            if (impDstA) { vkDestroyBuffer(A->dev, impDstA, NULL); vkFreeMemory(A->dev, mImpDstA, NULL); }
            if (!failed && verified && total_ms > 0) {
                double gbs = (double)bytes * iters / (total_ms * 1.0e6);
                p2p_gbs[i][j] = gbs;
                p2p_min[i][j] = mn > 0 ? (double)bytes / (mn * 1.0e6) : 0;
                p2p_max[i][j] = mx > 0 ? (double)bytes / (mx * 1.0e6) : 0;
                double hb = host_bounce_gbs(host_d2h[i], host_h2d[j]);
                double ep = host_d2h[i] < host_h2d[j] ? host_d2h[i] : host_h2d[j];
                if (hb > 0 && gbs <= hb * 1.2)
                    snprintf(p2p_note[i][j], sizeof(p2p_note[i][j]),
                             "P2P ~ host-bounce (%.1f GB/s): no direct-P2P gain", hb);
                else if (ep > 0 && gbs > ep * 1.5)
                    snprintf(p2p_note[i][j], sizeof(p2p_note[i][j]),
                             "SUSPECT: %.1f GB/s > endpoint PCIe (%.1f) - "
                             "import/migration, not link-bound", gbs, ep);
                else if (p2p_max[i][j] > p2p_min[i][j] * 3.0)
                    snprintf(p2p_note[i][j], sizeof(p2p_note[i][j]),
                             "unstable: min %.1f / max %.1f GB/s (first-touch/caching)",
                             p2p_min[i][j], p2p_max[i][j]);
                printf(LOG "  [%d->%d] %.1f GB/s (min %.1f / max %.1f)%s%s\n",
                       i, j, gbs, p2p_min[i][j], p2p_max[i][j],
                       p2p_note[i][j][0] ? "  -- " : "", p2p_note[i][j]);
            } else {
                p2p_gbs[i][j] = reason;
                printf(LOG "  [%d->%d] FAILED (submit=%s verify=%d reason=%d)\n",
                       i, j, failed ? "error" : "ok", verified, reason);
            }
            vkDestroyBuffer(B->dev, dstB, NULL);
            vkFreeMemory(B->dev, mDstB, NULL);
        }
    }
    printf("\n");
    print_p2p_matrix(devs, nd, p2p_gbs);

    /* Host-bounce matrix: the path used when direct P2P is unavailable (e.g.
     * cross-vendor).  Reports the effective A->host->B throughput, derived from
     * the measured one-way host<->device rates (harmonic mean of the two legs). */
    if (nd > 1) {
        printf("\n" LOG "=== Host-bounce matrix (A->host->B, rows=source, cols=dest, GB/s) ===\n");
        printf(LOG "     ");
        for (int j = 0; j < nd; j++) printf("%7d", j);
        printf("\n");
        for (int i = 0; i < nd; i++) {
            printf(LOG "%3d: ", i);
            for (int j = 0; j < nd; j++) {
                if (i == j || !devs[i].ok || !devs[j].ok) { printf("      -"); continue; }
                double gbs = host_bounce_gbs(host_d2h[i], host_h2d[j]);
                if (gbs < 0) { printf("  n/a "); continue; }
                printf("%7.1f", gbs);
            }
            printf("\n");
        }
        printf(LOG "n/a = no host<->device measurement; - = same/absent device\n");
    }

    /* ---- Final, human-readable summary of what was discovered ---- */
    printf("\n" LOG "==========================================================\n");
    printf(LOG "SUMMARY\n");
    printf(LOG "==========================================================\n");
    for (int i = 0; i < nd; i++) {
        const Dev *d = &devs[i];
        printf(LOG "  device[%d] %s (bdf=%04x:%02x:%02x.%x)\n", i, d->props.deviceName,
               d->pci_dom, d->pci_bus, d->pci_dev, d->pci_fn);
        printf(LOG "     vram=%.2f GiB  copy r+w=%.1f GB/s  pure read=%.1f GB/s  "
               "pure write=%.1f GB/s (fill %.1f)\n",
               (double)d->vram_bytes / (1024.0 * 1024.0 * 1024.0),
               dev_local_gbs[i], dev_read_gbs[i], dev_write_gbs[i], dev_fill_gbs[i]);
        printf(LOG "     host<->device: device->host %.1f GB/s, host->device %.1f GB/s\n",
               host_d2h[i], host_h2d[i]);
    }
    if (nd > 1) {
        printf(LOG "  device-to-device (P2P):\n");
        print_p2p_matrix(devs, nd, p2p_gbs);
        for (int i = 0; i < nd; i++) {
            for (int j = 0; j < nd; j++) {
                if (i == j) continue;
                double hb = host_bounce_gbs(host_d2h[i], host_h2d[j]);
                if (p2p_gbs[i][j] > 0) {
                    if (p2p_note[i][j][0])
                        printf(LOG "    device[%d] -> device[%d]: %s\n", i, j, p2p_note[i][j]);
                } else {
                    printf(LOG "    device[%d] -> device[%d]: P2P FAILED/unavailable", i, j);
                    if (hb > 0) printf("  => host-bounce estimate: %.1f GB/s", hb);
                    printf("\n");
                }
            }
        }
    }
    printf(LOG "==========================================================\n");

    for (int i = 0; i < nd; i++) {
        if (devs[i].ok) vkDestroyDevice(devs[i].dev, NULL);
    }
    vkDestroyInstance(inst, NULL);
    free(phys);
    return 0;
}
