/* vulkan/tools/mgpu_probe/mgpu_probe.c -- multi-GPU capability probe.
 *
 * Standalone (links only Vulkan): enumerates the physical devices, reports
 * per-device VRAM, measures device-local and host<->device bandwidth, and
 * tests cross-device (P2P) transfer via VK_KHR_external_memory_fd (dma-buf /
 * opaque fd) + optional external semaphores.
 *
 * Purpose: characterize the hardware BEFORE implementing the native Vulkan
 * multi-GPU backend (SPEC.md §8e / progress.md §1b), so placement and the
 * P2P-vs-host-bounce decision are made on measured facts.
 *
 * Build:  make mgpu-probe
 * Run:    ./mgpu-probe [--mb N] [--devices 0,1,2,3] [--iters N]
 * Env:    DS4_VULKAN_DEVICE_INDEX is ignored (this tool wants every device).
 *
 * Output: one block per device + a P2P matrix (ordered pairs).  Any
 * unsupported step is reported, never fatal.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define LOG "mgpu-probe: "

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
    int ok;
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
                     int host_visible, VkBuffer *buf_out,
                     VkDeviceMemory *mem_out) {
    VkBufferCreateInfo bci;
    memset(&bci, 0, sizeof(bci));
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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
    if (!find_mem_type(d, bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &mt) &&
        !find_mem_type(d, bits, 0, &mt)) {
        vkDestroyBuffer(d->dev, buf, NULL);
        return 0;
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

/* Run iters copies src->dst on the device and return total ms. */
static double copy_bench(const Dev *d, VkBuffer src, VkBuffer dst, uint64_t bytes,
                         int iters) {
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
        vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE);
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
    vkCmdFillBuffer(d->cb, buf, 0, bytes, 0u);
    vkEndCommandBuffer(d->cb);
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &d->cb;
    vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(d->queue);
}

/* Copy src (possibly imported / on another device) into dst, on d's queue. */
static double xcopy_bench(const Dev *d, VkBuffer src, VkBuffer dst,
                          uint64_t bytes, int iters) {
    return copy_bench(d, src, dst, bytes, iters);
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

int main(int argc, char **argv) {
    uint64_t mb = 256;
    int iters = 5;
    const char *devfilter = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mb") && i + 1 < argc) mb = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--devices") && i + 1 < argc) devfilter = argv[++i];
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
    int nd = 0;
    for (uint32_t i = 0; i < np && nd < 16; i++) {
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
        printf("\n" LOG "device[%d] (plat=%u): %s\n", nd, i, d->props.deviceName);
        printf(LOG "  type=%d api=%u.%u.%u vram=%.2f GiB\n", d->props.deviceType,
               VK_VERSION_MAJOR(d->props.apiVersion), VK_VERSION_MINOR(d->props.apiVersion),
               VK_VERSION_PATCH(d->props.apiVersion),
               (double)d->vram_bytes / (1024.0 * 1024.0 * 1024.0));
        print_mem_heaps(d);
        d->ok = init_device(d);
        nd++;
    }

    /* Per-device bandwidth + VRAM budget. */
    for (int i = 0; i < nd; i++) {
        Dev *d = &devs[i];
        if (!d->ok) { printf("\n" LOG "device[%d]: no device\n", i); continue; }
        uint64_t bytes = mb * 1024ull * 1024ull;
        if (bytes > d->vram_bytes / 8) bytes = d->vram_bytes / 8;
        VkBuffer a = VK_NULL_HANDLE, b = VK_NULL_HANDLE;
        VkDeviceMemory ma = VK_NULL_HANDLE, mbh = VK_NULL_HANDLE;
        VkBuffer ha = VK_NULL_HANDLE;
        VkDeviceMemory mha = VK_NULL_HANDLE;
        if (!alloc_buf(d, bytes, 0, 0, &a, &ma) || !alloc_buf(d, bytes, 0, 0, &b, &mbh)) {
            printf("\n" LOG "device[%d]: device-local alloc failed\n", i);
            continue;
        }
        double ms = copy_bench(d, a, b, bytes, iters);
        double gbs = (double)bytes * 2.0 * iters / (ms * 1.0e6);
        printf("\n" LOG "device[%d] %s: VRAM %.2f GiB, copy (r+w) %.1f GB/s (%llu MiB x%d)\n",
               i, d->props.deviceName,
               (double)d->vram_bytes / (1024.0 * 1024.0 * 1024.0), gbs,
               (unsigned long long)mb, iters);
        if (alloc_buf(d, bytes, 0, 1, &ha, &mha)) {
            double hms = copy_bench(d, a, ha, bytes, iters);
            double hgbs = (double)bytes * 2.0 * iters / (hms * 1.0e6);
            printf(LOG "  host<->device copy (r+w) %.1f GB/s\n", hgbs);
            vkDestroyBuffer(d->dev, ha, NULL);
            vkFreeMemory(d->dev, mha, NULL);
        }
        vkDestroyBuffer(d->dev, a, NULL);
        vkFreeMemory(d->dev, ma, NULL);
        vkDestroyBuffer(d->dev, b, NULL);
        vkFreeMemory(d->dev, mbh, NULL);
    }

    /* P2P matrix. */
    printf("\n" LOG "=== P2P transfer matrix (rows=source, cols=dest, GB/s) ===\n");
    printf(LOG "     ");
    for (int j = 0; j < nd; j++) printf("%7d", j);
    printf("\n");
    for (int i = 0; i < nd; i++) {
        printf(LOG "%3d: ", i);
        for (int j = 0; j < nd; j++) {
            if (i == j || !devs[i].ok || !devs[j].ok) { printf("      -"); continue; }
            Dev *A = &devs[i];
            Dev *B = &devs[j];
            if (!A->ext_mem_fd || !A->dma_buf || !pGetMemoryFdKHR) {
                printf("   n/a ");
                continue;
            }
            uint64_t bytes = mb * 1024ull * 1024ull;
            if (bytes > A->vram_bytes / 8) bytes = A->vram_bytes / 8;
            VkBuffer srcA = VK_NULL_HANDLE, dstB = VK_NULL_HANDLE;
            VkDeviceMemory mSrcA = VK_NULL_HANDLE, mDstB = VK_NULL_HANDLE;
            if (!alloc_buf(A, bytes, 1, 0, &srcA, &mSrcA)) { printf("   alloc"); continue; }
            fill_buf(A, srcA, bytes);
            int fd = -1;
            VkMemoryGetFdInfoKHR gfi;
            memset(&gfi, 0, sizeof(gfi));
            gfi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
            gfi.memory = mSrcA;
            gfi.handleType = A->dma_buf
                ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
                : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            if (pGetMemoryFdKHR(A->dev, &gfi, &fd) != VK_SUCCESS) {
                printf("  nofd ");
                vkDestroyBuffer(A->dev, srcA, NULL);
                vkFreeMemory(A->dev, mSrcA, NULL);
                continue;
            }
            VkBuffer impB = VK_NULL_HANDLE;
            VkDeviceMemory mImpB = VK_NULL_HANDLE;
            if (alloc_buf(B, bytes, 0, 0, &dstB, &mDstB) &&
                import_buf2(B, fd, bytes, &impB, &mImpB)) {
                double ms = xcopy_bench(B, impB, dstB, bytes, iters);
                double gbs = (double)bytes * iters / (ms * 1.0e6);
                printf("%7.1f", gbs);
            } else {
                printf("  fail ");
            }
            if (dstB) { vkDestroyBuffer(B->dev, dstB, NULL); vkFreeMemory(B->dev, mDstB, NULL); }
            if (impB) { vkDestroyBuffer(B->dev, impB, NULL); vkFreeMemory(B->dev, mImpB, NULL); }
            vkDestroyBuffer(A->dev, srcA, NULL);
            vkFreeMemory(A->dev, mSrcA, NULL);
        }
        printf("\n");
    }
    printf(LOG "n/a = external memory dma-buf unsupported; fail = import/copy failed; "
           "- = same/absent device\n");

    for (int i = 0; i < nd; i++) {
        if (devs[i].ok) vkDestroyDevice(devs[i].dev, NULL);
    }
    vkDestroyInstance(inst, NULL);
    free(phys);
    return 0;
}
