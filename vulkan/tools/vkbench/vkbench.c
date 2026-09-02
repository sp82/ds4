/* vkbench.c -- GPU benchmark per il server 4x RX 6900 XT.
 *
 * Uso: vkbench <device_index>
 * Misura (tutto su un device scelto per indice):
 *   1. Bandwidth VRAM  : saxpy (read a, read b, write out) + copy D2D
 *   2. Compute GFLOPS  : saxpy FLOPs
 *   3. Trasferimenti   : H2D e D2H via staging (buffer host-visible <-> device-local)
 *   4. Stabilità       : saxpy sostenuto per ~15 s, verifica device lost
 * Stampa anche PCI BDF (VK_EXT_pci_bus_info) e heap VRAM per mappare l'indice
 * RADV -> card/slot via sysfs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "saxpy_spv.h"
#include "madloop_spv.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static const char *vk_name(VkResult r) {
    switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    default: return "?";
    }
}

static VkInstance g_inst;
static VkPhysicalDevice g_phys;
static VkDevice g_dev;
static VkQueue g_q;
static VkPipelineLayout g_pl;
static VkPipeline g_pipe;
static VkDescriptorSetLayout g_dsl;
static VkDescriptorPool g_dpool;
static VkCommandPool g_cpool;
static VkCommandBuffer g_cb;
static VkPipeline g_pipe_mad;
static uint32_t g_mem_dev = UINT32_MAX, g_mem_host = UINT32_MAX;
static uint32_t g_phys_count = 0;
static char g_dev_name[256];
static int g_dev_idx = 0;
static int g_integrated = 0;

/* Riepilogo chiaro per chi non è esperto: giudizio su ogni metrica e il
 * comando ds4 consigliato. */
static void print_summary(double tflops, double vram_bw,
                          double h2d, double d2h, int stab_ok) {
    const double tr = h2d < d2h ? h2d : d2h;
    printf("\n");
    printf("====================================================================\n");
    printf(" RIEPILOGO (per chi non è esperto)\n");
    printf("====================================================================\n");
    printf("Scheda testata: %s  (indice %d di %u GPU rilevate)\n\n",
           g_dev_name, g_dev_idx, g_phys_count);

    printf("1) POTENZA DI CALCOLO ........... %.1f TFLOPS\n", tflops);
    printf("   -> %s\n",
           tflops >= 15.0 ? "OTTIMO: calcola come una RX 6900 XT sana." :
           tflops >= 10.0 ? "BUONO: normale per una GPU moderna." :
           tflops >= 5.0  ? "BASSO: possibile surriscaldamento (throttling) "
                            "o carico esterno." :
                            "ALLARMANTE: scheda molto lenta o difettosa.");

    printf("2) VELOCITA' MEMORIA VIDEO ....... %.0f GB/s\n", vram_bw);
    printf("   -> %s\n",
           vram_bw >= 150.0 ? "OTTIMO: memoria video veloce." :
           vram_bw >= 80.0  ? "BUONO: memoria normale." :
                              "BASSO: memoria lenta (GPU integrata o vecchia).");

    printf("3) TRASFERIMENTI HOST <-> GPU (PCIe):\n");
    printf("   upload   (PC -> GPU) .......... %.1f GB/s\n", h2d);
    printf("   download (GPU -> PC) .......... %.1f GB/s\n", d2h);
    if (g_integrated) {
        printf("   -> GPU INTEGRATA (iGPU): i 'trasferimenti' avvengono sulla RAM\n"
               "            di sistema condivisa, il giudizio PCIe non si applica.\n"
               "            Per ds4 va bene se la RAM di sistema e' sufficiente.\n");
    } else {
        printf("   -> %s\n",
               tr >= 10.0 ? "OTTIMO: la scheda e' su PCIe x16, PERFETTA per ds4 "
                            "(lo streaming degli esperti passa da qui)." :
               tr >= 2.0  ? "MEDIO: collegamento PCIe parziale (x8/x4), "
                            "utilizzabile ma non ideale." :
                            "LENTO: la scheda e' su PCIe x1, NON adatta a ds4 "
                            "(lo streaming sarebbe ~30x piu' lento).");
    }

    printf("4) STABILITA' (15 s di calcolo) ... %s\n",
           stab_ok ? "OK - nessun blocco o reset." :
                     "PROBLEMI - la scheda si e' bloccata o resettata.");

    printf("\n");
    if (g_integrated) {
        printf("CONCLUSIONE: GPU integrata, utilizzabile per ds4 se la RAM di "
               "sistema basta.\n");
        printf("Comando consigliato (indice %d):\n", g_dev_idx);
        printf("  ./ds4 --model <modello.gguf> --backend vulkan "
               "--ssd-streaming \\\n");
        printf("      --ssd-streaming-cache-experts 64 -c 256 -n 1 -t 4 "
               "--temp 0 -p \"Ciao\"\n");
    } else if (tr >= 10.0) {
        printf("CONCLUSIONE: questa scheda e' OTTIMA per ds4 Vulkan.\n");
        printf("Comando consigliato (usa SEMPRE l'indice %d):\n", g_dev_idx);
        printf("  DS4_VULKAN_DEVICE_INDEX=%d ./ds4 --model <modello.gguf> "
               "--backend vulkan \\\n", g_dev_idx);
        printf("      --ssd-streaming --ssd-streaming-cache-experts 64 "
               "-c 256 -n 1 -t 4 --temp 0 -p \"Ciao\"\n");
        printf("Nota: <modello.gguf> va sostituito col percorso del file del "
               "modello.\n");
    } else {
        printf("CONCLUSIONE: questa scheda NON e' adatta a ds4 "
               "(trasferimenti lenti).\n");
        printf("Trova quella giusta testandole tutte:\n");
        for (uint32_t i = 0; i < g_phys_count; i++) {
            printf("  vkbench %u\n", i);
        }
        printf("e usa l'indice che riporta trasferimenti oltre 10 GB/s.\n");
    }
    printf("====================================================================\n");
}

static int init(int dev_idx) {
    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "vkbench", 1, NULL, 0, VK_API_VERSION_1_0 };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &ai, 0, NULL, 0, NULL };
    if (vkCreateInstance(&ici, NULL, &g_inst) != VK_SUCCESS) { fprintf(stderr, "instance fail\n"); return 0; }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_inst, &n, NULL);
    g_phys_count = n;
    if ((uint32_t)dev_idx >= n) { fprintf(stderr, "no device %d (count %u)\n", dev_idx, n); return 0; }
    VkPhysicalDevice *devs = calloc(n, sizeof(*devs));
    vkEnumeratePhysicalDevices(g_inst, &n, devs);
    g_phys = devs[dev_idx];
    g_dev_idx = dev_idx;
    free(devs);

    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(g_phys, &p);
    g_integrated = (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    snprintf(g_dev_name, sizeof(g_dev_name), "%s", p.deviceName);
    printf("device[%d]: %s (api 0x%x, devID 0x%x)\n", dev_idx, p.deviceName, p.apiVersion, p.deviceID);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++) {
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            printf("  VRAM heap[%u]: %.2f GiB\n", i, (double)mp.memoryHeaps[i].size / 1073741824.0);
    }

    /* PCI BDF via VK_EXT_pci_bus_info (quando disponibile). */
    uint32_t extn = 0;
    vkEnumerateDeviceExtensionProperties(g_phys, NULL, &extn, NULL);
    VkExtensionProperties *exts = calloc(extn, sizeof(*exts));
    vkEnumerateDeviceExtensionProperties(g_phys, NULL, &extn, exts);
    int have_pci = 0;
    for (uint32_t i = 0; i < extn; i++)
        if (!strcmp(exts[i].extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME)) have_pci = 1;
    free(exts);
    if (have_pci) {
        VkPhysicalDeviceProperties2 p2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, NULL, p };
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT, NULL, 0,0,0,0 };
        p2.pNext = &pci;
        vkGetPhysicalDeviceProperties2(g_phys, &p2);
        printf("  PCI BDF: %04x:%02x:%02x.%d (dom:bus:dev.fn)\n",
               pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
    }

    /* queue family compute. */
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, NULL);
    VkQueueFamilyProperties *qf = calloc(qn, sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, qf);
    uint32_t qfam = UINT32_MAX;
    for (uint32_t i = 0; i < qn; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    free(qf);
    if (qfam == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 0; }

    const char *devext[1] = { VK_EXT_PCI_BUS_INFO_EXTENSION_NAME };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qfam, 1, &prio };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL,
                               have_pci ? 1u : 0u, have_pci ? devext : NULL, NULL };
    if (vkCreateDevice(g_phys, &dci, NULL, &g_dev) != VK_SUCCESS) { fprintf(stderr, "device fail\n"); return 0; }
    vkGetDeviceQueue(g_dev, qfam, 0, &g_q);

    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if (g_mem_dev == UINT32_MAX && (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) g_mem_dev = i;
        if (g_mem_host == UINT32_MAX && (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) g_mem_host = i;
    }
    if (g_mem_host == UINT32_MAX)
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) { g_mem_host = i; break; }
    printf("  mem: device-local=%u host-visible=%u\n", g_mem_dev, g_mem_host);
    if (g_mem_dev == UINT32_MAX || g_mem_host == UINT32_MAX) { fprintf(stderr, "no mem type\n"); return 0; }

    VkDescriptorSetLayoutBinding b[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dlci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, b };
    vkCreateDescriptorSetLayout(g_dev, &dlci, NULL, &g_dsl);
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_dsl, 1, &pcr };
    vkCreatePipelineLayout(g_dev, &plci, NULL, &g_pl);

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, saxpy_spv_len, (const uint32_t *)saxpy_spv };
    VkShaderModule mod;
    if (vkCreateShaderModule(g_dev, &smci, NULL, &mod) != VK_SUCCESS) { fprintf(stderr, "shader fail\n"); return 0; }
    VkPipelineShaderStageCreateInfo ss = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "saxpy", NULL };
    VkComputePipelineCreateInfo cpi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, ss, g_pl, VK_NULL_HANDLE, 0 };
    vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpi, NULL, &g_pipe);
    vkDestroyShaderModule(g_dev, mod, NULL);

    VkShaderModuleCreateInfo smci2 = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, madloop_spv_len, (const uint32_t *)madloop_spv };
    VkShaderModule mod2;
    vkCreateShaderModule(g_dev, &smci2, NULL, &mod2);
    VkPipelineShaderStageCreateInfo ss2 = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod2, "madloop", NULL };
    VkComputePipelineCreateInfo cpi2 = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, ss2, g_pl, VK_NULL_HANDLE, 0 };
    vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpi2, NULL, &g_pipe_mad);
    vkDestroyShaderModule(g_dev, mod2, NULL);

    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 3, 1, &ps };
    vkCreateDescriptorPool(g_dev, &dpci, NULL, &g_dpool);
    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfam };
    vkCreateCommandPool(g_dev, &cpci, NULL, &g_cpool);
    VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    vkAllocateCommandBuffers(g_dev, &cai, &g_cb);
    return 1;
}

typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceSize size;
    void *map;
} Buf;

static int buf_alloc(Buf *b, VkDeviceSize size, uint32_t mem_type, int host_map) {
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    if (vkCreateBuffer(g_dev, &bci, NULL, &b->buf) != VK_SUCCESS) return 0;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_dev, b->buf, &req);
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, mem_type };
    if (vkAllocateMemory(g_dev, &mai, NULL, &b->mem) != VK_SUCCESS) return 0;
    vkBindBufferMemory(g_dev, b->buf, b->mem, 0);
    b->size = size;
    b->map = NULL;
    if (host_map) vkMapMemory(g_dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map);
    return 1;
}

static void buf_free(Buf *b) {
    if (b->map) vkUnmapMemory(g_dev, b->mem);
    if (b->buf) vkDestroyBuffer(g_dev, b->buf, NULL);
    if (b->mem) vkFreeMemory(g_dev, b->mem, NULL);
    memset(b, 0, sizeof(*b));
}

static void bind_bufs(VkDescriptorSet set, Buf *a, Buf *b, Buf *out) {
    VkDescriptorBufferInfo infos[3] = {
        { a->buf, 0, a->size }, { b->buf, 0, b->size }, { out->buf, 0, out->size } };
    VkWriteDescriptorSet w[3] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &infos[0], NULL },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &infos[1], NULL },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &infos[2], NULL },
    };
    vkUpdateDescriptorSets(g_dev, 3, w, 0, NULL);
}

static int run_saxpy(VkDescriptorSet set, uint32_t groups) {
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
    if (vkBeginCommandBuffer(g_cb, &bi) != VK_SUCCESS) return 0;
    vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipe);
    vkCmdBindDescriptorSets(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_pl, 0, 1, &set, 0, NULL);
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
    for (int i = 0; i < 4; i++) {
        vkCmdDispatch(g_cb, groups, 1, 1);
        vkCmdPipelineBarrier(g_cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(g_cb);
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
    VkFence f;
    vkCreateFence(g_dev, &fci, NULL, &f);
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &g_cb, 0, NULL };
    VkResult r = vkQueueSubmit(g_q, 1, &si, f);
    if (r == VK_SUCCESS) {
        while (vkWaitForFences(g_dev, 1, &f, VK_TRUE, 1000000000ull) == VK_TIMEOUT) {}
    }
    vkDestroyFence(g_dev, f, NULL);
    return r == VK_SUCCESS;
}

int main(int argc, char **argv) {
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    if (!init(dev)) return 1;

    const uint32_t N = 64u * 1024u * 1024u;   /* 64M float = 256 MiB each */
    const VkDeviceSize bytes = (VkDeviceSize)N * 4;
    const uint32_t groups = (N + 255u) / 256u;

    Buf a = {0}, b = {0}, out = {0}, staging = {0};
    if (!buf_alloc(&a, bytes, g_mem_dev, 0) || !buf_alloc(&b, bytes, g_mem_dev, 0) ||
        !buf_alloc(&out, bytes, g_mem_dev, 0) || !buf_alloc(&staging, bytes, g_mem_host, 1)) {
        fprintf(stderr, "alloc fail\n"); return 1;
    }
    /* init staging so host pages are committed */
    for (uint64_t i = 0; i < N; i += 4096) ((float *)staging.map)[i] = 1.0f;

    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_dpool, 1, &g_dsl };
    VkDescriptorSet set;
    vkAllocateDescriptorSets(g_dev, &dsai, &set);
    bind_bufs(set, &a, &b, &out);

    struct { uint32_t n; float alpha; uint32_t pad0, pad1; } pc = { N, 2.0f, 0, 0 };
    vkCmdPushConstants(g_cb, g_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc); /* no-op warm */

    /* warmup */
    run_saxpy(set, groups);

    double fma_tflops = 0.0, vram_bw = 0.0, h2d_bw = 0.0, d2h_bw = 0.0;

    /* ---- 1. VRAM bandwidth + compute GFLOPS via saxpy ---- */
    const int iters = 10;
    double best = 1e18;
    for (int i = 0; i < iters; i++) {
        double t0 = now_ms();
        if (!run_saxpy(set, groups)) { printf("  saxpy FAILED\n"); return 1; }
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    /* saxpy: read a(4N) + read b(4N) + write out(4N) = 12N bytes */
    double bw = (double)N * 12.0 / (best / 1000.0) / 1e9;         /* GB/s */
    double gflops = (double)N * 2.0 / (best / 1000.0) / 1e9;       /* GFLOPS (mul+add) */
    printf("SAXPY:  best %.2f ms  ->  %.2f GB/s (read+write)  %.1f GFLOPS\n",
           best, bw, gflops);

    /* ---- 1b. compute-bound FP32 FMA throughput (madloop) ---- */
    {
        const uint32_t iters = 4096u;
        const int iters2 = 5;
        struct { uint32_t n; uint32_t iters; uint32_t p0, p1; } mpc = { N, iters, 0, 0 };
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
        vkBeginCommandBuffer(g_cb, &bi);
        vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipe_mad);
        vkCmdBindDescriptorSets(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_pl, 0, 1, &set, 0, NULL);
        vkCmdPushConstants(g_cb, g_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mpc), &mpc);
        vkCmdDispatch(g_cb, groups, 1, 1);
        vkEndCommandBuffer(g_cb);
        best = 1e18;
        for (int i = 0; i < iters2; i++) {
            double t0 = now_ms();
            VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
            VkFence f; vkCreateFence(g_dev, &fci, NULL, &f);
            VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &g_cb, 0, NULL };
            VkResult r = vkQueueSubmit(g_q, 1, &si, f);
            vkWaitForFences(g_dev, 1, &f, VK_TRUE, UINT64_MAX);
            vkDestroyFence(g_dev, f, NULL);
            double dt = now_ms() - t0;
            if (r == VK_SUCCESS && dt < best) best = dt;
        }
        double fma_flops = (double)N * (double)iters * 4.0 * 2.0;
        fma_tflops = fma_flops / (best / 1000.0) / 1e12;
        printf("FMA:    best %.2f ms  ->  %.2f TFLOPS FP32 (compute-bound)\n",
               best, fma_tflops);
    }

    /* ---- 2. copy D2D bandwidth (vkCmdCopyBuffer) ---- */
    {
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
        VkBufferCopy bc = { 0, 0, bytes };
        best = 1e18;
        for (int i = 0; i < iters; i++) {
            vkBeginCommandBuffer(g_cb, &bi);
            vkCmdCopyBuffer(g_cb, a.buf, out.buf, 1, &bc);
            vkEndCommandBuffer(g_cb);
            double t0 = now_ms();
            VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
            VkFence f; vkCreateFence(g_dev, &fci, NULL, &f);
            VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &g_cb, 0, NULL };
            VkResult r = vkQueueSubmit(g_q, 1, &si, f);
            vkWaitForFences(g_dev, 1, &f, VK_TRUE, UINT64_MAX);
            vkDestroyFence(g_dev, f, NULL);
            double dt = now_ms() - t0;
            if (r == VK_SUCCESS && dt < best) best = dt;
        }
        vram_bw = (double)bytes / (best / 1000.0) / 1e9;
        printf("COPY D2D: best %.2f ms  ->  %.2f GB/s\n", best, vram_bw);
    }

    /* ---- 3. transfer H2D / D2H (host-visible <-> device-local) ---- */
    {
        VkBufferCopy bc = { 0, 0, bytes };
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
        for (int dir = 0; dir < 2; dir++) {
            best = 1e18;
            for (int i = 0; i < iters; i++) {
                vkBeginCommandBuffer(g_cb, &bi);
                if (dir == 0) vkCmdCopyBuffer(g_cb, staging.buf, a.buf, 1, &bc);  /* H2D */
                else          vkCmdCopyBuffer(g_cb, a.buf, staging.buf, 1, &bc);  /* D2H */
                vkEndCommandBuffer(g_cb);
                double t0 = now_ms();
                VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
                VkFence f; vkCreateFence(g_dev, &fci, NULL, &f);
                VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &g_cb, 0, NULL };
                VkResult r = vkQueueSubmit(g_q, 1, &si, f);
                vkWaitForFences(g_dev, 1, &f, VK_TRUE, UINT64_MAX);
                vkDestroyFence(g_dev, f, NULL);
                double dt = now_ms() - t0;
                if (r == VK_SUCCESS && dt < best) best = dt;
            }
            double tr_bw = (double)bytes / (best / 1000.0) / 1e9;
            if (dir == 0) h2d_bw = tr_bw; else d2h_bw = tr_bw;
            printf("%s: best %.2f ms  ->  %.2f GB/s\n",
                   dir == 0 ? "H2D      " : "D2H      ",
                   best, tr_bw);
        }
    }

    /* ---- 4. stability: saxpy sostenuto per ~15 s ---- */
    {
        const double dur = 15000.0;
        double t0 = now_ms();
        long it = 0;
        VkResult bad = VK_SUCCESS;
        while (now_ms() - t0 < dur) {
            if (!run_saxpy(set, groups)) { bad = VK_ERROR_DEVICE_LOST; break; }
            it++;
        }
        double el = now_ms() - t0;
        double per = el / (double)it;
        printf("STABILITY: %ld iterazioni in %.1f s  ->  %.2f ms/iter  [%s]\n",
               it, el / 1000.0, per, bad == VK_SUCCESS ? "OK" : vk_name(bad));
        print_summary(fma_tflops, vram_bw, h2d_bw, d2h_bw,
                      bad == VK_SUCCESS);
    }

    buf_free(&staging); buf_free(&out); buf_free(&b); buf_free(&a);
    vkDeviceWaitIdle(g_dev);
    vkDestroyPipeline(g_dev, g_pipe, NULL);
    vkDestroyPipelineLayout(g_dev, g_pl, NULL);
    vkDestroyDescriptorSetLayout(g_dev, g_dsl, NULL);
    vkDestroyDescriptorPool(g_dev, g_dpool, NULL);
    vkDestroyCommandPool(g_dev, g_cpool, NULL);
    vkDestroyDevice(g_dev, NULL);
    vkDestroyInstance(g_inst, NULL);
    return 0;
}
