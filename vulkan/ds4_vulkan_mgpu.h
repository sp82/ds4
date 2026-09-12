/* vulkan/ds4_vulkan_mgpu.h -- multi-device Vulkan manager (Fase 8e / M1).
 *
 * Discovery + classification of the physical devices of one Vulkan instance.
 * The PCIe class is measured, not read from sysfs: the server's x1 cards
 * advertise x16 but run at ~0.8 GB/s (see AGENTS.md).  See SPECS_MGPU.md. */
#ifndef DS4_VULKAN_MGPU_H
#define DS4_VULKAN_MGPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_VK_MAX_DEVICES 16

typedef struct ds4_vk_dev_info {
    uint32_t vk_index;      /* Vulkan enumeration index (not stable across boot) */
    char     name[256];     /* VkPhysicalDeviceProperties.deviceName */
    char     bdf[32];       /* "dddd:bb:dd.f" or "n/a" */
    double   bw_gbps;       /* measured host<->device min(H2D,D2H) */
    uint64_t vram_bytes;    /* largest DEVICE_LOCAL heap */
    int      is_fast;       /* bw_gbps >= fast threshold */
    int      usable;        /* has a compute queue and is not CPU/virtual */
} ds4_vk_dev_info;

/* Enumerate and classify every physical device visible to a Vulkan instance.
 * Uses g_instance when already created, otherwise a throwaway one.  On success
 * returns the number of entries written (<= max_devices); 0 on failure. */
int ds4_vulkan_probe_devices(ds4_vk_dev_info *out, int max_devices);

#ifdef __cplusplus
}
#endif

#endif /* DS4_VULKAN_MGPU_H */
