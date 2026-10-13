/* CPU access bandwidth to host-visible GPU tensors (VRAM-BAR on large-BAR GPUs)
 * against plain host RAM.  Targets the open question in progress_qwen.md
 * ("verifica del costo lato CPU dei buffer passati a VRAM-BAR").
 *
 * Run on one GPU at a time:  DS4_VULKAN_DEVICE_INDEX=<n> ./tests/test_vram_bar_bw
 * Add DS4_VULKAN_DEBUG_MEM=1 to print which memory type the tensor landed in. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ds4_gpu.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void report(const char *tag, uint64_t bytes, double s) {
    printf("%-26s %7.1f MiB %9.3f s %12.1f MB/s\n", tag,
           (double)bytes / 1048576.0, s, (double)bytes / 1.0e6 / s);
}

int main(void) {
    const char *mib_env = getenv("DS4_BW_MIB");
    const uint64_t mib = mib_env ? (uint64_t)strtoull(mib_env, NULL, 10) : 256ull;
    const uint64_t bytes = (mib ? mib : 256ull) * 1024ull * 1024ull;
    setvbuf(stdout, NULL, _IONBF, 0);
    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        return 1;
    }
    uint8_t *src = (uint8_t *)malloc(bytes);
    uint8_t *dst = (uint8_t *)malloc(bytes);
    if (!src || !dst) return 1;
    memset(src, 0x5a, bytes);
    memset(dst, 0, bytes);

    for (int rep = 0; rep < 2; rep++) {
        double t0 = now_s();
        memcpy(dst, src, bytes);
        report("host RAM memcpy", bytes, now_s() - t0);
    }

    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    if (!t) {
        fprintf(stderr, "ds4_gpu_tensor_alloc(%llu) failed\n", (unsigned long long)bytes);
        return 1;
    }
    for (int rep = 0; rep < 2; rep++) {
        double t0 = now_s();
        int ok = ds4_gpu_tensor_write(t, 0, src, bytes);
        report(ok ? "GPU tensor write (CPU)" : "GPU tensor write FAILED", bytes, now_s() - t0);
    }
    for (int rep = 0; rep < 2; rep++) {
        double t0 = now_s();
        int ok = ds4_gpu_tensor_read(t, 0, dst, bytes);
        report(ok ? "GPU tensor read (CPU)" : "GPU tensor read FAILED", bytes, now_s() - t0);
    }
    if (memcmp(src, dst, bytes) != 0) {
        printf("DATA MISMATCH after write+read\n");
    }
    ds4_gpu_tensor_free(t);
    free(src);
    free(dst);
    return 0;
}
