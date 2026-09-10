/* vulkan/tools/kbench/kbench.c -- kernel microbenchmark per il backend
 * Vulkan di DS4 (Fase 7 tuning kernel).
 *
 * Misura il costo host+GPU dei matmul principali del decode al shape reale
 * (matvec 4096x4096, n_tok=1) e a shape configurabili via argomenti:
 *   kbench [in_dim [out_dim [n_tok [iters]]]]
 *
 * Kernels misurati:
 *   - matmul_f16   (proiezioni F16: 0.153 ms/call su RX 6900 XT @4096x4096)
 *   - matmul_q8_0  (path prequantizzato DOT4: v1 0.290 ms/call, ~67 GB/s vs
 *                   floor ~84 us; v2 0.124 ms/call -- Fase 7, 256 thread
 *                   attivi + letture vettorizzate)
 *   - matmul_f32   (fallback)
 *   Il matmul_q8_0 misura entrambe le varianti (DS4_VULKAN_FORCE_VARIANT) e
 *   ne confronta l'output (parity, tolleranza 1%).
 *
 * Nota: i kernel MoE IQ2/Q2K e attention decode si misurano sul modello reale
 * (DS4_METAL_GRAPH_LAYER_PROFILE + DS4_VULKAN_DEBUG_SUBMIT), vedi progress.md.
 *
 * Build: `make kbench` (locale) oppure, per il server Zen3:
 *   make NATIVE_CPU_FLAG=-march=znver3 kbench && scp kbench <server>:/tmp/
 * Il binario SI AUTO-SELEZIONA la GPU (probe bandwidth, veda ds4_vulkan.c) o
 * usa DS4_VULKAN_DEVICE_INDEX=N.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ds4_gpu.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static uint16_t f32_to_f16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t man = bits & 0x7fffffu;
    if (exp >= 31) return (uint16_t)(sign | 0x7bffu);
    if (exp <= 0) return sign;
    return (uint16_t)(sign | ((int32_t)exp << 10) | (man >> 13));
}

/* Close-enough for the 1% smoke tolerance: relative 1%, absolute floor near
 * zero (the v1-v2 accumulation order differs in the last ulps). */
static int ref_close(float a, float b) {
    float d = a > b ? a - b : b - a;
    float m = a > b ? a : b;
    if (m < 0.0f) m = -m;
    if (m < 1.0e-6f) return d < 1.0e-5f;
    return d / m <= 0.01f;
}

static int run_one(const char *name,
                   int (*fn)(ds4_gpu_tensor *out, const void *map,
                             uint64_t map_size, uint64_t off,
                             uint64_t in_dim, uint64_t out_dim,
                             const ds4_gpu_tensor *x, uint64_t n_tok),
                   const void *map, uint64_t map_size, uint64_t off,
                   uint32_t in_dim, uint32_t out_dim,
                   ds4_gpu_tensor *x, uint32_t n_tok, int iters,
                   float *out_host, uint32_t out_elems) {
    ds4_gpu_tensor *out =
        ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
    if (!out) { fprintf(stderr, "kbench: %s alloc failed\n", name); return 1; }
    double t0 = now_ms();
    /* Batch the iterations into one command scope: a bare dispatch is a
     * one-shot submit + device wait per call, which measures the round-trip
     * latency, not the kernel.  Both backends implement begin/end_commands. */
    ds4_gpu_begin_commands();
    for (int i = 0; i < iters; i++) {
        if (!fn(out, map, map_size, off, in_dim, out_dim, x, n_tok)) {
            fprintf(stderr, "kbench: %s failed at iter %d\n", name, i);
            ds4_gpu_tensor_free(out);
            return 1;
        }
    }
    ds4_gpu_end_commands();
    ds4_gpu_synchronize();
    double t1 = now_ms();
    ds4_gpu_tensor_read(out, 0, out_host,
                        (uint64_t)n_tok * out_dim * sizeof(float));
    ds4_gpu_tensor_free(out);
    /* sanity: finite and not all-zero */
    int nonzero = 0;
    for (uint32_t i = 0; i < out_elems; i++) {
        if (!(out_host[i] == out_host[i]) || out_host[i] == 0.0f) continue;
        if (out_host[i] > -1.0e30f && out_host[i] < 1.0e30f) nonzero++;
    }
    printf("kbench: %-14s in=%4u out=%4u tok=%u: %7.3f ms/call (%d iters) "
           "sanity=%s\n",
           name, in_dim, out_dim, n_tok, (t1 - t0) / (double)iters, iters,
           nonzero > 0 ? "ok" : "BAD");
    return 0;
}

int main(int argc, char **argv) {
    uint32_t in_dim = 4096u, out_dim = 4096u, n_tok = 1u, iters = 100;
    if (argc > 1) in_dim = (uint32_t)strtoul(argv[1], NULL, 10);
    if (argc > 2) out_dim = (uint32_t)strtoul(argv[2], NULL, 10);
    if (argc > 3) n_tok = (uint32_t)strtoul(argv[3], NULL, 10);
    if (argc > 4) iters = (int)strtoul(argv[4], NULL, 10);
    if (in_dim == 0 || out_dim == 0 || n_tok == 0 || iters <= 0) return 1;

    const uint64_t f16_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    const uint64_t q8_bytes = (uint64_t)out_dim * ((in_dim + 31u) / 32u) * 34u;
    const uint64_t f32_bytes = (uint64_t)out_dim * in_dim * sizeof(float);
    const uint64_t pad = 4096u;
    const uint64_t q8_off = ((f16_bytes + pad - 1) / pad) * pad;
    const uint64_t f32_off = ((q8_off + q8_bytes + pad - 1) / pad) * pad;
    const uint64_t buf_size =
        ((f32_off + f32_bytes + pad - 1) / pad) * pad;

    uint8_t *map = NULL;
    if (posix_memalign((void **)&map, 4096, (size_t)buf_size) != 0) return 1;
    memset(map, 0, (size_t)buf_size);
    /* F16 weights at 0 */
    for (uint64_t i = 0; i < f16_bytes / 2; i++) {
        ((uint16_t *)map)[i] =
            f32_to_f16(0.001f * (float)(int)(i % 1000) - 0.5f);
    }
    /* Q8_0 weights at q8_off: scale 1.0, qs pattern */
    for (uint64_t b = 0; b < q8_bytes / 34u; b++) {
        ((uint16_t *)map)[q8_off / 2u + b * 17u] = 0x3c00u;
        for (int i = 0; i < 32; i++) {
            map[q8_off + b * 34u + 2u + (uint64_t)i] = (uint8_t)(i % 100);
        }
    }
    /* F32 weights at f32_off */
    for (uint64_t i = 0; i < f32_bytes / 4; i++) {
        float v = 0.001f * (float)(int)(i % 1000) - 0.5f;
        memcpy(map + f32_off + i * 4, &v, 4);
    }

    if (ds4_gpu_init() == 0) { fprintf(stderr, "kbench: init failed\n"); return 1; }
    if (ds4_gpu_set_model_map(map, buf_size) == 0) {
        fprintf(stderr, "kbench: set_model_map failed\n");
        return 1;
    }

    ds4_gpu_tensor *x =
        ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
    if (!x) return 1;
    float *xv = (float *)malloc((size_t)n_tok * in_dim * sizeof(float));
    for (uint32_t i = 0; i < n_tok * in_dim; i++) {
        xv[i] = 0.01f * (float)(int)(i % 350) - 0.8f;
    }
    ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tok * in_dim * sizeof(float));

    float *out_host = (float *)malloc((size_t)n_tok * out_dim * sizeof(float));
    const uint32_t out_elems = n_tok * out_dim;

    run_one("matmul_f16", ds4_gpu_matmul_f16_tensor, map, buf_size, 0,
            in_dim, out_dim, x, n_tok, iters, out_host, out_elems);
    /* matmul_q8_0: misura la variante v1 e v2 (Fase 7 tuning) e ne confronta
     * l'output (tolleranza 1%, come lo smoke).  Le varianti sono forzate via
     * DS4_VULKAN_FORCE_VARIANT (valutato per dispatch, quindi il toggle a
     * metà processo funziona); il default a init è la v3 (dp4a) se il device
     * espone VK_KHR_shader_integer_dot_product, altrimenti la v2. */
    {
        float *q8_v1 = (float *)malloc((size_t)out_elems * sizeof(float));
        float *q8_v2 = (float *)malloc((size_t)out_elems * sizeof(float));
        float *q8_v3 = (float *)malloc((size_t)out_elems * sizeof(float));
        if (q8_v1 && q8_v2 && q8_v3) {
            setenv("DS4_VULKAN_FORCE_VARIANT", "Q8_PREQ:0", 1);
            run_one("matmul_q8_0", ds4_gpu_matmul_q8_0_tensor, map, buf_size,
                    q8_off, in_dim, out_dim, x, n_tok, iters, q8_v1, out_elems);
            setenv("DS4_VULKAN_FORCE_VARIANT", "Q8_PREQ:1", 1);
            run_one("matmul_q8_0_v2", ds4_gpu_matmul_q8_0_tensor, map,
                    buf_size, q8_off, in_dim, out_dim, x, n_tok, iters,
                    q8_v2, out_elems);
            setenv("DS4_VULKAN_FORCE_VARIANT", "Q8_PREQ:2", 1);
            run_one("matmul_q8_0_v3", ds4_gpu_matmul_q8_0_tensor, map,
                    buf_size, q8_off, in_dim, out_dim, x, n_tok, iters,
                    q8_v3, out_elems);
            unsetenv("DS4_VULKAN_FORCE_VARIANT");
            uint32_t bad = 0;
            for (uint32_t i = 0; i < out_elems; i++) {
                if (!ref_close(q8_v1[i], q8_v2[i])) bad++;
            }
            printf("kbench: parity q8 v1-v2: %s (%u/%u mismatches)\n",
                   bad == 0 ? "OK" : "FAIL", bad, out_elems);
            bad = 0;
            for (uint32_t i = 0; i < out_elems; i++) {
                if (!ref_close(q8_v2[i], q8_v3[i])) bad++;
            }
            printf("kbench: parity q8 v2-v3: %s (%u/%u mismatches)\n",
                   bad == 0 ? "OK" : "FAIL", bad, out_elems);
        }
        free(q8_v1);
        free(q8_v2);
        free(q8_v3);
    }
    run_one("matmul_f32", ds4_gpu_matmul_f32_tensor, map, buf_size, f32_off,
            in_dim, out_dim, x, n_tok, iters, out_host, out_elems);

    ds4_gpu_tensor_free(x);
    free(xv);
    free(out_host);
    ds4_gpu_cleanup();
    free(map);
    return 0;
}
