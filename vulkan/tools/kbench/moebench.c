/* vulkan/tools/kbench/moebench.c -- routed MoE microbenchmark for the DS4
 * Vulkan backend, using DeepSeek V4 Flash expert dimensions.
 *
 * Measures the MXFP4 routed MoE (gate/up/down experts) at the real shapes:
 *   in_dim=4096, mid_dim=2048, out_dim=4096, n_expert_used=6
 * for decode (1 token) and prefill (64 tokens).  The synthetic expert weights
 * are random MXFP4 blocks (any E2M1 code is valid); only the read+dequant+dot
 * cost matters, so the values are irrelevant to the timing.
 *
 * Build: make moebench   (Vulkan)   /   make moebench-cuda CUDA_ARCH=sm_120
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "ds4_gpu.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

#define MOE_IN_DIM   4096u
#define MOE_MID_DIM  2048u
#define MOE_OUT_DIM  4096u
#define MOE_N_TOTAL  64u
#define MOE_N_USED   6u
#define MOE_GATE_TYPE 39u   /* GGUF MXFP4 */
#define MOE_DOWN_TYPE 39u

/* gate_row_bytes = (in/32)*17, down_row_bytes = (mid/32)*17. */
static const uint64_t g_gate_row = (MOE_IN_DIM / 32u) * 17u;
static const uint64_t g_down_row = (MOE_MID_DIM / 32u) * 17u;
static const uint64_t g_gate_expert = (uint64_t)MOE_MID_DIM * g_gate_row;
static const uint64_t g_down_expert = (uint64_t)MOE_OUT_DIM * g_down_row;

static void fill_mxfp4(uint8_t *base, uint64_t rows, uint64_t row_bytes) {
    for (uint64_t r = 0; r < rows; r++) {
        uint8_t *blk = base + r * row_bytes;
        for (uint64_t off = 0; off < row_bytes; off += 17u) {
            blk[off] = (uint8_t)(124u + (uint32_t)(rand() % 7));
            for (uint32_t i = 1u; i < 17u; i++) {
                blk[off + i] = (uint8_t)(rand() & 0xffu);
            }
        }
    }
}

static void run_case(const char *name, const uint8_t *map, uint64_t map_size,
                     uint64_t up_off, uint64_t down_off, uint32_t n_tokens,
                     int iters) {
    const uint64_t pairs = (uint64_t)n_tokens * MOE_N_USED;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * MOE_IN_DIM * 4u);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(pairs * 4u);
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(pairs * 4u);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(pairs * MOE_MID_DIM * 4u);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(pairs * MOE_MID_DIM * 4u);
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(pairs * MOE_MID_DIM * 4u);
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(pairs * MOE_OUT_DIM * 4u);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * MOE_OUT_DIM * 4u);
    if (!x || !selected || !weights || !gate || !up || !mid || !down || !out) {
        fprintf(stderr, "moebench: %s alloc failed\n", name);
        return;
    }
    float *xv = (float *)malloc((size_t)n_tokens * MOE_IN_DIM * 4u);
    int32_t *sv = (int32_t *)malloc((size_t)pairs * 4u);
    float *wv = (float *)malloc((size_t)pairs * 4u);
    for (uint32_t i = 0; i < n_tokens * MOE_IN_DIM; i++) {
        xv[i] = 0.2f * (float)(i % 13) - 0.5f;
    }
    for (uint64_t i = 0; i < pairs; i++) {
        sv[i] = (int32_t)((i * 5u + 1u) % MOE_N_TOTAL);
        wv[i] = 0.3f + 0.1f * (float)(i % 5);
    }
    ds4_gpu_tensor_write(x, 0, xv, (uint64_t)n_tokens * MOE_IN_DIM * 4u);
    ds4_gpu_tensor_write(selected, 0, sv, pairs * 4u);
    ds4_gpu_tensor_write(weights, 0, wv, pairs * 4u);

    const int ok = ds4_gpu_begin_commands();
    const double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        if (!ds4_gpu_routed_moe_batch_tensor(
                    out, gate, up, mid, down, map, map_size,
                    0u, up_off, down_off, MOE_GATE_TYPE, MOE_DOWN_TYPE,
                    g_gate_expert, g_gate_row, g_down_expert, g_down_row,
                    MOE_IN_DIM, MOE_MID_DIM, MOE_OUT_DIM, selected, weights,
                    MOE_N_TOTAL, MOE_N_USED, 5.0f, x, 0u, n_tokens, NULL,
                    true)) {
            fprintf(stderr, "moebench: %s dispatch failed at %d\n", name, it);
            break;
        }
    }
    ds4_gpu_end_commands();
    ds4_gpu_synchronize();
    const double t1 = now_ms();
    (void)ok;
    const double ms = (t1 - t0) / (double)iters;
    const double mb = (double)pairs *
                      (double)(2ull * g_gate_expert + g_down_expert) / 1e6;
    printf("moebench: %-22s tok=%3u pairs=%3llu: %8.3f ms/call (%d iters) "
           "read<=%.1f MB -> %.1f GB/s\n",
           name, n_tokens, (unsigned long long)pairs, ms, iters, mb,
           mb / ms);

    free(xv); free(sv); free(wv);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(out);
}

int main(int argc, char **argv) {
    int iters = 50;
    if (argc > 1) iters = atoi(argv[1]);
    if (iters <= 0) iters = 50;

    const uint64_t gate_region = (uint64_t)MOE_N_TOTAL * g_gate_expert;
    const uint64_t down_region = (uint64_t)MOE_N_TOTAL * g_down_expert;
    const uint64_t up_off = gate_region;
    const uint64_t down_off = 2u * gate_region;
    const uint64_t map_size = (down_off + down_region + 4095u) &
                              ~(uint64_t)4095u;

    uint8_t *map = NULL;
    if (posix_memalign((void **)&map, 4096, (size_t)map_size) != 0) return 1;
    memset(map, 0, (size_t)map_size);
    srand(0x3f9au);
    fill_mxfp4(map, (uint64_t)MOE_N_TOTAL * MOE_MID_DIM, g_gate_row);
    fill_mxfp4(map + up_off, (uint64_t)MOE_N_TOTAL * MOE_MID_DIM, g_gate_row);
    fill_mxfp4(map + down_off, (uint64_t)MOE_N_TOTAL * MOE_OUT_DIM, g_down_row);
    printf("moebench: map %.1f MB, gate_expert %.2f MB, down_expert %.2f MB\n",
           (double)map_size / 1e6, (double)g_gate_expert / 1e6,
           (double)g_down_expert / 1e6);

    if (ds4_gpu_init() == 0) { fprintf(stderr, "moebench: init failed\n"); return 1; }
    if (ds4_gpu_set_model_map(map, map_size) == 0) {
        fprintf(stderr, "moebench: set_model_map failed\n");
        return 1;
    }
    run_case("mxfp4_decode", map, map_size, up_off, down_off, 1u, iters);
    run_case("mxfp4_prefill_64", map, map_size, up_off, down_off, 64u, iters / 2 + 1);
    ds4_gpu_cleanup();
    free(map);
    return 0;
}
