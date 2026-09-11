/* vulkan/tools/kbench/moebench.c -- routed MoE microbenchmark for the DS4
 * Vulkan backend, using DeepSeek V4 Flash expert dimensions.
 *
 * Measures the routed MoE (gate/up/down experts) at the real shapes:
 *   in_dim=4096, mid_dim=2048, out_dim=4096, n_expert_used=6
 * for decode (1 token) and prefill (8/16/32/64 tokens).  The synthetic expert
 * weights are random bytes (any code is valid for timing); only the
 * read+dequant+dot cost matters, so the values are irrelevant.
 *
 * Usage: moebench [iters [quant]]
 *   quant = mxfp4 (default) | iq2 | q4k
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

/* Expert quant sets: gate/up and down GGUF types + row byte counts.
 * row_bytes = (dim / block_elems) * block_bytes:
 *   MXFP4   (39): 17 B / 32    IQ2_XXS (16): 66 B / 256
 *   Q2_K    (10): 84 B / 256   Q4_K    (12): 144 B / 256 */
struct quant_set {
    const char *name;
    uint32_t    gate_type;
    uint32_t    down_type;
    uint64_t    gate_row;
    uint64_t    down_row;
};

static const struct quant_set g_sets[] = {
    { "mxfp4", 39u, 39u, (MOE_IN_DIM / 32u) * 17u,  (MOE_MID_DIM / 32u) * 17u  },
    { "iq2",   16u, 10u, (MOE_IN_DIM / 256u) * 66u, (MOE_MID_DIM / 256u) * 84u },
    { "q4k",   12u, 12u, (MOE_IN_DIM / 256u) * 144u,(MOE_MID_DIM / 256u) * 144u},
    { "q8",    8u,  8u,  (MOE_IN_DIM / 32u) * 34u,  (MOE_MID_DIM / 32u) * 34u  },
};

static uint32_t g_gate_type, g_down_type;
static uint64_t g_gate_row, g_down_row, g_gate_expert, g_down_expert;

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
                    0u, up_off, down_off, g_gate_type, g_down_type,
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
    const char *quant = "mxfp4";
    if (argc > 1) iters = atoi(argv[1]);
    if (argc > 2) quant = argv[2];
    if (iters <= 0) iters = 50;

    const struct quant_set *qs = &g_sets[0];
    for (size_t i = 0; i < sizeof(g_sets) / sizeof(g_sets[0]); i++) {
        if (!strcmp(g_sets[i].name, quant)) { qs = &g_sets[i]; break; }
    }
    g_gate_type = qs->gate_type;
    g_down_type = qs->down_type;
    g_gate_row = qs->gate_row;
    g_down_row = qs->down_row;
    g_gate_expert = (uint64_t)MOE_MID_DIM * g_gate_row;
    g_down_expert = (uint64_t)MOE_OUT_DIM * g_down_row;

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
    for (uint64_t i = 0; i < map_size; i++) map[i] = (uint8_t)(rand() & 0xff);
    printf("moebench: quant=%s map %.1f MB, gate_expert %.2f MB, "
           "down_expert %.2f MB\n", qs->name, (double)map_size / 1e6,
           (double)g_gate_expert / 1e6, (double)g_down_expert / 1e6);

    if (ds4_gpu_init() == 0) { fprintf(stderr, "moebench: init failed\n"); return 1; }
    if (ds4_gpu_set_model_map(map, map_size) == 0) {
        fprintf(stderr, "moebench: set_model_map failed\n");
        return 1;
    }
    run_case("decode", map, map_size, up_off, down_off, 1u, iters);
    run_case("prefill_32", map, map_size, up_off, down_off, 32u, iters / 2 + 1);
    run_case("prefill_64", map, map_size, up_off, down_off, 64u, iters / 2 + 1);
    ds4_gpu_cleanup();
    free(map);
    return 0;
}
