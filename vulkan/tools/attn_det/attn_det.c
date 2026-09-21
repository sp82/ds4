/* vulkan/tools/attn_det/attn_det.c -- standalone determinism + correctness +
 * performance harness for the Vulkan attention-decode kernels, WITHOUT loading
 * a model.
 *
 * Modes (env):
 *   ATTN_DET_BENCH=1         also time each attention kernel (ms/call).
 *
 * For each config it runs the kernel many times, reads back the heads and
 * hashes them: it reports DETERMINISM (all runs same hash) and CORRECTNESS
 * (max abs diff vs a CPU reference).  Configs cover the single-all path
 * (ds4_gpu_attention_decode_heads_tensor, ratio=0) and the mixed/compressed
 * path (ds4_gpu_attention_decode_mixed_batch_heads_tensor, ratio 4/128) — the
 * latter is what the real model uses and is where the subgroup kernel was
 * observed non-deterministic.
 *
 * Keep in vulkan/tools/attn_det/ (backend-private).  Link with the Vulkan
 * backend objects: ds4_vulkan.o ds4_vulkan_compat.o ds4_vulkan_unavailable.o
 * -lvulkan -lm.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "ds4_gpu.h"

/* The synthetic model buffer (attention sinks at offset 0). */
static const void *g_det_model = NULL;
static uint64_t g_det_model_size = 0;

static uint32_t fhash(const float *v, uint32_t n) {
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &v[i], 4);
        h ^= bits;
        h *= 16777619u;
    }
    return h;
}

static void fill(float *v, uint32_t n, uint32_t seed) {
    for (uint32_t i = 0; i < n; i++) {
        uint32_t r = seed ^ (i * 2654435761u) ^ (i >> 3);
        v[i] = ((float)(r % 1000003u) / 1000003.0f - 0.5f) * 2.0f;
    }
}

/* Q8_0 block (34 bytes: f16 scale + 32 int8) into dst. */
static void quant_q8_block(uint8_t *dst, const float *src, uint32_t n) {
    for (uint32_t b = 0; b < (n + 31u) / 32u; b++) {
        uint32_t i0 = b * 32u;
        uint32_t bn = (n - i0) < 32u ? (n - i0) : 32u;
        float amax = 0.0f;
        for (uint32_t i = 0; i < bn; i++) {
            float a = fabsf(src[i0 + i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        /* f16 scale */
        uint16_t h = 0;
        {
            float s = d;
            uint32_t bits;
            memcpy(&bits, &s, 4);
            uint32_t sign = (bits >> 16) & 0x8000u;
            int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
            uint32_t man = bits & 0x7fffffu;
            if (exp <= 0) { h = (uint16_t)sign; }
            else if (exp >= 31) { h = (uint16_t)(sign | 0x7c00u); }
            else { h = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13)); }
        }
        memcpy(dst + b * 34u, &h, 2);
        int8_t *qs = (int8_t *)(dst + b * 34u + 2);
        for (uint32_t i = 0; i < bn; i++) {
            int32_t q = (int32_t)roundf(src[i0 + i] * id);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            qs[i] = (int8_t)q;
        }
    }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* CPU reference: single-token, mixed raw+comp, ratio-aware (port of the smoke
 * test cpu_attn_decode plus the ratio path). */
static void cpu_attn_decode(float *heads, const float *sinks, const float *q,
                            const float *raw_kv, uint32_t n_raw,
                            uint32_t raw_cap, uint32_t raw_start,
                            const float *comp_kv, uint32_t n_comp,
                            const float *comp_mask, uint32_t use_mask,
                            uint32_t pos0, uint32_t window, uint32_t ratio,
                            uint32_t n_head, uint32_t head_dim) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    const uint32_t n_tokens = 1u;
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = (ratio != 0u)
        ? ((n_comp ? (qpos + 1u) / ratio : 0u))
        : n_comp;
    if (visible_comp > n_comp) visible_comp = n_comp;
    uint32_t raw_count = 0, raw_first_idx = 0;
    if (n_raw != 0u) {
        const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
        if (ratio == 0u) {
            raw_count = n_raw < 256u ? n_raw : 256u;
        } else if (qpos >= first_raw_pos) {
            uint32_t lo = first_raw_pos;
            if (window != 0u && qpos + 1u > window) {
                const uint32_t wlo = qpos + 1u - window;
                if (wlo > lo) lo = wlo;
            }
            const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
            if (hi >= lo) {
                raw_first_idx = lo - first_raw_pos;
                raw_count = hi - lo + 1u;
                if (raw_count > 256u) raw_count = 256u;
            }
        }
    }
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + (uint64_t)h * head_dim;
        float *oh = heads + (uint64_t)h * head_dim;
        float scores[2048];
        uint32_t n_score = raw_count + visible_comp;
        float local_max = sinks[h];
        for (uint32_t r = 0; r < raw_count; r++) {
            const uint32_t row = (raw_start + raw_first_idx + r) % raw_cap;
            const float *kvrow = raw_kv + (uint64_t)row * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float add = use_mask ? comp_mask[c] : 0.0f;
            float s = -3.402823466e38f;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                s = dot * scale + add;
            }
            scores[raw_count + c] = s;
            local_max = fmaxf(local_max, s);
        }
        float den = 0.0f;
        for (uint32_t i = 0; i < n_score; i++) {
            scores[i] = expf(scores[i] - local_max);
            den += scores[i];
        }
        den += expf(sinks[h] - local_max);
        for (uint32_t d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) {
                const uint32_t row = (raw_start + raw_first_idx + r) % raw_cap;
                acc += raw_kv[(uint64_t)row * head_dim + d] * scores[r];
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
            }
            oh[d] = acc / den;
        }
    }
}

typedef struct {
    const char *name;
    int mixed;                 /* 0 = single_all, 1 = mixed (ratio) */
    uint32_t n_head, head_dim, n_raw, raw_cap, raw_start;
    uint32_t n_comp, window, ratio, pos0;
    int use_mask;
} cfg_t;

static const cfg_t configs[] = {
    { "single-even",   0, 4u, 32u, 8u, 16u, 5u, 3u, 0u, 0u, 0u, 1 },
    { "single-partial",0, 8u, 32u, 10u, 16u, 5u, 5u, 0u, 0u, 0u, 1 },
    { "mixed-ratio4",  1, 64u, 512u, 128u, 128u, 0u, 64u, 128u, 4u, 200u, 1 },
    { "mixed-ratio128",1, 64u, 512u, 128u, 128u, 0u, 256u, 128u, 128u, 400u, 1 },
    { "mixed-uneven",  1, 32u, 128u, 37u, 128u, 0u, 9u, 128u, 4u, 100u, 1 },
    { "mixed-big",     1, 64u, 64u, 30u, 64u, 0u, 2u, 64u, 128u, 300u, 0 },
};

static int run_det(const cfg_t *c, int n_iter) {
    const uint64_t qb = (uint64_t)c->n_head * c->head_dim * sizeof(float);
    const uint64_t rb = (uint64_t)c->raw_cap * c->head_dim * sizeof(float);
    const uint64_t cb = (uint64_t)c->n_comp * c->head_dim * sizeof(float);
    const uint64_t mb = (uint64_t)c->n_comp * sizeof(float);
    const uint64_t hb = (uint64_t)c->n_head * c->head_dim * sizeof(float);

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(qb);
    ds4_gpu_tensor *raw_kv = ds4_gpu_tensor_alloc(rb);
    ds4_gpu_tensor *comp_kv = ds4_gpu_tensor_alloc(cb);
    ds4_gpu_tensor *comp_mask = ds4_gpu_tensor_alloc(mb);
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(hb);
    if (!q || !raw_kv || !comp_kv || !comp_mask || !heads) return -1;

    uint32_t seed = (uint32_t)(c - configs) * 977u + 13u;
    float *qv = malloc(qb), *rv = malloc(rb), *cv = malloc(cb),
          *mv = malloc(mb), *hv = malloc(hb), *ref = malloc(hb);
    fill(qv, (uint32_t)(qb / 4), seed);
    fill(rv, (uint32_t)(rb / 4), seed + 1);
    fill(cv, (uint32_t)(cb / 4), seed + 2);
    for (uint32_t i = 0; i < c->n_comp; i++)
        mv[i] = (i == 1) ? -1.0e30f : 0.5f * (float)i;
    ds4_gpu_tensor_write(q, 0, qv, qb);
    ds4_gpu_tensor_write(raw_kv, 0, rv, rb);
    ds4_gpu_tensor_write(comp_kv, 0, cv, cb);
    ds4_gpu_tensor_write(comp_mask, 0, mv, mb);

    /* The model map carries the sinks at offset 0 (see main). */
    cpu_attn_decode(ref, (const float *)g_det_model, qv, rv, c->n_raw,
                    c->raw_cap, c->raw_start, cv, c->n_comp, mv,
                    c->use_mask, c->pos0, c->window, c->ratio,
                    c->n_head, c->head_dim);

    uint32_t first_hash = 0;
    int all_equal = 1;
    float maxdiff = 0.0f;
    for (int it = 0; it < n_iter; it++) {
        memset(hv, 0, hb);
        int ok;
        if (c->mixed) {
            ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                    heads, g_det_model, g_det_model_size, 0u, q, raw_kv,
                    comp_kv, 0, c->use_mask ? comp_mask : NULL, c->use_mask,
                    1u, c->pos0, c->n_raw, c->raw_cap, c->raw_start,
                    c->n_comp, c->window, c->ratio, c->n_head, c->head_dim);
        } else {
            ok = ds4_gpu_attention_decode_heads_tensor(
                    heads, g_det_model, g_det_model_size, 0u, q, raw_kv,
                    c->n_raw, c->raw_cap, c->raw_start,
                    comp_kv, 0, c->n_comp,
                    c->use_mask ? comp_mask : NULL, c->use_mask,
                    c->n_head, c->head_dim);
        }
        if (ok == 0) { fprintf(stderr, "attn_det: kernel failed (%s)\n", c->name); return -1; }
        if (ds4_gpu_synchronize() == 0) return -1;
        ds4_gpu_tensor_read(heads, 0, hv, hb);
        uint32_t hsh = fhash(hv, (uint32_t)(hb / 4));
        if (it == 0) first_hash = hsh;
        else if (hsh != first_hash) all_equal = 0;
        for (uint32_t i = 0; i < hb / 4; i++) {
            float d = fabsf(hv[i] - ref[i]);
            if (d > maxdiff) maxdiff = d;
        }
    }

    printf("%-16s %-7s n_head=%u dim=%u n_raw=%u n_comp=%u ratio=%u mask=%d "
           "hash=%08x DET=%s maxdiff_vs_cpu=%.6f %s\n",
           c->name, c->mixed ? "mixed" : "single", c->n_head, c->head_dim,
           c->n_raw, c->n_comp, c->ratio, c->use_mask, first_hash,
           all_equal ? "OK" : "FAIL", maxdiff,
           maxdiff < 2e-3f ? "(correct)" : "(WRONG!)");

    free(qv); free(rv); free(cv); free(mv); free(hv); free(ref);
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(raw_kv);
    ds4_gpu_tensor_free(comp_kv); ds4_gpu_tensor_free(comp_mask);
    ds4_gpu_tensor_free(heads);
    return all_equal && (maxdiff < 2e-3f) ? 0 : 1;
}

/* --- performance harness: time each attention kernel with realistic shapes --
 * The model buffer (set in main) carries:
 *   0        : sinks (max_sinks floats)
 *   OUT_A_OFF: out_a Q8_0 weight for attn_output_low
 *   MATMUL_OFF: Q8_0 weight for matmul_q8_0 (in_dim x out_dim)
 */
#define MODEL_SIZE   (8u * 1024u * 1024u)
#define OUT_A_OFF    (1u * 1024u * 1024u)
#define MATMUL_OFF   (2u * 1024u * 1024u)

static void bench_all(int n_iter) {
    /* attn_decode mixed (n_head=64, head_dim=512, raw_cap=128, ratio=4).
     * Run each variant (serial/v2/v3) at n_tokens=1 (decode, latency-bound) and
     * n_tokens=128 (prefill/batch, bandwidth-bound, where v2's coalesced reads
     * should show) so the tool can distinguish the three kernels. */
    for (int pass = 0; pass < 2; pass++) {
        const uint32_t n_tokens = pass == 0 ? 1u : 128u;
        const char *label = pass == 0 ? "decode nt=1" : "prefill nt=128";
        const uint32_t n_head = 64u, head_dim = 512u, raw_cap = 128u,
                       n_raw = 128u, n_comp = 64u, window = 128u, ratio = 4u,
                       pos0 = 200u;
        const uint64_t qb = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
        const uint64_t rb = (uint64_t)raw_cap * head_dim * sizeof(float);
        const uint64_t cb = (uint64_t)n_comp * head_dim * sizeof(float);
        ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(qb);
        ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(rb);
        ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(cb);
        ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(qb);
        float *qv = malloc(qb), *rv = malloc(rb), *cv = malloc(cb);
        if (!q || !raw || !comp || !heads || !qv || !rv || !cv) return;
        fill(qv, (uint32_t)(qb / 4), 1); fill(rv, (uint32_t)(rb / 4), 2);
        fill(cv, (uint32_t)(cb / 4), 3);
        ds4_gpu_tensor_write(q, 0, qv, qb);
        ds4_gpu_tensor_write(raw, 0, rv, rb);
        ds4_gpu_tensor_write(comp, 0, cv, cb);
        for (int i = 0; i < 3; i++) {
            ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                heads, g_det_model, g_det_model_size, 0u, q, raw, comp, 0,
                NULL, 0, n_tokens, pos0, n_raw, raw_cap, 0u, n_comp, window,
                ratio, n_head, head_dim);
            ds4_gpu_synchronize();
        }
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++) {
            ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                heads, g_det_model, g_det_model_size, 0u, q, raw, comp, 0,
                NULL, 0, n_tokens, pos0, n_raw, raw_cap, 0u, n_comp, window,
                ratio, n_head, head_dim);
        }
        ds4_gpu_synchronize();
        printf("bench attn_decode   %-14s: %.4f ms/call\n",
               label, (now_ms() - t0) / (double)n_iter);
        free(qv); free(rv); free(cv);
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(raw);
        ds4_gpu_tensor_free(comp); ds4_gpu_tensor_free(heads);
    }

    /* attn_output_low (heads @ out_a^T, group_dim=512 rank=64 n_groups=8). */
    {
        const uint32_t group_dim = 512u, rank = 64u, n_groups = 8u, n_rows = 1u;
        const uint32_t low_dim = n_groups * rank;
        const uint64_t hb = (uint64_t)n_rows * n_groups * group_dim * sizeof(float);
        const uint64_t lb = (uint64_t)n_rows * low_dim * sizeof(float);
        ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(hb);
        ds4_gpu_tensor *low = ds4_gpu_tensor_alloc(lb);
        float *hvv = malloc(hb);
        fill(hvv, (uint32_t)(hb / 4), 7);
        ds4_gpu_tensor_write(heads, 0, hvv, hb);
        for (int i = 0; i < 3; i++) {
            ds4_gpu_attention_output_low_q8_tensor(
                low, g_det_model, g_det_model_size, OUT_A_OFF, group_dim, rank,
                n_groups, heads);
            ds4_gpu_synchronize();
        }
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++) {
            ds4_gpu_attention_output_low_q8_tensor(
                low, g_det_model, g_det_model_size, OUT_A_OFF, group_dim, rank,
                n_groups, heads);
        }
        ds4_gpu_synchronize();
        printf("bench attn_out_low  low_q8          : %.4f ms/call\n",
               (now_ms() - t0) / (double)n_iter);
        free(hvv);
        ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(low);
    }

    /* matmul_q8_0 (in_dim=4096 out_dim=1024). */
    {
        const uint32_t in_dim = 4096u, out_dim = 1024u;
        const uint64_t xb = (uint64_t)in_dim * sizeof(float);
        const uint64_t ob = (uint64_t)out_dim * sizeof(float);
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(xb);
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(ob);
        float *xv = malloc(xb);
        fill(xv, in_dim, 11);
        ds4_gpu_tensor_write(x, 0, xv, xb);
        for (int i = 0; i < 3; i++) {
            ds4_gpu_matmul_q8_0_tensor(out, g_det_model, g_det_model_size,
                                       MATMUL_OFF, in_dim, out_dim, x, 1u);
            ds4_gpu_synchronize();
        }
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++) {
            ds4_gpu_matmul_q8_0_tensor(out, g_det_model, g_det_model_size,
                                       MATMUL_OFF, in_dim, out_dim, x, 1u);
        }
        ds4_gpu_synchronize();
        printf("bench matmul_q8_0   (%ux%u)          : %.4f ms/call\n",
               in_dim, out_dim, (now_ms() - t0) / (double)n_iter);
        free(xv);
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out);
    }

    /* rms_norm_weight (n=4096). */
    {
        const uint32_t n = 4096u;
        const uint64_t b = (uint64_t)n * sizeof(float);
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(b);
        ds4_gpu_tensor *o = ds4_gpu_tensor_alloc(b);
        float *xv = malloc(b);
        fill(xv, n, 17);
        ds4_gpu_tensor_write(x, 0, xv, b);
        for (int i = 0; i < 3; i++) {
            ds4_gpu_rms_norm_weight_tensor(o, x, g_det_model, g_det_model_size,
                                           0u, n, 1e-6f);
            ds4_gpu_synchronize();
        }
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++) {
            ds4_gpu_rms_norm_weight_tensor(o, x, g_det_model, g_det_model_size,
                                           0u, n, 1e-6f);
        }
        ds4_gpu_synchronize();
        printf("bench rms_norm_wt   (n=4096)         : %.4f ms/call\n",
               (now_ms() - t0) / (double)n_iter);
        free(xv);
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(o);
    }

    /* rope_tail (n_head=64 head_dim=512 n_rot=64). */
    {
        const uint32_t n_head = 64u, head_dim = 512u, n_rot = 64u;
        const uint64_t b = (uint64_t)n_head * head_dim * sizeof(float);
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(b);
        float *xv = malloc(b);
        fill(xv, (uint32_t)(b / 4), 23);
        ds4_gpu_tensor_write(x, 0, xv, b);
        for (int i = 0; i < 3; i++) {
            ds4_gpu_rope_tail_tensor(x, 1u, n_head, head_dim, n_rot, 100u, 0u,
                                     false, 10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
            ds4_gpu_synchronize();
        }
        double t0 = now_ms();
        for (int i = 0; i < n_iter; i++) {
            ds4_gpu_rope_tail_tensor(x, 1u, n_head, head_dim, n_rot, 100u, 0u,
                                     false, 10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        }
        ds4_gpu_synchronize();
        printf("bench rope_tail     (64x512x64)     : %.4f ms/call\n",
               (now_ms() - t0) / (double)n_iter);
        free(xv);
        ds4_gpu_tensor_free(x);
    }
}

int main(void) {
    if (ds4_gpu_init() == 0) { fprintf(stderr, "attn_det: ds4_gpu_init failed\n"); return 1; }

    const uint32_t max_sinks = 64u;
    uint8_t *model = NULL;
    if (posix_memalign((void **)&model, 4096, MODEL_SIZE) != 0) return 1;
    memset(model, 0, MODEL_SIZE);
    float *sinkv = (float *)model;
    for (uint32_t h = 0; h < max_sinks; h++) sinkv[h] = 0.02f * (float)h - 0.1f;
    /* Synthetic attention weights (Q8_0): out_a (attn_output_low) and a matmul
     * weight, so the bench kernels read real-ish data. */
    {
        /* out_a: low_dim(=n_groups*rank=512) x group_dim(512) Q8_0. */
        const uint32_t group_dim = 512u, rank = 64u, n_groups = 8u;
        const uint32_t out_dim = n_groups * rank;
        const uint32_t blocks = (group_dim + 31u) / 32u;
        float *w = malloc((uint64_t)out_dim * group_dim * sizeof(float));
        fill(w, out_dim * group_dim, 101u);
        for (uint32_t r = 0; r < out_dim; r++)
            quant_q8_block(model + OUT_A_OFF + (uint64_t)r * blocks * 34u,
                           w + (uint64_t)r * group_dim, group_dim);
        free(w);
    }
    {
        /* matmul weight: out_dim(1024) x in_dim(4096) Q8_0. */
        const uint32_t in_dim = 4096u, out_dim = 1024u;
        const uint32_t blocks = (in_dim + 31u) / 32u;
        float *w = malloc((uint64_t)out_dim * in_dim * sizeof(float));
        fill(w, out_dim * in_dim, 202u);
        for (uint32_t r = 0; r < out_dim; r++)
            quant_q8_block(model + MATMUL_OFF + (uint64_t)r * blocks * 34u,
                           w + (uint64_t)r * in_dim, in_dim);
        free(w);
    }
    if (ds4_gpu_set_model_map(model, MODEL_SIZE) == 0) {
        fprintf(stderr, "attn_det: ds4_gpu_set_model_map failed\n"); return 1;
    }
    g_det_model = model;
    g_det_model_size = MODEL_SIZE;

    const int n_iter = 16;
    const uint32_t ncfg = sizeof(configs) / sizeof(configs[0]);
    int fail = 0;
    for (uint32_t ci = 0; ci < ncfg; ci++) {
        if (run_det(&configs[ci], n_iter) != 0) fail++;
    }
    printf("attn_det: %u/%u configs OK; %s\n", ncfg - fail, ncfg,
           fail == 0 ? "PASSED" : "FAILED");

    if (getenv("ATTN_DET_BENCH") != NULL) {
        bench_all(100);
    }

    free(model);
    return fail == 0 ? 0 : 1;
}
