/* vulkan/shaders/compressor.hlsl -- DS4 KV context compressor (Phase 5).
 *
 * Pooled compression of the KV context: store raw rows into a small rolling
 * state, then softmax-pool candidate rows into compressed rows at ratio
 * boundaries.  Parity with the CUDA kernels:
 *
 *   compressor_store      -- state[dst_row][j] = kv[t][j], score + ape.
 *   compressor_set_rows   -- copy rows [src0, src0+rows) into state at dst0.
 *   compressor_prefill_pool -- pool candidates into comp[c] (per dim).
 *   compressor_update_pool -- pool the state into one comp row (decode emit).
 *   compressor_shift_ratio4 -- shift the 8-row state down by 4 (duplicate).
 *
 * The ape weight tensor is read from the model buffer (binding 3) as f32
 * (type 0) or f16 (type 1) via model_scalar().  Params per entry below.
 *
 * Bindings: t0 kv, t1 sc, t3 model ape, u0 state_kv, u1 state_score,
 * u2 comp_cache / comp row.
 */

#include "common.hlsl"

StructuredBuffer<float> kv_buf : register(t0);
StructuredBuffer<float> sc_buf : register(t1);
ByteAddressBuffer ape_buf : register(t3);
RWStructuredBuffer<float> st_kv_buf : register(u0);
RWStructuredBuffer<float> st_sc_buf : register(u1);
RWStructuredBuffer<float> comp_buf : register(u2);

[[vk::push_constant]] DS4Params params;

/* state[ratio + pos_mod][j] = kv[t][j]; score += ape[pos_mod][j].
 * params: n=head_dim, rows=n_tokens, ratio=ratio, pos0=pos0,
 * index=ape_type. */
[numthreads(256, 1, 1)]
void compressor_store(uint3 gid : SV_DispatchThreadID) {
    uint coff = params.ratio == 4u ? 2u : 1u;
    uint width = coff * params.n;
    uint n = params.rows * width;
    if (gid.x >= n) return;
    uint t = gid.x / width;
    uint j = gid.x - t * width;
    uint pos_mod = (params.pos0 + t) % params.ratio;
    /* Several tokens in the batch can map to the same state row (their
     * pos_mod collide).  The sequential reference is last-write-wins, so only
     * the last token for each pos_mod writes; otherwise the concurrent writes
     * race (observed on NVIDIA).  t_last = largest t < rows with
     * (pos0 + t) % ratio == pos_mod. */
    uint t0 = (pos_mod + params.ratio - (params.pos0 % params.ratio)) %
              params.ratio;
    uint t_last = t0 + params.ratio *
                  ((params.rows - 1u - t0) / params.ratio);
    if (t != t_last) return;
    uint dst_row = params.ratio == 4u ? params.ratio + pos_mod : pos_mod;
    st_kv_buf[dst_row * width + j] = kv_buf[t * width + j];
    st_sc_buf[dst_row * width + j] =
        sc_buf[t * width + j] +
        model_scalar(ape_buf, 0u, params.index, pos_mod * width + j);
}

/* Copy rows [src0, src0+rows) of kv/sc into state at dst0 (used by the
 * prefill state construction).  params: n=width, rows=rows, ratio=ratio,
 * pos0=pos0, index=ape_type, aux=src0, blocks=dst0. */
[numthreads(256, 1, 1)]
void compressor_set_rows(uint3 gid : SV_DispatchThreadID) {
    uint n = params.rows * params.n;
    if (gid.x >= n) return;
    uint r = gid.x / params.n;
    uint j = gid.x - r * params.n;
    uint src = params.aux + r;
    uint dst = params.blocks + r;
    uint phase = (params.pos0 + src) % params.ratio;
    st_kv_buf[dst * params.n + j] = kv_buf[src * params.n + j];
    st_sc_buf[dst * params.n + j] =
        sc_buf[src * params.n + j] +
        model_scalar(ape_buf, 0u, params.index, phase * params.n + j);
}

/* Softmax-pool candidates into comp[c][d].  One block per head_dim group,
 * one comp row per blockIdx.y.  params: n=head_dim, index=ratio,
 * pos0=pos0, aux=ape_type, flags=replay. */
[numthreads(256, 1, 1)]
void compressor_prefill_pool(uint3 bid : SV_GroupID,
                             uint tid : SV_GroupThreadID) {
    uint d = bid.x * 256u + tid;
    uint c = bid.y;
    uint head_dim = params.n;
    uint ratio = params.index;
    if (d >= head_dim) return;
    uint coff = ratio == 4u ? 2u : 1u;
    uint width = coff * head_dim;
    float vals[16];
    float scores[16];
    float max_s = -1.0e30f;
    uint n_cand = 0u;
    if (ratio == 4u) {
        if ((params.flags & 1u) != 0u && c == 0u) {
            for (uint r = 0u; r < 4u; r++) {
                vals[n_cand] = st_kv_buf[r * width + d];
                scores[n_cand] = st_sc_buf[r * width + d];
                max_s = max(max_s, scores[n_cand++]);
            }
        } else if (c > 0u) {
            uint base = (c - 1u) * ratio;
            for (uint r = 0u; r < 4u; r++) {
                uint t = base + r;
                float ape = model_scalar(ape_buf, 0u, params.aux,
                                         ((uint64_t)(params.pos0 + t) % ratio) * width + d);
                vals[n_cand] = kv_buf[t * width + d];
                scores[n_cand] = sc_buf[t * width + d] + ape;
                max_s = max(max_s, scores[n_cand++]);
            }
        }
        uint base = c * ratio;
        for (uint r = 0u; r < 4u; r++) {
            uint t = base + r;
            float ape = model_scalar(ape_buf, 0u, params.aux,
                                     ((uint64_t)(params.pos0 + t) % ratio) * width + head_dim + d);
            vals[n_cand] = kv_buf[t * width + head_dim + d];
            scores[n_cand] = sc_buf[t * width + head_dim + d] + ape;
            max_s = max(max_s, scores[n_cand++]);
        }
    } else {
        uint base = c * ratio;
        for (uint r = 0u; r < ratio; r++) {
            uint t = base + r;
            float ape = model_scalar(ape_buf, 0u, params.aux,
                                     ((uint64_t)(params.pos0 + t) % ratio) * width + d);
            vals[n_cand] = kv_buf[t * width + d];
            scores[n_cand] = sc_buf[t * width + d] + ape;
            max_s = max(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint i = 0u; i < n_cand; i++) {
        float w = exp(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    comp_buf[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
}

/* Pool the current state into one compressed row (decode emit).  One thread
 * per dimension.  params: n=head_dim, index=ratio. */
[numthreads(256, 1, 1)]
void compressor_update_pool(uint3 gid : SV_DispatchThreadID) {
    uint d = gid.x;
    uint head_dim = params.n;
    uint ratio = params.index;
    if (d >= head_dim) return;
    uint coff = ratio == 4u ? 2u : 1u;
    uint width = coff * head_dim;
    float vals[16];
    float scores[16];
    float max_s = -1.0e30f;
    uint n_cand = 0u;
    if (ratio == 4u) {
        for (uint r = 0u; r < 4u; r++) {
            vals[n_cand] = st_kv_buf[r * width + d];
            scores[n_cand] = st_sc_buf[r * width + d];
            max_s = max(max_s, scores[n_cand++]);
        }
        for (uint r = 0u; r < 4u; r++) {
            vals[n_cand] = st_kv_buf[(ratio + r) * width + head_dim + d];
            scores[n_cand] = st_sc_buf[(ratio + r) * width + head_dim + d];
            max_s = max(max_s, scores[n_cand++]);
        }
    } else {
        for (uint r = 0u; r < ratio; r++) {
            vals[n_cand] = st_kv_buf[r * width + d];
            scores[n_cand] = st_sc_buf[r * width + d];
            max_s = max(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint i = 0u; i < n_cand; i++) {
        float w = exp(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    comp_buf[d] = den != 0.0f ? acc / den : 0.0f;
}

/* Shift the 8-row ratio4 state down by 4 (duplicating rows 4..7 into 0..3).
 * params: n=width. */
[numthreads(256, 1, 1)]
void compressor_shift_ratio4(uint3 gid : SV_DispatchThreadID) {
    uint half = 4u * params.n;
    if (gid.x >= half) return;
    float v = st_kv_buf[half + gid.x];
    float s = st_sc_buf[half + gid.x];
    st_kv_buf[gid.x] = v;
    st_sc_buf[gid.x] = s;
    st_kv_buf[half + gid.x] = v;
    st_sc_buf[half + gid.x] = s;
}
