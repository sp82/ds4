/* vulkan/shaders/qwen4_mtp.hlsl -- Qwen3.8-Flash-Next multi-token predictor
 * (nextn / MTP) support kernels.
 *
 * Ports the Metal reference (metal/qwen4.metal):
 *   - qwen4_argmax: two-pass argmax over n_vocab F32 logits, writing an int32
 *     token id (NaN skipped, +-Inf kept, ties to the lower index).  Pass 1
 *     reduces independent 4096-value chunks into uint2{value_bits, index}
 *     partials; pass 2 merges the chunk winners.
 *   - mtp_stage: builds the concat rows for the fused [W_e | W_h] projection.
 *     Row 0 is [rms(e)*g_e | 0]; row 1+s is [0 | R_s/rms(R)*g_h_s] with one
 *     RMS over all hc streams.  One threadgroup per row.
 *   - mtp_combine: R_out[s][d] = proj[0][d] + proj[1+s][d].
 *
 * Bindings (see ds4_vulkan_internal.h): A=t0, B=t1, C=t2, W=t3, OUT=u0,
 * OUT2=u1.  All model weights arrive as ByteAddressBuffer ranges (f32 here). */

#include "common.hlsl"

[[vk::push_constant]] DS4Params params;

/* --- predictor argmax ---------------------------------------------------- */

StructuredBuffer<float> am_logits_buf : register(t0);
RWStructuredBuffer<int> am_out_buf : register(u0);
RWStructuredBuffer<uint2> am_partials_buf : register(u1);

groupshared float am_val[256];
groupshared uint am_idx[256];

/* params.n = value count (pass 1) or chunk count (pass 2);
 * params.flags bit0 = finish (merge the partials into out_idx). */
[numthreads(256, 1, 1)]
void qwen4_argmax(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    const bool finish = (params.flags & 1u) != 0u;
    const uint begin = finish ? 0u : gid_grp.x * 4096u;
    const uint end = finish ? params.n : min(begin + 4096u, params.n);
    float best = -1.0e30f;
    uint index = 0u;
    for (uint i = begin + tid; i < end; i += 256u) {
        const uint2 p = finish ? am_partials_buf[i]
                               : uint2(asuint(am_logits_buf[i]), i);
        if ((p.x & 0x7fffffffu) > 0x7f800000u) continue;   /* skip NaN */
        const float v = asfloat(p.x);
        if (v > best || (v == best && p.y < index)) { best = v; index = p.y; }
    }
    am_val[tid] = best;
    am_idx[tid] = index;
    GroupMemoryBarrierWithGroupSync();
    for (uint st = 128u; st > 0u; st >>= 1u) {
        if (tid < st) {
            const float vb = am_val[tid + st];
            const uint ib = am_idx[tid + st];
            if (vb > am_val[tid] || (vb == am_val[tid] && ib < am_idx[tid])) {
                am_val[tid] = vb;
                am_idx[tid] = ib;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        if (finish) am_out_buf[0] = (int)am_idx[0];
        else am_partials_buf[gid_grp.x] = uint2(asuint(am_val[0]), am_idx[0]);
    }
}

/* --- MTP input staging --------------------------------------------------- */

StructuredBuffer<float> mtp_e_buf : register(t0);   /* [E] next-token embedding */
StructuredBuffer<float> mtp_R_buf : register(t1);   /* [hc*E] pre-mixer streams */
ByteAddressBuffer mtp_ge_buf : register(t2);        /* [E] f32 enorm */
ByteAddressBuffer mtp_gh_buf : register(t3);        /* [hc*E] f32 hnorm */
RWStructuredBuffer<float> mtp_cat_buf : register(u0); /* [1+hc][2E] */

groupshared float mtp_red[256];

/* params.n = n_embd (E); params.index = n_hc; params.eps = rms eps. */
[numthreads(256, 1, 1)]
void mtp_stage(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    const uint row = gid_grp.x;
    const uint E = params.n;
    const uint hc = params.index;
    if (row > hc) return;
    const bool emb = (row == 0u);
    const uint n_red = emb ? E : E * hc;
    float ss = 0.0f;
    if (emb) {
        for (uint i = tid; i < E; i += 256u) {
            const float v = mtp_e_buf[i];
            ss += v * v;
        }
    } else {
        for (uint i = tid; i < n_red; i += 256u) {
            const float v = mtp_R_buf[i];
            ss += v * v;
        }
    }
    mtp_red[tid] = ss;
    GroupMemoryBarrierWithGroupSync();
    for (uint st = 128u; st > 0u; st >>= 1u) {
        if (tid < st) mtp_red[tid] += mtp_red[tid + st];
        GroupMemoryBarrierWithGroupSync();
    }
    const float inv = rsqrt(mtp_red[0] / (float)n_red + params.eps);
    const uint src_base = emb ? 0u : (row - 1u) * E;
    const uint base = row * 2u * E;
    const uint lo = emb ? 0u : E;
    const uint hi = emb ? E : 0u;
    for (uint i = tid; i < E; i += 256u) {
        const float s = emb ? mtp_e_buf[i] : mtp_R_buf[src_base + i];
        const float g = emb ? asfloat(mtp_ge_buf.Load(i * 4u))
                            : asfloat(mtp_gh_buf.Load((src_base + i) * 4u));
        mtp_cat_buf[base + lo + i] = s * inv * g;
        mtp_cat_buf[base + hi + i] = 0.0f;
    }
}

/* --- MTP projection combine ---------------------------------------------- */

StructuredBuffer<float> mtp_proj_buf : register(t0);  /* [1+hc][E] */
RWStructuredBuffer<float> mtp_rout_buf : register(u0); /* [hc*E] */

/* params.n = n_embd (E); params.index = n_hc. */
[numthreads(256, 1, 1)]
void mtp_combine(uint3 gid : SV_DispatchThreadID) {
    const uint E = params.n;
    const uint n = E * params.index;
    if (gid.x >= n) return;
    const uint s = gid.x / E;
    const uint d = gid.x - s * E;
    mtp_rout_buf[gid.x] = mtp_proj_buf[d] + mtp_proj_buf[(s + 1u) * E + d];
}
