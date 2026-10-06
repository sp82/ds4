/* vulkan/shaders/qwen4_ple.hlsl -- Qwen3.8-Flash-Next PLE (n-gram).
 *
 * ple_gate: per (stream, token) RMSNorm of key and R, gated dot, sign-sqrt
 * gate, then RMSNorm of the gated value (CUDA ngram_gate).
 * ple_conv: dilated causal conv (H=(K-1)*dilation <= 8) that adds the gated
 * residual and silu(conv) back into R, with optional snapshots.
 *
 * Bindings: ple_gate R/key/val (t0..t2) + gk/gq/gc (t10..t12), gated (u0),
 * normed (u1); ple_conv gated (t0), w (t2), R (u0), history (u1),
 * snap (u2), snap2 (u3).
 */

#include "common.hlsl"

StructuredBuffer<float> a_buf : register(t0);
StructuredBuffer<float> b_buf : register(t1);
StructuredBuffer<float> c_buf : register(t2);
ByteAddressBuffer ple_w_buf : register(t3);
ByteAddressBuffer gk_buf : register(t10);
ByteAddressBuffer gq_buf : register(t11);
ByteAddressBuffer gc_buf : register(t12);
RWStructuredBuffer<float> out_buf : register(u0);
RWStructuredBuffer<float> out2_buf : register(u1);
RWStructuredBuffer<float> out3_buf : register(u2);
RWStructuredBuffer<float> out4_buf : register(u3);

[[vk::push_constant]] DS4Params params;

groupshared float ple_red[128];

float qwen4_sigmoid(float x) {
    const float e = exp(-abs(x));
    return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

float qwen4_silu(float x) { return x * qwen4_sigmoid(x); }

float qwen4_dense_scalar(ByteAddressBuffer w, uint idx, uint type) {
    if (type == 0u) return asfloat(w.Load(idx * 4u));
    if (type == 1u) return f16_at(w, idx * 2u);
    if (type == 30u) return bf16_at(w, idx * 2u);
    if (type == 8u) {
        uint block = idx >> 5u, i = idx & 31u, base = block * 34u;
        return q8_scale_at(w, base) * (float)q8_s8_at(w, base, i);
    }
    uint block = idx >> 5u, j = idx & 31u, base = block * 18u;
    uint nib = (q8_byte_at(w, base + 2u, j & 15u) >> (4u * (j >> 4u))) & 15u;
    return f16_at(w, base) * ((float)nib - 8.0f);
}

/* params: rows=T, in_dim=E, out_dim=hc, eps. */
[numthreads(128, 1, 1)]
void ple_gate(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint s = gid_grp.x, t = gid_grp.y;
    uint E = params.in_dim, hc = params.out_dim;
    if (s >= hc || t >= params.rows) return;
    const uint base = (t * hc + s) * E;

    float sk = 0.0f, sq = 0.0f;
    for (uint i = tid; i < E; i += 128u) {
        sk += b_buf[base + i] * b_buf[base + i];
        sq += a_buf[base + i] * a_buf[base + i];
    }
    ple_red[tid] = sk;
    GroupMemoryBarrierWithGroupSync();
    for (uint st = 64u; st > 0u; st >>= 1u) {
        if (tid < st) ple_red[tid] += ple_red[tid + st];
        GroupMemoryBarrierWithGroupSync();
    }
    const float ik = rsqrt(ple_red[0] / (float)E + params.eps);
    ple_red[tid] = sq;
    GroupMemoryBarrierWithGroupSync();
    for (uint st = 64u; st > 0u; st >>= 1u) {
        if (tid < st) ple_red[tid] += ple_red[tid + st];
        GroupMemoryBarrierWithGroupSync();
    }
    const float iq = rsqrt(ple_red[0] / (float)E + params.eps);

    float dot = 0.0f;
    for (uint i = tid; i < E; i += 128u) {
        float k = b_buf[base + i] * ik *
                  asfloat(gk_buf.Load((s * E + i) * 4u));
        float r = a_buf[base + i] * iq *
                  asfloat(gq_buf.Load((s * E + i) * 4u));
        dot += k * r;
    }
    ple_red[tid] = dot;
    GroupMemoryBarrierWithGroupSync();
    for (uint st = 64u; st > 0u; st >>= 1u) {
        if (tid < st) ple_red[tid] += ple_red[tid + st];
        GroupMemoryBarrierWithGroupSync();
    }
    const float a = ple_red[0] * rsqrt((float)E);
    const float mag = sqrt(max(abs(a), 1e-6f));
    const float gate = qwen4_sigmoid(a > 0.0f ? mag : (a < 0.0f ? -mag : 0.0f));

    float ss = 0.0f;
    for (uint i = tid; i < E; i += 128u) {
        float v = gate * c_buf[t * E + i];
        out_buf[base + i] = v;
        ss += v * v;
    }
    ple_red[tid] = ss;
    GroupMemoryBarrierWithGroupSync();
    for (uint st = 64u; st > 0u; st >>= 1u) {
        if (tid < st) ple_red[tid] += ple_red[tid + st];
        GroupMemoryBarrierWithGroupSync();
    }
    const float inv = rsqrt(ple_red[0] / (float)E + params.eps);
    for (uint i = tid; i < E; i += 128u)
        out2_buf[base + i] = out_buf[base + i] * inv *
                             asfloat(gc_buf.Load((s * E + i) * 4u));
}

/* params: n=C, rows=T, in_dim=K, index=dilation, rsvd3=type,
 * aux=snap_t (0xffffffff none), blocks=snap2_t. */
[numthreads(256, 1, 1)]
void ple_conv(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint c = gid_grp.x * 256u + tid;
    uint C = params.n;
    if (c >= C) return;
    uint T = params.rows, K = params.in_dim, dil = params.index;
    uint type = params.rsvd3;
    const uint H = (K - 1u) * dil;
    float hist[9], taps[4];
    for (uint i = 0u; i < H; i++) hist[i] = out2_buf[i * C + c];
    for (uint i = 0u; i < K; i++) taps[i] = qwen4_dense_scalar(ple_w_buf, c * K + i, type);
    for (uint t = 0u; t < T; t++) {
        const uint p = t * C + c;
        const float cur = b_buf[p];
        float v = taps[K - 1u] * cur;
        for (uint i = 0u; i + 1u < K; i++) v += taps[i] * hist[i * dil];
        for (uint i = 0u; i + 1u < H; i++) hist[i] = hist[i + 1u];
        hist[H - 1u] = cur;
        out_buf[p] += a_buf[p] + qwen4_silu(v);
        if (params.aux == t) for (uint i = 0u; i < H; i++) out3_buf[i * C + c] = hist[i];
        if (params.blocks == t) for (uint i = 0u; i < H; i++) out4_buf[i * C + c] = hist[i];
    }
    for (uint i = 0u; i < H; i++) out2_buf[i * C + c] = hist[i];
}
