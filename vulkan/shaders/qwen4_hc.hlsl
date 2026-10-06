/* vulkan/shaders/qwen4_hc.hlsl -- Qwen3.8-Flash-Next hyper-connections.
 *
 * Ports the CUDA reference: hc_norm (per-stream RMSNorm + low-rank injection
 * partials), hc_gate_mix (gate the low-rank projection with sigmoid and mix
 * the streams), hc_combine (inject the block output back into every stream),
 * hc_lo (silu(lo/hc)) and hc_mix_rows (prefill batched mix).  The T=1
 * combine+norm fusion is assembled host-side from hc_combine + hc_norm.
 *
 * Bindings: a (t0), b (t1, ByteAddressBuffer), c (t2), wi (t10,
 * ByteAddressBuffer), out (u0), out2 (u1).
 */

#include "common.hlsl"

StructuredBuffer<float> a_buf : register(t0);
ByteAddressBuffer b_buf : register(t1);
StructuredBuffer<float> c_buf : register(t2);
ByteAddressBuffer wi_buf : register(t10);
RWStructuredBuffer<float> out_buf : register(u0);
RWStructuredBuffer<float> out2_buf : register(u1);

[[vk::push_constant]] DS4Params params;

groupshared float hc_red[128];
groupshared float hc_acc[4][128];

float qwen4_sigmoid(float x) {
    const float e = exp(-abs(x));
    return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

float qwen4_silu(float x) { return x * qwen4_sigmoid(x); }

/* One element of a dense weight row (type 0 f32, 1 f16, 2 q4_0, 8 q8_0,
 * 30 bf16); `idx` is the element index. */
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

/* params: rows=T, in_dim=E, out_dim=hc, index=ni, rsvd3=type, eps. */
[numthreads(128, 1, 1)]
void hc_norm(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint stream = gid_grp.x >> 3u, chunk = gid_grp.x & 7u, tok = gid_grp.y;
    uint E = params.in_dim, hc = params.out_dim, ni = params.index;
    uint type = params.rsvd3;
    if (stream >= hc || tok >= params.rows) return;
    const uint dim = E * hc;
    const uint base = (tok * hc + stream) * E;

    float ss = 0.0f;
    for (uint i = tid; i < E; i += 128u) {
        float v = a_buf[base + i];
        ss += v * v;
    }
    hc_red[tid] = ss;
    GroupMemoryBarrierWithGroupSync();
    for (uint st = 64u; st > 0u; st >>= 1u) {
        if (tid < st) hc_red[tid] += hc_red[tid + st];
        GroupMemoryBarrierWithGroupSync();
    }
    const float inv = rsqrt(hc_red[0] / (float)E + params.eps);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const uint per = (E + 7u) / 8u;
    const uint end = min(E, (chunk + 1u) * per);
    for (uint i = chunk * per + tid; i < end; i += 128u) {
        float v = a_buf[base + i] * inv * asfloat(b_buf.Load((stream * E + i) * 4u));
        out_buf[base + i] = v;
        if (ni > 0u) acc0 += qwen4_dense_scalar(wi_buf, 0u * dim + stream * E + i, type) * v;
        if (ni > 1u) acc1 += qwen4_dense_scalar(wi_buf, 1u * dim + stream * E + i, type) * v;
        if (ni > 2u) acc2 += qwen4_dense_scalar(wi_buf, 2u * dim + stream * E + i, type) * v;
        if (ni > 3u) acc3 += qwen4_dense_scalar(wi_buf, 3u * dim + stream * E + i, type) * v;
    }
    hc_acc[0][tid] = acc0;
    hc_acc[1][tid] = acc1;
    hc_acc[2][tid] = acc2;
    hc_acc[3][tid] = acc3;
    GroupMemoryBarrierWithGroupSync();
    for (uint st = 64u; st > 0u; st >>= 1u) {
        if (tid < st) {
            hc_acc[0][tid] += hc_acc[0][tid + st];
            hc_acc[1][tid] += hc_acc[1][tid + st];
            hc_acc[2][tid] += hc_acc[2][tid + st];
            hc_acc[3][tid] += hc_acc[3][tid + st];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u && ni > 0u) {
        for (uint j = 0u; j < ni; j++)
            out2_buf[(tok * hc * 8u + stream * 8u + chunk) * ni + j] = hc_acc[j][0];
    }
}

/* params: rows=T, in_dim=E, out_dim=hc, index=rank, rsvd3=type. */
[numthreads(256, 1, 1)]
void hc_gate_mix(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint d = gid_grp.x * 256u + tid;
    uint t = gid_grp.y;
    uint E = params.in_dim, hc = params.out_dim, rank = params.index;
    uint type = params.rsvd3;
    if (d >= E || t >= params.rows) return;

    float m = 0.0f;
    for (uint s = 0u; s < hc; s++) {
        float acc = 0.0f;
        for (uint r = 0u; r < rank; r++)
            acc += qwen4_dense_scalar(b_buf, (s * E + d) * rank + r, type) *
                   qwen4_silu(c_buf[t * rank + r] / (float)hc);
        m += qwen4_sigmoid(acc) * a_buf[(t * hc + s) * E + d];
    }
    out_buf[t * E + d] = m / (float)hc;
}

/* params: rows=T, in_dim=E, out_dim=hc, index=hc. */
[numthreads(256, 1, 1)]
void hc_combine(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint d = gid_grp.x * 256u + tid;
    uint t = gid_grp.y;
    uint E = params.in_dim, hc = params.out_dim;
    if (d >= E || t >= params.rows) return;
    for (uint s = 0u; s < hc; s++) {
        float tot = 0.0f;
        for (uint i = 0u; i < hc * 8u; i++)
            tot += c_buf[(uint64_t)t * hc * hc * 8u + i * hc + s];
        const float wgt = 2.0f * qwen4_sigmoid(tot / (float)hc);
        out_buf[((uint64_t)t * hc + s) * E + d] += wgt * a_buf[t * E + d];
    }
}

/* params: n=total, index=hc. */
[numthreads(256, 1, 1)]
void hc_lo(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint i = gid_grp.x * 256u + tid;
    if (i >= params.n) return;
    out_buf[i] = qwen4_silu(a_buf[i] / (float)params.index);
}

/* params: rows=T, in_dim=E, out_dim=hc. */
[numthreads(256, 1, 1)]
void hc_mix_rows(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint d = gid_grp.x * 256u + tid;
    uint t = gid_grp.y;
    uint E = params.in_dim, hc = params.out_dim;
    if (d >= E || t >= params.rows) return;
    float v = 0.0f;
    for (uint s = 0u; s < hc; s++)
        v += qwen4_sigmoid(a_buf[(t * hc + s) * E + d]) * c_buf[(t * hc + s) * E + d];
    out_buf[t * E + d] = v / (float)hc;
}
