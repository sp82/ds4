/* vulkan/shaders/qwen4_router.hlsl -- Qwen3.8-Flash-Next MoE router.
 *
 * One block of 256 threads per token: softmax over the NE expert logits, a
 * strict top-NS selection (ties break to the lower expert id, matching the
 * CUDA reference router), renormalize the selected probabilities, and the
 * shared-expert gate dot sigmoid(shared) is left to moe_reduce.  The shared
 * gate logit is a single dot of the token activation against the one-row
 * ffn_gate_inp_shexp weight.
 *
 * This is NOT the DeepSeek router (router.hlsl): that one uses
 * sqrt(softplus(logit)) with a bias/hash table and a serial insertion sort.
 *
 * Bindings: logits (t0), gate weight (t1, ByteAddressBuffer), x (t2),
 * selected (u0), weights (u1), shared gate (u2).  Scalar params in DS4Params.
 */

#include "common.hlsl"

StructuredBuffer<float> logits_buf : register(t0);
ByteAddressBuffer gate_buf : register(t1);
StructuredBuffer<float> x_buf : register(t2);
RWStructuredBuffer<int> selected_buf : register(u0);
RWStructuredBuffer<float> weights_buf : register(u1);
RWStructuredBuffer<float> shared_buf : register(u2);

[[vk::push_constant]] DS4Params params;

groupshared float sp[512];
groupshared float red[256];
groupshared float keyv[256];
groupshared uint keyid[256];

/* One element of a dense weight row (type 0 f32, 1 f16, 2 q4_0, 8 q8_0,
 * 30 bf16); `idx` is the element index inside the row. */
float qwen4_dense_scalar(ByteAddressBuffer w, uint idx, uint type) {
    if (type == 0u) return asfloat(w.Load(idx * 4u));
    if (type == 1u) return f16_at(w, idx * 2u);
    if (type == 30u) return bf16_at(w, idx * 2u);
    if (type == 8u) {
        uint block = idx >> 5u, i = idx & 31u;
        uint base = block * 34u;
        return q8_scale_at(w, base) * (float)q8_s8_at(w, base, i);
    }
    /* Q4_0: 18-byte blocks / 32 values. */
    uint block = idx >> 5u, j = idx & 31u;
    uint base = block * 18u;
    float d = f16_at(w, base);
    uint packed = q8_byte_at(w, base + 2u, j & 15u);
    uint nib = (j < 16u) ? (packed & 0xfu) : (packed >> 4u);
    return d * ((float)nib - 8.0f);
}

/* Mirrors the CUDA reference router: in_dim=NE, out_dim=NS, rows=T,
 * index=K (shared-gate in_dim), rsvd3=gate weight type. */
[numthreads(256, 1, 1)]
void router_topk(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint t = gid_grp.x;
    uint NE = params.in_dim;
    uint NS = params.out_dim;
    uint K = params.index;
    uint type = params.rsvd3;
    if (t >= params.rows || NE == 0u || NE > 512u || NS == 0u || NS > 32u) return;

    /* max over the expert logits */
    float mx = -3.4e38f;
    for (uint e = tid; e < NE; e += 256u) mx = max(mx, logits_buf[t * NE + e]);
    red[tid] = mx;
    GroupMemoryBarrierWithGroupSync();
    for (uint st = 128u; st > 0u; st >>= 1u) {
        if (tid < st) red[tid] = max(red[tid], red[tid + st]);
        GroupMemoryBarrierWithGroupSync();
    }
    mx = red[0];

    /* softmax probabilities */
    float ps = 0.0f;
    for (uint e = tid; e < NE; e += 256u) {
        float v = exp(logits_buf[t * NE + e] - mx);
        sp[e] = v;
        ps += v;
    }
    red[tid] = ps;
    GroupMemoryBarrierWithGroupSync();
    for (uint st = 128u; st > 0u; st >>= 1u) {
        if (tid < st) red[tid] += red[tid + st];
        GroupMemoryBarrierWithGroupSync();
    }
    float total = red[0];
    for (uint e = tid; e < NE; e += 256u) sp[e] /= total;

    /* shared-expert gate logit */
    if (K != 0u) {
        float g = 0.0f;
        for (uint i = tid; i < K; i += 256u)
            g += qwen4_dense_scalar(gate_buf, i, type) * x_buf[t * K + i];
        red[tid] = g;
        GroupMemoryBarrierWithGroupSync();
        for (uint st = 128u; st > 0u; st >>= 1u) {
            if (tid < st) red[tid] += red[tid + st];
            GroupMemoryBarrierWithGroupSync();
        }
        if (tid == 0u) shared_buf[t] = red[0];
        GroupMemoryBarrierWithGroupSync();
    }

    /* strict top-NS (ties -> lower expert id); mark picked probs used */
    for (uint s = 0u; s < NS; s++) {
        float best = -1.0f;
        uint id = 0xffffffffu;
        for (uint e = tid; e < NE; e += 256u) {
            if (sp[e] > best) { best = sp[e]; id = e; }
        }
        keyv[tid] = best;
        keyid[tid] = id;
        GroupMemoryBarrierWithGroupSync();
        for (uint st = 128u; st > 0u; st >>= 1u) {
            if (tid < st && (keyv[tid + st] > keyv[tid] ||
                             (keyv[tid + st] == keyv[tid] &&
                              keyid[tid + st] < keyid[tid]))) {
                keyv[tid] = keyv[tid + st];
                keyid[tid] = keyid[tid + st];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (tid == 0u) {
            selected_buf[t * NS + s] = (int)keyid[0];
            red[s] = keyv[0];
            sp[keyid[0]] = -1.0f;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    /* renormalize the selected probabilities */
    if (tid < NS) {
        float denom = 0.0f;
        for (uint s = 0u; s < NS; s++) denom += red[s];
        weights_buf[t * NS + tid] = red[tid] / denom;
    }
}
