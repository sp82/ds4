/* vulkan/shaders/qwen4_moe_reduce.hlsl -- Qwen MoE combine + expert lists.
 *
 * moe_reduce mirrors the CUDA reference: out[t][d] = sum_{s<NS} w_s *
 * part[t][s][d] + sigmoid(gate[t]) * shared_slot, then the hyper-connection
 * injection R[t][s][d] += injection(s) * out[t][d].
 * moe_build_lists is the CUDA expert_lists: per-expert capped pair buckets.
 *
 * Bindings: part (t0), weights (t1), gate (t2), selected (t3), shared (t10),
 * out (u0), R (u1), inj (u2), lists (u3), counts (u4).
 */

#include "common.hlsl"

StructuredBuffer<float> part_buf : register(t0);
StructuredBuffer<float> weights_buf : register(t1);
StructuredBuffer<float> gate_buf : register(t2);
StructuredBuffer<int> selected_buf : register(t3);
StructuredBuffer<float> shared_buf : register(t10);
RWStructuredBuffer<float> out_buf : register(u0);
RWStructuredBuffer<float> R_buf : register(u1);
RWStructuredBuffer<float> inj_buf : register(u2);
RWStructuredBuffer<int> lists_buf : register(u3);
RWStructuredBuffer<int> counts_buf : register(u4);

[[vk::push_constant]] DS4Params params;

groupshared uint qwen4_list_counters[512];

float qwen4_sigmoid(float x) {
    const float e = exp(-abs(x));
    return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

/* params: rows=T, out_dim=D, in_dim=NS, ratio=stride, index=hc,
 * flags bit0=has_gate bit1=has_shared. */
[numthreads(256, 1, 1)]
void moe_reduce(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint d = gid_grp.x * 256u + tid;
    uint t = gid_grp.y;
    uint D = params.out_dim;
    uint NS = params.in_dim;
    uint stride = params.ratio;
    uint hc = params.index;
    if (d >= D || t >= params.rows) return;

    float v = 0.0f;
    for (uint s = 0u; s < NS; s++)
        v += weights_buf[t * NS + s] * part_buf[((uint64_t)t * stride + s) * D + d];
    if ((params.flags & 1u) != 0u) {
        float shared_val = ((params.flags & 2u) != 0u)
            ? shared_buf[t * D + d]
            : part_buf[((uint64_t)t * stride + NS) * D + d];
        v += qwen4_sigmoid(gate_buf[t]) * shared_val;
    }
    out_buf[t * D + d] = v;

    for (uint s = 0u; s < hc; s++) {
        float tot = 0.0f;
        for (uint i = 0u; i < hc * 8u; i++)
            tot += inj_buf[(uint64_t)t * hc * hc * 8u + i * hc + s];
        const float wgt = 2.0f * qwen4_sigmoid(tot / (float)hc);
        R_buf[((uint64_t)t * hc + s) * D + d] += wgt * v;
    }
}

/* params: index=NE, rows=pairs, out_dim=cap. */
[numthreads(256, 1, 1)]
void moe_build_lists(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint NE = params.index;
    uint pairs = params.rows;
    uint cap = params.out_dim;
    if (NE == 0u || NE > 512u) return;
    for (uint e = tid; e < NE; e += 256u) qwen4_list_counters[e] = 0u;
    GroupMemoryBarrierWithGroupSync();
    for (uint p = tid; p < pairs; p += 256u) {
        int e = selected_buf[p];
        if (e >= 0 && (uint)e < NE) {
            uint slot;
            InterlockedAdd(qwen4_list_counters[e], 1u, slot);
            if (slot < cap) lists_buf[(uint64_t)e * cap + slot] = (int)p;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint e = tid; e < NE; e += 256u)
        counts_buf[e] = (int)min(qwen4_list_counters[e], cap);
}
