/* vulkan/shaders/moe.hlsl -- routed MoE kernels (Phase 4).
 *
 * gate/up/mid, down projection and expert sum for Q8_0 quantized expert
 * weights (the same block format the matmul kernels consume).  Weight rows
 * are addressed by their absolute byte offset inside the bound model range:
 *   gate/up rows at expert*gate_expert_bytes + row*gate_row_bytes
 *   down rows at    expert*down_expert_bytes + row*down_row_bytes
 * (all offsets 32-bit; the host rejects expert regions >= 4 GiB).
 *
 * One binding set per file (dxc rule): t0->0, t1->1, t2->2, u0->4, u1->5,
 * u2->6, u3->7, u4->8.  a_buf carries x (gate/up), mid (down) or down
 * (sum) depending on the entry point; out4_buf carries mid, down or the
 * final out.  Router select lives in router.hlsl.  Scalar parameters in
 * DS4Params.
 */

#include "common.hlsl"

StructuredBuffer<float> a_buf : register(t0);
ByteAddressBuffer w_buf : register(t1);    /* gate_w (gate/up) or down_w */
ByteAddressBuffer up_w_buf : register(t2); /* up_w (gate/up only) */
RWStructuredBuffer<int> selected_buf : register(u0);
RWStructuredBuffer<float> weights_buf : register(u1);
RWStructuredBuffer<float> out2_buf : register(u2); /* gate */
RWStructuredBuffer<float> out3_buf : register(u3); /* up */
RWStructuredBuffer<float> out4_buf : register(u4); /* mid / down / out */
RWStructuredBuffer<int> tbl_buf : register(u5); /* expert->slot table (pool) */

[[vk::push_constant]] DS4Params params;

groupshared float moe_gate_red[256];
groupshared float moe_up_red[256];
groupshared float moe_down_red[256];

/* One block per (row, pair): gate[slot][row] and up[slot][row] are Q8_0
 * dots of the selected expert's row against x (quantized inline per block),
 * mid gets the clamped SwiGLU scaled by the router weight.  Mirrors the
 * CUDA moe_gate_up_mid kernels at the block level.
 * params: in_dim=expert_in_dim, out_dim=expert_mid_dim, rows=n_tokens,
 * index=n_expert, aux=gate_expert_bytes, ratio=gate_row_bytes,
 * blocks=in_dim/32, clamp=clamp. */
[numthreads(256, 1, 1)]
void moe_gate_up_mid_q8(uint3 gid_grp : SV_GroupID,
                        uint tid : SV_GroupThreadID) {
    uint pair = gid_grp.y;
    uint mid_dim = params.out_dim;
    uint n_expert = params.index;
    uint n_tokens = params.rows;
    uint in_dim = params.in_dim;
    uint blocks = params.blocks;
    uint expert_bytes = params.aux;
    uint row_bytes = params.ratio;
    if (pair >= n_tokens * n_expert) return;
    uint tok = pair / n_expert;
    uint slot = pair - tok * n_expert;
    int sel = selected_buf[tok * n_expert + slot];
    if (sel < 0) sel = 0;
    uint expert;
    if ((params.flags & 1u) != 0u) {
        int ts = tbl_buf[sel];
        if (ts < 0) ts = 0;
        expert = (uint)ts;
    } else {
        expert = (uint)sel;
    }
    uint xoff = tok * in_dim;
    MOE_ROWS_BEGIN(gid_grp.x, mid_dim)
    uint gbase = expert * expert_bytes + row * row_bytes;
    uint ubase = expert * expert_bytes + row * row_bytes;
    float gacc = 0.0f;
    float uacc = 0.0f;
    for (uint b = tid; b < blocks; b += 256u) {
        uint xb = xoff + b * 32u;
        uint gb = gbase + b * 34u;
        uint ub = ubase + b * 34u;
        float amax = 0.0f;
        for (uint i = 0u; i < 32u; i++) {
            amax = max(amax, abs(a_buf[xb + i]));
        }
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        int gd = 0;
        int ud = 0;
        for (uint i = 0u; i < 32u; i++) {
            int q = (int)round(a_buf[xb + i] * id);
            q = min(q, 127);
            q = max(q, -128);
            gd += q8_s8_at(w_buf, gb, i) * q;
            ud += q8_s8_at(up_w_buf, ub, i) * q;
        }
        gacc += q8_scale_at(w_buf, gb) * d * (float)gd;
        uacc += q8_scale_at(up_w_buf, ub) * d * (float)ud;
    }
    moe_gate_red[tid] = gacc;
    moe_up_red[tid] = uacc;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            moe_gate_red[tid] += moe_gate_red[tid + stride];
            moe_up_red[tid] += moe_up_red[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        float g = moe_gate_red[0];
        float u = moe_up_red[0];
        if (params.clamp > 1.0e-6f) {
            g = min(g, params.clamp);
            u = min(max(u, -params.clamp), params.clamp);
        }
        uint off = pair * mid_dim + row;
        out2_buf[off] = g;
        out3_buf[off] = u;
        out4_buf[off] = (g / (1.0f + exp(-g))) * u *
                        weights_buf[tok * n_expert + slot];
    }
    MOE_ROWS_END
}

/* One block per (row, pair): down[slot][row] = Q8_0 dot of the selected
 * expert's down row against mid[slot] (the weighted SwiGLU output).
 * params: in_dim=expert_mid_dim, out_dim=out_dim, rows=n_tokens,
 * index=n_expert, aux=down_expert_bytes, ratio=down_row_bytes,
 * blocks=in_dim/32. */
[numthreads(256, 1, 1)]
void moe_down_q8(uint3 gid_grp : SV_GroupID,
                 uint tid : SV_GroupThreadID) {
    uint pair = gid_grp.y;
    uint out_dim = params.out_dim;
    uint n_expert = params.index;
    uint n_tokens = params.rows;
    uint in_dim = params.in_dim;
    uint blocks = params.blocks;
    uint expert_bytes = params.aux;
    uint row_bytes = params.ratio;
    if (pair >= n_tokens * n_expert) return;
    uint tok = pair / n_expert;
    uint slot = pair - tok * n_expert;
    int sel = selected_buf[tok * n_expert + slot];
    if (sel < 0) sel = 0;
    uint expert;
    if ((params.flags & 1u) != 0u) {
        int ts = tbl_buf[sel];
        if (ts < 0) ts = 0;
        expert = (uint)ts;
    } else {
        expert = (uint)sel;
    }
    uint xoff = pair * in_dim;
    MOE_ROWS_BEGIN(gid_grp.x, out_dim)
    uint base = expert * expert_bytes + row * row_bytes;
    float acc = 0.0f;
    for (uint b = tid; b < blocks; b += 256u) {
        uint xb = xoff + b * 32u;
        uint wb = base + b * 34u;
        float amax = 0.0f;
        for (uint i = 0u; i < 32u; i++) {
            amax = max(amax, abs(a_buf[xb + i]));
        }
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        int dot = 0;
        for (uint i = 0u; i < 32u; i++) {
            int q = (int)round(a_buf[xb + i] * id);
            q = min(q, 127);
            q = max(q, -128);
            dot += q8_s8_at(w_buf, wb, i) * q;
        }
        acc += q8_scale_at(w_buf, wb) * d * (float)dot;
    }
    moe_down_red[tid] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            moe_down_red[tid] += moe_down_red[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        out4_buf[pair * out_dim + row] = moe_down_red[0];
    }
    MOE_ROWS_END
}

/* out[tok][row] = sum_slot down[(tok*n_expert+slot)][row].
 * params: rows=n_tokens, out_dim=out_dim, index=n_expert. */
[numthreads(256, 1, 1)]
void moe_sum(uint3 gid : SV_DispatchThreadID) {
    uint n = params.rows * params.out_dim;   /* n_tokens * out_dim */
    if (gid.x >= n) return;
    uint tok = gid.x / params.out_dim;
    uint row = gid.x - tok * params.out_dim;
    uint n_expert = params.index;
    float acc = 0.0f;
    for (uint e = 0u; e < n_expert; e++) {
        acc += a_buf[(tok * n_expert + e) * params.out_dim + row];
    }
    out4_buf[gid.x] = acc;
}
