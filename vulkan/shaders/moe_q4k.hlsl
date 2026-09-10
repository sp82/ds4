/* vulkan/shaders/moe_q4k.hlsl -- routed MoE kernels for the Q4_K (GGUF type 12)
 * expert quant format.
 *
 * Same structure as moe_iq2.hlsl (gate/up/mid, down projection, then the shared
 * moe_sum entry point in moe.hlsl sums the down rows): one block per (row,
 * pair), 256 threads, where each thread contributes the dot of one 32-value
 * sub-block and a tree reduce sums the block.  Weight rows are addressed by
 * absolute byte offset inside the bound model range:
 *   gate/up rows at expert*gate_expert_bytes + row*gate_row_bytes + b*144
 *   down rows at    expert*down_expert_bytes + row*down_row_bytes + b*144
 * where b indexes 256-value super-blocks (144 bytes Q4_K).  The activation x
 * is f32 (a_buf); the dequant mirrors ds4_vec_dot_q4_K_f32.
 *
 * Q4_K block (144 bytes / 256 values): d (f16) @0, dmin (f16) @2,
 * scales[12] @4, qs[128] @16.  Eight 32-value groups; group j uses
 * byte_off=(j>>1)*32 and shift=(j&1)*4 in qs, with a two-level scale/min
 * decoded from scales[] (q4_k_get_scale_min).  Value = d*sc*nib - dmin*m.
 */

#include "common.hlsl"

StructuredBuffer<float> a_buf : register(t0);
ByteAddressBuffer w_buf : register(t1);    /* gate_w (gate/up) or down_w */
ByteAddressBuffer up_w_buf : register(t2); /* up_w (gate/up only) */
RWStructuredBuffer<int> selected_buf : register(u0);
RWStructuredBuffer<float> weights_buf : register(u1);
RWStructuredBuffer<float> out2_buf : register(u2); /* gate */
RWStructuredBuffer<float> out3_buf : register(u3); /* up */
RWStructuredBuffer<float> out4_buf : register(u4); /* mid / down */
RWStructuredBuffer<int> tbl_buf : register(u5); /* expert->slot table (pool) */

[[vk::push_constant]] DS4Params params;

groupshared float moe_gate_red[256];
groupshared float moe_up_red[256];
groupshared float moe_down_red[256];

/* One block per (row, pair): gate[slot][row] and up[slot][row] are Q4_K dots
 * of the selected expert's row against x, mid gets the clamped SwiGLU scaled
 * by the router weight.
 * params: in_dim=expert_in_dim, out_dim=expert_mid_dim, rows=n_tokens,
 * index=n_expert, aux=gate_expert_bytes, ratio=gate_row_bytes,
 * blocks=in_dim/256, clamp=clamp. */
[numthreads(256, 1, 1)]
void moe_gate_up_mid_q4k(uint3 gid_grp : SV_GroupID,
                         uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x;
    uint pair = gid_grp.y;
    uint mid_dim = params.out_dim;
    uint n_expert = params.index;
    uint n_tokens = params.rows;
    uint blocks = params.blocks;
    uint expert_bytes = params.aux;
    uint row_bytes = params.ratio;
    if (row >= mid_dim || pair >= n_tokens * n_expert) return;
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
    uint gbase = expert * expert_bytes + row * row_bytes;
    uint ubase = expert * expert_bytes + row * row_bytes;
    uint sub_blocks = blocks * 8u;
    float gacc = 0.0f;
    float uacc = 0.0f;
    if (tid < sub_blocks) {
        uint b = tid >> 3u;
        uint sub = tid & 7u;
        uint xb = tok * params.in_dim + tid * 32u;
        gacc = q4k_sub_dot(w_buf, gbase + b * 144u, sub, a_buf, xb);
        uacc = q4k_sub_dot(up_w_buf, ubase + b * 144u, sub, a_buf, xb);
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
}

/* One block per (row, pair): down[slot][row] = Q4_K dot of the selected
 * expert's down row against mid[slot] (the weighted SwiGLU output).
 * params: in_dim=expert_mid_dim, out_dim=out_dim, rows=n_tokens,
 * index=n_expert, aux=down_expert_bytes, ratio=down_row_bytes,
 * blocks=in_dim/256. */
[numthreads(256, 1, 1)]
void moe_down_q4k(uint3 gid_grp : SV_GroupID,
                  uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x;
    uint pair = gid_grp.y;
    uint out_dim = params.out_dim;
    uint n_expert = params.index;
    uint n_tokens = params.rows;
    uint blocks = params.blocks;
    uint expert_bytes = params.aux;
    uint row_bytes = params.ratio;
    if (row >= out_dim || pair >= n_tokens * n_expert) return;
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
    uint bbase = expert * expert_bytes + row * row_bytes;
    uint sub_blocks = blocks * 8u;
    float acc = 0.0f;
    if (tid < sub_blocks) {
        uint b = tid >> 3u;
        uint sub = tid & 7u;
        uint xb = pair * params.in_dim + tid * 32u;
        acc = q4k_sub_dot(w_buf, bbase + b * 144u, sub, a_buf, xb);
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
}
