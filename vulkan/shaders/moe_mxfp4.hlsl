/* vulkan/shaders/moe_mxfp4.hlsl -- routed MoE kernels for the MXFP4 (GGUF
 * type 39) expert quant format.
 *
 * Same structure as moe_iq2.hlsl / moe_q4k.hlsl: one block per (row, pair),
 * 256 threads, each thread accumulates whole 32-value MXFP4 blocks and a tree
 * reduce sums the block.  Weight rows are addressed by absolute byte offset
 * inside the bound model range:
 *   gate/up rows at expert*gate_expert_bytes + row*gate_row_bytes + b*17
 *   down rows at    expert*down_expert_bytes + row*down_row_bytes + b*17
 * where b indexes 32-value MXFP4 blocks (17 bytes each).  The activation x is
 * f32 (a_buf); the dequant mirrors ds4_vec_dot_mxfp4_f32.
 *
 * MXFP4 block (17 bytes / 32 values): e (E8M0 exponent) @0, qs[16] @1.
 * qs[j] packs two 4-bit E2M1 codes: low nibble = element j, high nibble =
 * element j+16.  Value = e8m0_to_f32(e) * mxfp4_value(nib).
 */

#include "common.hlsl"

StructuredBuffer<float> a_buf : register(t0);
ByteAddressBuffer w_buf : register(t1);    /* gate_w (gate/up) or down_w */
ByteAddressBuffer up_w_buf : register(t2); /* up_w (gate/up only) */
StructuredBuffer<uint> order_buf : register(t3); /* expert-grouped pair order */
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

/* One block per (row, pair): gate[slot][row] and up[slot][row] are MXFP4 dots
 * of the selected expert's row against x, mid gets the clamped SwiGLU scaled
 * by the router weight.
 * params: in_dim=expert_in_dim, out_dim=expert_mid_dim, rows=n_tokens,
 * index=n_expert, aux=gate_expert_bytes, ratio=gate_row_bytes,
 * blocks=in_dim/32, clamp=clamp. */
[numthreads(256, 1, 1)]
void moe_gate_up_mid_mxfp4(uint3 gid_grp : SV_GroupID,
                           uint tid : SV_GroupThreadID) {
    uint pair = gid_grp.y;
    uint mid_dim = params.out_dim;
    uint n_expert = params.index;
    uint n_tokens = params.rows;
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
    MOE_ROWS_BEGIN(gid_grp.x, mid_dim)
    uint gbase = expert * expert_bytes + row * row_bytes;
    uint ubase = expert * expert_bytes + row * row_bytes;
    float gacc = 0.0f;
    float uacc = 0.0f;
    for (uint b = tid; b < blocks; b += 256u) {
        uint xb = tok * params.in_dim + b * 32u;
        gacc += mxfp4_block_dot(w_buf, gbase + b * 17u, a_buf, xb);
        uacc += mxfp4_block_dot(up_w_buf, ubase + b * 17u, a_buf, xb);
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

/* One block per (row, pair): down[slot][row] = MXFP4 dot of the selected
 * expert's down row against mid[slot] (the weighted SwiGLU output).
 * params: in_dim=expert_mid_dim, out_dim=out_dim, rows=n_tokens,
 * index=n_expert, aux=down_expert_bytes, ratio=down_row_bytes,
 * blocks=in_dim/32. */
[numthreads(256, 1, 1)]
void moe_down_mxfp4(uint3 gid_grp : SV_GroupID,
                    uint tid : SV_GroupThreadID) {
    uint pair = gid_grp.y;
    uint out_dim = params.out_dim;
    uint n_expert = params.index;
    uint n_tokens = params.rows;
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
    MOE_ROWS_BEGIN(gid_grp.x, out_dim)
    uint bbase = expert * expert_bytes + row * row_bytes;
    float acc = 0.0f;
    for (uint b = tid; b < blocks; b += 256u) {
        uint xb = pair * params.in_dim + b * 32u;
        acc += mxfp4_block_dot(w_buf, bbase + b * 17u, a_buf, xb);
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

/* Partial MXFP4 block dot over `bc` qs bytes starting at byte b0 (0..15) of
 * the block: byte b0+jj holds element (b0+jj) in its low nibble and element
 * (b0+jj+16) in its high nibble.  Used by the split v2 kernels to keep all
 * 256 lanes busy when the block count is below 256. */
static float mxfp4_partial_dot(ByteAddressBuffer w, uint bbase,
                               StructuredBuffer<float> x, uint xb,
                               uint b0, uint bc) {
    float d = e8m0_to_f32(q8_byte_at(w, bbase, 0u));
    float acc = 0.0f;
    for (uint jj = 0u; jj < bc; jj++) {
        uint q = q8_byte_at(w, bbase + 1u, b0 + jj);
        acc += d * mxfp4_value(q & 0xfu) * x[xb + b0 + jj];
        acc += d * mxfp4_value(q >> 4u) * x[xb + b0 + jj + 16u];
    }
    return acc;
}
/* v2: same as moe_gate_up_mid_mxfp4 but two lanes share each 32-value block
 * (half the qs bytes each) so all 256 lanes are active at in_dim/32 blocks
 * (e.g. 128 at in_dim=4096).  The tree reduce is unchanged.  Each workgroup
 * processes params.rsvd2 consecutive rows: the prefill grid is
 * dispatch-bound on NVIDIA (one tiny workgroup per (row, pair) = millions),
 * so amortising the per-workgroup cost over several rows is a large win.
 * rsvd2 (rows/workgroup) is set by the host; 0/1 = the classic one-row grid. */
[numthreads(256, 1, 1)]
void moe_gate_up_mid_mxfp4_v2(uint3 gid_grp : SV_GroupID,
                              uint tid : SV_GroupThreadID) {
    uint mid_dim = params.out_dim;
    uint n_expert = params.index;
    uint n_tokens = params.rows;
    uint blocks = params.blocks;
    uint expert_bytes = params.aux;
    uint row_bytes = params.ratio;
    uint pair = gid_grp.y;
    if (pair >= n_tokens * n_expert) return;
    uint apair = ((params.flags & 2u) != 0u) ? order_buf[pair] : pair;
    uint tok = apair / n_expert;
    uint slot = apair - tok * n_expert;
    int sel = selected_buf[apair];
    if (sel < 0) sel = 0;
    uint expert;
    if ((params.flags & 1u) != 0u) {
        int ts = tbl_buf[sel];
        if (ts < 0) ts = 0;
        expert = (uint)ts;
    } else {
        expert = (uint)sel;
    }
    uint R = params.rsvd2;
    if (R == 0u) R = 1u;
    uint b = tid >> 1u;
    uint h = tid & 1u;
    for (uint rr = 0u; rr < R; rr++) {
        uint row = gid_grp.x * R + rr;
        if (row >= mid_dim) break;
        uint gbase = expert * expert_bytes + row * row_bytes;
        float gacc = 0.0f;
        float uacc = 0.0f;
        for (uint bb = b; bb < blocks; bb += 128u) {
            uint xb = tok * params.in_dim + bb * 32u;
            gacc += mxfp4_partial_dot(w_buf, gbase + bb * 17u, a_buf, xb,
                                      h * 8u, 8u);
            uacc += mxfp4_partial_dot(up_w_buf, gbase + bb * 17u, a_buf, xb,
                                      h * 8u, 8u);
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
            uint off = apair * mid_dim + row;
            out2_buf[off] = g;
            out3_buf[off] = u;
            out4_buf[off] = (g / (1.0f + exp(-g))) * u *
                            weights_buf[apair];
        }
    }
}

/* v2 down: four lanes share each 32-value block (four qs bytes each) so all
 * 256 lanes are active at mid_dim/32 blocks (e.g. 64 at mid_dim=2048).  Like
 * the gate/up v2, each workgroup processes DS4_MOE_ROWS_PER_GROUP rows. */
[numthreads(256, 1, 1)]
void moe_down_mxfp4_v2(uint3 gid_grp : SV_GroupID,
                       uint tid : SV_GroupThreadID) {
    uint out_dim = params.out_dim;
    uint n_expert = params.index;
    uint n_tokens = params.rows;
    uint blocks = params.blocks;
    uint expert_bytes = params.aux;
    uint row_bytes = params.ratio;
    uint pair = gid_grp.y;
    if (pair >= n_tokens * n_expert) return;
    uint apair = ((params.flags & 2u) != 0u) ? order_buf[pair] : pair;
    uint tok = apair / n_expert;
    uint slot = apair - tok * n_expert;
    int sel = selected_buf[apair];
    if (sel < 0) sel = 0;
    uint expert;
    if ((params.flags & 1u) != 0u) {
        int ts = tbl_buf[sel];
        if (ts < 0) ts = 0;
        expert = (uint)ts;
    } else {
        expert = (uint)sel;
    }
    uint R = params.rsvd2;
    if (R == 0u) R = 1u;
    uint b = tid >> 2u;
    uint h = tid & 3u;
    for (uint rr = 0u; rr < R; rr++) {
        uint row = gid_grp.x * R + rr;
        if (row >= out_dim) break;
        uint bbase = expert * expert_bytes + row * row_bytes;
        float acc = 0.0f;
        for (uint bb = b; bb < blocks; bb += 64u) {
            uint xb = apair * params.in_dim + bb * 32u;
            acc += mxfp4_partial_dot(w_buf, bbase + bb * 17u, a_buf, xb,
                                     h * 4u, 4u);
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
            out4_buf[apair * out_dim + row] = moe_down_red[0];
        }
    }
}
