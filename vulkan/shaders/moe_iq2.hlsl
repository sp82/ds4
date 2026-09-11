/* vulkan/shaders/moe_iq2.hlsl -- routed MoE kernels for the IQ2_XXS (gate/up,
 * GGUF type 16) and Q2_K (down, GGUF type 10) expert quant formats.
 *
 * Same structure as moe.hlsl (gate/up/mid, down projection, then the shared
 * moe_sum entry point in moe.hlsl sums the down rows): one block per
 * (row, pair), 256 threads, where each thread contributes the dot of one
 * 32-value sub-block and a tree reduce sums the block.  Weight rows are
 * addressed by absolute byte offset inside the bound model range:
 *   gate/up rows at expert*gate_expert_bytes + row*gate_row_bytes + b*66
 *   down rows at    expert*down_expert_bytes + row*down_row_bytes + b*84
 * where b indexes 256-value super-blocks (66 bytes IQ2_XXS / 84 bytes Q2_K).
 * The IQ2_XXS lookup tables come from iq2_tables.hlsl (generated from
 * cuda/mmq/ggml-common.h).  One binding set per file (dxc rule).
 */

#include "common.hlsl"
#include "iq2_tables.hlsl"

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

/* Dot of one 32-value IQ2_XXS sub-block against x.  qoff is the byte offset
 * of the 8-byte sub-block group (2 bytes after the block's fp16 d), d the
 * block scale, xb the x element offset.  Mirrors dequantize_row_iq2_xxs. */
float iq2xxs_sub_dot(ByteAddressBuffer w, uint qoff, float d, uint xb) {
    uint b0 = q8_byte_at(w, qoff, 0u);
    uint b1 = q8_byte_at(w, qoff, 1u);
    uint b2 = q8_byte_at(w, qoff, 2u);
    uint b3 = q8_byte_at(w, qoff, 3u);
    uint b4 = q8_byte_at(w, qoff, 4u);
    uint b5 = q8_byte_at(w, qoff, 5u);
    uint b6 = q8_byte_at(w, qoff, 6u);
    uint b7 = q8_byte_at(w, qoff, 7u);
    uint aux1 = b4 | (b5 << 8u) | (b6 << 16u) | (b7 << 24u);
    float db = d * (0.5f + (float)(aux1 >> 28u)) * 0.25f;
    float acc = 0.0f;
    for (uint l = 0u; l < 4u; l++) {
        uint gi = (l == 0u) ? b0 : (l == 1u ? b1 : (l == 2u ? b2 : b3));
        uint signs = ksigns_iq2xs[(aux1 >> (7u * l)) & 127u];
        for (uint j = 0u; j < 8u; j++) {
            float wv = db * (float)iq2xxs_grid_byte(gi, j);
            if ((signs & (1u << j)) != 0u) wv = -wv;
            acc += wv * a_buf[xb + 8u * l + j];
        }
    }
    return acc;
}

/* Dot of one 32-value Q2_K sub-block against x.  bbase is the byte offset of
 * the 84-byte block (scales[16] | qs[64] | d | dmin), sub the sub-block
 * index 0..7, xb the x element offset.  Mirrors dequantize_row_q2_K. */
float q2k_sub_dot(ByteAddressBuffer w, uint bbase, uint sub, uint xb) {
    float d = f16_at(w, bbase + 80u);
    float dmin = f16_at(w, bbase + 82u);
    uint half = sub >> 2u;
    uint j = sub & 3u;
    uint shift = 2u * j;
    float acc = 0.0f;
    for (uint l = 0u; l < 32u; l++) {
        uint sc_idx = half * 8u + j * 2u + (l >= 16u ? 1u : 0u);
        uint sc = q8_byte_at(w, bbase, sc_idx);
        float dl = d * (float)(sc & 0xfu);
        float ml = dmin * (float)(sc >> 4u);
        uint qb = q8_byte_at(w, bbase + 16u, half * 32u + l);
        acc += (dl * (float)((qb >> shift) & 3u) - ml) * a_buf[xb + l];
    }
    return acc;
}

/* One block per (row, pair): gate[slot][row] and up[slot][row] are IQ2_XXS
 * dots of the selected expert's row against x, mid gets the clamped SwiGLU
 * scaled by the router weight.
 * params: in_dim=expert_in_dim, out_dim=expert_mid_dim, rows=n_tokens,
 * index=n_expert, aux=gate_expert_bytes, ratio=gate_row_bytes,
 * blocks=in_dim/256, clamp=clamp. */
[numthreads(256, 1, 1)]
void moe_gate_up_mid_iq2xxs(uint3 gid_grp : SV_GroupID,
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
    uint sub_blocks = blocks * 8u;
    uint lsub = tid >> 3u;
    uint sub = tid & 7u;
    uint xb = tok * params.in_dim + tid * 32u;
    MOE_ROWS_BEGIN(gid_grp.x, mid_dim)
    uint gbase = expert * expert_bytes + row * row_bytes;
    uint ubase = expert * expert_bytes + row * row_bytes;
    float gacc = 0.0f;
    float uacc = 0.0f;
    if (tid < sub_blocks) {
        gacc = iq2xxs_sub_dot(w_buf, gbase + lsub * 66u + 2u + 8u * sub,
                              f16_at(w_buf, gbase + lsub * 66u), xb);
        uacc = iq2xxs_sub_dot(up_w_buf, ubase + lsub * 66u + 2u + 8u * sub,
                              f16_at(up_w_buf, ubase + lsub * 66u), xb);
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

/* One block per (row, pair): down[slot][row] = Q2_K dot of the selected
 * expert's down row against mid[slot] (the weighted SwiGLU output).
 * params: in_dim=expert_mid_dim, out_dim=out_dim, rows=n_tokens,
 * index=n_expert, aux=down_expert_bytes, ratio=down_row_bytes,
 * blocks=in_dim/256. */
[numthreads(256, 1, 1)]
void moe_down_q2k(uint3 gid_grp : SV_GroupID,
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
    uint sub_blocks = blocks * 8u;
    uint lsub = tid >> 3u;
    uint sub = tid & 7u;
    uint xb = pair * params.in_dim + tid * 32u;
    MOE_ROWS_BEGIN(gid_grp.x, out_dim)
    uint bbase = expert * expert_bytes + row * row_bytes;
    float acc = 0.0f;
    if (tid < sub_blocks) {
        acc = q2k_sub_dot(w_buf, bbase + lsub * 84u, sub, xb);
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

/* ------------------------------------------------------------------ */
/* v2 variants (Fase 7): same math, vectorized dequant reads and all   */
/* 256 lanes active (2 threads per 32-value IQ2_XXS sub-block for       */
/* gate/up, 4 threads per Q2_K sub-block for down).  Accumulation order */
/* differs from v1 (ok, smoke tolerance 1%).                            */
/* ------------------------------------------------------------------ */

/* Dot of half (16 values, groups l = 2h, 2h+1) of one 32-value IQ2_XXS
 * sub-block.  qoff is the sub-block byte offset (2-mod-4 for even b,
 * 4-aligned for odd b), d the super-block scale, xb the x offset of the
 * half, h the half index.  Reads the 8 sub-block bytes as 2-3 aligned
 * words instead of 8 byte-wise loads. */
float iq2xxs_sub_dot_v2(ByteAddressBuffer w, uint qoff, float d, uint xb,
                        uint h) {
    uint grid, aux1;
    if ((qoff & 3u) == 0u) {
        uint w0 = w.Load(qoff);
        uint w1 = w.Load(qoff + 4u);
        grid = w0;
        aux1 = w1;
    } else {
        uint w0 = w.Load(qoff - 2u);
        uint w1 = w.Load(qoff + 2u);
        uint w2 = w.Load(qoff + 6u);
        grid = ((w0 >> 16u) & 0xffffu) | ((w1 & 0xffffu) << 16u);
        aux1 = ((w1 >> 16u) & 0xffffu) | ((w2 & 0xffffu) << 16u);
    }
    float db = d * (0.5f + (float)(aux1 >> 28u)) * 0.25f;
    float acc = 0.0f;
    [unroll]
    for (uint k = 0u; k < 2u; k++) {
        uint l = 2u * h + k;
        uint gi = (grid >> (8u * l)) & 0xffu;
        uint signs = ksigns_iq2xs[(aux1 >> (7u * l)) & 127u];
        [unroll]
        for (uint j = 0u; j < 8u; j++) {
            float wv = db * (float)iq2xxs_grid_byte(gi, j);
            if ((signs & (1u << j)) != 0u) wv = -wv;
            acc += wv * a_buf[xb + 8u * k + j];
        }
    }
    return acc;
}

/* Dot of one 8-value chunk (values 8h..8h+7) of a Q2_K sub-block.  bbase is
 * the 84-byte block offset (4-aligned), sub 0..7, h the chunk index, xb the
 * x offset.  Layout (this model): scales[16] | qs[64] | d | dmin.  Value e of
 * the block lives in qs byte [half*32 + (e & 31)] with shift 2*j (byte
 * shared by all j sub-blocks), so each chunk reads 8 consecutive qs bytes
 * (2 aligned words).  d/dmin from one word, the 2 scale bytes from one word. */
float q2k_sub_dot_v2(ByteAddressBuffer w, uint bbase, uint sub, uint h,
                     uint xb) {
    uint half = sub >> 2u;
    uint j = sub & 3u;
    uint shift = 2u * j;
    uint dd = w.Load(bbase + 80u);
    float d = f16tof32(dd & 0xffffu);
    float dmin = f16tof32(dd >> 16u);
    /* The 2 scale bytes sit at bbase + half*8 + j*2, which is 2-mod-4 for
     * odd j: load the aligned word covering them and shift out (same trick
     * as q8_byte). */
    uint saddr = bbase + half * 8u + j * 2u;
    uint scw = w.Load(saddr & ~3u);
    uint sshift = (saddr & 3u) * 8u;
    uint sc0 = (scw >> sshift) & 0xffu;
    uint sc1 = (scw >> (sshift + 8u)) & 0xffu;
    uint sc = (h < 2u) ? sc0 : sc1;
    float dl = d * (float)(sc & 0xfu);
    float ml = dmin * (float)(sc >> 4u);
    uint qoff = bbase + 16u + half * 32u + 8u * h;
    uint qw0 = w.Load(qoff);
    uint qw1 = w.Load(qoff + 4u);
    float acc = 0.0f;
    [unroll]
    for (uint l = 0u; l < 8u; l++) {
        uint qb = (l < 4u) ? ((qw0 >> (8u * l)) & 0xffu)
                           : ((qw1 >> (8u * (l - 4u))) & 0xffu);
        float qv = (float)((qb >> shift) & 3u);
        acc += (dl * qv - ml) * a_buf[xb + l];
    }
    return acc;
}

[numthreads(256, 1, 1)]
void moe_gate_up_mid_iq2xxs_v2(uint3 gid_grp : SV_GroupID,
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
    uint sub_blocks = blocks * 8u;
    uint sb = tid >> 1u;
    uint h = tid & 1u;
    /* params.rsvd2 = rows per workgroup (see moe_mxfp4.hlsl v2 / SPECS_AUTOTUNE
     * §2.5): the prefill grid is dispatch-bound on NVIDIA, so amortise the
     * per-workgroup cost over several rows. */
    uint R = params.rsvd2;
    if (R == 0u) R = 1u;
    for (uint rr = 0u; rr < R; rr++) {
        uint row = gid_grp.x * R + rr;
        if (row >= mid_dim) break;
        uint gbase = expert * expert_bytes + row * row_bytes;
        uint ubase = gbase;
        float gacc = 0.0f;
        float uacc = 0.0f;
        if (sb < sub_blocks) {
            uint b = sb >> 3u;
            uint sub = sb & 7u;
            uint xb = tok * params.in_dim + sb * 32u + h * 16u;
            uint qoff = gbase + b * 66u + 2u + 8u * sub;
            float gd = f16_at(w_buf, gbase + b * 66u);
            float ud = f16_at(up_w_buf, ubase + b * 66u);
            gacc = iq2xxs_sub_dot_v2(w_buf, qoff, gd, xb, h);
            uacc = iq2xxs_sub_dot_v2(up_w_buf, qoff, ud, xb, h);
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
}

[numthreads(256, 1, 1)]
void moe_down_q2k_v2(uint3 gid_grp : SV_GroupID,
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
    uint sub_blocks = blocks * 8u;
    uint sb = tid >> 2u;
    uint h = tid & 3u;
    uint R = params.rsvd2;
    if (R == 0u) R = 1u;
    for (uint rr = 0u; rr < R; rr++) {
        uint row = gid_grp.x * R + rr;
        if (row >= out_dim) break;
        uint bbase = expert * expert_bytes + row * row_bytes;
        float acc = 0.0f;
        if (sb < sub_blocks) {
            uint b = sb >> 3u;
            uint sub = sb & 7u;
            uint xb = pair * params.in_dim + sb * 32u + h * 8u;
            acc = q2k_sub_dot_v2(w_buf, bbase + b * 84u, sub, h, xb);
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
}
