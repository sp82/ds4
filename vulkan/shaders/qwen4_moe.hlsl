/* vulkan/shaders/qwen4_moe.hlsl -- Qwen3.8-Flash-Next routed MoE decode.
 *
 * Two entry points mirroring the CUDA reference moe_mv<TYPE,DOWN>:
 *   moe_mid  : mid[pair][row] = silu(gate_dot) * up_dot
 *   moe_down : part[pair][row] = down_dot
 * where pair = t*stride + slot, stride = NS + (shared ? 1 : 0) and slot==NS is
 * the shared expert (read from the dense sh0/sh1 tensors with shared_type).
 * Unlike the DeepSeek kernels these do NOT scale by the router weight (the
 * weight is applied in moe_reduce) and the shared expert is an extra slot.
 *
 * One 256-thread block per (row, slot, token).  The dot is block-cooperative:
 * each thread dequantises one 32-value sub-block (or a strided slice of the
 * element-wise formats) and a groupshared tree reduction sums the block.  This
 * replaces the earlier lane-0-only dot (32x underutilised) that dominated the
 * Qwen decode.  Weight rows are addressed by absolute byte offset inside the
 * bound model range (or the expert-pool slab when params.flags bit0 is set).
 *
 * Bindings: x (t0), w0 (t1), w1 (t2), shared w0 (t10), shared w1 (t11),
 * selected (u0), out (u4), pool table (u5).
 */

#include "common.hlsl"
#include "iq2_tables.hlsl"

StructuredBuffer<float> a_buf : register(t0);
ByteAddressBuffer w_buf : register(t1);
ByteAddressBuffer up_w_buf : register(t2);
ByteAddressBuffer sh0_buf : register(t10);
ByteAddressBuffer sh1_buf : register(t11);
RWStructuredBuffer<int> selected_buf : register(u0);
RWStructuredBuffer<float> out4_buf : register(u4);
RWStructuredBuffer<int> tbl_buf : register(u5);

[[vk::push_constant]] DS4Params params;

/* Output rows computed per workgroup.  The grid is (ceil(M/MOE_NR), slots,
 * tokens); one 256-thread block still does the cooperative dot, but now for
 * MOE_NR rows at once, so the per-block launch and reduction cost is amortized
 * (the previous one-row-per-block grid was 640x11 blocks for a single token). */
#define MOE_NR 4

groupshared float moe_gate_red[MOE_NR][256];
groupshared float moe_up_red[MOE_NR][256];
groupshared float moe_down_red[MOE_NR][256];

float qwen4_silu(float x) {
    const float e = exp(-abs(x));
    const float s = x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    return x * s;
}

/* One 32-value IQ2_XXS sub-block dot against x[xb..].  qoff is the byte offset
 * of the 8-byte sub-block group, d the super-block scale. */
float qwen4_iq2_sub_dot(ByteAddressBuffer w, uint qoff, float d, uint xb) {
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

/* One 32-value Q2_K sub-block dot; bbase is the 84-byte block offset, sub the
 * sub-block index 0..7 within the 256-value block, xb the x offset. */
float qwen4_q2k_sub_dot(ByteAddressBuffer w, uint bbase, uint sub, uint xb) {
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

/* This thread's contribution to the dot of the weight row at `base` against
 * x[xb..xb+K).  `tid` is the thread index; one 32-value sub-block per thread
 * for the block formats, a strided slice for the element-wise formats. */
float qwen4_dot_part(ByteAddressBuffer w, uint base, uint type, uint K,
                     uint xb, uint tid) {
    if (type == 0u || type == 1u || type == 30u) {
        float acc = 0.0f;
        for (uint i = tid; i < K; i += 256u) {
            float wv = (type == 0u) ? asfloat(w.Load(base + i * 4u))
                     : (type == 1u) ? f16_at(w, base + i * 2u)
                                    : bf16_at(w, base + i * 2u);
            acc += wv * a_buf[xb + i];
        }
        return acc;
    }
    uint nsub = (type == 10u || type == 12u || type == 16u)
              ? ((K + 31u) >> 5u) : (K >> 5u);
    if (tid >= nsub) return 0.0f;
    const uint sb = tid;
    const uint xb32 = xb + sb * 32u;
    if (type == 16u)
        return qwen4_iq2_sub_dot(w, base + (sb >> 3u) * 66u + 2u +
                                    8u * (sb & 7u),
                                 f16_at(w, base + (sb >> 3u) * 66u), xb32);
    if (type == 10u)
        return qwen4_q2k_sub_dot(w, base + (sb >> 3u) * 84u, sb & 7u, xb32);
    if (type == 12u)
        return q4k_sub_dot(w, base + (sb >> 3u) * 144u, sb & 7u, a_buf, xb32);
    if (type == 39u)
        return mxfp4_block_dot(w, base + sb * 17u, a_buf, xb32);
    if (type == 8u) {
        uint bb = base + sb * 34u;
        float d = q8_scale_at(w, bb);
        float acc = 0.0f;
        for (uint i = 0u; i < 32u; i++)
            acc += d * (float)q8_s8_at(w, bb, i) * a_buf[xb32 + i];
        return acc;
    }
    /* Q4_0 */
    uint bb = base + sb * 18u;
    float d = f16_at(w, bb);
    float acc = 0.0f;
    for (uint j = 0u; j < 16u; j++) {
        uint packed = q8_byte_at(w, bb + 2u, j);
        acc += d * ((float)(packed & 0xfu) - 8.0f) * a_buf[xb32 + j];
        acc += d * ((float)(packed >> 4u) - 8.0f) * a_buf[xb32 + j + 16u];
    }
    return acc;
}

/* params: n=K (activation len), rows=T, in_dim=M (out rows), out_dim=NS,
 * aux=expert_bytes, ratio=row_bytes, blocks=shared_type (0xffffffff=none),
 * index=shared row bytes, rsvd3=routed type, flags bit0=pool bit1=shared. */
[numthreads(256, 1, 1)]
void moe_mid(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint M = params.in_dim;
    uint NS = params.out_dim;
    uint K = params.n;
    uint row0 = gid_grp.x * MOE_NR;
    uint slot = gid_grp.y;
    uint t = gid_grp.z;
    const bool has_shared = (params.flags & 2u) != 0u;
    const bool use_shared = has_shared && slot == NS;
    const uint n_out = NS + (has_shared ? 1u : 0u);
    const uint pair = t * n_out + slot;
    if (row0 >= M || t >= params.rows) return;

    const uint xb = t * K;
    uint base0 = 0u;
    if (use_shared) {
        base0 = row0 * params.index;
    } else {
        int e = selected_buf[t * NS + slot];
        if ((params.flags & 1u) != 0u) {
            int ts = tbl_buf[e];
            if (ts >= 0) e = ts;
        }
        if (e < 0) {
            if (tid == 0u)
                [unroll] for (uint r = 0u; r < MOE_NR; r++)
                    if (row0 + r < M) out4_buf[pair * M + row0 + r] = 0.0f;
            return;
        }
        base0 = ((uint)e * M + row0) * params.ratio;
    }

    float ga[MOE_NR], ua[MOE_NR];
    [unroll] for (uint r = 0u; r < MOE_NR; r++) { ga[r] = 0.0f; ua[r] = 0.0f; }
    [unroll] for (uint r = 0u; r < MOE_NR; r++) {
        if (row0 + r >= M) break;
        if (use_shared) {
            ga[r] = qwen4_dot_part(sh0_buf, base0 + r * params.index, params.blocks, K, xb, tid);
            ua[r] = qwen4_dot_part(sh1_buf, base0 + r * params.index, params.blocks, K, xb, tid);
        } else {
            uint base = base0 + r * params.ratio;
            ga[r] = qwen4_dot_part(w_buf, base, params.rsvd3, K, xb, tid);
            ua[r] = qwen4_dot_part(up_w_buf, base, params.rsvd3, K, xb, tid);
        }
    }
    [unroll] for (uint r = 0u; r < MOE_NR; r++) {
        moe_gate_red[r][tid] = ga[r];
        moe_up_red[r][tid] = ua[r];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint st = 128u; st > 0u; st >>= 1u) {
        if (tid < st) {
            [unroll] for (uint r = 0u; r < MOE_NR; r++) {
                moe_gate_red[r][tid] += moe_gate_red[r][tid + st];
                moe_up_red[r][tid] += moe_up_red[r][tid + st];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u)
        [unroll] for (uint r = 0u; r < MOE_NR; r++)
            if (row0 + r < M)
                out4_buf[pair * M + row0 + r] =
                    qwen4_silu(moe_gate_red[r][0]) * moe_up_red[r][0];
}

[numthreads(256, 1, 1)]
void moe_down(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint M = params.in_dim;
    uint NS = params.out_dim;
    uint K = params.n;
    uint row0 = gid_grp.x * MOE_NR;
    uint slot = gid_grp.y;
    uint t = gid_grp.z;
    const bool has_shared = (params.flags & 2u) != 0u;
    const bool use_shared = has_shared && slot == NS;
    const uint n_out = NS + (has_shared ? 1u : 0u);
    const uint pair = t * n_out + slot;
    if (row0 >= M || t >= params.rows) return;

    const uint xb = pair * K;
    uint base0 = 0u;
    if (use_shared) {
        base0 = row0 * params.index;
    } else {
        int e = selected_buf[t * NS + slot];
        if ((params.flags & 1u) != 0u) {
            int ts = tbl_buf[e];
            if (ts >= 0) e = ts;
        }
        if (e < 0) {
            if (tid == 0u)
                [unroll] for (uint r = 0u; r < MOE_NR; r++)
                    if (row0 + r < M) out4_buf[pair * M + row0 + r] = 0.0f;
            return;
        }
        base0 = ((uint)e * M + row0) * params.ratio;
    }

    float a[MOE_NR];
    [unroll] for (uint r = 0u; r < MOE_NR; r++) a[r] = 0.0f;
    [unroll] for (uint r = 0u; r < MOE_NR; r++) {
        if (row0 + r >= M) break;
        if (use_shared) {
            a[r] = qwen4_dot_part(sh0_buf, base0 + r * params.index, params.blocks, K, xb, tid);
        } else {
            a[r] = qwen4_dot_part(w_buf, base0 + r * params.ratio, params.rsvd3, K, xb, tid);
        }
    }
    [unroll] for (uint r = 0u; r < MOE_NR; r++) moe_down_red[r][tid] = a[r];
    GroupMemoryBarrierWithGroupSync();
    for (uint st = 128u; st > 0u; st >>= 1u) {
        if (tid < st) {
            [unroll] for (uint r = 0u; r < MOE_NR; r++)
                moe_down_red[r][tid] += moe_down_red[r][tid + st];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u)
        [unroll] for (uint r = 0u; r < MOE_NR; r++)
            if (row0 + r < M) out4_buf[pair * M + row0 + r] = moe_down_red[r][0];
}
