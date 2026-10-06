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
 * One 128-thread block (4 warps) per (4 rows, slot, token); each warp owns one
 * output row.  The whole dot runs on lane 0 (correct; the warp-cooperative
 * variant is a later optimisation).  Weight rows are addressed by absolute
 * byte offset inside the bound model range.
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

/* Dot of one dequantised weight row at `base` against x[xb..xb+K). */
float qwen4_moe_dot(ByteAddressBuffer w, uint base, uint xb, uint K, uint type) {
    float acc = 0.0f;
    if (type == 0u) {
        for (uint i = 0u; i < K; i++) acc += asfloat(w.Load(base + i * 4u)) * a_buf[xb + i];
    } else if (type == 1u) {
        for (uint i = 0u; i < K; i++) acc += f16_at(w, base + i * 2u) * a_buf[xb + i];
    } else if (type == 30u) {
        for (uint i = 0u; i < K; i++) acc += bf16_at(w, base + i * 2u) * a_buf[xb + i];
    } else if (type == 8u) {
        uint blocks = K >> 5u;
        for (uint b = 0u; b < blocks; b++) {
            uint bb = base + b * 34u;
            float d = q8_scale_at(w, bb);
            for (uint i = 0u; i < 32u; i++)
                acc += d * (float)q8_s8_at(w, bb, i) * a_buf[xb + b * 32u + i];
        }
    } else if (type == 2u) {
        uint blocks = K >> 5u;
        for (uint b = 0u; b < blocks; b++) {
            uint bb = base + b * 18u;
            float d = f16_at(w, bb);
            for (uint j = 0u; j < 16u; j++) {
                uint packed = q8_byte_at(w, bb + 2u, j);
                acc += d * ((float)(packed & 0xfu) - 8.0f) * a_buf[xb + b * 32u + j];
                acc += d * ((float)(packed >> 4u) - 8.0f) * a_buf[xb + b * 32u + j + 16u];
            }
        }
    } else if (type == 39u) {
        uint blocks = K >> 5u;
        for (uint b = 0u; b < blocks; b++)
            acc += mxfp4_block_dot(w, base + b * 17u, a_buf, xb + b * 32u);
    } else if (type == 12u) {
        uint blocks = K >> 8u;
        for (uint b = 0u; b < blocks; b++) {
            uint bb = base + b * 144u;
            for (uint j = 0u; j < 8u; j++)
                acc += q4k_sub_dot(w, bb, j, a_buf, xb + b * 256u + j * 32u);
        }
    } else if (type == 10u) {
        uint subs = (K + 31u) >> 5u;
        for (uint s = 0u; s < subs; s++)
            acc += qwen4_q2k_sub_dot(w, base + (s >> 3u) * 84u, s & 7u, xb + s * 32u);
    } else if (type == 16u) {
        uint subs = K >> 5u;
        for (uint s = 0u; s < subs; s++) {
            uint b = s >> 3u;
            acc += qwen4_iq2_sub_dot(w, base + b * 66u + 2u + 8u * (s & 7u),
                                     f16_at(w, base + b * 66u), xb + s * 32u);
        }
    }
    return acc;
}

/* params: n=K (activation len), rows=T, in_dim=M (out rows), out_dim=NS,
 * aux=expert_bytes, ratio=row_bytes, blocks=shared_type (0xffffffff=none),
 * index=shared row bytes, rsvd3=routed type, flags bit0=pool bit1=shared. */
[numthreads(128, 1, 1)]
void moe_mid(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x * 4u + (tid >> 5u);
    uint slot = gid_grp.y;
    uint t = gid_grp.z;
    uint M = params.in_dim;
    uint NS = params.out_dim;
    uint K = params.n;
    if (row >= M || t >= params.rows) return;
    const bool has_shared = (params.flags & 2u) != 0u;
    const bool use_shared = has_shared && slot == NS;
    const uint stride = NS + (has_shared ? 1u : 0u);
    const uint pair = t * stride + slot;
    float a = 0.0f, b = 0.0f;
    if ((tid & 31u) == 0u) {
        if (use_shared) {
            uint srb = params.index;
            a = qwen4_moe_dot(sh0_buf, row * srb, t * K, K, params.blocks);
            b = qwen4_moe_dot(sh1_buf, row * srb, t * K, K, params.blocks);
        } else {
            int e = selected_buf[t * NS + slot];
            if ((params.flags & 1u) != 0u) {
                int ts = tbl_buf[e];
                if (ts >= 0) e = ts;
            }
            if (e >= 0) {
                uint off = ((uint)e * M + row) * params.ratio;
                a = qwen4_moe_dot(w_buf, off, t * K, K, params.rsvd3);
                b = qwen4_moe_dot(up_w_buf, off, t * K, K, params.rsvd3);
            }
        }
        out4_buf[pair * M + row] = qwen4_silu(a) * b;
    }
}

[numthreads(128, 1, 1)]
void moe_down(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x * 4u + (tid >> 5u);
    uint slot = gid_grp.y;
    uint t = gid_grp.z;
    uint M = params.in_dim;
    uint NS = params.out_dim;
    uint K = params.n;
    if (row >= M || t >= params.rows) return;
    const bool has_shared = (params.flags & 2u) != 0u;
    const bool use_shared = has_shared && slot == NS;
    const uint stride = NS + (has_shared ? 1u : 0u);
    const uint pair = t * stride + slot;
    float a = 0.0f;
    if ((tid & 31u) == 0u) {
        if (use_shared) {
            a = qwen4_moe_dot(sh0_buf, row * params.index, pair * K, K,
                              params.blocks);
        } else {
            int e = selected_buf[t * NS + slot];
            if ((params.flags & 1u) != 0u) {
                int ts = tbl_buf[e];
                if (ts >= 0) e = ts;
            }
            if (e >= 0) {
                uint off = ((uint)e * M + row) * params.ratio;
                a = qwen4_moe_dot(w_buf, off, pair * K, K, params.rsvd3);
            }
        }
        out4_buf[pair * M + row] = a;
    }
}
