/* vulkan/shaders/qwen4_moe_mm.hlsl -- Qwen MoE prefill (expert-grouped tiles).
 *
 * Scalar fallback port of the CUDA reference matrix<TYPE,DOWN,EXPERT=true>:
 * one 256-thread block (16 output rows x 16 tokens) per (row tile, expert,
 * token tile).  Weights are dequantised once per tile and reused across the 16
 * tokens; activations are gathered through the per-expert token lists.  No
 * shared expert here (the engine runs it as dense GEMMs).
 *
 *   moe_mm_mid : mid[tok*NO+slot][row] = silu(gate) * up
 *   moe_mm_down: part[tok*NO+slot][row] = down
 *
 * Bindings: x (t0), w0 (t1), w1 (t2), lists (t3), counts (t10), out (u0).
 */

#include "common.hlsl"
#include "iq2_tables.hlsl"

StructuredBuffer<float> x_buf : register(t0);
ByteAddressBuffer w0_buf : register(t1);
ByteAddressBuffer w1_buf : register(t2);
StructuredBuffer<int> lists_buf : register(t3);
StructuredBuffer<int> counts_buf : register(t10);
RWStructuredBuffer<float> out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared float mm_a[16][33];
groupshared float mm_b[16][33];
groupshared float mm_v[16][33];

float qwen4_silu(float x) {
    const float e = exp(-abs(x));
    const float s = x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    return x * s;
}

/* The expert-grouped tiled GEMMs use f16 tiles (like the CUDA
 * matrix_half_tile / Metal simdgroup_half8x8 paths) for the 256-value
 * super-block formats; q8_0 and the plain formats stay exact f32. */
bool qwen4_mm_half(uint type) {
    return type == 10u || type == 12u || type == 16u || type == 39u;
}

float qwen4_mm_round(float v, uint type) {
    if (qwen4_mm_half(type)) return f16tof32(ds4_f32_to_f16_bits_rne(v));
    return v;
}

/* One dequantised weight element at logical index `i` inside the row at
 * `base` (mirrors the CUDA value<TYPE>). */
float qwen4_value(ByteAddressBuffer w, uint base, uint i, uint type) {
    if (type == 0u) return asfloat(w.Load(base + i * 4u));
    if (type == 1u) return f16_at(w, base + i * 2u);
    if (type == 30u) return bf16_at(w, base + i * 2u);
    if (type == 8u) {
        uint b = base + (i >> 5u) * 34u;
        return q8_scale_at(w, b) * (float)q8_s8_at(w, b, i & 31u);
    }
    if (type == 2u) {
        uint b = base + (i >> 5u) * 18u;
        float d = f16_at(w, b);
        uint nib = (q8_byte_at(w, b + 2u, i & 15u) >> (4u * ((i & 31u) >> 4u))) & 15u;
        return d * ((float)nib - 8.0f);
    }
    if (type == 39u) {
        uint b = base + (i >> 5u) * 17u;
        uint q = (q8_byte_at(w, b + 1u, i & 15u) >> (4u * ((i & 31u) >> 4u))) & 15u;
        const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        float scale = e8m0_to_f32(q8_byte_at(w, b, 0u));
        float mag = levels[q & 7u];
        return ((q & 8u) != 0u ? -mag : mag) * scale;
    }
    if (type == 10u) {
        uint b = base + (i >> 8u) * 84u;
        uint j = i & 255u;
        uint sc = q8_byte_at(w, b, j >> 4u);
        uint q = (q8_byte_at(w, b + 16u, (j >> 7u) * 32u + (j & 31u)) >>
                  (2u * ((j & 127u) >> 5u))) & 3u;
        return f16_at(w, b + 80u) * (float)(sc & 15u) * (float)q -
               f16_at(w, b + 82u) * (float)(sc >> 4u);
    }
    if (type == 12u) {
        uint b = base + (i >> 8u) * 144u;
        uint j = i & 255u, group = j >> 5u;
        uint sc, mn;
        q4k_scale_min(w, b, group, sc, mn);
        uint q = (q8_byte_at(w, b + 16u, (j >> 6u) * 32u + (j & 31u)) >>
                  (4u * (group & 1u))) & 15u;
        return f16_at(w, b) * (float)sc * (float)q -
               f16_at(w, b + 2u) * (float)mn;
    }
    /* IQ2_XXS */
    uint b = base + (i >> 8u) * 66u;
    uint j = i & 255u, group = j >> 5u, sub = (j & 31u) >> 3u;
    uint p = b + 2u + group * 8u;
    uint grid_ids = q8_byte_at(w, p, 0u) | (q8_byte_at(w, p, 1u) << 8u) |
                    (q8_byte_at(w, p, 2u) << 16u) | (q8_byte_at(w, p, 3u) << 24u);
    uint signs_scale = q8_byte_at(w, p, 4u) | (q8_byte_at(w, p, 5u) << 8u) |
                       (q8_byte_at(w, p, 6u) << 16u) | (q8_byte_at(w, p, 7u) << 24u);
    uint gi = (grid_ids >> (8u * sub)) & 255u;
    uint si = (signs_scale >> (7u * sub)) & 127u;
    uint signs = ksigns_iq2xs[si];
    float v = (float)iq2xxs_grid_byte(gi, j & 7u);
    if ((signs & (1u << (j & 7u))) != 0u) v = -v;
    return f16_at(w, b) * (0.5f + (float)(signs_scale >> 28u)) * 0.25f * v;
}

/* params: n=K, rows=T, in_dim=M, out_dim=NO, index=NS, aux=expert_bytes,
 * ratio=rb, blocks=cap, rsvd3=type. */
[numthreads(256, 1, 1)]
void moe_mm_mid(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint K = params.n;
    uint M = params.in_dim;
    uint NO = params.out_dim;
    uint NS = params.index;
    uint cap = params.blocks;
    uint type = params.rsvd3;
    uint rb = params.ratio;
    uint e = gid_grp.y;
    uint count = (uint)counts_buf[e];
    uint r = tid & 15u, c = tid >> 4u;
    uint row = gid_grp.x * 16u + r;
    for (uint t0 = gid_grp.z * 16u; t0 < count; t0 += params.rsvd2 * 16u) {
        uint item = t0 + c;
        int pair = (item < count) ? lists_buf[e * cap + item] : (int)item;
        uint tok = (uint)pair / NS, slot = (uint)pair % NS;
        float acc = 0.0f, up = 0.0f;
        for (uint k0 = 0u; k0 < K; k0 += 32u) {
            for (uint j = tid; j < 16u * 32u; j += 256u) {
                uint rr = j >> 5u, kk = j & 31u;
                uint grow = gid_grp.x * 16u + rr, logical = k0 + kk;
                uint wbase = ((uint64_t)e * M + grow) * rb;
                bool valid = grow < M && logical < K;
                mm_a[rr][kk] = valid ? qwen4_mm_round(qwen4_value(w0_buf, wbase, logical, type), type) : 0.0f;
                mm_b[rr][kk] = valid ? qwen4_mm_round(qwen4_value(w1_buf, wbase, logical, type), type) : 0.0f;
                uint ti = t0 + rr;
                int pp = (ti < count) ? lists_buf[e * cap + ti] : (int)ti;
                uint tt = (uint)pp / NS;
                mm_v[rr][kk] = (ti < count && logical < K) ? qwen4_mm_round(x_buf[tt * K + logical], type) : 0.0f;
            }
            GroupMemoryBarrierWithGroupSync();
            for (uint k = 0u; k < 32u; k++) {
                acc += mm_a[r][k] * mm_v[c][k];
                up += mm_b[r][k] * mm_v[c][k];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (row < M && item < count)
            out_buf[(tok * NO + slot) * M + row] = qwen4_silu(acc) * up;
    }
}

[numthreads(256, 1, 1)]
void moe_mm_down(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint K = params.n;
    uint M = params.in_dim;
    uint NO = params.out_dim;
    uint NS = params.index;
    uint cap = params.blocks;
    uint type = params.rsvd3;
    uint rb = params.ratio;
    uint e = gid_grp.y;
    uint count = (uint)counts_buf[e];
    uint r = tid & 15u, c = tid >> 4u;
    uint row = gid_grp.x * 16u + r;
    for (uint t0 = gid_grp.z * 16u; t0 < count; t0 += params.rsvd2 * 16u) {
        uint item = t0 + c;
        int pair = (item < count) ? lists_buf[e * cap + item] : (int)item;
        uint tok = (uint)pair / NS, slot = (uint)pair % NS;
        float acc = 0.0f;
        for (uint k0 = 0u; k0 < K; k0 += 32u) {
            for (uint j = tid; j < 16u * 32u; j += 256u) {
                uint rr = j >> 5u, kk = j & 31u;
                uint grow = gid_grp.x * 16u + rr, logical = k0 + kk;
                uint wbase = ((uint64_t)e * M + grow) * rb;
                bool valid = grow < M && logical < K;
                mm_a[rr][kk] = valid ? qwen4_mm_round(qwen4_value(w0_buf, wbase, logical, type), type) : 0.0f;
                uint ti = t0 + rr;
                int pp = (ti < count) ? lists_buf[e * cap + ti] : (int)ti;
                uint tt = (uint)pp / NS, ss = (uint)pp % NS;
                mm_v[rr][kk] = (ti < count && logical < K)
                    ? qwen4_mm_round(x_buf[(tt * NO + ss) * K + logical], type) : 0.0f;
            }
            GroupMemoryBarrierWithGroupSync();
            for (uint k = 0u; k < 32u; k++) acc += mm_a[r][k] * mm_v[c][k];
            GroupMemoryBarrierWithGroupSync();
        }
        if (row < M && item < count)
            out_buf[(tok * NO + slot) * M + row] = acc;
    }
}
