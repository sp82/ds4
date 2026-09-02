/* vulkan/shaders/top1.hlsl -- fused Q8_0 matmul + argmax (single token).
 *
 * Computes the out_dim Q8_0 dot products of one token and reduces to the
 * maximum (ties break toward the lower index, matching the CUDA
 * matmul_q8_0_top1 path).  Writes values[0] (f32) and selected[0] (uint32 =
 * index + index_offset).  One block of 256 threads.
 */

#include "common.hlsl"

StructuredBuffer<float> x_buf : register(t0);
ByteAddressBuffer w_buf : register(t3);
RWStructuredBuffer<float> val_buf : register(u0);
RWStructuredBuffer<uint> sel_buf : register(u1);

[[vk::push_constant]] DS4Params params;

groupshared float top1_val[256];
groupshared uint top1_idx[256];

[numthreads(256, 1, 1)]
void matmul_q8_0_top1(uint3 gid : SV_DispatchThreadID) {
    uint out_dim = params.out_dim;
    uint in_dim = params.in_dim;
    uint blocks = params.blocks;
    uint index_offset = params.index;

    float best_v = -3.402823466e38f;
    uint best_i = 0xffffffffu;
    for (uint row = gid.x; row < out_dim; row += 256u) {
        float acc = 0.0f;
        for (uint b = 0u; b < blocks; b++) {
            uint i0 = b * 32u;
            uint bn = min(32u, in_dim - i0);
            uint wblock = row * blocks + b;
            float amax = 0.0f;
            for (uint i = 0u; i < bn; i++) {
                amax = max(amax, abs(x_buf[i0 + i]));
            }
            float d = amax / 127.0f;
            float id = d != 0.0f ? 1.0f / d : 0.0f;
            int dot = 0;
            for (uint i = 0u; i < bn; i++) {
                int q = (int)round(x_buf[i0 + i] * id);
                q = min(q, 127);
                q = max(q, -128);
                dot += q8_s8(w_buf, wblock, i) * q;
            }
            acc += q8_scale(w_buf, wblock) * d * (float)dot;
        }
        uint ridx = row + index_offset;
        if (acc > best_v || (acc == best_v && ridx < best_i)) {
            best_v = acc;
            best_i = ridx;
        }
    }

    top1_val[gid.x] = best_v;
    top1_idx[gid.x] = best_i;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (gid.x < stride) {
            float vb = top1_val[gid.x + stride];
            uint ib = top1_idx[gid.x + stride];
            if (vb > top1_val[gid.x] ||
                (vb == top1_val[gid.x] && ib < top1_idx[gid.x])) {
                top1_val[gid.x] = vb;
                top1_idx[gid.x] = ib;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gid.x == 0u) {
        val_buf[0] = top1_val[0];
        sel_buf[0] = top1_idx[0];
    }
}
