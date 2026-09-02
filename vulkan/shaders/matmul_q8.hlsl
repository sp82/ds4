/* vulkan/shaders/matmul_q8.hlsl -- fused Q8_0 matmul.
 *
 * out[tok][row] = sum_b scale_w[b] * scale_x[b] * dot(qs_w[b], qs_x[b]).
 *
 * One block of 256 threads computes one output element (row, tok); x is
 * quantized inline per block.  The model buffer (binding 3) is bound at the
 * weight tensor's byte offset, so in-shader block indices are relative to
 * that tensor.  Portable int8 dot (no VK_KHR_shader_integer_dot_product).
 */

#include "common.hlsl"

StructuredBuffer<float> x_buf : register(t0);
ByteAddressBuffer w_buf : register(t3);
RWStructuredBuffer<float> out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared float matmul_red[256];

[numthreads(256, 1, 1)]
void matmul_q8_0(uint3 gid : SV_DispatchThreadID,
                 uint3 gid_grp : SV_GroupID,
                 uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x;
    uint tok = gid_grp.y;
    uint out_dim = params.out_dim;
    uint in_dim = params.in_dim;
    uint blocks = params.blocks;
    if (row >= out_dim || tok >= params.rows) return;

    float acc = 0.0f;
    for (uint b = tid; b < blocks; b += 256u) {
        uint i0 = b * 32u;
        uint bn = min(32u, in_dim - i0);
        uint wblock = row * blocks + b;
        float amax = 0.0f;
        for (uint i = 0u; i < bn; i++) {
            amax = max(amax, abs(x_buf[tok * in_dim + i0 + i]));
        }
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        int dot = 0;
        for (uint i = 0u; i < bn; i++) {
            int q = (int)round(x_buf[tok * in_dim + i0 + i] * id);
            q = min(q, 127);
            q = max(q, -128);
            dot += q8_s8(w_buf, wblock, i) * q;
        }
        acc += q8_scale(w_buf, wblock) * d * (float)dot;
    }

    matmul_red[tid] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            matmul_red[tid] += matmul_red[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        out_buf[tok * out_dim + row] = matmul_red[0];
    }
}
