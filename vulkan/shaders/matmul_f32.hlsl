/* vulkan/shaders/matmul_f32.hlsl -- FP32 weight matmul fallback.
 *
 * out[tok][row] = sum_k w_f32[row][k] * x[tok][k].
 * One block of 256 threads per output element.  Weights are read from the
 * model buffer (binding 3) as packed float32, bound at the tensor's offset.
 */

#include "common.hlsl"

StructuredBuffer<float> x_buf : register(t0);
ByteAddressBuffer w_buf : register(t3);
RWStructuredBuffer<float> out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared float matmul_red[256];

[numthreads(256, 1, 1)]
void matmul_f32(uint3 gid : SV_DispatchThreadID,
                uint3 gid_grp : SV_GroupID,
                uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x;
    uint tok = gid_grp.y;
    uint out_dim = params.out_dim;
    uint in_dim = params.in_dim;
    if (row >= out_dim || tok >= params.rows) return;

    float acc = 0.0f;
    for (uint k = tid; k < in_dim; k += 256u) {
        acc += asfloat(w_buf.Load((row * in_dim + k) * 4u)) *
               x_buf[tok * in_dim + k];
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
