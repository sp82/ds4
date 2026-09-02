/* vulkan/shaders/directional.hlsl -- directional steering projection
 * (Phase 5).  x[row] -= scale * (x[row] . dir[layer]) * dir[layer] for each
 * row (parity with the CUDA directional_steering_project_kernel).
 *
 * Bindings: t0 directions, u0 x (in place).  params: index=layer,
 * n=width, rows=rows, weight=scale.
 */

#include "common.hlsl"

StructuredBuffer<float> dir_buf : register(t0);
RWStructuredBuffer<float> x_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared float ds_partial[256];

[numthreads(256, 1, 1)]
void directional_steering_project(uint3 bid : SV_GroupID,
                                  uint tid : SV_GroupThreadID) {
    uint row = bid.x;
    if (row >= params.rows || params.n == 0u) return;
    uint width = params.n;
    uint base = row * width;
    uint dir_base = params.index * width;
    float sum = 0.0f;
    for (uint i = tid; i < width; i += 256u) {
        sum += x_buf[base + i] * dir_buf[dir_base + i];
    }
    ds_partial[tid] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            ds_partial[tid] += ds_partial[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float coeff = params.weight * ds_partial[0];
    for (uint i = tid; i < width; i += 256u) {
        x_buf[base + i] -= coeff * dir_buf[dir_base + i];
    }
}
