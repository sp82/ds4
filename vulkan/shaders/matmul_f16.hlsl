/* vulkan/shaders/matmul_f16.hlsl -- FP16 weight matmul.
 *
 * out[tok][row] = sum_k w_f16[row][k] * x[tok][k].
 * One block of 256 threads per output element.  Weights are read from the
 * model buffer (binding 3) as packed uint16, bound at the tensor's offset.
 */

#include "common.hlsl"

StructuredBuffer<float> x_buf : register(t0);
ByteAddressBuffer w_buf : register(t3);
RWStructuredBuffer<float> out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared float matmul_red[256];

[numthreads(256, 1, 1)]
void matmul_f16(uint3 gid : SV_DispatchThreadID,
                uint3 gid_grp : SV_GroupID,
                uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x;
    uint tok = gid_grp.y;
    uint out_dim = params.out_dim;
    uint in_dim = params.in_dim;
    if (row >= out_dim || tok >= params.rows) return;

    float acc = 0.0f;
    const uint xoff = tok * in_dim;
    const uint base = row * in_dim * 2u;
    if ((base & 3u) == 0u) {
        for (uint kk = tid * 8u; kk + 8u <= in_dim; kk += 256u * 8u) {
            const uint off = base + kk * 2u;
            const uint4 v = uint4(w_buf.Load(off), w_buf.Load(off + 4u),
                                  w_buf.Load(off + 8u), w_buf.Load(off + 12u));
            acc += f16tof32(v.x & 0xffffu) * x_buf[xoff + kk];
            acc += f16tof32(v.x >> 16) * x_buf[xoff + kk + 1u];
            acc += f16tof32(v.y & 0xffffu) * x_buf[xoff + kk + 2u];
            acc += f16tof32(v.y >> 16) * x_buf[xoff + kk + 3u];
            acc += f16tof32(v.z & 0xffffu) * x_buf[xoff + kk + 4u];
            acc += f16tof32(v.z >> 16) * x_buf[xoff + kk + 5u];
            acc += f16tof32(v.w & 0xffffu) * x_buf[xoff + kk + 6u];
            acc += f16tof32(v.w >> 16) * x_buf[xoff + kk + 7u];
        }
        for (uint kk = (in_dim & ~7u) + tid; kk < in_dim; kk += 256u) {
            acc += f16_at(w_buf, base + kk * 2u) * x_buf[xoff + kk];
        }
    } else {
        for (uint kk = tid; kk < in_dim; kk += 256u) {
            acc += f16_at(w_buf, base + kk * 2u) * x_buf[xoff + kk];
        }
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
