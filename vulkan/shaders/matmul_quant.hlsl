/* vulkan/shaders/matmul_quant.hlsl -- dense quant matmul for the Q4_K and
 * Q4_0 weight formats (non-Q8_0 types that reach ds4_gpu_matmul_quant_tensor).
 *
 * out[tok][row] = dot(dequant(w[row]), x[tok]) with a raw f32 activation (no
 * inline quantization, unlike matmul_q8_0).  One block of 256 threads per
 * output element; the model buffer (binding 3) is bound at the weight tensor
 * byte offset, so in-shader row/block indices are relative to that tensor.
 *
 * Q4_K: 144-byte blocks / 256 values (dequant via q4k_sub_dot).
 * Q4_0: 18-byte blocks / 32 values (f16 d + 16 nibbles; elements 0..15 are
 * the low nibbles of qs[0..15], 16..31 the high; value = d * (nib - 8)).
 */

#include "common.hlsl"

StructuredBuffer<float> x_buf : register(t0);
ByteAddressBuffer w_buf : register(t3);
RWStructuredBuffer<float> out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared float matmul_red[256];

[numthreads(256, 1, 1)]
void matmul_q4k(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x;
    uint tok = gid_grp.y;
    uint out_dim = params.out_dim;
    uint in_dim = params.in_dim;
    uint blocks = params.blocks;   /* in_dim / 256 */
    if (row >= out_dim || tok >= params.rows) return;

    float acc = 0.0f;
    for (uint b = tid; b < blocks; b += 256u) {
        uint bbase = (row * blocks + b) * 144u;
        uint xb = tok * in_dim + b * 256u;
        for (uint j = 0u; j < 8u; j++) {
            acc += q4k_sub_dot(w_buf, bbase, j, x_buf, xb + j * 32u);
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

[numthreads(256, 1, 1)]
void matmul_q4_0(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x;
    uint tok = gid_grp.y;
    uint out_dim = params.out_dim;
    uint in_dim = params.in_dim;
    uint blocks = params.blocks;   /* in_dim / 32 */
    if (row >= out_dim || tok >= params.rows) return;

    float acc = 0.0f;
    for (uint b = tid; b < blocks; b += 256u) {
        uint bbase = (row * blocks + b) * 18u;
        float d = f16_at(w_buf, bbase);
        uint xb = tok * in_dim + b * 32u;
        for (uint j = 0u; j < 16u; j++) {
            uint packed = q8_byte_at(w_buf, bbase + 2u, j);
            acc += d * ((float)(packed & 0xfu) - 8.0f) * x_buf[xb + j];
            acc += d * ((float)(packed >> 4u) - 8.0f) * x_buf[xb + j + 16u];
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
