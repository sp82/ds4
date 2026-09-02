/* vulkan/shaders/quantize_q8.hlsl -- Q8_0 prequantization pass.
 *
 * Quantizes the activation tensor x into packed int8 blocks plus per-block
 * fp32 scales, mirroring quantize_q8_0_f32_kernel on CUDA:
 *   d = max|x|/127, q = round(x/d) clamped to [-128,127].
 * One thread per (block, token); each thread writes one 32-byte int8 block
 * (packed as 8 uint32) and one scale.  Grid = (blocks, n_tok).
 */

#include "common.hlsl"

StructuredBuffer<float> x_buf : register(t0);
RWByteAddressBuffer xq_buf : register(u0);
RWStructuredBuffer<float> xscale_buf : register(u1);

[[vk::push_constant]] DS4Params params;

[numthreads(256, 1, 1)]
void quantize_q8_0(uint3 gid : SV_DispatchThreadID) {
    uint b = gid.x;
    uint tok = gid.y;
    uint blocks = params.blocks;
    uint in_dim = params.in_dim;
    if (b >= blocks || tok >= params.rows) return;
    uint i0 = b * 32u;
    uint bn = min(32u, in_dim - i0);
    float amax = 0.0f;
    for (uint i = 0u; i < bn; i++) {
        amax = max(amax, abs(x_buf[tok * in_dim + i0 + i]));
    }
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    uint dst = (tok * blocks + b) * 32u;
    [unroll]
    for (uint j = 0u; j < 32u; j += 4u) {
        int4 q;
        q.x = (j + 0u < bn) ? (int)round(x_buf[tok * in_dim + i0 + j + 0u] * id) : 0;
        q.y = (j + 1u < bn) ? (int)round(x_buf[tok * in_dim + i0 + j + 1u] * id) : 0;
        q.z = (j + 2u < bn) ? (int)round(x_buf[tok * in_dim + i0 + j + 2u] * id) : 0;
        q.w = (j + 3u < bn) ? (int)round(x_buf[tok * in_dim + i0 + j + 3u] * id) : 0;
        q = min(max(q, int4(-128, -128, -128, -128)),
                int4(127, 127, 127, 127));
        xq_buf.Store(dst + j, pack8(q.x, q.y, q.z, q.w));
    }
    xscale_buf[tok * blocks + b] = d;
}
