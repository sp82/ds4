/* vulkan/shaders/hc_add_half.hlsl -- HC expand+add with an f16 add term
 * (shared_down_f16 path).  Parity with the ROCm hc_expand_add_half_kernel
 * (n_hc==4, split layout).  block_out is f32, block_add is f16.
 *
 * Bindings: t0 block_out (f32), t1 block_add (ByteAddressBuffer, f16),
 * t2 residual_hc, u0 out_hc, u1 split (ro).
 */

#include "common.hlsl"

StructuredBuffer<float> block_buf : register(t0);
ByteAddressBuffer add_buf : register(t1);
StructuredBuffer<float> res_buf : register(t2);
RWStructuredBuffer<float> out_buf : register(u0);
RWStructuredBuffer<float> split_buf : register(u1);

[[vk::push_constant]] DS4Params params;

[numthreads(256, 1, 1)]
void hc_expand4_add_half(uint3 gid : SV_DispatchThreadID) {
    uint n_elem = params.rows * params.index * params.n;
    if (gid.x >= n_elem) return;
    uint d = gid.x % params.n;
    uint tmp = gid.x / params.n;
    uint dst = tmp % params.index;
    uint t = tmp / params.index;
    uint n_hc = params.index;
    float bv = block_buf[t * params.n + d] +
               f16_at(add_buf, (t * params.n + d) * 2u);
    uint hc_base = t * n_hc * params.n + d;
    uint sp = t * 24u;
    float acc = bv * split_buf[sp + 4u + dst];
    for (uint src = 0u; src < 4u; src++) {
        acc += split_buf[sp + 8u + src * 4u + dst] *
               res_buf[hc_base + src * params.n];
    }
    out_buf[hc_base + dst * params.n] = acc;
}
