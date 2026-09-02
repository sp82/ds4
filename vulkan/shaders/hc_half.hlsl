/* vulkan/shaders/hc_half.hlsl -- HC expand with an f16 block output (the
 * attn_out_f16 / shared_down_f16 paths).  Parity with the ROCm
 * hc_expand4_half_kernel (n_hc==4, split layout: post = split+4, comb =
 * split+8).  block_out is stored as f16 (2 bytes/element).
 *
 * Bindings: t0 block_out (ByteAddressBuffer, f16), t1 residual_hc, t2 split,
 * u0 out_hc.
 */

#include "common.hlsl"

ByteAddressBuffer block_buf : register(t0);
StructuredBuffer<float> res_buf : register(t1);
StructuredBuffer<float> split_buf : register(t2);
RWStructuredBuffer<float> out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

[numthreads(256, 1, 1)]
void hc_expand4_half(uint3 gid : SV_DispatchThreadID) {
    uint n_elem = params.rows * params.index * params.n;
    if (gid.x >= n_elem) return;
    uint d = gid.x % params.n;
    uint tmp = gid.x / params.n;
    uint dst = tmp % params.index;
    uint t = tmp / params.index;
    uint n_hc = params.index;
    float bv = f16_at(block_buf, (t * params.n + d) * 2u);
    uint hc_base = t * n_hc * params.n + d;
    uint sp = t * 24u;
    float acc = bv * split_buf[sp + 4u + dst];
    for (uint src = 0u; src < 4u; src++) {
        acc += split_buf[sp + 8u + src * 4u + dst] *
               res_buf[hc_base + src * params.n];
    }
    out_buf[hc_base + dst * params.n] = acc;
}
