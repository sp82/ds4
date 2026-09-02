/* vulkan/shaders/f16_conv.hlsl -- packed f32 -> f16 conversion.
 *
 * Each thread converts TWO consecutive f32 elements and writes one aligned
 * uint32 word (half0 in the low 16 bits, half1 in the high 16).  One thread
 * per word removes the read-modify-write race that byte-pair stores from
 * independent blocks would have on the same 4-byte word.
 */

#include "common.hlsl"

StructuredBuffer<float> src_buf : register(t0);
RWByteAddressBuffer out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

/* f32 -> f16 bit pattern (round to nearest even), as a uint16 in a uint. */
uint f32_to_f16_bits(float v) {
    uint bits = asuint(v);
    uint sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint man = bits & 0x7fffffu;
    uint h;
    if (exp >= 31) {
        h = sign | 0x7bffu;
    } else if (exp <= 0) {
        uint mant = man | 0x800000u;
        uint shift = (uint)(1 - exp);
        h = sign | ((mant >> shift) >> 13);
    } else {
        h = sign | ((uint32_t)exp << 10) | (man >> 13);
    }
    return h;
}

[numthreads(256, 1, 1)]
void f32_to_f16(uint3 gid : SV_DispatchThreadID) {
    uint n = params.n;
    uint i0 = gid.x * 2u;
    if (i0 >= n) return;
    uint lo = f32_to_f16_bits(src_buf[i0]);
    uint hi = (i0 + 1u < n) ? f32_to_f16_bits(src_buf[i0 + 1u]) : 0u;
    out_buf.Store(gid.x * 4u, lo | (hi << 16));
}
