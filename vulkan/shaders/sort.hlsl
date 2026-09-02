/* vulkan/shaders/sort.hlsl -- ascending bitonic sort of int32 rows.
 *
 * row_width must be a power of two <= 2048 (matches the Metal kernel's
 * power-of-two requirement).  One block of 256 threads sorts one row.
 */

#include "common.hlsl"

StructuredBuffer<int> src_buf : register(t0);
RWStructuredBuffer<int> dst_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared int sort_tmp[2048];

[numthreads(256, 1, 1)]
void sort_i32_rows_asc(uint3 gid : SV_DispatchThreadID) {
    uint row = gid.y;
    uint w = params.n;
    uint base = row * w;
    for (uint i = gid.x; i < w; i += 256u) {
        sort_tmp[i] = src_buf[base + i];
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint k = 2u; k <= w; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint i = gid.x; i < w; i += 256u) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    int a = sort_tmp[i];
                    int b = sort_tmp[ixj];
                    bool asc = ((i & k) == 0u) ? (a > b) : (a < b);
                    if (asc) {
                        sort_tmp[i] = b;
                        sort_tmp[ixj] = a;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    for (uint i2 = gid.x; i2 < w; i2 += 256u) {
        dst_buf[base + i2] = sort_tmp[i2];
    }
}
