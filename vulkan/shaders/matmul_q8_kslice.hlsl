/* vulkan/shaders/matmul_q8_kslice.hlsl -- partial-K Q8_0 matmul.
 *
 * Computes a K-slice matvec: out[row] = W[row, k_off : k_off+k_cnt] @ x[0:k_cnt]
 * where the weight rows span full_in_dim (so the weight block index is
 * row*full_blocks + k_off/32 + sb) but only slice_blocks consecutive blocks
 * participate.  x rows carry only the k_cnt slice elements.  k_off and k_cnt
 * must be multiples of 32 (Q8_0 block).
 *
 * Param mapping: in_dim = k_cnt, blocks = full_blocks, index = k_off/32,
 * aux = slice_blocks (k_cnt/32).  One block of 256 threads per (row, token).
 */

#include "common.hlsl"

StructuredBuffer<float> x_buf : register(t0);
ByteAddressBuffer w_buf : register(t3);
RWStructuredBuffer<float> out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared float matmul_red[256];

[numthreads(256, 1, 1)]
void matmul_q8_0_kslice(uint3 gid : SV_DispatchThreadID,
                        uint3 gid_grp : SV_GroupID,
                        uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x;
    uint tok = gid_grp.y;
    uint out_dim = params.out_dim;
    uint slice = params.in_dim;          /* k_cnt elements per x row */
    uint full_blocks = params.blocks;
    uint block_start = params.index;     /* k_off / 32 */
    uint slice_blocks = params.aux;      /* k_cnt / 32 */
    if (row >= out_dim || tok >= params.rows || slice_blocks == 0u) return;

    float acc = 0.0f;
    for (uint sb = tid; sb < slice_blocks; sb += 256u) {
        uint wblock = row * full_blocks + block_start + sb;
        uint i0 = sb * 32u;
        uint bn = min(32u, slice - i0);
        float amax = 0.0f;
        for (uint i = 0u; i < bn; i++) {
            amax = max(amax, abs(x_buf[tok * slice + i0 + i]));
        }
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        int dot = 0;
        for (uint i = 0u; i < bn; i++) {
            int q = (int)round(x_buf[tok * slice + i0 + i] * id);
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
