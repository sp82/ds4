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

/* Multi-row variant for the Qwen dense path: a 64-thread block computes
 * F16_NR output rows for one token (the 1-row form spends a full 256-lane
 * block per output element -- only in_dim/8 lanes have work, e.g. 40 of 256
 * at the hc up-projections' in_dim 320).  The 8 x values of a group are
 * loaded once and reused across rows; weight rows are word-read when every
 * row base is 4-aligned (in_dim even).  Same per-element accumulation order
 * as the 1-row kernel; the tree sums 64 lanes instead of 256. */
#define F16_NR 4

groupshared float f16_red[F16_NR][64];

[numthreads(64, 1, 1)]
void matmul_f16_mr(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row0 = gid_grp.x * F16_NR;
    uint tok = gid_grp.y;
    uint out_dim = params.out_dim;
    uint in_dim = params.in_dim;
    if (row0 >= out_dim || tok >= params.rows) return;
    const uint nrows = min(F16_NR, out_dim - row0);

    float acc[F16_NR];
    [unroll] for (uint r = 0u; r < F16_NR; r++) acc[r] = 0.0f;

    const uint xoff = tok * in_dim;
    if ((in_dim & 1u) == 0u && (in_dim & 7u) == 0u) {
        /* every row base is 4-aligned (in_dim even); one thread per
         * 8-element group */
        for (uint g = tid; g * 8u < in_dim; g += 64u) {
            uint kk = g * 8u;
            float xv[8];
            [unroll] for (uint i = 0u; i < 8u; i++) xv[i] = x_buf[xoff + kk + i];
            [unroll] for (uint r = 0u; r < F16_NR; r++) {
                uint rr = row0 + r;
                if (rr >= out_dim) rr = row0;   /* tail: reuse a valid row */
                const uint off = rr * in_dim * 2u + kk * 2u;
                const uint4 v = uint4(w_buf.Load(off), w_buf.Load(off + 4u),
                                      w_buf.Load(off + 8u),
                                      w_buf.Load(off + 12u));
                acc[r] += f16tof32(v.x & 0xffffu) * xv[0];
                acc[r] += f16tof32(v.x >> 16) * xv[1];
                acc[r] += f16tof32(v.y & 0xffffu) * xv[2];
                acc[r] += f16tof32(v.y >> 16) * xv[3];
                acc[r] += f16tof32(v.z & 0xffffu) * xv[4];
                acc[r] += f16tof32(v.z >> 16) * xv[5];
                acc[r] += f16tof32(v.w & 0xffffu) * xv[6];
                acc[r] += f16tof32(v.w >> 16) * xv[7];
            }
        }
    } else {
        /* unaligned or non-multiple-of-8 in_dim: scalar, same order */
        for (uint kk = tid; kk < in_dim; kk += 64u) {
            float xv = x_buf[xoff + kk];
            [unroll] for (uint r = 0u; r < F16_NR; r++) {
                uint rr = row0 + r;
                if (rr >= out_dim) rr = row0;
                acc[r] += f16_at(w_buf, rr * in_dim * 2u + kk * 2u) * xv;
            }
        }
    }

    [unroll] for (uint r = 0u; r < F16_NR; r++) f16_red[r][tid] = acc[r];
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            [unroll] for (uint r = 0u; r < F16_NR; r++) {
                f16_red[r][tid] += f16_red[r][tid + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        [unroll] for (uint r = 0u; r < F16_NR; r++) {
            if (r < nrows) out_buf[tok * out_dim + row0 + r] = f16_red[r][0];
        }
    }
}
