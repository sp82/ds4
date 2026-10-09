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

/* Q8_0 weights against raw f32 activations (Qwen projections preserve f32
 * activations, unlike matmul_q8_0 which quantizes x inline).  in_dim must be
 * a multiple of 32; blocks = in_dim / 32.
 *
 * llama.cpp-style GEMV tiling: one 64-thread block computes Q8F32_NR output
 * rows for a single token.  The old form spent 256 lanes on one row while
 * only in_dim/32 lanes had work (176 idle at in_dim 2560) and read the weight
 * qs byte-wise (one Load per int8); here every lane has work, the 32 x values
 * of a block are loaded once and reused across rows, and the qs come from
 * aligned 32-bit words (the 34-byte Q8 block is 2-mod-4, so even blocks
 * repack from 5 overlapping words -- same values, 4x fewer loads). */
#define Q8F32_NR 4

groupshared float q8f32_red[Q8F32_NR][64];

[numthreads(64, 1, 1)]
void matmul_q8_0_f32(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row0 = gid_grp.x * Q8F32_NR;
    uint tok = gid_grp.y;
    uint out_dim = params.out_dim;
    uint in_dim = params.in_dim;
    uint blocks = params.blocks;   /* in_dim / 32 */
    if (row0 >= out_dim || tok >= params.rows) return;
    const uint nrows = min(Q8F32_NR, out_dim - row0);

    float acc[Q8F32_NR];
    [unroll] for (uint r = 0u; r < Q8F32_NR; r++) acc[r] = 0.0f;

    for (uint b = tid; b < blocks; b += 64u) {
        uint xb = tok * in_dim + b * 32u;
        float xv[32];
        [unroll] for (uint i = 0u; i < 32u; i++) xv[i] = x_buf[xb + i];
        [unroll] for (uint r = 0u; r < Q8F32_NR; r++) {
            uint rr = row0 + r;
            if (rr >= out_dim) rr = row0;   /* tail: reuse a valid row */
            uint wblock = rr * blocks + b;
            uint sw = w_buf.Load((wblock * 34u) & ~3u);
            float d = f16tof32(((wblock & 1u) != 0u ? (sw >> 16u) : sw) &
                               0xffffu);
            [unroll] for (uint h = 0u; h < 2u; h++) {
                uint w_byte = wblock * 34u + 2u + h * 16u;
                uint ww0, ww1, ww2, ww3;
                if ((wblock & 1u) != 0u) {
                    ww0 = w_buf.Load(w_byte);
                    ww1 = w_buf.Load(w_byte + 4u);
                    ww2 = w_buf.Load(w_byte + 8u);
                    ww3 = w_buf.Load(w_byte + 12u);
                } else {
                    /* per-half overlap word: covers the two qs bytes that
                     * precede this half (offset 34w+2h16 is 2 mod 4). */
                    uint wa = w_buf.Load(w_byte - 2u);
                    uint wb = w_buf.Load(w_byte + 2u);
                    uint wc = w_buf.Load(w_byte + 6u);
                    uint wd = w_buf.Load(w_byte + 10u);
                    uint we = w_buf.Load(w_byte + 14u);
                    ww0 = (wa >> 16u) | ((wb & 0xffffu) << 16u);
                    ww1 = (wb >> 16u) | ((wc & 0xffffu) << 16u);
                    ww2 = (wc >> 16u) | ((wd & 0xffffu) << 16u);
                    ww3 = (wd >> 16u) | ((we & 0xffffu) << 16u);
                }
                uint e = h * 16u;
                int4 q;
                q = s8x4((int)ww0);
                acc[r] += d * (float)q.x * xv[e + 0u];
                acc[r] += d * (float)q.y * xv[e + 1u];
                acc[r] += d * (float)q.z * xv[e + 2u];
                acc[r] += d * (float)q.w * xv[e + 3u];
                q = s8x4((int)ww1);
                acc[r] += d * (float)q.x * xv[e + 4u];
                acc[r] += d * (float)q.y * xv[e + 5u];
                acc[r] += d * (float)q.z * xv[e + 6u];
                acc[r] += d * (float)q.w * xv[e + 7u];
                q = s8x4((int)ww2);
                acc[r] += d * (float)q.x * xv[e + 8u];
                acc[r] += d * (float)q.y * xv[e + 9u];
                acc[r] += d * (float)q.z * xv[e + 10u];
                acc[r] += d * (float)q.w * xv[e + 11u];
                q = s8x4((int)ww3);
                acc[r] += d * (float)q.x * xv[e + 12u];
                acc[r] += d * (float)q.y * xv[e + 13u];
                acc[r] += d * (float)q.z * xv[e + 14u];
                acc[r] += d * (float)q.w * xv[e + 15u];
            }
        }
    }

    [unroll] for (uint r = 0u; r < Q8F32_NR; r++) q8f32_red[r][tid] = acc[r];
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            [unroll] for (uint r = 0u; r < Q8F32_NR; r++) {
                q8f32_red[r][tid] += q8f32_red[r][tid + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        [unroll] for (uint r = 0u; r < Q8F32_NR; r++) {
            if (r < nrows) out_buf[tok * out_dim + row0 + r] = q8f32_red[r][0];
        }
    }
}

[numthreads(256, 1, 1)]
void matmul_bf16(uint3 gid_grp : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x;
    uint tok = gid_grp.y;
    uint out_dim = params.out_dim;
    uint in_dim = params.in_dim;
    if (row >= out_dim || tok >= params.rows) return;

    float acc = 0.0f;
    const uint base = row * in_dim * 2u;
    const uint xoff = tok * in_dim;
    for (uint k = tid; k < in_dim; k += 256u) {
        acc += bf16_at(w_buf, base + k * 2u) * x_buf[xoff + k];
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
