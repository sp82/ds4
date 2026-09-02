/* vulkan/shaders/matmul_q8_preq.hlsl -- prequantized Q8_0 matmul.
 *
 * Consumes the quantize_q8_0 output: weights from the model buffer (t3),
 * prequantized activations (packed int8, t0) and scales (t1).  Matches the
 * CUDA matmul_q8_0_preq_warp8 structure at the block level; one block of 256
 * threads per (row, token), lanes split the block dimension.  The int8 dot
 * reads four lanes at a time from the packed xq words.
 *
 * Variants (Fase 7 tuning, selected by the host via g_pipes[slot]):
 *   matmul_q8_0_preq    -- v1: one thread per Q8 block (128 of 256 lanes
 *                          active at in_dim 4096), byte-wise unaligned reads
 *                          of the weight qs.
 *   matmul_q8_0_preq_v2 -- v2: two threads per Q8 block (all 256 lanes
 *                          active at 128 blocks), vectorized weight reads.
 *                          The 34-byte block is 2-mod-4: even blocks carry
 *                          qs at 2 (mod 4), odd blocks at 0 (mod 4), so the
 *                          even-block path loads 5 aligned words and repacks
 *                          the 16 bytes into 4 sign-extended groups with a
 *                          shift.  Accumulation order differs from v1 (ok,
 *                          smoke tolerance 1%).
 */

#include "common.hlsl"

ByteAddressBuffer xq_buf : register(t0);
StructuredBuffer<float> xscale_buf : register(t1);
ByteAddressBuffer w_buf : register(t3);
RWStructuredBuffer<float> out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared float matmul_red[256];

static int dot4(int4 a, int4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

[numthreads(256, 1, 1)]
void matmul_q8_0_preq(uint3 gid : SV_DispatchThreadID,
                      uint3 gid_grp : SV_GroupID,
                      uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x;
    uint tok = gid_grp.y;
    uint out_dim = params.out_dim;
    uint blocks = params.blocks;
    if (row >= out_dim || tok >= params.rows) return;

    float acc = 0.0f;
    for (uint b = tid; b < blocks; b += 256u) {
        uint wblock = row * blocks + b;
        uint xq_byte = (tok * blocks + b) * 32u;
        int dot = dot8x32_w_xq(w_buf, wblock, xq_buf, xq_byte);
        acc += q8_scale(w_buf, wblock) * xscale_buf[tok * blocks + b] *
               (float)dot;
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
void matmul_q8_0_preq_v2(uint3 gid : SV_DispatchThreadID,
                         uint3 gid_grp : SV_GroupID,
                         uint tid : SV_GroupThreadID) {
    uint row = gid_grp.x;
    uint tok = gid_grp.y;
    uint out_dim = params.out_dim;
    uint blocks = params.blocks;
    if (row >= out_dim || tok >= params.rows) return;

    float acc = 0.0f;
    uint b = tid >> 1u;   /* Q8 block */
    uint h = tid & 1u;    /* 16-element half of the block */
    for (; b < blocks; b += 128u) {
        uint wblock = row * blocks + b;
        uint xq_byte = (tok * blocks + b) * 32u + h * 16u;
        int xw0 = (int)xq_buf.Load(xq_byte);
        int xw1 = (int)xq_buf.Load(xq_byte + 4u);
        int xw2 = (int)xq_buf.Load(xq_byte + 8u);
        int xw3 = (int)xq_buf.Load(xq_byte + 12u);

        int ww0, ww1, ww2, ww3;
        uint w_byte = wblock * 34u + 2u + h * 16u;
        if ((wblock & 1u) != 0u) {
            ww0 = (int)w_buf.Load(w_byte);
            ww1 = (int)w_buf.Load(w_byte + 4u);
            ww2 = (int)w_buf.Load(w_byte + 8u);
            ww3 = (int)w_buf.Load(w_byte + 12u);
        } else {
            uint wa = w_buf.Load(w_byte - 2u);
            uint wb = w_buf.Load(w_byte + 2u);
            uint wc = w_buf.Load(w_byte + 6u);
            uint wd = w_buf.Load(w_byte + 10u);
            uint we = w_buf.Load(w_byte + 14u);
            ww0 = (int)((wa >> 16u) | ((wb & 0xffffu) << 16u));
            ww1 = (int)((wb >> 16u) | ((wc & 0xffffu) << 16u));
            ww2 = (int)((wc >> 16u) | ((wd & 0xffffu) << 16u));
            ww3 = (int)((wd >> 16u) | ((we & 0xffffu) << 16u));
        }

        int dot = dot4(s8x4(xw0), s8x4(ww0)) + dot4(s8x4(xw1), s8x4(ww1)) +
                  dot4(s8x4(xw2), s8x4(ww2)) + dot4(s8x4(xw3), s8x4(ww3));

        uint sw = w_buf.Load((wblock * 34u) & ~3u);
        uint sbits = (wblock & 1u) != 0u ? (sw >> 16u) : sw;
        acc += f16tof32(sbits & 0xffffu) *
               xscale_buf[tok * blocks + b] * (float)dot;
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
