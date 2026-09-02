/* vulkan/shaders/indexer.hlsl -- indexer score and QAT kernels (Phase 5).
 *
 * indexer_scores: score = scale * sum_h max(0, q[t][h] . index_comp[c]) *
 * weights[t][h].  One block per (comp, token); 256 lanes reduce the dot.
 * Causal masking writes -INF for c >= (pos0 + t + 1) / ratio.
 *
 * dsv4_indexer_qat: in-place Hadamard transform (head_dim==128) followed by
 * per-32-block FP4 (E2M1FN) quantize/dequantize, parity with CUDA
 * indexer_hadamard_fp4_kernel.
 *
 * Bindings: t0 q, t1 weights, t2 index_comp, u0 out.  Scalar parameters in
 * DS4Params.
 */

#include "common.hlsl"

StructuredBuffer<float> q_buf : register(t0);
StructuredBuffer<float> wgt_buf : register(t1);
StructuredBuffer<float> comp_buf : register(t2);
RWStructuredBuffer<float> out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared float isc_partial[256];
groupshared float qat_vals[128];
groupshared float qat_abs[128];

/* scores[t][c] = scale * sum_h max(0, dot) * w[t][h].
 * params: n=n_comp, rows=n_tokens, index=pos0, aux=n_head, blocks=head_dim,
 * ratio=ratio, weight=scale, flags=causal. */
[numthreads(256, 1, 1)]
void indexer_scores(uint3 bid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint c = bid.x;
    uint t = bid.y;
    uint n_comp = params.n;
    uint n_tokens = params.rows;
    if (c >= n_comp || t >= n_tokens) return;
    uint n_head = params.aux;
    uint head_dim = params.blocks;
    if ((params.flags & 1u) != 0u) {
        uint n_visible = (params.index + t + 1u) / params.ratio;
        if (c >= n_visible) {
            if (tid == 0u) out_buf[t * n_comp + c] = -1.0e30f;
            return;
        }
    }
    float total = 0.0f;
    for (uint h = 0u; h < n_head; h++) {
        float dot = 0.0f;
        for (uint d = tid; d < head_dim; d += 256u) {
            dot += q_buf[(t * n_head + h) * head_dim + d] *
                   comp_buf[c * head_dim + d];
        }
        isc_partial[tid] = dot;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                isc_partial[tid] += isc_partial[tid + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        total += max(isc_partial[0], 0.0f) * wgt_buf[t * n_head + h];
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) out_buf[t * n_comp + c] = total * params.weight;
}

/* In-place Hadamard transform + FP4 quantize (head_dim == 128).  One block
 * per row.  params: n=head_dim, rows=n_rows. */
[numthreads(128, 1, 1)]
void dsv4_indexer_qat(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.rows || params.n != 128u) return;
    uint base = row * 128u;
    qat_vals[tid] = out_buf[base + tid];
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint bb = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = qat_vals[bb];
            float b = qat_vals[bb + stride];
            qat_vals[bb] = a + b;
            qat_vals[bb + stride] = a - b;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float v = qat_vals[tid] * 0.08838834764831845f;
    uint fp4_block = tid >> 5u;
    uint lane = tid & 31u;
    uint block_base = fp4_block * 32u;
    qat_abs[tid] = abs(v);
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            qat_abs[block_base + lane] = max(qat_abs[block_base + lane],
                                             qat_abs[block_base + lane + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float amax = max(qat_abs[block_base], 7.052966104933725e-38f);
    float scale = exp2(ceil(log2(amax / 6.0f)));
    out_buf[base + tid] =
        ds4_e2m1fn_dequant(min(6.0f, max(-6.0f, v / scale))) * scale;
}
