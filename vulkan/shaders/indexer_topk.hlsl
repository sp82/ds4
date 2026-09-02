/* vulkan/shaders/indexer_topk.hlsl -- indexer top-k selection (Phase 5).
 *
 * indexer_topk: serial insertion sort of one token row, one thread per
 * token.  Handles any top_k (parity with the CUDA indexer_topk_kernel;
 * large top_k is a perf tuning item, not a correctness one).
 *
 * indexer_top1_value: per-token block reduce for top1 with an index offset
 * applied to the tie-break (parity with indexer_top1_value_kernel).
 *
 * Bindings: t0 scores, u0 selected (uint32), u1 values (float, top1 only).
 */

#include "common.hlsl"

StructuredBuffer<float> scores_buf : register(t0);
RWStructuredBuffer<uint> sel_buf : register(u0);
RWStructuredBuffer<float> val_buf : register(u1);

[[vk::push_constant]] DS4Params params;

groupshared float tk_vals[256];
groupshared uint tk_idxs[256];

bool tk_better(float av, uint ai, float bv, uint bi) {
    return av > bv || (av == bv && ai < bi);
}

/* selected[t][0..top_k) = descending top-k indices of scores[t].
 * params: n=n_comp, rows=n_tokens, index=top_k. */
[numthreads(1, 1, 1)]
void indexer_topk(uint3 gid : SV_DispatchThreadID) {
    uint t = gid.x;
    if (t >= params.rows) return;
    uint n_comp = params.n;
    uint top_k = params.index;
    if (top_k == 0u || top_k > n_comp) return;
    uint base = t * n_comp;
    uint out_base = t * top_k;
    for (uint k = 0u; k < top_k; k++) sel_buf[out_base + k] = 0u;
    for (uint c = 0u; c < n_comp; c++) {
        float v = scores_buf[base + c];
        for (uint k = 0u; k < top_k; k++) {
            if (k >= c || v > scores_buf[base + sel_buf[out_base + k]]) {
                for (uint j = top_k - 1u; j > k; j--) {
                    sel_buf[out_base + j] = sel_buf[out_base + j - 1u];
                }
                sel_buf[out_base + k] = c;
                break;
            }
        }
    }
}

/* selected[t] = argmax (with index_offset), values[t] = its score.
 * params: n=n_comp, rows=n_tokens, index=index_offset. */
[numthreads(256, 1, 1)]
void indexer_top1_value(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint t = gid.x;
    if (t >= params.rows) return;
    uint n_comp = params.n;
    uint index_offset = params.index;
    uint base = t * n_comp;
    float best_v = -1.0e30f;
    uint best_i = 0u;
    for (uint i = tid; i < n_comp; i += 256u) {
        float v = scores_buf[base + i];
        uint gi = index_offset + i;
        uint bgi = index_offset + best_i;
        if (tk_better(v, gi, best_v, bgi)) {
            best_v = v;
            best_i = i;
        }
    }
    tk_vals[tid] = best_v;
    tk_idxs[tid] = best_i;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float ov = tk_vals[tid + stride];
            uint oi = tk_idxs[tid + stride];
            uint gi = index_offset + tk_idxs[tid];
            if (tk_better(ov, index_offset + oi, tk_vals[tid], gi)) {
                tk_vals[tid] = ov;
                tk_idxs[tid] = oi;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        sel_buf[t] = index_offset + tk_idxs[0];
        val_buf[t] = tk_vals[0];
    }
}
