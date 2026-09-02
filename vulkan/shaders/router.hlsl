/* vulkan/shaders/router.hlsl -- MoE router select (Phase 4).
 *
 * One block of 256 threads per token: prob[i] = sqrt(softplus(logit[i]))
 * written to probs, then thread 0 does the insertion-sort top-6 selection
 * (ties -> lower index) with an optional bias, or reads the hash table in
 * hash mode.  Weights are normalized by sum * expert_weight_scale.
 *
 * Bindings: logits (t0), bias (t1), hash (t2), tokens (t3), probs (u0),
 * weights (u1), selected (u2).  bias/hash/tokens are model/tensor ranges
 * bound as ByteAddressBuffer.  Scalar parameters in DS4Params.
 */

#include "common.hlsl"

StructuredBuffer<float> logits_buf : register(t0);
ByteAddressBuffer bias_buf : register(t1);   /* optional (256 f32) */
ByteAddressBuffer hash_buf : register(t2);   /* optional (hash_rows x 6 i32) */
ByteAddressBuffer tokens_buf : register(t3); /* optional (batch token ids) */
RWStructuredBuffer<float> probs_buf : register(u0);
RWStructuredBuffer<float> weights_buf : register(u1);
RWStructuredBuffer<int> selected_buf : register(u2);

[[vk::push_constant]] DS4Params params;

groupshared float sprob[256];

/* Mirrors the CUDA serial router_select_kernel.
 * params: in_dim=n_expert, out_dim=n_expert_used, rows=n_tokens,
 * index=hash_rows, pos0=token (single), weight=expert_weight_scale,
 * flags=has_bias|(hash_mode<<1)|(has_tokens<<2). */
[numthreads(256, 1, 1)]
void router_select(uint3 gid_grp : SV_GroupID,
                   uint tid : SV_GroupThreadID) {
    uint t = gid_grp.x;
    uint n_tokens = params.rows;
    uint n_expert = params.in_dim;
    uint n_used = params.out_dim;
    uint hash_rows = params.index;
    int tok_scalar = (int)params.pos0;
    uint flags = params.flags;
    float scale = params.weight;
    if (t >= n_tokens || tid >= n_expert) return;
    int has_bias = (flags & 1u) != 0u;
    int hash_mode = (flags & 2u) != 0u;
    int has_tokens = (flags & 4u) != 0u;

    float p = sqrt(softplus(logits_buf[t * n_expert + tid]));
    sprob[tid] = p;
    probs_buf[t * n_expert + tid] = p;
    GroupMemoryBarrierWithGroupSync();
    if (tid != 0u) return;

    int tok = has_tokens ? asint(tokens_buf.Load(t * 4u)) : tok_scalar;
    if (tok < 0 || (uint)tok >= hash_rows) tok = 0;

    int sel[6];
    if (hash_mode) {
        for (uint j = 0u; j < n_used; j++) {
            sel[j] = asint(hash_buf.Load(((uint)tok * n_used + j) * 4u));
        }
    } else {
        for (uint j = 0u; j < n_used; j++) sel[j] = -1;
        for (uint e = 0u; e < n_expert; e++) {
            float score = sprob[e] +
                (has_bias ? asfloat(bias_buf.Load(e * 4u)) : 0.0f);
            for (uint j = 0u; j < n_used; j++) {
                int cur = sel[j];
                float cur_score = cur >= 0
                    ? sprob[(uint)cur] +
                          (has_bias ? asfloat(bias_buf.Load((uint)cur * 4u)) : 0.0f)
                    : 0.0f;
                if (cur < 0 || score > cur_score) {
                    for (uint k = n_used - 1u; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = (int)e;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (uint j = 0u; j < n_used; j++) {
        int e = sel[j];
        float v = (e >= 0 && (uint)e < n_expert) ? sprob[(uint)e] : 0.0f;
        selected_buf[t * n_used + j] = e;
        weights_buf[t * n_used + j] = v;
        sum += v;
    }
    sum = max(sum, 6.103515625e-5f);
    for (uint j = 0u; j < n_used; j++) {
        weights_buf[t * n_used + j] = weights_buf[t * n_used + j] / sum * scale;
    }
}
