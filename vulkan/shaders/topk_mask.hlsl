/* vulkan/shaders/topk_mask.hlsl -- indexer top-k attention mask (Phase 5).
 *
 * mask[t][c] = 0.0 if c is in topk[t], -INF otherwise (parity with the CUDA
 * topk_mask_kernel).  Flat over n_tokens * n_comp.
 *
 * Bindings: t0 topk (uint32), u0 mask.  params: n=n_comp, rows=n_tokens,
 * index=top_k.
 */

#include "common.hlsl"

StructuredBuffer<uint> topk_buf : register(t0);
RWStructuredBuffer<float> mask_buf : register(u0);

[[vk::push_constant]] DS4Params params;

[numthreads(256, 1, 1)]
void topk_mask(uint3 gid : SV_DispatchThreadID) {
    uint n = params.rows * params.n;
    if (gid.x >= n) return;
    uint t = gid.x / params.n;
    uint c = gid.x - t * params.n;
    uint top_k = params.index;
    float v = -1.0e30f;
    for (uint k = 0u; k < top_k; k++) {
        if (topk_buf[(uint64_t)t * top_k + k] == c) {
            v = 0.0f;
            break;
        }
    }
    mask_buf[gid.x] = v;
}
