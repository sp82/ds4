/* vulkan/shaders/argmax.hlsl -- argmax over n_vocab F32 logits.
 *
 * Writes the winning index as int32 at out[0]; ties break toward the lower
 * index (matches the host sample_argmax and CUDA indexer topk path).
 */

#include "common.hlsl"

StructuredBuffer<float> logits_buf : register(t0);
RWStructuredBuffer<int> out_idx_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared float argmax_val[256];
groupshared uint argmax_idx[256];

[numthreads(256, 1, 1)]
void argmax(uint3 gid : SV_DispatchThreadID) {
    float best_v = -3.402823466e38f;
    uint best_i = 0xffffffffu;
    for (uint i = gid.x; i < params.n; i += 256u) {
        float v = logits_buf[i];
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
    }
    argmax_val[gid.x] = best_v;
    argmax_idx[gid.x] = best_i;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (gid.x < stride) {
            float vb = argmax_val[gid.x + stride];
            uint ib = argmax_idx[gid.x + stride];
            if (vb > argmax_val[gid.x] ||
                (vb == argmax_val[gid.x] && ib < argmax_idx[gid.x])) {
                argmax_val[gid.x] = vb;
                argmax_idx[gid.x] = ib;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gid.x == 0u) {
        out_idx_buf[0] = (int)argmax_idx[0];
    }
}
