/* vulkan/shaders/dspark.hlsl -- DSpark Markov argmax (Phase 5).
 *
 * out = argmax_i(logits[i] + dot(w2[i], w1[prev_token])) over the vocab,
 * entirely on-device.  w1_row and w2 are Q8_0 (34-byte blocks).  Parity with
 * the CUDA dspark_markov_argmax_kernel, but dispatched with a single block:
 * the block-local reduction yields the global best and thread 0 writes the
 * packed uint64 key directly (no 64-bit atomics needed).
 *
 * Key format (little-endian): [0] = ~best_idx (low 32), [1] = monotonic
 * float key (high 32), matching the host's unsigned long long decode.
 *
 * Bindings: t0 logits, t3 model (ByteAddressBuffer: w1_row + w2), u0 out_key
 * (uint32[2]).  params: n=vocab, blocks=rank_blocks, aux=w1 offset (bytes,
 * relative to the bound model range), ratio=w2 offset (bytes, relative).
 */

#include "common.hlsl"

StructuredBuffer<float> logits_buf : register(t0);
ByteAddressBuffer w_buf : register(t3);
RWStructuredBuffer<uint> out_key_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared float mk_state[256];
groupshared float mk_vals[256];
groupshared uint mk_idxs[256];

bool mk_better(float av, uint ai, float bv, uint bi) {
    return av > bv || (av == bv && ai < bi);
}

[numthreads(256, 1, 1)]
void dspark_markov_argmax(uint tid : SV_GroupThreadID) {
    uint vocab = params.n;
    uint rank_blocks = params.blocks;
    if (tid < rank_blocks * 32u) {
        uint block = tid >> 5u;
        uint lane = tid & 31u;
        uint base = params.aux + block * 34u;
        mk_state[tid] = f16_at(w_buf, base) * (float)s8_at(w_buf, base + 2u + lane);
    }
    GroupMemoryBarrierWithGroupSync();

    float best_v = -1.0e30f;
    uint best_i = 0u;
    for (uint i = tid; i < vocab; i += 256u) {
        uint base = params.ratio + i * rank_blocks * 34u;
        float acc = 0.0f;
        for (uint block = 0u; block < rank_blocks; block++) {
            uint qb = base + block * 34u;
            float d = f16_at(w_buf, qb);
            float s = 0.0f;
            for (uint lane = 0u; lane < 32u; lane++) {
                s += (float)s8_at(w_buf, qb + 2u + lane) * mk_state[block * 32u + lane];
            }
            acc += d * s;
        }
        float v = logits_buf[i] + acc;
        if (mk_better(v, i, best_v, best_i)) {
            best_v = v;
            best_i = i;
        }
    }

    mk_vals[tid] = best_v;
    mk_idxs[tid] = best_i;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride &&
            mk_better(mk_vals[tid + stride], mk_idxs[tid + stride],
                      mk_vals[tid], mk_idxs[tid])) {
            mk_vals[tid] = mk_vals[tid + stride];
            mk_idxs[tid] = mk_idxs[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        uint bits = asuint(mk_vals[0]);
        uint fkey = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        out_key_buf[0] = ~mk_idxs[0];
        out_key_buf[1] = fkey;
    }
}
