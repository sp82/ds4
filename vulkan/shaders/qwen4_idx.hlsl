/* vulkan/shaders/qwen4_idx.hlsl -- Qwen3.8-Flash-Next QSA indexer.
 *
 *   idx_score    score[t][b] = sum_h relu(q_h . key_b); invisible blocks -3e38
 *   idx_tile_max per-8-block max of max(score,0) as a uint key (radix prefilter)
 *   idx_select   exact 4-pass radix top-k (stable greater-than then equal)
 *   idx_expand   selected blocks -> token list + contiguous tail
 *
 * The radix select is a new kernel: indexer_topk.hlsl (serial insertion) does
 * not cover this.  One warp per (token, block) keeps the groupshared reduction
 * free of block-wide divergence. */

#include "common.hlsl"

[[vk::push_constant]] DS4Params params;

/* ---- idx_score --------------------------------------------------------- */

StructuredBuffer<float> idx_q : register(t0);
ByteAddressBuffer idx_key : register(t1);      /* f16 [N][D] */
RWStructuredBuffer<float> idx_out : register(u0);

groupshared float idx_red[32];

/* grid (N, T), 32 threads.  params.n = N, rows = T, in_dim = H, out_dim = D,
 * pos0, ratio. */
[numthreads(32, 1, 1)]
void idx_score(uint3 gid : SV_GroupID, uint lane : SV_GroupThreadID) {
    const uint b = gid.x, t = gid.y;
    const uint N = params.n, H = params.in_dim, D = params.out_dim;
    const uint pos0 = params.pos0, ratio = params.ratio;
    float score;
    if (b >= (pos0 + t + 1u) / ratio) {
        score = -3e38f;
    } else {
        score = 0.0f;
        for (uint h = 0u; h < H; h++) {
            float a = 0.0f;
            for (uint i = lane; i < D; i += 32u)
                a += idx_q[(t * H + h) * D + i] *
                     f16_at(idx_key, (b * D + i) * 2u);
            idx_red[lane] = a;
            GroupMemoryBarrierWithGroupSync();
            if (lane == 0u) {
                float s = 0.0f;
                for (uint j = 0u; j < 32u; j++) s += idx_red[j];
                idx_red[0] = s;
            }
            GroupMemoryBarrierWithGroupSync();
            score += max(idx_red[0], 0.0f);
            GroupMemoryBarrierWithGroupSync();
        }
    }
    if (lane == 0u) idx_out[t * N + b] = score;
}

/* ---- idx_tile_max ------------------------------------------------------ */

StructuredBuffer<float> tm_score : register(t0);
RWStructuredBuffer<uint> tm_out : register(u0);

/* grid ((tiles+255)/256, T), 256 threads.  params.n = N, blocks = tiles. */
[numthreads(256, 1, 1)]
void idx_tile_max(uint3 gid : SV_DispatchThreadID) {
    const uint tile = gid.x, t = gid.y;
    const uint N = params.n, tiles = params.blocks;
    if (tile >= tiles) return;
    uint v = 0u;
    const uint end = min(N, tile * 8u + 8u);
    for (uint i = tile * 8u; i < end; i++)
        v = max(v, asuint(max(tm_score[t * N + i], 0.0f)));
    tm_out[t * tiles + tile] = v;
}

/* ---- idx_select -------------------------------------------------------- */

StructuredBuffer<float> sel_score : register(t0);
RWStructuredBuffer<int> sel_out : register(u0);

groupshared uint sel_hist[256];
groupshared uint sel_gt[256];
groupshared uint sel_eq[256];
groupshared uint sel_threshold;
groupshared uint sel_need;

/* grid (T), 256 threads.  params.n = N, out_dim = K.  Exact radix threshold
 * then stable gather (elements > threshold first, then == threshold). */
[numthreads(256, 1, 1)]
void idx_select(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    const uint t = gid.x;
    const uint N = params.n, K = params.out_dim;
    if (tid == 0u) { sel_threshold = 0u; sel_need = K; }
    GroupMemoryBarrierWithGroupSync();

    for (uint pass = 0u; pass < 4u; pass++) {
        const uint shift = 24u - 8u * pass;
        const uint mask = pass ? (0xffffffffu << (shift + 8u)) : 0u;
        sel_hist[tid] = 0u;
        GroupMemoryBarrierWithGroupSync();
        for (uint i = tid; i < N; i += 256u) {
            const uint key = asuint(max(sel_score[t * N + i], 0.0f));
            if ((key & mask) == sel_threshold) {
                uint orig;
                InterlockedAdd(sel_hist[(key >> shift) & 255u], 1u, orig);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid == 0u) {
            for (int d = 255; d >= 0; d--) {
                if (sel_hist[d] >= sel_need) {
                    sel_threshold |= (uint)d << shift;
                    break;
                }
                sel_need -= sel_hist[d];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const uint chunk = (N + 255u) / 256u;
    const uint begin = min(N, tid * chunk), end = min(N, begin + chunk);
    uint ng = 0u, ne = 0u;
    for (uint i = begin; i < end; i++) {
        const uint key = asuint(max(sel_score[t * N + i], 0.0f));
        ng += key > sel_threshold ? 1u : 0u;
        ne += key == sel_threshold ? 1u : 0u;
    }
    sel_gt[tid] = ng;
    sel_eq[tid] = ne;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u) {
        uint pg = 0u, pe = 0u;
        for (uint i = 0u; i < 256u; i++) {
            const uint g = sel_gt[i], e = sel_eq[i];
            sel_gt[i] = pg;
            sel_eq[i] = pe;
            pg += g;
            pe += e;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint g = sel_gt[tid], e = sel_eq[tid];
    for (uint i = begin; i < end; i++) {
        const uint key = asuint(max(sel_score[t * N + i], 0.0f));
        if (key > sel_threshold) {
            sel_out[t * K + g] = (int)i;
            g++;
        } else if (key == sel_threshold) {
            if (e < sel_need) sel_out[t * K + K - sel_need + e] = (int)i;
            e++;
        }
    }
}

/* ---- idx_expand -------------------------------------------------------- */

StructuredBuffer<int> exp_blocks : register(t0);
RWStructuredBuffer<int> exp_out : register(u0);
RWStructuredBuffer<uint> exp_count : register(u1);

/* grid (T), 256 threads.  params.n = K, in_dim = stride, ratio, pos0. */
[numthreads(256, 1, 1)]
void idx_expand(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    const uint t = gid.x;
    const uint K = params.n, ratio = params.ratio;
    const uint pos0 = params.pos0, stride = params.in_dim;
    const uint pos = pos0 + t, tail = (pos + 1u) / ratio * ratio;
    for (uint i = tid; i < K * ratio; i += 256u)
        exp_out[t * stride + i] =
            exp_blocks[t * K + i / ratio] * (int)ratio + (int)(i % ratio);
    for (uint i = tail + tid; i <= pos; i += 256u)
        exp_out[t * stride + K * ratio + i - tail] = (int)i;
    if (tid == 0u) exp_count[t] = K * ratio + pos + 1u - tail;
}
