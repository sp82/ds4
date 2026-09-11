/* vulkan/shaders/attention.hlsl -- DS4 attention kernels.
 *
 * Decode (attn_decode) implements the mixed decode semantics of the CUDA
 * attention_decode_mixed_kernel fallback: per (token, head) block, scores
 * over a raw ring-buffer span plus optionally compressed rows, online
 * softmax with the attention sink, then a weighted sum over the values.
 * Scores are staged in a global scratch buffer (binding u2) instead of a
 * huge groupshared array so large contexts never exceed the shared-memory
 * limit.
 *
 * Prefill (attn_prefill) implements the causal mixed kernel: raw rows are
 * read directly from raw_kv (rows [raw_start, t]), compressed rows from
 * comp_kv, with an optional per-token comp mask.
 *
 * Bindings: q (t0), raw_kv (t1), comp_kv (t2), sinks (t3 = model),
 * heads out (u0), comp_mask (u1), scores scratch (u2).
 */

#include "common.hlsl"

StructuredBuffer<float> q_buf : register(t0);
StructuredBuffer<float> raw_buf : register(t1);
StructuredBuffer<float> comp_buf : register(t2);
ByteAddressBuffer sink_buf : register(t3);
RWStructuredBuffer<float> heads_buf : register(u0);
RWStructuredBuffer<float> mask_buf : register(u1);
RWStructuredBuffer<float> score_buf : register(u2);
RWStructuredBuffer<uint> topk_buf : register(u3);

groupshared uint attn_idx_valid[256];
groupshared uint attn_comp_rows[512];
groupshared uint attn_comp_count;

[[vk::push_constant]] DS4Params params;

groupshared float attn_partial[256];
groupshared float attn_max_s;
groupshared float attn_denom;
groupshared uint attn_raw_count;
groupshared uint attn_raw_first;

float attn_sink(uint h) {
    return asfloat(sink_buf.Load(h * 4u));
}

/* Mixed decode: one block of 256 threads per (token, head).
 * params: n=head_dim, rows=n_head, in_dim=n_tokens, out_dim=n_raw,
 * n_rot=raw_cap, pos0=pos0, index=raw_start, blocks=n_comp, aux=window,
 * ratio=ratio, flags=use_comp_mask, rsvd2=score slab width. */
[numthreads(256, 1, 1)]
void attn_decode(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint t = gid.x;
    uint h = gid.y;
    uint n_head = params.rows;
    uint head_dim = params.n;
    uint n_tokens = params.in_dim;
    uint n_raw = params.out_dim;
    uint raw_cap = params.n_rot;
    uint pos0 = params.pos0;
    uint raw_start = params.index;
    uint n_comp = params.blocks;
    uint window = params.aux;
    uint ratio = params.ratio;
    uint use_comp_mask = params.flags;
    uint slab = params.rsvd2;
    if (t >= n_tokens || h >= n_head) return;

    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint qpos = pos0 + t;
    uint first_raw_pos = pos0 + n_tokens - n_raw;
    uint visible_comp = single_all ? n_comp
                                   : (n_comp ? (qpos + 1u) / ratio : 0u);
    visible_comp = min(visible_comp, n_comp);

    uint raw_count = 0u;
    uint raw_first_idx = 0u;
    if (n_raw != 0u) {
        uint raw_last_pos = first_raw_pos + n_raw - 1u;
        if (single_all) {
            raw_count = min(n_raw, 256u);
        } else if (qpos >= first_raw_pos) {
            uint lo = first_raw_pos;
            if (window != 0u && qpos + 1u > window) {
                lo = max(lo, qpos + 1u - window);
            }
            uint hi = min(qpos, raw_last_pos);
            if (hi >= lo) {
                raw_first_idx = lo - first_raw_pos;
                raw_count = min(hi - lo + 1u, 256u);
            }
        }
    }

    float scale = rsqrt((float)head_dim);
    uint qbase = (t * n_head + h) * head_dim;
    uint sbase = (t * n_head + h) * slab;
    float local_max = attn_sink(h);

    for (uint r = tid; r < raw_count; r += 256u) {
        uint row = (raw_start + raw_first_idx + r) % raw_cap;
        uint kvbase = row * head_dim;
        float dot = 0.0f;
        for (uint d = 0u; d < head_dim; d++) {
            dot += q_buf[qbase + d] * raw_buf[kvbase + d];
        }
        score_buf[sbase + r] = dot * scale;
        local_max = max(local_max, score_buf[sbase + r]);
    }
    for (uint c = tid; c < visible_comp; c += 256u) {
        float add = use_comp_mask ? mask_buf[t * n_comp + c] : 0.0f;
        float s = -3.402823466e38f;
        if (add > -1.0e20f) {
            uint kvbase = c * head_dim;
            float dot = 0.0f;
            for (uint d = 0u; d < head_dim; d++) {
                dot += q_buf[qbase + d] * comp_buf[kvbase + d];
            }
            s = dot * scale + add;
        }
        score_buf[sbase + raw_count + c] = s;
        local_max = max(local_max, s);
    }

    attn_partial[tid] = local_max;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            attn_partial[tid] = max(attn_partial[tid],
                                    attn_partial[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) attn_max_s = attn_partial[0];
    GroupMemoryBarrierWithGroupSync();

    float den_local = 0.0f;
    uint n_score = raw_count + visible_comp;
    for (uint i = tid; i < n_score; i += 256u) {
        score_buf[sbase + i] = exp(score_buf[sbase + i] - attn_max_s);
        den_local += score_buf[sbase + i];
    }
    attn_partial[tid] = den_local;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            attn_partial[tid] += attn_partial[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        attn_denom = attn_partial[0] + exp(attn_sink(h) - attn_max_s);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint d = tid; d < head_dim; d += 256u) {
        float acc = 0.0f;
        for (uint r = 0u; r < raw_count; r++) {
            uint row = (raw_start + raw_first_idx + r) % raw_cap;
            acc += raw_buf[row * head_dim + d] * score_buf[sbase + r];
        }
        for (uint c = 0u; c < visible_comp; c++) {
            acc += comp_buf[c * head_dim + d] * score_buf[sbase + raw_count + c];
        }
        heads_buf[qbase + d] = acc / attn_denom;
    }
}

/* Indexed mixed decode: per (token, head) block over the raw ring span plus
 * the compressed rows selected by the indexer topk (not all visible rows).
 * The per-token topk list is compacted stably into comp_rows[] (candidates
 * c in [0, visible_comp) kept in topk order) with a Hillis-Steele scan over
 * the valid mask, mirroring attention_indexed_mixed_kernel in ds4_cuda.cu.
 *
 * Bindings: q (t0), raw_kv (t1), comp_kv (t2), sinks (t3), heads out (u0),
 * scores scratch (u2), topk (u3).
 * params: n=head_dim, rows=n_head, in_dim=n_tokens, out_dim=n_raw,
 * n_rot=raw_cap, pos0=pos0, index=raw_start, blocks=n_comp, aux=window,
 * ratio=ratio, flags=top_k, rsvd2=score slab width. */
[numthreads(256, 1, 1)]
void attn_indexed_decode(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint t = gid.x;
    uint h = gid.y;
    uint n_head = params.rows;
    uint head_dim = params.n;
    uint n_tokens = params.in_dim;
    uint n_raw = params.out_dim;
    uint raw_cap = params.n_rot;
    uint pos0 = params.pos0;
    uint raw_start = params.index;
    uint n_comp = params.blocks;
    uint window = params.aux;
    uint ratio = params.ratio;
    uint top_k = params.flags;
    uint slab = params.rsvd2;
    if (t >= n_tokens || h >= n_head) return;

    uint qpos = pos0 + t;
    uint first_raw_pos = pos0 + n_tokens - n_raw;
    uint visible_comp = n_comp;
    if (ratio != 0u) {
        visible_comp = (qpos + 1u) / ratio;
        visible_comp = min(visible_comp, n_comp);
    }

    uint raw_count = 0u;
    uint raw_first_idx = 0u;
    if (n_raw != 0u) {
        uint raw_last_pos = first_raw_pos + n_raw - 1u;
        if (qpos >= first_raw_pos) {
            uint lo = first_raw_pos;
            if (window != 0u && qpos + 1u > window) {
                lo = max(lo, qpos + 1u - window);
            }
            uint hi = min(qpos, raw_last_pos);
            if (hi >= lo) {
                raw_first_idx = lo - first_raw_pos;
                raw_count = min(hi - lo + 1u, 256u);
            }
        }
    }

    /* Stable compaction of the per-token topk into comp_rows[]. */
    uint topk_base = t * top_k;
    uint my_valid0 = 0u;
    uint my_cand0 = 0u;
    if (tid < top_k) {
        int c = (int)topk_buf[topk_base + tid];
        if (c >= 0 && (uint)c < visible_comp) {
            my_valid0 = 1u;
            my_cand0 = (uint)c;
        }
    }
    uint my_valid1 = 0u;
    uint my_cand1 = 0u;
    if (tid + 256u < top_k) {
        int c = (int)topk_buf[topk_base + tid + 256u];
        if (c >= 0 && (uint)c < visible_comp) {
            my_valid1 = 1u;
            my_cand1 = (uint)c;
        }
    }

    attn_idx_valid[tid] = my_valid0;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        uint v = tid >= stride ? attn_idx_valid[tid - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        if (tid >= stride) attn_idx_valid[tid] += v;
        GroupMemoryBarrierWithGroupSync();
    }
    uint comp_total0 = attn_idx_valid[255];
    if (my_valid0) {
        uint pos = attn_idx_valid[tid] - my_valid0;
        attn_comp_rows[pos] = my_cand0;
    }
    GroupMemoryBarrierWithGroupSync();

    attn_idx_valid[tid] = my_valid1;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        uint v = tid >= stride ? attn_idx_valid[tid - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        if (tid >= stride) attn_idx_valid[tid] += v;
        GroupMemoryBarrierWithGroupSync();
    }
    if (my_valid1) {
        uint pos = comp_total0 + (attn_idx_valid[tid] - my_valid1);
        attn_comp_rows[pos] = my_cand1;
    }
    if (tid == 0u) attn_comp_count = comp_total0 + attn_idx_valid[255];
    GroupMemoryBarrierWithGroupSync();
    uint comp_count = attn_comp_count;

    float scale = rsqrt((float)head_dim);
    uint qbase = (t * n_head + h) * head_dim;
    uint sbase = (t * n_head + h) * slab;
    float local_max = attn_sink(h);

    for (uint r = tid; r < raw_count; r += 256u) {
        uint row = (raw_start + raw_first_idx + r) % raw_cap;
        uint kvbase = row * head_dim;
        float dot = 0.0f;
        for (uint d = 0u; d < head_dim; d++) {
            dot += q_buf[qbase + d] * raw_buf[kvbase + d];
        }
        score_buf[sbase + r] = dot * scale;
        local_max = max(local_max, score_buf[sbase + r]);
    }
    for (uint c = tid; c < comp_count; c += 256u) {
        uint row = attn_comp_rows[c];
        uint kvbase = row * head_dim;
        float dot = 0.0f;
        for (uint d = 0u; d < head_dim; d++) {
            dot += q_buf[qbase + d] * comp_buf[kvbase + d];
        }
        score_buf[sbase + raw_count + c] = dot * scale;
        local_max = max(local_max, score_buf[sbase + raw_count + c]);
    }

    attn_partial[tid] = local_max;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            attn_partial[tid] = max(attn_partial[tid],
                                    attn_partial[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) attn_max_s = attn_partial[0];
    GroupMemoryBarrierWithGroupSync();

    float den_local = 0.0f;
    uint n_score = raw_count + comp_count;
    for (uint i = tid; i < n_score; i += 256u) {
        score_buf[sbase + i] = exp(score_buf[sbase + i] - attn_max_s);
        den_local += score_buf[sbase + i];
    }
    attn_partial[tid] = den_local;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            attn_partial[tid] += attn_partial[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        attn_denom = attn_partial[0] + exp(attn_sink(h) - attn_max_s);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint d = tid; d < head_dim; d += 256u) {
        float acc = 0.0f;
        for (uint r = 0u; r < raw_count; r++) {
            uint row = (raw_start + raw_first_idx + r) % raw_cap;
            acc += raw_buf[row * head_dim + d] * score_buf[sbase + r];
        }
        for (uint c = 0u; c < comp_count; c++) {
            uint row = attn_comp_rows[c];
            acc += comp_buf[row * head_dim + d] * score_buf[sbase + raw_count + c];
        }
        heads_buf[qbase + d] = acc / attn_denom;
    }
}

/* Causal prefill: one block of 256 threads per (token, head).
 * params: n=head_dim, rows=n_head, in_dim=n_tokens, out_dim=n_comp,
 * index=window, aux=ratio, flags=use_comp_mask, rsvd2=score slab width.
 * raw_kv is read directly (rows [raw_start, t]); comp rows from comp_kv. */
[numthreads(256, 1, 1)]
void attn_prefill(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint t = gid.x;
    uint h = gid.y;
    uint n_head = params.rows;
    uint head_dim = params.n;
    uint n_tokens = params.in_dim;
    uint n_comp = params.out_dim;
    uint window = params.index;
    uint ratio = params.aux;
    uint use_comp_mask = params.flags;
    uint slab = params.rsvd2;
    if (t >= n_tokens || h >= n_head) return;

    uint raw_start = (window != 0u && t + 1u > window) ? t + 1u - window : 0u;
    uint raw_count = t + 1u - raw_start;
    uint visible_comp = n_comp ? (t + 1u) / ratio : 0u;
    visible_comp = min(visible_comp, n_comp);
    uint n_score = raw_count + visible_comp;

    float scale = rsqrt((float)head_dim);
    uint qbase = (t * n_head + h) * head_dim;
    uint sbase = (t * n_head + h) * slab;
    float local_max = attn_sink(h);

    for (uint r = tid; r < raw_count; r += 256u) {
        uint kvbase = (raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint d = 0u; d < head_dim; d++) {
            dot += q_buf[qbase + d] * raw_buf[kvbase + d];
        }
        score_buf[sbase + r] = dot * scale;
        local_max = max(local_max, score_buf[sbase + r]);
    }
    for (uint c = tid; c < visible_comp; c += 256u) {
        float add = use_comp_mask ? mask_buf[t * n_comp + c] : 0.0f;
        float s = -3.402823466e38f;
        if (add > -1.0e20f) {
            uint kvbase = c * head_dim;
            float dot = 0.0f;
            for (uint d = 0u; d < head_dim; d++) {
                dot += q_buf[qbase + d] * comp_buf[kvbase + d];
            }
            s = dot * scale + add;
        }
        score_buf[sbase + raw_count + c] = s;
        local_max = max(local_max, s);
    }

    attn_partial[tid] = local_max;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            attn_partial[tid] = max(attn_partial[tid],
                                    attn_partial[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) attn_max_s = attn_partial[0];
    GroupMemoryBarrierWithGroupSync();

    float den_local = 0.0f;
    for (uint i = tid; i < n_score; i += 256u) {
        score_buf[sbase + i] = exp(score_buf[sbase + i] - attn_max_s);
        den_local += score_buf[sbase + i];
    }
    attn_partial[tid] = den_local;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            attn_partial[tid] += attn_partial[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        attn_denom = attn_partial[0] + exp(attn_sink(h) - attn_max_s);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint d = tid; d < head_dim; d += 256u) {
        float acc = 0.0f;
        for (uint r = 0u; r < raw_count; r++) {
            acc += raw_buf[(raw_start + r) * head_dim + d] * score_buf[sbase + r];
        }
        for (uint c = 0u; c < visible_comp; c++) {
            acc += comp_buf[c * head_dim + d] * score_buf[sbase + raw_count + c];
        }
        heads_buf[qbase + d] = acc / attn_denom;
    }
}

/* Low-rank projection of the attention output (grouped Q8_0):
 * low[t][g*rank + r] = sum_d heads[t][g*group_dim + d] * out_a[(g*rank+r)][d].
 *
 * One block of 256 threads per output element; row = g*rank + r.  The model
 * buffer (t3) is bound at out_a's byte offset.  The group slice
 * [group0, group0+group_cnt) lets rows_exact reuse the same kernel.
 *
 * params: n=group_dim, in_dim=group_dim, out_dim=rank, blocks=group_dim/32,
 * index=n_groups (total groups in heads), aux=group0, n_rot=group_cnt,
 * rows=n_rows, rsvd2=group_dim (in floats, aliased via n). */
[numthreads(256, 1, 1)]
void attn_output_low_q8(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row = gid.x;            /* local output row: 0 .. group_cnt*rank-1 */
    uint t = gid.y;
    uint rank = params.out_dim;
    uint group_dim = params.n;
    uint blocks = params.blocks;
    uint n_groups_total = params.index;
    uint group0 = params.aux;
    uint group_cnt = params.n_rot;
    uint n_rows = params.rows;
    if (row >= group_cnt * rank || t >= n_rows) return;

    uint g = group0 + row / rank;
    uint r = row % rank;
    uint wrow = g * rank + r;    /* out_a row index */
    uint xbase = t * n_groups_total * group_dim + g * group_dim;

    float acc = 0.0f;
    for (uint b = tid; b < blocks; b += 256u) {
        uint i0 = b * 32u;
        uint bn = min(32u, group_dim - i0);
        uint wblock = wrow * blocks + b;
        float amax = 0.0f;
        for (uint i = 0u; i < bn; i++) {
            amax = max(amax, abs(q_buf[xbase + i0 + i]));
        }
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        int dot = 0;
        for (uint i = 0u; i < bn; i++) {
            int q = (int)round(q_buf[xbase + i0 + i] * id);
            q = min(q, 127);
            q = max(q, -128);
            dot += q8_s8(sink_buf, wblock, i) * q;
        }
        acc += q8_scale(sink_buf, wblock) * d * (float)dot;
    }

    attn_partial[tid] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            attn_partial[tid] += attn_partial[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        heads_buf[t * (group_cnt * rank) + row] = attn_partial[0];
    }
}

/* v2: packed-dot variant of attn_output_low_q8.  Same math and output as v1
 * (activation quantized inline identically: amax/127, round, clamp), but the
 * Q8 weight is read as 32-bit words and the int8 dot uses dot4add_i8packed
 * (OpSDot, requires VK_KHR_shader_integer_dot_product, compiled as cs_6_4).
 * The host selects this only when the device exposes the feature. */
[numthreads(256, 1, 1)]
void attn_output_low_q8_v2(uint3 gid : SV_GroupID,
                           uint tid : SV_GroupThreadID) {
    uint row = gid.x;
    uint t = gid.y;
    uint rank = params.out_dim;
    uint group_dim = params.n;
    uint blocks = params.blocks;
    uint n_groups_total = params.index;
    uint group0 = params.aux;
    uint group_cnt = params.n_rot;
    uint n_rows = params.rows;
    if (row >= group_cnt * rank || t >= n_rows) return;

    uint g = group0 + row / rank;
    uint r = row % rank;
    uint wrow = g * rank + r;
    uint xbase = t * n_groups_total * group_dim + g * group_dim;

    float acc = 0.0f;
    for (uint b = tid; b < blocks; b += 256u) {
        uint i0 = b * 32u;
        uint bn = min(32u, group_dim - i0);
        float amax = 0.0f;
        for (uint i = 0u; i < bn; i++) {
            amax = max(amax, abs(q_buf[xbase + i0 + i]));
        }
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        uint qw[8];
        for (uint j = 0u; j < 8u; j++) {
            uint packed = 0u;
            for (uint k = 0u; k < 4u; k++) {
                uint idx = j * 4u + k;
                int q = 0;
                if (idx < bn) {
                    q = (int)round(q_buf[xbase + i0 + idx] * id);
                    q = min(q, 127);
                    q = max(q, -128);
                }
                packed |= ((uint)(q & 0xff)) << (k * 8u);
            }
            qw[j] = packed;
        }
        uint wblock = wrow * blocks + b;
        uint w_byte = wblock * 34u + 2u;
        uint ww[8];
        if ((wblock & 1u) != 0u) {
            for (uint j = 0u; j < 8u; j++) {
                ww[j] = sink_buf.Load(w_byte + j * 4u);
            }
        } else {
            uint prev = sink_buf.Load(w_byte - 2u);
            for (uint j = 0u; j < 8u; j++) {
                uint nxt = sink_buf.Load(w_byte + 2u + j * 4u);
                ww[j] = (prev >> 16u) | (nxt << 16u);
                prev = nxt;
            }
        }
        int dot = 0;
        for (uint j = 0u; j < 8u; j++) {
            dot = dot4add_i8packed(qw[j], ww[j], dot);
        }
        acc += q8_scale(sink_buf, wblock) * d * (float)dot;
    }

    attn_partial[tid] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            attn_partial[tid] += attn_partial[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        heads_buf[t * (group_cnt * rank) + row] = attn_partial[0];
    }
}

/* Low-rank projection of the attention output (grouped Q4_K): same math and
 * layout as attn_output_low_q8, but out_a is Q4_K (144-byte 256-value blocks)
 * and the activation heads are used raw (f32, no inline Q8 quantization).
 *
 * params: n=group_dim, in_dim=group_dim, out_dim=rank, blocks=group_dim/256,
 * index=n_groups (total groups in heads), aux=group0, n_rot=group_cnt,
 * rows=n_rows.  group_dim must be a multiple of 256. */
[numthreads(256, 1, 1)]
void attn_output_low_q4k(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row = gid.x;            /* local output row: 0 .. group_cnt*rank-1 */
    uint t = gid.y;
    uint rank = params.out_dim;
    uint group_dim = params.n;
    uint blocks = params.blocks;
    uint n_groups_total = params.index;
    uint group0 = params.aux;
    uint group_cnt = params.n_rot;
    uint n_rows = params.rows;
    if (row >= group_cnt * rank || t >= n_rows) return;

    uint local_g = row / rank;
    uint r = row % rank;
    uint g = group0 + local_g;
    uint wrow = local_g * rank + r;    /* out_a row index within the slice */
    uint xbase = t * n_groups_total * group_dim + g * group_dim;

    float acc = 0.0f;
    for (uint b = tid; b < blocks; b += 256u) {
        uint bbase = (wrow * blocks + b) * 144u;
        uint xb = xbase + b * 256u;
        for (uint j = 0u; j < 8u; j++) {
            acc += q4k_sub_dot(sink_buf, bbase, j, q_buf, xb + j * 32u);
        }
    }

    attn_partial[tid] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            attn_partial[tid] += attn_partial[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        heads_buf[t * (group_cnt * rank) + row] = attn_partial[0];
    }
}
