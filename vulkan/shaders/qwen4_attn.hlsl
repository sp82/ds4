/* vulkan/shaders/qwen4_attn.hlsl -- Qwen3.8-Flash-Next full attention.
 *
 *   attn_prep    normalise q/k (+rope, +gamma), copy gate, write k/v/indexer
 *                caches, and the raw indexer key
 *   block_key    mean of `ratio` indexer rows, RMSNorm, rope, f16 output
 *   attn_decode  online-softmax attention, one warp per (head, token, split)
 *   attn_merge   reconnect the split partial softmaxes
 *
 * Multimodal NeoX-interleaved rope: pair i uses position component i%3
 * (t,h,w); text repeats one position.  Frequencies come from the host table
 * (ds4_gpu_qwen4_set_rope) when set, else from `base`.  K/V caches are f16,
 * written as packed 32-bit words so no two lanes race on one word. */

#include "common.hlsl"

[[vk::push_constant]] DS4Params params;

float qwen_sigmoid(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + exp(-x)) : exp(x) / (1.0f + exp(x));
}

/* Effective rope frequency i (i < nrot/2): table when set, else base. */
float qwen_freq(ByteAddressBuffer tab, uint i, float base, uint nrot, uint set) {
    if (set != 0u) return asfloat(tab.Load(i * 4u));
    return pow(base, -2.0f * (float)i / (float)nrot);
}

/* ---- attn_prep --------------------------------------------------------- */

RWStructuredBuffer<float> ap_q : register(u0);
RWStructuredBuffer<float> ap_gate : register(u1);
RWByteAddressBuffer ap_kc : register(u2);   /* f16 [cap][Hkv][D] */
RWByteAddressBuffer ap_vc : register(u3);
RWStructuredBuffer<float> ap_iq : register(u4);
RWStructuredBuffer<float> ap_ikc : register(u5);  /* f32 [cap][Di] */
StructuredBuffer<float> ap_qg : register(t0);     /* [T][H][2D] q|gate */
StructuredBuffer<float> ap_kp : register(t1);
StructuredBuffer<float> ap_vp : register(t2);
StructuredBuffer<float> ap_iq_in : register(t3);
StructuredBuffer<float> ap_ik : register(t10);
StructuredBuffer<uint> ap_pos3 : register(t11);
ByteAddressBuffer ap_gq : register(t12);
ByteAddressBuffer ap_gk : register(t13);
ByteAddressBuffer ap_giq : register(t14);
ByteAddressBuffer ap_rope : register(t15);

groupshared float ap_red[32];
groupshared float ap_row[256];

/* grid (H + Hkv + Hi + 1, T), 32 threads.  params.n = D, rows = T, in_dim = H,
 * out_dim = Hkv, aux = Hi, ratio = Di, pos0, n_rot, freq_base, freq_scale,
 * eps, flags bit1 = rope table set.  The last slot copies the raw indexer key. */
[numthreads(32, 1, 1)]
void attn_prep(uint3 gid : SV_GroupID, uint lane : SV_GroupThreadID) {
    const uint slot = gid.x, t = gid.y;
    const uint H = params.in_dim, Hkv = params.out_dim, D = params.n;
    const uint Hi = params.aux, Di = params.ratio;
    const uint pos = params.pos0 + t;
    if (slot == H + Hkv + Hi) {
        for (uint i = lane; i < Di; i += 32u)
            ap_ikc[pos * Di + i] = ap_ik[t * Di + i];
        return;
    }
    const bool isq = slot < H;
    const bool isk = !isq && slot < H + Hkv;
    const uint h = isq ? slot : isk ? slot - H : slot - H - Hkv;
    const uint dim = (isq || isk) ? D : Di;
    const uint npt = dim >> 5u;

    float ss = 0.0f;
    if (isq) {
        const uint b = (t * H + h) * 2u * D;
        for (uint i = 0u; i < npt; i++) { const float v = ap_qg[b + lane * npt + i]; ss += v * v; }
    } else if (isk) {
        const uint b = (t * Hkv + h) * D;
        for (uint i = 0u; i < npt; i++) { const float v = ap_kp[b + lane * npt + i]; ss += v * v; }
    } else {
        const uint b = (t * Hi + h) * Di;
        for (uint i = 0u; i < npt; i++) { const float v = ap_iq_in[b + lane * npt + i]; ss += v * v; }
    }
    ap_red[lane] = ss;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        float s = 0.0f;
        for (uint j = 0u; j < 32u; j++) s += ap_red[j];
        ap_red[0] = s;
    }
    GroupMemoryBarrierWithGroupSync();
    const float inv = rsqrt(ap_red[0] / (float)dim + params.eps);
    GroupMemoryBarrierWithGroupSync();

    if (isq) {
        const uint b = (t * H + h) * 2u * D;
        for (uint i = lane; i < dim; i += 32u)
            ap_row[i] = ap_qg[b + i] * inv * asfloat(ap_gq.Load(i * 4u));
    } else if (isk) {
        const uint b = (t * Hkv + h) * D;
        for (uint i = lane; i < dim; i += 32u)
            ap_row[i] = ap_kp[b + i] * inv * asfloat(ap_gk.Load(i * 4u));
    } else {
        const uint b = (t * Hi + h) * Di;
        for (uint i = lane; i < dim; i += 32u)
            ap_row[i] = ap_iq_in[b + i] * inv * asfloat(ap_giq.Load(i * 4u));
    }
    GroupMemoryBarrierWithGroupSync();

    const uint half_rot = params.n_rot >> 1u;
    if (lane < half_rot) {
        const float th = (float)ap_pos3[pos * 4u + (lane % 3u)] *
                         qwen_freq(ap_rope, lane, params.freq_base, params.n_rot,
                                   (params.flags >> 1u) & 1u);
        const float c = cos(th) * params.freq_scale;
        const float s = sin(th) * params.freq_scale;
        const float a = ap_row[lane], b = ap_row[lane + half_rot];
        ap_row[lane] = a * c - b * s;
        ap_row[lane + half_rot] = a * s + b * c;
    }
    GroupMemoryBarrierWithGroupSync();

    if (isq) {
        const uint b = (t * H + h) * D;
        for (uint i = lane; i < dim; i += 32u) {
            ap_q[b + i] = ap_row[i];
            ap_gate[b + i] = ap_qg[(t * H + h) * 2u * D + D + i];
        }
    } else if (isk) {
        for (uint i = lane * 2u; i < dim; i += 64u) {
            const uint off = ((pos * Hkv + h) * D + i) * 2u;
            ap_kc.Store(off, ds4_f32_to_f16_bits_rne(ap_row[i]) |
                             (ds4_f32_to_f16_bits_rne(ap_row[i + 1u]) << 16u));
            ap_vc.Store(off, ds4_f32_to_f16_bits_rne(ap_vp[(t * Hkv + h) * D + i]) |
                             (ds4_f32_to_f16_bits_rne(ap_vp[(t * Hkv + h) * D + i + 1u]) << 16u));
        }
    } else {
        const uint b = (t * Hi + h) * Di;
        for (uint i = lane; i < dim; i += 32u) ap_iq[b + i] = ap_row[i];
    }
}

/* ---- block_key --------------------------------------------------------- */

RWByteAddressBuffer bk_out : register(u0);   /* f16 [b0+N][D] */
StructuredBuffer<float> bk_ik : register(t0);
StructuredBuffer<uint> bk_pos3 : register(t1);
ByteAddressBuffer bk_gik : register(t2);
ByteAddressBuffer bk_rope : register(t3);

groupshared float bk_red[32];
groupshared float bk_row[128];

/* grid (N), 32 threads.  params.n = D, rows = N, in_dim = b0, ratio,
 * n_rot, freq_base, freq_scale, eps, flags bit1 = rope table set. */
[numthreads(32, 1, 1)]
void block_key(uint3 gid : SV_GroupID, uint lane : SV_GroupThreadID) {
    const uint b = params.in_dim + gid.x;
    const uint ratio = params.ratio, D = params.n;
    const uint npt = D >> 5u;
    float ss = 0.0f, v[4];
    for (uint i = 0u; i < npt; i++) {
        float a = 0.0f;
        for (uint t = 0u; t < ratio; t++)
            a += bk_ik[(b * ratio + t) * D + lane * npt + i];
        v[i] = a / (float)ratio;
        ss += v[i] * v[i];
    }
    bk_red[lane] = ss;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        float s = 0.0f;
        for (uint j = 0u; j < 32u; j++) s += bk_red[j];
        bk_red[0] = s;
    }
    GroupMemoryBarrierWithGroupSync();
    const float inv = rsqrt(bk_red[0] / (float)D + params.eps);
    GroupMemoryBarrierWithGroupSync();
    for (uint i = 0u; i < npt; i++)
        bk_row[lane * npt + i] = v[i] * inv * asfloat(bk_gik.Load((lane * npt + i) * 4u));
    GroupMemoryBarrierWithGroupSync();

    const uint half_rot = params.n_rot >> 1u;
    if (lane < half_rot) {
        const uint pbase = b * ratio;
        const float th = (float)bk_pos3[pbase * 4u + (lane % 3u)] *
                         qwen_freq(bk_rope, lane, params.freq_base, params.n_rot,
                                   (params.flags >> 1u) & 1u);
        const float c = cos(th) * params.freq_scale;
        const float s = sin(th) * params.freq_scale;
        const float a = bk_row[lane], bb = bk_row[lane + half_rot];
        bk_row[lane] = a * c - bb * s;
        bk_row[lane + half_rot] = a * s + bb * c;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint i = lane * 2u; i < D; i += 64u)
        bk_out.Store((b * D + i) * 2u,
                     ds4_f32_to_f16_bits_rne(bk_row[i]) |
                         (ds4_f32_to_f16_bits_rne(bk_row[i + 1u]) << 16u));
}

/* ---- attn_decode ------------------------------------------------------- */

RWStructuredBuffer<float> at_out : register(u0);
RWStructuredBuffer<float> at_part : register(u1);
StructuredBuffer<float> at_q : register(t0);
StructuredBuffer<float> at_gate : register(t1);
ByteAddressBuffer at_kc : register(t2);
ByteAddressBuffer at_vc : register(t3);
StructuredBuffer<int> at_sel : register(t10);
StructuredBuffer<uint> at_cnt : register(t11);

groupshared float at_red[32];

/* grid (H, T, splits), 32 threads.  params.n = D, rows = T, in_dim = H,
 * out_dim = Hkv, pos0, ratio = sel_stride, blocks = splits, weight = scale,
 * flags bit0 = sparse. */
[numthreads(32, 1, 1)]
void attn_decode(uint3 gid : SV_GroupID, uint lane : SV_GroupThreadID) {
    const uint h = gid.x, t = gid.y, split = gid.z;
    const uint H = params.in_dim, Hkv = params.out_dim, D = params.n;
    const uint pos0 = params.pos0, stride = params.ratio, splits = params.blocks;
    const bool sparse = (params.flags & 1u) != 0u;
    const float scale = params.weight;
    const uint kh = h / (H / Hkv);
    const uint npt = D >> 5u;
    const uint n = sparse ? at_cnt[t] : pos0 + t + 1u;
    const uint keys = sparse ? stride : pos0 + params.rows;
    const uint per = (keys + splits - 1u) / splits;

    float qv[8], acc[8];
    for (uint i = 0u; i < npt; i++) {
        qv[i] = at_q[(t * H + h) * D + lane + 32u * i] * scale;
        acc[i] = 0.0f;
    }
    float m = -3e38f, denom = 0.0f;
    const uint j1 = min(n, (split + 1u) * per);
    for (uint j = split * per; j < j1; j++) {
        const uint p = sparse ? (uint)at_sel[t * stride + j] : j;
        if (p > pos0 + t) continue;
        float score = 0.0f;
        for (uint i = 0u; i < npt; i++)
            score += qv[i] * f16_at(at_kc, ((p * Hkv + kh) * D + lane + 32u * i) * 2u);
        at_red[lane] = score;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u) {
            float s = 0.0f;
            for (uint k = 0u; k < 32u; k++) s += at_red[k];
            at_red[0] = s;
        }
        GroupMemoryBarrierWithGroupSync();
        score = at_red[0];
        GroupMemoryBarrierWithGroupSync();
        const float nm = max(m, score), corr = exp(m - nm), w = exp(score - nm);
        denom = denom * corr + w;
        for (uint i = 0u; i < npt; i++)
            acc[i] = acc[i] * corr + w * f16_at(at_vc, ((p * Hkv + kh) * D + lane + 32u * i) * 2u);
        m = nm;
    }

    if (splits == 1u) {
        for (uint i = 0u; i < npt; i++) {
            const uint p = (t * H + h) * D + lane + 32u * i;
            at_out[p] = (denom > 0.0f ? acc[i] / denom : 0.0f) * qwen_sigmoid(at_gate[p]);
        }
    } else {
        const uint base = ((t * H + h) * splits + split) * (D + 2u);
        if (lane == 0u) { at_part[base] = m; at_part[base + 1u] = denom; }
        for (uint i = 0u; i < npt; i++)
            at_part[base + 2u + lane + 32u * i] = acc[i];
    }
}

/* ---- attn_merge -------------------------------------------------------- */

RWStructuredBuffer<float> am_out : register(u0);
StructuredBuffer<float> am_part : register(t0);
StructuredBuffer<float> am_gate : register(t1);

/* grid (H, T), D threads.  params.n = D, in_dim = H, blocks = splits. */
[numthreads(256, 1, 1)]
void attn_merge(uint3 gid : SV_GroupID, uint d : SV_GroupThreadID) {
    const uint h = gid.x, t = gid.y;
    const uint H = params.in_dim, D = params.n, splits = params.blocks;
    if (d >= D) return;
    const uint base = ((t * H + h) * splits) * (D + 2u);
    float m = -3e38f, denom = 0.0f, acc = 0.0f;
    for (uint s = 0u; s < splits; s++)
        m = max(m, am_part[base + s * (D + 2u)]);
    for (uint s = 0u; s < splits; s++) {
        const uint row = base + s * (D + 2u);
        const float w = am_part[row + 1u] > 0.0f ? exp(am_part[row] - m) : 0.0f;
        denom += am_part[row + 1u] * w;
        acc += am_part[row + 2u + d] * w;
    }
    const uint i = (t * H + h) * D + d;
    am_out[i] = (denom > 0.0f ? acc / denom : 0.0f) * qwen_sigmoid(am_gate[i]);
}
