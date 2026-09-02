/* vulkan/shaders/hc.hlsl -- DS4 hyper-connection (HC) kernels (Phase 5).
 *
 * HC reduces four residual streams before a sublayer and expands the sublayer
 * output back into four streams afterward.  Parity with the CUDA/ROCm kernels
 * in ds4_rocm_hc.cuh:
 *
 *   - hc_split_sinkhorn: 4x4 sinkhorn split of a 24-float mixer per row
 *     (pre[4] + post[4] + comb[16]).  Serial per row (one thread).
 *   - hc_weighted_sum: out[t][d] = sum_h residual[t][h][d] * w[t*stride + h].
 *   - hc_expand (generic): out_hc[t][dst][d] = block_out*post[dst] +
 *     sum_src comb[dst][src] * residual[t][src][d] (optional block_add).
 *   - hc_expand4 (fused, n_hc==4, split layout): post = split+4, comb = split+8.
 *   - hc_split_weighted_sum_fused: sinkhorn split + immediate weighted sum.
 *   - output_hc_weights: sigmoid(pre*scale + base) + eps.
 *   - repeat_hc(_rows): broadcast a token row over n_hc streams.
 *
 * Bindings (set 0): t0..t2 StructuredBuffer, t3 model (ByteAddressBuffer),
 * u0..u2 RWStructuredBuffer.  Scalar parameters in DS4Params.
 */

#include "common.hlsl"

StructuredBuffer<float> a_buf : register(t0);
StructuredBuffer<float> b_buf : register(t1);
StructuredBuffer<float> c_buf : register(t2);
ByteAddressBuffer w_buf : register(t3);
RWStructuredBuffer<float> out_buf : register(u0);
RWStructuredBuffer<float> out2_buf : register(u1);
RWStructuredBuffer<float> out3_buf : register(u2);

[[vk::push_constant]] DS4Params params;

/* Serial 4x4 sinkhorn split (hc4_split_one).  dst = 24 floats:
 * pre (sigmoid + eps), post (2*sigmoid), comb (16, softmax rows then cols).
 * dst is `inout` (HLSL passes arrays by value; without inout the writes are
 * dead code and dxc's optimizer drops the whole computation). */
void hc4_split_one(inout float dst[24], float mix[24], uint scale_off,
                   uint base_off) {
    float pre_scale = asfloat(w_buf.Load(scale_off + 0u));
    float post_scale = asfloat(w_buf.Load(scale_off + 4u));
    float comb_scale = asfloat(w_buf.Load(scale_off + 8u));
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + asfloat(w_buf.Load(base_off + i * 4u));
        dst[i] = 1.0f / (1.0f + exp(-z)) + params.eps;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale +
                  asfloat(w_buf.Load(base_off + (4 + i) * 4u));
        dst[4 + i] = 2.0f / (1.0f + exp(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -1.0e30f;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale +
                      asfloat(w_buf.Load(base_off + (8 + r * 4 + col) * 4u));
            c[r * 4 + col] = v;
            m = max(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = exp(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) {
            c[r * 4 + col] = c[r * 4 + col] / s + params.eps;
        }
    }
    for (int col = 0; col < 4; col++) {
        float s = params.eps;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    uint iters = params.aux;
    for (uint iter = 1u; iter < iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = params.eps;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = params.eps;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) dst[8 + i] = c[i];
}

/* One thread per row: split a 24-float mixer into out[row][24].
 * params: n=n_hc (must be 4), rows=n_rows, aux=sinkhorn_iters, eps=eps,
 * blocks=scale offset (bytes, relative to the bound model range),
 * ratio=base offset (bytes, relative). */
[numthreads(256, 1, 1)]
void hc_split_sinkhorn(uint3 gid : SV_DispatchThreadID) {
    if (gid.x >= params.rows) return;
    float mix[24];
    uint base = gid.x * 24u;
    for (int i = 0; i < 24; i++) mix[i] = a_buf[base + i];
    float o[24];
    hc4_split_one(o, mix, params.blocks, params.ratio);
    for (int i = 0; i < 24; i++) out_buf[base + i] = o[i];
}

/* Flat weighted sum over n_hc streams.  params: n=n_embd, rows=n_tokens,
 * index=n_hc, ratio=weight_stride_f32 (n_hc for tensor weights, the full
 * split stride for hc_weighted_sum_split). */
[numthreads(256, 1, 1)]
void hc_weighted_sum(uint3 gid : SV_DispatchThreadID) {
    uint n = params.n * params.rows;
    if (gid.x >= n) return;
    uint d = gid.x % params.n;
    uint t = gid.x / params.n;
    uint n_hc = params.index;
    uint stride = params.ratio;
    float acc = 0.0f;
    for (uint h = 0u; h < n_hc; h++) {
        acc += a_buf[t * n_hc * params.n + h * params.n + d] *
               b_buf[t * stride + h];
    }
    out_buf[t * params.n + d] = acc;
}

/* Generic HC expand (post/comb as separate tensors).  params: n=n_embd,
 * rows=n_tokens, index=n_hc, aux=post_stride, blocks=comb_stride,
 * flags&1=has_add. */
[numthreads(256, 1, 1)]
void hc_expand(uint3 gid : SV_DispatchThreadID) {
    uint n_elem = params.rows * params.index * params.n;
    if (gid.x >= n_elem) return;
    uint d = gid.x % params.n;
    uint tmp = gid.x / params.n;
    uint dst_hc = tmp % params.index;
    uint t = tmp / params.index;
    uint n_hc = params.index;
    float block_v = a_buf[t * params.n + d];
    if ((params.flags & 1u) != 0u) {
        block_v += b_buf[t * params.n + d];
    }
    float acc = block_v * out2_buf[t * params.aux + dst_hc];
    for (uint src_hc = 0u; src_hc < n_hc; src_hc++) {
        float comb_v = out3_buf[t * params.blocks + dst_hc + src_hc * n_hc];
        float res_v = c_buf[t * n_hc * params.n + src_hc * params.n + d];
        acc += comb_v * res_v;
    }
    out_buf[t * n_hc * params.n + dst_hc * params.n + d] = acc;
}

/* Fused HC expand for the split layout (post = split+4, comb = split+8),
 * n_hc == 4.  params: n=n_embd, rows=n_tokens, index=n_hc, flags: bit0
 * has_add, bit1 has_add2.  Bindings: t0 block_out, t1 block_add, t2
 * block_add2, u1 residual_hc (ro), u2 split (ro). */
[numthreads(256, 1, 1)]
void hc_expand4(uint3 gid : SV_DispatchThreadID) {
    uint n_elem = params.rows * params.index * params.n;
    if (gid.x >= n_elem) return;
    uint d = gid.x % params.n;
    uint tmp = gid.x / params.n;
    uint dst = tmp % params.index;
    uint t = tmp / params.index;
    uint n_hc = params.index;
    float bv = a_buf[t * params.n + d];
    if ((params.flags & 1u) != 0u) bv += b_buf[t * params.n + d];
    if ((params.flags & 2u) != 0u) bv += c_buf[t * params.n + d];
    uint hc_base = t * n_hc * params.n + d;
    uint sp = t * 24u;
    float acc = bv * out3_buf[sp + 4u + dst];
    for (uint src = 0u; src < 4u; src++) {
        acc += out3_buf[sp + 8u + src * 4u + dst] *
               out2_buf[hc_base + src * params.n];
    }
    out_buf[hc_base + dst * params.n] = acc;
}

/* Fused sinkhorn split + immediate weighted sum (n_hc==4).  One block per
 * row.  params: n=n_embd, rows=n_rows, index=n_hc, aux=sinkhorn_iters,
 * eps=eps, blocks=scale offset, ratio=base offset.  split[row] receives
 * the 24-float split; out[row] the weighted sum.  The split is published
 * through out2_buf (global) so every lane reads it after the barrier. */
[numthreads(256, 1, 1)]
void hc_split_weighted_sum_fused(uint3 gid : SV_GroupID,
                                 uint tid : SV_GroupThreadID) {
    uint t = gid.x;
    if (t >= params.rows || params.index != 4u) return;
    uint base = t * 24u;
    if (tid == 0u) {
        float mix[24];
        float sp[24];
        for (int i = 0; i < 24; i++) mix[i] = a_buf[base + i];
        hc4_split_one(sp, mix, params.blocks, params.ratio);
        for (int i = 0; i < 24; i++) out2_buf[base + i] = sp[i];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint col = tid; col < params.n; col += 256u) {
        float acc = 0.0f;
        for (uint h = 0u; h < 4u; h++) {
            acc += b_buf[t * 4u * params.n + h * params.n + col] *
                   out2_buf[base + h];
        }
        out_buf[t * params.n + col] = acc;
    }
}

/* Output HC weights: out[gid] = sigmoid(pre[gid]*scale + base[h]) + eps.
 * params: n=n_hc, rows=n_tokens, eps=eps, blocks=scale offset (relative),
 * aux=base offset (relative). */
[numthreads(256, 1, 1)]
void output_hc_weights(uint3 gid : SV_DispatchThreadID) {
    uint n = params.rows * params.n;
    if (gid.x >= n) return;
    uint h = gid.x % params.n;
    float z = a_buf[gid.x] * asfloat(w_buf.Load(params.blocks)) +
              asfloat(w_buf.Load(params.aux + h * 4u));
    out_buf[gid.x] = 1.0f / (1.0f + exp(-z)) + params.eps;
}

/* Broadcast a token row over n_hc streams.  params: n=n_embd, rows=n_tokens
 * (1 for the single-row entry), index=n_hc. */
[numthreads(256, 1, 1)]
void repeat_hc(uint3 gid : SV_DispatchThreadID) {
    uint n = params.rows * params.index * params.n;
    if (gid.x >= n) return;
    uint tok = gid.x / (params.index * params.n);
    uint d = gid.x % params.n;
    out_buf[gid.x] = a_buf[tok * params.n + d];
}
