/* vulkan/shaders/rmsnorm.hlsl -- RMS normalization kernels.
 *
 * One block of 256 threads per row.  The `rows` parameter selects how many
 * rows the dispatch covers (grid.y); the plain vs plain_rows entry points are
 * the same shader launched with different grids.  Weighted variants read the
 * per-channel weights from the model buffer (binding 3), which the host binds
 * at the weight tensor's byte offset.
 */

#include "common.hlsl"

StructuredBuffer<float> x_buf : register(t0);
ByteAddressBuffer w_buf : register(t3);
RWStructuredBuffer<float> out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

groupshared float rms_red[256];

/* out[row][i] = x[row][i] / sqrt(mean(x^2) + eps) */
[numthreads(256, 1, 1)]
void rms_norm_plain(uint3 gid : SV_DispatchThreadID) {
    uint row = gid.y;
    uint n = params.n;
    uint base = row * n;
    float sum = 0.0f;
    for (uint i = gid.x; i < n; i += 256u) {
        float v = x_buf[base + i];
        sum += v * v;
    }
    rms_red[gid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (gid.x < stride) {
            rms_red[gid.x] += rms_red[gid.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float scale = rsqrt(rms_red[0] / (float)n + params.eps);
    for (uint i2 = gid.x; i2 < n; i2 += 256u) {
        out_buf[base + i2] = x_buf[base + i2] * scale;
    }
}

/* out[row][i] = x[row][i] * w[i] / sqrt(mean(x^2) + eps) */
[numthreads(256, 1, 1)]
void rms_norm_weight(uint3 gid : SV_DispatchThreadID) {
    uint row = gid.y;
    uint n = params.n;
    uint base = row * n;
    float sum = 0.0f;
    for (uint i = gid.x; i < n; i += 256u) {
        float v = x_buf[base + i];
        sum += v * v;
    }
    rms_red[gid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (gid.x < stride) {
            rms_red[gid.x] += rms_red[gid.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float scale = rsqrt(rms_red[0] / (float)n + params.eps);
    for (uint i2 = gid.x; i2 < n; i2 += 256u) {
        float wv = asfloat(w_buf.Load(i2 * 4u));
        out_buf[base + i2] = x_buf[base + i2] * scale * wv;
    }
}

/* In-place per-head RMS norm: x[rows][head_dim] with one block per
 * (token*head) row.  params: n=head_dim, rows=n_tok, index=n_head. */
[numthreads(256, 1, 1)]
void head_rms_norm(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.rows * params.index) return;
    uint n = params.n;
    uint base = row * n;
    float sum = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        float v = x_buf[base + i];
        sum += v * v;
    }
    rms_red[tid] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            rms_red[tid] += rms_red[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float scale = rsqrt(rms_red[0] / (float)n + params.eps);
    for (uint i2 = tid; i2 < n; i2 += 256u) {
        out_buf[base + i2] = x_buf[base + i2] * scale;
    }
}

/* In-place per-head RMS norm + rope on the tail (n_rot elements at the end
 * of each head).  One block per (token*head) row; rope position is
 * pos0 + token (all heads of a token share it).  Parity with the CUDA
 * head_rms_norm_rope_tail_kernel. */
[numthreads(256, 1, 1)]
void head_rms_norm_rope_tail(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row = gid.x;
    uint n_tok = params.rows;
    uint n_head = params.index;
    if (row >= n_tok * n_head) return;
    uint t = row / n_head;
    uint n = params.n;
    uint n_rot = params.n_rot;
    uint base = row * n;
    float sum = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        float v = x_buf[base + i];
        sum += v * v;
    }
    rms_red[tid] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            rms_red[tid] += rms_red[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float scale = rsqrt(rms_red[0] / (float)n + params.eps);
    uint n_nope = n - n_rot;
    for (uint i = tid; i < n_nope; i += 256u) {
        out_buf[base + i] = x_buf[base + i] * scale;
    }
    float corr0 = 0.0f, corr1 = 0.0f;
    if (params.ext_factor != 0.0f) {
        float denom = 2.0f * log(params.freq_base);
        corr0 = floor((float)n_rot *
                      log((float)params.n_ctx_orig / (params.beta_fast * 2.0f * 3.14159265f)) /
                      denom);
        corr1 = ceil((float)n_rot *
                     log((float)params.n_ctx_orig / (params.beta_slow * 2.0f * 3.14159265f)) /
                     denom);
        corr0 = max(0.0f, corr0);
        corr1 = min((float)(n_rot - 1), corr1);
    }
    for (uint pair = tid; pair < n_rot / 2u; pair += 256u) {
        uint i = pair * 2u;
        float theta_extrap = (float)(params.pos0 + t) *
                             pow(params.freq_base, -((float)i) / (float)n_rot);
        float theta_interp = params.freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = params.attn_factor;
        if (params.ext_factor != 0.0f) {
            float ramp = rope_yarn_ramp(corr0, corr1, (int)i) * params.ext_factor;
            theta = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
            mscale *= 1.0f + 0.1f * log(1.0f / params.freq_scale);
        }
        float c = cos(theta) * mscale;
        float s = sin(theta) * mscale;
        if (params.inverse != 0) s = -s;
        uint tail = base + n_nope + i;
        float x0 = x_buf[tail] * scale;
        float x1 = x_buf[tail + 1u] * scale;
        out_buf[tail] = x0 * c - x1 * s;
        out_buf[tail + 1u] = x0 * s + x1 * c;
    }
}
