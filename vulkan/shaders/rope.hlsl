/* vulkan/shaders/rope.hlsl -- DS4 tail-only RoPE (matches CUDA
 * rope_tail_kernel, including the YaRN/linear-scaling corrections).
 *
 * One thread rotates one (pair, head, token) of the tail.  Dispatch covers
 * n_tok * n_head * (n_rot / 2) threads.
 */

#include "common.hlsl"

RWStructuredBuffer<float> x_buf : register(u0);

[[vk::push_constant]] DS4Params params;

[numthreads(256, 1, 1)]
void rope_tail(uint3 gid : SV_DispatchThreadID) {
    uint n_tok = params.rows;
    uint n_head = params.index;
    uint head_dim = params.n;
    uint n_rot = params.n_rot;
    uint pairs = n_rot / 2u;
    if (pairs == 0u) return;
    uint gid_flat = gid.x;
    uint total = n_tok * n_head * pairs;
    if (gid_flat >= total) return;

    uint pair = gid_flat % pairs;
    uint tmp = gid_flat / pairs;
    uint h = tmp % n_head;
    uint t = tmp / n_head;
    uint n_nope = head_dim - n_rot;
    uint i = pair * 2u;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (params.ext_factor != 0.0f) {
        float denom = 2.0f * log(params.freq_base);
        corr0 = floor((float)n_rot *
                      log((float)params.n_ctx_orig /
                          (params.beta_fast * 2.0f * DS4_PI)) / denom);
        corr1 = ceil((float)n_rot *
                     log((float)params.n_ctx_orig /
                         (params.beta_slow * 2.0f * DS4_PI)) / denom);
        corr0 = max(0.0f, corr0);
        corr1 = min((float)(n_rot - 1), corr1);
    }

    float theta_extrap =
        (float)(params.pos0 + t * (params.aux != 0u ? params.aux : 1u)) *
        pow(params.freq_base, -((float)i) / (float)n_rot);
    float theta_interp = params.freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = params.attn_factor;
    if (params.ext_factor != 0.0f) {
        float ramp_mix =
            rope_yarn_ramp(corr0, corr1, (int)i) * params.ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / params.freq_scale);
    }
    float c = cos(theta) * mscale;
    float s = sin(theta) * mscale;
    if (params.inverse != 0) s = -s;

    uint base = (t * n_head + h) * head_dim + n_nope;
    float x0 = x_buf[base + i];
    float x1 = x_buf[base + i + 1u];
    x_buf[base + i] = x0 * c - x1 * s;
    x_buf[base + i + 1u] = x0 * s + x1 * c;
}
