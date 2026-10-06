/* vulkan/shaders/gdn.hlsl -- Qwen3.8-Flash-Next gated delta-net (GDN).
 *
 * Single-session kernels matching the CUDA reference (ds4_qwen4_cuda.cuh):
 *   conv_stream   depthwise causal conv K=2..4 (+ optional SiLU, snapshot)
 *   gdn_prep      L2-normalise q/k, decay a=exp(A*softplus(a+bias)), b=sigmoid
 *   gdn_scan      delta rule, sequential over tokens, one warp owns ROWS rows
 *   gdn_out       per-head RMSNorm * ssm_norm * sigmoid(z)
 *
 * The scan uses a groupshared tree reduction across the 32 lanes of each warp
 * (no subgroup ops): deterministic run-to-run and independent of the chunk
 * length, so a split prefill is byte-exact against a single batch call.
 *
 * Binding convention (see common.hlsl): SRVs t0..t3 = bindings 0..3, UAVs
 * u0..u4 = bindings 4..8.  Weight scalars live in the model buffer (t3). */

#include "common.hlsl"

[[vk::push_constant]] DS4Params params;

/* Stable sigmoid/silu, parity with the double-precision test reference
 * (sigmoid_d/silu_d in tests/test_qwen4_kernels.c). */
float gdn_sigmoid(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + exp(-x)) : exp(x) / (1.0f + exp(x));
}
float gdn_silu(float x) { return x * gdn_sigmoid(x); }

/* ---- conv_stream ------------------------------------------------------- */

RWStructuredBuffer<float> conv_x : register(u0);     /* qkv, updated in place */
RWStructuredBuffer<float> conv_hist : register(u1);  /* (K-1) window rows */
RWStructuredBuffer<float> conv_snap : register(u2);
RWStructuredBuffer<float> conv_snap2 : register(u3);
ByteAddressBuffer conv_w : register(t3);

/* params.n = C channels, params.rows = T, params.in_dim = K, flags bit0 =
 * activate, index = snap_t, aux = snap2_t.  Snapshots are only written when
 * the matching token index is reached; the host binds a dummy for absent
 * snapshots and uses 0xffffffff for the token index. */
[numthreads(256, 1, 1)]
void conv_stream(uint3 gid : SV_DispatchThreadID) {
    const uint c = gid.x;
    const uint C = params.n, T = params.rows, K = params.in_dim;
    if (c >= C) return;
    float win[3], taps[4];
    for (uint i = 0u; i < K - 1u; i++) win[i] = conv_hist[i * C + c];
    for (uint i = 0u; i < K; i++) taps[i] = asfloat(conv_w.Load((c * K + i) * 4u));
    const bool activate = (params.flags & 1u) != 0u;
    const uint snap_t = params.index, snap2_t = params.aux;
    for (uint t = 0u; t < T; t++) {
        const uint pos = t * C + c;
        const float raw = conv_x[pos];
        float v = taps[K - 1u] * raw;
        for (uint i = 0u; i < K - 1u; i++) v += taps[i] * win[i];
        for (uint i = 0u; i + 2u < K; i++) win[i] = win[i + 1u];
        win[K - 2u] = raw;
        conv_x[pos] = activate ? gdn_silu(v) : v;
        if (t == snap_t)
            for (uint i = 0u; i < K - 1u; i++) conv_snap[i * C + c] = win[i];
        if (t == snap2_t)
            for (uint i = 0u; i < K - 1u; i++) conv_snap2[i * C + c] = win[i];
    }
    for (uint i = 0u; i < K - 1u; i++) conv_hist[i * C + c] = win[i];
}

/* ---- gdn_prep ---------------------------------------------------------- */

RWStructuredBuffer<float> prep_qkv : register(u0);   /* q/k normalised in place */
RWStructuredBuffer<float> prep_a : register(u1);
RWStructuredBuffer<float> prep_b : register(u2);
ByteAddressBuffer prep_A : register(t3);     /* ssm_a [Hv] */
ByteAddressBuffer prep_bias : register(t2);  /* dt_bias [Hv] */

groupshared float prep_red[32];

/* grid (Hk, T), 32 threads.  params.n = D, rows = T, in_dim = Hk,
 * out_dim = Hv, eps = 1e-6. */
[numthreads(32, 1, 1)]
void gdn_prep(uint3 gid : SV_GroupID, uint lane : SV_GroupThreadID) {
    const uint h = gid.x, t = gid.y;
    const uint Hk = params.in_dim, Hv = params.out_dim, D = params.n;
    const uint npt = D >> 5u;
    const uint C = (2u * Hk + Hv) * D;
    const uint qbase = t * C + h * D + lane * npt;
    const uint kbase = qbase + Hk * D;

    float qs = 0.0f, ks = 0.0f;
    for (uint i = 0u; i < npt; i++) {
        const float q = prep_qkv[qbase + i], k = prep_qkv[kbase + i];
        qs += q * q;
        ks += k * k;
    }
    prep_red[lane] = qs;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) prep_red[lane] += prep_red[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    const float qsum = prep_red[0];
    GroupMemoryBarrierWithGroupSync();
    prep_red[lane] = ks;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) prep_red[lane] += prep_red[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    const float ksum = prep_red[0];
    GroupMemoryBarrierWithGroupSync();

    const float qscale = rsqrt(qsum + 1e-6f) * rsqrt((float)D);
    const float kscale = rsqrt(ksum + 1e-6f);
    for (uint i = 0u; i < npt; i++) {
        prep_qkv[qbase + i] *= qscale;
        prep_qkv[kbase + i] *= kscale;
    }

    if (h == 0u) {
        for (uint j = lane; j < Hv; j += 32u) {
            const uint p = t * Hv + j;
            const float A = asfloat(prep_A.Load(j * 4u));
            const float bias = asfloat(prep_bias.Load(j * 4u));
            prep_a[p] = exp(A * softplus(prep_a[p] + bias));
            prep_b[p] = gdn_sigmoid(prep_b[p]);
        }
    }
}

/* ---- gdn_scan ---------------------------------------------------------- */

RWStructuredBuffer<float> scan_out : register(u0);
RWStructuredBuffer<float> scan_state : register(u1);  /* [Hv][D][D] as [dv][dk] */
StructuredBuffer<float> scan_qkv : register(t0);
StructuredBuffer<float> scan_a : register(t1);
StructuredBuffer<float> scan_b : register(t2);
RWStructuredBuffer<float> scan_snap : register(u2);
RWStructuredBuffer<float> scan_snap2 : register(u3);

groupshared float scan_red[4][4][32];  /* [warp][row][lane] */

/* grid (D/(4*ROWS), Hv), 128 threads (4 warps).  params.n = D, rows = T,
 * in_dim = Hk, out_dim = Hv, index = snap_t, aux = snap2_t, rsvd2 = ROWS. */
[numthreads(128, 1, 1)]
void gdn_scan(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    const uint warp = tid.x >> 5u, lane = tid.x & 31u;
    const uint ROWS = max(params.rsvd2, 1u);
    const uint D = params.n, Hk = params.in_dim, Hv = params.out_dim;
    const uint npt = D >> 5u;
    const uint C = (2u * Hk + Hv) * D;
    const uint h = gid.y;
    const uint dv = (gid.x * 4u + warp) * ROWS;
    const uint kh = h % Hk;
    const uint k0 = lane * npt;
    const uint idx = (h * D + dv) * D + k0;
    const uint snap_t = params.index, snap2_t = params.aux;

    float s[4][4];
    [unroll] for (uint r = 0u; r < 4u; r++)
        if (r < ROWS)
            [unroll] for (uint i = 0u; i < 4u; i++)
                if (i < npt) s[r][i] = scan_state[idx + r * D + i];

    for (uint t = 0u; t < params.rows; t++) {
        const float decay = scan_a[t * Hv + h], beta = scan_b[t * Hv + h];
        const uint qbase = t * C + kh * D + k0;
        const uint kbase = qbase + Hk * D;
        float vv[4], part[4];

        [unroll] for (uint r = 0u; r < 4u; r++) {
            if (r < ROWS) {
                const float v = scan_qkv[t * C + 2u * Hk * D + h * D + dv + r];
                float u = 0.0f;
                [unroll] for (uint i = 0u; i < 4u; i++) {
                    if (i < npt) {
                        s[r][i] *= decay;
                        u += s[r][i] * scan_qkv[kbase + i];
                    }
                }
                vv[r] = v;
                part[r] = u;
            }
        }

        [unroll] for (uint r = 0u; r < 4u; r++)
            if (r < ROWS) scan_red[warp][r][lane] = part[r];
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 16u; stride > 0u; stride >>= 1u) {
            if (lane < stride)
                [unroll] for (uint r = 0u; r < 4u; r++)
                    if (r < ROWS)
                        scan_red[warp][r][lane] += scan_red[warp][r][lane + stride];
            GroupMemoryBarrierWithGroupSync();
        }
        float delta[4];
        [unroll] for (uint r = 0u; r < 4u; r++)
            if (r < ROWS) delta[r] = (vv[r] - scan_red[warp][r][0]) * beta;
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint r = 0u; r < 4u; r++) {
            if (r < ROWS) {
                float o = 0.0f;
                [unroll] for (uint i = 0u; i < 4u; i++) {
                    if (i < npt) {
                        s[r][i] += scan_qkv[kbase + i] * delta[r];
                        o += s[r][i] * scan_qkv[qbase + i];
                    }
                }
                part[r] = o;
            }
        }

        [unroll] for (uint r = 0u; r < 4u; r++)
            if (r < ROWS) scan_red[warp][r][lane] = part[r];
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 16u; stride > 0u; stride >>= 1u) {
            if (lane < stride)
                [unroll] for (uint r = 0u; r < 4u; r++)
                    if (r < ROWS)
                        scan_red[warp][r][lane] += scan_red[warp][r][lane + stride];
            GroupMemoryBarrierWithGroupSync();
        }

        if (lane == 0u)
            [unroll] for (uint r = 0u; r < 4u; r++)
                if (r < ROWS) scan_out[(t * Hv + h) * D + dv + r] = scan_red[warp][r][0];
        if (t == snap_t)
            [unroll] for (uint r = 0u; r < 4u; r++)
                if (r < ROWS)
                    [unroll] for (uint i = 0u; i < 4u; i++)
                        if (i < npt) scan_snap[idx + r * D + i] = s[r][i];
        if (t == snap2_t)
            [unroll] for (uint r = 0u; r < 4u; r++)
                if (r < ROWS)
                    [unroll] for (uint i = 0u; i < 4u; i++)
                        if (i < npt) scan_snap2[idx + r * D + i] = s[r][i];
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll] for (uint r = 0u; r < 4u; r++)
        if (r < ROWS)
            [unroll] for (uint i = 0u; i < 4u; i++)
                if (i < npt) scan_state[idx + r * D + i] = s[r][i];
}

/* ---- gdn_out ----------------------------------------------------------- */

RWStructuredBuffer<float> out_o : register(u0);
StructuredBuffer<float> out_z : register(t1);
ByteAddressBuffer out_w : register(t3);  /* ssm_norm [D] */

groupshared float out_red[32];

/* grid (H, T), 32 threads.  params.n = D, rows = T, in_dim = H, eps. */
[numthreads(32, 1, 1)]
void gdn_out(uint3 gid : SV_GroupID, uint lane : SV_GroupThreadID) {
    const uint h = gid.x, t = gid.y;
    const uint D = params.n, H = params.in_dim;
    const uint npt = D >> 5u, k0 = lane * npt;
    const uint idx = (t * H + h) * D + k0;

    float ss = 0.0f;
    for (uint i = 0u; i < npt; i++) ss += out_o[idx + i] * out_o[idx + i];
    out_red[lane] = ss;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) out_red[lane] += out_red[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = rsqrt(out_red[0] / (float)D + params.eps);
    GroupMemoryBarrierWithGroupSync();

    for (uint i = 0u; i < npt; i++) {
        const uint p = idx + i;
        out_o[p] = out_o[p] * r * asfloat(out_w.Load((k0 + i) * 4u)) * gdn_sigmoid(out_z[p]);
    }
}
