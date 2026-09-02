/* vulkan/shaders/kv.hlsl -- DS4 FP8 KV quantize and raw-cache store kernels.
 *
 * FP8 KV semantics (parity with ds4_cuda.cu fp8_kv_quantize_row /
 * store_raw_kv_batch_kernel / fp8_kv_quantize_store_rows_kernel):
 *
 *   - Only the nope part (head_dim - n_rot) is quantized; the rope tail stays
 *     f32.  Blocks of 64 elements: scale = exp2(ceil(log2(max(|v|,1e-4)/448))),
 *     each value rounded to E4M3FN (round-half-to-even on the representable
 *     lattice) and dequantized back to f32 in place.
 *   - Raw stores write __half2float(__float2half(x)) (f16 round trip).
 *
 * Bindings: x_buf (kv, RW) on u0, raw_buf (raw cache, RW) on u1.
 */

#include "common.hlsl"

RWStructuredBuffer<float> x_buf : register(u0);
RWStructuredBuffer<float> raw_buf : register(u1);

[[vk::push_constant]] DS4Params params;

groupshared float kv_red[64];

/* E4M3FN representable values: exp = (i>>3)&15, mant = i&7; exp==0 subnormal
 * -> mant * 2^-9, else (1 + mant/8) * 2^(exp-7).  i in [0,126]. */
float ds4_e4m3fn_value(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2((float)exp - 7.0f);
}

float ds4_e4m3fn_dequant(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = min(abs(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (ds4_e4m3fn_value(mid) <= ax) lo = mid; else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = abs(ax - ds4_e4m3fn_value(best));
        float nd = abs(ax - ds4_e4m3fn_value(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) {
            best++;
        }
    }
    return sign * ds4_e4m3fn_value(best);
}

/* In-place FP8 KV quantize of the nope part.  One block of 64 threads per
 * row (params.rows = n_tok).  params.n = head_dim, params.n_rot = n_rot. */
[numthreads(64, 1, 1)]
void fp8_kv_quantize(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint row = gid.x;
    uint head_dim = params.n;
    uint n_nope = head_dim - params.n_rot;
    uint base = row * head_dim;
    for (uint off = 0u; off < n_nope; off += 64u) {
        uint idx = off + tid;
        float v = idx < n_nope ? x_buf[base + idx] : 0.0f;
        kv_red[tid] = idx < n_nope ? abs(v) : 0.0f;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 32u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                kv_red[tid] = max(kv_red[tid], kv_red[tid + stride]);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        float scale = exp2(ceil(log2(max(kv_red[0], 1.0e-4f) / 448.0f)));
        if (idx < n_nope) {
            float q = ds4_e4m3fn_dequant(min(448.0f, max(-448.0f, v / scale))) * scale;
            x_buf[base + idx] = q;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

/* Store raw-cache rows with an f16 round trip:
 * raw[(pos0 + t) % raw_cap][d] = half2float(float2half(kv[t][d])).
 * Flat 1-D dispatch over n_tokens * head_dim.  params.rows = n_tokens,
 * params.pos0 = pos0, params.aux = raw_cap. */
[numthreads(256, 1, 1)]
void store_raw_kv(uint3 gid : SV_DispatchThreadID) {
    uint head_dim = params.n;
    uint n = params.rows * head_dim;
    if (gid.x >= n) return;
    uint d = gid.x % head_dim;
    uint t = gid.x / head_dim;
    uint row = (params.pos0 + t) % params.aux;
    raw_buf[row * head_dim + d] = f16tof32(ds4_f32_to_f16_bits_rne(x_buf[t * head_dim + d]));
}

/* Fused FP8 KV quantize + raw store: quantize the nope part of the row in
 * place, then write the whole row (head_dim) to the raw cache with an f16
 * round trip.  One block of 64 threads per row.  params.index = raw_start
 * (already modulo raw_cap by the host), params.aux = raw_cap. */
[numthreads(64, 1, 1)]
void kv_fp8_store_raw(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint head_dim = params.n;
    uint n_nope = head_dim - params.n_rot;
    uint base = gid.x * head_dim;
    for (uint off = 0u; off < n_nope; off += 64u) {
        uint idx = off + tid;
        float v = idx < n_nope ? x_buf[base + idx] : 0.0f;
        kv_red[tid] = idx < n_nope ? abs(v) : 0.0f;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 32u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                kv_red[tid] = max(kv_red[tid], kv_red[tid + stride]);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        float scale = exp2(ceil(log2(max(kv_red[0], 1.0e-4f) / 448.0f)));
        if (idx < n_nope) {
            float q = ds4_e4m3fn_dequant(min(448.0f, max(-448.0f, v / scale))) * scale;
            x_buf[base + idx] = q;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    uint raw_row = params.index;
    for (uint d = tid; d < head_dim; d += 64u) {
        raw_buf[raw_row * head_dim + d] = f16tof32(ds4_f32_to_f16_bits_rne(x_buf[base + d]));
    }
}
