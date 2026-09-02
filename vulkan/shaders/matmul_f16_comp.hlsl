/* vulkan/shaders/matmul_f16_comp.hlsl -- F16 paired projection fused with
 * the recurrent compressor-state store (decode).
 *
 * Mirrors the Metal kernel_mul_mv_f16_f32_pair_compressor_store_4: computes
 *   out_kv[row]    = sum_k w_kv_f16[row][k] * x[k]
 *   out_score[row] = sum_k w_score_f16[row][k] * x[k]
 * (single token), then stores into the rolling compressor state at the
 * pos/ratio row:
 *   pos_mod  = pos % ratio
 *   dst_row  = ratio == 4 ? ratio + pos_mod : pos_mod
 *   state_kv[dst_row*width + row]    = out_kv[row]
 *   state_score[dst_row*width + row] = out_score[row] + ape[pos_mod*width+row]
 * (ape read as f32 when index==0, f16 when index==1).
 *
 * One 256-thread block per output row (the same reduction tree as
 * matmul_f16); the block's reduced values are stored straight into both the
 * projection outputs and the state, so the fused result is bit-identical to
 * the separate matmul_f16_pair + compressor_store sequence.
 *
 * Bindings: t0 x, t1 w_kv, t2 w_score, t3 model ape, u0 out_kv, u1 out_score,
 * u2 state_kv, u3 state_score.  Params: in_dim, out_dim=width, ratio=ratio,
 * pos0=pos, index=ape_type.
 */

#include "common.hlsl"

StructuredBuffer<float> x_buf : register(t0);
ByteAddressBuffer w_kv_buf : register(t1);
ByteAddressBuffer w_sc_buf : register(t2);
ByteAddressBuffer ape_buf : register(t3);
RWStructuredBuffer<float> out_kv_buf : register(u0);
RWStructuredBuffer<float> out_sc_buf : register(u1);
RWStructuredBuffer<float> st_kv_buf : register(u2);
RWStructuredBuffer<float> st_sc_buf : register(u3);

[[vk::push_constant]] DS4Params params;

groupshared float mf16c_red_kv[256];
groupshared float mf16c_red_sc[256];

[numthreads(256, 1, 1)]
void matmul_f16_pair_compressor_store(uint3 gid : SV_GroupID,
                                      uint tid : SV_GroupThreadID) {
    uint row = gid.x;
    uint width = params.out_dim;
    uint in_dim = params.in_dim;
    if (row >= width) return;

    float acc_kv = 0.0f;
    float acc_sc = 0.0f;
    for (uint k = tid; k < in_dim; k += 256u) {
        float xv = x_buf[k];
        acc_kv += f16_at(w_kv_buf, (row * in_dim + k) * 2u) * xv;
        acc_sc += f16_at(w_sc_buf, (row * in_dim + k) * 2u) * xv;
    }

    mf16c_red_kv[tid] = acc_kv;
    mf16c_red_sc[tid] = acc_sc;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            mf16c_red_kv[tid] += mf16c_red_kv[tid + stride];
            mf16c_red_sc[tid] += mf16c_red_sc[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        out_kv_buf[row] = mf16c_red_kv[0];
        out_sc_buf[row] = mf16c_red_sc[0];
        uint pos_mod = params.pos0 % params.ratio;
        uint dst_row = params.ratio == 4u ? params.ratio + pos_mod : pos_mod;
        uint dst = dst_row * width + row;
        uint ape_i = pos_mod * width + row;
        st_kv_buf[dst] = mf16c_red_kv[0];
        st_sc_buf[dst] =
            mf16c_red_sc[0] + model_scalar(ape_buf, 0u, params.index, ape_i);
    }
}
