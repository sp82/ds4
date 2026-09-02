/* vulkan/shaders/unary.hlsl -- elementwise flat kernels (add, add3, swiglu).
 *
 * Compiled to SPIR-V with dxc (HLSL -> SPIR-V).  Binding convention (see
 * common.hlsl): t0/t1/t2 land on set 0 bindings 0/1/2, u0 on binding 4.
 * Scalar parameters travel in the fixed DS4Params push constant block.
 */

#include "common.hlsl"

StructuredBuffer<float> a_buf : register(t0);
StructuredBuffer<float> b_buf : register(t1);
StructuredBuffer<float> c_buf : register(t2);
RWStructuredBuffer<float> out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

/* out[i] = a[i] + b[i] */
[numthreads(256, 1, 1)]
void add(uint3 gid : SV_DispatchThreadID) {
    if (gid.x < params.n) {
        out_buf[gid.x] = a_buf[gid.x] + b_buf[gid.x];
    }
}

/* out[i] = a[i] + b[i] + c[i] */
[numthreads(256, 1, 1)]
void add3(uint3 gid : SV_DispatchThreadID) {
    if (gid.x < params.n) {
        out_buf[gid.x] = a_buf[gid.x] + b_buf[gid.x] + c_buf[gid.x];
    }
}

/* out[i] = silu(gate[i]) * up[i] * weight, with optional clamp on both
 * inputs (matches CUDA swiglu_kernel). */
[numthreads(256, 1, 1)]
void swiglu(uint3 gid : SV_DispatchThreadID) {
    if (gid.x >= params.n) return;
    float g = a_buf[gid.x];
    float u = b_buf[gid.x];
    if (params.clamp > 1.0e-6f) {
        g = min(g, params.clamp);
        u = min(max(u, -params.clamp), params.clamp);
    }
    float s = g / (1.0f + exp(-g));
    out_buf[gid.x] = s * u * params.weight;
}

/* out[i] = value (flat fill; used by the compressor state initialization).
 * params.n = count, params.weight = value. */
[numthreads(256, 1, 1)]
void fill_f32(uint3 gid : SV_DispatchThreadID) {
    if (gid.x < params.n) out_buf[gid.x] = params.weight;
}
