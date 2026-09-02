StructuredBuffer<float> a_buf : register(t0);
StructuredBuffer<float> b_buf : register(t1);
RWStructuredBuffer<float> out_buf : register(u0);
[[vk::push_constant]] struct P { uint n; float alpha; uint pad0; uint pad1; } p;
[numthreads(256,1,1)]
void saxpy(uint3 gid : SV_DispatchThreadID) {
    if (gid.x < p.n) out_buf[gid.x] = a_buf[gid.x] * p.alpha + b_buf[gid.x];
}
