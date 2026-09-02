RWStructuredBuffer<float> out_buf : register(u0);
[[vk::push_constant]] struct P { uint n; uint iters; uint pad0; uint pad1; } p;
[numthreads(256,1,1)]
void madloop(uint3 gid : SV_DispatchThreadID) {
    if (gid.x >= p.n) return;
    float acc = (float)(gid.x & 1023u) * 1.000001f + 1.0f;
    float x = 1.0000001f;
    for (uint k = 0u; k < p.iters; k++) {
        acc = acc * x + 1.0f;
        acc = acc * x + 1.0f;
        acc = acc * x + 1.0f;
        acc = acc * x + 1.0f;
    }
    out_buf[gid.x] = acc;
}
