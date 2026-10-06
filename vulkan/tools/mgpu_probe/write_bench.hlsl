/* write_bench.hlsl -- pure-write DRAM bandwidth kernel for mgpu-probe.
 *
 * Grid-stride write: p.n_total threads each write p.elem uint4s (16 B each),
 * striding by p.n_total, and nothing is ever read.  This measures PURE write
 * bandwidth from the shader (compute) path -- the counterpart to
 * read_bench.hlsl, and directly comparable to the DMA vkCmdFillBuffer result.
 *
 * Same rationale as read_bench: one thread per element would measure
 * block-dispatch throughput, not DRAM, so each thread writes many elements
 * with an 8-wide unrolled inner loop (8 independent stores in flight).
 * Coalescing comes from the stride: consecutive lanes write consecutive 16 B.
 *
 * Dispatch: 2D with ROW_W threads per row so no 1D count exceeds the driver's
 * maxComputeWorkCount (radv reports 2^20); the last row may overshoot and is
 * guarded by n_total.
 *
 * Binding layout (dxc -fvk-u-shift 2 0): u0 -> binding 2 (output) in set 0.
 * (The shared descriptor set also has binding 0, bound but unused here.)
 */
RWStructuredBuffer<uint4> out_buf : register(u0);
[[vk::push_constant]] struct P { uint n_total; uint elem; uint stride; uint seed; } p;
#define ROW_W 1048576u
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint gtid = tid.y * ROW_W + tid.x;
    if (gtid >= p.n_total) return;
    uint4 v = uint4(gtid, gtid ^ 0x9e3779b9u, p.seed, gtid + p.seed);
    uint idx = gtid;
    for (uint k = 0; k < p.elem; k += 8u) {
        out_buf[idx] = v; idx += p.stride;
        out_buf[idx] = v; idx += p.stride;
        out_buf[idx] = v; idx += p.stride;
        out_buf[idx] = v; idx += p.stride;
        out_buf[idx] = v; idx += p.stride;
        out_buf[idx] = v; idx += p.stride;
        out_buf[idx] = v; idx += p.stride;
        out_buf[idx] = v; idx += p.stride;
    }
}
