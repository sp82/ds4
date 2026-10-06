/* read_bench.hlsl -- pure-read DRAM bandwidth kernel for mgpu-probe.
 *
 * Grid-stride read: p.n_total threads each read p.elem uint4s (16 B each),
 * striding by p.n_total, XOR-reduce, and one thread per 256-thread group
 * writes a single uint to out_buf.  Write traffic = n_groups * 4 B (~0.01% of
 * the reads), so this measures PURE read bandwidth -- not a read+write mix like
 * vkCmdCopyBuffer (which moves 1 bus byte read + 1 written per copied byte and
 * is bounded by the TOTAL DRAM bus, e.g. ~512 GB/s on an RX 6900 XT).
 *
 * The inner loop is unrolled by 8 so 8 independent loads are in flight per
 * thread, while the loop itself keeps the thread resident for many iterations.
 * This matters: a kernel that launches one short-lived thread per 16 B (n_read
 * = 1, millions of tiny blocks) measures block-dispatch throughput on RDNA2,
 * NOT memory bandwidth -- it tops out around 16 GB/s no matter the access
 * pattern.  Coalescing is achieved by the stride: consecutive lanes read
 * consecutive 16 B in every load.
 *
 * Dispatch: 2D with ROW_W threads per row so no 1D count exceeds the driver's
 * maxComputeWorkCount (radv reports 2^20); the last row may overshoot and is
 * guarded by n_total.
 *
 * Binding layout (dxc -fvk-t-shift 0 0 / -fvk-u-shift 2 0): t0 -> binding 0
 * (input), u0 -> binding 2 (output).  Both in descriptor set 0.
 */
StructuredBuffer<uint4> in_buf : register(t0);
RWStructuredBuffer<uint> out_buf : register(u0);
[[vk::push_constant]] struct P { uint n_total; uint elem; uint stride; uint pad; } p;
#define ROW_W 1048576u
#define X4(a) ((a).x ^ (a).y) ^ ((a).z ^ (a).w)
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint gtid = tid.y * ROW_W + tid.x;
    if (gtid >= p.n_total) return;
    uint idx = gtid;
    uint v = 0;
    for (uint k = 0; k < p.elem; k += 8u) {
        uint4 a0 = in_buf[idx]; idx += p.stride;
        uint4 a1 = in_buf[idx]; idx += p.stride;
        uint4 a2 = in_buf[idx]; idx += p.stride;
        uint4 a3 = in_buf[idx]; idx += p.stride;
        uint4 a4 = in_buf[idx]; idx += p.stride;
        uint4 a5 = in_buf[idx]; idx += p.stride;
        uint4 a6 = in_buf[idx]; idx += p.stride;
        uint4 a7 = in_buf[idx]; idx += p.stride;
        v ^= X4(a0) ^ X4(a1) ^ X4(a2) ^ X4(a3) ^ X4(a4) ^ X4(a5) ^ X4(a6) ^ X4(a7);
    }
    if ((gtid & 255u) == 0u) out_buf[gtid >> 8] = v;
}
