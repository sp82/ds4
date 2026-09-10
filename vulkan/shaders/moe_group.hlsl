/* vulkan/shaders/moe_group.hlsl -- device-side counting sort of routed-MoE
 * (token, expert) pairs by expert.
 *
 * The routed-MoE kernels run one workgroup per (row, pair); without grouping
 * each expert's weights are re-read once per token that selected it.  This
 * kernel produces `order[p]` = the pair index at grouped position p, with all
 * pairs of one expert contiguous, so the main kernels walk the pairs in
 * expert order and reuse the expert's weights from L2.
 *
 * One workgroup.  params: index=n_total, rows=pair_count, flags bit0 =
 * expert ids go through the pool table (same convention as the MoE kernels).
 * Bindings: t0 selected, t1 expert->slot table (pool), u0 order out. */

#include "common.hlsl"

StructuredBuffer<int> selected_g : register(t0);
StructuredBuffer<int> tbl_g : register(t1);
RWStructuredBuffer<uint> order_g : register(u0);

[[vk::push_constant]] DS4Params params;

#define MOE_GROUP_MAX 512
groupshared uint hist[MOE_GROUP_MAX];
groupshared uint cursor[MOE_GROUP_MAX];

static uint moe_group_expert(int sel) {
    if (sel < 0) sel = 0;
    if ((params.flags & 1u) != 0u) {
        int ts = tbl_g[sel];
        return (uint)(ts < 0 ? 0 : ts);
    }
    return (uint)sel;
}

[numthreads(256, 1, 1)]
void moe_group(uint3 gid : SV_GroupID, uint tid : SV_GroupThreadID) {
    uint n_total = params.index;
    uint pair_count = params.rows;
    if (n_total == 0u || n_total > MOE_GROUP_MAX || pair_count == 0u) return;

    for (uint i = tid; i < n_total; i += 256u) hist[i] = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint p = tid; p < pair_count; p += 256u) {
        uint e = moe_group_expert(selected_g[p]);
        if (e < n_total) InterlockedAdd(hist[e], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0u) {
        uint acc = 0u;
        for (uint i = 0u; i < n_total; i++) {
            cursor[i] = acc;
            acc += hist[i];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint p = tid; p < pair_count; p += 256u) {
        uint e = moe_group_expert(selected_g[p]);
        if (e < n_total) {
            uint pos = 0u;
            InterlockedAdd(cursor[e], 1u, pos);
            order_g[pos] = p;
        }
    }
}
