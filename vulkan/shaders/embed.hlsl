/* vulkan/shaders/embed.hlsl -- token embedding dequant (Q8_0) and HC
 * embedding (f16, broadcast to n_hc copies).
 *
 * out[d] = scale(block) * qs(lane) for token `index` (single) or a batch of
 * tokens (tokens buffer on binding 0).  The model buffer (binding 3) is
 * bound at the embedding tensor's byte offset; block indices are relative.
 * The HC variants broadcast one f16 embedding row across n_hc copies:
 * out[e] = f16(w[token * n_embd + e % n_embd]).
 */

#include "common.hlsl"

StructuredBuffer<int> tokens_buf : register(t0);
ByteAddressBuffer w_buf : register(t3);
RWStructuredBuffer<float> out_buf : register(u0);

[[vk::push_constant]] DS4Params params;

/* Dequant one embedding element: block = (tok * row_blocks + d / 32),
 * lane = d % 32. */
float embed_q8_elem(int tok, uint d) {
    uint row_blocks = params.blocks;
    uint block = (uint)tok * row_blocks + (d >> 5u);
    uint lane = d & 31u;
    return q8_scale(w_buf, block) * (float)q8_s8(w_buf, block, lane);
}

/* f16 HC embedding element: token row replicated across n_hc copies.  The
 * model buffer is bound at the f16 token_embd byte offset.  f16_at already
 * returns a widened float -- do NOT wrap it in f16tof32 again (dxc then
 * emits float->uint->half, zeroing the value). */
float embed_hc_elem(uint tok, uint d) {
    return f16_at(w_buf, (tok * params.n + d) * 2u);
}

[numthreads(256, 1, 1)]
void embed_token_q8_0(uint3 gid : SV_DispatchThreadID) {
    if (gid.x >= params.n) return;
    out_buf[gid.x] = embed_q8_elem((int)params.index, gid.x);
}

[numthreads(256, 1, 1)]
void embed_tokens_q8_0(uint3 gid : SV_DispatchThreadID) {
    uint total = params.rows * params.n;
    if (gid.x >= total) return;
    uint t = gid.x / params.n;
    uint d = gid.x - t * params.n;
    int tok = tokens_buf[t];
    out_buf[gid.x] = embed_q8_elem(tok, d);
}

/* Single-token HC embedding: params.n = n_embd, params.aux = n_hc,
 * params.index = token. */
[numthreads(256, 1, 1)]
void embed_token_hc(uint3 gid : SV_DispatchThreadID) {
    uint n = params.n * params.aux;
    if (gid.x >= n) return;
    uint d = gid.x % params.n;
    out_buf[gid.x] = embed_hc_elem(params.index, d);
}

/* Batched HC embedding: params.n = n_embd, params.aux = n_hc,
 * params.rows = n_tokens, params.blocks = n_vocab. */
[numthreads(256, 1, 1)]
void embed_tokens_hc(uint3 gid : SV_DispatchThreadID) {
    uint total = params.rows * params.n * params.aux;
    if (gid.x >= total) return;
    uint d = gid.x % params.n;
    uint t = gid.x / (params.n * params.aux);
    int tok = tokens_buf[t];
    if (tok < 0) tok = 0;
    uint ub = (uint)tok;
    if (ub >= params.blocks) ub = 0u;
    out_buf[gid.x] = embed_hc_elem(ub, d);
}
