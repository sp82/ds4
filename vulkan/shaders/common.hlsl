/* vulkan/shaders/common.hlsl -- shared definitions for DS4 Vulkan compute
 * shaders.
 *
 * Compiled to SPIR-V with dxc (HLSL -> SPIR-V).  The binding convention is
 * fixed by the host: SRVs (StructuredBuffer / ByteAddressBuffer) land on set
 * 0 bindings 0..3 (via -fvk-t-shift 0 0), UAVs (RWStructuredBuffer) on
 * bindings 4..5 (via -fvk-u-shift 4 0).  Match these offsets in
 * ds4_vulkan.c when building the descriptor set layout and updating
 * descriptors.
 *
 * Scalar kernel parameters travel in one fixed push constant block
 * (DS4Params, 96 bytes).  All kernels share the same struct layout so the
 * host pushes a single blob for every dispatch.
 */

#ifndef DS4_VULKAN_COMMON_HLSL
#define DS4_VULKAN_COMMON_HLSL

#define DS4_PI 3.14159265358979323846

/* Fixed push constant block shared by every kernel.  Field semantics are
 * per-kernel; unused fields are ignored.  The host mirrors this layout in
 * struct ds4_vk_params (ds4_vulkan.c). */
struct DS4Params {
    uint  n;          /* element count / head_dim / n_embd / row_width */
    uint  rows;       /* n_tok / n_rows / n_heads * n_tok */
    uint  in_dim;     /* matmul in dim (elements) */
    uint  out_dim;    /* matmul out dim (rows) */
    uint  n_rot;      /* rope rotation count */
    uint  pos0;       /* rope starting position */
    uint  n_ctx_orig; /* rope original context */
    int   inverse;    /* rope inverse flag */
    float eps;        /* rms norm epsilon */
    float clamp;      /* swiglu clamp */
    float weight;     /* swiglu output weight */
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    uint  blocks;     /* q8_0 block count (in_dim / 32) */
    uint  index;      /* token id / n_head / generic index */
    uint  aux;        /* window / raw_cap / spare */
    uint  ratio;      /* compression ratio */
    uint  flags;      /* use_comp_mask / use_mask / spare */
    uint  rsvd2;
    uint  rsvd3;
};

/* Q8_0 block: { uint16_t scale_half; int8_t qs[32]; } = 34 bytes.  The
 * blocks are not 4-byte aligned (34 % 4 == 2), so the byte helpers below
 * read the aligned 32-bit word that covers a byte and shift it out. */
uint q8_byte(ByteAddressBuffer w, uint block, uint byte_in_block) {
    uint off = block * 34u + byte_in_block;
    return (w.Load(off & ~3u) >> ((off & 3u) * 8u)) & 0xffu;
}

float q8_scale(ByteAddressBuffer w, uint block) {
    uint bits = q8_byte(w, block, 0u) | (q8_byte(w, block, 1u) << 8u);
    return f16tof32(bits);
}

/* Sign-extend an int8 read as an unsigned byte. */
int q8_s8(ByteAddressBuffer w, uint block, uint i) {
    uint b = q8_byte(w, block, 2u + i);
    return b < 128u ? (int)b : (int)b - 256;
}

/* Sign-extend four int8 packed in the low 4 bytes of an int32 (byte 0 in
 * the lowest 8 bits, little-endian). */
int4 s8x4(int v) {
    return int4((v << 24) >> 24, (v << 16) >> 24, (v << 8) >> 24, v >> 24);
}

/* Read a single unaligned byte at an absolute byte offset `base + k`
 * (relative to the bound buffer).  The offset can exceed the 4-byte window
 * the existing q8_byte() covers, so MoE kernels that read weight rows at
 * expert-scoped byte offsets use these helpers instead. */
uint q8_byte_at(ByteAddressBuffer w, uint base, uint k) {
    uint off = base + k;
    return (w.Load(off & ~3u) >> ((off & 3u) * 8u)) & 0xffu;
}

float q8_scale_at(ByteAddressBuffer w, uint base) {
    uint bits = q8_byte_at(w, base, 0u) | (q8_byte_at(w, base, 1u) << 8u);
    return f16tof32(bits);
}

int q8_s8_at(ByteAddressBuffer w, uint base, uint i) {
    uint b = q8_byte_at(w, base, 2u + i);
    return b < 128u ? (int)b : (int)b - 256;
}

/* Numerically stable softplus (matches CUDA softplus_dev). */
float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    return log(1.0f + exp(x));
}

/* Pack four int8 into one uint32 (little-endian byte order). */
uint pack8(int a, int b, int c, int d) {
    return ((uint)a & 0xffu) | (((uint)b & 0xffu) << 8) |
           (((uint)c & 0xffu) << 16) | (((uint)d & 0xffu) << 24);
}

/* Integer dot of 32 signed bytes, one operand stored unaligned in the model
 * buffer (via q8_s8), the other prequantized as packed int32 words (read
 * 4-aligned from a ByteAddressBuffer). */
int dot8x32_w_xq(ByteAddressBuffer w, uint wblock, ByteAddressBuffer xq,
                 uint xq_byte) {
    int acc = 0;
    for (uint i = 0u; i < 32u; i += 4u) {
        int4 qw = s8x4((int)xq.Load(xq_byte + i));
        acc += qw.x * q8_s8(w, wblock, i) + qw.y * q8_s8(w, wblock, i + 1u) +
               qw.z * q8_s8(w, wblock, i + 2u) + qw.w * q8_s8(w, wblock, i + 3u);
    }
    return acc;
}

/* NTK/YaRN rope correction, mirroring llama.cpp rope_yarn_ramp. */
float rope_yarn_ramp(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

/* Read an unaligned uint16 and widen it to f16.  `off` is a byte offset
 * relative to the bound buffer (same window-shift trick as q8_byte). */
float f16_at(ByteAddressBuffer w, uint off) {
    uint bits = (w.Load(off & ~3u) >> ((off & 3u) * 8u)) & 0xffffu;
    return f16tof32(bits);
}

/* f32 -> f16 bit pattern, round-to-nearest-even, parity with CUDA
 * __float2half and the host f32_to_f16_rne reference.  Do NOT use HLSL
 * f32tof16 here: dxc lowers it to GLSL.std.450 PackHalf2x16, which is not
 * guaranteed correctly rounded -- RADV/LLVM emits the truncating
 * v_cvt_pkrtz_f16_f32 on RDNA2 (NAVI21), giving values half an f16 ULP off
 * from RNE on those GPUs.  The explicit bit manipulation is portable. */
uint ds4_f32_to_f16_bits_rne(float v) {
    uint bits = asuint(v);
    uint sign = (bits >> 16) & 0x8000u;
    int exp = (int)((bits >> 23) & 0xffu) - 127 + 15;
    uint man = bits & 0x7fffffu;
    uint h;
    if (exp >= 31) {
        h = sign | 0x7bffu;
    } else if (exp <= 0) {
        if (exp < -10) return sign;  /* underflow to zero */
        uint mant = man | 0x800000u;
        uint shift = (uint)(1 - exp);
        uint m = mant >> shift;
        h = sign | (m >> 13);
        uint rem = (mant >> shift) & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
        return h;
    } else {
        h = sign | ((uint)exp << 10) | (man >> 13);
        uint rem = man & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    }
    return h;
}

/* Sign-extended int8 read at an unaligned byte offset. */
int s8_at(ByteAddressBuffer w, uint off) {
    uint b = (w.Load(off & ~3u) >> ((off & 3u) * 8u)) & 0xffu;
    return b < 128u ? (int)b : (int)b - 256;
}

/* --- Q4_K (GGUF type 12) dequant helpers ---------------------------------
 * 144-byte blocks / 256 values: d (f16) @0, dmin (f16) @2, scales[12] @4,
 * qs[128] @16.  Eight 32-value groups; group j uses byte_off=(j>>1)*32 and
 * shift=(j&1)*4 in qs, with a two-level scale/min decoded from scales[]
 * (q4_k_get_scale_min).  Value = d*sc*nib - dmin*m (nib is not biased). */
void q4k_scale_min(ByteAddressBuffer w, uint bbase, uint j,
                   out uint sc, out uint m) {
    if (j < 4u) {
        sc = q8_byte_at(w, bbase + 4u, j) & 63u;
        m  = q8_byte_at(w, bbase + 4u, j + 4u) & 63u;
    } else {
        uint a = q8_byte_at(w, bbase + 4u, j + 4u);
        uint b = q8_byte_at(w, bbase + 4u, j - 4u);
        uint c = q8_byte_at(w, bbase + 4u, j);
        sc = (a & 0xfu) | ((b >> 6u) << 4u);
        m  = (a >> 4u)  | ((c >> 6u) << 4u);
    }
}

/* Dot of one 32-value Q4_K sub-block (group j, 0..7) at byte offset bbase
 * against x[xb..xb+31].  Mirrors ds4_vec_dot_q4_K_f32. */
float q4k_sub_dot(ByteAddressBuffer w, uint bbase, uint j,
                  StructuredBuffer<float> x, uint xb) {
    float d    = f16_at(w, bbase + 0u);
    float dmin = f16_at(w, bbase + 2u);
    uint sc, m;
    q4k_scale_min(w, bbase, j, sc, m);
    float scale = d * (float)sc;
    float minv  = dmin * (float)m;
    uint byte_off = (j >> 1u) * 32u;
    uint shift = (j & 1u) * 4u;
    float acc = 0.0f;
    for (uint l = 0u; l < 32u; l++) {
        uint qb = q8_byte_at(w, bbase + 16u, byte_off + l);
        acc += (scale * (float)((qb >> shift) & 0xfu) - minv) * x[xb + l];
    }
    return acc;
}

/* --- MXFP4 (GGUF type 39) dequant helpers --------------------------------
 * 17-byte blocks / 32 values: e (E8M0 exponent) @0 + qs[16].  qs[j] packs two
 * 4-bit FP4 (E2M1) codes: low nibble = element j, high nibble = element j+16.
 * Value = e8m0_to_f32(e) * mxfp4_value(nib).  Mirrors ds4_vec_dot_mxfp4_f32. */
float mxfp4_value(uint nib) {
    float mag;
    switch (nib & 7u) {
    case 0u: mag = 0.0f; break;
    case 1u: mag = 0.5f; break;
    case 2u: mag = 1.0f; break;
    case 3u: mag = 1.5f; break;
    case 4u: mag = 2.0f; break;
    case 5u: mag = 3.0f; break;
    case 6u: mag = 4.0f; break;
    default: mag = 6.0f; break;
    }
    return (nib & 8u) != 0u ? -mag : mag;
}

/* E8M0 exponent byte -> f32 (e == 0 -> 2^-127, else 2^(e-127)). */
float e8m0_to_f32(uint e) {
    uint bits = (e == 0u) ? 0x00400000u : (e << 23u);
    return asfloat(bits);
}

/* Dot of one 32-value MXFP4 block at byte offset bbase against
 * x[xb..xb+31]. */
float mxfp4_block_dot(ByteAddressBuffer w, uint bbase,
                      StructuredBuffer<float> x, uint xb) {
    float d = e8m0_to_f32(q8_byte_at(w, bbase, 0u));
    float acc = 0.0f;
    for (uint j = 0u; j < 16u; j++) {
        uint q = q8_byte_at(w, bbase + 1u, j);
        acc += d * mxfp4_value(q & 0xfu) * x[xb + j];
        acc += d * mxfp4_value(q >> 4u) * x[xb + j + 16u];
    }
    return acc;
}

/* Read a model scalar at a byte offset: type 0 = f32, 1 = f16 (the
 * compressor ape tensors).  Mirrors CUDA model_scalar_dev. */
float model_scalar(ByteAddressBuffer w, uint base, uint type, uint idx) {
    if (type == 1u) return f16_at(w, base + idx * 2u);
    return asfloat(w.Load(base + idx * 4u));
}

/* E2M1FN (fp4) representable magnitudes (index 0..7).  Used by the indexer
 * Hadamard-FP4 QAT path (dsv4_indexer_qat). */
float ds4_e2m1fn_value(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

/* Nearest E2M1FN representation (tie -> even), parity with CUDA
 * dsv4_e2m1fn_dequant_dev. */
float ds4_e2m1fn_dequant(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = min(abs(x), 6.0f);
    int best = 0;
    float best_diff = abs(ax - ds4_e2m1fn_value(0));
    for (int i = 1; i < 8; i++) {
        float diff = abs(ax - ds4_e2m1fn_value(i));
        if (diff < best_diff ||
            (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * ds4_e2m1fn_value(best);
}

#endif /* DS4_VULKAN_COMMON_HLSL */
