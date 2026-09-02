#!/usr/bin/env python3
"""vulkan/shaders/gen_shaders.py -- compile DS4 Vulkan HLSL shaders to SPIR-V
and emit a C header of embedded byte arrays (ds4_vulkan_shaders.inc).

Requires: dxc (DirectXShaderCompiler) on PATH, python3.
Output:   vulkan/shaders/ds4_vulkan_shaders.inc  (NOT committed)

The Makefile runs this before building ds4_vulkan.o.  Each compiled shader is
emitted as `static const uint32_t ds4_spv_<name>[]` plus its byte length.
"""

import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "ds4_vulkan_shaders.inc")

# (inc-identifier, source file, entry point) or
# (inc-identifier, source file, entry point, extra dxc args list)
SHADERS = [
    ("unary_add", "unary.hlsl", "add"),
    ("unary_add3", "unary.hlsl", "add3"),
    ("unary_swiglu", "unary.hlsl", "swiglu"),
    ("argmax", "argmax.hlsl", "argmax"),
    ("sort_i32_rows_asc", "sort.hlsl", "sort_i32_rows_asc"),
    ("rms_norm_plain", "rmsnorm.hlsl", "rms_norm_plain"),
    ("rms_norm_weight", "rmsnorm.hlsl", "rms_norm_weight"),
    ("rope_tail", "rope.hlsl", "rope_tail"),
    ("matmul_q8_0", "matmul_q8.hlsl", "matmul_q8_0"),
    ("matmul_q8_0_preq", "matmul_q8_preq.hlsl", "matmul_q8_0_preq"),
    ("matmul_q8_0_kslice", "matmul_q8_kslice.hlsl", "matmul_q8_0_kslice"),
    ("quantize_q8_0", "quantize_q8.hlsl", "quantize_q8_0"),
    ("top1", "top1.hlsl", "matmul_q8_0_top1"),
    ("f32_to_f16", "f16_conv.hlsl", "f32_to_f16"),
    ("matmul_f16", "matmul_f16.hlsl", "matmul_f16"),
    ("matmul_f16_pair_compressor_store", "matmul_f16_comp.hlsl",
     "matmul_f16_pair_compressor_store"),
    ("matmul_f32", "matmul_f32.hlsl", "matmul_f32"),
    ("embed_token_q8_0", "embed.hlsl", "embed_token_q8_0"),
    ("embed_tokens_q8_0", "embed.hlsl", "embed_tokens_q8_0"),
    ("embed_token_hc", "embed.hlsl", "embed_token_hc"),
    ("embed_tokens_hc", "embed.hlsl", "embed_tokens_hc"),
    ("fp8_kv_quantize", "kv.hlsl", "fp8_kv_quantize"),
    ("store_raw_kv", "kv.hlsl", "store_raw_kv"),
    ("kv_fp8_store_raw", "kv.hlsl", "kv_fp8_store_raw"),
    ("attn_decode", "attention.hlsl", "attn_decode"),
    ("attn_indexed_decode", "attention.hlsl", "attn_indexed_decode"),
    ("attn_prefill", "attention.hlsl", "attn_prefill"),
    ("attn_output_low_q8", "attention.hlsl", "attn_output_low_q8"),
    ("router_select", "router.hlsl", "router_select"),
    ("moe_gate_up_mid_q8", "moe.hlsl", "moe_gate_up_mid_q8"),
    ("moe_down_q8", "moe.hlsl", "moe_down_q8"),
    ("moe_sum", "moe.hlsl", "moe_sum"),
    ("moe_gate_up_mid_iq2xxs", "moe_iq2.hlsl", "moe_gate_up_mid_iq2xxs"),
    ("moe_down_q2k", "moe_iq2.hlsl", "moe_down_q2k"),
    ("fill_f32", "unary.hlsl", "fill_f32"),
    ("head_rms_norm", "rmsnorm.hlsl", "head_rms_norm"),
    ("head_rms_norm_rope_tail", "rmsnorm.hlsl", "head_rms_norm_rope_tail"),
    ("hc_split_sinkhorn", "hc.hlsl", "hc_split_sinkhorn"),
    ("hc_weighted_sum", "hc.hlsl", "hc_weighted_sum"),
    ("hc_expand", "hc.hlsl", "hc_expand"),
    ("hc_expand4", "hc.hlsl", "hc_expand4"),
    ("hc_split_weighted_sum_fused", "hc.hlsl", "hc_split_weighted_sum_fused"),
    ("output_hc_weights", "hc.hlsl", "output_hc_weights"),
    ("repeat_hc", "hc.hlsl", "repeat_hc"),
    ("hc_expand4_half", "hc_half.hlsl", "hc_expand4_half"),
    ("hc_expand4_add_half", "hc_add_half.hlsl", "hc_expand4_add_half"),
    ("indexer_scores", "indexer.hlsl", "indexer_scores"),
    ("dsv4_indexer_qat", "indexer.hlsl", "dsv4_indexer_qat"),
    ("indexer_topk", "indexer_topk.hlsl", "indexer_topk"),
    ("indexer_top1_value", "indexer_topk.hlsl", "indexer_top1_value"),
    ("topk_mask", "topk_mask.hlsl", "topk_mask"),
    ("dspark_markov_argmax", "dspark.hlsl", "dspark_markov_argmax"),
    ("compressor_store", "compressor.hlsl", "compressor_store"),
    ("compressor_set_rows", "compressor.hlsl", "compressor_set_rows"),
    ("compressor_prefill_pool", "compressor.hlsl", "compressor_prefill_pool"),
    ("compressor_update_pool", "compressor.hlsl", "compressor_update_pool"),
    ("compressor_shift_ratio4", "compressor.hlsl", "compressor_shift_ratio4"),
    ("directional_steering_project", "directional.hlsl",
     "directional_steering_project"),
    ("matmul_q8_0_preq_v2", "matmul_q8_preq.hlsl", "matmul_q8_0_preq_v2"),
    ("moe_gate_up_mid_iq2xxs_v2", "moe_iq2.hlsl", "moe_gate_up_mid_iq2xxs_v2"),
    ("moe_down_q2k_v2", "moe_iq2.hlsl", "moe_down_q2k_v2"),
]

# Binding shifts must match common.hlsl and the descriptor layout in
# ds4_vulkan.c: SRVs at set 0 bindings 0..3, UAVs at 4..7, cbuffer at 8.
DXC_FLAGS = [
    "-spirv",
    "-fspv-target-env=vulkan1.0",
    "-fvk-t-shift", "0", "0",
    "-fvk-u-shift", "4", "0",
    "-fvk-b-shift", "8", "0",
]


def find_dxc():
    exe = os.environ.get("DXC", "dxc")
    if shutil.which(exe):
        return exe
    sys.stderr.write("gen_shaders.py: dxc not found (set DXC or add it to PATH)\n")
    sys.exit(1)


def compile_spv(dxc, src_path, entry, out_path, profile):
    cmd = [dxc, "-E", entry] + profile + DXC_FLAGS
    cmd += ["-Fo", out_path, src_path]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)


def emit_array(f, name, data):
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    f.write("static const uint32_t ds4_spv_%s[] = {\n" % name)
    line = []
    for w in words:
        line.append("0x%08x," % w)
        if len(line) == 8:
            f.write("    " + " ".join(line) + "\n")
            line = []
    if line:
        f.write("    " + " ".join(line) + "\n")
    f.write("};\n")
    f.write("static const size_t ds4_spv_%s_len = sizeof(ds4_spv_%s);\n\n"
            % (name, name))


def main():
    dxc = find_dxc()
    tmp_spv = os.path.join(HERE, ".tmp.shader.spv")
    with open(OUTPUT, "w") as f:
        f.write("/* Generated by gen_shaders.py -- DO NOT EDIT.\n")
        f.write(" * Compiled from the HLSL sources in this directory with dxc\n")
        f.write(" * (HLSL to SPIR-V).\n")
        f.write(" */\n\n")
        f.write("#include <stddef.h>\n\n")
        for sh in SHADERS:
            name, src, entry = sh[0], sh[1], sh[2]
            profile = sh[3] if len(sh) > 3 else ["-T", "cs_6_0"]
            compile_spv(dxc, os.path.join(HERE, src), entry, tmp_spv, profile)
            with open(tmp_spv, "rb") as spv:
                data = spv.read()
            if len(data) % 4 != 0:
                data += b"\x00" * (4 - len(data) % 4)
            emit_array(f, name, data)
            os.unlink(tmp_spv)
    print("gen_shaders.py: wrote %s (%d shaders)" % (OUTPUT, len(SHADERS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
