#ifndef DS4_LAYER_PACK_H
#define DS4_LAYER_PACK_H

#include <stddef.h>
#include <stdio.h>

#define DS4_LAYER_PACK_MAX_GPUS 16
#define DS4_LAYER_PACK_CPU      (-1)

typedef struct {
    /* Per-device budget in BYTES (already net of per-device overhead such as
     * cuBLAS workspace, global scratch, logits buffer, MTP, steering, and the
     * configured safety margin). The packer treats these as raw capacity. */
    size_t gpu_budget_bytes[DS4_LAYER_PACK_MAX_GPUS];
    int    n_gpus;
} ds4_layer_pack_config;

#ifdef __cplusplus
extern "C" {
#endif

/* Compute a monotonic-contiguous layer placement.
 *
 * Inputs:
 *   entry_bytes[]     — per-entry byte footprint, in FORWARD ORDER:
 *                         entry 0           = embedding pseudo-layer
 *                         entry 1..n_layers = transformer layers
 *                         entry n_layers+1  = output-head pseudo-layer
 *   n_entries         — total number of entries (typically n_layers + 2)
 *   cfg               — per-device budgets
 *
 * Output:
 *   device_for_entry[] — filled with target device for each entry; CPU is
 *                        DS4_LAYER_PACK_CPU. Caller-owned, must be
 *                        large enough for n_entries int slots.
 *
 * Returns:
 *   0 on success.
 *   Nonzero on configuration errors (null pointer, n_entries < 0,
 *   n_gpus out of range).
 *
 * Per the design doc, an entry that exceeds every GPU budget spills to CPU;
 * by the monotonicity rule every entry after it also goes to CPU. There is
 * no error path for "entry too large" — the CPU tier is always available. */
int ds4_compute_layer_placement(const size_t *entry_bytes,
                                int n_entries,
                                const ds4_layer_pack_config *cfg,
                                int *device_for_entry);

/* Print a human-readable layout summary matching the design doc's format:
 *
 *   multi-GPU layout:
 *     GPU0: layers 0-21 + embedding   (38.4 / 40.0 GB)
 *     GPU1: layers 22-31              (11.7 / 12.0 GB)
 *     CPU : layers 32-42 + output head
 *
 * Entry-naming convention:
 *   entry 0          -> "embedding"
 *   entry i in 1..n_layers -> transformer layer (numbered i-1)
 *   entry n_layers+1 -> "output head"
 *
 * gpu_used_bytes[] and gpu_budget_bytes[] may be NULL — in that case the
 * "(used / budget)" line is omitted for the GPU lines. */
void ds4_layer_pack_print(FILE *out,
                          const int *device_for_entry,
                          int n_entries,
                          int n_layers,
                          const size_t *entry_bytes,
                          const size_t *gpu_used_bytes,
                          const size_t *gpu_budget_bytes,
                          int n_gpus);

/* -------------------------------------------------------------------------
 * Multi-GPU memory/placement config (SPECS_MGPU.md M2).
 *
 * The JSON file (--gpu-config) describes the per-device role: layer range,
 * static vs dynamic expert mode, expert cache budget, and the recency pin.
 * When ranges are absent the asymmetric auto-layout is computed: the slow
 * (x1) tiers take the first contiguous layers statically, the fast (x16)
 * tier takes the rest dynamically.
 * ------------------------------------------------------------------------- */

#define DS4_MGPU_AUTO   (-1)
#define DS4_MGPU_OUTPUT (-2)

typedef struct {
    int    index;             /* physical device index; DS4_MGPU_AUTO = by position */
    int    layer_start;       /* DS4_MGPU_AUTO or first layer (0-based) */
    int    layer_end;         /* inclusive; DS4_MGPU_OUTPUT = through output head */
    int    mode;              /* 0 static, 1 dynamic, DS4_MGPU_AUTO = by class */
    double expert_cache_gb;   /* <= 0 = auto */
    int    pin_experts;       /* DS4_MGPU_AUTO, 0/1 */
    int    pin_tokens;        /* DS4_MGPU_AUTO, 0/1 = count; -1 auto */
} ds4_mgpu_device_cfg;

typedef struct {
    int     version;
    int     placement_auto;   /* 1 = compute ranges (default) */
    double  safety_margin_gb;
    double  fast_link_gbps;
    int     n_devices;
    int     loaded;           /* 1 when read from a file */
    ds4_mgpu_device_cfg devices[DS4_LAYER_PACK_MAX_GPUS];
} ds4_mgpu_config;

void ds4_mgpu_config_defaults(ds4_mgpu_config *cfg);

/* Parse a JSON config file. Returns 0 on success, nonzero on error (err
 * populated). */
int ds4_mgpu_config_load(const char *path, ds4_mgpu_config *out,
                         char *err, size_t errlen);

void ds4_mgpu_config_dump(FILE *out, const ds4_mgpu_config *cfg);

/* Per-layer byte accounting for the planner (expert tensors separated). */
typedef struct {
    size_t dense_bytes;    /* non-expert weights of the layer */
    size_t expert_bytes;   /* all routed experts of the layer */
} ds4_mgpu_layer_bytes;

/* Resolved per-tier descriptor, in placement (tier) order. */
typedef struct {
    int    physical_index;
    int    is_fast;
    int    mode;               /* resolved: 0 static, 1 dynamic */
    size_t budget_bytes;       /* usable VRAM for weights + pool + scratch */
    size_t expert_cache_bytes; /* dynamic: pool budget (0 = all experts) */
    int    pin_experts;
    int    pin_tokens;
} ds4_mgpu_tier;

/* Build a contiguous placement over n_layers transformer layers.
 *
 * placement_out has n_layers + 2 entries: [0] = embedding, [1..n_layers] =
 * transformer layers, [n_layers+1] = output head. Each receives the tier
 * index or DS4_LAYER_PACK_CPU on spill. Static tiers are charged
 * dense_bytes + expert_bytes per layer; dynamic tiers are charged
 * dense_bytes only (experts stream). tiers[] is consumed in order: the
 * caller orders it so the slow tiers come first (auto-layout) or matches the
 * JSON device order (explicit). cfg may be NULL (pure auto). Returns 0 on
 * success, nonzero on error (err populated). */
int ds4_mgpu_plan(const ds4_mgpu_config *cfg,
                  const ds4_mgpu_tier *tiers, int n_tiers,
                  const ds4_mgpu_layer_bytes *layer_bytes, int n_layers,
                  size_t embedding_bytes, size_t output_bytes,
                  int *placement_out,
                  char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* DS4_LAYER_PACK_H */
