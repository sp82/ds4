/* Monotonic-contiguous multi-tier layer placement packer.
 *
 * Pure C99: no CUDA, no platform-specific code. Used by both the CUDA and
 * Metal/CPU builds. See ds4_layer_pack.h for the API contract and the
 * design doc docs/superpowers/specs/2026-05-26-multi-gpu-pp-v0-design.md
 * for the rationale. */

#include "ds4_layer_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ds4_compute_layer_placement(const size_t *entry_bytes,
                                int n_entries,
                                const ds4_layer_pack_config *cfg,
                                int *device_for_entry) {
    if (!entry_bytes || !cfg || !device_for_entry) return 1;
    if (n_entries < 0) return 2;
    if (cfg->n_gpus < 0 || cfg->n_gpus > DS4_LAYER_PACK_MAX_GPUS) return 3;

    /* Work over a local copy of budgets so the caller's struct is untouched. */
    size_t budget[DS4_LAYER_PACK_MAX_GPUS];
    for (int d = 0; d < cfg->n_gpus; d++) {
        budget[d] = cfg->gpu_budget_bytes[d];
    }

    int d = 0;
    for (int e = 0; e < n_entries; e++) {
        /* Advance until the current device can hold this entry, or we run
         * out of devices. Strict greater-than: exact fits stay on d. */
        while (d < cfg->n_gpus && entry_bytes[e] > budget[d]) {
            d++;
        }
        if (d < cfg->n_gpus) {
            device_for_entry[e] = d;
            budget[d] -= entry_bytes[e];
        } else {
            device_for_entry[e] = DS4_LAYER_PACK_CPU;
            /* d stays at cfg->n_gpus so every subsequent entry also lands
             * on the CPU tier. */
        }
    }
    return 0;
}

/* Internal helper: append " + embedding" or " + output head" to a buffer if
 * the matching pseudo-layer lives on this tier. */
static void append_pseudo_layer_tags(char *buf, size_t buflen,
                                      const int *device_for_entry,
                                      int tier,
                                      int n_entries,
                                      int n_layers) {
    /* entry 0 -> embedding, entry n_layers+1 -> output head */
    if (n_entries > 0 && device_for_entry[0] == tier) {
        strncat(buf, " + embedding", buflen - strlen(buf) - 1);
    }
    if (n_entries > 1 && n_layers + 1 < n_entries &&
        device_for_entry[n_layers + 1] == tier) {
        strncat(buf, " + output head", buflen - strlen(buf) - 1);
    }
}

/* Emit "L-R" for a contiguous transformer-layer range assigned to a tier.
 * Returns 1 if anything was printed (i.e. tier owned at least one layer). */
static int print_layer_range_for_tier(FILE *out, const int *device_for_entry,
                                      int tier, int n_entries, int n_layers) {
    /* Transformer layers live at indices 1..n_layers in device_for_entry.
     * Their human-facing numbers are 0..n_layers-1 (i.e. entry index - 1). */
    int first = -1;
    int last  = -1;
    for (int i = 1; i <= n_layers && i < n_entries; i++) {
        if (device_for_entry[i] == tier) {
            if (first < 0) first = i - 1;  /* human-facing index */
            last = i - 1;
        }
    }
    if (first < 0) {
        fprintf(out, "(no transformer layers)");
        return 0;
    }
    if (first == last) {
        fprintf(out, "layer %d", first);
    } else {
        fprintf(out, "layers %d-%d", first, last);
    }
    return 1;
}

void ds4_layer_pack_print(FILE *out,
                          const int *device_for_entry,
                          int n_entries,
                          int n_layers,
                          const size_t *entry_bytes,
                          const size_t *gpu_used_bytes,
                          const size_t *gpu_budget_bytes,
                          int n_gpus) {
    if (!out || !device_for_entry) return;
    (void)entry_bytes; /* not needed for the textual layout */

    fprintf(out, "multi-GPU layout:\n");
    for (int d = 0; d < n_gpus; d++) {
        fprintf(out, "  GPU%d: ", d);
        int have_layers = print_layer_range_for_tier(out, device_for_entry,
                                                     d, n_entries, n_layers);
        /* Pseudo-layer tags. */
        char tags[64];
        tags[0] = '\0';
        append_pseudo_layer_tags(tags, sizeof(tags), device_for_entry, d,
                                  n_entries, n_layers);
        if (tags[0]) {
            if (!have_layers) {
                /* Erase the "(no transformer layers)" stub and just write
                 * the tag with leading "+ " stripped. */
                /* The stub was already written; appending " + embedding"
                 * is still informative. */
            }
            fprintf(out, "%s", tags);
        }
        /* Usage line. */
        if (gpu_used_bytes && gpu_budget_bytes) {
            const double used_gb   = (double)gpu_used_bytes[d]   / 1073741824.0;
            const double budget_gb = (double)gpu_budget_bytes[d] / 1073741824.0;
            /* Two-space pad before the parens matches the design doc. */
            fprintf(out, "  (%.1f / %.1f GB)", used_gb, budget_gb);
        }
        fputc('\n', out);
    }
    /* CPU tier: only print if at least one entry is on CPU. */
    int cpu_present = 0;
    for (int i = 0; i < n_entries; i++) {
        if (device_for_entry[i] == DS4_LAYER_PACK_CPU) { cpu_present = 1; break; }
    }
    if (cpu_present) {
        fprintf(out, "  CPU : ");
        int have_layers = print_layer_range_for_tier(out, device_for_entry,
                                                      DS4_LAYER_PACK_CPU,
                                                      n_entries, n_layers);
        char tags[64];
        tags[0] = '\0';
        append_pseudo_layer_tags(tags, sizeof(tags), device_for_entry,
                                  DS4_LAYER_PACK_CPU, n_entries, n_layers);
        if (tags[0]) {
            (void)have_layers;
            fprintf(out, "%s", tags);
        }
        fputc('\n', out);
    }
}

/* -------------------------------------------------------------------------
 * Multi-GPU config (JSON) + asymmetric placement planner.
 * See ds4_layer_pack.h and vulkan/SPECS_MGPU.md.
 * ------------------------------------------------------------------------- */

static void mgpu_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
}

void ds4_mgpu_config_defaults(ds4_mgpu_config *cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->version = 1;
    cfg->placement_auto = 1;
    cfg->safety_margin_gb = 1.0;
    cfg->fast_link_gbps = 10.0;
    cfg->n_devices = 0;
    cfg->loaded = 0;
}

static const char *mgpu_skip_ws(const char *p) {
    while (p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static int mgpu_expect(const char **p, char c) {
    const char *q = mgpu_skip_ws(*p);
    if (*q != c) return -1;
    *p = q + 1;
    return 0;
}

static int mgpu_parse_string(const char **p, char *out, size_t outlen) {
    const char *q = mgpu_skip_ws(*p);
    if (*q != '"') return -1;
    q++;
    size_t n = 0;
    while (*q && *q != '"') {
        char c = *q++;
        if (c == '\\' && *q) c = *q++;
        if (n + 1 < outlen) out[n++] = c;
    }
    if (*q != '"') return -1;
    if (outlen) out[n] = '\0';
    *p = q + 1;
    return 0;
}

static int mgpu_parse_number(const char **p, double *out) {
    const char *q = mgpu_skip_ws(*p);
    char *end = NULL;
    double v = strtod(q, &end);
    if (end == q) return -1;
    *out = v;
    *p = end;
    return 0;
}

static int mgpu_parse_int(const char **p, int *out) {
    double v = 0.0;
    if (mgpu_parse_number(p, &v) != 0) return -1;
    *out = (int)v;
    return 0;
}

static int mgpu_parse_bool(const char **p, int *out) {
    const char *q = mgpu_skip_ws(*p);
    if (strncmp(q, "true", 4) == 0) { *out = 1; *p = q + 4; return 0; }
    if (strncmp(q, "false", 5) == 0) { *out = 0; *p = q + 5; return 0; }
    return -1;
}

static int mgpu_skip_value(const char **p);

static int mgpu_skip_container(const char **p, char open, char close) {
    const char *q = mgpu_skip_ws(*p);
    if (*q != open) return -1;
    q++;
    int depth = 1;
    while (*q && depth > 0) {
        if (*q == '"') {
            q++;
            while (*q && *q != '"') {
                if (*q == '\\' && q[1]) q++;
                q++;
            }
            if (*q != '"') return -1;
            q++;
            continue;
        }
        if (*q == open) depth++;
        else if (*q == close) depth--;
        if (depth == 0) { q++; break; }
        q++;
    }
    if (depth != 0) return -1;
    *p = q;
    return 0;
}

static int mgpu_skip_value(const char **p) {
    const char *q = mgpu_skip_ws(*p);
    if (*q == '{') return mgpu_skip_container(p, '{', '}');
    if (*q == '[') return mgpu_skip_container(p, '[', ']');
    if (*q == '"') {
        char tmp[2];
        return mgpu_parse_string(p, tmp, sizeof(tmp));
    }
    if (strncmp(q, "true", 4) == 0 || strncmp(q, "false", 5) == 0 ||
        strncmp(q, "null", 4) == 0) {
        while (*q && *q != ',' && *q != '}' && *q != ']' &&
               *q != ' ' && *q != '\n' && *q != '\t' && *q != '\r') q++;
        *p = q;
        return 0;
    }
    double d = 0.0;
    return mgpu_parse_number(p, &d);
}

static int mgpu_parse_layers(const char **p, ds4_mgpu_device_cfg *dev) {
    const char *q = mgpu_skip_ws(*p);
    if (*q == '"') {
        char s[16];
        if (mgpu_parse_string(p, s, sizeof(s)) != 0) return -1;
        if (strcmp(s, "auto") != 0) return -1;
        dev->layer_start = DS4_MGPU_AUTO;
        dev->layer_end = DS4_MGPU_AUTO;
        return 0;
    }
    if (mgpu_expect(p, '[') != 0) return -1;
    if (mgpu_parse_int(p, &dev->layer_start) != 0) return -1;
    if (mgpu_expect(p, ',') != 0) return -1;
    q = mgpu_skip_ws(*p);
    if (*q == '"') {
        char s[16];
        if (mgpu_parse_string(p, s, sizeof(s)) != 0) return -1;
        if (strcmp(s, "output") != 0) return -1;
        dev->layer_end = DS4_MGPU_OUTPUT;
    } else {
        if (mgpu_parse_int(p, &dev->layer_end) != 0) return -1;
    }
    if (mgpu_expect(p, ']') != 0) return -1;
    return 0;
}

static void mgpu_device_defaults(ds4_mgpu_device_cfg *dev) {
    memset(dev, 0, sizeof(*dev));
    dev->index = DS4_MGPU_AUTO;
    dev->layer_start = DS4_MGPU_AUTO;
    dev->layer_end = DS4_MGPU_AUTO;
    dev->mode = DS4_MGPU_AUTO;
    dev->expert_cache_gb = -1.0;
    dev->pin_experts = DS4_MGPU_AUTO;
    dev->pin_tokens = DS4_MGPU_AUTO;
}

static int mgpu_parse_device_obj(const char **p, ds4_mgpu_device_cfg *dev) {
    mgpu_device_defaults(dev);
    if (mgpu_expect(p, '{') != 0) return -1;
    const char *q = mgpu_skip_ws(*p);
    if (*q == '}') { *p = q + 1; return 0; }
    for (;;) {
        char key[32];
        if (mgpu_parse_string(p, key, sizeof(key)) != 0) return -1;
        if (mgpu_expect(p, ':') != 0) return -1;
        if (strcmp(key, "index") == 0) {
            if (mgpu_parse_int(p, &dev->index) != 0) return -1;
        } else if (strcmp(key, "layers") == 0) {
            if (mgpu_parse_layers(p, dev) != 0) return -1;
        } else if (strcmp(key, "mode") == 0) {
            char s[16];
            if (mgpu_parse_string(p, s, sizeof(s)) != 0) return -1;
            if (strcmp(s, "static") == 0) dev->mode = 0;
            else if (strcmp(s, "dynamic") == 0) dev->mode = 1;
            else return -1;
        } else if (strcmp(key, "expert_cache_gb") == 0) {
            if (mgpu_parse_number(p, &dev->expert_cache_gb) != 0) return -1;
        } else if (strcmp(key, "pin_experts") == 0) {
            int b = 0;
            if (mgpu_parse_bool(p, &b) != 0) return -1;
            dev->pin_experts = b;
        } else if (strcmp(key, "pin_tokens") == 0) {
            if (mgpu_parse_int(p, &dev->pin_tokens) != 0) return -1;
        } else {
            if (mgpu_skip_value(p) != 0) return -1;
        }
        q = mgpu_skip_ws(*p);
        if (*q == ',') { *p = q + 1; continue; }
        if (*q == '}') { *p = q + 1; return 0; }
        return -1;
    }
}

int ds4_mgpu_config_load(const char *path, ds4_mgpu_config *out,
                         char *err, size_t errlen) {
    if (!path || !out) { mgpu_err(err, errlen, "invalid argument"); return 1; }
    FILE *f = fopen(path, "rb");
    if (!f) { mgpu_err(err, errlen, "cannot open config file"); return 1; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); mgpu_err(err, errlen, "seek failed"); return 1; }
    long sz = ftell(f);
    if (sz <= 0 || sz > (1L << 20)) {
        fclose(f);
        mgpu_err(err, errlen, "config file size out of range");
        return 1;
    }
    rewind(f);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); mgpu_err(err, errlen, "out of memory"); return 1; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';

    ds4_mgpu_config_defaults(out);
    out->loaded = 1;

    const char *p = buf;
    if (mgpu_expect(&p, '{') != 0) { free(buf); mgpu_err(err, errlen, "expected '{'"); return 1; }
    const char *q = mgpu_skip_ws(p);
    if (*q != '}') {
        for (;;) {
            char key[32];
            if (mgpu_parse_string(&p, key, sizeof(key)) != 0) { free(buf); mgpu_err(err, errlen, "expected key"); return 1; }
            if (mgpu_expect(&p, ':') != 0) { free(buf); mgpu_err(err, errlen, "expected ':'"); return 1; }
            if (strcmp(key, "version") == 0) {
                if (mgpu_parse_int(&p, &out->version) != 0) { free(buf); mgpu_err(err, errlen, "bad version"); return 1; }
            } else if (strcmp(key, "placement") == 0) {
                char s[16];
                if (mgpu_parse_string(&p, s, sizeof(s)) != 0) { free(buf); mgpu_err(err, errlen, "bad placement"); return 1; }
                if (strcmp(s, "auto") == 0) out->placement_auto = 1;
                else if (strcmp(s, "explicit") == 0) out->placement_auto = 0;
                else { free(buf); mgpu_err(err, errlen, "placement must be auto|explicit"); return 1; }
            } else if (strcmp(key, "safety_margin_gb") == 0) {
                if (mgpu_parse_number(&p, &out->safety_margin_gb) != 0) { free(buf); mgpu_err(err, errlen, "bad safety_margin_gb"); return 1; }
            } else if (strcmp(key, "fast_link_gbps") == 0) {
                if (mgpu_parse_number(&p, &out->fast_link_gbps) != 0) { free(buf); mgpu_err(err, errlen, "bad fast_link_gbps"); return 1; }
            } else if (strcmp(key, "devices") == 0) {
                if (mgpu_expect(&p, '[') != 0) { free(buf); mgpu_err(err, errlen, "devices must be an array"); return 1; }
                const char *qq = mgpu_skip_ws(p);
                if (*qq != ']') {
                    for (;;) {
                        if (out->n_devices >= DS4_LAYER_PACK_MAX_GPUS) { free(buf); mgpu_err(err, errlen, "too many devices"); return 1; }
                        if (mgpu_parse_device_obj(&p, &out->devices[out->n_devices]) != 0) { free(buf); mgpu_err(err, errlen, "bad device object"); return 1; }
                        out->n_devices++;
                        qq = mgpu_skip_ws(p);
                        if (*qq == ',') { p = qq + 1; continue; }
                        break;
                    }
                }
                if (mgpu_expect(&p, ']') != 0) { free(buf); mgpu_err(err, errlen, "expected ']'"); return 1; }
            } else {
                if (mgpu_skip_value(&p) != 0) { free(buf); mgpu_err(err, errlen, "bad value"); return 1; }
            }
            q = mgpu_skip_ws(p);
            if (*q == ',') { p = q + 1; continue; }
            break;
        }
    }
    if (mgpu_expect(&p, '}') != 0) { free(buf); mgpu_err(err, errlen, "expected '}'"); return 1; }
    free(buf);
    return 0;
}

void ds4_mgpu_config_dump(FILE *out, const ds4_mgpu_config *cfg) {
    if (!out || !cfg) return;
    fprintf(out, "mgpu-config: v%d placement=%s n_devices=%d "
                 "safety=%.2fGiB fast_link=%.1fGB/s\n",
            cfg->version, cfg->placement_auto ? "auto" : "explicit",
            cfg->n_devices, cfg->safety_margin_gb, cfg->fast_link_gbps);
    for (int i = 0; i < cfg->n_devices; i++) {
        const ds4_mgpu_device_cfg *d = &cfg->devices[i];
        char layers[32];
        if (d->layer_start == DS4_MGPU_AUTO) {
            snprintf(layers, sizeof(layers), "auto");
        } else if (d->layer_end == DS4_MGPU_OUTPUT) {
            snprintf(layers, sizeof(layers), "%d-output", d->layer_start);
        } else {
            snprintf(layers, sizeof(layers), "%d-%d", d->layer_start, d->layer_end);
        }
        fprintf(out, "  device[%d] index=%d layers=%s mode=%s cache=%.1fGB "
                     "pin=%d tokens=%d\n",
                i, d->index, layers,
                d->mode == DS4_MGPU_AUTO ? "auto" : (d->mode ? "dynamic" : "static"),
                d->expert_cache_gb, d->pin_experts, d->pin_tokens);
    }
}

int ds4_mgpu_plan(const ds4_mgpu_config *cfg,
                  const ds4_mgpu_tier *tiers, int n_tiers,
                  const ds4_mgpu_layer_bytes *layer_bytes, int n_layers,
                  size_t embedding_bytes, size_t output_bytes,
                  int *placement_out, char *err, size_t errlen) {
    if (!tiers || n_tiers <= 0 || n_tiers > DS4_LAYER_PACK_MAX_GPUS ||
        !layer_bytes || n_layers <= 0 || !placement_out) {
        mgpu_err(err, errlen, "planner: invalid argument");
        return 1;
    }
    const int n_entries = n_layers + 2;
    for (int i = 0; i < n_entries; i++) placement_out[i] = DS4_LAYER_PACK_CPU;

    const int explicit_ranges =
        cfg && cfg->loaded && !cfg->placement_auto && cfg->n_devices > 0;
    if (explicit_ranges) {
        int covered = 0;
        for (int d = 0; d < cfg->n_devices && d < n_tiers; d++) {
            const ds4_mgpu_device_cfg *dc = &cfg->devices[d];
            if (dc->layer_start == DS4_MGPU_AUTO) continue;
            int s = dc->layer_start;
            int e = (dc->layer_end == DS4_MGPU_OUTPUT) ? n_layers - 1 : dc->layer_end;
            if (s < 0 || e < 0 || s > e || e >= n_layers) {
                mgpu_err(err, errlen, "planner: invalid explicit layer range");
                return 1;
            }
            for (int il = s; il <= e; il++) {
                if (placement_out[il + 1] != DS4_LAYER_PACK_CPU) {
                    mgpu_err(err, errlen, "planner: overlapping explicit layer ranges");
                    return 1;
                }
                placement_out[il + 1] = d;
                covered++;
            }
            if (dc->layer_end == DS4_MGPU_OUTPUT)
                placement_out[n_layers + 1] = d;
        }
        if (covered != n_layers) {
            mgpu_err(err, errlen, "planner: explicit ranges do not cover all layers");
            return 1;
        }
        placement_out[0] = placement_out[1];
        if (placement_out[n_layers + 1] == DS4_LAYER_PACK_CPU)
            placement_out[n_layers + 1] = placement_out[n_layers];
        return 0;
    }

    /* Auto: greedy monotonic fill.  Slow (static) tiers come first in the
     * caller-provided order, so they take the first contiguous layers; the
     * dynamic tier only pays for dense weights, so it absorbs the rest. */
    size_t rem[DS4_LAYER_PACK_MAX_GPUS];
    for (int t = 0; t < n_tiers; t++) {
        rem[t] = tiers[t].budget_bytes;
        if (tiers[t].mode == 1 && tiers[t].expert_cache_bytes > 0) {
            rem[t] = rem[t] > tiers[t].expert_cache_bytes
                   ? rem[t] - tiers[t].expert_cache_bytes : 0;
        }
    }
    /* The output head is resident (never streamed): reserve it on the last
     * tier up front so the layer fill cannot consume its space. */
    const int head_tier = n_tiers - 1;
    if (output_bytes > rem[head_tier]) {
        mgpu_err(err, errlen, "planner: output head does not fit the last tier");
        return 1;
    }
    rem[head_tier] -= output_bytes;

    if (embedding_bytes > rem[0]) {
        mgpu_err(err, errlen, "planner: embedding does not fit tier 0");
        return 1;
    }
    placement_out[0] = 0;
    rem[0] -= embedding_bytes;

    int t = 0;
    for (int il = 0; il < n_layers; il++) {
        const size_t cost = (tiers[t].mode == 0)
            ? layer_bytes[il].dense_bytes + layer_bytes[il].expert_bytes
            : layer_bytes[il].dense_bytes;
        while (t < n_tiers && cost > rem[t]) t++;
        if (t >= n_tiers) {
            placement_out[il + 1] = DS4_LAYER_PACK_CPU;
            continue;
        }
        placement_out[il + 1] = t;
        rem[t] -= cost;
    }
    placement_out[n_layers + 1] = head_tier;
    return 0;
}
