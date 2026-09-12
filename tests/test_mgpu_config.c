/* test_mgpu_config — M2 unit tests: JSON config loader + asymmetric planner.
 * Pure C99, no GPU.  See vulkan/SPECS_MGPU.md. */
#include "../ds4_layer_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
} while (0)

static int write_tmp(const char *name, const char *text) {
    char path[256];
    snprintf(path, sizeof(path), "/tmp/%s", name);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(text, 1, strlen(text), f);
    fclose(f);
    return 0;
}

static void test_defaults(void) {
    ds4_mgpu_config cfg;
    ds4_mgpu_config_defaults(&cfg);
    CHECK(cfg.version == 1, "version");
    CHECK(cfg.placement_auto == 1, "auto");
    CHECK(cfg.fast_link_gbps > 9.0 && cfg.fast_link_gbps < 11.0, "fast link");
    CHECK(cfg.n_devices == 0, "n_devices");
}

static void test_parse_explicit(void) {
    const char *json =
        "{\n"
        "  \"version\": 1,\n"
        "  \"placement\": \"explicit\",\n"
        "  \"safety_margin_gb\": 1.5,\n"
        "  \"fast_link_gbps\": 12.0,\n"
        "  \"devices\": [\n"
        "    {\"index\": 0, \"layers\": [0, 7], \"mode\": \"static\",\n"
        "     \"expert_cache_gb\": 10.0, \"pin_experts\": true, \"pin_tokens\": 0},\n"
        "    {\"index\": 3, \"layers\": [8, \"output\"], \"mode\": \"dynamic\",\n"
        "     \"expert_cache_gb\": 8.0, \"pin_experts\": false, \"pin_tokens\": 3}\n"
        "  ]\n"
        "}\n";
    CHECK(write_tmp("ds4_test_mgpu.json", json) == 0, "write tmp");
    ds4_mgpu_config cfg;
    char err[128] = {0};
    int rc = ds4_mgpu_config_load("/tmp/ds4_test_mgpu.json", &cfg, err, sizeof(err));
    CHECK(rc == 0, "load rc");
    CHECK(cfg.loaded == 1, "loaded");
    CHECK(cfg.placement_auto == 0, "explicit");
    CHECK(cfg.n_devices == 2, "n_devices");
    CHECK(cfg.devices[0].index == 0 && cfg.devices[0].layer_start == 0 &&
          cfg.devices[0].layer_end == 7 && cfg.devices[0].mode == 0 &&
          cfg.devices[0].pin_experts == 1, "device0 fields");
    CHECK(cfg.devices[1].index == 3 && cfg.devices[1].layer_end == DS4_MGPU_OUTPUT &&
          cfg.devices[1].mode == 1 && cfg.devices[1].pin_tokens == 3, "device1 fields");
    CHECK(cfg.devices[1].pin_experts == 0, "device1 pin_experts");
    ds4_mgpu_config_dump(stdout, &cfg);
}

static void test_parse_errors(void) {
    ds4_mgpu_config cfg;
    char err[128];
    CHECK(ds4_mgpu_config_load("/tmp/does-not-exist-ds4.json", &cfg, err, sizeof(err)) != 0,
          "missing file errors");
    CHECK(write_tmp("ds4_test_bad.json", "{ \"devices\": [ { ] }") == 0, "write bad");
    CHECK(ds4_mgpu_config_load("/tmp/ds4_test_bad.json", &cfg, err, sizeof(err)) != 0,
          "bad json errors");
}

static void test_auto_plan(void) {
    ds4_mgpu_tier tiers[4];
    memset(tiers, 0, sizeof(tiers));
    for (int i = 0; i < 3; i++) {
        tiers[i].physical_index = i;
        tiers[i].is_fast = 0;
        tiers[i].mode = 0;              /* static */
        tiers[i].budget_bytes = 200;    /* 50 emb + 1 layer (100) + head 50 */
        tiers[i].pin_experts = 1;
    }
    tiers[3].physical_index = 3;
    tiers[3].is_fast = 1;
    tiers[3].mode = 1;                  /* dynamic */
    tiers[3].budget_bytes = 1000;
    tiers[3].expert_cache_bytes = 100;
    tiers[3].pin_tokens = 3;

    ds4_mgpu_layer_bytes lb[8];
    for (int i = 0; i < 8; i++) { lb[i].dense_bytes = 10; lb[i].expert_bytes = 90; }

    int placement[10];
    char err[128] = {0};
    int rc = ds4_mgpu_plan(NULL, tiers, 4, lb, 8, 50, 50, placement, err, sizeof(err));
    CHECK(rc == 0, "auto plan rc");
    CHECK(placement[0] == 0, "embedding on tier0");
    CHECK(placement[1] == 0, "layer0 on slow0");
    CHECK(placement[2] >= 1, "layer1 moved off slow0");
    CHECK(placement[9] == 3, "output head on fast tier");
    /* monotonic non-decreasing over layers (contiguity) */
    for (int i = 1; i <= 8; i++) {
        CHECK(placement[i] >= placement[i - 1], "monotonic placement");
    }
    CHECK(placement[8] == 3, "last layer on fast tier");
}

static void test_explicit_plan(void) {
    const char *json =
        "{\"placement\": \"explicit\", \"devices\": ["
        " {\"index\": 0, \"layers\": [0, 3], \"mode\": \"static\", \"pin_experts\": true},"
        " {\"index\": 3, \"layers\": [4, \"output\"], \"mode\": \"dynamic\", \"pin_tokens\": 3}"
        "]}";
    CHECK(write_tmp("ds4_test_mgpu2.json", json) == 0, "write tmp2");
    ds4_mgpu_config cfg;
    char err[128] = {0};
    CHECK(ds4_mgpu_config_load("/tmp/ds4_test_mgpu2.json", &cfg, err, sizeof(err)) == 0,
          "load2");
    ds4_mgpu_tier tiers[2];
    memset(tiers, 0, sizeof(tiers));
    tiers[0].mode = 0; tiers[0].budget_bytes = (size_t)1 << 40;
    tiers[1].mode = 1; tiers[1].budget_bytes = (size_t)1 << 40;
    ds4_mgpu_layer_bytes lb[8];
    for (int i = 0; i < 8; i++) { lb[i].dense_bytes = 10; lb[i].expert_bytes = 90; }
    int placement[10];
    int rc = ds4_mgpu_plan(&cfg, tiers, 2, lb, 8, 50, 50, placement, err, sizeof(err));
    CHECK(rc == 0, "explicit plan rc");
    CHECK(placement[0] == 0 && placement[4] == 0, "explicit first block");
    CHECK(placement[5] == 1 && placement[8] == 1, "explicit second block");
    CHECK(placement[9] == 1, "explicit output head");
}

int main(void) {
    test_defaults();
    test_parse_explicit();
    test_parse_errors();
    test_auto_plan();
    test_explicit_plan();
    if (g_fail) {
        fprintf(stderr, "=== FAILED (%d) ===\n", g_fail);
        return 1;
    }
    printf("=== PASSED (0 failures) ===\n");
    return 0;
}
