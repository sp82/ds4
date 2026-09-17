# Progress — Backend Vulkan per DS4

> **FASE 8e — cache pesi persistente per-device FATTA (2026-09-14)**: il
> multi-GPU nativo non ri-staggia più i pesi a ogni token. Nuova cache
> **device-local persistente** in `vulkan/ds4_vulkan.c` (`g_weight_cache`,
> `vulkan_model_window_for` la consulta prima dei window transitori,
> `set_span_windows` filtra gli span già residenti), `ds4_gpu_device_cache_tensors`
> reale (prima stub muto), engine con `g_tier_weights_resident[]` che salta lo
> staging per-layer sui tier statici. Fix di un **bug latente** in
> `ds4_vulkan_init_multi` (`g_vk_ctx_active` restava -1 → i tier 2/3 salvavano lo
> stato del tier 0). Sul server i 3 tier statici passano da ~1500 ms/layer a
> **~50-150 ms/layer**; il tier dinamico resta **disk-bound** (1.86 GiB di esperti
> per layer riletti dal disco, 31 GiB RAM / 86 GiB modello) → serve la **pool
> esperti per-tier** (prossimo lavoro). Dettagli: `SPECS_MGPU.md` §12,
> `SPECS_CORE_CHANGES.md` §2e.
>
> **TODO concordati (2026-09-14, dettaglio in `SPECS_MGPU.md` §15)**: (1) soglia
> fast/slow 10→**5 GB/s** FATTA; (2) `--gpu-config-dump` (emit JSON dal default
> auto); (3) `--gpu-config FILE` (load JSON per ottimizzazione manuale);
> (4) bilanciamento auto con tutte le GPU dynamic (oggi il greedy mette tutto su
> tier 0); (5) seed hotlist per-tier; (6) **fix correttezza multi-tier**
> **FATTO (2026-09-15)**: il decode multi-tier lasciava command scope aperti ai
> confini di tier; ora `ds4_vulkan_set_current_device` li sottomette prima dello
> switch (+2 fix difensivi). Multi == single-GPU (hc-sum byte-identici), 0 TIER
> MISMATCH. Dettagli: `SPECS_CORE_CHANGES.md` §6. Idea async "static-first /
> esperti hot in background":
> realizzabile con l'architettura attuale (cache per-tier parallela + worker
> async pool + priorità hotness), ma dopo il fix di correttezza.
>
> **Store esperti in parallelo FATTO (2026-09-17)**: `vulkan_pool_store_batch`
> ora carica gli slab (gate|up|down) degli esperti non-residenti con un piccolo
> **pool di thread persistente** (lazy, `DS4_VULKAN_POOL_COPY_THREADS`, default 8)
> invece del loop seriale di memcpy. Il guadagno è sui **cold read**: più
> page-fault concorrenti → più queue depth sull'NVMe → più banda (il tier
> dinamico è disk-bound, ~1.86 GiB esperti/layer riletti). Le copie vanno a
> offset per-esperto disgiunti (nessuna lock sul data path); il GPU-copy resta
> il singolo `vkCmdCopyBuffer` batched. **Bug fisso durante il lavoro**: la
> main thread correva davanti ai thread appena creati (il contatore `active`
> è 0 sia "prima che partano" sia "quando hanno finito") → ora si aspetta
> `engaged>0` prima di attendere `active==0`. Smoke 396/396, multi-run e
> `DS4_VULKAN_POOL_COPY_THREADS=2/4/16` tutti verdi.
>
> **Findings e misure (2026-09-17) — server 4× RX 6900 XT, multi-GPU
> `--gpu-vram auto`**:
> - **A/B vecchio(serial) vs nuovo(pool 8 thread)** (stesso modello, n=30):
>   banda letture esperti **896→1939 MiB/s (×2.2)**, generation **2.21→2.71 t/s**,
>   prefill 1.06→1.91 t/s. Su single-GPU (senza `--gpu-vram`): 1409→3509 MiB/s,
>   gen 1.70→2.56 t/s.
> - **Copia GPU**: è **una sola `vkCmdCopyBuffer` batch** (n_missing×3 regioni,
>   ds4_vulkan.c). Misurato il fence-wait (`gpu_fence`, ora nel report):
>   **3.5 ms** su 5468 ms di letture (0.06%) → la copia GPU è trascurabile,
>   NON è il collo di bottiglia.
> - **Per-expert `vkCmdCopyBuffer`** (`DS4_VULKAN_POOL_COPY_PER_EXPERT`):
>   identico al batch (gen 2.71 vs 2.73, gpu_fence 3.6 vs 3.5 ms); sul path
>   async serializza i fence per-submit → leggermente più lento. Toggle
>   env-gated, default batch.
> - **16 thread**: peggiora (524 MiB/s vs 1939, gen 1.57 vs 2.71). Il server ha
>   12 CPU logiche; i thread sono I/O-bound ma troppi thread concorrenti
>   provocano contention/disco oltre il sweet spot. **Default resta 8.**
> - **`--ssd-part` (Optane/hotlist)**: c'è ed è cablato (sparse mirror, stesso
>   offset); compone col pool parallelo (`vulkan_expert_src` sceglie part_map vs
>   model_map per-esperto). La **split 3/5 (Optane/Samsung) NON è fattibile come
>   ratio fissa** — il part è selezionato dalla hotlist/popolarità + budget, e
>   un esperto non ospitato non è leggibile da Optane. Lasciato com'era.
> - **Hit/miss pool**: **NON è un bug**. n=30 → miss ~52% (cold), n=120 → miss
>   ~21% (warm), pool 97% residente (1648/1702), reload basso (~38). Il pool
>   scalda bene. Nota: in multi-GPU auto il budget pool NON viene dal count
>   `--ssd-streaming-cache-experts` (64 vs 512 → identico) ma dal budget byte
>   del tier dinamico (`dev_expert_cache_bytes = budget*4/10` ≈ 6.2 GiB →
>   ~11.3 GiB esperti). Da capire se l'overriding del count è voluto.
>
> **DA APPROFONDIRE (2026-09-18)**:
> - **Perché il ceiling dei t/s resta ~2.8** (non supera i 3) anche con la pool
>   calda (79% hit). Ipotesi: il **tier dinamico** (GPU3, layers 20-42 + head)
>   è disk-bound e concentra 23 layer + output head su una sola card → il
>   tempo di decode è il suo. Da profilare: **quota tempo per tier**, tempo GPU
>   vs tempo lettura disco per layer (con `DS4_VULKAN_DEBUG_POOL_TIME` +
>   `--vulkan-stats`), e verificare se serve ribilanciare (M5 pool per-tier) o
>   se il limite è la banda disco di GPU3.
> - **Overriding del count pool in multi-GPU auto** (vedi sopra): capire se
>   `--ssd-streaming-cache-experts` deve cappare la pool o se il budget byte del
>   tier è quello giusto.
> - **Server (`ds4-server`)**: fix fatto al guard del backend (14651) per usare
>   multi-GPU Vulkan con `--gpu-vram`; path separato e meno testato → decidere
>   se tenerlo o revertirlo.
> - **KV cache con contesti grandi**: i test finora sono con `-c 256` e KV in
>   VRAM (default, `DS4_VULKAN_KV_IN_RAM` non impostata). Con `-c` grandi la KV
>   (anche se compressa) cresce: **verificare se `DS4_VULKAN_KV_IN_RAM=1`
>   (KV in RAM, libera VRAM per la pool esperti) sposta qualcosa** con
>   `--ssd-streaming` multi-GPU, e misurare la dimensione KV reale a `-c` alto
>   (formula in `SPECS_KVCACHE.md`). Pronostico: impatto minimo (KV compressa),
>   ma da confermare a contesto grande.

Stato: **FASE 7 IN CORSO — 2026-09-09**. Fasi 0-6 chiuse (de-serializzazione
v1: tabella on-device esperto→slot; pool calda; store batch; mirror VRAM).
Tre target Fase 7 fatti e stabili: `matmul_q8_0_preq` v2, MoE IQ2/Q2K v2,
**overlap store esperti ↔ compute GPU** (worker async attivo, gate ON di
default, output identico al sync). **Flush early-submit FATTO** (double-buffer
g_cmd[2] + fence per-CB, begin senza drain, flush end+submit+switch ON di
default): la scope shared+attention viene sottomessa mentre il worker carica
gli esperti. **Misura onesta**: A/B interleaved ON vs sync (8+ run) → mediana
IDENTICA (~1.5 t/s): il decode è bound da max(store SSD esperti ~8-13 ms/layer,
GPU+PCIe ~7-15 ms/layer), l'overlap dei due è già al massimo in entrambi i
path; con page-cache caldo il sistema fa 2.6-4.7 t/s (varianza SSD, non
differenza di codice). **Validazione multi-GPU avviata (2026-09-01)**: il
pipeline distribuito 4-processi (SPEC §8d Opzione C) ESCE su 1 GPU condivisa
(prefill+generation) dopo i fix della static slice map e della pool per-slice;
resta un **race di correttezza** (output non-deterministico vs single-GPU) da
debbuggare — il debug definitivo richiede le 4 GPU (x1 da ricollegare, §1b).
**Nota (2026-09-03): il debug del race distribuito passa alla Fase 8** (vedi
§1b caveat 6): è la validazione dell'Opzione C (multi-GPU), non un blocco della
  Fase 7. **Hotness (leva immediata) FATTO (2026-09-03)**: priorità hotlist +
  eviction LFU+LRU implementati (item 6, §1). **Telemetria `--vulkan-stats`
  FATTA (2026-09-08, validata e2e 2026-09-09)**: contatori `std::atomic` per
  layer per fase nella pool + snapshot C → motore → blocco di report multi-riga
  a fine risposta (CLI e ds4-server per-request, item 7 §1).
  **ALLINEAMENTO A UPSTREAM/MAIN FATTO (2026-09-09)**: rebase su `6289c51`
  (197 commit upstream: Vision-Exp/GLM 5.3/iris/agent rework), fork main in FF,
  nuova superficie `ds4_gpu_*` stubbata, `ds4_image.o` nel link vulkan
  (dettagli in AGENTS.md). **GAP QUANT DEEPSEEK CHIUSO (2026-09-10)**: MoE
  routed Q4_K (12,12), attention-output Q4_K, MoE MXFP4 (39,39) e dense
  Q4_K/Q4_0 (item 8), smoke 450/450 — copre `ds4f-q4`/`ds4f-q2-q4`/Pro Q4,
  MTP-Q4K, `ds4f-mxfp4` e i modelli GLM densi. Prossimi passi della Fase 7 in
  ordine:
  (1) **autotuner a init** (SPECS_AUTOTUNE), (2) misura e2e robusta (anche per
  validare l'effetto hotness sul server e il nuovo blocco --vulkan-stats).
  **VALIDAZIONE NVIDIA (RTX 5080 16 GiB, 2026-09-11) — FATTA**: il backend gira
  su NVIDIA (driver 595.84, CUDA 13.2) con `--ssd-streaming` dopo i fix del
  **device-lost (Xid 109)**; MXFP4 e Q2 completano prefill+generazione (0 Xid),
  smoke **450/450** su NVIDIA, RX 6900 XT e iGPU 780M. Causa radice: lifetime
  dello scratch buffer (`a0875f0`). Nuovi kernel MXFP4 v2/grouping/LUT e Q8
  preq v3; misure in §2 e dettagli nell'item 9.

---

## 1. Fase 7 — cosa è fatto e cosa manca

### Fatto ✅

1. **`matmul_q8_0_preq` v2** (0.266 → **0.118-0.124 ms**, 2.2-4×): 256 thread
   attivi (2 thread/blocco, halves da 16), letture vettorizzate sul blocco
   34-byte 2-mod-4 (blocchi pari: 5 word allineate + repack a shift; dispari:
   4 word dirette). Variante in coda all'enum + `DS4_VULKAN_FORCE_VARIANT`
   (ponte autotuner). Parity v1-v2 esatta, smoke 396/396.
2. **MoE IQ2/Q2K v2** (scope MoE 2.47 → **1.03 ms mediana**, -58%): dequant
   vettorizzato (2-3 word allineate per sub-blocco IQ2; Q2K: d/dmin+scale+qs
   da 3 word), 256 lane attive (2 thread/sub-blocco IQ2, 4 per Q2K).
3. **Overlap store↔compute GPU** (il collo di bottiglia reale): il worker
   async del motore (già usato da Metal/CUDA) è stato attivato su Vulkan con
   sync readback reale e store async. **3 crash trovati e fixati** (vedi §4).
   Verifica: output IDENTICO al path sync (temp 0), smoke 396/396, e2e stabile
   **1.48-1.69 t/s** (gain ~5%: il flush early-submit è gated e i drain
   serializzano → il gain pieno richiede il punto 1 sotto).
4. **Flush early-submit** (il "gain pieno" dell'overlap): `ds4_gpu_flush_commands`
   fa ora davvero end+submit+reopen — **command buffer double-buffered**
   (`g_cmd[2]`, fence per-CB `g_cmd_fence[2]`, `g_cmd_i`), nessun drain alla
   riapertura (l'acquire attende solo la fence del CB riusato; la FIFO della
   coda ordina le scope). La signal sottomette la scope router e marca il CB
   con `g_readback_fence` (l'acquire la aspetta ma non la resetta: il worker
   la attende pure); il motore riapre una scope per lo shared (ds4.c,
   `#if DS4_VULKAN_BUILD`, solo col worker attivo) e il flush la sottomette
   subito → la GPU esegue shared+attention mentre il worker fa il memcpy
   degli esperti; il retry sync (fail del worker) chiude/riapre la scope.
   Gate: `DS4_VULKAN_FLUSH_END_BEGIN=0` = flush no-op legacy (bisect).
   Fix collegati: deadlock `DS4_VULKAN_DEBUG_SUBMIT` (fence resettata ma
   lasciata nello slot → acquire attende una fence mai segnalata; il path
   debug ora distrugge la fence), **descriptor-pool HWM drain** (con i drain
   per-layer rimossi, la prefill senza readback esauriva le 8192 set → drain
   raro a 6144 set in `vulkan_cb_acquire`). Verifica: output IDENTICO al sync
   (diff temp 0), smoke 396/396, e2e -n 60 pulito.
5. Strumenti: `kbench` misura v1+v2 + parity (1%), telemetrie pool-hit,
   build parallela `make -j$(nproc)` (~25 s). Fix pre-esistente: la regola
   Makefile dei vkbench header (2 target, 1 recipe) con `-j` eseguiva la
   recipe 2× in concorrenza → race su `/tmp/vkbench_spv_tmp.bin` → build
   flaky; ora target raggruppato `&:` (GNU make 4.3).
6. **Hotness esperti (leva immediata, SPECS_HOTNESS) FATTO**: hotness per
   (layer, esperto) dentro la pool (`route_hotness[]` per-layer, sopravvive
   all'eviction), **priorità della hotlist usate al seed** (non più scartate in
   `ds4_vulkan_compat.c`), **nota dinamica +1** a ogni selezione routata
   (seed selected/async/batch), **decay** ogni 16 seed routati (~token),
   **eviction LFU** (hotness minima) con tiebreak LRU su `slot_age` in
   `vulkan_pool_slot_reserve_only`. `reset_route_hotness` ora reale (chiamato
   dal motore a inizio prefill/decode). Zero modifiche al motore: le API
   `seed_experts`/`reset_route_hotness` erano già wired per entrambi i backend.
    Verifica: build 0 warning, smoke 396/396, gate `-fsyntax-only` puliti, ABI
    `nm` 0 `_Z`. Effetto e2e da misurare sul server (atteso: meno store dei
    layer con routing concentrato).
7. **Telemetria pool esperti `--vulkan-stats` FATTA (2026-09-08)**: flag
   opzionale (CLI + ds4-server) che stampa su stderr un blocco stats dopo ogni
   risposta/generazione: per fase (decode/prefill/hotlist) hit/miss % e MiB
   caricati del seed pool, più eviction, reload (thrash), occupazione pool e
   stall del `device_wait`. Contatori `std::atomic<uint64_t>` per layer in
   `g_pool_tel[]` (racesafe: un thread per layer alla volta) + timing wait in
   `vulkan_pool_seed_remap`/`ds4_vulkan_pool_commit_pending` (tutto dentro
   `vulkan/ds4_vulkan.c`); snapshot C `ds4_vulkan_telemetry_snapshot()` →
   compat wrapper → `ds4_vulkan_stats_report()` in ds4.c (baseline
   process-wide con mutex, delta = la richiesta appena chiusa; NULL se pool non
   pronta). Nessun overhead nel path runtime (un fetch-add per slot). Fase 1:
   solo residuance pool — la distinzione SSD-vs-RAM (mincore) è rimandata.
   Modifiche core (gate e impatto) in `SPECS_CORE_CHANGES.md` §2a.
   Verifica: build `make -j$(nproc) vulkan` 0 warning, flag accettato da
   entrambi i binari; e2e sul server (GPU index 0, RX 6900 XT): il blocco esce
   nel path `--temp 0` (argmax, log t/s dell'engine) e in quello sampled
   (`run_sampled_generation`) e REPL — il primo test e2e ha mostrato il blocco
   vuoto: il hook mancava nel branch argmax di `run_generation` (fixato).
   Test ds4-server via HTTP (`POST /v1/chat/completions`, `max_tokens:30`,
   temp 0): hook in `generate_job` emette il blocco nel log del server dopo la
   request (decode 49.3% hit, hotlist 864 miss, pool 946/946, evictions 4492).
8. **Gap quant DeepSeek CHIUSO (2026-09-10)**: MoE Q4_K, attention-output
   Q4_K, MoE MXFP4 e dense Q4_K/Q4_0. Copre i modelli Q4KExperts (`ds4f-q4`,
   `ds4f-q2-q4` layer 37-42, Pro Q4), le varianti con AProj Q4_K (MTP-Q4K),
   MXFP4 (`ds4f-mxfp4`, `ds4f-vision-mxfp4`) e i GLM densi (Q4_K/Q4_0).
   Dettagli in `SPECS_GAP.md` §4 (Fasi 1, 2a, 2b, 3).
   - **MoE Q4_K**: nuovo `vulkan/shaders/moe_q4k.hlsl`
     (`moe_gate_up_mid_q4k`, `moe_down_q4k`) + helper condiviso
     `q4k_sub_dot`/`q4k_scale_min` in `common.hlsl` (blocco 144 B/256, scale
     2-livello `d*sc*nib − dmin*m`, activation f32). `vulkan_routed_moe_launch`
     esteso a `q4_path = (gate==12 && down==12)`.
   - **Attention-output Q4_K**: `attn_output_low_q4k` in `attention.hlsl` +
     `vulkan_attn_output_low_q4k`; entry
     `ds4_gpu_attention_output_low_q4_K_slice_tensor` (decode) e
     `ds4_gpu_attention_output_q4_K_batch_tensor` (prefill); stub rimossi.
   - **MoE MXFP4**: nuovo `vulkan/shaders/moe_mxfp4.hlsl`
     (`moe_gate_up_mid_mxfp4`, `moe_down_mxfp4`) + helper
     `mxfp4_value`/`e8m0_to_f32`/`mxfp4_block_dot` in `common.hlsl` (blocco
     17 B/32, E8M0 + nibble E2M1). `vulkan_routed_moe_launch` esteso a
     `mxfp4_path = (gate==39 && down==39)`.
   - **Dense Q4_K/Q4_0**: nuovo `vulkan/shaders/matmul_quant.hlsl`
     (`matmul_q4k`, `matmul_q4_0`) + helper `vulkan_matmul_quant_dense`;
     `ds4_gpu_matmul_quant_tensor` esteso a type 12 (Q4_K) e 2 (Q4_0),
     activation f32.
   - Pipeline `DS4_VK_PIPE_COUNT` 61 → **68**; 7 tuple in `gen_shaders.py`;
     `moe_q4k.hlsl`/`moe_mxfp4.hlsl`/`matmul_quant.hlsl` nella dependency
     `Makefile`.
   - Verifica: build 5 binari 0 warning; smoke **450/450** (erano 397; +26 check
     Q4_K, +16 MXFP4, +11 dense Q4_K/Q4_0) su AMD Radeon 780M; gate
     `-fsyntax-only` (CUDA e `-DDS4_NO_GPU`) puliti; ABI `nm` 0 `_Z`.
   - **E2e server (RX 6900 XT, RADV NAVI21, 2026-09-10)**: build
     `-march=znver3`, deploy `/tmp`; smoke **450/450** anche sulla GPU target;
     `ds4f-q4` (MoE Q4_K) e `ds4f-mxfp4` (MoE MXFP4) scaricati e girati con
     `--ssd-streaming --ssd-streaming-cache-experts 64 -c 256 -n 30 --temp 0`:
     **exit 0, output coerente, 0 spam "Vulkan unavailable"** (prefill 0.15-0.17
     / gen 0.29-0.30 t/s, cache fredda). Non-regressione `ds4f-q2`: ok.
9. **Validazione NVIDIA RTX 5080 + fix device-lost (2026-09-11)**: il backend
   gira su NVIDIA (driver 595.84, CUDA 13.2) con `--ssd-streaming`; MXFP4 e Q2
   completano prefill+generazione a 16 GiB, **0 Xid**, smoke **450/450** su
   NVIDIA, AMD RX 6900 XT e iGPU Radeon 780M. Catena di fix del device-lost
   (Xid 109), dalla causa superficiale a quella radice:
   - `dd10ee8` **leak di `VkFence`** → `vkCreateFence` falliva con
     `VK_ERROR_OUT_OF_HOST_MEMORY` (-1) dopo migliaia di submit per scope;
   - `e481916` fence **persistente** per command buffer (create/destroy per
     scope → reset+reuse), marker in-flight separato;
   - `b659eba` serializzazione `vkDeviceWaitIdle`+reset pool con i submit (il
     worker non resetta più il descriptor pool);
   - `855ed96` binding flag dei descriptor array + feature
     `descriptorBindingPartiallyBound` (accessi OOB del pool descriptor);
   - `f8d7761` submit serializzati + pool/CB dedicati al worker;
   - `a0875f0` **causa radice**: `vulkan_scratch_a/b` liberava il buffer vecchio
     su grow con `vulkan_device_wait()`, che drena solo il **già sottomesso**,
     mentre i dispatch della **scope ancora aperta** lo referenziavano → GPU su
     memoria libera. Fix: `vulkan_scratch_retire()` = flush della scope aperta
     (`ds4_gpu_flush_commands`) prima del drain e della free.
   Debug: tool `DS4_VULKAN_CHECKPOINTS` (`VK_NV_device_diagnostic_checkpoints`)
   che nomina la pipeline al fault (dispatch + dump post-fault). Nuovi kernel:
   MXFP4 MoE **v2 split-lane**, **grouping** opt-in (`DS4_VULKAN_MOE_GROUP`),
   dequant MXFP4 **LUT branchless**, **Q8 preq v3** (`dot4add_i8packed`); tool
   `moebench` e `kbench-cuda`. Misure in §2.
   **MoE routed rows/workgroup (R)**: il prefill è **dispatch-bound** (un
   workgroup per (riga, pair) → 2.36M workgroup); **tutti** i kernel MoE
   gate/down (Q8, IQ2/Q2K, Q4_K, MXFP4; v1 e v2) ora processano **R righe per
   workgroup** (macro `MOE_ROWS_BEGIN/END` in `common.hlsl`, `params.rsvd2`
   via `DS4_VULKAN_MOE_ROWS`, default 8) → su NVIDIA guadagnano tutte le path
   (prefill64: MXFP4 113.8→23.6 = 4.8×, IQ2/Q2K 114.8→24.2 = 4.7×, Q4_K
   116.8→51.1 = 2.3×, Q8 111.3→64.6 = 1.7×), **neutro su AMD**. Sub-probe R
   (~0.2 s a init) nel piano autotune, `SPECS_AUTOTUNE.md` §2.5.

### Manca ⬜ (prossimi passi in ordine)

1. **Autotuner a init** (`vulkan/SPECS_AUTOTUNE.md`): le varianti v2 sono già
   registrate come ponte (`DS4_VULKAN_FORCE_VARIANT=<slot>:<idx>` in lista
   separata da virgole; slot Q8_PREQ/MOE_IQ2/MOE_DOWN). **Prima parte FATTA
   (2026-09-11)**: sub-probe **R** (righe/workgroup) a init (`§2.5`) — sceglie
   8 su NVIDIA (5.0×), neutro su AMD, ~0.2 s. Manca il framework di
   selezione per-device delle varianti (misura a init, `g_pipes[slot]` =
   vincente) + probe VRAM (`g_vulkan_max_alloc_bytes`). È il path runtime
   adattivo per il multi-GPU (device diversi → varianti diverse).
   **Attn-output FATTO (2026-09-11, `c243f90`)**: `vulkan_autotune_attn_out()`
   seleziona v1/v2/v3 a init (v3 su AMD e NVIDIA). Restano le altre varianti
   (Q8_PREQ, MoE IQ2/DOWN) e la probe VRAM.
2. **Misura e2e robusta** (incluso l'effetto hotness del nuovo eviction sul
   server): varianza run-to-run ±0.2-0.5 t/s dominata dal page-cache (modello
   80,76 GiB — IQ2XXS 0731 — su 31 GB di RAM) → confronti A/B interleaved,
   mediana, e diff output
   temp 0 per la correttezza. Il design definitivo hot/cold (pinning
   per-device, hotlist per-device) si definisce in Fase 8.
3. **Coda (analizzata, non implementata)**:
   - `attn_decode` — **sospeso**: ROI ~0 (0.1-0.3% della scope attention,
     dominata dalle proiezioni F16 a ~43% della banda).
   - **F16 matmul bandwidth** (219 GB/s su 512 GB/s teorici): la vera leva
     della scope attention, non attn_decode.
   - **Scope ~320 MB non attribuite** (ndisp=38-50 nel per-token).
   - **Kernel fused Metal-only** (`attn_q_b_f16_head_rms_rope_tail`,
     `attention_output_q8_batch_f16`): fallback OK, implementazione = ottimizzazione.
   - **Cooperative matrix / tensor core** (`VK_KHR_cooperative_matrix` + FP4
     nativo): **BASSA PRIORITÀ** (decisione 2026-09-11). Il backend è SPIR-V
     ALU portabile (scelta deliberata per RDNA2/780M, che non hanno coopmat). Su
     NVIDIA chiuderebbe il gap prefill MoE (~7× vs CUDA) e matmul (~12×) **solo
     per MXFP4 e i dense F16**; **NON per IQ2/Q2_K** — sono formati a 2 bit non
     standard, i tensor core non li leggono, e gonfiarli a FP16/INT8/FP4
     costerebbe più memoria/precisione di quanto acceleri un calcolo che non è
     il collo (decode I/O-bound; prefill limitato da memoria+dequant). Da
     rivalutare solo con un percorso NVIDIA-specifico + fallback non-coopmat.
4. **Fusione scope 4f** (SPECS_MONOKERNEL_RESEARCH.md P1.4): DOPO tuning +
   overlap al massimo (kernel veloci → un CS per-token sotto il watchdog).
5. **Quant**: gap DeepSeek chiuso (item 8; `SPECS_GAP.md` §4). Restano fuori
   scope GLM/Vision/BF16 e i tipi solo-esperti Q5_K/Q6_K/Q8_K (nessun modello
   target li usa).

---

## 1b. Multi-GPU — esecuzione distribuita (verifica nel codice, 2026-09-01)

> **Direzione (2026-09-12)**: il **target multi-GPU è il multi-device Vulkan nativo**
> (una `VkInstance`, **N `VkDevice`**, stesso processo; `SPEC.md` §8e), **NON** il path
> multi-processo TCP. Questo §1b documenta il **path distribuito multi-processo**, che è
> un **test delle funzionalità già presenti** (pipeline/layer-split, backend-agnostico) e
> resta un path separato: il race/validazione descritti qui **non bloccano** il
> multi-GPU nativo.
>
> TODO multi-GPU nativo: nuova TU `vulkan/ds4_vulkan_mgpu.c` che sostituisce lo shim
> single-GPU `vulkan/ds4_vulkan_compat.c` (`init_multi` con N device, `set_current_device`,
> `alloc_ptr_on` per tier) + stato **per-device** in `ds4_vulkan.c` + cross-device via
> `VK_KHR_external_memory_fd`/`VK_KHR_external_semaphore_fd` (peer dma-buf, fallback
> host-bounce). API già definita in `ds4_gpu_mgpu.h`. **Design operativo e fasi
> M1-M7: `SPECS_MGPU.md`.**
>
> **M1 FATTO (2026-09-12)**: discovery + classificazione **fast/slow** di tutti i
> device all'avvio (`ds4_vulkan_probe_devices`, banda reale misurata, cache-ata;
> soglia `DS4_VULKAN_FAST_LINK_GBPS`, default **5 GB/s dal 2026-09-14** — era 10;
> a 5 una PCIe 4.0 x4 (~6-7 GB/s) conta come FAST e abilita il layout simmetrico
> 4x4x4x4 con pool esperti per GPU). Validato sul server:
> `device[0]=06:00.0` 0,8 GB/s SLOW, `device[1]=0a:00.0` 0,8 SLOW,
> `device[2]=0d:00.0` 0,1 SLOW, `device[3]=13:00.0` 14,4 FAST, llvmpipe escluso;
> auto-pick → device[3]. Log:
> `ds4: Vulkan device[i] <nome> bdf=... bw=... GB/s vram=... GiB FAST|SLOW`.
>
> **M2 FATTO (2026-09-12)**: config JSON (`--gpu-config`) + planner asimmetrico
> in `ds4_layer_pack.c/.h` (`ds4_mgpu_config_load`, `ds4_mgpu_plan`): tier static
> = layer contigui dei primi layer sui device lenti, tier dynamic = restanti
> (solo dense + pool esperti). Unit test `tests/test_mgpu_config` PASSED.
> **M3-M7**: vincoli/blocchi e ordine consigliato in `SPECS_MGPU.md` §9.
>
> **M3/M4 infra FATTA (2026-09-12)**: probe auto multi-device Vulkan (slow-first),
> CLI `--gpu-vram auto` non forza più CUDA se il backend è VULKAN, planner
> asimmetrico collegato all'engine, contesto per-device in `ds4_vulkan.c`
> (`ds4_vk_dev_ctx` save/restore, `ds4_gpu_init_multi` con N `VkDevice`,
> `set_current_device`, `alloc_ptr_on`, `copy_xdev` host-bounce, free per-tier).
> Sul server: 4 device creati, placement `0-6 / 7-13 / 14-20 / 21-42`, ma il
> **prefill multi-tier fallisce** (`vulkan prefill failed`) → esecuzione M5/M6
> da completare. Dettagli in `SPECS_MGPU.md` §11.
>
> **Pre-flight (2026-09-12):** `make mgpu-probe` →
> `vulkan/tools/mgpu_probe/mgpu_probe.c` (standalone, solo Vulkan): VRAM/heap per device,
> banda device-local e host<->device, **matrice P2P** (dma-buf export/import tra device).
> Da girare sul server appena le 4 GPU sono raggiungibili, prima di scrivere il backend.
> Test locale (780M, 1 device): funziona; device-local ~42-63 GB/s, host<->device ~56 GB/s.
>
> **PREREQUISITO (2026-09-12):** prima del multi-GPU nativo va migliorato il **prefill**
> su Vulkan — oggi **token-per-token** e ~0,4-0,9 t/s (forzato da `ds4.c:32461` perché il
> seed esperti batch supera la pool). Specifica dedicata: **`SPECS_PREFILL.md`**
> (obiettivo centinaia di t/s, prompt lunghi + throughput, generico su tutte le quant).

La SPEC §8d (Opzione C, layer-split asimmetrico) NON richiede una TU
`ds4_vulkan_mgpu.c`: usa il **pipeline distribuito multi-processo già
esistente** del motore. Verificato nel codice (nessuna esecuzione):
`ds4_distributed.c` (8437 righe) ha **ZERO gate backend** (nessun
`__APPLE__`/`DS4_ROCM_BUILD`/`DS4_VULKAN_BUILD` in tutto il file) → il path è
backend-agnostico e, in linea di principio, funziona su Vulkan.

**Come funziona:**
- CLI: `--role coordinator|worker --layers START:END|START:output
  --listen <host> <port>` (coordinator) / `--coordinator <host> <port>`
  (worker) → `ds4_dist_run` (ds4_cli.c:2164 → ds4_distributed.c:8414).
- Coordinator (ds4_distributed.c:5707): listener TCP, **route plan con
  copertura contigua da layer 0 all'ultima** (`dist_coordinator_build_route_plan`,
  ds4_distributed.c:2230+), worker-chaining con **return-upstream** (i logits
  risalgono il pipeline), prefill a chunk con flow-window + `ack_only` per gli
  stage non-finali (solo KV side-effect) + readahead.
- Worker (ds4_distributed.c:7938): listener dati + connessione al coordinator,
  HELLO col proprio range, **prefetch del token successivo**
  (`dist_worker_read_loop_prefetch`) e reconnect automatico.
- Ogni nodo valuta la sua slice via `ds4_session_eval_layer_slice`
  (ds4_distributed.c:2733/3669/7501 → ds4.c:59689): graph engine + `ds4_gpu_*`
  → compatibile Vulkan. Check `prefix_hash` della timeline KV per-processo.
- Caricamento slice: `load_slice`/`load_layer_start/end`
  (`ds4_dist_prepare_engine_options`, ds4_distributed.c:8369); su Vulkan lo
  staging iniziale usa `weights_model_map_decode_static_slice_spans`
  (non-expert della slice, una volta al load, ds4.c:58279).

**Schema 4 processi (loopback TCP, una GPU ciascuno):**
```
# terminal 1 (coordinator):      DS4_VULKAN_DEVICE_INDEX=0 ./ds4 --backend vulkan --role coordinator \
#   --layers 0:10 --listen 127.0.0.1 9000 --model <modello> -c 256 -n 30 --temp 0 -p "Hi"
# terminal 2:                    DS4_VULKAN_DEVICE_INDEX=1 ./ds4 --backend vulkan --role worker \
#   --layers 11:21 --coordinator 127.0.0.1 9000 --model <modello>
# terminal 3:                    DS4_VULKAN_DEVICE_INDEX=2 ... --layers 22:32 --coordinator 127.0.0.1 9000
# terminal 4:                    DS4_VULKAN_DEVICE_INDEX=3 ... --layers 33:output --coordinator 127.0.0.1 9000
```
(il range 10/11/10/10 è un esempio; il bilanciamento si trova per misura —
§8d: lo split giusto è dove lo streaming esposto della x16 ≈ compute delle x1.)

**Caveat trovati nel codice (stato 2026-09-01, validazione su 1 GPU condivisa):**
1. **Il coordinator DEVE partire da layer 0** (ds4_distributed.c:8407,
   `dist_validate_layers_for_model`) → la x16 con la slice finale + output
   head deve essere un **WORKER**; i logits risalgono il pipeline (return
   upstream) fino al coordinator. Contraddice la nota §8d "coordinatore =
   GPU x16 con la slice finale" → da correggere in SPEC.
2. **Slice + `--ssd-streaming` su Vulkan: static slice map ATTIVATA (fix
   2026-09-01)** — `ds4_session_eval_layer_slice` ora stagia le finestre
   non-expert della slice UNA volta (`metal_graph_stream_map_decode_static_slice`,
   guardia `streaming_static_decode_map_current`) invece di ri-stagiarle per
   layer per token. Fix collegato: lo staging DEVE avvenire a scope chiusa
   (prima causava windows vuote → GPUVM fault/device lost), e
   `vulkan_streaming_pool_active` ora controlla OGNI layer caricato (prima
   solo `layer[0]`, NULL su una slice → pool ritenuta inattiva → staggio dei
   blob esperti interi ~20 GiB/slice invece della pool).
3. **Pool per-slice (fix 2026-09-01)**: `vulkan_streaming_pool_active` scansiona
   i layer caricati → su una slice i worker usano la pool (finestre non-expert
   ~3 GiB invece di ~20 GiB di blob esperti).
4. **Batch FFN seed** (n_tokens>1): il prefill di slice su Vulkan richiede
   `--dist-prefill-chunk 1` (il seed batch n_tokens×6 esperti supera il budget
   pool → "ffn batch encode failed"). Con chunk=1 ogni chunk è il path decode
   (funziona). NOTA: con chunk=1 il primo token (pos0=0) passa dal path decode
   grazie al guard `n_tokens == 1` (prima andava al batch).
5. **Kernel fused Metal-only stubbed** (`attn_q_b_f16_head_rms_rope_tail`,
   `attention_output_q8_batch_f16`): il fallback non-fuso funziona ma è più
   lento; implementarli su Vulkan = ottimizzazione.
6. **VALIDAZIONE su 4 GPU separate (2026-09-01 sera, x1 ricollegate)**:
   - **Hardware**: le 3 x1 ricollegate risultano **tutte su PCIe x16 16 GT/s**
     (l'asimmetria x1/x16 della §8d NON esiste più — da rivedere la premessa
     del layer-split asimmetrico; con 4×x16 l'Opzione B expert-shard torna
     competitiva).
   - Il pipeline 4-processi (0:10/11:21/22:32/33:output, una GPU ciascuno,
     `--ssd-streaming --dist-prefill-chunk 1`) **ESCE pulito** (exit 0,
     prefill+generation, prefill 0.10 / gen 0.7-0.8 t/s).
   - **2 processi (0:21 / 22:output): DETERMINISTICO e CORRETTO** (output
     identico al single-GPU, 2/2 run).
   - **3-4 processi: RACE nel relay worker-to-worker** — output
     NON-deterministico, divergenze a token near-tie in punti variabili
     ("The user just said "Hi" — a simple" vs "The user just said "Hi". This
     is a" vs "The user greeted me with "Hi," which"); il single-GPU è
     deterministico. **Esclusi**: contesa GPU condivisa (persiste su GPU
     separate), overlap async (persiste con `DS4_METAL_DISABLE_STREAMING_
     SELECTED_SHARED_OVERLAP=1`). **Sospetto**: il relay worker-to-worker
     (`dist_worker_forwarder_relay_main`, ds4_distributed.c:6437 + coda
      pending-request del forwarder) che appare solo con ≥3 processi. Test
      successivi suggeriti: `DS4_DIST_DISABLE_WORKER_PREFETCH=1` sui worker,
      `-n 1` ripetuto per isolare prefill vs decode, confronto hidden state
      al confine della slice.
      **STATO (2026-09-03): APERTO — passa alla Fase 8** (validazione
      dell'Opzione C / multi-GPU). Il debug definitivo richiede le 4 GPU;
      non è un blocco della Fase 7. Eventuali fix in
      `dist_worker_forwarder_relay_main` / coda pending-request
      (ds4_distributed.c:6437). RIVEDERE la §8d: le 4 GPU sono ora tutte x16
      (niente asimmetria → l'Opzione B expert-shard torna in gioco).
7. **Hardware**: 4 GPU collegate (tutte x16); `DS4_VULKAN_DEVICE_INDEX=0..3`.
8. **`--tensor-parallel` è un path separato** (ds4_tp.c; su Vulkan gli stub
   `tp_*` ritornano 0) → NON serve per l'Opzione C (layer-split).
9. **KV per-processo** dimensionata sulla slice (già gestito:
   `glm_graph_context_memory_estimate_for_compact_cap_slice`).

**Hotness (SPECS_HOTNESS) — rinvio a Fase 8**: la gestione esperti hot/cold
si definisce meglio con l'assegnazione esperto→device del multi-GPU (pinning
caldo/freddo per device, hotlist per-device, eviction LFU+LRU). Su una GPU
singola la leva immediata resta il store (retention hotlist + eviction), ma
NON investire ora nel design definitivo.

**Probe multi-GPU (2026-09-12, `make mgpu-probe`) — hardware reale.** Eseguito sul
server con le 4 GPU AMD collegate. Strumento: `vulkan/tools/mgpu_probe/mgpu_probe.c`
(standalone, solo Vulkan).

- **4× RX 6900 XT (RADV NAVI21, gfx1030)**, **15,98 GiB VRAM** ciascuna; **dma-buf
  importabile/esportabile** su tutte → P2P via `VK_KHR_external_memory_fd` possibile.
- **Banda device-local (memcpy, r+w)**: 242-249 GB/s (tetto copy-engine, ~50% dei 512).
- **Host<->device (r+w)**: **device0 ~54 GB/s (x16)**, device1/2 **1,8 GB/s (x1)**,
  device3 **0,5 GB/s (x1 più lento)** → **l'asimmetria del link È REALE** (contraddice
  il "tutte x16" scritto sopra in §1b, 2026-09-01).
- **Matrice P2P** (GB/s, riga=src, colonna=dst):
  ```
        0     1     2     3
   0:   -    0.7   0.3   0.2
   1:  0.7    -    6.0   1.7
   2:  0.7   6.1    -    1.7
   3:  0.2   6.1   6.0    -
  ```
  **Il P2P funziona**, ma non segue il link host: le GPU 1-2-3 (dietro il chipset X570)
  fanno P2P a ~6 GB/s tra coppie adiacenti; **qualsiasi coppia con device0 è lenta
  (0,2-0,7 GB/s)** — device0 (13:00.0) sta sul root complex della CPU (host veloce,
  P2P lento verso le altre).
- **Mappa indici Vulkan ↔ BDF ↔ link (2026-09-12, probe con `VK_EXT_pci_bus_info`):**
  **NON stabile** — l'ordine di enumerazione cambia tra reboot/BIOS/sessioni, e
  `sysfs` può riportare `x16` su tutte pur con prestazioni molto diverse. **Non
  esiste un indice fisso da usare**: a inizio sessione lanciare `mgpu-probe` e
  scegliere la GPU con la **banda host↔device massima** (vedi `AGENTS.md`).
  - **Osservato 2026-09-12**: i 4 device sono tutti `x16 16 GT/s` in `sysfs`, ma
    e2e Vulkan (`-n 50`): idx 0/1/2 lenti (~0,3-0,7 t/s), **idx 3 veloce
    (~2,0 t/s)**.
  - Storico (da non usare come regola): in sessioni precedenti la veloce era
    risultata `device[0]=13:00.0`; dopo un reboot la mappa è cambiata. La nota
    "card4/index 3 = x16" non è più valida come regola.
- **Implicazioni**: nel decode gli hidden state sono 64 KB/token → trascurabile con
  qualunque link; nel prefill (128 MB/confine) serve scegliere **P2P vs host-bounce per
  coppia** (misura host-bounce da aggiungere al probe). Store esperti su x1: 24 MB a
  0,9 GB/s ≈ **27 ms/layer** → **pinnare gli esperti residenti sulle x1 è essenziale**
  (regge §12.6). Carico one-time di una slice su x1 (~13 GB) ≈ **15 s/GPU**.
- **Prossimi passi**: (1) misura **host-bounce per coppia** nel probe; (2) **config di
  allocazione via JSON** (range di layer → GPU) — vedi `SPEC.md` §8e/§12.5.
  **Nota**: prima di implementare il multi-GPU nativo va riavviato il contesto e risolto
  un altro problema (indicazione dell'utente, 2026-09-12).

---

## 2. Misure chiave (kbench, RX 6900 XT, in=out=4096 tok=1)

| Kernel | ms/call | Note |
|---|---|---|
| matmul_f16 | 0.153 | ~219 GB/s (~43% della banda: headroom per tuning) |
| matmul_f32 | 0.169 | ok |
| matmul_q8_0 (preq) | **0.124 (v2)** | v1 0.266; floor ~0.084 |
| MoE IQ2 gate/up/mid + Q2K down | scope e2e **1.03 ms** | v1 2.47 ms |
| attn_decode | — | sospeso (ROI ~0) |

e2e decode (ds4f-q2, `-n 30`): **1.47-1.76 t/s** overlap ON (default),
A/B interleaved ON vs sync (mediana 1.50 vs 1.53 → identici; varianza
±0.2-0.5 dominata dal page-cache). Con store ~0 (page-cache caldo):
2.6-4.7 t/s osservati (outlier; non attribuibile al flush). Prefill
0.35-0.88 t/s (host-bound sull'encode). Floor teorica ~38 t/s.

### Misure 2026-09-11 (build con il fix dello scratch; kbench/moebench)

kbench (in=out=4096, tok=1, ms/call):

| Kernel | RTX 5080 (Vulkan) | RX 6900 XT (Vulkan) |
|---|---:|---:|
| matmul_f16 | 0.172 | 0.079 |
| matmul_q8_0 v1 | 0.201 | 0.106 |
| matmul_q8_0 v2 | 0.206 | 0.033 |
| matmul_q8_0 v3 (dp4a) | 0.227 | **0.027** |
| matmul_f32 | 0.175 | 0.122 |

Parity q8 v1/v2/v3 esatta su entrambe. Riferimento CUDA: `matmul_q8_0_preq`
~0.017 ms (NVIDIA). La 6900 XT batte la 5080 sul Q8 Vulkan (~7× su v3).

moebench MXFP4 (esperti reali), ms/call:

| Backend | decode (1 tok) | prefill (64 tok) |
|---|---:|---:|
| RTX 5080 Vulkan | 0.793 | 109.3 |
| RTX 5080 CUDA | 3.854 | **3.34** |
| RX 6900 XT Vulkan | **0.409** | 12.8 |

Q2 e2e, **256 prefill + 100 decode** (`--ssd-streaming`, cache fredda), t/s:

| GPU / backend | prefill 256 | decode 100 | 1° token (ms) |
|---|---:|---:|---:|
| RTX 5080 CUDA | 1.35 | 2.67 | 554.9 |
| RTX 5080 Vulkan | 1.40 | 2.76 | 573.6 |
| RX 6900 XT Vulkan | **2.39** | **3.93** | 412.2 |

Note: a queste taglie l'NVIDIA Vulkan ≈ CUDA (bound sullo **streaming SATA**,
non sul compute); CUDA domina il prefill MoE grazie alle **FP4 native
Blackwell**; la 6900 XT è la più veloce sul MoE (kernel più maturi + storage).
Il distacco dei microbench Vulkan-vs-CUDA **non** si vede nell'e2e perché la
GPU attende l'I/O.

---

## 3. Architettura dell'overlap (come funziona, dove)

Fondazione (de-serializzazione v1, Fase 6): la selezione del router vive
**on-device**; i kernel MoE (`moe.hlsl`, `moe_iq2.hlsl`) leggono
`router_selected` (expert id) e traducono via la **tabella esperto→slot**
(`l->meta`, host-visible GTT, binding `DS4_VK_BINDING_TBL`=9), aggiornata
solo su eviction. Il seed su cache-hit non scrive nulla (niente round-trip
per-token). Binding: t0→0..t3→3 (a/b/c/model), u0→4..u4→8, TBL=9,
`DS4_VK_MAX_BINDS`=10.

Il decode è bound dall'expert streaming: pool 6.24 GiB (VRAM piena: static
map 9.18 GiB) → miss 1-6/6 esperti per layer per token → store sincrono
7-27 ms/layer (`vulkan_pool_store_batch`: memcpy da mmap SSD + staging +
copy + device wait) ≈ **450-600 ms/token di solo store** su ~710 ms (1.4 t/s).

Flusso attivo (gate `metal_graph_use_iq2_selected_shared_overlap`, ds4.c,
ON di default; env `DS4_METAL_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP` per
spegnerlo):

1. **Main**: router scope → `ds4_gpu_signal_selected_readback_ready`
   (ds4_vulkan.c): end della scope + submit **diretto con `g_readback_fence`**
   (niente empty submit) → `event_value=1`.
2. **Worker** (`metal_graph_selected_async_load_worker_main`, ds4.c:21492):
   attende la fence, legge `router_selected` via `tensor_read_after_selected_event`
   (download worker-safe su `g_worker_cb`, mai `g_cmd`), poi
   `begin_selected_load_async` → `ds4_vulkan_stream_seed_selected_async`:
   **prepare** (memcpy esperti nello staging) + submit copy su `g_worker_cb`
   con `g_worker_fence` (niente wait).
3. **Main**: computa shared+attention, poi `metal_graph_selected_async_load_finish`
   (attende il worker) → MoE dispatch → `vulkan_pool_commit_pending`
   (wait per-submit della `g_worker_fence`; FIFO main queue = slot-reuse
   sicuro, niente device wait pieno).
   **Flush early-submit (2026-08-31 notte)**: la signal chiude la scope router
   (submit con `g_readback_fence`, che marca anche il CB in volo); il motore
   riapre una scope (`begin_commands` Vulkan-only, solo se il worker è
   partito), ci registra lo shared gate/up/down e `ds4_gpu_flush_commands`
   la sottomette subito (end+submit+fence per-CB + switch al CB gemello,
   **nessun drain**): la GPU esegue shared+attention mentre il worker fa il
   memcpy degli esperti. Il MoE si registra nella scope riaperta dal flush e
   la layer-end la sottomette. Le CB sono double-buffered (`g_cmd[2]`): ogni
   acquire aspetta solo la fence della CB riusata (max 2 scope in volo); il
   riuso della CB in volo (il vecchio fault GPUVM) resta impossibile.
4. Teardown: `metal_graph_selected_async_load_stop()` (join worker) prima di
   `ds4_gpu_cleanup()`.

File chiave: `vulkan/ds4_vulkan.c` (fence/CB worker, store async, commit,
double-buffer scope CB), `vulkan/ds4_vulkan_compat.c` (surface
`begin_selected_load_async`), `ds4.c` (worker + gate + stop). Env bisect:
`DS4_VULKAN_ASYNC_STORE_OFF=1` (store sync nel worker, per isolare),
`DS4_VULKAN_FLUSH_END_BEGIN=0` (flush no-op legacy: niente early-submit).

---

## 4. Crash fixati nell'overlap (2026-08-31, da NON ri-introdurre)

1. **GPUVM fault / context lost**: la signal sottommetteva la scope router con
   fence senza drain; i dispatch one-shot successivi ri-registravano `g_cmd`
   in volo. Fix: **drain in `vulkan_dispatch_begin`** (settle se
   `g_vulkan_device_dirty` prima di riusare g_cmd).
2. **Double free RADV**: `vkDeviceWaitIdle` + `vkResetDescriptorPool`
   concorrenti tra main e worker (grow dello staging). Fix: **mutex su
   `vulkan_device_wait`**.
3. **Riallocazione pool dal worker**: `vulkan_pool_layer_ensure` poteva
   liberare/riallocare `l->tensor` mentre il main lo bindava. Fix:
   **no-realloc nel seed async** (param `allow_realloc=0`; fail → il finish
   fa il retry sync sul main thread).
4. Teardown: worker mai joinato → **stop+join prima di `ds4_gpu_cleanup`**.
5. **Deadlock `DS4_VULKAN_DEBUG_SUBMIT`** (da me introdotto col double-buffer,
   fixato): il path debug di `end_commands` resettava la fence ma la lasciava
   nello slot → `vulkan_cb_acquire` aspettava una fence mai più segnalata
   (hang al begin del layer successivo, exit 124). Fix: il path debug ora
   distrugge la fence e azzera lo slot (la scope è completa).
6. **Descriptor-pool esaurita nella prefill** (da me introdotto, fixato):
   rimossi i drain per-layer (device_wait) → niente reset della pool durante
   la prefill decode-style senza readback (10 token × ~950 set > 8192).
   Fix: contatore `g_desc_sets_allocated` + **HWM drain** (6144 set) in
   `vulkan_cb_acquire` (drain raro, ~1/6-7 token di prefill).
7. **Segfault `DS4_VULKAN_DEBUG_SUBMIT` (fixato 2026-09-11)**: il fix dell'item 5
   distruggeva `g_cb_fence[g_cmd_i]` senza azzerarlo → al `vulkan_cb_submit`
   successivo `vkResetFences` su handle distrutto (exit 139). Fix: azzera solo
   il marker di in-flight (come `vulkan_cb_acquire`), **senza distruggere**; la
   fence persistente è reset+riusata. Inoltre la signal ora attende la fence di
   readback sotto `DEBUG_SUBMIT`, così il tempo GPU dell'attention è attribuibile.
   **Attribuzione per-layer (decode, RX 6900 XT, ds4f-q2)**: attention+router
   **~6-7 ms** (p50 6.68, picchi 7.4), MoE **~0.65 ms**, GPU totale ~7 ms/layer →
   è il **vero collo** (floor di banda ~0.6 ms/layer → ~38 t/s). Verifica: output
   `--temp 0` identico normal vs DEBUG, smoke PASSED (780M), normal path intatto
   (modifiche solo nei rami debug). Dettagli: `SPEC_CROSSLAYER_PREFETCH.md` §4.
8. **GPU timestamp scope timing + attribuzione corretta (2026-09-11)**: nuova env
   `DS4_VULKAN_DEBUG_GPU_TS=1` (richiede `DEBUG_SUBMIT=1` e overlap worker OFF) →
   timestamp device a inizio/fine scope + coppia attorno a `router_select`, letti
   dopo la fence (separa il tempo GPU reale da fwait/queue). **Risultato (RX 6900 XT,
   ds4f-q2, decode)**: attention **~1,5-2 ms**, `router_select` **~0,1-0,8 ms**
   (`rbench` isolato 0.14 ms; in-situ 0.8), MoE **~0.9 ms**, **store esperti ~4,5 ms**
   (p50) — e lo store è nella **stessa scope** dell'attention (verificato con
   `SKIP_PIPE=28`: la scope scende da ~7 a ~1.7 ms). Quindi la scope "attention" da
   ~7 ms è attention + router + **store**: il **router è trascurabile**, il collo è lo
   **store serializzato nella scope**, non l'attention. Dettagli:
   `SPEC_CROSSLAYER_PREFETCH.md` §4.
9. **Breakdown per-dispatch dell'attention + variante DP4A (2026-09-11)**: aggiunta
   `DS4_VULKAN_DEBUG_GPU_TS_VERBOSE=1` (timestamp per ogni dispatch, nome pipe).
   **Top consumer GPU nel decode (somma 43 layer, un token)**: `attn_output_low_q8`
   **183 ms** (4,27 ms/call — il collo), `router_select` 26, `matmul_q8_0_preq_v3` 21,
   `moe_gate_up_mid_iq2xxs` 14, `attn_decode` 10, rms norm ~14, `quantize_q8_0` 8.
   In isolamento (`rbench2`) `attn_output_low_q8` = **0,98 ms** (in-situ 4,7: la
   differenza è la stall dello store). Il kernel legge i Q8 **byte-per-byte** e
   **ri-quantizza l'attivazione per ogni riga** (rank=1024 per gruppo): 1,3× dal solo
   DP4A. **Fatto**: variante `attn_output_low_q8_v2` (DP4A `dot4add_i8packed`,
   `VK_KHR_shader_integer_dot_product`, cs_6_4; selettore `ATTN_OUT_LOW:0|1`) →
   **0,98 → 0,75 ms** isolato, output **bit-identico** a v1 (smoke + diff temp 0),
   e2e invariato (lo store domina). Prossimo passo per l'attention: **riusare
   l'attivazione tra righe** (una workgroup per gruppo/righe, quantizzare una volta):
   rimuove ~4× di traffico ridondante (stima ~5× sul kernel). Ma il ROI e2e resta
   basso finché lo **store (~4,5 ms) non è sovrapposto**.
10. **Misura NVIDIA RTX 5080 (2026-09-11, Vulkan)**: verbose GPU_TS (overlap OFF).
    **attention scope ~2,2 ms/layer → 94 ms/token (~78% del compute)**, MoE
    ~0,65 ms/layer → 26 ms/token. Dentro l'attention: `matmul_q8_0_preq_v3`
    **53 ms/token (0,29 ms/call)** — su AMD 0,069 ms/call, quindi il path
    DP4A/OpSDot Vulkan è **~4× più lento su NVIDIA**; `attn_output_low_q8_v2`
    13,5 ms/token (0,315 ms/call — su AMD 0,98: NVIDIA 3× più veloce);
    router 4,6, attn_decode 3,4, rms ~6. **Quindi il collo attention è diverso per
    vendor**: su AMD è `attn_output_low_q8`, su NVIDIA è il **matmul Q8 denso**.
    La variante v2 (DP4A) dà ~1,3× su AMD e ~1× su NVIDIA. e2e: ~4 t/s (output
    identico v1/v2); la quota non-attention (~130 ms/token) è store/IO.
11. **Rewrite attention output-low + autotuner a init (`c243f90`, 2026-09-11)**:
    variante `attn_output_low_q8_v3` (riuso attivazione): una workgroup calcola
    AO_RT=8 righe dello stesso gruppo; l'attivazione è quantizzata **una volta**
    in groupshared e riusata dalle 8 righe (32 lane/riga, dot int8 packed).
    L'ordine dell'albero di riduzione riproduce quello di v1 → risultato
    **bit-identico** (1 ulp in isolamento) e output temp-0 **uguale**. Isolato
    (gdim=4096 rank=1024): RX 6900 XT **0,98 → 0,19 ms**, RTX 5080 **0,32 →
    0,11 ms**; entrambe scelgono v3. **e2e AMD** (ds4f-q2, -n 16, temp 0):
    generation mediana **1,94 → 2,71 t/s (+40%)**. **Autotuner**:
    `vulkan_autotune_attn_out()` misura v1/v2/v3 a init (mappa Q8 sintetica, come
    il probe R della MoE) e fissa `g_attn_out_autotuned` per-device;
    `DS4_VULKAN_FORCE_VARIANT=ATTN_OUT_LOW:0|1|2` per bisect,
    `DS4_VULKAN_AUTOTUNE_OFF` per disabilitare. Smoke PASSED su 780M/AMD/NVIDIA.
    Nota: l'autotune del `q8_preq` è stato provato e **rimosso** (probe
    inaffidabile: dispatch senza effetto → tempi falsi); resta il default v3.
12. **Benchmark completi NVIDIA RTX 5080 (2026-09-12, driver open 595.99.02,
    kernel 7.0.0-31)**: tool ricompilati dalla sorgente corrente.
    - `vkbench` (dev 0): SAXPY 182,6 GB/s, FMA FP32 **97,6 TFLOPS**, COPY D2D
      204,9 GB/s, H2D/D2H 12,7/13,1 GB/s, stabilità 15 s OK.
    - `kbench` 4096² tok=1: f16 0,147 ms; q8 v1/v2/v3 0,184/0,194/0,179 ms
      (parità v1-v2/v2-v3 OK); f32 0,161. CUDA q8 v2/v3 **0,017 ms** (~10×),
      f16 1,56 ms (path lento). **A tok=32 il vantaggio CUDA sul q8 svanisce**
      (0,028 vs 3,7 ms) → il Vulkan **non batcia** i matmul.
    - `moebench` (MoE routed, dim reali) prefill_64: mxfp4 Vulkan **23,6 ms**
      (217 GB/s) vs CUDA **1,78 ms** (2886 GB/s); q4k 51,0 vs 2,84; iq2 24,2 vs
      3,27 → **CUDA ~8-13× sul prefill MoE** (quantificazione del gap §4.1 di
      `SPECS_PREFILL.md`). q8 CUDA non supportato.
    - `mgpu-probe`: 1 device, VRAM 15,92 GiB, copy device-local 819 GB/s,
      host↔device 27,1 GB/s, external OPAQUE_FD import/export OK.
    - **Autotune `attn_output_low` su NVIDIA = v2** (0,108 ms, gdim=4096
      rank=1024), non v3 come riportato all'item 11: la scelta dipende dal
      probe a init / dalla shape; da rivedere. MoE rows/workgroup = 8.

---

## 5. Build e test

```sh
make -j$(nproc) -B vulkan            # 5 binari, 0 warning, ~25 s
make -j$(nproc) -B tests/test_vulkan_smoke && ./tests/test_vulkan_smoke  # 396/396 (780M)
gcc -I. -fsyntax-only ds4.c          # gate motore
nm vulkan/ds4_vulkan.o | grep ' ds4_gpu_' | grep -c _Z   # = 0 (ABI non manglata)
make -j$(nproc) -B kbench && ./kbench [in_dim [out_dim [n_tok [iters]]]]  # matmul v1+v2+parity
```

Server (RX 6900 XT, Zen3, NO AVX-512; dxc NON installato → build locale e scp):

```sh
make -j$(nproc) -B NATIVE_CPU_FLAG=-march=znver3 vulkan
make -j$(nproc) -B NATIVE_CPU_FLAG=-march=znver3 tests/test_vulkan_smoke
scp ds4 tests/test_vulkan_smoke kbench <server>:/tmp/
```

**GPU del server**: ora c'è UNA SOLA GPU (le 3 x1 sono state scollegate
fisicamente dall'alimentatore) → **index 0** (NON più 3). Auto-probe comunque
attivo. Modello:
`<server-model-dir>/model/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`

```sh
DS4_VULKAN_DEVICE_INDEX=0 ./ds4 --model <modello> --backend vulkan \
  --ssd-streaming --ssd-streaming-cache-experts 64 -c 256 -n 30 --temp 0 -p "Hi"
# correttezza: diff output temp 0 tra path overlap e DS4_METAL_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP=1
```

---

## 6. Telemetrie (env, debug)

- `DS4_VULKAN_DEBUG_SUBMIT=1` — timing per scope (end/queue/fwait) + ndisp/binds/bytes.
- `DS4_VULKAN_DEBUG_POOL_HIT=1` — miss del seed pool per layer + `store_time`.
- `DS4_VULKAN_DEBUG_DISPATCH=1` — log di ogni dispatch (pipe, gx/gy/gz);
  `DS4_VULKAN_DEBUG_DISPATCH_TIME=1` — timing per-stage (<0.5 µs, non è il collo).
- `DS4_METAL_GRAPH_TOKEN_PROFILE=1` / `DS4_METAL_GRAPH_LAYER_PROFILE=1` —
  encode vs execute per token / per layer.
- `DS4_VULKAN_DEBUG_BINDS=1` — finestre staggiate + VA, bind NULL, GPU scelta.
- `DS4_VULKAN_DEBUG_ONESHOT=1` — submit+wait per dispatch (attribuisce un
  fault GPUVM); `DS4_VULKAN_SKIP_PIPE=<i,j,...>` — bisect saltando pipe.
- `DS4_VULKAN_DEVICE_INDEX=N` — forza la GPU (server: 0).
- `--vulkan-stats` (CLI `ds4`/`ds4-server`) — blocco stats della pool esperti
  a fine risposta (hit/miss %, MiB caricati, eviction/reload, occupazione,
  stall per fase decode/prefill/hotlist). Contatori: `g_pool_tel[]` (atomic
  per layer) + timing in `vulkan/ds4_vulkan.c`; snapshot C
  `ds4_vulkan_telemetry_snapshot()` → `ds4_vulkan_stats_report()` in ds4.c
  (richiede pool attiva = `--ssd-streaming-cache-experts`, altrimenti no-op).

---

## 7. Gotchas essenziali per ripartire

- **Build per il server = MAI riusare oggetti di build default**: la macchina
  locale compila con `-march=native` (AVX-512); oggetti stale linkati in una
  build `NATIVE_CPU_FLAG=-march=znver3` → SIGILL su Zen 3 (visto 2026-09-09 con
  `ds4_distributed.o`). Sempre `make -B NATIVE_CPU_FLAG=-march=znver3 ...` per
  il server, e ridistribuire anche i `.o` condivisi.
- **`--ssd-streaming` NON è supportato dalla build CPU** (solo metal/cuda/rocm):
  la versione CPU gira il modello via mmap senza expert pool.
- **Firme ABI C identiche a `ds4_gpu.h`** nei backend (C++ mangla → link fail);
  stub da rimuovere da `ds4_vulkan_unavailable.c` quando si implementa.
- **Stub Vulkan muti + report all'init** (2026-09-10): `ds4_vulkan_unavailable.c`
  non stampa più "Vulkan unavailable" al call-site. Ogni stub registra il suo
  nome a static-init; `ds4_vulkan_report_unavailable()` (chiamata da
  `ds4_gpu_init`) stampa **una volta** un riepilogo categorizzato (105 stub:
  GLM/KDA, Vision, multi-GPU/TP, decode graph, fusi/plumbing). Con
  `DS4_VULKAN_STUBS_VERBOSE=1` elenca tutti i nomi. Prima erano fail-loud: ora
  l'unica traccia di una funzione non implementata è il report di startup.
- **Gate motore**: blocchi positivi `#if defined(DS4_ROCM_BUILD) ||
  defined(DS4_VULKAN_BUILD)`; negativi `&& !defined(DS4_VULKAN_BUILD)`.
- **Watchdog amdgpu ~10 s**: mai CS lunghi (batched disabilitato su Vulkan);
  i kernel v2 tengono i dispatch singoli ben sotto.
- **Blocco Q8_0 34 byte (2-mod-4)**: letture vettorizzate = load allineato +
  shift (v2 già lo fa); `w.Load` HLSL richiede 4-allineato (gotcha Q2K scale).
- **`f32tof16` HLSL NON è RNE su RDNA2**: usare `ds4_f32_to_f16_bits_rne`.
- **Pipeline in `vulkan_compute_init()`, MAI lazy**; un `.hlsl` = binding
  coerenti (un register = una risorsa); push constant `DS4Params` (96 B,
  `common.hlsl`).
- **`tensor_free_in_place` NON chiama `tensor_free`** (double free).
- **Mai `(base+size)-offset`** nei range check (underflow → GPUVM fault).
- **Smoke flaky pre-esistente**: `compressor store` fallisce ~1/15 run (race
  nel kernel, NON regressione; verificato con git stash).
- **Teardown silenzioso (fixato 2026-09-09)**: `ds4_gpu_decode_graphs_invalidate`
  è CUDA-only (decode-island capture); il motore la chiama però
  incondizionatamente a chiusura del graph runtime → su Vulkan finiva sullo
  stub "Vulkan unavailable" a fine di ogni run. Ora no-op silenzioso reale in
  `ds4_vulkan.c` (rimossa la riga da `ds4_vulkan_unavailable.c`).
- **Makefile vkbench race (fixato)**: la regola `$(VKBENCH_HDRS):` (2 target,
  1 recipe) con `-j` eseguiva la recipe 2× in concorrenza (race su
  `/tmp/vkbench_spv_tmp.bin`) → build flaky. Ora `&:` (grouped target).
- **`gen_shaders.py` race (fixato 2026-09-11, `6176817`)**: con `-j` e più goal
  (`make vulkan tests/...`) la recipe gira 2× → race su `.tmp.shader.spv`
  (FileNotFoundError). Ora temp per-PID + `os.replace` atomico dell'`.inc`.
- **`-fspv-target-env=vulkan1.1` sui kernel MoE MXFP4 v2 = INUTILE**: testato
  su NVIDIA (RTX 5080) e AMD — i kernel usano solo f32/uint (niente subgroup né
  storage 8/16-bit), stessa dxc 1.9.0.1; con e senza il target env l'esito è
  identico. Rimosso dal branch. Il server NVIDIA lo aveva come residuo della
  debug device-lost (il fix vero era `vulkan_scratch_retire`, `a0875f0`).
- **`DS4_VULKAN_DEBUG_SUBMIT=1` ora aspetta ogni scope** (semantica di
  misura): con le CB double-buffered serve per attribuire i tempi per-scope;
  mai usarlo nelle misure e2e (serializza e rallenta il run).
- **LSP su ds4.c = rumore** (cast void*: il build reale è gcc `-std=c99`).
- **ROCm non buildabile per RX 6900 XT** (rocwmma solo CDNA/RDNA3+) → Vulkan.
- **Kernel batch prefill multi-layer** su Vulkan = limitazione nota (prefill
  tutto decode-style; `DS4_METAL_STREAMING_DECODE_PREFILL_MAX=UINT32_MAX`).

---

## 8. Come si aggiunge un kernel/variante (pattern)

1. Entry point in un `.hlsl` (set 0, bindings fissi, push `DS4Params`).
2. Tupla in `vulkan/shaders/gen_shaders.py` → `ds4_spv_<nome>[]`/`_len`.
3. In `vulkan_compute_init()`: riga in `pipes[]` + valore in **coda** all'enum
   `ds4_vk_pipe` (+ bump `DS4_VK_PIPE_COUNT`). MAI lazy, MAI riordinare.
4. Dispatch: `vulkan_dispatch(pipe, &params, sizeof(params), binds, nb, gx, gy, gz)`.
5. Variante = nuova pipeline, stesso binding set (SPECS_AUTOTUNE §2.1);
   selettore `vulkan_pipe_*` + env `DS4_VULKAN_FORCE_VARIANT` (già in uso).
6. Build+smoke+gate (§5).

---

## 9. Commit recenti e setup fork (branch `vulkan-backend`)

**Setup pubblico/privato (2026-08-31)**:
- Branch locale **`vulkan-backend`** = PRIVATO (storia completa, docs
  `vulkan/*.md` tracciati). MAI pushare questo branch.
- Branch locale **`vulkan-backend-public`** = squash in UN commit (`49d2127`)
  di tutto il lavoro SENZA `vulkan/*.md` (restano locali; in `.gitignore` sul
  branch pubblico). Pushato sul fork ricreato come branch `vulkan-backend`.
- Il fork `sp82/ds4` è stato **cancellato e ricreato** da upstream (la vecchia
  storia con i docs e il nome autore è sparita da GitHub).
- **Privacy**: nei file pubblici mai IP/percorsi/hardware personali
  (placeholder `<server>`, `<repo>`, `<server-model-dir>`, `<bdf>`);
  sanificazione fatta in `c968fe7`. Il flusso squash+push è in `AGENTS.md`.
- Allineamento con upstream: `git fetch origin && git rebase origin/main` sul
  branch privato (i `vulkan/*.md` non confliggono mai: upstream non ha la
  cartella `vulkan/`; i conflitti sono nei file motore ds4.c/ds4_gpu.h/Makefile).
  Il branch pubblico si rigenera con lo squash DOPO l'allineamento.

**Commit recenti (branch privato)**:
- `a0875f0` — **fix causa radice device-lost NVIDIA**: `vulkan_scratch_retire()`
  (flush della scope aperta prima del drain/free dello scratch in grow); tool
  `DS4_VULKAN_CHECKPOINTS`.
- `e481916` — fence persistente per command buffer (reset+reuse).
- `dd10ee8` — fix leak `VkFence` (host OOM dopo migliaia di scope).
- `b659eba` — serializzazione device-wait + reset pool con i submit.
- `3e993c8` — fix race `compressor_store` + tolleranza rope 2e-5 (NVIDIA libm).
- `f562835` — dequant MXFP4 branchless (magnitude LUT).
- `c3083f0` — grouping per esperto opt-in (`DS4_VULKAN_MOE_GROUP`).
- `3b487cf` — kernel MXFP4 MoE v2 split-lane + tool `moebench`.
- `e3df35d` — Q8 preq v3 (dp4a, `VK_KHR_shader_integer_dot_product`).
- `f8d7761` — submit coda serializzati + pool/CB dedicati al worker.
- `855ed96` — binding flag descriptor array + `descriptorBindingPartiallyBound`.
- `a3cd988` — fix link CUDA (chiamata async expert-load gated Vulkan).
- `ad8d230` — link `kbench-cuda` (`ds4_image.o`).
- `50ff610` — kbench: target di build CUDA (`kbench-cuda`).
- `3cc1e12` — gitignore binari `kbench`/`kbench-cuda`.
- *(non committato)* — telemetria `--vulkan-stats`: contatori pool per layer +
  snapshot C + report per-request (CLI e ds4-server); modifica core tracciata
  in `SPECS_CORE_CHANGES.md` §2a.
- *(non committato)* — flush early-submit: double-buffer `g_cmd[2]` + fence
  per-CB (begin senza drain, flush end+submit+switch), marker readback nel
  slot CB, HWM drain descriptor pool, fix deadlock debug-submit, retry sync
  end/begin nel path overlap (Vulkan), fix race Makefile vkbench `&:`,
  SPEC §8d + progress (verifica path distribuito multi-processo).
- `c968fe7` — sanificazione dati personali (IP/percorsi/hardware/BDF →
  placeholder; `.clangd`+`compile_commands.json` fuori dal tracking).
- `9604210` — progress.md senza riferimenti all'archivio + dettagli fondativi.
- `6ea0c00` — progress.md compatto + archivio del dettaglio storico.
- `8793739` — fix GPU fault + double free nell'overlap (drain dispatch_begin,
  mutex device_wait, no-realloc seed async, stop/join worker), gate ON default.
- `8d01b7c` — overlap store↔compute: sync readback fence + store async +
  worker (WIP, poi fixato in 8793739).
- `4fe26d0` — MoE IQ2/Q2K v2 (scope 2.47 → 1.03 ms) + attn default.
- `685f76b` — matmul Q8 preq v2 (0.266 → 0.118-0.124 ms) + kbench parity.
- `699eb31` — SPECS_AUTOTUNE.md (piano autotuner).
- `3942d8a` — Fase 7 pronta al lancio (kbench + docs).

File generati non tracciati: `vulkan/shaders/ds4_vulkan_shaders.inc` (61
shader), `iq2_tables.hlsl`/`iq2_tables_host.h`, `vulkan/tools/vkbench/*.h`.

---

## 10. Riferimenti

- `vulkan/SPEC.md` — fasi progetto, multi-GPU Opzione B/C (§8d).
- `vulkan/SPECS_KVCACHE.md` — dimensione KV cache reale (formula CSA, budget VRAM).
- `vulkan/SPECS_AUTOTUNE.md` — autotuner a init (varianti + probe VRAM).
- `vulkan/SPECS_MONOKERNEL_RESEARCH.md` — monokernel/fusione (P1.4 dopo tuning).
- `vulkan/SPECS_HOTNESS.md`, `vulkan/SPECS_GAP.md` — hotness esperti, gap funzionale.
- `vulkan/SPECS_NON_DETERMINISTIC.md` — non-determinismo run-to-run del decode
  single-GPU (indagine in corso, 2026-09-04; invalida il gate "diff temp 0").
- `vulkan/AGENTS.md` — note operative (build parallela `make -j$(nproc)`).
- Esecuzione distribuita: `ds4_distributed.c` / `ds4_distributed.h`
  (backend-agnostico; verifica in §1b).

Ultimo aggiornamento: 2026-09-11 — **VALIDAZIONE NVIDIA (RTX 5080 16 GiB,
driver 595.84, CUDA 13.2) e fix device-lost**. Il backend gira su NVIDIA con
`--ssd-streaming` dopo la catena di fix del device-lost (Xid 109), chiusa da
`a0875f0` (**lifetime dello scratch buffer**: la scope aperta referenziava un
buffer liberato dal grow → GPU su memoria libera). MXFP4 e Q2 completano
prefill+generazione a 16 GiB (0 Xid); smoke **450/450** su NVIDIA, RX 6900 XT
e iGPU 780M. Nuovi: MXFP4 MoE v2 split-lane + grouping opt-in + LUT branchless,
Q8 preq v3 (dp4a), tool `moebench`/`kbench-cuda`, debug
`DS4_VULKAN_CHECKPOINTS`. Misure §2: NVIDIA Vulkan ≈ CUDA sull'e2e I/O-bound,
CUDA vince il prefill MoE con **FP4 native**, la 6900 XT è la più veloce.
Aggiunto l'autotune del parametro **R** (righe/workgroup) di **tutti** i kernel
MoE routed (Q8, IQ2/Q2K, Q4_K, MXFP4; dispatch-bound: 1.7-4.8× su NVIDIA,
neutro su AMD); **prima parte dell'autotuner a init FATTA (2026-09-11)**:
sub-probe R (~0.2 s) che sceglie 8 su NVIDIA, `SPECS_AUTOTUNE.md` §2.5.
Prossimi passi
invariati (framework autotuner varianti + probe VRAM, misura e2e robusta);
nuova
direzione: validazione su GPU **96 GiB** (residente, non streaming).

Prima: 2026-09-09 — **smoke Vulkan vs CPU sul server + gotcha
build** (Ryzen 5 5600X = Zen 3): entrambi i backend girano e producono output
deterministico identico a `--temp 0` (CPU `-n 8` = prefisso del Vulkan `-n 16`).
Vulkan con `--ssd-streaming`: gen ~0.5 t/s (cache fredda). CPU SENZA
`--ssd-streaming` (non supportato su CPU): prefill 1.30 t/s, gen 1.42 t/s a
cache calda. **Gotcha SIGILL**: la prima CPU build dava "Illegal instruction"
a inizio `parse_options` — causa: oggetti `ds4_distributed.o` stale compilati
`-march=native` (AVX-512) riusati dal link CPU (istruzione `vmovdqu64 %zmm`,
illegale su Zen 3). Fix: **`make -B NATIVE_CPU_FLAG=-march=znver3 cpu`**
(recompile forzata di TUTTI gli oggetti; mai mescolare oggetti di build default
`-march=native` con build per il server). Prima oggi: **ALLINEAMENTO A
`upstream/main` `6289c51`**
(rebase dei 3 commit su 197 commit upstream: Vision-Exp, GLM 5.3 multimodale,
iris jpeg/png, agent rework, refactor static decode map
`enabled()`+`supported()`, TP/DSPark/kv-norm). Conflitti risolti su
Makefile/ds4.c/ds4_agent.c; 16 simboli `ds4_gpu_*` nuovi (vision/GLM53/DSpark)
stubbati in `ds4_vulkan_unavailable.c` (irraggiungibili su Vulkan), `ds4_image.o`
aggiunto al link vulkan; fork main in FF. Validazione: build 5 binari 0 warning,
gate c99 ×3, ABI `nm` 0 `_Z`, smoke PASSED, e2e CLI + ds4-server HTTP con
`--vulkan-stats` (zero spam "Vulkan unavailable"). Backup pre-rebase:
`align-backup-4737b65`. In precedenza oggi: fix teardown (`decode_graphs_invalidate`
no-op silenzioso) e validazione e2e della telemetria (numeri §1 item 7).
Prima: 2026-09-04 (non-determinismo
run-to-run del decode single-GPU sul server — SPECS_NON_DETERMINISTIC.md;
causa circoscritta al path async dell'overlap store↔compute, fix non ancora
applicato). Ancora prima: 2026-09-03 (hotness esperti implementata — priorità
hotlist + eviction LFU+LRU, item 6 "Fatto"; riorganizzazione fasi: il debug
del race distribuito passa alla Fase 8 — §1b caveat 6; prossimi passi Fase 7:
autotuner, misura e2e). Ancor prima: 2026-09-01 (flush early-submit fatto —
gain e2e ~0 nel regime I/O-bound, 2.6-4.7 t/s a cache caldo; verifica del path
distribuito multi-processo per l'Opzione C — §1b).

> **NOTA — portare la modifica async su upstream (2026-09-17)**: `a3cd988`
> ("cuda: fix link broken by Vulkan-only async expert-load call") gata
> `ds4_gpu_stream_expert_cache_begin_selected_load_async` (definita SOLO in
> `vulkan/ds4_vulkan_compat.c`, dichiarata in `ds4_gpu.h:309`) sotto
> `#if defined(DS4_VULKAN_BUILD)` e usa la variante sync
> `ds4_gpu_stream_expert_cache_begin_selected_load` per CUDA/Metal/ROCm. Su
> `upstream/main` la funzione async NON esiste (il motore usa solo la sync, es.
> `metal_graph_selected_async_load_run` ds4.c:23513). **TODO: cercare di
> portare la variante async expert-load su upstream** (in alternativa,
> rendere la variante async un'estensione realmente opzionale nei backend
> non-Vulkan) così da non dipendere da una guardia locale `DS4_VULKAN_BUILD`.
> Dopo il rebase serve la build di TUTTI i backend per verificare che non
> ricompaiano `undefined reference` simili.
