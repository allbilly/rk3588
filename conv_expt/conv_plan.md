# conv_expt Active Plan

## Goal

Build a clean FP16 CONV implementation for RK3588 NPU with:

- `examples/conv.py < 1000` lines.
- No CPU/GPU fallback.
- No RKNN hex/task-table replay in the final runtime.
- Unsupported shapes fenced before allocation.
- Eventual `217/217 PASS` on the canonical sweep.

## Rewrite Status 2026-06-04

`conv_expt/conv_1stprinciple.py` has now been completely rewritten as the compact
runtime seed. The old 5k-line proof harness is preserved as
`conv_expt/conv_1stprinciple_ref.py`, so the active file no longer needs to carry
RKNN capture materializers, parity reports, failed closure experiments, or long
shape-name allowlists.

The active runtime is now coverage-first and descriptor-driven:

```text
shape
  -> conv_tile_planner descriptor rows
  -> generic local schedule selection
  -> direct / BY_Y / BY_K / BY_YK setup-local / grouped / depthwise serial
  -> one small hardware submit per local tile
```

This is intentionally not the final high-performance exact-chain CONV compiler.
It is the clean base for replacing expensive local decomposition with real
`setup/k_half/k_tile/aux` formulas once those semantics are derived from RK3588
atomic-C/atomic-K hardware rules.

The final design stays descriptor-driven:

```text
shape
  -> conv_expt/conv_tile_planner.py
  -> NONE / BY_Y / BY_K / BY_YK
  -> descriptor rows
  -> generic emitter family
  -> safe hardware submit closure
```

## Current Truth

`examples/conv.py` is a hardware research scratchpad, not the final clean implementation.

Current observed state:

| File | Role | State |
|---|---|---|
| `examples/conv.py` | Hardware oracle/scratchpad | ~2927 lines, many prefix-replay-derived special cases. |
| `conv_expt/conv_tile_planner.py` | Planner source of truth | ~527 lines, owns descriptor planning only. |
| `conv_expt/conv_1stprinciple.py` | Active compact runtime seed | 375 lines, planner-backed, generic local decomposition, no exact11/capture materializer tables. |
| `conv_expt/conv_1stprinciple_ref.py` | Archived proof harness/oracle | ~5215 lines, historical 217/217 evidence, capture-derived materializers, parity/debug reports. |
| `examples/conv_tiles.py` | Large RKNN-topology reference | ~2588 lines, do not import/copy into clean runtime. |

The archived proof harness reached `runnable=217, fenced=0` and hardware-passed the canonical 217-shape sweep. Many promoted families still came from prefix-replay-derived evidence. That proves hardware behavior but not yet the final clean abstraction.

The active compact runtime is not yet claimed as the final 217/217 hardware implementation. It is the new base to validate generic local decomposition first, then selectively replace slow local paths with real exact-chain emitter formulas. Several parity-clean CONV closures remain retained only in `_ref.py` as analysis evidence because they failed numerically or were unsafe.

## Conv Python File Map

Use this map to decide what should feed first-principles work. High-value files provide planner rules, safe no-submit evidence, or hardware-proven register behavior. Medium-value files are references that should not be copied directly. Low-value files are historical or narrow probes.

| File | Role | Help for first-principles work | Why |
|---|---|---:|---|
| `conv_expt/conv_1stprinciple.py` | Active compact runtime seed | High | Target implementation; owns generic local schedule selection and submit flow. |
| `conv_expt/conv_1stprinciple_ref.py` | Archived proof harness/oracle | High | Historical hardware evidence; mine for formulas only, do not port shape tables. |
| `conv_expt/conv_tile_planner.py` | No-submit descriptor planner | High | Core planner contract for `NONE/BY_Y/BY_K/BY_YK`. |
| `conv_expt/conv_tile_cpu.py` | CPU-only planner validation harness | High | Safely validates descriptor math and shape coverage without NPU submit. |
| `conv_expt/conv_pershapepatch.py` | Large per-shape patched submit path | Medium | Useful historical register evidence, but too per-shape for final runtime. |
| `examples/conv.py` | Current hardware oracle/scratchpad | High | Provides packing, submit helpers, exact closure emitters, and proven hardware behavior. |
| `examples/conv_tiles.py` | Large RKNN-topology tiled CONV implementation | High | Strong evidence for RKNN-style descriptors, task layout, and closure behavior. |
| `examples/conv_tiles_no_submit.py` | No-submit decoded-reg scaffold | Medium | Safe fixture/reference for decoded register rows. |
| `examples/conv_no_submit_materializer.py` | No-submit descriptor/materializer model | High | Good reference for clean descriptor-to-register emitters. |
| `examples/conv_no_submit_fixtures.py` | Fixture registry | Medium | Records known windows, offsets, and fenced cases. |
| `examples/conv_no_submit_closure.py` | Symbolic closure/PC-chain utilities | Medium | Useful for PC-tail and multi-row closure reasoning. |
| `examples/conv_small_no_submit.py` | Small no-submit layout probe | Medium | Helps validate task/regcmd assumptions safely. |
| `examples/conv_h14_k_tile_no_submit.py` | H14/H7 BY_K no-submit audit | High | Important BY_K exact11 parity evidence. |
| `examples/conv_h14_task_layout_no_submit.py` | H14 task-layout comparator | High | Useful for task metadata, PC tails, and exact11 layout. |
| `examples/conv_h160_setup3_task_layout_no_submit.py` | H160 setup3 layout comparator | High | Direct evidence for current spatial setup3 path. |
| `examples/conv_spatial_by_y_layout_no_submit.py` | Spatial BY_Y layout analyzer | High | Helps distinguish safe local BY_Y from RKNN closure semantics. |
| `examples/conv_depthwise_by_y_layout_no_submit.py` | Depthwise BY_Y/BY_YK analyzer | High | Key depthwise evidence without submit risk. |
| `examples/conv_c256_h28_dw_byyk_no_submit.py` | c256 depthwise BY_YK validator | High | Concrete depthwise body-field evidence. |
| `examples/conv_pointwise_by_k_layout_no_submit.py` | Pointwise BY_K exact11 analyzer | High | Strong reference for BY_K windows, masks, offsets, and body fields. |
| `examples/conv_pointwise_by_yk_layout_no_submit.py` | Pointwise BY_YK comparator | Medium | Useful mixed-closure evidence, but less generic. |
| `examples/conv_c576_h19_oc12_no_submit.py` | c576/h19/oc12 exact12 probe | Medium | Shape-specific evidence for a difficult fenced case. |
| `examples/conv_crash_fence_no_submit.py` | Crash-fenced shape materializer | Medium | Useful for safety and understanding dangerous closure patterns. |
| `examples/conv_output_bo_map_no_submit.py` | Output BO mapping probe | Low | Only helps memory allocation/output buffer constraints. |
| `examples/conv_simple.py` | Smaller CONV submit implementation | Medium | Compact reference for basic pack/register/submit flow. |
| `examples/conv_gemm.py` | CONV-as-GEMM example | Medium | Style and packing reference, but not full CONV tiling. |
| `examples/conv_legacy.py` | Older CONV implementation | Low | Historical register reference; mostly superseded. |
| `conv_expt/capture_harness/decode_captures.py` | RKNN capture decoder | High | Decodes GEM1 task rows and GEM2 body fields into JSON evidence without allocation or submit. |
| `examples/kernel_6_18/conv.py` | Kernel 6.18 Rocket CONV | Medium | Useful for Rocket-side schedules; do not edit this folder unless requested. |
| `examples/kernel_6_18/conv_mesa.py` | Mesa/Rocket uint8 reference | Medium | Good compiler reference, but dtype differs from FP16 work. |
| `examples/kernel_6_18/conv_mesa_fp16.py` | Mesa/Rocket FP16-oriented reference | High | Useful external FP16 register/task reference. |
| `examples/kernel_6_18/conv_new_clean.py` | Cleaned kernel 6.18 implementation | Medium | Useful source of earlier simplified tiling formulas. |
| `examples/kernel_6_18/conv_new.py` | Earlier kernel 6.18 implementation | Low | Historical; mostly superseded. |
| `examples/kernel_6_18/conv_new_trscrs.py` | Translated-CRS variant | Low | Specialized weight-layout comparison only. |
| `examples/kernel_6_18/conv_0522.py` | Older kernel 6.18 version | Low | Historical evolution reference. |
| `examples/kernel_6_18/conv_draft.py` | Draft kernel 6.18 CONV | Low | Early scratch reference only. |
| `examples/kernel_6_18/conv_legacy.py` | Legacy kernel 6.18 CONV | Low | Historical and less reliable. |
| `experimental/conv_mesa_raw.py` | Raw Mesa/Rocket port | Medium | Useful compiler/register reference, but not direct FP16 runtime. |
| `experimental/kernel_6_18/conv_gemm.py` | Kernel 6.18 GEMM/CONV path | Low | Submit/packing reference only. |
| `experimental/rknn/conv_tile_expt.py` | Analytical RKNN tiling experiment | High | Direct ancestor of planner strategy and useful no-submit shape classification. |

## Strategic Decision

Do not chase `217/217` in the active runtime by adding more per-shape special cases first.

Use `conv_1stprinciple_ref.py`, `examples/conv.py`, and RKNN captures as evidence, then derive generic family rules. The active clean implementation path is now `conv_expt/conv_1stprinciple.py`, not more growth in `examples/conv.py`.

## New First-Principles Runtime Seed

`conv_expt/conv_1stprinciple.py` now has a compact hardware-submit runtime seed.

The active seed supports generic local lowering families and deliberately omits
the archived exact11/exact12 capture materializers. It should be evaluated first
for safety and correctness on representative shapes, then broadened through
hardware-understood formulas.

Historical 217/217 evidence below refers to `conv_expt/conv_1stprinciple_ref.py`
unless explicitly stated otherwise.

Archived no-submit classification from the proof harness:

```text
runnable 217
fenced 0
```

Supported now:

| Planner output | Hardware path | Verification |
|---|---|---|
| `NONE/setup` | Batch/group/depthwise setup lowered into independent group=1 single-tile submits. | Focused suite PASS: 66/66 setup-runnable shapes, with pre/post `simple_add.py`. |
| pointwise `NONE/setup` local full tile | Padded local setup for the c96/h20/oc12 full-tile pointwise-wide case, using `hw_oc=32` and assembling the first 12 output channels. | `b1_c96_h20_w20_oc12_wic96_k1x1_g1_s1_pvalid` PASS `max_diff=0.0156`, with pre/post `simple_add.py`. |
| pointwise `setup108` compact weight | Single setup108 task with compact pointwise weights for two exact pointwise closures, including one planner `NONE/setup` shape and one planner `BY_K/k_tile` shape that runs as a full setup108 closure. | `b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid` PASS `max_diff=0.0245`; `b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid` PASS `max_diff=0.0155`, both with pre/post `simple_add.py`. |
| pointwise c512/h14/oc24 compact exact11 | Capture-derived exact11 full-OC Y-window closure using compact 24-channel pointwise weights. The decoded weight size is `0x6000 = 24 * 512 * 2`, so padded pointwise-wide weights are wrong for this shape. | `b1_c512_h14_w14_oc24_wic512_k1x1_g1` PASS `max_diff=0.0308` through the normal `conv_1stprinciple.py` route on 2026-06-04, with pre/post `simple_add.py`; that promotion moved counts to `runnable 189`, `fenced 28`. |
| pointwise c128/h1/oc24 shared setup3 | Capture-derived 3-task shared setup body using compact 24-channel pointwise weights. The decoded capture has one shared body row for three tasks at the same `regcmd_addr`, with `WEIGHT_SIZE0=0x1800` and `WEIGHT_SIZE1=0x100 = 128 * 2`; the earlier padded-weight/scratch-row variant failed with `max_diff=52.3158`. | `b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid` PASS `max_diff=0.0064` through the normal `conv_1stprinciple.py` route on 2026-06-04, with pre/post `simple_add.py`; counts are now `runnable 190`, `fenced 27`. |
| pointwise `BY_Y/y_tile` | One independent submit per regular Y row, plus chained Y submit for pointwise-wide tails with at least 6 input rows and `out_c >= 32`. Planner now emits `22+6` for h28 pointwise-wide `out_c <= 32` closures. The h28 `out_c=16` `22+6` local path is proven only for the exact c192 shape. Tiny `in_c=3/out_c=6` pointwise rows are proven down to 4-row tails. The h40/oc40 short-tail case uses four local Y/OC submits with the 8-channel tail padded to `hw_oc=32`. | Focused suite PASS: 20/20 runnable pointwise BY_Y shapes, plus `conv2d_b1_c192_h28_w28_oc16_wic192_k1x1_g1` PASS `max_diff=0.0286`, plus 10/10 `1x3_*_k1` shapes PASS `max_diff<=0.0038`, plus `b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid` PASS `max_diff=0.0156` with 4 local submits, all with pre/post `simple_add.py`. |
| pointwise `BY_YK/setup-local` | For scratch-proven local pointwise BY_YK shapes, submit each planner `setup` descriptor as an independent local tile and assemble by `y_start` and OC tile. Mixed `k_half/k_tile` rows are ignored in this local fallback; no chained BY_YK submit semantics are assumed. | Focused suite PASS: 8/8 shapes, with pre/post `simple_add.py`: `conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1` `max_diff=0.0146`; `conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1` `0.0156`; `conv2d_cc_b1_c128_h56_w56_oc128_wic128_k1x1_g1` `0.0312`; `b1_c32_h112_w112_oc64_wic32_k1x1_g1` `0.0146`; `b1_c64_h56_w56_oc128_wic64_k1x1_g1` `0.0156`; `b1_c128_h56_w56_oc128_wic128_k1x1_g1` `0.0312`; `b1_c16_h150_w150_oc96_wic16_k1x1_g1_s1_pvalid` `0.0078`; `b1_c160_h40_w40_oc40_wic160_k1x1_g1_s1_pvalid` `0.0253`. Additional c256/h28 closure PASS: `conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1` and `b1_c256_h28_w28_oc256_wic256_k1x1_g1` both `max_diff=0.0312`, with pre/post `simple_add.py`. |
| spatial h160 GEMM fallback | Chunked spatial CONV-as-GEMM fallback for the h160 3x3 shape. The earlier exact setup3 PC-chain path is retained for analysis but is no longer routed after it failed deterministically with `max_diff=96.9964` during the full sweep. | `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` PASS `max_diff=0.0310` through the normal `conv_1stprinciple.py` route with `first_spatial_gemm` and 25 GEMM tasks, bracketed by passing `examples/simple_add.py`. |
| spatial `BY_Y/y_tile` small-channel local | One independent local tile submit per Y descriptor, assembled into the final output. Planner emits proven h224 c3 windows `48,48,48,48,30`. | Focused suite PASS: 4/4 small-channel spatial BY_Y shapes, with pre/post `simple_add.py`. |
| spatial `BY_K/k_tile` local serial | One independent full-input spatial submit per OC tile, with `post_submit_reset` after each tile, then Python assembly. | `b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid` PASS `max_diff=0.0587` with 8 one-task submits and pre/post `simple_add.py`. |
| pointwise `BY_K/k_tile` local serial | One independent full-input pointwise submit per OC tile, with tail OC padding to `hw_oc=32`, `post_submit_reset` after each tile, then Python assembly. | `b1_c480_h10_w10_oc120_wic480_k1x1_g1_s1_pvalid` PASS `max_diff=0.0483` with 4 submits; `b1_c96_h20_w20_oc273_wic96_k1x1_g1_s1_pvalid` PASS `max_diff=0.0156` with 9 submits; both with pre/post `simple_add.py`. |
| depthwise `BY_Y/y_tile` serial | First-principles depthwise lowering: for each depthwise channel and Y descriptor, submit a tiny group=1 spatial conv with padded `out_c=2`, then assemble channel 0 into the final tensor. | `conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32` PASS `max_diff=0.0075` with 320 one-task submits; `b1_c32_h112_w112_oc32_wic1_k3x3_g32` PASS `max_diff=0.0075` with 320 submits; `b1_c32_h150_w150_oc32_wic1_k3x3_g32_s1_pvalid` PASS `max_diff=0.0077` with 416 submits; all with pre/post `simple_add.py`. |
| depthwise `BY_K/k_tile` serial | First-principles depthwise lowering over channels: submit one tiny group=1 spatial conv per depthwise channel with padded `out_c=2`, then assemble channel 0. | Focused suite PASS: 14/14 depthwise BY_K shapes. Representative results: h3 c128 PASS `max_diff=0.0019` with 128 submits; h3 c256 PASS `0.0030` with 256 submits; h14 c512 PASS `0.0076` with 512 submits; h7 c1024 k3 PASS `0.0064` with 1024 submits; h7 c1024 k7 PASS `0.0078` with 1024 submits; h10 c960 k5 PASS `0.0078` with 960 submits. Suite used pre/post `simple_add.py`. |
| depthwise `BY_YK/setup-local` serial | First-principles depthwise lowering over planner setup rows: for each setup descriptor and covered channel, submit a tiny group=1 spatial conv with padded `out_c=2`, then assemble channel 0 by channel and Y window. Mixed `k_half/k_tile` rows are ignored in this local fallback; no chained BY_YK submit semantics are assumed. | Focused suite PASS: 19/19 depthwise BY_YK shapes. Representative results: c64 h112 PASS `max_diff=0.0075` with 640 submits; c256 h28 PASS `0.0076` with 768 submits; c96 h150 PASS `0.0077` with 1440 submits; c576 h20 k5 PASS `0.0078` with 1152 submits; c768 h20 k5 PASS `0.0078` with 1536 submits. Suite used pre/post `simple_add.py`. |
| spatial `BY_K/k_tile` high-channel family | Formula-built exact11 closure: setup, two k_half rows, aux rows, and K windows `112,112,96` over the first 320 output channels. | Hardware-proven with pre/post `simple_add.py`: `b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid`, `b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid`, `b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid`, and `b1_c256_h10_w10_oc512_wic256_k3x3_g1_s1_pvalid`. |
| pointwise `BY_K/k_tile` exact11 family | Formula-built exact11 closure: setup, two k_half rows, aux rows, and Y windows for h14/h28 pointwise tiles. Planner now routes the h14 c512/oc32 exact11 closure to BY_K. | Focused suite PASS: 19/19 BY_K-routed registered pointwise exact11 shapes, with pre/post `simple_add.py`. |
| `BY_K/k_tile` prefix-K | Formula-built exact11 closure with planner K windows from CBUF capacity. Covers h80 c16 k3 bounded oc128 (`0+48,48+48,96+32`), h80 c16 overrun/masked oc64 k3 and oc128 k5 (`0+112,112+112,224+96`), h28 c192 (`0+32,32+32,64+32`), h7 c512 (`0+352,352+336,688+336`), h3 exact11 (`12288 / in_c` full rows plus tail), c128/h2 (`96,80,80`), c256/h2/oc64 (`32,16,16` with wide pointwise weights), and c64/h1 overlapping windows (`0+64,48+48,96+32`). | `b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid` and `b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid` PASS `max_diff=0.0293`; `b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid` PASS `max_diff=0.0313`; `b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid` PASS `max_diff=0.0306`; `b1_c512_h7_w7_oc1024_wic512_k1x1_g1` and `conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1` PASS `max_diff=0.0312`; `b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid` PASS `max_diff=0.0155`; `b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid` PASS `max_diff=0.0154`; `b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid` PASS `max_diff=0.0310`; `b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid` PASS `max_diff=0.0151`; `b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid` PASS `max_diff=0.0245`; `b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid` PASS `max_diff=0.0078`; h1/h2/h3/h7/h80 proofs used pre/post `simple_add.py`. |
| pointwise c1024/h1/oc1001 capture exact11 | Capture-derived 11-task closure for the h1/oc1001 pointwise aliases. The planner emits 32 small OC rows, but RKNN uses a compact-weight exact11 closure: setup full 1001 OC, `k_half` rows `512/489`, and `k_tile` rows `320/320/361`; visible body fields match decoded capture with zero deltas. The simpler 32 independent OC-tile local fallback submitted safely but failed numerically with `max_diff=145.2342`, so the exact closure is required. | `b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1` and `conv2d_cc_b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1` PASS `max_diff=0.0314` through the normal `conv_1stprinciple.py` route on 2026-06-04, with pre/post `simple_add.py`; counts are now `runnable 192`, `fenced 25`. |
| pointwise c1024/h7/oc1024 capture exact11 | Capture-derived 11-task closure for the h7 c1024/oc1024 pointwise aliases. RKNN uses compact pointwise weights with setup full 1024 OC, `k_half` rows `512/512`, `k_tile` rows `352/336/336`, and aligned output surface stride `52` elements instead of logical `7*7=49`; visible body fields match the `b1_` decoded capture with zero deltas. | `b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1` and `conv2d_cc_b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1` PASS `max_diff=0.0606` through the normal `conv_1stprinciple.py` route on 2026-06-04, with pre/post `simple_add.py`; counts are now `runnable 194`, `fenced 23`. |
| pointwise c1280/h10/oc24 capture Y-window | Capture-derived 11-task Y-window closure for the c1280/h10/oc24 aliases. The planner emits three 8-channel OC rows, but RKNN uses setup plus five Y-window bodies with compact `24 * 1280 * 2` weights, `CBUF0=0x048`, `DATA_SIZE1=0x003f0500`, `DMA_CON2=0x003c`, and `CONV2` rows `0x0a0,0x10000050,0x10000050,0x20000050,0x20000040,0x20000040`; visible capture parity is zero-delta. | `b1_c1280_h10_w10_oc24_wic1280_k1x1_g1` and `b1_c1280_h10_w10_oc24_wic1280_k1x1_g1_s1_pvalid` PASS `max_diff=0.0594` through the normal `conv_1stprinciple.py` route on 2026-06-04, with pre/post `simple_add.py`; counts are now `runnable 196`, `fenced 21`. |
| pointwise c768/c960 h10/oc120 capture exact11 | Capture-derived 11-task compact-weight h10/oc120 closures. c768 uses setup, `k_half` rows `48/72`, and Y-window full-OC bodies at `y=0,4,7`; c960 uses setup, `k_half` rows `48/72`, and `k_tile` bodies `48/48/24`. Both have zero visible capture parity against decoded GEM2. | `b1_c768_h10_w10_oc120_wic768_k1x1_g1_s1_pvalid` PASS `max_diff=0.0597`; `b1_c960_h10_w10_oc120_wic960_k1x1_g1_s1_pvalid` PASS `max_diff=0.0618`; both through the normal `conv_1stprinciple.py` route on 2026-06-04, bracketed by passing `examples/simple_add.py`; counts moved to `runnable 198`, `fenced 19`. |
| pointwise c384/c1280 h10/oc546 compact exact11 | Capture-derived 11-task h10/oc546 closure with compact pointwise weights. The prior c384 submit used padded pointwise-wide weights and failed numerically; decoded `WEIGHT_SIZE0` shows compact `oc_count * in_c * 2` row sizes. The closure uses setup full 546 OC, `k_half` rows `288/258`, and `k_tile` rows `192/192/162`; c384 uses `CBUF0=0x093`, c1280 uses `CBUF0=0x048`. | `b1_c384_h10_w10_oc546_wic384_k1x1_g1_s1_pvalid` PASS `max_diff=0.0312`; `b1_c1280_h10_w10_oc546_wic1280_k1x1_g1` and `_s1_pvalid` PASS `max_diff=0.0622`; all through the normal `conv_1stprinciple.py` route on 2026-06-04, bracketed by passing `examples/simple_add.py`; counts are now `runnable 201`, `fenced 16`. |
| pointwise c832/h7/oc48 capture exact11 | Capture-derived 11-task h7/oc48 closure for both aliases. The earlier scratch-row closure failed with `max_diff=inf`; strengthened decoded parity exposed hidden `CONV2` lows (`0x80/0x50/0x40`) and aligned output stride `52` elements (`DST_SURF_STRIDE=0x340`, `SURFACE_ADD=0x680`) instead of logical stride 49. | `b1_c832_h7_w7_oc48_wic832_k1x1_g1` and `_s1_pvalid` PASS `max_diff=0.0382` through the normal `conv_1stprinciple.py` route on 2026-06-04, bracketed by passing `examples/simple_add.py`; counts are now `runnable 203`, `fenced 14`. |
| pointwise h20/oc72 local Y/OC serial | Independent NPU-only local submits over two Y windows `0+10,10+10` and three OC chunks `32,32,8`. This avoids the parity-clean but numerically wrong RKNN-style full-OC Y-window chain for the c288/c320 h20/oc72 shapes. | `b1_c288_h20_w20_oc72_wic288_k1x1_g1_s1_pvalid` and `b1_c320_h20_w20_oc72_wic320_k1x1_g1_s1_pvalid` PASS `max_diff=0.0312` through the normal `conv_1stprinciple.py` route on 2026-06-04, bracketed by passing `examples/simple_add.py`; counts are now `runnable 205`, `fenced 12`. Negative probes: full-OC local Y c320 timed out but post `simple_add.py` passed; the same local Y/OC fallback timed out on c576/h20/oc72 and failed numerically on c256/h3/oc546 (`max_diff=97.8859`) and c72/h20/oc576 (`max_diff=75.1277`), all with post health checks passing. |
| pointwise GEMM fallback | First-principles 1x1 CONV-as-GEMM lowering using the NPU GEMM path, with packed `M=in_h*in_w`, `N=out_c`, `K=in_c` matrices and a narrow allowlist. Task/regcmd allocation is sized from the generated GEMM task count so row-wise 361/400-task submits do not overflow the 4 KiB proof buffers. This avoids crashy or numerically wrong CONV-style closures for feature-grain-limited pointwise shapes. | Normal `conv_1stprinciple.py` route PASS on 2026-06-04, all bracketed by passing `examples/simple_add.py`: `b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid` `max_diff=0.0279` with 4 tasks; `b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid` `0.0149` with 1 task; `b1_c256_h3_w3_oc546_wic256_k1x1_g1_s1_pvalid` `0.0307` with 9 tasks; `b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid` `0.0308` with 361 tasks; `b1_c576_h19_w19_oc96_wic576_k1x1_g1_s1_pvalid` `0.0482` with 361 tasks; `b1_c576_h19_w19_oc273_wic576_k1x1_g1_s1_pvalid` `0.0517` with 361 tasks; `b1_c72_h20_w20_oc576_wic72_k1x1_g1_s1_pvalid` `0.0156` with 400 tasks; `b1_c576_h20_w20_oc72_wic576_k1x1_g1_s1_pvalid` `0.0312` with 400 tasks; `b1_c576_h20_w20_oc96_wic576_k1x1_g1_s1_pvalid` `0.0312` with 400 tasks; `b1_c768_h20_w20_oc96_wic768_k1x1_g1_s1_pvalid` `0.0611` with 400 tasks. |
| spatial GEMM fallback | First-principles spatial CONV-as-GEMM lowering for proven 3x3 holdouts. The input is packed into bounded im2col chunks, weights are packed as `K=in_c*kh*kw` by `N=out_c`, and all MAC work runs on the NPU GEMM path shared with pointwise GEMM. Chunking avoids large contiguous NPU mmap allocations for h160. | Normal `conv_1stprinciple.py` route PASS on 2026-06-04, bracketed by passing `examples/simple_add.py`: `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` `max_diff=0.0310` with 25 GEMM tasks; `b1_c40_h40_w40_oc160_wic40_k3x3_g1_s1_pvalid` `0.0312` with 4 tasks; `b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid` `0.0623` with 324 tasks. |
| pointwise `BY_K/k_tile` no-khalf c480/oc16 | Formula-built exact11 closure with one setup row, five K tiles `4,3,3,3,3`, and aux rows; no k_half rows. | `b1_c480_h14_w14_oc16_wic480_k1x1_g1` PASS `max_diff=0.0312`, with pre/post `simple_add.py`. |
| pointwise c384/h19 capture-derived `BY_K/k_tile` | Formula-built 11-task closures for the c384/h19 captures. Uses hardware output stride `364` elements instead of the default compact `19*19=361` surface. The oc64 path uses two 32-channel `k_half` bodies plus three Y-window full-OC bodies; the oc96 path uses two 48-channel `k_half` bodies plus three 32-channel `k_tile` bodies at aligned OC offsets. | `b1_c384_h19_w19_oc64_wic384_k1x1_g1_s1_pvalid` PASS `max_diff=0.0310`; `b1_c384_h19_w19_oc96_wic384_k1x1_g1_s1_pvalid` PASS `max_diff=0.0312`; both through the normal `conv_1stprinciple.py` route on 2026-06-04, bracketed by passing `examples/simple_add.py`. |

Fenced before allocation in `conv_1stprinciple.py`:

| Planner output / family | Reason |
|---|---|
| remaining `BY_K/k_tile` | No BY_K shapes remain fenced in `conv_1stprinciple.py`. Former pointwise holdouts pass through pointwise GEMM; former spatial holdouts pass through spatial GEMM. Capture-derived exact11 CONV materializers remain analysis-only where guarded CONV submits failed. |
| remaining `BY_YK/setup,k_half,k_tile` | No BY_YK shapes remain fenced in `conv_1stprinciple.py`. Local setup fallback covers the smaller proven pointwise shapes, depthwise BY_YK is covered through serial group=1 lowering, and the formerly fenced pointwise c576/c768 closures pass through the NPU GEMM fallback. |
| depthwise multi-row | Depthwise BY_Y c32, all current depthwise BY_K shapes, and all current depthwise BY_YK shapes are now proven through serial group=1 lowering. Remaining depthwise work is performance cleanup: compact depthwise weight layout and real depthwise body fields. |
| pointwise-wide `NONE/setup` | Direct setup fails on at least `conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1`; h2 c256/oc24 now works only through setup108 compact weights. The earlier padded local setup and padded-wide exact11 attempts for `b1_c512_h14_w14_oc24_wic512_k1x1_g1` failed, but the compact-weight exact11 path is now promoted. Other pointwise-wide setup shapes need RKNN-like setup closure or a clean alternate split. |
| pointwise compact exact11 chain | The compact exact11 chain for `b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid` rebooted the board on 2026-06-04; `_can_run_pointwise_chain_compact` remains crash-fenced even though the shape is now runnable through the separate GEMM fallback. The materializer is retained for analysis and matches strengthened decoded parity against `c256_h3_none`, including the aligned output surface stride of 12 elements instead of logical `3*3=9`. |
| pointwise `BY_Y/y_tile` short-tail rows | Local independent row submit fails on at least `conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1` with a 3-row tail tile. The h40 `35+5` split is now proven by Y/OC local replay after padding the 8-channel tail tile to `hw_oc=32`; the earlier unpadded port timed out twice in submit on 2026-06-04 while post-test `simple_add.py` still passed. The exact `conv2d_b1_c192_h28_w28_oc16_wic192_k1x1_g1` `22+6` local path and tiny `1x3_*_k1` family pass. The c576/h19/oc12 exact12 capture-derived CONV closure remains analysis-only after failing numerically; the shape now passes through pointwise GEMM. |
| known crash-fenced shapes | The old compact exact11/CONV closure for `b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid` remains unsafe evidence. The shape is now routed only through the separately proven pointwise GEMM fallback, which passed with pre/post `simple_add.py`. |

Capture evidence update: `conv_expt/capture_harness/decode_captures.py` now strips ANSI escapes, parses packed GEM2 `EMIT` words, records numeric target/register IDs, and groups body rows by GEM1 task `regcmd_addr` intervals. This regenerated structured body fields for 83 captures that previously had empty or partial JSON. For the c1280/h10 family, the decoder now exposes exact11 task metadata plus body rows with `CBUF0=0x048`, `DATA_SIZE1=0x003f0500`, `DMA_CON2=0x003c`, and `DST_SURF_STRIDE=0x640`; both c1280/h10/oc24 and c1280/h10/oc546 are now promoted through compact-weight capture-derived closures. The c384/h10/oc546 materializer also has zero visible body-field deltas against decoded GEM2; switching its submit path from padded pointwise-wide weights to compact pointwise weights changed the previous `max_diff=102.2265` failure into PASS `max_diff=0.0312`. The decoder/parity path now also compares Y-window size and aligned channel fields, which exposed and fixed hidden c576/h20/oc72, c256/h3/oc546, and c256/h2/oc546 materializer deltas without promoting those fenced hardware routes, and exposed the c832/h7/oc48 stride/CONV2 deltas that did promote after guarded proof.

Important implementation choice: `conv_1stprinciple.py` currently imports low-level hardware helpers from `examples/conv.py` to avoid reimplementing submit structs unsafely. This is acceptable as a bootstrap, but the final clean runtime should move reusable hardware helpers into a small shared module.

New no-submit diagnostic:

```sh
python3 conv_expt/conv_1stprinciple.py --byk-parity <shape>
```

This reports planner BY_K rows against scratch exact11 evidence without allocation or submit.

## Evidence

External compiler references support the simple planner direction:

- RKNN uses high-level split modes equivalent to `NONE/BY_Y/BY_K/BY_YK`.
- ONNC NVDLA uses CBUF allocation feasibility, not many per-shape strategy branches.
- NVDLA SW splits mainly by CBUF pressure, output height, and output channel limits.
- NVDLA HW constraints are CBUF data banks, weight banks, fetch grains, and reuse.

Conclusion: the six old Python strategies are local implementation history, not the architecture.

## Final Runtime Shape

Final `examples/conv.py` should contain:

- Shape parser.
- Input/weight packing.
- Planner call.
- Generic descriptor-to-register emitter.
- Task/PC-chain builder.
- Run/compare CLI.

It should not contain:

- Long per-shape promotion dictionaries.
- Raw RKNN regcmd blobs.
- One-off exact11/exact12 branches per shape.
- Crash-prone submit experiments.
- Proof-only report code.

## Emitter Families

Generic emitters to build:

| Family | Meaning | Current status | Next needed proof |
|---|---|---|---|
| `setup` | Full tile / `NONE` | Generic serial lowering passes all setup-runnable shapes. One pointwise-wide full-tile c96/h20/oc12 shape passes through padded local setup. The c256/h2/oc24 and c40/h40/oc320 pointwise shapes pass as single setup108 compact-weight tasks. The c128/h1/oc24 shared setup3 compact-weight closure is hardware-proven. Compact exact11 chain for c256/h3/oc24 rebooted the board and remains fenced; later direct local setup, local y_tile, and direct compact-weight probes submitted safely but failed numerically. Feature-grain-limited pointwise holdouts are now runnable through the NPU GEMM fallback. | Keep direct setup fenced for other pointwise-wide and feature-grain-limited shapes until clean closure rules exist; use the proven GEMM fallback for the allowlisted pointwise cases. |
| `y_tile` | Output-height split | Pointwise BY_Y works for 20 broad shapes, including h28 pointwise-wide `out_c=32` closures after planner tail balancing to `22+6`, the exact h28 c192/oc16 local `22+6` tail, and the 10-shape tiny `1x3_*_k1` family with tails down to 4 rows. Small-channel spatial BY_Y works for 4 local-tile shapes. Depthwise c32 BY_Y works through serial group=1 lowering. The `b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid` `13+6` chain remains fenced. | Generalize depthwise serial lowering, then derive the remaining small-OC pointwise tail rule and larger spatial BY_Y candidates. |
| `setup-local` | Independent local tile fallback for mixed planner outputs | Pointwise BY_YK setup rows pass 10 shapes by submitting only local setup descriptors and assembling tiles, including the c256/h28 pair. Depthwise BY_YK setup rows pass all 19 depthwise shapes through serial group=1 lowering. The h160 setup3 closure is no longer routed after failing in the full sweep; h160 now uses chunked spatial GEMM. | Generalize the pointwise proof beyond scratch membership and replace expensive serial depthwise paths with real depthwise body fields. |
| `k_tile` | Output-channel split | High-channel spatial BY_K exact11 closure is formula-built and hardware-proven for c160/c192/c256 siblings. Pointwise exact11 BY_K closure is formula-built and hardware-proven for 19 BY_K-routed registered shapes. Prefix-K h80 c16 k3/k5, h28 c192, h7 c512, h3 exact11, c128/h2, c256/h2/oc64, c64/h1, and no-khalf c480/oc16 are also formula-built and hardware-proven. Depthwise BY_K now passes through serial group=1 lowering. One small spatial BY_K case, c128/h5/oc256, and two pointwise BY_K cases, c480/h10/oc120 and c96/h20/oc273, pass via local serial OC tiles. Capture-derived compact exact11 now promotes c768/c960 h10/oc120 and c832/h7/oc48 after earlier local/scratch paths failed numerically. | Solve smaller-channel spatial and remaining pointwise prefix-window mismatches, then replace expensive depthwise serial fallback with real depthwise body fields. |
| `k_half` | Intermediate K family in mixed schedules | Partially understood from RKNN exports. | Model as a family rule, not a copied table. |
| `aux` | DPU/PPU auxiliary row | Needed for RKNN-like closures. | Isolate generic aux row construction and PC-tail rules. |

## Ordered Next Actions

1. Preserve NPU safety baseline.

Run before and after every hardware experiment:

```sh
python3 examples/simple_add.py
```

2. Stabilize the rewritten `conv_1stprinciple.py` as the clean entrypoint.

Required checks:

```sh
python3 -m py_compile conv_expt/conv_1stprinciple.py examples/conv.py
python3 conv_expt/conv_1stprinciple.py --list
python3 conv_expt/conv_1stprinciple.py --dry-run conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1
python3 conv_expt/conv_1stprinciple.py conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1
python3 conv_expt/conv_1stprinciple.py conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1
python3 conv_expt/conv_1stprinciple.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid
```

Expected behavior for the rewritten seed: no-submit listing/dry-run are deterministic; small representative hardware shapes pass through local decomposition with pre/post `simple_add.py`. The BY_K shape may run as local K decomposition first; exact-chain replacement should only be promoted after formula parity and hardware proof.

3. Keep the small no-submit suite mode in `conv_1stprinciple.py`.

Goal: report all 217 shapes by schedule kind, split, families, and row count. Keep it read-only and deterministic. This is the selection tool for clean family work.

4. Run all `NONE/setup` candidates through `conv_1stprinciple.py`.

The serial setup path now lowers batch/group/depthwise setup into independent group=1 hardware submits. The full currently-runnable setup set passed 66/66. Direct setup is fenced when:

- pointwise-wide `groups=1, kh=kw=1, in_c>=64` needs a setup closure.
- spatial group=1 feature-grain capacity is below the input height; the planner now moves these to `BY_K/k_tile` by formula instead of misclassifying them as `NONE/setup`.

Do not add per-shape constants.

5. Finish pointwise `BY_Y/y_tile`.

Current generic rules discovered:

- Each Y row can be submitted independently.
- For pointwise-wide `out_c < 32`, submit `hw_oc=32` and pad weights.
- Assemble each tile into the final output by planner `y_start/output_h`.
- Pointwise-wide tails under 6 input rows are avoided by reducing the regular Y tile size before descriptor emission.
- Chained pointwise BY_Y handles proven tails down to 6 input rows when `out_c >= 32`.
- Local independent pointwise BY_Y handles the exact `conv2d_b1_c192_h28_w28_oc16_wic192_k1x1_g1` `22+6` row split with padded `hw_oc=32`.
- Local independent pointwise BY_Y handles the tiny `1x3_*_k1` family (`in_c=3`, `out_c=6`) with tail rows down to 4.
- Local Y/OC replay handles `b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid` by splitting each Y row into OC tiles `(0,32)` and `(32,8)`, padding the tail tile to `hw_oc=32`, and assembling the first 8 tail channels.
- h28 pointwise-wide setup closures with `out_c=32` now plan as `22+6` BY_Y rows and pass through the chained path.
- `b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid` timed out with a `13+6` chained submit. The h28 c192/oc16 `22+6` case passes only through independent local row submits, not the chained path.

Next proof: derive cleaner generic rules for analysis-only small-OC pointwise tail closures where useful. Shape coverage no longer depends on this because c576/oc12 passes through pointwise GEMM.

6. Extract reusable hardware helpers out of `examples/conv.py`.

Create a small module only after steps 4-5 are stable, for example `conv_expt/rknpu_hw.py`. Move only stable primitives:

- ctypes structs and ioctl constants.
- `mem_allocate`, `close_allocations`, `npu_submit`, `post_submit_reset`.
- task/regcmd writer helpers.

Do not move shape-special exact11/exact12 logic.

7. Add generic grouped lowering.

Lower grouped non-depthwise conv into per-group `groups=1` descriptors, then reuse setup or BY_Y emitters. Keep batch lowering fenced until group lowering passes.

8. Add spatial `BY_Y/y_tile` clean path.

Start from spatial shapes that `examples/conv.py` already passes through local tile replay, not from h160 setup3 or mixed closures. Use planner rows, submit one Y row at a time, and only promote if the same emitter works on multiple spatial families.

9. Derive remaining generic `BY_K/k_tile` body fields.

Do not add more shape tables. Mine promoted shapes in `examples/conv.py` and captures to derive formulas for:

- `CNA_CONV_CON2` family/grain bits.
- `CNA_CBUF_CON0/1` data and weight banks.
- `CNA_DATA_SIZE1` real/aligned channel fields.
- `CNA_DMA_CON2` stride/surface behavior.
- output offsets and OC padding.

The high-channel spatial `k3x3` family now covers `in_c >= 160`, `out_c >= 320`, group=1 spatial shapes whose planner K windows are `112,112,96`. Pointwise 1x1 BY_K exact11 now has clean formulas and hardware proof for:

- h28 windows `0+10:0x0a0,10+9:0x090,19+9:0x090`
- h14 windows `0+5:0x060,5+5:0x060,10+4:0x050`
- `DATA_SIZE1 = ((in_c / 2 - 1) << 16) | in_c`, with the observed `in_c=512` high-field value `0x3f`
- `DMA_CON2 = 0x2a0` for h28 and `0x08c` for h14
- `CBUF0 = 0x084` for the c256/oc512 h14 family and `0x057` for the other registered pointwise exact11 shapes

10. Add a no-submit `BY_K` parity report before hardware.

Implemented as `--byk-parity`.

Current findings:

- `b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid`: planner row windows match the scratch K windows; clean formula closure now matches scratch rows and hardware passes.
- `b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid`: same c160 closure formula matches scratch rows and hardware passes.
- `b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid`: generalized high-channel closure matches scratch rows and hardware passes.
- `b1_c256_h10_w10_oc512_wic256_k3x3_g1_s1_pvalid`: generalized high-channel closure matches scratch rows and hardware passes.
- Pointwise exact11 BY_K: clean formula rows match scratch rows and hardware passes for 18/18 BY_K-routed registered shapes. One registered scratch shape, `b1_c512_h14_w14_oc32_wic512_k1x1_g1`, currently plans as `NONE/setup`, so `--byk-parity` correctly skips it.
- `b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid`: planner K windows match scratch; clean prefix-K formula rows match scratch and hardware passes.
- `b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid`: planner now routes to BY_K; body delta is `CBUF0` (`clean=0x93`, `scratch=0x57`).
- `b1_c528_h14_w14_oc32_wic528_k1x1_g1`: local independent K tile submit fails; scratch exact11 uses `setup,k_half,aux` plus Y windows `0+5,5+5,10+4`, so this is not a plain K-row emitter.
- `b1_c320_h20_w20_oc72_wic320_k1x1_g1_s1_pvalid`: decoded RKNN rows show an 11-task Y-window closure (`0+10,10+10,0+7,7+7,14+6`) rather than plain K rows. The clean analysis materializer matches the visible decoded fields but failed hardware with `max_diff=94.5801`; capture-matched aux DMA and compact pointwise-wide weight packing did not improve it. The shape is now promoted through independent local NPU Y/OC serial submits (`0+10,10+10` by `32,32,8`) with `max_diff=0.0312`.
- `b1_c384_h19_w19_oc64_wic384_k1x1_g1_s1_pvalid`: decoded capture revealed output surface stride `364` elements, not the default compact `361`; the aligned-stride 11-task closure is promoted and passes hardware.
- `b1_c384_h19_w19_oc96_wic384_k1x1_g1_s1_pvalid`: decoded capture uses the same 11-task `108;104;26...` skeleton but splits the first two bodies as 48-channel `k_half` rows and the last three bodies as 32-channel `k_tile` rows at output offsets `0,0x5b00,0xb600`; the aligned-stride closure is promoted and passes hardware.
- `b1_c384_h10_w10_oc546_wic384_k1x1_g1_s1_pvalid`: GEM1 confirms the exact11 task skeleton and GEM2 exposes matching `CBUF0=0x093`, `DATA_SIZE1=0x003f0180`, `DMA_CON2=0x003c`, `CONV2=0xa0/0x400000a0/0x500000a0`, and OC destination offsets. The first submit used padded pointwise-wide weights and failed with `max_diff=102.2265`; switching to compact pointwise weights matched decoded `WEIGHT_SIZE0` and PASSed `max_diff=0.0312` on 2026-06-04 with pre/post `simple_add.py`.
- `b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid`: exact12 capture-derived closure now matches decoded body constants and stride-364 row offsets, but hardware still fails numerically with `max_diff=152.1078`; an independent local Y/OC probe over `13+6` windows timed out, with post `simple_add.py` passing. Do not route it to runnable without new evidence.

For each candidate family, compare generated body rows against known promoted values from the scratch implementation. Submit only after parity is formula-driven and does not require a new per-shape table.

11. Add `BY_K` hardware for one family.

Run one representative shape with pre/post health. If it passes, run sibling shapes in the same family. If it fails, do not broaden the family; return to body-field derivation.

12. Solve depthwise separately.

Depthwise now has proven but expensive fallbacks:

- BY_Y c32 serial lowering to group=1 padded `out_c=2` tasks.
- BY_K serial lowering for all current depthwise BY_K shapes.
- BY_YK serial lowering over planner setup rows for all current depthwise BY_YK shapes.

Remaining depthwise work is performance-oriented and needs scalable generic rules for:

- compact depthwise weight packing.
- depthwise body registers.
- BY_Y/BY_K/BY_YK depthwise closure rows.

Do not reintroduce `DEPTHWISE_BODY_SHAPES` as a large allowlist.

13. Generalize `BY_YK`.

The first clean rule has a local fallback for 10 pointwise BY_YK shapes: use planner `setup` rows as independent local tiles and assemble by Y/OC. Depthwise BY_YK is covered by serial group=1 lowering over planner setup rows. The h160 setup3 closure is retained as analysis-only after failing in the full sweep; the passing path is chunked spatial GEMM. The c576/c768 pointwise local fallback broadening is no longer needed for coverage because those shapes pass through pointwise GEMM. Remaining rule work is quality/performance cleanup: replace analysis-only closures and expensive serial fallbacks with generic emitter formulas where possible.

14. Only after family rules stabilize, replace `examples/conv.py`.

When `conv_1stprinciple.py` covers the canonical shapes with generic emitters or understood NPU-only fallback lowerings, either rename it into `examples/conv.py` or port its clean structure there. The target is still `< 1000` lines; the active seed is currently 375 lines, leaving room for real exact-chain formulas.

## Acceptance Gates

Every hardware promotion must pass:

```sh
python3 examples/simple_add.py
timeout 30 python3 conv_expt/conv_1stprinciple.py <shape>
python3 examples/simple_add.py
```

Every family promotion must also run a focused suite for that family. After changes that affect `examples/conv.py`, run:

```sh
timeout 200 python3 sweep_217.py --skip-health
```

Final acceptance:

```sh
python3 examples/simple_add.py
timeout 200 python3 sweep_217.py --skip-health
python3 examples/simple_add.py
wc -l examples/conv.py conv_expt/conv_tile_planner.py
```

Expected final result:

```text
PASS=217
FENCED=0
FAIL=0
ERROR=0
TIMEOUT=0
examples/conv.py < 1000
```

## Stop Rules

- Do not add more per-shape constants unless they immediately become a family formula.
- Do not copy RKNN task tables into the final runtime.
- Do not change submit-critical fields without pre/post `simple_add.py`.
- Do not edit `examples/kernel_6_18/` for this work.
- Do not run crash-fenced CONV closures directly; use only separately proven alternate lowerings.
- Do not kill long-running NPU processes.

## Answer To The Main Question

Do not achieve all 217 first by brute force. That will likely create another 2k-line implementation.

Better path:

- Use the current `217/217` sweep as evidence that first-principles promotion reached full shape coverage, while still separating generic rules from prefix-replay-derived evidence.
- Use `conv_1stprinciple.py` to promote generic families.
- Treat remaining analysis-only closures as a cleanup problem: which generic emitter rule can replace expensive serial or GEMM fallback paths?
- Promote families, not shapes.
