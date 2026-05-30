# conv_expt Evidence

This file replaces the old progress logs, chat transcripts, and huge decomp note
dumps. It keeps only the facts that currently matter for the CONV planner work.

## Shape Set

Source: `examples/kernel_6_18/conv_new.py`

Total literal test shapes: `217`

The old runtime branch grouping is:

| Old strategy / path | Count | Share |
|---|---:|---:|
| `pointwise_oc_tile` | 87 | 40.1% |
| fallback/direct path | 54 | 24.9% |
| `depthwise_spatial_tiled` | 36 | 16.6% |
| `grouped_serial` | 15 | 6.9% |
| `spatial_oc_serial` | 12 | 5.5% |
| `spatial_im2col` | 11 | 5.1% |
| `pointwise_y_tile_hardcoded` | 2 | 0.9% |

Fallback/direct path detail:

| Subtype | Count | Notes |
|---|---:|---|
| true direct fallback | 43 | no explicit tiling branch |
| flat-output Y tile fallback | 11 | large small-channel `1x1`, mostly `ic=3, oc=6`, `52x52..72x72` |

## Rare Shapes

`pointwise_y_tile_hardcoded`: 2 shapes.

| Shape | Input | Output | Kernel |
|---|---|---|---|
| `conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1` | `144x28x28` | `32x28x28` | `1x1` |
| `conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1` | `192x28x28` | `32x28x28` | `1x1` |

`spatial_im2col`: 11 shapes.

| Shape | Input | Output | Kernel |
|---|---|---|---|
| `b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid` | `160x14x14` | `320x12x12` | `3x3` |
| `b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid` | `160x7x7` | `320x5x5` | `3x3` |
| `b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid` | `192x7x7` | `384x5x5` | `3x3` |
| `b1_c256_h10_w10_oc512_wic256_k3x3_g1_s1_pvalid` | `256x10x10` | `512x8x8` | `3x3` |
| `b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid` | `128x5x5` | `256x3x3` | `3x3` |
| `b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid` | `128x3x3` | `256x1x1` | `3x3` |
| `b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid` | `3x320x320` | `32x318x318` | `3x3` |
| `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` | `16x160x160` | `128x158x158` | `3x3` |
| `b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid` | `16x80x80` | `128x76x76` | `5x5` |
| `b1_c40_h40_w40_oc160_wic40_k3x3_g1_s1_pvalid` | `40x40x40` | `160x38x38` | `3x3` |
| `b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid` | `72x20x20` | `288x18x18` | `3x3` |

## RK3588 CBUF Constants

Sources:

- `ref/mesa/src/gallium/drivers/rocket/rkt_ml.h`
- `ref/rk3588-npu/include/npu_hw.h`

```c
#define CBUF_BANK_SIZE        32768
#define CBUF_BANKS            12
#define CBUF_ENTRIES_PER_BANK 256
#define CBUF_ENTRY_SIZE       (CBUF_BANK_SIZE / CBUF_ENTRIES_PER_BANK)

#define NPU_CBUF_BANK_SIZE 32768
#define NPU_CBUF_BANKS 12
```

Computed values:

| Target | Banks | Per bank | Total CBUF | Entry size | Entries/bank |
|---|---:|---:|---:|---:|---:|
| RK3588 / Rocket | 12 | 32 KiB | 384 KiB | 128 B | 256 |
| NVDLA `nv_full` | 16 | 32 KiB | 512 KiB | config width/depth differs | 512 config-depth units |

Implication: RK3588 has 25% less CBUF than NVDLA `nv_full`. ONNC no-split on
`nv_full` does not prove no-split on RK3588.

## ONNC / NVDLA Findings

ONNC evidence:

- Uses CBUF allocation classes, not our six Python branches.
- Important concepts: `getCbufAllocType`, `tryAllocateDataAndWeightsIntoCBuf`,
  `SplitConvPass`, `SplitGroupConvPass`, `NvDlaBackend::getMaxNumOfConvChannels`.
- Handles grouped conv by graph split into group-1 convs plus concat.
- Handles large input-channel conv by splitting input channels and summing.
- Useful as compiler design evidence, but not exact RK3588 behavior because ONNC
  targets NVDLA configs such as `nv_full`.

NVDLA SW evidence:

- Planning is in UMD compiler, mainly `ConvCoreNode::splitNodes()` and
  `splitNodesInternal()`.
- Decision order is CBUF-fit based: full data/full weight, hardware split-K
  modes, partial-H split, and final unsupported/error cases.
- KMD/firmware consumes planned descriptors; it does not invent tiling.

ONNC VP observed shape results:

| Shape | CONV ops | SDP ops | Split evidence |
|---|---:|---:|---|
| `conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1` | 1 | 1 | no ONNC software tiling |
| `conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1` | 1 | 1 | no ONNC software tiling |
| `conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1` | 2 | 2 | BY_Y, captured as `45 + 11` output rows |
| `conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1` | 2 | 2 | likely BY_Y |
| `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` | 2 | 2 | BY_Y |
| `b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid` | 7 | 7 | strong spatial split |

Captured BY_Y example:

- Shape: `conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1`
- Runtime result: `Test pass`
- Output rows: `45 + 11 = 56`
- Artifacts:
  `ref/onnc-tutorial/models/conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1/capture.log`,
  `parsed_capture.txt`, and `parsed_loadable.txt`.

## RKNN / librknnrt Findings

Source: `experimental/rknn/librknnrt_conv_channel_tile_decomp.md`

Critical conclusion: RKNN does not use our six Python strategy names. The useful
reverse-engineering result is that RKNN has one conv tiler with Y/K-style split
records plus reuse/group bookkeeping.

RKNN pipeline model:

```text
layer attrs
  -> build tile seed
  -> choose NONE/BY_Y/BY_K/BY_YK split
  -> build Y/K tile vectors
  -> emit ABC_T/BAC or KC_T register tasks
  -> extract tile parameters by mode
  -> patch reuse/DMA command values
  -> validate CBUF bank/reuse fields
  -> PC-chain tasks
```

Important correction from the decomp note:

- The high-level split enum is `0=NONE`, `1=BY_Y`, `2=BY_K`, `3=BY_YK`.
- Some values passed near `fcn.005f1998` are value-reader modes, not strategy
  enums. They select fields from a tile seed, such as 8/16/32/64-channel
  parameter groups.
- `fcn.005f38f0` has grouping behavior for method `2`, method `3`, and generic
  `N`, but those are split-vector grouping patterns, not our Python strategy
  names.

Important strings and functions from `librknnrt.so`:

| Evidence | Meaning |
|---|---|
| `min_weight_banks > 3 && m_Target == RKNNTarget::RKNPU_F2, do conv with ChannelTile` | RK3588/F2 switches to ChannelTile when minimum weight-bank pressure exceeds 3 banks. |
| `Conv min_weight_banks > 3, OutputName : %s` | threshold participates in conv planning diagnostics. |
| `fcn.00387e3c`, `fcn.00388a18` | ChannelTile predicates. They compute data/weight bank pressure and apply target threshold tables. |
| `fcn.00384ed0` | minimum weight-bank estimator. It converts weight work to CBUF pressure and participates in the F2 `>3` threshold. |
| `fcn.00383338` | data-tile estimator. Its result converts spare CBUF banks into a legal Y/data window budget. |
| `fcn.003873e8` | channel tile tuner. It halves/clamps channel tiles until the pressure fits target limits. |
| `fcn.00385988` | legal K/weight split helper. It returns a target-legal K step or `-1` when below the hardware minimum. |
| `fcn.005d4ac0` | layer-level conv split planner. Builds `xstart/ystart/kstart`, Y/K steps, bank counts, reuse flags, and MC treatment fields. |
| `fcn.005f38f0` | K/X split-vector composer. Dispatches `NONE/BY_Y/BY_K/BY_YK`-style vectors and rejects illegal methods. |
| `fcn.005f51c0` | top-level Y/K/multicore tiler. Checks X overflow, Y config generation, MC Y/K legality, and per-core tile vectors. |
| `fcn.00384880` | parameter extraction dispatch. Reads mode-specific tile parameters from seed/context fields. |
| `fcn.005a41f0`, `fcn.005a4e18` | ABC_T/BAC and KC_T/C1K1C2K2C3 register-emission families. Exact register names still require runtime trace. |
| `fcn.00307198`, `fcn.00312328` | post-planner data/weight reuse updater paths. |
| `fcn.002efd38`, `fcn.002efce0`, `fcn.00100a40` | command-value patchers. They patch the middle 32-bit value field of existing 64-bit regcmd entries. |
| `fcn.005c6b50` | workload bank/reuse validator. Checks `input_bank_num + weight_bank_num <= target banks` and reuse legality. |
| `fcn.005f1cd0` | tile-table formatter. Confirms tile rows carry `xstart`, `ystart`, `kstart`, reuse flags, and MC treatment fields. |
| CNA feature/weight/CSC group strings | RKNN tracks a separate resource-group mask beyond visible CBUF bank fields. |

Confirmed tile-record fields:

| Field | Meaning for our planner work |
|---|---|
| `xstart` | X/window start. Usually zero in current Python model, but present in RKNN records. |
| `ystart` | output/input Y-window start. |
| `kstart` | output-channel/kernel-window start. |
| `data_reuse` | adjacent tile can reuse feature/data window. |
| `weight_reuse` | adjacent tile can reuse weight window. |
| `input_bank_num` | per-tile feature CBUF bank allocation. |
| `weight_bank_num` | per-tile weight CBUF bank allocation. |
| `mc_treat_by_y_tile` | multicore Y distribution type. |
| `mc_treat_by_k_tile` | multicore K distribution type. |
| `mc_treat_by_1c_y_tile` | single-core fallback Y treatment. |
| `mc_treat_by_1c_k_tile` | single-core fallback K treatment. |

Reuse/patching finding:

- RKNN does not only set a visible `CNA_CBUF_CON0_WEIGHT_REUSE` bit. It patches
  pre-built command entries through indexed/list metadata.
- `fcn.00100a40` is exactly a `bfi x0, x1, #16, #32` style value patch, matching
  our 64-bit command encoding: target bits `63..48`, value bits `47..16`,
  register address bits `15..0`.
- Single-tile GDB trace proved `fcn.002efd38` patches DMA addresses for weight,
  internal, and data/task buffer phases. `fcn.002efce0` still needs a real
  multi-tile ChannelTile trace.

CNA group mask layout from the formatter:

| Bits | Meaning |
|---|---|
| `0..1` | CNA feature group0/group1 |
| `2..3` | CNA weight group0/group1 |
| `4..5` | CNA CSC group0/group1 |
| `6..7` | ACCU group0/group1 |
| `8..9` | DPU group0/group1 |
| `10..11` | PPU group0/group1 |
| `12..13` | DMA read/write error bits |

Descriptor-level contract extracted from RKNN exports:

| Descriptor field | Emitter use |
|---|---|
| `family_bits` | high bits of `CNA_CONV_CON2`, e.g. setup, `y_tile`, `k_half`, `k_tile`. |
| `grain_bits` | low bits of `CNA_CONV_CON2`; unresolved formula, must remain explicit. |
| `input_h`, `output_h`, `output_w`, `oc_count` | tile-local shape values; do not recompute only from full layer shape. |
| `feature_off` | added to input DMA for this descriptor. |
| `weight_off` | added to weight DMA for this descriptor. |
| `output_off` | added to output DMA for this descriptor. |
| `cbuf0` | CBUF bank/reuse field; high bank-selection bits can vary independently from `grain_bits`. |

Observed descriptor families:

| Family | `family_bits` | Evidence |
|---|---:|---|
| setup/full | `0x00000000` | full/simple run family. |
| `y_tile` | `0x20000000` | Y-window descriptors, including pointwise row splits. |
| `k_half` | `0x40000000` | half-K/channel family in mixed spatial schedules. |
| `k_tile` | `0x50000000` | K/output-channel tiled family. |

Mixed Y+K schedule finding:

- Official RKNN exports for `160->320 3x3` show independent Y and K windows.
- Descriptor order for the tested mixed spatial case is:

```text
for family in [setup, k_half, k_tile]:
  for k_window in family_k_windows:
    for y_window in y_windows:
      emit descriptor with additive offsets
```

- For that path, `feature_off` depends only on Y, `weight_off` depends only on K,
  and `output_off = output_k_base + output_y_base`.
- This is the strongest implementation hint: clean Python should produce
  independent Y/K window descriptors, then combine them into descriptor families
  with additive DMA offsets.

Mapping to current Python families:

| Current Python family | RKNN interpretation |
|---|---|
| direct/fallback | `NONE` single-tile descriptor. |
| `pointwise_oc_tile` | closest to RKNN ChannelTile / `BY_K`; Python lacks full reuse/group/emit state. |
| `pointwise_y_tile_hardcoded` | `BY_Y` or mixed `BY_YK`, likely descriptor-family driven. |
| `spatial_oc_serial` | Y/K tile records, but Python submits independently instead of RKNN PC-chain/reuse path. |
| `depthwise_spatial_tiled` | depthwise variant of Y/K records with channel granularity. |
| `spatial_im2col` | Python workaround; RKNN programs spatial conv directly or uses ConvStreaming. |
| `grouped_serial` | higher-level per-group lowering; the core planner handles one lowered group. |

Interpretation:

- RKNN supports the idea of `NONE/BY_Y/BY_K/BY_YK` planning.
- Naive OC slicing is not equal to full RKNN ChannelTile.
- The missing parts are likely KG/weight-start semantics, reuse-table patching,
  or CNA feature/weight/CSC group programming.
- The clean local target should be a planner that emits explicit descriptor
  records, not more strategy-specific submit loops.

Known RKNN gaps that should not be hidden:

- `ConvStreaming` (`fcn.002ba2c0` and variants) is a separate large-weight conv
  operator and is not reversed enough to implement.
- ABC_T/KC_T register emission structure is known, but exact target register
  destinations still need runtime trace.
- The producer/programmer of the CNA group mask is still unknown.
- `grain_bits` are not fully derived. They depend on template/Y-window/CBUF
  context and must remain an explicit descriptor field until solved.
- Multicore wrappers (`fcn.005d78e0`, `fcn.005d7bc8`) are not fully reversed.

## Hardware Proof Boundaries

Known good health check:

```sh
python3 examples/simple_add.py
```

Mainline/Rocket quick check:

```sh
python3 examples/kernel_6_18/simple_add.py
```

Important pass/fail facts preserved from the old logs:

| Experiment | Result | Current decision |
|---|---|---|
| Small safe default shapes in `examples/conv_tiles.py` | passed with pre/post `simple_add.py` | basic vendor-driver conv path usable |
| Case 14 broad 217-shape hardware run, `conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1` | timed out, post `simple_add.py` passed | do not rerun unchanged; broad hardware sweep is unsafe/noisy |
| H40 direct-spatial `single_stream` | completed but wrong result, `max_diff=311.1706` | do not promote old direct-spatial formula submit |
| C64/H56 sparse replay | can compute correct output but pollutes next PC job until timeout/reset | evidence only; not normal validation |
| tail-index sparse policy | crashed/rebooted board | do not recreate |
| IC-split plus DPU/RDMA add, narrow OC | passed for `512->24` and `1280->24` with IC chunks of 128 | promising replacement for some pointwise precision cases |
| IC-split large OC `1280->546` | passed with IC chunks of 256 and partial OC tile 16 | partial proof; needs full precision-risk sweep |
| direct forced weight-bank sweep | small 3-bank estimate shape passed down to 2 banks, failed at 1 | RK3588 has some weight-streaming tolerance |
| real dense overflow `160->320 3x3` direct weight streaming | failed | keep im2col fallback until KG/reuse is understood |
| naive direct ChannelTile small proxy `32->128 3x3` | per-submit OC tile works; PC-chain fails | output offset can work; chained changing-weight state is missing |
| naive direct ChannelTile real `160->320 3x3` | fails before output placement | naive OC slicing is insufficient |

## What This Means

Do next:

- Use `conv_tile_cpu.py` to prove the CBUF planner and generic CPU execution.
- Produce old-strategy versus `NONE/BY_Y/BY_K/BY_YK` cross-tab.
- Run VP only for targeted ambiguities.

Do not do next:

- Do not clean `examples/conv_tiles.py` broadly before the planner proof.
- Do not delete `spatial_im2col` from the hardware path yet.
- Do not promote IC-split or ChannelTile replacements without more sweeps.
- Do not rerun known timeout/crash probes unchanged.
