# CONV Tile Planner Result

Date: 2026-05-30

## Scope

Implemented offline planner reporting in `conv_expt/conv_tile_cpu.py` and a
no-submit hardware-side materialization scaffold in
`examples/conv_h14_k_tile_no_submit.py`. The registry-driven consumer scaffold
`examples/conv_tiles_no_submit.py` now exercises the same executable fixture
contracts without entering the submit path.

No NPU submit path is changed. The scaffold does not open the DRM device, does
not allocate NPU memory, does not write task structs, and does not call
`npu_submit`.

## Added Commands

```sh
python3 conv_expt/conv_tile_cpu.py --planner-report
python3 conv_expt/conv_tile_cpu.py --descriptor-dump [SHAPE]
python3 conv_expt/conv_tile_cpu.py --cross-tab
python3 conv_expt/conv_tile_cpu.py --cbuf-compare
python3 conv_expt/conv_tile_cpu.py --cbuf-compare-all
python3 conv_expt/conv_tile_cpu.py --generic-only
python3 conv_expt/conv_tile_cpu.py --evidence-check
python3 conv_expt/conv_tile_cpu.py --family-window-report
python3 conv_expt/conv_tile_cpu.py --family-coverage-report
python3 conv_expt/conv_tile_cpu.py --family-coverage-all
python3 conv_expt/conv_tile_cpu.py --pointwise-hardcoded-report
python3 conv_expt/conv_tile_cpu.py --k-tile-emitter-field-report
python3 conv_expt/conv_tile_cpu.py --h14-k-tile-trace-report
python3 conv_expt/conv_tile_cpu.py --h14-k-tile-emitter-diff
python3 conv_expt/conv_tile_cpu.py --h14-k-tile-no-submit-dry-run
python3 conv_expt/conv_tile_cpu.py --cna-group-mask-trace-report
python3 conv_expt/conv_tile_cpu.py --abc-kc-builder-dispatch-report
python3 conv_expt/conv_tile_cpu.py --difficult-shape-evidence-report
python3 conv_expt/conv_tile_cpu.py --targeted-vp-list-report
python3 conv_expt/conv_tile_cpu.py --unresolved-fence-report
python3 examples/conv_h14_k_tile_no_submit.py
python3 examples/conv_tiles_no_submit.py h14_k_tile
python3 examples/conv_tiles_no_submit.py h7_k_tile
python3 examples/conv_tiles_no_submit.py setup
python3 examples/conv_tiles_no_submit.py --list
python3 examples/conv_tiles_no_submit.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/conv_tiles_no_submit.py b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/conv_tiles_no_submit.py b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid
```

`--planner-report` reports one row per shape:

```text
name, old_strategy, split_method,
y_boundaries, k_boundaries,
descriptor_count,
descriptor_families,
unresolved_fields
```

`--descriptor-dump` reports descriptor rows with explicit unresolved values:

```text
family, family_bits, grain_bits,
y_start, input_h, output_h, output_w,
k_start, oc_count,
feature_off, weight_off, output_off,
input_bank_num, weight_bank_num, cbuf0,
data_reuse, weight_reuse,
semantic_status, rknn_executable_equivalent, unresolved_reason
```

## RK3588 Cross-Tab

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --cross-tab
```

Result:

| Split method | Descriptor families | Count | Old branches covered |
|---|---|---:|---|
| `BY_K` | `k_tile` | 72 | `depthwise_spatial_tiled:14`, `pointwise_oc_tile:49`, `spatial_im2col:9` |
| `BY_Y` | `y_tile` | 32 | `depthwise_spatial_tiled:3`, `fallback/direct:11`, `pointwise_oc_tile:13`, `spatial_im2col:2`, `spatial_oc_serial:3` |
| `BY_YK` | `setup;k_half;k_tile` | 35 | `depthwise_spatial_tiled:19`, `pointwise_oc_tile:16` |
| `NONE` | `setup` | 78 | `fallback/direct:43`, `grouped_serial:15`, `pointwise_oc_tile:9`, `pointwise_y_tile_hardcoded:2`, `spatial_oc_serial:9` |

All 217 shapes are classified into `NONE/BY_Y/BY_K/BY_YK`.

## RK3588 vs NVDLA `nv_full` CBUF

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --cbuf-compare
```

Profiles compared:

| Profile | Banks | Bank size | Total CBUF |
|---|---:|---:|---:|
| RK3588 / Rocket | 12 | 32 KiB | 384 KiB |
| NVDLA `nv_full` planner budget | 16 | 32 KiB | 512 KiB |

DeepWiki / `nvdla/hw` check: `nv_full` defines `NVDLA_CBUF_BANK_NUMBER=16`; generated large config uses bank width/depth parameters, while the C-model commonly uses 128-byte entries. For this local comparison the harness keeps RK/NVDLA C-model entry geometry and changes only bank count, because the goal is planner pressure comparison.

Summary:

```text
profiles: rk3588=12x32KiB, nvdla_full=16x32KiB planner budget
same=192 changed=25 total=217
```

Split transitions:

| RK3588 split | NVDLA `nv_full` split | Count |
|---|---|---:|
| `BY_K` | `BY_K` | 72 |
| `BY_Y` | `BY_Y` | 29 |
| `BY_Y` | `NONE` | 3 |
| `BY_YK` | `BY_K` | 6 |
| `BY_YK` | `BY_YK` | 29 |
| `NONE` | `NONE` | 78 |

Changed rows: 25 of 217.

Key changed cases:

| Shape | Old strategy | RK3588 | NVDLA `nv_full` | Main change |
|---|---|---|---|---|
| `conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1` | `pointwise_oc_tile` | `BY_Y` | `NONE` | larger CBUF removes Y split |
| `conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1` | `pointwise_oc_tile` | `BY_YK` | `BY_K` | larger CBUF removes Y split |
| `b1_c256_h28_w28_oc256_wic256_k1x1_g1` | `pointwise_oc_tile` | `BY_YK` | `BY_K` | larger CBUF removes Y split |
| `b1_c144_h38_w38_oc32_wic144_k1x1_g1_s1_pvalid` | `pointwise_oc_tile` | `BY_Y` | `NONE` | larger CBUF removes Y split |
| `b1_c576_h19_w19_oc96_wic576_k1x1_g1_s1_pvalid` | `pointwise_oc_tile` | `BY_YK` | `BY_K` | larger CBUF removes Y split |
| `b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid` | `pointwise_oc_tile` | `BY_Y` | `NONE` | larger CBUF removes Y split |
| `b1_c576_h19_w19_oc273_wic576_k1x1_g1_s1_pvalid` | `pointwise_oc_tile` | `BY_YK` | `BY_K` | larger CBUF removes Y split |
| `b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid` | `pointwise_oc_tile` | `BY_YK` | `BY_K` | larger CBUF removes Y split |
| `b1_c576_h20_w20_oc72_wic576_k1x1_g1_s1_pvalid` | `pointwise_oc_tile` | `BY_YK` | `BY_K` | larger CBUF removes Y split |

Other changed rows keep the same split class but use larger Y windows and fewer descriptors under `nv_full`, for example large spatial or depthwise shapes such as `b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid`, `b1_c96_h150_w150_oc96_wic1_k3x3_g96_s1_pvalid`, and `b1_c768_h20_w20_oc768_wic1_k3x3_g768_s1_pvalid`.

## Interpretation

The `nv_full` comparison confirms the plan's warning: ONNC/NVDLA `nv_full` no-split results cannot be treated as RK3588 no-split evidence. The 16-bank profile changes tiling for 25 shapes and specifically removes Y splitting for several high-pressure pointwise cases that RK3588 still tiles.

## Generic-Only CPU Execution

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --generic-only
```

Result:

```text
217 PASS, 0 FAIL out of 217 shapes
```

The generic-only path consumes descriptor rows directly and does not branch on the old strategy names for execution. It uses descriptor-local `y_start`, `input_h`, `output_h`, `output_w`, `k_start`, and `oc_count` for input/weight slicing and output placement.

For `BY_YK`, current descriptor families still duplicate the same CPU tile math across `setup`, `k_half`, and `k_tile`; the rows overwrite the same output tile and therefore prove descriptor-local coverage, not exact RKNN family compute semantics. Exact `grain_bits`, `cbuf0`, group-mask, and family-specific register emission remain unresolved hardware-emitter work.

## RKNN Evidence Check

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --evidence-check
```

Summary:

| Status | Count |
|---|---:|
| `PASS` | 12 |
| `GAP` | 2 |

Passing checks:

| Target | Check | Result |
|---|---|---|
| mixed `160->320 3x3 h40` | high-level split | `BY_YK`, Y windows `[0, 21, 38]`, K windows `[0, 32, ..., 320]` |
| mixed `160->320 3x3 h40` | independent Y/K windows | 2 Y windows and planner seed has 10 conservative K windows |
| mixed `160->320 3x3 h40` | family order | `setup`, `k_half`, `k_tile` |
| mixed `160->320 3x3 h40` | additive offsets | `feature_off` by Y, `weight_off` by K, `output_off` additive |
| mixed `160->320 3x3 h40` | exact RKNN `k_tile` OC windows | `112,112,96` |
| spatial `160->320 3x3 h14` | `k_tile` rows present | 3 `k_tile` rows |
| spatial `160->320 3x3 h14` | `cbuf0`/grain explicit | both remain `unknown` |
| spatial `160->320 3x3 h14` | exact RKNN `k_tile` OC windows | `112,112,96` |
| pointwise exported Y tile | `y_tile` rows present | `BY_Y`, Y windows `[0, 25, 28]` |
| pointwise exported Y tile | `grain_bits` explicit | all descriptor `grain_bits=unknown` |
| pointwise exported Y tile | `cbuf0` separate | all descriptor `cbuf0=unknown` |
| all descriptor rows | unresolved fields visible | checked 3222 descriptor rows |

Gaps found:

| Target | Gap |
|---|---|
| mixed `160->320 3x3 h40` | current descriptor count is 46, but RKNN export expects `setup x2`, `k_half x4`, `k_tile x6` |
| `pointwise_y_tile_hardcoded` | the two rare old-branch shapes still classify as `NONE`, not `BY_Y`/`BY_YK` |

Interpretation: the offline planner now matches the high-level RKNN evidence for split composition, family ordering, explicit unresolved fields, additive offsets, and `k_tile` OC windows for the two evidence shapes. It does not yet match exact RKNN `setup` and `k_half` semantics for mixed h40.

## Family Window Report

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --family-window-report
```

Focused targets:

| Target | Source | Family | Y windows | K windows | Descriptors | Max banks `input/weight/total` | Notes |
|---|---|---|---|---|---:|---|---|
| mixed `160->320 3x3 h40` | current | `setup` | `0:21;21:17` | ten `32`-channel windows | 20 | `9/3/12` | `semantic_status=unresolved`, `rknn_executable_equivalent=false` |
| mixed `160->320 3x3 h40` | current | `k_half` | `0:21;21:17` | ten `32`-channel windows | 20 | `9/3/12` | `semantic_status=unresolved`, `rknn_executable_equivalent=false` |
| mixed `160->320 3x3 h40` | current | `k_tile` | `0:21;21:17` | `0:112;112:112;224:96` | 6 | `9/10/19` | matches RKNN `k_tile` grouping |
| mixed `160->320 3x3 h40` | RKNN target | `setup` | `0:21;21:17` | `0:320` | 2 | `9/29/38` | target proves setup semantics cannot be modeled as normal full-weight CBUF residency |
| mixed `160->320 3x3 h40` | RKNN target | `k_half` | `0:21;21:17` | `0:160;160:160` | 4 | `9/15/24` | target still exceeds naive local CBUF sum |
| mixed `160->320 3x3 h40` | RKNN target | `k_tile` | `0:21;21:17` | `0:112;112:112;224:96` | 6 | `9/10/19` | target still needs family-specific reuse/group semantics |
| spatial `160->320 3x3 h14` | current | `k_tile` | `0:12` | `0:112;112:112;224:96` | 3 | `2/10/12` | matches RKNN `k_tile` grouping |
| spatial `160->320 3x3 h14` | RKNN target | `k_tile` | `0:12` | `0:112;112:112;224:96` | 3 | `2/10/12` | exactly fits 12-bank pressure by naive estimate |

Interpretation:

- The `k_tile` family now uses RKNN-like `112,112,96` windows for the two evidence shapes.
- `setup` and `k_half` still use the conservative seed K grid and are now explicitly marked unresolved, so they cannot be mistaken for executable RKNN-equivalent descriptors.
- For h40, even RKNN-like `k_tile` reaches `9/10/19` by the naive local bank sum, so h40 still needs reuse/group semantics before hardware emission.

## k_tile Emitter Field Report

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --k-tile-emitter-field-report
```

Source:

```text
experimental/rknn/models/b1_c160_h14_w14_oc320_wic160_k3x3_g1.rknn
experimental/rknn/rknn_parse_regcmd_runs.py --descriptors
experimental/rknn/librknnrt_conv_channel_tile_decomp.md lines 4138-4186, 4216-4247
```

Mined h14 `k_tile` fields:

| Run | File off | K window | `grain_bits` | `CNA_CONV_CON2` | `cbuf0` | `cbuf1` | Weight off | Output off | Planned match |
|---:|---|---|---|---|---|---|---|---|---|
| 6 | `0x32c0` | `0:112` | `0x000000f0` | `0x500000f0` | `0x000000a2` | `0x00000046` | `0x0` | `0x0` | true |
| 8 | `0x3740` | `112:112` | `0x000000f0` | `0x500000f0` | `0x000000a2` | `0x00000046` | `0x4ec00` | `0x7e00` | true |
| 10 | `0x3bc0` | `224:96` | `0x000000f0` | `0x500000f0` | `0x000000a2` | `0x00000046` | `0x9d800` | `0xfc00` | true |

Final register destinations mined from the RKNN regcmd dump:

```text
CNA_CONV_CON2      @ target 0x0201 reg 0x1010
CNA_CBUF_CON0      @ target 0x0201 reg 0x1040
CNA_CBUF_CON1      @ target 0x0201 reg 0x1044
CNA_DCOMP_ADDR0    @ target 0x0201 reg 0x1110
CORE_SIZE1         @ target 0x0801 reg 0x3018
DPU_DST_BASE_ADDR  @ target 0x1001 reg 0x4020
DPU_DST_C          @ target 0x1001 reg 0x403c
DPU_CHANNEL_END    @ target 0x1001 reg 0x4058
```

Channel-count encoding is consistent with RKNN evidence:

| K window | `CNA_WEIGHT_SIZE2` | `CORE_SIZE1` | `DPU_DST_C` | `DPU_CHANNEL_END` |
|---|---|---|---|---|
| `0:112` | `0x03030070` | `0x0000006f` | `0x006f006f` | `0x0000006f` |
| `112:112` | `0x03030070` | `0x0000006f` | `0x006f006f` | `0x0000006f` |
| `224:96` | `0x03030060` | `0x0000005f` | `0x005f005f` | `0x0000005f` |

Remaining emitter-field gaps:

| Field | Status |
|---|---|
| CNA group mask | still `unknown`; formatter/debug mask format is known, producer/programming site is not mined here |
| ABC_T/KC_T builder destination | final register destinations are mined, but the exact builder path remains `unknown` for this report |

## h14 k_tile Trace And Safety Report

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --h14-k-tile-trace-report
```

Focused answers:

| Question | Answer | Status |
|---|---|---|
| who produces/programs the CNA group mask | not identified offline; `fcn.00101be8` is a formatter only, and current static xrefs do not identify the producer/programming site | fenced |
| where ABC_T/BAC register writes are built | composer `fcn.005a41f0`, register-task builder `fcn.00597828`, then target-specific vtable writers | known builder path |
| where KC_T/C1K1C2K2C3 register writes are built | composer `fcn.005a4e18`, register-task builder `fcn.00598468`, then target-specific vtable writers | known builder path |
| where h14 `k_tile` final writes are observed | exported RKNN regcmd runs at file offsets `0x32c0`, `0x3740`, `0x3bc0` | known final regs |

Trace evidence sources:

```text
experimental/rknn/librknnrt_conv_channel_tile_decomp.md:1514-1552
experimental/rknn/librknnrt_conv_channel_tile_decomp.md:2457-2459
experimental/rknn/librknnrt_conv_channel_tile_decomp.md:2639-2730
experimental/rknn/librknnrt_conv_channel_tile_decomp.md:3028-3105
experimental/rknn/trace_librknnc_build.gdb
experimental/rknn/trace_channeltile_emit.gdb
```

Constant-field check across the three h14 K windows:

| Field | Values across K windows | Constant | Decision |
|---|---|---|---|
| `grain_bits` | `0x000000f0` | true | safe for h14 `k_tile` emission |
| `cbuf0` | `0x000000a2` | true | safe for h14 `k_tile` emission |
| `cbuf1` | `0x00000046` | true | safe for h14 `k_tile` emission |
| `CNA_CONV_CON2` | `0x500000f0` | true | safe for h14 `k_tile` emission |

Hardware-emission safety split:

| Field | Decision | Reason |
|---|---|---|
| `k_window` | safe for h14 `k_tile` emission | matches RKNN windows and coverage has no gaps/overlap |
| `family_bits` | safe for h14 `k_tile` emission | `CONV_CON2` high bits are `0x50000000` in all three RKNN rows |
| `grain_bits` | safe for h14 `k_tile` emission | constant `0xf0` across all three K windows |
| `cbuf0` | safe for h14 `k_tile` emission | constant `0xa2` across all three K windows |
| `cbuf1` | safe for h14 `k_tile` emission | constant `0x46` across all three K windows |
| `feature_off` | safe for h14 `k_tile` emission | pure K split, feature address does not advance |
| `weight_off` / `output_off` | safe for h14 `k_tile` emission | planned offsets match RKNN DCOMP/DPU destination offsets |
| channel-count registers | safe for h14 `k_tile` emission | RKNN uses `N` in `CNA_WEIGHT_SIZE2` and `N-1` in CORE/DPU fields |
| CNA group mask | fenced | mask format is known, producer/programming site is not identified |
| ABC_T/KC_T builder path | fenced for generic emission | functions are known, but the exact per-shape selected builder path remains a trace target |
| submit / PC-chain layout | fenced | this offline report does not prove `npu_submit`, `regcmd_addr`, or `regcfg_amount` safety |

## h14 k_tile Emitter Diff

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --h14-k-tile-emitter-diff
```

Summary:

| Status | Count |
|---|---:|
| `PASS` | 30 |
| `FENCED` | 3 |

The diff compares the shape-local decoded-reg fixture for h14 `160->320 3x3` `k_tile` against RKNN regcmd rows at file offsets `0x32c0`, `0x3740`, and `0x3bc0`.

Passing decoded-reg groups:

| Field group | Checks |
|---|---|
| CNA family/CBUF | `CNA_CONV_CON2=0x500000f0`, `CNA_CBUF_CON0=0x000000a2`, `CNA_CBUF_CON1=0x00000046` for all three K windows |
| K-window counts | `CNA_WEIGHT_SIZE2`, `CORE_SIZE1`, `DPU_DST_C`, and `DPU_CHANNEL_END` match `112,112,96` count encoding |
| offsets | weight/output offsets match `0/0`, `0x4ec00/0x7e00`, `0x9d800/0xfc00` |
| destinations | target/register pairs match the mined table for CNA CBUF/family, DCOMP, CORE, and DPU destination/channel fields |

Fenced fields, intentionally not defaulted:

| Field | Status | Reason |
|---|---|---|
| CNA group mask | `FENCED` | producer/programming site is still unknown |
| selected ABC_T/KC_T builder dispatch | `FENCED` | known builder functions, but per-shape selected vtable path is not inferred |
| submit / PC-chain | `FENCED` | `npu_submit`, `regcmd_addr`, `regcfg_amount`, and PC-chain tails are out of scope for offline diff |

## h14 k_tile No-Submit Dry-Run

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --h14-k-tile-no-submit-dry-run
```

Gate result:

| Check | Status | Detail | Safety |
|---|---|---|---|
| decoded-reg parity | `PASS` | `PASS=30 FENCED=3 FAIL=0` | offline only |
| register row order | `PASS` | `0x32c0;0x3740;0x3bc0` | preserves RKNN file-offset order |
| fenced fields | `FENCED` | CNA group mask; selected ABC_T/KC_T dispatch; submit/PC-chain tails | not emitted or defaulted |
| buffer layout | `PASS` | `0x0/0x0;0x4ec00/0x7e00;0x9d800/0xfc00` | `feature_off` remains `0x0` for pure K split |
| submit safety | `FENCED` | `npu_submit`, `task_count`, `regcmd_addr`, `regcfg_amount`, `enable_mask`, `core_mask`, `subcore_task`, PC-chain tails | no-submit dry-run; submit args are not modeled or written |

Materialized offline rows:

| Run | File off | K window | Row order | Feature off | Weight off | Output off | Submit state |
|---:|---|---|---:|---|---|---|---|
| 6 | `0x32c0` | `0:112` | 0 | `0x0` | `0x0` | `0x0` | `FENCED` |
| 8 | `0x3740` | `112:112` | 1 | `0x0` | `0x4ec00` | `0x7e00` | `FENCED` |
| 10 | `0x3bc0` | `224:96` | 2 | `0x0` | `0x9d800` | `0xfc00` | `FENCED` |

This is a hardware-shape packing gate only. It does not call `npu_submit`, does
not model submit args, and does not write PC-chain tails.

## CNA Group Mask Trace Report

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --cna-group-mask-trace-report
```

Trace status:

| Item | Status | Decision |
|---|---|---|
| mask format | `KNOWN` | usable for decoding only |
| formatter call site | `UNKNOWN` | do not infer producer |
| bank-validator trace points | `INSTALLED_NOT_HIT` | trace target remains valid but unresolved |
| h14 `k_tile` regcmd evidence | `NOT_PRESENT` | keep h14 group mask fenced |
| hardware emission | `FENCED` | do not default or emit CNA group mask |

Known bit layout from formatter `fcn.00101be8`:

| Bits | Meaning |
|---|---|
| `0..1` | CNA feature group0/group1 |
| `2..3` | CNA weight group0/group1 |
| `4..5` | CNA CSC group0/group1 |
| `6..7` | ACCU group0/group1 |
| `8..9` | DPU group0/group1 |
| `10..11` | PPU group0/group1 |
| `12..13` | DMA read/write error bits |

Trace targets remain:

| Target | Offset | Status | Next use |
|---|---|---|---|
| `cna_bank_validator_a` | `librknnc.so+0x007d13dc` | breakpoint installed, not hit for tested FP16 conv builds | rerun only for targeted model/build that should exercise group programming |
| `cna_bank_validator_b` | `librknnc.so+0x007d1438` | breakpoint installed, not hit for tested FP16 conv builds | dump caller chain and candidate mask words if hit |

## ABC/KC Builder Dispatch Report

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --abc-kc-builder-dispatch-report
```

Status:

| Path | Function | Status | Decision |
|---|---|---|---|
| ABC_T/BAC composer | `fcn.005a41f0` | `KNOWN_FUNCTION` | known builder family |
| ABC_T/BAC register-task builder | `fcn.00597828` | `KNOWN_FUNCTION` | known builder family |
| KC_T/C1K1C2K2C3 composer | `fcn.005a4e18` | `KNOWN_FUNCTION` | known builder family |
| KC_T/C1K1C2K2C3 register-task builder | `fcn.00598468` | `KNOWN_FUNCTION` | known builder family |
| h14 final reg replay | n/a | `SUFFICIENT_FOR_H14_FIXTURE` | do not require selected vtable dispatch for shape-local offline fixture |
| exact selected vtable dispatch | unknown | `FENCED_FOR_GENERIC_EMISSION` | do not claim generic ABC_T/KC_T emission |

Trace targets if final-reg replay becomes insufficient:

| Target | Offset | Prior status | When needed |
|---|---|---|---|
| `abc_or_channel_tile_emit_a` | `librknnc.so+0x015d0cc0` | installed, not hit in prior trace | only if final-reg replay becomes insufficient for a target shape |
| `abc_or_channel_tile_emit_b` | `librknnc.so+0x015d5ef0` | installed, not hit in prior trace | generic ABC_T/BAC builder dispatch proof |
| `kc_or_alt_emit` | `librknnc.so+0x015e1b60` | installed, not hit in prior trace | generic KC_T/C1K1C2K2C3 builder dispatch proof |

## Targeted VP / Trace List

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --targeted-vp-list-report
```

Reduced target list:

| Target | Trigger | Method | Expected output | Broad sweep |
|---|---|---|---|---|
| h14 `k_tile` CNA group mask | only if a future hardware fixture needs a concrete mask value | targeted compiler/runtime trace at `cna_bank_validator_a/b` before VP | producer/programming site or explicit keep-fenced decision | no |
| mixed h40 `setup`/`k_half` semantics | only if replacing the fallback/fence becomes necessary | targeted VP or RKNN trace for `setup x2` and `k_half x4` semantics | reuse/group semantics or permanent fallback decision | no |
| ABC_T/KC_T selected builder dispatch | only if final-reg replay is insufficient for a target shape | targeted builder trace at `abc_or_channel_tile_emit_a/b` or `kc_or_alt_emit` | selected vtable path or explicit generic-emission fence | no |

## Unresolved Fence Report

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --unresolved-fence-report
```

Focused h40 state:

| Family | Status | RKNN executable equivalent | Hardware decision | Reason |
|---|---|---|---|---|
| `setup` | unresolved | false | fenced | mixed h40 RKNN `setup` semantics exceed naive CBUF pressure and are not modeled |
| `k_half` | unresolved | false | fenced | mixed h40 RKNN `k_half` semantics exceed naive CBUF pressure and are not modeled |
| `k_tile` | planned | true | candidate after emitter fields | K windows are RKNN-like, but h40 still needs reuse/group semantics before emission |

## Family Coverage Report

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --family-coverage-report
```

Result:

```text
PASS 209
```

The report checks `k_tile` rows only. For every shape/Y-window with `k_tile` rows, the output-channel windows cover every output channel exactly once with no overlap.

Focused rows:

| Shape | Split | Y window | K windows | Missing | Overlap |
|---|---|---|---|---:|---:|
| `b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid` | `BY_K` | `0:12` | `0:112;112:112;224:96` | 0 | 0 |
| `evidence_mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1` | `BY_YK` | `0:21` | `0:112;112:112;224:96` | 0 | 0 |
| `evidence_mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1` | `BY_YK` | `21:17` | `0:112;112:112;224:96` | 0 | 0 |

## Pointwise Hardcoded Rows

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --pointwise-hardcoded-report
```

Result:

| Shape | Old strategy | Split | Descriptor | Banks `input/weight/total` | Decision |
|---|---|---|---|---|---|
| `conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1` | `pointwise_y_tile_hardcoded` | `NONE` | `setup` | `8/1/9` | old strategy noise: current CBUF model fits as `NONE` |
| `conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1` | `pointwise_y_tile_hardcoded` | `NONE` | `setup` | `10/1/11` | old strategy noise: current CBUF model fits as `NONE` |

## Fallback And Register Fields

`spatial_im2col` remains a hardware fallback. The planner can classify those shapes, but prior hardware evidence says direct dense `160->320 3x3` streaming fails and naive ChannelTile is insufficient. Do not remove the fallback until family K windows, KG/reuse, and group-mask semantics are clearer.

Register-emitter mining for the h14 `k_tile` evidence shape now has concrete `grain_bits`, `cbuf0`, `cbuf1`, offsets, and final register destinations. CNA group mask and exact ABC_T/KC_T builder path remain unresolved and should be mined from targeted compiler/runtime traces, not broad VP sweeps.

Remaining next actions 1-2 are now reduced to explicit fenced states:

1. Mixed h40 `setup`/`k_half` are fenced and cannot be consumed as RKNN-equivalent hardware descriptors.
2. h14 `k_tile` emitter mining and no-submit dry-run are complete enough for a shape-local offline fixture, but not for generic emission or hardware submit because generic selected builder dispatch and submit/PC-chain layout remain unresolved.
3. CNA group mask remains fenced for hardware emission: the bit layout is known, but the producer/programming site is not identified and the existing bank-validator trace points were installed but not hit for tested FP16 conv builds.
4. ABC/KC builder families are known and final-reg replay is sufficient for h14; exact selected vtable dispatch remains fenced for generic emission.
5. VP questions are reduced to three targeted trace/VP cases. Do not run a broad VP sweep.

## h14 Hardware-Side No-Submit Scaffold

Command:

```sh
python3 examples/conv_h14_k_tile_no_submit.py
```

Result:

```text
profile=h14_k_tile_fixture status=NO_SUBMIT submit_state=FENCED rows=3
decoded_rows=PASS detail=3
executable_fixture_registry=PASS detail=h14_k_tile;h7_k_tile;setup
registers_per_row=PASS detail=9;9;9
expected_reg_parity=PASS detail=PASS=27 FAIL=0
h14_k_tile_observed_compare=PASS detail=PASS=48 FAIL=0
fenced_negative_fixture=PASS detail=materialized_rows=0
second_executable_fixture=PASS detail=b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid PASS=27 FAIL=0
setup_family_fixture=PASS detail=b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid PASS=9 FAIL=0
setup_observed_compare=PASS detail=PASS=16 FAIL=0
y_tile_candidate=PASS detail=RKNN y_tile rows are mined, but planner emits setup-only and the 9-reg materializer does not emit feature/data/window regs
y_tile_candidate_filter=PASS detail=materialized_rows=0
y_tile_observed_compare=PASS detail=PASS=48 FAIL=0
k_half_candidate=PASS detail=RKNN k_half rows are mined, but planner emits setup-only for this shape and k_half family semantics are not solved
k_half_candidate_filter=PASS detail=materialized_rows=0
k_half_observed_compare=PASS detail=PASS=32 FAIL=0
observed_candidate_metadata=PASS detail=feature/window/channel regs match mined candidates
submit_state=PASS detail=FENCED;FENCED;FENCED
fenced_field_count=PASS detail=3;3;3
executable_filter=PASS detail=true
descriptor_source=PASS detail=planner;planner;planner
```

The scaffold now uses a reusable no-submit materialization contract:

| Component | Role |
|---|---|
| `Descriptor` | Carries planner row identity, K window, offsets, semantic status, `rknn_executable_equivalent`, and source. |
| `EmitterProfile` | Carries explicit fixture-local emitter constants: family bits, grain bits, CBUF values, kernel shape, and fenced fields. |
| `MaterializedRow` | Carries one executable-equivalent descriptor, decoded register rows, `submit_state=FENCED`, and fenced-field list. |
| `examples/conv_no_submit_materializer.py` | Importable no-submit module that owns generic register IDs, dataclasses, `descriptor_to_decoded_regs()`, and `materialize_no_submit_rows()`. |
| `EXECUTABLE_FIXTURES` | Registry for planner-fed executable no-submit fixtures. It currently contains `h14_k_tile`, `h7_k_tile`, and `setup`, with each entry owning its descriptor function, profile, expected windows/offsets/file offsets, expected decoded regs, optional observed-reg table, and run metadata. |
| `materialize_executable_fixture(key)` | Registry-backed materialization helper. Compatibility wrappers for h14, h7, and setup now call this helper. |
| `planner_h14_k_tile_descriptors()` | Imports `conv_tile_cpu.py`, extracts h14 `k_tile` planner rows, and attaches RKNN run/file-offset metadata for reporting. |
| `materialize_no_submit_rows(descriptors, profile)` | Reusable no-submit helper. It filters non-executable descriptors and emits decoded regs only. |
| `H14_EXPECTED_REGS` | Internal expected-value table for the 27 h14 decoded registers; `validate_rows()` asserts parity before printing. |
| `H14_K_TILE_OBSERVED_COMPARE_REGS` | Compare-only RKNN fields for the h14 executable `k_tile` fixture. It validates 48 fields across 3 rows, including expanded shape/window fields outside the 9-reg emitter contract. |
| `FENCED_NEGATIVE_DESCRIPTOR` | Negative fixture with `rknn_executable_equivalent=false`; the self-check proves it materializes zero rows. |
| `H7_EXPECTED_REGS` | Internal expected-value table for the 27 h7 decoded registers mined from `experimental/rknn/models/b1_c160_h7_w7_oc320_wic160_k3x3_g1.rknn`. |
| `H7_K_TILE_PROFILE` | h7-local profile: `grain_bits=0xa0`, `CNA_CONV_CON2=0x500000a0`, `cbuf0=0xb1`, `cbuf1=0x23`. |
| `SECOND_EXECUTABLE_FIXTURE_STATUS` | Now enabled for the h7 no-submit fixture after expected decoded-reg parity was mined. |
| `SETUP_EXPECTED_REGS` | Internal expected-value table for the setup row mined from `experimental/rknn/models/b1_c32_h14_w14_oc128_wic32_k3x3_g1.rknn`, run 1 at `0x1d80`. |
| `SETUP_PROFILE` | setup-row profile: `grain_bits=0x110`, `CNA_CONV_CON2=0x00000110`, `cbuf0=0xb1`, `cbuf1=0x0e`. |
| `SETUP_OBSERVED_COMPARE_REGS` | Compare-only RKNN fields for the setup row. It validates 16 fields for run 1, including expanded shape/window fields outside the 9-reg emitter contract. |
| `Y_TILE_RKNN_CANDIDATE` | Fenced mined `y_tile` rows from the same c32 RKNN export: runs `6;8;10`, file offsets `0x2a00;0x2e80;0x3300`, Y windows `0:4;4:4;8:4`, feature offsets `0x0;0x380;0x700`, output offsets `0x0;0x300;0x600`, `grain_bits=0x90`, `cbuf0=0xb1`, `cbuf1=0x0e`. |
| `Y_TILE_OBSERVED_DESCRIPTORS` | Observed-only descriptors with `source=rknn_observed_y_tile_candidate` and `rknn_executable_equivalent=false`; the self-check proves they materialize zero rows. |
| `Y_TILE_CANDIDATE_STATUS` | Explicit fence: no local RKNN-backed model has a planner-emitted `y_tile` match, and the current 9-reg materializer does not emit feature/data/window registers. |
| `Y_TILE_OBSERVED_COMPARE_REGS` | Compare-only RKNN fields for the fenced `y_tile` rows. It validates 48 fields across 3 rows without making them executable. Distinguishing fields include `CNA_DATA_SIZE0=0x000e0006`, `CNA_DATA_SIZE3=0x30`, `CORE_SIZE0=0x0003000b`, `DPU_DST_H=3`, `DPU_SIZE=0x0003000b`, feature offsets `0x0;0x380;0x700`, and output offsets `0x0;0x300;0x600`. |
| `K_HALF_RKNN_CANDIDATE` | Fenced mined `k_half` rows from the same c32 RKNN export: runs `2;4`, file offsets `0x20f8;0x2580`, K windows `0:64;64:64`, weight offsets `0x0;0x9000`, output offsets `0x0;0x4800`, `grain_bits=0x110`, `cbuf0=0xb1`, `cbuf1=0x0e`. |
| `K_HALF_OBSERVED_DESCRIPTORS` | Observed-only descriptors with `source=rknn_observed_k_half_candidate` and `rknn_executable_equivalent=false`; the self-check proves they materialize zero rows. |
| `K_HALF_OBSERVED_COMPARE_REGS` | Compare-only RKNN fields for the fenced `k_half` rows. It validates 32 fields across 2 rows without making them executable. Distinguishing fields include `CNA_CONV_CON2=0x40000110`, `CNA_WEIGHT_SIZE0=0x9000`, `CNA_WEIGHT_SIZE2=0x03030040`, `CORE_SIZE1=0x3f`, `DPU_DST_C=0x003f003f`, weight offsets `0x0;0x9000`, and output offsets `0x0;0x4800`. |

The enabled profiles are `h14_k_tile_fixture` and `h7_k_tile_fixture`. The
scaffold materializes the h14 `160->320 3x3` `k_tile` fixture rows, the h7
`160->320 3x3` `k_tile` fixture rows, and one setup-family row. All are
no-submit only:

| Row | Source | Run | File off | K window | Feature off | Weight off | Output off | Submit state |
|---:|---|---:|---|---|---|---|---|---|
| 0 | planner | 6 | `0x32c0` | `0:112` | `0x0` | `0x0` | `0x0` | `FENCED` |
| 1 | planner | 8 | `0x3740` | `112:112` | `0x0` | `0x4ec00` | `0x7e00` | `FENCED` |
| 2 | planner | 10 | `0x3bc0` | `224:96` | `0x0` | `0x9d800` | `0xfc00` | `FENCED` |

h7 materialized rows:

| Row | Source | Run | File off | K window | Feature off | Weight off | Output off | Submit state |
|---:|---|---:|---|---|---|---|---|---|
| 0 | planner | 6 | `0x32c0` | `0:112` | `0x0` | `0x0` | `0x0` | `FENCED` |
| 1 | planner | 8 | `0x3740` | `112:112` | `0x0` | `0x4ec00` | `0x1880` | `FENCED` |
| 2 | planner | 10 | `0x3bc0` | `224:96` | `0x0` | `0x9d800` | `0x3100` | `FENCED` |

setup materialized row:

| Row | Source | Run | File off | K window | Feature off | Weight off | Output off | Submit state |
|---:|---|---:|---|---|---|---|---|---|
| 0 | planner | 1 | `0x1d80` | `0:128` | `0x0` | `0x0` | `0x0` | `FENCED` |

The setup fixture validates the setup-family decoded-register row only. The same
RKNN export also contains extra `k_half` and `y_tile` rows, so this is not a
claim that the whole RKNN model schedule is planner-equivalent.

y_tile mined candidate, still fenced:

| Run | File off | Y window | Input H | Feature off | Output off | `CNA_CONV_CON2` | `cbuf0` | `cbuf1` | Status |
|---:|---|---|---:|---|---|---|---|---|---|
| 6 | `0x2a00` | `0:4` | 6 | `0x0` | `0x0` | `0x20000090` | `0xb1` | `0x0e` | fenced |
| 8 | `0x2e80` | `4:4` | 6 | `0x380` | `0x300` | `0x20000090` | `0xb1` | `0x0e` | fenced |
| 10 | `0x3300` | `8:4` | 6 | `0x700` | `0x600` | `0x20000090` | `0xb1` | `0x0e` | fenced |

Decision: do not expand the no-submit materializer for `y_tile` yet. The current
source contract is planner-fed descriptors, and no local RKNN-backed model was
found where the planner emits a matching `y_tile` descriptor. The mined RKNN rows
are therefore represented as observed-only descriptors with
`rknn_executable_equivalent=false`, and the self-check proves they materialize
zero rows. Also, the current 9 decoded registers are insufficient to prove
Y-window behavior because the distinguishing fields are feature/data and window
registers such as `CNA_FEATURE_DATA_ADDR`, `CNA_DATA_SIZE0`, `CORE_SIZE0`, and
`DPU_SIZE`.

k_half mined candidate, still fenced:

| Run | File off | K window | Weight off | Output off | `CNA_CONV_CON2` | `cbuf0` | `cbuf1` | Status |
|---:|---|---|---|---|---|---|---|---|
| 2 | `0x20f8` | `0:64` | `0x0` | `0x0` | `0x40000110` | `0xb1` | `0x0e` | fenced |
| 4 | `0x2580` | `64:64` | `0x9000` | `0x4800` | `0x40000110` | `0xb1` | `0x0e` | fenced |

Decision: do not expand the no-submit materializer for `k_half` yet. These rows
are observed-only for the c32 RKNN export, while the planner emits the shape as a
single setup row. Mixed h40 `k_half` semantics also remain unresolved by the
family-window report, so `k_half` must stay non-executable until family compute
semantics are solved.

The h7 fixture required one planner offset correction: descriptor `output_off`
now uses `out_width_stride` rather than raw `out_h * out_w`. This preserves h14
offsets (`12*12` is already aligned) and matches h7 RKNN offsets where `5*5` is
aligned to 28 elements per output channel.

Decoded registers emitted per row:

| Register | Target/reg | Source |
|---|---|---|
| `CNA_CONV_CON2` | `0201:1010` | constant h14 fixture value `0x500000f0` |
| `CNA_CBUF_CON0` | `0201:1040` | constant h14 fixture value `0x000000a2` |
| `CNA_CBUF_CON1` | `0201:1044` | constant h14 fixture value `0x00000046` |
| `CNA_WEIGHT_SIZE2` | `0201:1038` | K-window count encoding, `N` in low field |
| `CNA_DCOMP_ADDR0` | `0201:1110` | descriptor `weight_off` |
| `CORE_SIZE1` | `0801:3018` | `oc_count - 1` |
| `DPU_DST_BASE_ADDR` | `1001:4020` | descriptor `output_off` |
| `DPU_DST_C` | `1001:403c` | `oc_count - 1` in both channel fields |
| `DPU_CHANNEL_END` | `1001:4058` | `oc_count - 1` |

Still fenced and not emitted:

| Field | Reason |
|---|---|
| CNA group mask | producer/programming site remains unidentified |
| selected ABC_T/KC_T builder dispatch | final-reg replay is enough for the h14/h7 fixtures, not generic emission |
| submit/PC-chain | submit args and PC-chain tails remain out of scope |

Intentional mismatch probe:

```sh
python3 -c "import examples.conv_h14_k_tile_no_submit as h; from examples.conv_no_submit_materializer import EmitterProfile; rows=h.materialize_no_submit_rows(h.planner_h14_k_tile_descriptors(), EmitterProfile(name='bad', family_bits=0x50000000, grain_bits=0, cbuf0=0xa2, cbuf1=0x46, kernel_h=3, kernel_w=3, fenced_fields=h.FENCED_FIELDS)); mismatches=h.expected_parity_mismatches(rows); assert mismatches and mismatches[0][1] == 'CNA_CONV_CON2'"
```

This confirms the scaffold catches expected-value drift before any submit path is
introduced.

## Registry-Driven No-Submit Consumer

Command:

```sh
python3 examples/conv_tiles_no_submit.py h14_k_tile
python3 examples/conv_tiles_no_submit.py h7_k_tile
python3 examples/conv_tiles_no_submit.py setup
python3 examples/conv_tiles_no_submit.py --list
python3 examples/conv_tiles_no_submit.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/conv_tiles_no_submit.py b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/conv_tiles_no_submit.py b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid
! python3 examples/conv_tiles_no_submit.py y_tile
! python3 examples/conv_tiles_no_submit.py k_half
! python3 examples/conv_tiles_no_submit.py b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid
! python3 examples/conv_tiles_no_submit.py unknown
```

Result:

| Check | Status | Detail |
|---|---|---|
| `h14_k_tile` | `PASS` | 3 planner-fed rows, 9 decoded regs per row, `submit_state=FENCED` |
| `h7_k_tile` | `PASS` | 3 planner-fed rows, 9 decoded regs per row, `submit_state=FENCED` |
| `setup` | `PASS` | 1 planner-fed row, 9 decoded regs, `submit_state=FENCED` |
| `--list` | `PASS` | lists the three executable fixture keys and exact shape names |
| exact executable shape names | `PASS` | resolve to the same h14/h7/setup fixture rows as fixture keys |
| `y_tile` | `PASS` | rejected as observed-only fenced candidate |
| `k_half` | `PASS` | rejected as observed-only fenced candidate |
| h40 unsupported shape | `PASS` | rejected before materialization |
| unknown key | `PASS` | rejected before materialization |

`examples/conv_tiles_no_submit.py` is intentionally no-submit only. It imports the
fixture registry, accepts either executable fixture keys or exact fixture shape
names, validates expected windows/offsets/file offsets, decoded-reg parity,
optional observed compares, and `submit_state=FENCED`, then prints a deterministic
decoded-register listing. It does not open DRM, allocate NPU memory, write task
structs, model submit args, or call `npu_submit`.

## Action 34 Difficult-Shape Evidence Matrix

Scope: build a targeted NVDLA VP / ONNC / KMD evidence matrix from existing
artifacts and local planner descriptor dumps. This is evidence work only: no
hardware submit path was changed, no VP broad sweep was run, and no
`examples/kernel_6_18/` file was edited.

Commands:

```sh
python3 conv_expt/conv_tile_cpu.py --descriptor-dump conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1
python3 conv_expt/conv_tile_cpu.py --descriptor-dump conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1
python3 conv_expt/conv_tile_cpu.py --descriptor-dump conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1
python3 conv_expt/conv_tile_cpu.py --descriptor-dump conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1
python3 conv_expt/conv_tile_cpu.py --descriptor-dump b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid
python3 conv_expt/conv_tile_cpu.py --descriptor-dump b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid
```

Existing ONNC/VP source: `conv_expt/shape_stratgery.md`, lines 88-330.

| Shape | Old branch pressure | Local planner split/family | Local windows | ONNC/VP evidence | Interpretation | Next evidence need |
|---|---|---|---|---|---|---|
| `conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1` | `pointwise_y_tile_hardcoded` | `NONE` / `setup` | one full row: `y=0:28`, `k=0:32`, banks `8/1` | ONNC parses as `1 CONV + 1 SDP`, no software tiling | `planner-confirmed` for no split under current RK3588 CBUF model; old branch is likely local noise unless RKNN proves otherwise | none before cleanup; keep as `setup` candidate |
| `conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1` | `pointwise_y_tile_hardcoded` | `NONE` / `setup` | one full row: `y=0:28`, `k=0:32`, banks `10/1` | ONNC parses as `1 CONV + 1 SDP`, no software tiling | `planner-confirmed` for no split under current RK3588 CBUF model; old branch is likely local noise unless RKNN proves otherwise | none before cleanup; keep as `setup` candidate |
| `conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1` | related pointwise/Y pressure | `BY_Y` / `y_tile` | local rows `0:32`, `32:24`, banks `11/1`, `8/1` | ONNC parses as `2 CONV + 2 SDP`; VP/KMD capture passes and splits output height `45 + 11` | `planner-confirmed` at split class level; exact row cuts differ because ONNC `nv_full` has a 16-bank budget while RK3588 uses 12 banks | no new VP needed unless exact NVDLA row-cut formula becomes a cleanup blocker |
| `conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1` | related pointwise/Y pressure | `BY_Y` / `y_tile` | local rows `0:20`, `20:20`, `40:16`, banks `11/1`, `11/1`, `9/1` | ONNC parses as `2 CONV + 2 SDP`, BY_Y likely | `planner-confirmed` at split class level; local RK3588 needs smaller Y windows than ONNC `nv_full` likely due to 12-bank pressure | parse/capture only if exact row cuts become needed; no broad VP sweep |
| `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` | large spatial/Y pressure | `BY_Y` / `y_tile` | local rows `0:62`, `62:62`, `124:34`, banks `10/2`, `10/2`, `6/2` | ONNC parses as `2 CONV + 2 SDP`, BY_Y / output-height split | `planner-confirmed` that direct spatial Y split is compiler-like; local row cuts remain RK3588-profile specific | parse/capture only if replacing fallback for this family requires exact runtime offsets |
| `b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid` | old `spatial_im2col` | `BY_Y` / `y_tile` | local rows `0:65`, `65:65`, `130:65`, `195:65`, `260:58`, banks `11/1`, `11/1`, `11/1`, `11/1`, `10/1` | ONNC parses as `7 CONV + 7 SDP`, strong spatial split candidate | `planner-confirmed` that this should be direct spatial Y descriptors rather than permanent Python im2col; exact row count differs by profile | targeted VP/runtime capture only if promoting direct spatial hardware emission for this shape |

Action 34 conclusion:

- The matrix supports the simple planner direction. The difficult shapes collapse
  into `NONE` or `BY_Y` with descriptor families; none require preserving the old
  six Python strategy names as compiler concepts.
- ONNC `nv_full` evidence is useful for split class but not exact RK3588 row cuts.
  `nv_full` has 16 banks while RK3588 has 12, so a different tile count is not a
  contradiction.
- The two old `pointwise_y_tile_hardcoded` shapes are now classified as
  `planner-confirmed` no-split candidates unless RKNN evidence proves a hidden
  reason for Y tiling.
- The large spatial shapes support replacing old `spatial_im2col` with direct
  spatial `y_tile` descriptors eventually, but hardware emission is still gated by
  the existing `y_tile` decoded-register and submit fences.
- No new VP capture is needed before the next cleanup step. New VP work should be
  targeted only if exact row cuts, runtime offsets, or family-specific register
  fields become blockers.

## Action 35 Executable Difficult-Shape Evidence Report

Scope: make the Action 34 matrix executable in `conv_tile_cpu.py` so cleanup work
can use a stable offline gate instead of relying only on markdown.

Command:

```sh
python3 conv_expt/conv_tile_cpu.py --difficult-shape-evidence-report
```

Result:

| Shape | Old branch pressure | Local split | Families | Local Y windows | Local K windows | Banks | ONNC/VP evidence | Interpretation | Next evidence need |
|---|---|---|---|---|---|---|---|---|---|
| `conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1` | `pointwise_y_tile_hardcoded` | `NONE` | `setup` | `0:28` | `0:32` | `8/1` | ONNC `1 CONV + 1 SDP`, no software tiling | `planner-confirmed` | none before cleanup; keep setup candidate |
| `conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1` | `pointwise_y_tile_hardcoded` | `NONE` | `setup` | `0:28` | `0:32` | `10/1` | ONNC `1 CONV + 1 SDP`, no software tiling | `planner-confirmed` | none before cleanup; keep setup candidate |
| `conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1` | related pointwise/Y pressure | `BY_Y` | `y_tile` | `0:32;32:24` | `0:24` | `11/1;8/1` | ONNC `2 CONV + 2 SDP`, VP/KMD pass, output H `45+11` | planner-confirmed split class; RK3588 row cuts differ | none unless exact NVDLA row cuts block cleanup |
| `conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1` | related pointwise/Y pressure | `BY_Y` | `y_tile` | `0:20;20:20;40:16` | `0:24` | `11/1;9/1` | ONNC `2 CONV + 2 SDP`, BY_Y likely | planner-confirmed split class; RK3588 row cuts differ | parse/capture only if exact row cuts become needed |
| `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` | large spatial/Y pressure | `BY_Y` | `y_tile` | `0:62;62:62;124:34` | `0:128` | `10/2;6/2` | ONNC `2 CONV + 2 SDP`, BY_Y output-height split | planner-confirmed direct spatial BY_Y candidate | capture only if runtime offsets block fallback removal |
| `b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid` | `spatial_im2col` | `BY_Y` | `y_tile` | `0:65;65:65;130:65;195:65;260:58` | `0:32` | `11/1;10/1` | ONNC `7 CONV + 7 SDP`, strong spatial split candidate | planner-confirmed direct spatial BY_Y candidate | targeted VP only if promoting this shape to hardware |

Action 35 conclusion:

- The difficult-shape matrix is now generated from live local planner descriptors
  plus static ONNC/VP evidence constants.
- The report is offline-only. It does not parse new loadables, run VP, open DRM,
  allocate NPU memory, or touch submit fields.
- This report is now the cleanup gate for preventing the old
  `pointwise_y_tile_hardcoded` and `spatial_im2col` names from re-entering the
  design as strategy branches.

## Validation

Commands run successfully:

```sh
python3 -m py_compile conv_expt/conv_tile_cpu.py
python3 conv_expt/conv_tile_cpu.py --cross-tab
python3 conv_expt/conv_tile_cpu.py --planner-report
python3 conv_expt/conv_tile_cpu.py --descriptor-dump b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid
python3 conv_expt/conv_tile_cpu.py --descriptor-dump b1_c32_h112_w112_oc64_wic32_k1x1_g1
python3 conv_expt/conv_tile_cpu.py --cbuf-compare
python3 conv_expt/conv_tile_cpu.py --descriptor-dump conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1
python3 conv_expt/conv_tile_cpu.py --generic-only
python3 conv_expt/conv_tile_cpu.py --evidence-check
python3 conv_expt/conv_tile_cpu.py --descriptor-dump b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid
python3 conv_expt/conv_tile_cpu.py --descriptor-dump b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid
python3 experimental/rknn/rknn_parse_regcmd_runs.py --descriptors experimental/rknn/models/b1_c32_h14_w14_oc128_wic32_k3x3_g1.rknn
python3 conv_expt/conv_tile_cpu.py --family-window-report
python3 conv_expt/conv_tile_cpu.py --family-coverage-report
python3 conv_expt/conv_tile_cpu.py --family-coverage-all
python3 conv_expt/conv_tile_cpu.py --pointwise-hardcoded-report
python3 conv_expt/conv_tile_cpu.py --k-tile-emitter-field-report
python3 conv_expt/conv_tile_cpu.py --h14-k-tile-trace-report
python3 conv_expt/conv_tile_cpu.py --h14-k-tile-emitter-diff
python3 conv_expt/conv_tile_cpu.py --h14-k-tile-no-submit-dry-run
python3 conv_expt/conv_tile_cpu.py --cna-group-mask-trace-report
python3 conv_expt/conv_tile_cpu.py --abc-kc-builder-dispatch-report
python3 conv_expt/conv_tile_cpu.py --difficult-shape-evidence-report
python3 conv_expt/conv_tile_cpu.py --targeted-vp-list-report
python3 conv_expt/conv_tile_cpu.py --unresolved-fence-report
python3 conv_expt/conv_tile_cpu.py --descriptor-dump evidence_mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1
python3 conv_expt/conv_tile_cpu.py --descriptor-dump conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1
python3 conv_expt/conv_tile_cpu.py --descriptor-dump conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1
python3 conv_expt/conv_tile_cpu.py --descriptor-dump conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1
python3 conv_expt/conv_tile_cpu.py --descriptor-dump conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1
python3 conv_expt/conv_tile_cpu.py --descriptor-dump b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid
python3 conv_expt/conv_tile_cpu.py --descriptor-dump b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid
python3 -m py_compile examples/conv_no_submit_materializer.py examples/conv_h14_k_tile_no_submit.py
python3 -m py_compile examples/conv_tiles_no_submit.py
python3 examples/conv_h14_k_tile_no_submit.py
python3 examples/conv_tiles_no_submit.py h14_k_tile
python3 examples/conv_tiles_no_submit.py h7_k_tile
python3 examples/conv_tiles_no_submit.py setup
python3 examples/conv_tiles_no_submit.py --list
python3 examples/conv_tiles_no_submit.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/conv_tiles_no_submit.py b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/conv_tiles_no_submit.py b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid
! python3 examples/conv_tiles_no_submit.py y_tile
! python3 examples/conv_tiles_no_submit.py k_half
! python3 examples/conv_tiles_no_submit.py b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid
! python3 examples/conv_tiles_no_submit.py unknown
python3 -c "import examples.conv_h14_k_tile_no_submit as h; from examples.conv_no_submit_materializer import EmitterProfile; rows=h.materialize_no_submit_rows(h.planner_h14_k_tile_descriptors(), EmitterProfile(name='bad', family_bits=0x50000000, grain_bits=0, cbuf0=0xa2, cbuf1=0x46, kernel_h=3, kernel_w=3, fenced_fields=h.FENCED_FIELDS)); mismatches=h.expected_parity_mismatches(rows); assert mismatches and mismatches[0][1] == 'CNA_CONV_CON2'"
```
