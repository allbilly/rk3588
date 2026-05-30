# CONV Tile Planner Result

Date: 2026-05-30

## Scope

Implemented offline planner reporting in `conv_expt/conv_tile_cpu.py`.

This is planner-only work. No NPU submit path or hardware example was changed.

## Added Commands

```sh
python3 conv_expt/conv_tile_cpu.py --planner-report
python3 conv_expt/conv_tile_cpu.py --descriptor-dump [SHAPE]
python3 conv_expt/conv_tile_cpu.py --cross-tab
python3 conv_expt/conv_tile_cpu.py --cbuf-compare
python3 conv_expt/conv_tile_cpu.py --cbuf-compare-all
python3 conv_expt/conv_tile_cpu.py --generic-only
python3 conv_expt/conv_tile_cpu.py --evidence-check
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
data_reuse, weight_reuse
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
| `PASS` | 10 |
| `GAP` | 4 |

Passing checks:

| Target | Check | Result |
|---|---|---|
| mixed `160->320 3x3 h40` | high-level split | `BY_YK`, Y windows `[0, 21, 38]`, K windows `[0, 32, ..., 320]` |
| mixed `160->320 3x3 h40` | independent Y/K windows | 2 Y windows and 10 K windows |
| mixed `160->320 3x3 h40` | family order | `setup`, `k_half`, `k_tile` |
| mixed `160->320 3x3 h40` | additive offsets | `feature_off` by Y, `weight_off` by K, `output_off` additive |
| spatial `160->320 3x3 h14` | `k_tile` rows present | 10 `k_tile` rows |
| spatial `160->320 3x3 h14` | `cbuf0`/grain explicit | both remain `unknown` |
| pointwise exported Y tile | `y_tile` rows present | `BY_Y`, Y windows `[0, 25, 28]` |
| pointwise exported Y tile | `grain_bits` explicit | all descriptor `grain_bits=unknown` |
| pointwise exported Y tile | `cbuf0` separate | all descriptor `cbuf0=unknown` |
| all descriptor rows | unresolved fields visible | checked 3236 descriptor rows |

Gaps found:

| Target | Gap |
|---|---|
| mixed `160->320 3x3 h40` | current descriptor count is 60, but RKNN export expects `setup x2`, `k_half x4`, `k_tile x6` |
| mixed `160->320 3x3 h40` | current `k_tile` OC windows are ten `32`-channel windows; RKNN export expects `112,112,96` |
| spatial `160->320 3x3 h14` | current `k_tile` OC windows are ten `32`-channel windows; RKNN export expects `112,112,96` |
| `pointwise_y_tile_hardcoded` | the two rare old-branch shapes still classify as `NONE`, not `BY_Y`/`BY_YK` |

Interpretation: the offline planner now matches the high-level RKNN evidence for split composition, family ordering, explicit unresolved fields, and additive offsets. It does not yet match exact RKNN family-specific K-window grouping. The next useful step is to investigate those gaps by family before hardware cleanup.

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
```
