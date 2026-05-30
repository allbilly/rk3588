# conv_new.py Shape Strategy Grouping

Source: `examples/kernel_6_18/conv_new.py`

Total literal test shapes: `217`

The grouping below follows the actual `run_conv2d()` branch order in
`conv_new.py`, not just shape names. The special branches are checked before the
fallback/direct path:

1. `spatial_im2col`
2. `grouped_serial`
3. `spatial_oc_serial`
4. `depthwise_spatial_tiled`
5. `pointwise_oc_tile`
6. `pointwise_y_tile_hardcoded`
7. fallback/direct path

## Count By Strategy

| Strategy / path | Count | Share |
|---|---:|---:|
| `pointwise_oc_tile` | 87 | 40.1% |
| fallback/direct path | 54 | 24.9% |
| `depthwise_spatial_tiled` | 36 | 16.6% |
| `grouped_serial` | 15 | 6.9% |
| `spatial_oc_serial` | 12 | 5.5% |
| `spatial_im2col` | 11 | 5.1% |
| `pointwise_y_tile_hardcoded` | 2 | 0.9% |

Fallback/direct path detail:

| Fallback subtype | Count | Notes |
|---|---:|---|
| true direct fallback | 43 | No explicit tiling branch selected |
| flat-output Y tile fallback | 11 | Large small-channel `1x1`, mostly `ic=3, oc=6`, `52x52..72x72` |

## Rare / Difficult Groups

### `pointwise_y_tile_hardcoded`: 2/217

These are the rarest shapes. They currently need the explicit
`_needs_pointwise_tile_schedule()` hardcoded Y-tiling path.

| Shape | Input | Output | Kernel | Groups |
|---|---|---|---|---:|
| `conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1` | `144x28x28` | `32x28x28` | `1x1` | 1 |
| `conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1` | `192x28x28` | `32x28x28` | `1x1` | 1 |

Related shapes with the same family but caught earlier by `pointwise_oc_tile`:

| Shape | Reason |
|---|---|
| `conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1` | OC tile planner catches it first |
| `conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1` | OC tile planner catches it first |
| `conv2d_b1_c192_h28_w28_oc16_wic192_k1x1_g1` | OC tile planner catches it first |
| `conv2d_b1_c256_h28_w28_oc32_wic256_k1x1_g1` | OC tile planner catches it first |

### `spatial_im2col`: 11/217

These are dense spatial shapes with high weight or output pressure. This is the
least desirable permanent path because it materializes im2col in Python and turns
spatial conv into many `1x1` tiles.

| Shape | Input | Output | Kernel | Groups |
|---|---|---|---|---:|
| `b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid` | `160x14x14` | `320x12x12` | `3x3` | 1 |
| `b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid` | `160x7x7` | `320x5x5` | `3x3` | 1 |
| `b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid` | `192x7x7` | `384x5x5` | `3x3` | 1 |
| `b1_c256_h10_w10_oc512_wic256_k3x3_g1_s1_pvalid` | `256x10x10` | `512x8x8` | `3x3` | 1 |
| `b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid` | `128x5x5` | `256x3x3` | `3x3` | 1 |
| `b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid` | `128x3x3` | `256x1x1` | `3x3` | 1 |
| `b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid` | `3x320x320` | `32x318x318` | `3x3` | 1 |
| `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` | `16x160x160` | `128x158x158` | `3x3` | 1 |
| `b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid` | `16x80x80` | `128x76x76` | `5x5` | 1 |
| `b1_c40_h40_w40_oc160_wic40_k3x3_g1_s1_pvalid` | `40x40x40` | `160x38x38` | `3x3` | 1 |
| `b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid` | `72x20x20` | `288x18x18` | `3x3` | 1 |

### `fallback_flat_y_tile`: 11/217

These are not part of the six explicit strategies, but they are still tiled in
the fallback path due to flat output stride or output-buffer pressure.

| Shape family | Count | Notes |
|---|---:|---|
| `conv2d_b1_c3_h52_w52_oc6_wic3_k1x1_g1` plus `1x3_54x54_k1..1x3_72x72_k1` | 11 | Large small-channel `1x1`, mostly artificial stride-stress sweep |

## ONNC / VP Findings

These checks used ONNC-generated `out_nv_small.nvdla` loadables on the `onnc/vp`
environment. Despite the file name, this is the ONNC FP16 / `nv_full`-style flow,
not true INT8 `nv_small` hardware. Tiling is therefore useful evidence for compiler
partitioning, not for exact INT8 trace matching.

### Parsed Operation Counts

The loadable parser was run with `python3 parse.py --cfg-writes <out_nv_small.nvdla>`.

| Shape | CONV ops | SDP ops | ONNC split evidence |
|---|---:|---:|---|
| `conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1` | 1 | 1 | No software tiling in ONNC |
| `conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1` | 1 | 1 | No software tiling in ONNC |
| `conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1` | 2 | 2 | BY_Y / output-height split |
| `conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1` | 2 | 2 | BY_Y likely |
| `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` | 2 | 2 | BY_Y / output-height split |
| `b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid` | 7 | 7 | Strong spatial split candidate |

Most `spatial_im2col` listed shapes still parse as `1 CONV + 1 SDP` in ONNC. The
large spatial-pressure examples above are the ones that actually showed multiple
hardware CONV/SDP operations.

### Captured BY_Y Example

Captured with `examples/vp/opendla_reglog.ko` to avoid full blob printk overload:

```text
conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1
```

Artifacts:

| File | Purpose |
|---|---|
| `ref/onnc-tutorial/models/conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1/capture.log` | VP/KMD register log |
| `ref/onnc-tutorial/models/conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1/parsed_capture.txt` | Parsed runtime writes |
| `ref/onnc-tutorial/models/conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1/parsed_loadable.txt` | Parsed loadable writes |

Runtime result:

```text
Test pass
```

Compiler split:

| Operation | Register | Value | Meaning |
|---|---|---:|---|
| CONV index 0 | `NVDLA_CACC.D_DATAOUT_SIZE_0_0` | `0x002c0037` | width 56, height 45 |
| CONV index 2 | `NVDLA_CACC.D_DATAOUT_SIZE_0_0` | `0x000a0037` | width 56, height 11 |

Total output height is `45 + 11 = 56`, so this is a clear ONNC BY_Y split.

The runtime DMA offsets also confirm tile placement:

| Tile | Register | Captured address |
|---|---|---:|
| first input tile | `NVDLA_CDMA.D_DAIN_ADDR_LOW_0_0` | `0xc0100000` |
| second input tile | `NVDLA_CDMA.D_DAIN_ADDR_LOW_0_0` | `0xc0113b00` |
| first output tile | `NVDLA_SDP.D_DST_BASE_ADDR_LOW_0` | `0xc0040000` |
| second output tile | `NVDLA_SDP.D_DST_BASE_ADDR_LOW_0` | `0xc0053b00` |

Captured scheduling order:

```text
Program Convolution operation index 0
Program Convolution operation index 2
Program SDP operation index 1
Program SDP operation index 3
```

So the KMD does not invent the tiling. It consumes compiler-produced descriptors,
programs both CONV tiles, then programs the matching SDP output tiles.

Diff notes:

| Difference | Interpretation |
|---|---|
| `S_POINTER` values | Runtime ping-pong group selection |
| DMA base addresses | Runtime-resolved IOVA addresses |
| `S_INTR_STATUS` writes | Runtime interrupt clears |
| SDP order vs loadable order | Scheduler dependency behavior, not a shape mismatch |

### Parser Note

The c96 capture exposed and fixed a parser packing bug: CONV/CDMA/CSC
`data_reuse`, `weight_reuse`, `skip_data_rls`, and `skip_weight_rls` fields are
at hardware bits `16`, `20`, `24`, and `28`, not consecutive bits `16..19`.

## Interpretation

The current six-strategy structure is mostly an artifact of local reverse
engineering and incremental fixes. The shape distribution suggests only a few
real pressure families:

| Pressure family | Current branches |
|---|---|
| output-channel / K pressure | `pointwise_oc_tile`, `spatial_oc_serial`, part of `spatial_im2col` |
| spatial / Y pressure | `depthwise_spatial_tiled`, `fallback_flat_y_tile`, `pointwise_y_tile_hardcoded`, part of `spatial_im2col` |
| grouped conv lowering | `grouped_serial` |
| dense spatial high weight pressure | `spatial_im2col`, should ideally become direct spatial Y/K split |

That maps naturally to a single planner with `split_method`:

| Split method | Meaning |
|---|---|
| `NONE` | full input and full weight fit |
| `BY_Y` | split output rows / input height window |
| `BY_K` | split output channels / kernels |
| `BY_YK` | split both rows and output channels |

## DeepWiki Notes

DeepWiki on `ONNC/onnc`:

- ONNC's NVDLA backend uses a CBUF allocation feasibility model rather than six
  separate runtime strategies.
- Important concepts/functions reported by DeepWiki: `getCbufAllocType`,
  `tryAllocateDataAndWeightsIntoCBuf`, `SplitConvPass`, `SplitGroupConvPass`, and
  `NvDlaBackend::getMaxNumOfConvChannels`.
- It uses allocation categories like full-data/full-weight, full-data/partial-weight,
  split-data/full-weight, split-data/partial-weight, minimum-weight, and unfeasible.

DeepWiki on `nvdla/sw`:

- Planning is mainly in the UMD compiler, not the KMD submit path.
- Reported source focus: `umd/core/src/compiler/engine-ast/ConvCoreNode.cpp`.
- Important functions reported by DeepWiki: `splitNodes()`, `splitNodesInternal()`,
  `determineSplitDataRatios()`, and `splitData()`.
- The planner evaluates full input/full weight, full input with hardware weight
  split, partial-H data split, and software split-K fallback in a prioritized
  decision tree.
- Firmware/KMD register code mostly consumes precomputed `data_bank` and
  `weight_bank` values.

## Recommendation

Do not spend the next step doing a broad cleanup of `examples/conv_tiles.py` yet.

Reason: `examples/conv_tiles.py` is already implementing the RKNN reverse result,
but it is still likely encoding the current understanding too literally. If we
clean it now, we risk polishing the wrong abstraction and still keeping many
strategy-specific branches.

Recommended next step:

1. Use NVDLA VP / ONNC / NVDLA UMD to inspect the difficult shapes first,
   especially the `spatial_im2col` and `pointwise_y_tile_hardcoded` groups.
2. Compare their split decisions against a simple local `NONE/BY_Y/BY_K/BY_YK`
   planner.
3. After that, clean or rewrite toward one small planner, preferably in a new
   compact file or by reducing `examples/conv_tiles.py` around that planner.

Practical short-term target:

- Keep `examples/conv_tiles.py` as the working/reference implementation for now.
- Do not clean all 2k lines yet.
- Add a small offline planner/probe script that classifies these 217 shapes into
  `NONE/BY_Y/BY_K/BY_YK` using CBUF bank pressure.
- Use the NVDLA VP result on the rare/difficult shapes as the deciding evidence
  before deleting branches from `conv_tiles.py` or other `conv*.py` files.

If the NVDLA VP/ONNC split behavior matches the simple planner on the rare cases,
then the likely clean path is a `<1000` line `conv.py` built around one Y/K tiler.
If it does not match, keep the rare branches isolated and documented rather than
spreading them through the main implementation.

# ONNC
rk3588npu
- TRM: 1024 int8 MAC operations per cycle, 512 float16 MAC operations per cycle

RK3588 / Rocket CBUF size:

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

Compute:

```text
per bank = 32768 bytes = 32 KiB
entry size = 32768 / 256 = 128 bytes
total CBUF = 12 * 32768 = 393216 bytes = 384 KiB
```

So:
- RK3588 NPU CBUF = 12 banks
- per bank = 32 KiB
- total = 384 KiB
- entry size = 128 bytes
- entries per bank = 256

Register note: `CNA_CBUF_CON0.DATA_BANK` and `WEIGHT_BANK` are 4-bit fields, so
the register encoding can represent more than 12, but the RK3588/Mesa constants
use 12 physical CBUF banks for planning.

NVDLA nv_full
```
#define CBUF_BANK_NUMBER_16
#define CBUF_BANK_WIDTH_64
#define CBUF_BANK_DEPTH_512
Compute:
CBUF total = bank_count * bank_width * bank_depth
           = 16 * 64 bytes * 512
           = 524288 bytes
           = 512 KiB
So:
nv_full CBUF = 16 banks
per bank = 64 * 512 = 32768 bytes = 32 KiB
total = 16 * 32 KiB = 512 KiB
```

Comparison:

| Target | Banks | Per bank | Total CBUF | Entry size | Entries/bank |
|---|---:|---:|---:|---:|---:|
| RK3588 / Rocket | 12 | 32 KiB | 384 KiB | 128 B | 256 |
| NVDLA `nv_full` | 16 | 32 KiB | 512 KiB | 64 B from ONNC config width unit, or 128 B effective if comparing RK-style entries | 512 config-depth units |

Important implication: RK3588 has the same 32 KiB bank size as `nv_full`, but only
12 banks instead of 16. Any ONNC/NVDLA `nv_full` shape that fits without tiling can
still need BY_Y/BY_K tiling on RK3588 because RK3588 has 25% less total CBUF.

Confirmed tiled by ONNC:
- conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1: 2 CONV + 2 SDP, captured, BY_Y split 45 + 11 output rows.
- conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1: 2 CONV + 2 SDP, likely BY_Y.
- b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid: 2 CONV + 2 SDP, BY_Y.
- b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid: 7 CONV + 7 SDP, strong spatial/Y split.
Confirmed not tiled by ONNC:
- conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1: 1 CONV + 1 SDP.
- conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1: 1 CONV + 1 SDP.
