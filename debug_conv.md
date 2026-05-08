# Debugging failed shape in `conv.py`

## Shape

See the follow-up sections below for the active cc/debug shapes.

Meaning: in_c=2, out_c=2, kernel 1x1, input 4x4, weight_in_c=2 (= in_c). Name `known_2x2_1x1_4x4` = known marker, 2 in-ch × 2 out-ch, 1×1 kernel, 4×4 input.

## Symptom

Row 0 of output is correct (max_diff < 0.001), rows 1-3 are garbage (max_diff=inf, values like 704, -40448, inf).

## Step 1: Check if shape works in `~/npu/ops_reg`

Build and run:
```sh
cd ~/npu/ops_reg
gcc -o main main.c -ldrm -lm -I../include
./main conv2d 1 2 4 4 2 2 1 1 1
```

This calls `test_conv2d()` at `main.c:6273` with positional args:
`<batch> <in_c> <in_h> <in_w> <out_c> <weight_in_c> <kh> <kw> [groups]`

- **PASS** → bug is in `conv.py`'s register generation or data pack/unpack
- **FAIL** → the conv2d runtime logic in `ops_reg` also has the same bug

## Step 2: If ops_reg passes but conv.py fails

Remaining register differences after `_feature_grains` fix (`min` not `max`):

| Register | conv.py | RKNN (working) |
|---|---|---|
| CNA_CONV_CON2 FEATURE_GRAINS | 5 (was 1024 before fix ✓) | 5 |
| CNA_CBUF_CON0 DATA_BANK | 1 (✓) | 1 |
| CNA_CBUF_CON1 DATA_ENTRIES | 16 | 1 |
| CNA_CVT_CON5 (0x1180) | 0x07 (input_pack_c2 - 1 = 7) | not written |
| _unpack_flat_1x1_output | contiguous read | N/A (RKNN uses OutputOperator on CPU) |

### Suspect A: `CNA_CVT_CON5` at `conv.py:432`

Writes `input_pack_c2 - 1 = 7` to `PER_CHANNEL_CVT_EN`, enabling per-channel conversion on 3 pseudo-channels. RKNN doesn't write this register. Even with `CVT_BYPASS=1`, enabling per-channel mode may alter internal data routing.

**Fix:** Comment out or conditionalize for NHWC pack only.

### Suspect B: Output unpack (`_unpack_flat_1x1_output` at `conv.py:322`)

Reads contiguous 128 FP16 elements (`c1 * out_width_stride * c2 = 1 * 16 * 8`). The DPU writes with `DST_SURF_STRIDE=256` bytes between rows. If the NPU writes output with row stride, the contiguous read only captures row 0 + padding.

**Fix:** Verify raw output buffer layout. If strided, read row-by-row with stride.

### Suspect C: `DATA_ENTRIES=16`

RKNN uses 1 entry (128 bytes). conv.py allocates 16 entries (2048 bytes). Unlikely to cause wrong computation, but worth checking if CBUF allocation conflicts with weight.

## Step 3: Dump registers from ops_rknn for comparison

Already done — captured RKNN compiler registers to `/tmp/ops_rknn_emit.txt`.

To re-dump:
```sh
cd ~/npu/ops_rknn
echo 'set pagination off
break rknn_destroy
commands
    shell python3 dump.py 2 2>&1 | grep EMIT | sed "s/\\x1B\\[[0-9;]*[a-zA-Z]//g" | grep -v "0x00000000" > /tmp/ops_rknn_emit.txt
    continue
end
run --case known_2x2_1x1_4x4 1 2 4 4 2 1 1 1
quit' > /tmp/dump_rknn.gdb
gdb -batch -x /tmp/dump_rknn.gdb ./conv2d_multi
```

## Step 4: Apply fixes to `conv.py`

1. `_feature_grains` — change `max` to `min` (already applied)
2. `CNA_CVT_CON5` — tested removing for non-NHWC; rejected because it regressed `conv2d_16x16_1x1_8x8`
3. Output unpack — checked against `experimental/conv_mesa.py`; current flat 1x1 unpack matches that reference
4. Root cause — 2-channel small 1x1 input must use NHWC/pixel input path, like the existing 1- and 3-channel small cases. Fix `should_use_nhwc_pack()` to include `channels == 2 and c2 // channels == 4`.

## Result

`python examples/conv.py` now passes the full sweep, including:

```text
known_2x2_1x1_4x4  2x4x4 -> 2x4x4  kh=1 kw=1 g=1  PASS  (max_diff=0.0017)
```

## Follow-up: known issue sweep

All previously commented `known_issue_shapes` were enabled one-by-one and now pass in `examples/conv.py`:

```text
known_2x2_1x1_4x4
known_8x8_1x1_5x5
known_10x20_3x3_9x9
known_16x16_3x3_9x9
known_2x4_3x3_6x6
known_2x4_2x2_5x5
known_1x32_5x5_10x10
known_8x4_4x4_10x10
```

For `known_10x20_3x3_9x9`, `ops_reg` also failed, so the working RKNN compiler path was dumped from `~/npu/ops_rknn` using `conv2d_fail_10x20_3x3_9x9.rknn`. RKNN showed:

```text
Native input:  NC1HWC2, dims 1 2 9 9 8
Native output: NC1HWC2, dims 1 4 1 52 8
CNA_WEIGHT_SIZE2 kernels=20
DPU_DST_SURF_STRIDE=52
DPU_SURFACE_ADD=104
```

Fixes derived from that dump:

1. Use input `C2=8` for non-depthwise/grouped inputs with `in_c > 4`.
2. Pack KH-major spatial weights in output-channel blocks of 16.
3. Read non-grouped spatial output as flattened `NC1HWC2` (`height=1`, `width=out_h*out_w`, `stride=out_width_stride`).
4. Read aligned output planes using `align_out_c / 8`, not only `ceil(out_c / 8)`.
5. Program `DPU_SURFACE_ADD` with 16-channel surface grouping (`max(2, effective_align_out // 16)`).

## Follow-up: large RKNN pc-chain shape

Added and fixed:

```text
conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1
input:  1x3x224x224
weight: 32x3x3x3
output: 1x32x222x222
```

`~/npu/ops_reg` is expected to fail this because it does not support the RKNN pc-chain schedule for the large convolution. RKNN was dumped from `~/npu/ops_rknn` using:

```sh
python3 gen_conv2d_models.py --custom --batch 1 --in-ch 3 --height 224 --width 224 --out-ch 32 --k-h 3 --k-w 3 --groups 1 --name conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1 --force --out-dir models
gdb -q ./conv2d_multi -ex "set pagination off" -ex "break rknn_outputs_get" -ex "run --case conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1 1 3 224 224 32 3 3 1 conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1" -ex "shell python3 dump.py 1 2 3 4 5 > dump/conv2d_cc_dump.txt" -ex "continue" -ex "quit"
```

RKNN showed five height tiles: four `50`-input-row tiles producing `48` output rows each, followed by one `32`-input-row tile producing `30` output rows. Important register values from the dump:

```text
CNA_CONV_CON2 FEATURE_GRAINS=50 for 50-row tiles
CNA_CBUF_CON0 DATA_BANK=11, WEIGHT_BANK=1
DPU_DST_SURF_STRIDE=49284 (full output surface stride)
DPU_SURFACE_ADD=98568
```

Fixes:

1. Large non-grouped spatial convs now split into 48-output-row tiles and submit those tiles as one PC chain.
2. Each chained tile gets its own input DMA offset and an output DMA base offset for the tile row.
3. `make_conv2d_regs()` accepts an output stride override so tiled DPU writes still use the full output surface stride.
4. NHWC spatial convs now use `FEATURE_GRAINS=in_h` and all 11 data CBUF banks, matching RKNN and avoiding partial-row corruption.

Result:

```text
conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1 3x224x224 -> 32x222x222 kh=3 kw=3 g=1 PASS (max_diff=0.0151)
```

## Follow-up: ResNet-style cc shapes

The new cc shapes listed at the top of this file were added to `known_issue_shapes` in `examples/conv.py` and are being debugged one by one.

### Fixed so far

```text
conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32  32x112x112 -> 32x110x110  kh=3 kw=3 g=32  PASS (max_diff=0.0075)
conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1  32x112x112 -> 64x112x112  kh=1 kw=1 g=1   PASS (max_diff=0.0146)
conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1   64x56x56   -> 128x56x56   kh=1 kw=1 g=1   PASS (max_diff=0.0156)
```

### RKNN dumps captured

```text
~/npu/ops_rknn/dump/conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32_dump.txt
~/npu/ops_rknn/dump/conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1_dump.txt
~/npu/ops_rknn/dump/conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1_dump.txt
```

### Depthwise 32x112 fix

RKNN schedules `conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32` as a pc-chain, not as a single full-height task. Key register values:

```text
CNA_CONV_CON2 FEATURE_GRAINS=15
CNA_CBUF_CON0 DATA_BANK=11, WEIGHT_BANK=1
CNA_CBUF_CON1 DATA_ENTRIES=112
CNA_DATA_SIZE0 height=50 for 48-output-row tiles
DPU_DST_SURF_STRIDE=12100
DPU_SURFACE_ADD=48400
```

Fixes applied:

1. Depthwise spatial convs with `out_h > 48` now enter the shared pc-chain height tiling path.
2. `_feature_grains()` returns `15` for depthwise spatial convs, matching RKNN.
3. `_data_bank()` returns all 11 data banks for depthwise spatial convs.
4. The 224 spatial pc-chain shape was moved to the end of the sweep because running it before depthwise pc-chain shapes poisons later chained submissions even after explicit resets.

### Pointwise 32->64 fix

RKNN schedules `conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1` as height-tiled pc-chain tasks. Important findings:

```text
CNA_CONV_CON2 FEATURE_GRAINS=10
CNA_CBUF_CON0 DATA_BANK=11
CNA_CBUF_CON1 DATA_ENTRIES=112
CNA_FC_DATA_SIZE1 DMA_CHANNEL=32
```

The raw path also needed output-channel sub-tiling: without it, channels `0..15` were correct and channels `16+` were corrupt.

Fixes applied:

1. Large non-spatial pointwise convs with `out_h > 50` now use pc-chain height tiling.
2. `FEATURE_GRAINS`, `DATA_BANK`, `CNA_CBUF_CON1`, and `CNA_FC_DATA_SIZE1` now use the aligned real input channel count for wide inputs, not just the CBUF lane width.
3. The 32->64 pointwise shape uses 16-output-channel block tasks with weight and output-surface offsets.
4. Those output-channel block tasks are submitted as one PC chain, then the whole chain is submitted twice. Submitting each 16-channel tile as an independent job still left intermittent stale-state failures across repeated `python examples/conv.py` runs.

### Pointwise 64->128 fix

`conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1` failed with `max_diff` around `74` after enabling the next known-issue shape. The first 16 output channels in each wide tile were close, but later channels were corrupt.

Root cause: the default 1x1 weight pack serialized each output channel with all 64 input channels contiguous. That happens to match the 16x32 hardware order when `in_c == 32`, but not when `in_c >= 64`. The wide pointwise path needs the same 16-output-channel by 32-input-channel chunk order used by the GEMM references:

```text
weight.reshape(out_c/16, 16, in_c/32, 32).transpose(0, 2, 1, 3)
```

Fix applied:

1. Added `_pack_pointwise_wide()` for non-grouped 1x1 convolutions with `in_c >= 64`.
2. Kept the proven 16-output-channel task tiling; switching to 64-output-channel tiles made only the first 16 channels of each tile correct.

Verified:

```text
conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1 PASS (max_diff=0.0156)
python examples/conv.py PASS with this known-issue shape enabled
```

### Final sweep result

`python examples/conv.py` passed the active sweep for this debugging session.

Additional depthwise fixes applied:

1. Depthwise spatial weight packing now handles all spatial kernels, not just 3x3. This fixed `conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024`.
2. Large depthwise spatial shapes use 32-channel block execution with per-block input and weight repacking.
3. The 28x28 depthwise case is split into two 13-output-row tiles because a single 26-output-row tile is unstable.
4. Depthwise spatial `FEATURE_GRAINS` is capped at `13`, which avoids the unstable 28x28 grain settings while preserving the passing larger/smaller cases.

Known remaining work:

1. Remaining wider non-grouped 1x1 pointwise cc shapes beyond the active 64->128 case still need to be enabled and validated one by one.
2. The empirical double-submit state warm-ups should be reduced once a register-level reset/isolation sequence is identified.

Representative final output:

```text
conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1    PASS (max_diff=0.0146)
conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32    PASS (max_diff=0.0075)
conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64    PASS (max_diff=0.0075)
conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1      PASS (max_diff=0.0151)
conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1     PASS (max_diff=0.0156)
```

Next real NPU work:

1. Continue enabling the remaining wider pointwise 1x1 cases and validate whether the 16x32 weight packing generalizes.
2. Keep state-transition regressions covered in `examples/conv.py` while reducing empirical double-submit workarounds where possible.

### Known regressions avoided

Do not globally remove or zero `CNA_CVT_CON5`: removing it previously regressed `conv2d_16x16_1x1_8x8`. A later test of zeroing it for wider pointwise shapes did not fix `64->128` and was reverted.

### Intermittent 32->64 pointwise failure fix

`conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1` could fail intermittently, including cold first-run failures in isolation (`max_diff` around `48`) and failures after large depthwise work in the full sweep (`max_diff` around `32`). The issue was stale task/ping-pong state, not `CNA_CVT_CON5` or weight packing.

Fixes applied in `examples/conv.py`:

1. `submit_conv_tasks()` now uses `RKNPU_JOB_PC | RKNPU_JOB_BLOCK`, matching the Mesa-style reference submit path, instead of also setting `RKNPU_JOB_PINGPONG` for every raw conv task.
2. Depthwise channel-tiled pc-chain blocks are submitted twice, keeping the existing stale-state warm-up strategy local to the depthwise block path.
3. The 32->64 and 64->128 pointwise cases exercise the NPU path; remaining wider 1x1 pointwise cases still need separate validation.

Verified:

```text
python examples/conv.py
```

The current sweep passes, including the active cc transition coverage. Representative cc results:

```text
conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1    PASS (max_diff=0.0146)
conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32    PASS (max_diff=0.0075)
conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1      PASS (max_diff=0.0151)
conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64    PASS (max_diff=0.0075)
conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1     PASS (max_diff=0.0156)
```

### 224 spatial pc-chain poisoning following depthwise

`conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1` and `conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32` both passed independently, but running the 112-depthwise case after the 224 spatial pc-chain failed (`max_diff` around `14.67`). This reproduced as a persistent stale PC-chain state failure: subsequent 112-depthwise retries in the same process also failed.

Root cause: the last task in `write_regs_to_npu_task()` only emitted `PC_OPERATION_ENABLE`. RKNN dumps show chain boundaries clearing `REG_PC_REGISTER_AMOUNTS` to `0`, and stale nonzero PC next-task state can survive into the next raw conv despite per-submit reset.

Fix applied:

1. Final PC tails now write `PC_BASE_ADDRESS=0` and `PC_REGISTER_AMOUNTS=0` before `PC_OPERATION_ENABLE`.
2. The main sweep includes `conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1` immediately before `conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32` so the poisoning order is covered by `python examples/conv.py`.

Verified:

```text
conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1   PASS (max_diff=0.0151)
conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32 PASS (max_diff=0.0075)
```

### 32->64 pointwise intermittent follow-up

`conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1` still showed intermittent first-run/full-sweep failures (`max_diff` around `49`) after the PC tail clear fix. The failing path is the 16-output-channel independent tile loop used by the 32->64 pointwise NPU implementation.

Fix applied:

1. Each independent pointwise output-channel tile is submitted twice before moving to the next output-channel tile.
2. This was re-tested after final PC tail clearing, so the earlier pointwise-to-depthwise poisoning did not recur.

Follow-up fix:

1. The per-output-channel independent submits were still intermittently stale across repeated full-process sweeps (`max_diff` around `44-52`).
2. The pointwise output-channel tile regs are now accumulated into one PC chain and the complete chain is submitted twice.
3. Testing `PC_BASE_ADDRESS=1` as a regcmd preamble regressed the 224 spatial conv, and post-submit reset did not fix the instability, so both experiments were rejected.

Verified:

```text
conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1 PASS (max_diff=0.0146)
conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64 PASS (max_diff=0.0075)
```

Also verified `python examples/conv.py` plus three additional full sweeps in a row.

Latest verification:

```text
python examples/conv.py                  PASS, 10 consecutive full-process sweeps
targeted cc transition stress sequence   PASS, 10 consecutive runs
```

## Files involved

| File | Location |
|---|---|
| Register programming | `~/rk3588/examples/conv.py` |
| ops_reg test runner | `~/npu/ops_reg/main.c` (test_conv2d at line 6273) |
| RKNN reference dump | `/tmp/ops_rknn_emit.txt` |
| GDB dump script | `~/npu/ops_rknn/rknn.gdb` |
| Register decoder | `~/npu/ops_rknn/dump.py` |
| Register definitions | `~/npu/ops_rknn/registers.xml` |
| RKNN test model | `~/npu/ops_rknn/ops_rknn/models/known_2x2_1x1_4x4.rknn` |
