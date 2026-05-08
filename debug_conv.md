# Debugging failed shape in `conv.py`

## Shape

```
dict(name="known_2x2_1x1_4x4",  batch=1, in_c=2, in_h=4, in_w=4, out_c=2, weight_in_c=2, kh=1, kw=1, groups=1),
```

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
