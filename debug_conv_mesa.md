# Debugging conv shapes with Mesa reference source

Mesa source code is at `/home/orangepi/rk3588/ref/mesa`.

The Mesa Rocket driver (`src/gallium/drivers/rocket/`) is the canonical software
reference for RK3588 NPU register programming.  It is not a standalone debug
runner like `conv.py`; it is the driver that processes TFLite operations through
the Teflon delegate and emits register command streams.

Use this file to navigate the Mesa source when debugging a `conv.py` failure.

---

## Mesa source files

| File | Lines | Purpose |
|---|---|---|
| `rkt_ml.c` | 638 | Operation lowering: TFLite/Teflon → Rocket operations.  Records input/output shapes, padding, stride, depthwise, zero points, scales, weight/bias buffers. |
| `rkt_ml.h` | 155 | Data structures: `rkt_operation`, `split_task`, CBUF constants (`CBUF_BANK_SIZE=32768`, `CBUF_BANKS=12`, `FEATURE_ATOMIC_SIZE=16`, `WEIGHT_ATOMIC_SIZE=32`). |
| `rkt_task.c` | 327 | Task splitting: CBUF entries-per-slice calculation, input bank budget, weight bank budget, `fill_task()`, `rkt_split_tasks()`. |
| `rkt_regcmd.c` | 544 | Register emission: `fill_first_regcmd()` programs all CNA/CORE/DPU registers for one task slice. |
| `rkt_coefs.c` | 215 | Weight packing: `rkt_fill_weights()` reorders quantized weights into hardware layout. |
| `intercept.c` | — | Debug: reads back register command buffers for logging. |
| `rkt_device.c/h` | — | Device init, BO management, submit helpers. |

Upstream kernel register definitions:

| File | Purpose |
|---|---|
| `ref/linux/drivers/accel/rocket/rocket_registers.h` | Auto-generated bitfield masks and shifts for every NPU register.  Ground truth for register field encodings. |

---

## Mapping `conv.py` functions to Mesa source

| `conv.py` function | Mesa equivalent | Key behavior |
|---|---|---|
| `_conv_params()` | `fill_task()` in `rkt_task.c` | Computes aligned channels, strides, output dimensions |
| `_conv_align_c()` | `FEATURE_ATOMIC_SIZE = 16` alignment | Input channels aligned to 16 for non-small cases |
| `_conv_tiles()` | `rkt_split_tasks()` in `rkt_task.c` | Splits conv into height-tiled tasks fitting CBUF |
| `_feature_grains()` | `CNA_CONV_CON2` field in `rkt_regcmd.c:74-75` | Feature grains = 50 + stride_y + 1 (magic) |
| `_cbuf_entries()` | `calc_entries_per_slice()` in `rkt_task.c:10-26` | CBUF entries from channel atoms × width |
| `_data_bank()` | `calc_input_banks()` in `rkt_task.c:29-34` | Input CBUF banks from entries × height |
| `_pack_pointwise_wide()` | `rkt_fill_weights()` in `rkt_coefs.c` | Weight reorder: 16-OC × 32-IC chunks |
| `make_conv2d_regs()` | `fill_first_regcmd()` in `rkt_regcmd.c` | Emits all CNA/CORE/DPU registers |
| `write_regs_to_npu_task()` | `rkt_device.c` submit path | Writes regs to BO, submits via DRM |

---

## Step 1: Reproduce shape on real Mesa/Teflon

Run the full TFLite model through Mesa's Teflon delegate to produce ground-truth
registers and outputs:

```sh
cd ~/mesa
source .venv/bin/activate

# mobilenetv1 (fast, ~11ms on NPU)
TEFLON_DEBUG=verbose ETNA_MESA_DEBUG=ml_msgs python3.10 \
  src/gallium/frontends/teflon/tests/classification.py \
  -i ./grace_hopper.bmp \
  -m src/gallium/targets/teflon/tests/models/mobilenetv1/mobilenet_v1_1_224_quant.tflite \
  -l src/gallium/frontends/teflon/tests/labels_mobilenet_quant_v1_224.txt \
  -e build/src/gallium/targets/teflon/libteflon.so
```

The `TEFLON_DEBUG=verbose` output logs every conv op's input/output shapes,
padding, stride, and whether it is depthwise.

---

## Step 2: Extract shape from Mesa logs

Find the conv op in the `TEFLON_DEBUG=verbose` output that matches your failing
shape.  Mesa logs tensors in NHWC order:

```
operation 0: CONV
  input: 1x224x224x3
  weight: 32x3x3x3
  output: 1x112x112x32
  padding: 1 1 1 1, stride: 2
```

Translate to `conv.py` meaning:

- Input NCHW from NHWC: `1x3x224x224`
- Weight OIHW from OHWI: `32x3x3x3`
- output NCHW: `1x32x112x112`
- If stride != 1 or padding != valid → Mesa-only shape, not exact `conv.py`

---

## Step 3: Compare CBUF bank allocation

Mesa's `rkt_task.c:calc_input_banks()` and `calc_weights_banks()` determine the
CBUF split.  Key formulas:

```c
// calc_entries_per_slice (rkt_task.c:10-26)
total_c_atomics = ceil(input_channels * 1B / 16B)  // 16B FEATURE_ATOMIC_SIZE
atomics_per_entry = 128B / 16B = 8                  // CBUF_ENTRY_SIZE / FEATURE_ATOMIC_SIZE
int_c_entries = (total_c_atomics / 8) * input_width
frac_c_entries = ... for remainder >= 3

// calc_input_banks (rkt_task.c:29-34)
entries_per_slice * input_height / 256 entries_per_bank

// calc_weights_banks (rkt_task.c:37-54)
bytes = kw * kh * input_channels * 1B
if not depthwise: bytes *= output_channels
entries = ceil(bytes / 128)
banks = ceil(entries / 256) + 1   // +1 safety bank
```

In `conv.py`, compare:

- `_cbuf_entries()` → `calc_entries_per_slice()`
- `_tile_data_bank()` / `_data_bank()` → `calc_input_banks()`
- Weight bank is implicit in `CNA_CBUF_CON0` → `calc_weights_banks()`

If `conv.py` uses a different number of data banks or weight banks than Mesa,
the CBUF allocation may be wrong and cause data corruption.

---

## Step 4: Compare register fields

Mesa's `rkt_regcmd.c:fill_first_regcmd()` emits the full register sequence.
For each register, check that `conv.py`'s `make_conv2d_regs()` produces the same
value.

| Register | `rkt_regcmd.c` line | `conv.py` line | Check |
|---|---|---|---|
| `CNA_CBUF_CON0` | 44-48 | 568-571 | `WEIGHT_BANK`, `DATA_BANK`, `WEIGHT_REUSE` |
| `CNA_CONV_CON1` | 54-63 | 543-547 | precision, pixel flags, depthwise mode |
| `CNA_CONV_CON2` | 74-75 | 548-550 | `FEATURE_GRAINS` (Mesa: `50 + stride_y + 1`) |
| `CNA_CONV_CON3` | 76-77 | 551-553 | stride values |
| `CNA_DATA_SIZE0` | 78-80 | 554-556 | input width, height |
| `CNA_DATA_SIZE1` | 82-84 | 557-559 | `DATAIN_CHANNEL` (aligned), `DATAIN_CHANNEL_REAL` |
| `CNA_DATA_SIZE2` | 86 | 560 | output width |
| `CNA_DATA_SIZE3` | 87 | 561 | output atomics (out_w * out_h) |
| `CNA_WEIGHT_SIZE0` | 88-89 | 562 | total weight bytes |
| `CNA_WEIGHT_SIZE1` | 90-91 | 563 | bytes per kernel |
| `CNA_WEIGHT_SIZE2` | 92-95 | 564-567 | kernel width, height, count |
| `CNA_CBUF_CON1` | 99 | 572 | data entries |
| `CNA_CVT_CON0` | 124-126 | 573-575 | bypass, type, sign |
| `CNA_DMA_CON1` | 143 | 584 | line stride |
| `CNA_DMA_CON2` | 144 | 585 | surface stride |
| `CNA_FC_DATA_SIZE0` | 146-148 | 586-588 | DMA width, height |
| `CNA_FC_DATA_SIZE1` | 150-151 | 589 | DMA channel |
| `CNA_DCOMP_ADDR0` | 154 | 590 | weight base address |
| `CNA_CVT_CON5` | 173-176 | 591 | per-channel CVT enable mask |
| `CORE_MISC_CFG` | 192-196 | 592-594 | depthwise, spatial enable |
| `CORE_DATAOUT_SIZE_0` | 197-199 | 595-597 | output height, width |
| `CORE_DATAOUT_SIZE_1` | 200-201 | 598 | output channel |
| `DPU_FEATURE_MODE_CFG` | 206-210 | 600-603 | burst, output mode, conv mode |
| `DPU_DATA_FORMAT` | 212 | 604-607 | precision fields |
| `DPU_DST_BASE_ADDR` | 214-216 | 608 | output base address |
| `DPU_DST_SURF_STRIDE` | 217-218 | 609 | output surface stride |
| `DPU_DATA_CUBE_WIDTH` | 219-220 | 610 | cube width |
| `DPU_DATA_CUBE_CHANNEL` | 224-226 | 613-615 | `ORIG_CHANNEL`, `CHANNEL` |
| `DPU_BS_CFG` | 227-229 | 616-619 | bypass config |

---

## Step 5: Check output channel alignment

Mesa's `fill_task()` in `rkt_task.c:86-92` pads output channels:

```c
task->output_channels = align(MAX2(operation->output_channels, 32), 32);
if (operation->depthwise) {
   if (task->output_channels_real <= 32)
      task->output_channels *= 2;
   task->output_channels = align(task->output_channels, 64);
}
```

Key rule: **output channels are always aligned up to 32** (not 16).  Depthwise
doubles to 64 when real <= 32.

In `conv.py`:
- `align_out_c = max(16, _align_up(out_c, 16))` — aligns to 16, not 32
- Depthwise `out_channel_field = _align_up(align_out_c, 32) - 1` — aligns to 32,
  then another `align_up` to 64 if <= 32 real

Mesa always uses 32 as the minimum output channel alignment, while `conv.py` uses
16.  This affects `CORE_DATAOUT_SIZE_1`, `DPU_DATA_CUBE_CHANNEL`, and
`DPU_SURFACE_ADD`.  For shapes where `out_c < 32`, Mesa programs 32 output
channels and only reads back the real count.

---

## Step 6: Check input channel alignment

Mesa's `fill_task()` in `rkt_task.c:76-78`:

```c
task->input_channels = align(MAX2(operation->input_channels, FEATURE_ATOMIC_SIZE), FEATURE_ATOMIC_SIZE);
task->input_channels_real = operation->input_channels;
```

Input channels always aligned to `FEATURE_ATOMIC_SIZE = 16`, never less.  In
`conv.py`, `align_c` can be 8 for small-input cases (`in_c <= 4` and not
depthwise/grouped).  This small-channel path is valid but does not match Mesa's
minimum-16 rule.

---

## Step 7: Check weight packing

Mesa's `rkt_coefs.c:rkt_fill_weights()` packs quantized uint8 weights.  The
layout is:

```c
// Loop order: oc1 → ic1 → x → y → oc2 → ic2
for oc1 in ceil(out_c / 32):
    for ic1 in ceil(in_c / 32):
        for x in kw:
            for y in kh:
                for oc2 in min(32, out_c):
                    for ic2 in min(32, in_c):
                        // output at index =
                        //   oc1*ic1_blocks*kh*kw*32*32 +
                        //   ic1*kh*kw*32*32 +
                        //   x*kh*32*32 +
                        //   y*32*32 +
                        //   oc2*32 +
                        //   ic2
```

Key properties:
- Groups of 32 output channels × 32 input channels
- Within each group: spatial loop outermost, then OC, then IC innermost
- `WEIGHT_ATOMIC_SIZE = 32` (not 16)

In `conv.py`, `_pack_pointwise_wide()` for `in_c >= 64` uses:
- 16-output-channel × 32-input-channel chunks
- Different OC group size (16 vs 32)

If weight packing doesn't match what the registers describe (bytes per kernel,
total bytes), the hardware reads wrong data.

---

## Step 8: Compare with upstream kernel register definitions

When bitfield values seem wrong, check `ref/linux/drivers/accel/rocket/rocket_registers.h`.
This header is auto-generated from Mesa's `registers.xml` and is the **ground
truth** for bitfield masks and shifts.

```c
// Example from rocket_registers.h:
#define CNA_CBUF_CON0_DATA_BANK_SHIFT           0
#define CNA_CBUF_CON0_DATA_BANK_MASK            0x0000000f
#define CNA_CBUF_CON0_WEIGHT_BANK_SHIFT         4
#define CNA_CBUF_CON0_WEIGHT_BANK_MASK          0x000000f0
```

Compare `conv.py`'s raw register values against what the bitfield macros
produce in Mesa's `rkt_regcmd.c`:

```c
// Mesa rkt_regcmd.c:44-45
uint32_t con0 = CNA_CBUF_CON0_WEIGHT_BANK(task->weights_banks) |
                CNA_CBUF_CON0_DATA_BANK(task->input_banks);
```

If `conv.py` emits `E(reg.CNA, reg.CNA_CBUF_CON0, 0xab)` but Mesa would have
emitted `0xba`, the bank split is swapped.

---

## Step 9: Mesa-Teflon one-layer validation

Mesa can generate a one-layer TFLite model and validate against the NPU:

```sh
cd ~/mesa
source .venv/bin/activate

# Generate a TFLite model for a single conv layer
build/src/gallium/targets/teflon/test_teflon generate_model \
  conv 1 4 4 3 2 2 4 1 /tmp/test_conv.tflite

# Run with CPU vs Teflon delegate
python3.10 src/gallium/frontends/teflon/tests/single_op_test.py \
  /tmp/test_conv.tflite \
  -e build/src/gallium/targets/teflon/libteflon.so
```

This confirms whether the Mesa driver itself handles a given shape correctly.
If Mesa passes but `conv.py` fails, the bug is in `conv.py`'s register
generation or data packing.

---

## Step 10: Mesa logs for register values

Enable Mesa debug logging to capture emitted register values:

```sh
# ML operation messages (tensor shapes, layer types)
ETNA_MESA_DEBUG=ml_msgs

# Rocket register debug
ROCKET_DEBUG=dbg_msgs

# Verbose Teflon delegate
TEFLON_DEBUG=verbose

# Combine all three:
ETNA_MESA_DEBUG=ml_msgs ROCKET_DEBUG=dbg_msgs TEFLON_DEBUG=verbose
```

The `ROCKET_DEBUG=dbg_msgs` output shows:
- Task structure (input/output channels, banks, offsets)
- Register write addresses and values

Pipe through `intercept.c`'s register readback for the full register dump.

---

## Register field reference

From `ref/linux/drivers/accel/rocket/rocket_registers.h`:

| Register | Field | Shift | Mask | Notes |
|---|---|---|---|---|
| `CNA_CBUF_CON0` | `DATA_BANK` | 0 | 0xf | data CBUF banks |
| | `WEIGHT_BANK` | 4 | 0xf | weight CBUF banks |
| | `WEIGHT_REUSE` | 13 | 0x1 | reuse weights across slices |
| `CNA_CONV_CON1` | `CONV_MODE` | 0 | 0xf | 0=normal, 3=depthwise |
| | `IN_PRECISION` | 4 | 0x7 | 2=int8, 3=int16, 5=fp16 |
| | `PROC_PRECISION` | 7 | 0x7 | same encoding |
| | `NONALIGN_DMA` | 30 | 0x1 | pixel/nonalign DMA mode |
| | `ARGB_IN` | 12 | 0xf | pixel channel count |
| `CNA_CONV_CON2` | `FEATURE_GRAINS` | 4 | 0x7ff | CBUF grain count |
| `CNA_DATA_SIZE1` | `DATAIN_CHANNEL` | 0 | 0xffff | aligned channel count |
| | `DATAIN_CHANNEL_REAL` | 16 | 0x3fff | real channel count - 1 |
| `CNA_WEIGHT_SIZE2` | `WEIGHT_KERNELS` | 0 | 0x3fff | output channels |
| | `WEIGHT_HEIGHT` | 16 | 0x1f | kernel height |
| | `WEIGHT_WIDTH` | 24 | 0x1f | kernel width |
| `CORE_DATAOUT_SIZE_1` | `DATAOUT_CHANNEL` | 0 | 0x1fff | aligned out channels - 1 |
| `DPU_DATA_CUBE_CHANNEL` | `CHANNEL` | 0 | 0x1fff | aligned out channels - 1 |
| | `ORIG_CHANNEL` | 16 | 0x1fff | real out channels - 1 |
| `DPU_DST_SURF_STRIDE` | `DST_SURF_STRIDE` | 4 | 0x0fffffff | output surface stride (<<4 in value) |
| `DPU_SURFACE_ADD` | `SURF_ADD` | 4 | 0x0fffffff | surface add offset (<<4 in value) |
| `DPU_DATA_FORMAT` | `IN_PRECISION` | 0 | 0x7 | 2=fp16 |
| | `PROC_PRECISION` | 26 | 0x7 | 2=fp16 |
| | `OUT_PRECISION` | 29 | 0x7 | 2=fp16, 5=fp32 |

---

## Known Mesa-conv.py differences

| Aspect | Mesa (ref/mesa) | conv.py |
|---|---|---|
| Data type | uint8 quantized (runtime) | FP16 standalone |
| Weight packing | 32-OC × 32-IC groups (WEIGHT_ATOMIC_SIZE=32) | 16-OC × 32-IC groups for wide pointwise |
| Output channel align | minimum 32, depthwise 64 | minimum 16 |
| Input channel align | minimum 16 (FEATURE_ATOMIC_SIZE) | 8 for small cases (in_c=2,3,4) |
| CBUF bank formula | `ceil(entries/256) + 1` | `ceil(bytes/CBUF_BANK_SIZE)` clipped to [1,11] |
| Feature grains | `50 + stride_y + 1` | `_feature_grains()` from row bytes |
| Height tiling | `rkt_split_tasks()` CBUF-aware slices | `_conv_tiles()` with fixed row limits |
| Input packing | NC1HW (quantized) or NHWC (pixel) | NC1HWC2 or NHWC |
| Stride/padding | native register support | no (valid-only conv) |
| NPU reset | not per-operation | not per-operation |
| Double-submit | never | for stale-state workarounds |

---

## Files involved

| File | Location |
|---|---|
| Mesa Rocket driver | `ref/mesa/src/gallium/drivers/rocket/` |
| Mesa Teflon delegate | `ref/mesa/src/gallium/targets/teflon/` |
| Upstream register defs | `ref/linux/drivers/accel/rocket/rocket_registers.h` |
| conv.py register program | `examples/kernel_6_18/conv.py` |
| Mesa-Teflon shape matrix | `mesa_example_run.md` |
| Mesa build/runtime | `~/mesa` (local checkout, not ref/) |
