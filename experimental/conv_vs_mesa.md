# RK3588 Conv: Mesa Rocket vs Python Examples

This note explains how Mesa's Rocket driver programs convolution, then compares it with `examples/conv.py` and `examples/conv_mesa.py`.

## Current Status

Both regression suites pass on the board after pinning submit to core 0:

```bash
python /home/orangepi/rk3588/test/test_conv.py --submit
python /home/orangepi/rk3588/test/test_conv_mesa.py --submit
```

The key runtime issue was not convolution math. The kernel logs showed NPU jobs timing out/aborting on rotating cores even though the ioctl wrapper returned success. Pinning `core_mask=1` makes the jobs execute consistently.

Run submit tests sequentially. Running `test_conv.py --submit` and `test_conv_mesa.py --submit` concurrently can corrupt results because both scripts use the same NPU device and reset/submit globally. Occasional isolated `WARN` lines on non-`1x1` decomposed shapes have also been observed immediately after stressful runs; rerunning the same suite serially clears them. Treat stable WARNs as bugs, but do not use parallel submit runs as evidence.

## Mesa Rocket Conv Pipeline

Relevant Mesa files:

- `mesa/src/gallium/drivers/rocket/rkt_ml.c`
- `mesa/src/gallium/drivers/rocket/rkt_task.c`
- `mesa/src/gallium/drivers/rocket/rkt_regcmd.c`
- `mesa/src/gallium/drivers/rocket/rkt_coefs.c`

Mesa lowers `PIPE_ML_OPERATION_TYPE_CONVOLUTION` into an internal `struct rkt_operation` in `lower_convolution()`.

It records:

- input/output tensor indices and NHWC dimensions
- kernel width/height
- stride and padding mode
- quantization zero points/scales
- whether the op is depthwise
- packed weights and biases resources

Depthwise detection is in `rkt_is_depthwise()`:

```c
return poperation->conv.depthwise && input_channels > 1 && output_channels > 1;
```

## Weight And Bias Packing

Mesa packs coefficients in `rkt_coefs.c`.

`rkt_fill_weights()` creates a weight BO with a tiled order over:

- output channel atomic groups
- input channel groups
- kernel x/y
- inner output/input lanes

For normal conv, Mesa aligns output channels to an even count and input channels to at least `FEATURE_ATOMIC_SIZE`.

For depthwise conv, Mesa sets `output_channels = 1` in the packed weight layout and doubles the input channel grouping:

```c
if (rkt_is_depthwise(poperation))
   output_channels = 1;

unsigned input_channel_groups = WEIGHT_ATOMIC_SIZE;
if (rkt_is_depthwise(poperation))
   input_channel_groups *= 2;
```

The weights are converted from unsigned quantized representation by subtracting `0x80`.

`rkt_fill_biases()` also applies bias correction for input/weight zero points and chooses `truncate_bits` for some quantized scales.

The Python examples are FP16-only. They do not implement Mesa's full int8 bias/requant path.

## Task Splitting

Mesa computes split tasks in `rkt_split_tasks()`.

The key CBUF calculations are:

- `calc_entries_per_slice()`
- `calc_input_banks()`
- `calc_weights_banks()`

Mesa uses a CBUF bank budget to decide whether a convolution fits as one task or must be split vertically.

If weights fit with room left:

```c
operation->reuse_weights_cbuf = true;
available_input_banks = CBUF_BANKS - weights_banks_required;
```

Otherwise Mesa uses a partial-weights/partial-input path:

```c
operation->reuse_weights_cbuf = false;
available_input_banks = 7;
available_weights_banks = CBUF_BANKS - available_input_banks;
```

If the full input fits, Mesa emits one `split_task`.

If not, Mesa emits multiple vertical slices. It tracks:

- `top_slice`
- `bottom_slice`
- overlap between adjacent tasks
- retained slices
- per-task `input_height`
- per-task `output_height`
- `input_offset`
- `output_offset`

The overlap logic is important for kernels taller than 1. Adjacent input windows overlap by `weights_height - 1` rows so output rows at slice boundaries have enough source rows.

## Register Emission

Mesa emits register command streams in `rkt_regcmd.c`.

The central function is `fill_first_regcmd()`.

Important register groups:

- CNA: input DMA, CBUF allocation, convolution size, weight location, conversion/padding
- CORE: convolution mode and output dimensions
- DPU: output formatting, bypass paths, optional quantization/addition
- DPU RDMA: used when fused addition is present
- PC: command chaining and operation enable

Mesa emits `CNA_CBUF_CON0` from per-task bank allocation:

```c
uint32_t con0 = CNA_CBUF_CON0_WEIGHT_BANK(task->weights_banks) |
                CNA_CBUF_CON0_DATA_BANK(task->input_banks);
if (task_num > 0 && operation->reuse_weights_cbuf)
   con0 |= CNA_CBUF_CON0_WEIGHT_REUSE(1);
```

Mesa sets convolution mode bits based on input and depthwise state:

```c
if (task->input_channels_real == 1)
   con1 |= CNA_CONV_CON1_NONALIGN_DMA(1) |
           CNA_CONV_CON1_GROUP_LINE_OFF(1) |
           CNA_CONV_CON1_ARGB_IN(8);

if (operation->depthwise)
   con1 |= CNA_CONV_CON1_CONV_MODE(3);
```

Mesa emits per-task data/weight sizes:

- `REG_CNA_DATA_SIZE0/1/2/3`
- `REG_CNA_WEIGHT_SIZE0/1/2`
- `REG_CORE_DATAOUT_SIZE_0/1`
- `REG_DPU_DATA_CUBE_WIDTH/HEIGHT/CHANNEL`

Mesa also emits `REG_CNA_PAD_CON1`, but that value is part of its quantized/int8 pipeline. In our FP16 Python path, adding this write caused intermittent numeric warnings, so `conv_mesa.py` computes it but does not currently emit it.

## PC Chaining

Mesa supports multiple split tasks through PC command chaining.

In `compile_operation()`, Mesa builds one regcmd buffer per split task, then patches the previous task's PC tail with:

- next regcmd physical address via `REG_PC_BASE_ADDRESS`
- next register count via `REG_PC_REGISTER_AMOUNTS`

At the end of each command stream, Mesa emits PC op enable:

```c
emit_raw(regs, 0x81, REG_PC_OPERATION_ENABLE,
         PC_OPERATION_ENABLE_RESERVED_0(14) |
         PC_OPERATION_ENABLE_OP_EN(1));
```

The Python examples currently submit one task at a time through the kernel task descriptor. They do not yet implement Mesa's multi-task PC chain.

## `examples/conv.py`

`conv.py` is the current passing FP16 reference path for these tests.

Key properties:

- one low-level NPU submit per direct `1x1` task
- channel slicing for `1x1` with more than 4 input channels
- depthwise channel slicing for more than 8 channels
- non-`1x1` convolution lowered to many exact `1x1` submits in `_run_conv2d_spatial_decomposed()`
- grouped convolution is expanded or decomposed in software where needed
- submit pinned to `core_mask=1`
- input/output BOs are non-cacheable IOMMU BOs
- weights BO remains cacheable and is synced to device

The important correctness fallback is `_run_conv2d_spatial_decomposed()`.

Instead of programming direct `3x3`, `5x5`, etc. convolution, it loops over every `(kh, kw)` kernel point, crops the matching input window, runs a pointwise `1x1` NPU submit, and accumulates in FP16 order:

```python
for kh_idx in range(kernel_h):
    for kw_idx in range(kernel_w):
        input_crop = input_nchw[:, ic:ic + 1,
                                kh_idx:kh_idx + out_h,
                                kw_idx:kw_idx + out_w].copy()
        weight_1x1 = weight_ochw[oc_start:oc_end,
                                 ic_local:ic_local + 1,
                                 kh_idx:kh_idx + 1,
                                 kw_idx:kw_idx + 1].copy()
        result[:, oc_start:oc_end] += _npu_submit_single_input_pointwise(...)
```

This is slower than direct hardware spatial convolution, but it is stable and matches the tests.

## `examples/conv_mesa.py`

`conv_mesa.py` started as an experiment to move `conv.py` toward Mesa's task model.

It adds Mesa-inspired helpers:

- `PC_TASK_COUNT_CLEAR`
- `PC_TASK_NUMBER_MASK`
- `_calc_pad_con1()`
- `_packed_line_stride_channels()`
- `_output_line_stride_bytes()`
- `_calc_weight_banks()`
- `_calc_slice_budget_in_rows()`
- `_run_conv2d_spatial_tiled()`

It also records Mesa-like values in `compute_conv2d_params()`:

- `pad_con1`
- `weights_banks`

By default, the current passing path still disables direct spatial tiling:

```python
needs_spatial_tiling = False
needs_spatial_decomposition = not is_1x1
```

That means normal `conv_mesa.py` runs keep the Mesa-inspired calculations and experimental tiling function in the file, but use the same stable decomposition route as `conv.py` for non-`1x1` tests.

There is now an opt-in direct-spatial path behind `CONV_MESA_DIRECT=1`. It is guarded by a shape allowlist and only bypasses decomposition for shapes that have passed board-side NumPy comparison. `CONV_MESA_DIRECT_ALL=1` exists only as an unsafe exploration override for testing candidate shapes.

This is deliberate. The direct spatial tiling path produced full but sometimes numerically mismatched output for some non-`1x1` shapes. The stable route is to keep it disabled until full PC chaining and direct spatial register parity are implemented.

## Current Diff: `conv.py` vs `conv_mesa.py`

Both files now share the same runtime reliability fixes:

- `core_mask=1` instead of `core_mask=0`
- input/output BOs use `RKNPU_MEM_NON_CACHEABLE | RKNPU_MEM_IOMMU`
- input/output mem syncs are conditional on `data_is_cacheable`
- output `FROM_DEVICE` sync exists for cacheable mode but is skipped for non-cacheable BOs

## Comparison Table

| Area | Mesa Rocket | `examples/conv.py` | Better Today |
|---|---|---|---|
| Goal | General driver/compiler backend for real ML graphs | Minimal FP16 reverse-engineering/test harness | Depends on task |
| Supported graph shape | Handles compiled subgraphs and can fuse some operations like add | Runs one conv-shaped test at a time | Mesa |
| Data type focus | Quantized uint8/int8-style path with scales, zero points, bias correction, requantization | FP16-only path | `conv.py` for FP16 bring-up, Mesa for quantized models |
| Correctness in current tests | Not directly used by our Python regression tests | Passes current `test_conv.py --submit` | `conv.py` |
| Simplicity | Complex: task splitting, PC chaining, quantization, bias BOs, RDMA, graph resources | Small and direct: pack buffers, emit regs, submit, unpack | `conv.py` |
| Debuggability | Harder: many layers and generated register macros | Easier: one Python file, readable register values, deterministic random inputs | `conv.py` |
| Direct spatial convolution | Designed to run real spatial kernels directly | Avoids direct spatial conv; decomposes non-`1x1` into many `1x1` submits | Mesa architecturally, `conv.py` practically today |
| Performance | Better design: fewer submits, direct kernels, CBUF reuse, PC chaining | Slower for non-`1x1` because it does software decomposition and many submits | Mesa |
| CBUF scheduling | Computes input/weight bank budget and splits tasks | Uses simple local rules; no real PC task chain | Mesa |
| Weight reuse | Supports `CNA_CBUF_CON0_WEIGHT_REUSE` across split tasks | Reloads weights per submit | Mesa |
| Multi-task execution | Builds chained regcmd buffers and patches PC next-address/count | One kernel task descriptor per submit | Mesa |
| Output stitching | Hardware writes each task directly to output offsets | Python may stitch or accumulate in software | Mesa |
| Group/depthwise handling | Has dedicated depthwise mode and quantized coefficient layout | Uses explicit software slicing/expansion/decomposition | Mesa for production, `conv.py` for transparency |
| Register coverage | Much broader: DPU RDMA, bias, EW/add, LUT-related zeroing, quant output conversion | Minimal subset needed for current FP16 conv tests | Mesa |
| Risk of hidden magic | High: many scale-specific constants and TODOs | Lower: fewer magic values, easier to bisect | `conv.py` |
| Hardware parity | Closer to a real driver stack | Closer to our known-good experimental shapes | Depends on target |
| Failure mode | Harder to isolate because many subsystems interact | Easier to isolate individual packing/register bugs | `conv.py` |
| Maintainability for production | Better long-term if completed/upstreamed | Not suitable as a production compiler backend | Mesa |
| Maintainability for reverse engineering | Too broad for quick experiments | Excellent as a controlled reference script | `conv.py` |

## Where Mesa Is Better

Mesa's approach is better as an architecture for a real driver.

It models convolution as a compiled operation, allocates graph resources, packs weights and biases, splits work based on CBUF limits, and emits a regcmd stream that can chain multiple hardware tasks. This is the correct direction for performance and production use.

The main advantages are:

- fewer kernel submissions for large or spatial convolutions
- direct hardware spatial convolution instead of software lowering
- CBUF-aware tiling
- weight reuse across split tasks
- direct output offsets rather than Python-side stitching
- support for quantization, bias, and fused add paths
- a path toward full model execution instead of isolated operators

## Where `conv.py` Is Better

`conv.py` is better as a bring-up and reverse-engineering tool.

It intentionally avoids parts of the hardware path that are not fully understood yet. For non-`1x1` kernels, it decomposes convolution into exact pointwise submits, which is slow but stable and easy to validate against NumPy.

The main advantages are:

- easier to read and modify
- deterministic random test inputs/weights
- smaller register surface area
- faster bug isolation
- stable FP16 correctness for current tests
- no dependency on Mesa's quantized coefficient assumptions
- useful as a known-good baseline when experimenting with direct spatial scheduling

In short: Mesa is the better target design; `conv.py` is the better current oracle.

The remaining file-level differences are Mesa-experiment additions in `conv_mesa.py`.

### Extra Constants

`conv_mesa.py` adds:

```python
PC_TASK_COUNT_CLEAR = 1 << 12
PC_TASK_NUMBER_MASK = 0xFFF
```

These are intended for future PC-chain register programming.

### Extra Mesa-Style Helpers

`conv_mesa.py` adds:

```python
_calc_pad_con1()
_packed_line_stride_channels()
_output_line_stride_bytes()
_calc_weight_banks()
_calc_slice_budget_in_rows()
```

These mirror pieces of Mesa's `rkt_task.c` and `rkt_regcmd.c` planning logic.

Only `weights_banks` and `pad_con1` are currently stored in params. `pad_con1` is not emitted in the current passing path.

### Extra Params

`conv_mesa.py` adds these fields to `compute_conv2d_params()`:

```python
'pad_con1': pad_con1
'weights_banks': ...
```

These are scaffolding for direct Mesa-like spatial task programming.

### Experimental Spatial Tiling

`conv_mesa.py` adds `_run_conv2d_spatial_tiled()`.

It computes a CBUF row budget, splits input rows into overlapped vertical slices, submits each slice independently, and stitches output rows in software.

This approximates Mesa's `rkt_split_tasks()`, but it is not equivalent because:

- it does not program a single chained PC command buffer
- it does not set Mesa's exact per-task `input_offset`/`output_offset` registers in one task chain
- it does not use weight reuse through `CNA_CBUF_CON0_WEIGHT_REUSE`
- it still relies on repeated standalone Python submits

Default routing keeps it disabled:

```python
needs_spatial_tiling = False
```

Opt-in routing can use direct spatial for allowlisted shapes with:

```bash
CONV_MESA_DIRECT=1 python /home/orangepi/rk3588/test/test_conv_mesa.py --submit
```

As of this note, direct spatial is verified for small `ic=1/3, oc=6` spatial kernels, grouped `3->6 g=3 3x3`, and `4->4 3x3 9x9`. Larger regular conv, several mismatched-channel kernels, depthwise spatial, and high-channel spatial still fall back to decomposition.

### PAD_CON1 Difference

Mesa emits `REG_CNA_PAD_CON1` in `rkt_regcmd.c`.

`conv_mesa.py` computes the value, but the current passing Python FP16 path does not emit it. This is because the Python path is not Mesa's full quantized pipeline, and emitting `PAD_CON1` caused intermittent non-`1x1` warnings.

## Why Direct Mesa Tiling Is Not Done Yet

Mesa's direct convolution path is a coordinated system:

- packed int8 weights/biases
- exact CBUF bank split
- per-slice input/output offsets
- overlap/retain slice accounting
- optional CBUF weight reuse
- PC command chaining
- DPU quantization and bypass configuration

`conv_mesa.py` currently has only part of that system. It can estimate slice sizes and perform software stitching, but it does not yet reproduce Mesa's exact multi-task command chain.

Until full direct-spatial parity is implemented, the robust default path is:

- use direct pointwise hardware for `1x1`
- decompose non-allowlisted non-`1x1` into exact `1x1` submits
- keep the Mesa tiling code as scaffolding, disabled by default

## Next Implementation Step

To make `conv_mesa.py` truly Mesa-like, implement PC-chained multi-task submission:

1. Build a list of split tasks matching `rkt_split_tasks()`.
2. Emit one regcmd segment per task.
3. Patch each segment tail with next `REG_PC_BASE_ADDRESS` and `REG_PC_REGISTER_AMOUNTS`.
4. Set task descriptor `task_number`/`subcore_task` for the actual chain.
5. Program `CNA_CBUF_CON0_WEIGHT_REUSE` for later slices when weights fit in CBUF.
6. Use per-task input/output DMA offsets instead of software stitching.
7. Re-enable direct spatial tiling only after register dumps match Mesa for small known shapes.
