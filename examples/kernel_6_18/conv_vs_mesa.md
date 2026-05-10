# RK3588 Convolution: Mesa Rocket vs `conv.py`

This note compares the Mesa Rocket convolution implementation in `ref/mesa/src/gallium/drivers/rocket/` with `examples/kernel_6_18/conv.py`.

Short answer: Mesa is cleaner because it treats convolution as a compiled tensor operation. `conv.py` treats convolution as a standalone register experiment. Mesa has a small number of general mechanisms that cover padding, stride, depthwise, quantization, CBUF tiling, output offsets, and PC chaining. `conv.py` has many shape-specific workarounds because it bypasses most of that compiler structure and tries to make individual FP16 shapes work directly from Python.

## Evidence

`mesa_example_run.md` contains two different kinds of evidence.

| Evidence | Meaning |
|---|---|
| Real Teflon model runs | `mobilenetv1`, `mobilenetv2`, `inception`, `ssdmobilenetv2`, and `mobiledet` compile and run through Mesa/Teflon for their supported subgraphs. |
| Extracted shape matrix | A union of `conv.py` regression shapes and extracted Mesa/Teflon convolution rows, replayed through different paths. |

The important matrix summary is:

| Target | PASS | FAIL | ERROR | SKIP | UNSUPPORTED | NOT_RUN | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mainline6_18/conv.py` | 359 | 46 | 1 | 47 | 0 | 0 | 453 |
| `mainline6_18/conv_mesa.py` | 65 | 49 | 320 | 10 | 5 | 4 | 453 |
| real Mesa custom one-layer TFLite | 399 | 5 | 4 | 0 | 45 | 0 | 453 |

Do not read this as "Mesa passes every possible `conv.py` shape". It means Mesa passes many real TFLite convolution shapes and, for many rows where the Python experiments fail or error, the real Mesa custom TFLite path passes. Some rows are unsupported by the local generated-model helper, not necessarily by the Rocket hardware. The real Teflon model runs are the stronger evidence for production-shape coverage.

## Source Map

Relevant Mesa files:

| File | Role |
|---|---|
| `rkt_ml.c` | Lowers `PIPE_ML_OPERATION_TYPE_CONVOLUTION`, creates tensors, compiles split tasks, submits Rocket jobs, packs/unpacks user tensors. |
| `rkt_ml.h` | Defines `struct rkt_operation` and `struct split_task`. |
| `rkt_coefs.c` | Packs quantized weights and biases into hardware layout. |
| `rkt_task.c` | Computes CBUF entries/banks and splits large convolutions into vertical tasks. |
| `rkt_regcmd.c` | Emits CNA, CORE, DPU, DPU RDMA, and PC register streams. |
| `rkt_registers.h` | Generated register field macros used by `rkt_regcmd.c`. |

Relevant Python sections:

| `conv.py` area | Role |
|---|---|
| `_conv_params()` | Reconstructs alignment, packing, width stride, output stride, and layout choices. |
| `pack_conv_weights_for_shape()` | Selects one of several FP16 weight layouts. |
| `_pack_conv_input_fp16()` | Packs NCHW FP16 input into RK hardware order. |
| `_conv_tiles()` | Shape-specific tiling rules for pointwise, spatial, and depthwise cases. |
| `make_conv2d_regs()` | Emits a handwritten register list. |
| `write_regs_to_npu_task()` | Writes command streams and a minimal PC chain tail. |
| `run_conv2d()` | Chooses direct submit, output-channel slicing, depthwise slicing, or DPU add accumulation. |
| `known_issue_shapes` | Documents Mesa-like shapes that the Python path does not model cleanly. |

## Mesa Pipeline

Mesa's convolution path is organized as a compiler pipeline.

1. `rkt_ml.c::lower_convolution()` records graph-level semantics.

It stores input/output tensor indices, NHWC dimensions, padding, stride, depthwise state, zero points, scales, weight shape, packed weights, and packed biases in `struct rkt_operation`.

This is the first major difference from `conv.py`: Mesa does not infer shape semantics later from ad-hoc register rules. The operation already knows whether it is padded, strided, depthwise, quantized, or fused with addition.

2. `rkt_coefs.c::rkt_fill_weights()` creates the hardware coefficient layout.

Mesa aligns input channels to at least `FEATURE_ATOMIC_SIZE`, aligns normal output channels to an even count, and switches depthwise weights to a single output-kernel layout:

```c
input_channels = MAX2(input_channels, FEATURE_ATOMIC_SIZE);
output_channels = align(output_channels, 2);
if (rkt_is_depthwise(poperation))
   output_channels = 1;
```

The packing loops walk output-channel atomic groups, input-channel groups, kernel width, kernel height, output lanes, and input lanes. For depthwise, `input_channel_groups` is doubled from `WEIGHT_ATOMIC_SIZE` to `64`.

Mesa also converts quantized unsigned tensor bytes into the signed domain expected by this hardware path by subtracting `0x80`.

3. `rkt_coefs.c::rkt_fill_biases()` corrects bias for input and weight zero points.

The bias path subtracts the contribution of quantized zero points and chooses `truncate_bits` for known scale cases. This is part of why Mesa can run real TFLite models: the convolution result is not just MAC output; it is quantized model arithmetic.

`conv.py` is FP16-only. It compares against NumPy FP64/FP16 reference values, but it does not implement Mesa's int8 zero-point, bias correction, clip truncate, or output requantization contract.

4. `rkt_task.c::rkt_split_tasks()` schedules the work against CBUF capacity.

Mesa computes:

| Function | Purpose |
|---|---|
| `calc_entries_per_slice()` | How many CBUF entries one input slice needs. |
| `calc_input_banks()` | How many banks the full input needs. |
| `calc_weights_banks()` | How many banks the weights need. |

If weights fit with room left, Mesa keeps full weights in CBUF and enables weight reuse for later tasks:

```c
if (weights_banks_required + 1 < CBUF_BANKS)
   operation->reuse_weights_cbuf = true;
```

If the input does not fit, Mesa splits vertically. It tracks `top_slice`, `bottom_slice`, overlap slices, retained slices, per-task padding, per-task input height, output height, input offset, and output offset. The overlap logic is important for `3x3`, `5x5`, and `7x7` kernels because adjacent tiles need shared source rows.

5. `rkt_regcmd.c::fill_first_regcmd()` emits per-task registers from `struct split_task`.

Mesa programs the major blocks as one coherent register stream:

| Block | Mesa responsibilities |
|---|---|
| CNA | CBUF banks, convolution mode, stride, input size, output atomics, weight size, padding, input DMA address, input DMA strides, weight address, CVT config. |
| CORE | QD/depthwise mode, output size, clip truncate. |
| DPU | Output format, output base address, output stride, cube dimensions, bias/scale path, output conversion. |
| DPU RDMA | Bias reads and optional fused add reads. |
| PC | Next regcmd address, next register count, version sentinel, operation enable. |

Examples of shape-semantic registers that are first-class in Mesa:

```c
EMIT(REG_CNA_PAD_CON0, CNA_PAD_CON0_PAD_LEFT(task->pad_left) |
                         CNA_PAD_CON0_PAD_TOP(task->pad_top));
EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_X_STRIDE(task->stride_x) |
                         CNA_CONV_CON3_CONV_Y_STRIDE(task->stride_y));
EMIT(REG_CNA_FEATURE_DATA_ADDR,
     rkt_get_tensor(subgraph, operation->input_index)->phys_addr +
        task->input_offset);
EMIT(REG_DPU_DST_BASE_ADDR,
     rkt_get_tensor(subgraph, operation->output_index)->phys_addr +
        task->output_offset);
```

Those four fields explain many `conv.py` differences. Mesa expresses padding, stride, input slice location, and output slice location directly in the hardware command stream. `conv.py` mostly avoids padding/stride, and when it tiles it often packs separate temporary inputs and stitches or accumulates results on the host side around NPU submits.

6. `rkt_ml.c::compile_operation()` patches PC chaining.

Mesa builds one regcmd buffer per split task, then patches the previous stream's tail with the next command address and register count. The patch points are the PC base address and PC register amount entries near the end of the regcmd stream.

This is different from submitting many unrelated standalone Python tasks. The hardware receives a planned sequence, and later tasks can reuse weights in CBUF when `reuse_weights_cbuf` is true.

7. `rkt_ml.c::rkt_ml_subgraph_invoke()` submits graph jobs.

When weights are reused, Mesa submits all tasks for an operation as one Rocket job so they stay on the same core. When weights are not reused, it may spread tasks as separate jobs for parallelism.

This matters because CBUF reuse is only valid if tasks execute on the expected core and in the expected sequence.

## `conv.py` Pipeline

`conv.py` is useful, but it is not a compiler. It is a reverse-engineering harness for a single FP16 operator family.

The Python pipeline is:

1. Generate deterministic FP16 input and weights.
2. Convert NCHW tensors into an inferred RK packing.
3. Pick a weight layout from several shape predicates.
4. Emit one handwritten register list in `make_conv2d_regs()`.
5. For harder shapes, split by rows, output channels, depthwise channels, or input-channel chunks.
6. Submit one or more NPU tasks.
7. Unpack raw output and compare against a NumPy reference.

That is excellent for hardware bring-up because every moving part is visible. It is not clean as a production convolution implementation because many choices are encoded as special cases.

Examples of shape-specific logic in `conv.py`:

| Python rule | Why it exists |
|---|---|
| `_needs_c96_oc24_pointwise_schedule()` | One explicit MobileNetV2-style pointwise schedule. |
| `_pointwise_split_add_chunks()` | Splits large `IC >> OC` pointwise shapes and accumulates partials with DPU add. |
| `_is_kh_major()` | Switches spatial normal-conv weight layout. |
| `_reorder_grouped_spatial_weights_block16()` | Pre-shuffles grouped spatial weights so the default serializer happens to match hardware order. |
| `_feature_grains()` | Uses RK-specific clamps and depthwise/small-channel exceptions. |
| `_data_bank()` and `_tile_data_bank()` | Pick CBUF banks with local heuristics rather than one general scheduler. |
| `_conv_tiles()` | Encodes many separate tiling thresholds for output stride, height, channel tiling, and depthwise. |
| `_direct_submit_repeat()` | Repeats some submits to stabilize observed hardware behavior. |

The handwritten register path also has important omissions compared with Mesa:

| Area | Mesa | `conv.py` |
|---|---|---|
| Tensor layout contract | NHWC Teflon tensors converted into Rocket tensor BOs. | NCHW FP16 arrays packed manually per shape. |
| Quantization | Input/weight/output zero points, scales, bias correction, truncate, requantization. | FP16 only, with optional FP16 writeback; no TFLite quantized arithmetic. |
| Padding | Programs `REG_CNA_PAD_CON0` and `REG_CNA_PAD_CON1`. | Mostly valid convolution; no general padding contract in `run_conv2d()`. |
| Stride | Programs `REG_CNA_CONV_CON3` from the operation. | Main API assumes stride 1. |
| Split scheduling | General CBUF bank budget and vertical slicing with overlap. | Shape-specific tiling thresholds and temporary input slices. |
| Input/output offsets | Per-task hardware DMA offsets. | Often repacks each tile at BO offset 0 or computes local offsets manually. |
| Weight reuse | `CNA_CBUF_CON0_WEIGHT_REUSE` when tasks stay on one core. | Limited boolean use, but not the same full Mesa scheduling model. |
| PC chaining | Patches next regcmd address/count for split tasks. | Has a minimal tail helper, but most logic is still standalone task construction. |
| Fused add | Integrated DPU RDMA/EW path for graph add. | Separate FP16 DPU add helper for partial accumulation. |
| Register definitions | Generated macros and field names. | Handwritten numeric register addresses and bit comments. |

## Why The Known-Failed Shape List Exists

The `known_issue_shapes` list in `conv.py` is not evidence that the RK3588 NPU cannot run those shapes. It is evidence that this Python FP16 register harness does not yet have Mesa's full compiler model.

The listed reasons fall into three broad categories.

| Category in `conv.py` | What it really means | Why Mesa can pass many of them |
|---|---|---|
| `mesa_semantics (spatial/depthwise with large channels)` | Direct spatial/depthwise scheduling is not fully modeled by Python for that shape class. | Mesa has depthwise-specific coefficient packing, CBUF splitting, overlap, padding, output offsets, and depthwise DPU/CORE mode as one path. |
| `ic>>oc pointwise numerical precision` | Python's FP16 path, chunking, output format, or accumulation order diverges from the expected reference for wide input / narrow output pointwise layers. | Mesa is running quantized TFLite arithmetic with bias correction and requantization, not the same FP16 accumulation contract. |
| `buffer size error` | The Python fixed BO sizes, output read count, or staging strategy cannot represent the generated shape. | Mesa allocates pipe resources sized from tensor dimensions and operation output size. |

Several rows in `mesa_example_run.md` show exactly this pattern: `conv.py` or `conv_mesa.py` fails/errors while the `real Mesa custom one-layer TFLite` column passes. Examples include large depthwise MobileNet shapes, large `1x1` pointwise shapes, and large RGB pointwise sweep shapes.

The most important point: Mesa passes those rows under Mesa/TFLite semantics. `conv.py` tests FP16 valid-convolution semantics. If a shape differs by quantization, padding, stride, output layout, or expected rounding, a direct pass/fail comparison is not apples-to-apples.

## Why Mesa Is Cleaner

Mesa is cleaner because each hard problem has one owner.

| Hard problem | Mesa owner | Python state |
|---|---|---|
| Operation semantics | `struct rkt_operation` in `rkt_ml.h` populated by `lower_convolution()`. | Spread across function arguments, local predicates, and known-shape lists. |
| Hardware task shape | `struct split_task` populated by `rkt_split_tasks()`. | Recomputed in several branches with local `tile_p` dictionaries. |
| CBUF capacity | `calc_entries_per_slice()`, `calc_input_banks()`, `calc_weights_banks()`. | Multiple heuristics in `_feature_grains()`, `_data_bank()`, `_tile_data_bank()`, `_conv_tiles()`. |
| Coefficient layout | `rkt_fill_weights()` and `rkt_fill_biases()`. | Several packers plus pre-reorders selected by shape predicates. |
| Register fields | Generated macros in `rkt_registers.h`. | Handwritten bit shifts and comments. |
| Multi-task execution | `compile_operation()` plus PC patching plus Rocket jobs. | Mixture of task lists, repeated submits, manual output offsets, and post-submit unpack/stitch. |
| Graph integration | Tensors are persistent pipe resources, operations are compiled once, invoke submits jobs. | One standalone op call creates random data, submits, unpacks, and returns. |

Mesa is not tiny and it still has reverse-engineering scars. `rkt_regcmd.c` contains magic scale cases, `FEATURE_GRAINS(50 + stride + 1)`, and special input-stride cases. But those quirks are isolated inside a compiler structure. In `conv.py`, similar quirks become public control flow that multiplies as more shapes are added.

That is the real cleanliness difference. Mesa has a messy hardware target behind a clean operation/task/register boundary. `conv.py` exposes the messy target directly to every shape.

## Why `conv.py` Still Matters

`conv.py` is still useful for this project.

It is better than Mesa for quick register bring-up because it is small, deterministic, FP16-focused, and easy to bisect. When a single register or packing rule is wrong, the Python file makes that visible quickly. Mesa's graph/compiler stack is harder to reduce to one isolated experiment.

The right interpretation is:

| Use case | Better tool |
|---|---|
| Understand one register bit or one packing hypothesis | `conv.py` |
| Run real TFLite convolution graphs | Mesa Rocket |
| Test FP16 standalone NPU behavior | `conv.py` |
| Implement broad convolution coverage | Mesa-style operation/task/regcmd pipeline |
| Debug one failed shape quickly | `conv.py`, then compare with Mesa dumps |

## What To Copy From Mesa

If the Python path is meant to become broad and clean, it should not keep adding per-shape exceptions. The next useful changes are Mesa-shaped structural changes.

1. Add an explicit operation object that records padding, stride, dtype, zero points, scales, and tensor layout.
2. Add a `split_task` equivalent and compute all per-task dimensions, offsets, padding, banks, and overlap before emitting registers.
3. Replace `_conv_tiles()` heuristics with CBUF calculations equivalent to `rkt_task.c`.
4. Emit `REG_CNA_PAD_CON0`, `REG_CNA_PAD_CON1`, and stride registers from operation semantics, not from local shape assumptions.
5. Make direct spatial/depthwise use Mesa's per-task input/output offsets instead of Python-side temporary tiling where possible.
6. Treat PC chaining as the normal multi-task path, not a partial helper around repeated standalone submits.
7. Keep FP16 arithmetic separate from Mesa's quantized path so comparisons do not mix FP16 reference expectations with TFLite uint8/int8 expectations.

## Single-File Mesa-Equivalent Python Plan

To write one Python file that passes the same convolution shapes as Mesa, do not start from the current `conv.py` control flow. Start from Mesa's data model and port it almost mechanically. The goal is not "make FP16 `conv.py` pass more shapes"; the goal is "make a Python Mesa clone for the same supported quantized TFLite-style convolution semantics".

The file should be organized in the same order as Mesa:

| Python section | Mesa source to port | Purpose |
|---|---|---|
| Register constants/macros | `rkt_registers.h`, `rkt_regcmd.c` | Emit the same bitfields and packed 64-bit register commands. |
| Operation dataclass | `struct rkt_operation` in `rkt_ml.h` | Hold tensor shapes, padding, stride, zero points, scales, depthwise flag, BOs, and split tasks. |
| SplitTask dataclass | `struct split_task` in `rkt_ml.h` | Hold per-task dimensions, banks, offsets, padding, and regcmd address/count. |
| Input tensor conversion | `rkt_ml.c::rkt_ml_subgraph_invoke()` | Convert NHWC uint8 input into Rocket tensor layout and signed domain. |
| Weight packing | `rkt_coefs.c::rkt_fill_weights()` | Produce exactly Mesa's normal/depthwise weight BO layout. |
| Bias packing | `rkt_coefs.c::rkt_fill_biases()` | Apply zero-point correction and truncate-bit choice. |
| Task splitting | `rkt_task.c::rkt_split_tasks()` | Compute CBUF bank allocation, vertical slices, overlaps, and hardware offsets. |
| Regcmd emission | `rkt_regcmd.c::fill_first_regcmd()` | Emit the exact CNA/CORE/DPU/RDMA/PC stream for each split task. |
| PC patching | `rkt_ml.c::compile_operation()` | Patch next regcmd address and fetch amount for chained tasks. |
| Submit | `rkt_ml.c::rkt_ml_subgraph_invoke()` | Submit Rocket jobs with one task list when weight reuse is required. |
| Output conversion | `rkt_ml.c::rkt_ml_subgraph_read_outputs()` | Convert raw Rocket output BO back to NHWC uint8. |

The most important design rule is: the Python file should use Mesa's NHWC uint8/int8 quantized contract, not the current FP16 NCHW contract. If the file keeps FP16 data, FP16 NumPy references, or `conv.py` output unpacking, it will not be "exact same as Mesa" even if some registers match.

### 1. Define Mesa-Like Data Structures

Use explicit dataclasses rather than dictionaries and shape predicates:

```python
@dataclass
class Operation:
    depthwise: bool
    reuse_weights_cbuf: bool = False
    truncate_bits: int = 0
    padding_top: int = 0
    padding_bottom: int = 0
    padding_left: int = 0
    padding_right: int = 0
    stride: int = 1
    input_width: int = 0
    input_height: int = 0
    input_channels: int = 0
    input_zero_point: int = 0
    input_scale: float = 1.0
    output_width: int = 0
    output_height: int = 0
    output_channels: int = 0
    output_zero_point: int = 0
    output_scale: float = 1.0
    weights_width: int = 0
    weights_height: int = 0
    weights_zero_point: int = 0
    weights_scale: float = 1.0
    add_tensor: int = -1
    addition_input: bool = False
    addition_offset: int = 0
    addition_scale: float = 1.0
    tasks: list[SplitTask] = field(default_factory=list)
```

`SplitTask` should mirror every field in Mesa's `struct split_task`, including `top_slice`, `bottom_slice`, `num_overlap_slices`, `num_retain_slices`, `input_offset`, `output_offset`, `input_banks`, `weights_banks`, `atomic_count`, and `surfaces_per_row`.

Do not replace this with `_conv_params()` plus local branch variables. That is exactly how the current Python path became shape-specific.

### 2. Match Mesa Tensor Semantics

The single-file runner should accept tensors in the same logical format as Teflon/Mesa:

| Tensor | Logical order | dtype |
|---|---|---|
| Input | `NHWC` | `uint8` or `int8` as model requires, with zero point and scale. |
| Weights normal conv | `OHWI` as Mesa reads it: `OC x KW x KH x IC` according to the current Mesa indexing. | `uint8` quantized. |
| Weights depthwise | Mesa's depthwise weight tensor shape where output-kernel count is `1` and channels live in the input-channel axis. | `uint8` quantized. |
| Bias | `OC` int32. | `int32`. |
| Output | `NHWC` | `uint8` after DPU output conversion. |

The current `conv.py` API uses `NCHW` FP16 with valid-conv dimensions derived as `in_h - kh + 1`. That cannot represent Mesa's padded/strided rows in `mesa_example_run.md` without changing the API.

### 3. Port Input Packing Exactly

Port `rkt_ml.c::rkt_ml_subgraph_invoke()` input conversion instead of using `_pack_conv_input_fp16()`.

Mesa uses two paths:

| Condition | Packing behavior |
|---|---|
| `output_channels == 1 && input_channels == 1 && !addition_input && add_tensor == -1` | Direct buffer copy. |
| `input_channels == 1` | Iterate `x`, then `y` up to `max(input_height, FEATURE_ATOMIC_SIZE)`, pad with input zero point. |
| Otherwise | Iterate channel atom group `u`, then `x`, then `y`, then `c`, store `input - 0x80` for real channels and `zero_point - 0x80` for padding channels. |

This order is a common source of false failures. Mesa's loops use `input_width`, `input_height`, and `FEATURE_ATOMIC_SIZE` exactly; the Python clone should preserve the same loop order even if a NumPy transpose looks more natural.

### 4. Port Weight Packing Exactly

Port `rkt_coefs.c::rkt_fill_weights()` as a direct loop first. Optimize later only after byte-for-byte dumps match.

Key details to preserve:

```c
input_channels = MAX2(input_channels, FEATURE_ATOMIC_SIZE);
output_channels = align(output_channels, 2);
if (rkt_is_depthwise(poperation))
   output_channels = 1;

weights_size = weights_width * weights_height * output_channels *
               align(input_channels, WEIGHT_ATOMIC_SIZE) * 2;

input_channel_groups = WEIGHT_ATOMIC_SIZE;
if (depthwise)
   input_channel_groups *= 2;
```

The loop order must be:

1. `oc1`
2. `ic1`
3. `x`
4. `y`
5. `oc2`
6. `ic2`

The emitted value must be `weights_in[...] - 0x80` for real channels and `zero_point - 0x80` for padded channels, with Mesa's special skip/zero behavior for aligned output-channel tails.

Do not reuse `_pack_default()`, `_pack_kh_major()`, `_pack_pointwise_wide()`, or `_pack_dw_spatial_major()` for the Mesa-equivalent file. Those are FP16 reverse-engineering layouts, not Mesa's quantized coefficient layout.

### 5. Port Bias Correction Exactly

Implement `calculate_bias_correction()` and `rkt_fill_biases()` exactly enough to reproduce Mesa output.

For normal conv, correction is the sum over all `x`, `y`, and `ic`:

```c
(weights[oc][x][y][ic] - weight_zero_point) * (input_zero_point - 0x80)
```

For depthwise, correction uses `weights[0][x][y][oc]`.

Then Mesa writes:

```c
biases[oc] = (biases_in[oc] - correction) / (1 << truncate_bits);
```

The scale-specific `truncate_bits` table in Mesa looks ugly, but a Python file that claims Mesa equivalence must preserve it until a better derivation is proven.

### 6. Port CBUF Task Splitting Exactly

Port these functions directly from `rkt_task.c`:

```python
calc_entries_per_slice(operation)
calc_input_banks(operation)
calc_weights_banks(operation)
calc_line_stride(width)
fill_task(operation, task)
split_tasks(operation)
```

Important differences from current `conv.py`:

| Mesa behavior | Why it matters |
|---|---|
| `calc_weights_banks()` adds one extra bank. | Mesa relies on this conservative bank count. Removing it changes task split and CBUF assignment. |
| Full weights + partial input sets `reuse_weights_cbuf = true`. | Later tasks can set `CNA_CBUF_CON0_WEIGHT_REUSE`. |
| Partial weights + partial input reserves 7 input banks. | This changes both task count and register values. |
| Vertical split computes overlap and retained slices. | Spatial kernels need shared rows at tile boundaries. |
| `input_offset` and `output_offset` are byte offsets derived from line stride. | Hardware writes each task directly into final output placement. |

Do not port `_conv_tiles()`. Its thresholds are Python-specific and should disappear from the Mesa-equivalent file.

### 7. Emit Registers From Mesa Fields

The register emitter should be a direct Python translation of `rkt_regcmd.c::fill_first_regcmd()`.

Required rules:

| Rule | Reason |
|---|---|
| Use generated-mask semantics from `rkt_registers.h`, translated into Python helper functions. | Avoid handwritten bit drift. |
| Emit duplicate `REG_CNA_CBUF_CON0` and duplicate `REG_CNA_CONV_CON1` exactly where Mesa does. | Matching dumps should come before cleanup. |
| Emit all zeroing registers that Mesa emits. | They clear stale state between operations. |
| Emit `REG_CNA_PAD_CON0` and `REG_CNA_PAD_CON1`. | Padding semantics and quantized pad values live there. |
| Emit DPU RDMA bias registers even if no fused add is present. | Mesa's bias path uses DPU RDMA. |
| Emit `REG_DPU_OUT_CVT_*` from quantized scale math. | This is required for TFLite output equality. |
| Emit the final PC base/count/version/operation-enable tail. | The kernel fetches exactly the regcmd stream. |

The 64-bit command packing should remain:

```python
def emit_raw(target, reg_addr, value):
    return (target << 48) | ((value & 0xffffffff) << 16) | reg_addr
```

For field macros, prefer small Python functions named like Mesa macros:

```python
def CNA_CBUF_CON0_WEIGHT_BANK(x): return (x & MASK) << SHIFT
def CNA_CBUF_CON0_DATA_BANK(x): return (x & MASK) << SHIFT
```

The masks and shifts should come from `rkt_registers.h`, not from guesses in the old Python script.

### 8. Patch PC Chaining Like Mesa

Mesa first emits each task's regcmd with placeholder PC next-address/count fields. Then `compile_operation()` patches the previous task's last four entries.

The Python clone should do the same:

1. Build `regcmds: list[list[int]]`, one list per split task.
2. Allocate one contiguous regcmd BO sized to `sum(align(len(task) * 8, 64))`.
3. For each task except the last, patch entry `reg_count - 4` with next physical address and entry `reg_count - 3` with the next fetch count.
4. Copy each task stream into the aligned BO offset.
5. Store `task.regcfg_addr` and `task.regcfg_amount` from the actual BO physical address and qword count.

The fetch-count formula should initially match Mesa exactly:

```c
regs_to_fetch = util_dynarray_num_elements(&regcfgs[i + 1], uint64_t);
regs_to_fetch -= 4;
regs_to_fetch = align(regs_to_fetch / 2, 2);
```

Do not "simplify" this until a hardware dump proves the simplification is equivalent.

### 9. Submit Like Mesa

The Python submit path should use the mainline Rocket ioctl format, but job construction should mirror Mesa:

| Condition | Submit shape |
|---|---|
| `operation.reuse_weights_cbuf == true` | One Rocket job with all split tasks, so all tasks stay on the same core and can reuse CBUF weights. |
| `operation.reuse_weights_cbuf == false` | One job per split task is allowed, matching Mesa's parallelism path. |

Each `drm_rocket_task` should point at the task's patched regcmd address and regcmd count. The input BO list must include the input tensor BO and optional add tensor BO. The output BO list must include the output tensor BO.

This is not the same as repeatedly calling a one-task helper. Weight reuse and PC chaining depend on task grouping.

### 10. Unpack Output Like Mesa

Port `rkt_ml.c::rkt_ml_subgraph_read_outputs()`:

```c
for (int oc = 0; oc < operation->output_channels; oc++) {
   for (int x = 0; x < operation->output_width; x++) {
      for (int y = 0; y < operation->output_height; y++) {
         unsigned c = oc % FEATURE_ATOMIC_SIZE;
         unsigned g = oc / FEATURE_ATOMIC_SIZE;
         output_out[y][x][oc] = output_in[g][y][x][c] + 0x80;
      }
   }
}
```

This is Mesa's raw output layout and quantized-domain conversion. Do not reuse `_unpack_flat_1x1_output()` or `_unpack_nc1hwc2_output()` from the FP16 script.

### 11. Validation Strategy

The quickest way to make this converge is byte-for-byte comparison against Mesa dumps before checking numerical output.

Recommended validation order:

1. Enable Mesa BO dumps for a tiny known passing generated model.
2. Run the Python clone with the same input, weights, bias, zero points, scales, stride, and padding.
3. Compare input BO bytes.
4. Compare weight BO bytes.
5. Compare bias BO bytes.
6. Compare split task count and every `SplitTask` field.
7. Compare each regcmd qword before PC address patching.
8. Compare patched regcmd qwords, allowing only physical address differences.
9. Compare raw output BO bytes.
10. Compare final NHWC uint8 output.

Only after those pass for small normal conv, small depthwise conv, stride-2, SAME padding, large pointwise, and large depthwise should the script be run across the full shape matrix.

### 12. Expected Coverage And Limits

A faithful Python Mesa clone should pass the same class of shapes as Mesa's Rocket driver, not every row that appears in `conv.py`.

Expected pass set:

| Shape class | Should pass if cloned correctly |
|---|---|
| Quantized normal conv with stride 1 or 2 and supported padding | Yes. |
| Quantized depthwise conv using Mesa's depthwise definition | Yes. |
| Large pointwise layers from MobileNet/SSD/MobileDet | Yes, under quantized semantics. |
| CBUF-split spatial/depthwise layers | Yes, if split tasks, offsets, overlap, and PC patching match. |
| Fused add cases that Mesa supports | Yes, if DPU RDMA/EW path and scale cases are copied. |

Expected non-goals:

| Shape class | Reason |
|---|---|
| FP16 `conv.py` exact-reference rows | Different dtype, packing, arithmetic, output conversion, and tolerances. |
| Batch > 1 standalone rows | Mesa's Teflon path mostly treats model tensors with batch 1 in the tested examples. Add only after matching Mesa behavior. |
| Per-axis quantized models | Mesa explicitly rejects per-axis quantization today. |
| Dilation | Mesa rejects dilation factors other than 1. |
| Arbitrary grouped convolution | Mesa's path covers normal conv and its depthwise definition, not every PyTorch-style grouped conv in `conv.py`. |

### 13. Minimal File Skeleton

The single file should look like this before optimizations:

```text
mesa_conv_single.py
  constants and register macro helpers
  Rocket ioctl structs and BO helpers
  Operation and SplitTask dataclasses
  align/div helpers
  is_depthwise()
  create_operation_from_tflite_like_args()
  pack_input_like_mesa()
  fill_weights_like_mesa()
  fill_biases_like_mesa()
  calc_entries_per_slice()
  calc_input_banks()
  calc_weights_banks()
  fill_task()
  split_tasks_like_mesa()
  emit_regcmd_like_mesa()
  compile_operation_like_mesa()
  submit_operation_like_mesa()
  read_output_like_mesa()
  cpu_quantized_reference_or_tflite_reference()
  shape_matrix_runner()
```

Keep the first implementation boring and literal. A readable direct port of Mesa is better than a clever vectorized Python version because the first milestone is byte-for-byte parity with Mesa, not speed.

## Bottom Line

Mesa passes many shapes that `conv.py` marks as known issues because Mesa solves the whole convolution compilation problem. It owns tensor semantics, quantized arithmetic, coefficient packing, CBUF scheduling, split-task offsets, register emission, and job submission together.

`conv.py` is valuable precisely because it is not that. It is a transparent FP16 reverse-engineering script. Its failed/known-issue shape list is the cost of using a shape-driven script where Mesa uses an operation-driven compiler pipeline.
