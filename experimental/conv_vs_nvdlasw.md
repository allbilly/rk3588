# RK3588 Convolution: NVDLA SW vs `conv.py`

This note compares the NVIDIA NVDLA software stack in `ref/nvdla/sw/` with `examples/kernel_6_18/conv.py`.

Short answer: NVDLA SW is cleaner as a software architecture, but it is not a drop-in register template for RK3588 Rocket convolution. NVDLA SW compiles graph operations into loadable descriptors, then KMD firmware translates those descriptors into CDMA/CSC/CMAC/CACC/SDP register writes. `conv.py` skips that whole descriptor and firmware boundary and emits Rocket CNA/CORE/DPU/RDMA/PC register commands directly from Python.

The useful comparison is therefore not "copy these NVDLA registers into `conv.py`". The useful comparison is "copy the descriptor-driven organization so shape semantics, memory surfaces, dependencies, and register programming have separate owners".

## Evidence

The local NVDLA SW tree shows a full compiler/runtime/KMD path:

| Evidence | Meaning |
|---|---|
| `CompilerFeatures.md` | NVDLA SW advertises convolution FP16/INT8 support, FP16 Winograd, FP16 group convolution, and dilation. |
| `umd/core/src/compiler/engine-ast/ConvolutionOp.cpp` | Compiler emits `dla_conv_op_desc` and `dla_conv_surface_desc` fields for one convolution operation. |
| `umd/core/src/compiler/engine-ast/ConvCoreNode.cpp` | Compiler owns weight preprocessing, grouped convolution normalization, image-input channel extension, tensor surface sizing, strides, and offsets. |
| `umd/core/src/compiler/engine-ast/EngineNode.cpp` | Compiler emits dependency descriptors and data cube descriptors. |
| `umd/core/src/compiler/engine-ast/EngineGraph.cpp` | Compiler builds memory, address, tensor, relocation, and dependency-graph lists for a loadable. |
| `umd/core/src/runtime/Runtime.cpp` | Runtime deserializes the loadable, allocates/binds memory, fills task address lists, and submits tasks. |
| `umd/port/linux/nvdla.c` | UMD converts runtime tasks into `DRM_IOCTL_NVDLA_SUBMIT`. |
| `kmd/firmware/conv.c` | Firmware programs CDMA, CSC, CMAC, and CACC registers from the conv descriptors. |
| `kmd/firmware/include/dla_interface.h` | Shared ABI for network descriptors, common op descriptors, data cubes, conv op descriptors, and conv surface descriptors. |

`conv.py` has a different evidence shape:

| Evidence | Meaning |
|---|---|
| `_conv_params()` | Reconstructs RK/Rocket packing, alignment, width stride, output stride, and output atomics from one shape. |
| `pack_conv_weights_for_shape()` | Selects an FP16 weight serializer from shape predicates. |
| `_pack_conv_input_fp16()` | Packs NCHW FP16 input into the inferred RK hardware order. |
| `_conv_tiles()` | Encodes shape-specific tiling rules. |
| `make_conv2d_regs()` | Emits Rocket regcmd qwords directly. |
| `write_regs_to_npu_task()` | Writes command streams and a minimal PC chain tail. |
| `run_conv2d()` | Chooses direct submit, row tiling, output-channel tiling, depthwise channel tiling, or DPU add accumulation. |
| `compute_expected_nchw()` | Uses a simple FP64 NCHW valid-convolution reference. |

## Source Map

Relevant NVDLA SW files:

| File | Role |
|---|---|
| `CompilerFeatures.md` | High-level feature matrix and limitations. |
| `umd/core/src/compiler/engine-ast/ConvolutionOp.cpp` | Captures canonical conv params and emits `dla_conv_op_desc` / `dla_conv_surface_desc`. |
| `umd/core/src/compiler/engine-ast/ConvCoreNode.cpp` | Handles SDP joint ops, surface dims/strides/sizes/offsets, INT8 quantization, image convolution weight transforms, grouped convolution squashing, and conv+SDP merging. |
| `umd/core/src/compiler/engine-ast/EngineNode.cpp` | Emits dependency descriptors and data cube accessors. |
| `umd/core/src/compiler/engine-ast/EngineGraph.cpp` | Assigns annotation IDs, verifies dependencies, resolves memory, creates address list entries, and emits relocations. |
| `umd/core/src/runtime/Runtime.cpp` | Loads flatbuffer loadables, allocates memory, fills address lists, and submits tasks. |
| `umd/port/linux/nvdla.c` | Linux UMD memory allocation and submit ioctl wrapper. |
| `kmd/firmware/include/dla_interface.h` | Firmware ABI structs: `dla_network_desc`, `dla_common_op_desc`, `dla_data_cube`, `dla_conv_surface_desc`, `dla_conv_op_desc`. |
| `kmd/firmware/conv.c` | Converts `dla_conv_op_desc` and `dla_conv_surface_desc` into hardware register writes. |

Relevant Python sections:

| `conv.py` area | Role |
|---|---|
| `reg` | Handwritten Rocket target IDs and register addresses. |
| `drm_rocket_*` structs | Mainline Rocket ioctl ABI. |
| `_conv_params()` | One local shape-to-hardware-parameter function. |
| `_pack_*()` functions | FP16 input and weight packing experiments. |
| `_feature_grains()`, `_data_bank()`, `_tile_data_bank()` | CBUF/bank heuristics. |
| `_conv_tiles()` | Shape-specific tiling scheduler. |
| `make_conv2d_regs()` | Direct register stream emitter. |
| `submit_conv_tasks()` | Writes regcmds and submits Rocket tasks. |
| `run_conv2d()` | End-to-end standalone FP16 experiment. |

## NVDLA SW Pipeline

NVDLA SW is a compiler, runtime, and firmware stack.

1. The compiler captures graph-level convolution semantics.

`ConvCoreConvolutionOpNode::captureCanonicalParams()` copies bias, weight dimensions, padding, padding value, stride, dilation, raw weights, and group count from the canonical graph node.

That is the first big difference from `conv.py`. NVDLA SW records padding, stride, dilation, groups, weight layout intent, and precision as operation semantics before register programming. `conv.py` derives most of its hardware behavior later from shape predicates.

2. The compiler emits a conv operation descriptor.

`ConvCoreConvolutionOpNode::emitOp()` fills `dla_conv_op_desc` fields such as:

| Descriptor field | Meaning |
|---|---|
| `conv_mode` | Direct or Winograd. |
| `data_reuse`, `weight_reuse` | CBUF reuse behavior. |
| `skip_data_rls`, `skip_weight_rls` | Release behavior between dependent ops. |
| `entry_per_slice`, `data_bank`, `weight_bank`, `release` | CBUF scheduling parameters. |
| `conv_stride_x`, `conv_stride_y` | Convolution stride. |
| `pad_x_left`, `pad_x_right`, `pad_y_top`, `pad_y_bottom`, `pad_val` | Padding semantics. |
| `dilation_x`, `dilation_y` | Dilation semantics. |
| `input_width_csc`, `input_height_csc`, `input_channel_csc` | CSC input dimensions. |
| `kernel_width_csc`, `kernel_height_csc`, `kernel_channel_csc` | CSC kernel dimensions. |
| `input_width_cmac`, `input_height_cmac` | CMAC output dimensions. |
| `bytes_per_kernel` | Weight layout size information. |
| `in_cvt`, `out_cvt` | Input and output conversion/truncation parameters. |

This is not a register stream. It is a stable firmware ABI. The firmware later decides how these fields map to actual hardware registers.

3. The compiler emits surface descriptors.

The same `emitOp()` fills `dla_conv_surface_desc` through `setDataCubeAccessor()`:

| Data cube | Meaning |
|---|---|
| `src_data` | Input tensor memory type, address-list index, offset, size, dimensions, line stride, surface stride. |
| `weight_data` | Weight memory object and shape. |
| `wmb_data`, `wgs_data` | Optional compressed-weight metadata. |
| `dst_data` | Output tensor memory object and shape. |
| `offset_u`, `in_line_uv_stride` | Image/YUV input support. |

NVDLA's `dla_data_cube.address` is not a physical address. It is an index into the task's address list. The runtime/KMD resolves it later. `conv.py` writes physical DMA addresses into regcmd qwords immediately.

4. The compiler owns surface layout.

`ConvCoreNode.cpp` has separate functions for surface dimensions, line stride, surface stride, total surface size, and buffer offsets. It also handles split input offsets from upstream split nodes.

This is cleaner than `conv.py` because the memory layout contract lives with tensor surface descriptors. In `conv.py`, stride and offsets are recomputed locally in `_dma_strides()`, `make_conv2d_regs()`, `submit_pointwise_to_output()`, and several branches inside `run_conv2d()`.

5. The compiler preprocesses weights before emission.

NVDLA SW has explicit passes for:

| Pass | Purpose |
|---|---|
| `quantizeAuxData()` | Quantize weights for INT8 profiles, with per-kernel or per-filter scale handling. |
| `mandatoryChnlExtForIMG()` | Pre-channel extension for image convolution weights. |
| `optionalChnlExtForIMG()` | Optional post-channel extension for better MAC utilization. |
| `squashWeightGroups()` | Convert grouped convolution into a normalized weight representation. |
| `preProcessAuxData()` | Orchestrates image, grouped, and quantized weight preprocessing. |

`conv.py` also preprocesses weights, but it does it by selecting one of several FP16 packers and pre-reorders from local predicates. The distinction is important: NVDLA SW transforms model weights into the compiler's surface format, while `conv.py` transforms random FP16 test weights into an inferred RK register experiment format.

6. The compiler emits dependency graph descriptors.

`Node::emitDependencyParams()` fills `dla_common_op_desc` with:

| Field | Meaning |
|---|---|
| `index` | Operation annotation ID. |
| `op_type` | Firmware engine type, such as `DLA_OP_CONV` or `DLA_OP_SDP`. |
| `dependency_count` | Number of dependencies that must signal before this op. |
| `consumers[]` | Per-engine consumer index and event. |
| `fused_parent` | Fused producer relationship. |

This is how NVDLA SW represents engine ordering. `conv.py` mostly represents ordering by Python list order plus PC chain tail registers. It does not have an operation dependency graph separate from the command stream.

7. The compiler creates loadable memory and address lists.

`EngineGraph.cpp` assigns memory IDs and address IDs for pools, buffers, surfaces, input/output bindings, and content blobs. It also adds relocation entries for bindable surfaces so runtime-side memory can rewrite line and surface strides if needed.

This boundary is absent from `conv.py`. The Python file allocates a fixed set of BOs at import time:

| BO | Role |
|---|---|
| `task_map` | Task descriptors. |
| `regcmd_map` | Register command stream. |
| `input_map` | Input tensor and temporary staging. |
| `weight_map` | Weights and sometimes temporary staging. |
| `output_map` | Output tensor. |

Those fixed BOs are good for experiments, but they are the reason `conv.py` contains buffer-size and staging rules that would normally belong to a runtime allocator.

8. Runtime deserializes and submits loadable tasks.

`Runtime::load()` deserializes a loadable, collects task, submit, memory, address, tensor-desc, and relocation entries, allocates memory, and prepares address lists. `Runtime::submitInternal()` reloads dependency graph memory and calls `NvDlaSubmit()` for each DLA task.

The Linux UMD wrapper in `umd/port/linux/nvdla.c` converts the runtime task address list into `DRM_IOCTL_NVDLA_SUBMIT` arguments. The KMD receives handles and offsets, not a Python-built regcmd BO.

9. KMD firmware programs registers from descriptors.

`kmd/firmware/conv.c::processor_conv_program()` is the NVDLA register-programming equivalent of `conv.py::make_conv2d_regs()`, but it starts from descriptors instead of shape arguments.

It programs the major NVDLA blocks:

| Block | Register-programming source |
|---|---|
| CACC | Output size, output address, strides, dataout map, clip config. |
| CMAC A/B | Convolution mode and processing precision. |
| CSC | Input ext size, weight size, atomics, release, stride, dilation, padding, bank allocation. |
| CDMA | Input/weight memory addresses, input format, dimensions, strides, fetch grain, weight bytes, CVT, stride, padding, bank allocation. |

NVDLA's firmware explicitly waits for idle groups, validates address/stride alignment, resolves address-list entries into DMA addresses, and writes register groups in a fixed order. `conv.py` writes Rocket regcmd qwords and relies on the Rocket kernel's command processor path to fetch those qwords.

## `conv.py` Pipeline

`conv.py` is not a compiler. It is a direct RK3588 convolution register experiment.

The Python pipeline is:

1. Generate deterministic FP16 input and weights.
2. Compute valid-convolution output dimensions as `in_h - kh + 1` and `in_w - kw + 1`.
3. Derive RK alignment, input packing, width stride, output stride, CBUF entries, feature grains, data banks, and output atomics from `_conv_params()` plus local helper functions.
4. Pack NCHW FP16 input into an inferred Rocket layout.
5. Pack weights using one of several shape-specific FP16 serializers.
6. Emit Rocket CNA/CORE/DPU/RDMA/PC regcmd qwords.
7. For harder shapes, split by rows, output channels, depthwise channels, or input-channel chunks.
8. Submit one or more Rocket tasks.
9. Unpack raw output and compare with a NumPy FP64 valid-convolution reference.

That directness is useful. A single wrong bit or packing hypothesis is easy to bisect. The cost is that operation semantics, memory layout, scheduling, and register emission are interleaved.

Examples of shape-specific logic in `conv.py`:

| Python rule | NVDLA-style owner that is missing |
|---|---|
| `_needs_c96_oc24_pointwise_schedule()` | Compiler scheduler or task splitter. |
| `_pointwise_split_add_chunks()` | Graph-level split plus fused/explicit SDP accumulation semantics. |
| `_is_kh_major()` | Weight layout policy owned by weight translation. |
| `_reorder_grouped_spatial_weights_block16()` | Grouped-conv weight preprocessing pass. |
| `_feature_grains()` | CBUF scheduler. |
| `_data_bank()` and `_tile_data_bank()` | CBUF bank allocator. |
| `_conv_tiles()` | Split-task planner. |
| `_direct_submit_repeat()` | Hardware-state workaround outside a scheduler/firmware boundary. |

## Register Model Difference

Do not read NVDLA register names as RK3588 register names.

| Concept | NVDLA SW | `conv.py` / Rocket |
|---|---|---|
| Data DMA and conv sequencing | CDMA + CSC. | Mostly CNA registers. |
| MAC array | CMAC A/B. | CORE target plus CNA scheduling. |
| Accumulator | CACC. | CORE/DPU output path. |
| Post-processing | SDP. | DPU and DPU RDMA targets. |
| Program ordering | Firmware scheduler and dependency graph. | PC chain tail plus Rocket task list. |
| Register emission unit | KMD firmware writes MMIO registers from descriptors. | Python emits 64-bit regcmd qwords. |
| Memory references | Address-list index plus offset in `dla_data_cube`. | Physical DMA address in regcmd value. |

There are conceptual similarities: input size, output size, weight bytes, stride, padding, banks, and precision exist in both worlds. But the bitfields and sequencing are not interchangeable.

For example, NVDLA firmware writes `CDMA_D_BANK` and `CSC_D_BANK` from `conv_op->data_bank` and `conv_op->weight_bank`. `conv.py` writes Rocket `CNA_CBUF_CON0` with data and weight bank fields. Those solve the same scheduling problem, but they are not the same register ABI.

## Semantic Difference

The biggest difference is not C versus Python. It is the semantic contract.

| Area | NVDLA SW | `conv.py` |
|---|---|---|
| Input API | Compiled network graph, usually Caffe-origin tensors. | Standalone function arguments and random tensors. |
| Tensor layout | Tensor surface descriptors with format, dims, strides, memory IDs, offsets. | NCHW FP16 arrays manually packed per shape. |
| Precision | FP16 and INT8 profiles, with conversion/truncation descriptors. | FP16 input/weights, FP16 or FP32-ish output readback. |
| Padding | First-class `pad_x_left/right`, `pad_y_top/bottom`, `pad_val`. | Mostly valid convolution. Some padding-like workarounds are local. |
| Stride | First-class `conv_stride_x/y`. | Main path is stride 1. |
| Dilation | First-class `dilation_x/y` in descriptors and firmware registers. | Not modeled as a public API. |
| Winograd | Compiler mode with transformed dims and weights. | Not modeled. |
| Group conv | Compiler squashes/preprocesses group weights. | Expanded or reordered locally. |
| Fused post-op | Conv streams to SDP joint/fused nodes through dependency graph. | Separate DPU add helper for pointwise partial accumulation. |
| Memory | Loadable memory/address/tensor lists plus runtime binding. | Fixed BOs and manual offsets. |
| Scheduling | Dependency graph plus firmware processor groups. | Python list ordering, PC chain tail, repeat submits. |
| Register ownership | KMD firmware. | `make_conv2d_regs()`. |

## Why NVDLA SW Is Cleaner

NVDLA SW is cleaner because each hard problem has one owner.

| Hard problem | NVDLA SW owner | Python state |
|---|---|---|
| Operation semantics | `ConvCoreConvolutionOpNode` params and `dla_conv_op_desc`. | Function arguments plus local predicates. |
| Surface layout | `TensorSurfaceDesc` and `dla_data_cube`. | `_pack_conv_input_fp16()`, `_dma_strides()`, output unpackers, local offsets. |
| Weight transformation | `ConvCoreNode` preprocessing and `WeightTranslationUnit`. | Multiple FP16 packers selected by shape. |
| CBUF scheduling | Descriptor fields: `entry_per_slice`, `fetch_grain`, `data_bank`, `weight_bank`, `release`, reuse flags. | `_feature_grains()`, `_data_bank()`, `_tile_data_bank()`, `_conv_tiles()`. |
| Dependencies | `dla_common_op_desc` and graph annotation IDs. | Task list order and PC chain tail. |
| Memory binding | Loadable memory/address lists, runtime binding, UMD/KMD address resolution. | Fixed BOs and direct DMA addresses. |
| Register programming | `kmd/firmware/conv.c`. | `make_conv2d_regs()`. |

This structure is the main lesson for `conv.py`. NVDLA SW is not small, and it has its own limitations. But quirks are contained in compiler passes, descriptor emission, or firmware programming. In `conv.py`, a new quirk often becomes another condition inside the same end-to-end function.

## Where NVDLA SW Is Not Cleaner For This Project

NVDLA SW is not automatically better for RK3588 bring-up.

| Issue | Why it matters |
|---|---|
| Different register ABI | NVDLA CDMA/CSC/CMAC/CACC/SDP register writes do not directly match Rocket CNA/CORE/DPU/RDMA/PC regcmd qwords. |
| Different runtime model | NVDLA uses loadables and firmware descriptors; Rocket mainline accepts regcmd task addresses. |
| Different frontend | NVDLA SW is Caffe-oriented; `conv.py` is a standalone NumPy/Rocket harness. |
| Different output contract | NVDLA conv normally streams to SDP for post-processing; `conv.py` reads DPU output directly. |
| Different validation target | NVDLA validates network compiler behavior; `conv.py` validates individual FP16 register hypotheses. |

So the right use of `ref/nvdla/sw` is structural and semantic, not literal register copying.

## What To Copy From NVDLA SW

If `conv.py` is meant to grow into a cleaner broad convolution runner, copy the data model first.

1. Add an explicit convolution operation object.

It should hold at least input/output dims, kernel dims, groups/depthwise mode, padding, stride, dilation, precision, output conversion mode, and reuse flags. This is the Python equivalent of `dla_conv_op_desc`, not a replacement for Rocket register constants.

2. Add explicit surface/data-cube objects.

Track memory object, offset, size, width, height, channel, line stride, surface stride, and logical tensor layout. This is the Python equivalent of `dla_data_cube` / `dla_conv_surface_desc`.

3. Separate weight preprocessing from register emission.

Create a weight translation section with named transforms for normal, depthwise, grouped, pointwise-wide, and spatial layouts. Register emission should consume an already-prepared weight BO and size metadata.

4. Separate task splitting from register emission.

Create a split-task object that records tile input dims, output dims, source offset, destination offset, weight offset, banks, reuse flags, and PC-chain metadata before any qword is emitted.

5. Treat dependencies as data.

Even if Rocket submission still uses a PC chain, represent dependencies and fused/accumulate steps explicitly before writing PC tail registers. This would make DPU add accumulation less ad hoc.

6. Keep KMD-style register programming as the last step.

The Python equivalent of `processor_conv_program()` should consume operation, surface, and task descriptors and emit Rocket regcmd qwords. It should not rediscover shape semantics while emitting registers.

## What Not To Copy Directly

Do not directly copy these from NVDLA SW into `conv.py`:

| NVDLA item | Why not direct-copy |
|---|---|
| `CDMA_D_*`, `CSC_D_*`, `CMAC_*`, `CACC_*` register fields | Rocket uses different target IDs and register ABI. |
| Loadable flatbuffer format | Useful architecture, but too much machinery for a standalone examples file. |
| Caffe parser and canonical AST | Not needed for RK register experiments. |
| Firmware scheduler | Rocket mainline submit path already has its own task ABI. |
| NVDLA image/YUV channel extension rules | Only copy if matching Rocket image-input behavior is being implemented. |
| NVDLA compressed weights | `conv.py` currently uses uncompressed FP16 experiments. |

## Mapping For A Cleaner Python Design

One practical way to use NVDLA SW is to build a Rocket-native equivalent of the descriptor boundary.

| Python concept to add | Inspired by | Purpose |
|---|---|---|
| `ConvOp` dataclass | `dla_conv_op_desc` | Semantic convolution fields independent of registers. |
| `DataCube` dataclass | `dla_data_cube` | Memory object, offset, size, dims, and strides. |
| `ConvSurface` dataclass | `dla_conv_surface_desc` | Input, weight, and output surfaces for one op. |
| `TaskDesc` dataclass | `dla_common_op_desc` plus split-task info | Dependencies, tile geometry, reuse, and output placement. |
| `WeightTranslator` functions | `ConvCoreNode` weight preprocessing | Keep packing choices out of the register emitter. |
| `plan_conv_tasks()` | NVDLA scheduling/descriptor phase | Compute all tiles, offsets, and banks before register qwords. |
| `emit_rocket_conv_regs()` | `processor_conv_program()` | Final translation from descriptors to Rocket regcmds. |
| `submit_rocket_tasks()` | `Runtime::submitInternal()` | Handle BO address lists and Rocket task grouping. |

This structure can still live in one Python file if desired. The important rule is not file count; it is ownership. Register emission should be the last phase, not the place where shape policy is invented.

## NVDLA Descriptor Fields Worth Mirroring

These fields are especially useful because they map to problems currently spread across `conv.py`:

| NVDLA field | Python/Rocket equivalent to model |
|---|---|
| `conv_mode` | Direct/Winograd or Rocket-specific convolution schedule. |
| `data_reuse`, `weight_reuse` | Reuse flags in Rocket CBUF/PC-chain task planning. |
| `skip_data_rls`, `skip_weight_rls`, `release` | Explicit CBUF lifetime policy. |
| `entry_per_slice`, `fetch_grain` | CBUF entry and fetch-grain calculations. |
| `data_bank`, `weight_bank` | Bank allocator output. |
| `conv_stride_x/y` | Public stride API instead of hardcoded stride 1. |
| `pad_*`, `pad_val` | Public padding API instead of valid-conv-only assumptions. |
| `dilation_x/y` | Future dilation support boundary. |
| `input_*_csc`, `kernel_*_csc`, `input_*_cmac` | Separate source/kernel/output scheduler dimensions. |
| `bytes_per_kernel` | Weight packing contract. |
| `in_cvt`, `out_cvt` | Input/output conversion and truncation policy. |
| `src_data`, `weight_data`, `dst_data` | Address/stride/offset data cubes. |

## Why Known-Failed `conv.py` Shapes Are Different

The known-failed shape classes in `conv.py` are not proof that the RK3588 NPU lacks those capabilities. They mostly show that a direct FP16 register harness is missing compiler structure.

| `conv.py` failure class | NVDLA SW lesson |
|---|---|
| Spatial/depthwise large-channel shapes | Need a real task planner with CBUF lifetime, bank allocation, offsets, and dependency semantics. |
| Wide-input/narrow-output pointwise precision issues | Need an explicit accumulation/output-conversion contract, not post-hoc DPU add branches. |
| Buffer size or staging errors | Need runtime-sized memory/address objects instead of fixed BO assumptions. |
| Grouped spatial layout issues | Need a weight preprocessing pass with a clear source and destination layout. |

NVDLA SW would not directly pass these RK FP16 rows because it targets a different software and register ABI. But it shows how to avoid adding one workaround per shape.

## Single-File Rocket-Native Plan

If the next step is a cleaner single Python file, do not port NVDLA SW wholesale. Port the boundary pattern.

The file should be organized like this:

```text
conv_clean.py
  constants and Rocket register helpers
  Rocket ioctl structs and BO helpers
  ConvOp, DataCube, ConvSurface, TaskDesc dataclasses
  align/div helpers
  create_conv_op_from_args()
  create_surfaces_for_conv()
  pack_input_fp16_for_surface()
  translate_weights_fp16_for_op()
  plan_conv_tasks()
  allocate_or_slice_bo_regions()
  emit_rocket_conv_regs(task)
  compile_pc_chain(tasks)
  submit_tasks(tasks)
  unpack_output(surface)
  cpu_reference_for_same_semantics()
  shape_matrix_runner()
```

The first milestone should be parity with current `conv.py` for a few known-good FP16 direct-conv shapes. Only then add padding, stride, or new shape classes. Otherwise the rewrite will mix semantic changes with structural changes and become hard to debug.

## Validation Strategy

A descriptor-shaped rewrite should validate in layers:

1. For an existing passing shape, compare packed input bytes against current `conv.py`.
2. Compare packed weight bytes.
3. Compare computed data cubes: dimensions, offsets, line stride, surface stride, size.
4. Compare planned task count and every task's input/output/weight offsets.
5. Compare emitted regcmd qwords, allowing only intentional cleanup differences.
6. Run the shape on the NPU and compare output to current `conv.py` and NumPy.
7. Add one new semantic feature at a time: padding, then stride, then better split scheduling.

Do not validate a structural rewrite by immediately running the full shape matrix. Byte-for-byte parity on tiny shapes is the fastest way to find layout drift.

## Bottom Line

NVDLA SW is a strong reference for convolution software architecture: descriptors, surfaces, dependency graph, memory/address lists, runtime submission, and firmware register programming are separate phases.

`conv.py` is a strong reference for RK3588 register bring-up: it is direct, deterministic, FP16-focused, and exposes every qword written to the NPU.

For this project, use NVDLA SW to clean up the shape of the Python implementation, not as a literal register source. The best next improvement is to make `conv.py` descriptor-driven while keeping Rocket-specific register values, BO handling, and FP16 validation intact.
