# RK3588 Convolution: ONNC NVDLA vs `conv.py`

This note compares the ONNC NVDLA convolution backend in `ref/onnc/` with `examples/kernel_6_18/conv.py`.

Short answer: ONNC is cleaner as a compiler, but it is not a drop-in source of RK3588 Rocket register values. ONNC lowers ONNX `Conv` into an NVDLA loadable with typed operation descriptors, memory lists, address lists, dependency graphs, and task lists. `conv.py` writes RK3588 Rocket register commands directly from Python. The useful part to copy from ONNC is the operation-centered compiler structure: explicit Conv attributes, cube/layout objects, CBUF feasibility calculations, graph rewrites for channel/group splitting, weight/bias packing boundaries, and final task serialization. The unsafe part to copy blindly is the NVDLA descriptor/register contract, because RK3588's mainline Rocket command stream is not the same ABI.

## Evidence

This comparison is source-based. I did not find an ONNC-vs-`conv.py` shape matrix equivalent to `mesa_example_run.md` in this tree.

| Evidence | Meaning |
|---|---|
| `Operator.md` | ONNC advertises `Conv` support mapped to `CONV` plus `SDP`, with limits on constant weights/bias, padding, stride, dilation, and group. |
| `NvDlaBackend.cpp` pass pipeline | ONNC lowers ONNX to compute IR, legalizes attributes, splits large convolutions by channel, splits grouped convolution, allocates memory, emits NVDLA operations, builds tasks, and serializes a loadable. |
| `CodeEmitVisitor/Conv.inc` | The concrete Conv path packs weights/biases, computes cube info, splits by height when CBUF cannot hold full data, emits a CONV operation plus fused SDP bias/add output stage, and records dependency edges. |
| `NvDlaMeta.*` | Tensor cube sizes, strides, entry-per-slice, bank counts, memory/address metadata, and operation lists are centralized. |
| `conv.py` | The RK3588 experiment manually packs FP16 input/weights, emits handwritten Rocket registers, tiles with shape-specific rules, submits NPU tasks, unpacks output, and checks NumPy references. |

The important constraint: ONNC targets NVIDIA NVDLA `nv_full` loadables. RK3588's NPU lineage is NVDLA-like, but the mainline Rocket driver consumes Rocket register command buffers, not ONNC `dla_network_desc` blobs. So ONNC is useful as a compiler design reference, not as a literal register table for `make_conv2d_regs()`.

## Source Map

Relevant ONNC files:

| File | Role |
|---|---|
| `Operator.md` | Documents supported ONNX operators, Conv execution-unit mapping, and high-level limitations. |
| `lib/Transforms/TensorSel/ConvLower.cpp` | Converts ONNX `Conv` nodes into ONNC `onnc::Conv` compute operators and copies attributes. |
| `include/onnc/IR/Compute/Conv.h` | Defines the Conv operation object: inputs `X/W/B`, output `Y`, `auto_pad`, `dilations`, `group`, `kernel_shape`, `pads`, and `strides`. |
| `lib/IR/Compute/Conv.cpp` | Sets default Conv attributes such as `auto_pad="NOTSET"` and `group=1`. |
| `lib/Transforms/Optimizations/SplitConvPass.cpp` | Splits large normal convolutions along input channels, emits multiple Conv nodes, and combines outputs with `Sum`. |
| `lib/Target/NvDla/SplitGroupConvPass.cpp` | Splits grouped convolutions into `Split -> Conv... -> Concat` with per-group sliced weights/biases. |
| `lib/Target/NvDla/NvDlaBackend.cpp` | Defines the NVDLA backend pipeline, CBUF allocation type selection, max-channel calculation, memory allocation, code emit, task submit, and file generation passes. |
| `lib/Target/NvDla/Config/NvFull.cpp` | Defines the `nv_full` hardware constants: FP16 precision, atom sizes, MAC atom sizes, CBUF geometry, and tensor data type. |
| `lib/Target/NvDla/NvDlaMeta.h` / `.cpp` | Owns memory/address/task/loadable metadata and `NvDlaCubeInfo` size, stride, EPS, and bank calculations. |
| `lib/Target/NvDla/CodeEmitVisitor.h` / `.cpp` | Packs weights/biases, issues DLA addresses, emits operation descriptors, and links dependencies. |
| `lib/Target/NvDla/CodeEmitVisitor/Conv.inc` | Implements Conv-specific emission: attributes, group handling, CBUF split, CONV descriptor, surface descriptor, SDP bias/output stage. |
| `lib/Target/NvDla/NvDlaTaskSubmitPass.cpp` | Converts emitted DLA/EMU operation lists into loadable task blobs: network desc, dependency graph, op list, surface list, LUT list, task list, and submit list. |
| `lib/Target/NvDla/NvDlaFileGenPass.cpp` | Serializes the final loadable lists. |

Relevant Python sections:

| `conv.py` area | Role |
|---|---|
| `_conv_params()` | Infers RK alignment, width stride, output stride, atom layout, and input packing flags from a shape. |
| `_pack_conv_input_fp16()` | Packs logical NCHW FP16 input into an inferred RK feature-memory order. |
| `pack_conv_weights_for_shape()` | Chooses among FP16 weight layouts using shape predicates. |
| `_conv_tiles()` | Chooses row/channel tiling with handwritten thresholds. |
| `make_conv2d_regs()` | Emits handwritten Rocket CNA, CORE, DPU, RDMA, and PC-related register commands. |
| `write_regs_to_npu_task()` | Places command streams in a BO and appends a minimal PC chaining tail. |
| `run_conv2d()` | Generates input/weights, chooses split/add/depthwise/output-channel schedules, submits tasks, unpacks output, and compares against NumPy. |
| `compute_expected_nchw()` | Computes valid-convolution FP64 reference output. |
| shape list in `__main__` | Regression shapes and known-problem model-like shapes for the Python harness. |

## ONNC Pipeline

ONNC's Conv path is a real compiler pipeline.

1. `ConvLower` preserves ONNX Conv semantics.

`lib/Transforms/TensorSel/ConvLower.cpp` checks the ONNX node shape, creates an `onnc::Conv`, fills default attributes, then copies `auto_pad`, `dilations`, `group`, `kernel_shape`, `pads`, and `strides` from the ONNX node.

The Conv IR object has one owner for those semantics:

| Attribute | ONNC owner |
|---|---|
| Input/output tensors | `Conv::kX`, `kW`, `kB`, `kY` in `Conv.h` |
| Padding | `Conv::m_Pads` |
| Stride | `Conv::m_Strides` |
| Dilation | `Conv::m_Dilations` |
| Groups | `Conv::m_Group` |
| Kernel shape | `Conv::m_KernelShape` |
| Bias presence | `Conv::hasBias()` |

This is already cleaner than `conv.py`, where padding is mostly absent, stride is hardcoded to 1 in the register emitter, and most behavior is inferred from local shape predicates.

2. The NVDLA backend runs legality and graph-rewrite passes.

`NvDlaBackend::addOnncIrOptimization()` enables legalization and optimization passes, then adds Conv-specific transforms:

| Pass | Conv effect |
|---|---|
| `LegalizePadPass` | Enforces Conv pad limits; the setting gives Conv a pad limit of 31. |
| `ReplaceGemmByConv` | Lets GEMM-like layers use the Conv path. |
| `SplitConvPass` | Splits large `group=1` Conv by input channels when CBUF feasibility requires it. |
| `SplitGroupConvPass` | Rewrites grouped Conv into split inputs, per-group Conv nodes, and a channel-axis Concat. |

The two splitting passes are important because they operate on graph semantics before code emission. `conv.py` performs comparable work much later inside `run_conv2d()` and `_conv_tiles()`, after packing and register assumptions are already mixed together.

3. `NvDlaCubeInfo` centralizes layout, stride, EPS, and bank accounting.

For feature cubes, ONNC computes:

| Field | Meaning |
|---|---|
| `stride_channel` | Element byte stride. |
| `stride_line` | Width stride aligned from `dim_w * FEATURE_ATOM_CUBE_SIZE`. |
| `stride_surface` | `stride_line * dim_h`. |
| `size` | Feature cube size across aligned channel surfaces. |
| `eps` | Entries per slice, matching NVDLA CBUF expectations. |
| `banks` | `ceil(eps * dim_h / CBUF_BANK_DEPTH)`. |

For weight cubes, ONNC computes total aligned blob size and delays CBUF bank selection until allocation type is known.

This is conceptually what `conv.py` needs more of: a single cube/layout object. Today the Python path spreads equivalent state across `_conv_params()`, `_cbuf_entries()`, `_feature_grains()`, `_data_bank()`, `_tile_data_bank()`, and per-branch output-offset logic.

4. `NvDlaBackend::getCbufAllocType()` chooses a CBUF strategy.

ONNC tries these cases in order:

| CBUF case | Meaning |
|---|---|
| `kFullDataFullWeight` | Full feature data and full weights fit. |
| `kFullDataPartialWeight` | Full feature data and partial weights fit. |
| `kSplitDataFullWeight` | Split feature data and full weights fit. |
| `kSplitDataPartialWeight` | Split feature data and partial weights fit. |
| `kFullDataMinimumWeight` | Full feature data and minimum weight buffering fit. |
| `kSplitDataMinimumWeight` | Split feature data and minimum weight buffering fit. |
| `kUnfeasible` | No modeled CBUF layout fits. |

`CodeEmitVisitor::tryAllocateDataAndWeightsIntoCBuf()` then sets weight-bank mode and returns whether weights should be reused.

This is cleaner than `_conv_tiles()` because the decision is capacity-driven. `conv.py` uses thresholds such as output height, output channels, `RK_MAX_CONV_FLAT_STRIDE`, special `c96_oc24`, depthwise height caps, and channel tiling flags. Those thresholds are useful reverse-engineering knowledge, but they are not a general scheduler.

5. `CodeEmitVisitor::packWeight()` owns coefficient layout.

ONNC converts FloatTensor weights to FP16 and writes a DLA weight blob. The core direct-weight layout iterates output-channel MAC groups, kernel height/width, output lanes, and input channel lanes, with front padding for group alignment. Blob size is aligned to `WEIGHT_ATOM_CUBE_SIZE`.

The equivalent Python code has several independent packers:

| Python packer | Why it exists |
|---|---|
| `_pack_default()` | Simple OC/KH/KW/IC layout. |
| `_pack_kh_major()` | Spatial normal-conv layout. |
| `_pack_pointwise_wide()` | Wide pointwise layout. |
| `_pack_dw_spatial_major()` | Depthwise spatial layout. |
| `_reorder_grouped_spatial_weights_block16()` | Pre-shuffle for grouped spatial weights. |

The ONNC lesson is not that its exact NVDLA weight layout should replace these RK layouts. The lesson is that weight packing should be one backend-owned contract selected from operation/task metadata, not a set of shape predicates exposed throughout the runner.

6. `CodeEmitVisitor/Conv.inc` emits a CONV operation plus an SDP stage.

For each group and each height split, ONNC creates an `NvDlaDlaOperation` with `DLA_OP_CONV`. It fills `dla_conv_op_desc` fields for direct convolution:

| Descriptor area | Examples from ONNC |
|---|---|
| CBUF behavior | `data_reuse`, `weight_reuse`, `skip_weight_rls`, `entry_per_slice`, `data_bank`, `weight_bank`. |
| Input/kernel geometry | `input_width_csc`, `input_height_csc`, `input_channel_csc`, `kernel_width_csc`, `kernel_height_csc`, `kernel_channel_csc`. |
| Output geometry | `input_width_cmac`, `input_height_cmac`. |
| Conv semantics | `conv_stride_x`, `conv_stride_y`, `pad_x_left`, `pad_x_right`, `pad_y_top`, `pad_y_bottom`, `dilation_x`, `dilation_y`. |
| Precision/conversion | `in_precision`, `out_precision`, `out_cvt.scale`, `out_cvt.enable`, `pra_truncate`. |

It then creates a fused `DLA_OP_SDP` operation for bias/output handling. If bias exists, SDP `x1_op` is enabled in per-kernel sum mode and points at the packed bias blob. The SDP destination is the final memory-backed output tensor.

`conv.py` also programs convolution and DPU/SDP-like output behavior together, but it does so as raw Rocket register writes in `make_conv2d_regs()`. The Python code has to know every target register address and bitfield directly; ONNC writes typed descriptors and lets the NVDLA runtime/firmware/KMD interpret them.

7. ONNC splits height inside Conv emission.

`Conv.inc` computes an affordable convolution height from CBUF capacity:

```text
affordable_conv_height = CBUF_BANK_DEPTH * (CBUF_BANK_NUM - winfo.banks) / fcube_group.eps
```

Then it walks split layers, computing:

| Field | Purpose |
|---|---|
| `input_h_idx` | Source input row for this split. |
| `output_h_idx` | Destination output row for this split. |
| `input_split_height` | Input rows consumed by this split. |
| `output_split_height` | Output rows produced by this split. |
| `split_pad_top` / `split_pad_bottom` | Padding only on first/last split. |
| `is_weight_reuse` | Whether later splits can reuse buffered weights. |

This is very close to the structural fix `conv.py` needs. The Python path already tiles by height, but often repacks temporary input tiles and emits standalone task streams instead of treating source/output offsets and reuse as first-class task fields.

8. ONNC serializes a loadable, not raw Rocket tasks.

`NvDlaTaskSubmitPass` groups consecutive DLA operations into task blobs:

| Blob/list | Content |
|---|---|
| `task-*-addr0` | `dla_network_desc` or `emu_network_desc`. |
| `task-*-dep_graph` | `dla_common_op_desc` dependency descriptors. |
| `task-*-op_list` | Operation containers such as `dla_conv_op_desc` and `dla_sdp_op_desc`. |
| `task-*-surf_list` | Surface containers such as `dla_conv_surface_desc` and `dla_sdp_surface_desc`. |
| `task-*-lut_list` | LUT parameters, when needed. |
| task/submit/event lists | Loadable scheduling and synchronization. |

`NvDlaFileGenPass` then serializes the memory, tensor, address, event, task, and submit lists into the loadable.

This is not the same output format as `write_regs_to_npu_task()`, which writes 64-bit Rocket register commands and task descriptors for the mainline Rocket ioctl path.

## `conv.py` Pipeline

`conv.py` is a direct RK3588 register harness rather than a compiler.

The Python pipeline is:

1. Generate deterministic FP16 NCHW input and weights.
2. Infer RK packing and alignment in `_conv_params()`.
3. Pack FP16 input into one of the observed RK input layouts.
4. Pack FP16 weights using shape-selected layouts.
5. Emit CNA, CORE, DPU, RDMA, and PC register commands in `make_conv2d_regs()`.
6. For hard shapes, split by rows, output channels, depthwise channels, or input-channel chunks.
7. Use DPU add to accumulate some pointwise partials.
8. Submit one or more Rocket tasks.
9. Unpack raw output and compare against `compute_expected_nchw()`.

This is good for bring-up because every register and packing hypothesis is visible. It is not clean as a general compiler because shape semantics, memory layout, CBUF scheduling, task splitting, register emission, submission, and result verification are interleaved.

Examples of shape-specific logic in `conv.py`:

| Python rule | Meaning |
|---|---|
| `_needs_c96_oc24_pointwise_schedule()` | Hard-coded special case for one MobileNetV2-style pointwise shape. |
| `_pointwise_split_add_chunks()` | Splits wide input-channel pointwise conv and accumulates partials with DPU add. |
| `_is_kh_major()` | Selects a spatial weight layout by shape. |
| `_reorder_grouped_spatial_weights_block16()` | Pre-shuffles grouped spatial weights so another packer serializes them in hardware order. |
| `_feature_grains()` | Adjusts CBUF fetch grains with depthwise/spatial/small-channel exceptions. |
| `_data_bank()` / `_tile_data_bank()` | Select CBUF banks from local heuristics. |
| `_conv_tiles()` | Encodes multiple height/channel/depthwise tiling thresholds. |
| `_direct_submit_repeat()` | Repeats some submissions to stabilize observed hardware behavior. |

## Side-By-Side Differences

| Area | ONNC NVDLA | `conv.py` |
|---|---|---|
| Target ABI | NVDLA loadable descriptors and blobs. | RK3588 Rocket register command BOs and ioctls. |
| Operation representation | `onnc::Conv` object with preserved ONNX attributes. | Function args plus inferred shape dictionaries. |
| Logical layout | ONNX-style NCHW tensors represented by `NvDlaDims` and cube metadata. | Logical NCHW FP16 arrays packed manually into RK layouts. |
| Precision | `nv_full` backend is configured for FP16 (`PRECISION_FP16`, element size 2). | FP16 input/weights; output can be FP16 or read/cast as FP32 for some shapes. |
| Bias | Bias is a first-class optional Conv input and packed into an SDP operand. | No general Conv bias path in the main `run_conv2d()` reference. |
| Padding | Conv pads are preserved, legalized, split-aware, and emitted into descriptors. | Main API is valid convolution; no general padding contract. |
| Stride | Conv strides are preserved and emitted into descriptors. | `make_conv2d_regs()` programs stride 1. |
| Dilation | Conv dilation is preserved and emitted into descriptors. | Not modeled in the main runner. |
| Groups | Backend rewrites grouped Conv or loops group descriptors depending on path. | Depthwise and grouped behavior is handled by shape-specific pack/unpack paths. |
| Large normal Conv | `SplitConvPass` splits by input channel and sums outputs. | `_pointwise_split_add_chunks()` handles selected pointwise cases. |
| CBUF model | `NvDlaCubeInfo` and `getCbufAllocType()` drive banks/reuse/splitting. | `_feature_grains()`, `_data_bank()`, `_tile_data_bank()`, and `_conv_tiles()` heuristics. |
| Height splitting | Split fields are computed before descriptor emission with input/output row offsets. | Often repacks temporary input tiles and computes output offsets locally. |
| Weight reuse | Descriptor fields `weight_reuse` and `skip_weight_rls` are derived from CBUF strategy. | A limited `weight_reuse` flag is passed into handwritten registers. |
| Chaining/dependencies | Dependency graph and task list in the loadable. | Minimal PC chain tail in the register command buffer. |
| Output stage | Conv is followed by fused SDP for bias/output conversion. | DPU registers handle writeback; separate DPU add helper handles partial accumulation. |
| Validation | Compiler emits loadable; runtime validation is outside this note. | Self-contained shape runner compares with NumPy. |

## What ONNC Explains About `conv.py` Failures

ONNC reinforces the same conclusion as the Mesa comparison: many hard shapes are hard because a convolution implementation needs a scheduler and task model, not just more register cases.

| `conv.py` pain point | ONNC-style interpretation |
|---|---|
| Spatial/depthwise large-channel known issues | Need split tasks with explicit source/destination offsets, split-specific padding, CBUF banks, and reuse semantics. |
| `ic >> oc` pointwise numerical issues | Need a defined accumulation and output-stage contract. ONNC uses FP16 NVDLA CONV plus SDP descriptors; `conv.py` sometimes uses multiple FP16 partials and DPU add. |
| Buffer size errors | Tensor memory sizes should come from cube metadata and graph allocation, not fixed global BO assumptions. |
| Grouped conv quirks | ONNC rewrites group Conv structurally into split/conv/concat; Python currently keeps group behavior in local pack/reorder/unpack code. |
| Padding/stride gaps | ONNC treats pads and strides as operation attributes from lowering through descriptor emission. |

The caveat is important: ONNC passing or supporting a Conv shape on NVDLA would not prove the same raw RK3588 register sequence is correct. It only proves that the shape is expressible in a cleaner compiler model.

## What To Copy From ONNC

If `conv.py` is going to grow beyond a bring-up script, the useful ONNC ideas are structural.

1. Add an explicit Conv operation object that records input/output shapes, padding, stride, dilation, groups, dtype, bias, and output policy.
2. Add a cube/layout object that computes line stride, surface stride, total size, entries per slice, and bank requirements in one place.
3. Replace shape-specific channel and group rewrites with graph-like transforms before packing/register emission.
4. Replace `_conv_tiles()` thresholds with a capacity-driven CBUF allocation function.
5. Represent each hardware task as a data object with input offsets, output offsets, split padding, data banks, weight banks, reuse flags, and output dimensions.
6. Keep coefficient packing behind backend-owned functions selected by operation/task metadata.
7. Treat bias and output conversion as an explicit output stage, even if the first RK implementation only supports bypass/no-bias.
8. Generate the final Rocket regcmd stream from task objects, not directly from top-level shape args.
9. Keep PC chaining and dependency handling as part of task compilation, not a helper appended after handwritten register emission.

## What Not To Copy Literally

Do not directly port these ONNC details into Rocket code without checking Rocket register definitions and hardware behavior:

| ONNC detail | Why not literal for RK3588 Rocket |
|---|---|
| `dla_conv_op_desc` / `dla_sdp_op_desc` | These are NVDLA loadable ABI structures, not Rocket regcmd qwords. |
| `CBUF_BANK_NUM = 16` in `NvFull.cpp` | `conv.py` currently models RK CBUF as 12 banks. |
| NVDLA `FEATURE_ATOM_CUBE_SIZE`, `WEIGHT_ATOM_CUBE_SIZE`, `MAC_ATOMIC_C/K` values | Useful reference points, but RK Rocket field meanings and bank geometry must be checked against Rocket docs/registers and hardware. |
| ONNC direct-weight blob offsets | RK weight layouts observed in `conv.py` differ by pointwise/spatial/depthwise cases. |
| NVDLA loadable task/dependency format | Mainline Rocket uses register command buffers and Rocket ioctls. |
| NVDLA SDP bias/add descriptors | RK DPU/RDMA registers need separate mapping. |

## Single-File ONNC-Inspired Python Plan

For an ONNC-inspired rewrite of `conv.py`, keep the target Rocket-specific but copy ONNC's boundaries.

Recommended single-file sections:

| Python section | ONNC source to study | Purpose |
|---|---|---|
| Operation dataclass | `Conv.h`, `ConvLower.cpp` | Preserve Conv semantics explicitly. |
| CubeInfo dataclass | `NvDlaMeta.h`, `NvDlaMeta.cpp` | Compute size, stride, EPS, and bank needs centrally. |
| Legalization helpers | `Operator.md`, `NvDlaBackend.cpp` | Reject unsupported pads/strides/dilations/groups early. |
| Group/channel split transforms | `SplitConvPass.cpp`, `SplitGroupConvPass.cpp` | Rewrite hard shapes before hardware task emission. |
| Weight/bias packers | `CodeEmitVisitor.cpp` | Put coefficient layout behind one backend boundary. |
| Task dataclass | `CodeEmitVisitor/Conv.inc` split loop | Store per-task dimensions, offsets, padding, banks, and reuse. |
| CBUF allocator | `NvDlaBackend::getCbufAllocType()` | Replace threshold-based tiling. |
| Register emitter | `make_conv2d_regs()` plus Rocket register references | Emit Rocket regcmds from task fields. |
| Task compiler | `NvDlaTaskSubmitPass.cpp` conceptually | Build one task list/chain from emitted task streams. |
| Validation runner | Existing `compute_expected_nchw()` path | Keep deterministic FP16 tests and shape matrix. |

Unlike the Mesa-equivalent plan, this should remain an RK/Rocket file. ONNC does not provide the right final register ABI, but it provides a good example of where each compiler responsibility should live.

## Bottom Line

ONNC is cleaner than `conv.py` because it is a compiler backend. It carries Conv semantics from ONNX lowering into graph rewrites, memory allocation, cube layout, CBUF allocation, operation descriptors, surface descriptors, dependency graphs, and loadable task serialization.

`conv.py` is still more useful for RK3588 register bring-up because it talks directly to the Rocket NPU and tests hardware immediately. Its weakness is that every new shape tends to add another branch to packing, tiling, register emission, or output unpacking.

The practical path is not to port ONNC descriptors. The practical path is to make the Python RK path ONNC-shaped internally: explicit operation, explicit cube/task objects, capacity-driven splits, isolated packing, and one final Rocket register emitter.
