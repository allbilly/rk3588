# RK3588 TRM register campaign

Silicon: Orange Pi RK3588, mainline Rocket ABI.  Backend target:
`~/tinygrad` branch `rockchip-2608-ew-restart`, commit `e3e93fa43`.

This file distinguishes register descriptions from results actually observed on
the NPU. DPU probes use the established one-task DPU/RDMA submission shape
(`enable_mask=0x18`); PPU and CNA probes use their established decoded Rocket
recipes. New cases are followed by `examples/kernel_6_18/simple_add.py`.
The one exception is explicit below: a correct two-task CNA weight-reuse chain
polluted the first following health job before a second health job recovered.

The exhaustive field-by-field triage, including fields that are not useful
backend operations, is in `trm_register_matrix.md`.

## DPU register and auxiliary-DMA results

| Fields | Silicon result | Operand format | Direct backend use |
|---|---|---|---|
| `EW_OP_SRC=0`, `EW_OP_VALUE_0..7`, `ERDMA_DISABLE=1` | **PASS**: FP16 ADD and INT16 ADD without an EW buffer | FP16 ALU operand is FP32 bits; INT16 ALU operand is signed low 16 bits | Stop materializing broadcast constants in scratch and omit their ERDMA read |
| BS configured ALU | **PASS**: scalar FP16 ADD | `BS_ALU_CFG` is FP32 bits | Fuse a scalar ALU before EW |
| BN configured ALU | **PASS**: scalar FP16 ADD | `BN_ALU_CFG` is FP32 bits | Fuse a second scalar ALU before EW |
| BS/BN configured MUL | **PASS** | FP16 bits in `BS_MUL_CFG[31:16]` / `BN_MUL_CFG[31:16]` | Fuse affine scaling |
| EW configured MUL | **PASS** | FP16 bits in `EW_OP_VALUE_n[15:0]`, unlike EW configured ALU | Eliminate scalar MUL buffer and ERDMA read |
| BS + BN + EW configured ALUs | **PASS**: three additions in one task | All three configured ALU constants are FP32 | Collapse three scalar elementwise kernels |
| BS and BN ALU+MUL plus EW | **PASS**: five scalar ops in one task | BS computes `(x + alu) * mul`; BN computes `x * mul + alu` | Lower two affine stages plus a final EW op as one task |
| BS/BN `RELUX_EN` | **PASS** in both stages | FP32 clamp limit; exact `[0, limit]` | Fuse ReLU/ReLU-X before EW |
| BS/BN `MUL_PRELU` | **PASS** in both stages | FP16 slope in MUL operand | Fuse leaky ReLU before EW |
| `RDMA_BRDMA_CFG` → BS ALU | **PASS**: per-channel tensor ADD | BRDMA ALU tensor is FP32 | Add a third tensor without another DPU task |
| `RDMA_NRDMA_CFG` → BN ALU | **PASS**: per-channel tensor ADD | NRDMA ALU tensor is FP32 | Add a fourth tensor without another DPU task |
| BRDMA/NRDMA external MUL lanes | **PASS** for both clients | One FP16 multiplier is consumed per element in MUL-only mode | Fuse tensor MUL in BS/BN for the proved layout |
| main + BRDMA + NRDMA + ERDMA | **PASS**: exact four-input sum in one task | main/ERDMA FP16, BRDMA/NRDMA ALU FP32 | Four-input reductions and chained binary fusion |
| EW/ERDMA data mode 0 | **PASS**: one FP16 value per channel repeats across pixels and channel surfaces, including immediately after a four-input auxiliary-DMA task | 16 operands produced a 16-channel × 2-pixel broadcast exactly; `DST_SURF_STRIDE` and `RDMA_EW_SURF_STRIDE` must both be explicit | Eliminate host expansion for channel-wise bias/scale |
| EW/ERDMA data modes 2/3 | **FAIL for ordinary multi-surface broadcast**: second 8-channel surface became zero | Do not select without another documented layout recipe | Keep mode 1 fallback |
| BRDMA/NRDMA disable bit | **PASS**: `*_CFG[0]=1` disables the client; zero enables it | Explicitly disable both clients in streams that do not consume them | Prevent reads through stale auxiliary addresses |
| auxiliary-DMA → EW mode-0 transition | **PASS after completing the stream**: the prior zero output was stale `RDMA_EW_SURF_STRIDE`, not an arithmetic or ping-pong limitation | Explicit 16-byte operand-surface and 32-byte output-surface strides | Safe to compose when every banked layout register is emitted |
| two 513-entry DPU LUTs | **PASS**: SiLU max error 0.0001221, sigmoid 0.0000610, tanh exact at FP16 on probe vectors | signed int16 table values plus a known output scale | Native approximate nonlinear activations; avoids decomposing them into unsupported transcendental pieces |
| LUT reuse across separate Rocket submissions | **FAIL**: a second task without table writes produced unrelated values for clear, rearm, and untouched pointer states | Reload all 1,026 entries per standalone submission; same-submit chaining remains unproved | Do not build a cross-submit LUT cache yet |
| FP16 comparison result → INT16 WDMA | **PASS** | Captured compare mask writes exact INT16 0/1 | Keep comparison masks in an integer pipeline |
| INT16 EW result → external INT32 WDMA | **PASS** | Exact sign-preserving INT32 output | Produce external INT32 indices/scores without host conversion |
| `EQUAL_EN` on ordinary MIN/MAX | **NO EFFECT in tested layout** | Value output remained ordinary MIN/MAX | Not a proved comparison primitive |
| `BINARY_EN` | **UNRESOLVED/negative**: standard FP16 and raw-output recipes emitted zeros | Packed binary layout is not the normal FP16 layout | Do not lower booleans through it |
| `MINMAX_CTL` sweep | **UNRESOLVED/negative**: odd values yielded stable non-semantic packed bytes | No useful value mapping found | No backend use yet |
| `ERDMA_NONALIGN` | **NO EFFECT** in the one-atom test | Same full atom was read | Does not solve arbitrary scalar tails |
| DPU non-aligned output controls | **UNRESOLVED/negative**: obvious encodings wrote zeros; `SURF_LEN=1` still emitted all eight lanes | Not a proved flat-tail mask | Keep padded/tiled tails |
| C1WC2 layout sweep | **PASS**: 90 exact ADD streams | INT8/INT16/INT32, widths 1/2/3, channels 3/7/8/9/15/16/17/31/32/33 | Proves partial atoms, complete surfaces, and partial final surfaces for the main+ERDMA EW path |
| Integer edge/overflow sweep | **PASS with captured limits** | INT8/INT16 ADD/SUB/MUL/ABS/NEG saturate; INT32 ADD saturates, ABS/NEG preserve `INT32_MIN`, tested SUB with an `INT32_MIN` EW operand emits `INT32_MIN`, and INT32 MUL consumes signed-INT16 EW operands | Lower with explicit dtype/range rules; never promise saturating unary INT32 at `INT32_MIN` |
| BRDMA/NRDMA MUL layout sweep | **PASS/limited**: 60 streams characterize both clients | Width broadcast 1/2/3 and channels through the first 8-lane FP16 atom pass; complete surfaces at 16/32 pass; partial final surfaces after a complete surface (9/15/17/31/33) are unavailable/zero | Use only for complete 8-channel surfaces or a sole partial first surface |
| Simultaneous outside ALU+MUL | **NEGATIVE** for the original-NVDLA interleaved BOTH packing | BS wrote zeros; BN wrote non-semantic values; both following health checks passed | RK does not inherit that NVDLA operand packing; keep ALU-only and MUL-only recipes separate |

The executable proofs are in `elementwise_int.py`:

- `run_fp16_add_register_operand`
- `run_fp16_register_pipeline`
- `run_fp16_five_scalar_ops`
- `run_fp16_bs_bn_activations`
- `run_fp16_four_input_pipeline`
- `run_fp16_per_channel_broadcast`
- `run_fp16_silu_lut`
- `run_fp16_compare_to_int16`
- `run_int16_add_to_int32`

The isolated recovery probes are opt-in and are not included in the default
hardware matrix.  Configured MINUS passed in BS and BN with main-minus-operand
direction; BRDMA and NRDMA MUL-only passed with one FP16 multiplier per
element.  The regroup recipe submitted but returned eight zeros:

- `run_fp16_bs_bn_minus`
- `run_fp16_bs_bn_external_mul`
- `run_fp16_regroup_stride2`

`python3 examples/expt/elementwise_int.py --validate-probes` checks their five
decoded streams without opening the NPU device.

### Quarantined converter observations

Do not use the following observations as backend proofs.  A controlled
`OUT_CVT` probe produced the expected FP16-to-INT16 affine conversion for
offset 1, scale 1, and shift 1.  A following INT16 `EW_CVT` probe submitted
but returned `[2, 5, 8, 11, 14, 17, 20, 23]` instead of the nearest-even
expectation `[2, 6, 8, 12, 14, 18, 20, 24]`; that result is consistent with
rounding negative and positive half-way values downward.  The user reported a
crash immediately after this sequence.  The current-boot kernel log contains
no Rocket/RKNPU fault, IOMMU fault, oops, panic, or watchdog record, so the
failure level cannot be determined from retained logs.

Both converter probes were removed from the executable example.  Their fields
must remain disabled in the tinygrad backend until each probe is isolated after
a clean recovery and followed by the known-good NPU health check.

## External cross-check: `gregordinary/rockchip-npu-notes`

The repository was inspected at commit
`e0c72133186dcb473f81714b616a5448155328c3`.  It is useful as a register-probe
roadmap, not as silicon proof for this tree.  Its README states that the notes
were primarily AI-authored and that their accuracy is not guaranteed.  Most of
the cited HW gates under `tests/` and implementations under `src/` are in a
separate `rocket-userspace` project and are not available in the notes clone;
the PPU capture scripts and decoded Teflon-add capture are the main directly
inspectable artifacts here.

High-value leads to revisit only after the NPU is confirmed recovered:

- `encodings/out-cvt-converter.md` gives an integer accumulator recipe
  `(acc * uint16_scale) >> shift` and reports arithmetic right-shift rounding
  toward negative infinity.  This is compatible with the direction of the
  quarantined half-way-value observation above, but it concerns `OUT_CVT`, not
  enough to validate the crashed `EW_CVT` sequence.
- `teflon-add-capture/regcmd.decoded.txt`,
  `encodings/k-accumulation.md`, and `encodings/mrdma-trap.md` identify
  `COMB_USE=5`, explicit main/EW addresses, and nonzero surface notch/stride as
  a captured combined MRDMA+ERDMA recipe.  They also warn that an unfed enabled
  MRDMA can time out and wedge later submissions.  This is a plausible failure
  class for future diagnosis, not a retained-log diagnosis of this crash.
- `encodings/cbuf-reuse.md` supplies concrete CNA `DATA_REUSE` and
  `WEIGHT_REUSE` task-order recipes.  These could reduce feature or weight DMA
  in a future convolution/matmul backend, but the selected tinygrad branch is
  currently a DPU-EW backend and emits neither CNA nor CORE work.
- `encodings/regcmd-task-model.md` corroborates full per-task register streams,
  `S_POINTER=0xe`, and explicit PC links.  It reports contiguous one-kick
  chaining as FP16-only and integer chaining as corrupt after task zero.  The
  current tinygrad PC chain is already limited to ordinary FP16 EW stages;
  stateful comparison/conversion stages are separated.
- `encodings/ppu-reduce-mean.md` proposes multi-pass spatial global
  average/max/min and reports a PPU-written sub-4 intermediate hazard.  This is
  new test coverage worth adding later; it does not add ArgMax, gather, sort,
  or unpool hardware.
- `encodings/conv-transpose.md` reports no native deconvolution mode.  Its
  implementation is a host-packed dilated input plus rotated/transposed weights
  sent through ordinary forward convolution, so it is a lowering technique,
  not a helpful unproven register.

Two local vendor-capture details sharpen those leads:

- `experimental/rknn/channeltile_6tile_emit.txt` contains an exact FP16
  three-channel CNA input recipe: `NONALIGN_DMA=1`, `GROUP_LINE_OFF=1`,
  `ARGB_IN=10`, and `DATAIN_CHANNEL_REAL=2`.  This upgrades the combination
  from an undocumented guess to **CAPTURED**, but not to a local silicon PASS.
- The same stream repeatedly emits a standalone PPU+PPU_RDMA `1x1x32` maximum
  task with PPU_RDMA `IN_PRECISION=1`, PPU `PROC_PRECISION=0`, constant 16-byte
  strides, fixed scratch addresses, and PC enable mask `0x60`.  TRM section
  36.6.8 defines the two-bit RDMA field by storage width, where 1 means 8-bit;
  PPU processing precision 0 is INT8 in the locally passing integer recipe.
  The capture is therefore a same-format INT8 token task, not evidence of an
  INT16-to-INT8 conversion.

One broad claim conflicts with this machine's checked-in silicon proofs and
must not be imported:

- The notes describe the per-element EW ALU as float-only.  `elementwise_int.py`
  proves same-format integer MAX/MIN/ADD/MINUS/ABS/NEG for INT8, INT16, and
  INT32, plus the documented MUL limits.

The notes' negative native-INT8 PPU result initially appeared to expose an
error in the local cross-check, but the local TRM and vendor capture disprove
that reinterpretation.  `PPU_RDMA_DATA_FORMAT.IN_PRECISION` is named like a
dtype selector but encodes input storage width: 0/1/2/3 mean 4/8/16/32 bits.
The old INT8 and INT16 probes correctly programmed 1 and 2.  Recovered-silicon
adversarial signed same-format probes also pass exactly, proving signed maximum
ordering for both widths.

`python3 examples/expt/pooling.py --validate` performs a
device-free audit of all 36 decoded streams, including signed INT8 and INT16
maximum, the vendor same-format INT8 `1x1x32` geometry, FP16 `-inf` padding,
and every `USE_CNT` encoding.  It currently passes.  On recovered silicon,
signed INT8/INT16 maximum/minimum and FP16 `-inf` padding pass.  Integer
average is exact with Q16 reciprocals for 2x2, 3x3, and 2x3; the older FP16
reciprocal encoding remains a recorded negative control. All eight `USE_CNT`
encodings return the include-padding result 2 rather than exclude-padding 8,
so this field is not an exclude-padding switch in the tested recipe.

Three-pass average, maximum, and minimum also pass through direct NPU-written
8x8→4x4→2x2→1x1 intermediates. A 16x16 `INDEX_EN` probe proves the index is
only six bits (`row%8`, `column%8`), while padding+stride exposes a non-local
vertical phase. Explicit PPU_RDMA line/surface strides work; nonzero notch does
not act as an additive line pitch.

Therefore the clone materially narrows several future experiments, but it does
not change any PASS/FAIL status in this report without a local decoded-register
probe and post-probe health check.

## Mesa Rocket compiler cross-check

A sparse checkout of `chaotic-cx/mesa-mirror` at
`f56d52b26b2937b93a518e3346282e312cfb894f` provides complete compiler recipes,
not just field definitions:

- `rkt_regcmd.c` sets `CNA_CBUF_CON0.WEIGHT_REUSE` only on task numbers greater
  than zero, and `rkt_task.c` selects that path only when the complete weight
  tile remains resident while the input is spatially split.  This establishes
  the required adjacent-task ordering, but not a local performance result.
- For one real INT8 input channel, Mesa enables CNA input conversion with four
  identical scale/offset/truncate lanes and `CNA_CVT_CON5=0x0000ffff`.  That is
  the exact locally passing probe recipe; it does not establish arbitrary per-channel
  masks or FP16 conversion.
- Its fused convolution-plus-add path uses `COMB_USE=5`, both main and EW
  addresses, `ew_stride=max(output_width*output_height,12)`, and matching main
  and EW surface notches.  These fields form one coupled recipe rather than
  independent generic-view knobs.
- Mesa contains no compiler use of PPU, DPU-to-PPU fly-in, deconvolution,
  transpose, regroup, `DATA_REUSE`, or unpool.  Their presence in
  `registers.xml` is therefore not implementation evidence.

These recipes were then exercised where a safe complete stream existed.
`conv_simple.py --rocket-rgb` proves the captured three-channel FP16 path with
`max_diff=0.0071`. The exact one-channel INT8 converter passes with Mesa and
Mesa-compatible translated packing and fails with generic CRS translation
(`max_diff=234`). A two-task spatial split with `WEIGHT_REUSE` produces exact
output, but the first following DPU health task reads stale `48576`; a second
health task recovers. Therefore converter/RGB are positive CNA proofs, while
weight reuse remains unsafe for general dispatch and `DATA_REUSE` was not
speculatively submitted.

`conv_simple.py` contains the exact decoded Mesa
words behind these classifications.  Running
`python3 examples/expt/conv_simple.py --validate-mesa-cna-regs` checks the four
converter lanes, `PER_CHANNEL_CVT_EN`, the first/reused CBUF writes,
`COMB_USE=5`, EW stride, and matching main/EW surface notches without opening
the NPU device.  That offline check passes; it validates the encoder independently
of the silicon results above.

## Complete register-definition audit

The semantic matrix was checked against every non-reserved bitfield in
`experimental/registers.xml`: 199 register records and 512 field occurrences
across PC, CNA, CORE, DPU, DPU_RDMA, PPU, PPU_RDMA, DDMA, SDMA, and GLOBAL.
The per-domain counts and the grouping rule for baseline/control fields are in
`trm_register_matrix.md`.

That enumeration found four families which were previously implicit or
misclassified:

- PPU_RDMA `IN_PRECISION` is a 4/8/16/32-bit storage-width selector, not the
  DPU/PPU arithmetic dtype enumeration.  This correction removes the claimed
  vendor INT16-to-INT8 PPU boundary.
- DPU `EW_DATA_MODE` and `EDATA_SIZE` must be paired with ERDMA data mode and
  size.  Per-pixel 8/16/32-bit and FP16 per-channel mode 0 are proved; mode 2
  and 4-bit payloads are not.
- `EW_TRUNCATE_NEG`, `BN_MUL_SHIFT_VALUE_NEG`, and
  `BS_MUL_SHIFT_VALUE_NEG` are separate unproved signed-shift controls.  No
  retained Mesa/vendor stream sets them nonzero, and original NVDLA cannot be
  copied because its shift register layout differs.
- CNA/CORE expose INT8, INT16, BF16, INT4, and TF32 precision values, while DPU
  exposes INT4/BF16 pointwise encodings.  The checked-in local convolution
  matrix proves FP16 only; alternate precisions are source/reference leads,
  not local DPU-EW proofs.

## Why this helps the selected tinygrad branch

The current renderer sets `_EW_OP_SRC_DMA` in `_EW_CFG_COMMON` and turns Python
float leaves into scratch `RKArg`s (`tinygrad/renderer/rockchip.py`, around
lines 166 and 1563-1663).  The runtime then materializes gather/scratch data on
the host.  It never emits `EW_OP_VALUE_n`, `ERDMA_DISABLE`, BRDMA, or NRDMA.

The minimum useful integration order is:

1. Represent scalar leaves as immediate operands and emit `EW_OP_SRC=0` plus
   `EW_OP_VALUE_n`; set `ERDMA_DISABLE` when EW has no tensor operand.
2. Fuse compatible scalar chains into BS and BN, preserving the observed stage
   order and the distinct ALU/MUL encodings.
3. Assign up to two extra per-channel inputs to BRDMA/NRDMA and one to ERDMA.
   Use FP32 payloads for the proved auxiliary ALU path and one FP16 component
   per element for the proved MUL-only path.  Emit all surface-stride fields
   when changing ERDMA data modes.
4. Retain the existing EW-only fallback for layouts/broadcasts not yet proved.

At the selected commit, the costs and replacements are:

| Current backend behavior | Register finding | Defensible effect |
|---|---|---|
| `lower_ew` assigns every unique FP16 scalar a scratch slot; the runtime repeats its two-byte value across `_scratch_bytes(count)` before execution | `EW_OP_SRC=0` plus `EW_OP_VALUE_0..7`, with ERDMA disabled, is locally proved | Remove one scratch BO and one host fill/ERDMA stream per unique scalar operand represented this way; the scalar encoding differs for ALU and MUL |
| Every compatible half ALU node becomes an `RKEWOp` body, although bodies normally share one PC-chain ioctl | Configured BS/BN ALU+MUL plus EW executes five scalar operations in one body | Replace as many as five order-compatible scalar bodies with one and remove their intermediate scratch traffic; do not promise five fewer ioctls |
| A four-input sum needs three binary EW bodies and intermediate values | Main + BRDMA + NRDMA + ERDMA exact sum is locally proved | One DPU body for the proved one-pixel/per-channel FP32 auxiliary layout; general per-pixel BRDMA/NRDMA layout is not yet proved |
| Channel-wise constants/broadcasts can be expanded into scratch values | Coupled EW/ERDMA mode 0 repeats one FP16 operand per channel across pixels and surfaces | Avoid host expansion and reduce operand scratch traffic when the view is exactly channel-wise; mode 2 remains unavailable |
| Generic `lower_ew` only accepts half ALU nodes; integer support is confined to specialized paths | EW MAX/MIN/ADD/MINUS/ABS/NEG is locally proved for INT8/INT16/INT32 across 90 C1WC2 layout streams; MUL is proved for INT8/INT16 and has a signed-INT16 second-operand limit for INT32 | Enables a future typed integer EW lowerer with explicit overflow and operand-width rules |
| Simple pooling is expanded through DPU EW/gather work and usually already shares one ioctl | PPU FP16 max/min/average, global average, indices, large kernel/stride, and nonmultiple channels are locally proved | Replace pool-specific task/scratch work after adding PPU serialization/runtime support; does not remove sort/WHERE/indexing workarounds |
| Renderer/runtime contain no CNA or CORE engine path | Convolution geometry, RGB nonaligned input, and one-channel INT8 CVT now have local numeric proofs; weight reuse has a post-chain state-hygiene failure | No immediate line saving. These matter only after a separate CNA/CORE backend and safe task-boundary handling are implemented |

External BS/BN MUL can turn compatible tensor multiplication into one body for
the proved layout.  Regroup/notch and DPU-to-PPU fly-in remain conditional and
must not be counted as savings until a silicon probe passes and the required
layout is represented in the backend.

These findings do not remove gather/scatter, sort, or arbitrary indexing.  They
do remove constant materialization and can reduce compatible scalar chains or
four-input pointwise expressions from several DPU task bodies to one.  The
selected backend already packs ordinary EW bodies into one PC-chain ioctl, so
the direct savings are fewer task descriptors/regcmds, fewer intermediate
scratch buffers, and less DMA; submit count only falls where an existing
barrier forces separate ioctls.

The same distinction applies to PPU.  For example, the selected branch already
expects one submit for simple max-pool and padded average-pool tests.  Native
PPU replaces their expanded EW/gather computation and scratch movement; it
does not by itself prove an ioctl-count reduction.

## Other tested units

PPU pooling, index, precision, packing, and limit results are maintained in
`ppu_pooling_findings.md`, with decoded streams in `pooling.py`.  FP16 native
value pooling is real. Signed INT8/INT16 maximum, minimum, and Q16-reciprocal
average passed with the TRM's 8/16-bit RDMA widths; the FP16 reciprocal is kept
as an inexact negative control. FP16 negative-infinity padding also passed, while
`USE_CNT=0..7` did not change include-padding average semantics.  The external-RDMA
32-bit INT32/FP32 attempts and DPU-RDMA unpool remain negative or unresolved as
documented there.

## CNA/CORE convolution results

The decoded Rocket reference `kernel_6_18/simple_conv_fp16.py` was exercised
without editing it.  Its complete 217-shape matrix passed on silicon.  The
matrix contains 57 depthwise cases, 56 grouped cases, kernels from 1x1 through
7x7, batches 1/2/4/8, channel tiling, and weight-reuse chains.  Therefore these
are not merely TRM claims: ordinary FP16 CNA/CORE convolution,
`CORE_MISC_CFG.DW_EN`, CNA depthwise mode, grouped decomposition, spatial and
pointwise modes, and the existing weight-reuse scheduling are proved.

Additional controlled mutations used the known-passing
4-channel, 9x9, 3x3 Rocket stream and left the submit ABI unchanged:

| Fields | Silicon result | Direct backend use |
|---|---|---|
| `CNA_CONV_CON3.CONV_X/Y_STRIDE=2` | **PASS**: the first 4x4 outputs matched stride-2 convolution, max error 0.00693 | Native strided convolution instead of input gathering |
| `CNA_CONV_CON3.ATROUS_X/Y_DILATION=1` | **PASS**: field value 1 implements mathematical dilation 2; first 5x5 outputs matched with max error 0.00587 | Native dilated convolution instead of expanded kernels/inputs |
| `CNA_PAD_CON0.PAD_TOP/PAD_LEFT=1`, `PAD_VALUE=0` | **PASS**: zero-padded convolution matched with max error 0.00744 | Eliminate host-side padding for top/left and derive right/bottom from output geometry |
| `CNA_PAD_CON1.PAD_VALUE` | **PASS**: FP16 `2.0` is encoded as low-half bits `0x4000`; FP32 `0x40000000` behaved as zero | Native constant padding with the correct FP16 encoding |
| DPU FP16 convolution writeback | **PASS**: exact equality to the FP16-rounded reference on the 4x9x9 case | Avoid external FP32 output when the graph consumes FP16 |
| `CORE_CLIP_TRUNCATE` values 0, 1, 2, `0x40`, `0x41` with FP16 processing | **NO EFFECT** on the tested FP32 output | Do not expect this integer-accumulator control to scale FP16 convolution |

The root-level decoded builder in `conv_simple.py` has a device-free
`--validate-rgb-regs` check and a mainline Rocket `--rocket-rgb` runner. It proves
that the prepared 3-channel FP16 stream
emits `CNA_CONV_CON1=0x6000a120` (`NONALIGN_DMA=1`, `GROUP_LINE_OFF=1`,
`ARGB_IN=10`), `DATAIN_CHANNEL_REAL=2`, aligned channel field 8, and NHWC
packing with the expected padded line. The Rocket result matches the FP16
reference with `max_diff=0.0071`. The old vendor-ioctl path remains unsuitable
for this kernel and was not used.

For the stride and dilation probes, the existing output geometry was left
larger deliberately and only the mathematically valid leading region was
checked.  A compiler must set CNA/CORE/DPU output dimensions to the usual
stride/dilation formula before using the full result.

## Remaining high-value TRM queue

These are not claimed working until a silicon probe is added:

- Simultaneous BS/BN external ALU+MUL packing needs a new RK capture. The
  original-NVDLA interleaved BOTH format is now disproved; configured MINUS and
  isolated external MUL-only remain proved separately.
- Converter rounding/truncation and output-converter modes.  The first
  `BINARY_EN`, `EQUAL_EN`, and `MINMAX_CTL` sweeps were negative as recorded
  above; revisit only with a captured packed-output recipe.
- Add checked-in sigmoid/tanh/GELU table generators around the proved generic
  mapping; cross-submit reload is required, while same-submit reuse is unproved.
- ERDMA modes 2/3, DPU notch addressing, and `COMB_USE` arbitration behavior.
  Mode 0 per-channel (including an auxiliary-DMA transition) and mode 1
  per-element are already proved; the obvious non-align recipes were negative
  as recorded above.
- DPU-RDMA unpooling/index routing (existing safe probes have not produced a
  valid unpool result).
- CNA/CORE deconvolution, `DATA_REUSE`, generalized input converter modes, and
  integer accumulator clipping.  Depthwise, stride-2, dilation-2, top/left
  padding, constant FP16 padding, the exact one-channel INT8 CVT, numeric
  weight-reuse output, FP32 output, and FP16 writeback are proved above. Weight
  reuse is not dispatch-safe until its immediate post-chain health check passes.
- PC chaining/ping-pong and multicore fields only after topology-safe proofs;
  submit ABI fields must not be guessed.
