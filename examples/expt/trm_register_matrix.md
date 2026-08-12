# RK3588 NPU semantic-register matrix

This is the exhaustive triage of non-reserved RK3588 NPU fields that can
change an operation, data layout, conversion, reuse policy, or task sequence.
Ordinary dimensions, base addresses, strides, precision selectors, burst
lengths, block enables, and read-only status fields are accounted for at the
end instead of being repeated register by register.

The target backend is `~/tinygrad` branch `rockchip-2608-ew-restart`, commit
`e3e93fa43e140570792e8c970d75188a55a56e3a`.  That branch currently emits
DPU+DPU_RDMA elementwise work only; it has no CNA/CORE convolution emitter and
no PPU emitter.  "Future" below means useful only after one of those engines is
added.

Status vocabulary:

- **PASS** / **NEGATIVE**: observed locally on this Orange Pi RK3588.
- **CAPTURED**: a decoded vendor/Mesa stream exists, but there is no isolated
  local silicon proof in `examples/expt/`.
- **REFERENCE**: another project reports a hardware result, but its gate was
  not reproduced locally.
- **UNPROVED**: only the TRM/register definition or an incomplete recipe exists.
- **QUARANTINED**: do not submit; it is in the sequence after which the user
  reported a crash, or it changes equally dangerous task/DMA topology.

## PC and ping-pong control

| Fields | Status | Potential value | Decision |
|---|---|---|---|
| `PC_BASE_ADDRESS`, `PC_REGISTER_AMOUNTS`, `GLOBAL.OPERATION_ENABLE` | **PASS**, mandatory in every decoded stream | Dispatch only | Already emitted; no graph capability |
| All block `S_POINTER` fields | **PASS** with `0xe`; external notes also report stale/empty groups without full reinitialization | Full per-task register banking | Emit the full configuration and `0xe`; never inherit a prior job |
| `TASK_NUMBER`, PC forward links | **REFERENCE** for contiguous FP16 chaining; tinygrad already restricts its ordinary chain to FP16 EW bodies | Fewer kicks/IRQs | Keep current restriction; integer/stateful chains need separate proof |
| `TASK_PP_EN`, `TASK_COUNT_CLEAR`, `TASK_DMA_BASE_ADDR` | **QUARANTINED/UNPROVED** locally | Prefetch or offset-addressed multi-task submission | Kernel/submit ownership; do not mutate from an example |
| `INTERRUPT_{MASK,CLEAR,STATUS,RAW_STATUS}`, `TASK_STATUS`, block `S_STATUS` | Register-defined diagnostics | Better hang diagnosis | Runtime/driver instrumentation only, not a renderer operation |
| Pointer clear, pointer/executer select bits | **QUARANTINED** except the established `0xe` recipe | Recovery or manual bank selection | Never sweep while submitting work |

## CNA input, convolution, and CBUF

| Fields | Status | Potential value | Decision |
|---|---|---|---|
| Direct `CONV_MODE`, precision, geometry, kernel groups, CBUF bank/entry fields | **PASS** through the complete 217-case FP16 convolution matrix | Native convolution/matmul foundation | Future CNA backend baseline |
| CNA/CORE precision values INT8=0, INT16=1, BF16=3, INT4=6, TF32=7 | INT8 has a Mesa compiler recipe; the others are **TRM/REFERENCE**, not locally proved in the checked-in convolution matrix | Quantized or alternate-precision convolution/matmul; INT4 principally reduces model footprint | Future CNA backend only. Preserve dtype-specific activation/weight packing and accumulator output type; do not infer DPU-EW support from CNA precision support |
| Depthwise `CONV_MODE=3` + `CORE.DW_EN` | **PASS** in 57 matrix cases | Depthwise convolution | Future native lowering |
| Grouped decomposition | **PASS** in 56 matrix cases | Grouped convolution | Future native lowering; it is task decomposition, not a new selector |
| `CONV_X/Y_STRIDE` | **PASS** for mathematical stride 2 | Avoid input gathering | High-value future CNA feature |
| `ATROUS_X/Y_DILATION` | **PASS**: field 1 means mathematical dilation 2 | Avoid expanded input/kernel | High-value future CNA feature |
| `PAD_TOP`, `PAD_LEFT`, `PAD_VALUE` | **PASS**, including FP16 constant pad encoded in the low halfword | Avoid host padding | High-value future CNA feature |
| `DECONV`, `DECONV_X/Y_STRIDE` | **UNPROVED** locally; `rockchip-npu-notes` instead reports host dilation/weight rotation followed by forward conv | Native ConvTranspose would avoid zero insertion | High value but medium/high risk; isolate from a captured vendor deconv stream before any mutation |
| `NONALIGN_DMA`, `ARGB_IN` | **PASS**: the one-task Rocket RGB convolution `b1_c3_h5_w5_oc16_wic3_k3x3` matches the FP16 reference (`max_diff=0.0071`) | Native RGB/small-channel input without host padding to a full atom | Proven CNA recipe; not useful until a CNA backend exists |
| `GROUP_LINE_OFF`, `FEATURE_GRAINS` | **PASS as part of** the three-channel `NONALIGN_DMA` recipe; independent effect unproved | Fetch scheduling and nonaligned line handling | Preserve the complete proved combination; benchmark individual mutations only after correctness |
| `KERNEL_GROUP` | **UNPROVED as an independent knob**; TRM defines output-kernel groups of 32 for INT8 and 16 for INT16/FP16 | Describe larger output-channel groups | Treat as compiler geometry, not a new operation; require a vendor stream with a nonzero value before emitting |
| `NN_MODE` | **UNPROVED** locally; TRM calls it multicore co-work mode | Cooperative MAC-array shapes | High topology risk; mainline Rocket already schedules cores through fds, so do not sweep |
| `SURF_MODE` | Baseline values exercised by the convolution matrix, alternate modes not isolated | Two/four serialized surfaces | Future layout optimization only |
| `WEIGHT_REUSE` | **NUMERIC PASS, STATE-HYGIENE FAIL**: a captured two-task spatial split with task 1 bit 13 set produced exact INT8 convolution output, but the first following known-good DPU add returned stale `48576`; a second add recovered | Skip repeated weight DMA across spatial split tasks | Do not enable generally until the post-chain state transition is understood and one immediate health job passes |
| `DATA_REUSE` | **REFERENCE**, not submitted locally: current Mesa Rocket does not emit it, and the weight-reuse chain polluted the first following job | Skip repeated input-feature DMA | Quarantined pending an exact adjacent output-channel capture and clean state-hygiene proof |
| Input `CVT_BYPASS=0`, four scale/offset/truncate lanes | **PASS** for the exact one-real-channel INT8 Mesa recipe | Fused input normalization/quantization | Mesa and Mesa-compatible translated packing are exact; generic `translate_crs_int8` is numerically wrong. Do not generalize to arbitrary channels or FP16 conversion |
| `PER_CHANNEL_CVT_EN` | **PASS as part of** the one-real-channel converter recipe at `0x0000ffff`; independent lane meaning unproved | Enable the captured converter lanes | Preserve the exact mask; do not generalize it to all 32 bits |
| Input `CVT_TYPE`, `ROUND_TYPE`, `DATA_SIGN` outside that recipe | **UNPROVED** locally | Other signedness/rounding modes | Do not sweep independently; change one field only after the complete Mesa recipe passes |
| Decoded Mesa CVT/reuse/notch encoder | **OFFLINE PASS** through `conv_simple.py --validate-mesa-cna-regs`; CVT also has the exact silicon proof above | Preserve coupled compiler recipes | Encoder proof does not upgrade unsubmitted reuse/notch combinations |
| `FC_SKIP_EN`, `FC_SKIP_DATA`, `DATA_OFFSET`, `WEIGHT_OFFSET`, `FC_DATA_BANK`, FC DMA sizes | **UNPROVED**; TRM describes feature-value zero skipping | Sparse fully-connected/matmul weight-traffic reduction | Potentially large future gain, but needs the FC-specific packed layout; not a one-bit conv mutation |
| `DCOMP_*` | Dense pass-through is present in references; compressed CWT/WMB/WGS is **REFERENCE**, not locally proved | Compressed sparse weights and zero skipping | Lower priority than CBUF reuse; requires a complete compressed buffer recipe |
| `CSC_DO_EN`, `CSC_WO_EN`, `CMD_FIFO_SRST` | Debug/disable/reset controls | None for graph lowering | Do not probe |
| CNA clock-gate disable bits | Unproved performance/power controls | At most latency tuning | Driver/performance work, not backend semantics |
| `OV4K_BYPASS`, data/weight burst lengths | Established baseline only | DMA tuning | Benchmark only after the functional backend exists |

## CORE accumulator

| Fields | Status | Potential value | Decision |
|---|---|---|---|
| `PROC_PRECISION`, output geometry | **PASS** in convolution matrix | Mandatory | Future CNA/CORE baseline |
| `QD_EN` | Exercised only as part of existing reference streams, not isolated for integer requantization | Integer output path | Revisit with a captured INT8 convolution, not FP16 |
| `CLIP_TRUNCATE`, `ROUND_TYPE` | **NEGATIVE** for FP16 values 0, 1, 2, `0x40`, `0x41`; integer path unproved | Integer accumulator scaling/rounding | No FP16 benefit; future INT8-only probe |
| `MAC_GATING`, `SOFT_GATING` | Baseline settings only | Power/possibly sparse work suppression | No graph capability; do not sweep during correctness work |

## DPU pipeline and writeback

| Fields | Status | Potential value | Decision |
|---|---|---|---|
| `FLYING_MODE=1`, outside output mode | **PASS** in every standalone EW example | DPU-only operation | Current backend baseline |
| DPU-to-PPU output mode, `PPU.DPU_FLYIN` | **UNPROVED** locally; no complete producer/consumer capture was found in the local vendor/reference trees | Fuse elementwise/conv directly into pooling without DRAM RDMA | Do not guess the combined enable/topology fields; require one complete captured stream first |
| `COMB_USE` in DPU feature mode | Ordinary value is baseline; combined topology not locally isolated | Coordinate combined main/elementwise streams | Treat with the DPU_RDMA `COMB_USE` risk below |
| `NONALIGN`, `SURF_LEN`, `MC_SURF_OUT` | **NEGATIVE/unresolved** for flat DPU tail probes | Avoid padded scalar/tail tiles | Do not lower tails through current recipes |
| `RGP_TYPE`, `RGP_CNTER` | **NEGATIVE/UNRESOLVED** for the decoded one-in-two FP16 recipe: submission succeeded but all eight outputs were zero | Regular lane selection (1 of 2/4/8), regrouping | Do not lower regular subsampling without a complete captured writer-layout recipe |
| `TP_EN`, `TP_ORG_EN`, `TP_PRECISION` | **REFERENCE** says full INT16 matmul output is available only transposed and narrowed; no local proof | Transpose/narrow output and possible matrix-layout savings | Future CNA backend probe; does not help present EW-only renderer |
| `BS_OW_OP`, `OD_BYPASS`, `OW_SRC` (CPEND) | `BS_OW_OP=0x80-weight_zp` is **CAPTURED/REFERENCE** in quantized depthwise streams; ordinary paths bypass CPEND | Weight-zero-point correction before writeback | Relevant to a future quantized CNA backend, not current FP16/elementwise lowering; outside CPEND packing remains unproved |
| `OFFSET_PEND`, `SIZE_E_*` | `OFFSET_PEND=0` and Size-E baseline are captured; nonzero offset/extra-channel behavior is unproved | Fill padded channels and control writer geometry | Medium future tail/layout probe; malformed writer geometry can corrupt memory |
| `MINMAX_CTL` | **NEGATIVE/unresolved** sweep produced stable non-semantic packed bytes | Possible min/max probability format | No tinygrad lowering |
| `DST_SURF_STRIDE`, `SURFACE_ADD.SURF_ADD`, `WDMA_SIZE_*`, `SIZE_E_*` baseline combinations | **PASS** in established EW/convolution and INT16-to-INT32 layouts; independent arbitrary values are not operations | Correct multi-surface writer packing | Treat as derived layout invariants. `SURF_ADD` is a byte-address increment (low four bits fixed zero), not a surface-count scalar despite the TRM wording |
| `EW_DATA_MODE`, `EDATA_SIZE` coupled to ERDMA `DATA_MODE`, `DATA_SIZE` | **PASS** for per-pixel modes with 8/16/32-bit operands and for FP16 per-channel mode 0; mode 2 and 4-bit size remain unproved | Per-channel broadcast and future compact operands without host expansion | Emit both sides consistently. Current tinygrad can use proved mode 0/1; do not use mode 2 or infer dtype from the two-bit size alone |
| DPU INT4 (`IN/PROC/OUT_PRECISION=6`, `EDATA_SIZE=0`) and BF16 (`=3`) EW paths | **UNPROVED locally**; the TRM exposes the encodings but no retained standalone RK stream proves EW packing/arithmetic | Compact integer EW or BF16 graphs | Low immediate value because the selected renderer advertises only FP16; revisit only with a captured packed operand/output recipe |
| BS/BN configured ADD and MUL | **PASS**, including five fused scalar operations | Fuse affine/scalar chains | Immediate tinygrad opportunity: fewer task bodies/regcmd bytes and scratch intermediates; ordinary EW is already PC-chained into one ioctl |
| BS/BN configured MINUS algorithm | **PASS** for both stages; direction is main minus configured FP32 operand | Subtract a scalar before EW | Immediate tinygrad opportunity alongside configured ADD |
| BS/BN `RELUX_EN`, ReLU, `MUL_PRELU` | **PASS** | Fuse clamp/ReLU/leaky-ReLU | Immediate tinygrad opportunity |
| BS/BN outside ALU via BRDMA/NRDMA | **PASS**, including four-input sum; tested operands are FP32 | Fuse two extra per-channel additive tensors | Immediate tinygrad opportunity with strict packing |
| BS/BN outside MUL via BRDMA/NRDMA `DATA_USE[2]` | **PASS** for both clients with one FP16 multiplier per element | Fuse tensor multiplication before EW | Useful for the exact proved layout; extend multi-surface/tail coverage before generic lowering |
| BS/BN simultaneous outside ALU+MUL, `*_TRUNCATE_SRC`, positive/negative shifts | **NEGATIVE** for original-NVDLA interleaved BOTH packing: BS writes zero and BN writes non-semantic values; shifts/truncate remain unproved | Per-channel affine quantization in one stage | RK does not inherit that NVDLA packing. Keep proved ALU-only and MUL-only recipes separate |
| `EW_TRUNCATE_NEG`, `BN_MUL_SHIFT_VALUE_NEG`, `BS_MUL_SHIFT_VALUE_NEG` | **UNPROVED**: no local vendor/Mesa stream programs a nonzero value, and original NVDLA has a different shift-register layout | Asymmetric signed requantization/rounding | Potential future integer lowerer feature; first probe must use mixed positive/negative lanes with all other shifts zero and must not be combined with the quarantined converters |
| BRDMA/NRDMA CPEND/TRT `DATA_USE` lanes | **UNPROVED** | External writer-pend or per-channel shift operands | Needs captured packing; lower priority than simultaneous ALU+MUL |
| EW MAX/MIN/ADD/MINUS/ABS/NEG | **PASS** for INT8, INT16, INT32 across widths 1/2/3 and partial/complete C1WC2 surfaces | Native integer elementwise | INT8/INT16 overflow saturates; INT32 ABS/NEG preserve `INT32_MIN`, and the tested INT32 subtract with `INT32_MIN` EW operand emits `INT32_MIN` |
| EW MUL | **PASS** for INT8/INT16 across multi-surface layouts; INT32 second operand is signed 16-bit | Native integer multiply within limits | Immediate tinygrad opportunity with operand-width guard |
| BRDMA/NRDMA MUL broadcast layout | **PASS/limited** across 60 streams: widths 1/2/3, channels ≤8, and complete 8-channel surfaces pass; a partial final surface after a complete surface is zero/unavailable | Tensor multiply broadcast | Restrict to a sole partial first surface or complete FP16 atoms |
| EW DIV/FLOOR/CEIL | **PASS as floating semantics**, not integer arithmetic | FP16 division/rounding | Do not map integer ops to these selectors |
| EW ReLU/ReLU-X/PReLU | Existing decoded recipes, but not isolated by this integer campaign | Leave BS/BN free for other fused work | Current tinygrad already emits these FP16 forms; keep their existing regression coverage |
| `EW_EQUAL_EN`, `EW_BINARY_EN` | **NEGATIVE/unresolved** in ordinary layouts | Packed compare/mask output | No lowering without a captured packed-output recipe |
| FP16 comparison pipeline to INT16 WDMA | **PASS**, exact 0/1 | Integer masks without host conversion | Immediate tinygrad opportunity |
| INT16 result to external INT32 WDMA | **PASS**, sign preserving | External INT32 indices/scores | Immediate tinygrad opportunity |
| `EW_OP_SRC=0`, `EW_OP_VALUE_0..7`, `ERDMA_DISABLE=1` | **PASS** for scalar ALU/MUL | Remove scalar scratch allocation and ERDMA | Highest-value immediate simplification |
| `EW_CVT_*`, `OUT_CVT_*` affine modes | **QUARANTINED** after the reported crash; observations are recorded separately | Mixed-precision and requant boundaries | Keep disabled until clean recovery and one-field isolated probes |
| `FP32TOFP16_EN` in established FP16 conv/LUT streams | **PASS** only in those established recipes | Narrow FP32 accumulator/LUT result | Use only within the proved recipe; not evidence for arbitrary conversion |
| LUT access/config/info/start/end/slope fields | **PASS** for checked-in SiLU/sigmoid/tanh mappings | Nonlinear activation | Useful, but every standalone submit must reload both 513-entry tables |
| Cross-submit LUT reuse | **NEGATIVE** | Avoid table programming | Do not cache across jobs |

## DPU_RDMA

| Fields | Status | Potential value | Decision |
|---|---|---|---|
| MRDMA main + ERDMA per-pixel operand | **PASS** in standalone EW | Two-input DPU operation | Current backend baseline |
| `MRDMA_DISABLE=1`, `ERDMA_DISABLE=1` | **PASS** when the corresponding stream is unused | Prevent stale/unfed reads | Emit explicitly |
| BRDMA/NRDMA `DATA_USE` ALU lane | **PASS** for both clients | Extra per-channel additive inputs | Immediate tinygrad opportunity |
| ERDMA data mode 0 | **PASS** per-channel broadcast with explicit output/EW surface strides | Eliminate expanded channel constants | Immediate tinygrad opportunity |
| ERDMA data mode 1 | **PASS** per-pixel/element | Ordinary tensor operand | Current baseline |
| ERDMA data modes 2/3, `SURF_MODE` | **NEGATIVE/unresolved** for ordinary multi-surface broadcast | Alternate serialized layout | Do not select without a captured layout |
| `ERDMA_NONALIGN` | **NEGATIVE** in a one-atom test | Tail reads | No current use |
| `EW_LINE_NOTCH_ADDR`, main/EW `SURF_NOTCH`, line notch | **MESA COMPILER SOURCE/CAPTURED**, not locally isolated: combined conv+add uses `ew_stride=max(output_area,12)` and the same derived surface notch for main/EW | Advance through split convolution surfaces without host expansion | Preserve the complete conv+add formula; not yet a generic strided-view recipe |
| `COMB_USE=5` with main and EW addresses | **MESA COMPILER SOURCE/CAPTURED** for combined convolution-main + ERDMA operand | Conv epilogue, K accumulation, residual fusion | High value but **QUARANTINED**: an enabled unfed MRDMA can wedge the engine |
| `MRDMA_FP16TOFP32_EN` | Present in proved FP16 paths but not isolated | Wider internal main operand | Keep recipe-specific |
| `UNPOOLING_EN`, unpool kernel/stride/pad/method | **NEGATIVE/unresolved**: safe bypass mutations returned zero | Pool-index scatter/upsample | No implementation until a vendor capture supplies complete geometry/index routing |
| RDMA arbiter weights | Explicit nonzero values work in multi-input probes | Fair service for active clients | Emit deterministic defaults; optimization only |
| `OV4K_BYPASS`, burst length | Baseline only | DMA tuning | No semantic benefit |

## PPU and PPU_RDMA

| Fields | Status | Potential value | Decision |
|---|---|---|---|
| Average/max/min selectors | **PASS** | Native spatial pooling | Future PPU backend specialization; replaces expanded EW/gather work, but simple pools already use one PC-chain ioctl |
| `INDEX_EN`, `INDEX_ADD` | **PASS/limited**: ordinary local position works, but row/column alias modulo 8; padding+stride exposes a non-local vertical phase | MaxPool indices | Use only for ordinary windows ≤8 with a separately proved geometry; not general ArgMax |
| FP16 precision | **PASS** through external PPU_RDMA | Native value pooling | Supported with the checked-in packing |
| Same-format INT8/INT16 precision | **PASS** for signed maximum/minimum and average with RDMA storage widths 1/2 | Native integer pooling | Integer average must use Q16 reciprocals, not the FP16 reciprocal encoding; 2x2, 3x3, and 2x3 pass exactly |
| PPU_RDMA width 0 (4-bit input) with a matching PPU processing precision | **UNPROVED**; defined only as a storage width by the TRM | Compact integer pooling | Requires nibble packing and a proved PPU arithmetic-precision value; low priority for the FP16-only selected renderer |
| Mixed PPU_RDMA width / PPU processing precision | **UNPROVED**: the supposed vendor INT16-to-INT8 task is actually RDMA width 1 (8-bit) plus PROC 0 (INT8) | Possible cast/narrowing | No captured mixed-format recipe remains; do not lower a cast from the independent fields alone |
| INT32/FP32 processing precision | **NEGATIVE for external-RDMA width 3**: PROC values 4/5 submitted and wrote zero | Wider pooling | Do not use on external PPU_RDMA; DPU-flyin remains a separate unproved path |
| Kernel/stride through 16, padding geometry through 7 | **PASS** individually; kernel 16 + pad 7 combination is **NEGATIVE** | Cover normal pool geometry | Do not assume independently valid maximum field values compose |
| `PADDING_VALUE_1/2_CFG` nonzero encoding | **PASS** for FP16 `-inf` (`0xFC00` in the low halfword, high part zero) | Correct padded max on all-negative inputs | Use this encoding for FP16 maximum padding |
| Reciprocals | **PASS** with the checked-in 17-bit encoding, including direct 3x5 global average | Average/global average | Use `pooling.py` encoding, not an assumed host `/H` repair |
| `USE_CNT` | **NEGATIVE** as an exclude-padding control in the tested average recipe: all values 0..7 return include-pad result 2, not exclude-pad result 8 | Undocumented/conditional counter behavior | Do not use it to implement `count_include_pad=False` without another complete recipe |
| `NOTCH_ADDR`, PPU/PPU_RDMA unusual line/surface strides | **PASS** for explicit 80-byte line/256-byte surface strides; `NOTCH_ADDR=2` corrupts later values | Strided/subview pooling | Use explicit RDMA strides and keep notch zero; notch is not an additive line pitch in this recipe |
| `MC_SURF_OUT`, `NONALIGN`, `SURF_LEN` | **UNPROVED** | Nonmultiple channels/tail writer layout | Existing padded-channel packing already works; low priority |
| PPU input directly from DPU | **UNPROVED** | Remove intermediate DRAM read/task | High-value but topology-changing; one known-safe DPU output into one 2x2 pool is the first probe |
| Multi-pass spatial global avg/max/min | **PASS** for 8x8→4x4→2x2→1x1 through NPU-written BOs | Kernels larger than 16; reduce over arbitrary smooth spatial sizes | Separately fence each pass; no host copy/repair is required for the tested sub-4 intermediates |
| PPU_RDMA dimensions, address, line/surface strides | **PASS**, mandatory | External input | Baseline only; precision support is tracked separately above |

## DDMA, SDMA, and non-semantic fields

| Fields | Status | Potential value | Decision |
|---|---|---|---|
| DDMA/SDMA outstanding counts, read/write weights, fixed/weighted arbitration, QoS, AXI cache/protection/burst/size, WSTRB | Register-defined only in this userspace campaign | System-level bandwidth/latency tuning | Driver-owned global behavior, no graph capability; never sweep from an example |
| DDMA/SDMA FIFO clear | Recovery/debug control | Clear DMA state | Destructive global action; driver only |
| DDMA/SDMA error IDs and idle status | Diagnostic | Better fault reporting | Read-only driver instrumentation |
| Version, all block status, ordinary geometry/address/stride/channel/size registers | **PASS** wherever their owning engine is proved | Required configuration | Not independent operations and not separate probe candidates |
| Clock gates, soft resets, operation-enable shadow registers | Baseline/debug only | Power or recovery | Keep established values; never interpret as graph operations |

## XML inventory completeness

The source inventory is `experimental/registers.xml`.  A mechanical enumeration
of every `reg32` containing at least one non-reserved bitfield produces 199
register records and 512 non-reserved field occurrences.  Repeated interrupt
bits are counted at each mask/clear/status address.  The totals below make the
scope of “exhaustive” checkable; reserved fields alone are excluded.

| Domain | Register records | Non-reserved field occurrences | Classification in this matrix |
|---|---:|---:|---|
| PC | 12 | 67 | Dispatch, task/ping-pong control, and diagnostics |
| CNA | 52 | 104 | Convolution/input/CVT/CBUF plus DCOMP; ordinary geometry grouped |
| CORE | 8 | 20 | Accumulator precision, quantization, clipping, and gating |
| DPU | 52 | 139 | Writer layout, BS/BN/EW, conversion, regroup/transpose, and LUT |
| DPU_RDMA | 20 | 51 | Main/BS/BN/EW clients, layout/notches, unpool, and arbitration |
| PPU | 20 | 42 | Pool semantics, padding, index, flying, and output packing |
| PPU_RDMA | 10 | 17 | External input geometry, address/strides, and storage width |
| DDMA | 12 | 33 | Global DMA policy, diagnostics, and reset |
| SDMA | 12 | 33 | Global DMA policy, diagnostics, and reset |
| GLOBAL | 1 | 6 | Per-engine operation enables |
| **Total** | **199** | **512** | All semantic candidates above; baseline/control fields grouped explicitly |

This accounting found and corrected four previously implicit or misclassified
families: PPU_RDMA storage width, DPU `EW_DATA_MODE`/`EDATA_SIZE`, the three
negative-data shift fields, and DPU/CNA alternate precision values.  Ordinary
address, dimension, stride, channel, enable, status, interrupt, burst, QoS,
clock, and reset fields remain accounted for but are not independent graph-op
probes.

## Absence results

No register domain contains an indexed address generator, arbitrary permutation,
sort network, general `WHERE`, or scatter destination index.  `PPU.INDEX_EN`
reports only a local pool-window coordinate, CNA/DPU notch fields express regular
strides, and DPU regrouping expresses regular lane selection.  Therefore arbitrary
gather/scatter, sort, and fancy indexing cannot be claimed from the TRM surface.

The recovery-gated silicon order and result is:

1. One known-safe FP16 health task.
2. BS configured MINUS, then a health task; BN configured MINUS, then health: **PASS**.
3. BRDMA MUL-only with one FP16 component per element, then health; repeat for NRDMA: **PASS**.
4. DPU FP16 regroup one-in-two, then health: **NEGATIVE/UNRESOLVED**, all-zero output.
5. PPU signed INT8/INT16 maximum, FP16 `-inf` padding, and `USE_CNT=0..7`, each followed by health: signed pooling and padding **PASS**; every `USE_CNT` value retains include-padding semantics.
6. PPU-written 8x8→4x4→2x2→1x1 average/max/min, separately fenced: **PASS**.
7. PPU explicit strided RDMA: **PASS** with explicit line/surface strides;
   nonzero notch is **NEGATIVE** for additive-pitch semantics.
8. Captured three-channel CNA convolution and exact one-channel INT8 CVT:
   **PASS**. Generic CRS input packing is **NEGATIVE**.
9. Captured two-task CNA weight reuse: numeric **PASS**, but the first following
   health task **FAILS** with stale state and the second recovers. `DATA_REUSE`
   was therefore not submitted.

DPU-to-PPU flying, converters outside the exact CNA recipe, combined MRDMA,
PC/task mutations, deconvolution, unpooling, reset, and global DMA fields remain
quarantined until a complete capture and clean recovery protocol exist.
