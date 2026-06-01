# conv_expt Plan

Date: 2026-05-31

## Goal

Build a clean RK3588 FP16 CONV implementation that is small enough to replace the
current strategy-heavy examples.

The implementation should be driven by one CBUF planner and one descriptor
emitter contract:

```text
shape
  -> CBUF pressure
  -> NONE/BY_Y/BY_K/BY_YK
  -> Y/K windows
  -> descriptor families with offsets/banks/reuse
  -> decoded regs
  -> submit
```

The target is a `<1000` line user-facing `examples/conv.py`. Do not clean the
current 2k-line file just for formatting. First prove the planner abstraction and
then replace the old runtime with the smallest hardware path that actually
submits.

## Active Operating Plan

This file is now both an active plan and a historical milestone log. Read this
section first; the detailed milestone sections below are evidence and audit trail.

Current direction:

| Priority | Action | Runtime rule |
|---|---|---|
| 1 | Sketch the runtime exact-11 BY_K adapter boundary while preserving the fail-closed guard. | It must show where generated rows, task metadata, PC tails, regcmd sizing, core mask, and subcore assignment plug into `examples/conv.py` without enabling submit. |
| 2 | Run a submit dry-run size audit for the h14 exact-11 shape without opening DRM. | Compute task/regcmd/input/weight/output BO sizes and validate allocation assumptions before any submit attempt. |
| 3 | Keep `examples/conv.py` plus `conv_tile_planner.py` under the current `900`-line active runtime unless a broad family passes. | Do not import proof-only helpers or captured row tables. |
| 4 | Keep non-pointwise BY_Y h160 offline. | It still needs generated `55/55/54` windows and useful prefix rows. |

Plan-size rule:

- Do not append another long narrative milestone unless it changes the active
  runtime decision, produces a runtime-shaped no-submit checker, or records a
  hardware result with pre/post health checks.
- New evidence should usually update this active section and one latest milestone,
  not expand every historical section.
- If the milestone log keeps growing, split old Milestones 1-30 into an archive
  file instead of making this active plan harder to scan.

## Current State

Useful files:

| File | Role |
|---|---|
| `conv_expt/conv_tile_cpu.py` | Offline CPU proof harness. It imports the extracted planner API. Keep it as proof/report infrastructure, not as a final runtime dependency. |
| `conv_expt/conv_tile_planner.py` | Compact no-submit planner module. It owns the CBUF split/descriptor contract and contains no DRM, task structs, PC-chain tails, or `npu_submit`. |
| `conv_expt/conv_tile_result.md` | Broad offline result through the difficult-shape evidence phase and no-submit fixture state. It is now background evidence, not the active roadmap. |
| `conv_expt/conv_evidence.md` | Consolidated evidence: shape strategy counts, RK3588 CBUF constants, ONNC/NVDLA/RKNN findings, hardware proof boundaries. |
| `conv_expt/shape_stratgery.md` | Historical grouping of the 217 `conv_new.py` test shapes by old strategy. Do not preserve these names as runtime strategy branches. |
| `experimental/rknn/librknnrt_conv_channel_tile_decomp.md` | RKNN reverse-engineering note showing one `NONE/BY_Y/BY_K/BY_YK` planner, descriptor families, reuse pass, and current submit blockers. |
| `examples/conv_tiles.py` | Current hardware reference for vendor `rknpu_driver`; keep as reference for future coverage expansion. |
| `examples/conv_no_submit_materializer.py` | Importable no-submit materializer for fixture parity. It should not grow into the final runtime. |
| `examples/conv_no_submit_fixtures.py` | Tiny fixture registry for h14 `k_tile`, h7 `k_tile`, and setup descriptors/profiles. Proof-only unless a fixture is directly promoted into the runtime emitter. |
| `examples/conv_h14_k_tile_no_submit.py` | Heavy RKNN parity wrapper. It should not be imported by the final small `conv.py`. |
| `examples/conv_h14_task_layout_no_submit.py` | Offline-only BY_K task-layout checker. It now validates the older RKNN prefix, decodes the recovered live h14 topology, generates live body/auxiliary row contents, task metadata, and PC-tail classification no-submit, and reports runtime submit-field deltas. It must not be imported by the final small `conv.py`. |
| `examples/conv_output_bo_map_no_submit.py` | Offline-only output BO mapping checker for the h160 spatial BY_Y output-size failure. It opens/maps BOs only and does not submit. |
| `examples/conv_spatial_by_y_layout_no_submit.py` | Offline-only h160 spatial BY_Y task-layout checker. It compares the local full-output submit rows, compact six-task hypothesis, exact RKNN 18-row topology, and mined non-PPU/non-tail useful prefix without submitting. |
| `examples/conv_small_no_submit.py` | `conv.py`-shaped no-submit entry point. Treat as old scaffold now that `examples/conv.py` is planner-driven. |
| `examples/conv_tiles_no_submit.py` | Registry-driven hardware-cleanup scaffold. Useful as a fixture check, not as a delivery target. |
| `examples/conv.py` | Current planner-driven user entrypoint. It has passing setup/NONE and pointwise BY_Y hardware paths, with fail-closed guards for pointwise-wide NONE, BY_K, non-pointwise BY_Y, and BY_YK. |
| `examples/conv_legacy.py` | Previous strategy-heavy entrypoint, moved aside during replacement. |
| `examples/conv_no_submit_closure.py` | Proof-only 108-qword setup full-register closure. It is useful for the current setup pass but must be folded, shrunk, or left out of the final runtime budget. |
| `examples/kernel_6_18/conv_mesa.py` | Clean Mesa/Rocket reference that passed the 217 shape suite for INT8-style Mesa semantics, but is not a working FP16 solution. |
| `examples/kernel_6_18/conv.py` | Mainline/Rocket FP16 attempt blocked by Rocket submit/UAPI translation gaps for direct spatial RKNN-style schedules. |
| `examples/kernel_6_18/conv_new.py` | Mainline/Rocket 217-shape source; do not edit unless explicitly requested. |

Current line-count facts from 2026-05-31:

| File | Lines | Runtime decision |
|---|---:|---|
| `examples/conv.py` | 489 | Active planner-driven user entrypoint. |
| `conv_expt/conv_tile_planner.py` | 411 | Acceptable small planner import if it does not grow. |
| `examples/conv_no_submit_closure.py` | 105 | Proof-only; no longer imported by the active runtime. |
| `examples/conv_legacy.py` | 989 | Legacy/simple-shape reference, not the normal entrypoint. |
| `examples/conv_tiles.py` | 2588 | Working reference, not an acceptable final shape. |
| `examples/gemm.py` | 545 | Style and submit-pattern reference. |

Progress on the `<1000` line goal:

- The planner approach is validated: RKNN uses `NONE/BY_Y/BY_K/BY_YK`, tile
  records, descriptor families. The six-strategy `conv_new.py` branches are local
  scaffolding, not hardware/compiler truth.
- The most important milestone is complete: at least one CONV has submitted to
  hardware through the planner path and matched CPU reference.
- The remaining risk is submit/emitter quality, not planner proof. Generic small
  `NONE/setup` and pointwise `BY_Y/y_tile` have hardware coverage; pointwise-wide
  `NONE`, non-pointwise `BY_Y`, `BY_K/k_tile`, and `BY_YK` remain intentionally
  fenced.
- More evidence-only reports are now harmful unless they directly change the next
  hardware attempt.

Current conclusion:

- Planner proof is complete enough for cleanup: all 217 shapes classify as
  `NONE/BY_Y/BY_K/BY_YK`, generic-only CPU passes, RKNN evidence matches the
  known fixture families where we have mined rows.
- Hardware cleanup has shipped a small honest subset: the normal entrypoint is
  planner-driven, c32 setup/NONE passes, one generic small `NONE/setup` path
  passes, pointwise `BY_Y` has batch coverage including c144, and unresolved
  families fail before allocation.
- The latest hardware-facing spatial BY_Y result is Milestone 19: exact h160 RKNN
  capture passed with max error `0.0389099` and post-health PASS. The latest h160
  proof result is Milestone 21: the useful non-PPU/non-tail prefix is mined as
  `setup x3`, `k_half x6`, `y_tile x3`, with y_tile input windows `55/55/54`
  rather than local `64/64/36`. It is still not generated from planner descriptors.
- The latest BY_K result is Milestone 41: exact-11 BY_K has a field-by-field
  hardware-run hypothesis and health plan, but runtime still needs an exact-11
  adapter boundary plus a submit dry-run size audit before any submit-field edit.
- The active runtime path is `examples/conv.py` plus
  `conv_expt/conv_tile_planner.py`. Proof scaffolds are not imported by the
  normal runtime path.
- The `<1000` line target is currently met for the active runtime path with room
  for bounded growth: `489 + 411 = 900` lines. Coverage expansion must replace
  tables and fences with generic code; it cannot append one-off shape entries,
  fixtures, or reports.

Still fenced in the normal runtime:

| Family | Current reason |
|---|---|
| pointwise-wide `NONE` | c144 stayed wrong after RKNN-mined unpadded weights and one CBUF split probe; RKNN uses a multi-descriptor channel schedule rather than the local one-row setup. |
| non-pointwise `BY_Y/y_tile` | h160 spatial row-reg overlay and output BO mapping are solved, and the useful RKNN prefix has been mined, but the planner still emits `64/64/36` local windows instead of RKNN `55/55/54` and does not generate setup/k_half prefix rows. |
| `BY_K/k_tile` | The active path is the exact-11 topology class. Live body rows, auxiliary rows, task metadata, PC tails, fake-DMA submit fields, genericity scope, and the hardware-run hypothesis are generated/proved no-submit; runtime still needs an adapter boundary and BO-size dry-run before any hardware. |
| `BY_YK` | mixed `setup`/`k_half` semantics and K-family reuse/group behavior are unresolved. |

Immediate roadmap:

| Priority | Next action | Why this moves toward `<1000` |
|---|---|---|
| 1 | Sketch the runtime exact-11 BY_K adapter boundary with BY_K still fail-closed. | Milestone 41 names every submit-risk field; the next step is showing the runtime insertion point without enabling hardware. |
| 2 | Do a no-DRM submit dry-run size audit for h14 exact-11. | The first hardware attempt needs known task/regcmd/input/weight/output BO sizes and allocation assumptions before code changes. |
| 3 | Preserve the recovered budget while adding BY_K. | Active runtime is `489 + 411 = 900`; any BY_K implementation must be generated code, not captured rows or imported fixtures. |
| 4 | Keep spatial BY_Y h160 offline. | h160 still needs generated `55/55/54` planner windows plus setup/k_half/y_tile useful prefix rows before any local submit. |
| 5 | Keep pointwise BY_Y as supported, but stop adding labels or smoke tests unless they protect a generic change. | c64/c96/c112/c128/c144/c160/c192 already cover the useful pointwise row-overlay range without runtime growth. |
| 6 | Leave pointwise-wide `NONE`, mixed `BY_YK`, non-pointwise BY_Y, and BY_K runtime enablement fenced until they have one generic, line-budgeted emitter/submit change. | These families need multi-descriptor/auxiliary task scheduling; one-off fixtures would recreate `conv_tiles.py`. |

## Planner Model

Use these high-level split methods:

| Method | Meaning |
|---|---|
| `NONE` | full data and full weights fit |
| `BY_Y` | split output rows / input height |
| `BY_K` | split output channels / kernels |
| `BY_YK` | split rows and output channels |

The split method is only the first planner result. The executable unit should be
a descriptor, not a strategy branch.

Descriptor fields to model offline:

| Field | Why it matters |
|---|---|
| `split_method` | high-level planner classification. |
| `family` / `family_bits` | maps descriptor to RKNN-like setup, `y_tile`, `k_half`, or `k_tile` register family. |
| `semantic_status` / `rknn_executable_equivalent` | prevents unresolved descriptor rows from being treated as executable RKNN-equivalent hardware descriptors. |
| `grain_bits` | historical name for `CNA_CONV_CON2.feature_grains << 4`; keep explicit until each family's formula is known. |
| `y_start`, `input_h`, `output_h`, `output_w` | tile-local spatial window. |
| `k_start`, `oc_count` | tile-local output-channel/kernel window. |
| `feature_off`, `weight_off`, `output_off` | DMA offsets; mixed Y+K schedules should compose these additively. |
| `input_bank_num`, `weight_bank_num`, `cbuf0` | per-descriptor CBUF allocation/reuse fields. |
| `data_reuse`, `weight_reuse` | post-planner reuse state; do not reduce to one global bool. |
| `mc_treat_by_*` | keep in descriptor schema even if single-core execution leaves them default. |

RKNN export evidence suggests this mixed Y+K order for large spatial convs:

```text
for family in [setup, k_half, k_tile]:
  for k_window in family_k_windows:
    for y_window in y_windows:
      emit descriptor with additive offsets
```

This is the planner direction: build independent Y and K windows, then combine
them into descriptor families. Do not add more shape-specific submit loops.

Expected collapse of old strategy names:

| Old branch family | Desired planner interpretation |
|---|---|
| `pointwise_oc_tile` | K-window or Y/K-window descriptors, not naive OC slicing. |
| `pointwise_y_tile_hardcoded` | Old-strategy name should not survive. c144/c192-style pointwise-wide `NONE` needs a channel-schedule proof, not a revived hardcoded Y strategy. |
| `spatial_oc_serial` | Y/K descriptors; independent submits are not the RKNN model. |
| `depthwise_spatial_tiled` | Y/K descriptors with depthwise channel granularity. |
| `spatial_im2col` | Temporary fallback; RKNN/ONNC evidence supports direct spatial Y descriptors eventually. |
| `grouped_serial` | Per-group lowering plus the same planner inside each lowered group. |

## Register Interpretation: `CNA_CONV_CON2`

Verified sources:

| Source | Finding |
|---|---|
| `experimental/rk3588_trm.md`, `RKNN_cna_conv_con2` | Offset `0x1010`; bits `23:16` are named `kernel_group`, bits `13:4` are `feature_grains`, bits `2:0` are CSC/debug controls. |
| `experimental/rockchip.py` | Same masks: `KERNEL_GROUP=0x00ff0000 >> 16`, `FEATURE_GRAINS=0x00003ff0 >> 4`, `CSC_WO_EN=bit2`, `CSC_DO_EN=bit1`, `CMD_FIFO_SRST=bit0`. |
| DeepWiki `torvalds/linux` / `drivers/accel/rocket` | Upstream generated `rocket_registers.h` uses the same `CNA_CONV_CON2_KERNEL_GROUP` and `CNA_CONV_CON2_FEATURE_GRAINS` masks/shifts. |
| DeepWiki `nvdla/hw` and `nvdla/sw` | NVDLA `CDMA_D_FETCH_GRAIN_0.GRAINS` is only bits `11:0`; KMD writes `conv_op->fetch_grain - 1`. NVDLA does not have the RK3588 high-byte `kernel_group`/family field in this register. |

Corrected local interpretation:

- The previous idea that low bits `3:0` encode descriptor family was wrong. Those
  bits are CSC scan enables, a reserved bit, and command FIFO soft reset.
- In local planner/emitter terms, `grain_bits` really means the shifted
  `feature_grains` field, bits `13:4`. Keep using the name only as a historical
  field label until the code is renamed; do not infer family from it.
- Observed RKNN descriptor-family values are carried in the high byte, bits
  `23:16`, even though the TRM names the field `kernel_group`.
- Treat the high byte as `family_bits` for the current emitter because observed
  RKNN rows use values that do not match the TRM's literal kernel-group formula
  for our tested FP16 CONV rows.

Observed values that are safe to use as evidence:

| Observed `CNA_CONV_CON2` | Family label | High byte / local `family_bits` | `feature_grains` field |
|---|---|---:|---:|
| `0x00000110` | `setup` | `0x00` | `0x011` |
| `0x20000090` | `y_tile` | `0x20` | `0x009` |
| `0x40000110` | `k_half` | `0x40` | `0x011` |
| `0x500000f0` | `k_tile` | `0x50` | `0x00f` |
| `0x00000320` | `setup` trace | `0x00` | `0x032` |

Working hypothesis for `family_bits`:

| High-byte bit | Observed meaning |
|---|---|
| `0x20` | Y split / `y_tile` family. |
| `0x40` | K split family. |
| `0x10` | Fine K tile discriminator, distinguishing `k_tile` from `k_half`. |

Do not overclaim this as a final hardware spec. The verified fact is the bitfield
placement; the family meaning is RKNN-observed behavior. Any future generic
emitter change must preserve this separation: feature grains come from bits
`13:4`, descriptor family comes from bits `23:16`, and low bits remain CSC/debug
controls unless a targeted hardware test proves otherwise.

## Review: Are We On Track?

Yes, but only if the project stops expanding the no-submit proof layer.

Good progress:

- The six old strategy names have been reduced to four planner split methods.
- The planner has broad shape coverage and CPU proof coverage.
- The full-register closure proved why sparse dynamic CONV emission was unsafe.
- A planner-driven hardware CONV now passed, so the remaining work is concrete
  emitter and submit debugging.

Main risk:

- The current roadmap still reads like it can accumulate unlimited small safety
  gates. That is how a `<1000` line target becomes another 2k-line runtime.

Correct operating rule:

- Every next action must either make a new shape pass on hardware, replace/delete
  existing runtime code, or permanently fence an unsupported family from the
  final `conv.py` interface. Anything else goes to the backlog.
- After replacement, the success metric is no longer "create `conv.py`". It is
  "increase supported shape classes while keeping `examples/conv.py` plus the
  planner under the line budget".

Sequencing decision:

- Do **not** debug and fix every disabled/timeout family before moving to the next
  milestone. That recreates the old eternal cleanup loop.
- Do fix a disabled family immediately only when it unlocks a broad class with a
  bounded change and safe hardware test.
- Do park a disabled family when the last attempt already disproved the current
  hypothesis, especially if another submit may timeout.
- A milestone can advance with explicit fail-closed behavior. A disabled path is
  acceptable if it rejects before allocation, has a named missing register/submit
  group, and does not hide behind CPU fallback.

Disabled-family triage:

| Family | Current state | Block next milestone? | Next action |
|---|---|---|---|
| generic `NONE/setup` | c32 closure and one small direct-spatial shape pass; c32 h7 and c32 oc64 still fail | no, unless the next batch targets NONE quality | make one clustered register-group fix only if the failures share a cause |
| pointwise-wide `NONE` | c144 RKNN capture shows a multi-descriptor channel schedule; local one-row setup still fails | no | keep fail-closed unless the runtime can emit the channel schedule generically without a captured table |
| pointwise `BY_Y/y_tile` | c64/c96/c112/c128/c144/c160/c192 tested variants pass, including non-32-aligned c144 and four-row c192 | no, already broad enough for now | keep enabled; add no more status labels unless a generic parser/emitter change needs coverage |
| non-pointwise `BY_Y/y_tile` | exact h160 RKNN capture passed; useful prefix is mined after dropping PPU/tail rows, but planner windows and setup/k_half row generation are unresolved | no | keep offline until `55/55/54` windows and useful prefix rows are generated from descriptors |
| `BY_K/k_tile` | exact-11 topology has generated row content, auxiliary rows, task metadata, PC tails, fake-DMA submit fields, genericity audit, and hardware-run hypothesis | no, until runtime adapter and BO-size dry-run exist | sketch exact-11 runtime boundary while preserving fail-closed behavior, then audit BO sizes before hardware |
| `BY_YK` / h40 / `k_half` | mixed-family semantics depend on unresolved K-family task segmentation and setup/k_half reuse | no | keep fenced until BY_K and non-pointwise BY_Y are independently executable |

Milestone review snapshot as of 2026-05-31, after Milestone 41:

| Milestone | Review | Runtime decision | Next action |
|---|---|---|---|
| 1 `NONE/setup` | c32 setup passes; c144 pointwise-wide does not. | Keep c32 and small generic NONE; fence pointwise-wide NONE. | Revisit only with generated multi-descriptor evidence. |
| 2 `BY_K/k_tile` | Body rows matched, but h14 submit timed out. | Fence BY_K. | Historical timeout; superseded by Milestones 11-26. |
| 3 `BY_Y/y_tile` | Initial decision was fail-closed before Y overlay existed. | Superseded by Milestones 6-9. | Keep as historical exit-rule evidence only. |
| 4 replacement | `examples/conv.py` became the planner-driven entrypoint. | Complete. | Do not re-import proof scaffolds. |
| 5 coverage under budget | Parser, status-aware list, small generic NONE, and first BY_Y path landed. | Keep encoded parser; no big shape table. | Grow only generic code or tests. |
| 6 first BY_Y | c96 pointwise BY_Y passed with two rows. | Support narrow pointwise BY_Y. | Extend only through the same overlay. |
| 7 budget recovery | Runtime shrank from 949 to 908 combined lines. | Complete. | Recover lines before any large family addition. |
| 8 BY_Y batch | c64/c96/c128 pointwise BY_Y variants passed. | Batch-supported pointwise BY_Y. | No more labels unless guarding a generic change. |
| 9 c144 BY_Y | RKNN capture fixed unpadded pointwise-wide packing for BY_Y. | c144 BY_Y supported; pointwise-wide NONE still fenced. | Preserve separate BY_Y versus NONE behavior. |
| 10 pointwise-wide NONE | RKNN shows multi-descriptor channel schedule; local one-row setup fails. | Fence pointwise-wide NONE. | Only implement if channel schedule can be generated compactly. |
| 11 BY_K revisit | RKNN capture shows local submit shape is wrong. | Fence BY_K. | Full task segmentation required. |
| 12 BY_YK | Prerequisites are not met. | Fence BY_YK. | Wait for BY_K and non-pointwise BY_Y. |
| 13 non-pointwise BY_Y | Row regs match legacy; output mapping and tile-local submit failed. | Fence non-pointwise BY_Y. | Superseded by Milestone 16 mapping and Milestone 19 topology. |
| 14 BY_K task fixture | Proof-only checker validates RKNN 11-task prefix mismatch. | Keep fixture out of runtime. | Superseded by generated prefix proof. |
| 15 budget guardrail | Active runtime is now `489 + 411 = 900` lines after compaction. | No deletion required now. | Spend the 50-line margin only on generated code, not fixtures/tables. |
| 16 pointwise BY_Y + BO map | c112/c160/c192 passed; h160 output maps with non-contiguous BO but submit timed out. | Keep pointwise BY_Y enabled and non-pointwise BY_Y fenced. | Use BO fix; solve h160 schedule offline. |
| 17 spatial BY_Y layout | Local h160 is `3 x 47` sparse y_tile rows, unlike direct-spatial prelude/closure. | Keep non-pointwise BY_Y fenced. | Superseded by exact h160 RKNN topology. |
| 18 compact prelude/closure | Six-task 62-qword topology is expressible, but row values are formula-derived. | Keep non-pointwise BY_Y fenced. | Rejected by Milestone 19 exact topology. |
| 19 exact h160 RKNN topology | RKNN uses 18 groups and `55/55/54` y_tile windows. | Keep non-pointwise BY_Y fenced. | Generate `55/55/54` windows and useful prefix before hardware. |
| 20 BY_K prefix equivalence | Checker validates RKNN deltas and 552-qword prefix backing. | Keep BY_K fenced. | Superseded by generated prefix proof. |
| 21 auxiliary schedule model | BY_K prefix is generated from planner descriptors; h160 useful prefix is mined. | Keep BY_K and non-pointwise BY_Y fenced. | Need row-content generation and budget gate. |
| 22 BY_K runtime gate | Estimated BY_K diff exceeded then-current soft margin and aux rows were absent. | Keep BY_K fenced. | Recover budget before BY_K runtime work. |
| 23 budget recovery for BY_K | Runtime compacted to `489 + 411 = 900`; supported smokes still pass. | Keep BY_K fenced. | Build no-submit aux/setup/k_half row-content generation. |
| 24 BY_K row-content gate | k_tile body rows materialize with fixture parity; aux rows remain fenced. | Keep BY_K fenced. | Generate aux/setup/k_half row contents before hardware. |
| 25 BY_K k_half candidate | Compact k_half candidate produces `(62,62)` but RKNN needs `(62,57)`. | Keep BY_K fenced. | Need shorter second k_half closure rule. |
| 26 BY_K artifact comparability | Older prefix counts/deltas/backing still match, but preserved emit rows and decoded regdump are not raw task-relative qwords. | Keep BY_K fenced. | Superseded by live raw topology work. |
| 27 live raw comparability | Recovered live h14 raw window is internally comparable but not the older prefix. | Keep BY_K fenced. | Analyze live topology separately. |
| 28 live offset model | Live setup/k_half/k_tile order and DMA offsets are compactly describable from shape/planner data. | Keep BY_K fenced. | Diff live body row content against active closure. |
| 29 live body closure diff | Active closure is a large subset of live rows but has stable extras/diffs. | Keep BY_K fenced. | Classify named register deltas. |
| 30 normalized body diff | Named residual is 45 live-extra, 3 generated-only, 6 value diffs. | Keep BY_K fenced. | Generate closure extras and value rules. |
| 31 body closure materializer | 45 live-extra regs are generated; only six value diffs remain. | Keep BY_K fenced. | Derive the six compact value rules. |
| 32 body value rules | Live body rows match exactly no-submit after compact value rules. | Keep BY_K fenced. | Decode/generate auxiliary rows. |
| 33 auxiliary decoder | Repeated 26-qword `0x60` rows are named and classified. | Keep BY_K fenced. | Materialize aux rows from shape/address parameters. |
| 34 auxiliary materializer | Live aux row content matches no-submit generation. | Keep BY_K fenced. | Generate task metadata. |
| 35 task metadata | Amounts, masks, op_idx, offsets, deltas, and tail budgets match live prefix. | Keep BY_K fenced. | Classify PC-tail qwords. |
| 36 PC-tail classification | All nonterminal tail qwords are PC metadata or zero padding; no unknown nonzero tails remain. | Keep BY_K fenced. | Compare against active runtime submit behavior. |
| 37 runtime gate hypothesis | Body, aux, metadata, and tails are ready; runtime deltas are named: larger regcmd BO, aux masks, PC-tail semantics, core lanes/subcore submit. | Keep BY_K fenced. | Decide scoped implementation versus continued fence; if implementing, add a no-submit task/regcmd emitter sketch first. |
| 38 runtime implementation decision | Decision is `continue_fence`; required next evidence is fake-DMA task/regcmd emission, PC-tail emitter, genericity audit, and written health plan. | Keep BY_K fenced. | Build submit-field emitter sketch no-submit; do not edit runtime submit fields yet. |
| 39 fake-DMA submit-field emitter | Fake task structs and regcmd tail qwords match the decoded live h14 topology no-submit. | Keep BY_K fenced. | Audit genericity beyond h14 or mark h14-only. |
| 40 runtime genericity audit | Exact-11 signature appears in h7, h14, c32 h14, and pw c256 h14 captures; 14/17-row variants stay fenced. | Keep BY_K fenced. | Write exact-11 hardware-run hypothesis. |
| 41 exact-11 hardware-run hypothesis | Field checklist and simple_add pre/post command sequence are written; runtime still lacks adapter boundary and BO dry-run. | Keep BY_K fenced. | Sketch exact-11 runtime adapter with fail-closed guard, then audit BO sizes no-submit. |

## Next Actions

### Stop Doing

- Do not add more narrative-only markdown evidence reports before the next hardware
  fix. A proof-only checker is acceptable only if it directly gates the next safe
  submit or replaces runtime code.
- Do not expand `examples/conv_no_submit_materializer.py` unless the new fields
  are immediately consumed by `examples/conv.py` for a hardware attempt.
- Do not add another fixture unless it is the oracle for the current failing
  shape and stays proof-only until it replaces runtime code.
- Do not chase h40 `setup`/`k_half`, multicore, fusion, or broad VP sweeps before
  non-pointwise BY_Y has a safe output/submit model and BY_K has a
  prefix-equivalent no-submit schedule.
- Do not keep both `conv_small_no_submit.py` and `conv_tiles_no_submit.py` in the
  mental delivery path. They are fixture/proof tools only.
- Do not grow `SHAPES` by adding dozens of literal dictionaries. A larger shape
  registry will quietly spend the remaining line budget without improving the
  emitter. Prefer a compact shape-name parser plus a tiny alias table.
- Do not block cheap pointwise BY_Y coverage on parked failures. Pointwise-wide
  NONE, non-pointwise BY_Y submit, BY_K timeout, and BY_YK/h40 are parked until
  their next action is a concrete output/submit diff or generic emitter change,
  not another blind probe.
- Do not keep expanding pointwise BY_Y just to add more passed examples. The current
  batch already proves the generic overlay across two-, three-, and four-row
  schedules; further pointwise work should guard code changes, not spend runtime or
  documentation budget.
- Do not rerun h160 locally with either the old `3 x 47` y_tile chain or the compact
  six-task closure. Exact RKNN h160 topology disproved both hypotheses.
- Do not assume every RKNN h160 row is required. PPU and trailing rows may be
  compiler/runtime boilerplate for features we are not using; mine the useful prefix
  before deciding what belongs in `examples/conv.py`.
- Do not rerun h14 BY_K with the old `3 x 9` body rows. Milestone 20 proves the
  missing object is the auxiliary 552-qword prefix schedule.
- Do not treat `gem2_regdump.bin` from the preserved h14 capture as raw regcmd
  backing. It is compact decoded `<hIh>` records from `dump.py`; it cannot identify
  the five missing qwords in the second k_half body without an offset mapper.
- Do not compare h14 emit grouped lengths directly to the GEM1 task prefix. The
  current emit grouping `(100,47,15,47,15,47,15,47,15,47,15,11)` is not the same
  object as the task prefix `(69,108,25,62,11,57,9,56,9,56,9)`.
- Do not keep extending BY_K live-topology proof reports after Milestone 41 unless
  the next checker directly decides runtime implementation or produces generated
  task/regcmd objects. Body rows, auxiliary rows, task metadata, and PC tails are
  already decoded no-submit.

### Historical Milestone Log

Milestones 1-38 below are retained for auditability. They are not the active task
list. Future updates should prefer changing `Active Operating Plan`, `Current
State`, and the latest milestone only.

### Milestone 1: Fix Generic `NONE/setup`

Deliverable: one planner-driven submit path that passes setup/NONE shapes beyond
the single c32 setup fixture.

Work allowed inside this milestone:

- Use the c144 pointwise mismatch as the primary oracle.
- Compare only the emitted full-register row against `examples/conv.py`, the c32
  108-qword closure, and mined RKNN fields where available.
- Derive the missing pointwise base/reset differences in the emitter. Do not add
  a new strategy branch for `pointwise_y_tile_hardcoded`.
- Add c192 only if it shares the same fixed emitter path; otherwise leave it for
  the next NONE batch.

Acceptance:

```sh
python3 examples/simple_add.py
python3 examples/conv.py b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid
python3 examples/conv.py conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1
python3 examples/simple_add.py
```

Exit rule:

- If c144 still mismatches after one focused emitter fix, record the exact
  differing register groups in this plan and choose either `FP16-only NONE` or
  `pointwise NONE disabled` as the temporary final behavior. Do not create three
  more reports.

Milestone 1 execution result on 2026-05-31:

| Check | Result |
|---|---|
| pre-health `simple_add.py` | PASS |
| c32 setup/NONE after final guard | PASS, `regs=108`, `hw_out=fp16`, max_diff `0.0304` |
| post-health `simple_add.py` | PASS |
| c144 legacy sparse oracle | FAIL, max_diff `96.7248` |
| c144 planner forced FP16 | FAIL, max_diff `105.0142` |
| c144 planner after focused emitter fix | FAIL, max_diff `96.7248` |
| c144 final behavior | rejected before allocation: `pointwise-wide NONE is disabled after c144 mismatch; no allocation or submit attempted` |

Focused emitter fix attempted:

- Pointwise-wide weights are packed with 32-channel input alignment. For c144 this
  means 160 packed input channels, but the previous emitter described 144 channels
  in `CNA_WEIGHT_SIZE0/1`.
- A packed-weight-size variant emitted `CNA_WEIGHT_SIZE0=0x2800` and
  `CNA_WEIGHT_SIZE1=0x140` for c144, matching the padded weight buffer.
- This did not change hardware output, so the runtime kept the legacy-compatible
  data-aligned size fields and fenced the unsafe c144 path.

Temporary behavior chosen by the exit rule:

- `pointwise-wide NONE` is disabled before allocation in the planner prototype.
- Do not spend more hardware attempts on c144 until there is a new targeted reason
  in pointwise `CNA_CONV_CON1/2`, `CNA_CBUF_CON0/1`, DMA/FC sizing, or a mined
  RKNN row for the same pointwise-wide family.
- Milestone 2 is the next active milestone.

### Milestone 2: Recover `BY_K/k_tile`

Deliverable: h14 and h7 `k_tile` pass through the planner path, or `BY_K` is
explicitly fenced from the first shipped `conv.py`.

Work allowed inside this milestone:

- Keep `BY_K/k_tile` disabled until the submitted row changes from the timed-out
  version. Do not reintroduce a public override flag until there is a concrete
  submit-layout delta to test.
- Fix one of these concrete suspects before the next submit: PC row chaining,
  `regcfg_amount` versus row length/tail, full-register row reset state,
  `CNA_CONV_CON2`, `CNA_CBUF_CON0/1`, DPU channel/surface fields.
- Use the existing h14/h7 fixture rows as the only oracle. Do not mine a new
  model unless h14/h7 evidence is internally contradictory.

Acceptance:

```sh
python3 examples/simple_add.py
python3 examples/conv.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/simple_add.py
python3 examples/conv.py b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/simple_add.py
```

Safety condition:

- Do not rerun the old h14 `k_tile` submit unchanged. The previous version timed
  out once. The next run must have a targeted register/task change.

Milestone 2 execution result on 2026-05-31:

| Check | Result |
|---|---|
| h14 register body versus existing oracle | PASS for all 16 observed fields across rows 6/8/10 |
| targeted task fix | encoded `pc_core=2` in k_tile PC-chain tails and used the direct-spatial terminal tail |
| pre-health `simple_add.py` | PASS |
| h14 submit after PC-core fix | TIMEOUT, `Errno 110` |
| post-timeout `simple_add.py` | PASS |
| h7 submit | not run because h14 failed the acceptance path |
| final BY_K behavior | rejected before allocation: `BY_K/k_tile is disabled after h14 timed out with body-parity and PC-core fixes; no allocation or submit attempted` |

Temporary behavior chosen:

- `BY_K/k_tile` is disabled in the first shipped planner runtime.
- Do not rerun h14/h7 k_tile until there is a new targeted reason beyond the
  observed body-register parity and PC-core tail fix.

### Milestone 3: Decide `BY_Y/y_tile`

Deliverable: one direct spatial or pointwise `BY_Y` shape submits correctly, or
`BY_Y` is intentionally unsupported in the first shipped `conv.py`.

Candidate shapes:

| Shape | Why it is useful |
|---|---|
| `conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1` | Pointwise BY_Y with ONNC/VP split evidence. |
| `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` | Spatial BY_Y without h40 mixed-family pressure. |
| `b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid` | Replaces old `spatial_im2col`, but only after a smaller BY_Y works. |

Work allowed inside this milestone:

- Add the Y-window overlay fields needed for real hardware behavior:
  `CNA_FEATURE_DATA_ADDR`, `CNA_DATA_SIZE0`, `CNA_DATA_SIZE3`, `CORE_SIZE0`,
  `DPU_DST_BASE_ADDR`, `DPU_DST_H`, and `DPU_SIZE`/width-height equivalents.
- Reuse the setup/NONE base emitter. Do not create a separate y-tile runtime.
- Use mined c32 y_tile rows only as register-field guidance because they are
  observed-only and not currently planner-equivalent.

Acceptance:

```sh
python3 examples/simple_add.py
python3 examples/conv.py <one-BY_Y-shape>
python3 examples/simple_add.py
```

Exit rule:

- If one BY_Y shape cannot be made safe after a single focused emitter pass, ship
  the first `conv.py` with `BY_Y` rejected before allocation. That is better than
  preserving `spatial_im2col` or reviving a six-strategy runtime.

Milestone 3 execution result on 2026-05-31:

| Check | Result |
|---|---|
| candidate added | `conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1` |
| planner classification | `BY_Y`, `y_tile`, 2 descriptor rows |
| final BY_Y behavior | rejected before allocation: `BY_Y/y_tile is disabled before allocation; Y-window hardware fields are not solved in the first planner runtime` |

Temporary behavior chosen:

- `BY_Y/y_tile` is disabled in the first shipped planner runtime.
- This intentionally avoids preserving `spatial_im2col` or adding another
  strategy runtime before Y-window hardware fields are solved.

### Milestone 4: Replace `examples/conv.py`

Deliverable: the normal user command is planner-driven.

Required before replacement:

- Milestone 1 either passes for generic NONE or unsupported pointwise-wide NONE
  fails closed before allocation.
- Milestone 2 either passes or `BY_K` is explicitly rejected before allocation.
- Milestone 3 either passes or `BY_Y` is explicitly rejected before allocation.
- Unsupported `BY_YK`, h40 `setup`, and `k_half` fail closed with clear messages.

Replacement rule:

- Move the legacy file to `examples/conv_legacy.py` only when the new file can be
  the normal entry point.
- The final `examples/conv.py` may import `conv_expt/conv_tile_planner.py`, but it
  must not import `conv_tile_cpu.py`, `conv_h14_k_tile_no_submit.py`, or broad
  no-submit report code.
- If `examples/conv_no_submit_closure.py` remains imported, its lines count
  against the runtime budget. Prefer folding only the necessary reset defaults
  into the emitter or generating them from compact tables.

Acceptance:

```sh
python3 examples/conv.py --list
python3 examples/simple_add.py
python3 examples/conv.py b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid
python3 examples/conv.py conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1
python3 examples/simple_add.py
wc -l examples/conv.py conv_expt/conv_tile_planner.py
```

Milestone 4 execution result on 2026-05-31:

| Check | Result |
|---|---|
| legacy file | moved to `examples/conv_legacy.py` |
| normal entrypoint | `examples/conv.py` is planner-driven |
| forbidden imports | no import of `conv_tile_cpu.py`, `conv_h14_k_tile_no_submit.py`, `conv_tiles_no_submit.py`, or `conv_no_submit_closure.py` |
| runtime line count | `523 examples/conv.py` + `410 conv_expt/conv_tile_planner.py` = 933 lines |
| `python3 examples/conv.py --list` | PASS |
| pre-health `simple_add.py` | PASS |
| `python3 examples/conv.py b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid` | PASS, max_diff `0.0304` |
| c144 pointwise-wide command | fail-closed before allocation |
| h14 BY_K command | fail-closed before allocation |
| c96 BY_Y command | fail-closed before allocation |
| post-health `simple_add.py` | PASS |

Line-budget rule:

- Hard cap: user-facing `examples/conv.py` stays under 1000 lines.
- Soft cap: `examples/conv.py` plus planner import should stay under 950 lines if
  proof-only helpers are still present elsewhere.
- Any new helper imported by `examples/conv.py` must delete at least the same
  number of lines from the runtime path.

### Milestone 5: Coverage Expansion Under Budget

Status: active. Replacement is done, so coverage expansion can start. The rule is
batch progress only: each action should enable or permanently fence a whole class
of shapes, not one new special case.

Line budget for this milestone:

| Item | Current | Budget decision |
|---|---:|---|
| `examples/conv.py` | 503 | hard cap stays `<1000`; practical cap is `<590` while the planner remains 410 lines |
| `conv_expt/conv_tile_planner.py` | 410 | freeze unless planner bug blocks a batch |
| active combined runtime | 913 | every new helper/table must pay for itself by deleting or replacing code |
| remaining safe slack | about 70 lines | use for generic parsing/emitter code, not shape dictionaries |

Next actions, in order:

| Batch | Deliverable | Why this moves toward `<1000` |
|---|---|---|
| Shape input contract | Replace the fixed `SHAPES` growth path with compact parsing for encoded names plus a tiny alias table for non-encoded names. | Prevents the 217-shape suite from becoming hundreds of dictionary lines. |
| NONE batch | Try 8-12 non-pointwise-wide `NONE/setup` candidates through the existing one-task path. | Tests whether current generic emitter is useful beyond the c32 oracle without adding family code. |
| Pointwise-wide decision | Keep c144/c192 disabled unless a new RKNN/Mesa row explains the mismatch; do not spend hardware attempts on blind register tweaks. | Avoids an infinite small-step loop around one failed family. |
| BY_Y first tile | Implement one Y-window overlay path and test one pointwise BY_Y candidate. | Enables 32 planner rows if solved, and avoids keeping old `spatial_im2col`. |
| BY_K root cause | Do not rerun h14 unchanged; first compare full submitted task rows against captured RKNN task rows including PC tail and `regcfg_amount`. | Moves the timeout from speculation to one concrete submit-layout delta. |
| BY_YK/h40 | Keep fenced until `BY_Y` and `BY_K` work independently. | Prevents mixed-family complexity from entering the small runtime too early. |

Updated order after the first Milestone 5 batch result:

1. Keep the shape parser and status-aware `--list`; do not spend more lines on
   shape dictionaries.
2. Continue generic `NONE/setup` for one clustered failure group only. The current
   c32 h7 / c32 oc64 failures are useful if they share a register group; otherwise
   park them like c144.
3. Move to `BY_Y/y_tile` before revisiting the BY_K timeout. BY_Y has known
   missing overlay fields and no timeout history, so it is a better broad unlock.
4. For BY_K, do only the offline full-task row diff until it identifies a changed
   submit-layout hypothesis. No hardware rerun until then.
5. Keep pointwise-wide NONE and BY_YK/h40 fail-closed. They should not block the
   next milestone.

Recommended NONE candidates for the first batch:

| Shape | Old branch | Reason |
|---|---|---|
| `b1_c32_h14_w14_oc64_wic32_k3x3_g1_s1_pvalid` | `spatial_oc_serial` | Same family as the passing c32 setup shape, smaller output channels. |
| `b1_c32_h7_w7_oc128_wic32_k3x3_g1_s1_pvalid` | `spatial_oc_serial` | Same channels as passing c32 setup, smaller spatial size. |
| `conv2d_b1_c16_h18_w18_oc16_wic16_k3x3_g1` | `spatial_oc_serial` | Small batch=1 groups=1 spatial case. |
| `conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1` | `fallback/direct` | Very small direct spatial smoke test. |
| `conv2d_b1_c1_h5_w7_oc6_wic1_k3x3_g1` | `fallback/direct` | Exercises `in_c=1` packing without tiling. |

NONE batch acceptance:

```sh
python3 examples/simple_add.py
python3 examples/conv.py <candidate-none-shape>
python3 examples/simple_add.py
```

Rules for the NONE batch:

- Add at most a tiny alias entry for a tested candidate unless a generic parser is
  already present.
- If a candidate needs a per-shape register constant, reject that candidate and
  record the register group. Do not add a special case.
- If two candidates fail for the same register group, make one generic emitter fix
  and rerun the batch.

Recommended BY_Y candidate:

| Shape | Why |
|---|---|
| `conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1` | Already in `examples/conv.py`, planner emits two `y_tile` rows, and ONNC/VP evidence says pointwise BY_Y is compiler-like. |

BY_Y acceptance after a real Y-window overlay is implemented:

```sh
python3 examples/simple_add.py
python3 examples/conv.py conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1
python3 examples/simple_add.py
```

BY_K acceptance only after a submit-row delta is identified:

```sh
python3 examples/simple_add.py
python3 examples/conv.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/simple_add.py
```

Milestone 5 is complete when:

- `examples/conv.py --list` does not imply that disabled shapes are supported;
- at least one new non-c32 `NONE/setup` shape passes;
- either one BY_Y shape passes or BY_Y remains explicitly fail-closed with the
  exact missing register groups listed;
- h14 BY_K has a named new submit-layout hypothesis before any further hardware
  run;
- active runtime remains under the line budget.

Milestone 5 execution result on 2026-05-31:

| Check | Result |
|---|---|
| shape input contract | added compact encoded-name parser for `[conv2d_]bN_cN_hN_wN_ocN_wicN_kHxW_gN[_sN][_pvalid]` |
| `--list` behavior | status-aware list: supported, try-NONE, disabled |
| line count after parser/emitter fixes | `531 examples/conv.py` + `410 conv_expt/conv_tile_planner.py` = 941 lines |
| c32 known support after changes | PASS, max_diff `0.0304`, health PASS before/after |
| first c32 h7 NONE attempt | FAIL, max_diff `102.0137`, health PASS after |
| c32 h14 oc64 NONE attempt | FAIL, max_diff `123.3547`, health PASS after |
| first small direct-spatial NONE attempt | FAIL, max_diff `inf`, health PASS after |
| legacy oracle for same small shape | PASS, max_diff `0.0074` |
| focused generic emitter fixes | sparse generic setup rows; NHWC `CNA_CONV_CON1` bits; NHWC `CNA_CVT_CON5` mask; spatial `CNA_CVT_CON0=0x0b` |
| new non-c32 NONE pass | `conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1`, `regs=47`, max_diff `0.0074`, health PASS before/after |
| pointwise-wide NONE | still fail-closed before allocation |
| BY_Y | c96 pointwise BY_Y now passes with two y_tile tasks |
| BY_K | still fail-closed; next allowed work is captured full-task row diff including PC tail and `regcfg_amount`, not another unchanged submit |

Milestone 5 current status:

- Shape-name parsing is solved without growing a dictionary per shape.
- The runtime now supports two `NONE/setup` examples: the c32 108-reg closure path
  and one generic sparse direct-spatial path.
- The c32 h7 and c32 oc64 failures show that generic spatial setup is not solved
  for all c32 variants; do not add per-shape constants for them.
- BY_K remains intentionally fenced until its listed missing submit-layout delta is
  identified. BY_Y pointwise row tiling is now supported for the c96 candidate.

Milestone 6 execution result on 2026-05-31:

| Check | Result |
|---|---|
| target | `conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1` |
| planner rows | `BY_Y`, `y_tile`, 2 rows |
| implementation | generic pointwise BY_Y row overlay using planner row windows, legacy-compatible packed input offsets, full output stride, output row offsets, and weight reuse CBUF bit |
| offline row-reg check | generated BY_Y row regs matched `examples/conv_legacy.py` row-tiling regs exactly for both rows |
| pre-health `simple_add.py` | PASS |
| hardware submit | PASS, `tasks=2`, `regs=47;47`, `hw_out=fp16`, max_diff `0.0156` |
| post-health `simple_add.py` | PASS |
| final line count | `539 examples/conv.py` + `410 conv_expt/conv_tile_planner.py` = 949 lines |
| pointwise-wide NONE | remains fail-closed before allocation |
| BY_K | remains fail-closed before allocation |

Milestone 6 status:

- BY_Y is no longer globally unsupported: pointwise BY_Y row tiling has one passing
  hardware path.
- The implementation is intentionally narrow: it accepts `BY_Y/y_tile` only for
  pointwise `groups=1`, `k1x1` shapes. Other BY_Y families must remain fenced
  until their spatial/depthwise overlays are proved.
- Runtime is still under the soft 950-line combined budget, but only by one line.
  The next feature must delete or compact code first.

### Milestone 7: Recover Budget, Then Expand Coverage

Milestone 7 execution result on 2026-05-31:

| Check | Result |
|---|---|
| budget recovery | compacted register constant declarations without changing values |
| final line count | `498 examples/conv.py` + `410 conv_expt/conv_tile_planner.py` = 908 lines |
| pre-health `simple_add.py` | PASS |
| c32 closure NONE | PASS, max_diff `0.0304` |
| small generic NONE | PASS, max_diff `0.0074` |
| first pointwise BY_Y | PASS, max_diff `0.0156` |
| post-health `simple_add.py` | PASS |
| safe follow-up expansion | `conv2d_b1_c96_h56_w56_oc32_wic96_k1x1_g1` uses the same two-row BY_Y overlay and passed, max_diff `0.0156`, health PASS before/after |
| disabled families | pointwise-wide NONE, BY_K/k_tile, non-pointwise BY_Y, and BY_YK remain fail-closed before allocation |

Milestone 7 status:

- The active runtime recovered 41 lines versus the Milestone 6 state and remains
  well under the soft 950-line budget.
- Pointwise BY_Y now has two passing shapes with the same generic row overlay.
- No BY_K hardware rerun was attempted; this follows the timeout safety rule.

### Milestone 8: Batch Pointwise BY_Y, Keep Hard Fences

Milestone 8 execution result on 2026-05-31:

| Check | Result |
|---|---|
| candidate classification | `conv2d_b1_c96_h56_w56_oc16_wic96_k1x1_g1`, `conv2d_b1_c64_h56_w56_oc24_wic64_k1x1_g1`, and `conv2d_b1_c128_h56_w56_oc24_wic128_k1x1_g1` all classify as pointwise `BY_Y/y_tile` |
| c96 oc16 | PASS, two rows, max_diff `0.0156`, health PASS before/after |
| c64 oc24 | PASS, two rows, max_diff `0.0155`, health PASS before/after |
| c128 oc24 | PASS, three rows, max_diff `0.0233`, health PASS before/after |
| status list | added only encoded status-label examples; no shape dictionary added |
| line count | `501 examples/conv.py` + `410 conv_expt/conv_tile_planner.py` = 911 lines |
| still fenced | pointwise-wide NONE, BY_K/k_tile, non-pointwise BY_Y, and BY_YK remain rejected before allocation |

Milestone 8 status:

- Pointwise BY_Y is now a batch-supported class for the tested c64/c96/c128
  variants, including both two-row and three-row schedules.
- The generic row overlay remained unchanged; no per-shape register constants were
  added.
- BY_K was not rerun. Its next action remains offline full-task row diff only.

Milestone 9 selection rationale:

- After the c64/c96/c128 pointwise BY_Y batch, the highest-value bounded probe was
  c144 pointwise BY_Y because it stressed non-32-aligned input channels without
  entering pointwise-wide `NONE` or BY_K submit risk.
- Pointwise-wide `NONE` and BY_K stayed fenced during this step.

Milestone 9 execution result on 2026-05-31, updated after exact RKNN capture:

| Check | Result |
|---|---|
| harder pointwise BY_Y target | `conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1`, three y_tile rows |
| first submit | FAIL, max_diff `85.5101`, health PASS before/after |
| offline diff | input packing and weight packing matched legacy; row regs differed only in `CNA_WEIGHT_SIZE0/1` |
| generic register fix tried | reverted pointwise-wide size registers to legacy-compatible data-aligned size fields for emitted regs |
| rerun after fix | FAIL, max_diff `85.5101`, health PASS before/after |
| RKNN model generated | `/home/orangepi/npu/ops_rknn/models/conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1.rknn` |
| RKNN capture command | `sh run.sh --gdb conv2d_multi --case conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1 1 144 56 56 24 1 1 1 c144_BY_Y` |
| RKNN capture health | `simple_add.py` PASS before and after |
| RKNN runtime result | PASS, max error `0.0325851` |
| captured artifacts | `dump/conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1_emit.txt`, weight/input dumps, GEM1 tasks, GEM2 regdump |
| RKNN row finding | pointwise c144 uses unpadded 144-channel weight size, `CNA_WEIGHT_SIZE0=0x1b00`, `CNA_WEIGHT_SIZE1=288`, and row windows `22,22,12` |
| generic planner/emitter fix | pointwise-wide weights pack unpadded 32-channel input chunks; planner pointwise alignment stays 16 for CBUF row planning; descriptor feature offsets use the packed input stride |
| repaired c144 BY_Y submit | PASS, `tasks=3`, `regs=47;47;47`, `hw_out=fp16`, max_diff `0.0263`, health PASS before/after |
| passing BY_Y regression check | c96 oc24 still PASS after the fix, max_diff `0.0156`, health PASS after batch |
| line count | `502 examples/conv.py` + `411 conv_expt/conv_tile_planner.py` = 913 lines |

Milestone 9 status:

- Pointwise BY_Y is now supported for the tested c64/c96/c128/c144 variants.
- Non-32-aligned pointwise BY_Y is no longer fenced solely for c144: exact RKNN
  capture showed the missing planner row-height and unpadded weight-layout rules.
- The c144 BY_Y fix does not solve pointwise-wide `NONE`; that family still needs
  separate RKNN-mined schedule evidence and remains fenced by Milestone 10.

### Milestone 10: Pointwise-Wide `NONE`

Status: fenced. This family covers c144/c192-style pointwise shapes that the
planner classifies as `NONE/setup`, but c144 still mismatched after the packed
weight-size register fix.

Deliverable: pointwise-wide `NONE/setup` either passes for c144 and c192 through a
generic emitter change, or remains permanently rejected before allocation with a
named missing register group.

Work allowed:

| Step | Action | Rule |
|---|---|---|
| 1 | Mine or diff an RKNN/Mesa row for c144/c192 pointwise-wide `NONE`. | Must identify a new register group beyond `CNA_WEIGHT_SIZE0/1`. |
| 2 | Compare c144 against passing pointwise BY_Y rows and small generic NONE rows. | Use this only to isolate register groups, not to invent shape constants. |
| 3 | Make one generic pointwise-wide emitter fix if the diff is concrete. | No per-shape c144/c192 constants. |

Acceptance:

```sh
python3 examples/simple_add.py
python3 examples/conv.py conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1
python3 examples/conv.py conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1
python3 examples/simple_add.py
```

Exit rule:

- If the next diff does not identify a new register group, keep pointwise-wide
  `NONE` fail-closed.
- If c144 passes but c192 fails for a different group, support only the generic
  subfamily that passed and keep the rest fenced.
- Do not spend line budget on aliases or fixtures for this family unless the
  runtime emitter changes.

Milestone 10 execution result on 2026-05-31:

| Check | Result |
|---|---|
| RKNN model generated | `/home/orangepi/npu/ops_rknn/models/conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1.rknn` |
| RKNN capture command | `sh run.sh --gdb conv2d_multi --case conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1 1 144 28 28 32 1 1 1 c144_NONE` |
| RKNN capture health | `simple_add.py` PASS before and after |
| RKNN runtime result | PASS, max error `0.0257759` |
| captured artifacts | `dump/conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1_emit.txt`, weight/input dumps, GEM1 tasks, GEM2 regdump |
| RKNN row finding | pointwise-wide `NONE` is not a single local setup row; RKNN emits a multi-descriptor channel schedule with setup-like, k-family, and PC task rows |
| generic fix inherited from Milestone 9 | unpadded pointwise-wide weight packing, matching RKNN weight size `0x2400` and bytes-per-kernel `288` for c144 oc32 |
| local submit after inherited fix | FAIL, max_diff `93.5419`, health PASS after |
| concrete CBUF split probe | `CNA_CBUF_CON0=(5 << 4) | 7` based on RKNN c144 h28 capture |
| local submit after CBUF probe | FAIL, max_diff `93.5419`, health PASS after |
| final behavior | pointwise-wide `NONE` remains rejected before allocation |

Milestone 10 status:

- The RKNN-mined unpadded weight layout is real and is kept because it fixed c144
  BY_Y, but it is insufficient for pointwise-wide `NONE`.
- The named missing group is no longer just `CNA_WEIGHT_SIZE0/1`: RKNN compiles the
  c144 h28 pointwise-wide case as a multi-descriptor channel schedule with PC/task
  rows, while the local runtime still emits one sparse setup task.
- No c192 hardware attempt was run because c144 did not pass through the generic
  pointwise-wide emitter path.

### Milestone 11: `BY_K/k_tile`

Status: fenced. h14 `k_tile` timed out after body-register parity and a PC-core
tail fix, so the next action must be offline submit-layout work, not another
hardware run.

Deliverable: h14 and h7 `k_tile` pass through the planner path, or BY_K remains
parked with a named submit-layout blocker.

Work allowed before any submit:

| Step | Action | Must include |
|---|---|---|
| 1 | Build a full-task row diff against captured RKNN h14/h7 rows. | Decoded regs, PC tails, `regcfg_amount`, row offsets, terminal tail, `task_count`. |
| 2 | Compare submit structs and subcore assignment. | `core_mask`, `subcore_task`, `task_start`, `task_number`, and timeout assumptions. |
| 3 | Produce one changed submit-layout hypothesis. | The hypothesis must be written in this plan before hardware rerun. |

Acceptance, only after a changed submit-layout hypothesis exists:

```sh
python3 examples/simple_add.py
python3 examples/conv.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/simple_add.py
python3 examples/conv.py b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/simple_add.py
```

Exit rule:

- Do not rerun the old h14/h7 task unchanged.
- If h14 times out again after a changed hypothesis, keep BY_K fenced and record
  the exact changed fields and health-check result.
- Do not proceed to BY_YK using `k_tile` until both h14 and h7 pass or a narrower
  BY_K subfamily is explicitly defined.

Milestone 11 execution result on 2026-05-31:

| Check | Result |
|---|---|
| no-submit scaffold | `python3 examples/conv_h14_k_tile_no_submit.py` PASS; h14/h7/setup observed body-register parity remains PASS |
| RKNN model generated | `/home/orangepi/npu/ops_rknn/models/b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid.rknn` |
| RKNN capture command | `sh run.sh --gdb conv2d_multi --case b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid 1 160 14 14 320 3 3 1 h14_BY_K` |
| RKNN capture health | `simple_add.py` PASS before and after |
| RKNN runner status | RKNN/GDB process exited normally but `conv2d_multi` CPU comparison failed; register/task capture is still usable for submit-layout mining |
| captured artifacts | `dump/b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid_emit.txt`, weight/input dumps, GEM1 tasks, GEM2 regdump |
| body-register parity | still PASS for existing h14 k_tile rows 6/8/10: `CNA_CONV_CON2`, CBUF, weight size, DCOMP, CORE, and DPU channel/output fields match observed rows |
| task-layout finding | exact RKNN task buffer has 11 task entries with reg counts `69,108,25,62,11,57,9,56,9,56,9`, not a simple 3-task k_tile submit |
| submit-layout blocker | local BY_K previously submitted the body-parity k_tile rows directly; RKNN wraps them in setup/k_half/k_tile/PC rows with alternating `enable_mask` values `0x18` and `0x0d` and different row lengths |
| final behavior | BY_K remains rejected before allocation; no local BY_K hardware rerun was attempted |

Changed submit-layout hypothesis before any future BY_K rerun:

- The timeout is likely caused by submitting only the three k_tile body rows with a
  synthetic PC chain. RKNN's h14 task stream is an 11-entry schedule that includes
  setup and k_half/channel-family rows before and around the k_tile rows, with
  task-specific `regcfg_amount` and `enable_mask` values.
- The next safe BY_K hardware attempt must reproduce the RKNN task segmentation or
  prove a smaller equivalent task sequence in no-submit form first. A body-register
  parity-only rerun remains forbidden.

### Milestone 12: `BY_YK` / Mixed Families

Status: fenced. `BY_YK` combines Y windows with K families and includes unresolved
mixed h40 `setup`/`k_half` semantics. It is not a safe runtime target until BY_Y
and BY_K work independently.

Deliverable: one mixed Y+K shape emits RKNN-equivalent descriptor families and
passes hardware, or BY_YK remains explicitly unsupported with named unresolved
family semantics.

Prerequisites:

| Prerequisite | Required state |
|---|---|
| pointwise BY_Y | A small batch passes through the generic row overlay. |
| non-pointwise BY_Y | At least one spatial or depthwise BY_Y overlay is understood, or BY_YK scope is limited to pointwise-only shapes. |
| BY_K/k_tile | h14 and h7 pass, or the BY_YK candidate does not consume `k_tile`. |
| `setup`/`k_half` semantics | Mixed h40 rows are mined or permanently excluded from the first BY_YK runtime. |

Work allowed:

| Step | Action | Rule |
|---|---|---|
| 1 | Choose one BY_YK candidate with existing RKNN evidence. | Prefer a pointwise candidate before h40 spatial mixed-family pressure. |
| 2 | Compose descriptor offsets additively from existing Y and K windows. | No new strategy loop. |
| 3 | Emit only families with executable-equivalent semantics. | `setup`/`k_half` stay fenced if still unresolved. |

Acceptance, after prerequisites are met:

```sh
python3 examples/simple_add.py
python3 examples/conv.py <one-BY_YK-shape>
python3 examples/simple_add.py
```

Exit rule:

- If `setup`/`k_half` still exceed the modeled CBUF/reuse semantics, keep them
  non-executable and do not fake RKNN equivalence.
- If BY_K is still fenced, BY_YK must stay fenced too unless the selected shape has
  a valid non-`k_tile` plan.
- Any BY_YK code must reuse the planner descriptor loop and remain inside the
  runtime line budget; no captured RKNN schedule table.

Milestone 12 execution result on 2026-05-31:

| Check | Result |
|---|---|
| pointwise BY_Y prerequisite | satisfied for c64/c96/c128/c144 tested variants |
| non-pointwise BY_Y prerequisite | not satisfied; spatial/depthwise Y overlays remain fenced |
| BY_K/k_tile prerequisite | not satisfied; h14 task-layout blocker remains |
| setup/k_half semantics prerequisite | not satisfied for mixed families |
| hardware run | not attempted |
| final behavior | BY_YK remains rejected before allocation |

Milestone 12 status:

- BY_YK is intentionally unchanged. The new c144 BY_Y pass broadens pointwise Y
  tiling only; it does not make K-family or mixed setup/k_half rows executable.
- The h14 RKNN capture strengthened the reason to keep BY_YK fenced: mixed-family
  execution depends on the same RKNN task segmentation that BY_K has not yet
  reproduced safely.

### Milestone 13: Non-Pointwise `BY_Y/y_tile`

Status: proposed next hardware-facing milestone. Pointwise BY_Y is now a generic
batch-supported class, so the next broad unlock is one spatial or depthwise BY_Y
shape. This should happen before revisiting pointwise-wide `NONE`, BY_K hardware,
or BY_YK runtime code.

Deliverable: one non-pointwise BY_Y shape passes through the same planner-driven
descriptor loop, or non-pointwise BY_Y remains rejected before allocation with a
named missing overlay group.

Candidate order:

| Shape | Reason |
|---|---|
| `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` | Spatial BY_Y with ONNC split evidence and no mixed h40 K-family pressure. |
| `b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid` | Replaces old `spatial_im2col`, but only after the smaller spatial BY_Y candidate is understood. |
| `b1_c96_h150_w150_oc96_wic1_k3x3_g96_s1_pvalid` | Depthwise BY_Y candidate; useful only after one normal spatial BY_Y path works. |

Work allowed:

| Step | Action | Rule |
|---|---|---|
| 1 | Do a no-submit row-reg diff for one spatial BY_Y candidate against legacy/RKNN evidence. | Include `CNA_FEATURE_DATA_ADDR`, `CNA_DATA_SIZE0/3`, `CORE_DATAOUT_SIZE_0`, `DPU_DST_BASE_ADDR`, `DPU_WDMA_SIZE_1`, `DPU_SURFACE_ADD`, and input/output offsets. |
| 2 | Reuse `make_y_tile_regs()` if possible. | Extend the existing row overlay; do not create a separate spatial strategy runtime. |
| 3 | Run one hardware attempt only after the row diff has a named change. | Use `simple_add.py` before and after. |

Acceptance:

```sh
wc -l examples/conv.py conv_expt/conv_tile_planner.py
python3 examples/simple_add.py
python3 examples/conv.py b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid
python3 examples/simple_add.py
```

Exit rule:

- If the spatial candidate needs a per-shape register constant, reject it and keep
  non-pointwise BY_Y fenced.
- If the row overlay is generic but fails hardware, record the exact differing
  register group and do not add a fallback/im2col path to `examples/conv.py`.
- Runtime must remain under the soft combined 950-line budget unless this milestone
  unlocks a whole non-pointwise BY_Y class.

Milestone 13 execution result on 2026-05-31:

| Check | Result |
|---|---|
| candidate | `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` |
| planner rows | `BY_Y/y_tile`, rows `(y=0,input_h=64,out_h=62)`, `(62,64,62)`, `(124,36,34)` |
| no-submit row-reg diff | PASS against the legacy spatial row-tile overlay for `CNA_FEATURE_DATA_ADDR`, `CNA_DATA_SIZE0/3`, `CORE_DATAOUT_SIZE_0`, `DPU_DST_BASE_ADDR`, `DPU_WDMA_SIZE_1`, `DPU_SURFACE_ADD`, CBUF, and `CNA_CONV_CON2` |
| full-output hardware attempt | pre-health PASS, then failed before submit while mapping the required `~6.1 MiB` FP16 output BO: `OSError: [Errno 6] No such device or address`; post-health PASS |
| tile-local output hypothesis | changed runtime temporarily to submit each planner Y tile with local output stride/output BO, avoiding the oversized output map |
| tile-local hardware attempt | pre-health PASS, first tile `npu_submit` timed out with `Errno 110`; post-health `simple_add.py` PASS |
| final runtime behavior | reverted the temporary enablement; non-pointwise BY_Y again rejects before allocation with `BY_Y/y_tile is disabled before allocation except pointwise row tiling` |
| line budget after revert | `502 examples/conv.py` + `411 conv_expt/conv_tile_planner.py` = `913` lines |

Milestone 13 status:

- The generic spatial row overlay is not the blocker: required register fields match
  the legacy row-tile emitter without per-shape constants.
- The safe submit model is still unresolved. The full-output PC chain cannot be run
  through the current fixed BO path because the output allocation exceeds what this
  vendor-driver mapping path accepted, and the tile-local independent-submit
  hypothesis timed out.
- Non-pointwise BY_Y remains fail-closed. Do not add an im2col fallback or per-shape
  constants to `examples/conv.py`; the next attempt needs an output-buffer/submit
  layout change, not another row-reg tweak.

### Milestone 14: BY_K Task-Layout Fixture, No Hardware Rerun

Status: proposed offline milestone. Milestone 11 found that RKNN h14 uses an
11-task schedule, while the local timeout came from submitting only the three
body-parity `k_tile` rows. The next BY_K work should create a no-submit fixture
for that task segmentation before touching the live submit path.

Deliverable: a no-submit task-layout fixture that describes the RKNN h14 11-task
schedule enough to explain `regcfg_amount`, PC tails, task offsets, and
`enable_mask` ordering, without importing that fixture into `examples/conv.py`.

Work allowed:

| Step | Action | Rule |
|---|---|---|
| 1 | Materialize the RKNN h14 task row lengths `69,108,25,62,11,57,9,56,9,56,9`. | Offline only; no DRM open and no `npu_submit`. |
| 2 | Annotate rows by family/setup/PC role and `enable_mask`. | Keep body-register parity separate from submit-layout parity. |
| 3 | Compare local synthetic 3-task layout against RKNN 11-task layout. | Produce one minimal changed submit hypothesis for a later milestone. |

Acceptance:

```sh
python3 examples/conv_h14_k_tile_no_submit.py
python3 <new-or-existing-no-submit-task-layout-check>
wc -l examples/conv.py conv_expt/conv_tile_planner.py
```

Exit rule:

- Do not import the no-submit task-layout fixture into `examples/conv.py`.
- Do not run h14/h7 BY_K hardware in this milestone.
- If the fixture grows large, keep it proof-only and preserve the active runtime
  line budget.

Milestone 14 execution result on 2026-05-31:

| Check | Result |
|---|---|
| new proof-only checker | `examples/conv_h14_task_layout_no_submit.py` |
| runtime import status | not imported by `examples/conv.py` |
| RKNN task dump parsed | `/home/orangepi/npu/ops_rknn/dump/b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid_gem1_tasks.txt` |
| parsed task rows | `831` total rows in the full GEM1 dump; checker validates the first 11-task h14 BY_K prefix |
| expected prefix reg counts | `69,108,25,62,11,57,9,56,9,56,9` |
| expected prefix masks | `0x18,0x0d,0x18,0x0d,0x18,0x0d,0x18,0x0d,0x18,0x0d,0x18` |
| checker result | PASS for task dump presence, prefix length, reg counts, enable masks, `op_idx=2`, mismatch versus local `(9,9,9)` layout, and mismatch versus local all-`0x0d` mask assumption |
| local comparison | local synthetic BY_K was `3 x 9` direct k_tile body rows with `enable_mask=0x0d`; RKNN starts with setup/k_half auxiliaries and alternates `0x18/0x0d` before/around k_tile body rows |
| hardware rerun | not attempted |

Changed submit-layout hypothesis produced by Milestone 14:

- BY_K cannot safely rerun as local `3 x 9` direct k_tile rows. A future hardware
  attempt must either reproduce the RKNN prefix segmentation (`69,108,25,62,11,57,
  9,56,9,56,9`) or prove a smaller no-submit-equivalent sequence that preserves the
  auxiliary rows, `op_idx=2`, and alternating `0x18/0x0d` enable-mask structure.
- Body-register parity for k_tile rows remains useful but is insufficient; the
  timeout hypothesis is now specifically submit/task segmentation, not the 9 body
  qwords themselves.

### Milestone 15: Runtime Budget Guardrail

Status: standing guardrail for all future milestones. The hard requirement is
`examples/conv.py < 1000` lines, but the practical rule remains the combined
runtime path under about 950 lines while the planner is imported.

Next budget actions:

| Trigger | Required action |
|---|---|
| `examples/conv.py + conv_tile_planner.py >= 950` | Recover at least 20 lines before adding another family. |
| A new helper is imported by `examples/conv.py` | Delete at least the same number of active-runtime lines. |
| A status/example list grows beyond a few labels | Replace it with generated status from encoded names or move examples to docs. |
| A fixture table is needed | Keep it no-submit/proof-only unless it replaces more runtime code than it adds. |

Preferred deletion targets if budget tightens:

| Target | Reason |
|---|---|
| `LIST_SHAPES` labels | Easy to move to docs or generate from a tiny tuple of shape/status pairs. |
| duplicated `_conv_params` planner math | Worth touching only if sharing code is net-negative in lines and does not make submit code harder to audit. |
| c32 108-reg closure literals | Can be compacted further, but only if the passing c32 setup path remains byte-for-byte equivalent after patching DMA addresses. |

Milestone 15 guardrail result on 2026-05-31:

| Check | Result |
|---|---|
| active runtime line count | `502 examples/conv.py` + `411 conv_expt/conv_tile_planner.py` = `913` lines |
| proof-only fixture line count | `examples/conv_h14_task_layout_no_submit.py` is outside the active runtime path |
| runtime imports | `examples/conv.py` imports only `conv_expt/conv_tile_planner.py` from project CONV helpers |
| budget action | no runtime deletion required; combined runtime is below the `950` soft guardrail |

Guardrail status:

- The temporary Milestone 13 spatial BY_Y enablement was reverted after timeout, so
  the active runtime returned to the pre-milestone `913` combined-line budget.
- The BY_K task-layout checker is proof-only and does not count against the active
  runtime path unless it is imported later, which is forbidden by Milestone 14.
- Milestone 16 added a three-line generic large-output BO mapping guard to the active
  runtime, so the current combined line count is `916`, still below the soft
  guardrail.

### Milestone 16: Pointwise BY_Y Batch And Output BO Mapping

Status: complete. This milestone made one zero-growth pointwise BY_Y coverage gain
and isolated the next non-pointwise BY_Y blocker.

Deliverable: more pointwise BY_Y coverage with no runtime growth, plus one named
output-buffer rule for non-pointwise BY_Y. BY_K remains no-submit-only unless a
prefix-equivalent task schedule is proved.

Planned actions:

| Priority | Action | Acceptance |
|---|---|---|
| 1 | Run 2-4 additional encoded pointwise BY_Y shapes that classify into the already-supported subfamily. | Each pass uses existing parser/overlay code, with `simple_add.py` before and after; no new shape dictionary entries. |
| 2 | Investigate the Milestone 13 full-output BO mapping failure with an allocation/mapping-only probe before touching CONV submit. | Record whether the `~6.1 MiB` output failure is size alignment, handle lifecycle, mmap length, or driver limit. |
| 3 | If output mapping is solved, retry the spatial BY_Y candidate only with the full-output PC-chain model, not the timed-out tile-local independent submit. | The plan must name the changed output-buffer or submit fields before hardware. |
| 4 | Extend the BY_K no-submit checker from prefix validation to a compact prefix-equivalent emitter sketch. | It must describe task row lengths, masks, `op_idx`, PC tails, and `regcfg_amount` without being imported by `examples/conv.py`. |
| 5 | If combined runtime reaches 950 lines, recover at least 20 lines before adding more CONV family code. | `wc -l examples/conv.py conv_expt/conv_tile_planner.py` stays below the guardrail unless a whole new family passes. |

Prior `progress.md` evidence that can help:

| Evidence | How to use it now | Runtime rule |
|---|---|---|
| `examples/conv_tiles.py` H7/H14 six-desc paths passed hardware with post-health checks. | Treat BY_K as a schedule-reproduction problem, not a body-register or hardware impossibility. Mine the passing task topology, separator rows, PC links, and core assignment. | Do not import `conv_tiles.py` or copy captured schedules into `examples/conv.py`. |
| C64/H56 RKNN runtime profile used cacheable non-contiguous BO flags plus MEM_SYNC, and offline source checks explain the mmap path. | Use this for the Milestone 13 output mapping probe: compare current non-cacheable fixed BO allocation against RKNN-style `NON_CONTIGUOUS|CACHEABLE|IOMMU_LIMIT_IOVA_ALIGNMENT` and page-aligned sizes before any CONV submit. | A compute path using cacheable BOs must add explicit MEM_SYNC; do not silently switch runtime flags. |
| C64/H56 raw sparse run produced correct output but poisoned the next PC job. | Keep `simple_add.py` before/after every risky submit and treat post-job health as part of pass/fail, especially for spatial/BY_K schedule experiments. | Do not rerun known pollution or tail-layout probes unchanged. |
| `experimental/rknn/compare_conv_tiles_offline.py` parses task objects, regcmd windows, MEM_SYNC sequences, free rows, MMIO skips, and PC trace logs. | Reuse its parsing ideas for proof-only checks if BY_K prefix emission or output allocation needs validation. | Keep these checks outside the active runtime path unless they replace more code than they add. |
| `formula_direct_spatial_schedule()` contains partial RKNN-derived schedule formulas. | Use only as a mining reference for descriptor ordering and row/window hypotheses. The current planner remains the source of runtime descriptors. | Do not reintroduce the old strategy/template scheduler. |

Other `conv*.py` files that can help:

| File | Useful for | Do not use for |
|---|---|---|
| `examples/conv_simple.py` | Side-by-side reference for the small planner runtime before the latest compaction. Useful when a compacted register constant or guard in `examples/conv.py` looks suspicious. | Do not keep it as a second delivery path; delete or archive later if it stops adding debugging value. |
| `examples/conv_legacy.py` | Legacy full-register oracle for setup/NONE and pointwise BY_Y row overlays. It was already useful for Milestone 13 spatial row-reg parity. | Do not revive its strategy branches or CPU fallback behavior. |
| `examples/conv_tiles.py` | Passed H40 sparse, H7/H14 six-desc, C32-H14, and PW-C256-H14 with decoded math plus captured schedule topology. Best source for BY_K/BY_YK task topology and RKNN-like PC/link behavior. | Do not import it or copy captured templates into the small runtime. |
| `examples/conv_h14_task_layout_no_submit.py` | Current proof-only BY_K 11-task prefix checker. Extend this before any BY_K hardware rerun. | Do not import it into `examples/conv.py`. |
| `examples/conv_h14_k_tile_no_submit.py`, `examples/conv_no_submit_fixtures.py`, `examples/conv_no_submit_materializer.py`, `examples/conv_tiles_no_submit.py` | Existing no-submit register/body parity fixtures for setup, h7/h14 k_tile, observed y_tile/k_half rows. | Do not expand them unless they are the oracle for the current failing shape. |
| `conv_expt/conv_tile_cpu.py` and `experimental/rknn/conv_tile_expt.py` | Offline classification, CBUF pressure, and historical 217-shape grouping. Useful for candidate selection and parser status labels. | Do not make them runtime dependencies. |
| `examples/kernel_6_18/conv_mesa.py` and `experimental/conv_mesa_raw.py` | Mesa/Rocket compiler reference for CBUF, register names, row tiling, and task-building ideas. | Do not treat INT8 Mesa behavior as FP16 correctness proof. |
| `examples/kernel_6_18/conv.py`, `conv_new.py`, `conv_new_clean.py`, `conv_new_trscrs.py`, `conv_0522.py`, `conv_draft.py` | Historical Rocket/mainline attempts and old strategy sources. Useful for understanding why the six-strategy path grew too large. | Do not edit `examples/kernel_6_18/` or port its strategy runtime back. |
| `examples/conv_gemm.py` and `experimental/kernel_6_18/conv_gemm.py` | 1x1/matmul special-case register and packing reference. | Do not let matmul-specific assumptions leak into generic spatial CONV. |

Explicit non-goals for Milestone 16:

- Do not enable pointwise-wide `NONE` by copying RKNN task rows into the runtime.
- Do not rerun h14/h7 BY_K with the local `3 x 9` body-row submit.
- Do not revive `spatial_im2col` as a fallback in `examples/conv.py`.
- Do not add captured schedule tables to the active runtime path.

Milestone 16 execution result on 2026-05-31:

| Check | Result |
|---|---|
| added pointwise BY_Y coverage | `conv2d_b1_c112_h56_w56_oc24_wic112_k1x1_g1` PASS, `tasks=2`, max_diff `0.0156` |
| added pointwise BY_Y coverage | `conv2d_b1_c160_h56_w56_oc24_wic160_k1x1_g1` PASS, `tasks=3`, max_diff `0.0294` |
| added pointwise BY_Y coverage | `conv2d_b1_c192_h56_w56_oc24_wic192_k1x1_g1` PASS, `tasks=4`, max_diff `0.0310` |
| health behavior | pre-health passed for each CONV; immediate post-health sometimes timed out transiently, but repeated `simple_add.py` passed and final health passed |
| output BO mapping probe | new proof-only `examples/conv_output_bo_map_no_submit.py`; no submit path |
| contiguous BO probe | current allocation order maps `4 MiB` but fails for `5 MiB`, h160 `6,389,760` bytes, and `8 MiB` with `OSError: [Errno 6]` |
| non-contiguous BO probe | maps `4 MiB`, `5 MiB`, h160 `6,389,760` bytes, and `8 MiB` successfully with `RKNPU_MEM_NON_CONTIGUOUS` |
| runtime output BO change | large output BOs now use non-contiguous non-cacheable allocation; no cacheable BO or MEM_SYNC behavior was introduced |
| spatial BY_Y full-output retry | h160 got past BO mapping, but full-output PC-chain submit timed out with `Errno 110`; post-health PASS |
| final spatial BY_Y behavior | non-pointwise BY_Y remains rejected before allocation; pointwise BY_Y remains enabled |
| active runtime line count | `505 examples/conv.py` + `411 conv_expt/conv_tile_planner.py` = `916` lines |

Milestone 16 status:

- Pointwise BY_Y coverage expanded without adding any shape dictionary entries or
  per-shape constants. The existing encoded parser and generic row overlay handled
  c112, c160, and c192.
- The Milestone 13 output mapping blocker is solved generically for large outputs:
  use non-contiguous non-cacheable output BOs above the 4 MiB contiguous mapping
  limit. This is kept in `examples/conv.py` because it is a generic allocation rule,
  not a captured schedule.
- The remaining non-pointwise BY_Y blocker is now specifically full-output PC-chain
  submit/task behavior. Do not rerun h160 unchanged; the next attempt must name a
  changed PC/task/output-stride field or use a no-submit checker to prove it first.

### Milestone 17: Spatial BY_Y Submit Layout, No Blind Hardware

Status: complete as a proof-only milestone. Spatial BY_Y has coherent row registers
and a working large-output BO mapping rule, but the full-output PC-chain submit
timed out. This milestone identified the local task-topology mismatch before any
new hardware run.

Deliverable: a no-submit spatial BY_Y task-layout check that compares the current
three-row full-output submit against a known-good reference topology or a mined RKNN
task prefix, then either names one changed submit hypothesis or keeps spatial BY_Y
parked.

Work allowed:

| Step | Action | Rule |
|---|---|---|
| 1 | Materialize the local h160 spatial BY_Y task layout offline. | Include row lengths, PC tails, `PC_REGISTER_AMOUNTS`, `regcmd_addr` alignment, `regcfg_amount`, `enable_mask`, and output offsets. |
| 2 | Compare against `conv_tiles.py` or RKNN task topology without importing either into runtime. | The comparison is proof-only and must not submit. |
| 3 | Produce one changed submit hypothesis. | Examples: different `enable_mask`, `op_idx`, separator/aux rows, PC amount formula, output BO flag interaction, or subcore assignment. |

Acceptance:

```sh
python3 examples/simple_add.py
python3 <new-or-existing-spatial-by-y-no-submit-layout-check>
python3 examples/conv_output_bo_map_no_submit.py
wc -l examples/conv.py conv_expt/conv_tile_planner.py
```

Exit rule:

- Do not rerun `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` until the no-submit
  checker identifies a changed submit/PC-layout hypothesis.
- Keep non-pointwise BY_Y fail-closed if the only available change is another
  independent tile-local submit or copied schedule table.

Milestone 17 execution result on 2026-05-31:

| Check | Result |
|---|---|
| new proof-only checker | `examples/conv_spatial_by_y_layout_no_submit.py` |
| runtime import status | not imported by `examples/conv.py` |
| NPU pre-health | `python3 examples/simple_add.py` PASS |
| h160 planner rows | `BY_Y/y_tile`, rows `(0,64,62)`, `(62,64,62)`, `(124,36,34)` |
| local full-output task layout | three sparse y_tile tasks, `regcfg_amount=(47,47,47)`, `enable_mask=0x0d` for all rows, `op_idx=1`, `regcmd_qword=(0,52,104)` |
| local PC tails | `PC_REGISTER_AMOUNTS=(0x19,0x19,0x0)`, so the current full-output chain is core0-only and points to 47-qword successor rows |
| local output offsets | emitted byte offsets `(0,156736,313472)` match the current runtime formula `y_start*out_w*UNPACK_C2*2`; planner `output_off` remains descriptor units `(0,19592,39184)` |
| conv_tiles reference comparison | known direct-spatial c32 h14 topology starts with `setup,k_half,k_half,y_tile,y_tile,y_tile`, uses six 62-qword rows, and PC core lanes `(0,1,1,2,2,2)` |
| topology mismatch | PASS: h160 local `3 x 47` direct y_tile submit is not equivalent to the direct-spatial reference topology |
| output BO mapping regression | `python3 examples/conv_output_bo_map_no_submit.py` PASS; h160 output maps only with non-contiguous BO |
| active runtime line count | `505 examples/conv.py` + `411 conv_expt/conv_tile_planner.py` = `916` lines |

Changed submit-layout hypothesis produced by Milestone 17:

- Do not rerun h160 as the current three sparse y_tile tasks. The next safe attempt
  must add a direct-spatial prelude/closure like the `conv_tiles.py` setup+k_half
  rows and 62-qword register closure, or prove a smaller equivalent sequence first.
- A minimal smaller-equivalent hypothesis must at least account for y_tile PC core
  lane bits and the missing reset/clear qwords; the current core0-only
  `PC_REGISTER_AMOUNTS=(0x19,0x19,0x0)` remains forbidden for another hardware run.
- Non-pointwise BY_Y remains rejected before allocation until such a no-submit
  equivalent emitter exists. Copying a captured schedule table into `examples/conv.py`
  is not allowed under the line-budget rule.

### Milestone 18: Compact Direct-Spatial Prelude/Closure

Status: complete as a proof-only milestone. This step proved a compact six-task
topology can be modeled generically, but rejected h160 hardware rerun until exact
RKNN h160 setup/k_half row values are mined or another oracle proves equivalence.

Deliverable: a proof-only compact emitter sketch for the h160 spatial BY_Y candidate
that explains the missing direct-spatial prelude/closure, or a permanent decision
to keep non-pointwise BY_Y fenced until a generic schedule formula is available.

Work allowed:

| Step | Action | Rule |
|---|---|---|
| 1 | Compare h160 planner y_tile rows against the smallest passing direct-spatial schedule family in `examples/conv_tiles.py`. | Mine topology and qword groups only; do not import or replay the captured schedule. |
| 2 | Identify the minimal generic prelude/closure register groups missing from the local `3 x 47` chain. | Must name groups, not shape constants: setup/k_half auxiliary rows, DPU clear/reset rows, PC core lane bits, separator rows, or PC amount formula. |
| 3 | Add or extend a no-submit checker if the hypothesis can be expressed compactly. | The checker must stay proof-only and outside the active runtime path. |
| 4 | Only after the no-submit checker passes, decide whether a small runtime change is justified. | If the runtime change would push combined lines near 950, recover lines first or keep the family fenced. |

Acceptance before any h160 hardware rerun:

```sh
python3 examples/simple_add.py
python3 examples/conv_spatial_by_y_layout_no_submit.py
python3 examples/conv_output_bo_map_no_submit.py
wc -l examples/conv.py conv_expt/conv_tile_planner.py
```

Exit rule:

- Do not run h160 hardware again with the current `3 x 47` core0-only chain.
- Do not add a captured direct-spatial schedule table to `examples/conv.py`.
- If the compact prelude/closure cannot be stated as generic code, keep
  non-pointwise BY_Y fail-closed and move to BY_K no-submit prefix equivalence.

Milestone 18 execution result on 2026-05-31:

| Check | Result |
|---|---|
| proof-only checker extended | `examples/conv_spatial_by_y_layout_no_submit.py` now materializes a compact y_tile closure sketch |
| runtime import status | not imported by `examples/conv.py` |
| NPU pre-health | `python3 examples/simple_add.py` PASS |
| local h160 baseline | still `3 x 47` sparse y_tile rows with core0-only `PC_REGISTER_AMOUNTS=(0x19,0x19,0x0)` |
| compact closure groups | generic RDMA-zero qwords at RDMA offsets `0x1020,0x1040,0x1044` plus DPU clear qwords `0x4100..0x412c` |
| compact y_tile closure | PASS: local h160 y_tile rows become `3 x 62` qwords, matching the known direct-spatial row closure size without a captured schedule table |
| compact y_tile PC lanes | PASS: y_tile core-lane PC amounts become `(0x20020,0x20020,0x0)` for closed 62-qword successor rows |
| compact prelude topology | PASS: formula-derived setup, two k_half rows, and three y_tile rows form six 62-qword tasks with PC lanes `0/1/1/2/2/2` |
| prelude row-value status | unresolved: setup/k_half row values are formula-derived for h160, not RKNN-captured or hardware-proved |
| output BO mapping regression | `python3 examples/conv_output_bo_map_no_submit.py` PASS |
| active runtime line count | `505 examples/conv.py` + `411 conv_expt/conv_tile_planner.py` = `916` lines |

Milestone 18 decision:

- Do not run h160 hardware with only the compact `3 x 62` y_tile closure or with the
  formula-derived six-task prelude. The topology now matches the known direct-spatial
  shape, but setup/k_half row values still need an h160 RKNN or equivalent oracle.
- Non-pointwise BY_Y remains fail-closed before allocation. The next spatial BY_Y
  action must mine exact h160 RKNN task/register topology or park spatial BY_Y and
  move to BY_K no-submit prefix equivalence.
- The active runtime remains unchanged and below the line-budget guardrail.

Recommended next work after Milestone 18:

| Priority | Action | Reason |
|---|---|---|
| 1 | Capture or mine exact h160 RKNN topology. | Six-task topology is expressible; row values need an oracle before local submit code changes. |
| 2 | Continue BY_K prefix-equivalent no-submit work if h160 RKNN mining is not practical. | BY_K and BY_YK remain blocked by task segmentation, not by simple body regs. |
| 3 | Recover runtime lines before any accepted spatial BY_Y runtime change. | Current combined runtime is `916`; a family addition should stay below the soft `950` guardrail. |
| 4 | Keep pointwise-wide `NONE` parked. | RKNN shows a multi-descriptor channel schedule; local one-row setup has failed. |

### Milestone 19: Exact H160 RKNN Topology

Status: complete as a proof-only milestone. This captured the exact h160 spatial
BY_Y RKNN run and used it to reject the compact six-task topology before any local
hardware submit change.

Deliverable: preserve exact h160 RKNN artifacts, extend the no-submit checker with
the mined topology, and decide whether the next safe action is local hardware or
offline schedule redesign.

Milestone 19 execution result on 2026-05-31:

| Check | Result |
|---|---|
| RKNN model generated | `/home/orangepi/npu/ops_rknn/models/b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid.rknn` |
| RKNN capture command | `sh run.sh --gdb conv2d_multi --case b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid 1 16 160 160 128 3 3 1 h160_BY_Y` |
| RKNN capture health | `python3 examples/simple_add.py` PASS before and after |
| RKNN runtime result | PASS, max error `0.0389099` |
| captured artifacts | `dump/b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid_emit.txt`, input/weight dumps, GEM2 task/regdump copies |
| no-submit checker update | `examples/conv_spatial_by_y_layout_no_submit.py` now parses the exact h160 RKNN emit rows |
| RKNN row lengths | `(53,49,16,47,47,47,15,47,47,47,15,47,15,47,15,47,15,17)` |
| RKNN family sequence | `setup,setup,setup,k_half,k_half,k_half,ppu,k_half,k_half,k_half,ppu,y_tile,ppu,y_tile,ppu,y_tile,ppu,tail` |
| RKNN y_tile input windows | `(55,55,54)`, not the local planner windows `(64,64,36)` |
| compact six-task hypothesis | rejected: it is not RKNN-equivalent for h160 |
| local hardware rerun | not attempted |

Milestone 19 decision:

- Non-pointwise BY_Y remains fail-closed before allocation. The exact RKNN schedule
  disproves the current local `3 x 47` chain and the compact six-task closure as
  safe h160 hardware candidates.
- The next spatial BY_Y action must be offline redesign of planner/emitter schedule:
  mine the useful prefix from RKNN's setup triplet, k_half triplets, PPU/separator
  rows, and tail, then prove which rows are actually required for our no-pooling
  case. RKNN's `55/55/54` y_tile windowing is a real planner clue, but PPU/trailing
  rows must not be assumed mandatory without a smaller-equivalent proof. Do not
  submit h160 locally until a no-submit checker proves that new schedule shape.
- BY_K remains the alternate next path if spatial BY_Y schedule redesign grows too
  much or cannot be expressed generically under the line budget.

### Milestone 20: BY_K Prefix-Equivalent No-Submit Check

Status: complete as a proof-only milestone. This tightened the BY_K no-submit
blocker by validating the RKNN prefix backing layout, not just task counts and
enable masks.

Deliverable: extend the h14 BY_K checker to prove the captured RKNN prefix needs
auxiliary/setup/k_half rows and a much larger backing stream before the 9-qword
k_tile bodies, while keeping BY_K out of runtime.

Milestone 20 execution result on 2026-05-31:

| Check | Result |
|---|---|
| proof-only checker extended | `examples/conv_h14_task_layout_no_submit.py` |
| runtime import status | not imported by `examples/conv.py` |
| RKNN prefix reg counts | `(69,108,25,62,11,57,9,56,9,56,9)` |
| RKNN prefix enable masks | alternating `(0x18,0x0d,0x18,0x0d,0x18,0x0d,0x18,0x0d,0x18,0x0d,0x18)` |
| RKNN prefix op index | all prefix rows use `op_idx=2` |
| RKNN regcmd deltas | `(80,112,32,72,16,64,16,64,16,64,16)` qwords |
| RKNN prefix backing | first 11 rows advance `552` qwords before the next task |
| local mismatch | local `3 x 9` body-row submit has no setup/k_half auxiliary prefix and all-`0x0d` mask assumption is wrong |
| hardware rerun | not attempted |

Milestone 20 decision:

- BY_K remains fail-closed before allocation. A safe h14 rerun needs either a
  generated RKNN-prefix-equivalent schedule with the 552-qword backing layout and
  alternating masks, or a no-submit proof of a smaller equivalent. Body-register
  parity alone is not enough.
- Do not copy captured prefix rows into `examples/conv.py`. Any future BY_K runtime
  change must generate the auxiliary/setup/k_half/k_tile structure generically and
  stay within the line budget.

### Milestone 21: Compact Auxiliary Schedule Model

Status: complete as a proof-only milestone. Milestones 19 and 20 both point to the
same root problem: the active runtime can emit simple descriptor rows, but RKNN uses
auxiliary task schedules around those rows. This milestone proved the BY_K prefix
can be generated from planner descriptors, and separated h160's useful non-PPU rows
from full RKNN boilerplate without entering `examples/conv.py`.

Deliverable: a proof-only generated schedule representation that proves one of two
things:

- BY_K h14 prefix can be generated from planner descriptors plus compact family
  rules, matching the 11-task RKNN prefix shape closely enough for a guarded
  hardware attempt.
- Or BY_K cannot be made generic cheaply, in which case BY_K stays fenced and the
  project parks K-family runtime support to protect the `<1000` line target.

Work allowed:

| Step | Action | Rule |
|---|---|---|
| 1 | Extend `examples/conv_h14_task_layout_no_submit.py` or a tiny sibling checker to generate, not replay, the h14 BY_K prefix roles. | Must preserve expected counts `(69,108,25,62,11,57,9,56,9,56,9)`, masks, `op_idx=2`, deltas, and 552-qword backing. |
| 2 | Compare generated rows against local active-runtime row generation. | The output should name exactly which runtime emitter groups are missing: setup rows, k_half auxiliaries, PC separator rows, or tail rows. |
| 3 | Estimate runtime line cost before enabling anything. | If the generated runtime change would push combined lines over 950, recover lines first or keep BY_K fenced. |
| 4 | Keep h160 spatial BY_Y in the same checker as a useful-prefix mining case. | It should distinguish required setup/k_half/y_tile rows from optional PPU/trailing rows; 18-row replay is not the target. |

Acceptance:

```sh
python3 examples/conv_h14_task_layout_no_submit.py
python3 examples/conv_spatial_by_y_layout_no_submit.py
wc -l examples/conv.py conv_expt/conv_tile_planner.py
git diff --check -- conv_expt/conv_plan.md
```

Exit rule:

- If the BY_K prefix can be generated compactly, write the exact hardware-run
  hypothesis into this plan before changing `examples/conv.py` submit code.
- If the model needs copied RKNN rows or shape tables, stop and keep BY_K fenced.
- Do not run h14 hardware in Milestone 21; the first hardware rerun belongs to a
  later milestone only after the active-runtime diff is small and auditable.

Milestone 21 execution result on 2026-05-31:

| Check | Result |
|---|---|
| BY_K checker extended | `examples/conv_h14_task_layout_no_submit.py` now generates the prefix roles from the planner's three k_tile descriptors |
| generated BY_K counts | `(69,108,25,62,11,57,9,56,9,56,9)` match RKNN prefix |
| generated BY_K masks | alternating `(0x18,0x0d,0x18,0x0d,0x18,0x0d,0x18,0x0d,0x18,0x0d,0x18)` match RKNN prefix |
| generated BY_K deltas | `(80,112,32,72,16,64,16,64,16,64,16)` match RKNN prefix |
| generated BY_K backing | `552` qwords before the next task |
| h160 useful-prefix mining | `examples/conv_spatial_by_y_layout_no_submit.py` now separates useful rows from PPU/tail rows |
| h160 useful families | `setup,setup,setup,k_half,k_half,k_half,k_half,k_half,k_half,y_tile,y_tile,y_tile` |
| runtime import status | no proof-only checker is imported by `examples/conv.py` |
| hardware rerun | not attempted |

Milestone 21 decision:

- BY_K is now the preferred next hardware-facing unlock, but not in this milestone.
  The next step is a no-submit active-runtime emitter sketch and line-cost estimate:
  generate setup/k_half/k_tile auxiliary rows and the 552-qword backing layout from
  descriptor rows, without copying RKNN command rows into `examples/conv.py`.
- Spatial BY_Y h160 remains offline. The useful prefix is mined, but it is not yet
  generated from planner descriptors, and the planner still emits `64/64/36` rather
  than RKNN's `55/55/54` y_tile input windows.
- BY_K and non-pointwise BY_Y stay fail-closed before allocation until a later
  milestone produces a small, auditable runtime diff and a named hardware-run
  hypothesis.

Recommended Milestone 22, only if Milestone 21 passes:

| Candidate | Action | Acceptance |
|---|---|---|
| BY_K h14 generated prefix | Implement the smallest runtime emitter/submit diff for one generated prefix-equivalent schedule. | `simple_add.py` before and after, `examples/conv.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid` passes or fails with named changed fields, active runtime stays under budget. |
| Spatial BY_Y h160 | Do not implement yet unless the same schedule model compactly explains a useful prefix from the 18 RKNN row groups and `55/55/54` windows. | No local h160 submit until the checker generates the required prefix without a captured table and explicitly justifies dropping PPU/trailing rows. |
| Pointwise-wide NONE | Keep parked unless the auxiliary schedule model naturally covers the c144 h28 multi-descriptor channel schedule. | No new c144 hardware probe from weight-size or CBUF tweaks alone. |

Current recommendation after Milestone 21:

- Prefer BY_K h14 for the next hardware-facing unlock. Its schedule can now be
  generated in proof-only form from planner descriptors, so the next useful work is
  a line-budgeted runtime emitter sketch, still no hardware submit.
- Keep spatial BY_Y h160 offline until its useful prefix and `55/55/54` windows can
  be generated from planner descriptors rather than mined from one capture.

### Milestone 22: BY_K Runtime Gate And Line-Cost Estimate

Status: complete as a proof-only milestone. This milestone intentionally did not
enable BY_K and did not run h14 hardware. It converted the generated prefix proof
into an active-runtime gate: what must change in `examples/conv.py`, whether it fits
the soft line budget, and what row groups are still missing.

Deliverable: extend the BY_K checker with the smallest plausible runtime-diff
estimate and a named missing-group list, then decide whether BY_K can move directly
to hardware or must remain fenced.

Milestone 22 execution result on 2026-05-31:

| Check | Result |
|---|---|
| proof-only checker extended | `examples/conv_h14_task_layout_no_submit.py` |
| active runtime lines | `916` combined lines for `examples/conv.py` plus `conv_expt/conv_tile_planner.py` |
| soft budget margin | `34` lines below the 950-line guardrail |
| estimated BY_K runtime diff | minimum `45` lines for generated metadata, row materializers, larger regcmd backing, and PC-tail changes |
| runtime budget gate | PASS as a blocker: estimated diff exceeds soft margin |
| missing runtime group | variable per-task `op_idx`/`enable_mask` |
| missing runtime group | explicit regcmd qword deltas and `>4096` regcmd BO sizing |
| missing runtime group | `setup_aux/setup_body` and `k_half_aux/k_half_body` row materializers |
| missing runtime group | k_tile auxiliary rows between generated body descriptors |
| missing runtime group | PC tail core-lane/register-amount programming for the `0x18/0x0d` alternating schedule |
| hardware rerun | not attempted |

Milestone 22 decision:

- BY_K remains fail-closed before allocation. The generated schedule is promising,
  but enabling it now would likely cross the soft budget and still lacks generated
  aux/setup/k_half row contents.
- The next BY_K milestone should recover at least 15 active-runtime lines or find a
  smaller design than the 45-line estimate, then add a no-submit row-content
  generator. Only after that should the plan name a hardware-run diff for h14.
- Do not solve this by copying RKNN task rows into `examples/conv.py`; the required
  rows must be generated from descriptor/family rules.

Current recommendation after Milestone 22:

- First recover line budget in `examples/conv.py` or `conv_expt/conv_tile_planner.py`.
- Then build BY_K row-content generation no-submit. Hardware remains blocked until
  that checker passes and the active-runtime diff is under budget.

### Milestone 23: Runtime Budget Recovery For BY_K

Status: complete. This milestone recovered the line budget requested by Milestone
22 without changing submit behavior or enabling BY_K.

Deliverable: shrink the active runtime enough that the estimated 45-line BY_K diff
could fit under the soft 950-line guardrail, then verify existing supported paths.

Milestone 23 execution result on 2026-05-31:

| Check | Result |
|---|---|
| runtime compaction | `LIST_SHAPES` compressed and ioctl constant declarations compacted |
| active runtime lines | `489 examples/conv.py` + `411 conv_expt/conv_tile_planner.py` = `900` lines |
| soft budget margin | `50` lines below the 950-line guardrail |
| BY_K estimate gate | `examples/conv_h14_task_layout_no_submit.py` now reports enough soft margin for the 45-line estimate |
| `examples/conv.py --list` | PASS; output unchanged |
| pre-health `simple_add.py` | PASS |
| generic NONE smoke | `conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1` PASS, max_diff `0.0074` |
| pointwise BY_Y smoke | `conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1` PASS, max_diff `0.0156` |
| post-health `simple_add.py` | first immediate run timed out transiently after pointwise BY_Y; repeated run PASS |
| hardware BY_K | not attempted |

Milestone 23 decision:

- BY_K remains fail-closed before allocation. The budget gate is now cleared, but
  the row-content gate is not: setup/k_half auxiliary rows, k_tile aux rows,
  variable per-task metadata, explicit regcmd deltas, and PC core-lane tails still
  need a no-submit generator.
- The next milestone should add no-submit BY_K row-content generation. Do not run
  h14 hardware until that checker proves the generated rows and task metadata are
  coherent and the active-runtime diff remains within budget.

Current recommendation after Milestone 23:

- Build BY_K no-submit row-content generation next.
- Keep spatial BY_Y, pointwise-wide NONE, and BY_YK parked.

### Milestone 24: BY_K Row-Content Gate

Status: complete as a proof-only milestone. This milestone integrated the existing
h14 k_tile row-content fixture into the main BY_K task-layout checker. It proves
the body rows are available, and narrows the remaining blocker to auxiliary row
contents.

Deliverable: one BY_K checker output that reports schedule generation, soft budget,
k_tile body row-content parity, and auxiliary row-content status.

Milestone 24 execution result on 2026-05-31:

| Check | Result |
|---|---|
| proof-only checker extended | `examples/conv_h14_task_layout_no_submit.py` imports the existing proof-only `conv_h14_k_tile_no_submit` fixture; still not imported by `examples/conv.py` |
| generated prefix | still matches RKNN counts, masks, deltas, `op_idx=2`, and 552-qword backing |
| body row content | `body_row_content_generated=PASS`, three k_tile body rows with `9;9;9` decoded regs |
| body row parity | existing h14 k_tile fixture parity passes for all 27 checked body registers |
| auxiliary row status | `aux_row_content_still_fenced=PASS`; k_half observed regs are mined, but no planner-generated aux/k_half rows are executable |
| active runtime lines | still `489 + 411 = 900` |
| hardware rerun | not attempted |

Milestone 24 decision:

- BY_K remains fail-closed before allocation. The body rows are not the blocker;
  the missing piece is generated `setup_aux/setup_body`, `k_half_aux/k_half_body`,
  and `k_tile_aux` row contents plus matching task metadata.
- The next BY_K work should generate auxiliary row contents no-submit. If that
  requires copied RKNN rows or grows beyond the recovered budget, keep BY_K fenced.

Current recommendation after Milestone 24:

- Work on aux/setup/k_half row-content generation only if it can be expressed as
  compact descriptor/family rules. Do not run h14 hardware yet.

### Milestone 25: BY_K Auxiliary Row-Content Generation

Status: complete as a proof-only blocker milestone. Milestone 24 proved that the
three k_tile body rows are not the blocker. This milestone tried the smallest
compact k_half row-content generator and rejected it because it does not match the
RKNN prefix lengths.

Deliverable: extend proof-only BY_K generation so it emits decoded-register content
for the non-body prefix rows, or explicitly parks BY_K if those rows require copied
RKNN tables.

Work allowed:

| Step | Action | Rule |
|---|---|---|
| 1 | Generate `setup_aux/setup_body` decoded regs no-submit. | Use descriptor/family rules or compact constants already justified by setup fixtures; no captured task-row replay. |
| 2 | Generate `k_half_aux/k_half_body` decoded regs no-submit. | Existing observed k_half rows may be an oracle, but generated rows must come from planner descriptors and family rules. |
| 3 | Generate `k_tile_aux` rows around the already-passing k_tile body rows. | Keep body rows from the current fixture parity path; do not duplicate their constants. |
| 4 | Recompute active-runtime line cost. | If the real runtime diff exceeds the 50-line soft margin, recover lines before any hardware attempt. |
| 5 | Write the exact h14 hardware hypothesis only after no-submit row-content passes. | Must name `task_count`, `regcfg_amount`, `regcmd_addr` deltas/backing size, `op_idx`, `enable_mask`, and PC tail/core-lane behavior. |

Acceptance:

```sh
python3 examples/conv_h14_task_layout_no_submit.py
wc -l examples/conv.py conv_expt/conv_tile_planner.py
git diff --check -- conv_expt/conv_plan.md
```

Exit rule:

- If auxiliary rows can be generated compactly, Milestone 26 may implement the
  smallest runtime BY_K submit diff and run h14 with `simple_add.py` before/after.
- If auxiliary rows require copied RKNN rows, keep BY_K fenced and do not spend the
  recovered line budget on schedule replay.
- Do not work on spatial BY_Y h160 runtime until BY_K either passes h14 or is
  explicitly parked again; h160 still needs generated `55/55/54` windows and useful
  prefix rows.

Milestone 25 execution result on 2026-05-31:

| Check | Result |
|---|---|
| proof-only checker extended | `examples/conv_h14_task_layout_no_submit.py` |
| generated candidate | two k_half body rows from active `make_regs()`, patched to `0x40000000` family bits, plus RDMA-zero/DPU-clear direct-spatial closure |
| generated k_half body lengths | `(62,62)` |
| RKNN h14 k_half body lengths | `(62,57)` from the 11-task prefix |
| blocker status | candidate rejected as not RKNN-equivalent; second k_half body has a shorter closure/auxiliary rule |
| hardware rerun | not attempted |

Milestone 25 decision:

- BY_K remains fail-closed before allocation. The first k_half body length can be
  reproduced by compact closure, but the second cannot. A hardware rerun would still
  be a blind submit-layout probe.
- The next unblocked work is to mine or derive the shorter second k_half closure
  rule. Specifically, identify which five qwords are omitted versus the 62-qword
  direct-spatial closure, or prove that the task dump's 57-qword body is not
  comparable to the generated row model.

Current recommendation after Milestone 25:

- Continue BY_K offline only. Compare RKNN regcmd backing windows or emit rows to
  resolve the `(62,62)` versus `(62,57)` k_half discrepancy.
- Do not run h14 hardware yet.

### Milestone 26: BY_K Artifact Comparability Check

Status: complete as a proof-only blocker milestone. DeepWiki confirms that, in a
matching capture, task `regcfg_amount`/`regcmd_addr` rows should point into the same
regcmd stream represented by emitted command logs. The local preserved h14 artifacts
do not currently form such a comparable pair.

Deliverable: prove whether the preserved h14 task dump, emit log, and regdump can be
used to identify the missing five qwords in the second k_half body.

Milestone 26 execution result on 2026-05-31:

| Check | Result |
|---|---|
| DeepWiki reference | `allbilly/rknpu_driver` / `allbilly/npu` says task rows and emit logs are comparable when they come from the same regcmd stream |
| h14 task prefix lengths | `(69,108,25,62,11,57,9,56,9,56,9)` |
| h14 emit grouped lengths | `(100,47,15,47,15,47,15,47,15,47,15,11)` |
| comparability check | `emit_rows_not_task_prefix=PASS`; current emit grouping is not the GEM1 task prefix |
| preserved `gem2_regdump.bin` | produced by `dump.py` as compact decoded `<hIh>` records, not raw qwords with task-relative offsets |
| raw backing status | no stable preserved raw regcmd window tied to the task dump was found in the named h14 artifacts |
| hardware rerun | not attempted |

Milestone 26 decision:

- BY_K remains fail-closed before allocation. The current preserved artifacts cannot
  identify the exact five omitted qwords for the second k_half body.
- The next action needs a fresh or recovered RKNN h14 capture that preserves a raw
  regcmd GEM/window paired with the task dump, or an equivalent parser that maps the
  existing dump back to task-relative offsets. Do not run local h14 BY_K hardware.

Current recommendation after Milestone 26:

- Preferred: recapture RKNN h14 with raw regcmd GEM preservation and health checks
  before/after, then compare the task-relative backing windows for the first and
  second k_half bodies directly.
- If recapture is not available, build a proof-only offset mapper from the existing
  decoded dumps only if it can be validated against known prefix counts, deltas, and
  the 552-qword backing. Do not infer omitted qwords from grouped emit lengths alone.
- Otherwise keep BY_K parked and switch to another non-risky offline milestone.

Historical recommendation after Milestone 26, superseded by Milestones 27-37:

| Candidate | Action | Acceptance |
|---|---|---|
| BY_K raw backing comparability | Preserve or reconstruct a raw h14 regcmd window tied to the GEM1 task dump. | Proof-only checker reports task prefix counts `(69,108,25,62,11,57,9,56,9,56,9)`, deltas `(80,112,32,72,16,64,16,64,16,64,16)`, 552-qword backing, and direct task-relative windows for both k_half bodies. |
| BY_K k_half closure rule | Derive which five qwords are omitted from the second k_half body, or prove the generated `(62,62)` rows are not comparable to RKNN body rows. | No-submit generated k_half rows match `(62,57)` without copied RKNN row replay, or BY_K is explicitly parked again. |
| BY_K h14 hardware | Only after the first two rows pass, add the smallest generated-prefix runtime path for `b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid`. | `python3 examples/simple_add.py`, h14 CONV, `python3 examples/simple_add.py`; pass or fail with exact changed submit fields and post-health status. |
| BY_K h7 follow-up | Only after h14 passes or fails safely with useful evidence, adapt the same generated prefix to h7. | No new h7 submit until h14 result is understood. |
| Spatial BY_Y h160 | Keep no-submit-only. | Requires generated useful prefix and `55/55/54` planner windows before any local submit. |

### Milestone 27: BY_K Live Raw Window Comparability

Status: complete as a proof-only partial milestone. This recovered and validated a
raw h14 regcmd window from the preserved live submit log, but it is not the same
task topology as the older `69/108/...` GEM1 task dump. It therefore proves an
important negative: the live raw window is internally comparable, but it cannot be
used to identify the five qwords missing from the older second `k_half_body`.

Milestone 27 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now parses `experimental/rknn/capture_rknpu_submit_dump_gems_h14_live_regcmd_832_20260524_124752.log` |
| live raw tasks | 11 task rows parsed from GEM1 |
| live raw qwords | 832 qwords parsed from GEM2 raw `QWORD_WINDOW` |
| live raw counts | `(108,104,26,104,26,104,26,104,26,104,26)` |
| live raw masks | `(0x0d,0x0d,0x60,0x0d,0x60,0x0d,0x60,0x0d,0x60,0x0d,0x60)` |
| live raw deltas | `(112,112,32,112,32,112,32,112,32,112)` measurable adjacent deltas |
| live raw window coverage | every live task row has its full `regcfg_amount` qwords in the captured window |
| comparability decision | `live_raw_not_old_task_prefix=PASS`; live raw counts/masks do not match the older `(69,108,25,62,11,57,9,56,9,56,9)` prefix |
| old prefix status | still validates counts, masks, deltas, `op_idx=2`, and 552-qword backing, but still has no matching raw qword window |
| hardware rerun | not attempted |

Milestone 27 decision:

- BY_K remains fail-closed before allocation. The recovered raw window is useful as
  a parser/mapper proof, but it belongs to a different h14 schedule with `op_idx=1`
  and `0x0d/0x60` masks, not the older prefix with `op_idx=2` and `0x18/0x0d`
  masks.
- Do not compare the live raw qwords to the old generated `(62,62)` versus
  `(62,57)` k_half body discrepancy. That would mix two non-equivalent artifacts.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| BY_K old-prefix raw capture | Recapture or recover a raw h14 regcmd window whose GEM1 task rows exactly match `(69,108,25,62,11,57,9,56,9,56,9)`, `op_idx=2`, and masks `(0x18,0x0d,...)`. | Checker reports full task-relative qword windows for task 3 and task 5 of that same prefix, then compares the two k_half bodies directly. |
| BY_K live-topology analysis | If old-prefix raw capture is unavailable, analyze the live `(108,104,26,...)` topology as a separate schedule. | Checker names whether the live topology can be generated compactly from planner descriptors without using captured rows; no runtime enablement until it also has row-content rules. |
| Spatial BY_Y h160 | Keep no-submit-only. | Generate `55/55/54` windows and useful prefix rows from descriptors before any submit. |

### Milestone 28: BY_K Live Topology Offset Model

Status: complete as a proof-only partial milestone. This milestone analyzed the
live `(108,104,26,...)` topology separately from the older `(69,108,25,...)`
prefix. It proved the live schedule's body-family order and DMA offsets are
compactly describable from shape/planner data, while preserving the fence on row
contents and task metadata.

Milestone 28 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` decodes live raw qword windows into register maps |
| live body families | `setup,k_half,k_half,k_tile,k_tile,k_tile` from `CNA_CONV_CON2` values `(0xf0,0x400000f0,0x400000f0,0x500000f0,0x500000f0,0x500000f0)` |
| live k_half offsets | generated split at 160 output channels: weight offsets `(0x0,0x70800)`, output offsets `(0x0,0xb400)` |
| live k_tile offsets | match planner rows exactly: weight offsets `(0x0,0x4ec00,0x9d800)`, output offsets `(0x0,0x7e00,0xfc00)` |
| live auxiliary rows | five repeated `26`-qword `0x60` rows remain captured-only PPU/SDP auxiliaries |
| old prefix status | unchanged; old `(69,108,25,62,11,57,...)` prefix still lacks a matching raw qword window |
| hardware rerun | not attempted |

Milestone 28 decision:

- The live topology is more promising than a pure captured replay: body ordering and
  weight/output offsets are generatable from descriptors and a simple half-channel
  split.
- BY_K still remains fail-closed before allocation. The checker does not yet
  generate the 104-qword body row contents, the repeated 26-qword auxiliary rows,
  or the `op_idx=1`/`0x0d,0x60` task metadata from local runtime rules.
- Do not submit either the old prefix or the live topology until a no-submit
  materializer can produce row contents without captured qword replay.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Live topology body row-content diff | Compare live 104/108-qword body rows against active `make_regs()` closures after normalizing DMA bases and family bits. | Checker reports a small named delta set, or proves live body rows are not derivable from current runtime closures. |
| Live auxiliary row generator | Decode the repeated 26-qword `0x60` rows and determine whether they are a generic PPU/SDP tail. | Checker generates those rows from output shape/address parameters, or keeps them fenced with named unknown registers. |
| Old-prefix raw capture | Still valid if available. | Exact old-prefix raw task windows for tasks 3 and 5 would directly resolve the `(62,62)` versus `(62,57)` issue. |

### Milestone 29: BY_K Live Body Closure Diff

Status: complete as a proof-only partial milestone. This milestone compared each
live 104/108-qword body row against the active runtime's generated CONV register
closure, using the same `make_regs()` path and direct-spatial closure helper used
by earlier k_half checks.

Milestone 29 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now builds generated live-body register maps for `setup`, two `k_half`, and three `k_tile` rows |
| generated source | active `examples/conv.py make_regs()` plus family-bit patching and direct-spatial RDMA/DPU closure |
| body diff shape | every live body row has the same summary against generated closure: `live_extra=45`, `generated_only=3`, `value_diff=9` |
| stable gate | `live_body_closure_diff_stable=PASS` |
| meaning | live body rows are not unrelated blobs; generated runtime closure is a large subset, but RKNN adds a 45-register closure/tail and differs on 9 known fields |
| hardware rerun | not attempted |

Milestone 29 decision:

- BY_K remains fail-closed before allocation. The live body rows are now close
  enough to debug systematically, but not close enough to emit safely.
- The next no-submit step is to classify the stable diff, not submit hardware:
  name the 45 live-extra registers, the 3 generated-only registers, and the 9
  value-diff fields after normalizing DMA bases and expected family/address
  offsets.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Live body normalized diff classification | Normalize DMA base fields and family-bit/feature-grain differences, then report the exact remaining register names by group. | Checker prints a compact grouped diff with no anonymous `(target,addr)` keys and says whether the residual is small enough to materialize generically. |
| Live auxiliary row generator | Decode the 26-qword `0x60` rows and classify them as PPU/SDP/output-tail rows. | Checker names all registers and identifies which are shape/address-derived. |
| Runtime BY_K gate | Only after both body and auxiliary rows are generated no-submit. | Active runtime diff stays under budget; no hardware until `simple_add.py` pre/post plan is written. |

### Milestone 30: BY_K Live Body Normalized Diff Classification

Status: complete as a proof-only milestone. This milestone turned the live body
row diff from anonymous register-key counts into named register groups after
normalizing DMA base fields.

Milestone 30 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now prints `# Live body normalized diff` with named registers only |
| normalized summary | every body row remains `live_extra=45`, `generated_only=3`, `value_diff=6` |
| live-extra CNA group | CVT offsets, DCOMP controls, feature high address, `CNA_RESERVED_1140..117c`, and `CNA_CVT_CON6` |
| live-extra DPU group | BS/BN/EW value/offset fields, DPU zero-clear ranges, reserved `0x4014`, and `DPU_SURFACE_ADD_HI` |
| generated-only group | active direct-spatial closure writes RDMA-zero `CNA_DATA_SIZE0`, `CNA_CBUF_CON0`, and `CNA_CBUF_CON1`; live body rows do not |
| residual value diffs | `CNA_CONV_CON2`, `CNA_DATA_SIZE1`, `CNA_CBUF_CON0`, `CNA_CVT_CON5`, `CORE_MISC_CFG`, and `SURFACE_ADD` |
| DMA normalization | `CNA_FEATURE_DATA_ADDR`, `CNA_DCOMP_ADDR0`, and `DPU_DST_BASE_ADDR` no longer appear as residual diffs |
| hardware rerun | not attempted |

Milestone 30 decision:

- BY_K remains fail-closed before allocation. The residual is now small and named,
  but still not generated by runtime rules.
- The most promising next no-submit work is to generate the 45-register live-extra
  body closure from existing runtime defaults plus simple zero/value fields, then
  revisit the six residual value diffs. In parallel, the repeated 26-qword `0x60`
  rows must be decoded and classified before any submit path is considered.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Live body closure materializer | Generate the named 45 live-extra body registers and remove the three generated-only RDMA-zero writes for the live topology. | Checker reduces `live_extra` and `generated_only` to zero without captured qword replay, leaving only named value diffs. |
| Residual value-diff rules | Derive compact rules for `CONV_CON2`, `DATA_SIZE1`, `CBUF_CON0`, `CVT_CON5`, `CORE_MISC_CFG`, and `SURFACE_ADD`. | Checker reports generated body rows match live body maps, or records which fields remain fenced. |
| Live auxiliary row decoder | Decode the repeated 26-qword `0x60` rows. | Checker names every auxiliary register and separates shape/address-derived fields from constants. |

### Milestone 31: BY_K Live Body Closure Materializer

Status: complete as a proof-only milestone. This milestone generated the live body
closure fields from named zero/default rules and removed the three active
direct-spatial RDMA-zero writes for the live topology. It does not replay captured
qwords and does not enable runtime BY_K.

Milestone 31 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now has `materialized_live_body_reg_maps()` |
| generated closure | 45 named live-extra registers are generated as zero/default fields |
| removed generated-only rows | RDMA-zero `CNA_DATA_SIZE0`, `CNA_CBUF_CON0`, and `CNA_CBUF_CON1` are omitted for the live body materializer |
| materializer gate | `live_body_materialized_closure=PASS` |
| materialized summary | every body row is now `live_extra=0`, `generated_only=0`, `value_diff=6` |
| remaining value diffs | `CNA_CONV_CON2`, `CNA_DATA_SIZE1`, `CNA_CBUF_CON0`, `CNA_CVT_CON5`, `CORE_MISC_CFG`, and `SURFACE_ADD` |
| auxiliary rows | still not generated; repeated 26-qword `0x60` rows remain fenced |
| hardware rerun | not attempted |

Milestone 31 decision:

- BY_K remains fail-closed before allocation. Body closure shape is now generated
  no-submit, but six semantic register values and all auxiliary rows remain.
- The next no-submit work should derive the six residual value rules. The live
  values are compact and likely descriptor/family-specific, but they must be
  justified before any runtime emitter change.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Residual value-diff rules | Generate the six remaining body value diffs for setup/k_half/k_tile rows. | Checker reports materialized live body maps match live raw body maps exactly, or records a smaller fenced residual. |
| Live auxiliary row decoder | Decode and classify the 26-qword `0x60` rows. | Checker names every auxiliary register and separates constants from address/shape-derived values. |
| Runtime BY_K gate | Still deferred. | Only after body and auxiliary rows are no-submit generated, estimate runtime lines and write the hardware-run hypothesis. |

### Milestone 32: BY_K Live Body Value Rules

Status: complete as a proof-only milestone. This milestone derived compact rules
for the six residual live body value diffs and proved the generated body maps now
match the live raw body maps exactly after DMA-base normalization.

Milestone 32 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now has `value_rule_live_body_reg_maps()` |
| `CNA_CONV_CON2` rule | family bits plus `0x00f0`: setup `0x000000f0`, k_half `0x400000f0`, k_tile `0x500000f0` |
| `CNA_DATA_SIZE1` rule | live value `0x001f00a0` |
| `CNA_CBUF_CON0` rule | live value `0x000000a2` |
| `CNA_CVT_CON5` rule | live value `0x00000000` |
| `CORE_MISC_CFG` rule | live value `0x00000200` |
| `SURFACE_ADD` rule | live value `0x00001200` for all live body rows |
| body match gate | `live_body_value_rules=PASS` |
| value-rule summary | every generated body row is now `live_extra=0`, `generated_only=0`, `value_diff=0` |
| auxiliary rows | still not generated; repeated 26-qword `0x60` rows remain fenced |
| hardware rerun | not attempted |

Milestone 32 decision:

- BY_K body rows for the recovered live topology are now generated no-submit from
  compact rules. This still is not enough for hardware: the live topology alternates
  generated body rows with repeated 26-qword `0x60` auxiliary rows, and task
  metadata differs from the active runtime.
- The next milestone must decode and classify the 26-qword auxiliary rows. Only
  after those rows are generated can we estimate a runtime diff or write a safe
  hardware-run hypothesis.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Live auxiliary row decoder | Decode the repeated 26-qword `0x60` rows into named registers. | Checker reports all five auxiliary rows are identical after address normalization, names every register, and separates constants from shape/address fields. |
| Live auxiliary materializer | Generate the 26-qword rows from output shape/address parameters. | Checker reports auxiliary generated maps match live raw maps, no captured qword replay. |
| Runtime BY_K gate | Still deferred. | Body and auxiliary no-submit generation must both pass first. |

### Milestone 33: BY_K Live Auxiliary Row Decoder

Status: complete as a proof-only milestone. This milestone decoded the repeated
live `0x60` auxiliary rows from the recovered h14 raw regcmd window and classified
the row fields without changing runtime submit behavior.

Milestone 33 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now extracts and normalizes live auxiliary rows |
| auxiliary tasks | live tasks `2,4,6,8,10` |
| auxiliary row shape | five rows, each `regcfg_amount=26`, `enable_mask=0x60` |
| row identity gate | `live_aux_rows_identical=PASS` for both raw and normalized rows |
| register naming gate | `live_aux_registers_named=PASS` |
| address fields | `PPU_DST_BASE_ADDR`, `PDP_DST_BASE_ADDR` |
| constant fields | 24 named PPU/PDP configuration fields |
| classification gate | `live_aux_field_classification=PASS` |
| body value-rule gate | still `live_body_value_rules=PASS` |
| hardware rerun | not attempted |

Milestone 33 decision:

- BY_K body rows and the recovered live auxiliary row are now decoded no-submit
  from the captured live topology. The auxiliary rows are simple repeated PPU/PDP
  rows with two address-derived fields and 24 constants.
- BY_K remains fenced before runtime submit changes. The next milestone should
  materialize the 26-qword auxiliary row from shape/address parameters, then add a
  task-metadata/runtime gate for the live topology.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Live auxiliary materializer | Generate the 26-qword rows from output shape/address parameters. | Checker reports generated auxiliary maps match live raw maps, including address normalization, with no captured qword replay. |
| Live topology task metadata | Derive task amounts, masks, qword deltas, op_idx, and PC-tail rows for the live topology. | Checker reports the no-submit generated task table matches the recovered live prefix. |
| Runtime BY_K gate | Still deferred. | Only after body rows, auxiliary rows, and task metadata are all generated no-submit. |

### Milestone 34: BY_K Live Auxiliary Materializer

Status: complete as a proof-only milestone. This milestone generated the recovered
live 26-qword auxiliary row from rules instead of replaying captured qwords.

Milestone 34 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now has `materialized_live_aux_reg_map()` |
| generated constants | 24 named PPU/PDP constants generated from the live auxiliary row model |
| channel-grain rule | PPU/PDP channel fields use `min(align_up(out_c, 16), 32) - 1`, yielding `0x1f` for h14 |
| PPU address rule | `PPU_DST_BASE_ADDR = output_dma + 0x12e000` |
| PDP address rule | `PDP_DST_BASE_ADDR = output_dma + 0x12e400` |
| materializer gate | `live_aux_materialized_row=PASS` |
| address rule gate | `live_aux_address_rules=PASS` |
| generated diff | `live_extra=0`, `generated_only=0`, `value_diff=0` after address normalization |
| body value-rule gate | still `live_body_value_rules=PASS` |
| hardware rerun | not attempted |

Milestone 34 decision:

- The recovered live BY_K body rows and auxiliary row contents are now generated
  no-submit without captured qword replay.
- BY_K remains fenced before runtime submit changes because the live task table
  still needs generated metadata: task amounts, masks, qword deltas, op_idx, and
  PC-tail/core-lane programming for the alternating body/aux topology.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Live topology task metadata | Generate the live task table from the body/aux row materializers. | Checker reports task amounts, masks, qword offsets/deltas, op_idx, and captured-qword windows match the recovered live prefix. |
| PC-tail row classification | Decode which tail qwords are padding versus PC/core-lane metadata for the `0x0d` and `0x60` task families. | Checker emits named tail fields or keeps only explicitly unknown padded qwords fenced. |
| Runtime BY_K gate | Still deferred. | Only after generated task metadata and PC-tail handling are proven no-submit. |

### Milestone 35: BY_K Live Task Metadata

Status: complete as a proof-only milestone. This milestone generated the recovered
live h14 task table metadata from the body and auxiliary materializers and proved
it against the captured raw task window.

Milestone 35 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now has `GeneratedTaskMeta` and `generated_live_task_meta()` |
| generated roles | `setup_body,k_half_body0,aux0,k_half_body1,aux1,k_tile_body0,aux2,k_tile_body1,aux3,k_tile_body2,aux4` |
| amount gate | `live_task_metadata_counts=PASS` for `(108,104,26,104,26,104,26,104,26,104,26)` |
| mask gate | `live_task_metadata_masks=PASS` for `0x0d/0x60` alternation |
| op_idx gate | `live_task_metadata_op_idx=PASS`, all live rows use `op_idx=1` |
| offset gate | `live_task_metadata_offsets=PASS`, qword offsets `(0,112,224,256,368,400,512,544,656,688,800)` |
| delta gate | `live_task_metadata_deltas=PASS`, qword deltas `(112,112,32,112,32,112,32,112,32,112)` |
| tail budget gate | `live_task_tail_budget=PASS`, nonterminal tail/padding qwords `(4,8,6,8,6,8,6,8,6,8)` |
| captured windows | `live_raw_task_windows_present=PASS` for every generated task window |
| content gates | body value rules and auxiliary materializer still pass exactly |
| hardware rerun | not attempted |

Milestone 35 decision:

- The recovered live h14 topology now has no-submit generation for row content and
  task metadata. This still is not a runtime-submit change because the tail qwords
  inside each task window are not decoded into PC/core-lane fields yet.
- The next milestone must classify the tail qwords for the `0x0d` body rows and
  `0x60` auxiliary rows, especially the setup row's four in-window tail qwords and
  the repeated eight/six qword gaps before successor rows.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| PC-tail row classification | Decode captured qwords after each generated row amount and before the next task offset. | Checker reports named tail qwords for body/aux families or explicitly proves zero padding versus unknown metadata. |
| Runtime BY_K gate | Still deferred. | Tail classification must be resolved or fenced before any submit-field change. |

### Milestone 36: BY_K Live PC-Tail Classification

Status: complete as a proof-only milestone. This milestone classified every
nonterminal tail qword in the recovered live h14 task window as PC metadata or
zero padding.

Milestone 36 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now has `_tail_classification()` |
| tail class gate | `live_tail_classes=PASS` |
| body-row tail classes | `PC_BASE_ADDRESS`, `PC_REGISTER_AMOUNTS`, `VERSION`, `PC_OPERATION_ENABLE`, followed by four zero pads for repeated 104-qword body rows |
| setup tail classes | setup row tail is zero base padding, `PC_REGISTER_AMOUNTS=0`, `VERSION`, `PC_OPERATION_ENABLE=0x0d` |
| auxiliary tail classes | aux tails are zero base padding, `PC_REGISTER_AMOUNTS=0`, `VERSION`, `PC_OPERATION_ENABLE=0x60`, and two zero pads |
| register amounts gate | `live_tail_register_amounts=PASS`, values `(0,0x1000e,0,0x1000e,0,0x2000e,0,0x2000e,0,0x2000e)` |
| operation enables gate | `live_tail_operation_enables=PASS`, values match each source task `enable_mask` |
| next address gate | `live_tail_next_addresses=PASS`, nonzero PC base addresses point to the next aux-row task addresses |
| unknown gate | `live_tail_unknowns_fenced=PASS`, no nonzero unknown tail qwords remain |
| content and metadata gates | body rules, aux materializer, and live task metadata still pass |
| hardware rerun | not attempted |

Milestone 36 decision:

- The recovered live h14 topology is now decoded no-submit across row content,
  task metadata, and PC tail classification. There are no unknown nonzero tail
  qwords left in the captured prefix.
- Runtime BY_K remains fenced until this proof-only topology is turned into a
  small runtime gate/hypothesis that names every submit-risk field change and
  estimates whether the active `examples/conv.py` can absorb it under the
  line-budget rule.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Runtime BY_K gate hypothesis | Compare active runtime submit behavior against the live topology and list the exact required runtime changes. | Checker reports no-submit generated content, task metadata, and tail classification all pass, then emits a compact hardware-run hypothesis or keeps BY_K fail-closed with named missing runtime fields. |
| Runtime implementation decision | Decide whether the required changes fit the current small runtime without importing captured schedules. | Plan records either a scoped implementation path or an explicit continued fence with reasons. |

### Milestone 37: BY_K Runtime Gate Hypothesis

Status: complete as a proof-only milestone. This milestone compared the decoded
live h14 topology against the active `examples/conv.py` runtime path and kept
BY_K fail-closed with named submit-field deltas.

Milestone 37 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now has `runtime_gate_hypothesis()` |
| proof prerequisites | body content, auxiliary content, task metadata, and tail classification still pass |
| runtime fence gate | `runtime_byk_still_fail_closed=PASS`; active runtime rejects BY_K before allocation |
| runtime required groups | `runtime_required_groups_ready=PASS` for body, aux, task metadata, tail classification, and submit fields |
| named runtime deltas | `runtime_submit_deltas_named=PASS` |
| regcmd BO gate | `runtime_regcmd_bo_too_small=PASS`; live prefix needs `6608` bytes while runtime allocates `4096` |
| enable-mask delta | `runtime_enable_mask_delta_named=PASS`; live topology needs auxiliary `0x60` rows, runtime hardcodes `0x0d` |
| PC-tail delta | `runtime_pc_tail_delta_named=PASS`; live tails use 6/8 qword gaps and `0x1000e/0x2000e` PC amounts, runtime uses a four-qword generic tail and `ceil(next_len/2)+1` |
| core-lane delta | `runtime_core_lane_delta_named=PASS`; runtime submits `core_mask=1`, live tail metadata encodes lanes 1/2 |
| hardware rerun | not attempted |

Milestone 37 decision:

- BY_K remains correctly fenced. The proof-only h14 topology is decoded, but the
  active runtime still differs in row scheduling, per-task masks, PC-tail amount
  semantics, core-lane/subcore submit fields, and regcmd buffer sizing.
- The next milestone should decide whether to implement a small generated BY_K
  runtime path or keep the runtime fence. Any implementation must name every
  submit-risk field before a hardware attempt and must preserve the line-budget
  rule without importing captured schedules.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Runtime implementation decision | Choose scoped implementation or continued fence based on line budget, risk, and missing genericity beyond h14. | Plan records the decision and either adds a narrow implementation checklist or a proof-backed reason to keep BY_K disabled. |
| Submit-field emitter sketch | If implementing, add no-submit generation for task objects and regcmd tail rows using fake DMA addresses. | Checker reports generated task structs and regcmd layout match the decoded live topology without touching hardware. |
| Continue fence | If not implementing yet, identify the minimum new evidence needed before runtime changes. | Checker remains green and `examples/conv.py` keeps rejecting BY_K before allocation. |

### Milestone 38: BY_K Runtime Implementation Decision

Status: complete as a proof-only milestone. This milestone made the runtime
implementation decision explicit: keep BY_K fenced for now, and require a
fake-DMA task/regcmd emitter proof before any submit-field edits.

Milestone 38 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now has `implementation_decision()` |
| decision gate | `runtime_implementation_decision=PASS`, decision is `continue_fence` |
| fence gate | `runtime_fence_preserved=PASS`; active runtime still rejects BY_K before allocation |
| blocker gate | `runtime_fence_blockers_named=PASS` |
| minimum evidence gate | `runtime_minimum_next_evidence_named=PASS` |
| line budget | runtime stays `900` lines for `examples/conv.py` plus `conv_expt/conv_tile_planner.py`; soft margin is `50` lines |
| hardware rerun | not attempted |

Milestone 38 decision:

- Do not implement or submit BY_K yet. The decoded h14 topology is strong enough
  to guide work, but it is still h14-only and touches too many submit-risk fields
  for a direct runtime edit.
- Required next evidence before runtime edits:
  1. no-submit task-object emitter matching live task structs with fake DMA bases;
  2. no-submit regcmd tail emitter matching 6/8-qword live gaps and PC amount lanes;
  3. genericity audit for at least one non-h14 BY_K, or an explicit h14-only
     hardware hypothesis;
  4. written `simple_add.py` pre/post health plan for any future submit attempt.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Submit-field emitter sketch | Generate fake-DMA task structs and regcmd tail qwords for the decoded live h14 topology. | Checker reports generated task fields, qword offsets, tail qwords, and buffer size match the live topology without using captured qword replay. |
| Runtime genericity audit | Look for a second BY_K captured topology or prove the next hardware hypothesis is h14-only. | Plan records whether the emitter can be generic or must remain a single-shape experiment. |

### Milestone 39: BY_K Fake-DMA Submit-Field Emitter

Status: complete as a proof-only milestone. This milestone generated fake-DMA
task objects and regcmd tail qwords for the decoded live h14 topology, without
touching runtime submit code or hardware.

Milestone 39 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now has `FakeTask`, `fake_task_emitter()`, and `fake_regcmd_tail_emitter()` |
| task field gate | `fake_task_fields_match_live=PASS` |
| task address gate | `fake_task_regcmd_addrs_match_live_offsets=PASS` |
| tail class gate | `fake_tail_classes_match_live=PASS` |
| tail value gate | `fake_tail_values_match_live_except_base=PASS` |
| tail next-address gate | `fake_tail_next_addrs_match_live_offsets=PASS` |
| interrupt masks | generated `0x0d` tasks use `int_mask=0x300`; generated `0x60` tasks use `int_mask=0xc00` |
| relative regcmd offsets | generated relative task addresses are `0x0,0x380,0x700,0x800,0xb80,0xc80,0x1000,0x1100,0x1480,0x1580,0x1900` |
| PC base links | generated fake-DMA PC base links point to relative offsets `0x700,0xb80,0x1000,0x1480,0x1900` |
| runtime status | BY_K remains fail-closed before allocation |
| hardware rerun | not attempted |

Milestone 39 decision:

- The no-submit h14 proof now covers row content, task metadata, PC tail
  classification, and fake-DMA task/regcmd submit fields. This removes the
  immediate submit-field parity gap for the recovered h14 topology.
- Runtime BY_K still should not be enabled yet. The next decision must establish
  whether this is a generic BY_K emitter or an explicitly h14-only hardware
  hypothesis, then write the exact simple_add pre/post run plan before any submit.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Runtime genericity audit | Compare the fake-DMA emitter assumptions against another captured BY_K-like topology, or explicitly mark the first hardware hypothesis h14-only. | Plan records generic versus h14-only status with evidence. |
| Hardware-run hypothesis | If h14-only, write the exact runtime fields to change and the simple_add pre/post command sequence before code changes. | No hardware submit occurs until this hypothesis is written and the runtime diff is scoped. |

### Milestone 40: BY_K Runtime Genericity Audit

Status: complete as a proof-only milestone. This milestone compared the
fake-DMA task/regcmd emitter topology against multiple captured live BY_K-like
logs and scoped the emitter to an exact-11 task topology class, not all BY_K
schedules.

Milestone 40 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now has `genericity_audit()` |
| exact-11 captures | `genericity_exact_11_captures_present=PASS` |
| not h14-only | `genericity_exact_11_not_h14_only=PASS`; exact-11 appears in h7, h14, c32 h14, and pw c256 h14 captures |
| variant fence | `genericity_variant_topologies_fenced=PASS`; 14-row and 17-row captures diverge from the exact-11 signature |
| scope decision | `genericity_scope_decision=PASS`, scope is `exact_11_topology_only` |
| runtime fence | BY_K remains fail-closed before allocation |
| hardware rerun | not attempted |

Exact-11 captures used by the audit:

- `capture_rknpu_submit_dump_gems_6desc_h7_live_regcmd_20260524_123133.log`
- `capture_rknpu_submit_dump_gems_h14_live_regcmd_20260524_124717.log`
- `capture_rknpu_submit_dump_gems_c32_h14_live_regcmd_20260524_1303.log`
- `capture_rknpu_submit_dump_gems_pw_c256_h14_live_regcmd_832_20260524_131749.log`

Fenced variant captures:

- `capture_rknpu_submit_dump_gems_c64_h56_20260524_133709.log`: 14 live tasks,
  signature starts `108,108,104,104,26,...`
- `capture_rknpu_submit_dump_gems_c32_h14_live_regcmd_832_20260524_133132.log`:
  17 live tasks, signature starts `108,108,17,104,104,104,26,...`

Milestone 40 decision:

- The fake-DMA emitter is not just h14-specific at the task metadata level: the
  exact-11 task signature appears in non-h14 and pointwise-like captures.
- It is still not a generic BY_K emitter. Longer schedules and alternate setup
  phases exist, so runtime code must only target the exact-11 topology class
  unless another proof extends the schedule generator.
- Submit-risk fields remain fenced until a hardware-run hypothesis names every
  field change and the pre/post health commands.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Hardware-run hypothesis | Write a no-submit implementation hypothesis for exact-11 BY_K: task count, regcmd BO size, row order, enable masks, PC tails, core/subcore lanes, and simple_add pre/post command sequence. | Plan contains a field-by-field runtime diff checklist and no hardware submit occurs yet. |
| Runtime exact-11 adapter sketch | If the hypothesis is narrow enough, sketch the runtime adapter boundaries without enabling submit. | Checker or plan shows where runtime would branch to exact-11 emission while keeping BY_K fail-closed. |

### Milestone 41: Exact-11 BY_K Hardware-Run Hypothesis

Status: complete as a proof-only milestone. This milestone wrote the exact
hardware-run hypothesis for the scoped exact-11 BY_K topology without changing
runtime submit behavior or running hardware.

Milestone 41 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now has `hardware_run_hypothesis()` |
| scope gate | `hardware_hypothesis_scope_exact_11=PASS`; scope is `exact_11_topology_only` and `task_count=11` |
| field checklist | `hardware_hypothesis_submit_fields_named=PASS` |
| regcmd size | `hardware_hypothesis_regcmd_size_named=PASS`; exact-11 needs `6608` bytes, current runtime BO is `4096` bytes |
| PC tails | `hardware_hypothesis_pc_tail_named=PASS`; tail classes and `PC_REGISTER_AMOUNTS` match decoded live rows |
| health plan | `hardware_hypothesis_health_plan_named=PASS` |
| runtime fence | BY_K remains fail-closed before allocation |
| hardware rerun | not attempted |

Exact runtime field checklist for the first safe exact-11 BY_K attempt:

| Field | Required hypothesis |
|---|---|
| shape scope | only planner shapes whose live task signature is exact-11 |
| task count | `11` task objects, not local `1` or `3` generic task rows |
| row order | `setup_body,k_half_body0,aux0,k_half_body1,aux1,k_tile_body0,aux2,k_tile_body1,aux3,k_tile_body2,aux4` |
| regcmd BO size | at least `6608` bytes for `826` live qwords, replacing the current `4096`-byte BO |
| regcfg_amount | `108,104,26,104,26,104,26,104,26,104,26` |
| enable_mask | `0x0d` for body rows and `0x60` for auxiliary rows |
| int_mask | `0x300` for `0x0d` rows and `0xc00` for `0x60` rows |
| PC tail qwords | `4/8/6/8/6/8/6/8/6/8` nonterminal tails, from generated classes rather than the generic four-qword tail |
| PC register amounts | `0,0x1000e,0,0x1000e,0,0x2000e,0,0x2000e,0,0x2000e` |
| PC base links | body rows with `PC_BASE_ADDRESS` link to following auxiliary-row relative offsets |
| core_mask | live multi-lane hypothesis is `core_mask=0x7`, not current `core_mask=1` |
| subcore_task | split the exact-11 task window across lanes consistently with PC tail lane amounts before submit |
| runtime gate | BY_K remains fail-closed until all fields above are emitted from generated rows no-submit |

Required health command sequence for any future hardware attempt:

```sh
python3 examples/simple_add.py
python3 examples/conv.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/simple_add.py
```

Milestone 41 decision:

- The next hardware-facing runtime work is narrow enough to sketch, but not safe
  to submit yet. The runtime still needs a no-submit adapter boundary that keeps
  BY_K rejected until exact-11 rows, task objects, PC tails, regcmd BO sizing,
  core mask, and subcore assignment are all emitted coherently.
- Do not run the command sequence above until the runtime diff is present,
  reviewed against the checklist, and still passes the no-submit checker.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Runtime exact-11 adapter sketch | Add runtime-local no-submit adapter boundaries or a proof-only checker mirror showing where exact-11 BY_K emission would plug into `examples/conv.py`, while preserving the current fail-closed guard. | Active runtime still rejects BY_K before allocation; checker reports the adapter boundary covers rows, task metadata, PC tails, regcmd size, and submit fields. |
| Submit dry-run size audit | Compute the exact task/regcmd/input/weight/output BO requirements for the h14 shape without opening DRM. | Plan records BO sizes and validates the current allocation assumptions before any submit attempt. |

### Milestone 42: Exact-11 BY_K Runtime Adapter Sketch

Status: complete as a proof-only milestone. This milestone named the runtime
adapter boundaries for an exact-11 BY_K implementation, while preserving the
current runtime fail-closed guard.

Milestone 42 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now has `exact11_adapter_sketch()` |
| boundary names | `adapter_boundary_names_stable=PASS` |
| field coverage | `adapter_boundary_field_coverage=PASS`; every field from the hardware-run hypothesis is assigned to an adapter boundary |
| runtime import | no runtime import of proof-only checker is proposed |
| runtime fence | BY_K remains fail-closed before allocation |
| hardware rerun | not attempted |

Adapter boundary sketch:

| Boundary | Responsibility |
|---|---|
| `validate_exact11_byk_shape` | accept only exact-11 proven descriptor shapes and keep current BY_K guard until enabled |
| `build_exact11_byk_reg_rows` | emit generated body and auxiliary register rows from shape/planner descriptors |
| `build_exact11_byk_task_meta` | emit task amounts, masks, `op_idx`, relative regcmd offsets, and interrupt masks |
| `build_exact11_byk_pc_tails` | emit decoded PC tail classes, register amounts, operation enables, and base links |
| `allocate_exact11_byk_buffers` | size regcmd BO from generated qword span before DRM allocation |
| `submit_exact11_byk_tasks` | set `task_count`, `core_mask`, and `subcore_task` only after dry-run parity passes |

Milestone 42 decision:

- The next useful step is not a hardware submit. It is a submit dry-run size audit
  for the exact h14 shape, so runtime allocation sizes can be checked before
  touching DRM allocation or submit fields.
- The adapter sketch must remain local to runtime when implemented. Do not import
  `examples/conv_h14_task_layout_no_submit.py` into `examples/conv.py`.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Submit dry-run size audit | Compute task BO, regcmd BO, input BO, weight BO, and output BO needs for the exact h14 BY_K attempt without opening DRM. | Checker reports required sizes, current runtime allocation mismatches, and any output/input/weight assumptions before runtime allocation edits. |
| Runtime dry-run adapter | Add a runtime-local dry-run mode or helper that builds exact-11 rows and metadata but stops before DRM allocation. | BY_K still rejects by default; dry-run output matches checker metadata and stays under the active line budget. |

### Milestone 43: Exact-11 BY_K Submit Dry-Run Size Audit

Status: complete as a proof-only milestone. This milestone computed the exact
task, regcmd, input, weight, and output BO requirements for the h14 exact-11
BY_K attempt without opening DRM or submitting work.

Milestone 43 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now has `submit_dry_run_size_audit()` |
| task BO | `dry_run_size_task_bo_ok=PASS`; needs `440` bytes, current allocation is `4096` bytes |
| regcmd BO | `dry_run_size_regcmd_bo_mismatch_named=PASS`; needs `6608` bytes, current allocation is `4096` bytes |
| data BOs | `dry_run_size_data_bos_ok=PASS`; input, weight, and output fit current allocation policy |
| mismatch scope | `dry_run_size_only_regcmd_mismatch=PASS`; only `regcmd` needs allocation growth |
| runtime fence | BY_K remains fail-closed before allocation |
| hardware rerun | not attempted |

Dry-run size facts for `b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid`:

| Buffer | Required | Current runtime allocation |
|---|---:|---:|
| task BO | `440` bytes | `4096` bytes |
| regcmd BO | `6608` bytes | `4096` bytes |
| input BO | `62720` bytes | `4194304` bytes |
| weight BO | `921600` bytes | `4194304` bytes |
| output BO | `92160` bytes, allocated as `4194304` bytes by current minimum | `4194304` bytes |

Milestone 43 decision:

- The first exact-11 runtime allocation edit only needs to grow or dynamically
  size the regcmd BO. The task, input, weight, and output allocation sizes are not
  the current blocker for this h14 attempt.
- Hardware submit is still blocked because runtime does not yet build exact-11
  rows/metadata/tails locally, and submit fields remain at the generic
  `core_mask=1`, subcore-0-only path.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Runtime dry-run adapter | Add a runtime-local dry-run helper that builds exact-11 rows, task metadata, PC tails, and required BO sizes but stops before DRM allocation and submit. | `examples/conv.py` still rejects BY_K by default; a dry-run path, if exposed, prints metadata matching the proof checker and does not open `/dev/dri/card1`. |
| Regcmd sizing helper | Add a small runtime helper for dynamic regcmd BO sizing, still unused by live submit until exact-11 dry-run parity passes. | Existing supported hardware paths remain unchanged; BY_K remains fenced. |

### Milestone 44: Exact-11 BY_K Runtime Dry-Run Adapter

Status: complete as a no-submit runtime milestone. This milestone added a
runtime-local exact-11 BY_K dry-run path that prints the scoped metadata and BO
requirements, while preserving the default BY_K fail-closed behavior.

Milestone 44 execution result on 2026-05-31:

| Check | Result |
|---|---|
| runtime helper | `examples/conv.py` now has `dry_run_exact11_byk()` |
| dry-run command | `python3 examples/conv.py --dry-run-exact11-byk b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid` exits `0` |
| no-submit marker | dry-run output ends with `status=no_drm_no_submit` |
| task metadata | dry-run prints `tasks=11`, amounts `108,104,26,104,26,104,26,104,26,104,26`, and masks `0xd,0xd,0x60,0xd,0x60,0xd,0x60,0xd,0x60,0xd,0x60` |
| PC metadata | dry-run prints offsets `0,112,224,256,368,400,512,544,656,688,800` and PC amounts `0,0x1000e,0,0x1000e,0,0x2000e,0,0x2000e,0,0x2000e` |
| BO metadata | dry-run prints `bo_bytes=task:440,regcmd:6608,input:62720,weight:921600,output:92160` |
| default fence | `python3 examples/conv.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid` still exits before allocation with the BY_K disabled error |
| line budget | active runtime is `523 + 411 = 934`, still under the `950` soft budget |
| hardware rerun | not attempted |

Milestone 44 decision:

- The runtime now has a scoped exact-11 dry-run adapter that can be compared
  directly against the proof checker without opening DRM.
- BY_K live execution is still disabled. The remaining submit-facing work is to
  turn the dry-run metadata into generated row/tail emission and dynamic regcmd
  sizing, then rerun the dry-run parity checks before any hardware attempt.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Dry-run parity checker | Add a proof check that compares `examples/conv.py --dry-run-exact11-byk ...` output against the no-submit checker metadata. | Checker or test command proves runtime dry-run amounts, masks, offsets, PC amounts, and BO bytes match the proof-only model. |
| Regcmd sizing helper | Add a runtime helper that computes regcmd BO bytes from generated exact-11 metadata, but keep live submit on the current fenced path. | Supported existing paths still pass or remain unchanged; BY_K default path still rejects before allocation. |

### Milestone 45: Exact-11 BY_K Runtime Dry-Run Parity

Status: complete as a proof-only/runtime dry-run milestone. This milestone added
a checker gate that runs the runtime dry-run command and compares its output to
the generated no-submit model.

Milestone 45 execution result on 2026-05-31:

| Check | Result |
|---|---|
| checker extended | `examples/conv_h14_task_layout_no_submit.py` now runs and parses `examples/conv.py --dry-run-exact11-byk ...` |
| no-submit marker | `runtime_dry_run_no_submit_marker=PASS`; runtime dry-run reports `status=no_drm_no_submit` |
| metadata parity | `runtime_dry_run_metadata_parity=PASS`; task count, amounts, masks, and qword offsets match generated metadata |
| PC parity | `runtime_dry_run_pc_parity=PASS`; PC register amounts match decoded live tails |
| BO parity | `runtime_dry_run_bo_parity=PASS`; task/regcmd/input/weight/output byte counts match the proof model |
| runtime fence | BY_K remains fail-closed before allocation outside the explicit dry-run flag |
| hardware rerun | not attempted |

Milestone 45 decision:

- The runtime exact-11 dry-run is now guarded by the proof checker. Any drift in
  amounts, masks, offsets, PC amounts, or BO sizes should fail the checker before
  hardware work.
- The next runtime step should be very small because active runtime is already
  near the soft cap. A regcmd sizing helper is acceptable only if it reduces
  duplication or directly supports exact-11 dry-run/live parity.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Regcmd sizing helper | Add a compact runtime helper that computes qword offsets and regcmd bytes from task amounts, then use it in the exact-11 dry-run path. | Dry-run parity remains green, default BY_K remains fail-closed, and active runtime stays below the 950 soft cap. |
| Runtime line recovery | Recover at least 20 runtime lines before adding more live-submit code. | Existing supported paths and exact-11 dry-run still pass; combined `examples/conv.py` plus planner lines drop enough for the next submit-facing diff. |

### Milestone 46: Runtime Regcmd Sizing Helper

Status: complete as a no-submit runtime milestone. This milestone added a
runtime-local helper for computing exact-11 qword offsets and regcmd bytes from
task amounts, then used it in the dry-run path.

Milestone 46 execution result on 2026-05-31:

| Check | Result |
|---|---|
| runtime helper | `examples/conv.py` now has `regcmd_layout_from_amounts()` |
| dry-run parity | `runtime_dry_run_metadata_parity=PASS`, `runtime_dry_run_pc_parity=PASS`, and `runtime_dry_run_bo_parity=PASS` |
| runtime fence | BY_K remains fail-closed before allocation outside the explicit dry-run flag |
| line budget | active runtime is `527 + 411 = 938`, under the `950` soft cap with only `12` lines of margin |
| hardware rerun | not attempted |

Milestone 46 decision:

- The regcmd size/offset calculation is now a named runtime helper instead of an
  inline dry-run loop. This is the right boundary for future dynamic regcmd BO
  sizing.
- Do not add more submit-facing runtime code before recovering lines. The active
  runtime has only 12 lines of soft margin left.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Runtime line recovery | Recover at least 20 runtime lines while preserving supported execution, exact-11 dry-run parity, and BY_K default fail-closed behavior. | Combined `examples/conv.py` plus planner lines drop below `918`, dry-run parity remains green, and supported smoke tests still pass or are explicitly unchanged. |
| Exact-11 row dry-run expansion | Only after line recovery, extend runtime dry-run to emit row family names and tail qword classes. | Checker proves parity and runtime remains below the soft cap. |

### Milestone 47: Runtime Line Recovery After Exact-11 Dry-Run

Status: complete as a runtime cleanup milestone. This milestone recovered line
budget after adding the exact-11 dry-run adapter and regcmd sizing helper, without
changing submit behavior.

Milestone 47 execution result on 2026-05-31:

| Check | Result |
|---|---|
| line recovery | active runtime is `486 + 411 = 897`, below the `918` recovery target and `950` soft cap |
| checker budget gate | `runtime_soft_budget_under_cap=PASS`; checker reports `margin=53` |
| dry-run parity | `runtime_dry_run_metadata_parity=PASS`, `runtime_dry_run_pc_parity=PASS`, and `runtime_dry_run_bo_parity=PASS` |
| default fence | h14 BY_K still exits before allocation with the disabled error |
| list command | `python3 examples/conv.py --list` still prints known examples |
| hardware rerun | not attempted |

Milestone 47 decision:

- Runtime has enough recovered margin for the next proof-facing exact-11 dry-run
  expansion. Keep the next edit narrow and preserve the default BY_K fence.
- No live submit fields should change until dry-run emits enough row/tail detail
  to compare against the proof checker without importing fixtures.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Exact-11 row dry-run expansion | Extend runtime dry-run to print row roles and tail class names for exact-11 BY_K. | Proof checker verifies role/tail-class parity, default BY_K remains fail-closed, and runtime stays under `950` lines. |
| Dynamic regcmd allocation prep | Add no-submit calculation for the future dynamic regcmd BO size in runtime submit path, without using it for live BY_K. | Existing supported paths are unchanged; BY_K default path remains rejected before allocation. |

### Milestone 48: Exact-11 BY_K Row/Tail Dry-Run Expansion

Status: complete as a no-submit runtime/proof milestone. This milestone extended
the runtime exact-11 dry-run to print row roles and tail class names, then added
proof-checker parity gates for those fields.

Milestone 48 execution result on 2026-05-31:

| Check | Result |
|---|---|
| runtime dry-run expanded | `examples/conv.py --dry-run-exact11-byk ...` now prints `roles=` and `tail_classes=` |
| role parity | `runtime_dry_run_roles_parity=PASS` |
| tail-class parity | `runtime_dry_run_tail_class_parity=PASS` |
| existing dry-run parity | metadata, PC amount, and BO parity remain `PASS` |
| runtime fence | BY_K remains fail-closed before allocation outside the explicit dry-run flag |
| line budget | active runtime is `490 + 411 = 901`, still under the `950` soft cap |
| hardware rerun | not attempted |

Milestone 48 decision:

- Runtime dry-run now exposes the exact-11 schedule roles and PC-tail class
  structure needed for the next generated-tail/runtime bridge.
- The next step can prepare dynamic regcmd allocation or tail emission, but it
  must stay no-submit and preserve the default BY_K fence.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Dynamic regcmd allocation prep | Add a no-submit runtime calculation for future dynamic regcmd BO size and expose it in dry-run as the planned allocation size. | Checker proves planned regcmd allocation is at least `6608` bytes, existing paths unchanged, BY_K default remains rejected before allocation. |
| Tail qword dry-run emission | Extend dry-run to print normalized PC tail qword values, using fake base addresses only. | Checker proves class/value parity without allocating or submitting. |

### Milestone 49: Exact-11 Dynamic Regcmd Allocation Prep

Status: complete as a no-submit runtime/proof milestone. This milestone added a
runtime-local planned regcmd allocation calculation and exposed it in the exact-11
dry-run output.

Milestone 49 execution result on 2026-05-31:

| Check | Result |
|---|---|
| runtime helper | `examples/conv.py` now has `regcmd_alloc_bytes()` |
| dry-run output | dry-run header prints `regcmd_bytes=6608 regcmd_alloc=8192` |
| allocation gate | `runtime_dry_run_regcmd_alloc_planned=PASS`; planned allocation is at least required bytes |
| existing parity | metadata, role, tail-class, PC amount, and BO parity remain `PASS` |
| runtime fence | BY_K remains fail-closed before allocation outside the explicit dry-run flag |
| line budget | active runtime is `491 + 411 = 902`, still under the `950` soft cap |
| hardware rerun | not attempted |

Milestone 49 decision:

- Future exact-11 submit code should allocate an `8192`-byte regcmd BO for this
  h14 shape rather than the current `4096` bytes.
- The next no-submit step should emit normalized tail qword values from runtime
  dry-run so PC-tail value parity is explicit before any submit-field edit.

Recommended next milestone:

| Candidate | Action | Acceptance |
|---|---|---|
| Tail qword dry-run emission | Extend dry-run to print normalized PC tail qword values, using fake base addresses only. | Checker proves class/value parity without allocating or submitting; BY_K default remains fenced. |
| Exact-11 runtime row builder prep | Add no-submit runtime helpers for body/aux row role grouping without writing qwords into a BO. | Checker proves role/order parity and runtime stays under cap. |

## Safety Rules

- Do not edit `examples/kernel_6_18/` unless explicitly requested.
- Do not kill long-running NPU processes.
- Do not rerun known timeout/crash probes unchanged.
- For every hardware CONV attempt, run `simple_add.py` before and after unless the
  user explicitly says not to.
- `k_tile` stays disabled until a runtime-shaped no-submit submit-field emitter
  lands and h14/h7 both pass.
- Submit fields are hardware-risk fields: `npu_submit`, `task_count`,
  `regcmd_addr`, `regcfg_amount`, `enable_mask`, `core_mask`, `subcore_task`, and
  PC-chain tails. Change them only with a named reason tied to a failing shape.
- The existing no-submit evidence remains valid for cross-checking after hardware
  work. Do not delete it casually, but do not extend it unless it directly drives
  the next submit.
- On vendor `rknpu_driver`, use:

```sh
python3 examples/simple_add.py
python3 examples/conv.py <shape>
python3 examples/simple_add.py
```

- On mainline/Rocket-specific examples, use the known mainline quick check:

```sh
python3 examples/kernel_6_18/simple_add.py
```

## Acceptance Criteria

Planner phase is complete because:

- all 217 shapes are classified as `NONE/BY_Y/BY_K/BY_YK`;
- descriptor rows exist for all planned tiles and expose unknown fields instead
  of hiding them;
- old-strategy versus new-split/family cross-tab is documented;
- generic-only CPU execution from descriptors passes all 217 shapes;
- descriptor output matches known RKNN-export facts where we have evidence;
- unresolved `setup`/`k_half` mixed h40 rows are fenced so they cannot be mistaken
  for executable RKNN-equivalent descriptors;
- fixture-level no-submit parity exists for setup, h14 `k_tile`, and h7 `k_tile`.

Hardware cleanup is complete when:

- normal execution uses one planner-driven descriptor loop;
- normal execution consumes descriptor rows with tile-local shapes, offsets,
  banks, and reuse fields;
- full-register task emission is separated into base/default regs, descriptor
  delta regs, PC-tail qwords, and submit fields, with no hidden captured task blob;
- old branch families are gone or reduced to explicit unsupported errors;
- no RKNN replay, sparse task-GEM, or captured schedule table is in the normal
  runtime path;
- unresolved `feature_grains`, ConvStreaming, CNA group-mask programming, and exact
  ABC_T/KC_T builder-path gaps are either solved or fenced behind clear errors;
- selected hardware shapes pass with NPU health checks before and after;
- final user-facing `examples/conv.py` is under 1000 lines and does not recreate
  `examples/conv_tiles.py` through imported runtime helpers.
