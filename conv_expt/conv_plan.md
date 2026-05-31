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
| `examples/conv_small_no_submit.py` | `conv.py`-shaped no-submit entry point. Treat as old scaffold now that `examples/conv.py` is planner-driven. |
| `examples/conv_tiles_no_submit.py` | Registry-driven hardware-cleanup scaffold. Useful as a fixture check, not as a delivery target. |
| `examples/conv.py` | Current planner-driven user entrypoint. It has one passing setup/NONE hardware CONV and fail-closed guards for pointwise-wide NONE, BY_K, BY_Y, and BY_YK. |
| `examples/conv_legacy.py` | Previous strategy-heavy entrypoint, moved aside during replacement. |
| `examples/conv_no_submit_closure.py` | Proof-only 108-qword setup full-register closure. It is useful for the current setup pass but must be folded, shrunk, or left out of the final runtime budget. |
| `examples/kernel_6_18/conv_mesa.py` | Clean Mesa/Rocket reference that passed the 217 shape suite for INT8-style Mesa semantics, but is not a working FP16 solution. |
| `examples/kernel_6_18/conv.py` | Mainline/Rocket FP16 attempt blocked by Rocket submit/UAPI translation gaps for direct spatial RKNN-style schedules. |
| `examples/kernel_6_18/conv_new.py` | Mainline/Rocket 217-shape source; do not edit unless explicitly requested. |

Current line-count facts from 2026-05-31:

| File | Lines | Runtime decision |
|---|---:|---|
| `examples/conv.py` | 539 | Active planner-driven user entrypoint. |
| `conv_expt/conv_tile_planner.py` | 410 | Acceptable small planner import if it does not grow. |
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
- The remaining risk is runtime emitter quality, not planner proof. Pointwise-wide
  `NONE` is now intentionally disabled after the c144 mismatch, and h14 `k_tile`
  timeout is the next hardware bug to fix.
- More evidence-only reports are now harmful unless they directly change the next
  hardware attempt.

Current conclusion:

- Planner proof is complete enough for cleanup: all 217 shapes classify as
  `NONE/BY_Y/BY_K/BY_YK`, generic-only CPU passes, RKNN evidence matches the
  known fixture families where we have mined rows.
- Hardware cleanup has shipped a small honest subset: the normal entrypoint is
  planner-driven, c32 setup/NONE passes, and unresolved families fail before
  allocation.
- The active runtime path is `examples/conv.py` plus
  `conv_expt/conv_tile_planner.py`. Proof scaffolds are not imported by the
  normal runtime path.
- The `<1000` line target is currently met for the active runtime path, but only
  barely enough for growth: `539 + 410 = 949` lines. Coverage expansion must
  replace tables and fences with generic code; it cannot append one-off shape
  entries, fixtures, or reports.

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
| `grain_bits` | low bits of `CNA_CONV_CON2`; keep explicit until formula is known. |
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
| `pointwise_y_tile_hardcoded` | Mostly old-strategy noise; current c144/c192 evidence says `NONE/setup` should work once the generic emitter is fixed. |
| `spatial_oc_serial` | Y/K descriptors; independent submits are not the RKNN model. |
| `depthwise_spatial_tiled` | Y/K descriptors with depthwise channel granularity. |
| `spatial_im2col` | Temporary fallback; RKNN/ONNC evidence supports direct spatial Y descriptors eventually. |
| `grouped_serial` | Per-group lowering plus the same planner inside each lowered group. |

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
| generic `NONE/setup` | c32 closure and one small direct-spatial shape pass; several nearby NONE shapes still fail | yes, but only as a batch-quality issue | run a small NONE batch, then make one generic emitter fix if failures cluster |
| pointwise-wide `NONE` | c144 failed after the weight-size fix | no | keep fail-closed until a mined RKNN/Mesa row identifies a new register group |
| `BY_Y/y_tile` | no hardware submit yet; missing Y-window feature/data/core/DPU fields are known | yes, next broad unlock after NONE | implement one generic Y-window overlay and test the c96 pointwise BY_Y candidate |
| `BY_K/k_tile` | h14 timed out after body parity and PC-core fix | no, unless a full-task diff reveals a concrete new delta | do offline full-task row diff first; do not submit unchanged |
| `BY_YK` / h40 / `k_half` | mixed-family semantics unresolved | no | keep fenced until `BY_Y` and `BY_K` work independently |

## Next Actions

### Stop Doing

- Do not add more markdown evidence reports before the next hardware fix.
- Do not expand `examples/conv_no_submit_materializer.py` unless the new fields
  are immediately consumed by `examples/conv.py` for a hardware attempt.
- Do not add another fixture unless it is the oracle for the current failing
  shape.
- Do not chase h40 `setup`/`k_half`, multicore, fusion, or broad VP sweeps before
  `NONE`, `BY_K`, and one `BY_Y` are decided.
- Do not keep both `conv_small_no_submit.py` and `conv_tiles_no_submit.py` in the
  mental delivery path. They are fixture/proof tools only.
- Do not grow `SHAPES` by adding dozens of literal dictionaries. A larger shape
  registry will quietly spend the remaining line budget without improving the
  emitter. Prefer a compact shape-name parser plus a tiny alias table.
- Do not block a broad next step on a parked failure. Pointwise-wide NONE, BY_K
  timeout, and BY_YK/h40 are parked until their next action is a concrete diff or
  generic overlay, not another blind probe.

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
- `examples/conv.py` now emits `CNA_WEIGHT_SIZE0=0x2800` and
  `CNA_WEIGHT_SIZE1=0x140` for c144, matching the packed weight buffer.
- This did not change hardware output, so the mismatch is not only the weight-size
  register group.

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
| `examples/conv.py` | 539 | hard cap stays `<1000`; practical cap is `<590` while the planner remains 410 lines |
| `conv_expt/conv_tile_planner.py` | 410 | freeze unless planner bug blocks a batch |
| active combined runtime | 949 | every new helper/table must pay for itself by deleting or replacing code |
| remaining safe slack | about 40 lines | use for generic parsing/emitter code, not shape dictionaries |

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

Recommended next milestone:

- **Milestone 7 should first recover line budget, then expand BY_Y or NONE.** The
  current combined runtime is 949 lines, so adding another family without deleting
  code violates the soft budget.
- The best next code target is to compact or generate the 108-reg setup closure,
  or move repeated register-packing boilerplate into a smaller table-driven form.
- After recovering at least 20 lines, choose either a second pointwise BY_Y shape
  or one clustered c32 NONE failure. Do not revisit BY_K hardware until the
  full-task row diff identifies a changed submit-layout hypothesis.

Milestone 7 proposed acceptance:

```sh
wc -l examples/conv.py conv_expt/conv_tile_planner.py
python3 examples/simple_add.py
python3 examples/conv.py b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid
python3 examples/conv.py conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1
python3 examples/conv.py conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1
python3 examples/simple_add.py
```

## Safety Rules

- Do not edit `examples/kernel_6_18/` unless explicitly requested.
- Do not kill long-running NPU processes.
- Do not rerun known timeout/crash probes unchanged.
- For every hardware CONV attempt, run `simple_add.py` before and after unless the
  user explicitly says not to.
- `k_tile` stays disabled until a targeted fix lands and h14/h7 both pass.
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
- unsupported `grain_bits`, ConvStreaming, CNA group-mask programming, and exact
  ABC_T/KC_T builder-path gaps are either solved or fenced behind clear errors;
- selected hardware shapes pass with NPU health checks before and after;
- final user-facing `examples/conv.py` is under 1000 lines and does not recreate
  `examples/conv_tiles.py` through imported runtime helpers.

# extract notes

Rockchip TRM (local experimental/rk3588_trm.md, line 2247)
The actual TRM bit layout for RKNN_cna_conv_con2 at offset 0x1010:
Bit     Attr  Field            Description
31:24   RO    reserved
23:16   RW    kernel_group     Kernels per group (int16/fp16: 16 kernels per group)
15:14   RO    reserved
13:4    RW    feature_grains   Feature rows to buffer. Suggested: y_stride+kh+1
3       RO    reserved
2       RW    csc_wo_en       Weight scan enable (0=enable)
1       RW    csc_do_en       Data scan enable  (0=enable)
0       RW    cmd_fifo_srst   Command FIFO soft reset (debug only)
NVDLA HW (nvdla/hw via DeepWiki)
D_FETCH_GRAIN at offset 0x5060 — a single field register, GRAINS at bits [11:0] (12 bits). No kernel_group, no reserved upper bits, no CSC enable bits. NVDLA keeps this register pure: just a grain count.
NVDLA SW (nvdla/sw via DeepWiki)
fetch_grain is calculated by the compiler, passed as dla_conv_op_desc.fetch_grain, then decremented by 1 in the KMD before the hardware write. The KMD writes SHIFT(fetch_grain - 1, GRAINS).
Your earlier bits 3:0 claim was wrong — that came from reverse engineering, not the TRM.
The correct mapping:
Observed RKNN value	Family	TRM kernel_group field	TRM feature_grains field
0x00000110	setup	0x00 (0)	0x011 = 17
0x40000110	k_half	0x40 (64)	0x011 = 17
0x500000f0	k_tile	0x50 (80)	0x00f = 15
0x00000320	setup (trace)	0x00 (0)	0x032 = 50
So the descriptor family (setup/y_tile/k_half/k_tile) is encoded in the kernel_group field (bits 23:16), not bits 3:0. And our code's grain_bits concept is really the feature_grains field (bits 13:4), shifted by 4. The bits 3:0 are just CSC control bits and a reserved debug field — nothing to do with descriptor family.


0x50 = 80 decimal = 0101 0000 binary.
It is not a kernel group count despite the TRM's kernel_group field name. The TRM says kernel_group = (total_kernels / kernels_per_group) - 1, but these values don't satisfy that formula for any of the shapes. For example, c32 setup with oc=128 would need (128/16) - 1 = 7, not 0x00.
RKNN repurposes this byte as a descriptor family opcode:
Value
0x00
0x20
0x40
0x50
The encoding pattern:
- bit 5 (0x20) → "Y dimension split"
- bit 6 (0x40) → "K dimension split"  
- bit 4 (0x10) → "fine-grained tile" (distinguishes halves vs 3+ tiles)
So 0x50 = "K split, fine-grained" = k_tile.
The hardware may not even use this as kernel_group for CBUF weight banking — that is handled by CNA_CBUF_CON0 (bits 3:0 = weight banks, bits 7:4 = data banks). The bits 23:16 byte is better understood as a pipeline discriminator: the CSC/CORE stages need to know the descriptor family to apply different output-channel accumulation and weight-reuse semantics per-tile.