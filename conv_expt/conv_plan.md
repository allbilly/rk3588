# conv_expt Active Plan

Date: 2026-05-31

## Goal

Ship a clean RK3588 FP16 CONV implementation with a user-facing
`examples/conv.py < 1000` lines.

The runtime must stay planner-driven:

```text
shape
  -> CBUF pressure
  -> NONE/BY_Y/BY_K/BY_YK
  -> descriptor rows
  -> generated regs/tasks
  -> submit
```

The old milestone log grew too granular and is now archived at
`conv_expt/conv_plan_archive_2026-05-31.md`. Treat that file as evidence, not the
active roadmap.

## Verdict

Progress is good, but the previous plan was too long and encouraged eternal
no-submit milestones.

What is good:

- `examples/conv.py` is already planner-driven.
- User-facing `examples/conv.py` is still well under the hard goal at about 530
  lines.
- Active runtime plus planner recovered line budget:
  about `530 examples/conv.py + 411 conv_tile_planner.py = 941` lines.
- Hardware coverage exists for c32 setup/NONE, one generic small NONE/setup path,
  and a useful pointwise BY_Y batch.
- Unsupported families fail before allocation; there is no CPU/GPU fallback.
- BY_K exact-11 now has a removed guarded attempt, a retained no-DRM dry-run, and
  an offline mismatch triage table.

What is not good:

- The old plan had 41 tiny milestones. That made the documentation look busy but
  did not force runtime decisions.
- BY_K still has no passing hardware result. Current no-submit evidence matches
  register rows, task layout, PC tails, and submit-lane fields, so the h14 output
  mismatch is no longer explained by the known submit metadata.
- More BY_K proof-only reports are not useful unless they identify a new mismatch
  class or produce a small guarded runtime diff.
- Non-pointwise BY_Y has executable h160 no-submit field parity, a generated
  no-DRM runtime dry-run, and first RKNN/GDB prefix replay evidence. Default
  local runtime hardware execution remains fail-closed before allocation until
  submit rows for the complete RKNN schedule are understood.
- Difficult tiled shapes should continue through RKNN/GDB incremental task replay
  and output dumps, not by adding longer local guarded retries.

## Current Runtime State

| File | Lines | Runtime decision |
|---|---:|---|
| `examples/conv.py` | ~530 | Active planner-driven user entrypoint. |
| `conv_expt/conv_tile_planner.py` | 411 | Active planner import; freeze unless it blocks a broad family. |
| `examples/conv_legacy.py` | 989 | Reference only. Do not revive strategies. |
| `examples/conv_tiles.py` | 2588 | Hardware/topology reference only. Do not import or copy tables. |

Supported in the active runtime:

| Family | Status |
|---|---|
| small `NONE/setup` | Supported for c32 closure and one generic sparse direct-spatial path. |
| pointwise `BY_Y/y_tile` | Supported for tested c64/c96/c112/c128/c144/c160/c192 variants. |

Still fenced before allocation:

| Family | Reason |
|---|---|
| pointwise-wide `NONE` | RKNN uses a multi-descriptor channel schedule; local one-row setup failed. |
| non-pointwise `BY_Y/y_tile` | h160 planner now emits the proven three setup windows `64/64/36`; `--dry-run-h160-spatial-by-y` reports those rows. The explicit guarded `--allow-h160-setup3-submit` path passes this one h160 shape; default hardware remains fenced. |
| `BY_K/k_tile` | exact-11 default remains fenced; guarded h14 submit ran and failed output parity with post-health PASS, so guarded submit code was removed and only no-DRM dry-run evidence remains. |
| `BY_YK` | Verified fail-closed for h40 mixed BY_YK; depends on K-family and mixed setup/k_half semantics. |

## Completed Gates

These are evidence gates, not more work to expand.

| Gate | Status | Result |
|---|---|---|
| A. BY_K adapter sketch | Done | Exact-11 row order, task metadata, PC tails, masks, offsets, regcmd sizing, `core_mask`, and `subcore_task` are named in `examples/conv_h14_task_layout_no_submit.py`. |
| B. BY_K dry-run sizing | Done | `examples/conv.py --dry-run-exact11-byk ...h14...` reports no-DRM sizes: task 440, regcmd 6608/alloc 8192, input 62720, weight 921600, output 92160. |
| C. BY_K guarded hardware attempt | Done, failed | h14 returned `max_diff=309.1068`; post-health `python3 examples/simple_add.py` passed; guarded submit code was removed. |
| D. BY_K generalization | Parked | No h7 run because h14 did not pass. |
| E. Spatial BY_Y shape work | Done no-submit | h160 checker generates RKNN-shaped `55/55/54` setup/k_half/y_tile useful-prefix rows. |
| F. Line-budget recovery | Done | Active runtime plus planner is about `941` lines, leaving room under the user-facing `conv.py < 1000` goal. |
| G. BY_K submit-lane parity | Done | h14 read-only ioctl evidence matches `core_mask=0`, `task_start=0`, `task_number=3`, and `subcore_task=(0,1),(0,1),(0,1),(0,0),(0,0)`. |
| H. Route decision | Done | Chose the preferred spatial BY_Y no-submit route; no optional BY_K guarded retry was reintroduced. |
| I. Spatial BY_Y executable no-submit | Done | `examples/conv_spatial_by_y_layout_no_submit.py` proves h160 executable field parity against RKNN emit for setup/k_half `64/64/36` windows and y_tile `55/55/54` windows, covering feature/output offsets and size/height registers. |
| J. Spatial BY_Y dry-run promotion | Done | `examples/conv.py --dry-run-h160-spatial-by-y ...h160...` emits generated 12-row h160 spatial BY_Y no-DRM runtime metadata; default h160 execution remains fail-closed before allocation. |
| K. Spatial BY_Y first hardware | Parked | RKNN emits PPU/PDP rows that should not be omitted and may need distinct enable/int/task metadata; current runtime does not safely emit those submit rows. |
| L. BY_K decision | Done | BY_K remains dry-run-only; no h14 guarded retry was reintroduced and h7 remains blocked. |
| M. BY_YK/k_half semantics | Done for now | BY_YK remains fail-closed; `b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid` was verified fenced before allocation. |
| N. 217-shape planner gate | Done | Planner classification over 217 known shapes passed with zero failures: `NONE=78`, `BY_Y=32`, `BY_K=72`, `BY_YK=35`. |
| P. RKNN prefix replay for difficult shapes | Done for h160 setup3 guarded path; broader BY_Y remains fenced | Added `conv_expt/gdb/rknn_prefix_replay.gdb`. h160 `KEEP_TASKS=1` and `KEEP_TASKS=2` land exactly on successive 62-row output boundaries, and h160 `KEEP_TASKS=3` passes the full RKNN output compare on a single patched subcore lane. The active planner now classifies h160 as three setup windows `64/64/36`. `examples/conv_h160_setup3_task_layout_no_submit.py` passes and names RKNN task fields, spans, PC tail amounts, dynamic patch fields, one-lane submit metadata, and a generated setup3 closure materializer. The materializer derives RKNN emit-name row lengths `53,49,16`, qword offsets `0,112,224`, spans `0x380,0x380,0x0c0`, PC amounts `55,10,0`, and exact non-address value parity for the first three setup rows. `examples/conv.py --allow-h160-setup3-submit ...h160...` now passes hardware with `max_diff=0.0310` and pre/post `python3 examples/simple_add.py` PASS. Default h160 remains fenced; only the explicit guarded one-shape flag runs this path. Artifacts live under `~/npu/ops_rknn/dump/prefix_h160_keep{1,2,3}_gem5/`. |

Conclusion: BY_K h14 failure is no longer explained by known register rows, task
layout, PC tails, BO sizing, or submit-lane fields. BY_K should not consume the
main path; use RKNN prefix replay to find the first divergent intermediate state.

## RKNN Incremental Task-Dump Route

This is now the preferred way to understand difficult tiled shapes that run too
long or need intermediate outputs. Let RKNN emit the known-good schedule, then
use GDB to submit only a prefix of the task stream and dump the intermediate
output. Repeat with prefixes `0..1`, `0..2`, `0..3`, etc. to map which row first
produces the expected tile and which row changes the output unexpectedly.

Existing tooling:

| File | Use |
|---|---|
| `conv_expt/gdb/rknn_prefix_replay.gdb` | Current GDB submit-prefix patcher. Configure with `KEEP_TASKS=N DUMP_GEM=5 DUMP_DIR=...`; default `PREFIX_MODE=linear` patches `subcore_task[0]` to `{0,N}`. Use `PREFIX_MODE=subcores` for RKNN schedules that distribute one task per subcore, such as h14/h7/c144/c192. It dumps the selected GEM at `rknn_destroy`. |
| `conv_expt/rknn_prefix_replay.py` | No-submit manifest for difficult-shape prefix replay commands and existing RKNN emit prefix order. |
| `examples/conv.py` | Local runtime now uses the same intermediate-output debugging style for generated tiles. The h40 pointwise `BY_YK` fix was found by checking each local tile output immediately after its NPU submit, mirroring RKNN prefix replay. |
| `ioctl.gdb` | Decodes `DRM_IOCTL_RKNPU_SUBMIT`; `KEEP_TASKS` controls the patched submit task count in `_patch_submit`. Uncomment `_patch_submit(regs["arg"])` only for intentional prefix runs. The helper writes `task_start=0`; its current print string says `start=1`, so trust the decoded submit after patch, not that message. |
| `patch_task_buffer.py` | Patches task/reg GEM buffers through `/dev/dri/card1`; supports task truncation, regcmd zeroing, and direct regcmd writes. |
| `conv2d_taskmodify.gdb` | Conv2d example that locates the task buffer at `rknn_run`, zeroes task descriptors after task 0, optionally zeroes regcmd ranges, and dumps GEM buffers before/after. |
| `dump.py` | Dumps GEM buffers after each patched run. |

First h160 hardware prefix results:

| Field | h160 result |
|---|---|
| Shape | `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` |
| Command form | `KEEP_TASKS=N DUMP_GEM=5 DUMP_DIR=... gdb -q -x /home/orangepi/rk3588/conv_expt/gdb/rknn_prefix_replay.gdb --args ./conv2d_multi --case ...` |
| Submit before patch | `task_start=0`, `task_number=9`, subcores `(0,3),(0,3),(0,3),(0,0),(0,0)` |
| Output GEM | `gem name: 5`, dumped at `rknn_destroy` |
| `KEEP_TASKS=1` | Submit after patch is `task_start=0`, `task_number=1`, subcores `(0,1),(0,0),(0,0),(0,0),(0,0)`. First compare mismatch is index `9796 = 158 * 62`. First raw GEM zero range starts at `0x00026440 = 158 * 8 * 2 * 62`. |
| `KEEP_TASKS=2` | Submit after patch is `task_start=0`, `task_number=2`, subcores `(0,2),(0,0),(0,0),(0,0),(0,0)`. First compare mismatch is index `19592 = 158 * 124`. First raw GEM zero range starts at `0x0004c880 = 158 * 8 * 2 * 124`. |
| `KEEP_TASKS=3` | Submit after patch is `task_start=0`, `task_number=3`, subcores `(0,3),(0,0),(0,0),(0,0),(0,0)`. Full RKNN output compare passed with `max error: 0.0389099`. The only large raw GEM zero range starts at `0x00618400`, exactly the output tensor end; trailing aligned padding is zero. |
| Health | pre/post `python3 examples/simple_add.py` passed for each prefix. |

Interpretation: `KEEP_TASKS=1` and `KEEP_TASKS=2` are not bad-output failures;
they stop RKNN after expected h160 Y boundaries and preserve NPU health.
`KEEP_TASKS=3` proves the first three h160 task descriptors are sufficient for a
single-lane correct final output. The planner has been fixed to name those as
the h160 setup windows, and the guarded local setup3 runtime path now emits the
RKNN-shaped `regcfg_amount=108,108,17` closure and passes hardware. Default h160
remains fenced until this rule is generalized safely.

Rules for this route:

- Prefer RKNN prefix replay over local `examples/conv.py` guarded submit when a
  shape is difficult, long-running, or needs intermediate state.
- Start from `KEEP_TASKS=1`; increase by exactly one task at a time.
- Keep `task_start=0` unless proving a separate start offset; patching start and
  count at the same time hides which row changed behavior.
- For each prefix, save submit metadata, task rows, regcmd ranges, input/weight
  dumps, and output dump. The useful artifact is the first prefix where output
  diverges from RKNN's full run or from the expected tile.
- Do not copy captured RKNN tables into `examples/conv.py`; use the dumps to
  derive descriptor rules for setup/k_half/y_tile/ppu/pdp rows.
- Do not kill long-running NPU processes. If a prefix run hangs, stop expanding
  the prefix series and run `python3 examples/simple_add.py` only after the
  process exits normally.

Initial target shapes for prefix replay:

| Shape | Reason |
|---|---|
| `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` | Spatial BY_Y prefix replay is complete through `KEEP_TASKS=3`; do not run longer h160 prefixes unless task/regcmd closure mapping exposes a concrete unanswered question. |
| `b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid` | BY_K h14 prefix replay is complete for the original three-task topology. The only timeout/failure mode found was the invalid linear patch for `KEEP_TASKS=2`; `PREFIX_MODE=subcores` fixes it. |
| `b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid` | Pointwise BY_YK prefix replay is complete for the original six-task topology: `KEEP_TASKS=1` is an expected partial output, `KEEP_TASKS=2` passes, and unpatched RKNN passes. |

Remaining difficult families and hardware-prefix backlog:

| Priority | Family | Shape(s) | Prefix-test status | What the prefix must answer |
|---:|---|---|---|---|
| 0 | non-pointwise spatial `BY_Y` task/regcmd closure | `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` | Done for one-shape guarded path. | Root cause was both closure shape and exact value parity. The passing guarded path emits `108,108,17`, spans `0x380,0x380,0x0c0`, rows `53,49,16`, subcores `(0,3),(0,0),(0,0),(0,0),(0,0)`, and RKNN-equivalent CBUF/grain/DMA/core/surface fields. Do not generalize beyond this shape without new no-submit value parity. |
| 1 | `BY_K/k_tile` exact-11 | `b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid` | Prefix replay done. | Original submit is three one-task subcores. Linear `KEEP_TASKS=2` was invalid and caused submit failure; `PREFIX_MODE=subcores` fixes the timeout/failure. Prefixes `1/2/3` then complete, dumps are byte-identical, and unpatched RKNN shows the same small strict-compare error (`0.0362344`) as the prefixes. |
| 2 | `BY_K/k_tile` small follow-up | `b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid` | Prefix replay done. | Generated RKNN model; original submit is three one-task subcores. Prefixes `1/2/3` pass with `max error=0.0666199`, dumps are byte-identical, and all health checks passed. |
| 3 | pointwise `BY_YK` | `b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid` | Prefix replay done. | Generated RKNN model. Original submit is six tasks, two per subcore. `KEEP_TASKS=1` is expected partial output; `KEEP_TASKS=2` on one lane passes with `max error=0.0236588`; unpatched RKNN passes. |
| 4 | spatial mixed `BY_YK` | `b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid` / `evidence_mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1` | Prefix replay done. | Generated RKNN model. Original submit is six tasks, two per subcore. `KEEP_TASKS=1` is expected partial output; `KEEP_TASKS=2` and unpatched RKNN both only fail the strict compare by small FP16 errors around `0.04`, so this is baseline RKNN behavior, not a prefix timeout. |
| 5 | pointwise-wide `NONE/setup` multi-descriptor channel schedule | `conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1`, `conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1` | Prefix replay done. | c144 existing model and c192 generated model both use three one-task subcores. Prefixes `1/2/3` pass for both shapes (`c144 max error=0.0257759`, `c192 max error=0.0311127`) with health checks passing. |
| 6 | broad direct-spatial `BY_Y` candidate | `b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid` | Prefix replay done. | Generated RKNN model. Original submit is 30 tasks, ten per subcore. Prefixes `1..10` all exited normally with health checks passing; `KEEP_TASKS=10` passes with `max error=0.0185509`, and unpatched RKNN also passes. |

This table is the complete active prefix-test backlog. Do not start broad sweeps;
add a new shape here only when it represents a new family rule or a blocker for
runtime promotion.

The same prefix/intermediate-output route now works for both RKNN and the local
runtime. For RKNN, `conv_expt/gdb/rknn_prefix_replay.gdb` truncates submits and
dumps GEM output. For `examples/conv.py`, tile-by-tile local submits expose the
same kind of intermediate output; this isolated the h40 pointwise `BY_YK` bug to
the first Y tile and fixed it by using the proven `make_y_tile_regs` path per OC/Y
tile. The local command now passes:

```sh
python3 examples/conv.py b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid
```

Local runtime promotion status after prefix/intermediate replay:

| Shape | Local `examples/conv.py` status |
|---|---|
| `b1_c128_h40_w40_oc40_wic128_k1x1_g1_s1_pvalid` | Passes via local tile replay, 4 one-task NPU submits, `max_diff=0.0156`. |
| `conv2d_b1_c144_h28_w28_oc32_wic144_k1x1_g1` | Passes via local row replay, 2 one-task NPU submits, `max_diff=0.0283`. |
| `conv2d_b1_c192_h28_w28_oc32_wic192_k1x1_g1` | Passes via local row replay, 2 one-task NPU submits, `max_diff=0.0286`. |
| `b1_c3_h320_w320_oc32_wic3_k3x3_g1_s1_pvalid` | Passes via local row replay, 10 one-task NPU submits, `max_diff=0.0114`. |
| `b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid` | Default path now uses the prefix-proven setup3 closure and passes with `max_diff=0.0310`. |
| `b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid` | Fixed. Default path now uses generated exact-11 BY_K body/aux rows and passes with `max_diff=0.0624`. |
| `b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid` | Fixed. Default path now uses the same exact-11 BY_K topology with h7 surface stride and passes with `max_diff=0.0621`. |
| `b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid` | Fixed. Default path now uses generated h40 exact17 setup/k_half/k_tile rows and passes with `max_diff=0.0625`. |

Latest dense-spatial debug update:

- `examples/conv.py --allow-exact11-byk-submit b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid` now emits the generated exact-11 body/aux rows, RKNN-style `108/104/26` amounts, masks, PC tails, 8192-byte regcmd BO, and `core_mask=0` / subcores `(0,1),(0,1),(0,1),(0,0),(0,0)`.
- The guarded h14 submit completes and post-health `python3 examples/simple_add.py` passes, but output still fails with the same `max_diff=309.1068`; per-OC debug shows every 32-channel group is wrong. This proves task topology parity is not sufficient.
- Default h14/h7/h40 behavior remains fail-closed before allocation. The h14 exact-11 path is opt-in only for continued diagnosis.
- Next suspect is dense-spatial data/weight/register semantic parity beyond the no-submit map: packed data/weight layout, PC lane execution semantics (`submit_task_number=3` versus the 11 task objects), or register ordering/duplicate writes.
- Prefix/output replay comparison completed: fresh RKNN h14 `KEEP_TASKS=1/2/3 PREFIX_MODE=subcores` dumps in `~/npu/ops_rknn/dump/prefix_h14_compare_keep{1,2,3}_gem5` are byte-identical to each other. Running our Python exact-11 submit with RKNN dumped packed input/weight buffers produces byte-identical packed output versus all three RKNN prefix dumps (`max=0 mean=0`). The remaining h14 local failure is therefore local random input/weight packing or CPU reference data parity, not submit/register replay.
- Dense-spatial weight packing fixed: spatial weights use RKNN's chunked layout by OC16, IC32, KH, KW. The generated h14 weight buffer now matches the RKNN weight dump byte-for-byte. h14 and h7 exact-11 BY_K pass by default.
- h40 fixed: quiet RKNN GEM1/GEM2 dumps showed the spatial h40 shape uses a 17-task topology with amounts `108,108,104,104,26,104,104,26,104,104,26,104,104,26,104,104,26`, masks `0x0d` for body rows and `0x60` for aux rows, original submit `task_number=6` and subcores `(0,2),(0,2),(0,2),(0,0),(0,0)`. Generated Python rows now match the RKNN regcmd dump for all 17 rows, and the default path passes with `max_diff=0.0625`.

## End-To-End Roadmap

The goal is not to prove every family before shipping. The goal is a clean
`examples/conv.py < 1000` that passes a useful broad set and fails closed for the
rest.

| Milestone | Deliverable | Exit rule |
|---|---|---|
| O. Release cleanup | Remove status noise, keep only descriptor-driven code paths, update docs. | `examples/conv.py < 1000`, no proof-helper imports, no captured tables, final checks pass. |
| P. RKNN prefix replay for difficult shapes | Use `conv_expt/gdb/rknn_prefix_replay.gdb` and RKNN `conv2d_multi` to run task prefixes and dump intermediate outputs. | Complete for all active backlog rows in this plan; remaining work is deriving runtime descriptor rules from the captured task/regcmd families. |

## Immediate Next Action

Milestone P hardware-prefix backlog is complete. Next work should mine the new
task/regcmd dumps into descriptor rules before adding runtime support; do not add
more prefix shapes unless they represent a new family rule.

Selected route completed:

- Spatial BY_Y executable no-submit rows now distinguish setup/k_half `64/64/36`
  windows from y_tile `55/55/54` windows.
- Field parity covers registers that distinguish real Y windows:
  `CNA_FEATURE_DATA_ADDR`, `CNA_DATA_SIZE0`, `CNA_DATA_SIZE3`, `CORE_SIZE0`,
  `DPU_DST_H`, `DPU_SIZE`, feature offsets, and output offsets.
- Guarded h160 setup3 hardware now passes behind `--allow-h160-setup3-submit`;
  default h160 remains fenced until this is generalized safely.

Next investigation route:

- Use `python3 conv_expt/rknn_prefix_replay.py` to print target-specific prefix
  order and exact `conv2d_multi --case ...` commands before touching GDB.
- For h160, stop increasing prefixes for now. `KEEP_TASKS=3` passes, the planner
  emits the three setup windows, and the guarded local setup3 path passes. Do not
  generalize h160 without a new no-submit exact-value parity gate.
- Dump output and GEM/task/regcmd metadata for each future prefix.
- Increase `KEEP_TASKS` by one only after checking health and interpreting the
  previous output boundary; preserve the original per-subcore topology when a
  shape uses one task per subcore.
- Bring only the inferred scheduling rule back to this repo.

Baseline acceptance before runtime promotion:

```sh
python3 examples/conv_h14_task_layout_no_submit.py
python3 examples/conv.py --dry-run-exact11-byk b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/conv.py b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/conv_spatial_by_y_layout_no_submit.py
python3 examples/conv_h160_setup3_task_layout_no_submit.py
python3 examples/conv.py --dry-run-h160-spatial-by-y b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid
python3 examples/conv.py b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid
python3 examples/simple_add.py
python3 examples/conv.py --allow-h160-setup3-submit b1_c16_h160_w160_oc128_wic16_k3x3_g1_s1_pvalid
python3 examples/simple_add.py
wc -l examples/conv.py conv_expt/conv_tile_planner.py
git diff --check -- conv_expt/conv_plan.md examples/conv.py examples/conv_spatial_by_y_layout_no_submit.py
```

Expected behavior for the non-flag BY_K and h160 BY_Y `conv.py` commands is
fail-closed before allocation.

## Stop Rules

- Do not add another milestone unless it moves one roadmap row toward runtime,
  executable no-submit parity, or a guarded hardware result.
- Do not append narrative evidence that only rephrases existing BY_K facts.
- Do not import proof-only helpers into `examples/conv.py`.
- Do not copy captured RKNN task rows into runtime.
- Do not run new h7 local guarded code until h14 BY_K runtime rules are derived;
  RKNN h7 prefix replay has already passed.
- Do not run h160 through default local `examples/conv.py` hardware submit until
  the missing broader schedule is modeled safely; the explicit setup3 guarded
  flag is the only passing h160 hardware path.
- Do not promote BY_K/BY_YK from RKNN prefix replay until the result is expressed
  as descriptor rules, not captured task tables.
- Do not spend line budget on shape dictionaries or status labels.
- Do not edit `examples/kernel_6_18/` unless explicitly requested.

## Hardware Safety

For every future CONV hardware attempt on vendor `rknpu_driver`:

```sh
python3 examples/simple_add.py
python3 examples/conv.py <shape>
python3 examples/simple_add.py
```

Submit-risk fields must be named before changing them:

```text
npu_submit, task_count, regcmd_addr, regcfg_amount, enable_mask,
core_mask, subcore_task, PC-tail qwords
```

Do not kill long-running NPU processes.

## Evidence Index

Use these files for details instead of growing this active plan:

| File | Use |
|---|---|
| `conv_expt/conv_plan_archive_2026-05-31.md` | Full historical milestone log through Milestone 41. |
| `conv_expt/conv_tile_result.md` | Offline planner reports and early no-submit fixture evidence. |
| `conv_expt/conv_evidence.md` | Consolidated evidence and external references. |
| `experimental/rknn/librknnrt_conv_channel_tile_decomp.md` | RKNN reverse-engineering details. |
| `examples/conv_h14_task_layout_no_submit.py` | Current BY_K exact-11 proof-only checker. |
| `examples/conv_h160_setup3_task_layout_no_submit.py` | Current h160 setup3 proof-only closure checker and generated closure materializer for RKNN task fields, emit-name row lengths, regcmd spans, PC tail amounts, dynamic patch fields, and one-lane submit metadata. |
| `examples/conv_spatial_by_y_layout_no_submit.py` | Spatial BY_Y h160 proof-only checker. |
| `examples/conv_output_bo_map_no_submit.py` | Large-output BO mapping checker. |
| `examples/conv_tiles.py` | Working hardware topology reference; never import into small runtime. |
| `~/npu/ops_rknn/ioctl.gdb` | GDB ioctl decoder and `KEEP_TASKS` submit-prefix patch point for RKNN runs. |
| `~/npu/ops_rknn/patch_task_buffer.py` | GEM task/regcmd patch helper for prefix replay and targeted regcmd writes. |
| `~/npu/ops_rknn/conv2d_taskmodify.gdb` | Existing conv2d task-buffer truncation and dump script. |
| `conv_expt/rknn_prefix_replay.py` | No-submit manifest for difficult-shape prefix replay commands and existing RKNN emit prefix order. |
| `conv_expt/gdb/rknn_prefix_replay.gdb` | Current hardware prefix replay GDB script using `KEEP_TASKS`, `PREFIX_MODE`, `DUMP_GEM`, and `DUMP_DIR`. |
| `~/npu/ops_rknn/dump/prefix_h160_keep1_gem5/` | First h160 hardware prefix replay artifacts: patched-submit log and output GEM dump. |
| `~/npu/ops_rknn/dump/prefix_h160_keep2_gem5/` | Second h160 hardware prefix replay artifacts: patched-submit log and output GEM dump; boundary at output row 124. |
| `~/npu/ops_rknn/dump/prefix_h160_keep3_gem5/` | Third h160 hardware prefix replay artifacts: patched-submit log and output GEM dump; full RKNN output compare passed. |
| `~/npu/ops_rknn/dump/prefix_h14_*gem5/` | h14 BY_K prefix artifacts; `prefix_h14_subcores{2,3}_gem5` are the fixed subcore-mode runs. |
| `~/npu/ops_rknn/dump/prefix_h7_*gem5/` | h7 BY_K prefix artifacts; generated model lives in `~/npu/ops_rknn/models/`. |
| `~/npu/ops_rknn/dump/prefix_h40_byyk_*gem5/` and `prefix_h40_spatial_byyk_*gem5/` | Pointwise and spatial h40 BY_YK prefix artifacts. |
| `~/npu/ops_rknn/dump/prefix_c144_h28_*gem5/` and `prefix_c192_h28_*gem5/` | Pointwise-wide NONE prefix artifacts. |
| `~/npu/ops_rknn/dump/prefix_h320_direct_by_y_keep{1..10}_gem5/` | Direct-spatial BY_Y prefix artifacts; `KEEP_TASKS=10` is the passing single-lane closure. |

## Current Sweep Snapshot

Latest full `examples/conv.py` 217-shape refresh:

```text
/tmp/opencode/conv_py_217_refresh_20260601_114043_summary.txt
pre_health_rc=0
post_health_rc=0
total=217 counts={'PASS': 103, 'FENCED': 114, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}
```

Newly verified against the current worktree:

- The local pointwise `BY_YK` replay family now passes for the eight manifest
  shapes: `b1_c160_h40_w40_oc40_wic160_k1x1_g1_s1_pvalid`,
  `conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1`,
  `conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1`,
  `conv2d_cc_b1_c128_h56_w56_oc128_wic128_k1x1_g1`,
  `b1_c32_h112_w112_oc64_wic32_k1x1_g1`,
  `b1_c64_h56_w56_oc128_wic64_k1x1_g1`,
  `b1_c128_h56_w56_oc128_wic128_k1x1_g1`, and
  `b1_c16_h150_w150_oc96_wic16_k1x1_g1_s1_pvalid`.
- The small grouped serial manifest shapes currently pass in the refreshed
  sweep, except the broader grouped/depthwise families remain fenced.
- Remaining uncovered buckets from
  `python3 conv_expt/rknn_prefix_replay.py --coverage-summary` are:
  `BY_K/k_tile=67`, `BY_YK/setup,k_half,k_tile=24`, `NONE/setup=9`, and
  `BY_Y/y_tile=2`.
- `replay_ready` hints remain only for exact-11-like BY_K evidence:
  `conv2d_cc_b1_c128_h28_w28_oc256_wic128_k1x1_g1` and
  `conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1`.
- Pointwise exact-11 BY_K diagnostic attempts are not promotable yet. The first
  task-topology match using setup/k_half/y_tile rows failed all four scoped
  shapes, and the pointwise register-value patch for `CBUF_CON0`, `DATA_SIZE1`,
  dynamic `CONV_CON2` grains, and `SURFACE_ADD` still failed. Artifacts:
  `/tmp/opencode/pointwise_exact11_byk_attempt_20260601_114512.txt` and
  `/tmp/opencode/pointwise_exact11_byk_patch_attempt_20260601_114610.txt`.
  Default pointwise BY_K remains fenced before allocation.
- `conv2d_b1_c32_h32_w32_oc32_wic1_k1x1_g32` was tested as a scoped 32-submit
  grouped/depthwise serial candidate and failed numerically with post-health
  PASS. Artifact: `/tmp/opencode/g32_pointwise_serial_20260601_114654.txt`.
  It remains fenced.
- Pointwise-wide `NONE` one-task local replay was probed for
  `b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid`,
  `b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid`, and
  `b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid`; all failed numerically with
  post-health PASS. Artifact:
  `/tmp/opencode/pointwise_none_local_probe_20260601_114832.txt`.
- Fresh RKNN evidence was generated for
  `b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid`. The model lives in
  `~/npu/ops_rknn/models/`; prefix artifacts are
  `/tmp/opencode/rknn_c128_h1_none_prefix_20260601_114936.txt`,
  `/tmp/opencode/rknn_c128_h1_none_taskdump_20260601_115006.txt`,
  `~/npu/ops_rknn/dump/prefix_c128_h1_none_keep1_gem5/`, and
  `~/npu/ops_rknn/dump/prefix_c128_h1_none_keep1_gem1/`.
  RKNN's original submit uses `task_number=3`, `core_mask=0`, subcores
  `(0,1),(0,1),(0,1),(0,0),(0,0)`. The decoded task descriptors are
  `108,108,108`, all `enable_mask=0x0d`, all pointing at the same regcmd
  address. A local three-lane same-regcmd probe timed out but recovered with
  post-health PASS:
  `/tmp/opencode/c128_h1_three_lane_probe_20260601_115048.txt`.
  Keep this shape fenced until the actual RKNN 108-row regcmd body is decoded.

## Success Criteria

The project is on track only while all of these remain true:

- `examples/conv.py < 1000` lines.
- Active runtime plus planner stays near the current `941` lines unless a broad
  class passes hardware; the next runtime feature should first recover line
  budget.
- New support is generated from descriptors/family rules, not copied capture
  tables.
- Unsupported families fail before allocation.
- Hardware attempts include pre/post `simple_add.py` health checks.
