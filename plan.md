# Plan: clean formula-driven `examples/conv_tiles.py`

## Objective

Build `examples/conv_tiles.py` as a clean downstream RKNPU CONV example with
the same style goal as `examples/gemm.py`: decoded registers, compact control
flow, one normal submit path, and no opaque hex blobs.

The direct-spatial CONV path should infer RKNN's tiling math from captured RKNN
templates. Captures are reference evidence and regression tests, not shape
dispatch tables.

## Current Problem

The current `conv_tiles.py` has drifted into an RKNN replay/debug harness:

- captured shape schedules are encoded directly;
- multiple diagnostic submit policies are live in the example;
- RKNN allocation/sync/action replay is mixed into normal runtime code;
- timeout/MMIO/status investigation is driving the file structure;
- `conv_new.py`-style multi-strategy fallback logic is still conceptually
  present, just with more direct-spatial capture special cases.

That is the opposite of the desired result. The target is one RKNN-like CONV
strategy, with formulas that generate tiles for shapes, not per-shape schedule
constants.

## Target Shape Of `conv_tiles.py`

Keep the file close to `gemm.py`:

- downstream ABI structs and ioctls;
- fixed BO allocation policy suitable for the example;
- one `npu_submit()` path;
- one `write_regs_to_npu_task()` path;
- CONV parameter math;
- input/weight packing;
- direct-spatial tile planner;
- register generation from decoded fields;
- output unpack/check code;
- small shape list for quick validation.

Move or keep outside `conv_tiles.py`:

- RKNN byte-for-byte replay modes;
- sparse task-GEM experiments;
- rejected submit policies;
- ACTION/MEM_SYNC reproduction experiments;
- MMIO/debug/status capture helpers;
- large capture parsers and offline forensic checks.

Those belong under `experimental/rknn/`.

## Design Rule

No direct-spatial shape schedule table in final `conv_tiles.py`.

Allowed constants:

- hardware constants: CBUF banks, bank size, atom sizes, alignment, PC tail
  qwords, register offsets;
- small universal thresholds if they are derived from hardware limits;
- comments explaining the RKNN-derived formula.

Not allowed in final `conv_tiles.py`:

- tables keyed by specific shapes such as C64/H56, H40, H7, C32/H14;
- record-count arrays copied from captures;
- per-shape active regcmd offsets/windows;
- env-selected replay policies;
- "tail-layout" or other crash/debug hypotheses.

If a formula cannot yet explain a captured schedule, keep that shape in
`experimental/rknn/` and do not promote it into the clean example.

## Formula Work

Derive and implement one planner that emits a list of tile descriptors:

```text
tile = {
  input_y,
  input_h,
  output_y,
  output_h,
  oc_start,
  oc_count,
  phase,
}
```

The formula should derive:

- output dimensions from kernel/stride/padding-valid math;
- aligned input/output channels;
- weight bank usage from `kh * kw * input_channels * oc_tile`;
- available feature banks after weight reservation;
- maximum output-row tile height from CBUF feature capacity;
- output-channel tile width from weight CBUF capacity;
- split mode from whether Y and/or K exceed one tile;
- tile order from RKNN evidence, but expressed as a deterministic ordering
  rule over computed Y/K partitions;
- setup/k-half/k-tile/y-tile phases from computed split mode, not shape names;
- PC links and task records from the generated descriptor list, not captured
  record arrays.

Use captured RKNN schedules only to validate the formula:

- H40 160->320 k3x3 valid;
- H7/H14 variants;
- C32/H14;
- C64/H56 pointwise;
- selected MobileNet-like pointwise/depthwise shapes.

Each capture should become an offline assertion:

```text
shape params -> formula schedule == captured schedule
```

If the assertion needs a shape-specific exception, the formula is not ready.

## Implementation Phases

1. Freeze current forensic work in `progress.md`.

   Do not continue appending runtime-debug history to `plan.md`.

2. Extract capture comparison helpers out of `conv_tiles.py`.

   Keep `experimental/rknn/compare_conv_tiles_offline.py` as the place for
   RKNN task/regcmd byte comparisons and MMIO/status parsing.

3. Create a clean planner interface inside `conv_tiles.py`.

   Start with pure functions:

   ```python
   conv_params(...)
   plan_direct_spatial_tiles(...)
   make_conv_regs(...)
   write_regs_to_npu_task(...)
   run_conv(...)
   ```

4. Replace shape-template schedules with formulas.

   Remove direct constants like H40/H7/C64_H56 schedule arrays from the example
   once equivalent formulas pass offline schedule checks.

5. Collapse submit policy to the `gemm.py` pattern.

   The clean example should use normal downstream RKNPU PC tasks. Any RKNN
   sparse replay remains experimental until it is understood and safe.

6. Keep validation small in the example.

   `conv_tiles.py` should run a short set of representative shapes. Broad RKNN
   capture regression belongs in `experimental/rknn/`.

## Acceptance Criteria

`examples/conv_tiles.py` is acceptable when:

- it is substantially closer to `gemm.py` in structure and readability;
- direct-spatial tiles are computed from formulas, not shape lookup tables;
- there is one normal submit path;
- `python3 -m py_compile examples/conv_tiles.py` passes;
- selected safe shapes pass on NPU and are bracketed by
  `python3 examples/simple_add.py`;
- offline RKNN schedule comparisons pass for promoted formulas;
- unsafe raw sparse-task probes are not required for normal operation.

## Safety Rules

- Do not edit `examples/kernel_6_18/` unless explicitly requested.
- Do not rerun unchanged raw sparse-task direct-spatial probes that previously
  timed out or crashed.
- Always run `python3 examples/simple_add.py` before and after any hardware
  CONV probe.
- Be conservative around `npu_submit`, `task_count`, `regcmd_addr`,
  `regcfg_amount`, and `enable_mask`.
- Do not kill long-running NPU processes.

## Immediate Next Step

Start by writing a small offline schedule-derivation script under
`experimental/rknn/` that compares formula output against the existing captured
RKNN schedules. Only after the formula explains the captures should
`conv_tiles.py` be simplified around that planner.
