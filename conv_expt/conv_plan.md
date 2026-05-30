# conv_expt Plan

Date: 2026-05-30

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

The target is a `<1000` line example. Do not clean the current 2k-line file just
for formatting. First prove the planner abstraction.

## Current State

Useful files:

| File | Role |
|---|---|
| `conv_expt/conv_tile_cpu.py` | Offline CPU proof harness. Already contains `_compute_k_step()`, `_compute_y_step()`, and `_plan_conv_tiles()`. |
| `conv_expt/conv_evidence.md` | Consolidated evidence: shape strategy counts, RK3588 CBUF constants, ONNC/NVDLA/RKNN findings, hardware proof boundaries. |
| `examples/conv_tiles.py` | Current hardware reference for vendor `rknpu_driver`; keep as reference until the planner is proven. |
| `examples/kernel_6_18/conv_new.py` | Mainline/Rocket 217-shape source; do not edit unless explicitly requested. |

Current conclusion:

- RKNN, ONNC, and NVDLA do not justify keeping six unrelated Python strategies.
- RKNN evidence points to one Y/K-style planner plus descriptor families,
  reuse/group details, bank fields, and register-emission modes.
- ONNC/NVDLA evidence confirms CBUF-fit-driven splitting, but `nv_full` has more
  CBUF than RK3588, so ONNC no-split results are not enough for RK3588.
- The immediate proof target is not just `split_method`; it is an explicit
  descriptor contract carrying tile-local shape, offsets, CBUF fields, and known
  unresolved values such as `grain_bits`.
- `conv_tile_cpu.py` is the right next workspace. Do not start by rewriting
  `examples/conv_tiles.py`.

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
| `pointwise_oc_tile` | closest to RKNN ChannelTile / `BY_K`; should become K-window descriptors, not naive OC slicing. |
| `pointwise_y_tile_hardcoded` | `BY_Y` or `BY_YK`, probably descriptor-family driven. |
| `spatial_oc_serial` | Y/K descriptors; current independent submits are not the RKNN model. |
| `depthwise_spatial_tiled` | Y/K descriptors with depthwise channel granularity. |
| `spatial_im2col` | temporary fallback; RKNN would program spatial conv directly or use ConvStreaming. |
| `grouped_serial` | per-group lowering plus the same planner inside each lowered group. |

## Next Actions

1. Add a planner-report mode to `conv_expt/conv_tile_cpu.py`.

    Report these fields for all 217 shapes:

    ```text
    name, old_strategy, split_method,
    y_boundaries, k_boundaries,
    descriptor_count,
    descriptor_families,
    unresolved_fields
    ```

2. Add a descriptor dump mode.

   Each descriptor row should include:

   ```text
   family, family_bits, grain_bits,
   y_start, input_h, output_h, output_w,
   k_start, oc_count,
   feature_off, weight_off, output_off,
   input_bank_num, weight_bank_num, cbuf0,
   data_reuse, weight_reuse
   ```

   Use `unknown`/`None` for fields that are not solved yet. Do not hide unknowns
   behind recomputed defaults.

3. Generate a cross-tab from old branches to new split methods and descriptor
   families.

    Required summary:

    | Split method | Descriptor families | Count | Old branches covered |
    |---|---|---:|---|
    | `NONE` | TBD | TBD | TBD |
    | `BY_Y` | TBD | TBD | TBD |
    | `BY_K` | TBD | TBD | TBD |
    | `BY_YK` | TBD | TBD | TBD |

4. Add a generic-only CPU execution mode.

   The current CPU harness still has old branch-specific execution paths. Add a
   mode that executes only the descriptor rows and prove whether all 217 shapes
   pass without old strategy loops. CPU execution should use descriptor-local
   `input_h`, `output_h`, `output_w`, `oc_count`, and offsets, not the old
   strategy branch names.

5. Compare descriptor output against known RKNN-export observations where
   evidence exists.

   Priority checks:

   | Evidence | Expected check |
   |---|---|
   | mixed `160->320 3x3` | independent Y/K windows, family order, additive output offsets. |
   | pointwise exported schedules | `y_tile` row descriptors and explicit `grain_bits` unknowns. |
   | spatial `k_tile` sweeps | OC windows and `cbuf0`/grain separation. |

6. Investigate failures by family.

    Priority groups:

   | Group | Why |
   |---|---|
   | 2 `pointwise_y_tile_hardcoded` shapes | rarest branch; should not justify a permanent special strategy |
   | 11 `spatial_im2col` shapes | worst fallback; materializes im2col and should become direct Y/K if possible |
   | large depthwise spatial | checks whether depthwise is a tile descriptor dimension, not a separate executor |
   | grouped spatial | should become group lowering plus generic planner |

7. Run targeted VP only if the planner/descriptor report leaves ambiguity.

    Do not run a broad VP sweep. Useful VP targets are listed in
    `conv_evidence.md`.

8. Only after offline proof, reduce hardware code.

   Preferred implementation path:

    1. Keep `examples/conv_tiles.py` as reference.
    2. Prove the descriptor planner and CPU execution in `conv_expt/conv_tile_cpu.py`.
    3. Port the generic descriptor loop to a new small hardware implementation or
       reduce `examples/conv_tiles.py` around it.
    4. Hardware-test selected shapes with `simple_add.py` before and after.

## Safety Rules

- Do not edit `examples/kernel_6_18/` unless explicitly requested.
- Do not kill long-running NPU processes.
- Do not rerun known timeout/crash probes unchanged.
- Do not casually change submit fields: `npu_submit`, `task_count`,
  `task_obj_addr`, `regcmd_addr`, `regcfg_amount`, `enable_mask`, `core_mask`,
  `subcore_task`, or PC-chain tails.
- Prefer offline planner work until the cross-tab is clear.
- For hardware CONV tests, use:

```sh
python3 examples/simple_add.py
python3 examples/conv_tiles.py <shape>
python3 examples/simple_add.py
```

On mainline/Rocket-specific examples, use the known mainline quick check:

```sh
python3 examples/kernel_6_18/simple_add.py
```

## Acceptance Criteria

Planner phase is complete when:

- all 217 shapes are classified as `NONE/BY_Y/BY_K/BY_YK`;
- descriptor rows exist for all planned tiles and expose unknown fields instead
  of hiding them;
- old-strategy versus new-split/family cross-tab is documented;
- generic-only CPU execution from descriptors passes all 217 shapes, or every
  failure has a clear planner/hardware reason;
- descriptor output matches known RKNN-export facts where we have evidence;
- rare and difficult groups are explained without hand-wavy special cases;
- remaining VP questions are reduced to a short target list.

Hardware cleanup starts only after the planner phase is complete.

Hardware cleanup is complete when:

- normal execution uses one planner-driven tile loop;
- normal execution consumes descriptor rows with tile-local shapes, offsets,
  banks, and reuse fields;
- old branch families are gone or reduced to small packing/materialization or
  explicitly temporary fallback cases;
- no RKNN replay, sparse task-GEM, or captured schedule table is in the normal
  runtime path;
- unresolved `grain_bits`, ConvStreaming, CNA group-mask programming, and exact
  ABC_T/KC_T emission gaps are either solved or fenced behind explicit fallback;
- selected hardware shapes pass with NPU health checks before and after;
- the final example is plausibly under the `<1000` line target.
