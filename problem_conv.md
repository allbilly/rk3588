# conv.py — Readability & Correctness Problems

## Problem 1: Magic-Dimension Special Cases

**Where:** `should_use_nhwc_pack` / `is_131128_3133_g3` (line 103–110), `input_pack_c2` overrides (lines 293–297), `run_conv2d` dimension-encoded booleans (lines 258–263)

**Root cause:** Experimentally discovered hardware behaviors encoded as dimension tuples with zero rationale.

**Fix:**
- Replace opaque names like `is_131128_3133_g3` with named constants like `DEPTHWISE_3x3_G3_ERRA_1` and a comment referencing what goes wrong without them (e.g., "without this, NPU produces garbage output — suspected DMA alignment quirk")
- Move all overrides into a single lookup table keyed by `(batch, in_c, in_h, in_w, out_c, kh, kw, groups)` with a rationale string, instead of scattered `if` statements

## Problem 2: Weight Packing with Boolean Flags

**Where:** `pack_conv_weights_fp16` — `use_kh_major` condition (lines 162–168) enumerates specific dimension tuples

**Root cause:** Each new RKNN dump reveals a different weight layout with no documented pattern.

**Fix:**
- Extract each packing variant into a named function (`pack_dw_spatial_major`, `pack_kh_major`, `pack_default`)
- Add a comment per variant documenting *which model/layer* produced the reference layout ("observed in YOLOv5s layer 17 RKNN dump")
- Prefer a dispatch table `(out_c, in_c, kh, kw, groups) -> pack_fn` over boolean flags

## Problem 3: Parameter Computation and Override Interleaving

**Where:** `compute_conv2d_params` mutates `width_stride`, `out_width_stride`, `align_c` mid-function (lines 258–276, 293–297)

**Root cause:** Default computation happens first, then special cases patch the result inline.

**Fix:**
- Two-phase approach:
  1. `_compute_default_params()` — returns default values
  2. `_apply_overrides(params)` — returns overridden copy
- Makes it explicit which values come from the general algorithm vs. a special case

## Problem 4: Implicit Slicing in `run_conv2d`

**Where:** Lines 463–478 — when `is_1x1 && in_channels >= 5`, silently slices input channels into groups of 4 and sums results

**Root cause:** 1x1 conv with >4 input channels requires channel-wise tiling due to HW limitation (non-aligned DMA max width).

**Fix:**
- Rename the intent: `_run_conv2d_with_channel_slicing` or at minimum add a docstring explaining *why* (HW max 4 in-channels for non-aligned DMA)
- Expose it explicitly in the public API rather than hiding inside `run_conv2d`

## Problem 5: No Validation Against Reference

**Where:** Everywhere — special cases are never checked against ground truth.

**Fix:**
- Add a `--validate` mode that runs the same conv2d through a CPU reference and reports mismatches per shape
- Would catch when a new weight-packing variant is needed or an old one is stale

## Problem 6: Non-1x1 Kernels Produce Partial Output

**Where:** `test_conv.py` line 7 — known hardware limit documented in test

**Description:** All non-1x1 kernels emit sparse output — only some spatial positions are non-zero vs. the expected dense result. The test treats this as WARN, not FAIL: prints `nz=<nonzero_count>/<expected_count>`.

**Affected shapes (12 of 20 test cases):**
- 3x3: (4,4,3,3,9x9), (16,16,3,3,9x9), (2,4,3,3,6x6), (1,6,3,3,5x7), (16,16,3,3,18x18)
- 2x2: (2,4,2,2,5x5)
- 5x5: (1,32,5,5,10x10)
- 4x4: (8,4,4,4,10x10)
- Depthwise 3x3: (3,3,3,3,11x28,g=3)
- Non-square: (1,6,3,1,5x7), (3,6,1,3,5x5)

**Working shapes (8 of 20):** All 1x1 kernels:
- (2,2,1,1,4x4), (1,6,1,1,4x4), (3,3,1,1,4x4), (4,2,1,1,4x4), (4,4,1,1,9x9), (8,8,1,1,5x5), (16,16,1,1,8x8), (16,16,1,1,32x32)

**Root cause:** Unknown — likely missing DPU kernel accumulation or CNA sliding-window configuration for `k_h > 1 || k_w > 1`.

**Fix:** Investigate whether the NPU requires a different CNA conv mode (`CNA_CONV_CON1_CONV_MODE`), additional accumulation registers, or a multi-pass approach for spatial kernels.

## Problem 7: Cross-Process Isolation Required

**Where:** conv and gemm cannot share a Python process — NPU state persists between separate `/dev/dri/card1` FDs even after `reset_npu()`.

**Fix:** Always run conv and gemm tests in isolated subprocess invocations. Document that mixing them in one process produces garbage.
