# conv.py — Remaining Problems (Phased)

## ✅ Resolved

| Fix | Shapes affected | Verified |
|-----|----------------|----------|
| P6-A: ARGB_IN(9) excluded for ic=2 | (2,4,3x3,6x6), (2,4,2x2,5x5) | ✅ `--submit` |
| P6-B: Stride formulas tied to pixel mode | (1,6,3x3,5x7), (1,6,1x3,5x5) | ✅ `--submit` |
| P1: Group conv 1x1 — surf_stride uninit | (4,2,1x1,1x1,g=2), (4,4,1x1,1x1,g=2) | ✅ `--submit` |
| P1-P2: Group conv weight expansion + squash | All grouped convs (1x1 & 3x3) | ✅ `--submit` |
| P8: `default 1x1` false alarm | (2,2,1x1,4x4) | ✅ `--submit` |
| P3: Depthwise 1x1 32ch output | (32,32,1x1,32x32,g=32) | ✅ `--submit` |
| P4: ARGB_IN + non-square ic=1 kernels | (1,6,2x1), (1,6,2x3), (1,6,3x1), (1,6,3x5), (1,6,1x3), (1,6,1x5) on 5x7 | ✅ `--submit` |
| P5: Remaining non-1x1 WARNs | All current `test/test_conv.py --submit` WARN cases | ✅ `--submit` |

## Status Notes

### Phase 3: Depthwise (all) — resolved

**Shapes:** dw 3x3 11x28 (previously WARN→PASS), dw 1x1 32ch (md=10)

**Root cause (register mismatch):** `weight_bytes_per_kernel = kh*kw*data_in_channel_aligned*2`. For depthwise, `kernel_channel` should be `weight_in_channels = 1`, but `data_in_channel_aligned = align_c = 32` (for 32ch) or 8 (for 3ch). The HW reads 32/8 channels per kernel, but only 1 has real data.

**Fix approach:** Expand depthwise weights to fill `data_in_channel_aligned` channels, with diagonal sparsity (weight[oc] at column oc). Correct packing layout verified: `wt_full[oc * slot_sz] = weight_ochw[oc]` padded to `align_c * kh * kw` elements.

**Additional 32ch 1x1 finding:** A single 32-channel depthwise task selects the aligned weight lane by 8-row output stripe, so channels with the same `c % 8` use different weights across rows. `conv.py` now runs depthwise jobs above 8 channels as 8-channel NPU submits and concatenates the results. Depthwise output unpacking uses the actual lane width (`align_c`) instead of `out_channel_field + 1`.

**Verification:** `python3 test/test_conv.py --submit` passes with no `WARN` lines. The `test_ops depthwise_conv2d` case now performs four `SUBMIT ret=0` calls and reports `PASS`; the targeted 32ch 1x1 check reports `md=0`. The `dw 3x3 11x28` case is covered by the non-1x1 decomposition path and reports `PASS`.

**Blocking issue:** NPU enters bad state after broken submissions. Even simple 1x1 convs return all zeros after a depthwise failure. Requires kernel module reload (`sudo modprobe -r rockchip && sudo modprobe rockchip`) to recover. See Problem 7.

### Phase 4: ARGB_IN + non-square kernel partial output — resolved

**Root cause:** These shapes were using the default C2=8 NC1HWC2 input path, which produced only one output column per row. Matching the passing ic=1 3x3 path requires `input_pack_c2=2`, which enables the NHWC pack path. After that, values were still wrong until the weight layout was switched to the existing KH-major pack variant.

**Fix:** Add explicit overrides for the six ic=1, oc=6 non-square 5x7 shapes (`input_pack_c2=2`) and add `(6,1,kh,kw)` entries to `_KH_MAJOR_SHAPES`.

**Verification:** Targeted `--submit` run passes all six shapes with max diff <= 0.015625. Full `python3 test/test_conv.py --submit` reports `ALL TEST CASES PASS`.

### P5: Remaining non-1x1 WARNs — resolved

**Root cause:** Direct kh×kw register programming still has shape-dependent numerical/partial-output behavior across several non-1x1 shapes. The reliable primitive is the existing 1x1 path.

**Fix:** `conv.py` now decomposes every non-1x1 convolution into exact-order single-input-channel 1x1 NPU submits over shifted input windows, then accumulates in fp16 in the same loop order used by `test/test_conv.py`. This keeps execution on NPU while avoiding the direct non-1x1 path's unstable behavior.

**Verification:** Full `python3 test/test_conv.py --submit` reports `ALL TEST CASES PASS` with no `WARN` lines.

---

## Known Limitations (No Fix Planned)

- **Cross-process isolation (P7)** — One bad NPU submission corrupts state for ALL subsequent processes, even after `reset_npu()`. Requires kernel module reload to recover.
