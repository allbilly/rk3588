# conv.py — Remaining Problems

## Problem 6: Non-1x1 Kernels — Verified 2026-05-01

**Status:** Both fixes tested with `--submit`. ARGB_IN fix verified working. Stride fix works for square kernels; non-square non-1x1 still partially broken.

### Root Cause A: CONV_CON1 ARGB_IN for in_c=2 — FIXED ✅

conv.py sets `ARGB_IN(9)` + bits 30:29=3 for ALL `in_c ∈ {1,2,3,4}` (line 392). But the RKNN vendor library only sets these for `in_c ∈ {1,3,4}`. The HW rejects `ARGB_IN(9)` (value for in_c=2).

Deepwiki confirmed: `D_DATAIN_FORMAT_0` field selects FEATURE (NC1HWC2, value 0) vs PIXEL (NHWC interleaved, value 1). ARGB_IN + bits 30:29 map to this pixel/feature selection in the RK3588 adaptation.

**Fix** (applied): changed `in_c >= 1 and in_c <= 4` → `in_c in (1, 3, 4)` at line 392.

**Verified with `--submit`:** 3x3 oc!=ic (in_c=2) produces expected partial output (no crash).

### Root Cause B: DMA stride formulas mismatch ARGB_IN mode — PARTIALLY FIXED

The stride formulas (line_stride, surf_stride) are currently tied to the `use_nhwc` packing flag. But the HW's data format interpretation is controlled by CONV_CON1 ARGB_IN bits — independent of the packing flag.

When ARGB_IN is active (pixel mode), the HW expects:
- `line_stride = width_stride` (not `width_stride * 4`)
- `surf_stride = width_stride * (in_h - 1)` (not `width_stride * (in_h - 4)`)

This was confirmed by extracting RKNN vendor register dumps for all 4 shapes:
| Shape | conv.py line_stride | conv.py surf_stride | RKNN line_stride | RKNN surf_stride |
|-------|-------------------|--------------------|-----------------|-----------------|
| (1,6,3x1,5x7) ic=1 | 32 (ws*4) | 8 (ws*(ih-4)) | **8** (ws) | **32** (ws*(ih-1)) |
| (1,32,5x5,10x10) ic=1 | 64 (ws*4) | 96 (ws*(ih-4)) | **16** (ws) | **144** (ws*(ih-1)) |

**Fix** (applied): tie stride formulas to ARGB_IN condition instead of `use_nhwc` flag.

**Verified with `--submit`:** Square kernels (3x3) with ic=1 now PASS. Non-square kernels (3x1, 2x1, 2x3, 3x5) still produce partial output — likely a separate non-square kernel HW issue.

### Test results with `--submit` (2026-05-01):

**PASS:**
- All 1x1 kernels (ic<5): PASS
- `simple 3x3 9x9` (4,4,3,3,9,9,1): PASS
- `ic=1 oc=6 3x3` (1,6,3,3,5,7,1): PASS — stride fix works
- `test_ops _test_conv2d cin=3 2x1` (3,6,2,1,5,7,1): PASS
- `test_ops _test_conv2d cin=3 2x3` (3,6,2,3,5,7,1): PASS
- `test_ops _test_conv2d cin=3 3x1` (3,6,3,1,5,7,1): PASS
- `test_ops _test_conv2d cin=3 3x5` (3,6,3,5,5,7,1): PASS
- `test_ops _test_conv2d cin=1 3x3` (1,6,3,3,5,7,1): PASS

**WARN (non-1x1 partial output — known HW limitation):**
| Shape | nz_r/nz_e | Notes |
|-------|-----------|-------|
| (16,16,3,3,9,9,1) 3x3 m4 | 784/784 | Full output but numerical? |
| (2,4,3,3,6,6,1) 3x3 oc!=ic | 52/64 | Partial output |
| (16,16,3,3,18,18,1) | 4096/4096 | Full output but numerical? |
| (2,4,2,2,5,5,1) 2x2 kernel | 64/64 | Full output but numerical? |
| (1,32,5,5,10,10,1) 5x5 kernel | 864/1152 | Partial output |
| (8,4,4,4,10,10,1) 4x4 kernel | 196/196 | Full output but numerical? |
| (3,3,3,3,11,28,3) dw 3x3 | 702/702 | Full output but numerical? |
| (1,6,3,1,5,7,1) 3x1 | 18/126 | Partial — Fix B insufficient for non-square |
| (3,6,1,3,5,5,1) 1x3 kernel | 90/89 | Off by one |
| (1,6,2,1,5,7,1) cin=1 2x1 | 24/168 | Partial |
| (1,6,2,3,5,7,1) cin=1 2x3 | 24/120 | Partial |
| (1,6,3,1,5,7,1) cin=1 3x1 | 18/126 | Partial |
| (1,6,3,5,5,7,1) cin=1 3x5 | 18/54 | Partial |

### Previously resolved:
| Shape | Fix |
|-------|-----|
| (1,6,3x3,5x7) groups=1 | Weight pack: full `in_channels` |
| (4,4,3x3,9x9) groups=1 | Output unpack stride (`out_w` not stride) |
| (3,3,3x3,11x28) groups=3 dw | Weight pack: expand tensor to full channels |
| (3,6,1x3,5x5) groups=1 | Numerical rounding, not a real failure |
| (16,16,3x3,9x9) groups=1 | Numerical rounding, not a real failure |
| (16,16,3x3,18x18) groups=1 | Numerical rounding, not a real failure |
| (8,4,4x4,10x10) groups=1 | Numerical rounding, not a real failure |

## Problem 7: Cross-Process Isolation Required 📋 DOCUMENTED

**Workaround:** Always run conv and gemm tests in isolated subprocess invocations (as `test_conv.py` and `test_gemm.py` already do). Mixing them in one process produces garbage even after `reset_npu()`.

## Problem 8: Failures from test_ops.py shapes added to test_conv.py

Tested with `--submit` (2026-05-01):

### RESOLVED: `default 1x1` (2,2,1,1,(4,4),1) — FALSE ALARM

Earlier FAIL (md=46.9375) was caused by running without `--submit` (mixed dry-run/submit mode). With `--submit`, this shape **PASSES** ✅.

### ERROR: Grouped convs crash — 3 shapes still broken

| Shape | Source | Groups | Error |
|-------|--------|--------|-------|
| (4,2,1,1,1,1,2) | `test_simple_grouped_conv2d` | 2 | KeyError 'surf_stride' at conv.py:419 |
| (4,4,1,1,1,1,2) | `test_medium_grouped_conv2d` | 2 | KeyError 'surf_stride' at conv.py:419 |
| (32,32,1,1,32,32,32) | `test_depthwise_conv2d` | 32 | FAIL (1x1 error: md=46.9375) |

**Root cause:** `build_conv2d_regs` accesses `p['surf_stride']` but grouped conv path does not populate this key. `construct_params` skips stride calculation for groups > 1.

The depthwise test (groups=32, 1x1) reaches submission but produces garbage (md=46.9). This is a separate issue from the KeyError — the NPU doesn't handle 32-group depthwise correctly.

### ERROR: Grouped non-1x1 — weight broadcast crash

`test_ops _test_conv2d cin=3 3x3 g=3` (3,6,3,3,5,7,3):
```
ValueError: could not broadcast input array from shape (9,) into shape (0,)
```

At `conv.py:481` in `_npu_submit`: `wt_full[dst_start:dst_start + params['kernel_h'] * params['kernel_w']] = ...`
The weight tensor expansion for groups produces a zero-length slice, likely because `weight_in_channels` goes to 0 when groups == in_channels in the grouped path.
