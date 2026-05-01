# conv.py — Remaining Problems

## Problem 6: Non-1x1 Kernels — Root cause found, fix pending test

**Status:** Both root causes identified via RKNN vendor register dump comparison + NVDLA C-model (deepwiki). Fixes applied in code but not yet tested with `--submit`.

### Root Cause A: CONV_CON1 ARGB_IN for in_c=2

conv.py sets `ARGB_IN(9)` + bits 30:29=3 for ALL `in_c ∈ {1,2,3,4}` (line 392). But the RKNN vendor library only sets these for `in_c ∈ {1,3,4}`. The HW rejects `ARGB_IN(9)` (value for in_c=2).

Deepwiki confirmed: `D_DATAIN_FORMAT_0` field selects FEATURE (NC1HWC2, value 0) vs PIXEL (NHWC interleaved, value 1). ARGB_IN + bits 30:29 map to this pixel/feature selection in the RK3588 adaptation.

**Fix** (applied): changed `in_c >= 1 and in_c <= 4` → `in_c in (1, 3, 4)` at line 392.

### Root Cause B: DMA stride formulas mismatch ARGB_IN mode

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

### Remaining failing shapes:
| Shape | nz/total | Root cause | Fix applied | Tested? |
|-------|----------|-----------|-------------|---------|
| (1,6,3x1,5x7) groups=1 | 18/126 | Bug B (stride mismatch when ARGB_IN active) | ✅ | ❌ |
| (2,4,3x3,6x6) groups=1 | 48/64 | Bug A (ARGB_IN(9) rejected for ic=2) | ✅ | ❌ |
| (2,4,2x2,5x5) groups=1 | 28/64 | Bug A (same) | ✅ | ❌ |
| (1,32,5x5,10x10) groups=1 | 864/1152 | Bug B (stride mismatch when ARGB_IN active) | ✅ | ❌ |

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
