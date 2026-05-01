# conv.py — Remaining Problems (Phased)

## ✅ Resolved

| Fix | Shapes affected | Verified |
|-----|----------------|----------|
| P6-A: ARGB_IN(9) excluded for ic=2 | (2,4,3x3,6x6), (2,4,2x2,5x5) | ✅ `--submit` |
| P6-B: Stride formulas tied to pixel mode | (1,6,3x3,5x7), (1,6,1x3,5x5) | ✅ `--submit` |
| P1: Group conv 1x1 — surf_stride uninit | (4,2,1x1,1x1,g=2), (4,4,1x1,1x1,g=2) | ✅ `--submit` |
| P1: Group conv 1x1 — weight expansion bug | Same + (3,6,3x3,5x7,g=3) | ✅ `--submit` |
| P8: `default 1x1` false alarm | (2,2,1x1,4x4) | ✅ `--submit` |

---

## Active Phases

### Phase 2: Grouped non-1x1 — weight broadcast

**Status:** ❌ Not started. Remaining: `_test_conv2d cin=3 3x3 g=3` with non-1x1. The 1x1 case was fixed in Phase 1; the non-1x1 case needs the same group-squash logic adapted for non-square outputs.

### Phase 3: Depthwise 1x1 32-group FAIL

**Status:** ❌ Not started.

**Shape:** `(32,32,1,1,32,32,32)` — `test_depthwise_conv2d`

**Error:** FAIL (1x1 error: md=27.3) — NPU produces wrong output for depthwise 1x1 with 32 groups.

**Diagnosis:** `is_depthwise=True` (groups=32 = in_c = out_c). The weight expansion for depthwise is correct (each oc maps to its own ic), but the output is garbage. Possible causes:
1. `align_c` uses `max_align=32` for depthwise, affecting data layout
2. The ABUF_IN (depthwise) bit at CONV_CON1 line 398 may need additional configuration
3. 32-channel depthwise with 32x32 input may exceed CBUF capacity or have wrong cbuf_entries

### Phase 4: ARGB_IN + non-square kernel partial output

**Status:** ❌ Not started.

**Shapes:** 6 shapes with ic=1 (ARGB_IN/pixel mode) and non-square kernels produce ~14-33% output. Square 3x3 with ic=1 **PASSES** after P6-B fix, so the issue is specific to pixel mode + non-square geometry.

---

## Known Limitations (No Fix Planned)

- **Non-1x1 kernels produce partial/numerical output** — Expected HW limitation. Tracked as WARN, not FAIL.
- **Cross-process isolation** — Run conv/gemm as separate processes. Workaround documented.
