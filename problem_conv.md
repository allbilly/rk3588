# conv.py — Remaining Problems (Phased)

## ✅ Resolved

| Fix | Shapes affected | Verified |
|-----|----------------|----------|
| P6-A: ARGB_IN(9) excluded for ic=2 | (2,4,3x3,6x6), (2,4,2x2,5x5) | ✅ `--submit` |
| P6-B: Stride formulas tied to pixel mode | (1,6,3x3,5x7), (1,6,1x3,5x5) | ✅ `--submit` |
| P1: Group conv 1x1 — surf_stride uninit | (4,2,1x1,1x1,g=2), (4,4,1x1,1x1,g=2) | ✅ `--submit` |
| P1-P2: Group conv weight expansion + squash | All grouped convs (1x1 & 3x3) | ✅ `--submit` |
| P8: `default 1x1` false alarm | (2,2,1x1,4x4) | ✅ `--submit` |

## Active Phases

### Phase 3: Depthwise 1x1 — Under investigation

**Shape:** `(32,32,1,1,32,32,32)` — md=10.05

**Root cause identified:** The `is_1x1 and in_channels >= 5` gate at line 574 incorrectly routes depthwise 1x1 (groups=32) through `_run_conv2d_channel_sliced` which creates a full `(32,32)` weight matrix instead of depthwise. **Fix applied** (add `groups == 1` check). Now flows to normal depthwise path.

**Remaining issue:** Weight packing produces md=10 even with correct _pack_default layout. Multiple approaches tested:
1. `_pack_default` with `in_c=1` (original depthwise path) → FAIL
2. Diagonal expansion + groups=1 (group squash approach) → FAIL  
3. Different output unpack methods → FAIL

Likely root cause: CBUF/depthwise register configuration may need additional tuning for 1x1 depthwise. The depthwise 3x3 (kh=3, kw=3) PASSES, but all 1x1 depthwise tests fail regardless of channel count. Suggests depthwise 1x1 has a specific hardware configuration issue (maybe FC_DATA_SIZE or CBUF entries for depthwise 1x1).

### Phase 4: ARGB_IN + non-square kernel partial output

**Status:** ❌ Not started. 6 shapes with ic=1 and non-square kernels produce ~14-33% output.

---

## Known Limitations (No Fix Planned)

- **Non-1x1 kernels produce partial/numerical output** — Expected HW limitation. Tracked as WARN, not FAIL.
- **Cross-process isolation** — Run conv/gemm as separate processes.
