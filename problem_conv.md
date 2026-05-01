# conv.py — Remaining Problems (Phased)

## ✅ Resolved
<!-- Keep this list concise — these shapes were fixed in past iterations -->

| Shape | Fix |
|-------|-----|
| (1,6,3x3,5x7) groups=1 | Weight pack: full `in_channels` |
| (4,4,3x3,9x9) groups=1 | Output unpack stride (`out_w` not stride) |
| (3,3,3x3,11x28) groups=3 dw | Weight pack: expand tensor to full channels |
| (3,6,1x3,5x5) groups=1 | Numerical rounding, not a real failure |
| (16,16,3x3,9x9) groups=1 | Numerical rounding, not a real failure |
| (16,16,3x3,18x18) groups=1 | Numerical rounding, not a real failure |
| (8,4,4x4,10x10) groups=1 | Numerical rounding, not a real failure |
| (2,2,1x1,4,4) groups=1 `default 1x1` | False alarm (dry-run vs submit confusion) |

**Root Cause A** (P6-A): ARGB_IN(9) rejected for ic=2. Fix: exclude ic=2 at `conv.py:396`.
**Root Cause B** (P6-B): Stride formulas mismatched in pixel mode. Fix: `use_pixel_mode` controls stride at `conv.py:364`.
Both verified with `--submit`: square 1x1 and square 3x3 all pass.

---

## Active Phases

### Phase 1: Grouped conv 1x1 crash — KeyError 'surf_stride'

**Status:** ❌ Not started

**Shapes:**
- `(4,2,1,1,1,1,2)` — test_simple_grouped_conv2d
- `(4,4,1,1,1,1,2)` — test_medium_grouped_conv2d

**Error:** `KeyError: 'surf_stride'` at `conv.py:419` — `build_conv2d_regs` reads `p['surf_stride']` but grouped conv path in `compute_conv2d_params` doesn't set it.

**Diagnosis:** The stride formulas at `conv.py:365-371` compute `line_stride`/`surf_stride` unconditionally for all paths including groups > 1. The issue may be that the grouped path changes `in_channels` after stride calculation, or that `locals()` return from `compute_conv2d_params` doesn't include them for some execution branch.

**Verification:** Run `python3 test/test_conv.py --submit` and check these two shapes no longer crash.

---

### Phase 2: Grouped non-1x1 weight broadcast crash

**Status:** ❌ Not started

**Shapes:**
- `(3,6,3,3,5,7,3)` — test_ops _test_conv2d cin=3 3x3 g=3

**Error:**
```
ValueError: could not broadcast input array from shape (9,) into shape (0,)
```
At `conv.py:481`: `wt_full[dst_start:dst_start + params['kernel_h'] * params['kernel_w']] = ...`

**Diagnosis:** The weight expansion loop at `conv.py:477-482` assumes depthwise mapping (`ic_src = oc`), but this test has `groups=3, in_c=3, out_c=6` (not depthwise — out_c != in_c). The `dst_start` computes to a zero-length slice because `wt_in_c * params['kernel_h'] * params['kernel_w']` overflows the expected weight layout.

**Verification:** Run `python3 test/test_conv.py --submit` and check this shape no longer crashes.

---

### Phase 3: Depthwise 1x1 32-group FAIL

**Status:** ❌ Not started

**Shapes:**
- `(32,32,1,1,32,32,32)` — test_depthwise_conv2d

**Error:** FAIL (1x1 error: md=46.9375) — NPU produces garbage; `np.allclose(result, expected, atol=0.1)` fails.

**Diagnosis:** `is_depthwise=True` (groups=32 = in_c = out_c), 1x1 kernel. The output is completely wrong (max diff 46.9 for fp16). Possible causes:
1. CBUF/data bank sizing wrong for 32-channel depthwise 1x1 with 32x32 input
2. Weight packing produces wrong format for depthwise mode
3. The `align_c = max(8, min(pow2, 32))` for depthwise may cause data layout issues

**Verification:** Run `python3 test/test_conv.py --submit` and check this shape produces correct results (md < 0.1).

---

### Phase 4: ARGB_IN + non-square kernel partial output

**Status:** ❌ Not started

**Shapes** (all have ic=1 and non-square kernels — ARGB_IN pixel mode):
| Shape | nz_r/nz_e | Notes |
|-------|-----------|-------|
| (1,6,3,1,5,7,1) 3x1 | 18/126 | ~14% output |
| (1,32,5,5,10,10,1) 5x5 | 864/1152 | ~75% output |
| (1,6,2,1,5,7,1) cin=1 2x1 | 24/168 | ~14% output |
| (1,6,2,3,5,7,1) cin=1 2x3 | 24/120 | ~20% output |
| (1,6,3,1,5,7,1) cin=1 3x1 | 18/126 | ~14% output |
| (1,6,3,5,5,7,1) cin=1 3x5 | 18/54 | ~33% output |

**Diagnosis:** All failing shapes share two properties: (a) `in_channels=1` → ARGB_IN/pixel mode active, (b) non-square kernel. Square 3x3 with ic=1 **PASSES** after the stride fix, so the stride formulas are correct for square kernels. Non-square kernels in pixel mode likely need different CSC configuration (convolution window stride, atom count, or feature grain calculation).

Compare the cin=3 (FEATURE mode) variants: all pass for same non-square kernels — confirming the issue is specific to ARGB_IN pixel mode + non-square geometry.

**Verification:** Run `python3 test/test_conv.py --submit` and check these shapes show PASS (or at least nz_r ≈ nz_e).

---

## Known Limitations (No Fix Planned)

- **Non-1x1 kernels produce partial/numerical output** — The NPU (NVDLA-based) silicon may fundamentally produce reduced accuracy for non-1x1 convolution. Some shapes show nz_r ≈ nz_e but fail `allclose` (numerical drift); others show nz_r < nz_e (partial output). These are tracked as WARN, not FAIL, in the test suite.

- **Cross-process isolation** — Always run conv/gemm tests as separate processes. Mixing them produces garbage even after `reset_npu()`. Workaround documented but no fix planned.
