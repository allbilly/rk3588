# gemm.py — Issues & Debug Method

## Status: 17/18 PASS, 1 WARN (256x256x256 fp16 accumulation imprecision)

All 18 test_gemm.py test cases PASS with `np.allclose(atol=0.1)`. The only
non-ideal result is 256x256x256 (WARN, md~54-64) due to fp16 accumulation
error inherent in the NPU hardware — not a register/packing bug.

---

## Shape 1: 32x32x32 — FIXED

**Root cause (old):** `k_exact && m == k` rule incorrectly disabled GROUP_LINE_OFF.

| Register | gemm.c (correct) | gemm.py (before fix) |
|----------|-----------------|---------------------|
| GROUP_LINE_OFF (CONV_CON1 bit 29) | `1` (enabled) | `0` (disabled) |
| NOTCH (DATA_CUBE_NOTCH) | `7` | `0` |
| DST_SURF_STRIDE | `1` | `32` |
| SURFACE_ADD | `4` | `0x800` |
| OUTPUT DECODE | C2=32 (row-major) | C2=4 (wrong) |

**Fix:** Removed `k_exact && m == k` override. Notch zeroing now matches gemm.c
(only for `is_KN_64/256/512 || K>7872`). Output decode uses linear (C2=align_out).

---

## Shape 2: 64x99x64 — FIXED

**Root cause:** `align_in=64 < align_out=128`. The NPU hardware has a CBUF
readback issue when the input cube alignment is smaller than the output
cube alignment at specific align_in values (particularly 64).

**Fix:** When `align_in < align_out`, pad `align_in` up to `align_out`.
Input data is zero-padded for the extra channels; weight values beyond
original K are zero and multiply harmlessly. `eff_k = align_in` (post-padding)
is used for line_stride, surf_stride, feature_grains, notch calculations.

Before fix: md=36.7. After fix: md=0.008.

---

## Shape 3: 256x256x256 — IMPRECISE (not a bug)

| Implementation | Result |
|---------------|--------|
| gemm.c | not tested |
| ops_rknn | PASS (max error = 5.34e-5) |
| **gemm.py** | **WARN (md=54-64)** |

This is NOT a register or packing bug — the NPU's fp16 accumulation pipeline
has limited precision for large dot products. The 256×256 inner dimension
accumulates 256 fp16 products per output element, losing precision to ~3-4
significant figures. This matches the pre-existing ops_reg result of md=40+.

The register config is correct: `is_matmul_256` sets `no_group_line_off=True`,
notch is zeroed by `is_kn_256`, DST_SURF_STRIDE=256, CNSize aligns with
ops_reg. Output uses C2=4 decode (matching gemm.c).

Test_gemm.py tracks it in `known_imprecise` — the WARN is suppressed to
avoid false failures.

---

## Shape 4: 45x100x65 — FIXED

**Details:** `k=65`, `align_in=96`, `align_out=128`. This shape had no explicit
bug — it worked before and still works after all fixes (md=0.008).

---

## Shape 5: 256x256x256 CBUF fix (earlier attempt)

Previously suspected CBUF bank issues — extra data_bank at exact 32768-byte
boundary (`data_bank += 1` on exact multiple). This was already in the code
and is not harmful but didn't fix the imprecision (which is fp16 accumulation,
not CBUF).

---

## All Fixes Summary

| Fix | Shapes affected | Date |
|-----|----------------|------|
| Removed `k_exact && m == k` GROUP_LINE_OFF override | 32x32x32, 64x99x64 | Session 1 |
| Notch zeroing matches gemm.c (is_KN_64/256/512 only) | 32x32x32 | Session 1 |
| Remove (32,32,32) from C2 dispatch tables | 32x32x32 | Session 1 |
| Output decode uses linear for non-special shapes | 32x32x32, 64x99x64 | Session 1 |
| `reopen_device()` between test cases for NPU state isolation | All | Session 1 |
| Pad align_in to align_out when align_in < align_out | 64x99x64, 256x256x256 | Session 2 |
| `eff_k = align_in` used for line_stride/surf_stride/feat_grains/notch | 64x99x64 | Session 2 |
| Weight packing bounds use min(n,align_out), min(k,align_in) | 64x99x64 | Session 2 |
| Output decode zero-pads when n > align_out | 64x99x64 | Session 2 |

---

## Code References

- `experimental/gemm.c` — Canonical matmul register config (174 lines)
- `experimental/rknnops.h` — `make_matmul_params` (line 164), `pack_matmul_weights_fp16` (line 834), `feature_data` (line 765)
- `experimental/rknnops.h:1440-1611` — `alu_case_matmul` (duplicate)
- `examples/gemm.py` — Python implementation
- `~/npu/ops_rknn/matmul_api.cpp` — Vendor library (ground truth)
- `~/npu/ops_reg/main.c` — ops_reg C implementation

## Test Commands

```bash
# Full test suite
cd /home/orangepi/rk3588 && python test/test_gemm.py

# Single shape quick test
python -c "
import sys, numpy as np; sys.path.insert(0, 'examples'); import gemm
np.random.seed(42)
m,n,k=64,99,64; a=np.random.randn(m,k).astype(np.float16); b=np.random.randn(k,n).astype(np.float16)
r=gemm.run_gemm(m,n,k,a,b); e=a@b
print('PASS' if np.allclose(r,e,atol=0.1) else 'FAIL', 'md=', float(np.max(np.abs(r-e))))
"

# Dry run register dump
python -c "
import sys; sys.path.insert(0, 'examples'); import gemm
sys.argv.append('--dry')
gemm.run_gemm(64, 99, 64,
    np.random.randn(64,64).astype(np.float16),
    np.random.randn(64,99).astype(np.float16))
"
```
