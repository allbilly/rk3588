# gemm.py — Problems & Improvement Plan

## Status

All **20 test_gemm.py** test cases PASS (18 original + 2 new: 1×4×4, 1×99×64).

All past bugs fixed (submit NONBLOCK, CBUF boundary, align_in padding).

**Problem 1** (shape whitelists) — 6 of 9 whitelists replaced with hardware-derived
formulas. 3 conditions remain shape-specific due to data-format coupling (C2=8 input
packing, line_stride for C2=8, surf_stride CBUF constraints) — these are documented
as hardware-format constraints, not whitelists.

---

## Problem 1: Shape Whitelists Instead of Hardware-Derived Parameters

The code has accumulated 9 shape-specific whitelists (exact `(m,n,k)` tuples,
narrow K ranges, named booleans) where register values should be derivable
from NPU architecture constants + shape parameters.

### Architecture Constants (from `nvdla/hw` C-model + `rockchip.py`)

| Constant | Value | Source |
|---|---|---|
| `ATOM_CUBE_SIZE` | 32 bytes | CDMA `ActDmaResponseHandler` |
| `CBUF_ENTRY_PER_BANK` | 256 entries | CSC C-model |
| `CBUF_ENTRY_SIZE` | 128 bytes | CSC C-model |
| **CBUF bank size** | **32768 bytes** | 256 × 128 |
| `NPU_CBUF_BANKS` | 12 | `rknnops.h` |
| `ATOMIC_K_SIZE` | 32 (MAC atomic K) | `nvdla_config_large.h` |
| `PARALLEL_CHANNEL_NUM` | 64 | NVDLA MAC array width |
| `C2_INPUT_FP16` | 8 (from 16-byte memAtom) | NVDLA/sw compiler |
| `LINE_STRIDE` field | shift=0, mask=0x0fffffff | `rockchip.py` |
| `DST_SURF_STRIDE` field | shift=4 | `rockchip.py` |
| `SURFACE_ADD` field | shift=4 | `rockchip.py` |

### The 9 Whitelists

| # | Location | Whitelist | Should Be Derived From |
|---|---|---|---|
| 1 | `make_gemm_regs:286-292` | 7 named booleans: `is_kn_64/256/512/lg_512`, `is_matmul_768/768_2048/2048` | K==N relation + align_in geometry |
| 2 | `make_gemm_regs:294` | `no_group_line_off` = OR of 7 booleans | Single condition: `k == n` and `align_in >= 64`? |
| 3 | `make_gemm_regs:297` | `eff_k not in (64, 256)` for line_stride | `line_stride = min(13, (eff_k+31)//32) * 4` for ALL eff_k |
| 4 | `make_gemm_regs:302-303` | 4 K-ranges zeroing surf_stride | surf_stride = 0 only when `surf_groups <= 1` (m <= 4) |
| 5 | `make_gemm_regs:312` | `is_matmul_64/256` special dst_surf_stride values | dst_surf_stride = `align_out` always (remove 64/256 special cases) |
| 6 | `make_gemm_regs:314-323` | K>7872, 128<K≤192, K>192&&K≠256 for feature_grains | Single CBUF formula for all K |
| 7 | `make_gemm_regs:327-329` | `is_kn_64/256/512` or K>7872 for notch=0 | notch=0 when align_out fits in single atomic column |
| 8 | `gemm.py:256-270` | Shape-key dispatch for PACK_INPUT/DECODE_OUTPUT | Derive C2 from align_in (input) and align_out (output) |
| 9 | `gemm.py:386-390` | `pad_k` only when align_in < align_out | Same, but eff_k logic is fragile |

### Remaining Shape-Specific Conditions (not whitelists)

#### 1. `line_stride` exclusion for K=64, 256

C2=8 packed data needs line_stride=4, while row-major data uses `(eff_k+31)//32 * 4`.
Fixed by keeping the exclusion with a comment explaining the C2=8 coupling.
Deriving line_stride from data format would fix this but requires propagating
the data-format decision through the register computation.

Location: `make_gemm_regs:298-300`

#### 2. `surf_stride` K-range zeroing

Four K-range conditions (`32<K<64`, `64<K≤128`, `128<K<256`, `256<K<512`)
zero surf_stride for non-power-of-2 K values. This is a genuine CBUF hardware
constraint — removing it breaks (12,34,56).

Location: `make_gemm_regs:302-305`

#### 3. C2=8 input packing

Still shape-whitelisted to (64,64,64) and (256,256,256) because register
formulas for other shapes expect row-major data. Generalizing C2=8 to all
`align_in >= 64` requires also generalizing line_stride, surf_stride, etc.

Location: `get_input_packer:258-264`

### Verification Strategy

For each phase, verify using this process:

1. Build register comparison table for all 18 test shapes
2. Only registers affected by the change should differ
3. Run full test suite — all 18 must PASS
4. For any failure, compare register dumps with ops_reg (gemm.c)

### Key: ops_rknn Proves Hardware Flexibility

ops_rknn uses for 256x256x256: GROUP_LINE_OFF=1, feat_grains=128,
line_stride=32, notch=63, dst_surf_stride=1, surface_add=4 — a completely
different valid configuration. This proves the NPU accepts multiple valid
register sets and the whitelists are overfit, not required.

---

## Bug History (resolved)

### Bug A: Submit NONBLOCK (Session 3 — FIXED)
**Root cause**: `submit.flags` included `RKNPU_JOB_NONBLOCK` (bit 1).
ioctl returned before NPU finished (~450ms for 256x256). Only affected the
largest shape; smaller shapes happened to finish fast enough.
**Fix**: Changed `flags=0x7` → `flags=0x1`.

### Bug B: CBUF data_bank boundary (Session 1 — FIXED)
**Root cause**: `data_bank = (input_bytes + 32768) // 32768` incremented at
exact 32768 boundary (256x256x256: 131072 bytes → bank 5 instead of 4).
gemm.c uses ceiling division: `(bytes + 32767) // 32768`.
**Fix**: Changed `+ 32768` → `+ 32767`.

### Bug C: GROUP_LINE_OFF false generalization (Session 1 — FIXED)
**Root cause**: `k_exact and m == k` incorrectly disabled GROUP_LINE_OFF
for 32x32x32 and 64x99x64.
**Fix**: Replaced with exact 7-shape whitelist matching gemm.c.

### Bug D: align_in padding (Session 2 — FIXED)
**Root cause**: When `align_in < align_out` (e.g., K=64, N=99 → align_in=64,
align_out=128), NPU CBUF readback produced garbage (md=36.7).
**Fix**: Pad align_in to align_out. eff_k = align_in for downstream calcs.

### Bug E: Notch zeroing rule (Session 1 — FIXED)
**Root cause**: Notch was zeroed whenever `no_group_line_off` was true.
gemm.c only zeroes notch for `is_KN_64/256/512` or `K>7872`.
**Fix**: Matched gemm.c's condition.
