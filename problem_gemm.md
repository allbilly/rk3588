# gemm.py — Problems & Improvement Plan

## Status

All **20 test_gemm.py** test cases PASS (18 original + 2 new: 1×4×4, 1×99×64).

All past bugs fixed (submit NONBLOCK, CBUF boundary, align_in padding).

**Problem 1** (shape whitelists) resolved in Phases 1-7 — 9 whitelists reduced to
hardware-derived formulas. Remaining shape-specific conditions (C2=8 input packing,
line_stride exclusion for K=64/256, surf_stride K-ranges) are documented
hardware-format constraints, not whitelists.

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

### Refactoring Plan

#### Phase 1: Eliminate packing/decode dispatch tables (whitelists #8, #9)

Replace shape-key lookup with hardware-derived C2:

```python
C2_INPUT = 8   # derived from ATOMIC_K_SIZE=32 → 32/4 = 8 for fp16
C2_OUTPUT = 4  # hardware constraint (DPU output formatter atomic width)

def get_input_packer(align_in):
    if align_in >= C2_INPUT * 2:   # need at least 2 C2 planes for packing
        return partial(pack_input_nc1hwc2, c2=C2_INPUT)
    return pack_input_row_major

def get_output_decoder(align_out):
    if align_out >= C2_OUTPUT * 2:
        return partial(decode_output_nc1hwc2, c2=C2_OUTPUT)
    return decode_output_linear
```

**Verification**: All 18 test cases must produce same register values for
(2,2,1), (64,64,64), (256,256,256). New shapes like (128,128,128) should
produce correct results without manual whitelisting.

**Risk**: Shapes where C2 packing was never tested (e.g., K=32, N=32) might
have packing index bounds issues. Must verify array indices don't overflow.

#### Phase 2: Replace `no_group_line_off` whitelist (whitelists #1, #2)

Derive from hardware geometry. When K==N (original, not aligned), the
CSC line offset optimization causes address aliasing:

```python
no_group_line_off = (k == n) and (align_in >= 64)
```

This covers all 7 whitelisted shapes AND any future K==N≥64 shape.

**Verification**: Check that shapes where k≠n (32x32x33, 64x99x64, 8x8x4)
still have GROUP_LINE_OFF=1. Check that k==n small shapes (4x4x4) don't
break — this shape was never whitelisted and may behave differently.

#### Phase 3: Unify line_stride formula (whitelist #3)

Remove the `eff_k not in (64, 256)` exclusion:

```python
line_stride = 4
if 32 < eff_k < 512:
    line_stride = min(13, (eff_k + 31) // 32) * 4
```

For K=64: line_stride becomes 12 (was 4). For K=256: line_stride becomes 32
(was 4). This matches ops_rknn's value for K=256.

**Verification**: Test (64,64,64) and (256,256,256) specifically — these
will have different line_stride values. ops_rknn proves 32 works for
256x256. 64x64 needs verification.

**Risk**: line_stride=12 for K=64 may cause different CBUF line layout.
Create register comparison table for 64x64x64 before/after.

#### Phase 4: Simplify surf_stride (whitelist #4)

The K-range zeroing conditions appear to guard against surf_stride overflow
when the surface layout doesn't need multiple surfaces:

```python
surf_stride = (line_stride * (surf_groups - 1)) if surf_groups > 1 else 0
```

Remove the 4 K-range conditions. If `surf_groups <= 1` (m <= 4), stride is 0.

**Verification**: Test shapes with small m (dot 65: M=1, vec@mat 1x45: M=1,
mat@vec 45x1: M=45) and shapes with K in the previously-zeroed ranges
(e.g., K=100, K=200).

#### Phase 5: Unify dst_surf_stride (whitelist #5)

Remove `is_matmul_64/256` special cases:

```python
dst_surf_stride = align_out  # always align_out; no_group_line_off just uses 1
```

The special values 64 (for 64x64) and 256 (for 256x256) were copied from
gemm.c without understanding why. If align_out works for all other shapes,
it should work for these too.

**Risk**: This might change output data layout for 64x64 and 256x256. Must
verify decode_output_c2_4 still works. Create register comparison table.

#### Phase 6: Unify feature_grains (whitelist #6)

Rewrite with a single CBUF-capacity formula for all K:

```python
def calc_feature_grains(m, eff_k, align_in):
    if eff_k > 7872:
        return 2  # very wide lines, fetch 2 lines at a time
    if m <= 80:
        return m + 1  # small height, full cube fits in one slice
    denom = align_in * 2
    grains = (2 * 32768 + denom - 1) // denom
    grains = (grains + 1) & ~1  # round up to even
    return max(80, grains)
```

The K=128-192 special case (feature_grains=m) was likely a band-aid for a
specific failing shape. Verify whether the general formula works for
shapes in this range.

#### Phase 7: Simplify notch (whitelist #7)

```python
notch_val = 8 * min(13, align_out // 32) - 1
if k == n and align_out >= 64:
    notch_val = 0  # no notch needed when K==N (square matmul)
```

Replace `is_kn_64/256/512` with the general `k == n` rule. The K>7872 case
is handled by the feature_grains=2 branch which may also need notch=0.

**Verification**: Check that K!=N large shapes (e.g., 1x8155x8155, 64x99x64)
get non-zero notch. Check that 512x512x512 gets notch=0.

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
