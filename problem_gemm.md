# gemm.py — Remaining Issues & Debug Method

## Debug Method: Cross-Reference with gemm.c

When any shape fails, follow this process:

1. **Check `experimental/gemm.c`** — This is the canonical register reference. It contains the `alu_case_matmul` block (identical to `experimental/rknnops.h` lines 1440-1611). This code is known to pass 32x32x32, 64x64x64, 128x128x128.

2. **Check `experimental/rknnops.h`** — Contains the supporting functions (`make_matmul_params`, `pack_matmul_weights_fp16`, `feature_data`) that gemm.c depends on.

3. **Check ops_rknn** — The vendor library (`~/npu/ops_rknn/matmul_api.cpp`) passes ALL shapes. It is the ground truth. Use `gdb -x matmul.gdb ./matmul_api` with `dump.py` to capture register dumps.

4. **Build a register comparison table** like below for the failing shape.

5. **Write a quick Python script** to dump gemm.py's registers and compare against the expected values from gemm.c.

---

## Status: 15/18 PASS, 3 FAIL

ops_rknn passes all 3. gemm.py has bugs.

---

## Shape 1: 32x32x32

| Implementation | Result |
|---------------|--------|
| gemm.c     | PASS (max diff = 6e-6) |
| ops_rknn   | PASS (max error = 5.72e-6) |
| **gemm.py**    | **FAIL** |

### Register Comparison Table for 32x32x32

| Register | gemm.c (correct) | gemm.py (buggy) | ops_rknn dump | Bug? |
|----------|-----------------|-----------------|---------------|------|
| **GROUP_LINE_OFF** (CONV_CON1 bit 29) | `1` (enabled) | `0` (disabled, `k_exact&&m==k`) | `1` (enabled) | **ROOT CAUSE** |
| **NOTCH** (DATA_CUBE_NOTCH) | `7` (align_out=32→8*1-1) | `0` (zeroed by `no_group_line_off`) | `7` | **CASCADE** |
| **DST_SURF_STRIDE** | `1` (out_width_stride=1) | `32` (align_out) | `1` | **CASCADE** |
| **SURFACE_ADD** | `4` (stride*4=1*4) | `0x800` (32*4<<4) | `64` | **CASCADE** |
| **OUTPUT DECODE** | C2=32 (row-major) | C2=4 (wrong) | N/A | **PACKING BUG** |
| DATA_SIZE0 W/H | W=1, H=32 | W=32, H=1 | W=32, H=1 | NOT (swapped OK) |
| line_stride | `4` | `4` | `4` | MATCH |
| data_bank | `1` | `1` | - | MATCH |
| feature_grains | `33` | `33` | `33` | MATCH |

### Root Cause

**Line 291-292 of gemm.py**: `if k_exact and m == k: no_group_line_off = True`
This incorrectly disables GROUP_LINE_OFF for 32x32x32 (k_exact=32, m=k=32). gemm.c only disables it for explicit shapes like (64,64), (256,256), (512,512), etc.

This single bug cascades to wrong DST_SURF_STRIDE, NOTCH, and SURFACE_ADD.

### Fix applied

1. Removed `k_exact and m == k` override (lines 291-292)
2. Removed `no_group_line_off` from notch zeroing (line 327) — gemm.c only zeros for `is_KN_64/256/512 || K>7872`
3. Removed (32,32,32) from PACK_INPUT dispatch (uses row-major, matching `feature_data(align_in, M, 1, align_in, k, m, 1)`)
4. Removed (32,32,32) from DECODE_OUTPUT dispatch (uses linear, matching gemm.c's `unpack_matmul_output_fp32` with C2=align_out=32)

---

## Shape 2: 64x99x64

| Implementation | Result |
|---------------|--------|
| gemm.c | untested |
| ops_rknn | PASS (max error = 1.91e-5) |
| **gemm.py** | **FAIL** (before fix) |

Same root cause as 32x32x32: `k_exact && m == k` (k=64, m=64, but n=99 not 64) incorrectly set `no_group_line_off=True` via the `k_exact && m == k` rule.

After fix: `no_group_line_off=False` (not is_KN_64 since n=99). Registers match gemm.c formula.

### Fix applied

Same as 32x32x32. Also removed (64,99,64) from DECODE_OUTPUT dispatch (uses linear decode, not C2=4).

---

## Shape 3: 256x256x256

| Implementation | Result |
|---------------|--------|
| gemm.c | untested |
| ops_rknn | PASS (max error = 5.34e-5) |
| **gemm.py** | **FAIL** (35-66 md) |

This shape uses `is_matmul_256` which correctly sets `no_group_line_off=True` (matching gemm.c's `is_matmul_256`). The register config should be correct after the notch fix (notch zeroed by `is_kn_256`, not `no_group_line_off`).

May have CBUF bank allocation issues. CBUF fix already applied (extra bank at exact boundary).

---

## Code References

- `experimental/gemm.c` — Canonical matmul register config (174 lines)
- `experimental/rknnops.h` — Supporting functions: `make_matmul_params` (line 164), `pack_matmul_weights_fp16` (line 834), `feature_data` (line 765)
- `experimental/rknnops.h:1440-1611` — `alu_case_matmul` register config (duplicate of gemm.c)
- `examples/gemm.py` — Python implementation
- `~/npu/ops_rknn/matmul_api.cpp` — Vendor library test

## Test Commands

```bash
# gemm.c (ops_reg)
cd ~/npu/ops_reg && ./build/ops_reg matmul 32 32 32

# ops_rknn
cd ~/npu/ops_rknn && ./matmul_api

# gemm.py
cd /home/orangepi/rk3588 && python -c "
import sys, numpy as np; sys.path.insert(0, 'examples'); import gemm
np.random.seed(42)
m,n,k=32,32,32; a=np.random.randn(m,k).astype(np.float16); b=np.random.randn(k,n).astype(np.float16)
r=gemm.run_gemm(m,n,k,a,b); e=a@b
print('PASS' if np.allclose(r,e,atol=0.1) else 'FAIL', 'md=', float(np.max(np.abs(r-e))))
"
```
