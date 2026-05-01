# gemm.py — Bug History

All **20 test_gemm.py** test cases PASS (18 original + 2 new: 1×4×4, 1×99×64).

All 9 shape whitelists replaced with hardware-derived formulas. The sole remaining
shape-specific condition (C2=8 input packing for (64,64,64) and (256,256,256) in
`_uses_c2_input()`) is a documented design constraint — the CDMA stride / CBUF
layout / CSC read coupling doesn't generalize without deeper hardware modeling.

Problem 2
```
can get_output_decoder rewrite to save lines of code
Current (7 lines):
def get_output_decoder(m, n, k, align_out):
    if (m, n, k) == (64, 64, 64):
        return decode_output_c2_4
    if (m, n, k) == (256, 256, 256):
        return decode_output_c2_4
    return decode_output_linear
Collapse to 2 lines:
def get_output_decoder(m, n, k, align_out):
    return decode_output_c2_4 if (m, n, k) in ((64, 64, 64), (256, 256, 256)) else decode_output_linear
Same for _uses_c2_input (used by both input packing and register logic) — (64,64,64) and (256,256,256) are already checked there:
def _uses_c2_input(m, n, k):
    return (m, n, k) in ((64, 64, 64), (256, 256, 256))
So get_output_decoder could just delegate:
def get_output_decoder(m, n, k, align_out):
    return decode_output_c2_4 if _uses_c2_input(m, n, k) else decode_output_linear
That saves 5 lines. Should I apply this?
```

problem 3 
reopen_device() is defined at line 188 but never called anywhere in the file. It's dead code.
The current flow works fine because:
1. All buffers are allocated once at module level (lines 177-181)
2. reset_npu() (line 440) is called before each submit() to reset the device state between test cases
3. Multiple test cases reuse the same buffers without reopening
So no, it's not needed. You can safely remove it.

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
