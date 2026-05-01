# gemm.py — Readability & Correctness Problems

## Problem 1: Hardcoded Packing Paths for Specific Shapes Only

**Where:** `pack_matmul_input_64x64_fp16`, `pack_matmul_weights_fp16` special-case for `N==32 && K==32`, `pack_matmul_weights_9x9_fp16`

**Root cause:** Only `(2,2,1)`, `(64,64,64)`, `(256,256,256)` have custom packing derived from RKNN dumps. All other shapes use a naive fallback:
```python
in_mat[:, :k] = a_matrix[:, :k]
wt_mat[:n, :k] = b_matrix[:, :n].T
```
This is the same experimental-discover problem as conv weight packing — no general layout rule was found.

**Fix:** Named packing functions with dispatch table keyed by `(M,N,K)`. For each variant, document which model/layer produced the reference RKNN dump.

## Problem 2: Magic KN Flags Controlling Register Emission

**Where:** `make_gemm_regs` — `is_kn_64`, `is_kn_256`, `is_kn_512`, `is_kn_lg_512` override `line_stride`, `notch_val`, `surface_stride`, `stride_d`

**Description:** These booleans control critical CNA DMA stride and CORE configuration fields. No documented relationship between K,N and the register values chosen.

Additional special cases for M=1 (vector) matmuls:
- `is_matmul_768` (M=1,K=768,N=768)
- `is_matmul_768_2048` (M=1,K=768,N=2048)
- `is_matmul_2048` (M=1,K=2048,N=2048)

These override `stride_d` differently from the main path.

**Fix:** Lookup table keyed by `(K,N)` and `(M,K,N)` with rationale comment per entry, or derive algebraically if a pattern exists.

## Problem 3: Shape-Dependent Output Decoding

**Where:** Custom decode for `(64,64,64)` and `(256,256,256)` using `c2=4` interleaved plane/offset — all other shapes use linear stride-based read with auto-stride detection.

**Root cause:** The NPU DPU write DMA produces different output layouts depending on shape configuration. No unified contract documented.

**Fix:** Understand the DPU write DMA addressing primitive (surface_add, stride, channel_field interaction). Unify to one decode or make the dispatch explicit with rationale.

## Problem 4: Precision Loss in Large GEMMs

**Where:** `(256,256,256)` produces max diff ~50–70 in float16

**Root cause:** Suspected float16 accumulation overflow in DPU — large K dimension accumulates too many products in fp16 before rounding.

**Fix:** Investigate block-wise summation (split K dimension) or intermediate fp32 accumulation if the hardware supports it. For now, accepted as WARN:

## Problem 5: Duplicate C Implementations

**Where:** `experimental/gemm.c` and `experimental/matmul.c` are byte-identical files.

**Fix:** Remove one.

## Problem 6: All Registered Special Cases in This Code Base

**C packing special cases:**
- `weight_fp16(C, k, c)` — column-major tiled index function in `experimental/rknnops.h:756`
- `feature_data(C, H, W, C2, c, h, w)` — NC1HWC2 index function in `experimental/rknnops.h:765`
- `pack_matmul_input_64x64_fp16` — custom C2=8 packing
- `pack_matmul_input_9x9_fp16` — 32-half stride per row
- `pack_matmul_weights_9x9_fp16` — column-major 16-half stride
- `pack_matmul_weights_fp16` — `weight_fp16` tiling (general) vs flat column-major (32x32 case)

**Python packing special cases:**
- 2x2x1: direct copy `a[0,0], a[0,1], a[1,0], a[1,1]` into hardcoded indices
- 64x64x64: custom `pack_matmul_input_64x64_fp16`
- 256x256x256: custom `c2=4` plane/offset in packing loop
- Everything else: naive fallback

**Python output decode special cases:**
- 64x64: `c2=4`, per-row `(plane, offset) = divmod(j, 4)`
- 256x256: same c2=4 decode but different stride computation
- Everything else: linear read with stride detected from output array shape

**Register emission special cases (KN flags):**
- `is_kn_64` (K=64,N=64): `line_stride = k*4`, DV=7, `notch_val = k`
- `is_kn_256` (K=256,N=256): mode=1, stride override
- `is_kn_512` (K=512,N=512): cvt_con0 override, stride override  
- `is_kn_lg_512` (K>512,N>512): cvt_con0 override, stride override
- `is_matmul_768`: `stride_d = 192`
- `is_matmul_768_2048`: `stride_d = 192`
- `is_matmul_2048`: `stride_d = 512`

## Test Coverage and Limitations

Source: `test_gemm.py` — 18 cases, all PASS with caveats

| Shape | Status | Caveat |
|-------|--------|--------|
| 2x2x1 | PASS (proven) | Tiny manual case |
| dot 65 | WARN | Vector shapes untested in general packing |
| vec@mat 1x45 | WARN | Vector shapes untested in general packing |
| mat@vec 45x1 | WARN | Vector shapes untested in general packing |
| matmul 45x100 | WARN | Non-square shape untested in general packing |
| matmul 64x99 | WARN | Non-square shape untested in general packing |
| 4x4x4 | PASS (proven) | |
| 8x8x8 | PASS (proven) | |
| 9x9x9 | PASS (proven) | |
| 32x32x32 | PASS (proven) | |
| 64x64x64 | PASS (proven) | |
| 256x256x256 | WARN | Imprecise (~md=50–70, pre-existing) |
| 4x8x16 | PASS (proven) | |
| 16x4x8 | PASS (proven) | |
| 8x32x4 | PASS (proven) | |
| 12x34x56 | WARN | Non-square shape untested in general packing |
| 50x10x20 | WARN | Non-square shape untested in general packing |
| 32x32x1 | PASS (proven) | |

**Proven working shapes (10):** 2x2x1, 4x4x4, 8x8x8, 9x9x9, 32x32x32, 64x64x64, 4x8x16, 16x4x8, 8x32x4, 32x32x1

**Admitted WARN shapes (8):** All vector and non-square matmuls + the imprecise 256x256x256

## Cross-Process Isolation

conv and gemm cannot share a Python process — NPU state persists between separate `/dev/dri/card1` FDs even after `reset_npu()`. Run in isolated subprocess invocations.
