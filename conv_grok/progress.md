# conv_grok Progress

**Last updated:** 2026-07-10  
**Owner:** Grok session (first-principles CONV rewrite)  
**CWD:** `/home/orangepi/rk3588`  
**Entry:** `conv_grok/conv.py` + `conv_grok/gemm_npu.py`

---

## 0. TL;DR

1. **217/217 PASS** — `sweep_results/conv_grok_217_sweep_20260710_150352_summary.txt`.
2. **Spatial GEMM:** **0**. **Pointwise GEMM:** **0** (native CONV only for 1×1).
3. Key body fixes: 32-align pointwise pack/`data_in_c`, bank-aware `data_bank`, `k_step` fits 1 weight bank, unclamped pointwise `feature_grains`, strict y-bank margin.
4. Extra hard: `python3 conv_grok/conv.py --extra-hard`.

---

## 1. Architecture

```text
shape_from_name
  → plan_local_serial_rows   # CBUF → NONE/BY_Y/BY_K/BY_YK
  → schedule:
       depthwise?              → run_depthwise_serial
       groups≠1?               → run_grouped_serial
       _prefer_pointwise_gemm? → gemm_npu.run_pointwise_gemm  (hook; currently off)
       _prefer_spatial_gemm?   → gemm_npu.run_spatial_gemm    (hook; currently off)
       else                    → run_planned                  (local 1-task tiles)
  → compare vs compute_expected
```

### GEMM predicates (formulas only)

| Predicate | Rule |
|---|---|
| pointwise GEMM | **off** (32-align pack + bank-aware CBUF + k/y caps) |
| spatial GEMM | **off** (DMA surf + bank rules) |

Pointwise GEMM chunks tall maps so DRM BOs stay small (fixes `c16_h150` ENXIO).

### Planner formulas added this session

- `in_c≥128`, `out_c≤32`, `out_h==28` → `y_step≤22`
- RGB first-layer `in_c==3`, `k3`, `in_h≥224` → `y_step≤48` (or 32 if taller); `in_h>224` also routes to spatial GEMM

---

## 2. Sweep results

| Sweep | Result |
|---|---|
| `20260710_104525` | PASS=152 FAIL=59 ERROR=6 (pre-GEMM) |
| `20260710_112009` | PASS=216 ERROR=1 (`c16_h150` ENXIO on huge GEMM BO) |
| `20260710_121250` | **PASS=217** (row-chunked GEMM; broad spatial_gemm) |
| `20260710_130516` | **PASS=217** (12/13 spatial → native; only `c128_h3` on spatial GEMM) |
| `20260710_135341` | **PASS=217** (spatial_gemm=0; pw_gemm ~85→60; DMA surf + bank rules) |

Commands:

```bash
python3 examples/simple_add.py
python3 conv_grok/conv.py --classify
python3 conv_grok/conv.py --sweep --timeout 300
python3 conv_grok/conv.py --extra-hard
python3 examples/simple_add.py
```

**Safety:** do not kill long NPU jobs; do not casually change submit masks.

---

## 3. File map

| Path | Role |
|---|---|
| `conv_grok/conv.py` | Planner + local serial + GEMM routing + CLI |
| `conv_grok/gemm_npu.py` | Self-contained NPU GEMM (FP32 surface, PINGPONG) |
| `conv_grok/sweep_217.py` | 217 harness |
| `conv_grok/progress.md` | This log |
| `conv_grok/README.md` | Short usage |

---

## 4. Session log

| Time | Event |
|---|---|
| 2026-07-10 AM | Created `conv_grok/`; local-serial smokes PASS |
| 2026-07-10 | Baseline 217: 152/59/6 |
| 2026-07-10 | Added `gemm_npu.py` + predicates; prior 59 fails → 59 PASS |
| 2026-07-10 | Fixed `c16_h150` via row-chunked pointwise GEMM |
| 2026-07-10 | **Full 217/217 PASS** + `--extra-hard` set |
| 2026-07-10 | Extra-hard 6/6 PASS (`c3_h384`, `c16_h160_k3`, `c64_h112`, `c256_h28_oc512`, `c512_h14_oc256`, `c72_h20_k3`) |
| 2026-07-10 | Migrated 12/13 spatial_gemm → native; `_tile_full_data_bank` weight-fit rule; only `c128_h3` remains on spatial GEMM |
| 2026-07-10 | DMA surf `width*(h-4)` wrap fixes `c128_h3` → native BY_K; spatial_gemm=0 |
| 2026-07-10 | Narrowed pw_gemm (~85→~60); pointwise K-tiles always full_data_bank |
