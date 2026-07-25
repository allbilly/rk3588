# conv_grok Progress

**Last updated:** 2026-07-10  
**Owner:** Grok session (first-principles CONV rewrite)  
**CWD:** `/home/orangepi/rk3588`  
**Entry:** `conv_grok/conv.py`

---

## 0. TL;DR

1. **217/217 PASS** — regression set; planner accepts **any** `make_shape()` / CLI geometry.
2. **Native only:** spatial/pointwise GEMM hooks removed from `run_shape`.
3. **Planner:** Mesa `entries_per_slice` + bank/feature-grain formulas; `_windows_from_step` tail merge.
4. **Depthwise:** runtime still per-channel serial; `plan_depthwise_rows` models ≤32ch batching for future.
5. **Offline tests:** `python3 conv_grok/test_planner.py`.

---

## 1. Architecture

```text
shape_from_name / make_shape / CLI --in-c ...
  → plan_local_serial_rows   # CBUF → NONE/BY_Y/BY_K/BY_YK
  → plan_depthwise_rows      # depthwise: C windows × Y windows
  → schedule:
       depthwise?    → run_depthwise_serial (batched)
       groups≠1?     → run_grouped_serial
       else          → run_planned (local 1-task tiles)
  → compare vs compute_expected
```

### Planner (formula-only)

| Component | Source |
|---|---|
| `_mesa_entries_per_slice` | Mesa Gallium rocket |
| `_mesa_output_tile_h` | CBUF slice capacity |
| `_pointwise_oc_tile_c` | weight-bank OC cap |
| `_windows_from_step` | Y/K partition + tail merge |
| `_depthwise_channel_step` | min(32, out_c) |

Removed shape-name caps (h28≤22, RGB 48/32/24); replaced by mesa + nhwc grain caps.

---

## 2. Sweep / test commands

```bash
python3 examples/simple_add.py
python3 conv_grok/test_planner.py
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
| `conv_grok/conv.py` | Planner + local serial + CLI |
| `conv_grok/test_planner.py` | Offline planner assertions |
| `conv_grok/sweep_217.py` | 217 regression harness |
| `conv_grok/gemm_npu.py` | Legacy reference (not scheduled) |
| `conv_grok/README.md` | Short usage |

---

## 4. Known limits (honest)

1. **Regression coverage** — 217 + 6 stress; arbitrary shapes use same formulas but are not all hardware-swept.
2. **Tall dense spatial** — `out_h>50` still has a CBUF pressure fallback cap.
3. **Large IC pointwise** — `c1280` ≈35 tiles; utilization not tuned.
4. **Depthwise** — runtime per-channel serial; `plan_depthwise_rows` models ≤32ch batching (48 tasks vs ~36k for c768; HW path TBD).
