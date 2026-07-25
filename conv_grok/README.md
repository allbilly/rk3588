# conv_grok — first-principles FP16 CONV

Standalone CONV for RK3588 NPU: Mesa-style CBUF planner + local serial tiles. **No GEMM path in the hot schedule** (native CONV only).

## Scope

| What | Detail |
|---|---|
| **Regression set** | 217 shapes from `conv_new.py` + 6 extra-hard stress cases |
| **Any shape** | `make_shape()` or CLI `--in-c/--in-h/...` — planner is formula-driven, not name tables |
| **Limits** | Correctness validated on regression set; huge IC still uses many K-tiles (utilization TBD) |

## Progress (2026-07-10)

| Milestone | Status |
|---|---|
| 217-shape sweep | **217/217 PASS** |
| GEMM escape | **0** (hooks removed from schedule) |
| Planner | Mesa `entries_per_slice` + bank/feature-grain formulas |
| Depthwise | Serial per-channel (proven); `plan_depthwise_rows` = batched target (~48 vs ~36k for c768) |
| Planner tests | `conv_grok/test_planner.py` (offline, no NPU) |

### Planner formulas (not shape-name tables)

1. **DMA surf stride:** `width_stride*(in_h-4)` even when `in_h<4` (uint32 wrap).
2. **Pointwise 32-align:** pack + `data_in_c = align_up(in_c, 32)`; wide threshold `in_c≥32`.
3. **Bank-aware tiling:** weight banks reserved; `k_step` from `_pointwise_oc_tile_c`; 2-bank upgrade when features fit.
4. **Y step:** `_mesa_output_tile_h` + feature grains + strict bank headroom; `_windows_from_step` merges tiny tails.
5. **Depthwise:** `_depthwise_channel_step=32` + same Y planner on full `in_c`.

## Design

```text
shape_from_name / make_shape / CLI fields
  → plan_local_serial_rows        # NONE / BY_Y / BY_K / BY_YK
  → plan_depthwise_rows           # depthwise: C×Y batched tiles
  → depthwise?     → run_depthwise_serial
     groups≠1?    → run_grouped_serial
     else         → run_planned (1-task local serial per tile)
```

- No hex blobs / OVERRIDES / exact11
- `gemm_npu.py` kept for reference; not called from `run_shape`

## Usage

```bash
python3 examples/simple_add.py
python3 conv_grok/test_planner.py          # offline planner asserts
python3 conv_grok/conv.py --list
python3 conv_grok/conv.py --dry-run <shape>
python3 conv_grok/conv.py --classify       # 217 split summary, no NPU
python3 conv_grok/conv.py --sweep --timeout 300
python3 conv_grok/conv.py --extra-hard
# arbitrary shape (no encoded name needed):
python3 conv_grok/conv.py --in-c 64 --in-h 56 --out-c 128 --kh 3 --dry-run
python3 examples/simple_add.py
```

## Verified families (217 regression)

| Path | Examples |
|---|---|
| NONE / BY_Y / BY_K / BY_YK | `c4_h9`, `c3_h224/320/384`, `c40_h40_k3`, `c72_h20_k3`, `c128_h3_k3`, `c96_h56` 1×1 |
| depthwise serial | through `c1024`; batched planner in `plan_depthwise_rows` (not scheduled yet) |
| grouped serial | `g>1` divisible IC/OC |
