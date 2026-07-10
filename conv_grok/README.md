# conv_grok — first-principles FP16 CONV

Standalone CONV for RK3588 NPU: CBUF planner + local serial + narrow predicate GEMM.

## Progress (2026-07-10)

| Milestone | Status |
|---|---|
| 217-shape sweep | **217/217 PASS** — `sweep_results/conv_grok_217_sweep_20260710_150352_summary.txt` |
| Spatial GEMM | **0** |
| Pointwise GEMM | **0** — all 1×1 on native CONV |
| Extra-hard | `c3_h384` native BY_Y; spatial extras native |

### Key formula fixes (this session)

1. **DMA surf stride:** `width_stride*(in_h-4)` even when `in_h<4` (uint32 wrap).
2. **Pointwise 32-align:** pack + `data_in_c = align_up(in_c, 32)`; wide threshold `in_c≥32` (fixes `c40`/`c72`).
3. **Bank-aware `full_data_bank`:** leave weight banks for OC tile; `k_step` capped to 1-bank OC (`c528`/`c576`/`c832`).
4. **Unclamped pointwise `feature_grains=in_h`** (fixes `c960`); **strict y-bank margin** (fixes `c768_h20` y=11).
5. **GEMM hooks off** for both pointwise and spatial.

### Path policy

| Path | When | CONV pipeline? |
|---|---|---|
| native `NONE/BY_Y/BY_K/BY_YK` | all 217 shapes (via local serial / DW / grouped) | **yes** |
| pointwise / spatial GEMM | hooks only (currently unused) | — |

### Next

Optional: speed (fewer tiles for huge IC), or revisit GEMM only if a new unsafe native body appears.

## Design

- CBUF planner: `NONE / BY_Y / BY_K / BY_YK` (BY_YK = Y×K cartesian local tiles)
- Local serial: one 1-task submit per tile, Python assemble
- Predicate GEMM (`gemm_npu.py`): wide/tall 1×1 escape only — **no shape-name allowlists**
- Depthwise → per-channel `group=1` serial; grouped → per-group serial
- No hex blobs / OVERRIDES / exact11

## Usage

```bash
python3 examples/simple_add.py
python3 conv_grok/conv.py --list
python3 conv_grok/conv.py --dry-run <shape>
python3 conv_grok/conv.py                    # default smoke
python3 conv_grok/conv.py <shape>
python3 conv_grok/conv.py --classify
python3 conv_grok/conv.py --sweep --timeout 300
python3 conv_grok/conv.py --extra-hard
python3 examples/simple_add.py
```

## Verified families

| Path | Examples |
|---|---|
| NONE / BY_Y / BY_K / BY_YK native | `c4_h9`, `c3_h224/320/384`, `c16_h80`, `c40_h40_k3`, `c72_h20_k3`, `c128_h3_k3`, `c160_h14_k3`, `c96_h56` 1×1, `c64_h56` 1×1 |
| pointwise GEMM | `c128_h28_oc256`, `c576_h19`, `c1024_h7`, `c40_h40_oc320` 1×1 |
| depthwise serial | through `c1024` |
