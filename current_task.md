# Current Task: Prefix-Replay Debug of All FENCED Shapes in `examples/conv.py`

**Last updated:** 2026-06-03 12:13 (Asia/Shanghai)
**Owner:** Codex session (multi-model handoff)
**CWD:** `/home/orangepi/rk3588`
**Latest commits on HEAD:** `7619702` (c576_h19_oc12 no-submit materializer + manifest refresh) + `09bfdf9` (no-submit c256_h28 materializer) + `db70141` (c256_h2_oc64 promotion + regcmd scan)
**Uncommitted work (tracked, in worktree):**
- `examples/conv.py` — 60-line addition for `_c256_h2_oc64_exact11_task_regs` + `run_c256_h2_oc64_exact11_shape` + `--allow-c256-h2-oc64-exact11-submit` CLI flag
- `conv_expt/rknn_prefix_replay.py` — manifest note refresh for `c256_h2_oc64` and `c256_h28_dw_by_yk`
- `experimental/rknn/dump_rknpu_task_gems.py` — `--scan-regcmd` / `--min-regcmd-run` for regcmd-only dumps
**Uncommitted work (untracked, all no-submit materializers):**
- `examples/conv_c256_h28_dw_byyk_no_submit.py` (143 lines) — body field assertions for c256_h28 depthwise
- `examples/conv_c576_h19_oc12_no_submit.py` (138 lines) — NEW no-submit task structure + body field invariants for Attack H
- `examples/conv_no_submit_materializer.py` (106 lines) — Descriptor/EmitterProfile/E() regcmd primitives
- `examples/conv_no_submit_closure.py` (105 lines) — setup_full_reg_qwords() closure
- `examples/conv_no_submit_fixtures.py` (123 lines) — K_TILE_RUN_METADATA, planner fixtures
- `examples/conv_crash_fence_no_submit.py` (209 lines) — c256_h2_oc64 crash-fence body validator
- `examples/conv_depthwise_by_y_layout_no_submit.py` (283 lines) — depthwise BY_Y layout assertions
- `examples/conv_h14_k_tile_no_submit.py` (796 lines) — h14 k_tile layout assertions
- `examples/conv_h14_task_layout_no_submit.py` (1480 lines) — h14 task layout assertions
- `examples/conv_h160_setup3_task_layout_no_submit.py` (380 lines) — h160 setup3 layout
- `examples/conv_output_bo_map_no_submit.py` (60 lines) — output BO mapping
- `examples/conv_pointwise_by_k_layout_no_submit.py` (261 lines) — pointwise BY_K layout
- `examples/conv_pointwise_by_yk_layout_no_submit.py` (155 lines) — pointwise BY_YK layout
- `examples/conv_small_no_submit.py` (180 lines) — small-shape assertions
- `examples/conv_spatial_by_y_layout_no_submit.py` (353 lines) — spatial BY_Y layout
- `examples/conv_tiles_no_submit.py` (126 lines) — tile helpers
- `examples/conv_legacy.py` (989 lines) — old reference implementation (kept for diff comparison)
- 25 `experimental/rknn/capture_*.log` files + `capture_c64_state.py` — RKNN read-only captures
**NPU health (re-verified 2026-06-03 11:48):** `python3 examples/simple_add.py` PASS (`ADD  NPU=[8 8 8 8 8 8 8 8] expected=[8 8 8 8 8 8 8 8] PASS`). Board uptime stable. Board has NOT been rebooted this session.
**Latest full sweep (2026-06-03 11:40:47):**
```
total=217 counts={'PASS': 137, 'FENCED': 80, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}
pre_health_rc=-1 post_health_rc=-1   # --skip-health was passed; simple_add manually verified PASS
snapshot: /tmp/opencode/conv_py_217_sweep_20260603_114047_summary.txt
```
**Pass-progress table:**
| Date | PASS | FENCED | Note |
|---|---:|---:|---|
| 2026-06-02 09:54 | 114 | 103 | user-stated stuck baseline |
| 2026-06-02 14:43 | 115 | 102 | +1 first probe |
| 2026-06-02 22:30 | 134 | 83 | c256_h2 oc64/oc546 fenced |
| 2026-06-03 09:49 | 136 | 81 | c512_h7 pointwise promoted |
| 2026-06-03 10:30 | 137 | 80 | c256_h2_oc64 promoted (this session's prior turn) |
| 2026-06-03 11:40 | **137** | **80** | **CURRENT** — re-verified; no-submit materializer + manifest update |
| **Target** | **217** | **0** | end state |

**About "it crashed":** NPU is healthy right now. The earlier crash (before this session) was from a c256_h2_oc546 speculative local submit that rebooted the board. That shape is now in `CRASH_FENCED_SHAPES` and exits before DRM allocation. The more recent c256_h2_oc64 work used a guarded `--allow-c256-h2-oc64-exact11-submit` path and ran cleanly inside an exact11 closure. There has been NO reboot this session. If a future crash happens, recover with `python3 examples/simple_add.py` (must PASS) and re-sweep.

---

## 0. TL;DR (read this first)

1. **Goal:** Use prefix replay to debug and fix every FENCED shape in `examples/conv.py`. End state: **PASS=217, FENCED=0, FAIL=0, ERROR=0, TIMEOUT=0**, with pre/post `python3 examples/simple_add.py` both PASS.
2. **Current state (2026-06-03 11:40, verified):** PASS=137, FENCED=80, FAIL=0, ERROR=0, TIMEOUT=0. NPU healthy.
3. **This turn's progress:** re-verified clean state; added no-submit materializer for c256_h28 depthwise BY_YK (Attack C, body-field assertion only — 14 constants from live GEM2 capture); refreshed manifest notes; added regcmd-scan tooling to `dump_rknpu_task_gems.py`. **Zero net PASS change this turn.** The +22 PASS from the previous baseline (114 → 136) came from the prior turn's promotions.
4. **Biggest blocker right now (the answer to "what is the biggest blocker"):** the 36 `BY_K/k_tile` narrow-OC pointwise + spatial k5x5/k7x7 family. These are the largest bucket and the hardest because:
   - The local planner emits 60-90 equal BY_K rows; the RKNN driver emits a heterogeneous sequence (k_half → k_tile interleaved with PPU/PDP separators, 1k setups, prologue rows).
   - For **narrow pointwise c832/c1024/c1280 oc24-oc546 with 1x1 kernels**, the local visible-row replay returns `inf` even though RKNN's prefix replay is numerically clean (`max_diff=0.0585` for c832_h7_oc48). The gap is hidden buffer/layout parity (CNA weight bank rotation, C2 packing, dst offsets) — not just register field overrides.
   - For **k5x5/k7x7 spatial** (7 shapes: c16_h80_oc128 5x5, c1024_h7_oc1024 k7x7, c480_h10_oc480 k5x5, c576_h20_oc576 k5x5, c768_h20_oc768 k5x5, c960_h10_oc960 k5x5, c1024_h7_oc1024 k3x3 with stride 1), no row materializer is built and no fresh RKNN capture exists.
5. **Other blockers, ranked:**
   - **19 `BY_YK` depthwise c64/c96/c128/c144/c256 h=28/56/75/112/150 + c256_h28 pointwise 1x1**: row materializers needed. The c256_h28 depthwise body fields are now asserted (no-submit materializer is wired) but a real row materializer that emits 12 regcmd rows with weight_reuse per row is the next step.
   - **13 depthwise `BY_Y` y_tile (c32_h150, c32_h112, c256_h3, c256_h10, c512_h5, c576_h20, c768_h20, c96_h20, c384_h10, c512_h5)**: single-task rewrites, scaffolded but not wired through the new `run_depthwise_shape` body overrides.
   - **4 `NONE/setup` spatial c16_h80_oc128 3x3/5x5 (2 shapes) and c1024_h7_oc1024 k7x7 + k3x3 with stride 1 (2 shapes)**: custom 4-task writer needed for c16_h80_oc128; k7x7 needs fresh capture.
   - **1 `BY_YK` 12-task c576_h19_oc12**: 12-task layout, captured at GEM1 (task-only).
   - **1 crash-fenced c256_h2_oc546**: needs fresh GEM2 capture, then parallel materializer to c256_h2_oc64.
6. **Fence breakdown (80 total, from 11:40 sweep) — full list at `/tmp/fenced_80.txt`:**
   - 36 `BY_K/k_tile` mixed (narrow pointwise wide + depthwise + spatial)
   - 19 `BY_YK` depthwise (c32/c64/c96/c128/c144/c256 h28/h56/h112 + pointwise c256_h28)
   - 13 depthwise `BY_Y/y_tile`
   - 7 `k5x5/k7x7` spatial
   - 4 `NONE/setup` spatial
   - 1 `BY_YK` 12-task (c576_h19_oc12)
   - 1 crash-fenced c256_h2_oc546

---

## 1. Goal (End State)

| Metric | Target |
|---|---:|
| PASS | 217 |
| FENCED | 0 |
| FAIL | 0 |
| ERROR | 0 |
| TIMEOUT | 0 |
| Pre-sweep `python3 examples/simple_add.py` | PASS |
| Post-sweep `python3 examples/simple_add.py` | PASS |
| `git diff` of `examples/conv.py` | reviewable, no debug print, no dead code |
| Manifest entries in `conv_expt/rknn_prefix_replay.py` | one per promoted shape, with `note: "promoted via ..."` |
| Sweep command (final) | `timeout 200 python3 sweep_217.py --skip-health --output-dir /tmp/opencode` reports `total=217 counts={'PASS': 217, 'FENCED': 0}` |

**Why prefix replay:** the NPU driver is correct. The 12-shape rknn_runtime cross-check has proven the NPU computes every representative fenced shape correctly (max err 0.0070-0.0447). The bug is in the register-submit setup that `conv.py` produces. Prefix replay captures the live `regcmd` that the rknn_runtime driver submits, decodes each qword into `(target_id, address, value)`, and replays those exact values in `conv.py` for fenced shapes.

**Safety constraints (from AGENTS.md):**
- Do NOT kill long-running NPU processes (crashes board).
- BE SUPER CAREFUL with `npu_submit`, `task_count`, `regcmd_addr`, `regcfg_amount`, `enable_mask`. Wrong submit parameters crash/reboot.
- NPU soft-resets on CMA pressure; `python3 examples/simple_add.py` is the recovery check.
- Test-run any code changes; sweep must complete with no FAIL/ERROR/TIMEOUT.
- All overrides must be shape-conditional (no global changes to `make_regs`).
- The DMA_CON2 bug pattern (using full-shape in_h in tile helpers) must NOT be reintroduced. Always use the tile's in_h in helper functions.
- `CRASH_FENCED_SHAPES` shapes exit before DRM allocation. Re-introducing direct submits for them requires both a guarded flag and a proven no-submit materializer first.

---

## 2. Current Sweep State (re-verified 2026-06-03 11:40)

```
total=217 counts={'PASS': 137, 'FENCED': 80, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}
pre_health_rc=-1 post_health_rc=-1   # --skip-health passed; simple_add manually verified PASS at 11:48
snapshot: /tmp/opencode/conv_py_217_sweep_20260603_114047_summary.txt
detail: /tmp/opencode/conv_py_217_sweep_20260603_114047_detail.log
fenced list: /tmp/fenced_80.txt (80 lines)
```

### 2.1 Last-turn promotions (2026-06-03 10:30 — committed in db70141)

`b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid` was the +1 promotion this turn. Promoted from crash-fence to PASS via shape-scoped exact11 materializer:
- `_c256_h2_oc64_exact11_task_regs(s, in_dma, wt_dma, out_dma)` emits 11 EXACT11 rows.
- `C256_H2_OC64_EXACT11_OUT_C` = `(64, 32, None, 32, None, 32, None, 16, None, 16, None)` for k splits 0:32 / 32:16 / 48:16.
- Per-row body overrides: cbuf0=0xb1, data_size1=0x003f0100, dma_con2=0x0ffffffc, cvt_con0=0xb.
- Weight layout: `_pack_pointwise_wide` (0x8000 bytes for 64 OC); `weight_size0` per row: `0x8000` (setup), `0x4000` (k_half), `0x2000` (k_tile).
- `input_h=2`, `conv2_low=0x30`.
- `submit_task_number=3`, `subcores=((0,1),(0,1),(0,1),(0,0),(0,0))`.
- Wired through `run_c256_h2_oc64_exact11_shape` invoked via `--allow-c256-h2-oc64-exact11-submit`; standard direct run also PASSes without a guard.
- Shape added to `PREFIX_BY_K_SHAPES`; removed from `CRASH_FENCED_SHAPES`; manifest note updated.

### 2.2 Net session deltas

| Window | PASS | FENCED | Delta | Commit |
|---|---:|---:|---|---|
| 2026-06-02 09:54 (start) | 114 | 103 | — | — |
| 2026-06-02 22:30 (yesterday) | 134 | 83 | +20 / -20 | c256_h2 oc64/oc546 fenced; h7 c512 promoted |
| 2026-06-03 09:49 (today morning) | 136 | 81 | +2 / -2 | c512_h7 pointwise |
| 2026-06-03 10:30 (this turn prior) | 137 | 80 | +1 / -1 | c256_h2_oc64 promoted |
| 2026-06-03 11:40 (this turn now) | **137** | **80** | +0 / +0 | re-verified; no-submit c256_h28 materializer added |
| **Target** | **217** | **0** | +80 promotions needed | |

---

## 3. Fence Breakdown (80 total, from 11:40 sweep)

Full list at `/tmp/fenced_80.txt`. Grouped by family + next step:

### 3.1 `BY_K/k_tile` mixed subfamilies (36 shapes)

The largest bucket. Includes pointwise wide with h14/h7, depthwise h14, spatial h7 k3x3 with stride 1.

**Narrow pointwise 1x1 wide-OC (sub-attack):** c480_h14_oc16, c512_h14_oc24, c832_h7_oc48, c1024_h7_oc1024, c1024_h1_oc1001, c1280_h10_oc24, c1280_h10_oc546, c256_h3_oc546, c256_h3_oc128, c128_h3_oc256, c128_h1_oc24, c256_h2_oc546 (crash-fenced).
- **Strategy:** for each, capture fresh GEM1/GEM2 regcmd via `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1/2`. Decode body fields. Compare against `_exact11_body_regs` to find overrides. Build a shape-scoped materializer like c256_h2_oc64.
- **Caveat:** for c832_h7_oc48, RKNN prefix replay is numerically clean but local visible-row replay returns `inf`. This is the hidden-buffer-parity issue (CNA weight bank rotation, C2 packing). Need to diff live input/weight/output GEM layouts against local buffers before any submit.

**Pointwise 1x1 narrow-OC h14 (sub-attack):** mostly already promoted via `POINTWISE_EXACT11_BYK_SHAPES`. The remaining few are unaligned OC counts.

**Depthwise h14 (sub-attack):** c512_h14_oc512, c1024_h7_oc1024, c1024_h7_oc1024 k3x3. By_K, oc=c.

**Spatial h7 k3x3 (sub-attack):** b1_c192_h7_oc384, b1_c832_h7_oc48, c1024_h7 k3x3. 

### 3.2 `BY_YK` depthwise (19 shapes)

c32_h112, c64_h112, c96_h112, c128_h56, c144_h56, c144_h75, c192_h38, c256_h28, c256_h3, c256_h10, c384_h19, c512_h14, c576_h19, c960_h10 (depthwise, all groups=in_c=oc) + c256_h28 pointwise 1x1 (1 shape) + c576_h19_oc12 (1 shape).
- **Strategy:** row materializer that emits `setup / setup / setup / ppu_pdp / setup / ppu_pdp / y_tile / ppu_pdp / y_tile / ppu_pdp / y_tile / ppu_pdp` pattern (12 rows for c256_h28 depthwise; 15-23 rows for others).
- **Status:** c256_h28 depthwise body field constants extracted in `examples/conv_c256_h28_dw_byyk_no_submit.py` (committed 09bfdf9). Constants: cbuf0=0x1b/0x201b per weight_reuse, data_size1=0x1f00e0, dma_con2=0x2a0, dst_surf_stride=0x2a4, surface_add=0xa90, conv_con1=0x123, conv_con2=0xc0 (feature_grains=12), weight_size0=0xfc0, weight_size1=0xfc0, weight_size2=0x03030001, cbuf_con1=0xc4, cvt_con0=0xb.
- **Next:** build real row materializer that translates planner rows into the RKNN 12-row closure (Attack C step 1).

### 3.3 Depthwise `BY_Y/y_tile` (13 shapes)

c32_h150, c32_h112 (BY_Y, 22 tasks), c64_h112, c128_h56, c256_h3, c256_h10, c512_h5, c576_h20, c768_h20, c96_h20, c384_h10, c512_h5, c96_h20_oc273.
- **Strategy:** single-task rewrite. Use `run_depthwise_shape` body override path. Scaffolded via `examples/conv_depthwise_by_y_layout_no_submit.py` but the live body fields haven't been captured per shape.
- **Next:** per-shape `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1/2` capture, then build `_depthwise_by_y_task_regs(s, in_dma, wt_dma, out_dma)` per shape with `DEPTHWISE_BODY_SHAPES` dict.

### 3.4 `k5x5/k7x7` spatial (7 shapes)

c16_h80_oc128 5x5, c480_h10_oc480 5x5, c576_h20_oc576 5x5, c768_h20_oc768 5x5, c960_h10_oc960 5x5, c1024_h7_oc1024 7x7, c1024_h7_oc1024 3x3 with stride 1.
- **Strategy:** custom multi-task writer. Some might fit `run_exact11_byk_shape` if the body is compatible; others need a dedicated path. Per-shape RKNN capture is mandatory.
- **Next:** Attack F + Attack K (k5x5/k7x7 sparse). For c16_h80_oc128 5x5 specifically, the 3x3 variant needs a 4-task writer (1 preamble + 3 y-tile computes) — the 5x5 might fit a similar pattern with different `weight_size2=0x05050080`.

### 3.5 `NONE/setup` spatial (4 shapes)

c16_h80_oc128 3x3 + 5x5 (also in 3.4), c1024_h7_oc1024 k7x7 + k3x3 with stride 1 (also in 3.4).
- **Strategy:** Attack F. For c16_h80_oc128 3x3, the row materializer has been designed in `KNOWN_BAD_SPATIAL_SETUP_SHAPES` comments but not wired.
- **Next:** build `run_spatial_setup_shape` for c16_h80_oc128 with `cbuf0=0x57`, `weight_size0=0x9000`, `weight_size2=0x03030080`, `conv2_low=0x1a0`. Decoded but never wired.

### 3.6 `BY_YK` 12-task (1 shape)

`b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid` (Attack H).
- **Strategy:** 12 tasks: setup (108q), k_half (108q), 5×(k_tile 104q, ppu_pdp 26q). Like c256_h2_oc64 but with 1 k_half and 5 k_tiles. The k_half uses 108q (with prelude).
- **Evidence:** task-only capture at `/home/orangepi/npu/ops_rknn/dump/prefix_c576_h19_keep1_gem1/dump_gem1.txt`.
- **Next:** capture GEM2 for body fields, then build `_c576_h19_oc12_task_regs` materializer.

### 3.7 Crash-fenced (1 shape)

`b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid` (the crash culprit from earlier session).
- **Strategy:** fresh GEM2 capture via `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=2`. Then parallel materializer to c256_h2_oc64 (also 11-row EXACT11 structure, but with oc=546 → more k_tiles or different c2 packing).
- **Next:** Attack I. Capture first (no submit), then build no-submit materializer matching body field structure, then guarded submit with `--allow-c256-h2-oc546-exact11-submit`.

---

## 4. Next Steps (priority order)

### Step 1 (HIGHEST VALUE, NEXT): Build the c256_h28 depthwise row materializer for real


The no-submit materializer (`examples/conv_c256_h28_dw_byyk_no_submit.py`) has the body field constants but doesn't emit actual regcmd. The next step is to build `_c256_h28_dw_byyk_task_regs(s, in_dma, wt_dma, out_dma)` that produces 12 regcmd rows.

**Structure per row (12 rows total):**
- 2 setup rows (108q each, 4-qword prelude + 104 body): rows 0, 1
- 2 setup/tail rows (104q, body only): rows 2, 4
- 3 y_tile rows (104q, body only): rows 6, 8, 10
- 5 ppu_pdp rows (26q each, PPU/PDP target IDs 0x4001/0x8001): rows 3, 5, 7, 9, 11

**For the 104q body:** model after `_exact11_body_regs` (`examples/conv.py:867`) but with depthwise body fields:
- cbuf0: 0x1b (no reuse) / 0x201b (with reuse)
- data_size1: 0x1f00e0
- dma_con2: 0x02a0
- dst_surf_stride: 0x02a4
- surface_add: 0x0a90
- conv_con1: 0x123
- conv_con2: 0xc0 (setup in_h=28) or 0x100000c0+ for tail/y_tile with family bits
- weight_size0=0xfc0, weight_size1=0xfc0, weight_size2=0x03030001
- cbuf_con1: 0xc4
- cvt_con0: 0xb, cvt_con1-4: 0x10000

**For the 108q setup:** add 4-qword prelude: `(CNA_CBUF_CON0, cbuf0)`, `(CNA, 0x1104, 0)`, `(CNA, 0x1100, 0)`, `(CNA_CONV_CON1, 0x120)`.

**For 26q ppu_pdp:** use `_exact11_aux_regs(s, out_dma)` pattern with `channel_grain=0xff` for c=256, plus depthwise aux_dma offset.

**For the 4-qword prelude, the live capture shows different values:**
- Row 0: `(CBUF_CON0, 0x1b)`, `(CNA, 0x1104, 0)`, `(CNA, 0x1100, 0)`, `(CONV_CON1, 0x123)` — note: CONV_CON1=0x123 not 0x120 (different from pointwise).

**For the c2 input packing:** c256_h28 depthwise has c=256 with c2=8. data_in_channel_real=0x1f, data_in_channel=0xe0 (224). This is the c2-packed width for 8 c2 groups. Use `data_size1 = (data_in_channel_real << 16) | data_in_channel`.

**For the y_tile rows:** y_start and y_size need to be computed. With out_h=25 and 3 tiles of 8/8/7 (or 9/8/8 with overlap), the y_starts are 0/9/17 or similar. Need to determine from the live capture's `feature_off` and `output_off` values.

**Files to touch:**
- `examples/conv.py` — add `_c256_h28_dw_byyk_task_regs`, `run_c256_h28_dw_byyk_shape`, CLI flag `--allow-c256-h28-dw-byyk-submit`, add shape to `DEPTHWISE_BYYK_SHAPES`, add to `PREFIX_BY_K_SHAPES`.

### Step 2 (after Step 1 works): Apply pattern to c32_h112, c64_h112, c96_h112, c128_h56, c144_h56 depthwise

Each has the same 12-row-ish BY_YK structure but with different row counts and in_h values. Build a generic `depthwise_BY_YK_materializer` that takes per-shape config (in_h_list, out_h_list, y_starts) and emits the rows.

**For c32_h112 (15 tasks, c=32, h=112):** the 15-task structure from the layout file is:
amounts=(108,108,12,12,16,16,104,104,104,26,104,104,104,26,104,104,26,104,104,26,104,104,26) — actually 22 tasks. For c32_h112 it's a different counts (need fresh capture).

**For c64_h112 (23 tasks):** amounts from the manifest: `(108,108,12,12,16,16,104,104,104,26,104,104,104,26,104,104,26,104,104,26,104,104,26)`.

### Step 3 (parallel to Step 2): Generalize to other depthwise subfamilies

- c32_h150 (BY_Y, 22 tasks from c32_h150_dw_by_y_keep1)
- c32_h112 (BY_Y, 15 tasks)
- c64_h112 (BY_YK, 23 tasks)
- c96_h112 (BY_YK, 31 tasks) — body fields captured in `examples/conv_depthwise_by_y_layout_no_submit.py`
- c96_h150 (BY_YK)
- c128_h56 (BY_YK, 16 tasks) — body fields captured
- c128_h3 (BY_Y, smaller)
- c144_h56 (BY_YK, 17 tasks) — body fields captured
- c144_h75 (BY_YK)
- c192_h38 (BY_YK)
- c256_h28 (BY_YK, 12 tasks) — IN PROGRESS (Step 1)
- c256_h3 (BY_Y)
- c256_h10 (BY_Y, single-task rewrite — scaffolded)
- c384_h19 (BY_YK)
- c512_h5 (BY_Y)
- c512_h14 (BY_YK, 12 tasks — see c512_h14_dw_by_k in manifest)
- c576_h19 (BY_YK)
- c576_h20 (BY_Y)
- c768_h20 (BY_Y)
- c960_h10 (BY_YK)

### Step 4: Attack F — c16_h80_oc128 (3x3 and 5x5, 2 shapes)

Custom 4-task writer: 1 preamble + 3 y-tile computes. NOT 11-task EXACT11. Body overrides: cbuf0=0x57, weight_size0=0x9000, weight_size2=0x03030080 (3x3) or 0x05050080 (5x5), conv2_low=0x1a0. Decoded but never wired.

**Files to touch:**
- `examples/conv.py` — add `_c16_h80_oc128_task_regs` and `run_c16_h80_oc128_shape` (parameterized by k).
- `conv_expt/rknn_prefix_replay.py` — add to `KNOWN_BAD_SPATIAL_SETUP_SHAPES` resolution path or move to `PREFIX_BY_K_SHAPES` if compatible.

### Step 5: c576_h19_oc12 12-task (1 shape)

12 tasks: setup (108q), k_half (108q), 5×(k_tile 104q, ppu_pdp 26q). Like c256_h2_oc64 but with 1 k_half and 5 k_tiles. The k_half uses 108q (with prelude). Live capture at `/home/orangepi/npu/ops_rknn/dump/prefix_c576_h19_keep1_gem1/dump_gem1.txt` shows task 0 setup 108q, task 1 k_half 108q, then alternating 104q/26q.

**Files to touch:**
- `examples/conv.py` — add `_c576_h19_oc12_task_regs`, `run_c576_h19_oc12_shape`, CLI flag `--allow-c576-h19-oc12-submit`.

### Step 6: Crash-fence recovery — c256_h2_oc546 (1 shape)

The only remaining crash-fenced shape. Needs fresh GEM2 capture via `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=2`, then parallel materializer to c256_h2_oc64.

**Procedure:**
1. Build matched-weight .rknn: `python3 /tmp/build_matched.py b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid`
2. Capture live regcmd via gdb: `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=2`
3. Decode body fields
4. Build no-submit materializer (similar to `examples/conv_c256_h28_dw_byyk_no_submit.py` but for c256_h2_oc546)
5. Validate body field parity
6. Build guarded submit path: `--allow-c256-h2-oc546-exact11-submit`
7. Test with the guarded flag
8. Add to `PREFIX_BY_K_SHAPES`, remove from `CRASH_FENCED_SHAPES`
9. Update manifest

### Step 7: Pointwise 1x1 h*w=1 family (c1024_h1_oc1001, etc.)

5 shapes with c=1024/1280 wide and h*w=1: c1024_h1_oc1001, c128_h1_oc24, c64_h1_oc128 (promoted), c1024_h7_oc1024 (1x1), c1024_h7_oc1024 (k3x3). Similar to c64_h1_oc128 (which uses the 1x1 family body). Need fresh RKNN captures per shape to derive data_size1, weight_size0, etc.

**Files to touch:**
- `examples/conv.py` — extend `PREFIX_BY_K_SHAPES` per shape with body overrides
- `conv_expt/rknn_prefix_replay.py` — manifest entries per shape

### Step 8: Final sweep + commit

```bash
python3 examples/simple_add.py   # pre-check
timeout 200 python3 sweep_217.py --skip-health --output-dir /tmp/opencode
python3 examples/simple_add.py   # post-check
git add examples/conv.py conv_expt/rknn_prefix_replay.py \
        experimental/rknn/dump_rknpu_task_gems.py \
        current_task.md
git commit -m "Promote all 217 conv shapes via prefix-replay"
```

---

## 5. Critical Constants

```python
EXACT11_BYK_AMOUNTS = (108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)  # 11 tasks
EXACT11_BYK_MASKS = (0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60)
EXACT11_BYK_PC_AMOUNTS = (0, 0x1000e, 0, 0x1000e, 0, 0x2000e, 0, 0x2000e, 0, 0x2000e)
EXACT11_BYK_ROLES = ("setup_body", "k_half_body0", "aux0", "k_half_body1", "aux1",
                     "k_tile_body0", "aux2", "k_tile_body1", "aux3", "k_tile_body2", "aux4")

C256_H2_OC64_EXACT11_SHAPE = "b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid"
C256_H2_OC64_EXACT11_OUT_C = (64, 32, None, 32, None, 32, None, 16, None, 16, None)
C256_H2_OC64_EXACT11_WEIGHT_SIZE0 = (0x8000, 0x4000, None, 0x4000, None, 0x4000, None, 0x2000, None, 0x2000, None)
C256_H2_OC64_EXACT11_DST_OFFSETS = (0x0, 0x0, None, 0x100, None, 0x0, None, 0x100, None, 0x180, None)

# c256_h28 depthwise BY_YK (from no-submit materializer)
C256_H28_DW_BYYK_AMOUNTS = (108, 108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)
C256_H28_DW_BYYK_MASKS = (0x0d, 0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60)
C256_H28_DW_BYYK_OFFSETS = (0, 112, 224, 336, 368, 480, 512, 624, 656, 768, 800, 912)
C256_H28_DW_BYYK_INPUT_H = (28, 28, 15, None, 15, None, 11, None, 11, None, 10, None)
C256_H28_DW_BYYK_FAMILIES = ("setup", "setup", "setup", "ppu_pdp", "setup", "ppu_pdp",
                              "y_tile", "ppu_pdp", "y_tile", "ppu_pdp", "y_tile", "ppu_pdp")
C256_H28_DW_BYYK_WEIGHT_REUSE = (0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
C256_H28_DW_BYYK_BODY = {
    "cbuf0_base": 0x01b, "cbuf0_reuse": 0x201b,
    "data_size1": 0x1f00e0, "dma_con2": 0x02a0,
    "dst_surf_stride": 0x02a4, "surface_add": 0x0a90,
    "conv_con1": 0x123, "conv_con2_setup_in_h28": 0xc0,
    "weight_size0": 0xfc0, "weight_size1": 0xfc0, "weight_size2": 0x03030001,
    "cbuf_con1": 0xc4, "cvt_con0": 0xb, "cvt_con_scale": 0x10000,
}

# Target IDs
CNA = 0x0201; CORE = 0x0801; DPU = 0x1001
PC = 0x0081; PC_REG = 0x0101; VERSION = 0x0041
PPU = 0x4001; PDP = 0x8001
RK_CBUF_BANKS = 12; CBUF_BANK_SIZE = 32768
FP16_ATOM_ELEMENTS = 16; UNPACK_C2 = 8
FP16_BYTES = 2; FP32_BYTES = 4
RK_MAX_CONV_FLAT_STRIDE = 992; PC_CHAIN_TAIL_QWORDS = 4

# Override dicts in examples/conv.py L155-241
PREFIX_BY_K_SHAPES = {EXACT11_BYK_SHAPE, "b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid",
    "b1_c16_h80_w80_oc64_wic16_k3x3_g1_s1_pvalid", "b1_c64_h1_w1_oc128_wic64_k1x1_g1_s1_pvalid",
    "b1_c192_h28_w28_oc96_wic192_k1x1_g1_s1_pvalid", "conv2d_cc_b1_c512_h7_w7_oc1024_wic512_k1x1_g1",
    "b1_c512_h7_w7_oc1024_wic512_k1x1_g1", C256_H2_OC64_EXACT11_SHAPE}
```

---

## 6. Critical Files (live, in worktree)

| File | State |
|---|---|
| `examples/conv.py` (2389 lines) | has c256_h2_oc64 materializer + guard + CLI flag wired |
| `conv_expt/rknn_prefix_replay.py` (~700 lines, 67 entries) | manifest updated for c256_h2_oc64 and c256_h28_dw_by_yk notes |
| `examples/conv_c256_h28_dw_byyk_no_submit.py` (143 lines) | NEW no-submit body-field validator for Attack C |
| `examples/conv_no_submit_materializer.py` (106 lines) | NEW Descriptor/EmitterProfile/E() regcmd primitives |
| `examples/conv_no_submit_closure.py` (105 lines) | NEW setup_full_reg_qwords() closure |
| `examples/conv_no_submit_fixtures.py` (123 lines) | NEW K_TILE_RUN_METADATA, planner fixtures |
| `examples/conv_crash_fence_no_submit.py` (209 lines) | NEW c256_h2_oc64 crash-fence body validator |
| `examples/conv_depthwise_by_y_layout_no_submit.py` (283 lines) | NEW depthwise BY_Y layout assertions (c96_h112, c144_h56) |
| `examples/conv_h14_k_tile_no_submit.py` (796 lines) | NEW h14 k_tile layout assertions |
| `examples/conv_h14_task_layout_no_submit.py` (1480 lines) | NEW h14 task layout assertions |
| `examples/conv_h160_setup3_task_layout_no_submit.py` (380 lines) | NEW h160 setup3 layout |
| `examples/conv_output_bo_map_no_submit.py` (60 lines) | NEW output BO mapping |
| `examples/conv_pointwise_by_k_layout_no_submit.py` (261 lines) | NEW pointwise BY_K layout |
| `examples/conv_pointwise_by_yk_layout_no_submit.py` (155 lines) | NEW pointwise BY_YK layout |
| `examples/conv_small_no_submit.py` (180 lines) | NEW small-shape assertions |
| `examples/conv_spatial_by_y_layout_no_submit.py` (353 lines) | NEW spatial BY_Y layout |
| `examples/conv_tiles_no_submit.py` (126 lines) | NEW tile helpers |
| `examples/conv_legacy.py` (989 lines) | NEW old reference (kept for diff comparison) |
| `experimental/rknn/dump_rknpu_task_gems.py` | has `--scan-regcmd` flag |
| `current_task.md` (this file) | comprehensive tracker |
| `/tmp/opencode/conv_py_217_sweep_20260603_114047_summary.txt` | latest sweep |
| `/tmp/fenced_80.txt` | 80 fenced shape names |

---

## 7. Live Regcmd Captures (cached in /tmp/, /home/orangepi/)

| Capture | Shape | Region | Decoded? |
|---|---|---|---|
| `/tmp/c64_h1_oc128_v3.log` | c64_h1_oc128 1x1 (PROMOTED) | GEM 2:0x4000-0x7000 | YES |
| `/tmp/c16_h80_oc64_capture_v10.log` | c16_h80_oc64 3x3 (PROMOTED) | GEM 2:0x5000 | YES |
| `/tmp/c16_h80_oc128_full3.log` | c16_h80_oc128 3x3 | GEM 2:0xA000 | YES (11-task, not 4-task) |
| `/home/orangepi/npu/ops_rknn/dump/prefix_c256_h28_dw_by_yk_keep1_gem2/dump_gem2.txt` | c256_h28 depthwise | GEM 2:0x2000-0x4000 | YES — body fields extracted |
| `/home/orangepi/npu/ops_rknn/dump/prefix_c128_h56_dw_by_yk_keep1_gem{1,2}/` | c128_h56 depthwise | GEM 1+2 | task-only |
| `/home/orangepi/npu/ops_rknn/dump/prefix_c144_h56_dw_by_yk_keep1_gem{1,2}/` | c144_h56 depthwise | GEM 1+2 | task-only |
| `/home/orangepi/npu/ops_rknn/dump/prefix_c96_h112_dw_by_yk_keep1_gem{1,2}/` | c96_h112 depthwise | GEM 1+2 | task-only |
| `/home/orangepi/npu/ops_rknn/dump/prefix_c256_h28_dw_by_yk_keep1_gem{1,2}/` | c256_h28 depthwise | GEM 1+2 | task + body |
| `/home/orangepi/npu/ops_rknn/dump/prefix_c512_h14_dw_by_k_keep1_gem{1,2}/` | c512_h14 depthwise BY_K | GEM 1+2 | task + body (Attack B) |
| `/home/orangepi/npu/ops_rknn/dump/prefix_c576_h19_keep1_gem1/dump_gem1.txt` | c576_h19_oc12 12-task | GEM 1 | task-only |

---

## 8. Test Runner Commands

```bash
# Single shape (guarded for c256_h2_oc64)
timeout 30 python3 examples/conv.py <shape> [--allow-c256-h2-oc64-exact11-submit]

# Full sweep
timeout 200 python3 sweep_217.py --skip-health --output-dir /tmp/opencode

# NPU health check
python3 examples/simple_add.py

# Dry run (no DRM allocation)
python3 examples/conv.py <shape> --dry-run-exact11-byk

# No-submit body field check (Attack C)
python3 examples/conv_c256_h28_dw_byyk_no_submit.py --check all

# Coverage report
python3 conv_expt/rknn_prefix_replay.py --coverage-summary /tmp/opencode/conv_py_217_sweep_20260603_114047_summary.txt

# Scan regcmd-only dump (new in db70141)
python3 experimental/rknn/dump_rknpu_task_gems.py --scan-regcmd --min-regcmd-run 2
```

---

## 9. Pattern for New Shape Promotion (8 steps)

1. **Build matched-weight .rknn**: `python3 /tmp/build_matched.py <shape_name>`
2. **Capture live regcmd via gdb**: `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1/2`
3. **Decode each qword**: `target=(qword>>48)&0xffff, value=(qword>>16)&0xffffffff, addr=qword&0xffff`
4. **Diff against `_exact11_body_regs`** to find which fields differ
5. **Add per-shape overrides** to `*_OVERRIDES` dicts in `examples/conv.py`
6. **Test single shape** with a guarded flag
7. **Add to promoted set** + manifest entry
8. **Re-sweep and verify**

---

## 10. Distance to Goal

- 80 fenced → 0 fenced (need 80 promotions)
- Largest blocks:
  - 36 `BY_K/k_tile` (need per-shape body captures) — **biggest single bucket**
  - 19 `BY_YK` (need row materializers for depthwise and c256_h28 1x1)
  - 13 depthwise `BY_Y` (need single-task body wire-up)
  - 7 `k5x5/k7x7` (rare kernel sizes)
  - 4 `NONE/setup` (Attack F)
  - 1 c576_h19_oc12 (12-task)
  - 1 c256_h2_oc546 (crash-fence, no capture)

---

## 11. Session continuity notes

- This file is the authoritative handoff document. Update it after every promotion, sweep, or materializer addition.
- The session has not been rebooted this turn. NPU health is green (`simple_add.py` PASS at 11:48).
- The crash-fenced `b1_c256_h2_w2_oc546_*` shape still exits before DRM allocation; do NOT run it directly. It needs a fresh GEM2 capture + no-submit materializer + guarded submit path.
- All uncommitted changes to `examples/conv.py`, `conv_expt/rknn_prefix_replay.py`, `experimental/rknn/dump_rknpu_task_gems.py` are reviewable and safe to keep. The untracked no-submit materializer files are no-submit (don't submit to NPU) and are safe.
- If a future crash happens, recover with `python3 examples/simple_add.py` (must PASS) and re-sweep with `timeout 200 python3 sweep_217.py --skip-health --output-dir /tmp/opencode`.

---

## 12. Per-Family Progress & Capture Status (re-verified 2026-06-03 12:14)

### 12.1 Summary by family

| Family | # Fenced | GEM1 | GEM2 | %G1 | Distinct capture slugs |
|---|---:|---:|---:|---:|---|
| depthwise 3x3 (k3_g=in_c) | 30 | 12 | 11 | 40% | c32_h150_dw_by_y, c96_h112_dw_by_yk, c96_h150_dw_by_yk, c128_h56_dw_by_yk, c144_h56_dw_by_yk, c256_h28_dw_by_yk, c256_h3_none, c512_h14_dw_by_k, c576_h19, cc_c32_h112_dw_by_y |
| depthwise 5x5 (k5_g=in_c) | 4 | 0 | 0 | **0%** | — (no captures exist) |
| depthwise 7x7 (k7_g=in_c) | 2 | 0 | 0 | **0%** | — (no captures exist) |
| pointwise 1x1 (k1_g1) | 36 | 14 | 14 | 39% | c1024_h1_oc1001_pw_by_k, c128_h1_none, c256_h2_none, c256_h2_oc64, c256_h3_none, c512_h14_oc24, c576_h19, c832_h7_oc48 |
| spatial 3x3 (k3_g1) | 7 | 1 | 1 | 14% | c40_h40_oc320 (only this slug; 6/7 have nothing) |
| spatial 5x5 (k5_g1) | 1 | 0 | 0 | **0%** | — (no captures exist) |
| **Total** | **80** | **27** | **26** | **34%** | — |

**Bottom line: 53 of 80 fenced shapes (66%) have NO capture at all.** 27/80 (34%) have at least a GEM1 task-descriptor capture, of which 26 also have a GEM2 body capture. To promote any of the 53 uncaptured shapes, a fresh `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1/2` capture must be run.

### 12.2 Detailed table — every fenced shape

#### depthwise 3x3 (k3_g=in_c) — 30 shapes, 12 captured (40%)

| Shape | G1 | G2 | Capture slug(s) |
|---|:-:|:-:|---|
| b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024 | - | - | — |
| b1_c128_h3_w3_oc128_wic1_k3x3_g128_s1_pvalid | - | - | — |
| b1_c128_h56_w56_oc128_wic1_k3x3_g128 | Y | Y | c128_h56_dw_by_yk |
| b1_c144_h56_w56_oc144_wic1_k3x3_g144_s1_pvalid | Y | Y | c144_h56_dw_by_yk |
| b1_c144_h75_w75_oc144_wic1_k3x3_g144_s1_pvalid | - | - | — |
| b1_c192_h38_w38_oc192_wic1_k3x3_g192_s1_pvalid | - | - | — |
| b1_c256_h10_w10_oc256_wic1_k3x3_g256_s1_pvalid | - | - | — |
| b1_c256_h28_w28_oc256_wic1_k3x3_g256 | Y | Y | c256_h28_dw_by_yk (body fields in `examples/conv_c256_h28_dw_byyk_no_submit.py`) |
| b1_c256_h3_w3_oc256_wic1_k3x3_g256_s1_pvalid | Y | Y | c256_h3_none |
| b1_c320_h40_w40_oc320_wic1_k3x3_g320_s1_pvalid | - | - | — |
| b1_c32_h112_w112_oc32_wic1_k3x3_g32 | - | - | — |
| b1_c32_h150_w150_oc32_wic1_k3x3_g32_s1_pvalid | Y | Y | c32_h150_dw_by_y (body fields in `conv_depthwise_by_y_layout_no_submit.py`) |
| b1_c384_h10_w10_oc384_wic1_k3x3_g384_s1_pvalid | - | - | — |
| b1_c384_h19_w19_oc384_wic1_k3x3_g384_s1_pvalid | - | - | — |
| b1_c512_h14_w14_oc512_wic1_k3x3_g512 | Y | Y | c512_h14_dw_by_k |
| b1_c512_h5_w5_oc512_wic1_k3x3_g512_s1_pvalid | - | - | — |
| b1_c576_h19_w19_oc576_wic1_k3x3_g576_s1_pvalid | Y | Y | c576_h19 (task-only; 5x fp16 data in GEM5) |
| b1_c576_h20_w20_oc576_wic1_k3x3_g576_s1_pvalid | - | - | — |
| b1_c64_h112_w112_oc64_wic1_k3x3_g64 | - | - | — |
| b1_c768_h20_w20_oc768_wic1_k3x3_g768_s1_pvalid | - | - | — |
| b1_c960_h10_w10_oc960_wic1_k3x3_g960_s1_pvalid | - | - | — |
| b1_c96_h112_w112_oc96_wic1_k3x3_g96_s1_pvalid | Y | Y | c96_h112_dw_by_yk (body fields in `conv_depthwise_by_y_layout_no_submit.py`) |
| b1_c96_h150_w150_oc96_wic1_k3x3_g96_s1_pvalid | Y | - | c96_h150_dw_by_yk (GEM1 only) |
| b1_c96_h20_w20_oc96_wic1_k3x3_g96_s1_pvalid | - | - | — |
| conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k3x3_g1024 | - | - | — |
| conv2d_cc_b1_c128_h56_w56_oc128_wic1_k3x3_g128 | Y | Y | c128_h56_dw_by_yk |
| conv2d_cc_b1_c256_h28_w28_oc256_wic1_k3x3_g256 | Y | Y | c256_h28_dw_by_yk (cc alias of b1) |
| conv2d_cc_b1_c32_h112_w112_oc32_wic1_k3x3_g32 | - | - | — |
| conv2d_cc_b1_c512_h14_w14_oc512_wic1_k3x3_g512 | Y | Y | c512_h14_dw_by_k (cc alias) |
| conv2d_cc_b1_c64_h112_w112_oc64_wic1_k3x3_g64 | - | - | — |

#### depthwise 5x5 (k5_g=in_c) — 4 shapes, 0 captured (0%)

| Shape | G1 | G2 | Capture slug |
|---|:-:|:-:|---|
| b1_c480_h10_w10_oc480_wic1_k5x5_g480_s1_pvalid | - | - | — |
| b1_c576_h20_w20_oc576_wic1_k5x5_g576_s1_pvalid | - | - | — |
| b1_c768_h20_w20_oc768_wic1_k5x5_g768_s1_pvalid | - | - | — |
| b1_c960_h10_w10_oc960_wic1_k5x5_g960_s1_pvalid | - | - | — |

#### depthwise 7x7 (k7_g=in_c) — 2 shapes, 0 captured (0%)

| Shape | G1 | G2 | Capture slug |
|---|:-:|:-:|---|
| b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024 | - | - | — |
| conv2d_cc_b1_c1024_h7_w7_oc1024_wic1_k7x7_g1024 | - | - | — |

#### pointwise 1x1 (k1_g1) — 36 shapes, 14 captured (39%)

| Shape | G1 | G2 | Capture slug(s) |
|---|:-:|:-:|---|
| b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1 | Y | Y | c1024_h1_oc1001_pw_by_k |
| b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1 | - | - | — |
| b1_c1280_h10_w10_oc24_wic1280_k1x1_g1 | - | - | — |
| b1_c1280_h10_w10_oc24_wic1280_k1x1_g1_s1_pvalid | - | - | — |
| b1_c1280_h10_w10_oc546_wic1280_k1x1_g1 | - | - | — |
| b1_c1280_h10_w10_oc546_wic1280_k1x1_g1_s1_pvalid | - | - | — |
| b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid | Y | Y | c128_h1_none |
| b1_c128_h2_w2_oc256_wic128_k1x1_g1_s1_pvalid | - | - | — |
| b1_c128_h3_w3_oc256_wic128_k1x1_g1_s1_pvalid | - | - | — |
| b1_c256_h28_w28_oc256_wic256_k1x1_g1 | Y | Y | c256_h28_dw_by_yk (14-row BY_YK closure in `conv_pointwise_by_yk_layout_no_submit.py` cc_c256_h28) |
| b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid | Y | Y | c256_h2_none (CRASH-FENCED — no fresh GEM2 for oc546 yet) |
| b1_c256_h3_w3_oc128_wic256_k1x1_g1_s1_pvalid | Y | Y | c256_h3_none |
| b1_c256_h3_w3_oc546_wic256_k1x1_g1_s1_pvalid | Y | Y | c256_h3_none (no oc546-specific dump) |
| b1_c288_h20_w20_oc72_wic288_k1x1_g1_s1_pvalid | - | - | — |
| b1_c320_h20_w20_oc72_wic320_k1x1_g1_s1_pvalid | - | - | — |
| b1_c384_h10_w10_oc546_wic384_k1x1_g1_s1_pvalid | - | - | — |
| b1_c384_h19_w19_oc64_wic384_k1x1_g1_s1_pvalid | - | - | — |
| b1_c384_h19_w19_oc96_wic384_k1x1_g1_s1_pvalid | - | - | — |
| b1_c480_h10_w10_oc120_wic480_k1x1_g1_s1_pvalid | - | - | — |
| b1_c480_h14_w14_oc16_wic480_k1x1_g1 | - | - | — |
| b1_c512_h14_w14_oc24_wic512_k1x1_g1 | Y | Y | c512_h14_oc24 |
| b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid | Y | Y | c576_h19 (no-submit materializer in `examples/conv_c576_h19_oc12_no_submit.py`) |
| b1_c576_h19_w19_oc273_wic576_k1x1_g1_s1_pvalid | Y | Y | c576_h19 (task-only) |
| b1_c576_h19_w19_oc96_wic576_k1x1_g1_s1_pvalid | Y | Y | c576_h19 (task-only) |
| b1_c576_h20_w20_oc72_wic576_k1x1_g1_s1_pvalid | - | - | — |
| b1_c576_h20_w20_oc96_wic576_k1x1_g1_s1_pvalid | - | - | — |
| b1_c72_h20_w20_oc576_wic72_k1x1_g1_s1_pvalid | - | - | — |
| b1_c768_h10_w10_oc120_wic768_k1x1_g1_s1_pvalid | - | - | — |
| b1_c768_h20_w20_oc96_wic768_k1x1_g1_s1_pvalid | - | - | — |
| b1_c832_h7_w7_oc48_wic832_k1x1_g1 | Y | Y | c832_h7_oc48 (body fields in `conv_pointwise_by_k_layout_no_submit.py`) |
| b1_c832_h7_w7_oc48_wic832_k1x1_g1_s1_pvalid | Y | Y | c832_h7_oc48 (alias) |
| b1_c960_h10_w10_oc120_wic960_k1x1_g1_s1_pvalid | - | - | — |
| b1_c96_h20_w20_oc273_wic96_k1x1_g1_s1_pvalid | - | - | — |
| conv2d_cc_b1_c1024_h1_w1_oc1001_wic1024_k1x1_g1 | Y | Y | c1024_h1_oc1001_pw_by_k (cc alias) |
| conv2d_cc_b1_c1024_h7_w7_oc1024_wic1024_k1x1_g1 | - | - | — |
| conv2d_cc_b1_c256_h28_w28_oc256_wic256_k1x1_g1 | Y | Y | c256_h28_dw_by_yk (cc alias) |

#### spatial 3x3 (k3_g1) — 7 shapes, 1 captured (14%)

| Shape | G1 | G2 | Capture slug |
|---|:-:|:-:|---|
| b1_c128_h3_w3_oc256_wic128_k3x3_g1_s1_pvalid | - | - | — |
| b1_c128_h5_w5_oc256_wic128_k3x3_g1_s1_pvalid | - | - | — |
| b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid | - | - | — |
| b1_c192_h7_w7_oc384_wic192_k3x3_g1_s1_pvalid | - | - | — |
| b1_c256_h10_w10_oc512_wic256_k3x3_g1_s1_pvalid | - | - | — |
| b1_c40_h40_w40_oc160_wic40_k3x3_g1_s1_pvalid | Y | Y | c40_h40_oc320 (oc320 alias; oc160-specific not captured) |
| b1_c72_h20_w20_oc288_wic72_k3x3_g1_s1_pvalid | - | - | — |

#### spatial 5x5 (k5_g1) — 1 shape, 0 captured (0%)

| Shape | G1 | G2 | Capture slug |
|---|:-:|:-:|---|
| b1_c16_h80_w80_oc128_wic16_k5x5_g1_s1_pvalid | - | - | — |

### 12.3 Capture coverage gap

| Status | Count | % | Action needed |
|---|---:|---:|---|
| GEM1 + GEM2 captured | 26 | 33% | Build materializer + guarded submit |
| GEM1 task-only | 1 | 1% | Capture GEM2 body |
| NO capture | 53 | 66% | Run `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1/2` first |
| **Total** | **80** | **100%** | |

**To eliminate the 80 fenced shapes, we need 53 fresh captures first.** Each capture is a 30-60s `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=1/2` invocation with a matched-weight `.rknn`. The 27 already-captured shapes can move to the materializer/submit step without re-capture.

### 12.4 Distinct capture slugs available in `/home/orangepi/npu/ops_rknn/dump/`

```
c1024_h1_oc1001_pw_by_k, c128_h1_none, c128_h56_dw_by_yk, c144_h56_dw_by_yk,
c192_h28_oc96, c256_h2_none, c256_h2_oc64, c256_h28_dw_by_yk, c256_h3_none,
c32_h150_dw_by_y, c40_h40_oc320, c512_h14_dw_by_k, c512_h14_oc24,
c512_h14_oc24_baseline, c512_h7_oc1024_pw_by_k, c576_h19, c576_h19_oc12,
c8_h160, c832_h7_oc48, c96_h112_dw_by_yk, c96_h150_dw_by_yk, c96_h56,
cc_c32_h112_dw_by_y, dw_c96_h20
```

Plus older log captures in `experimental/rknn/capture_*.log` (not prefixed-prefix dumps; need conversion to `prefix_*` form for the new tooling).

