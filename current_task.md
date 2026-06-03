# Current Task: Prefix-Replay Debug of All FENCED Shapes in `examples/conv.py`

**Last updated:** 2026-06-03 13:15 (Asia/Shanghai)
**Owner:** Codex session (multi-model handoff)
**CWD:** `/home/orangepi/rk3588`
**NPU health (verified 13:15):** `python3 examples/simple_add.py` PASS — board has NOT been rebooted this session.
**HEAD:** `cffb380` (current_task refresh). Working tree has manifest updates (4 new entries) staged for commit.
**Latest full sweep:** `timeout 200 python3 sweep_217.py --skip-health` →
```
total=217 counts={'PASS': 137, 'FENCED': 80, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}
pre_health_rc=-1 post_health_rc=-1
snapshot: /home/orangepi/rk3588/sweep_results/conv_py_217_sweep_20260603_131332_summary.txt
detail:   /home/orangepi/rk3588/sweep_results/conv_py_217_sweep_20260603_131332_detail.log
current:  /home/orangepi/rk3588/sweep_results/conv_py_217_current_summary.txt (snapshot of latest)
fenced:   /home/orangepi/rk3588/sweep_results/fenced_80.txt (80 lines)
```

---

## 0. TL;DR

1. **Goal:** use **prefix replay** to debug and fix every FENCED shape in `examples/conv.py`. End state: **PASS=217, FENCED=0, FAIL=0, ERROR=0, TIMEOUT=0**, with pre/post `python3 examples/simple_add.py` both PASS.
2. **Current pass progress:** **PASS=137 / 217 (63.1%)**, **FENCED=80 (36.9%)**, **FAIL=0, ERROR=0, TIMEOUT=0**. NPU healthy.
3. **Capture status:** **100% (80/80 fenced shapes have BOTH GEM1 and GEM2 captures)**. Captures stored at `/home/orangepi/npu/ops_rknn/dump/prefix_<slug>_keep1_gem{1,2}/` (NEVER `/tmp`). 117 distinct capture slugs in dump, 94 in `prefix_*_keep1_gem{1,2}/` form.
4. **This turn's progress (2026-06-03 13:15):** verified 12 "newly promoted" shapes from the objective are ALL PASSING in the latest sweep. Added 4 missing manifest entries to `conv_expt/rknn_prefix_replay.py` for shapes that were promoted in code but not yet in the manifest. Re-ran sweep, confirmed 137/80 unchanged. Manifest parses OK.
5. **Did we do prefix replay?** YES. Prefix replay IS the methodology. For each fenced shape, gdb captures the live regcmd that the rknn_runtime driver submits, decodes each qword to `(target_id, address, value)`, then replays those exact values in `conv.py`.
6. **Storage rule (non-negotiable):** NEVER store important files in `/tmp` — they are lost on crash/reboot. All captures go to `/home/orangepi/npu/ops_rknn/dump/`. All sweep outputs go to `/home/orangepi/rk3588/sweep_results/`.

---

## 1. Pass-Progress Table

| Date | PASS | FENCED | Note |
|---|---:|---:|---|
| 2026-06-02 09:54 | 114 | 103 | user-stated stuck baseline |
| 2026-06-02 14:43 | 115 | 102 | +1 first probe |
| 2026-06-02 22:30 | 134 | 83 | c256_h2 oc64/oc546 fenced; h7 c512 promoted |
| 2026-06-03 09:49 | 136 | 81 | c512_h7 pointwise promoted |
| 2026-06-03 10:30 | 137 | 80 | c256_h2_oc64 promoted via EXACT11 materializer |
| 2026-06-03 12:51 | 137 | 80 | 100% capture coverage achieved (80/80 GEM1+2) |
| 2026-06-03 12:55 | 137 | 80 | current_task.md refreshed |
| 2026-06-03 13:15 | **137** | **80** | **CURRENT** — 12 newly promoted shapes verified, 4 manifest entries added |
| **Target** | **217** | **0** | +80 promotions needed |

**Net session delta vs user-stated baseline (114):** **+23 PASS, -23 FENCED**. The 137 PASS plateau is real — no new promotions since `db70141` (c256_h2_oc64 at 10:30). All progress this turn was on the manifest/capture/document side, not new code promotions.

---

## 2. This Turn's Verification (2026-06-03 13:15)

### 2.1 Verified: 12 "newly promoted" shapes from objective — all PASS in latest sweep

| # | Shape | Family | Status | Manifest entry |
|---|---|---|---|---|
| 1 | `conv2d_b1_c96_h56_w56_oc24_wic96_k1x1_g1` | pointwise BY_Y | **PASS** (max_diff=0.0156) | `c96_h56_pw_by_y` (line 218) |
| 2 | `conv2d_b1_c144_h56_w56_oc24_wic144_k1x1_g1` | pointwise BY_Y | **PASS** (max_diff=0.0263) | `c144_h56_pw_by_y` (line 263) |
| 3 | `b1_c144_h75_w75_oc24_wic144_k1x1_g1_s1_pvalid` | pointwise BY_Y | **PASS** (max_diff=0.0238) | `c144_h75_pw_by_y` (line 268) |
| 4 | `b1_c144_h38_w38_oc32_wic144_k1x1_g1_s1_pvalid` | pointwise BY_Y | **PASS** (max_diff=0.0156) | `c144_h38_pw_by_y` (line 273) |
| 5 | `b1_c192_h38_w38_oc32_wic192_k1x1_g1_s1_pvalid` | pointwise BY_Y | **PASS** (max_diff=0.0300) | `c192_h38_pw_by_y` (line 278) |
| 6 | `b1_c128_h80_w80_oc16_wic128_k1x1_g1_s1_pvalid` | pointwise BY_Y | **PASS** (max_diff=0.0245) | `c128_h80_pw_by_y` (line 468) |
| 7 | `b1_c64_h80_w80_oc16_wic64_k1x1_g1_s1_pvalid` | pointwise BY_Y | **PASS** (max_diff=0.0156) | `c64_h80_pw_by_y` (line 473) |
| 8 | `b1_c96_h20_w20_oc12_wic96_k1x1_g1_s1_pvalid` | pointwise BY_Y | **PASS** (max_diff=0.0156) | **NEW** `c96_h20_oc12_pw_by_y` (this turn) |
| 9 | `conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1` | spatial BY_Y | **PASS** (max_diff=0.0151) | `c3_h224_spatial_by_y` (line 207) |
| 10 | `b1_c3_h224_w224_oc32_wic3_k3x3_g1` | spatial BY_Y | **PASS** (max_diff=0.0151) | **NEW** `b1_c3_h224_spatial_by_y` (this turn) |
| 11 | `conv2d_b1_c4_h1_w1_oc2_wic2_k1x1_g2` | grouped_serial | **PASS** (max_diff=0.0004) | **NEW** `g2_c4_h1_w1_oc2_serial` (this turn) |
| 12 | `b1_c4_h1_w1_oc4_wic2_k1x1_g2` | grouped_serial | **PASS** (max_diff=0.0006) | **NEW** `g2_c4_h1_w1_oc4_serial` (this turn) |

**All 12 newly promoted shapes are PASSING in the latest sweep (131332) and have manifest entries.** 4 of the 12 had no manifest entry before this turn; those 4 are added now.

### 2.2 Discrepancy correction: "Important failed/kept-fenced findings" from objective vs. actual

The objective listed 4 "kept fenced" findings. Cross-checked against the latest sweep:

| Objective claim | Actual status | Notes |
|---|---|---|
| `b1_c576_h19_w19_oc12_*` timed out with local replay; remains fenced | **CONFIRMED FENCED** | Real blocker. 12-task closure (108/108/104/26 × 5) needed. |
| `b1_c8_h160_w160_oc16_*` prefix 138/20 windows; local replay numerically wrong; remains fenced | **INCORRECT** | Actually **PASS** in latest sweep via `c8_h160_oc16_promoted_via_local_tile_replay` (line 512 in manifest, "promoted via local_tile_replay"). The 138/20 was the GEM5 output dump artifact (a separate, kept-fenced evidence entry); the actual shape is promoted. |
| Broad grouped spatial serial still times out on `conv2d_b1_c3_h11_w28_oc3_wic1_k3x3_g3`; grouped default remains fenced except the two proven pointwise 1x1 cases | **INCORRECT** | Actually **PASS** in latest sweep via `g3_c3_h11_w28_spatial_serial` (line 348, "promoted after one-off local grouped_serial replay passed"). Sibling `conv2d_b2_c3_h11_w28_oc3_wic1_k3x3_g3` also passes (6 submits). The "2 proven pointwise 1x1 cases" from the objective are now 3 (`c2_wic2`, `c4_wic2` are added this turn, plus the original conv2d_b1_c4_h1_w1_oc2_wic2_k1x1_g2 set). |
| Simple OC-tile replay for BY_K `b1_c40_h40_w40_oc320_wic40_k1x1_g1_s1_pvalid` is numerically wrong, confirming BY_K still needs RKNN closure semantics | **INCORRECT** | Actually **PASS** in latest sweep via `c40_h40_oc320_pw_by_yk` (line 489, "promoted via RKNN prefix replay and local one-task setup108 compact-weight replay"). max_diff=0.0155. This is a counter-example to the broad claim that BY_K always fails. |

**Net result of corrections:** 3 of the 4 "kept fenced" findings are outdated. The actual remaining blocker is the c576_h19_oc12 12-task closure (Attack H), plus the 80 still-fenced shapes that span 5 different split families.

### 2.3 This turn's commit-ready work

- `conv_expt/rknn_prefix_replay.py`: added 4 manifest entries (24 new lines):
  - `c96_h20_oc12_pw_by_y` (after `c64_h56_pw_by_y`)
  - `b1_c3_h224_spatial_by_y` (after `c3_h224_spatial_by_y`)
  - `g2_c4_h1_w1_oc2_serial` (after `g2_c4_h5_oc4_spatial_serial`)
  - `g2_c4_h1_w1_oc4_serial`
- Manifest file: 947 → 971 lines. Parses OK as Python.
- Sweep: re-ran 131332, **137/80 unchanged** (no regressions from manifest edits, which are documentation only).
- `sweep_results/conv_py_217_current_summary.txt`: latest snapshot saved (NOT in /tmp).

---

## 3. Per-Family Progress and Capture Status

### 3.1 Summary by family

| Family | # Fenced | GEM1 | GEM2 | %G2 | Promoted (this turn) | Materializer status |
|---|---:|---:|---:|---:|---:|---|
| depthwise 3x3 (k3_g=in_c) | 30 | 30 | 30 | **100%** | 0 | c256_h28 + c32_h150 + c96_h112 have body fields (no-submit). Others need row materializers. |
| depthwise 5x5 (k5_g=in_c) | 4 | 4 | 4 | **100%** | 0 | Captures only; no materializer. |
| depthwise 7x7 (k7_g=in_c) | 2 | 2 | 2 | **100%** | 0 | Captures only; no materializer. |
| pointwise 1x1 (k1_g1) | 36 | 36 | 36 | **100%** | 0 (8 confirmed pre-existing) | EXACT11 materializer (c256_h2_oc64) + 8 pointwise BY_Y local tile replays + 12 BY_YK closures + 14-row cc_c256_h28 closure. 35 to go. |
| spatial 3x3 (k3_g1) | 7 | 7 | 7 | **100%** | 0 (2 confirmed pre-existing) | c3_h224 prefix-derived (2 variants), c16_h80_oc128 4-task (Attack F), g3_c3_h11_w28 grouped_serial, g2_c4_h5 spatial_serial (4 variants), c8_h160_oc16, c40_h40_oc320 |
| spatial 5x5 (k5_g1) | 1 | 1 | 1 | **100%** | 0 | Captures only. |
| **Total** | **80** | **80** | **80** | **100%** | **0 this turn** | — |

### 3.2 Capture coverage gap

| Status | Count | % |
|---|---:|---:|
| GEM1 + GEM2 captured | **80** | **100%** |
| GEM1 task-only | 0 | 0% |
| NO capture | 0 | 0% |
| **Total** | **80** | **100%** |

**The capture gap is fully closed.** The remaining 80 promotions are the bottleneck.

### 3.3 Remaining fence groups (per objective)

| Group | Count | Status |
|---|---:|---|
| BY_K/k_tile closure needed | 56 | 4 of these already pass via special paths (c40_h40_oc320, c256_h2_oc64 EXACT11, etc.). Most still need parameterized EXACT11 materializer. |
| grouped/depthwise still fenced | 54 | Wide range: c32_h112, c64_h112, c96_h112, c128_h56, c144_h56, c144_h75, c192_h38, c256_h28, c256_h3, c256_h10, c384_h19, c512_h14, c576_h19, c960_h10 + grouped_serial variants |
| BY_YK unresolved | 15 | Need parameterized row materializer (setup/setup/setup/ppu_pdp/setup/ppu_pdp/y_tile/ppu_pdp/y_tile/ppu_pdp/y_tile/ppu_pdp) |
| pointwise-wide NONE closure needed | 6 | Narrow-OC pointwise 1x1 with h*w=1 or h<10 (c1024_h1_oc1001, c128_h1_oc24, etc.) |
| known-bad spatial setup | 4 | c16_h80_oc128 k3x3 + k5x5, k5x5/k7x7 depthwise (c480, c576, c768, c960) |
| pointwise-wide BY_Y | 1 | c256_h28_oc16 (max_diff=119.6110 — kept fenced, needs RKNN closure) |
| spatial BY_Y closure | 1 | c8_h160_oc16 — actually **PROMOTED** (line 512 manifest) but the GEM5 138/20 artifact is a separate kept-fenced evidence entry |

---

## 4. Critical Constants

```python
# 11-row EXACT11 (c256_h2_oc64 template)
EXACT11_BYK_AMOUNTS = (108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)
EXACT11_BYK_MASKS = (0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60)
EXACT11_BYK_PC_AMOUNTS = (0, 0x1000e, 0, 0x1000e, 0, 0x2000e, 0, 0x2000e, 0, 0x2000e)
EXACT11_BYK_ROLES = ("setup_body", "k_half_body0", "aux0", "k_half_body1", "aux1",
                     "k_tile_body0", "aux2", "k_tile_body1", "aux3", "k_tile_body2", "aux4")

# c256_h2_oc64 (PROMOTED as template)
C256_H2_OC64_EXACT11_SHAPE = "b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid"
C256_H2_OC64_EXACT11_OUT_C = (64, 32, None, 32, None, 32, None, 16, None, 16, None)

# c256_h28 depthwise BY_YK (12 rows, body fields in no-submit materializer)
C256_H28_DW_BYYK_AMOUNTS = (108, 108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)
C256_H28_DW_BYYK_MASKS = (0x0d, 0x0d, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60, 0x0d, 0x60)
C256_H28_DW_BYYK_OFFSETS = (0, 112, 224, 336, 368, 480, 512, 624, 656, 768, 800, 912)
C256_H28_DW_BYYK_BODY = {
    "cbuf0_base": 0x01b, "cbuf0_reuse": 0x201b,
    "data_size1": 0x1f00e0, "dma_con2": 0x02a0,
    "dst_surf_stride": 0x02a4, "surface_add": 0x0a90,
    "conv_con1": 0x123, "conv_con2_setup_in_h28": 0xc0,
    "weight_size0": 0xfc0, "weight_size1": 0xfc0, "weight_size2": 0x03030001,
    "cbuf_con1": 0xc4, "cvt_con0": 0xb, "cvt_con_scale": 0x10000,
}

# c576_h19_oc12 12-task (FENCED, in no-submit materializer)
# amounts = (108, 108, 104, 26, 104, 26, 104, 26, 104, 26, 104, 26)
# k_tile_splits: 4+3+2+1+2=12 for c576_h19_oc12 (oc=12)
# cbuf0=0xb1, data_size1=0x8f0240, conv2_low=0x30 (educated guesses from c256_h2_oc64)

# Target IDs
CNA = 0x0201; CORE = 0x0801; DPU = 0x1001
PC = 0x0081; PC_REG = 0x0101; VERSION = 0x0041
PPU = 0x4001; PDP = 0x8001
RK_CBUF_BANKS = 12; CBUF_BANK_SIZE = 32768
FP16_ATOM_ELEMENTS = 16; UNPACK_C2 = 8
```

---

## 5. Critical Files

| File | State |
|---|---|
| `examples/conv.py` (2389 lines) | has c256_h2_oc64 materializer + guard + CLI flag; PREFIX_BY_Y_SHAPES, PREFIX_BY_K_SHAPES, GROUPED_SERIAL_SHAPES, LOCAL_TILE_REPLAY_SHAPES, CRASH_FENCED_SHAPES sets |
| `conv_expt/rknn_prefix_replay.py` (971 lines) | manifest with 4 new entries added this turn: `c96_h20_oc12_pw_by_y`, `b1_c3_h224_spatial_by_y`, `g2_c4_h1_w1_oc2_serial`, `g2_c4_h1_w1_oc4_serial`. 98+ shape entries total. |
| `examples/conv_c576_h19_oc12_no_submit.py` (138 lines) | 12-task task structure + body field invariants for Attack H |
| `examples/conv_c256_h28_dw_byyk_no_submit.py` (143 lines) | no-submit body-field validator for Attack C |
| `examples/conv_crash_fence_no_submit.py` (209 lines) | c256_h2_oc64 crash-fence body validator |
| `examples/conv_pointwise_by_yk_layout_no_submit.py` (155 lines) | pointwise BY_YK layout (c256_h28 14-row closure) |
| `examples/conv_pointwise_by_k_layout_no_submit.py` (261 lines) | pointwise BY_K layout (c832_h7_oc48 body fields) |
| `examples/conv_depthwise_by_y_layout_no_submit.py` (283 lines) | depthwise BY_Y layout (c32_h150, c96_h112 body fields) |
| `conv_expt/capture_harness/capture_all_uncaptured.py` | walks fenced list, builds matched .rknn, runs gdb capture |
| `conv_expt/capture_harness/decode_captures.py` | decodes captured qwords |
| `conv_expt/capture_harness/decoded/` | decoded capture data |
| `conv_expt/build_progress_table.py` | per-family capture coverage from live dump dir |
| `sweep_217.py` | default --output-dir `sweep_results/` in worktree |
| `sweep_results/` (committed) | all sweep outputs; safe from reboot |
| `sweep_results/conv_py_217_current_summary.txt` | **NEW this turn** — latest snapshot |
| `sweep_results/conv_py_217_sweep_20260603_131332_summary.txt` | latest sweep |
| `current_task.md` (this file) | authoritative handoff |
| `/home/orangepi/npu/ops_rknn/dump/prefix_*_keep1_gem{1,2}/` | 160 capture dumps, 117 distinct slugs |

---

## 6. Test Runner Commands

```bash
# NPU health check (MUST pass before and after any sweep)
python3 examples/simple_add.py

# Single shape (guarded for c256_h2_oc64)
timeout 30 python3 examples/conv.py <shape> [--allow-c256-h2-oc64-exact11-submit]

# Full sweep (output to worktree, NOT /tmp)
timeout 200 python3 sweep_217.py --skip-health
# Latest: sweep_results/conv_py_217_sweep_<timestamp>_summary.txt
# Current snapshot: sweep_results/conv_py_217_current_summary.txt
# Fenced list: sweep_results/fenced_80.txt

# Re-print capture coverage table
python3 conv_expt/build_progress_table.py

# No-submit body field checks
python3 examples/conv_c576_h19_oc12_no_submit.py --check all
python3 examples/conv_c256_h28_dw_byyk_no_submit.py --check all
```

---

## 7. Safety Constraints (from AGENTS.md)

- **Do NOT kill long-running NPU processes** (crashes board).
- **BE SUPER CAREFUL** with `npu_submit`, `task_count`, `regcmd_addr`, `regcfg_amount`, `enable_mask`. Wrong submit parameters crash/reboot.
- **NPU soft-resets on CMA pressure**; `python3 examples/simple_add.py` is the recovery check.
- **Test-run any code changes**; sweep must complete with no FAIL/ERROR/TIMEOUT.
- **All overrides must be shape-conditional** (no global changes to `make_regs`).
- **The DMA_CON2 bug pattern** (using full-shape in_h in tile helpers) must NOT be reintroduced.
- **`CRASH_FENCED_SHAPES`** shapes exit before DRM allocation. Re-introducing direct submits for them requires both a guarded flag and a proven no-submit materializer first.
- **Do NOT store important files in `/tmp`** (lost on crash/reboot). Use worktree or `/home/orangepi/npu/ops_rknn/dump/`.
- **No-submit materializers are SAFE** — they don't allocate DRM or submit to NPU.

---

## 8. Session continuity notes

- This file is the authoritative handoff document. Update it after every promotion, sweep, or materializer addition.
- NPU health is green (`simple_add.py` PASS at 13:15).
- The crash-fenced `b1_c256_h2_w2_oc546_*` shape still exits before DRM allocation; do NOT run it directly.
- All captures stored at `/home/orangepi/npu/ops_rknn/dump/prefix_<slug>_keep1_gem{1,2}/` (NOT `/tmp`).
- All sweep outputs stored at `/home/orangepi/rk3588/sweep_results/` (NOT `/tmp`).
- If a future crash happens, recover with `python3 examples/simple_add.py` (must PASS) and re-sweep with `timeout 200 python3 sweep_217.py --skip-health`.

---

## 9. Distance to Goal

- 80 fenced → 0 fenced (need 80 promotions)
- Largest blocks:
  - **36 pointwise 1x1** (need parameterized BY_K materializer) — **biggest single bucket**
  - **30 depthwise 3x3** (need row materializers per BY pattern: 19 BY_YK + 11 BY_Y)
  - 7 spatial 3x3 (need BY_K writers + 1 c16_h80_oc128 4-task writer)
  - 6 k5x5/k7x7 (need 4-task writers)
  - 1 spatial 5x5 (c16_h80_oc128 5x5)
  - 1 c256_h2_oc546 (crash-fence, has captures, needs guarded submit)
- **Captures: DONE (100%)** — no longer a blocker
- **Materializers: 1 done (c256_h2_oc64 EXACT11), 79 to go** — current blocker
- **Net promotions this turn: 0** (only manifest entries and documentation)

---

## 10. Next Steps (ordered by tractability)

### 10.1 IMMEDIATE — verify state and confirm NPU is still healthy

```bash
cd /home/orangepi/rk3588
python3 examples/simple_add.py                                    # must PASS
timeout 200 python3 sweep_217.py --skip-health                     # full sweep
python3 examples/simple_add.py                                    # must PASS post
```

Expected: `total=217 counts={'PASS': 137, 'FENCED': 80, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}`. Output goes to `sweep_results/`.

### 10.2 STEP 1 — Build a parameterized pointwise-1x1 BY_K materializer (target: 5-10 promotions)

Copy `_c256_h2_oc64_exact11_task_regs` into a generic `_pointwise_by_k_exact11_task_regs(s, in_dma, wt_dma, out_dma, overrides_dict)`. Per-shape overrides live in `POINTWISE_BY_K_OVERRIDES` dict at the top of `conv.py`. Each entry has keys: `cbuf0`, `data_size1`, `dma_con2`, `weight_size0`, `weight_size1`, `weight_size2`, `conv2_low`, `dst_offsets`, `out_c_splits`, `subcores`.

**Targets (easiest first):** c128_h1_oc24, c512_h14_oc24, c832_h7_oc48, c1024_h1_oc1001, c1280_h10_oc24.

**Files to touch:** `examples/conv.py`, `conv_expt/rknn_prefix_replay.py` (manifest entries for each promoted shape).

### 10.3 STEP 2 — Build c576_h19_oc12 12-task materializer (Attack H, target: 1 promotion)

12 tasks: setup (108q), k_half (108q), 5×(k_tile 104q, ppu_pdp 26q). Like c256_h2_oc64 but with 1 k_half and 5 k_tiles.

**Files to touch:** `examples/conv.py` (add `_c576_h19_oc12_task_regs`, `run_c576_h19_oc12_shape`, CLI flag `--allow-c576-h19-oc12-submit`).

### 10.4 STEP 3 — Build parameterized depthwise BY_YK row materializer (target: 5-10 promotions)

19 fenced depthwise shapes follow the BY_YK pattern. Build a `depthwise_BY_YK_writer(in_h_list, out_h_list, y_starts, body_overrides, weight_reuse_per_row)` that emits 12-23 rows. Start with c256_h28 (body fields already extracted), then generalize.

**Files to touch:** `examples/conv.py` (add `_depthwise_by_yk_task_regs`, `run_depthwise_by_yk_shape`, `DEPTHWISE_BYYK_OVERRIDES` dict).

### 10.5 STEP 4 — Build depthwise BY_Y/y_tile single-task writer (target: 5-10 promotions)

13 fenced depthwise shapes with in_h<32 use the BY_Y/y_tile single-task pattern.

**Files to touch:** `examples/conv.py` (add `_depthwise_by_y_task_regs`, `run_depthwise_by_y_shape`, `DEPTHWISE_BYY_OVERRIDES` dict).

### 10.6 STEP 5 — Build c16_h80_oc128 k3x3/k5x5 4-task writer (Attack F, target: 2 promotions)

Custom 4-task writer: 1 preamble + 3 y-tile computes. NOT 11-task EXACT11. Body overrides: cbuf0=0x57, weight_size0=0x9000, weight_size2=0x03030080 (3x3) or 0x05050080 (5x5), conv2_low=0x1a0.

### 10.7 STEP 6 — Build k5x5/k7x7 spatial 4-task writers (target: 4-6 promotions)

4 k5x5 depthwise (c480, c576, c768, c960) + 2 k7x7 depthwise (c1024) need 4-task writers similar to Attack F but with different body fields.

### 10.8 STEP 7 — Crash-fence recovery — c256_h2_oc546 (target: 1 promotion)

The only remaining crash-fenced shape. Needs no-submit materializer (similar to `examples/conv_crash_fence_no_submit.py` but for c256_h2_oc546), then parallel materializer to c256_h2_oc64 with guarded submit.

### 10.9 STEP 8 — Final sweep + commit

```bash
python3 examples/simple_add.py
timeout 200 python3 sweep_217.py --skip-health
python3 examples/simple_add.py
git add examples/conv.py conv_expt/rknn_prefix_replay.py \
        experimental/rknn/dump_rknpu_task_gems.py \
        current_task.md sweep_results/ \
        conv_expt/capture_harness/ conv_expt/build_progress_table.py
git commit -m "Promote all 217 conv shapes via prefix-replay"
```
