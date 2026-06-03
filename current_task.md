# Current Task: Prefix-Replay Debug of All FENCED Shapes in `examples/conv.py`

**Last updated:** 2026-06-03 15:55 (Asia/Shanghai)
**Owner:** Codex session (continuing multi-model handoff)
**CWD:** `/home/orangepi/rk3588`
**NPU health (verified 15:42):** `python3 examples/simple_add.py` returns `ret=0, handle=5` and `ADD NPU=[8 8 8 8 8 8 8 8] expected=[8 8 8 8 8 8 8 8] PASS`. Board has NOT been rebooted this session.
**HEAD:** `12c7a96` (c16_h80_oc128_3x3 promotion, 12 commits ahead of origin/main). Working tree: `examples/conv.py` and `current_task.md` and `conv_expt/rknn_prefix_replay.py` and `sweep_results/` modified (c16_h80_oc128_5x5 promotion in progress). `shape_stratgery.md` deleted (unrelated).
**Latest full sweep (20260603_155357):** `total=217 counts={'PASS': 139, 'FENCED': 78, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}` (2 new promotions this session: c16_h80_oc128_3x3 + c16_h80_oc128_5x5). NPU health verified pre/post via `python3 examples/simple_add.py` (both PASS).
**Storage rule (always observed):** NEVER store important files in `/tmp`. Sweep outputs go to `/home/orangepi/rk3588/sweep_results/`. Captures go to `/home/orangepi/npu/ops_rknn/dump/`. The in-progress materializer is at `/home/orangepi/rk3588/conv_expt/in_progress/c576_h19_oc12_addition.py` (NOT `/tmp`).

---

## 0. TL;DR

1. **Goal:** use **prefix replay** to debug and fix every FENCED shape in `examples/conv.py`. End state: **`PASS=217, FENCED=0, FAIL=0, ERROR=0, TIMEOUT=0`** in `timeout 200 python3 sweep_217.py --skip-health`, with pre/post `python3 examples/simple_add.py` both PASS. 79 fenced shapes must be promoted via prefix-replay methodology.
2. **Current pass progress:** **`PASS=139 / 217` (64.1%)**, **`FENCED=78 / 217` (35.9%)**, **`FAIL=0, ERROR=0, TIMEOUT=0`**. Net new promotions since the user-stated 114/103 stuck baseline: **+25 PASS** (114→139). Net new promotions this session: **+2** (c16_h80_oc128_3x3 at 15:41, c16_h80_oc128_5x5 at 15:50). Distance to goal: **78 more promotions**.
3. **Capture coverage:** **100% (78/78 fenced shapes have BOTH GEM1 and GEM2 captures)**. Captures at `/home/orangepi/npu/ops_rknn/dump/prefix_<slug>_keep1_gem{1,2}/` (84 distinct `_keep1_gem2` directories; 117 distinct prefix slugs). Captures are no longer a blocker. **YES, every fenced shape has a capture already.**
4. **Biggest blocker (stated explicitly):** **per-shape body-field constants for the 79 remaining fenced shapes**. The generic 11-task EXACT11 BY_K path works for the existing 9 promoted shapes but has hard-coded body fields (CBUF0, DATA_SIZE1, DMA_CON2, CVT_CON0, CONV2_LOW, weight size, k_tile OC splits) that don't match the (ic, oc, kh, kw, in_h) tuple of every other shape. The c16_h80_oc128_3x3 promotion this turn showed that the spatial-3x3 with in_c=16 family shares body field constants (CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0) across oc values, with KT_TILE_SPLITS auto-parameterized for the OC count. This same template can likely be reused for the other 5 spatial-3x3 siblings.
5. **In-flight work (this turn):** c16_h80_oc128_3x3 PROMOTED at 15:41 (max_diff=0.0293). c16_h80_oc128_5x5 PROMOTED at 15:50 (max_diff=0.0313). Both used the same body field constants as c16_h80_oc64 (CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0). The 5x5 differed only in weight size (auto-scaled by EXACT11 BY_K path: weight per kernel changes from 288 to 800 for 5x5). Full sweep 139/78. The c576_h19_oc12 materializer (committed in 3b520a0/3704e1c/40b6133) is still FAIL max_diff=152; left as-is. c16_h80 family is now fully promoted (3x3 and 5x5 both done).

---

## 1. Pass-Progress Table (chronological)

| Date         | PASS | FENCED | Note |
|---|---:|---:|---|
| 2026-06-02 09:54 | 114 | 103 | user-stated stuck baseline |
| 2026-06-02 22:30 | 134 |  83 | c256_h2 oc64/oc546 fenced; h7 c512 promoted |
| 2026-06-03 09:49 | 136 |  81 | c512_h7 pointwise promoted |
| 2026-06-03 10:30 | 137 |  80 | c256_h2_oc64 promoted via EXACT11 materializer |
| 2026-06-03 12:51 | 137 |  80 | 100% capture coverage achieved (80/80 GEM1+2) |
| 2026-06-03 13:15 | 137 |  80 | 4 manifest entries added; 12 "newly promoted" shapes verified |
| 2026-06-03 13:25 | 137 |  80 | investigated 5 promotion paths, all blocked; net 0 |
| 2026-06-03 13:32 | 137 |  80 | latest sweep, FENCED list frozen at 80 |
| 2026-06-03 14:10 | 137 |  80 | honest analysis, 5 paths blocked, c576_h19_oc12 drafted |
| 2026-06-03 14:22 | 137 |  80 | sweep 142207 confirms FENCED=80 (c16_h80_oc128 still FENCED, spatial setup/NONE) |
| 2026-06-03 15:17 | 137 |  80 | c576_h19_oc12 committed (3b520a0/3704e1c/40b6133) but still FAIL max_diff=152 |
| 2026-06-03 15:30 | 137 |  80 | c16_h80_oc128_3x3 added to PREFIX_BY_K_SHAPES (uncommitted); direct run FAIL max_diff=inf |
| 2026-06-03 15:41 | 138 |  79 | c16_h80_oc128_3x3 promoted: per-shape body field overrides (CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0) added; PASS max_diff=0.0293 |
| 2026-06-03 15:42 | 138 |  79 | sweep 154116 confirms no regressions; c16_h80_oc128_5x5 still FENCED |
| 2026-06-03 15:50 | 139 |  78 | c16_h80_oc128_5x5 promoted: same body field constants as 3x3 sibling (CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0); PASS max_diff=0.0313 |
| 2026-06-03 15:55 | **139** | **78** | **CURRENT** — sweep 155357 confirms; c16_h80 family now fully promoted (3x3 + 5x5); 4 spatial-3x3 siblings remain (c128_h3_oc256, c128_h5_oc256, c192_h7_oc384, c256_h10_oc512, c40_h40_oc160, c72_h20_oc288) |
| **Target**      | **217** |   **0** | +79 promotions needed |

---

## 2. Per-Family Progress + Capture Status

**Source:** `python3 conv_expt/build_progress_table.py` (re-run 15:42 against latest sweep `conv_py_217_sweep_20260603_154116_summary.txt`). Full output at `sweep_results/family_progress_table.txt` (139 lines).

| Family                    | # Fenced | GEM1 | GEM2 | %   | Promotion status (per family) |
|---|---:|---:|---:|---:|---|
| depthwise 3x3 (k3_g=in_c) | 30 | 30 | 30 | 100% | 0 promoted; per-shape body fields differ (DEPTHWISE_BODY_SHAPES empty since 2813f92). Per-row path handles BY_Y (c32_h150); BY_K and BY_YK need new code paths. |
| depthwise 5x5 (k5_g=in_c) |  4 |  4 |  4 | 100% | 0 promoted. Same blockers as 3x3 plus kernel size 5x5. |
| depthwise 7x7 (k7_g=in_c) |  2 |  2 |  2 | 100% | 0 promoted. Captures at `c1024_h7_oc1024` shared with depthwise 3x3. |
| pointwise 1x1 (k1_g1)     | 36 | 36 | 36 | 100% | 1 promoted (c256_h2_oc64 via EXACT11). 30 unblocked narrow-OC shapes; 5 in crash-fence (c256_h2_oc546 still FENCED). c576_h19_oc12 attack (3 commits) still FAIL max_diff=152. |
| spatial 3x3 (k3_g1)       |  6 |  6 |  6 | 100% | 2 promoted (c16_h80_oc64 + c16_h80_oc128 via EXACT11 BY_K). 4 others (c128_h3_oc256, c128_h5_oc256, c192_h7_oc384, c256_h10_oc512, c40_h40_oc160, c72_h20_oc288) need per-shape body field derivation. |
| spatial 5x5 (k5_g1)       |  0 |  0 |  0 | 100% | 1 promoted (c16_h80_oc128_5x5 via EXACT11 BY_K, same body fields as 3x3 sibling). c16_h80 family now fully promoted. |
| **Total**                 | **78** | **78** | **78** | **100%** | 4 promoted. 74 to go. |

**Are all shapes captured already?** **YES — 100% capture coverage achieved**. Every one of the 79 fenced slugs has BOTH a `prefix_<slug>_keep1_gem1/` and a `prefix_<slug>_keep1_gem2/` directory under `/home/orangepi/npu/ops_rknn/dump/`. Captures are no longer the blocker; the blocker is **deriving body-field register values from those captures and writing per-shape materializers**.

---

## 3. Goal (End State)

PASS=217, FENCED=0, FAIL=0, ERROR=0, TIMEOUT=0 in `timeout 200 python3 sweep_217.py --skip-health`, with pre/post `python3 examples/simple_add.py` both PASS. 79 fenced shapes must be promoted via prefix-replay methodology (capture RKNN prefix dump → decode body fields → build per-shape materializer in `conv.py` → guarded submit → manifest entry → sweep verification).

**Promotion criteria for each shape (per AGENTS.md):**
- Live NPU submit with `python3 examples/conv.py <shape> [--allow-...-submit]` PASSes.
- `max_diff` ≤ 0.12 (fp16 output) or 0.10 (fp32 output), with `np.allclose` returning True.
- Pre-sweep `simple_add.py` PASS and post-sweep `simple_add.py` PASS (board did not hang/reboot).
- A manifest entry in `conv_expt/rknn_prefix_replay.py` documenting the dump path, the body field constants, and the body/task fields that were the missing closure.
- The shape is removed from `fenced_80.txt` (it now shows up as PASS in the next sweep).

**Methodology constraints:**
- **All overrides must be shape-conditional** (no global changes to `make_regs`).
- **Test-run any code changes**; sweep must complete with no FAIL/ERROR/TIMEOUT.
- **Do NOT kill long-running NPU processes** (crashes board).
- **BE SUPER CAREFUL** with `npu_submit`, `task_count`, `regcmd_addr`, `regcfg_amount`, `enable_mask`.
- **NPU soft-resets on CMA pressure**; `python3 examples/simple_add.py` is the recovery check.

---

## 4. Task on Hand (THIS TURN, 15:42)

**Primary task:** promote one or more fenced shapes per turn via the prefix-replay methodology.

**Steps completed (this turn):**
1. Verified NPU health with `python3 examples/simple_add.py` (PASS at 15:42).
2. Read latest sweep (`sweep_results/conv_py_217_sweep_20260603_142207_summary.txt`) — confirmed `total=217, PASS=137, FENCED=80, FAIL=0, ERROR=0, TIMEOUT=0`.
3. Re-ran `python3 conv_expt/build_progress_table.py` — confirmed 100% capture coverage (80/80 GEM1+2).
4. Read `git log` and `git diff` — confirmed HEAD=`40b6133`, uncommitted edit adds `c16_h80_oc128_3x3` to `PREFIX_BY_K_SHAPES` on line 63.
5. Tested the c16_h80_oc128_3x3 uncommitted edit directly: `FAIL max_diff=inf` (the "crash" the user reported). The BY_K path takes the shape but the body field constants (CBUF0=0x57 for pointwise 1x1, DATA_SIZE1=0x000F0010 for oc=64) are wrong for spatial 3x3 with oc=128.
6. Decoded the c16_h80_oc128 GEM2 capture at `/home/orangepi/npu/ops_rknn/dump/prefix_c16_h80_oc128_s1pvalid_keep1_gem2/dump_gem2.txt`. Confirmed the body field constants for the spatial-3x3 in_c=16 family: CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0 (= FEATURE_GRAINS(26) << 4), weight bytes per kernel=288. All identical to c16_h80_oc64.
7. **PROMOTED c16_h80_oc128_3x3** at 15:41 by adding it to PREFIX_BY_K_SHAPES (line 63), CBUF0_OVERRIDES (line 288 = 0x057), DATA_SIZE1_OVERRIDES (line 304 = 0x000F0010), CONV2_LOW_OVERRIDES (line 395 = 0x1a0). KT_TILE_SPLITS already had the parameterization ((0, 48), (48, 48), (96, 32)) summing to 128. Direct run: PASS max_diff=0.0293.
8. Ran full sweep at 15:41: `total=217 counts={'PASS': 138, 'FENCED': 79, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}`. No regressions. c16_h80_oc128_3x3 removed from FENCED list. NPU health PASS pre/post.
9. Added manifest entry `c16_h80_oc128_promoted_via_exact11_byk` to `conv_expt/rknn_prefix_replay.py`. Updated sibling `c16_h80_oc64_promoted_via_exact11_byk` note to reflect the new promotion.
10. Wrote this `current_task.md` with the updated state.

**Steps remaining in this turn (in order):**
1. **Commit the c16_h80_oc128_3x3 promotion** with the 4-line edit + manifest update. Commit message: "Promote c16_h80_oc128_3x3 via EXACT11 BY_K with body field overrides (CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0); max_diff=0.0293 PASS, sweep 138/79."
2. **Try c16_h80_oc128_5x5** with the same approach (different body field constants expected since weight per kernel changes from 288 to 800 for 5x5). ~30 min for fresh capture + decode + test.
3. **Move to the next promotion path** based on remaining time and board health.

---

## 5. The "crash" the user reported (15:30) — RESOLVED

The user added `b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid` to `PREFIX_BY_K_SHAPES` in `examples/conv.py` (uncommitted edit, line 63). They then ran `python3 examples/conv.py b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid` and got:

```
exact11_byk_submit tasks=11 submit_task_number=3 amounts=108;104;26;104;26;104;26;104;26;104;26 masks=0xd;0xd;0x60;0xd;0x60;0xd;0x60;0xd;0x60;0xd;0x60 subcores=(0,1),(0,1),(0,1),(0,0),(0,0)
shape=b1_c16_h80_w80_oc128_wic16_k3x3_g1_s1_pvalid guarded=exact11_byk tasks=11 submit_tasks=3 FAIL max_diff=inf
debug_exact11_byk_oc=0:inf;32:inf;64:inf;96:inf
debug_exact11_byk_abs got=inf expected=77.8900
AssertionError: output mismatch max_diff=inf
```

**Root cause analysis:**
- The PREFIX_BY_K_SHAPES dispatch path (run_exact11_byk_shape) takes the shape and calls _exact11_task_regs with the BY_K layout.
- The 11-task layout uses 3 k-tile splits (KT_TILE_SPLITS for c16_h80_oc128 = `((0, 48), (48, 48), (96, 32))` = 48+48+32=128 OC) plus 5 aux rows + 1 setup + 1 k_half = 11 rows.
- The body field overrides for c16_h80_oc64 (CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0) were NOT being applied to c16_h80_oc128 because the OVERRIDES dicts had no per-shape entry for c16_h80_oc128.
- The default CBUF0=0x0a2 (used for c512_h7_oc1024, c160_h7_oc320 — pointwise family) was being applied to spatial 3x3, causing the NPU to misconfigure the CBUF and produce inf output.

**Fix applied (15:41):** added c16_h80_oc128 to 3 OVERRIDES dicts (CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0) and the PREFIX_BY_K_SHAPES set. All other spatial-3x3 in_c=16 shapes should be able to use the same constants (with KT_TILE_SPLITS parameterized for their specific OC count).

**Resolution:** c16_h80_oc128_3x3 now PASSes with max_diff=0.0293 (same as c16_h80_oc64). Full sweep 138/79, no regressions.

---

## 6. Risk Analysis for the Goal

**Per-promotion cost (estimated):**
- Fresh GEM2 capture (KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=2): ~5-10 min/shape (NPU time, blocking).
- Body field decode from the GEM2 dump: ~10-20 min/shape.
- Materializer code in conv.py: ~50-200 lines/shape, ~20-60 min.
- Guarded test + sweep verification: ~5-10 min/shape.
- **Total: ~1-2 hours per promotion.**

**Total ETA to 217/217:**
- 30 narrow-OC pointwise 1x1: ~6-8 hours.
- 30 depthwise 3x3: ~6-8 hours.
- 8 spatial 3x3 + 5x5: ~1-2 hours (one promoted, 6-7 to go).
- 12 mixed (depthwise 5x5/7x7, c256_h2_oc546 crash-fence, c256_h28 pointwise BY_YK, c576_h19_oc12 chained/aux, etc.): ~3-4 hours.
- **Total: ~16-22 hours of focused work for 79 promotions.**

**Biggest blockers (ranked by impact):**
1. **Body field parameterization for spatial 3x3/5x5** (6-7 shapes). The existing _exact11_task_regs is hard-coded for 1x1 pointwise. The c16_h80_oc128_3x3 promotion shows that for the (in_c=16, in_h=80, kh=kw=3) family, the body fields are identical to c16_h80_oc64 except KT_TILE_SPLITS. Reusable for c128_h3_oc256, c128_h5_oc256, c192_h7_oc384, c256_h10_oc512, c40_h40_oc160, c72_h20_oc288 (each needs fresh verification of body field constants).
2. **Body field parameterization for depthwise 3x3** (30 shapes). Per-row BY_Y path is the only thing working; BY_K and BY_YK need new code paths with per-shape body fields.
3. **c576_h19_oc12 chained/aux pipeline** (1 shape, but it's the most-studied). The 12-task structure is real but the 5 k_tile rows overlap in (oc, y), so they're not independent output producers. Needs ~2-3 more hours of analysis.
4. **c256_h2_oc546 crash-fence** (1 shape). Captures exist. The shape still exits before DRM allocation. Reintroducing direct submits requires both a guarded flag and proven no-submit materializer.
5. **C256_h28_oc256 pointwise BY_YK** (1 shape). 14-row BY_YK closure evidence is in `examples/conv_pointwise_by_yk_layout_no_submit.py`. Needs a real materializer.
6. **c16_h80_oc128_5x5 spatial 5x5** (1 shape, the remaining c16_h80 family shape). Needs k5x5 body field overrides (weight per kernel = 25*16*2=800 instead of 288). CBUF0 and CONV2_LOW need fresh decode from the existing c16_h80_oc128 GEM2 capture (it covers both 3x3 and 5x5 layouts).

---

## 7. Next Steps (ordered by tractability)

### 7.1 DONE: c16_h80_oc128_3x3 PROMOTED at 15:41
- Added to PREFIX_BY_K_SHAPES (line 63)
- Added to CBUF0_OVERRIDES = 0x057 (line 288)
- Added to DATA_SIZE1_OVERRIDES = 0x000F0010 (line 304)
- Added to CONV2_LOW_OVERRIDES = 0x1a0 (line 395)
- KT_TILE_SPLITS already had the parameterization ((0, 48), (48, 48), (96, 32)) summing to 128
- Manifest entry added to conv_expt/rknn_prefix_replay.py
- Direct run: PASS max_diff=0.0293
- Full sweep 20260603_154116: PASS=138, FENCED=79, FAIL=0, ERROR=0, TIMEOUT=0

### 7.2 NEXT TURN: Commit the c16_h80_oc128_3x3 promotion
```bash
cd /home/orangepi/rk3588
git add examples/conv.py conv_expt/rknn_prefix_replay.py current_task.md sweep_results/
git commit -m "Promote c16_h80_oc128_3x3 via EXACT11 BY_K with body field overrides (CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0); max_diff=0.0293 PASS, sweep 138/79"
```

### 7.3 NEXT TURN: Try c16_h80_oc128_5x5 with the same approach (Attack J, target: 1 promotion)
- Add c16_h80_oc128_5x5 to PREFIX_BY_K_SHAPES + body field overrides.
- Body field constants will differ from 3x3 sibling (weight per kernel changes from 288 to 800 for 5x5). Need fresh decode of the c16_h80_oc128 GEM2 capture or fresh capture.
- Reuse KT_TILE_SPLITS ((0, 48), (48, 48), (96, 32)) if the OC split logic is the same.

### 7.4 NEXT TURN: Parameterized spatial-3x3 BY_K materializer for the other 4 spatial 3x3 siblings (Attack I-bulk, target: 1-4 promotions)
- Reusable template from c16_h80_oc128_3x3 promotion: CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0 (verified against c16_h80_oc128 GEM2 dump).
- Candidates: c128_h3_oc256, c128_h5_oc256, c192_h7_oc384, c256_h10_oc512, c40_h40_oc160, c72_h20_oc288. They differ in (in_c, in_h, oc) so need re-verification of body field constants via fresh GEM2 capture for each.

### 7.5 LATER: c256_h2_oc546 crash-fence recovery (target: 1 promotion)
- Captures exist. Need no-submit materializer + guarded submit path.
- The crash-fence exits before DRM allocation; reintroducing direct submits requires both a guarded flag and proven no-submit materializer.

### 7.6 LATER: c256_h28_oc256 pointwise 1x1 (BY_YK) materializer (Attack F, target: 1 promotion)
- The 14-row BY_YK closure evidence is in `examples/conv_pointwise_by_yk_layout_no_submit.py` (DATA_SIZE1=0x003f0100, DMA_CON2=0x2a0, DST_SURF_STRIDE=0x310, SURFACE_ADD=0x620, CBUF0=0x48/0x2048).
- Build a real materializer using these constants. Add a NEW `BY_YK_ALLOWED_SHAPES` set, modify the fence to skip for shapes in the set. Build a guarded submit path.

### 7.7 LATER: Build parameterized pointwise-1x1 BY_K materializer (Attack B, target: 5-10 promotions)
- The pattern from c160_h14_oc320 / c256_h2_oc64 / c256_h3_oc24 / c40_h40_oc320 / c40_h40_oc24 / c64_h1_oc128 is: 11-task exact-11 BY_K body, CBUF0=0x57 (pointwise family), DATA_SIZE1 = `((ic//4 - 1) << 16) | (oc // 4)` (HYPOTHESIS, needs validation per (ic, oc) class), DMA_CON2=0x8c (or 0x0ffffffc for c256_h2_oc64), CVT_CON0=0xb.
- Choose 5-10 narrow-OC pointwise shapes NOT in the negative-probe set and NOT in CRASH_FENCED_SHAPES. Build matched-weight .rknn for each, capture fresh GEM2, decode body fields, build per-shape materializer in `conv.py`, test, sweep.

### 7.8 LATER: Add per-shape DEPTHWISE_BODY_OVERRIDES for c256_h28 + c32_h150 (Attack C, target: 1-2 promotions)
- The body field constants are extracted in `examples/conv_c256_h28_dw_byyk_no_submit.py` (12-row BY_YK closure for c256_h28 depthwise) and `examples/conv_depthwise_by_y_layout_no_submit.py` (15-task BY_Y closure for c32_h150).
- Add per-shape `DEPTHWISE_BODY_OVERRIDES` dict in `conv.py`. Add the shapes to `DEPTHWISE_BODY_SHAPES` to unblock the per-row path.
- The per-row path needs to be extended to handle BY_YK (12 mixed families) — currently it only handles BY_Y (single family y_tile).

### 7.9 LATER: Final sweep + commit
```bash
python3 examples/simple_add.py
timeout 200 python3 sweep_217.py --skip-health
python3 examples/simple_add.py
git add examples/conv.py conv_expt/rknn_prefix_replay.py current_task.md sweep_results/
```

---

## 8. Key Code Templates (copy-paste-ready)

### 8.1 c256_h2_oc64 EXACT11 materializer (the template for narrow-OC pointwise)
```python
C256_H2_OC64_EXACT11_SHAPE = "b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid"
C256_H2_OC64_EXACT11_OUT_C = (64, 32, None, 32, None, 32, None, 16, None, 16, None)
C256_H2_OC64_EXACT11_WEIGHT_SIZE0 = (0x8000, 0x4000, None, 0x4000, None, 0x4000, None, 0x2000, None, 0x2000, None)
C256_H2_OC64_EXACT11_DST_OFFSETS = (0x0, 0x0, None, 0x100, None, 0x0, None, 0x100, None, 0x180, None)
# Amounts/masks same as c256_h3_oc24: 108,104,26,104,26,104,26,104,26,104,26
# CBUF0=0xb1, DATA_SIZE1=0x003f0100, DMA_CON2=0x0ffffffc, conv2_low=0x30
# Weight layout: _pack_pointwise_wide (0x8000 bytes for 64 OC)
# K splits 0:32/32:16/48:16
# Local examples/conv.py exact11_byk submit PASS max_diff=0.0245 with pre/post simple_add PASS
```

### 8.2 c16_h80_oc64 + c16_h80_oc128 EXACT11 materializer (the template for spatial 3x3 narrow-OC, in_c=16)
```python
# Both shapes promoted via EXACT11 BY_K closure with per-shape overrides:
#   CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0
# c16_h80_oc64: NOT in KT_TILE_SPLITS, uses default ((0, 112), (112, 112), (224, 96)) but NPU masks
# c16_h80_oc128: KT_TILE_SPLITS = ((0, 48), (48, 48), (96, 32)) summing to 128
# WEIGHT_SIZE0 auto-scales: c16_h80_oc64 setup=0x4800, c16_h80_oc128 setup=0x9000
# WEIGHT_SIZE1 = 0x120 (288 bytes per kernel for in_c=16, kh=kw=3)
# WEIGHT_SIZE2 = (3 << 24) | (3 << 16) | kernels (64 for c16_h80_oc64, 128 for c16_h80_oc128)
# Both pass with max_diff=0.0293, pre/post simple_add PASS
# Reusable for other spatial-3x3 in_c=16 shapes with the same body field constants
```

### 8.3 c576_h19_oc12 EXACT12 materializer (committed, still FAIL)
```python
C576_H19_OC12_EXACT12_SHAPE = "b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid"
# GEM2 dump body field constants (from rknn_prefix_replay.py note):
# CBUF0=0x1b (NOT 0xb1), DATA_SIZE1=0x3f0240, DMA_CON2=0x011d, CVT_CON0=0x000b
# CONV2 family bits: setup=0x80, k_half=0x40, k_tile=0x60
# WEIGHT_SIZE0=0x3600 (12*576*2=13824), WEIGHT_SIZE1=0x480, WEIGHT_SIZE2=0x1010100c
# 12-task structure: 1 setup(108q) + 1 k_half(108q, 4q prelude) + 5 k_tile(104q) + 5 aux(26q)
# k_splits 4+3+2+1+2=12
# max_diff=152.1078 with simple y_start=0 for all rows
# Per-row y_offset patches needed (stride 19*8*2=304 bytes/row):
#   0x1300 forward (y_start=16), 0x850 forward (y_start=7), 0xf70 forward (y_start=13)
#   -0x420 backward is not an integer multiple of 304 (needs further analysis)
# Existing in conv_expt/in_progress/c576_h19_oc12_addition.py
# Committed in 3b520a0, 3704e1c, 40b6133 -- leave as-is, document as known FAIL
```

### 8.4 The five CRASH_FENCED_SHAPES (line 187 of conv.py)
```python
CRASH_FENCED_SHAPES = {
    "b1_c256_h2_w2_oc546_wic256_k1x1_g1_s1_pvalid",  # crash-fenced, captured but no materializer
    "b1_c256_h2_w2_oc64_wic256_k1x1_g1_s1_pvalid",   # WAS crash-fenced, EXACT11 materializer UN-FENCED at db70141
    ...
}
# (full set of 5 is in conv.py; the only remaining crash-fence is c256_h2_oc546)
```

---

## 9. Safety Constraints (from AGENTS.md)

- **Do NOT kill long-running NPU processes** (crashes board).
- **BE SUPER CAREFUL** with `npu_submit`, `task_count`, `regcmd_addr`, `regcfg_amount`, `enable_mask`. Wrong submit parameters crash/reboot.
- **NPU soft-resets on CMA pressure**; `python3 examples/simple_add.py` is the recovery check.
- **Test-run any code changes**; sweep must complete with no FAIL/ERROR/TIMEOUT.
- **All overrides must be shape-conditional** (no global changes to `make_regs`).
- **Do NOT store important files in `/tmp`** (lost on crash/reboot). Use worktree or `/home/orangepi/npu/ops_rknn/dump/` or `/home/orangepi/rk3588/conv_expt/in_progress/`.

---

## 10. Session Continuity Notes

- This file is the authoritative handoff document. Update it after every promotion, sweep, or materializer addition.
- NPU health is green (`simple_add.py` PASS at 15:42, pre and post the c16_h80_oc128 promotion).
- The crash-fenced `b1_c256_h2_w2_oc546_*` shape still exits before DRM allocation; do NOT run it directly.
- All captures stored at `/home/orangepi/npu/ops_rknn/dump/prefix_<slug>_keep1_gem{1,2}/` (NOT `/tmp`).
- All sweep outputs stored at `/home/orangepi/rk3588/sweep_results/` (NOT `/tmp`).
- The 79 fenced shapes need actual promotion work — manifest updates and analysis are insufficient. Next turn should pick ONE specific promotion path (e.g., c16_h80_oc128_5x5 with the same approach) and see it through with a fresh GEM2 capture + guarded submit + manifest entry + sweep verification.
- The c16_h80_oc128_3x3 promotion is uncommitted in `examples/conv.py` lines 63, 288, 304, 395 and `conv_expt/rknn_prefix_replay.py` (new manifest entry). Body field constants were identical to c16_h80_oc64, so the fix was 4 line additions to the OVERRIDES dicts.
- The c576_h19_oc12 materializer is committed (3 commits) but still FAILs; leave as-is, document as known FAIL.
- The in-progress c576_h19_oc12 backup materializer is at `conv_expt/in_progress/c576_h19_oc12_addition.py` (NOT `/tmp`).

---

## 11. Distance to Goal

- 78 fenced → 0 fenced (need 78 promotions)
- **Captures: 100% done** (no longer a blocker)
- **Materializers: 4 done (c256_h2_oc64 EXACT11, c16_h80_oc64 EXACT11, c16_h80_oc128_3x3 EXACT11, c16_h80_oc128_5x5 EXACT11), 74 to go** (current blocker)
- **Per-promotion cost:** ~1-2 hours (fresh GEM2 capture + body field decode + materializer code + guarded test + manifest entry + sweep verification). **c16_h80_oc128_3x3 took ~10 min** because the body field constants were identical to c16_h80_oc64.
- **Net promotions this turn: +2** (c16_h80_oc128_3x3 + c16_h80_oc128_5x5)
- **Net promotions this session: +2 (137→139)**
- **Net promotions since 114 baseline: +25**
- **Target: 78 more promotions to reach 217/217**
- **ETA: 15-21 hours of focused work** (c16_h80 family took ~20 min total)
