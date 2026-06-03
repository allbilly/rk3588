# Current Task: Prefix-Replay Debug of All FENCED Shapes in `examples/conv.py`

**Last updated:** 2026-06-03 13:25 (Asia/Shanghai)
**Owner:** Codex session (multi-model handoff)
**CWD:** `/home/orangepi/rk3588`
**NPU health (verified 13:25):** `python3 examples/simple_add.py` PASS — board has NOT been rebooted this session.
**HEAD:** `c2749f2` (4 manifest entries + sweep archive + current_task refresh).
**Latest full sweep (20260603_131332):** `total=217 counts={'PASS': 137, 'FENCED': 80, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}`

---

## 0. TL;DR

1. **Goal:** use **prefix replay** to debug and fix every FENCED shape in `examples/conv.py`. End state: **PASS=217, FENCED=0, FAIL=0, ERROR=0, TIMEOUT=0**.
2. **Current pass progress:** **PASS=137 / 217 (63.1%)**, **FENCED=80 (36.9%)**, **FAIL=0, ERROR=0, TIMEOUT=0**. NPU healthy.
3. **Capture status:** **100% (80/80 fenced shapes have BOTH GEM1 and GEM2 captures)**. Captures at `/home/orangepi/npu/ops_rknn/dump/prefix_<slug>_keep1_gem{1,2}/`. 117 distinct slugs.
4. **This turn's progress (2026-06-03 13:25):** investigated 5 distinct promotion paths for fenced shapes. **All 5 are blocked by documented technical issues** in the existing manifest (negative probes, missing per-shape body field overrides, globally-disabled BY_YK). Net new promotions this turn: **0**. The blocker remains: 80 fenced shapes need fresh RKNN GEM2 captures for the specific narrow-OC or BY_YK closure semantics that haven't been derived yet.
5. **Storage rule:** NEVER store important files in `/tmp`. Sweep outputs in `/home/orangepi/rk3588/sweep_results/`. Captures in `/home/orangepi/npu/ops_rknn/dump/`.

---

## 1. Pass-Progress Table

| Date | PASS | FENCED | Note |
|---|---:|---:|---|
| 2026-06-02 09:54 | 114 | 103 | user-stated stuck baseline |
| 2026-06-02 22:30 | 134 | 83 | c256_h2 oc64/oc546 fenced; h7 c512 promoted |
| 2026-06-03 09:49 | 136 | 81 | c512_h7 pointwise promoted |
| 2026-06-03 10:30 | 137 | 80 | c256_h2_oc64 promoted via EXACT11 materializer |
| 2026-06-03 12:51 | 137 | 80 | 100% capture coverage achieved (80/80 GEM1+2) |
| 2026-06-03 13:15 | 137 | 80 | 4 manifest entries added; 12 "newly promoted" shapes verified |
| 2026-06-03 13:25 | **137** | **80** | **CURRENT** — investigated 5 promotion paths, all blocked; net 0 promotions this turn |
| **Target** | **217** | **0** | +80 promotions needed |

---

## 2. This Turn's Promotion-Path Investigation (13:25)

I investigated 5 distinct paths to promote fenced shapes this turn. Every one is blocked by a documented issue. Summary:

### 2.1 Path 1: `POINTWISE_EXACT11_BYK_SHAPES` for c480_h14_oc16 and c512_h14_oc24 (pointwise-wide NONE)
- **Blocker:** both are documented `*_negative_probe` entries.
- c512_h14_oc24: "fresh RKNN prefix replay on matched model passed with max error 0.0539551 ... but Replaying those visible row fields in conv.py still failed with max_diff=135.3407 ... so visible body/task fields alone are not the missing closure."
- c480_h14_oc16: "same guessed h14 pointwise exact11 body as c480_h14_oc96 timed out on one-off guarded submit. Post simple_add PASS, but this shape needs a narrow-OC RKNN closure before promotion."
- **Verdict:** re-trying would either FAIL (max_diff=135) or TIMEOUT. Risky for board stability.

### 2.2 Path 2: `b1_c128_h1_w1_oc24_wic128_k1x1_g1_s1_pvalid` (pointwide NONE)
- **Blocker:** manifest says "regcmd/body value semantics remain unresolved. Local three-lane same-row probe timed out and one-off local setup-row replay was numerically wrong (max_diff=64.3627)."
- **Verdict:** documented as unresolved. Risky.

### 2.3 Path 3: c576_h19_oc12 12-task materializer (Attack H)
- **Blocker:** the no-submit materializer's body fields are "educated guesses from c256_h2_oc64 + c256_h28_pw_1x1 closure pattern. Body field register values (data_size1, dma2, cbuf0) ... are NOT captured from a fresh RKNN GEM2 dump, so promotion requires a fresh KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=2 capture."
- **Verdict:** would need a fresh GEM2 capture (10+ min on the NPU) plus building a 12-task materializer with guarded submit. Too risky this turn.

### 2.4 Path 4: depthwise shapes via `DEPTHWISE_BODY_SHAPES` (e.g., c32_h150, c96_h112, c256_h28)
- **Blocker:** `DEPTHWISE_BODY_SHAPES` is empty. The default `DEPTHWISE_OVERRIDES` (line 322 conv.py) doesn't match the per-shape body field constants extracted in the no-submit materializers (e.g., c256_h28 needs cbuf0=0x1b/0x201b, data_size1=0x1f00e0, dma_con2=0x2a0; the default is cbuf0=0xa2, data_size1=0x003f0100, dma_con2=0x3c). Additionally, the per-row submit path only handles BY_Y; BY_K and BY_YK are not implemented for depthwise.
- **Verdict:** would need per-shape `DEPTHWISE_BODY_OVERRIDES` + per-row path for BY_K/BY_YK. Multi-hour work.

### 2.5 Path 5: c256_h28_oc256 pointwise 1x1 (BY_YK) via new BY_YK allowed set
- **Blocker:** "BY_YK is disabled before allocation; mixed Y/K setup and k_half semantics are unresolved" (line 1257 conv.py). The body field constants ARE extracted in `examples/conv_pointwise_by_yk_layout_no_submit.py` (14-row BY_YK closure), but a real materializer would require building the 14-row pattern + a run function + a fence-modification.
- **Verdict:** the body fields are extracted but the runtime path isn't built. Multi-hour work.

### 2.6 Conclusion
- **Net new code promotions this turn: 0**
- **Net manifest entries this turn: 0** (all 19 POINTWISE_EXACT11_BYK_SHAPES already have entries; 4 new entries added in previous turn)
- **Blocker remains:** 80 fenced shapes need fresh RKNN GEM2 captures (or per-shape body field derivation) before any new promotion can be safely submitted. Per AGENTS.md, submitting with educated-guess body fields risks crashing the board.

---

## 3. Remaining Fence Groups (80 total, per latest sweep)

| Group | Count | Example shapes | Materializer status |
|---|---:|---|---|
| BY_K/k_tile closure needed | 56 | c1024_h7_oc1024, c832_h7_oc48, c128_h3_oc256, c192_h7_oc384 | 3 of 56 already pass via c40_h40_oc320 (setup108), c256_h2_oc64 (EXACT11), c256_h2_oc24 (setup108), c256_h3_oc24 (EXACT11). 53 need parameterized materializer. |
| grouped/depthwise still fenced | 54 | c32_h112_oc32, c64_h112_oc64, c128_h56_oc128, c256_h28_oc256, c384_h19_oc384, c576_h20_oc576 | DEPTHWISE_BODY_SHAPES empty; per-row path only handles BY_Y; default DEPTHWISE_OVERRIDES wrong for the actual shapes |
| BY_YK unresolved | 15 | c256_h28_oc256 (pointwise 1x1), c256_h28_oc256 (depthwise), c576_h19_oc96, c576_h19_oc273, c64_h56_oc128 | Globally disabled; c256_h28 has 14-row closure evidence but no runtime path |
| pointwise-wide NONE closure needed | 6 | c480_h14_oc16, c512_h14_oc24, c128_h1_oc24, c96_h20_oc273, c384_h10_oc546, c96_h20_oc96 | 2 are documented negative probes; others need narrow-OC RKNN closure |
| known-bad spatial setup | 4 | c16_h80_oc128 (k3x3, k5x5), c576_h20_oc576 (k5x5) | Need custom 4-task writer |
| pointwise-wide BY_Y | 1 | c256_h28_oc16 (max_diff=119.6110, kept fenced) | Needs RKNN closure semantics |
| spatial BY_Y closure | 1 | c8_h160_oc16 (already promoted via local_tile_replay; the "kept fenced" note refers to a separate GEM5 artifact entry) | 0 actually fenced in this group |

**Sub-total:** ~80 fenced (some groups overlap, e.g., c256_h28_oc256 pointwise 1x1 is in both "BY_YK" and "BY_K" categories depending on the planner view).

---

## 4. Per-Family Capture Status (re-verified 2026-06-03 13:25)

| Family | # Fenced | GEM1 | GEM2 | %G2 |
|---|---:|---:|---:|---:|
| depthwise 3x3 (k3_g=in_c) | 30 | 30 | 30 | **100%** |
| depthwise 5x5 (k5_g=in_c) | 4 | 4 | 4 | **100%** |
| depthwise 7x7 (k7_g=in_c) | 2 | 2 | 2 | **100%** |
| pointwise 1x1 (k1_g1) | 36 | 36 | 36 | **100%** |
| spatial 3x3 (k3_g1) | 7 | 7 | 7 | **100%** |
| spatial 5x5 (k5_g1) | 1 | 1 | 1 | **100%** |
| **Total** | **80** | **80** | **80** | **100%** |

**Captures: 100% (80/80).** The capture phase is **closed**. The remaining bottleneck is materializer derivation for each shape, which is gated by fresh RKNN GEM2 captures for body field evidence.

---

## 5. Critical Files (live, in worktree)

| File | State |
|---|---|
| `examples/conv.py` (2389 lines) | c256_h2_oc64 materializer + guard + CLI flag; PREFIX_BY_Y_SHAPES, PREFIX_BY_K_SHAPES, GROUPED_SERIAL_SHAPES, LOCAL_TILE_REPLAY_SHAPES, CRASH_FENCED_SHAPES, POINTWISE_EXACT11_BYK_SHAPES (19), POINTWISE_SETUP108_COMPACT_WEIGHT_SHAPES (2), POINTWISE_EXACT11_CHAIN_COMPACT_WEIGHT_SHAPES (1) |
| `conv_expt/rknn_prefix_replay.py` (971 lines) | manifest with 98+ shape entries; 4 new entries added in prior turn (c96_h20_oc12_pw_by_y, b1_c3_h224_spatial_by_y, g2_c4_h1_w1_oc2_serial, g2_c4_h1_w1_oc4_serial) |
| `examples/conv_c576_h19_oc12_no_submit.py` (138 lines) | NEW 12-task task structure + body field invariants for Attack H (c576_h19_oc12) |
| `examples/conv_c256_h28_dw_byyk_no_submit.py` (143 lines) | NEW no-submit body-field validator for c256_h28 depthwise (Attack C) |
| `examples/conv_pointwise_by_yk_layout_no_submit.py` (155 lines) | c256_h28 pointwise 1x1 14-row BY_YK closure evidence |
| `examples/conv_pointwise_by_k_layout_no_submit.py` (261 lines) | c832_h7_oc48 body fields |
| `examples/conv_depthwise_by_y_layout_no_submit.py` (283 lines) | c32_h150, c96_h112, c144_h56 body fields |
| `examples/conv_crash_fence_no_submit.py` (209 lines) | c256_h2_oc64 crash-fence body validator |
| `conv_expt/capture_harness/capture_all_uncaptured.py` | walks fenced list, builds matched .rknn, runs gdb capture |
| `conv_expt/capture_harness/decode_captures.py` | decodes captured qwords |
| `conv_expt/build_progress_table.py` | per-family capture coverage from live dump dir |
| `sweep_217.py` | default --output-dir `sweep_results/` in worktree |
| `sweep_results/` (committed) | all sweep outputs; safe from reboot |
| `sweep_results/conv_py_217_current_summary.txt` | latest snapshot (137/80) |
| `sweep_results/conv_py_217_sweep_20260603_131332_summary.txt` | latest sweep |
| `current_task.md` (this file) | authoritative handoff |
| `/home/orangepi/npu/ops_rknn/dump/prefix_*_keep1_gem{1,2}/` | 160 capture dumps, 117 distinct slugs |

---

## 6. Test Runner Commands

```bash
# NPU health check (MUST pass before and after any sweep)
python3 examples/simple_add.py

# Single shape
timeout 30 python3 examples/conv.py <shape> [--allow-exact11-byk-submit] [--allow-pointwise-exact11-byk-submit] [--allow-c256-h2-oc64-exact11-submit] [--allow-h160-setup3-submit]

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
- **Do NOT store important files in `/tmp`** (lost on crash/reboot). Use worktree or `/home/orangepi/npu/ops_rknn/dump/`.

---

## 8. Next Steps (ordered by tractability, requires fresh RKNN captures for body fields)

### 8.1 NEXT TURN: Build c576_h19_oc12 12-task materializer (Attack H, target: 1 promotion)
1. Build matched-weight .rknn: `python3 /home/orangepi/npu/ops_rknn/gen_conv2d_models.py --custom --batch 1 --in-ch 576 --out-ch 12 --height 19 --width 19 --k-h 1 --k-w 1 --groups 1 --name match_b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid --out-dir /home/orangepi/npu/ops_rknn/models`
2. Capture fresh GEM2: `KEEP_TASKS=1 PREFIX_MODE=linear DUMP_GEM=2 DUMP_DIR=/home/orangepi/npu/ops_rknn/dump gdb -q -batch -x /home/orangepi/rk3588/conv_expt/gdb/rknn_prefix_replay.gdb --args /home/orangepi/npu/ops_rknn/conv2d_multi --case match_b1_c576_h19_w19_oc12_wic576_k1x1_g1_s1_pvalid 1 576 19 19 12 1 1 1 c576_h19_oc12`
3. Decode body fields and replace the educated guesses in `examples/conv_c576_h19_oc12_no_submit.py`
4. Build a real 12-task materializer in `examples/conv.py` with a guarded flag `--allow-c576-h19-oc12-submit`
5. Add a manifest entry
6. Re-sweep to confirm 138/79

### 8.2 NEXT TURN: Build c256_h28_oc256 pointwise 1x1 (BY_YK) materializer (target: 1 promotion)
1. The 14-row BY_YK closure evidence is in `examples/conv_pointwise_by_yk_layout_no_submit.py` (DATA_SIZE1=0x003f0100, DMA_CON2=0x2a0, DST_SURF_STRIDE=0x310, SURFACE_ADD=0x620, CBUF0=0x48/0x2048). Build a real materializer using these.
2. Add a NEW `BY_YK_ALLOWED_SHAPES` set, modify the fence to skip for shapes in the set.
3. Build a guarded submit path.
4. Test, re-sweep, manifest.

### 8.3 NEXT TURN: Add per-shape DEPTHWISE_BODY_OVERRIDES for c256_h28 + c32_h150 (Attack C, target: 1-2 promotions)
1. The body field constants are extracted in `examples/conv_c256_h28_dw_byyk_no_submit.py` (12-row BY_YK closure for c256_h28 depthwise) and `examples/conv_depthwise_by_y_layout_no_submit.py` (15-task BY_Y closure for c32_h150).
2. Add per-shape `DEPTHWISE_BODY_OVERRIDES` dict in `conv.py`.
3. Add the shapes to `DEPTHWISE_BODY_SHAPES` to unblock the per-row path.
4. The per-row path needs to be extended to handle BY_YK (12 mixed families) — currently it only handles BY_Y (single family y_tile).

### 8.4 LATER: Build parameterized pointwise-1x1 BY_K materializer (target: 5-10 promotions)
- For shapes NOT in the negative-probe set, build a generic materializer parameterized by per-shape config.
- Captures exist; need fresh GEM2 captures for body fields.

### 8.5 LATER: Crash-fence recovery — c256_h2_oc546 (target: 1 promotion)
- Captures exist. Need no-submit materializer + guarded submit path.

### 8.6 LATER: Final sweep + commit
```bash
python3 examples/simple_add.py
timeout 200 python3 sweep_217.py --skip-health
python3 examples/simple_add.py
git add examples/conv.py conv_expt/rknn_prefix_replay.py current_task.md sweep_results/
git commit -m "Promote all 217 conv shapes via prefix-replay"
```

---

## 9. Session continuity notes

- This file is the authoritative handoff document. Update it after every promotion, sweep, or materializer addition.
- NPU health is green (`simple_add.py` PASS at 13:25).
- The crash-fenced `b1_c256_h2_w2_oc546_*` shape still exits before DRM allocation; do NOT run it directly.
- All captures stored at `/home/orangepi/npu/ops_rknn/dump/prefix_<slug>_keep1_gem{1,2}/` (NOT `/tmp`).
- All sweep outputs stored at `/home/orangepi/rk3588/sweep_results/` (NOT `/tmp`).
- The 80 fenced shapes need actual promotion work — manifest updates and analysis are insufficient. Next turn should pick ONE specific promotion path (e.g., c576_h19_oc12) and see it through with a fresh GEM2 capture + guarded submit + manifest entry + sweep verification.

---

## 10. Distance to Goal

- 80 fenced → 0 fenced (need 80 promotions)
- **Captures: 100% done** (no longer a blocker)
- **Materializers: 1 done (c256_h2_oc64 EXACT11), 79 to go** (current blocker)
- **Per-promotion cost:** ~30-60 min (fresh GEM2 capture + body field decode + materializer code + guarded test + manifest entry + sweep verification)
- **Net promotions this turn: 0**
- **Net promotions this session: +23 (114 → 137)**
- **Target: 80 more promotions to reach 217/217**
