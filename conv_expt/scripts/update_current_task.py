#!/usr/bin/env python3
"""Apply targeted updates to current_task.md in-place."""
import re
from pathlib import Path

PATH = Path("/home/orangepi/rk3588/current_task.md")
text = PATH.read_text()

# Update 1: header block (timestamps, NPU health, HEAD, sweep, table file)
old1 = """# Current Task: Prefix-Replay Debug of All FENCED Shapes in `examples/conv.py`

**Last updated:** 2026-06-03 17:05 (Asia/Shanghai)
**Owner:** Codex session (continuing multi-model handoff; previous session was "it crashed" - see Session Continuity SS 10.2)
**CWD:** `/home/orangepi/rk3588`
**NPU health (verified 17:05):** `python3 examples/simple_add.py` returns `ret=0, handle=5` and `ADD NPU=[8 8 8 8 8 8 8 8] expected=[8 8 8 8 8 8 8 8] PASS`. Board has NOT been rebooted this session.
**HEAD:** `eb45bfb` (c128_h3_oc256_3x3 promotion, 16 commits ahead of `origin/main`).
**Working tree:** `M conv_expt/build_progress_table.py` (path updated, not in /tmp). `D shape_stratgery.md` (unrelated, was already deleted last session). Other `??` files in `examples/` and `experimental/` are pre-existing scratch and intentionally untouched.
**Latest full sweep (20260603_170242):** `total=217 counts={'PASS': 142, 'FENCED': 75, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}`. Pre/post health rc was -1; NPU verified manually at 17:05 PASS.
**Per-family table file:** `sweep_results/family_progress_table_20260603_1600.txt` (135 lines; safe location, NOT /tmp)."""

new1 = """# Current Task: Prefix-Replay Debug of All FENCED Shapes in `examples/conv.py`

**Last updated:** 2026-06-03 17:30 (Asia/Shanghai)
**Owner:** Codex session (continuing multi-model handoff; previous session was "it crashed" - see Session Continuity SS 10.2)
**CWD:** `/home/orangepi/rk3588`
**NPU health (verified 17:30):** `python3 examples/simple_add.py` returns `ret=0, handle=5` and `ADD NPU=[8 8 8 8 8 8 8 8] expected=[8 8 8 8 8 8 8 8] PASS`. Board has NOT been rebooted this session.
**HEAD:** `4d16dcc` (per-family table refresh, 22 commits ahead of `origin/main`).
**Working tree:** CLEAN. The uncommitted c1280_h10_oc24 attempt at 17:20 (re-using c128 family body fields) produced max_diff=225 and was reverted via `git checkout examples/conv.py` at 17:30. The lesson: c128 family body fields do NOT transfer to c1280 family; c1280 needs a fresh body field decode. Untracked `??` files in `examples/` and `experimental/` are pre-existing scratch and intentionally untouched.
**Latest full sweep (20260603_172056):** `total=217 counts={'PASS': 142, 'FENCED': 75, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}`. Pre/post health rc was -1; NPU verified manually at 17:30 PASS.
**Per-family table file:** `sweep_results/family_progress_table_20260603_1730.txt` (192 lines; safe location, NOT /tmp)."""

assert old1 in text, "header block not found"
text = text.replace(old1, new1)

# Update 2: TL;DR (78 -> 75; new fence reason breakdown)
old2 = """1. **Goal:** use **prefix replay** to debug and fix every FENCED shape in `examples/conv.py`. End state: **`PASS=217, FENCED=0, FAIL=0, ERROR=0, TIMEOUT=0`** in `timeout 200 python3 sweep_217.py --skip-health`, with pre/post `python3 examples/simple_add.py` both PASS. 78 fenced shapes must be promoted via prefix-replay methodology.
2. **Current pass progress:** **`PASS=142 / 217` (65.4%)**, **`FENCED=75 / 217` (34.6%)**, **`FAIL=0, ERROR=0, TIMEOUT=0`**. Net new promotions since the user-stated 114/103 stuck baseline: **+28 PASS** (114->142). Net new promotions this session (post-current_task.md-rewrite): **+3** (c256_h3_oc128_1x1, c128_h3_oc256_1x1, c128_h3_oc256_3x3). Distance to goal: **75 more promotions**.
3. **Capture coverage:** **100% (78/78 fenced shapes have BOTH GEM1 and GEM2 captures)**. Captures at `/home/orangepi/npu/ops_rknn/dump/prefix_<slug>_keep1_gem{1,2}/` (84 distinct `_keep1_gem2` directories; 117 distinct prefix slugs total). Captures are **no longer a blocker**. **YES, every fenced shape has a capture already.** Detail in SS 2 below.
4. **Biggest blocker (stated explicitly):** **per-shape body-field constants for the 78 remaining fenced shapes**. The generic 11-task EXACT11 BY_K path works for the existing 9 promoted shapes but has hard-coded body fields (CBUF0, DATA_SIZE1, DMA_CON2, CVT_CON0, CONV2_LOW, weight size, k_tile OC splits) that don't match the (ic, oc, kh, kw, in_h) tuple of every other shape. The c16_h80_oc128_3x3 promotion last turn showed that the spatial-3x3 with in_c=16 family shares body field constants (CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0) across oc values, with KT_TILE_SPLITS auto-parameterized for the OC count.
5. **In-flight work (last session):** c16_h80_oc128_3x3 PROMOTED at 15:41 (max_diff=0.0293). c16_h80_oc128_5x5 PROMOTED at 15:50 (max_diff=0.0313). Both used the same body field constants as c16_h80_oc64 (CBUF0=0x57, DATA_SIZE1=0x000F0010, CONV2_LOW=0x1a0). The 5x5 differed only in weight size (auto-scaled by EXACT11 BY_K path: weight per kernel changes from 288 to 800 for 5x5). c16_h80 family is now fully promoted (3x3 and 5x5 both done). A **batch promotion of 6 spatial 3x3 siblings FAILED** with max_diff=163-368 and was reverted via `git checkout examples/conv.py`; NPU still healthy. The c576_h19_oc12 materializer (committed in 3b520a0/3704e1c/40b6133) is still FAIL max_diff=152; left as-is."""

new2 = """1. **Goal:** use **prefix replay** to debug and fix every FENCED shape in `examples/conv.py`. End state: **`PASS=217, FENCED=0, FAIL=0, ERROR=0, TIMEOUT=0`** in `timeout 200 python3 sweep_217.py --skip-health`, with pre/post `python3 examples/simple_add.py` both PASS. 75 fenced shapes must be promoted via prefix-replay methodology.
2. **Current pass progress:** **`PASS=142 / 217` (65.4%)**, **`FENCED=75 / 217` (34.6%)**, **`FAIL=0, ERROR=0, TIMEOUT=0`**. Net new promotions since the user-stated 114/103 stuck baseline: **+28 PASS** (114->142). Net new promotions this session: **+3** (c256_h3_oc128_1x1, c128_h3_oc256_1x1, c128_h3_oc256_3x3). Distance to goal: **75 more promotions**.
3. **Capture coverage:** **100% (75/75 fenced shapes have BOTH GEM1 and GEM2 captures). YES, every fenced shape already has a capture.** Captures at `/home/orangepi/npu/ops_rknn/dump/prefix_<slug>_keep1_gem{1,2}/` (84 distinct `_keep1_gem2` directories; 117 distinct prefix slugs total). **The capture phase is COMPLETE and is no longer a blocker.** See SS 2 for the per-family capture table.
4. **Biggest blocker:** **per-shape body-field constants for the 75 remaining fenced shapes**. The 9 promoted shapes each had a fresh body field decode from their GEM2 capture. The c16_h80 family (3x3 + 5x5) showed that when in_c is the same, body field constants can transfer across oc values; this session's c128_h3 family (1x1 + 3x3) showed the same for sibling (ic, in_h, oc) tuples. Other families (c1280, c1024, c832, c480, c384, c288, c72) require fresh per-family body field decoding because body field constants are (ic, in_h, kh)-dependent, not just (ic)-dependent. See SS 2.4 for fence reason breakdown.
5. **Fence reason breakdown (75 total, classified by sweep_217.py error message):**
   - **27 BY_K/k_tile** (pending RKNN 108/104/26-row closure) — pointwise 1x1 with k_tile partitioning + spatial 3x3
   - **19 depthwise BY_YK** (needs DEPTHWISE_BODY_SHAPES membership)
   - **14 depthwise BY_K** (needs DEPTHWISE_BODY_SHAPES membership)
   - **7 BY_YK disabled at planner level** (mixed Y/K setup, k_half semantics unresolved — not tractable via current path)
   - **3 depthwise BY_Y** (needs DEPTHWISE_BODY_SHAPES membership)
   - **3 pointwise-wide NONE** (pending 108-row closure — includes c128_h1_oc24, c480_h14_oc16, c512_h14_oc24)
   - **1 pointwise-wide BY_Y** (c576_h19_oc12 — pending row closure; the in_progress materializer is at conv_expt/in_progress/c576_h19_oc12_addition.py)
   - **1 crash-fenced** (b1_c256_h2_w2_oc546 — DO NOT submit directly; causes reboot)
6. **In-flight work (this session, 17:30):** c1280_h10_oc24_s1pvalid attempt at 17:25 used c128 family body fields (CBUF0=0x0b1, DATA_SIZE1=0x04ff0500, DMA_CON2=0x0ffffffd) and produced `max_diff=225.82`; reverted. c1280 family needs a fresh per-shape body field decode from its capture, not a c128 family transplant."""

assert old2 in text, "TL;DR block not found"
text = text.replace(old2, new2)

PATH.write_text(text)
print(f"OK, current_task.md is now {len(text.splitlines())} lines")
