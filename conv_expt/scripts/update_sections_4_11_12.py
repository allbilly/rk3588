#!/usr/bin/env python3
"""Update sections 4 (Task on Hand), 11 (Distance), 12 (Spatial 3x3) with the 17:30 state."""
from pathlib import Path

PATH = Path("/home/orangepi/rk3588/current_task.md")
text = PATH.read_text()

# Update 1: Section 4 Task on Hand (replace existing 4.x content)
old4 = """## 4. Task on Hand (this session, ordered by tractability)

The user reported `it crashed` at the top of this session, which contextually refers to last session's reverted batch promotion of 6 spatial 3x3 siblings (max_diff=163-368, then `git checkout examples/conv.py` to revert). NPU is now healthy (16:00 simple_add PASS), the worktree is clean of that attempt, and we need to do promotion work ONE SHAPE AT A TIME from here on, with the per-family table as the scoreboard.

### 4.1 Highest priority: c40_h40_oc160_3x3 alone

This is the smallest of the 6 remaining spatial 3x3 siblings. Body field constants already extracted from `/home/orangepi/npu/ops_rknn/dump/prefix_c40_h40_oc160_s1pvalid_keep1_gem2/dump_gem2.txt`:

- CBUF0 = 0x87
- DATA_SIZE1 = 0x00270028
- CONV2_LOW = 0x160
- KERNELS pattern from capture: 3 k_tiles x 160 (full OC per k_tile)

**Approach:** add 4-line edit to `examples/conv.py` (CBUF0, DATA_SIZE1, CONV2_LOW, KT_TILE_SPLITS) with `KT_TILE_SPLITS = ((0, 160), (0, 160), (0, 160))` for full-OC k_tiles. Run guarded. If PASS, commit and proceed to next sibling.

**Risk:** last attempt's 6-shape batch FAILED with the same body field constants, attributed to wrong k_tile OC partitioning. The full-OC approach is a new hypothesis that needs validation on this one shape first.

### 4.2 c16_h80 family - DONE last session

c16_h80_oc128_3x3 (commit 12c7a96, max_diff=0.0293) and c16_h80_oc128_5x5 (commit 8a50477, max_diff=0.0313) are PROMOTED. No further work on this family.

### 4.3 c576_h19_oc12 - committed but FAIL, leave as-is

3 commits (3b520a0, 3704e1c, 40b6133) document a guarded submit attempt; max_diff=152.1078, not promoted. Known FAIL; do not retry without a fundamentally new approach (likely needs the 108/104/26-row RKNN closure derivation, not just body field alignment).

### 4.4 Remaining spatial 3x3 (5 siblings, after c40_h40 is unblocked)

If c40_h40_oc160_3x3 promotion succeeds, generalize to:
- c72_h20_oc288_3x3 (CBUF0=0xa7, DATA_SIZE1=0x00070048, CONV2_LOW=0x140, 3 k_tiles x 96)
- c128_h3_oc128_3x3 (CBUF0=0xb7, DATA_SIZE1=0x003f0080, CONV2_LOW=0x060, 4 k_tiles summing to 384)
- c128_h5_oc256_3x3 (CBUF0=0xb7, DATA_SIZE1=0x003f0080, CONV2_LOW=0x080, 4 k_tiles)
- c192_h7_oc384_3x3 (CBUF0=0xb7, DATA_SIZE1=0x003f00c0, CONV2_LOW=0x0a0, 3 k_tiles x 128)
- c256_h10_oc512_3x3 (CBUF0=0xa7, DATA_SIZE1=0x003f0100, CONV2_LOW=0x0d0, 3 k_tiles 176+176+160)

### 4.5 Pointwise 1x1 next (36 shapes)

Two sub-families have shared body constants that suggest batch promotion is feasible:

**Sub-family A: c40_h40 narrow-OC (DATA_SIZE1=0x00270028)**
- c40_h40_oc160_3x3 (spatial, shares body - not pointwise)
- c512_h14_oc24, c384_h19_oc64, c384_h19_oc96, c1280_h10_oc24, c1280_h10_oc546, c480_h10_oc120, c768_h10_oc120, c960_h10_oc120 (pointwise 1x1 with this DATA_SIZE1)

**Sub-family B: large-ic pointwise (ic >= 1024)**
- c64_h1_oc128 already promoted (CBUF0=0xb1, DATA_SIZE1=0x003f0040 pattern)
- c1024_h1_oc1001, c1024_h7_oc1024, c1280_h10_oc24, c1280_h10_oc546 (need body field decode from their captures; many have DATA_SIZE1 patterns already extracted in the per-family table)

### 4.6 Depthwise families (36 shapes total: 30 + 4 + 2)

Per-row BY_Y is the only working depthwise path (c32_h150 was promoted this way). New BY_K/BY_YK closures would need per-shape DEPTHWISE_BODY_SHAPES membership. Lower priority than spatial 3x3 because c128_h3_oc128 is known to time out on BY_K (6s NPU submit timeout) and depthwise is a separate code path."""

new4 = """## 4. Task on Hand (this session, ordered by tractability)

NPU is healthy (17:30 simple_add PASS), working tree is CLEAN, 142/75 confirmed by sweep_172056. Distance to goal: 75 more promotions. **ALL 75 fenced shapes already have capture** (per SS 2.3); capture is no longer work to be done.

The task is per-shape body field derivation for 75 fenced shapes. The methodology is: read the GEM2 capture, decode the body field EMITs, add 4-7 line edits to OVERRIDES dicts in `examples/conv.py`, run `timeout 30 python3 examples/conv.py <shape>`, add manifest entry to `conv_expt/rknn_prefix_replay.py`, then run sweep and commit.

### 4.1 Highest priority: spatial 3x3 (5 remaining)

The 5 remaining spatial 3x3 siblings are all BY_K/k_tile-fenced. Body field constants are known from GEM2 captures. The c128_h3_oc256_3x3 promotion this session (max_diff=0.0310) confirms the EXACT11 BY_K path works for spatial 3x3 with the right body fields.

**Order of attempts (smallest to largest):**
1. **c128_h5_oc256_3x3** (CBUF0=0xb7, DATA_SIZE1=0x003f0080, CONV2_LOW=0x080) - 4 k_tiles, sibling of c128 family
2. **c40_h40_oc160_3x3** (CBUF0=0x87, DATA_SIZE1=0x00270028, CONV2_LOW=0x160) - 3 k_tiles x 160 (full OC hypothesis)
3. **c72_h20_oc288_3x3** (CBUF0=0xa7, DATA_SIZE1=0x00070048, CONV2_LOW=0x140) - 3 k_tiles x 96
4. **c192_h7_oc384_3x3** (CBUF0=0xb7, DATA_SIZE1=0x003f00c0, CONV2_LOW=0x0a0) - 3 k_tiles x 128
5. **c256_h10_oc512_3x3** (CBUF0=0xa7, DATA_SIZE1=0x003f0100, CONV2_LOW=0x0d0) - 3 k_tiles 176+176+160

**Lesson from 16:00 batch attempt:** the 6-shape batch FAILED because the k_tile OC partitioning was wrong. The full-OC hypothesis `((0, oc), (0, oc), (0, oc))` was never validated because attempt #5 raised a RuntimeError. Need to test full-OC partitioning on c40_h40_oc160_3x3 first.

### 4.2 Pointwise 1x1 (34 remaining)

**Sub-family A: c1280_h10 family (4 shapes) - c128 family body fields FAILED, need fresh decode**
- c1280_h10_oc24, c1280_h10_oc24_s1pvalid, c1280_h10_oc546, c1280_h10_oc546_s1pvalid
- This session: c128 family body fields (CBUF0=0x0b1, DATA_SIZE1=0x04ff0500) produced max_diff=225; reverted
- Need to read GEM2 capture directly: `/home/orangepi/npu/ops_rknn/dump/prefix_c1280_h10_oc24_s1pvalid_keep1_gem2/dump_gem2.txt` to find actual body field EMITs
- Likely candidates for body fields: CBUF0=0x14a or 0x250 pattern (per `nvdla/hw/cmod/cdma`), DATA_SIZE1=0x04ff0500 (ic=1280, in_h=10)

**Sub-family B: c1024 family (4 shapes) - c64_h1 body fields FAILED, need fresh decode**
- c1024_h1_oc1001, c1024_h7_oc1024 (b1 + cc variants)
- This session: c64_h1_oc128 body fields (CBUF0=0x0b1, DATA_SIZE1=0x003f0040) failed with max_diff=77
- Try natural DATA_SIZE1=0x03ff0400 (in_c=1024, in_h=1, derived from standard formula)

**Sub-family C: c72_h20 / c288 / c320 / c96 (6 shapes)**
- c72_h20_oc576, c288_h20_oc72, c320_h20_oc72, c96_h20_oc273 (pointwise 1x1, in_h=20)
- Need fresh body decode per family; the (in_c, in_h) tuple is uncommon

**Sub-family D: c384_h19 (3 shapes)**
- c384_h19_oc64, c384_h19_oc96, c384_h10_oc546
- in_h=19/10, in_c=384; needs fresh body decode

**Sub-family E: c480_h10 (2 shapes)**
- c480_h10_oc120, c768_h10_oc120
- in_h=10, in_c=480/768; c16_h80 family body fields don't transfer

**Sub-family F: c832_h7 (2 shapes)**
- c832_h7_oc48, c832_h7_oc48_s1pvalid
- OVERRIDES already exist in conv.py dicts but not in PREFIX_BY_K_SHAPES. Adding to PREFIX_BY_K_SHAPES gives max_diff=inf; need fresh body decode

**Sub-family G: pointwise-wide NONE (3 shapes)**
- c128_h1_oc24_s1pvalid, c480_h14_oc16, c512_h14_oc24
- Try c64_h1_oc128 or c256_h2_oc64 sibling body fields

**Sub-family H: BY_YK disabled (5 shapes)**
- c576_h19_oc273, c576_h19_oc96, c576_h20_oc72, c576_h20_oc96, c768_h20_oc96, c256_h28_oc256_k1x1
- Blocked at planner level; needs fundamentally different closure

**Sub-family I: c256_h3_oc546 + c256_h2_oc546 (1 active + 1 crash-fenced)**
- c256_h3_oc546: closest to passing (max_diff=35.69). First 512 OC are correct (0.01-0.03); OC 512-543 wrong.
- c256_h2_oc546: crash-fenced, do NOT submit directly

### 4.3 Depthwise (36 remaining)

All need `DEPTHWISE_BODY_SHAPES` membership in the depthwise code path. The c32_h150 family was the only depthwise promotion. Per-row BY_Y closure is the only working depthwise path so far.

**Priority order (largest = highest value):**
1. **c256_h28_oc256 (depthwise 3x3)** and its `cc` variant (2 shapes) - in_c=256, h=28, the largest depthwise after c1024
2. **c512_h14_oc512** and its `cc` variant (2 shapes) - in_c=512, h=14
3. **c1024_h7_oc1024** and its `cc` variant (depthwise 3x3 + 7x7) (4 shapes) - in_c=1024, the largest
4. **c576_h19 / c576_h20** (5 shapes) - in_c=576
5. **c384 / c320 / c192 / c144 / c128** depthwise (rest)

### 4.4 Known FAIL materializers (do not retry)

- c576_h19_oc12 (commit 3b520a0, max_diff=152): in_progress materializer at `conv_expt/in_progress/c576_h19_oc12_addition.py`. Needs fundamentally different approach.
- c256_h2_oc546 (crash-fenced): do NOT submit directly; reboots board.
- c1280_h10_oc24 (this session, reverted): c128 family body fields don't transfer; needs fresh decode.

### 4.5 Working tree state

CLEAN. The c1280_h10_oc24 attempt at 17:20 was reverted via `git checkout examples/conv.py` at 17:30. Untracked `??` files in `examples/` and `experimental/` are pre-existing scratch, intentionally untouched."""

assert old4 in text, "section 4 block not found"
text = text.replace(old4, new4)

# Update 2: Section 11 Distance to Goal
old11 = """## 11. Distance to Goal

- 78 fenced -> 0 fenced (need 78 promotions)
- **Captures: 100% done** (no longer a blocker; SS 2)
- **Materializers done in this session: 3** (c256_h3_oc128_1x1, c128_h3_oc256_1x1, c128_h3_oc256_3x3)
- **Materializers done prior session: 2** (c16_h80_oc128 3x3 + 5x5)
- **Total materializers done: 9 promoted shapes** (c256_h2_oc64, c256_h2_oc24, c256_h3_oc24, c64_h1_oc128, c192_h28_oc96, c256_h28_oc256, c512_h7_oc1024, c832_h7_oc48, c16_h80_oc64, c16_h80_oc128 3x3, c16_h80_oc128 5x5, c40_h40_oc320, c256_h2_oc546 NOT promoted)
- **Per-promotion cost:** ~1-2 hours (fresh GEM2 capture + body field decode + materializer code + guarded test + manifest entry + sweep verification). **c16_h80_oc128_3x3 took ~10 min** because the body field constants were identical to c16_h80_oc64.
- **Net promotions since 114 baseline: +28**
- **Target: 75 more promotions to reach 217/217**
- **ETA: 15-21 hours of focused work** (c16_h80 family took ~20 min total; spatial 3x3 batch is 6 more at ~10 min each if the full-OC k_tile hypothesis holds; pointwise 1x1 batch is 36 at ~5-10 min each after the first establishes the pattern; depthwise is the slow track at ~30 min each with new closure derivation)"""

new11 = """## 11. Distance to Goal

- **75 fenced -> 0 fenced (need 75 promotions)**
- **Captures: 100% done** (75/75 fenced have BOTH GEM1 and GEM2 captures; SS 2.3)
- **Materializers done this session: 3** (c256_h3_oc128_1x1, c128_h3_oc256_1x1, c128_h3_oc256_3x3)
- **Materializers done prior session: 2** (c16_h80_oc128 3x3 + 5x5)
- **Materializers done in earlier sessions: 10** (c256_h2_oc64, c256_h2_oc24, c256_h3_oc24, c64_h1_oc128, c192_h28_oc96, c256_h28_oc256, c512_h7_oc1024, c832_h7_oc48, c16_h80_oc64, c40_h40_oc320)
- **Total promoted shapes: 15** (some from prior sessions, e.g. c256_h2_oc546 NOT promoted because crash-fenced)
- **Per-promotion cost (recent):** c128_h3 family took ~10 min each (sibling-capture body fields transferred cleanly). c1280_h10_oc24 took 5 min for the attempt + revert. Per-shape fresh decode typically 30-60 min.
- **Net promotions since 114 baseline: +28**
- **Target: 75 more promotions to reach 217/217**
- **ETA: 15-25 hours of focused work**, broken down by tractability:
  - 5 spatial 3x3 siblings (BY_K/k_tile): ~50 min each if full-OC k_tile hypothesis works, ~2 hours each if not = 4-10 hours
  - 23 pointwise 1x1 BY_K/k_tile: ~30-60 min each after first establishes pattern = 12-23 hours
  - 3 pointwise-wide NONE: ~30 min each = 1.5 hours
  - 36 depthwise (after DEPTHWISE_BODY_SHAPES is added): ~1-2 hours each = 36-72 hours (long track)
  - 7 BY_YK disabled: BLOCKED, no ETA
  - 1 c576_h19_oc12: BLOCKED at this approach
  - 1 crash-fenced: BLOCKED, cannot submit
- **Highest ROI per minute:** pointwise 1x1 with sibling-capture body fields (c128 family model)."""

assert old11 in text, "section 11 block not found"
text = text.replace(old11, new11)

PATH.write_text(text)
print(f"OK, current_task.md is now {len(text.splitlines())} lines")
