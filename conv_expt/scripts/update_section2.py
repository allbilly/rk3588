#!/usr/bin/env python3
"""Replace section 2 (per-family progress table) with the new comprehensive version including fence reason and capture status per shape."""
from pathlib import Path

PATH = Path("/home/orangepi/rk3588/current_task.md")
text = PATH.read_text()

# Find the section 2 block and the start of section 3
start_marker = "## 2. Per-Family Progress + Capture Status (78 FENCED)"
end_marker = "## 3. Goal Definition (Explicit)"

s = text.index(start_marker)
e = text.index(end_marker)
old_block = text[s:e]

# Read the per-shape detail from the new table file
TABLE = Path("/home/orangepi/rk3588/sweep_results/family_progress_table_20260603_1730.txt").read_text()
# Extract the per-shape detail (everything from "=== PER-SHAPE DETAIL" onward)
detail_start = TABLE.index("=== PER-SHAPE DETAIL")
shape_detail = TABLE[detail_start:].rstrip()

# Build the new section
new_block = """## 2. Per-Family Progress + Capture Status (75 FENCED)

**ANSWER: YES, ALL 75 FENCED SHAPES ALREADY HAVE CAPTURE. The capture phase is COMPLETE.**

```
Total fenced: 75
With any capture (GEM1 or GEM2): 75/75 (100%)
With GEM2 body capture: 75/75 (100%)
With NO capture at all: 0/75 (0%)
```

| Family                              | Fenced | G1 | G2 | %G2 | Path needed                                       | Promotion candidates this session |
|---|---:|---:|---:|---:|---|---|
| **pointwise 1x1 (k1_g1)**            | 34 | 34 | 34 | 100% | EXACT11 BY_K body overrides; per-shape CBUF0/DATA_SIZE1/CONV2_LOW/CVT_CON0/DMA_CON2 + KT_TILE_SPLITS | c1280_h10_oc24 (reverted), c1024_h1_oc1001, c832_h7_oc48 — body fields from sibling family DON'T transfer |
| **depthwise 3x3 (k3_g=in_c)**       | 30 | 30 | 30 | 100% | DEPTHWISE_BODY_SHAPES membership + per-row body; c32_h150 family is the only working depthwise path | 0 (c128_h3_oc128 was a known BY_K timeout) |
| **spatial 3x3 (k3_g1)**             |  5 |  5 |  5 | 100% | EXACT11 BY_K body overrides; c16_h80 family done (3x3 + 5x5), c128_h3 family done (1x1 + 3x3) | c40_h40_oc160, c72_h20_oc288, c192_h7_oc384, c256_h10_oc512, c128_h5_oc256 — all need full-OC k_tile hypothesis (still blocked) |
| **depthwise 5x5 (k5_g=in_c)**       |  4 |  4 |  4 | 100% | DEPTHWISE_BODY_SHAPES + new weight per-kernel constant (800 bytes for 5x5) | 0 |
| **depthwise 7x7 (k7_g=in_c)**       |  2 |  2 |  2 | 100% | Shares c1024_h7_oc1024 capture; needs new kernel-size-7 path | 0 |
| **TOTAL**                           | **75** | **75** | **75** | **100%** | - | **+3 this session** |

### 2.1 Fence-reason sub-classification (by sweep_217.py error message)

| Fence reason | Count | Tractability | Notes |
|---|---:|---|---|
| BY_K/k_tile (108/104/26 closure) | 27 | TRACTABLE | 23 pointwise 1x1 + 4 spatial 3x3; the path exists, need per-shape body fields |
| depthwise BY_YK (DEPTHWISE_BODY) | 19 | TRACTABLE once DEPTHWISE_BODY_SHAPES exists | c128_h56, c144_h56, c192_h38, c256_h28, c320_h40, c384_h19, c576_h19/20, c64/96_h112/150 |
| depthwise BY_K (DEPTHWISE_BODY) | 14 | TRACTABLE once DEPTHWISE_BODY_SHAPES exists | c128_h3, c256_h3, c256_h10, c384_h10, c512_h5/14, c1024_h7 (3x3+7x7) |
| BY_YK disabled (planner) | 7 | BLOCKED at planner | b1_c256_h28_oc256_k1x1, c576_h19_oc273/96, c576_h20_oc72/96, c768_h20_oc96, cc_c256_h28_oc256_k1x1 — mixed Y/K setup and k_half semantics unresolved |
| depthwise BY_Y (DEPTHWISE_BODY) | 3 | TRACTABLE once DEPTHWISE_BODY_SHAPES exists | c32_h112, c32_h150 (b1), cc_c32_h112 |
| pointwise-wide NONE (108-row closure) | 3 | TRACTABLE via c64_h1_oc128 or c256_h2_oc64 sibling | c128_h1_oc24, c480_h14_oc16, c512_h14_oc24 |
| pointwise-wide BY_Y (row closure) | 1 | BLOCKED | c576_h19_oc12 — in_progress materializer at conv_expt/in_progress/c576_h19_oc12_addition.py, max_diff=152 |
| crash-fenced (do NOT submit) | 1 | BLOCKED | b1_c256_h2_w2_oc546 — reboots board; do not run directly |

### 2.2 Promotion path summary (in order of tractability)

1. **27 BY_K/k_tile** (pointwise 1x1 + spatial 3x3): the EXACT11 BY_K closure exists; need per-shape body field overrides. **3 promotions this session** (c256_h3_oc128, c128_h3_oc256, c128_h3_oc256_3x3) all used sibling-capture body fields.
2. **36 depthwise** (19 BY_YK + 14 BY_K + 3 BY_Y): all need `DEPTHWISE_BODY_SHAPES` membership. The per-row BY_Y path works for c32_h150 (1 promotion prior session). Other depthwise needs per-row BY_Y or new BY_K/BY_YK closure with body field derivation.
3. **3 pointwise-wide NONE**: tractable via sibling body fields (c64_h1_oc128 or c256_h2_oc64 patterns).
4. **7 BY_YK disabled**: blocked at planner level. Needs a fundamentally different closure (mixed Y/K setup and k_half semantics).
5. **1 pointwise-wide BY_Y (c576_h19_oc12)**: in-progress materializer exists, FAIL max_diff=152.
6. **1 crash-fenced (c256_h2_oc546)**: do NOT run directly, reboots board.

### 2.3 Per-shape capture status + fence reason (75 FENCED, all 100% captured)

"""
new_block += "```\n" + shape_detail + "\n```\n\n"
# Now add the section 3 header (we consumed it in slicing)
new_block += "## 3. Goal Definition (Explicit)\n"

text = text[:s] + new_block + text[e+len(end_marker):]
PATH.write_text(text)
print(f"OK, current_task.md is now {len(text.splitlines())} lines")
