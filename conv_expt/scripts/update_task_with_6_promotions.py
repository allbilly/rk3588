#!/usr/bin/env python3
"""Update current_task.md with this session's 6 promotions."""
from pathlib import Path
PATH = Path("/home/orangepi/rk3588/current_task.md")
text = PATH.read_text()

# Update header
text = text.replace(
    "**Last updated:** 2026-06-03 17:30 (Asia/Shanghai)",
    "**Last updated:** 2026-06-03 18:00 (Asia/Shanghai)"
)
text = text.replace(
    "**NPU health (verified 17:30):**",
    "**NPU health (verified 18:00):**"
)
text = text.replace(
    "**HEAD:** `4d16dcc` (per-family table refresh, 22 commits ahead of `origin/main`).",
    "**HEAD:** `dd8d652` (6 promotions this session, 25 commits ahead of `origin/main`)."
)
text = text.replace(
    "**Latest full sweep (20260603_172056):** `total=217 counts={'PASS': 142, 'FENCED': 75, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}`",
    "**Latest full sweep (20260603_175618):** `total=217 counts={'PASS': 145, 'FENCED': 72, 'FAIL': 0, 'ERROR': 0, 'TIMEOUT': 0}`"
)

# Update TL;DR progress
text = text.replace(
    "**`PASS=142 / 217` (65.4%)**, **`FENCED=75 / 217` (34.6%)**",
    "**`PASS=145 / 217` (66.8%)**, **`FENCED=72 / 217` (33.2%)**"
)
text = text.replace(
    "Net new promotions since the user-stated 114/103 stuck baseline: **+28 PASS** (114->142). Net new promotions this session: **+3** (c256_h3_oc128_1x1, c128_h3_oc256_1x1, c128_h3_oc256_3x3). Distance to goal: **75 more promotions**.",
    "Net new promotions since the user-stated 114/103 stuck baseline: **+31 PASS** (114->145). Net new promotions this session: **+6** (c256_h3_oc128_1x1, c128_h3_oc256_1x1, c128_h3_oc256_3x3, c128_h2_oc256_1x1, c192_h7_oc384_3x3, c256_h10_oc512_3x3). Distance to goal: **72 more promotions**."
)

# Update section 4.1
old4_anchor = "### 4.1 Highest priority: spatial 3x3 (5 remaining)"
if old4_anchor in text:
    # Replace section 4.1 content
    start_idx = text.index(old4_anchor)
    # Find next "###" or "## " 
    next_section_idx = text.index("\n### 4.2", start_idx)
    new4 = """### 4.1 Highest priority: spatial 3x3 (3 remaining)

**PROMOTED 2 of 5 spatial 3x3 shapes this session (c192_h7_oc384 max_diff=0.0624, c256_h10_oc512 max_diff=0.1121). 3 remaining:**

The 3 remaining spatial 3x3 siblings are all BY_K/k_tile-fenced. c40 and c72 have a DIFFERENT family_bits structure than standard EXACT11 (k_setup instead of k_half), which the standard 11-task code does not write. c128_h5_oc256 needs sibling-capture body field decoding (capture has different CBUF0 from c128 family).

**Order of attempts:**
1. **c40_h40_oc160_3x3** (CBUF0=0x84, DATA_SIZE1=0x00270028, CONV2_LOW=0x160) - capture has k_setup+k_tile family bits, not k_half. Standard path FAIL max_diff=163.
2. **c72_h20_oc288_3x3** (CBUF0=0x0a2, DATA_SIZE1=0x00070048, CONV2_LOW=0x140) - same issue. Standard path FAIL max_diff=204.
3. **c128_h5_oc256_3x3** (CBUF0=0x0b1 or 0x0b7, DATA_SIZE1=0x003f0080, CONV2_LOW=0x080) - tried both 0x0b1 and 0x0b7, FAIL max_diff=259. Needs fresh body field decode from GEM2.

**Path forward:** write a special `_exact11_task_regs` case for c40_h40_oc160 and c72_h20_oc288 (like the c832_h7_oc48 case) that writes the correct k_setup+k_tile structure. Or use a 6-task closure (1 setup + 2 k_setup + 3 k_tile) instead of the standard 11-task."""
    text = text[:start_idx] + new4 + text[next_section_idx:]
    print("Section 4.1 updated")

# Update section 4.2 with c128_h2_oc256 promotion
old5_anchor = "**Sub-family A: c1280_h10 family (4 shapes) - c128 family body fields FAILED, need fresh decode**"
if old5_anchor in text:
    insert_text = """**PROMOTED c128_h2_oc256_1x1 (in_c=128, in_h=2, oc=256) at 17:46, max_diff=0.0151.** Key finding: DMA_CON2=0x0ffffffc (NOT 0x0ffffffd like c128_h3 family). All other body fields (CBUF0, DATA_SIZE1, CVT_CON0) match c128 family. KT_TILE_SPLITS=((0, 96), (96, 96), (192, 64)) summing to 256.

"""
    text = text.replace(old5_anchor, insert_text + old5_anchor)
    print("Section 4.2 c128_h2 promotion note added")

# Update SS 4.5 with net promotions
text = text.replace(
    "**Materializers done this session: 3** (c256_h3_oc128_1x1, c128_h3_oc256_1x1, c128_h3_oc256_3x3)",
    "**Materializers done this session: 6** (c256_h3_oc128_1x1, c128_h3_oc256_1x1, c128_h3_oc256_3x3, c128_h2_oc256_1x1, c192_h7_oc384_3x3, c256_h10_oc512_3x3)"
)
# Update total promoted count
text = text.replace(
    "- **Total promoted shapes: 15**",
    "- **Total promoted shapes: 18**"
)
# Update net promotions
text = text.replace(
    "- **Net promotions since 114 baseline: +28**",
    "- **Net promotions since 114 baseline: +31**"
)
text = text.replace(
    "- **Target: 75 more promotions to reach 217/217**",
    "- **Target: 72 more promotions to reach 217/217**"
)

# Add new section 13.6 about this session's additional 3 promotions
new_section = """

### 13.6 Additional promotions (this session, 17:46-18:00)

After the 16:35-16:40 promotions, continued with capture-derived body field decoding for more shapes:

**PROMOTED 3 more shapes via GEM2 capture body field decode:**

| Shape | Commit | max_diff | Key constants | Lesson |
|---|---|---|---|---|
| c128_h2_oc256_1x1 | `d023650` | 0.0151 | CBUF0=0x0b1, DATA_SIZE1=0x003f0080, CVT_CON0=0x000b, DMA_CON2=**0x0ffffffc** (NOT 0x0ffffffd), KT_TILE_SPLITS=((0,96),(96,96),(192,64)), CONV2_LOW=0x008 | DMA_CON2 differs from c128_h3 family (0x0ffffffc vs 0x0ffffffd) |
| c192_h7_oc384_3x3 | `dd8d652` | 0.0624 | CBUF0=0x0b1, DATA_SIZE1=0x003f00c0, CVT_CON0=0x000b, DMA_CON2=0x0015, KT_TILE_SPLITS=((0,128),(128,128),(256,128)), CONV2_LOW=0x0a0 (FEATURE_GRAINS=10) | First 192-family spatial 3x3 promotion |
| c256_h10_oc512_3x3 | `dd8d652` | 0.1121 | CBUF0=0x0a2, DATA_SIZE1=0x003f0100, CVT_CON0=0x000b, DMA_CON2=0x003c, KT_TILE_SPLITS=((0,176),(176,176),(352,160)), CONV2_LOW=0x0d0 (FEATURE_GRAINS=13) | CBUF0=0x0a2 (not 0x0b1) for in_c=256 family |

**Failed attempts (reverted):**
- c72_h20_oc288_3x3 (max_diff=204): different family_bits structure (k_setup not k_half)
- c40_h40_oc160_3x3 (max_diff=163): same family_bits issue
- c128_h5_oc256_3x3 (max_diff=259): c128 family body fields don't transfer
- c1280_h10_oc24 (max_diff=225, earlier): c128 family body fields don't transfer to c1280
- c1024_h1_oc1001, c1024_h7_oc1024 (max_diff=145-220): pointwise-wide special path needed

**Other 75 fenced shapes verified captured (per SS 2.3); no capture work needed.**

**Key methodological insight from this session:**
- The 6 promotions all used GEM2 capture EMIT statements to extract per-row body field constants
- Family bits (RESERVED_0 in CONV_CON2) determine which k_tile structure to use:
  - 0x10000000 = k_half (standard EXACT11)
  - 0x40000000 = k_setup (some spatial 3x3 with large in_h)
  - 0x50000000 = k_tile (standard)
- Spatial 3x3 with k_setup+k_tile family bits (c40, c72) need a custom `_exact11_task_regs` branch
"""

text = text + new_section
PATH.write_text(text)
print(f"OK, current_task.md is now {len(text.splitlines())} lines")
