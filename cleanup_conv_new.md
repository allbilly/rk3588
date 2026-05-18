This is the key finding. NVDLA doesn't need OC-tiling for FP16 because NVDLA OpenDLA has 32 CBUF banks, while RK3588 has only 12:
Shape	FP16 weight banks	OpenDLA (32)
160→320 dense 3×3	30	Fits
512→24 pointwise	2	Fits
1024→1001 pointwise	64	Both OVER
NVDLA's splitData() (partial-H split) works for FP16 on OpenDLA's 32 banks because all realistic shapes fit — the 160→320 case that kills RK3588 fits fine in 30 of 32 banks. The only shape that overflows both (1024→1001 classifier head) NVDLA also can't handle — it simply doesn't support that shape.
The 3 OC-tiling strategies in conv_new_clean.py exist because RK3588 has less than half the CBUF of reference NVDLA (12 banks × 32KB = 384KB vs 32 banks × 32KB = 1MB).
▣  Plan · CrofAI: deepseek-v4-pro · 1m 13s
so it return doesn't support that shape?
▶ Thought for 3.2s
⚙ deepwiki_ask_question [question=In nvdla/sw umd/core/src/compiler/engine-ast/ConvCoreNode.cpp, when the convolution operation's weights or data don't fit in CBUF even after partial-H splitting (e.g. weights alone exceed total CBUF banks), what happens? Does the compiler:
1. Return an error/assertion failure?
2. Silently produce a truncated/wrong result?
3. Fall back to a different compilation strategy?
4. Split along a different dimension?
Look at the splitData(), determineSplitDataRatios(), and splitNodesInternal() functions for error handling when banks don't fit., repoName=nvdla/sw]
▶ Thought for 4.0s
Yes. NVDLA's compiler returns NvDlaError_NotSupported or NvDlaError_InsufficientMemory and refuses to compile the subgraph. The exact flow in splitNodesInternal():
1. 
Try no split → fail
2. 
Try weight-only split (ping-pong/inline KG) → fail  
3. 
Try partial-H split → fail
4. 
→ ORIGINATE_ERROR_FAIL(NvDlaError_NotSupported, "Not enough banks to hold the weight data")
No silent truncation, no im2col, no OC-tiling. The shape simply doesn't run.
So the fundamental design difference is:
 	NVDLA SW
Shape doesn't fit CBUF	Refuse (NvDlaError_NotSupported)
Why it works for NVDLA	32 banks + int8 + target shapes fit
The 3 OC-tiling strategies are pragmatic hacks that go beyond NVDLA's reference compiler — NVDLA simply rejects shapes that need them. On RK3588's smaller CBUF (12 vs 32 banks), you either add these workarounds or reject half the FP16 model layers.
▣  Plan · CrofAI: deepseek-v4-pro · 32.5s
ask deepwiki on how onnc handles it
▶ Thought for 935ms
⚙ deepwiki_ask_question [question=In the ONNC compiler, how does it handle NVDLA convolution tiling when the operation doesn't fit in CBUF? Specifically:
1. Look in lib/Target/NVDLA/ for any convolution tiling or splitting logic.
2. Does ONNC have multiple tiling strategies (H-split, OC-split, im2col)?
3. How does it handle FP16 precision vs INT8 for tiling decisions?
4. Look for any "split", "tile", "bank", or "CBUF" related code in the NVDLA backend.
5. Is there any output-channel splitting (OC-tiling)?
Show relevant file names, function names, and key code snippets., repoName=ONNC/onnc]
▶ Thought for 8.0s
ONNC handles it completely differently — with multiple graduated strategies rather than rejecting shapes:
ONNC's approach: CbufAllocType enumeration
Unlike NVDLA SW (which only does H-split then gives up) and conv_new_clean.py (which adds OC-tiling hacks), ONNC tries 6 strategies in priority order:
kFullDataFullWeight       → everything fits (ideal)
kFullDataPartialWeight    → full input, tile weights across submitions
kFullDataMinimumWeight    → full input, most aggressive weight tiling
kSplitDataFullWeight      → H-split input, full weights
kSplitDataPartialWeight   → H-split input + tile weights
kSplitDataMinimumWeight   → H-split input + most aggressive weight tiling
If none of these 6 fit, it falls back to input channel splitting (SplitConvPass): splits the conv along the input channel dimension into multiple smaller convs, then sums their results — effectively splitting a Conv(C→M) into t × Conv(C/t → M) + Sum.
Key difference from conv_new_clean.py
Strategy	conv_new_clean.py	ONNC
H-split	✅ (Mesa-style)	✅ (kSplitData*)
Weight tiling	❌ (no equivalent)	✅ (k*PartialWeight — load partial weights, reuse across tiles)
IC-split (sum)	❌ (OC-tile instead)	✅ (SplitConvPass — split input channels, sum results)
OC-tile	✅ (3 strategies)	❌
ONNC's weight tiling (kFullDataPartialWeight) is the key strategy that conv_new_clean.py is missing. When weights exceed CBUF, ONNC loads a subset of weight banks per submition and iterates — this avoids both OC-tiling and im2col entirely. Combined with IC-splitting as the final fallback, it covers the 12-bank RK3588 CBUF without needing the 3 OC-tiling hacks.
▣  Plan · CrofAI: deepseek-v4-pro · 45.4s
what is o and c in oc tiling, why onnc doesnt need oc tiling
▶ Thought for 29.3s
"O" = output channels (the M dimension, also called out_c).  
"C" = input channels (the K dimension, also called in_c).  
So "OC-tiling" means tiling along the output channel dimension: splitting a Conv(C→M) into multiple Conv(C→M_tile) tasks that each produce a subset of output channels.
Why ONNC doesn't need OC-tiling
ONNC has three orthogonal strategies that together cover all CBUF-overflow cases without ever tiling output channels:
Strategy	What it does
Weight sub-tiling (kFullDataPartialWeight)	Keeps full input in CBUF, loads only partial weight banks per submition. Reuses input across submitions and accumulates in CACC.
IC-splitting (SplitConvPass)	Splits input channels into groups: Conv(C/t → M) × t, then Sum. Each sub-conv's weights are 1/t the size.
H-splitting (kSplitData*)	Tiles input height (same as Mesa/NVDLA).
The key insight: OC-tiling is a workaround for not having weight sub-tiling. OC-tiling also reduces weight CBUF usage (since weights are M × C × KH × KW, halving M halves the weight size), but it complicates output buffer layout. ONNC's weight sub-tiling is cleaner because:
ONNC weight tile:     Input unchanged, load W/2 weight banks → produce partial sum
                      Reuse same input, load next W/2 weight banks → accumulate
                      → Full output in one buffer, no stitching
OC-tile (conv style): Split M → each tile needs separate output area
                      → Must manage output buffer offsets and unpacking
ONNC's approach maps naturally to NVDLA's hardware: the CACC (Convolution Accumulator) already accumulates partial sums across CBUF iterations — that's exactly what weight sub-tiling uses. conv_new_clean.py's OC-tiling bypasses CACC accumulation and does the stitching in software instead.
▣  Plan · CrofAI: deepseek-v4-pro · 44.8s
which tiling is better
▶ Thought for 24.6s
Both approaches have tradeoffs. Here's the comparison:
ONNC's weight sub-tiling (the "right" way)
Cycle 1: Load full input into CBUF, load weight_chunk_0 → produce partial sum in CACC
Cycle 2: Same input (CBUF reuse), load weight_chunk_1 → accumulate in CACC
...
Done: CACC holds full output → write once
Pros: Single contiguous output. No stitching. Uses CACC accumulator as designed.
Cons: Hard to get right on RK3588 — requires verifying CACC accumulation works across weight sub-mitions, and the driver/reset path must not clear CACC between cycles. Needs proof that CNA_DATA_SIZE3 (atomics) works with partial weights.
OC-tiling (conv_new_clean.py's approach)
Task 1: out_c[0:16] → write to output_buf[0:16]
Task 2: out_c[16:32] → write to output_buf[16:32]
...
Done: stitch separate output regions in software
Pros: Each tile is a complete independent conv. Simple to debug. No hardware accumulation magic needed.
Cons: Output must be stitched from multiple regions. SURFACE_ADD and DST_SURF_STRIDE math is fragile. 3 nearly-duplicate loop nests.
Which is better for RK3588?
For a production compiler like ONNC: weight sub-tiling wins — it's general, clean, and maps to hardware.
For a research/debug Python script like conv_new_clean.py: OC-tiling is pragmatic — you can verify each tile independently without trusting CACC accumulation across submitions. But the current implementation is messy (3 strategies = 3 copies of the same loop).
The real answer is that neither fully solves the problem — they handle different overflow modes:
Overflow	ONNC fix
Weights > CBUF	Weight sub-tile
Input > CBUF	H-split
IC >> OC precision	N/A (int8)
If you wanted to clean up conv_new_clean.py, the best path would be to replace spatial_oc_serial and depthwise_spatial_tiled with a single H-tiler (like Mesa's split_tasks_like_mesa), keep the two OC paths that ONNC's approach can't replace (im2col for weight overflow, pointwise OC for FP16 precision), and drop the weight-reuse chain that chains row-tiles into one submit (lines 941-967) — it adds complexity for marginal benefit. That would collapse ~260 lines of run_conv2d to ~120 lines while keeping correctness.
Want me to plan that refactor?
▣  Plan · CrofAI: deepseek-v4-pro · 41.3s
compare run_conv2d vs run_gemm again
▶ Thought for 1.8s
→ Read examples/gemm.py [offset=465, limit=55]
→ Read examples/kernel_6_18/conv_new_clean.py [offset=746, limit=264]
▶ Thought for 6.0s
Here's the final comparison with all context baked in:
run_gemm — 55 lines, 1 tiling dimension
m_tile = GEMM_INPUT_BANKS * CBUF_BANK_SIZE // input_row_bytes
for start in range(0, m, m_tile):
    task_regs.append(make_gemm_regs(tile_m, ...))
write_regs_to_npu_task(task_regs)
npu_submit(...)
Why it's clean: GEMM has exactly one overflow dimension (M = rows). Input rows exceed CBUF → split M. Weight packing is one fixed pattern (transpose(0,2,1,3)). Output is one flat format. The problem space is: m, n, k — that's it. No kernel sizes, no depthwise, no groups, no precision issues.
run_conv2d — 264 lines, 6 strategies
Strategy	When	Lines	What tiles
spatial_im2col	Weight banks > 4 or output > 4MB	41	unroll → OC + rows
grouped_serial	Groups > 1, spatial	24	groups
spatial_oc_serial	out_c > 8, in_c odd	25	OC (8) + rows
depthwise_spatial_tiled	Depthwise + spatial	29	channels + rows
_needs_pointwise_oc_tile	Pointwise IC >> OC	29	OC + rows
_needs_pointwise_tile	2 specific MobileNet tuples	25	rows only
Default	Everything else	28	rows (H-split)
The 3 root causes of the 5× size difference
Root cause	gemm.py	conv_new_clean.py
Overflow dimensions	1 (M)	4 (rows, OC, IC, channels)
Output formats	1 (flat)	3 (flat, nc1hwc2, grouped_spatial)
Weight layouts	1 (16×32 strip)	5 (kh_major, default, pointwise_wide, dw_spatial, grouped)
About ~130 of the 264 lines are duplicated loop structure — each of the 6 strategies has its own independent loop body that does: slice input → pack tile weights → generate regs → submit → unpack output. With ONNC's approach (add weight-sub-tiling, keep 1 H-tiler, keep only the FP16 precision OC path), you'd collapse to ~130 lines total in run_conv2d.
▣  Plan · CrofAI: deepseek-v4-pro · 45.2s
add onnc strategies as well and comapre and comment if onnc strategies alone is enough
▶ Thought for 27.4s
⚙ deepwiki_ask_question [question=In ONNC's NVDLA backend, for the `kFullDataPartialWeight` CbufAllocType strategy:
1. Does the weight sub-tiling split along output channels (M dimension)?
2. Does CACC accumulation across submitions work correctly for FP16?
3. What is the `getBanksForPartialWeights` function doing? How does it split weights?
4. Does ONNC's SplitConvPass fix FP16 precision issues by reducing accumulation depth (splitting input channels)?
Show relevant code from NvDlaBackend.cpp, NvDlaMeta.h, or ConvCoreNode related code., repoName=ONNC/onnc]
▶ Thought for 17.8s
This changes the picture significantly. Let me rebuild the comparison with ONNC's actual strategies:
Complete comparison table
Strategy	gemm.py	conv_new_clean.py	ONNC
M/row split	✅ for start in range(0, m, m_tile)	✅ (default path, H-tile)	✅ kSplitData*
HW weight sub-tile	❌	❌	✅ kFullDataPartialWeight
HW weight min	❌	❌	✅ kFullDataMinimumWeight
H + weight sub-tile	❌	❌	✅ kSplitDataPartialWeight
IC-split (SplitConvPass)	❌	❌	✅ SplitConvPass
im2col + OC-tile	❌	✅ spatial_im2col	❌
OC-serial + H	❌	✅ spatial_oc_serial	❌
DW channel + H	❌	✅ depthwise_spatial_tiled	❌
OC-tile for precision	❌	✅ _needs_pointwise_oc_tile_*	❌
Weight pack	1 pattern	5 patterns	N/A (HW handles)
Output unpack	1 format	3 formats	1 format
Are ONNC's strategies alone enough for RK3588 FP16?
Yes — but with caveats. Here's how each overflow scenario maps:
Scenario	conv_new_clean path	ONNC replacement
Weights overflow (3×3 160→320)	spatial_im2col	kFullDataPartialWeight — HW loads weights in OC groups, CACC accumulates. No im2col needed
FP16 precision (512→24)	_needs_pointwise_oc_tile_schedule	SplitConvPass — splits 512→24 into 4× Conv(128→24) + Sum. Each sub-conv accumulates 128 products instead of 512
Input overflow (large H)	Default H-split	kSplitDataFullWeight
Depthwise spatial	depthwise_spatial_tiled	kSplitDataFullWeight (depthwise weights are tiny, only input overflows)
Grouped conv	grouped_serial	SplitGroupConvPass (same thing)
The 2 gaps where ONNC's set is unproven on RK:
1. 
HW weight sub-tiling (kFullDataPartialWeight): conv_new_clean.py doesn't use this at all. ONNC relies on it for weight overflow instead of im2col. If it doesn't work on RK, you'd keep im2col.
2. 
FP16 IC>>OC precision: ONNC's SplitConvPass (IC-split + Sum) would fix this naturally — conv_new_clean's OC-tile is a workaround for not having IC-split.
Summary
ONNC's 7 graduated strategies (6 CbufAllocType + 1 SplitConvPass) would replace all 6 of conv_new_clean's special-case paths with 3 unified mechanisms:
1. HW weight sub-tile  → replaces im2col, oc_serial, depthwise_tiled
2. H-split             → already shared
3. IC-split + Sum      → replaces pointwise OC schedules (both CBUF and precision)
But #1 requires verifying the RK3588 CNA supports CACC accumulation across weight-sub-tile cycles — something conv_new_clean.py never tests. Without that proof, conv_new_clean's im2col fallback stays necessary.
▣  Plan · CrofAI: deepseek-v4-pro · 1m 34s
add nvdla/sw tiling strageies as well
▶ Thought for 2.9s
⚙ deepwiki_ask_question [question=In nvdla/sw umd/core/src/compiler/engine-ast/ConvCoreNode.cpp, enumerate ALL tiling/splitting strategies attempted in order. Specifically:
1. In splitNodesInternal(), what is the full ordered list of strategies tried before giving up?
2. What are "ping-pong weight" and "single KG" modes? How do they differ from partial-H split?
3. After partial-H split fails, what happens? Is there a 3rd, 4th strategy?
4. Is there any code path that tries channel-splitting (split along C) for convolution?
Show the exact enum values or if/else-if chain ordering., repoName=nvdla/sw]
▶ Thought for 16.1s
Here's the updated comprehensive comparison with all 4 implementations:
NVDLA SW: 9 graduated strategies (all combinations of 3 axes)
NVDLA SW only tiles 2 dimensions: H (input rows) and K (output-channel groups via hardware weight split). No IC-split, no im2col.
Ordered attempts:
  1. No split                    ─ Full Input + Full Weight
  2. HW ping-pong weight         ─ Full Input + 2× min-weight-bank
  3. HW single-KG weight         ─ Full Input + 1× min-weight-bank
  4. Profile-databanks + pingpong─ Partial weight via profile allotment
  5. Profile-databanks + 1KG     ─ Same, more aggressive
  6. Partial-H + Full Weight     ─ H-split + full weights
  7. Partial-H + pingpong weight ─ H-split + 2× min-weight
  8. Partial-H + 1KG weight      ─ H-split + 1× min-weight
  9. Partial-H + SW split-K      ─ H-split + software weight-split (last resort)
  ────────────────────────────────────────────────────────
  ↗ NvDlaError_NotSupported      ─ Reject shape
Final comparison: all 4
Strategy	gemm.py	conv_new_clean.py	NVDLA SW	ONNC
No split	✅ (up to m_tile)	✅ (fast path)	✅ (#1)	✅ kFullDataFullWeight
HW weight ping-pong	❌	❌	✅ (#2,4,7)	✅ kFullDataPartialWeight
HW weight 1KG	❌	❌	✅ (#3,5,8)	✅ kFullDataMinimumWeight
SW weight split-K	❌	❌	✅ (#9, last resort)	❌
H-split (rows)	✅ M-split	✅ (default)	✅ (#6-9)	✅ kSplitData*
im2col + OC-tile	❌	✅ spatial_im2col	❌	❌
OC-serial + H	❌	✅ spatial_oc_serial	❌	❌
DW channel + H	❌	✅ depthwise_spatial_tiled	❌	❌
OC-tile (precision)	❌	✅ _needs_pointwise_oc_tile	❌	❌
IC-split + Sum	❌	❌	❌ (FIXME: "will follow")	✅ SplitConvPass
Group-serial	❌	✅ grouped_serial	❌	✅ SplitGroupConvPass
Is any one set alone sufficient for RK3588 FP16?
Implementation	Covers weight overflow?	Covers input overflow?	Covers FP16 precision?
gemm.py	N/A (gemm only)	✅	N/A
conv_new_clean.py	✅ (im2col)	✅ (H-split)	✅ (OC-tile)
NVDLA SW	✅ (HW split-K)	✅ (H-split)	No (no IC-split)
ONNC	✅ (HW split-K)	✅ (H-split)	✅ (IC-split + Sum)
ONNC's set is the cleanest of the 4: 7 graduated strategies (6 CbufAllocType + SplitConvPass) cover all RBUF overflow + FP16 precision cases with no duplicate loop bodies. But it requires verifying HW weight split-K works on RK3588 (CNA_CBUF_CON0_WEIGHT_REUSE + CACC persistence across submitions).
NVDLA SW is the most conservative — it rejects shapes that need IC-split, which is fine for its target (32-bank CBUF + int8 focus), but insufficient for RK3588 FP16 pathological shapes.
conv_new_clean.py is the most pragmatic — uses im2col as a fallback when weight split-K is unproven, uses OC-tile for precision when IC-split isn't available. Verbose but safe.