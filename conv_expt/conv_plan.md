# conv_expt Active Plan

## Goal

Build a clean FP16 CONV implementation for RK3588 NPU with:

- `examples/conv.py < 1000` lines
- no CPU/GPU fallback
- no RKNN hex/task table replay in the final runtime
- all unsupported shapes fenced before allocation
- eventual `217/217 PASS`

The final design is descriptor-driven:

shape
  -> CBUF pressure model
  -> NONE / BY_Y / BY_K / BY_YK
  -> descriptor rows
  -> generic register emitter family
  -> safe submit closure

## Current Truth

`examples/conv.py` is currently a hardware research scratchpad, not the final clean implementation.

Current observed state:

- `examples/conv.py`: ~2926 lines
- `conv_expt/conv_tile_planner.py`: ~438 lines
- latest handoff: PASS=146, FENCED=71, FAIL=0
- many recent passes came from prefix-replay-derived per-shape body constants
- this proves the hardware path, but not the clean abstraction yet

The old plan is stale because it still describes a ~530-line runtime and a 103/114 sweep state.

## Strategic Decision

Do not chase 217/217 by adding more per-shape special cases first.

Instead:

- Use current `examples/conv.py` as the oracle/scratchpad.
- Extract generic emitter rules from the promoted families.
- Build a smaller descriptor-emitter runtime.
- Use the 217-shape sweep as the acceptance test, not as an excuse to grow shape tables.

## Evidence

External compiler references support the simple planner direction:

- RKNN uses high-level split modes equivalent to `NONE/BY_Y/BY_K/BY_YK`.
- ONNC NVDLA uses CBUF allocation feasibility, not many ad hoc per-shape strategies.
- NVDLA SW splits mainly by CBUF pressure, output height, and output channel limits.
- NVDLA HW constraints are CBUF data banks, weight banks, fetch grains, and reuse.

Conclusion: the six old Python strategies are local implementation history, not the architecture.

## Final Runtime Shape

Final `examples/conv.py` should contain:

- shape parser
- input/weight packing
- planner call
- generic descriptor-to-register emitter
- task/PC-chain builder
- run/compare CLI

It should not contain:

- long per-shape promotion dictionaries
- raw RKNN regcmd blobs
- one-off exact11/exact12 branches per shape
- crash-prone submit experiments
- proof-only report code

## Emitter Families

Implement generic emitters for:

| Family | Meaning | Status |
|---|---|---|
| `setup` | full tile / NONE | already broadly understood |
| `y_tile` | output-height split | works for pointwise; spatial needs generic closure |
| `k_half` | intermediate K family | partially understood from RKNN |
| `k_tile` | output-channel split | many shape-local passes; needs generic body derivation |
| `aux` | DPU/PPU auxiliary row | needed for RKNN-like closures |

The task is to make these families formula-driven.

## Near-Term Milestones

1. Extract body-field formulas from promoted pointwise/spatial BY_K shapes.
2. Replace `CBUF0_OVERRIDES`, `DATA_SIZE1_OVERRIDES`, `DMA_CON2_OVERRIDES`, and `KT_TILE_SPLITS` with derived fields where possible.
3. Keep only a small allowlist for genuinely unresolved or crash-fenced shapes.
4. Solve one depthwise family generically instead of adding `DEPTHWISE_BODY_SHAPES` entries one by one.
5. Solve BY_YK by composing independent Y/K windows plus family emitters.
6. Run the 217 sweep after each generic family promotion.

## Acceptance Gates

Every runtime promotion must pass:

```sh
python3 examples/simple_add.py
timeout 30 python3 examples/conv.py <shape>
python3 examples/simple_add.py
timeout 200 python3 sweep_217.py --skip-health
Final acceptance:
python3 examples/simple_add.py
timeout 200 python3 sweep_217.py --skip-health
python3 examples/simple_add.py
wc -l examples/conv.py
Expected final result:
- PASS=217
- FENCED=0
- FAIL=0
- ERROR=0
- TIMEOUT=0
- examples/conv.py < 1000
Stop Rules
- Do not add more per-shape constants unless they are used to derive a family rule.
- Do not copy RKNN task tables into the final runtime.
- Do not change submit-critical fields without a pre/post health check.
- Do not edit examples/kernel_6_18/ for this work.
- Do not run crash-fenced shapes directly.

**Answer To The Main Question**
Do **not** achieve all 217 first by brute force. That will likely create another 2k-line implementation.

Better path:

- Use the current 146/217 as evidence that prefix replay works.
- Stop treating each fenced shape as a promotion target.
- Treat remaining fenced shapes as a classification problem: which generic emitter rule is missing?
- Promote families, not shapes.
