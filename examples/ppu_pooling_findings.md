# RK3588 PPU capability findings

`python examples/pooling.py` is the executable decoded-register proof for the
silicon results in this table.

| Capability | Result | Evidence |
| --- | --- | --- |
| Average / maximum / minimum pool | Proven | Native PPU selector values 0/1/2 all produce the reference result. Minimum is tested directly, not synthesized with negation. |
| Global average | Proven without host repair | A 3x5 pool uses dimension-specific HLS FP17 reciprocals and directly returns the mean. The old `0x7800`/`0x7800` recipe is only correct for 2x2 and otherwise returns sum/4. |
| Global maximum | Proven | The kernel/stride-16 probe is also a direct 16x16 global maximum. |
| `INDEX_EN` | Proven, local only | It writes `(row << 3) | column` for each channel. This is a pool-window position, not general ArgMax. |
| PPU FP16 / INT8 / INT16 | Proven | Native maximum-pool probes match exactly. |
| PPU INT32 / FP32 | Disproven on the tested external-RDMA path | Both submit successfully but write zero for nonzero expected output. |
| Kernel and stride 16 | Proven in isolation | Direct 16x16 maximum with stride 16 matches. |
| Padding 7 | Proven in isolation | 8x8 maximum with all four pads set to 7 matches all 15x15 outputs. |
| Kernel 16 combined with padding 7 | Not supported by the tested recipe | The isolated features pass, but the combined probe returned only the first input value. Do not assume all maximum field values compose. |
| Non-multiple channel count | Proven with packing | Three logical FP16 channels work while each pixel remains padded to a 16-byte atom. |
| External PPU RDMA (`FLYING_MODE=1`) | Proven | Every executable probe in `pooling.py` uses it. |
| PPU input from DPU (`FLYING_MODE=0`) | Register-defined, not silicon-proven here | A speculative combined submit would change the unit enable topology, so it was not used as proof. |
| Unpool / upsample | Not a PPU operation; PoC unresolved | `UNPOOLING_EN` is in DPU_RDMA. Several safe DPU_RDMA bypass probes returned zero, while the identical bypass without unpool worked; no working local capture was found. That is insufficient to disprove the hardware block. |
| General ArgMax, sort, fancy gather/scatter, WHERE, IEEE helpers, integer bitops | Not PPU capabilities | The complete PPU operation-mode register has only the three pool selectors plus local `INDEX_EN`; there are no selectors for these families. |

The register-field limits are kernel/stride encodings up to 16 and padding
encodings up to 7. RKNN recipes may impose smaller software limits; those are
not silicon limits. PPU specialization therefore replaces pooling code, but it
does not implement the unrelated workaround families in the last row.

The workspace copy of `ref/tinygrad` has no RK3588 renderer, so the quoted
renderer sizes (~5588 total, ~3.5k special cases, ~458 pool-index/unpool) cannot
be independently line-counted from this checkout. No line-savings estimate is
presented as measured fact here.
