# RK3588 PPU capability findings

`python examples/expt/pooling.py` is the executable decoded-register proof for the
silicon results in this table.

| Capability | Result | Evidence |
| --- | --- | --- |
| Average / maximum / minimum pool | Proven | Native PPU selector values 0/1/2 all produce the reference result. Minimum is tested directly, not synthesized with negation. |
| Global average | Proven without host repair | A 3x5 pool uses dimension-specific HLS FP17 reciprocals and directly returns the mean. The old `0x7800`/`0x7800` recipe is only correct for 2x2 and otherwise returns sum/4. |
| Global maximum | Proven | The kernel/stride-16 probe is also a direct 16x16 global maximum. |
| `INDEX_EN` | Proven, local and limited | It writes three row bits plus three column bits. A 16x16 sweep proves coordinates alias modulo 8: `(row & 7) << 3 \| (column & 7)`. With padding+stride, values remain exact but the lower output row retains a non-local vertical phase (`18` then `2`), so the simple local-coordinate model is valid only for the ordinary unpadded recipe. This is not general ArgMax. |
| PPU FP16 | Proven | Native maximum-pool probes match exactly. |
| PPU INT8 / INT16 maximum and minimum | Proven for signed values | Adversarial maximum vectors and negative minimum vectors pass exactly. The local TRM's RDMA storage widths 1 and 2 mean 8-bit and 16-bit and pair with same-format PPU processing precisions. |
| PPU INT8 / INT16 average | Proven with integer Q16 reciprocals | The FP16-encoded reciprocal recipe is inexact, but literal Q16 `round(65536/kernel_dimension)` passes exactly for 2x2, 3x3, and asymmetric 2x3 pools with positive and negative means in both INT8 and INT16. |
| Mixed PPU_RDMA width / PPU processing dtype | No captured numerical proof | The repeated vendor `1x1x32` token uses RDMA width 1 (8-bit) and PPU processing precision 0 (INT8), so it is same-format INT8—not INT16-to-INT8. `vendor_int8_1x1_c32` preserves that geometry; the former mixed-cast hypotheses were removed. |
| PPU INT32 / FP32 | Disproven on the tested external-RDMA path | Both submit successfully but write zero for nonzero expected output. |
| Kernel and stride 16 | Proven in isolation | Direct 16x16 maximum with stride 16 matches. |
| Padding 7 geometry | Proven in isolation | 8x8 maximum with all four pads set to 7 matches all 15x15 outputs for a positive input and zero pad fill. |
| FP16 `-inf` padding value (`0xFC00`) | Proven | `negative_max_padding_neg_inf` matches all 72 all-negative border and interior results, proving the low-halfword FP16 padding encoding. |
| `USE_CNT=0..7` | Does not select exclude-padding semantics in the tested recipe | Every encoding returns 2 for the padded 2x2-average probe. Exclude-padding would return 8, so the field is ignored here or requires another undocumented dependency. |
| Kernel 16 combined with padding 7 | Not supported by the tested recipe | The isolated features pass, but the combined probe returned only the first input value. Do not assume all maximum field values compose. |
| Non-multiple channel count | Proven with packing | Three logical FP16 channels work while each pixel remains padded to a 16-byte atom. |
| Multi-pass avg/max/min | Proven through NPU-written intermediates | Three separately fenced PPU submissions reduce 8x8→4x4→2x2→1x1 exactly. The second and third passes read the preceding PPU output BO directly; no host copy or repair is used. |
| Explicit PPU_RDMA line/surface strides | Proven | A 3x3x16 source with 80-byte lines and 256-byte surfaces pools exactly. `NOTCH_ADDR=2` corrupts later outputs, proving notch is not an additive line-pitch field; keep it zero for this external-RDMA recipe. |
| External PPU RDMA (`FLYING_MODE=1`) | Proven | Every executable probe in `pooling.py` uses it. |
| PPU input from DPU (`FLYING_MODE=0`) | Register-defined, not silicon-proven here | A speculative combined submit would change the unit enable topology, so it was not used as proof. |
| Unpool / upsample | Not a PPU operation; PoC unresolved | `UNPOOLING_EN` is in DPU_RDMA. Several safe DPU_RDMA bypass probes returned zero, while the identical bypass without unpool worked; no working local capture was found. That is insufficient to disprove the hardware block. |
| General ArgMax, sort, fancy gather/scatter, WHERE, IEEE helpers, integer bitops | Not PPU capabilities | The complete PPU operation-mode register has only the three pool selectors plus local `INDEX_EN`; there are no selectors for these families. |

`python3 examples/expt/pooling.py --validate` checks all 36 decoded precision, padding,
`USE_CNT`, address-stride, enable-mask, and multi-surface packing assumptions
without opening the NPU device.  Remaining cases marked `[isolated semantics probe]`
by `--list` are intentionally excluded from the default hardware matrix.
`--validate-multi-pass` additionally checks the nine chained-pass streams.

The register-field limits are kernel/stride encodings up to 16 and padding
encodings up to 7. RKNN recipes may impose smaller software limits; those are
not silicon limits. PPU specialization therefore replaces pooling code, but it
does not implement the unrelated workaround families in the last row.

The selected `~/tinygrad` branch is `rockchip-2608-ew-restart` at
`e3e93fa43e140570792e8c970d75188a55a56e3a`; its Rockchip renderer is 1,932
lines and its runtime is 287 lines.  It emits DPU+DPU_RDMA only.  Simple max
pool and padded average pool are already expected to use one PC-chain ioctl,
so adding PPU would principally replace expanded EW/gather task bodies and
scratch movement rather than automatically reduce submit count.  It still
cannot remove the renderer's unrelated sort, WHERE, arbitrary indexing, IEEE,
or integer-workaround families.
