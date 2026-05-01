# conv.py — Remaining Problems (Phased)

## ✅ Resolved

| Fix | Shapes affected | Verified |
|-----|----------------|----------|
| P6-A: ARGB_IN(9) excluded for ic=2 | (2,4,3x3,6x6), (2,4,2x2,5x5) | ✅ `--submit` |
| P6-B: Stride formulas tied to pixel mode | (1,6,3x3,5x7), (1,6,1x3,5x5) | ✅ `--submit` |
| P1: Group conv 1x1 — surf_stride uninit | (4,2,1x1,1x1,g=2), (4,4,1x1,1x1,g=2) | ✅ `--submit` |
| P1-P2: Group conv weight expansion + squash | All grouped convs (1x1 & 3x3) | ✅ `--submit` |
| P8: `default 1x1` false alarm | (2,2,1x1,4x4) | ✅ `--submit` |

## Active Phases

### Phase 3: Depthwise (all) — Under investigation

**Shapes:** dw 3x3 11x28 (previously WARN→PASS), dw 1x1 32ch (md=10)

**Root cause (register mismatch):** `weight_bytes_per_kernel = kh*kw*data_in_channel_aligned*2`. For depthwise, `kernel_channel` should be `weight_in_channels = 1`, but `data_in_channel_aligned = align_c = 32` (for 32ch) or 8 (for 3ch). The HW reads 32/8 channels per kernel, but only 1 has real data.

**Fix approach:** Expand depthwise weights to fill `data_in_channel_aligned` channels, with diagonal sparsity (weight[oc] at column oc). Correct packing layout verified: `wt_full[oc * slot_sz] = weight_ochw[oc]` padded to `align_c * kh * kw` elements.

**Blocking issue:** NPU enters bad state after broken submissions. Even simple 1x1 convs return all zeros after a depthwise failure. Requires kernel module reload (`sudo modprobe -r rockchip && sudo modprobe rockchip`) to recover. See Problem 7.

### Phase 4: ARGB_IN + non-square kernel partial output

**Status:** ❌ Not started. 6 shapes with ic=1 and non-square kernels produce ~14-33% output.

---

## Known Limitations (No Fix Planned)

- **Non-1x1 kernels produce partial/numerical output** — Expected HW limitation. Tracked as WARN, not FAIL.
- **Cross-process isolation (P7)** — One bad NPU submission corrupts state for ALL subsequent processes, even after `reset_npu()`. Requires kernel module reload to recover.
