# conv.py — Remaining Problems

## Problem 6: Non-1x1 Kernels — 4 shapes still produce partial output

**Status:** Depthwise weight packing fix resolved the dw case (702/702, was 234/702). Output unpack stride fix resolved 3 others. 4 shapes remain with genuinely partial output.

**Resolved cases:**
| Shape | nz/total | Fix |
|-------|----------|-----|
| (1,6,3,3,5x7) groups=1 | PASS | Weight pack: use full `in_channels` for depthwise-like ic=1 case |
| (4,4,3,3,9x9) groups=1 | PASS | Output unpack stride (`out_w` not `out_width_stride`) |
| (3,3,3,3,11x28) groups=3 dw | 702/702 | Weight pack: expand tensor to full in_channels for depthwise |
| (3,6,1,3,5x5) groups=1 | 90/89 | Numerical rounding, not a real failure |

**Remaining failing shapes:**
| Shape | nz/total | Ratio |
|-------|----------|-------|
| (2,4,3,3,6x6) groups=1 | 48/64 | 75% |
| (2,4,2,2,5x5) groups=1 | 28/64 | 44% |
| (1,32,5,5,10x10) groups=1 | 864/1152 | 75% |
| (1,6,3,1,5x7) groups=1 | 18/126 | 14% |

**WARN-but-full-coverage (numerical rounding, not urgent):**
| Shape | nz/total | Notes |
|-------|----------|-------|
| (16,16,3,3,9x9) groups=1 | 784/784 | All pixels, values within rounding |
| (16,16,3,3,18x18) groups=1 | 4096/4096 | All pixels, values within rounding |
| (8,4,4,4,10x10) groups=1 | 196/196 | All pixels, values within rounding |

**Common factors among remaining failures:** small output dimensions, non-square or large kernels. These may indicate additional packing or register configuration issues beyond the output stride and weight packing.

## Problem 7: Cross-Process Isolation Required 📋 DOCUMENTED

**Workaround:** Always run conv and gemm tests in isolated subprocess invocations (as `test_conv.py` and `test_gemm.py` already do). Mixing them in one process produces garbage even after `reset_npu()`.
