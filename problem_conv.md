# conv.py — Remaining Problems

## Remaining Cleanup Targets

- **Weight layout dispatch (`_KH_MAJOR_SHAPES`)**: still a shape table keyed by `(out_c, in_c, kh, kw)` with group filters. It is contained, but the underlying hardware selector is not decoded yet.
- **Raw register values in `build_conv2d_regs()`**: many register bitfields are still emitted as raw constants. Replacing them should be done against `experimental/rockchip.py`/upstream register definitions to avoid changing working programming.
- **Script entrypoint hard-codes one smoke shape**: useful for quick manual testing but not representative. `test/test_conv.py --submit` is the real coverage.

## Known Limitations

- **Cross-process isolation (P7)**: one bad NPU submission can corrupt state for later processes, even after `reset_npu()`. Recovery still requires reloading the kernel module.
- **Direct spatial convolution**: raw non-1x1 programming can still produce partial or shape-dependent output, so the tested path decomposes spatial kernels into 1x1 NPU submits.
