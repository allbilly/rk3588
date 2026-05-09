# conv_gemm.py problem

## Symptom
Combined default run (`python examples/conv_gemm.py`) fails on certain conv shapes after GEMM phase.

- conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1 (32x112x112 -> 64x112x112, 1x1 pointwise)
- Runs fine in conv-only mode (`python examples/conv_gemm.py conv`).
- Runs fine in conv.py standalone.
- Fails only when GEMM runs first in the same process: max_diff=58.1240

## Debugging status
- `DATA_CUBE_NOTCH=0` fix resolved the first spatial conv failure (16x18x18 k3x3) after GEMM, but the 32x112x112 1x1 still fails.
- Conv-only test on this shape passes (max_diff=0.0146).
- Conv-after-GEMM test fails (max_diff=58.1240).
- Suspected stale DPU register state from GEMM that isn't cleared by `npu_reset` or the conv register list.

## Today (2026-05-09) last test
- Running conv after a 518x518x518 GEMM + a warmup conv (16x18x18 k3x3) hit `OSError: [Errno 22] Invalid argument` in `npu_submit`.
- Kernel log shows CMA allocation failures and job timeout → likely memory exhaustion from repeated test runs. Reboot needed.

## Resolution
- No reboot was needed for the functional fix.
- The failure starts after multi-task GEMM PC-chain submits (`512x512x512` is enough to reproduce).
- Reopening `/dev/dri/card1` and reallocating the NPU BO context between the GEMM and CONV phases clears the stale state.
- `examples/conv_gemm.py` now calls `reopen_npu_context()` after GEMM tests when the CONV phase will run.
- Verified `python examples/conv_gemm.py`: all GEMM and CONV tests pass, including `conv2d_cc_b1_c32_h112_w112_oc64_wic32_k1x1_g1` with `max_diff=0.0146`.

## Next steps
1. Optional: keep narrowing the exact leaked hardware/driver state so the context reopen can be replaced by register-level cleanup.
