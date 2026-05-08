# conv.py — Remaining Problems

## Known Limitations

- **`test_sd_big_conv` timeout on target**: large direct spatial conv schedule times out (>180s) without a correctness failure. Next step is RKNN capture for the exact shape or a targeted register comparison against a known-good vendor run.
- **Large direct spatial conv (`16->6, 5x2, 64x64`) fails numerically**: matches CPU output partially (`nz=18143/22678`). Also fails in the C `ops_reg` CLI, so it is not a Python transcription issue.
- **Cross-process isolation**: one bad NPU submission can corrupt state for later processes even after `reset_npu()`. Recovery still requires reloading the kernel module.

## Crash Notes (Still Relevant)

### Crash: blind qword patching in conv rawbuf captures

`experimental/conv_rawbuf_common.py`'s `patch_regcmd_buffer()` scans the entire captured `weight` GEM as `uint64` words and rewrites any qword whose middle 32 bits look like an old DMA address. This is unsafe because the `weight` GEM contains more than just PC regcmd descriptors — it also contains real convolution weight bytes or padding. An accidental address-looking bit pattern there can be rewritten into garbage, corrupting data that the NPU later consumes.

The decoded path is safer: `decoded_regcmd_segments()` walks only the task descriptors, patches each descriptor's known regcmd body plus the four-qword PC tail, and leaves unrelated weight payload bytes unchanged.

**Fix**: decoded segment patching is now the default. The old whole-GEM patch path requires `--raw-blind-patch`.

### Crash: parallel NPU submits

The board crashed after `experimental/conv_rawbuf_small.py` and `experimental/conv_rawbuf_big.py` were launched in parallel. The NPU path must be single-process and single-submit-step during bring-up.

**Root cause**: the downstream driver has per-core job queues and internal locks for IRQ/job queue, power, reset, and IOMMU domain state, but reset-vs-submit from separate processes is not globally serialized at the userspace device level.

**Fix**: `/tmp/rk3588_npu_submit.lock` around the full open/allocate/map/reset/submit/readback path in every experimental script that touches `/dev/dri/card1`. However, the lock only prevents overlap; it does not make submit storms safe. Avoid parallel tool execution for any command that opens `/dev/dri/card1`, resets the NPU, allocates NPU BOs, or submits to the NPU.

### Crash: decomposed 1x1 spatial conv submit storms

The old `_run_conv2d_spatial_decomposed()` path expanded non-1x1 conv into one-input-channel 1x1 pointwise submits. For `test_large_input_conv2d` shape `(in=16, out=6, kernel=5x2, input=64x64)`:
- `16 * 5 * 2 * 4 * 4 * 1 = 2560 NPU submits`
- Each submit called `reset_npu(fd)`, making even a "passing" run a reset/submit stress test.

For `test_sd_big_conv`: `256 * 3 * 3 * 4 * 4 * 32 = 1,179,648 NPU submits` — crashed the board.

**Fix**: `examples/conv.py` no longer routes non-1x1 conv through 1x1 pointwise submits. Direct non-1x1 register schedule is now emitted for dry-run. Do not re-run `test_sd_big_conv` until the submit-count problem has a new plan.

