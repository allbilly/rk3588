# conv.py — Remaining Problems

## Current Progress

- Standalone inline conv rawbuf and decoded PC-chain replay scripts are in place and do not load runtime dump files during normal replay.
- Working serialized replay coverage:
  - `experimental/conv_rawbuf.py --case 1`
  - `experimental/conv_rawbuf.py --case 4`
  - `experimental/conv_rawbuf_pcchain.py` (defaults to large case 4)
  - `experimental/conv_pcchain.py --case 1`
  - `experimental/conv_pcchain.py --case 4`
- `examples/conv.py --submit` and filtered `test/test_conv.py --submit` now take `/tmp/rk3588_npu_submit.lock` before touching `/dev/dri/card1`.
- The temporary decomposed spatial-conv path crashed the board on `test_sd_big_conv`; `examples/conv.py` no longer routes non-1x1 conv through 1x1 pointwise submits.
- `test_sd_big_conv` remains unresolved on target. Dry-run register generation now uses the direct conv path; target validation must still be done one case at a time because earlier decomposed submits triggered driver/device failures.
- Too much repeated `--submit` traffic can still crash the target even with the lock in place. The lock only prevents overlap; it does not make submit storms safe. Keep the submit path single-process and single-case, and avoid broad back-to-back target runs until the direct non-1x1 schedule is stable.

## Standalone Operator Scripts

These scripts are self-contained and do not import hex blobs from another file or read runtime dump files during normal replay:

- `experimental/add_pcchain.py`
- `experimental/add_pcchain_length.py`
- `experimental/add_rawbuf_pcchain.py`
- `experimental/conv_pcchain.py`
- `experimental/conv_rawbuf.py`
- `experimental/conv_rawbuf_big.py`
- `experimental/conv_rawbuf_pcchain.py`
- `experimental/conv_rawbuf_small.py`
- `experimental/gemm_pcchain.py`
- `experimental/gemm_rawbuf.py`
- `experimental/gemm_rawbuf_big.py`
- `experimental/gemm_rawbuf_medium.py`
- `experimental/gemm_rawbuf_pcchain.py`
- `experimental/gemm_rawbuf_small.py`
- `experimental/min_add_pcchain.py`
- `experimental/multicore_probe.py`
- `experimental/ops_rockchip_standalone.py`
- `experimental/elementwise_appleane.py`

Helper-backed or non-standalone scripts:

- `experimental/conv_rawbuf_common.py` is capture-only and still reads `gem*-dump` files when refreshing blobs.
- `experimental/min_add_small.py`, `experimental/min_add_medium.py`, `experimental/min_add_large.py`, `experimental/min_add_common.py`, `experimental/multicore_elementwise.py`, and `experimental/multicore_gemm.py` share helper modules instead of being fully standalone.
- `experimental/ops_rockchip.py` still shells out to `dump.py` in debug paths, so it is not a standalone replay script.

## Crash Note: `experimental/conv_rawbuf_small.py`

- The small rawbuf script is not a separate implementation; it calls `run_replay_main(1, ...)`, the same captured case used by `experimental/conv_rawbuf_pcchain.py` and `experimental/conv_pcchain.py`.
- The likely crash cause is the default raw patch path in `experimental/conv_rawbuf_common.py`: `patch_regcmd_buffer()` scans the entire captured `weight` GEM as `uint64` words and rewrites any qword whose middle 32 bits look like an old DMA address.
- That is unsafe for conv captures because the `weight` GEM contains more than just PC regcmd descriptors. It can also contain real convolution weight bytes or padding. An accidental address-looking bit pattern there can be rewritten into garbage, corrupting data that the NPU later consumes.
- The decoded path is safer: `decoded_regcmd_segments()` walks only the task descriptors, patches each descriptor's known regcmd body plus the four-qword PC tail, and leaves unrelated weight payload bytes unchanged.
- The other rawbuf/PC-chain scripts likely avoided the crash because they did not hit the same bad combination of capture contents and patch mode. `conv_pcchain.py --decoded` reconstructs only known regcmd segments, so unrelated weight bytes are not touched. `conv_rawbuf_big.py` uses a different capture case, so the blind scan may simply not encounter an address-looking payload word at a dangerous offset. `conv_rawbuf_pcchain.py` currently wraps the same case-1 entrypoint as `conv_rawbuf_small.py`; if run with the same default opaque patch path, it should be treated as equally risky unless the exact command used `--decoded` or otherwise restricted patching.
- This is consistent with the PC-chain/driver ground truth: RK/Rocket task descriptors use absolute DMA addresses for regcmd buffers, and PC-chain submits need the captured descriptor layout plus `RKNPU_JOB_PC|RKNPU_JOB_BLOCK|RKNPU_JOB_PINGPONG` (`0x7`). The fix should therefore avoid blind whole-buffer qword patching rather than changing the submit ABI first.

## Crash Note: parallel rawbuf wrappers, 2026-05-04

- The board crashed after `experimental/conv_rawbuf_small.py` and `experimental/conv_rawbuf_big.py` were launched in parallel.
- Treat this as an unsafe target race. The NPU path must be single-process and single-submit-step during bring-up.
- Do not use parallel tool execution for any command that opens `/dev/dri/card1`, resets the NPU, allocates NPU BOs, or submits to the NPU.
- `experimental/conv_rawbuf.py` now takes `/tmp/rk3588_npu_submit.lock` around the full rawbuf open/allocate/reset/submit path so two standalone rawbuf processes refuse to overlap.
- This lock change has now been target-run for standalone case 1 only; keep validating one command at a time.

How to reproduce the two-target crash pattern, if physical reset access is available:

```bash
# Terminal 1
python3 experimental/conv_rawbuf_small.py --mode core0 --alloc-mode official --flags 0x7 --verbose

# Terminal 2, started before terminal 1 has fully exited
python3 experimental/conv_rawbuf_big.py --mode core0 --alloc-mode official --flags 0x7 --verbose
```

Do not run this during normal bring-up. A close equivalent happened accidentally when both commands were launched through a parallel tool call. The likely race window includes `/dev/dri/card1` open, BO allocation/mapping, `reset_npu()`, and `DRM_IOCTL_RKNPU_SUBMIT`, not just the final submit ioctl.

DeepWiki check on `allbilly/rknpu_driver`:

- The downstream driver has per-core job queues and internal locks for IRQ/job queue, power, reset, and IOMMU domain state.
- Submit is scheduled through per-core queues; it is not a single global userspace-facing device lock.
- Reset paths are separately locked, but a reset from one process can still be hostile to another process that is allocating/submitting/waiting.
- Therefore user-space replay tools should serialize the whole NPU target section, not just individual ioctl calls.

Suggested fixes:

- Keep `/tmp/rk3588_npu_submit.lock` around open/allocate/map/reset/submit/readback for every experimental script that touches `/dev/dri/card1`.
- Add the same lock to `examples/conv.py` before broad `test/test_conv.py --submit` runs. This is now implemented.
- Avoid `multi_tool_use.parallel` for target NPU commands entirely.
- Kernel-side hardening would be a global device/job mutex around reset-vs-submit and stricter validation of task ranges/regcmd DMA before hardware programming, but local bring-up should not depend on a kernel patch.

Suggested fix direction:

- Make decoded segment patching the default for conv rawbuf replay, or delete/disable the blind `patch_regcmd_buffer()` path for conv captures. This is now implemented in `experimental/conv_rawbuf_common.py`; the old whole-GEM patch path requires `--raw-blind-patch`.
- Patch only qwords reachable from the captured `rknpu_task.regcmd_addr`/`regcfg_amount` ranges, including the four PC-tail qwords after each body.
- Before any NPU submit, dump the reconstructed register stream and compare it carefully against a known-good `~/npu/ops_reg`/RKNN capture for the same shape. Build a table by task and by `(target, register)` showing expected value, reconstructed value, and whether a difference is only a DMA relocation. Do not submit if non-DMA register values, task counts, `regcfg_amount`, PC tails, or descriptor addresses differ unexpectedly.
- Keep submit flags at `0x7` and keep `mode=core0` while validating; only revisit `core_mask`/`subcore_task` layout after the decoded single-core replay is stable.
- Add a dry/verbose checker that prints every patched qword offset and rejects patches outside known regcmd segments before allowing `/dev/dri/card1` submit. This is now available as `--preflight-only --verbose`.

## Tiny Safe Experiment Plan

Do not jump directly to conv multicore. The current board has already shown that invalid task ranges or bad regcmd DMA pointers can panic the kernel. Use this order:

Reference examples to keep open while comparing:

- `experimental/add_pcchain.py` and `experimental/add_rawbuf_pcchain.py` for decoded/raw ADD PC-chain layout.
- `experimental/gemm_pcchain.py` for a decoded RKNN-style multi-task PC-chain that matches a captured GEMM stream.
- `examples/pool_pcchain.py` for a compact working PC-tail example.
- `examples/pool_multicore.py` for independent-task multicore shape. Treat this as a scheduling/layout reference only; do not assume conv can use the same register body.
- `experimental/conv_rawbuf_common.py` / `experimental/conv_pcchain.py` for the captured RKNN conv replay path.

1. **Dry-check known working local examples, no NPU submit**
   - `python3 examples/pool_pcchain.py`
   - `python3 examples/pool_multicore.py`
   - These are useful references because `pool_pcchain.py` shows a working PC tail layout and `pool_multicore.py` shows a working independent-task multicore shape for PPU.
   - 2026-05-04 dry checks pass: `POOL PCCHAIN DRY RUN PASS` and `POOL MULTICORE DRY RUN PASS`.

2. **Capture RKNN conv and inspect only, no raw replay submit, only when refreshing inline blobs**
   - Use `experimental/conv_rawbuf_common.py` helpers for this capture-refresh path; the normal `conv_rawbuf_small.py` entrypoint is now standalone and does not capture.
   - Required pass criteria:
     - decoded identity mismatches are `0`
     - all task `regcmd_addr` values land inside the captured weight GEM
     - each segment includes the body plus the four-qword PC tail
     - blind outside-segment candidates are `0`; if nonzero, do not use `--raw-blind-patch`

3. **Repeat preflight for the second conv capture case, only when refreshing inline blobs**
   - Use the same explicit capture-refresh path for case 4.
   - Same pass criteria as small. This checks that case-specific capture layout is stable before any raw submit.

4. **Run one decoded core-0 replay only**
   - `python3 experimental/conv_rawbuf_small.py --mode core0 --alloc-mode official --flags 0x7 --verbose`
   - This now uses decoded patching by default. Stop here if it fails, times out, or writes a nonzero/NaN output.

5. **Only after small replay passes, run the decoded PC-chain wrapper**
   - `python3 experimental/conv_pcchain.py --mode core0 --alloc-mode official --flags 0x7 --verbose`
   - Expected descriptor shape is still the captured RKNN 11-task stream, not a guessed local descriptor layout.

6. **Only after both case-1 decoded replays pass, try case 2**
   - `python3 experimental/conv_rawbuf_big.py --mode core0 --alloc-mode official --flags 0x7 --verbose`

7. **No conv multicore until decoded single-core replay is repeatable**
   - Do not use `core_mask=0x7`, nonzero cores, or `subcore_task[2..4]` for conv until the captured single-core stream can be replayed repeatedly without timeout.
   - When multicore is revisited, start from `examples/pool_multicore.py` as the working independent-task pattern and from `allbilly/rknpu_driver`'s 3-core indexing rule as a driver constraint, not from a guessed split.

Preflight result from 2026-05-04 for case 1:

```text
tasks=11
decoded identity mismatches=0
blind address-like qwords: inside_segments=33 outside_segments=0
CONV PREFLIGHT PASS
```

Preflight result from 2026-05-04 for case 4:

```text
tasks=11
decoded identity mismatches=0
blind address-like qwords: inside_segments=33 outside_segments=0
CONV PREFLIGHT PASS
```

Step result from 2026-05-04 for case 1 decoded core-0 rawbuf replay:

```text
python3 experimental/conv_rawbuf_small.py --mode core0 --alloc-mode official --flags 0x7 --verbose
case=1 mode=core0 tasks=11 submit ret=0
output offset=0x80 size=0x300 max_abs=0.000000
conv rawbuf small PASS
```

Standalone rawbuf update from 2026-05-04:

- Added `experimental/conv_rawbuf.py` as a standalone inline replay script.
- `experimental/conv_rawbuf_small.py` and `experimental/conv_rawbuf_big.py` now use inline capture blobs and do not invoke GDB, `dump.py`, `conv2d_dump_regs`, or load `gem*-dump` files at runtime.
- `experimental/conv_rawbuf_pcchain.py` now uses the standalone inline path and defaults to the large capture case (`--case 4`).
- `experimental/conv_pcchain.py` is now the decoded-register PC-chain entrypoint. It does not load runtime dump files and does not depend on `conv_rawbuf_common.py`; the entrypoint reconstructs register commands through decoded `(target, register, value)` handling instead of submitting an opaque regcmd file.

Step result from 2026-05-04 for standalone inline case 1:

```text
python3 experimental/conv_rawbuf.py --case 1 --mode core0 --alloc-mode official --flags 0x7 --verbose
case=1 mode=core0 tasks=11 submit ret=0
output offset=0x80 size=0x300 max_abs=0.000000
conv rawbuf inline case 1 PASS
```

Step result from 2026-05-04 for standalone inline case 1 after adding the submit lock:

```text
python3 experimental/conv_rawbuf.py --case 1 --mode core0 --alloc-mode official --flags 0x7 --verbose
case=1 mode=core0 tasks=11 submit ret=0
output offset=0x80 size=0x300 max_abs=0.000000
conv rawbuf inline case 1 PASS
```

Step result from 2026-05-04 for standalone inline case 4 after adding the submit lock:

```text
python3 experimental/conv_rawbuf.py --case 4 --mode core0 --alloc-mode official --flags 0x7 --verbose
case=4 mode=core0 tasks=11 submit ret=0
output offset=0x140 size=0x900 max_abs=0.000000
conv rawbuf inline case 4 PASS
```

Step result from 2026-05-04 for standalone inline case 4:

```text
python3 experimental/conv_rawbuf.py --case 4 --mode core0 --alloc-mode official --flags 0x7 --verbose
case=4 mode=core0 tasks=11 submit ret=0
output offset=0x140 size=0x900 max_abs=0.000000
conv rawbuf inline case 4 PASS
```

Step result from 2026-05-04 for case 1 decoded core-0 PC-chain wrapper:

```text
python3 experimental/conv_pcchain.py --mode core0 --alloc-mode official --flags 0x7 --verbose
case=1 mode=core0 tasks=11 submit ret=0
output offset=0x80 size=0x300 max_abs=0.000000
conv pcchain decoded PASS
```

Step result from 2026-05-04 for standalone decoded PC-chain wrapper case 1:

```text
python3 experimental/conv_pcchain.py --case 1 --mode core0 --alloc-mode official --flags 0x7 --verbose
case=1 mode=core0 tasks=11 submit ret=0
output offset=0x80 size=0x300 max_abs=0.000000
conv pcchain decoded case 1 PASS
```

Step result from 2026-05-04 for standalone decoded PC-chain wrapper case 4:

```text
python3 experimental/conv_pcchain.py --case 4 --mode core0 --alloc-mode official --flags 0x7 --verbose
case=4 mode=core0 tasks=11 submit ret=0
output offset=0x140 size=0x900 max_abs=0.000000
conv pcchain decoded case 4 PASS
```

Step result from 2026-05-04 for `test/test_conv.py` dry-run:

```text
python3 test/test_conv.py
ALL TEST CASES PASS
```

This is a non-target register-generation check only; it does not submit to the NPU.

Harness safety update from 2026-05-04:

- `examples/conv.py` now takes `/tmp/rk3588_npu_submit.lock` before opening `/dev/dri/card1` in `--submit` mode.
- `test/test_conv.py` now supports `--filter <substring>` so target validation can run one shape at a time instead of jumping directly to the whole harness.
- `python3 test/test_conv.py --filter 'default 1x1'` passed as a non-target dry-run.

Step result from 2026-05-04 for filtered target harness `default 1x1`:

```text
python3 test/test_conv.py --submit --filter 'default 1x1'
default 1x1: SUBMIT ret=0
PASS
ALL TEST CASES PASS
```

Step result from 2026-05-04 for filtered target harness `ic=1 oc=6 1x1`:

```text
python3 test/test_conv.py --submit --filter 'ic=1 oc=6 1x1'
ic=1 oc=6 1x1: SUBMIT ret=0
PASS
ALL TEST CASES PASS
```

Step result from 2026-05-04 for filtered target harness `ic=8 oc=8 1x1 5x5`:

```text
python3 test/test_conv.py --submit --filter 'ic=8 oc=8 1x1 5x5'
ic=8 oc=8 1x1 5x5: SUBMIT ret=0
SUBMIT ret=0
PASS
ALL TEST CASES PASS
```

Step result from 2026-05-04 for filtered target harness `simple 3x3 9x9`:

```text
python3 test/test_conv.py --submit --filter 'simple 3x3 9x9'
simple 3x3 9x9: PASS
ALL TEST CASES PASS
```

Step result from 2026-05-04 for filtered target harness `test_large_input_conv2d`:

```text
python3 test/test_conv.py --submit --filter 'test_large_input_conv2d'
test_ops test_large_input_conv2d: PASS
ALL TEST CASES PASS
```

Regression result from 2026-05-04 after first spatial batching attempt:

```text
python3 test/test_conv.py --submit --filter 'test_large_input_conv2d'
test_ops test_large_input_conv2d: FAIL (non-1x1 partial: nz=22679/22678)
SOME TEST CASES FAILED - see above
```

Interpretation: batching input channels changed the numerical/output pattern for a shape that previously passed. A later attempt to gate batching to large input-channel cases still crashed on `test_sd_big_conv`, so the batching path was fully reverted.

Crash note from 2026-05-04 for `test_sd_big_conv` after narrowed batching:

```text
timeout 240 python3 test/test_conv.py --submit --filter 'test_sd_big_conv'
test_ops test_sd_big_conv:
<board crashed before pass/fail output>
```

Conclusion from the old fallback: decomposed 1x1 spatial conv is unsafe for large shapes. The fix direction is to use a direct `ops_reg`-style non-1x1 conv register schedule, not to batch or retry decomposed pointwise submits.

Step result from 2026-05-04 for filtered target harness `test_sd_big_conv`:

```text
timeout 180 python3 test/test_conv.py --submit --filter 'test_sd_big_conv'
test_ops test_sd_big_conv:
exit code 124
```

No correctness failure was reported before timeout. Treat this as the next conv performance/tiling problem, not as a register mismatch yet.

Reverted spatial decomposition attempt from 2026-05-04:

- `_run_conv2d_spatial_decomposed()` briefly batched up to 4 input channels per decomposed 1x1 submit and tried full-width row tiles.
- That attempt was reverted after `test_large_input_conv2d` produced a correctness regression and `test_sd_big_conv` crashed the board.
- `examples/conv.py` now removes the decomposed fallback and emits direct non-1x1 conv registers for dry-run.
- `python3 test/test_conv.py --filter 'test_large_input_conv2d'` passes as a non-target dry-run and shows one direct `kernel=(5,2)` register schedule.
- Re-run target validation one case at a time; do not re-run `test_sd_big_conv` until the submit-count problem has a new plan.

Historical step result from 2026-05-04 for filtered target harness `test_large_input_conv2d` after reverting batching, before removing the decomposed fallback:

```text
timeout 120 python3 test/test_conv.py --submit --filter 'test_large_input_conv2d'
test_ops test_large_input_conv2d: PASS
ALL TEST CASES PASS
```

Direct-conv target result from 2026-05-04 after removing the 1x1 decomposition fallback:

```text
timeout 120 python3 test/test_conv.py --submit --filter 'test_large_input_conv2d'
test_ops test_large_input_conv2d: SUBMIT ret=0
FAIL (non-1x1 partial: nz=18143/22678)
SOME TEST CASES FAILED - see above
```

The same exact shape also fails in the C `ops_reg` CLI built from `experimental/main.c`:

```text
timeout 120 ./build/ops_reg conv2d 1 16 64 64 6 16 5 2 1
conv2d_cli_b1_c16_h64_w64_oc6_wic16_k5x2_g1: matches CPU -> NO (mismatches=22671)
```

Small direct spatial target cases still pass:

```text
timeout 60 python3 test/test_conv.py --submit --filter 'ic=1 oc=6 3x3'
ic=1 oc=6 3x3: SUBMIT ret=0
PASS
ALL TEST CASES PASS

timeout 60 python3 test/test_conv.py --submit --filter 'simple 3x3 9x9'
simple 3x3 9x9: SUBMIT ret=0
PASS
ALL TEST CASES PASS
```

Interpretation: the large `16->6, 5x2, 64x64` direct schedule is not validated by current `ops_reg`; it is not merely a Python transcription issue. The next reference should be an RKNN capture for this exact shape or a targeted register comparison against a known-good vendor run.

Crash-analysis note from 2026-05-04:

- The board rebooted at `2026-05-04 19:52 CST` after a later target attempt. Current boot logs show an unclean journal, but the actual crash stack is not visible without root-only `/sys/fs/pstore` access.
- Accessible previous-boot journal entries show the same failure family: repeated `RKNPU: soft reset, num: 6`, IOMMU page faults reading IOVA `0x0`, `RKNPU: job timeout`, and `RKNPU: job commit failed` / `job abort ret: -22`.
- Before the direct-conv fix, `test_large_input_conv2d` shape `(in=16, out=6, kernel=5x2, input=64x64)` entered `_run_conv2d_spatial_decomposed()` and expanded to one-input-channel 1x1 pointwise submits:

```text
in_channels * kernel_h * kernel_w * output_row_tiles * output_col_tiles * output_channel_chunks
= 16 * 5 * 2 * ceil(60/16) * ceil(63/16) * ceil(6/16)
= 16 * 5 * 2 * 4 * 4 * 1
= 2560 NPU submits
```

- Each `_npu_submit()` called `reset_npu(fd)` before programming the command buffer, so even the "passing" large-input case was a reset/submit stress test, not one normal conv submission.
- The old decomposed `test_sd_big_conv` expansion was much worse:

```text
256 * 3 * 3 * ceil(62/16) * ceil(62/16) * ceil(512/16)
= 1,179,648 NPU submits
```

- Do not target-run the old decomposed `test_sd_big_conv` path. The direct non-1x1 register schedule must be validated cautiously with one target command at a time.

Step result from 2026-05-04 for filtered target harness `simple 3x3 9x9` before the unsafe batching experiment:

```text
python3 test/test_conv.py --submit --filter 'simple 3x3 9x9'
simple 3x3 9x9: PASS
ALL TEST CASES PASS
```

Step result from 2026-05-04 for `examples/conv.py --submit` smoke:

```text
python3 examples/conv.py --submit
SUBMIT ret=0
CONV 2x2x1x1 -> 2x4x4: PASS (max_diff=0.0010)
```

Step result from 2026-05-04 for case 4 decoded core-0 rawbuf replay:

```text
python3 experimental/conv_rawbuf_big.py --mode core0 --alloc-mode official --flags 0x7 --verbose
case=4 mode=core0 tasks=11 submit ret=0
output offset=0x140 size=0x900 max_abs=0.000000
conv rawbuf big PASS
```

Step result from 2026-05-04 for case 1 decoded core-0 rawbuf PC-chain wrapper:

```text
python3 experimental/conv_rawbuf_pcchain.py --mode core0 --alloc-mode official --flags 0x7 --verbose
case=1 mode=core0 tasks=11 submit ret=0
output offset=0x80 size=0x300 max_abs=0.000000
conv rawbuf pcchain PASS
```

Step result from 2026-05-04 for standalone large rawbuf PC-chain wrapper after adding the submit lock:

```text
python3 experimental/conv_rawbuf_pcchain.py --mode core0 --alloc-mode official --flags 0x7 --verbose
case=4 mode=core0 tasks=11 submit ret=0
output offset=0x140 size=0x900 max_abs=0.000000
conv rawbuf pcchain large PASS
```

## Fix Log

- **2026-05-03: no-skip/no-warning conv harness cleanup**
  - Removed the unsupported-case footer from `test/test_conv.py` so the harness does not print `SKIP` or warning-style lines.
  - Do not use a CPU fallback for large spatial conv. The large-shape fix must use NPU execution, preferably the RKNN-style conv PC-chain path proven by `experimental/conv_pcchain.py`.

- **2026-05-03: captured conv rawbuf/PC-chain replay**
  - Added `experimental/conv_rawbuf_small.py`, `experimental/conv_rawbuf_big.py`, `experimental/conv_rawbuf_pcchain.py`, and `experimental/conv_pcchain.py`.
  - The scripts capture RKNN conv task/regcmd buffers from `~/npu/ops_rknn/conv2d_dump_regs`, patch DMA addresses, and replay the captured 11-task stream directly through `/dev/dri/card1`.
  - `conv_pcchain.py` rebuilds regcmd qwords from decoded `(target, register, value)` triples before submit by default.
  - Required submit flags for captured conv streams are `0x7` (`PC|BLOCK|PINGPONG`); omitting `BLOCK` caused timeouts.

- **2026-05-03: large spatial conv fallback now tiles output rows**
  - Failure: the decomposed non-1x1 path produced partial outputs for `test_ops test_sd_big_conv` and `test_ops test_large_input_conv2d`.
  - Cause: one 1x1 submit could only deliver the first ~48 output rows in this flattened schedule, leaving later rows zero.
  - Historical fix: `_run_conv2d_spatial_decomposed()` sliced output height into 32-row tiles and accumulated each tile with separate 1x1 submits.
  - Superseded: `examples/conv.py` no longer uses this decomposed fallback for non-1x1 conv.

- **2026-05-03: 1x1 packed output sizing for large decomposed spatial convs**
  - Failure: `test_ops test_sd_big_conv` and `test_ops test_large_input_conv2d` raised `buffer is smaller than requested size` in `test/test_conv.py --submit`.
  - Cause: `_packed_sizes()` sized all output buffers as `out_h * out_width_stride * align_out_c`, but 1x1 schedules flatten the output surface to `H=1, W=out_h*out_w`; `_unpack_output()` already decodes that flattened layout. The inflated size exceeded the fixed 4 MiB output mmap for large spatial convolutions decomposed into 1x1 submits.
  - Fix: make `_packed_sizes()` use one output row for `kernel_h == kernel_w == 1`, matching `experimental/rknnops.h::float16_conv2d` packed-output sizing and the existing unpack path.

## Remaining Cleanup Targets

- **Weight layout dispatch (`_KH_MAJOR_SHAPES`)**: still a shape table keyed by `(out_c, in_c, kh, kw)` with group filters. It is contained, but the underlying hardware selector is not decoded yet.
- **Raw register values in `build_conv2d_regs()`**: many register bitfields are still emitted as raw constants. Replacing them should be done against `experimental/rockchip.py`/upstream register definitions to avoid changing working programming.
- **Script entrypoint hard-codes one smoke shape**: useful for quick manual testing but not representative. `test/test_conv.py --submit` is the real coverage.

## Known Limitations

- **Cross-process isolation (P7)**: one bad NPU submission can corrupt state for later processes, even after `reset_npu()`. Recovery still requires reloading the kernel module.
- **Direct spatial convolution target status**: `examples/conv.py` now emits the direct non-1x1 register schedule instead of 1x1 decomposition, but target correctness still needs one-at-a-time validation because previous fallback submits crashed the board.
