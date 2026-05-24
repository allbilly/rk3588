# Current progress: `examples/conv_tiles.py` RKNN/RKNPU CONV bring-up

Date: 2026-05-24

This is a detailed handoff of the current RK3588 NPU CONV work. It is written
to be usable without relying on chat history.

## Objective

Current working objective:

```text
Create/promote examples/conv_tiles.py as the top-level downstream RKNPU version
of the kernel_6_18 CONV work, using decoded registers rather than opaque blobs,
and continue the direct-spatial/RKNN-captured sparse task path until the next
real blocker is understood.
```

Relevant project rules:

- Do not offload op work to CPU/GPU unless explicitly requested.
- Do not edit files in `examples/kernel_6_18/` unless explicitly requested.
- Do not kill long-running NPU processes.
- Be very careful around `npu_submit`, `task_count`, `regcmd_addr`,
  `regcfg_amount`, and `enable_mask`.
- For all `ops_rknn` work, source is under `~/npu/ops_rknn`.
- Use `python3 examples/simple_add.py` as the quick NPU health/recovery check.

## Short answer to the latest questions

### Why MMIO read was needed

MMIO read is not needed to compute the conv output. It was requested as evidence
for the current C64/H56 blocker.

The raw C64/H56 sparse path returns a correct conv result, but the next PC job
(`simple_add.py`) times out. Userspace-visible RKNN metadata already matches:
task bytes, live regcmd window, active offset, allocation sizes, ACTION sequence,
submit metadata, and MEM_SYNC sequence. Because those all match, the remaining
suspect is hidden hardware/driver state after completion: PC/DPU/interrupt/task
status that is not visible from normal userspace ioctls.

The specific state we want to compare is:

- after an RKNN C64/H56 run that does not poison the next job;
- after the raw `conv_tiles.py` C64/H56 run that computes correctly but poisons
  the next job;
- before submitting the next `simple_add.py`.

Non-root sysfs is not enough for that. It only shows runtime/devfreq state, not
per-core PC/DPU/IRQ status. `/dev/mem` MMIO access is currently blocked:

```text
MMIO_SNAPSHOT baseline_after_recovery
  MMIO_UNAVAILABLE device=/dev/mem errno=13 error=Permission denied
```

So MMIO is a diagnostic path, not part of the conv implementation.

### Current blocker

The blocker is not that one conv shape timed out during its own run. The C64/H56
conv itself completed and produced correct output.

The blocker is:

```text
C64/H56 raw sparse CONV succeeds, but it leaves the NPU/driver state polluted so
the next PC job times out until the downstream RKNPU driver timeout path performs
a soft reset.
```

Observed next-job timeout:

```text
post simple_add TimeoutError: [Errno 110] Connection timed out

RKNPU: job: ..., mask: 0x1, ... timeout: 6000000us
RKNPU: failed to wait job, task counter: 0, flags: 0x5
RKNPU: job timeout, flags: 0x0:
RKNPU:     core 0 irq status: 0x0, raw status: 0xc0000000, require mask: 0x300, task counter: 0x0
RKNPU: soft reset, num: 6
```

After that timeout/soft-reset, `python3 examples/simple_add.py` passes again.

### Did we rerun RKNN with RKNPU and capture again

Yes. The refreshed RKNN/RKNPU capture set includes:

```text
experimental/rknn/capture_rknpu_ioctl_sweep_action_readonly_20260524_135914.log
experimental/rknn/capture_rknpu_ioctl_c64_h56_readonly_20260524_133736.log
experimental/rknn/capture_rknpu_submit_dump_gems_c64_h56_20260524_133709.log
experimental/rknn/capture_rknpu_submit_dump_gems_c64_h56_live_regcmd_1168_20260524_133725.log
```

The C64/H56 RKNN capture showed the schedule-specific runtime profile now
modeled in `conv_tiles.py`:

- task BO size: `560`
- regcmd BO size: `28352`
- active regcmd offset: `0x4800`
- active regcmd bytes: `9344`
- weight/internal BO size: `1204224`
- input BO size: `401408`
- output BO size: `802816`
- task object: 14 records
- live active regcmd window: 1168 qwords
- one extra input `MEM_SYNC FROM_DEVICE` before submit
- four output `MEM_SYNC FROM_DEVICE` calls after submit

The offline verifier confirms this capture is modeled:

```text
offline verifier PASS
PASS C64/H56 RKNN-mode runtime profile
PASS C64/H56 RKNN-mode sync profile
PASS unpromoted C64/H56 live regcmd evidence
```

### Is `conv_tiles.py` hardcoding tile shapes from RKNN or using formulas

It is mixed.

The direct-spatial schedules are not a complete formulaic tiler yet. They are
RKNN-captured templates plus decoded register math.

Captured/template-driven pieces:

- record amounts;
- record count/topology;
- PC links;
- PC root/separator layout;
- per-record PC core assignment for promoted special cases;
- selected CBUF/grain overrides;
- active regcmd offsets/windows;
- RKNN-mode BO sizes for captured shape profiles;
- RKNN-mode MEM_SYNC sequence variants.

Formula/math-derived pieces:

- many register field encodings;
- DMA addresses and offsets from BO base plus tensor layout;
- input/output tensor placement;
- NC1HWC2/flat/direct-spatial packing logic;
- output offset computation;
- weight/input/output buffer population;
- register qword generation from descriptor objects;
- normalization/comparison of live RKNN regcmd windows against generated qwords.

So the honest status is:

```text
conv_tiles.py is not yet a general RKNN tiling compiler. Promoted shapes are
template schedules captured from RKNN, with register bodies and buffer layout
computed by decoded math. A shape is promoted only after byte-level offline
equality against RKNN capture plus one gated hardware probe bracketed by
simple_add before and after.
```

## Main files and roles

### `examples/conv_tiles.py`

Top-level downstream RKNPU CONV experiment.

Current behavior:

- Based on `examples/kernel_6_18/conv.py`, but adapted to downstream Rockchip
  `rknpu` ABI.
- Uses `/dev/dri/card1`.
- Uses `DRM_IOCTL_RKNPU_*`.
- Defines downstream structs:
  - `rknpu_mem_create`
  - `rknpu_mem_map`
  - `rknpu_mem_sync`
  - `rknpu_submit`
  - `rknpu_subcore_task`
  - `struct_rknpu_task`
- Uses downstream `op_idx=1`.
- Normal register-list task descriptors use downstream RKNPU PC-tail semantics:

```text
npu_tasks[idx].regcfg_amount = len(regs)
```

Important runtime gates:

```text
RK3588_CONV_NO_DEVICE=1
RK3588_CONV_DIRECT_SPATIAL=1
RK3588_CONV_DIRECT_SPATIAL_UNSAFE=1
RK3588_CONV_DIRECT_SPATIAL_TASKS=rknpu_sparse_task_gem
RK3588_CONV_RKNN_MEM_SYNC=1
RK3588_CONV_RKNN_SKIP_RESET=1
RK3588_CONV_RKNN_INIT_ACTIONS=1
RK3588_CONV_C64_H56_SPARSE_UNSAFE=1
```

Important safety detail:

- `RK3588_CONV_C64_H56_SPARSE_UNSAFE=1` is required for C64/H56 sparse hardware
  submit.
- Without it, C64/H56 sparse submit is guarded and raises before hardware work.
- This is intentional because C64/H56 currently poisons the next PC job.

Rejected/crashy policy:

```text
RK3588_CONV_DIRECT_SPATIAL_TASKS=rknpu_sparse_task_gem_tail
```

That tail-index hypothesis previously crashed/rebooted the board. Do not rerun
it.

### `experimental/rknn/compare_conv_tiles_offline.py`

Offline verifier. It imports `examples/conv_tiles.py` with:

```text
RK3588_CONV_NO_DEVICE=1
```

It does not submit NPU work.

Coverage currently includes:

- downstream RKNPU ABI layout checks;
- captured RKNN task object comparison;
- live RKNN regcmd window comparison;
- H40 sparse task/regcmd comparison;
- six-desc H7/H14/PW C256/H14/C32-H14 comparisons;
- C64/H56 live regcmd comparison;
- C64/H56 RKNN-mode runtime BO profile comparison;
- C64/H56 RKNN-mode sync profile comparison;
- RKNN ioctl sequence comparison;
- RKNN allocation metadata comparison;
- DMA binding checks;
- RKNPU driver commit candidate model;
- IRQ completion source model;
- timeout source model;
- sysfs snapshot parsing;
- MMIO snapshot parsing, including `MMIO_UNAVAILABLE` skip behavior;
- PC commit trace patch/log parser checks.

Representative command:

```sh
python3 experimental/rknn/compare_conv_tiles_offline.py \
  --ioctl-log experimental/rknn/capture_rknpu_ioctl_sweep_action_readonly_20260524_135914.log \
  --mmio-log experimental/rknn/mmio_snapshot_baseline_after_recovery_20260524.log
```

Latest relevant result:

```text
offline verifier PASS
SKIP rknpu MMIO snapshot comparison
MMIO unavailable: /dev/mem errno=13 error=Permission denied
```

### `experimental/rknn/capture_c64_state.py`

Guarded state-capture harness for the current C64/H56 blocker.

Default is dry-run. It prints commands and does not submit work.

Dry-run examples:

```sh
python3 experimental/rknn/capture_c64_state.py --mode rknn --mmio --sysfs --tag dryrun
python3 experimental/rknn/capture_c64_state.py --mode raw --tag dryrun
```

Raw execution requires both:

```text
--execute --allow-raw-c64
```

Safety check result:

```sh
python3 experimental/rknn/capture_c64_state.py --mode raw --execute --tag should_refuse
```

```text
--execute --mode raw requires --allow-raw-c64
```

Purpose:

- RKNN mode can run the existing ACTION-aware GDB capture with optional sysfs
  and MMIO snapshots.
- Raw mode brackets raw C64/H56 with before/after sysfs/MMIO snapshots.
- It intentionally does not blindly continue into next-job checks because that
  next job is known to timeout while the pollution remains.

### `experimental/rknn/snapshot_rknpu_mmio_readonly.py`

Read-only `/dev/mem` MMIO snapshot helper for selected RKNPU core state.

It now reports permission failure cleanly instead of throwing a traceback:

```text
MMIO_UNAVAILABLE device=/dev/mem errno=13 error=Permission denied
```

This is currently blocked in the non-root shell.

### `experimental/rknn/snapshot_rknpu_sysfs.py`

Non-privileged sysfs snapshot helper.

Latest baseline after recovery:

```text
runtime_status = suspended
runtime_usage = 0
cur_freq = 1000000000
target_freq = 1000000000
governor = rknpu_ondemand
```

This proves runtime/devfreq state after recovery, but it does not expose the
PC/DPU/IRQ state needed for the C64 pollution diagnosis.

### `experimental/rknn/rknpu_pc_commit_trace.patch`

Patch for the downstream RKNPU driver to log PC commit/IRQ/timeout state.

This is the fallback if root MMIO snapshots are not available.

It logs:

- `pc commit prelude`
- `pc commit trace`
- `pc commit regs`
- `pc commit armed`
- `valid irq trace`
- `invalid irq pc regs`
- `timeout pc regs`

Offline parser command:

```sh
python3 experimental/rknn/compare_conv_tiles_offline.py \
  --pc-trace-log experimental/rknn/rknpu_pc_commit_trace.log
```

Current parser status:

- `pc commit prelude` is parsed and compared.
- `valid irq trace` is parsed and compared.
- `invalid irq pc regs` is parsed and compared.
- Trace regexes accept both `0x...` and bare `0`, matching Linux `%#x` output
  where zero prints as `0`.
- A synthetic trace log with prelude, commit, armed, and valid IRQ rows passes
  `compare_pc_trace_log()`.

Additional teardown evidence:

- The offline verifier now parses RKNN debug `free memory` rows.
- H40 and C64/H56 captures both free BO debug rows in this order:
  `internal`, `weight`, `task`, `input`, `output`.
- Each free row is checked against the corresponding allocation row for handle,
  virtual address, object address, DMA address, aligned size, flags, and IOMMU
  domain.
- The existing captures do not show an extra RKNPU destroy ioctl; teardown
  evidence visible from current logs is unsupported `POWER_OFF` action followed
  by RKNN debug free rows.

## Verified working hardware paths

### NPU health check

Command:

```sh
python3 examples/simple_add.py
```

Known latest pass:

```text
SUBMIT ret=0
ADD  NPU=[8 8 8 8 8 8 8 8] expected=[8 8 8 8 8 8 8 8] PASS
```

Use this before and after every risky NPU probe.

### H40 sparse direct-spatial path

Shape:

```text
b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid
```

Controlled command:

```sh
python3 examples/simple_add.py
RK3588_CONV_DIRECT_SPATIAL=1 \
RK3588_CONV_DIRECT_SPATIAL_UNSAFE=1 \
RK3588_CONV_DIRECT_SPATIAL_TASKS=rknpu_sparse_task_gem \
RK3588_CONV_RKNN_MEM_SYNC=1 \
RK3588_CONV_RKNN_SKIP_RESET=1 \
RK3588_CONV_RKNN_INIT_ACTIONS=1 \
python3 examples/conv_tiles.py b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/simple_add.py
```

Result:

```text
pre simple_add PASS
conv_tiles H40 sparse rknpu PASS (max_diff=0.0625)
post simple_add PASS
```

Conclusion: H40 sparse RKNPU path is promoted under the RKNN-mode runtime
conditions above.

### H7 six-desc path

Shape:

```text
b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid
```

Known result:

```text
pre-probe simple_add PASS
b1_c160_h7_w7_oc320_wic160_k3x3_g1_s1_pvalid PASS (max_diff=0.0621)
post-probe simple_add PASS
```

### H14 six-desc path

Shape:

```text
b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid
```

Known result:

```text
pre-probe simple_add PASS
b1_c160_h14_w14_oc320_wic160_k3x3_g1_s1_pvalid PASS (max_diff=0.0625)
post-probe simple_add PASS
```

### C32/H14 six-desc path

Shape:

```text
b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid
```

Known result:

```text
pre-probe simple_add PASS
b1_c32_h14_w14_oc128_wic32_k3x3_g1_s1_pvalid PASS (max_diff=0.0304)
post-probe simple_add PASS
```

### PW C256/H14 path

Shape:

```text
b1_c256_h14_w14_oc512_wic256_k1x1_g1
```

Known result:

```text
pre-probe simple_add PASS
b1_c256_h14_w14_oc512_wic256_k1x1_g1 PASS (max_diff=0.0311)
post-probe simple_add PASS
```

## C64/H56 detailed status

Shape:

```text
b1_c64_h56_w56_oc128_wic64_k1x1_g1
```

RKNN model path:

```text
/home/orangepi/npu/ops_rknn/models/conv2d_cc_b1_c64_h56_w56_oc128_wic64_k1x1_g1.rknn
```

Current modeled evidence:

- 14-record sparse task stream;
- active regcmd offset `0x4800`;
- active regcmd bytes `9344`;
- live active regcmd window `1168` qwords;
- task BO size `560`;
- regcmd BO size `28352`;
- weight/internal BO size `1204224`;
- input BO size `401408`;
- output BO size `802816`;
- captured C64/H56 MEM_SYNC sequence;
- captured ACTION/ioctl sequence;
- captured submit metadata.

Offline status:

```text
PASS unpromoted C64/H56 live regcmd evidence
C64/H56 14-record sparse stream matches live RKNN GEM2 window after DMA/PC base normalization
PASS C64/H56 RKNN-mode runtime profile
PASS C64/H56 RKNN-mode sync profile
```

Hardware probe command used after adding captured C64 sync profile:

```sh
python3 examples/simple_add.py
RK3588_CONV_DIRECT_SPATIAL=1 \
RK3588_CONV_DIRECT_SPATIAL_UNSAFE=1 \
RK3588_CONV_C64_H56_SPARSE_UNSAFE=1 \
RK3588_CONV_DIRECT_SPATIAL_TASKS=rknpu_sparse_task_gem \
RK3588_CONV_RKNN_MEM_SYNC=1 \
RK3588_CONV_RKNN_SKIP_RESET=1 \
RK3588_CONV_RKNN_INIT_ACTIONS=1 \
python3 examples/conv_tiles.py b1_c64_h56_w56_oc128_wic64_k1x1_g1
python3 examples/simple_add.py
```

Result:

```text
pre simple_add PASS
C64/H56 sparse conv PASS (max_diff=0.0156)
post simple_add TimeoutError: [Errno 110] Connection timed out
recovery simple_add PASS after driver timeout soft reset
```

Conclusion:

```text
The missing C64/H56 cleanup/finalization semantic is not the captured MEM_SYNC
profile. The raw C64/H56 sparse path matches the known RKNN userspace-visible
runtime behavior and computes the correct output, but still leaves hidden
PC/DPU/IRQ/core state in a bad condition for the next PC job.
```

Do not rerun C64/H56 sparse hardware unchanged unless the run is explicitly for
state capture.

## Current safety rules

Always use `simple_add.py` before and after intentional hardware probes:

```sh
python3 examples/simple_add.py
# exactly one probe
python3 examples/simple_add.py
```

Do not run C64/H56 raw sparse hardware casually:

```text
RK3588_CONV_C64_H56_SPARSE_UNSAFE=1
```

Only use that flag for a deliberately bracketed state-capture run.

Do not run the old tail-layout policy:

```text
RK3588_CONV_DIRECT_SPATIAL_TASKS=rknpu_sparse_task_gem_tail
```

It previously crashed/rebooted the board.

Do not change submit plumbing without rechecking every field:

- `task_count`
- `task_obj_addr`
- `regcmd_addr`
- `regcfg_amount`
- `int_mask`
- `enable_mask`
- `subcore_task`
- `flags`
- `core_mask`
- PC-chain tail qwords

## Next useful work

### Best next evidence path with root MMIO

Run RKNN C64 state capture:

```sh
sudo -E python3 experimental/rknn/capture_c64_state.py \
  --mode rknn --execute --mmio --sysfs --tag rknn_c64_mmio
```

Run raw C64 state capture:

```sh
sudo -E python3 experimental/rknn/capture_c64_state.py \
  --mode raw --execute --allow-raw-c64 --tag raw_c64_mmio
```

Compare:

- PC data/address registers after RKNN C64 vs raw C64;
- DPU/PPU/RDMA completion or pending state;
- interrupt status/raw status/mask state;
- task counter state;
- whether RKNN leaves any cleanup state that raw path misses.

### Best next evidence path without root MMIO

Use the driver trace patch:

```text
experimental/rknn/rknpu_pc_commit_trace.patch
```

Collect dmesg for:

- RKNN C64 run;
- raw C64 run;
- next `simple_add.py` timeout.

Save as:

```text
experimental/rknn/rknpu_pc_commit_trace.log
```

Then run:

```sh
python3 experimental/rknn/compare_conv_tiles_offline.py \
  --pc-trace-log experimental/rknn/rknpu_pc_commit_trace.log
```

This should show whether the raw C64 run leaves PC commit/IRQ/task state
different from RKNN even though the userspace task/regcmd/sync metadata matches.

### Formulaic tiler work, separate from blocker

Current direct-spatial schedules are still capture-template based. A future
cleanup can replace the captured tables with formulas for:

- y-window splitting;
- k/channel splitting;
- setup/k-half/k-tile/y-tile descriptor ordering;
- record amount generation;
- PC link generation;
- PC core assignment;
- CBUF/grain selection;
- active regcmd offset/window sizing;
- RKNN-mode BO sizing;
- RKNN-mode sync profile selection.

This should be treated as a separate compiler/tiler task. It should not be mixed
with the C64 pollution diagnosis unless it directly explains the hidden hardware
state difference.

## Verification commands

After Python edits:

```sh
python3 -m py_compile \
  examples/conv_tiles.py \
  experimental/rknn/compare_conv_tiles_offline.py \
  experimental/rknn/capture_c64_state.py \
  experimental/rknn/snapshot_rknpu_mmio_readonly.py \
  experimental/rknn/snapshot_rknpu_sysfs.py
```

Offline verifier:

```sh
python3 experimental/rknn/compare_conv_tiles_offline.py \
  --ioctl-log experimental/rknn/capture_rknpu_ioctl_sweep_action_readonly_20260524_135914.log \
  --mmio-log experimental/rknn/mmio_snapshot_baseline_after_recovery_20260524.log
```

NPU health:

```sh
python3 examples/simple_add.py
```

## Dirty worktree notes

The worktree is dirty and contains many untracked capture logs. Do not clean or
revert unrelated files.

Important modified/untracked files observed:

```text
examples/conv_tiles.py
experimental/rknn/compare_conv_tiles_offline.py
experimental/rknn/capture_c64_state.py
experimental/rknn/snapshot_rknpu_mmio_readonly.py
experimental/rknn/snapshot_rknpu_sysfs.py
experimental/rknn/mmio_snapshot_baseline_after_recovery_20260524.log
experimental/rknn/sysfs_snapshot_baseline_after_recovery_20260524.log
experimental/rknn/rknpu_pc_commit_trace.patch
experimental/rknn/capture_rknpu_ioctl_sweep_action_readonly_20260524_135914.log
experimental/rknn/capture_rknpu_ioctl_c64_h56_readonly_20260524_133736.log
experimental/rknn/capture_rknpu_submit_dump_gems_c64_h56_live_regcmd_1168_20260524_133725.log
plan.md
progress.md
```

The capture logs under `experimental/rknn/` are evidence. Do not delete them as
noise.

## Completion status

Completed:

- Created top-level `examples/conv_tiles.py`.
- Converted the top-level CONV experiment from Rocket-style plumbing to
  downstream RKNPU ABI plumbing.
- Added RKNN-style cacheable/IOMMU BO allocation and MEM_SYNC path.
- Added RKNN init ACTION modeling.
- Proved H40 sparse direct-spatial hardware path.
- Proved H7/H14/C32-H14/PW-C256-H14 promoted paths.
- Re-ran/captured RKNN/RKNPU ioctl and live regcmd evidence.
- Modeled C64/H56 task/regcmd/runtime/sync profile against RKNN captures.
- Ran C64/H56 raw sparse hardware once with captured sync profile and confirmed
  correct output plus next-job pollution.
- Added guarded C64 state-capture harness.
- Added/updated MMIO/sysfs snapshot handling and offline verifier skip behavior
  for non-root `/dev/mem` denial.
- Tightened the PC commit trace parser/verifier so instrumented driver logs can
  compare prelude, valid IRQ, invalid IRQ, and timeout PC state against the
  modeled commit state.
- Added RKNN debug free-row teardown comparison for H40 and C64/H56 captures.

Not complete:

- C64/H56 sparse hardware promotion.
- Root MMIO or driver-trace evidence explaining the post-C64 next-job pollution.
- General formulaic tiler to replace RKNN-captured schedule templates.
