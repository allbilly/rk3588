# RK3588 NPU Multicore Plan

## Current Conclusion

Multicore on RK3588 is definitely possible, but there are two different open-source models that must not be conflated:

- Tomeu Vizoso's Rocket/Mesa path has proven multicore support through a new mainline-style Rocket UAPI and kernel scheduler.
- The vendor-style `rknpu_submit.core_mask + subcore_task[]` path also appears to support multicore in `allbilly/rknpu_driver`, but the expected `subcore_task[]` indexing is more specific than our original guess.

The biggest new finding from `allbilly/rknpu_driver` is that 3-core RK3588 jobs reportedly read per-core task ranges from `subcore_task[2]`, `subcore_task[3]`, and `subcore_task[4]`, not `subcore_task[0]`, `subcore_task[1]`, and `subcore_task[2]`.

That may explain why our previous raw `split3` layout was unsafe or invalid.

Tomeu Vizoso's Rocket/Mesa path uses a different UAPI:
- Userspace submits an array of independent jobs with `DRM_IOCTL_ROCKET_SUBMIT`.
- The kernel has one scheduler per NPU core.
- Userspace does **not** choose core 0/1/2 directly.
- The kernel scheduler assigns each job to a core.
- Multiple tasks inside one job execute sequentially on the same core to preserve SRAM/CBUF residency.
- Parallelism comes from submitting multiple independent jobs, not from manually filling `subcore_task[1]` or `[2]`.

That means our local raw ioctl experiments should treat downstream `subcore_task[]` as a driver-specific ABI. Safe work still stays on core 0 by default. Risky multicore experiments must explicitly choose a layout and remain behind `--allow-unsafe-submit`.

Local working references to use before more risky probes:

- `examples/pool_pcchain.py` dry-runs a known-good PC-tail layout and can submit a chained PPU workload only with `--submit`.
- `examples/pool_multicore.py` dry-runs an independent-task multicore layout and can submit only with `--submit`.
- `examples/gemm.py` is the current cleaned-up GEMM implementation and PC-chain reference.
- `experimental/add_pcchain.py` / `experimental/add_rawbuf_pcchain.py` are the ADD/elementwise PC-chain references.
- `experimental/gemm_pcchain.py` is only the historical decoded RKNN `394x394x394` capture reference.
- For conv capture refresh, use `experimental/conv_rawbuf_common.py` preflight helpers before updating inline blobs. The normal `conv_rawbuf_small.py`, `conv_rawbuf_big.py`, `conv_rawbuf_pcchain.py`, and `conv_pcchain.py` entrypoints are standalone and do not load runtime dump files.

Current raw experiment layouts:
- `direct`: maps core 0/1/2 to `subcore_task[0]`, `[1]`, `[2]`; this was our original guessed layout.
- `rk3588-tricore-tail`: maps core 0/1/2 to `subcore_task[2]`, `[3]`, `[4]`; this matches the `allbilly/rknpu_driver` analysis for `use_core_num == 3`.

## Crash Incident

Running `experimental/multicore_elementwise.py` with `--core-mask 0x7 --core-ranges split3` caused a full kernel panic. Later, even some separate raw submits aimed at core 1/2 caused timeout or hard lock depending on task object layout.

Confirmed unsafe patterns on the current downstream driver:
- Single downstream `rknpu_submit` with multiple nonzero `subcore_task[]` entries.
- Raw `core_mask=0x2` / `0x4` experiments without official runtime setup.
- Shifting `task_obj_addr` into the middle of the task BO.
- Launching two target NPU replay commands in parallel. A 2026-05-04 parallel run of `experimental/conv_rawbuf_small.py` and `experimental/conv_rawbuf_big.py` crashed the board; target submits must be serialized.

These crashes do not disprove multicore. They show that the downstream ABI is not the same abstraction as Rocket and should not be guessed into shape.

DeepWiki check on `allbilly/rknpu_driver` for the parallel-target crash:

- The driver has per-core scheduling queues plus IRQ, reset, power, and IOMMU-domain locks.
- Job submission is queued per core rather than globally serialized for all userspace processes.
- Reset/action paths are separate from submit/wait paths, so two replay processes can still interfere if one resets or times out while another is preparing or waiting on a job.
- User-space replay tools should take a process lock around the whole target path: open `/dev/dri/card1`, allocate/map BOs, reset, submit, wait, and read back.
- Kernel-side fixes would be stricter task/regcmd validation and stronger reset-vs-submit serialization, but the immediate local rule is one target NPU command at a time.

## Ground Truth From allbilly/rknpu_driver

DeepWiki source analysis of `allbilly/rknpu_driver` says the vendor-style driver supports RK3588 multicore execution:

- RK3588 config advertises `core_mask = 0x7` for three cores.
- `rknpu_submit` exposes `core_mask` and `subcore_task[5]` to userspace.
- `rknpu_job_alloc` derives `job->use_core_num` from `args->core_mask`.
- `rknpu_job_commit` commits work to each selected core.
- `run_count` and `interrupt_count` are initialized from `use_core_num`, so a multicore job is not complete until all selected cores finish.
- `rknpu_job_schedule` handles `RKNPU_CORE_AUTO_MASK` by choosing a core, while explicit masks select the participating cores.

Important `subcore_task[]` indexing detail from `rknpu_job_subcore_commit_pc`:

```text
use_core_num == 1 or 2:
  core i uses subcore_task[i]

use_core_num == 3:
  core 0 uses subcore_task[2]
  core 1 uses subcore_task[3]
  core 2 uses subcore_task[4]
```

Implications:
- A 3-core raw submit with only `subcore_task[0..2]` populated may be invalid.
- The driver may read `subcore_task[3]` and `[4]` as zero if userspace did not populate them.
- Zero or wrong task ranges could lead to failed submit, NPU fault, timeout, or hard lock.
- Task ranges are expected to be explicit per-core ranges. They should normally be disjoint unless the compiler/runtime intentionally creates duplicated task streams.

This means the earlier captured RKNN layout:

```text
task_number = 3
subcore_task[0] = {0, 1}
subcore_task[1] = {0, 1}
subcore_task[2] = {0, 1}
```

is not enough to prove a valid 3-core raw layout. If `core_mask=0x7` and the driver uses the 3-core path, the relevant entries would be `[2]`, `[3]`, and `[4]`.

Updated experiment code:
- `experimental/rknpu_common.py` now supports `subcore_layout="direct"` and `subcore_layout="rk3588-tricore-tail"`.
- `experimental/multicore_elementwise.py` and `experimental/multicore_gemm.py` expose `--subcore-layout`.
- `experimental/multicore_probe.py` exposes `--use-rk3588-tricore-tail-layout` for risky split3 probes.
- All nonzero-core or multi-range raw submits still require `--allow-unsafe-submit`.

Candidate risky command, only with physical reset access:

```bash
python experimental/multicore_elementwise.py --n 4096 --ops ADD,MUL,SUB --core-mask 0x7 --core-ranges split3 --subcore-layout rk3588-tricore-tail --allow-unsafe-submit
```

Result on this board/runtime:

```text
flags=0x5
core_mask=0x7
task_number=3
subcore_task[0]=(0,0)
subcore_task[1]=(0,0)
subcore_task[2]=(0,1)
subcore_task[3]=(1,1)
subcore_task[4]=(2,1)
submit ret=[0] elapsed_ms=0.218
ADD task[0] PASS sentinel=True max_diff=0.0000
MUL task[1] FAIL sentinel=False max_diff=nan
SUB task[2] FAIL sentinel=False max_diff=nan
```

Interpretation:
- The `rk3588-tricore-tail` layout did **not** hard-lock the board.
- The ioctl returned success.
- Only task 0 produced output.
- Tasks assigned to cores 1 and 2 left output sentinels untouched.
- Therefore, multicore execution is still **not working** in our raw userspace experiment.
- This suggests either cores 1/2 were not actually committed/enabled, the captured allbilly indexing is not sufficient for this kernel/runtime, or additional compiler/runtime metadata/register setup is required for nonzero cores.

Follow-up: core 0 with all three task descriptors in one submit also timed out:

```bash
python experimental/multicore_elementwise.py --n 4096 --ops ADD,MUL,SUB --core-mask 0x1 --core-ranges core0
```

This supports the PC-chaining concern from `conv_vs_mesa.md`: before a `task_number > 1` range can work reliably, the register command streams may need Mesa-style PC tail patching:

- `REG_PC_BASE_ADDRESS` points to the next regcmd buffer.
- `REG_PC_REGISTER_AMOUNTS` gives the next regcmd count.
- the chain ends with zero next-address/count.

`experimental/multicore_elementwise.py` now has `--pc-chain` to patch regcmd tails this way for elementwise experiments.

Narrow core-0 chain test:

```bash
python experimental/multicore_elementwise.py --n 4096 --ops ADD,ADD,ADD --core-mask 0x1 --core-ranges core0 --pc-chain --timeout 6000
```

Result:

```text
task[0] op=ADD regcfg_amount=18 regcfg_offset=0
task[1] op=ADD regcfg_amount=18 regcfg_offset=256
task[2] op=ADD regcfg_amount=18 regcfg_offset=512
TimeoutError: [Errno 110] Connection timed out
```

Conclusion: before multicore can work, we need to solve **single-core multi-task PC submit**. Multicore experiments should pause until one core can complete three chained/descriptor tasks in one submit.

Plan to solve single-core multi-task PC submit:

1. Find a known-good multi-task submit from official RKNN or `~/npu` artifacts.
2. Capture or locate the exact `rknpu_submit` fields for that known-good run:
   - `flags`
   - `task_number`
   - `task_obj_addr`
   - `task_base_addr`
   - `core_mask`
   - `subcore_task[]`
3. Decode the first several `struct_rknpu_task` descriptors:
   - `regcmd_addr`
   - `regcfg_amount`
   - `regcfg_offset`
   - `enable_mask`
   - `int_mask`
4. Dump the regcmd tail of each task and compare:
   - whether `REG_PC_BASE_ADDRESS` / `REG_PC_REGISTER_AMOUNTS` are embedded
   - whether the final task uses zeros
   - whether `REG_PC_OPERATION_ENABLE` value is `0x0d`, `0x18`, or another mask
5. Compare task BO layout:
   - whether descriptor array starts at `task_obj_addr`
   - whether `task_base_addr` is zero or points to task DMA base
   - whether `regcfg_offset` mode or absolute `regcmd_addr` mode is used
6. Reproduce the known-good single-core multi-task shape in `experimental/multicore_elementwise.py` before retrying `core_mask=0x7`.

`~/npu` reference found:

- File: `~/npu/include/rknnops.h`
- Key section: `build_handles()` around lines 4380-4470 and `submitTask()` around lines 4473-4505.
- It supports multiple PC tasks with one submit.
- It emits `REG_PC_BASE_ADDRESS` and `REG_PC_REGISTER_AMOUNTS` placeholders inside each task's regcmd stream.
- During finalization, it patches:
  - `REG_PC_BASE_ADDRESS` to `PC_BASE_ADDRESS_PC_SOURCE_ADDR((uint32_t)(next_addr >> 4))`
  - `REG_PC_REGISTER_AMOUNTS` to `PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT((uint32_t)reg_task_lengths[i])`
- It pads each task regcmd segment to 64-byte alignment.
- It sets each task descriptor:
  - `op_idx = 1`
  - `enable_mask = 0x0d` for normal DPU/CNA/Core style tasks
  - `int_mask = 0x300`
  - `regcfg_amount = reg_task_lengths[i]`
  - `regcfg_offset = reg_task_offsets[i] * 8`
  - `regcmd_addr = reg_base_addr + regcmd_offset`
- It submits:
  - `flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | optional RKNPU_JOB_PINGPONG`
  - `task_number = task_count`
  - `task_obj_addr = tasks_obj`
  - `task_base_addr = 0`
  - `core_mask = 0` in that reference
  - `subcore_task[0] = {0, task_count}`

Important differences from our first `--pc-chain` attempt:
- Our previous `REG_PC_BASE_ADDRESS` encoding incorrectly set `PC_SEL=1`; `rknnops.h` leaves it as the encoded `next_addr >> 4` field only.
- Our previous `REG_PC_REGISTER_AMOUNTS` used the next task length; `rknnops.h` patches each task's amount field to that same task's `reg_task_lengths[i]`.
- `rknnops.h` uses `op_idx=1`, while our raw elementwise task currently uses `op_idx=4`.
- `rknnops.h` includes `RKNPU_JOB_BLOCK` in submit flags, while our raw helper originally had `RKNPU_JOB_BLOCK = 0` and did not expose it.

Mesa Rocket PC-chain ground truth from local source:

- DeepWiki's public `mesa/mesa` index did not include the Rocket driver, so the local `mesa/` checkout is the source of truth here.
- Register definitions: `mesa/src/gallium/drivers/rocket/registers.xml`:
  - `REG_PC_BASE_ADDRESS = 0x0010`
  - `PC_SOURCE_ADDR` bits `[31:4]`
  - `PC_SEL` bit `0`
  - `REG_PC_REGISTER_AMOUNTS = 0x0014`
  - `PC_DATA_AMOUNT` bits `[15:0]`
- Regcmd emission: `mesa/src/gallium/drivers/rocket/rkt_regcmd.c`:
  - if `num_tasks == 1`, Mesa appends a literal zero where the next-address command would be.
  - if `num_tasks > 1`, Mesa emits `REG_PC_BASE_ADDRESS, 0`.
  - Mesa then emits `REG_PC_REGISTER_AMOUNTS, 0`.
  - Mesa then appends raw `0x0041000000000000`.
  - Mesa then emits `REG_PC_OPERATION_ENABLE` with `PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1)`.
- Tail layout is therefore the final four qwords of each regcmd segment:
  1. `REG_PC_BASE_ADDRESS` or literal zero
  2. `REG_PC_REGISTER_AMOUNTS`
  3. raw `0x0041000000000000`
  4. `REG_PC_OPERATION_ENABLE`
- Patching: `mesa/src/gallium/drivers/rocket/rkt_ml.c::compile_operation()`:
  - patches `reg_count - 4` with `addr << 16`, where `addr` is the next regcmd physical address.
  - patches `reg_count - 3` with `regs_to_fetch << 16`.
  - `regs_to_fetch = len(next_regcfg)` initially.
  - `regs_to_fetch -= 4` to exclude the next segment's four-qword PC tail.
  - `regs_to_fetch = ALIGN(regs_to_fetch / 2, 2)`.
  - final task remains zero/zero and terminates the chain.
- Mesa task descriptors are Rocket UAPI `drm_rocket_task`, not vendor `struct rknpu_task`:
  - `regcmd = task->regcfg_addr`
  - `regcmd_count = task->regcfg_amount`

Important difference from our second `--pc-chain` attempt:
- We patched `REG_PC_REGISTER_AMOUNTS` with the raw current segment length.
- Mesa patches it with the transformed **next** segment fetch amount: `ALIGN((next_len - 4) / 2, 2)`.
- This may explain the core-0 PC-chain timeout or no-output behavior.

Additional `~/npu` dump artifact finding:

- `~/npu/ops_rknn/dump/gem1-dump` shows a valid official RKNN PC tail:
  - literal zero at the next-address slot for a single-task chain
  - `REG_PC_REGISTER_AMOUNTS` encoded as raw qword `0x0101000000000014`
  - raw `0x0041000000000000`
  - `REG_PC_OPERATION_ENABLE` encoded as raw qword `0x00810000000d0008`
- This means `REG_PC_BASE_ADDRESS` and `REG_PC_REGISTER_AMOUNTS` use target `0x101`, not `0x81`.
- `REG_PC_OPERATION_ENABLE` uses target `0x81`.
- Our previous PC-chain attempts used target `0x81` for `BASE_ADDRESS` and `REGISTER_AMOUNTS`, which likely made those writes ineffective or wrong.

Focused official RKNN matmul split dump:

- `~/npu/README` says `424x424x424` is the known matmul boundary case and the FAQ says too-large matmul is split along `N`.
- `~/npu/ops_rknn/matmul_api.cpp` now accepts CLI dimensions, so the focused dump command was:

```bash
cd ~/npu/ops_rknn
RKNN_CORE_MASK=1 sh ./run.sh --gdb matmul_api 424 424 424
```

Result:
- The RKNN matmul run completed on core 0 and passed.
- Runtime allocated a task BO of 80 bytes and a `shared_cmd_tensor` BO of 1792 bytes.
- The task BO contained exactly two valid descriptors:

```text
Task 0:
  flags         = 0
  op_idx        = 0
  enable_mask   = 0x0d
  int_mask      = 0x300
  int_clear     = 0x1ffff
  regcfg_amount = 108
  regcfg_offset = 0
  regcmd_addr   = 0xfff76000

Task 1:
  flags         = 0
  op_idx        = 0
  enable_mask   = 0x0d
  int_mask      = 0x300
  int_clear     = 0x1ffff
  regcfg_amount = 108
  regcfg_offset = 0
  regcmd_addr   = 0xfff76380
```

PC-chain details from the first segment tail:

```text
REG_PC_BASE_ADDRESS      -> 0xfff76380 >> 4
REG_PC_REGISTER_AMOUNTS  -> 55
0x0041000000000000
REG_PC_OPERATION_ENABLE  -> 0x000d0008
```

Final segment tail:

```text
literal zero for next address
REG_PC_REGISTER_AMOUNTS -> 0
0x0041000000000000
REG_PC_OPERATION_ENABLE -> 0x000d0008
```

Important correction:
- Official RKNN does **not** set `PC_REGISTER_AMOUNTS` to the descriptor `regcfg_amount` (`108`).
- Official RKNN does **not** use Mesa's `ALIGN((next_len - 4) / 2, 2)` formula (`52` for this dump).
- For `regcfg_amount=108`, official RKNN uses `55`, i.e. `regcfg_amount / 2 + 1`.
- The four-qword PC tail is included in descriptor `regcfg_amount=108`; the next segment starts at `0x380` bytes because each segment is padded to `112` qwords / 64-byte alignment.
- Descriptors use absolute `regcmd_addr` with `regcfg_offset=0`; this matches our `--regcmd-mode absolute`, not the default `offset` mode.

Submit header capture for `424x424x424` without forced mask:

```text
flags=0x5
timeout=6000
task_number=6
task_obj_addr=<kernel task obj addr>
task_base_addr=0
core_mask=0
subcore_task[0]={0,2}
subcore_task[1]={0,2}
subcore_task[2]={0,2}
subcore_task[3]={0,0}
subcore_task[4]={0,0}
```

This submit passed. The task BO still contains only two descriptors, so RKNN reports `task_number=6` because the same two-descriptor PC chain is made available to three possible cores. With `RKNN_CORE_MASK=1`, the submit header becomes `core_mask=1` but the same `task_number=6` / three duplicate subcore ranges were used, and this shape timed out in that particular forced-core run. Therefore, the passed focused dump came from RKNN's auto/core-mask-zero path, not forced core 0.

New GEMM harness attempt:

- Added `experimental/gemm_multicore.py` as a GEMM-specific reproducer that imports `examples/gemm.py` helpers and can run either an official baseline or explicit row tiles.
- The baseline path still times out on `examples/gemm.py` for `424x424x424`, which matches the README note that this shape was a known failure in that branch.
- The explicit official-style submit header (`flags=0x5`, `task_number=6`, `core_mask=0`, `subcore_task[0..2]=(0,2)`) also timed out when paired with the raw GEMM register stream from `examples/gemm.py`.

Conclusion:
- The official RKNN `ops_rknn` dump is the useful ground truth for the shape.
- `examples/gemm.py` is still a good register-layout reference, but it is not the working execution path for `424x424x424` on this branch.
- Next step should be a register diff between the official `ops_rknn` dump and `examples/gemm.py`'s emitted GEMM stream, especially around the task header and PC tail, before trying more split submits.

Experiment updates after comparing `rknnops.h`:
- `experimental/rknpu_common.py` now defines `RKNPU_JOB_BLOCK = 0 << 1`; bit 1 is `RKNPU_JOB_NONBLOCK`.
- `experimental/multicore_elementwise.py --pc-chain` now encodes `REG_PC_BASE_ADDRESS` without setting `PC_SEL`.
- `--pc-chain` now patches `REG_PC_REGISTER_AMOUNTS` with the current segment length, matching `rknnops.h`.
- Added `--task-op-idx` to test `op_idx=1` versus the previous `op_idx=4`.
- Added `--submit-block` to include `RKNPU_JOB_BLOCK` in default submit flags.

Core-0 chain test matching `rknnops.h` descriptor flags more closely:

```bash
python experimental/multicore_elementwise.py --n 4096 --ops ADD,ADD,ADD --core-mask 0x1 --core-ranges core0 --pc-chain --task-op-idx 1 --submit-block --timeout 10000
```

Result:

```text
flags=0x5
core_mask=0x1
task_number=3
subcore_task[0]=(0,3)
submit ret=[0]
ADD task[0] FAIL sentinel=False
ADD task[1] FAIL sentinel=False
ADD task[2] FAIL sentinel=False
```

Core-0 chain test with our previous known-good `op_idx=4` plus `RKNPU_JOB_BLOCK`:

```bash
python experimental/multicore_elementwise.py --n 4096 --ops ADD,ADD,ADD --core-mask 0x1 --core-ranges core0 --pc-chain --submit-block --timeout 10000
```

Result:

```text
flags=0x5
core_mask=0x1
task_number=3
subcore_task[0]=(0,3)
submit ret=[0]
ADD task[0] FAIL sentinel=False
ADD task[1] FAIL sentinel=False
ADD task[2] FAIL sentinel=False
```

Interpretation:
- `RKNPU_JOB_BLOCK` changes failure mode from timeout to quick successful ioctl with no output writes.
- `op_idx=1` from `rknnops.h` is not directly compatible with this raw elementwise DPU/RDMA task descriptor/register mix.
- `op_idx=4` plus block also does not execute the chain.
- The remaining difference is likely the exact register stream shape used by `rknnops.h` for multi-task PC mode, not just the submit header. We need to build or run an actual `~/npu/ops_reg` case that produces `task_count > 1`, then dump its descriptors/regcmds and copy that layout.

Safer dry/static command:

```bash
python -m py_compile experimental/rknpu_common.py experimental/multicore_elementwise.py experimental/multicore_gemm.py experimental/multicore_probe.py
```

## Ground Truth From Rocket/Mesa

Local source: `./mesa` Rocket branch.

### Rocket UAPI

File: `mesa/include/drm-uapi/rocket_accel.h`

Important structs:
```c
struct drm_rocket_task {
    __u32 regcmd;
    __u32 regcmd_count;
};

struct drm_rocket_job {
    __u64 tasks;
    __u64 in_bo_handles;
    __u64 out_bo_handles;
    __u32 task_count;
    __u32 task_struct_size;
    __u32 in_bo_handle_count;
    __u32 out_bo_handle_count;
};

struct drm_rocket_submit {
    __u64 jobs;
    __u32 job_count;
    __u32 job_struct_size;
    __u64 reserved;
};
```

Key comment in `drm_rocket_job`:
```text
All tasks in the same job will be executed sequentially on the same core,
to benefit from memory residency in SRAM.
```

This is the core scheduling contract.

### Mesa Job Construction

File: `mesa/src/gallium/drivers/rocket/rkt_ml.c`

In `rkt_ml_subgraph_invoke`, Mesa converts each operation into one or more Rocket jobs.

If weights can be reused in CBUF:
```c
/* Submit all tasks to the same core, so weights can be reused */
job.task_count = task_count;
```

If weights cannot be reused:
```c
/* Spread tasks among cores, for parallelism */
job.task_count = 1;
```

Then Mesa submits all jobs at once:
```c
submit.jobs = (uint64_t)util_dynarray_begin(&jobs);
submit.job_count = util_dynarray_num_elements(&jobs, struct drm_rocket_job);
drmIoctl(screen->fd, DRM_IOCTL_ROCKET_SUBMIT, &submit);
```

Interpretation:
- A task is one register-command stream.
- A job is the scheduler unit.
- Multiple tasks in one job are intentionally serialized on one core.
- Multiple jobs are eligible to run concurrently across the three cores.
- Userspace exposes dependencies with input/output BO handles, not with PC-chain subcore ranges.

### Mesa Task Splitting

File: `mesa/src/gallium/drivers/rocket/rkt_task.c`

`rkt_split_tasks` splits convolution work by input slices when one operation cannot fit in CBUF. It computes per-task:
- `top_slice` / `bottom_slice`
- `input_offset`
- `output_offset`
- overlap and retained slices
- per-task padding
- per-task output height

This is the practical tiling strategy we should copy conceptually for Phase 3. The split is not arbitrary: each task has adjusted input/output offsets and shape registers.

### Mesa PC Chaining

File: `mesa/src/gallium/drivers/rocket/rkt_ml.c`

`compile_operation` builds one register command block per split task and patches next-task PC metadata when tasks are in the same job:
```c
if (i < num_tasks - 1) {
    /* Patch next address and amount of regs to fetch */
    *next_address_reg |= addr << 16;
    *reg_count_reg |= regs_to_fetch << 16;
}
```

Interpretation:
- PC chaining is used for sequential tasks inside one job.
- PC chaining is not the mechanism for distributing work across cores.
- Distribution across cores happens at the Rocket **job** level.

## Ground Truth From Official Rockchip Runtime

Official RKNN/RKLLM also confirms that multicore is not simply a userspace bitmask on arbitrary register streams.

Evidence from `rknn-toolkit2`:
- `rknn_api.h` exposes `rknn_set_core_mask` with `RKNN_NPU_CORE_0=1`, `RKNN_NPU_CORE_1=2`, `RKNN_NPU_CORE_2=4`, `RKNN_NPU_CORE_0_1_2=7`.
- `rknn_matmul_api.h` exposes `rknn_matmul_set_core_mask`.
- Matmul demo only calls `rknn_matmul_set_core_mask`; it does not touch low-level `subcore_task[]`.

Evidence from `rknn-llm`:
- RKLLM export passes `num_npu_core=3` at build time.
- This suggests compiler/model metadata participates in multicore partitioning.

Evidence from `librknnrt.so` strings:
```text
This model has not multicore config, fallback single core in current, if you want to use multicore mode please rebuild model
Job can not submit multi core tasks
Illegal job core_mask %d
Not support core mask: %x, fallback to single core auto mode
For rk3588/rk3576, core mask for matmul api will fall back to RKNN_NPU_CORE_AUTO mode
```

Interpretation:
- Official runtime validates whether a job/model may run multicore.
- Normal raw register streams do not contain that validation metadata.
- RKNN multicore and Rocket multicore both use a higher-level scheduling concept than our guessed downstream `subcore_task[]` layout.

## Martin Chang RWKV Evidence

Martin Chang's “Porting RWKV to RK3588 NPU” slides describe a different successful multicore strategy for LLM/RWKV workloads.

Repository check:
- `https://github.com/marty1885/rk3588-matmul-bench` is a benchmark for the RKNN matmul API.
- Its `bench.cpp` creates one `rknn_matmul_ctx` per `(M,K,N,type,layout)` case and repeatedly calls `rknn_matmul_run(ctx)`.
- It does not call `rknn_matmul_set_core_mask`.
- It does not split `N` across multiple contexts.
- It does not implement the RWKV/GGML multicore stitching described in the slides.

Branch audit of `https://github.com/marty1885/llama.cpp`:
- Cloned all public branches with `git clone --no-single-branch --filter=blob:none`.
- Public branches checked: `backend-dev`, `backend-dev-2`, `core-refactor`, `core-refactor-merge-mm-merge-upstream`, `et-dev`, `et-dev-progress`, `et-uberkernel`, `master`, `metalium-*`, `rknpu2-backend`, `rwkv-edit`, `rwkv-probe`, `tensorfma-cxx`.
- Only `rknpu2-backend` contains `ggml-rknpu2.c` and `ggml-rknpu2.h`.
- `rknpu2-backend` creates one cached RKNN matmul context per `(M,K,N,type)` and calls `rknn_matmul_run`.
- `rknpu2-backend` has a single explicit core-mask call: `rknn_matmul_set_core_mask(kernel->matmul_ctx, RKNN_NPU_CORE_1)`.
- `rknpu2-backend` does not split `N`, does not create per-core matmul contexts for output-column tiles, and does not stitch partial outputs.
- `rwkv-probe` and `rwkv-edit` contain RWKV probing/editing tools under `tools/rwkv-probe`, but no RKNN/RKNPU backend files and no RKNN matmul multicore implementation.

Therefore, the public `rk3588-matmul-bench` repo and public `marty1885/llama.cpp` branches explain the RKNN matmul backend and benchmarking work, but not the multicore RWKV split. The N-axis multicore implementation from the slides appears unpublished, in another repository, or in a branch that is no longer public.

Important points from the slides:
- RK3588 NPU has 3 cores, but RKNN matmul API in the tested SDK used only one NPU core per matmul.
- The useful LLM workload is mostly GEMV/GEMM, especially `M=1` token-generation matmuls.
- RKNN matmul is fast for square/large `M`, but not ideal for GEMV (`M=1`) because runtime/driver latency dominates.
- Martin manually split matmul along the `N` axis into multiple pieces.
- Each NPU core computes a different output-column slice.
- The partial outputs are stitched back together on CPU.
- A 2-core split improved token time from about `83ms/token` to `76ms/token`.
- A 3-NPU-core plus 1-CPU-core split reached about `65ms/token`, close to CPU speed but unstable.

This is important because it confirms a vendor-driver-compatible multicore strategy that does **not** require one downstream `rknpu_submit` with multiple `subcore_task[]` entries.

Martin-style model:
```text
C = A @ B
A shape: M x K, often M=1 for token generation
B shape: K x N

Split B by output columns:
B0 = B[:, 0:N0]
B1 = B[:, N0:N1]
B2 = B[:, N1:N]

Core/job 0 computes C0 = A @ B0
Core/job 1 computes C1 = A @ B1
Core/job 2 computes C2 = A @ B2

CPU stitches C = concat(C0, C1, C2, axis=N)
```

Implications for this repo:
- For matmul/GEMV, the first serious multicore experiment should be **N-axis tiling**, not `experimental/multicore_gemm.py --core-ranges split3`.
- Each tile should be a complete standalone matmul with its own weight slice and output buffer.
- For official RKNN matmul API, this likely means multiple `rknn_matmul_ctx` objects, each created for smaller `N_tile`.
- For raw `gemm.py`, this means generating independent register streams with different weight/output base addresses and dimensions.
- Correctness can be proven safely by running all N-tiles sequentially on core 0, then stitching and comparing to CPU.
- Only after sequential tiling works should we try multiple contexts/processes and nonzero core masks.

## Difference Between Rocket and Downstream RKNPU ABI

### Rocket model

```text
submit
  job[0] -> scheduler chooses a core -> task(s) run sequentially on that core
  job[1] -> scheduler chooses a core -> task(s) run sequentially on that core
  job[2] -> scheduler chooses a core -> task(s) run sequentially on that core
```

Userspace provides:
- jobs
- tasks per job
- input BO handles
- output BO handles

Userspace does not provide:
- `core_mask`
- `subcore_task[]`
- per-core task ranges

### Downstream RKNPU model in this repo

```text
rknpu_submit
  core_mask
  task_number
  subcore_task[0]
  subcore_task[1]
  subcore_task[2]
```

This ABI looks like userspace can assign task ranges to cores, but local tests show guessed assignments can hard-lock the downstream kernel. It may require hidden runtime setup, PC-linked tasks, model metadata, or a different interpretation of fields.

## Revised Strategy

The correct direction is:

1. Treat Rocket as the open-source design target.
2. Learn from Rocket's job/task split and scheduler model.
3. Stop assuming `subcore_task[]` is the raw equivalent of Rocket jobs.
4. For local scripts, prioritize manual tiling into independent jobs/submits with separate outputs.
5. Only test downstream `subcore_task[]` multi-range after a staged capture plan, and only late because crash is accepted there.

## Safe Script State

Current guard changes in this repo:
- `experimental/multicore_elementwise.py` defaults to `core0`, `core_mask=0x1`.
- `experimental/multicore_gemm.py` defaults to `core0`, `core_mask=0x1`.
- Raw nonzero-core and multi-range paths require `--allow-unsafe-submit`.
- `experimental/multicore_probe.py` requires explicit opt-in for risky probes.
- `test/test_multicore_*.py` are static guard tests and do not submit to NPU.

## Experiment Run Log

Date: 2026-05-03.

Environment:
- Kernel: `Linux orangepi5 6.1.99-rockchip-rk3588 #1.2.2 SMP Thu Jun 26 14:42:53 CST 2025 aarch64`.
- Device: `/dev/dri/card1` exists.
- RKNN runtime reported by official binaries: `librknnrt version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)`.
- RKNN driver reported by official binaries: `0.9.8`.
- `/sys/kernel/debug/rknpu/*` exists but is not readable as the current user.

Tools added/modified for the experiment:
- Added `capture_rknpu_submit.gdb`, a read-only GDB script that decodes `DRM_IOCTL_RKNPU_SUBMIT` without patching the submit buffer.
- Patched `~/npu/ops_rknn/add.cpp` to accept `RKNN_CORE_MASK=<mask>` and call `rknn_set_core_mask(ctx, mask)` after `rknn_init`.

### Experiment 1: Hardware Health With Local Raw Core 0

Command:
```text
python test/test_add.py
```

Result:
```text
45x68 Tensor.add: PASS
45x68 operator add: PASS
scalar add: PASS
ALL TEST CASES PASS
```

Conclusion:
- The NPU and downstream driver are not globally wedged.
- Known-good local raw core-0 submissions still work.

### Experiment 2: Official RKNN `add` Without Forced Core Mask

Command:
```text
cd ~/npu/ops_rknn
./add
```

Result:
- Runtime initialized successfully.
- Submit failed after roughly 18 seconds:
```text
failed to submit!, op id: 4, op name: Add:/Add, flags: 0x5, task start: 0, task number: 1, run task counter: 0, int status: 0
rknn_run failed: -1
```

GDB capture before timeout showed:
```text
flags=0x00000005
task_number=3
core_mask=0x00000000
subcore_task[0]={task_start=0, task_number=1}
subcore_task[1]={task_start=0, task_number=1}
subcore_task[2]={task_start=0, task_number=1}
```

Conclusion:
- Official runtime default/auto mode emits a multicore-looking downstream submit on this model/kernel.
- This layout fails on this board/kernel/runtime combination.
- It is not identical to our original `split3`; all three subcore entries point at `task_start=0`, not `0/1/2`.

### Experiment 3: Official RKNN `add` Forced To Core 0

Command:
```text
cd ~/npu/ops_rknn
RKNN_CORE_MASK=1 sh ./run.sh add
```

Result:
- `rknn_set_core_mask(1) ret=0`.
- Submit completed; no timeout.
- The available model is `models/add_float16_1x2.rknn`, but the test harness compares four values, so the output check reports mismatch for elements not produced by the model:
```text
output : 2 2.24219 0 0
expect : 2 2.24268 2.29492 2.17188
NPU result match CPU: NO
```

GDB capture:
```text
flags=0x00000005
task_number=3
core_mask=0x00000001
subcore_task[0]={task_start=0, task_number=1}
subcore_task[1]={task_start=0, task_number=1}
subcore_task[2]={task_start=0, task_number=1}
```

Conclusion:
- For this official model, forcing `core_mask=1` avoids the submit failure.
- The runtime still fills all three `subcore_task[]` entries; `core_mask` gates actual execution.
- This is the first solid downstream clue: `subcore_task[]` entries alone do not mean all cores are used; `core_mask` is decisive.

### Experiment 4: Official RKNN `add` Forced To Core 1

Command:
```text
cd ~/npu/ops_rknn
RKNN_CORE_MASK=2 sh ./run.sh add
```

Result:
- `rknn_set_core_mask(2) ret=0`.
- Submit failed after roughly 18 seconds with the same `failed to submit!` message.

GDB capture:
```text
flags=0x00000005
task_number=3
core_mask=0x00000002
subcore_task[0]={task_start=0, task_number=1}
subcore_task[1]={task_start=0, task_number=1}
subcore_task[2]={task_start=0, task_number=1}
```

Conclusion:
- The runtime accepts core mask 2 at API level, but the downstream driver/hardware path does not complete the job on core 1.
- This matches earlier raw experiments where nonzero-core execution did not produce output or timed out.

### Experiment 5: Official RKNN `add` Forced To Core 2

Command:
```text
cd ~/npu/ops_rknn
RKNN_CORE_MASK=4 sh ./run.sh add
```

Result:
- `rknn_set_core_mask(4) ret=0`.
- Submit failed after roughly 18 seconds with the same `failed to submit!` message.

Conclusion:
- Core 2 behaves like core 1 on this downstream driver/runtime setup: accepted by API, but execution does not complete.

### Experiment 6: Official RKNN Matmul Baseline

Command:
```text
cd ~/npu/ops_rknn
./matmul_api
```

Result without forced core mask:
- All tested matmul shapes failed submit with the same timeout-style error.
- Summary: `passed=0, failed=5`.

Then `~/npu/ops_rknn/matmul_api.cpp` was patched to accept `RKNN_CORE_MASK=<mask>` and call `rknn_matmul_set_core_mask(ctx, mask)` after context creation.

Command:
```text
cd ~/npu/ops_rknn
RKNN_CORE_MASK=1 sh ./run.sh matmul_api
```

Result with forced core 0:
- `32x32x32` failed submit.
- `64x99x64` passed, `0.118ms`.
- `64x64x64` passed, `0.160ms`.
- `256x256x256` passed, `0.807ms`.
- `M=1, K=N=8155` passed, `12.158ms`.
- Summary: `passed=4, failed=1`.

Command:
```text
cd ~/npu/ops_rknn
RKNN_CORE_MASK=2 sh ./run.sh matmul_api
```

Result with forced core 1:
- The board crashed/hard-locked.
- Hardware testing was stopped immediately after this.

Conclusion:
- Official matmul API is usable only when forced to core 0 on this board/kernel/runtime combination.
- Nonzero-core official matmul (`RKNN_CORE_MASK=2`) can hard-lock the machine.
- Martin-style N-axis splitting is still useful for **safe sequential core-0 correctness validation**, but not safe for nonzero-core execution on this vendor kernel without further driver/device-tree investigation.

### Experiment Stop Decision

Do not proceed to raw nonzero-core or raw multi-range tests yet.

Reason:
- Official runtime core 1 and core 2 already fail cleanly without crashing.
- Official matmul API with `RKNN_CORE_MASK=2` hard-locked the board.
- Local raw nonzero-core tests previously caused hard lockups.
- There is little value in repeating raw core 1/2 until we know why official runtime cannot complete on those cores.

Next useful experiments:
- Determine whether NPU cores 1/2 are disabled, power-gated, clock-gated, or unsupported by this board's downstream kernel/device tree.
- Compare `/proc/device-tree` NPU nodes and IRQs against a known-good RK3588 multicore setup.
- If root access is available, inspect `/sys/kernel/debug/rknpu/version`, `/sys/kernel/debug/rknpu/load`, and kernel logs after core 1/2 failures.
- Build/run Tomeu's Rocket driver path on a mainline kernel if true multicore is the goal.

## Redesigned Experiment Plan

The first stages must not crash. Risky downstream raw-core tests come late.

### Stage 0: Recovery Setup

Risk: none.

Before hardware experiments:
- Confirm physical reset access.
- Open serial console if available.
- Run `dmesg -w` in another shell.
- Record kernel version and driver path.
- Confirm whether testing Rocket DRM driver or downstream `/dev/dri/card1` RKNPU driver.

Success criteria:
- Recovery path is known.
- Logs are being captured.

### Stage 1: Static Rocket Study

Risk: none.

Read these files:
- `mesa/include/drm-uapi/rocket_accel.h`
- `mesa/src/gallium/drivers/rocket/rkt_ml.c`
- `mesa/src/gallium/drivers/rocket/rkt_task.c`
- `mesa/src/gallium/drivers/rocket/rkt_regcmd.c`
- `mesa/src/gallium/frontends/teflon/tfl_device.c`

Questions:
- How many jobs are submitted for a model?
- Which operations become multi-task jobs?
- When does Mesa group tasks into one job versus split into many jobs?
- Which register fields change per split task?

Success criteria:
- We can explain Rocket's job/task model without running hardware.

### Stage 2: Build/Run Rocket Path If Kernel Supports It

Risk: low to medium.

This follows Tomeu's path and should not use local downstream raw scripts.

Goal:
- Run a known TFLite model through Teflon/Rocket.
- Confirm one-core baseline.
- Then confirm job-level multicore if the Rocket kernel driver is present.

Expected command shape depends on local build, but conceptually:
```text
LD_LIBRARY_PATH=<mesa-build>/src/gallium/targets/teflon \
<tflite-benchmark-or-demo> --external_delegate_path=<mesa-build>/src/gallium/targets/teflon/libteflon.so
```

Collect:
- FPS / latency
- `ROCKET_DEBUG=dbg_msgs` output
- `ROCKET_DEBUG=dump_bos` artifacts if needed
- number of submitted jobs if instrumented

Success criteria:
- A model runs through Rocket/Teflon.
- We know whether the local kernel has Rocket or only downstream RKNPU.

### Stage 3: Instrument Rocket Job Count

Risk: low.

Patch Mesa locally to print in `rkt_ml_subgraph_invoke`:
```text
operation index
operation type
reuse_weights_cbuf
number of split tasks
number of jobs appended
submit.job_count
```

Do not change scheduling behavior yet.

Expected observations:
- `reuse_weights_cbuf=true`: one job with multiple sequential tasks.
- `reuse_weights_cbuf=false`: multiple one-task jobs that can be scheduled across cores.

Success criteria:
- We have a concrete model/shape that produces multiple Rocket jobs.
- This model/shape is the best candidate for downstream raw tiling experiments.

### Stage 4: Replicate Rocket's Tiling Logic in Analysis Only

Risk: none.

For one convolution that Rocket splits, record per task:
- top/bottom slice
- input offset
- output offset
- input height
- output height
- padding
- regcmd count

Map that to local `examples/conv.py` register fields:
- source base address / input offset
- destination base address / output offset
- data cube dimensions
- padding fields
- line/surface stride
- CBUF bank allocation

Success criteria:
- A written register-field mapping exists.
- No downstream multicore ioctl was run.

### Stage 5: Safe Downstream Single-Core Tiled Prototype

Risk: low.

Implement one logical convolution or elementwise workload as several independent downstream core-0 submits:
```text
tile 0 -> core 0 -> output slice 0
tile 1 -> core 0 -> output slice 1
tile 2 -> core 0 -> output slice 2
```

This verifies tiling correctness without multicore.

For elementwise, split by contiguous vector ranges.
For convolution, split output rows/channels using Rocket's slice logic.

Success criteria:
- All tiled outputs reassemble correctly.
- This proves the register-level tiling is correct before multicore risk.

### Stage 5B: Martin-Style Matmul N-Axis Split

Risk: low for sequential core 0, medium for multiple official contexts, high for nonzero raw core masks.

Use a GEMV/GEMM shape that resembles RWKV token generation first:
```text
M=1, K=1024, N=1024
M=1, K=2048, N=2048
M=32, K=1024, N=1024
```

Tile along `N`:
```text
N tiles = 2: [0:512], [512:1024]
N tiles = 3: [0:342], [342:684], [684:1024]
N tiles = 4: three NPU tiles plus one CPU tile, matching Martin's 3 NPU + 1 CPU experiment
```

Safe validation order:
1. CPU reference full `A @ B`.
2. Sequential RKNN/raw core-0 tile 0, tile 1, tile 2.
3. Stitch output columns and compare to CPU.
4. Benchmark sequential tiled runtime against one full matmul.
5. Only then test multiple processes/contexts.

Expected outcome from Martin's slides:
- Tiling can use more cores, but overhead is high.
- GEMV (`M=1`) may still be slower than optimized CPU/GGML.
- Larger `M` should favor RKNN/NPU more.

Success criteria:
- Stitched output matches full CPU reference.
- Sequential tiling is correct before any risky core masks.
- We know whether the local RKNN matmul API can be forced to a stable single-core baseline.

### Stage 6: Safe Parallelism Through Processes, Core 0 Only

Risk: low to medium.

Run multiple independent processes that each submit a valid single-core core-0 job. This tests whether the downstream driver serializes safely and whether concurrent opens are stable, without nonzero-core masks.

Success criteria:
- No crash.
- Results correct.
- Timing tells whether downstream driver serializes all work anyway.

### Stage 7: Official RKNN Runtime Capture

Risk: medium.

Use `~/npu/ops_rknn/ioctl.gdb` to capture official runtime submits for a model using `rknn_set_core_mask` or RKLLM `num_npu_core=3`.

Record:
- `flags`
- `task_number`
- `core_mask`
- all `subcore_task[]`
- task register addresses and counts
- whether multiple submits are made

Question:
- Does official downstream runtime resemble Rocket jobs, or does it use hidden metadata and only one visible subcore range?

Success criteria:
- We know official downstream behavior before raw risky tests.

### Stage 8: First Risky Downstream Nonzero-Core Probe

Risk: high. Crash accepted from this stage onward.

Only run after Stage 5 passes and Stage 7 is understood.

Start with one independent tile, not split3:
```text
python experimental/multicore_elementwise.py --core-mask 0x2 --core-ranges 0:0:1 --allow-unsafe-submit
```

Try only one hypothesis at a time:
- `core_mask=0x2`, `subcore_task[0]=(0,1)`
- `core_mask=0x2`, `subcore_task[1]=(0,1)`
- `core_mask=0`, `subcore_task[1]=(0,1)`

Stop on first hang or wrong-output pattern.

Success criteria:
- Identify whether downstream raw ioctl can safely target core 1/2 at all.

### Stage 9: Risky Downstream Multi-Job Emulation

Risk: high.

If Stage 8 finds a safe nonzero-core pattern, emulate Rocket job-level parallelism with separate submits per independent tile:
```text
tile 0 -> core pattern A
tile 1 -> core pattern B
tile 2 -> core pattern C
```

Do not use multi-range `subcore_task[]` yet.

Success criteria:
- Correct outputs.
- No crash.
- Some speedup or confirmed serialization.

### Stage 10: Last Resort Multi-Range / PC-Chain Probe

Risk: very high.

Only run if official capture shows downstream runtime really uses multiple nonzero `subcore_task[]` entries.

Test order:
```text
2 tasks, 2 cores, no pingpong
2 tasks, 2 cores, PC-chain matching official layout
3 tasks, 3 cores, no pingpong
3 tasks, 3 cores, PC-chain matching official layout
```

Success criteria:
- No crash.
- Outputs correct.
- Captured layout matches official runtime.

## PC-Chain Update

The GEMM PC-chain work now gives a concrete regcmd reference that should be
carried into any later multicore experiment:

- `REG_PC_BASE_ADDRESS` and `REG_PC_REGISTER_AMOUNTS` are emitted as `TARGET_PC_REG` (`0x0101`) records.
- `REG_PC_OPERATION_ENABLE` is the `TARGET_PC` (`0x0081`) record.
- The 394x394x394 GEMM chain matches the raw capture qword-for-qword when built from decoded register writes.
- The last segment ends with `PC_REGISTER_AMOUNTS = 0`.

That means the older wrapper-style assumption is not good enough for elementwise
or multicore work.

Current elementwise probe results from `experimental/multicore_elementwise.py`:

```text
--core-ranges 0:0:1 --core-mask 0x1 --ops ADD --regcmd-mode offset
  ioctl returned EINVAL

--core-ranges 0:0:1 --core-mask 0x1 --ops ADD --regcmd-mode absolute
  ioctl returned EINVAL

--core-ranges core0 --core-mask 0x1 --ops ADD,MUL,SUB --regcmd-mode absolute
  ioctl returned EINVAL

--core-ranges core0 --core-mask 0x1 --ops ADD,MUL,SUB --regcmd-mode absolute --pc-chain --pc-chain-style rknnops
  submit returned 0
  task[0] PASS
  task[1] FAIL
  task[2] FAIL

--core-ranges core0 --core-mask 0x1 --ops ADD,ADD,ADD --regcmd-mode absolute --pc-chain --pc-chain-style rknnops
  submit timed out

--core-ranges core0 --core-mask 0x1 --ops ADD,MUL,SUB --regcmd-mode absolute --pc-chain --pc-chain-style mesa
  ioctl returned EINVAL

--core-ranges core0 --core-mask 0x1 --ops ADD,MUL,SUB --regcmd-mode absolute --pc-chain --pc-chain-style rknn-matmul
  ioctl returned EINVAL
```

Additional focused tests after separating descriptor `enable_mask` from regcmd
`REG_PC_OPERATION_ENABLE`:

```text
--pc-chain --pc-chain-style rknnops --regcmd-mode offset --pc-op-enable 0x18
  submit returned 0
  ADD,MUL,SUB: task[0] PASS, task[1]/task[2] FAIL

--pc-chain --pc-chain-style rknnops --regcmd-mode absolute --pc-op-enable 0x18
  ioctl returned EINVAL

--pc-chain --pc-chain-style rknnops --task-op-idx 1 --enable-mask 0x0d --submit-block --pc-op-enable 0x0d
  submit returned 0
  no outputs written

--pc-chain --pc-chain-style rknnops --task-op-idx 1 --enable-mask 0x0d --submit-block --pc-op-enable 0x18
  submit returned 0
  no outputs written

--pc-chain --pc-chain-style rknn-gemm --regcmd-mode offset --pc-op-enable 0x18 --ops ADD,ADD,ADD
  submit returned 0
  task[0] PASS, task[1]/task[2] FAIL

--pc-chain --pc-chain-style rknn-gemm --regcmd-mode offset --pc-op-enable 0x18 --ops ADD,MUL,SUB
  ioctl returned EINVAL
```

Interpretation:

- The chain still needs more than a PC tail.
- The `rknnops` amount style is the only one that gets mixed ADD/MUL/SUB past submit on this board.
- The GEMM-derived amount style `ceil(next_body_regs / 2) + 1` is valid enough to submit a uniform ADD chain, but it still does not make later tasks correct.
- The official `op_idx=1`, descriptor `enable_mask=0x0d`, and `RKNPU_JOB_BLOCK` combination is not compatible with this raw elementwise register body: it returns quickly with no output writes.
- The raw elementwise body still needs regcmd `PC_OPERATION_ENABLE = 0x18` for task 0 to write output.
- Later elementwise tasks are not executing correctly yet, so the next fix is
  likely task descriptor shape, op/enable mix, or the body register stream, not
  just the tail patch.
- Therefore, yes: single-core elementwise PC-chain correctness is the immediate
  prerequisite before any multicore test. Multicore should remain paused until
  `task0 -> task1 -> task2` works on core 0.

## What To Fix In Local Scripts

The local scripts should not try to make `split3` the primary path. Instead:

1. Add a Rocket-style abstraction:
```text
Job = one independent output-producing tile
Task = one regcmd stream inside a job
```

2. Add a tiled execution mode:
```text
--tiles 3
--tile-axis rows|channels|flat
--execution sequential-core0|separate-submits|unsafe-core-mask
```

3. Make `sequential-core0` the default validation path.

4. Keep `unsafe-core-mask` behind `--allow-unsafe-submit`.

5. Do not expose `split3` as a normal test. It is Stage 10 only.

For Martin-style RWKV/GEMV experiments, add a separate script instead of reusing `multicore_gemm.py` blindly:
```text
examples/matmul_nsplit.py
```

It should support:
```text
--m 1 --k 1024 --n 1024
--tiles 2|3|4
--backend raw-gemm|rknn-matmul-api
--execution sequential-core0|multi-process|unsafe-core-mask
--cpu-tail              # optional final tile on CPU
--allow-unsafe-submit   # required for nonzero raw core masks
```

The default must be:
```text
--execution sequential-core0
```

This matches Martin's successful approach while keeping the first experiments safe.

## Success Criteria

Short-term success:
- We can reproduce Rocket's multicore scheduling model from source.
- We can run or at least instrument Rocket/Teflon if the kernel driver is present.
- We can implement tiled downstream core-0 execution correctly.

Medium-term success:
- We know whether downstream raw ioctl can safely target core 1/2.
- We can run independent tiles across cores through a safe discovered pattern.

Long-term success:
- One logical workload is tiled, submitted as independent jobs, reassembled correctly, and is faster than single-core baseline.

## 2026-05-03 ADD PC-Chain Progress

Single-core ADD PC-chain is now working for the minimal raw DPU/RDMA body and
for the integrated elementwise helper. This unblocks the next stage of multicore
debugging, but it does not prove multicore by itself.

Verified commands:

```bash
python3 experimental/add_rawbuf_pcchain.py --amount-style body --regcmd-mode absolute --descriptor-amount segment --segment-elements 8
python3 experimental/min_add_pcchain.py
python3 experimental/add_pcchain.py --segment-elements 4096
python3 experimental/multicore_elementwise.py --n 4096 --ops ADD --task-count 3 --core-ranges core0 --core-mask 0x1 --pc-chain --pc-chain-style add --regcmd-mode absolute --submit-block --timeout 10000
```

Results:

```text
add_rawbuf_pcchain.py, 8 values/segment:
  ADD RAWBUF PCCHAIN PASS

min_add_pcchain.py, 4096 values/segment:
  ADD RAWBUF PCCHAIN PASS

add_pcchain.py, decoded 4096 values/segment:
  ADD PCCHAIN PASS

multicore_elementwise.py, core0 three-task PC-chain:
  submit ret=[0]
  ADD task[0] PASS
  ADD task[1] PASS
  ADD task[2] PASS
```

Important implementation details:

- `REG_PC_BASE_ADDRESS` / `REG_PC_REGISTER_AMOUNTS` use target `0x0101`.
- `REG_PC_OPERATION_ENABLE` uses target `0x0081`.
- ADD/DPU-RDMA PC-chain uses descriptor `regcfg_amount = body + 4-qword PC tail`.
- The working amount style for the minimal ADD chain is `PC_REGISTER_AMOUNTS = next body qword count`.
- Every chained segment re-arms both `DPU.S_POINTER = 0x0e` and `DPU_RDMA.S_POINTER = 0x0e`.
- `multicore_elementwise.py` now defaults to one core-0 task. Three-task tests must be explicit with `--task-count 3` or three comma-separated ops.
- `--separate-submits` now means sequential core-0 submits; it no longer maps task 1/2 to cores 1/2 by default.

Official RKNN runtime multicore is still not a solved path for this ADD model.
This command was tested:

```bash
RKNN_CORE_MASK=7 /home/orangepi/npu/ops_rknn/add 16
```

It failed in `rknn_run`:

```text
failed to submit!, op id: 2, flags: 0x5, task start: 13, task number: 9
rknn_run failed: -1
```

Core-0 RKNN ADD references still pass through:

```bash
python3 experimental/min_add_small.py
python3 experimental/min_add_medium.py
python3 experimental/min_add_large.py
```

Next safe multicore step:

1. Keep raw nonzero-core submit behind `--allow-unsafe-submit`.
2. First test split execution only after the core-0 PC-chain command above
   remains green.
3. Prefer official/runtime captures for any `core_mask != 1` path before
   guessing raw `subcore_task[]` layouts.

## 2026-05-03 Chain-Length Caution

`experimental/add_pcchain_length.py` was added to measure how many ADD PC-chain
segments can run on single core. The only trustworthy hardware runs are the
serialized ones. Do not trust results from multiple NPU submit commands launched
in parallel; those can race the downstream driver/runtime and leave later
outputs at zero even for configurations that previously passed.

Serialized evidence before the accidental parallel probes:

```text
segments=1 elements=4096 PASS
segments=2 elements=4096 PASS
segments=4 elements=4096 PASS
segments=8 elements=4096 PASS
segments=16 elements=4096 FAIL at task 14
```

Because later tests were accidentally launched in parallel, the boundary above
should be treated as provisional. The script now defaults to the known-stable
range only:

```bash
python3 experimental/add_pcchain_length.py --max-segments 8 --segment-elements 4096 --stop-on-fail
```

Lengths above 8 require an explicit opt-in:

```bash
python3 experimental/add_pcchain_length.py --lengths 16 --segment-elements 4096 --stop-on-fail --allow-risky-length
```

Use that only with physical reset access and after confirming no other RKNN/raw
NPU process is running. The current conservative conclusion is:

- 8 chained ADD segments at 4096 FP16 values/segment is verified stable.
- 16 segments is not verified; one serialized run failed near task 14.
- `experimental/add_pcchain_length.py` and `experimental/rknpu_common.py` use
  `/tmp/rk3588_npu_submit.lock` and refuse parallel submits instead of waiting.
- Multicore tests should not be resumed until the board/runtime can again pass
  the known-good core-0 PC-chain smoke test.

## 2026-05-03 Tiled Elementwise Multicore Plan

`experimental/multicore_elementwise.py` now has `--tile-flat` so multicore tests
use one logical ADD vector split into independent tiles instead of three
unrelated test tensors.

Safe baseline command, after the board/runtime is healthy:

```bash
python3 experimental/multicore_elementwise.py --tile-flat --ops ADD --n 12288 --tiles 3 --execution sequential-core0 --timeout 10000
```

Fresh post-reboot tiled ADD evidence:

```bash
python3 experimental/multicore_elementwise.py --tile-flat --ops ADD --n 12288 --tiles 3 --execution sequential-core0 --timeout 10000
```

```text
execution=sequential-core0 submit ret=[0, 0, 0]
ADD tile[0] PASS
ADD tile[1] PASS
ADD tile[2] PASS
ADD tiled_output PASS
```

Core-0 one-submit PC-chain tiled validation also passed:

```bash
python3 experimental/multicore_elementwise.py --tile-flat --ops ADD --n 12288 --tiles 3 --execution pc-chain-core0 --timeout 10000
```

```text
flags=0x7
core_mask=0x1
task_number=3
subcore_task[0]=(0,3)
execution=pc-chain-core0 submit ret=[0]
ADD tile[0] PASS
ADD tile[1] PASS
ADD tile[2] PASS
ADD tiled_output PASS
```

Raw split-core tiled ADD now has a working evidence path. Guarded command:

```bash
timeout 25s python3 experimental/multicore_elementwise.py --tile-flat --ops ADD --n 12288 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit
```

Working submit layout:

```text
flags=0x7
core_mask=0x7
task_number=3
subcore_task[0]=(0,0)
subcore_task[1]=(0,0)
subcore_task[2]=(0,1)
subcore_task[3]=(1,1)
subcore_task[4]=(2,1)
```

```text
execution=unsafe-split3 submit ret=[0]
ADD tile[0] PASS
ADD tile[1] PASS
ADD tile[2] PASS
ADD tiled_output PASS
```

Larger split-core tiled ADD also passed:

```bash
timeout 25s python3 experimental/multicore_elementwise.py --tile-flat --ops ADD --n 24576 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit
```

```text
tile[0] n=8192 PASS
tile[1] n=8192 PASS
tile[2] n=8192 PASS
ADD tiled_output PASS
```

Current conclusion:

- The safe model is confirmed: split one logical flat ADD into independent
  output tiles.
- The raw vendor submit that works on this board is the RK3588 tail layout:
  `subcore_task[2]=(0,1)`, `[3]=(1,1)`, `[4]=(2,1)`.
- Each tile is a standalone DPU/RDMA ADD task with absolute `regcmd_addr`.
- `unsafe-split3` uses one submit with PC-chain tails and
  `PC | BLOCK | PINGPONG` flags.
- Keep the explicit `--allow-unsafe-submit` guard. This path works for tiled ADD
  now, but it is still a raw downstream-driver path, not a general proof for all
  ops or GEMM.

Additional tiled elementwise coverage:

```bash
python3 experimental/multicore_elementwise.py --tile-flat --ops MUL --n 12288 --tiles 3 --execution sequential-core0 --timeout 10000
python3 experimental/multicore_elementwise.py --tile-flat --ops MUL --n 12288 --tiles 3 --execution pc-chain-core0 --timeout 10000
timeout 25s python3 experimental/multicore_elementwise.py --tile-flat --ops MUL --n 12288 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit
```

Results:

```text
MUL sequential-core0: PASS, all 3 tiles and stitched output
MUL pc-chain-core0:   PASS, all 3 tiles and stitched output
MUL unsafe-split3:    PASS, all 3 tiles and stitched output
```

```bash
python3 experimental/multicore_elementwise.py --tile-flat --ops SUB --n 12288 --tiles 3 --execution sequential-core0 --timeout 10000
python3 experimental/multicore_elementwise.py --tile-flat --ops SUB --n 12288 --tiles 3 --execution pc-chain-core0 --timeout 10000
timeout 25s python3 experimental/multicore_elementwise.py --tile-flat --ops SUB --n 12288 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit
```

Results:

```text
SUB sequential-core0: PASS, all 3 tiles and stitched output
SUB pc-chain-core0:   PASS, all 3 tiles and stitched output
SUB unsafe-split3:    PASS, all 3 tiles and stitched output
```

Updated supported scope:

- Flat tiled ADD/MUL/SUB are verified through sequential core-0, core-0
  PC-chain, and raw split3 on this board.
- The split3 evidence is for independent flat tiles only. It should not be
  generalized to chained dependent tasks, convolution, or GEMM without separate
  tile-specific validation.

`MAX` is also verified:

```bash
python3 experimental/multicore_elementwise.py --tile-flat --ops MAX --n 12288 --tiles 3 --execution sequential-core0 --timeout 10000
python3 experimental/multicore_elementwise.py --tile-flat --ops MAX --n 12288 --tiles 3 --execution pc-chain-core0 --timeout 10000
timeout 25s python3 experimental/multicore_elementwise.py --tile-flat --ops MAX --n 12288 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit
```

Results:

```text
MAX sequential-core0: PASS, all 3 tiles and stitched output
MAX pc-chain-core0:   PASS, all 3 tiles and stitched output
MAX unsafe-split3:    PASS, all 3 tiles and stitched output
```

The verified flat tiled elementwise set is now:

```text
ADD, MUL, SUB, MAX
```

Larger split3 tile size also passes for the verified set:

```bash
timeout 25s python3 experimental/multicore_elementwise.py --tile-flat --ops ADD --n 24576 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit
timeout 25s python3 experimental/multicore_elementwise.py --tile-flat --ops MUL --n 24576 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit
timeout 25s python3 experimental/multicore_elementwise.py --tile-flat --ops SUB --n 24576 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit
timeout 25s python3 experimental/multicore_elementwise.py --tile-flat --ops MAX --n 24576 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit
```

Results:

```text
ADD 3x8192 split3: PASS
MUL 3x8192 split3: PASS
SUB 3x8192 split3: PASS
MAX 3x8192 split3: PASS
```

Next expansion targets:

- GEMM N-axis tiling

`NEG` is verified:

```bash
python3 experimental/multicore_elementwise.py --tile-flat --ops NEG --n 12288 --tiles 3 --execution sequential-core0 --timeout 10000
python3 experimental/multicore_elementwise.py --tile-flat --ops NEG --n 12288 --tiles 3 --execution pc-chain-core0 --timeout 10000
timeout 25s python3 experimental/multicore_elementwise.py --tile-flat --ops NEG --n 12288 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit
```

Results:

```text
NEG sequential-core0: PASS, all 3 tiles and stitched output
NEG pc-chain-core0:   PASS, all 3 tiles and stitched output
NEG unsafe-split3:    PASS, all 3 tiles and stitched output
```

`FDIV` is verified with `--atol 0.2`:

```bash
python3 experimental/multicore_elementwise.py --tile-flat --ops FDIV --n 12288 --tiles 3 --execution sequential-core0 --timeout 10000 --atol 0.2
python3 experimental/multicore_elementwise.py --tile-flat --ops FDIV --n 12288 --tiles 3 --execution pc-chain-core0 --timeout 10000 --atol 0.2
timeout 25s python3 experimental/multicore_elementwise.py --tile-flat --ops FDIV --n 12288 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit --atol 0.2
```

Results:

```text
FDIV sequential-core0: PASS, all 3 tiles and stitched output, max_diff=0.0039
FDIV pc-chain-core0:   PASS, all 3 tiles and stitched output, max_diff=0.0039
FDIV unsafe-split3:    PASS, all 3 tiles and stitched output, max_diff=0.0039
```

The verified flat tiled elementwise set is now all ops currently exposed by
`experimental/multicore_elementwise.py`:

```text
ADD, MUL, SUB, MAX, NEG, FDIV
```

Remaining major target:

- GEMM N-axis tiling

## 2026-05-03 GEMM Multicore Status

`experimental/multicore_gemm.py` now imports the local GEMM helper from
`examples/gemm.py`, but it is not yet a usable multicore path. Safe core0 smoke:

```bash
timeout 25s python3 experimental/multicore_gemm.py --m 32 --n 32 --k 32 --core-ranges core0 --core-mask 0x1 --regcmd-mode absolute --timeout 10000
```

Result:

```text
task[0] GEMM 32x32x32 regcfg_amount=45 regcmd_addr=...
task[1] GEMM 32x32x32 regcfg_amount=45 regcmd_addr=...
task[2] GEMM 32x32x32 regcfg_amount=45 regcmd_addr=...
TimeoutError: [Errno 110] Connection timed out
```

Conclusion:

- The existing GEMM helper is still the old three-independent-descriptor shape.
- It does not yet reproduce the working GEMM PC-chain approach.
- Do not run GEMM split-core until a core0 GEMM PC-chain/tiled baseline passes.

Next GEMM work:

1. Use the working `examples/gemm.py` PC-chain path as the tiled logical GEMM
   baseline.
2. Validate one logical GEMM split along N on core 0.
3. Validate the same N tiles as independent sequential core0 submits.
4. Only then try `rk3588-tricore-tail` split3 for GEMM.

Update: generic small GEMM N-axis tiling now works for independent tiles. First,
the helper was fixed to default to one safe core-0 task; the previous three-task
core0 submit timeout was caused by submitting multiple GEMM descriptors together
without a valid chain. Single-task smoke:

```bash
timeout 25s python3 experimental/multicore_gemm.py --m 32 --n 32 --k 32 --task-count 1 --core-ranges core0 --core-mask 0x1 --regcmd-mode absolute --timeout 10000
```

Result:

```text
flags=0x5
core_mask=0x1
task_number=1
GEMM task[0] PASS max_diff=0.0044
```

Three independent 32x32x32 GEMMs also pass on split3:

```bash
timeout 25s python3 experimental/multicore_gemm.py --m 32 --n 32 --k 32 --task-count 3 --core-ranges split3 --core-mask 0x7 --subcore-layout rk3588-tricore-tail --regcmd-mode absolute --timeout 10000 --allow-unsafe-submit
```

Result:

```text
flags=0x5
core_mask=0x7
task_number=3
subcore_task[2]=(0,1)
subcore_task[3]=(1,1)
subcore_task[4]=(2,1)
GEMM task[0] PASS
GEMM task[1] PASS
GEMM task[2] PASS
```

Logical N-axis tiled GEMM now has a working baseline. This treats `--n` as total
N and splits the weight/output columns across three independent GEMM tasks:

```bash
timeout 25s python3 experimental/multicore_gemm.py --tile-n --m 32 --n 96 --k 32 --tiles 3 --execution sequential-core0 --timeout 10000 --atol 0.1
```

Result:

```text
execution=sequential-core0 submit ret=[0, 0, 0]
GEMM task[0] PASS max_diff=0.0052
GEMM task[1] PASS max_diff=0.0054
GEMM task[2] PASS max_diff=0.0065
GEMM tiled_output PASS max_diff=0.0065
```

Split3 N-axis tiled GEMM also passes:

```bash
timeout 25s python3 experimental/multicore_gemm.py --tile-n --m 32 --n 96 --k 32 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit --atol 0.1
```

Result:

```text
flags=0x5
core_mask=0x7
task_number=3
subcore_task[2]=(0,1)
subcore_task[3]=(1,1)
subcore_task[4]=(2,1)
execution=unsafe-split3 submit ret=[0]
GEMM task[0] PASS
GEMM task[1] PASS
GEMM task[2] PASS
GEMM tiled_output PASS max_diff=0.0065
```

Current GEMM scope:

- Verified: small 32x96x32 logical GEMM split into three N-axis tiles.
- Verified: split3 tail layout works for independent GEMM tasks.
- Not yet verified: larger GEMM/GEMV tile sizes, 394x394x394-style PC-chain
  row splits, or performance speedup.

Important failing large-shape check:

```bash
timeout 25s python3 experimental/multicore_gemm.py --m 395 --n 395 --k 295 --task-count 1 --core-ranges core0 --core-mask 0x1 --regcmd-mode absolute --timeout 10000 --atol 0.2
```

Result:

```text
task[0] GEMM 395x395x295 regcfg_amount=45 regcmd_addr=...
TimeoutError: [Errno 110] Connection timed out
```

Conclusion:

- `395x395x295` is still not solved.
- It fails before multicore: one single core, one descriptor times out.
- The new split3 success only covers independent GEMM tiles whose individual
  tile shape already works, such as `32x32x32`.
- `395x395x295` needs the GEMM register/packing path fixed or rewritten around
  the known-good RKNN/GEMM PC-chain capture style before multicore can help.
- Since `N=395` is not divisible by 3, N-axis tiling also needs uneven tile
  support, for example `132 + 132 + 131`, after each tile shape is proven on
  core 0.

`395x395x395` has the same current failure mode:

```bash
timeout 25s python3 experimental/multicore_gemm.py --m 395 --n 395 --k 395 --task-count 1 --core-ranges core0 --core-mask 0x1 --regcmd-mode absolute --timeout 10000 --atol 0.2
```

Result:

```text
task[0] GEMM 395x395x395 regcfg_amount=45 regcmd_addr=...
TimeoutError: [Errno 110] Connection timed out
```

Conclusion:

- `395x395x395` is not covered by the small N-axis tiled GEMM success.
- It fails before multicore, in the single-task core0 path.
- This likely needs the cleaned-up tiled PC-chain path in `examples/gemm.py`, not
  the older one-descriptor GEMM experiments.

Historical specialized GEMM PC-chain check:

```bash
timeout 25s python3 experimental/gemm_pcchain.py --constant-data --mode core0 --timeout 10000
timeout 25s python3 experimental/gemm_pcchain.py --constant-data --mode official --timeout 10000
timeout 25s python3 experimental/gemm_pcchain.py --mode official --seed 0 --timeout 10000
```

Results:

```text
394x394x394 pcchain core0 constant:    PASS max_diff=0.000000
394x394x394 pcchain official constant: PASS max_diff=0.000000
394x394x394 pcchain official seed 0:   PASS max_diff=0.031067
```

Interpretation:

- The hardcoded RKNN-style GEMM PC-chain reproducer worked on core0.
- The existing `official` mode also passes. It uses the RKNN-style submit shape:
  `task_number = TASKS * 3`, `core_mask = 0`, and duplicate
  `subcore_task[0..2] = (0,TASKS)`.
- This remains useful historical evidence for the captured RKNN layout, but the
  active GEMM PC-chain implementation is now `examples/gemm.py`.

This should run three independent tile submits on core 0, stitch the output in
host memory, and verify the full logical vector. This is the Rocket-style model:

```text
one logical workload
  -> tile 0: valid standalone ADD task
  -> tile 1: valid standalone ADD task
  -> tile 2: valid standalone ADD task
host verifies concat(tile outputs)
```

Core-0 one-submit PC-chain tiled validation:

```bash
python3 experimental/multicore_elementwise.py --tile-flat --ops ADD --n 12288 --tiles 3 --execution pc-chain-core0 --timeout 10000
```

Only after both commands pass should raw split-core be tested:

```bash
python3 experimental/multicore_elementwise.py --tile-flat --ops ADD --n 12288 --tiles 3 --execution unsafe-split3 --timeout 10000 --allow-unsafe-submit
```

`unsafe-split3` forces:

```text
core_ranges      = split3
core_mask        = 0x7
subcore_layout   = rk3588-tricore-tail
pc_chain         = add
regcmd_mode      = absolute
submit flags     = PC | BLOCK | PINGPONG
```

Do not run `unsafe-split3` while the board is segfaulting or after failed
parallel submit experiments. The immediate next evidence needed is a fresh
post-reboot pass of:

```bash
python3 experimental/multicore_elementwise.py --tile-flat --ops ADD --n 12288 --tiles 3 --execution sequential-core0 --timeout 10000
```
