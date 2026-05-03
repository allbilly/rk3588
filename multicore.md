# RK3588 NPU Multicore Plan

## Current Conclusion

Multicore on RK3588 is definitely possible, but the working open-source model is **not** the downstream Rockchip `rknpu_submit.core_mask + subcore_task[]` model we were guessing against.

Tomeu Vizoso's Rocket/Mesa path uses a different UAPI:
- Userspace submits an array of independent jobs with `DRM_IOCTL_ROCKET_SUBMIT`.
- The kernel has one scheduler per NPU core.
- Userspace does **not** choose core 0/1/2 directly.
- The kernel scheduler assigns each job to a core.
- Multiple tasks inside one job execute sequentially on the same core to preserve SRAM/CBUF residency.
- Parallelism comes from submitting multiple independent jobs, not from manually filling `subcore_task[1]` or `[2]`.

That means our local raw ioctl experiments should be redesigned around **job-level parallelism** and **tiling into independent jobs**, not around one downstream submit with multiple `subcore_task[]` ranges.

## Crash Incident

Running `examples/multicore_elementwise.py` with `--core-mask 0x7 --core-ranges split3` caused a full kernel panic. Later, even some separate raw submits aimed at core 1/2 caused timeout or hard lock depending on task object layout.

Confirmed unsafe patterns on the current downstream driver:
- Single downstream `rknpu_submit` with multiple nonzero `subcore_task[]` entries.
- Raw `core_mask=0x2` / `0x4` experiments without official runtime setup.
- Shifting `task_obj_addr` into the middle of the task BO.

These crashes do not disprove multicore. They show that the downstream ABI is not the same abstraction as Rocket and should not be guessed into shape.

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
- For matmul/GEMV, the first serious multicore experiment should be **N-axis tiling**, not `examples/multicore_gemm.py --core-ranges split3`.
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
- `examples/multicore_elementwise.py` defaults to `core0`, `core_mask=0x1`.
- `examples/multicore_gemm.py` defaults to `core0`, `core_mask=0x1`.
- Raw nonzero-core and multi-range paths require `--allow-unsafe-submit`.
- `examples/multicore_probe.py` requires explicit opt-in for risky probes.
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
python examples/multicore_elementwise.py --core-mask 0x2 --core-ranges 0:0:1 --allow-unsafe-submit
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
