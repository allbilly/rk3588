# Plan: Clean `examples/kernel_6_18/conv.py`

## Goal

Make `examples/kernel_6_18/conv.py` a clean RKNN-style standalone example, with
`examples/gemm.py` as the style reference.

Scope is only `examples/kernel_6_18/conv.py`. Do not spend this cleanup cycle on
top-level `examples/conv.py`.

Target shape for the next work:

```text
b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid
```

Desired final shape of the file:

- short readable flow: inputs -> RKNN descriptor plan -> decoded registers ->
  guarded Rocket submit -> unpack -> compare
- decoded register construction, no opaque conv hex blob
- no CPU/GPU offload for the intended NPU path
- minimal helper families, close to the `examples/gemm.py` style
- comments preserved while editing

## Review

The current `plan.md` was too long because it mixed a cleanup plan, a probe log,
and several rejected submit theories. The useful content is narrower:

- `examples/kernel_6_18/conv.py` already has the right raw ingredients:
  `run_conv()`, `plan_conv_descriptors()`, decoded register builders, RKNN
  direct-spatial descriptors, and unsafe hardware gates.
- The file is not clean yet because it still carries multiple competing
  direct-spatial task policies: `pc_root6`, `single_stream`, `sparse_single`,
  plus older fallback/materializer branches.
- Cosmetic cleanup is not the next step. Removing those branches before the
  direct-spatial submit model is proven would either hide the real issue or make
  later comparison harder.
- The cleanup should happen after one local Rocket submit model matches RKNN
  semantics for the target shape.

## Main Blocker

The blocker is not register decoding or high-level tiling anymore. The blocker
is the local Rocket PC-chain/task execution model for the RKNN direct-spatial
stream.

Known failures:

- `single_stream` submitted without crashing, but produced wrong output.
- `sparse_single` matched more of the RKNN sparse backing layout offline, but
  also produced wrong output on hardware.
- `rocket_record_amounts` matched the offline per-record Rocket inverse
  candidate and runtime parity verifier, but produced wrong output on hardware:

```text
b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid FAIL max_diff=231.3182
```

- A failed `sparse_single` probe briefly polluted NPU state: the first
  post-probe `simple_add.py` returned zeros, while a retry passed.
- The failed `rocket_record_amounts` probe repeated that state-pollution
  pattern: first post-probe `simple_add.py` returned zeros, second retry passed.

Current best hypothesis:

```text
conv.py is close on descriptor/register bytes, but still wrong on PC/task amount
semantics or vendor task selection semantics.
```

Specific evidence:

- RKNN task records carry fields local Rocket does not expose directly:
  `enable_mask`, `int_mask`, `int_clear`, `op_idx`, `regcfg_amount`,
  `regcmd_addr`.
- Local Rocket task ABI is minimal: `regcmd` and `regcmd_count`.
- RKNN logs show `task_number=6` and subcore task metadata; local Rocket does
  not provide the same submit surface.
- Current offline verifier output says vendor auto-core selection resolves to
  one active queue entry for this local evidence:

```text
selected subcore_task[0]=(0, 2)
conv@0x0+108q[0:108]
conv@0x380+108q[112:220]
```

- The sparse backing stream has embedded RKNN `PC_BASE_ADDRESS` links and
  `PC_REGISTER_AMOUNTS` values. Those values do not directly map to the local
  Rocket PC amount model.
- Offline verifier work found that a simple Mesa/Rocket PC amount rewrite cuts
  a record, while the stronger per-record Rocket inverse candidate passes
  offline with `53/51/12` style amounts.
- Hardware rejected that stronger candidate, so changing embedded
  `PC_REGISTER_AMOUNTS` alone is not the missing semantic.
- `--verify-rocket-task-mapping-proof` still fails: no observed vendor submit
  interpretation is a complete, unambiguous, segment-aligned local Rocket job.
- `--verify-local-rocket-pc-commit-surface-gap` now passes and makes the
  current gap explicit. Vendor `rknpu_job_subcore_commit_pc()` programs:

```text
PC_DATA_ADDR
PC_DATA_AMOUNT
INT_MASK
INT_CLEAR
PC_TASK_CONTROL
PC_DMA_BASE_ADDR
PC_OP_EN
```

For the observed selected range it derives:

```text
subcore_task[0]=(0,2)
PC_DATA_AMOUNT=0x37
PC_TASK_CONTROL=0x6002
INT_MASK=0x300
INT_CLEAR=0x300
first_fetch_span=[0:112]
```

Local `drm_rocket_task` exposes only:

```text
regcmd
regcmd_count
```

So stream-only rewrites cannot express the vendor PC commit surface directly.
- Upstream Rocket `rocket_job.c` confirms the local kernel synthesizes its own
  PC commit per `drm_rocket_task`:

```text
PC_REGISTER_AMOUNTS = (regcmd_count + 1) / 2 - 1
PC_TASK_CONTROL uses TASK_NUMBER(1) and TASK_PP_EN(1)
PC_TASK_DMA_BASE_ADDR = 0
PC_OPERATION_ENABLE toggles 1 then later 0
```

This means every local Rocket task is its own single-task PC commit. The next
candidate must choose Rocket task spans that are valid under that model, not
under vendor `task_number=2` or `task_number=6` semantics.
- `--verify-runtime-rocket-policy-commit-models` now prints the exact local
  commit model for each current diagnostic policy:

```text
single_stream          task_count=1 regcmd_count=1280 PC_DATA_AMOUNT=639  TASK_CONTROL=0x6001
pc_root6               task_count=6 regcmd_count=224/256/.../32 TASK_CONTROL=0x6001
sparse_single          task_count=1 regcmd_count=1498 PC_DATA_AMOUNT=748  TASK_CONTROL=0x6001
rocket_record_amounts  task_count=1 regcmd_count=1498 PC_DATA_AMOUNT=748  TASK_CONTROL=0x6001
```

The key difference is now explicit: local Rocket PC data amount decodes to
`regcmd_count`, while vendor RKNN commit fetches `regcfg_amount + 4`.
- `--verify-rocket-segment-aligned-pc-partition-rejection` now rejects the
  obvious "split at boundaries that are both PC targets and compiler segment
  boundaries" idea:

```text
pc_targets_in_stream=[0,112,336,448,592,704,848,960,1104,1216]
segment_pc_candidate_boundaries=[0,960,1280]
boundary 960 starts row9:pc_tail
executable_boundaries=[0,1280]
```

The only internal candidate boundary starts at a PC tail, not a descriptor body,
so it is not a valid Rocket task root. Excluding that boundary leaves only the
already-rejected single-stream model.

Therefore, the next blocker to work is:

```text
identify the RKNN/vendor task-root, unit-enable, interrupt, or PC-commit
semantic that local Rocket still does not model for the sparse direct-spatial
stream.
```

## Next Work

1. Keep the target narrow.

Only work on:

```text
b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid
```

Do not broaden shape support and do not refactor fallback paths until this
target path has a passing model or a stronger negative proof.

2. Keep the `rocket_record_amounts` diagnostic policy as rejected evidence.

`examples/kernel_6_18/conv.py` now has a diagnostic policy that starts from
`direct_spatial_sparse_backing_stream(descs)` and patches only embedded
`PC_REGISTER_AMOUNTS` to the Rocket per-record inverse candidate.

Implemented name:

```text
rocket_record_amounts
```

It must stay guarded and rejected. Do not re-submit it unchanged.

3. Keep runtime parity verification for the rejected candidate.

`experimental/compare_conv_direct_regs.py` can prove that
`examples/kernel_6_18/conv.py` emits exactly the same `rocket_record_amounts`
stream and task span model as the offline candidate.

Minimum checks:

- sparse record starts stay unchanged
- embedded `PC_BASE_ADDRESS` targets still land on valid record starts
- embedded `PC_REGISTER_AMOUNTS` match the per-record Rocket inverse candidate
- fetched ranges do not cut records and do not fetch sparse gaps
- runtime task span/count is exactly what the local Rocket submit path will use

4. Re-run offline gates.

Before any direct-spatial hardware submit:

```sh
python3 -m py_compile examples/kernel_6_18/conv.py experimental/compare_conv_direct_regs.py
python3 experimental/compare_conv_direct_regs.py sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1 --verify-rocket-record-amounts-candidate
python3 experimental/compare_conv_direct_regs.py sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1 --verify-runtime-rocket-record-amounts-stream
python3 experimental/compare_conv_direct_regs.py sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1 --verify-runtime-pc-root6-guard
python3 examples/kernel_6_18/simple_add.py
```

These passed before the rejected `rocket_record_amounts` hardware probe.

5. Use RKNN only for missing evidence.

If local evidence is insufficient, inspect RKNN behavior read-only on:

```sh
ssh orangepi@192.168.192.36
cd ~/npu/ops_rknn
```

Useful questions there:

- What task records does RKNN submit for the target model?
- Which regcmd records are roots versus PC-linked children?
- Does RKNN patch PC amounts at runtime, or are they compiler-emitted?
- Does RKNN rely on `enable_mask`/interrupt semantics that local Rocket cannot
  express directly?
- Does RKNN execute all six task roots, only selected roots, or a vendor
  scheduler-expanded graph that local one-task Rocket submit does not mirror?

Do not kill long-running NPU processes.

If the evidence points to a local Rocket driver limitation rather than a
register-stream problem, use the remote RKNN driver on
`orangepi@192.168.192.36:~/npu/ops_rknn` as the comparison target. Keep that
comparison controlled: prefer existing RKNN models/logs first, then run the
smallest matching RKNN conv only when needed. Do not use remote RKNN success as
proof that local Rocket submit semantics are solved; use it to separate
compiler/register correctness from Rocket ABI translation issues.

Current remote RKNN comparison:

```sh
cd ~/npu/ops_rknn
python3 gen_conv2d_models.py --custom --out-dir models \
  --batch 1 --in-ch 160 --height 40 --width 40 \
  --out-ch 320 --k-h 3 --k-w 3 --groups 1 \
  --name conv2d_target_160x40_320x3x3
./conv2d_multi --case conv2d_target_160x40_320x3x3 \
  1 160 40 40 320 3 3 1 target_160x40_320x3x3
```

The exact model runs through the vendor RKNN driver and reports native shape
`1 20 40 40 8 -> 1 40 38 38 8`. The harness marks it failed because a few
absolute errors exceed its `3.5e-2` tolerance, but the observed errors are about
`0.04`, not the `231+` local Rocket direct-spatial failure. Treat this as
evidence that the large local failure is a Rocket submit/PC translation problem,
not a basic compiler/register-shape problem.

`tmp_remote_rknn/conv2d_target_160x40_320x3x3.rknn` was copied back locally and
`--verify-remote-exact-export-equivalent` passes:

```text
local_size=948726 remote_size=948726
local_stream_qwords=1280 remote_stream_qwords=1280
PASS remote exact export equivalence
```

The full file hashes differ, but the decoded compiler descriptor stream is the
same. This makes the comparison valid at the register-command level.

Read-only remote GDB submit tracing with
`experimental/remote_rknn_submit_readonly.gdb` on the generated exact model
confirms the runtime submit metadata is the same shape as the saved local RKNN
logs:

```text
SUBMIT[1] cmd=0xc0686441 size=104
flags=0x5 timeout=6000 task_start=0 task_number=6 core_mask=0x0 fence_fd=-1
subcore_task[0]={task_start=0, task_number=2}
subcore_task[1]={task_start=0, task_number=2}
subcore_task[2]={task_start=0, task_number=2}
subcore_task[3]={task_start=0, task_number=0}
subcore_task[4]={task_start=0, task_number=0}
```

The task object pointer is a kernel object address and is not directly readable
from userspace in this read-only trace, so task-record contents should still use
the existing GEM dump logs or a separate GEM-map trace.

Next offline/remote focus:

- Trace or reconstruct how RKNN turns `subcore_task[0]=(0,2)` plus embedded PC
  links into real work. The current local one-task sparse backing model does
  not mirror that.
- Determine whether local Rocket should submit only the two selected root
  records, several root records, or a rewritten PC-root stream rather than one
  giant sparse task.
- Check whether the vendor-only `enable_mask`/`int_mask` fields change unit
  enable, interrupt clear, or PC commit behavior in a way local Rocket must
  encode inside the regcmd stream.
- Inspect or reconstruct local Rocket kernel-side submit handling. The next
  proof must explain how Rocket derives the equivalent of vendor
  `PC_DATA_AMOUNT`, `PC_TASK_CONTROL`, interrupt clear/mask, and `PC_OP_EN` from
  `{regcmd, regcmd_count}` before another candidate is worth probing.
- Specifically, model Rocket's per-task `TASK_NUMBER(1)` behavior against the
  RKNN PC-linked stream. A candidate that needs one task to execute multiple
  vendor task records is suspect unless its embedded PC links prove that chain
  continues correctly under Rocket's single-task commit.
- Next candidate work should start by deriving task spans from Rocket commit
  semantics first, then checking whether the resulting stream preserves RKNN
  PC-link targets and does not cut descriptor row segments. Avoid more
  `PC_REGISTER_AMOUNTS`-only rewrites.
- The segment-aligned PC-target partition is now rejected offline. The next
  useful route is remote RKNN comparison or source tracing to determine whether
  the vendor driver depends on hidden PC task-table execution semantics that
  Rocket's `TASK_NUMBER(1)` commit cannot reproduce from userspace.

6. Hardware probe only after a new proof changes.

Do not rerun `single_stream`, `sparse_single`, or `rocket_record_amounts`
unchanged. For any new direct-spatial probe, require both gates:

```sh
RK3588_CONV_DIRECT_SPATIAL=1 \
RK3588_CONV_DIRECT_SPATIAL_UNSAFE=1 \
RK3588_CONV_DIRECT_SPATIAL_TASKS=<new-policy> \
python3 examples/kernel_6_18/conv.py b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid
```

Run `python3 examples/kernel_6_18/simple_add.py` before and after the probe.

## Cleanup After Blocker

After the target direct-spatial path passes, simplify
`examples/kernel_6_18/conv.py` in this order:

1. Keep one canonical RKNN/Rocket direct-spatial policy.
2. Remove or quarantine rejected policies: `pc_root6`, `single_stream`,
   `sparse_single`, and any temporary `rocket_record_amounts` diagnostic naming.
3. Collapse planner/materializer branches that only exist because the
   direct-spatial path was unproven.
4. Keep decoded register builders explicit, but move probe-only diagnostics out
   of the main runnable path.
5. Recompare structure against `examples/gemm.py`: short constants, small
   helpers, one obvious execution path, clear final PASS/FAIL.

## Done Criteria

The cleanup goal is done when:

- the target shape passes through the NPU direct-spatial path
- `examples/kernel_6_18/conv.py` has no conv hex blob
- the file reads like `examples/gemm.py` in structure and top-level flow
- rejected direct-spatial policies are removed or isolated outside the clean
  example path
- `py_compile` and `examples/kernel_6_18/simple_add.py` pass

Until then, the main blocker remains the missing RKNN/vendor task execution
semantic for the sparse direct-spatial stream.

## Root Cause: Rocket UAPI Gap

The upstream Linux Rocket driver (`drivers/accel/rocket/`) cannot express the
RKNN vendor task semantics needed for direct-spatial conv.

### Current Rocket UAPI (`drm_rocket_task`)

```c
struct drm_rocket_task {
    u64 regcmd;
    u32 regcmd_count;
};
```

`rocket_job_hw_submit` hard-codes per-job values:

```c
rocket_pc_writel(core, OPERATION_ENABLE, 1);
rocket_pc_writel(core, INTERRUPT_MASK, DPU_0 | DPU_1);
rocket_pc_writel(core, TASK_CON, PC_TASK_CON_TASK_NUMBER(1));
```

### What RKNN vendor needs per task

| Capability | Rocket today | RKNN `rknpu_task` |
|---|---|---|
| `enable_mask` | `OP_EN(1)` global | Per-task: `0x0d` vs `0x60` |
| `int_mask`/`int_clear` | Global `DPU_0\|DPU_1` | Per-task: `0x300` vs `0xc00` |
| `TASK_NUMBER` | Hard-coded `1` | Up to `2` per subcore |
| Multi-root PC chain | 1 `regcmd` per task | N task records in GEM, each with own `regcmd_addr` |

### What a kernel UAPI patch would add

```c
struct drm_rocket_task {
    u64 regcmd;
    u32 regcmd_count;
    u32 enable_mask;    // → OPERATION_ENABLE
    u32 int_mask;       // → INTERRUPT_MASK
    u32 int_clear;      // → INTERRUPT_CLEAR
    u32 task_number;    // → TASK_CON_TASK_NUMBER
};
```

And `rocket_job_hw_submit` would write per-task register values instead of
hard-coded globals. Until this patch lands in mainline, conv shapes requiring
multi-task execution with per-task enable/interrupt masks cannot be expressed
through the local Rocket userspace ABI.

### Practical consequence for conv.py

Every direct-spatial probe that failed locally (`single_stream`,
`sparse_single`, `rocket_record_amounts`, 3-task grouping) failed because
the Rocket driver cannot program the correct per-task `enable_mask`/`int_mask`
or `TASK_NUMBER`. The vendor rknpu driver does all of this internally from
the `struct rknpu_task` array in the task GEM object.

The offline verifiers and `.rknn` model parser are complete — the register
stream is correct. The blocker is purely at the UAPI/driver level.

---

## Appendix: progress.md (snapshot)

Below is the full content of the earlier `progress.md` document, which captured
the state of the cleanup effort at an intermediate milestone. It is preserved
here for reference.

# Progress: `examples/kernel_6_18/conv.py` RKNN-style cleanup

## Current goal

Make `examples/kernel_6_18/conv.py` a clean standalone RKNN-style convolution
example, using `examples/gemm.py` as the style reference.

The cleanup target is not top-level `examples/conv.py`. Work should stay focused
on:

```text
examples/kernel_6_18/conv.py
```

The active functional target shape is:

```text
b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid
```

The desired final file shape is:

- decoded register construction, no opaque conv hex blob
- short readable flow like `examples/gemm.py`
- clear path: inputs -> descriptor plan -> decoded registers -> guarded NPU
  submit -> unpack -> compare
- no CPU/GPU offload for the intended NPU path
- minimal helper families and minimal line/style drift from `examples/gemm.py`
- comments preserved while editing

## Important scope correction

An earlier plan drifted toward `examples/conv.py`. That is wrong for this phase.
The file under active cleanup is:

```text
examples/kernel_6_18/conv.py
```

Top-level `examples/conv.py` and `examples/conv.py.bak` should not drive this
cleanup unless explicitly requested later.

## Current repository state observed

The working tree is dirty and has many user/research artifacts. Do not revert
or clean them as part of this task.

Relevant files currently present:

```text
plan.md
examples/kernel_6_18/conv.py
examples/gemm.py
experimental/compare_conv_direct_regs.py
tmp_remote_rknn/conv2d_target_160x40_320x3x3.rknn
```

`progress.md` did not exist before this snapshot.

## What is already implemented in `examples/kernel_6_18/conv.py`

The file already has most of the raw machinery needed for the RKNN-style conv
path:

- `run_conv()` exists as the main operation entry.
- `plan_conv_descriptors()` and related planning code can produce tile records.
- `build_direct_spatial_descs()` builds direct-spatial descriptors.
- `attach_direct_spatial_regs()` attaches decoded register lists to descriptor
  records.
- `make_direct_spatial_tile_desc()` can materialize a direct-spatial tile.
- `direct_spatial_task_regs()` selects one of several direct-spatial task
  policies.
- The direct-spatial path is gated by environment variables.
- Hardware direct-spatial execution requires both:

```text
RK3588_CONV_DIRECT_SPATIAL=1
RK3588_CONV_DIRECT_SPATIAL_UNSAFE=1
```

The current direct-spatial task policies in the file are:

```text
single_stream
sparse_single
rocket_record_amounts
pc_root6
```

These policies are diagnostic/research scaffolding. They are not the desired
final clean example shape.

## What has been learned

The main problem is no longer high-level descriptor planning or basic register
decoding. The current evidence points to a missing translation between RKNN
vendor task/PC semantics and the local upstream Rocket driver submit model.

The current best statement of the blocker is:

```text
The sparse direct-spatial RKNN register stream is close, but the local Rocket
PC-chain/task execution model still does not match the vendor RKNN execution
semantics for the target conv.
```

This means cosmetic cleanup should not happen first. If the diagnostic branches
are removed before the execution model is understood, useful evidence will be
lost and the file may look cleaner while still being functionally wrong.

## Hardware results so far

Several candidate task policies have already been tried and rejected. Do not
rerun them unchanged.

### `single_stream`

Result:

```text
submitted without crashing, but produced wrong output
```

Interpretation:

The compact compiler-style stream alone is not enough to reproduce RKNN vendor
execution through local Rocket.

### `sparse_single`

Result:

```text
submitted, but produced wrong output
```

Additional observation:

After a failed `sparse_single` probe, the NPU briefly appeared polluted:

```text
first post-probe simple_add.py -> zeros
second post-probe simple_add.py -> passed
```

Interpretation:

The sparse backing layout is closer to RKNN's emitted backing stream, but one
large local Rocket task does not reproduce the vendor task execution.

### `rocket_record_amounts`

Result:

```text
b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid FAIL max_diff=231.3182
```

Additional observation:

The same NPU state-pollution pattern appeared:

```text
first post-probe simple_add.py -> zeros
second post-probe simple_add.py -> passed
```

Interpretation:

Patching embedded `PC_REGISTER_AMOUNTS` to the per-record Rocket inverse
candidate is not sufficient. The missing semantic is not only the PC amount
field inside the stream.

## Offline verifier progress

`experimental/compare_conv_direct_regs.py` now contains a lot of targeted
offline proof work around the direct-spatial stream.

Important verifier conclusions already captured in `plan.md`:

- The runtime stream for `rocket_record_amounts` matches the offline candidate.
- Sparse record starts stay stable.
- Embedded `PC_BASE_ADDRESS` targets can be checked against valid record starts.
- Embedded `PC_REGISTER_AMOUNTS` can be patched to a per-record Rocket inverse
  candidate.
- The patched candidate avoids obvious record-cutting in offline analysis.
- The runtime span/count model can be compared against what the local Rocket
  submit path will actually pass.

Useful commands from the current plan:

```sh
python3 -m py_compile examples/kernel_6_18/conv.py experimental/compare_conv_direct_regs.py
python3 experimental/compare_conv_direct_regs.py sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1 --verify-rocket-record-amounts-candidate
python3 experimental/compare_conv_direct_regs.py sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1 --verify-runtime-rocket-record-amounts-stream
python3 experimental/compare_conv_direct_regs.py sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1 --verify-runtime-pc-root6-guard
python3 examples/kernel_6_18/simple_add.py
```

These gates were reported as passing before the rejected
`rocket_record_amounts` hardware probe.

## Local Rocket submit model evidence

The local Rocket task ABI is much smaller than the vendor RKNN task ABI.

Vendor RKNN task records include fields such as:

```text
enable_mask
int_mask
int_clear
op_idx
regcfg_amount
regcmd_addr
```

Local Rocket userspace task submission exposes only:

```text
regcmd
regcmd_count
```

The upstream Rocket driver synthesizes the PC commit sequence from that minimal
task description. The current model says every local Rocket task gets a
single-task PC commit:

```text
PC_REGISTER_AMOUNTS = (regcmd_count + 1) / 2 - 1
PC_TASK_CONTROL uses TASK_NUMBER(1) and TASK_PP_EN(1)
PC_TASK_DMA_BASE_ADDR = 0
PC_OPERATION_ENABLE toggles 1 then later 0
```

That differs from the observed RKNN vendor commit model.

For the selected vendor runtime range, current evidence says RKNN derives:

```text
subcore_task[0]=(0,2)
PC_DATA_AMOUNT=0x37
PC_TASK_CONTROL=0x6002
INT_MASK=0x300
INT_CLEAR=0x300
first_fetch_span=[0:112]
```

Current local Rocket diagnostic policies instead model as:

```text
single_stream          task_count=1 regcmd_count=1280 PC_DATA_AMOUNT=639  TASK_CONTROL=0x6001
pc_root6               task_count=6 regcmd_count=224/256/.../32 TASK_CONTROL=0x6001
sparse_single          task_count=1 regcmd_count=1498 PC_DATA_AMOUNT=748  TASK_CONTROL=0x6001
rocket_record_amounts  task_count=1 regcmd_count=1498 PC_DATA_AMOUNT=748  TASK_CONTROL=0x6001
```

The important mismatch is:

```text
RKNN vendor commit fetches regcfg_amount + 4 style ranges and uses task_number=2
for the selected subcore range, while local Rocket derives a single-task commit
from regcmd_count and programs TASK_NUMBER(1).
```

## Rejected partition idea

The obvious idea of splitting the sparse stream at boundaries that are both PC
targets and compiler segment boundaries has been rejected offline.

Current evidence:

```text
pc_targets_in_stream=[0,112,336,448,592,704,848,960,1104,1216]
segment_pc_candidate_boundaries=[0,960,1280]
boundary 960 starts row9:pc_tail
executable_boundaries=[0,1280]
```

Interpretation:

The only internal candidate boundary starts at a PC tail, not a descriptor body.
That is not a valid Rocket task root. If that boundary is excluded, the only
remaining model is the already-rejected single-stream model.

## Remote RKNN comparison progress

A matching RKNN model was generated and run on the remote RKNN environment:

```sh
ssh orangepi@192.168.192.36
cd ~/npu/ops_rknn
python3 gen_conv2d_models.py --custom --out-dir models \
  --batch 1 --in-ch 160 --height 40 --width 40 \
  --out-ch 320 --k-h 3 --k-w 3 --groups 1 \
  --name conv2d_target_160x40_320x3x3
./conv2d_multi --case conv2d_target_160x40_320x3x3 \
  1 160 40 40 320 3 3 1 target_160x40_320x3x3
```

Observed native RKNN layout:

```text
input  1 20 40 40 8
output 1 40 38 38 8
```

The RKNN harness marked the model failed because a few absolute errors exceeded
its `3.5e-2` tolerance, but the observed errors were around `0.04`, not the
`231+` failure seen through local Rocket. This is important evidence:

```text
The large local failure is likely a Rocket submit/PC translation issue, not a
basic compiler/register-shape issue.
```

The exact remote RKNN export was copied back locally:

```text
tmp_remote_rknn/conv2d_target_160x40_320x3x3.rknn
```

The local/remote decoded compiler descriptor stream equivalence check passes:

```text
local_size=948726 remote_size=948726
local_stream_qwords=1280 remote_stream_qwords=1280
PASS remote exact export equivalence
```

The full file hashes differ, but the decoded compiler descriptor stream is the
same. That makes the comparison valid at the register-command level.

## Remote submit tracing progress

Read-only remote GDB submit tracing used:

```text
experimental/remote_rknn_submit_readonly.gdb
```

The generated exact model produced submit metadata matching the saved local RKNN
logs:

```text
SUBMIT[1] cmd=0xc0686441 size=104
flags=0x5 timeout=6000 task_start=0 task_number=6 core_mask=0x0 fence_fd=-1
subcore_task[0]={task_start=0, task_number=2}
subcore_task[1]={task_start=0, task_number=2}
subcore_task[2]={task_start=0, task_number=2}
subcore_task[3]={task_start=0, task_number=0}
subcore_task[4]={task_start=0, task_number=0}
```

The task object pointer in this trace is a kernel object address and was not
directly readable from userspace in the read-only trace. Task-record contents
still need to come from existing GEM dump logs or a separate GEM-map trace.

## Current main blocker

The main blocker to work on next is:

```text
Identify the RKNN/vendor task-root, unit-enable, interrupt, or PC-commit
semantic that local Rocket still does not model for the sparse direct-spatial
stream.
```

More specifically:

- Determine how RKNN executes `task_number=6` globally while each active subcore
  gets `{task_start=0, task_number=2}`.
- Determine whether the local Rocket path should submit two selected root
  records, six root records, or a rewritten PC-root stream.
- Determine whether RKNN relies on `enable_mask`, `int_mask`, or `int_clear`
  fields that local Rocket cannot express through `{regcmd, regcmd_count}`.
- Determine whether the vendor driver has hidden task-table/PC scheduler
  semantics that cannot be reproduced by a plain local Rocket userspace stream.
- Model Rocket's `TASK_NUMBER(1)` commit behavior against RKNN's PC-linked
  stream before creating another hardware candidate.

## What not to do next

Do not rerun these hardware policies unchanged:

```text
single_stream
sparse_single
rocket_record_amounts
```

Do not do broad cleanup first. In particular, do not delete the diagnostic
direct-spatial policies until there is either:

- a passing local Rocket model for the target direct-spatial shape, or
- a strong proof that local Rocket cannot express the required RKNN semantics
  from userspace.

Do not broaden shape support yet. Stay on:

```text
b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid
```

Do not submit unsafe direct-spatial hardware probes without both env gates.

Do not kill long-running NPU processes. Killing stuck/long NPU work can crash or
reboot the board.

## Safe next work

The next useful work should be offline or read-only first:

1. Reconstruct RKNN task roots from GEM dump logs or a controlled GEM-map trace.
2. Compare RKNN task records against local Rocket's generated PC commit model.
3. Derive a candidate task span model from Rocket semantics first, not from
   cosmetic stream shape.
4. Prove the candidate offline:
   - task roots begin at real descriptor bodies
   - PC links target valid record starts
   - fetch ranges do not cut records
   - fetch ranges do not depend on sparse gaps being executable data
   - local Rocket `PC_DATA_AMOUNT` and `TASK_CONTROL` are explicitly modeled
5. Only after a new proof changes, run hardware with:

```sh
python3 -m py_compile examples/kernel_6_18/conv.py experimental/compare_conv_direct_regs.py
python3 examples/kernel_6_18/simple_add.py
RK3588_CONV_DIRECT_SPATIAL=1 \
RK3588_CONV_DIRECT_SPATIAL_UNSAFE=1 \
RK3588_CONV_DIRECT_SPATIAL_TASKS=<new-policy> \
python3 examples/kernel_6_18/conv.py b1_c160_h40_w40_oc320_wic160_k3x3_g1_s1_pvalid
python3 examples/kernel_6_18/simple_add.py
```

If the first post-probe `simple_add.py` returns zeros, rerun it once before
concluding the NPU is unusable. This has already happened after rejected probes.

## Cleanup path after blocker is solved

Once the target direct-spatial path passes or has a clear local-Rocket limitation
proof, clean `examples/kernel_6_18/conv.py` in this order:

1. Keep one canonical RKNN/Rocket direct-spatial policy.
2. Remove or quarantine rejected policies:
   - `pc_root6`
   - `single_stream`
   - `sparse_single`
   - temporary `rocket_record_amounts` diagnostic naming
3. Collapse planner/materializer branches that exist only because direct-spatial
   execution was unproven.
4. Keep decoded register builders explicit.
5. Move probe-only diagnostics out of the main runnable path.
6. Recompare structure against `examples/gemm.py`:
   - short constants section
   - small helper set
   - one obvious execution path
   - clear final PASS/FAIL

## Done criteria

This cleanup goal is done when:

- `examples/kernel_6_18/conv.py` is the cleaned file, not `examples/conv.py`
- the target shape passes through the intended NPU direct-spatial path, or the
  local Rocket limitation is proven strongly enough to define the clean fallback
- there is no opaque conv hex blob in the cleaned example
- the file reads like `examples/gemm.py` in structure and top-level flow
- rejected direct-spatial policies are removed or isolated outside the clean
  runnable example path
- `python3 -m py_compile examples/kernel_6_18/conv.py` passes
- `python3 examples/kernel_6_18/simple_add.py` passes before and after any new
  hardware probe
