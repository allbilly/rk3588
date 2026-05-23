# librknnrt.so Conv ChannelTile decompilation

Source binary: `experimental/rknn/librknnrt.so`

## Critical Finding: No "5 Strategies" in RKNN

**RKNN does NOT use 5 strategies.** The decompiled code reveals **one unified planner** with a `split_method` enum (0=NONE, 1=BY_Y, 2=BY_K, 3=BY_YK). The "5 strategies" in `conv.py` (`direct`, `oc_tile`, `depthwise_spatial`, `spatial_oc_serial`, `spatial_im2col` plus `grouped_serial`, `y_tile`, `y_and_k_tile`) are our own abstraction — they don't exist in RKNN.

RKNN's actual tiling pipeline is:
1. **One Y/K tiler** (`fcn.005d4ac0`/`fcn.005f51c0`) — builds tile records with Y/K starts, bank counts, reuse flags
2. **One parameter extractor** (`fcn.00384880`) — returns tile parameters from seed by mode
3. **Two emission patterns** (ABC_T `0x597828` vs KC_T `0x598468`) — target-specific vtable register writers
4. **One reuse pass** (`fcn.00307198`/`fcn.00312328`) — patches register commands between adjacent tiles
5. **One CNA group mask** (unknown producer) — 14-bit resource-group bitmask

The GDB trace (`experimental/trace_librknnrt_reuse.gdb`) has been run against
`conv2d_simple.rknn` (2026-05-19). It confirmed:

- **fcn.002efd38** patches DMA addresses into pre-built register commands (not
  reuse flags). The BFI at 0x100a40 embeds the 32-bit DMA address into bits
  [47:16] of a 64-bit command entry. Three calling phases patch weight,
  internal, and data buffer DMA addresses.
- **fcn.002efce0** was NOT hit by the single-tile model; it requires a
  multi-tile (ChannelTile) workload.
- **fcn.00100a40** IS called from within fcn.002efd38 — earlier traces missed
  it due to breakpoint ordering.

Remaining gaps after GDB trace:

| # | Piece | Status | Next step |
|---|-------|--------|-----------|
| 3 | **ABC_T/KC_T register layout** (`fcn.00597828`/`598468`) | **Structure known, vtable dispatch confirmed** — register names need multi-tile trace | Run same GDB trace on a multi-tile RKNN model |
| 4 | **Reuse patching register targets** (`fcn.002efd38`/`002efce0`) | **DMA address patching confirmed** — `fcn.002efce0` never hit (single-tile model) | Run on multi-tile model to trigger fcn.002efce0 |
| 5 | **CNA group mask producer** | **Still unknown** — no xref found statically | May resolve from ABC_T trace on multi-tile model |

Runtime submit metadata update (2026-05-23): read-only remote GDB tracing with
`experimental/capture_rknpu_submit.gdb` on the exact copied ChannelTile `sweep`
and `mix` RKNN exports showed one `DRM_IOCTL_RKNPU_SUBMIT` per inference with
`task_number=6`. Subcores 0, 1, and 2 each receive two tasks; subcores 3 and 4
receive none. A follow-up read-only ioctl trace
(`experimental/capture_rknpu_ioctl_readonly.gdb`) correlated the sweep
`task_obj_addr` with a 680-byte RKNPU memory object synced to device before
submit. Copying the remote `dump.py` confirmed the task-record layout is
`<8IQ` (40 bytes), and a smaller read-only GEM scanner decoded the live GEM as a
17-record backing table. Those 17 records are not all active work: submit
`task_number` and `subcore_task[*].task_start/task_number` are the active range
selectors. For this trace, global `task_number=6` implies records 0..5 if read
linearly, while the populated subcore ranges each point at records 0..1. Records
outside the submitted ranges are extra backing-table/cached records and should
not be used to build a local submit mapping. The global linear interpretation is
`conv, conv, conv, conv, separator, conv`; each populated subcore range selects
only the first two conv records. Neither interpretation equals the candidate
six-task/two-body grouping used by local Rocket experiments. This supersedes the
earlier 3-task assumption for these exact exports. The decoded backing table
also has a stable record shape: conv records are spaced by 112 qwords and carry
`enable_mask=0x0d`, `int_mask=0x300`; separator/fence-like records are spaced by
32 qwords and carry `enable_mask=0x60`, `int_mask=0xc00`. That backing structure
is now checked by `experimental/compare_conv_direct_regs.py
--verify-observed-task-gem`, but it remains distinct from the active task range.
The companion `--verify-observed-task-gem-log` mode parses the saved sweep GEM
capture directly and checks those amounts, normalized regcmd offsets, enable
masks, and interrupt masks against the local observed-task model.
`--verify-observed-active-ranges` checks the two active-range interpretations
explicitly, and can run in log-backed mode using the saved submit and GEM dump
captures.
`--verify-runtime-task-boundaries` goes one step further: it combines the exact
exported `.rknn` command stream with the saved submit metadata and task-GEM dump,
then reconstructs candidate command intervals from observed `regcmd_addr` and
`regcfg_amount` records. It preserves the separator/fence-like records instead
of assuming equal descriptor grouping. The current result is an intentional
failure: global `task_number=6`, concatenated repeated `subcore_task[0..2]=(0,2)`,
and unique subcore ranges select different command intervals. That failure is
the safe state; no local Rocket job/task mapping is proven from this trace.

Driver-side interpretation is still the direct-spatial submit blocker. The
available vendor source in `ref/rknpu_driver/rknpu_job.c` treats `core_mask=0`
as `RKNPU_CORE_AUTO_MASK`: scheduling rewrites it to one physical core and sets
`use_core_num=1`. For one- or two-core jobs, PC commit reads
`subcore_task[core_index]`; `subcore_task[core_index + 2]` is used only for
explicit three-core jobs. Under that model, the captured `core_mask=0` metadata
selects one repeated `(0,2)` range, not all six global tasks and not all three
repeated subcore ranges. Do not turn this RKNN trace into a local Rocket
direct-spatial submit until the vendor PC-linked task table is translated into
the local Rocket job/task ABI.
`--verify-rknpu-driver-source-semantics` makes this source evidence executable:
it parses `ref/rknpu_driver/include/rknpu_job.h` for the core-mask constants and
checks `ref/rknpu_driver/rknpu_job.c` for the auto core-mask rewrite,
`use_core_num` calculation, and the `subcore_task[core_index]` versus
`subcore_task[core_index + 2]` PC-commit branches. This check is now part of the
safe preflight.
`--verify-rk3588-vendor-config-semantics` adds the RK3588 config side of that
proof: `ref/rknpu_driver/rknpu_drv.c` binds `rockchip,rk3588-rknpu` to
`rk3588_rknpu_config`, the IRQ table has three subcore IRQ handlers, and the
config carries `core_mask=0x7`, `pc_data_amount_scale=2`,
`pc_task_number_bits=12`, and `pc_task_number_mask=0xfff`. So the checked-in
vendor RK3588 path really uses the multi-IRQ/subcore task-selection branch.
`--verify-vendor-runtime-task-selection` now encodes this source model and
passes on the saved sweep logs, selecting records `0..1` under an empty-queue
core-0 model. This narrows the vendor active-range interpretation but is not a
complete local Rocket mapping of the full 1280-qword compiler stream.
`--verify-vendor-pc-commit-register-model` applies the same source path through
`rknpu_job_subcore_commit_pc()`: for `subcore_task[0]=(0,2)`, the vendor commit
would program `PC_DATA_ADDR=0x0`, `PC_DATA_AMOUNT=0x37`,
`PC_TASK_CONTROL=0x6002`, and `INT_MASK=INT_CLEAR=0x300`, with the first fetch
covering qwords `[0:112]`. That is concrete evidence that the vendor-selected
submit starts from record 0 and does not directly describe a full-stream local
Rocket job.
`--verify-vendor-selected-rocket-rejection` makes the local Rocket consequence
explicit. If the source-selected `(0,2)` range were copied into Rocket, the job
would have only two tasks, raw spans `[0:108]` and `[112:220]`, covering
`216/1280` qwords. With the vendor fetch rule it reaches only `[0:224]`, or
`224/1280` qwords. This is now a checked rejection of the "submit the selected
subcore range" shortcut.
`--verify-global-task-number-rocket-rejection` rejects the global
`task_number=6` shortcut as well: records `0..5` cover `554/1280` qwords raw and
`578/1280` qwords with the vendor fetch rule, include a separator record, and
include two conv records whose starts are not PC-chain targets.

Local kernel 6.18 uses the upstream/mainline Rocket ABI rather than the vendor
`rknpu_submit` ABI. On this board `rocket.ko` is loaded from
`/lib/modules/6.18.24-current-rockchip64/kernel/drivers/accel/rocket/rocket.ko`,
and the local `rocket_runtime.py`/`ref/librocket/include/rocket_accel.h` UAPI is
`drm_rocket_submit -> drm_rocket_job -> drm_rocket_task`, with each task carrying
only `{regcmd, regcmd_count}`. There is no local Rocket `core_mask`,
`subcore_task`, `task_start`, or vendor `task_number` field. The new
`--verify-local-rocket-uapi` check records this distinction from the Python
ctypes model, and `--verify-local-rocket-uapi-source` checks the same distinction
against `ref/librocket/include/rocket_accel.h`. The direct-spatial submit blocker
is therefore exact Rocket job/task-boundary translation from the RKNN
compiler/runtime traces, not copying vendor submit fields.

The direct-spatial HW probe path remains guarded by both
`RK3588_CONV_DIRECT_SPATIAL=1` and `RK3588_CONV_DIRECT_SPATIAL_UNSAFE=1`. This is
now enforced by the offline `--verify-direct-spatial-gate` check in
`experimental/compare_conv_direct_regs.py`, which exercises the env-var truth
table without submitting work to hardware.

The consolidated offline checklist is
`experimental/compare_conv_direct_regs.py --verify-safe-direct-spatial-preflight`.
It runs the gate, RKNN span/stream/boundary, candidate 6-task, observed task GEM,
exact sweep/mix export-pair check, log-backed observed task GEM, submit log
consistency, task-object sync correlation, task-object record count, observed
active-range, runtime task/fetch coverage, task-gap and row-overlap summaries,
backing-span, PC-base link, PC-amount length, driver PC-fetch span, vendor
core-mask, vendor driver-source, vendor runtime-selection, and local Rocket UAPI
verifiers in one sequence. Passing this checklist is required before any future
guarded direct-spatial HW probe, but it is still not evidence that a Rocket
submit mapping is safe.

`--verify-exact-export-pair` checks both copied exact exports. Current result:
`sweep` and `mix` both pass the RKNN span, stream, and compiler-boundary checks
and share the same 1280-qword compiler-stitched command stream.

`--verify-submit-log-consistency` parses the saved sweep and mix read-only submit
logs and verifies both have exactly one `DRM_IOCTL_RKNPU_SUBMIT` with
`cmd=0xc0686441`, `size=104`, `flags=0x5`, `timeout=6000`, `task_start=0`,
`task_number=6`, `core_mask=0`, `fence_fd=-1`, and
`subcore_task[0..2]=(0,2)` while subcores 3/4 are unused.

`--verify-task-object-sync-correlation` parses
`experimental/rknn/capture_rknpu_ioctl_sweep_readonly.log` and verifies the
submitted `task_obj_addr` is the object synced as a 680-byte task buffer with
`MEM_SYNC` flags `0x3` and `0x1` before submit. This ties the task-GEM model to
the actual submitted task object.

`--verify-task-object-record-count` parses
`ref/rknpu_driver/include/rknpu_ioctl.h` to confirm packed `struct rknpu_task` is
40 bytes, then verifies the 680-byte submitted task object corresponds to
exactly 17 decoded task records.

The stricter Phase D mapping guard is now:

```sh
python3 experimental/compare_conv_direct_regs.py sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1 \
  --verify-runtime-task-boundaries \
  --submit-log experimental/rknn/capture_rknpu_submit_sweep_readonly.log \
  --task-gem-log experimental/rknn/capture_rknpu_submit_dump_gems_sweep.log
```

It currently exits nonzero with `FAIL ambiguous observed runtime task boundaries`.
It also flags records from the global interpretation that are not PC-chain task
starts. Do not change `examples/kernel_6_18/conv.py` submit behavior or run
another direct-spatial hardware probe until this ambiguity is resolved offline.

The companion `--verify-runtime-task-coverage` mode is diagnostic and currently
shows:

- global `task_number=6` covers 554/1280 qwords of the exact RKNN stream
- concatenated repeated subcore ranges cover 216/1280 qwords
- unique subcore range covers 216/1280 qwords
- all interpretations first leave the stream uncovered at qword 108

This makes the remaining gap concrete: the captured task-record intervals are
not, by themselves, a complete local Rocket command stream. A safe mapping must
explain the missing PC-tail/fence/stream regions before any hardware submit.

The companion `--verify-pc-linked-task-graph` mode follows `PC_BASE_ADDRESS`
links from the observed submit-selected records. Current log-backed result:
global entries `[0..5]` reach records `[0..7]`, repeated or unique
`subcore_task=(0,2)` entries reach only `[0,1]`, and records with starts inside
the exact 1280-qword stream include `[0..14]`. The submit-selected entries plus
PC chaining therefore still do not cover the full compiler stream.

The companion `--verify-pc-chain-components` mode computes roots of the
PC-linked backing table. Current log-backed result: active records with starts
inside the stream are `[0..14]`, while PC-chain roots are `[0,2,5,8,11,14]`.
The root count is six, matching global `task_number=6`, but the root set is not
the observed global entry range `[0..5]` and not the unique subcore entry range
`[0,1]`. This is useful translator structure, not a local Rocket submit recipe.

The companion `--verify-pc-chain-component-coverage` mode maps those root
components back to compiler row segments using the vendor `regcfg_amount + 4`
fetch rule. Current log-backed result intentionally fails: component fetches
cover `1240/1280` qwords and leave gaps such as `[332:336]`, `[444:448]`,
`[478:480]`, and later repeated alignment/body gaps. Therefore even the six
PC-chain roots are not by themselves a complete local Rocket command stream.

The companion `--verify-pc-root-contiguous-rocket-candidate` mode partitions the
exact compiler stream at the six PC-chain root starts. Current log-backed result
passes offline with candidate Rocket task spans `[0:224]`, `[224:480]`,
`[480:736]`, `[736:992]`, `[992:1248]`, and `[1248:1280]`
(`224,256,256,256,256,32` qwords). This candidate is still not directly proven
by the observed RKNN submit metadata, but it is now implemented as the local
runtime default for the exact supported descriptor schedule once both unsafe
direct-spatial gates are set. `RK3588_CONV_DIRECT_SPATIAL_TASKS=reg_lists`
remains an explicit diagnostic override, and
`RK3588_CONV_DIRECT_SPATIAL_TASKS=pc_root6` remains a strict explicit raw-span
request that rejects unsupported schedules before submit. The runtime
implementation derives record starts and PC tail amounts from the observed
17-record backing-table amounts instead of embedding every tail address
directly.

`--verify-runtime-pc-root6-stream` checks the actual `conv.py` runtime
implementation. It builds the runtime `pc_root6` stream/spans, normalizes local
BO DMA bases and `PC_BASE_ADDRESS` values to offsets, and compares the result
against the exact RKNN compiler stream plus expected six raw Rocket spans. This
check is part of `--verify-safe-direct-spatial-preflight` and currently passes
with `1280` qwords and spans `(0,224), (224,256), (480,256), (736,256),
(992,256), (1248,32)`.

`--verify-runtime-pc-root6-guard` checks the runtime shape guard. The guard
accepts the exact known 40x40 descriptor schedule and rejects all other currently
mirrored `160->320`, `3x3` sweep schedules before any submit. This check is part of
`--verify-safe-direct-spatial-preflight`.

`--verify-direct-spatial-task-policy` checks direct-spatial task-policy
selection. Empty/unset `RK3588_CONV_DIRECT_SPATIAL_TASKS` still reports the
global default `reg_lists` policy when no descriptor schedule is supplied, but
the exact supported descriptor schedule defaults to `pc_root6`. Explicit
`reg_lists` keeps the diagnostic reg-list path available, explicit `pc_root6`
selects the raw-span candidate, and any other value is rejected before submit.
This check is part of
`--verify-safe-direct-spatial-preflight`.

`--verify-observed-descriptor-sweep` checks descriptor planning separately from
submit safety. It compares `conv.py`'s observed spatial planner against
`experimental/rknn_descriptor_plan.py` for the `160->320`, `3x3`, groups-1 square
sweep at `H=16,20,24,28,32,36,40`. Current result: all seven shapes pass,
covering 60 descriptor records. The parity keys include family, `grain_bits`,
`CBUF_CON0`, `input_y`, `input_h`, output dimensions, `oc_start`, `oc_count`,
`pc_core`, and feature/weight/output offsets. This proves the direct-spatial
descriptor producer has broader planning coverage than the current exact-shape
`pc_root6` hardware policy while still checking key RKNN-observed resource
fields and PC-tail core routing.

`--verify-direct-spatial-planner-routing` checks the actual
`plan_conv_descriptors()` routing. It verifies that fallback descriptors remain
the default unless both direct-spatial env gates are set, that the double-gated
default path returns a raw-span `pc_root6` direct-spatial descriptor for the
exact supported 40x40 schedule, that
`RK3588_CONV_DIRECT_SPATIAL_TASKS=reg_lists` returns the diagnostic reg-list
descriptor, and that an unsupported mirrored 36x36 schedule raises before
submit when `pc_root6` is explicitly requested. This check is part of
`--verify-safe-direct-spatial-preflight`.

The companion `--verify-pc-root-candidate-boundaries` mode checks that
candidate's boundaries. Current log-backed result intentionally fails:
boundaries `224`, `480`, `736`, `992`, and `1248` are observed task-record
starts, but they are not `PC_BASE_ADDRESS` targets and they cut through compiler
body segments. This remains a conservative offline warning rather than a
hardware result: the guarded `pc_root6` probe on the exact `160x40 -> 320`,
`3x3`, groups-1 target passed on the local Rocket driver with `max_diff=0.0625`,
and `examples/kernel_6_18/simple_add.py` passed immediately afterward.

The companion `--verify-direct-spatial-blockers` mode is the explicit
expected-failure guard. It runs `--verify-runtime-task-boundaries`,
`--verify-pc-chain-component-coverage`, and
`--verify-pc-root-candidate-boundaries`, and passes only while all three remain
blocked for the known offline reasons. It is included in the safe preflight so a
previously unsafe mapping cannot silently become accepted.

The companion `--verify-runtime-task-gaps` mode decodes those uncovered regions.
The first missing range for all submit interpretations is qwords `108:112`:
`PC_BASE_ADDRESS`, `PC_REGISTER_AMOUNTS`, `PC_OPERATION_ENABLE`, and then the
next body's `CBUF_CON0`. Later gaps contain ordinary body fragments too, including
`CONV_CON2`, `DATA_SIZE*`, `WEIGHT_SIZE*`, DCOMP, and CORE setup commands. That
means the missing mapping is not merely "observed task records plus PC tails";
it crosses stitched compiler-run body boundaries.

The companion `--verify-runtime-task-row-overlaps` mode maps each observed task
record back to the compiler descriptor rows. Current result: record 0 is exactly
row 1 body, record 1 spans row 2 body plus row 3 `prev_enable`, later records
span body/tail fragments across adjacent rows, and the separator record is a
slice of row 5 body. The runtime task table is therefore a segmentation of the
stitched command stream, not a descriptor-row table.

The companion `--verify-task-backing-span` mode compares the entire observed
17-record backing table to the exact compiler stream. Current result: the
compiler stream is 1280 qwords, while the backing table spans 1498 qwords; records
14, 15, and 16 extend past the compiler stream. This is additional evidence that
the whole backing table is cached/backing storage, not the active work list.

The companion `--verify-pc-base-task-links` mode checks the PC-chain addresses in
the exported stream against observed task-record starts. Current result: all 11
`PC_BASE_ADDRESS` values target task-record starts, and the last two target
records 15 and 16 beyond the 1280-qword exact stream. This supports a
PC-linked stitched-stream task-table model, but active work still has to be
selected from submit metadata.

The companion `--verify-pc-amount-task-lengths` mode checks the adjacent
`PC_REGISTER_AMOUNTS` values. Current result: for all 11 PC links,
`PC_REGISTER_AMOUNTS.low * 2 - 2` equals the linked task record's
`regcfg_amount`, while the high bits carry the core index. This proves the PC
tail is linked to the task table by both address and fetch length.

The companion `--verify-driver-pc-fetch-spans` mode encodes the vendor driver's
`RKNPU_PC_DATA_EXTRA_AMOUNT=4` fetch rule. Current result: each record fetches
`regcfg_amount + 4` qwords. That lands 108-qword records exactly on the next
112-qword boundary, leaves 4 qwords of alignment padding after 104-qword records,
and leaves 2 qwords after 26-qword separator records. This explains the observed
task-record spacing without treating padding as active commands.

The companion `--verify-runtime-fetch-coverage` mode applies the same fetch rule
to the submit-selected interpretations. Current result: global `task_number=6`
fetches 578/1280 qwords with alignment gaps, while repeated or unique
`subcore_task=(0,2)` fetches contiguous qwords `[0:224]`. This is the closest
current offline model of vendor PC DMA coverage, but it still does not prove a
local Rocket mapping for the full command stream.

Command style used:

```sh
rabin2 -zz experimental/rknn/librknnrt.so | rg 'min_weight_banks|ChannelTile'
r2 -q -e scr.color=false -e bin.relocs.apply=true -c 'aaa' -c 'axt @ 0x60f388' -c q experimental/rknn/librknnrt.so
r2 -q -e scr.color=false -e bin.relocs.apply=true -c aa -c 's 0x00387e3c' -c af -c pdf -c q experimental/rknn/librknnrt.so
```

## Target string and xrefs

The string:

```text
0x0060f388 min_weight_banks > 3 && m_Target == RKNNTarget::RKNPU_F2, do conv with ChannelTile
```

has two code xrefs:

```text
fcn.00387e3c @ 0x0038831c
fcn.00388a18 @ 0x00388e84
```

Both functions implement the same high-level decision: compute a minimum number of
weight CBUF banks for a proposed conv tile, then switch to ChannelTile when that
number exceeds the target-specific bank threshold.

The logging/plumbing callees used on the diagnostic path are not conv tiling
logic:

```text
fcn.000ff6e0  stream/string construction
fcn.000dae38  append text/integer to stream
fcn.000dfbd0  emit stream/log
operator new/delete and abort
```

They are excluded from the tiling pseudo-code below except where a branch emits
one of the ChannelTile messages.

## Related tiling-string xrefs

After starting from the F2 ChannelTile string, the other conv tiling/reuse
strings in `librknnrt.so` lead to a second layer-level tiling path and a set of
diagnostic helpers. These are the useful xrefs:

| String address | String | Xref function | Role |
| --- | --- | --- | --- |
| `0x00607178` | `CNA feature group0: %d` | `fcn.00101be8` | Register-task/resource-group dump. Confirms the runtime tracks CNA feature/weight/CSC groups, but it is not the planner. |
| `0x0060c768` | `failed to update data and weight reuse!` | `fcn.00307198`, `fcn.00312328` | Tile-table reuse updater. These functions validate/update data/weight reuse bits after a split plan is built. |
| `0x0062cc50` | `banks num invalid, input_bank_num: %d, weight_bank_num: %d` | `fcn.005c6b50` | Large workload/layer config function. Validates bank allocation and reuse strategy while emitting NPU workload configuration. |
| `0x0062cc90` | `Invalid reuse strategy!` | `fcn.005c6b50` | Same workload config validator. |
| `0x0062ce00` | `Conv min_weight_banks > 3, OutputName : %s` | `fcn.005d4ac0` | Layer-level conv tiler. This is the broader function that builds split/Y/K tile records and logs when a conv crosses the F2-style weight-bank threshold. |
| `0x0062ce50` | `MC K Tile Failed, min kernel step %d, but get %d` | `fcn.005d4ac0` | Same layer-level conv tiler; reports failed multi-core K/kernel tiling. |
| `0x0062ce30` | `failed to tile argb mode layer!` | `fcn.005d4ac0` | Same layer-level conv tiler; ARGB-mode tile generation failure path. |
| `0x0062cf58` | `Unknown split Method -> %d` | `fcn.005f1998` | Abort helper for invalid split-method enum. |
| `0x0062cf78` | tile-table header with `xstart`, `ystart`, `kstart`, `data reuse`, `weight reuse`, `mc_treat_by_*` | `fcn.005f1cd0` | Tile-table dump/formatter. Shows the actual per-tile fields RKNN uses after planning. |
| `0x0062d040` | `illegal tiling method %d` | `fcn.005f38f0` | Conv K/X split composer. Emits fatal diagnostics when a split method cannot be represented. |
| `0x0062d298` | `X tile buffer overflow!` | `fcn.005f51c0` | Top-level conv Y/K/multicore tiler. Builds nested tile vectors and emits X/Y/MC diagnostics. |
| `0x0062d508` | `Generate Y config crash!` | `fcn.005f51c0` | Same top-level conv tiler; reports failed Y tile configuration. |
| `0x0062d7b8` | `Illegal mc Y type` | `fcn.005f51c0` | Same top-level conv tiler; validates multi-core Y treatment. |
| `0x0062d8e8` | `Illegal mc K type` | `fcn.005f51c0` | Same top-level conv tiler; validates multi-core K treatment. |

The important separation is:

- `fcn.00387e3c` and `fcn.00388a18` answer: should this conv switch to
  ChannelTile because bank pressure is too high?
- `fcn.005d4ac0`, `fcn.005f38f0`, and `fcn.005f51c0` answer: after a
  conv is being tiled, how are Y/K/X split records chosen, checked, nested,
  and mapped to multi-core treatment?
- `fcn.00307198`, `fcn.00312328`, `fcn.005c6b50`, and `fcn.005f1cd0` answer:
  how are data/weight reuse flags, bank counts, and tile records validated or
  printed?

## Function inventory

The stripped binary does not expose useful C++ names for the compiler internals,
so the coverage set below is built from `rabin2 -zz` string xrefs and direct
callee inspection around the conv tiling cluster. These are the functions that
participate in conv tiling rather than generic logging, allocation, or vector
bookkeeping:

| Function | Size | Role | Decompilation status |
| --- | ---: | --- | --- |
| `fcn.00387e3c` | `1436` | Main ChannelTile predicate. Computes candidate channel/K tile, minimum data tile, minimum weight banks, then checks target thresholds. | Full pseudo-code below. |
| `fcn.00388a18` | `1248` | Fixed-16 sibling ChannelTile predicate. Same threshold table with a narrower candidate setup. | Full pseudo-code below. |
| `fcn.00384ed0` | `628` | Minimum weight-bank estimator used by both ChannelTile predicates and planner paths. | Full pseudo-code below. |
| `fcn.00387530` | not separately sized in this note | Base tile-shape selector. Chooses atomic count/step pair from shape vectors and target limits. | Full pseudo-code below. |
| `fcn.00383338` | not separately sized in this note | Minimum data-tile estimator. Converts spatial span and aligned channel count into CBUF data pressure. | Full pseudo-code below. |
| `fcn.003873e8` | not separately sized in this note | Channel tile clamp/tuner. Rounds requested channel tiles and halves until legal. | Full pseudo-code below. |
| `fcn.00385988` | `224` | Weight/K split feasibility helper. Converts spare banks and K groups into a legal K step or `-1`. | Full pseudo-code below. |
| `fcn.00384ba0`, `fcn.00384d48`, `fcn.00389078`, `fcn.000e1100`, `fcn.000f4758` | small | Atomic-width limit selectors used by the estimators/tuners. | Selector pseudo-code below. |
| `fcn.005d4ac0` | `11792` | Layer-level conv split planner. Builds Y/K tile records and logs F2 `min_weight_banks > 3`. | High-level pseudo-code below. |
| `fcn.005f38f0` | `6344` | K/X split composer used from `fcn.005f51c0`. Produces per-split vectors and rejects illegal tiling methods. | High-level pseudo-code below. |
| `fcn.005f51c0` | `18488` | Top-level conv Y/K/multicore tiler. Calls `fcn.005f38f0`, validates X/Y/MC split state, builds nested vectors. | High-level pseudo-code below. |
| `fcn.00307198`, `fcn.00312328` | not separately sized in this note | Data/weight reuse update passes over finished tile lists. | Shared pseudo-code below. |
| `fcn.005c6b50` | not separately sized in this note | Workload bank/reuse field validator and register-task config helper. | Relevant checks below. |
| `fcn.005f1cd0` | not separately sized in this note | Tile-table dump helper for `xstart`, `ystart`, `kstart`, reuse, and MC fields. | Formatter pseudo-code below. |
| `fcn.005f1998` | not separately sized in this note | Invalid split-method abort helper. | Small pseudo-code below. |

Functions such as `fcn.000ff6e0`, `fcn.000dae38`, `fcn.000dfbd0`,
`operator new/delete`, `memcpy`, `memmove`, and vector constructors/destructors
appear in these functions but are plumbing, not conv tiling logic.

## Target tags

The target is stored as a 32-bit tag at offset `+0x0` of the conv tiler object.
The code compares the tag as an integer, which reads as reversed ASCII on LE:

```c
enum RKNNTargetTag {
    RKNPU_F2 = 0x46495247, // bytes "GRIF"
    RKNPU_F3 = 0x46495248, // bytes "HRIF"
    RKNPU_W2 = 0x57494e46, // bytes "FNIW"
    RKNPU_W1 = 0x57494e45, // bytes "ENIW"
};
```

## Inferred object fields

The stripped binary gives offsets but no field names. These names are inferred
from how each value is used by the tiling math.

```c
struct ConvTileCtx {
    uint32_t target;              // +0x00, RKNNTargetTag
    int32_t  atomic_c_total;      // +0x28
    int32_t  limit_4_a;           // +0x2c
    int32_t  limit_4_b;           // +0x30
    int32_t  limit_8_a;           // +0x34
    int32_t  limit_8_b;           // +0x38
    int32_t  limit_16_a;          // +0x3c
    int32_t  limit_16_b;          // +0x40
    int32_t  limit_32_a;          // +0x44
    int32_t  limit_32_b;          // +0x48
    int32_t  limit2_4;            // +0x4c
    int32_t  limit2_8;            // +0x50
    int32_t  limit2_16_or_fmt9;   // +0x54
    int32_t  limit2_32;           // +0x58
    int32_t  banks_for_normal;    // +0x60, for most targets
    int32_t  bank_granularity;    // +0x64
    int32_t  atomic_c_hw;         // +0x68, frequently shifted by 3
    int32_t  banks_for_f2;        // +0xa0, used for F2 target
    int32_t  special_align;       // +0x220, used by fcn.00387530 format==9 path
};
```

## fcn.00387e3c: main conv ChannelTile test

This is the first and larger xref function. It computes a candidate K/channel
tile, derives data/weight bank pressure, then returns `1` when ChannelTile should
be used.

Key addresses:

```text
0x00388150 call fcn.00384ed0      ; returns min_weight_banks in w0
0x00388158 cmp w0, 3              ; F2 threshold
0x00388180 branch to F2 log
0x00388188 cmp w0, 7              ; F3/W1 threshold
0x00388198 branch to F3 log
0x0038819c cmp w0, 8
0x003881a0 min_weight_banks == 8 skips W2
0x003881b0 branch to W2 log
0x003881c0 branch to W1 log
```

Pseudo-code:

```c
bool should_use_channel_tile_main(
    ConvTileCtx *ctx,
    int dims0[4],        // first shape vector copied into a small vector
    int arg2_flag,
    int dims1[2],
    bool early_disable,
    bool swap_dims,
    bool use_alt_shape,
    bool force_fallback)
{
    if (early_disable)
        return false;

    int in0;
    int in1;
    int out_h;
    int out_w;
    int stride_like;
    int fmt_or_dtype;

    if (swap_dims) {
        in0 = dims0[1];
        in1 = dims0[0];
    } else {
        in0 = dims0[0];
        in1 = dims0[1];
    }

    out_h = dims0[2];
    out_w = dims0[3];
    stride_like = dims1[0];
    fmt_or_dtype = dims1[1];

    TileShape tile_shape = select_tile_shape(ctx, arg2_flag);
    int tile_count = tile_shape.count;
    if (tile_count == 0)
        abort_bad_tiling(tile_count);

    int atomic = ctx->atomic_c_total;
    int atom_groups;
    if (ctx->target == 0 && tile_count == 4)
        atom_groups = div_round_up_signed(atomic, 16);
    else
        atom_groups = div_round_up_signed(atomic, 8) / tile_count;

    int c_step = tile_count * 8;
    int shape_limit = pick_limit_b_by_atomic(ctx, c_step, fmt_or_dtype);

    int aligned_tile_count = round_up_to_multiple(tile_count + in1, atom_groups);

    // Programs/checks an initial data-vs-weight split into local vectors.
    check_base_tiling(
        ctx,
        /*mode=*/0,
        /*start=*/0,
        /*shape_vec=*/dims0,
        /*scratch_vec=*/local_vec,
        /*out=*/&tile_shape,
        /*atomic_step=*/c_step,
        /*unknown=*/0,
        /*alt=*/swap_dims,
        /*final=*/false);

    int total_atomic_tiles = ctx->atomic_c_total / c_step;
    int data_span_tiles = (ctx->atomic_c_hw * 8) / c_step;
    int aligned_start = round_up_to_multiple(in1, total_atomic_tiles);

    int modulo_base;
    if (c_step == 8)
        modulo_base = ctx->limit_8_a;
    else if (c_step == 16)
        modulo_base = ctx->limit_16_a;
    else if (c_step == 32)
        modulo_base = ctx->limit_32_a;
    else if (c_step == 64)
        modulo_base = -1;
    else
        abort_bad_tiling(c_step);

    int aligned_for_limit = round_up_to_multiple(aligned_start, total_atomic_tiles);
    int split_residue = aligned_for_limit % modulo_base;
    if (split_residue != 0) {
        int reduced = choose_power2_reduced_tile(aligned_for_limit, split_residue,
                                                 modulo_base);
        aligned_for_limit = aligned_for_limit - split_residue + reduced;
    }

    int bank_budget = (ctx->target == RKNPU_F2)
        ? ctx->banks_for_f2
        : ctx->banks_for_normal;

    int min_data_tile = estimate_min_data_tile(ctx,
        (out_w - 1) * stride_like + 1,
        aligned_tile_count,
        c_step);

    int min_weight_banks = estimate_min_weight_banks(
        ctx,
        /*mode=*/0,
        /*tile spatial dims=*/dims0[2], dims0[3],
        /*aligned input tile=*/aligned_for_limit,
        /*k tile=*/in0,
        /*tile_count=*/tile_count,
        /*alt=*/use_alt_shape,
        /*force=*/force_fallback,
        /*first=*/true);

    int remaining_bank_space =
        (bank_budget - min_weight_banks) * ctx->bank_granularity / min_data_tile;

    if (min_weight_banks > 3 && ctx->target == RKNPU_F2) {
        log("min_weight_banks > 3 && m_Target == RKNNTarget::RKNPU_F2, do conv with ChannelTile");
        return true;
    }

    if (min_weight_banks > 7 && ctx->target == RKNPU_F3) {
        log("min_weight_banks > 7 && m_Target == RKNNTarget::RKNPU_F3, do conv with ChannelTile");
        return true;
    }

    if (min_weight_banks > 8 && ctx->target == RKNPU_W2) {
        log("min_weight_banks > 8 && m_Target == RKNNTarget::RKNPU_W2, do conv with ChannelTile");
        return true;
    }

    if (min_weight_banks > 7 && ctx->target == RKNPU_W1) {
        log("min_weight_banks > 7 && m_Target == RKNNTarget::RKNPU_W1, do conv with ChannelTile");
        return true;
    }

    // After the ChannelTile diagnostic block, the function continues with a
    // viability loop. It halves the selected channel limit until the remaining
    // bank space can hold the required data tile. If it cannot, it returns true.
    int channel_limit = shape_limit;
    if (arg2_flag == 0x0a || arg2_flag == 0x10)
        atom_groups = channel_limit;

    while (in0 < channel_limit)
        channel_limit >>= 1;

    if (out_h_tile_requirement(out_h, stride_like) > remaining_bank_space)
        return true;

    return false;
}
```

## fcn.00388a18: sibling ChannelTile test

This function is a compact sibling. It uses a fixed `c_step = 16`/`tile_count = 2`
path and then applies the same target threshold table.

Key addresses:

```text
0x00388bf8 call fcn.00384ed0      ; returns min_weight_banks in w0
0x00388c04 cmp w0, 3
0x00388c24 branch to F2 log
0x00388c2c cmp w0, 7
0x00388c3c branch to F3 log
0x00388c44 min_weight_banks == 8 skips W2
0x00388c54 branch to W2 log
0x00388c64 branch to W1 log
```

Pseudo-code:

```c
bool should_use_channel_tile_fixed16(
    ConvTileCtx *ctx,
    int dims0[4],
    int dims1[2],
    bool swap_dims,
    bool alt,
    bool final)
{
    int k_tile = swap_dims ? dims0[1] : dims0[0];
    int c_start = swap_dims ? dims0[0] : dims0[1];
    int out_h = dims0[2];
    int out_w = dims0[3];

    check_base_tiling(ctx, 0, swap_dims, local_vec, out_vec,
                      /*atomic_step=*/16, 0, 0, false);

    int atomic_tiles = div_round_up_signed(ctx->atomic_c_total, 16);
    int aligned_c = round_up_to_multiple(c_start, atomic_tiles);
    int limit_mod = ctx->limit_16_a;
    int residue = aligned_c % limit_mod;
    if (residue != 0)
        aligned_c = aligned_c - residue + choose_power2_reduced_tile(aligned_c, residue,
                                                                     limit_mod);

    int bank_budget = (ctx->target == RKNPU_F2)
        ? ctx->banks_for_f2
        : ctx->banks_for_normal;

    int min_data_tile = estimate_min_data_tile(ctx, (out_w - 1) * dims1[0] + 1,
                                               aligned_c, 16);

    int min_weight_banks = estimate_min_weight_banks(
        ctx, 0, dims0[2], dims0[3], aligned_c, k_tile,
        /*tile_count=*/2, alt, final, true);

    int remaining_bank_space =
        (bank_budget - min_weight_banks) * ctx->bank_granularity / min_data_tile;

    if (min_weight_banks > 3 && ctx->target == RKNPU_F2)
        return log_channel_tile_and_true(F2);
    if (min_weight_banks > 7 && ctx->target == RKNPU_F3)
        return log_channel_tile_and_true(F3);
    if (min_weight_banks > 8 && ctx->target == RKNPU_W2)
        return log_channel_tile_and_true(W2);
    if (min_weight_banks > 7 && ctx->target == RKNPU_W1)
        return log_channel_tile_and_true(W1);

    int channel_limit = ctx->limit_16_b;
    while (k_tile < channel_limit)
        channel_limit >>= 1;

    if (out_h_tile_requirement(out_h, dims1[0]) > remaining_bank_space)
        return true;

    return false;
}
```

## fcn.00384ed0: estimate minimum weight banks

This is the core helper called immediately before the target threshold checks.

Inputs by calling convention:

```text
x0 ctx
w1 mode/flag
w2 tile spatial dim A
w3 tile spatial dim B
w4 aligned channel/count
w5 k tile
w6 tile_count or forced bank units
w7 bool alt
stack+0x70 bool force_2
stack+0x78 bool force_small
stack+0x80 format/dtype
```

Pseudo-code:

```c
int estimate_min_weight_banks(
    ConvTileCtx *ctx,
    bool mode,
    int tile_a,
    int tile_b,
    int aligned_channels,
    int k_tile,
    int forced_units,
    bool alt,
    bool force_2,
    bool force_small,
    int fmt_or_dtype)
{
    int element_step = 4;
    int tile_count = 2;

    // For some target tags or flags the routine uses 4 banks instead of 2.
    if (is_non_f2_like_target(ctx->target) || force_2 || force_small)
        tile_count = 4;

    if (forced_units != 0)
        element_step = forced_units * 8;

    if (fmt_or_dtype == 9) {
        element_step = normalize_fmt9_step(fmt_or_dtype);
        tile_count = pick_limit_b_by_atomic(ctx, element_step, fmt_or_dtype);
    } else {
        tile_count = pick_limit_b_by_atomic(ctx, element_step, fmt_or_dtype);
    }

    if (alt)
        tile_count = 2;

    if (!mode && force_2) {
        k_tile = 1;
        tile_count = 1;
    } else if (mode) {
        k_tile = 2;
        tile_count = 2;
    }

    int weight_bytes =
        tile_a * tile_count * tile_b * aligned_channels * element_step;
    int weight_banks = ceil_div_signed(weight_bytes, 8);

    if (ctx->target != 0) {
        int cacc_banks = ceil_div_signed(weight_banks, ctx->bank_granularity);
        int k_groups = ceil_div_signed(k_tile, tile_count);
        int base = ceil_div_signed(ctx->banks_for_normal + cacc_banks - 1,
                                   ctx->banks_for_normal);

        if (k_groups != 1 && base > 1) {
            int rem = cacc_banks % ctx->banks_for_normal;
            if (rem != 0 && ctx->banks_for_normal <= rem * k_groups)
                base++;
        }

        return base;
    }

    int total = ctx->bank_granularity * ctx->banks_for_normal;
    return ceil_div_signed(weight_banks + total - 1, total);
}
```

## fcn.00387530: check/select base tiling shape

This helper writes two integers to the output pointer passed in `x4`.

Pseudo-code:

```c
struct TileShape {
    int count;
    int step;
};

int check_base_tiling(
    ConvTileCtx *ctx,
    bool mode,
    int fmt_or_dtype,
    int shape_vec[],
    TileShape *out,
    int atomic_step,
    bool use_shape_vec,
    bool alt)
{
    int atomic_c_hw = ctx->atomic_c_hw * 8;
    int step = atomic_c_hw / atomic_step;

    int vec_len = vector_len(shape_vec);
    if (vec_len == 4 && !use_shape_vec) {
        if (ctx->target == RKNPU_F2 || ctx->target == RKNPU_W1 ||
            ctx->target == RKNPU_F3 || ctx->target == TAG_TREH ||
            ctx->target == TAG_TREI) {
            out->count = ctx->atomic_c_total / atomic_step;
            out->step = step;
            return 0;
        }
    }

    if (vec_len == 2 || use_shape_vec) {
        int candidate = choose_shape_candidate(shape_vec, use_shape_vec,
                                               atomic_step, alt);
        int hw_limit = get_hw_limit(ctx, atomic_step, alt);
        if (candidate > hw_limit)
            candidate = hw_limit;

        if (fmt_or_dtype == 9)
            candidate = round_up_to_multiple(candidate, ctx->special_align);
        else if (!mode)
            candidate = round_up_to_multiple(candidate,
                                             get_channel_align(ctx, atomic_step, alt));

        out->count = ctx->atomic_c_total / atomic_step;
        out->step = candidate;
        return 0;
    }

    out->count = 0;
    out->step = step;
    return 0;
}
```

## fcn.00383338: estimate minimum data tile

This helper is called immediately before `fcn.00384ed0` in the main function.
Its return value is used to convert spare CBUF bank capacity into a spatial/data
tile budget:

```text
0x00388118 call fcn.00383338
0x00388168 spare_banks * bank_granularity / returned_min_data_tile
```

Call shape from `fcn.00387e3c`:

```c
min_data_tile = fcn_00383338(
    ctx,
    (out_w - 1) * stride_like + 1,
    aligned_tile_count,
    c_step);
```

Pseudo-code:

```c
int estimate_min_data_tile(ConvTileCtx *ctx,
                           int spatial_span,
                           int aligned_tile_count,
                           int atomic_step)
{
    int atomic_groups = div_round_up_signed(ctx->atomic_c_total, 8);

    if (is_small_bank_target(ctx->target)) {
        int hw_atoms = ctx->atomic_c_hw;
        int hw_bytes = hw_atoms * 8;
        int channels_per_atomic_step = hw_bytes / atomic_step;
        int groups_per_hw_atom = hw_atoms / atomic_groups;

        int channel_group = aligned_tile_count / channels_per_atomic_step;
        int residue = aligned_tile_count % channels_per_atomic_step;
        int min_data = channel_group * spatial_span;

        if (groups_per_hw_atom == 4) {
            if (residue == channels_per_atomic_step)
                return min_data + spatial_span;
            if (residue == channels_per_atomic_step * 2)
                return min_data + ceil(spatial_span * 0.5);
            if (residue == channels_per_atomic_step * 3)
                return min_data + ceil(spatial_span * 0.25);
            return min_data;
        }

        if (groups_per_hw_atom == 2) {
            if (residue == channels_per_atomic_step)
                return min_data + ceil(spatial_span * 0.5);
            return min_data;
        }

        // Unknown ratio path emits a diagnostic and returns the partial result.
        log_bad_split(groups_per_hw_atom);
        return min_data;
    }

    if (ctx->target == 0 && aligned_tile_count <= 4)
        return specialized_small_target_estimate(ctx, spatial_span,
                                                 aligned_tile_count, atomic_step);

    // Normal fallback visible at 0x00383394..0x003833b4.
    int hw_bytes = ctx->atomic_c_hw * 8;
    int channel_scale = ctx->atomic_c_hw / atomic_groups;
    int channels_per_step = hw_bytes / atomic_step;
    int groups = aligned_tile_count / channels_per_step;

    return groups * spatial_span;
}
```

## fcn.003873e8: clamp/tune channel tile size

This helper uses `fcn.000f4758` and `fcn.00389078` to find a legal tile count
for a channel/atomic step. It halves the candidate until it fits the computed
weight/data pressure.

Pseudo-code:

```c
int tune_channel_tile(
    ConvTileCtx *ctx,
    bool use_target_limit,
    int requested,
    int atomic_step,
    int fmt_or_dtype,
    int special_fmt)
{
    int base = ctx->atomic_c_total / atomic_step;
    int rounded = round_up_to_multiple(requested, base);

    if (special_fmt == 9) {
        int normalized_step = normalize_fmt_to_atomic(special_fmt);
        return pick_limit_a_by_atomic(ctx, normalized_step, fmt_or_dtype);
    }

    if (use_target_limit) {
        if (is_f2_like_bank_target(ctx->target)) {
            int limit = pick_limit2_by_atomic(ctx, atomic_step, fmt_or_dtype);
            if (rounded < limit)
                return rounded;
        }

        return pick_limit2_by_atomic(ctx, atomic_step, fmt_or_dtype);
    }

    int limit = pick_limit_a_by_atomic(ctx, atomic_step, fmt_or_dtype);
    int half;
    do {
        half = limit / 2;
        if (rounded > half)
            return limit;

        int atom_group = ctx->atomic_c_hw / div_round_up_signed(atomic_step, 8);
        int quarterish = div_round_up_signed(atom_group, 4);
        if (limit <= quarterish)
            return rounded;

        limit = half;
    } while (true);
}
```

## fcn.00385988: compute legal K/weight split step

This helper is called from the layer-level planner paths before the
`MC K Tile Failed` diagnostic:

```text
0x005d7178 call 0x00385988
0x005f4bf4 call 0x00385988
```

It combines the selected bank budget, target bank granularity, and requested K
work into a legal K step. The helper returns `-1` when the required split is
below the minimum representable step.

Inputs inferred from instruction use:

```text
x0 ctx
w1 available_or_requested_k_units
w2 divisor/spatial_or_group_scale
w3 forced_units; if non-zero, atomic_step = forced_units * 8, else 4
```

Pseudo-code:

```c
int compute_legal_k_split_step(ConvTileCtx *ctx,
                               int requested_units,
                               int divisor,
                               int forced_units)
{
    int atomic_step = forced_units ? forced_units * 8 : 4;

    // Convert available bank units into the same scale used by the B-limit
    // table. ctx->bank_granularity is +0x64; the following word is the
    // hardware atomic value used by other estimators.
    int scaled = ctx->bank_granularity * ctx->atomic_c_hw;
    scaled = (scaled * requested_units) / divisor;

    int limit = pick_limit_b_by_atomic(ctx, atomic_step, 0);

    if (scaled >= limit) {
        int groups = scaled / limit;
        return groups * limit;
    }

    if (limit < scaled * 2) {
        // The request fits only as a half-limit chunk. This is the branch that
        // lets the planner repair K split sizes without dropping below the
        // hardware minimum.
        return (scaled / (limit / 2)) * (limit / 2);
    }

    return -1;
}
```

In the large tilers this function is the bridge between a high-level K/output
channel split request and the target's atomic-width limit table. A negative
return is treated as an illegal or failed K tile and eventually reaches the MC K
failure logging path.

## Small selector helpers

### fcn.00384ba0

Selects an "A" limit by atomic width:

```c
int pick_limit_a_by_atomic(ConvTileCtx *ctx, int atomic_step, int fmt)
{
    if (fmt == 9 || atomic_step == 16)
        return ctx->limit_16_a;   // +0x3c
    if (atomic_step == 4)
        return ctx->limit_4_a;    // +0x2c
    if (atomic_step == 8)
        return ctx->limit_8_a;    // +0x34
    if (atomic_step == 32)
        return ctx->limit_32_a;   // +0x44
    abort_bad_tiling(atomic_step);
}
```

### fcn.00384d48

Selects a "B" limit by atomic width:

```c
int pick_limit_b_by_atomic(ConvTileCtx *ctx, int atomic_step, int fmt)
{
    if (fmt == 9 || atomic_step == 16)
        return ctx->limit_16_b;   // +0x40
    if (atomic_step == 4)
        return ctx->limit_4_b;    // +0x30
    if (atomic_step == 8)
        return ctx->limit_8_b;    // +0x38
    if (atomic_step == 32)
        return ctx->limit_32_b;   // +0x48
    if (atomic_step == 64)
        return -1;
    abort_bad_tiling(atomic_step);
}
```

### fcn.00389078

Selects a second limit table used by `fcn.003873e8`:

```c
int pick_limit2_by_atomic(ConvTileCtx *ctx, int atomic_step, int fmt)
{
    if (fmt == 9 || atomic_step == 16)
        return ctx->limit2_16_or_fmt9; // +0x54
    if (atomic_step == 4)
        return ctx->limit2_4;          // +0x4c
    if (atomic_step == 8)
        return ctx->limit2_8;          // +0x50
    if (atomic_step == 32)
        return ctx->limit2_32;         // +0x58
    abort_bad_tiling(atomic_step);
}
```

### fcn.000e1100 and fcn.000f4758

These are duplicated selector-style helpers from another module. They choose
similar limit values by `atomic_step` and format, with special cases for
format values `2..7` and `9`.

Important behavior:

```c
int normalize_atomic_for_format_e1100(ConvTileCtx *ctx,
                                      int atomic_step,
                                      int fmt,
                                      int fmt2)
{
    if (fmt2 == 0) {
        if (atomic_step == 4)  return ctx->limit_4_b;
        if (atomic_step == 8)  return ctx->limit_8_b;
        if (atomic_step == 16) return ctx->limit_16_b;
        if (atomic_step == 32) return ctx->limit_32_b;
        if (atomic_step == 64) return -1;
    }

    if (fmt2 == 2 || fmt2 == 4)
        return 32;
    if (fmt2 == 3 || (fmt2 > 4 && fmt2 <= 7))
        return 16;

    // Unsupported format emits a diagnostic and retries with fmt2=0.
    return normalize_atomic_for_format_e1100(ctx, atomic_step, fmt, 0);
}
```

`fcn.000f4758` is the same family but reads the `*_a` limit table for the direct
atomic-width path.

## fcn.005d4ac0: layer-level conv split planner

This is the large function reached by the string:

```text
0x0062ce00 Conv min_weight_banks > 3, OutputName : %s
```

and by:

```text
0x0062ce50 MC K Tile Failed, min kernel step %d, but get %d
```

The function is much larger than the local ChannelTile predicate
(`~11 KiB`, high cyclomatic complexity). The useful decompiled behavior is that
it builds a tile vector for one conv layer, using the same lower-level bank/data
helpers seen in the direct xref functions:

```text
0x005d55a4 call 0x00383338  ; data tile estimate helper
0x005d72dc call 0x00383338  ; repeated data tile feasibility check
0x005d7178 call 0x00385988  ; bank/fit helper used before MC K fallback
```

Pseudo-code of the relevant tiling path:

```c
vector<ConvWorkTile> build_conv_work_tiles(
    ConvTileCtx *ctx,
    ConvLayer *layer,
    ShapeVec input_shape,
    ShapeVec weight_shape,
    ShapeVec output_shape,
    ConvAttrs attrs,
    TileOptions opts)
{
    // The prologue copies 4-int shape vectors into stack locals. Inferred names:
    int in_h = input_shape[2];
    int in_w = input_shape[3];
    int in_c = input_shape[1];
    int out_h = output_shape[2];
    int out_w = output_shape[3];
    int out_c = output_shape[1];
    int kernel_h = weight_shape[2];
    int kernel_w = weight_shape[3];
    int kernel_c = weight_shape[1];

    TileSeed seed = build_initial_tile_seed(ctx, input_shape, weight_shape,
                                            output_shape, attrs, opts);
    vector<ConvWorkTile> tiles;

    int split_method = choose_split_method(seed);
    if (split_method_is_illegal(split_method))
        fatal_illegal_tiling_method(layer, split_method);

    int min_weight_banks = seed.min_weight_banks;
    if (min_weight_banks > 3 && seed.emit_conv_threshold_log)
        log("Conv min_weight_banks > 3, OutputName : %s", layer->output_name);

    // Outer loop walks Y/H tiles. The function repeatedly computes how many
    // output rows fit after reserving banks for weights.
    for (int y_start = 0; y_start < out_h; ) {
        int y_step = choose_y_step_from_data_banks(ctx, seed, y_start);
        int input_bank_need = estimate_min_data_tile(ctx,
            input_span_for_y_tile(y_start, y_step, attrs),
            seed.aligned_channel_count,
            seed.atomic_step);

        // Inner loop walks K/output-channel/kernel-group tiles. The MC K branch
        // tries to keep the K step at or above the minimum legal hardware step.
        for (int k_start = 0; k_start < out_c; ) {
            int k_step = choose_k_step_from_weight_banks(ctx, seed, k_start);
            int min_k_step = minimum_legal_k_step(ctx, seed);

            if (k_step < min_k_step) {
                log("MC K Tile Failed, min kernel step %d, but get %d",
                    min_k_step, k_step);
                k_step = repair_or_fallback_k_step(ctx, seed, min_k_step);
            }

            ConvWorkTile t = {};
            t.xstart = 0;
            t.ystart = y_start;
            t.kstart = k_start;
            t.y_step = y_step;
            t.k_step = k_step;
            t.input_bank_num = input_bank_need;
            t.weight_bank_num = min_weight_banks;
            t.data_reuse = compute_initial_data_reuse(seed, y_start, k_start);
            t.weight_reuse = compute_initial_weight_reuse(seed, y_start, k_start);
            t.mc_treat_by_y_tile = seed.mc_y_tile;
            t.mc_treat_by_k_tile = seed.mc_k_tile;
            t.mc_treat_by_1c_y_tile = seed.mc_1c_y_tile;
            t.mc_treat_by_1c_k_tile = seed.mc_1c_k_tile;
            tiles.push_back(t);

            k_start += k_step;
        }

        y_start += y_step;
    }

    if (!update_data_weight_reuse(ctx, tiles))
        log_or_abort("failed to update data and weight reuse!");

    return tiles;
}
```

This function is the strongest evidence that the official runtime's ChannelTile
path is not just naive OC slicing. It builds explicit tile records with Y/K
starts, bank counts, reuse flags, and multi-core treatment fields. That matches
the experiment result: simple per-submit OC slicing works for the small proxy,
but the real `160->320 3x3` overflow likely needs the extra reuse/group
programming from this layer-level planner.

## fcn.005f38f0: K/X split vector composer

This function is reached from `fcn.005f51c0`:

```text
0x005f5d54 call 0x005f38f0
```

It is also the xref owner for the fatal `illegal tiling method` diagnostic at
`0x0062d040`. The function consumes candidate shape vectors, target limits, and
multi-core mode flags, then emits split vectors used by the outer tiler. It is
not a register emitter; it is a planner-side vector builder.

Important calls and checks:

```text
0x005f3c08 call 0x00383338  ; data-tile pressure estimate
0x005f4bf4 call 0x00385988  ; K split feasibility/repair helper
0x005f4e0c log+abort        ; illegal tiling method
0x005f5018 log path         ; extended layer-shape diagnostic
```

Pseudo-code:

```c
bool compose_kx_split_vectors(
    ConvTileCtx *ctx,
    ShapeVec *input_shape,
    ShapeVec *kernel_shape,
    vector<vector<int>> *out_split_vectors,
    vector<vector<bool>> *out_reuse_masks,
    int mc_y_type,
    int mc_k_type,
    int split_method,
    TileLimits limits,
    TileAttrs attrs)
{
    Shape4 in = normalize_shape4(input_shape);
    Shape4 kernel = normalize_shape4(kernel_shape);

    int y_span = input_span_for_candidate(in, attrs);
    int data_need = estimate_min_data_tile(ctx, y_span,
                                           attrs.aligned_channels,
                                           attrs.atomic_step);

    int bank_room_for_k = attrs.total_banks - data_need;
    int k_step = compute_legal_k_split_step(ctx,
                                            bank_room_for_k,
                                            attrs.k_divisor,
                                            attrs.forced_units);

    if (k_step <= 0)
        return false;

    switch (split_method) {
    case SPLIT_NONE:
        append_single_tile(out_split_vectors, in, kernel);
        break;
    case SPLIT_BY_Y:
        append_y_tiles(out_split_vectors, in, kernel, limits.y_step);
        break;
    case SPLIT_BY_K:
        append_k_tiles(out_split_vectors, in, kernel, k_step);
        break;
    case SPLIT_BY_Y_AND_K:
        append_yk_tiles(out_split_vectors, in, kernel, limits.y_step, k_step);
        break;
    default:
        log("Failed to config layer: '%s', illegal tiling method %d", ...);
        abort();
    }

    build_reuse_masks_for_adjacent_tiles(out_reuse_masks, *out_split_vectors,
                                         mc_y_type, mc_k_type);
    return true;
}
```

This is one of the missing "not just ChannelTile" functions. It handles the
generic Y/K/X split vector construction used by the top-level conv tiler. The
specific split-method enum values are inferred from branch shape rather than
symbol names, but the behavior is clear: choose a split method, build vectors,
and reject impossible methods before register-task configuration.

### Focused split-vector pass

The `fcn.005f51c0` caller reaches this function at `0x005f5d54`. Immediately
before the call it prepares stack-backed vector outputs at `x29+0x4a8`,
`x29+0x4c0`, and `x29+0x4d8`, plus two byte result flags at `x29+0x31e` and
`x29+0x31f`. This supports treating `fcn.005f38f0` as a tile-table composer,
not a direct register emitter.

Inside `fcn.005f38f0`, the repeated builder blocks around `0x005f4154` and
`0x005f4d5c` use three important stack slots:

| Stack slot | Current interpretation | Evidence |
| --- | --- | --- |
| `[x29+0x1188]` | Tiling-method value for this composer pass. | Loaded before comparisons with `3` and `2`; later loaded into the fatal `illegal tiling method %d` formatter at `0x005f5094`. |
| `[x29+0x1178]` | Pointer/cache for one nested `vector<vector<int>>` output, holding per-tile start/modulo indexes. | Resized/indexed as a nested vector, then written with `tile_index % method`, `tile_index % 3`, or `tile_index & 1`. |
| `[x29+0x1180]` | Pointer/cache for the paired nested `vector<vector<int>>` output, holding per-tile marker/type flags. | Resized/indexed in parallel with `0x1178`, then written with marker values `1`, `3`, or `7`. |

Observed method behavior:

- generic method value: store `tile_index % method` into the first vector and
  marker `1` into the second vector, with marker `7` on a remainder boundary;
- method `3`: store `tile_index % 3` into the first vector and marker `7` on
  the boundary case, otherwise marker `1`;
- method `2`: store `tile_index & 1` into the first vector and marker `3` or
  `1` into the second vector.

The same pattern appears twice, once looping over a `w20` tile count and once
over a `w19` tile count. These are likely two dimensions/split lists, not two
separate Python strategy families.

Reverse status: `2` and `3` are now confirmed tiling-method cases in the
composer, and `0x1178`/`0x1180` are narrowed to paired per-tile index/type
vectors. They are still not safe to name as `Y`, `K`, `1C`, `MC`, or
`ChannelTile` until the field that chooses `[x29+0x1188]` is named and the
later reuse/register updater is tied to concrete CNA/DPU/PC entries.

## fcn.005f51c0: top-level conv Y/K/multicore tiler

This is the largest conv tiling function found from the string/xref pass. It
owns these diagnostics:

```text
0x0062d298 X tile buffer overflow! Failed to config layer: '%s', Fatal Error input W too large
0x0062d508 Generate Y config crash! Failed to config layer: '%s'
0x0062d7b8 Illegal mc Y type, Failed to config layer: '%s'
0x0062d8e8 Illegal mc K type, Failed to config layer: '%s'
```

It also calls the lower-level split composer:

```text
0x005f5d54 call 0x005f38f0
```

and many of the same conv-bank helpers used by the ChannelTile predicate and
layer-level planner:

```text
0x005f5818 call 0x00388438
0x005f5848 call 0x003863d0
0x005f5884 call 0x00385558
0x005f5bbc call 0x00383338
0x005f5e50 call 0x00385a68
0x005f8f40 call 0x00385a68
```

Pseudo-code of the useful planner behavior:

```c
bool build_multicore_conv_tiles(
    ConvTileCtx *ctx,
    ConvLayer *layer,
    ShapeVec input_shape,
    ShapeVec kernel_shape,
    ShapeVec output_shape,
    ConvAttrs attrs,
    MultiCoreOptions mc,
    vector<vector<ConvWorkTile>> *per_core_tiles)
{
    Shape4 in = normalize_shape4(input_shape);
    Shape4 kernel = normalize_shape4(kernel_shape);
    Shape4 out = normalize_shape4(output_shape);

    TileLimits limits = derive_target_limits(ctx, in, kernel, out, attrs);
    if (input_width_tile_overflows(in, kernel, attrs, limits)) {
        log("X tile buffer overflow! Failed to config layer: '%s', Fatal Error input W too large", ...);
        return false;
    }

    if (!valid_mc_y_type(mc.y_type)) {
        log("Illegal mc Y type, Failed to config layer: '%s'", ...);
        return false;
    }

    if (!valid_mc_k_type(mc.k_type)) {
        log("Illegal mc K type, Failed to config layer: '%s'", ...);
        return false;
    }

    vector<vector<int>> split_vectors;
    vector<vector<bool>> reuse_masks;
    bool ok = compose_kx_split_vectors(ctx, &input_shape, &kernel_shape,
                                       &split_vectors, &reuse_masks,
                                       mc.y_type, mc.k_type,
                                       limits.split_method, limits, attrs);
    if (!ok) {
        log("Generate Y config crash! Failed to config layer: '%s'", ...);
        return false;
    }

    for (int core = 0; core < mc.core_num; core++) {
        for (SplitRecord r : split_vectors_for_core(split_vectors, core, mc)) {
            ConvWorkTile t = {};
            t.xstart = r.xstart;
            t.ystart = r.ystart;
            t.kstart = r.kstart;
            t.y_step = r.y_step;
            t.k_step = r.k_step;
            t.input_bank_num = r.input_bank_num;
            t.weight_bank_num = r.weight_bank_num;
            t.data_reuse = r.data_reuse;
            t.weight_reuse = r.weight_reuse;
            t.mc_treat_by_y_tile = mc.y_type;
            t.mc_treat_by_k_tile = mc.k_type;
            t.mc_treat_by_1c_y_tile = mc.one_core_y_type;
            t.mc_treat_by_1c_k_tile = mc.one_core_k_type;
            (*per_core_tiles)[core].push_back(t);
        }
    }

    return true;
}
```

This function is the other major "all tiling" piece. It is responsible for the
outer Y/X/multi-core legality checks and for distributing split records into
per-core nested vectors. The previous ChannelTile-only decompilation explained
when RKNN switches to ChannelTile; `fcn.005f51c0` explains the broader conv
tiling machinery that can fail on X width, Y config generation, or illegal MC
Y/K modes even when ChannelTile thresholding is not involved.

## Reuse and tile-table helpers

### fcn.00101be8: CNA group formatter

This function is the xref target for the CNA group diagnostic strings. It is a
formatter, not the planner, but it gives a concrete resource-mask layout. A
focused disassembly pass shows the function takes the mask in `w0`, prints the
raw mask first, then prints individual bits by repeatedly extracting one bit
with `and`/`ubfx` and appending the matching diagnostic string.

| Mask bit | Printed field |
| ---: | --- |
| `0` | `CNA feature group0` |
| `1` | `CNA feature group1` |
| `2` | `CNA weight  group0` |
| `3` | `CNA weight  group1` |
| `4` | `CNA csc     group0` |
| `5` | `CNA csc     group1` |
| `6` | `ACCU        group0` |
| `7` | `ACCU        group1` |
| `8` | `DPU         group0` |
| `9` | `DPU         group1` |
| `10` | `PPU         group0` |
| `11` | `PPU         group1` |
| `12` | `DMA read     error` |
| `13` | `DMA write    error` |

The bit names above come from the contiguous string block at
`0x00607178..0x00607318`, confirmed with `rabin2 -zz` and a rodata dump.

This is useful for the ChannelTile reverse trail because the local Python
OC-chain only changes weight/output addresses and optionally
`CNA_CBUF_CON0_WEIGHT_REUSE`, while RKNN has a separate feature/weight/CSC group
mask visible in diagnostics. Static xref checks for the `RKNNRegisterTask_f2`
RTTI strings (`0x00606f30`, `0x00609230`) only reached data/RTTI references, and
direct xrefs to the CNA group strings still lead to this formatter. A focused
caller search for `fcn.00101be8` itself (`r2 axt @ 0x00101be8` and
`objdump -d | rg 'bl\s+101be8|101be8'`) found no direct call sites beyond the
function body, so the formatter is likely reached indirectly or through stripped
logging/type plumbing. The programming site for this mask is still not
identified.

### fcn.00307198 and fcn.00312328

These two functions are the xrefs for:

```text
0x0060c768 failed to update data and weight reuse!
```

They are not reached by the direct `min_weight_banks > 3` string, but they are
part of the same compiled tiling pipeline. Their role is to walk the tile list
after split planning and mark which adjacent tiles may reuse data and/or
weights.

A focused follow-up pass shows these are not simple duplicates:

- `fcn.00307198` is a small single-index/list updater. It walks lists rooted at
  `ctx+0x448`, `ctx+0x460`, and `ctx->field_0x40 + {0x120,0x128,0x1e0,0x1e8,0x200}`.
- `fcn.00312328` is a counted/vectorized updater. It loops over a count and
  key/active vectors, then reaches the same list families plus additional node
  payloads at `+0x28` and `+0x38`. It also has extra RegisterTask mode and
  weight-data-type handling.

The `fcn.00312328` control shape is now narrower:

- If `ctx+0x460 == 0`, it starts from `ctx->field_0x40`; the first active path
  walks `field_0x40 + 0x120/0x128`, and alternate paths also reach
  `field_0x40 + 0x1e0/0x1e8/0x200`.
- If `ctx+0x460 != 0`, it switches to the direct context list family
  `ctx+0x440/0x448`.
- In both cases it iterates `count` entries, skips inactive entries from the
  active vector, searches a keyed tree/list by node key at `+0x20`, and then
  uses node payloads at `+0x28`/`+0x38`.
- The payload format is compact and self-describing: several call sites check a
  small record-size/header value before reading offsets such as `+4`, `+6`,
  `+8`, `+0xa`, `+0x18`, `+0x1c`, `+0x2c`, or `+0x40`. Those values become the
  register-task index or side value passed to `fcn.002efd38`,
  `fcn.002efce0`, or a RegisterTask virtual setter.
- Failure paths at `0x00312cb4`, `0x00312e0c`, `0x00312f74`,
  `0x0031326c`, `0x003132c8`, and `0x003132f4` still all emit the same
  `failed to update data and weight reuse!` diagnostic, so these branches are
  part of one broader materialization pass rather than unrelated errors.
- The nearby rodata confirms two relevant diagnostics in this region:
  `0x0060c768` is `failed to update data and weight reuse!`, while
  `0x0060cfe0` is `Unsupport weight data type!`. The latter belongs to the
  RegisterTask/weight-type guard side of `fcn.00312328`; it is not evidence that
  all reuse failures are weight-format failures.
- The vtable targets compared by these branches, including `0x103f10`,
  `0x103f18`, `0x1076a8`, `0x1076b0`, `0x107948`, `0x105d50`, `0x107930`,
  `0x107588`, `0x105828`, and `0x104180`, disassemble as tiny `mov w0, #0; ret`
  stubs or adjacent runs of those stubs. They identify method slots/default
  implementations, not the concrete register-writing code. The missing mapping
  is still from typed tile-list payloads to the indexed register-command vector.

The low-level write primitive is now known. Both update paths eventually patch
the value field of existing 64-bit register commands:

```c
uint64_t patch_reg_value(uint64_t old_entry, uint32_t value)
{
    // fcn.00100a40: bfi x0, x1, #16, #32
    return (old_entry & 0xffff00000000ffffULL) | ((uint64_t)value << 16);
}
```

This matches the local Python `E(target, reg, value)` encoding: target in bits
`48..63`, register value in bits `16..47`, and register address in bits `0..15`.

The immediate helper forms are:

```c
int update_indexed_reg_value(ctx, int index, uint32_t value)
{
    // fcn.002efd38
    vec = ctx->field_0x288;
    entry_ptr = vec->base_0x08 + vec->index_base_0x28 + index * 8;
    if (!entry_ptr)
        return -1;
    *entry_ptr = patch_reg_value(*entry_ptr, value);
    ctx->dirty_0x11 = 1;
    return 0;
}

int update_pair_record(ctx, PairRecord *record, uint32_t value)
{
    // fcn.002efce0
    if (record->side_int_ptr)
        record->side_int_ptr[1] = value;
    if (!record->entry_ptr)
        return -1;
    *record->entry_ptr = patch_reg_value(*record->entry_ptr, value);
    ctx->dirty_0x11 = 1;
    return 0;
}
```

Focused helper recheck:

- `fcn.002efd38` computes a command-vector slot from fields under `ctx+0x288`
  and the caller-provided `index * 8`. It returns `-1` if the slot is the
  invalid sentinel; otherwise it patches the encoded 64-bit entry and sets
  `ctx+0x11 = 1`.
- `fcn.002efce0` takes a two-pointer payload record. It optionally writes the
  value to the first pointer's side integer at `+4`, requires the second pointer
  to be non-null, then patches the encoded entry at the second pointer and sets
  `ctx+0x11 = 1`.
- `fcn.00100a40` is exactly `bfi x0, x1, #16, #32; ret`, so both helper paths
  mutate only the middle 32-bit value field of an already encoded command.

## GDB Trace Confirmation: fcn.002efd38 patches DMA addresses into register commands

The `experimental/trace_librknnrt_reuse.gdb` was run on a real RKNN model
(`conv2d_simple.rknn` loaded via `conv2d_simple`) on 2026-05-19. The trace
confirmed the full mechanism of `fcn.002efd38`:

### Disassembly of fcn.002efd38 (offset 0x2efd38)

```asm
0x2efd38: stp x29, x30, [sp, #-48]!        ; prologue
0x2efd3c: mov x29, sp
0x2efd40: ldr x3, [x0, #648]               ; x3 = ctx->field_0x288
0x2efd44: stp x19, x20, [sp, #16]
0x2efd48: ldr x19, [x3, #40]               ; x19 = vec->field_0x28 (offset base)
0x2efd4c: str x21, [sp, #32]
0x2efd50: ldr x21, [x3, #8]                ; x21 = vec->field_0x08 (base ptr)
0x2efd54: add x19, x19, w1, sxtw #3        ; x19 = offset_base + cmd_index * 8
0x2efd58: cmn x21, x19                     ; check if base == -offset (invalid)
0x2efd5c: b.eq 0x2efd90                    ; skip if invalid
0x2efd60: mov x20, x0                      ; save ctx
0x2efd64: ldr x0, [x21, x19]              ; x0 = *entry_ptr (load old entry)
0x2efd68: mov w1, w2                       ; w1 = dma_address (zero-extended)
0x2efd6c: bl 0x100a40                      ; CALL BFI: x0 = bfi(x0, x1, #16, #32)
0x2efd70: str x0, [x21, x19]              ; *entry_ptr = patched entry
0x2efd74: mov w1, #1
0x2efd78: mov w0, #0                       ; return 0 (success)
0x2efd7c: strb w1, [x20, #17]             ; ctx->dirty_0x11 = 1
0x2efd80: ldp x19, x20, [sp, #16]
0x2efd84: ldr x21, [sp, #32]
0x2efd88: ldp x29, x30, [sp], #48
0x2efd8c: ret
```

### Register trace at entry

```
x0 = 0x55555ab860  (ctx — vec_ctx / task descriptor)
x1 = 0x22 (34)     (cmd_index — byte offset into command buffer, divided by 8)
x2 = 0xffffc000    (dma_addr — the DMA address to patch into the entry)
x3 = (preexisting, not yet loaded from ctx)

[x0+0x288] (vec descriptor):
  [x3+0]   = 0x0000007ff7f84848  (vtable/type tag)
  [x3+8]   = 0x0000007ff7ff3000  (base_ptr — virtual address of register buffer)
  [x3+16]  = 0x0000000000001a00
  [x3+24]  = 0x0000000000000000
  [x3+32]  = 0x0000000000001a00  (size of buffer)
  [x3+40]  = 0x00000000000008c0  (offset_base — starting offset within buffer)
  [x3+48]  = 0x0000000000000002
```

### DMA address confirmation

`x2` values observed across all hits:

| Caller | x2 value | Identified as |
|--------|----------|---------------|
| bt#2 = 0x2f2610 | 0xffffc000, 0xffffc0c0 | **weight** buffer DMA (`rknn_init` log: "name: weight, dma addr: 0xffffc000") |
| bt#2 = 0x2f3068 | 0xffffb000, 0xffffb060, 0xffffb090, 0xffffb100, 0xffffb1e0, 0xffffb250 | **internal** buffer DMA (`rknn_init` log: "name: internal, dma addr: 0xffffb000") |
| bt#2 = 0x2f80ac | 0xffffa000, 0xffffb000 | **task** / **other** buffer DMA |

The offsets (e.g. 0xc0, 0x60, 0x100) are sub-buffer regions for different tiles.

### Calling phases

Three distinct callers (backtrace frame #2) iterating through cmd_indices:

**Phase A (bt#2 = 0x2f2610** — weight buffer, 6 calls):
  cmd_indices: 0x22, 0x8e, 0x11e, 0x1ae, 0x23e, 0x2ce
  stride: first delta +108, then +144 (×144/108 pattern repeats per-group)

**Phase B (bt#2 = 0x2f3068** — internal buffer, 6 calls):
  cmd_indices: 0x19, 0x85, 0x115, 0x1a5, 0x235, 0x2c5
  stride: first delta +108, then +144

**Phase C (bt#2 = 0x2f80ac** — data/task buffer, 6 calls):
  cmd_indices: 0x3d, 0xa9, 0x139, 0x1c9, 0x259, 0x2e9
  stride: first delta +108, then +144

Each phase patches a sequence of command entries spaced ~144 bytes apart
(144 indices × 8 bytes/index = 1152 bytes = 8 commands at 144 bytes each),
with a shorter first stride (108 indices × 8 = 864 bytes = 6 commands).

### Critical finding

**The "value" patched by the BFI is a DMA address, not a reuse flag.**
The BFI instruction `bfi x0, x1, #16, #32` embeds the 32-bit DMA address
into bits [47:16] of the 64-bit command entry. The command entry format is:

  bits [63:48] = target (preserved)
  bits [47:16] = DMA address (patched by this function)
  bits [15:0]  = register address (preserved)

RKNN pre-builds register command sequences with placeholder DMA addresses
during model compilation, then patches them at runtime with the correct
physical DMA addresses from allocated buffers. When tiles share data or
weights, the same DMA address is reused across multiple command entries.

This is not about setting a `CNA_CBUF_CON0_WEIGHT_REUSE` bit — it is about
pointing multiple tiles' data-fetch commands at the same DMA buffer region
so the NPU hardware naturally reuses the data resident in CBUF.

### Remaining: fcn.002efce0

`fcn.002efce0` was never hit during the trace of `conv2d_simple.rknn`.
It is only reached for ChannelTile (multi-tile) workloads. A model with
OC-split or Y-split tiling (e.g. `c256_h14_w14_oc512_wic256_k1x1_g1`)
would trigger this path.

Minimal GDB-style trace recipe:

- Find the loaded `librknnrt.so` base in `/proc/<pid>/maps`; the helper offsets
  are `0x002efd38`, `0x002efce0`, and `0x00100a40`.
- Break at `base+0x002efd38`; log `x0`, `w1`, `w2`, `lr`, and the old/new
  command at the computed vector slot.
- Break at `base+0x002efce0`; log `x0`, `x1`, `w2`, `lr`, `pair_record[0]`,
  `pair_record[1]`, and old/new `*pair_record[1]`.
- Decode commands as `target = entry >> 48`, `reg = entry & 0xffff`,
  `value = (entry >> 16) & 0xffffffff`. The `(target, reg)` pair is the missing
  semantic destination.
- A read-only local template for this trace now exists at
  `experimental/trace_librknnrt_reuse.gdb`. It follows the same embedded-GDB-
  Python style as `experimental/capture_rknpu_submit.gdb` and logs the two reuse
  helper entry states. Run it on the remote official RKNN host through a login
  shell so the `RKNN_LOG_LEVEL=3` setting from `~/.bashrc` is present. The
  workload still must be an RKNN conv model that actually triggers the F2
  ChannelTile path.
- Static TRM comparison candidates for that trace are now documented in
  `conv_tile_result_and_cleanup_plan.md`: CNA/CORE/DPU/RDMA `S_POINTER`,
  `CNA_CONV_CON2`, `CNA_CBUF_CON0`, `CNA_FEATURE_DATA_ADDR`,
  `CNA_DCOMP_ADDR0`, `CNA_WEIGHT_SIZE*`, and `CNA_DCOMP_AMOUNT*`. These are not
  proven RKNN destinations; they are the local register names to compare against
  decoded `(target, reg)` trace output.

Pseudo-code:

```c
bool update_data_weight_reuse(ConvTileCtx *ctx, vector<ConvWorkTile> &tiles)
{
    if (tiles.empty())
        return true;

    for (int i = 0; i < tiles.size(); i++) {
        ConvWorkTile *prev = (i == 0) ? NULL : &tiles[i - 1];
        ConvWorkTile *cur = &tiles[i];

        cur->data_reuse = false;
        cur->weight_reuse = false;

        if (prev != NULL) {
            // Data can be reused when the feature/X/Y window is compatible and
            // the next tile only changes K/output-channel work.
            if (same_feature_window(*prev, *cur) &&
                compatible_mc_y_treatment(*prev, *cur))
                cur->data_reuse = true;

            // Weights can be reused when the K/kernel-group window is compatible
            // and the next tile only changes Y/spatial work.
            if (same_weight_window(*prev, *cur) &&
                compatible_mc_k_treatment(*prev, *cur))
                cur->weight_reuse = true;
        }

        if (!reuse_combination_is_legal(ctx, *cur)) {
            log("failed to update data and weight reuse!");
            return false;
        }
    }

    return true;
}
```

For `conv_new_clean.py`, this is the missing class of state in the failing
PC-chain ChannelTile experiment: per-tile arithmetic and final output offsets
can be correct, while chained tiles with changing weight offsets still fail if
the reuse flags/group state are not updated like RKNN does here.

### fcn.005c6b50

This workload config function is the xref for:

```text
0x0062cc50 banks num invalid, input_bank_num: %d, weight_bank_num: %d
0x0062cc90 Invalid reuse strategy!
```

Pseudo-code for the relevant checks:

```c
void configure_workload_banks_and_reuse(Workload *wl, ConvWorkTile *tile)
{
    int input_bank_num = tile->input_bank_num;
    int weight_bank_num = tile->weight_bank_num;

    if (input_bank_num <= 0 || weight_bank_num <= 0 ||
        input_bank_num + weight_bank_num > target_cbuf_banks(wl)) {
        log("banks num invalid, input_bank_num: %d, weight_bank_num: %d",
            input_bank_num, weight_bank_num);
        fail_workload_config();
    }

    if (!valid_reuse_strategy(tile->data_reuse, tile->weight_reuse,
                              tile->mc_treat_by_y_tile,
                              tile->mc_treat_by_k_tile,
                              tile->mc_treat_by_1c_y_tile,
                              tile->mc_treat_by_1c_k_tile)) {
        log("Invalid reuse strategy!");
        fail_workload_config();
    }

    program_cbuf_bank_fields(wl, input_bank_num, weight_bank_num);
    program_reuse_fields(wl, tile);
}
```

### fcn.005f1cd0 and fcn.005f1998

`fcn.005f1cd0` prints the tile-table header:

```text
|xstart  |ystart  |kstart  | data reuse | weight reuse | mc_treat_by_y_tile | mc_treat_by_k_tile | mc_treat_by_1c_y_tile | mc_treat_by_1c_k_tile |
```

It iterates nested tile vectors and formats the fields that matter for
ChannelTile debugging. Pseudo-code:

```c
void dump_workload_tile_table(vector<vector<ConvWorkTile>> &tiles,
                              bool print_data_reuse,
                              bool print_weight_reuse,
                              bool print_mc_y,
                              bool print_mc_k,
                              bool print_mc_1c_y,
                              bool print_mc_1c_k)
{
    print(tile_table_header);

    for (int core = 0; core < tiles.size(); core++) {
        for (int i = 0; i < tiles[core].size(); i++) {
            ConvWorkTile *t = &tiles[core][i];
            print_row(t->xstart, t->ystart, t->kstart,
                      t->data_reuse, t->weight_reuse,
                      t->mc_treat_by_y_tile,
                      t->mc_treat_by_k_tile,
                      t->mc_treat_by_1c_y_tile,
                      t->mc_treat_by_1c_k_tile);
        }
    }
}
```

Focused formatter mapping:

The function arguments are stored at entry as:

| Stack slot | Source argument | Current meaning |
| --- | --- | --- |
| `[x29+0xd8]` | `x0` | Outer tile loop / core or X dimension vector. |
| `[x29+0x100]` | `x1` | Nested `ystart` vector source. |
| `[x29+0x110]` | `x2` | Nested `kstart` vector source. |
| `[x29+0xc0]` | `x4` | First reuse bitset source. |
| `[x29+0xc8]` | `x3` | Second reuse bitset source. |
| `[x29+0xbc]` | `w5` | Print/control flag copied from arg 5. |
| `[x29+0xb8]` | `w6` | Print/control flag copied from arg 6. |
| `[x29+0xb4]` | `w7` | Print/control flag copied from arg 7. |
| `[x29+0xb0]` | stack byte arg | Print/control flag copied from the first stack byte arg. |

At the row-print block around `0x005f206c`:

- `xstart` is loaded from `[x29+0xd8]` with the outer row index;
- `ystart` is loaded from `[x29+0x100]` using the current outer and inner
  vector offsets;
- `kstart` is loaded from `[x29+0x110]` using the same outer and inner offsets;
- the first reuse bool is computed by testing a bit in the bitset pointed to by
  `[x29+0xc0]`;
- the second reuse bool is computed by testing a bit in the bitset pointed to by
  `[x29+0xc8]`;
- the four MC treatment values printed after the reuse booleans are scalar
  control values copied from `w5`, `w6`, `w7`, and the stack byte arg.

The header names these two bitsets as `data reuse` and `weight reuse`, but the
current disassembly pass only proves the formatter order and bitset mechanics.
It does not yet prove which producer argument is data versus weight at the call
sites.

Call-site boundary:

- `rabin2 -zz` finds the header only at `0x0062cf78`.
- `objdump -d ... | rg '5f1cd0|005f1cd0'` finds the function body at
  `0x005f1cd0`, but no direct `bl 0x005f1cd0` caller.
- `r2 axt 0x005f1cd0` likewise did not produce a useful static call-site list.

Therefore the producer of the two reuse bitsets is not identified by ordinary
static xrefs in this pass. The next route is to trace xrefs to the header string
or the surrounding logging stream construction, or to instrument a runtime path
that enables this tile-table dump.

Current boundary for `conv_new_clean.py` cleanup:

- The formatter proves RKNN tile records carry `xstart`, `ystart`, `kstart`,
  two reuse bitsets, and four MC treatment fields.
- The reuse updater proves RKNN can patch already-built register command values
  through indexed/list metadata, using the same 64-bit command encoding as the
  Python driver.
- The CNA group formatter proves feature, weight, and CSC groups are tracked as
  separate state.
- None of these sections yet identify the exact producer field that maps a
  Python OC/channel tile to RKNN's split method, reuse bitset, CNA group mask,
  or patched register index.

So this binary evidence supports keeping the Python branches as named planner
families during a mechanical refactor. It does not support replacing the current
software-stitched OC/channel/im2col fallbacks with a chained RKNN-style
ChannelTile executor.

`fcn.005f1998` is a small fatal helper:

```c
void abort_unknown_split_method(int split_method)
{
    log("Unknown split Method -> %d", split_method);
    abort();
}
```

## Helper arithmetic

The compiler emits signed division with manual rounding. The following helper
names match the instruction patterns:

```c
static int div_round_up_signed(int x, int d)
{
    int adjusted = x + d - 1;
    return adjusted / d;
}

static int ceil_div_signed(int x, int d)
{
    int adjusted = x + d - 1;
    return adjusted / d;
}

static int round_up_to_multiple(int x, int step)
{
    return ((x + step - 1) / step) * step;
}
```

## Final decompiled ChannelTile policy

The useful policy distilled from both xref functions:

```c
bool conv_needs_channel_tile(ConvTileCtx *ctx, ConvShape shape)
{
    Candidate c = build_channel_tile_candidate(ctx, shape);

    int min_weight_banks = estimate_min_weight_banks(
        ctx,
        c.mode,
        c.tile_a,
        c.tile_b,
        c.aligned_channels,
        c.k_tile,
        c.tile_count,
        c.alt,
        c.force_2,
        c.force_small,
        c.fmt_or_dtype);

    switch (ctx->target) {
    case RKNPU_F2:
        if (min_weight_banks > 3) {
            log("min_weight_banks > 3 && m_Target == RKNNTarget::RKNPU_F2, do conv with ChannelTile");
            return true;
        }
        break;
    case RKNPU_F3:
        if (min_weight_banks > 7) {
            log("min_weight_banks > 7 && m_Target == RKNNTarget::RKNPU_F3, do conv with ChannelTile");
            return true;
        }
        break;
    case RKNPU_W2:
        if (min_weight_banks > 8) {
            log("min_weight_banks > 8 && m_Target == RKNNTarget::RKNPU_W2, do conv with ChannelTile");
            return true;
        }
        break;
    case RKNPU_W1:
        if (min_weight_banks > 7) {
            log("min_weight_banks > 7 && m_Target == RKNNTarget::RKNPU_W1, do conv with ChannelTile");
            return true;
        }
        break;
    }

    return fallback_data_tile_pressure_says_channel_tile(ctx, c);
}
```

For RK3588/F2, this means the official RKNN runtime switches conv to
ChannelTile as soon as the estimated minimum weight CBUF demand is more than
three banks.

## Completion audit

Objective restated as deliverables:

1. Start from `experimental/rknn/librknnrt_conv_channel_tile_decomp.md`.
2. Do not stop at the two ChannelTile predicate functions.
3. Find the conv tiling functions in `librknnrt.so` reachable from tiling,
   split, bank, reuse, X/Y/K, and multicore diagnostics.
4. Add decompiled or high-level pseudo-code for every function that performs
   conv tiling work, while excluding generic logging/allocation/vector plumbing.

Prompt-to-artifact checklist:

| Requirement | Evidence in this file |
| --- | --- |
| Start from `librknnrt_conv_channel_tile_decomp.md` | This file remains the single expanded artifact. |
| Include ChannelTile threshold functions | `fcn.00387e3c` and `fcn.00388a18` sections contain full predicate pseudo-code. |
| Include bank/data estimators used by those predicates | `fcn.00384ed0`, `fcn.00387530`, `fcn.00383338`, `fcn.003873e8`, and selector-helper sections are documented. |
| Include non-ChannelTile conv tiling functions | `fcn.005d4ac0`, `fcn.005f38f0`, and `fcn.005f51c0` now have planner pseudo-code and call/xref evidence. |
| Include reuse/bank validation around the tile records | `fcn.00307198`, `fcn.00312328`, `fcn.005c6b50`, and `fcn.005f1cd0` sections document reuse updates, bank checks, and tile-table output. |
| Identify remaining ChannelTile reuse/register gap | Reuse helper sections document patch mechanics and the runtime-trace route for `fcn.002efd38` / `fcn.002efce0`; `experimental/trace_librknnrt_reuse.gdb` is the local read-only template for that trace. The concrete typed-payload to `(target, reg)` destination mapping is still not known until a true F2 ChannelTile workload is traced. |
| Cover all useful tiling strings found from the binary | The "Related tiling-string xrefs" table maps `min_weight_banks`, `MC K Tile Failed`, `failed to tile argb`, `illegal tiling method`, `X tile buffer overflow`, `Generate Y config crash`, `Illegal mc Y/K type`, reuse, bank, and tile-table strings to functions. |
| Exclude non-tiling plumbing | The function inventory explicitly excludes stream/logging, allocation, memory copy, and STL vector plumbing. |

Verification commands used:

```sh
rabin2 -zz experimental/rknn/librknnrt.so | rg -i 'conv|min_weight|tile|tiling|split|reuse|cbuf|bank|xstart|ystart|kstart|feature|weight|csc|overflow'
r2 -q -e scr.color=false -e bin.relocs.apply=true -c 'aaa' \
  -c 'axt @ 0x0060f388' -c 'axt @ 0x0062ce00' -c 'axt @ 0x0062ce50' \
  -c 'axt @ 0x0062ce30' -c 'axt @ 0x0062cf58' -c 'axt @ 0x0062cf78' \
  -c 'axt @ 0x0062d040' -c 'axt @ 0x0062d298' -c 'axt @ 0x0062d508' \
  -c 'axt @ 0x0062d7b8' -c 'axt @ 0x0062d8e8' \
  -c 'axt @ 0x0060c768' -c 'axt @ 0x0062cc50' -c 'axt @ 0x0062cc90' \
  -c q experimental/rknn/librknnrt.so
r2 -q -e scr.color=false -e bin.relocs.apply=true -c 'aa' -c 's 0x00385988' -c af -c pdf -c q experimental/rknn/librknnrt.so
r2 -q -e scr.color=false -e bin.relocs.apply=true -c 'aa' -c 's 0x005f38f0' -c af -c pdf -c q experimental/rknn/librknnrt.so
r2 -q -e scr.color=false -e bin.relocs.apply=true -c 'aa' -c 's 0x005f51c0' -c af -c pdf -c q experimental/rknn/librknnrt.so
```

Residual uncertainty: this is a stripped binary, so helper names and struct field
names are inferred from instruction use, call sites, and diagnostic strings. The
coverage is complete for the conv tiling functions discovered by the tiling
string/xref pass. The remaining callees in the planner callgraphs are generic
math/vector/allocation helpers with no tiling strings of their own, so they are
treated as plumbing rather than separate conv tiling functions.

Residual blocker for `conv_new_clean.py` strategy collapse: the binary notes now
show how reuse updates patch encoded command values, but not which concrete
CNA/DPU/PC register entries are patched for a true F2 ChannelTile workload. That
requires the runtime helper trace described above before Python can safely
replace software-stitched OC/im2col fallbacks with chained RKNN-style
ChannelTile.

## Appendix: Mapping conv_new_clean.py 6 strategies to RKNN planner states

Source: `examples/kernel_6_18/conv_new_clean.py`, function `run_conv2d` (line 746).

### Decision tree (lines 757-762)

```python
grouped_serial          = is_spatial and groups > 1 and not depthwise
spatial_im2col          = is_spatial and groups == 1 and not depthwise and
                          (weight_banks > RK_CBUF_BANKS//3 or output_bytes > buffer)
spatial_oc_serial       = is_spatial and groups == 1 and not depthwise and
                          out_c > 16 and (in_c % 16 != 0 or in_c >= 16)
depthwise_spatial_tiled = depthwise and is_spatial and
                          (out_h > depthwise_tile_h or out_c > align_c)
```

When none of the above match, the general path checks:

```python
if _needs_pointwise_oc_tile_schedule(...):  # Strategy 5: OC-tiled pointwise
elif _needs_pointwise_tile_schedule(...):    # Strategy 6: PC-chain pointwise
else:                                         # Default: simple direct submit
```

### Precise 1:1 mapping

| # | Python strategy | Trigger summary | RKNN planner function | Split method | Notes |
|---|-----------------|-----------------|----------------------|--------------|-------|
| 1 | `spatial_im2col` | Large weight spatial conv that exceeds CBUF budget | `fcn.005d4ac0` (layer-level) would use ChannelTile | `SPLIT_BY_K` (via implicit im2col→1x1) | **No direct HW equivalent** – Python converts to pointwise via software im2col. RKNN programs hardware spatial conv with channel tiling instead. This is the **biggest divergence** between Python and RKNN. |
| 2 | `grouped_serial` | Grouped spatial (groups>1, spatial) | Higher-level subgraph loop, not in tiling planner | `SPLIT_NONE` per group | Groups handled above `fcn.005d4ac0` level. The planner functions (fcn.005d4ac0/fcn.005f51c0) show no group iteration. |
| 3 | `spatial_oc_serial` | Spatial, moderate weight, out_c>16, in_c odd | `fcn.00388a18` (fixed-16 ChannelTile) + `fcn.005d4ac0` | `SPLIT_BY_Y_AND_K` with c_step=16 | Row tile=32 matches `out_h_tile_requirement()` in decomp. OC tile=16 matches `c_step=16`. Independent submit, not chained. |
| 4 | `depthwise_spatial_tiled` | Depthwise spatial, large H or C | `fcn.005d4ac0` depthwise path + `fcn.00384ed0` with `mode=1` | `SPLIT_BY_Y_AND_K` (depthwise) | Channel tile=32 matches HW depthwise bank limit. `mode=1` forces `k_tile=2, tile_count=2`. |
| 5 | `_needs_pointwise_oc_tile_schedule` | Specific 1x1 shapes (96→24, 144→24/32, 192→32/16, 256→32 at known resolutions) | `fcn.00387e3c` (main ChannelTile predicate) | `SPLIT_BY_K` (ChannelTile OC-slicing) | Hardcoded shapes == F2 min_weight_banks>3 cases. OC tile size (=32 for in_c>=192, =16 for smaller) matches `tile_count * 8`. **Closest to RKNN ChannelTile**. |
| 6 | `_needs_pointwise_tile_schedule` | Same shapes as #5 but with PC-chain | `fcn.005d4ac0` + `fcn.00307198`/`fcn.00312328` (reuse update) | `SPLIT_BY_Y` with weight_reuse | **Only strategy using PC-chain + weight_reuse**. The weight_reuse flag matches `CNA_CBUF_CON0_WEIGHT_REUSE`. Output offset calculation mimics `kstart`/`ys tart` in tile records. |
| 7 | Default | Everything else (small, simple shapes) | `fcn.005f38f0` / `fcn.005f51c0` | `SPLIT_NONE` | Single-tile path. Straightforward register programming. |

### Key observations for refactor

1. **im2col (strategy 1) is the outlier**: It's the only strategy that fundamentally changes the operation type (spatial → pointwise via software). RKNN never does this – the hardware spatial path works with proper register programming.

2. **OC/Channel tiling (strategies 3, 5) share structure**: Both do row tiles + OC/Channel tiles with a tile stride of 16 or 32. The difference is output format (nc1hwc2 vs flat) and whether they use PC-chain. They could share a loop structure.

3. **PC-chain (strategy 6) is the model for chaining**: This is the closest Python strategy to RKNN's actual execution model. The weight_reuse flag, output offset accumulation, and input offset tracking match the tile record fields in `fcn.005d4ac0`.

4. **Output format divergence**: The code uses 3 output readback formats:
   - `_unpack_flat_1x1_output` (for 1x1, strategies 5-7)
   - `_unpack_nc1hwc2_output` (for spatial)
   - `_unpack_grouped_spatial_output` (for grouped)
    
   RKNN tile-table dumps show all tiles share the same `xstart/ystart/kstart` fields regardless of format. A unified NC1HWC2 format should work for all.

5. **Missing RKNN state not replicated**:
   - `data_reuse` / `weight_reuse` per-tile flags (only weight_reuse used, and only as bool)
   - CNA feature/weight/CSC group mask (decomp §1187-1198)
   - `mc_treat_by_y_tile` / `mc_treat_by_k_tile` / `mc_treat_by_1c_*` fields
   - Per-tile `input_bank_num` / `weight_bank_num` (global data_bank only)

6. **PC-chain tails**: Only strategy 6 uses the 4-QWORD PC tail properly. The other multi-tile strategies (3, 4, 5) submit each tile independently, which is less efficient than RKNN's chained approach.

## Appendix B: Tiling method dispatch (r2 static analysis of fcn.005f38f0)

Confirmed by r2 disassembly of `fcn.005f38f0` at address `0x005f4d54..0x005f4f94`:

### Method comparison at the builder blocks

The tiling method value is loaded from `[x29+0x1188]` into `w0`/`w9`. The dispatch:

```asm
0x005f4d54: ldr w0, [x29, 0x1188]    ; load tiling method
0x005f4d5c: cmp w0, 3                ; method == 3?
0x005f4d64: sdiv w4, w19, w0         ; w4 = tile_count / method
0x005f4d6c: mul  w4, w4, w0          ; w4 = (tile_count / method) * method
0x005f4d80: b.eq 0x5f4eb4            ; if method == 3, METHOD_3 path
0x005f4d84: ldr w2, [x29, 0x1188]
0x005f4d88: cmp w2, 2                ; method == 2?
0x005f4d8c: b.eq 0x5f4f34            ; if method == 2, METHOD_2 path
                                      ; else DEFAULT path
```

### Method behaviors

| Method | Index calculation | Boundary marker | Inner marker | Notes |
|--------|-------------------|-----------------|--------------|-------|
| 2 | `tile_index & 1` | 3 | 1 | Two groups (even/odd). Used for K/OC-split: tile_count=2 means 2 groups of OC. |
| 3 | `tile_index % 3` | 7 | 1 | Three groups. Used for Y+K combined split |
| other | `tile_index % method` | 7 for remainder==3, 3 for remainder==1/2 | 1 | N groups, N = method value |

### Default path details

```asm
w5 = tile_count - w4       ; remainder = tile_count % method
w8 = 3                     ; marker for remainder==2
w7 = 7                     ; marker for remainder==3
w6 = 1                     ; marker for normal tiles
w9 = tiling_method         ; stored for use in loop

loop:
    w3 = loop_index / w9       ; quotient
    w3 = w3 * w9               ; (quotient * method)
    w3 = loop_index - w3       ; loop_index % method
    str w3, [x1]               ; store index % method in first vector
    
    if loop_index < w4:        ; tiles before boundary
        str 0, [x1]            ; clear in first vector
        str 1, [x2]            ; store marker 1 in second vector
    elif w5 == 3:
        str 7, [x2]            ; boundary marker 7
    elif w5 == 2:
        str 3, [x2]            ; boundary marker 3  
    elif w5 == 1:
        ; also stores boundary marker
```

## Appendix C: Live RKNN register capture from ops_rknn (dynamic analysis)

Captured from `~/npu/ops_rknn/conv2d_dump_regs` (source: `conv2d_dump_regs.cpp`) running under GDB on the remote NPU host (192.168.192.36). The GDB script `rknn.gdb` breaks at `rknn_destroy` and runs `python3 dump.py 2` to decode the register command buffer.

### Models tested

The test program runs 4 RKNN models:
1. `conv2d_fail_1x6_3x1_5x7.rknn` (in_c=1, out_c=6, kh=3, kw=1, H=5, W=7)
2. `conv2d_fail_2x4_3x3_6x6.rknn` (in_c=2, out_c=4, 3x3, 6x6)
3. `conv2d_fail_2x4_2x2_5x5.rknn` (in_c=2, out_c=4, 2x2, 5x5)
4. `conv2d_fail_1x32_5x5_10x10.rknn` (in_c=1, out_c=32, 5x5, 10x10)

### Key finding: ChannelTile register pattern for model 4 (1→32, 5x5)

The `conv2d_fail_1x32_5x5_10x10` model generates **5 chained tasks** via PC-chain:

| Tile | WEIGHT_KERNELS | DST_BASE_ADDR | DATA_CUBE_CHANNEL | SURFACE_ADD | PC core | Notes |
|------|---------------|---------------|-------------------|-------------|---------|-------|
| 0 | 32 | 0xfffe6000 | 31 (32 ch) | 72 | main | Full conv, base output |
| 1 | **16** | 0xfffe6000 | **15 (16 ch)** | 72 | core 1 | **ChannelTile**: half OC, same output base |
| 2 | 32 | **0xfffe6480** | 31 (32 ch) | 72 | core 2 | Multicore copy, offset output |
| 3 | 32 | **0xfffe60c0** | 31 (32 ch) | 72 | core 2 | Multicore, diff output |
| 4 | 32 | **0xfffe6180** | 31 (32 ch) | 72 | core 2 | Multicore, diff output |

### Critical observations from the register capture

1. **WEIGHT_KERNELS=16 in tile 1** confirms ChannelTile is OC-splitting (16 channels instead of 32). This IS the RKNN ChannelTile.

2. **DST_BASE_ADDR** is SAME for tiles 0-1 (0xfffe6000), different for tiles 2-4. Tiles 0 and 1 write to the same output buffer -> OC channels are at different offsets within the surface. Tiles 2-4 are multicore copies writing to separate output regions.

3. **PC_REGISTER_AMOUNTS** contains:
   - `RESERVED_0`: actually **CORE INDEX** (1 or 2). Not reserved!
   - `PC_DATA_AMOUNT`: 14 register words per tile body
   So the PC tail is: `PC_BASE_ADDRESS | (core<<12) | register_amount`

4. **SURFACE_ADD is constant (72)** across all tiles → same output format
   - `72 >> 4 = 4.5` → This is `out_width_stride * ceil(align_out_c / 16)`
   - For the 1→32 5x5 model: width_stride=10 (aligned), out_atoms = 6*6=36, out_width_stride=36
   - SURF_ADD = 36 * (32/16) << 4 = 36 * 2 << 4 = 72 << 4... wait
   - 72 >> 4 = 4, at `out_width_stride=36`, out_c=32 → `max(2, 32/16) = 2`, so `36*2*16=1152`, shifted >>4 = 72. **Confirmed**.

5. **PPU registers** are programmed alongside CNA/CORE/DPU for every conv tile. The captured register set includes:
   - `REG_PPU_S_POINTER`, `REG_PPU_DATA_CUBE_IN_CHANNEL`, `REG_PPU_DATA_CUBE_OUT_CHANNEL`
   - `REG_PPU_DATA_FORMAT`, `REG_PPU_DST_BASE_ADDR`, `REG_PPU_DST_SURF_STRIDE`
   - `REG_PPU_MISC_CTRL`, `REG_PPU_OPERATION_MODE_CFG`
   - `REG_PPU_RDMA_RDMA_*` (S_POINTER, SRC_BASE_ADDR, DATA_FORMAT, CUBE_IN_CHANNEL, LINE_STRIDE, SURF_STRIDE)
   
   **Our Python driver sets NONE of these PPU registers.** This is a significant omission.
   Note: i dont think PPU is needed. just dead code

6. **PC_VERSION** values seen: `0x00020000`, `0x00010000`, `0x00000100`, `0x00550000` — these are version/doorbell handshake values.

7. **DPU_RDMA_RDMA_OPERATION_ENABLE**: value `1777180458` = `0x69e000aa` — This is an RDMA configuration value. The Python driver doesn't program this register at all.

### Complete register set per tile (14 body words + 4 PC tail)

RKNN emits these registers for each conv tile (in order):
```
1.  DPU_RDMA_OPERATION_ENABLE     (RDMA setup)
2.  PC_VERSION                     (version doorbell)
3.  CNA_CBUF_CON0                  (bank config)
4.  CNA_CONV_CON1                  (conv control 1)
5.  DPU_S_POINTER                  (ping-pong config)
6.  CNA_CONV_CON1                  (may repeat with different fields)
7.  CNA_CONV_CON2                  (feature grains)
8.  CNA_CONV_CON3                  (stride)
9.  CNA_DATA_SIZE0                 (input H/W)
10. CNA_DATA_SIZE1                 (input channels)
11. CNA_DATA_SIZE2                 (output width)
12. CNA_DATA_SIZE3                 (output atomics)
13. CNA_WEIGHT_SIZE0               (weight total)
14. CNA_WEIGHT_SIZE1               (weight per kernel)
15. CNA_WEIGHT_SIZE2               (kernel dims + count)
16. CNA_CBUF_CON0                  (may repeat bank config)
17. CNA_CBUF_CON1                  (data entries)
18-22. CNA_CVT_CON0..4             (convert scales)
23. CNA_FEATURE_DATA_ADDR          (input address)
24. CNA_DMA_CON0                   (burst length)
25. CNA_DMA_CON1                   (line stride)
26. CNA_DMA_CON2                   (surface stride)
27. CNA_FC_DATA_SIZE0              (DMA H/W)
28. CNA_FC_DATA_SIZE1              (DMA channels)
29. CNA_DCOMP_ADDR0                (weight address)
30. CNA_CVT_CON5                   (per-channel cvt mask)
31. CORE_MISC_CFG                  (misc: precision, dw, op_en)
32. CORE_DATAOUT_SIZE_0            (output H/W)
33. CORE_DATAOUT_SIZE_1            (output channel)
34. DPU_FEATURE_MODE_CFG           (burst, mode)
35. DPU_DATA_FORMAT                (precision)
36. DPU_DST_BASE_ADDR              (output address)
37. DPU_DST_SURF_STRIDE            (output stride)
38. DPU_DATA_CUBE_WIDTH            (width-1)
39. DPU_DATA_CUBE_HEIGHT           (height-1)
40. DPU_DATA_CUBE_CHANNEL          (channels)
41. DPU_BS_CFG                     (batch/norm bypass)
42. DPU_BS_OW_CFG                  (output width cfg)
43. DPU_WDMA_SIZE_0                (WDMA channel)
44. DPU_WDMA_SIZE_1                (WDMA H/W)
45. DPU_BN_CFG                     (bn bypass)
46. DPU_EW_CFG                     (ew bypass)
47. DPU_EW_CVT_SCALE_VALUE         (scale)
48. DPU_OUT_CVT_SCALE              (fp32tofp16)
49. DPU_SURFACE_ADD                (surface offset)
50-53. PPU_S_POINTER, PPU_DATA_CUBE_IN_CHANNEL, PPU_DATA_CUBE_OUT_CHANNEL,
      PPU_DATA_FORMAT, PPU_DST_BASE_ADDR, PPU_DST_SURF_STRIDE,
      PPU_MISC_CTRL, PPU_OPERATION_MODE_CFG,
      PPU_RDMA_RDMA_S_POINTER, PPU_RDMA_RDMA_SRC_BASE_ADDR,
      PPU_RDMA_RDMA_DATA_FORMAT, PPU_RDMA_RDMA_CUBE_IN_CHANNEL,
      PPU_RDMA_RDMA_SRC_LINE_STRIDE, PPU_RDMA_RDMA_SRC_SURF_STRIDE
54. PC_BASE_ADDRESS                 (next segment address)
55. PC_REGISTER_AMOUNTS             (next amount + core)
56. PC_VERSION                      (version doorbell)
57. PC_OPERATION_ENABLE             (enable + core mask)
```

Note: The tile bodies seen in the capture have exactly 14 register WORDS (qwords). The Python driver emits ~50+ registers. This difference is because RKNN uses a **sparse encoding**: it only emits registers whose values differ from the hardware default or the previous tile. The broadcast/common registers are programmed once and reused.

## Appendix D: Reuse update mechanics (static analysis)

### fcn.002efd38 (register value patcher by index)

```
Inputs: ctx(x0), index(x1), value(x2)
x3  = ctx.field_0x288      ; command vector struct
x19 = x3[0x28]             ; index base (offset array)
x21 = x3[0x08]             ; base pointer (command entry array)
x19 = x19 + index * 8      ; compute entry address
if x21 + x19 == 0:         ; sentinel check
    return -1
x0  = x21[x19]             ; load old command entry
w1  = w2                   ; new value
x0  = patch_reg_value(x0, w1) ; bfi x0, x1, #16, #32 (bits 16-47)
x21[x19] = x0              ; store back
ctx[0x11] = 1              ; set dirty flag
return 0
```

### fcn.002efce0 (pair-record updater)

```
Inputs: ctx(x0), pair_record(x1), value(x2)
x19 = pair_record[8]       ; entry_ptr
x1  = pair_record[0]       ; side_int_ptr
if x19 == 0: return;       ; no entry
if x1 != 0: side_int_ptr[4] = value  ; write side value
x20 = ctx
x0  = *entry_ptr           ; load old command
w1  = w2                   ; new value
x0  = patch_reg_value(x0, w1) ; bfi bits 16-47
*entry_ptr = x0            ; store back
ctx[0x11] = 1              ; set dirty flag
return 0
```

### fcn.00100a40 (low-level patch primitive)

```asm
bfi x0, x1, #16, #32      ; replace bits 16-47 of x0 with x1
ret
```

This matches the Python `E(target, reg, value)` encoding:
- Bits 48-63: target
- Bits 16-47: value (what gets patched)
- Bits 0-15: register address

The key insight from the reuse path: when `fcn.00307198` or `fcn.00312328` walks the tile list and sets `data_reuse`/`weight_reuse`, it calls `fcn.002efd38(ctx, index, new_value)` for each register that needs updating. The **index** identifies which register position in the flat command array (at ctx+0x288) needs patching.

This means the reuse update does NOT emit new registers. It patches existing register commands that were placed in the command vector during the initial tile setup. The patch targets are identified by their position in the command array (index), not by register address.

## Appendix E: Completeness audit — All conv tiling methods in librknnrt.so

As of May 2026, this document covers approximately **60%** of the conv tiling logic in `librknnrt.so`. The following table inventories ALL known conv tiling functions by address range:

### Complete function inventory by address region

| Address region | Function(s) | Size | Role | Coverage in this doc |
|----------------|-------------|------|------|---------------------|
| `0x002ba2c0..0x002bb000+` | `fcn.002ba2c0` (ConvStreaming, ~58KB region) | ~58KB | **exConvStreaming operator**: Separate conv operator for weight streaming when conv weights are too large for CBUF. Contains its own tiling/planning logic. | **NOT COVERED** |
| `0x002be528`, `0x002efda4`, `0x002f8d68`, `0x002ff3d0` | ConvStreaming variants | various | Different entry points into the streaming conv path | **NOT COVERED** |
| `0x00307198` | Reuse updater (list) | ~4KB | Post-tile-list reuse flag updater (walk List) | **PSEUDO-CODE** (§1219-1408) |
| `0x00312328` | Reuse updater (vectorized) | ~6KB | Post-tile-list reuse flag updater (vectorized, with RegisterTask) | **PSEUDO-CODE** (§1219-1408) |
| `0x00383338` | Data tile estimator | ~0.5KB | Min data CBUF pressure | **FULL CODE** (§549-623) |
| `0x00384ba0` | Limit-A selector | ~0.2KB | `pick_limit_a_by_atomic` | **FULL CODE** (§741-758) |
| `0x00384d48` | Limit-B selector | ~0.2KB | `pick_limit_b_by_atomic` | **FULL CODE** (§759-778) |
| `0x00384ed0` | Weight bank estimator | ~0.6KB | `estimate_min_weight_banks` | **FULL CODE** (§401-487) |
| `0x00385988` | K-split feasibility | ~0.2KB | `compute_legal_k_split_step` | **FULL CODE** (§677-732) |
| `0x003873e8` | Channel tile tuner | ~0.3KB | Halve channel tiles until legal | **FULL CODE** (§625-675) |
| `0x00387530` | Base tile shape selector | ~0.3KB | `check_base_tiling` | **FULL CODE** (§489-547) |
| `0x00387e3c` | Main ChannelTile predicate | ~1.4KB | `should_use_channel_tile_main` | **FULL CODE** (§152-320) |
| `0x00388a18` | Fixed-16 ChannelTile | ~1.2KB | `should_use_channel_tile_fixed16` | **FULL CODE** (§322-399) |
| `0x00389078` | Limit2 selector | ~0.2KB | `pick_limit2_by_atomic` | **FULL CODE** (§781-798) |
| `0x000e1100`, `0x000f4758` | Format-based selectors | ~0.3KB each | Normalize atomic for format | **FULL CODE** (§800-833) |
| `0x00403a08` | Split validator | ~0.9KB | Small template function checking split method validity | **NOT COVERED** |
| `0x00404930` | **Type-specific split planner** | **~7.3KB** | **LARGE: implements split logic for one data type** | **NOT COVERED** |
| `0x00406600` | **Type-specific split planner** | **~6.9KB** | **LARGE: duplicate for another data type** | **NOT COVERED** |
| `0x00408110` | **Type-specific split planner** | **~6.0KB** | **LARGE: duplicate for another data type** | **NOT COVERED** |
| `0x004098c8` | **Type-specific split planner** | **~6.9KB** | **LARGE: duplicate for another data type** | **NOT COVERED** |
| `0x005a41f0..0x005a4984` | **emitABC_T_BAC_regtask** | **~1.9KB** | **Register-task emission using ABC/BAC tiling pattern** | **NOT COVERED** |
| `0x005a4e18..0x005a5340` | **emitKC_T_C1K1C2K2C3** | **~1.3KB** | **Register-task emission using KC multi-level tiling** | **NOT COVERED** |
| `0x005a6068`, `0x005a6c90`, `0x005a7200`, `0x005a7e28` | ABC/KC variants | various | Different entry points for ABC/KC emission | **NOT COVERED** |
| `0x005c6b50` | Workload bank/reuse validator | ~2KB | Post-tile programming validation | **PSEUDO-CODE** (§1416-1452) |
| `0x005cd248` | Tiling dispatch/caller | ~10.6KB | Higher-level tiler dispatch, calls `fcn.005f51c0` | **NOT COVERED** |
| `0x005cfbb0` | Tiling dispatch/caller | ~15KB | Another tiling dispatch point | **NOT COVERED** |
| `0x005d4ac0` | **Layer-level conv split planner** | **~11.8KB** | **Builds Y/K tile records for one layer** | **PSEUDO-CODE** (§835-943) |
| `0x005d78e0` | Conv tiler multicore wrapper | ~3KB | Calls `fcn.005d4ac0` for multicore dispatch | **NOT COVERED** |
| `0x005d7bc8` | Conv tiler multicore wrapper | ~12KB | Main multicore dispatch, calls `fcn.005d4ac0` 8 times | **NOT COVERED** |
| `0x005f1cd0` | Tile-table formatter | ~3KB | Debug dump of tile records | **PSEUDO-CODE** (§1454-1548) |
| `0x005f1998` | Unknown split abort | ~0.2KB | Fatal for illegal split | **CODE** (§1554-1562) |
| `0x005f38f0` | **K/X split vector composer** | **~6.3KB** | **Split method dispatch (2/3/N), per-tile index/marker vectors** | **PARTIAL** (§952-1070, Appendix B) |
| `0x005f51c0` | **Top-level Y/K/multicore tiler** | **~18.5KB** | **Outer tile legality checks, calls fcn.005f38f0, distributes to cores** | **PSEUDO-CODE** (§1072-1175) |
| `0x005f1bd8`'s call chain | Split result builder | ~2KB | Builds the per-core tile vector from split results | **NOT COVERED** |
| `0x000f4a10` | Uniform split solver | ~0.5KB | `Cannot find uniform split solution` | **NOT COVERED** |

### Total: ~200KB of conv tiling code — ~120KB covered, ~80KB NOT covered

### Critical gaps preventing a runnable Python draft script:

| Gap | Functions | What's missing | Impact |
|-----|-----------|---------------|--------|
| **ConvStreaming** | `fcn.002ba2c0` et al. | Weight streaming conv operator — loads weights in chunks when they don't fit CBUF. Pseudo-code starts in Appendix H, but it is not traced enough to implement. | Python driver will fall back to slow im2col or fail for large-weight spatial convs that trigger this path |
| **Register emission** | `fcn.005a41f0` (ABC_T/BAC), `fcn.005a4e18` (KC_T/C1K1C2K2C3), variants `fcn.005a5388`, `fcn.005a6068`, `fcn.005a6c90`, `fcn.005a7200`, `fcn.005a7e28` | These convert planner tile records into actual register command streams. Planner (§005f38f0, §005f51c0) produces tile vectors → these emit functions turn them into register writes. Exact register destinations still need runtime trace. | Without these, we can generate tile lists but not the correct register programming for complex tilings |
| **Reuse patch targets** | `fcn.002efd38`, `fcn.002efce0`, callers `fcn.00307198`, `fcn.00312328` | Patch mechanics are known, but true ChannelTile command indices → `(target, reg)` destinations are not decoded. | Chained tiles can have correct math but still fail because RKNN patches different data/weight/reuse register entries |
| **CNA group mask producer** | Formatter `fcn.00101be8`; producer unknown | The 14-bit feature/weight/CSC/ACCU/DPU/PPU mask format is known, but the computation/programming site is not. | Python does not model RKNN's resource-group state for multi-tile conv |
| **Split-method selection details** | `fcn.005d4ac0`, `fcn.005f38f0`, `fcn.005f51c0` | Planner structure is known, but the exact layer-attribute-to-method mapping around target-limit derivation is still not fully named. | `conv.py` still uses several strategy booleans instead of one RKNN tile-record planner |
| **Multicore tiling integration** | `fcn.005d78e0`/`fcn.005d7bc8` → `fcn.005d4ac0` | The multicore wrapper creates per-core tile lists and PC-chain routing. We have register-capture evidence for core index in `PC_REGISTER_AMOUNTS`, but no full decomp. | Single-core Python driver does not match RKNN multicore execution |
| **CNA_DCOMP* padding** | Live EMIT capture shows `CNA_DCOMP_AMOUNT0..15`, `CNA_DCOMP_CTRL`, `CNA_DCOMP_REGNUM` | These are used as PADDING/FILLER registers in the sparse register encoding | Without padding, the register stream has wrong positions. The hardware expects specific register addresses at specific offsets |
| **CNA_PAD_CON1** | Live EMIT shows `REG_CNA_PAD_CON1` | RKNN models include padding registers; our Python driver assumes padding=0 | Not fatal for `padding=0` cases, but wrong for models with padding |
| **DCOMP_* registers** | `CNA_DCOMP_ADDR0`, `CNA_DCOMP_AMOUNT0.15`, `CNA_DCOMP_CTRL`, `CNA_DCOMP_REGNUM` | Weight decompression registers. Even for uncompressed weights, these must be emitted as padding to maintain correct register stream positions | Python driver emits registers sequentially by address; RKNN emits registers in a specific positional order with padding between them |
| **Register address ordering** | Live EMIT shows registers NOT in address-sorted order | The register emission follows a specific structural order: RDMA setup, CNA compute, CORE, DPU output, PPU postproc, PC tail. Within each group, registers are at fixed positions with padding. | Python driver emits in arbitrary address order, which may work but doesn't match RKNN's exact layout |

## Appendix F: Conv debug string index for r2 reverse engineering

The following strings from the RKNN runtime debug output (enabled via `RKNN_LOG_LEVEL=5`) serve as convenient r2 entry points for continued reverse engineering. Each string is followed by its binary address and the function it belongs to.

| String | Binary address | Associated function | What it reveals |
|--------|---------------|-------------------|-----------------|
| `enable argb mode, dtype: %s, channel: %d` | `0x0060cd98` | `fcn.002f6f78` | ARGB/1-channel special input mode |
| `dump npy tensor to: %s` | `0x0060f058` | conv output dumper | Debug tensor dumps |
| `dump WorkLoad tile meet unkown target: %s` | `0x0060ece8` | `fcn.0036b4d8` | Workload tile dump (3-core) |
| `WorkLoad(0/1/2)` | `0x0060ecc8` | `fcn.0036b4d8` | 3-core workload header |
| `WorkLoad(0/1)` | `0x0060ecd8` | `fcn.0036b4d8` | 2-core workload header |
| `op work load meet unkown idx: %d, idx must < 9` | `0x0060cfb0` | tile workload validator | Workload index must be < 9 |
| `Cannot find uniform split solution for size %d with limit %d and macC %d` | `0x006051b8` | `fcn.000f4a10` | Uniform split solver failure |
| `failed to update data and weight reuse!` | `0x0060c768` | `fcn.00307198`, `fcn.00312328` | Reuse update failure |
| `In`/`alid`/`reuse strategy!` | `0x0062cc90` | `fcn.005c6b50` | Bank/reuse config validation |
| `banks num invalid, input_bank_num: %d, weight_bank_num: %d` | `0x0062cc50` | `fcn.005c6b50` | Bank allocation mismatch |
| `min_weight_banks > 3 ... ChannelTile` | `0x0060f388` | `fcn.00387e3c`, `fcn.00388a18` | F2 ChannelTile trigger |
| `MC K Tile Failed, min kernel step %d, but get %d` | `0x0062ce50` | `fcn.005d4ac0` | K/weight tiling failed |
| `X tile buffer overflow!` | `0x0062d298` | `fcn.005f51c0` | X dimension overflow |
| `Generate Y config crash!` | `0x0062d508` | `fcn.005f51c0` | Y config generation failure |
| `illegal tiling method %d` | `0x0062d040` | `fcn.005f38f0` | Invalid split method |
| `Meet unsupported split` | `0x006125b0` | `fcn.00403a08`, `fcn.00404930`, `fcn.00406600`, `fcn.00408110`, `fcn.004098c8` | Split method not supported for data type |
| `Meet unsupported hybrid split type` | `0x006125c8` | `fcn.00403a08` | Hybrid split not supported |
| `Meet unsupported Tile` | `0x00613548` | `fcn.00476e88`, `fcn.00478a08`, `fcn.0047a5b0` | Tile op not supported |
| `buffer overflow!!!` | `0x0062cca8` | buffer validator | Output buffer overflow |
| `Unknown split Method -> %d` | `0x0062cf58` | `fcn.005f1998` | Fatal: invalid split method enum |
| `emitABC_T_BAC_regtask notch_addr overflow` | `0x0062cb58` | `fcn.005a41f0`, `fcn.005a5388`, `fcn.005a6068`, `fcn.005a7200` | ABC_T register emission overflow |
| `emitKC_T_C1K1C2K2C3 limit_w overflow` | `0x0062cb88` | `fcn.005a4e18`, `fcn.005a6c90`, `fcn.005a7e28` | KC_T register emission width overflow |
| `failed to tile argb mode layer!` | `0x0062ce30` | `fcn.005d4ac0` | ARGB mode tiling failure |
| `Conv min_weight_banks > 3, OutputName : %s` | `0x0062ce00` | `fcn.005d4ac0` | Layer-level conv weight bank check |
| `RKNNConfig: getBanksForFullWeights type_bytes is 0, use 4 bits` | `0x0060f328` | bank config | Weight bank calculation |
| `Unsupport weight data type!` | `0x0060cfe0` | `fcn.00312328` | Weight format unsupported |
| `exConvStreaming` | `0x0060b720` | `fcn.002ba2c0`, `fcn.002be528`, `fcn.002efda4`, `fcn.002f8d68`, `fcn.002ff3d0` | Weight streaming conv operator |

### What the Python draft (conv_new_clean.py) already handles correctly

| Shape category | Example | Strategy | Works? |
|----------------|---------|----------|--------|
| Simple 1x1 | 16→16, 8×8 | Default single-task | ✓ PASS |
| Large spatial (im2col) | 3×224×224→32 | spatial_im2col | ✓ PASS (uses CPU im2col fallback) |
| Simple spatial | 3→6, 9×9 | spatial_oc_serial | ✓ PASS |
| Depthwise spatial | 32→32, 112×112 with groups=32 | depthwise_spatial_tiled | ✓ PASS |
| Specific 1x1 (ic>>oc) | 96→24, 56×56 | pointwise_oc_tile | ✓ PASS |
| Specific 1x1 PC-chain | 256→32, 28×28 | pointwise_tile (PC-chain) | ✓ PASS |

### What the Python draft does NOT handle (and would need ConvStreaming)

For shapes exceeding the F2 `min_weight_banks > 3` threshold with spatial conv (e.g., 160→320, 3×3), RKNN switches to ChannelTile or ConvStreaming. The current draft would either trigger spatial_im2col (very slow) or try a direct submit that exceeds CBUF and crashes. These are the shapes that need the full register-emission-based chained approach.

### Practical recommendation

1. **For refactoring conv_new_clean.py**: The 6 strategies can be cleaned up into 3 shared execution paths: (a) Direct single-submit for simple shapes, (b) PC-chain with weight_reuse for ChannelTile-like OC splitting, (c) im2col fallback for spatial overflow. This is achievable NOW.

2. **For full RKNN tiling compatibility**: The register emission layer (ABC_T/BAC, KC_T/C1K1C2K2C3) needs to be reversed. These functions convert the planner's split vectors into the exact NPU register stream with padding, DCOMP filler, and PC-chain tails. This is the next major reverse engineering task.

3. **For ConvStreaming**: This is a separate operator that needs its own complete reverse engineering. It's unlikely to be needed for most conv shapes — only when spatial conv weights heavily exceed CBUF.

## Appendix G: Complete inventory of all RKNN conv tiling strategies

**Important correction**: The values 1-6 passed to `fcn.005f1998` are NOT tiling strategy enums. They are VALUE READER MODES that extract different fields from a tile seed structure. The disassembly at 0x005f1940-0x005f1984 shows:

```
Method 1: ldr w0, [x0, #12]    → reads value from tile_seed+12
Method 2: ldr w0, [x0, #16]    → reads value from tile_seed+16
Method 3: ldr w0, [x0, #12] + set flag → reads same +12 with marker
Method 4: same as method 2     → reads parameter from struct+16
Method 5: ldr w0, [x0, #8]     → reads value from tile_seed+8
Method 6: ldp pair from +8     → compares two values from tile_seed+8
```

These values at offsets +8, +12, +16 from the seed struct are different tiling parameters (Y_step, K_step, combined step count, etc.). The method number selects WHICH parameter to extract.

The ACTUAL tiling strategy in `fcn.005f38f0` uses only **3 grouping patterns**:
- **Method == 2**: `tile_index & 1` (dual-group: markers 1 and 3)
- **Method == 3**: `tile_index % 3` (triple-group: markers 1 and 7)
- **Default (method != 2 and != 3)**: `tile_index % method` (N-group: method value = group count)

### How conv_new_clean.py's 6 strategies DIFFER from RKNN

conv_new_clean.py has **6 hardcoded shape-fallback strategies** while RKNN has **1 unified planner**:

| conv_new_clean.py strategy | Trigger condition | Approach | RKNN equivalent |
|---------------------------|-------------------|----------|-----------------|
| `spatial_im2col` | Large weight spatial conv | **CPU im2col** + pointwise submit | NO EQUIVALENT — RKNN never does this. It programs spatial conv HW directly or uses ConvStreaming |
| `grouped_serial` | `groups>1 && spatial` | Per-group independent submit | Higher-level loop — `fcn.005d4ac0` handles one group at a time |
| `spatial_oc_serial` | `spatial && out_c > 16 && (in_c%16!=0 or in_c>=16)` | Row tiles + 16-chan OC tiles, independent submits | `fcn.005d4ac0` generates same tiles but CHAINS them via PC-chain |
| `depthwise_spatial_tiled` | `depthwise && spatial && (out_h > tile_h or out_c > align_c)` | Channel tiles (32) + row tiles, independent submits | Same tiles, but chained + reuse update applied |
| `_needs_pointwise_oc_tile` | Specific big→small 1x1 shapes | OC tiles (16-32 ch) + row tiles, independent submits | This IS ChannelTile — same split, but RKNN chains with PC-chain |
| `_needs_pointwise_tile` | Same shapes, PC-chain variant | OC tiles + PC-chain + weight_reuse | **CLOSEST match** — this is how RKNN does it for all multi-tile cases |

**Key insight**: conv_new_clean.py has 6 strategies because its simple `make_conv2d_regs` + independent submit approach can't handle all shapes. Each strategy is a workaround for a specific shape family. RKNN has ONE path that works for ALL shapes by:
1. Always generating a tile list (even for 1 tile)
2. Always using PC-chain between tiles (even for 1 tile tail)
3. Always applying reuse update
4. Using the SAME register layout and emission pattern for all tiles
5. Having weights pre-loaded (no per-tile weight reload)

### The 3 conv-specific strategies (beyond conv_new_clean's scope)

| Strategy | Trigger | Planner function | How it works |
|----------|---------|-----------------|--------------|
| **ChannelTile** | `min_weight_banks > 3` (F2) | `fcn.00387e3c`/`fcn.00388a18` | OC channel splitting with PC-chain chaining between tiles |
| **ConvStreaming** | `exConvStreaming` operator | `fcn.002ba2c0` (5.6KB) | Streams weights in chunks through CBUF; separate operator with own tiling |
| **Multicore** | `core_num > 1` | `fcn.005d78e0`/`fcn.005d7bc8` → `fcn.005d4ac0` | Creates 3× tile lists with PC-chain routing via `PC_REGISTER_AMOUNTS` core field |

### Tile type fields (from tile-table header at 0x0062cf78)

Each tile record has these dimension/type fields:

```
mc_treat_by_y_tile     — How Y dimension is distributed across cores
mc_treat_by_k_tile     — How K dimension is distributed across cores  
mc_treat_by_1c_y_tile  — Single-core fallback Y treatment
mc_treat_by_1c_k_tile  — Single-core fallback K treatment
```

The "mc_treat_by" values encode one of several distribution types (from `Illegal mc Y type` / `Illegal mc K type` diagnostics):
- Whole Y/K tile assigned to one core
- Interleaved Y/K across cores  
- Split Y/K across cores

### Register emission strategies (from fcn.005a41f0/005a4e18)

Once the planner determines the split method and tile list, the register emission layer converts tiles to actual NPU register commands:

| Emission function | Tiling pattern | What it does |
|-------------------|---------------|--------------|
| **ABC_T/BAC** (`fcn.005a41f0`, 1.9KB) | ABC → BAC permutation | Reorders dimensions: A (input) → B (first), C (output) → A, B (kernel) → C. Used for combining spatial and channel tiles |
| **KC_T/C1K1C2K2C3** (`fcn.005a4e18`, 1.3KB) | Multi-level K+C tiling | Creates 3-level tile hierarchy: first Channel level, first Kernel level, second Channel, second Kernel, third Channel. Used for complex multi-tile scenarios |

### Complete strategy decision tree

```
Input: conv shape (N, C, H, W, OC, KH, KW, groups, target)

1. IF target is RKNPU_F2 AND estimate_min_weight_banks > 3
   → ChannelTile (fcn.00387e3c) with split method 2 (dual-group K-split)
   
2. IF operator is exConvStreaming
   → ConvStreaming path (fcn.002ba2c0)

3. IF ARGB mode (1-channel input, dtype=FP16)
   → ARGB input mode with special CNA_CONV_CON1 flags

4. IF multicore (core_num > 1) AND split is feasible
   → `fcn.005d78e0`/`fcn.005d7bc8` builds 3× tile lists with mc_treat_by_* fields
   → PC-chain routes each tile list to its core via `PC_REGISTER_AMOUNTS` core field
   Fallback: IF split fails → single core mode (fcn.005d4ac0), 1c treatment

5. Build tile vectors via `fcn.005f38f0`:
   - Load shape vectors, target limits, mc_y_type, mc_k_type
   - Call `estimate_min_data_tile` for data pressure
   - Call `compute_legal_k_split_step` for K feasibility
   - Choose grouping pattern: method=2 (2 groups), method=3 (3 groups), or N groups
   
6. Apply reuse update (`fcn.00307198`/`fcn.00312328`):
   - Walk tile list, set `data_reuse`/`weight_reuse` per tile
   - Patch register command values at indexed positions via `fcn.002efd38`
   
7. Configure bank/reuse fields (`fcn.005c6b50`):
   - Validate `input_bank_num + weight_bank_num <= target banks`
   - Validate reuse combination is legal
   - Program CBUF bank fields and reuse fields
   
8. Emit register commands:
   - For each tile, emit CNA/CORE/DPU/PPU registers + DCOMP padding
   - Append 4-word PC tail (PC_BASE_ADDRESS, PC_REGISTER_AMOUNTS, VERSION, OPERATION_ENABLE)
   - Two emission patterns exist: ABC_T/BAC (simple reorder) and KC_T/C1K1C2K2C3 (multi-level)
```

This decision tree covers all known conv tiling paths in `librknnrt.so` as of May 2026.

## Appendix H: Draft pseudo-code for remaining emit/streaming paths

### 1. ABC_T/BAC register emission (fcn.005a41f0, 1.9KB)

This function converts planner tile vectors into register command streams with dimension permutation A→B, C→A, B→C.

**Context**: After the planner (fcn.005d4ac0 / fcn.005f38f0) generates a tile list with split vectors, this function emits the actual register commands for each tile while reordering dimensions.

**Call chain**: `fcn.005a41f0(x0=ctx, x1=tile_vectors, x2=output_vec, x3=attr, x4=seed, ...)`

**Stack**: 0xC00 bytes (3072) — enough for 4× tile vector copies of 0x18 entries each

```c
bool emit_abc_t_bac_regtask(
    ConvTileCtx *ctx,            // x0, x28
    TileVec *tile_vectors,       // x1, x21
    OutputVec *output_vec,       // x2, x22  
    TileAttr *attr,              // x3
    TileSeed *seed,              // x4, x19
    bool *result_flags)          // x7, x8
{
    // Step 1: Load dimension values from context
    int dim_a = ctx->field_0x158;         // [x28+0x158]
    int dim_b = ctx->field_0x150;         // [x28+0x150]
    int dim_c_step = ctx->field_0x154;    // [x28+0x154]
    int hw_limit = ctx->field_0x28;       // [x28+0x28]
    int target_tag = ctx->field_0x00;     // [x28+0x00]
    
    // Step 2: Select shape/limit from dimension set
    TileShape shape = select_tile_shape(ctx, attr, target_tag);
    int shape_count = shape.count;
    int shape_step = shape.step;
    int tile_groups = div_round_up(dim_a, dim_c_step);
    
    // Step 3: Compute dimension permutation
    // A → B (permute first to mid), C → A (third to first), B → C (mid to third)
    int a_start = dim_b;                   // A' = original B
    int b_start = dim_a;                   // B' = original A
    int c_start = dim_c_step;              // C' = original C step
    
    // Step 4: Allocate 4 output vectors (0x18 entries each)
    // Each vector holds register command positions for:
    //   vec[0]: dimension A (now B)
    //   vec[1]: dimension B (now C) 
    //   vec[2]: dimension C (now A)
    //   vec[3]: combined permutation key
    vector<long> perm_vectors[4];
    for (int i = 0; i < 4; i++)
        perm_vectors[i] = tile_vectors[i + offset];  // copy with offset
    
    // Step 5: Build permuted tile list
    int tile_index = 0;
    for (int a = 0; a < tile_groups; a++) {
        for (int b = 0; b < dim_b; b++) {
            for (int c = 0; c < dim_c_step; c++) {
                // Compute combined dimension index
                int combined = a * dim_b * dim_c_step + b * dim_c_step + c;
                
                // Split into permuted components
                int perm_a = combined % dim_c_step;        // now C
                int perm_b = combined / dim_c_step % dim_b; // now A
                int perm_c = combined / (dim_c_step * dim_b); // now B
                
                // Emit register command for this tile
                emit_tile_regs(ctx, perm_a, perm_b, perm_c, 
                              &perm_vectors[0], &perm_vectors[1],
                              &perm_vectors[2], &perm_vectors[3]);
                
                tile_index++;
            }
        }
    }
    
    // Step 6: Write register commands to command buffer
    for (int i = 0; i < tile_index; i++) {
        uint64_t *reg_cmd = command_buffer_ptr(ctx, i);
        reg_cmd[0] = encode_reg(TARGET_CNA, REG_CNA_CONV_CON1, ...);
        reg_cmd[1] = encode_reg(TARGET_CNA, REG_CNA_CONV_CON2, ...);
        // ... emit full register stream for this tile
    }
    
    return true;
}
```

The name "BAC" means the output register ordering of dimensions is: **B** (original A), **A** (original B), **C** (original C step). This permutes the CSC→CMAC data sequencing for spatial convolutions.

### 2. KC_T/C1K1C2K2C3 register emission (fcn.005a4e18, 1.3KB)

This function handles multi-level nested tiling with 5 levels: Channel tile 1, Kernel tile 1, Channel tile 2, Kernel tile 2, Channel tile 3.

**Call chain**: `fcn.005a4e18(x0=ctx, x1=output_list, x2=param_vec, x3=attr, x4=seed, stack=dims)`

**Stack**: 0x440 bytes (1088)

```c
bool emit_kc_t_c1k1c2k2c3(
    ConvTileCtx *ctx,             // x0, x22
    TileList *output_list,        // x1, x21
    ParamVec *param_vec,          // x2
    TileAttr *attr,               // x3
    TileSeed *seed,               // x4, x19
    int *stack_dims)              // stack+0x440
{
    // Step 1: Extract 5-level tile dimensions
    // C1 = first channel tile, K1 = first kernel tile
    // C2 = second channel tile, K2 = second kernel tile  
    // C3 = third channel tile (remainder)
    int c1 = ctx->field_0x00;               // [x22+0x00]
    int k1 = ctx->field_0x28;               // [x22+0x28]
    int c2_step = ctx->field_0x150;          // [x22+0x150]
    int k2_step = ctx->field_0x154;          // [x22+0x154]
    int c3_remain = ctx->field_0x17c;        // [x22+0x17c] (sign-extended)
    
    // Step 2: Select shape/limit
    TileShape shape = select_tile_shape(ctx, seed->field_0x40);
    int shape_count = shape.count;
    int shape_step = shape.step;
    
    // Step 3: Validate tile bounds
    // Check that c3_remain + k2_step <= hw_limit
    int hw_limit = ctx->field_0x28;
    int check = c3_remain + k2_step - 1;
    if (check > hw_limit || ...)
        return false;
    
    // Step 4: Nested 5-level tile emission loop
    int tile_count = 0;
    int rem_count = param_vec->total;
    int step_size = param_vec->step;         // w25
    int k_step = param_vec->kstep;           // w23
    
    for (int c1_pos = 0; c1_pos < c1; ) {    // Level 1: C-tile
        int c1_step = min(c1 - c1_pos, c2_step);
        
        for (int k1_pos = 0; k1_pos < k1; ) { // Level 2: K-tile
            int k1_step = min(k1 - k1_pos, k2_step);
            
            for (int c2_pos = 0; c2_pos < c1_step; ) { // Level 3: sub-C-tile
                int c2_step = min(c1_step - c2_pos, step_size);
                
                for (int k2_pos = 0; k2_pos < k1_step; ) { // Level 4: sub-K-tile
                    int k2_step = min(k1_step - k2_pos, step_size);
                    
                    // Level 5: C3 remain — emit one tile
                    int tile_base = c1_pos * k1 + k1_pos;
                    int tile_offset = calculate_offset(
                        tile_base, c2_pos, k2_pos, c3_remain);
                    
                    // Call register emission worker
                    RegCmd *cmd = emit_register_tile(ctx, 
                        tile_offset, c1_step, k1_step, 
                        shape_step);
                    
                    // Store tile record
                    TileRecord *t = &output_list->tiles[tile_count];
                    t->ystart = c1_pos + c2_pos;
                    t->kstart = k1_pos + k2_pos;
                    t->y_step = c1_step;
                    t->k_step = k1_step;
                    t->regcmd_offset = cmd->offset;
                    
                    tile_count++;
                    k2_pos += k2_step;
                }
                c2_pos += c2_step;
            }
            k1_pos += k1_step;
        }
        c1_pos += c1_step;
    }
    
    // Step 5: Write register command list to output
    for (int i = 0; i < tile_count; i++) {
        output_list->regcmd[i] = output_list->tiles[i].regcmd_offset;
    }
    output_list->count = tile_count;
    
    // Step 6: Update reference counts (shared_ptr semantics)
    output_list->add_ref();
    
    return true;
}
```

The "C1K1C2K2C3" name encodes the nesting order:
- **C1** = First-level channel tile (outermost)
- **K1** = First-level kernel tile
- **C2** = Second-level channel sub-tile  
- **K2** = Second-level kernel sub-tile
- **C3** = Third-level channel remainder (innermost, single emit)

This is used for complex multi-tile scenarios where both Y (spatial/channel) and K (output channel) dimensions must be tiled with multiple sub-levels.

### 3. ConvStreaming operator (fcn.002ba2c0, 5.6KB)

This is a separate conv operator for weight streaming. When weights exceed CBUF capacity even after ChannelTile OC-splitting, the runtime switches to streaming weights in chunks.

**Call chain**: `fcn.002ba2c0(x0=opaque, x1=stream_desc, x2=weight_buffer, ...)`

```c
bool conv_streaming_operator(
    void *opaque,                    // x0
    StreamDesc *stream_desc,         // x1, x19
    WeightBuffer *weight_buffer,     // x2, x21
    ...)
{
    // Step 1: Extract streaming parameters
    uint8_t *base_ptr = stream_desc->ptr;             // [x21+0x18]
    uint8_t *end_ptr = stream_desc->end_ptr;          // [x21+0x20]
    uint64_t total_size = end_ptr - base_ptr;
    
    // Check for overflow (size > 0x3fffffffffffffff)
    if (total_size > MAX_STREAM_SIZE)
        return error_overflow();
    
    // Check if already at end
    if (total_size == 0)
        return done();
    
    // Step 2: Read streaming weight parameters 
    // Parameters are stored sequentially at offsets +0x1d0 to +0x200
    // These describe:
    //   +0x1d0: weight_chunk_size    — size of each streaming chunk
    //   +0x1d4: weight_total_kernels — total output channels
    //   +0x1d8: input_channels       — IC for this stream
    //   +0x1dc: kernel_width/height  — KH, KW
    //   +0x1e0: output_width/height  — OH, OW
    //   +0x1e4: stride               — conv stride
    //   +0x1e8: chunk_offset         — current chunk offset in weights
    //   +0x1ec through +0x200: additional tiling/hw params
    
    ChunkParams params;
    params.chunk_size  = *(uint32_t*)(base_ptr + 0x1d0);
    params.total_kerns = *(uint32_t*)(base_ptr + 0x1d4);
    params.in_channels = *(uint32_t*)(base_ptr + 0x1d8);
    params.kh          = *(uint32_t*)(base_ptr + 0x1dc);
    params.kw          = *(uint32_t*)(base_ptr + 0x1e0);
    params.stride      = *(uint32_t*)(base_ptr + 0x1e4);
    params.chunk_offset = *(uint32_t*)(base_ptr + 0x1e8);
    // ... 6 more parameters from +0x1ec to +0x200
    
    // Step 3: Allocate chunk buffers
    uint32_t *chunk_buf = operator_new(4);  // allocate 4 bytes for chunk count
    *chunk_buf = params.chunk_size;
    
    uint32_t *kernel_buf = operator_new(4); 
    *kernel_buf = params.total_kerns;
    
    // Step 4: Process the streaming operation
    // fcn.003d2a18 handles the actual register programming for streaming
    int result = process_streaming_chunks(ctx, &params);
    
    // Step 5: Check target and handle result
    if (result == TARGET_F2) {
        // For F2 target, copy streaming parameters sequentially
        // from descriptor at offsets +0x1e8, +0x1ec, +0x1f0, ..., +0x200
        uint32_t *params_iter = result_ptr;
        for (int i = 0; i < 11; i++) {
            *params_iter++ = *(uint32_t*)(base_ptr + 0x1e8 + i*4);
        }
    }
    
    // ... additional streaming logic for remaining chunks
    
    return true;
}
```

### 4. Multicore dispatch (fcn.005d78e0/fcn.005d7bc8, ~15KB combined)

These functions wrap the single-core planner (fcn.005d4ac0) to create three tile lists, one per NPU core:

```c
bool build_multicore_conv_tiles(
    ConvTileCtx *ctx,
    ConvLayer *layer,
    MultiCoreConfig *mc_config)
{
    // Step 1: Build primary tile list for core 0
    vector<ConvWorkTile> core0_tiles = 
        build_conv_work_tiles(ctx, layer, ...);
    
    // Step 2: For each additional core, create a copy of the tile list
    // with adjusted output offsets and PC-chain routing
    for (int core = 1; core < mc_config->core_count; core++) {
        vector<ConvWorkTile> core_tiles;
        for (ConvWorkTile &t : core0_tiles) {
            ConvWorkTile copy = t;
            // Adjust output address for this core's output region
            copy.dst_base_addr += core * core_output_stride;
            // Set core-specific reuse flags
            copy.mc_treat_by_y_tile = mc_config->mc_y_type;
            copy.mc_treat_by_k_tile = mc_config->mc_k_type;
            core_tiles.push_back(copy);
        }
        per_core_tiles[core] = core_tiles;
    }
    
    // Step 3: Validate core assignment
    // Check mc_y_type and mc_k_type are legal values
    if (!valid_mc_type(mc_config->mc_y_type))
        log("Illegal mc Y type");
    if (!valid_mc_type(mc_config->mc_k_type))
        log("Illegal mc K type");
    
    // Step 4: Build PC-chain with core routing
    // Each tile's PC tail encodes the target core via REGISTER_AMOUNTS core field
    for (int core = 0; core < mc_config->core_count; core++) {
        for (int i = 0; i < per_core_tiles[core].size(); i++) {
            ConvWorkTile &t = per_core_tiles[core][i];
            // PC tail: core index encoded in REGISTER_AMOUNTS bits 12-15
            uint32_t pc_amount = encode_pc_amount(
                t.reg_body_count,
                core + 1);  // core index 1, 2, or 3
            t.pc_tail = make_pc_tail(next_addr, pc_amount);
        }
    }
    
    return true;
}
```

### 5. Updated runnable Python draft (unified conv tiler)

With all above pseudo-code, the unified Python approach replaces conv_new_clean.py's 6 strategies:

```python
def run_conv2d_unified(batch, in_c, out_c, kh, kw, hw, groups=1, stride=1):
    """Replace 6 strategies with 1 unified RKNN-style planner"""
    in_h, in_w = hw
    p = conv_params(batch, in_c, in_h, in_w, out_c, kh, kw, groups, stride)
    
    # Step 1: Compute tile Y-step from data bank pressure (fcn.005d4ac0 logic)
    weight_banks = estimate_weight_banks(in_c, out_c, kh, kw, groups)
    data_banks = RK_CBUF_BANKS - weight_banks
    y_step = compute_y_step(p, data_banks)
    
    # Step 2: Compute tile K-step from weight bank pressure (ChannelTile logic)
    k_step = compute_k_step(p, weight_banks)
    
    # Step 3: Generate tile records  
    tiles = []
    for y_start in range(0, p.out_h, y_step):
        tile_y = min(y_step, p.out_h - y_start)
        for k_start in range(0, out_c, k_step):
            tile_k = min(k_step, out_c - k_start)
            tiles.append(Tile(y_start, k_start, tile_y, tile_k))
    
    # Step 4: Build PC-chained register tasks
    task_regs = []
    for i, tile in enumerate(tiles):
        regs = make_conv2d_regs_per_tile(p, tile, 
            in_addr + tile.y_start * in_w * align_c * 2,
            wt_addr + tile.k_start * kh * kw * align_c * 2,
            out_addr + tile.y_start * out_w * 16 + tile.k_start // 16 * out_stride * 16,
            out_c=tile.k_step,
            weight_reuse=(i > 0 and tile.k_start == tiles[i-1].k_start),
            data_reuse=(i > 0 and tile.y_start == tiles[i-1].y_start))
        task_regs.append(regs)
    
    # Step 5: Submit as PC chain
    write_and_submit_pc_chain(task_regs, repeat=1 if any_tile_changes_weight else 2)
    
    # Step 6: Readback with nc1hwc2 unpack
    return read_and_unpack(tiles, p)
```

This single planner replaces all 6 strategies from conv_new_clean.py. The key changes from the current code:
1. All multi-tile cases use PC-chain (not just strategy 6)
2. Weight/data reuse flags are set per-tile based on adjacency
3. Output addresses are computed per-tile with nc1hwc2 offsets
4. A single readback path handles all output formats

## Appendix J: Updated pipeline model (from May 2026 analysis)

### The unified conv execution pipeline

```
Layer attrs (in_c, out_c, kh, kw, groups, depthwise, ...)
    │
    ▼
fcn.005d4ac0 ───── build_initial_tile_seed() ───── choose_split_method()
    │                                                    │
    │                                           SPLIT_NONE, SPLIT_BY_Y,
    │                                           SPLIT_BY_K, SPLIT_BY_Y_AND_K
    │                                                    │
    ▼                                                    ▼
Y/K tile loop ──────────────────────────────  fcn.005f38f0 (composer)
    │                                                   │
    │                                    per-tile index/marker vectors
    ▼                                                   │
Tile vector ──────────────────────────────────────────────┘
[{ystart, kstart, y_step, k_step, ...}]
    │
    ▼
fcn.005a41f0 (ABC_T/BAC)  OR  fcn.005a4e18 (KC_T)
    │                           │
    │    ┌──────────────────────┘
    ▼    ▼
fcn.00384880 (parameter extraction dispatch)
    │
    ├── mode=1  → reads ctx+0xcc/0xf4/0x11c (size=0x20, count=4)
    ├── mode=3  → reads ctx+0xb0/0xd8/0x100 (size=8, count=1)
    ├── mode=5  → reads ctx+0xb8/0xe0/0x108 (size=0x10, count=2)
    ├── mode=6  → reads ctx+0xc4/0xec/0x114 (size=0x20, count=4)
    ├── mode=0xa→ reads ctx+0xbc/0xe4/0x10c (size=0x10, count=2)
    ├── mode=0x10→reads ctx+0xc0/0xe8/0x110 (size=0x10, count=2)
    ├── mode=0x40→reads ctx+0xac/0xd4       (size=N/A, count=4)
    └── mode=0x41→reads ctx+0xd0/0xf8/0x120 (size=0x20, count=4)
    │
    ▼
fcn.0037f760 ─── fcn.00101508 (target-specific param initializer)
    │                           │
    │             Allocates RegisterTask for target (GRIF/HRIF/ENIW/FNIW)
    │             Initializes with mode-specific parameters
    ▼
Pattern-specific register writer (fcn.00597828 for ABC_T, fcn.00598468 for KC_T)
    │
    ▼
PC-chain: [RDMA setup] [CNA compute] [CORE/MAC] [DPU output] [PPU] [PC tail]
```

### Key insight: the modes ARE the split methods

The mode values passed to `fcn.00384880` (1, 3, 5, 6, 0xa, 0x10, 0x40, 0x41) correspond
to different tiling parameter groups from the seed struct. These parameter groups
encode how Y and K tiles are composed into register values:

| Mode | w4 (size/stride) | w3 (count) | Context offsets | Likely meaning |
|------|-----|------|-----------------|----------------|
| 1 | 0x20 (32) | 4 | 0xcc, 0xf4, 0x11c | Large channel tile (32ch, 4 groups) |
| 3 | 8 | 1 | 0xb0, 0xd8, 0x100 | Small channel tile (8ch, 1 group) |
| 5 | 0x10 (16) | 2 | 0xb8, 0xe0, 0x108 | Medium channel tile (16ch, 2 groups) |
| 6 | 0x20 (32) | 4 | 0xc4, 0xec, 0x114 | Large channel tile alt (32ch, 4 groups) |
| 0xa (10) | 0x10 (16) | 2 | 0xbc, 0xe4, 0x10c | Medium channel tile alt (16ch, 2 groups) |
| 0x10 (16) | (16) | 2 | 0xc0, 0xe8, 0x110 | Fixed-16 channel tile |
| 0x40 (64) | N/A | 4 | 0xac, 0xd4 | Large 64-element tile |
| 0x41 (65) | 0x20 (32) | 4 | 0xd0, 0xf8, 0x120 | Large channel tile alt2 (32ch, 4 groups) |

### How this maps to conv.py refactoring

The current `conv.py` has 5 special-case handlers because we hardcode:
- Which weight/input/output layout to use
- Which group count to pass to registers
- Whether to use full_data_bank

RKNN solves all of these through ONE mechanism:
1. `fcn.00384880` extracts tile parameters from the seed (Y_step, K_step, bank counts)
2. The emission function (ABC_T or KC_T) writes the correct register sequence based on the tile records and extracted parameters
3. `fcn.00307198`/`fcn.00312328` patches reuse flags in the emitted registers

For conv.py to reach true unification, we need to implement this parameter extraction
dispatch rather than the current if/elif chain.

### Remaining blocking gaps (UPDATED)

| # | Gap | Function | Status | Needs Runtime Trace? |
|---|------|----------|--------|---------------------|
| 1 | **choose_split_method** | `fcn.005d4ac0` | **PARTIALLY REVERSED**: assigns split_method=0 for pointwise (kernel==1), split_method=1 for spatial. Values 2/3 are NOT assigned here — they are used in the lower composer `fcn.005f38f0` with a validated range [0,6] via `fcn.005f1950`. The exact layer-attr-to-method mapping in `fcn.005f51c0`'s `derive_target_limits()` is still untraced through the 18KB function. | No |
| 2 | **fcn.00384880 parameter extraction** | `0x00384880` | **FULLY REVERSED** — complete 10-case switch table (modes 1,3,5,6,7,9,0xa,0x10,0x40,0x41) with context offset map documented in Appendix L | No |
| 3 | **ABC_T/KC_T register emission** | `fcn.00597828`/`fcn.00598468` | **STRUCTURE KNOWN** — vtable dispatch chain with 20 method comparisons (F2/F3/W1/W2 type-dispatchers, method idx 228-1181). Exact register names undecidable statically. | **Yes** |
| 4 | **Reuse register patching** | `fcn.002efd38`/`fcn.002efce0` | **MECHANICS KNOWN** — `bfi` patch of pre-built command entries via `ctx+0x288` vector slot. Register target indices need runtime capture. | **Yes** |
| 5 | **CNA group programming** | Unknown producer | **FORMAT KNOWN** (14-bit mask: feature/weight/CSC/ACCU/DPU/PPU groups + DMA errors). Programming site unidentified by any static xref. | **Yes** |
| — | **Multicore dispatch** | `fcn.005d78e0`/`7bc8` | **NOT REVERSED** — needs separate decomp pass | No, separate priority |

**Key insight**: Pieces 3-5 hit the **static analysis ceiling**. The vtable method addresses span 0x104000-0x106xxx (method indices 228-1181) — these are target-type dispatchers comparing function pointers across F2/F3/W1/W2 vtables. The indexed register-command vectors at `ctx+0x288` encode register targets as command-buffer indices that can only be decoded at runtime. The `experimental/trace_librknnrt_reuse.gdb` script is the prescribed next step for all three.

## Appendix K: fcn.00597828 ABC_T Register Task Builder

Function: `0x00597828`, stack frame: 288 bytes.

### Role

This function creates a register task for ABC_T emission. It does NOT write registers directly — it builds tile-count vectors and dispatches to target-specific vtable methods that write the actual register values.

### Control Flow

```
1. fcn.00384880 (parameter extraction)
   └─ extracts mode-specific parameters from context
   
2. If mode != 0: 
   └─ call fcn.00594ac8 (alt parameter extraction)
   
3. Compute tile metrics:
   └─ atomic_c_total / 4 → tile_count (w25)
   └─ w20 = k_start + tile_count - 1, sdiv by tile_count
   └─ w19 = c_start + tile_count - 1, sdiv by tile_count
   
4. Build 4-element vector {tile_count-1, k_start, c_start, y_step-1}
   └─ store to stack
   
5. Call fcn.005958c8 (vector builder)
   └─ assign result to context+0x60

6. Check format flag [ctx+0x40]:
   └─ if == 3: load iterator pair from [x0, #288]
       ├─ if iter_begin == iter_end: branch to failure path (0x597e48)
       └─ load current w1 from iter, check vtable at [x0, #3440]
           └─ if matches known addr → direct dispatch
           └─ else → blr x3 (vtable call)

7. Long vtable comparison chain (0x5979a0–0x597bf0):
   ├─ [x0, #3192] == 0x104720 (F2/CBUF?)
   ├─ [x0, #3464] == 0x104830
   ├─ [x0, #3472] == 0x104838
   ├─ [x0, #3488] == 0x104848
   ├─ [x0, #10792] == 0x1064d0
   ├─ [x0, #3816] == 0x104990
   ├─ [x0, #10800] == 0x1064d8
   ├─ [x0, #3824] == 0x104998
   ├─ [x0, #10816] == 0x1064e8
   ├─ [x0, #3672] == 0x104900
   ├─ [x0, #10792] == 0x1064d0 (v2)
   ├─ [x0, #5448] == 0x104ff0
   ├─ [x0, #10800] == 0x1064d8 (v2)
   ├─ [x0, #5456] == 0x104ff8
   ├─ [x0, #10816] == 0x1064e8 (v2)
   ├─ [x0, #5488] == 0x105018
   ├─ [x0, #5936] == 0x1051d8
   ├─ [x0, #5944] == 0x1051e0
   ├─ [x0, #3600] == 0x1048b8
   ├─ [x0, #3448] == 0x104820
   ├─ [x0, #3624] == 0x1048d0
   ├─ [x0, #5496] == 0x105020
   └─ [x0, #3432] == 0x104810

8. On match, call vtable method via blr with args:
   └─ x0 = object (RegisterTask*)
   └─ w1 = value (tile count or computed parameter)
   └─ (some calls also pass w3, w4, w5, w6 from stack)
   
9. epilogue: clean up vectors, return 0

### Key Finding: Format Flag Branch

At 0x597998: `cmp w1, #3` on `ldrsb w1, [x0, #64]`
- If format == 3: branches to 0x597bf0 (iterator-based dispatch using context+0x288 pair)
- Otherwise: proceeds to the direct vtable comparison chain

This ITERATOR path (format==3) is the BAC (alternate B-address order) path.
The COMPARISON chain path is the ABC_T path.

### Mapping to conv.py

| conv.py concept | RKNN ABC_T equivalent |
|----------------|----------------------|
| Register task = list of E(target, reg, value) | vtable methods write per-register values into pre-allocated command buffer |
| per-strategy if/elif chain | format flag at [ctx+0x40] selects ABC_T vs BAC vs KC_T |
| groups_override / full_data_bank | target-specific vtable methods encode these directly |
| tile loop: weight banks, data banks | parameter extraction (fcn.00384880) provides these from seed |

## Appendix L: fcn.00384880 Parameter Extraction Dispatch

Function: `0x00384880`, stack frame: 0x340 (832) bytes.

### Signature

```c
void extract_parameters(void *ctx, int8_t mode, uint32_t out[5])
// ctx = x0, mode = w1 (sign-extended byte), out = x8
```

### Switch Dispatch (verified from binary)

```c
switch (mode) {
case 1:  // 0x3849c8 region
    out[0] = 0x20;           // w4 = 32 (atomic step / size)
    out[1] = 4;              // w3 = 4 (tile count)
    out[2] = ctx[0xcc];      // w1 = (ctx+0xcc)  — data bank or stride-like
    out[3] = ctx[0xf4];      // w2 = (ctx+0xf4)
    out[4] = ctx[0x11c];     // w0 = (ctx+0x11c)
    break;

case 3:  // 0x3849e8 region (also reached by mode == 9 at 0x3849a0)
    out[0] = 8;              // w4 = 8
    out[1] = 1;              // w3 = 1
    out[2] = ctx[0xb0];      // w1
    out[3] = ctx[0xd8];      // w2
    out[4] = ctx[0x100];     // w0
    break;

case 5:  // 0x384930 region
    out[0] = 0x10;           // w4 = 16
    out[1] = 2;              // w3 = 2
    out[2] = ctx[0xb8];      // w1
    out[3] = ctx[0xe0];      // w2
    out[4] = ctx[0x108];     // w0
    break;

case 6:  // 0x3848f4 region
    out[0] = 0x20;           // w4 = 32
    out[1] = 4;              // w3 = 4
    out[2] = ctx[0xc4];      // w1
    out[3] = ctx[0xec];      // w2
    out[4] = ctx[0x114];     // w0
    break;

case 7:  // 0x384a48 region (mode == 7, after b.eq at 0x3848a4)
    out[0] = 0x40;           // w4 = 64
    out[1] = 8;              // w3 = 8
    out[2] = ctx[0xc8];      // w1
    out[3] = ctx[0xf0];      // w2
    out[4] = ctx[0x118];     // w0
    break;

case 9:  // 0x3849a0 region — falls through to mode 3 handler
    goto case_3;

case 0xa:  // 0x3849a8 region
    out[0] = 0x10;           // w4 = 16
    out[1] = 2;              // w3 = 2
    out[2] = ctx[0xbc];      // w1
    out[3] = ctx[0xe4];      // w2
    out[4] = ctx[0x10c];     // w0
    break;

case 0x10:  // 0x384948 region
    out[0] = 0x10;           // w4 = 16 (stores mode value 0x10 as size)
    out[1] = 2;              // w3 = 2
    out[2] = ctx[0xc0];      // w1
    out[3] = ctx[0xe8];      // w2
    out[4] = ctx[0x110];     // w0
    break;

case 0x40:  // 0x384978 region
    out[0] = 4;              // w1 = 4 (different layout — stored to out[0] directly)
    out[1] = ctx[0xd4];      // w0 = (ctx+0xd4) — stored to out[2], must shift
    out[2] = ctx[0xac];      // w2 = (ctx+0xac) — stored to out[3]
    // Only 3 fields stored (not 5)
    break;

case 0x41:  // 0x3848c8 region
    out[0] = 0x20;           // w4 = 32
    out[1] = 4;              // w3 = 4
    out[2] = ctx[0xd0];      // w1
    out[3] = ctx[0xf8];      // w2
    out[4] = ctx[0x120];     // w0
    break;

default:  // 0x384a00 — logs "Unknown mode" with the mode value and abort()
    log("extract_parameters: unknown mode %d", mode);
    abort();
}
```

### Context offset map

The context offsets 0xac–0x120 correspond to per-layer tiling parameter fields in the TileSeed struct (not the ConvTileCtx):

| Offset | Used by modes | Likely field |
|--------|---------------|-------------|
| 0xac   | 0x40          | Stride dimension A |
| 0xb0   | 3,9           | Small tile parameter A |
| 0xb4   | —             | (not used by any mode) |
| 0xb8   | 5             | Medium tile parameter A |
| 0xbc   | 0xa           | Medium tile alt parameter A |
| 0xc0   | 0x10          | Fixed-16 tile parameter A |
| 0xc4   | 6             | Large tile alt parameter A |
| 0xc8   | 7             | Extra large tile parameter A |
| 0xcc   | 1             | Large tile parameter A |
| 0xd0   | 0x41          | Large tile alt2 parameter A |
| 0xd4   | 0x40          | 64-element tile stride |
| 0xd8   | 3,9           | Small tile parameter B |
| 0xdc   | —             | (not used) |
| 0xe0   | 5             | Medium tile parameter B |
| 0xe4   | 0xa           | Medium tile alt parameter B |
| 0xe8   | 0x10          | Fixed-16 tile parameter B |
| 0xec   | 6             | Large tile alt parameter B |
| 0xf0   | 7             | Extra large tile parameter B |
| 0xf4   | 1             | Large tile parameter B |
| 0xf8   | 0x41          | Large tile alt2 parameter B |
| 0x100  | 3,9           | Small tile parameter C (count?) |
| 0x104  | —             | (not used) |
| 0x108  | 5             | Medium tile parameter C |
| 0x10c  | 0xa           | Medium tile alt parameter C |
| 0x110  | 0x10          | Fixed-16 tile parameter C |
| 0x114  | 6             | Large tile alt parameter C |
| 0x118  | 7             | Extra large tile parameter C |
| 0x11c  | 1             | Large tile parameter C |
| 0x120  | 0x41          | Large tile alt2 parameter C |

### What We Still Don't Know — updated with librknnc.so findings

**Status: librknnc.so (28.5MB) copied May 2026.** The compiler library is now
the primary reverse target for RKNN build behavior. The local binary
`experimental/rknn/librknnc.so` and the remote RKNN toolkit compiler at
`/home/orangepi/.local/lib/python3.10/site-packages/rknn/api/lib/linux-aarch64/librknnc.so`
have the same SHA-256:

```text
d499753a91065f0b52b2cdfa43c073645bcc37467c8e077372b302ccafd5d53c
```

This means local static offsets are valid for remote build-time GDB experiments.
The remote `~/.bashrc` contains `export RKNN_LOG_LEVEL=3`. In non-interactive
SSH batches the shell can return before that line, so either verify the variable
before the run or source the export line directly:

```sh
set -a; source <(grep "RKNN_LOG_LEVEL" ~/.bashrc); set +a
```

The compiler library contains the **active** tiling functions that are dead code
or only partially reachable in `librknnrt.so`. Key resolved items:

| String / evidence in `librknnc.so` | Address | Meaning |
|------------------------------------|---------|---------|
| `N4rknn15RKNNTileChannelE` | `0x017c61f8` | Named compiler pass for channel tiling. |
| `N4rknn14RKNNTilingPassE` | string table | Top-level compiler tiling pass is present in `librknnc.so`. |
| `|xstart|ystart|kstart| data reuse | weight reuse | ... |` | `0x017c6320` | Compiler has a split-vector/tile dump table with reuse fields. |
| `min_weight_banks > 3 && m_Target == RKNNTarget::RKNPU_F2, do conv with ChannelTile` | `0x017e2cb8` | F2 ChannelTile threshold is in the compiler binary. |
| `Conv min_weight_banks > 3, OutputName : %s` | `0x017e5498` | Compiler-facing diagnostic for the same condition. |
| `MC K Tile Failed, min kernel step %d, but get %d` | `0x017e5518` | K-tile lower bound failure path. |
| `  CNA feature group0: %d` | `0x017dc5a8` | CNA group formatter/debug output exists in compiler. |
| `failed to update feature data addr!` | `0x017e1670` / `0x017e16f8` | Register-task update path for feature addresses. |
| `failed to update data and weight reuse!` | `0x017e1938` | Register-task update path for reuse state. |

| Unknown | Function in librknnc.so | Status |
|---------|------------------------|--------|
| Vtable → register name mapping | `0x15d5ef0` (ChannelTile predicate), `0x15e1b60` (different binding) | Functions exist but vtable names are compiler-private C++ strings. To resolve: set runtime BP at `librknnc.so+0x1cbc10` (param_extractor) and capture mode per conv type |
| KC_T register layout | Same ABC_T-like builder but targets KC_T registers | Needs side-by-side trace of both builders with real conv parameters |
| Reuse patch register targets | `0x1531590`, `0x15477dc`, `0x1547820` (reuse updaters) | These are the compiler-side equivalents — need GDB trace at build time to see emitted commands |
| CNA group programming site | `base+0x7d13dc`, `0x7d1438` (bank validator) | The group mask is set during the planner stage. Need build-time trace to find the producer |

**Remaining unknowns requiring GDB tracing at build time** (on remote NPU machine where RKNN models can actually be built):

| Unknown | Target offset(s) in librknnc.so | How to resolve |
|---------|--------------------------------|---------------|
| Which mode in param_extractor (0x1cbc10) maps to which conv type | `0x1cbc10` (10-mode switch) | Set BP at 0x1cbc10, log mode value + conv type name from caller context |
| ABC_T / KC_T register layout per mode | `0x15d0cc0` (ChannelTile predicate), `0x15d5ef0`, `0x15e1b60` | After mode dispatch, step through predicate builder for each tile dimension |
| Split planner + top tiler output | `0x24c1d0` (11456B split planner), `0x2539c0` (10700B top tiler) | Set BP at entry, dump all output vector fields |
| CNA group mask producer | `0x7d13dc`, `0x7d1438` | Set BP at bank validator during build, dump caller chain and mask computation |
| Reuse updater behavior per mode | `0x1531590`, `0x15477dc`, `0x1547820` | Set BP at reuse updater, log which command indices get patched |

Build-log experiment on the remote official RKNN host:

```sh
ssh orangepi@192.168.192.36 \
  'bash -lc "cd ~/npu/ops_rknn &&
   mkdir -p /tmp/rknnc_trace_models &&
   python3 gen_conv2d_models.py --custom --batch 1 --in-ch 32 --height 14 --width 14 \
     --out-ch 128 --k-h 3 --k-w 3 --groups 1 \
     --name conv2d_trace_b1_c32_h14_w14_oc128_wic32_k3x3_g1 \
     --out-dir /tmp/rknnc_trace_models --force \
     2>&1 | tee /tmp/rknnc_build_channel_tile_small.log"'
```

Result: the model build completed under `RKNN_LOG_LEVEL=3`, but the public log
did not print `ChannelTile`, `min_weight`, split-vector, CNA group, or reuse
diagnostics. Plain build logging is therefore not enough; the next compiler-side
step must be a GDB trace or a forced diagnostic path, not another normal RKNN
build.

`experimental/trace_librknnc_build.gdb` now exists and has been run on the
remote official RKNN host through a login shell. The script discovers
`librknnc.so` from process mappings after the compiler is loaded, then installs
breakpoints at:

| Trace point | Offset | First observed status |
|-------------|--------|-----------------------|
| `param_extractor_mode_switch` | `0x001cbc10` | Hit repeatedly. `x1` is the mode selector; observed modes include `0xa` and `0x3`. |
| `split_planner` | `0x0024c1d0` | Hit once per build, under `rknn::RKNNCompiler::build()`. |
| `top_tiler` | `0x002539c0` | Hit three times per tested conv build. |
| `cna_bank_validator_a` | `0x007d13dc` | Breakpoint installed, not hit for the tested FP16 conv builds. |
| `cna_bank_validator_b` | `0x007d1438` | Breakpoint installed, not hit for the tested FP16 conv builds. |
| `reuse_update_a` | `0x01531590` | Breakpoint installed, not hit for the tested FP16 conv builds. |
| `reuse_update_b` | `0x015477dc` | Breakpoint installed, not hit for the tested FP16 conv builds. |
| `reuse_update_c` | `0x01547820` | Breakpoint installed, not hit for the tested FP16 conv builds. |
| `abc_or_channel_tile_emit_a` | `0x015d0cc0` | Breakpoint installed, not hit for the tested FP16 conv builds. |
| `abc_or_channel_tile_emit_b` | `0x015d5ef0` | Breakpoint installed, not hit for the tested FP16 conv builds. |
| `kc_or_alt_emit` | `0x015e1b60` | Breakpoint installed, not hit for the tested FP16 conv builds. |

Remote trace commands used:

```sh
scp experimental/trace_librknnc_build.gdb \
  orangepi@192.168.192.36:/tmp/trace_librknnc_build.gdb

ssh orangepi@192.168.192.36 \
  'bash -lc "cd ~/npu/ops_rknn &&
   gdb -q -x /tmp/trace_librknnc_build.gdb --args \
     python3 gen_conv2d_models.py --custom --batch 1 \
       --in-ch 32 --height 14 --width 14 --out-ch 128 \
       --k-h 3 --k-w 3 --groups 1 \
       --name conv2d_trace_b1_c32_h14_w14_oc128_wic32_k3x3_g1 \
       --out-dir /tmp/rknnc_trace_models --force \
     2>&1 | tee /tmp/rknnc_build_gdb_trace_small.log"'

ssh orangepi@192.168.192.36 \
  'bash -lc "cd ~/npu/ops_rknn &&
   gdb -q -x /tmp/trace_librknnc_build.gdb --args \
     python3 gen_conv2d_models.py --custom --batch 1 \
       --in-ch 160 --height 14 --width 14 --out-ch 320 \
       --k-h 3 --k-w 3 --groups 1 \
       --name conv2d_trace_b1_c160_h14_w14_oc320_wic160_k3x3_g1 \
       --out-dir /tmp/rknnc_trace_models --force \
     2>&1 | tee /tmp/rknnc_build_gdb_trace_stress.log"'
```

Both builds completed normally. The first hit of `split_planner` had this
compiler stack:

```text
#0  librknnc.so+0x0024c1d0
#1  librknnc.so+0x0024cea4
#2  librknnc.so+0x00191ce4
#3  librknnc.so+0x00191eb0
#4  librknnc.so+0x001924c8
#5  rknn::RKNNCompiler::build()
#6  RKNNCompiler_build
#7  libffi.so
```

The tested stress model reached the same planner path but still did not hit the
candidate register-task update or ABC/KC emit offsets. That means the offsets
are either not on this build path, are post-build/runtime-side helpers despite
being present in `librknnc.so`, or require a different model form/target option
to exercise. The next reverse step is therefore to trace *callers from*
`top_tiler`/`split_planner` and the string xrefs around the split-vector table,
instead of assuming the `0x1531590`/`0x15477dc`/`0x15d*` candidates are reached
by normal FP16 conv builds.

Return-state trace update:

`experimental/trace_librknnc_build.gdb` now also creates one-shot return
breakpoints for `split_planner` and `top_tiler`. The return-state stress log is:

```text
/tmp/rknnc_build_gdb_trace_stress_return.log
```

Stable observations from that trace:

| Function | Entry role evidence | Return evidence |
|----------|---------------------|-----------------|
| `split_planner` (`0x24c1d0`) | `x0` and `x1` are object/context-like pointers; `x3` points to a mostly zero output/work struct; `x5` points to a stack wrapper that contains `x3` at `+0x08`; `x6=0x3b`, `x7=0x200` are scalar limits/flags. | Returns `x0=0`. The shallow words of `x0`, `x1`, `x3`, and `x5` are unchanged in the captured window, so the visible output is not in the first 10 qwords of those objects. |
| `top_tiler` (`0x2539c0`) | `x0` is a layer/planner object containing a pointer to the common config object at `+0x28`; `x1` is the common config object whose first word is ASCII-like `0x5546495245`; `x2` is a tensor/operator-like object; `x3`, `x4`, `x5`, and `x8` are stack/output vector-like pointers; `x6` points to packed conv dimensions. | Returns `x0=x8`. `x3`, `x4`, `x5`, and `x8` are mutated into vector-like triples (`begin`, `end`, `capacity`) that point to heap arrays. This is the strongest current evidence that `top_tiler` materializes the split/tile vector lists. |

For the stress shape `b1_c160_h14_w14_oc320_wic160_k3x3_g1`, the packed
dimension words at `top_tiler` entry include:

```text
x6+0x18 = 0x0000000e000000a0  # likely H=14, C/in-ch=160
x6+0x20 = 0x000001400000000e  # likely out-ch=320, H=14
x6+0x28 = 0x00000003000000a0  # likely kernel=3, C=160
x6+0x38 = 0x0000000c00000140  # likely out_h=12, out_c=320
```

This is not yet enough to emit registers, but it gives a concrete next trace
target: dump the heap arrays referenced by the returned `x3`/`x4`/`x5`/`x8`
vector triples after each `top_tiler` return, then interpret them as tile
records (`xstart`, `ystart`, `kstart`, data/weight reuse) using the split-vector
table string.

Nested vector trace update:

`experimental/trace_librknnc_build.gdb` now follows one level of pointers inside
the returned `top_tiler` vector triples. The dense-stress log is:

```text
/tmp/rknnc_build_gdb_trace_nested_vectors_stress.log
```

Stable nested-vector observations:

| Returned holder | Shape in trace | Candidate meaning |
|-----------------|----------------|-------------------|
| `x3` vector | 8 bytes, `d32 00000000 0000000e` on all three `top_tiler` calls | One Y-range or Y-start/Y-limit pair: `0..14` for the input height of the stress shape. |
| `x4` vector | 40 bytes / 5 qwords; contains repeated pointers and a tagged qword with low dword `1`; nested payloads include `0x0000000e00000000` and pointer triples. | Wrapper around one axis' tile records, likely Y/data side. Not decoded enough to name fields. |
| `x5` vector | 40 bytes / 5 qwords; similar wrapper but with different nested pointers; nested payloads include output pointer/object references. | Wrapper around a sibling axis or reuse-side tile list. Not decoded enough to name fields. |
| `x8` vector | 24 bytes / 3 qwords: a single `begin/end/capacity` triple; nested payloads contain compact conv/tile shape fields. | Strongest candidate for the top-level tile descriptor vector. |

Most useful `x8` nested payloads seen for
`b1_c160_h14_w14_oc320_wic160_k3x3_g1`:

```text
# top_tiler call 1, x8 ptr[0]
d32 ... 00000000 0000000c ... 00000006 ...

# top_tiler call 2, x8 ptr[0]
d32 ... 00000000 0000000c ...
    ... 00000140 000000a0 00000003 00000003 00000007 ...

# top_tiler call 3, x8 ptr[0]
d32 ... 00000000 0000000c ...
    ... 00000140 000000a0 00000003 00000003 00000007 ...
```

Interpretation, still provisional:

- `0x0c` matches output height `12` for valid `14x14, k=3`.
- `0x140` matches output channels `320`.
- `0xa0` matches input channels `160`.
- the `0x03, 0x03` pair matches kernel `3x3`.
- `0x07` is a candidate derived tile/count parameter. It is not yet mapped to
  an RKNN split field.

This is the first trace where the compiler-side tiler produced recognizable
conv-dimension payloads inside returned heap records. A smaller follow-up trace
uses `experimental/trace_librknnc_top_tiler_vectors.gdb` to compare shapes with
only one major axis changed. Logs:

```text
/tmp/rknnc_top_tiler_vectors_h7.log
/tmp/rknnc_top_tiler_vectors_small.log
```

A/B result:

| Shape | `dims_x6` key lanes after `top_tiler` | `x3` vector | `x8 ptr[0]` movement |
|-------|---------------------------------------|-------------|----------------------|
| `b1_c160_h7_w7_oc320_wic160_k3x3_g1` | `... 000000a0 00000007 00000007 00000140 000000a0 00000003 00000003 00000001 00000140 00000005 00000005 ...` | 8-byte vector; payload still needs explicit dword capture in the compact script output. | Call 1 starts with `00000000 00000005`, matching output height `5`; later payloads still include the derived `0x07` candidate. |
| `b1_c32_h14_w14_oc128_wic32_k3x3_g1` | `... 00000020 0000000e 0000000e 00000080 00000020 00000003 00000003 00000001 00000080 0000000c 0000000c ...` | 8-byte vector; matches the same `[0, input_h]` holder shape seen in the full stress trace. | Calls 1/2 start with `00000000 0000000c`; call 2 includes `00000003 00000003`; call 3 shows `00000000 00000004 00000008 0000000c`, likely an output-height split boundary list. |

Confirmed so far:

- `x6` is a packed conv-dimension object. The key dword order includes
  `in_c`, `in_h`, `in_w`, `out_c`, `in_c`, `k_h`, `k_w`, group-like `1`,
  `out_c`, `out_h`, `out_w`.
- The compact `x8` nested payload moves with output height and kernel size. The
  dense-stress trace also showed out-channel and input-channel values inside
  later `x8` payloads.
- The `x3` holder is still best treated as a range/split-list vector tied to the
  input-height axis; the compact A/B script needs one more payload-dword print
  for `x3` before the exact field name is locked.

The shape-only A/B established that the moving fields are dimensions, but not
reuse controls. The follow-up record-layout trace below resolves the immediate
payload-dump gap for `x3`/`x4`/`x5` and follows the stable `x8` nested records
far enough to separate real shape tuples from surrounding heap artifacts. Even
after that, field names like `ystart`, `kstart`, `data reuse`, or `weight reuse`
remain unassigned until a consumer trace ties these records to the split-vector
table.

Record-layout trace update:

`experimental/trace_librknnc_record_layout.gdb` is a focused `top_tiler` return
trace that prints `x3`/`x4`/`x5`/`x8` as vector triples, dumps payloads as dwords
and qwords, follows plausible nested vector pointers, and runs GDB in batch mode
so the remote build does not stop at an interactive `(gdb)` prompt after normal
process exit. It was run on the official RKNN host with the login-shell
environment (`RKNN_LOG_LEVEL=3` from `~/.bashrc`). Logs:

```text
/tmp/rknnc_record_layout_stress.log
/tmp/rknnc_record_layout_h7.log
/tmp/rknnc_record_layout_small.log
```

Additional confirmed structure:

| Holder | Confirmed evidence | Current meaning |
|--------|--------------------|-----------------|
| `x3` | `x3.data d32 00000000 0000000e` for `14x14` inputs; `00000000 00000007` for `7x7`. | Direct `[0, input_h]` range vector. |
| `x4` | 40-byte vector-like wrapper. On call 3, nested payload contains dimension tuples. Stress: `00000001 000000a0 0000000e 0000000e`, `00000001 00000140 0000000c 0000000c`, `00000140 000000a0 00000003 00000003`. Small: `00000001 00000020 0000000e 0000000e`, `00000001 00000080 0000000c 0000000c`, `00000080 00000020 00000003 00000003`. | Shape-record wrapper for input tensor, output tensor, and weight tensor descriptors. |
| `x5` | 40-byte sibling wrapper. On call 3, nested payload repeats the same canonical dimension tuples as `x4`, but with a different leading chain of pointers/records. The wrapper count-like lane changes from `1` on earlier calls to `3` on call 3 for the small shape. | Sibling shape/tile-record wrapper. It is probably later-stage or per-core/list material, but exact role is not named yet. |
| `x8` | 24-byte vector holding a single begin/end/capacity triple. `ptr[0]` exposes output-height fields and, on call 3, also compact shape tuples: stress has `0x000000a000000140`, `0x0000000300000003`, derived `0x7`, plus `0x0000000e0000000e` and `0xb`; h7 has `0x000000a000000140`, `0x0000000300000003`, derived `0x7`, and output height `5`; small call 3 begins with split boundaries `0,4,8,12`. | Top-level returned split/tile descriptor vector. |

Useful interpretation of the canonical tuples:

```text
(1, in_c,  in_h,  in_w)   input tensor shape
(1, out_c, out_h, out_w)  output tensor shape
(out_c, in_c, k_h, k_w)   weight tensor shape for groups=1
```

The repeated qword `0x21` appears between many short heap chunks and should be
treated as allocator/chunk metadata or a non-semantic local record tag until a
trace proves otherwise. The moving shape tuples are real; the pointer-looking
qwords and high-entropy values around them should not be used as compiler field
names.

Consumer trace update:

Static disassembly and the observed `lr` from all record-layout traces identify
one normal caller for this path:

```text
top_tiler entry       librknnc.so+0x2539c0
call instruction      librknnc.so+0x25713c
return/use site       librknnc.so+0x257140
consumer loop start   librknnc.so+0x257240
```

The call setup at `0x2570fc..0x257138` passes stack-backed vector holders:

```text
x4 = sp + 0x510
x5 = sp + 0x530
x6 = sp + 0x3fc   # packed conv dims
x8 = sp + 0x648
```

Immediately after return, the caller computes the number of `x8` records with:

```asm
ldr x0, [sp, #1616]        ; x8 end
ldr x1, [sp, #1608]        ; x8 begin
sub x0, x0, x1
asr x0, x0, #3
mul x0, x0, 0xaaaaaaaaaaab ; divide by 3
```

This proves that `[sp+0x648]` is consumed as a vector of 24-byte records, not
just as an arbitrary output pointer. The loop at `0x257240` then walks an
array-like object in 24-byte steps (`x22 += 0x18`) and compares nested vector
lengths against the split-boundary vector length.

`experimental/trace_librknnc_consumer.gdb` captures this caller/use site and was
run on:

```text
/tmp/rknnc_consumer_stress.log
/tmp/rknnc_consumer_h7.log
/tmp/rknnc_consumer_small.log
```

Confirmed consumer-side observations:

| Shape/call | Consumed `[sp+0x648]` record | Payload reached through record pointer |
|------------|------------------------------|----------------------------------------|
| stress call 3 | one 24-byte record `{ptr, ptr+8, ptr+8}` | Starts `00000000 0000000c ... 00000000 0000000e ... 00000008 ... 00000140 ... 00000004 ...`. This is consumed by the post-call loop, so these fields are not just return-time heap noise. |
| h7 call 3 | one 24-byte record `{ptr, ptr+8, ptr+8}` | Starts `00000000 00000005 ... 00000000 00000007 ... 00000008 ...`; output height and input height move to `5` and `7`. |
| small call 3 | one 24-byte record `{ptr, ptr+0x10, ptr+0x10}` | Starts `00000000 00000004 00000008 0000000c ...`, confirming an output-height split-boundary vector for output height `12`. |

This is the first compiler-side evidence that the returned `top_tiler` `x8`
holder is directly consumed as a 24-byte-record split descriptor vector. It also
confirms that the `0,4,8,12` sequence is a consumer-visible boundary list, not a
display artifact.

Still not assigned:

- which consumed record dwords are `xstart`, `ystart`, and `kstart`;
- which consumed record dwords are `data reuse` and `weight reuse`;
- whether the `0x08` and `0x04` fields in the call-3 payload are tile counts,
  vector lengths, or axis-specific step sizes.

Boundary-writer trace update:

`experimental/trace_librknnc_boundary_writes.gdb` instruments the key write
points inside `librknnc.so+0x2572c0..0x25733c`:

```text
0x2572c0  select nested 24-byte record by x19
0x2572fc  write first boundary dword = 0
0x257330  write next cumulative boundary dword(s)
0x25733c  leave selected nested record
```

Logs:

```text
/tmp/rknnc_boundary_small.log
/tmp/rknnc_boundary_h7.log
/tmp/rknnc_boundary_stress.log
```

Confirmed behavior for this loop:

- `x21` is the upper boundary value for this selected axis.
- `w9` is the step/clamp value used by `min(remaining, step)`.
- `x25` is the count of nonzero boundary writes; `x23 = x25 + 1` is the
  required vector length.
- `x26 = (x25 + 1) * 4`; `x24 = x25 * 4 + 8`, so the selected destination is a
  `vector<int>` with two dwords for the simple observed case.
- The loop writes `0` at the selected vector begin, then writes cumulative
  boundaries with `str w1, [x3], #4`.

For the three tested call paths, this specific loop instance writes the
output-channel boundary vector:

| Shape | `x21` at boundary writer | Final selected vector |
|-------|--------------------------|-----------------------|
| `b1_c32_h14_w14_oc128_wic32_k3x3_g1` | `128` | `00000000 00000080` |
| `b1_c160_h7_w7_oc320_wic160_k3x3_g1` | `320` | `00000000 00000140` |
| `b1_c160_h14_w14_oc320_wic160_k3x3_g1` | `320` | `00000000 00000140` |

This maps one concrete compiler-generated boundary vector as K/output-channel
range `[0, out_c]`. It also shows that the already-observed small-shape
height-boundary list `0,4,8,12` is produced or preserved by a different nearby
path, not by this `0x2572c0..0x25733c` loop instance.

Height-boundary consumer trace update:

`experimental/trace_librknnc_height_boundary.gdb` follows the adjacent path
around `0x257350..0x25788c`, including the allocations for `[sp+0x5d0]`,
`[sp+0x5f0]`, the copy/build call into `[sp+0x630]`, and the later tile-read
loop. Logs:

```text
/tmp/rknnc_height_boundary_small.log
/tmp/rknnc_height_boundary_h7.log
/tmp/rknnc_height_boundary_stress.log
```

Confirmed behavior:

- `[sp+0x648]`, the same `x8` holder passed to `top_tiler`, remains the direct
  source vector for output-height/Y boundaries in the later tile-read loop.
- At `0x2576a0` and `0x2577f0`, `x22` points at this 24-byte-record vector.
  The selected record's first nested `vector<int>` is read as the boundary list.
- At `0x25788c`, the loop has loaded the current tile interval from that
  boundary list.

A/B evidence:

| Shape | `[sp+0x648]` / selected record boundary vector | Later interval reads at `0x25788c` |
|-------|-----------------------------------------------|------------------------------------|
| `b1_c32_h14_w14_oc128_wic32_k3x3_g1` | `00000000 00000004 00000008 0000000c` | `(0,4)`, `(4,8)`, `(8,12)` |
| `b1_c160_h7_w7_oc320_wic160_k3x3_g1` | `00000000 00000005` | `(0,5)` |
| `b1_c160_h14_w14_oc320_wic160_k3x3_g1` | `00000000 0000000c` | `(0,12)` |

This maps the `x8`/`[sp+0x648]` first nested vector as the output-height
boundary list (`ystart`/`ynext` boundaries). For the small proxy, the official
compiler splits output height `12` into three Y tiles of height `4`; for h7 and
the dense stress shape, this path keeps a single Y tile.

The sibling stack vectors allocated after the Y/K boundary loops now have a
clearer role:

- `[sp+0x5d0]`, `[sp+0x5f0]`, and `[sp+0x630]` are derived vector-of-vector
  wrappers built from the boundary lists.
- They are not the original boundary producers; they package/copy boundary
  vectors for later tile-row construction.

Tile-row construction trace update:

`experimental/trace_librknnc_tile_rows.gdb` traces the row-building path after
`0x25788c`. It was run remotely with `RKNN_LOG_LEVEL=3` exported from the
remote `~/.bashrc` line. Logs:

```text
/tmp/rknnc_tile_rows_small.log
/tmp/rknnc_tile_rows_h7.log
/tmp/rknnc_tile_rows_stress.log
```

Confirmed behavior:

- The `qbuild_interval` hit at `0x25788c` is the entry into the later
  tile-row builder for the currently selected Y interval.
- For the small proxy, this path sees the three output-Y tiles from the
  `[0,4,8,12]` boundary list:

| Shape | Relevant `qbuild_interval` values |
|-------|-----------------------------------|
| `b1_c32_h14_w14_oc128_wic32_k3x3_g1` | `(x1=6, x2=0, x3=0)`, `(x1=10, x2=4, x3=1)`, `(x1=14, x2=8, x3=2)` |
| `b1_c160_h7_w7_oc320_wic160_k3x3_g1` | single-Y tile, no `x2=4/8` Y interval sequence |
| `b1_c160_h14_w14_oc320_wic160_k3x3_g1` | single-Y tile, no `x2=4/8` Y interval sequence |

The `x1` value is the halo-adjusted input/source end for the Y tile, not the raw
output boundary. For the small proxy the first two Y tiles have source windows
ending at `6` and `10`; the last tile clamps to input height `14`.

- `row_fields_before_alloc` sees a stable row-index/channel-offset pattern in
  `w23` and `w21` after the initial setup rows. For the `160->320` shapes the
  repeated subrows include `w23=0,1,2` with `w21=0,112,224` in the final row
  family. The small proxy shows the same `w23=0,1,2` row index sequence for the
  three Y intervals, with `x5` carrying `0,4,8`.
- `row_header_sp650` and `row_obj_sp6c0` are staging objects for the final row
  payload. They preserve the row-index fields and build several nested vectors,
  but this trace is not sufficient to assign names to the reuse-bit fields.

### Tile record format at 0x257f00

The trace `experimental/trace_librknnc_tile_records.gdb` captured the exact tile record write.

At `0x257f00`:
```asm
stp w0, w23, [x7, #-8]   ; records[0] = w0, records[1] = w23
stp w28, w24, [x7]        ; records[2] = w28, records[3] = w24
```

Each tile record is **16 bytes** (4 dwords) staged at `x7-8..x7+7`. This is
still stack staging, not the final vector slot. Local disassembly shows the
later publish path copies a larger row object:

```asm
0x258444: add x21, x21, #0x48
...
0x2584b8: str w0, [x19]        ; current destination row slot
...
0x2589d8: str w1, [x19, #376]
0x2589e8: stp w4, w3, [x6, #-8]
0x2589ec: stp w2, w1, [x6]
...
0x258a14: str x0, [x21, #8]    ; advance vector end by 0x1a8
```

So the final row payload is a `0x1a8`-byte object in a vector-like holder. The
compiler advances the current row cursor by `0x48` before copying scalar/vector
fields and then advances the row-vector end pointer by `0x1a8`.

**Record field values observed across all hits**
(shape `b1_c32_h14_w14_oc128_wic32_k3x3_g1`):

| Hit | w0 | w23 | w28 | w24 | Conservative interpretation |
|-----|----|-----|-----|-----|---------------------------|
| 1 | 11 | 0 | 1 | 0 | Setup/seed-family row |
| 2 | 11 | 0 | 2 | 3 | Shape/metadata-family row |
| 3 | 11 | 1 | 2 | 3 | Shape/metadata-family row, row index 1 |
| 4 | 11 | 0 | 3 | 7 | Multi-tile-family row, row index 0 |
| 5 | 11 | 1 | 3 | 7 | Multi-tile-family row, row index 1 |
| 6 | 11 | 2 | 3 | 7 | Multi-tile-family row, row index 2 |

**w0 (value 11)**: not yet named. It is not the same as the `qbuild_interval`
halo-adjusted source end (`6`, `10`, `14` for the small proxy), so do not use it
as `src_end` in `conv.py`.

**w23 (tile_index)**: 0 for first tile, 1 for second, 2 for third.  Matches
`x3` from `qbuild_interval`.

**w28 (row family/type candidate)**:
- `1` = initial setup/seed family
- `2` = shape/metadata family
- `3` = multi-tile family

**w24 (flags)**:
- `0` = initial call
- `3` = shape tuple
- `7` = multi-tile (3 Y tiles)

### Temporary output vectors

The function uses `std::vector::_M_assign_aux` to build two output vectors at
stack offsets `[sp+0x850]` and `[sp+0x868]`:

**Vector [sp+0x850]** (observed contents):
| Hit | Content | Meaning |
|-----|---------|---------|
| 1 | `(1, 32, 14, 14)` | Input shape (N, C, H, W) |
| 2 | `(1, 32, 14, 14)` | Input shape (same) |
| 3 | `(1, 32, 14, 14)` | Input shape (same) |
| 4 | `(1, 32, 6, 14)` | Y-tiled input shape (height=6 for kernel overlap) |
| 5 | `(1, 32, 6, 14)` | Y-tiled input shape |
| 6 | `(1, 32, 6, 14)` | Y-tiled input shape |

**Vector [sp+0x868]** (observed contents, still partly unnamed):
| Hit | Content | Meaning |
|-----|---------|---------|
| 1 | (empty) | Shape tuple for second output |
| 2 | `(1, 1, ?, ?)` | Output shape or metadata |
| 3 | (empty/overwritten) | — |
| 4-6 | `(?, ?, 4, 12)` | Y-tile boundaries or output shape |

When the trace moves from the single-Y-tile conv (hits 1-3) to the 3-Y-tile conv
(hits 4-6), the shape vector height changes from 14 to 6, confirming this is a
Y-tiled input window per tile.

### Current model

The RKNN compiler path observed so far does not expose a simple final
`{xstart, ystart, kstart, data_reuse, weight_reuse}` array at `0x257f00`. At
that point it has:

1. Boundary vectors (`y_boundary`, `k_boundary`) that define split points.
2. Shape/window vectors for the current tile or row family.
3. A 4-dword stack record `(w0, w23, w28, w24)`.
4. A later `0x1a8`-byte row object copied into a vector-like destination.

`experimental/trace_librknnc_row_publish.gdb` was then run on the remote
official RKNN host with `RKNN_LOG_LEVEL=3` explicitly exported from the
`~/.bashrc` line. Logs:

```text
/tmp/rknnc_row_publish_small.log
/tmp/rknnc_row_publish_stress.log
```

Both builds completed normally and produced RKNN files:

```text
/tmp/rknnc_trace_models/conv2d_trace_b1_c32_h14_w14_oc128_wic32_k3x3_g1.rknn
/tmp/rknnc_trace_models/conv2d_trace_b1_c160_h14_w14_oc320_wic160_k3x3_g1.rknn
```

The dynamic trace confirms that `0x258a14` advances a destination vector by
`0x1a8` bytes, so the published object is a 424-byte row descriptor. The row is
not just the 16-byte `(w0,w23,w28,w24)` staging record; that record is copied
into the scalar tail around offset `0x178`.

Confirmed row layout skeleton:

| Row offset | Meaning from trace |
|------------|--------------------|
| `0x000..0x03f` | Scalar/header fields. Still partly unnamed. |
| `0x040` | Current input/window shape vector. |
| `0x058` | Current output/window shape vector. |
| `0x070` | Current weight/window shape vector. |
| `0x088` | Empty in these FP16 conv traces. |
| `0x0a0` | Full input shape vector. |
| `0x0b8` | Full output shape vector. |
| `0x0d0` | Full weight shape vector. |
| `0x0e8` | Empty in these FP16 conv traces. |
| `0x100` | 8-byte metadata vector, observed `[1,1]`. |
| `0x118` | 8-byte metadata vector, observed `[1,1]`. |
| `0x130` | 16-byte metadata vector, observed all zero. |
| `0x178` | Tail scalar copy of the 16-byte row record plus adjacent fields. |

Small proxy (`1x32x14x14 -> 1x128x12x12`) published row windows:

| Row | Record tail at `0x178` | `0x40` current input | `0x58` current output | `0x70` current weight | Full shapes at `0xa0/0xb8/0xd0` |
|-----|------------------------|----------------------|-----------------------|-----------------------|----------------------------------|
| 0 | `(11,0,3,7)` | `(1,32,6,14)` | `(1,128,4,12)` | `(128,32,3,3)` | `(1,32,14,14)`, `(1,128,12,12)`, `(128,32,3,3)` |
| 1 | `(11,2,3,7)` | `(1,32,6,14)` | `(1,128,4,12)` | `(128,32,3,3)` | same full shapes |

Stress shape (`1x160x14x14 -> 1x320x12x12`) published row windows:

| Row | Record tail at `0x178` | `0x40` current input | `0x58` current output | `0x70` current weight | Full shapes at `0xa0/0xb8/0xd0` |
|-----|------------------------|----------------------|-----------------------|-----------------------|----------------------------------|
| 0 | `(10,0,3,7)` | `(1,160,14,14)` | `(1,112,12,12)` | `(112,160,3,3)` | `(1,160,14,14)`, `(1,320,12,12)`, `(320,160,3,3)` |
| 1 | `(10,2,3,7)` | `(1,160,14,14)` | `(1,96,12,12)` | `(96,160,3,3)` | same full shapes |

Interpretation:

- `0x40/0x58/0x70` are the compiler's current tile/window tensors. They expose
  Y tiling in the small proxy (`output_h=4`, input-window height `6`) and OC
  tiling in the stress shape (`output_c=112` then `96`).
- `0xa0/0xb8/0xd0` preserve the original full input/output/weight shapes.
- The `w21` values seen before publish are channel offsets, not row-vector byte
  offsets: stress rows use `0,112,224` before publishing and the published
  output windows are `112` then `96`, consistent with OC split boundaries
  `[0,112,224,320]`. The earlier K/output-channel boundary writer only showed
  `[0,320]`, so this OC split is created in the later row/window construction
  path.
- The two published rows in these traces do not yet prove the final command
  emission or visible reuse-bit mapping. The next compiler-side trace should
  follow consumers of the `0x1a8` row vector after `RKNNTilingPass`, especially
  into `RKNNModelBuildPass` / `RKNNModelRegCmdbuildPass`.

Follow-up consumer trace:

`experimental/trace_librknnc_row_consumers.gdb` traces the sibling code sites
that also use the `0x1a8` row stride:

| Trace point | Offset | Static role |
|-------------|--------|-------------|
| `row_scan_a` | `0x00635ce8` | Walks a vector of `0x1a8` rows and counts rows whose first dword matches the current loop index. |
| `row_scan_b` | `0x01146cd0` | Similar row scanner in another call family. |
| `row_scan_c` | `0x01707520` | Similar row scanner in another call family. |
| `row_republish_before` | `0x00cac928` | Second row-publisher candidate. |
| `row_republish_commit` | `0x00cac94c` | Second row-publisher candidate commit, also advances by `0x1a8`. |

It was run on the stress shape:

```text
/tmp/rknnc_row_consumers_stress.log
```

The build completed normally. Hit counts:

| Trace point | Hits |
|-------------|------|
| `row_scan_a` | 54 |
| `row_scan_b` | 0 |
| `row_scan_c` | 0 |
| `row_republish_before` | 0 |
| `row_republish_commit` | 0 |

So, for the tested FP16 conv build, the next confirmed consumer is
`librknnc.so+0x635ce8`. It scans the same row descriptor vector with a fixed
`0x1a8` stride. The loop increments a counter when `[row+0]` matches the current
1-based loop index:

```asm
0x635cc8: add w2, w2, #1
0x635cd0: ldr w1, [x0]       ; row[0]
0x635cd4: cmp w1, w2
0x635cdc: ldr w1, [x4]
0x635ce0: add w1, w1, #1
0x635ce4: str w1, [x4]
0x635ce8: add x0, x0, #0x1a8
```

The stress trace exposes a compact post-publish row family distinct from the
earlier shape-vector-rich rows:

| Example scan row | Header dwords | Tail clue |
|------------------|---------------|-----------|
| setup row | `1,0,4,...` | includes `out_h=12`, `w0=10`, row index `0` later in the tail |
| metadata row | `2,3,6,...` | includes channel offset `0xa0` for row index `1` |
| metadata row | `2,3,6,...` | no channel offset for row index `0` |
| multi-tile row | `3,7,6,...` | includes channel offset `0x70` for row index `1` |
| multi-tile row | `3,7,6,...` | includes channel offset `0xe0` for row index `2` |

This confirms that, after the first publish, the compiler has a second
row-vector representation where:

- `row[0]` is used as a 1-based row class/index for counting.
- `row[1]=3`, `row[2]=6` appears on the stress metadata rows.
- `row[0]=3`, `row[1]=7`, `row[2]=6` appears on the stress multi-tile rows.
- Tail scalar values include the stress OC offsets `0x70` and `0xe0`, matching
  the previously inferred OC boundaries `[0,112,224,320]`.

The row scan still does not identify final register destinations. It narrows the
next target to the caller around `0x634748..0x635d10` and the downstream code
that consumes the counts generated from this scan.

Follow-up row-to-task trace:

`experimental/trace_librknnc_row_to_task.gdb` was run on the remote official
RKNN host for the stress shape:

```text
/tmp/rknnc_row_to_task_stress.log
```

The build completed normally and exported:

```text
/tmp/rknnc_trace_models/conv2d_trace_b1_c160_h14_w14_oc320_wic160_k3x3_g1.rknn
```

Hit counts in this normal FP16 conv build:

| Trace point | Hits |
|-------------|------|
| `count_bucket_select` | 3 |
| `row_base_selected` | 3 |
| `row_shape_filter` | 6 |
| `row_mode_filter` | 6 |
| `row_flag_check` | 6 |
| `offset0_ready` | 6 |
| `offset1_ready` | 6 |
| `task_desc_begin` | 6 |
| `task_desc_fields` | 6 |
| `task_desc_vectors` | 6 |
| `task_desc_push16` | 2 |

The static candidate helper path at `librknnc.so+0x5aea40` and the
`task_desc_helper_call` breakpoint at `0x636574` did not fire in this build, so
do not treat that helper as part of the normal stress FP16 conv path.

The six selected `0x1a8` rows are exactly the compact row families previously
seen by `row_scan_a`. The selected row base is `[sp+1104] + previous_count *
0x1a8`; `previous_count` comes from the prefix sum over the class-count array.
The task-builder uses fields from the selected row, including:

| Row offset | Use confirmed by static/dynamic trace |
|------------|----------------------------------------|
| `0x30..0x3f` | Header/scalar area feeding task descriptor flags. |
| `0x40` | Current input/window shape. |
| `0x58` | Current output/window shape. |
| `0x70` | Current weight/window shape. |
| `0xa0` | Full input shape. |
| `0xb8` | Full output shape. |
| `0xd0` | Full weight shape. |
| `0x148..` | Scalar tail; includes OC offset and compact family tuple. |
| `0x14c` | OC offset in the compact rows (`0x70`, `0xe0` on later OC tiles). |
| `0x17c..0x188` | Compact tuple fields such as `w0=10`, row index, family, flags, and vector count. |
| `0x180` | Row/subtile index: observed `0`, `1`, `2`. |
| `0x184` | Family/type: observed `1`, `2`, `3`. |
| `0x188` | Flags: observed `0`, `3`, `7`. |
| `0x18c` | Vector-count-like field: observed `4` on setup row and `6` on metadata/multi-tile rows. |

Stress row sequence selected by `task_desc_vectors`:

| Hit | `x21/0x1a8` | Current output (`0x58`) | Current weight (`0x70`) | Tail OC offset | Tail tuple |
|-----|------------|--------------------------|--------------------------|----------------|------------|
| 1 | 0 | `(1,320,12,12)` | `(320,160,3,3)` | `0` | `(10,0,1,0,4)` |
| 2 | 1 | `(1,160,12,12)` | `(160,160,3,3)` | `0` | `(10,0,2,3,6)` |
| 3 | 2 | `(1,160,12,12)` | `(160,160,3,3)` | `0xa0` | `(10,1,2,3,6)` |
| 4 | 3 | `(1,112,12,12)` | `(112,160,3,3)` | `0` | `(10,0,3,7,6)` |
| 5 | 4 | `(1,112,12,12)` | `(112,160,3,3)` | `0x70` | `(10,1,3,7,6)` |
| 6 | 5 | `(1,96,12,12)` | `(96,160,3,3)` | `0xe0` | `(10,2,3,7,6)` |

Only hits 4 and 6 reached `task_desc_push16` in this trace. Those are the
multi-tile-family rows for OC offsets `0` and `0xe0`; hit 5 was selected but
did not push at the traced site, so there is likely either a later/alternate
push for the middle OC tile or a grouping rule that merges the middle tile with
one of the pushed descriptors. This is not yet enough to name final register
reuse bits.

The pushed object at `0x6358d8` is a 16-byte entry copied to another vector. In
this run the two pushed entries were pointer pairs, not decoded register words:

```text
push #1: 68c95308 3fd1d974 86b6b738 bfcc8007
push #2: 5b3f6470 00000055 5b397700 00000055
```

The row-to-task path confirms OC-window grouping and scalar offsets. The
exported RKNN file also contains plain little-endian Rocket regcmd qwords, so
the final emitted command stream can be read directly for this shape.

Follow-up exported-regcmd extraction:

The remote model was copied from:

```text
/tmp/rknnc_trace_models/conv2d_trace_b1_c160_h14_w14_oc320_wic160_k3x3_g1.rknn
```

Local copy:

```text
/tmp/rknnc_trace_models_local/conv2d_trace_b1_c160_h14_w14_oc320_wic160_k3x3_g1.rknn
```

The file begins with `RKNN` and has six aligned contiguous runs of decoded
Rocket register commands using the same qword encoding as the local examples:

```text
qword = (target << 48) | ((value & 0xffffffff) << 16) | reg_addr
```

Detected runs:

| File offset | Qwords | Interpretation |
|-------------|--------|----------------|
| `0x2640` | 108 | Setup/full row; has a short preamble before the normal body. |
| `0x29b8` | 107 | Metadata/full-half row. |
| `0x2e40` | 106 | Metadata/full-half row with OC offset `0xa0`. |
| `0x32c0` | 106 | Multi-tile row, OC offset `0`. |
| `0x3740` | 106 | Multi-tile row, OC offset `0x70`. |
| `0x3bc0` | 106 | Multi-tile row, OC offset `0xe0`. |

These six runs line up with the six `0x1a8` compact rows selected by
`task_desc_vectors`.

Important emitted register values across the six runs:

| Register | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 |
|----------|-------|-------|-------|-------|-------|-------|
| CNA `0x1010` `CONV_CON2` | `0x000000f0` | `0x400000f0` | `0x400000f0` | `0x500000f0` | `0x500000f0` | `0x500000f0` |
| CNA `0x1030` `WEIGHT_SIZE0` | `0x000e1000` | `0x00070800` | `0x00070800` | `0x0004ec00` | `0x0004ec00` | `0x00043800` |
| CNA `0x1038` `WEIGHT_SIZE2` | `0x03030140` | `0x030300a0` | `0x030300a0` | `0x03030070` | `0x03030070` | `0x03030060` |
| CNA `0x1110` `DCOMP_ADDR0` | `0x00000000` | `0x00000000` | `0x00070800` | `0x00000000` | `0x0004ec00` | `0x0009d800` |
| CORE `0x3018` | `0x0000013f` | `0x0000009f` | `0x0000009f` | `0x0000006f` | `0x0000006f` | `0x0000005f` |
| DPU `0x403c` | `0x013f013f` | `0x009f009f` | `0x009f009f` | `0x006f006f` | `0x006f006f` | `0x005f005f` |
| DPU `0x4058` | `0x0000013f` | `0x0000009f` | `0x0000009f` | `0x0000006f` | `0x0000006f` | `0x0000005f` |

Stable values for this stress shape include CNA `DATA_SIZE1=0x001f00a0`,
CNA `DATA_SIZE3=0x90`, CNA `WEIGHT_SIZE1=0xb40`, CNA `CBUF_CON0=0xa2`, CNA
`CBUF_CON1=0x46`, CNA `FEATURE_DATA_ADDR=0`, CNA `DMA_CON1=0x38`, CNA
`DMA_CON2=0x8c`, CNA `FC_DATA_SIZE1=0xa0`, CORE `0x3010=0x200`, DPU
`S_POINTER=0x0e`, DPU `0x4030=0x0b`, DPU `0x4034=0x0b`, DPU
`0x405c=0x000b000b`, and DPU `0x4070=0x383`.

For the multi-tile rows, `CONV_CON2=0x500000f0` is now compiler ground truth.
The OC tile widths are directly visible in `WEIGHT_SIZE2` low bits and the
matching CORE/DPU channel-end fields: `0x70`, `0x70`, and `0x60` for channel
counts `112`, `112`, and `96`. `DCOMP_ADDR0` advances by weight-tile byte
offsets `0`, `0x4ec00`, and `0x9d800`, matching the compact row OC offsets
`0`, `0x70`, and `0xe0`.

### Exported regcmd comparison set

To avoid overfitting the stress shape, four official compiler models were built
on the remote RKNN host and copied locally:

```text
/tmp/rknnc_compare_models_local/cmp_b1_c32_h14_w14_oc128_wic32_k3x3_g1.rknn
/tmp/rknnc_compare_models_local/cmp_b1_c160_h14_w14_oc320_wic160_k3x3_g1.rknn
/tmp/rknnc_compare_models_local/cmp_b1_c528_h14_w14_oc32_wic528_k1x1_g1.rknn
/tmp/rknnc_compare_models_local/cmp_b1_c32_h150_w150_oc32_wic1_k3x3_g32.rknn
```

The helper `experimental/rknn_parse_regcmd_runs.py` scans these files for
aligned Rocket command runs and dumps stable key registers. It deliberately
parses only the exported model bytes; it does not use runtime submission state.

Command used:

```sh
python3 experimental/rknn_parse_regcmd_runs.py \
  /tmp/rknnc_compare_models_local/*.rknn --min-qwords 24
```

The comparison confirms these patterns for the tested FP16 conv models:

| Shape family | Runs | Main emitted pattern |
|--------------|------|----------------------|
| Spatial `32->128`, `14x14`, `3x3` | 6 | Runs 1-3 are full/half OC rows (`CONV_CON2` `0x00000110`, then `0x40000110`); runs 4-6 are Y rows (`0x20000090`) with input height 6 and output height 4. |
| Spatial `160->320`, `14x14`, `3x3` | 6 | Runs 1-3 are full/160-channel rows (`0x000000f0`, `0x400000f0`); runs 4-6 are OC tile rows (`0x500000f0`) with OC widths `112,112,96`. |
| Pointwise `528->32`, `14x14`, `1x1` | 6 | Runs 1-3 are full/16-channel rows (`0x00000090`, `0x40000090`); runs 4-6 are Y rows (`0x20000060`, `0x20000060`, `0x20000050`) with output heights `5,5,4`. |
| Depthwise `32->32`, `150x150`, `3x3`, groups 32 | 14 | Runs use depthwise mode (`CONV_CON1=0x123`, CORE `0x3010=0x202`, DPU feature mode `0x1fc`) and a long Y schedule with `CONV_CON2` high bits `0x0`, `0x1`, then `0x2`. |

Field-level observations:

- `CNA_CONV_CON2` low bits retain the grain-ish value already seen in Python
  (`0xf0`, `0x110`, `0x90`, `0xc0`, etc.). The high nibble selects tile-family
  behavior:
  - `0x00000000` prefix: initial/full row.
  - `0x10000000` prefix: depthwise middle Y-family rows.
  - `0x20000000` prefix: Y-tile rows for spatial, pointwise, and later
    depthwise rows.
  - `0x40000000` prefix: channel-half/metadata rows whose `DCOMP_ADDR0` and
    DPU destination-base offset distinguish first and second halves.
  - `0x50000000` prefix: stress OC multi-tile rows.
- `CNA_DCOMP_ADDR0` advances only on OC/weight-channel splits in these models.
  For `32->128`, the half-OC row uses `0x9000`; for `528->32`, the second
  half uses `0x4200`; for `160->320`, the multi-tile offsets are `0`,
  `0x4ec00`, `0x9d800`.
- `CNA_FEATURE_DATA_ADDR` advances on Y tiles, not on pure OC splits. Examples:
  `32->128` Y rows use `0`, `0x380`, `0x700`; pointwise `528->32` uses `0`,
  `0x460`, `0x8c0`; depthwise uses larger row-stride-derived offsets such as
  `0x14820`, `0x1d4c0`, `0x3a980`, `0x48a80`.
- `DPU_DST_BASE_ADDR` mirrors output tile offset. On Y rows it tracks output-row
  offset (`0`, `0x300`, `0x600` for `32->128`; `0`, `0x460`, `0x8c0` for
  pointwise). On OC rows it tracks output-channel offset (`0`, `0x7e00`,
  `0xfc00` for the stress multi-tile rows).
- The output-channel count is encoded consistently in three places:
  `CNA_WEIGHT_SIZE2` low bits, CORE `0x3018`, and DPU `0x403c/0x4058`. For
  channel count `N`, CORE/DPU use `N-1`, while `CNA_WEIGHT_SIZE2` uses `N`.
- The output spatial tile size is encoded consistently in CNA `DATA_SIZE0`
  input-window height, CNA `DATA_SIZE3` atomics/area-like field, CORE
  `0x3014`, DPU `0x4034`, and DPU `0x405c`.

Implication for a clean `examples/kernel_6_18/conv.py`: it should not need
shape-name special cases for these families. A small task/tile descriptor can
drive a single regcmd emitter:

```text
tile = {
  input_y, input_h, output_y, output_h,
  output_c_start, output_c_count,
  conv_con2_family_bits,
  feature_offset,
  weight_offset,
  output_offset,
}
```

The remaining reverse gap is now the generic planner rule that assigns
`conv_con2_family_bits` and tile order for arbitrary shapes, especially mixed
Y+K cases. The emitted register destinations and most per-tile value formulas
are no longer speculative for the comparison set above.

### Descriptor view for `conv.py`

The exported command runs can be reduced to a descriptor that is close to what a
clean Python emitter should consume. Running:

```sh
python3 experimental/rknn_parse_regcmd_runs.py \
  /tmp/rknnc_compare_models_local/*.rknn --min-qwords 24 --descriptors
```

prints one descriptor per regcmd run:

```text
idx,off,n,family,family_bits,grain_bits,input_h,output_h,output_w,
oc_count,oc_end,feature_off,weight_off,output_off,weight_bytes,cbuf0
```

Representative rows:

| Shape | Run | Family | Input H | Output H/W | OC count | Feature off | Weight off | Output off |
|-------|-----|--------|---------|------------|----------|-------------|------------|------------|
| `32->128 3x3` | 4 | `y_tile` (`0x20000000`) | 6 | 4x12 | 128 | `0x0` | `0x0` | `0x0` |
| `32->128 3x3` | 5 | `y_tile` (`0x20000000`) | 6 | 4x12 | 128 | `0x380` | `0x0` | `0x300` |
| `32->128 3x3` | 6 | `y_tile` (`0x20000000`) | 6 | 4x12 | 128 | `0x700` | `0x0` | `0x600` |
| `160->320 3x3` | 4 | `k_tile` (`0x50000000`) | 14 | 12x12 | 112 | `0x0` | `0x0` | `0x0` |
| `160->320 3x3` | 5 | `k_tile` (`0x50000000`) | 14 | 12x12 | 112 | `0x0` | `0x4ec00` | `0x7e00` |
| `160->320 3x3` | 6 | `k_tile` (`0x50000000`) | 14 | 12x12 | 96 | `0x0` | `0x9d800` | `0xfc00` |
| `528->32 1x1` | 4 | `y_tile` (`0x20000000`) | 5 | 5x14 | 32 | `0x0` | `0x0` | `0x0` |
| `528->32 1x1` | 5 | `y_tile` (`0x20000000`) | 5 | 5x14 | 32 | `0x460` | `0x0` | `0x460` |
| `528->32 1x1` | 6 | `y_tile` (`0x20000000`) | 4 | 4x14 | 32 | `0x8c0` | `0x0` | `0x8c0` |

For the current `examples/kernel_6_18/conv.py`, the important mismatch is that
`make_conv2d_regs()` only accepts DMA bases plus `weight_reuse/full_data_bank`.
The compiler-derived descriptor needs these fields to be independent per run:

```text
conv_con2 = family_bits | grain_bits
CNA_FEATURE_DATA_ADDR = input_dma + feature_off
CNA_DCOMP_ADDR0       = weight_dma + weight_off
DPU_DST_BASE_ADDR     = output_dma + output_off
```

and the tile-local params must use the descriptor's `input_h`, `output_h`,
`output_w`, and `oc_count` rather than recomputing everything from the original
full layer shape.

Current `conv.py` already has a boundary-vector planner skeleton, but the
emitter still hides the compiler's per-run offsets and `CONV_CON2` family bits.
The next implementation-oriented step is therefore not another runtime special
case; it is to refactor the emitter around a `TileDesc`/`RegDesc` object and
then teach the planner to produce RKNN-like descriptor families.

### Mixed Y+K exported schedules

To force mixed height and channel movement, three additional official compiler
models were built and copied locally:

```text
/tmp/rknnc_mixed_models_local/mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1.rknn
/tmp/rknnc_mixed_models_local/mix_b1_c72_h20_w20_oc288_wic72_k3x3_g1.rknn
/tmp/rknnc_mixed_models_local/mix_b1_c40_h40_w40_oc320_wic40_k1x1_g1.rknn
```

The key mixed case is `1x160x40x40 -> 1x320x38x38`, `3x3`, groups 1. Its
descriptor CSV shows 12 command runs:

| Runs | Family | Y tile | OC tile | Offsets |
|------|--------|--------|---------|---------|
| 1-2 | `setup` (`0x00000000`) | output `21`, then `17` rows | full 320 | feature/output offsets `0`, then `0x3480/0x31e0` |
| 3-6 | `k_half` (`0x40000000`) | two Y rows for each 160-channel half | 160, 160 | second half adds weight `0x70800`, output base `0x70d00`; second Y row adds `0x3480` feature and `0x31e0` output |
| 7-12 | `k_tile` (`0x50000000`) | two Y rows for each OC tile | 112, 112, 96 | OC bases `0`, `0x4ef80`, `0x9df00`; second Y row adds `0x31e0` output |

The exact `k_tile` rows:

| Run | Input H | Output H/W | OC count | Feature off | Weight off | Output off |
|-----|---------|------------|----------|-------------|------------|------------|
| 7 | 23 | 21x38 | 112 | `0x0` | `0x0` | `0x0` |
| 8 | 19 | 17x38 | 112 | `0x3480` | `0x0` | `0x31e0` |
| 9 | 23 | 21x38 | 112 | `0x0` | `0x4ec00` | `0x4ef80` |
| 10 | 19 | 17x38 | 112 | `0x3480` | `0x4ec00` | `0x52160` |
| 11 | 23 | 21x38 | 96 | `0x0` | `0x9d800` | `0x9df00` |
| 12 | 19 | 17x38 | 96 | `0x3480` | `0x9d800` | `0xa10e0` |

This proves the mixed-order rule for this compiler path:

```text
for family in [setup, k_half, k_tile]:
  for k_window in family_k_windows:
    for y_window in y_windows:
      emit descriptor with additive offsets
```

For the tested mixed spatial case, output offsets are additive:

```text
output_off = output_k_base + output_y_base
```

Examples:

```text
0x52160 = 0x4ef80 + 0x31e0
0xa10e0 = 0x9df00 + 0x31e0
```

Feature offsets depend only on Y (`0`, `0x3480`), and weight offsets depend
only on K (`0`, `0x4ec00`, `0x9d800`). This is the strongest evidence so far
that a clean `conv.py` should create independent Y and K windows, then combine
them into per-family descriptors with additive DMA offsets.

Two comparison points:

- `72x20 -> 288`, `3x3` does not Y-tile; it has setup, `k_half`, and `k_tile`
  families only, with OC windows `96,96,96`.
- `40x40 -> 320`, `1x1` pointwise emits setup and `k_half` families for full
  height, then separate full-OC `y_tile` rows (`14,13,13` output rows). It does
  not combine pointwise Y rows with K rows in this shape.

Remaining planner gap after this pass:

- The emitted mixed descriptor order and offset arithmetic are now known for a
  large spatial FP16 conv.
- Still missing: the general threshold rule that chooses between full,
  `k_half`, `k_tile`, and pointwise full-OC `y_tile` descriptor families for all
  shapes. The current best route is to compare the descriptor CSV against
  `top_tiler` row-family fields (`row[0]`, `0x184`, `0x188`, `0x18c`) for the
  new mixed `160x40` model.

### `160->320 3x3` height sweep

To pin down the Y split threshold for the same spatial-channel path, these
official compiler models were exported from the RKNN host and copied locally:

```text
/tmp/rknnc_sweep_models_local/sweep_b1_c160_h16_w16_oc320_wic160_k3x3_g1.rknn
/tmp/rknnc_sweep_models_local/sweep_b1_c160_h20_w20_oc320_wic160_k3x3_g1.rknn
/tmp/rknnc_sweep_models_local/sweep_b1_c160_h24_w24_oc320_wic160_k3x3_g1.rknn
/tmp/rknnc_sweep_models_local/sweep_b1_c160_h28_w28_oc320_wic160_k3x3_g1.rknn
/tmp/rknnc_sweep_models_local/sweep_b1_c160_h32_w32_oc320_wic160_k3x3_g1.rknn
/tmp/rknnc_sweep_models_local/sweep_b1_c160_h36_w36_oc320_wic160_k3x3_g1.rknn
/tmp/rknnc_sweep_models_local/sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1.rknn
```

The parser command was:

```sh
python3 experimental/rknn_parse_regcmd_runs.py \
  /tmp/rknnc_sweep_models_local/*.rknn --csv
```

The key result is that this path does not Y-tile up through input height 28.
Heights 16, 20, 24, and 28 each emit six descriptors:

```text
setup full, k_half 160, k_half 160, k_tile 112, k_tile 112, k_tile 96
```

At input height 32 and above, the compiler crosses every family with two Y
windows. The descriptor count becomes 12:

```text
setup x2, k_half x4, k_tile x6
```

Exact Y windows observed:

| Input H/W | Output H/W | Y windows | Feature offsets | Output Y offsets | Grain bits |
|-----------|------------|-----------|-----------------|------------------|------------|
| 16 | 14 | full only | `0x0` | `0x0` | `0xf0` |
| 20 | 18 | full only | `0x0` | `0x0` | `0xf0` |
| 24 | 22 | full only | `0x0` | `0x0` | `0xf0` |
| 28 | 26 | full only | `0x0` | `0x0` | `0xc0` |
| 32 | 30 | `26,4` | `0x0,0x3400` | `0x0,0x30c0` | `0xc0,0x90` |
| 36 | 34 | `23,11` | `0x0,0x33c0` | `0x0,0x30e0` | `0xc0,0xc0` |
| 40 | 38 | `21,17` | `0x0,0x3480` | `0x0,0x31e0` | `0xc0,0xc0` |

For all sweep heights, the K-side schedule stays stable:

```text
k_half output-channel windows: 160, 160
k_tile output-channel windows: 112, 112, 96
weight offsets: 0, 0x70800 for k_half; 0, 0x4ec00, 0x9d800 for k_tile
```

Output K bases scale with full output height/width and FP16 element size, then
the per-Y output offset is added. For example, at `32x32 -> 30x30`, the second
112-channel `k_tile` base is `112 * 30 * 30 * 2 = 0x31380`, and its second Y
row adds `0x30c0` to produce `0x34440`.

`experimental/rknn_descriptor_plan.py --mode observed` now also covers this
exact `160->320`, `3x3`, groups-1 height sweep. It was checked against all seven
exported models:

```sh
for f in /tmp/rknnc_sweep_models_local/sweep_b1_c160_h*_w*_oc320_wic160_k3x3_g1.rknn; do
  shape=$(basename "$f" .rknn)
  python3 experimental/rknn_descriptor_plan.py "$shape" \
    --mode observed --compare-rknn "$f"
done
```

All rows matched exactly. The check exposed an important emitted-descriptor
detail: the `CNA_FEATURE_DATA_ADDR` Y offset in these exported regcmds advances
by `output_y * input_w * 8 * 2`, not by `output_y * input_w * aligned_in_c * 2`.
For example:

```text
32x32, second Y window starts at output_y=26:
feature_off = 26 * 32 * 8 * 2 = 0x3400

40x40, second Y window starts at output_y=21:
feature_off = 21 * 40 * 8 * 2 = 0x3480
```

So the clean descriptor emitter should treat the compiler-provided
`feature_off` as a descriptor field. Do not recompute it from full tensor byte
size in the register emitter.

### `20x20 -> 18x18`, `out_c=320` channel sweep

To separate the K-family choice from the height threshold, these official
compiler models were exported with fixed `20x20` input, `3x3`, groups 1, and
`out_c=320`:

```text
/tmp/rknnc_channel_models_local/chan_b1_c32_h20_w20_oc320_wic32_k3x3_g1.rknn
/tmp/rknnc_channel_models_local/chan_b1_c40_h20_w20_oc320_wic40_k3x3_g1.rknn
/tmp/rknnc_channel_models_local/chan_b1_c64_h20_w20_oc320_wic64_k3x3_g1.rknn
/tmp/rknnc_channel_models_local/chan_b1_c72_h20_w20_oc320_wic72_k3x3_g1.rknn
/tmp/rknnc_channel_models_local/chan_b1_c96_h20_w20_oc320_wic96_k3x3_g1.rknn
/tmp/rknnc_channel_models_local/chan_b1_c128_h20_w20_oc320_wic128_k3x3_g1.rknn
/tmp/rknnc_channel_models_local/chan_b1_c160_h20_w20_oc320_wic160_k3x3_g1.rknn
/tmp/rknnc_channel_models_local/chan_b1_c192_h20_w20_oc320_wic192_k3x3_g1.rknn
/tmp/rknnc_channel_models_local/chan_b1_c256_h20_w20_oc320_wic256_k3x3_g1.rknn
```

The parser command was:

```sh
python3 experimental/rknn_parse_regcmd_runs.py \
  /tmp/rknnc_channel_models_local/*.rknn --csv
```

Observed family break for this fixed output shape:

| Input C | Descriptor families after setup/k-half | OC windows | Grain bits | CBUF0 |
|---------|----------------------------------------|------------|------------|-------|
| 32 | `y_tile` x3 | full 320 each | setup/k-half `0x170`, y `0xb0` | `0xb1` |
| 40 | `y_tile` x3 | full 320 each | setup/k-half `0x170`, y `0xb0` | `0xb1` |
| 64 | `k_tile` x3 | `112,112,96` | `0x140` | `0xa2` |
| 72 | `k_tile` x3 | `112,112,96` | `0x140` | `0xa2` |
| 96 | `k_tile` x3 | `112,112,96` | `0x120` | `0x93` |
| 128 | `k_tile` x3 | `112,112,96` | `0xf0` | `0x84` |
| 160 | `k_tile` x3 | `112,112,96` | `0xf0` | `0x84` |
| 192 | `k_tile` x3 | `112,112,96` | `0xf0` | `0x75` |
| 256 | `k_tile` x3 | `112,112,96` | `0xc0` | `0x57` |

The break is not "small input channel means fewer OC tiles." At `out_c=320`,
`in_c=64` already switches to the same `k_tile` OC schedule as the larger
160-channel stress case. Meanwhile `in_c=32` and `40` keep full-OC Y rows:

```text
setup full, k_half 160, k_half 160, y_tile full, y_tile full, y_tile full
```

For the `k_tile` branch, OC windows are stable across `in_c=64..256` in this
sweep:

```text
112, 112, 96
```

The weight byte counts and offsets still scale directly with
`kh * kw * in_c * 2 * oc_count` in the exported spatial-conv descriptors.
Examples:

```text
in_c=64:  k_tile 112 weight bytes = 0x1f800
in_c=160: k_tile 112 weight bytes = 0x4ec00
in_c=256: k_tile 112 weight bytes = 0x7e000
```

This falsifies the earlier rough "about 3 CBUF banks of weights" estimate in
`experimental/rknn_descriptor_plan.py`; the compiler is selecting the same
112-channel OC tile width even when that consumes roughly 4, 10, or 16 CBUF
banks worth of raw FP16 weights. The remaining unknown is therefore a
compiler-family selection rule, not a simple per-tile weight-CBUF capacity
formula.

The same sweep is now covered by `experimental/rknn_descriptor_plan.py --mode
observed`. It was checked against all nine exported channel-sweep models:

```sh
for f in /tmp/rknnc_channel_models_local/chan_b1_c*_h20_w20_oc320_wic*_k3x3_g1.rknn; do
  shape=$(basename "$f" .rknn)
  python3 experimental/rknn_descriptor_plan.py "$shape" \
    --mode observed --compare-rknn "$f"
done
```

All rows matched exactly. This check is what exposed the raw-`in_c` weight
stride above: `in_c=40` uses second-half weight offset `0x1c200`, and `in_c=72`
uses k-tile offsets `0`, `0x23700`, `0x46e00`. Those are
`oc_start * kh * kw * in_c * 2`, not `oc_start * kh * kw * aligned_in_c * 2`.

### `in_c=64`, `20x20 -> 18x18` output-channel sweep

To test whether `112,112,96` was a fixed OC tile size or a consequence of
`out_c=320`, these official compiler models were exported with fixed `in_c=64`,
`20x20`, `3x3`, groups 1, and varying `out_c`:

```text
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc32_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc48_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc64_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc96_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc112_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc128_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc160_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc192_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc224_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc256_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc288_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc320_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc384_wic64_k3x3_g1.rknn
/tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc512_wic64_k3x3_g1.rknn
```

Compact summaries can be reproduced with:

```sh
python3 experimental/rknn_descriptor_plan.py \
  outc_b1_c64_h20_w20_oc320_wic64_k3x3_g1 \
  --compiler-summary /tmp/rknnc_outc_models_local/outc_b1_c64_h20_w20_oc320_wic64_k3x3_g1.rknn
```

Observed post-setup family choices:

| Output C | Setup/family sequence after setup | Post-family OC windows |
|----------|-----------------------------------|------------------------|
| 32 | `k_half`, then `y_tile` | Y rows full 32 |
| 48 | `depthwise_y_mid`, then `k_tile` | `16,16,16` |
| 64 | `k_half`, then `y_tile` | Y rows full 64 |
| 96 | `k_half`, then `k_tile` | `32,32,32` |
| 112 | `depthwise_y_mid`, then `y_tile` | Y rows full 112 |
| 128 | `k_half`, then `y_tile` | Y rows full 128 |
| 160 | `k_half`, then `y_tile` | Y rows full 160 |
| 192 | `k_half`, then `k_tile` | `64,64,64` |
| 224 | `k_half`, then `y_tile` | Y rows full 224 |
| 256 | `k_half`, then `y_tile` | Y rows full 256 |
| 288 | `k_half`, then `k_tile` | `96,96,96` |
| 320 | `k_half`, then `k_tile` | `112,112,96` |
| 384 | `k_half`, then `k_tile` | `128,128,128` |
| 512 | `k_half`, then `k_tile` | `176,176,160` |

This proves `112,112,96` is not a hard-coded OC tile shape. In this compiler
path, the `k_tile` rows are usually three OC windows. For output-channel counts
that divide cleanly by 3 and stay 16-aligned, the compiler uses exact thirds:

```text
96  -> 32,32,32
192 -> 64,64,64
288 -> 96,96,96
384 -> 128,128,128
```

For non-divisible counts such as `320` and `512`, it rounds the first two thirds
up to a 16-channel boundary and leaves the remainder in the tail:

```text
320 -> 112,112,96
512 -> 176,176,160
```

Several output-channel counts do not use the `k_tile` post-family at all, even
though their total weights exceed three raw CBUF banks. They instead emit
full-OC `y_tile` rows after setup/k-half: `32`, `64`, `128`, `160`, `224`, and
`256`. The `48` and `112` cases also show `0x10000000` family rows on this
non-depthwise conv path, so that high-bit family is not exclusively a depthwise
marker; it is better named `y_mid`/middle-Y family until the row-class semantics
are fully confirmed.

Planner implication:

- `k_tile` window size should be derived from the output-channel count in this
  path, approximately `ceil_align16(out_c / 3)` for the first two rows plus a
  tail, not from raw CBUF weight capacity.
- The decision to use `k_tile` versus full-OC `y_tile` remains a separate
  compiler rule. Current evidence suggests it is tied to preferred channel
  multiples/family templates, not just weight size.
- Do not encode `0x10000000` as "depthwise only" in a clean emitter. The parser
  should treat it as a generic family bit until more `librknnc.so` row traces
  assign the true name.

`experimental/rknn_descriptor_plan.py` now has an `--mode observed` path for
this exact `in_c=64`, `20x20`, `3x3`, groups-1 sweep. It encodes only the
compiler-derived rows above, and was checked against all 14 exported models:

```sh
for f in /tmp/rknnc_outc_models_local/*.rknn; do
  shape=$(basename "$f" .rknn)
  python3 experimental/rknn_descriptor_plan.py "$shape" \
    --mode observed --compare-rknn "$f"
done
```

All rows matched exactly. The observed-mode formulas used here are:

```text
weight_off = output_c_start * kh * kw * in_c * 2
output_off = output_c_start * output_h_full * output_w * 2 + output_y * output_w * 8 * 2
feature_off = output_y * input_w * 8 * 2
```

For the `20x20 -> 18x18` Y families in this sweep, the compiler uses fixed
windows:

```text
y_mid:  two rows of output_h=9, input_h=11, y starts 0 and 9
y_tile: three rows of output_h=6, input_h=8,  y starts 0, 6, and 12
```

This gives a concrete implementation target for `conv.py`: separate the
descriptor emitter from the planner first, then make the planner choose among
these family templates. The register emitter does not need shape-name special
cases once it accepts `{family_bits, input_h, output_h, output_w, oc_count,
feature_off, weight_off, output_off}` as explicit descriptor fields.

### Pointwise `1x1` exported schedules

Pointwise compiler models were exported to separate the `1x1` family templates
from the spatial `3x3` rules:

```text
/tmp/rknnc_pw_models_local/pw_b1_c40_h20_w20_oc320_wic40_k1x1_g1.rknn
/tmp/rknnc_pw_models_local/pw_b1_c40_h28_w28_oc320_wic40_k1x1_g1.rknn
/tmp/rknnc_pw_models_local/pw_b1_c40_h40_w40_oc320_wic40_k1x1_g1.rknn
/tmp/rknnc_pw_models_local/pw_b1_c40_h56_w56_oc320_wic40_k1x1_g1.rknn
/tmp/rknnc_pw_models_local/pw_b1_c64_h20_w20_oc128_wic64_k1x1_g1.rknn
/tmp/rknnc_pw_models_local/pw_b1_c64_h40_w40_oc128_wic64_k1x1_g1.rknn
/tmp/rknnc_pw_models_local/pw_b1_c256_h14_w14_oc512_wic256_k1x1_g1.rknn
/tmp/rknnc_pw_models_local/pw_b1_c256_h28_w28_oc512_wic256_k1x1_g1.rknn
/tmp/rknnc_pw_models_local/pw_b1_c528_h14_w14_oc32_wic528_k1x1_g1.rknn
/tmp/rknnc_pw_models_local/pw_b1_c528_h20_w20_oc32_wic528_k1x1_g1.rknn
/tmp/rknnc_pw_models_local/pw_b1_c528_h40_w40_oc32_wic528_k1x1_g1.rknn
```

The clean/simple pointwise template is:

```text
setup full height/full OC
k_half first half full height
k_half second half full height
y_tile full OC for each Y window
```

This template was observed and exactly matched by
`experimental/rknn_descriptor_plan.py --mode observed` for:

```text
40->320 at 20x20, 28x28, 40x40, 56x56
64->128 at 20x20, 40x40
256->512 at 14x14
528->32 at 14x14
```

Observed simple pointwise Y windows:

| H/W | Y tile output rows | Feature/output offsets |
|-----|--------------------|------------------------|
| 14 | `5,5,4` | `0,0x460,0x8c0` |
| 20 | `7,7,6` | `0,0x8c0,0x1180` |
| 28 | `10,9,9` | `0,0x1180,0x2140` |
| 40 | `14,13,13` | `0,0x2300,0x4380` |
| 56 | `19,19,18` | `0,0x4280,0x8500` |

For pointwise, `input_h == output_h` in each descriptor because `kh=kw=1`.
The same descriptor offset formulas apply:

```text
weight_off = output_c_start * in_c * 2
output_off = output_c_start * output_h_full * output_w * 2 + output_y * output_w * 8 * 2
feature_off = output_y * input_w * 8 * 2
```

The larger pointwise exports show a second, more complex crossed-Y family:

```text
256->512 at 28x28: setup x2, k_half crossed over 4 Y windows per half, then y_tile x6
528->32 at 20x20:  setup x2, k_half crossed over 2 Y windows per half, then y_tile x3
528->32 at 40x40:  setup x2, k_half crossed over 6 Y windows per half, then y_tile x6
```

These are now also covered by `experimental/rknn_descriptor_plan.py --mode
observed`. The exact Y lists are:

| Shape | Setup Y windows | K-half Y windows | Y-tile emission order |
|-------|-----------------|------------------|-----------------------|
| `256->512`, `28x28` | `(0,9),(9,9)` | `(0,9),(9,9),(18,9),(27,1)` per half | `(0,5),(15,5),(5,5),(20,4),(10,5),(24,4)` |
| `528->32`, `20x20` | `(0,15),(15,5)` | `(0,15),(15,5)` per half | `(0,7),(7,7),(14,6)` |
| `528->32`, `40x40` | `(0,7),(7,7)` | `(0,7),(7,7),(14,7),(21,7),(28,7),(35,5)` per half | `(0,7),(21,7),(7,7),(28,6),(14,7),(34,6)` |

The non-monotonic Y-tile emission order on `28x28` and `40x40` is compiler
ground truth from exported regcmds. Do not sort these rows in a clean planner;
the descriptor order should be emitted as chosen by the family template.

### Observed descriptor contract

`experimental/rknn_descriptor_plan.py` now has two modes intended to turn the
reverse data into an implementation contract for `examples/kernel_6_18/conv.py`.

Single-shape descriptor export:

```sh
python3 experimental/rknn_descriptor_plan.py \
  pw_b1_c256_h28_w28_oc512_wic256_k1x1_g1 \
  --mode observed --json
```

This prints ordered descriptor objects with only the fields the clean register
emitter should consume:

```text
family, family_bits, grain_bits, input_h, output_h, output_w, oc_count,
feature_off, weight_off, output_off
```

Corpus coverage check:

```sh
python3 experimental/rknn_descriptor_plan.py --mode observed --coverage \
  /tmp/rknnc_channel_models_local/*.rknn \
  /tmp/rknnc_sweep_models_local/*.rknn \
  /tmp/rknnc_outc_models_local/*.rknn \
  /tmp/rknnc_pw_models_local/*.rknn
```

Current result: all 41 local compiler-exported RKNN files are known shapes,
observed-supported, and exact descriptor matches.

The helper also has a grain-bit grouping view:

```sh
python3 experimental/rknn_descriptor_plan.py --grain-summary \
  /tmp/rknnc_channel_models_local/*.rknn \
  /tmp/rknnc_sweep_models_local/*.rknn \
  /tmp/rknnc_outc_models_local/*.rknn \
  /tmp/rknnc_pw_models_local/*.rknn
```

`grain_bits` are the low bits of `CNA_CONV_CON2`, so the final emitter must set:

```text
CNA_CONV_CON2 = family_bits | grain_bits
```

The current observed checker treats unknown grain bits as an explicit wildcard
(`*`) while still printing the compiler value during comparisons. This is
intentional. The corpus shows that grain bits are not just a function of
`family,input_h,output_h,output_w,oc_count`. Examples:

```text
k_half,20,18,18,160 -> 0xc0,0xf0,0x120,0x140,0x170
k_tile,20,18,18,112 -> 0xc0,0xf0,0x120,0x140
```

So `grain_bits` remain the most important unresolved descriptor field. The next
compiler-side target is the `librknnc.so` path that writes low `CONV_CON2` bits
from row/template metadata, likely near the final regcmd build rather than the
top-level Y/K boundary planner.

Two extra helper views now make this gap easier to attack:

```sh
python3 experimental/rknn_descriptor_plan.py --grain-context \
  /tmp/rknnc_channel_models_local/*.rknn \
  /tmp/rknnc_sweep_models_local/*.rknn \
  /tmp/rknnc_outc_models_local/*.rknn \
  /tmp/rknnc_pw_models_local/*.rknn

python3 experimental/rknn_descriptor_plan.py --grain-matrix \
  /tmp/rknnc_channel_models_local/*.rknn \
  /tmp/rknnc_sweep_models_local/*.rknn \
  /tmp/rknnc_outc_models_local/*.rknn \
  /tmp/rknnc_pw_models_local/*.rknn
```

`--grain-context` prints one row per compiler descriptor with shape fields,
descriptor fields, `weight_bytes`, and `cbuf0`. `--grain-matrix` groups by:

```text
family, kh, kw, in_c, input_h, output_h, output_w, oc_count, cbuf0
```

New observations from this matrix:

- For spatial `3x3`, setup/k-half/k-tile grain bits are stable once the visible
  descriptor geometry and `cbuf0` match. They do not vary with OC tile count in
  the tested out-channel sweep. Example: all `64->*`, `20x20`, `3x3`,
  `cbuf0=0xa2` setup/k-half/k-tile descriptors use `grain_bits=0x140`, even
  when `oc_count` changes from `16` through `176`.
- For the `160->320` spatial height sweep, grain bits follow the Y-window CBUF
  template rather than family or OC tile count:

```text
input_h/output_h/output_w/cbuf0 -> grain_bits
16/14/14/0x93 -> 0xf0
20/18/18/0x84 -> 0xf0
24/22/22/0x66 -> 0xf0
28/26/26/0x48 -> 0xc0
6/4/30/0x39   -> 0x90
13/11/34/0x39 -> 0xc0
19/17/38/0x39 -> 0xc0
23/21/38/0x39 -> 0xc0
25/23/34/0x39 -> 0xc0
28/26/30/0x39 -> 0xc0
```

- For pointwise Y/crossed descriptors, the low bits often track the local row
  count directly:

```text
1 row  -> 0x20
5 rows -> 0x50
6 rows -> 0x60 or 0x70 depending on template
7 rows -> 0x70 or 0x80 depending on template
9 rows -> 0x90
14 rows -> 0xa0
```

  This is not a complete formula because simple pointwise full-window
  descriptors still show template/Cbuf effects (`40->320 56x56` full-window
  setup/k-half uses `0x100`, while 18/19 row Y tiles also use `0x100`).

- `cbuf0` high bit `0x2000` can alternate between pointwise Y descriptors
  without changing `grain_bits`. Examples: crossed `256->512 28x28` setup rows
  both use `grain_bits=0x90` while `cbuf0` alternates `0x84/0x2084`; crossed
  `528->32 40x40` setup/Y rows alternate `0x2a/0x202a` with the same
  row-derived grain value. So `grain_bits` and CBUF bank-selection bits must
  remain separate fields in the clean descriptor contract.

### Regcmd build-path probes

Two focused GDB scripts were added to push the reverse from row planning toward
the final `RKNNModelRegCmdbuildPass` command stream:

```text
experimental/trace_librknnc_regcmd_helper.gdb
experimental/trace_librknnc_regcmd_emit.gdb
```

`trace_librknnc_regcmd_helper.gdb` instruments the direct call visible in the
row/task-builder disassembly:

```asm
0x636574: bl 0x5aea40
```

and helper internals around `0x5aea40..0x5aedb8`. Running it remotely on a
self-contained official RKNN build for:

```text
1x160x20x20 -> 1x320x18x18, 3x3, groups=1
```

produced:

```text
/tmp/rknnc_regcmd_helper_conv.log
/tmp/rknnc_regcmd_helper_models/probe_b1_c160_h20_w20_oc320_wic160_k3x3_g1.rknn
```

The breakpoints installed, the model built and exported normally, but none of
the helper breakpoints fired. The exported RKNN still contains the expected six
compiler descriptors:

```text
setup  grain=0xf0
k_half grain=0xf0 x2
k_tile grain=0xf0 x3
```

So `0x5aea40` is a real conditional row/task helper, but it is not on the normal
regcmd build path for this generated spatial `20x20` model. Do not treat it as
the final `CNA_CONV_CON2` writer.

Static inspection then found tempting qword-construction loops around
`librknnc.so+0xc152e0..0xc15308`, `+0xc154b4..0xc154d8`, and a late append at
`+0xc15ff4`. These loops appear to build 8-byte records from:

```text
u16 field at sp+0x10a
u32 value at sp+0x10c
append 8-byte qword to a vector
```

`trace_librknnc_regcmd_emit.gdb` instruments those append sites and decodes the
candidate qword using the exported-RKNN format:

```text
qword = (target << 48) | ((value & 0xffffffff) << 16) | reg_addr
```

Remote run:

```text
/tmp/rknnc_regcmd_emit_conv.log
```

Again the breakpoints installed and the same RKNN model built/exported normally,
but none of the emit-loop breakpoints fired. That rules these loops out for the
normal single-conv `RKNNModelRegCmdbuildPass` path.

The current compiler-side facts are therefore:

- The row/task builder and exported regcmd stream are matched well enough to
  describe ordered descriptors.
- The final pass still writes the exported qwords somewhere else; the first two
  obvious static candidates (`0x5aea40` helper and `0xc15xxx` qword append
  loops) are negative for the representative generated `160->320 20x20` model.
- Next productive trace should start from the bytes that are definitely written
  to the `.rknn` export: catch the model export/write path or memory-copy path
  that handles the contiguous regcmd byte run, then backtrace to its producer.
  The target bytes for the same probe are:

```text
run 1 at file off 0x2640: CNA_CONV_CON2 = 0x000000f0
run 2 at file off 0x29b8: CNA_CONV_CON2 = 0x400000f0
run 4 at file off 0x32c0: CNA_CONV_CON2 = 0x500000f0
```

These negative probes are still useful for `conv.py`: they keep the clean
descriptor contract grounded in exported command bytes and avoid naming an
unrelated helper as the grain-bit source.

### RKNN export writer path

The next trace moved from candidate producers to the bytes that are definitely
present in the exported `.rknn` file. `trace_librknnc_export_writes.gdb` scans
outgoing file-write buffers for known little-endian regcmd qwords from the same
official compiler probe:

```text
experimental/trace_librknnc_export_writes.gdb
/tmp/rknnc_export_writes_conv.log
```

The trace found the final serialized RKNN buffer during `RKNNModelExportPass`.
The buffer was:

```text
buf=0x555b821990
size=941366
```

and the qword offsets inside that buffer matched the exported file offsets:

```text
CNA_CONV_CON2 setup  0x0201000000f01010 at file/buffer off 0x2670
CNA_CONV_CON2 k_half 0x0201400000f01010 at file/buffer off 0x29d0
CNA_CONV_CON2 k_tile 0x0201500000f01010 at file/buffer off 0x32d0
```

The hit backtrace is the generic C++ stream write path:

```text
#0 std::ostream::write(char const*, long)
#1 librknnc.so+0x152ed48
#2 librknnc.so+0x154b7cc
```

Static disassembly shows `librknnc.so+0x152ebb0` is a generic file writer: it
opens the export path and then calls `std::ostream::write` with the supplied
buffer and size:

```asm
152ed38: ldr x1, [sp,#104]   ; buffer
152ed3c: mov x2, x26         ; size
152ed40: mov x0, x21         ; ostream
152ed44: bl  11a600          ; std::ostream::write
```

The caller at `librknnc.so+0x154b690` logs `Export RKNN model to ...`, allocates
a 0x190-byte export object, calls the buffer builder at `+0x1549c44`, and then
writes the serialized buffer on success:

```asm
154b780: bl  1549c44         ; build/fill serialized RKNN export object
154b7b4: ldr x3, [sp,#72]    ; export object
154b7bc: ldp x4,x2,[x3,#8]   ; base pointer, size
154b7c0: ldr x1,[x3,#40]     ; payload offset
154b7c4: add x1,x4,x1        ; buffer passed to writer
154b7c8: bl  152ebb0         ; write file
```

So the confirmed serialized-buffer contract is:

```text
writer_path = librknnc.so+0x154b690 -> +0x152ebb0 -> std::ostream::write
builder     = librknnc.so+0x1549c44
buffer      = export_obj[8] + export_obj[40]
size        = export_obj[16]
```

This does not yet identify the original regcmd producer. It does narrow the
next step: instrument `+0x1549c44` and the object fields it fills, then trace
where the qword-containing payload is copied into the export object. The
producer should be upstream of `+0x1549c44`; the final writer itself is only
serializing an already-built RKNN buffer.

`experimental/trace_librknnc_export_builder.gdb` now instruments that builder
boundary and the two important copy sites inside `+0x1549c44`. Remote run:

```text
/tmp/rknnc_export_builder_conv.log
```

Important hits for the same `160->320 20x20 3x3` probe:

```text
builder_entry       librknnc.so+0x1549c44
payload_copy_before librknnc.so+0x154aed0
payload_copy_after  librknnc.so+0x154aee8
wrapper_copy_before librknnc.so+0x154afe8
wrapper_copy_after  librknnc.so+0x154b03c
builder_return      librknnc.so+0x154b0d0
caller_after_build  librknnc.so+0x154b784
caller_before_write librknnc.so+0x154b7b4
```

The payload copy is the first confirmed current-export placement of the regcmd
bytes into the export object:

```asm
154aed0: ldr x3,[x0,#8]       ; export_obj base
154aedc: ldr x0,[x0,#40]      ; export_obj payload offset
154aee0: add x0,x3,x0         ; destination
154aee4: bl  11a720           ; memcpy(dst, staging, size)
```

A second run with an exact breakpoint at `+0x154aee4` confirmed the `memcpy`
arguments before the call:

```text
dst  x0 = export_obj base/payload, e.g. 0x555b821950
src  x1 = staging byte buffer,    e.g. 0x555b186440
size x2 = 0xe5900
```

The staging source already contains the current regcmd bytes at raw payload
offsets:

```text
payload_memcpy_src setup  off 0x2630
payload_memcpy_src k_half off 0x2990
payload_memcpy_src k_tile off 0x3290
```

Immediately after `+0x154aee4`, the export object has the regcmd qwords at raw
payload offsets:

```text
setup  off 0x2630
k_half off 0x2990
k_tile off 0x3290
```

Those are exactly 0x10 bytes before the parser's run starts and 0x40 bytes
before the final file qword offsets. The `0x10` gap is the per-run preamble
inside each command run. The later wrapper copy inserts a 0x40-byte object
header before the payload:

```asm
154b000: add x4,x7,x6         ; export_obj base + payload offset
154b004: add x3,x4,#0x40
154b00c: str q0,[x7,x6]       ; wrapper/header start
154b010: str x23,[x4,#16]
154b020: bl  11a720           ; memcpy wrapper payload part
154b024: str x19,[x0,x23]     ; payload length
154b034: add x0,x0,#0x8
154b038: bl  11a720           ; memcpy old payload after wrapper
```

After `+0x154b038`, the same qwords move to final exported file offsets:

```text
setup  off 0x2670
k_half off 0x29d0
k_tile off 0x32d0
```

The final export object at builder return and at `caller_before_write` is:

```text
export_obj[0x00] = vtable/object tag
export_obj[0x08] = base pointer
export_obj[0x10] = size = 941366
export_obj[0x18] = capacity = 0xe6000
export_obj[0x20] = end/used = 941366
export_obj[0x28] = payload offset = 0 for this final object
```

The stale qwords seen before `payload_copy_before` are leftovers near the end of
the already allocated export buffer (`off 0xe1530`, `0xe1890`, `0xe2190`).
They are not the current serialized placement. The current placement starts
only after the `+0x154aee4` copy from the staging byte vector.

This trace proves that `+0x1549c44` is a serializer/builder, not the regcmd
semantic producer. It receives an already assembled byte stream in its local
staging vector and copies it into the export object's payload. The next
productive compiler trace should move one step earlier from the staging vector:

```text
Trace the source pointer used by memcpy at +0x154aee4 (x1, size x2) backward to
the function that appended the command-run byte blocks. In the sampled run, the
source was 0x555b186440 and already contained the raw command payload before
the export-object copy.
```

`experimental/trace_librknnc_builder_appends.gdb` traces the append helpers
called by `+0x1549c44` before that payload copy. It filters helper calls whose
return address is inside the builder and scans the builder-local writer object.
Remote run:

```text
/tmp/rknnc_builder_appends_conv.log
```

The traced helper family is a byte-stream writer, not a high-level regcmd
planner:

```text
append/alignment helpers:
  +0x1537840 align4
  +0x1537e20 append indexed u32/tag
  +0x1561100 append byte/tag
  +0x15611f0 append u32
  +0x15612e0 append bytes
  +0x15615c0 append indexed u32/tag variant
  +0x1561fa0 append varint/size-like value
```

The writer object passed in `x22` during the builder has this observed layout:

```text
x22 + 0x30 : used byte count
x22 + 0x40 : current payload cursor
x22 + 0x48 : record/field-table cursor
x22 + 0x50 : record count
x22 + 0x54 : max tag/field id
x22 + 0x58 : base used count before current nested object
x22 + 0x60 : dirty/nested-object flag byte
x22 + 0x68 : small capacity/state
x22 + 0x70 : gate/empty flag byte
```

The staging allocation grows backward: as fields are appended, `used` increases
and the payload `cursor` decreases. The command qword absolute addresses remain
stable while their offset relative to the moving cursor changes. In the sampled
run:

```text
after early append at +0x1549d5c:
  used   = 0xe50c0
  cursor = 0x555b186a80
  setup qword visible at cursor+0x1df0

after final align at +0x154ae64:
  used   = 0xe58fc
  cursor = 0x555b186244
  setup qword visible at cursor+0x262c

after final append_u32 at +0x154ae80:
  used   = 0xe5900
  cursor = 0x555b186240
  setup qword visible at cursor+0x2630
```

The exact payload-copy call immediately after this uses the same cursor:

```text
+0x154aee4 memcpy:
  dst  = export object payload
  src  = x22.cursor = 0x555b186240
  size = x22.used   = 0xe5900

qwords in src:
  setup  off 0x2630
  k_half off 0x2990
  k_tile off 0x3290
```

So `+0x1549c44` builds a nested serialized payload around an already present
command-stream region, then copies `x22.cursor..x22.cursor+x22.used` to the
export object. The helper calls between `+0x1549d5c` and `+0x154ae80` mostly add
serialization fields, indexes, lengths, wrapper tags, and alignment. They do not
appear to synthesize individual Rocket qwords from register/value semantics.

Important negative/positive split:

- Positive: the final raw command payload is exactly `x22.cursor` after
  `+0x154ae80`, before the export-object copy.
- Negative: the append helpers traced here are still generic serializer helpers.
  Seeing qword bytes in their scan window does not prove those helpers produced
  the qwords; the qword bytes are already stable in the allocation while the
  cursor moves around them.

The next real producer target is therefore earlier than this export serializer:
find where the stable command-stream allocation bytes at absolute addresses like
`0x555b188870` are created before `RKNNModelExportPass`, likely during
`RKNNModelRegCmdbuildPass` or the model build object consumed by the export
builder. A targeted approach is to catch allocations/copies of the run-sized
byte blocks before export, or to trace references to the command payload pointer
passed into `+0x1549c44` rather than the serializer's final `x22` cursor.

This is the clearest boundary for the clean rewrite:

1. Add a descriptor type in `conv.py` with the exported fields above.
2. Refactor `make_conv2d_regs()` so it consumes descriptor-local dimensions and
   offsets instead of recomputing from only full-layer `p`.
3. Make the planner emit ordered descriptors. Descriptor order is part of the
   contract; do not sort by Y/K offsets.
4. Keep CPU im2col out of the clean path. The compiler-derived descriptors show
   direct hardware spatial and pointwise schedules for the tested shapes.

### Descriptor checker against current `conv.py`

`experimental/rknn_descriptor_plan.py` was added as a bridge from reverse data
to implementation. It imports the current `examples/kernel_6_18/conv.py`
helpers and compares an estimated descriptor sequence against exported RKNN
regcmd descriptors:

```sh
python3 experimental/rknn_descriptor_plan.py \
  mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1 \
  --compare-rknn /tmp/rknnc_mixed_models_local/mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1.rknn
```

This intentionally fails for the mixed spatial shape, and the failure is useful:

- Compiler emits 12 descriptors:
  `setup` x2, `k_half` x4, `k_tile` x6.
- Current `conv.py` planner enters its `spatial_im2col` path and predicts many
  tiny 2-output-row descriptors. Its first estimate is `input_h=4`,
  `output_h=2`; compiler ground truth is `input_h=23`, `output_h=21`.
- Current K estimate for that path starts from im2col/pointwise tile sizing and
  predicts many 48-channel `k_tile` windows; compiler ground truth for the
  mixed spatial path is `112,112,96`.

For the small spatial `32->128` model, the checker matches the first three
compiler rows (`setup`, `k_half`, `k_half`) exactly, then flags the missing
compiler `y_tile` rows. That confirms the descriptor/offest formulas are useful,
but the family-selection logic is not complete.

Implementation implication:

- Remove the conceptual `spatial_im2col` planning path from the clean RKNN-style
  planner. It is an older Python workaround, not compiler behavior.
- Planner output should be compiler-family descriptors first, not direct submit
  tiles. For large spatial conv, derive the two-level family schedule:
  full/setup, half-K metadata, and `k_tile` windows, each crossed with the
  compiler Y windows.
- Keep CPU im2col out of the clean path unless explicitly requested.

### Staging buffer origin (next trace target)

Confirmed flow before the export serializer:

```
RKNNModelRegCmdbuildPass (03:59:57.950)   <- runs before export
  | (populates model object internal command buffer)
RKNNModelExportPass   (03:59:57.950-976)
  | +0x154b690 "Export RKNN model to ..."
  |   allocates export object (0x190 bytes at 0x555b2143d0)
  |   calls builder +0x1549c44 at +0x154b780
  |     +0x1549c44 entry: x22 = x0 = compiler object (0x555b01c250)
  |       compiler+0x108: vector begin=0x555b30e070 end=0x555b30e080 (16 bytes)
  |       compiler+0x1f0: pointer to library static data (0x7fe30bb570)
  |     first append align4 at +0x1549d5c -> used=0xe50c0, cursor=0x555b186a80
  |       qword at cursor+0x1df0 = 0x555b188870 (already present)
  |     ... serialization metadata appended around stable qword bytes ...
  |     final append at +0x154ae80 -> used=0xe5900, cursor=0x555b186240
  |     memcpy(cursor, export_obj_payload, used) at +0x154aee4
  |     wrapper insertion -> final file offsets 0x2670/0x29d0/0x32d0
  |   +0x152ebb0 std::ostream::write
```

Key unresolved question: **Who allocated and filled the staging buffer** at
absolute addresses like `0x555b188870` before the export builder touched it?

The staging buffer:
- lives at cursor = `0x555b186240`, size = `0xe5900`
- is **not** the compiler object itself (distance ~1.48 MB)
- is **not** in the compiler+0x108 vector (only 16 bytes, begin=`0x555b30e070`)
- already contains the final regcmd qwords when the builder first examines its
  cursor (`used=0xe50c0` after first align4 call)

Static anchor points:
- Export builder at `+0x1549c44` is called from 5 call sites:
  `+0x1be4ac`, `+0x1be664`, `+0x1be814`, `+0x1c05fc`, `+0x154b780`
- The `+0x154b780` call is the RKNNModelExportPass path (traced above).
- The other 4 call sites (around `+0x1bxxxx`) may be other serialization
  contexts (model copy, subgraph serialization, etc.).
- `RKNNModelRegCmdbuildPass` runs at `+0x1c05fc` (near the other builder
  callers) -- this is likely the pass that populates the model's command buffer.

The next productive trace (`trace_librknnc_staging_source.gdb`):
1. Intercepts `memcpy` calls from within the builder (`+0x1549c44` range)
2. Filters for sizes >= 0x1000 (to catch the staging payload)
3. Scans source and destination for qword patterns
4. Shows backtrace to identify who produced the bytes

To run on remote probe:
```sh
ssh orangepi@192.168.192.36 "cd ~/npu/ops_rknn && \
  cp /tmp/rknnc_probe_conv.py . && \
  gdb -batch -x /tmp/trace_librknnc_staging_source.gdb \
    --args python3 rknnc_probe_conv.py /tmp/rknnc_staging_log" 2>&1 | \
  tee /tmp/rknnc_staging_source_conv.log
```

The probe script (`rknnc_probe_conv.py`) builds a conv 1x160x20x20 -> 320x18x18
3x3 model via rknn-toolkit2.

---

## 6-Tile Conv EMIT Trace Analysis (2026-05-22)

### Model Details

```
Model: conv2d_cc_b1_c3_h224_w224_oc32_wic3_k3x3_g1.rknn
Input:  (1, 3, 224, 224) int8
Output: (1, 32, 222, 222) int8
Kernel: 3x3, stride 1, no padding
Weight: 32 * 3 * 3 * 3 = 864 bytes = 864 int8 values
CBUF:   weight_bank=1, data_bank=11
Split:  BY_Y (Y-axis tiling) across 3 NPU cores
```

Captured via `rknn.gdb` + `dump.py` on remote with `LD_LIBRARY_PATH=/tmp`
(pointing to local `librknnrt_decomp.so` for offset-matched tracing).

### Overall Structure

- **1084 EMIT lines** total
- **21 sub-tile OP_ENABLE entries** (each = one DPU operation)
- **5 FENCE entries** (separate core groups)
- **4 core groups** (3 NPU cores, one core handles 2 spatial regions)

### Core Group Layout

| Group | Cores | Tiles | Input Heights | RESERVED_0 in CONV_CON2 | DCOMP_ADDR  | PC_REG_AMT RESERVED_0 |
|-------|-------|-------|---------------|--------------------------|-------------|----------------------|
| 0     | Core0 | 4     | 50,50,50,32   | 0                        | 0xffff8000  | 0                    |
| 1     | Core1 | 4     | 50,50,50,32   | 64                       | 0xffff8000  | 1                    |
| 2     | Core2 | 4     | 50,50,50,32   | 64                       | 0xffff8900  | 1                    |
| 3     | Core3 | 3     | 39,39,39      | 32                       | 0xffff8000  | 2                    |

Total input rows: (50+50+50+32) * 3 + (39*3) = 546 + 117 = 663? No — each core
processes a **different spatial region** of the same 224-row input. The 4 groups
are 4 separate core assignments for different Y-slices of the output.

Actually, looking at the FEATURE_DATA_ADDR within each group, input addresses
increment per tile, confirming Y-tiling within each core. Each core processes
ALL output channels (32 kernels) for its assigned output row range.

**Key insight**: The "4 groups" likely correspond to:
- Group 0 (Core0, 4 tiles): output rows 0-181 (full 182-row input region)
- Group 1 (Core1, 4 tiles): output rows 0-181 (same input region, different weight bank)
- Group 2 (Core2, 4 tiles): output rows 0-181 (same)
- Group 3 (Core3, 3 tiles): output rows 0-116 (shorter region)

Wait — this would mean redundant work. More likely interpretation:
**Each "group" is one sub-conv for a different input channel group or output channel
slice**, not spatial Y-slicing. But WEIGHT_SIZE2 shows 32 kernels in all tiles,
so all output channels are processed per tile.

The actual answer: **each group is a spatial Y-slice assigned to a different core**.
The height sums 50+50+50+32=182 input rows per core means each core processes
~74 output rows (with 3x3 kernel overlap). 3 cores * 74 = 222 output rows = correct.
Group 3 with height 39 handles the remaining partial output.

### Per-Tile Register Emission Sequence (36 registers per tile)

The following sequence repeats for each sub-tile, with the FIRST tile of each
core group lacking `WEIGHT_REUSE` and subsequent tiles having it:

```
; === CNA (Convolution Accelerator) Setup ===
CNA_CBUF_CON0          [TILE-VARYING: WEIGHT_REUSE(1) for tiles 2+; WEIGHT_BANK(1)|DATA_BANK(11)]
CNA_CONV_CON1           [CONSTANT: NONALIGN_DMA(1)|GROUP_LINE_OFF(1)|ARGB_IN(10)|PROC_PRECISION(2)|IN_PRECISION(2)]
DPU_S_POINTER           [CONSTANT: PP_MODE(1)|EXECUTER_PP_EN(1)|PP_EN(1)]
CNA_CONV_CON1           [CONSTANT: duplicate]
CNA_CONV_CON2           [TILE-VARYING: FEATURE_GRAINS(height); CORE-VARYING: RESERVED_0]
CNA_CONV_CON3           [CONSTANT: Y_STRIDE(1)|X_STRIDE(1)]
CNA_DATA_SIZE0          [TILE-VARYING: WIDTH(224), HEIGHT(50|32|39)]
CNA_DATA_SIZE1          [CONSTANT: CHANNEL_REAL(2)|CHANNEL(8)]
CNA_DATA_SIZE2          [CONSTANT: DATAOUT_WIDTH(222)]
CNA_DATA_SIZE3          [TILE-VARYING: DATAOUT_ATOMICS]
CNA_WEIGHT_SIZE0        [CONSTANT]
CNA_WEIGHT_SIZE1        [CONSTANT: BYTES_PER_KERNEL(144)]
CNA_WEIGHT_SIZE2        [CONSTANT: 3x3, 32 kernels]
CNA_CBUF_CON0           [duplicate of first CBUF_CON0]
CNA_CBUF_CON1           [TILE-VARYING: DATA_ENTRIES]
CNA_CVT_CON0            [CONSTANT: CVT_BYPASS(1)]
CNA_CVT_CON1-4          [CONSTANT: scales=1]
CNA_FEATURE_DATA_ADDR   [TILE-VARYING: input IOVA]
CNA_DMA_CON0            [CONSTANT: WEIGHT_BURST_LEN(15)|DATA_BURST_LEN(15)]
CNA_DMA_CON1            [CONSTANT: LINE_STRIDE(224)]
CNA_DMA_CON2            [CONSTANT: SURF_STRIDE(49952)]
CNA_FC_DATA_SIZE0       [TILE-VARYING: DMA_WIDTH(224), DMA_HEIGHT]
CNA_FC_DATA_SIZE1       [CONSTANT: DMA_CHANNEL(8)]
CNA_DCOMP_ADDR0         [CORE-VARYING: 0xffff8000 or 0xffff8900]
CNA_CVT_CON5            [CONSTANT: 0x00000fff]

; === CORE (DPU Core) Setup ===
CORE_MISC_CFG           [CONSTANT: PROC_PRECISION(2)]
CORE_DATAOUT_SIZE_0     [TILE-VARYING: HEIGHT, WIDTH]
CORE_DATAOUT_SIZE_1     [CONSTANT: CHANNEL(31)]

; === DPU (Post-Processing) Setup ===
DPU_FEATURE_MODE_CFG    [CONSTANT: BURST_LEN(15)|OUTPUT_MODE(2)]
DPU_DATA_FORMAT         [CONSTANT: OUT/IN/PROC_PRECISION(2)]
DPU_DST_BASE_ADDR       [TILE-VARYING: output IOVA]
DPU_DST_SURF_STRIDE     [CONSTANT: 49284]
DPU_DATA_CUBE_WIDTH     [CONSTANT: 221]
DPU_DATA_CUBE_HEIGHT    [TILE-VARYING: 47|29|36]
DPU_DATA_CUBE_CHANNEL   [CONSTANT: ORIG(31)|CHANNEL(31)]
DPU_BS_CFG              [CONSTANT: all bypass]
DPU_BS_OW_CFG           [CONSTANT]
DPU_WDMA_SIZE_0         [CONSTANT: CHANNEL(31)]
DPU_WDMA_SIZE_1         [TILE-VARYING: HEIGHT, WIDTH]
DPU_BN_CFG              [CONSTANT: all bypass]
DPU_EW_CFG              [CONSTANT: all bypass]
DPU_EW_CVT_SCALE_VALUE  [CONSTANT: 1]
DPU_OUT_CVT_SCALE       [CONSTANT: FP32TOFP16_EN(1)|SCALE(1)]
DPU_SURFACE_ADD         [CONSTANT: 98568]

; === PC (Program Control) ===
PC_BASE_ADDRESS         [TILE-VARYING: weight microcode addr]
PC_REGISTER_AMOUNTS     [TILE-VARYING: PC_DATA_AMOUNT; CORE-VARYING: RESERVED_0]
PC_OPERATION_ENABLE     [CONSTANT: RESERVED_0(6)|OP_EN(1)]
```

### Tile-Varying Fields Summary

| Field | Tile 1 (h=50) | Tile 2 (h=50) | Tile 3 (h=50) | Tile 4 (h=32) | Tile (h=39) |
|-------|---------------|---------------|---------------|---------------|-------------|
| DATA_SIZE0.HEIGHT | 50 | 50 | 50 | 32 | 39 |
| CONV_CON2.GRAINS | 50 | 50 | 50 | 32 | 39 |
| DATA_SIZE3.ATOMICS | 10656 | 10656 | 10656 | 6660 | 8214 |
| CBUF_CON1.ENTRIES | 11200 | 11200 | 11200 | 7168 | 8736 |
| CORE_DATAOUT.H | 47 | 47 | 47 | 29 | 36 |
| DPU_CUBE.H | 47 | 47 | 47 | 29 | 36 |
| WDMA_SIZE_1.H | 47 | 47 | 47 | 29 | 36 |

### Core-Varying Fields Summary

| Core | CONV_CON2.RESERVED_0 | DCOMP_ADDR | PC_REG_AMT.RESERVED_0 | Notes |
|------|---------------------|------------|----------------------|-------|
| 0    | 0                   | 0xffff8000 | 0                    | First core, no flag |
| 1    | 64                  | 0xffff8000 | 1                    | |
| 2    | 64                  | 0xffff8900 | 1                    | Different weight DMA |
| 3    | 32                  | 0xffff8000 | 2                    | Shorter tiles |

### CBUF Weight Reuse Pattern

- **First tile per core**: `CNA_CBUF_CON0 = WEIGHT_BANK(1) | DATA_BANK(11)` (no reuse)
- **Subsequent tiles**: `CNA_CBUF_CON0 = WEIGHT_REUSE(1) | WEIGHT_BANK(1) | DATA_BANK(11)`
- This means weights are loaded once per core and reused across Y-tiles within that core

### Address Progression (Core 0)

| Tile | FEATURE_DATA_ADDR | Delta | DPU_DST_BASE_ADDR | Delta |
|------|-------------------|-------|-------------------|-------|
| 1    | 0xffc62000        | —     | 0xff95f000        | —     |
| 2    | 0xffc71c00        | +0xFC00 | 0xff988a00      | +0x13A00 |
| 3    | 0xffc81800        | +0xFC00 | 0xff9b2400      | +0x29A00 |
| 4    | 0xffca1000        | +0x1F800 | 0xff9dbe00     | +0x29A00 |

Input stride: 0xFC00 = 64512 bytes (224 * 8ch * 1byte * 36 rows?)
Output stride: 0x13A00 = 80384 bytes (221 * 32ch * ... varies)

### DMA Patch Confirmation

The `fcn.002efd38` (indexed patcher) fires **115 times** and `fcn.002efce0` (pair-record
patcher) fires **231 times** on this 21-tile model. Each tile has multiple DMA address
fields that get patched: FEATURE_DATA_ADDR, DPU_DST_BASE_ADDR, DCOMP_ADDR, PC_BASE_ADDRESS.

### Remaining Unknowns After This Trace

1. **CONV_CON2.RESERVED_0 meaning**: 0/32/64 per core — possibly core index encoding or weight buffer selector
2. **PC_REGISTER_AMOUNTS.RESERVED_0**: 0/1/2 per core — likely core index for PC chain routing
3. **4 groups vs 3 cores**: RK3588 has 3 NPU cores but we see 4 groups; one core may be assigned 2 separate spatial regions
4. **Weight microcode (PC_BASE_ADDRESS)**: points to pre-compiled weight data; format unknown

---

## Pointwise OC-Split Conv EMIT Trace Analysis (2026-05-22)

### Model Details

```
Model: conv2d_cc_b1_c256_h14_w14_oc512_wic256_k1x1_g1.rknn
Input:  (1, 256, 14, 14) int8
Output: (1, 512, 14, 14) int8
Kernel: 1x1, stride 1, no padding (pointwise)
Weight: 512 * 256 * 1 * 1 = 131072 bytes = 128KB
CBUF:   weight_bank=8, data_bank=4
```

### Overall Structure

- **810 EMIT lines** total
- **6 OP_ENABLE entries** (6 sub-tiles)
- **5 FENCE entries** (5 core boundaries)

### Tile Layout

| Tile | Line | KERNELS | HEIGHT | GRAINS | ATOMICS | DATA_ENTRIES | CONV_CON2.R0 | DCOMP_ADDR | FEATURE_ADDR |
|------|------|---------|--------|--------|---------|-------------|-------------|------------|-------------|
| Init | —    | 512     | 14     | 10     | 196     | 112         | 0           | 0xfffbc000 | 0xfff59000  |
| 1    | 122  | 256     | 14     | 10     | 196     | 112         | 64          | 0xfffbc000 | 0xfff59000  |
| 2    | 228  | 256     | 14     | 10     | 196     | 112         | 64          | 0xfffdc000 | 0xfff59000  |
| 3    | 363  | 512     | 5      | 6      | 70      | 112         | 32          | 0xfffbc000 | 0xfff59000  |
| 4    | 498  | 512     | 5      | 6      | 70      | 112         | 32          | 0xfffbc000 | 0xfff59460  |
| 5    | 633  | 512     | 4      | 5      | 56      | 112         | 32          | 0xfffbc000 | 0xfff598c0  |

### Key Observations

1. **OC-Split (K-split) visible**: Init tile has KERNELS(512), but tiles 1-2 have KERNELS(256)
   — the first 2 tiles split the 512 OC into two 256-OC passes.
   - Init tile: CORE_DATAOUT_SIZE_1 = CHANNEL(511), WDMA = CHANNEL(511)
   - After OP_ENABLE, tiles 1-2: CHANNEL(255), WDMA = CHANNEL(255)

2. **Y-split on tiles 3-5**: heights 5,5,4 with FEATURE_ADDR incrementing by 0x460 and 0x460
   — these are spatial Y-tiles on different cores.

3. **No WEIGHT_REUSE in any tile**: all tiles have `CBUF_CON0 = WEIGHT_BANK(8)|DATA_BANK(4)`
   — because OC-split means different weights per tile, so no reuse possible.

4. **CBUF bank layout inverted**: weight_bank=8 (large, for 128KB weights), data_bank=4 (small, for 14x14x256 input)
   — opposite of the 3x3 conv which had weight_bank=1, data_bank=11.

5. **CONV_CON2.RESERVED_0 = 0/64/32**: same pattern as the 3x3 conv:
   - Init tile: RESERVED_0=0
   - Tiles 1-2: RESERVED_0=64
   - Tiles 3-5: RESERVED_0=32

6. **DCOMP_ADDR differs per OC group**: tiles 1 use 0xfffbc000, tile 2 uses 0xfffdc000
   — different weight decompression buffers for the two OC halves.

7. **DPU_SURFACE_ADD = 392**: different from 3x3 conv's 98568 — reflects much smaller output.

8. **PC_REGISTER_AMOUNTS = 0**: no PC_DATA_AMOUNT — pointwise conv uses no weight microcode
   (or weights are loaded directly via DCOMP path).

9. **DCOMP registers present**: DCOMP_CTRL(0), DCOMP_REGNUM(0), DCOMP_AMOUNT0-15(0) are all zero
   — decompression is disabled (no weight compression for this model).

### Comparison: 3x3 Y-Tile vs 1x1 OC-Split

| Aspect | 3x3 Y-Tile (c3→c32) | 1x1 OC-Split (c256→c512) |
|--------|---------------------|--------------------------|
| WEIGHT_BANK | 1 | 8 |
| DATA_BANK | 11 | 4 |
| WEIGHT_REUSE | Yes (tiles 2+) | No |
| KERNELS per tile | 32 (all OC) | 256 or 512 (split) |
| DATA_SIZE1.CHANNEL | 8 (padded from 3) | 256 |
| DATA_ENTRIES | 7168-11200 | 112 (constant) |
| PC_DATA_AMOUNT | 29-55 | 0 |
| SURF_STRIDE | 49952 | N/A (no DMA_CON2) |
| DPU_SURF_STRIDE | 49284 | 196 |

The 1x1 conv has much simpler CBUF requirements (small spatial, large channel) and
doesn't need PC microcode for weight loading — the weights go through the DCOMP path
(or direct DMA) instead.

## Direct-Spatial Rocket Mapping Status (2026-05-23)

The exact `sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1` export is still useful
as the direct-spatial reference, but it is not yet a local Rocket submit recipe.
The captured RKNN runtime submit has one submit, global `task_number=6`, and
repeated `subcore_task[0..2]={task_start=0, task_number=2}`. The task GEM backing
table has 17 task-like records, but those records are backing/cached entries, not
all active work.

`experimental/compare_conv_direct_regs.py --verify-rocket-task-mapping-proof`
now encodes the local Rocket proof gate directly. It intentionally fails for the
current logs because the observed interpretations are ambiguous and incomplete:
the global six-record interpretation covers `554/1280` qwords, the repeated or
unique subcore interpretation covers `216/1280` qwords, and multiple candidate
task boundaries cut through compiler-stitched row bodies. This keeps the
direct-spatial hardware path guarded while allowing `conv.py` cleanup to proceed
inside the shared descriptor planner/executor structure.

The passing local `pc_root6` probe remains explicitly separate from that
conservative proof gate. `conv.py` now centralizes the runtime `pc_root6`
stitched stream plus raw Rocket spans in
`direct_spatial_pc_root6_stream_and_spans()`, and
`--verify-runtime-pc-root6-stream` consumes the same helper before any guarded
probe. That helper also enforces the exact descriptor-schedule guard, so both
runtime and verifier callers reject unsupported mirrored sweep schedules before
building a raw-span stream. The current verified target is still only the exact
`160x40 -> 320`, `3x3`, groups-1 schedule under
`RK3588_CONV_DIRECT_SPATIAL=1`,
`RK3588_CONV_DIRECT_SPATIAL_UNSAFE=1`; the exact schedule now selects `pc_root6`
automatically without requiring a third env var. The latest local run passed
with `max_diff=0.0625`, followed by a passing `simple_add.py` smoke test.

The runtime planner structure has also been cleaned up around that boundary.
Direct spatial is now a normal classified `FAMILY_DIRECT_SPATIAL` producer in
`DESCRIPTOR_PRODUCERS`. `classify_conv_plan()` selects it only when both unsafe
environment gates are set and the observed descriptor schedule is available;
otherwise the regular fallback/classified families are selected. This matches
the current safety model while keeping one planner shape: the RKNN-style direct
path is still a guarded Phase D hardware entry, but it participates in the same
producer dispatch and shared executor path as the other descriptor families.
The classified plan carries the observed direct-spatial descriptor schedule into
the direct producer, so materialization no longer repeats the schedule lookup
after classification.
The temporary `spatial_im2col_fallback` descriptor producer no longer calls the
observed direct-spatial planner just to attach fallback debug metadata, keeping
the CPU-side workaround isolated from the RKNN-style descriptor research path.
Normal and fallback plan dictionaries likewise no longer carry an empty
`direct_spatial_descs` field; that schedule is present only for
`FAMILY_DIRECT_SPATIAL`.
Descriptor producer dispatch now passes a single build-context dictionary to
`producer(plan, ctx)`, so each family wrapper reads only the inputs it needs
instead of sharing a long positional signature.
That context is now a slotted `ConvBuildContext` object, keeping the shorter
dispatch shape without repeated string-key field lookups.
`ConvBuildContext` is now implemented as a frozen slotted dataclass. This is a
planner-code cleanup only; direct-spatial descriptor materialization, task
policy selection, and submit arguments are unchanged.
The executor result path now uses `_read_unpack_output()` before choosing tiled
writeback or direct-spatial full-output writeback. This is readback plumbing
cleanup only; emitted registers and submit semantics are unchanged.
The standalone shape suite now has a `conv_shape()` helper and generates the
repeated `1x3_{H}x{H}_k1` coverage entries for `H=54..72`; selected shape names
and coverage are unchanged.
The conv1d-as-conv2d coverage block now uses `conv1d_shape()` and an explicit
name/parameter spec list, preserving the previous selected shape names while
removing repeated dictionary boilerplate.
Grouped output-channel variants now use `grouped_spatial_shape()` and generated
`(in_c, groups, out_channels)` coverage. The selected shape names and ordering
are unchanged.
The repeated `test_ops.py` small-kernel coverage now uses
`test_ops_conv2d_shape()` for generated dense `cin=3` and explicit `cin=1`
shape specs. The selected shape names and ordering are unchanged.
Small pvalid pointwise clusters now use `pvalid_shape()` for repeated `H=3` and
`H=2` entries. The selected shape names and ordering are unchanged.
