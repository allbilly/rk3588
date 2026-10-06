# Mainline 6.18 model runtime work

INTENT: code supports vendor RKNPU only; the task expects GPT-2 and openpilot on mainline Rocket; the examples document Rocket BO synchronization and command counts including PC trailers.

- [x] Inspect both runners, mainline examples, experimental notes and upstream v6.18 UAPI/job/GEM implementations.
- [x] Prove this board's NPU works with `python3 examples/simple_add.py`.
- [x] Add driver discovery, Rocket BO ownership and blocking model submissions.
- [x] Adapt decoded PC trailers and active task ranges without changing tensor arithmetic or coefficients.
- [x] Expose driver/device selection in both model CLIs and document usage.
- [x] Test ABI, schedules, ownership, errors and all shipped base-model programs.
- [x] Run all three openpilot fixtures and GPT-2 generation on the vendor NPU.
- [ ] Run all three openpilot fixtures and GPT-2 generation on mainline Rocket hardware.

The current board runs `6.1.99-rockchip-rk3588` with the vendor RKNPU driver.
It has no `/dev/accel` or Rocket-bound DRM device. A mainline test host has been
requested; mainline numerical verification remains pending until one is available.

References:

- `examples/kernel_6_18/README.md` and `simple_add.py`
- `experimental/kernel_6_18/rocket_runtime.py` and `problem.md`
- `ref/nvdla/REGISTER_MAP.md` and `ARCHITECTURE.md`
- https://github.com/torvalds/linux/blob/v6.18/include/uapi/drm/rocket_accel.h
- https://github.com/torvalds/linux/blob/v6.18/drivers/accel/rocket/rocket_job.c
- https://github.com/torvalds/linux/blob/v6.18/drivers/accel/rocket/rocket_gem.c

The requested DeepWiki tool is not installed in this session. On continuation,
the public MCP endpoint was used directly for `ask_wiki_question` on
`torvalds/linux`, `chaotic-cx/mesa-mirror` and `nvdla/hw`. Its response was checked
against the fetched upstream v6.18 kernel and Mesa source.

## Observed checks (2026-10-06)

- `python3 -B examples/simple_add.py`: NPU ADD passed. The first attempt with
  `-S` failed to import the example's NumPy dependency; the normal interpreter
  passed without any hardware error.
- `python3 -S -B openpilot/test_rocket.py`: tests reconstruct all seven shipped
  base programs through the real Model initializer with sparse test BOs. They
  verify command bodies, trailer termination, active ranges, exact UAPI layouts,
  BO ownership/deadlines, cleanup and submission failures. No neural arithmetic
  is simulated or offloaded. GPT-2's other layers share these base geometries.
  All nine tests passed.
- `python3 -S -B openpilot/test_tensor_io.py`: all five byte-layout tests passed.
- `python3 -S -B openpilot/infer.py MODEL --verify` for navigation,
  dmonitoring and supercombo: all 420, 84 and 6768 outputs respectively match
  the selected caches' RKNN fixtures exactly (`max_abs_vs_rknn = 0`).
- Persistent GPT-2 decoder, resets between the three prompts in
  `gpt2/validation.json`: five reference tokens match for `Hello`,
  `The quick brown fox`, and `The capital of France is`.
- `python3 -S -B gpt2/generate.py --driver auto --device /dev/dri/card1
  --prompt Hello --tokens 5 --json`: passed, with the same five reference tokens.
- `python3 -S -B openpilot/verify_models.py --driver rknpu`: passed all three
  openpilot fixtures twice in persistent models and all three GPT-2 prompts in
  one persistent decoder. The report identifies kernel 6.1.99 and `/dev/dri/card1`.
- Existing comments in the three edited runtime Python files are unchanged.
- `--driver rocket` on this board fails during discovery, before allocation or
  submission: `No rocket NPU device found in /dev/accel or /dev/dri`.

Mainline hardware completion requires all three `openpilot/infer.py MODEL
--driver rocket --verify` commands and the three GPT-2 five-token comparisons
to pass on a Rocket-bound RK3588. Software ABI/schedule tests and vendor runs
do not prove that result. No kernel/boot configuration was changed.

The complete reproducible hardware command is:

```sh
python3 -S openpilot/verify_models.py --driver rocket --report /tmp/models-rocket.json
```

Both caches must already be prepared. A successful report records the actual
kernel, device, driver, two exact fixture comparisons per openpilot model and
the GPT-2 reference token comparisons. This work has not produced a successful
Rocket hardware report yet.

## Continuation audit

INTENT: Rocket accepts any captured unit mask; stock Linux 6.18 completes only DPU tasks; submitted tasks must include DPU completion.

Stock v6.18 `rocket_job_hw_submit` arms only DPU completion (`0x300`), and
`rocket_job_irq_handler` ignores PPU completion (`0xc00`). The adapter now
rejects standalone and mixed PPU tasks before submitting to that driver.
Every active task in the seven shipped base programs uses masks `0x0d`,
`0x18` or `0x1d`; navigation's six `0x60` descriptors (indexes 155, 159, 161,
226, 230, 232) belong to unused core ranges. No kernel pooling patch is
required for these active schedules.

`python3 -S -B openpilot/test_rocket.py` now passes all ten tests, including
active DPU selection with an unused PPU descriptor and rejection of standalone
or mixed PPU submissions. A native C program compiled against the upstream
v6.18 `rocket_accel.h` independently matches all seven ctypes struct sizes and
field offsets and all five ioctl numbers. The temporary C source and executable
were removed automatically after that comparison.

Many captured tasks are partial register updates (for example, 20 of the 91
active navigation tasks contain fewer than 50 configuration words). Rocket
reinitializes CNA/CORE `S_POINTER` before every task, while these captures came
from vendor PC chains. DeepWiki highlighted possible producer/consumer bank
state differences; it did not prove a failure or establish a valid rewrite.
The hardware comparison must check this explicitly. No speculative register
expansion or tensor arithmetic fallback has been introduced.

The continuation's combined hardware command failed before NPU allocation:
`No rocket NPU device found in /dev/accel or /dev/dri`. The running kernel still
is 6.1.99, only that module tree is installed, and `/sys/class/drm/card1` remains
bound to `RKNPU`. No NPU inference process is running or awaiting observation.

DeepWiki query:
https://deepwiki.com/search/we-are-porting-captured-rk3588_aaf1b891-d54f-49ba-8391-436f7ae5d3de

## Cleanup fixes (2026-10-07)

INTENT: code leaks a new BO or stays busy after rejection; the review expects safe cleanup; driver docs require completion before freeing buffers.

A rejected `SUBMIT` ioctl now clears the busy flag, allowing context cleanup
and preserving the original error. Linux v6.18's returned submit errors occur
before queueing; the one-job adapter does not clear the guard for Python
interruptions or failed asynchronous completion waits. A newly allocated BO
whose initial `PREP_BO` fails now closes its mapping and GEM handle before
the constructor raises, while it is still absent from `device.buffers`.

The new rejection/context-cleanup and initial-sync tests both reproduced the
defects before the fix. All 13 Rocket tests and five tensor-layout tests now
pass, including interruption and failed-fence guards. The combined vendor NPU
check also passes: each openpilot fixture matches exactly on both repeats,
and all three GPT-2 reference continuations match. Syntax, existing runtime
comments and whitespace checks pass. Mainline hardware verification remains
pending; this board still runs vendor 6.1.99 without a Rocket-bound device.

TWINS: searched busy assignments and initial buffer sync calls - found 0 other matching Rocket defects.

The vendor busy flag already has a blocking-ioctl `finally` reset. The cleanup
reasoning was checked against pinned Linux v6.18 `rocket_job.c` after asking
DeepWiki; its wider error list included newer code, so it was not used as the
authority for v6.18 error propagation.

DeepWiki query:
https://deepwiki.com/search/for-the-exact-linux-v618-drive_c5a384ab-ccac-47a5-bd81-ab87c1eea065
