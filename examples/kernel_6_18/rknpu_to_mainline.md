# Porting examples/*.py to mainline rocket driver

## Goal

`examples/*.py` targets the downstream `rknpu` DRM driver (`/dev/dri/card1`,
`DRM_IOCTL_RKNPU_*`).  Porting to `experimental/mainline6_18/` makes the same
register logic work on the upstream `rocket` DRM driver (`/dev/accel/accel*` or
`/dev/dri/renderD*`, `DRM_IOCTL_ROCKET_*`).

The core register programming and data packing logic stays **exactly the same**.
Only the submission plumbing changes.

## Changes needed (reference: gemm.py)

1.  **Import rocket_runtime** — add at the top:

    ```python
    import rocket_runtime as rt
    ```

2.  **Remove legacy structs & IOCTLs** — delete these blocks entirely:
    - `class rknpu_mem_create`
    - `class rknpu_mem_map`
    - `class rknpu_action`
    - `class rknpu_subcore_task`
    - `class rknpu_submit`
    - `def _IOWR`
    - `DRM_IOCTL_RKNPU_*` constants

    Only keep `class struct_rknpu_task` (needed by rocket_runtime to build
    `drm_rocket_task`).

3.  **Open device** — replace hardcoded path:

    ```python
    # old
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    # new
    fd = rt.open_rocket_device()
    ```

4.  **mem_allocate** — replace body to delegate:

    ```python
    def mem_allocate(fd, size, flags=0):
        return rt.mem_allocate(fd, size, flags)
    ```

5.  **npu_reset** — replace body (no-op for rocket):

    ```python
    def npu_reset(fd):
        return rt.reset_npu(fd)
    ```

6.  **npu_submit** — replace legacy IOCTL with rocket submit:

    ```python
    def npu_submit(task_count=1):
        npu_reset(fd)
        for bo in (regcmd_mem_create, input_mem_create, weight_mem_create, output_mem_create):
            rt.fini_bo(fd, bo)
        ret = rt.submit(fd, npu_tasks, task_count,
            in_bos=[regcmd_mem_create, input_mem_create, weight_mem_create],
            out_bos=[output_mem_create])
        rt.prep_bo(fd, output_mem_create)
        return ret
    ```

    Key differences from rknpu version:
    - No `task_obj_addr` parameter (rocket reads from `npu_tasks` array directly)
    - Buffer sync via `rt.fini_bo` / `rt.prep_bo` before/after submit
    - Pass `npu_tasks` pointer and `task_count` to `rt.submit`

7.  **regcfg_amount must include PC chain tail** — in `write_regs_to_npu_task`:

    ```python
    # rknpu: HW fetches past regcfg_amount to find PC chain tail
    npu_tasks[idx].regcfg_amount = len(regs)

    # rocket: HW fetches exactly regcfg_amount regs, so tail must be counted
    npu_tasks[idx].regcfg_amount = len(regs) + len(tails)
    ```

    The rknpu driver's PC engine reads past `regcfg_amount` to find inline PC
    chain instructions.  The rocket driver's `drm_rocket_task.regcmd_count`
    tells the HW exactly how many 64-bit regs to fetch — it does **not** read
    past the count.  Without including the tail, `OPERATION_ENABLE` is never
    executed and the NPU does nothing.

8.  **Remove unused flags** — the rocket driver doesn't use `RKNPU_JOB_PC`,
    `RKNPU_JOB_BLOCK`, `RKNPU_JOB_PINGPONG`.  Remove them or leave as dead
    code (they won't be referenced).

All core logic (`make_*_regs`, pack/unpack helpers, `run_*`, test shapes)
remains untouched.

## Verification checklist

- [ ] `make_gemm_regs` / `make_conv2d_regs` unchanged
- [ ] Weight/input packing functions unchanged
- [ ] Output decode functions unchanged
- [ ] `write_regs_to_npu_task` keeps the same PC chain tail structure
- [ ] `regcfg_amount` includes `len(tails)`
- [ ] `fd` uses `rt.open_rocket_device()`
- [ ] `npu_submit` syncs BOs, calls `rt.submit`, syncs output back
- [ ] Test shapes unchanged from original `examples/*.py`
