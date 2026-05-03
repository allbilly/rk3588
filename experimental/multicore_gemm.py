import argparse
import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from rknpu_common import (
    RKNPU_JOB_PC,
    RKNPU_JOB_PINGPONG,
    RKNPU_MEM_KERNEL_MAPPING,
    RKNPU_MEM_NON_CACHEABLE,
    mem_allocate,
    print_submit_layout,
    reset_npu,
    struct_rknpu_task,
    submit,
)


MAX_TASKS = 3
REG_BLOCK_QWORDS = 64


def align_up(x, align):
    return ((x + align - 1) // align) * align


def parse_core_ranges(spec, task_count):
    if spec == "split3":
        return {0: (0, 1), 1: (1, 1), 2: (2, 1)}
    if spec == "core0":
        return {0: (0, task_count)}
    ranges = {}
    for item in spec.split(","):
        core_s, start_s, count_s = item.split(":")
        ranges[int(core_s, 0)] = (int(start_s, 0), int(count_s, 0))
    return ranges


def assigned_tasks(core_ranges):
    assigned = set()
    for start, count in core_ranges.values():
        assigned.update(range(start, start + count))
    return assigned


def validate_core_ranges(core_ranges, task_count):
    assigned = assigned_tasks(core_ranges)
    invalid = sorted(task for task in assigned if task < 0 or task >= task_count)
    if invalid:
        raise ValueError(f"core range references missing task indices {invalid}; task_count={task_count}")


def build_tasks(tasks, regcmd, regcmd_dma, regcmd_mode, task_regs):
    for task_idx, regs in enumerate(task_regs):
        block_qword = task_idx * REG_BLOCK_QWORDS
        for i in range(REG_BLOCK_QWORDS):
            regcmd[block_qword + i] = 0
        for i, value in enumerate(regs):
            regcmd[block_qword + i] = value

        tasks[task_idx].flags = 0
        tasks[task_idx].op_idx = 4
        tasks[task_idx].enable_mask = 0x18
        tasks[task_idx].int_mask = 0x300
        tasks[task_idx].int_clear = 0x1ffff
        tasks[task_idx].int_status = 0
        tasks[task_idx].regcfg_amount = len(regs)
        if regcmd_mode == "offset":
            tasks[task_idx].regcfg_offset = block_qword * ctypes.sizeof(ctypes.c_uint64)
            tasks[task_idx].regcmd_addr = regcmd_dma
        else:
            tasks[task_idx].regcfg_offset = 0
            tasks[task_idx].regcmd_addr = regcmd_dma + block_qword * ctypes.sizeof(ctypes.c_uint64)


def copy_task(dst, src):
    dst.flags = src.flags
    dst.op_idx = src.op_idx
    dst.enable_mask = src.enable_mask
    dst.int_mask = src.int_mask
    dst.int_clear = src.int_clear
    dst.int_status = src.int_status
    dst.regcfg_amount = src.regcfg_amount
    dst.regcfg_offset = src.regcfg_offset
    dst.regcmd_addr = src.regcmd_addr


def run(args):
    import gemm

    if args.tile_n:
        if args.n % args.tiles:
            raise ValueError("--n must be divisible by --tiles for --tile-n")
        task_count = args.tiles
        if args.execution == "sequential-core0":
            args.separate_submits = True
            args.core_ranges = "core0"
            args.core_mask = 0x1
            args.regcmd_mode = "absolute"
        elif args.execution == "unsafe-split3":
            if not args.allow_unsafe_submit:
                raise RuntimeError("--execution unsafe-split3 requires explicit --allow-unsafe-submit")
            if args.tiles != 3:
                raise ValueError("--execution unsafe-split3 requires --tiles 3")
            args.separate_submits = False
            args.core_ranges = "split3"
            args.core_mask = 0x7
            args.subcore_layout = "rk3588-tricore-tail"
            args.regcmd_mode = "absolute"
        else:
            raise ValueError(f"unknown execution mode: {args.execution}")
    else:
        task_count = args.task_count
    if task_count < 1 or task_count > MAX_TASKS:
        raise ValueError(f"--task-count must be 1..{MAX_TASKS}")

    core_ranges_for_guard = parse_core_ranges(args.core_ranges, task_count)
    validate_core_ranges(core_ranges_for_guard, task_count)
    uses_nonzero_core = any(core != 0 and count for core, (_, count) in core_ranges_for_guard.items())
    uses_multi_range = sum(1 for _, count in core_ranges_for_guard.values() if count) > 1
    if (uses_nonzero_core or uses_multi_range) and not args.allow_unsafe_submit:
        raise RuntimeError(
            "Unsafe raw multicore GEMM submit refused. Official RKNN multicore requires compiler/runtime "
            "multicore metadata; this raw ioctl path has locked up this kernel. Re-run with "
            "--allow-unsafe-submit only with physical reset access."
        )

    fd = gemm.fd
    m, n_total, k = args.m, args.n, args.k
    n = n_total // task_count if args.tile_n else n_total
    align_in, align_out, _, pad_k = gemm._gemm_layout(m, n, k)
    input_elems = align_in * m
    weight_elems = align_in * align_out
    output_bytes = max(256, ((m - 1) * align_out + n) * gemm.FP32_BYTES)
    input_stride = align_up(input_elems * np.dtype(np.float16).itemsize, 4096)
    weight_stride = align_up(weight_elems * np.dtype(np.float16).itemsize, 4096)
    output_stride = align_up(output_bytes, 4096)

    task_map, task_mc = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE, args.verbose)
    regcmd_map, regcmd_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE, args.verbose)
    input_map, input_mc = mem_allocate(fd, input_stride * task_count, RKNPU_MEM_NON_CACHEABLE, args.verbose)
    weight_map, weight_mc = mem_allocate(fd, weight_stride * task_count, RKNPU_MEM_NON_CACHEABLE, args.verbose)
    output_map, output_mc = mem_allocate(fd, output_stride * task_count, RKNPU_MEM_NON_CACHEABLE, args.verbose)

    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

    pack_input = gemm.get_input_packer(m, n, k, align_in)
    pack_weight = gemm.get_weight_packer(m, n, k, align_in)
    if pad_k:
        pack_input = gemm.pack_input_row_major
    decode_output = gemm.get_output_decoder(m, n, k, align_out)

    rng = np.random.default_rng(args.seed)
    expected = []
    if args.tile_n:
        a_full = rng.normal(size=(m, k)).astype(np.float16)
        b_full = rng.normal(size=(k, n_total)).astype(np.float16)
    for task_idx in range(task_count):
        if args.tile_n:
            a = a_full
            b = b_full[:, task_idx * n:(task_idx + 1) * n]
        else:
            a = rng.normal(size=(m, k)).astype(np.float16)
            b = rng.normal(size=(k, n)).astype(np.float16)
        in_pack = np.zeros(input_elems, dtype=np.float16)
        wt_pack = np.zeros(weight_elems, dtype=np.float16)
        pack_input(m, n, k, a, in_pack, align_in)
        pack_weight(m, n, k, b, wt_pack, align_in, align_out)

        in_view = np.frombuffer(input_map, dtype=np.float16, count=input_elems, offset=task_idx * input_stride)
        wt_view = np.frombuffer(weight_map, dtype=np.float16, count=weight_elems, offset=task_idx * weight_stride)
        out_view = np.frombuffer(output_map, dtype=np.float32, count=output_stride // gemm.FP32_BYTES, offset=task_idx * output_stride)
        in_view[:] = in_pack
        wt_view[:] = wt_pack
        out_view[:] = np.nan
        expected.append((a @ b).astype(np.float32))

    task_regs = []
    for task_idx in range(task_count):
        task_regs.append(gemm.make_gemm_regs(
            m,
            n,
            k,
            input_mc.dma_addr + task_idx * input_stride,
            weight_mc.dma_addr + task_idx * weight_stride,
            output_mc.dma_addr + task_idx * output_stride,
        ))
    build_tasks(tasks, regcmd, regcmd_mc.dma_addr, args.regcmd_mode, task_regs)

    for task_idx in range(task_count):
        print(
            f"task[{task_idx}] GEMM {m}x{n}x{k} regcfg_amount={tasks[task_idx].regcfg_amount} "
            f"regcfg_offset={tasks[task_idx].regcfg_offset} regcmd_addr=0x{tasks[task_idx].regcmd_addr:x}"
        )

    core_ranges = parse_core_ranges(args.core_ranges, task_count)
    active_tasks = assigned_tasks(core_ranges)
    flags = args.flags if args.flags is not None else RKNPU_JOB_PC | (0 if args.no_pingpong else RKNPU_JOB_PINGPONG)
    reset_npu(fd)
    start = time.perf_counter()
    submit_results = []
    if args.separate_submits:
        for task_idx in range(task_count):
            copy_task(tasks[0], tasks[task_idx])
            ret, submit_struct = submit(
                fd,
                task_mc.obj_addr,
                1,
                core_ranges={0: (0, 1)},
                core_mask=0x1,
                flags=flags,
                timeout=args.timeout,
                subcore_layout=args.subcore_layout,
            )
            submit_results.append((ret, submit_struct))
            print(f"sequential submit task[{task_idx}] core=0 ret={ret}")
            print_submit_layout(submit_struct)
    else:
        ret, submit_struct = submit(
            fd,
            task_mc.obj_addr,
            task_count,
            core_ranges=core_ranges,
            core_mask=args.core_mask,
            flags=flags,
            timeout=args.timeout,
            subcore_layout=args.subcore_layout,
        )
        submit_results.append((ret, submit_struct))
        print_submit_layout(submit_struct)
    elapsed = time.perf_counter() - start
    print(f"execution={args.execution if args.tile_n else 'raw'} submit ret={[ret for ret, _ in submit_results]} elapsed_ms={elapsed * 1000:.3f}")

    all_pass = all(ret == 0 for ret, _ in submit_results)
    outputs = []
    for task_idx in range(task_count):
        raw = np.frombuffer(
            output_map,
            dtype=np.float32,
            count=output_bytes // gemm.FP32_BYTES,
            offset=task_idx * output_stride,
        ).copy()
        out = decode_output(m, n, k, raw, align_out)
        outputs.append(out)
        sentinel_overwritten = not np.all(np.isnan(raw))
        if task_idx not in active_tasks:
            kept_sentinel = not sentinel_overwritten
            print(f"GEMM task[{task_idx}] {'SKIP' if kept_sentinel else 'FAIL'} unassigned sentinel_kept={kept_sentinel}")
            all_pass &= kept_sentinel
            continue
        ok = sentinel_overwritten and np.allclose(out, expected[task_idx], atol=args.atol, equal_nan=False)
        md = float(np.nanmax(np.abs(out - expected[task_idx]))) if out.size else 0.0
        print(f"GEMM task[{task_idx}] {'PASS' if ok else 'FAIL'} sentinel={sentinel_overwritten} max_diff={md:.4f}")
        all_pass &= ok
    if args.tile_n and all_pass:
        got_full = np.concatenate(outputs, axis=1)
        expected_full = a_full @ b_full
        ok_full = np.allclose(got_full, expected_full, atol=args.atol, equal_nan=False)
        md_full = float(np.nanmax(np.abs(got_full - expected_full))) if got_full.size else 0.0
        print(f"GEMM tiled_output {'PASS' if ok_full else 'FAIL'} max_diff={md_full:.4f}")
        all_pass &= ok_full
    return 0 if all_pass else 1


def main():
    parser = argparse.ArgumentParser(description="Submit independent GEMM tasks across RK3588 NPU cores. Safe default is one core-0 task.")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--tile-n", action="store_true", help="Treat --n as total N and split the GEMM across --tiles N-axis tiles.")
    parser.add_argument("--tiles", type=int, default=3)
    parser.add_argument(
        "--execution",
        choices=("sequential-core0", "unsafe-split3"),
        default="sequential-core0",
        help="Execution mode for --tile-n.",
    )
    parser.add_argument("--core-ranges", default="core0", help="'split3', 'core0', or core:start:count comma list. split3 is risky on this kernel.")
    parser.add_argument("--core-mask", type=lambda x: int(x, 0), default=0x1)
    parser.add_argument("--regcmd-mode", choices=("offset", "absolute"), default="offset")
    parser.add_argument("--separate-submits", action="store_true", help="Run each GEMM task as a sequential core-0 submit.")
    parser.add_argument("--flags", type=lambda x: int(x, 0), default=None)
    parser.add_argument("--no-pingpong", action="store_true")
    parser.add_argument("--allow-unsafe-submit", action="store_true", help="Allow raw nonzero-core or multi-range submits that may hard-lock the kernel.")
    parser.add_argument(
        "--subcore-layout",
        choices=("direct", "rk3588-tricore-tail"),
        default="direct",
        help="Map core ranges to subcore_task[]. allbilly/rknpu_driver uses indices 2..4 for 3-core RK3588 jobs.",
    )
    parser.add_argument("--timeout", type=int, default=10000)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    gemm_mod = None
    try:
        raise SystemExit(run(args))
    finally:
        gemm_mod = sys.modules.get("gemm")
        if gemm_mod is not None and getattr(gemm_mod, "fd", None) is not None:
            os.close(gemm_mod.fd)


if __name__ == "__main__":
    main()
