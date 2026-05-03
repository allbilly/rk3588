import argparse
import ctypes
import os
import time

import numpy as np

from rknpu_common import (
    RKNPU_JOB_PC,
    RKNPU_JOB_BLOCK,
    RKNPU_JOB_PINGPONG,
    RKNPU_MEM_KERNEL_MAPPING,
    RKNPU_MEM_NON_CACHEABLE,
    mem_allocate,
    open_device,
    print_submit_layout,
    reset_npu,
    struct_rknpu_task,
    submit,
)


class reg:
    TARGET_DPU = 0x1001
    TARGET_RDMA = 0x2001
    TARGET_PC = 0x0081
    TARGET_PC_REG = 0x0101

    OPERATION_ENABLE = 0x0008
    BASE_ADDRESS = 0x0010
    REGISTER_AMOUNTS = 0x0014
    S_POINTER = 0x4004
    FEATURE_MODE_CFG = 0x400c
    DATA_FORMAT = 0x4010
    DST_BASE_ADDR = 0x4020
    DATA_CUBE_WIDTH = 0x4030
    DATA_CUBE_HEIGHT = 0x4034
    DATA_CUBE_CHANNEL = 0x403c
    EW_CFG = 0x4070
    OUT_CVT_SCALE = 0x4084
    RDMA_DATA_CUBE_WIDTH = 0x500c
    RDMA_DATA_CUBE_HEIGHT = 0x5010
    RDMA_DATA_CUBE_CHANNEL = 0x5014
    RDMA_ERDMA_CFG = 0x5034
    RDMA_SRC_BASE_ADDR = 0x5018
    RDMA_EW_BASE_ADDR = 0x5038
    RDMA_FEATURE_MODE_CFG = 0x5044
    RDMA_S_POINTER = 0x5004


_EW_BASE = 0x108002c0
OPS = {
    "ADD": (_EW_BASE | (2 << 16), lambda a, b: a + b, {}),
    "MUL": (_EW_BASE | (1 << 2) | (1 << 8), lambda a, b: a * b, {}),
    "SUB": (_EW_BASE | (4 << 16), lambda a, b: a - b, {}),
    "MAX": (_EW_BASE, lambda a, b: np.maximum(a, b), {}),
    "NEG": (_EW_BASE | (1 << 2) | (1 << 8), lambda a, b: -a, {"neg_op": True}),
    "FDIV": (_EW_BASE | (3 << 16) | (1 << 8), lambda a, b: a / b, {"fdiv_op": True}),
}


REG_BLOCK_QWORDS = 32
MAX_TASKS = 3


def _regcmd(target, addr, value):
    return (target << 48) | ((value & 0xFFFFFFFF) << 16) | addr


def _pc_base_address_value(regcmd_dma):
    # rknnops.h patches PC_BASE_ADDRESS with PC_SOURCE_ADDR(next_addr >> 4),
    # which encodes back to the aligned low 32 bits of next_addr.
    return regcmd_dma & 0xFFFFFFF0


def _align_up(x, align):
    return ((x + align - 1) // align) * align


def _mesa_pc_data_amount(reg_count):
    # Mesa patches REG_PC_REGISTER_AMOUNTS with ALIGN((next_len - 4) / 2, 2).
    return _align_up((reg_count - 4) // 2, 2)


def _rknn_matmul_pc_data_amount(segment_qwords):
    # RKNN matmul 424x424x424 dump: aligned segment stride=112 qwords,
    # PC_DATA_AMOUNT=55, so the count is 128-bit fetches minus one.
    return (segment_qwords // 2) - 1


def _rknn_gemm_pc_data_amount(reg_count):
    # Verified GEMM PC-chain captures use ceil(next_body_regs / 2) + 1:
    # 108 -> 55, 13 -> 8, 12 -> 7, 17 -> 10.
    return (reg_count + 1) // 2 + 1


def make_regs(ew_cfg, n, input_dma, weight_dma, output_dma, fdiv_op=False, pc_op_enable=0x18):
    dataout_width = (n + 7) // 8 - 1
    out_cvt = 1 if fdiv_op else 0x10001
    feat_cfg = 0x00017841 if fdiv_op else 0x00017849
    return [
        _regcmd(reg.TARGET_DPU, reg.S_POINTER, 0x0000000e),
        _regcmd(reg.TARGET_DPU, reg.FEATURE_MODE_CFG, 0x000001e5),
        _regcmd(reg.TARGET_DPU, reg.DATA_FORMAT, 0x48000002),
        _regcmd(reg.TARGET_DPU, reg.DATA_CUBE_WIDTH, dataout_width),
        _regcmd(reg.TARGET_DPU, reg.DATA_CUBE_HEIGHT, 0),
        _regcmd(reg.TARGET_DPU, reg.DATA_CUBE_CHANNEL, 0x00070007),
        _regcmd(reg.TARGET_DPU, reg.EW_CFG, ew_cfg),
        _regcmd(reg.TARGET_DPU, reg.OUT_CVT_SCALE, out_cvt),
        _regcmd(reg.TARGET_RDMA, reg.RDMA_S_POINTER, 0x0000000e),
        _regcmd(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_WIDTH, dataout_width),
        _regcmd(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_HEIGHT, 0),
        _regcmd(reg.TARGET_RDMA, reg.RDMA_DATA_CUBE_CHANNEL, 0x00000007),
        _regcmd(reg.TARGET_RDMA, reg.RDMA_ERDMA_CFG, 0x40000008),
        _regcmd(reg.TARGET_DPU, reg.DST_BASE_ADDR, output_dma & 0xFFFFFFFF),
        _regcmd(reg.TARGET_RDMA, reg.RDMA_SRC_BASE_ADDR, input_dma & 0xFFFFFFFF),
        _regcmd(reg.TARGET_RDMA, reg.RDMA_EW_BASE_ADDR, weight_dma & 0xFFFFFFFF),
        _regcmd(reg.TARGET_RDMA, reg.RDMA_FEATURE_MODE_CFG, feat_cfg),
        0,
        _regcmd(reg.TARGET_PC_REG, reg.REGISTER_AMOUNTS, 0),
        0x0041000000000000,
        _regcmd(reg.TARGET_PC, reg.OPERATION_ENABLE, pc_op_enable),
    ]


def _pad_before_pc_tail(regs, align):
    body = list(regs[:-4])
    tail = list(regs[-4:])
    while (len(body) + len(tail)) % align:
        body.append(0)
    return body + tail


def parse_core_ranges(spec, task_count):
    ranges = {}
    if spec == "split3":
        return {0: (0, 1), 1: (1, 1), 2: (2, 1)}
    if spec == "core0":
        return {0: (0, task_count)}
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


def build_tasks(tasks, regcmd, regcmd_dma, regcmd_mode, task_regs, pc_chain=False, pc_chain_style="rknnops", op_idx=4):
    for task_idx, regs in enumerate(task_regs):
        if pc_chain:
            if pc_chain_style == "rknn-matmul":
                regs = _pad_before_pc_tail(regs, 2)
            next_addr = 0
            if task_idx + 1 < len(task_regs):
                next_addr = regcmd_dma + (task_idx + 1) * REG_BLOCK_QWORDS * ctypes.sizeof(ctypes.c_uint64)
            regs = list(regs)
            regs[-4] = _regcmd(reg.TARGET_PC_REG, reg.BASE_ADDRESS, _pc_base_address_value(next_addr)) if next_addr else 0
            if pc_chain_style == "mesa":
                amount = _mesa_pc_data_amount(len(task_regs[task_idx + 1])) if task_idx + 1 < len(task_regs) else 0
            elif pc_chain_style == "add":
                amount = len(regs) - 4 if task_idx + 1 < len(task_regs) else 0
            elif pc_chain_style == "rknnops":
                amount = len(regs)
            elif pc_chain_style == "rknn-gemm":
                amount = _rknn_gemm_pc_data_amount(len(task_regs[task_idx + 1])) if task_idx + 1 < len(task_regs) else 0
            elif pc_chain_style == "rknn-matmul":
                amount = _rknn_matmul_pc_data_amount(REG_BLOCK_QWORDS) if task_idx + 1 < len(task_regs) else 0
            else:
                raise ValueError(f"unknown PC chain style: {pc_chain_style}")
            regs[-3] = _regcmd(reg.TARGET_PC_REG, reg.REGISTER_AMOUNTS, amount)

        block_qword = task_idx * REG_BLOCK_QWORDS
        for i in range(REG_BLOCK_QWORDS):
            regcmd[block_qword + i] = 0
        for i, value in enumerate(regs):
            regcmd[block_qword + i] = value

        tasks[task_idx].flags = 0
        tasks[task_idx].op_idx = op_idx
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


def _make_inputs(rng, n, op_name):
    _, expected_fn, kw = OPS[op_name]
    a = rng.uniform(-2, 2, size=n).astype(np.float16)
    b = rng.uniform(0.25, 2, size=n).astype(np.float16)
    if kw.get("neg_op"):
        b[:] = -1.0
    return a, b, expected_fn(a, b).astype(np.float16), kw


def _run_tiled(args):
    op_name = args.ops.strip().upper()
    if "," in args.ops or op_name not in OPS:
        raise ValueError(f"--tile-flat currently accepts exactly one op from: {', '.join(OPS)}")
    if args.tiles < 1 or args.tiles > MAX_TASKS:
        raise ValueError(f"--tiles must be 1..{MAX_TASKS}")
    if args.n % args.tiles:
        raise ValueError("--n must be divisible by --tiles for flat tiling")

    if args.execution == "sequential-core0":
        args.separate_submits = True
        args.core_ranges = "core0"
        args.core_mask = 0x1
        args.regcmd_mode = "absolute"
        args.submit_block = True
    elif args.execution == "pc-chain-core0":
        args.separate_submits = False
        args.core_ranges = "core0"
        args.core_mask = 0x1
        args.pc_chain = True
        args.pc_chain_style = "add"
        args.regcmd_mode = "absolute"
        args.submit_block = True
    elif args.execution == "unsafe-split3":
        if not args.allow_unsafe_submit:
            raise RuntimeError("--execution unsafe-split3 requires explicit --allow-unsafe-submit")
        if args.tiles != 3:
            raise ValueError("--execution unsafe-split3 requires --tiles 3")
        args.separate_submits = False
        args.core_ranges = "split3"
        args.core_mask = 0x7
        args.subcore_layout = "rk3588-tricore-tail"
        args.pc_chain = True
        args.pc_chain_style = "add"
        args.regcmd_mode = "absolute"
        args.submit_block = True
    else:
        raise ValueError(f"unknown execution mode: {args.execution}")

    tile_count = args.tiles
    tile_n = args.n // tile_count
    core_ranges_for_guard = parse_core_ranges(args.core_ranges, tile_count)
    validate_core_ranges(core_ranges_for_guard, tile_count)
    uses_nonzero_core = any(core != 0 and count for core, (_, count) in core_ranges_for_guard.items())
    uses_multi_range = sum(1 for _, count in core_ranges_for_guard.values() if count) > 1
    if (uses_nonzero_core or uses_multi_range) and not args.allow_unsafe_submit:
        raise RuntimeError(
            "Unsafe raw multicore submit refused. Re-run with --allow-unsafe-submit only with physical reset access."
        )

    fd = open_device(args.device)
    try:
        byte_stride = max(4096, tile_n * np.dtype(np.float16).itemsize)
        total_bytes = byte_stride * tile_count
        task_map, task_mc = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE, args.verbose)
        regcmd_map, regcmd_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE, args.verbose)
        input_map, input_mc = mem_allocate(fd, total_bytes, RKNPU_MEM_NON_CACHEABLE, args.verbose)
        weight_map, weight_mc = mem_allocate(fd, total_bytes, RKNPU_MEM_NON_CACHEABLE, args.verbose)
        output_map, output_mc = mem_allocate(fd, total_bytes, RKNPU_MEM_NON_CACHEABLE, args.verbose)

        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
        regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

        rng = np.random.default_rng(args.seed)
        a_full, b_full, expected_full, kw = _make_inputs(rng, args.n, op_name)
        for tile_idx in range(tile_count):
            start = tile_idx * tile_n
            end = start + tile_n
            in_view = np.frombuffer(input_map, dtype=np.float16, count=tile_n, offset=tile_idx * byte_stride)
            wt_view = np.frombuffer(weight_map, dtype=np.float16, count=tile_n, offset=tile_idx * byte_stride)
            out_view = np.frombuffer(output_map, dtype=np.float16, count=tile_n, offset=tile_idx * byte_stride)
            in_view[:] = a_full[start:end]
            wt_view[:] = b_full[start:end]
            out_view[:] = np.nan

        ew_cfg, _, _ = OPS[op_name]
        task_regs = []
        for tile_idx in range(tile_count):
            task_regs.append(make_regs(
                ew_cfg,
                tile_n,
                input_mc.dma_addr + tile_idx * byte_stride,
                weight_mc.dma_addr + tile_idx * byte_stride,
                output_mc.dma_addr + tile_idx * byte_stride,
                fdiv_op=kw.get("fdiv_op", False),
                pc_op_enable=args.pc_op_enable,
            ))

        build_tasks(
            tasks,
            regcmd,
            regcmd_mc.dma_addr,
            args.regcmd_mode,
            task_regs,
            pc_chain=args.pc_chain,
            pc_chain_style=args.pc_chain_style,
            op_idx=args.task_op_idx,
        )
        if args.enable_mask is not None:
            for tile_idx in range(tile_count):
                tasks[tile_idx].enable_mask = args.enable_mask

        for tile_idx in range(tile_count):
            print(
                f"tile[{tile_idx}] op={op_name} n={tile_n} regcfg_amount={tasks[tile_idx].regcfg_amount} "
                f"regcmd_addr=0x{tasks[tile_idx].regcmd_addr:x}"
            )

        default_flags = RKNPU_JOB_PC
        if args.submit_block:
            default_flags |= RKNPU_JOB_BLOCK
        if not args.no_pingpong:
            default_flags |= RKNPU_JOB_PINGPONG
        flags = args.flags if args.flags is not None else default_flags

        reset_npu(fd)
        start_time = time.perf_counter()
        submit_results = []
        if args.separate_submits:
            for tile_idx in range(tile_count):
                copy_task(tasks[0], tasks[tile_idx])
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
                print(f"sequential submit tile[{tile_idx}] core=0 ret={ret}")
                print_submit_layout(submit_struct)
        else:
            core_ranges = parse_core_ranges(args.core_ranges, tile_count)
            ret, submit_struct = submit(
                fd,
                task_mc.obj_addr,
                tile_count,
                core_ranges=core_ranges,
                core_mask=args.core_mask,
                flags=flags,
                timeout=args.timeout,
                subcore_layout=args.subcore_layout,
            )
            submit_results.append((ret, submit_struct))
            print_submit_layout(submit_struct)

        elapsed = time.perf_counter() - start_time
        print(f"execution={args.execution} submit ret={[ret for ret, _ in submit_results]} elapsed_ms={elapsed * 1000:.3f}")

        all_pass = all(ret == 0 for ret, _ in submit_results)
        output_full = np.empty(args.n, dtype=np.float16)
        active_tiles = set(range(tile_count)) if args.separate_submits else assigned_tasks(parse_core_ranges(args.core_ranges, tile_count))
        for tile_idx in range(tile_count):
            start = tile_idx * tile_n
            end = start + tile_n
            out = np.frombuffer(output_map, dtype=np.float16, count=tile_n, offset=tile_idx * byte_stride).copy()
            output_full[start:end] = out
            sentinel_overwritten = not np.all(np.isnan(out))
            if tile_idx not in active_tiles:
                kept_sentinel = not sentinel_overwritten
                print(f"{op_name} tile[{tile_idx}] {'SKIP' if kept_sentinel else 'FAIL'} unassigned sentinel_kept={kept_sentinel}")
                all_pass &= kept_sentinel
                continue
            expected = expected_full[start:end]
            ok = sentinel_overwritten and np.allclose(out, expected, atol=args.atol, equal_nan=False)
            md = float(np.nanmax(np.abs(out - expected))) if out.size else 0.0
            print(f"{op_name} tile[{tile_idx}] {'PASS' if ok else 'FAIL'} sentinel={sentinel_overwritten} max_diff={md:.4f}")
            all_pass &= ok

        if all_pass:
            ok_full = np.allclose(output_full, expected_full, atol=args.atol, equal_nan=False)
            md_full = float(np.nanmax(np.abs(output_full - expected_full))) if output_full.size else 0.0
            print(f"{op_name} tiled_output {'PASS' if ok_full else 'FAIL'} max_diff={md_full:.4f}")
            all_pass &= ok_full
        return 0 if all_pass else 1
    finally:
        os.close(fd)


def run(args):
    if args.tile_flat:
        return _run_tiled(args)

    op_names = [x.strip().upper() for x in args.ops.split(",")]
    if any(name not in OPS for name in op_names):
        raise ValueError(f"--ops must name ops from: {', '.join(OPS)}")
    task_count = args.task_count if args.task_count is not None else len(op_names)
    if task_count < 1 or task_count > MAX_TASKS:
        raise ValueError(f"--task-count must be 1..{MAX_TASKS}")
    if len(op_names) == 1:
        op_names *= task_count
    if len(op_names) != task_count:
        raise ValueError("--ops must name one op or exactly --task-count comma-separated ops")

    core_ranges_for_guard = parse_core_ranges(args.core_ranges, task_count)
    validate_core_ranges(core_ranges_for_guard, task_count)
    uses_nonzero_core = any(core != 0 and count for core, (_, count) in core_ranges_for_guard.items())
    uses_multi_range = sum(1 for _, count in core_ranges_for_guard.values() if count) > 1
    if (uses_nonzero_core or uses_multi_range) and not args.allow_unsafe_submit:
        raise RuntimeError(
            "Unsafe raw multicore submit refused. Official RKNN multicore requires compiler/runtime "
            "multicore metadata; this raw ioctl path has locked up this kernel. Re-run with "
            "--allow-unsafe-submit only with physical reset access."
        )

    fd = open_device(args.device)
    try:
        n = args.n
        byte_stride = max(4096, n * np.dtype(np.float16).itemsize)
        task_map, task_mc = mem_allocate(fd, 4096, RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_NON_CACHEABLE, args.verbose)
        regcmd_map, regcmd_mc = mem_allocate(fd, 4096, RKNPU_MEM_NON_CACHEABLE, args.verbose)
        input_map, input_mc = mem_allocate(fd, byte_stride * task_count, RKNPU_MEM_NON_CACHEABLE, args.verbose)
        weight_map, weight_mc = mem_allocate(fd, byte_stride * task_count, RKNPU_MEM_NON_CACHEABLE, args.verbose)
        output_map, output_mc = mem_allocate(fd, byte_stride * task_count, RKNPU_MEM_NON_CACHEABLE, args.verbose)

        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(struct_rknpu_task))
        regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))

        rng = np.random.default_rng(args.seed)
        expected = []
        for task_idx, op_name in enumerate(op_names):
            _, expected_fn, kw = OPS[op_name]
            a = rng.uniform(-2, 2, size=n).astype(np.float16)
            b = rng.uniform(0.25, 2, size=n).astype(np.float16)
            if kw.get("neg_op"):
                b[:] = -1.0
            in_view = np.frombuffer(input_map, dtype=np.float16, count=n, offset=task_idx * byte_stride)
            wt_view = np.frombuffer(weight_map, dtype=np.float16, count=n, offset=task_idx * byte_stride)
            out_view = np.frombuffer(output_map, dtype=np.float16, count=n, offset=task_idx * byte_stride)
            in_view[:] = a
            wt_view[:] = b
            out_view[:] = np.nan
            expected.append(expected_fn(a, b).astype(np.float16))

        task_regs = []
        for task_idx, op_name in enumerate(op_names):
            ew_cfg, _, kw = OPS[op_name]
            task_regs.append(make_regs(
                ew_cfg,
                n,
                input_mc.dma_addr + task_idx * byte_stride,
                weight_mc.dma_addr + task_idx * byte_stride,
                output_mc.dma_addr + task_idx * byte_stride,
                fdiv_op=kw.get("fdiv_op", False),
                pc_op_enable=args.pc_op_enable,
            ))
        build_tasks(
            tasks,
            regcmd,
            regcmd_mc.dma_addr,
            args.regcmd_mode,
            task_regs,
            pc_chain=args.pc_chain,
            pc_chain_style=args.pc_chain_style,
            op_idx=args.task_op_idx,
        )

        if args.enable_mask is not None:
            for task_idx in range(task_count):
                tasks[task_idx].enable_mask = args.enable_mask

        for task_idx in range(task_count):
            print(
                f"task[{task_idx}] op={op_names[task_idx]} regcfg_amount={tasks[task_idx].regcfg_amount} "
                f"regcfg_offset={tasks[task_idx].regcfg_offset} regcmd_addr=0x{tasks[task_idx].regcmd_addr:x}"
            )

        default_flags = RKNPU_JOB_PC
        if args.submit_block:
            default_flags |= RKNPU_JOB_BLOCK
        if not args.no_pingpong:
            default_flags |= RKNPU_JOB_PINGPONG
        flags = args.flags if args.flags is not None else default_flags
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
                print(f"separate submit task[{task_idx}] core=0 ret={ret}")
                print_submit_layout(submit_struct)
        else:
            core_ranges = parse_core_ranges(args.core_ranges, task_count)
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
        elapsed = time.perf_counter() - start
        if not args.separate_submits:
            print_submit_layout(submit_results[0][1])
        print(f"submit ret={[ret for ret, _ in submit_results]} elapsed_ms={elapsed * 1000:.3f}")

        active_tasks = set(range(task_count)) if args.separate_submits else assigned_tasks(parse_core_ranges(args.core_ranges, task_count))
        all_pass = all(ret == 0 for ret, _ in submit_results)
        for task_idx, op_name in enumerate(op_names):
            out = np.frombuffer(output_map, dtype=np.float16, count=n, offset=task_idx * byte_stride).copy()
            sentinel_overwritten = not np.all(np.isnan(out))
            if task_idx not in active_tasks:
                kept_sentinel = not sentinel_overwritten
                print(f"{op_name} task[{task_idx}] {'SKIP' if kept_sentinel else 'FAIL'} unassigned sentinel_kept={kept_sentinel}")
                all_pass &= kept_sentinel
                continue
            ok = sentinel_overwritten and np.allclose(out, expected[task_idx], atol=args.atol, equal_nan=False)
            md = float(np.nanmax(np.abs(out - expected[task_idx]))) if out.size else 0.0
            print(f"{op_name} task[{task_idx}] {'PASS' if ok else 'FAIL'} sentinel={sentinel_overwritten} max_diff={md:.4f}")
            all_pass &= ok
        return 0 if all_pass else 1
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description="Submit elementwise tasks to RK3588 NPU. Safe default is one core-0 task.")
    parser.add_argument("--device", default="/dev/dri/card1")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--ops", default="ADD", help="One op for all tasks or one comma-separated op per task.")
    parser.add_argument("--task-count", type=int, default=None, help="Number of task descriptors to build. Defaults to the number of --ops entries.")
    parser.add_argument("--tile-flat", action="store_true", help="Treat --n as one logical flat vector split across --tiles independent ADD tasks.")
    parser.add_argument("--tiles", type=int, default=3, help="Tile count for --tile-flat.")
    parser.add_argument(
        "--execution",
        choices=("sequential-core0", "pc-chain-core0", "unsafe-split3"),
        default="sequential-core0",
        help="Execution mode for --tile-flat. unsafe-split3 mutates core mask/layout and requires physical reset access.",
    )
    parser.add_argument("--core-ranges", default="core0", help="'split3', 'core0', or core:start:count comma list. split3 is risky on this kernel.")
    parser.add_argument("--core-mask", type=lambda x: int(x, 0), default=0x1)
    parser.add_argument("--separate-submits", action="store_true", help="Run each task as a sequential core-0 submit.")
    parser.add_argument("--allow-unsafe-submit", action="store_true", help="Allow raw nonzero-core or multi-range submits that may hard-lock the kernel.")
    parser.add_argument(
        "--subcore-layout",
        choices=("direct", "rk3588-tricore-tail"),
        default="direct",
        help="Map core ranges to subcore_task[]. allbilly/rknpu_driver uses indices 2..4 for 3-core RK3588 jobs.",
    )
    parser.add_argument("--pc-chain", action="store_true", help="Patch each regcmd tail with PC next-address/count like Mesa split tasks.")
    parser.add_argument("--pc-chain-style", choices=("add", "rknnops", "rknn-gemm", "mesa", "rknn-matmul"), default="rknnops", help="Choose PC_REGISTER_AMOUNTS patching style for --pc-chain.")
    parser.add_argument("--task-op-idx", type=lambda x: int(x, 0), default=4, help="Task descriptor op_idx. rknnops.h uses 1; current raw examples used 4.")
    parser.add_argument("--enable-mask", type=lambda x: int(x, 0), default=None, help="Task descriptor enable_mask. rknnops.h uses 0x0d.")
    parser.add_argument("--pc-op-enable", type=lambda x: int(x, 0), default=0x18, help="REG_PC_OPERATION_ENABLE value in each regcmd tail.")
    parser.add_argument("--regcmd-mode", choices=("offset", "absolute"), default="offset")
    parser.add_argument("--flags", type=lambda x: int(x, 0), default=None)
    parser.add_argument("--submit-block", action="store_true", help="Include RKNPU_JOB_BLOCK in default submit flags, matching rknnops.h submitTask.")
    parser.add_argument("--no-pingpong", action="store_true")
    parser.add_argument("--timeout", type=int, default=6000)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
