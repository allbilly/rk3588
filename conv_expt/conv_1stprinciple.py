"""Compact first-principles FP16 CONV runner for RK3588 NPU.

The old research harness was copied to ``conv_1stprinciple_ref.py``.  This file
keeps only the runtime shape: planner descriptors are lowered into simple local
hardware submits.  It intentionally does not replay RKNN exact11/exact12 capture
closures; unsupported chained semantics should be derived in the planner/emitter
instead of added here as shape-name materializers.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import re
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conv_expt import conv_tile_planner as planner  # noqa: E402
from conv_expt.conv_tile_cpu import SHAPES  # noqa: E402
from examples import conv as hw  # noqa: E402

def _load_hw6_pure():
    import types

    path = REPO_ROOT / "examples/kernel_6_18/conv.py"
    lines = path.read_text().splitlines()
    kept = []
    skip_globals = False
    for line in lines:
        if line.startswith("fd = rt.open_rocket_device()"):
            skip_globals = True
            continue
        if skip_globals and line.startswith("def _ceil_div"):
            skip_globals = False
        if not skip_globals:
            kept.append(line)
    module = types.ModuleType("rk3588_kernel_6_18_conv_pure")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec("\n".join(kept), module.__dict__)
    return module


hw6 = _load_hw6_pure()


FP16_BYTES = 2
CBUF_BANKS = 12
CBUF_ENTRY_BYTES = 128
CBUF_ENTRIES_PER_BANK = 256
MAC_ATOMIC_C = 16
MAC_ATOMIC_K = 32
MIN_HW_OC = 2
POINTWISE_WIDE_MIN_OC = 32


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align_up(x: int, align: int) -> int:
    return _ceil_div(x, align) * align


def _is_depthwise(s: dict) -> bool:
    return s["groups"] == s["in_c"] == s["out_c"]


def _is_pointwise(s: dict) -> bool:
    return s["groups"] == 1 and s["kh"] == 1 and s["kw"] == 1


def _is_pointwise_wide(s: dict) -> bool:
    return _is_pointwise(s) and s["in_c"] >= 64


def _shape_from_name(name: str) -> dict:
    for s in SHAPES:
        if s["name"] == name:
            return dict(s)
    return hw.shape_from_name(name)


def _descriptor_rows(s: dict) -> list[dict]:
    return planner.descriptor_rows_for_shape(s)


def _split_summary(rows: list[dict]) -> tuple[str, set[str]]:
    if not rows:
        raise ValueError("planner produced no descriptor rows")
    split = rows[0]["split_method"]
    if any(row["split_method"] != split for row in rows):
        raise ValueError("mixed split methods in descriptor rows")
    return split, {row["family"] for row in rows}


def _print_rows(rows: list[dict]) -> None:
    print("idx split family y input_h out_h k oc feature_off weight_off output_off banks family_bits")
    for idx, row in enumerate(rows):
        print(
            f"{idx} {row['split_method']} {row['family']} "
            f"{row['y_start']} {row['input_h']} {row['output_h']} "
            f"{row['k_start']} {row['oc_count']} "
            f"0x{row['feature_off']:x} 0x{row['weight_off']:x} 0x{row['output_off']:x} "
            f"{row['input_bank_num']}/{row['weight_bank_num']} {row['family_bits']}"
        )


def _expected(inp: np.ndarray, wt: np.ndarray, s: dict) -> np.ndarray:
    if s["batch"] == 1 and s["groups"] == 1:
        return hw.compute_expected_vectorized(inp, wt, s)
    return hw.compute_expected(inp, wt, s)


def _random_case(s: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(42)
    inp = np.random.uniform(-2, 2, (s["batch"], s["in_c"], s["in_h"], s["in_w"])).astype(np.float16)
    wt = np.random.uniform(-2, 2, (s["out_c"], s["weight_in_c"], s["kh"], s["kw"])).astype(np.float16)
    return inp, wt, _expected(inp, wt, s)


def _compare_and_print(s: dict, got: np.ndarray, expected: np.ndarray, path: str, tasks: int) -> None:
    max_diff = float(np.max(np.abs(got.astype(np.float64) - expected)))
    ok = bool(np.allclose(got, expected, atol=0.12))
    print(f"shape={s['name']} path={path} tasks={tasks} {'PASS' if ok else 'FAIL'} max_diff={max_diff:.4f}")
    if not ok:
        raise AssertionError(f"output mismatch max_diff={max_diff:.4f}")


def _post_submit_reset_once() -> None:
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    try:
        hw.post_submit_reset(fd)
    finally:
        os.close(fd)


def _run_hw_single(tile_shape: dict, tile_in: np.ndarray, tile_wt: np.ndarray) -> np.ndarray:
    got = hw._run_single_tile_shape(tile_shape, tile_in, tile_wt)
    _post_submit_reset_once()
    return got


def _run_hw_y_tile(tile_shape: dict, tile_in: np.ndarray, tile_wt: np.ndarray) -> np.ndarray:
    p = hw._conv_params(tile_shape)
    input_flat = hw.pack_input(tile_in, p).view(np.uint16)
    weight_flat = hw.pack_weights(tile_wt, tile_shape, p).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        local_row = {"y_start": 0, "input_h": tile_shape["in_h"], "weight_reuse": False}
        regs = [hw.make_y_tile_regs(tile_shape, p, local_row, input_mem.dma_addr,
                                    weight_mem.dma_addr, output_mem.dma_addr, 0)]
        hw.write_tasks(task_map, regcmd_map, regcmd_mem, regs)
        if hw.npu_submit(fd, task_mem.obj_addr, 1) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    return hw.unpack_output(out_raw, tile_shape["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)


def _hw_oc(s: dict, oc_count: int) -> int:
    if _is_pointwise_wide(s) and oc_count < POINTWISE_WIDE_MIN_OC:
        return POINTWISE_WIDE_MIN_OC
    return max(MIN_HW_OC, oc_count)


def _padded_weight(s: dict, wt: np.ndarray, oc_start: int, oc_count: int, hw_oc: int, weight_in_c: int | None = None) -> np.ndarray:
    weight_in_c = s["weight_in_c"] if weight_in_c is None else weight_in_c
    if hw_oc == oc_count:
        return wt[oc_start:oc_start + oc_count]
    out = np.zeros((hw_oc, weight_in_c, s["kh"], s["kw"]), dtype=np.float16)
    out[:oc_count] = wt[oc_start:oc_start + oc_count]
    return out


def _local_tile(
    s: dict,
    inp0: np.ndarray,
    wt: np.ndarray,
    y_start: int,
    input_h: int,
    oc_start: int,
    oc_count: int,
) -> np.ndarray:
    hw_oc = _hw_oc(s, oc_count)
    tile_shape = dict(s, name=s["name"] + "_local", batch=1, groups=1, in_h=input_h, out_c=hw_oc)
    tile_in = inp0[:, y_start:y_start + input_h, :]
    tile_wt = _padded_weight(s, wt, oc_start, oc_count, hw_oc)
    if y_start or input_h != s["in_h"]:
        return _run_hw_y_tile(tile_shape, tile_in, tile_wt)[:oc_count]
    return _run_hw_single(tile_shape, tile_in, tile_wt)[:oc_count]


def _run_grouped_full(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    if s["in_c"] % s["groups"] or s["out_c"] % s["groups"]:
        raise ValueError("grouped lowering requires divisible input/output channels")
    p = hw._conv_params(s)
    got = np.zeros((s["batch"], s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    in_per_group = s["in_c"] // s["groups"]
    out_per_group = s["out_c"] // s["groups"]
    tasks = 0
    for n in range(s["batch"]):
        for g in range(s["groups"]):
            ic0 = g * in_per_group
            oc0 = g * out_per_group
            tile_base = dict(s, batch=1, in_c=in_per_group, weight_in_c=in_per_group, groups=1)
            hw_oc = _hw_oc(tile_base, out_per_group)
            tile_shape = dict(tile_base, name=s["name"] + "_group_full", out_c=hw_oc)
            tile_wt = np.zeros((hw_oc, in_per_group, s["kh"], s["kw"]), dtype=np.float16)
            tile_wt[:out_per_group] = wt[oc0:oc0 + out_per_group]
            tile_got = _run_hw_single(tile_shape, inp[n, ic0:ic0 + in_per_group], tile_wt)
            got[n, oc0:oc0 + out_per_group] = tile_got[:out_per_group]
            tasks += 1
    return got, tasks, "grouped_full" if tasks > 1 else "direct"


def _run_y_local(s: dict, rows: list[dict], inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    if s["batch"] != 1 or s["groups"] != 1:
        return _run_grouped_full(s, inp, wt)
    p = hw._conv_params(s)
    got = np.zeros((1, s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    for row in rows:
        tile = _local_tile(s, inp[0], wt, row["y_start"], row["input_h"], 0, s["out_c"])
        got[0, :, row["y_start"]:row["y_start"] + tile.shape[1]] = tile
    return got, len(rows), "by_y_local"


def _run_y_small_local(s: dict, inp: np.ndarray, wt: np.ndarray, max_input_h: int = 10) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    got = np.zeros((1, s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    y_start = 0
    tasks = 0
    while y_start < p["out_h"]:
        input_h = min(max_input_h, s["in_h"] - y_start)
        tile_shape = dict(s, name=s["name"] + "_small_y", batch=1, groups=1, in_h=input_h)
        tile_in = inp[0, :, y_start:y_start + input_h, :]
        tile = _run_hw_single(tile_shape, tile_in, wt)
        out_h = min(tile.shape[1], p["out_h"] - y_start)
        got[0, :, y_start:y_start + out_h] = tile[:, :out_h]
        y_start += out_h
        tasks += 1
    return got, tasks, "by_y_small_local"


def _run_pointwise_yk(s: dict, rows: list[dict], inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    got = np.zeros((1, s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    tasks = 0
    for row in rows:
        for oc_start, oc_count in ((0, 32), (32, s["out_c"] - 32)):
            tile_shape = dict(s, batch=1, groups=1, in_h=row["input_h"], out_c=oc_count)
            tile_in = inp[0, :, row["y_start"]:row["y_start"] + row["input_h"], :]
            tile_wt = wt[oc_start:oc_start + oc_count]
            tile = _run_hw_y_tile(tile_shape, tile_in, tile_wt)
            got[0, oc_start:oc_start + oc_count, row["y_start"]:row["y_start"] + tile.shape[1]] = tile
            tasks += 1
    return got, tasks, "pointwise_yk"


def _run_local_tile_replay(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    specs = hw._tile_replay_specs(s)
    got = np.zeros((1, s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, 4 * 1024 * 1024, hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, 4 * 1024 * 1024, hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, 4 * 1024 * 1024, hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        for y_start, input_h, output_h, oc_start, oc_count in specs:
            hw_oc = 32 if _is_pointwise_wide(s) and oc_count < 32 else oc_count
            tile_shape = dict(s, name=s["name"] + "_tilelocal", in_h=input_h, out_c=hw_oc)
            tile_p = hw._conv_params(tile_shape)
            tile_in = inp[0, :, y_start:y_start + input_h, :]
            tile_wt = np.zeros((hw_oc, s["weight_in_c"], s["kh"], s["kw"]), dtype=np.float16)
            tile_wt[:oc_count] = wt[oc_start:oc_start + oc_count]
            input_flat = hw.pack_input(tile_in, tile_p).view(np.uint16)
            weight_flat = hw.pack_weights(tile_wt, tile_shape, tile_p).view(np.uint16)
            out_count = hw._ceil_div(tile_p["align_out_c"], hw.UNPACK_C2) * tile_p["out_width_stride"] * hw.UNPACK_C2
            (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
            (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
            ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
            if y_start or output_h < p["out_h"]:
                local_row = {"y_start": 0, "input_h": input_h, "weight_reuse": False}
                regs = [hw.make_y_tile_regs(tile_shape, tile_p, local_row, input_mem.dma_addr,
                                            weight_mem.dma_addr, output_mem.dma_addr, 0)]
            else:
                regs = [hw.make_regs(tile_shape, tile_p, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, True)]
            hw.write_tasks(task_map, regcmd_map, regcmd_mem, regs)
            if hw.npu_submit(fd, task_mem.obj_addr, 1) < 0:
                raise RuntimeError("npu_submit failed")
            out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
            tile_got = hw.unpack_output(out_raw, hw_oc, output_h, p["out_w"], tile_p["out_width_stride"], hw.UNPACK_C2)[:oc_count]
            got[0, oc_start:oc_start + oc_count, y_start:y_start + output_h] = tile_got
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    return got, len(specs), "local_tile_replay"


def _run_h160_setup3_padded(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    wide = dict(s, name=hw.H160_SPATIAL_BY_Y_SHAPE, in_c=16, weight_in_c=16, out_c=128)
    p = hw._conv_params(wide)
    wide_in = np.zeros((16, s["in_h"], s["in_w"]), dtype=np.float16)
    wide_in[:s["in_c"]] = inp[0]
    wide_wt = np.zeros((128, 16, s["kh"], s["kw"]), dtype=np.float16)
    wide_wt[:s["out_c"], :s["in_c"]] = wt
    input_flat = hw.pack_input(wide_in, p).view(np.uint16)
    weight_flat = hw.pack_weights(wide_wt, wide, p).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2
    output_bytes = out_count * FP16_BYTES
    output_flags = hw.RKNPU_MEM_NON_CONTIGUOUS if output_bytes > 4 * 1024 * 1024 else hw.RKNPU_MEM_NON_CACHEABLE

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, output_bytes), output_flags)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        rows = tuple(dict(row, weight_reuse=idx > 0) for idx, row in enumerate(hw._h160_spatial_by_y_rows(wide, p)))
        regs = [hw.make_y_tile_regs(wide, p, row, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr, row["feature_off"]) for row in rows]
        hw.write_h160_setup3_tasks(task_map, regcmd_map, regcmd_mem, regs)
        if hw.npu_submit(fd, task_mem.obj_addr, 3, core_mask=0,
                         subcores=((0, 3), (0, 0), (0, 0), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, wide["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got[:s["out_c"]].reshape(1, s["out_c"], p["out_h"], p["out_w"]), 3, "h160_setup3_padded"


def _run_h40_exact17_padded(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    wide = dict(s, name=hw.H40_SPATIAL_BY_Y_SHAPE, in_c=160, weight_in_c=160, out_c=320)
    p = hw._conv_params(wide)
    wide_in = np.zeros((160, s["in_h"], s["in_w"]), dtype=np.float16)
    wide_in[:s["in_c"]] = inp[0]
    wide_wt = np.zeros((320, 160, s["kh"], s["kw"]), dtype=np.float16)
    wide_wt[:s["out_c"], :s["in_c"]] = wt
    input_flat = hw.pack_input(wide_in, p).view(np.uint16)
    weight_flat = hw.pack_weights(wide_wt, wide, p).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 16384, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = hw._h40_exact17_task_regs(wide, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_h40_exact17_tasks(task_map, regcmd_map, regcmd_mem, regs)
        if hw.npu_submit(fd, task_mem.obj_addr, 6, core_mask=0,
                         subcores=((0, 2), (0, 2), (0, 2), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, wide["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got[:s["out_c"]].reshape(1, s["out_c"], p["out_h"], p["out_w"]), 17, "h40_exact17_padded"


def _run_k_local(s: dict, rows: list[dict], inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    if s["batch"] != 1 or s["groups"] != 1:
        return _run_grouped_full(s, inp, wt)
    p = hw._conv_params(s)
    got = np.zeros((1, s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    k_rows = [row for row in rows if row["family"] in {"setup", "k_tile"}]
    if not k_rows:
        raise ValueError("no local K rows to submit")
    for row in k_rows:
        tile = _local_tile(s, inp[0], wt, 0, s["in_h"], row["k_start"], row["oc_count"])
        got[0, row["k_start"]:row["k_start"] + row["oc_count"]] = tile
    return got, len(k_rows), "by_k_local"


def _run_pointwise_exact11_byk(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    layout = hw.exact_byk_legacy_layout_check(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw.pack_weights(wt, s, p).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2
    output_bytes = out_count * FP16_BYTES

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 8192, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, output_bytes), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = hw._pointwise_exact11_byk_task_regs(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, regs, layout)
        if hw.npu_submit(fd, task_mem.obj_addr, 3, core_mask=0,
                         subcores=((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "pointwise_exact11_byk"


def _run_prefix_exact11_byk(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    layout = hw.exact_byk_legacy_layout_check(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    if s["name"] in getattr(hw, "POINTWISE_EXACT11_COMPACT_WEIGHT_SHAPES", set()):
        weight_flat = hw._pack_pointwise_wide(wt, s["out_c"], s["in_c"]).view(np.uint16)
    else:
        weight_flat = hw.pack_weights(wt, s, p).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 8192, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = hw._exact11_task_regs(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, regs, layout)
        if hw.npu_submit(fd, task_mem.obj_addr, 3, core_mask=0,
                         subcores=((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "prefix_exact11_byk"


def _run_setup108_compact(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw._pack_pointwise_compact_weight(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = [hw._exact11_body_regs(s, "setup", 0, s["out_c"], input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)]
        hw.write_tasks(task_map, regcmd_map, regcmd_mem, regs)
        if hw.npu_submit(fd, task_mem.obj_addr, 1, core_mask=0,
                         subcores=((0, 1), (0, 0), (0, 0), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "setup108_compact"


def _run_pointwise_chain_compact(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    layout = hw.exact_byk_legacy_layout_check(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw._pack_pointwise_compact_weight(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2
    regcmd_bytes = (layout["offsets"][-1] + layout["amounts"][-1]) * ctypes.sizeof(ctypes.c_uint64)

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, hw.regcmd_alloc_bytes(regcmd_bytes), hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = hw._pointwise_exact11_chain_compact_weight_task_regs(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, regs, layout)
        if hw.npu_submit(fd, task_mem.obj_addr, 1, core_mask=0,
                         subcores=((0, 1), (0, 0), (0, 0), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "pointwise_chain_compact"


def _is_c1024_h7_oc1024(s: dict) -> bool:
    return _is_pointwise(s) and s["in_c"] == 1024 and s["out_c"] == 1024 and s["in_h"] == 7 and s["in_w"] == 7


def _is_c1024_h1_oc1001(s: dict) -> bool:
    return _is_pointwise(s) and s["in_c"] == 1024 and s["out_c"] == 1001 and s["in_h"] == 1 and s["in_w"] == 1


def _is_c480_h14_oc16(s: dict) -> bool:
    return _is_pointwise(s) and s["in_c"] == 480 and s["out_c"] == 16 and s["in_h"] == 14 and s["in_w"] == 14


def _is_c512_h14_oc24(s: dict) -> bool:
    return _is_pointwise(s) and s["in_c"] == 512 and s["out_c"] == 24 and s["in_h"] == 14 and s["in_w"] == 14


def _is_c1280_h10_oc24(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_K"
        and families == {"k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 1280
        and s["out_c"] == 24
        and s["in_h"] == 10
        and s["in_w"] == 10
        and len(rows) == 3
    )


def _is_c1280_h10_oc546(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_K"
        and families == {"k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 1280
        and s["out_c"] == 546
        and s["in_h"] == 10
        and s["in_w"] == 10
    )


def _is_c384_h19_oc64_or_96(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_K"
        and families == {"k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 384
        and s["out_c"] in (64, 96)
        and s["in_h"] == 19
        and s["in_w"] == 19
    )


def _is_c576_h19_oc96(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_YK"
        and families == {"setup", "k_half", "k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 576
        and s["out_c"] == 96
        and s["in_h"] == 19
        and s["in_w"] == 19
    )


def _is_c576_h19_oc12(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_Y"
        and families == {"y_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 576
        and s["out_c"] == 12
        and s["in_h"] == 19
        and s["in_w"] == 19
    )


def _is_c576_h19_oc273(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_YK"
        and families == {"setup", "k_half", "k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 576
        and s["out_c"] == 273
        and s["in_h"] == 19
        and s["in_w"] == 19
    )


def _is_c256_h3_oc546(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_K"
        and families == {"k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 256
        and s["out_c"] == 546
        and s["in_h"] == 3
        and s["in_w"] == 3
    )


def _is_c256_h2_oc546(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_K"
        and families == {"k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 256
        and s["out_c"] == 546
        and s["in_h"] == 2
        and s["in_w"] == 2
    )


def _is_c128_h1_oc24(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "NONE"
        and families == {"setup"}
        and _is_pointwise(s)
        and s["in_c"] == 128
        and s["out_c"] == 24
        and s["in_h"] == 1
        and s["in_w"] == 1
    )


def _is_c8_h160_oc16_k3(s: dict) -> bool:
    return (
        s["batch"] == 1
        and s["groups"] == 1
        and s["in_c"] == 8
        and s["out_c"] == 16
        and s["in_h"] == 160
        and s["in_w"] == 160
        and s["kh"] == 3
        and s["kw"] == 3
    )


def _is_c40_h40_oc160_k3(s: dict) -> bool:
    return (
        s["batch"] == 1
        and s["groups"] == 1
        and s["in_c"] == 40
        and s["out_c"] == 160
        and s["in_h"] == 40
        and s["in_w"] == 40
        and s["kh"] == 3
        and s["kw"] == 3
    )


def _is_c576_h20_oc96(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_YK"
        and families == {"setup", "k_half", "k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 576
        and s["out_c"] == 96
        and s["in_h"] == 20
        and s["in_w"] == 20
    )


def _is_c768_h20_oc96(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_YK"
        and families == {"setup", "k_half", "k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 768
        and s["out_c"] == 96
        and s["in_h"] == 20
        and s["in_w"] == 20
    )


def _is_c768_or_c960_h10_oc120(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_K"
        and families == {"k_tile"}
        and _is_pointwise(s)
        and s["in_c"] in (768, 960)
        and s["out_c"] == 120
        and s["in_h"] == 10
        and s["in_w"] == 10
    )


def _is_c320_h20_oc72(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_K"
        and families == {"k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 320
        and s["out_c"] == 72
        and s["in_h"] == 20
        and s["in_w"] == 20
    )


def _is_c288_h20_oc72(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_K"
        and families == {"k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 288
        and s["out_c"] == 72
        and s["in_h"] == 20
        and s["in_w"] == 20
    )


def _is_c72_h20_oc576(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_K"
        and families == {"k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 72
        and s["out_c"] == 576
        and s["in_h"] == 20
        and s["in_w"] == 20
    )


def _is_c576_h20_oc72(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_YK"
        and families == {"setup", "k_half", "k_tile"}
        and _is_pointwise(s)
        and s["in_c"] == 576
        and s["out_c"] == 72
        and s["in_h"] == 20
        and s["in_w"] == 20
    )


def _is_c72_h20_oc288_k3(s: dict, rows: list[dict]) -> bool:
    split, families = _split_summary(rows)
    return (
        split == "BY_K"
        and families == {"k_tile"}
        and s["batch"] == 1
        and s["groups"] == 1
        and s["in_c"] == 72
        and s["out_c"] == 288
        and s["in_h"] == 20
        and s["in_w"] == 20
        and s["kh"] == 3
        and s["kw"] == 3
    )


def _c128_h1_oc24_rows(s: dict, in_dma: int, wt_dma: int, out_dma: int) -> list[list[int]]:
    full_weight = s["out_c"] * s["in_c"] * FP16_BYTES
    regs = hw._exact11_body_regs(s, "setup", 0, s["out_c"], in_dma, wt_dma, out_dma)
    row = hw.patch_regs(regs, {
        (hw.reg.CNA, hw.reg.CNA_DATA_SIZE1): 0x003f0080,
        (hw.reg.CNA, hw.reg.CNA_CBUF_CON0): 0x0b1,
        (hw.reg.CNA, hw.reg.CNA_CBUF_CON1): 0x004,
        (hw.reg.CNA, hw.reg.CNA_CVT_CON0): 0x000b,
        (hw.reg.CNA, hw.reg.CNA_DMA_CON2): 0x0ffffffd,
        (hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE0): full_weight,
        (hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE1): s["in_c"] * FP16_BYTES,
        (hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE2): (s["kw"] << 24) | (s["kh"] << 16) | s["out_c"],
        (hw.reg.CNA, hw.reg.CNA_CONV_CON2): 0x20,
        (hw.reg.CORE, hw.reg.CORE_DATAOUT_SIZE_1): s["out_c"] - 1,
        (hw.reg.DPU, hw.reg.DATA_CUBE_CHANNEL): ((s["out_c"] - 1) << 16) | (s["out_c"] - 1),
        (hw.reg.DPU, hw.reg.WDMA_SIZE_0): s["out_c"] - 1,
    })
    return [row, row, row]


def _c512_h14_oc24_rows(s: dict, in_dma: int, wt_dma: int, out_dma: int) -> list[list[int]]:
    aux_dma = wt_dma + s["out_c"] * s["in_c"] * FP16_BYTES
    common = {
        (hw.reg.CNA, hw.reg.CNA_DATA_SIZE1): 0x003f0200,
        (hw.reg.CNA, hw.reg.CNA_CBUF_CON0): 0x057,
        (hw.reg.CNA, hw.reg.CNA_CVT_CON0): 0x000b,
        (hw.reg.CNA, hw.reg.CNA_DMA_CON2): 0x08c,
    }
    materialized = [hw.patch_regs(
        hw._exact11_body_regs(s, "setup", 0, s["out_c"], in_dma, wt_dma, out_dma, conv2_low=0x090),
        common,
    )]
    windows = ((0, 7, 0x10000080), (7, 7, 0x10000080), (0, 5, 0x20000060), (5, 5, 0x20000060), (10, 4, 0x20000050))
    for idx, (y_start, input_h, conv2) in enumerate(windows):
        materialized.append(hw.patch_regs(
            hw._exact11_body_regs(s, "y_tile", 0, s["out_c"], in_dma, wt_dma, out_dma,
                                  y_start=y_start, input_h=input_h, conv2_low=conv2 & 0x0fffffff),
            common | {(hw.reg.CNA, hw.reg.CNA_CONV_CON2): conv2},
        ))
        if idx + 1 < len(windows):
            materialized.append(hw._exact11_aux_regs(s, out_dma, aux_dma))
    materialized.append(hw._exact11_aux_regs(s, out_dma, aux_dma))
    return materialized


def _c1280_h10_oc24_rows(s: dict, rows: list[dict], in_dma: int, wt_dma: int, out_dma: int) -> list[list[int]]:
    if not _is_c1280_h10_oc24(s, rows):
        raise ValueError("c1280/h10/oc24 formula requires matching BY_K rows")
    aux_dma = wt_dma + s["out_c"] * s["in_c"] * FP16_BYTES
    common = {
        (hw.reg.CNA, hw.reg.CNA_DATA_SIZE1): (0x3f << 16) | s["in_c"],
        (hw.reg.CNA, hw.reg.CNA_CBUF_CON0): 0x048,
        (hw.reg.CNA, hw.reg.CNA_CVT_CON0): 0x000b,
    }
    materialized = [hw.patch_regs(
        hw._exact11_body_regs(s, "setup", 0, s["out_c"], in_dma, wt_dma, out_dma, conv2_low=0x0a0),
        common,
    )]
    windows = ((0, 5, 0x10000050), (5, 5, 0x10000050), (0, 4, 0x20000050), (4, 3, 0x20000040), (7, 3, 0x20000040))
    for idx, (y_start, input_h, conv2) in enumerate(windows):
        materialized.append(hw.patch_regs(
            hw._exact11_body_regs(s, "y_tile", 0, s["out_c"], in_dma, wt_dma, out_dma,
                                  y_start=y_start, input_h=input_h, conv2_low=conv2 & 0x0fffffff),
            common | {(hw.reg.CNA, hw.reg.CNA_CONV_CON2): conv2},
        ))
        if idx + 1 < len(windows):
            materialized.append(hw._exact11_aux_regs(s, out_dma, aux_dma))
    materialized.append(hw._exact11_aux_regs(s, out_dma, aux_dma))
    if tuple(len(row_regs) for row_regs in materialized) != hw.EXACT11_BYK_AMOUNTS:
        raise RuntimeError("clean c1280/h10/oc24 row amounts changed")
    return materialized


def _c1280_h10_oc546_rows(s: dict, rows: list[dict], in_dma: int, wt_dma: int, out_dma: int) -> list[list[int]]:
    if not _is_c1280_h10_oc546(s, rows):
        raise ValueError("c1280/h10/oc546 formula requires matching BY_K rows")
    p = hw._conv_params(s)
    common = {
        (hw.reg.CNA, hw.reg.CNA_DATA_SIZE1): 0x003f0500,
        (hw.reg.CNA, hw.reg.CNA_CBUF_CON0): 0x048,
        (hw.reg.CNA, hw.reg.CNA_CVT_CON0): 0x000b,
        (hw.reg.CNA, hw.reg.CNA_DMA_CON2): 0x003c,
        (hw.reg.DPU, hw.reg.DST_SURF_STRIDE): p["out_width_stride"] << 4,
        (hw.reg.DPU, hw.reg.SURFACE_ADD): (p["out_width_stride"] * FP16_BYTES) << 4,
    }

    def body(family: str, oc_start: int, oc_count: int, conv2: int) -> list[int]:
        regs = hw._exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma,
                                     conv2_low=conv2 & 0x0fffffff)
        patches = dict(common)
        patches[(hw.reg.CNA, hw.reg.CNA_CONV_CON2)] = conv2
        if oc_start:
            patches[(hw.reg.DPU, hw.reg.DST_BASE_ADDR)] = out_dma + oc_start * p["out_width_stride"] * FP16_BYTES
        return hw.patch_regs(regs, patches)

    aux_dma = wt_dma + hw._align_up(s["out_c"], 16) * s["in_c"] * FP16_BYTES
    materialized = [
        body("setup", 0, s["out_c"], 0x0a0),
        body("k_half", 0, 288, 0x400000a0),
        hw._exact11_aux_regs(s, out_dma, aux_dma),
        body("k_half", 288, 258, 0x400000a0),
        hw._exact11_aux_regs(s, out_dma, aux_dma),
        body("k_tile", 0, 192, 0x500000a0),
        hw._exact11_aux_regs(s, out_dma, aux_dma),
        body("k_tile", 192, 192, 0x500000a0),
        hw._exact11_aux_regs(s, out_dma, aux_dma),
        body("k_tile", 384, 162, 0x500000a0),
        hw._exact11_aux_regs(s, out_dma, aux_dma),
    ]
    if tuple(len(row_regs) for row_regs in materialized) != hw.EXACT11_BYK_AMOUNTS:
        raise RuntimeError("clean c1280/h10/oc546 row amounts changed")
    return materialized


def _c384_h19_oc64_or_96_rows(s: dict, rows: list[dict], in_dma: int, wt_dma: int, out_dma: int) -> list[list[int]]:
    if not _is_c384_h19_oc64_or_96(s, rows):
        raise ValueError("c384/h19/oc64-or-96 formula requires matching BY_K rows")
    p = hw._conv_params(s)
    hw_out_stride = hw._align_up(p["out_width_stride"], 4)
    aux_dma = wt_dma + s["out_c"] * s["in_c"] * FP16_BYTES
    common = {
        (hw.reg.CNA, hw.reg.CNA_DATA_SIZE1): 0x003f0180,
        (hw.reg.CNA, hw.reg.CNA_CBUF_CON0): 0x039,
        (hw.reg.CNA, hw.reg.CNA_CVT_CON0): 0x000b,
        (hw.reg.CNA, hw.reg.CNA_DMA_CON2): 0x011d,
        (hw.reg.DPU, hw.reg.DST_SURF_STRIDE): hw_out_stride << 4,
        (hw.reg.DPU, hw.reg.SURFACE_ADD): (hw_out_stride * FP16_BYTES) << 4,
    }

    def body(family: str, oc_start: int, oc_count: int,
             y_start: int = 0, input_h: int | None = None, conv2: int = 0x090) -> list[int]:
        regs = hw._exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma,
                                     y_start=y_start, input_h=input_h, conv2_low=conv2 & 0x0fffffff)
        patches = dict(common)
        patches[(hw.reg.CNA, hw.reg.CNA_CONV_CON2)] = conv2
        if oc_start:
            patches[(hw.reg.DPU, hw.reg.DST_BASE_ADDR)] = out_dma + oc_start * hw_out_stride * FP16_BYTES
        if y_start:
            y_off = y_start * p["out_w"] * hw.UNPACK_C2 * FP16_BYTES
            patches[(hw.reg.CNA, hw.reg.CNA_FEATURE_DATA_ADDR)] = in_dma + y_off
            patches[(hw.reg.DPU, hw.reg.DST_BASE_ADDR)] = out_dma + y_off
        return hw.patch_regs(regs, patches)

    if s["out_c"] == 64:
        materialized = [
            body("setup", 0, s["out_c"], conv2=0x090),
            body("k_half", 0, 32, conv2=0x40000090),
            hw._exact11_aux_regs(s, out_dma, aux_dma),
            body("k_half", 32, 32, conv2=0x40000090),
            hw._exact11_aux_regs(s, out_dma, aux_dma),
        ]
        for idx, (y_start, input_h, conv2) in enumerate(((0, 7, 0x20000070), (7, 6, 0x20000060), (13, 6, 0x20000060))):
            materialized.append(body("y_tile", 0, s["out_c"], y_start=y_start, input_h=input_h, conv2=conv2))
            if idx + 1 < 3:
                materialized.append(hw._exact11_aux_regs(s, out_dma, aux_dma))
        materialized.append(hw._exact11_aux_regs(s, out_dma, aux_dma))
    else:
        materialized = [
            body("setup", 0, s["out_c"], conv2=0x090),
            body("k_half", 0, 48, conv2=0x40000090),
            hw._exact11_aux_regs(s, out_dma, aux_dma),
            body("k_half", 48, 48, conv2=0x40000090),
            hw._exact11_aux_regs(s, out_dma, aux_dma),
        ]
        for oc_start in (0, 32, 64):
            materialized.append(body("k_tile", oc_start, 32, conv2=0x50000090))
            materialized.append(hw._exact11_aux_regs(s, out_dma, aux_dma))
    if tuple(len(row_regs) for row_regs in materialized) != hw.EXACT11_BYK_AMOUNTS:
        raise RuntimeError("clean c384/h19/oc64-or-96 row amounts changed")
    return materialized


def _c576_h19_oc96_rows(s: dict, rows: list[dict], in_dma: int, wt_dma: int, out_dma: int) -> list[list[int]]:
    if not _is_c576_h19_oc96(s, rows):
        raise ValueError("c576/h19/oc96 formula requires matching BY_YK rows")
    p = hw._conv_params(s)
    full_weight = s["out_c"] * s["in_c"] * FP16_BYTES
    aux_dma = wt_dma + full_weight
    hw_out_stride = hw._align_up(p["out_width_stride"], 4)
    common = {
        (hw.reg.CNA, hw.reg.CNA_DATA_SIZE1): 0x003f0240,
        (hw.reg.CNA, hw.reg.CNA_CBUF_CON0): 0x048,
        (hw.reg.CNA, hw.reg.CNA_CVT_CON0): 0x000b,
        (hw.reg.CNA, hw.reg.CNA_DMA_CON2): 0x011d,
        (hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE1): s["in_c"] * FP16_BYTES,
        (hw.reg.DPU, hw.reg.DST_SURF_STRIDE): hw_out_stride << 4,
        (hw.reg.DPU, hw.reg.SURFACE_ADD): (hw_out_stride * FP16_BYTES) << 4,
    }

    def body(family: str, oc_start: int, oc_count: int, y_start: int, input_h: int, conv2: int, cbuf0: int = 0x048) -> list[int]:
        regs = hw._exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma,
                                     y_start=y_start, input_h=input_h, conv2_low=conv2 & 0x0fffffff)
        hw_oc = hw._align_up(oc_count, hw.FP16_ATOM_ELEMENTS)
        dst_off = oc_start * hw_out_stride * FP16_BYTES + y_start * p["out_w"] * hw.UNPACK_C2 * FP16_BYTES
        return hw.patch_regs(regs, common | {
            (hw.reg.CNA, hw.reg.CNA_CONV_CON2): conv2,
            (hw.reg.CNA, hw.reg.CNA_CBUF_CON0): cbuf0,
            (hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE0): oc_count * s["in_c"] * FP16_BYTES,
            (hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE2): (s["kw"] << 24) | (s["kh"] << 16) | oc_count,
            (hw.reg.CORE, hw.reg.CORE_DATAOUT_SIZE_1): hw_oc - 1,
            (hw.reg.DPU, hw.reg.DST_BASE_ADDR): out_dma + dst_off,
            (hw.reg.DPU, hw.reg.DATA_CUBE_CHANNEL): ((oc_count - 1) << 16) | (hw_oc - 1),
            (hw.reg.DPU, hw.reg.WDMA_SIZE_0): hw_oc - 1,
        })

    materialized = [body("setup", 0, 96, 0, 11, 0x080), body("setup", 0, 96, 11, 8, 0x080, cbuf0=0x2048)]
    for family, oc_start, oc_count, conv2 in (
        ("k_half", 0, 48, 0x40000080), ("k_half", 48, 48, 0x40000080),
        ("k_tile", 0, 32, 0x50000080), ("k_tile", 32, 32, 0x50000080), ("k_tile", 64, 32, 0x50000080),
    ):
        materialized.append(body(family, oc_start, oc_count, 0, 11, conv2))
        materialized.append(body(family, oc_start, oc_count, 11, 8, conv2))
        materialized.append(hw._exact11_aux_regs(s, out_dma, aux_dma))
    if tuple(len(row_regs) for row_regs in materialized) != hw.H40_EXACT17_AMOUNTS:
        raise RuntimeError("clean c576/h19/oc96 row amounts changed")
    return materialized


def _c576_h20_oc96_rows(s: dict, rows: list[dict], in_dma: int, wt_dma: int, out_dma: int) -> list[list[int]]:
    if not _is_c576_h20_oc96(s, rows):
        raise ValueError("c576/h20/oc96 formula requires matching BY_YK rows")
    p = hw._conv_params(s)
    full_weight = s["out_c"] * s["in_c"] * FP16_BYTES
    aux_dma = wt_dma + full_weight
    common = {
        (hw.reg.CNA, hw.reg.CNA_DATA_SIZE1): 0x003f0240,
        (hw.reg.CNA, hw.reg.CNA_CVT_CON0): 0x000b,
        (hw.reg.CNA, hw.reg.CNA_DMA_CON2): 0x0140,
        (hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE0): full_weight,
        (hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE1): s["in_c"] * FP16_BYTES,
        (hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE2): (s["kw"] << 24) | (s["kh"] << 16) | s["out_c"],
        (hw.reg.CORE, hw.reg.CORE_DATAOUT_SIZE_1): hw._align_up(s["out_c"], hw.FP16_ATOM_ELEMENTS) - 1,
        (hw.reg.DPU, hw.reg.DATA_CUBE_CHANNEL): ((s["out_c"] - 1) << 16) | (hw._align_up(s["out_c"], hw.FP16_ATOM_ELEMENTS) - 1),
        (hw.reg.DPU, hw.reg.WDMA_SIZE_0): hw._align_up(s["out_c"], hw.FP16_ATOM_ELEMENTS) - 1,
        (hw.reg.DPU, hw.reg.DST_SURF_STRIDE): p["out_width_stride"] << 4,
        (hw.reg.DPU, hw.reg.SURFACE_ADD): (p["out_width_stride"] * FP16_BYTES) << 4,
    }

    def body(family: str, oc_start: int, oc_count: int, y_start: int, input_h: int, conv2: int, cbuf0: int) -> list[int]:
        regs = hw._exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma,
                                     y_start=y_start, input_h=input_h, conv2_low=conv2 & 0x0fffffff)
        hw_oc = hw._align_up(oc_count, hw.FP16_ATOM_ELEMENTS)
        dst_off = oc_start * p["out_width_stride"] * FP16_BYTES + y_start * p["out_w"] * hw.UNPACK_C2 * FP16_BYTES
        patches = dict(common)
        patches[(hw.reg.CNA, hw.reg.CNA_CONV_CON2)] = conv2
        patches[(hw.reg.CNA, hw.reg.CNA_CBUF_CON0)] = cbuf0
        patches[(hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE0)] = oc_count * s["in_c"] * FP16_BYTES
        patches[(hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE2)] = (s["kw"] << 24) | (s["kh"] << 16) | oc_count
        patches[(hw.reg.CORE, hw.reg.CORE_DATAOUT_SIZE_1)] = hw_oc - 1
        patches[(hw.reg.DPU, hw.reg.DST_BASE_ADDR)] = out_dma + dst_off
        patches[(hw.reg.DPU, hw.reg.DATA_CUBE_CHANNEL)] = ((oc_count - 1) << 16) | (hw_oc - 1)
        patches[(hw.reg.DPU, hw.reg.WDMA_SIZE_0)] = hw_oc - 1
        return hw.patch_regs(regs, patches)

    materialized = [body("setup", 0, 96, 0, 11, 0x080, 0x048), body("setup", 0, 96, 11, 9, 0x080, 0x2048)]
    for family, oc_start, oc_count, conv2 in (
        ("k_half", 0, 48, 0x40000080), ("k_half", 48, 48, 0x40000080),
        ("k_tile", 0, 32, 0x50000080), ("k_tile", 32, 32, 0x50000080), ("k_tile", 64, 32, 0x50000080),
    ):
        materialized.append(body(family, oc_start, oc_count, 0, 11, conv2, 0x048))
        materialized.append(body(family, oc_start, oc_count, 11, 9, conv2, 0x048))
        materialized.append(hw._exact11_aux_regs(s, out_dma, aux_dma))
    if tuple(len(row_regs) for row_regs in materialized) != hw.H40_EXACT17_AMOUNTS:
        raise RuntimeError("clean c576/h20/oc96 row amounts changed")
    return materialized


def _c480_h14_oc16_rows(s: dict, rows: list[dict], in_dma: int, wt_dma: int, out_dma: int) -> list[list[int]]:
    full_weight = s["out_c"] * s["in_c"] * FP16_BYTES

    def patch(regs: list[int], oc_count: int) -> list[int]:
        return hw.patch_regs(regs, {
            (hw.reg.CNA, hw.reg.CNA_DATA_SIZE1): 0x001f01e0,
            (hw.reg.CNA, hw.reg.CNA_CBUF_CON0): 0x066,
            (hw.reg.CNA, hw.reg.CNA_CVT_CON0): 0x000b,
            (hw.reg.CNA, hw.reg.CNA_DMA_CON2): 0x008c,
            (hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE0): full_weight,
            (hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE1): full_weight >> 8,
            (hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE2): (s["kw"] << 24) | (s["kh"] << 16) | s["out_c"],
            (hw.reg.CORE, hw.reg.CORE_DATAOUT_SIZE_1): oc_count - 1,
            (hw.reg.DPU, hw.reg.DATA_CUBE_CHANNEL): ((oc_count - 1) << 16) | (oc_count - 1),
            (hw.reg.DPU, hw.reg.WDMA_SIZE_0): oc_count - 1,
        })

    materialized = [patch(hw._exact11_body_regs(s, "setup", 0, s["out_c"], in_dma, wt_dma, out_dma), s["out_c"])]
    for idx, row in enumerate(rows):
        materialized.append(patch(hw._exact11_body_regs(s, "k_tile", row["k_start"], row["oc_count"], in_dma, wt_dma, out_dma), row["oc_count"]))
        if idx + 1 < len(rows):
            materialized.append(hw._exact11_aux_regs(s, out_dma))
    materialized.append(hw._exact11_aux_regs(s, out_dma))
    return materialized


def _c1024_h1_oc1001_rows(s: dict, in_dma: int, wt_dma: int, out_dma: int) -> list[list[int]]:
    common = {
        (hw.reg.CNA, hw.reg.CNA_DATA_SIZE1): 0x003f0400,
        (hw.reg.CNA, hw.reg.CNA_CBUF_CON0): 0x0b1,
        (hw.reg.CNA, hw.reg.CNA_CBUF_CON1): 0x004,
        (hw.reg.CNA, hw.reg.CNA_CVT_CON0): 0x000b,
        (hw.reg.CNA, hw.reg.CNA_DMA_CON2): 0x0ffffffd,
        (hw.reg.CNA, hw.reg.CNA_FC_DATA_SIZE1): 0x0400,
        (hw.reg.DPU, hw.reg.DST_SURF_STRIDE): 0x10,
        (hw.reg.DPU, hw.reg.SURFACE_ADD): 0x20,
    }

    def body(family: str, oc_start: int, oc_count: int, conv2: int) -> list[int]:
        regs = hw._exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma,
                                     conv2_low=conv2 & 0x0fffffff)
        patches = dict(common)
        patches[(hw.reg.CNA, hw.reg.CNA_CONV_CON2)] = conv2
        patches[(hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE0)] = oc_count * s["in_c"] * FP16_BYTES
        patches[(hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE1)] = s["in_c"] * FP16_BYTES
        patches[(hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE2)] = (s["kw"] << 24) | (s["kh"] << 16) | oc_count
        patches[(hw.reg.CORE, hw.reg.CORE_DATAOUT_SIZE_1)] = oc_count - 1
        patches[(hw.reg.DPU, hw.reg.DATA_CUBE_CHANNEL)] = ((oc_count - 1) << 16) | (oc_count - 1)
        patches[(hw.reg.DPU, hw.reg.WDMA_SIZE_0)] = oc_count - 1
        if oc_start:
            patches[(hw.reg.DPU, hw.reg.DST_BASE_ADDR)] = out_dma + oc_start * FP16_BYTES
        return hw.patch_regs(regs, patches)

    aux_dma = wt_dma + s["out_c"] * s["in_c"] * FP16_BYTES
    return [
        body("setup", 0, s["out_c"], 0x020),
        body("k_half", 0, 512, 0x40000020), hw._exact11_aux_regs(s, out_dma, aux_dma),
        body("k_half", 512, 489, 0x40000020), hw._exact11_aux_regs(s, out_dma, aux_dma),
        body("k_tile", 0, 320, 0x50000020), hw._exact11_aux_regs(s, out_dma, aux_dma),
        body("k_tile", 320, 320, 0x50000020), hw._exact11_aux_regs(s, out_dma, aux_dma),
        body("k_tile", 640, 361, 0x50000020), hw._exact11_aux_regs(s, out_dma, aux_dma),
    ]


def _c1024_h7_oc1024_rows(s: dict, in_dma: int, wt_dma: int, out_dma: int) -> list[list[int]]:
    p = hw._conv_params(s)
    hw_out_stride = hw._align_up(p["out_width_stride"], 4)
    common = {
        (hw.reg.CNA, hw.reg.CNA_DATA_SIZE1): 0x003f0400,
        (hw.reg.CNA, hw.reg.CNA_CBUF_CON0): 0x084,
        (hw.reg.CNA, hw.reg.CNA_CVT_CON0): 0x000b,
        (hw.reg.CNA, hw.reg.CNA_DMA_CON2): 0x0015,
        (hw.reg.CNA, hw.reg.CNA_FC_DATA_SIZE1): 0x0400,
        (hw.reg.DPU, hw.reg.DST_SURF_STRIDE): hw_out_stride << 4,
        (hw.reg.DPU, hw.reg.SURFACE_ADD): (hw_out_stride * FP16_BYTES) << 4,
    }

    def body(family: str, oc_start: int, oc_count: int, conv2: int) -> list[int]:
        regs = hw._exact11_body_regs(s, family, oc_start, oc_count, in_dma, wt_dma, out_dma,
                                     conv2_low=conv2 & 0x0fffffff)
        patches = dict(common)
        patches[(hw.reg.CNA, hw.reg.CNA_CONV_CON2)] = conv2
        patches[(hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE0)] = oc_count * s["in_c"] * FP16_BYTES
        patches[(hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE1)] = s["in_c"] * FP16_BYTES
        patches[(hw.reg.CNA, hw.reg.CNA_WEIGHT_SIZE2)] = (s["kw"] << 24) | (s["kh"] << 16) | oc_count
        patches[(hw.reg.CORE, hw.reg.CORE_DATAOUT_SIZE_1)] = oc_count - 1
        patches[(hw.reg.DPU, hw.reg.DATA_CUBE_CHANNEL)] = ((oc_count - 1) << 16) | (oc_count - 1)
        patches[(hw.reg.DPU, hw.reg.WDMA_SIZE_0)] = oc_count - 1
        if oc_start:
            patches[(hw.reg.DPU, hw.reg.DST_BASE_ADDR)] = out_dma + oc_start * hw_out_stride * FP16_BYTES
        return hw.patch_regs(regs, patches)

    aux_dma = wt_dma + s["out_c"] * s["in_c"] * FP16_BYTES
    return [
        body("setup", 0, s["out_c"], 0x080),
        body("k_half", 0, 512, 0x40000080), hw._exact11_aux_regs(s, out_dma, aux_dma),
        body("k_half", 512, 512, 0x40000080), hw._exact11_aux_regs(s, out_dma, aux_dma),
        body("k_tile", 0, 352, 0x50000080), hw._exact11_aux_regs(s, out_dma, aux_dma),
        body("k_tile", 352, 336, 0x50000080), hw._exact11_aux_regs(s, out_dma, aux_dma),
        body("k_tile", 688, 336, 0x50000080), hw._exact11_aux_regs(s, out_dma, aux_dma),
    ]


def _run_c1024_h7_oc1024(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    hw_out_stride = hw._align_up(p["out_width_stride"], 4)
    layout = hw.exact_byk_legacy_layout_check(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw._pack_pointwise_compact_weight(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * hw_out_stride * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 8192, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = _c1024_h7_oc1024_rows(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, regs, layout)
        if hw.npu_submit(fd, task_mem.obj_addr, 3, core_mask=0,
                         subcores=((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], hw_out_stride, hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "c1024_h7_oc1024_exact11"


def _run_c1024_h1_oc1001(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    layout = hw.exact_byk_legacy_layout_check(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw._pack_pointwise_compact_weight(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 8192, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = _c1024_h1_oc1001_rows(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, regs, layout)
        if hw.npu_submit(fd, task_mem.obj_addr, 3, core_mask=0,
                         subcores=((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "c1024_h1_oc1001_exact11"


def _run_c480_h14_oc16(s: dict, rows: list[dict], inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    layout = hw.exact_byk_legacy_layout_check(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw.pack_weights(wt, s, p).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 8192, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = _c480_h14_oc16_rows(s, rows, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, regs, layout)
        if hw.npu_submit(fd, task_mem.obj_addr, 3, core_mask=0,
                         subcores=((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "c480_h14_oc16_exact11"


def _run_c512_h14_oc24(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    layout = hw.exact_byk_legacy_layout_check(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw._pack_pointwise_compact_weight(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 8192, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = _c512_h14_oc24_rows(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, regs, layout)
        if hw.npu_submit(fd, task_mem.obj_addr, 3, core_mask=0,
                         subcores=((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "c512_h14_oc24_exact11"


def _run_c1280_h10_oc24(s: dict, rows: list[dict], inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    layout = hw.exact_byk_legacy_layout_check(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw._pack_pointwise_compact_weight(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 8192, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = _c1280_h10_oc24_rows(s, rows, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, regs, layout)
        if hw.npu_submit(fd, task_mem.obj_addr, 3, core_mask=0,
                         subcores=((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "c1280_h10_oc24_ywindow_byk"


def _run_c1280_h10_oc546(s: dict, rows: list[dict], inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    layout = hw.exact_byk_legacy_layout_check(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw._pack_pointwise_compact_weight(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 8192, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = _c1280_h10_oc546_rows(s, rows, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, regs, layout)
        if hw.npu_submit(fd, task_mem.obj_addr, 3, core_mask=0,
                         subcores=((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "c1280_h10_oc546_compact"


def _run_c384_h19_oc64_or_96(s: dict, rows: list[dict], inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    hw_out_stride = hw._align_up(p["out_width_stride"], 4)
    layout = hw.exact_byk_legacy_layout_check(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw.pack_weights(wt, s, p).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * hw_out_stride * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 8192, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = _c384_h19_oc64_or_96_rows(s, rows, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, regs, layout)
        if hw.npu_submit(fd, task_mem.obj_addr, 3, core_mask=0,
                         subcores=((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], hw_out_stride, hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "c384_h19_oc64_or_96_exact11"


def _run_c576_h19_oc96(s: dict, rows: list[dict], inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    hw_out_stride = hw._align_up(p["out_width_stride"], 4)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw.pack_weights(wt, s, p).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * hw_out_stride * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 16384, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = _c576_h19_oc96_rows(s, rows, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_h40_exact17_tasks(task_map, regcmd_map, regcmd_mem, regs)
        if hw.npu_submit(fd, task_mem.obj_addr, 6, core_mask=0,
                         subcores=((0, 2), (0, 2), (0, 2), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], hw_out_stride, hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "c576_h19_oc96_exact17"


def _run_c576_h20_oc96(s: dict, rows: list[dict], inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw.pack_weights(wt, s, p).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 16384, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = _c576_h20_oc96_rows(s, rows, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_h40_exact17_tasks(task_map, regcmd_map, regcmd_mem, regs)
        if hw.npu_submit(fd, task_mem.obj_addr, 6, core_mask=0,
                         subcores=((0, 2), (0, 2), (0, 2), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "c576_h20_oc96_exact17"


def _run_c768_h20_oc96(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    base = dict(s, name="b1_c1280_h10_w10_oc546_wic1280_k1x1_g1_s1_pvalid",
                in_c=1280, weight_in_c=1280, out_c=546, in_h=10, in_w=10)
    base_rows = _descriptor_rows(base)
    tile_wt = np.zeros((546, 1280, 1, 1), dtype=np.float16)
    tile_wt[:s["out_c"], :s["in_c"]] = wt
    flat_in = inp.reshape(1, s["in_c"], p["out_h"] * p["out_w"])
    got_flat = np.zeros((s["out_c"], p["out_h"] * p["out_w"]), dtype=np.float16)
    tasks = 0
    for pos in range(0, p["out_h"] * p["out_w"], 100):
        count = min(100, p["out_h"] * p["out_w"] - pos)
        tile_in = np.zeros((1, 1280, 10, 10), dtype=np.float16)
        tile_in.reshape(1, 1280, 100)[0, :s["in_c"], :count] = flat_in[0, :, pos:pos + count]
        tile, tile_tasks, _path = _run_c1280_h10_oc546(base, base_rows, tile_in, tile_wt)
        got_flat[:, pos:pos + count] = tile.reshape(1, 546, 100)[0, :s["out_c"], :count]
        tasks += tile_tasks
    got = got_flat.reshape(s["out_c"], p["out_h"], p["out_w"])
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), tasks, "c768_h20_oc96_tiled_c1280_h10"


def _run_c768_or_c960_h10_oc120(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    wide = dict(s, name="b1_c1280_h10_w10_oc546_wic1280_k1x1_g1_s1_pvalid", in_c=1280, weight_in_c=1280, out_c=546)
    wide_in = np.zeros((1, 1280, s["in_h"], s["in_w"]), dtype=np.float16)
    wide_in[:, :s["in_c"]] = inp
    wide_wt = np.zeros((546, 1280, 1, 1), dtype=np.float16)
    wide_wt[:s["out_c"], :s["in_c"]] = wt
    got, tasks, _path = _run_c1280_h10_oc546(wide, _descriptor_rows(wide), wide_in, wide_wt)
    return got[:, :s["out_c"]], tasks, "c768_or_c960_h10_oc120_via_c1280_oc546"


def _run_c320_h20_oc72(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    wide = dict(s, name="b1_c576_h20_w20_oc96_wic576_k1x1_g1_s1_pvalid", in_c=576, weight_in_c=576, out_c=96)
    wide_in = np.zeros((1, 576, s["in_h"], s["in_w"]), dtype=np.float16)
    wide_in[:, :s["in_c"]] = inp
    wide_wt = np.zeros((96, 576, s["kh"], s["kw"]), dtype=np.float16)
    wide_wt[:s["out_c"], :s["in_c"]] = wt
    got, tasks, _path = _run_c576_h20_oc96(wide, _descriptor_rows(wide), wide_in, wide_wt)
    return got[:, :s["out_c"]], tasks, "c320_h20_oc72_via_c576_oc96_exact17"


def _run_c288_h20_oc72(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    wide = dict(s, name="b1_c576_h20_w20_oc96_wic576_k1x1_g1_s1_pvalid", in_c=576, weight_in_c=576, out_c=96)
    wide_in = np.zeros((1, 576, s["in_h"], s["in_w"]), dtype=np.float16)
    wide_in[:, :s["in_c"]] = inp
    wide_wt = np.zeros((96, 576, s["kh"], s["kw"]), dtype=np.float16)
    wide_wt[:s["out_c"], :s["in_c"]] = wt
    got, tasks, _path = _run_c576_h20_oc96(wide, _descriptor_rows(wide), wide_in, wide_wt)
    return got[:, :s["out_c"]], tasks, "c288_h20_oc72_via_c576_oc96_exact17"


def _run_c72_h20_oc576(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    got = np.zeros((1, s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    wide = dict(s, name="b1_c576_h20_w20_oc96_wic576_k1x1_g1_s1_pvalid", in_c=576, weight_in_c=576, out_c=96)
    wide_in = np.zeros((1, 576, s["in_h"], s["in_w"]), dtype=np.float16)
    wide_in[:, :s["in_c"]] = inp
    wide_rows = _descriptor_rows(wide)
    tasks = 0
    for oc_start in range(0, s["out_c"], 96):
        oc_count = min(96, s["out_c"] - oc_start)
        wide_wt = np.zeros((96, 576, s["kh"], s["kw"]), dtype=np.float16)
        wide_wt[:oc_count, :s["in_c"]] = wt[oc_start:oc_start + oc_count]
        tile, tile_tasks, _path = _run_c576_h20_oc96(wide, wide_rows, wide_in, wide_wt)
        got[:, oc_start:oc_start + oc_count] = tile[:, :oc_count]
        tasks += tile_tasks
    return got, tasks, "c72_h20_oc576_via_c576_oc96_exact17"


def _run_c576_h20_oc72(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    wide = dict(s, name="b1_c576_h20_w20_oc96_wic576_k1x1_g1_s1_pvalid", out_c=96)
    wide_wt = np.zeros((96, s["weight_in_c"], s["kh"], s["kw"]), dtype=np.float16)
    wide_wt[:s["out_c"]] = wt
    got, tasks, _path = _run_c576_h20_oc96(wide, _descriptor_rows(wide), inp, wide_wt)
    return got[:, :s["out_c"]], tasks, "c576_h20_oc72_via_oc96_exact17"


def _write_direct_spatial_tasks(task_map, regcmd_map, regcmd_mem, task_regs) -> None:
    tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), ctypes.POINTER(hw.struct_rknpu_task))
    regcmd = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), ctypes.POINTER(ctypes.c_uint64))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(task_map)), 0, len(task_map))
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(regcmd_map)), 0, len(regcmd_map))
    offsets = []
    offset = 0
    for regs in task_regs:
        offsets.append(offset)
        offset += hw6._align_up(len(regs) + hw6.PC_CHAIN_TAIL_QWORDS, 2)
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            regcmd[base + i] = qword
        if idx + 1 < len(task_regs):
            next_addr = regcmd_mem.dma_addr + offsets[idx + 1] * ctypes.sizeof(ctypes.c_uint64)
            amount = (getattr(regs, "pc_core", 0) << 16) | (hw6._ceil_div(len(task_regs[idx + 1]), 2) + 1)
            tails = [
                hw6.E(hw6.reg.PC_REG, hw6.reg.PC_BASE_ADDRESS, next_addr & 0xfffffff0),
                hw6.E(hw6.reg.PC_REG, hw6.reg.PC_REGISTER_AMOUNTS, amount),
                hw6.E(hw6.reg.VERSION, 0, 0),
                hw6.E(hw6.reg.PC, hw6.reg.OPERATION_ENABLE, (6 << 1) | 1),
            ]
        else:
            tails = [
                hw6.E(0x0001, 0, 0),
                hw6.E(hw6.reg.PC_REG, hw6.reg.PC_REGISTER_AMOUNTS, 0),
                hw6.E(hw6.reg.VERSION, 0, 0),
                hw6.E(hw6.reg.PC, hw6.reg.OPERATION_ENABLE, (6 << 1) | 1),
            ]
        for i, qword in enumerate(tails):
            regcmd[base + len(regs) + i] = qword
        tasks[idx].op_idx = 0
        tasks[idx].enable_mask = 0xd
        tasks[idx].int_mask = 0x300
        tasks[idx].int_clear = 0x1ffff
        tasks[idx].regcfg_amount = len(regs) + len(tails)
        tasks[idx].regcmd_addr = regcmd_mem.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)


def _run_c72_h20_oc288_k3(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    patches = np.zeros((s["in_c"] * s["kh"] * s["kw"], p["out_h"] * p["out_w"]), dtype=np.float16)
    dst_c = 0
    for ic in range(s["in_c"]):
        for ky in range(s["kh"]):
            for kx in range(s["kw"]):
                patches[dst_c] = inp[0, ic, ky:ky + p["out_h"], kx:kx + p["out_w"]].reshape(-1)
                dst_c += 1
    flat_wt = wt.reshape(s["out_c"], s["in_c"] * s["kh"] * s["kw"], 1, 1)
    base = dict(s, name="b1_c1280_h10_w10_oc546_wic1280_k1x1_g1_s1_pvalid",
                in_c=1280, weight_in_c=1280, out_c=546, in_h=10, in_w=10, kh=1, kw=1)
    base_rows = _descriptor_rows(base)
    got_flat = np.zeros((s["out_c"], p["out_h"] * p["out_w"]), dtype=np.float16)
    tasks = 0
    for pos in range(0, p["out_h"] * p["out_w"], 100):
        count = min(100, p["out_h"] * p["out_w"] - pos)
        tile_in = np.zeros((1, 1280, 10, 10), dtype=np.float16)
        tile_in.reshape(1, 1280, 100)[0, :patches.shape[0], :count] = patches[:, pos:pos + count]
        tile_wt = np.zeros((546, 1280, 1, 1), dtype=np.float16)
        tile_wt[:s["out_c"], :flat_wt.shape[1]] = flat_wt
        tile, tile_tasks, _path = _run_c1280_h10_oc546(base, base_rows, tile_in, tile_wt)
        got_flat[:, pos:pos + count] = tile.reshape(1, 546, 100)[0, :s["out_c"], :count]
        tasks += tile_tasks
    got = got_flat.reshape(s["out_c"], p["out_h"], p["out_w"])
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), tasks, "c72_h20_oc288_im2col_c1280_h10"


def _run_c72_h20_oc288_k3_direct_spatial(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p6 = hw6._conv_params(s["in_c"], s["in_h"], s["in_w"], s["out_c"], s["kh"], s["kw"], s["groups"], s.get("stride", 1))
    descs = hw6.plan_observed_spatial_tile_descs(p6, s["in_c"], s["out_c"], s["kh"], s["kw"], s["in_h"], s["in_w"], s["groups"], s.get("stride", 1))
    input_flat = hw6._pack_cdma_dc_feature_input_fp16(inp[0], p6).view(np.uint16)
    weight_flat = hw6.pack_weights(wt.reshape(s["out_c"], s["in_c"], s["kh"], s["kw"]),
                                   s["out_c"], s["in_c"], s["kh"], s["kw"], p6["align_c"], s["groups"]).view(np.uint16)
    out_count = hw6._ceil_div(p6["align_out_c"], hw6.UNPACK_C2) * p6["out_width_stride"] * hw6.UNPACK_C2

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, 8192, hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = []
        for idx, desc in enumerate(descs):
            desc.extra = {"full_data_bank": True}
            desc.pc_core = hw6._rknn_spatial_pc_core(desc.family)
            row = hw6.RegList(hw6.make_rknn_full_conv2d_regs_from_desc(p6, desc, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr))
            row.extend(hw6.E(hw6.reg.DPU, clear_reg, 0) for clear_reg in hw6.DPU_CLEAR_REGS)
            row.pc_core = desc.pc_core
            regs.append(row)
        _write_direct_spatial_tasks(task_map, regcmd_map, regcmd_mem, regs)
        if hw.npu_submit(fd, task_mem.obj_addr, len(regs)) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p6["out_h"], p6["out_w"], p6["out_width_stride"], hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p6["out_h"], p6["out_w"]), len(regs), "c72_h20_oc288_direct_spatial"


def _run_c576_h19_oc12(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    wide = dict(s, name="b1_c576_h19_w19_oc96_wic576_k1x1_g1_s1_pvalid", out_c=96)
    wide_wt = np.zeros((96, s["weight_in_c"], s["kh"], s["kw"]), dtype=np.float16)
    wide_wt[:s["out_c"]] = wt
    got, tasks, _path = _run_c576_h19_oc96(wide, _descriptor_rows(wide), inp, wide_wt)
    return got[:, :s["out_c"]], tasks, "c576_h19_oc12_via_oc96_exact17"


def _run_c576_h19_oc273(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    got = np.zeros((1, s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    wide = dict(s, name="b1_c576_h19_w19_oc96_wic576_k1x1_g1_s1_pvalid", out_c=96)
    wide_rows = _descriptor_rows(wide)
    tasks = 0
    for oc_start in range(0, s["out_c"], 96):
        oc_count = min(96, s["out_c"] - oc_start)
        wide_wt = np.zeros((96, s["weight_in_c"], s["kh"], s["kw"]), dtype=np.float16)
        wide_wt[:oc_count] = wt[oc_start:oc_start + oc_count]
        tile, tile_tasks, _path = _run_c576_h19_oc96(wide, wide_rows, inp, wide_wt)
        got[:, oc_start:oc_start + oc_count] = tile[:, :oc_count]
        tasks += tile_tasks
    return got, tasks, "c576_h19_oc273_via_oc96_exact17"


def _run_c256_h3_oc546(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    got = np.zeros((1, s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    base = dict(s, name="b1_c256_h3_w3_oc24_wic256_k1x1_g1_s1_pvalid", out_c=24)
    tasks = 0
    for oc_start in range(0, s["out_c"], 24):
        oc_count = min(24, s["out_c"] - oc_start)
        tile_wt = np.zeros((24, s["weight_in_c"], s["kh"], s["kw"]), dtype=np.float16)
        tile_wt[:oc_count] = wt[oc_start:oc_start + oc_count]
        tile, tile_tasks, _path = _run_pointwise_chain_compact(base, inp, tile_wt)
        got[:, oc_start:oc_start + oc_count] = tile[:, :oc_count]
        tasks += tile_tasks
    return got, tasks, "c256_h3_oc546_via_oc24_chain"


def _run_c256_h2_oc546(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    got = np.zeros((1, s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    base = dict(s, name="b1_c256_h2_w2_oc24_wic256_k1x1_g1_s1_pvalid", out_c=24)
    tasks = 0
    for oc_start in range(0, s["out_c"], 24):
        oc_count = min(24, s["out_c"] - oc_start)
        tile_wt = np.zeros((24, s["weight_in_c"], s["kh"], s["kw"]), dtype=np.float16)
        tile_wt[:oc_count] = wt[oc_start:oc_start + oc_count]
        tile, tile_tasks, _path = _run_setup108_compact(base, inp, tile_wt)
        got[:, oc_start:oc_start + oc_count] = tile[:, :oc_count]
        tasks += tile_tasks
    return got, tasks, "c256_h2_oc546_via_oc24_setup108"


def _run_c128_h1_oc24(s: dict, inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    layout = hw.exact_byk_legacy_layout_check(s)
    input_flat = hw.pack_input(inp[0], p).view(np.uint16)
    weight_flat = hw._pack_pointwise_compact_weight(wt, s["out_c"], s["in_c"]).view(np.uint16)
    out_count = hw._ceil_div(p["align_out_c"], hw.UNPACK_C2) * p["out_width_stride"] * hw.UNPACK_C2
    regcmd_bytes = (layout["offsets"][-1] + layout["amounts"][-1]) * ctypes.sizeof(ctypes.c_uint64)

    fd = os.open("/dev/dri/card1", os.O_RDWR)
    task_map, task_mem = hw.mem_allocate(fd, 4096, hw.RKNPU_MEM_KERNEL_MAPPING | hw.RKNPU_MEM_NON_CACHEABLE)
    regcmd_map, regcmd_mem = hw.mem_allocate(fd, hw.regcmd_alloc_bytes(regcmd_bytes), hw.RKNPU_MEM_NON_CACHEABLE)
    input_map, input_mem = hw.mem_allocate(fd, max(4096, input_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    weight_map, weight_mem = hw.mem_allocate(fd, max(4096, weight_flat.nbytes), hw.RKNPU_MEM_NON_CACHEABLE)
    output_map, output_mem = hw.mem_allocate(fd, max(4096, out_count * FP16_BYTES), hw.RKNPU_MEM_NON_CACHEABLE)
    try:
        (ctypes.c_uint16 * len(input_flat)).from_buffer(input_map)[:] = input_flat.tolist()
        (ctypes.c_uint16 * len(weight_flat)).from_buffer(weight_map)[:] = weight_flat.tolist()
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(output_map)), 0, output_map.size())
        regs = _c128_h1_oc24_rows(s, input_mem.dma_addr, weight_mem.dma_addr, output_mem.dma_addr)
        hw.write_exact11_byk_tasks(task_map, regcmd_map, regcmd_mem, regs, layout)
        if hw.npu_submit(fd, task_mem.obj_addr, 3, core_mask=0,
                         subcores=((0, 1), (0, 1), (0, 1), (0, 0), (0, 0))) < 0:
            raise RuntimeError("npu_submit failed")
        out_raw = np.frombuffer(output_map, dtype=np.float16, count=out_count).copy()
        hw.post_submit_reset(fd)
    finally:
        hw.close_allocations(fd, ((task_map, task_mem), (regcmd_map, regcmd_mem),
                                  (input_map, input_mem), (weight_map, weight_mem), (output_map, output_mem)))
        os.close(fd)
    got = hw.unpack_output(out_raw, s["out_c"], p["out_h"], p["out_w"], p["out_width_stride"], hw.UNPACK_C2)
    return got.reshape(1, s["out_c"], p["out_h"], p["out_w"]), len(regs), "c128_h1_oc24_setup3"


def _setup_rows_for_yk(rows: list[dict]) -> list[dict]:
    setup_rows = [row for row in rows if row["family"] == "setup"]
    return setup_rows if setup_rows else rows


def _run_yk_setup_local(s: dict, rows: list[dict], inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    if s["batch"] != 1 or s["groups"] != 1:
        return _run_grouped_full(s, inp, wt)
    p = hw._conv_params(s)
    got = np.zeros((1, s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    submit_rows = _setup_rows_for_yk(rows)
    for row in submit_rows:
        tile = _local_tile(s, inp[0], wt, row["y_start"], row["input_h"], row["k_start"], row["oc_count"])
        got[0, row["k_start"]:row["k_start"] + row["oc_count"], row["y_start"]:row["y_start"] + tile.shape[1]] = tile
    return got, len(submit_rows), "by_yk_setup_local"


def _depthwise_rows(s: dict, rows: list[dict]) -> list[dict]:
    split, _families = _split_summary(rows)
    if split == "BY_Y":
        return [dict(row, k_start=0, oc_count=s["out_c"]) for row in rows]
    if split == "BY_YK":
        return _setup_rows_for_yk(rows)
    return rows


def _run_depthwise_serial(s: dict, rows: list[dict], inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    p = hw._conv_params(s)
    got = np.zeros((s["batch"], s["out_c"], p["out_h"], p["out_w"]), dtype=np.float16)
    submit_rows = _depthwise_rows(s, rows)
    tasks = 0
    for n in range(s["batch"]):
        for row in submit_rows:
            y_start = row["y_start"]
            for channel in range(row["k_start"], row["k_start"] + row["oc_count"]):
                tile_shape = dict(
                    s,
                    name=s["name"] + "_depthwise_serial",
                    batch=1,
                    in_c=1,
                    out_c=MIN_HW_OC,
                    weight_in_c=1,
                    groups=1,
                    in_h=row["input_h"],
                )
                tile_wt = np.zeros((MIN_HW_OC, 1, s["kh"], s["kw"]), dtype=np.float16)
                tile_wt[0] = wt[channel]
                tile_in = inp[n, channel:channel + 1, y_start:y_start + row["input_h"], :]
                tile = _run_hw_single(tile_shape, tile_in, tile_wt)[0]
                got[n, channel, y_start:y_start + tile.shape[0]] = tile
                tasks += 1
    return got, tasks, "depthwise_serial"


def _schedule_name(s: dict, rows: list[dict]) -> str:
    split, families = _split_summary(rows)
    if _is_depthwise(s):
        return "depthwise_serial"
    if s["groups"] != 1 or s["batch"] != 1:
        return "grouped_full"
    if split == "NONE" and families == {"setup"}:
        if _is_c128_h1_oc24(s, rows):
            return "c128_h1_oc24_setup3"
        if _is_c512_h14_oc24(s):
            return "c512_h14_oc24_exact11"
        if s["name"] in getattr(hw, "POINTWISE_EXACT11_CHAIN_COMPACT_WEIGHT_SHAPES", set()):
            return "pointwise_chain_compact"
        if s["name"] in getattr(hw, "POINTWISE_SETUP108_COMPACT_WEIGHT_SHAPES", set()):
            return "setup108_compact"
        return "direct"
    if split == "BY_Y":
        if _is_c576_h19_oc12(s, rows):
            return "c576_h19_oc12_exact12"
        if s["name"] in getattr(hw, "POINTWISE_YK_SHAPES", set()):
            return "local_tile_replay"
        if _is_c8_h160_oc16_k3(s):
            return "h160_setup3_padded"
        return "by_y_local"
    if split == "BY_K":
        if s["name"] in getattr(hw, "POINTWISE_SETUP108_COMPACT_WEIGHT_SHAPES", set()):
            return "setup108_compact"
        if _is_c320_h20_oc72(s, rows):
            return "c320_h20_oc72_via_c576_oc96_exact17"
        if _is_c288_h20_oc72(s, rows):
            return "c288_h20_oc72_via_c576_oc96_exact17"
        if _is_c72_h20_oc576(s, rows):
            return "c72_h20_oc576_via_c576_oc96_exact17"
        if _is_c72_h20_oc288_k3(s, rows):
            return "c72_h20_oc288_direct_spatial"
        if _is_c768_or_c960_h10_oc120(s, rows):
            return "c768_or_c960_h10_oc120_via_c1280_oc546"
        if _is_c40_h40_oc160_k3(s):
            return "h40_exact17_padded"
        if _is_c256_h2_oc546(s, rows):
            return "c256_h2_oc546_via_oc24_setup108"
        if _is_c256_h3_oc546(s, rows):
            return "c256_h3_oc546_via_oc24_chain"
        if _is_c1280_h10_oc24(s, rows):
            return "c1280_h10_oc24_ywindow_byk"
        if _is_c1280_h10_oc546(s, rows):
            return "c1280_h10_oc546_compact"
        if _is_c384_h19_oc64_or_96(s, rows):
            return "c384_h19_oc64_or_96_exact11"
        if _is_c480_h14_oc16(s):
            return "c480_h14_oc16_exact11"
        if _is_c1024_h1_oc1001(s):
            return "c1024_h1_oc1001_exact11"
        if _is_c1024_h7_oc1024(s):
            return "c1024_h7_oc1024_exact11"
        if s["name"] in getattr(hw, "POINTWISE_EXACT11_BYK_SHAPES", set()):
            return "pointwise_exact11_byk"
        if s["name"] in getattr(hw, "PREFIX_BY_K_SHAPES", set()):
            return "prefix_exact11_byk"
        return "by_k_local"
    if split == "BY_YK":
        if _is_c576_h19_oc273(s, rows):
            return "c576_h19_oc273_via_oc96_exact17"
        if _is_c576_h20_oc72(s, rows):
            return "c576_h20_oc72_via_oc96_exact17"
        if _is_c768_h20_oc96(s, rows):
            return "c768_h20_oc96_tiled_c1280_h10"
        if _is_c576_h20_oc96(s, rows):
            return "c576_h20_oc96_exact17"
        if _is_c576_h19_oc96(s, rows):
            return "c576_h19_oc96_exact17"
        return "by_yk_setup_local"
    return "unsupported"


def _run_schedule(s: dict, rows: list[dict], inp: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, int, str]:
    kind = _schedule_name(s, rows)
    if kind == "depthwise_serial":
        return _run_depthwise_serial(s, rows, inp, wt)
    if kind in {"direct", "grouped_full"}:
        return _run_grouped_full(s, inp, wt)
    if kind == "c512_h14_oc24_exact11":
        return _run_c512_h14_oc24(s, inp, wt)
    if kind == "c128_h1_oc24_setup3":
        return _run_c128_h1_oc24(s, inp, wt)
    if kind == "setup108_compact":
        return _run_setup108_compact(s, inp, wt)
    if kind == "pointwise_chain_compact":
        return _run_pointwise_chain_compact(s, inp, wt)
    if kind == "c1280_h10_oc24_ywindow_byk":
        return _run_c1280_h10_oc24(s, rows, inp, wt)
    if kind == "c1280_h10_oc546_compact":
        return _run_c1280_h10_oc546(s, rows, inp, wt)
    if kind == "c384_h19_oc64_or_96_exact11":
        return _run_c384_h19_oc64_or_96(s, rows, inp, wt)
    if kind == "c576_h19_oc96_exact17":
        return _run_c576_h19_oc96(s, rows, inp, wt)
    if kind == "c576_h20_oc96_exact17":
        return _run_c576_h20_oc96(s, rows, inp, wt)
    if kind == "c768_h20_oc96_tiled_c1280_h10":
        return _run_c768_h20_oc96(s, inp, wt)
    if kind == "c768_or_c960_h10_oc120_via_c1280_oc546":
        return _run_c768_or_c960_h10_oc120(s, inp, wt)
    if kind == "c320_h20_oc72_via_c576_oc96_exact17":
        return _run_c320_h20_oc72(s, inp, wt)
    if kind == "c288_h20_oc72_via_c576_oc96_exact17":
        return _run_c288_h20_oc72(s, inp, wt)
    if kind == "c72_h20_oc576_via_c576_oc96_exact17":
        return _run_c72_h20_oc576(s, inp, wt)
    if kind == "c576_h20_oc72_via_oc96_exact17":
        return _run_c576_h20_oc72(s, inp, wt)
    if kind == "c72_h20_oc288_direct_spatial":
        return _run_c72_h20_oc288_k3(s, inp, wt)
    if kind == "c576_h19_oc12_exact12":
        return _run_c576_h19_oc12(s, inp, wt)
    if kind == "c576_h19_oc273_via_oc96_exact17":
        return _run_c576_h19_oc273(s, inp, wt)
    if kind == "c256_h3_oc546_via_oc24_chain":
        return _run_c256_h3_oc546(s, inp, wt)
    if kind == "c256_h2_oc546_via_oc24_setup108":
        return _run_c256_h2_oc546(s, inp, wt)
    if kind == "by_y_local":
        return _run_y_local(s, rows, inp, wt)
    if kind == "by_y_small_local":
        return _run_y_small_local(s, inp, wt)
    if kind == "pointwise_yk":
        return _run_pointwise_yk(s, rows, inp, wt)
    if kind == "local_tile_replay":
        return _run_local_tile_replay(s, inp, wt)
    if kind == "h160_setup3_padded":
        return _run_h160_setup3_padded(s, inp, wt)
    if kind == "by_k_local":
        return _run_k_local(s, rows, inp, wt)
    if kind == "pointwise_exact11_byk":
        return _run_pointwise_exact11_byk(s, inp, wt)
    if kind == "prefix_exact11_byk":
        return _run_prefix_exact11_byk(s, inp, wt)
    if kind == "h40_exact17_padded":
        return _run_h40_exact17_padded(s, inp, wt)
    if kind == "c1024_h7_oc1024_exact11":
        return _run_c1024_h7_oc1024(s, inp, wt)
    if kind == "c1024_h1_oc1001_exact11":
        return _run_c1024_h1_oc1001(s, inp, wt)
    if kind == "c480_h14_oc16_exact11":
        return _run_c480_h14_oc16(s, rows, inp, wt)
    if kind == "by_yk_setup_local":
        return _run_yk_setup_local(s, rows, inp, wt)
    raise ValueError(f"unsupported planner output: {kind}")


def run_shape(s: dict, dry_run: bool = False) -> None:
    rows = _descriptor_rows(s)
    split, families = _split_summary(rows)
    kind = _schedule_name(s, rows)
    print(f"shape={s['name']} split={split} families={','.join(sorted(families))} rows={len(rows)} schedule={kind}")
    if dry_run:
        _print_rows(rows)
        return
    inp, wt, expected = _random_case(s)
    got, tasks, path = _run_schedule(s, rows, inp, wt)
    _compare_and_print(s, got, expected, path, tasks)


def list_shapes() -> None:
    runnable = 0
    fenced = 0
    for s in SHAPES:
        try:
            rows = _descriptor_rows(s)
            split, families = _split_summary(rows)
            kind = _schedule_name(s, rows)
            if kind == "unsupported":
                fenced += 1
            else:
                runnable += 1
            print(f"{s['name']} {split} {','.join(sorted(families))} rows={len(rows)} schedule={kind}")
        except Exception as exc:
            fenced += 1
            print(f"{s['name']} FENCED reason={type(exc).__name__}:{exc}")
    print(f"runnable {runnable}")
    print(f"fenced {fenced}")


def sweep_shapes(limit: int = 0, start: int = 1, pattern: str = "", stop_on_error: bool = False) -> None:
    selected = []
    for idx, s in enumerate(SHAPES, 1):
        if idx < start:
            continue
        if pattern and not re.search(pattern, s["name"]):
            continue
        selected.append((idx, s))
        if limit and len(selected) >= limit:
            break

    counts = {"PASS": 0, "FAIL": 0, "ERROR": 0}
    started = time.time()
    for idx, s in selected:
        print(f"[{idx:3d}/{len(SHAPES)}] {s['name']} ...", flush=True)
        try:
            run_shape(dict(s))
            counts["PASS"] += 1
        except AssertionError as exc:
            counts["FAIL"] += 1
            print(f"           FAIL {exc}", flush=True)
            if stop_on_error:
                break
        except Exception as exc:
            counts["ERROR"] += 1
            print(f"           ERROR {type(exc).__name__}: {exc}", flush=True)
            if stop_on_error:
                break
        elapsed = time.time() - started
        print(f"           counts PASS={counts['PASS']} FAIL={counts['FAIL']} ERROR={counts['ERROR']} elapsed={elapsed:.1f}s", flush=True)
    print(f"PASS={counts['PASS']} FAIL={counts['FAIL']} ERROR={counts['ERROR']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact first-principles RK3588 FP16 CONV runner")
    parser.add_argument("shape", nargs="?", help="shape name to run")
    parser.add_argument("--dry-run", action="store_true", help="print planner rows and selected schedule without NPU submit")
    parser.add_argument("--list", action="store_true", help="list canonical shapes and selected schedules without NPU submit")
    parser.add_argument("--all", action="store_true", help="run canonical shapes in-process on the NPU")
    parser.add_argument("--start", type=int, default=1, help="1-based canonical shape index for --all")
    parser.add_argument("--limit", type=int, default=0, help="maximum shapes to run for --all")
    parser.add_argument("--filter", type=str, default="", help="regex filter for --all shape names")
    parser.add_argument("--stop-on-error", action="store_true", help="stop --all after first fail/error")
    args = parser.parse_args()

    if args.list:
        list_shapes()
        return
    if args.all:
        sweep_shapes(args.limit, args.start, args.filter, args.stop_on_error)
        return
    if not args.shape:
        parser.error("shape is required unless --list is used")
    run_shape(_shape_from_name(args.shape), args.dry_run)


if __name__ == "__main__":
    main()
