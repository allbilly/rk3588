#!/usr/bin/env python3
"""Proof experiment for pointwise IC-split plus NPU-side FP16 sum."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONV_PATH = ROOT / "examples" / "kernel_6_18" / "conv_new_clean.py"
NPU_LOCK = Path("/tmp/rk3588_npu_submit.lock")

EW_CFG_ADD = 0x108002c0 | (2 << 16)

SHAPES = [
    dict(name="b1_c512_h14_w14_oc24_wic512_k1x1_g1",
         batch=1, in_c=512, in_h=14, in_w=14, out_c=24,
         weight_in_c=512, kh=1, kw=1, groups=1),
    dict(name="b1_c1280_h10_w10_oc24_wic1280_k1x1_g1",
         batch=1, in_c=1280, in_h=10, in_w=10, out_c=24,
         weight_in_c=1280, kh=1, kw=1, groups=1),
    dict(name="b1_c1280_h10_w10_oc546_wic1280_k1x1_g1",
         batch=1, in_c=1280, in_h=10, in_w=10, out_c=546,
         weight_in_c=1280, kh=1, kw=1, groups=1),
]


@contextlib.contextmanager
def npu_lock():
    NPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with NPU_LOCK.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def load_conv_module():
    spec = importlib.util.spec_from_file_location("conv_new_clean_ic_split_exp", CONV_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CONV_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def allocate_output_bo(mod, size):
    out_map, out_bo = mod.mem_allocate(mod.fd, size=size, flags=mod.RKNPU_MEM_NON_CACHEABLE)
    return out_map, out_bo


def copy_raw_to_bo(raw, out_map, out_bo):
    flat = np.ascontiguousarray(raw.view(np.uint16))
    clear_map(out_map)
    assert flat.nbytes <= out_bo.size, "clone BO too small"
    ctypes.memmove(ctypes.addressof(ctypes.c_char.from_buffer(out_map)), flat.ctypes.data, flat.nbytes)


def clear_map(buf):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(buf)), 0, buf.size())


def submit_conv_to_bo(mod, task_regs, out_bo):
    mod.write_regs_to_npu_task(task_regs)
    for bo in (mod.regcmd_mem_create, mod.input_mem_create, mod.weight_mem_create, out_bo):
        mod.rt.fini_bo(mod.fd, bo)
    ret = mod.rt.submit(
        mod.fd,
        mod.npu_tasks,
        len(task_regs),
        in_bos=[mod.regcmd_mem_create, mod.input_mem_create, mod.weight_mem_create],
        out_bos=[out_bo],
    )
    mod.rt.prep_bo(mod.fd, out_bo)
    if ret != 0:
        raise RuntimeError(f"conv submit failed: {ret}")


def run_partial_conv(mod, input_tile, weight_tile, out_map, out_bo):
    out_c, in_c, kh, kw = weight_tile.shape
    _, in_h, in_w = input_tile.shape
    p = mod._conv_params(1, in_c, in_h, in_w, out_c, kh, kw, 1)
    wt_flat = mod.pack_weights(weight_tile, out_c, in_c, kh, kw, p["align_c"], 1)
    mod.copy_input_tile(input_tile, p)
    mod.copy_weight_flat(wt_flat)
    clear_map(out_map)
    regs = [mod.make_conv2d_regs_from_params(
        p,
        mod.input_mem_create.dma_addr,
        mod.weight_mem_create.dma_addr,
        out_bo.dma_addr,
        full_data_bank=True,
    )]
    submit_conv_to_bo(mod, regs, out_bo)
    return p


def run_partial_conv_oc_tiled(mod, input_tile, weight_tile, out_map, out_bo, oc_tile):
    out_c, in_c, kh, kw = weight_tile.shape
    _, in_h, in_w = input_tile.shape
    full_p = mod._conv_params(1, in_c, in_h, in_w, out_c, kh, kw, 1)
    clear_map(out_map)

    for oc_start in range(0, out_c, oc_tile):
        oc_end = min(oc_start + oc_tile, out_c)
        tile_out_c = oc_end - oc_start
        hw_out_c = oc_tile if tile_out_c < oc_tile else tile_out_c
        tile_p = mod._conv_params(1, in_c, in_h, in_w, hw_out_c, kh, kw, 1)
        tile_weight = np.zeros((hw_out_c, in_c, kh, kw), dtype=np.float16)
        tile_weight[:tile_out_c] = weight_tile[oc_start:oc_end]
        wt_flat = mod.pack_weights(tile_weight, hw_out_c, in_c, kh, kw, tile_p["align_c"], 1)
        mod.copy_input_tile(input_tile, tile_p)
        mod.copy_weight_flat(wt_flat)
        output_offset = (oc_start // mod.UNPACK_C2) * full_p["out_width_stride"] * mod.UNPACK_C2 * mod.FP16_BYTES
        regs = [mod.make_conv2d_regs_from_params(
            tile_p,
            mod.input_mem_create.dma_addr,
            mod.weight_mem_create.dma_addr,
            out_bo.dma_addr + output_offset,
            out_width_stride_override=full_p["out_width_stride"],
            full_data_bank=True,
        )]
        submit_conv_to_bo(mod, regs, out_bo)

    return full_p


def write_dpu_add_task(mod, a_bo, b_bo, out_bo, count, variant="current"):
    if variant == "simple":
        width = (count + 7) // 8 - 1
        regs = [
            mod.E(mod.reg.DPU,  mod.reg.S_POINTER, 0x0000000E),
            mod.E(mod.reg.DPU,  mod.reg.FEATURE_MODE_CFG, (15 << 5) | (2 << 1) | 1),
            mod.E(mod.reg.DPU,  mod.reg.DATA_FORMAT, (2 << 29) | (2 << 26) | 2),
            mod.E(mod.reg.DPU,  mod.reg.DATA_CUBE_WIDTH, width),
            mod.E(mod.reg.DPU,  mod.reg.DATA_CUBE_HEIGHT, 0),
            mod.E(mod.reg.DPU,  mod.reg.DATA_CUBE_NOTCH, 0),
            mod.E(mod.reg.DPU,  mod.reg.DATA_CUBE_CHANNEL, (7 << 16) | 7),
            mod.E(mod.reg.DPU,  mod.reg.EW_CFG, 0x108202c0),
            mod.E(mod.reg.DPU,  mod.reg.OUT_CVT_SCALE, (1 << 16) | 1),
            mod.E(mod.reg.RDMA, mod.reg.RDMA_S_POINTER, 0x0000000E),
            mod.E(mod.reg.RDMA, mod.reg.RDMA_DATA_CUBE_WIDTH, width),
            mod.E(mod.reg.RDMA, mod.reg.RDMA_DATA_CUBE_HEIGHT, 0),
            mod.E(mod.reg.RDMA, mod.reg.RDMA_DATA_CUBE_CHANNEL, 7),
            mod.E(mod.reg.RDMA, mod.reg.RDMA_ERDMA_CFG, (1 << 30) | (2 << 2)),
            mod.E(mod.reg.DPU,  mod.reg.DST_BASE_ADDR, out_bo.dma_addr),
            mod.E(mod.reg.RDMA, mod.reg.RDMA_SRC_BASE_ADDR, a_bo.dma_addr),
            mod.E(mod.reg.RDMA, mod.reg.RDMA_EW_BASE_ADDR, b_bo.dma_addr),
            mod.E(mod.reg.RDMA, mod.reg.RDMA_FEATURE_MODE_CFG, (2 << 15) | (15 << 11) | (2 << 5) | (1 << 3) | 1),
        ]
    elif variant == "current":
        regs = _current_dpu_add_regs(mod, a_bo, b_bo, out_bo, count)
    else:
        raise ValueError(f"unknown add variant {variant!r}")

    _write_one_dpu_add_task(mod, regs)


def _current_dpu_add_regs(mod, a_bo, b_bo, out_bo, count):
    width = (count + mod.UNPACK_C2 - 1) // mod.UNPACK_C2 - 1
    assert count % mod.UNPACK_C2 == 0, "DPU add runner expects c2-aligned count"
    channel = mod.UNPACK_C2 - 1
    return [
        mod.E(mod.reg.DPU, mod.reg.FEATURE_MODE_CFG, 0x1e5),
        mod.E(mod.reg.DPU, mod.reg.DATA_FORMAT, 0x48000002),
        mod.E(mod.reg.DPU, mod.reg.DATA_CUBE_CHANNEL, (channel << 16) | channel),
        mod.E(mod.reg.DPU, mod.reg.DATA_CUBE_HEIGHT, 0),
        mod.E(mod.reg.DPU, mod.reg.DATA_CUBE_WIDTH, width),
        mod.E(mod.reg.DPU, mod.reg.EW_CFG, EW_CFG_ADD),
        mod.E(mod.reg.RDMA, mod.reg.RDMA_DATA_CUBE_WIDTH, width),
        mod.E(mod.reg.RDMA, mod.reg.RDMA_DATA_CUBE_HEIGHT, 0),
        mod.E(mod.reg.RDMA, mod.reg.RDMA_DATA_CUBE_CHANNEL, (channel << 16) | channel),
        mod.E(mod.reg.RDMA, mod.reg.RDMA_ERDMA_CFG, 0x40000008),
        mod.E(mod.reg.DPU, mod.reg.DST_BASE_ADDR, out_bo.dma_addr),
        mod.E(mod.reg.RDMA, mod.reg.RDMA_SRC_BASE_ADDR, a_bo.dma_addr),
        mod.E(mod.reg.RDMA, mod.reg.RDMA_EW_BASE_ADDR, b_bo.dma_addr),
        mod.E(mod.reg.RDMA, mod.reg.RDMA_FEATURE_MODE_CFG, 0x17849),
    ]


def _write_one_dpu_add_task(mod, regs):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(mod.task_map)), 0, mod.task_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(mod.regcmd_map)), 0, mod.regcmd_map.size())
    for i, qword in enumerate(regs + [
        mod.E(mod.reg.PC_REG, mod.reg.PC_BASE_ADDRESS, 0),
        mod.E(mod.reg.PC_REG, mod.reg.PC_REGISTER_AMOUNTS, 0),
        mod.E(mod.reg.VERSION, 0, 0),
        mod.E(mod.reg.PC, mod.reg.OPERATION_ENABLE, 0x18),
    ]):
        mod.npu_regcmd[i] = qword
    mod.npu_tasks[0].flags = 0
    mod.npu_tasks[0].op_idx = 4
    mod.npu_tasks[0].enable_mask = 0x18
    mod.npu_tasks[0].int_mask = 0x300
    mod.npu_tasks[0].int_clear = 0x1ffff
    mod.npu_tasks[0].int_status = 0
    mod.npu_tasks[0].regcfg_amount = len(regs) + 4
    mod.npu_tasks[0].regcfg_offset = 0
    mod.npu_tasks[0].regcmd_addr = mod.regcmd_mem_create.dma_addr


def dpu_add(mod, a_bo, b_bo, out_map, out_bo, count, variant="current"):
    clear_map(out_map)
    write_dpu_add_task(mod, a_bo, b_bo, out_bo, count, variant=variant)
    for bo in (mod.regcmd_mem_create, a_bo, b_bo, out_bo):
        mod.rt.fini_bo(mod.fd, bo)
    ret = mod.rt.submit(
        mod.fd,
        mod.npu_tasks,
        1,
        in_bos=[mod.regcmd_mem_create, a_bo, b_bo],
        out_bos=[out_bo],
    )
    mod.rt.prep_bo(mod.fd, out_bo)
    if ret != 0:
        raise RuntimeError(f"dpu add submit failed: {ret}")


def run_known_good(mod, shape):
    result, inp, wt = mod.run_conv2d(
        shape["batch"], shape["in_c"], shape["out_c"], shape["kh"], shape["kw"],
        (shape["in_h"], shape["in_w"]),
        groups=shape["groups"],
        weight_in_c=shape["weight_in_c"],
        stride=shape.get("stride", 1),
    )
    expected = mod.compute_expected_nchw(
        inp, wt,
        shape["batch"], shape["in_c"], shape["in_h"], shape["in_w"],
        shape["out_c"], shape["kh"], shape["kw"],
        groups=shape["groups"],
        stride=shape.get("stride", 1),
    )
    return result, inp, wt, expected


def run_ic_split(mod, shape, ic_tile, add_variant="current", partial_oc_tile=0):
    known, inp, wt, expected = run_known_good(mod, shape)
    full_p = mod._conv_params(1, shape["in_c"], shape["in_h"], shape["in_w"],
                              shape["out_c"], shape["kh"], shape["kw"], 1)
    count = mod._output_count(full_p)

    partial_maps = []
    partial_bos = []
    partial_diffs = []
    partial_raws = []
    partial_p = None
    for start in range(0, shape["in_c"], ic_tile):
        end = min(start + ic_tile, shape["in_c"])
        in_chunk = inp[0, start:end]
        wt_chunk = wt[:, start:end]
        out_map, out_bo = allocate_output_bo(mod, mod.output_mem_create.size)
        if partial_oc_tile:
            partial_p = run_partial_conv_oc_tiled(mod, in_chunk, wt_chunk, out_map, out_bo, partial_oc_tile)
        else:
            partial_p = run_partial_conv(mod, in_chunk, wt_chunk, out_map, out_bo)
        partial_count = mod._output_count(partial_p)
        partial_raw = np.frombuffer(out_map, dtype=np.uint16, count=partial_count).copy().view(np.float16)
        partial = mod._unpack_flat_1x1_output(
            partial_raw,
            shape["out_c"],
            partial_p["out_h"],
            partial_p["out_w"],
            partial_p["out_width_stride"],
            mod.UNPACK_C2,
        )
        partial_expected = mod.compute_expected_nchw(
            in_chunk.reshape(1, end - start, shape["in_h"], shape["in_w"]),
            wt_chunk,
            1,
            end - start,
            shape["in_h"],
            shape["in_w"],
            shape["out_c"],
            shape["kh"],
            shape["kw"],
            groups=1,
            stride=shape.get("stride", 1),
        )[0]
        partial_diffs.append(float(np.max(np.abs(partial.astype(np.float64) - partial_expected))))
        partial_raws.append(partial_raw)
        partial_maps.append(out_map)
        partial_bos.append(out_bo)

    assert partial_p is not None
    cpu_raw_sum = partial_raws[0].copy()
    for raw in partial_raws[1:]:
        cpu_raw_sum = (cpu_raw_sum + raw).astype(np.float16)
    cpu_raw_summed = mod._unpack_flat_1x1_output(
        cpu_raw_sum,
        shape["out_c"],
        full_p["out_h"],
        full_p["out_w"],
        full_p["out_width_stride"],
        mod.UNPACK_C2,
    )
    cpu_raw_md = float(np.max(np.abs(cpu_raw_summed.astype(np.float64) - expected)))

    pair_map, pair_bo = allocate_output_bo(mod, mod.output_mem_create.size)
    dpu_add(mod, partial_bos[0], partial_bos[1], pair_map, pair_bo, count, variant=add_variant)
    pair_raw = np.frombuffer(pair_map, dtype=np.uint16, count=count).copy().view(np.float16)
    pair_expected = (partial_raws[0] + partial_raws[1]).astype(np.float16)
    pair_add_md = float(np.max(np.abs(pair_raw.astype(np.float64) - pair_expected.astype(np.float64))))

    clone_a_map, clone_a_bo = allocate_output_bo(mod, mod.output_mem_create.size)
    clone_b_map, clone_b_bo = allocate_output_bo(mod, mod.output_mem_create.size)
    clone_pair_map, clone_pair_bo = allocate_output_bo(mod, mod.output_mem_create.size)
    copy_raw_to_bo(partial_raws[0], clone_a_map, clone_a_bo)
    copy_raw_to_bo(partial_raws[1], clone_b_map, clone_b_bo)
    dpu_add(mod, clone_a_bo, clone_b_bo, clone_pair_map, clone_pair_bo, count, variant=add_variant)
    clone_pair_raw = np.frombuffer(clone_pair_map, dtype=np.uint16, count=count).copy().view(np.float16)
    clone_pair_add_md = float(np.max(np.abs(clone_pair_raw.astype(np.float64) - pair_expected.astype(np.float64))))

    ping_map, ping_bo = allocate_output_bo(mod, mod.output_mem_create.size)
    pong_map, pong_bo = allocate_output_bo(mod, mod.output_mem_create.size)
    acc_bo = partial_bos[0]
    acc_map = partial_maps[0]
    use_ping = True
    add_tasks = 0
    for next_bo in partial_bos[1:]:
        dst_map, dst_bo = (ping_map, ping_bo) if use_ping else (pong_map, pong_bo)
        dpu_add(mod, acc_bo, next_bo, dst_map, dst_bo, count, variant=add_variant)
        acc_map, acc_bo = dst_map, dst_bo
        use_ping = not use_ping
        add_tasks += 1

    summed_raw = np.frombuffer(acc_map, dtype=np.uint16, count=count).copy().view(np.float16)
    summed = mod._unpack_flat_1x1_output(
        summed_raw,
        shape["out_c"],
        full_p["out_h"],
        full_p["out_w"],
        full_p["out_width_stride"],
        mod.UNPACK_C2,
    )

    known_md = float(np.max(np.abs(known.astype(np.float64) - expected)))
    split_md = float(np.max(np.abs(summed.astype(np.float64) - expected)))
    cross_md = float(np.max(np.abs(summed.astype(np.float64) - known.astype(np.float64))))
    known_ok = bool(np.allclose(known, expected, atol=0.2) and not np.any(np.isinf(known)))
    split_ok = bool(np.allclose(summed, expected, atol=0.2) and not np.any(np.isinf(summed)))
    cross_ok = bool(np.allclose(summed, known, atol=0.2) and not np.any(np.isinf(summed)))
    return {
        "chunks": len(partial_bos),
        "add_tasks": add_tasks,
        "known_md": known_md,
        "split_md": split_md,
        "cross_md": cross_md,
        "partial_max_md": max(partial_diffs),
        "partial_diffs": partial_diffs,
        "cpu_raw_sum_md": cpu_raw_md,
        "pair_add_md": pair_add_md,
        "clone_pair_add_md": clone_pair_add_md,
        "known_ok": known_ok,
        "split_ok": split_ok,
        "cross_ok": cross_ok,
    }


def choose_ic_split_schedule(shape, args):
    if args.planner == "manual":
        return args.ic_tile, args.partial_oc_tile
    if shape["out_c"] > 128:
        return 256, 16
    return 128, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--ic-tile", type=int, default=128)
    parser.add_argument("--add-variant", choices=("current", "simple"), default="current")
    parser.add_argument("--partial-oc-tile", type=int, default=0)
    parser.add_argument("--planner", choices=("manual", "auto"), default="manual")
    parser.add_argument("--shape", default="")
    args = parser.parse_args()

    with npu_lock():
        mod = load_conv_module()
        try:
            failures = []
            shapes = SHAPES if args.limit == 0 else SHAPES[:args.limit]
            if args.shape:
                shapes = [shape for shape in shapes if args.shape in shape["name"]]
                if not shapes:
                    print(f"no shapes matched --shape {args.shape!r}")
                    return 2
            for shape in shapes:
                print(f"\n{shape['name']}")
                ic_tile, partial_oc_tile = choose_ic_split_schedule(shape, args)
                print(f"  schedule ic_tile={ic_tile} partial_oc_tile={partial_oc_tile} add_variant={args.add_variant}")
                stats = run_ic_split(
                    mod, shape, ic_tile, add_variant=args.add_variant,
                    partial_oc_tile=partial_oc_tile)
                print(f"  chunks={stats['chunks']} npu_sum_tasks={stats['add_tasks']}")
                print(f"  partial_vs_numpy max_diff={stats['partial_max_md']:.6f}")
                print(f"  cpu_raw_sum_vs_numpy max_diff={stats['cpu_raw_sum_md']:.6f}")
                print(f"  first_pair_dpu_add_vs_cpu_raw max_diff={stats['pair_add_md']:.6f}")
                print(f"  first_pair_clone_dpu_add_vs_cpu_raw max_diff={stats['clone_pair_add_md']:.6f}")
                print(f"  known_vs_numpy max_diff={stats['known_md']:.6f} {'PASS' if stats['known_ok'] else 'FAIL'}")
                print(f"  split_vs_numpy max_diff={stats['split_md']:.6f} {'PASS' if stats['split_ok'] else 'FAIL'}")
                print(f"  split_vs_known max_diff={stats['cross_md']:.6f} {'PASS' if stats['cross_ok'] else 'FAIL'}")
                if not (stats["known_ok"] and stats["split_ok"] and stats["cross_ok"]):
                    failures.append(shape["name"])
            if failures:
                print("\nIC_SPLIT_NPU_SUM FAIL")
                print("failed_shapes=" + ",".join(failures))
                return 1
            print("\nIC_SPLIT_NPU_SUM PASS")
            return 0
        finally:
            os.close(mod.fd)


if __name__ == "__main__":
    raise SystemExit(main())
