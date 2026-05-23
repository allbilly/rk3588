#!/usr/bin/env python3
"""Proof experiment for direct spatial conv with limited CBUF weight banks."""

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

SHAPES = [
    dict(name="bank_sweep_c16_h16_w16_oc8_wic16_k12x12_g1",
         batch=1, in_c=16, in_h=16, in_w=16, out_c=8,
         weight_in_c=16, kh=12, kw=12, groups=1),
    dict(name="b1_c160_h14_w14_oc320_wic160_k3x3_g1",
         batch=1, in_c=160, in_h=14, in_w=14, out_c=320,
         weight_in_c=160, kh=3, kw=3, groups=1),
    dict(name="b1_c32_h14_w14_oc128_wic32_k3x3_g1",
         batch=1, in_c=32, in_h=14, in_w=14, out_c=128,
         weight_in_c=32, kh=3, kw=3, groups=1),
]


@contextlib.contextmanager
def npu_lock():
    NPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with NPU_LOCK.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def load_conv_module():
    spec = importlib.util.spec_from_file_location("conv_new_clean_hw_weight_exp", CONV_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CONV_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generate_tensors(shape):
    np.random.seed(42)
    inp = np.random.uniform(
        -2, 2,
        (shape["batch"], shape["in_c"], shape["in_h"], shape["in_w"]),
    ).astype(np.float16)
    wt = np.random.uniform(
        -2, 2,
        (shape["out_c"], shape["weight_in_c"], shape["kh"], shape["kw"]),
    ).astype(np.float16)
    return inp, wt


def run_current(mod, shape):
    return mod.run_conv2d(
        shape["batch"], shape["in_c"], shape["out_c"], shape["kh"], shape["kw"],
        (shape["in_h"], shape["in_w"]),
        groups=shape["groups"],
        weight_in_c=shape["weight_in_c"],
        stride=shape.get("stride", 1),
    )[0]


def run_direct_spatial(mod, shape, inp, wt):
    assert shape["batch"] == 1
    assert shape["groups"] == 1
    p = mod._conv_params(
        1,
        shape["in_c"], shape["in_h"], shape["in_w"],
        shape["out_c"], shape["kh"], shape["kw"],
        1,
        stride=shape.get("stride", 1),
    )
    wt_flat = mod.pack_weights(
        wt,
        shape["out_c"], shape["in_c"], shape["kh"], shape["kw"],
        p["align_c"],
        1,
    )
    regs = [mod.make_conv2d_regs_from_params(
        p,
        mod.input_mem_create.dma_addr,
        mod.weight_mem_create.dma_addr,
        mod.output_mem_create.dma_addr,
    )]
    result = mod.submit_and_unpack_nc1(
        inp[0],
        p,
        wt_flat,
        regs,
        shape["out_c"],
        p["out_h"],
        p["out_w"],
    )
    return result.reshape(1, shape["out_c"], p["out_h"], p["out_w"])


def patch_cbuf_con0(mod, regs, data_banks, weight_banks):
    patched = list(regs)
    for idx, qword in enumerate(patched):
        if (qword & 0xffff) == mod.reg.CNA_CBUF_CON0:
            patched[idx] = mod.E(mod.reg.CNA, mod.reg.CNA_CBUF_CON0,
                                 ((weight_banks << 4) | data_banks))
            return patched
    raise RuntimeError("CNA_CBUF_CON0 not found")


def run_direct_spatial_with_banks(mod, shape, inp, wt, weight_banks):
    assert shape["batch"] == 1
    assert shape["groups"] == 1
    p = mod._conv_params(
        1,
        shape["in_c"], shape["in_h"], shape["in_w"],
        shape["out_c"], shape["kh"], shape["kw"],
        1,
        stride=shape.get("stride", 1),
    )
    wt_flat = mod.pack_weights(
        wt,
        shape["out_c"], shape["in_c"], shape["kh"], shape["kw"],
        p["align_c"],
        1,
    )
    regs = [mod.make_conv2d_regs_from_params(
        p,
        mod.input_mem_create.dma_addr,
        mod.weight_mem_create.dma_addr,
        mod.output_mem_create.dma_addr,
    )]
    data_banks = mod.RK_CBUF_BANKS - weight_banks
    regs[0] = patch_cbuf_con0(mod, regs[0], data_banks, weight_banks)
    result = mod.submit_and_unpack_nc1(
        inp[0],
        p,
        wt_flat,
        regs,
        shape["out_c"],
        p["out_h"],
        p["out_w"],
    )
    return result.reshape(1, shape["out_c"], p["out_h"], p["out_w"])


def _copy_weight_flat_at(mod, wt_flat, offset):
    flat = np.ascontiguousarray(wt_flat)
    assert offset + flat.nbytes <= mod.weight_mem_create.size, "weight buffer too small"
    ctypes.memmove(mod._weight_ptr() + offset, flat.ctypes.data, flat.nbytes)
    return flat.nbytes


def _output_dma_offset_nc1(mod, full_p, oc_start):
    assert oc_start % mod.UNPACK_C2 == 0
    return (oc_start // mod.UNPACK_C2) * full_p["out_width_stride"] * mod.UNPACK_C2 * mod.FP16_BYTES


def _raw_output_offset_elems_nc1(mod, full_p, oc_start):
    assert oc_start % mod.UNPACK_C2 == 0
    return (oc_start // mod.UNPACK_C2) * full_p["out_width_stride"] * mod.UNPACK_C2


def _align_bytes(value, alignment):
    return (value + alignment - 1) & ~(alignment - 1)


def submit_task_regs_mesa_pc_patch(mod, task_regs):
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(mod.task_map)), 0, mod.task_map.size())
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(mod.regcmd_map)), 0, mod.regcmd_map.size())

    sizes = [len(regs) + mod.PC_CHAIN_TAIL_QWORDS for regs in task_regs]
    offsets = []
    byte_offset = 0
    for size_qwords in sizes:
        offsets.append(byte_offset // ctypes.sizeof(ctypes.c_uint64))
        byte_offset += _align_bytes(size_qwords * ctypes.sizeof(ctypes.c_uint64), 64)
    assert byte_offset <= mod.regcmd_mem_create.size, "regcmd buffer too small"

    enable = mod.E(mod.reg.PC, mod.reg.OPERATION_ENABLE, (6 << 1) | 1)
    for idx, regs in enumerate(task_regs):
        base = offsets[idx]
        for i, qword in enumerate(regs):
            mod.npu_regcmd[base + i] = qword

        if idx + 1 < len(task_regs):
            next_addr = mod.regcmd_mem_create.dma_addr + offsets[idx + 1] * ctypes.sizeof(ctypes.c_uint64)
            next_size = sizes[idx + 1]
            regs_to_fetch = _align_bytes(((next_size - mod.PC_CHAIN_TAIL_QWORDS) // 2), 2)
            tails = [
                mod.E(mod.reg.PC_REG, mod.reg.PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0),
                mod.E(mod.reg.PC_REG, mod.reg.PC_REGISTER_AMOUNTS, regs_to_fetch),
                mod.E(mod.reg.VERSION, 0, 0),
                enable,
            ]
        else:
            tails = [
                mod.E(mod.reg.PC_REG, mod.reg.PC_BASE_ADDRESS, 0),
                mod.E(mod.reg.PC_REG, mod.reg.PC_REGISTER_AMOUNTS, 0),
                mod.E(mod.reg.VERSION, 0, 0),
                enable,
            ]

        for i, qword in enumerate(tails):
            mod.npu_regcmd[base + len(regs) + i] = qword

        mod.npu_tasks[idx].regcmd_addr = mod.regcmd_mem_create.dma_addr + base * ctypes.sizeof(ctypes.c_uint64)
        mod.npu_tasks[idx].regcfg_amount = sizes[idx]
        mod.npu_tasks[idx].op_idx = 1
        mod.npu_tasks[idx].enable_mask = 0xd
        mod.npu_tasks[idx].int_mask = (1 << 8) | (1 << 9)
        mod.npu_tasks[idx].int_clear = 0x1ffff

    mod.npu_submit(task_count=len(task_regs))


def run_direct_channel_tile_stitch(mod, shape, inp, wt, oc_tile, full_data_bank=False):
    assert shape["batch"] == 1
    assert shape["groups"] == 1
    full_p = mod._conv_params(
        1,
        shape["in_c"], shape["in_h"], shape["in_w"],
        shape["out_c"], shape["kh"], shape["kw"],
        1,
        stride=shape.get("stride", 1),
    )
    result = np.zeros((1, shape["out_c"], full_p["out_h"], full_p["out_w"]), dtype=np.float16)
    for oc_start in range(0, shape["out_c"], oc_tile):
        oc_end = min(oc_start + oc_tile, shape["out_c"])
        tile_out_c = oc_end - oc_start
        tile_p = mod._conv_params(
            1,
            shape["in_c"], shape["in_h"], shape["in_w"],
            tile_out_c, shape["kh"], shape["kw"],
            1,
            stride=shape.get("stride", 1),
        )
        tile_weight = wt[oc_start:oc_end].reshape(tile_out_c, shape["in_c"], shape["kh"], shape["kw"])
        wt_flat = mod.pack_weights(
            tile_weight,
            tile_out_c, shape["in_c"], shape["kh"], shape["kw"],
            tile_p["align_c"],
            1,
        )
        task_regs = [mod.make_conv2d_regs_from_params(
            tile_p,
            mod.input_mem_create.dma_addr,
            mod.weight_mem_create.dma_addr,
            mod.output_mem_create.dma_addr,
            full_data_bank=full_data_bank,
        )]
        result[0, oc_start:oc_end] = mod.submit_and_unpack_nc1(
            inp[0],
            tile_p,
            wt_flat,
            task_regs,
            tile_out_c,
            full_p["out_h"],
            full_p["out_w"],
        )
    return result


def _weight_reuse_enabled(mode, task_index):
    if mode == "none":
        return False
    if mode == "after-first":
        return task_index > 0
    if mode == "all":
        return True
    raise ValueError(f"unknown weight_reuse mode {mode!r}")


def build_channel_tile_task_regs(mod, shape, wt, oc_tile, full_data_bank=False, weight_reuse_mode="none"):
    assert shape["batch"] == 1
    assert shape["groups"] == 1
    assert oc_tile % mod.UNPACK_C2 == 0
    full_p = mod._conv_params(
        1,
        shape["in_c"], shape["in_h"], shape["in_w"],
        shape["out_c"], shape["kh"], shape["kw"],
        1,
        stride=shape.get("stride", 1),
    )

    task_regs = []
    tile_meta = []
    weight_offset = 0
    for oc_start in range(0, shape["out_c"], oc_tile):
        oc_end = min(oc_start + oc_tile, shape["out_c"])
        tile_out_c = oc_end - oc_start
        tile_p = mod._conv_params(
            1,
            shape["in_c"], shape["in_h"], shape["in_w"],
            tile_out_c, shape["kh"], shape["kw"],
            1,
            stride=shape.get("stride", 1),
        )
        tile_weight = wt[oc_start:oc_end].reshape(tile_out_c, shape["in_c"], shape["kh"], shape["kw"])
        wt_flat = mod.pack_weights(
            tile_weight,
            tile_out_c, shape["in_c"], shape["kh"], shape["kw"],
            tile_p["align_c"],
            1,
        )
        out_offset = _output_dma_offset_nc1(mod, full_p, oc_start)
        regs = mod.make_conv2d_regs_from_params(
            tile_p,
            mod.input_mem_create.dma_addr,
            mod.weight_mem_create.dma_addr + weight_offset,
            mod.output_mem_create.dma_addr + out_offset,
            out_width_stride_override=full_p["out_width_stride"],
            full_data_bank=full_data_bank,
            weight_reuse=_weight_reuse_enabled(weight_reuse_mode, len(task_regs)),
        )
        task_regs.append(regs)
        tile_meta.append(dict(
            oc_start=oc_start,
            oc_end=oc_end,
            weight_offset=weight_offset,
            weight_bytes=wt_flat.nbytes,
            output_offset=out_offset,
        ))
        weight_offset = mod._align_up(weight_offset + wt_flat.nbytes, 16)
    return full_p, task_regs, tile_meta


def run_direct_channel_tile_final_buffer(mod, shape, inp, wt, oc_tile, full_data_bank=False, prefill=0, pc_patch="helper", weight_reuse_mode="none"):
    assert shape["batch"] == 1
    assert shape["groups"] == 1
    assert oc_tile % mod.UNPACK_C2 == 0
    full_p = mod._conv_params(
        1,
        shape["in_c"], shape["in_h"], shape["in_w"],
        shape["out_c"], shape["kh"], shape["kw"],
        1,
        stride=shape.get("stride", 1),
    )
    mod.copy_input_tile(inp[0], full_p)
    ctypes.memset(mod._output_ptr(), prefill, mod.output_mem_create.size)
    ctypes.memset(mod._weight_ptr(), 0, mod.weight_mem_create.size)

    task_regs = []
    weight_offset = 0
    for oc_start in range(0, shape["out_c"], oc_tile):
        oc_end = min(oc_start + oc_tile, shape["out_c"])
        tile_out_c = oc_end - oc_start
        tile_p = mod._conv_params(
            1,
            shape["in_c"], shape["in_h"], shape["in_w"],
            tile_out_c, shape["kh"], shape["kw"],
            1,
            stride=shape.get("stride", 1),
        )
        tile_weight = wt[oc_start:oc_end].reshape(tile_out_c, shape["in_c"], shape["kh"], shape["kw"])
        wt_flat = mod.pack_weights(
            tile_weight,
            tile_out_c, shape["in_c"], shape["kh"], shape["kw"],
            tile_p["align_c"],
            1,
        )
        copied = _copy_weight_flat_at(mod, wt_flat, weight_offset)
        task_regs.append(mod.make_conv2d_regs_from_params(
            tile_p,
            mod.input_mem_create.dma_addr,
            mod.weight_mem_create.dma_addr + weight_offset,
            mod.output_mem_create.dma_addr + _output_dma_offset_nc1(mod, full_p, oc_start),
            out_width_stride_override=full_p["out_width_stride"],
            full_data_bank=full_data_bank,
            weight_reuse=_weight_reuse_enabled(weight_reuse_mode, len(task_regs)),
        ))
        weight_offset = mod._align_up(weight_offset + copied, 16)

    if pc_patch == "helper":
        mod.submit_task_regs(task_regs)
    elif pc_patch == "mesa":
        submit_task_regs_mesa_pc_patch(mod, task_regs)
    else:
        raise ValueError(f"unknown pc_patch {pc_patch!r}")
    raw = mod.read_output_fp16(mod._output_count(full_p))
    return mod._unpack_nc1hwc2_output(
        raw,
        shape["out_c"],
        full_p["out_h"],
        full_p["out_w"],
        mod.UNPACK_C2,
        full_p["out_width_stride"],
    ).reshape(1, shape["out_c"], full_p["out_h"], full_p["out_w"])


def _target_name(mod, target):
    for name, value in vars(mod.reg).items():
        if name.isupper() and value == target:
            return name
    return f"target_{target:#x}"


def _addr_name(mod, addr):
    for name, value in vars(mod.reg).items():
        if name.isupper() and value == addr:
            return name
    return f"reg_{addr:#06x}"


def _decode_reg(mod, qword):
    target = (int(qword) >> 48) & 0xffff
    value = (int(qword) >> 16) & 0xffffffff
    addr = int(qword) & 0xffff
    return target, addr, value


def _reg_map(mod, regs):
    return {(_decode_reg(mod, qword)[0], _decode_reg(mod, qword)[1]): _decode_reg(mod, qword)[2]
            for qword in regs}


def dump_channel_tile_reg_diff(mod, shape, oc_tile, full_data_bank=False, weight_reuse_mode="none"):
    _, wt = generate_tensors(shape)
    _, task_regs, tile_meta = build_channel_tile_task_regs(
        mod, shape, wt, oc_tile,
        full_data_bank=full_data_bank,
        weight_reuse_mode=weight_reuse_mode,
    )
    print(f"  dump_oc_tile={oc_tile} full_data_bank={int(full_data_bank)} weight_reuse_mode={weight_reuse_mode}")
    for idx, meta in enumerate(tile_meta):
        print(
            "  tile[{idx}] oc={oc_start}:{oc_end} weight_offset={weight_offset} "
            "weight_bytes={weight_bytes} output_offset={output_offset}".format(idx=idx, **meta)
        )
    if len(task_regs) < 2:
        print("  only one tile; no adjacent diff")
        return True

    first = _reg_map(mod, task_regs[0])
    for idx in range(1, len(task_regs)):
        cur = _reg_map(mod, task_regs[idx])
        print(f"  diff tile[0] -> tile[{idx}]")
        for key in sorted(set(first) | set(cur)):
            a = first.get(key)
            b = cur.get(key)
            if a == b:
                continue
            target, addr = key
            print(
                f"    {_target_name(mod, target)}.{_addr_name(mod, addr)} "
                f"0x{(a if a is not None else 0):08x} -> 0x{(b if b is not None else 0):08x}"
            )
    return True


def run_direct_channel_tile_final_buffer_per_submit(mod, shape, inp, wt, oc_tile, full_data_bank=False, prefill=0):
    assert shape["batch"] == 1
    assert shape["groups"] == 1
    assert oc_tile % mod.UNPACK_C2 == 0
    full_p = mod._conv_params(
        1,
        shape["in_c"], shape["in_h"], shape["in_w"],
        shape["out_c"], shape["kh"], shape["kw"],
        1,
        stride=shape.get("stride", 1),
    )
    ctypes.memset(mod._output_ptr(), prefill, mod.output_mem_create.size)
    for oc_start in range(0, shape["out_c"], oc_tile):
        oc_end = min(oc_start + oc_tile, shape["out_c"])
        tile_out_c = oc_end - oc_start
        tile_p = mod._conv_params(
            1,
            shape["in_c"], shape["in_h"], shape["in_w"],
            tile_out_c, shape["kh"], shape["kw"],
            1,
            stride=shape.get("stride", 1),
        )
        tile_weight = wt[oc_start:oc_end].reshape(tile_out_c, shape["in_c"], shape["kh"], shape["kw"])
        wt_flat = mod.pack_weights(
            tile_weight,
            tile_out_c, shape["in_c"], shape["kh"], shape["kw"],
            tile_p["align_c"],
            1,
        )
        task_regs = [mod.make_conv2d_regs_from_params(
            tile_p,
            mod.input_mem_create.dma_addr,
            mod.weight_mem_create.dma_addr,
            mod.output_mem_create.dma_addr + _output_dma_offset_nc1(mod, full_p, oc_start),
            out_width_stride_override=full_p["out_width_stride"],
            full_data_bank=full_data_bank,
        )]
        mod.copy_input_tile(inp[0], tile_p)
        mod.copy_weight_flat(wt_flat)
        mod.submit_task_regs(task_regs)

    raw = mod.read_output_fp16(mod._output_count(full_p))
    return mod._unpack_nc1hwc2_output(
        raw,
        shape["out_c"],
        full_p["out_h"],
        full_p["out_w"],
        mod.UNPACK_C2,
        full_p["out_width_stride"],
    ).reshape(1, shape["out_c"], full_p["out_h"], full_p["out_w"])


def inspect_channel_tile_raw_layout(mod, shape, inp, wt, oc_tile, pc_patch="helper", weight_reuse_mode="none"):
    full_p = mod._conv_params(
        1,
        shape["in_c"], shape["in_h"], shape["in_w"],
        shape["out_c"], shape["kh"], shape["kw"],
        1,
        stride=shape.get("stride", 1),
    )
    probe = np.zeros_like(wt)
    for oc in range(min(oc_tile, shape["out_c"])):
        probe[oc] = wt[oc]
    only_first = run_direct_channel_tile_final_buffer(
        mod, shape, inp, probe, oc_tile, full_data_bank=False, prefill=0,
        pc_patch=pc_patch, weight_reuse_mode=weight_reuse_mode)
    live = np.where(np.max(np.abs(only_first[0].astype(np.float64)), axis=(1, 2)) > 1e-6)[0]
    expected_live = np.arange(min(oc_tile, shape["out_c"]))
    ok = np.array_equal(live, expected_live)
    print(f"  raw_layout_probe live_channels={live[:40].tolist()} count={live.size} {'PASS' if ok else 'FAIL'}")
    return ok


def cbuf_info(mod, shape):
    p = mod._conv_params(
        1, shape["in_c"], shape["in_h"], shape["in_w"],
        shape["out_c"], shape["kh"], shape["kw"], 1,
        stride=shape.get("stride", 1),
    )
    full_weight_banks = mod._mesa_weight_banks(
        shape["kw"], shape["kh"], shape["in_c"], shape["out_c"], False,
    )
    data_in_channel_aligned = mod._align_up(shape["in_c"], p["align_c"])
    input_row_bytes = p["width_stride"] * data_in_channel_aligned * mod.FP16_BYTES
    feature_grains = mod._feature_grains(input_row_bytes, p["in_h"] + p["kh"], p["use_nhwc"], True, False)
    data_bank = mod._data_bank(p["width_stride"], feature_grains, data_in_channel_aligned, p["use_nhwc"], True, False)
    return full_weight_banks, data_bank, mod.RK_CBUF_BANKS - data_bank


def run_shape(mod, shape):
    inp, wt = generate_tensors(shape)
    expected = mod.compute_expected_nchw(
        inp, wt,
        shape["batch"], shape["in_c"], shape["in_h"], shape["in_w"],
        shape["out_c"], shape["kh"], shape["kw"],
        groups=shape["groups"],
        stride=shape.get("stride", 1),
    )
    full_weight_banks, data_banks, allotted_weight_banks = cbuf_info(mod, shape)
    print(f"  full_weight_banks={full_weight_banks} data_banks={data_banks} allotted_weight_banks={allotted_weight_banks}")

    current = run_current(mod, shape)
    direct = run_direct_spatial(mod, shape, inp, wt)

    current_md = float(np.max(np.abs(current.astype(np.float64) - expected)))
    direct_md = float(np.max(np.abs(direct.astype(np.float64) - expected)))
    cross_md = float(np.max(np.abs(direct.astype(np.float64) - current.astype(np.float64))))
    current_ok = bool(np.allclose(current, expected, atol=0.2) and not np.any(np.isinf(current)))
    direct_ok = bool(np.allclose(direct, expected, atol=0.2) and not np.any(np.isinf(direct)))
    cross_ok = bool(np.allclose(direct, current, atol=0.2) and not np.any(np.isinf(direct)))
    print(f"  current_vs_numpy max_diff={current_md:.6f} {'PASS' if current_ok else 'FAIL'}")
    print(f"  direct_vs_numpy  max_diff={direct_md:.6f} {'PASS' if direct_ok else 'FAIL'}")
    print(f"  direct_vs_current max_diff={cross_md:.6f} {'PASS' if cross_ok else 'FAIL'}")
    return current_ok and direct_ok and cross_ok


def run_bank_sweep(mod, shape):
    inp, wt = generate_tensors(shape)
    expected = mod.compute_expected_nchw(
        inp, wt,
        shape["batch"], shape["in_c"], shape["in_h"], shape["in_w"],
        shape["out_c"], shape["kh"], shape["kw"],
        groups=shape["groups"],
        stride=shape.get("stride", 1),
    )
    full_weight_banks, data_banks, allotted_weight_banks = cbuf_info(mod, shape)
    print(f"  full_weight_banks={full_weight_banks} natural_data_banks={data_banks} natural_weight_banks={allotted_weight_banks}")
    failures = []
    for weight_banks in range(mod.RK_CBUF_BANKS - 1, 0, -1):
        result = run_direct_spatial_with_banks(mod, shape, inp, wt, weight_banks)
        md = float(np.max(np.abs(result.astype(np.float64) - expected)))
        ok = bool(np.allclose(result, expected, atol=0.2) and not np.any(np.isinf(result)))
        label = "PASS" if ok else "FAIL"
        print(f"  forced_weight_banks={weight_banks:2d} forced_data_banks={mod.RK_CBUF_BANKS - weight_banks:2d} max_diff={md:.6f} {label}")
        if weight_banks >= full_weight_banks and not ok:
            failures.append(weight_banks)
    return not failures


def run_channel_tile_shape(mod, shape, oc_tiles=(16, 32), prefill=0x00, pc_patches=("helper",), weight_reuse_modes=("none",)):
    inp, wt = generate_tensors(shape)
    expected = mod.compute_expected_nchw(
        inp, wt,
        shape["batch"], shape["in_c"], shape["in_h"], shape["in_w"],
        shape["out_c"], shape["kh"], shape["kw"],
        groups=shape["groups"],
        stride=shape.get("stride", 1),
    )
    full_weight_banks, data_banks, allotted_weight_banks = cbuf_info(mod, shape)
    print(f"  full_weight_banks={full_weight_banks} data_banks={data_banks} allotted_weight_banks={allotted_weight_banks}")
    current = run_current(mod, shape)
    current_md = float(np.max(np.abs(current.astype(np.float64) - expected)))
    current_ok = bool(np.allclose(current, expected, atol=0.2) and not np.any(np.isinf(current)))
    print(f"  current_vs_numpy max_diff={current_md:.6f} {'PASS' if current_ok else 'FAIL'}")

    failures = []
    for oc_tile in oc_tiles:
        if oc_tile > shape["out_c"] or shape["out_c"] % oc_tile != 0:
            continue
        print(f"  oc_tile={oc_tile}")
        layout_ok_by_variant = {}
        for pc_patch in pc_patches:
            for weight_reuse_mode in weight_reuse_modes:
                variant = f"{pc_patch}_wr_{weight_reuse_mode}"
                layout_ok_by_variant[variant] = inspect_channel_tile_raw_layout(
                    mod, shape, inp, wt, oc_tile,
                    pc_patch=pc_patch,
                    weight_reuse_mode=weight_reuse_mode)
        for full_data_bank in (False, True):
            label = "full_data" if full_data_bank else "natural_data"
            stitched = run_direct_channel_tile_stitch(
                mod, shape, inp, wt, oc_tile, full_data_bank=full_data_bank)
            stitched_md = float(np.max(np.abs(stitched.astype(np.float64) - expected)))
            stitched_ok = bool(np.allclose(stitched, expected, atol=0.2) and not np.any(np.isinf(stitched)))
            print(f"    {label}_cpu_stitch_vs_numpy max_diff={stitched_md:.6f} {'PASS' if stitched_ok else 'FAIL'}")

            final_ok_by_variant = {}
            cross_ok_by_variant = {}
            for pc_patch in pc_patches:
                for weight_reuse_mode in weight_reuse_modes:
                    variant = f"{pc_patch}_wr_{weight_reuse_mode}"
                    final = run_direct_channel_tile_final_buffer(
                        mod, shape, inp, wt, oc_tile, full_data_bank=full_data_bank,
                        prefill=prefill, pc_patch=pc_patch,
                        weight_reuse_mode=weight_reuse_mode)
                    final_md = float(np.max(np.abs(final.astype(np.float64) - expected)))
                    final_cross = float(np.max(np.abs(final.astype(np.float64) - stitched.astype(np.float64))))
                    final_ok = bool(np.allclose(final, expected, atol=0.2) and not np.any(np.isinf(final)))
                    cross_ok = bool(np.allclose(final, stitched, atol=0.2) and not np.any(np.isinf(final)))
                    final_ok_by_variant[variant] = final_ok
                    cross_ok_by_variant[variant] = cross_ok
                    print(f"    {label}_{variant}_finalbuf_vs_numpy  max_diff={final_md:.6f} {'PASS' if final_ok else 'FAIL'}")
                    print(f"    {label}_{variant}_finalbuf_vs_stitch max_diff={final_cross:.6f} {'PASS' if cross_ok else 'FAIL'}")

            per_submit = run_direct_channel_tile_final_buffer_per_submit(
                mod, shape, inp, wt, oc_tile, full_data_bank=full_data_bank, prefill=prefill)
            per_submit_md = float(np.max(np.abs(per_submit.astype(np.float64) - expected)))
            per_submit_cross = float(np.max(np.abs(per_submit.astype(np.float64) - stitched.astype(np.float64))))
            per_submit_ok = bool(np.allclose(per_submit, expected, atol=0.2) and not np.any(np.isinf(per_submit)))
            per_submit_cross_ok = bool(np.allclose(per_submit, stitched, atol=0.2) and not np.any(np.isinf(per_submit)))
            print(f"    {label}_per_submit_final_vs_numpy  max_diff={per_submit_md:.6f} {'PASS' if per_submit_ok else 'FAIL'}")
            print(f"    {label}_per_submit_final_vs_stitch max_diff={per_submit_cross:.6f} {'PASS' if per_submit_cross_ok else 'FAIL'}")
            if not (all(layout_ok_by_variant.values()) and stitched_ok and
                    all(final_ok_by_variant.values()) and all(cross_ok_by_variant.values()) and
                    per_submit_ok and per_submit_cross_ok):
                failures.append((oc_tile, label))
    return current_ok and not failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--mode", choices=("weight-stream", "channel-tile", "dump-channel-tile-regs"), default="channel-tile")
    parser.add_argument("--oc-tiles", default="16,32")
    parser.add_argument("--pc-patches", default="helper")
    parser.add_argument("--weight-reuse-modes", default="none")
    parser.add_argument("--shape", default="")
    parser.add_argument("--prefill", type=lambda x: int(x, 0), default=0)
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
                if args.mode == "dump-channel-tile-regs" and not shape["name"].startswith("bank_sweep"):
                    oc_tiles = tuple(int(x) for x in args.oc_tiles.split(",") if x)
                    weight_reuse_modes = tuple(x for x in args.weight_reuse_modes.split(",") if x)
                    ok = True
                    for oc_tile in oc_tiles:
                        if oc_tile > shape["out_c"] or shape["out_c"] % oc_tile != 0:
                            continue
                        for weight_reuse_mode in weight_reuse_modes:
                            ok = (dump_channel_tile_reg_diff(
                                mod, shape, oc_tile,
                                full_data_bank=False,
                                weight_reuse_mode=weight_reuse_mode) and
                                  dump_channel_tile_reg_diff(
                                      mod, shape, oc_tile,
                                      full_data_bank=True,
                                      weight_reuse_mode=weight_reuse_mode) and ok)
                elif args.mode == "channel-tile" and not shape["name"].startswith("bank_sweep"):
                    oc_tiles = tuple(int(x) for x in args.oc_tiles.split(",") if x)
                    pc_patches = tuple(x for x in args.pc_patches.split(",") if x)
                    weight_reuse_modes = tuple(x for x in args.weight_reuse_modes.split(",") if x)
                    ok = run_channel_tile_shape(
                        mod, shape, oc_tiles=oc_tiles, prefill=args.prefill,
                        pc_patches=pc_patches,
                        weight_reuse_modes=weight_reuse_modes)
                elif shape["name"].startswith("bank_sweep"):
                    ok = run_bank_sweep(mod, shape)
                else:
                    ok = run_shape(mod, shape)
                if not ok:
                    failures.append(shape["name"])
            if failures:
                print(f"\n{args.mode.upper().replace('-', '_')} FAIL")
                print("failed_shapes=" + ",".join(failures))
                return 1
            print(f"\n{args.mode.upper().replace('-', '_')} PASS")
            return 0
        finally:
            os.close(mod.fd)


if __name__ == "__main__":
    raise SystemExit(main())
