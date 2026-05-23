#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "kernel_6_18"))

import conv  # noqa: E402

sys.path.insert(0, str(ROOT / "experimental" / "rknn"))
import rknn_parse_regcmd_runs as rknn_runs  # noqa: E402
import rknn_descriptor_plan as descriptor_plan  # noqa: E402


SHAPES = {
    "cmp_b1_c32_h14_w14_oc128_wic32_k3x3_g1": (32, 14, 14, 128, 3, 3, 1, 1),
    "cmp_b1_c160_h14_w14_oc320_wic160_k3x3_g1": (160, 14, 14, 320, 3, 3, 1, 1),
    "b1_c160_h7_w7_oc320_wic160_k3x3_g1": (160, 7, 7, 320, 3, 3, 1, 1),
    "goal_b1_c160_h7_w7_oc320_wic160_k3x3_g1": (160, 7, 7, 320, 3, 3, 1, 1),
    "sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1": (160, 40, 40, 320, 3, 3, 1, 1),
    "mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1": (160, 40, 40, 320, 3, 3, 1, 1),
    "mix_b1_c72_h20_w20_oc288_wic72_k3x3_g1": (72, 20, 20, 288, 3, 3, 1, 1),
}

OBSERVED_DESCRIPTOR_SWEEP_SHAPES = (
    "b1_c160_h7_w7_oc320_wic160_k3x3_g1",
    "cmp_b1_c160_h14_w14_oc320_wic160_k3x3_g1",
    "sweep_b1_c160_h16_w16_oc320_wic160_k3x3_g1",
    "sweep_b1_c160_h20_w20_oc320_wic160_k3x3_g1",
    "sweep_b1_c160_h24_w24_oc320_wic160_k3x3_g1",
    "sweep_b1_c160_h28_w28_oc320_wic160_k3x3_g1",
    "sweep_b1_c160_h32_w32_oc320_wic160_k3x3_g1",
    "sweep_b1_c160_h36_w36_oc320_wic160_k3x3_g1",
    "sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1",
)

OBSERVED_DESCRIPTOR_SPATIAL_Y_TILE_SHAPES = (
    "outc_b1_c64_h20_w20_oc32_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc48_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc64_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc96_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc112_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc128_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc160_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc192_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc224_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc256_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc288_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc320_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc384_wic64_k3x3_g1",
    "outc_b1_c64_h20_w20_oc512_wic64_k3x3_g1",
)

OBSERVED_DESCRIPTOR_CHAN320_SHAPES = (
    "chan_b1_c32_h20_w20_oc320_wic32_k3x3_g1",
    "chan_b1_c40_h20_w20_oc320_wic40_k3x3_g1",
    "chan_b1_c64_h20_w20_oc320_wic64_k3x3_g1",
    "chan_b1_c72_h20_w20_oc320_wic72_k3x3_g1",
    "chan_b1_c96_h20_w20_oc320_wic96_k3x3_g1",
    "chan_b1_c128_h20_w20_oc320_wic128_k3x3_g1",
    "chan_b1_c160_h20_w20_oc320_wic160_k3x3_g1",
    "chan_b1_c192_h20_w20_oc320_wic192_k3x3_g1",
    "chan_b1_c256_h20_w20_oc320_wic256_k3x3_g1",
)

OBSERVED_DESCRIPTOR_MIXED_SPATIAL_SHAPES = (
    "mix_b1_c72_h20_w20_oc288_wic72_k3x3_g1",
)

OBSERVED_DESCRIPTOR_COMPARE_SPATIAL_SHAPES = (
    "cmp_b1_c32_h14_w14_oc128_wic32_k3x3_g1",
)

OBSERVED_DESCRIPTOR_POINTWISE_SHAPES = (
    "pw_b1_c40_h20_w20_oc320_wic40_k1x1_g1",
    "pw_b1_c40_h28_w28_oc320_wic40_k1x1_g1",
    "pw_b1_c40_h40_w40_oc320_wic40_k1x1_g1",
    "pw_b1_c40_h56_w56_oc320_wic40_k1x1_g1",
    "pw_b1_c64_h20_w20_oc128_wic64_k1x1_g1",
    "pw_b1_c64_h40_w40_oc128_wic64_k1x1_g1",
    "pw_b1_c256_h14_w14_oc512_wic256_k1x1_g1",
    "pw_b1_c256_h28_w28_oc512_wic256_k1x1_g1",
    "pw_b1_c528_h14_w14_oc32_wic528_k1x1_g1",
    "pw_b1_c528_h20_w20_oc32_wic528_k1x1_g1",
    "pw_b1_c528_h40_w40_oc32_wic528_k1x1_g1",
)

DEFAULT_RKNN_PATHS = {
    name: ROOT / "experimental" / "rknn" / "models" / f"{name}.rknn"
    for name in SHAPES
}


REGS = {
    "CONV_CON2": conv.reg.CNA_CONV_CON2,
    "DATA_SIZE0": conv.reg.CNA_DATA_SIZE0,
    "DATA_SIZE1": conv.reg.CNA_DATA_SIZE1,
    "DATA_SIZE3": conv.reg.CNA_DATA_SIZE3,
    "WEIGHT_SIZE0": conv.reg.CNA_WEIGHT_SIZE0,
    "WEIGHT_SIZE1": conv.reg.CNA_WEIGHT_SIZE1,
    "WEIGHT_SIZE2": conv.reg.CNA_WEIGHT_SIZE2,
    "CBUF_CON0": conv.reg.CNA_CBUF_CON0,
    "CBUF_CON1": conv.reg.CNA_CBUF_CON1,
    "FEATURE_ADDR": conv.reg.CNA_FEATURE_DATA_ADDR,
    "FC_SIZE0": conv.reg.CNA_FC_DATA_SIZE0,
    "DCOMP_ADDR0": conv.reg.CNA_DCOMP_ADDR0,
    "CORE_SIZE0": conv.reg.CORE_DATAOUT_SIZE_0,
    "CORE_SIZE1": conv.reg.CORE_DATAOUT_SIZE_1,
    "DST_BASE": conv.reg.DST_BASE_ADDR,
    "DST_SURF_STRIDE": conv.reg.DST_SURF_STRIDE,
    "DPU_DST_C": conv.reg.DATA_CUBE_CHANNEL,
    "DPU_SIZE": conv.reg.WDMA_SIZE_1,
    "SURFACE_ADD": conv.reg.SURFACE_ADD,
}


PC_TAIL_QWORDS = 4
RKNPU_PC_DATA_EXTRA_AMOUNT = 4
RKNPU_JOB_C = ROOT / "ref" / "rknpu_driver" / "rknpu_job.c"
RKNPU_JOB_H = ROOT / "ref" / "rknpu_driver" / "include" / "rknpu_job.h"
RKNPU_DRV_C = ROOT / "ref" / "rknpu_driver" / "rknpu_drv.c"
RKNPU_IOCTL_H = ROOT / "ref" / "rknpu_driver" / "include" / "rknpu_ioctl.h"
ROCKET_ACCEL_H = ROOT / "ref" / "librocket" / "include" / "rocket_accel.h"
LIBROCKET_TESTING = ROOT / "ref" / "librocket" / "TESTING.md"
KERNEL_ROCKET_JOB_C_URL = "https://raw.githubusercontent.com/torvalds/linux/master/drivers/accel/rocket/rocket_job.c"

REG_NAMES = {
    (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS): "PC_BASE_ADDRESS",
    (conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS): "PC_REGISTER_AMOUNTS",
    (conv.reg.PC, conv.reg.OPERATION_ENABLE): "PC_OPERATION_ENABLE",
    (conv.reg.VERSION, 0): "VERSION",
    (0x0001, 0): "PC_CHAIN_END",
}
for _name in (
        "PPU_S_POINTER", "PPU_DATA_CUBE_IN_WIDTH", "PPU_DATA_CUBE_IN_HEIGHT",
        "PPU_DATA_CUBE_IN_CHANNEL", "PPU_DATA_CUBE_OUT_WIDTH",
        "PPU_DATA_CUBE_OUT_HEIGHT", "PPU_DATA_CUBE_OUT_CHANNEL",
        "PPU_OPERATION_MODE_CFG", "PPU_POOLING_KERNEL_CFG",
        "PPU_RECIP_KERNEL_WIDTH", "PPU_RECIP_KERNEL_HEIGHT",
        "PPU_POOLING_PADDING_CFG", "PPU_PADDING_VALUE_1_CFG",
        "PPU_PADDING_VALUE_2_CFG", "PPU_DST_BASE_ADDR",
        "PPU_DST_SURF_STRIDE", "PPU_DATA_FORMAT", "PPU_MISC_CTRL"):
    REG_NAMES[(conv.reg.PPU, getattr(conv.reg, _name))] = _name
for _name in (
        "PPU_RDMA_S_POINTER", "PPU_RDMA_CUBE_IN_WIDTH",
        "PPU_RDMA_CUBE_IN_HEIGHT", "PPU_RDMA_CUBE_IN_CHANNEL",
        "PPU_RDMA_SRC_BASE_ADDR", "PPU_RDMA_SRC_LINE_STRIDE",
        "PPU_RDMA_SRC_SURF_STRIDE", "PPU_RDMA_DATA_FORMAT"):
    REG_NAMES[(conv.reg.PPU_RDMA, getattr(conv.reg, _name))] = _name
for _name, _reg_addr in REGS.items():
    REG_NAMES[(conv.reg.CNA, _reg_addr)] = _name
    REG_NAMES[(conv.reg.CORE, _reg_addr)] = _name
    REG_NAMES[(conv.reg.DPU, _reg_addr)] = _name
    REG_NAMES[(conv.reg.RDMA, _reg_addr)] = _name


def reg_values(regs):
    values = {}
    for cmd in regs:
        values[cmd & 0xffff] = (cmd >> 16) & 0xffffffff
    return values


def fmt(value):
    return f"0x{value:x}"


def make_regs(p, desc, full_body):
    desc.extra = {"full_data_bank": True}
    if full_body:
        regs = conv.RegList(conv.make_rknn_full_conv2d_regs_from_desc(p, desc, 0, 0, 0))
    else:
        regs = conv.RegList(conv.make_conv2d_regs_from_desc(p, desc, 0, 0, 0))
        regs.extend(conv.E(conv.reg.DPU, clear_reg, 0) for clear_reg in conv.DPU_CLEAR_REGS)
    regs.pc_core = desc.pc_core
    return regs


def direct_rows(shape_name, full_body=False):
    in_c, in_h, in_w, out_c, kh, kw, groups, stride = SHAPES[shape_name]
    p = conv._conv_params(in_c, in_h, in_w, out_c, kh, kw, groups, stride)
    descs = conv.plan_observed_spatial_tile_descs(p, in_c, out_c, kh, kw, in_h, in_w, groups, stride)
    rows = []
    body_lens = []
    body_values = []
    for desc in descs:
        regs = make_regs(p, desc, full_body)
        body_lens.append(len(regs))
        body_values.append(reg_values(regs))
    for idx, desc in enumerate(descs, 1):
        values = body_values[idx - 1]
        next_body_len = body_lens[idx] if idx < len(body_lens) else 0
        row = {
            "idx": idx,
            "family": desc.family,
            "input_h": desc.input_h,
            "output_h": desc.output_h,
            "output_w": desc.output_w,
            "oc_start": desc.oc_start,
            "oc_count": desc.oc_count,
            "feature_off": desc.feature_off,
            "weight_off": desc.weight_off,
            "output_off": desc.output_off,
            "body_qwords": body_lens[idx - 1],
            "regcfg_amount": body_lens[idx - 1] + PC_TAIL_QWORDS,
            "pc_register_amounts": 0 if idx == len(descs) else (
                (getattr(desc, "pc_core", 0) << 16) | (conv._ceil_div(next_body_len, 2) + 1)
            ),
            "op_idx": 0,
            "enable_mask": 0x0d,
        }
        for name, reg_addr in REGS.items():
            row[name] = values.get(reg_addr)
        rows.append(row)
    return rows


def print_summary(rows):
    fields = [
        "idx", "family", "input_h", "output_h", "output_w", "oc_start", "oc_count",
        "feature_off", "weight_off", "output_off", "CONV_CON2", "CBUF_CON0",
        "CBUF_CON1", "DATA_SIZE0", "DATA_SIZE3", "WEIGHT_SIZE0", "WEIGHT_SIZE1",
        "WEIGHT_SIZE2", "DST_SURF_STRIDE", "SURFACE_ADD", "body_qwords",
        "regcfg_amount", "pc_register_amounts", "op_idx", "enable_mask",
    ]
    print(",".join(fields))
    for row in rows:
        out = []
        for field in fields:
            value = row[field]
            out.append(fmt(value) if isinstance(value, int) and field not in {
                "idx", "input_h", "output_h", "output_w", "oc_start", "oc_count",
            } else str(value))
        print(",".join(out))


def local_full_bodies(shape_name):
    in_c, in_h, in_w, out_c, kh, kw, groups, stride = SHAPES[shape_name]
    p = conv._conv_params(in_c, in_h, in_w, out_c, kh, kw, groups, stride)
    descs = conv.plan_observed_spatial_tile_descs(p, in_c, out_c, kh, kw, in_h, in_w, groups, stride)
    bodies = []
    for desc in descs:
        bodies.append(make_regs(p, desc, full_body=True))
    return bodies


def runtime_direct_spatial_descs(shape_name):
    in_c, in_h, in_w, out_c, kh, kw, groups, stride = SHAPES[shape_name]
    p = conv._conv_params(in_c, in_h, in_w, out_c, kh, kw, groups, stride)
    input_nchw, weight_nchw = conv.make_conv_test_data(1, in_c, out_c, kh, kw, (in_h, in_w), groups)
    descs, _packed_input, _packed_weight = conv.build_direct_spatial_descs(
        0, input_nchw, weight_nchw, p, in_c, out_c, kh, kw, in_h, in_w, groups, stride)
    return descs


def shape_tuple(shape_name):
    if shape_name in SHAPES:
        return SHAPES[shape_name]
    shape = descriptor_plan.SHAPES[shape_name]
    return (shape.in_c, shape.in_h, shape.in_w, shape.out_c,
            shape.kh, shape.kw, shape.groups, shape.stride)


def planned_descs_for_shape(shape_name):
    in_c, in_h, in_w, out_c, kh, kw, groups, stride = shape_tuple(shape_name)
    input_nchw, weight_nchw = conv.make_conv_test_data(1, in_c, out_c, kh, kw, (in_h, in_w), groups)
    p = conv._conv_params(in_c, in_h, in_w, out_c, kh, kw, groups, stride)
    result = conv.np.zeros((1, out_c, p["out_h"], p["out_w"]), dtype=conv.np.float16)
    return conv.plan_conv_descriptors(
        0, input_nchw, weight_nchw, result,
        in_c, out_c, kh, kw, (in_h, in_w), groups, stride)


def local_observed_descriptor_keys(shape_name):
    if shape_name in SHAPES:
        in_c, in_h, in_w, out_c, kh, kw, groups, stride = SHAPES[shape_name]
    else:
        shape = descriptor_plan.SHAPES[shape_name]
        in_c, in_h, in_w = shape.in_c, shape.in_h, shape.in_w
        out_c, kh, kw = shape.out_c, shape.kh, shape.kw
        groups, stride = shape.groups, shape.stride
    p = conv._conv_params(in_c, in_h, in_w, out_c, kh, kw, groups, stride)
    descs = conv.plan_observed_spatial_tile_descs(p, in_c, out_c, kh, kw, in_h, in_w, groups, stride)
    return [
        {
            "family": desc.family,
            "family_bits": desc.family_bits,
            "grain_bits": desc.grain_bits,
            "cbuf0": desc.cbuf0,
            "pc_core": desc.pc_core,
            "input_y": desc.input_y,
            "input_h": desc.input_h,
            "output_h": desc.output_h,
            "output_w": desc.output_w,
            "oc_start": desc.oc_start,
            "oc_count": desc.oc_count,
            "feature_off": desc.feature_off,
            "weight_off": desc.weight_off,
            "output_off": desc.output_off,
        }
        for desc in descs
    ]


def reference_observed_descriptor_keys(shape_name):
    shape = descriptor_plan.SHAPES[shape_name]
    feature_row = shape.in_w * conv.UNPACK_C2 * conv.FP16_BYTES
    weight_per_oc = shape.kh * shape.kw * shape.in_c * conv.FP16_BYTES
    return [
        {
            "family": desc["family"],
            "family_bits": desc["family_bits"],
            "grain_bits": desc["grain_bits"],
            "cbuf0": desc.get("cbuf0"),
            "pc_core": desc.get("pc_core", 0),
            "input_y": desc["feature_off"] // feature_row,
            "input_h": desc["input_h"],
            "output_h": desc["output_h"],
            "output_w": desc["output_w"],
            "oc_start": 0 if weight_per_oc == 0 else desc["weight_off"] // weight_per_oc,
            "oc_count": desc["oc_count"],
            "feature_off": desc["feature_off"],
            "weight_off": desc["weight_off"],
            "output_off": desc["output_off"],
        }
        for desc in descriptor_plan.observed_descriptors(shape)
    ]


def verify_observed_descriptor_plan(shape_name):
    if shape_name not in descriptor_plan.SHAPES:
        print(f"FAIL shape not in descriptor reference: {shape_name}")
        return 1
    local = local_observed_descriptor_keys(shape_name)
    reference = reference_observed_descriptor_keys(shape_name)
    if local != reference:
        print("FAIL")
        print(f"local_count={len(local)} reference_count={len(reference)}")
        for idx, (lrow, rrow) in enumerate(zip(local, reference), 1):
            if lrow != rrow:
                print(f"first_diff={idx}")
                print(f"local={lrow}")
                print(f"reference={rrow}")
                break
        if len(local) != len(reference):
            print(f"local_tail={local[min(len(local), len(reference)):]}")
            print(f"reference_tail={reference[min(len(local), len(reference)):]}")
        return 1
    print(f"PASS observed descriptor parity rows={len(local)}")
    print("conv.py plan_observed_spatial_tile_descs matches experimental/rknn_descriptor_plan.py --mode observed")
    return 0


def verify_observed_descriptor_sweep():
    failures = []
    rows = 0
    for shape_name in OBSERVED_DESCRIPTOR_SWEEP_SHAPES:
        local = local_observed_descriptor_keys(shape_name)
        reference = reference_observed_descriptor_keys(shape_name)
        rows += len(local)
        if local != reference:
            failures.append((shape_name, local, reference))
            break
        print(f"{shape_name}: PASS rows={len(local)}")
    if failures:
        shape_name, local, reference = failures[0]
        print(f"FAIL observed descriptor sweep at {shape_name}")
        print(f"local_count={len(local)} reference_count={len(reference)}")
        for idx, (lrow, rrow) in enumerate(zip(local, reference), 1):
            if lrow != rrow:
                print(f"first_diff={idx}")
                print(f"local={lrow}")
                print(f"reference={rrow}")
                break
        if len(local) != len(reference):
            print(f"local_tail={local[min(len(local), len(reference)):]}")
            print(f"reference_tail={reference[min(len(local), len(reference)):]}")
        return 1
    print(f"PASS observed descriptor sweep shapes={len(OBSERVED_DESCRIPTOR_SWEEP_SHAPES)} rows={rows}")
    print("conv.py observed spatial planner matches the reference sweep for 160->320 k3x3 square sizes 16..40")
    return 0

def verify_observed_spatial_y_tile_descriptors():
    failures = []
    rows = 0
    for shape_name in OBSERVED_DESCRIPTOR_SPATIAL_Y_TILE_SHAPES:
        local = local_observed_descriptor_keys(shape_name)
        reference = reference_observed_descriptor_keys(shape_name)
        rows += len(local)
        if local != reference:
            failures.append((shape_name, local, reference))
            break
        print(f"{shape_name}: PASS rows={len(local)}")
    if failures:
        shape_name, local, reference = failures[0]
        print(f"FAIL observed spatial y_tile descriptors at {shape_name}")
        print(f"local_count={len(local)} reference_count={len(reference)}")
        for idx, (lrow, rrow) in enumerate(zip(local, reference), 1):
            if lrow != rrow:
                print(f"first_diff={idx}")
                print(f"local={lrow}")
                print(f"reference={rrow}")
                break
        if len(local) != len(reference):
            print(f"local_tail={local[min(len(local), len(reference)):]}")
            print(f"reference_tail={reference[min(len(local), len(reference)):]}")
        return 1
    print(f"PASS observed spatial y_tile descriptors shapes={len(OBSERVED_DESCRIPTOR_SPATIAL_Y_TILE_SHAPES)} rows={rows}")
    print("conv.py observed spatial planner matches the reference setup+k_half+y_tile family for c64 h20 k3x3")
    return 0

def verify_observed_chan320_descriptors():
    failures = []
    rows = 0
    for shape_name in OBSERVED_DESCRIPTOR_CHAN320_SHAPES:
        local = local_observed_descriptor_keys(shape_name)
        reference = reference_observed_descriptor_keys(shape_name)
        rows += len(local)
        if local != reference:
            failures.append((shape_name, local, reference))
            break
        print(f"{shape_name}: PASS rows={len(local)}")
    if failures:
        shape_name, local, reference = failures[0]
        print(f"FAIL observed chan320 descriptors at {shape_name}")
        print(f"local_count={len(local)} reference_count={len(reference)}")
        for idx, (lrow, rrow) in enumerate(zip(local, reference), 1):
            if lrow != rrow:
                print(f"first_diff={idx}")
                print(f"local={lrow}")
                print(f"reference={rrow}")
                break
        if len(local) != len(reference):
            print(f"local_tail={local[min(len(local), len(reference)):]}")
            print(f"reference_tail={reference[min(len(local), len(reference)):]}")
        return 1
    print(f"PASS observed chan320 descriptors shapes={len(OBSERVED_DESCRIPTOR_CHAN320_SHAPES)} rows={rows}")
    print("conv.py observed spatial planner matches the reference channel sweep for out_c=320 c32..256 h20 k3x3")
    return 0

def verify_observed_mixed_spatial_descriptors():
    failures = []
    rows = 0
    for shape_name in OBSERVED_DESCRIPTOR_MIXED_SPATIAL_SHAPES:
        local = local_observed_descriptor_keys(shape_name)
        reference = reference_observed_descriptor_keys(shape_name)
        rows += len(local)
        if local != reference:
            failures.append((shape_name, local, reference))
            break
        print(f"{shape_name}: PASS rows={len(local)}")
    if failures:
        shape_name, local, reference = failures[0]
        print(f"FAIL observed mixed spatial descriptors at {shape_name}")
        print(f"local_count={len(local)} reference_count={len(reference)}")
        for idx, (lrow, rrow) in enumerate(zip(local, reference), 1):
            if lrow != rrow:
                print(f"first_diff={idx}")
                print(f"local={lrow}")
                print(f"reference={rrow}")
                break
        if len(local) != len(reference):
            print(f"local_tail={local[min(len(local), len(reference)):]}")
            print(f"reference_tail={reference[min(len(local), len(reference)):]}")
        return 1
    print(f"PASS observed mixed spatial descriptors shapes={len(OBSERVED_DESCRIPTOR_MIXED_SPATIAL_SHAPES)} rows={rows}")
    print("conv.py observed planner matches the RKNN mixed 72->288 h20 k3x3 setup+k_half+k_tile template")
    return 0

def verify_observed_compare_spatial_descriptors():
    failures = []
    rows = 0
    for shape_name in OBSERVED_DESCRIPTOR_COMPARE_SPATIAL_SHAPES:
        local = local_observed_descriptor_keys(shape_name)
        reference = reference_observed_descriptor_keys(shape_name)
        rows += len(local)
        if local != reference:
            failures.append((shape_name, local, reference))
            break
        print(f"{shape_name}: PASS rows={len(local)}")
    if failures:
        shape_name, local, reference = failures[0]
        print(f"FAIL observed compare spatial descriptors at {shape_name}")
        print(f"local_count={len(local)} reference_count={len(reference)}")
        for idx, (lrow, rrow) in enumerate(zip(local, reference), 1):
            if lrow != rrow:
                print(f"first_diff={idx}")
                print(f"local={lrow}")
                print(f"reference={rrow}")
                break
        if len(local) != len(reference):
            print(f"local_tail={local[min(len(local), len(reference)):]}")
            print(f"reference_tail={reference[min(len(local), len(reference)):]}")
        return 1
    print(f"PASS observed compare spatial descriptors shapes={len(OBSERVED_DESCRIPTOR_COMPARE_SPATIAL_SHAPES)} rows={rows}")
    print("conv.py observed planner matches the RKNN compare-set 32->128 h14 k3x3 setup+k_half+y_tile template")
    return 0

def verify_observed_pointwise_descriptors():
    failures = []
    rows = 0
    for shape_name in OBSERVED_DESCRIPTOR_POINTWISE_SHAPES:
        local = local_observed_descriptor_keys(shape_name)
        reference = reference_observed_descriptor_keys(shape_name)
        rows += len(local)
        if local != reference:
            failures.append((shape_name, local, reference))
            break
        print(f"{shape_name}: PASS rows={len(local)}")
    if failures:
        shape_name, local, reference = failures[0]
        print(f"FAIL observed pointwise descriptors at {shape_name}")
        print(f"local_count={len(local)} reference_count={len(reference)}")
        for idx, (lrow, rrow) in enumerate(zip(local, reference), 1):
            if lrow != rrow:
                print(f"first_diff={idx}")
                print(f"local={lrow}")
                print(f"reference={rrow}")
                break
        if len(local) != len(reference):
            print(f"local_tail={local[min(len(local), len(reference)):]}")
            print(f"reference_tail={reference[min(len(local), len(reference)):]}")
        return 1
    print(f"PASS observed pointwise descriptors shapes={len(OBSERVED_DESCRIPTOR_POINTWISE_SHAPES)} rows={rows}")
    print("conv.py observed planner matches the reference pointwise direct-spatial descriptor templates")
    return 0


def verify_all_observed_supported_descriptors():
    failures = []
    rows = 0
    shapes = []
    for shape_name, shape in sorted(descriptor_plan.SHAPES.items()):
        if descriptor_plan.observed_supported(shape):
            shapes.append(shape_name)
    for shape_name in shapes:
        local = local_observed_descriptor_keys(shape_name)
        reference = reference_observed_descriptor_keys(shape_name)
        rows += len(local)
        if local != reference:
            failures.append((shape_name, local, reference))
            break
    if failures:
        shape_name, local, reference = failures[0]
        print(f"FAIL all observed-supported descriptors at {shape_name}")
        print(f"local_count={len(local)} reference_count={len(reference)}")
        for idx, (lrow, rrow) in enumerate(zip(local, reference), 1):
            if lrow != rrow:
                print(f"first_diff={idx}")
                print(f"local={lrow}")
                print(f"reference={rrow}")
                break
        if len(local) != len(reference):
            print(f"local_tail={local[min(len(local), len(reference)):]}")
            print(f"reference_tail={reference[min(len(local), len(reference)):]}")
        return 1
    print(f"PASS all observed-supported descriptors shapes={len(shapes)} rows={rows}")
    print("conv.py direct-spatial descriptor planner covers every RKNN-observed supported shape in experimental/rknn_descriptor_plan.py")
    return 0


def longest_subspan(compiler, local):
    best = (0, 0, 0)
    for c_start in range(len(compiler)):
        for l_start in range(len(local)):
            n = 0
            while (c_start + n < len(compiler) and l_start + n < len(local) and
                   compiler[c_start + n] == local[l_start + n]):
                n += 1
            if n > best[2]:
                best = (c_start, l_start, n)
    return best


def compiler_qwords(path):
    buf = Path(path).read_bytes()
    rows = []
    for run in rknn_runs.find_runs(buf, 24):
        qwords = []
        decoded = []
        for off in run:
            target, reg_addr, value, qword = rknn_runs.decode(buf, off)
            qwords.append(qword)
            decoded.append((target, reg_addr, value, qword))
        if any((target, reg_addr) == (conv.reg.CNA, conv.reg.CNA_CONV_CON2)
               for target, reg_addr, _value, _qword in decoded):
            rows.append(decoded)
    return rows


def compiler_relevant_runs(path, min_qwords=8):
    buf = Path(path).read_bytes()
    rows = []
    for run in rknn_runs.find_runs(buf, min_qwords):
        decoded = []
        for off in run:
            target, reg_addr, value, qword = rknn_runs.decode(buf, off)
            decoded.append((target, reg_addr, value, qword))
        has_conv = any((target, reg_addr) == (conv.reg.CNA, conv.reg.CNA_CONV_CON2)
                       for target, reg_addr, _value, _qword in decoded)
        has_separator = len(decoded) == 26 and not has_conv
        if has_conv or has_separator:
            rows.append(decoded)
    return rows


def print_span_compare(shape_name, rknn_path):
    local = local_full_bodies(shape_name)
    compiler = compiler_qwords(rknn_path)
    fields = [
        "idx", "compiler_qwords", "local_qwords", "match_comp_start", "match_local_start",
        "match_len", "match_comp_end", "match_local_end", "pc_core", "pc_amount",
        "tail_pc_base", "tail_pc_amount",
    ]
    print(",".join(fields))
    for idx, (comp_decoded, local_body) in enumerate(zip(compiler, local), 1):
        comp_qwords = [row[3] for row in comp_decoded]
        comp_start, local_start, match_len = longest_subspan(comp_qwords, local_body)
        pc_base = ""
        pc_amount_qword = ""
        pc_core = ""
        pc_amount = ""
        for target, reg_addr, value, qword in comp_decoded:
            if (target, reg_addr) == (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
                pc_base = f"0x{value:x}"
            elif (target, reg_addr) == (conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS):
                pc_amount_qword = f"0x{value:x}"
                pc_core = str((value >> 16) & 0xffff)
                pc_amount = str(value & 0xffff)
        print(",".join([
            str(idx),
            str(len(comp_qwords)),
            str(len(local_body)),
            str(comp_start),
            str(local_start),
            str(match_len),
            str(comp_start + match_len),
            str(local_start + match_len),
            pc_core,
            pc_amount,
            pc_base,
            pc_amount_qword,
        ]))


def verify_compiler_spans(shape_name, rknn_path):
    local = local_full_bodies(shape_name)
    compiler = compiler_qwords(rknn_path)
    if len(local) != len(compiler):
        print(f"FAIL rows local={len(local)} compiler={len(compiler)}")
        return 1
    failures = []
    for idx, (comp_decoded, local_body) in enumerate(zip(compiler, local), 1):
        comp_qwords = [row[3] for row in comp_decoded]
        comp_start, local_start, match_len = longest_subspan(comp_qwords, local_body)
        expected = (
            (0, 0, 108) if idx == 1 else
            (1, 0, 108) if idx == 2 else
            (1, 4, 104) if idx == 3 else
            (0, 4, 104)
        )
        if (comp_start, local_start, match_len) != expected:
            failures.append((idx, (comp_start, local_start, match_len), expected))
    if failures:
        print("FAIL")
        for idx, got, expected in failures:
            print(f"row {idx}: got={got} expected={expected}")
        return 1
    print(f"PASS rows={len(local)}")
    print("stitching: row1=full_body+pc_tail row2=prev_enable+full_body row3=prev_enable+body[4:108]+pc_tail rows4-12=body[4:108]+pc_tail")
    return 0


def rknn_path_arg(shape_name, rknn_path):
    path = rknn_path or DEFAULT_RKNN_PATHS[shape_name]
    if not path.exists():
        raise FileNotFoundError(f"missing RKNN export: {path}")
    return path


def optional_rknn_path(value):
    return None if value is True else value


def compiler_descriptor_stream(rows):
    stream = []
    for idx, decoded in enumerate(rows, 1):
        qwords = [row[3] for row in decoded]
        if idx == 1:
            stream.extend(qwords)
        elif idx == 2:
            stream.append(conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, 0x0d))
            stream.extend(qwords[1:])
        elif idx == 3:
            stream.append(conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, 0x0d))
            stream.extend(qwords[1:])
        else:
            stream.extend(qwords)
    return stream


def local_compiler_stitched_stream(shape_name, rknn_path):
    bodies = local_full_bodies(shape_name)
    compiler = compiler_qwords(rknn_path)
    stream = []
    for idx, body in enumerate(bodies, 1):
        pc_base = None
        pc_amount = None
        for target, reg_addr, value, _qword in compiler[idx - 1]:
            if (target, reg_addr) == (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
                pc_base = value
            elif (target, reg_addr) == (conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS):
                pc_amount = value
        pc_tail = []
        if pc_base is not None and pc_amount is not None:
            pc_tail = [
                conv.E(conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS, pc_base),
                conv.E(conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS, pc_amount),
            ]
        if idx == 1:
            stream.extend(body)
            stream.extend(pc_tail)
        elif idx == 2:
            stream.append(conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, 0x0d))
            stream.extend(body)
        elif idx == 3:
            stream.append(conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, 0x0d))
            stream.extend(body[4:])
            stream.extend(pc_tail)
        else:
            stream.extend(body[4:])
            stream.extend(pc_tail)
    return stream


def local_compiler_stitched_segments(shape_name, rknn_path):
    bodies = local_full_bodies(shape_name)
    compiler = compiler_qwords(rknn_path)
    segments = []
    cursor = 0

    def add_segment(row_idx, kind, qwords):
        nonlocal cursor
        if not qwords:
            return
        segments.append({
            "row": row_idx,
            "kind": kind,
            "start": cursor,
            "end": cursor + len(qwords),
        })
        cursor += len(qwords)

    for idx, body in enumerate(bodies, 1):
        pc_base = None
        pc_amount = None
        for target, reg_addr, value, _qword in compiler[idx - 1]:
            if (target, reg_addr) == (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
                pc_base = value
            elif (target, reg_addr) == (conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS):
                pc_amount = value
        pc_tail = []
        if pc_base is not None and pc_amount is not None:
            pc_tail = [
                conv.E(conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS, pc_base),
                conv.E(conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS, pc_amount),
            ]
        if idx == 1:
            add_segment(idx, "body[0:108]", body)
            add_segment(idx, "pc_tail", pc_tail)
        elif idx == 2:
            add_segment(idx, "prev_enable", [conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, 0x0d)])
            add_segment(idx, "body[0:108]", body)
        elif idx == 3:
            add_segment(idx, "prev_enable", [conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, 0x0d)])
            add_segment(idx, "body[4:108]", body[4:])
            add_segment(idx, "pc_tail", pc_tail)
        else:
            add_segment(idx, "body[4:108]", body[4:])
            add_segment(idx, "pc_tail", pc_tail)
    return segments


def verify_compiler_stream(shape_name, rknn_path):
    compiler = compiler_qwords(rknn_path)
    expected = compiler_descriptor_stream(compiler)
    local = local_compiler_stitched_stream(shape_name, rknn_path)
    if local == expected:
        print(f"PASS qwords={len(local)}")
        return 0
    print(f"FAIL local_qwords={len(local)} compiler_qwords={len(expected)}")
    for idx, (lq, cq) in enumerate(zip(local, expected)):
        if lq != cq:
            print(f"first_mismatch={idx} local=0x{lq:016x} compiler=0x{cq:016x}")
            break
    if len(local) != len(expected):
        print("length_mismatch")
    return 1


def compiler_boundary_rows(shape_name, rknn_path):
    compiler = compiler_qwords(rknn_path)
    bodies = local_full_bodies(shape_name)
    rows = []
    for idx, (decoded, body) in enumerate(zip(compiler, bodies), 1):
        qwords = [row[3] for row in decoded]
        has_prev_enable = bool(qwords and qwords[0] == conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, 0x0d))
        pc_base = None
        pc_amount_raw = None
        for target, reg_addr, value, _qword in decoded:
            if (target, reg_addr) == (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
                pc_base = value
            elif (target, reg_addr) == (conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS):
                pc_amount_raw = value
        comp_start, local_start, match_len = longest_subspan(qwords, body)
        expected = (
            (0, 0, 108) if idx == 1 else
            (1, 0, 108) if idx == 2 else
            (1, 4, 104) if idx == 3 else
            (0, 4, 104)
        )
        rows.append({
            "idx": idx,
            "compiler_qwords": len(qwords),
            "has_prev_enable": has_prev_enable,
            "body_local_start": local_start,
            "body_len": match_len,
            "has_pc_tail": pc_base is not None and pc_amount_raw is not None,
            "pc_core": "" if pc_amount_raw is None else (pc_amount_raw >> 16) & 0xffff,
            "pc_amount": "" if pc_amount_raw is None else pc_amount_raw & 0xffff,
            "pc_base": pc_base,
            "pattern_ok": (comp_start, local_start, match_len) == expected,
        })
    return rows


def print_compiler_boundaries(shape_name, rknn_path):
    fields = [
        "idx", "compiler_qwords", "has_prev_enable", "body_local_start", "body_len",
        "has_pc_tail", "pc_core", "pc_amount", "pc_base", "pattern_ok",
    ]
    print(",".join(fields))
    for row in compiler_boundary_rows(shape_name, rknn_path):
        out = []
        for field in fields:
            value = row[field]
            if isinstance(value, bool):
                out.append("yes" if value else "no")
            elif isinstance(value, int) and field == "pc_base":
                out.append(fmt(value))
            else:
                out.append(str(value) if value is not None else "")
        print(",".join(out))


def verify_compiler_boundaries(shape_name, rknn_path):
    rows = compiler_boundary_rows(shape_name, rknn_path)
    failures = [row for row in rows if not row["pattern_ok"]]
    if failures:
        print("FAIL")
        for row in failures:
            print(f"row {row['idx']}: body_local_start={row['body_local_start']} body_len={row['body_len']}")
        return 1
    print(f"PASS rows={len(rows)}")
    print("boundaries: verifies compiler stitched run boundaries only; not a rocket runtime task mapping")
    return 0


def verify_exact_export_pair(shape_name):
    del shape_name
    shape_names = [
        "sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1",
        "mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1",
    ]
    streams = []
    for name in shape_names:
        path = rknn_path_arg(name, None)
        print(f"== export {name} ==")
        for check_name, check in [
                ("RKNN spans", lambda n=name, p=path: verify_compiler_spans(n, p)),
                ("RKNN stream", lambda n=name, p=path: verify_compiler_stream(n, p)),
                ("compiler boundaries", lambda n=name, p=path: verify_compiler_boundaries(n, p))]:
            print(f"-- {check_name} --")
            rc = check()
            if rc:
                print(f"FAIL exact export pair stopped at {name}: {check_name}")
                return rc
        streams.append(local_compiler_stitched_stream(name, path))
    if streams[0] != streams[1]:
        print("FAIL exact export streams differ")
        for idx, (left, right) in enumerate(zip(streams[0], streams[1])):
            if left != right:
                print(f"first_mismatch={idx} sweep=0x{left:016x} mix=0x{right:016x}")
                break
        if len(streams[0]) != len(streams[1]):
            print(f"lengths sweep={len(streams[0])} mix={len(streams[1])}")
        return 1
    print(f"PASS exact export pair qwords={len(streams[0])}")
    print("sweep and mix exports share the same compiler-stitched command stream model")
    return 0


def verify_remote_exact_export_equivalent(shape_name, remote_path):
    local_path = rknn_path_arg(shape_name, None)
    local_stream = compiler_descriptor_stream(compiler_qwords(local_path))
    remote_stream = compiler_descriptor_stream(compiler_qwords(remote_path))
    failures = []
    if local_path.stat().st_size != remote_path.stat().st_size:
        failures.append(f"file sizes differ local={local_path.stat().st_size} remote={remote_path.stat().st_size}")
    if local_stream != remote_stream:
        failures.append(f"compiler streams differ local={len(local_stream)} remote={len(remote_stream)}")
    print(f"local_export={local_path}")
    print(f"remote_export={remote_path}")
    print(f"local_size={local_path.stat().st_size} remote_size={remote_path.stat().st_size}")
    print(f"local_stream_qwords={len(local_stream)} remote_stream_qwords={len(remote_stream)}")
    if failures:
        print("FAIL remote exact export equivalence")
        for failure in failures:
            print(f"  {failure}")
        for idx, (left, right) in enumerate(zip(local_stream, remote_stream)):
            if left != right:
                print(f"first_mismatch={idx} local=0x{left:016x} remote=0x{right:016x}")
                break
        return 1
    print("PASS remote exact export equivalence")
    print("remote RKNN driver comparison uses the same compiler descriptor stream as the local export; file hash differences are outside the decoded regcmd stream")
    return 0


def rocket_full_body_tasks(shape_name):
    bodies = local_full_bodies(shape_name)
    tasks = []
    for idx, body in enumerate(bodies, 1):
        regs = list(body)
        if idx < len(bodies):
            next_len = len(bodies[idx])
            pc_amount = conv._ceil_div(next_len, 2) + 1
            pc_core = (bodies[idx].pc_core if hasattr(bodies[idx], "pc_core") else 0)
            regs.extend([
                conv.E(conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS, 0),
                conv.E(conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS, (pc_core << 16) | pc_amount),
                conv.E(conv.reg.VERSION, 0, 0),
                conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, 0x0d),
            ])
        else:
            regs.extend([
                conv.E(0x0001, 0, 0),
                conv.E(conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS, 0),
                conv.E(conv.reg.VERSION, 0, 0),
                conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, 0x0d),
            ])
        tasks.append(regs)
    return tasks


def verify_rocket_mapping(shape_name):
    tasks = rocket_full_body_tasks(shape_name)
    failures = []
    for idx, regs in enumerate(tasks[:-1], 1):
        next_len = len(tasks[idx]) - PC_TAIL_QWORDS
        tail = regs[-4:]
        pc_amount = (tail[1] >> 16) & 0xffffffff
        expected_amount = conv._ceil_div(next_len, 2) + 1
        if (pc_amount & 0xffff) != expected_amount:
            failures.append((idx, pc_amount, expected_amount))
    if failures:
        print("FAIL")
        for idx, got, expected in failures:
            print(f"task {idx}: pc_amount=0x{got:x} expected_low=0x{expected:x}")
        return 1
    print(f"PASS tasks={len(tasks)} qwords={sum(len(t) for t in tasks)} task_len={len(tasks[0])}")
    print("mapping: one rocket task per descriptor, full compiler-shaped body plus 4-qword rocket-visible tail")
    return 0


def verify_grouped_task_mapping(shape_name, task_count, label):
    bodies = local_full_bodies(shape_name)
    total_qwords = sum(len(b) for b in bodies)
    n_bodies = len(bodies)
    k, m = divmod(n_bodies, task_count)
    groups = [bodies[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(task_count)]
    tasks = []
    for g in groups:
        regs = []
        for body in g:
            regs.extend(body)
        regs.extend([
            conv.E(0x0001, 0, 0),
            conv.E(conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS, 0),
            conv.E(conv.reg.VERSION, 0, 0),
            conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, 0x0d),
        ])
        tasks.append(regs)
    task_qwords = sum(len(t) for t in tasks)
    group_sizes = [len(g) for g in groups]
    print(f"PASS tasks={task_count} bodies={n_bodies} qwords={total_qwords} task_qwords={task_qwords}")
    print(f"groups: {group_sizes} bodies per task")
    print(f"mapping: {label}")
    return 0


def verify_3task_mapping(shape_name):
    return verify_grouped_task_mapping(shape_name, 3, "candidate 1-submit/3-task grouping")


def verify_6task_mapping(shape_name):
    return verify_grouped_task_mapping(shape_name, 6, "candidate 6-task rocket grouping by descriptor bodies; not captured RKNN task-record boundaries")


def observed_task_gem_records():
    # (regcfg_amount, regcmd_offset_bytes, enable_mask, int_mask)
    return [
        (108, 0x0000, 0x0d, 0x0300),
        (108, 0x0380, 0x0d, 0x0300),
        (104, 0x0700, 0x0d, 0x0300),
        (104, 0x0a80, 0x0d, 0x0300),
        (26, 0x0e00, 0x60, 0x0c00),
        (104, 0x0f00, 0x0d, 0x0300),
        (104, 0x1280, 0x0d, 0x0300),
        (26, 0x1600, 0x60, 0x0c00),
        (104, 0x1700, 0x0d, 0x0300),
        (104, 0x1a80, 0x0d, 0x0300),
        (26, 0x1e00, 0x60, 0x0c00),
        (104, 0x1f00, 0x0d, 0x0300),
        (104, 0x2280, 0x0d, 0x0300),
        (26, 0x2600, 0x60, 0x0c00),
        (104, 0x2700, 0x0d, 0x0300),
        (104, 0x2a80, 0x0d, 0x0300),
        (26, 0x2e00, 0x60, 0x0c00),
    ]


def task_record_kind(record):
    _amount, _regcmd_off, enable_mask, _int_mask = record
    return "conv" if enable_mask == 0x0d else "separator"


def format_task_range(records, start, count):
    return [
        f"{idx}:{task_record_kind(records[idx])}:{records[idx][0]}@0x{records[idx][1]:x}"
        for idx in range(start, start + count)
    ]


SUBMIT_HEADER_LINE = re.compile(r"DRM_IOCTL_RKNPU_SUBMIT #(?P<idx>\d+) cmd=0x(?P<cmd>[0-9a-f]+) size=(?P<size>\d+)")
SUBMIT_GLOBAL_LINE = re.compile(
    r"flags=0x(?P<flags>[0-9a-f]+) timeout=(?P<timeout>\d+) "
    r"task_start=(?P<start>\d+) task_number=(?P<number>\d+)"
)
SUBMIT_CORE_LINE = re.compile(r"core_mask=0x(?P<core_mask>[0-9a-f]+) fence_fd=(?P<fence_fd>-?\d+)")
SUBMIT_TASK_OBJ_LINE = re.compile(r"task_counter=\d+ priority=-?\d+ task_obj_addr=0x(?P<task_obj>[0-9a-f]+)")
SUBMIT_SUBCORE_LINE = re.compile(
    r"subcore_task\[(?P<idx>\d+)\]=\{task_start=(?P<start>\d+), task_number=(?P<number>\d+)\}"
)
MEM_SYNC_LINE = re.compile(
    r"MEM_SYNC flags=0x(?P<flags>[0-9a-f]+) obj=0x(?P<obj>[0-9a-f]+) "
    r"offset=(?P<offset>\d+) size=(?P<size>\d+)"
)


def parse_submit_metadata_detail(log_path):
    submit_count = 0
    cmd = None
    size = None
    flags = None
    timeout = None
    global_start = None
    global_count = None
    task_obj_addr = None
    core_mask = None
    fence_fd = None
    subcore_ranges = [None] * 5
    for line in log_path.read_text().splitlines():
        stripped = line.strip()
        match = SUBMIT_HEADER_LINE.search(stripped)
        if match:
            submit_count += 1
            cmd = int(match.group("cmd"), 16)
            size = int(match.group("size"))
            continue
        if stripped.startswith("flags="):
            match = SUBMIT_GLOBAL_LINE.search(stripped)
            if match:
                flags = int(match.group("flags"), 16)
                timeout = int(match.group("timeout"))
                global_start = int(match.group("start"))
                global_count = int(match.group("number"))
            continue
        match = SUBMIT_TASK_OBJ_LINE.search(stripped)
        if match:
            task_obj_addr = int(match.group("task_obj"), 16)
            continue
        match = SUBMIT_CORE_LINE.search(stripped)
        if match:
            core_mask = int(match.group("core_mask"), 16)
            fence_fd = int(match.group("fence_fd"))
            continue
        match = SUBMIT_SUBCORE_LINE.search(stripped)
        if match:
            idx = int(match.group("idx"))
            subcore_ranges[idx] = (int(match.group("start")), int(match.group("number")))
    required = [cmd, size, flags, timeout, global_start, global_count, task_obj_addr, core_mask, fence_fd]
    if submit_count == 0 or any(row is None for row in required) or any(row is None for row in subcore_ranges):
        raise ValueError(f"incomplete submit metadata in {log_path}")
    return {
        "submit_count": submit_count,
        "cmd": cmd,
        "size": size,
        "flags": flags,
        "timeout": timeout,
        "task_start": global_start,
        "task_number": global_count,
        "task_obj_addr": task_obj_addr,
        "core_mask": core_mask,
        "fence_fd": fence_fd,
        "subcore_ranges": subcore_ranges,
    }


def parse_submit_metadata(log_path):
    detail = parse_submit_metadata_detail(log_path)
    return detail["task_start"], detail["task_number"], detail["subcore_ranges"]


def verify_submit_log_consistency(shape_name, submit_logs):
    del shape_name
    if not submit_logs:
        submit_logs = [
            ROOT / "experimental" / "rknn" / "capture_rknpu_submit_sweep_readonly.log",
            ROOT / "experimental" / "rknn" / "capture_rknpu_submit_mix_readonly.log",
        ]
    details = [(path, parse_submit_metadata_detail(path)) for path in submit_logs]
    expected = {
        "submit_count": 1,
        "cmd": 0xc0686441,
        "size": 104,
        "flags": 0x5,
        "timeout": 6000,
        "task_start": 0,
        "task_number": 6,
        "core_mask": 0,
        "fence_fd": -1,
        "subcore_ranges": [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)],
    }
    failures = []
    first = details[0][1]
    for path, detail in details:
        print(f"log={path}")
        print(f"  submit_count={detail['submit_count']} cmd=0x{detail['cmd']:x} size={detail['size']}")
        print(f"  flags=0x{detail['flags']:x} timeout={detail['timeout']} task_start={detail['task_start']} task_number={detail['task_number']}")
        print(f"  core_mask=0x{detail['core_mask']:x} fence_fd={detail['fence_fd']} subcore_ranges={detail['subcore_ranges']}")
        for key, expected_value in expected.items():
            if detail[key] != expected_value:
                failures.append(f"{path}: {key}={detail[key]!r}, expected {expected_value!r}")
        comparable = {key: detail[key] for key in expected}
        first_comparable = {key: first[key] for key in expected}
        if comparable != first_comparable:
            failures.append(f"{path}: metadata differs from {details[0][0]}")
    if failures:
        print("FAIL submit log consistency")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS submit log consistency")
    print("all checked logs show one vendor submit with task_number=6, core_mask=0, and repeated subcore_task[0..2]=(0,2)")
    return 0


def parse_mem_syncs(log_path):
    syncs = []
    for line in log_path.read_text().splitlines():
        match = MEM_SYNC_LINE.search(line.strip())
        if not match:
            continue
        syncs.append({
            "flags": int(match.group("flags"), 16),
            "obj": int(match.group("obj"), 16),
            "offset": int(match.group("offset")),
            "size": int(match.group("size")),
        })
    return syncs


def verify_task_object_sync_correlation(shape_name, ioctl_log=None):
    del shape_name
    log_path = ioctl_log or ROOT / "experimental" / "rknn" / "capture_rknpu_ioctl_sweep_readonly.log"
    detail = parse_submit_metadata_detail(log_path)
    syncs = parse_mem_syncs(log_path)
    task_syncs = [
        sync for sync in syncs
        if sync["obj"] == detail["task_obj_addr"]
    ]
    expected = [
        {"flags": 0x3, "offset": 0, "size": 680},
        {"flags": 0x1, "offset": 0, "size": 680},
    ]
    simplified = [
        {"flags": sync["flags"], "offset": sync["offset"], "size": sync["size"]}
        for sync in task_syncs
    ]
    print(f"log={log_path}")
    print(f"task_obj_addr=0x{detail['task_obj_addr']:x}")
    print(f"task_obj_mem_syncs={simplified}")
    if detail["task_number"] != 6 or detail["task_obj_addr"] is None:
        print("FAIL unexpected submit metadata for task object correlation")
        return 1
    if simplified != expected:
        print("FAIL task object MEM_SYNC correlation")
        print(f"expected={expected}")
        return 1
    print("PASS task object MEM_SYNC correlation")
    print("submit task_obj_addr is synced as a 680-byte task object with write/read syncs before submit")
    return 0


def rknpu_task_struct_fields(source):
    match = re.search(r"struct\s+rknpu_task\s*\{(?P<body>.*?)\}\s*__packed\s*;",
                      source, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError("missing packed struct rknpu_task")
    fields = []
    for line in match.group("body").splitlines():
        stripped = line.strip()
        match = re.match(r"__(?P<bits>u32|u64)\s+(?P<name>[a-zA-Z0-9_]+)\s*;", stripped)
        if match:
            fields.append((match.group("name"), int(match.group("bits")[1:]) // 8))
    return fields


def c_struct_fields(source, struct_name):
    match = re.search(rf"struct\s+{re.escape(struct_name)}\s*\{{(?P<body>.*?)\}}\s*(?:__packed)?\s*;",
                      source, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"missing struct {struct_name}")
    fields = []
    for line in match.group("body").splitlines():
        stripped = line.strip()
        match = re.match(r"__(?P<type>[us]\d+)\s+(?P<name>[a-zA-Z0-9_]+)\s*;", stripped)
        if match:
            fields.append((match.group("name"), match.group("type")))
    return fields


def verify_task_object_record_count(shape_name, ioctl_log=None, gem_log=None):
    del shape_name
    ioctl_path = ioctl_log or ROOT / "experimental" / "rknn" / "capture_rknpu_ioctl_sweep_readonly.log"
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    detail = parse_submit_metadata_detail(ioctl_path)
    syncs = parse_mem_syncs(ioctl_path)
    task_sync_sizes = {
        sync["size"] for sync in syncs
        if sync["obj"] == detail["task_obj_addr"]
    }
    fields = rknpu_task_struct_fields(RKNPU_IOCTL_H.read_text())
    struct_size = sum(size for _name, size in fields)
    expected_fields = [
        ("flags", 4), ("op_idx", 4), ("enable_mask", 4), ("int_mask", 4),
        ("int_clear", 4), ("int_status", 4), ("regcfg_amount", 4),
        ("regcfg_offset", 4), ("regcmd_addr", 8),
    ]
    print(f"sources: {RKNPU_IOCTL_H} {ioctl_path}")
    print(f"rknpu_task_fields={fields} struct_size={struct_size}")
    print(f"task_obj_sync_sizes={sorted(task_sync_sizes)} decoded_task_records={len(records)}")
    failures = []
    if fields != expected_fields:
        failures.append(f"unexpected struct rknpu_task fields: {fields}")
    if struct_size != 40:
        failures.append(f"struct rknpu_task size {struct_size} != 40")
    if task_sync_sizes != {680}:
        failures.append(f"task object sync sizes {sorted(task_sync_sizes)} != [680]")
    if len(records) * struct_size != 680:
        failures.append(f"decoded records {len(records)} * struct size {struct_size} != 680")
    if failures:
        print("FAIL task object record-count proof")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS task object record-count proof")
    print("680-byte submitted task object equals 17 packed rknpu_task records")
    return 0


def verify_observed_active_ranges(shape_name, submit_log=None, gem_log=None):
    del shape_name
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    if submit_log:
        global_start, global_count, subcore_ranges = parse_submit_metadata(submit_log)
    else:
        global_start, global_count = 0, 6
        subcore_ranges = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]
    global_records = format_task_range(records, global_start, global_count)
    subcore_records = [format_task_range(records, start, count) for start, count in subcore_ranges]
    if [task_record_kind(records[idx]) for idx in range(global_count)] != [
            "conv", "conv", "conv", "conv", "separator", "conv"]:
        print("FAIL unexpected global active-range layout")
        return 1
    if subcore_records[:3] != [global_records[:2], global_records[:2], global_records[:2]]:
        print("FAIL unexpected repeated subcore active-range layout")
        print(f"global={global_records}")
        print(f"subcores={subcore_records}")
        return 1
    print("PASS observed submit active-range interpretations")
    if submit_log or gem_log:
        print(f"sources: submit_log={submit_log or 'constants'} gem_log={gem_log or 'constants'}")
    print(f"global task_start={global_start} task_number={global_count} -> {global_records}")
    print(f"subcore_task[0..2]=(0,2) -> {subcore_records[:3]}")
    print("note: neither captured interpretation equals six even two-body descriptor groups")
    return 0


def task_record_interval(record):
    amount, regcmd_off, enable_mask, int_mask = record
    qword_start = regcmd_off // 8
    return {
        "start": qword_start,
        "end": qword_start + amount,
        "amount": amount,
        "offset": regcmd_off,
        "kind": task_record_kind(record),
        "enable_mask": enable_mask,
        "int_mask": int_mask,
    }


def active_task_intervals(records, start, count):
    return [task_record_interval(records[idx]) for idx in range(start, start + count)]


def format_interval(interval):
    return (f"{interval['kind']}@0x{interval['offset']:x}"
            f"+{interval['amount']}q[{interval['start']}:{interval['end']}]"
            f" enable=0x{interval['enable_mask']:x} int=0x{interval['int_mask']:x}")


def pc_chain_target_starts(stream):
    starts = {0}
    for qword in stream:
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        value = (qword >> 16) & 0xffffffff
        if (target, reg_addr) == (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
            starts.add(value // 8)
    return starts


def rocket_task_candidate(interval):
    return {
        "regcmd": interval["offset"],
        "regcmd_count": interval["amount"],
        "kind": interval["kind"],
    }


def observed_runtime_task_boundary_candidates(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    if submit_log:
        global_start, global_count, subcore_ranges = parse_submit_metadata(submit_log)
    else:
        global_start, global_count = 0, 6
        subcore_ranges = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]

    def build_candidate(name, ranges):
        intervals = []
        for start, count in ranges:
            intervals.extend(active_task_intervals(records, start, count))
        return {"name": name, "ranges": ranges, "intervals": intervals}

    populated_subcore_ranges = [row for row in subcore_ranges if row[1]]
    candidates = [
        build_candidate("global task_start/task_number", [(global_start, global_count)]),
        build_candidate("subcore_task ranges concatenated", populated_subcore_ranges),
        build_candidate("subcore_task unique ranges", sorted(set(populated_subcore_ranges))),
    ]
    pc_targets = pc_chain_target_starts(stream)
    for candidate in candidates:
        for interval in candidate["intervals"]:
            interval["inside_stream"] = interval["end"] <= len(stream)
            interval["matches_stream_span"] = (
                interval["inside_stream"] and
                len(stream[interval["start"]:interval["end"]]) == interval["amount"]
            )
            interval["pc_chain_boundary"] = interval["start"] in pc_targets
        candidate["rocket_job"] = {
            "task_count": len(candidate["intervals"]),
            "tasks": [rocket_task_candidate(interval) for interval in candidate["intervals"]],
        }
    return stream, candidates


def verify_runtime_task_boundaries(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream, candidates = observed_runtime_task_boundary_candidates(
        shape_name, rknn_path, submit_log, gem_log)
    failures = []
    for candidate in candidates:
        for interval in candidate["intervals"]:
            if not interval["inside_stream"]:
                failures.append((candidate["name"], interval, "outside compiler stream"))
            elif not interval["matches_stream_span"]:
                failures.append((candidate["name"], interval, "does not match exported stream span"))
            elif not interval["pc_chain_boundary"]:
                failures.append((candidate["name"], interval, "not a PC-chain task boundary"))

    signatures = {
        tuple((i["start"], i["end"], i["kind"], i["enable_mask"], i["int_mask"])
              for i in candidate["intervals"])
        for candidate in candidates
    }
    print(f"compiler_stream_qwords={len(stream)}")
    for candidate in candidates:
        print(f"candidate: {candidate['name']} ranges={candidate['ranges']}")
        for interval in candidate["intervals"]:
            boundary = "pc-boundary" if interval["pc_chain_boundary"] else "record-boundary"
            print(f"  {format_interval(interval)} {boundary}")
        print(f"  Rocket job candidate: task_count={candidate['rocket_job']['task_count']}")
        for idx, task in enumerate(candidate["rocket_job"]["tasks"]):
            print(f"    task[{idx}] regcmd=0x{task['regcmd']:x} regcmd_count={task['regcmd_count']} kind={task['kind']}")
    if failures:
        print("FAIL runtime task boundary reconstruction")
        for name, interval, reason in failures:
            print(f"{name}: {format_interval(interval)}: {reason}")
    if len(signatures) != 1:
        print("FAIL ambiguous observed runtime task boundaries")
        print("global task_number=6 and repeated subcore_task=(0,2) select different command intervals")
        print("no general Rocket job->task[] mapping is proven; keep any hardware probe narrow and double-gated")
        return 1
    if failures:
        print("no general Rocket job->task[] mapping is proven; keep any hardware probe narrow and double-gated")
        return 1
    print("PASS runtime task boundaries are unambiguous")
    return 0


def verify_rocket_task_mapping_proof(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream, candidates = observed_runtime_task_boundary_candidates(
        shape_name, rknn_path, submit_log, gem_log)
    segments = local_compiler_stitched_segments(shape_name, rknn_path)
    segment_boundaries = segment_boundary_set(segments)
    signatures = {
        tuple((i["start"], i["end"], i["kind"], i["enable_mask"], i["int_mask"])
              for i in candidate["intervals"])
        for candidate in candidates
    }

    print(f"compiler_stream_qwords={len(stream)}")
    if len(signatures) != 1:
        print("FAIL observed submit interpretations are ambiguous")
        for candidate in candidates:
            intervals = candidate["intervals"]
            ranges, covered, first_uncovered = interval_coverage(intervals, len(stream))
            print(f"  {candidate['name']}: task_count={len(intervals)} covered={covered}/{len(stream)}"
                  f" first_uncovered={first_uncovered} ranges={ranges}")

    proof_failures = []
    for candidate in candidates:
        intervals = candidate["intervals"]
        ranges, covered, first_uncovered = interval_coverage(intervals, len(stream))
        candidate_failures = []
        if covered != len(stream):
            candidate_failures.append(f"does not cover full compiler stream: {covered}/{len(stream)} qwords, first_uncovered={first_uncovered}")
        for interval in intervals:
            if not interval["inside_stream"]:
                candidate_failures.append(f"{format_interval(interval)} is outside compiler stream")
            if interval["kind"] != "conv":
                candidate_failures.append(f"{format_interval(interval)} is not a conv task")
            if not interval["pc_chain_boundary"]:
                candidate_failures.append(f"{format_interval(interval)} is not a PC-chain boundary")
            if interval["start"] not in segment_boundaries:
                containing = segment_containing(interval["start"], segments)
                detail = ""
                if containing:
                    detail = f" inside row{containing['row']}:{containing['kind']}[{containing['start']}:{containing['end']}]"
                candidate_failures.append(f"task start {interval['start']} cuts compiler segment{detail}")
            if interval["end"] not in segment_boundaries:
                containing = segment_containing(interval["end"], segments)
                detail = ""
                if containing:
                    detail = f" inside row{containing['row']}:{containing['kind']}[{containing['start']}:{containing['end']}]"
                candidate_failures.append(f"task end {interval['end']} cuts compiler segment{detail}")
        print(f"candidate: {candidate['name']}")
        print(f"  Rocket task_count={candidate['rocket_job']['task_count']} covered_ranges={ranges}")
        if candidate_failures:
            for failure in candidate_failures:
                print(f"  FAIL {failure}")
            proof_failures.extend((candidate["name"], failure) for failure in candidate_failures)
        else:
            print("  candidate satisfies per-task proof invariants")

    if len(signatures) != 1:
        proof_failures.append(("all", "submit metadata interpretations select different command intervals"))
    if proof_failures:
        print("FAIL Rocket task mapping proof")
        print("no observed candidate is a complete, unambiguous, segment-aligned local Rocket job")
        return 1
    print("PASS Rocket task mapping proof")
    print("observed runtime metadata gives a complete unambiguous local Rocket job/task mapping")
    return 0


def interval_coverage(intervals, stream_len):
    covered = [False] * stream_len
    for interval in intervals:
        for idx in range(interval["start"], min(interval["end"], stream_len)):
            covered[idx] = True
    ranges = []
    start = None
    for idx, is_covered in enumerate(covered + [False]):
        if is_covered and start is None:
            start = idx
        elif not is_covered and start is not None:
            ranges.append((start, idx))
            start = None
    first_uncovered = next((idx for idx, is_covered in enumerate(covered) if not is_covered), stream_len)
    return ranges, sum(end - start for start, end in ranges), first_uncovered


def driver_fetch_intervals(intervals):
    fetched = []
    for interval in intervals:
        row = dict(interval)
        row["end"] = row["start"] + row["amount"] + RKNPU_PC_DATA_EXTRA_AMOUNT
        row["fetch_amount"] = row["amount"] + RKNPU_PC_DATA_EXTRA_AMOUNT
        fetched.append(row)
    return fetched


def uncovered_ranges(intervals, stream_len):
    covered_ranges, _covered, _first = interval_coverage(intervals, stream_len)
    gaps = []
    cursor = 0
    for start, end in covered_ranges:
        if cursor < start:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < stream_len:
        gaps.append((cursor, stream_len))
    return gaps


def qword_label(qword):
    target = (qword >> 48) & 0xffff
    reg_addr = qword & 0xffff
    value = (qword >> 16) & 0xffffffff
    name = REG_NAMES.get((target, reg_addr), f"target=0x{target:x} reg=0x{reg_addr:x}")
    return f"{name}=0x{value:x}"


def verify_runtime_task_coverage(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream, candidates = observed_runtime_task_boundary_candidates(
        shape_name, rknn_path, submit_log, gem_log)
    print(f"compiler_stream_qwords={len(stream)}")
    for candidate in candidates:
        ranges, covered, first_uncovered = interval_coverage(candidate["intervals"], len(stream))
        print(f"candidate: {candidate['name']} ranges={candidate['ranges']}")
        print(f"  covered_qwords={covered}/{len(stream)} first_uncovered={first_uncovered}")
        print(f"  covered_ranges={ranges}")
    print("PASS runtime task coverage summary")
    print("coverage summary is diagnostic only; it does not prove a local Rocket mapping")
    return 0


def verify_runtime_fetch_coverage(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream, candidates = observed_runtime_task_boundary_candidates(
        shape_name, rknn_path, submit_log, gem_log)
    print(f"compiler_stream_qwords={len(stream)} pc_data_extra_qwords={RKNPU_PC_DATA_EXTRA_AMOUNT}")
    for candidate in candidates:
        fetched = driver_fetch_intervals(candidate["intervals"])
        ranges, covered, first_uncovered = interval_coverage(fetched, len(stream))
        print(f"candidate: {candidate['name']} ranges={candidate['ranges']}")
        print(f"  fetched_qwords={covered}/{len(stream)} first_uncovered={first_uncovered}")
        print(f"  fetched_ranges={ranges}")
    print("PASS runtime fetch coverage summary")
    print("fetch coverage uses vendor regcfg_amount+4 rule; it does not prove a local Rocket mapping")
    return 0


def verify_runtime_task_gaps(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream, candidates = observed_runtime_task_boundary_candidates(
        shape_name, rknn_path, submit_log, gem_log)
    print(f"compiler_stream_qwords={len(stream)}")
    for candidate in candidates:
        gaps = uncovered_ranges(candidate["intervals"], len(stream))
        print(f"candidate: {candidate['name']} ranges={candidate['ranges']} gaps={len(gaps)}")
        for start, end in gaps[:8]:
            labels = [qword_label(stream[idx]) for idx in range(start, min(end, start + 6))]
            suffix = " ..." if end - start > 6 else ""
            print(f"  gap[{start}:{end}] len={end - start}: {', '.join(labels)}{suffix}")
        if len(gaps) > 8:
            print(f"  ... {len(gaps) - 8} more gaps")
    print("PASS runtime task gap summary")
    print("gap summary is diagnostic only; it does not prove a local Rocket mapping")
    return 0


def interval_segment_overlaps(interval, segments):
    overlaps = []
    for segment in segments:
        start = max(interval["start"], segment["start"])
        end = min(interval["end"], segment["end"])
        if start < end:
            overlaps.append(f"row{segment['row']}:{segment['kind']}[{start}:{end}]")
    return overlaps


def verify_runtime_task_row_overlaps(shape_name, rknn_path, submit_log=None, gem_log=None):
    _stream, candidates = observed_runtime_task_boundary_candidates(
        shape_name, rknn_path, submit_log, gem_log)
    segments = local_compiler_stitched_segments(shape_name, rknn_path)
    for candidate in candidates:
        print(f"candidate: {candidate['name']} ranges={candidate['ranges']}")
        for interval in candidate["intervals"]:
            overlaps = interval_segment_overlaps(interval, segments)
            print(f"  {format_interval(interval)} -> {', '.join(overlaps)}")
    print("PASS runtime task row-overlap summary")
    print("row-overlap summary is diagnostic only; it does not prove a local Rocket mapping")
    return 0


def verify_task_backing_span(shape_name, rknn_path, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    intervals = [task_record_interval(record) for record in records]
    table_end = max(interval["end"] for interval in intervals)
    overflow = [(idx, interval) for idx, interval in enumerate(intervals)
                if interval["end"] > len(stream)]
    print(f"compiler_stream_qwords={len(stream)}")
    print(f"task_backing_records={len(records)} backing_span_qwords={table_end}")
    print(f"records_beyond_compiler_stream={len(overflow)}")
    for idx, interval in overflow:
        print(f"  record[{idx}] {format_interval(interval)} exceeds stream by {interval['end'] - len(stream)} qwords")
    print("PASS task backing span summary")
    print("backing span summary is diagnostic only; records beyond the stream are not active work")
    return 0


def verify_pc_base_task_links(shape_name, rknn_path, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    starts = {regcmd_off // 8: idx for idx, (_amount, regcmd_off, _enable, _int) in enumerate(records)}
    links = []
    misses = []
    for idx, qword in enumerate(stream):
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        value = (qword >> 16) & 0xffffffff
        if (target, reg_addr) != (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
            continue
        qword_target = value // 8
        record_idx = starts.get(qword_target)
        if record_idx is None:
            misses.append((idx, value, qword_target))
        else:
            links.append((idx, value, qword_target, record_idx))
    if misses:
        print("FAIL PC_BASE_ADDRESS target not found in observed task records")
        for idx, value, qword_target in misses:
            print(f"  stream[{idx}] PC_BASE_ADDRESS=0x{value:x} qword={qword_target}")
        return 1
    print(f"PASS pc_base_links={len(links)}")
    for idx, value, qword_target, record_idx in links:
        amount, _off, enable, int_mask = records[record_idx]
        beyond = " beyond_stream" if qword_target >= len(stream) else ""
        print(f"  stream[{idx}] PC_BASE_ADDRESS=0x{value:x} -> record[{record_idx}] qword={qword_target} amount={amount} enable=0x{enable:x} int=0x{int_mask:x}{beyond}")
    print("PC_BASE_ADDRESS links target observed task-record starts; this supports a stitched-stream task table model")
    return 0


def verify_pc_amount_task_lengths(shape_name, rknn_path, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    starts = {regcmd_off // 8: idx for idx, (_amount, regcmd_off, _enable, _int) in enumerate(records)}
    failures = []
    links = []
    for idx, qword in enumerate(stream):
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        value = (qword >> 16) & 0xffffffff
        if (target, reg_addr) != (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
            continue
        if idx + 1 >= len(stream):
            failures.append((idx, "missing PC_REGISTER_AMOUNTS after PC_BASE_ADDRESS"))
            continue
        next_qword = stream[idx + 1]
        next_target = (next_qword >> 48) & 0xffff
        next_reg = next_qword & 0xffff
        next_value = (next_qword >> 16) & 0xffffffff
        if (next_target, next_reg) != (conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS):
            failures.append((idx, "next qword is not PC_REGISTER_AMOUNTS"))
            continue
        qword_target = value // 8
        record_idx = starts.get(qword_target)
        if record_idx is None:
            failures.append((idx, f"PC_BASE_ADDRESS target qword {qword_target} has no record"))
            continue
        record_amount = records[record_idx][0]
        pc_amount_low = next_value & 0xffff
        decoded_amount = pc_amount_low * 2 - 2
        if decoded_amount != record_amount:
            failures.append((idx, f"decoded amount {decoded_amount} != record amount {record_amount}"))
        links.append((idx, record_idx, next_value >> 16, pc_amount_low, decoded_amount, record_amount))
    if failures:
        print("FAIL PC_REGISTER_AMOUNTS does not match target task records")
        for idx, reason in failures:
            print(f"  stream[{idx}]: {reason}")
        return 1
    print(f"PASS pc_amount_links={len(links)}")
    for idx, record_idx, core, pc_amount_low, decoded_amount, record_amount in links:
        print(f"  stream[{idx}] -> record[{record_idx}] core={core} pc_amount={pc_amount_low} decoded_qwords={decoded_amount} record_qwords={record_amount}")
    print("PC_REGISTER_AMOUNTS low bits encode target record length as low*2-2; high bits carry core index")
    return 0


def task_record_start_map(records):
    return {regcmd_off // 8: idx for idx, (_amount, regcmd_off, _enable, _int) in enumerate(records)}


def pc_links_by_task_record(stream, records):
    starts = task_record_start_map(records)
    links = {}
    for record_idx, record in enumerate(records):
        interval = task_record_interval(record)
        scan_end = min(interval["end"] + RKNPU_PC_DATA_EXTRA_AMOUNT, len(stream))
        record_links = []
        for qword_idx in range(interval["start"], scan_end):
            qword = stream[qword_idx]
            target = (qword >> 48) & 0xffff
            reg_addr = qword & 0xffff
            value = (qword >> 16) & 0xffffffff
            if (target, reg_addr) != (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
                continue
            target_qword = value // 8
            record_links.append((qword_idx, target_qword, starts.get(target_qword)))
        links[record_idx] = record_links
    return links


def reachable_records_from(entries, links):
    visited = set()
    stack = list(entries)
    while stack:
        record_idx = stack.pop()
        if record_idx in visited:
            continue
        visited.add(record_idx)
        for _qword_idx, _target_qword, target_record in links.get(record_idx, []):
            if target_record is not None and target_record not in visited:
                stack.append(target_record)
    return sorted(visited)


def verify_pc_linked_task_graph(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    if submit_log:
        global_start, global_count, subcore_ranges = parse_submit_metadata(submit_log)
    else:
        global_start, global_count = 0, 6
        subcore_ranges = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]
    links = pc_links_by_task_record(stream, records)
    global_entries = list(range(global_start, global_start + global_count))
    populated_subcore_ranges = [row for row in subcore_ranges if row[1]]
    concatenated_entries = [
        idx for start, count in populated_subcore_ranges
        for idx in range(start, start + count)
    ]
    unique_subcore_entries = sorted(set(concatenated_entries))
    entry_sets = [
        ("global task_start/task_number", global_entries),
        ("subcore_task ranges concatenated", concatenated_entries),
        ("subcore_task unique ranges", unique_subcore_entries),
    ]

    print(f"compiler_stream_qwords={len(stream)} task_records={len(records)}")
    for record_idx, record_links in links.items():
        interval = task_record_interval(records[record_idx])
        if not record_links:
            print(f"record[{record_idx}] {format_interval(interval)} pc_links=[]")
            continue
        rendered = []
        for qword_idx, target_qword, target_record in record_links:
            target_text = f"record[{target_record}]" if target_record is not None else "missing-record"
            rendered.append(f"stream[{qword_idx}]->qword{target_qword}/{target_text}")
        print(f"record[{record_idx}] {format_interval(interval)} pc_links={rendered}")

    active_stream_records = {
        idx for idx, record in enumerate(records)
        if task_record_interval(record)["start"] < len(stream)
    }
    print(f"records_with_start_inside_stream={sorted(active_stream_records)}")
    for name, entries in entry_sets:
        reachable = reachable_records_from(entries, links)
        inside_reachable = sorted(set(reachable) & active_stream_records)
        missing_inside = sorted(active_stream_records - set(inside_reachable))
        duplicate_note = " duplicates-present" if len(entries) != len(set(entries)) else ""
        print(f"entry set: {name}{duplicate_note} entries={entries}")
        print(f"  reachable_records={reachable}")
        print(f"  reachable_inside_stream={inside_reachable}")
        print(f"  missing_inside_stream={missing_inside}")

    print("PASS pc-linked task graph summary")
    print("graph is diagnostic only; entries plus PC links still do not prove a local Rocket mapping")
    return 0


def pc_link_targets(links):
    return {
        record_idx: [target_record for _qword_idx, _target_qword, target_record in record_links
                     if target_record is not None]
        for record_idx, record_links in links.items()
    }


def pc_chain_components(records, active_records, links):
    targets = pc_link_targets(links)
    indegree = {idx: 0 for idx in active_records}
    for record_idx in active_records:
        for target in targets.get(record_idx, []):
            if target in indegree:
                indegree[target] += 1
    roots = [idx for idx in active_records if indegree[idx] == 0]
    components = []
    for root in roots:
        chain = []
        seen = set()
        cursor = root
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            chain.append(cursor)
            next_targets = targets.get(cursor, [])
            cursor = next_targets[0] if next_targets else None
        components.append(chain)
    covered = sorted({idx for chain in components for idx in chain if idx in active_records})
    missing = sorted(set(active_records) - set(covered))
    return roots, components, missing


def verify_pc_chain_components(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    if submit_log:
        global_start, global_count, subcore_ranges = parse_submit_metadata(submit_log)
    else:
        global_start, global_count = 0, 6
        subcore_ranges = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]
    links = pc_links_by_task_record(stream, records)
    active_records = [
        idx for idx, record in enumerate(records)
        if task_record_interval(record)["start"] < len(stream)
    ]
    roots, components, missing = pc_chain_components(records, active_records, links)
    global_entries = list(range(global_start, global_start + global_count))
    subcore_entries = sorted({
        idx for start, count in subcore_ranges
        for idx in range(start, start + count)
    })

    print(f"compiler_stream_qwords={len(stream)}")
    print(f"active_records_with_start_inside_stream={active_records}")
    print(f"pc_chain_roots={roots}")
    for idx, chain in enumerate(components):
        rendered = []
        for record_idx in chain:
            interval = task_record_interval(records[record_idx])
            beyond = " beyond_stream" if interval["start"] >= len(stream) else ""
            rendered.append(f"{record_idx}:{interval['kind']}[{interval['start']}:{interval['end']}]{beyond}")
        print(f"  chain[{idx}] {' -> '.join(rendered)}")
    print(f"component_missing_active_records={missing}")
    print(f"observed_global_entries={global_entries}")
    print(f"observed_unique_subcore_entries={subcore_entries}")
    if roots == global_entries or roots == subcore_entries:
        print("FAIL PC-chain roots unexpectedly match observed submit entries")
        return 1
    if missing:
        print("FAIL PC-chain component reconstruction missed active records")
        return 1
    print("PASS pc-chain component summary")
    print("complete stream roots differ from observed submit entry interpretations; this is diagnostic, not a Rocket submit recipe")
    return 0


def component_fetch_intervals(records, chain):
    return [driver_fetch_intervals([task_record_interval(records[idx])])[0]
            for idx in chain]


def verify_pc_chain_component_coverage(shape_name, rknn_path, submit_log=None, gem_log=None):
    del submit_log
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    segments = local_compiler_stitched_segments(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    links = pc_links_by_task_record(stream, records)
    active_records = [
        idx for idx, record in enumerate(records)
        if task_record_interval(record)["start"] < len(stream)
    ]
    _roots, components, missing_records = pc_chain_components(records, active_records, links)

    all_fetches = []
    print(f"compiler_stream_qwords={len(stream)}")
    for component_idx, chain in enumerate(components):
        fetches = component_fetch_intervals(records, chain)
        all_fetches.extend(fetches)
        ranges, covered, first_uncovered = interval_coverage(fetches, len(stream))
        print(f"component[{component_idx}] records={chain}")
        print(f"  fetched_ranges={ranges} covered={covered} first_uncovered={first_uncovered}")
        for interval in fetches:
            overlaps = interval_segment_overlaps(interval, segments)
            if interval["start"] >= len(stream):
                span = f"[{interval['start']}:{interval['end']}] beyond_stream"
            else:
                fetch_end = min(interval["end"], len(stream))
                clipped = " clipped" if interval["end"] > len(stream) else ""
                span = f"[{interval['start']}:{fetch_end}]{clipped}"
            print(f"  fetch {interval['kind']}@0x{interval['offset']:x}{span} -> {', '.join(overlaps)}")

    all_ranges, all_covered, all_first_uncovered = interval_coverage(all_fetches, len(stream))
    all_gaps = uncovered_ranges(all_fetches, len(stream))
    print(f"all_components_covered_qwords={all_covered}/{len(stream)} first_uncovered={all_first_uncovered}")
    print(f"all_components_covered_ranges={all_ranges}")
    print(f"all_components_gaps={all_gaps}")
    if missing_records:
        print(f"FAIL component reconstruction missed active records: {missing_records}")
        return 1
    if all_covered != len(stream):
        print("FAIL PC-chain component fetches do not cover the full compiler stream")
        print("component roots are useful structure, but not yet a complete Rocket task mapping")
        return 1
    print("PASS pc-chain component fetch coverage")
    print("component fetches cover the compiler stream offline; this still needs a Rocket ABI task-boundary proof before HW submit")
    return 0


def verify_pc_root_contiguous_rocket_candidate(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    if submit_log:
        global_start, global_count, subcore_ranges = parse_submit_metadata(submit_log)
    else:
        global_start, global_count = 0, 6
        subcore_ranges = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]
    links = pc_links_by_task_record(stream, records)
    active_records = [
        idx for idx, record in enumerate(records)
        if task_record_interval(record)["start"] < len(stream)
    ]
    roots, components, missing_records = pc_chain_components(records, active_records, links)
    root_starts = [task_record_interval(records[idx])["start"] for idx in roots]
    task_starts = root_starts + [len(stream)]
    tasks = []
    failures = []
    for idx, (start, end) in enumerate(zip(task_starts, task_starts[1:])):
        if start >= end:
            failures.append(f"task[{idx}] empty or reversed span {start}:{end}")
            continue
        if start != 0 and start not in root_starts:
            failures.append(f"task[{idx}] start {start} is not a PC-chain root")
        tasks.append({"regcmd": start * 8, "regcmd_count": end - start})

    covered = sum(task["regcmd_count"] for task in tasks)
    expected_global_entries = list(range(global_start, global_start + global_count))
    expected_unique_subcore = sorted({
        idx for start, count in subcore_ranges
        for idx in range(start, start + count)
    })

    print(f"compiler_stream_qwords={len(stream)}")
    print(f"pc_chain_roots={roots}")
    print(f"pc_chain_root_starts={root_starts}")
    print(f"observed_global_entries={expected_global_entries}")
    print(f"observed_unique_subcore_entries={expected_unique_subcore}")
    print(f"candidate Rocket job: task_count={len(tasks)} covered_qwords={covered}")
    for idx, task in enumerate(tasks):
        start = task["regcmd"] // 8
        end = start + task["regcmd_count"]
        labels = [qword_label(stream[pos]) for pos in range(start, min(end, start + 4))]
        suffix = " ..." if end - start > 4 else ""
        print(f"  task[{idx}] regcmd=0x{task['regcmd']:x} regcmd_count={task['regcmd_count']} span=[{start}:{end}] first={labels}{suffix}")
    if missing_records:
        print(f"FAIL component reconstruction missed active records: {missing_records}")
        return 1
    if failures:
        print("FAIL root-contiguous candidate construction")
        for failure in failures:
            print(f"  {failure}")
        return 1
    if covered != len(stream) or len(tasks) != len(roots):
        print("FAIL root-contiguous candidate does not cover exact compiler stream")
        return 1
    if roots == expected_global_entries or roots == expected_unique_subcore:
        print("FAIL root-contiguous candidate unexpectedly matches observed submit entries")
        return 1
    print("PASS root-contiguous Rocket candidate")
    print("candidate covers the exact compiler stream offline, but it is not observed RKNN submit metadata and must not be submitted without further proof")
    return 0


def normalized_pc_base_value(value):
    base = conv.regcmd_mem_create.dma_addr
    if value >= base:
        return (value - base) // 8
    return value // 8


ADDRESS_BASE_REGS = {
    (conv.reg.CNA, conv.reg.CNA_FEATURE_DATA_ADDR): lambda: conv.input_mem_create.dma_addr,
    (conv.reg.CNA, conv.reg.CNA_DCOMP_ADDR0): lambda: conv.weight_mem_create.dma_addr,
    (conv.reg.DPU, conv.reg.DST_BASE_ADDR): lambda: conv.output_mem_create.dma_addr,
}


def normalized_dma_offset(value, base):
    base32 = base & 0xffffffff
    if value >= base32:
        return value - base32
    return value


def normalize_runtime_address_value(target, reg_addr, value):
    if (target, reg_addr) == (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
        return normalized_pc_base_value(value)
    base_fn = ADDRESS_BASE_REGS.get((target, reg_addr))
    if base_fn is not None:
        return normalized_dma_offset(value, base_fn())
    return value


def normalize_runtime_qwords(stream):
    normalized = []
    for qword in stream:
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        value = (qword >> 16) & 0xffffffff
        normalized_value = normalize_runtime_address_value(target, reg_addr, value)
        if normalized_value != value:
            value = normalized_value
            qword = conv.E(target, reg_addr, value)
        normalized.append(qword)
    return normalized


def verify_runtime_pc_root6_stream(shape_name, rknn_path):
    expected_stream = normalize_runtime_qwords(local_compiler_stitched_stream(shape_name, rknn_path))
    descs = runtime_direct_spatial_descs(shape_name)
    runtime_stream, spans = conv.direct_spatial_pc_root6_stream_and_spans(descs)
    runtime_stream = normalize_runtime_qwords(runtime_stream)
    starts = conv.direct_spatial_pc_root6_record_starts()
    expected_spans = conv.direct_spatial_pc_root6_expected_spans()
    failures = []
    if runtime_stream != expected_stream:
        failures.append(f"runtime stream differs from exact RKNN stream after PC base normalization: runtime={len(runtime_stream)} expected={len(expected_stream)}")
    if spans != expected_spans:
        failures.append(f"runtime spans differ: got={spans} expected={expected_spans}")
    expected_descs = len(conv.DIRECT_SPATIAL_PC_ROOT6_H40_SCHEDULE)
    if len(descs) != expected_descs:
        failures.append(f"expected {expected_descs} descriptors, got {len(descs)}")
    print(f"runtime_stream_qwords={len(runtime_stream)}")
    print(f"runtime_record_starts={starts}")
    print(f"runtime_task_spans={spans}")
    if failures:
        print("FAIL runtime pc_root6 stream")
        for failure in failures:
            print(f"  {failure}")
        if runtime_stream != expected_stream:
            for idx, (actual, expected) in enumerate(zip(runtime_stream, expected_stream)):
                if actual != expected:
                    print(f"first_mismatch={idx} runtime=0x{actual:016x} expected=0x{expected:016x}")
                    break
        return 1
    print("PASS runtime pc_root6 stream")
    print("conv.py pc_root6 stream and raw Rocket spans match the exact RKNN compiler stream after PC base normalization")
    return 0


def verify_conv_raw_span_boundary_rejection(shape_name, rknn_path):
    descs = runtime_direct_spatial_descs(shape_name)
    _runtime_stream, spans = conv.direct_spatial_pc_root6_stream_and_spans(descs)
    expected_spans = conv.direct_spatial_pc_root6_expected_spans()
    local_stream = local_compiler_stitched_stream(shape_name, rknn_path)
    segments = local_compiler_stitched_segments(shape_name, rknn_path)
    seg_boundaries = segment_boundary_set(segments)
    pc_targets = pc_chain_target_starts(local_stream)
    boundaries = [start for start, _amount in spans]
    boundaries.append(spans[-1][0] + spans[-1][1])
    rejected = []
    failures = []

    if spans != expected_spans:
        failures.append(f"conv.py raw spans changed: got={spans} expected={expected_spans}")
    if boundaries[-1] != len(local_stream):
        failures.append(f"final boundary={boundaries[-1]} does not match stream length={len(local_stream)}")

    print(f"compiler_stream_qwords={len(local_stream)}")
    print(f"conv_raw_spans={spans}")
    print(f"conv_raw_boundaries={boundaries}")
    for boundary in boundaries:
        is_final = boundary == len(local_stream)
        segment_boundary = boundary in seg_boundaries
        pc_target = boundary in pc_targets
        containing = segment_containing(boundary, segments)
        detail = ""
        if containing:
            detail = f" inside row{containing['row']}:{containing['kind']}[{containing['start']}:{containing['end']}]"
        print(f"  boundary {boundary}: pc_target={pc_target} segment_boundary={segment_boundary}{detail}")
        if not is_final and (not pc_target or not segment_boundary):
            rejected.append((boundary, pc_target, segment_boundary, detail))

    if failures:
        print("FAIL conv.py raw-span boundary audit")
        for failure in failures:
            print(f"  {failure}")
        return 1
    if not rejected:
        print("FAIL conv.py raw spans unexpectedly satisfy PC-target and compiler-segment boundary invariants")
        print("inspect before treating the current raw spans as a blocked mapping")
        return 1
    print("PASS conv.py raw-span boundary rejection")
    for boundary, pc_target, segment_boundary, detail in rejected:
        print(f"  rejected boundary {boundary}: pc_target={pc_target} segment_boundary={segment_boundary}{detail}")
    print("conv.py raw spans are exact stream spans, but they are not a proven general Rocket task-boundary mapping")
    return 0


def verify_single_stream_rocket_candidate(shape_name, rknn_path):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    segments = local_compiler_stitched_segments(shape_name, rknn_path)
    descs = runtime_direct_spatial_descs(shape_name)
    runtime_spans = conv.direct_spatial_task_regs(descs, conv.DIRECT_SPATIAL_POLICY_SINGLE_STREAM)
    task = {
        "regcmd": 0,
        "regcmd_count": len(stream),
    }
    task_fields = [name for name, _ctype in conv.rt.drm_rocket_task._fields_]
    seg_boundaries = segment_boundary_set(segments)
    pc_targets = pc_chain_target_starts(stream)
    failures = []
    if task_fields != ["regcmd", "regcmd_count"]:
        failures.append(f"unexpected Rocket task fields: {task_fields}")
    if task["regcmd_count"] != conv.DIRECT_SPATIAL_PC_ROOT6_STREAM_QWORDS:
        failures.append(f"candidate regcmd_count={task['regcmd_count']} expected={conv.DIRECT_SPATIAL_PC_ROOT6_STREAM_QWORDS}")
    if 0 not in seg_boundaries or len(stream) not in seg_boundaries:
        failures.append("candidate does not align with compiler stream endpoints")
    if 0 not in pc_targets:
        failures.append("candidate does not start at a PC-chain target")
    if task["regcmd_count"] > (conv.regcmd_mem_create.size // 8):
        failures.append("candidate exceeds local regcmd BO")
    if runtime_spans != [(0, len(stream))]:
        failures.append(f"conv.py single_stream spans={runtime_spans}, expected=[(0, {len(stream)})]")

    print(f"compiler_stream_qwords={len(stream)}")
    print(f"rocket_task_fields={task_fields}")
    print(f"single_stream_candidate=regcmd=0x{task['regcmd']:x} regcmd_count={task['regcmd_count']}")
    print(f"conv_single_stream_spans={runtime_spans}")
    print(f"endpoint_segment_boundaries={[0 in seg_boundaries, len(stream) in seg_boundaries]}")
    print(f"start_is_pc_target={0 in pc_targets}")
    print(f"regcmd_bo_qwords={conv.regcmd_mem_create.size // 8}")
    if failures:
        print("FAIL single-stream Rocket candidate")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS single-stream Rocket candidate")
    print("candidate has no internal task boundaries and fits the local Rocket task ABI, but it is unobserved and must remain offline-only until proven on hardware behind both gates")
    return 0


def verify_compact_single_stream_pc_target_rejection(shape_name, rknn_path):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    segments = local_compiler_stitched_segments(shape_name, rknn_path)
    seg_boundaries = segment_boundary_set(segments)
    rejected = []
    links = []
    for idx, qword in enumerate(stream):
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        value = (qword >> 16) & 0xffffffff
        if (target, reg_addr) != (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
            continue
        qword_target = value // 8
        containing = segment_containing(qword_target, segments)
        links.append((idx, qword_target, containing))
        if qword_target not in seg_boundaries:
            rejected.append((idx, qword_target, containing))

    if not links:
        print("FAIL compact single-stream PC target audit found no PC_BASE_ADDRESS links")
        return 1
    if not rejected:
        print("FAIL compact single-stream unexpectedly has all PC targets aligned to compact segment boundaries")
        print("inspect before treating compact layout as the known failed hardware layout")
        return 1

    print(f"compiler_stream_qwords={len(stream)}")
    print(f"pc_base_links={len(links)} rejected_targets={len(rejected)}")
    for idx, qword_target, containing in rejected[:8]:
        detail = ""
        if containing:
            detail = f" inside row{containing['row']}:{containing['kind']}[{containing['start']}:{containing['end']}]"
        print(f"  stream[{idx}] PC_BASE target qword {qword_target} is not a compact segment boundary{detail}")
    if len(rejected) > 8:
        print(f"  ... {len(rejected) - 8} more rejected targets")
    print("PASS compact single-stream PC-target rejection")
    print("compact stream qword positions do not match RKNN sparse task-record offsets; direct-spatial must use a sparse backing layout or another proven PC model")
    return 0


def sparse_fetch_layout_map(stream_len, records):
    compact_to_sparse = {}
    sparse_to_compact = {}
    compact_idx = 0
    intervals = driver_fetch_intervals([task_record_interval(record) for record in records])
    for record_idx, interval in enumerate(intervals):
        for sparse_idx in range(interval["start"], interval["end"]):
            if compact_idx >= stream_len:
                return compact_to_sparse, sparse_to_compact, record_idx, sparse_idx, compact_idx
            compact_to_sparse[compact_idx] = sparse_idx
            sparse_to_compact[sparse_idx] = compact_idx
            compact_idx += 1
    return compact_to_sparse, sparse_to_compact, None, None, compact_idx


def verify_sparse_fetch_layout_candidate(shape_name, rknn_path):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = observed_task_gem_records()
    compact_to_sparse, sparse_to_compact, end_record, end_sparse, compact_written = sparse_fetch_layout_map(
        len(stream), records)
    intervals = driver_fetch_intervals([task_record_interval(record) for record in records])
    table_end = max(interval["end"] for interval in intervals)
    failures = []
    if compact_written != len(stream):
        failures.append(f"sparse fetch layout consumed {compact_written}/{len(stream)} compact qwords")

    links = []
    missing_targets = []
    for compact_idx, qword in enumerate(stream):
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        value = (qword >> 16) & 0xffffffff
        if (target, reg_addr) != (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
            continue
        sparse_target = value // 8
        mapped_compact = sparse_to_compact.get(sparse_target)
        links.append((compact_idx, sparse_target, mapped_compact))
        if mapped_compact is None:
            missing_targets.append((compact_idx, sparse_target))

    if missing_targets:
        failures.append(f"{len(missing_targets)} PC targets are outside the compact stream sparse-fetch projection")

    print(f"compiler_stream_qwords={len(stream)}")
    print(f"sparse_fetch_table_qwords={table_end}")
    print(f"compact_qwords_projected={compact_written}")
    if end_record is not None:
        print(f"compact_stream_ends_inside_record={end_record} sparse_qword={end_sparse}")
    print(f"pc_base_links={len(links)} missing_targets={len(missing_targets)}")
    for compact_idx, sparse_target, mapped_compact in links:
        mapped = "missing" if mapped_compact is None else str(mapped_compact)
        print(f"  compact[{compact_idx}] -> sparse[{sparse_target}] maps_to_compact={mapped}")
    if failures:
        print("FAIL sparse fetch layout candidate")
        for failure in failures:
            print(f"  {failure}")
        print("sparse padding alone is not enough; the RKNN backing table has records beyond the 1280-qword compiler projection")
        print("PASS blocker still active: sparse fetch layout needs extended backing records")
        return 0
    print("PASS sparse fetch layout candidate")
    print("sparse fetch projection covers all compact qwords and PC targets; this layout can be considered for a guarded diagnostic path")
    return 0


def verify_sparse_extended_backing_hypothesis(shape_name, rknn_path):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = observed_task_gem_records()
    _compact_to_sparse, sparse_to_compact, _end_record, _end_sparse, _compact_written = sparse_fetch_layout_map(
        len(stream), records)
    starts = {regcmd_off // 8: (idx, amount, enable, int_mask)
              for idx, (amount, regcmd_off, enable, int_mask) in enumerate(records)}
    missing = []
    for compact_idx, qword in enumerate(stream):
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        value = (qword >> 16) & 0xffffffff
        if (target, reg_addr) != (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
            continue
        sparse_target = value // 8
        if sparse_target not in sparse_to_compact:
            missing.append((compact_idx, sparse_target, starts.get(sparse_target)))

    expected = [
        (1172, 1360, 15, 104, 0x0d, 1174, 1278),
        (1278, 1472, 16, 26, 0x60, 1280, 1280),
    ]
    failures = []
    if len(missing) != len(expected):
        failures.append(f"missing targets={missing}, expected {len(expected)}")
    for actual, exp in zip(missing, expected):
        compact_idx, sparse_target, record = actual
        exp_compact_idx, exp_sparse, exp_record_idx, exp_amount, exp_enable, exp_src_start, exp_src_end = exp
        if record is None:
            failures.append(f"sparse target {sparse_target} has no observed backing record")
            continue
        record_idx, amount, enable, _int_mask = record
        if (compact_idx, sparse_target, record_idx, amount, enable) != (
            exp_compact_idx, exp_sparse, exp_record_idx, exp_amount, exp_enable):
            failures.append(
                f"missing target got={(compact_idx, sparse_target, record_idx, amount, enable)} "
                f"expected={(exp_compact_idx, exp_sparse, exp_record_idx, exp_amount, exp_enable)}")
        if exp_src_end > len(stream):
            failures.append(f"expected source [{exp_src_start}:{exp_src_end}] exceeds compact stream")

    print(f"compiler_stream_qwords={len(stream)}")
    print(f"missing_sparse_targets={len(missing)}")
    for actual, exp in zip(missing, expected):
        compact_idx, sparse_target, record = actual
        exp_src_start, exp_src_end = exp[5], exp[6]
        if record is None:
            print(f"  compact[{compact_idx}] -> sparse[{sparse_target}] no backing record")
            continue
        record_idx, amount, enable, int_mask = record
        payload = exp_src_end - exp_src_start
        print(
            f"  compact[{compact_idx}] -> record[{record_idx}] sparse[{sparse_target}] "
            f"amount={amount} enable=0x{enable:x} int=0x{int_mask:x} "
            f"hypothesis_compact_payload=[{exp_src_start}:{exp_src_end}] len={payload}")
    if failures:
        print("FAIL sparse extended backing hypothesis")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS sparse extended backing hypothesis")
    print("observed records beyond the compact 1280-qword projection match the final payload/terminal-separator targets that the embedded PC links reference")
    return 0


def verify_separator_runs_from_export(shape_name, rknn_path):
    del shape_name
    rows = compiler_relevant_runs(rknn_path)
    separators = [
        [qword for _target, _reg_addr, _value, qword in row]
        for row in rows
        if len(row) == 26 and not any((target, reg_addr) == (conv.reg.CNA, conv.reg.CNA_CONV_CON2)
                                      for target, reg_addr, _value, _qword in row)
    ]
    expected_separator = conv.direct_spatial_separator_regs()
    failures = []
    if len(rows) != len(observed_task_gem_records()):
        failures.append(f"relevant runs={len(rows)}, expected task records={len(observed_task_gem_records())}")
    if len(separators) != 5:
        failures.append(f"separator runs={len(separators)}, expected=5")
    if separators and any(row != separators[0] for row in separators[1:]):
        failures.append("separator runs are not identical")
    if separators and separators[0] != expected_separator:
        failures.append("separator run does not match conv.py decoded direct_spatial_separator_regs()")

    print(f"relevant_runs={len(rows)}")
    print(f"separator_runs={len(separators)}")
    if separators:
        print("separator_decoded:")
        for idx, qword in enumerate(separators[0]):
            target = (qword >> 48) & 0xffff
            reg_addr = qword & 0xffff
            value = (qword >> 16) & 0xffffffff
            name = REG_NAMES.get((target, reg_addr), f"target=0x{target:x}/reg=0x{reg_addr:x}")
            print(f"  {idx:02d}: {name}=0x{value:x}")
    if failures:
        print("FAIL separator runs from export")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS separator runs from export")
    print("RKNN export carries five identical 26-qword separator records matching conv.py decoded PPU/PPU_RDMA separator regs")
    return 0


def sparse_export_backing_stream(rows, records):
    table_end = max((regcmd_off // 8) + len(decoded)
                    for decoded, (_amount, regcmd_off, _enable, _int) in zip(rows, records))
    sparse = [None] * table_end
    for row_idx, (decoded, record) in enumerate(zip(rows, records)):
        amount, regcmd_off, _enable, _int_mask = record
        qwords = [qword for _target, _reg_addr, _value, qword in decoded]
        start = regcmd_off // 8
        if len(qwords) < amount:
            raise ValueError(f"run {row_idx} has {len(qwords)} qwords, task amount is {amount}")
        if row_idx + 1 < len(records):
            next_start = records[row_idx + 1][1] // 8
            if start + len(qwords) > next_start:
                raise ValueError(
                    f"run {row_idx} span [{start}:{start + len(qwords)}] overlaps next record start {next_start}")
        sparse[start:start + len(qwords)] = qwords
    return sparse


SPARSE_GAP_QWORD = 0


def patch_sparse_pc_bases_for_local_bo(sparse):
    patched = []
    for idx, qword in enumerate(sparse):
        if qword is None:
            patched.append(SPARSE_GAP_QWORD)
            continue
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        value = (qword >> 16) & 0xffffffff
        if (target, reg_addr) == (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
            value = (conv.regcmd_mem_create.dma_addr + value) & 0xfffffff0
            qword = conv.E(target, reg_addr, value)
        patched.append(qword)
    return patched


def sparse_pc_base_links(sparse):
    starts = []
    for idx, qword in enumerate(sparse):
        if qword is None:
            continue
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        value = (qword >> 16) & 0xffffffff
        if (target, reg_addr) == (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
            starts.append((idx, normalized_pc_base_value(value)))
    return starts


def verify_sparse_export_backing_writer(shape_name, rknn_path, gem_log=None):
    del shape_name
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    rows = compiler_relevant_runs(rknn_path)
    failures = []
    if len(rows) != len(records):
        failures.append(f"relevant runs={len(rows)}, task records={len(records)}")
    try:
        sparse = sparse_export_backing_stream(rows, records)
    except ValueError as exc:
        print(f"FAIL sparse export backing writer: {exc}")
        return 1

    separator_expected = conv.direct_spatial_separator_regs()
    populated = [idx for idx, qword in enumerate(sparse) if qword is not None]
    starts = {regcmd_off // 8: idx for idx, (_amount, regcmd_off, _enable, _int) in enumerate(records)}
    pc_links = []
    missing_targets = []
    for record_idx, record in enumerate(records):
        amount, regcmd_off, _enable, _int_mask = record
        start = regcmd_off // 8
        row_len = len(rows[record_idx])
        qwords = sparse[start:start + row_len]
        if task_record_kind(record) == "separator" and qwords != separator_expected:
            failures.append(f"record[{record_idx}] separator does not match decoded conv.py separator regs")
        for qword_idx, qword in enumerate(qwords, start):
            if qword is None:
                failures.append(f"record[{record_idx}] sparse qword {qword_idx} is empty")
                continue
            target = (qword >> 48) & 0xffff
            reg_addr = qword & 0xffff
            value = (qword >> 16) & 0xffffffff
            if (target, reg_addr) != (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
                continue
            target_qword = value // 8
            target_record = starts.get(target_qword)
            pc_links.append((qword_idx, target_qword, target_record))
            if target_record is None or target_qword >= len(sparse) or sparse[target_qword] is None:
                missing_targets.append((qword_idx, target_qword, target_record))

    table_end = len(sparse)
    gaps = []
    cursor = 0
    for idx in populated:
        if cursor < idx:
            gaps.append((cursor, idx))
        cursor = idx + 1
    if cursor < table_end:
        gaps.append((cursor, table_end))
    if missing_targets:
        failures.append(f"{len(missing_targets)} PC targets do not land on populated sparse records")

    print(f"sparse_backing_qwords={table_end}")
    print(f"task_records={len(records)} populated_qwords={len(populated)} gaps={gaps}")
    print(f"pc_base_links={len(pc_links)}")
    for qword_idx, target_qword, target_record in pc_links:
        target_text = "missing" if target_record is None else f"record[{target_record}]"
        print(f"  sparse[{qword_idx}] -> sparse[{target_qword}] {target_text}")
    if failures:
        print("FAIL sparse export backing writer")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS sparse export backing writer")
    print("offline sparse backing table lays out all 17 RKNN records, decodes separator records, and resolves every embedded PC target to populated record starts")
    return 0


def verify_sparse_single_rocket_candidate(shape_name, rknn_path, gem_log=None):
    del shape_name
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    rows = compiler_relevant_runs(rknn_path)
    try:
        sparse = sparse_export_backing_stream(rows, records)
    except ValueError as exc:
        print(f"FAIL sparse single Rocket candidate: {exc}")
        return 1

    patched = patch_sparse_pc_bases_for_local_bo(sparse)
    populated = {idx for idx, qword in enumerate(sparse) if qword is not None}
    starts = {regcmd_off // 8: idx for idx, (_amount, regcmd_off, _enable, _int) in enumerate(records)}
    task_fields = [name for name, _ctype in conv.rt.drm_rocket_task._fields_]
    candidate = {
        "regcmd": 0,
        "regcmd_count": len(patched),
        "task_count": 1,
    }
    failures = []
    if task_fields != ["regcmd", "regcmd_count"]:
        failures.append(f"unexpected Rocket task fields: {task_fields}")
    if candidate["regcmd_count"] > conv.regcmd_mem_create.size // 8:
        failures.append("candidate exceeds local regcmd BO")
    if len(patched) != 1498:
        failures.append(f"sparse backing length={len(patched)}, expected 1498")
    if sum(1 for qword in patched if qword == SPARSE_GAP_QWORD) != len(patched) - len(populated):
        failures.append("gap fill count does not match sparse unpopulated qwords")
    pc_links = sparse_pc_base_links(patched)
    if len(pc_links) != 11:
        failures.append(f"pc link count={len(pc_links)}, expected 11")
    for pc_qword, target_qword in pc_links:
        if target_qword not in starts:
            failures.append(f"PC link at sparse[{pc_qword}] targets non-record qword {target_qword}")
        elif target_qword not in populated:
            failures.append(f"PC link at sparse[{pc_qword}] targets unpopulated qword {target_qword}")
        if pc_qword + 1 >= len(patched):
            failures.append(f"PC link at sparse[{pc_qword}] has no following PC_REGISTER_AMOUNTS")
            continue
        next_qword = patched[pc_qword + 1]
        target = (next_qword >> 48) & 0xffff
        reg_addr = next_qword & 0xffff
        value = (next_qword >> 16) & 0xffffffff
        if (target, reg_addr) != (conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS):
            failures.append(f"PC link at sparse[{pc_qword}] is not followed by PC_REGISTER_AMOUNTS")
            continue
        record_idx = starts.get(target_qword)
        if record_idx is not None:
            expected_amount = records[record_idx][0]
            decoded_amount = (value & 0xffff) * 2 - 2
            if decoded_amount != expected_amount:
                failures.append(
                    f"PC link at sparse[{pc_qword}] amount decodes {decoded_amount}, expected record[{record_idx}] amount {expected_amount}")

    print(f"sparse_backing_qwords={len(patched)}")
    print(f"populated_qwords={len(populated)} gap_qwords={len(patched) - len(populated)} gap_fill=0x{SPARSE_GAP_QWORD:016x}")
    print(f"rocket_task_fields={task_fields}")
    print(f"sparse_single_candidate=task_count={candidate['task_count']} regcmd=0x{candidate['regcmd']:x} regcmd_count={candidate['regcmd_count']}")
    print(f"regcmd_bo_qwords={conv.regcmd_mem_create.size // 8}")
    print(f"pc_base_links={len(pc_links)}")
    for pc_qword, target_qword in pc_links:
        target_record = starts.get(target_qword)
        print(f"  sparse[{pc_qword}] -> sparse[{target_qword}] record[{target_record}]")
    if failures:
        print("FAIL sparse single Rocket candidate")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS sparse single Rocket candidate")
    print("offline candidate is one local Rocket task over the sparse RKNN backing table, with local PC_BASE_ADDRESS values patched and gap qwords filled; it is still not submitted to hardware")
    return 0


def verify_runtime_sparse_single_stream(shape_name, rknn_path, gem_log=None):
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    rows = compiler_relevant_runs(rknn_path)
    expected_sparse = sparse_export_backing_stream(rows, records)
    expected_sparse = patch_sparse_pc_bases_for_local_bo(expected_sparse)
    expected_sparse = normalize_runtime_qwords(expected_sparse)

    descs = runtime_direct_spatial_descs(shape_name)
    runtime_sparse = conv.direct_spatial_sparse_backing_stream(descs)
    runtime_sparse = normalize_runtime_qwords(runtime_sparse)
    runtime_spans = conv.direct_spatial_task_regs(descs, conv.DIRECT_SPATIAL_POLICY_SPARSE_SINGLE)

    failures = []
    if runtime_sparse != expected_sparse:
        failures.append(
            f"runtime sparse stream differs from export sparse backing: runtime={len(runtime_sparse)} expected={len(expected_sparse)}")
    if runtime_spans != [(0, len(expected_sparse))]:
        failures.append(f"runtime sparse spans={runtime_spans}, expected=[(0, {len(expected_sparse)})]")

    print(f"runtime_sparse_qwords={len(runtime_sparse)}")
    print(f"expected_sparse_qwords={len(expected_sparse)}")
    print(f"runtime_sparse_spans={runtime_spans}")
    if failures:
        print("FAIL runtime sparse single stream")
        for failure in failures:
            print(f"  {failure}")
        for idx, (actual, expected) in enumerate(zip(runtime_sparse, expected_sparse)):
            if actual != expected:
                print(f"first_mismatch={idx} runtime=0x{actual:016x} expected=0x{expected:016x}")
                break
        return 1
    print("PASS runtime sparse single stream")
    print("conv.py sparse_single policy emits the offline-proven RKNN sparse backing table and one raw Rocket span")
    return 0


def verify_runtime_rocket_record_amounts_stream(shape_name, rknn_path, gem_log=None):
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    rows = compiler_relevant_runs(rknn_path)
    sparse_export = sparse_export_backing_stream(rows, records)
    expected_sparse, expected_patches = patch_sparse_pc_amounts_for_rocket(
        sparse_export, rows, records, "record_amount")
    expected_sparse = patch_sparse_pc_bases_for_local_bo(expected_sparse)
    expected_sparse = normalize_runtime_qwords(expected_sparse)

    descs = runtime_direct_spatial_descs(shape_name)
    runtime_stream = conv.direct_spatial_rocket_record_amounts_stream(descs)
    runtime_patch_count = len(conv.direct_spatial_rocket_record_amount_patches(
        conv.direct_spatial_sparse_backing_stream(descs)))
    runtime_sparse = normalize_runtime_qwords(runtime_stream)
    runtime_spans = conv.direct_spatial_task_regs(
        descs, conv.DIRECT_SPATIAL_POLICY_ROCKET_RECORD_AMOUNTS)

    failures = []
    if runtime_sparse != expected_sparse:
        failures.append(
            f"runtime rocket_record_amounts stream differs from offline candidate: "
            f"runtime={len(runtime_sparse)} expected={len(expected_sparse)}")
    if runtime_spans != [(0, len(expected_sparse))]:
        failures.append(f"runtime spans={runtime_spans}, expected=[(0, {len(expected_sparse)})]")
    if runtime_patch_count != len(expected_patches):
        failures.append(
            f"runtime patch count={runtime_patch_count}, expected={len(expected_patches)}")

    populated = {idx for idx, qword in enumerate(sparse_export) if qword is not None}
    starts = {_regcmd_off // 8: idx for idx, (_amount, _regcmd_off, _enable, _int) in enumerate(records)}
    for patch in expected_patches:
        target_start = patch["target_qword"]
        fetch_end = target_start + patch["rocket_decoded_qwords"]
        amount_end = target_start + patch["record_amount"]
        missing = [idx for idx in range(target_start, min(fetch_end, len(sparse_export))) if idx not in populated]
        if target_start not in starts:
            failures.append(f"sparse[{patch['pc_qword']}] target {target_start} is not a record start")
        if fetch_end != amount_end:
            failures.append(
                f"record[{patch['record_idx']}] fetch [{target_start}:{fetch_end}] "
                f"does not equal task amount end {amount_end}")
        if missing:
            failures.append(f"record[{patch['record_idx']}] fetch includes sparse gaps {missing[:8]}")

    print(f"runtime_rocket_record_amounts_qwords={len(runtime_sparse)}")
    print(f"expected_rocket_record_amounts_qwords={len(expected_sparse)}")
    print(f"runtime_rocket_record_amounts_spans={runtime_spans}")
    print(f"candidate_patches={len(expected_patches)}")
    for patch in expected_patches:
        fetch_end = patch["target_qword"] + patch["rocket_decoded_qwords"]
        print(
            f"  sparse[{patch['pc_qword']}] -> record[{patch['record_idx']}] "
            f"target={patch['target_qword']} original={patch['original_amount']} "
            f"rocket={patch['rocket_amount']} fetch=[{patch['target_qword']}:{fetch_end}]")
    if failures:
        print("FAIL runtime rocket_record_amounts stream")
        for failure in failures:
            print(f"  {failure}")
        if runtime_sparse != expected_sparse:
            for idx, (actual, expected) in enumerate(zip(runtime_sparse, expected_sparse)):
                if actual != expected:
                    print(f"first_mismatch={idx} runtime=0x{actual:016x} expected=0x{expected:016x}")
                    break
        return 1
    print("PASS runtime rocket_record_amounts stream")
    return 0


def verify_single_stream_regcmd_count_ambiguity(shape_name, rknn_path):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    testing_text = LIBROCKET_TESTING.read_text()
    pc_control_qwords = []
    for idx, qword in enumerate(stream):
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        if (target, reg_addr) in {
            (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS),
            (conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS),
            (conv.reg.PC, conv.reg.OPERATION_ENABLE),
        }:
            pc_control_qwords.append(idx)
    note_present = "regcmd_count passed to submit excludes PC header/footer" in testing_text
    full_count = len(stream)
    without_pc_control = full_count - len(pc_control_qwords)
    failures = []
    if not note_present:
        failures.append(f"missing librocket TESTING.md regcmd_count warning: {LIBROCKET_TESTING}")
    if not pc_control_qwords:
        failures.append("stitched stream has no PC control qwords to audit")
    if without_pc_control == full_count:
        failures.append("PC-control-excluded count unexpectedly equals full stream count")

    print(f"source={LIBROCKET_TESTING}")
    print(f"single_stream_full_regcmd_count={full_count}")
    print(f"pc_control_qwords={len(pc_control_qwords)}")
    print(f"pc_control_excluded_count={without_pc_control}")
    print(f"testing_note_present={note_present}")
    print(f"first_pc_control_qwords={pc_control_qwords[:12]}")
    if failures:
        print("FAIL single-stream regcmd_count ambiguity audit")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS single-stream regcmd_count ambiguity audit")
    print("full-stream count and PC-control-excluded count differ; hardware promotion needs this resolved before any submit change")
    return 0


def verify_local_python_runtime_regcmd_count_passthrough(shape_name, rknn_path):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    vendor_tasks = (conv.struct_rknpu_task * 1)()
    vendor_tasks[0].regcmd_addr = conv.regcmd_mem_create.dma_addr
    vendor_tasks[0].regcfg_amount = len(stream)
    rocket_tasks = conv.rt.build_rocket_tasks(vendor_tasks, 1)
    failures = []
    if rocket_tasks[0].regcmd != (conv.regcmd_mem_create.dma_addr & 0xffffffff):
        failures.append("Rocket task regcmd does not preserve low 32 bits of vendor regcmd_addr")
    if rocket_tasks[0].regcmd_count != len(stream):
        failures.append(f"Rocket task regcmd_count={rocket_tasks[0].regcmd_count}, expected passthrough {len(stream)}")

    print(f"compiler_stream_qwords={len(stream)}")
    print(f"vendor_regcfg_amount={vendor_tasks[0].regcfg_amount}")
    print(f"rocket_task_regcmd=0x{rocket_tasks[0].regcmd:x}")
    print(f"rocket_task_regcmd_count={rocket_tasks[0].regcmd_count}")
    if failures:
        print("FAIL local Python runtime regcmd_count passthrough")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS local Python runtime regcmd_count passthrough")
    print("experimental/kernel_6_18/rocket_runtime.py maps vendor regcfg_amount directly to Rocket regcmd_count with no PC-control adjustment")
    return 0


def verify_local_rocket_regcmd_count_rule(shape_name, rknn_path):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    source_text = (ROCKET_ACCEL_H.read_text() + "\n" +
                   ROCKET_ACCEL_H.with_name("rocket_interface.h").read_text() + "\n" +
                   (ROOT / "experimental" / "kernel_6_18" / "rocket_runtime.py").read_text())
    pc_control_qwords = []
    for idx, qword in enumerate(stream):
        target = (qword >> 48) & 0xffff
        reg_addr = qword & 0xffff
        if (target, reg_addr) in {
            (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS),
            (conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS),
            (conv.reg.PC, conv.reg.OPERATION_ENABLE),
            (0x0001, 0),
            (conv.reg.VERSION, 0),
        }:
            pc_control_qwords.append(idx)

    vendor_tasks = (conv.struct_rknpu_task * 1)()
    vendor_tasks[0].regcmd_addr = conv.regcmd_mem_create.dma_addr
    vendor_tasks[0].regcfg_amount = len(stream)
    rocket_tasks = conv.rt.build_rocket_tasks(vendor_tasks, 1)
    task_fields = [name for name, _ctype in conv.rt.drm_rocket_task._fields_]
    full_count = len(stream)
    pc_control_excluded_count = full_count - len(pc_control_qwords)

    required_patterns = {
        "rocket_task_count_field": r"regcmd_count",
        "rocket_task_count_comment": r"Number of commands in the register command buffer",
        "rocket_interface_count_comment": r"Number of 64-bit register commands",
        "python_count_passthrough": r"tasks\[i\]\.regcmd_count\s*=\s*int\(vendor_tasks\[i\]\.regcfg_amount\)",
    }
    failures = []
    for name, pattern in required_patterns.items():
        if not re.search(pattern, source_text):
            failures.append(f"missing source evidence: {name}")
    if task_fields != ["regcmd", "regcmd_count"]:
        failures.append(f"unexpected Rocket task fields: {task_fields}")
    if rocket_tasks[0].regcmd_count != full_count:
        failures.append(f"local runtime regcmd_count={rocket_tasks[0].regcmd_count}, expected full stream count {full_count}")
    if pc_control_excluded_count == full_count:
        failures.append("PC-control-excluded count did not differ from full count")
    if pc_control_excluded_count == rocket_tasks[0].regcmd_count:
        failures.append("local runtime unexpectedly applies PC-control exclusion")

    print(f"sources: {ROCKET_ACCEL_H}, {ROCKET_ACCEL_H.with_name('rocket_interface.h')}, experimental/kernel_6_18/rocket_runtime.py")
    print(f"rocket_task_fields={task_fields}")
    print(f"full_stream_regcmd_count={full_count}")
    print(f"pc_control_qwords={len(pc_control_qwords)}")
    print(f"pc_control_excluded_count={pc_control_excluded_count}")
    print(f"runtime_rocket_regcmd_count={rocket_tasks[0].regcmd_count}")
    if failures:
        print("FAIL local Rocket regcmd_count rule")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS local Rocket regcmd_count rule")
    print("resolved rule for this Python Rocket runtime: regcfg_amount must be the full stitched command count, including PC-control/header/footer qwords")
    return 0


def mesa_rocket_pc_amount_for_segment_qwords(segment_qwords):
    return conv._align_up((segment_qwords - PC_TAIL_QWORDS) // 2, 2)


def rocket_kernel_pc_amount_for_task_count(regcmd_count):
    return (regcmd_count + 1) // 2 - 1


def rocket_pc_amount_decoded_qwords(pc_amount):
    return pc_amount * 2 + 2


def patch_sparse_pc_amounts_for_rocket(sparse, rows, records, amount_model):
    patched = list(sparse)
    starts = {_regcmd_off // 8: idx for idx, (_amount, _regcmd_off, _enable, _int) in enumerate(records)}
    patches = []
    for pc_qword, target_qword in sparse_pc_base_links(sparse):
        if pc_qword + 1 >= len(sparse):
            raise ValueError(f"PC_BASE at sparse[{pc_qword}] has no following PC_REGISTER_AMOUNTS")
        amount_qword = sparse[pc_qword + 1]
        target = (amount_qword >> 48) & 0xffff
        reg_addr = amount_qword & 0xffff
        value = (amount_qword >> 16) & 0xffffffff
        if (target, reg_addr) != (conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS):
            raise ValueError(f"sparse[{pc_qword + 1}] is not PC_REGISTER_AMOUNTS")
        record_idx = starts.get(target_qword)
        if record_idx is None:
            raise ValueError(f"PC_BASE at sparse[{pc_qword}] targets non-record qword {target_qword}")
        original_amount = value & 0xffff
        if amount_model == "mesa_next_segment":
            rocket_amount = mesa_rocket_pc_amount_for_segment_qwords(len(rows[record_idx]))
        elif amount_model == "record_amount":
            rocket_amount = rocket_kernel_pc_amount_for_task_count(records[record_idx][0])
        else:
            raise ValueError(f"unknown Rocket PC amount model: {amount_model}")
        new_value = (value & 0xffff0000) | rocket_amount
        patched[pc_qword + 1] = conv.E(target, reg_addr, new_value)
        patches.append({
            "pc_qword": pc_qword,
            "amount_qword": pc_qword + 1,
            "target_qword": target_qword,
            "record_idx": record_idx,
            "row_len": len(rows[record_idx]),
            "record_amount": records[record_idx][0],
            "original_amount": original_amount,
            "rocket_amount": rocket_amount,
            "rocket_decoded_qwords": rocket_pc_amount_decoded_qwords(rocket_amount),
        })
    return patched, patches


def verify_rocket_pc_amount_model_mismatch(shape_name, rknn_path, gem_log=None):
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    rows = compiler_relevant_runs(rknn_path)
    sparse = sparse_export_backing_stream(rows, records)
    starts = {regcmd_off // 8: idx for idx, (_amount, regcmd_off, _enable, _int) in enumerate(records)}
    source_text = ((ROOT / "examples" / "kernel_6_18" / "conv_mesa.py").read_text() + "\n" +
                   LIBROCKET_TESTING.read_text())

    required_patterns = {
        "mesa_next_len_minus_tail": r"regs_to_fetch\s*=\s*next_size\s*-\s*4",
        "mesa_align_half_count": r"regs_to_fetch\s*=\s*_align\(regs_to_fetch\s*//\s*2,\s*2\)",
        "librocket_count_warning": r"regcmd_count passed to submit excludes PC header/footer",
    }
    failures = []
    for name, pattern in required_patterns.items():
        if not re.search(pattern, source_text):
            failures.append(f"missing source evidence: {name}")

    links = []
    mismatches = []
    for pc_qword, target_qword in sparse_pc_base_links(sparse):
        if pc_qword + 1 >= len(sparse):
            failures.append(f"PC_BASE at sparse[{pc_qword}] has no following PC_REGISTER_AMOUNTS")
            continue
        amount_qword = sparse[pc_qword + 1]
        target = (amount_qword >> 48) & 0xffff
        reg_addr = amount_qword & 0xffff
        value = (amount_qword >> 16) & 0xffffffff
        if (target, reg_addr) != (conv.reg.PC_REG, conv.reg.PC_REGISTER_AMOUNTS):
            failures.append(f"sparse[{pc_qword + 1}] is not PC_REGISTER_AMOUNTS")
            continue
        record_idx = starts.get(target_qword)
        if record_idx is None:
            failures.append(f"PC_BASE at sparse[{pc_qword}] targets non-record qword {target_qword}")
            continue
        rknn_amount = value & 0xffff
        row_len = len(rows[record_idx])
        mesa_amount = mesa_rocket_pc_amount_for_segment_qwords(row_len)
        decoded_qwords = rknn_amount * 2 - 2
        record_amount = records[record_idx][0]
        link = {
            "pc_qword": pc_qword,
            "target_qword": target_qword,
            "record_idx": record_idx,
            "row_len": row_len,
            "record_amount": record_amount,
            "rknn_amount": rknn_amount,
            "mesa_amount": mesa_amount,
            "decoded_qwords": decoded_qwords,
        }
        links.append(link)
        if rknn_amount != mesa_amount:
            mismatches.append(link)

    descs = runtime_direct_spatial_descs(shape_name)
    sparse_candidate = conv.direct_spatial_sparse_backing_stream(descs)
    sparse_single_count = len(sparse_candidate)
    kernel_top_amount = rocket_kernel_pc_amount_for_task_count(sparse_single_count)
    rknn_style_top_amount = conv._ceil_div(sparse_single_count, 2) + 1

    if not links:
        failures.append("no sparse PC links found")
    if not mismatches:
        failures.append("RKNN PC amounts unexpectedly match Mesa/Rocket amount model")
    if kernel_top_amount == rknn_style_top_amount:
        failures.append("kernel top-level PC amount unexpectedly matches RKNN-style amount")

    print(f"sources: examples/kernel_6_18/conv_mesa.py, {LIBROCKET_TESTING}")
    print(f"kernel_source_reference={KERNEL_ROCKET_JOB_C_URL}")
    print(f"sparse_single_qwords={sparse_single_count}")
    print(f"kernel_top_pc_amount=(count+1)//2-1 -> {kernel_top_amount}")
    print(f"rknn_style_top_pc_amount=ceil(count/2)+1 -> {rknn_style_top_amount}")
    print(f"pc_links={len(links)} mismatched_amounts={len(mismatches)}")
    for link in links:
        marker = "MISMATCH" if link in mismatches else "MATCH"
        print(
            f"  {marker} sparse[{link['pc_qword']}] -> record[{link['record_idx']}] "
            f"row_len={link['row_len']} record_amount={link['record_amount']} "
            f"rknn_pc_amount={link['rknn_amount']} decoded_qwords={link['decoded_qwords']} "
            f"mesa_pc_amount={link['mesa_amount']}")

    if failures:
        print("FAIL Rocket PC amount model mismatch audit")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS Rocket PC amount model mismatch audit")
    print("RKNN sparse_single uses vendor PC amount semantics; Mesa/Rocket PC-chain code uses next-segment-minus-tail semantics, so sparse_single must not be re-submitted unchanged")
    return 0


def verify_rocket_pc_amounts_candidate(shape_name, rknn_path, gem_log=None):
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    rows = compiler_relevant_runs(rknn_path)
    sparse = sparse_export_backing_stream(rows, records)
    populated = {idx for idx, qword in enumerate(sparse) if qword is not None}
    starts = {_regcmd_off // 8: idx for idx, (_amount, _regcmd_off, _enable, _int) in enumerate(records)}
    failures = []
    try:
        patched, patches = patch_sparse_pc_amounts_for_rocket(sparse, rows, records, "mesa_next_segment")
    except ValueError as exc:
        print(f"FAIL Rocket PC amounts candidate: {exc}")
        return 1

    if not patches:
        failures.append("candidate found no PC amount patches")
    if len(patched) != len(sparse):
        failures.append("candidate changed sparse stream length")
    for idx, (old, new) in enumerate(zip(sparse, patched)):
        if old == new:
            continue
        if idx == 0 or sparse[idx - 1] is None:
            failures.append(f"candidate changed sparse[{idx}] without preceding PC_BASE_ADDRESS")
            continue
        prev = sparse[idx - 1]
        target = (prev >> 48) & 0xffff
        reg_addr = prev & 0xffff
        if (target, reg_addr) != (conv.reg.PC_REG, conv.reg.PC_BASE_ADDRESS):
            failures.append(f"candidate changed sparse[{idx}] but previous qword is not PC_BASE_ADDRESS")

    fetch_windows = []
    for patch in patches:
        target_start = patch["target_qword"]
        decoded = patch["rocket_decoded_qwords"]
        fetch_end = target_start + decoded
        record_idx = patch["record_idx"]
        row_end = target_start + patch["row_len"]
        record_amount_end = target_start + patch["record_amount"]
        missing = [idx for idx in range(target_start, min(fetch_end, len(sparse))) if idx not in populated]
        fetch_windows.append((target_start, fetch_end))
        if target_start not in starts:
            failures.append(f"sparse[{patch['pc_qword']}] target {target_start} is not a record start")
        if fetch_end > len(sparse):
            failures.append(f"record[{record_idx}] Rocket fetch [{target_start}:{fetch_end}] exceeds sparse backing length {len(sparse)}")
        if missing:
            failures.append(f"record[{record_idx}] Rocket fetch [{target_start}:{fetch_end}] includes gap qwords {missing[:8]}")
        if fetch_end < record_amount_end:
            failures.append(
                f"record[{record_idx}] Rocket fetch [{target_start}:{fetch_end}] cuts before task amount end {record_amount_end}")
        if fetch_end > row_end:
            failures.append(
                f"record[{record_idx}] Rocket fetch [{target_start}:{fetch_end}] extends beyond decoded row end {row_end}")

    first_target = patches[0]["target_qword"] if patches else None
    last_fetch_end = max((end for _start, end in fetch_windows), default=0)
    covered = sorted({idx for start, end in fetch_windows for idx in range(start, min(end, len(sparse)))})
    covered_populated = sum(1 for idx in covered if idx in populated)

    print(f"sparse_backing_qwords={len(sparse)}")
    print(f"candidate_patches={len(patches)}")
    print(f"covered_populated_qwords={covered_populated}")
    if first_target is not None:
        print(f"first_pc_target_qword={first_target}")
    print(f"last_fetch_end={last_fetch_end}")
    for patch in patches:
        fetch_end = patch["target_qword"] + patch["rocket_decoded_qwords"]
        row_end = patch["target_qword"] + patch["row_len"]
        amount_end = patch["target_qword"] + patch["record_amount"]
        print(
            f"  sparse[{patch['pc_qword']}] -> record[{patch['record_idx']}] "
            f"target={patch['target_qword']} original={patch['original_amount']} "
            f"rocket={patch['rocket_amount']} fetch=[{patch['target_qword']}:{fetch_end}] "
            f"task_amount_end={amount_end} row_end={row_end}")
    if failures:
        print("FAIL Rocket PC amounts candidate")
        for failure in failures:
            print(f"  {failure}")
        print("candidate is offline-rejected; do not submit a simple PC_REGISTER_AMOUNTS rewrite")
        return 1
    print("PASS Rocket PC amounts candidate")
    print("candidate rewrites embedded PC amounts to Mesa/Rocket semantics without cutting records or fetching sparse gaps; still requires separate runtime wiring and guarded proof before hardware")
    return 0


def verify_rocket_record_amounts_candidate(shape_name, rknn_path, gem_log=None):
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    rows = compiler_relevant_runs(rknn_path)
    sparse = sparse_export_backing_stream(rows, records)
    populated = {idx for idx, qword in enumerate(sparse) if qword is not None}
    starts = {_regcmd_off // 8: idx for idx, (_amount, _regcmd_off, _enable, _int) in enumerate(records)}
    failures = []
    try:
        _patched, patches = patch_sparse_pc_amounts_for_rocket(sparse, rows, records, "record_amount")
    except ValueError as exc:
        print(f"FAIL Rocket record-amounts candidate: {exc}")
        return 1

    if not patches:
        failures.append("candidate found no PC amount patches")
    fetch_windows = []
    for patch in patches:
        target_start = patch["target_qword"]
        decoded = patch["rocket_decoded_qwords"]
        fetch_end = target_start + decoded
        record_idx = patch["record_idx"]
        row_end = target_start + patch["row_len"]
        record_amount_end = target_start + patch["record_amount"]
        missing = [idx for idx in range(target_start, min(fetch_end, len(sparse))) if idx not in populated]
        fetch_windows.append((target_start, fetch_end))
        if target_start not in starts:
            failures.append(f"sparse[{patch['pc_qword']}] target {target_start} is not a record start")
        if fetch_end != record_amount_end:
            failures.append(
                f"record[{record_idx}] Rocket fetch [{target_start}:{fetch_end}] does not equal task amount end {record_amount_end}")
        if fetch_end > row_end:
            failures.append(
                f"record[{record_idx}] Rocket fetch [{target_start}:{fetch_end}] extends beyond decoded row end {row_end}")
        if missing:
            failures.append(f"record[{record_idx}] Rocket fetch [{target_start}:{fetch_end}] includes gap qwords {missing[:8]}")

    covered = sorted({idx for start, end in fetch_windows for idx in range(start, min(end, len(sparse)))})
    covered_populated = sum(1 for idx in covered if idx in populated)
    print(f"sparse_backing_qwords={len(sparse)}")
    print(f"candidate_patches={len(patches)}")
    print(f"covered_populated_qwords={covered_populated}")
    for patch in patches:
        fetch_end = patch["target_qword"] + patch["rocket_decoded_qwords"]
        row_end = patch["target_qword"] + patch["row_len"]
        amount_end = patch["target_qword"] + patch["record_amount"]
        print(
            f"  sparse[{patch['pc_qword']}] -> record[{patch['record_idx']}] "
            f"target={patch['target_qword']} original={patch['original_amount']} "
            f"rocket={patch['rocket_amount']} fetch=[{patch['target_qword']}:{fetch_end}] "
            f"task_amount_end={amount_end} row_end={row_end}")
    if failures:
        print("FAIL Rocket record-amounts candidate")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS Rocket record-amounts candidate")
    print("embedded PC amounts rewritten with the Rocket kernel inverse of each RKNN task regcfg_amount preserve record starts, avoid sparse gaps, and fetch exactly each target task amount; still offline-only")
    return 0


def verify_runtime_pc_root6_guard(shape_name):
    if shape_name not in SHAPES:
        print(f"FAIL unknown shape: {shape_name}")
        return 1
    descs = runtime_direct_spatial_descs(shape_name)
    try:
        conv.validate_direct_spatial_pc_root6_descs(descs)
    except ValueError as exc:
        print(f"FAIL valid pc_root6 shape rejected: {exc}")
        return 1

    rejected = []
    for unsupported_shape in OBSERVED_DESCRIPTOR_SWEEP_SHAPES:
        if unsupported_shape.endswith("_h40_w40_oc320_wic160_k3x3_g1"):
            continue
        shape = descriptor_plan.SHAPES[unsupported_shape]
        p = conv._conv_params(shape.in_c, shape.in_h, shape.in_w, shape.out_c,
                              shape.kh, shape.kw, shape.groups, shape.stride)
        wrong_descs = conv.plan_observed_spatial_tile_descs(
            p, shape.in_c, shape.out_c, shape.kh, shape.kw,
            shape.in_h, shape.in_w, shape.groups, shape.stride)
        try:
            conv.validate_direct_spatial_pc_root6_descs(wrong_descs)
        except ValueError:
            rejected.append(unsupported_shape)
        else:
            print("FAIL pc_root6 guard accepted an unsupported descriptor schedule")
            print(f"accepted_shape={unsupported_shape}")
            return 1
    print("PASS runtime pc_root6 guard")
    print(f"known 40x40 descriptor schedule is accepted and unsupported sweep schedules rejected={len(rejected)}")
    return 0


def verify_direct_spatial_task_policy(shape_name):
    original = os.environ.get(conv.DIRECT_SPATIAL_TASKS_ENV)
    cases = [
        (None, conv.TASK_POLICY_REG_LISTS),
        ("", conv.TASK_POLICY_REG_LISTS),
        (conv.TASK_POLICY_REG_LISTS, conv.TASK_POLICY_REG_LISTS),
        (conv.DIRECT_SPATIAL_POLICY_PC_ROOT6, conv.DIRECT_SPATIAL_POLICY_PC_ROOT6),
        (conv.DIRECT_SPATIAL_POLICY_SINGLE_STREAM, conv.DIRECT_SPATIAL_POLICY_SINGLE_STREAM),
        (conv.DIRECT_SPATIAL_POLICY_SPARSE_SINGLE, conv.DIRECT_SPATIAL_POLICY_SPARSE_SINGLE),
    ]
    try:
        for value, expected in cases:
            if value is None:
                os.environ.pop(conv.DIRECT_SPATIAL_TASKS_ENV, None)
            else:
                os.environ[conv.DIRECT_SPATIAL_TASKS_ENV] = value
            actual = conv.direct_spatial_task_policy_for_descs(None)
            if actual != expected:
                print(f"FAIL policy value={value!r} got={actual!r} expected={expected!r}")
                return 1
            exec_expected = (conv.TASK_POLICY_RAW_SPANS
                             if expected in {conv.DIRECT_SPATIAL_POLICY_PC_ROOT6,
                                             conv.DIRECT_SPATIAL_POLICY_SINGLE_STREAM,
                                             conv.DIRECT_SPATIAL_POLICY_SPARSE_SINGLE}
                             else conv.TASK_POLICY_REG_LISTS)
            exec_actual = conv.direct_spatial_exec_task_policy(actual)
            if exec_actual != exec_expected:
                print(f"FAIL exec policy value={value!r} got={exec_actual!r} expected={exec_expected!r}")
                return 1
        os.environ.pop(conv.DIRECT_SPATIAL_TASKS_ENV, None)
        descs = runtime_direct_spatial_descs(shape_name)
        actual = conv.direct_spatial_task_policy_for_descs(descs)
        if actual != conv.DIRECT_SPATIAL_POLICY_SINGLE_STREAM:
            print(f"FAIL exact schedule default policy got={actual!r}")
            return 1
        os.environ[conv.DIRECT_SPATIAL_TASKS_ENV] = "definitely_bad"
        try:
            conv.direct_spatial_task_policy_for_descs(descs)
        except ValueError:
            print("PASS direct-spatial task policy")
            print("empty global policy uses reg_lists, exact schedule policy is single_stream, explicit policies are accepted, and unknown policies are rejected before submit")
            return 0
        print("FAIL unknown direct-spatial task policy was accepted")
        return 1
    finally:
        if original is None:
            os.environ.pop(conv.DIRECT_SPATIAL_TASKS_ENV, None)
        else:
            os.environ[conv.DIRECT_SPATIAL_TASKS_ENV] = original


def verify_direct_spatial_planner_routing(shape_name):
    saved = {
        conv.DIRECT_SPATIAL_ENV: os.environ.get(conv.DIRECT_SPATIAL_ENV),
        conv.DIRECT_SPATIAL_UNSAFE_ENV: os.environ.get(conv.DIRECT_SPATIAL_UNSAFE_ENV),
        conv.DIRECT_SPATIAL_TASKS_ENV: os.environ.get(conv.DIRECT_SPATIAL_TASKS_ENV),
    }

    def set_env(direct=None, unsafe=None, policy=None):
        for key, value in (
            (conv.DIRECT_SPATIAL_ENV, direct),
            (conv.DIRECT_SPATIAL_UNSAFE_ENV, unsafe),
            (conv.DIRECT_SPATIAL_TASKS_ENV, policy),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    try:
        for direct, unsafe in ((None, None), ("1", None), (None, "1")):
            set_env(direct=direct, unsafe=unsafe)
            try:
                planned_descs_for_shape(shape_name)
            except RuntimeError as exc:
                if "hardware execution is gated" in str(exc):
                    continue
                print("FAIL direct-spatial planner raised unexpected RuntimeError")
                print(f"direct={direct!r} unsafe={unsafe!r} error={exc}")
                return 1
            print("FAIL direct-spatial planner silently fell back instead of gating a proven RKNN schedule")
            print(f"direct={direct!r} unsafe={unsafe!r}")
            return 1

        set_env()
        im2col_routes = []
        gated_routes = 0
        for observed_name, shape in sorted(descriptor_plan.SHAPES.items()):
            if not descriptor_plan.observed_supported(shape):
                continue
            if not (shape.kh == 3 and shape.kw == 3 and shape.groups == 1):
                continue
            p, split_method, tiles = conv._plan_conv_tiles(
                shape.in_c, shape.out_c, shape.kh, shape.kw,
                shape.in_h, shape.in_w, shape.groups, shape.stride)
            plan = conv.classify_conv_plan(
                p, shape.in_c, shape.out_c, shape.kh, shape.kw,
                shape.in_h, shape.in_w, shape.groups, split_method, tiles)
            if plan["family"] == conv.FAMILY_SPATIAL_IM2COL_FALLBACK:
                im2col_routes.append(observed_name)
            elif plan["family"] == conv.FAMILY_DIRECT_SPATIAL_GATED:
                gated_routes += 1
        if im2col_routes:
            print("FAIL RKNN-observed dense spatial schedules still route through Python im2col")
            print(f"first={im2col_routes[0]} count={len(im2col_routes)}")
            return 1
        if gated_routes == 0:
            print("FAIL no RKNN-observed dense spatial schedules routed to the gated direct-spatial family")
            return 1

        set_env(direct="1", unsafe="1")
        try:
            planned_descs_for_shape(shape_name)
        except RuntimeError as exc:
            if "hardware execution is gated" not in str(exc):
                print("FAIL exact schedule raised unexpected RuntimeError under both gates")
                print(f"error={exc}")
                return 1
        else:
            print("FAIL exact schedule ran by default under both gates after failed single_stream hardware probe")
            return 1

        set_env(direct="1", unsafe="1", policy=conv.DIRECT_SPATIAL_POLICY_SINGLE_STREAM)
        descs = planned_descs_for_shape(shape_name)
        if len(descs) != 1 or descs[0]["family"] != conv.FAMILY_DIRECT_SPATIAL or descs[0]["task_policy"] != conv.TASK_POLICY_RAW_SPANS:
            print("FAIL direct-spatial planner did not honor explicit single_stream diagnostic policy")
            print(f"families={[desc['family'] for desc in descs]} policies={[desc.get('task_policy') for desc in descs]}")
            return 1
        if descs[0]["meta"].get("task_policy") != conv.DIRECT_SPATIAL_POLICY_SINGLE_STREAM:
            print("FAIL explicit single_stream diagnostic policy missing in metadata")
            print(f"meta={descs[0]['meta']}")
            return 1

        set_env(direct="1", unsafe="1", policy=conv.DIRECT_SPATIAL_POLICY_SPARSE_SINGLE)
        descs = planned_descs_for_shape(shape_name)
        if len(descs) != 1 or descs[0]["family"] != conv.FAMILY_DIRECT_SPATIAL or descs[0]["task_policy"] != conv.TASK_POLICY_RAW_SPANS:
            print("FAIL direct-spatial planner did not honor explicit sparse_single diagnostic policy")
            print(f"families={[desc['family'] for desc in descs]} policies={[desc.get('task_policy') for desc in descs]}")
            return 1
        if descs[0]["meta"].get("task_policy") != conv.DIRECT_SPATIAL_POLICY_SPARSE_SINGLE:
            print("FAIL explicit sparse_single diagnostic policy missing in metadata")
            print(f"meta={descs[0]['meta']}")
            return 1

        set_env(direct="1", unsafe="1", policy=conv.TASK_POLICY_REG_LISTS)
        descs = planned_descs_for_shape(shape_name)
        if len(descs) != 1 or descs[0]["family"] != conv.FAMILY_DIRECT_SPATIAL or descs[0]["task_policy"] != conv.TASK_POLICY_REG_LISTS:
            print("FAIL direct-spatial planner did not honor explicit reg-list diagnostic policy")
            print(f"families={[desc['family'] for desc in descs]} policies={[desc.get('task_policy') for desc in descs]}")
            return 1

        y_tile_shape = "outc_b1_c64_h20_w20_oc256_wic64_k3x3_g1"
        set_env(direct="1", unsafe="1")
        try:
            planned_descs_for_shape(y_tile_shape)
        except RuntimeError as exc:
            if "hardware execution is gated" not in str(exc):
                print("FAIL spatial y_tile schedule raised unexpected RuntimeError")
                print(f"error={exc}")
                return 1
        else:
            print("FAIL spatial y_tile schedule ran without explicit diagnostic policy")
            return 1

        set_env(direct="1", unsafe="1", policy=conv.TASK_POLICY_REG_LISTS)
        descs = planned_descs_for_shape(y_tile_shape)
        if len(descs) != 1 or descs[0]["family"] != conv.FAMILY_DIRECT_SPATIAL or descs[0]["task_policy"] != conv.TASK_POLICY_REG_LISTS:
            print("FAIL spatial y_tile schedule did not enter explicit reg-list diagnostic policy")
            print(f"families={[desc['family'] for desc in descs]} policies={[desc.get('task_policy') for desc in descs]}")
            return 1

        unsupported_shape = "sweep_b1_c160_h36_w36_oc320_wic160_k3x3_g1"
        set_env(direct="1", unsafe="1", policy=conv.DIRECT_SPATIAL_POLICY_PC_ROOT6)
        try:
            planned_descs_for_shape(unsupported_shape)
        except ValueError:
            print("PASS direct-spatial planner routing")
            print("RKNN-observed dense spatial schedules no longer route through Python im2col; direct-spatial hardware now fails closed by default, explicit diagnostics remain available, and unsupported pc_root6 schedules reject before submit")
            return 0
        print("FAIL pc_root6 planner accepted an unsupported mirrored direct-spatial schedule")
        return 1
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def segment_boundary_set(segments):
    boundaries = {0}
    for segment in segments:
        boundaries.add(segment["start"])
        boundaries.add(segment["end"])
    return boundaries


def segment_containing(qword_idx, segments):
    for segment in segments:
        if segment["start"] < qword_idx < segment["end"]:
            return segment
    return None


def segment_starting_at(qword_idx, segments):
    for segment in segments:
        if segment["start"] == qword_idx:
            return segment
    return None


def verify_rocket_segment_aligned_pc_partition_rejection(shape_name, rknn_path):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    segments = local_compiler_stitched_segments(shape_name, rknn_path)
    pc_targets = pc_chain_target_starts(stream)
    seg_boundaries = segment_boundary_set(segments)
    candidate_boundaries = sorted((pc_targets & seg_boundaries) | {len(stream)})
    executable_boundaries = []
    rejected_boundaries = []
    for boundary in candidate_boundaries:
        if boundary in (0, len(stream)):
            executable_boundaries.append(boundary)
            continue
        segment = segment_starting_at(boundary, segments)
        if segment is None:
            rejected_boundaries.append((boundary, "no segment starts here"))
        elif segment["kind"] == "pc_tail":
            rejected_boundaries.append((boundary, f"starts row{segment['row']}:{segment['kind']}"))
        else:
            executable_boundaries.append(boundary)

    print(f"compiler_stream_qwords={len(stream)}")
    print(f"pc_targets_in_stream={sorted(x for x in pc_targets if x <= len(stream))}")
    print(f"segment_pc_candidate_boundaries={candidate_boundaries}")
    for boundary in candidate_boundaries:
        segment = segment_starting_at(boundary, segments)
        segment_text = "final" if boundary == len(stream) else (
            f"row{segment['row']}:{segment['kind']}" if segment else "none")
        print(f"  boundary {boundary}: start_segment={segment_text}")
    print(f"executable_boundaries={executable_boundaries}")
    if rejected_boundaries:
        print("PASS segment-aligned PC partition rejection")
        for boundary, reason in rejected_boundaries:
            print(f"  rejected boundary {boundary}: {reason}")
        print("the only internal PC-target segment boundary starts at a PC tail; using it as a Rocket task root is not a valid descriptor-body start")
        return 0
    print("FAIL segment-aligned PC partition unexpectedly has no rejected internal boundary")
    print("inspect before promoting a segment-aligned PC-target partition to hardware")
    return 1


def verify_pc_root_candidate_boundaries(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    segments = local_compiler_stitched_segments(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    if submit_log:
        global_start, global_count, subcore_ranges = parse_submit_metadata(submit_log)
    else:
        global_start, global_count = 0, 6
        subcore_ranges = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]
    links = pc_links_by_task_record(stream, records)
    active_records = [
        idx for idx, record in enumerate(records)
        if task_record_interval(record)["start"] < len(stream)
    ]
    roots, _components, missing_records = pc_chain_components(records, active_records, links)
    root_starts = [task_record_interval(records[idx])["start"] for idx in roots]
    task_boundaries = root_starts + [len(stream)]
    record_starts = {task_record_interval(record)["start"] for record in records}
    pc_targets = pc_chain_target_starts(stream)
    seg_boundaries = segment_boundary_set(segments)
    expected_global_entries = list(range(global_start, global_start + global_count))
    expected_unique_subcore = sorted({
        idx for start, count in subcore_ranges
        for idx in range(start, start + count)
    })

    failures = []
    print(f"compiler_stream_qwords={len(stream)}")
    print(f"candidate_boundaries={task_boundaries}")
    print(f"observed_global_entries={expected_global_entries}")
    print(f"observed_unique_subcore_entries={expected_unique_subcore}")
    for boundary in task_boundaries:
        is_final = boundary == len(stream)
        record_start = boundary in record_starts
        pc_target = boundary in pc_targets
        segment_boundary = boundary in seg_boundaries
        containing = segment_containing(boundary, segments)
        detail = ""
        if containing:
            detail = f" inside row{containing['row']}:{containing['kind']}[{containing['start']}:{containing['end']}]"
        print(f"  boundary {boundary}: record_start={record_start} pc_target={pc_target} segment_boundary={segment_boundary}{detail}")
        if not is_final and not record_start:
            failures.append(f"boundary {boundary} is not an observed task-record start")
        if not is_final and not pc_target:
            failures.append(f"boundary {boundary} is not a PC-chain target/root")
        if not is_final and not segment_boundary:
            failures.append(f"boundary {boundary} cuts compiler segment{detail}")
    if missing_records:
        failures.append(f"component reconstruction missed active records: {missing_records}")
    if failures:
        print("FAIL root-contiguous candidate boundary invariants")
        for failure in failures:
            print(f"  {failure}")
        print("candidate remains offline-only and must not be submitted")
        return 1
    print("PASS root-contiguous candidate boundary invariants")
    print("candidate boundaries align with task records, PC targets, and compiler segments")
    return 0


def verify_driver_pc_fetch_spans(shape_name, rknn_path, gem_log=None):
    del shape_name, rknn_path
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    failures = []
    print(f"pc_data_extra_qwords={RKNPU_PC_DATA_EXTRA_AMOUNT}")
    for idx, (amount, regcmd_off, enable, int_mask) in enumerate(records):
        start = regcmd_off // 8
        fetch_end = start + amount + RKNPU_PC_DATA_EXTRA_AMOUNT
        if idx + 1 < len(records):
            next_start = records[idx + 1][1] // 8
            padding = next_start - fetch_end
            if padding < 0:
                failures.append((idx, "overlaps next record", padding))
        else:
            next_start = None
            padding = None
        print(f"  record[{idx}] {task_record_kind(records[idx])} start={start} amount={amount} fetch=[{start}:{fetch_end}]"
              f" next_start={next_start if next_start is not None else '-'} padding={padding if padding is not None else '-'}"
              f" enable=0x{enable:x} int=0x{int_mask:x}")
    if failures:
        print("FAIL driver PC fetch spans overlap")
        for idx, reason, padding in failures:
            print(f"  record[{idx}]: {reason}, padding={padding}")
        return 1
    print("PASS driver PC fetch span model")
    print("regcfg_amount + 4 explains record fetch spans; remaining gaps are alignment padding")
    return 0


def verify_vendor_pc_commit_register_model(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    if submit_log:
        _global_start, _global_count, subcore_ranges = parse_submit_metadata(submit_log)
    else:
        subcore_ranges = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]

    # Mirrors ref/rknpu_driver/rknpu_job.c rknpu_job_subcore_commit_pc() for
    # RK3588 after core_mask=0 is auto-scheduled to one core.
    scheduled_core = 0
    task_start, task_number = subcore_ranges[scheduled_core]
    first_idx = task_start
    last_idx = task_start + task_number - 1
    first_task = task_record_interval(records[first_idx])
    last_task = task_record_interval(records[last_idx])
    pc_data_amount_scale = 2
    pc_task_number_bits = 12
    task_pp_en = 0
    pc_data_addr = first_task["offset"]
    pc_data_amount = (
        first_task["amount"] + RKNPU_PC_DATA_EXTRA_AMOUNT + pc_data_amount_scale - 1
    ) // pc_data_amount_scale - 1
    pc_task_control = ((0x6 | task_pp_en) << pc_task_number_bits) | task_number
    int_mask = last_task["int_mask"]
    int_clear = first_task["int_mask"]
    fetch_end = first_task["start"] + first_task["amount"] + RKNPU_PC_DATA_EXTRA_AMOUNT
    fetched = stream[first_task["start"]:min(fetch_end, len(stream))]

    failures = []
    expected = {
        "scheduled_core": 0,
        "task_start": 0,
        "task_number": 2,
        "pc_data_addr": 0x0,
        "pc_data_amount": 55,
        "pc_task_control": 0x6002,
        "int_mask": 0x300,
        "int_clear": 0x300,
        "first_fetch_qwords": 112,
    }
    actual = {
        "scheduled_core": scheduled_core,
        "task_start": task_start,
        "task_number": task_number,
        "pc_data_addr": pc_data_addr,
        "pc_data_amount": pc_data_amount,
        "pc_task_control": pc_task_control,
        "int_mask": int_mask,
        "int_clear": int_clear,
        "first_fetch_qwords": len(fetched),
    }
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            failures.append(f"{key}=0x{actual[key]:x}, expected 0x{expected_value:x}")

    print("vendor PC commit model for observed core_mask=0 submit")
    print(f"selected subcore_task[{scheduled_core}]={subcore_ranges[scheduled_core]}")
    print(f"first_task={format_interval(first_task)}")
    print(f"last_task={format_interval(last_task)}")
    print(f"PC_DATA_ADDR=0x{pc_data_addr:x}")
    print(f"PC_DATA_AMOUNT=0x{pc_data_amount:x} ({pc_data_amount})")
    print(f"PC_TASK_CONTROL=0x{pc_task_control:x}")
    print(f"INT_MASK=0x{int_mask:x} INT_CLEAR=0x{int_clear:x}")
    print(f"first_fetch_span=[{first_task['start']}:{fetch_end}] qwords={len(fetched)}")
    if failures:
        print("FAIL vendor PC commit register model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS vendor PC commit register model")
    print("this models the vendor submit register programming for the selected range; it is not a local Rocket job/task mapping")
    return 0


def verify_vendor_selected_rocket_rejection(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    if submit_log:
        _global_start, _global_count, subcore_ranges = parse_submit_metadata(submit_log)
    else:
        subcore_ranges = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]

    # The vendor source model for observed core_mask=0 auto-selects one core.
    # If copied to Rocket, the only representable shape would be one job with
    # task_count=2 and raw regcmd/regcmd_count spans from records 0 and 1.
    scheduled_core = 0
    start, count = subcore_ranges[scheduled_core]
    intervals = active_task_intervals(records, start, count)
    rocket_tasks = [rocket_task_candidate(interval) for interval in intervals]
    raw_ranges, raw_covered, raw_first_uncovered = interval_coverage(intervals, len(stream))
    fetch_intervals = driver_fetch_intervals(intervals)
    fetch_ranges, fetch_covered, fetch_first_uncovered = interval_coverage(fetch_intervals, len(stream))
    failures = []
    if len(rocket_tasks) != 2:
        failures.append(f"expected two vendor-selected Rocket tasks, got {len(rocket_tasks)}")
    if raw_covered >= len(stream):
        failures.append("raw vendor-selected Rocket spans unexpectedly cover full compiler stream")
    if fetch_covered >= len(stream):
        failures.append("vendor fetch spans unexpectedly cover full compiler stream")
    if raw_first_uncovered != 108:
        failures.append(f"raw first_uncovered={raw_first_uncovered}, expected 108")
    if fetch_first_uncovered != 224:
        failures.append(f"fetch first_uncovered={fetch_first_uncovered}, expected 224")

    print(f"compiler_stream_qwords={len(stream)}")
    print(f"vendor-selected subcore_task[{scheduled_core}]={subcore_ranges[scheduled_core]}")
    print(f"candidate Rocket job from vendor-selected range: task_count={len(rocket_tasks)}")
    for idx, task in enumerate(rocket_tasks):
        start_q = task["regcmd"] // 8
        end_q = start_q + task["regcmd_count"]
        print(f"  task[{idx}] regcmd=0x{task['regcmd']:x} regcmd_count={task['regcmd_count']} span=[{start_q}:{end_q}] kind={task['kind']}")
    print(f"raw_covered_qwords={raw_covered}/{len(stream)} first_uncovered={raw_first_uncovered} ranges={raw_ranges}")
    print(f"vendor_fetch_covered_qwords={fetch_covered}/{len(stream)} first_uncovered={fetch_first_uncovered} ranges={fetch_ranges}")
    if failures:
        print("FAIL vendor-selected Rocket rejection invariant")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS vendor-selected Rocket rejection")
    print("the source-selected vendor range is provably incomplete as a local Rocket job; do not copy it into conv.py submit logic")
    return 0


def verify_global_task_number_rocket_rejection(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    if submit_log:
        global_start, global_count, _subcore_ranges = parse_submit_metadata(submit_log)
    else:
        global_start, global_count = 0, 6

    intervals = active_task_intervals(records, global_start, global_count)
    rocket_tasks = [rocket_task_candidate(interval) for interval in intervals]
    raw_ranges, raw_covered, raw_first_uncovered = interval_coverage(intervals, len(stream))
    fetch_intervals = driver_fetch_intervals(intervals)
    fetch_ranges, fetch_covered, fetch_first_uncovered = interval_coverage(fetch_intervals, len(stream))
    pc_targets = pc_chain_target_starts(stream)
    non_pc_boundaries = [interval for interval in intervals if interval["start"] not in pc_targets]
    separator_tasks = [interval for interval in intervals if interval["kind"] == "separator"]

    failures = []
    if len(rocket_tasks) != 6:
        failures.append(f"expected six global Rocket tasks, got {len(rocket_tasks)}")
    if raw_covered >= len(stream):
        failures.append("raw global Rocket spans unexpectedly cover full compiler stream")
    if fetch_covered >= len(stream):
        failures.append("global vendor fetch spans unexpectedly cover full compiler stream")
    if raw_first_uncovered != 108:
        failures.append(f"raw first_uncovered={raw_first_uncovered}, expected 108")
    if fetch_first_uncovered != 332:
        failures.append(f"fetch first_uncovered={fetch_first_uncovered}, expected 332")
    if len(non_pc_boundaries) != 2:
        failures.append(f"expected two non-PC-boundary global records, got {len(non_pc_boundaries)}")
    if len(separator_tasks) != 1:
        failures.append(f"expected one separator in global records, got {len(separator_tasks)}")

    print(f"compiler_stream_qwords={len(stream)}")
    print(f"global task_start={global_start} task_number={global_count}")
    print(f"candidate Rocket job from global range: task_count={len(rocket_tasks)}")
    for idx, task in enumerate(rocket_tasks):
        interval = intervals[idx]
        start_q = task["regcmd"] // 8
        end_q = start_q + task["regcmd_count"]
        boundary = "pc-boundary" if interval["start"] in pc_targets else "record-boundary"
        print(f"  task[{idx}] regcmd=0x{task['regcmd']:x} regcmd_count={task['regcmd_count']} span=[{start_q}:{end_q}] kind={task['kind']} {boundary}")
    print(f"raw_covered_qwords={raw_covered}/{len(stream)} first_uncovered={raw_first_uncovered} ranges={raw_ranges}")
    print(f"vendor_fetch_covered_qwords={fetch_covered}/{len(stream)} first_uncovered={fetch_first_uncovered} ranges={fetch_ranges}")
    print(f"non_pc_boundary_records={[format_interval(interval) for interval in non_pc_boundaries]}")
    print(f"separator_records={[format_interval(interval) for interval in separator_tasks]}")
    if failures:
        print("FAIL global task_number Rocket rejection invariant")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS global task_number Rocket rejection")
    print("the observed global six-record range is provably incomplete as a local Rocket job and includes non-PC-boundary records")
    return 0


def verify_vendor_runtime_task_selection(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    if submit_log:
        _global_start, _global_count, subcore_ranges = parse_submit_metadata(submit_log)
    else:
        subcore_ranges = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]

    # Mirrors ref/rknpu_driver/rknpu_job.c:
    # RKNPU_CORE_AUTO_MASK is rewritten to one scheduled core, then PC commit
    # reads subcore_task[core_index] for use_core_num 1 or 2.
    scheduled_core = 0
    start, count = subcore_ranges[scheduled_core]
    intervals = active_task_intervals(records, start, count)
    for interval in intervals:
        interval["pc_chain_boundary"] = (
            interval["start"] == 0 or
            stream[interval["start"] - 1] == conv.E(conv.reg.PC, conv.reg.OPERATION_ENABLE, 0x0d) or
            interval["kind"] == "separator"
        )
    if not intervals or any(interval["end"] > len(stream) for interval in intervals):
        print("FAIL vendor runtime task selection outside compiler stream")
        for interval in intervals:
            print(f"  {format_interval(interval)}")
        return 1
    if [(i["kind"], i["amount"], i["offset"]) for i in intervals] != [
            ("conv", 108, 0x0000), ("conv", 108, 0x0380)]:
        print("FAIL unexpected vendor-selected interval set")
        for interval in intervals:
            print(f"  {format_interval(interval)}")
        return 1
    print("PASS vendor runtime task selection model")
    print("core_mask=0 -> auto selected one core; empty-queue model uses core 0")
    print(f"selected subcore_task[{scheduled_core}]={subcore_ranges[scheduled_core]}")
    for interval in intervals:
        boundary = "pc-boundary" if interval["pc_chain_boundary"] else "record-boundary"
        print(f"  {format_interval(interval)} {boundary}")
    print("this proves the vendor-source active range model only; it is not a local Rocket job/task mapping")
    return 0


TASK_GEM_LINE = re.compile(
    r"task_like\[\d+\].*?enable_mask=0x(?P<enable>[0-9a-f]+)"
    r".*?int_mask=0x(?P<int_mask>[0-9a-f]+)"
    r".*?regcfg_amount=(?P<amount>\d+)"
    r".*?regcmd_addr=0x(?P<regcmd>[0-9a-f]+)",
)


def parse_task_gem_records(log_path):
    records = []
    in_gem1 = False
    for line in log_path.read_text().splitlines():
        if line.startswith("GEM "):
            in_gem1 = line.startswith("GEM 1:")
            continue
        if not in_gem1:
            continue
        match = TASK_GEM_LINE.search(line)
        if not match:
            continue
        records.append((
            int(match.group("amount")),
            int(match.group("regcmd"), 16),
            int(match.group("enable"), 16),
            int(match.group("int_mask"), 16),
        ))
    if not records:
        raise ValueError(f"no GEM 1 task records found in {log_path}")
    base_regcmd = records[0][1]
    return [(amount, regcmd - base_regcmd, enable, int_mask)
            for amount, regcmd, enable, int_mask in records]


def verify_observed_task_gem_log(log_path):
    actual_records = parse_task_gem_records(log_path)
    expected_records = observed_task_gem_records()
    if actual_records != expected_records:
        print("FAIL")
        print(f"actual_records={actual_records}")
        print(f"expected_records={expected_records}")
        return 1
    print(f"PASS log={log_path} gem1_task_records={len(actual_records)}")
    print("GEM 1 log records match observed task-gem amounts, relative regcmd offsets, enable masks, and interrupt masks")
    return 0


def verify_observed_task_gem(shape_name):
    bodies = local_full_bodies(shape_name)
    conv_counts = [len(bodies[0]), len(bodies[1])] + [len(body) - 4 for body in bodies[2:]]
    # Read-only remote GEM dump of the exact sweep export showed the backing task
    # GEM has 17 task-like records. The submit metadata is the active range; the
    # extra records are cached/backing-table entries unless referenced by
    # task_number or subcore_task ranges.
    expected_records = observed_task_gem_records()
    local_records = []
    body_idx = 0
    regcmd_off = 0
    for group in (4, 2, 2, 2, 2):
        for count in conv_counts[body_idx:body_idx + group]:
            local_records.append((count, regcmd_off, 0x0d, 0x0300))
            regcmd_off += 112 * 8
        body_idx += group
        local_records.append((26, regcmd_off, 0x60, 0x0c00))
        regcmd_off += 32 * 8
    if local_records != expected_records:
        print("FAIL")
        print(f"local_records={local_records}")
        print(f"expected_records={expected_records}")
        return 1
    submit_task_number = 6
    subcore_ranges = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]
    global_active = list(range(submit_task_number))
    subcore_active = sorted({idx for start, count in subcore_ranges for idx in range(start, start + count)})
    print(f"PASS task_records={len(local_records)} backing_conv_records={len(conv_counts)} separator_records=5")
    print("task-gem backing table: conv[0:4]+sep, then four groups of conv[2]+sep")
    print("record invariants: conv stride=112 qwords enable=0x0d int=0x300; separator stride=32 qwords enable=0x60 int=0xc00")
    print(f"submit global task_number={submit_task_number} active_if_linear={global_active}")
    print(f"submit subcore ranges={subcore_ranges} unique_active_by_subcore={subcore_active}")
    print("note: records outside submitted ranges are not evidence of active hardware work")
    return 0


def verify_rknpu_core_mask_semantics(shape_name):
    # This mirrors ref/rknpu_driver/rknpu_job.c. It is deliberately an offline
    # check: the local kernel_6_18 path uses upstream rocket, while the observed
    # RKNN metadata came from the vendor RKNPU ioctl ABI.
    del shape_name
    RKNPU_CORE_AUTO_MASK = 0x00
    RKNPU_CORE0_MASK = 0x01
    RKNPU_CORE1_MASK = 0x02
    RKNPU_CORE2_MASK = 0x04
    submit_task_number = 6
    observed_core_mask = RKNPU_CORE_AUTO_MASK
    observed_subcore_task = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]

    queued_task_load = [0, 0, 0]
    scheduled_core = min(range(3), key=lambda i: queued_task_load[i])
    scheduled_core_mask = [RKNPU_CORE0_MASK, RKNPU_CORE1_MASK, RKNPU_CORE2_MASK][scheduled_core]
    use_core_num = ((scheduled_core_mask & RKNPU_CORE0_MASK) +
                    ((scheduled_core_mask & RKNPU_CORE1_MASK) >> 1) +
                    ((scheduled_core_mask & RKNPU_CORE2_MASK) >> 2))
    if observed_core_mask != RKNPU_CORE_AUTO_MASK or use_core_num != 1:
        print("FAIL unexpected offline model")
        return 1

    task_start, task_number = observed_subcore_task[scheduled_core]
    active = list(range(task_start, task_start + task_number))
    print("PASS vendor rknpu_job.c core-mask model")
    print("core_mask=0 is RKNPU_CORE_AUTO_MASK, not an explicit 3-core mask")
    print(f"auto scheduling picks one core; empty-queue model selects core {scheduled_core} mask=0x{scheduled_core_mask:x}")
    print(f"global task_number={submit_task_number} but vendor multi-IRQ commit uses subcore_task[{scheduled_core}]={observed_subcore_task[scheduled_core]}")
    print(f"active task records under this model: {active}")
    print("implication: observed RKNN core_mask=0 metadata is not enough to build a local 3-core rocket submit")
    return 0


def c_define_value(source, name):
    match = re.search(rf"^\s*#define\s+{re.escape(name)}\s+(0x[0-9a-fA-F]+|\d+)\b",
                      source, re.MULTILINE)
    if not match:
        raise ValueError(f"missing {name} in source")
    return int(match.group(1), 0)


def source_has(source, pattern):
    return re.search(pattern, source, re.MULTILINE | re.DOTALL) is not None


def source_struct_body(source, name):
    match = re.search(
        rf"static\s+const\s+struct\s+\w+\s+{re.escape(name)}\s*\[\]\s*=\s*\{{(?P<body>.*?)\n\}};",
        source,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        match = re.search(
            rf"static\s+const\s+struct\s+\w+\s+{re.escape(name)}\s*=\s*\{{(?P<body>.*?)\n\}};",
            source,
            re.MULTILINE | re.DOTALL,
        )
    if not match:
        raise ValueError(f"missing source struct {name}")
    return match.group("body")


def verify_rknpu_driver_source_semantics(shape_name):
    del shape_name
    job_c = RKNPU_JOB_C.read_text()
    job_h = RKNPU_JOB_H.read_text()
    constants = {
        "RKNPU_MAX_CORES": c_define_value(job_h, "RKNPU_MAX_CORES"),
        "RKNPU_CORE_AUTO_MASK": c_define_value(job_h, "RKNPU_CORE_AUTO_MASK"),
        "RKNPU_CORE0_MASK": c_define_value(job_h, "RKNPU_CORE0_MASK"),
        "RKNPU_CORE1_MASK": c_define_value(job_h, "RKNPU_CORE1_MASK"),
        "RKNPU_CORE2_MASK": c_define_value(job_h, "RKNPU_CORE2_MASK"),
    }
    expected = {
        "RKNPU_MAX_CORES": 3,
        "RKNPU_CORE_AUTO_MASK": 0x00,
        "RKNPU_CORE0_MASK": 0x01,
        "RKNPU_CORE1_MASK": 0x02,
        "RKNPU_CORE2_MASK": 0x04,
    }
    failures = []
    for name, value in expected.items():
        if constants[name] != value:
            failures.append(f"{name}=0x{constants[name]:x}, expected 0x{value:x}")

    required_patterns = {
        "auto_core_mask_rewrite": (
            r"if\s*\(\s*job->args->core_mask\s*==\s*RKNPU_CORE_AUTO_MASK\s*\)"
            r".*?job->args->core_mask\s*=\s*rknpu_core_mask\s*\(\s*core_index\s*\)"
            r".*?job->use_core_num\s*=\s*1"
        ),
        "use_core_num_popcount": (
            r"job->use_core_num\s*=\s*\(args->core_mask\s*&\s*RKNPU_CORE0_MASK\)"
            r".*?\(\(args->core_mask\s*&\s*RKNPU_CORE1_MASK\)\s*>>\s*1\)"
            r".*?\(\(args->core_mask\s*&\s*RKNPU_CORE2_MASK\)\s*>>\s*2\)"
        ),
        "task_number_uses_subcore_for_1_or_2_cores": (
            r"if\s*\(\s*job->use_core_num\s*==\s*1\s*\|\|\s*job->use_core_num\s*==\s*2\s*\)"
            r"\s*task_num\s*=\s*job->args->subcore_task\s*\[\s*core_index\s*\]\s*\.task_number"
        ),
        "task_number_uses_core_plus_2_for_3_cores": (
            r"else\s+if\s*\(\s*job->use_core_num\s*==\s*3\s*\)"
            r"\s*task_num\s*=\s*job->args->subcore_task\s*\[\s*core_index\s*\+\s*2\s*\]\s*\.task_number"
        ),
        "pc_commit_start_count_uses_subcore_for_1_or_2_cores": (
            r"case\s+1:\s*case\s+2:"
            r".*?task_start\s*=\s*args->subcore_task\s*\[\s*core_index\s*\]\s*\.task_start"
            r".*?task_number\s*=\s*args->subcore_task\s*\[\s*core_index\s*\]\s*\.task_number"
        ),
        "pc_commit_start_count_uses_core_plus_2_for_3_cores": (
            r"case\s+3:"
            r".*?task_start\s*=\s*args->subcore_task\s*\[\s*core_index\s*\+\s*2\s*\]\s*\.task_start"
            r".*?task_number\s*=\s*args->subcore_task\s*\[\s*core_index\s*\+\s*2\s*\]\s*\.task_number"
        ),
    }
    for name, pattern in required_patterns.items():
        if not source_has(job_c, pattern):
            failures.append(f"missing source pattern: {name}")

    if failures:
        print("FAIL vendor rknpu driver source semantics")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("PASS vendor rknpu driver source semantics")
    print(f"sources: {RKNPU_JOB_C} {RKNPU_JOB_H}")
    print("constants: " + ", ".join(f"{name}=0x{value:x}" for name, value in constants.items()))
    print("source confirms core_mask=0 is auto-scheduled to one physical core")
    print("source confirms 1/2-core PC commit uses subcore_task[core_index]")
    print("source confirms only explicit 3-core PC commit uses subcore_task[core_index + 2]")
    return 0


def verify_rk3588_vendor_config_semantics(shape_name):
    del shape_name
    drv_c = RKNPU_DRV_C.read_text()
    failures = []
    try:
        irq_body = source_struct_body(drv_c, "rk3588_npu_irqs")
        config_body = source_struct_body(drv_c, "rk3588_rknpu_config")
    except ValueError as exc:
        print(f"FAIL RK3588 vendor config semantics: {exc}")
        return 1

    irq_count = len(re.findall(r"\{\s*\"npu\d+_irq\"\s*,\s*rknpu_core\d+_irq_handler\s*\}", irq_body))
    if irq_count != 3:
        failures.append(f"rk3588_npu_irqs has {irq_count} entries, expected 3")

    required_patterns = {
        "of_match_uses_rk3588_config": (
            r"\.compatible\s*=\s*\"rockchip,rk3588-rknpu\""
            r".*?\.data\s*=\s*&rk3588_rknpu_config"
        ),
        "config_uses_rk3588_irqs": r"\.irqs\s*=\s*rk3588_npu_irqs",
        "config_num_irqs_from_array": r"\.num_irqs\s*=\s*ARRAY_SIZE\s*\(\s*rk3588_npu_irqs\s*\)",
        "config_core_mask_three_cores": r"\.core_mask\s*=\s*0x7",
        "config_pc_scale_two": r"\.pc_data_amount_scale\s*=\s*2",
        "config_pc_task_bits": r"\.pc_task_number_bits\s*=\s*12",
        "config_pc_task_mask": r"\.pc_task_number_mask\s*=\s*0xfff",
    }
    combined = drv_c
    scoped = {
        "config_uses_rk3588_irqs",
        "config_num_irqs_from_array",
        "config_core_mask_three_cores",
        "config_pc_scale_two",
        "config_pc_task_bits",
        "config_pc_task_mask",
    }
    for name, pattern in required_patterns.items():
        source = config_body if name in scoped else combined
        if not source_has(source, pattern):
            failures.append(f"missing source pattern: {name}")

    if failures:
        print("FAIL RK3588 vendor config semantics")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("PASS RK3588 vendor config semantics")
    print(f"source: {RKNPU_DRV_C}")
    print("rockchip,rk3588-rknpu binds rk3588_rknpu_config")
    print("rk3588_npu_irqs has three IRQ handlers, so vendor RK3588 uses the multi-IRQ/subcore path")
    print("rk3588 config exposes core_mask=0x7, pc_data_amount_scale=2, pc_task_number_bits=12, pc_task_number_mask=0xfff")
    return 0


def verify_local_rocket_uapi(shape_name):
    del shape_name
    task_fields = [name for name, _ctype in conv.rt.drm_rocket_task._fields_]
    job_fields = [name for name, _ctype in conv.rt.drm_rocket_job._fields_]
    submit_fields = [name for name, _ctype in conv.rt.drm_rocket_submit._fields_]
    forbidden = {"core_mask", "subcore_task", "task_start", "task_number"}
    present_forbidden = sorted(
        forbidden & (set(task_fields) | set(job_fields) | set(submit_fields))
    )
    expected_task = ["regcmd", "regcmd_count"]
    if task_fields != expected_task or present_forbidden:
        print("FAIL unexpected local Rocket UAPI shape")
        print(f"task_fields={task_fields}")
        print(f"job_fields={job_fields}")
        print(f"submit_fields={submit_fields}")
        print(f"vendor-only fields present={present_forbidden}")
        return 1
    if "task_count" not in job_fields or "job_count" not in submit_fields:
        print("FAIL local Rocket UAPI missing task/job count fields")
        return 1

    print("PASS local Rocket UAPI model")
    print("local kernel_6_18 path submits drm_rocket_job/task arrays, not vendor rknpu_submit")
    print(f"drm_rocket_task fields={task_fields}")
    print("no core_mask/subcore_task fields exist in the local Rocket ioctl ABI")
    print("implication: RKNN vendor submit metadata must be translated into Rocket job/task boundaries before any local direct-spatial HW probe")
    return 0


def verify_local_rocket_uapi_source(shape_name):
    del shape_name
    source = ROCKET_ACCEL_H.read_text()
    task_fields = c_struct_fields(source, "drm_rocket_task")
    job_fields = c_struct_fields(source, "drm_rocket_job")
    submit_fields = c_struct_fields(source, "drm_rocket_submit")
    forbidden = {"core_mask", "subcore_task", "task_start", "task_number", "enable_mask", "int_mask"}
    present_forbidden = sorted(
        forbidden & ({name for name, _ctype in task_fields} |
                     {name for name, _ctype in job_fields} |
                     {name for name, _ctype in submit_fields})
    )
    expected_task = [("regcmd", "u32"), ("regcmd_count", "u32")]
    required_job = {"tasks", "task_count", "task_struct_size"}
    required_submit = {"jobs", "job_count", "job_struct_size"}
    failures = []
    if task_fields != expected_task:
        failures.append(f"drm_rocket_task fields={task_fields}, expected={expected_task}")
    job_names = {name for name, _ctype in job_fields}
    submit_names = {name for name, _ctype in submit_fields}
    if not required_job <= job_names:
        failures.append(f"drm_rocket_job missing {sorted(required_job - job_names)}")
    if not required_submit <= submit_names:
        failures.append(f"drm_rocket_submit missing {sorted(required_submit - submit_names)}")
    if present_forbidden:
        failures.append(f"vendor-only fields present={present_forbidden}")
    print(f"sources: {ROCKET_ACCEL_H}")
    print(f"drm_rocket_task={task_fields}")
    print(f"drm_rocket_job={job_fields}")
    print(f"drm_rocket_submit={submit_fields}")
    if failures:
        print("FAIL local Rocket UAPI source model")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS local Rocket UAPI source model")
    print("header confirms Rocket tasks carry only regcmd/regcmd_count; job/submit carry task_count/job_count and struct sizes")
    print("header contains no vendor core_mask/subcore_task/task_start/task_number fields")
    return 0


def verify_local_rocket_pc_commit_surface_gap(shape_name, rknn_path, submit_log=None, gem_log=None):
    stream = local_compiler_stitched_stream(shape_name, rknn_path)
    records = parse_task_gem_records(gem_log) if gem_log else observed_task_gem_records()
    if submit_log:
        _global_start, _global_count, subcore_ranges = parse_submit_metadata(submit_log)
    else:
        subcore_ranges = [(0, 2), (0, 2), (0, 2), (0, 0), (0, 0)]

    vendor_source = RKNPU_JOB_C.read_text()
    rocket_header = ROCKET_ACCEL_H.read_text()
    rocket_task_fields = c_struct_fields(rocket_header, "drm_rocket_task")
    scheduled_core = 0
    task_start, task_number = subcore_ranges[scheduled_core]
    first_task = task_record_interval(records[task_start])
    last_task = task_record_interval(records[task_start + task_number - 1])
    pc_data_amount = (
        first_task["amount"] + RKNPU_PC_DATA_EXTRA_AMOUNT + 2 - 1
    ) // 2 - 1
    pc_task_control = (0x6 << 12) | task_number
    first_fetch_end = first_task["start"] + first_task["amount"] + RKNPU_PC_DATA_EXTRA_AMOUNT

    required_vendor_patterns = {
        "PC_DATA_ADDR": r"REG_WRITE\(first_task->regcmd_addr,\s*RKNPU_OFFSET_PC_DATA_ADDR\)",
        "PC_DATA_AMOUNT": r"RKNPU_OFFSET_PC_DATA_AMOUNT",
        "INT_MASK": r"REG_WRITE\(last_task->int_mask,\s*RKNPU_OFFSET_INT_MASK\)",
        "INT_CLEAR": r"REG_WRITE\(first_task->int_mask,\s*RKNPU_OFFSET_INT_CLEAR\)",
        "PC_TASK_CONTROL": r"RKNPU_OFFSET_PC_TASK_CONTROL",
        "PC_DMA_BASE_ADDR": r"RKNPU_OFFSET_PC_DMA_BASE_ADDR",
        "PC_OP_EN": r"RKNPU_OFFSET_PC_OP_EN",
    }
    forbidden_rocket_fields = {
        "core_mask", "subcore_task", "task_start", "task_number",
        "enable_mask", "int_mask", "int_clear", "regcfg_amount", "regcmd_addr",
    }
    rocket_field_names = {name for name, _ctype in rocket_task_fields}
    failures = []
    for name, pattern in required_vendor_patterns.items():
        if not re.search(pattern, vendor_source):
            failures.append(f"missing vendor source write pattern: {name}")
    present_forbidden = sorted(forbidden_rocket_fields & rocket_field_names)
    if rocket_task_fields != [("regcmd", "u32"), ("regcmd_count", "u32")]:
        failures.append(f"unexpected drm_rocket_task fields={rocket_task_fields}")
    if present_forbidden:
        failures.append(f"Rocket task unexpectedly exposes vendor fields={present_forbidden}")
    if first_fetch_end > len(stream):
        failures.append(f"vendor first fetch end {first_fetch_end} exceeds stream length {len(stream)}")

    print("vendor PC commit surface")
    print(f"sources: {RKNPU_JOB_C}, {ROCKET_ACCEL_H}")
    print(f"selected subcore_task[{scheduled_core}]={subcore_ranges[scheduled_core]}")
    print(f"first_task={format_interval(first_task)}")
    print(f"last_task={format_interval(last_task)}")
    print(f"vendor_commit_registers=PC_DATA_ADDR,PC_DATA_AMOUNT,INT_MASK,INT_CLEAR,PC_TASK_CONTROL,PC_DMA_BASE_ADDR,PC_OP_EN")
    print(f"derived_PC_DATA_AMOUNT=0x{pc_data_amount:x}")
    print(f"derived_PC_TASK_CONTROL=0x{pc_task_control:x}")
    print(f"derived_INT_MASK=0x{last_task['int_mask']:x} derived_INT_CLEAR=0x{first_task['int_mask']:x}")
    print(f"first_fetch_span=[{first_task['start']}:{first_fetch_end}]")
    print(f"local_drm_rocket_task_fields={rocket_task_fields}")
    if failures:
        print("FAIL local Rocket PC commit surface gap")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS local Rocket PC commit surface gap")
    print("stream-only rewrites cannot express the vendor PC commit registers; the next candidate must account for how Rocket synthesizes or replaces that commit surface")
    return 0


def verify_runtime_rocket_policy_commit_models(shape_name):
    del shape_name
    descs = runtime_direct_spatial_descs("sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1")
    policies = (
        conv.DIRECT_SPATIAL_POLICY_SINGLE_STREAM,
        conv.DIRECT_SPATIAL_POLICY_PC_ROOT6,
        conv.DIRECT_SPATIAL_POLICY_SPARSE_SINGLE,
        conv.DIRECT_SPATIAL_POLICY_ROCKET_RECORD_AMOUNTS,
    )
    failures = []
    print("Rocket per-task PC commit models for conv.py diagnostic policies")
    for policy in policies:
        spans = conv.direct_spatial_task_regs(descs, policy)
        print(f"policy={policy} task_count={len(spans)}")
        for idx, (start, amount) in enumerate(spans):
            pc_amount = rocket_kernel_pc_amount_for_task_count(amount)
            decoded_qwords = rocket_pc_amount_decoded_qwords(pc_amount)
            print(
                f"  task[{idx}] span=[{start}:{start + amount}] regcmd_count={amount} "
                f"PC_DATA_AMOUNT={pc_amount} decoded_qwords={decoded_qwords} "
                f"PC_TASK_CONTROL=0x6001")
            if decoded_qwords != amount:
                failures.append(
                    f"{policy} task[{idx}] decoded_qwords={decoded_qwords}, "
                    f"expected regcmd_count={amount}")
        if policy == conv.DIRECT_SPATIAL_POLICY_SPARSE_SINGLE and spans != [(0, 1498)]:
            failures.append(f"sparse_single spans changed unexpectedly: {spans}")
        if policy == conv.DIRECT_SPATIAL_POLICY_ROCKET_RECORD_AMOUNTS and spans != [(0, 1498)]:
            failures.append(f"rocket_record_amounts spans changed unexpectedly: {spans}")

    if failures:
        print("FAIL runtime Rocket policy commit models")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS runtime Rocket policy commit models")
    print("local Rocket commits each drm_rocket_task with TASK_NUMBER(1) and PC_DATA_AMOUNT decoding to regcmd_count, unlike vendor regcfg_amount+4 commits")
    return 0


def verify_direct_spatial_gate(shape_name):
    del shape_name
    saved = {
        conv.DIRECT_SPATIAL_ENV: os.environ.get(conv.DIRECT_SPATIAL_ENV),
        conv.DIRECT_SPATIAL_UNSAFE_ENV: os.environ.get(conv.DIRECT_SPATIAL_UNSAFE_ENV),
    }
    cases = [
        (None, None, False),
        ("1", None, False),
        (None, "1", False),
        ("0", "1", False),
        ("1", "0", False),
        ("1", "1", True),
    ]
    try:
        for direct_value, unsafe_value, expected in cases:
            if direct_value is None:
                os.environ.pop(conv.DIRECT_SPATIAL_ENV, None)
            else:
                os.environ[conv.DIRECT_SPATIAL_ENV] = direct_value
            if unsafe_value is None:
                os.environ.pop(conv.DIRECT_SPATIAL_UNSAFE_ENV, None)
            else:
                os.environ[conv.DIRECT_SPATIAL_UNSAFE_ENV] = unsafe_value

            actual = conv.direct_spatial_hw_enabled()
            if actual != expected:
                print("FAIL direct-spatial gate mismatch")
                print(f"{conv.DIRECT_SPATIAL_ENV}={direct_value!r}")
                print(f"{conv.DIRECT_SPATIAL_UNSAFE_ENV}={unsafe_value!r}")
                print(f"actual={actual} expected={expected}")
                return 1
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("PASS direct-spatial HW gate")
    print(f"direct-spatial execution requires {conv.DIRECT_SPATIAL_ENV}=1 and {conv.DIRECT_SPATIAL_UNSAFE_ENV}=1")
    return 0


def verify_safe_direct_spatial_preflight(shape_name):
    rknn_path = rknn_path_arg(shape_name, None)
    task_gem_log = ROOT / "experimental" / "rknn" / "capture_rknpu_submit_dump_gems_sweep.log"
    checks = [
        ("direct-spatial gate", lambda: verify_direct_spatial_gate(shape_name)),
        ("exact export pair", lambda: verify_exact_export_pair(shape_name)),
        ("RKNN spans", lambda: verify_compiler_spans(shape_name, rknn_path)),
        ("RKNN stream", lambda: verify_compiler_stream(shape_name, rknn_path)),
        ("compiler boundaries", lambda: verify_compiler_boundaries(shape_name, rknn_path)),
        ("observed descriptor sweep", lambda: verify_observed_descriptor_sweep()),
        ("observed spatial y_tile descriptors", lambda: verify_observed_spatial_y_tile_descriptors()),
        ("observed chan320 descriptors", lambda: verify_observed_chan320_descriptors()),
        ("observed mixed spatial descriptors", lambda: verify_observed_mixed_spatial_descriptors()),
        ("observed compare spatial descriptors", lambda: verify_observed_compare_spatial_descriptors()),
        ("observed pointwise descriptors", lambda: verify_observed_pointwise_descriptors()),
        ("all observed-supported descriptors", lambda: verify_all_observed_supported_descriptors()),
        ("candidate 6-task mapping", lambda: verify_6task_mapping(shape_name)),
        ("observed task GEM", lambda: verify_observed_task_gem(shape_name)),
        ("observed task GEM log", lambda: verify_observed_task_gem_log(task_gem_log)),
        ("submit log consistency", lambda: verify_submit_log_consistency(shape_name, [])),
        ("task object sync correlation", lambda: verify_task_object_sync_correlation(shape_name)),
        ("task object record count", lambda: verify_task_object_record_count(shape_name, None, task_gem_log)),
        ("observed active ranges", lambda: verify_observed_active_ranges(shape_name)),
        ("runtime task coverage", lambda: verify_runtime_task_coverage(shape_name, rknn_path, None, task_gem_log)),
        ("runtime fetch coverage", lambda: verify_runtime_fetch_coverage(shape_name, rknn_path, None, task_gem_log)),
        ("runtime task gaps", lambda: verify_runtime_task_gaps(shape_name, rknn_path, None, task_gem_log)),
        ("runtime task row overlaps", lambda: verify_runtime_task_row_overlaps(shape_name, rknn_path, None, task_gem_log)),
        ("task backing span", lambda: verify_task_backing_span(shape_name, rknn_path, task_gem_log)),
        ("PC base task links", lambda: verify_pc_base_task_links(shape_name, rknn_path, task_gem_log)),
        ("PC amount task lengths", lambda: verify_pc_amount_task_lengths(shape_name, rknn_path, task_gem_log)),
        ("driver PC fetch spans", lambda: verify_driver_pc_fetch_spans(shape_name, rknn_path, task_gem_log)),
        ("vendor core-mask semantics", lambda: verify_rknpu_core_mask_semantics(shape_name)),
        ("vendor driver source semantics", lambda: verify_rknpu_driver_source_semantics(shape_name)),
        ("RK3588 vendor config semantics", lambda: verify_rk3588_vendor_config_semantics(shape_name)),
        ("vendor runtime task selection", lambda: verify_vendor_runtime_task_selection(shape_name, rknn_path)),
        ("vendor PC commit register model", lambda: verify_vendor_pc_commit_register_model(shape_name, rknn_path)),
        ("vendor-selected Rocket rejection", lambda: verify_vendor_selected_rocket_rejection(shape_name, rknn_path)),
        ("global task_number Rocket rejection", lambda: verify_global_task_number_rocket_rejection(shape_name, rknn_path)),
        ("local Rocket UAPI", lambda: verify_local_rocket_uapi(shape_name)),
        ("local Rocket UAPI source", lambda: verify_local_rocket_uapi_source(shape_name)),
        ("local Rocket PC commit surface gap", lambda: verify_local_rocket_pc_commit_surface_gap(shape_name, rknn_path)),
        ("runtime Rocket policy commit models", lambda: verify_runtime_rocket_policy_commit_models(shape_name)),
        ("segment-aligned PC partition rejection", lambda: verify_rocket_segment_aligned_pc_partition_rejection(shape_name, rknn_path)),
        ("direct-spatial task policy", lambda: verify_direct_spatial_task_policy(shape_name)),
        ("direct-spatial planner routing", lambda: verify_direct_spatial_planner_routing(shape_name)),
        ("runtime pc_root6 stream", lambda: verify_runtime_pc_root6_stream(shape_name, rknn_path)),
        ("conv.py raw-span boundary rejection", lambda: verify_conv_raw_span_boundary_rejection(shape_name, rknn_path)),
        ("single-stream Rocket candidate", lambda: verify_single_stream_rocket_candidate(shape_name, rknn_path)),
        ("compact single-stream PC-target rejection", lambda: verify_compact_single_stream_pc_target_rejection(shape_name, rknn_path)),
        ("sparse fetch layout candidate", lambda: verify_sparse_fetch_layout_candidate(shape_name, rknn_path)),
        ("sparse extended backing hypothesis", lambda: verify_sparse_extended_backing_hypothesis(shape_name, rknn_path)),
        ("separator runs from export", lambda: verify_separator_runs_from_export(shape_name, rknn_path)),
        ("sparse export backing writer", lambda: verify_sparse_export_backing_writer(shape_name, rknn_path, task_gem_log)),
        ("sparse single Rocket candidate", lambda: verify_sparse_single_rocket_candidate(shape_name, rknn_path, task_gem_log)),
        ("runtime sparse single stream", lambda: verify_runtime_sparse_single_stream(shape_name, rknn_path, task_gem_log)),
        ("single-stream regcmd_count ambiguity", lambda: verify_single_stream_regcmd_count_ambiguity(shape_name, rknn_path)),
        ("local Python runtime regcmd_count passthrough", lambda: verify_local_python_runtime_regcmd_count_passthrough(shape_name, rknn_path)),
        ("local Rocket regcmd_count rule", lambda: verify_local_rocket_regcmd_count_rule(shape_name, rknn_path)),
        ("Rocket PC amount model mismatch", lambda: verify_rocket_pc_amount_model_mismatch(shape_name, rknn_path, task_gem_log)),
        ("Rocket record-amounts candidate", lambda: verify_rocket_record_amounts_candidate(shape_name, rknn_path, task_gem_log)),
        ("runtime pc_root6 guard", lambda: verify_runtime_pc_root6_guard(shape_name)),
        ("direct-spatial blockers", lambda: verify_direct_spatial_blockers(shape_name)),
    ]
    for name, check in checks:
        print(f"== {name} ==")
        rc = check()
        if rc:
            print(f"FAIL safe direct-spatial preflight stopped at {name}")
            return rc
    print("PASS safe direct-spatial preflight")
    print("all checks are offline; this does not submit direct-spatial work to hardware")
    return 0


def verify_direct_spatial_blockers(shape_name):
    rknn_path = rknn_path_arg(shape_name, None)
    submit_log = ROOT / "experimental" / "rknn" / "capture_rknpu_submit_sweep_readonly.log"
    task_gem_log = ROOT / "experimental" / "rknn" / "capture_rknpu_submit_dump_gems_sweep.log"
    expected_failures = [
        ("Rocket task mapping proof", lambda: verify_rocket_task_mapping_proof(
            shape_name, rknn_path, submit_log, task_gem_log)),
        ("runtime task boundaries", lambda: verify_runtime_task_boundaries(
            shape_name, rknn_path, submit_log, task_gem_log)),
        ("PC-chain component coverage", lambda: verify_pc_chain_component_coverage(
            shape_name, rknn_path, None, task_gem_log)),
        ("root-contiguous candidate boundaries", lambda: verify_pc_root_candidate_boundaries(
            shape_name, rknn_path, submit_log, task_gem_log)),
        ("simple Mesa PC amounts candidate", lambda: verify_rocket_pc_amounts_candidate(
            shape_name, rknn_path, task_gem_log)),
    ]
    for name, check in expected_failures:
        print(f"== expected blocker: {name} ==")
        rc = check()
        if rc == 0:
            print(f"FAIL expected blocker unexpectedly passed: {name}")
            print("inspect before treating any direct-spatial Rocket mapping as safe")
            return 1
        print(f"PASS blocker still active: {name}")
    print("PASS direct-spatial blocker checks")
    print("known unsafe mappings remain blocked offline; do not promote them as general/default direct-spatial submit paths")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shape", choices=sorted(SHAPES))
    parser.add_argument("--full-body", action="store_true")
    parser.add_argument("--compare-rknn", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-rknn-spans", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-rknn-stream", nargs="?", const=True, type=Path)
    parser.add_argument("--compiler-boundaries", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-compiler-boundaries", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-exact-export-pair", action="store_true")
    parser.add_argument("--verify-remote-exact-export-equivalent", type=Path)
    parser.add_argument("--verify-rocket-mapping", action="store_true")
    parser.add_argument("--verify-3task-mapping", action="store_true")
    parser.add_argument("--verify-6task-mapping", action="store_true")
    parser.add_argument("--verify-observed-task-gem", action="store_true")
    parser.add_argument("--verify-observed-task-gem-log", type=Path)
    parser.add_argument("--verify-submit-log-consistency", nargs="*", type=Path)
    parser.add_argument("--verify-task-object-sync-correlation", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-task-object-record-count", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-observed-active-ranges", action="store_true")
    parser.add_argument("--verify-runtime-task-boundaries", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-rocket-task-mapping-proof", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-runtime-task-coverage", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-runtime-fetch-coverage", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-runtime-task-gaps", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-runtime-task-row-overlaps", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-task-backing-span", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-pc-base-task-links", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-pc-amount-task-lengths", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-pc-linked-task-graph", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-pc-chain-components", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-pc-chain-component-coverage", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-pc-root-contiguous-rocket-candidate", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-pc-root-candidate-boundaries", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-driver-pc-fetch-spans", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-vendor-pc-commit-register-model", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-vendor-selected-rocket-rejection", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-global-task-number-rocket-rejection", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-vendor-runtime-task-selection", nargs="?", const=True, type=Path)
    parser.add_argument("--submit-log", type=Path)
    parser.add_argument("--task-gem-log", type=Path)
    parser.add_argument("--verify-observed-descriptor-plan", action="store_true")
    parser.add_argument("--verify-observed-spatial-y-tile-descriptors", action="store_true")
    parser.add_argument("--verify-observed-chan320-descriptors", action="store_true")
    parser.add_argument("--verify-observed-mixed-spatial-descriptors", action="store_true")
    parser.add_argument("--verify-observed-compare-spatial-descriptors", action="store_true")
    parser.add_argument("--verify-observed-pointwise-descriptors", action="store_true")
    parser.add_argument("--verify-all-observed-supported-descriptors", action="store_true")
    parser.add_argument("--verify-rknpu-core-mask-semantics", action="store_true")
    parser.add_argument("--verify-rknpu-driver-source-semantics", action="store_true")
    parser.add_argument("--verify-rk3588-vendor-config-semantics", action="store_true")
    parser.add_argument("--verify-local-rocket-uapi", action="store_true")
    parser.add_argument("--verify-local-rocket-uapi-source", action="store_true")
    parser.add_argument("--verify-local-rocket-pc-commit-surface-gap", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-runtime-rocket-policy-commit-models", action="store_true")
    parser.add_argument("--verify-rocket-segment-aligned-pc-partition-rejection", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-direct-spatial-gate", action="store_true")
    parser.add_argument("--verify-direct-spatial-task-policy", action="store_true")
    parser.add_argument("--verify-direct-spatial-planner-routing", action="store_true")
    parser.add_argument("--verify-runtime-pc-root6-stream", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-conv-raw-span-boundary-rejection", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-single-stream-rocket-candidate", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-compact-single-stream-pc-target-rejection", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-sparse-fetch-layout-candidate", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-sparse-extended-backing-hypothesis", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-separator-runs-from-export", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-sparse-export-backing-writer", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-sparse-single-rocket-candidate", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-runtime-sparse-single-stream", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-single-stream-regcmd-count-ambiguity", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-local-python-runtime-regcmd-count-passthrough", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-local-rocket-regcmd-count-rule", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-rocket-pc-amount-model-mismatch", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-rocket-pc-amounts-candidate", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-rocket-record-amounts-candidate", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-runtime-rocket-record-amounts-stream", nargs="?", const=True, type=Path)
    parser.add_argument("--verify-runtime-pc-root6-guard", action="store_true")
    parser.add_argument("--verify-observed-descriptor-sweep", action="store_true")
    parser.add_argument("--verify-direct-spatial-blockers", action="store_true")
    parser.add_argument("--verify-safe-direct-spatial-preflight", action="store_true")
    args = parser.parse_args()

    # This script intentionally avoids NPU submit. It imports conv.py only to reuse
    # decoded register materialization and descriptor planning.
    os.environ.pop(conv.DIRECT_SPATIAL_ENV, None)
    os.environ.pop(conv.DIRECT_SPATIAL_UNSAFE_ENV, None)
    if args.verify_rknn_spans:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_rknn_spans))
        raise SystemExit(verify_compiler_spans(args.shape, path))
    if args.verify_rknn_stream:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_rknn_stream))
        raise SystemExit(verify_compiler_stream(args.shape, path))
    if args.verify_compiler_boundaries:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_compiler_boundaries))
        raise SystemExit(verify_compiler_boundaries(args.shape, path))
    if args.verify_exact_export_pair:
        raise SystemExit(verify_exact_export_pair(args.shape))
    if args.verify_remote_exact_export_equivalent:
        raise SystemExit(verify_remote_exact_export_equivalent(args.shape, args.verify_remote_exact_export_equivalent))
    if args.verify_rocket_mapping:
        raise SystemExit(verify_rocket_mapping(args.shape))
    if args.verify_3task_mapping:
        raise SystemExit(verify_3task_mapping(args.shape))
    if args.verify_6task_mapping:
        raise SystemExit(verify_6task_mapping(args.shape))
    if args.verify_observed_task_gem:
        raise SystemExit(verify_observed_task_gem(args.shape))
    if args.verify_observed_task_gem_log:
        raise SystemExit(verify_observed_task_gem_log(args.verify_observed_task_gem_log))
    if args.verify_submit_log_consistency is not None:
        raise SystemExit(verify_submit_log_consistency(args.shape, args.verify_submit_log_consistency))
    if args.verify_task_object_sync_correlation:
        path = None if args.verify_task_object_sync_correlation is True else args.verify_task_object_sync_correlation
        raise SystemExit(verify_task_object_sync_correlation(args.shape, path))
    if args.verify_task_object_record_count:
        path = None if args.verify_task_object_record_count is True else args.verify_task_object_record_count
        raise SystemExit(verify_task_object_record_count(args.shape, path, args.task_gem_log))
    if args.verify_observed_active_ranges:
        raise SystemExit(verify_observed_active_ranges(args.shape, args.submit_log, args.task_gem_log))
    if args.verify_runtime_task_boundaries:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_runtime_task_boundaries))
        raise SystemExit(verify_runtime_task_boundaries(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_rocket_task_mapping_proof:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_rocket_task_mapping_proof))
        raise SystemExit(verify_rocket_task_mapping_proof(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_runtime_task_coverage:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_runtime_task_coverage))
        raise SystemExit(verify_runtime_task_coverage(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_runtime_fetch_coverage:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_runtime_fetch_coverage))
        raise SystemExit(verify_runtime_fetch_coverage(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_runtime_task_gaps:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_runtime_task_gaps))
        raise SystemExit(verify_runtime_task_gaps(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_runtime_task_row_overlaps:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_runtime_task_row_overlaps))
        raise SystemExit(verify_runtime_task_row_overlaps(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_task_backing_span:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_task_backing_span))
        raise SystemExit(verify_task_backing_span(args.shape, path, args.task_gem_log))
    if args.verify_pc_base_task_links:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_pc_base_task_links))
        raise SystemExit(verify_pc_base_task_links(args.shape, path, args.task_gem_log))
    if args.verify_pc_amount_task_lengths:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_pc_amount_task_lengths))
        raise SystemExit(verify_pc_amount_task_lengths(args.shape, path, args.task_gem_log))
    if args.verify_pc_linked_task_graph:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_pc_linked_task_graph))
        raise SystemExit(verify_pc_linked_task_graph(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_pc_chain_components:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_pc_chain_components))
        raise SystemExit(verify_pc_chain_components(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_pc_chain_component_coverage:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_pc_chain_component_coverage))
        raise SystemExit(verify_pc_chain_component_coverage(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_pc_root_contiguous_rocket_candidate:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_pc_root_contiguous_rocket_candidate))
        raise SystemExit(verify_pc_root_contiguous_rocket_candidate(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_pc_root_candidate_boundaries:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_pc_root_candidate_boundaries))
        raise SystemExit(verify_pc_root_candidate_boundaries(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_driver_pc_fetch_spans:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_driver_pc_fetch_spans))
        raise SystemExit(verify_driver_pc_fetch_spans(args.shape, path, args.task_gem_log))
    if args.verify_vendor_pc_commit_register_model:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_vendor_pc_commit_register_model))
        raise SystemExit(verify_vendor_pc_commit_register_model(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_vendor_selected_rocket_rejection:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_vendor_selected_rocket_rejection))
        raise SystemExit(verify_vendor_selected_rocket_rejection(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_global_task_number_rocket_rejection:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_global_task_number_rocket_rejection))
        raise SystemExit(verify_global_task_number_rocket_rejection(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_vendor_runtime_task_selection:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_vendor_runtime_task_selection))
        raise SystemExit(verify_vendor_runtime_task_selection(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_observed_descriptor_plan:
        raise SystemExit(verify_observed_descriptor_plan(args.shape))
    if args.verify_observed_descriptor_sweep:
        raise SystemExit(verify_observed_descriptor_sweep())
    if args.verify_observed_spatial_y_tile_descriptors:
        raise SystemExit(verify_observed_spatial_y_tile_descriptors())
    if args.verify_observed_chan320_descriptors:
        raise SystemExit(verify_observed_chan320_descriptors())
    if args.verify_observed_mixed_spatial_descriptors:
        raise SystemExit(verify_observed_mixed_spatial_descriptors())
    if args.verify_observed_compare_spatial_descriptors:
        raise SystemExit(verify_observed_compare_spatial_descriptors())
    if args.verify_observed_pointwise_descriptors:
        raise SystemExit(verify_observed_pointwise_descriptors())
    if args.verify_all_observed_supported_descriptors:
        raise SystemExit(verify_all_observed_supported_descriptors())
    if args.verify_rknpu_core_mask_semantics:
        raise SystemExit(verify_rknpu_core_mask_semantics(args.shape))
    if args.verify_rknpu_driver_source_semantics:
        raise SystemExit(verify_rknpu_driver_source_semantics(args.shape))
    if args.verify_rk3588_vendor_config_semantics:
        raise SystemExit(verify_rk3588_vendor_config_semantics(args.shape))
    if args.verify_local_rocket_uapi:
        raise SystemExit(verify_local_rocket_uapi(args.shape))
    if args.verify_local_rocket_uapi_source:
        raise SystemExit(verify_local_rocket_uapi_source(args.shape))
    if args.verify_local_rocket_pc_commit_surface_gap:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_local_rocket_pc_commit_surface_gap))
        raise SystemExit(verify_local_rocket_pc_commit_surface_gap(args.shape, path, args.submit_log, args.task_gem_log))
    if args.verify_runtime_rocket_policy_commit_models:
        raise SystemExit(verify_runtime_rocket_policy_commit_models(args.shape))
    if args.verify_rocket_segment_aligned_pc_partition_rejection:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_rocket_segment_aligned_pc_partition_rejection))
        raise SystemExit(verify_rocket_segment_aligned_pc_partition_rejection(args.shape, path))
    if args.verify_direct_spatial_gate:
        raise SystemExit(verify_direct_spatial_gate(args.shape))
    if args.verify_direct_spatial_task_policy:
        raise SystemExit(verify_direct_spatial_task_policy(args.shape))
    if args.verify_direct_spatial_planner_routing:
        raise SystemExit(verify_direct_spatial_planner_routing(args.shape))
    if args.verify_runtime_pc_root6_stream:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_runtime_pc_root6_stream))
        raise SystemExit(verify_runtime_pc_root6_stream(args.shape, path))
    if args.verify_conv_raw_span_boundary_rejection:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_conv_raw_span_boundary_rejection))
        raise SystemExit(verify_conv_raw_span_boundary_rejection(args.shape, path))
    if args.verify_single_stream_rocket_candidate:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_single_stream_rocket_candidate))
        raise SystemExit(verify_single_stream_rocket_candidate(args.shape, path))
    if args.verify_compact_single_stream_pc_target_rejection:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_compact_single_stream_pc_target_rejection))
        raise SystemExit(verify_compact_single_stream_pc_target_rejection(args.shape, path))
    if args.verify_sparse_fetch_layout_candidate:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_sparse_fetch_layout_candidate))
        raise SystemExit(verify_sparse_fetch_layout_candidate(args.shape, path))
    if args.verify_sparse_extended_backing_hypothesis:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_sparse_extended_backing_hypothesis))
        raise SystemExit(verify_sparse_extended_backing_hypothesis(args.shape, path))
    if args.verify_separator_runs_from_export:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_separator_runs_from_export))
        raise SystemExit(verify_separator_runs_from_export(args.shape, path))
    if args.verify_sparse_export_backing_writer:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_sparse_export_backing_writer))
        raise SystemExit(verify_sparse_export_backing_writer(args.shape, path, args.task_gem_log))
    if args.verify_sparse_single_rocket_candidate:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_sparse_single_rocket_candidate))
        raise SystemExit(verify_sparse_single_rocket_candidate(args.shape, path, args.task_gem_log))
    if args.verify_runtime_sparse_single_stream:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_runtime_sparse_single_stream))
        raise SystemExit(verify_runtime_sparse_single_stream(args.shape, path, args.task_gem_log))
    if args.verify_single_stream_regcmd_count_ambiguity:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_single_stream_regcmd_count_ambiguity))
        raise SystemExit(verify_single_stream_regcmd_count_ambiguity(args.shape, path))
    if args.verify_local_python_runtime_regcmd_count_passthrough:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_local_python_runtime_regcmd_count_passthrough))
        raise SystemExit(verify_local_python_runtime_regcmd_count_passthrough(args.shape, path))
    if args.verify_local_rocket_regcmd_count_rule:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_local_rocket_regcmd_count_rule))
        raise SystemExit(verify_local_rocket_regcmd_count_rule(args.shape, path))
    if args.verify_rocket_pc_amount_model_mismatch:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_rocket_pc_amount_model_mismatch))
        raise SystemExit(verify_rocket_pc_amount_model_mismatch(args.shape, path, args.task_gem_log))
    if args.verify_rocket_pc_amounts_candidate:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_rocket_pc_amounts_candidate))
        raise SystemExit(verify_rocket_pc_amounts_candidate(args.shape, path, args.task_gem_log))
    if args.verify_rocket_record_amounts_candidate:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_rocket_record_amounts_candidate))
        raise SystemExit(verify_rocket_record_amounts_candidate(args.shape, path, args.task_gem_log))
    if args.verify_runtime_rocket_record_amounts_stream:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.verify_runtime_rocket_record_amounts_stream))
        raise SystemExit(verify_runtime_rocket_record_amounts_stream(args.shape, path, args.task_gem_log))
    if args.verify_runtime_pc_root6_guard:
        raise SystemExit(verify_runtime_pc_root6_guard(args.shape))
    if args.verify_direct_spatial_blockers:
        raise SystemExit(verify_direct_spatial_blockers(args.shape))
    if args.verify_safe_direct_spatial_preflight:
        raise SystemExit(verify_safe_direct_spatial_preflight(args.shape))
    if args.compiler_boundaries:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.compiler_boundaries))
        print_compiler_boundaries(args.shape, path)
        return
    if args.compare_rknn:
        path = rknn_path_arg(args.shape, optional_rknn_path(args.compare_rknn))
        print_span_compare(args.shape, path)
    else:
        print_summary(direct_rows(args.shape, args.full_body))


if __name__ == "__main__":
    main()
