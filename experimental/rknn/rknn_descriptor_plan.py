#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "kernel_6_18"))

import conv  # noqa: E402

sys.path.insert(0, str(ROOT / "experimental" / "rknn"))
import rknn_parse_regcmd_runs as regcmd_runs  # noqa: E402


@dataclass(frozen=True)
class Shape:
    in_c: int
    in_h: int
    in_w: int
    out_c: int
    kh: int
    kw: int
    groups: int
    stride: int = 1


SHAPES = {
    "cmp_b1_c32_h14_w14_oc128_wic32_k3x3_g1": Shape(32, 14, 14, 128, 3, 3, 1),
    "cmp_b1_c160_h14_w14_oc320_wic160_k3x3_g1": Shape(160, 14, 14, 320, 3, 3, 1),
    "b1_c160_h7_w7_oc320_wic160_k3x3_g1": Shape(160, 7, 7, 320, 3, 3, 1),
    "goal_b1_c160_h7_w7_oc320_wic160_k3x3_g1": Shape(160, 7, 7, 320, 3, 3, 1),
    "cmp_b1_c528_h14_w14_oc32_wic528_k1x1_g1": Shape(528, 14, 14, 32, 1, 1, 1),
    "cmp_b1_c32_h150_w150_oc32_wic1_k3x3_g32": Shape(32, 150, 150, 32, 3, 3, 32),
    "mix_b1_c160_h40_w40_oc320_wic160_k3x3_g1": Shape(160, 40, 40, 320, 3, 3, 1),
    "mix_b1_c72_h20_w20_oc288_wic72_k3x3_g1": Shape(72, 20, 20, 288, 3, 3, 1),
    "mix_b1_c40_h40_w40_oc320_wic40_k1x1_g1": Shape(40, 40, 40, 320, 1, 1, 1),
    "sweep_b1_c160_h16_w16_oc320_wic160_k3x3_g1": Shape(160, 16, 16, 320, 3, 3, 1),
    "sweep_b1_c160_h20_w20_oc320_wic160_k3x3_g1": Shape(160, 20, 20, 320, 3, 3, 1),
    "sweep_b1_c160_h24_w24_oc320_wic160_k3x3_g1": Shape(160, 24, 24, 320, 3, 3, 1),
    "sweep_b1_c160_h28_w28_oc320_wic160_k3x3_g1": Shape(160, 28, 28, 320, 3, 3, 1),
    "sweep_b1_c160_h32_w32_oc320_wic160_k3x3_g1": Shape(160, 32, 32, 320, 3, 3, 1),
    "sweep_b1_c160_h36_w36_oc320_wic160_k3x3_g1": Shape(160, 36, 36, 320, 3, 3, 1),
    "sweep_b1_c160_h40_w40_oc320_wic160_k3x3_g1": Shape(160, 40, 40, 320, 3, 3, 1),
    "chan_b1_c32_h20_w20_oc320_wic32_k3x3_g1": Shape(32, 20, 20, 320, 3, 3, 1),
    "chan_b1_c40_h20_w20_oc320_wic40_k3x3_g1": Shape(40, 20, 20, 320, 3, 3, 1),
    "chan_b1_c64_h20_w20_oc320_wic64_k3x3_g1": Shape(64, 20, 20, 320, 3, 3, 1),
    "chan_b1_c72_h20_w20_oc320_wic72_k3x3_g1": Shape(72, 20, 20, 320, 3, 3, 1),
    "chan_b1_c96_h20_w20_oc320_wic96_k3x3_g1": Shape(96, 20, 20, 320, 3, 3, 1),
    "chan_b1_c128_h20_w20_oc320_wic128_k3x3_g1": Shape(128, 20, 20, 320, 3, 3, 1),
    "chan_b1_c160_h20_w20_oc320_wic160_k3x3_g1": Shape(160, 20, 20, 320, 3, 3, 1),
    "chan_b1_c192_h20_w20_oc320_wic192_k3x3_g1": Shape(192, 20, 20, 320, 3, 3, 1),
    "chan_b1_c256_h20_w20_oc320_wic256_k3x3_g1": Shape(256, 20, 20, 320, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc32_wic64_k3x3_g1": Shape(64, 20, 20, 32, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc48_wic64_k3x3_g1": Shape(64, 20, 20, 48, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc64_wic64_k3x3_g1": Shape(64, 20, 20, 64, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc96_wic64_k3x3_g1": Shape(64, 20, 20, 96, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc112_wic64_k3x3_g1": Shape(64, 20, 20, 112, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc128_wic64_k3x3_g1": Shape(64, 20, 20, 128, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc160_wic64_k3x3_g1": Shape(64, 20, 20, 160, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc192_wic64_k3x3_g1": Shape(64, 20, 20, 192, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc224_wic64_k3x3_g1": Shape(64, 20, 20, 224, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc256_wic64_k3x3_g1": Shape(64, 20, 20, 256, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc288_wic64_k3x3_g1": Shape(64, 20, 20, 288, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc320_wic64_k3x3_g1": Shape(64, 20, 20, 320, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc384_wic64_k3x3_g1": Shape(64, 20, 20, 384, 3, 3, 1),
    "outc_b1_c64_h20_w20_oc512_wic64_k3x3_g1": Shape(64, 20, 20, 512, 3, 3, 1),
    "pw_b1_c40_h20_w20_oc320_wic40_k1x1_g1": Shape(40, 20, 20, 320, 1, 1, 1),
    "pw_b1_c40_h28_w28_oc320_wic40_k1x1_g1": Shape(40, 28, 28, 320, 1, 1, 1),
    "pw_b1_c40_h40_w40_oc320_wic40_k1x1_g1": Shape(40, 40, 40, 320, 1, 1, 1),
    "pw_b1_c40_h56_w56_oc320_wic40_k1x1_g1": Shape(40, 56, 56, 320, 1, 1, 1),
    "pw_b1_c64_h20_w20_oc128_wic64_k1x1_g1": Shape(64, 20, 20, 128, 1, 1, 1),
    "pw_b1_c64_h40_w40_oc128_wic64_k1x1_g1": Shape(64, 40, 40, 128, 1, 1, 1),
    "pw_b1_c256_h14_w14_oc512_wic256_k1x1_g1": Shape(256, 14, 14, 512, 1, 1, 1),
    "pw_b1_c256_h28_w28_oc512_wic256_k1x1_g1": Shape(256, 28, 28, 512, 1, 1, 1),
    "pw_b1_c528_h14_w14_oc32_wic528_k1x1_g1": Shape(528, 14, 14, 32, 1, 1, 1),
    "pw_b1_c528_h20_w20_oc32_wic528_k1x1_g1": Shape(528, 20, 20, 32, 1, 1, 1),
    "pw_b1_c528_h40_w40_oc32_wic528_k1x1_g1": Shape(528, 40, 40, 32, 1, 1, 1),
}


OUTC64_H20_POST = {
    32: ("k_half_y_tile", None),
    48: ("y_mid_k_tile", [16, 16, 16]),
    64: ("k_half_y_tile", None),
    96: ("k_half_k_tile", [32, 32, 32]),
    112: ("y_mid_y_tile", None),
    128: ("k_half_y_tile", None),
    160: ("k_half_y_tile", None),
    192: ("k_half_k_tile", [64, 64, 64]),
    224: ("k_half_y_tile", None),
    256: ("k_half_y_tile", None),
    288: ("k_half_k_tile", [96, 96, 96]),
    320: ("k_half_k_tile", [112, 112, 96]),
    384: ("k_half_k_tile", [128, 128, 128]),
    512: ("k_half_k_tile", [176, 176, 160]),
}


SWEEP160_Y_WINDOWS = {
    7: [(0, 7, 5)],
    14: [(0, 14, 12)],
    16: [(0, 16, 14)],
    20: [(0, 20, 18)],
    24: [(0, 24, 22)],
    28: [(0, 28, 26)],
    32: [(0, 28, 26), (26, 6, 4)],
    36: [(0, 25, 23), (23, 13, 11)],
    40: [(0, 23, 21), (21, 19, 17)],
}


CHAN320_IN_C = {32, 40, 64, 72, 96, 128, 160, 192, 256}


MIX72_H20_K_TILE_COUNTS = [96, 96, 96]


CMP32_H14_Y_WINDOWS = [(0, 6, 4), (4, 6, 4), (8, 6, 4)]


PW_SIMPLE_Y_WINDOWS = {
    14: [(0, 5), (5, 5), (10, 4)],
    20: [(0, 7), (7, 7), (14, 6)],
    28: [(0, 10), (10, 9), (19, 9)],
    40: [(0, 14), (14, 13), (27, 13)],
    56: [(0, 19), (19, 19), (38, 18)],
}


PW_CROSSED = {
    (256, 28, 512): {
        "setup": [(0, 9), (9, 9)],
        "k_half": [(0, 9), (9, 9), (18, 9), (27, 1)],
        "y_tile": [(0, 5), (15, 5), (5, 5), (20, 4), (10, 5), (24, 4)],
    },
    (528, 20, 32): {
        "setup": [(0, 15), (15, 5)],
        "k_half": [(0, 15), (15, 5)],
        "y_tile": [(0, 7), (7, 7), (14, 6)],
    },
    (528, 40, 32): {
        "setup": [(0, 7), (7, 7)],
        "k_half": [(0, 7), (7, 7), (14, 7), (21, 7), (28, 7), (35, 5)],
        "y_tile": [(0, 7), (21, 7), (7, 7), (28, 6), (14, 7), (34, 6)],
    },
}


def align_up(value, align):
    return ((value + align - 1) // align) * align


def family_bits(name):
    return {
        "setup": 0x00000000,
        "y_mid": 0x10000000,
        "y_tile": 0x20000000,
        "k_half": 0x40000000,
        "k_tile": 0x50000000,
    }[name]


def infer_y_windows(shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    _p, _split, y_boundary, _k_boundary, _reason, _ys, _ks = conv.plan_conv_tiles(
        shape.in_c, shape.out_c, shape.kh, shape.kw,
        shape.in_h, shape.in_w, shape.groups, shape.stride,
    )
    out_h = p["out_h"]
    out_w = p["out_w"]
    windows = []
    for idx, start in enumerate(y_boundary[:-1]):
        out_rows = y_boundary[idx + 1] - start
        if out_rows <= 0:
            continue
        input_rows = (out_rows - 1) * shape.stride + shape.kh
        if out_rows == out_h:
            input_rows = shape.in_h
        windows.append({
            "y_start": start,
            "input_h": input_rows,
            "output_h": out_rows,
            "output_w": out_w,
        })
    return windows


def bytes_per_output_row(shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    c2 = conv.UNPACK_C2
    return p["out_w"] * c2 * conv.FP16_BYTES


def feature_row_bytes(shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    if p["is_depthwise"]:
        tile_in_c = min(32, shape.in_c)
    else:
        tile_in_c = shape.in_c
    aligned = align_up(tile_in_c, p["align_c"])
    return p["width_stride"] * aligned * conv.FP16_BYTES


def k_windows_for_family(shape, name):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    if name == "setup":
        return [(0, shape.out_c)]
    if name == "k_half":
        half = align_up(shape.out_c // 2, 16)
        return [(0, half), (half, shape.out_c - half)]
    if name == "k_tile":
        if p["is_depthwise"]:
            step = min(32, shape.out_c)
        elif shape.kh == 1 and shape.kw == 1:
            step = conv._pointwise_oc_tile_c(shape.in_c)
        else:
            # Observed RKNN spatial k_tile uses about 3 CBUF banks of weights.
            data_in = align_up(shape.in_c, p["align_c"])
            per_oc = shape.kh * shape.kw * data_in * conv.FP16_BYTES
            step = max(16, (3 * conv.CBUF_BANK_SIZE) // per_oc)
            step = align_up(step, 16)
        out = []
        start = 0
        while start < shape.out_c:
            count = min(step, shape.out_c - start)
            out.append((start, count))
            start += count
        return out
    return []


def families_for_shape(shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    if p["is_depthwise"]:
        return ["setup", "y_mid", "y_tile"]
    if shape.out_c > 32:
        return ["setup", "k_half", "k_tile"]
    return ["setup", "k_half", "y_tile"]


def estimate_descriptors(shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    y_windows = infer_y_windows(shape)
    full_y = [{
        "y_start": 0,
        "input_h": shape.in_h,
        "output_h": p["out_h"],
        "output_w": p["out_w"],
    }]
    feature_stride = feature_row_bytes(shape)
    output_stride = bytes_per_output_row(shape)
    descs = []
    for family in families_for_shape(shape):
        y_iter = y_windows if (family in {"setup", "y_mid", "y_tile"} and len(y_windows) > 1) else full_y
        if family in {"k_half", "k_tile"} and len(y_windows) > 1:
            y_iter = y_windows
        for k_start, k_count in k_windows_for_family(shape, family):
            data_in = align_up(k_count if p["is_depthwise"] else shape.in_c, p["align_c"])
            weight_per_oc = shape.kh * shape.kw * data_in * conv.FP16_BYTES
            weight_off = 0 if p["is_depthwise"] else k_start * weight_per_oc
            output_k_base = 0 if p["is_depthwise"] else k_start * p["out_h"] * p["out_w"] * conv.FP16_BYTES
            for y in y_iter:
                y_start = y["y_start"]
                feature_off = y_start * shape.stride * feature_stride
                output_y_base = y_start * output_stride
                descs.append({
                    "family": family,
                    "family_bits": family_bits(family),
                    "input_h": y["input_h"],
                    "output_h": y["output_h"],
                    "output_w": y["output_w"],
                    "oc_count": 1 if p["is_depthwise"] else k_count,
                    "feature_off": feature_off,
                    "weight_off": weight_off,
                    "output_off": output_k_base + output_y_base,
                })
    return descs


def desc(family, input_h, output_h, output_w, oc_count, feature_off, weight_off, output_off):
    return {
        "family": family,
        "family_bits": family_bits(family),
        "grain_bits": None,
        "cbuf0": None,
        "pc_core": 0,
        "input_h": input_h,
        "output_h": output_h,
        "output_w": output_w,
        "oc_count": oc_count,
        "feature_off": feature_off,
        "weight_off": weight_off,
        "output_off": output_off,
    }


def add_k_half(descs, shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    half = align_up(shape.out_c // 2, 16)
    weight_per_oc = shape.kh * shape.kw * shape.in_c * conv.FP16_BYTES
    output_per_oc = p["out_width_stride"] * conv.FP16_BYTES
    for k_start, k_count in [(0, half), (half, shape.out_c - half)]:
        descs.append(desc(
            "k_half", shape.in_h, p["out_h"], p["out_w"], k_count,
            0, k_start * weight_per_oc, k_start * output_per_oc,
        ))


def add_full_y_tiles(descs, shape, family="y_tile"):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    feature_row = shape.in_w * 16
    output_row = p["out_w"] * conv.UNPACK_C2 * conv.FP16_BYTES
    for y_start, out_rows in [(0, 6), (6, 6), (12, 6)]:
        input_rows = (out_rows - 1) * shape.stride + shape.kh
        descs.append(desc(
            family, input_rows, out_rows, p["out_w"], shape.out_c,
            y_start * feature_row, 0, y_start * output_row,
        ))


def add_y_mid(descs, shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    feature_row = shape.in_w * 16
    output_row = p["out_w"] * conv.UNPACK_C2 * conv.FP16_BYTES
    for y_start in (0, 9):
        descs.append(desc(
            "y_mid", 11, 9, p["out_w"], shape.out_c,
            y_start * feature_row, 0, y_start * output_row,
        ))


def add_k_tiles(descs, shape, counts):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    weight_per_oc = shape.kh * shape.kw * shape.in_c * conv.FP16_BYTES
    output_per_oc = p["out_width_stride"] * conv.FP16_BYTES
    k_start = 0
    for k_count in counts:
        descs.append(desc(
            "k_tile", shape.in_h, p["out_h"], p["out_w"], k_count,
            0, k_start * weight_per_oc, k_start * output_per_oc,
        ))
        k_start += k_count


def add_family_cross_y(descs, shape, family, k_windows, y_windows):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    weight_per_oc = shape.kh * shape.kw * shape.in_c * conv.FP16_BYTES
    output_per_oc = p["out_width_stride"] * conv.FP16_BYTES
    feature_row = shape.in_w * conv.UNPACK_C2 * conv.FP16_BYTES
    output_row = p["out_w"] * conv.UNPACK_C2 * conv.FP16_BYTES
    for k_start, k_count in k_windows:
        for y_start, input_rows, output_rows in y_windows:
            descs.append(desc(
                family, input_rows, output_rows, p["out_w"], k_count,
                y_start * feature_row,
                k_start * weight_per_oc,
                k_start * output_per_oc + y_start * output_row,
            ))


def observed_outc64_h20_descriptors(shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    mode, k_counts = OUTC64_H20_POST[shape.out_c]
    descs = [desc("setup", shape.in_h, p["out_h"], p["out_w"], shape.out_c, 0, 0, 0)]
    if mode.startswith("k_half"):
        add_k_half(descs, shape)
    elif mode.startswith("y_mid"):
        add_y_mid(descs, shape)
    if mode.endswith("y_tile"):
        add_full_y_tiles(descs, shape)
    elif mode.endswith("k_tile"):
        add_k_tiles(descs, shape, k_counts)
    return descs


def observed_sweep160_descriptors(shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    y_windows = SWEEP160_Y_WINDOWS[shape.in_h]
    descs = []
    add_family_cross_y(descs, shape, "setup", [(0, shape.out_c)], y_windows)
    add_family_cross_y(descs, shape, "k_half", [(0, 160), (160, 160)], y_windows)
    add_family_cross_y(descs, shape, "k_tile", [(0, 112), (112, 112), (224, 96)], y_windows)
    grain_bits = 0xc0 if shape.in_h >= 28 else 0xf0
    cbuf0 = 0x39 if shape.in_h >= 32 else {7: 0xb1, 14: None, 16: 0x93, 20: 0x84, 24: 0x66, 28: 0x48}[shape.in_h]
    for d in descs:
        d["grain_bits"] = 0xa0 if shape.in_h == 7 else grain_bits
        d["cbuf0"] = cbuf0
        d["pc_core"] = {"setup": 0, "k_half": 1, "k_tile": 2}[d["family"]]
    assert all(d["output_w"] == p["out_w"] for d in descs)
    return descs


def observed_chan320_descriptors(shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    descs = [desc("setup", shape.in_h, p["out_h"], p["out_w"], shape.out_c, 0, 0, 0)]
    add_k_half(descs, shape)
    if shape.in_c in {32, 40}:
        add_full_y_tiles(descs, shape)
    else:
        add_k_tiles(descs, shape, [112, 112, 96])
    return descs


def observed_mix72_h20_descriptors(shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    descs = [desc("setup", shape.in_h, p["out_h"], p["out_w"], shape.out_c, 0, 0, 0)]
    add_k_half(descs, shape)
    add_k_tiles(descs, shape, MIX72_H20_K_TILE_COUNTS)
    return descs


def observed_cmp32_h14_descriptors(shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    descs = [desc("setup", shape.in_h, p["out_h"], p["out_w"], shape.out_c, 0, 0, 0)]
    descs[0]["grain_bits"] = 0x110
    add_k_half(descs, shape)
    for d in descs[1:]:
        d["grain_bits"] = 0x110
    feature_row = shape.in_w * conv.UNPACK_C2 * conv.FP16_BYTES
    output_row = p["out_w"] * conv.UNPACK_C2 * conv.FP16_BYTES
    for y_start, input_rows, output_rows in CMP32_H14_Y_WINDOWS:
        d = desc(
            "y_tile", input_rows, output_rows, p["out_w"], shape.out_c,
            y_start * feature_row, 0, y_start * output_row,
        )
        d["grain_bits"] = 0x90
        descs.append(d)
    return descs


def observed_pointwise_simple_descriptors(shape):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    descs = [desc("setup", shape.in_h, p["out_h"], p["out_w"], shape.out_c, 0, 0, 0)]
    add_k_half(descs, shape)
    feature_row = shape.in_w * conv.UNPACK_C2 * conv.FP16_BYTES
    output_row = p["out_w"] * conv.UNPACK_C2 * conv.FP16_BYTES
    for y_start, out_rows in PW_SIMPLE_Y_WINDOWS[shape.in_h]:
        descs.append(desc(
            "y_tile", out_rows, out_rows, p["out_w"], shape.out_c,
            y_start * feature_row, 0, y_start * output_row,
        ))
    return descs


def add_pointwise_y_list(descs, shape, family, k_windows, y_windows):
    p = conv._conv_params(
        shape.in_c, shape.in_h, shape.in_w, shape.out_c,
        shape.kh, shape.kw, shape.groups, shape.stride,
    )
    weight_per_oc = shape.in_c * conv.FP16_BYTES
    output_per_oc = p["out_h"] * p["out_w"] * conv.FP16_BYTES
    feature_row = shape.in_w * conv.UNPACK_C2 * conv.FP16_BYTES
    output_row = p["out_w"] * conv.UNPACK_C2 * conv.FP16_BYTES
    for k_start, k_count in k_windows:
        for y_start, out_rows in y_windows:
            descs.append(desc(
                family, out_rows, out_rows, p["out_w"], k_count,
                y_start * feature_row,
                k_start * weight_per_oc,
                k_start * output_per_oc + y_start * output_row,
            ))


def observed_pointwise_crossed_descriptors(shape):
    cfg = PW_CROSSED[(shape.in_c, shape.in_h, shape.out_c)]
    half = align_up(shape.out_c // 2, 16)
    descs = []
    add_pointwise_y_list(descs, shape, "setup", [(0, shape.out_c)], cfg["setup"])
    add_pointwise_y_list(descs, shape, "k_half", [(0, half), (half, shape.out_c - half)], cfg["k_half"])
    add_pointwise_y_list(descs, shape, "y_tile", [(0, shape.out_c)], cfg["y_tile"])
    return descs


def observed_descriptors(shape):
    if (
        shape.in_c == 64 and shape.in_h == 20 and shape.in_w == 20
        and shape.kh == 3 and shape.kw == 3 and shape.groups == 1
        and shape.out_c in OUTC64_H20_POST
    ):
        return observed_outc64_h20_descriptors(shape)
    if (
        shape.in_c == 160 and shape.in_h == shape.in_w and shape.out_c == 320
        and shape.kh == 3 and shape.kw == 3 and shape.groups == 1
        and shape.in_h in SWEEP160_Y_WINDOWS
    ):
        return observed_sweep160_descriptors(shape)
    if (
        shape.in_c in CHAN320_IN_C and shape.in_h == 20 and shape.in_w == 20
        and shape.out_c == 320 and shape.kh == 3 and shape.kw == 3
        and shape.groups == 1
    ):
        return observed_chan320_descriptors(shape)
    if (
        shape.in_c == 72 and shape.in_h == 20 and shape.in_w == 20
        and shape.out_c == 288 and shape.kh == 3 and shape.kw == 3
        and shape.groups == 1
    ):
        return observed_mix72_h20_descriptors(shape)
    if (
        shape.in_c == 32 and shape.in_h == 14 and shape.in_w == 14
        and shape.out_c == 128 and shape.kh == 3 and shape.kw == 3
        and shape.groups == 1
    ):
        return observed_cmp32_h14_descriptors(shape)
    if (
        shape.kh == 1 and shape.kw == 1 and shape.groups == 1
        and shape.in_h == shape.in_w and shape.in_h in PW_SIMPLE_Y_WINDOWS
        and (
            (shape.in_c, shape.out_c) in {(40, 320), (64, 128)}
            or (shape.in_c, shape.out_c, shape.in_h) in {(528, 32, 14), (256, 512, 14)}
        )
    ):
        return observed_pointwise_simple_descriptors(shape)
    if (
        shape.kh == 1 and shape.kw == 1 and shape.groups == 1
        and shape.in_h == shape.in_w
        and (shape.in_c, shape.in_h, shape.out_c) in PW_CROSSED
    ):
        return observed_pointwise_crossed_descriptors(shape)
    return estimate_descriptors(shape)


def observed_supported(shape):
    return observed_descriptors(shape) != estimate_descriptors(shape)


def compiler_descriptors(path):
    buf = path.read_bytes()
    out = []
    for idx, run in enumerate(regcmd_runs.find_runs(buf, 24), 1):
        desc = regcmd_runs.infer_descriptor(regcmd_runs.run_values(buf, run))
        if desc["output_c_count"] == 0 and desc["input_h"] == 0:
            continue
        out.append({
            "idx": idx,
            "family": desc["family"],
            "family_bits": desc["family_bits"],
            "grain_bits": desc["grain_bits"],
            "input_h": desc["input_h"],
            "output_h": desc["output_h"],
            "output_w": desc["output_w"],
            "input_w": desc["input_w"],
            "kernel_h": desc["kernel_h"],
            "kernel_w": desc["kernel_w"],
            "oc_count": desc["output_c_count"],
            "oc_end": desc["output_c_end"],
            "feature_off": desc["feature_offset"],
            "weight_off": desc["weight_offset"],
            "output_off": desc["output_offset"],
            "weight_bytes": desc["weight_bytes"],
            "cbuf0": desc["cbuf0"],
        })
    return out


def descriptor_key(desc):
    grain_bits = desc.get("grain_bits")
    return (
        desc["family"],
        desc["family_bits"],
        grain_bits if grain_bits is not None else "any",
        desc["input_h"],
        desc["output_h"],
        desc["output_w"],
        desc["oc_count"],
        desc["feature_off"],
        desc["weight_off"],
        desc["output_off"],
    )


def shape_name_from_path(path):
    return path.name[:-5] if path.name.endswith(".rknn") else path.stem


def descriptors_for_mode(shape, mode):
    return observed_descriptors(shape) if mode == "observed" else estimate_descriptors(shape)


def compare(shape_name, rknn_path, mode):
    shape = SHAPES[shape_name]
    estimated = descriptors_for_mode(shape, mode)
    compiler = compiler_descriptors(rknn_path)
    max_len = max(len(estimated), len(compiler))
    print("idx,status,compiler,estimated")
    for idx in range(max_len):
        c = compiler[idx] if idx < len(compiler) else None
        e = estimated[idx] if idx < len(estimated) else None
        if c is None:
            status = "extra_estimated"
        elif e is None:
            status = "missing_estimated"
        elif descriptor_match(c, e):
            status = "match"
        else:
            status = "diff"
        print(f"{idx + 1},{status},{format_desc(c)},{format_desc(e)}")


def coverage(paths, mode):
    print("file,shape_known,observed_supported,runs,status")
    failed = False
    for path in paths:
        shape_name = shape_name_from_path(path)
        if shape_name not in SHAPES:
            print(f"{path.name},no,no,0,unknown_shape")
            failed = True
            continue
        shape = SHAPES[shape_name]
        expected = descriptors_for_mode(shape, mode)
        actual = compiler_descriptors(path)
        status = "match"
        if len(expected) != len(actual):
            status = "diff"
        else:
            for c, e in zip(actual, expected):
                if not descriptor_match(c, e):
                    status = "diff"
                    break
        supported = "yes" if observed_supported(shape) else "no"
        print(f"{path.name},yes,{supported},{len(actual)},{status}")
        if status != "match":
            failed = True
    return 1 if failed else 0


def grain_summary(paths):
    groups = {}
    for path in paths:
        for desc in compiler_descriptors(path):
            key = (
                desc["family"],
                desc["input_h"],
                desc["output_h"],
                desc["output_w"],
                desc["oc_count"],
            )
            groups.setdefault(key, set()).add(desc["grain_bits"])
    print("family,input_h,output_h,output_w,oc_count,grain_bits")
    for key in sorted(groups):
        grains = ",".join(f"0x{value:08x}" for value in sorted(groups[key]))
        print(f"{key[0]},{key[1]},{key[2]},{key[3]},{key[4]},{grains}")


def grain_context(paths):
    print(
        "file,idx,family,grain_bits,in_c,out_c,kh,kw,in_h,in_w,"
        "input_h,input_w,output_h,output_w,oc_count,oc_end,"
        "feature_off,weight_off,output_off,weight_bytes,cbuf0"
    )
    for path in paths:
        shape_name = shape_name_from_path(path)
        shape = SHAPES.get(shape_name)
        for desc in compiler_descriptors(path):
            shape_fields = {
                "in_c": shape.in_c if shape else "",
                "out_c": shape.out_c if shape else "",
                "kh": shape.kh if shape else "",
                "kw": shape.kw if shape else "",
                "in_h": shape.in_h if shape else "",
                "in_w": shape.in_w if shape else "",
            }
            print(
                f"{path.name},{desc['idx']},{desc['family']},"
                f"0x{desc['grain_bits']:08x},"
                f"{shape_fields['in_c']},{shape_fields['out_c']},"
                f"{shape_fields['kh']},{shape_fields['kw']},"
                f"{shape_fields['in_h']},{shape_fields['in_w']},"
                f"{desc['input_h']},{desc['input_w']},"
                f"{desc['output_h']},{desc['output_w']},"
                f"{desc['oc_count']},{desc['oc_end']},"
                f"0x{desc['feature_off']:x},0x{desc['weight_off']:x},"
                f"0x{desc['output_off']:x},0x{desc['weight_bytes']:x},"
                f"0x{desc['cbuf0']:x}"
            )


def grain_matrix(paths):
    groups = {}
    examples = {}
    for path in paths:
        shape_name = shape_name_from_path(path)
        shape = SHAPES.get(shape_name)
        for desc in compiler_descriptors(path):
            key = (
                desc["family"],
                shape.kh if shape else "",
                shape.kw if shape else "",
                shape.in_c if shape else "",
                desc["input_h"],
                desc["output_h"],
                desc["output_w"],
                desc["oc_count"],
                f"0x{desc['cbuf0']:x}",
            )
            groups.setdefault(key, set()).add(desc["grain_bits"])
            examples.setdefault(key, path.name)
    print("family,kh,kw,in_c,input_h,output_h,output_w,oc_count,cbuf0,grain_bits,example")
    for key in sorted(groups):
        grains = "|".join(f"0x{value:08x}" for value in sorted(groups[key]))
        print(",".join(str(item) for item in key) + f",{grains},{examples[key]}")


def compiler_summary(rknn_path):
    descs = compiler_descriptors(rknn_path)
    families = []
    for desc in descs:
        item = f"{desc['family']}:{desc['oc_count']}x{desc['output_h']}"
        if not families or families[-1] != item:
            families.append(item)
    k_tile_ocs = [desc["oc_count"] for desc in descs if desc["family"] == "k_tile"]
    y_tile_rows = [desc["output_h"] for desc in descs if desc["family"] == "y_tile"]
    y_windows = []
    for desc in descs:
        window = (desc["input_h"], desc["output_h"], desc["feature_off"], desc["output_off"])
        if window not in y_windows:
            y_windows.append(window)
    print(f"file={rknn_path.name}")
    print(f"runs={len(descs)}")
    print(f"sequence={' '.join(families)}")
    print(f"k_tile_ocs={','.join(str(v) for v in k_tile_ocs) or '-'}")
    print(f"y_tile_output_rows={','.join(str(v) for v in y_tile_rows) or '-'}")
    print(
        "unique_y_windows="
        + ";".join(
            f"ih{ih}:oh{oh}:fin0x{fin:x}:out0x{out:x}"
            for ih, oh, fin, out in y_windows
        )
    )


def format_desc(desc):
    if desc is None:
        return "-"
    return (
        f"{desc['family']} fh=0x{desc['family_bits']:08x} "
        f"gr={format_optional_hex(desc.get('grain_bits'))} "
        f"ih={desc['input_h']} oh={desc['output_h']} ow={desc['output_w']} "
        f"oc={desc['oc_count']} fin=0x{desc['feature_off']:x} "
        f"wt=0x{desc['weight_off']:x} out=0x{desc['output_off']:x}"
    )


def format_optional_hex(value):
    return "*" if value is None else f"0x{value:08x}"


def descriptor_match(compiler, estimated):
    if compiler is None or estimated is None:
        return False
    for key in (
        "family",
        "family_bits",
        "input_h",
        "output_h",
        "output_w",
        "oc_count",
        "feature_off",
        "weight_off",
        "output_off",
    ):
        if compiler[key] != estimated[key]:
            return False
    expected_grain = estimated.get("grain_bits")
    return expected_grain is None or compiler.get("grain_bits") == expected_grain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shape", nargs="?", choices=sorted(SHAPES))
    parser.add_argument("--compare-rknn", type=Path)
    parser.add_argument("--compiler-summary", type=Path)
    parser.add_argument("--coverage", nargs="+", type=Path)
    parser.add_argument("--grain-context", nargs="+", type=Path)
    parser.add_argument("--grain-matrix", nargs="+", type=Path)
    parser.add_argument("--grain-summary", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--mode", choices=["current", "observed"], default="current")
    args = parser.parse_args()
    if args.coverage:
        raise SystemExit(coverage(args.coverage, args.mode))
    if args.grain_summary:
        grain_summary(args.grain_summary)
        return
    if args.grain_context:
        grain_context(args.grain_context)
        return
    if args.grain_matrix:
        grain_matrix(args.grain_matrix)
        return
    if args.shape is None:
        parser.error("shape is required unless --coverage is used")
    if args.compiler_summary:
        compiler_summary(args.compiler_summary)
        return
    if args.compare_rknn:
        compare(args.shape, args.compare_rknn, args.mode)
        return
    descriptors = descriptors_for_mode(SHAPES[args.shape], args.mode)
    if args.json:
        print(json.dumps(descriptors, indent=2, sort_keys=True))
        return
    print("idx,family,family_bits,grain_bits,input_h,output_h,output_w,oc_count,feature_off,weight_off,output_off")
    for idx, desc in enumerate(descriptors, 1):
        print(
            f"{idx},{desc['family']},0x{desc['family_bits']:08x},"
            f"{format_optional_hex(desc.get('grain_bits'))},"
            f"{desc['input_h']},{desc['output_h']},{desc['output_w']},"
            f"{desc['oc_count']},0x{desc['feature_off']:x},"
            f"0x{desc['weight_off']:x},0x{desc['output_off']:x}"
        )


if __name__ == "__main__":
    main()
