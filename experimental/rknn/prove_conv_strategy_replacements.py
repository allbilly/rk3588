#!/usr/bin/env python3
"""A/B experiments for conv_new_clean strategy replacements.

This file intentionally lives under experimental/ and imports
examples/kernel_6_18/conv_new_clean.py without editing it.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONV_PATH = ROOT / "examples" / "kernel_6_18" / "conv_new_clean.py"
NPU_LOCK = Path("/tmp/rk3588_npu_submit.lock")


LAYOUT_FALLBACK_SHAPES = [
    dict(name="b1_c32_h112_w112_oc64_wic32_k1x1_g1",
         batch=1, in_c=32, in_h=112, in_w=112, out_c=64,
         weight_in_c=32, kh=1, kw=1, groups=1),
    dict(name="b1_c64_h56_w56_oc128_wic64_k1x1_g1",
         batch=1, in_c=64, in_h=56, in_w=56, out_c=128,
         weight_in_c=64, kh=1, kw=1, groups=1),
    dict(name="b1_c256_h14_w14_oc512_wic256_k1x1_g1",
         batch=1, in_c=256, in_h=14, in_w=14, out_c=512,
         weight_in_c=256, kh=1, kw=1, groups=1),
]


@contextlib.contextmanager
def npu_lock():
    NPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with NPU_LOCK.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def load_conv_module():
    spec = importlib.util.spec_from_file_location("conv_new_clean_exp", CONV_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CONV_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shape_stride(shape):
    return shape.get("stride", 1)


def run_shape(mod, shape):
    stride = shape_stride(shape)
    result, inp, wt = mod.run_conv2d(
        shape["batch"],
        shape["in_c"],
        shape["out_c"],
        shape["kh"],
        shape["kw"],
        (shape["in_h"], shape["in_w"]),
        groups=shape["groups"],
        weight_in_c=shape["weight_in_c"],
        stride=stride,
    )
    expected = mod.compute_expected_nchw(
        inp, wt,
        shape["batch"],
        shape["in_c"],
        shape["in_h"],
        shape["in_w"],
        shape["out_c"],
        shape["kh"],
        shape["kw"],
        groups=shape["groups"],
        stride=stride,
    )
    max_diff = float(np.max(np.abs(result.astype(np.float64) - expected)))
    ok = bool(np.allclose(result, expected, atol=0.2) and not np.any(np.isinf(result)))
    return result, expected, max_diff, ok


@contextlib.contextmanager
def force_direct_pointwise(mod):
    old_oc = mod._needs_pointwise_oc_tile_schedule
    old_chain = mod._needs_pointwise_tile_schedule
    mod._needs_pointwise_oc_tile_schedule = lambda *args, **kwargs: False
    mod._needs_pointwise_tile_schedule = lambda *args, **kwargs: False
    try:
        yield
    finally:
        mod._needs_pointwise_oc_tile_schedule = old_oc
        mod._needs_pointwise_tile_schedule = old_chain


def prove_layout_replacement(mod, shapes):
    failures = []
    for shape in shapes:
        name = shape["name"]
        print(f"\n{name}")
        print("  current=pointwise_oc_h_layout_fallback")
        current, current_expected, current_md, current_ok = run_shape(mod, shape)
        print(f"    current_vs_numpy max_diff={current_md:.6f} {'PASS' if current_ok else 'FAIL'}")

        print("  candidate=direct_h_split")
        with force_direct_pointwise(mod):
            direct, direct_expected, direct_md, direct_ok = run_shape(mod, shape)
        same_expected = np.array_equal(current_expected, direct_expected)
        cross_md = float(np.max(np.abs(direct.astype(np.float64) - current.astype(np.float64))))
        cross_ok = bool(np.allclose(direct, current, atol=0.2) and not np.any(np.isinf(direct)))
        print(f"    direct_vs_numpy  max_diff={direct_md:.6f} {'PASS' if direct_ok else 'FAIL'}")
        print(f"    direct_vs_current max_diff={cross_md:.6f} {'PASS' if cross_ok else 'FAIL'}")
        print(f"    deterministic_reference {'PASS' if same_expected else 'FAIL'}")

        if not (current_ok and direct_ok and cross_ok and same_expected):
            failures.append(name)

    if failures:
        print("\nLAYOUT_REPLACEMENT FAIL")
        print("failed_shapes=" + ",".join(failures))
        return 1

    print("\nLAYOUT_REPLACEMENT PASS")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["layout"], default="layout")
    parser.add_argument("--limit", type=int, default=1,
                        help="number of shapes to run; use 0 for all selected shapes")
    args = parser.parse_args()

    with npu_lock():
        mod = load_conv_module()
        try:
            shapes = LAYOUT_FALLBACK_SHAPES if args.limit == 0 else LAYOUT_FALLBACK_SHAPES[:args.limit]
            if args.mode == "layout":
                return prove_layout_replacement(mod, shapes)
            raise AssertionError(args.mode)
        finally:
            os.close(mod.fd)


if __name__ == "__main__":
    raise SystemExit(main())
