#!/usr/bin/env python3
"""Offline planner checks — no NPU. Run: python3 conv_grok/test_planner.py"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from conv_grok.conv import (  # noqa: E402
    _windows_from_step,
    make_shape,
    plan_depthwise_rows,
    plan_local_serial_rows,
    plan_shape,
    shape_from_name,
)


def _assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "assertion failed")


def test_windows_merge_tail():
    wins = _windows_from_step(28, 22)
    _assert(wins == [(0, 22), (22, 6)], f"tail merge: {wins}")
    _assert(_windows_from_step(10, 10) == [(0, 10)])
    # step < min_tail: must not emit n<=0 (was infinite growth / OOM on c320 y_step=5)
    w5 = _windows_from_step(28, 5)
    _assert(all(n > 0 for _, n in w5) and sum(n for _, n in w5) == 28, f"step5: {w5}")
    _assert(len(w5) <= 28)


def test_make_shape_arbitrary():
    s = make_shape(in_c=48, in_h=37, in_w=41, out_c=96, kh=3, kw=5, name="adhoc")
    _assert(s["in_c"] == 48 and s["in_h"] == 37 and s["kw"] == 5)
    rows, p, y_step, k_step = plan_local_serial_rows(s)
    _assert(p["out_h"] == 35 and p["out_w"] == 37)
    _assert(len(rows) >= 1 and rows[0]["split_method"] in ("NONE", "BY_Y", "BY_K", "BY_YK"))


def test_pointwise_h28_mesa_not_hardcoded():
    s = shape_from_name("b1_c256_h28_w28_oc512_wic256_k1x1_g1_s1_pvalid")
    rows, _, y_step, k_step = plan_local_serial_rows(s)
    _assert(1 < y_step < 28, f"h28 y_step={y_step}")
    _assert(k_step == 32 and rows[0]["split_method"] == "BY_YK")


def test_pointwise_large_ic_tiles():
    s = shape_from_name("b1_c1280_h7_w7_oc1280_wic1280_k1x1_g1_s1_pvalid")
    rows, _, _, k_step = plan_local_serial_rows(s)
    _assert(k_step in (16, 32, 64, 128), f"c1280 k_step={k_step}")
    _assert(len(rows) <= 200, f"c1280 rows={len(rows)}")


def test_rgb_first_layer_nhwc():
    s = shape_from_name("b1_c3_h224_w224_oc32_wic3_k3x3_g1_s1_pvalid")
    rows, _, y_step, _ = plan_local_serial_rows(s)
    _assert(8 <= y_step <= 48, f"rgb y_step={y_step}")
    _assert(rows[0]["split_method"] == "BY_Y" and len(rows) <= 12)


def test_nhwc_only_for_c1_to_c4():
    from conv_grok.conv import _conv_params, make_shape, should_use_nhwc_pack
    _assert(should_use_nhwc_pack(3, 8) and should_use_nhwc_pack(4, 8))
    _assert(not should_use_nhwc_pack(5, 8) and not should_use_nhwc_pack(7, 8))
    for ic, want in ((3, True), (4, True), (5, False), (7, False), (8, False)):
        p = _conv_params(make_shape(in_c=ic, in_h=32, in_w=32, out_c=16, kh=3, kw=3))
        _assert(p["use_nhwc"] is want, f"ic={ic} use_nhwc={p['use_nhwc']} want={want}")


def test_stride_hw_limit():
    from conv_grok.conv import MAX_CONV_STRIDE, make_shape
    _assert(MAX_CONV_STRIDE == 7)
    try:
        make_shape(in_c=32, in_h=54, in_w=54, out_c=64, kh=3, kw=3, stride=8, pvalid=True)
        _assert(False, "stride=8 should raise")
    except ValueError as e:
        _assert("stride" in str(e).lower())
    s = make_shape(in_c=32, in_h=54, in_w=54, out_c=64, kh=3, kw=3, stride=7, pvalid=True)
    _assert(s["stride"] == 7)


def test_depthwise_batched_tasks():
    s = shape_from_name("b1_c768_h20_w20_oc768_wic1_k5x5_g768_s1_pvalid")
    rows, _, y_step, c_step = plan_depthwise_rows(s)
    _assert(c_step == 32)
    est = len(rows)
    _assert(est < 500, f"c768_k5x5_dw est_tasks={est} (was ~36k serial)")
    _assert(est == _ceil_div(768, 32) * len(_windows_from_step(
        (s["in_h"] - s["kh"]) // s.get("stride", 1) + 1, y_step)), f"rows={est}")


def _ceil_div(a, b):
    return (a + b - 1) // b


def test_geometry_only_planner():
    s = make_shape(in_c=17, in_h=83, in_w=91, out_c=29, kh=3, kw=5)
    p = plan_shape(s)
    _assert(p["split"] in ("NONE", "BY_Y", "BY_K", "BY_YK", "depthwise"))
    _assert(p["y_step"] >= 1 and p["k_step"] >= 1)


def test_regression_sample_splits():
    samples = {
        "conv2d_b1_c4_h9_w9_oc4_wic4_k3x3_g1": "NONE",
        "b1_c128_h3_w3_oc128_wic128_k3x3_g1_s1_pvalid": "BY_K",
        "b1_c768_h20_w20_oc768_wic1_k5x5_g768_s1_pvalid": "depthwise",
    }
    for name, expect in samples.items():
        s = shape_from_name(name)
        if expect == "depthwise":
            rows, _, _, _ = plan_depthwise_rows(s)
            _assert(len(rows) < 500, f"{name} dw rows={len(rows)}")
        else:
            rows, _, _, _ = plan_local_serial_rows(s)
            _assert(rows[0]["split_method"] == expect, f"{name} got {rows[0]['split_method']}")


def main():
    tests = [
        test_windows_merge_tail,
        test_make_shape_arbitrary,
        test_pointwise_h28_mesa_not_hardcoded,
        test_pointwise_large_ic_tiles,
        test_rgb_first_layer_nhwc,
        test_nhwc_only_for_c1_to_c4,
        test_stride_hw_limit,
        test_depthwise_batched_tasks,
        test_geometry_only_planner,
        test_regression_sample_splits,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"ALL {len(tests)} planner tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
