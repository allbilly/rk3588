"""Grain bits correctness sweep.
Tests whether different feature_grains values affect output for spatial conv shapes."""
import os, sys, numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "examples/kernel_6_18")))
import conv

# Shapes that go through spatial paths
TEST_SHAPES = [
    # spatial_oc_serial (low-IC spatial, full_data_bank)
    dict(batch=1, in_c=8,  out_c=16, kh=3, kw=3, in_h=160, in_w=160, groups=1, stride=1, name="c8_h160_oc16"),
    dict(batch=1, in_c=16, out_c=128, kh=3, kw=3, in_h=160, in_w=160, groups=1, stride=1, name="c16_h160_oc128"),
    # generic_yk spatial (high-IC spatial)
    dict(batch=1, in_c=64, out_c=96, kh=3, kw=3, in_h=20, in_w=20, groups=1, stride=1, name="c64_h20_oc96"),
    # pointwise
    dict(batch=1, in_c=96, out_c=24, kh=1, kw=1, in_h=56, in_w=56, groups=1, stride=1, name="c96_h56_oc24"),
]

# Values to sweep
GRAIN_VALUES = [2, 8, 12, 15, 20, 23, 30, 52]

# Current conv.py _feature_grains baseline
BASELINE = {}

for shape in TEST_SHAPES:
    s = shape
    name = s["name"]
    BASELINE[name] = {}
    for g in GRAIN_VALUES:
        os.environ["GRAIN_OVERRIDE"] = str(g)
        try:
            r, inp, wt = conv.run_conv(s["batch"], s["in_c"], s["out_c"], s["kh"], s["kw"],
                                       (s["in_h"], s["in_w"]), groups=s["groups"], stride=s["stride"])
            e = conv.compute_expected_nchw(inp, wt, s["batch"], s["in_c"], s["in_h"], s["in_w"],
                                           s["out_c"], s["kh"], s["kw"], groups=s["groups"], stride=s["stride"])
            md = float(np.max(np.abs(r.astype(np.float64) - e)))
            ok = np.allclose(r, e, atol=0.2) and not np.any(np.isinf(r))
            BASELINE[name][g] = (md, ok)
            result = "PASS" if ok else "FAIL"
            print(f"  {name} grain={g:3d}  -> {result}  (max_diff={md:.4f})")
        except Exception as ex:
            print(f"  {name} grain={g:3d}  -> ERROR: {ex}")
    # Show variation
    vals = BASELINE[name]
    if len(set(v[0] for v in vals.values())) > 1:
        print(f"  *** {name}: grain_bits AFFECTS correctness (diff varies)")
    else:
        print(f"  {name}: grain_bits does NOT affect correctness (all diff={list(vals.values())[0][0]:.4f})")
    print()
