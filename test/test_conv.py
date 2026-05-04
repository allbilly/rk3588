import sys, numpy as np
sys.path.insert(0, 'examples')
import conv

def _arg_value(name):
    prefix = name + "="
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith(prefix):
            return arg[len(prefix):]
        if arg == name and i + 2 <= len(sys.argv[1:]):
            return sys.argv[i + 2]
    return None

CASE_FILTER = _arg_value("--filter")

# Conv shapes derived from test_ops.py + NPU-specific edge cases
# Format: (in_c, out_c, kh, kw, (ih, iw), groups, desc)
# Known NPU limitations: non-1x1 kernels produce partial output (hardware limit)
test_cases = [
    # ── 1x1 kernels (fully supported via NHWC mode + channel slicing for ic>=5) ──
    (2, 2, 1, 1, (4, 4), 1, "default 1x1"),
    (1, 6, 1, 1, (4, 4), 1, "ic=1 oc=6 1x1"),
    (3, 3, 1, 1, (4, 4), 1, "ic=3 1x1"),
    (4, 2, 1, 1, (4, 4), 1, "ic=4 1x1"),
    (4, 4, 1, 1, (9, 9), 1, "1x1 9x9"),                  # test_ops test_simple_conv2d_1x1
    (8, 8, 1, 1, (5, 5), 1, "ic=8 oc=8 1x1 5x5"),         # test_ops test_biased_conv2d
    (16, 16, 1, 1, (8, 8), 1, "1x1 ic=16 oc=16 8x8"),     # test_ops test_simple_conv2d_1x1_m4-like
    (16, 16, 1, 1, (32, 32), 1, "1x1 32x32"),              # test_ops test_simple_conv2d_1x1_m4
    (10, 20, 3, 3, (9, 9), 1, "test_ops simple_conv2d_nhwc shape"),

    # ── Non-1x1 kernels (partial output — known NPU hardware limitation) ──
    (4, 4, 3, 3, (9, 9), 1, "simple 3x3 9x9"),             # test_ops test_simple_conv2d
    (16, 16, 3, 3, (9, 9), 1, "3x3 m4"),                   # test_ops test_simple_conv2d_m4
    (2, 4, 3, 3, (6, 6), 1, "3x3 oc!=ic"),
    (1, 6, 3, 3, (5, 7), 1, "ic=1 oc=6 3x3"),
    (16, 16, 3, 3, (18, 18), 1, "16x16 3x3 18x18"),

    # Various kernel sizes
    (2, 4, 2, 2, (5, 5), 1, "2x2 kernel"),
    (1, 32, 5, 5, (10, 10), 1, "5x5 kernel"),
    (8, 4, 4, 4, (10, 10), 1, "4x4 kernel"),

    # Depthwise
    (3, 3, 3, 3, (11, 28), 3, "dw 3x3 11x28"),            # test_ops test_fancy_conv2d

    # Non-square kernels
    (1, 6, 3, 1, (5, 7), 1, "ic=1 oc=6 3x1"),
    (3, 6, 1, 3, (5, 5), 1, "1x3 kernel"),

    # ── Missing shapes from test_ops.py _test_conv2d(cin=3): (1,3,5,7) @ (6,1,kh,kw) ──
    (3, 6, 3, 3, (5, 7), 3, "test_ops _test_conv2d cin=3 3x3 g=3"),
    (3, 6, 2, 1, (5, 7), 1, "test_ops _test_conv2d cin=3 2x1"),
    (3, 6, 2, 3, (5, 7), 1, "test_ops _test_conv2d cin=3 2x3"),
    (3, 6, 3, 1, (5, 7), 1, "test_ops _test_conv2d cin=3 3x1"),
    (3, 6, 3, 5, (5, 7), 1, "test_ops _test_conv2d cin=3 3x5"),

    # ── Missing from test_ops.py _test_conv2d(cin=1): (1,1,5,7) @ (6,1,kh,kw) ──
    (1, 6, 3, 3, (5, 7), 1, "test_ops _test_conv2d cin=1 3x3"),
    (1, 6, 2, 1, (5, 7), 1, "test_ops _test_conv2d cin=1 2x1"),
    (1, 6, 2, 3, (5, 7), 1, "test_ops _test_conv2d cin=1 2x3"),
    (1, 6, 3, 1, (5, 7), 1, "test_ops _test_conv2d cin=1 3x1"),
    (1, 6, 3, 5, (5, 7), 1, "test_ops _test_conv2d cin=1 3x5"),

    # ── Grouped/depthwise convs from test_ops.py ──
    (4, 2, 1, 1, (1, 1), 2, "test_ops simple_grouped_conv2d"),
    (4, 4, 1, 1, (1, 1), 2, "test_ops medium_grouped_conv2d"),
    (32, 32, 1, 1, (32, 32), 32, "test_ops depthwise_conv2d"),
    (15, 35, 3, 3, (5, 5), 5, "test_ops grouped_conv2d"),
]

# Exact shape coverage for listed slow 2D forward convs.
slow_test_cases = [
    (256, 512, 3, 3, (64, 64), 1, "test_ops test_sd_big_conv"),
    (2048, 1, 3, 3, (3, 3), 1, "test_ops test_large_ic_conv"),
    (16, 6, 5, 2, (64, 64), 1, "test_ops test_large_input_conv2d"),
]

# Listed tinygrad helper_test_op cases this harness does not cover yet because
# conv.run_conv2d currently exposes only batch=1, unbiased, unpadded,
# stride=1/dilation=1, forward 2D convolution.
def compute_expected(result, inp, wt, in_c, out_c, kh, kw, oh, ow, groups):
    b = 1
    expected = np.zeros((b, out_c, oh, ow), dtype=np.float16)
    if groups > 1:
        oc_per_group = out_c // groups
        ic_per_group = in_c // groups
        for n in range(b):
            for o in range(out_c):
                g = o // oc_per_group
                for c_local in range(ic_per_group):
                    c = g * ic_per_group + c_local
                    for i in range(kh):
                        for j in range(kw):
                            expected[n, o] += inp[n, c, i:i+oh, j:j+ow] * float(wt[o, c_local, i, j])
    else:
        for n in range(b):
            for o in range(out_c):
                for c in range(in_c):
                    for i in range(kh):
                        for j in range(kw):
                            expected[n, o] += inp[n, c, i:i+oh, j:j+ow] * float(wt[o, c, i, j])
    return expected

all_pass = True
selected = 0
for in_c, out_c, kh, kw, (ih, iw), groups, desc in test_cases + slow_test_cases:
    if CASE_FILTER and CASE_FILTER not in desc:
        continue
    selected += 1
    sys.stdout.write(f"  {desc}: ")
    sys.stdout.flush()
    try:
        result, inp, wt = conv.run_conv2d(in_c, out_c, kh, kw, (ih, iw), groups)
        if result is None:
            print("PASS (dry-run)")
            continue
        b, oc, oh, ow = result.shape
        expected = compute_expected(result, inp, wt, in_c, out_c, kh, kw, oh, ow, groups)

        correct = np.allclose(result, expected, atol=0.1) and not np.any(np.isinf(result))
        if correct:
            print("PASS")
        elif kh == 1 and kw == 1:
            md = float(np.max(np.abs(result - expected))) if not np.any(np.isinf(result)) else float('inf')
            print(f"FAIL (1x1 error: md={md:.4f})")
            all_pass = False
        else:
            nz_r = np.count_nonzero(np.abs(result) > 0.001)
            nz_e = np.count_nonzero(np.abs(expected) > 0.001)
            print(f"FAIL (non-1x1 partial: nz={nz_r}/{nz_e})")
            all_pass = False
    except Exception as e:
        print(f"FAIL (exception: {e})")
        all_pass = False

if selected == 0:
    print(f"NO TEST CASES MATCHED FILTER: {CASE_FILTER!r}")
    sys.exit(1)
elif all_pass:
    print("ALL TEST CASES PASS")
else:
    print("SOME TEST CASES FAILED - see above")
    sys.exit(1)
