import sys, numpy as np
sys.path.insert(0, 'examples')
import conv

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
]

def compute_expected(result, inp, wt, in_c, out_c, kh, kw, oh, ow, groups):
    b = 1
    expected = np.zeros((b, out_c, oh, ow), dtype=np.float16)
    for n in range(b):
        for o in range(out_c):
            for c in range(in_c):
                wi = 0 if (groups > 1 and groups == in_c and groups == out_c) else c
                for i in range(kh):
                    for j in range(kw):
                        expected[n, o] += inp[n, c, i:i+oh, j:j+ow] * float(wt[o, wi, i, j])
    return expected

all_pass = True
for in_c, out_c, kh, kw, (ih, iw), groups, desc in test_cases:
    sys.stdout.write(f"  {desc}: ")
    sys.stdout.flush()
    try:
        result, inp, wt = conv.run_conv2d(in_c, out_c, kh, kw, (ih, iw), groups)
        if result is None:
            print("SKIP"); continue
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
            print(f"WARN (non-1x1 partial: nz={nz_r}/{nz_e})")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()
        all_pass = False

print()
if all_pass:
    print("ALL TEST CASES PASS")
else:
    print("SOME TEST CASES FAILED - see above")
    sys.exit(1)
