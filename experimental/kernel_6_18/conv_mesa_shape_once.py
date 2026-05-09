import argparse
import importlib.util
import os
import sys

import numpy as np


def expected_conv(inp, wt, batch, in_c, in_h, in_w, out_c, kh, kw, groups):
    out_h = in_h - kh + 1
    out_w = in_w - kw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("invalid_output")

    exp = np.zeros((batch, out_c, out_h, out_w), dtype=np.float64)
    for n in range(batch):
        for group in range(groups):
            oc0 = group * out_c // groups
            oc1 = (group + 1) * out_c // groups
            ic0 = group * in_c // groups
            ic1 = (group + 1) * in_c // groups
            for oc in range(oc0, oc1):
                for ic in range(ic0, ic1):
                    for i in range(kh):
                        for j in range(kw):
                            exp[n, oc] += (
                                inp[n, ic, i:i + out_h, j:j + out_w].astype(np.float64)
                                * wt[oc, ic - ic0, i, j].astype(np.float64)
                            )
    return exp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-id", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--in-c", type=int, required=True)
    parser.add_argument("--in-h", type=int, required=True)
    parser.add_argument("--in-w", type=int, required=True)
    parser.add_argument("--out-c", type=int, required=True)
    parser.add_argument("--weight-in-c", type=int, required=True)
    parser.add_argument("--kh", type=int, required=True)
    parser.add_argument("--kw", type=int, required=True)
    parser.add_argument("--groups", type=int, required=True)
    args = parser.parse_args()

    if args.batch != 1:
        print("UNSUPPORTED batch")
        return 0
    if args.groups <= 0 or args.in_c % args.groups or args.out_c % args.groups:
        print("UNSUPPORTED groups")
        return 0
    if args.weight_in_c != args.in_c // args.groups:
        print("UNSUPPORTED weight_in_c")
        return 0
    if args.in_h - args.kh + 1 <= 0 or args.in_w - args.kw + 1 <= 0:
        print("UNSUPPORTED invalid_output")
        return 0

    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, script_dir)
    sys.argv = ["conv_mesa_shape_once", "--submit"]

    spec = importlib.util.spec_from_file_location(
        "conv_mesa", os.path.join(script_dir, "conv_mesa.py")
    )
    conv_mesa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv_mesa)

    result, inp, wt = conv_mesa.run_conv2d(
        args.in_c,
        args.out_c,
        args.kh,
        args.kw,
        (args.in_h, args.in_w),
        groups=args.groups,
    )
    exp = expected_conv(
        inp,
        wt,
        args.batch,
        args.in_c,
        args.in_h,
        args.in_w,
        args.out_c,
        args.kh,
        args.kw,
        args.groups,
    )
    max_diff = float(np.max(np.abs(result.astype(np.float64) - exp)))
    ok = np.allclose(result, exp, atol=0.1) and not np.any(np.isinf(result))
    print(("PASS" if ok else "FAIL") + f" max_diff={max_diff:.4f}")

    try:
        os.close(conv_mesa.fd)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
