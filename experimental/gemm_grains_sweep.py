import os
import time

import numpy as np

import gemm


def run_with_override(m, n, k, a, b, attr, value):
    old = getattr(gemm, attr)
    setattr(gemm, attr, value)
    try:
        t0 = time.perf_counter()
        out = gemm.run_gemm(m, n, k, a, b)
        return out, time.perf_counter() - t0
    finally:
        setattr(gemm, attr, old)


def defaults(m, n, k):
    align_in, align_out, eff_k = gemm._gemm_layout(m, n, k)
    input_row_bytes = align_in * gemm.FP16_BYTES
    even_rows_per_two_banks = (gemm._ceil_div(2 * gemm.CBUF_BANK_SIZE, input_row_bytes) + 1) & ~1
    feature_grains = max(gemm.RK_MIN_WIDE_FEATURE_GRAINS, even_rows_per_two_banks)
    line_stride = 4 if eff_k <= gemm.MIN_CHANNEL_TILE or eff_k >= gemm.RK_KN_LINE_STRIDE_START else min(gemm.RK_LINE_STRIDE_GROUP_CAP, gemm._ceil_div(eff_k, gemm.MIN_CHANNEL_TILE)) * 4
    notch_val = 8 * min(align_out // gemm.MIN_CHANNEL_TILE, gemm.RK_LINE_STRIDE_GROUP_CAP) - 1
    return feature_grains, line_stride, notch_val


def sweep_shape(m, n, k, name, attr, values, trials):
    rng = np.random.default_rng(42 + m)
    a = rng.standard_normal((m, k)).astype(np.float16)
    b = rng.standard_normal((k, n)).astype(np.float16)
    expected = a @ b
    feature_grains, line_stride, notch_val = defaults(m, n, k)

    print(f"\n{m}x{n}x{k} {name} sweep (defaults: grains={feature_grains}, line_stride={line_stride}, notch_val={notch_val})")
    print(f"{name:>12} {'avg_ms':>10} {'max_diff':>12} {'status':>8}")
    print("-" * 44)

    for value in values:
        times = []
        diffs = []
        for _ in range(trials):
            out, elapsed = run_with_override(m, n, k, a, b, attr, value)
            times.append(elapsed * 1000)
            diffs.append(float(np.max(np.abs(out - expected))))
        worst = max(diffs)
        print(f"{value:12d} {np.mean(times):10.3f} {worst:12.6f} {'PASS' if worst <= 0.1 else 'FAIL':>8}")


if __name__ == "__main__":
    grains_256 = [1, 2, 3, 4, 5, 10, 20, 40, 60, 80, 100, 128, 150, 200, 257, 300, 500]
    grains_512 = [1, 2, 5, 10, 20, 40, 80, 128, 200, 300, 500, 513, 800, 1000]
    line_stride_values = [0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 52, 56, 64]
    notch_values = [0, 1, 3, 7, 8, 15, 16, 31, 32, 47, 48, 63, 64, 79, 80, 95, 96, 103, 104, 127, 128]

    sweep_shape(256, 256, 256, "grains", "_feature_grains_override", grains_256, 3)
    sweep_shape(256, 256, 256, "line_stride", "_line_stride_override", line_stride_values, 3)
    sweep_shape(256, 256, 256, "notch_val", "_notch_val_override", notch_values, 3)
    sweep_shape(512, 512, 512, "grains", "_feature_grains_override", grains_512, 2)
    sweep_shape(512, 512, 512, "line_stride", "_line_stride_override", line_stride_values, 2)
    sweep_shape(512, 512, 512, "notch_val", "_notch_val_override", notch_values, 2)
    os.close(gemm.fd)
