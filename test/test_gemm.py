import sys

import numpy as np


sys.path.insert(0, "examples")
sys.path.insert(0, "experimental")

import gemm
import gemm_int8
import gemm_int16


def _fp16_cases():
    rng = np.random.default_rng(42)
    cases = [
        (
            2,
            2,
            1,
            np.array([[1], [3]], dtype=np.float16),
            np.array([[5, 6]], dtype=np.float16),
            "2x2x1 manual",
        )
    ]

    a = rng.standard_normal(65).astype(np.float16)
    b = rng.standard_normal(65).astype(np.float16)
    cases.append((1, 1, 65, a.reshape(1, 65), b.reshape(65, 1), "dot 65"))

    a = rng.standard_normal(65).astype(np.float16)
    b = rng.standard_normal((65, 45)).astype(np.float16)
    cases.append((1, 45, 65, a.reshape(1, 65), b, "vec@mat 1x45"))

    a = rng.standard_normal((45, 65)).astype(np.float16)
    b = rng.standard_normal(65).astype(np.float16)
    cases.append((45, 1, 65, a, b.reshape(65, 1), "mat@vec 45x1"))

    shapes = [
        (45, 100, 65),
        (64, 99, 64),
        (1, 4, 4),
        (1, 99, 64),
        (4, 4, 4),
        (8, 8, 8),
        (9, 9, 9),
        (32, 32, 32),
        (64, 64, 64),
        (96, 96, 96),
        (128, 128, 128),
        (192, 192, 192),
        (256, 256, 256),
        (4, 8, 16),
        (16, 4, 8),
        (8, 32, 4),
        (12, 34, 56),
        (50, 10, 20),
    ]
    for m, n, k in shapes:
        a = rng.standard_normal((m, k)).astype(np.float16)
        b = rng.standard_normal((k, n)).astype(np.float16)
        cases.append((m, n, k, a, b, f"{m}x{n}x{k}"))

    a = rng.standard_normal((32, 1)).astype(np.float16)
    b = rng.standard_normal((1, 32)).astype(np.float16)
    cases.append((32, 32, 1, a, b, "32x32x1 outer"))
    return cases


def _int_cases(dtype, low, high, shapes, seed_base):
    cases = [
        (
            2,
            2,
            1,
            np.array([[1], [3]], dtype=dtype),
            np.array([[5, 6]], dtype=dtype),
            "2x2x1 manual",
        )
    ]
    for idx, (m, n, k) in enumerate(shapes):
        rng = np.random.default_rng(seed_base + idx)
        a = rng.integers(low, high, size=(m, k), dtype=dtype)
        b = rng.integers(low, high, size=(k, n), dtype=dtype)
        cases.append((m, n, k, a, b, f"{m}x{n}x{k}"))
    return cases


def _run_fp16_case(m, n, k, a, b, desc):
    result = gemm.run_gemm(m, n, k, a, b)
    assert result is not None, desc
    expected = a @ b
    assert result.shape == (m, n), desc
    np.testing.assert_allclose(result, expected, atol=0.1, err_msg=desc)


def _run_int_case(module, m, n, k, a, b, desc):
    result = module.run_gemm(m, n, k, a, b)
    expected = a.astype(np.int32) @ b.astype(np.int32)
    assert result.dtype == np.int32, desc
    assert result.shape == (m, n), desc
    np.testing.assert_array_equal(result, expected, err_msg=desc)


def test_gemm_fp16():
    for case in _fp16_cases():
        _run_fp16_case(*case)


def test_gemm_int8():
    shapes = [(32, 32, 32), (64, 99, 64), (64, 64, 64), (128, 128, 128)]
    for case in _int_cases(np.int8, -8, 8, shapes, 100):
        _run_int_case(gemm_int8, *case)


def test_gemm_int16():
    shapes = [(32, 32, 32), (64, 64, 64), (96, 96, 96), (64, 99, 64), (128, 128, 128)]
    for case in _int_cases(np.int16, -16, 16, shapes, 200):
        _run_int_case(gemm_int16, *case)


if __name__ == "__main__":
    failed = False
    for name, fn in (
        ("gemm.py fp16", test_gemm_fp16),
        ("gemm_int8.py int8", test_gemm_int8),
        ("gemm_int16.py int16", test_gemm_int16),
    ):
        print(f"{name}: ", end="", flush=True)
        try:
            fn()
            print("PASS")
        except Exception as exc:
            failed = True
            print(f"FAIL ({exc})")
    if failed:
        raise SystemExit(1)
