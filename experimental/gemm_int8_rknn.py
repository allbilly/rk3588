import os
import sys

import ctypes
import numpy as np

from rknn_matmul import (
    MatmulSession,
    RKNN_INT8_MM_INT8_TO_INT32,
    RKNN_MEMORY_SYNC_FROM_DEVICE,
    RKNN_MEMORY_SYNC_TO_DEVICE,
    RKNN_TENSOR_INT32,
    RKNN_TENSOR_INT8,
    attr_summary,
)


def _align_up(x, align):
    return ((x + align - 1) // align) * align


def _write_tensor(mem, values):
    arr = np.asarray(values, dtype=np.asarray(values).dtype, order="C")
    ctypes.memset(int(mem.contents.virt_addr), 0, int(mem.contents.size))
    ctypes.memmove(int(mem.contents.virt_addr), arr.ctypes.data, arr.nbytes)


def _read_tensor(mem, count, dtype):
    ptr = int(mem.contents.virt_addr)
    c_type = {
        np.int8: ctypes.c_int8,
        np.int32: ctypes.c_int32,
    }[np.dtype(dtype).type]
    buf = (c_type * count).from_address(ptr)
    return np.ctypeslib.as_array(buf).copy()


def run_gemm(m, n, k, a_matrix, b_matrix, dry=False):
    align_k = _align_up(k, 32)
    align_n = _align_up(n, 32)
    session = MatmulSession(m, align_k, align_n, RKNN_INT8_MM_INT8_TO_INT32)
    try:
        a_mem = session.alloc(session.io_attr.A.size)
        b_mem = session.alloc(session.io_attr.B.size)
        c_mem = session.alloc(session.io_attr.C.size)

        a = np.zeros((m, align_k), dtype=np.int8)
        b = np.zeros((align_k, align_n), dtype=np.int8)
        a[:, :k] = np.asarray(a_matrix, dtype=np.int8, order="C").reshape(m, k)
        b[:k, :n] = np.asarray(b_matrix, dtype=np.int8, order="C").reshape(k, n)
        _write_tensor(a_mem, a)
        _write_tensor(b_mem, b)
        ctypes.memset(int(c_mem.contents.virt_addr), 0, int(c_mem.contents.size))

        session.sync(a_mem, RKNN_MEMORY_SYNC_TO_DEVICE)
        session.sync(b_mem, RKNN_MEMORY_SYNC_TO_DEVICE)
        session.sync(c_mem, RKNN_MEMORY_SYNC_TO_DEVICE)

        session.set_io_mem(a_mem, session.io_attr.A)
        session.set_io_mem(b_mem, session.io_attr.B)
        session.set_io_mem(c_mem, session.io_attr.C)

        if dry or "--dry" in sys.argv:
            print(f"\n=== GEMM INT8 {m}x{n}x{k} DRY RUN ===")
            print("  A", attr_summary(session.io_attr.A))
            print("  B", attr_summary(session.io_attr.B))
            print("  C", attr_summary(session.io_attr.C))
            return None

        session.run()
        session.sync(c_mem, RKNN_MEMORY_SYNC_FROM_DEVICE)

        raw = _read_tensor(c_mem, m * align_n, np.int32)
        out = raw.reshape(m, align_n)[:, :n].copy()
        return out
    finally:
        session.close()


if __name__ == "__main__":
    np.random.seed(42)
    cases = [
        (2, 2, 1),
        (32, 32, 32),
        (64, 99, 64),
        (64, 64, 64),
        (128, 128, 128),
    ]

    ok_all = True
    for m, n, k in cases:
        a = np.random.randint(-128, 128, size=(m, k), dtype=np.int8)
        b = np.random.randint(-128, 128, size=(k, n), dtype=np.int8)
        print(f"\n{m}x{n}x{k}:")
        result = run_gemm(m, n, k, a, b)
        if result is None:
            continue
        expected = (a.astype(np.int32) @ b.astype(np.int32)).astype(np.int32)
        ok = np.array_equal(result, expected)
        md = int(np.max(np.abs(result - expected))) if result.size else 0
        print(f"  {'PASS' if ok else 'FAIL'} (max_diff={md})")
        ok_all &= ok

    if not ok_all:
        raise SystemExit(1)
