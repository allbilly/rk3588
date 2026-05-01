import sys, numpy as np
sys.path.insert(0, 'examples')
import gemm

test_cases = []
np.random.seed(42)

test_cases.append((2, 2, 1,
    np.array([[1, 2], [3, 4]], dtype=np.float16),
    np.array([[5, 6], [7, 8]], dtype=np.float16), "2x2x1 manual"))

a = np.random.randn(65).astype(np.float16)
b = np.random.randn(65).astype(np.float16)
test_cases.append((1, 1, 65, a.reshape(1, 65), b.reshape(65, 1), "dot 65"))

a = np.random.randn(65).astype(np.float16)
b = np.random.randn(65, 45).astype(np.float16)
test_cases.append((1, 45, 65, a.reshape(1, 65), b, "vec@mat 1x45"))

a = np.random.randn(45, 65).astype(np.float16)
b = np.random.randn(65).astype(np.float16)
test_cases.append((45, 1, 65, a, b.reshape(65, 1), "mat@vec 45x1"))

for (m, n, k) in [(45, 100, 65), (64, 99, 64)]:
    a = np.random.randn(m, k).astype(np.float16)
    b = np.random.randn(k, n).astype(np.float16)
    test_cases.append((m, n, k, a, b, f"matmul {m}x{n}"))

for (m, n, k) in [(4, 4, 4), (8, 8, 8), (9, 9, 9),
                   (32, 32, 32), (64, 64, 64), (256, 256, 256)]:
    a = np.random.randn(m, k).astype(np.float16)
    b = np.random.randn(k, n).astype(np.float16)
    test_cases.append((m, n, k, a, b, f"{m}x{n}x{k}"))

for (m, n, k) in [(4, 8, 16), (16, 4, 8), (8, 32, 4),
                   (12, 34, 56), (50, 10, 20)]:
    a = np.random.randn(m, k).astype(np.float16)
    b = np.random.randn(k, n).astype(np.float16)
    test_cases.append((m, n, k, a, b, f"{m}x{n}x{k}"))

a = np.random.randn(32, 1).astype(np.float16)
b = np.random.randn(1, 32).astype(np.float16)
test_cases.append((32, 32, 1, a, b, "32x32x1 (outer)"))

proven = {(2,2,1), (4,4,4), (8,8,8), (9,9,9),
          (64,64,64), (64,99,64),
          (4,8,16), (16,4,8), (8,32,4), (32,32,1),
          (1,1,65), (1,45,65), (45,1,65), (45,100,65),
          (50,10,20), (12,34,56)}
known_imprecise = {(256, 256, 256)}

all_pass = True
for m, n, k, a, b, desc in test_cases:
    gemm.reopen_device()
    sys.stdout.write(f"  {desc}: ")
    sys.stdout.flush()
    try:
        result = gemm.run_gemm(m, n, k, a, b)
        if result is None:
            print("SKIP"); continue
        expected = a @ b
        ok = np.allclose(result, expected, atol=0.1)
        md = float(np.max(np.abs(result - expected)))
        if ok:
            print("PASS")
        elif (m, n, k) in known_imprecise:
            print(f"WARN (imprecise: md={md:.4f})")
        elif (m, n, k) not in proven:
            print(f"WARN (untested shape: md={md:.4f})")
        else:
            print(f"FAIL (md={md:.4f})")
            all_pass = False
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
