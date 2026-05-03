import os, sys, numpy as np

sys.path.insert(0, "examples")
import elementwise


np.random.seed(0)
submit_count = 0
_real_submit = elementwise.submit


def tracked_submit(task_obj_addr):
    global submit_count
    submit_count += 1
    return _real_submit(task_obj_addr)


elementwise.submit = tracked_submit

# Shapes copied from TestOps.test_sub in test/test_ops.py:
#   helper_test_op([(45,65), (45,65)], lambda x,y: x-y, Tensor.sub)
#   helper_test_op([(45,65), (45,65)], lambda x,y: x-y)
#   helper_test_op([(), ()], lambda x,y: x-y)
# plus scalar_sub and scalar_rsub shapes from test_ops.py.
test_cases = [
    ((45, 65), (45, 65), "45x65 Tensor.sub", lambda a, b: a - b, lambda a, b: (a, b)),
    ((45, 65), (45, 65), "45x65 operator sub", lambda a, b: a - b, lambda a, b: (a, b)),
    ((), (), "scalar sub", lambda a, b: a - b, lambda a, b: (a, b)),
    ((45, 65), None, "45x65 scalar sub", lambda a, b: a - np.float16(2), lambda a, b: (a, np.full(a.shape, 2, dtype=np.float16))),
    ((), None, "scalar scalar sub", lambda a, b: a - np.float16(2), lambda a, b: (a, np.full(a.shape, 2, dtype=np.float16))),
    ((45, 65), None, "45x65 scalar rsub", lambda a, b: np.float16(2) - a, lambda a, b: (np.full(a.shape, 2, dtype=np.float16), a)),
    ((), None, "scalar scalar rsub", lambda a, b: np.float16(2) - a, lambda a, b: (np.full(a.shape, 2, dtype=np.float16), a)),
]


def make_input(shape):
    return np.random.uniform(-2, 2, size=shape).astype(np.float16)


all_pass = True
for ashape, bshape, desc, expected_fn, npu_inputs_fn in test_cases:
    sys.stdout.write(f"  {desc}: ")
    sys.stdout.flush()
    try:
        a = make_input(ashape)
        b = make_input(bshape) if bshape is not None else None
        expected = expected_fn(a, b).astype(np.float16)
        npu_a, npu_b = npu_inputs_fn(a, b)
        a_flat = np.asarray(npu_a, dtype=np.float16).reshape(-1)
        b_flat = np.asarray(npu_b, dtype=np.float16).reshape(-1)
        n = len(a_flat)
        before_submits = submit_count
        (np.frombuffer(elementwise.output_map, dtype=np.float16, count=n))[:] = np.nan
        result = np.array(elementwise.run_op(elementwise.EW_CFG_SUB, a_flat, b_flat), dtype=np.float16).reshape(expected.shape)
        if submit_count != before_submits + 1:
            print(f"FAIL (submit_count {before_submits}->{submit_count})")
            all_pass = False
            continue
        ok = np.allclose(result, expected, atol=0.1) and not np.any(np.isinf(result))
        if ok:
            print("PASS")
        else:
            md = float(np.max(np.abs(result - expected))) if result.size else 0.0
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

os.close(elementwise.fd)
