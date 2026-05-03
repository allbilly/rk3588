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

# Shapes copied only from TestOps.test_add in test/test_ops.py:
#   helper_test_op([(45,68), (45,68)], lambda x,y: x+y, Tensor.add)
#   helper_test_op([(45,68), (45,68)], lambda x,y: x+y)
#   helper_test_op([(), ()], lambda x,y: x+y)
test_cases = [
    ((45, 68), (45, 68), "45x68 Tensor.add"),
    ((45, 68), (45, 68), "45x68 operator add"),
    ((), (), "scalar add"),
]


def make_input(shape):
    return np.random.uniform(-2, 2, size=shape).astype(np.float16)


all_pass = True
for ashape, bshape, desc in test_cases:
    sys.stdout.write(f"  {desc}: ")
    sys.stdout.flush()
    try:
        a = make_input(ashape)
        b = make_input(bshape)
        a_flat = np.asarray(a, dtype=np.float16).reshape(-1)
        b_flat = np.asarray(b, dtype=np.float16).reshape(-1)
        n = len(a_flat)
        before_submits = submit_count
        # Pre-fill the mapped NPU output buffer. If run_op does not submit and
        # read back hardware-written output, this sentinel survives and fails.
        (np.frombuffer(elementwise.output_map, dtype=np.float16, count=n))[:] = np.nan
        result = np.array(elementwise.run_op(elementwise.EW_CFG_ADD, a_flat, b_flat), dtype=np.float16).reshape(ashape)
        if submit_count != before_submits + 1:
            print(f"FAIL (submit_count {before_submits}->{submit_count})")
            all_pass = False
            continue
        expected = (a + b).astype(np.float16)
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
