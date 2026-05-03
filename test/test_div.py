import os, sys, numpy as np

sys.path.insert(0, "examples")
import elementwise


np.random.seed(0)
submit_count = 0
_real_submit = elementwise.submit
EW_CFG_FDIV = elementwise._EW_BASE | (3 << 16) | (1 << 8)


def tracked_submit(task_obj_addr):
    global submit_count
    submit_count += 1
    return _real_submit(task_obj_addr)


elementwise.submit = tracked_submit

# Shapes copied from TestOps.test_div in test/test_ops.py:
#   helper_test_op([(45,65), (45,65)], lambda x,y: x/y, Tensor.div)
#   helper_test_op([(45,65), (45,65)], lambda x,y: x/y)
#   helper_test_op([(), ()], lambda x,y: x/y)
test_cases = [
    ((45, 65), (45, 65), "45x65 Tensor.div"),
    ((45, 65), (45, 65), "45x65 operator div"),
    ((), (), "scalar div"),
]


def make_input(shape, low=-2, high=2):
    return np.random.uniform(low, high, size=shape).astype(np.float16)


all_pass = True
for ashape, bshape, desc in test_cases:
    sys.stdout.write(f"  {desc}: ")
    sys.stdout.flush()
    try:
        a = make_input(ashape)
        # Keep denominators away from zero to avoid huge fp16 relative errors.
        b = make_input(bshape, low=0.5, high=2.0)
        a_flat = np.asarray(a, dtype=np.float16).reshape(-1)
        b_flat = np.asarray(b, dtype=np.float16).reshape(-1)
        n = len(a_flat)
        before_submits = submit_count
        (np.frombuffer(elementwise.output_map, dtype=np.float16, count=n))[:] = np.nan
        result = np.array(elementwise.run_op(EW_CFG_FDIV, a_flat, b_flat, fdiv_op=True), dtype=np.float16).reshape(ashape)
        if submit_count != before_submits + 1:
            print(f"FAIL (submit_count {before_submits}->{submit_count})")
            all_pass = False
            continue
        expected = (a / b).astype(np.float16)
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
