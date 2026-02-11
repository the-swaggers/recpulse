import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import math

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        print(f"FAIL: {name}")

def read_scalar(t):
    return t.sum_all()


x1 = rp.values([2.0, 3.0], dtype='float32', device='cpu')
x2 = rp.values([2.0, 3.0], dtype='float32', device='cpu')
x1.requires_grad_()
x2.requires_grad_()
out = x1.op_power(x2)
out.backward()
check("power simple CPU x1.grad exists", x1.grad is not None)
check("power simple CPU x2.grad exists", x2.grad is not None)


try:
    x1_gpu = rp.values([2.0, 3.0], dtype='float32', device='cuda')
    x2_gpu = rp.values([2.0, 3.0], dtype='float32', device='cuda')
    x1_gpu.requires_grad_()
    x2_gpu.requires_grad_()
    out_gpu = x1_gpu.op_power(x2_gpu)
    out_gpu.backward()
    check("power simple GPU grads computed",
          x1_gpu.grad is not None and x2_gpu.grad is not None)
except Exception as e:
    print(f"GPU test skipped: {e}")


x1 = rp.values([2.0, 3.0, 4.0, 5.0], dtype='float64', device='cpu')
x2 = rp.values([3.0, 2.0, 0.5, 1.0], dtype='float64', device='cpu')
x1.requires_grad_()
x2.requires_grad_()
out = x1.op_power(x2)
out.backward()
g1 = x1.grad
g2 = x2.grad
check("power 4-element x1.grad exists", g1 is not None)
check("power 4-element x2.grad exists", g2 is not None)
if g1 is not None:
    gv1 = list(g1.data())
    expected_g1 = [
        3.0 * 2.0**2.0,
        2.0 * 3.0**1.0,
        0.5 * 4.0**(-0.5),
        1.0 * 5.0**0.0,
    ]
    check("power 4-element x1.grad values",
          all(abs(a - b) < 1e-8 for a, b in zip(gv1, expected_g1)))
if g2 is not None:
    gv2 = list(g2.data())
    expected_g2 = [
        (2.0**3.0) * math.log(2.0),
        (3.0**2.0) * math.log(3.0),
        (4.0**0.5) * math.log(4.0),
        (5.0**1.0) * math.log(5.0),
    ]
    check("power 4-element x2.grad values",
          all(abs(a - b) < 1e-8 for a, b in zip(gv2, expected_g2)))


x1 = rp.values([2.0, 3.0, 4.0], dtype='float64', device='cpu')
x2 = rp.values([3.0, 2.0, 1.5], dtype='float64', device='cpu')
x1.requires_grad_()
out = x1.op_power(x2)
s = out.op_sum_all()
s.backward()
g = x1.grad
check("power numerical check grad exists", g is not None)
if g is not None:
    import numpy as np
    x_np = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    e_np = np.array([3.0, 2.0, 1.5], dtype=np.float64)
    eps = 1e-5
    gv = list(g.data())
    for i in range(3):
        xp = x_np.copy(); xp[i] += eps
        xm = x_np.copy(); xm[i] -= eps
        numerical = ((xp ** e_np).sum() - (xm ** e_np).sum()) / (2 * eps)
        check(f"power numerical check x1[{i}]", abs(gv[i] - numerical) < 1e-4)

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if failed > 0:
    print("SOME TESTS FAILED")
    sys.exit(1)
else:
    print("All tests passed!")
