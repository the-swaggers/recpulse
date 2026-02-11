import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import math
import numpy as np

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


x1 = rp.values([2.0], dtype='float32', device='cpu')
x2 = rp.values([3.0], dtype='float32', device='cpu')
x1.requires_grad_()
x2.requires_grad_()
out = x1.op_power(x2)
out.backward()
check("power CPU forward", abs(read_scalar(out) - 8.0) < 1e-5)
check("power CPU x1.grad exists", x1.grad is not None)
check("power CPU x2.grad exists", x2.grad is not None)
if x1.grad and x2.grad:
    gx1 = read_scalar(x1.grad)
    gx2 = read_scalar(x2.grad)
    check("power CPU grad_x1 = 3*2^2 = 12", abs(gx1 - 12.0) < 1e-4)
    check("power CPU grad_x2 = 8*ln(2)", abs(gx2 - 8.0 * math.log(2.0)) < 1e-4)


try:
    x1_gpu = rp.values([2.0], dtype='float32', device='cuda')
    x2_gpu = rp.values([3.0], dtype='float32', device='cuda')
    x1_gpu.requires_grad_()
    x2_gpu.requires_grad_()
    out_gpu = x1_gpu.op_power(x2_gpu)
    out_gpu.backward()
    check("power GPU grads computed", x1_gpu.grad is not None and x2_gpu.grad is not None)
except Exception as e:
    print(f"GPU test skipped: {e}")


x = rp.values([4.0, 9.0, 16.0], dtype='float64', device='cpu')
exp = rp.values([0.5, 0.5, 0.5], dtype='float64', device='cpu')
x.requires_grad_()
exp.requires_grad_()
out = x.op_power(exp)
s = out.op_sum_all()
s.backward()
x_grad = x.grad
check("power fractional (sqrt via x^0.5) forward",
      abs(read_scalar(out) - (2.0 + 3.0 + 4.0)) < 1e-10)
if x_grad is not None:
    gvals = list(x_grad.data())
    expected = [0.5 * 4.0**(-0.5), 0.5 * 9.0**(-0.5), 0.5 * 16.0**(-0.5)]
    check("power fractional grad_x values",
          all(abs(g - e) < 1e-10 for g, e in zip(gvals, expected)))


eps = 1e-5
x_np = np.array([2.0, 3.0, 5.0], dtype=np.float64)
e_np = np.array([3.0, 2.0, 0.5], dtype=np.float64)

x = rp.values(x_np.tolist(), dtype='float64', device='cpu')
e = rp.values(e_np.tolist(), dtype='float64', device='cpu')
x.requires_grad_()
out = x.op_power(e)
s = out.op_sum_all()
s.backward()
grad_x = x.grad
gvals = list(grad_x.data())
for i in range(3):
    x_plus = x_np.copy(); x_plus[i] += eps
    x_minus = x_np.copy(); x_minus[i] -= eps
    numerical = ((x_plus ** e_np).sum() - (x_minus ** e_np).sum()) / (2 * eps)
    check(f"power numerical gradient x[{i}]", abs(gvals[i] - numerical) < 1e-4)


x = rp.values([2.0, 3.0, 4.0, 5.0], dtype='float64', device='cpu')
e = rp.values([2.0, 3.0, 0.5, 1.0], dtype='float64', device='cpu')
x.requires_grad_()
out = x.op_power(e)
s = out.op_sum_all()
s.backward()
g = x.grad
check("power multi-element grad exists", g is not None)
if g is not None:
    gv = list(g.data())
    expected = [
        2.0 * 2.0**(2.0 - 1.0),
        3.0 * 3.0**(3.0 - 1.0),
        0.5 * 4.0**(0.5 - 1.0),
        1.0 * 5.0**(1.0 - 1.0),
    ]
    check("power multi-element grad values",
          all(abs(gv[i] - expected[i]) < 1e-8 for i in range(4)))

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if failed > 0:
    print("SOME TESTS FAILED")
    sys.exit(1)
else:
    print("All tests passed!")
