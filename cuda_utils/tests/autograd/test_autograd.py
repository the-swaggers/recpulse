import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp

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


x = rp.values([1.0, 2.0, 3.0], device='cpu').requires_grad_()
y = rp.values([4.0, 5.0, 6.0], device='cpu').requires_grad_()
z = x.op_add(y)
z.backward()
x_grad = x.grad
y_grad = y.grad
check("add backward x.grad exists", x_grad is not None)
check("add backward y.grad exists", y_grad is not None)
if x_grad and y_grad:
    check("add backward x.grad = [1,1,1]",
          all(abs(g - 1.0) < 1e-6 for g in x_grad.data()))
    check("add backward y.grad = [1,1,1]",
          all(abs(g - 1.0) < 1e-6 for g in y_grad.data()))


x = rp.values([5.0, 7.0, 9.0], device='cpu').requires_grad_()
y = rp.values([2.0, 3.0, 4.0], device='cpu').requires_grad_()
z = x.op_sub(y)
z.backward()
x_grad = x.grad
y_grad = y.grad
check("sub backward x.grad exists", x_grad is not None)
check("sub backward y.grad exists", y_grad is not None)
if x_grad and y_grad:
    check("sub backward x.grad = [1,1,1]",
          all(abs(g - 1.0) < 1e-6 for g in x_grad.data()))
    check("sub backward y.grad = [-1,-1,-1]",
          all(abs(g - (-1.0)) < 1e-6 for g in y_grad.data()))
check("sub forward values",
      all(abs(a - b) < 1e-6 for a, b in zip(z.data(), [3.0, 4.0, 5.0])))


x = rp.values([2.0, 3.0, 4.0], device='cpu').requires_grad_()
y = rp.values([5.0, 6.0, 7.0], device='cpu').requires_grad_()
z = x.op_mul(y)
z.backward()
x_grad = x.grad
y_grad = y.grad
check("mul backward x.grad exists", x_grad is not None)
check("mul backward y.grad exists", y_grad is not None)
if x_grad and y_grad:
    check("mul backward x.grad = y = [5,6,7]",
          all(abs(g - e) < 1e-6 for g, e in zip(x_grad.data(), [5.0, 6.0, 7.0])))
    check("mul backward y.grad = x = [2,3,4]",
          all(abs(g - e) < 1e-6 for g, e in zip(y_grad.data(), [2.0, 3.0, 4.0])))
check("mul forward values",
      all(abs(a - b) < 1e-6 for a, b in zip(z.data(), [10.0, 18.0, 28.0])))


x = rp.values([6.0, 12.0, 20.0], device='cpu').requires_grad_()
y = rp.values([2.0, 3.0, 4.0], device='cpu').requires_grad_()
z = x.op_div(y)
z.backward()
x_grad = x.grad
y_grad = y.grad
check("div backward x.grad exists", x_grad is not None)
check("div backward y.grad exists", y_grad is not None)
if x_grad and y_grad:
    expected_gx = [1.0/2.0, 1.0/3.0, 1.0/4.0]
    expected_gy = [-6.0/4.0, -12.0/9.0, -20.0/16.0]
    check("div backward x.grad = 1/y",
          all(abs(g - e) < 1e-5 for g, e in zip(x_grad.data(), expected_gx)))
    check("div backward y.grad = -x/y^2",
          all(abs(g - e) < 1e-5 for g, e in zip(y_grad.data(), expected_gy)))
check("div forward values",
      all(abs(a - b) < 1e-6 for a, b in zip(z.data(), [3.0, 4.0, 5.0])))

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if failed > 0:
    print("SOME TESTS FAILED")
    sys.exit(1)
else:
    print("All tests passed!")
