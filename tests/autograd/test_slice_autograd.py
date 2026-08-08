import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import math

passed = 0
failed = 0

def read_grad_vals(grad, n):
    flat = grad.flatten()
    vals = []
    for i in range(n):
        v = flat.slice(start=[i], stop=[i+1]).sum_all()
        vals.append(v)
    return vals

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        print(f"FAIL: {name}")


def test_basic_1d_slice_backward():
    x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    sliced = x.slice(start=[1], stop=[4])
    s = sliced.op_sum_all()
    s.backward()

    grad = x.grad
    grad_vals = read_grad_vals(grad, 5)
    expected = [0.0, 1.0, 1.0, 1.0, 0.0]
    ok = all(abs(g - e) < 1e-10 for g, e in zip(grad_vals, expected))
    check("basic 1D slice backward [1:4] -> sum_all", ok)
    if not ok:
        print(f"  expected: {expected}")
        print(f"  got:      {grad_vals}")


def test_slice_with_op_chain():
    x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    sliced = x.slice(start=[0], stop=[3])
    squared = sliced.op_square()
    s = squared.op_sum_all()
    s.backward()

    grad = x.grad
    grad_vals = read_grad_vals(grad, 6)
    expected = [2.0, 4.0, 6.0, 0.0, 0.0, 0.0]
    ok = all(abs(g - e) < 1e-10 for g, e in zip(grad_vals, expected))
    check("slice [0:3] -> square -> sum_all backward", ok)
    if not ok:
        print(f"  expected: {expected}")
        print(f"  got:      {grad_vals}")


def test_2d_slice_dim0():
    x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float64', device='cpu')
    x2d = x.reshape([2, 3])
    x2d.requires_grad_(True)

    sliced = x2d.slice(start=[0, 0], stop=[1, 3])
    s = sliced.op_sum_all()
    s.backward()

    grad = x2d.grad
    grad_vals = read_grad_vals(grad, 6)
    expected = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    ok = all(abs(g - e) < 1e-10 for g, e in zip(grad_vals, expected))
    check("2D slice dim=0 [0:1, 0:3] -> sum_all backward", ok)
    if not ok:
        print(f"  expected: {expected}")
        print(f"  got:      {grad_vals}")


def test_multiple_slices_same_tensor():
    x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    s1 = x.slice(start=[0], stop=[3])
    s2 = x.slice(start=[3], stop=[6])
    sum1 = s1.op_sum_all()
    sum2 = s2.op_sum_all()
    total = sum1.op_add(sum2)
    total.backward()

    grad = x.grad
    grad_vals = read_grad_vals(grad, 6)
    expected = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ok = all(abs(g - e) < 1e-10 for g, e in zip(grad_vals, expected))
    check("multiple slices from same tensor (both contribute)", ok)
    if not ok:
        print(f"  expected: {expected}")
        print(f"  got:      {grad_vals}")


def test_slice_numerical_gradient():
    eps = 1e-5
    data = [1.5, 2.3, -0.7, 3.1, 0.4]

    x = rp.values(data, dtype='float64', device='cpu')
    x.requires_grad_(True)

    sliced = x.slice(start=[1], stop=[4])
    squared = sliced.op_square()
    s = squared.op_sum_all()
    s.backward()

    grad = x.grad
    grad_vals = read_grad_vals(grad, 5)

    ok = True
    for i in range(5):
        d_plus = list(data)
        d_minus = list(data)
        d_plus[i] += eps
        d_minus[i] -= eps

        def f(d):
            return sum(v * v for v in d[1:4])

        numerical = (f(d_plus) - f(d_minus)) / (2 * eps)
        if abs(grad_vals[i] - numerical) > 1e-4:
            print(f"  numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}")
            ok = False

    check("slice backward numerical gradient check", ok)


if __name__ == '__main__':
    test_basic_1d_slice_backward()
    test_slice_with_op_chain()
    test_2d_slice_dim0()
    test_multiple_slices_same_tensor()
    test_slice_numerical_gradient()

    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("All slice autograd tests passed!")
