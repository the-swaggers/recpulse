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


def test_same_tensor_added_to_itself():
    x = rp.values([1.0, 2.0, 3.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    y = x.op_add(x)
    s = y.op_sum_all()
    s.backward()

    grad = x.grad
    grad_vals = read_grad_vals(grad, 3)
    expected = [2.0, 2.0, 2.0]
    ok = all(abs(g - e) < 1e-10 for g, e in zip(grad_vals, expected))
    check("same tensor added to itself (x + x)", ok)
    if not ok:
        print(f"  expected: {expected}")
        print(f"  got:      {grad_vals}")


def test_same_tensor_two_different_ops():
    x = rp.values([2.0, 3.0, 4.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    a = x.op_square()
    b = x.op_mul_scalar(3.0)
    c = a.op_add(b)
    s = c.op_sum_all()
    s.backward()

    grad = x.grad
    grad_vals = read_grad_vals(grad, 3)
    expected = [7.0, 9.0, 11.0]
    ok = all(abs(g - e) < 1e-10 for g, e in zip(grad_vals, expected))
    check("same tensor in square + mul_scalar(3) branches", ok)
    if not ok:
        print(f"  expected: {expected}")
        print(f"  got:      {grad_vals}")


def test_same_tensor_used_3_times():
    x = rp.values([1.0, 2.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    a = x.op_mul_scalar(1.0)
    b = x.op_mul_scalar(2.0)
    c = x.op_mul_scalar(3.0)
    ab = a.op_add(b)
    abc = ab.op_add(c)
    s = abc.op_sum_all()
    s.backward()

    grad = x.grad
    grad_vals = read_grad_vals(grad, 2)
    expected = [6.0, 6.0]
    ok = all(abs(g - e) < 1e-10 for g, e in zip(grad_vals, expected))
    check("same tensor used 3 times (1x + 2x + 3x = 6x)", ok)
    if not ok:
        print(f"  expected: {expected}")
        print(f"  got:      {grad_vals}")


def test_tensor_both_sides_of_multiply():
    x = rp.values([2.0, 3.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    y = x.op_mul(x)
    s = y.op_sum_all()
    s.backward()

    grad = x.grad
    grad_vals = read_grad_vals(grad, 2)
    expected = [4.0, 6.0]
    ok = all(abs(g - e) < 1e-10 for g, e in zip(grad_vals, expected))
    check("tensor used in both sides of multiply (x*x = x^2, grad=2x)", ok)
    if not ok:
        print(f"  expected: {expected}")
        print(f"  got:      {grad_vals}")


def test_deep_chain_with_fan_out():
    x = rp.values([3.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    sq = x.op_square()
    lin = x.op_mul_scalar(2.0)
    s = sq.op_add(lin)
    c = s.op_add_scalar(1.0)
    loss = c.op_sum_all()
    loss.backward()

    grad = x.grad
    grad_vals = read_grad_vals(grad, 1)
    expected = [8.0]
    ok = all(abs(g - e) < 1e-10 for g, e in zip(grad_vals, expected))
    check("x^2 + 2x + 1 at x=3, grad = 2*3 + 2 = 8", ok)
    if not ok:
        print(f"  expected: {expected}")
        print(f"  got:      {grad_vals}")


def test_sigmoid_relu_same_input():
    data = [1.0, -1.0, 0.5]
    eps = 1e-5

    x = rp.values(data, dtype='float64', device='cpu')
    x.requires_grad_(True)

    a = x.op_sigmoid()
    b = x.op_relu()
    c = a.op_add(b)
    s = c.op_sum_all()
    s.backward()

    grad = x.grad
    grad_vals = read_grad_vals(grad, 3)

    ok = True
    for i in range(3):
        d_plus = list(data)
        d_minus = list(data)
        d_plus[i] += eps
        d_minus[i] -= eps

        def f(d):
            total = 0.0
            for v in d:
                sig = 1.0 / (1.0 + math.exp(-v))
                rel = max(0.0, v)
                total += sig + rel
            return total

        numerical = (f(d_plus) - f(d_minus)) / (2 * eps)
        if abs(grad_vals[i] - numerical) > 1e-4:
            print(f"  numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}")
            ok = False

    check("sigmoid + relu on same input (numerical gradient check)", ok)


if __name__ == '__main__':
    test_same_tensor_added_to_itself()
    test_same_tensor_two_different_ops()
    test_same_tensor_used_3_times()
    test_tensor_both_sides_of_multiply()
    test_deep_chain_with_fan_out()
    test_sigmoid_relu_same_input()

    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("All gradient accumulation tests passed!")
