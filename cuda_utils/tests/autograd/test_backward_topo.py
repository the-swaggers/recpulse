import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp


def read_scalar(t):
    return t.sum_all()


def read_grad_vals(grad, n):
    vals = []
    for i in range(n):
        v = grad.slice(start=[i], stop=[i+1]).sum_all()
        vals.append(v)
    return vals


def test_diamond_graph():
    x = rp.values([3.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    x_sq = x.op_square()
    x_3 = x.op_mul_scalar(3.0)
    c = x_sq.op_add(x_3)
    loss = c.op_sum_all()
    loss.backward()

    grad = x.grad
    g = read_scalar(grad)
    expected = 2.0 * 3.0 + 3.0
    assert abs(g - expected) < 1e-10, f"Expected {expected}, got {g}"
    print("PASS: diamond graph (x^2 + 3x)")


def test_deep_diamond():
    x = rp.values([2.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    x_sq = x.op_square()
    x_2 = x.op_mul_scalar(2.0)
    c = x_sq.op_mul(x_2)
    loss = c.op_sum_all()
    loss.backward()

    grad = x.grad
    g = read_scalar(grad)
    expected = 6.0 * 2.0 * 2.0
    assert abs(g - expected) < 1e-10, f"Expected {expected}, got {g}"
    print("PASS: deep diamond (x^2 * 2x)")


def test_triple_use():
    x = rp.values([2.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    a = x.op_square()
    b = x.op_mul_scalar(3.0)
    c = x.op_mul_scalar(5.0)
    ab = a.op_add(b)
    abc = ab.op_add(c)
    loss = abc.op_sum_all()
    loss.backward()

    grad = x.grad
    g = read_scalar(grad)
    expected = 2.0 * 2.0 + 3.0 + 5.0
    assert abs(g - expected) < 1e-10, f"Expected {expected}, got {g}"
    print("PASS: triple use (x^2 + 3x + 5x)")


def test_long_chain():
    x = rp.values([1.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    intermediates = []
    cur = x
    for _ in range(200):
        cur = cur.op_add_scalar(0.0)
        intermediates.append(cur)
    loss = cur.op_sum_all()
    loss.backward()

    grad = x.grad
    g = read_scalar(grad)
    assert abs(g - 1.0) < 1e-10, f"Expected 1.0, got {g}"
    print("PASS: long chain (200 nodes)")


def test_multi_input_diamond():
    x = rp.values([2.0, 3.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    a = x.op_square()
    b = x.op_mul_scalar(2.0)
    c = a.op_add(b)
    loss = c.op_sum_all()
    loss.backward()

    grad = x.grad
    vals = read_grad_vals(grad, 2)
    expected = [2.0 * 2.0 + 2.0, 2.0 * 3.0 + 2.0]
    for i in range(2):
        assert abs(vals[i] - expected[i]) < 1e-10, f"Expected grad[{i}]={expected[i]}, got {vals[i]}"
    print("PASS: multi-input diamond (x^2 + 2x, 2 elements)")


def test_zero_grad():
    x = rp.values([3.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    sq = x.op_square()
    loss = sq.op_sum_all()
    loss.backward()
    assert x.has_grad is True
    g1 = read_scalar(x.grad)
    assert abs(g1 - 6.0) < 1e-10

    x.zero_grad()
    assert x.has_grad is False

    sq2 = x.op_square()
    loss2 = sq2.op_sum_all()
    loss2.backward()
    g2 = read_scalar(x.grad)
    assert abs(g2 - 6.0) < 1e-10
    print("PASS: zero_grad")


def test_has_grad():
    x = rp.values([1.0], dtype='float64', device='cpu')
    x.requires_grad_(True)
    assert x.has_grad is False

    sq = x.op_square()
    loss = sq.op_sum_all()
    loss.backward()
    assert x.has_grad is True

    x.zero_grad()
    assert x.has_grad is False
    print("PASS: has_grad property")


def test_wide_fan_out():
    x = rp.values([2.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    a = x.op_mul_scalar(1.0)
    b = x.op_mul_scalar(2.0)
    c = x.op_mul_scalar(3.0)
    d = x.op_mul_scalar(4.0)

    ab = a.op_add(b)
    cd = c.op_add(d)
    total = ab.op_add(cd)
    loss = total.op_sum_all()
    loss.backward()

    grad = x.grad
    g = read_scalar(grad)
    expected = 1.0 + 2.0 + 3.0 + 4.0
    assert abs(g - expected) < 1e-10, f"Expected {expected}, got {g}"
    print("PASS: wide fan-out (x used in 4 ops)")


def test_gradient_accumulation_multi_path():
    x = rp.values([3.0], dtype='float64', device='cpu')
    x.requires_grad_(True)

    p1 = x.op_square()
    p2 = x.op_mul_scalar(5.0)
    p3 = x.op_mul_scalar(7.0)
    s1 = p1.op_add(p2)
    s2 = s1.op_add(p3)
    loss = s2.op_sum_all()
    loss.backward()

    grad = x.grad
    g = read_scalar(grad)
    expected = 2.0 * 3.0 + 5.0 + 7.0
    assert abs(g - expected) < 1e-10, f"Expected {expected}, got {g}"
    print("PASS: gradient accumulation from multiple paths (x^2 + 5x + 7x)")


if __name__ == '__main__':
    passed = 0
    total = 0

    tests = [
        test_diamond_graph,
        test_deep_diamond,
        test_triple_use,
        test_long_chain,
        test_multi_input_diamond,
        test_zero_grad,
        test_has_grad,
        test_wide_fan_out,
        test_gradient_accumulation_multi_path,
    ]

    for t in tests:
        total += 1
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")

    print(f"\nResults: {passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)
    print("All backward topology tests passed!")
