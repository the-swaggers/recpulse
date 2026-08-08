import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import math

PASS_COUNT = 0
FAIL_COUNT = 0
TOL = 1e-4


def numerical_grad(fn, x_data, eps=1e-5):
    grads = []
    for i in range(len(x_data)):
        x_plus = list(x_data)
        x_plus[i] += eps
        x_minus = list(x_data)
        x_minus[i] -= eps
        t_plus = rp.values(x_plus, dtype='float64', device='cpu')
        val_plus = fn(t_plus)
        t_minus = rp.values(x_minus, dtype='float64', device='cpu')
        val_minus = fn(t_minus)
        grads.append((val_plus - val_minus) / (2 * eps))
    return grads


def check(name, condition, msg=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {name} -- {msg}")


def test_gelu_grad():
    print("\n--- test_gelu_grad ---")
    x_data = [0.5, 1.0, -0.5, 2.0]
    x = rp.values(x_data, dtype='float64', device='cpu')
    x.requires_grad_()
    y = x.op_gelu()
    s = y.op_sum_all()
    s.backward()
    g = x.grad
    analytical = g.data()

    for i, xi in enumerate(x_data):
        sqrt_2_over_pi = 0.7978845608
        coeff = 0.044715
        coeff3 = 3.0 * coeff
        inner = sqrt_2_over_pi * (xi + coeff * xi ** 3)
        tanh_inner = math.tanh(inner)
        sech2_inner = 1.0 - tanh_inner ** 2
        d_inner = sqrt_2_over_pi * (1.0 + coeff3 * xi ** 2)
        expected = 0.5 * (1.0 + tanh_inner + xi * sech2_inner * d_inner)
        check(f"gelu analytical [{i}]",
              abs(analytical[i] - expected) < TOL,
              f"got {analytical[i]}, expected {expected}")

    def f(t):
        r = t.op_gelu()
        return r.sum_all()
    num = numerical_grad(f, x_data)
    for i in range(len(x_data)):
        check(f"gelu numerical [{i}]",
              abs(analytical[i] - num[i]) < TOL,
              f"analytical={analytical[i]}, numerical={num[i]}")


def test_silu_grad():
    print("\n--- test_silu_grad ---")
    x_data = [0.5, 1.0, -0.5, 2.0]
    x = rp.values(x_data, dtype='float64', device='cpu')
    x.requires_grad_()
    y = x.op_silu()
    s = y.op_sum_all()
    s.backward()
    g = x.grad
    analytical = g.data()

    for i, xi in enumerate(x_data):
        sig = 1.0 / (1.0 + math.exp(-xi))
        expected = sig + xi * sig * (1.0 - sig)
        check(f"silu analytical [{i}]",
              abs(analytical[i] - expected) < TOL,
              f"got {analytical[i]}, expected {expected}")

    def f(t):
        r = t.op_silu()
        return r.sum_all()
    num = numerical_grad(f, x_data)
    for i in range(len(x_data)):
        check(f"silu numerical [{i}]",
              abs(analytical[i] - num[i]) < TOL,
              f"analytical={analytical[i]}, numerical={num[i]}")


def test_rsqrt_grad():
    print("\n--- test_rsqrt_grad ---")
    x_data = [1.0, 4.0, 9.0, 16.0]
    x = rp.values(x_data, dtype='float64', device='cpu')
    x.requires_grad_()
    y = x.op_rsqrt()
    s = y.op_sum_all()
    s.backward()
    g = x.grad
    analytical = g.data()

    for i, xi in enumerate(x_data):
        expected = -0.5 * xi ** (-1.5)
        check(f"rsqrt analytical [{i}]",
              abs(analytical[i] - expected) < TOL,
              f"got {analytical[i]}, expected {expected}")

    def f(t):
        r = t.op_rsqrt()
        return r.sum_all()
    num = numerical_grad(f, x_data)
    for i in range(len(x_data)):
        check(f"rsqrt numerical [{i}]",
              abs(analytical[i] - num[i]) < TOL,
              f"analytical={analytical[i]}, numerical={num[i]}")


def test_abs_grad():
    print("\n--- test_abs_grad ---")
    x_data = [2.0, -3.0, 0.5, -1.5]
    x = rp.values(x_data, dtype='float64', device='cpu')
    x.requires_grad_()
    y = x.op_abs()
    s = y.op_sum_all()
    s.backward()
    g = x.grad
    analytical = g.data()

    for i, xi in enumerate(x_data):
        expected = 1.0 if xi > 0 else -1.0
        check(f"abs analytical [{i}]",
              abs(analytical[i] - expected) < TOL,
              f"got {analytical[i]}, expected {expected}")

    def f(t):
        r = t.op_abs()
        return r.sum_all()
    num = numerical_grad(f, x_data)
    for i in range(len(x_data)):
        check(f"abs numerical [{i}]",
              abs(analytical[i] - num[i]) < TOL,
              f"analytical={analytical[i]}, numerical={num[i]}")


def test_asin_grad():
    print("\n--- test_asin_grad ---")
    x_data = [0.0, 0.5, -0.5, 0.3]
    x = rp.values(x_data, dtype='float64', device='cpu')
    x.requires_grad_()
    y = x.op_asin()
    s = y.op_sum_all()
    s.backward()
    g = x.grad
    analytical = g.data()

    for i, xi in enumerate(x_data):
        expected = 1.0 / math.sqrt(1.0 - xi * xi)
        check(f"asin analytical [{i}]",
              abs(analytical[i] - expected) < TOL,
              f"got {analytical[i]}, expected {expected}")

    def f(t):
        r = t.op_asin()
        return r.sum_all()
    num = numerical_grad(f, x_data)
    for i in range(len(x_data)):
        check(f"asin numerical [{i}]",
              abs(analytical[i] - num[i]) < TOL,
              f"analytical={analytical[i]}, numerical={num[i]}")


def test_acos_grad():
    print("\n--- test_acos_grad ---")
    x_data = [0.0, 0.5, -0.5, 0.3]
    x = rp.values(x_data, dtype='float64', device='cpu')
    x.requires_grad_()
    y = x.op_acos()
    s = y.op_sum_all()
    s.backward()
    g = x.grad
    analytical = g.data()

    for i, xi in enumerate(x_data):
        expected = -1.0 / math.sqrt(1.0 - xi * xi)
        check(f"acos analytical [{i}]",
              abs(analytical[i] - expected) < TOL,
              f"got {analytical[i]}, expected {expected}")

    def f(t):
        r = t.op_acos()
        return r.sum_all()
    num = numerical_grad(f, x_data)
    for i in range(len(x_data)):
        check(f"acos numerical [{i}]",
              abs(analytical[i] - num[i]) < TOL,
              f"analytical={analytical[i]}, numerical={num[i]}")


def test_atan_grad():
    print("\n--- test_atan_grad ---")
    x_data = [0.0, 1.0, -1.0, 2.0]
    x = rp.values(x_data, dtype='float64', device='cpu')
    x.requires_grad_()
    y = x.op_atan()
    s = y.op_sum_all()
    s.backward()
    g = x.grad
    analytical = g.data()

    for i, xi in enumerate(x_data):
        expected = 1.0 / (1.0 + xi * xi)
        check(f"atan analytical [{i}]",
              abs(analytical[i] - expected) < TOL,
              f"got {analytical[i]}, expected {expected}")

    def f(t):
        r = t.op_atan()
        return r.sum_all()
    num = numerical_grad(f, x_data)
    for i in range(len(x_data)):
        check(f"atan numerical [{i}]",
              abs(analytical[i] - num[i]) < TOL,
              f"analytical={analytical[i]}, numerical={num[i]}")


def test_tan_grad():
    print("\n--- test_tan_grad ---")
    x_data = [0.1, 0.5, -0.3, 1.0]
    x = rp.values(x_data, dtype='float64', device='cpu')
    x.requires_grad_()
    y = x.op_tan()
    s = y.op_sum_all()
    s.backward()
    g = x.grad
    analytical = g.data()

    for i, xi in enumerate(x_data):
        expected = 1.0 / (math.cos(xi) ** 2)
        check(f"tan analytical [{i}]",
              abs(analytical[i] - expected) < TOL,
              f"got {analytical[i]}, expected {expected}")

    def f(t):
        r = t.op_tan()
        return r.sum_all()
    num = numerical_grad(f, x_data)
    for i in range(len(x_data)):
        check(f"tan numerical [{i}]",
              abs(analytical[i] - num[i]) < TOL,
              f"analytical={analytical[i]}, numerical={num[i]}")


def test_leaky_relu_grad():
    print("\n--- test_leaky_relu_grad ---")
    x_data = [2.0, -3.0, 0.5, -1.5]
    alpha = 0.01
    x = rp.values(x_data, dtype='float64', device='cpu')
    x.requires_grad_()
    y = x.op_leaky_relu()
    s = y.op_sum_all()
    s.backward()
    g = x.grad
    analytical = g.data()

    for i, xi in enumerate(x_data):
        expected = 1.0 if xi > 0 else alpha
        check(f"leaky_relu analytical [{i}]",
              abs(analytical[i] - expected) < TOL,
              f"got {analytical[i]}, expected {expected}")

    def f(t):
        r = t.op_leaky_relu()
        return r.sum_all()
    num = numerical_grad(f, x_data)
    for i in range(len(x_data)):
        check(f"leaky_relu numerical [{i}]",
              abs(analytical[i] - num[i]) < TOL,
              f"analytical={analytical[i]}, numerical={num[i]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Unary Autograd Gradient Value Tests")
    print("=" * 60)

    test_gelu_grad()
    test_silu_grad()
    test_rsqrt_grad()
    test_abs_grad()
    test_asin_grad()
    test_acos_grad()
    test_atan_grad()
    test_tan_grad()
    test_leaky_relu_grad()

    print("\n" + "=" * 60)
    total = PASS_COUNT + FAIL_COUNT
    print(f"Results: {PASS_COUNT}/{total} checks passed, {FAIL_COUNT} failed")
    if FAIL_COUNT == 0:
        print("All tests passed!")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
