import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import math

PASS_COUNT = 0
FAIL_COUNT = 0
TOL = 1e-4


def numerical_grad(make_loss, x_data, eps=1e-5):
    grads = []
    for i in range(len(x_data)):
        x_plus = list(x_data)
        x_plus[i] += eps
        x_minus = list(x_data)
        x_minus[i] -= eps
        loss_plus = make_loss(x_plus)
        loss_minus = make_loss(x_minus)
        grads.append((loss_plus - loss_minus) / (2 * eps))
    return grads


def read_grad_vals(grad, n):
    flat = grad.flatten()
    vals = []
    for i in range(n):
        v = flat.slice(start=[i], stop=[i+1]).sum_all()
        vals.append(v)
    return vals


def check(name, condition, msg=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {name} -- {msg}")


def test_linear_layer():
    print("\n--- test_linear_layer: y = x @ W + b ---")

    x_data = [1.0, 2.0]
    w_data = [0.5, -0.3, 0.2, 0.7]
    b_data = [0.1, -0.1]

    x_1d = rp.values(x_data, dtype='float64', device='cpu')
    x = x_1d.reshape([1, 2])
    x.requires_grad_()
    w_1d = rp.values(w_data, dtype='float64', device='cpu')
    W = w_1d.reshape([2, 2])
    W.requires_grad_()
    b_1d = rp.values(b_data, dtype='float64', device='cpu')
    b = b_1d.reshape([1, 2])
    b.requires_grad_()

    out = x.op_matmul(W)
    out_b = out.op_add(b)
    loss = out_b.op_sum_all()
    loss.backward()

    def make_loss_x(xd):
        xt_1d = rp.values(xd, dtype='float64', device='cpu')
        xt = xt_1d.reshape([1, 2])
        wt_1d = rp.values(w_data, dtype='float64', device='cpu')
        wt = wt_1d.reshape([2, 2])
        bt_1d = rp.values(b_data, dtype='float64', device='cpu')
        bt = bt_1d.reshape([1, 2])
        r1 = xt.op_matmul(wt)
        r2 = r1.op_add(bt)
        r3 = r2.op_sum_all()
        return r3.sum_all()

    num_x = numerical_grad(make_loss_x, x_data)
    grad_x = x.grad
    grad_x_vals = read_grad_vals(grad_x, 2)
    for i in range(2):
        check(f"x.grad[{i}]", abs(grad_x_vals[i] - num_x[i]) < TOL,
              f"autograd={grad_x_vals[i]}, numerical={num_x[i]}")

    def make_loss_w(wd):
        xt_1d = rp.values(x_data, dtype='float64', device='cpu')
        xt = xt_1d.reshape([1, 2])
        wt_1d = rp.values(wd, dtype='float64', device='cpu')
        wt = wt_1d.reshape([2, 2])
        bt_1d = rp.values(b_data, dtype='float64', device='cpu')
        bt = bt_1d.reshape([1, 2])
        r1 = xt.op_matmul(wt)
        r2 = r1.op_add(bt)
        r3 = r2.op_sum_all()
        return r3.sum_all()

    num_w = numerical_grad(make_loss_w, w_data)
    grad_w = W.grad
    grad_w_vals = read_grad_vals(grad_w, 4)
    for i in range(4):
        check(f"W.grad[{i}]", abs(grad_w_vals[i] - num_w[i]) < TOL,
              f"autograd={grad_w_vals[i]}, numerical={num_w[i]}")

    def make_loss_b(bd):
        xt_1d = rp.values(x_data, dtype='float64', device='cpu')
        xt = xt_1d.reshape([1, 2])
        wt_1d = rp.values(w_data, dtype='float64', device='cpu')
        wt = wt_1d.reshape([2, 2])
        bt_1d = rp.values(bd, dtype='float64', device='cpu')
        bt = bt_1d.reshape([1, 2])
        r1 = xt.op_matmul(wt)
        r2 = r1.op_add(bt)
        r3 = r2.op_sum_all()
        return r3.sum_all()

    num_b = numerical_grad(make_loss_b, b_data)
    grad_b = b.grad
    grad_b_vals = read_grad_vals(grad_b, 2)
    for i in range(2):
        check(f"b.grad[{i}]", abs(grad_b_vals[i] - num_b[i]) < TOL,
              f"autograd={grad_b_vals[i]}, numerical={num_b[i]}")


def test_matmul_relu_sum():
    print("\n--- test_matmul_relu_sum: loss = sum(relu(x @ W)) ---")

    x_data = [1.0, -1.0, 0.5, 2.0]
    w_data = [0.3, -0.5, 0.8, 0.1]

    x_1d = rp.values(x_data, dtype='float64', device='cpu')
    x = x_1d.reshape([2, 2])
    x.requires_grad_()
    w_1d = rp.values(w_data, dtype='float64', device='cpu')
    W = w_1d.reshape([2, 2])
    W.requires_grad_()

    h = x.op_matmul(W)
    a = h.op_relu()
    loss = a.op_sum_all()
    loss.backward()

    def make_loss_x(xd):
        xt_1d = rp.values(xd, dtype='float64', device='cpu')
        xt = xt_1d.reshape([2, 2])
        wt_1d = rp.values(w_data, dtype='float64', device='cpu')
        wt = wt_1d.reshape([2, 2])
        r1 = xt.op_matmul(wt)
        r2 = r1.op_relu()
        r3 = r2.op_sum_all()
        return r3.sum_all()

    num_x = numerical_grad(make_loss_x, x_data)
    grad_x = x.grad
    grad_x_vals = read_grad_vals(grad_x, 4)
    for i in range(4):
        check(f"x.grad[{i}]", abs(grad_x_vals[i] - num_x[i]) < TOL,
              f"autograd={grad_x_vals[i]}, numerical={num_x[i]}")

    def make_loss_w(wd):
        xt_1d = rp.values(x_data, dtype='float64', device='cpu')
        xt = xt_1d.reshape([2, 2])
        wt_1d = rp.values(wd, dtype='float64', device='cpu')
        wt = wt_1d.reshape([2, 2])
        r1 = xt.op_matmul(wt)
        r2 = r1.op_relu()
        r3 = r2.op_sum_all()
        return r3.sum_all()

    num_w = numerical_grad(make_loss_w, w_data)
    grad_w = W.grad
    grad_w_vals = read_grad_vals(grad_w, 4)
    for i in range(4):
        check(f"W.grad[{i}]", abs(grad_w_vals[i] - num_w[i]) < TOL,
              f"autograd={grad_w_vals[i]}, numerical={num_w[i]}")


def test_sigmoid_cross_entropy():
    print("\n--- test_sigmoid_cross_entropy: loss = -sum(y * log(sigmoid(x))) ---")

    x_data = [1.0, -0.5, 2.0]
    y_data = [1.0, 0.0, 1.0]

    x = rp.values(x_data, dtype='float64', device='cpu')
    x.requires_grad_()
    y_t = rp.values(y_data, dtype='float64', device='cpu')

    sig = x.op_sigmoid()
    lg = sig.op_log()
    prod = lg.op_mul(y_t)
    s = prod.op_sum_all()
    neg = s.op_mul_scalar(-1.0)
    neg.backward()

    def make_loss_x(xd):
        xt = rp.values(xd, dtype='float64', device='cpu')
        yt = rp.values(y_data, dtype='float64', device='cpu')
        r1 = xt.op_sigmoid()
        r2 = r1.op_log()
        r3 = r2.op_mul(yt)
        r4 = r3.op_sum_all()
        r5 = r4.op_mul_scalar(-1.0)
        return r5.sum_all()

    num_x = numerical_grad(make_loss_x, x_data)
    grad_x = x.grad
    grad_x_vals = grad_x.data()
    for i in range(3):
        check(f"x.grad[{i}]", abs(grad_x_vals[i] - num_x[i]) < TOL,
              f"autograd={grad_x_vals[i]}, numerical={num_x[i]}")


def test_quadratic_form():
    print("\n--- test_quadratic_form: loss = x^T A x ---")

    x_data = [1.0, 2.0]
    a_data = [2.0, 1.0, 1.0, 3.0]

    x_1d = rp.values(x_data, dtype='float64', device='cpu')
    x = x_1d.reshape([2, 1])
    x.requires_grad_()
    a_1d = rp.values(a_data, dtype='float64', device='cpu')
    A = a_1d.reshape([2, 2])

    Ax = A.op_matmul(x)
    prod = x.op_mul(Ax)
    loss = prod.op_sum_all()
    loss.backward()

    def make_loss_x(xd):
        xt_1d = rp.values(xd, dtype='float64', device='cpu')
        xt = xt_1d.reshape([2, 1])
        at_1d = rp.values(a_data, dtype='float64', device='cpu')
        at = at_1d.reshape([2, 2])
        r1 = at.op_matmul(xt)
        r2 = xt.op_mul(r1)
        r3 = r2.op_sum_all()
        return r3.sum_all()

    num_x = numerical_grad(make_loss_x, x_data)
    grad_x = x.grad
    grad_x_vals = read_grad_vals(grad_x, 2)
    for i in range(2):
        check(f"x.grad[{i}]", abs(grad_x_vals[i] - num_x[i]) < TOL,
              f"autograd={grad_x_vals[i]}, numerical={num_x[i]}")


def test_exp_sum():
    print("\n--- test_exp_sum: loss = sum(exp(x)) ---")

    x_data = [1.0, 2.0, 3.0]

    x = rp.values(x_data, dtype='float64', device='cpu')
    x.requires_grad_()

    e = x.op_exp()
    loss = e.op_sum_all()
    loss.backward()

    grad_x = x.grad
    grad_x_vals = grad_x.data()

    for i in range(3):
        expected = math.exp(x_data[i])
        check(f"x.grad[{i}] analytical", abs(grad_x_vals[i] - expected) < TOL,
              f"autograd={grad_x_vals[i]}, expected={expected}")

    def make_loss_x(xd):
        xt = rp.values(xd, dtype='float64', device='cpu')
        r1 = xt.op_exp()
        r2 = r1.op_sum_all()
        return r2.sum_all()

    num_x = numerical_grad(make_loss_x, x_data)
    for i in range(3):
        check(f"x.grad[{i}] numerical", abs(grad_x_vals[i] - num_x[i]) < TOL,
              f"autograd={grad_x_vals[i]}, numerical={num_x[i]}")


def test_multi_layer():
    print("\n--- test_multi_layer: loss = sum(tanh(relu(x * w1) * w2)) ---")

    x_data = [0.5, 1.5, -0.5]
    w1_data = [2.0, -1.0, 0.5]
    w2_data = [1.0, 0.5, -1.0]

    x = rp.values(x_data, dtype='float64', device='cpu')
    x.requires_grad_()
    w1 = rp.values(w1_data, dtype='float64', device='cpu')
    w1.requires_grad_()
    w2 = rp.values(w2_data, dtype='float64', device='cpu')
    w2.requires_grad_()

    h1 = x.op_mul(w1)
    a1 = h1.op_relu()
    h2 = a1.op_mul(w2)
    out = h2.op_tanh()
    loss = out.op_sum_all()
    loss.backward()

    def make_loss_x(xd):
        xt = rp.values(xd, dtype='float64', device='cpu')
        w1t = rp.values(w1_data, dtype='float64', device='cpu')
        w2t = rp.values(w2_data, dtype='float64', device='cpu')
        r1 = xt.op_mul(w1t)
        r2 = r1.op_relu()
        r3 = r2.op_mul(w2t)
        r4 = r3.op_tanh()
        r5 = r4.op_sum_all()
        return r5.sum_all()

    num_x = numerical_grad(make_loss_x, x_data)
    grad_x = x.grad
    grad_x_vals = grad_x.data()
    for i in range(3):
        check(f"x.grad[{i}]", abs(grad_x_vals[i] - num_x[i]) < TOL,
              f"autograd={grad_x_vals[i]}, numerical={num_x[i]}")

    def make_loss_w1(w1d):
        xt = rp.values(x_data, dtype='float64', device='cpu')
        w1t = rp.values(w1d, dtype='float64', device='cpu')
        w2t = rp.values(w2_data, dtype='float64', device='cpu')
        r1 = xt.op_mul(w1t)
        r2 = r1.op_relu()
        r3 = r2.op_mul(w2t)
        r4 = r3.op_tanh()
        r5 = r4.op_sum_all()
        return r5.sum_all()

    num_w1 = numerical_grad(make_loss_w1, w1_data)
    grad_w1 = w1.grad
    grad_w1_vals = grad_w1.data()
    for i in range(3):
        check(f"w1.grad[{i}]", abs(grad_w1_vals[i] - num_w1[i]) < TOL,
              f"autograd={grad_w1_vals[i]}, numerical={num_w1[i]}")

    def make_loss_w2(w2d):
        xt = rp.values(x_data, dtype='float64', device='cpu')
        w1t = rp.values(w1_data, dtype='float64', device='cpu')
        w2t = rp.values(w2d, dtype='float64', device='cpu')
        r1 = xt.op_mul(w1t)
        r2 = r1.op_relu()
        r3 = r2.op_mul(w2t)
        r4 = r3.op_tanh()
        r5 = r4.op_sum_all()
        return r5.sum_all()

    num_w2 = numerical_grad(make_loss_w2, w2_data)
    grad_w2 = w2.grad
    grad_w2_vals = grad_w2.data()
    for i in range(3):
        check(f"w2.grad[{i}]", abs(grad_w2_vals[i] - num_w2[i]) < TOL,
              f"autograd={grad_w2_vals[i]}, numerical={num_w2[i]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Complex Autograd Chain Tests")
    print("=" * 60)

    test_linear_layer()
    test_matmul_relu_sum()
    test_sigmoid_cross_entropy()
    test_quadratic_form()
    test_exp_sum()
    test_multi_layer()

    print("\n" + "=" * 60)
    total = PASS_COUNT + FAIL_COUNT
    print(f"Results: {PASS_COUNT}/{total} checks passed, {FAIL_COUNT} failed")
    if FAIL_COUNT == 0:
        print("All tests passed!")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
