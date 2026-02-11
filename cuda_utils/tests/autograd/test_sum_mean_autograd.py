import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import numpy as np

def read_grad_vals(grad, n):
    vals = []
    for i in range(n):
        v = grad.slice(start=[i], stop=[i+1]).sum_all()
        vals.append(v)
    return vals

def test_op_sum_all_forward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t = rp.values(data, dtype='float32', device='cpu')
    result = t.op_sum_all()
    assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
    assert result.size == 1
    val = result.sum_all()
    assert abs(val - 21.0) < 1e-5, f"Expected 21.0, got {val}"
    print("PASS: op_sum_all forward")

def test_op_mean_all_forward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t = rp.values(data, dtype='float32', device='cpu')
    result = t.op_mean_all()
    assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
    val = result.sum_all()
    assert abs(val - 3.5) < 1e-5, f"Expected 3.5, got {val}"
    print("PASS: op_mean_all forward")

def test_op_sum_all_backward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    result = t.op_sum_all()
    result.backward()

    grad = t.grad
    assert grad is not None, "Gradient should not be None"
    assert grad.shape == (6,), f"Expected grad shape (6,), got {grad.shape}"
    assert abs(grad.sum_all() - 6.0) < 1e-10, f"Expected sum of grads = 6.0, got {grad.sum_all()}"

    grad_vals = read_grad_vals(grad, 6)
    for i, v in enumerate(grad_vals):
        assert abs(v - 1.0) < 1e-10, f"Expected grad[{i}] = 1.0, got {v}"

    print("PASS: op_sum_all backward")

def test_op_mean_all_backward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    result = t.op_mean_all()
    result.backward()

    grad = t.grad
    assert grad is not None, "Gradient should not be None"
    assert grad.shape == (6,), f"Expected grad shape (6,), got {grad.shape}"
    assert abs(grad.sum_all() - 1.0) < 1e-10, f"Expected sum of grads = 1.0, got {grad.sum_all()}"

    expected = 1.0 / 6.0
    grad_vals = read_grad_vals(grad, 6)
    for i, v in enumerate(grad_vals):
        assert abs(v - expected) < 1e-10, f"Expected grad[{i}] = {expected}, got {v}"

    print("PASS: op_mean_all backward")

def test_chain_sum_all():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    sq = t.op_square()
    out = sq.op_sum_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    expected = [2.0, 4.0, 6.0, 8.0]
    grad_vals = read_grad_vals(grad, 4)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: chain square -> sum_all backward")

def test_chain_mean_all():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    sq = t.op_square()
    out = sq.op_mean_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    expected = [2.0 * 1.0 / 4.0, 2.0 * 2.0 / 4.0, 2.0 * 3.0 / 4.0, 2.0 * 4.0 / 4.0]
    grad_vals = read_grad_vals(grad, 4)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: chain square -> mean_all backward")

def test_numerical_gradient_sum():
    eps = 1e-5
    data = [1.5, -0.5, 2.3, 0.7]
    t_np = np.array(data, dtype=np.float64)

    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)
    out = t.op_sum_all()
    out.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 4)

    for i in range(4):
        t_plus = t_np.copy(); t_plus[i] += eps
        t_minus = t_np.copy(); t_minus[i] -= eps
        numerical = (t_plus.sum() - t_minus.sum()) / (2 * eps)
        assert abs(grad_vals[i] - numerical) < 1e-4, \
            f"Numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}"

    print("PASS: numerical gradient check (sum_all)")

def test_numerical_gradient_mean():
    eps = 1e-5
    data = [1.5, -0.5, 2.3, 0.7]
    t_np = np.array(data, dtype=np.float64)

    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)
    out = t.op_mean_all()
    out.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 4)

    for i in range(4):
        t_plus = t_np.copy(); t_plus[i] += eps
        t_minus = t_np.copy(); t_minus[i] -= eps
        numerical = (t_plus.mean() - t_minus.mean()) / (2 * eps)
        assert abs(grad_vals[i] - numerical) < 1e-4, \
            f"Numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}"

    print("PASS: numerical gradient check (mean_all)")

def test_2d_sum_backward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([2, 3])
    t.requires_grad_(True)

    out = t.op_sum_all()
    out.backward()

    grad = t.grad
    assert grad is not None, "Gradient should not be None"
    assert grad.shape == (2, 3), f"Expected grad shape (2, 3), got {grad.shape}"
    assert abs(grad.sum_all() - 6.0) < 1e-10, f"Expected sum of grads = 6.0, got {grad.sum_all()}"
    print("PASS: 2D sum_all backward")

def test_2d_mean_backward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([2, 3])
    t.requires_grad_(True)

    out = t.op_mean_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    assert grad.shape == (2, 3), f"Expected grad shape (2, 3), got {grad.shape}"
    assert abs(grad.sum_all() - 1.0) < 1e-10, f"Expected sum of grads = 1.0, got {grad.sum_all()}"
    print("PASS: 2D mean_all backward")

def test_longer_chain():
    data = [1.0, -1.0, 2.0, -2.0, 3.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    r = t.op_relu()
    s = r.op_square()
    out = s.op_sum_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 5)
    expected = [2.0, 0.0, 4.0, 0.0, 6.0]
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-6, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: longer chain relu -> square -> sum_all")

def test_float32_backward():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float32', device='cpu')
    t.requires_grad_(True)

    out = t.op_sum_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    assert abs(grad.sum_all() - 4.0) < 1e-5, f"Expected sum=4.0, got {grad.sum_all()}"
    print("PASS: float32 sum_all backward")

def test_sum_all_larger_tensor():
    data = [float(i) for i in range(1, 101)]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    result = t.op_sum_all()
    result.backward()

    grad = t.grad
    assert grad is not None, "Gradient should not be None for 100-element tensor"
    assert grad.shape == (100,), f"Expected grad shape (100,), got {grad.shape}"
    assert abs(grad.sum_all() - 100.0) < 1e-10, \
        f"Expected sum of grads = 100.0, got {grad.sum_all()}"

    grad_vals = read_grad_vals(grad, 100)
    for i, v in enumerate(grad_vals):
        assert abs(v - 1.0) < 1e-10, f"Expected grad[{i}] = 1.0, got {v}"

    print("PASS: sum_all larger tensor (100 elements)")


def test_mean_all_gradient_value():
    for n in [3, 5, 10, 25]:
        data = [float(i) for i in range(1, n + 1)]
        t = rp.values(data, dtype='float64', device='cpu')
        t.requires_grad_(True)

        result = t.op_mean_all()
        result.backward()

        grad = t.grad
        assert grad is not None
        expected_each = 1.0 / n
        grad_vals = read_grad_vals(grad, n)
        for i, v in enumerate(grad_vals):
            assert abs(v - expected_each) < 1e-10, \
                f"n={n}, grad[{i}]: expected {expected_each}, got {v}"

    print("PASS: mean_all gradient = 1/n for n in {3,5,10,25}")


if __name__ == '__main__':
    test_op_sum_all_forward()
    test_op_mean_all_forward()
    test_op_sum_all_backward()
    test_op_mean_all_backward()
    test_chain_sum_all()
    test_chain_mean_all()
    test_numerical_gradient_sum()
    test_numerical_gradient_mean()
    test_2d_sum_backward()
    test_2d_mean_backward()
    test_longer_chain()
    test_float32_backward()
    test_sum_all_larger_tensor()
    test_mean_all_gradient_value()
    print("\nAll tests passed!")
