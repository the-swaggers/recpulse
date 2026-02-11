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

def test_op_add_scalar_forward():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float32', device='cpu')
    result = t.op_add_scalar(5.0)
    assert result.shape == (4,), f"Expected shape (4,), got {result.shape}"
    assert abs(result.sum_all() - 30.0) < 1e-5, f"Expected 30.0, got {result.sum_all()}"
    print("PASS: op_add_scalar forward")

def test_op_sub_scalar_forward():
    data = [10.0, 20.0, 30.0, 40.0]
    t = rp.values(data, dtype='float32', device='cpu')
    result = t.op_sub_scalar(5.0)
    assert result.shape == (4,), f"Expected shape (4,), got {result.shape}"
    assert abs(result.sum_all() - 80.0) < 1e-5, f"Expected 80.0, got {result.sum_all()}"
    print("PASS: op_sub_scalar forward")

def test_op_add_scalar_backward():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_add_scalar(5.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None, "Gradient should not be None"
    assert grad.shape == (4,), f"Expected grad shape (4,), got {grad.shape}"

    grad_vals = read_grad_vals(grad, 4)
    for i, v in enumerate(grad_vals):
        assert abs(v - 1.0) < 1e-10, f"Expected grad[{i}] = 1.0, got {v}"

    print("PASS: op_add_scalar backward")

def test_op_sub_scalar_backward():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_sub_scalar(5.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None, "Gradient should not be None"
    assert grad.shape == (4,), f"Expected grad shape (4,), got {grad.shape}"

    grad_vals = read_grad_vals(grad, 4)
    for i, v in enumerate(grad_vals):
        assert abs(v - 1.0) < 1e-10, f"Expected grad[{i}] = 1.0, got {v}"

    print("PASS: op_sub_scalar backward")

def test_chain_add_scalar():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    sq = t.op_square()
    added = sq.op_add_scalar(10.0)
    out = added.op_sum_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    expected = [2.0, 4.0, 6.0, 8.0]
    grad_vals = read_grad_vals(grad, 4)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: chain square -> add_scalar -> sum_all backward")

def test_chain_sub_scalar():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    sq = t.op_square()
    subbed = sq.op_sub_scalar(10.0)
    out = subbed.op_sum_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    expected = [2.0, 4.0, 6.0, 8.0]
    grad_vals = read_grad_vals(grad, 4)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: chain square -> sub_scalar -> sum_all backward")

def test_numerical_gradient_add_scalar():
    eps = 1e-5
    scalar = 3.7
    data = [1.5, -0.5, 2.3, 0.7]
    t_np = np.array(data, dtype=np.float64)

    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)
    added = t.op_add_scalar(scalar)
    out = added.op_sum_all()
    out.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 4)

    for i in range(4):
        t_plus = t_np.copy(); t_plus[i] += eps
        t_minus = t_np.copy(); t_minus[i] -= eps
        numerical = ((t_plus + scalar).sum() - (t_minus + scalar).sum()) / (2 * eps)
        assert abs(grad_vals[i] - numerical) < 1e-4, \
            f"Numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}"

    print("PASS: numerical gradient check (add_scalar)")

def test_numerical_gradient_sub_scalar():
    eps = 1e-5
    scalar = 3.7
    data = [1.5, -0.5, 2.3, 0.7]
    t_np = np.array(data, dtype=np.float64)

    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)
    subbed = t.op_sub_scalar(scalar)
    out = subbed.op_sum_all()
    out.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 4)

    for i in range(4):
        t_plus = t_np.copy(); t_plus[i] += eps
        t_minus = t_np.copy(); t_minus[i] -= eps
        numerical = ((t_plus - scalar).sum() - (t_minus - scalar).sum()) / (2 * eps)
        assert abs(grad_vals[i] - numerical) < 1e-4, \
            f"Numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}"

    print("PASS: numerical gradient check (sub_scalar)")

def test_add_scalar_mean_chain():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    added = t.op_add_scalar(100.0)
    out = added.op_mean_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    expected_each = 1.0 / 4.0
    grad_vals = read_grad_vals(grad, 4)
    for i, v in enumerate(grad_vals):
        assert abs(v - expected_each) < 1e-10, f"grad[{i}]: expected {expected_each}, got {v}"

    print("PASS: add_scalar -> mean_all backward")

def test_float32_scalar_backward():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float32', device='cpu')
    t.requires_grad_(True)

    out = t.op_add_scalar(5.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    assert abs(grad.sum_all() - 4.0) < 1e-5, f"Expected sum=4.0, got {grad.sum_all()}"
    print("PASS: float32 add_scalar backward")

def test_op_mul_scalar_forward():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float32', device='cpu')
    result = t.op_mul_scalar(3.0)
    assert abs(result.sum_all() - 30.0) < 1e-5, f"Expected 30.0, got {result.sum_all()}"
    print("PASS: op_mul_scalar forward")

def test_op_div_scalar_forward():
    data = [10.0, 20.0, 30.0, 40.0]
    t = rp.values(data, dtype='float32', device='cpu')
    result = t.op_div_scalar(2.0)
    assert abs(result.sum_all() - 50.0) < 1e-5, f"Expected 50.0, got {result.sum_all()}"
    print("PASS: op_div_scalar forward")

def test_op_mul_scalar_backward():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_mul_scalar(3.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 4)
    for i, v in enumerate(grad_vals):
        assert abs(v - 3.0) < 1e-10, f"Expected grad[{i}] = 3.0, got {v}"

    print("PASS: op_mul_scalar backward")

def test_op_div_scalar_backward():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_div_scalar(4.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 4)
    for i, v in enumerate(grad_vals):
        assert abs(v - 0.25) < 1e-10, f"Expected grad[{i}] = 0.25, got {v}"

    print("PASS: op_div_scalar backward")

def test_chain_mul_scalar():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    sq = t.op_square()
    scaled = sq.op_mul_scalar(2.0)
    out = scaled.op_sum_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    expected = [4.0, 8.0, 12.0, 16.0]
    grad_vals = read_grad_vals(grad, 4)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: chain square -> mul_scalar -> sum_all backward")

def test_chain_div_scalar():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    sq = t.op_square()
    divided = sq.op_div_scalar(2.0)
    out = divided.op_sum_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    expected = [1.0, 2.0, 3.0, 4.0]
    grad_vals = read_grad_vals(grad, 4)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: chain square -> div_scalar -> sum_all backward")

def test_numerical_gradient_mul_scalar():
    eps = 1e-5
    scalar = 2.5
    data = [1.5, -0.5, 2.3, 0.7]
    t_np = np.array(data, dtype=np.float64)

    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)
    scaled = t.op_mul_scalar(scalar)
    out = scaled.op_sum_all()
    out.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 4)

    for i in range(4):
        t_plus = t_np.copy(); t_plus[i] += eps
        t_minus = t_np.copy(); t_minus[i] -= eps
        numerical = ((t_plus * scalar).sum() - (t_minus * scalar).sum()) / (2 * eps)
        assert abs(grad_vals[i] - numerical) < 1e-4, \
            f"Numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}"

    print("PASS: numerical gradient check (mul_scalar)")

def test_numerical_gradient_div_scalar():
    eps = 1e-5
    scalar = 2.5
    data = [1.5, -0.5, 2.3, 0.7]
    t_np = np.array(data, dtype=np.float64)

    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)
    divided = t.op_div_scalar(scalar)
    out = divided.op_sum_all()
    out.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 4)

    for i in range(4):
        t_plus = t_np.copy(); t_plus[i] += eps
        t_minus = t_np.copy(); t_minus[i] -= eps
        numerical = ((t_plus / scalar).sum() - (t_minus / scalar).sum()) / (2 * eps)
        assert abs(grad_vals[i] - numerical) < 1e-4, \
            f"Numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}"

    print("PASS: numerical gradient check (div_scalar)")

def test_mul_div_scalar_chain():
    data = [2.0, 4.0, 6.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    m = t.op_mul_scalar(3.0)
    d = m.op_div_scalar(2.0)
    out = d.op_sum_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    expected_each = 3.0 / 2.0
    grad_vals = read_grad_vals(grad, 3)
    for i, v in enumerate(grad_vals):
        assert abs(v - expected_each) < 1e-10, f"grad[{i}]: expected {expected_each}, got {v}"

    print("PASS: mul_scalar -> div_scalar -> sum_all backward")

def test_op_pow_scalar_forward():
    data = [2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float32', device='cpu')
    result = t.op_pow_scalar(2.0)
    assert abs(result.sum_all() - 29.0) < 1e-5, f"Expected 29.0, got {result.sum_all()}"
    print("PASS: op_pow_scalar forward")

def test_op_pow_scalar_backward():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_pow_scalar(3.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    expected = [3.0, 12.0, 27.0, 48.0]
    grad_vals = read_grad_vals(grad, 4)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: op_pow_scalar backward")

def test_pow_scalar_square_equiv():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_pow_scalar(2.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    expected = [2.0, 4.0, 6.0, 8.0]
    grad_vals = read_grad_vals(grad, 4)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: pow_scalar(2) matches square gradient")

def test_chain_pow_scalar():
    data = [1.0, 2.0, 3.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    added = t.op_add_scalar(1.0)
    powered = added.op_pow_scalar(2.0)
    out = powered.op_sum_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    expected = [4.0, 6.0, 8.0]
    grad_vals = read_grad_vals(grad, 3)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: chain add_scalar -> pow_scalar -> sum_all backward")

def test_numerical_gradient_pow_scalar():
    eps = 1e-5
    scalar = 3.0
    data = [1.5, 2.0, 0.5, 3.0]
    t_np = np.array(data, dtype=np.float64)

    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)
    powered = t.op_pow_scalar(scalar)
    out = powered.op_sum_all()
    out.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 4)

    for i in range(4):
        t_plus = t_np.copy(); t_plus[i] += eps
        t_minus = t_np.copy(); t_minus[i] -= eps
        numerical = ((t_plus ** scalar).sum() - (t_minus ** scalar).sum()) / (2 * eps)
        assert abs(grad_vals[i] - numerical) < 1e-4, \
            f"Numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}"

    print("PASS: numerical gradient check (pow_scalar)")

def test_pow_scalar_fractional():
    data = [1.0, 4.0, 9.0, 16.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_pow_scalar(0.5)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    expected = [0.5, 0.25, 0.5/3.0, 0.125]
    grad_vals = read_grad_vals(grad, 4)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: pow_scalar(0.5) sqrt-equivalent backward")

def test_op_rsub_scalar_forward():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float32', device='cpu')
    result = t.op_rsub_scalar(10.0)
    assert abs(result.sum_all() - 30.0) < 1e-5, f"Expected 30.0, got {result.sum_all()}"
    print("PASS: op_rsub_scalar forward")

def test_op_rsub_scalar_backward():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_rsub_scalar(10.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 4)
    for i, v in enumerate(grad_vals):
        assert abs(v - (-1.0)) < 1e-10, f"Expected grad[{i}] = -1.0, got {v}"

    print("PASS: op_rsub_scalar backward")

def test_numerical_gradient_rsub_scalar():
    eps = 1e-5
    scalar = 5.0
    data = [1.5, -0.5, 2.3, 0.7]
    t_np = np.array(data, dtype=np.float64)

    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)
    rsubbed = t.op_rsub_scalar(scalar)
    out = rsubbed.op_sum_all()
    out.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 4)

    for i in range(4):
        t_plus = t_np.copy(); t_plus[i] += eps
        t_minus = t_np.copy(); t_minus[i] -= eps
        numerical = ((scalar - t_plus).sum() - (scalar - t_minus).sum()) / (2 * eps)
        assert abs(grad_vals[i] - numerical) < 1e-4, \
            f"Numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}"

    print("PASS: numerical gradient check (rsub_scalar)")

def test_op_rdiv_scalar_forward():
    data = [1.0, 2.0, 4.0, 5.0]
    t = rp.values(data, dtype='float32', device='cpu')
    result = t.op_rdiv_scalar(20.0)
    assert abs(result.sum_all() - 39.0) < 1e-4, f"Expected 39.0, got {result.sum_all()}"
    print("PASS: op_rdiv_scalar forward")

def test_op_rdiv_scalar_backward():
    data = [1.0, 2.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_rdiv_scalar(8.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    expected = [-8.0, -2.0, -0.5]
    grad_vals = read_grad_vals(grad, 3)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: op_rdiv_scalar backward")

def test_numerical_gradient_rdiv_scalar():
    eps = 1e-5
    scalar = 6.0
    data = [1.5, 2.0, 3.0, 0.5]
    t_np = np.array(data, dtype=np.float64)

    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)
    rdived = t.op_rdiv_scalar(scalar)
    out = rdived.op_sum_all()
    out.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 4)

    for i in range(4):
        t_plus = t_np.copy(); t_plus[i] += eps
        t_minus = t_np.copy(); t_minus[i] -= eps
        numerical = ((scalar / t_plus).sum() - (scalar / t_minus).sum()) / (2 * eps)
        assert abs(grad_vals[i] - numerical) < 1e-4, \
            f"Numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}"

    print("PASS: numerical gradient check (rdiv_scalar)")

def test_op_rpow_scalar_forward():
    data = [1.0, 2.0, 3.0]
    t = rp.values(data, dtype='float32', device='cpu')
    result = t.op_rpow_scalar(2.0)
    assert abs(result.sum_all() - 14.0) < 1e-4, f"Expected 14.0, got {result.sum_all()}"
    print("PASS: op_rpow_scalar forward")

def test_op_rpow_scalar_backward():
    data = [1.0, 2.0, 3.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_rpow_scalar(2.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    ln2 = np.log(2.0)
    expected = [2.0 * ln2, 4.0 * ln2, 8.0 * ln2]
    grad_vals = read_grad_vals(grad, 3)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: op_rpow_scalar backward")

def test_numerical_gradient_rpow_scalar():
    eps = 1e-5
    scalar = 3.0
    data = [0.5, 1.0, 1.5, 2.0]
    t_np = np.array(data, dtype=np.float64)

    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)
    rpowed = t.op_rpow_scalar(scalar)
    out = rpowed.op_sum_all()
    out.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 4)

    for i in range(4):
        t_plus = t_np.copy(); t_plus[i] += eps
        t_minus = t_np.copy(); t_minus[i] -= eps
        numerical = ((scalar ** t_plus).sum() - (scalar ** t_minus).sum()) / (2 * eps)
        assert abs(grad_vals[i] - numerical) < 1e-4, \
            f"Numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}"

    print("PASS: numerical gradient check (rpow_scalar)")

def test_op_logb_scalar_forward():
    data = [1.0, 10.0, 100.0]
    t = rp.values(data, dtype='float64', device='cpu')
    result = t.op_logb_scalar(10.0)
    expected = [0.0, 1.0, 2.0]
    grad_vals = read_grad_vals(result, 3)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"val[{i}]: expected {exp}, got {got}"
    print("PASS: op_logb_scalar forward")

def test_op_logb_scalar_backward():
    data = [1.0, 2.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_logb_scalar(2.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    ln2 = np.log(2.0)
    expected = [1.0 / (1.0 * ln2), 1.0 / (2.0 * ln2), 1.0 / (4.0 * ln2)]
    grad_vals = read_grad_vals(grad, 3)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: op_logb_scalar backward")

def test_numerical_gradient_logb_scalar():
    eps = 1e-5
    scalar = 10.0
    data = [1.5, 2.0, 5.0, 0.5]
    t_np = np.array(data, dtype=np.float64)

    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)
    logged = t.op_logb_scalar(scalar)
    out = logged.op_sum_all()
    out.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 4)

    for i in range(4):
        t_plus = t_np.copy(); t_plus[i] += eps
        t_minus = t_np.copy(); t_minus[i] -= eps
        f_plus = (np.log(t_plus) / np.log(scalar)).sum()
        f_minus = (np.log(t_minus) / np.log(scalar)).sum()
        numerical = (f_plus - f_minus) / (2 * eps)
        assert abs(grad_vals[i] - numerical) < 1e-4, \
            f"Numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}"

    print("PASS: numerical gradient check (logb_scalar)")

def test_op_rlogb_scalar_forward():
    data = [2.0, 4.0, 8.0]
    t = rp.values(data, dtype='float64', device='cpu')
    result = t.op_rlogb_scalar(16.0)
    ln16 = np.log(16.0)
    expected = [ln16 / np.log(2.0), ln16 / np.log(4.0), ln16 / np.log(8.0)]
    grad_vals = read_grad_vals(result, 3)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"val[{i}]: expected {exp}, got {got}"
    print("PASS: op_rlogb_scalar forward")

def test_op_rlogb_scalar_backward():
    data = [2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_rlogb_scalar(8.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    ln8 = np.log(8.0)
    expected = [-ln8 / (x * np.log(x)**2) for x in [2.0, 3.0, 4.0]]
    grad_vals = read_grad_vals(grad, 3)
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: op_rlogb_scalar backward")

def test_numerical_gradient_rlogb_scalar():
    eps = 1e-5
    scalar = 10.0
    data = [2.0, 3.0, 5.0, 7.0]
    t_np = np.array(data, dtype=np.float64)

    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)
    rlogged = t.op_rlogb_scalar(scalar)
    out = rlogged.op_sum_all()
    out.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 4)

    for i in range(4):
        t_plus = t_np.copy(); t_plus[i] += eps
        t_minus = t_np.copy(); t_minus[i] -= eps
        f_plus = (np.log(scalar) / np.log(t_plus)).sum()
        f_minus = (np.log(scalar) / np.log(t_minus)).sum()
        numerical = (f_plus - f_minus) / (2 * eps)
        assert abs(grad_vals[i] - numerical) < 1e-4, \
            f"Numerical check failed at [{i}]: autograd={grad_vals[i]}, numerical={numerical}"

    print("PASS: numerical gradient check (rlogb_scalar)")

def test_chain_mul_then_add_scalar():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    scaled = t.op_mul_scalar(3.0)
    shifted = scaled.op_add_scalar(10.0)
    out = shifted.op_sum_all()
    out.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 4)
    for i, v in enumerate(grad_vals):
        assert abs(v - 3.0) < 1e-10, f"grad[{i}]: expected 3.0, got {v}"

    print("PASS: chain mul_scalar(3) -> add_scalar(10) -> sum backward")


def test_float64_mul_scalar_precision():
    data = [1.123456789012345, 2.987654321098765, 0.000000001]
    t = rp.values(data, dtype='float64', device='cpu')
    t.requires_grad_(True)

    out = t.op_mul_scalar(7.0)
    s = out.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 3)
    for i, v in enumerate(grad_vals):
        assert abs(v - 7.0) < 1e-10, f"grad[{i}]: expected 7.0, got {v}"

    print("PASS: float64 mul_scalar precision test")


if __name__ == '__main__':
    test_op_add_scalar_forward()
    test_op_sub_scalar_forward()
    test_op_add_scalar_backward()
    test_op_sub_scalar_backward()
    test_chain_add_scalar()
    test_chain_sub_scalar()
    test_numerical_gradient_add_scalar()
    test_numerical_gradient_sub_scalar()
    test_add_scalar_mean_chain()
    test_float32_scalar_backward()
    test_op_mul_scalar_forward()
    test_op_div_scalar_forward()
    test_op_mul_scalar_backward()
    test_op_div_scalar_backward()
    test_chain_mul_scalar()
    test_chain_div_scalar()
    test_numerical_gradient_mul_scalar()
    test_numerical_gradient_div_scalar()
    test_mul_div_scalar_chain()
    test_op_pow_scalar_forward()
    test_op_pow_scalar_backward()
    test_pow_scalar_square_equiv()
    test_chain_pow_scalar()
    test_numerical_gradient_pow_scalar()
    test_pow_scalar_fractional()
    test_op_rsub_scalar_forward()
    test_op_rsub_scalar_backward()
    test_numerical_gradient_rsub_scalar()
    test_op_rdiv_scalar_forward()
    test_op_rdiv_scalar_backward()
    test_numerical_gradient_rdiv_scalar()
    test_op_rpow_scalar_forward()
    test_op_rpow_scalar_backward()
    test_numerical_gradient_rpow_scalar()
    test_op_logb_scalar_forward()
    test_op_logb_scalar_backward()
    test_numerical_gradient_logb_scalar()
    test_op_rlogb_scalar_forward()
    test_op_rlogb_scalar_backward()
    test_numerical_gradient_rlogb_scalar()
    test_chain_mul_then_add_scalar()
    test_float64_mul_scalar_precision()
    print("\nAll tests passed!")
