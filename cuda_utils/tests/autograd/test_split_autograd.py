import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import numpy as np

def read_grad_vals(grad, n):
    flat = grad.flatten()
    vals = []
    for i in range(n):
        v = flat.slice(start=[i], stop=[i+1]).sum_all()
        vals.append(v)
    return vals

def test_split_forward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([6])

    parts = t.split([2, 4], 0)
    assert len(parts) == 2, f"Expected 2 parts, got {len(parts)}"
    assert parts[0].shape == (2,), f"Expected (2,), got {parts[0].shape}"
    assert parts[1].shape == (4,), f"Expected (4,), got {parts[1].shape}"
    assert abs(parts[0].sum_all() - 3.0) < 1e-10
    assert abs(parts[1].sum_all() - 18.0) < 1e-10
    print("PASS: split forward")

def test_split_forward_2d():
    data = list(range(1, 13))
    t1d = rp.values([float(x) for x in data], dtype='float64', device='cpu')
    t = t1d.reshape([3, 4])

    parts = t.split([1, 2], 0)
    assert len(parts) == 2
    assert parts[0].shape == (1, 4), f"Expected (1, 4), got {parts[0].shape}"
    assert parts[1].shape == (2, 4), f"Expected (2, 4), got {parts[1].shape}"
    assert abs(parts[0].sum_all() - 10.0) < 1e-10
    assert abs(parts[1].sum_all() - 68.0) < 1e-10
    print("PASS: split forward 2D")

def test_split_backward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([6])
    t.requires_grad_(True)

    parts = t.split([2, 4], 0)
    s0 = parts[0].op_sum_all()
    s0.backward()

    grad = t.grad
    assert grad is not None, "Gradient should not be None"
    grad_vals = read_grad_vals(grad, 6)
    expected = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"
    print("PASS: split backward (first part)")

def test_split_backward_second():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([6])
    t.requires_grad_(True)

    parts = t.split([2, 4], 0)
    s1 = parts[1].op_sum_all()
    s1.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 6)
    expected = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"
    print("PASS: split backward (second part)")

def test_split_backward_2d():
    data = [float(x) for x in range(1, 13)]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([3, 4])
    t.requires_grad_(True)

    parts = t.split([1, 2], 0)
    sq = parts[1].op_square()
    s = sq.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 12)

    np_t = np.array(data).reshape(3, 4)
    np_parts = np.split(np_t, [1], axis=0)
    np_grad = np.zeros_like(np_t)
    np_grad[1:, :] = 2.0 * np_parts[1]

    for i, (got, exp) in enumerate(zip(grad_vals, np_grad.flatten())):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"
    print("PASS: split backward 2D")

def test_split_backward_dim1():
    data = [float(x) for x in range(1, 13)]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([3, 4])
    t.requires_grad_(True)

    parts = t.split([1, 3], 1)
    assert parts[0].shape == (3, 1), f"Expected (3, 1), got {parts[0].shape}"
    assert parts[1].shape == (3, 3), f"Expected (3, 3), got {parts[1].shape}"

    s = parts[0].op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 12)

    expected = [1.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0]
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"
    print("PASS: split backward dim=1")

def test_split_numerical_gradient():
    eps = 1e-5
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    np_data = np.array(data, dtype=np.float64)

    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([6])
    t.requires_grad_(True)

    parts = t.split([2, 4], 0)
    sq = parts[1].op_square()
    s = sq.op_sum_all()
    s.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 6)

    for idx in range(6):
        d_plus = np_data.copy(); d_plus[idx] += eps
        d_minus = np_data.copy(); d_minus[idx] -= eps
        p_plus = np.split(d_plus, [2])[1]
        p_minus = np.split(d_minus, [2])[1]
        numerical = ((p_plus ** 2).sum() - (p_minus ** 2).sum()) / (2 * eps)
        assert abs(grad_vals[idx] - numerical) < 1e-4, \
            f"Numerical check failed for [{idx}]: autograd={grad_vals[idx]}, numerical={numerical}"
    print("PASS: split numerical gradient check")

def test_split_equal_forward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([6])

    parts = t.split_equal(3, 0)
    assert len(parts) == 3, f"Expected 3 parts, got {len(parts)}"
    for p in parts:
        assert p.shape == (2,), f"Expected (2,), got {p.shape}"
    assert abs(parts[0].sum_all() - 3.0) < 1e-10
    assert abs(parts[1].sum_all() - 7.0) < 1e-10
    assert abs(parts[2].sum_all() - 11.0) < 1e-10
    print("PASS: split_equal forward")

def test_split_equal_forward_2d():
    data = [float(x) for x in range(1, 13)]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([4, 3])

    parts = t.split_equal(2, 0)
    assert len(parts) == 2
    assert parts[0].shape == (2, 3)
    assert parts[1].shape == (2, 3)
    print("PASS: split_equal forward 2D")

def test_split_equal_backward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([6])
    t.requires_grad_(True)

    parts = t.split_equal(3, 0)
    s = parts[1].op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 6)
    expected = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"
    print("PASS: split_equal backward (middle part)")

def test_split_equal_backward_2d():
    data = [float(x) for x in range(1, 13)]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([4, 3])
    t.requires_grad_(True)

    parts = t.split_equal(2, 0)
    sq = parts[0].op_square()
    s = sq.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 12)

    np_t = np.array(data).reshape(4, 3)
    np_grad = np.zeros_like(np_t)
    np_grad[:2, :] = 2.0 * np_t[:2, :]

    for i, (got, exp) in enumerate(zip(grad_vals, np_grad.flatten())):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"
    print("PASS: split_equal backward 2D")

def test_split_equal_numerical_gradient():
    eps = 1e-5
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    np_data = np.array(data, dtype=np.float64)

    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([6])
    t.requires_grad_(True)

    parts = t.split_equal(3, 0)
    sq = parts[2].op_square()
    s = sq.op_sum_all()
    s.backward()

    grad = t.grad
    grad_vals = read_grad_vals(grad, 6)

    for idx in range(6):
        d_plus = np_data.copy(); d_plus[idx] += eps
        d_minus = np_data.copy(); d_minus[idx] -= eps
        p_plus = np.split(d_plus, 3)[2]
        p_minus = np.split(d_minus, 3)[2]
        numerical = ((p_plus ** 2).sum() - (p_minus ** 2).sum()) / (2 * eps)
        assert abs(grad_vals[idx] - numerical) < 1e-4, \
            f"Numerical check failed for [{idx}]: autograd={grad_vals[idx]}, numerical={numerical}"
    print("PASS: split_equal numerical gradient check")

def test_split_chain():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([6])
    t.requires_grad_(True)

    parts = t.split([3, 3], 0)
    sq0 = parts[0].op_square()
    sq1 = parts[1].op_square()
    s0 = sq0.op_sum_all()
    s1 = sq1.op_sum_all()
    total = s0.op_add(s1)
    total.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 6)
    expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"
    print("PASS: split chain (both parts contribute to loss)")

def test_split_float64():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([6])
    t.requires_grad_(True)

    parts = t.split([3, 3], 0)
    s = parts[0].op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 6)
    expected = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"
    print("PASS: split float64 backward")


def test_split_gradient_to_specific_chunks():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    t1d = rp.values(data, dtype='float64', device='cpu')
    t = t1d.reshape([8])
    t.requires_grad_(True)

    parts = t.split([2, 3, 3], 0)

    sq = parts[1].op_square()
    s = sq.op_sum_all()
    s.backward()

    grad = t.grad
    assert grad is not None
    grad_vals = read_grad_vals(grad, 8)

    expected = [0.0, 0.0, 6.0, 8.0, 10.0, 0.0, 0.0, 0.0]
    for i, (got, exp) in enumerate(zip(grad_vals, expected)):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"
    print("PASS: split gradient routing to specific chunk (middle)")


if __name__ == '__main__':
    test_split_forward()
    test_split_forward_2d()
    test_split_backward()
    test_split_backward_second()
    test_split_backward_2d()
    test_split_backward_dim1()
    test_split_numerical_gradient()
    test_split_equal_forward()
    test_split_equal_forward_2d()
    test_split_equal_backward()
    test_split_equal_backward_2d()
    test_split_equal_numerical_gradient()
    test_split_chain()
    test_split_float64()
    test_split_gradient_to_specific_chunks()
    print("\nAll split/split_equal tests passed!")
