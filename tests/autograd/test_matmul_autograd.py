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

def test_matmul_forward():
    a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    a1d = rp.values(a_data, dtype='float64', device='cpu')
    a = a1d.reshape([2, 3])
    b1d = rp.values(b_data, dtype='float64', device='cpu')
    b = b1d.reshape([3, 2])

    c = a.op_matmul(b)
    assert c.shape == (2, 2), f"Expected shape (2, 2), got {c.shape}"

    a_np = np.array(a_data, dtype=np.float64).reshape(2, 3)
    b_np = np.array(b_data, dtype=np.float64).reshape(3, 2)
    c_np = a_np @ b_np

    assert abs(c.sum_all() - c_np.sum()) < 1e-10, \
        f"Expected sum {c_np.sum()}, got {c.sum_all()}"

    print("PASS: matmul forward")

def test_matmul_backward_a():
    a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b_data = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]

    a1d = rp.values(a_data, dtype='float64', device='cpu')
    a = a1d.reshape([2, 3])
    a.requires_grad_(True)

    b1d = rp.values(b_data, dtype='float64', device='cpu')
    b = b1d.reshape([3, 2])

    c = a.op_matmul(b)
    s = c.op_sum_all()
    s.backward()

    grad = a.grad
    assert grad is not None, "Gradient of a should not be None"
    assert grad.shape == (2, 3), f"Expected grad shape (2, 3), got {grad.shape}"

    a_np = np.array(a_data, dtype=np.float64).reshape(2, 3)
    b_np = np.array(b_data, dtype=np.float64).reshape(3, 2)
    grad_np = np.ones((2, 2)) @ b_np.T

    grad_flat = read_grad_vals(grad, 6)
    expected_flat = grad_np.flatten().tolist()
    for i, (got, exp) in enumerate(zip(grad_flat, expected_flat)):
        assert abs(got - exp) < 1e-10, f"grad_a[{i}]: expected {exp}, got {got}"

    print("PASS: matmul backward (grad A)")

def test_matmul_backward_b():
    a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b_data = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]

    a1d = rp.values(a_data, dtype='float64', device='cpu')
    a = a1d.reshape([2, 3])

    b1d = rp.values(b_data, dtype='float64', device='cpu')
    b = b1d.reshape([3, 2])
    b.requires_grad_(True)

    c = a.op_matmul(b)
    s = c.op_sum_all()
    s.backward()

    grad = b.grad
    assert grad is not None, "Gradient of b should not be None"
    assert grad.shape == (3, 2), f"Expected grad shape (3, 2), got {grad.shape}"

    a_np = np.array(a_data, dtype=np.float64).reshape(2, 3)
    grad_np = a_np.T @ np.ones((2, 2))

    grad_flat = read_grad_vals(grad, 6)
    expected_flat = grad_np.flatten().tolist()
    for i, (got, exp) in enumerate(zip(grad_flat, expected_flat)):
        assert abs(got - exp) < 1e-10, f"grad_b[{i}]: expected {exp}, got {got}"

    print("PASS: matmul backward (grad B)")

def test_matmul_backward_both():
    a_data = [1.0, 2.0, 3.0, 4.0]
    b_data = [5.0, 6.0, 7.0, 8.0]

    a1d = rp.values(a_data, dtype='float64', device='cpu')
    a = a1d.reshape([2, 2])
    a.requires_grad_(True)

    b1d = rp.values(b_data, dtype='float64', device='cpu')
    b = b1d.reshape([2, 2])
    b.requires_grad_(True)

    c = a.op_matmul(b)
    s = c.op_sum_all()
    s.backward()

    a_np = np.array(a_data, dtype=np.float64).reshape(2, 2)
    b_np = np.array(b_data, dtype=np.float64).reshape(2, 2)

    grad_a_np = np.ones((2, 2)) @ b_np.T
    grad_b_np = a_np.T @ np.ones((2, 2))

    grad_a = a.grad
    assert grad_a is not None
    grad_a_flat = read_grad_vals(grad_a, 4)
    for i, (got, exp) in enumerate(zip(grad_a_flat, grad_a_np.flatten())):
        assert abs(got - exp) < 1e-10, f"grad_a[{i}]: expected {exp}, got {got}"

    grad_b = b.grad
    assert grad_b is not None
    grad_b_flat = read_grad_vals(grad_b, 4)
    for i, (got, exp) in enumerate(zip(grad_b_flat, grad_b_np.flatten())):
        assert abs(got - exp) < 1e-10, f"grad_b[{i}]: expected {exp}, got {got}"

    print("PASS: matmul backward (both A and B)")

def test_numerical_gradient_matmul():
    eps = 1e-5
    a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64).reshape(2, 3)
    b_np = np.array([0.5, 1.5, -0.5, 1.0, 2.0, -1.0], dtype=np.float64).reshape(3, 2)

    a1d = rp.values(a_np.flatten().tolist(), dtype='float64', device='cpu')
    a = a1d.reshape([2, 3])
    a.requires_grad_(True)
    b1d = rp.values(b_np.flatten().tolist(), dtype='float64', device='cpu')
    b = b1d.reshape([3, 2])

    c = a.op_matmul(b)
    s = c.op_sum_all()
    s.backward()

    grad_a = a.grad
    grad_a_flat = read_grad_vals(grad_a, 6)

    for idx in range(6):
        i, j = idx // 3, idx % 3
        a_plus = a_np.copy(); a_plus[i, j] += eps
        a_minus = a_np.copy(); a_minus[i, j] -= eps
        numerical = ((a_plus @ b_np).sum() - (a_minus @ b_np).sum()) / (2 * eps)
        assert abs(grad_a_flat[idx] - numerical) < 1e-4, \
            f"Numerical check failed for A[{i},{j}]: autograd={grad_a_flat[idx]}, numerical={numerical}"

    print("PASS: numerical gradient check (matmul, grad A)")

def test_numerical_gradient_matmul_b():
    eps = 1e-5
    a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64).reshape(2, 3)
    b_np = np.array([0.5, 1.5, -0.5, 1.0, 2.0, -1.0], dtype=np.float64).reshape(3, 2)

    a1d = rp.values(a_np.flatten().tolist(), dtype='float64', device='cpu')
    a = a1d.reshape([2, 3])
    b1d = rp.values(b_np.flatten().tolist(), dtype='float64', device='cpu')
    b = b1d.reshape([3, 2])
    b.requires_grad_(True)

    c = a.op_matmul(b)
    s = c.op_sum_all()
    s.backward()

    grad_b = b.grad
    grad_b_flat = read_grad_vals(grad_b, 6)

    for idx in range(6):
        i, j = idx // 2, idx % 2
        b_plus = b_np.copy(); b_plus[i, j] += eps
        b_minus = b_np.copy(); b_minus[i, j] -= eps
        numerical = ((a_np @ b_plus).sum() - (a_np @ b_minus).sum()) / (2 * eps)
        assert abs(grad_b_flat[idx] - numerical) < 1e-4, \
            f"Numerical check failed for B[{i},{j}]: autograd={grad_b_flat[idx]}, numerical={numerical}"

    print("PASS: numerical gradient check (matmul, grad B)")

def test_matmul_chain():
    a_data = [1.0, 2.0, 3.0, 4.0]
    b_data = [1.0, 0.0, 0.0, 1.0]

    a1d = rp.values(a_data, dtype='float64', device='cpu')
    a = a1d.reshape([2, 2])
    a.requires_grad_(True)

    b1d = rp.values(b_data, dtype='float64', device='cpu')
    b = b1d.reshape([2, 2])

    c = a.op_matmul(b)
    scaled = c.op_mul_scalar(2.0)
    out = scaled.op_sum_all()
    out.backward()

    grad = a.grad
    assert grad is not None

    a_np = np.array(a_data, dtype=np.float64).reshape(2, 2)
    b_np = np.array(b_data, dtype=np.float64).reshape(2, 2)
    grad_np = 2.0 * np.ones((2, 2)) @ b_np.T

    grad_flat = read_grad_vals(grad, 4)
    for i, (got, exp) in enumerate(zip(grad_flat, grad_np.flatten())):
        assert abs(got - exp) < 1e-10, f"grad[{i}]: expected {exp}, got {got}"

    print("PASS: matmul -> mul_scalar -> sum_all chain backward")

def test_matmul_nonsquare():
    a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64).reshape(2, 4)
    b_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    dtype=np.float64).reshape(4, 3)

    a1d = rp.values(a_np.flatten().tolist(), dtype='float64', device='cpu')
    a = a1d.reshape([2, 4])
    a.requires_grad_(True)

    b1d = rp.values(b_np.flatten().tolist(), dtype='float64', device='cpu')
    b = b1d.reshape([4, 3])
    b.requires_grad_(True)

    c = a.op_matmul(b)
    assert c.shape == (2, 3), f"Expected (2, 3), got {c.shape}"

    s = c.op_sum_all()
    s.backward()

    grad_a_expected = np.ones((2, 3)) @ b_np.T
    grad_b_expected = a_np.T @ np.ones((2, 3))

    grad_a = a.grad
    assert grad_a is not None
    grad_a_flat = read_grad_vals(grad_a, 8)
    for i, (got, exp) in enumerate(zip(grad_a_flat, grad_a_expected.flatten())):
        assert abs(got - exp) < 1e-10, f"grad_a[{i}]: expected {exp}, got {got}"

    grad_b = b.grad
    assert grad_b is not None
    grad_b_flat = read_grad_vals(grad_b, 12)
    for i, (got, exp) in enumerate(zip(grad_b_flat, grad_b_expected.flatten())):
        assert abs(got - exp) < 1e-10, f"grad_b[{i}]: expected {exp}, got {got}"

    print("PASS: matmul non-square (2x4 @ 4x3)")


def test_matmul_float64():
    a_np = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64).reshape(2, 2)
    b_np = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64).reshape(2, 2)

    a1d = rp.values(a_np.flatten().tolist(), dtype='float64', device='cpu')
    a = a1d.reshape([2, 2])
    a.requires_grad_(True)

    b1d = rp.values(b_np.flatten().tolist(), dtype='float64', device='cpu')
    b = b1d.reshape([2, 2])
    b.requires_grad_(True)

    c = a.op_matmul(b)
    s = c.op_sum_all()
    s.backward()

    c_np = a_np @ b_np
    assert abs(s.sum_all() - c_np.sum()) < 1e-10, \
        f"Float64 forward mismatch: {s.sum_all()} vs {c_np.sum()}"

    grad_a_expected = np.ones((2, 2)) @ b_np.T
    grad_b_expected = a_np.T @ np.ones((2, 2))

    grad_a = a.grad
    assert grad_a is not None
    grad_a_flat = read_grad_vals(grad_a, 4)
    for i, (got, exp) in enumerate(zip(grad_a_flat, grad_a_expected.flatten())):
        assert abs(got - exp) < 1e-10, f"float64 grad_a[{i}]: expected {exp}, got {got}"

    grad_b = b.grad
    assert grad_b is not None
    grad_b_flat = read_grad_vals(grad_b, 4)
    for i, (got, exp) in enumerate(zip(grad_b_flat, grad_b_expected.flatten())):
        assert abs(got - exp) < 1e-10, f"float64 grad_b[{i}]: expected {exp}, got {got}"

    print("PASS: matmul float64")


if __name__ == '__main__':
    test_matmul_forward()
    test_matmul_backward_a()
    test_matmul_backward_b()
    test_matmul_backward_both()
    test_numerical_gradient_matmul()
    test_numerical_gradient_matmul_b()
    test_matmul_chain()
    test_matmul_nonsquare()
    test_matmul_float64()
    print("\nAll tests passed!")
