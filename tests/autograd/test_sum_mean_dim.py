import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import numpy as np


def read_1d(tensor):
    vals = []
    for i in range(tensor.size):
        s = tensor.slice(start=[i], stop=[i+1])
        v = s.sum_all()
        vals.append(v)
    return vals

def to_numpy(tensor):
    t_copy = tensor.copy()
    flat = t_copy.reshape([tensor.size])
    vals = read_1d(flat)
    return np.array(vals).reshape(tensor.shape)

def make_tensor(data, shape, dtype='float64', requires_grad=False):
    t1d = rp.values(data if isinstance(data, list) else data.tolist(),
                    dtype=dtype, device='cpu')
    t = t1d.reshape(list(shape))
    if requires_grad:
        t.requires_grad_(True)
    return t1d, t


def test_sum_dim_forward_2d():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    base, t = make_tensor(data, [2, 3])
    np_t = np.array(data).reshape(2, 3)

    r0 = t.sum_dim(dim=0)
    np_r0 = np_t.sum(axis=0)
    assert r0.shape == (3,), f"Expected (3,), got {r0.shape}"
    got = to_numpy(r0)
    assert np.allclose(got, np_r0, atol=1e-10), f"dim=0: {got} vs {np_r0}"

    r1 = t.sum_dim(dim=1)
    np_r1 = np_t.sum(axis=1)
    assert r1.shape == (2,), f"Expected (2,), got {r1.shape}"
    got = to_numpy(r1)
    assert np.allclose(got, np_r1, atol=1e-10), f"dim=1: {got} vs {np_r1}"

    print("PASS: sum_dim forward 2D")


def test_mean_dim_forward_2d():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    base, t = make_tensor(data, [2, 3])
    np_t = np.array(data).reshape(2, 3)

    r0 = t.mean_dim(dim=0)
    np_r0 = np_t.mean(axis=0)
    assert r0.shape == (3,), f"Expected (3,), got {r0.shape}"
    got = to_numpy(r0)
    assert np.allclose(got, np_r0, atol=1e-10), f"dim=0: {got} vs {np_r0}"

    r1 = t.mean_dim(dim=1)
    np_r1 = np_t.mean(axis=1)
    assert r1.shape == (2,), f"Expected (2,), got {r1.shape}"
    got = to_numpy(r1)
    assert np.allclose(got, np_r1, atol=1e-10), f"dim=1: {got} vs {np_r1}"

    print("PASS: mean_dim forward 2D")


def test_sum_dim_forward_3d():
    data = [float(x) for x in range(1, 25)]
    base, t = make_tensor(data, [2, 3, 4])
    np_t = np.array(data, dtype=np.float64).reshape(2, 3, 4)

    for dim in range(3):
        r = t.sum_dim(dim=dim)
        np_r = np_t.sum(axis=dim)
        assert r.shape == tuple(np_r.shape), f"dim={dim}: shape {r.shape} vs {np_r.shape}"
        got = to_numpy(r)
        assert np.allclose(got, np_r, atol=1e-10), f"dim={dim}: values mismatch"

    print("PASS: sum_dim forward 3D")


def test_mean_dim_forward_3d():
    data = [float(x) for x in range(1, 25)]
    base, t = make_tensor(data, [2, 3, 4])
    np_t = np.array(data, dtype=np.float64).reshape(2, 3, 4)

    for dim in range(3):
        r = t.mean_dim(dim=dim)
        np_r = np_t.mean(axis=dim)
        assert r.shape == tuple(np_r.shape), f"dim={dim}: shape {r.shape} vs {np_r.shape}"
        got = to_numpy(r)
        assert np.allclose(got, np_r, atol=1e-10), f"dim={dim}: values mismatch"

    print("PASS: mean_dim forward 3D")


def test_negative_dim():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    base, t = make_tensor(data, [2, 3])
    np_t = np.array(data).reshape(2, 3)

    r = t.sum_dim(dim=-1)
    np_r = np_t.sum(axis=-1)
    got = to_numpy(r)
    assert np.allclose(got, np_r, atol=1e-10)

    r2 = t.sum_dim(dim=-2)
    np_r2 = np_t.sum(axis=-2)
    got2 = to_numpy(r2)
    assert np.allclose(got2, np_r2, atol=1e-10)

    print("PASS: negative dim")


def test_keepdim():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    base, t = make_tensor(data, [2, 3])
    np_t = np.array(data).reshape(2, 3)

    r = t.sum_dim(dim=0, keepdim=True)
    assert r.shape == (1, 3), f"Expected (1, 3), got {r.shape}"
    np_r = np_t.sum(axis=0, keepdims=True)
    got = to_numpy(r)
    assert np.allclose(got, np_r, atol=1e-10)

    r1 = t.sum_dim(dim=1, keepdim=True)
    assert r1.shape == (2, 1), f"Expected (2, 1), got {r1.shape}"
    np_r1 = np_t.sum(axis=1, keepdims=True)
    got1 = to_numpy(r1)
    assert np.allclose(got1, np_r1, atol=1e-10)

    r2 = t.mean_dim(dim=0, keepdim=True)
    assert r2.shape == (1, 3), f"Expected (1, 3), got {r2.shape}"

    print("PASS: keepdim")


def test_1d_reduction():
    data = [1.0, 2.0, 3.0, 4.0]
    t = rp.values(data, dtype='float64', device='cpu')

    r = t.sum_dim(dim=0)
    assert r.shape == (1,), f"Expected (1,), got {r.shape}"
    assert abs(r.sum_all() - 10.0) < 1e-10

    r_k = t.sum_dim(dim=0, keepdim=True)
    assert r_k.shape == (1,), f"Expected (1,), got {r_k.shape}"

    print("PASS: 1D reduction")


def test_op_sum_dim_forward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    base, t = make_tensor(data, [2, 3])
    np_t = np.array(data).reshape(2, 3)

    r = t.op_sum_dim(dim=1)
    np_r = np_t.sum(axis=1)
    assert r.shape == (2,), f"Expected (2,), got {r.shape}"
    got = to_numpy(r)
    assert np.allclose(got, np_r, atol=1e-10)

    print("PASS: op_sum_dim forward")


def test_op_mean_dim_forward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    base, t = make_tensor(data, [2, 3])
    np_t = np.array(data).reshape(2, 3)

    r = t.op_mean_dim(dim=0)
    np_r = np_t.mean(axis=0)
    assert r.shape == (3,), f"Expected (3,), got {r.shape}"
    got = to_numpy(r)
    assert np.allclose(got, np_r, atol=1e-10)

    print("PASS: op_mean_dim forward")


def test_sum_dim_backward_analytical():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    base, t = make_tensor(data, [2, 3], requires_grad=True)

    out = t.op_sum_dim(dim=1)
    loss = out.op_sum_all()
    loss.backward()

    grad = t.grad
    assert grad is not None
    assert grad.shape == (2, 3)
    got = to_numpy(grad)
    expected = np.ones((2, 3))
    assert np.allclose(got, expected, atol=1e-10), f"sum_dim backward: {got} vs {expected}"

    print("PASS: sum_dim backward analytical")


def test_sum_dim_backward_dim0():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    base, t = make_tensor(data, [2, 3], requires_grad=True)

    out = t.op_sum_dim(dim=0)
    loss = out.op_sum_all()
    loss.backward()

    grad = t.grad
    assert grad is not None
    assert grad.shape == (2, 3)
    got = to_numpy(grad)
    expected = np.ones((2, 3))
    assert np.allclose(got, expected, atol=1e-10), f"sum_dim(0) backward: {got} vs {expected}"

    print("PASS: sum_dim backward dim=0")


def test_mean_dim_backward_analytical():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    base, t = make_tensor(data, [2, 3], requires_grad=True)

    out = t.op_mean_dim(dim=1)
    loss = out.op_sum_all()
    loss.backward()

    grad = t.grad
    assert grad is not None
    assert grad.shape == (2, 3)
    got = to_numpy(grad)
    expected = np.full((2, 3), 1.0 / 3.0)
    assert np.allclose(got, expected, atol=1e-10), f"mean_dim backward: {got} vs {expected}"

    print("PASS: mean_dim backward analytical")


def test_mean_dim_backward_dim0():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    base, t = make_tensor(data, [2, 3], requires_grad=True)

    out = t.op_mean_dim(dim=0)
    loss = out.op_sum_all()
    loss.backward()

    grad = t.grad
    assert grad is not None
    assert grad.shape == (2, 3)
    got = to_numpy(grad)
    expected = np.full((2, 3), 1.0 / 2.0)
    assert np.allclose(got, expected, atol=1e-10), f"mean_dim(0) backward: {got} vs {expected}"

    print("PASS: mean_dim backward dim=0")


def test_numerical_gradient_sum_dim():
    eps = 1e-5
    np_x = np.array([1.5, -0.5, 2.3, 0.7, -1.2, 3.1], dtype=np.float64).reshape(2, 3)

    for dim in range(2):
        base, t = make_tensor(np_x.flatten(), [2, 3], requires_grad=True)
        out = t.op_sum_dim(dim=dim)
        loss = out.op_sum_all()
        loss.backward()
        got = to_numpy(t.grad)

        numerical = np.zeros_like(np_x)
        for idx in np.ndindex(*np_x.shape):
            x_plus = np_x.copy(); x_plus[idx] += eps
            x_minus = np_x.copy(); x_minus[idx] -= eps
            numerical[idx] = (x_plus.sum(axis=dim).sum() - x_minus.sum(axis=dim).sum()) / (2 * eps)

        assert np.allclose(got, numerical, atol=1e-4), \
            f"Numerical check failed for sum_dim(dim={dim}): autograd={got}, numerical={numerical}"

    print("PASS: numerical gradient sum_dim")


def test_numerical_gradient_mean_dim():
    eps = 1e-5
    np_x = np.array([1.5, -0.5, 2.3, 0.7, -1.2, 3.1], dtype=np.float64).reshape(2, 3)

    for dim in range(2):
        base, t = make_tensor(np_x.flatten(), [2, 3], requires_grad=True)
        out = t.op_mean_dim(dim=dim)
        loss = out.op_sum_all()
        loss.backward()
        got = to_numpy(t.grad)

        numerical = np.zeros_like(np_x)
        for idx in np.ndindex(*np_x.shape):
            x_plus = np_x.copy(); x_plus[idx] += eps
            x_minus = np_x.copy(); x_minus[idx] -= eps
            numerical[idx] = (x_plus.mean(axis=dim).sum() - x_minus.mean(axis=dim).sum()) / (2 * eps)

        assert np.allclose(got, numerical, atol=1e-4), \
            f"Numerical check failed for mean_dim(dim={dim}): autograd={got}, numerical={numerical}"

    print("PASS: numerical gradient mean_dim")


def test_chain_square_sum_dim():
    np_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64).reshape(2, 3)

    base, t = make_tensor(np_x.flatten(), [2, 3], requires_grad=True)

    sq = t.op_square()
    out = sq.op_sum_dim(dim=1)
    loss = out.op_sum_all()
    loss.backward()

    got = to_numpy(t.grad)
    expected = 2.0 * np_x
    assert np.allclose(got, expected, atol=1e-10), f"chain: {got} vs {expected}"

    print("PASS: chain square -> sum_dim -> sum_all backward")


def test_chain_square_mean_dim():
    np_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64).reshape(2, 3)

    base, t = make_tensor(np_x.flatten(), [2, 3], requires_grad=True)

    sq = t.op_square()
    out = sq.op_mean_dim(dim=1)
    loss = out.op_sum_all()
    loss.backward()

    got = to_numpy(t.grad)
    expected = 2.0 * np_x / 3.0
    assert np.allclose(got, expected, atol=1e-10), f"chain: {got} vs {expected}"

    print("PASS: chain square -> mean_dim -> sum_all backward")


def test_3d_backward():
    np_x = np.arange(1, 25, dtype=np.float64).reshape(2, 3, 4)

    for dim in range(3):
        base, t = make_tensor(np_x.flatten(), [2, 3, 4], requires_grad=True)
        out = t.op_sum_dim(dim=dim)
        loss = out.op_sum_all()
        loss.backward()
        got = to_numpy(t.grad)
        expected = np.ones_like(np_x)
        assert np.allclose(got, expected, atol=1e-10), f"3D sum_dim(dim={dim}): {got} vs {expected}"

    for dim in range(3):
        base, t = make_tensor(np_x.flatten(), [2, 3, 4], requires_grad=True)
        out = t.op_mean_dim(dim=dim)
        loss = out.op_sum_all()
        loss.backward()
        got = to_numpy(t.grad)
        expected = np.full_like(np_x, 1.0 / np_x.shape[dim])
        assert np.allclose(got, expected, atol=1e-10), f"3D mean_dim(dim={dim}): {got} vs {expected}"

    print("PASS: 3D backward")


def test_float32():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    base, t = make_tensor(data, [2, 3], dtype='float32')
    np_t = np.array(data, dtype=np.float32).reshape(2, 3)

    r = t.sum_dim(dim=1)
    got = to_numpy(r)
    expected = np_t.sum(axis=1)
    assert np.allclose(got, expected, atol=1e-5), f"f32 sum: {got} vs {expected}"

    base2, t2 = make_tensor(data, [2, 3], dtype='float32', requires_grad=True)
    out = t2.op_sum_dim(dim=0)
    loss = out.op_sum_all()
    loss.backward()
    grad = t2.grad
    assert grad is not None
    got_grad = to_numpy(grad)
    assert np.allclose(got_grad, np.ones((2, 3)), atol=1e-5)

    print("PASS: float32")


def test_keepdim_backward():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    base, t = make_tensor(data, [2, 3], requires_grad=True)

    out = t.op_sum_dim(dim=1, keepdim=True)
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"
    loss = out.op_sum_all()
    loss.backward()

    grad = t.grad
    assert grad is not None
    assert grad.shape == (2, 3)
    got = to_numpy(grad)
    expected = np.ones((2, 3))
    assert np.allclose(got, expected, atol=1e-10)

    print("PASS: keepdim backward")


if __name__ == '__main__':
    test_sum_dim_forward_2d()
    test_mean_dim_forward_2d()
    test_sum_dim_forward_3d()
    test_mean_dim_forward_3d()
    test_negative_dim()
    test_keepdim()
    test_1d_reduction()
    test_op_sum_dim_forward()
    test_op_mean_dim_forward()
    test_sum_dim_backward_analytical()
    test_sum_dim_backward_dim0()
    test_mean_dim_backward_analytical()
    test_mean_dim_backward_dim0()
    test_numerical_gradient_sum_dim()
    test_numerical_gradient_mean_dim()
    test_chain_square_sum_dim()
    test_chain_square_mean_dim()
    test_3d_backward()
    test_float32()
    test_keepdim_backward()
    print("\nAll sum_dim/mean_dim tests passed!")
