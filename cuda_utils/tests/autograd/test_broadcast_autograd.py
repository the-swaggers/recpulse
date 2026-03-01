import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp

passed = 0
failed = 0
EPS = 1e-6
ATOL = 1e-4


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)


def read_flat_values(tensor, total_size):
    flat_base = tensor
    flat = flat_base.reshape([total_size])
    vals = []
    for i in range(total_size):
        s = flat.slice(start=[i], stop=[i + 1])
        vals.append(s.sum_all())
    return vals


def make_tensor(data, shape=None):
    base = rp.values(data, dtype='float64', device='cpu')
    if shape is not None:
        view = base.reshape(shape)
        return base, view
    return base, base


def compute_numerical_grad(forward_fn, data_list, shape, total_size):
    grads = []
    for i in range(total_size):
        data_plus = list(data_list)
        data_plus[i] += EPS
        data_minus = list(data_list)
        data_minus[i] -= EPS

        loss_plus = forward_fn(data_plus, shape)
        loss_minus = forward_fn(data_minus, shape)

        grads.append((loss_plus - loss_minus) / (2.0 * EPS))
    return grads


def compare_grads(analytical, numerical, name, atol=ATOL):
    if len(analytical) != len(numerical):
        check(name, False, f"length mismatch: {len(analytical)} vs {len(numerical)}")
        return
    max_diff = 0.0
    for i in range(len(analytical)):
        diff = abs(analytical[i] - numerical[i])
        if diff > max_diff:
            max_diff = diff
    ok = max_diff < atol
    detail = f"max_diff={max_diff:.2e}"
    check(name, ok, detail)


def test_add_broadcast_3x1_1x4():
    print("\n--- test_add: [3,1] + [1,4] ---")

    a_data = [1.0, 2.0, 3.0]
    b_data = [10.0, 20.0, 30.0, 40.0]

    a_base, a = make_tensor(a_data, [3, 1])
    a.requires_grad_(True)
    b_base, b = make_tensor(b_data, [1, 4])
    b.requires_grad_(True)

    c = a.op_add(b)
    loss = c.op_sum_all()
    loss.backward()

    grad_a = a.grad
    grad_b = b.grad
    check("a.grad exists", grad_a is not None)
    check("b.grad exists", grad_b is not None)
    check("a.grad shape", grad_a.shape == (3, 1))
    check("b.grad shape", grad_b.shape == (1, 4))

    analytical_a = read_flat_values(grad_a, 3)
    analytical_b = read_flat_values(grad_b, 4)

    def forward_a(data, shape):
        ta_base, ta = make_tensor(data, shape)
        tb_base, tb = make_tensor(b_data, [1, 4])
        out = ta.op_add(tb)
        return out.op_sum_all().sum_all()

    def forward_b(data, shape):
        ta_base, ta = make_tensor(a_data, [3, 1])
        tb_base, tb = make_tensor(data, shape)
        out = ta.op_add(tb)
        return out.op_sum_all().sum_all()

    numerical_a = compute_numerical_grad(forward_a, a_data, [3, 1], 3)
    numerical_b = compute_numerical_grad(forward_b, b_data, [1, 4], 4)

    compare_grads(analytical_a, numerical_a, "a grad numerical check")
    compare_grads(analytical_b, numerical_b, "b grad numerical check")


def test_sub_broadcast_3x1_1x4():
    print("\n--- test_sub: [3,1] - [1,4] ---")

    a_data = [5.0, 6.0, 7.0]
    b_data = [1.0, 2.0, 3.0, 4.0]

    a_base, a = make_tensor(a_data, [3, 1])
    a.requires_grad_(True)
    b_base, b = make_tensor(b_data, [1, 4])
    b.requires_grad_(True)

    c = a.op_sub(b)
    loss = c.op_sum_all()
    loss.backward()

    grad_a = a.grad
    grad_b = b.grad
    check("a.grad exists", grad_a is not None)
    check("b.grad exists", grad_b is not None)
    check("a.grad shape", grad_a.shape == (3, 1))
    check("b.grad shape", grad_b.shape == (1, 4))

    analytical_a = read_flat_values(grad_a, 3)
    analytical_b = read_flat_values(grad_b, 4)

    def forward_a(data, shape):
        ta_base, ta = make_tensor(data, shape)
        tb_base, tb = make_tensor(b_data, [1, 4])
        out = ta.op_sub(tb)
        return out.op_sum_all().sum_all()

    def forward_b(data, shape):
        ta_base, ta = make_tensor(a_data, [3, 1])
        tb_base, tb = make_tensor(data, shape)
        out = ta.op_sub(tb)
        return out.op_sum_all().sum_all()

    numerical_a = compute_numerical_grad(forward_a, a_data, [3, 1], 3)
    numerical_b = compute_numerical_grad(forward_b, b_data, [1, 4], 4)

    compare_grads(analytical_a, numerical_a, "a grad numerical check")
    compare_grads(analytical_b, numerical_b, "b grad numerical check")


def test_mul_broadcast_3x1_1x4():
    print("\n--- test_mul: [3,1] * [1,4] ---")

    a_data = [2.0, 3.0, 4.0]
    b_data = [5.0, 6.0, 7.0, 8.0]

    a_base, a = make_tensor(a_data, [3, 1])
    a.requires_grad_(True)
    b_base, b = make_tensor(b_data, [1, 4])
    b.requires_grad_(True)

    c = a.op_mul(b)
    loss = c.op_sum_all()
    loss.backward()

    grad_a = a.grad
    grad_b = b.grad
    check("a.grad exists", grad_a is not None)
    check("b.grad exists", grad_b is not None)
    check("a.grad shape", grad_a.shape == (3, 1))
    check("b.grad shape", grad_b.shape == (1, 4))

    analytical_a = read_flat_values(grad_a, 3)
    analytical_b = read_flat_values(grad_b, 4)

    def forward_a(data, shape):
        ta_base, ta = make_tensor(data, shape)
        tb_base, tb = make_tensor(b_data, [1, 4])
        out = ta.op_mul(tb)
        return out.op_sum_all().sum_all()

    def forward_b(data, shape):
        ta_base, ta = make_tensor(a_data, [3, 1])
        tb_base, tb = make_tensor(data, shape)
        out = ta.op_mul(tb)
        return out.op_sum_all().sum_all()

    numerical_a = compute_numerical_grad(forward_a, a_data, [3, 1], 3)
    numerical_b = compute_numerical_grad(forward_b, b_data, [1, 4], 4)

    compare_grads(analytical_a, numerical_a, "a grad numerical check")
    compare_grads(analytical_b, numerical_b, "b grad numerical check")


def test_div_broadcast_3x1_1x4():
    print("\n--- test_div: [3,1] / [1,4] ---")

    a_data = [12.0, 24.0, 36.0]
    b_data = [2.0, 3.0, 4.0, 6.0]

    a_base, a = make_tensor(a_data, [3, 1])
    a.requires_grad_(True)
    b_base, b = make_tensor(b_data, [1, 4])
    b.requires_grad_(True)

    c = a.op_div(b)
    loss = c.op_sum_all()
    loss.backward()

    grad_a = a.grad
    grad_b = b.grad
    check("a.grad exists", grad_a is not None)
    check("b.grad exists", grad_b is not None)
    check("a.grad shape", grad_a.shape == (3, 1))
    check("b.grad shape", grad_b.shape == (1, 4))

    analytical_a = read_flat_values(grad_a, 3)
    analytical_b = read_flat_values(grad_b, 4)

    def forward_a(data, shape):
        ta_base, ta = make_tensor(data, shape)
        tb_base, tb = make_tensor(b_data, [1, 4])
        out = ta.op_div(tb)
        return out.op_sum_all().sum_all()

    def forward_b(data, shape):
        ta_base, ta = make_tensor(a_data, [3, 1])
        tb_base, tb = make_tensor(data, shape)
        out = ta.op_div(tb)
        return out.op_sum_all().sum_all()

    numerical_a = compute_numerical_grad(forward_a, a_data, [3, 1], 3)
    numerical_b = compute_numerical_grad(forward_b, b_data, [1, 4], 4)

    compare_grads(analytical_a, numerical_a, "a grad numerical check")
    compare_grads(analytical_b, numerical_b, "b grad numerical check")


def test_mul_broadcast_different_ndim():
    print("\n--- test_mul: [4] * [2,4] (different ndim) ---")

    a_data = [1.0, 2.0, 3.0, 4.0]
    b_data = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    a = rp.values(a_data, dtype='float64', device='cpu')
    a.requires_grad_(True)
    b_base, b = make_tensor(b_data, [2, 4])
    b.requires_grad_(True)

    c = a.op_mul(b)
    loss = c.op_sum_all()
    loss.backward()

    grad_a = a.grad
    grad_b = b.grad
    check("a.grad exists", grad_a is not None)
    check("b.grad exists", grad_b is not None)
    check("a.grad shape", grad_a.shape == (4,))
    check("b.grad shape", grad_b.shape == (2, 4))

    analytical_a = read_flat_values(grad_a, 4)
    analytical_b = read_flat_values(grad_b, 8)

    def forward_a(data, shape):
        ta = rp.values(data, dtype='float64', device='cpu')
        tb_base, tb = make_tensor(b_data, [2, 4])
        out = ta.op_mul(tb)
        return out.op_sum_all().sum_all()

    def forward_b(data, shape):
        ta = rp.values(a_data, dtype='float64', device='cpu')
        tb_base, tb = make_tensor(data, shape)
        out = ta.op_mul(tb)
        return out.op_sum_all().sum_all()

    numerical_a = compute_numerical_grad(forward_a, a_data, [4], 4)
    numerical_b = compute_numerical_grad(forward_b, b_data, [2, 4], 8)

    compare_grads(analytical_a, numerical_a, "a grad numerical check")
    compare_grads(analytical_b, numerical_b, "b grad numerical check")


def test_add_broadcast_scalar():
    print("\n--- test_add: [1] + [5] (scalar broadcast) ---")

    a_data = [7.0]
    b_data = [1.0, 2.0, 3.0, 4.0, 5.0]

    a = rp.values(a_data, dtype='float64', device='cpu')
    a.requires_grad_(True)
    b = rp.values(b_data, dtype='float64', device='cpu')
    b.requires_grad_(True)

    c = a.op_add(b)
    loss = c.op_sum_all()
    loss.backward()

    grad_a = a.grad
    grad_b = b.grad
    check("a.grad exists", grad_a is not None)
    check("b.grad exists", grad_b is not None)
    check("a.grad shape", grad_a.shape == (1,))
    check("b.grad shape", grad_b.shape == (5,))

    analytical_a = read_flat_values(grad_a, 1)
    analytical_b = read_flat_values(grad_b, 5)

    def forward_a(data, shape):
        ta = rp.values(data, dtype='float64', device='cpu')
        tb = rp.values(b_data, dtype='float64', device='cpu')
        out = ta.op_add(tb)
        return out.op_sum_all().sum_all()

    def forward_b(data, shape):
        ta = rp.values(a_data, dtype='float64', device='cpu')
        tb = rp.values(data, dtype='float64', device='cpu')
        out = ta.op_add(tb)
        return out.op_sum_all().sum_all()

    numerical_a = compute_numerical_grad(forward_a, a_data, [1], 1)
    numerical_b = compute_numerical_grad(forward_b, b_data, [5], 5)

    compare_grads(analytical_a, numerical_a, "a grad numerical check")
    compare_grads(analytical_b, numerical_b, "b grad numerical check")


def test_power_broadcast_3x1_1x3():
    print("\n--- test_power: [3,1] ** [1,3] ---")

    a_data = [2.0, 3.0, 4.0]
    b_data = [1.0, 2.0, 3.0]

    a_base, a = make_tensor(a_data, [3, 1])
    a.requires_grad_(True)
    b_base, b = make_tensor(b_data, [1, 3])
    b.requires_grad_(True)

    c = a.op_power(b)
    loss = c.op_sum_all()
    loss.backward()

    grad_a = a.grad
    grad_b = b.grad
    check("a.grad exists", grad_a is not None)
    check("b.grad exists", grad_b is not None)
    check("a.grad shape", grad_a.shape == (3, 1))
    check("b.grad shape", grad_b.shape == (1, 3))

    analytical_a = read_flat_values(grad_a, 3)
    analytical_b = read_flat_values(grad_b, 3)

    def forward_a(data, shape):
        ta_base, ta = make_tensor(data, shape)
        tb_base, tb = make_tensor(b_data, [1, 3])
        out = ta.op_power(tb)
        return out.op_sum_all().sum_all()

    def forward_b(data, shape):
        ta_base, ta = make_tensor(a_data, [3, 1])
        tb_base, tb = make_tensor(data, shape)
        out = ta.op_power(tb)
        return out.op_sum_all().sum_all()

    numerical_a = compute_numerical_grad(forward_a, a_data, [3, 1], 3)
    numerical_b = compute_numerical_grad(forward_b, b_data, [1, 3], 3)

    compare_grads(analytical_a, numerical_a, "a grad numerical check")
    compare_grads(analytical_b, numerical_b, "b grad numerical check")


def test_mul_broadcast_chain():
    print("\n--- test_mul chain: sum((a * b) * c) where a=[2,1], b=[1,3], c=[2,3] ---")

    a_data = [2.0, 3.0]
    b_data = [4.0, 5.0, 6.0]
    c_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    a_base, a = make_tensor(a_data, [2, 1])
    a.requires_grad_(True)
    b_base, b = make_tensor(b_data, [1, 3])
    b.requires_grad_(True)
    c_base, c = make_tensor(c_data, [2, 3])
    c.requires_grad_(True)

    ab = a.op_mul(b)
    abc = ab.op_mul(c)
    loss = abc.op_sum_all()
    loss.backward()

    grad_a = a.grad
    grad_b = b.grad
    grad_c = c.grad
    check("a.grad exists", grad_a is not None)
    check("b.grad exists", grad_b is not None)
    check("c.grad exists", grad_c is not None)
    check("a.grad shape", grad_a.shape == (2, 1))
    check("b.grad shape", grad_b.shape == (1, 3))
    check("c.grad shape", grad_c.shape == (2, 3))

    analytical_a = read_flat_values(grad_a, 2)
    analytical_b = read_flat_values(grad_b, 3)
    analytical_c = read_flat_values(grad_c, 6)

    def forward_a(data, shape):
        ta_base, ta = make_tensor(data, shape)
        tb_base, tb = make_tensor(b_data, [1, 3])
        tc_base, tc = make_tensor(c_data, [2, 3])
        tab = ta.op_mul(tb)
        tabc = tab.op_mul(tc)
        return tabc.op_sum_all().sum_all()

    def forward_b(data, shape):
        ta_base, ta = make_tensor(a_data, [2, 1])
        tb_base, tb = make_tensor(data, shape)
        tc_base, tc = make_tensor(c_data, [2, 3])
        tab = ta.op_mul(tb)
        tabc = tab.op_mul(tc)
        return tabc.op_sum_all().sum_all()

    def forward_c(data, shape):
        ta_base, ta = make_tensor(a_data, [2, 1])
        tb_base, tb = make_tensor(b_data, [1, 3])
        tc_base, tc = make_tensor(data, shape)
        tab = ta.op_mul(tb)
        tabc = tab.op_mul(tc)
        return tabc.op_sum_all().sum_all()

    numerical_a = compute_numerical_grad(forward_a, a_data, [2, 1], 2)
    numerical_b = compute_numerical_grad(forward_b, b_data, [1, 3], 3)
    numerical_c = compute_numerical_grad(forward_c, c_data, [2, 3], 6)

    compare_grads(analytical_a, numerical_a, "a grad numerical check")
    compare_grads(analytical_b, numerical_b, "b grad numerical check")
    compare_grads(analytical_c, numerical_c, "c grad numerical check")


def test_div_broadcast_different_ndim():
    print("\n--- test_div: [3] / [2,3] (different ndim) ---")

    a_data = [6.0, 12.0, 18.0]
    b_data = [1.0, 2.0, 3.0, 4.0, 6.0, 9.0]

    a = rp.values(a_data, dtype='float64', device='cpu')
    a.requires_grad_(True)
    b_base, b = make_tensor(b_data, [2, 3])
    b.requires_grad_(True)

    c = a.op_div(b)
    loss = c.op_sum_all()
    loss.backward()

    grad_a = a.grad
    grad_b = b.grad
    check("a.grad exists", grad_a is not None)
    check("b.grad exists", grad_b is not None)
    check("a.grad shape", grad_a.shape == (3,))
    check("b.grad shape", grad_b.shape == (2, 3))

    analytical_a = read_flat_values(grad_a, 3)
    analytical_b = read_flat_values(grad_b, 6)

    def forward_a(data, shape):
        ta = rp.values(data, dtype='float64', device='cpu')
        tb_base, tb = make_tensor(b_data, [2, 3])
        out = ta.op_div(tb)
        return out.op_sum_all().sum_all()

    def forward_b(data, shape):
        ta = rp.values(a_data, dtype='float64', device='cpu')
        tb_base, tb = make_tensor(data, shape)
        out = ta.op_div(tb)
        return out.op_sum_all().sum_all()

    numerical_a = compute_numerical_grad(forward_a, a_data, [3], 3)
    numerical_b = compute_numerical_grad(forward_b, b_data, [2, 3], 6)

    compare_grads(analytical_a, numerical_a, "a grad numerical check")
    compare_grads(analytical_b, numerical_b, "b grad numerical check")


def test_sub_broadcast_scalar_like():
    print("\n--- test_sub: [1] - [2,3] (scalar-like broadcast) ---")

    a_data = [10.0]
    b_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    a = rp.values(a_data, dtype='float64', device='cpu')
    a.requires_grad_(True)
    b_base, b = make_tensor(b_data, [2, 3])
    b.requires_grad_(True)

    c = a.op_sub(b)
    loss = c.op_sum_all()
    loss.backward()

    grad_a = a.grad
    grad_b = b.grad
    check("a.grad exists", grad_a is not None)
    check("b.grad exists", grad_b is not None)
    check("a.grad shape", grad_a.shape == (1,))
    check("b.grad shape", grad_b.shape == (2, 3))

    analytical_a = read_flat_values(grad_a, 1)
    analytical_b = read_flat_values(grad_b, 6)

    def forward_a(data, shape):
        ta = rp.values(data, dtype='float64', device='cpu')
        tb_base, tb = make_tensor(b_data, [2, 3])
        out = ta.op_sub(tb)
        return out.op_sum_all().sum_all()

    def forward_b(data, shape):
        ta = rp.values(a_data, dtype='float64', device='cpu')
        tb_base, tb = make_tensor(data, shape)
        out = ta.op_sub(tb)
        return out.op_sum_all().sum_all()

    numerical_a = compute_numerical_grad(forward_a, a_data, [1], 1)
    numerical_b = compute_numerical_grad(forward_b, b_data, [2, 3], 6)

    compare_grads(analytical_a, numerical_a, "a grad numerical check")
    compare_grads(analytical_b, numerical_b, "b grad numerical check")


if __name__ == "__main__":
    print("=" * 60)
    print("Broadcasting Autograd Tests (Numerical Gradient Check)")
    print("=" * 60)

    test_add_broadcast_3x1_1x4()
    test_sub_broadcast_3x1_1x4()
    test_mul_broadcast_3x1_1x4()
    test_div_broadcast_3x1_1x4()
    test_mul_broadcast_different_ndim()
    test_add_broadcast_scalar()
    test_power_broadcast_3x1_1x3()
    test_mul_broadcast_chain()
    test_div_broadcast_different_ndim()
    test_sub_broadcast_scalar_like()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("All broadcast autograd tests passed.")
        sys.exit(0)
