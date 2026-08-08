import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
from recpulse.optim import SGD, Adam


def read_scalar(t):
    return t.sum_all()


def test_sgd_basic():
    x = rp.values([5.0], dtype='float64', device='cpu')
    x.requires_grad_(True)
    opt = SGD([x], lr=0.1)

    for _ in range(100):
        opt.zero_grad()
        sq = x.op_square()
        loss = sq.op_sum_all()
        loss.backward()
        opt.step()

    val = read_scalar(x)
    assert abs(val) < 1e-4, f"Expected ~0, got {val}"
    print("PASS: SGD minimizes x^2")


def test_sgd_momentum():
    x = rp.values([5.0], dtype='float64', device='cpu')
    x.requires_grad_(True)
    opt = SGD([x], lr=0.01, momentum=0.9)

    for _ in range(200):
        opt.zero_grad()
        sq = x.op_square()
        loss = sq.op_sum_all()
        loss.backward()
        opt.step()

    val = read_scalar(x)
    assert abs(val) < 0.1, f"Expected ~0, got {val}"
    print("PASS: SGD with momentum minimizes x^2")


def test_adam_basic():
    x = rp.values([5.0], dtype='float64', device='cpu')
    x.requires_grad_(True)
    opt = Adam([x], lr=0.1)

    for _ in range(200):
        opt.zero_grad()
        sq = x.op_square()
        loss = sq.op_sum_all()
        loss.backward()
        opt.step()

    val = read_scalar(x)
    assert abs(val) < 1e-2, f"Expected ~0, got {val}"
    print("PASS: Adam minimizes x^2")


def test_multi_param():
    x = rp.values([3.0], dtype='float64', device='cpu')
    x.requires_grad_(True)
    y = rp.values([4.0], dtype='float64', device='cpu')
    y.requires_grad_(True)

    opt = SGD([x, y], lr=0.1)

    for _ in range(100):
        opt.zero_grad()
        x_sq = x.op_square()
        y_sq = y.op_square()
        s = x_sq.op_add(y_sq)
        loss = s.op_sum_all()
        loss.backward()
        opt.step()

    xv = read_scalar(x)
    yv = read_scalar(y)
    assert abs(xv) < 1e-4, f"Expected x~0, got {xv}"
    assert abs(yv) < 1e-4, f"Expected y~0, got {yv}"
    print("PASS: multi-param SGD minimizes x^2 + y^2")


def test_zero_grad_works():
    x = rp.values([3.0], dtype='float64', device='cpu')
    x.requires_grad_(True)
    opt = SGD([x], lr=0.1)

    sq = x.op_square()
    loss = sq.op_sum_all()
    loss.backward()
    assert x.has_grad is True

    opt.zero_grad()
    assert x.has_grad is False
    print("PASS: optimizer zero_grad clears gradients")


def test_adam_multi_param():
    x = rp.values([3.0], dtype='float64', device='cpu')
    x.requires_grad_(True)
    y = rp.values([-4.0], dtype='float64', device='cpu')
    y.requires_grad_(True)

    opt = Adam([x, y], lr=0.1)

    for _ in range(200):
        opt.zero_grad()
        x_sq = x.op_square()
        y_sq = y.op_square()
        s = x_sq.op_add(y_sq)
        loss = s.op_sum_all()
        loss.backward()
        opt.step()

    xv = read_scalar(x)
    yv = read_scalar(y)
    assert abs(xv) < 1e-2, f"Expected x~0, got {xv}"
    assert abs(yv) < 1e-2, f"Expected y~0, got {yv}"
    print("PASS: Adam multi-param minimizes x^2 + y^2")


def test_sgd_weight_decay():
    x_wd = rp.values([5.0], dtype='float64', device='cpu')
    x_wd.requires_grad_(True)
    opt_wd = SGD([x_wd], lr=0.01, weight_decay=0.5)

    x_no_wd = rp.values([5.0], dtype='float64', device='cpu')
    x_no_wd.requires_grad_(True)
    opt_no_wd = SGD([x_no_wd], lr=0.01, weight_decay=0.0)

    for _ in range(20):
        opt_wd.zero_grad()
        sq_wd = x_wd.op_square()
        loss_wd = sq_wd.op_sum_all()
        loss_wd.backward()
        opt_wd.step()

        opt_no_wd.zero_grad()
        sq_no_wd = x_no_wd.op_square()
        loss_no_wd = sq_no_wd.op_sum_all()
        loss_no_wd.backward()
        opt_no_wd.step()

    val_wd = abs(read_scalar(x_wd))
    val_no_wd = abs(read_scalar(x_no_wd))
    return val_wd < val_no_wd, f"weight_decay={val_wd:.6f} < no_decay={val_no_wd:.6f}"


def test_sgd_nesterov():
    x = rp.values([5.0], dtype='float64', device='cpu')
    x.requires_grad_(True)
    opt = SGD([x], lr=0.01, momentum=0.9, nesterov=True)

    for _ in range(200):
        opt.zero_grad()
        sq = x.op_square()
        loss = sq.op_sum_all()
        loss.backward()
        opt.step()

    val = read_scalar(x)
    return abs(val) < 0.1, f"val={val:.6f}"


def test_adam_convergence_rate():
    xa_big = rp.values([100.0], dtype='float64', device='cpu')
    xa_big.requires_grad_(True)
    xa_small = rp.values([0.01], dtype='float64', device='cpu')
    xa_small.requires_grad_(True)
    opt_adam = Adam([xa_big, xa_small], lr=0.5)

    xs_big = rp.values([100.0], dtype='float64', device='cpu')
    xs_big.requires_grad_(True)
    xs_small = rp.values([0.01], dtype='float64', device='cpu')
    xs_small.requires_grad_(True)
    opt_sgd = SGD([xs_big, xs_small], lr=0.001)

    steps = 100
    for _ in range(steps):
        opt_adam.zero_grad()
        sq_ab = xa_big.op_square()
        sq_as = xa_small.op_square()
        s_a = sq_ab.op_add(sq_as)
        loss_a = s_a.op_sum_all()
        loss_a.backward()
        opt_adam.step()

        opt_sgd.zero_grad()
        sq_sb = xs_big.op_square()
        sq_ss = xs_small.op_square()
        s_s = sq_sb.op_add(sq_ss)
        loss_s = s_s.op_sum_all()
        loss_s.backward()
        opt_sgd.step()

    adam_big = abs(read_scalar(xa_big))
    adam_small = abs(read_scalar(xa_small))
    sgd_big = abs(read_scalar(xs_big))
    sgd_small = abs(read_scalar(xs_small))
    adam_loss = adam_big**2 + adam_small**2
    sgd_loss = sgd_big**2 + sgd_small**2
    return adam_loss < sgd_loss, f"adam_loss={adam_loss:.6f} < sgd_loss={sgd_loss:.6f}"


def test_adam_amsgrad():
    x = rp.values([5.0], dtype='float64', device='cpu')
    x.requires_grad_(True)
    opt = Adam([x], lr=0.1, amsgrad=True)

    for _ in range(200):
        opt.zero_grad()
        sq = x.op_square()
        loss = sq.op_sum_all()
        loss.backward()
        opt.step()

    val = read_scalar(x)
    return abs(val) < 0.05, f"val={val:.6f}"


def test_sgd_50_steps():
    x = rp.values([3.0], dtype='float64', device='cpu')
    x.requires_grad_(True)
    opt = SGD([x], lr=0.1)

    for _ in range(50):
        opt.zero_grad()
        sq = x.op_square()
        loss = sq.op_sum_all()
        loss.backward()
        opt.step()

    val = read_scalar(x)
    return abs(val) < 1e-3, f"val={val:.8f}"


def test_float64_optimizer():
    x = rp.values([4.0], dtype='float64', device='cpu')
    x.requires_grad_(True)
    opt = SGD([x], lr=0.1)

    for _ in range(100):
        opt.zero_grad()
        sq = x.op_square()
        loss = sq.op_sum_all()
        loss.backward()
        opt.step()

    val = read_scalar(x)
    passed = abs(val) < 1e-6
    return passed, f"val={val:.10f}, dtype={x.dtype}"


def test_zero_learning_rate():
    x = rp.values([5.0], dtype='float64', device='cpu')
    x.requires_grad_(True)
    opt = SGD([x], lr=0.0)

    initial_val = read_scalar(x)

    for _ in range(10):
        opt.zero_grad()
        sq = x.op_square()
        loss = sq.op_sum_all()
        loss.backward()
        opt.step()

    final_val = read_scalar(x)
    return abs(final_val - initial_val) < 1e-15, f"initial={initial_val}, final={final_val}"


if __name__ == '__main__':
    test_sgd_basic()
    test_sgd_momentum()
    test_adam_basic()
    test_multi_param()
    test_zero_grad_works()
    test_adam_multi_param()

    passed = 0
    failed = 0
    new_tests = [
        ("SGD with weight_decay", test_sgd_weight_decay),
        ("SGD with nesterov", test_sgd_nesterov),
        ("Adam convergence rate vs SGD", test_adam_convergence_rate),
        ("Adam with amsgrad", test_adam_amsgrad),
        ("SGD 50 steps on x^2", test_sgd_50_steps),
        ("Float64 optimizer", test_float64_optimizer),
        ("Zero learning rate", test_zero_learning_rate),
    ]

    for name, test_fn in new_tests:
        try:
            ok, detail = test_fn()
            if ok:
                print(f"PASS: {name} ({detail})")
                passed += 1
            else:
                print(f"FAIL: {name} ({detail})")
                failed += 1
        except Exception as e:
            print(f"FAIL: {name} (exception: {e})")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed} new tests")
    if failed > 0:
        sys.exit(1)
    print("All optimizer tests passed!")
