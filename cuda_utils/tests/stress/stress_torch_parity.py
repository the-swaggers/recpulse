#!/usr/bin/env python3
from stress_common import *

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("SKIP: torch not available — parity suite not run")
    sys.exit(0)

RNG = np.random.default_rng(4242)
torch.manual_seed(0)


def tt(a, requires_grad=False):
    return torch.tensor(np.asarray(a, dtype=np.float64), requires_grad=requires_grad)


UNARY_PARITY = [
    ('exp', torch.exp, (-1, 1)),
    ('log', torch.log, (0.5, 3.0)),
    ('sqrt', torch.sqrt, (0.5, 3.0)),
    ('rsqrt', torch.rsqrt, (0.5, 3.0)),
    ('abs', torch.abs, (-2, 2)),
    ('sin', torch.sin, (-3, 3)),
    ('cos', torch.cos, (-3, 3)),
    ('tan', torch.tan, (-1, 1)),
    ('asin', torch.asin, (-0.95, 0.95)),
    ('acos', torch.acos, (-0.95, 0.95)),
    ('atan', torch.atan, (-3, 3)),
    ('sinh', torch.sinh, (-2, 2)),
    ('cosh', torch.cosh, (-2, 2)),
    ('tanh', torch.tanh, (-3, 3)),
    ('relu', torch.relu, (-2, 2)),
    ('sigmoid', torch.sigmoid, (-4, 4)),
    ('silu', F.silu, (-3, 3)),
    ('square', torch.square, (-2, 2)),
    ('ceil', torch.ceil, (-3, 3)),
    ('floor', torch.floor, (-3, 3)),
    ('trunc', torch.trunc, (-3, 3)),
]


def test_unary_forward_parity():
    for name, tfn, dom in UNARY_PARITY:
        a = RNG.uniform(*dom, (5, 7))
        got = to_np(getattr(t_from_np(a), name)())
        want = tfn(tt(a)).numpy()
        assert_close(got, want, rtol=1e-8, atol=1e-10, ctx=f"cpu f64 {name}")


def test_unary_forward_parity_f32_gpu():
    if not has_cuda():
        return
    for name, tfn, dom in UNARY_PARITY:
        a = RNG.uniform(*dom, (5, 7))
        got = to_np(getattr(t_from_np(a, dtype='float32', device='cuda'), name)())
        want = tfn(tt(a).float()).numpy()
        assert_close(got, want, rtol=1e-5, atol=1e-6, ctx=f"gpu f32 {name}")


def test_gelu_parity_tanh_approx():
    a = RNG.uniform(-3, 3, (5, 7))
    got = to_np(t_from_np(a).gelu())
    want_exact = F.gelu(tt(a)).numpy()
    want_tanh = F.gelu(tt(a), approximate='tanh').numpy()
    assert_close(got, want_tanh, rtol=1e-7, atol=1e-9, ctx="gelu vs torch tanh-approx")
    print(f"    (note: recpulse gelu is the tanh approximation; "
          f"max dev from exact erf gelu = {np.abs(got - want_exact).max():.1e})")


def test_binary_forward_parity():
    a = RNG.uniform(0.5, 2.0, (4, 6))
    b = RNG.uniform(0.5, 2.0, (4, 6))
    pairs = [
        ('add', to_np(t_from_np(a).add(t_from_np(b))), a + b),
        ('sub', to_np(t_from_np(a).sub(t_from_np(b))), a - b),
        ('mul', to_np(t_from_np(a).mul(t_from_np(b))), a * b),
        ('divide', to_np(t_from_np(a).divide(t_from_np(b))), a / b),
        ('power', to_np(t_from_np(a).power(t_from_np(b))), a ** b),
    ]
    for name, got, want in pairs:
        assert_close(got, want, rtol=1e-10, ctx=f"binary {name}")


def test_matmul_parity():
    for (m, k, n) in [(3, 4, 5), (16, 32, 8), (1, 7, 1), (64, 64, 64)]:
        a = RNG.uniform(-1, 1, (m, k))
        b = RNG.uniform(-1, 1, (k, n))
        got = to_np(t_from_np(a).op_matmul(t_from_np(b)))
        want = (tt(a) @ tt(b)).numpy()
        assert_close(got, want, rtol=1e-9, atol=1e-10, ctx=f"matmul {m}x{k}x{n}")
        if has_cuda():
            got_g = to_np(t_from_np(a, device='cuda').op_matmul(t_from_np(b, device='cuda')))
            assert_close(got_g, want, rtol=1e-9, atol=1e-10, ctx=f"gpu matmul {m}x{k}x{n}")


def test_reduction_parity():
    a = RNG.uniform(-2, 2, (4, 5, 6))
    t = t_from_np(a)
    assert abs(t.sum_all() - a.sum()) < 1e-8, "sum_all"
    assert abs(t.mean_all() - a.mean()) < 1e-10, "mean_all"
    for dim in (0, 1, 2):
        for keep in (False, True):
            got = to_np(t.sum_dim(dim, keep))
            want = a.sum(axis=dim, keepdims=keep)
            assert_close(got, want, rtol=1e-9, ctx=f"sum_dim({dim},{keep})")
            got = to_np(t.mean_dim(dim, keep))
            want = a.mean(axis=dim, keepdims=keep)
            assert_close(got, want, rtol=1e-9, ctx=f"mean_dim({dim},{keep})")


def test_softmax_parity():
    a = RNG.uniform(-4, 4, (5, 9))
    for dim in (-1, 0, 1):
        got = to_np(t_from_np(a).op_softmax(dim))
        want = F.softmax(tt(a), dim=dim).numpy()
        assert_close(got, want, rtol=1e-9, ctx=f"softmax dim={dim}")
        got = to_np(t_from_np(a).op_log_softmax(dim))
        want = F.log_softmax(tt(a), dim=dim).numpy()
        assert_close(got, want, rtol=1e-9, ctx=f"log_softmax dim={dim}")


def test_loss_parity():
    x = RNG.uniform(-2, 2, (6, 8))
    tgt = RNG.uniform(-2, 2, (6, 8))
    for red in ('mean', 'sum'):
        got = t_from_np(x).op_mse_loss(t_from_np(tgt), red).data()[0]
        want = F.mse_loss(tt(x), tt(tgt), reduction=red).item()
        assert abs(got - want) < 1e-8 * max(1, abs(want)), f"mse {red}: {got} vs {want}"

    p = RNG.uniform(0.02, 0.98, (6, 8))
    y = RNG.uniform(0, 1, (6, 8)).round()
    got = t_from_np(p).op_bce_loss(t_from_np(y), 'mean', 0).data()[0]
    want = F.binary_cross_entropy(tt(p), tt(y)).item()
    assert abs(got - want) < 1e-8, f"bce probs: {got} vs {want}"

    z = RNG.uniform(-3, 3, (6, 8))
    got = t_from_np(z).op_bce_loss(t_from_np(y), 'mean', 1).data()[0]
    want = F.binary_cross_entropy_with_logits(tt(z), tt(y)).item()
    assert abs(got - want) < 1e-8, f"bce logits: {got} vs {want}"

    logits = RNG.uniform(-2, 2, (5, 7))
    targets = [0, 3, 6, 2, 1]
    got = t_from_np(logits).op_cross_entropy_loss(targets, 'mean', 1).data()[0]
    want = F.cross_entropy(tt(logits), torch.tensor(targets)).item()
    assert abs(got - want) < 1e-8, f"cross_entropy: {got} vs {want}"


def test_backward_parity_mlp():
    x = RNG.uniform(-1, 1, (8, 10))
    w1 = RNG.uniform(-0.5, 0.5, (10, 16))
    w2 = RNG.uniform(-0.5, 0.5, (16, 4))
    tgt = RNG.uniform(-1, 1, (8, 4))

    xr = t_from_np(x, requires_grad=True)
    w1r = t_from_np(w1, requires_grad=True)
    w2r = t_from_np(w2, requires_grad=True)
    tgr = t_from_np(tgt)
    h = xr.op_matmul(w1r)
    a = h.op_tanh()
    o = a.op_matmul(w2r)
    loss = o.op_mse_loss(tgr, 'mean')
    loss.backward()

    xt = tt(x, True); w1t = tt(w1, True); w2t = tt(w2, True)
    F.mse_loss(torch.tanh(xt @ w1t) @ w2t, tt(tgt)).backward()

    assert_close(grad_np(xr), xt.grad.numpy(), rtol=1e-7, atol=1e-9, ctx="grad x")
    assert_close(grad_np(w1r), w1t.grad.numpy(), rtol=1e-7, atol=1e-9, ctx="grad w1")
    assert_close(grad_np(w2r), w2t.grad.numpy(), rtol=1e-7, atol=1e-9, ctx="grad w2")


def conv_case(shape_x, shape_w, stride, pad, dilation, bias=True, device='cpu'):
    x = RNG.uniform(-1, 1, shape_x)
    w = RNG.uniform(-0.5, 0.5, shape_w)
    b = RNG.uniform(-0.5, 0.5, (shape_w[0],)) if bias else None

    xr = t_from_np(x, device=device, requires_grad=True)
    wr = t_from_np(w, device=device, requires_grad=True)
    br = t_from_np(b, device=device, requires_grad=True) if bias else None
    y = xr.op_conv2d(wr, br, stride[0], stride[1], pad[0], pad[1], dilation[0], dilation[1])
    loss = y.op_sum_all()
    loss.backward()

    xt = tt(x, True); wt = tt(w, True)
    bt = tt(b, True) if bias else None
    yt = F.conv2d(xt, wt, bt, stride=stride, padding=pad, dilation=dilation)
    yt.sum().backward()

    ctx = f"conv2d s={stride} p={pad} d={dilation} bias={bias} dev={device}"
    assert_close(to_np(y), yt.detach().numpy(), rtol=1e-7, atol=1e-9, ctx=ctx + " fwd")
    assert_close(grad_np(xr), xt.grad.numpy(), rtol=1e-7, atol=1e-9, ctx=ctx + " dx")
    assert_close(grad_np(wr), wt.grad.numpy(), rtol=1e-7, atol=1e-9, ctx=ctx + " dw")
    if bias:
        assert_close(grad_np(br), bt.grad.numpy(), rtol=1e-7, atol=1e-9, ctx=ctx + " db")


def test_conv2d_parity_matrix():
    conv_case((2, 3, 8, 8), (5, 3, 3, 3), (1, 1), (0, 0), (1, 1))
    conv_case((2, 3, 8, 8), (5, 3, 3, 3), (2, 2), (1, 1), (1, 1))
    conv_case((1, 2, 9, 7), (4, 2, 3, 2), (2, 1), (1, 2), (1, 1), bias=False)
    conv_case((2, 3, 10, 10), (4, 3, 3, 3), (1, 1), (2, 2), (2, 2))
    conv_case((1, 1, 5, 5), (1, 1, 1, 1), (1, 1), (0, 0), (1, 1))


def test_conv2d_parity_gpu():
    if not has_cuda():
        return
    conv_case((2, 3, 8, 8), (5, 3, 3, 3), (2, 2), (1, 1), (1, 1), device='cuda')


def test_pooling_parity():
    x = RNG.uniform(-1, 1, (2, 3, 8, 8))
    xr = t_from_np(x, requires_grad=True)
    y = xr.op_maxpool2d(2, 2)
    y.op_sum_all().backward()
    xt = tt(x, True)
    yt = F.max_pool2d(xt, 2)
    yt.sum().backward()
    assert_close(to_np(y), yt.detach().numpy(), rtol=1e-9, ctx="maxpool fwd")
    assert_close(grad_np(xr), xt.grad.numpy(), rtol=1e-9, ctx="maxpool grad")

    xr2 = t_from_np(x, requires_grad=True)
    y2 = xr2.op_avgpool2d(2, 2, 2, 2, 0, 0)
    y2.op_sum_all().backward()
    xt2 = tt(x, True)
    yt2 = F.avg_pool2d(xt2, 2)
    yt2.sum().backward()
    assert_close(to_np(y2), yt2.detach().numpy(), rtol=1e-9, ctx="avgpool fwd")
    assert_close(grad_np(xr2), xt2.grad.numpy(), rtol=1e-9, ctx="avgpool grad")


def test_layer_norm_parity():
    x = RNG.uniform(-2, 2, (4, 10))
    w = RNG.uniform(0.5, 1.5, (10,))
    b = RNG.uniform(-0.5, 0.5, (10,))
    xr = t_from_np(x, requires_grad=True)
    y = xr.op_layer_norm([10], t_from_np(w), t_from_np(b), 1e-5)
    y.op_sum_all().backward()
    xt = tt(x, True)
    yt = F.layer_norm(xt, (10,), tt(w), tt(b), 1e-5)
    yt.sum().backward()
    assert_close(to_np(y), yt.detach().numpy(), rtol=1e-7, ctx="layer_norm fwd")
    assert_close(grad_np(xr), xt.grad.numpy(), rtol=1e-6, atol=1e-8, ctx="layer_norm grad")


def test_batch_norm_eval_parity():
    x = RNG.uniform(-2, 2, (4, 3, 5, 5))
    w = RNG.uniform(0.5, 1.5, (3,))
    b = RNG.uniform(-0.5, 0.5, (3,))
    rm = RNG.uniform(-0.5, 0.5, (3,))
    rv = RNG.uniform(0.5, 1.5, (3,))
    y = t_from_np(x).op_batch_norm(
        t_from_np(w), t_from_np(b), t_from_np(rm), t_from_np(rv), 1e-5, 0.1, False)
    yt = F.batch_norm(tt(x), tt(rm), tt(rv), tt(w), tt(b), False, 0.1, 1e-5)
    assert_close(to_np(y), yt.numpy(), rtol=1e-7, ctx="batch_norm eval fwd")


def optimizer_case(make_rp_opt, make_torch_opt, steps=10):
    p0 = RNG.uniform(-1, 1, (6, 6))
    tgt = RNG.uniform(-1, 1, (6, 6))

    pr = t_from_np(p0, dtype='float64', requires_grad=True)
    tr = t_from_np(tgt, dtype='float64')
    opt_r = make_rp_opt([pr])
    for _ in range(steps):
        opt_r.zero_grad()
        loss = pr.op_mse_loss(tr, 'mean')
        loss.backward()
        opt_r.step()
        del loss

    pt = tt(p0, True)
    tt_tgt = tt(tgt)
    opt_t = make_torch_opt([pt])
    for _ in range(steps):
        opt_t.zero_grad()
        F.mse_loss(pt, tt_tgt).backward()
        opt_t.step()

    assert_close(to_np(pr), pt.detach().numpy(), rtol=1e-6, atol=1e-8,
                 ctx=f"param after {steps} steps")


def test_sgd_parity():
    from recpulse.optim import SGD
    optimizer_case(lambda ps: SGD(ps, lr=0.1),
                   lambda ps: torch.optim.SGD(ps, lr=0.1))


def test_sgd_momentum_parity():
    from recpulse.optim import SGD
    optimizer_case(lambda ps: SGD(ps, lr=0.05, momentum=0.9),
                   lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9))


def test_sgd_nesterov_wd_parity():
    from recpulse.optim import SGD
    optimizer_case(lambda ps: SGD(ps, lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01),
                   lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01))


def test_adam_parity():
    from recpulse.optim import Adam
    optimizer_case(lambda ps: Adam(ps, lr=0.01),
                   lambda ps: torch.optim.Adam(ps, lr=0.01))


def test_adam_amsgrad_wd_parity():
    from recpulse.optim import Adam
    optimizer_case(lambda ps: Adam(ps, lr=0.01, weight_decay=0.01, amsgrad=True),
                   lambda ps: torch.optim.Adam(ps, lr=0.01, weight_decay=0.01, amsgrad=True))


def test_grad_clip_parity():
    g = RNG.uniform(-3, 3, (5, 5))
    pr = t_from_np(np.zeros((5, 5)), requires_grad=True)
    gt = t_from_np(g)
    m = pr.op_mul(gt)
    loss = m.op_sum_all()
    loss.backward()
    total = rp.clip_grad_norm([pr], 1.0)

    pt = tt(np.zeros((5, 5)), True)
    (pt * tt(g)).sum().backward()
    total_t = torch.nn.utils.clip_grad_norm_([pt], 1.0).item()

    assert abs(total - total_t) < 1e-6, f"total norm {total} vs {total_t}"
    assert_close(grad_np(pr), pt.grad.numpy(), rtol=1e-6, ctx="clipped grad")


def test_half_precision_forward():
    a = RNG.uniform(-2, 2, (8, 8))
    b = RNG.uniform(0.5, 1.5, (8, 8))
    for dt, tdt, tol in (('float16', torch.float16, 2e-3), ('bfloat16', torch.bfloat16, 2e-2)):
        got = t_from_np(a, dtype=dt).add(t_from_np(b, dtype=dt)).to(dtype='float32')
        want = (tt(a).to(tdt) + tt(b).to(tdt)).float().numpy()
        assert_close(to_np(got), want, rtol=tol, atol=tol, ctx=f"{dt} add")
        got = t_from_np(a, dtype=dt).mul(t_from_np(b, dtype=dt)).to(dtype='float32')
        want = (tt(a).to(tdt) * tt(b).to(tdt)).float().numpy()
        assert_close(to_np(got), want, rtol=tol, atol=tol, ctx=f"{dt} mul")


def test_scheduler_steplr_parity():
    from recpulse.optim import SGD
    from recpulse.scheduler import StepLR
    p = t_from_np(np.ones((2, 2)), requires_grad=True)
    opt = SGD([p], lr=1.0)
    sched = StepLR(opt, step_size=3, gamma=0.5)
    lrs = []
    for _ in range(10):
        sched.step()
        lrs.append(round(sched.get_lr(), 10))

    pt = tt(np.ones((2, 2)), True)
    opt_t = torch.optim.SGD([pt], lr=1.0)
    sched_t = torch.optim.lr_scheduler.StepLR(opt_t, step_size=3, gamma=0.5)
    lrs_t = []
    for _ in range(10):
        sched_t.step()
        lrs_t.append(round(opt_t.param_groups[0]['lr'], 10))

    assert lrs == lrs_t, f"StepLR schedule differs: rp={lrs} torch={lrs_t}"


run_test("unary forward parity f64 cpu", test_unary_forward_parity)
run_test("unary forward parity f32 gpu", test_unary_forward_parity_f32_gpu)
run_test("gelu parity (tanh approximation)", test_gelu_parity_tanh_approx)
run_test("binary forward parity", test_binary_forward_parity)
run_test("matmul parity cpu+gpu", test_matmul_parity)
run_test("reduction parity", test_reduction_parity)
run_test("softmax/log_softmax parity", test_softmax_parity)
run_test("loss parity (mse/bce/ce)", test_loss_parity)
run_test("backward parity: tanh-mlp", test_backward_parity_mlp)
run_test("conv2d parity matrix (stride/pad/dilation)", test_conv2d_parity_matrix)
run_test("pooling parity", test_pooling_parity)
run_test("layer_norm parity", test_layer_norm_parity)
run_test("batch_norm eval parity", test_batch_norm_eval_parity)
run_test("SGD parity", test_sgd_parity)
run_test("SGD momentum parity", test_sgd_momentum_parity)
run_test("SGD nesterov+wd parity", test_sgd_nesterov_wd_parity)
run_test("Adam parity", test_adam_parity)
run_test("Adam amsgrad+wd parity", test_adam_amsgrad_wd_parity)
run_test("clip_grad_norm parity", test_grad_clip_parity)
run_test("half precision forward vs torch", test_half_precision_forward)
run_test("StepLR schedule parity", test_scheduler_steplr_parity)
run_test("conv2d parity gpu (crash-prone, runs last)", test_conv2d_parity_gpu)

finish()
