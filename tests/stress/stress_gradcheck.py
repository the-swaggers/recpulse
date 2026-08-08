#!/usr/bin/env python3
from stress_common import *

EPS = 1e-6
RNG = np.random.default_rng(1234)

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def loss_value(make_graph, arrs, device='cpu', dtype='float64'):
    ts = [t_from_np(a, dtype=dtype, device=device) for a in arrs]
    loss, kept = make_graph(ts)
    if device != 'cpu':
        loss = loss.to(device='cpu')
    return loss.data()[0]


def gradcheck(make_graph, arrs, rtol=1e-4, atol=1e-6, check_inputs=None):
    arrs = [np.asarray(a, dtype=np.float64) for a in arrs]
    if check_inputs is None:
        check_inputs = list(range(len(arrs)))
    ts = [t_from_np(a, requires_grad=True) for a in arrs]
    loss, kept = make_graph(ts)
    loss.backward()
    for k in check_inputs:
        ana = grad_np(ts[k]).reshape(arrs[k].shape)
        num = np.zeros_like(arrs[k])
        flat = num.reshape(-1)
        for i in range(flat.size):
            plus = [a.copy() for a in arrs]
            minus = [a.copy() for a in arrs]
            plus[k].reshape(-1)[i] += EPS
            minus[k].reshape(-1)[i] -= EPS
            flat[i] = (loss_value(make_graph, plus) - loss_value(make_graph, minus)) / (2 * EPS)
        assert_close(ana, num, rtol=rtol, atol=atol, ctx=f"input{k} analytical-vs-numerical")


GPU_SAFE = [False]


def reduce_loss(z):
    if GPU_SAFE[0]:
        tgt = rp.zeros(list(z.shape), dtype=z.dtype, device=z.device)
        loss = z.op_mse_loss(tgt, 'sum')
        return loss, [tgt, loss]
    loss = z.op_sum_all()
    return loss, [loss]


def device_grad_parity(make_graph, arrs, dtype='float64', rtol=1e-7, atol=1e-9):
    arrs = [np.asarray(a, dtype=np.float64) for a in arrs]
    grads = {}
    GPU_SAFE[0] = True
    try:
        for device in ('cpu', 'cuda'):
            ts = [t_from_np(a, dtype=dtype, device=device, requires_grad=True) for a in arrs]
            loss, kept = make_graph(ts)
            loss.backward()
            grads[device] = [grad_np(t) for t in ts if t.has_grad]
    finally:
        GPU_SAFE[0] = False
    assert len(grads['cpu']) == len(grads['cuda'])
    for i, (gc, gd) in enumerate(zip(grads['cpu'], grads['cuda'])):
        assert_close(gd, gc, rtol=rtol, atol=atol, ctx=f"grad{i} cpu-vs-cuda ({dtype})")


def wsum(y, w):
    z = y.op_mul(w)
    loss, kept = reduce_loss(z)
    return loss, [y, w, z] + kept


UNARY_SPECS = [
    ('op_exp', -1.0, 1.0, None),
    ('op_log', 0.5, 2.0, None),
    ('op_sqrt', 0.5, 2.0, None),
    ('op_rsqrt', 0.5, 2.0, None),
    ('op_tanh', -2.0, 2.0, None),
    ('op_sigmoid', -3.0, 3.0, None),
    ('op_square', -2.0, 2.0, None),
    ('op_sin', -1.5, 1.5, None),
    ('op_cos', -1.5, 1.5, None),
    ('op_tan', -1.0, 1.0, None),
    ('op_asin', -0.9, 0.9, None),
    ('op_acos', -0.9, 0.9, None),
    ('op_atan', -2.0, 2.0, None),
    ('op_sinh', -2.0, 2.0, None),
    ('op_cosh', -2.0, 2.0, None),
    ('op_gelu', -2.0, 2.0, None),
    ('op_silu', -2.0, 2.0, None),
    ('op_relu', 0.1, 1.0, 'signed'),
    ('op_abs', 0.1, 1.0, 'signed'),
    ('op_leaky_relu', 0.1, 1.0, 0.1),
]


def make_unary_case(opname, low, high, extra):
    x = RNG.uniform(low, high, (3, 4))
    if extra == 'signed' or isinstance(extra, float):
        x = x * np.where(RNG.uniform(size=x.shape) < 0.5, -1.0, 1.0)
    w = RNG.uniform(0.5, 1.5, (3, 4))

    def make_graph(ts):
        wt = t_from_np(w, dtype=ts[0].dtype, device=ts[0].device)
        if isinstance(extra, float):
            y = getattr(ts[0], opname)(extra)
        else:
            y = getattr(ts[0], opname)()
        return wsum(y, wt)
    return make_graph, [x]


for _opname, _lo, _hi, _extra in UNARY_SPECS:
    def _reg(opname=_opname, lo=_lo, hi=_hi, extra=_extra):
        mg, xs = make_unary_case(opname, lo, hi, extra)

        @case(f"unary {opname}")
        def _t():
            gradcheck(mg, xs)

        @case(f"unary {opname} [gpu-parity]")
        def _tg():
            if not has_cuda():
                return
            device_grad_parity(mg, xs)
    _reg()


BINARY_SPECS = [
    ('op_add', (-1, 1), (-1, 1)),
    ('op_sub', (-1, 1), (-1, 1)),
    ('op_mul', (-1, 1), (-1, 1)),
    ('op_div', (-1, 1), (0.5, 2.0)),
    ('op_power', (0.3, 2.0), (0.5, 2.0)),
    ('op_logb', (0.5, 3.0), (1.5, 3.0)),
]

BROADCAST_SHAPES = [((3, 4), (3, 4)), ((3, 1), (1, 4)), ((4,), (3, 4)), ((2, 3, 4), (3, 4)), ((1,), (3, 4))]


def make_binary_case(opname, d1, d2, s1, s2):
    x1 = RNG.uniform(*d1, s1)
    x2 = RNG.uniform(*d2, s2)
    out_shape = np.broadcast_shapes(s1, s2)
    w = RNG.uniform(0.5, 1.5, out_shape)

    def make_graph(ts):
        wt = t_from_np(w, dtype=ts[0].dtype, device=ts[0].device)
        y = getattr(ts[0], opname)(ts[1])
        return wsum(y, wt)
    return make_graph, [x1, x2]


for _opname, _d1, _d2 in BINARY_SPECS:
    for _s1, _s2 in BROADCAST_SHAPES:
        def _reg(opname=_opname, d1=_d1, d2=_d2, s1=_s1, s2=_s2):
            mg, xs = make_binary_case(opname, d1, d2, s1, s2)
            tag = f"binary {opname} {s1}x{s2}"

            @case(tag)
            def _t():
                gradcheck(mg, xs)

            @case(tag + " [gpu-parity]")
            def _tg():
                if not has_cuda():
                    return
                device_grad_parity(mg, xs)
        _reg()


SCALAR_SPECS = [
    ('op_add_scalar', 0.7, (-1, 1)),
    ('op_sub_scalar', 0.7, (-1, 1)),
    ('op_mul_scalar', -1.3, (-1, 1)),
    ('op_div_scalar', 1.7, (-1, 1)),
    ('op_pow_scalar', 2.5, (0.3, 2.0)),
    ('op_rsub_scalar', 0.7, (-1, 1)),
    ('op_rdiv_scalar', 1.5, (0.5, 2.0)),
    ('op_rpow_scalar', 1.8, (-1, 1)),
    ('op_logb_scalar', 3.0, (0.5, 3.0)),
    ('op_rlogb_scalar', 2.0, (1.5, 3.0)),
]

for _opname, _s, _dom in SCALAR_SPECS:
    def _reg(opname=_opname, s=_s, dom=_dom):
        x = RNG.uniform(*dom, (3, 4))
        w = RNG.uniform(0.5, 1.5, (3, 4))

        def mg(ts):
            wt = t_from_np(w, dtype=ts[0].dtype, device=ts[0].device)
            y = getattr(ts[0], opname)(s)
            return wsum(y, wt)

        @case(f"scalar {opname}({s})")
        def _t():
            gradcheck(mg, [x])

        @case(f"scalar {opname}({s}) [gpu-parity]")
        def _tg():
            if not has_cuda():
                return
            device_grad_parity(mg, [x])
    _reg()


@case("matmul (3,4)@(4,2)")
def _():
    a = RNG.uniform(-1, 1, (3, 4))
    b = RNG.uniform(-1, 1, (4, 2))
    w = RNG.uniform(0.5, 1.5, (3, 2))

    def mg(ts):
        wt = t_from_np(w, dtype=ts[0].dtype, device=ts[0].device)
        y = ts[0].op_matmul(ts[1])
        return wsum(y, wt)
    gradcheck(mg, [a, b])
    if has_cuda():
        device_grad_parity(mg, [a, b])


@case("matmul with transposed input (non-contiguous)")
def _():
    a = RNG.uniform(-1, 1, (4, 3))
    b = RNG.uniform(-1, 1, (4, 2))
    w = RNG.uniform(0.5, 1.5, (3, 2))

    def mg(ts):
        wt = t_from_np(w, dtype=ts[0].dtype, device=ts[0].device)
        at = ts[0].transpose(0, 1)
        y = at.op_matmul(ts[1])
        loss, kept = wsum(y, wt)
        return loss, kept + [at]
    gradcheck(mg, [a, b])
    if has_cuda():
        device_grad_parity(mg, [a, b])


REDUCTION_SPECS = [
    ('op_sum_all', None, None),
    ('op_mean_all', None, None),
    ('op_sum_dim', 0, False),
    ('op_sum_dim', 1, True),
    ('op_mean_dim', 1, False),
    ('op_mean_dim', 0, True),
]

for _opname, _dim, _keep in REDUCTION_SPECS:
    def _reg(opname=_opname, dim=_dim, keep=_keep):
        x = RNG.uniform(-1, 1, (3, 4))

        def mg(ts):
            if dim is None:
                y = getattr(ts[0], opname)()
                return y, [y]
            y = getattr(ts[0], opname)(dim, keep)
            w = np.arange(1.0, y_size(x.shape, dim, keep) + 1.0)
            wt = t_from_np(w.reshape(y_shape(x.shape, dim, keep)),
                           dtype=ts[0].dtype, device=ts[0].device)
            return wsum(y, wt)

        tag = f"reduction {opname}" + (f"(dim={dim},keepdim={keep})" if dim is not None else "")

        @case(tag)
        def _t():
            gradcheck(mg, [x])

        @case(tag + " [gpu-parity]")
        def _tg():
            if not has_cuda() or dim is None:
                return
            device_grad_parity(mg, [x])
    _reg()


def y_shape(shape, dim, keep):
    s = list(shape)
    if keep:
        s[dim] = 1
    else:
        s.pop(dim)
    return s


def y_size(shape, dim, keep):
    return int(np.prod(y_shape(shape, dim, keep)))


def shape_case(name, fn, in_shape=(3, 4), n_inputs=1):
    x_arrs = [RNG.uniform(-1, 1, in_shape) for _ in range(n_inputs)]
    probe = fn(*[t_from_np(a) for a in x_arrs])
    w = RNG_W.uniform(0.5, 1.5, tuple(probe.shape))

    def mg(ts):
        y = fn(*ts)
        wt = t_from_np(w, dtype=ts[0].dtype, device=ts[0].device)
        loss, kept = wsum(y, wt)
        return loss, kept + [y]

    @case(f"shape {name}")
    def _t():
        gradcheck(mg, x_arrs)

    @case(f"shape {name} [gpu-parity]")
    def _tg():
        if not has_cuda():
            return
        device_grad_parity(mg, x_arrs)


RNG_W = np.random.default_rng(99)

shape_case("reshape [3,4]->[4,3]", lambda t: t.reshape([4, 3]))
shape_case("reshape [3,4]->[12]", lambda t: t.reshape([12]))
shape_case("view [3,4]->[2,6]", lambda t: t.view([2, 6]))
shape_case("transpose(0,1)", lambda t: t.transpose(0, 1))
shape_case("permute [1,0]", lambda t: t.permute([1, 0]))
shape_case("permute 3d [2,0,1]", lambda t: t.permute([2, 0, 1]), in_shape=(2, 3, 4))
shape_case("squeeze", lambda t: t.squeeze(1), in_shape=(3, 1, 4))
shape_case("unsqueeze(1)", lambda t: t.unsqueeze(1))
shape_case("flatten", lambda t: t.flatten(), in_shape=(2, 3, 4))
shape_case("flatten(1,2)", lambda t: t.flatten(1, 2), in_shape=(2, 3, 4))
shape_case("expand (3,1)->(3,4)", lambda t: t.expand([3, 4]), in_shape=(3, 1))
shape_case("repeat [2,3]", lambda t: t.repeat([2, 3]))
shape_case("slice middle", lambda t: t.slice([1, 1], [3, 3], [1, 1]), in_shape=(4, 4))
shape_case("cat dim0", lambda a, b: rp.cat([a, b], 0), n_inputs=2)
shape_case("cat dim1", lambda a, b: rp.cat([a, b], 1), n_inputs=2)


def _chunk_mg_factory():
    w1 = RNG.uniform(0.5, 1.5, (3, 2))
    w2 = RNG.uniform(1.5, 2.5, (3, 2))

    def mg(ts):
        parts = ts[0].chunk(2, 1)
        a, b = parts[0], parts[1]
        wt1 = t_from_np(w1, dtype=ts[0].dtype, device=ts[0].device)
        wt2 = t_from_np(w2, dtype=ts[0].dtype, device=ts[0].device)
        za = a.op_mul(wt1)
        zb = b.op_mul(wt2)
        s = za.op_add(zb)
        loss, kept = reduce_loss(s)
        return loss, [a, b, wt1, wt2, za, zb, s] + kept
    return mg


_CHUNK_X = RNG.uniform(-1, 1, (3, 4))
_CHUNK_MG = _chunk_mg_factory()


@case("shape chunk(2, dim=1) both outputs used")
def _():
    gradcheck(_CHUNK_MG, [_CHUNK_X])


@case("shape chunk(2, dim=1) [gpu-parity]")
def _():
    if not has_cuda():
        return
    device_grad_parity(_CHUNK_MG, [_CHUNK_X])


@case("shape split([1,3], dim=1) both outputs used")
def _():
    x = RNG.uniform(-1, 1, (3, 4))
    w1 = RNG.uniform(0.5, 1.5, (3, 1))
    w2 = RNG.uniform(1.5, 2.5, (3, 3))

    def mg(ts):
        parts = ts[0].split([1, 3], 1)
        a, b = parts[0], parts[1]
        wt1 = t_from_np(w1, dtype=ts[0].dtype, device=ts[0].device)
        wt2 = t_from_np(w2, dtype=ts[0].dtype, device=ts[0].device)
        ma = a.op_mul(wt1)
        mb = b.op_mul(wt2)
        za = ma.op_sum_all()
        zb = mb.op_sum_all()
        loss = za.op_add(zb)
        return loss, [a, b, wt1, wt2, ma, mb, za, zb, loss]
    gradcheck(mg, [x])


@case("gather(dim=1)")
def _():
    x = RNG.uniform(-1, 1, (3, 4))
    idx = np.array([[0, 2], [1, 3], [2, 0]], dtype=np.float64)
    w = RNG.uniform(0.5, 1.5, (3, 2))

    def mg(ts):
        it = t_from_np(idx, dtype='float32', device=ts[0].device)
        y = ts[0].op_gather(1, it)
        wt = t_from_np(w, dtype=ts[0].dtype, device=ts[0].device)
        loss, kept = wsum(y, wt)
        return loss, kept + [it]
    gradcheck(mg, [x])


for _dim in (-1, 0):
    def _reg(dim=_dim):
        x = RNG.uniform(-2, 2, (3, 4))
        w = RNG.uniform(0.5, 1.5, (3, 4))

        def mg_sm(ts):
            wt = t_from_np(w, dtype=ts[0].dtype, device=ts[0].device)
            y = ts[0].op_softmax(dim)
            return wsum(y, wt)

        def mg_lsm(ts):
            wt = t_from_np(w, dtype=ts[0].dtype, device=ts[0].device)
            y = ts[0].op_log_softmax(dim)
            return wsum(y, wt)

        @case(f"softmax(dim={dim})")
        def _t():
            gradcheck(mg_sm, [x])

        @case(f"log_softmax(dim={dim})")
        def _t2():
            gradcheck(mg_lsm, [x])

        @case(f"softmax(dim={dim}) [gpu-parity]")
        def _tg():
            if not has_cuda():
                return
            device_grad_parity(mg_sm, [x])
    _reg()


@case("mse_loss reduction='none' composed with mean_all")
def _():
    x = RNG.uniform(-1, 1, (3, 4))
    tgt = RNG.uniform(-1, 1, (3, 4))

    def mg(ts):
        tt = t_from_np(tgt, dtype=ts[0].dtype, device=ts[0].device)
        per_elem = ts[0].op_mse_loss(tt, 'none')
        loss = per_elem.op_mean_all()
        return loss, [tt, per_elem, loss]
    gradcheck(mg, [x])


@case("cross_entropy on probabilities (softmax upstream)")
def _():
    x = RNG.uniform(-2, 2, (3, 5))
    targets = [0, 3, 2]

    def mg(ts):
        p = ts[0].op_softmax(-1)
        loss = p.op_cross_entropy_loss(targets, 'mean', 0)
        return loss, [p, loss]
    gradcheck(mg, [x])


@case("mse_loss mean + sum")
def _():
    x = RNG.uniform(-1, 1, (3, 4))
    tgt = RNG.uniform(-1, 1, (3, 4))
    for reduction in ('mean', 'sum'):
        def mg(ts, reduction=reduction):
            tt = t_from_np(tgt, dtype=ts[0].dtype, device=ts[0].device)
            loss = ts[0].op_mse_loss(tt, reduction)
            return loss, [tt, loss]
        gradcheck(mg, [x])


@case("bce_loss on probabilities")
def _():
    x = RNG.uniform(0.05, 0.95, (3, 4))
    tgt = RNG.uniform(0.0, 1.0, (3, 4)).round()

    def mg(ts):
        tt = t_from_np(tgt, dtype=ts[0].dtype, device=ts[0].device)
        loss = ts[0].op_bce_loss(tt, 'mean', 0)
        return loss, [tt, loss]
    gradcheck(mg, [x])


@case("bce_loss from logits")
def _():
    x = RNG.uniform(-3, 3, (3, 4))
    tgt = RNG.uniform(0.0, 1.0, (3, 4)).round()

    def mg(ts):
        tt = t_from_np(tgt, dtype=ts[0].dtype, device=ts[0].device)
        loss = ts[0].op_bce_loss(tt, 'mean', 1)
        return loss, [tt, loss]
    gradcheck(mg, [x])


@case("cross_entropy from logits")
def _():
    x = RNG.uniform(-2, 2, (3, 5))
    targets = [0, 3, 2]

    def mg(ts):
        loss = ts[0].op_cross_entropy_loss(targets, 'mean', 1)
        return loss, [loss]
    gradcheck(mg, [x])
    if has_cuda():
        device_grad_parity(mg, [x])


@case("nll(log_softmax) composite")
def _():
    x = RNG.uniform(-2, 2, (3, 5))
    targets = [1, 4, 0]

    def mg(ts):
        ls = ts[0].op_log_softmax(-1)
        loss = ls.op_nll_loss(targets, 'mean')
        return loss, [ls, loss]
    gradcheck(mg, [x])


@case("same tensor twice: x*x")
def _():
    x = RNG.uniform(0.5, 1.5, (3, 4))

    def mg(ts):
        y = ts[0].op_mul(ts[0])
        loss = y.op_sum_all()
        return loss, [y, loss]
    gradcheck(mg, [x])


@case("diamond: y=x^2 used by two branches")
def _():
    x = RNG.uniform(0.5, 1.5, (3, 4))
    w = RNG.uniform(0.5, 1.5, (3, 4))

    def mg(ts):
        wt = t_from_np(w, dtype=ts[0].dtype, device=ts[0].device)
        y = ts[0].op_square()
        z1 = y.op_mul(wt)
        z2 = y.op_add(wt)
        s = z1.op_add(z2)
        loss = s.op_sum_all()
        return loss, [wt, y, z1, z2, s, loss]
    gradcheck(mg, [x])


@case("mlp composite: relu(x@W1)@W2 -> mse")
def _():
    x = RNG.uniform(-1, 1, (4, 5))
    w1 = RNG.uniform(-0.5, 0.5, (5, 6))
    w2 = RNG.uniform(-0.5, 0.5, (6, 2))
    tgt = RNG.uniform(-1, 1, (4, 2))

    def mg(ts):
        tt = t_from_np(tgt, dtype=ts[0].dtype, device=ts[0].device)
        h = ts[0].op_matmul(ts[1])
        a = h.op_relu()
        o = a.op_matmul(ts[2])
        loss = o.op_mse_loss(tt, 'mean')
        return loss, [h, a, o, tt, loss]
    gradcheck(mg, [x, w1, w2])
    if has_cuda():
        device_grad_parity(mg, [x, w1, w2])


@case("deep chain of 30 scalar ops")
def _():
    x = RNG.uniform(0.5, 1.5, (2, 3))

    def mg(ts):
        kept = []
        y = ts[0]
        for i in range(30):
            y = y.op_add_scalar(0.05) if i % 2 == 0 else y.op_mul_scalar(1.01)
            kept.append(y)
        loss = y.op_sum_all()
        kept.append(loss)
        return loss, kept
    gradcheck(mg, [x])


def _conv_mg(ts):
    y = ts[0].op_conv2d(ts[1], ts[2], 2, 2, 1, 1, 1, 1)
    loss, kept = reduce_loss(y)
    return loss, [y] + kept


_CONV_XS = [RNG.uniform(-1, 1, (2, 3, 5, 5)),
            RNG.uniform(-0.5, 0.5, (4, 3, 3, 3)),
            RNG.uniform(-0.5, 0.5, (4,))]


@case("conv2d gradcheck (stride=2, pad=1)")
def _():
    gradcheck(_conv_mg, _CONV_XS, rtol=3e-4)


@case("conv2d [gpu-parity]")
def _():
    if not has_cuda():
        return
    device_grad_parity(_conv_mg, _CONV_XS)


@case("maxpool2d + avgpool2d gradcheck")
def _():
    x = RNG.uniform(-1, 1, (2, 2, 6, 6))
    x += np.arange(x.size).reshape(x.shape) * 1e-3

    def mg_max(ts):
        y = ts[0].op_maxpool2d(2, 2)
        loss = y.op_sum_all()
        return loss, [y, loss]

    def mg_avg(ts):
        y = ts[0].op_avgpool2d(2, 2)
        loss = y.op_sum_all()
        return loss, [y, loss]
    gradcheck(mg_max, [x])
    gradcheck(mg_avg, [x])


@case("layer_norm gradcheck")
def _():
    x = RNG.uniform(-1, 1, (3, 6))
    w = RNG.uniform(0.5, 1.5, (6,))
    b = RNG.uniform(-0.5, 0.5, (6,))
    ww = RNG.uniform(0.5, 1.5, (3, 6))

    def mg(ts):
        wt = t_from_np(ww, dtype=ts[0].dtype, device=ts[0].device)
        y = ts[0].op_layer_norm([6], ts[1], ts[2], 1e-5)
        return wsum(y, wt)
    gradcheck(mg, [x, w, b], rtol=3e-4)


@case("embedding gradcheck")
def _():
    table = RNG.uniform(-1, 1, (5, 3))
    idx = [0, 2, 2, 4]
    w = RNG.uniform(0.5, 1.5, (4, 3))

    def mg(ts):
        wt = t_from_np(w, dtype=ts[0].dtype, device=ts[0].device)
        y = ts[0].op_embedding(idx)
        return wsum(y, wt)
    gradcheck(mg, [table])


def _order(item):
    name = item[0]
    gpu = '[gpu-parity]' in name
    risky = gpu and any(k in name for k in ('slice', 'cat', 'chunk', 'split'))
    return (2 if risky else 1) if gpu else 0


CASES.sort(key=_order)

print(f"stress_gradcheck: {len(CASES)} cases (cuda={'yes' if has_cuda() else 'NO'})")
for name, fn in CASES:
    run_test(name, fn)
finish()
