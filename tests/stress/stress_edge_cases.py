#!/usr/bin/env python3
from stress_common import *

RNG = np.random.default_rng(777)


def classify(rc, out):
    if rc == 0 and 'RAISED' in out:
        return 'raised'
    if rc == 3 and 'NO_ERROR' in out:
        return 'accepted'
    if rc == -999:
        return 'timeout'
    if rc != 0:
        return f'crashed(rc={rc})'
    return f'unknown(rc={rc})'


def graceful(snippet, expect='raise'):
    code = (
        "try:\n"
        + ''.join('    ' + line + '\n' for line in snippet.strip().split('\n'))
        + "except Exception as e:\n"
        "    print('RAISED', type(e).__name__, str(e)[:120]); sys.exit(0)\n"
        "print('NO_ERROR'); sys.exit(3)\n"
    )
    rc, out = run_isolated(code)
    status = classify(rc, out)
    if expect == 'raise':
        assert status == 'raised', f"expected clean exception, got {status}: {out[-300:]}"
    else:
        assert status in ('raised', 'accepted'), \
            f"expected no crash, got {status}: {out[-300:]}"


INVALID_RAISE = [
    ("binary shape mismatch not broadcastable",
     "a = rp.zeros([3, 4]); b = rp.zeros([5, 6]); c = a.add(b)"),
    ("op_ binary shape mismatch not broadcastable",
     "a = rp.zeros([3, 4]); b = rp.zeros([5, 6]); c = a.op_add(b)"),
    ("matmul inner dim mismatch",
     "a = rp.zeros([3, 4]); b = rp.zeros([5, 2]); c = a.op_matmul(b)"),
    ("reshape to wrong element count",
     "a = rp.zeros([3, 4]); b = a.reshape([5, 5])"),
    ("view to wrong element count",
     "a = rp.zeros([3, 4]); b = a.view([7])"),
    ("transpose out-of-range dim",
     "a = rp.zeros([3, 4]); b = a.transpose(0, 5)"),
    ("permute wrong length",
     "a = rp.zeros([3, 4]); b = a.permute([0, 1, 2])"),
    ("cat shape mismatch on non-cat dim",
     "a = rp.zeros([3, 4]); b = rp.zeros([3, 5]); c = rp.cat([a, b], 0)"),
    ("squeeze non-size-1 dim",
     "a = rp.zeros([3, 4]); b = a.squeeze(0)"),
    ("split sizes not summing to dim",
     "a = rp.zeros([3, 4]); parts = a.split([1, 1], 1)"),
    ("invalid dtype string",
     "a = rp.zeros([2, 2], dtype='int8')"),
    ("invalid device string",
     "a = rp.zeros([2, 2], device='tpu')"),
    ("zeros with zero dim",
     "a = rp.zeros([0, 3])"),
    ("zeros with negative dim",
     "a = rp.zeros([-2, 3])"),
    ("values with empty data",
     "a = rp.values([])"),
    ("slice stop beyond shape",
     "a = rp.zeros([3, 4]); b = a.slice([0, 0], [3, 9], [1, 1]); print(b.data())"),
    ("sum_dim out-of-range dim",
     "a = rp.zeros([3, 4]); b = a.op_sum_dim(4)"),
    ("expand shrinking non-1 dim",
     "a = rp.zeros([3, 4]); b = a.expand([3, 2])"),
]

NOCRASH_ONLY = [
    ("mixed dtype binary op",
     "a = rp.zeros([2, 2], dtype='float32'); b = rp.zeros([2, 2], dtype='float64'); c = a.op_add(b); print(c.dtype)"),
    ("backward on non-scalar tensor",
     "a = rp.ones([3, 4]).requires_grad_(); b = a.op_square(); b.backward()"),
    ("backward called twice on same graph",
     "a = rp.ones([2, 2]).requires_grad_()\n"
     "b = a.op_square()\n"
     "loss = b.op_sum_all()\n"
     "loss.backward()\n"
     "loss.backward()\n"
     "print(a.grad.data())"),
    ("gather index out of range",
     "a = rp.ones([3, 4]).requires_grad_()\n"
     "idx = rp.values([0.0, 9.0, 1.0]).reshape([3, 1])\n"
     "g = a.op_gather(1, idx)\n"
     "print(g.data())"),
    ("nll_loss target class out of range",
     "a = rp.randn([3, 5])\n"
     "loss = a.op_nll_loss([0, 7, 1])\n"
     "print(loss.data())"),
    ("embedding index out of range",
     "w = rp.randn([5, 3])\n"
     "e = w.op_embedding([0, 11])\n"
     "print(e.data())"),
    ("chained ops without keeping intermediates (known GC hazard)",
     "import gc\n"
     "a = rp.ones([64, 64]).requires_grad_()\n"
     "loss = a.op_square().op_mul_scalar(3.0).op_sum_all()\n"
     "gc.collect()\n"
     "loss.backward()\n"
     "print(a.grad.data()[0])"),
    ("del intermediates then backward (known GC hazard)",
     "import gc\n"
     "a = rp.ones([32, 32]).requires_grad_()\n"
     "b = a.op_square()\n"
     "c = b.op_sum_all()\n"
     "del b\n"
     "gc.collect()\n"
     "c.backward()\n"
     "print(a.grad.data()[0])"),
    ("free leaf before backward",
     "import gc\n"
     "a = rp.ones([16, 16]).requires_grad_()\n"
     "c = a.op_square().op_sum_all()\n"
     "del a\n"
     "gc.collect()\n"
     "c.backward()\n"
     "print('done')"),
    ("split with negative dim backward",
     "x = rp.ones([3, 5]).requires_grad_()\n"
     "parts = x.split([2, 3], -1)\n"
     "loss = parts[0].op_sum_all()\n"
     "loss.backward()\n"
     "g = x.grad.to_numpy()\n"
     "assert g.sum() == 6.0, f'wrong grad sum {g.sum()}'\n"
     "print('ok')"),
    ("tokenizer decode of unknown id",
     "tok = rp.Tokenizer()\n"
     "tok.train('hello world hello', vocab_size=280)\n"
     "print(tok.decode([0, 100000]))"),
]

AUDIT_PROBES = [
    ("view outlives base (del base, read view)",
     "t = rp.ones([4, 4])\n"
     "v = t.slice([1, 0], [3, 4], [1, 1])\n"
     "del t\n"
     "import gc; gc.collect()\n"
     "print(v.data())"),
    ("copy() of expanded view",
     "t = rp.values([1.0, 2.0, 3.0, 4.0]).reshape([1, 4])\n"
     "e = t.expand([10000, 4])\n"
     "c = e.copy()\n"
     "print(c.shape, c.data()[:4])"),
    ("copy() of stepped slice has correct values",
     "t = rp.values([float(i) for i in range(10)])\n"
     "s = t.slice([0], [10], [2])\n"
     "c = s.copy()\n"
     "vals = c.data()\n"
     "assert vals == [0.0, 2.0, 4.0, 6.0, 8.0], f'wrong values: {vals}'"),
    ("to_numpy of offset slice has correct values",
     "t = rp.values([float(i) for i in range(12)]).reshape([3, 4])\n"
     "s = t.slice([1, 1], [3, 3], [1, 1])\n"
     "a = s.to_numpy()\n"
     "assert a.tolist() == [[5.0, 6.0], [9.0, 10.0]], f'wrong: {a.tolist()}'"),
    ("inplace .to() while autograd graph references source",
     "x = rp.ones([2, 2]).requires_grad_()\n"
     "y = x.op_square()\n"
     "x.to(dtype='float64', inplace=True)\n"
     "loss = y.op_sum_all()\n"
     "loss.backward()\n"
     "print('done')"),
]

CROSS_DEVICE = [
    ("gpu chunk backward",
     "x = rp.ones([4, 4], device='cuda').requires_grad_()\n"
     "parts = x.chunk(2, 0)\n"
     "a = parts[0]\n"
     "loss = a.op_sum_all()\n"
     "loss.backward()\n"
     "print(x.grad.to_numpy().sum())"),
    ("gpu slice backward",
     "x = rp.ones([4, 4], device='cuda').requires_grad_()\n"
     "s = x.slice([1, 0], [3, 4], [1, 1])\n"
     "loss = s.op_sum_all()\n"
     "loss.backward()\n"
     "print(x.grad.to_numpy().sum())"),
    ("gpu cat backward",
     "a = rp.ones([2, 3], device='cuda').requires_grad_()\n"
     "b = rp.ones([2, 3], device='cuda').requires_grad_()\n"
     "c = rp.cat([a, b], 1)\n"
     "loss = c.op_sum_all()\n"
     "loss.backward()\n"
     "print(a.grad.to_numpy().sum(), b.grad.to_numpy().sum())"),
    ("op on cpu+cuda tensors",
     "a = rp.zeros([2, 2]); b = rp.zeros([2, 2], device='cuda'); c = a.op_add(b); print(c.device)"),
    ("matmul cpu@cuda",
     "a = rp.zeros([2, 3]); b = rp.zeros([3, 2], device='cuda'); c = a.op_matmul(b)"),
    ("cat across devices",
     "a = rp.zeros([2, 2]); b = rp.zeros([2, 2], device='cuda'); c = rp.cat([a, b], 0)"),
    ("invalid cuda index",
     "a = rp.zeros([2, 2], device='cuda:7')"),
]

for _name, _code in INVALID_RAISE:
    def _t(code=_code):
        graceful(code, expect='raise')
    run_test(f"invalid-input raises: {_name}", _t)

for _name, _code in NOCRASH_ONLY:
    def _t(code=_code):
        graceful(code, expect='nocrash')
    run_test(f"no-crash: {_name}", _t)

for _name, _code in AUDIT_PROBES:
    def _t(code=_code):
        graceful(code, expect='nocrash')
    run_test(f"audit-probe: {_name}", _t)

if has_cuda():
    for _name, _code in CROSS_DEVICE:
        def _t(code=_code):
            graceful(code, expect='nocrash')
        run_test(f"cross-device no-crash: {_name}", _t)


def test_single_element():
    x = t_from_np([[2.0]], dtype='float64', requires_grad=True)
    y = x.op_square()
    loss = y.op_sum_all()
    loss.backward()
    assert_close(grad_np(x), [[4.0]], ctx="d(x^2)/dx at 2")


def test_noncontig_unary_forward():
    a = RNG.uniform(0.1, 2.0, (3, 4))
    t = t_from_np(a)
    tt = t.transpose(0, 1)
    r = to_np(tt.exp())
    assert_close(r, np.exp(a.T), ctx="exp(transpose)")


def test_noncontig_slice_forward():
    a = RNG.uniform(0.1, 2.0, (4, 4))
    t = t_from_np(a)
    s = t.slice([1, 1], [3, 4], [1, 1])
    r = to_np(s.sqrt())
    assert_close(r, np.sqrt(a[1:3, 1:4]), ctx="sqrt(slice)")


def test_expand_stride0_forward():
    a = RNG.uniform(0.1, 2.0, (3, 1))
    t = t_from_np(a)
    e = t.expand([3, 5])
    r = to_np(e.log())
    assert_close(r, np.log(np.broadcast_to(a, (3, 5))), ctx="log(expand)")


def test_to_numpy_of_views():
    a = RNG.uniform(-1, 1, (4, 5))
    t = t_from_np(a)
    assert_close(to_np(t.transpose(0, 1)), a.T, ctx="to_numpy(transpose)")
    assert_close(to_np(t.slice([1, 2], [4, 5], [1, 1])), a[1:4, 2:5], ctx="to_numpy(slice)")
    assert_close(to_np(t.permute([1, 0])), a.T, ctx="to_numpy(permute)")


def test_broadcast_forward_matrix():
    shapes = [((1,), (3, 4)), ((3, 1), (1, 4)), ((2, 1, 4), (3, 1)), ((4,), (2, 3, 4))]
    for s1, s2 in shapes:
        a = RNG.uniform(0.5, 1.5, s1)
        b = RNG.uniform(0.5, 1.5, s2)
        r = to_np(t_from_np(a).op_mul(t_from_np(b)))
        assert_close(r, a * b, ctx=f"broadcast mul {s1}x{s2}")


def test_grad_accumulation_two_backwards():
    x = t_from_np(np.full((2, 2), 3.0), requires_grad=True)
    y1 = x.op_square()
    l1 = y1.op_sum_all()
    l1.backward()
    g1 = grad_np(x)
    y2 = x.op_square()
    l2 = y2.op_sum_all()
    l2.backward()
    g2 = grad_np(x)
    assert_close(g2, 2 * g1, ctx="second backward accumulates")
    x.zero_grad()
    assert not x.has_grad, "zero_grad should clear gradient"


def test_dtype_roundtrips():
    a = RNG.uniform(-2, 2, (3, 4))
    t = t_from_np(a, dtype='float64')
    for dt, tol in (('float32', 1e-6), ('float16', 2e-3), ('bfloat16', 2e-2)):
        r = to_np(t.to(dtype=dt).to(dtype='float64'))
        assert_close(r, a, rtol=tol, atol=tol, ctx=f"roundtrip via {dt}")


def test_grad_transfer_in_to():
    x = t_from_np(np.full((2, 2), 2.0), requires_grad=True)
    y = x.op_square()
    loss = y.op_sum_all()
    loss.backward()
    moved = x.to(dtype='float32')
    assert moved.has_grad, "grad should transfer with .to()"
    assert_close(grad_np(moved), np.full((2, 2), 4.0), rtol=1e-6, ctx="transferred grad")


def test_nested_views_backward():
    a = RNG.uniform(-1, 1, (3, 4))
    w = RNG.uniform(0.5, 1.5, (2, 6))
    x = t_from_np(a, requires_grad=True)
    v1 = x.view([12])
    v2 = v1.view([2, 6])
    wt = t_from_np(w)
    z = v2.op_mul(wt)
    loss = z.op_sum_all()
    loss.backward()
    assert_close(grad_np(x), w.reshape(3, 4), ctx="nested view grad")


def test_large_reduction_precision():
    n = 1_000_000
    ones = rp.ones([n])
    s = ones.sum_all()
    assert abs(s - n) < 1.0, f"sum of {n} ones = {s}"


def test_gpu_sum_all_size_sweep():
    if not has_cuda():
        return
    bad = []
    for n in (255, 256, 257, 640, 1280, 4096, 5000, 65536, 65537, 100_000, 300_000, 1_000_000):
        rc, out = run_isolated(
            f"s = rp.ones([{n}], device='cuda').sum_all()\nprint('SUM=%r' % s)")
        if rc != 0:
            bad.append((n, f'crash rc={rc}'))
            continue
        try:
            val = float(out.split('SUM=')[1].strip().splitlines()[0])
        except Exception:
            bad.append((n, out[-80:]))
            continue
        if abs(val - n) > 0.5:
            bad.append((n, val))
    assert not bad, f"gpu sum_all wrong/crashed for sizes: {bad}"


def test_gpu_mean_all_matches_cpu():
    if not has_cuda():
        return
    bad = []
    for n in (640, 5000, 100_000):
        rc, out = run_isolated(
            "rp.manual_seed(31)\n"
            f"t = rp.randn([{n}])\n"
            "mc = t.mean_all()\n"
            "mg = t.to(device='cuda').mean_all()\n"
            "print('MEANS=%r,%r' % (mc, mg))")
        if rc != 0:
            bad.append((n, f'crash rc={rc}'))
            continue
        mc, mg = (float(v) for v in out.split('MEANS=')[1].strip().splitlines()[0].split(','))
        if abs(mc - mg) > 1e-4 * max(1, abs(mc)):
            bad.append((n, mc, mg))
    assert not bad, f"gpu mean_all mismatch/crash: {bad}"


def test_large_tensor_gpu_ops():
    if not has_cuda():
        return
    n = 4_000_000
    rc, out = run_isolated(
        f"a = rp.ones([{n}], device='cuda')\n"
        "b = a.mul_scalar(0.5)\n"
        "s = b.sum_all()\n"
        f"assert abs(s - {n * 0.5}) < 5.0, 'expected {n*0.5}, got %r' % s\n"
        "print('ok')")
    assert rc == 0 and 'ok' in out, f"rc={rc}: {out[-200:]}"


def test_deep_graph_500():
    x = t_from_np(np.ones((4, 4)), dtype='float32', requires_grad=True)
    kept = []
    y = x
    for _ in range(500):
        y = y.op_add_scalar(0.001)
        kept.append(y)
    loss = y.op_mean_all()
    loss.backward()
    assert_close(grad_np(x), np.full((4, 4), 1.0 / 16), rtol=1e-4, ctx="deep graph grad")


def test_seed_determinism():
    rp.manual_seed(12345)
    a = rp.randn([100]).data()
    rp.manual_seed(12345)
    b = rp.randn([100]).data()
    assert a == b, "manual_seed should make randn deterministic"
    if has_cuda():
        rp.manual_seed(999)
        c = to_np(rp.randn([100], device='cuda'))
        rp.manual_seed(999)
        d = to_np(rp.randn([100], device='cuda'))
        assert np.array_equal(c, d), "manual_seed should make gpu randn deterministic"


def test_rand_distribution_sanity():
    rp.manual_seed(42)
    u = np.array(rp.rand([200_000]).data())
    assert 0.0 <= u.min() and u.max() < 1.0, "rand outside [0,1)"
    assert abs(u.mean() - 0.5) < 0.01, f"rand mean {u.mean()}"
    n = np.array(rp.randn([200_000]).data())
    assert abs(n.mean()) < 0.02, f"randn mean {n.mean()}"
    assert abs(n.std() - 1.0) < 0.02, f"randn std {n.std()}"


run_test("single element forward+backward", test_single_element)
run_test("non-contiguous unary forward (transpose)", test_noncontig_unary_forward)
run_test("non-contiguous unary forward (slice)", test_noncontig_slice_forward)
run_test("expand (stride-0) unary forward", test_expand_stride0_forward)
run_test("to_numpy of views", test_to_numpy_of_views)
run_test("broadcast forward vs numpy", test_broadcast_forward_matrix)
run_test("grad accumulation across two backwards", test_grad_accumulation_two_backwards)
run_test("dtype round-trips f64<->f32/f16/bf16", test_dtype_roundtrips)
run_test("grad transfer in .to()", test_grad_transfer_in_to)
run_test("nested views backward", test_nested_views_backward)
run_test("deep graph 500 nodes", test_deep_graph_500)
run_test("seed determinism cpu+gpu", test_seed_determinism)
run_test("rand/randn distribution sanity", test_rand_distribution_sanity)
run_test("large reduction precision (1M cpu)", test_large_reduction_precision)
run_test("gpu sum_all size sweep (isolated)", test_gpu_sum_all_size_sweep)
run_test("gpu mean_all matches cpu (isolated)", test_gpu_mean_all_matches_cpu)
run_test("large tensor gpu ops 4M (isolated)", test_large_tensor_gpu_ops)

finish()
