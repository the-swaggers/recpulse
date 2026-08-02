#!/usr/bin/env python3
import gc
import subprocess
from stress_common import *
from recpulse.module import Module, Linear
from recpulse.optim import Adam, SGD


def rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024.0
    return -1.0


def vram_mb():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        return float(out)
    except Exception:
        return -1.0


def measure_growth(step_fn, iters=300, warmup=30, gpu=False):
    for _ in range(warmup):
        step_fn()
    gc.collect()
    before = vram_mb() if gpu else rss_mb()
    for _ in range(iters):
        step_fn()
    gc.collect()
    after = vram_mb() if gpu else rss_mb()
    return after - before


class TinyMLP(Module):
    def __init__(self, din=64, dh=128, dout=8):
        super().__init__()
        self.fc1 = Linear(din, dh)
        self.fc2 = Linear(dh, dout)

    def forward(self, x):
        h = self.keep(self.fc1(x))
        a = self.keep(h.op_relu())
        return self.fc2(a)


def test_tensor_create_free_cpu():
    def step():
        t = rp.randn([256, 256])
        s = t.sum_all()
        del t
    growth = measure_growth(step, iters=400)
    print(f"    rss growth: {growth:.1f} MB")
    assert growth < 25, f"RSS grew {growth:.1f} MB over 400 create/free iters"


def test_view_chain_create_free_cpu():
    def step():
        t = rp.randn([128, 128])
        v = t.transpose(0, 1)
        w = v.reshape([64, 256])
        s = w.slice([0, 0], [32, 128], [1, 1])
        c = s.copy()
        del t, v, w, s, c
    growth = measure_growth(step, iters=400)
    print(f"    rss growth: {growth:.1f} MB")
    assert growth < 25, f"RSS grew {growth:.1f} MB over 400 view-chain iters"


def test_raw_autograd_loop_cpu():
    x = rp.randn([64, 64]).requires_grad_()

    def step():
        y = x.op_square()
        z = y.op_mul_scalar(0.5)
        loss = z.op_mean_all()
        loss.backward()
        x.zero_grad()
        del y, z, loss
    growth = measure_growth(step, iters=400)
    print(f"    rss growth: {growth:.1f} MB")
    assert growth < 25, f"RSS grew {growth:.1f} MB over 400 fwd/bwd iters"


def make_training_step(device):
    model = TinyMLP().to(device=device)
    opt = Adam(model.parameters(), lr=1e-3)
    rp.manual_seed(7)
    xb = rp.randn([32, 64], device=device)
    yb = rp.randn([32, 8], device=device)

    def step():
        model.zero_grad()
        out = model(xb)
        loss = out.op_mse_loss(yb, 'mean')
        loss.backward()
        opt.step()
        del out, loss
    return step


def test_training_loop_rss_cpu():
    step = make_training_step('cpu')
    growth = measure_growth(step, iters=300)
    print(f"    rss growth: {growth:.1f} MB")
    assert growth < 30, f"RSS grew {growth:.1f} MB over 300 training steps"


def test_training_loop_vram_gpu():
    if not has_cuda():
        return
    step = make_training_step('cuda')
    growth = measure_growth(step, iters=300, gpu=True)
    print(f"    vram growth: {growth:.1f} MB")
    assert growth < 100, f"VRAM grew {growth:.1f} MB over 300 training steps"


def test_eval_loop_rss_known_hazard():
    model = TinyMLP()
    model.eval()
    xb = rp.randn([32, 64])

    def step():
        out = model(xb)
        del out
    growth = measure_growth(step, iters=400)
    print(f"    rss growth (eval loop, no zero_grad): {growth:.1f} MB")
    assert growth < 25, \
        f"KNOWN HAZARD confirmed: eval-only loop leaked {growth:.1f} MB (kept intermediates never cleared)"


def test_optimizer_zero_grad_only_known_hazard():
    model = TinyMLP()
    opt = SGD(model.parameters(), lr=1e-3)
    xb = rp.randn([32, 64])
    yb = rp.randn([32, 8])

    def step():
        opt.zero_grad()
        out = model(xb)
        loss = out.op_mse_loss(yb, 'mean')
        loss.backward()
        opt.step()
        del out, loss
    growth = measure_growth(step, iters=300)
    print(f"    rss growth (opt.zero_grad pattern): {growth:.1f} MB")
    assert growth < 30, \
        f"KNOWN HAZARD confirmed: optimizer.zero_grad()-only loop leaked {growth:.1f} MB"


def test_gpu_tensor_create_free():
    if not has_cuda():
        return
    def step():
        t = rp.randn([512, 512], device='cuda')
        u = t.mul_scalar(2.0)
        del t, u
    growth = measure_growth(step, iters=300, gpu=True)
    print(f"    vram growth: {growth:.1f} MB")
    assert growth < 50, f"VRAM grew {growth:.1f} MB over 300 gpu create/free iters"


def test_transfer_loop():
    if not has_cuda():
        return
    def step():
        t = rp.randn([256, 256])
        g = t.to(device='cuda')
        b = g.to(device='cpu')
        del t, g, b
    growth_v = measure_growth(step, iters=200, gpu=True)
    print(f"    vram growth: {growth_v:.1f} MB")
    assert growth_v < 50, f"VRAM grew {growth_v:.1f} MB over 200 transfer round-trips"


def test_serialize_loop():
    import os
    state = {f"p{i}": rp.randn([64, 64]) for i in range(4)}
    path = '/tmp/stress_serialize_loop.rpt'

    def step():
        rp.save(state, path)
        loaded = rp.load(path)
        del loaded
    growth = measure_growth(step, iters=200)
    os.remove(path)
    print(f"    rss growth: {growth:.1f} MB")
    assert growth < 25, f"RSS grew {growth:.1f} MB over 200 save/load iters"


def test_backward_graph_teardown():
    def step():
        x = rp.randn([64, 64]).requires_grad_()
        kept = []
        y = x
        for _ in range(20):
            y = y.op_add_scalar(0.01)
            kept.append(y)
        loss = y.op_sum_all()
        loss.backward()
        del x, y, loss, kept
    growth = measure_growth(step, iters=200)
    print(f"    rss growth: {growth:.1f} MB")
    assert growth < 25, f"RSS grew {growth:.1f} MB over 200 deep-graph iters"


run_test("tensor create/free loop (cpu rss)", test_tensor_create_free_cpu)
run_test("view chain create/free loop (cpu rss)", test_view_chain_create_free_cpu)
run_test("raw autograd fwd/bwd loop (cpu rss)", test_raw_autograd_loop_cpu)
run_test("training loop Module+Adam (cpu rss)", test_training_loop_rss_cpu)
run_test("training loop Module+Adam (gpu vram)", test_training_loop_vram_gpu)
run_test("eval-only loop (known hazard: kept intermediates)", test_eval_loop_rss_known_hazard)
run_test("optimizer.zero_grad-only loop (known hazard)", test_optimizer_zero_grad_only_known_hazard)
run_test("gpu tensor create/free loop (vram)", test_gpu_tensor_create_free)
run_test("cpu<->gpu transfer loop (vram)", test_transfer_loop)
run_test("save/load loop (cpu rss)", test_serialize_loop)
run_test("deep graph build/teardown loop (cpu rss)", test_backward_graph_teardown)

finish()
