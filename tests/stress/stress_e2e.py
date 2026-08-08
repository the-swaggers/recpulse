#!/usr/bin/env python3
import os
from stress_common import *
from recpulse.module import Module, Linear, BatchNorm2d, Dropout
from recpulse.optim import Adam, SGD
from recpulse.scheduler import StepLR

RNG = np.random.default_rng(2026)

N, DIN, DH, NCLASS = 96, 16, 32, 3
TEACHER = RNG.normal(size=(DIN, NCLASS))
X_NP = RNG.normal(size=(N, DIN))
Y_LIST = [int(i) for i in np.argmax(X_NP @ TEACHER + 0.1 * RNG.normal(size=(N, NCLASS)), axis=1)]


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(DIN, DH)
        self.fc2 = Linear(DH, NCLASS)

    def forward(self, x):
        h = self.keep(self.fc1(x))
        a = self.keep(h.op_relu())
        return self.fc2(a)


def make_model(seed, device='cpu'):
    rp.manual_seed(seed)
    m = MLP()
    if device != 'cpu':
        m = m.to(device=device)
    return m


def train(model, device='cpu', steps=60, lr=1e-2, opt=None):
    xb = t_from_np(X_NP, dtype='float32', device=device)
    if opt is None:
        opt = Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        model.zero_grad()
        logits = model(xb)
        loss = logits.op_cross_entropy_loss(Y_LIST, 'mean', 1)
        loss.backward()
        opt.step()
        losses.append(loss.to(device='cpu').data()[0] if device != 'cpu' else loss.data()[0])
        del logits, loss
    return losses, opt


def accuracy(model, device='cpu'):
    xb = t_from_np(X_NP, dtype='float32', device=device)
    logits = model(xb)
    pred = np.argmax(to_np(logits), axis=1)
    model.zero_grad()
    return float((pred == np.array(Y_LIST)).mean())


def test_train_cpu():
    model = make_model(5)
    losses, _ = train(model, steps=80)
    acc = accuracy(model)
    print(f"    loss {losses[0]:.3f} -> {losses[-1]:.3f}, acc {acc:.2%}")
    assert losses[-1] < 0.5 * losses[0], f"loss did not halve: {losses[0]} -> {losses[-1]}"
    assert acc > 0.85, f"accuracy too low: {acc}"


def test_train_gpu():
    if not has_cuda():
        return
    model = make_model(5, device='cuda')
    losses, _ = train(model, device='cuda', steps=80)
    acc = accuracy(model, device='cuda')
    print(f"    loss {losses[0]:.3f} -> {losses[-1]:.3f}, acc {acc:.2%}")
    assert losses[-1] < 0.5 * losses[0], f"loss did not halve: {losses[0]} -> {losses[-1]}"
    assert acc > 0.85, f"accuracy too low: {acc}"


def test_determinism():
    m1 = make_model(11)
    l1, _ = train(m1, steps=30)
    m2 = make_model(11)
    l2, _ = train(m2, steps=30)
    assert abs(l1[-1] - l2[-1]) < 1e-10, f"non-deterministic: {l1[-1]} vs {l2[-1]}"


def test_save_load_roundtrip_rpt():
    model = make_model(7)
    train(model, steps=30)
    path = '/tmp/stress_e2e_mlp.rpt'
    rp.save(model.tracked, path)

    fresh = make_model(99)
    fresh.load_state(rp.load(path))
    os.remove(path)

    xb = t_from_np(X_NP, dtype='float32')
    la = model(xb).op_cross_entropy_loss(Y_LIST, 'mean', 1).data()[0]
    model.zero_grad()
    lb = fresh(xb).op_cross_entropy_loss(Y_LIST, 'mean', 1).data()[0]
    fresh.zero_grad()
    assert abs(la - lb) < 1e-6, f"loss after load differs: {la} vs {lb}"


def test_save_load_resume_training():
    model = make_model(13)
    train(model, steps=20)
    path = '/tmp/stress_e2e_resume.rpt'
    rp.save(model.tracked, path)

    resumed = make_model(50)
    resumed.load_state(rp.load(path))
    os.remove(path)
    l1, _ = train(model, steps=10)
    l2, _ = train(resumed, steps=10)
    assert abs(l1[0] - l2[0]) < 1e-6, f"first resumed loss differs: {l1[0]} vs {l2[0]}"


def test_safetensors_roundtrip():
    try:
        import safetensors  # noqa: F401
    except ImportError:
        print("    (safetensors not installed, skipping)")
        return
    from recpulse.serialize import save, load
    state = {"w": t_from_np(RNG.normal(size=(4, 5)), dtype='float32'),
             "b": t_from_np(RNG.normal(size=(5,)), dtype='float32')}
    path = '/tmp/stress_e2e.safetensors'
    save(state, path)
    loaded = load(path)
    os.remove(path)
    for k in state:
        assert_close(to_np(loaded[k]), to_np(state[k]), rtol=1e-6, ctx=f"safetensors {k}")


def test_rpt_dtype_preservation():
    state = {
        "f64": t_from_np(RNG.normal(size=(3, 3)), dtype='float64'),
        "f32": t_from_np(RNG.normal(size=(3, 3)), dtype='float32'),
        "f16": t_from_np(RNG.normal(size=(3, 3)), dtype='float16'),
    }
    path = '/tmp/stress_e2e_dtypes.rpt'
    rp.save(state, path)
    loaded = rp.load(path)
    os.remove(path)
    for k, t in state.items():
        assert loaded[k].dtype == t.dtype, f"{k}: dtype {loaded[k].dtype} != {t.dtype}"
        assert_close(to_np(loaded[k].to(dtype='float64')), to_np(t.to(dtype='float64')),
                     rtol=1e-3, atol=1e-3, ctx=f"rpt {k}")


class BNWrapper(Module):
    def __init__(self):
        super().__init__()
        self.bn = BatchNorm2d(3)

    def forward(self, x):
        return self.bn(x)


def test_batchnorm_running_stats_in_checkpoint():
    model = BNWrapper()
    model.train()
    for i in range(5):
        x = t_from_np(RNG.normal(loc=2.0, size=(8, 3, 4, 4)), dtype='float32')
        out = model(x)
        del out
    model.zero_grad()

    attr_mean = to_np(model.bn.running_mean)
    assert np.abs(attr_mean).max() > 0.1, \
        f"bn.running_mean attribute never updated: {attr_mean}"

    tracked_mean = to_np(model.tracked['bn.running_mean'])
    assert_close(tracked_mean, attr_mean, rtol=1e-6,
                 ctx="model.tracked['bn.running_mean'] vs live bn.running_mean (checkpoint staleness)")


def test_batchnorm_train_eval_consistency():
    model = BNWrapper()
    model.train()
    xs = t_from_np(RNG.normal(loc=1.0, scale=2.0, size=(64, 3, 4, 4)), dtype='float32')
    for _ in range(30):
        out = model(xs)
        del out
    model.zero_grad()
    model.eval()
    out_eval = to_np(model(xs))
    model.zero_grad()
    mean = out_eval.mean(axis=(0, 2, 3))
    std = out_eval.std(axis=(0, 2, 3))
    assert np.abs(mean).max() < 0.2, f"eval BN output mean not ~0: {mean}"
    assert np.abs(std - 1).max() < 0.2, f"eval BN output std not ~1: {std}"


def test_dropout_train_eval():
    d = Dropout(p=0.5)
    x = t_from_np(np.ones((64, 64)), dtype='float32')
    d.train()
    y = to_np(d(x))
    frac_zero = (y == 0).mean()
    assert 0.3 < frac_zero < 0.7, f"train-mode dropout zero fraction {frac_zero}"
    nz = y[y != 0]
    assert_close(nz, np.full_like(nz, 2.0), rtol=1e-5, ctx="inverted dropout scaling")
    d.eval()
    y = to_np(d(x))
    assert (y == 1.0).all(), "eval-mode dropout should be identity"


def test_optimizer_created_before_to_device():
    if not has_cuda():
        return
    model = make_model(17)
    opt = Adam(model.parameters(), lr=1e-2)
    model.to(device='cuda')
    losses, _ = train(model, device='cuda', steps=30, opt=opt)
    assert losses[-1] < 0.9 * losses[0], \
        (f"KNOWN HAZARD confirmed: optimizer created before .to() trains orphaned tensors "
         f"(loss {losses[0]:.4f} -> {losses[-1]:.4f})")


def test_train_gpu_then_migrate_cpu():
    if not has_cuda():
        return
    model = make_model(23, device='cuda')
    l_gpu, _ = train(model, device='cuda', steps=30)
    model.to(device='cpu')
    l_cpu, _ = train(model, device='cpu', steps=30)
    assert l_cpu[0] < l_gpu[0], f"migration lost progress: {l_gpu[0]} -> {l_cpu[0]}"
    assert l_cpu[-1] < l_cpu[0], "no further progress after migration"


def test_scheduler_integration():
    model = make_model(29)
    opt = Adam(model.parameters(), lr=1e-2)
    sched = StepLR(opt, step_size=10, gamma=0.5)
    xb = t_from_np(X_NP, dtype='float32')
    for _ in range(25):
        model.zero_grad()
        logits = model(xb)
        loss = logits.op_cross_entropy_loss(Y_LIST, 'mean', 1)
        loss.backward()
        rp.clip_grad_norm(model.parameters(), 5.0)
        opt.step()
        sched.step()
        del logits, loss
    assert abs(sched.get_lr() - 1e-2 * 0.25) < 1e-9, f"lr after 25 steps: {sched.get_lr()}"


def test_tokenizer_roundtrip():
    text = ("the quick brown fox jumps over the lazy dog. " * 20 +
            "pack my box with five dozen liquor jugs. " * 20)
    tok = rp.Tokenizer()
    tok.train(text, vocab_size=300, special_tokens=['<pad>', '<eos>'])
    assert tok.vocab_size >= 258, f"vocab {tok.vocab_size}"
    sample = "the quick brown dog jumps"
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample, f"roundtrip failed: {tok.decode(ids)!r}"
    path = '/tmp/stress_e2e_tok.bin'
    tok.save(path)
    tok2 = rp.load_tokenizer(path)
    os.remove(path)
    assert tok2.decode(tok2.encode(sample)) == sample, "roundtrip after save/load failed"


def test_sgd_full_training():
    model = make_model(31)
    opt = SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-4)
    losses, _ = train(model, steps=80, opt=opt)
    print(f"    loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < 0.5 * losses[0], f"SGD did not train: {losses[0]} -> {losses[-1]}"


run_test("train MLP to convergence (cpu)", test_train_cpu)
run_test("train MLP to convergence (gpu)", test_train_gpu)
run_test("deterministic training with manual_seed", test_determinism)
run_test("save/load .rpt roundtrip preserves loss", test_save_load_roundtrip_rpt)
run_test("save/load then resume training", test_save_load_resume_training)
run_test("safetensors roundtrip", test_safetensors_roundtrip)
run_test(".rpt dtype preservation (f64/f32/f16)", test_rpt_dtype_preservation)
run_test("batchnorm running stats reach checkpoint dict", test_batchnorm_running_stats_in_checkpoint)
run_test("batchnorm train->eval statistics", test_batchnorm_train_eval_consistency)
run_test("dropout train/eval semantics", test_dropout_train_eval)
run_test("optimizer created before .to(device) still trains", test_optimizer_created_before_to_device)
run_test("train on gpu then migrate to cpu", test_train_gpu_then_migrate_cpu)
run_test("scheduler + grad clipping integration", test_scheduler_integration)
run_test("BPE tokenizer train/encode/decode/save/load", test_tokenizer_roundtrip)
run_test("SGD nesterov full training", test_sgd_full_training)

finish()
