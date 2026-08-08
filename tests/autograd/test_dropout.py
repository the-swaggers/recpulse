import sys
import math
sys.path.insert(0, '.')
import recpulse_cuda as rp
from recpulse.module import Module, Linear, Dropout
from recpulse.optim import SGD

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1

def approx(a, b, tol=0.05):
    if abs(b) < 1e-6:
        return abs(a - b) < tol
    return abs(a - b) / max(abs(b), 1e-6) < tol

def get_vals(t):
    t32 = t.to(dtype='float32') if t.dtype != 'float32' else t
    if 'cuda' in t32.device:
        t32 = t32.to(device='cpu')
    return [rp.values([0.0]).op_add(t32.slice([i], [i+1], [1])).sum_all() for i in range(t32.size)]

HAS_CUDA = False
try:
    _t = rp.zeros([1], device='cuda')
    HAS_CUDA = True
except:
    pass

print("=" * 60)
print("Dropout Tests (C/CUDA kernel)")
print("=" * 60)

print("\n--- op_dropout forward (CPU) ---")

rp.manual_seed(42)
x = rp.ones([10000])
out = x.op_dropout(0.5)
check("output shape", out.shape == (10000,))
check("output dtype", out.dtype == 'float32')

total = out.sum_all()
kept_count = total / 2.0
check(f"~50% kept (got {kept_count:.0f}/10000)", abs(kept_count - 5000) < 500)

vals = get_vals(out.slice([0], [20], [1]))
nonzero = sum(1 for v in vals if v != 0.0)
zero = sum(1 for v in vals if v == 0.0)
check("values are either 0 or ~2.0", all(v == 0.0 or approx(v, 2.0, 0.01) for v in vals))

print("\n--- op_dropout p=0 (identity) ---")

x = rp.values([1.0, 2.0, 3.0])
out_p0 = x.op_dropout(0.0)
check("p=0 is identity", approx(out_p0.sum_all(), 6.0, 0.001))

print("\n--- op_dropout p=1 (all zeros) ---")

out_p1 = x.op_dropout(1.0)
check("p=1 is all zeros", approx(out_p1.sum_all(), 0.0, 0.001))

print("\n--- op_dropout different p values ---")

rp.manual_seed(123)
x_large = rp.ones([50000])

for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    out_p = x_large.op_dropout(p)
    scale = 1.0 / (1.0 - p)
    expected_sum = 50000.0
    actual_sum = out_p.sum_all()
    check(f"p={p} mean preservation (got {actual_sum/50000:.2f}, exp ~1.0)",
          approx(actual_sum / 50000.0, 1.0, 0.1))

print("\n--- op_dropout backward (CPU) ---")

rp.manual_seed(42)
x = rp.randn([100])
x.requires_grad_(True)
out = x.op_dropout(0.5)
loss = out.op_sum_all()
loss.backward()
check("backward has_grad", x.has_grad)
g = x.grad
check("backward grad shape", g.shape == (100,))

grad_vals = get_vals(g.slice([0], [20], [1]))
check("grad values: 0 or ~2.0", all(v == 0.0 or approx(v, 2.0, 0.01) for v in grad_vals))

print("\n--- op_dropout backward correctness ---")

rp.manual_seed(42)
x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0])
x.requires_grad_(True)
out = x.op_dropout(0.5)
out_vals = get_vals(out)
loss = out.op_sum_all()
loss.backward()
g = x.grad
grad_vals = get_vals(g)

for i in range(5):
    if out_vals[i] == 0.0:
        check(f"  elem {i}: dropped -> grad=0", approx(grad_vals[i], 0.0, 0.001))
    else:
        check(f"  elem {i}: kept -> grad=2.0", approx(grad_vals[i], 2.0, 0.01))

print("\n--- op_dropout float64 ---")

rp.manual_seed(42)
x64 = rp.randn([100], dtype='float64')
out64 = x64.op_dropout(0.5)
check("float64 dtype", out64.dtype == 'float64')

x64.requires_grad_(True)
out64 = x64.op_dropout(0.5)
out64.op_sum_all().backward()
check("float64 backward", x64.has_grad)

print("\n--- op_dropout float16 ---")

x16 = rp.randn([100], dtype='float16')
out16 = x16.op_dropout(0.5)
check("float16 dtype", out16.dtype == 'float16')

print("\n--- Dropout Module (train vs eval) ---")

drop = Dropout(0.5)
x = rp.ones([1000])

drop.train()
out_train = drop(x)
kept_train = out_train.sum_all() / 2.0
check(f"module train: ~50% kept ({kept_train:.0f}/1000)", abs(kept_train - 500) < 100)

drop.eval()
out_eval = drop(x)
check("module eval: identity", approx(out_eval.sum_all(), 1000.0, 0.001))

print("\n--- Dropout Module backward ---")

drop.train()
x2 = rp.randn([100])
x2.requires_grad_(True)
out2 = drop(x2)
out2.op_sum_all().backward()
check("module backward", x2.has_grad)

print("\n--- Training with Dropout ---")

rp.manual_seed(789)

class DropMLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 16)
        self.drop = Dropout(0.3)
        self.fc2 = Linear(16, 2)

    def forward(self, x):
        h = self.keep(self.fc1(x))
        h = self.keep(h.op_relu())
        h = self.keep(self.drop(h))
        return self.fc2(h)

model = DropMLP()
opt = SGD(model.parameters(), lr=0.5)
x = rp.randn([16, 4])
targets = [0, 1] * 8

losses = []
for step in range(40):
    model.zero_grad()
    out = model(x)
    loss = out.op_cross_entropy_loss(targets)
    loss.backward()
    opt.step()
    losses.append(loss.sum_all())

check(f"training converges ({losses[0]:.3f} -> {losses[-1]:.3f})", losses[-1] < losses[0] * 0.5)

model.eval()
out_eval = model(x)
loss_eval = out_eval.op_cross_entropy_loss(targets)
check("eval mode loss reasonable", loss_eval.sum_all() < 1.0)

print("\n--- Reproducibility ---")

rp.manual_seed(42)
a = rp.ones([100]).op_dropout(0.5)
rp.manual_seed(42)
b = rp.ones([100]).op_dropout(0.5)
diff = a.op_sub(b).op_abs().sum_all()
check("same seed same output", approx(diff, 0.0, 0.001))

rp.manual_seed(42)
c = rp.ones([100]).op_dropout(0.5)
rp.manual_seed(99)
d = rp.ones([100]).op_dropout(0.5)
diff2 = c.op_sub(d).op_abs().sum_all()
check("different seed different output", diff2 > 1.0)

if HAS_CUDA:
    print("\n--- CUDA ---")

    rp.manual_seed(42)
    x_gpu = rp.ones([10000], device='cuda')
    out_gpu = x_gpu.op_dropout(0.5)
    check("CUDA shape", out_gpu.shape == (10000,))
    check("CUDA device", 'cuda' in out_gpu.device)

    total_gpu = out_gpu.to(device='cpu').sum_all()
    kept_gpu = total_gpu / 2.0
    check(f"CUDA ~50% kept ({kept_gpu:.0f}/10000)", abs(kept_gpu - 5000) < 500)

    x_gpu2 = rp.randn([100], device='cuda')
    x_gpu2.requires_grad_(True)
    out_gpu2 = x_gpu2.op_dropout(0.5)
    out_gpu2.op_sum_all().backward()
    check("CUDA backward", x_gpu2.has_grad)
else:
    print("\n  SKIP: No CUDA device")

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All dropout tests passed!")
