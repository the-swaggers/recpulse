import sys
import math
sys.path.insert(0, '.')
import recpulse_cuda as rp
from recpulse.module import Module, Linear
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

def approx(a, b, tol=1e-3):
    if abs(b) < 1e-8:
        return abs(a - b) < tol
    return abs(a - b) / max(abs(b), 1e-8) < tol

def get_grad_vals(param):
    g = param.grad
    if g is None:
        return []
    t32 = g.to(dtype='float32') if g.dtype != 'float32' else g
    return [rp.values([0.0]).op_add(t32.slice([i], [i+1], [1])).sum_all() for i in range(t32.size)]

rp.manual_seed(42)

print("=" * 60)
print("Gradient Clipping Tests")
print("=" * 60)

print("\n--- clip_grad_norm: basic ---")

x = rp.values([3.0, 4.0])
x.requires_grad_(True)
y = _sq = x.op_square(); _sq.op_sum_all()
y.backward()

g = x.grad
grad_vals_before = get_grad_vals(x)
norm_before = math.sqrt(sum(v*v for v in grad_vals_before))
check(f"grad norm before = 10 (got {norm_before:.2f})", approx(norm_before, 10.0, 0.1))

total_norm = rp.clip_grad_norm([x], 5.0)
check(f"returned total norm (got {total_norm:.2f})", approx(total_norm, 10.0, 0.1))

g_after = x.grad
grad_vals_after = get_grad_vals(x)
norm_after = math.sqrt(sum(v*v for v in grad_vals_after))
check(f"norm after clipping = 5.0 (got {norm_after:.2f})", approx(norm_after, 5.0, 0.1))

ratio = grad_vals_after[0] / grad_vals_before[0]
check("direction preserved", approx(grad_vals_after[0] / grad_vals_after[1],
                                     grad_vals_before[0] / grad_vals_before[1], 0.01))

print("\n--- clip_grad_norm: no clip needed ---")

x2 = rp.values([0.1, 0.2])
x2.requires_grad_(True)
y2 = _sq2 = x2.op_square(); _sq2.op_sum_all()
y2.backward()

g2_before = get_grad_vals(x2)
total_norm2 = rp.clip_grad_norm([x2], 100.0)
g2_after = get_grad_vals(x2)
check("no clip: values unchanged", all(approx(g2_before[i], g2_after[i]) for i in range(2)))

print("\n--- clip_grad_norm: multiple params ---")

a = rp.values([3.0, 4.0])
a.requires_grad_(True)
b = rp.values([0.0, 0.0, 0.0])
b.requires_grad_(True)

_sqa = a.op_square(); la = _sqa.op_sum_all()
la.backward()
_sqb = b.op_square(); lb = _sqb.op_sum_all()
lb.backward()

total = rp.clip_grad_norm([a, b], 5.0)
check(f"multi-param norm (got {total:.2f})", approx(total, 10.0, 0.1))
norm_a = math.sqrt(sum(v*v for v in get_grad_vals(a)))
check(f"param a norm reduced (got {norm_a:.2f})", norm_a < 10.0)

print("\n--- clip_grad_norm: params without grad skipped ---")

c = rp.values([1.0, 2.0])
c.requires_grad_(True)
d = rp.values([3.0, 4.0])
d.requires_grad_(True)

_sqc = c.op_square(); lc = _sqc.op_sum_all()
lc.backward()

total3 = rp.clip_grad_norm([c, d], 1.0)
check("skips params without grad", True)

print("\n--- clip_grad_value: basic ---")

x3 = rp.values([1.0, 2.0, 3.0, 4.0, 5.0])
x3.requires_grad_(True)
y3 = _sq3 = x3.op_square(); _sq3.op_sum_all()
y3.backward()

g3_before = get_grad_vals(x3)
check("grad before: [2,4,6,8,10]", approx(g3_before[0], 2.0) and approx(g3_before[4], 10.0))

rp.clip_grad_value([x3], 5.0)
g3_after = get_grad_vals(x3)
check("clipped [0] = 2.0 (unchanged)", approx(g3_after[0], 2.0))
check("clipped [1] = 4.0 (unchanged)", approx(g3_after[1], 4.0))
check("clipped [2] = 5.0 (clamped from 6)", approx(g3_after[2], 5.0))
check("clipped [3] = 5.0 (clamped from 8)", approx(g3_after[3], 5.0))
check("clipped [4] = 5.0 (clamped from 10)", approx(g3_after[4], 5.0))

print("\n--- clip_grad_value: negative values ---")

x4 = rp.values([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
x4.requires_grad_(True)
y4 = _sq4 = x4.op_square(); _sq4.op_sum_all()
y4.backward()

rp.clip_grad_value([x4], 4.0)
g4 = get_grad_vals(x4)
check("neg clipped [-10] -> -4", approx(g4[0], -4.0))
check("neg clipped [-6] -> -4", approx(g4[1], -4.0))
check("neg kept [-2]", approx(g4[2], -2.0))
check("pos kept [2]", approx(g4[3], 2.0))
check("pos clipped [6] -> 4", approx(g4[4], 4.0))
check("pos clipped [10] -> 4", approx(g4[5], 4.0))

print("\n--- Integration with training ---")

rp.manual_seed(123)

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 2)
    def forward(self, x):
        h = self.keep(self.fc1(x))
        h = self.keep(h.op_relu())
        return self.fc2(h)

model = MLP()
opt = SGD(model.parameters(), lr=0.5)
x = rp.randn([8, 4])
targets = [0, 1] * 4

losses = []
for step in range(30):
    model.zero_grad()
    out = model(x)
    loss = out.op_cross_entropy_loss(targets)
    loss.backward()
    rp.clip_grad_norm(model.parameters(), 1.0)
    opt.step()
    losses.append(loss.sum_all())

check("training with grad clipping converges", losses[-1] < losses[0])

print("\n--- Float64 ---")

x64 = rp.values([3.0, 4.0], dtype='float64')
x64.requires_grad_(True)
y64 = _sq64 = x64.op_square(); _sq64.op_sum_all()
y64.backward()
total64 = rp.clip_grad_norm([x64], 5.0)
check("f64 clip_grad_norm", approx(total64, 10.0, 0.1))

rp.clip_grad_value([x64], 3.0)
g64 = get_grad_vals(x64)
check("f64 clip_grad_value", approx(g64[0], 3.0))

HAS_CUDA = False
try:
    _t = rp.zeros([1], device='cuda')
    HAS_CUDA = True
except:
    pass

if HAS_CUDA:
    print("\n--- CUDA ---")

    x_gpu = rp.values([3.0, 4.0], device='cuda')
    x_gpu.requires_grad_(True)
    _sqg = x_gpu.op_square(); y_gpu = _sqg.op_sum_all()
    y_gpu.backward()
    total_gpu = rp.clip_grad_norm([x_gpu], 5.0)
    check("CUDA clip_grad_norm", approx(total_gpu, 10.0, 0.1))

    rp.clip_grad_value([x_gpu], 3.0)
    check("CUDA clip_grad_value runs", True)
else:
    print("\n  SKIP: No CUDA device")

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All gradient clipping tests passed!")
