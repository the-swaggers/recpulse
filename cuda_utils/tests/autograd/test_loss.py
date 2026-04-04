import sys
import math
sys.path.insert(0, '.')
import recpulse_cuda as rp

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

def sv(t):
    t32 = t.to(dtype='float32') if t.dtype != 'float32' else t
    if 'cuda' in t32.device:
        t32 = t32.to(device='cpu')
    return t32.sum_all()

_refs = []
def make_nd(data, shape, dtype='float32'):
    t = rp.values(data, dtype=dtype)
    _refs.append(t)
    return t.reshape(shape)

HAS_CUDA = False
try:
    _t = rp.zeros([1], device='cuda')
    HAS_CUDA = True
except:
    pass

print("=" * 60)
print("Loss Function Tests")
print("=" * 60)

print("\n--- MSE Loss forward ---")

pred = rp.values([1.0, 2.0, 3.0])
target = rp.values([1.5, 2.5, 3.5])
loss = pred.op_mse_loss(target)
check("MSE mean", approx(sv(loss), 0.25))

loss_sum = pred.op_mse_loss(target, reduction='sum')
check("MSE sum", approx(sv(loss_sum), 0.75))

loss_none = pred.op_mse_loss(target, reduction='none')
check("MSE none shape", loss_none.shape == (3,))

pred_zero = rp.values([1.0, 2.0, 3.0])
target_zero = rp.values([1.0, 2.0, 3.0])
check("MSE zero", approx(sv(pred_zero.op_mse_loss(target_zero)), 0.0))

print("\n--- MSE Loss backward ---")

pred = rp.values([1.0, 2.0, 3.0])
pred.requires_grad_(True)
target = rp.values([2.0, 2.0, 2.0])
loss = pred.op_mse_loss(target)
loss.backward()
check("MSE backward has_grad", pred.has_grad)
g = pred.grad
check("MSE backward shape", g.shape == (3,))

print("\n--- BCE Loss forward ---")

pred = rp.values([0.8, 0.2, 0.9])
target = rp.values([1.0, 0.0, 1.0])
loss = pred.op_bce_loss(target, from_logits=0)
expected_bce = -(math.log(0.8) + math.log(0.8) + math.log(0.9)) / 3.0
check(f"BCE from probs", approx(sv(loss), expected_bce, 0.01))

print("\n--- BCE Loss from logits ---")

logits = rp.values([2.0, -2.0, 3.0])
target = rp.values([1.0, 0.0, 1.0])
logits.requires_grad_(True)
loss = logits.op_bce_loss(target, from_logits=1)
check("BCE from logits runs", loss is not None)
loss.backward()
check("BCE from logits backward", logits.has_grad)

print("\n--- BCE auto-detect ---")

logits2 = rp.values([2.0, -2.0, 3.0])
logits2.requires_grad_(True)
probs2 = logits2.op_sigmoid()
target2 = rp.values([1.0, 0.0, 1.0])
loss_auto = probs2.op_bce_loss(target2)
check("BCE auto-detect from sigmoid", loss_auto is not None)

print("\n--- NLL Loss forward ---")

log_probs = make_nd([-2.4076, -1.4076, -0.4076, -0.4076, -1.4076, -2.4076], [2, 3])
loss = log_probs.op_nll_loss([2, 0])
check("NLL mean", approx(sv(loss), 0.4076, 0.01))

loss_sum = log_probs.op_nll_loss([2, 0], reduction='sum')
check("NLL sum", approx(sv(loss_sum), 0.8152, 0.01))

print("\n--- NLL Loss backward ---")

log_probs2 = make_nd([-2.4076, -1.4076, -0.4076, -0.4076, -1.4076, -2.4076], [2, 3])
log_probs2.requires_grad_(True)
loss = log_probs2.op_nll_loss([2, 0])
loss.backward()
check("NLL backward has_grad", log_probs2.has_grad)

print("\n--- Cross-Entropy Loss forward ---")

logits = make_nd([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], [2, 3])
loss = logits.op_cross_entropy_loss([2, 0])
check("CE from logits", approx(sv(loss), 0.4076, 0.01))

print("\n--- Cross-Entropy Loss backward ---")

logits2 = make_nd([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], [2, 3])
logits2.requires_grad_(True)
loss = logits2.op_cross_entropy_loss([2, 0])
loss.backward()
check("CE backward has_grad", logits2.has_grad)
g = logits2.grad
check("CE backward grad shape", g.shape == (2, 3))

print("\n--- CE auto-detect logits vs probs ---")

logits3 = make_nd([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], [2, 3])
logits3.requires_grad_(True)
loss_logits = logits3.op_cross_entropy_loss([2, 0])

logits4 = make_nd([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], [2, 3])
logits4.requires_grad_(True)
probs4 = logits4.op_softmax(dim=1)
loss_probs = probs4.op_cross_entropy_loss([2, 0])

check("CE auto-detect equal", approx(sv(loss_logits), sv(loss_probs), 0.001))

print("\n--- CE from_logits override ---")

logits5 = make_nd([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], [2, 3])
loss_forced = logits5.op_cross_entropy_loss([2, 0], from_logits=1)
check("CE forced from_logits", approx(sv(loss_forced), 0.4076, 0.01))

print("\n--- Numerical gradient check (MSE) ---")

eps = 1e-4
x_vals = [1.0, 2.5, 3.0]
t_vals = [2.0, 2.0, 2.0]
numerical = []
for i in range(3):
    xp = list(x_vals); xp[i] += eps
    xm = list(x_vals); xm[i] -= eps
    lp = sum((xp[j]-t_vals[j])**2 for j in range(3)) / 3.0
    lm = sum((xm[j]-t_vals[j])**2 for j in range(3)) / 3.0
    numerical.append((lp - lm) / (2 * eps))

pred = rp.values(x_vals)
pred.requires_grad_(True)
target = rp.values(t_vals)
loss = pred.op_mse_loss(target)
loss.backward()
g = pred.grad
analytic = [rp.values([0.0]).op_add(g.slice([i],[i+1],[1])).sum_all() for i in range(3)]
check("MSE numerical grad", all(approx(analytic[i], numerical[i], 1e-3) for i in range(3)))

print("\n--- Numerical gradient check (CE) ---")

def softmax_ref(vals):
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]

x_vals = [1.0, 2.0, 0.5]
targets = [1]
numerical_ce = []
for i in range(3):
    xp = list(x_vals); xp[i] += eps
    xm = list(x_vals); xm[i] -= eps
    sp = softmax_ref(xp)
    sm = softmax_ref(xm)
    lp = -math.log(sp[targets[0]])
    lm = -math.log(sm[targets[0]])
    numerical_ce.append((lp - lm) / (2 * eps))

logits = make_nd(x_vals, [1, 3])
logits.requires_grad_(True)
loss = logits.op_cross_entropy_loss(targets)
loss.backward()
g = logits.grad
analytic_ce = [rp.values([0.0]).op_add(g.reshape([3]).slice([i],[i+1],[1])).sum_all() for i in range(3)]
check("CE numerical grad", all(approx(analytic_ce[i], numerical_ce[i], 1e-3) for i in range(3)))

print("\n--- Float64 ---")

pred64 = rp.values([1.0, 2.0, 3.0], dtype='float64')
target64 = rp.values([1.5, 2.5, 3.5], dtype='float64')
pred64.requires_grad_(True)
loss64 = pred64.op_mse_loss(target64)
check("MSE float64", approx(sv(loss64), 0.25))
loss64.backward()
check("MSE float64 backward", pred64.has_grad)

if HAS_CUDA:
    print("\n--- CUDA ---")

    pred_gpu = rp.values([1.0, 2.0, 3.0], device='cuda')
    target_gpu = rp.values([1.5, 2.5, 3.5], device='cuda')
    loss_gpu = pred_gpu.op_mse_loss(target_gpu)
    check("CUDA MSE", approx(sv(loss_gpu), 0.25))

    pred_gpu2 = rp.values([1.0, 2.0, 3.0], device='cuda')
    pred_gpu2.requires_grad_(True)
    target_gpu2 = rp.values([2.0, 2.0, 2.0], device='cuda')
    loss_gpu2 = pred_gpu2.op_mse_loss(target_gpu2)
    loss_gpu2.backward()
    check("CUDA MSE backward", pred_gpu2.has_grad)

    logits_gpu_base = rp.values([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], device='cuda')
    logits_gpu = logits_gpu_base.reshape([2, 3])
    logits_gpu.requires_grad_(True)
    loss_gpu3 = logits_gpu.op_cross_entropy_loss([2, 0])
    check("CUDA CE forward", approx(sv(loss_gpu3), 0.4076, 0.01))
    loss_gpu3.backward()
    check("CUDA CE backward", logits_gpu.has_grad)
else:
    print("\n  SKIP: No CUDA device")

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All loss function tests passed!")
