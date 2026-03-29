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

def get_vals(t):
    t32 = t.to(dtype='float32') if t.dtype != 'float32' else t
    if 'cuda' in t32.device:
        t32 = t32.to(device='cpu')
    return [rp.values([0.0]).op_add(t32.slice([i], [i+1], [1])).sum_all() for i in range(t32.size)]

def approx(a, b, tol=1e-4):
    if abs(b) < 1e-8:
        return abs(a - b) < tol
    return abs(a - b) / max(abs(b), 1e-8) < tol

def softmax_ref(vals, dim_size, offset=0):
    chunk = vals[offset:offset + dim_size]
    m = max(chunk)
    exps = [math.exp(v - m) for v in chunk]
    s = sum(exps)
    return [e / s for e in exps]

def log_softmax_ref(vals, dim_size, offset=0):
    chunk = vals[offset:offset + dim_size]
    m = max(chunk)
    lse = math.log(sum(math.exp(v - m) for v in chunk))
    return [v - m - lse for v in chunk]

HAS_CUDA = False
try:
    _t = rp.zeros([1], device='cuda')
    HAS_CUDA = True
except:
    pass

print("=" * 60)
print("Softmax / Log-Softmax Tests")
print("=" * 60)

print("\n--- Softmax forward (1D) ---")

x = rp.values([1.0, 2.0, 3.0])
s = x.op_softmax()
check("softmax 1D shape", s.shape == (3,))
check("softmax 1D dtype", s.dtype == 'float32')

vals = get_vals(s)
ref = softmax_ref([1.0, 2.0, 3.0], 3)
check("softmax 1D values", all(approx(vals[i], ref[i]) for i in range(3)))

total = sum(vals)
check("softmax sums to 1", approx(total, 1.0, 1e-5))

print("\n--- Softmax forward (2D, dim=-1) ---")

x_base = rp.values([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
x2d = x_base.reshape([2, 3])
s2d = x2d.op_softmax(dim=-1)
check("softmax 2D shape", s2d.shape == (2, 3))

vals2d = get_vals(s2d.reshape([6]))
ref_row0 = softmax_ref([1.0, 2.0, 3.0], 3)
ref_row1 = softmax_ref([4.0, 1.0, 2.0], 3)
check("softmax 2D row 0", all(approx(vals2d[i], ref_row0[i]) for i in range(3)))
check("softmax 2D row 1", all(approx(vals2d[3+i], ref_row1[i]) for i in range(3)))
check("softmax 2D row sums", approx(sum(vals2d[:3]), 1.0) and approx(sum(vals2d[3:]), 1.0))

print("\n--- Softmax forward (2D, dim=0) ---")

s2d_d0 = x2d.op_softmax(dim=0)
check("softmax dim=0 shape", s2d_d0.shape == (2, 3))
vals_d0 = get_vals(s2d_d0.reshape([6]))
for j in range(3):
    col_sum = vals_d0[j] + vals_d0[3+j]
    check(f"softmax dim=0 col {j} sums to 1", approx(col_sum, 1.0))

print("\n--- Log-softmax forward (1D) ---")

x = rp.values([1.0, 2.0, 3.0])
ls = x.op_log_softmax()
check("log_softmax 1D shape", ls.shape == (3,))

vals = get_vals(ls)
ref = log_softmax_ref([1.0, 2.0, 3.0], 3)
check("log_softmax 1D values", all(approx(vals[i], ref[i]) for i in range(3)))

check("log_softmax all negative", all(v < 0 for v in vals))
check("exp(log_softmax) sums to 1", approx(sum(math.exp(v) for v in vals), 1.0))

print("\n--- Log-softmax forward (2D) ---")

ls2d = x2d.op_log_softmax(dim=-1)
check("log_softmax 2D shape", ls2d.shape == (2, 3))

print("\n--- Softmax backward ---")

x = rp.values([1.0, 2.0, 3.0])
x.requires_grad_(True)
s = x.op_softmax()
loss = s.op_sum_all()
loss.backward()
check("softmax backward has_grad", x.has_grad)
check("softmax backward grad shape", x.grad.shape == (3,))

grad_vals = get_vals(x.grad)
check("softmax backward grad near zero (sum grad = 0)", approx(sum(grad_vals), 0.0, 1e-5))

print("\n--- Softmax backward correctness (numerical) ---")

eps = 1e-4
x_vals = [2.0, 1.0, 0.5]
numerical_grads = []
for i in range(3):
    x_plus = list(x_vals)
    x_plus[i] += eps
    x_minus = list(x_vals)
    x_minus[i] -= eps

    sp = softmax_ref(x_plus, 3)
    sm = softmax_ref(x_minus, 3)
    numerical_grads.append(sum((sp[j] - sm[j]) / (2 * eps) for j in range(3)))

x = rp.values(x_vals)
x.requires_grad_(True)
s = x.op_softmax()
loss = s.op_sum_all()
loss.backward()
analytic = get_vals(x.grad)

check("softmax numerical grad check", all(approx(analytic[i], numerical_grads[i], 1e-3) for i in range(3)))

print("\n--- Log-softmax backward ---")

x = rp.values([1.0, 2.0, 3.0])
x.requires_grad_(True)
ls = x.op_log_softmax()
loss = ls.op_sum_all()
loss.backward()
check("log_softmax backward has_grad", x.has_grad)
check("log_softmax backward grad shape", x.grad.shape == (3,))

print("\n--- Log-softmax backward correctness (numerical) ---")

numerical_grads = []
for i in range(3):
    x_plus = list(x_vals)
    x_plus[i] += eps
    x_minus = list(x_vals)
    x_minus[i] -= eps

    lsp = log_softmax_ref(x_plus, 3)
    lsm = log_softmax_ref(x_minus, 3)
    numerical_grads.append(sum((lsp[j] - lsm[j]) / (2 * eps) for j in range(3)))

x = rp.values(x_vals)
x.requires_grad_(True)
ls = x.op_log_softmax()
loss = ls.op_sum_all()
loss.backward()
analytic = get_vals(x.grad)

check("log_softmax numerical grad check", all(approx(analytic[i], numerical_grads[i], 1e-3) for i in range(3)))

print("\n--- Softmax 2D backward ---")

x_base = rp.values([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
x2d = x_base.reshape([2, 3])
x2d.requires_grad_(True)
s2d = x2d.op_softmax(dim=1)
loss2d = s2d.op_sum_all()
loss2d.backward()
check("softmax 2D backward has_grad", x2d.has_grad)
check("softmax 2D backward grad shape", x2d.grad.shape == (2, 3))

print("\n--- Float64 ---")

x_f64 = rp.values([1.0, 2.0, 3.0], dtype='float64')
s_f64 = x_f64.op_softmax()
check("softmax float64 dtype", s_f64.dtype == 'float64')

x_f64.requires_grad_(True)
s_f64 = x_f64.op_softmax()
s_f64.op_sum_all().backward()
check("softmax float64 backward", x_f64.has_grad)

print("\n--- Float16 ---")

x_f16 = rp.values([1.0, 2.0, 3.0], dtype='float16')
s_f16 = x_f16.op_softmax()
check("softmax float16 dtype", s_f16.dtype == 'float16')
check("softmax float16 sums ~1", approx(s_f16.to(dtype='float32').sum_all(), 1.0, 0.01))

ls_f16 = x_f16.op_log_softmax()
check("log_softmax float16 dtype", ls_f16.dtype == 'float16')

print("\n--- Large numerical stability ---")

x_large = rp.values([1000.0, 1001.0, 1002.0])
s_large = x_large.op_softmax()
vals_large = get_vals(s_large)
check("softmax large values stable", approx(sum(vals_large), 1.0) and all(v > 0 for v in vals_large))

x_neg = rp.values([-1000.0, -999.0, -998.0])
s_neg = x_neg.op_softmax()
vals_neg = get_vals(s_neg)
check("softmax large neg values stable", approx(sum(vals_neg), 1.0) and all(v > 0 for v in vals_neg))

print("\n--- Chained ops ---")

x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
x.requires_grad_(True)
x2d = x.reshape([2, 3])
s = x2d.op_softmax(dim=1)
s_flat = s.reshape([6])
loss = s_flat.op_sum_all()
loss.backward()
check("softmax chain backward", x.has_grad)

if HAS_CUDA:
    print("\n--- CUDA ---")

    x_gpu = rp.values([1.0, 2.0, 3.0], device='cuda')
    s_gpu = x_gpu.op_softmax()
    check("CUDA softmax shape", s_gpu.shape == (3,))
    check("CUDA softmax device", 'cuda' in s_gpu.device)
    s_cpu = s_gpu.to(device='cpu')
    vals_cuda = get_vals(s_cpu)
    check("CUDA softmax sums to 1", approx(sum(vals_cuda), 1.0))

    ls_gpu = x_gpu.op_log_softmax()
    check("CUDA log_softmax device", 'cuda' in ls_gpu.device)

    x_gpu2 = rp.values([1.0, 2.0, 3.0], device='cuda')
    x_gpu2.requires_grad_(True)
    s_gpu2 = x_gpu2.op_softmax()
    s_gpu2.op_sum_all().backward()
    check("CUDA softmax backward", x_gpu2.has_grad)

    x_gpu3 = rp.values([1.0, 2.0, 3.0], device='cuda')
    x_gpu3.requires_grad_(True)
    ls_gpu3 = x_gpu3.op_log_softmax()
    ls_gpu3.op_sum_all().backward()
    check("CUDA log_softmax backward", x_gpu3.has_grad)
else:
    print("\n  SKIP: No CUDA device")

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All softmax tests passed!")
