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
print("LayerNorm C Kernel Tests")
print("=" * 60)

print("\n--- op_layer_norm forward (no affine) ---")

x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
_xb = x
x2d = x.reshape([2, 3])
out = x2d.op_layer_norm([3])
check("shape preserved", out.shape == (2, 3))

vals = get_vals(out.reshape([6]))
row0_mean = sum(vals[0:3]) / 3.0
row1_mean = sum(vals[3:6]) / 3.0
check(f"row 0 mean ~0 (got {row0_mean:.6f})", abs(row0_mean) < 0.01)
check(f"row 1 mean ~0 (got {row1_mean:.6f})", abs(row1_mean) < 0.01)

row0_var = sum((v - row0_mean)**2 for v in vals[0:3]) / 3.0
check(f"row 0 var ~1 (got {row0_var:.4f})", abs(row0_var - 1.0) < 0.1)

print("\n--- op_layer_norm forward (with affine) ---")

w = rp.values([2.0, 2.0, 2.0])
b = rp.values([1.0, 1.0, 1.0])
out_aff = x2d.op_layer_norm([3], weight=w, bias=b)
check("affine shape", out_aff.shape == (2, 3))
vals_aff = get_vals(out_aff.reshape([6]))
row0_mean_aff = sum(vals_aff[0:3]) / 3.0
check(f"affine row 0 mean ~1 (got {row0_mean_aff:.4f})", abs(row0_mean_aff - 1.0) < 0.1)

print("\n--- op_layer_norm backward (x grad) ---")

x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
_xb2 = x
x2d = x.reshape([2, 3])
x2d.requires_grad_(True)
out = x2d.op_layer_norm([3])
out.op_sum_all().backward()
check("x has_grad", x2d.has_grad)
check("x grad shape", x2d.grad.shape == (2, 3))

print("\n--- op_layer_norm backward (weight + bias grad) ---")

x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
_xb3 = x
x2d = x.reshape([2, 3])
x2d.requires_grad_(True)
w = rp.values([1.0, 1.0, 1.0])
w.requires_grad_(True)
b = rp.values([0.0, 0.0, 0.0])
b.requires_grad_(True)
out = x2d.op_layer_norm([3], weight=w, bias=b)
out.op_sum_all().backward()
check("weight has_grad", w.has_grad)
check("bias has_grad", b.has_grad)
check("weight grad shape", w.grad.shape == (3,))
check("bias grad shape", b.grad.shape == (3,))

bg = b.grad
bias_grad_vals = get_vals(bg)
check("bias grad = [2,2,2] (sum over 2 rows)", all(approx(v, 2.0) for v in bias_grad_vals))

print("\n--- Numerical gradient check ---")

eps = 1e-4
x_vals = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]

def layer_norm_manual(vals, start, end):
    chunk = vals[start:end]
    n = len(chunk)
    m = sum(chunk) / n
    v = sum((x - m)**2 for x in chunk) / n
    rstd = 1.0 / math.sqrt(v + 1e-5)
    return [rstd * (x - m) for x in chunk]

def compute_loss(vals):
    r0 = layer_norm_manual(vals, 0, 3)
    r1 = layer_norm_manual(vals, 3, 6)
    return sum(r0) + sum(r1)

numerical = []
for i in range(6):
    vp = list(x_vals); vp[i] += eps
    vm = list(x_vals); vm[i] -= eps
    numerical.append((compute_loss(vp) - compute_loss(vm)) / (2 * eps))

x = rp.values(x_vals)
_xb4 = x
x2d = x.reshape([2, 3])
x2d.requires_grad_(True)
out = x2d.op_layer_norm([3])
out.op_sum_all().backward()
g = x2d.grad
analytic = get_vals(g.reshape([6]))

check("numerical grad check", all(approx(analytic[i], numerical[i], 1e-2) for i in range(6)))

print("\n--- Float64 ---")

x64 = rp.values([1.0, 2.0, 3.0], dtype='float64')
out64 = x64.op_layer_norm([3])
check("f64 dtype", out64.dtype == 'float64')

x64.requires_grad_(True)
out64 = x64.op_layer_norm([3])
out64.op_sum_all().backward()
check("f64 backward", x64.has_grad)

print("\n--- Float16 ---")

x16 = rp.values([1.0, 2.0, 3.0], dtype='float16')
out16 = x16.op_layer_norm([3])
check("f16 dtype", out16.dtype == 'float16')

if HAS_CUDA:
    print("\n--- CUDA ---")

    x_gpu = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cuda')
    x_gpu_2d = x_gpu.reshape([2, 3])
    out_gpu = x_gpu_2d.op_layer_norm([3])
    check("CUDA shape", out_gpu.shape == (2, 3))
    check("CUDA device", 'cuda' in out_gpu.device)

    cpu_out = out_gpu.to(device='cpu')
    vals_cuda = get_vals(cpu_out.reshape([6]))
    cuda_row0_mean = sum(vals_cuda[0:3]) / 3.0
    check(f"CUDA mean ~0 (got {cuda_row0_mean:.6f})", abs(cuda_row0_mean) < 0.01)

    x_gpu2 = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cuda')
    x_gpu2_2d = x_gpu2.reshape([2, 3])
    x_gpu2_2d.requires_grad_(True)
    out_gpu2 = x_gpu2_2d.op_layer_norm([3])
    out_gpu2.op_sum_all().backward()
    check("CUDA backward", x_gpu2_2d.has_grad)
else:
    print("\n  SKIP: No CUDA device")

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All LayerNorm kernel tests passed!")
