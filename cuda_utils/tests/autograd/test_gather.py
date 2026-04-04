import sys
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

def sv(t):
    t32 = t.to(dtype='float32') if t.dtype != 'float32' else t
    if 'cuda' in t32.device:
        t32 = t32.to(device='cpu')
    return t32.sum_all()

def get_vals(t):
    t32 = t.to(dtype='float32') if t.dtype != 'float32' else t
    if 'cuda' in t32.device:
        t32 = t32.to(device='cpu')
    t_flat = t32.reshape([t32.size]) if t32.ndim > 1 else t32
    return [rp.values([0.0]).op_add(t_flat.slice([i], [i+1], [1])).sum_all() for i in range(t32.size)]

def get_grad_vals(t):
    g = t.grad
    return get_vals(g)

def approx(a, b, tol=1e-4):
    if abs(b) < 1e-8:
        return abs(a - b) < tol
    return abs(a - b) / max(abs(b), 1e-8) < tol

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
print("Gather / Scatter Tests")
print("=" * 60)

print("\n--- Gather forward (2D, dim=1) ---")

x = make_nd([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [2, 3])
idx = make_nd([2.0, 0.0, 1.0, 2.0], [2, 2])
out = x.op_gather(1, idx)
check("gather shape", out.shape == (2, 2))
vals = get_vals(out.reshape([4]))
check("gather values [30,10,50,60]", approx(vals[0], 30) and approx(vals[1], 10) and approx(vals[2], 50) and approx(vals[3], 60))

print("\n--- Gather forward (2D, dim=0) ---")

x = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
idx = make_nd([1.0, 0.0, 1.0], [1, 3])
out = x.op_gather(0, idx)
check("gather dim=0 shape", out.shape == (1, 3))
vals = get_vals(out.reshape([3]))
check("gather dim=0 values [4,2,6]", approx(vals[0], 4) and approx(vals[1], 2) and approx(vals[2], 6))

print("\n--- Gather forward (1D) ---")

x = rp.values([10.0, 20.0, 30.0, 40.0, 50.0])
idx = rp.values([4.0, 0.0, 2.0])
out = x.op_gather(0, idx)
check("gather 1D shape", out.shape == (3,))
vals = get_vals(out)
check("gather 1D values [50,10,30]", approx(vals[0], 50) and approx(vals[1], 10) and approx(vals[2], 30))

print("\n--- Gather backward (2D, dim=1) ---")

x = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
x.requires_grad_(True)
idx = make_nd([0.0, 2.0, 1.0, 0.0], [2, 2])
g = x.op_gather(1, idx)
s = g.op_sum_all()
s.backward()
check("gather backward has_grad", x.has_grad)
check("gather backward grad shape", x.grad.shape == (2, 3))

grad_vals = get_grad_vals(x)
check("gather grad [0] = 1 (selected once from row 0)", approx(grad_vals[0], 1.0))
check("gather grad [2] = 1 (selected once from row 0)", approx(grad_vals[2], 1.0))
check("gather grad [1] = 0 (not selected from row 0)", approx(grad_vals[1], 0.0))
check("gather grad [4] = 1 (selected once from row 1)", approx(grad_vals[4], 1.0))
check("gather grad [3] = 1 (selected once from row 1)", approx(grad_vals[3], 1.0))

print("\n--- Gather backward with duplicate indices ---")

x = make_nd([1.0, 2.0, 3.0], [1, 3])
x.requires_grad_(True)
idx = make_nd([1.0, 1.0, 1.0], [1, 3])
g = x.op_gather(1, idx)
s = g.op_sum_all()
s.backward()
grad_vals = get_grad_vals(x)
check("gather dup idx: grad at idx=1 is 3", approx(grad_vals[1], 3.0))
check("gather dup idx: grad at idx=0 is 0", approx(grad_vals[0], 0.0))
check("gather dup idx: grad at idx=2 is 0", approx(grad_vals[2], 0.0))

print("\n--- Gather backward (1D) ---")

x = rp.values([10.0, 20.0, 30.0])
x.requires_grad_(True)
idx = rp.values([2.0, 0.0])
g = x.op_gather(0, idx)
s = g.op_sum_all()
s.backward()
check("gather 1D backward", x.has_grad)
grad_vals = get_grad_vals(x)
check("gather 1D grad [0]=1,[1]=0,[2]=1", approx(grad_vals[0], 1.0) and approx(grad_vals[1], 0.0) and approx(grad_vals[2], 1.0))

print("\n--- Float64 ---")

x = rp.values([1.0, 2.0, 3.0], dtype='float64')
x.requires_grad_(True)
idx = rp.values([2.0, 0.0], dtype='float64')
g = x.op_gather(0, idx)
check("gather float64 dtype", g.dtype == 'float64')
g.op_sum_all().backward()
check("gather float64 backward", x.has_grad)

print("\n--- Float16 ---")

x = rp.values([1.0, 2.0, 3.0], dtype='float16')
idx = rp.values([2.0, 0.0], dtype='float16')
g = x.op_gather(0, idx)
check("gather float16 dtype", g.dtype == 'float16')

if HAS_CUDA:
    print("\n--- CUDA ---")

    x_gpu = rp.values([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], device='cuda')
    x_gpu_2d = x_gpu.reshape([2, 3])
    idx_base = rp.values([2.0, 0.0, 1.0, 2.0])
    idx_gpu = idx_base.reshape([2, 2])
    out_gpu = x_gpu_2d.op_gather(1, idx_gpu)
    check("CUDA gather shape", out_gpu.shape == (2, 2))
    check("CUDA gather device", 'cuda' in out_gpu.device)
    out_cpu = out_gpu.to(device='cpu')
    vals = get_vals(out_cpu)
    check("CUDA gather values", approx(vals[0], 30) and approx(vals[1], 10))

    x_gpu2 = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cuda')
    x_gpu2_2d = x_gpu2.reshape([2, 3])
    x_gpu2_2d.requires_grad_(True)
    idx_base2 = rp.values([0.0, 2.0, 1.0, 0.0])
    idx2 = idx_base2.reshape([2, 2])
    g_gpu = x_gpu2_2d.op_gather(1, idx2)
    g_gpu.op_sum_all().backward()
    check("CUDA gather backward", x_gpu2_2d.has_grad)
else:
    print("\n  SKIP: No CUDA device")

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All gather tests passed!")
