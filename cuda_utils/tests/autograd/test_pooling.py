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
def make_4d(data, shape):
    t = rp.values(data)
    _refs.append(t)
    return t.reshape(shape)

HAS_CUDA = False
try:
    _t = rp.zeros([1], device='cuda')
    HAS_CUDA = True
except:
    pass

print("=" * 60)
print("Pooling Tests")
print("=" * 60)

print("\n--- MaxPool2d forward ---")

data = [float(i+1) for i in range(16)]
x = make_4d(data, [1, 1, 4, 4])

out = x.op_maxpool2d(2, 2)
check("maxpool shape (1,1,2,2)", out.shape == (1, 1, 2, 2))
check("maxpool sum = 44 (6+8+14+16)", approx(sv(out), 44.0))

print("\n--- MaxPool2d stride ---")

out_s1 = x.op_maxpool2d(2, 2, stride_h=1, stride_w=1)
check("maxpool stride=1 shape (1,1,3,3)", out_s1.shape == (1, 1, 3, 3))

print("\n--- MaxPool2d padding ---")

x_small = make_4d([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])
out_pad = x_small.op_maxpool2d(2, 2, pad_h=1, pad_w=1)
check("maxpool padding shape (1,1,2,2)", out_pad.shape == (1, 1, 2, 2))

print("\n--- MaxPool2d multi-channel batch ---")

x_mc = make_4d([float(i+1) for i in range(32)], [2, 1, 4, 4])
out_mc = x_mc.op_maxpool2d(2, 2)
check("maxpool batch shape (2,1,2,2)", out_mc.shape == (2, 1, 2, 2))

print("\n--- AvgPool2d forward ---")

x = make_4d(data, [1, 1, 4, 4])
out = x.op_avgpool2d(2, 2)
check("avgpool shape (1,1,2,2)", out.shape == (1, 1, 2, 2))
check("avgpool sum = 34 (3.5+5.5+11.5+13.5)", approx(sv(out), 34.0))

print("\n--- AvgPool2d stride ---")

out_s1 = x.op_avgpool2d(2, 2, stride_h=1, stride_w=1)
check("avgpool stride=1 shape (1,1,3,3)", out_s1.shape == (1, 1, 3, 3))

print("\n--- MaxPool2d backward ---")

x = make_4d(data, [1, 1, 4, 4])
x.requires_grad_(True)
out = x.op_maxpool2d(2, 2)
out.op_sum_all().backward()
check("maxpool backward has_grad", x.has_grad)
check("maxpool backward shape", x.grad.shape == (1, 1, 4, 4))

g = x.grad
g_flat = g.reshape([16])
g_vals = [rp.values([0.0]).op_add(g_flat.slice([i],[i+1],[1])).sum_all() for i in range(16)]
check("maxpool grad: only max positions get 1", approx(g_vals[5], 1.0) and approx(g_vals[7], 1.0) and approx(g_vals[13], 1.0) and approx(g_vals[15], 1.0))
check("maxpool grad: non-max positions get 0", approx(g_vals[0], 0.0) and approx(g_vals[1], 0.0) and approx(g_vals[4], 0.0))

print("\n--- AvgPool2d backward ---")

x = make_4d(data, [1, 1, 4, 4])
x.requires_grad_(True)
out = x.op_avgpool2d(2, 2)
out.op_sum_all().backward()
check("avgpool backward has_grad", x.has_grad)
check("avgpool backward shape", x.grad.shape == (1, 1, 4, 4))

g = x.grad
g_flat = g.reshape([16])
g_vals = [rp.values([0.0]).op_add(g_flat.slice([i],[i+1],[1])).sum_all() for i in range(16)]
check("avgpool grad: all 0.25 (each elem in exactly one 2x2 window)", all(approx(v, 0.25) for v in g_vals))

print("\n--- Numerical gradient check (maxpool) ---")

eps = 1e-3
x_vals = [float(i+1) for i in range(16)]

def maxpool_manual(vals):
    out = 0.0
    for oh in range(2):
        for ow in range(2):
            m = -1e30
            for kh in range(2):
                for kw in range(2):
                    m = max(m, vals[(oh*2+kh)*4 + ow*2+kw])
            out += m
    return out

numerical = []
for i in range(16):
    xp = list(x_vals); xp[i] += eps
    xm = list(x_vals); xm[i] -= eps
    numerical.append((maxpool_manual(xp) - maxpool_manual(xm)) / (2*eps))

x = make_4d(x_vals, [1, 1, 4, 4])
x.requires_grad_(True)
out = x.op_maxpool2d(2, 2)
out.op_sum_all().backward()
g = x.grad
g_flat = g.reshape([16])
analytic = [rp.values([0.0]).op_add(g_flat.slice([i],[i+1],[1])).sum_all() for i in range(16)]
check("maxpool numerical grad", all(approx(analytic[i], numerical[i], 0.01) for i in range(16)))

print("\n--- Float64 ---")

x64 = make_4d(data, [1, 1, 4, 4]).to(dtype='float64')
out64 = x64.op_maxpool2d(2, 2)
check("maxpool float64", out64.dtype == 'float64' and approx(sv(out64), 44.0))
out64_avg = x64.op_avgpool2d(2, 2)
check("avgpool float64", out64_avg.dtype == 'float64' and approx(sv(out64_avg), 34.0))

if HAS_CUDA:
    print("\n--- CUDA ---")

    x_gpu = rp.values(data, device='cuda')
    x_gpu_4d = x_gpu.reshape([1, 1, 4, 4])

    out_gpu = x_gpu_4d.op_maxpool2d(2, 2)
    check("CUDA maxpool shape", out_gpu.shape == (1, 1, 2, 2))
    check("CUDA maxpool sum", approx(sv(out_gpu), 44.0))

    out_gpu_avg = x_gpu_4d.op_avgpool2d(2, 2)
    check("CUDA avgpool sum", approx(sv(out_gpu_avg), 34.0))

    x_gpu2 = rp.values(data, device='cuda')
    x_gpu2_4d = x_gpu2.reshape([1, 1, 4, 4])
    x_gpu2_4d.requires_grad_(True)
    out_gpu2 = x_gpu2_4d.op_maxpool2d(2, 2)
    out_gpu2.op_sum_all().backward()
    check("CUDA maxpool backward", x_gpu2_4d.has_grad)

    x_gpu3 = rp.values(data, device='cuda')
    x_gpu3_4d = x_gpu3.reshape([1, 1, 4, 4])
    x_gpu3_4d.requires_grad_(True)
    out_gpu3 = x_gpu3_4d.op_avgpool2d(2, 2)
    out_gpu3.op_sum_all().backward()
    check("CUDA avgpool backward", x_gpu3_4d.has_grad)
else:
    print("\n  SKIP: No CUDA device")

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All pooling tests passed!")
