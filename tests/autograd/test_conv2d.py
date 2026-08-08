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
print("Conv2d Tests")
print("=" * 60)

print("\n--- Basic forward (1x1x4x4 input, 1x1x3x3 kernel) ---")

inp = make_4d([1.0]*16, [1, 1, 4, 4])
w = make_4d([1.0]*9, [1, 1, 3, 3])
out = inp.op_conv2d(w)
check("output shape (1,1,2,2)", out.shape == (1, 1, 2, 2))
check("output sum = 36", approx(sv(out), 36.0))

print("\n--- Non-trivial kernel values ---")

inp = make_4d([1.0, 2.0, 3.0, 4.0,
               5.0, 6.0, 7.0, 8.0,
               9.0, 10.0, 11.0, 12.0,
               13.0, 14.0, 15.0, 16.0], [1, 1, 4, 4])
w = make_4d([1.0, 0.0, -1.0,
             2.0, 0.0, -2.0,
             1.0, 0.0, -1.0], [1, 1, 3, 3])
out = inp.op_conv2d(w)
check("non-trivial shape", out.shape == (1, 1, 2, 2))

print("\n--- Padding ---")

inp = make_4d([1.0]*4, [1, 1, 2, 2])
w = make_4d([1.0]*9, [1, 1, 3, 3])
out = inp.op_conv2d(w, pad_h=1, pad_w=1)
check("padded output shape (1,1,2,2)", out.shape == (1, 1, 2, 2))

print("\n--- Stride ---")

inp = make_4d([1.0]*36, [1, 1, 6, 6])
w = make_4d([1.0]*9, [1, 1, 3, 3])
out = inp.op_conv2d(w, stride_h=2, stride_w=2)
check("strided output shape (1,1,2,2)", out.shape == (1, 1, 2, 2))
check("strided output sum = 36", approx(sv(out), 36.0))

print("\n--- Multi-channel ---")

inp = make_4d([1.0]*32, [1, 2, 4, 4])
w = make_4d([1.0]*54, [3, 2, 3, 3])
out = inp.op_conv2d(w)
check("multi-channel shape (1,3,2,2)", out.shape == (1, 3, 2, 2))

print("\n--- Batch ---")

inp = make_4d([1.0]*32, [2, 1, 4, 4])
w = make_4d([1.0]*9, [1, 1, 3, 3])
out = inp.op_conv2d(w)
check("batch shape (2,1,2,2)", out.shape == (2, 1, 2, 2))
check("batch sum = 72", approx(sv(out), 72.0))

print("\n--- With bias ---")

inp = make_4d([1.0]*16, [1, 1, 4, 4])
w = make_4d([1.0]*9, [1, 1, 3, 3])
b = rp.values([10.0])
out = inp.op_conv2d(w, bias=b)
check("bias shape", out.shape == (1, 1, 2, 2))
check("bias sum = 36 + 4*10 = 76", approx(sv(out), 76.0))

print("\n--- Backward (input grad) ---")

inp = make_4d([1.0]*16, [1, 1, 4, 4])
inp.requires_grad_(True)
w = make_4d([1.0]*9, [1, 1, 3, 3])
out = inp.op_conv2d(w)
loss = out.op_sum_all()
loss.backward()
check("input grad has_grad", inp.has_grad)
check("input grad shape", inp.grad.shape == (1, 1, 4, 4))

print("\n--- Backward (weight grad) ---")

inp = make_4d([1.0]*16, [1, 1, 4, 4])
w = make_4d([1.0]*9, [1, 1, 3, 3])
w.requires_grad_(True)
out = inp.op_conv2d(w)
loss = out.op_sum_all()
loss.backward()
check("weight grad has_grad", w.has_grad)
check("weight grad shape", w.grad.shape == (1, 1, 3, 3))

print("\n--- Backward (both grads) ---")

inp = make_4d([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], [1, 1, 4, 4])
inp.requires_grad_(True)
w = make_4d([1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0], [1, 1, 3, 3])
w.requires_grad_(True)
out = inp.op_conv2d(w)
loss = out.op_sum_all()
loss.backward()
check("both grads: input", inp.has_grad)
check("both grads: weight", w.has_grad)

print("\n--- Backward (bias grad) ---")

inp = make_4d([1.0]*16, [1, 1, 4, 4])
w = make_4d([1.0]*9, [1, 1, 3, 3])
b = rp.values([0.0])
b.requires_grad_(True)
out = inp.op_conv2d(w, bias=b)
loss = out.op_sum_all()
loss.backward()
check("bias grad has_grad", b.has_grad)
g = b.grad
check("bias grad shape", g.shape == (1,))
check("bias grad = 4 (4 spatial positions)", approx(sv(g), 4.0))

print("\n--- Numerical gradient check (weight) ---")

eps = 1e-3
w_vals = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0]
inp_vals = [float(i+1) for i in range(16)]

def conv2d_manual(inp_v, w_v):
    out = 0.0
    for oh in range(2):
        for ow in range(2):
            val = 0.0
            for kh in range(3):
                for kw in range(3):
                    val += inp_v[(oh+kh)*4 + (ow+kw)] * w_v[kh*3+kw]
            out += val
    return out

numerical_w = []
for i in range(9):
    wp = list(w_vals); wp[i] += eps
    wm = list(w_vals); wm[i] -= eps
    lp = conv2d_manual(inp_vals, wp)
    lm = conv2d_manual(inp_vals, wm)
    numerical_w.append((lp - lm) / (2 * eps))

inp3 = make_4d(inp_vals, [1, 1, 4, 4])
w3 = make_4d(w_vals, [1, 1, 3, 3])
w3.requires_grad_(True)
out3 = inp3.op_conv2d(w3)
out3.op_sum_all().backward()
g3 = w3.grad
analytic_w = [rp.values([0.0]).op_add(g3.reshape([9]).slice([i],[i+1],[1])).sum_all() for i in range(9)]
check("weight numerical grad", all(approx(analytic_w[i], numerical_w[i], 0.01) for i in range(9)))

print("\n--- Float64 ---")

inp64 = make_4d([1.0]*16, [1, 1, 4, 4])
inp64_f64 = inp64.to(dtype='float64')
w64 = make_4d([1.0]*9, [1, 1, 3, 3])
w64_f64 = w64.to(dtype='float64')
out64 = inp64_f64.op_conv2d(w64_f64)
check("float64 output", out64.dtype == 'float64')
check("float64 sum = 36", approx(sv(out64), 36.0))

if HAS_CUDA:
    print("\n--- CUDA ---")

    inp_gpu = rp.values([1.0]*16, device='cuda')
    inp_gpu_4d = inp_gpu.reshape([1, 1, 4, 4])
    w_gpu = rp.values([1.0]*9, device='cuda')
    w_gpu_4d = w_gpu.reshape([1, 1, 3, 3])
    out_gpu = inp_gpu_4d.op_conv2d(w_gpu_4d)
    check("CUDA shape", out_gpu.shape == (1, 1, 2, 2))
    check("CUDA sum = 36", approx(sv(out_gpu), 36.0))

    inp_gpu2 = rp.values([1.0]*16, device='cuda')
    inp_gpu2_4d = inp_gpu2.reshape([1, 1, 4, 4])
    inp_gpu2_4d.requires_grad_(True)
    w_gpu2 = rp.values([1.0]*9, device='cuda')
    w_gpu2_4d = w_gpu2.reshape([1, 1, 3, 3])
    w_gpu2_4d.requires_grad_(True)
    out_gpu2 = inp_gpu2_4d.op_conv2d(w_gpu2_4d)
    out_gpu2.op_sum_all().backward()
    check("CUDA input grad", inp_gpu2_4d.has_grad)
    check("CUDA weight grad", w_gpu2_4d.has_grad)
else:
    print("\n  SKIP: No CUDA device")

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All conv2d tests passed!")
