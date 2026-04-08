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

def approx(a, b, tol=1e-2):
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
print("BatchNorm C Kernel Tests")
print("=" * 60)

print("\n--- op_batch_norm forward (train, 4D) ---")

_refs = []
def make_4d(data, shape):
    t = rp.values(data)
    _refs.append(t)
    return t.reshape(shape)

x = make_4d([float(i) for i in range(24)], [2, 3, 2, 2])
rm = rp.zeros([3])
rv = rp.ones([3])
w = rp.ones([3])
b = rp.zeros([3])

out = x.op_batch_norm(weight=w, bias=b, running_mean=rm, running_var=rv, training=True)
check("4D shape", out.shape == (2, 3, 2, 2))

check("running_mean updated", rm.sum_all() != 0.0)

print("\n--- op_batch_norm forward (eval, 4D) ---")

out_eval = x.op_batch_norm(weight=w, bias=b, running_mean=rm, running_var=rv, training=False)
check("eval shape", out_eval.shape == (2, 3, 2, 2))

print("\n--- op_batch_norm forward (train, no affine) ---")

rm2 = rp.zeros([3])
rv2 = rp.ones([3])
out_na = x.op_batch_norm(running_mean=rm2, running_var=rv2, training=True)
check("no affine shape", out_na.shape == (2, 3, 2, 2))

print("\n--- op_batch_norm normalization check ---")

data = [1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0]
x2 = make_4d(data, [2, 3, 2, 2])
rm3 = rp.zeros([3])
rv3 = rp.ones([3])
out2 = x2.op_batch_norm(running_mean=rm3, running_var=rv3, training=True)

vals = get_vals(out2.reshape([24]))
chan0 = vals[0:4] + vals[12:16]
chan0_mean = sum(chan0) / len(chan0)
check(f"channel 0 mean ~0 (got {chan0_mean:.4f})", abs(chan0_mean) < 0.1)

print("\n--- op_batch_norm momentum ---")

x3 = make_4d([1.0]*24, [2, 3, 2, 2])
rm4 = rp.zeros([3])
rv4 = rp.ones([3])
x3.op_batch_norm(running_mean=rm4, running_var=rv4, momentum=0.5, training=True)
rm_val = rm4.sum_all()
check(f"momentum 0.5 updates running_mean (got {rm_val:.4f})", abs(rm_val - 1.5) < 0.1)

print("\n--- Float64 ---")

x64 = make_4d([float(i) for i in range(24)], [2, 3, 2, 2]).to(dtype='float64')
rm64 = rp.zeros([3], dtype='float64')
rv64 = rp.ones([3], dtype='float64')
out64 = x64.op_batch_norm(running_mean=rm64, running_var=rv64, training=True)
check("f64 dtype", out64.dtype == 'float64')

print("\n--- Float16 ---")

x16 = make_4d([float(i) for i in range(24)], [2, 3, 2, 2]).to(dtype='float16')
rm16 = rp.zeros([3], dtype='float16')
rv16 = rp.ones([3], dtype='float16')
out16 = x16.op_batch_norm(running_mean=rm16, running_var=rv16, training=True)
check("f16 dtype", out16.dtype == 'float16')

if HAS_CUDA:
    print("\n--- CUDA ---")

    x_gpu = rp.values([float(i) for i in range(24)], device='cuda')
    x_gpu_4d = x_gpu.reshape([2, 3, 2, 2])
    rm_gpu = rp.zeros([3], device='cuda')
    rv_gpu = rp.ones([3], device='cuda')
    w_gpu = rp.ones([3], device='cuda')

    out_gpu = x_gpu_4d.op_batch_norm(weight=w_gpu, running_mean=rm_gpu, running_var=rv_gpu, training=True)
    check("CUDA shape", out_gpu.shape == (2, 3, 2, 2))
    check("CUDA device", 'cuda' in out_gpu.device)

    out_gpu_eval = x_gpu_4d.op_batch_norm(weight=w_gpu, running_mean=rm_gpu, running_var=rv_gpu, training=False)
    check("CUDA eval shape", out_gpu_eval.shape == (2, 3, 2, 2))
else:
    print("\n  SKIP: No CUDA device")

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All BatchNorm kernel tests passed!")
