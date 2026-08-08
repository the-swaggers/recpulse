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
    if 'cuda' in (t32.device if hasattr(t32, 'device') else 'cpu'):
        t32 = t32.to(device='cpu')
    return [rp.values([0.0]).op_add(t32.slice([i], [i+1], [1])).sum_all() for i in range(t32.size)]

def sv(t):
    t32 = t
    if t.dtype != 'float32':
        t32 = t.to(dtype='float32')
    if 'cuda' in t32.device:
        t32 = t32.to(device='cpu')
    return t32.sum_all()

HAS_CUDA = False
try:
    _t = rp.zeros([1], device='cuda')
    HAS_CUDA = True
except:
    pass

N = 10000

print("=" * 60)
print("Random Number Generation Tests")
print("=" * 60)

print("\n--- rp.rand() basics ---")

rp.manual_seed(123)
t = rp.rand([3, 4])
check("rand shape", t.shape == (3, 4))
check("rand dtype", t.dtype == 'float32')
check("rand device", t.device == 'cpu')

t_large = rp.rand([N])
vals = get_vals(t_large)
check("rand range [0,1)", all(0.0 <= v < 1.0 for v in vals))
mean = sum(vals) / len(vals)
check(f"rand mean ~0.5 (got {mean:.4f})", abs(mean - 0.5) < 0.02)

t_f64 = rp.rand([100], dtype='float64')
check("rand float64", t_f64.dtype == 'float64')

t_f16 = rp.rand([100], dtype='float16')
check("rand float16", t_f16.dtype == 'float16')
vals_f16 = get_vals(t_f16)
check("rand float16 range", all(0.0 <= v < 1.0 for v in vals_f16))

t_bf16 = rp.rand([100], dtype='bfloat16')
check("rand bfloat16", t_bf16.dtype == 'bfloat16')

print("\n--- rp.randn() Box-Muller ---")

rp.manual_seed(456)
t = rp.randn([N])
vals = get_vals(t)
mean = sum(vals) / len(vals)
std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
check(f"randn mean ~0.0 (got {mean:.4f})", abs(mean) < 0.05)
check(f"randn std ~1.0 (got {std:.4f})", abs(std - 1.0) < 0.05)
check("randn shape", t.shape == (N,))

t_odd = rp.randn([7])
check("randn odd size", t_odd.shape == (7,))

t_one = rp.randn([1])
check("randn single element", t_one.shape == (1,))

t_f64 = rp.randn([100], dtype='float64')
check("randn float64", t_f64.dtype == 'float64')

t_f16 = rp.randn([100], dtype='float16')
check("randn float16", t_f16.dtype == 'float16')

print("\n--- rp.randn() Ziggurat ---")

rp.manual_seed(789)
t = rp.randn([N], method='ziggurat')
vals = get_vals(t)
mean = sum(vals) / len(vals)
std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
check(f"ziggurat mean ~0.0 (got {mean:.4f})", abs(mean) < 0.05)
check(f"ziggurat std ~1.0 (got {std:.4f})", abs(std - 1.0) < 0.05)

print("\n--- rp.randn() Inverse CDF ---")

rp.manual_seed(101)
t = rp.randn([N], method='inverse_cdf')
vals = get_vals(t)
mean = sum(vals) / len(vals)
std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
check(f"inverse_cdf mean ~0.0 (got {mean:.4f})", abs(mean) < 0.05)
check(f"inverse_cdf std ~1.0 (got {std:.4f})", abs(std - 1.0) < 0.05)

print("\n--- rp.randint() ---")

rp.manual_seed(202)
t = rp.randint(0, 10, [1000])
vals = get_vals(t)
check("randint range [0,10)", all(0 <= v < 10 for v in vals))
check("randint integer-valued", all(v == math.floor(v) for v in vals))
unique = set(int(v) for v in vals)
check(f"randint covers range (got {len(unique)} unique)", len(unique) >= 8)

t_neg = rp.randint(-5, 5, [500])
vals_neg = get_vals(t_neg)
check("randint negative range", all(-5 <= v < 5 for v in vals_neg))

print("\n--- rp.manual_seed() reproducibility ---")

rp.manual_seed(42)
a = rp.rand([100])
rp.manual_seed(42)
b = rp.rand([100])
diff = sv(a.op_sub(b).op_abs())
check(f"rand reproducibility (diff={diff})", diff == 0.0)

rp.manual_seed(42)
a = rp.randn([100])
rp.manual_seed(42)
b = rp.randn([100])
diff = sv(a.op_sub(b).op_abs())
check(f"randn reproducibility (diff={diff})", diff == 0.0)

rp.manual_seed(42)
a = rp.randn([100], method='ziggurat')
rp.manual_seed(42)
b = rp.randn([100], method='ziggurat')
diff = sv(a.op_sub(b).op_abs())
check(f"ziggurat reproducibility (diff={diff})", diff == 0.0)

rp.manual_seed(42)
a = rp.randn([100], method='inverse_cdf')
rp.manual_seed(42)
b = rp.randn([100], method='inverse_cdf')
diff = sv(a.op_sub(b).op_abs())
check(f"inverse_cdf reproducibility (diff={diff})", diff == 0.0)

rp.manual_seed(42)
a = rp.rand([100])
rp.manual_seed(99)
b = rp.rand([100])
diff = sv(a.op_sub(b).op_abs())
check(f"different seeds differ (diff={diff:.4f})", diff > 0.1)

print("\n--- Edge cases ---")

t = rp.rand([1])
check("rand single element", t.shape == (1,) and t.size == 1)

t = rp.rand([1000000])
check("rand 1M elements", t.shape == (1000000,))

t = rp.randn([3])
check("randn size 3 (odd, Box-Muller pair)", t.shape == (3,))

try:
    rp.randn([3], method='invalid')
    check("invalid method raises error", False)
except ValueError:
    check("invalid method raises error", True)

if HAS_CUDA:
    print("\n--- CUDA tests ---")

    t = rp.rand([N], device='cuda')
    check("CUDA rand shape", t.shape == (N,))
    check("CUDA rand device", 'cuda' in t.device)
    cpu_t = t.to(device='cpu')
    vals = get_vals(cpu_t)
    check("CUDA rand range", all(0.0 <= v < 1.0 for v in vals))
    mean = sum(vals) / len(vals)
    check(f"CUDA rand mean ~0.5 (got {mean:.4f})", abs(mean - 0.5) < 0.03)

    t = rp.randn([N], device='cuda')
    vals = get_vals(t.to(device='cpu'))
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
    check(f"CUDA randn mean ~0.0 (got {mean:.4f})", abs(mean) < 0.05)
    check(f"CUDA randn std ~1.0 (got {std:.4f})", abs(std - 1.0) < 0.1)

    t = rp.randn([N], device='cuda', method='ziggurat')
    vals = get_vals(t.to(device='cpu'))
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
    check(f"CUDA ziggurat mean ~0.0 (got {mean:.4f})", abs(mean) < 0.05)
    check(f"CUDA ziggurat std ~1.0 (got {std:.4f})", abs(std - 1.0) < 0.1)

    t = rp.randn([N], device='cuda', method='inverse_cdf')
    vals = get_vals(t.to(device='cpu'))
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
    check(f"CUDA inverse_cdf mean ~0.0 (got {mean:.4f})", abs(mean) < 0.05)
    check(f"CUDA inverse_cdf std ~1.0 (got {std:.4f})", abs(std - 1.0) < 0.1)

    t = rp.randint(0, 10, [1000], device='cuda')
    vals = get_vals(t.to(device='cpu'))
    check("CUDA randint range", all(0 <= v < 10 for v in vals))
    check("CUDA randint integer-valued", all(v == math.floor(v) for v in vals))

    t_f16 = rp.rand([100], dtype='float16', device='cuda')
    check("CUDA rand float16", t_f16.dtype == 'float16')

    t_bf16 = rp.rand([100], dtype='bfloat16', device='cuda')
    check("CUDA rand bfloat16", t_bf16.dtype == 'bfloat16')

    t_f16 = rp.randn([100], dtype='float16', device='cuda')
    check("CUDA randn float16", t_f16.dtype == 'float16')

    rp.manual_seed(42)
    a = rp.rand([100], device='cuda')
    rp.manual_seed(42)
    b = rp.rand([100], device='cuda')
    diff = sv(a.op_sub(b).op_abs().to(device='cpu'))
    check(f"CUDA rand reproducibility (diff={diff})", diff == 0.0)

    rp.manual_seed(42)
    a = rp.randn([100], device='cuda')
    rp.manual_seed(42)
    b = rp.randn([100], device='cuda')
    diff = sv(a.op_sub(b).op_abs().to(device='cpu'))
    check(f"CUDA randn reproducibility (diff={diff})", diff == 0.0)

else:
    print("\n  SKIP: No CUDA device")

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All random tests passed!")
