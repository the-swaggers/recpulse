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

def scalar_val(t):
    return t.to(dtype='float32').sum_all()

print("=" * 60)
print("Testing fp16/bf16 support")
print("=" * 60)

for dtype_name in ['float16', 'bfloat16']:
    print(f"\n--- {dtype_name} ---")

    t = rp.zeros([4], dtype=dtype_name)
    check(f"{dtype_name} zeros creation", t is not None)
    check(f"{dtype_name} dtype", t.dtype == dtype_name)
    check(f"{dtype_name} shape", t.shape == (4,))
    check(f"{dtype_name} size", t.size == 4)

    t1 = rp.ones([4], dtype=dtype_name)
    check(f"{dtype_name} ones creation", t1 is not None)

    t2 = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    check(f"{dtype_name} values creation", t2 is not None)

    t_f32 = t2.to(dtype='float32')
    check(f"{dtype_name} -> float32 conversion", t_f32.dtype == 'float32')

    t_back = t_f32.to(dtype=dtype_name)
    check(f"float32 -> {dtype_name} conversion", t_back.dtype == dtype_name)

    r = t1.op_add(t1)
    check(f"{dtype_name} add", r is not None and r.dtype == dtype_name)

    r = t2.op_mul(t1)
    check(f"{dtype_name} mul", r is not None and r.dtype == dtype_name)

    r = t2.op_sub(t1)
    check(f"{dtype_name} sub", r is not None and r.dtype == dtype_name)

    r = t2.op_exp()
    check(f"{dtype_name} exp", r is not None and r.dtype == dtype_name)

    r = t2.op_relu()
    check(f"{dtype_name} relu", r is not None and r.dtype == dtype_name)

    r = t2.op_sigmoid()
    check(f"{dtype_name} sigmoid", r is not None and r.dtype == dtype_name)

    r = t2.op_square()
    check(f"{dtype_name} square", r is not None and r.dtype == dtype_name)

    r = t2.op_sin()
    check(f"{dtype_name} sin", r is not None and r.dtype == dtype_name)

    r = t2.op_cos()
    check(f"{dtype_name} cos", r is not None and r.dtype == dtype_name)

    r = t2.op_tanh()
    check(f"{dtype_name} tanh", r is not None and r.dtype == dtype_name)

    r = t2.op_abs()
    check(f"{dtype_name} abs", r is not None and r.dtype == dtype_name)

    r = t2.op_sqrt()
    check(f"{dtype_name} sqrt", r is not None and r.dtype == dtype_name)

    r = t2.op_log()
    check(f"{dtype_name} log", r is not None and r.dtype == dtype_name)

    r = t2.op_gelu()
    check(f"{dtype_name} gelu", r is not None and r.dtype == dtype_name)

    r = t2.op_silu()
    check(f"{dtype_name} silu", r is not None and r.dtype == dtype_name)

    r = t2.op_rsqrt()
    check(f"{dtype_name} rsqrt", r is not None and r.dtype == dtype_name)

    r = t2.op_add_scalar(2.0)
    check(f"{dtype_name} add_scalar", r is not None and r.dtype == dtype_name)

    r = t2.op_mul_scalar(3.0)
    check(f"{dtype_name} mul_scalar", r is not None and r.dtype == dtype_name)

    r = t2.op_sum_all()
    check(f"{dtype_name} sum_all", r is not None and r.dtype == dtype_name)

    r = t2.op_mean_all()
    check(f"{dtype_name} mean_all", r is not None and r.dtype == dtype_name)

    r = t2.reshape([2, 2])
    check(f"{dtype_name} reshape", r is not None and r.dtype == dtype_name and r.shape == (2, 2))

    print(f"\n--- {dtype_name} value verification ---")

    vals = rp.values([2.0, 3.0, 4.0, 5.0], dtype=dtype_name)

    sum_val = scalar_val(vals.op_sum_all())
    check(f"{dtype_name} sum_all value correct (got {sum_val})",
          abs(sum_val - 14.0) < 0.5)

    mean_val = scalar_val(vals.op_mean_all())
    check(f"{dtype_name} mean_all value correct (got {mean_val})",
          abs(mean_val - 3.5) < 0.5)

    add_val = scalar_val(vals.op_add(rp.ones([4], dtype=dtype_name)).op_sum_all())
    check(f"{dtype_name} add values correct (got {add_val})",
          abs(add_val - 18.0) < 0.5)

    sq_val = scalar_val(rp.values([2.0], dtype=dtype_name).op_square().op_sum_all())
    check(f"{dtype_name} square value correct (got {sq_val})",
          abs(sq_val - 4.0) < 0.5)

    print(f"\n--- {dtype_name} autograd ---")

    x = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x.requires_grad_(True)
    y = x.op_square()
    s = y.op_sum_all()
    s.backward()
    check(f"{dtype_name} square backward runs", x.has_grad)
    grad = x.grad
    check(f"{dtype_name} grad dtype", grad.dtype == dtype_name)
    check(f"{dtype_name} grad shape", grad.shape == (4,))

    x2 = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x2.requires_grad_(True)
    y2 = x2.op_relu()
    s2 = y2.op_sum_all()
    s2.backward()
    check(f"{dtype_name} relu backward runs", x2.has_grad)

    x3 = rp.values([0.5, 1.0, 1.5, 2.0], dtype=dtype_name)
    x3.requires_grad_(True)
    y3 = x3.op_exp()
    s3 = y3.op_sum_all()
    s3.backward()
    check(f"{dtype_name} exp backward runs", x3.has_grad)

    x4 = rp.values([0.5, 1.0, 1.5, 2.0], dtype=dtype_name)
    x4.requires_grad_(True)
    y4 = x4.op_sigmoid()
    s4 = y4.op_sum_all()
    s4.backward()
    check(f"{dtype_name} sigmoid backward runs", x4.has_grad)

    x5 = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x5.requires_grad_(True)
    y5 = x5.op_mul_scalar(2.0)
    s5 = y5.op_sum_all()
    s5.backward()
    check(f"{dtype_name} mul_scalar backward runs", x5.has_grad)

    x6 = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x6.requires_grad_(True)
    x6b = rp.values([0.5, 0.5, 0.5, 0.5], dtype=dtype_name)
    y6 = x6.op_add(x6b)
    s6 = y6.op_sum_all()
    s6.backward()
    check(f"{dtype_name} add backward runs", x6.has_grad)

    x7 = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x7.requires_grad_(True)
    y7 = x7.op_sin()
    s7 = y7.op_sum_all()
    s7.backward()
    check(f"{dtype_name} sin backward runs", x7.has_grad)

    x8 = rp.values([0.5, 1.0, 1.5, 2.0], dtype=dtype_name)
    x8.requires_grad_(True)
    y8 = x8.op_tanh()
    s8 = y8.op_sum_all()
    s8.backward()
    check(f"{dtype_name} tanh backward runs", x8.has_grad)

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All half-precision tests passed!")
