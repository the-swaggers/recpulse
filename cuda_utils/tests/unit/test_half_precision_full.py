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

def sv(t):
    return t.to(dtype='float32').sum_all()

def vals_f32(t):
    t32 = t.to(dtype='float32')
    return [rp.values([0.0]).op_add(t32.slice([i], [i+1], [1])).sum_all() for i in range(t.size)]

def approx(a, b, tol=0.05):
    if abs(b) < 1e-6:
        return abs(a - b) < tol
    return abs(a - b) / max(abs(b), 1e-6) < tol

_refs = []
def make_nd(data, shape, dtype):
    t = rp.values(data, dtype=dtype)
    _refs.append(t)
    return t.reshape(shape)

HAS_CUDA = False
try:
    _t = rp.zeros([1], device='cuda')
    HAS_CUDA = True
except:
    pass

for dtype_name in ['float16', 'bfloat16']:
    print("=" * 70)
    print(f"COMPREHENSIVE {dtype_name.upper()} TESTING")
    print("=" * 70)

    tol = 0.02 if dtype_name == 'float16' else 0.04

    print(f"\n{'='*50}")
    print(f"1. TENSOR CREATION & CONVERSION ({dtype_name})")
    print(f"{'='*50}")

    t_z = rp.zeros([3, 4], dtype=dtype_name)
    check(f"zeros [3,4]", t_z is not None and t_z.shape == (3, 4) and t_z.dtype == dtype_name)
    check(f"zeros value", abs(sv(t_z)) < 0.01)

    t_o = rp.ones([2, 3], dtype=dtype_name)
    check(f"ones [2,3]", t_o is not None and t_o.shape == (2, 3) and t_o.dtype == dtype_name)
    check(f"ones value", abs(sv(t_o) - 6.0) < 0.1)

    t_v = rp.values([1.5, -2.0, 3.25, 0.0], dtype=dtype_name)
    check(f"values creation", t_v is not None and t_v.size == 4 and t_v.dtype == dtype_name)

    t_f32 = t_v.to(dtype='float32')
    check(f"-> float32", t_f32.dtype == 'float32')
    check(f"-> float32 values preserved", abs(sv(t_f32) - 2.75) < 0.1)

    t_f64 = t_v.to(dtype='float64')
    check(f"-> float64", t_f64.dtype == 'float64')

    t_back = t_f32.to(dtype=dtype_name)
    check(f"float32 -> back", t_back.dtype == dtype_name)

    t_back64 = t_f64.to(dtype=dtype_name)
    check(f"float64 -> back", t_back64.dtype == dtype_name)

    if dtype_name == 'float16':
        t_to_bf = t_v.to(dtype='bfloat16')
        check(f"float16 -> bfloat16", t_to_bf.dtype == 'bfloat16')
        check(f"float16 -> bfloat16 value", abs(sv(t_to_bf) - 2.75) < 0.2)
    else:
        t_to_fp = t_v.to(dtype='float16')
        check(f"bfloat16 -> float16", t_to_fp.dtype == 'float16')
        check(f"bfloat16 -> float16 value", abs(sv(t_to_fp) - 2.75) < 0.2)

    t_copy = t_v.copy()
    check(f"copy", t_copy is not None and t_copy.dtype == dtype_name)
    check(f"copy value", abs(sv(t_copy) - sv(t_v)) < 0.01)

    print(f"\n{'='*50}")
    print(f"2. BINARY OPS - FORWARD VALUES ({dtype_name})")
    print(f"{'='*50}")

    a = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    b = rp.values([0.5, 1.0, 1.5, 2.0], dtype=dtype_name)

    r = a.op_add(b)
    check(f"add dtype", r.dtype == dtype_name)
    check(f"add value", approx(sv(r), 15.0, tol))

    r = a.op_sub(b)
    check(f"sub dtype", r.dtype == dtype_name)
    check(f"sub value", approx(sv(r), 5.0, tol))

    r = a.op_mul(b)
    check(f"mul dtype", r.dtype == dtype_name)
    check(f"mul value", approx(sv(r), 15.0, tol))

    r = a.op_div(b)
    check(f"div dtype", r.dtype == dtype_name)
    check(f"div value", approx(sv(r), 8.0, tol))

    base = rp.values([2.0, 3.0], dtype=dtype_name)
    exp_v = rp.values([3.0, 2.0], dtype=dtype_name)
    r = base.op_power(exp_v)
    check(f"power dtype", r.dtype == dtype_name)
    check(f"power value (2^3 + 3^2 = 17)", approx(sv(r), 17.0, tol))

    base2 = rp.values([8.0, 9.0], dtype=dtype_name)
    bases = rp.values([2.0, 3.0], dtype=dtype_name)
    r = base2.op_logb(bases)
    check(f"logb dtype", r.dtype == dtype_name)
    check(f"logb value (log2(8) + log3(9) = 5)", approx(sv(r), 5.0, tol))

    print(f"\n{'='*50}")
    print(f"3. SCALAR OPS - FORWARD VALUES ({dtype_name})")
    print(f"{'='*50}")

    x = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)

    r = x.op_add_scalar(10.0)
    check(f"add_scalar dtype", r.dtype == dtype_name)
    v = sv(r); check(f"add_scalar value (got {v}, exp 50)", approx(v, 50.0, tol))

    r = x.op_sub_scalar(1.0)
    v = sv(r); check(f"sub_scalar value (got {v}, exp 6)", approx(v, 6.0, tol))

    r = x.op_mul_scalar(2.0)
    v = sv(r); check(f"mul_scalar value (got {v}, exp 20)", approx(v, 20.0, tol))

    r = x.op_div_scalar(2.0)
    v = sv(r); check(f"div_scalar value (got {v}, exp 5)", approx(v, 5.0, tol))

    r = rp.values([2.0, 3.0], dtype=dtype_name).op_pow_scalar(2.0)
    v = sv(r); check(f"pow_scalar value (got {v}, exp 13)", approx(v, 13.0, tol))

    r = x.op_rsub_scalar(10.0)
    v = sv(r); check(f"rsub_scalar value (got {v}, exp 30)", approx(v, 30.0, tol))

    r = rp.values([2.0, 4.0], dtype=dtype_name).op_rdiv_scalar(8.0)
    v = sv(r); check(f"rdiv_scalar value (got {v}, exp 6)", approx(v, 6.0, tol))

    r = rp.values([2.0, 3.0], dtype=dtype_name).op_rpow_scalar(2.0)
    v = sv(r); check(f"rpow_scalar value (got {v}, exp 12)", approx(v, 12.0, tol))

    r = rp.values([8.0, 27.0], dtype=dtype_name).op_logb_scalar(2.0)
    check(f"logb_scalar dtype", r.dtype == dtype_name)

    r = rp.values([2.0, 4.0], dtype=dtype_name).op_rlogb_scalar(16.0)
    check(f"rlogb_scalar dtype", r.dtype == dtype_name)

    print(f"\n{'='*50}")
    print(f"4. ALL UNARY OPS - FORWARD ({dtype_name})")
    print(f"{'='*50}")

    pos = rp.values([0.5, 1.0, 1.5, 2.0], dtype=dtype_name)
    mixed = rp.values([-1.0, 0.5, -0.5, 2.0], dtype=dtype_name)
    small = rp.values([0.1, 0.3, 0.5, 0.7], dtype=dtype_name)

    r = pos.op_exp()
    check(f"exp dtype", r.dtype == dtype_name)
    expected_exp = math.exp(0.5) + math.exp(1.0) + math.exp(1.5) + math.exp(2.0)
    check(f"exp value", approx(sv(r), expected_exp, tol))

    r = pos.op_log()
    check(f"log dtype", r.dtype == dtype_name)

    r = pos.op_sqrt()
    check(f"sqrt dtype", r.dtype == dtype_name)

    r = mixed.op_abs()
    check(f"abs dtype", r.dtype == dtype_name)
    check(f"abs value", approx(sv(r), 4.0, tol))

    r = pos.op_sin()
    check(f"sin dtype", r.dtype == dtype_name)
    expected_sin = math.sin(0.5) + math.sin(1.0) + math.sin(1.5) + math.sin(2.0)
    check(f"sin value", approx(sv(r), expected_sin, tol))

    r = pos.op_cos()
    check(f"cos dtype", r.dtype == dtype_name)

    r = small.op_tan()
    check(f"tan dtype", r.dtype == dtype_name)

    r = small.op_asin()
    check(f"asin dtype", r.dtype == dtype_name)

    r = small.op_acos()
    check(f"acos dtype", r.dtype == dtype_name)

    r = small.op_atan()
    check(f"atan dtype", r.dtype == dtype_name)

    r = pos.op_sinh()
    check(f"sinh dtype", r.dtype == dtype_name)

    r = pos.op_cosh()
    check(f"cosh dtype", r.dtype == dtype_name)

    r = pos.op_tanh()
    check(f"tanh dtype", r.dtype == dtype_name)

    r = mixed.op_relu()
    check(f"relu dtype", r.dtype == dtype_name)
    check(f"relu value", approx(sv(r), 2.5, tol))

    r = pos.op_sigmoid()
    check(f"sigmoid dtype", r.dtype == dtype_name)

    r = mixed.op_leaky_relu()
    check(f"leaky_relu dtype", r.dtype == dtype_name)

    r = pos.op_square()
    check(f"square dtype", r.dtype == dtype_name)
    check(f"square value (0.25+1+2.25+4=7.5)", approx(sv(r), 7.5, tol))

    r = pos.op_rsqrt()
    check(f"rsqrt dtype", r.dtype == dtype_name)

    r = pos.op_gelu()
    check(f"gelu dtype", r.dtype == dtype_name)

    r = pos.op_silu()
    check(f"silu dtype", r.dtype == dtype_name)

    print(f"\n{'='*50}")
    print(f"5. REDUCTION OPS ({dtype_name})")
    print(f"{'='*50}")

    x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype_name)

    r = x.op_sum_all()
    check(f"sum_all", approx(sv(r), 21.0, tol))

    r = x.op_mean_all()
    check(f"mean_all", approx(sv(r), 3.5, tol))

    x2d_base = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype_name)
    x2d = x2d_base.reshape([2, 3])
    r = x2d.op_sum_dim(1)
    check(f"sum_dim(1) dtype", r.dtype == dtype_name)
    check(f"sum_dim(1) shape", r.shape == (2,))
    check(f"sum_dim(1) value", approx(sv(r), 21.0, tol))

    r = x2d.op_mean_dim(0)
    check(f"mean_dim(0) dtype", r.dtype == dtype_name)
    check(f"mean_dim(0) shape", r.shape == (3,))

    r = x2d.op_sum_dim(1, keepdim=True)
    check(f"sum_dim keepdim shape", r.shape == (2, 1))

    print(f"\n{'='*50}")
    print(f"6. MATMUL ({dtype_name})")
    print(f"{'='*50}")

    a_mat = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], dtype_name)
    b_mat = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], dtype_name)

    r = a_mat.op_matmul(b_mat)
    check(f"matmul dtype", r.dtype == dtype_name)
    check(f"matmul shape", r.shape == (2, 2))
    check(f"matmul value", approx(sv(r), 22+28+49+64, tol))

    print(f"\n{'='*50}")
    print(f"7. SHAPE OPS ({dtype_name})")
    print(f"{'='*50}")

    x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype_name)

    r = x.reshape([2, 3])
    check(f"reshape", r.shape == (2, 3) and r.dtype == dtype_name)

    r = x.reshape([3, 2]).view([6])
    check(f"view", r.shape == (6,) and r.dtype == dtype_name)

    r = x.reshape([2, 3]).transpose(0, 1)
    check(f"transpose", r.shape == (3, 2) and r.dtype == dtype_name)

    r = x.reshape([1, 6]).squeeze(0)
    check(f"squeeze", r.shape == (6,) and r.dtype == dtype_name)

    r = x.unsqueeze(0)
    check(f"unsqueeze", r.shape == (1, 6) and r.dtype == dtype_name)

    r = x.reshape([2, 3]).flatten()
    check(f"flatten", r.shape == (6,) and r.dtype == dtype_name)

    r = x.reshape([2, 3]).permute([1, 0])
    check(f"permute", r.shape == (3, 2) and r.dtype == dtype_name)

    a_cat = rp.values([1.0, 2.0], dtype=dtype_name)
    b_cat = rp.values([3.0, 4.0], dtype=dtype_name)
    r = rp.cat([a_cat, b_cat], dim=0)
    check(f"cat", r.shape == (4,) and r.dtype == dtype_name)
    check(f"cat value", approx(sv(r), 10.0, tol))

    r = x.slice([1], [4], [1])
    check(f"slice", r.shape == (3,) and r.dtype == dtype_name)
    check(f"slice value", approx(sv(r), 9.0, tol))

    x_2d = x.reshape([2, 3])
    r = x_2d.expand([3, 2, 3])
    check(f"expand", r.shape == (3, 2, 3) and r.dtype == dtype_name)

    rep_base = rp.values([1.0, 2.0, 3.0], dtype=dtype_name)
    r = rep_base.repeat([2])
    check(f"repeat", r.shape == (6,) and r.dtype == dtype_name)

    chunks = x.chunk(3, 0)
    check(f"chunk count", len(chunks) == 3)
    check(f"chunk dtype", all(c.dtype == dtype_name for c in chunks))

    splits = x.split_equal(2, 0)
    check(f"split_equal count", len(splits) == 2)
    check(f"split_equal dtype", all(s.dtype == dtype_name for s in splits))

    print(f"\n{'='*50}")
    print(f"8. ALL BACKWARD OPS ({dtype_name})")
    print(f"{'='*50}")

    def test_backward(name, input_vals, op_fn, extra_tol=None):
        t = extra_tol if extra_tol else tol
        x = rp.values(input_vals, dtype=dtype_name)
        x.requires_grad_(True)
        try:
            y = op_fn(x)
            s = y.op_sum_all()
            s.backward()
            ok = x.has_grad and x.grad.dtype == dtype_name and x.grad.shape == x.shape
            check(f"{name} backward", ok)
        except Exception as e:
            check(f"{name} backward (exception: {e})", False)

    test_backward("exp", [0.5, 1.0, 1.5, 2.0], lambda x: x.op_exp())
    test_backward("log", [0.5, 1.0, 1.5, 2.0], lambda x: x.op_log())
    test_backward("sqrt", [0.5, 1.0, 1.5, 2.0], lambda x: x.op_sqrt())
    test_backward("abs", [1.0, -2.0, 3.0, -4.0], lambda x: x.op_abs())
    test_backward("sin", [0.5, 1.0, 1.5, 2.0], lambda x: x.op_sin())
    test_backward("cos", [0.5, 1.0, 1.5, 2.0], lambda x: x.op_cos())
    test_backward("tan", [0.1, 0.3, 0.5, 0.7], lambda x: x.op_tan())
    test_backward("asin", [0.1, 0.3, 0.5, 0.7], lambda x: x.op_asin())
    test_backward("acos", [0.1, 0.3, 0.5, 0.7], lambda x: x.op_acos())
    test_backward("atan", [0.5, 1.0, 1.5, 2.0], lambda x: x.op_atan())
    test_backward("sinh", [0.5, 1.0, 1.5, 2.0], lambda x: x.op_sinh())
    test_backward("cosh", [0.5, 1.0, 1.5, 2.0], lambda x: x.op_cosh())
    test_backward("tanh", [0.5, 1.0, 1.5, 2.0], lambda x: x.op_tanh())
    test_backward("relu", [-1.0, 0.5, -0.5, 2.0], lambda x: x.op_relu())
    test_backward("sigmoid", [0.5, 1.0, 1.5, 2.0], lambda x: x.op_sigmoid())
    test_backward("leaky_relu", [-1.0, 0.5, -0.5, 2.0], lambda x: x.op_leaky_relu())
    test_backward("square", [1.0, 2.0, 3.0, 4.0], lambda x: x.op_square())
    test_backward("rsqrt", [0.5, 1.0, 2.0, 4.0], lambda x: x.op_rsqrt())
    test_backward("gelu", [0.5, 1.0, 1.5, 2.0], lambda x: x.op_gelu())
    test_backward("silu", [0.5, 1.0, 1.5, 2.0], lambda x: x.op_silu())

    print(f"\n--- Binary backward ({dtype_name}) ---")

    def test_binary_backward(name, vals_a, vals_b, op_fn):
        a = rp.values(vals_a, dtype=dtype_name)
        a.requires_grad_(True)
        b = rp.values(vals_b, dtype=dtype_name)
        b.requires_grad_(True)
        try:
            y = op_fn(a, b)
            s = y.op_sum_all()
            s.backward()
            ok_a = a.has_grad and a.grad.dtype == dtype_name
            ok_b = b.has_grad and b.grad.dtype == dtype_name
            check(f"{name} backward (both grads)", ok_a and ok_b)
        except Exception as e:
            check(f"{name} backward (exception: {e})", False)

    test_binary_backward("add", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], lambda a, b: a.op_add(b))
    test_binary_backward("sub", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], lambda a, b: a.op_sub(b))
    test_binary_backward("mul", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], lambda a, b: a.op_mul(b))
    test_binary_backward("div", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], lambda a, b: a.op_div(b))
    test_binary_backward("power", [2.0, 3.0, 4.0], [2.0, 2.0, 2.0], lambda a, b: a.op_power(b))
    test_binary_backward("logb", [8.0, 27.0, 16.0], [2.0, 3.0, 4.0], lambda a, b: a.op_logb(b))

    print(f"\n--- Scalar backward ({dtype_name}) ---")

    def test_scalar_backward(name, input_vals, op_fn):
        x = rp.values(input_vals, dtype=dtype_name)
        x.requires_grad_(True)
        try:
            y = op_fn(x)
            s = y.op_sum_all()
            s.backward()
            ok = x.has_grad and x.grad.dtype == dtype_name
            check(f"{name} backward", ok)
        except Exception as e:
            check(f"{name} backward (exception: {e})", False)

    test_scalar_backward("add_scalar", [1.0, 2.0, 3.0], lambda x: x.op_add_scalar(5.0))
    test_scalar_backward("sub_scalar", [1.0, 2.0, 3.0], lambda x: x.op_sub_scalar(1.0))
    test_scalar_backward("mul_scalar", [1.0, 2.0, 3.0], lambda x: x.op_mul_scalar(3.0))
    test_scalar_backward("div_scalar", [1.0, 2.0, 3.0], lambda x: x.op_div_scalar(2.0))
    test_scalar_backward("pow_scalar", [2.0, 3.0, 4.0], lambda x: x.op_pow_scalar(2.0))
    test_scalar_backward("rsub_scalar", [1.0, 2.0, 3.0], lambda x: x.op_rsub_scalar(10.0))
    test_scalar_backward("rdiv_scalar", [1.0, 2.0, 4.0], lambda x: x.op_rdiv_scalar(8.0))
    test_scalar_backward("rpow_scalar", [1.0, 2.0, 3.0], lambda x: x.op_rpow_scalar(2.0))
    test_scalar_backward("logb_scalar", [8.0, 16.0, 32.0], lambda x: x.op_logb_scalar(2.0))
    test_scalar_backward("rlogb_scalar", [2.0, 4.0, 8.0], lambda x: x.op_rlogb_scalar(16.0))

    print(f"\n--- Reduction backward ({dtype_name}) ---")

    x = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x.requires_grad_(True)
    s = x.op_sum_all()
    s.backward()
    check(f"sum_all backward", x.has_grad and x.grad.dtype == dtype_name)

    x = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x.requires_grad_(True)
    m = x.op_mean_all()
    m.backward()
    check(f"mean_all backward", x.has_grad and x.grad.dtype == dtype_name)

    x = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], dtype_name)
    x.requires_grad_(True)
    s = x.op_sum_dim(1)
    s2 = s.op_sum_all()
    s2.backward()
    check(f"sum_dim backward", x.has_grad and x.grad.dtype == dtype_name)

    x = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], dtype_name)
    x.requires_grad_(True)
    m = x.op_mean_dim(0)
    m2 = m.op_sum_all()
    m2.backward()
    check(f"mean_dim backward", x.has_grad and x.grad.dtype == dtype_name)

    print(f"\n--- Matmul backward ({dtype_name}) ---")

    x = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], dtype_name)
    x.requires_grad_(True)
    w = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], dtype_name)
    w.requires_grad_(True)
    y = x.op_matmul(w)
    s = y.op_sum_all()
    s.backward()
    check(f"matmul backward x", x.has_grad and x.grad.dtype == dtype_name and x.grad.shape == (2, 3))
    check(f"matmul backward w", w.has_grad and w.grad.dtype == dtype_name and w.grad.shape == (3, 2))

    print(f"\n--- Shape ops backward ({dtype_name}) ---")

    x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype_name)
    x.requires_grad_(True)
    y = x.reshape([2, 3])
    s = y.op_sum_all()
    s.backward()
    check(f"reshape backward", x.has_grad and x.grad.shape == (6,))

    x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype_name)
    x.requires_grad_(True)
    x_2d = x.reshape([2, 3])
    y = x_2d.transpose(0, 1)
    s = y.op_sum_all()
    s.backward()
    check(f"transpose backward", x.has_grad and x.grad.shape == (6,))

    a = rp.values([1.0, 2.0, 3.0], dtype=dtype_name)
    a.requires_grad_(True)
    b = rp.values([4.0, 5.0, 6.0], dtype=dtype_name)
    b.requires_grad_(True)
    c = rp.cat([a, b], dim=0)
    s = c.op_sum_all()
    s.backward()
    check(f"cat backward", a.has_grad and b.has_grad)

    x = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x.requires_grad_(True)
    y = x.slice([1], [3], [1])
    s = y.op_sum_all()
    s.backward()
    check(f"slice backward", x.has_grad and x.grad.shape == (4,))

    x = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 6], dtype_name)
    x.requires_grad_(True)
    y = x.squeeze(0)
    s = y.op_sum_all()
    s.backward()
    check(f"squeeze backward", x.has_grad)

    x = rp.values([1.0, 2.0, 3.0], dtype=dtype_name)
    x.requires_grad_(True)
    y = x.unsqueeze(0)
    s = y.op_sum_all()
    s.backward()
    check(f"unsqueeze backward", x.has_grad)

    x = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], dtype_name)
    x.requires_grad_(True)
    y = x.flatten()
    s = y.op_sum_all()
    s.backward()
    check(f"flatten backward", x.has_grad)

    x = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], dtype_name)
    x.requires_grad_(True)
    y = x.permute([1, 0])
    s = y.op_sum_all()
    s.backward()
    check(f"permute backward", x.has_grad)

    x = make_nd([1.0, 2.0, 3.0], [1, 3], dtype_name)
    x.requires_grad_(True)
    y = x.expand([4, 3])
    s = y.op_sum_all()
    s.backward()
    check(f"expand backward", x.has_grad)

    x = rp.values([1.0, 2.0, 3.0], dtype=dtype_name)
    x.requires_grad_(True)
    y = x.repeat([2])
    s = y.op_sum_all()
    s.backward()
    check(f"repeat backward", x.has_grad)

    x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype_name)
    x.requires_grad_(True)
    chunks = x.chunk(3, 0)
    s = chunks[0].op_sum_all()
    s.backward()
    check(f"chunk backward", x.has_grad)

    print(f"\n{'='*50}")
    print(f"9. GRADIENT VALUE CORRECTNESS ({dtype_name})")
    print(f"{'='*50}")

    x = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x.requires_grad_(True)
    y = x.op_square()
    s = y.op_sum_all()
    s.backward()
    grad_vals = vals_f32(x.grad)
    check(f"square grad values [2,4,6,8]",
          approx(grad_vals[0], 2.0, tol) and approx(grad_vals[1], 4.0, tol) and
          approx(grad_vals[2], 6.0, tol) and approx(grad_vals[3], 8.0, tol))

    x = rp.values([-1.0, 0.5, -0.5, 2.0], dtype=dtype_name)
    x.requires_grad_(True)
    y = x.op_relu()
    s = y.op_sum_all()
    s.backward()
    grad_vals = vals_f32(x.grad)
    check(f"relu grad values [0,1,0,1]",
          approx(grad_vals[0], 0.0, tol) and approx(grad_vals[1], 1.0, tol) and
          approx(grad_vals[2], 0.0, tol) and approx(grad_vals[3], 1.0, tol))

    x = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x.requires_grad_(True)
    y = x.op_mul_scalar(3.0)
    s = y.op_sum_all()
    s.backward()
    grad_vals = vals_f32(x.grad)
    check(f"mul_scalar(3) grad values [3,3,3,3]",
          all(approx(g, 3.0, tol) for g in grad_vals))

    x = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x.requires_grad_(True)
    s = x.op_sum_all()
    s.backward()
    grad_vals = vals_f32(x.grad)
    check(f"sum_all grad values [1,1,1,1]",
          all(approx(g, 1.0, tol) for g in grad_vals))

    x = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x.requires_grad_(True)
    m = x.op_mean_all()
    m.backward()
    grad_vals = vals_f32(x.grad)
    check(f"mean_all grad values [0.25,0.25,0.25,0.25]",
          all(approx(g, 0.25, tol) for g in grad_vals))

    print(f"\n{'='*50}")
    print(f"10. CHAINED OPS ({dtype_name})")
    print(f"{'='*50}")

    x = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
    x.requires_grad_(True)
    y = x.op_square()
    z = y.op_sum_all()
    w = z
    w.backward()
    check(f"square -> sum_all chain", x.has_grad and x.grad.dtype == dtype_name)

    x = rp.values([0.5, 1.0, 1.5, 2.0], dtype=dtype_name)
    x.requires_grad_(True)
    y = x.op_mul_scalar(2.0)
    z = y.op_relu()
    s = z.op_sum_all()
    s.backward()
    check(f"mul_scalar -> relu -> sum chain", x.has_grad)

    x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype_name)
    x.requires_grad_(True)
    y = x.reshape([2, 3])
    z = y.op_sum_dim(1)
    s = z.op_sum_all()
    s.backward()
    check(f"reshape -> sum_dim -> sum chain", x.has_grad)

    x = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], dtype_name)
    x.requires_grad_(True)
    w = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], dtype_name)
    w.requires_grad_(True)
    y = x.op_matmul(w)
    z = y.op_relu()
    s = z.op_sum_all()
    s.backward()
    check(f"matmul -> relu -> sum chain", x.has_grad and w.has_grad)

    print(f"\n{'='*50}")
    print(f"11. BROADCASTING ({dtype_name})")
    print(f"{'='*50}")

    a = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], dtype_name)
    b = rp.values([10.0, 20.0, 30.0], dtype=dtype_name)
    a.requires_grad_(True)
    b.requires_grad_(True)
    r = a.op_add(b)
    check(f"broadcast add shape", r.shape == (2, 3))
    s = r.op_sum_all()
    s.backward()
    check(f"broadcast add backward a", a.has_grad and a.grad.shape == (2, 3))
    check(f"broadcast add backward b", b.has_grad and b.grad.shape == (3,))

    a = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], dtype_name)
    b = rp.values([2.0, 3.0, 4.0], dtype=dtype_name)
    a.requires_grad_(True)
    b.requires_grad_(True)
    r = a.op_mul(b)
    check(f"broadcast mul shape", r.shape == (2, 3))
    s = r.op_sum_all()
    s.backward()
    check(f"broadcast mul backward a", a.has_grad and a.grad.shape == (2, 3))
    check(f"broadcast mul backward b", b.has_grad and b.grad.shape == (3,))

    if HAS_CUDA:
        print(f"\n{'='*50}")
        print(f"12. CUDA ({dtype_name})")
        print(f"{'='*50}")

        t_gpu = rp.zeros([4], dtype=dtype_name, device='cuda')
        check(f"CUDA zeros creation", t_gpu is not None and t_gpu.dtype == dtype_name)
        check(f"CUDA device", 'cuda' in t_gpu.device)

        t_gpu = rp.ones([4], dtype=dtype_name, device='cuda')
        check(f"CUDA ones creation", t_gpu is not None)

        t_cpu = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name)
        t_gpu = t_cpu.to(device='cuda')
        check(f"CPU -> CUDA transfer", 'cuda' in t_gpu.device and t_gpu.dtype == dtype_name)

        t_back = t_gpu.to(device='cpu')
        check(f"CUDA -> CPU transfer", t_back.device == 'cpu' and t_back.dtype == dtype_name)
        check(f"roundtrip value", approx(sv(t_back), 10.0, tol))

        t_gpu32 = t_gpu.to(dtype='float32')
        check(f"CUDA dtype convert", t_gpu32.dtype == 'float32' and 'cuda' in t_gpu32.device)

        a_gpu = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name, device='cuda')
        b_gpu = rp.values([0.5, 1.0, 1.5, 2.0], dtype=dtype_name, device='cuda')
        r = a_gpu.op_add(b_gpu)
        check(f"CUDA add", r.dtype == dtype_name and 'cuda' in r.device)
        check(f"CUDA add value", approx(sv(r.to(device='cpu')), 15.0, tol))

        r = a_gpu.op_mul(b_gpu)
        check(f"CUDA mul value", approx(sv(r.to(device='cpu')), 15.0, tol))

        r = a_gpu.op_sub(b_gpu)
        check(f"CUDA sub value", approx(sv(r.to(device='cpu')), 5.0, tol))

        r = a_gpu.op_div(b_gpu)
        check(f"CUDA div value", approx(sv(r.to(device='cpu')), 8.0, tol))

        r = a_gpu.op_exp()
        check(f"CUDA exp", r.dtype == dtype_name)

        r = a_gpu.op_relu()
        check(f"CUDA relu", r.dtype == dtype_name)

        r = a_gpu.op_sigmoid()
        check(f"CUDA sigmoid", r.dtype == dtype_name)

        r = a_gpu.op_square()
        check(f"CUDA square", r.dtype == dtype_name)

        r = a_gpu.op_sin()
        check(f"CUDA sin", r.dtype == dtype_name)

        r = a_gpu.op_cos()
        check(f"CUDA cos", r.dtype == dtype_name)

        r = a_gpu.op_tanh()
        check(f"CUDA tanh", r.dtype == dtype_name)

        r = a_gpu.op_gelu()
        check(f"CUDA gelu", r.dtype == dtype_name)

        r = a_gpu.op_silu()
        check(f"CUDA silu", r.dtype == dtype_name)

        r = a_gpu.op_rsqrt()
        check(f"CUDA rsqrt", r.dtype == dtype_name)

        r = a_gpu.op_abs()
        check(f"CUDA abs", r.dtype == dtype_name)

        r = a_gpu.op_sqrt()
        check(f"CUDA sqrt", r.dtype == dtype_name)

        r = a_gpu.op_log()
        check(f"CUDA log", r.dtype == dtype_name)

        r = a_gpu.op_leaky_relu()
        check(f"CUDA leaky_relu", r.dtype == dtype_name)

        r = a_gpu.op_add_scalar(5.0)
        check(f"CUDA add_scalar", r.dtype == dtype_name)

        r = a_gpu.op_mul_scalar(2.0)
        check(f"CUDA mul_scalar value", approx(sv(r.to(device='cpu')), 20.0, tol))

        r = a_gpu.op_sum_all()
        check(f"CUDA sum_all", approx(sv(r.to(device='cpu')), 10.0, tol))

        r = a_gpu.op_mean_all()
        check(f"CUDA mean_all", approx(sv(r.to(device='cpu')), 2.5, tol))

        a_mat = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], dtype_name).to(device='cuda')
        b_mat = make_nd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], dtype_name).to(device='cuda')
        r = a_mat.op_matmul(b_mat)
        check(f"CUDA matmul", r.dtype == dtype_name and r.shape == (2, 2))

        x_gpu = rp.values([1.0, 2.0, 3.0, 4.0], dtype=dtype_name, device='cuda')
        x_gpu.requires_grad_(True)
        y_gpu = x_gpu.op_square()
        s_gpu = y_gpu.op_sum_all()
        s_gpu.backward()
        check(f"CUDA autograd (square backward)", x_gpu.has_grad)
        grad_cpu = x_gpu.grad.to(device='cpu')
        gv = vals_f32(grad_cpu)
        check(f"CUDA grad values [2,4,6,8]",
              approx(gv[0], 2.0, tol) and approx(gv[1], 4.0, tol) and
              approx(gv[2], 6.0, tol) and approx(gv[3], 8.0, tol))

        x_gpu = rp.values([0.5, 1.0, 1.5, 2.0], dtype=dtype_name, device='cuda')
        x_gpu.requires_grad_(True)
        y_gpu = x_gpu.op_exp()
        s_gpu = y_gpu.op_sum_all()
        s_gpu.backward()
        check(f"CUDA exp backward", x_gpu.has_grad and x_gpu.grad.dtype == dtype_name)

        x_gpu = rp.values([-1.0, 0.5, -0.5, 2.0], dtype=dtype_name, device='cuda')
        x_gpu.requires_grad_(True)
        y_gpu = x_gpu.op_relu()
        s_gpu = y_gpu.op_sum_all()
        s_gpu.backward()
        check(f"CUDA relu backward", x_gpu.has_grad)

        x_gpu = rp.values([0.5, 1.0, 1.5, 2.0], dtype=dtype_name, device='cuda')
        x_gpu.requires_grad_(True)
        y_gpu = x_gpu.op_sigmoid()
        s_gpu = y_gpu.op_sum_all()
        s_gpu.backward()
        check(f"CUDA sigmoid backward", x_gpu.has_grad)

        a_gpu = rp.values([1.0, 2.0, 3.0], dtype=dtype_name, device='cuda')
        a_gpu.requires_grad_(True)
        b_gpu = rp.values([4.0, 5.0, 6.0], dtype=dtype_name, device='cuda')
        b_gpu.requires_grad_(True)
        y_gpu = a_gpu.op_mul(b_gpu)
        s_gpu = y_gpu.op_sum_all()
        s_gpu.backward()
        check(f"CUDA mul backward", a_gpu.has_grad and b_gpu.has_grad)

        x_base = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype_name, device='cuda')
        x_gpu = x_base.reshape([2, 3])
        x_gpu.requires_grad_(True)
        w_base = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype_name, device='cuda')
        w_gpu = w_base.reshape([3, 2])
        w_gpu.requires_grad_(True)
        y_gpu = x_gpu.op_matmul(w_gpu)
        s_gpu = y_gpu.op_sum_all()
        s_gpu.backward()
        check(f"CUDA matmul backward", x_gpu.has_grad and w_gpu.has_grad)

        x_base2 = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype_name, device='cuda')
        x_gpu = x_base2.reshape([2, 3])
        x_gpu.requires_grad_(True)
        w_base2 = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype_name, device='cuda')
        w_gpu = w_base2.reshape([3, 2])
        w_gpu.requires_grad_(True)
        y_gpu = x_gpu.op_matmul(w_gpu)
        z_gpu = y_gpu.op_relu()
        s_gpu = z_gpu.op_sum_all()
        s_gpu.backward()
        check(f"CUDA matmul->relu->sum chain", x_gpu.has_grad and w_gpu.has_grad)
    else:
        print(f"\n  SKIP: No CUDA device available")

print("\n" + "=" * 70)
print(f"TOTAL: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All comprehensive half-precision tests passed!")
