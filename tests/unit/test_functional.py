#!/usr/bin/env python3

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import recpulse_cuda as rp
import math

passed = 0
failed = 0

def run_test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS: {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL: {name} -- {e}")

def assert_close(actual, expected, tolerance=1e-5, label=""):
    diff = abs(actual - expected)
    assert diff <= tolerance, f"{label}: expected {expected}, got {actual} (diff: {diff})"

def assert_data_close(tensor, expected, tolerance=1e-5, label=""):
    data = tensor.data()
    assert len(data) == len(expected), f"{label}: length mismatch {len(data)} != {len(expected)}"
    for i, (a, e) in enumerate(zip(data, expected)):
        assert_close(a, e, tolerance, f"{label}[{i}]")


def test_add():
    a = rp.values([2.0, 3.0, 4.0])
    b = rp.values([1.0, 2.0, 3.0])
    c = a.add(b)
    assert_data_close(c, [3.0, 5.0, 7.0], label="add")

def test_sub():
    a = rp.values([2.0, 3.0, 4.0])
    b = rp.values([1.0, 2.0, 3.0])
    c = a.sub(b)
    assert_data_close(c, [1.0, 1.0, 1.0], label="sub")

def test_mul():
    a = rp.values([2.0, 3.0, 4.0])
    b = rp.values([1.0, 2.0, 3.0])
    c = a.mul(b)
    assert_data_close(c, [2.0, 6.0, 12.0], label="mul")

def test_divide():
    a = rp.values([2.0, 3.0, 4.0])
    b = rp.values([1.0, 2.0, 3.0])
    c = a.divide(b)
    assert_data_close(c, [2.0, 1.5, 4.0/3.0], label="divide")

def test_power():
    a = rp.values([2.0, 3.0, 4.0])
    b = rp.values([1.0, 2.0, 3.0])
    c = a.power(b)
    assert_data_close(c, [2.0, 9.0, 64.0], label="power")

def test_logb():
    a = rp.values([8.0, 27.0, 100.0])
    b = rp.values([2.0, 3.0, 10.0])
    c = a.logb(b)
    assert_data_close(c, [3.0, 3.0, 2.0], tolerance=1e-4, label="logb")

def test_binary_f64():
    a = rp.values([2.0, 3.0, 4.0], dtype='float64')
    b = rp.values([1.0, 2.0, 3.0], dtype='float64')
    c = a.add(b)
    assert c.dtype == 'float64', f"Expected float64, got {c.dtype}"
    assert_data_close(c, [3.0, 5.0, 7.0], label="add_f64")
    c = a.mul(b)
    assert_data_close(c, [2.0, 6.0, 12.0], label="mul_f64")

def test_add_scalar():
    a = rp.values([1.0, 2.0, 3.0])
    c = a.add_scalar(5.0)
    assert_data_close(c, [6.0, 7.0, 8.0], label="add_scalar")

def test_sub_scalar():
    a = rp.values([10.0, 20.0, 30.0])
    c = a.sub_scalar(3.0)
    assert_data_close(c, [7.0, 17.0, 27.0], label="sub_scalar")

def test_mul_scalar():
    a = rp.values([1.0, 2.0, 3.0])
    c = a.mul_scalar(2.0)
    assert_data_close(c, [2.0, 4.0, 6.0], label="mul_scalar")

def test_div_scalar():
    a = rp.values([10.0, 20.0, 30.0])
    c = a.div_scalar(5.0)
    assert_data_close(c, [2.0, 4.0, 6.0], label="div_scalar")

def test_pow_scalar():
    a = rp.values([2.0, 3.0, 4.0])
    c = a.pow_scalar(2.0)
    assert_data_close(c, [4.0, 9.0, 16.0], label="pow_scalar")

def test_rsub_scalar():
    a = rp.values([1.0, 2.0, 3.0])
    c = a.rsub_scalar(10.0)
    assert_data_close(c, [9.0, 8.0, 7.0], label="rsub_scalar")

def test_rdiv_scalar():
    a = rp.values([1.0, 2.0, 3.0])
    c = a.rdiv_scalar(12.0)
    assert_data_close(c, [12.0, 6.0, 4.0], label="rdiv_scalar")

def test_rpow_scalar():
    a = rp.values([1.0, 2.0, 3.0])
    c = a.rpow_scalar(2.0)
    assert_data_close(c, [2.0, 4.0, 8.0], label="rpow_scalar")

def test_logb_scalar():
    a = rp.values([8.0, 16.0, 32.0])
    c = a.logb_scalar(2.0)
    assert_data_close(c, [3.0, 4.0, 5.0], tolerance=1e-4, label="logb_scalar")

def test_rlogb_scalar():
    a = rp.values([2.0, 3.0, 10.0])
    c = a.rlogb_scalar(8.0)
    expected = [math.log(8.0)/math.log(2.0), math.log(8.0)/math.log(3.0), math.log(8.0)/math.log(10.0)]
    assert_data_close(c, expected, tolerance=1e-4, label="rlogb_scalar")

def test_scalar_f64():
    a = rp.values([1.0, 2.0, 3.0], dtype='float64')
    c = a.add_scalar(10.0)
    assert c.dtype == 'float64', f"Expected float64, got {c.dtype}"
    assert_data_close(c, [11.0, 12.0, 13.0], label="add_scalar_f64")

def test_exp():
    a = rp.values([0.0, 1.0, 2.0])
    c = a.exp()
    assert_data_close(c, [math.exp(0.0), math.exp(1.0), math.exp(2.0)], label="exp")

def test_log():
    a = rp.values([1.0, math.e, math.e**2])
    c = a.log()
    assert_data_close(c, [0.0, 1.0, 2.0], tolerance=1e-4, label="log")

def test_sqrt():
    a = rp.values([1.0, 4.0, 9.0, 16.0])
    c = a.sqrt()
    assert_data_close(c, [1.0, 2.0, 3.0, 4.0], label="sqrt")

def test_abs():
    a = rp.values([-2.0, -1.0, 0.0, 1.0, 2.0])
    c = a.abs()
    assert_data_close(c, [2.0, 1.0, 0.0, 1.0, 2.0], label="abs")

def test_sin():
    a = rp.values([0.0, math.pi/2, math.pi])
    c = a.sin()
    assert_data_close(c, [0.0, 1.0, 0.0], tolerance=1e-4, label="sin")

def test_cos():
    a = rp.values([0.0, math.pi/2, math.pi])
    c = a.cos()
    assert_data_close(c, [1.0, 0.0, -1.0], tolerance=1e-4, label="cos")

def test_tan():
    a = rp.values([0.0, math.pi/4])
    c = a.tan()
    assert_data_close(c, [0.0, 1.0], tolerance=1e-4, label="tan")

def test_asin():
    a = rp.values([0.0, 0.5, 1.0])
    c = a.asin()
    assert_data_close(c, [0.0, math.asin(0.5), math.pi/2], tolerance=1e-4, label="asin")

def test_acos():
    a = rp.values([1.0, 0.5, 0.0])
    c = a.acos()
    assert_data_close(c, [0.0, math.acos(0.5), math.pi/2], tolerance=1e-4, label="acos")

def test_atan():
    a = rp.values([0.0, 1.0, -1.0])
    c = a.atan()
    assert_data_close(c, [0.0, math.pi/4, -math.pi/4], tolerance=1e-4, label="atan")

def test_sinh():
    a = rp.values([0.0, 1.0, -1.0])
    c = a.sinh()
    assert_data_close(c, [math.sinh(0.0), math.sinh(1.0), math.sinh(-1.0)], tolerance=1e-4, label="sinh")

def test_cosh():
    a = rp.values([0.0, 1.0, -1.0])
    c = a.cosh()
    assert_data_close(c, [math.cosh(0.0), math.cosh(1.0), math.cosh(-1.0)], tolerance=1e-4, label="cosh")

def test_tanh():
    a = rp.values([0.0, 1.0, -1.0])
    c = a.tanh()
    assert_data_close(c, [math.tanh(0.0), math.tanh(1.0), math.tanh(-1.0)], tolerance=1e-4, label="tanh")

def test_relu():
    a = rp.values([-2.0, -1.0, 0.0, 1.0, 2.0])
    c = a.relu()
    assert_data_close(c, [0.0, 0.0, 0.0, 1.0, 2.0], label="relu")

def test_sigmoid():
    a = rp.values([-2.0, 0.0, 2.0])
    c = a.sigmoid()
    expected = [1.0/(1.0 + math.exp(2.0)), 0.5, 1.0/(1.0 + math.exp(-2.0))]
    assert_data_close(c, expected, tolerance=1e-4, label="sigmoid")

def test_leaky_relu():
    a = rp.values([-2.0, -1.0, 0.0, 1.0, 2.0])
    c = a.leaky_relu(0.1)
    assert_data_close(c, [-0.2, -0.1, 0.0, 1.0, 2.0], label="leaky_relu")

def test_gelu():
    a = rp.values([-1.0, 0.0, 1.0])
    c = a.gelu()
    sqrt_2_over_pi = 0.7978845608
    coeff = 0.044715
    def gelu_ref(x):
        return 0.5 * x * (1.0 + math.tanh(sqrt_2_over_pi * (x + coeff * x**3)))
    assert_data_close(c, [gelu_ref(-1.0), gelu_ref(0.0), gelu_ref(1.0)], label="gelu")

def test_silu():
    a = rp.values([-1.0, 0.0, 1.0])
    c = a.silu()
    def silu_ref(x):
        return x / (1.0 + math.exp(-x))
    assert_data_close(c, [silu_ref(-1.0), silu_ref(0.0), silu_ref(1.0)], label="silu")

def test_square():
    a = rp.values([1.0, 2.0, 3.0, 4.0])
    c = a.square()
    assert_data_close(c, [1.0, 4.0, 9.0, 16.0], label="square")

def test_ceil():
    a = rp.values([1.2, 2.5, 3.7, -1.3])
    c = a.ceil()
    assert_data_close(c, [2.0, 3.0, 4.0, -1.0], label="ceil")

def test_floor():
    a = rp.values([1.2, 2.5, 3.7, -1.3])
    c = a.floor()
    assert_data_close(c, [1.0, 2.0, 3.0, -2.0], label="floor")

def test_trunc():
    a = rp.values([1.7, 2.5, -1.7, -2.5])
    c = a.trunc()
    assert_data_close(c, [1.0, 2.0, -1.0, -2.0], label="trunc")

def test_rsqrt():
    a = rp.values([1.0, 4.0, 9.0, 16.0])
    c = a.rsqrt()
    assert_data_close(c, [1.0, 0.5, 1.0/3.0, 0.25], tolerance=1e-4, label="rsqrt")

def test_unary_f64():
    a = rp.values([0.0, 1.0, 2.0], dtype='float64')
    c = a.exp()
    assert c.dtype == 'float64', f"Expected float64, got {c.dtype}"
    assert_data_close(c, [math.exp(0.0), math.exp(1.0), math.exp(2.0)], tolerance=1e-10, label="exp_f64")
    b = rp.values([1.0, 4.0, 9.0], dtype='float64')
    c = b.sqrt()
    assert_data_close(c, [1.0, 2.0, 3.0], tolerance=1e-10, label="sqrt_f64")

def test_sum_all():
    a = rp.values([1.0, 2.0, 3.0, 4.0, 5.0])
    result = a.sum_all()
    assert_close(result, 15.0, label="sum_all")

def test_mean_all():
    a = rp.values([1.0, 2.0, 3.0, 4.0, 5.0])
    result = a.mean_all()
    assert_close(result, 3.0, label="mean_all")

def test_sum_large():
    a = rp.ones([100])
    assert_close(a.sum_all(), 100.0, label="sum_all_large")
    assert_close(a.mean_all(), 1.0, label="mean_all_large")

def test_matmul():
    a_1d = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = a_1d.reshape([2, 3])
    b_1d = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = b_1d.reshape([3, 2])
    c = a.op_matmul(b)
    assert c.shape == (2, 2), f"Expected (2, 2), got {c.shape}"
    expected = [
        1*1 + 2*3 + 3*5,
        1*2 + 2*4 + 3*6,
        4*1 + 5*3 + 6*5,
        4*2 + 5*4 + 6*6,
    ]
    assert_data_close(c, expected, label="matmul")

def test_matmul_identity():
    a_1d = rp.values([1.0, 0.0, 0.0, 1.0])
    a = a_1d.reshape([2, 2])
    b_1d = rp.values([5.0, 6.0, 7.0, 8.0])
    b = b_1d.reshape([2, 2])
    c = a.op_matmul(b)
    assert_data_close(c, [5.0, 6.0, 7.0, 8.0], label="matmul_identity")

def test_dtype_preservation_f32():
    a = rp.values([1.0, 2.0], dtype='float32')
    b = rp.values([3.0, 4.0], dtype='float32')
    c = a.add(b)
    assert c.dtype == 'float32', f"Expected float32, got {c.dtype}"

def test_dtype_preservation_f64():
    a = rp.values([1.0, 2.0], dtype='float64')
    b = rp.values([3.0, 4.0], dtype='float64')
    c = a.add(b)
    assert c.dtype == 'float64', f"Expected float64, got {c.dtype}"

def test_shape_preservation_1d():
    a = rp.values([1.0, 2.0, 3.0])
    c = a.add_scalar(1.0)
    assert c.shape == a.shape, f"Shape mismatch: {c.shape} != {a.shape}"

def test_shape_preservation_2d():
    a = rp.ones([3, 4])
    c = a.mul_scalar(2.0)
    assert c.shape == a.shape, f"Shape mismatch: {c.shape} != {a.shape}"


print("=" * 60)
print("  test_functional.py - Functional Layer Operations")
print("=" * 60)

print("\n-- Binary Operations --")
run_test("add", test_add)
run_test("sub", test_sub)
run_test("mul", test_mul)
run_test("divide", test_divide)
run_test("power", test_power)
run_test("logb", test_logb)
run_test("binary ops float64", test_binary_f64)

print("\n-- Scalar Operations --")
run_test("add_scalar", test_add_scalar)
run_test("sub_scalar", test_sub_scalar)
run_test("mul_scalar", test_mul_scalar)
run_test("div_scalar", test_div_scalar)
run_test("pow_scalar", test_pow_scalar)
run_test("rsub_scalar", test_rsub_scalar)
run_test("rdiv_scalar", test_rdiv_scalar)
run_test("rpow_scalar", test_rpow_scalar)
run_test("logb_scalar", test_logb_scalar)
run_test("rlogb_scalar", test_rlogb_scalar)
run_test("scalar ops float64", test_scalar_f64)

print("\n-- Unary Operations --")
run_test("exp", test_exp)
run_test("log", test_log)
run_test("sqrt", test_sqrt)
run_test("abs", test_abs)
run_test("sin", test_sin)
run_test("cos", test_cos)
run_test("tan", test_tan)
run_test("asin", test_asin)
run_test("acos", test_acos)
run_test("atan", test_atan)
run_test("sinh", test_sinh)
run_test("cosh", test_cosh)
run_test("tanh", test_tanh)

print("\n-- Activation Functions --")
run_test("relu", test_relu)
run_test("sigmoid", test_sigmoid)
run_test("leaky_relu", test_leaky_relu)
run_test("gelu", test_gelu)
run_test("silu", test_silu)

print("\n-- Utility Operations --")
run_test("square", test_square)
run_test("ceil", test_ceil)
run_test("floor", test_floor)
run_test("trunc", test_trunc)
run_test("rsqrt", test_rsqrt)
run_test("unary ops float64", test_unary_f64)

print("\n-- Reduction Operations --")
run_test("sum_all", test_sum_all)
run_test("mean_all", test_mean_all)
run_test("sum/mean large", test_sum_large)

print("\n-- Matrix Operations --")
run_test("matmul 2x3 @ 3x2", test_matmul)
run_test("matmul identity", test_matmul_identity)

print("\n-- Type and Shape Consistency --")
run_test("dtype preservation float32", test_dtype_preservation_f32)
run_test("dtype preservation float64", test_dtype_preservation_f64)
run_test("shape preservation 1D", test_shape_preservation_1d)
run_test("shape preservation 2D", test_shape_preservation_2d)

print()
print(f"{passed}/{passed + failed} tests passed")
if failed > 0:
    sys.exit(1)
