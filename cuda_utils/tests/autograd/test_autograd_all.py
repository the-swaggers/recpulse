import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import math

print("=" * 70)
print("  RecPulse Autograd - Comprehensive Test")
print("=" * 70)

def test_binary_op(op_name, op_func, x_val, y_val, expected_grad_x, expected_grad_y):
    """Test a binary operation with autograd"""
    print(f"\n{op_name}:")
    x = rp.values([x_val], device='cpu').requires_grad_()
    y = rp.values([y_val], device='cpu').requires_grad_()

    z = op_func(x, y)
    z.backward()

    actual_grad_x = x.grad.data()[0]
    actual_grad_y = y.grad.data()[0]

    x_correct = abs(actual_grad_x - expected_grad_x) < 1e-5
    y_correct = abs(actual_grad_y - expected_grad_y) < 1e-5

    status = "✅" if (x_correct and y_correct) else "❌"
    print(f"  {status} z = {op_name}({x_val}, {y_val}) = {z.data()[0]:.6f}")
    print(f"     dz/dx = {actual_grad_x:.6f} (expected {expected_grad_x:.6f})")
    print(f"     dz/dy = {actual_grad_y:.6f} (expected {expected_grad_y:.6f})")

    return x_correct and y_correct

def test_unary_op(op_name, op_func, x_val, expected_output, expected_grad):
    """Test a unary operation with autograd"""
    print(f"\n{op_name}:")
    x = rp.values([x_val], device='cpu').requires_grad_()

    z = op_func(x)
    z.backward()

    actual_output = z.data()[0]
    actual_grad = x.grad.data()[0]

    output_correct = abs(actual_output - expected_output) < 1e-5
    grad_correct = abs(actual_grad - expected_grad) < 1e-5

    status = "✅" if (output_correct and grad_correct) else "❌"
    print(f"  {status} z = {op_name}({x_val}) = {actual_output:.6f} (expected {expected_output:.6f})")
    print(f"     dz/dx = {actual_grad:.6f} (expected {expected_grad:.6f})")

    return output_correct and grad_correct

print("\n" + "=" * 70)
print("  BINARY OPERATIONS")
print("=" * 70)

results = []

# Test op_add: z = x + y, dz/dx = 1, dz/dy = 1
results.append(test_binary_op(
    "op_add", lambda x, y: x.op_add(y),
    2.0, 3.0,
    expected_grad_x=1.0,
    expected_grad_y=1.0
))

# Test op_mul: z = x * y, dz/dx = y, dz/dy = x
results.append(test_binary_op(
    "op_mul", lambda x, y: x.op_mul(y),
    2.0, 3.0,
    expected_grad_x=3.0,  # y
    expected_grad_y=2.0   # x
))

# Test op_sub: z = x - y, dz/dx = 1, dz/dy = -1
results.append(test_binary_op(
    "op_sub", lambda x, y: x.op_sub(y),
    5.0, 2.0,
    expected_grad_x=1.0,
    expected_grad_y=-1.0
))

# Test op_div: z = x / y, dz/dx = 1/y, dz/dy = -x/y^2
results.append(test_binary_op(
    "op_div", lambda x, y: x.op_div(y),
    6.0, 2.0,
    expected_grad_x=0.5,   # 1/2
    expected_grad_y=-1.5   # -6/4
))

print("\n" + "=" * 70)
print("  UNARY OPERATIONS")
print("=" * 70)

# Test op_exp: z = exp(x), dz/dx = exp(x)
x_val = 1.0
results.append(test_unary_op(
    "op_exp", lambda x: x.op_exp(),
    x_val,
    expected_output=math.exp(x_val),
    expected_grad=math.exp(x_val)
))

# Test op_log: z = log(x), dz/dx = 1/x
x_val = 2.0
results.append(test_unary_op(
    "op_log", lambda x: x.op_log(),
    x_val,
    expected_output=math.log(x_val),
    expected_grad=1.0/x_val
))

# Test op_sqrt: z = sqrt(x), dz/dx = 1/(2*sqrt(x))
x_val = 4.0
results.append(test_unary_op(
    "op_sqrt", lambda x: x.op_sqrt(),
    x_val,
    expected_output=math.sqrt(x_val),
    expected_grad=1.0/(2.0*math.sqrt(x_val))
))

# Test op_tanh: z = tanh(x), dz/dx = 1 - tanh(x)^2
x_val = 0.5
tanh_val = math.tanh(x_val)
results.append(test_unary_op(
    "op_tanh", lambda x: x.op_tanh(),
    x_val,
    expected_output=tanh_val,
    expected_grad=1.0 - tanh_val**2
))

print("\n" + "=" * 70)
print("  CHAIN OPERATION TEST")
print("=" * 70)

def test_chain_op():
    x = rp.values([2.0], device='cpu').requires_grad_()
    y = rp.values([3.0], device='cpu').requires_grad_()
    z = rp.values([1.0], device='cpu').requires_grad_()

    xy = x.op_mul(y)
    result = xy.op_add(z)
    s = result.op_sum_all()
    s.backward()

    gx = x.grad.data()[0]
    gy = y.grad.data()[0]
    gz = z.grad.data()[0]

    gx_ok = abs(gx - 3.0) < 1e-5
    gy_ok = abs(gy - 2.0) < 1e-5
    gz_ok = abs(gz - 1.0) < 1e-5

    status = "PASS" if (gx_ok and gy_ok and gz_ok) else "FAIL"
    print(f"\n{status}: chain x*y + z backward")
    print(f"  dL/dx = {gx:.6f} (expected 3.0)")
    print(f"  dL/dy = {gy:.6f} (expected 2.0)")
    print(f"  dL/dz = {gz:.6f} (expected 1.0)")
    return gx_ok and gy_ok and gz_ok

results.append(test_chain_op())

print("\n" + "=" * 70)
print("  FLOAT64 TEST")
print("=" * 70)

def test_float64_mul():
    x = rp.values([2.0], dtype='float64', device='cpu').requires_grad_()
    y = rp.values([3.0], dtype='float64', device='cpu').requires_grad_()

    z = x.op_mul(y)
    z.backward()

    gx = x.grad.data()[0]
    gy = y.grad.data()[0]

    gx_ok = abs(gx - 3.0) < 1e-10
    gy_ok = abs(gy - 2.0) < 1e-10

    status = "PASS" if (gx_ok and gy_ok) else "FAIL"
    print(f"\n{status}: float64 mul backward")
    print(f"  dz/dx = {gx:.10f} (expected 3.0)")
    print(f"  dz/dy = {gy:.10f} (expected 2.0)")
    return gx_ok and gy_ok

results.append(test_float64_mul())

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

passed = sum(results)
total = len(results)
print(f"\nPassed: {passed}/{total}")

if passed == total:
    print("\nALL AUTOGRAD OPERATIONS WORK CORRECTLY!")
else:
    print(f"\n{total - passed} test(s) failed")
    sys.exit(1)

print("\n" + "=" * 70)
