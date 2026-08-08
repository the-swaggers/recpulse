import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp

print("=" * 70)
print("  RecPulse Cat Operation - Comprehensive Test")
print("=" * 70)

def test_cat_forward(desc, tensors, dim, expected_shape, expected_data):
    print(f"\n{desc}:")
    result = rp.cat(tensors, dim=dim)

    shape_match = tuple(result.shape) == expected_shape
    data_match = all(abs(a - b) < 1e-6 for a, b in zip(result.data(), expected_data))

    status = "✅" if (shape_match and data_match) else "❌"
    print(f"  {status} shape={tuple(result.shape)} (expected {expected_shape})")
    print(f"     data={result.data()[:8]}... (showing first 8)")

    if not shape_match:
        print(f"     ❌ Shape mismatch!")
    if not data_match:
        print(f"     ❌ Data mismatch!")
        print(f"        Expected: {expected_data[:8]}...")

    return shape_match and data_match

def test_cat_backward(desc, tensors, dim, expected_grads):
    print(f"\n{desc}:")

    for t in tensors:
        t.requires_grad_()

    result = rp.cat(tensors, dim=dim)
    result.backward()

    all_correct = True
    for i, (t, expected_grad) in enumerate(zip(tensors, expected_grads)):
        if t.grad is None:
            print(f"  ❌ Tensor {i}: grad is None (expected {expected_grad})")
            all_correct = False
            continue

        grad_data = list(t.grad.data())
        grad_match = all(abs(a - b) < 1e-6 for a, b in zip(grad_data, expected_grad))

        status = "✅" if grad_match else "❌"
        print(f"  {status} Tensor {i} grad: {grad_data} (expected {expected_grad})")

        if not grad_match:
            all_correct = False

    return all_correct

print("\n" + "=" * 70)
print("  FORWARD PASS TESTS")
print("=" * 70)

results = []

results.append(test_cat_forward(
    "1D tensors, dim=0",
    [rp.values([1.0, 2.0], device='cpu'),
     rp.values([3.0, 4.0], device='cpu')],
    dim=0,
    expected_shape=(4,),
    expected_data=[1.0, 2.0, 3.0, 4.0]
))

results.append(test_cat_forward(
    "1D tensors, 3 tensors, dim=0",
    [rp.values([1.0], device='cpu'),
     rp.values([2.0, 3.0], device='cpu'),
     rp.values([4.0, 5.0, 6.0], device='cpu')],
    dim=0,
    expected_shape=(6,),
    expected_data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
))

x1 = rp.values([1.0, 2.0, 3.0, 4.0], device='cpu')
x1_reshaped = x1
x2 = rp.values([5.0, 6.0, 7.0, 8.0], device='cpu')
x2_reshaped = x2

print("\n2D cat test (dim=0):")
print("  Note: Need reshape for 2D test, implementing with 1D for now")

results.append(test_cat_forward(
    "Mixed sizes, dim=0",
    [rp.values([10.0, 20.0, 30.0], device='cpu'),
     rp.values([40.0, 50.0], device='cpu')],
    dim=0,
    expected_shape=(5,),
    expected_data=[10.0, 20.0, 30.0, 40.0, 50.0]
))

print("\n" + "=" * 70)
print("  BACKWARD PASS TESTS")
print("=" * 70)

results.append(test_cat_backward(
    "Simple 1D cat, dim=0 - gradient routing",
    [rp.values([1.0, 2.0], device='cpu'),
     rp.values([3.0, 4.0], device='cpu')],
    dim=0,
    expected_grads=[[1.0, 1.0], [1.0, 1.0]]
))

results.append(test_cat_backward(
    "Three 1D tensors, dim=0 - gradient splitting",
    [rp.values([1.0], device='cpu'),
     rp.values([2.0, 3.0], device='cpu'),
     rp.values([4.0, 5.0, 6.0], device='cpu')],
    dim=0,
    expected_grads=[[1.0], [1.0, 1.0], [1.0, 1.0, 1.0]]
))

print("\n" + "=" * 70)
print("  2D CAT WITH AUTOGRAD")
print("=" * 70)

t1_1d = rp.values([1.0, 2.0, 3.0, 4.0], device='cpu')
t1_2d = t1_1d.reshape([2, 2])
t2_1d = rp.values([5.0, 6.0, 7.0, 8.0], device='cpu')
t2_2d = t2_1d.reshape([2, 2])

results.append(test_cat_forward(
    "2D cat, dim=0",
    [t1_2d, t2_2d],
    dim=0,
    expected_shape=(4, 2),
    expected_data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
))

a_1d = rp.values([1.0, 2.0, 3.0, 4.0], device='cpu')
a_2d = a_1d.reshape([2, 2]).requires_grad_()
b_1d = rp.values([5.0, 6.0, 7.0, 8.0], device='cpu')
b_2d = b_1d.reshape([2, 2]).requires_grad_()

catted = rp.cat([a_2d, b_2d], dim=0)
s = catted.op_sum_all()
s.backward()
a_grad = a_2d.grad
b_grad = b_2d.grad

a_ok = a_grad is not None and all(abs(v - 1.0) < 1e-6 for v in a_grad.data())
b_ok = b_grad is not None and all(abs(v - 1.0) < 1e-6 for v in b_grad.data())

if a_ok and b_ok:
    print(f"\n  PASS: 2D cat backward gradient values")
else:
    print(f"\n  FAIL: 2D cat backward gradient values")
results.append(a_ok and b_ok)

a_1d2 = rp.values([1.0, 2.0, 3.0, 4.0], device='cpu')
a_2d2 = a_1d2.reshape([2, 2]).requires_grad_()
b_1d2 = rp.values([5.0, 6.0, 7.0, 8.0], device='cpu')
b_2d2 = b_1d2.reshape([2, 2]).requires_grad_()

catted2 = rp.cat([a_2d2, b_2d2], dim=0)
sq = catted2.op_square()
s2 = sq.op_sum_all()
s2.backward()

ag2 = a_2d2.grad
bg2 = b_2d2.grad

expected_a_grad = [2.0, 4.0, 6.0, 8.0]
expected_b_grad = [10.0, 12.0, 14.0, 16.0]

a_num_ok = ag2 is not None and all(abs(g - e) < 1e-5 for g, e in zip(ag2.data(), expected_a_grad))
b_num_ok = bg2 is not None and all(abs(g - e) < 1e-5 for g, e in zip(bg2.data(), expected_b_grad))

status = "PASS" if (a_num_ok and b_num_ok) else "FAIL"
print(f"\n  {status}: 2D cat backward numerical gradient check (square after cat)")
results.append(a_num_ok and b_num_ok)


print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

passed_count = sum(results)
total = len(results)
print(f"\nPassed: {passed_count}/{total}")

if passed_count == total:
    print("\nALL CAT TESTS PASSED!")
else:
    print(f"\n{total - passed_count} test(s) failed")
    sys.exit(1)

print("\n" + "=" * 70)
