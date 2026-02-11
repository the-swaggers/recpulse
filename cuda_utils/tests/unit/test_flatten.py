#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import recpulse_cuda as rp

passed = 0
failed = 0

def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"PASS: {name}")
        passed += 1
    except AssertionError as e:
        print(f"FAIL: {name} - {e}")
        failed += 1
    except Exception as e:
        print(f"FAIL: {name} - {type(e).__name__}: {e}")
        failed += 1


def test_flatten_first_two_dims():
    t = rp.zeros([2, 3, 4], dtype='float32', device='cpu')
    flat = t.flatten(0, 1)
    assert flat.shape == (6, 4), f"Expected (6, 4), got {flat.shape}"
    assert flat.size == 24, f"Expected size 24, got {flat.size}"

def test_flatten_all_dims():
    t = rp.ones([2, 3, 4], dtype='float32', device='cpu')
    flat = t.flatten(0, -1)
    assert flat.shape == (24,), f"Expected (24,), got {flat.shape}"
    assert flat.ndim == 1, f"Expected ndim 1, got {flat.ndim}"

def test_flatten_middle_dims():
    t = rp.zeros([2, 3, 4, 5], dtype='float32', device='cpu')
    flat = t.flatten(1, 2)
    assert flat.shape == (2, 12, 5), f"Expected (2, 12, 5), got {flat.shape}"

def test_flatten_negative_indices():
    t = rp.ones([2, 3, 4], dtype='float32', device='cpu')
    flat = t.flatten(-2, -1)
    assert flat.shape == (2, 12), f"Expected (2, 12), got {flat.shape}"

def test_flatten_value_preservation():
    data = [float(i) for i in range(24)]
    t = rp.values(data, dtype='float32', device='cpu')
    t3d = t.reshape([2, 3, 4])
    flat = t3d.flatten(0, -1)
    assert flat.shape == (24,), f"Expected (24,), got {flat.shape}"
    flat_data = flat.data()
    for i in range(24):
        assert abs(flat_data[i] - data[i]) < 1e-6, f"Value mismatch at {i}: {flat_data[i]} vs {data[i]}"

def test_flatten_partial_value_preservation():
    data = [float(i) for i in range(24)]
    t = rp.values(data, dtype='float32', device='cpu')
    t3d = t.reshape([2, 3, 4])
    flat = t3d.flatten(0, 1)
    assert flat.shape == (6, 4), f"Expected (6, 4), got {flat.shape}"
    flat_data = flat.data()
    for i in range(24):
        assert abs(flat_data[i] - data[i]) < 1e-6, f"Value mismatch at {i}: {flat_data[i]} vs {data[i]}"

def test_flatten_single_dim_noop():
    t = rp.zeros([2, 3, 4], dtype='float32', device='cpu')
    flat = t.flatten(0, 0)
    assert flat.shape == (2, 3, 4), f"Expected (2, 3, 4), got {flat.shape}"
    assert flat.ndim == 3, f"Expected ndim 3, got {flat.ndim}"

def test_flatten_single_dim_middle_noop():
    t = rp.zeros([2, 3, 4], dtype='float32', device='cpu')
    flat = t.flatten(1, 1)
    assert flat.shape == (2, 3, 4), f"Expected (2, 3, 4), got {flat.shape}"

def test_flatten_5d_to_2d():
    t = rp.ones([2, 3, 4, 5, 6], dtype='float32', device='cpu')
    flat = t.flatten(1, 3)
    assert flat.shape == (2, 60, 6), f"Expected (2, 60, 6), got {flat.shape}"
    assert flat.size == 720, f"Expected size 720, got {flat.size}"

def test_flatten_5d_all():
    t = rp.zeros([2, 3, 4, 5, 6], dtype='float32', device='cpu')
    flat = t.flatten(0, -1)
    assert flat.shape == (720,), f"Expected (720,), got {flat.shape}"
    assert flat.ndim == 1, f"Expected ndim 1, got {flat.ndim}"

def test_flatten_float64():
    data = [float(i) for i in range(12)]
    t = rp.values(data, dtype='float64', device='cpu')
    t3d = t.reshape([2, 3, 2])
    assert t3d.dtype == 'float64', f"Expected float64, got {t3d.dtype}"
    flat = t3d.flatten(0, 1)
    assert flat.shape == (6, 2), f"Expected (6, 2), got {flat.shape}"
    assert flat.dtype == 'float64', f"Expected float64, got {flat.dtype}"
    flat_data = flat.data()
    for i in range(12):
        assert abs(flat_data[i] - data[i]) < 1e-12, f"Value mismatch at {i}"

def test_flatten_preserves_size():
    t = rp.ones([3, 4, 5, 2], dtype='float32', device='cpu')
    flat = t.flatten(1, 3)
    assert flat.size == t.size, f"Expected size {t.size}, got {flat.size}"
    assert flat.shape == (3, 40), f"Expected (3, 40), got {flat.shape}"

def test_flatten_last_two_dims():
    t = rp.zeros([2, 3, 4, 5], dtype='float32', device='cpu')
    flat = t.flatten(2, 3)
    assert flat.shape == (2, 3, 20), f"Expected (2, 3, 20), got {flat.shape}"

def test_flatten_1d_noop():
    t = rp.ones([10], dtype='float32', device='cpu')
    flat = t.flatten(0, 0)
    assert flat.shape == (10,), f"Expected (10,), got {flat.shape}"
    assert flat.size == 10, f"Expected size 10, got {flat.size}"


print("=" * 60)
print("Testing flatten operation")
print("=" * 60)

run_test("flatten first two dims [2,3,4] -> [6,4]", test_flatten_first_two_dims)
run_test("flatten all dims [2,3,4] -> [24]", test_flatten_all_dims)
run_test("flatten middle dims [2,3,4,5] -> [2,12,5]", test_flatten_middle_dims)
run_test("flatten negative indices [2,3,4] -> [2,12]", test_flatten_negative_indices)
run_test("flatten all dims value preservation", test_flatten_value_preservation)
run_test("flatten partial dims value preservation", test_flatten_partial_value_preservation)
run_test("flatten single dim no-op (0,0)", test_flatten_single_dim_noop)
run_test("flatten single dim no-op (1,1)", test_flatten_single_dim_middle_noop)
run_test("flatten 5D partial [2,3,4,5,6] -> [2,60,6]", test_flatten_5d_to_2d)
run_test("flatten 5D all -> [720]", test_flatten_5d_all)
run_test("flatten float64", test_flatten_float64)
run_test("flatten preserves total size", test_flatten_preserves_size)
run_test("flatten last two dims [2,3,4,5] -> [2,3,20]", test_flatten_last_two_dims)
run_test("flatten 1D no-op", test_flatten_1d_noop)

print("=" * 60)
print(f"{passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"{failed} test(s) FAILED")
    sys.exit(1)
