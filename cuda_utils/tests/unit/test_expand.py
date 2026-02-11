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


def test_expand_size1_dim():
    t = rp.zeros([3, 1, 5], dtype='float32', device='cpu')
    expanded = t.expand([3, 4, 5])
    assert expanded.shape == (3, 4, 5), f"Expected (3, 4, 5), got {expanded.shape}"
    assert expanded.size == 60, f"Expected size 60, got {expanded.size}"

def test_expand_prepend_dim():
    t = rp.ones([5], dtype='float32', device='cpu')
    expanded = t.expand([3, 5])
    assert expanded.shape == (3, 5), f"Expected (3, 5), got {expanded.shape}"

def test_expand_multiple_dims():
    t = rp.zeros([1, 1, 5], dtype='float32', device='cpu')
    expanded = t.expand([3, 4, 5])
    assert expanded.shape == (3, 4, 5), f"Expected (3, 4, 5), got {expanded.shape}"

def test_expand_prepend_and_broadcast():
    t = rp.ones([1, 5], dtype='float32', device='cpu')
    expanded = t.expand([2, 3, 5])
    assert expanded.shape == (2, 3, 5), f"Expected (2, 3, 5), got {expanded.shape}"

def test_expand_identity():
    t = rp.zeros([3, 4, 5], dtype='float32', device='cpu')
    expanded = t.expand([3, 4, 5])
    assert expanded.shape == (3, 4, 5), f"Expected (3, 4, 5), got {expanded.shape}"

def test_expand_invalid_fails():
    t = rp.ones([3, 4, 5], dtype='float32', device='cpu')
    try:
        t.expand([3, 2, 5])
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

def test_expand_all_broadcast():
    t = rp.zeros([1, 1, 1], dtype='float32', device='cpu')
    expanded = t.expand([2, 3, 4])
    assert expanded.shape == (2, 3, 4), f"Expected (2, 3, 4), got {expanded.shape}"

def test_expand_batch():
    t = rp.ones([1, 128], dtype='float32', device='cpu')
    expanded = t.expand([32, 128])
    assert expanded.shape == (32, 128), f"Expected (32, 128), got {expanded.shape}"

def test_expand_ones_values():
    t = rp.ones([1, 3], dtype='float32', device='cpu')
    expanded = t.expand([4, 3])
    assert expanded.shape == (4, 3), f"Expected (4, 3), got {expanded.shape}"
    contiguous = expanded.reshape([12])
    data = contiguous.data()
    assert len(data) == 12, f"Expected 12 elements, got {len(data)}"
    for i in range(12):
        assert abs(data[i] - 1.0) < 1e-6, f"Expected 1.0 at index {i}, got {data[i]}"

def test_expand_values_broadcast():
    t = rp.values([1.0, 2.0, 3.0], dtype='float32', device='cpu')
    t2d = t.reshape([1, 3])
    expanded = t2d.expand([3, 3])
    assert expanded.shape == (3, 3), f"Expected (3, 3), got {expanded.shape}"
    contiguous = expanded.reshape([9])
    data = contiguous.data()
    expected = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    for i in range(9):
        assert abs(data[i] - expected[i]) < 1e-6, f"Value mismatch at {i}: {data[i]} vs {expected[i]}"

def test_expand_then_reshape_contiguous():
    t = rp.values([10.0, 20.0], dtype='float32', device='cpu')
    t2d = t.reshape([1, 2])
    expanded = t2d.expand([5, 2])
    assert expanded.shape == (5, 2), f"Expected (5, 2), got {expanded.shape}"
    contiguous = expanded.reshape([10])
    data = contiguous.data()
    for i in range(5):
        assert abs(data[i * 2] - 10.0) < 1e-6, f"Expected 10.0 at index {i*2}, got {data[i*2]}"
        assert abs(data[i * 2 + 1] - 20.0) < 1e-6, f"Expected 20.0 at index {i*2+1}, got {data[i*2+1]}"

def test_expand_float64():
    t = rp.values([1.0, 2.0, 3.0], dtype='float64', device='cpu')
    t2d = t.reshape([1, 3])
    assert t2d.dtype == 'float64', f"Expected float64, got {t2d.dtype}"
    expanded = t2d.expand([2, 3])
    assert expanded.shape == (2, 3), f"Expected (2, 3), got {expanded.shape}"
    assert expanded.dtype == 'float64', f"Expected float64, got {expanded.dtype}"
    contiguous = expanded.reshape([6])
    data = contiguous.data()
    expected = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    for i in range(6):
        assert abs(data[i] - expected[i]) < 1e-12, f"Value mismatch at {i}: {data[i]} vs {expected[i]}"

def test_expand_preserves_device():
    t = rp.ones([1, 3], dtype='float32', device='cpu')
    expanded = t.expand([4, 3])
    assert expanded.device == 'cpu', f"Expected cpu, got {expanded.device}"

def test_expand_size_correct():
    t = rp.zeros([1, 1, 5], dtype='float32', device='cpu')
    expanded = t.expand([3, 4, 5])
    assert expanded.size == 3 * 4 * 5, f"Expected {3*4*5}, got {expanded.size}"

def test_expand_ndim():
    t = rp.ones([5], dtype='float32', device='cpu')
    expanded = t.expand([3, 5])
    assert expanded.ndim == 2, f"Expected ndim 2, got {expanded.ndim}"

def test_expand_scalar_like():
    t = rp.values([42.0], dtype='float32', device='cpu')
    t3d = t.reshape([1, 1, 1])
    expanded = t3d.expand([2, 3, 4])
    assert expanded.shape == (2, 3, 4), f"Expected (2, 3, 4), got {expanded.shape}"
    contiguous = expanded.reshape([24])
    data = contiguous.data()
    for i in range(24):
        assert abs(data[i] - 42.0) < 1e-6, f"Expected 42.0 at index {i}, got {data[i]}"


print("=" * 60)
print("Testing expand operation")
print("=" * 60)

run_test("expand size-1 dim [3,1,5] -> [3,4,5]", test_expand_size1_dim)
run_test("expand prepend dim [5] -> [3,5]", test_expand_prepend_dim)
run_test("expand multiple dims [1,1,5] -> [3,4,5]", test_expand_multiple_dims)
run_test("expand prepend and broadcast [1,5] -> [2,3,5]", test_expand_prepend_and_broadcast)
run_test("identity expand [3,4,5] -> [3,4,5]", test_expand_identity)
run_test("invalid expand fails", test_expand_invalid_fails)
run_test("expand all broadcast [1,1,1] -> [2,3,4]", test_expand_all_broadcast)
run_test("batch expand [1,128] -> [32,128]", test_expand_batch)
run_test("expand ones tensor values", test_expand_ones_values)
run_test("expand values tensor broadcast", test_expand_values_broadcast)
run_test("expand then reshape contiguous", test_expand_then_reshape_contiguous)
run_test("expand float64", test_expand_float64)
run_test("expand preserves device", test_expand_preserves_device)
run_test("expand size correct", test_expand_size_correct)
run_test("expand ndim", test_expand_ndim)
run_test("expand scalar-like [1,1,1] -> [2,3,4] value check", test_expand_scalar_like)

print("=" * 60)
print(f"{passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"{failed} test(s) FAILED")
    sys.exit(1)
