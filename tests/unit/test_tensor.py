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
        passed += 1
        print(f"  PASS: {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL: {name} -- {e}")


def test_zeros_creation_f32():
    t = rp.zeros([2, 3])
    assert t.shape == (2, 3), f"Expected (2, 3), got {t.shape}"
    assert t.dtype == 'float32', f"Expected float32, got {t.dtype}"
    assert t.device == 'cpu', f"Expected cpu, got {t.device}"
    assert t.size == 6, f"Expected 6, got {t.size}"
    assert t.ndim == 2, f"Expected 2, got {t.ndim}"
    for v in t.data():
        assert v == 0.0, f"Expected 0.0, got {v}"

def test_ones_creation_f32():
    t = rp.ones([4, 5])
    assert t.shape == (4, 5), f"Expected (4, 5), got {t.shape}"
    assert t.dtype == 'float32', f"Expected float32, got {t.dtype}"
    assert t.size == 20, f"Expected 20, got {t.size}"
    for v in t.data():
        assert v == 1.0, f"Expected 1.0, got {v}"

def test_zeros_creation_f64():
    t = rp.zeros([3, 2], dtype='float64')
    assert t.dtype == 'float64', f"Expected float64, got {t.dtype}"
    assert t.shape == (3, 2), f"Expected (3, 2), got {t.shape}"
    assert t.size == 6, f"Expected 6, got {t.size}"
    for v in t.data():
        assert v == 0.0, f"Expected 0.0, got {v}"

def test_ones_creation_f64():
    t = rp.ones([2, 2], dtype='float64')
    assert t.dtype == 'float64', f"Expected float64, got {t.dtype}"
    for v in t.data():
        assert v == 1.0, f"Expected 1.0, got {v}"

def test_values_basic():
    t = rp.values([1.0, 2.0, 3.0])
    assert t.shape == (3,), f"Expected (3,), got {t.shape}"
    assert t.ndim == 1, f"Expected 1, got {t.ndim}"
    assert t.size == 3, f"Expected 3, got {t.size}"
    data = t.data()
    assert list(data) == [1.0, 2.0, 3.0], f"Expected [1.0, 2.0, 3.0], got {list(data)}"

def test_values_f64():
    t = rp.values([10.5, 20.5], dtype='float64')
    assert t.dtype == 'float64', f"Expected float64, got {t.dtype}"
    data = t.data()
    assert abs(data[0] - 10.5) < 1e-10, f"Expected 10.5, got {data[0]}"
    assert abs(data[1] - 20.5) < 1e-10, f"Expected 20.5, got {data[1]}"

def test_values_negative():
    t = rp.values([-5.0, -3.0, 0.0, 3.0, 5.0])
    data = t.data()
    expected = (-5.0, -3.0, 0.0, 3.0, 5.0)
    for i in range(5):
        assert data[i] == expected[i], f"Index {i}: expected {expected[i]}, got {data[i]}"

def test_values_single():
    t = rp.values([42.0])
    assert t.shape == (1,), f"Expected (1,), got {t.shape}"
    assert t.size == 1, f"Expected 1, got {t.size}"
    assert t.data()[0] == 42.0, f"Expected 42.0, got {t.data()[0]}"

def test_single_element_zeros():
    t = rp.zeros([1])
    assert t.shape == (1,), f"Expected (1,), got {t.shape}"
    assert t.size == 1, f"Expected 1, got {t.size}"
    assert t.ndim == 1, f"Expected 1, got {t.ndim}"
    assert t.data()[0] == 0.0, f"Expected 0.0, got {t.data()[0]}"

def test_large_tensor():
    t = rp.zeros([1000])
    assert t.shape == (1000,), f"Expected (1000,), got {t.shape}"
    assert t.size == 1000, f"Expected 1000, got {t.size}"
    for v in t.data():
        assert v == 0.0, f"Expected 0.0, got {v}"

def test_large_ones():
    t = rp.ones([1000])
    assert t.size == 1000, f"Expected 1000, got {t.size}"
    for v in t.data():
        assert v == 1.0, f"Expected 1.0, got {v}"

def test_properties_zeros_3d():
    t = rp.zeros([2, 3, 4])
    assert t.shape == (2, 3, 4), f"Expected (2, 3, 4), got {t.shape}"
    assert t.ndim == 3, f"Expected 3, got {t.ndim}"
    assert t.size == 24, f"Expected 24, got {t.size}"
    assert t.dtype == 'float32', f"Expected float32, got {t.dtype}"
    assert t.device == 'cpu', f"Expected cpu, got {t.device}"

def test_properties_1d():
    t = rp.values([1.0, 2.0, 3.0, 4.0, 5.0])
    assert t.ndim == 1, f"Expected 1, got {t.ndim}"
    assert t.shape == (5,), f"Expected (5,), got {t.shape}"
    assert t.size == 5, f"Expected 5, got {t.size}"

def test_copy_same_values():
    t = rp.values([1.0, 2.0, 3.0])
    c = t.copy()
    assert c.shape == t.shape, f"Copy shape {c.shape} != original {t.shape}"
    assert c.dtype == t.dtype, f"Copy dtype {c.dtype} != original {t.dtype}"
    assert c.device == t.device, f"Copy device {c.device} != original {t.device}"
    assert c.data() == t.data(), f"Copy data {c.data()} != original {t.data()}"
    assert c is not t, "Copy should be a different object"

def test_copy_independence():
    t = rp.values([10.0, 20.0, 30.0])
    c = t.copy()
    assert c.data() == t.data(), f"Copy data should match original"
    assert list(c.data()) == [10.0, 20.0, 30.0], f"Expected [10.0, 20.0, 30.0], got {list(c.data())}"

def test_to_dtype_conversion():
    t = rp.values([1.0, 2.0, 3.0], dtype='float32')
    t64 = t.to(dtype='float64')
    assert t64.dtype == 'float64', f"Expected float64, got {t64.dtype}"
    assert t.dtype == 'float32', f"Original should stay float32, got {t.dtype}"
    for i, (a, b) in enumerate(zip(t64.data(), t.data())):
        assert abs(a - b) < 1e-5, f"Index {i}: {a} != {b}"

def test_to_dtype_f64_to_f32():
    t = rp.values([1.5, 2.5, 3.5], dtype='float64')
    t32 = t.to(dtype='float32')
    assert t32.dtype == 'float32', f"Expected float32, got {t32.dtype}"
    for i, (a, b) in enumerate(zip(t32.data(), (1.5, 2.5, 3.5))):
        assert abs(a - b) < 1e-5, f"Index {i}: {a} != {b}"

def test_multidim_zeros():
    t = rp.zeros([2, 3, 4, 5])
    assert t.shape == (2, 3, 4, 5), f"Expected (2, 3, 4, 5), got {t.shape}"
    assert t.ndim == 4, f"Expected 4, got {t.ndim}"
    assert t.size == 120, f"Expected 120, got {t.size}"

def test_values_many_elements():
    data = [float(i) for i in range(100)]
    t = rp.values(data)
    assert t.size == 100, f"Expected 100, got {t.size}"
    result = t.data()
    for i in range(100):
        assert result[i] == float(i), f"Index {i}: expected {float(i)}, got {result[i]}"


print("=" * 60)
print("  test_tensor.py - Tensor Creation and Properties")
print("=" * 60)

run_test("zeros creation float32", test_zeros_creation_f32)
run_test("ones creation float32", test_ones_creation_f32)
run_test("zeros creation float64", test_zeros_creation_f64)
run_test("ones creation float64", test_ones_creation_f64)
run_test("values basic", test_values_basic)
run_test("values float64", test_values_f64)
run_test("values negative", test_values_negative)
run_test("values single element", test_values_single)
run_test("single element zeros", test_single_element_zeros)
run_test("large tensor (1000)", test_large_tensor)
run_test("large ones (1000)", test_large_ones)
run_test("properties 3D zeros", test_properties_zeros_3d)
run_test("properties 1D values", test_properties_1d)
run_test("copy same values", test_copy_same_values)
run_test("copy independence", test_copy_independence)
run_test("to dtype f32 -> f64", test_to_dtype_conversion)
run_test("to dtype f64 -> f32", test_to_dtype_f64_to_f32)
run_test("multidimensional zeros", test_multidim_zeros)
run_test("values many elements", test_values_many_elements)

print()
print(f"{passed}/{passed + failed} tests passed")
if failed > 0:
    sys.exit(1)
