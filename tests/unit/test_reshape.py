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


def test_basic_reshape():
    t = rp.values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    r = t.reshape([2, 6])
    assert r.shape == (2, 6), f"Expected (2, 6), got {r.shape}"
    assert r.size == 12, f"Expected 12, got {r.size}"

def test_reshape_chain():
    t = rp.values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    r = t.reshape([2, 6]).reshape([3, 4])
    assert r.shape == (3, 4), f"Expected (3, 4), got {r.shape}"
    assert r.size == 12, f"Expected 12, got {r.size}"

def test_reshape_infer_dim():
    t = rp.zeros([24])
    r1 = t.reshape([2, -1])
    assert r1.shape == (2, 12), f"Expected (2, 12), got {r1.shape}"
    r2 = t.reshape([-1, 6])
    assert r2.shape == (4, 6), f"Expected (4, 6), got {r2.shape}"
    r3 = t.reshape([2, 3, -1])
    assert r3.shape == (2, 3, 4), f"Expected (2, 3, 4), got {r3.shape}"

def test_reshape_multidim():
    t = rp.zeros([2, 3, 4])
    r1 = t.reshape([6, 4])
    assert r1.shape == (6, 4), f"Expected (6, 4), got {r1.shape}"
    r2 = t.reshape([24])
    assert r2.shape == (24,), f"Expected (24,), got {r2.shape}"
    r3 = t.reshape([2, 12])
    assert r3.shape == (2, 12), f"Expected (2, 12), got {r3.shape}"

def test_reshape_invalid():
    t = rp.zeros([12])
    try:
        t.reshape([3, 5])
        assert False, "Should have raised an error for invalid reshape"
    except Exception:
        pass

def test_reshape_preserves_values():
    data = [float(i) for i in range(1, 13)]
    t = rp.values(data)
    r = t.reshape([3, 4])
    assert r.shape == (3, 4), f"Expected (3, 4), got {r.shape}"
    r_data = r.data()
    for i in range(12):
        assert r_data[i] == data[i], f"Index {i}: expected {data[i]}, got {r_data[i]}"

def test_reshape_values_roundtrip():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t = rp.values(data)
    r2d = t.reshape([2, 3])
    r1d = r2d.reshape([6])
    r_data = r1d.data()
    for i in range(6):
        assert r_data[i] == data[i], f"Index {i}: expected {data[i]}, got {r_data[i]}"

def test_flatten_to_1d():
    t_1d = rp.values([float(i) for i in range(24)])
    t = t_1d.reshape([2, 3, 4])
    flat = t.reshape([24])
    assert flat.shape == (24,), f"Expected (24,), got {flat.shape}"
    assert flat.ndim == 1, f"Expected 1, got {flat.ndim}"
    for i in range(24):
        assert flat.data()[i] == float(i), f"Index {i}: expected {float(i)}, got {flat.data()[i]}"

def test_flatten_and_back():
    data = [float(i) for i in range(12)]
    t_1d = rp.values(data)
    t = t_1d.reshape([3, 4])
    flat = t.reshape([12])
    back = flat.reshape([3, 4])
    assert back.shape == (3, 4), f"Expected (3, 4), got {back.shape}"
    for i in range(12):
        assert back.data()[i] == data[i], f"Index {i}: expected {data[i]}, got {back.data()[i]}"

def test_4d_to_2d():
    t = rp.zeros([2, 3, 4, 5])
    assert t.size == 120, f"Expected 120, got {t.size}"
    r = t.reshape([6, 20])
    assert r.shape == (6, 20), f"Expected (6, 20), got {r.shape}"
    assert r.size == 120, f"Expected 120, got {r.size}"

def test_2d_to_4d():
    data = [float(i) for i in range(24)]
    t_1d = rp.values(data)
    t = t_1d.reshape([4, 6])
    r = t.reshape([2, 2, 2, 3])
    assert r.shape == (2, 2, 2, 3), f"Expected (2, 2, 2, 3), got {r.shape}"
    assert r.size == 24, f"Expected 24, got {r.size}"
    for i in range(24):
        assert r.data()[i] == data[i], f"Index {i}: expected {data[i]}, got {r.data()[i]}"

def test_view_semantics():
    t = rp.values([1.0, 2.0, 3.0, 4.0])
    r = t.reshape([2, 2])
    assert r.shape == (2, 2), f"Expected (2, 2), got {r.shape}"
    assert r.data() == t.data(), f"Data should match original"


print("=" * 60)
print("  test_reshape.py - Reshape Operations")
print("=" * 60)

run_test("basic reshape", test_basic_reshape)
run_test("reshape chain", test_reshape_chain)
run_test("reshape infer dim (-1)", test_reshape_infer_dim)
run_test("reshape multidim", test_reshape_multidim)
run_test("reshape invalid", test_reshape_invalid)
run_test("reshape preserves values", test_reshape_preserves_values)
run_test("reshape values roundtrip", test_reshape_values_roundtrip)
run_test("flatten to 1D", test_flatten_to_1d)
run_test("flatten and back", test_flatten_and_back)
run_test("4D to 2D", test_4d_to_2d)
run_test("2D to 4D", test_2d_to_4d)
run_test("view semantics", test_view_semantics)

print()
print(f"{passed}/{passed + failed} tests passed")
if failed > 0:
    sys.exit(1)
