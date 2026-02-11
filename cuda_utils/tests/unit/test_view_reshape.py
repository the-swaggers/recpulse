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


def test_view_contiguous():
    t = rp.ones([3, 4])
    v = t.view([12])
    assert v.shape == (12,), f"Expected (12,), got {v.shape}"
    assert v.size == 12, f"Expected 12, got {v.size}"

def test_reshape_contiguous():
    t = rp.ones([3, 4])
    r = t.reshape([12])
    assert r.shape == (12,), f"Expected (12,), got {r.shape}"

def test_view_fails_noncontiguous():
    t = rp.ones([3, 4])
    t_t = t.transpose(0, 1)
    try:
        t_t.view([12])
        assert False, "view should fail on non-contiguous tensor"
    except RuntimeError:
        pass

def test_reshape_succeeds_noncontiguous():
    t = rp.ones([3, 4])
    t_t = t.transpose(0, 1)
    r = t_t.reshape([12])
    assert r.shape == (12,), f"Expected (12,), got {r.shape}"

def test_view_infer_dim():
    t = rp.ones([2, 3, 4])
    v = t.view([2, -1])
    assert v.shape == (2, 12), f"Expected (2, 12), got {v.shape}"

def test_view_values_preserved():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t = rp.values(data)
    v = t.view([2, 3])
    assert v.shape == (2, 3), f"Expected (2, 3), got {v.shape}"
    v_data = v.data()
    for i in range(6):
        assert v_data[i] == data[i], f"Index {i}: expected {data[i]}, got {v_data[i]}"

def test_view_values_2d_to_3d():
    data = [float(i) for i in range(24)]
    t_1d = rp.values(data)
    t = t_1d.reshape([4, 6])
    v = t.view([2, 2, 6])
    assert v.shape == (2, 2, 6), f"Expected (2, 2, 6), got {v.shape}"
    v_data = v.data()
    for i in range(24):
        assert v_data[i] == data[i], f"Index {i}: expected {data[i]}, got {v_data[i]}"

def test_reshape_chain():
    data = [float(i) for i in range(12)]
    t = rp.values(data)
    r1 = t.reshape([3, 4])
    r2 = r1.reshape([2, 6])
    r3 = r2.reshape([12])
    r3_data = r3.data()
    for i in range(12):
        assert r3_data[i] == data[i], f"Index {i}: expected {data[i]}, got {r3_data[i]}"

def test_multiple_reshape_chains():
    data = [float(i) for i in range(24)]
    t = rp.values(data)
    r1 = t.reshape([2, 12])
    r2 = r1.reshape([2, 3, 4])
    r3 = r2.reshape([6, 4])
    r = r3.reshape([24])
    r_data = r.data()
    for i in range(24):
        assert r_data[i] == data[i], f"Index {i}: expected {data[i]}, got {r_data[i]}"

def test_reshape_contiguous_is_zerocopy():
    t = rp.ones([2, 3])
    r = t.reshape([6])
    assert r.shape == (6,), f"Expected (6,), got {r.shape}"
    assert r.data() == t.data(), f"Data should match"


print("=" * 60)
print("  test_view_reshape.py - View vs Reshape Operations")
print("=" * 60)

run_test("view on contiguous", test_view_contiguous)
run_test("reshape on contiguous", test_reshape_contiguous)
run_test("view fails on non-contiguous", test_view_fails_noncontiguous)
run_test("reshape succeeds on non-contiguous", test_reshape_succeeds_noncontiguous)
run_test("view infer dim (-1)", test_view_infer_dim)
run_test("view preserves values", test_view_values_preserved)
run_test("view 2D to 3D values", test_view_values_2d_to_3d)
run_test("reshape chain preserves values", test_reshape_chain)
run_test("multiple reshape chains", test_multiple_reshape_chains)
run_test("reshape contiguous zero-copy", test_reshape_contiguous_is_zerocopy)

print()
print(f"{passed}/{passed + failed} tests passed")
if failed > 0:
    sys.exit(1)
