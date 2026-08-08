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


def test_2d_transpose_shape():
    t = rp.zeros([2, 3])
    t_T = t.transpose(0, 1)
    assert t_T.shape == (3, 2), f"Expected (3, 2), got {t_T.shape}"

def test_T_property():
    t = rp.zeros([2, 3])
    assert t.T.shape == (3, 2), f"Expected (3, 2), got {t.T.shape}"

def test_3d_transpose():
    t = rp.ones([2, 3, 4])
    t_t = t.transpose(0, 2)
    assert t_t.shape == (4, 3, 2), f"Expected (4, 3, 2), got {t_t.shape}"

def test_negative_indices():
    t = rp.ones([2, 3, 4])
    t_neg = t.transpose(-1, 0)
    assert t_neg.shape == (4, 3, 2), f"Expected (4, 3, 2), got {t_neg.shape}"

def test_same_dim_noop():
    t = rp.zeros([2, 3])
    t_same = t.transpose(0, 0)
    assert t_same.shape == (2, 3), f"Expected (2, 3), got {t_same.shape}"

def test_transpose_values_2d():
    t_1d = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    t = t_1d.reshape([2, 3])
    t_T = t.transpose(0, 1)
    assert t_T.shape == (3, 2), f"Expected (3, 2), got {t_T.shape}"
    t_contig = t_T.reshape([6])
    data = t_contig.data()
    assert data[0] == 1.0, f"Expected 1.0 at [0,0], got {data[0]}"
    assert data[1] == 4.0, f"Expected 4.0 at [0,1], got {data[1]}"
    assert data[2] == 2.0, f"Expected 2.0 at [1,0], got {data[2]}"
    assert data[3] == 5.0, f"Expected 5.0 at [1,1], got {data[3]}"
    assert data[4] == 3.0, f"Expected 3.0 at [2,0], got {data[4]}"
    assert data[5] == 6.0, f"Expected 6.0 at [2,1], got {data[5]}"

def test_double_transpose_shape():
    t = rp.zeros([2, 3])
    t_double = t.transpose(0, 1).transpose(0, 1)
    assert t_double.shape == (2, 3), f"Expected (2, 3), got {t_double.shape}"

def test_double_transpose_values():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t_1d = rp.values(data)
    t = t_1d.reshape([2, 3])
    t_T = t.transpose(0, 1)
    t_double = t_T.transpose(0, 1)
    t_contig = t_double.reshape([6])
    result = t_contig.data()
    for i in range(6):
        assert result[i] == data[i], f"Index {i}: expected {data[i]}, got {result[i]}"

def test_4d_transpose():
    t = rp.zeros([2, 3, 4, 5])
    t_t = t.transpose(1, 3)
    assert t_t.shape == (2, 5, 4, 3), f"Expected (2, 5, 4, 3), got {t_t.shape}"

def test_4d_transpose_02():
    t = rp.zeros([2, 3, 4, 5])
    t_t = t.transpose(0, 2)
    assert t_t.shape == (4, 3, 2, 5), f"Expected (4, 3, 2, 5), got {t_t.shape}"

def test_transpose_ndim_preserved():
    t = rp.zeros([2, 3, 4])
    t_t = t.transpose(0, 2)
    assert t_t.ndim == 3, f"Expected 3, got {t_t.ndim}"
    assert t_t.size == 24, f"Expected 24, got {t_t.size}"

def test_transpose_is_view():
    t_1d = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    t = t_1d.reshape([2, 3])
    t_T = t.transpose(0, 1)
    assert t_T.size == t.size, f"Size mismatch: {t_T.size} != {t.size}"

def test_T_property_values():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t_1d = rp.values(data)
    t = t_1d.reshape([2, 3])
    t_T = t.T
    assert t_T.shape == (3, 2), f"Expected (3, 2), got {t_T.shape}"
    t_contig = t_T.reshape([6])
    result = t_contig.data()
    assert result[0] == 1.0, f"Expected 1.0, got {result[0]}"
    assert result[1] == 4.0, f"Expected 4.0, got {result[1]}"


print("=" * 60)
print("  test_transpose.py - Transpose Operations")
print("=" * 60)

run_test("2D transpose shape", test_2d_transpose_shape)
run_test(".T property", test_T_property)
run_test("3D transpose (0, 2)", test_3d_transpose)
run_test("negative indices", test_negative_indices)
run_test("same dim no-op", test_same_dim_noop)
run_test("transpose values 2D", test_transpose_values_2d)
run_test("double transpose shape", test_double_transpose_shape)
run_test("double transpose values", test_double_transpose_values)
run_test("4D transpose (1, 3)", test_4d_transpose)
run_test("4D transpose (0, 2)", test_4d_transpose_02)
run_test("transpose preserves ndim/size", test_transpose_ndim_preserved)
run_test("transpose is view", test_transpose_is_view)
run_test(".T property values", test_T_property_values)

print()
print(f"{passed}/{passed + failed} tests passed")
if failed > 0:
    sys.exit(1)
