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


def test_basic_permute_shape():
    t = rp.zeros([2, 3, 4], dtype='float32', device='cpu')
    permuted = t.permute([2, 0, 1])
    assert permuted.shape == (4, 2, 3), f"Expected (4, 2, 3), got {permuted.shape}"

def test_permute_as_transpose_shape():
    t = rp.ones([2, 3, 4], dtype='float32', device='cpu')
    permuted = t.permute([0, 2, 1])
    assert permuted.shape == (2, 4, 3), f"Expected (2, 4, 3), got {permuted.shape}"

def test_reverse_dims_shape():
    t = rp.zeros([2, 3, 4, 5], dtype='float32', device='cpu')
    permuted = t.permute([3, 2, 1, 0])
    assert permuted.shape == (5, 4, 3, 2), f"Expected (5, 4, 3, 2), got {permuted.shape}"

def test_negative_indices_shape():
    t = rp.ones([2, 3, 4], dtype='float32', device='cpu')
    permuted = t.permute([-1, 0, 1])
    assert permuted.shape == (4, 2, 3), f"Expected (4, 2, 3), got {permuted.shape}"

def test_identity_permute_shape():
    t = rp.zeros([2, 3, 4], dtype='float32', device='cpu')
    permuted = t.permute([0, 1, 2])
    assert permuted.shape == (2, 3, 4), f"Expected (2, 3, 4), got {permuted.shape}"

def test_invalid_permute_fails():
    t = rp.ones([2, 3, 4], dtype='float32', device='cpu')
    try:
        t.permute([0, 1, 1])
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

def test_permute_preserves_size():
    t = rp.zeros([2, 3, 4], dtype='float32', device='cpu')
    permuted = t.permute([2, 0, 1])
    assert permuted.size == t.size, f"Expected size {t.size}, got {permuted.size}"

def test_permute_preserves_ndim():
    t = rp.zeros([2, 3, 4, 5], dtype='float32', device='cpu')
    permuted = t.permute([3, 2, 1, 0])
    assert permuted.ndim == t.ndim, f"Expected ndim {t.ndim}, got {permuted.ndim}"

def test_permute_value_verification_2d():
    t = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float32', device='cpu')
    t2d = t.reshape([2, 3])
    permuted = t2d.permute([1, 0])
    assert permuted.shape == (3, 2), f"Expected (3, 2), got {permuted.shape}"
    contiguous = permuted.reshape([6])
    data = contiguous.data()
    expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    for i in range(6):
        assert abs(data[i] - expected[i]) < 1e-6, f"Value mismatch at {i}: {data[i]} vs {expected[i]}"

def test_permute_value_verification_3d():
    t = rp.values([float(i) for i in range(1, 25)], dtype='float32', device='cpu')
    t3d = t.reshape([2, 3, 4])
    permuted = t3d.permute([2, 0, 1])
    assert permuted.shape == (4, 2, 3), f"Expected (4, 2, 3), got {permuted.shape}"
    contiguous = permuted.reshape([24])
    data = contiguous.data()
    expected = [
        1.0, 5.0, 9.0,
        13.0, 17.0, 21.0,
        2.0, 6.0, 10.0,
        14.0, 18.0, 22.0,
        3.0, 7.0, 11.0,
        15.0, 19.0, 23.0,
        4.0, 8.0, 12.0,
        16.0, 20.0, 24.0,
    ]
    for i in range(24):
        assert abs(data[i] - expected[i]) < 1e-6, f"Value mismatch at {i}: {data[i]} vs {expected[i]}"

def test_4d_permute_shape():
    t = rp.ones([2, 3, 4, 5], dtype='float32', device='cpu')
    permuted = t.permute([0, 3, 1, 2])
    assert permuted.shape == (2, 5, 3, 4), f"Expected (2, 5, 3, 4), got {permuted.shape}"

def test_4d_permute_nhwc_to_nchw():
    t = rp.zeros([8, 32, 32, 3], dtype='float32', device='cpu')
    permuted = t.permute([0, 3, 1, 2])
    assert permuted.shape == (8, 3, 32, 32), f"Expected (8, 3, 32, 32), got {permuted.shape}"

def test_double_permute_restores_shape():
    t = rp.ones([2, 3, 4], dtype='float32', device='cpu')
    permuted = t.permute([2, 0, 1])
    assert permuted.shape == (4, 2, 3), f"Expected (4, 2, 3), got {permuted.shape}"
    inverse = [0, 0, 0]
    dims = [2, 0, 1]
    for i in range(3):
        inverse[dims[i]] = i
    restored = permuted.permute(inverse)
    assert restored.shape == t.shape, f"Expected {t.shape}, got {restored.shape}"

def test_double_permute_restores_values():
    data = [float(i) for i in range(1, 7)]
    t = rp.values(data, dtype='float32', device='cpu')
    t2d = t.reshape([2, 3])
    permuted = t2d.permute([1, 0])
    restored = permuted.permute([1, 0])
    restored_flat = restored.reshape([6])
    restored_data = restored_flat.data()
    for i in range(6):
        assert abs(restored_data[i] - data[i]) < 1e-6, f"Value mismatch at {i}: {restored_data[i]} vs {data[i]}"

def test_permute_float64():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t = rp.values(data, dtype='float64', device='cpu')
    t2d = t.reshape([2, 3])
    assert t2d.dtype == 'float64', f"Expected float64, got {t2d.dtype}"
    permuted = t2d.permute([1, 0])
    assert permuted.shape == (3, 2), f"Expected (3, 2), got {permuted.shape}"
    assert permuted.dtype == 'float64', f"Expected float64, got {permuted.dtype}"
    contiguous = permuted.reshape([6])
    p_data = contiguous.data()
    expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    for i in range(6):
        assert abs(p_data[i] - expected[i]) < 1e-12, f"Value mismatch at {i}: {p_data[i]} vs {expected[i]}"

def test_permute_preserves_device():
    t = rp.ones([2, 3, 4], dtype='float32', device='cpu')
    permuted = t.permute([2, 0, 1])
    assert permuted.device == 'cpu', f"Expected cpu, got {permuted.device}"


print("=" * 60)
print("Testing permute operation")
print("=" * 60)

run_test("basic permute [2,3,4] -> [4,2,3]", test_basic_permute_shape)
run_test("permute as transpose [2,3,4] -> [2,4,3]", test_permute_as_transpose_shape)
run_test("reverse dims [2,3,4,5] -> [5,4,3,2]", test_reverse_dims_shape)
run_test("negative indices [2,3,4] -> [4,2,3]", test_negative_indices_shape)
run_test("identity permute", test_identity_permute_shape)
run_test("invalid permute fails", test_invalid_permute_fails)
run_test("permute preserves size", test_permute_preserves_size)
run_test("permute preserves ndim", test_permute_preserves_ndim)
run_test("2D permute value verification", test_permute_value_verification_2d)
run_test("3D permute value verification", test_permute_value_verification_3d)
run_test("4D permute shape", test_4d_permute_shape)
run_test("4D NHWC to NCHW permute", test_4d_permute_nhwc_to_nchw)
run_test("double permute restores shape", test_double_permute_restores_shape)
run_test("double permute restores values", test_double_permute_restores_values)
run_test("permute float64", test_permute_float64)
run_test("permute preserves device", test_permute_preserves_device)

print("=" * 60)
print(f"{passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"{failed} test(s) FAILED")
    sys.exit(1)
