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


def test_squeeze_dim1():
    t = rp.ones([3, 1, 5, 1], dtype='float32', device='cpu')
    result = t.squeeze(1)
    assert result.shape == (3, 5, 1), f"Expected (3, 5, 1), got {result.shape}"
    assert result.size == t.size, f"Expected size {t.size}, got {result.size}"

def test_squeeze_dim3():
    t = rp.ones([3, 1, 5, 1], dtype='float32', device='cpu')
    result = t.squeeze(3)
    assert result.shape == (3, 1, 5), f"Expected (3, 1, 5), got {result.shape}"

def test_squeeze_all():
    t = rp.ones([3, 1, 5, 1], dtype='float32', device='cpu')
    result = t.squeeze(-1)
    assert result.shape == (3, 5), f"Expected (3, 5), got {result.shape}"
    assert result.ndim == 2, f"Expected ndim 2, got {result.ndim}"

def test_squeeze_non_size1_fails():
    t = rp.ones([3, 1, 5, 1], dtype='float32', device='cpu')
    try:
        t.squeeze(0)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

def test_unsqueeze_dim0():
    t = rp.ones([3, 5], dtype='float32', device='cpu')
    result = t.unsqueeze(0)
    assert result.shape == (1, 3, 5), f"Expected (1, 3, 5), got {result.shape}"

def test_unsqueeze_dim1():
    t = rp.ones([3, 5], dtype='float32', device='cpu')
    result = t.unsqueeze(1)
    assert result.shape == (3, 1, 5), f"Expected (3, 1, 5), got {result.shape}"

def test_unsqueeze_neg1():
    t = rp.ones([3, 5], dtype='float32', device='cpu')
    result = t.unsqueeze(-1)
    assert result.shape == (3, 5, 1), f"Expected (3, 5, 1), got {result.shape}"

def test_round_trip_shape():
    t = rp.ones([2, 3, 4], dtype='float32', device='cpu')
    unsqueezed = t.unsqueeze(1)
    assert unsqueezed.shape == (2, 1, 3, 4), f"Expected (2, 1, 3, 4), got {unsqueezed.shape}"
    squeezed = unsqueezed.squeeze(1)
    assert squeezed.shape == t.shape, f"Expected {t.shape}, got {squeezed.shape}"

def test_squeeze_value_preservation():
    t = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float32', device='cpu')
    t2 = t.reshape([2, 1, 3])
    squeezed = t2.squeeze(1)
    assert squeezed.shape == (2, 3), f"Expected (2, 3), got {squeezed.shape}"
    original_data = t.data()
    squeezed_data = squeezed.data()
    assert len(original_data) == len(squeezed_data), f"Data length mismatch: {len(original_data)} vs {len(squeezed_data)}"
    for i in range(len(original_data)):
        assert abs(original_data[i] - squeezed_data[i]) < 1e-6, f"Value mismatch at {i}: {original_data[i]} vs {squeezed_data[i]}"

def test_unsqueeze_squeeze_round_trip_values():
    t = rp.values([10.0, 20.0, 30.0], dtype='float32', device='cpu')
    unsqueezed = t.unsqueeze(0)
    assert unsqueezed.shape == (1, 3), f"Expected (1, 3), got {unsqueezed.shape}"
    squeezed = unsqueezed.squeeze(0)
    assert squeezed.shape == (3,), f"Expected (3,), got {squeezed.shape}"
    original_data = t.data()
    result_data = squeezed.data()
    for i in range(len(original_data)):
        assert abs(original_data[i] - result_data[i]) < 1e-6, f"Value mismatch at {i}: {original_data[i]} vs {result_data[i]}"

def test_squeeze_all_size1_dims():
    t = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype='float32', device='cpu')
    t5d = t.reshape([1, 3, 1, 4, 1])
    assert t5d.shape == (1, 3, 1, 4, 1), f"Expected (1, 3, 1, 4, 1), got {t5d.shape}"
    result = t5d.squeeze(0)
    assert result.shape == (3, 1, 4, 1), f"After squeeze(0): expected (3, 1, 4, 1), got {result.shape}"
    result = result.squeeze(1)
    assert result.shape == (3, 4, 1), f"After squeeze(1): expected (3, 4, 1), got {result.shape}"
    result = result.squeeze(2)
    assert result.shape == (3, 4), f"After squeeze(2): expected (3, 4), got {result.shape}"
    result_data = result.data()
    original_data = t.data()
    for i in range(len(original_data)):
        assert abs(original_data[i] - result_data[i]) < 1e-6, f"Value mismatch at {i}"

def test_squeeze_all_at_once():
    t = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype='float32', device='cpu')
    t5d = t.reshape([1, 3, 1, 4, 1])
    result = t5d.squeeze(-1)
    assert result.shape == (3, 4), f"Expected (3, 4), got {result.shape}"
    result_data = result.data()
    original_data = t.data()
    for i in range(len(original_data)):
        assert abs(original_data[i] - result_data[i]) < 1e-6, f"Value mismatch at {i}"

def test_squeeze_float64():
    t = rp.values([1.0, 2.0, 3.0], dtype='float64', device='cpu')
    t2 = t.reshape([1, 3, 1])
    assert t2.dtype == 'float64', f"Expected float64, got {t2.dtype}"
    squeezed = t2.squeeze(0)
    assert squeezed.shape == (3, 1), f"Expected (3, 1), got {squeezed.shape}"
    assert squeezed.dtype == 'float64', f"Expected float64, got {squeezed.dtype}"
    squeezed_data = squeezed.data()
    original_data = t.data()
    for i in range(len(original_data)):
        assert abs(original_data[i] - squeezed_data[i]) < 1e-12, f"Value mismatch at {i}"

def test_unsqueeze_float64():
    t = rp.values([4.0, 5.0, 6.0], dtype='float64', device='cpu')
    result = t.unsqueeze(1)
    assert result.shape == (3, 1), f"Expected (3, 1), got {result.shape}"
    assert result.dtype == 'float64', f"Expected float64, got {result.dtype}"
    result_data = result.data()
    original_data = t.data()
    for i in range(len(original_data)):
        assert abs(original_data[i] - result_data[i]) < 1e-12, f"Value mismatch at {i}"

def test_squeeze_preserves_device():
    t = rp.ones([2, 1, 3], dtype='float32', device='cpu')
    result = t.squeeze(1)
    assert result.device == 'cpu', f"Expected cpu, got {result.device}"

def test_unsqueeze_multiple():
    t = rp.ones([4], dtype='float32', device='cpu')
    r1 = t.unsqueeze(0)
    assert r1.shape == (1, 4), f"Expected (1, 4), got {r1.shape}"
    r2 = r1.unsqueeze(2)
    assert r2.shape == (1, 4, 1), f"Expected (1, 4, 1), got {r2.shape}"
    r3 = r2.unsqueeze(0)
    assert r3.shape == (1, 1, 4, 1), f"Expected (1, 1, 4, 1), got {r3.shape}"


print("=" * 60)
print("Testing squeeze / unsqueeze operations")
print("=" * 60)

run_test("squeeze dim 1 from [3,1,5,1]", test_squeeze_dim1)
run_test("squeeze dim 3 from [3,1,5,1]", test_squeeze_dim3)
run_test("squeeze all size-1 dims with -1", test_squeeze_all)
run_test("squeeze non-size-1 dim fails", test_squeeze_non_size1_fails)
run_test("unsqueeze dim 0", test_unsqueeze_dim0)
run_test("unsqueeze dim 1", test_unsqueeze_dim1)
run_test("unsqueeze dim -1", test_unsqueeze_neg1)
run_test("unsqueeze then squeeze round-trip shape", test_round_trip_shape)
run_test("squeeze value preservation", test_squeeze_value_preservation)
run_test("unsqueeze/squeeze round-trip values", test_unsqueeze_squeeze_round_trip_values)
run_test("squeeze all size-1 dims from [1,3,1,4,1] one by one", test_squeeze_all_size1_dims)
run_test("squeeze all size-1 dims at once with -1", test_squeeze_all_at_once)
run_test("squeeze float64", test_squeeze_float64)
run_test("unsqueeze float64", test_unsqueeze_float64)
run_test("squeeze preserves device", test_squeeze_preserves_device)
run_test("multiple unsqueeze operations", test_unsqueeze_multiple)

print("=" * 60)
print(f"{passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"{failed} test(s) FAILED")
    sys.exit(1)
