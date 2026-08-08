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


def test_uneven_chunk_dim0():
    t = rp.zeros([10, 5], dtype='float32', device='cpu')
    chunks = t.chunk(3, 0)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    assert chunks[0].shape == (4, 5), f"Expected (4, 5), got {chunks[0].shape}"
    assert chunks[1].shape == (4, 5), f"Expected (4, 5), got {chunks[1].shape}"
    assert chunks[2].shape == (2, 5), f"Expected (2, 5), got {chunks[2].shape}"

def test_exact_chunk_dim0():
    t = rp.ones([12, 3], dtype='float32', device='cpu')
    chunks = t.chunk(3, 0)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    for i, c in enumerate(chunks):
        assert c.shape == (4, 3), f"Chunk {i}: expected (4, 3), got {c.shape}"

def test_more_chunks_than_size():
    t = rp.zeros([5, 3], dtype='float32', device='cpu')
    chunks = t.chunk(10, 0)
    assert len(chunks) == 5, f"Expected 5 chunks (clamped), got {len(chunks)}"
    for i, c in enumerate(chunks):
        assert c.shape == (1, 3), f"Chunk {i}: expected (1, 3), got {c.shape}"

def test_chunk_along_dim1():
    t = rp.ones([3, 10], dtype='float32', device='cpu')
    chunks = t.chunk(4, 1)
    assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}"
    assert chunks[0].shape == (3, 3), f"Expected (3, 3), got {chunks[0].shape}"
    assert chunks[1].shape == (3, 3), f"Expected (3, 3), got {chunks[1].shape}"
    assert chunks[2].shape == (3, 3), f"Expected (3, 3), got {chunks[2].shape}"
    assert chunks[3].shape == (3, 1), f"Expected (3, 1), got {chunks[3].shape}"

def test_single_chunk():
    t = rp.zeros([10, 5], dtype='float32', device='cpu')
    chunks = t.chunk(1, 0)
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert chunks[0].shape == (10, 5), f"Expected (10, 5), got {chunks[0].shape}"

def test_chunk_negative_dim():
    t = rp.ones([3, 10], dtype='float32', device='cpu')
    chunks = t.chunk(3, -1)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    total_along_dim = sum(c.shape[1] for c in chunks)
    assert total_along_dim == 10, f"Expected total dim-1 size 10, got {total_along_dim}"

def test_chunk_default_dim():
    t = rp.zeros([10, 5], dtype='float32', device='cpu')
    chunks = t.chunk(3)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    total_along_dim = sum(c.shape[0] for c in chunks)
    assert total_along_dim == 10, f"Expected total dim-0 size 10, got {total_along_dim}"

def test_chunk_sizes_sum_to_original_dim0():
    t = rp.ones([17, 4], dtype='float32', device='cpu')
    chunks = t.chunk(5, 0)
    total = sum(c.shape[0] for c in chunks)
    assert total == 17, f"Expected chunk dim-0 sizes to sum to 17, got {total}"
    for c in chunks:
        assert c.shape[1] == 4, f"Expected dim-1 = 4, got {c.shape[1]}"

def test_chunk_sizes_sum_to_original_dim1():
    t = rp.zeros([3, 13], dtype='float32', device='cpu')
    chunks = t.chunk(4, 1)
    total = sum(c.shape[1] for c in chunks)
    assert total == 13, f"Expected chunk dim-1 sizes to sum to 13, got {total}"
    for c in chunks:
        assert c.shape[0] == 3, f"Expected dim-0 = 3, got {c.shape[0]}"

def test_chunk_value_verification_dim0():
    data = [float(i) for i in range(12)]
    t = rp.values(data, dtype='float32', device='cpu')
    t2d = t.reshape([4, 3])
    chunks = t2d.chunk(2, 0)
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    c0_data = chunks[0].data()
    c1_data = chunks[1].data()
    expected_c0 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    expected_c1 = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    for i in range(6):
        assert abs(c0_data[i] - expected_c0[i]) < 1e-6, f"Chunk 0 mismatch at {i}: {c0_data[i]} vs {expected_c0[i]}"
        assert abs(c1_data[i] - expected_c1[i]) < 1e-6, f"Chunk 1 mismatch at {i}: {c1_data[i]} vs {expected_c1[i]}"

def test_chunk_value_verification_3_chunks():
    data = [float(i) for i in range(18)]
    t = rp.values(data, dtype='float32', device='cpu')
    t2d = t.reshape([6, 3])
    chunks = t2d.chunk(3, 0)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    for ci, chunk in enumerate(chunks):
        chunk_data = chunk.data()
        offset = ci * 6
        for i in range(6):
            expected = float(offset + i)
            assert abs(chunk_data[i] - expected) < 1e-6, f"Chunk {ci} mismatch at {i}: {chunk_data[i]} vs {expected}"

def test_chunk_float64():
    data = [float(i) for i in range(12)]
    t = rp.values(data, dtype='float64', device='cpu')
    t2d = t.reshape([4, 3])
    assert t2d.dtype == 'float64', f"Expected float64, got {t2d.dtype}"
    chunks = t2d.chunk(2, 0)
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    for c in chunks:
        assert c.dtype == 'float64', f"Expected float64, got {c.dtype}"
    c0_data = chunks[0].data()
    expected_c0 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    for i in range(6):
        assert abs(c0_data[i] - expected_c0[i]) < 1e-12, f"Chunk 0 mismatch at {i}"

def test_chunk_1d_tensor():
    data = [float(i) for i in range(10)]
    t = rp.values(data, dtype='float32', device='cpu')
    chunks = t.chunk(3)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    total = sum(c.size for c in chunks)
    assert total == 10, f"Expected total size 10, got {total}"
    c0_data = chunks[0].data()
    for i in range(len(c0_data)):
        assert abs(c0_data[i] - float(i)) < 1e-6, f"Chunk 0 mismatch at {i}"

def test_chunk_preserves_device():
    t = rp.ones([6, 3], dtype='float32', device='cpu')
    chunks = t.chunk(2, 0)
    for c in chunks:
        assert c.device == 'cpu', f"Expected cpu, got {c.device}"

def test_chunk_preserves_ndim():
    t = rp.zeros([6, 4, 3], dtype='float32', device='cpu')
    chunks = t.chunk(2, 0)
    for c in chunks:
        assert c.ndim == 3, f"Expected ndim 3, got {c.ndim}"


print("=" * 60)
print("Testing chunk operation")
print("=" * 60)

run_test("uneven chunk along dim 0", test_uneven_chunk_dim0)
run_test("exact chunk along dim 0", test_exact_chunk_dim0)
run_test("more chunks than size (clamped)", test_more_chunks_than_size)
run_test("chunk along dim 1", test_chunk_along_dim1)
run_test("single chunk", test_single_chunk)
run_test("chunk negative dim", test_chunk_negative_dim)
run_test("chunk default dim", test_chunk_default_dim)
run_test("chunk sizes sum to original dim 0", test_chunk_sizes_sum_to_original_dim0)
run_test("chunk sizes sum to original dim 1", test_chunk_sizes_sum_to_original_dim1)
run_test("chunk value verification dim 0 (2 chunks)", test_chunk_value_verification_dim0)
run_test("chunk value verification dim 0 (3 chunks)", test_chunk_value_verification_3_chunks)
run_test("chunk float64", test_chunk_float64)
run_test("chunk 1D tensor", test_chunk_1d_tensor)
run_test("chunk preserves device", test_chunk_preserves_device)
run_test("chunk preserves ndim", test_chunk_preserves_ndim)

print("=" * 60)
print(f"{passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"{failed} test(s) FAILED")
    sys.exit(1)
