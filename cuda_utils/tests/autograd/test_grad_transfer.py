import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import numpy as np

def test_grad_transfer_cpu_to_gpu():
    print("\n=== Test Grad Transfer: CPU -> GPU ===")

    x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_square()
    y.backward()

    grad_before = np.array(list(x.grad.data()))

    x_gpu = x.to(device='cuda')

    assert x_gpu.has_grad, "Gradient should be transferred"
    grad_after = np.array(list(x_gpu.grad.data()))

    if np.allclose(grad_before, grad_after, rtol=1e-5):
        print("PASS")
        return True
    else:
        print(f"FAIL: expected {grad_before}, got {grad_after}")
        return False

def test_grad_transfer_gpu_to_cpu():
    print("\n=== Test Grad Transfer: GPU -> CPU ===")

    x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cuda')
    x.requires_grad_()

    y = x.op_square()
    y.backward()

    grad_before = np.array(list(x.grad.data()))

    x_cpu = x.to(device='cpu')

    assert x_cpu.has_grad, "Gradient should be transferred"
    grad_after = np.array(list(x_cpu.grad.data()))

    if np.allclose(grad_before, grad_after, rtol=1e-5):
        print("PASS")
        return True
    else:
        print(f"FAIL: expected {grad_before}, got {grad_after}")
        return False

def test_grad_transfer_dtype_f32_to_f64():
    print("\n=== Test Grad Transfer: f32 -> f64 ===")

    x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_square()
    y.backward()

    grad_before = np.array(list(x.grad.data()))

    x_f64 = x.to(dtype='float64')

    assert x_f64.has_grad, "Gradient should be transferred"
    assert x_f64.grad.dtype == 'float64', "Grad dtype should match tensor dtype"
    grad_after = np.array(list(x_f64.grad.data()))

    if np.allclose(grad_before, grad_after, rtol=1e-5):
        print("PASS")
        return True
    else:
        print(f"FAIL: expected {grad_before}, got {grad_after}")
        return False

def test_grad_transfer_combined():
    print("\n=== Test Grad Transfer: CPU f32 -> GPU f64 ===")

    x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_square()
    y.backward()

    grad_before = np.array(list(x.grad.data()))

    x_new = x.to(device='cuda', dtype='float64')

    assert x_new.has_grad, "Gradient should be transferred"
    assert x_new.grad.dtype == 'float64', "Grad dtype should be float64"
    assert x_new.grad.device == 'cuda:0', "Grad device should be cuda:0"
    grad_after = np.array(list(x_new.grad.data()))

    if np.allclose(grad_before, grad_after, rtol=1e-5):
        print("PASS")
        return True
    else:
        print(f"FAIL: expected {grad_before}, got {grad_after}")
        return False

def test_no_grad_transfer():
    print("\n=== Test No Grad: Transfer preserves NULL grad ===")

    x = rp.values([1.0, 2.0, 3.0], dtype='float32', device='cpu')
    x.requires_grad_()

    x_gpu = x.to(device='cuda')

    assert not x_gpu.has_grad, "Should have no gradient"
    print("PASS")
    return True

def test_grad_transfer_same_device_dtype():
    print("\n=== Test Grad Transfer: Same device/dtype (copy path) ===")

    x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_square()
    y.backward()

    grad_before = np.array(list(x.grad.data()))

    x_copy = x.to(device='cpu', dtype='float32')

    assert x_copy.has_grad, "Gradient should be transferred on copy"
    grad_after = np.array(list(x_copy.grad.data()))

    if np.allclose(grad_before, grad_after, rtol=1e-5):
        print("PASS")
        return True
    else:
        print(f"FAIL: expected {grad_before}, got {grad_after}")
        return False

def test_grad_transfer_values_correct():
    print("\n=== Test Grad Transfer: Values correctness ===")

    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_square()
    y.backward()

    expected_grad = 2.0 * x_data

    x_gpu = x.to(device='cuda')
    actual_grad = np.array(list(x_gpu.grad.data()))

    if np.allclose(actual_grad, expected_grad, rtol=1e-5):
        print("PASS")
        return True
    else:
        print(f"FAIL: expected {expected_grad}, got {actual_grad}")
        return False

def test_grad_transfer_inplace():
    print("\n=== Test Grad Transfer: Inplace ===")

    x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_square()
    y.backward()

    expected_grad = 2.0 * x_data

    x_gpu = x.to(device='cuda', inplace=True)

    assert x_gpu.has_grad, "Gradient should be present after inplace transfer"
    actual_grad = np.array(list(x_gpu.grad.data()))

    if np.allclose(actual_grad, expected_grad, rtol=1e-5):
        print("PASS")
        return True
    else:
        print(f"FAIL: expected {expected_grad}, got {actual_grad}")
        return False


if __name__ == '__main__':
    print("Testing Gradient Transfer in tensor_to()")
    print("=" * 50)

    results = []
    results.append(test_grad_transfer_cpu_to_gpu())
    results.append(test_grad_transfer_gpu_to_cpu())
    results.append(test_grad_transfer_dtype_f32_to_f64())
    results.append(test_grad_transfer_combined())
    results.append(test_no_grad_transfer())
    results.append(test_grad_transfer_same_device_dtype())
    results.append(test_grad_transfer_values_correct())
    results.append(test_grad_transfer_inplace())

    print("\n" + "=" * 50)
    print(f"Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED")
        sys.exit(1)
