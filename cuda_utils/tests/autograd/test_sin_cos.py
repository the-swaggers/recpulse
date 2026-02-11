import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import numpy as np

def test_sin_backward_cpu():
    """Test sin backward on CPU"""
    print("\n=== Test Sin Gradient (CPU) ===")

    x_data = np.array([0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_sin()
    y.backward()

    expected_output = np.sin(x_data)
    expected_grad = np.cos(x_data)

    actual_output = np.array(list(y.data()))
    actual_grad = np.array(list(x.grad.data()))

    print(f"Input: {x_data}")
    print(f"Output sin(x) expected: {expected_output}")
    print(f"Output sin(x) actual:   {actual_output}")
    print(f"Gradient cos(x) expected: {expected_grad}")
    print(f"Gradient cos(x) actual:   {actual_grad}")

    output_ok = np.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
    grad_ok = np.allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)

    if output_ok and grad_ok:
        print("PASSED")
        return True
    else:
        print(f"FAILED - output_ok={output_ok}, grad_ok={grad_ok}")
        return False

def test_cos_backward_cpu():
    """Test cos backward on CPU"""
    print("\n=== Test Cos Gradient (CPU) ===")

    x_data = np.array([0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_cos()
    y.backward()

    expected_output = np.cos(x_data)
    expected_grad = -np.sin(x_data)

    actual_output = np.array(list(y.data()))
    actual_grad = np.array(list(x.grad.data()))

    print(f"Input: {x_data}")
    print(f"Output cos(x) expected: {expected_output}")
    print(f"Output cos(x) actual:   {actual_output}")
    print(f"Gradient -sin(x) expected: {expected_grad}")
    print(f"Gradient -sin(x) actual:   {actual_grad}")

    output_ok = np.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
    grad_ok = np.allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)

    if output_ok and grad_ok:
        print("PASSED")
        return True
    else:
        print(f"FAILED - output_ok={output_ok}, grad_ok={grad_ok}")
        return False

def test_sin_backward_gpu():
    """Test sin backward on GPU"""
    print("\n=== Test Sin Gradient (GPU) ===")

    x_data = np.array([0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cuda')
    x.requires_grad_()

    y = x.op_sin()
    y.backward()

    expected_grad = np.cos(x_data)
    actual_grad = np.array(list(x.grad.data()))

    print(f"Gradient cos(x) expected: {expected_grad}")
    print(f"Gradient cos(x) actual:   {actual_grad}")

    grad_ok = np.allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)

    if grad_ok:
        print("PASSED")
        return True
    else:
        print(f"FAILED")
        return False

def test_sin_chain_rule():
    """Test chain rule: z = sin(sin(x)), dz/dx = cos(sin(x)) * cos(x)"""
    print("\n=== Test Sin Chain Rule (CPU) ===")

    x_data = np.array([0.1, 0.5, 1.0, 1.5], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_sin()
    z = y.op_sin()
    z.backward()

    expected_grad = np.cos(np.sin(x_data)) * np.cos(x_data)
    actual_grad = np.array(list(x.grad.data()))

    print(f"Input x: {x_data}")
    print(f"y = sin(x): {np.sin(x_data)}")
    print(f"z = sin(sin(x)): {np.sin(np.sin(x_data))}")
    print(f"Expected gradient: {expected_grad}")
    print(f"Actual gradient:   {actual_grad}")

    grad_ok = np.allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)

    if grad_ok:
        print("PASSED")
        return True
    else:
        print(f"FAILED")
        return False

def test_sin_cos_mixed():
    """Test mixed sin/cos: z = sin(x) * cos(x), dz/dx = cos^2(x) - sin^2(x) = cos(2x)"""
    print("\n=== Test Sin*Cos Mixed (CPU) ===")

    x_data = np.array([0.1, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y_sin = x.op_sin()
    y_cos = x.op_cos()
    z = y_sin.op_mul(y_cos)
    z.backward()

    expected_grad = np.cos(2 * x_data)
    actual_grad = np.array(list(x.grad.data()))

    print(f"Input x: {x_data}")
    print(f"z = sin(x)*cos(x): {np.sin(x_data) * np.cos(x_data)}")
    print(f"Expected gradient (cos(2x)): {expected_grad}")
    print(f"Actual gradient:             {actual_grad}")

    grad_ok = np.allclose(actual_grad, expected_grad, rtol=1e-4, atol=1e-4)

    if grad_ok:
        print("PASSED")
        return True
    else:
        print(f"FAILED - max error: {np.max(np.abs(actual_grad - expected_grad))}")
        return False

def test_tan_backward_cpu():
    print("\n=== Test Tan Gradient (CPU) ===")

    x_data = np.array([0.0, 0.3, 0.5, -0.3, 1.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_tan()
    y.backward()

    expected_grad = 1.0 / (np.cos(x_data) ** 2)
    actual_grad = np.array(list(x.grad.data()))

    grad_ok = np.allclose(actual_grad, expected_grad, rtol=1e-4, atol=1e-4)

    if grad_ok:
        print("PASSED")
        return True
    else:
        print(f"FAILED - max error: {np.max(np.abs(actual_grad - expected_grad))}")
        return False


def test_sin_cos_chain():
    print("\n=== Test sin(cos(x)) Chain Backward (CPU) ===")

    x_data = np.array([0.1, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    cos_x = x.op_cos()
    sin_cos_x = cos_x.op_sin()
    sin_cos_x.backward()

    expected_grad = np.cos(np.cos(x_data)) * (-np.sin(x_data))
    actual_grad = np.array(list(x.grad.data()))

    grad_ok = np.allclose(actual_grad, expected_grad, rtol=1e-4, atol=1e-4)

    if grad_ok:
        print("PASSED")
        return True
    else:
        print(f"FAILED - expected: {expected_grad}, actual: {actual_grad}")
        return False


if __name__ == '__main__':
    print("Testing Sin/Cos/Tan Backward Pass Implementation")
    print("=" * 50)

    results = []
    results.append(test_sin_backward_cpu())
    results.append(test_cos_backward_cpu())

    try:
        results.append(test_sin_backward_gpu())
    except Exception as e:
        print(f"\nGPU test skipped: {e}")

    results.append(test_sin_chain_rule())
    results.append(test_sin_cos_mixed())
    results.append(test_tan_backward_cpu())
    results.append(test_sin_cos_chain())

    print("\n" + "=" * 50)
    print(f"Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED")
        sys.exit(1)
