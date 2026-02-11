import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import numpy as np

def test_square_grad_cpu():
    """Test square backward on CPU"""
    print("\n=== Test Square Gradient (CPU) ===")

    # Create input tensor (1D to avoid reshape bug)
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    # Forward pass
    y = x.op_square()

    # Backward pass
    y.backward()

    # Expected gradient: dy/dx = 2x
    expected_grad = 2.0 * x_data

    # Get actual gradient
    actual_grad = np.array(list(x.grad.data()))

    print(f"Input: {x_data}")
    print(f"Output (x²): {np.array(list(y.data()))}")
    print(f"Expected gradient (2x): {expected_grad}")
    print(f"Actual gradient: {actual_grad}")

    # Check if gradients match
    if np.allclose(actual_grad, expected_grad, rtol=1e-5):
        print("✓ CPU gradient test PASSED")
        return True
    else:
        print("✗ CPU gradient test FAILED")
        print(f"Max error: {np.max(np.abs(actual_grad - expected_grad))}")
        return False

def test_square_grad_gpu():
    """Test square backward on GPU"""
    print("\n=== Test Square Gradient (GPU) ===")

    # Create input tensor (1D to avoid reshape bug)
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cuda')
    x.requires_grad_()

    # Forward pass
    y = x.op_square()

    # Backward pass
    y.backward()

    # Expected gradient: dy/dx = 2x
    expected_grad = 2.0 * x_data

    # Get actual gradient
    actual_grad = np.array(list(x.grad.data()))

    print(f"Input: {x_data}")
    print(f"Output (x²): {np.array(list(y.data()))}")
    print(f"Expected gradient (2x): {expected_grad}")
    print(f"Actual gradient: {actual_grad}")

    # Check if gradients match
    if np.allclose(actual_grad, expected_grad, rtol=1e-5):
        print("✓ GPU gradient test PASSED")
        return True
    else:
        print("✗ GPU gradient test FAILED")
        print(f"Max error: {np.max(np.abs(actual_grad - expected_grad))}")
        return False

def test_square_chain_rule():
    """Test chain rule with square: z = (x²)² = x⁴, dz/dx = 4x³"""
    print("\n=== Test Square Chain Rule (CPU) ===")

    # Create input tensor
    x_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    # Forward pass: z = square(square(x)) = x⁴
    y = x.op_square()
    z = y.op_square()

    # Backward pass
    z.backward()

    # Expected gradient: dz/dx = 4x³
    expected_grad = 4.0 * (x_data ** 3)

    # Get actual gradient
    actual_grad = np.array(list(x.grad.data()))

    print(f"Input x:\n{x_data}")
    print(f"y = x²:\n{np.array(list(y.data()))}")
    print(f"z = y² = x⁴:\n{np.array(list(z.data()))}")
    print(f"Expected gradient (4x³):\n{expected_grad}")
    print(f"Actual gradient:\n{actual_grad}")

    # Check if gradients match
    if np.allclose(actual_grad, expected_grad, rtol=1e-5):
        print("✓ Chain rule test PASSED")
        return True
    else:
        print("✗ Chain rule test FAILED")
        print(f"Max error: {np.max(np.abs(actual_grad - expected_grad))}")
        return False

def test_square_mixed_ops():
    """Test square mixed with multiplication: z = x² * x = x³, dz/dx = 3x²"""
    print("\n=== Test Square Mixed Operations (CPU) ===")

    # Create input tensor
    x_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    # Forward pass: z = x² * x = x³
    y = x.op_square()
    z = y.op_mul(x)

    # Backward pass
    z.backward()

    # Expected gradient: dz/dx = 3x²
    expected_grad = 3.0 * (x_data ** 2)

    # Get actual gradient
    actual_grad = np.array(list(x.grad.data()))

    print(f"Input x:\n{x_data}")
    print(f"y = x²:\n{np.array(list(y.data()))}")
    print(f"z = y * x = x³:\n{np.array(list(z.data()))}")
    print(f"Expected gradient (3x²):\n{expected_grad}")
    print(f"Actual gradient:\n{actual_grad}")

    # Check if gradients match
    if np.allclose(actual_grad, expected_grad, rtol=1e-5):
        print("✓ Mixed operations test PASSED")
        return True
    else:
        print("✗ Mixed operations test FAILED")
        print(f"Max error: {np.max(np.abs(actual_grad - expected_grad))}")
        return False

def test_square_negative_values():
    print("\n=== Test Square Gradient (Negative Values) ===")

    x_data = np.array([-3.0, -2.0, -1.0, -0.5], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_square()
    y.backward()

    expected_grad = 2.0 * x_data
    actual_grad = np.array(list(x.grad.data()))

    if np.allclose(actual_grad, expected_grad, rtol=1e-5):
        print("PASS: square gradient with negative values")
        return True
    else:
        print(f"FAIL: expected {expected_grad}, got {actual_grad}")
        return False


def test_square_zeros():
    print("\n=== Test Square Gradient (Zeros) ===")

    x_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_square()
    y.backward()

    expected_grad = np.zeros_like(x_data)
    actual_grad = np.array(list(x.grad.data()))

    if np.allclose(actual_grad, expected_grad):
        print("PASS: square gradient at zeros")
        return True
    else:
        print(f"FAIL: expected {expected_grad}, got {actual_grad}")
        return False


def test_square_float64():
    print("\n=== Test Square Gradient (float64) ===")

    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    x = rp.values(x_data.tolist(), dtype='float64', device='cpu')
    x.requires_grad_()

    y = x.op_square()
    y.backward()

    expected_grad = 2.0 * x_data
    actual_grad = np.array(list(x.grad.data()))

    if np.allclose(actual_grad, expected_grad, rtol=1e-10):
        print("PASS: square gradient float64")
        return True
    else:
        print(f"FAIL: max error: {np.max(np.abs(actual_grad - expected_grad))}")
        return False


if __name__ == '__main__':
    print("Testing Square Backward Pass Implementation")
    print("=" * 50)

    results = []
    results.append(test_square_grad_cpu())
    results.append(test_square_grad_gpu())
    results.append(test_square_chain_rule())
    results.append(test_square_mixed_ops())
    results.append(test_square_negative_values())
    results.append(test_square_zeros())
    results.append(test_square_float64())

    print("\n" + "=" * 50)
    print(f"Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED")
        sys.exit(1)
