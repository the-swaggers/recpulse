import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import numpy as np

def test_sinh_grad():
    print("Testing sinh gradient...")

    x_np = np.array([0.0, 0.5, 1.0, -0.5, 2.0], dtype=np.float32)

    x = rp.values(x_np.tolist())
    x.requires_grad_(True)

    y = x.op_sinh()
    y.backward()

    computed_grad = np.array(x.grad.data(), dtype=np.float32)
    expected_grad = np.cosh(x_np)

    print(f"  Input:          {x_np}")
    print(f"  Computed grad:  {computed_grad}")
    print(f"  Expected grad:  {expected_grad}")

    if np.allclose(computed_grad, expected_grad, rtol=1e-5, atol=1e-5):
        print("  PASSED")
        return True
    else:
        print("  FAILED")
        return False

def test_cosh_grad():
    print("Testing cosh gradient...")

    x_np = np.array([0.0, 0.5, 1.0, -0.5, 2.0], dtype=np.float32)

    x = rp.values(x_np.tolist())
    x.requires_grad_(True)

    y = x.op_cosh()
    y.backward()

    computed_grad = np.array(x.grad.data(), dtype=np.float32)
    expected_grad = np.sinh(x_np)

    print(f"  Input:          {x_np}")
    print(f"  Computed grad:  {computed_grad}")
    print(f"  Expected grad:  {expected_grad}")

    if np.allclose(computed_grad, expected_grad, rtol=1e-5, atol=1e-5):
        print("  PASSED")
        return True
    else:
        print("  FAILED")
        return False

def test_sinh_grad_chain():
    print("Testing sinh gradient with chain rule (sinh -> mul)...")

    x_np = np.array([0.0, 0.5, 1.0, -0.5, 2.0], dtype=np.float32)
    scale_np = np.array([2.0, 0.5, 1.0, 3.0, 0.1], dtype=np.float32)

    x = rp.values(x_np.tolist())
    x.requires_grad_(True)

    scale = rp.values(scale_np.tolist())

    y = x.op_sinh()
    z = y.op_mul(scale)
    z.backward()

    computed_grad = np.array(x.grad.data(), dtype=np.float32)
    expected_grad = scale_np * np.cosh(x_np)

    print(f"  Input:          {x_np}")
    print(f"  Scale:          {scale_np}")
    print(f"  Computed grad:  {computed_grad}")
    print(f"  Expected grad:  {expected_grad}")

    if np.allclose(computed_grad, expected_grad, rtol=1e-5, atol=1e-5):
        print("  PASSED")
        return True
    else:
        print("  FAILED")
        return False

def test_cosh_grad_chain():
    print("Testing cosh gradient with chain rule (cosh -> mul)...")

    x_np = np.array([0.0, 0.5, 1.0, -0.5, 2.0], dtype=np.float32)
    scale_np = np.array([2.0, 0.5, 1.0, 3.0, 0.1], dtype=np.float32)

    x = rp.values(x_np.tolist())
    x.requires_grad_(True)

    scale = rp.values(scale_np.tolist())

    y = x.op_cosh()
    z = y.op_mul(scale)
    z.backward()

    computed_grad = np.array(x.grad.data(), dtype=np.float32)
    expected_grad = scale_np * np.sinh(x_np)

    print(f"  Input:          {x_np}")
    print(f"  Scale:          {scale_np}")
    print(f"  Computed grad:  {computed_grad}")
    print(f"  Expected grad:  {expected_grad}")

    if np.allclose(computed_grad, expected_grad, rtol=1e-5, atol=1e-5):
        print("  PASSED")
        return True
    else:
        print("  FAILED")
        return False

def test_sinh_float64():
    print("Testing sinh gradient (float64)...")

    x_np = np.array([0.0, 0.5, 1.0, -0.5, 2.0], dtype=np.float64)

    x = rp.values(x_np.tolist(), dtype='float64')
    x.requires_grad_(True)

    y = x.op_sinh()
    y.backward()

    computed_grad = np.array(x.grad.data(), dtype=np.float64)
    expected_grad = np.cosh(x_np)

    if np.allclose(computed_grad, expected_grad, rtol=1e-10, atol=1e-10):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: max error {np.max(np.abs(computed_grad - expected_grad))}")
        return False


def test_cosh_float64():
    print("Testing cosh gradient (float64)...")

    x_np = np.array([0.0, 0.5, 1.0, -0.5, 2.0], dtype=np.float64)

    x = rp.values(x_np.tolist(), dtype='float64')
    x.requires_grad_(True)

    y = x.op_cosh()
    y.backward()

    computed_grad = np.array(x.grad.data(), dtype=np.float64)
    expected_grad = np.sinh(x_np)

    if np.allclose(computed_grad, expected_grad, rtol=1e-10, atol=1e-10):
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: max error {np.max(np.abs(computed_grad - expected_grad))}")
        return False


def test_sinh_at_zero():
    print("Testing sinh gradient at x=0...")

    x = rp.values([0.0], dtype='float64')
    x.requires_grad_(True)

    y = x.op_sinh()
    y.backward()

    grad_val = x.grad.data()[0]
    expected = np.cosh(0.0)

    if abs(grad_val - expected) < 1e-10:
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: expected {expected}, got {grad_val}")
        return False


def test_cosh_at_zero():
    print("Testing cosh gradient at x=0...")

    x = rp.values([0.0], dtype='float64')
    x.requires_grad_(True)

    y = x.op_cosh()
    y.backward()

    grad_val = x.grad.data()[0]
    expected = np.sinh(0.0)

    if abs(grad_val - expected) < 1e-10:
        print("  PASSED")
        return True
    else:
        print(f"  FAILED: expected {expected}, got {grad_val}")
        return False


if __name__ == "__main__":
    results = []
    results.append(test_sinh_grad())
    results.append(test_cosh_grad())
    results.append(test_sinh_grad_chain())
    results.append(test_cosh_grad_chain())
    results.append(test_sinh_float64())
    results.append(test_cosh_float64())
    results.append(test_sinh_at_zero())
    results.append(test_cosh_at_zero())

    print("\n" + "="*50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if all(results):
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)
