import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import recpulse_cuda as rp
import numpy as np


def test_relu_gradient_cpu():
    print("Testing ReLU gradient on CPU...")

    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 0.5, -0.5, 1.5, -1.5, 0.0], dtype=np.float32)

    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_relu()

    y.backward()

    analytical_grad = np.array(list(x.grad.data()), dtype=np.float32)
    output_data = np.array(list(y.data()), dtype=np.float32)

    expected_output = np.maximum(0, x_data)
    expected_grad = (x_data > 0).astype(np.float32)

    print("Input:", x_data)
    print("Output:", output_data)
    print("Expected output:", expected_output)
    print("Analytical gradient:", analytical_grad)
    print("Expected gradient:", expected_grad)
    print("Output match:", np.allclose(output_data, expected_output))
    print("Gradient match:", np.allclose(analytical_grad, expected_grad))

    assert np.allclose(output_data, expected_output), "CPU output mismatch!"
    assert np.allclose(analytical_grad, expected_grad), "CPU gradient mismatch!"
    print("PASS: relu gradient CPU")
    return True


def test_relu_gradient_cuda():
    print("Testing ReLU gradient on CUDA...")

    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 0.5, -0.5, 1.5, -1.5, 0.0], dtype=np.float32)

    x = rp.values(x_data.tolist(), dtype='float32', device='cuda')
    x.requires_grad_()

    y = x.op_relu()

    y.backward()

    x_cpu = x.to('cpu')
    y_cpu = y.to('cpu')
    grad_cpu = x.grad.to('cpu')

    analytical_grad = np.array(list(grad_cpu.data()), dtype=np.float32)
    output_data = np.array(list(y_cpu.data()), dtype=np.float32)

    expected_output = np.maximum(0, x_data)
    expected_grad = (x_data > 0).astype(np.float32)

    print("Input:", x_data)
    print("Output:", output_data)
    print("Expected output:", expected_output)
    print("Analytical gradient:", analytical_grad)
    print("Expected gradient:", expected_grad)
    print("Output match:", np.allclose(output_data, expected_output))
    print("Gradient match:", np.allclose(analytical_grad, expected_grad))

    assert np.allclose(output_data, expected_output), "CUDA output mismatch!"
    assert np.allclose(analytical_grad, expected_grad), "CUDA gradient mismatch!"
    print("PASS: relu gradient CUDA")
    return True


def test_relu_chain_rule():
    print("Testing ReLU in a computation chain...")

    x_data = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float32)

    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_relu()
    z = y.op_mul(y)

    z.backward()

    if x.grad is None:
        print("ERROR: x.grad is None!")
        print("y.grad:", y.grad)
        print("z.grad:", z.grad)
        raise RuntimeError("Gradient not computed for x")

    analytical_grad = np.array(list(x.grad.data()), dtype=np.float32)

    relu_out = np.maximum(0, x_data)
    expected_grad = 2 * relu_out * (x_data > 0)

    print("Input:", x_data)
    print("ReLU output:", np.array(list(y.data()), dtype=np.float32))
    print("z = relu(x)^2:", np.array(list(z.data()), dtype=np.float32))
    print("Analytical gradient (dz/dx):", analytical_grad)
    print("Expected gradient:", expected_grad)
    print("Match:", np.allclose(analytical_grad, expected_grad))

    assert np.allclose(analytical_grad, expected_grad), "Chain rule gradient mismatch!"
    print("PASS: relu chain rule")
    return True


def test_leaky_relu_gradient():
    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 0.5, -0.5], dtype=np.float32)

    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_leaky_relu()
    y.backward()

    actual_grad = np.array(list(x.grad.data()), dtype=np.float32)
    alpha = 0.01
    expected_grad = np.where(x_data > 0, 1.0, alpha).astype(np.float32)

    assert np.allclose(actual_grad, expected_grad, rtol=1e-5), \
        f"Leaky ReLU gradient mismatch: {actual_grad} vs {expected_grad}"
    print("PASS: leaky_relu gradient")
    return True


def test_relu_all_negative():
    x_data = np.array([-5.0, -3.0, -1.0, -0.5, -0.01], dtype=np.float32)

    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_relu()
    y.backward()

    actual_grad = np.array(list(x.grad.data()), dtype=np.float32)
    expected_grad = np.zeros_like(x_data)

    assert np.allclose(actual_grad, expected_grad), \
        f"All-negative ReLU gradient mismatch: {actual_grad} vs {expected_grad}"
    print("PASS: relu all-negative input (grad all zeros)")
    return True


def test_relu_all_positive():
    x_data = np.array([0.01, 0.5, 1.0, 3.0, 5.0], dtype=np.float32)

    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_relu()
    y.backward()

    actual_grad = np.array(list(x.grad.data()), dtype=np.float32)
    expected_grad = np.ones_like(x_data)

    assert np.allclose(actual_grad, expected_grad), \
        f"All-positive ReLU gradient mismatch: {actual_grad} vs {expected_grad}"
    print("PASS: relu all-positive input (grad all ones)")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("ReLU Backward Pass Tests")
    print("=" * 60 + "\n")

    results = []
    results.append(test_relu_gradient_cpu())
    results.append(test_relu_gradient_cuda())
    results.append(test_relu_chain_rule())
    results.append(test_leaky_relu_gradient())
    results.append(test_relu_all_negative())
    results.append(test_relu_all_positive())

    print("\n" + "=" * 60)
    p = sum(1 for r in results if r)
    t = len(results)
    print(f"Results: {p}/{t} tests passed")
    if p == t:
        print("All tests passed!")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
