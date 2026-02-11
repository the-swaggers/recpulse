import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import recpulse_cuda as rp
import numpy as np


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def test_sigmoid_gradient_cpu():
    print("Testing Sigmoid gradient on CPU...")

    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_sigmoid()

    y.backward()

    analytical_grad = np.array(list(x.grad.data()), dtype=np.float32)
    output_data = np.array(list(y.data()), dtype=np.float32)

    expected_output = sigmoid_np(x_data)
    sigmoid_out = sigmoid_np(x_data)
    expected_grad = sigmoid_out * (1 - sigmoid_out)

    print("Input:", x_data)
    print("Output:", output_data)
    print("Expected output:", expected_output)
    print("Analytical gradient:", analytical_grad)
    print("Expected gradient:", expected_grad)
    print("Output match:", np.allclose(output_data, expected_output, rtol=1e-5))
    print("Gradient match:", np.allclose(analytical_grad, expected_grad, rtol=1e-5))

    assert np.allclose(output_data, expected_output, rtol=1e-5), "CPU output mismatch!"
    assert np.allclose(analytical_grad, expected_grad, rtol=1e-5), "CPU gradient mismatch!"
    print("PASS: sigmoid gradient CPU")
    return True


def test_sigmoid_gradient_cuda():
    print("Testing Sigmoid gradient on CUDA...")

    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    x = rp.values(x_data.tolist(), dtype='float32', device='cuda')
    x.requires_grad_()

    y = x.op_sigmoid()

    y.backward()

    y_cpu = y.to('cpu')
    grad_cpu = x.grad.to('cpu')

    analytical_grad = np.array(list(grad_cpu.data()), dtype=np.float32)
    output_data = np.array(list(y_cpu.data()), dtype=np.float32)

    expected_output = sigmoid_np(x_data)
    sigmoid_out = sigmoid_np(x_data)
    expected_grad = sigmoid_out * (1 - sigmoid_out)

    print("Input:", x_data)
    print("Output:", output_data)
    print("Expected output:", expected_output)
    print("Analytical gradient:", analytical_grad)
    print("Expected gradient:", expected_grad)
    print("Output match:", np.allclose(output_data, expected_output, rtol=1e-5))
    print("Gradient match:", np.allclose(analytical_grad, expected_grad, rtol=1e-5))

    assert np.allclose(output_data, expected_output, rtol=1e-5), "CUDA output mismatch!"
    assert np.allclose(analytical_grad, expected_grad, rtol=1e-5), "CUDA gradient mismatch!"
    print("PASS: sigmoid gradient CUDA")
    return True


def test_sigmoid_chain_rule():
    print("Testing Sigmoid in a computation chain...")

    x_data = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)

    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_sigmoid()
    z = y.op_mul(y)

    z.backward()

    analytical_grad = np.array(list(x.grad.data()), dtype=np.float32)

    sigmoid_out = sigmoid_np(x_data)
    expected_grad = 2 * sigmoid_out * sigmoid_out * (1 - sigmoid_out)

    print("Input:", x_data)
    print("Sigmoid output:", np.array(list(y.data()), dtype=np.float32))
    print("z = sigmoid(x)^2:", np.array(list(z.data()), dtype=np.float32))
    print("Analytical gradient (dz/dx):", analytical_grad)
    print("Expected gradient:", expected_grad)
    print("Match:", np.allclose(analytical_grad, expected_grad, rtol=1e-5))

    assert np.allclose(analytical_grad, expected_grad, rtol=1e-5), "Chain rule gradient mismatch!"
    print("PASS: sigmoid chain rule")
    return True


def test_sigmoid_extreme_values():
    x_data = np.array([-50.0, -20.0, 20.0, 50.0], dtype=np.float32)

    x = rp.values(x_data.tolist(), dtype='float32', device='cpu')
    x.requires_grad_()

    y = x.op_sigmoid()
    y.backward()

    actual_grad = np.array(list(x.grad.data()), dtype=np.float32)

    for g in actual_grad:
        assert g >= 0.0, f"Sigmoid gradient should be non-negative, got {g}"
        assert g < 1.0, f"Sigmoid gradient should be < 1, got {g}"

    assert actual_grad[0] < 1e-6, f"Sigmoid grad at x=-50 should be ~0, got {actual_grad[0]}"
    assert actual_grad[3] < 1e-6, f"Sigmoid grad at x=50 should be ~0, got {actual_grad[3]}"

    print("PASS: sigmoid extreme values (large positive/negative)")
    return True


def test_sigmoid_float64():
    x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)

    x = rp.values(x_data.tolist(), dtype='float64', device='cpu')
    x.requires_grad_()

    y = x.op_sigmoid()
    y.backward()

    analytical_grad = np.array(list(x.grad.data()), dtype=np.float64)
    output_data = np.array(list(y.data()), dtype=np.float64)

    expected_output = 1.0 / (1.0 + np.exp(-x_data))
    sigmoid_out = expected_output
    expected_grad = sigmoid_out * (1 - sigmoid_out)

    assert np.allclose(output_data, expected_output, rtol=1e-10), \
        f"float64 output mismatch: {output_data} vs {expected_output}"
    assert np.allclose(analytical_grad, expected_grad, rtol=1e-10), \
        f"float64 gradient mismatch: {analytical_grad} vs {expected_grad}"

    print("PASS: sigmoid float64 precision")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Sigmoid Backward Pass Tests")
    print("=" * 60 + "\n")

    results = []
    results.append(test_sigmoid_gradient_cpu())
    results.append(test_sigmoid_gradient_cuda())
    results.append(test_sigmoid_chain_rule())
    results.append(test_sigmoid_extreme_values())
    results.append(test_sigmoid_float64())

    print("\n" + "=" * 60)
    p = sum(1 for r in results if r)
    t = len(results)
    print(f"Results: {p}/{t} tests passed")
    if p == t:
        print("All tests passed!")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
