import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import numpy as np

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")

def values_from_np(arr):
    flat = rp.values(arr.flatten().tolist(), dtype='float64', device='cpu')
    if arr.ndim == 1:
        return flat
    reshaped = flat.reshape(list(arr.shape))
    result = reshaped.copy()
    return result

def assert_close(actual_data, expected_flat, tol=1e-10):
    if len(actual_data) != len(expected_flat):
        return False
    return all(abs(a - e) < tol for a, e in zip(actual_data, expected_flat))


def test_same_shape():
    print("\n[1] Same-shape (fast path): [3] + [3]")
    a_np = np.array([1.0, 2.0, 3.0])
    b_np = np.array([10.0, 20.0, 30.0])

    a = values_from_np(a_np)
    b = values_from_np(b_np)
    result = a.op_add(b)

    expected = a_np + b_np
    check("result is not None", result is not None)
    if result is not None:
        check("shape is (3,)", result.shape == (3,))
        check("values match", assert_close(result.data(), expected.flatten().tolist()))


def test_scalar_broadcast():
    print("\n[2] Scalar broadcast: [1] + [4]")
    a_np = np.array([5.0])
    b_np = np.array([1.0, 2.0, 3.0, 4.0])

    a = values_from_np(a_np)
    b = values_from_np(b_np)
    result = a.op_add(b)

    expected = a_np + b_np
    check("result is not None", result is not None)
    if result is not None:
        check("shape is (4,)", result.shape == (4,))
        check("values match", assert_close(result.data(), expected.flatten().tolist()))


def test_vector_matrix():
    print("\n[3] Vector + Matrix: [3] + [2,3]")
    a_np = np.array([1.0, 2.0, 3.0])
    b_np = np.array([[10.0, 20.0, 30.0],
                     [40.0, 50.0, 60.0]])

    a = values_from_np(a_np)
    b = values_from_np(b_np)
    result = a.op_add(b)

    expected = a_np + b_np
    check("result is not None", result is not None)
    if result is not None:
        check("shape is (2, 3)", result.shape == (2, 3))
        check("values match", assert_close(result.data(), expected.flatten().tolist()))


def test_column_row():
    print("\n[4] Column + Row: [3,1] + [1,4]")
    a_np = np.array([[1.0], [2.0], [3.0]])
    b_np = np.array([[10.0, 20.0, 30.0, 40.0]])

    a = values_from_np(a_np)
    b = values_from_np(b_np)
    result = a.op_add(b)

    expected = a_np + b_np
    check("result is not None", result is not None)
    if result is not None:
        check("shape is (3, 4)", result.shape == (3, 4))
        check("values match", assert_close(result.data(), expected.flatten().tolist()))


def test_ndim_mismatch():
    print("\n[5] Different ndim: [3] * [2,3]")
    a_np = np.array([100.0, 200.0, 300.0])
    b_np = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]])

    a = values_from_np(a_np)
    b = values_from_np(b_np)
    result = a.op_mul(b)

    expected = a_np * b_np
    check("result is not None", result is not None)
    if result is not None:
        check("shape is (2, 3)", result.shape == (2, 3))
        check("values match", assert_close(result.data(), expected.flatten().tolist()))


def test_higher_dim():
    print("\n[6] Higher dim: [2,1,4] + [3,4]")
    a_np = np.array([[[1.0, 2.0, 3.0, 4.0]],
                     [[5.0, 6.0, 7.0, 8.0]]])
    b_np = np.array([[10.0, 20.0, 30.0, 40.0],
                     [50.0, 60.0, 70.0, 80.0],
                     [90.0, 100.0, 110.0, 120.0]])

    a = values_from_np(a_np)
    b = values_from_np(b_np)
    result = a.op_add(b)

    expected = a_np + b_np
    check("result is not None", result is not None)
    if result is not None:
        check("shape is (2, 3, 4)", result.shape == (2, 3, 4))
        check("values match", assert_close(result.data(), expected.flatten().tolist()))


def test_all_ops_broadcast():
    print("\n[7] All 6 ops with broadcasting: [2,1] op [1,3]")
    a_np = np.array([[2.0], [3.0]])
    b_np = np.array([[4.0, 5.0, 6.0]])

    a = values_from_np(a_np)
    b = values_from_np(b_np)

    expected_shape = (2, 3)

    result_add = a.op_add(b)
    expected_add = a_np + b_np
    check("add: result is not None", result_add is not None)
    if result_add is not None:
        check("add: shape", result_add.shape == expected_shape)
        check("add: values", assert_close(result_add.data(), expected_add.flatten().tolist()))

    result_sub = a.op_sub(b)
    expected_sub = a_np - b_np
    check("sub: result is not None", result_sub is not None)
    if result_sub is not None:
        check("sub: shape", result_sub.shape == expected_shape)
        check("sub: values", assert_close(result_sub.data(), expected_sub.flatten().tolist()))

    result_mul = a.op_mul(b)
    expected_mul = a_np * b_np
    check("mul: result is not None", result_mul is not None)
    if result_mul is not None:
        check("mul: shape", result_mul.shape == expected_shape)
        check("mul: values", assert_close(result_mul.data(), expected_mul.flatten().tolist()))

    result_div = a.op_div(b)
    expected_div = a_np / b_np
    check("div: result is not None", result_div is not None)
    if result_div is not None:
        check("div: shape", result_div.shape == expected_shape)
        check("div: values", assert_close(result_div.data(), expected_div.flatten().tolist(), tol=1e-9))

    result_pow = a.op_power(b)
    expected_pow = a_np ** b_np
    check("power: result is not None", result_pow is not None)
    if result_pow is not None:
        check("power: shape", result_pow.shape == expected_shape)
        check("power: values", assert_close(result_pow.data(), expected_pow.flatten().tolist(), tol=1e-6))

    a_logb_np = np.array([[100.0], [1000.0]])
    b_logb_np = np.array([[10.0, 10.0, 10.0]])
    a_logb = values_from_np(a_logb_np)
    b_logb = values_from_np(b_logb_np)
    result_logb = a_logb.op_logb(b_logb)
    expected_logb = np.log(a_logb_np) / np.log(b_logb_np)
    check("logb: result is not None", result_logb is not None)
    if result_logb is not None:
        check("logb: shape", result_logb.shape == expected_shape)
        check("logb: values", assert_close(result_logb.data(), expected_logb.flatten().tolist(), tol=1e-9))


def test_shape_verification():
    print("\n[8] Shape verification for various broadcast combos")

    configs = [
        ([1], [5], (5,)),
        ([5], [1], (5,)),
        ([1, 3], [4, 3], (4, 3)),
        ([4, 1], [4, 3], (4, 3)),
        ([3], [4, 3], (4, 3)),
        ([1, 1], [3, 4], (3, 4)),
        ([2, 1, 3], [1, 4, 3], (2, 4, 3)),
    ]

    for a_shape, b_shape, expected_shape in configs:
        a_size = 1
        for s in a_shape:
            a_size *= s
        b_size = 1
        for s in b_shape:
            b_size *= s

        a_np = np.arange(1.0, a_size + 1.0).reshape(a_shape)
        b_np = np.arange(1.0, b_size + 1.0).reshape(b_shape)

        a = values_from_np(a_np)
        b = values_from_np(b_np)
        result = a.op_add(b)

        expected = a_np + b_np
        label = f"{list(a_shape)} + {list(b_shape)} -> {list(expected_shape)}"
        check(f"{label}: not None", result is not None)
        if result is not None:
            check(f"{label}: shape", result.shape == expected_shape)
            check(f"{label}: values", assert_close(result.data(), expected.flatten().tolist()))


def test_value_verification():
    print("\n[9] Detailed value verification")

    a_np = np.array([[1.0, 2.0, 3.0]])
    b_np = np.array([[10.0], [20.0], [30.0]])

    a = values_from_np(a_np)
    b = values_from_np(b_np)

    result_add = a.op_add(b)
    expected_add = np.array([[11.0, 12.0, 13.0],
                             [21.0, 22.0, 23.0],
                             [31.0, 32.0, 33.0]])
    check("manual add values",
          result_add is not None and assert_close(result_add.data(), expected_add.flatten().tolist()))

    result_mul = a.op_mul(b)
    expected_mul = np.array([[10.0, 20.0, 30.0],
                             [20.0, 40.0, 60.0],
                             [30.0, 60.0, 90.0]])
    check("manual mul values",
          result_mul is not None and assert_close(result_mul.data(), expected_mul.flatten().tolist()))

    result_sub = a.op_sub(b)
    expected_sub = np.array([[-9.0, -8.0, -7.0],
                             [-19.0, -18.0, -17.0],
                             [-29.0, -28.0, -27.0]])
    check("manual sub values",
          result_sub is not None and assert_close(result_sub.data(), expected_sub.flatten().tolist()))

    result_div = a.op_div(b)
    expected_div = a_np / b_np
    check("manual div values",
          result_div is not None and assert_close(result_div.data(), expected_div.flatten().tolist(), tol=1e-9))


def test_incompatible_shapes():
    print("\n[10] Incompatible shapes raise RuntimeError")

    a = rp.values([1.0, 2.0, 3.0], dtype='float64', device='cpu')
    b = rp.values([1.0, 2.0, 3.0, 4.0], dtype='float64', device='cpu')

    got_error = False
    try:
        result = a.op_add(b)
        check("[3] + [4] returns None", result is None)
    except RuntimeError:
        got_error = True
        check("[3] + [4] raises RuntimeError", True)

    a2_flat = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float64', device='cpu')
    a2 = a2_flat.reshape([2, 3])
    a2 = a2.copy()
    b2_flat = rp.values([1.0, 2.0, 3.0, 4.0], dtype='float64', device='cpu')
    b2 = b2_flat.reshape([2, 2])
    b2 = b2.copy()

    got_error2 = False
    try:
        result2 = a2.op_add(b2)
        check("[2,3] + [2,2] returns None", result2 is None)
    except RuntimeError:
        got_error2 = True
        check("[2,3] + [2,2] raises RuntimeError", True)

    a3_flat = rp.values([float(x) for x in range(1, 13)], dtype='float64', device='cpu')
    a3 = a3_flat.reshape([3, 4])
    a3 = a3.copy()
    b3_flat = rp.values([float(x) for x in range(1, 11)], dtype='float64', device='cpu')
    b3 = b3_flat.reshape([2, 5])
    b3 = b3.copy()

    got_error3 = False
    try:
        result3 = a3.op_mul(b3)
        check("[3,4] * [2,5] returns None", result3 is None)
    except RuntimeError:
        got_error3 = True
        check("[3,4] * [2,5] raises RuntimeError", True)


def test_broadcast_commutativity():
    print("\n[11] Broadcast commutativity: a+b == b+a, a*b == b*a")
    a_np = np.array([1.0, 2.0, 3.0])
    b_np = np.array([[10.0], [20.0]])

    a = values_from_np(a_np)
    b = values_from_np(b_np)

    result_ab = a.op_add(b)
    result_ba = b.op_add(a)
    check("add a+b shape == b+a shape",
          result_ab is not None and result_ba is not None and result_ab.shape == result_ba.shape)
    if result_ab is not None and result_ba is not None:
        check("add a+b values == b+a values",
              assert_close(result_ab.data(), result_ba.data()))

    result_ab_mul = a.op_mul(b)
    result_ba_mul = b.op_mul(a)
    check("mul a*b shape == b*a shape",
          result_ab_mul is not None and result_ba_mul is not None and result_ab_mul.shape == result_ba_mul.shape)
    if result_ab_mul is not None and result_ba_mul is not None:
        check("mul a*b values == b*a values",
              assert_close(result_ab_mul.data(), result_ba_mul.data()))


def test_broadcast_with_ones_tensor():
    print("\n[12] Broadcasting with ones/zeros tensors")
    a_np = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]])
    a = values_from_np(a_np)
    ones = rp.values([1.0], dtype='float64', device='cpu')

    result_mul = a.op_mul(ones)
    check("tensor * [1] == tensor",
          result_mul is not None and assert_close(result_mul.data(), a_np.flatten().tolist()))

    zeros = rp.values([0.0], dtype='float64', device='cpu')
    result_add = a.op_add(zeros)
    check("tensor + [0] == tensor",
          result_add is not None and assert_close(result_add.data(), a_np.flatten().tolist()))


def test_broadcast_4d():
    print("\n[13] 4D broadcast: [2,1,1,3] + [1,4,1,1]")
    a_np = np.arange(1.0, 7.0).reshape(2, 1, 1, 3)
    b_np = np.arange(10.0, 50.0, 10.0).reshape(1, 4, 1, 1)

    a = values_from_np(a_np)
    b = values_from_np(b_np)
    result = a.op_add(b)

    expected = a_np + b_np
    check("result is not None", result is not None)
    if result is not None:
        check("shape is (2, 4, 1, 3)", result.shape == (2, 4, 1, 3))
        check("values match numpy",
              assert_close(result.data(), expected.flatten().tolist()))


if __name__ == '__main__':
    print("=" * 60)
    print("Broadcasting Forward Pass Tests")
    print("=" * 60)

    test_same_shape()
    test_scalar_broadcast()
    test_vector_matrix()
    test_column_row()
    test_ndim_mismatch()
    test_higher_dim()
    test_all_ops_broadcast()
    test_shape_verification()
    test_value_verification()
    test_incompatible_shapes()
    test_broadcast_commutativity()
    test_broadcast_with_ones_tensor()
    test_broadcast_4d()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
