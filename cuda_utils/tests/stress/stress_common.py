import sys, os, subprocess

STRESS_DIR = os.path.dirname(os.path.abspath(__file__))
CU_DIR = os.path.abspath(os.path.join(STRESS_DIR, '..', '..'))
for _p in (CU_DIR, STRESS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import recpulse_cuda as rp

passed = 0
failed = 0
failures = []


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS: {name}", flush=True)
    except Exception as e:
        failed += 1
        failures.append(name)
        print(f"  FAIL: {name} -- {type(e).__name__}: {e}", flush=True)


def finish():
    print()
    print(f"Results: {passed}/{passed + failed} tests passed")
    if failures:
        print("Failed tests:")
        for n in failures:
            print(f"  - {n}")
    sys.exit(1 if failed else 0)


_HAS_CUDA = None


def has_cuda():
    global _HAS_CUDA
    if _HAS_CUDA is None:
        try:
            rp.zeros([1], device='cuda')
            _HAS_CUDA = True
        except Exception:
            _HAS_CUDA = False
    return _HAS_CUDA


def t_from_np(a, dtype='float64', device='cpu', requires_grad=False):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    t = rp.from_numpy(a)
    if dtype != 'float64' or device != 'cpu':
        t = t.to(dtype=dtype, device=device)
    if requires_grad:
        t = t.requires_grad_()
    return t


def to_np(t):
    return np.asarray(t.to_numpy(), dtype=np.float64)


def grad_np(t):
    g = t.grad
    assert g is not None, "expected a gradient, got None"
    return np.asarray(g.to_numpy(), dtype=np.float64)


def assert_close(a, b, rtol=1e-5, atol=1e-7, ctx=""):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{ctx} shape mismatch {a.shape} vs {b.shape}"
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = np.abs(a - b)
        rel = diff / np.maximum(np.abs(b), 1e-12)
        raise AssertionError(
            f"{ctx} mismatch: max_abs={diff.max():.3e} max_rel={rel.max():.3e} "
            f"(rtol={rtol}, atol={atol})")


def run_isolated(code, timeout=120):
    pre = (
        "import sys\n"
        f"sys.path.insert(0, {CU_DIR!r})\n"
        "import recpulse_cuda as rp\n"
        "import numpy as np\n"
    )
    try:
        r = subprocess.run([sys.executable, '-c', pre + code],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return -999, "TIMEOUT"
