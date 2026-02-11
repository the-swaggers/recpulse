#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import recpulse_cuda as rp
import math

print("=" * 70)
print("  RecPulse Shape Operations - Comprehensive Test Suite")
print("=" * 70)

EPSILON = 1e-5
GRAD_EPSILON = 1e-4

def allclose(a, b, rtol=1e-5, atol=1e-5):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > atol + rtol * abs(y):
            return False
    return True


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

    def record(self, name, passed, msg=""):
        if passed:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            self.errors.append((name, msg))
            print(f"  ❌ {name}: {msg}")

    def skip(self, name, reason=""):
        self.skipped += 1
        print(f"  ⏭️  {name}: SKIPPED - {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 70}")
        print(f"  SUMMARY: {self.passed}/{total} tests passed ({self.skipped} skipped)")
        if self.errors:
            print(f"\n  Failed tests:")
            for name, msg in self.errors:
                print(f"    - {name}: {msg}")
        print("=" * 70)
        return self.failed == 0


results = TestResult()

print("\n" + "=" * 70)
print("  SECTION 1: Forward Pass Tests")
print("=" * 70)

print("\n--- view ---")
t = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cpu')
v = t.view([2, 3])
results.record("view: 1D -> [2,3]",
               v.shape == (2, 3))
results.record("view: data preserved",
               allclose(list(v.data()), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

v2 = t.view([3, 2])
results.record("view: 1D -> [3,2]",
               v2.shape == (3, 2))

v3 = t.view([6])
results.record("view: 1D -> [6]",
               v3.shape == (6,))

print("\n--- reshape ---")
t = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cpu').view([2, 3])
r = t.reshape([3, 2])
results.record("reshape: [2,3] -> [3,2]",
               r.shape == (3, 2))

print("\n--- transpose ---")
t = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cpu').view([2, 3])
tr = t.transpose(0, 1)
results.record("transpose: [2,3] -> [3,2]",
               tr.shape == (3, 2))

print("\n--- squeeze ---")
t = rp.ones([3, 1, 5], device='cpu')
sq = t.squeeze(1)
results.record("squeeze: [3,1,5] dim=1 -> [3,5]",
               sq.shape == (3, 5))

t2 = rp.ones([1, 3, 1, 5, 1], device='cpu')
sq2 = t2.squeeze(-1)
results.record("squeeze: [1,3,1,5,1] dim=-1 (all) -> [3,5]",
               sq2.shape == (3, 5))

print("\n--- unsqueeze ---")
t = rp.ones([3, 5], device='cpu')
usq = t.unsqueeze(0)
results.record("unsqueeze: [3,5] dim=0 -> [1,3,5]",
               usq.shape == (1, 3, 5))

usq2 = t.unsqueeze(1)
results.record("unsqueeze: [3,5] dim=1 -> [3,1,5]",
               usq2.shape == (3, 1, 5))

usq3 = t.unsqueeze(-1)
results.record("unsqueeze: [3,5] dim=-1 -> [3,5,1]",
               usq3.shape == (3, 5, 1))

print("\n--- flatten ---")
t = rp.ones([2, 3, 4], device='cpu')
fl = t.flatten(0, 1)
results.record("flatten: [2,3,4] dims 0-1 -> [6,4]",
               fl.shape == (6, 4))

fl2 = t.flatten(0, -1)
results.record("flatten: [2,3,4] dims 0-(-1) -> [24]",
               fl2.shape == (24,))

fl3 = t.flatten(1, 2)
results.record("flatten: [2,3,4] dims 1-2 -> [2,12]",
               fl3.shape == (2, 12))

print("\n--- permute ---")
t = rp.values(list(range(1, 25)), dtype='float32', device='cpu').view([2, 3, 4])
p = t.permute([2, 0, 1])
results.record("permute: [2,3,4] -> [4,2,3]",
               p.shape == (4, 2, 3))

p2 = t.permute([0, 2, 1])
results.record("permute: [2,3,4] -> [2,4,3]",
               p2.shape == (2, 4, 3))

print("\n--- chunk ---")
t = rp.ones([12, 5], device='cpu')
chunks = t.chunk(3, 0)
results.record("chunk: [12,5] into 3 along dim 0",
               len(chunks) == 3 and all(c.shape == (4, 5) for c in chunks))

t2 = rp.ones([10, 5], device='cpu')
chunks2 = t2.chunk(3, 0)
results.record("chunk: [10,5] into 3 (uneven)",
               len(chunks2) == 3 and
               chunks2[0].shape == (4, 5) and
               chunks2[1].shape == (4, 5) and
               chunks2[2].shape == (2, 5))

print("\n--- expand ---")
t = rp.values([1.0, 2.0, 3.0], device='cpu').view([1, 3])
ex = t.expand([4, 3])
results.record("expand: [1,3] -> [4,3]",
               ex.shape == (4, 3))

t2 = rp.ones([3, 1, 5], device='cpu')
ex2 = t2.expand([3, 4, 5])
results.record("expand: [3,1,5] -> [3,4,5]",
               ex2.shape == (3, 4, 5))

print("\n--- repeat ---")
x = rp.ones([2, 3], device='cpu')
rep = x.repeat([2, 2])
results.record("repeat: [2,3] x [2,2] -> [4,6]",
               rep.shape == (4, 6))
results.record("repeat: data correctly tiled",
               allclose(list(rep.data()), [1.0] * 24))

print("\n" + "=" * 70)
print("  SECTION 2: Autograd Tests (Backward Pass)")
print("=" * 70)

print("\n--- view backward ---")
x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cpu').view([2, 3]).requires_grad_()
y = x.reshape([3, 2])
y.backward()
if x.grad is not None:
    grad_data = list(x.grad.data())
    results.record("view/reshape backward: gradient shape matches input",
                   x.grad.shape == x.shape)
    results.record("view/reshape backward: gradient is ones",
                   allclose(grad_data, [1.0] * 6))
else:
    results.record("view/reshape backward: gradient computed", False, "grad is None")

print("\n--- transpose backward ---")
x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cpu').view([2, 3]).requires_grad_()
y = x.transpose(0, 1)
y.backward()
if x.grad is not None:
    results.record("transpose backward: gradient shape matches input",
                   x.grad.shape == x.shape)
    results.record("transpose backward: gradient is ones",
                   allclose(list(x.grad.data()), [1.0] * 6))
else:
    results.record("transpose backward: gradient computed", False, "grad is None")

print("\n--- squeeze backward ---")
x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cpu').view([2, 1, 3]).requires_grad_()
y = x.squeeze(1)
if y.shape == (2, 3):
    y.backward()
    if x.grad is not None:
        results.record("squeeze backward: gradient shape matches input",
                       x.grad.shape == x.shape)
        results.record("squeeze backward: gradient is ones",
                       allclose(list(x.grad.data()), [1.0] * 6))
    else:
        results.record("squeeze backward: gradient computed", False, "grad is None")
else:
    results.record("squeeze backward", False, f"forward failed: shape={y.shape}")

print("\n--- unsqueeze backward ---")
x = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cpu').view([2, 3]).requires_grad_()
y = x.unsqueeze(1)
if y.shape == (2, 1, 3):
    y.backward()
    if x.grad is not None:
        results.record("unsqueeze backward: gradient shape matches input",
                       x.grad.shape == x.shape)
        results.record("unsqueeze backward: gradient is ones",
                       allclose(list(x.grad.data()), [1.0] * 6))
    else:
        results.record("unsqueeze backward: gradient computed", False, "grad is None")
else:
    results.record("unsqueeze backward", False, f"forward failed: shape={y.shape}")

print("\n--- flatten backward ---")
x = rp.values(list(range(1, 25)), dtype='float32', device='cpu').view([2, 3, 4]).requires_grad_()
y = x.flatten(0, 1)
if y.shape == (6, 4):
    y.backward()
    if x.grad is not None:
        results.record("flatten backward: gradient shape matches input",
                       x.grad.shape == x.shape)
        results.record("flatten backward: gradient is ones",
                       allclose(list(x.grad.data()), [1.0] * 24))
    else:
        results.record("flatten backward: gradient computed", False, "grad is None")
else:
    results.record("flatten backward", False, f"forward failed: shape={y.shape}")

print("\n--- permute backward ---")
x = rp.values(list(range(1, 25)), dtype='float32', device='cpu').view([2, 3, 4]).requires_grad_()
y = x.permute([2, 0, 1])
if y.shape == (4, 2, 3):
    y.backward()
    if x.grad is not None:
        results.record("permute backward: gradient shape matches input",
                       x.grad.shape == x.shape)
        results.record("permute backward: gradient is ones",
                       allclose(list(x.grad.data()), [1.0] * 24))
    else:
        results.record("permute backward: gradient computed", False, "grad is None")
else:
    results.record("permute backward", False, f"forward failed: shape={y.shape}")

print("\n--- expand backward ---")
x = rp.values([1.0, 2.0, 3.0], device='cpu').view([1, 3]).requires_grad_()
y = x.expand([4, 3])
if y.shape == (4, 3):
    y.backward()
    if x.grad is not None:
        results.record("expand backward: gradient shape matches input",
                       x.grad.shape == x.shape)
        results.record("expand backward: gradient sums along expanded dims",
                       allclose(list(x.grad.data()), [4.0, 4.0, 4.0]))
    else:
        results.record("expand backward: gradient computed", False, "grad is None")
else:
    results.record("expand backward", False, f"forward failed: shape={y.shape}")

print("\n--- repeat backward ---")
x = rp.values([1.0, 2.0, 3.0], device='cpu').view([1, 3]).requires_grad_()
y = x.repeat([2, 3])
if y.shape == (2, 9):
    y.backward()
    if x.grad is not None:
        results.record("repeat backward: gradient shape matches input",
                       x.grad.shape == x.shape)
        results.record("repeat backward: gradient sums from repeated copies",
                       allclose(list(x.grad.data()), [6.0, 6.0, 6.0]))
    else:
        results.record("repeat backward: gradient computed", False, "grad is None")
else:
    results.record("repeat backward", False, f"forward failed: shape={y.shape}")

print("\n--- chunk backward ---")
x = rp.values(list(range(1, 13)), dtype='float32', device='cpu').requires_grad_()
chunks = x.chunk(3, 0)
if len(chunks) == 3:
    for c in chunks:
        c.backward()
    if x.grad is not None:
        results.record("chunk backward: gradient shape matches input",
                       x.grad.shape == x.shape)
        results.record("chunk backward: gradient accumulates from all chunks",
                       allclose(list(x.grad.data()), [1.0] * 12))
    else:
        results.record("chunk backward: gradient computed", False, "grad is None")
else:
    results.record("chunk backward", False, f"wrong number of chunks: {len(chunks)}")

print("\n" + "=" * 70)
print("  SECTION 3: Numerical Gradient Check")
print("=" * 70)
print("  Note: Numerical gradient tests skipped - requires backward implementations")
print("  These tests will work once the backward passes are implemented.")

print("\n" + "=" * 70)
print("  SECTION 4: GPU Tests")
print("=" * 70)

try:
    t_gpu = rp.ones([2, 3], device='cuda')

    print("\n--- GPU forward tests ---")
    v_gpu = t_gpu.reshape([3, 2])
    results.record("GPU reshape: [2,3] -> [3,2]",
                   v_gpu.shape == (3, 2) and v_gpu.device.startswith('cuda'))

    sq_gpu = rp.ones([2, 1, 3], device='cuda').squeeze(1)
    results.record("GPU squeeze: [2,1,3] -> [2,3]",
                   sq_gpu.shape == (2, 3))

    usq_gpu = rp.ones([2, 3], device='cuda').unsqueeze(0)
    results.record("GPU unsqueeze: [2,3] -> [1,2,3]",
                   usq_gpu.shape == (1, 2, 3))

    fl_gpu = rp.ones([2, 3, 4], device='cuda').flatten(0, 1)
    results.record("GPU flatten: [2,3,4] -> [6,4]",
                   fl_gpu.shape == (6, 4))

    tr_gpu = t_gpu.transpose(0, 1)
    results.record("GPU transpose: [2,3] -> [3,2]",
                   tr_gpu.shape == (3, 2))

    p_gpu = rp.ones([2, 3, 4], device='cuda').permute([2, 0, 1])
    results.record("GPU permute: [2,3,4] -> [4,2,3]",
                   p_gpu.shape == (4, 2, 3))

    ex_gpu = rp.ones([1, 3], device='cuda').expand([4, 3])
    results.record("GPU expand: [1,3] -> [4,3]",
                   ex_gpu.shape == (4, 3))

    chunks_gpu = rp.ones([12, 3], device='cuda').chunk(4, 0)
    results.record("GPU chunk: [12,3] into 4",
                   len(chunks_gpu) == 4 and all(c.shape == (3, 3) for c in chunks_gpu))

    print("\n--- GPU backward tests ---")
    x_gpu = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cuda').view([2, 3]).requires_grad_()
    y_gpu = x_gpu.reshape([3, 2])
    y_gpu.backward()
    if x_gpu.grad is not None:
        results.record("GPU reshape backward: gradient computed",
                       x_gpu.grad.shape == x_gpu.shape)
    else:
        results.record("GPU reshape backward: gradient computed", False, "grad is None")

    x_gpu2 = rp.values([1.0, 2.0, 3.0], device='cuda').view([1, 3]).requires_grad_()
    y_gpu2 = x_gpu2.expand([4, 3])
    y_gpu2.backward()
    if x_gpu2.grad is not None:
        grad_cpu = x_gpu2.grad.to('cpu')
        results.record("GPU expand backward: gradient sums correctly",
                       allclose(list(grad_cpu.data()), [4.0, 4.0, 4.0]))
    else:
        results.record("GPU expand backward: gradient computed", False, "grad is None")

except Exception as e:
    print(f"\n⚠️  GPU tests skipped: {e}")

print("\n" + "=" * 70)
print("  SECTION 5: Edge Cases and Error Handling")
print("=" * 70)

print("\n--- Invalid operations (should raise errors) ---")
try:
    t = rp.ones([2, 3], device='cpu')
    bad = t.squeeze(0)
    results.record("squeeze non-size-1 dim raises error", False, "should have raised")
except RuntimeError:
    results.record("squeeze non-size-1 dim raises error", True)

try:
    t = rp.ones([2, 3], device='cpu')
    bad = t.view([5, 5])
    results.record("view with wrong total size raises error", False, "should have raised")
except RuntimeError:
    results.record("view with wrong total size raises error", True)

try:
    t = rp.ones([2, 3, 4], device='cpu')
    bad = t.permute([0, 1, 1])
    results.record("permute with duplicate dims raises error", False, "should have raised")
except RuntimeError:
    results.record("permute with duplicate dims raises error", True)

try:
    t = rp.ones([3, 4, 5], device='cpu')
    bad = t.expand([3, 2, 5])
    results.record("expand with incompatible size raises error", False, "should have raised")
except RuntimeError:
    results.record("expand with incompatible size raises error", True)

print("\n--- reshape then transpose chain backward ---")
x = rp.values(list(range(1, 7)), dtype='float32', device='cpu').view([2, 3]).requires_grad_()
reshaped = x.reshape([3, 2])
transposed = reshaped.transpose(0, 1)
transposed.backward()
if x.grad is not None:
    results.record("reshape->transpose chain backward: gradient shape",
                   x.grad.shape == x.shape)
    results.record("reshape->transpose chain backward: gradient is ones",
                   allclose(list(x.grad.data()), [1.0] * 6))
else:
    results.record("reshape->transpose chain backward: gradient computed", False, "grad is None")

print("\n--- float64 shape op backward ---")
x64 = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float64', device='cpu').view([2, 3]).requires_grad_()
y64 = x64.flatten(0, -1)
y64.backward()
if x64.grad is not None:
    results.record("float64 flatten backward: gradient shape",
                   x64.grad.shape == x64.shape)
    results.record("float64 flatten backward: gradient is ones",
                   allclose(list(x64.grad.data()), [1.0] * 6))
else:
    results.record("float64 flatten backward: gradient computed", False, "grad is None")

print("\n--- Round-trip operations ---")
t = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cpu').view([2, 3])

usq = t.unsqueeze(1)
sq = usq.squeeze(1)
results.record("unsqueeze + squeeze round-trip",
               sq.shape == t.shape and allclose(list(sq.data()), list(t.data())))

fl = t.flatten(0, -1)
rs = fl.reshape([2, 3])
results.record("flatten + reshape round-trip",
               rs.shape == t.shape and allclose(list(rs.data()), list(t.data())))

tr = t.transpose(0, 1).transpose(0, 1)
results.record("double transpose round-trip",
               tr.shape == t.shape)

t4d = rp.ones([2, 3, 4, 5], device='cpu')
p = t4d.permute([3, 2, 1, 0])
p_inv = p.permute([3, 2, 1, 0])
results.record("permute + inverse permute round-trip",
               p_inv.shape == t4d.shape)

results.summary()

sys.exit(0 if results.failed == 0 else 1)
