import sys
sys.path.insert(0, '.')
import recpulse_cuda as rp
from recpulse.module import Module, Linear, Embedding
from recpulse.optim import SGD

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1

def approx(a, b, tol=1e-3):
    if abs(b) < 1e-8:
        return abs(a - b) < tol
    return abs(a - b) / max(abs(b), 1e-8) < tol

rp.manual_seed(42)

print("=" * 60)
print("Embedding Tests")
print("=" * 60)

print("\n--- Embedding basics ---")

emb = Embedding(10, 4)
check("weight shape", emb.weight.shape == (10, 4))
check("tracked", "weight" in emb.tracked)
check("learnable", len(emb.learnable()) == 1)
check("requires_grad", emb.weight.requires_grad)

print("\n--- Forward with list indices ---")

out = emb([0, 3, 7])
check("list forward shape", out.shape == (3, 4))

print("\n--- Forward with tensor indices ---")

idx = rp.values([1.0, 5.0, 2.0])
out = emb(idx)
check("tensor forward shape", out.shape == (3, 4))

print("\n--- Forward value correctness ---")

emb2 = Embedding(5, 3)
w_data = [0.1, 0.2, 0.3,
          0.4, 0.5, 0.6,
          0.7, 0.8, 0.9,
          1.0, 1.1, 1.2,
          1.3, 1.4, 1.5]
_w_base = rp.values(w_data)
emb2.weight = _w_base.reshape([5, 3])
emb2.weight.requires_grad_(True)
emb2.tracked["weight"] = emb2.weight

out = emb2([2, 0, 4])
check("value sum (0.7+0.8+0.9+0.1+0.2+0.3+1.3+1.4+1.5 = 7.2)", approx(out.sum_all(), 7.2, 0.01))

print("\n--- Backward ---")

emb3 = Embedding(5, 3)
out = emb3([1, 3, 1])
loss = out.op_sum_all()
loss.backward()
check("backward has_grad", emb3.weight.has_grad)
check("backward grad shape", emb3.weight.grad.shape == (5, 3))

g = emb3.weight.grad
g_flat = g.reshape([15])
g_vals = [rp.values([0.0]).op_add(g_flat.slice([i], [i+1], [1])).sum_all() for i in range(15)]

check("grad row 0 = 0 (not selected)", all(approx(g_vals[j], 0.0) for j in [0, 1, 2]))
check("grad row 1 = 2 (selected twice)", all(approx(g_vals[j], 2.0) for j in [3, 4, 5]))
check("grad row 2 = 0 (not selected)", all(approx(g_vals[j], 0.0) for j in [6, 7, 8]))
check("grad row 3 = 1 (selected once)", all(approx(g_vals[j], 1.0) for j in [9, 10, 11]))
check("grad row 4 = 0 (not selected)", all(approx(g_vals[j], 0.0) for j in [12, 13, 14]))

print("\n--- Training with Embedding ---")

rp.manual_seed(123)

class EmbModel(Module):
    def __init__(self):
        super().__init__()
        self.emb = Embedding(5, 8)
        self.fc = Linear(8, 2)

    def forward(self, indices):
        h = self.keep(self.emb(indices))
        return self.fc(h)

model = EmbModel()
from recpulse.optim import Adam
opt = Adam(model.parameters(), lr=0.05)

indices = [0, 1, 2, 3, 4, 0, 1, 2]
targets = [0, 1, 0, 1, 0, 1, 0, 1]

losses = []
for step in range(80):
    model.zero_grad()
    out = model(indices)
    loss = out.op_cross_entropy_loss(targets)
    loss.backward()
    opt.step()
    losses.append(loss.sum_all())

check(f"training loss decreases ({losses[0]:.3f} -> {losses[-1]:.3f})", losses[-1] < losses[0])

print("\n--- Single index ---")

out_single = emb([3])
check("single index shape", out_single.shape == (1, 4))

print("\n--- Large vocabulary ---")

rp.manual_seed(42)
emb_large = Embedding(10000, 128)
check("large emb weight shape", emb_large.weight.shape == (10000, 128))
out_large = emb_large([0, 5000, 9999])
check("large emb forward shape", out_large.shape == (3, 128))

print("\n--- Submodule in larger model ---")

class FullModel(Module):
    def __init__(self):
        super().__init__()
        self.emb = Embedding(10, 4)
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 2)

    def forward(self, indices):
        h = self.keep(self.emb(indices))
        h = self.keep(self.fc1(h))
        h = self.keep(h.op_relu())
        return self.fc2(h)

full = FullModel()
check("full model tracked", len(full.tracked) == 5)
check("full model params", len(full.parameters()) == 5)

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All embedding tests passed!")
