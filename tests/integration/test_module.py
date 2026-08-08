import sys
sys.path.insert(0, '.')
import recpulse_cuda as rp
from recpulse.module import Module, Linear, Conv2d, MaxPool2d, AvgPool2d
from recpulse.optim import SGD, Adam

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
print("Module / Linear / Built-in Module Tests")
print("=" * 60)

print("\n--- Module basics ---")

class EmptyModel(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

m = EmptyModel()
check("empty tracked", len(m.tracked) == 0)
check("empty learnable", len(m.learnable()) == 0)
check("empty frozen", len(m.frozen()) == 0)
check("train mode default", m._training == True)

print("\n--- Linear ---")

fc = Linear(4, 3)
check("linear tracked keys", set(fc.tracked.keys()) == {"weight", "bias"})
check("linear weight shape", fc.weight.shape == (4, 3))
check("linear bias shape", fc.bias.shape == (3,))
check("linear weight requires_grad", fc.weight.requires_grad)
check("linear bias requires_grad", fc.bias.requires_grad)
check("linear learnable count", len(fc.learnable()) == 2)
check("linear parameters count", len(fc.parameters()) == 2)

x = rp.randn([5, 4])
out = fc(x)
check("linear forward shape", out.shape == (5, 3))

print("\n--- Linear no bias ---")

fc_nb = Linear(4, 3, bias=False)
check("no bias tracked", len(fc_nb.tracked) == 1)
check("no bias is None", fc_nb.bias is None)
out_nb = fc_nb(x)
check("no bias forward shape", out_nb.shape == (5, 3))

print("\n--- Submodule registration ---")

class TwoLayer(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 2)

    def forward(self, x):
        h = self.keep(self.fc1(x))
        h = self.keep(h.op_relu())
        return self.fc2(h)

model = TwoLayer()
check("submodule tracked keys", set(model.tracked.keys()) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"})
check("submodule learnable count", len(model.learnable()) == 4)
check("submodule parameters count", len(model.parameters()) == 4)
check("submodules registered", "fc1" in model._modules and "fc2" in model._modules)

print("\n--- Forward pass ---")

x = rp.randn([3, 4])
out = model(x)
check("two layer forward shape", out.shape == (3, 2))

print("\n--- Training loop (MLP) ---")

rp.manual_seed(123)

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 16)
        self.fc2 = Linear(16, 2)

    def forward(self, x):
        h = self.keep(self.fc1(x))
        h = self.keep(h.op_relu())
        return self.fc2(h)

mlp = MLP()
optimizer = SGD(mlp.parameters(), lr=0.5)

x = rp.randn([16, 4])
targets = [0, 1] * 8

initial_loss = None
final_loss = None

for step in range(50):
    mlp.zero_grad()
    out = mlp(x)
    loss = out.op_cross_entropy_loss(targets)
    loss.backward()
    optimizer.step()

    l = loss.sum_all()
    if step == 0:
        initial_loss = l
    if step == 49:
        final_loss = l

check(f"training loss decreases ({initial_loss:.4f} -> {final_loss:.4f})", final_loss < initial_loss * 0.5)
check("training converges", final_loss < 0.3)

print("\n--- Training with Adam ---")

rp.manual_seed(456)

mlp2 = MLP()
optimizer2 = Adam(mlp2.parameters(), lr=0.01)

x2 = rp.randn([16, 4])

for step in range(100):
    mlp2.zero_grad()
    out2 = mlp2(x2)
    loss2 = out2.op_cross_entropy_loss(targets)
    loss2.backward()
    optimizer2.step()

adam_loss = loss2.sum_all()
check(f"adam training converges ({adam_loss:.4f})", adam_loss < 0.3)

print("\n--- zero_grad ---")

mlp.zero_grad()
out = mlp(x)
loss = out.op_cross_entropy_loss(targets)
loss.backward()

has_grads = all(t.has_grad for t in mlp.parameters())
check("grads exist after backward", has_grads)

mlp.zero_grad()
no_grads = all(not t.has_grad for t in mlp.parameters())
check("grads cleared after zero_grad", no_grads)

print("\n--- train/eval mode ---")

mlp.eval()
check("eval mode", mlp._training == False)
check("eval propagates to fc1", mlp.fc1._training == False)

mlp.train()
check("train mode", mlp._training == True)
check("train propagates to fc1", mlp.fc1._training == True)

print("\n--- Conv2d module ---")

conv = Conv2d(1, 4, kernel_size=3, padding=1)
check("conv2d tracked", "weight" in conv.tracked and "bias" in conv.tracked)
check("conv2d weight shape", conv.weight.shape == (4, 1, 3, 3))
check("conv2d bias shape", conv.bias.shape == (4,))

x_img = rp.randn([2, 1, 8, 8])
x_img_base = x_img
out_conv = conv(x_img_base)
check("conv2d forward shape", out_conv.shape == (2, 4, 8, 8))

print("\n--- MaxPool2d module ---")

pool = MaxPool2d(2)
x_pool = rp.randn([1, 1, 4, 4])
x_pool_base = x_pool
out_pool = pool(x_pool_base)
check("maxpool forward shape", out_pool.shape == (1, 1, 2, 2))
check("maxpool no tracked params", len(pool.tracked) == 0)

print("\n--- AvgPool2d module ---")

avgpool = AvgPool2d(2)
out_avg = avgpool(x_pool_base)
check("avgpool forward shape", out_avg.shape == (1, 1, 2, 2))

print("\n--- MSE loss training ---")

rp.manual_seed(789)
reg = Linear(3, 1)
opt_reg = SGD(reg.parameters(), lr=0.01)

x_reg = rp.randn([10, 3])
y_reg = rp.randn([10, 1])

for step in range(100):
    reg.zero_grad()
    pred = reg(x_reg)
    loss_reg = pred.op_mse_loss(y_reg)
    loss_reg.backward()
    opt_reg.step()

mse_final = loss_reg.sum_all()
check(f"mse training converges ({mse_final:.4f})", mse_final < 1.0)

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All module tests passed!")
