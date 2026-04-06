import sys
sys.path.insert(0, '.')
import recpulse_cuda as rp
from recpulse.module import Module, Linear, LayerNorm, BatchNorm2d
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

def approx(a, b, tol=1e-2):
    if abs(b) < 1e-6:
        return abs(a - b) < tol
    return abs(a - b) / max(abs(b), 1e-6) < tol

rp.manual_seed(42)

print("=" * 60)
print("LayerNorm / BatchNorm2d Tests")
print("=" * 60)

print("\n--- LayerNorm forward ---")

ln = LayerNorm(4)
x = rp.randn([3, 4])
out = ln(x)
check("layernorm shape", out.shape == (3, 4))

m = out.op_mean_dim(1)
check("layernorm mean ~0", approx(m.sum_all(), 0.0, 0.01))

check("layernorm tracked", "weight" in ln.tracked and "bias" in ln.tracked)
check("layernorm learnable", len(ln.learnable()) == 2)

print("\n--- LayerNorm backward ---")

x = rp.randn([3, 4])
x.requires_grad_(True)
ln = LayerNorm(4)
out = ln(x)
out.op_sum_all().backward()
check("layernorm x grad", x.has_grad)
check("layernorm weight grad", ln.weight.has_grad)
check("layernorm bias grad", ln.bias.has_grad)

print("\n--- LayerNorm no affine ---")

ln_na = LayerNorm(4, elementwise_affine=False)
out_na = ln_na(rp.randn([3, 4]))
check("layernorm no affine shape", out_na.shape == (3, 4))
check("layernorm no affine no tracked", len(ln_na.tracked) == 0)

print("\n--- LayerNorm 2D normalized_shape ---")

ln_2d = LayerNorm([3, 4])
x_3d_base = rp.randn([2 * 3 * 4])
x_3d = x_3d_base.reshape([2, 3, 4])
out_3d = ln_2d(x_3d)
check("layernorm 2d shape", out_3d.shape == (2, 3, 4))

print("\n--- BatchNorm2d forward (train) ---")

bn = BatchNorm2d(3)
x_base = rp.randn([2 * 3 * 4 * 4])
x_4d = x_base.reshape([2, 3, 4, 4])
out_bn = bn(x_4d)
check("batchnorm shape", out_bn.shape == (2, 3, 4, 4))
check("batchnorm tracked", "weight" in bn.tracked and "bias" in bn.tracked)
check("batchnorm running stats tracked", "running_mean" in bn.tracked and "running_var" in bn.tracked)
check("batchnorm learnable count", len(bn.learnable()) == 2)

rm_sum = bn.running_mean.sum_all()
check("batchnorm running_mean updated", rm_sum != 0.0)

print("\n--- BatchNorm2d backward ---")

bn2 = BatchNorm2d(2)
x_base2 = rp.randn([3 * 2 * 4 * 4])
x_4d2 = x_base2.reshape([3, 2, 4, 4])
x_4d2.requires_grad_(True)
out_bn2 = bn2(x_4d2)
out_bn2.op_sum_all().backward()
check("batchnorm x grad", x_4d2.has_grad)
check("batchnorm weight grad", bn2.weight.has_grad)

print("\n--- BatchNorm2d eval mode ---")

bn.eval()
out_eval = bn(x_4d)
check("batchnorm eval shape", out_eval.shape == (2, 3, 4, 4))

bn.train()
check("batchnorm back to train", bn._training == True)

print("\n--- BatchNorm2d no affine ---")

bn_na = BatchNorm2d(3, affine=False)
out_na = bn_na(x_4d)
check("batchnorm no affine shape", out_na.shape == (2, 3, 4, 4))

print("\n--- Training with LayerNorm ---")

rp.manual_seed(123)

class LNModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.ln = LayerNorm(8)
        self.fc2 = Linear(8, 2)

    def forward(self, x):
        h = self.keep(self.fc1(x))
        h = self.keep(self.ln(h))
        h = self.keep(h.op_relu())
        return self.fc2(h)

model = LNModel()
opt = SGD(model.parameters(), lr=0.5)
x = rp.randn([8, 4])
targets = [0, 1] * 4

losses = []
for step in range(30):
    model.zero_grad()
    out = model(x)
    loss = out.op_cross_entropy_loss(targets)
    loss.backward()
    opt.step()
    losses.append(loss.sum_all())

check(f"LN training converges ({losses[0]:.3f} -> {losses[-1]:.3f})", losses[-1] < losses[0] * 0.3)

print("\n--- Training with BatchNorm2d ---")

rp.manual_seed(456)

class BNConvModel(Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(1, 4, 3, padding=1)
        self.bn = BatchNorm2d(4)
        self.fc = Linear(4 * 4 * 4, 2)

    def forward(self, x):
        h = self.keep(self.conv(x))
        h = self.keep(self.bn(h))
        h = self.keep(h.op_relu())
        h_flat = self.keep(h.reshape([x.shape[0], 4 * 4 * 4]))
        return self.fc(h_flat)

from recpulse.module import Conv2d

conv_model = BNConvModel()
opt2 = SGD(conv_model.parameters(), lr=0.1)
x_img_base = rp.randn([4 * 1 * 4 * 4])
x_img = x_img_base.reshape([4, 1, 4, 4])
targets2 = [0, 1, 0, 1]

losses2 = []
for step in range(20):
    conv_model.zero_grad()
    out2 = conv_model(x_img)
    loss2 = out2.op_cross_entropy_loss(targets2)
    loss2.backward()
    opt2.step()
    losses2.append(loss2.sum_all())

check(f"BN+Conv training converges ({losses2[0]:.3f} -> {losses2[-1]:.3f})", losses2[-1] < losses2[0] * 0.7)

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All normalization tests passed!")
