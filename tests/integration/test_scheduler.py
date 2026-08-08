import sys
sys.path.insert(0, '.')
import recpulse_cuda as rp
from recpulse.module import Module, Linear
from recpulse.optim import SGD, Adam
from recpulse.scheduler import StaticLR, StepLR, ReduceLROnPlateau

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

def approx(a, b, tol=1e-6):
    if abs(b) < 1e-12:
        return abs(a - b) < tol
    return abs(a - b) / max(abs(b), 1e-12) < tol

rp.manual_seed(42)

print("=" * 60)
print("LR Scheduler Tests")
print("=" * 60)

print("\n--- StaticLR ---")

opt = SGD([rp.randn([3])], lr=0.1)
sched = StaticLR(opt)

check("static initial lr", approx(sched.get_lr(), 0.1))

for _ in range(10):
    sched.step()

check("static lr unchanged after 10 steps", approx(sched.get_lr(), 0.1))
check("static optimizer lr unchanged", approx(opt.defaults['lr'], 0.1))

print("\n--- StepLR ---")

opt2 = SGD([rp.randn([3])], lr=1.0)
sched2 = StepLR(opt2, step_size=3, gamma=0.5)

check("step initial lr", approx(sched2.get_lr(), 1.0))

lrs = [sched2.get_lr()]
for i in range(9):
    sched2.step()
    lrs.append(sched2.get_lr())

check("step lr at epoch 1", approx(lrs[1], 1.0))
check("step lr at epoch 2", approx(lrs[2], 1.0))
check("step lr at epoch 3 (drop)", approx(lrs[3], 0.5))
check("step lr at epoch 4", approx(lrs[4], 0.5))
check("step lr at epoch 5", approx(lrs[5], 0.5))
check("step lr at epoch 6 (drop again)", approx(lrs[6], 0.25))
check("step lr at epoch 9 (drop 3rd)", approx(lrs[9], 0.125))
check("step optimizer lr matches", approx(opt2.defaults['lr'], sched2.get_lr()))

print("\n--- StepLR with different gamma ---")

opt3 = SGD([rp.randn([3])], lr=0.01)
sched3 = StepLR(opt3, step_size=5, gamma=0.1)

for _ in range(5):
    sched3.step()

check("gamma=0.1 after 5 steps", approx(sched3.get_lr(), 0.001))

for _ in range(5):
    sched3.step()

check("gamma=0.1 after 10 steps", approx(sched3.get_lr(), 0.0001))

print("\n--- ReduceLROnPlateau (min mode) ---")

opt4 = SGD([rp.randn([3])], lr=0.1)
sched4 = ReduceLROnPlateau(opt4, patience=3, factor=0.5, mode='min')

check("plateau initial lr", approx(sched4.get_lr(), 0.1))

sched4.step(1.0)
sched4.step(0.9)
sched4.step(0.8)
check("plateau lr after improving", approx(sched4.get_lr(), 0.1))

sched4.step(0.85)
sched4.step(0.85)
sched4.step(0.85)
check("plateau lr after 3 bad (patience=3)", approx(sched4.get_lr(), 0.05))

sched4.step(0.85)
sched4.step(0.85)
sched4.step(0.85)
check("plateau lr drops again", approx(sched4.get_lr(), 0.025))

print("\n--- ReduceLROnPlateau (max mode) ---")

opt5 = Adam([rp.randn([3])], lr=0.01)
sched5 = ReduceLROnPlateau(opt5, patience=2, factor=0.1, mode='max')

sched5.step(0.5)
sched5.step(0.6)
sched5.step(0.7)
check("plateau max: improving", approx(sched5.get_lr(), 0.01))

sched5.step(0.65)
sched5.step(0.65)
check("plateau max: 2 bad epochs -> drop", approx(sched5.get_lr(), 0.001))

print("\n--- ReduceLROnPlateau min_lr ---")

opt6 = SGD([rp.randn([3])], lr=0.01)
sched6 = ReduceLROnPlateau(opt6, patience=1, factor=0.1, min_lr=0.001)

sched6.step(1.0)
sched6.step(2.0)
check("min_lr: first drop", approx(sched6.get_lr(), 0.001))

sched6.step(3.0)
check("min_lr: clamped", approx(sched6.get_lr(), 0.001))

print("\n--- ReduceLROnPlateau reset on improvement ---")

opt7 = SGD([rp.randn([3])], lr=1.0)
sched7 = ReduceLROnPlateau(opt7, patience=3, factor=0.5, mode='min')

sched7.step(1.0)
sched7.step(1.1)
sched7.step(1.1)
check("plateau bad count before reset", sched7.num_bad == 2)

sched7.step(0.5)
check("plateau bad count after improvement", sched7.num_bad == 0)
check("plateau lr not dropped (reset)", approx(sched7.get_lr(), 1.0))

print("\n--- Integration with training loop ---")

rp.manual_seed(123)

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 2)
    def forward(self, x):
        h = self.keep(self.fc1(x))
        h = self.keep(h.op_relu())
        return self.fc2(h)

model = MLP()
opt = SGD(model.parameters(), lr=0.5)
sched = StepLR(opt, step_size=10, gamma=0.5)

x = rp.randn([8, 4])
targets = [0, 1] * 4

losses = []
lrs_log = []
for epoch in range(30):
    model.zero_grad()
    out = model(x)
    loss = out.op_cross_entropy_loss(targets)
    loss.backward()
    opt.step()
    sched.step()
    losses.append(loss.sum_all())
    lrs_log.append(sched.get_lr())

check("training with scheduler converges", losses[-1] < losses[0])
check("lr dropped at step 10", lrs_log[9] < lrs_log[8])
check("lr dropped at step 20", lrs_log[19] < lrs_log[18])

print("\n--- Works with Adam ---")

opt_adam = Adam([rp.randn([3])], lr=0.001)
sched_adam = StepLR(opt_adam, step_size=5, gamma=0.5)

for _ in range(5):
    sched_adam.step()

check("adam lr after step", approx(opt_adam.defaults['lr'], 0.0005))

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All scheduler tests passed!")
