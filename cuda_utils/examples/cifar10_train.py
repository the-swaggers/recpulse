import sys
import time
import gc
sys.path.insert(0, '..')

import recpulse_cuda as rp
from recpulse.module import Module, Linear, Conv2d, MaxPool2d, Dropout
from recpulse.optim import Adam
from recpulse.scheduler import ReduceLROnPlateau
from recpulse.data import load_cifar10, get_batch_4d

rp.manual_seed(42)
DEVICE = 'cuda'
BS = 64
NUM_EPOCHS = 100

print("Loading CIFAR-10...")
train_img, train_lbl, test_img, test_lbl = load_cifar10('../data/cifar10')

class VGGNet(Module):
    def __init__(self):
        super().__init__()
        # Block 1: 3x32x32 -> 64x16x16
        self.conv1a = Conv2d(3, 64, 3, padding=1)
        self.conv1b = Conv2d(64, 64, 3, padding=1)
        self.pool1 = MaxPool2d(2)
        self.drop1 = Dropout(0.25)

        # Block 2: 64x16x16 -> 128x8x8
        self.conv2a = Conv2d(64, 128, 3, padding=1)
        self.conv2b = Conv2d(128, 128, 3, padding=1)
        self.pool2 = MaxPool2d(2)
        self.drop2 = Dropout(0.25)

        # Block 3: 128x8x8 -> 256x4x4
        self.conv3a = Conv2d(128, 256, 3, padding=1)
        self.conv3b = Conv2d(256, 256, 3, padding=1)
        self.pool3 = MaxPool2d(2)
        self.drop3 = Dropout(0.25)

        # Classifier: 256*4*4=4096 -> 512 -> 10
        self.fc1 = Linear(256 * 4 * 4, 512)
        self.drop4 = Dropout(0.5)
        self.fc2 = Linear(512, 10)

    def forward(self, x):
        h = self.keep(self.conv1a(x))
        h = self.keep(h.op_relu())
        h = self.keep(self.conv1b(h))
        h = self.keep(h.op_relu())
        h = self.keep(self.pool1(h))
        h = self.keep(self.drop1(h))

        h = self.keep(self.conv2a(h))
        h = self.keep(h.op_relu())
        h = self.keep(self.conv2b(h))
        h = self.keep(h.op_relu())
        h = self.keep(self.pool2(h))
        h = self.keep(self.drop2(h))

        h = self.keep(self.conv3a(h))
        h = self.keep(h.op_relu())
        h = self.keep(self.conv3b(h))
        h = self.keep(h.op_relu())
        h = self.keep(self.pool3(h))
        h = self.keep(self.drop3(h))

        h = self.keep(h.reshape([x.shape[0], 256 * 4 * 4]))
        h = self.keep(self.fc1(h))
        h = self.keep(h.op_relu())
        h = self.keep(self.drop4(h))
        return self.fc2(h)

def accuracy(model, images, labels):
    model.eval()
    c = 0; t = 0
    for b in range((len(labels) + 199) // 200):
        bi, bl = get_batch_4d(images, labels, b, 200)
        if bi is None:
            break
        o = model(bi.to(device=DEVICE)).to(device='cpu').to_numpy().argmax(axis=1)
        for i in range(len(o)):
            if o[i] == bl[i]:
                c += 1
            t += 1
        del bi, o
    model.train()
    return c / t

model = VGGNet()
model.to(device=DEVICE)
opt = Adam(model.parameters(), lr=0.001)
sched = ReduceLROnPlateau(opt, patience=5, factor=0.5, min_lr=1e-6)
num_batches = 50000 // BS

print(f"Params: {sum(t.size for t in model.parameters()):,}")
print(f"Training: {NUM_EPOCHS} epochs, batch_size={BS}, device={DEVICE}")
print(f"{'Epoch':>5}  {'Loss':>8}  {'Acc':>8}  {'LR':>10}  {'Time':>6}")
print("-" * 50)

best_acc = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    tl = 0; n = 0
    start = time.time()

    for b in range(num_batches):
        bi, bl = get_batch_4d(train_img, train_lbl, b, BS)
        if bi is None:
            break
        bg = bi.to(device=DEVICE)
        model.zero_grad()
        out = model(bg)
        loss = out.op_cross_entropy_loss(bl)
        tl += loss.to(device='cpu').sum_all(); n += 1
        loss.backward()
        rp.clip_grad_norm(model.parameters(), 2.0)
        opt.step()
        del bg, out, loss, bi
        if b % 50 == 0:
            gc.collect()

    elapsed = time.time() - start
    avg_loss = tl / n
    acc = accuracy(model, test_img, test_lbl)
    sched.step(avg_loss)
    gc.collect()

    if acc > best_acc:
        best_acc = acc
        rp.save(model.tracked, "../data/cifar10_best.rpt")

    print(f"{epoch+1:5d}  {avg_loss:8.4f}  {acc:7.2%}  {sched.get_lr():10.6f}  {elapsed:5.0f}s", flush=True)

print(f"\nBest accuracy: {best_acc:.2%}")
print(f"Model saved to data/cifar10_best.rpt")
