import sys
import os
import time
import gc
import math
import pickle
sys.path.insert(0, '..')

import numpy as np
import recpulse_cuda as rp
from recpulse.module import Module, Linear, Conv2d, BatchNorm2d, AvgPool2d
from recpulse.optim import SGD

rp.manual_seed(42)
np.random.seed(42)

DEVICE = os.environ.get('DEVICE', 'cuda')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 128))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 160))
MAX_BATCHES = int(os.environ.get('MAX_BATCHES', 0))
LR = float(os.environ.get('LR', 0.1))
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SCHEDULE = os.environ.get('SCHEDULE', 'step')  # 'step' or 'cosine'
MILESTONES = {int(x) for x in os.environ.get('MILESTONES', '80,120').split(',') if x.strip()}
LR_GAMMA = 0.1
GRAD_CLIP = float(os.environ.get('GRAD_CLIP', 1.0))
WARMUP_EPOCHS = int(os.environ.get('WARMUP_EPOCHS', 1))
LOG_EVERY = int(os.environ.get('LOG_EVERY', 50))

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)

OUT_DIR = os.environ.get('OUT_DIR', '../data/cifar10_resnet_run')
os.makedirs(OUT_DIR, exist_ok=True)


def load_cifar10_numpy(data_dir='../data/cifar10'):
    batch_dir = os.path.join(data_dir, 'cifar-10-batches-py')

    def load_batch(path):
        with open(path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        return d[b'data'], d[b'labels']

    train_x, train_y = [], []
    for i in range(1, 6):
        x, y = load_batch(os.path.join(batch_dir, f'data_batch_{i}'))
        train_x.append(x)
        train_y.extend(y)
    train_x = np.concatenate(train_x).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_x, test_y = load_batch(os.path.join(batch_dir, 'test_batch'))
    test_x = np.array(test_x).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return train_x, np.array(train_y, dtype=np.int64), test_x, np.array(test_y, dtype=np.int64)


print("Loading CIFAR-10...", flush=True)
train_images, train_labels, test_images, test_labels = load_cifar10_numpy()
print(f"Train: {train_images.shape}  Test: {test_images.shape}", flush=True)

test_images_norm = (test_images - CIFAR_MEAN) / CIFAR_STD


def augment_batch(x):
    n = x.shape[0]
    padded = np.pad(x, ((0, 0), (0, 0), (4, 4), (4, 4)), mode='reflect')
    out = np.empty_like(x)
    h_off = np.random.randint(0, 9, size=n)
    w_off = np.random.randint(0, 9, size=n)
    flips = np.random.rand(n) < 0.5
    for i in range(n):
        crop = padded[i, :, h_off[i]:h_off[i] + 32, w_off[i]:w_off[i] + 32]
        if flips[i]:
            crop = crop[:, :, ::-1]
        out[i] = crop
    return np.ascontiguousarray(out)


class ResBlock(Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_c)
        self.conv2 = Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_c)
        self.has_proj = (stride != 1) or (in_c != out_c)
        if self.has_proj:
            self.proj_conv = Conv2d(in_c, out_c, 1, stride=stride, padding=0, bias=False)
            self.proj_bn = BatchNorm2d(out_c)

    def forward(self, x):
        out = self.keep(self.conv1(x))
        out = self.keep(self.bn1(out))
        out = self.keep(out.op_relu())
        out = self.keep(self.conv2(out))
        out = self.keep(self.bn2(out))
        if self.has_proj:
            identity = self.keep(self.proj_conv(x))
            identity = self.keep(self.proj_bn(identity))
        else:
            identity = x
        out = self.keep(out.op_add(identity))
        return self.keep(out.op_relu())


class ResNet20(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = BatchNorm2d(16)
        self.s1_b1 = ResBlock(16, 16); self.s1_b2 = ResBlock(16, 16); self.s1_b3 = ResBlock(16, 16)
        self.s2_b1 = ResBlock(16, 32, stride=2); self.s2_b2 = ResBlock(32, 32); self.s2_b3 = ResBlock(32, 32)
        self.s3_b1 = ResBlock(32, 64, stride=2); self.s3_b2 = ResBlock(64, 64); self.s3_b3 = ResBlock(64, 64)
        self.pool = AvgPool2d(8)
        self.fc = Linear(64, num_classes)

    def forward(self, x):
        h = self.keep(self.conv1(x)); h = self.keep(self.bn1(h)); h = self.keep(h.op_relu())
        h = self.keep(self.s1_b1(h)); h = self.keep(self.s1_b2(h)); h = self.keep(self.s1_b3(h))
        h = self.keep(self.s2_b1(h)); h = self.keep(self.s2_b2(h)); h = self.keep(self.s2_b3(h))
        h = self.keep(self.s3_b1(h)); h = self.keep(self.s3_b2(h)); h = self.keep(self.s3_b3(h))
        h = self.keep(self.pool(h)); h = self.keep(h.reshape([h.shape[0], 64]))
        return self.fc(h)


def accuracy(model, images_np, labels_np, batch_size=200):
    model.eval()
    c = 0
    t = 0
    n = images_np.shape[0]
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        bx = rp.from_numpy(images_np[s:e]).to(device=DEVICE)
        out = model(bx).to(device='cpu').to_numpy().argmax(axis=1)
        c += int((out == labels_np[s:e]).sum())
        t += (e - s)
        del bx, out
        model._intermediates.clear()
        model._clear_all_intermediates()
    gc.collect()
    model.train()
    return c / t


model = ResNet20()
model.to(device=DEVICE)
opt = SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

num_train = train_images.shape[0]
num_batches = num_train // BATCH_SIZE
if MAX_BATCHES > 0:
    num_batches = min(num_batches, MAX_BATCHES)
indices = np.arange(num_train)

print(f"Params: {sum(t.size for t in model.parameters()):,}", flush=True)
print(f"Training: {NUM_EPOCHS} epochs, batch={BATCH_SIZE}, optim=SGD lr={LR} mom={MOMENTUM} wd={WEIGHT_DECAY}, milestones={sorted(MILESTONES)}", flush=True)
print(f"{'Epoch':>5}  {'Loss':>8}  {'Acc':>8}  {'LR':>10}  {'Time':>6}", flush=True)
print("-" * 50, flush=True)

best_acc = 0.0
history = {'loss': [], 'acc': [], 'lr': []}

base_lr = opt.defaults['lr']
params = model.parameters()

for epoch in range(NUM_EPOCHS):
    model.train()
    np.random.shuffle(indices)
    tl = 0.0
    n = 0
    start = time.time()

    if SCHEDULE == 'cosine' and not (WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS):
        prog = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
        prog = min(1.0, max(0.0, prog))
        opt.defaults['lr'] = 0.5 * LR * (1.0 + math.cos(math.pi * prog))
    max_loss = 0.0
    max_norm = 0.0
    n_skipped = 0

    for b in range(num_batches):
        if WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS:
            step_frac = (epoch * num_batches + b + 1) / float(WARMUP_EPOCHS * num_batches)
            opt.defaults['lr'] = base_lr * step_frac

        bi = indices[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        bx = train_images[bi]
        by = train_labels[bi].tolist()
        bx = augment_batch(bx)
        bx = (bx - CIFAR_MEAN) / CIFAR_STD
        bg = rp.from_numpy(bx).to(device=DEVICE)

        model.zero_grad()
        out = model(bg)
        loss = out.op_cross_entropy_loss(by)
        loss_val = float(loss.to(device='cpu').sum_all())

        if not (loss_val == loss_val) or loss_val > 1e6:
            print(f"  [skip b={b}] loss={loss_val:.4g} (NaN/explode)", flush=True)
            n_skipped += 1
            del bg, out, loss
            if b % 50 == 0:
                gc.collect()
            continue

        tl += loss_val
        n += 1
        if loss_val > max_loss:
            max_loss = loss_val
        loss.backward()
        gnorm = rp.clip_grad_norm(params, GRAD_CLIP)
        if gnorm > max_norm:
            max_norm = gnorm
        opt.step()

        if LOG_EVERY > 0 and b % LOG_EVERY == 0:
            print(f"  e{epoch+1} b{b:4d}/{num_batches}  loss={loss_val:7.4f}  gnorm={gnorm:8.3f}  lr={opt.defaults['lr']:.5f}", flush=True)

        del bg, out, loss
        if b % 50 == 0:
            gc.collect()

    if WARMUP_EPOCHS > 0 and epoch + 1 == WARMUP_EPOCHS:
        opt.defaults['lr'] = base_lr

    if SCHEDULE == 'step' and (epoch + 1) in MILESTONES:
        base_lr *= LR_GAMMA
        opt.defaults['lr'] = base_lr

    elapsed = time.time() - start
    avg_loss = tl / max(n, 1)
    acc = accuracy(model, test_images_norm, test_labels)
    gc.collect()

    if n_skipped:
        print(f"  (epoch {epoch+1}: skipped {n_skipped} batches, max_loss={max_loss:.4g}, max_gnorm={max_norm:.3f})", flush=True)

    history['loss'].append(avg_loss)
    history['acc'].append(acc)
    history['lr'].append(opt.defaults['lr'])

    if acc > best_acc:
        best_acc = acc
        rp.save(model.tracked, os.path.join(OUT_DIR, 'best.rpt'))

    print(f"{epoch + 1:5d}  {avg_loss:8.4f}  {acc:7.2%}  {opt.defaults['lr']:10.6f}  gn={max_norm:7.2f}  {elapsed:5.0f}s", flush=True)

rp.save(model.tracked, os.path.join(OUT_DIR, 'final.rpt'))
np.save(os.path.join(OUT_DIR, 'history.npy'), history)

print(f"\nBest accuracy: {best_acc:.2%}", flush=True)
print(f"Outputs saved to {OUT_DIR}/", flush=True)
