import sys, time, gc
sys.path.insert(0, '..')

import recpulse_cuda as rp
from recpulse.module import Module, Linear, Conv2d, BatchNorm2d, AvgPool2d
from recpulse.optim import Adam
from recpulse.scheduler import ReduceLROnPlateau
from recpulse.data import load_cifar10, get_batch_4d

rp.manual_seed(42)
DEVICE = 'cuda'
print(f'RecPulse loaded (device={DEVICE})')

train_images, train_labels, test_images, test_labels = load_cifar10('../data/cifar10')
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f'Train: {train_images.shape}')
print(f'Test:  {test_images.shape}')

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

        self.s1_b1 = ResBlock(16, 16)
        self.s1_b2 = ResBlock(16, 16)
        self.s1_b3 = ResBlock(16, 16)

        self.s2_b1 = ResBlock(16, 32, stride=2)
        self.s2_b2 = ResBlock(32, 32)
        self.s2_b3 = ResBlock(32, 32)

        self.s3_b1 = ResBlock(32, 64, stride=2)
        self.s3_b2 = ResBlock(64, 64)
        self.s3_b3 = ResBlock(64, 64)

        self.pool = AvgPool2d(8)
        self.fc = Linear(64, num_classes)

    def forward(self, x):
        h = self.keep(self.conv1(x))
        h = self.keep(self.bn1(h))
        h = self.keep(h.op_relu())
        h = self.keep(self.s1_b1(h))
        h = self.keep(self.s1_b2(h))
        h = self.keep(self.s1_b3(h))
        h = self.keep(self.s2_b1(h))
        h = self.keep(self.s2_b2(h))
        h = self.keep(self.s2_b3(h))
        h = self.keep(self.s3_b1(h))
        h = self.keep(self.s3_b2(h))
        h = self.keep(self.s3_b3(h))
        h = self.keep(self.pool(h))
        h = self.keep(h.reshape([h.shape[0], 64]))
        return self.fc(h)


model = ResNet20()
model.to(device=DEVICE)
num_params = sum(t.size for t in model.parameters())
print(f'Parameters: {num_params:,}')

optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-6)

BATCH_SIZE = 64
NUM_EPOCHS = 5  # local test; bump on Athena for real results
num_train = train_images.shape[0]
num_batches = (num_train + BATCH_SIZE - 1) // BATCH_SIZE

print(f'Batch size: {BATCH_SIZE}')
print(f'Batches per epoch: {num_batches}')
print(f'Epochs: {NUM_EPOCHS}')

def compute_accuracy(model, images, labels, batch_size=200):
    model.eval()
    correct = 0
    total = 0
    for b in range((len(labels) + batch_size - 1) // batch_size):
        batch_img, batch_lbl = get_batch_4d(images, labels, b, batch_size)
        if batch_img is None:
            break
        if DEVICE != 'cpu':
            batch_img = batch_img.to(device=DEVICE)
        out = model(batch_img)
        if DEVICE != 'cpu':
            out = out.to(device='cpu')
        preds = out.to_numpy().argmax(axis=1)
        for i in range(len(preds)):
            if preds[i] == batch_lbl[i]:
                correct += 1
            total += 1
        del batch_img, out
    model.train()
    return correct / total

history = {'loss': [], 'acc': [], 'lr': []}

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_loss = 0
    start = time.time()

    for b in range(num_batches):
        batch_img, batch_lbl = get_batch_4d(train_images, train_labels, b, BATCH_SIZE)
        if batch_img is None:
            break
        if DEVICE != 'cpu':
            batch_img = batch_img.to(device=DEVICE)

        model.zero_grad()
        out = model(batch_img)
        loss = out.op_cross_entropy_loss(batch_lbl)
        epoch_loss += float(loss.to(device='cpu').sum_all())
        n_loss += 1
        loss.backward()
        rp.clip_grad_norm(model.parameters(), 2.0)
        optimizer.step()
        del batch_img, out, loss
        if b % 50 == 0:
            gc.collect()

    elapsed = time.time() - start
    avg_loss = epoch_loss / n_loss
    history['loss'].append(avg_loss)
    history['lr'].append(scheduler.get_lr())

    acc = compute_accuracy(model, test_images, test_labels)
    history['acc'].append(acc)
    scheduler.step(avg_loss)
    gc.collect()

    print(f'Epoch {epoch+1:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}  acc={acc:.2%}  lr={scheduler.get_lr():.6f}  ({elapsed:.0f}s)', flush=True)

best_acc = max(history['acc'])
best_epoch = history['acc'].index(best_acc) + 1

print(f'Best test accuracy: {best_acc:.2%} (epoch {best_epoch})')
print(f'Final test accuracy: {history["acc"][-1]:.2%}')
print(f'Final training loss: {history["loss"][-1]:.4f}')
print()
print('Training history:')
print(f"{'Epoch':>5}  {'Loss':>8}  {'Acc':>8}  {'LR':>10}")
print('-' * 40)
for i in range(len(history['loss'])):
    print(f"{i+1:5d}  {history['loss'][i]:8.4f}  {history['acc'][i]:7.2%}  {history['lr'][i]:10.6f}")

rp.save(model.tracked, '../data/cifar10_resnet.rpt')
print('Model saved to data/cifar10_resnet.rpt')
