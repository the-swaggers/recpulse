import sys
import os
import time
import gc
sys.path.insert(0, '..')

import numpy as np
import recpulse_cuda as rp
from recpulse.module import Module, Linear, Dropout
from recpulse.optim import Adam
from recpulse.data import load_mnist, get_batch

rp.manual_seed(42)
DEVICE = 'cuda'
LATENT = 100
BATCH_SIZE = 128
NUM_EPOCHS = 200
SAMPLE_EVERY = 10

OUT_DIR = '../data/mnist_gan_run'
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading MNIST...", flush=True)
train_images, _, _, _ = load_mnist('../data/mnist')
arr = train_images.to_numpy() * 2.0 - 1.0
train_images = rp.from_numpy(arr.astype(np.float32))
train_labels = [0] * train_images.shape[0]
print(f"Train: {train_images.shape}", flush=True)


class Generator(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(LATENT, 256)
        self.fc2 = Linear(256, 512)
        self.fc3 = Linear(512, 1024)
        self.fc4 = Linear(1024, 784)

    def forward(self, z):
        h = self.keep(self.fc1(z))
        h = self.keep(h.op_leaky_relu(0.2))
        h = self.keep(self.fc2(h))
        h = self.keep(h.op_leaky_relu(0.2))
        h = self.keep(self.fc3(h))
        h = self.keep(h.op_leaky_relu(0.2))
        h = self.keep(self.fc4(h))
        return self.keep(h.op_tanh())


class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 512)
        self.drop1 = Dropout(0.3)
        self.fc2 = Linear(512, 256)
        self.drop2 = Dropout(0.3)
        self.fc3 = Linear(256, 1)

    def forward(self, x):
        h = self.keep(self.fc1(x))
        h = self.keep(h.op_leaky_relu(0.2))
        h = self.keep(self.drop1(h))
        h = self.keep(self.fc2(h))
        h = self.keep(h.op_leaky_relu(0.2))
        h = self.keep(self.drop2(h))
        h = self.keep(self.fc3(h))
        return self.keep(h.reshape([h.shape[0]]))


def save_sample_grid(generator, epoch, n=16, side=4):
    generator.eval()
    z = rp.randn([n, LATENT], device=DEVICE)
    samples = generator(z).to(device='cpu').to_numpy()
    samples = (samples + 1) / 2
    samples = samples.reshape(n, 28, 28).clip(0, 1)
    grid = np.zeros((side * 28, side * 28), dtype=np.float32)
    for i in range(n):
        r, c = i // side, i % side
        grid[r*28:(r+1)*28, c*28:(c+1)*28] = samples[i]
    np.save(os.path.join(OUT_DIR, f'samples_epoch_{epoch:03d}.npy'), grid)
    generator.train()


G = Generator(); G.to(device=DEVICE)
D = Discriminator(); D.to(device=DEVICE)
optG = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

g_params = sum(t.size for t in G.parameters())
d_params = sum(t.size for t in D.parameters())
print(f"G params: {g_params:,}  D params: {d_params:,}", flush=True)

ones_dev = rp.from_numpy(np.ones(BATCH_SIZE, dtype=np.float32)).to(device=DEVICE)
zeros_dev = rp.from_numpy(np.zeros(BATCH_SIZE, dtype=np.float32)).to(device=DEVICE)

num_train = train_images.shape[0]
num_batches = num_train // BATCH_SIZE
print(f"Batch size {BATCH_SIZE}, batches/epoch {num_batches}, epochs {NUM_EPOCHS}", flush=True)
print(f"{'Epoch':>5}  {'D loss':>8}  {'G loss':>8}  {'Time':>6}", flush=True)
print("-" * 40, flush=True)

save_sample_grid(G, 0)

history = {'d_loss': [], 'g_loss': []}
for epoch in range(NUM_EPOCHS):
    G.train(); D.train()
    d_total = 0.0; g_total = 0.0; n = 0
    start = time.time()

    for b in range(num_batches):
        real_cpu, _ = get_batch(train_images, train_labels, b, BATCH_SIZE)
        if real_cpu is None or real_cpu.shape[0] != BATCH_SIZE:
            break
        real = real_cpu.to(device=DEVICE)

        D.zero_grad(); G.zero_grad()
        z1 = rp.randn([BATCH_SIZE, LATENT], device=DEVICE)
        fake1 = G(z1)
        d_real = D(real)
        d_fake = D(fake1)
        loss_dr = d_real.op_bce_loss(ones_dev)
        loss_df = d_fake.op_bce_loss(zeros_dev)
        d_loss = loss_dr.op_add(loss_df)
        d_total += float(d_loss.to(device='cpu').sum_all())
        d_loss.backward()
        optD.step()

        D.zero_grad(); G.zero_grad()
        z2 = rp.randn([BATCH_SIZE, LATENT], device=DEVICE)
        fake2 = G(z2)
        d_fake2 = D(fake2)
        g_loss = d_fake2.op_bce_loss(ones_dev)
        g_total += float(g_loss.to(device='cpu').sum_all())
        g_loss.backward()
        optG.step()

        n += 1
        if b % 50 == 0:
            gc.collect()

    elapsed = time.time() - start
    history['d_loss'].append(d_total / n)
    history['g_loss'].append(g_total / n)
    print(f"{epoch+1:5d}  {d_total/n:8.4f}  {g_total/n:8.4f}  {elapsed:5.0f}s", flush=True)

    if (epoch + 1) % SAMPLE_EVERY == 0 or (epoch + 1) == NUM_EPOCHS:
        save_sample_grid(G, epoch + 1)

rp.save(G.tracked, os.path.join(OUT_DIR, 'G_final.rpt'))
rp.save(D.tracked, os.path.join(OUT_DIR, 'D_final.rpt'))
np.save(os.path.join(OUT_DIR, 'history.npy'),
        {'d_loss': history['d_loss'], 'g_loss': history['g_loss']})

print(f"\nDone. Outputs in {OUT_DIR}/", flush=True)
