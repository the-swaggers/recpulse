import sys
import os
import time
sys.path.insert(0, '..')

import numpy as np
import recpulse_cuda as rp
from recpulse.module import Module, LSTMCell, Linear, Embedding
from recpulse.optim import Adam

rp.manual_seed(42)
np.random.seed(42)

DEVICE = 'cuda'
EMBED_DIM = 64
HIDDEN = 256
SEQ_LEN = 40
BATCH_SIZE = 64
NUM_ITERS = 4000
LR = 2e-3
SAMPLE_EVERY = 500
SAMPLE_LEN = 300

DATA_PATH = '../data/tinyshakespeare.txt'
OUT_DIR = '../data/char_lstm_run'
os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int64)
n_train = int(0.95 * len(data))
train_data = data[:n_train]
val_data = data[n_train:]
print(f"Corpus: {len(text)} chars, vocab {vocab}", flush=True)


class CharLSTM(Module):
    def __init__(self, vocab, embed_dim, hidden):
        super().__init__()
        self.embed = Embedding(vocab, embed_dim)
        self.cell = LSTMCell(embed_dim, hidden)
        self.head = Linear(hidden, vocab)

    def forward(self, idx_steps):
        state = None
        logits = []
        for idx in idx_steps:
            x = self.embed(idx)
            h, c = self.cell(x, state)
            state = (h, c)
            logits.append(self.head(h))
        return logits


model = CharLSTM(vocab, EMBED_DIM, HIDDEN)
model.to(device=DEVICE)
opt = Adam(model.parameters(), lr=LR)
print(f"Params: {sum(p.size for p in model.parameters())}", flush=True)


def get_batch(source):
    starts = np.random.randint(0, len(source) - SEQ_LEN - 1, size=BATCH_SIZE)
    x = np.stack([source[s:s + SEQ_LEN] for s in starts])
    y = np.stack([source[s + 1:s + SEQ_LEN + 1] for s in starts])
    return x, y


def sequence_loss(x_np, y_np):
    idx_steps = [x_np[:, t].tolist() for t in range(x_np.shape[1])]
    logits = model(idx_steps)
    total = None
    for t, lg in enumerate(logits):
        l_t = lg.op_cross_entropy_loss(y_np[:, t].tolist(), 'mean', 1)
        total = l_t if total is None else total.op_add(l_t)
    return total.op_div_scalar(float(len(logits)))


def sample(prefix='\n', length=SAMPLE_LEN, temperature=0.8):
    idx = [stoi.get(c, 0) for c in prefix]
    state = None
    out_chars = list(prefix)
    for c in idx:
        x = model.embed([c])
        h, cc = model.cell(x, state)
        state = (h, cc)
    cur = idx[-1]
    for _ in range(length):
        x = model.embed([cur])
        h, cc = model.cell(x, state)
        state = (h, cc)
        logits = model.head(h).to_numpy().reshape(-1).astype(np.float64)
        logits = logits / temperature
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        cur = int(np.random.choice(vocab, p=p))
        out_chars.append(itos[cur])
    return ''.join(out_chars)


print("Training...", flush=True)
t0 = time.time()
losses = []
for it in range(1, NUM_ITERS + 1):
    x_np, y_np = get_batch(train_data)
    model.zero_grad()
    loss = sequence_loss(x_np, y_np)
    loss_val = loss.sum_all()
    loss.backward()
    opt.step()
    losses.append(loss_val)

    if it % 100 == 0:
        avg = sum(losses[-100:]) / len(losses[-100:])
        print(f"iter {it:5d} | loss {avg:.4f} | {(time.time() - t0):.0f}s", flush=True)

    if it % SAMPLE_EVERY == 0:
        vx, vy = get_batch(val_data)
        vloss = sequence_loss(vx, vy).sum_all()
        print(f"--- val loss {vloss:.4f}", flush=True)
        txt = sample()
        print(f"--- sample @ iter {it} ---\n{txt}\n--------------------", flush=True)
        with open(os.path.join(OUT_DIR, f'sample_{it:05d}.txt'), 'w') as f:
            f.write(txt)

np.save(os.path.join(OUT_DIR, 'losses.npy'), np.array(losses))
rp.save(model.tracked, os.path.join(OUT_DIR, 'char_lstm.rpt'))
print(f"Done in {(time.time() - t0) / 60:.1f} min. Final loss {sum(losses[-100:]) / 100:.4f}", flush=True)
