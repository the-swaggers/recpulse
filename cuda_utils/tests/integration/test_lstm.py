import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np
import recpulse_cuda as rp
from recpulse.module import LSTMCell, GRUCell, Linear, Module
from recpulse.optim import Adam


def test_lstm_shapes():
    cell = LSTMCell(8, 16)
    x = rp.from_numpy(np.random.randn(4, 8).astype(np.float32))
    h, c = cell(x)
    assert h.shape == (4, 16), f"h shape {h.shape}"
    assert c.shape == (4, 16), f"c shape {c.shape}"
    h2, c2 = cell(x, (h, c))
    assert h2.shape == (4, 16)
    print("PASS: LSTMCell forward shapes")


def test_gru_shapes():
    cell = GRUCell(8, 16)
    x = rp.from_numpy(np.random.randn(4, 8).astype(np.float32))
    h = cell(x)
    assert h.shape == (4, 16), f"h shape {h.shape}"
    h2 = cell(x, h)
    assert h2.shape == (4, 16)
    print("PASS: GRUCell forward shapes")


def test_lstm_grad_flow():
    cell = LSTMCell(4, 8)
    x = rp.from_numpy(np.random.randn(2, 4).astype(np.float32))
    h, c = cell(x)
    h2, c2 = cell(x, (h, c))
    loss = h2.op_square().op_sum_all()
    loss.backward()
    for name, p in cell.learnable().items():
        assert p.has_grad, f"no grad for {name}"
        g = np.abs(p.grad.to_numpy()).sum()
        assert g > 0, f"zero grad for {name}"
    print("PASS: LSTM gradients flow through 2 unrolled steps")


def test_lstm_learns():
    np.random.seed(0)
    rp.manual_seed(0)

    class TinyLSTM(Module):
        def __init__(self):
            super().__init__()
            self.cell = LSTMCell(2, 16)
            self.head = Linear(16, 1)

        def forward(self, steps):
            state = None
            for x in steps:
                h, c = self.cell(x, state)
                state = (h, c)
            return self.head(state[0])

    model = TinyLSTM()
    opt = Adam(model.parameters(), lr=0.01)

    seqs = np.random.randn(8, 3, 2).astype(np.float32)
    targets = seqs.sum(axis=(1, 2), keepdims=False).reshape(8, 1).astype(np.float32)

    steps = [rp.from_numpy(seqs[:, t, :]) for t in range(3)]
    tgt = rp.from_numpy(targets)

    first_loss = None
    last_loss = None
    for it in range(150):
        opt.zero_grad()
        out = model(steps)
        loss = out.op_mse_loss(tgt, 'mean')
        val = loss.sum_all()
        if first_loss is None:
            first_loss = val
        last_loss = val
        loss.backward()
        opt.step()

    assert last_loss < first_loss * 0.2, f"LSTM did not learn: {first_loss} -> {last_loss}"
    print(f"PASS: LSTM learns sequence sum ({first_loss:.4f} -> {last_loss:.4f})")


def test_lstm_torch_parity():
    try:
        import torch
    except ImportError:
        print("SKIP: torch not available")
        return

    np.random.seed(1)
    B, I, H = 3, 5, 7
    x_np = np.random.randn(B, I).astype(np.float64)
    h_np = np.random.randn(B, H).astype(np.float64)
    c_np = np.random.randn(B, H).astype(np.float64)
    w_ih = np.random.randn(I, 4 * H).astype(np.float64) * 0.3
    w_hh = np.random.randn(H, 4 * H).astype(np.float64) * 0.3
    b_ih = np.random.randn(4 * H).astype(np.float64) * 0.1
    b_hh = np.random.randn(4 * H).astype(np.float64) * 0.1

    cell = LSTMCell(I, H)
    cell.weight_ih.copy_(rp.from_numpy(w_ih))
    cell.weight_ih.requires_grad_(True)
    cell.weight_hh.copy_(rp.from_numpy(w_hh))
    cell.weight_hh.requires_grad_(True)
    cell.bias_ih.copy_(rp.from_numpy(b_ih))
    cell.bias_ih.requires_grad_(True)
    cell.bias_hh.copy_(rp.from_numpy(b_hh))
    cell.bias_hh.requires_grad_(True)
    cell.tracked = {"weight_ih": cell.weight_ih, "weight_hh": cell.weight_hh,
                    "bias_ih": cell.bias_ih, "bias_hh": cell.bias_hh}

    x = rp.from_numpy(x_np)
    h0 = rp.from_numpy(h_np)
    c0 = rp.from_numpy(c_np)
    h1, c1 = cell(x, (h0, c0))

    tcell = torch.nn.LSTMCell(I, H).double()
    with torch.no_grad():
        tcell.weight_ih.copy_(torch.from_numpy(w_ih.T.copy()))
        tcell.weight_hh.copy_(torch.from_numpy(w_hh.T.copy()))
        tcell.bias_ih.copy_(torch.from_numpy(b_ih))
        tcell.bias_hh.copy_(torch.from_numpy(b_hh))
    th1, tc1 = tcell(torch.from_numpy(x_np),
                     (torch.from_numpy(h_np), torch.from_numpy(c_np)))

    err_h = np.abs(h1.to_numpy() - th1.detach().numpy()).max()
    err_c = np.abs(c1.to_numpy() - tc1.detach().numpy()).max()
    assert err_h < 1e-10, f"h mismatch vs torch: {err_h}"
    assert err_c < 1e-10, f"c mismatch vs torch: {err_c}"
    print(f"PASS: LSTMCell matches torch.nn.LSTMCell (h err {err_h:.2e}, c err {err_c:.2e})")


def test_gru_learns():
    np.random.seed(2)
    rp.manual_seed(2)
    cell = GRUCell(2, 12)
    head = Linear(12, 1)
    params = cell.parameters() + head.parameters()
    opt = Adam(params, lr=0.01)

    seqs = np.random.randn(8, 3, 2).astype(np.float32)
    targets = seqs.sum(axis=(1, 2)).reshape(8, 1).astype(np.float32)
    steps = [rp.from_numpy(seqs[:, t, :]) for t in range(3)]
    tgt = rp.from_numpy(targets)

    first_loss = None
    last_loss = None
    for it in range(150):
        for p in params:
            p.zero_grad()
        h = None
        for x in steps:
            h = cell(x, h)
        out = head(h)
        loss = out.op_mse_loss(tgt, 'mean')
        val = loss.sum_all()
        if first_loss is None:
            first_loss = val
        last_loss = val
        loss.backward()
        opt.step()

    assert last_loss < first_loss * 0.2, f"GRU did not learn: {first_loss} -> {last_loss}"
    print(f"PASS: GRU learns sequence sum ({first_loss:.4f} -> {last_loss:.4f})")


if __name__ == "__main__":
    test_lstm_shapes()
    test_gru_shapes()
    test_lstm_grad_flow()
    test_lstm_learns()
    test_lstm_torch_parity()
    test_gru_learns()
    print("All LSTM/GRU tests passed")
