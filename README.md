# RecPulse

**Version 0.2.0b1** — a machine learning framework built from scratch:
C/CUDA tensor backend with autograd, exposed to Python through CPython bindings.

This project aims to create a new machine learning framework. Initially the goal was to learn to build a framework for fun and to implement weird ideas for some models, to which I haven't got. Currently my main goal is to deeply understand how the large frameworks like PyTorch, JAX and TF work.

## What's implemented

- **Tensor core** (C/CUDA): float64/float32/float16/bfloat16, CPU + multi-GPU with P2P, views/strides
- **Autograd**: 55+ ops with backward passes (element-wise, broadcasting binary ops, matmul, conv2d, pooling, reductions, shape ops, softmax, losses), dependency-counting topological sort
- **Framework layer** (Python): `Module`, Linear/Conv2d/BatchNorm2d/LayerNorm/Dropout/Embedding/pooling/LSTMCell/GRUCell, SGD + Adam optimizers, LR schedulers, gradient clipping
- **Serialization**: native `.rpt` checkpoints + safetensors, NumPy interop
- **BPE tokenizer** in C
- **Working examples**: MNIST MLP (98.1%), CIFAR-10 CNN, ResNet20, MNIST GAN, char-level LSTM text generation

## Quick start

```bash
cd cuda_utils
make clean && make        # builds recpulse_cuda.so (needs CUDA toolkit + Python headers)
make test_all             # run the test suite
```

```python
import recpulse_cuda as rp
from recpulse.module import Module, Linear
from recpulse.optim import Adam

x = rp.randn([32, 16], device='cuda').requires_grad_()
loss = x.op_square().op_mean_all()
loss.backward()
```

See `cuda_utils/examples/` for full training scripts and notebooks.

## Status

This is a 0.x beta: APIs and internals are still unstable, but every defect from the 0.1.0
audit ([`docs/RELEASE_READINESS_0.1.0.md`](docs/RELEASE_READINESS_0.1.0.md)) is fixed and the
full stress suite — numerical gradcheck, adversarial edge cases, memory-leak loops, PyTorch
parity, end-to-end training — is green on CPU and GPU (59/59 test files, A100). Tensor
lifetime is reference-counted (chaining ops and inline arguments are safe), non-contiguous
views compute correctly everywhere, and optimizers/losses/normalization match PyTorch.
Change history: [`CHANGELOG.md`](CHANGELOG.md).
