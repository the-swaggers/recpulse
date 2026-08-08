import sys
import os
sys.path.insert(0, '.')
import recpulse_cuda as rp
from recpulse.module import Module, Linear
from recpulse.serialize import save, load

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

def approx(a, b, tol=1e-4):
    if abs(b) < 1e-8:
        return abs(a - b) < tol
    return abs(a - b) / max(abs(b), 1e-8) < tol

rp.manual_seed(42)

print("=" * 60)
print("Serialization Tests")
print("=" * 60)

print("\n--- rp.save / rp.load (.rpt format) ---")

t1 = rp.values([1.0, 2.0, 3.0, 4.0])
t2 = rp.randn([2, 3])
state = {"weight": t1, "bias": t2}

path_rpt = "/tmp/test_model.rpt"
rp.save(state, path_rpt)
check("rpt save", os.path.exists(path_rpt))

loaded = rp.load(path_rpt)
check("rpt load type", isinstance(loaded, dict))
check("rpt load keys", set(loaded.keys()) == {"weight", "bias"})
check("rpt weight shape", loaded["weight"].shape == (4,))
check("rpt bias shape", loaded["bias"].shape == (2, 3))
check("rpt weight values", approx(loaded["weight"].sum_all(), 10.0))

t2_copy = t2.copy()
check("rpt bias values", approx(loaded["bias"].sum_all(), t2_copy.sum_all()))

os.remove(path_rpt)

print("\n--- rp.save / rp.load roundtrip ---")

rp.manual_seed(123)
state2 = {
    "fc1.weight": rp.randn([4, 8]),
    "fc1.bias": rp.zeros([8]),
    "fc2.weight": rp.randn([8, 2]),
    "fc2.bias": rp.zeros([2]),
}

path2 = "/tmp/test_roundtrip.rpt"
rp.save(state2, path2)
loaded2 = rp.load(path2)

for key in state2:
    orig = state2[key].copy()
    load_t = loaded2[key]
    check(f"roundtrip {key} shape", orig.shape == load_t.shape)
    check(f"roundtrip {key} dtype", orig.dtype == load_t.dtype)
    check(f"roundtrip {key} values", approx(orig.sum_all(), load_t.sum_all()))

os.remove(path2)

print("\n--- Float64 ---")

t64 = rp.randn([3, 3], dtype='float64')
path64 = "/tmp/test_f64.rpt"
rp.save({"w": t64}, path64)
loaded64 = rp.load(path64)
check("f64 dtype preserved", loaded64["w"].dtype == 'float64')
t64_copy = t64.copy()
check("f64 values", approx(loaded64["w"].sum_all(), t64_copy.sum_all()))
os.remove(path64)

print("\n--- Float16 ---")

t16 = rp.randn([3, 3], dtype='float16')
path16 = "/tmp/test_f16.rpt"
rp.save({"w": t16}, path16)
loaded16 = rp.load(path16)
check("f16 dtype preserved", loaded16["w"].dtype == 'float16')
os.remove(path16)

print("\n--- to_numpy ---")

import numpy as np
t = rp.values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
_tb = t
t2d = t.reshape([2, 3])
arr = t2d.to_numpy()
check("to_numpy type", isinstance(arr, np.ndarray))
check("to_numpy shape", arr.shape == (2, 3))
check("to_numpy values", np.allclose(arr, [[1, 2, 3], [4, 5, 6]]))
check("to_numpy dtype", arr.dtype == np.float32)

t64 = rp.values([1.0, 2.0], dtype='float64')
arr64 = t64.to_numpy()
check("to_numpy f64", arr64.dtype == np.float64)

print("\n--- safetensors save / load ---")

try:
    path_st = "/tmp/test_model.safetensors"
    t1 = rp.values([1.0, 2.0, 3.0])
    t2_base = rp.randn([2, 4])
    state_st = {"weight": t1, "bias": t2_base}

    save(state_st, path_st)
    check("safetensors save", os.path.exists(path_st))

    loaded_st = load(path_st)
    check("safetensors load keys", set(loaded_st.keys()) == {"weight", "bias"})
    check("safetensors weight shape", loaded_st["weight"].shape == (3,))
    check("safetensors bias shape", loaded_st["bias"].shape == (2, 4))
    t1_copy = t1.copy()
    check("safetensors weight values", approx(loaded_st["weight"].sum_all(), t1_copy.sum_all()))

    os.remove(path_st)
except ImportError:
    print("  SKIP: safetensors not installed")

print("\n--- Module save / load ---")

rp.manual_seed(456)

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

x_test = rp.randn([3, 4])
out_before = model(x_test)
loss_before = out_before.sum_all()

path_model = "/tmp/test_mlp.rpt"
rp.save(model.tracked, path_model)

model2 = MLP()
loaded_state = rp.load(path_model)
model2.load_state(loaded_state)

out_after = model2(x_test)
loss_after = out_after.sum_all()

check("model save/load same output", approx(loss_before, loss_after))

w1_orig = model.tracked["fc1.weight"].copy()
w1_loaded = model2.tracked["fc1.weight"].copy()
check("model weights match", approx(w1_orig.sum_all(), w1_loaded.sum_all()))

check("loaded weights have requires_grad", model2.fc1.weight.requires_grad)
check("loaded bias has requires_grad", model2.fc1.bias.requires_grad)

os.remove(path_model)

print("\n--- Module save as safetensors ---")

try:
    path_mlp_st = "/tmp/test_mlp.safetensors"
    save(model.tracked, path_mlp_st)

    model3 = MLP()
    loaded_st2 = load(path_mlp_st)
    model3.load_state(loaded_st2)

    out_st = model3(x_test)
    check("safetensors model output matches", approx(loss_before, out_st.sum_all(), 0.01))

    os.remove(path_mlp_st)
except ImportError:
    print("  SKIP: safetensors not installed")

print("\n--- Auto format detection ---")

rp.save({"t": rp.ones([3])}, "/tmp/test_auto.rpt")
loaded_auto = rp.load("/tmp/test_auto.rpt")
check("auto detect .rpt", approx(loaded_auto["t"].sum_all(), 3.0))
os.remove("/tmp/test_auto.rpt")

try:
    save({"t": rp.ones([3])}, "/tmp/test_auto.safetensors")
    loaded_auto_st = load("/tmp/test_auto.safetensors")
    check("auto detect .safetensors", approx(loaded_auto_st["t"].sum_all(), 3.0))
    os.remove("/tmp/test_auto.safetensors")
except ImportError:
    print("  SKIP: safetensors not installed")

print("\n" + "=" * 60)
print(f"Results: {passed}/{passed + failed} tests passed")
if failed > 0:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("All serialization tests passed!")
