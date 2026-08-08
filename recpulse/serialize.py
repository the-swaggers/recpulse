import recpulse_cuda as rp
import numpy as np


def save(state_dict, path):
    if path.endswith('.safetensors'):
        save_safetensors(state_dict, path)
    else:
        rp.save(state_dict, path)


def load(path):
    if path.endswith('.safetensors'):
        return load_safetensors(path)
    else:
        return rp.load(path)


def save_safetensors(state_dict, path):
    from safetensors.numpy import save_file

    np_dict = {}
    for name, tensor in state_dict.items():
        np_dict[name] = tensor.to_numpy()

    save_file(np_dict, path)


def load_safetensors(path):
    from safetensors.numpy import load_file

    np_dict = load_file(path)
    result = {}
    for name, arr in np_dict.items():
        dtype_str = 'float32'
        if arr.dtype == np.float64:
            dtype_str = 'float64'
        elif arr.dtype == np.float16:
            dtype_str = 'float16'

        flat = arr.flatten().tolist()
        base = rp.values(flat, dtype=dtype_str)

        if len(arr.shape) > 1:
            view = base.reshape(list(arr.shape))
            tensor = view.copy()
        else:
            tensor = base.copy()

        result[name] = tensor

    return result
