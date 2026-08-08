# Flatten Function Implementation

## Overview

The `flatten` function flattens dimensions from `start_dim` to `end_dim` (inclusive) into a single dimension. It's a zero-copy operation when the tensor is contiguous, otherwise it creates a contiguous copy.

## API

### C API (functional layer)
```c
Tensor* rp_flatten(Tensor* src, int start_dim, int end_dim);
```

### C API (ops layer with autograd support)
```c
Tensor* op_flatten(Tensor* src, int start_dim, int end_dim);
```

### Python API
```python
tensor.flatten(start_dim=0, end_dim=-1)
```

## Parameters

- `src` / `tensor`: Input tensor
- `start_dim`: First dimension to flatten (default: 0)
  - Supports negative indexing
- `end_dim`: Last dimension to flatten (default: -1, meaning last dimension)
  - Supports negative indexing

## Behavior

- Flattens dimensions from `start_dim` to `end_dim` (inclusive) into a single dimension
- All dimensions before `start_dim` and after `end_dim` remain unchanged
- Uses `rp_reshape` internally, which is zero-copy if contiguous

## Examples

### C Example
```c
int shape[] = {2, 3, 4};
Tensor* t = ones_host_tensor(DTYPE_FLOAT32, 3, shape, NULL);

// Flatten first two dimensions: [2, 3, 4] -> [6, 4]
Tensor* flat = rp_flatten(t, 0, 1);

// Flatten all dimensions: [2, 3, 4] -> [24]
Tensor* flat_all = rp_flatten(t, 0, -1);
```

### Python Example
```python
import recpulse_cuda as rp

t = rp.zeros([2, 3, 4], dtype='float32', device='cpu')

# Flatten first two dimensions: [2, 3, 4] -> [6, 4]
flat = t.flatten(0, 1)

# Flatten all dimensions: [2, 3, 4] -> [24]
flat_all = t.flatten(0, -1)

# Flatten middle dimensions: [2, 3, 4, 5] -> [2, 12, 5]
t2 = rp.zeros([2, 3, 4, 5])
flat_mid = t2.flatten(1, 2)
```

## Implementation Details

### Files Modified
- `functional/functional.h` - Added declaration
- `functional/functional.c` - Implemented `rp_flatten`
- `ops/ops.h` - Added declaration for autograd wrapper
- `ops/ops.c` - Implemented `op_flatten` with autograd support
- `python/tensor_python.c` - Added Python bindings
- `Makefile` - Added test_flatten target
- `.claude/rules/shape_ops.md` - Updated documentation

### Algorithm
1. Handle negative indices for `start_dim` and `end_dim`
2. Validate dimension bounds
3. Calculate the flattened size (product of dimensions from `start_dim` to `end_dim`)
4. Construct new shape:
   - Keep dimensions before `start_dim`
   - Single flattened dimension
   - Keep dimensions after `end_dim`
5. Use `rp_reshape` to create the result (zero-copy if contiguous)

### Performance
- **Zero-copy** when tensor is contiguous (just updates shape/strides metadata)
- **Requires copy** when tensor is non-contiguous (creates contiguous version first)

## Testing

### C Test
```bash
make test_flatten
```

### Python Test
```bash
python3 test_flatten.py
```

## Status
✅ Fully implemented and tested
- C functional layer implementation
- C ops layer with autograd support
- Python bindings
- Unit tests (C and Python)
