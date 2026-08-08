# Permute Function Implementation

## Overview

The `permute` function reorders tensor dimensions according to a specified permutation array. It's a generalized version of `transpose` that can rearrange any number of dimensions in a single operation. This is always a zero-copy operation - it only updates shape and strides metadata.

## API

### C API (functional layer)
```c
Tensor* rp_permute(Tensor* src, int* dims);
```

### C API (ops layer with autograd support)
```c
Tensor* op_permute(Tensor* src, int* dims);
```

### Python API
```python
tensor.permute(dims)
```

## Parameters

- `src` / `tensor`: Input tensor
- `dims`: Array/list of dimension indices in the new order
  - Must have length equal to tensor's ndim
  - Must contain each dimension exactly once
  - Supports negative indexing

## Behavior

- Reorders dimensions according to the `dims` array
- Always zero-copy (creates a view with reordered shape/strides)
- Validates that all dimensions appear exactly once
- `transpose(i, j)` is a special case of `permute([..., j, ..., i, ...])`

## Examples

### C Example
```c
int shape[] = {2, 3, 4};
Tensor* t = ones_host_tensor(DTYPE_FLOAT32, 3, shape, NULL);

// Permute to [4, 2, 3] - last dim becomes first
int dims1[] = {2, 0, 1};
Tensor* permuted1 = rp_permute(t, dims1);

// Reverse all dimensions: [2, 3, 4] -> [4, 3, 2]
int dims2[] = {2, 1, 0};
Tensor* permuted2 = rp_permute(t, dims2);

// Identity (no change): [2, 3, 4] -> [2, 3, 4]
int dims3[] = {0, 1, 2};
Tensor* permuted3 = rp_permute(t, dims3);
```

### Python Example
```python
import recpulse_cuda as rp

t = rp.zeros([2, 3, 4], dtype='float32', device='cpu')

# Basic permutation: [2, 3, 4] -> [4, 2, 3]
permuted = t.permute([2, 0, 1])

# Same as transpose(1, 2): [2, 3, 4] -> [2, 4, 3]
permuted = t.permute([0, 2, 1])

# Reverse all dimensions: [2, 3, 4] -> [4, 3, 2]
permuted = t.permute([2, 1, 0])

# Using negative indices: [2, 3, 4] -> [4, 2, 3]
permuted = t.permute([-1, 0, 1])

# 4D example - complex reordering
t2 = rp.zeros([2, 3, 4, 5])
permuted = t2.permute([3, 1, 0, 2])  # -> [5, 3, 2, 4]
```

## Comparison with Transpose

| Operation | Flexibility | Use Case |
|-----------|-------------|----------|
| `transpose(i, j)` | Swaps two dimensions only | Simple 2D transpose or swapping two dims |
| `permute(dims)` | Reorders any number of dims | Complex dimension reordering, batch reordering |

**Example equivalence:**
```python
# These are equivalent:
t.transpose(1, 2)        # Swap dims 1 and 2
t.permute([0, 2, 1])     # Reorder to swap dims 1 and 2
```

## Implementation Details

### Files Modified
- `functional/functional.h` - Added declaration
- `functional/functional.c` - Implemented `rp_permute`
- `ops/ops.h` - Added declaration for autograd wrapper
- `ops/ops.c` - Implemented `op_permute` with autograd support
- `python/tensor_python.c` - Added Python bindings
- `Makefile` - Added test_permute target
- `.claude/rules/shape_ops.md` - Updated documentation

### Algorithm
1. Validate dims array:
   - Handle negative indices
   - Check all values are in valid range [0, ndim)
   - Ensure each dimension appears exactly once (no duplicates or missing dims)
2. Create new shape and strides arrays by reordering according to dims
3. Create view tensor with:
   - Same data pointer (zero-copy)
   - Reordered shape and strides
   - owns_data = false
   - Pointer to base tensor

### Performance
- **Always zero-copy** - only metadata (shape/strides) is updated
- **O(ndim)** time complexity for validation and reordering
- **O(ndim)** space complexity for new shape/strides arrays

### Error Handling
The function validates:
- Null pointers
- Dimension bounds (each dim in [0, ndim))
- No duplicate dimensions
- All dimensions present

## Use Cases

### 1. Batch First ↔ Channel First
```python
# Convert from [batch, height, width, channels] to [batch, channels, height, width]
x = rp.zeros([32, 224, 224, 3])
x_channels_first = x.permute([0, 3, 1, 2])  # -> [32, 3, 224, 224]
```

### 2. Matrix Transpose in Batched Operations
```python
# Transpose matrices in a batch: [batch, m, n] -> [batch, n, m]
batch_matrices = rp.zeros([16, 10, 20])
transposed = batch_matrices.permute([0, 2, 1])  # -> [16, 20, 10]
```

### 3. Dimension Reordering for Operations
```python
# Prepare for operation that expects different dimension order
x = rp.zeros([10, 20, 30, 40])
# Bring dimension 3 to front for easier processing
x_reordered = x.permute([3, 0, 1, 2])  # -> [40, 10, 20, 30]
```

## Testing

### C Test
```bash
make test_permute
```

Tests include:
- Basic permutation
- Permute as transpose
- Reverse all dimensions
- Negative indices
- Identity permutation
- Invalid permutation (duplicate dimension)

### Python Test
```bash
python3 test_permute.py
```

## Status
✅ Fully implemented and tested
- C functional layer implementation
- C ops layer with autograd support
- Python bindings
- Unit tests (C and Python)
- Zero-copy view operation
- Full validation of permutation array
