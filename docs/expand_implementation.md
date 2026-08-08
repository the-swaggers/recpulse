# Expand Function Implementation

## Overview

The `expand` function broadcasts a tensor to a larger shape by expanding dimensions of size 1 to larger sizes. This is always a zero-copy operation - it creates a view with stride=0 for broadcasted dimensions, enabling efficient memory-sharing for broadcasting operations.

## API

### C API (functional layer)
```c
Tensor* rp_expand(Tensor* src, int ndim, int* shape);
```

### C API (ops layer with autograd support)
```c
Tensor* op_expand(Tensor* src, int ndim, int* shape);
```

### Python API
```python
tensor.expand(shape)
```

## Parameters

- `src` / `tensor`: Input tensor
- `ndim` / `shape`: Target shape
  - Must have `ndim >= src->ndim` (can only add/expand dimensions)
  - List/tuple of integers in Python

## Broadcasting Rules

The function follows standard broadcasting rules:

1. **Dimensions are aligned from the right** (trailing dimensions)
2. **New dimensions can be prepended** to the left
3. **For existing dimensions:**
   - Size 1 can expand to any size (stride becomes 0)
   - Same size keeps the same stride
   - Different non-1 sizes are invalid and cause an error

## Behavior

- Expands dimensions of size 1 to larger sizes via broadcasting
- Can prepend new dimensions
- **Always zero-copy** - creates view with stride=0 for broadcasted dimensions
- Non-size-1 dimensions must match target shape
- Validates all broadcasting rules

## Examples

### C Example
```c
// Expand size-1 dimension
int shape1[] = {3, 1, 5};
Tensor* t1 = ones_host_tensor(DTYPE_FLOAT32, 3, shape1, NULL);
int new_shape1[] = {3, 4, 5};
Tensor* expanded1 = rp_expand(t1, 3, new_shape1);
// Result: [3, 4, 5] with strides [5, 0, 1]

// Prepend new dimension
int shape2[] = {5};
Tensor* t2 = ones_host_tensor(DTYPE_FLOAT32, 1, shape2, NULL);
int new_shape2[] = {3, 5};
Tensor* expanded2 = rp_expand(t2, 2, new_shape2);
// Result: [3, 5] with strides [0, 1]

// Expand multiple dimensions
int shape3[] = {1, 1, 5};
Tensor* t3 = ones_host_tensor(DTYPE_FLOAT32, 3, shape3, NULL);
int new_shape3[] = {3, 4, 5};
Tensor* expanded3 = rp_expand(t3, 3, new_shape3);
// Result: [3, 4, 5] with strides [0, 0, 1]
```

### Python Example
```python
import recpulse_cuda as rp

# Expand size-1 dimension
t = rp.zeros([3, 1, 5], dtype='float32', device='cpu')
expanded = t.expand([3, 4, 5])
print(expanded.shape)  # (3, 4, 5)

# Prepend new dimension
t = rp.ones([5], dtype='float32', device='cpu')
expanded = t.expand([3, 5])
print(expanded.shape)  # (3, 5)

# Expand for broadcasting
bias = rp.ones([1, 128], dtype='float32', device='cpu')
expanded_bias = bias.expand([32, 128])  # Broadcast to batch
print(expanded_bias.shape)  # (32, 128)

# Broadcast scalar-like tensor
t = rp.ones([1, 1, 1], dtype='float32', device='cpu')
expanded = t.expand([2, 3, 4])
print(expanded.shape)  # (2, 3, 4)
```

## Implementation Details

### Files Modified
- `functional/functional.h` - Added declaration
- `functional/functional.c` - Implemented `rp_expand`
- `ops/ops.h` - Added declaration for autograd wrapper
- `ops/ops.c` - Implemented `op_expand` with autograd support
- `python/tensor_python.c` - Added Python bindings
- `Makefile` - Added test_expand target
- `.claude/rules/shape_ops.md` - Updated documentation

### Algorithm
1. Validate inputs:
   - Target ndim must be >= source ndim
   - All target dimensions must be positive
2. Allocate new shape and strides arrays
3. Calculate offset for dimension alignment:
   - `src_offset = target_ndim - src_ndim`
   - Prepended dimensions get stride=0
4. For each target dimension:
   - **Prepended dimensions** (i < src_offset): Set stride=0
   - **Existing dimensions** (i >= src_offset):
     - If source dim is 1: Set stride=0 (broadcast)
     - If source dim equals target: Keep original stride
     - Otherwise: Error (invalid broadcast)
5. Calculate total size from new shape
6. Create view tensor with:
   - New shape and strides
   - Same data pointer (zero-copy)
   - owns_data = false
   - Pointer to base tensor

### Broadcasting Mechanism

The key to efficient broadcasting is using **stride=0** for broadcasted dimensions:

```
Original: [3, 1, 5] with strides [5, 5, 1]
Expanded: [3, 4, 5] with strides [5, 0, 1]
                                     ^
                                stride=0 means:
                                "reuse same element for all positions"
```

When accessing element [i, j, k]:
- **Normal stride**: data + i*5 + j*5 + k*1
- **With stride=0**: data + i*5 + j*0 + k*1 (j doesn't advance pointer!)

This creates the illusion of duplicated data without actually copying.

### Performance
- **Always zero-copy** - only metadata (shape/strides) is updated
- **O(ndim)** time complexity for validation and metadata creation
- **O(ndim)** space complexity for shape/strides arrays
- **No memory overhead** - all expanded views share the same underlying data

## Use Cases

### 1. Broadcasting for Element-wise Operations
```python
# Add bias to batch
data = rp.zeros([32, 128])      # Batch data
bias = rp.ones([1, 128])         # Bias vector
expanded_bias = bias.expand([32, 128])
# Now can do element-wise add (when implemented)
```

### 2. Attention Mechanisms
```python
# Expand attention mask
mask = rp.ones([1, 1, 512])      # Mask for one head
expanded_mask = mask.expand([32, 8, 512])  # [batch, heads, seq_len]
```

### 3. Repeating Constants
```python
# Create constant tensor for all batch elements
constant = rp.ones([1, 64])
batch_constants = constant.expand([100, 64])
```

### 4. Preparing Tensors for Broadcasting
```python
# Reshape and expand for broadcasting
weights = rp.ones([128])          # [features]
weights_2d = weights.expand([1, 128])  # [1, features]
weights_batch = weights_2d.expand([32, 128])  # [batch, features]
```

## Error Handling

The function validates and rejects:
- Target ndim < source ndim (cannot remove dimensions)
- Non-positive target dimensions
- Expanding non-1 dimension to different size:
  ```python
  t = rp.zeros([3, 4, 5])
  t.expand([3, 2, 5])  # Error! Cannot expand 4 to 2
  ```

## Testing

### C Test
```bash
make test_expand
```

Tests include:
- Expand size-1 dimension
- Prepend new dimensions
- Expand multiple dimensions
- Prepend and broadcast simultaneously
- Identity expansion (no change)
- Invalid expansion (rejected)
- Broadcast from all-ones shape

### Python Test
```bash
python3 test_expand.py
```

## Comparison with Repeat

| Function | Memory | Use Case |
|----------|--------|----------|
| `expand` | Zero-copy, stride=0 | Broadcasting, element-wise ops with different shapes |
| `repeat` | Copies data | Actually duplicating data, different semantics |

**expand** is used for broadcasting (mathematical operation semantics)
**repeat** is used for actual data duplication (not yet implemented)

## Status
✅ Fully implemented and tested
- C functional layer implementation
- C ops layer with autograd support
- Python bindings
- Unit tests (C and Python)
- Zero-copy view operation with stride=0
- Full broadcasting rule validation
- Efficient memory sharing

## Notes
- This completes the core broadcasting infrastructure for RecPulse
- With `expand`, element-wise operations can efficiently handle different shapes
- The stride=0 technique is crucial for efficient broadcasting in ML frameworks
