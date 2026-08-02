# Tensor Reshape Operation

## Overview

The reshape operation changes the shape of a tensor while preserving the total number of elements. It supports automatic dimension inference using `-1`.

## Implementation

### Files Modified
- `core/tensor.h` - Added function declarations
- `core/tensor.c` - Device-agnostic wrapper with dimension inference
- `core/tensor_host.c` - CPU implementation
- `core/tensor_device.cu` - GPU implementation

### Function Signature

```c
Tensor* tensor_reshape(Tensor* tensor, int new_ndim, int* new_shape);
```

### Features

1. **Automatic Dimension Inference**: Use `-1` for one dimension to automatically calculate its size
   ```c
   int shape[] = {2, 3, 4};  // 24 elements
   int new_shape[] = {-1, 6};  // Infers to [4, 6]
   Tensor* reshaped = tensor_reshape(t, 2, new_shape);
   ```

2. **Size Validation**: Ensures the total number of elements matches
   ```c
   int shape[] = {2, 3};  // 6 elements
   int bad_shape[] = {2, 4};  // 8 elements - ERROR
   ```

3. **Device Agnostic**: Works on both HOST (CPU) and DEVICE (GPU) tensors

4. **Memory Safety**: Uses goto-based error handling for proper cleanup

### Implementation Details

#### Dimension Inference Algorithm
1. Scan new_shape for `-1` dimension (only one allowed)
2. Calculate known_elements from all non-negative dimensions
3. Infer missing dimension: `inferred_dim = total_elements / known_elements`
4. Validate that division is exact (no remainder)

#### Memory Layout
- Creates a new tensor with the new shape
- Copies all data from source to destination
- Recalculates strides for C-contiguous layout
- Metadata is shallow-copied (grad and grad_fn set to NULL)

#### Device-Specific Implementations
- **HOST**: Uses `malloc()` and `memcpy()`
- **DEVICE**: Uses `cudaMalloc()` and `cudaMemcpy()` with DeviceToDevice

### Error Handling

The function returns `NULL` and prints an error message for:
- Invalid arguments (NULL pointers, ndim <= 0)
- Multiple `-1` dimensions
- Zero or negative dimension sizes
- Size mismatch (total elements don't match)
- Memory allocation failures

### Testing

Tests cover:
- Basic reshape operations (2D to 1D, 3D to 2D, etc.)
- Automatic dimension inference with `-1`
- Multi-dimensional inference (e.g., `[2, -1, 2]`)
- Both HOST and DEVICE tensors
- Error cases (size mismatch, multiple -1, zero dimensions)

Run tests with:
```bash
make test_reshape
```

### Performance Considerations

- **Memory Copy**: Current implementation always copies data
- **Future Optimization**: For contiguous tensors, could implement view-based reshape (no copy)
  - Would require reference counting or shared data pointers
  - Tensor struct already has `base_tensor` field for this purpose
- **Data Layout**: Assumes and maintains C-contiguous layout (row-major)

### Usage Example

```c
// Create a 2x3x4 tensor
int shape[] = {2, 3, 4};
Tensor* t = zeros_host_tensor(DTYPE_FLOAT32, 3, shape, NULL);

// Reshape to 6x4
int new_shape1[] = {6, 4};
Tensor* r1 = tensor_reshape(t, 2, new_shape1);

// Reshape to flat array with inference
int new_shape2[] = {-1};  // Infers to [24]
Tensor* r2 = tensor_reshape(t, 1, new_shape2);

// Reshape with partial inference
int new_shape3[] = {2, -1, 2};  // Infers to [2, 6, 2]
Tensor* r3 = tensor_reshape(t, 3, new_shape3);

// Clean up
free_tensor(r3);
free_tensor(r2);
free_tensor(r1);
free_tensor(t);
```

## Next Steps

Other shape manipulation operations to implement:
- `squeeze` - Remove dimensions of size 1
- `unsqueeze` - Add dimension of size 1 at specified position
- `permute` - Generalized transpose (arbitrary dimension reordering)
- `flatten` - Collapse to 1D or 2D
- `split` - Split tensor into multiple tensors
- `concat` - Join multiple tensors along a dimension
- `stack` - Join tensors along a new dimension
