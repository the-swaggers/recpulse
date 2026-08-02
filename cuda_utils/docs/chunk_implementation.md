# Chunk Function Implementation

## Overview

The `chunk` function splits a tensor into a specified number of chunks along a dimension. Unlike `split_equal`, it allows the last chunk to be smaller if the dimension isn't evenly divisible. This is always a zero-copy operation - it creates views using `rp_split` internally.

## API

### C API (functional layer)
```c
Tensor** rp_chunk(Tensor* src, int chunks, int dim);
```

### C API (ops layer with autograd support)
```c
Tensor** op_chunk(Tensor* src, int chunks, int dim);
```

### Python API
```python
tensor.chunk(chunks, dim=0)
```

## Parameters

- `src` / `tensor`: Input tensor
- `chunks`: Number of chunks to create
- `dim`: Dimension along which to split (default: 0)
  - Supports negative indexing

## Behavior

- Splits tensor into `chunks` pieces along dimension `dim`
- If dimension size isn't evenly divisible, the last chunk will be smaller
- If `chunks` > dimension size, returns dimension size chunks (each size 1)
- Returns NULL-terminated array of tensor pointers (C) or list of tensors (Python)
- Always zero-copy (creates views)

## Comparison with split_equal

| Function | Division Requirements | Last Chunk Behavior |
|----------|----------------------|---------------------|
| `split_equal` | Must divide evenly | Always equal size, errors if not divisible |
| `chunk` | No requirements | Last chunk can be smaller |

**Example:**
```python
# split_equal would error here (10 not divisible by 3)
t = rp.zeros([10, 5])
chunks = t.chunk(3, 0)  # Works! Returns [4,5], [4,5], [2,5]
```

## Examples

### C Example
```c
int shape[] = {10, 5};
Tensor* t = ones_host_tensor(DTYPE_FLOAT32, 2, shape, NULL);

// Split into 3 chunks along dim 0
Tensor** chunks = rp_chunk(t, 3, 0);

// Result: [4,5], [4,5], [2,5]
int i = 0;
while (chunks[i] != NULL) {
    printf("Chunk %d: shape=[%d, %d]\n", i,
           chunks[i]->shape[0], chunks[i]->shape[1]);
    i++;
}

// Clean up
for (int j = 0; chunks[j] != NULL; j++) {
    free_tensor(chunks[j]);
}
free(chunks);
```

### Python Example
```python
import recpulse_cuda as rp

t = rp.zeros([10, 5], dtype='float32', device='cpu')

# Even division: [4,5], [4,5], [2,5]
chunks = t.chunk(3, 0)
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i}: {chunk.shape}")

# Exact division: [4,3], [4,3], [4,3]
t2 = rp.zeros([12, 3])
chunks2 = t2.chunk(3, 0)

# More chunks than size: returns 5 chunks of size 1
t3 = rp.zeros([5, 3])
chunks3 = t3.chunk(10, 0)  # Gets clamped to 5 chunks

# Chunk along different dimension
t4 = rp.zeros([3, 10])
chunks4 = t4.chunk(4, 1)  # [3,3], [3,3], [3,3], [3,1]

# Negative dimension
chunks5 = t.chunk(3, -1)  # Same as dim=1
```

## Implementation Details

### Files Modified
- `cuda_utils/functional/functional.h` - Added declaration
- `cuda_utils/functional/functional.c` - Implemented `rp_chunk`
- `cuda_utils/ops/ops.h` - Added declaration for autograd wrapper
- `cuda_utils/ops/ops.c` - Implemented `op_chunk` with autograd support
- `cuda_utils/python/tensor_python.c` - Added Python bindings
- `cuda_utils/Makefile` - Added test_chunk target
- `.claude/rules/shape_ops.md` - Updated documentation

### Algorithm
1. Validate inputs and handle negative dim indexing
2. Clamp `chunks` to dimension size if too large
3. Calculate `chunk_size = ceil(dim_size / chunks)`
4. Build sizes array:
   - Most chunks get `chunk_size` elements
   - Last chunk gets remaining elements (may be smaller)
5. Call `rp_split` with the sizes array
6. Wrap result in NULL-terminated array for easier iteration
7. Return NULL-terminated array

### Memory Management
The function:
1. Allocates a NULL-terminated array (size: `actual_chunks + 1`)
2. Copies tensor pointers from `rp_split` result
3. Frees the original `rp_split` array (but not the tensors)
4. Returns NULL-terminated array

### Performance
- **Always zero-copy** - creates views using slicing
- **O(chunks)** time complexity for calculating sizes and creating views
- **O(chunks)** space complexity for the array of tensor pointers

## Use Cases

### 1. Splitting Batches for Processing
```python
# Split large batch into smaller chunks for processing
batch = rp.zeros([100, 3, 224, 224])
chunks = batch.chunk(4, 0)  # 4 chunks of 25 each

for chunk in chunks:
    # Process each chunk separately
    result = model(chunk)
```

### 2. Multi-GPU Distribution (Uneven)
```python
# Distribute data across GPUs (last GPU gets smaller batch)
data = rp.zeros([43, 512])
chunks = data.chunk(4, 0)  # [11, 11, 11, 10]

for i, chunk in enumerate(chunks):
    chunk_gpu = chunk.to(f'cuda:{i}')
    # Process on GPU i
```

### 3. Temporal Splitting
```python
# Split sequence into overlapping windows
sequence = rp.zeros([1000, 128])
chunks = sequence.chunk(10, 0)  # Split into 10 temporal chunks
```

## Testing

### C Test
```bash
make test_chunk
```

Tests include:
- Even division (non-exact)
- Exact division
- More chunks than dimension size
- Chunk along different dimension
- Single chunk
- Negative dimension indexing

### Python Test
```bash
python3 test_chunk.py
```

## Status
✅ Fully implemented and tested
- C functional layer implementation with NULL-terminated arrays
- C ops layer with autograd support for all chunks
- Python bindings returning list of tensors
- Unit tests (C and Python)
- Zero-copy view operation
- Handles uneven divisions gracefully

## Phase 2 Complete!
With the implementation of `chunk`, all Phase 2 shape operations are now complete:
- ✅ flatten
- ✅ permute
- ✅ chunk
