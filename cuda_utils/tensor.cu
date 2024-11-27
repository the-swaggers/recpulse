#include "tensor.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void fill_kernel(float* vals, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vals[idx] = value;
    }
}

Tensor32* create_tensor(int* shape, int ndim) {
    Tensor32* tensor = (Tensor32*)malloc(sizeof(Tensor32));
    tensor->ndim = ndim;
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    
    size_t size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        size *= shape[i];
    }
    tensor->size = size;
    
    cudaMalloc(&tensor->vals, size * sizeof(float));
    return tensor;
}

void free_tensor(Tensor32* tensor) {
    cudaFree(tensor->vals);
    free(tensor->shape);
    free(tensor);
}

void fill_tensor(Tensor32* tensor, float value) {
    int block_size = 256;
    int num_blocks = (tensor->size + block_size - 1) / block_size;
    
    fill_kernel<<<num_blocks, block_size>>>(tensor->vals, value, tensor->size);
    cudaDeviceSynchronize();
}

